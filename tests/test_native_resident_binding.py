from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import sysconfig
import threading
import time
from types import ModuleType
from typing import Any, Sequence

import pytest

from neuralfn.native_inference import (
    GenerationConfig,
    KVCacheConfig,
    NativeInferenceCancelledError,
    NativeInferenceCapabilityError,
    NativeInferenceClosedError,
    NativeInferenceModel,
    _turboquant_binding_tables,
)
from neuralfn.native_cli import NativeArtifactCLIConfig, run_native_artifact_cli
from neuralfn.native_ir import migrate_graph_to_native
from neuralfn.turboquant import TurboQuantReferenceCodec
from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config


ROOT = Path(__file__).resolve().parents[1]


def _bf16(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0] >> 16


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _python_dense_logits(
    checkpoint: Path,
    token_ids: Sequence[int],
    *,
    use_qk_norm: bool,
    logit_softcap: float,
    mlp_activation: str = "gelu_exact",
) -> list[float]:
    """Small independent formula oracle for the resident dense-v5 fixture."""

    raw = checkpoint.read_bytes()
    (
        _magic,
        _version,
        max_seq_len,
        vocab_size,
        num_layers,
        num_heads,
        channels,
        padded_vocab_size,
    ) = struct.unpack_from("<8i", raw)
    bf16 = struct.unpack(f"<{(len(raw) - 1024) // 2}H", raw[1024:])
    weights = [struct.unpack("<f", struct.pack("<I", value << 16))[0] for value in bf16]
    offset = 0

    def take(count: int) -> list[float]:
        nonlocal offset
        result = weights[offset : offset + count]
        offset += count
        return result

    wte = take(padded_vocab_size * channels)
    wpe = take(max_seq_len * channels)
    blocks: list[dict[str, list[float]]] = []
    for _layer in range(num_layers):
        blocks.append(
            {
                "ln1w": take(channels),
                "ln1b": take(channels),
                "qkvw": take(3 * channels * channels),
                "qkvb": take(3 * channels),
                "attnw": take(channels * channels),
                "attnb": take(channels),
                "ln2w": take(channels),
                "ln2b": take(channels),
                "fcw": take(4 * channels * channels),
                "fcb": take(4 * channels),
                "mlpw": take(4 * channels * channels),
                "mlpb": take(channels),
            }
        )
    final_lnw = take(channels)
    final_lnb = take(channels)
    assert offset == len(weights)

    def layer_norm(
        rows: list[list[float]],
        weight: list[float],
        bias: list[float],
    ) -> list[list[float]]:
        result: list[list[float]] = []
        for row in rows:
            mean = sum(row) / channels
            variance = sum((value - mean) ** 2 for value in row) / channels
            inverse = 1.0 / math.sqrt(variance + 1.0e-5)
            result.append(
                [_float32((value - mean) * inverse * weight[i] + bias[i]) for i, value in enumerate(row)]
            )
        return result

    def linear(
        rows: list[list[float]],
        weight: list[float],
        bias: list[float],
        output_dim: int,
    ) -> list[list[float]]:
        result: list[list[float]] = []
        input_dim = len(rows[0])
        for row in rows:
            output: list[float] = []
            for out in range(output_dim):
                value = float(bias[out])
                for index in range(input_dim):
                    value += row[index] * weight[out * input_dim + index]
                output.append(_float32(value))
            result.append(output)
        return result

    hidden = [
        [
            _float32(wte[token * channels + channel] + wpe[row * channels + channel])
            for channel in range(channels)
        ]
        for row, token in enumerate(token_ids)
    ]
    head_dim = channels // num_heads
    for block in blocks:
        normalized = layer_norm(hidden, block["ln1w"], block["ln1b"])
        qkv = linear(normalized, block["qkvw"], block["qkvb"], channels * 3)
        if use_qk_norm:
            for row in qkv:
                for segment in range(2):
                    for head in range(num_heads):
                        start = segment * channels + head * head_dim
                        squared = sum(row[start + dim] ** 2 for dim in range(head_dim))
                        inverse = 1.0 / math.sqrt(squared / head_dim + 1.0e-6)
                        for dim in range(head_dim):
                            row[start + dim] = _float32(row[start + dim] * inverse)
        attention = [[0.0] * channels for _row in token_ids]
        scale = 1.0 / math.sqrt(head_dim)
        for row_index in range(len(token_ids)):
            for head in range(num_heads):
                q_start = head * head_dim
                scores: list[float] = []
                for key_row in range(row_index + 1):
                    k_start = channels + head * head_dim
                    score = sum(
                        qkv[row_index][q_start + dim] * qkv[key_row][k_start + dim]
                        for dim in range(head_dim)
                    ) * scale
                    scores.append(score)
                maximum = max(scores)
                probabilities = [math.exp(score - maximum) for score in scores]
                denominator = sum(probabilities)
                for dim in range(head_dim):
                    value = sum(
                        probabilities[key_row]
                        / denominator
                        * qkv[key_row][channels * 2 + head * head_dim + dim]
                        for key_row in range(row_index + 1)
                    )
                    attention[row_index][head * head_dim + dim] = _float32(value)
        projected = linear(attention, block["attnw"], block["attnb"], channels)
        residual = [
            [_float32(left + right) for left, right in zip(hidden_row, projected_row)]
            for hidden_row, projected_row in zip(hidden, projected)
        ]
        normalized = layer_norm(residual, block["ln2w"], block["ln2b"])
        expanded = linear(normalized, block["fcw"], block["fcb"], channels * 4)
        for row in expanded:
            for index, source in enumerate(row):
                if mlp_activation == "gelu_exact":
                    activated = 0.5 * source * (1.0 + math.erf(source / math.sqrt(2.0)))
                elif mlp_activation == "gelu":
                    activated = 0.5 * source * (
                        1.0
                        + math.tanh(
                            0.7978845608028654
                            * (source + 0.044715 * source * source * source)
                        )
                    )
                elif mlp_activation == "relu":
                    activated = max(0.0, source)
                elif mlp_activation == "silu":
                    activated = source / (1.0 + math.exp(-source))
                elif mlp_activation == "relu2":
                    activated = max(0.0, source) ** 2
                else:
                    raise AssertionError(f"unknown test MLP activation {mlp_activation!r}")
                row[index] = _float32(activated)
        mlp = linear(expanded, block["mlpw"], block["mlpb"], channels)
        hidden = [
            [_float32(left + right) for left, right in zip(residual_row, mlp_row)]
            for residual_row, mlp_row in zip(residual, mlp)
        ]
    final_hidden = layer_norm(hidden, final_lnw, final_lnb)[-1]
    logits: list[float] = []
    for token in range(vocab_size):
        value = sum(
            final_hidden[channel] * wte[token * channels + channel]
            for channel in range(channels)
        )
        if logit_softcap > 0.0:
            value = logit_softcap * math.tanh(value / logit_softcap)
        logits.append(_float32(value))
    return logits


def _write_tiny_dense_v5(
    path: Path,
    *,
    nontrivial: bool = False,
    max_seq_len: int = 8,
    vocab_size: int = 4,
    num_layers: int = 1,
    num_heads: int = 1,
    channels: int = 2,
    padded_vocab_size: int | None = None,
) -> None:
    padded_vocab_size = padded_vocab_size or vocab_size
    header = [0] * 256
    header[:8] = [
        20240326,
        5,
        max_seq_len,
        vocab_size,
        num_layers,
        num_heads,
        channels,
        padded_vocab_size,
    ]
    expected = (
        padded_vocab_size * channels
        + max_seq_len * channels
        + num_layers * (12 * channels * channels + 13 * channels)
        + 2 * channels
    )
    weights = [0.0] * (padded_vocab_size * channels)
    if (
        max_seq_len == 8
        and vocab_size == 4
        and padded_vocab_size == 4
        and num_layers == 1
        and num_heads == 1
        and channels == 2
    ):
        weights[:] = [0.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.5, 0.5]
    weights.extend([0.0] * (max_seq_len * channels))
    for _layer in range(num_layers):
        weights.extend([1.0] * channels)  # ln_1.weight
        weights.extend([0.0] * channels)  # ln_1.bias
        weights.extend([0.0] * (3 * channels * channels))
        weights.extend([0.0] * (3 * channels))
        weights.extend([0.0] * (channels * channels))
        weights.extend([0.0] * channels)
        weights.extend([1.0] * channels)  # ln_2.weight
        weights.extend([0.0] * channels)  # ln_2.bias
        weights.extend([0.0] * (4 * channels * channels))
        weights.extend([0.0] * (4 * channels))
        weights.extend([0.0] * (4 * channels * channels))
        weights.extend([0.0] * channels)
    weights.extend([1.0] * channels)  # ln_f.weight
    weights.extend([0.0] * channels)  # ln_f.bias
    assert len(weights) == expected
    if nontrivial:
        weights = [
            0.08 * math.sin(index * 0.37) + 0.02 * math.cos(index * 0.11)
            for index in range(expected)
        ]
    path.write_bytes(
        struct.pack("<256i", *header)
        + struct.pack(f"<{len(weights)}H", *(_bf16(value) for value in weights))
    )


def _manifest(
    checkpoint: Path,
    *,
    artifact_path: str | None = None,
    turboquant: bool = False,
    use_qk_norm: bool = False,
    logit_softcap: float = 0.0,
    moa_activation: str | None = None,
    moa_interval: int = 50,
) -> dict[str, Any]:
    checkpoint_bytes = checkpoint.read_bytes()
    (
        _magic,
        _version,
        max_seq_len,
        vocab_size,
        num_layers,
        num_heads,
        channels,
        _padded_vocab_size,
    ) = struct.unpack_from("<8i", checkpoint_bytes)
    topology_graphs: list[dict[str, Any]] = []
    if use_qk_norm:
        for layer in range(num_layers):
            graph_path = (
                f"root/nodes/model/subgraph/nodes/block_{layer}/subgraph/"
                "nodes/attention/subgraph"
            )

            def qk_node_path(instance_id: str) -> str:
                return f"{graph_path}/nodes/{instance_id}"

            topology_graphs.append(
                {
                    "path": graph_path,
                    "nodes": [
                        {
                            "path": qk_node_path("q_heads"),
                            "instance_id": "q_heads",
                            "kind": "module",
                            "operation": "reshape_heads",
                            "module_config": {"num_heads": num_heads},
                        },
                        {
                            "path": qk_node_path("k_heads"),
                            "instance_id": "k_heads",
                            "kind": "module",
                            "operation": "reshape_heads",
                            "module_config": {"num_heads": num_heads},
                        },
                        {
                            "path": qk_node_path("qk_norm"),
                            "instance_id": "qk_norm",
                            "kind": "module",
                            "operation": "qk_norm",
                            "module_config": {"eps": 1.0e-6},
                        },
                        {
                            "path": qk_node_path("sdpa"),
                            "instance_id": "sdpa",
                            "kind": "module",
                            "operation": "scaled_dot_product_attention",
                            "module_config": {"is_causal": True, "dropout_p": 0.0},
                        },
                    ],
                    "edges": [
                        {
                            "src_node": qk_node_path("q_heads"),
                            "src_port": 0,
                            "dst_node": qk_node_path("qk_norm"),
                            "dst_port": 0,
                        },
                        {
                            "src_node": qk_node_path("k_heads"),
                            "src_port": 0,
                            "dst_node": qk_node_path("qk_norm"),
                            "dst_port": 1,
                        },
                        {
                            "src_node": qk_node_path("qk_norm"),
                            "src_port": 0,
                            "dst_node": qk_node_path("sdpa"),
                            "dst_port": 0,
                        },
                        {
                            "src_node": qk_node_path("qk_norm"),
                            "src_port": 1,
                            "dst_node": qk_node_path("sdpa"),
                            "dst_port": 1,
                        },
                    ],
                }
            )
    if logit_softcap > 0.0:
        graph_path = "root/nodes/model/subgraph"

        def softcap_node_path(instance_id: str) -> str:
            return f"{graph_path}/nodes/{instance_id}"

        topology_graphs.append(
            {
                "path": graph_path,
                "nodes": [
                    {
                        "path": softcap_node_path("tied_lm_head"),
                        "instance_id": "tied_lm_head",
                        "kind": "module",
                        "operation": "tied_lm_head",
                        "module_config": {},
                    },
                    {
                        "path": softcap_node_path("softcap"),
                        "instance_id": "softcap",
                        "kind": "module",
                        "operation": "logit_softcap",
                        "module_config": {"softcap": logit_softcap},
                    },
                    {
                        "path": softcap_node_path("ce"),
                        "instance_id": "ce",
                        "kind": "module",
                        "operation": "token_cross_entropy",
                        "module_config": {},
                    },
                ],
                "edges": [
                    {
                        "src_node": softcap_node_path("tied_lm_head"),
                        "src_port": 0,
                        "dst_node": softcap_node_path("softcap"),
                        "dst_port": 0,
                    },
                    {
                        "src_node": softcap_node_path("softcap"),
                        "src_port": 0,
                        "dst_node": softcap_node_path("ce"),
                        "dst_port": 0,
                    },
                ],
            }
        )
    if not topology_graphs:
        topology_graphs.append({"path": "root", "nodes": [], "edges": []})
    graph_sha256 = "ab" * 32
    block_spec: dict[str, Any] = {
        "norm_type": "layernorm",
        "mlp_type": "gelu",
        "pos_encoding": "absolute",
        "attention_variant": "dense",
        "residual_type": "add",
        "compression": "none",
        "activation_mode": "moa" if moa_activation is not None else "single",
        "linear_bias": True,
        "use_qk_norm": use_qk_norm,
        "dropout_p": 0.0,
        "num_heads": num_heads,
        "num_kv_heads": None,
    }
    checkpoint_contract: dict[str, Any] = {
        "format": "neuralfn.native_dense_gpt.v5",
        "artifact_path": artifact_path or checkpoint.name,
        "target_nbytes": len(checkpoint_bytes),
        "target_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
    }
    if moa_activation is not None:
        block_spec.update(
            {
                "moa_activations": ["gelu", "relu", "silu", "relu2"],
                "moa_interval": moa_interval,
            }
        )
        checkpoint_contract["moa"] = {
            "schema": "neuralfn.native_dense_moa.inference_checkpoint",
            "version": 1,
            "preset": "gpt2_moa",
            "selected_activation": moa_activation,
            "candidate_activations": ["gelu", "relu", "silu", "relu2"],
            "interval": moa_interval,
            "source_graph_sha256": graph_sha256,
            "metadata_sha256": "cd" * 32,
        }
    return {
        "schema": "neuralfn.native_execution_manifest",
        "version": 1,
        "source_graph": {
            "path": None,
            "sha256": graph_sha256,
            "serialization_changed": False,
        },
        "model": {
            "family": "gpt2",
            "family_class": "autoregressive_transformer",
            "template_spec": {
                "model_dim": channels,
                "num_layers": num_layers,
                "vocab_size": vocab_size,
                "tie_embeddings": True,
                "logit_softcap": logit_softcap,
                "block_spec": block_spec,
            },
        },
        "topology": {"graphs": topology_graphs},
        "checkpoint": checkpoint_contract,
        "kernel_abi": {
            "resident_inference": {"version": 1, "status": "ready"},
            "session_prefix_cow": {
                "version": 1,
                "status": "ready",
                "profile": "dense-full-cache-kv-final-hidden-v1",
                "operation": "fork_session",
            },
            "session_prefix_cow_cpu_turboquant": {
                "version": 1 if turboquant else None,
                "status": "ready" if turboquant else "not_implemented",
                "profile": (
                    "dense-cpu-turboquant-mse-qjl-packed-kv-final-hidden-v1"
                    if turboquant
                    else None
                ),
                "operation": "fork_session" if turboquant else None,
                "backend": "cpu-reference-packed" if turboquant else None,
            },
            "turboquant_cache": {
                "version": 1 if turboquant else None,
                "status": "ready" if turboquant else "not_implemented",
                "backend": "cpu-reference-packed" if turboquant else None,
            },
        },
        "context_limits": {"max_context_tokens": max_seq_len, "max_output_tokens": None},
        "capabilities": {
            "native_inference": True,
            "resident_inference": True,
            "lossless_kv_cache": True,
            "session_prefix_cow": True,
            "session_prefix_cow_cpu_turboquant": turboquant,
            "turboquant_kv_cache": turboquant,
            "function_tools": False,
            "structured_output": False,
        },
        "session_state_kinds": ["token_history", "kv", "final_hidden_history"],
        "stop_tokens": [],
    }


def _manifest_topology_node(
    manifest: dict[str, Any],
    operation: str,
) -> dict[str, Any]:
    return next(
        node
        for graph in manifest["topology"]["graphs"]
        for node in graph["nodes"]
        if node.get("operation") == operation
    )


@pytest.fixture(scope="session")
def resident_binding(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    output = tmp_path_factory.mktemp("native-resident-binding") / (
        "_native_inference" + (sysconfig.get_config_var("EXT_SUFFIX") or ".so")
    )
    subprocess.run(
        ["bash", str(ROOT / "tools" / "build_native_inference_binding.sh"), str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    spec = importlib.util.spec_from_file_location("_native_inference", output)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resident_binding_reports_glimmer_cuda_vision_feature(
    resident_binding: ModuleType,
) -> None:
    capabilities = resident_binding.resident_inference_capabilities()
    assert capabilities["vision"] is True
    assert capabilities["vision_cpu"] is True
    assert capabilities["vision_cuda"] is True
    assert capabilities["media_encoder_abi"] == {
        "version": 1,
        "load_operation": "load_companion",
        "encode_operation": "encode_media",
        "prefill_operation": "prefill_with_embeddings",
        "projection_width": 6656,
    }
    source = (ROOT / "neuralfn/csrc/native_gpt2/resident_binding.cpp").read_text(
        encoding="utf-8"
    )
    assert (
        'handle->glimmer->vision_whole_model_cuda() ? Py_True : Py_False'
        in source
    )


def test_resident_binding_loads_once_and_runs_real_dense_recompute(
    resident_binding: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model_00000001.bin"
    _write_tiny_dense_v5(checkpoint)
    manifest = _manifest(checkpoint)
    (tmp_path / "native-execution-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    events: list[tuple[int, tuple[int, ...]]] = []
    with NativeInferenceModel.load(
        tmp_path,
        binding=resident_binding,
        kv_cache=KVCacheConfig(mode="off"),
    ) as model:
        checkpoint.unlink()  # The resident model must not reopen it for decode.
        with model.create_session(seed=17) as session:
            assert session.prefill([1]) == {
                "prefix_tokens": 1,
                "prefix_reused": 0,
                "prefilled_tokens": 1,
            }
            result = session.decode(
                GenerationConfig(max_new_tokens=2, temperature=0.0),
                on_token=lambda event: events.append((event.token_id, session.token_ids)),
            )
            assert result.token_ids == (1, 1)
            assert events == [(1, (1, 1)), (1, (1, 1, 1))]
            assert session.token_ids == (1, 1, 1)
            session_stats = session.stats()
            assert session_stats["token_count"] == 3
            assert session_stats["decode_calls"] == 2
            assert session_stats["effective_cache"] == "off"
            assert session_stats["recompute_full_prefix"] is True

        stats = model.stats()
        assert stats["backend"] == "cpu-reference-resident"
        assert stats["weights_load_count"] == 1
        assert stats["forward_calls"] == 2
        assert stats["subprocess_spawns"] == 0
        assert stats["lossless_kv_cache"] is True
        # Binding-wide support is not an effective model capability: this
        # artifact deliberately did not prove the TurboQuant ABI.
        assert stats["turboquant_kv_cache"] is False


def test_resident_sessions_are_isolated_and_support_lifecycle(
    resident_binding: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model_2.bin"
    _write_tiny_dense_v5(checkpoint)
    manifest = _manifest(checkpoint)
    model = resident_binding.load_model(str(tmp_path), manifest)
    left = resident_binding.create_session(
        model, {"seed": 1, "kv_cache": {"effective_mode": "full"}}
    )
    right = resident_binding.create_session(
        model, {"seed": 2, "kv_cache": {"effective_mode": "full"}}
    )
    try:
        resident_binding.prefill(model, left, [1], 0)
        resident_binding.prefill(model, right, [2], 0)
        strict = {
            "temperature": -0.0,
            "top_k": None,
            "top_p": 1.0,
            "seed": None,
            "stop_token_ids": [],
            "strict_model_compute": True,
        }
        assert resident_binding.decode_one(model, left, strict)["token_id"] == 1
        assert resident_binding.decode_one(model, right, strict)["token_id"] == 2
        assert resident_binding.session_stats(model, left)["token_count"] == 2
        assert resident_binding.session_stats(model, right)["token_count"] == 2
        assert resident_binding.session_stats(model, left)["cached_tokens"] == 2
        assert resident_binding.session_stats(model, right)["cached_tokens"] == 2

        resident_binding.truncate_session(model, left, 1)
        assert resident_binding.session_stats(model, left)["token_count"] == 1
        resident_binding.cancel_session(model, left)
        with pytest.raises(InterruptedError, match="cancelled"):
            resident_binding.decode_one(model, left, strict)
        resident_binding.reset_session(model, left)
        assert resident_binding.session_stats(model, left)["token_count"] == 0
        assert resident_binding.session_stats(model, left)["cancelled"] is False
    finally:
        resident_binding.close_session(model, left)
        resident_binding.close_session(model, right)
        resident_binding.close_model(model)


def test_dense_full_cache_prefix_fork_is_native_copy_on_write(
    resident_binding: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model_prefix_cow.bin"
    _write_tiny_dense_v5(checkpoint, nontrivial=True)
    manifest = _manifest(checkpoint)
    model = resident_binding.load_model(str(tmp_path), manifest)
    parent = resident_binding.create_session(
        model, {"seed": 1, "kv_cache": {"effective_mode": "full"}}
    )
    left = None
    right = None
    baseline = None
    try:
        resident_binding.prefill(model, parent, [1, 2, 1], 0)
        parent_logits = resident_binding.current_logits(model, parent)
        left = resident_binding.fork_session(
            model, parent, {"token_count": 3, "seed": 11}
        )
        right = resident_binding.fork_session(
            model, parent, {"token_count": 2, "seed": 22}
        )

        parent_shared = resident_binding.session_stats(model, parent)
        left_shared = resident_binding.session_stats(model, left)
        right_shared = resident_binding.session_stats(model, right)
        assert parent_shared["prefix_cow_forks_created"] == 2
        assert parent_shared["prefix_cow_storage_use_count"] == 3
        assert parent_shared["prefix_cow_shared_storage"] is True
        assert parent_shared["prefix_cow_shared_cached_tokens"] == 3
        assert parent_shared["prefix_cow_shared_cached_tokens_scope"] == (
            "this-session-valid-rows-in-shared-allocation"
        )
        assert parent_shared["prefix_cow_shared_capacity_bytes"] == parent_shared[
            "cache_capacity_bytes"
        ]
        assert left_shared["prefix_cow_forked_from_tokens"] == 3
        assert right_shared["prefix_cow_forked_from_tokens"] == 2
        assert right_shared["prefix_cow_shared_cached_tokens"] == 2
        assert left_shared["prefix_cow_detach_count"] == 0
        assert resident_binding.current_logits(model, left) == pytest.approx(
            parent_logits, abs=1.0e-7
        )
        baseline = resident_binding.create_session(
            model, {"seed": 0, "kv_cache": {"effective_mode": "full"}}
        )
        resident_binding.prefill(model, baseline, [1, 2], 0)
        assert resident_binding.current_logits(model, right) == pytest.approx(
            resident_binding.current_logits(model, baseline), abs=1.0e-7
        )

        # Each writer must detach all three native cache components before
        # overwriting its first divergent row.  The parent remains unchanged.
        strict = {
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "seed": None,
            "stop_token_ids": [],
            "strict_model_compute": True,
        }
        left_token = resident_binding.decode_one(model, left, strict)["token_id"]
        left_detached = resident_binding.session_stats(model, left)
        parent_after_left = resident_binding.session_stats(model, parent)
        assert left_detached["prefix_cow_storage_use_count"] == 1
        assert left_detached["prefix_cow_shared_storage"] is False
        assert left_detached["prefix_cow_detach_count"] == 1
        assert left_detached["prefix_cow_detached_capacity_bytes"] == left_detached[
            "cache_capacity_bytes"
        ]
        assert parent_after_left["prefix_cow_storage_use_count"] == 2
        assert resident_binding.current_logits(model, parent) == pytest.approx(
            parent_logits, abs=1.0e-7
        )

        resident_binding.prefill(model, right, [3], 2)
        assert resident_binding.session_stats(model, right)["prefix_cow_detach_count"] == 1
        assert resident_binding.session_stats(model, parent)[
            "prefix_cow_storage_use_count"
        ] == 1
        assert resident_binding.session_stats(model, parent)[
            "prefix_cow_shared_storage"
        ] is False

        resident_binding.reset_session(model, baseline)
        resident_binding.prefill(model, baseline, [1, 2, 1, left_token], 0)
        assert resident_binding.current_logits(model, left) == pytest.approx(
            resident_binding.current_logits(model, baseline), abs=1.0e-7
        )
        resident_binding.reset_session(model, baseline)
        resident_binding.prefill(model, baseline, [1, 2, 3], 0)
        assert resident_binding.current_logits(model, right) == pytest.approx(
            resident_binding.current_logits(model, baseline), abs=1.0e-7
        )

        resident_binding.close_session(model, left)
        left = None
        assert resident_binding.session_stats(model, parent)["token_count"] == 3
        resident_binding.reset_session(model, right)
        assert resident_binding.session_stats(model, right)["token_count"] == 0
        assert resident_binding.session_stats(model, parent)["token_count"] == 3
        resident_binding.truncate_session(model, parent, 2)
        assert resident_binding.session_stats(model, parent)["token_count"] == 2
        assert resident_binding.session_stats(model, right)["token_count"] == 0
    finally:
        if baseline is not None:
            resident_binding.close_session(model, baseline)
        if left is not None:
            resident_binding.close_session(model, left)
        if right is not None:
            resident_binding.close_session(model, right)
        resident_binding.close_session(model, parent)
        resident_binding.close_model(model)


@pytest.mark.parametrize("profile", ["mse-3.5", "qjl-3.5"])
def test_dense_cpu_turboquant_prefix_fork_shares_packed_storage_and_isolates_tails(
    resident_binding: ModuleType,
    tmp_path: Path,
    profile: str,
) -> None:
    checkpoint = tmp_path / f"model_turboquant_prefix_cow_{profile}.bin"
    _write_tiny_dense_v5(checkpoint, nontrivial=True)
    manifest = _manifest(checkpoint, turboquant=True)
    model = resident_binding.load_model(str(tmp_path), manifest)
    tables = _turboquant_binding_tables(
        channels=2,
        num_heads=1,
        profile=profile,
    )

    def new_session(seed: int = 0) -> Any:
        return resident_binding.create_session(
            model,
            {
                "seed": seed,
                "kv_cache": {
                    "effective_mode": "turboquant",
                    "turboquant_profile": profile,
                    "tables": tables,
                },
            },
        )

    parent = new_session(1)
    children: list[Any] = []
    baseline = new_session(99)
    strict = {
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "seed": None,
        "stop_token_ids": [],
        "strict_model_compute": True,
    }
    try:
        resident_binding.prefill(model, parent, [1, 2, 1], 0)
        parent_logits = resident_binding.current_logits(model, parent)
        parent_calls = resident_binding.session_stats(model, parent)[
            "turboquant_cpu_compressed_attention_calls"
        ]
        assert parent_calls > 0

        left = resident_binding.fork_session(
            model, parent, {"token_count": 3, "seed": 11}
        )
        right = resident_binding.fork_session(
            model, parent, {"token_count": 2, "seed": 22}
        )
        children.extend([left, right])
        for session, valid_rows in ((parent, 3), (left, 3), (right, 2)):
            stats = resident_binding.session_stats(model, session)
            assert stats["prefix_cow_storage_use_count"] == 3
            assert stats["prefix_cow_shared_storage"] is True
            assert stats["prefix_cow_shared_cached_tokens"] == valid_rows
            assert stats["prefix_cow_shared_capacity_bytes"] == stats[
                "cache_capacity_bytes"
            ]
        assert resident_binding.session_stats(model, left)[
            "turboquant_cpu_compressed_attention_calls"
        ] == 0
        assert resident_binding.session_stats(model, right)[
            "turboquant_cpu_compressed_attention_calls"
        ] == 0

        # Logical-only mutations retain the shared allocation and do not count
        # as a detach.
        logical = resident_binding.fork_session(
            model, parent, {"token_count": 3, "seed": 33}
        )
        children.append(logical)
        resident_binding.truncate_session(model, logical, 1)
        resident_binding.reset_session(model, logical)
        logical_stats = resident_binding.session_stats(model, logical)
        assert logical_stats["prefix_cow_shared_storage"] is True
        assert logical_stats["prefix_cow_shared_cached_tokens"] == 0
        assert logical_stats["prefix_cow_detach_count"] == 0
        resident_binding.close_session(model, logical)
        children.remove(logical)

        # Both a complete-prefix child and a partial-prefix child detach the
        # packed K/V plus final-hidden allocation before overwriting a tail.
        resident_binding.prefill(model, left, [2], 3)
        left_stats = resident_binding.session_stats(model, left)
        assert left_stats["prefix_cow_storage_use_count"] == 1
        assert left_stats["prefix_cow_shared_storage"] is False
        assert left_stats["prefix_cow_detach_count"] == 1
        assert left_stats["prefix_cow_detached_capacity_bytes"] == left_stats[
            "cache_capacity_bytes"
        ]
        assert left_stats["turboquant_cpu_compressed_attention_calls"] > 0
        assert resident_binding.session_stats(model, parent)[
            "turboquant_cpu_compressed_attention_calls"
        ] == parent_calls

        resident_binding.prefill(model, right, [3], 2)
        right_stats = resident_binding.session_stats(model, right)
        assert right_stats["prefix_cow_detach_count"] == 1
        assert right_stats["turboquant_cpu_compressed_attention_calls"] > 0
        assert resident_binding.session_stats(model, parent)[
            "prefix_cow_storage_use_count"
        ] == 1
        assert resident_binding.current_logits(model, parent) == pytest.approx(
            parent_logits, abs=1.0e-7
        )

        for session, history in ((left, [1, 2, 1, 2]), (right, [1, 2, 3])):
            resident_binding.reset_session(model, baseline)
            resident_binding.prefill(model, baseline, history, 0)
            assert resident_binding.current_logits(model, session) == pytest.approx(
                resident_binding.current_logits(model, baseline), abs=1.0e-7
            )

        # Closing one still-shared owner only drops that owner's references.
        survivor = resident_binding.fork_session(
            model, parent, {"token_count": 3, "seed": 44}
        )
        close_peer = resident_binding.fork_session(
            model, parent, {"token_count": 3, "seed": 55}
        )
        children.extend([survivor, close_peer])
        assert resident_binding.session_stats(model, survivor)[
            "prefix_cow_storage_use_count"
        ] == 3
        resident_binding.close_session(model, close_peer)
        children.remove(close_peer)
        assert resident_binding.session_stats(model, survivor)[
            "prefix_cow_storage_use_count"
        ] == 2

        # The original parent is also a writer. Its decode detaches without
        # changing the still-shared child's prefix or per-session CPU counter.
        survivor_calls = resident_binding.session_stats(model, survivor)[
            "turboquant_cpu_compressed_attention_calls"
        ]
        resident_binding.decode_one(model, parent, strict)
        assert resident_binding.session_stats(model, parent)[
            "prefix_cow_detach_count"
        ] == 1
        assert resident_binding.current_logits(model, survivor) == pytest.approx(
            parent_logits, abs=1.0e-7
        )
        assert resident_binding.session_stats(model, survivor)[
            "turboquant_cpu_compressed_attention_calls"
        ] == survivor_calls
    finally:
        resident_binding.close_session(model, baseline)
        for child in children:
            resident_binding.close_session(model, child)
        resident_binding.close_session(model, parent)
        resident_binding.close_model(model)


def test_prefix_fork_binding_rejects_non_full_cross_model_closed_and_bad_prefix(
    resident_binding: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model_prefix_cow_reject.bin"
    _write_tiny_dense_v5(checkpoint)
    manifest = _manifest(checkpoint, turboquant=True)
    model = resident_binding.load_model(str(tmp_path), manifest)
    other_model = resident_binding.load_model(str(tmp_path), manifest)
    full = resident_binding.create_session(
        model, {"seed": 0, "kv_cache": {"effective_mode": "full"}}
    )
    off = resident_binding.create_session(
        model, {"seed": 0, "kv_cache": {"effective_mode": "off"}}
    )
    turboquant = resident_binding.create_session(
        model,
        {
            "seed": 0,
            "kv_cache": {
                "effective_mode": "turboquant",
                "turboquant_profile": "mse-3.5",
                "tables": _turboquant_binding_tables(
                    channels=2, num_heads=1, profile="mse-3.5"
                ),
            },
        },
    )
    try:
        assert resident_binding.model_stats(model)["open_sessions"] == 3
        assert resident_binding.model_stats(other_model)["open_sessions"] == 0
        resident_binding.prefill(model, full, [1, 2], 0)
        resident_binding.prefill(model, off, [1], 0)
        resident_binding.prefill(model, turboquant, [1], 0)
        with pytest.raises(RuntimeError, match="non-empty cached prefix"):
            resident_binding.fork_session(
                model, full, {"token_count": 0, "seed": 0}
            )
        with pytest.raises(RuntimeError, match="non-empty cached prefix"):
            resident_binding.fork_session(
                model, full, {"token_count": 3, "seed": 0}
            )
        assert resident_binding.model_stats(model)["open_sessions"] == 3
        with pytest.raises(RuntimeError, match="full-cache"):
            resident_binding.fork_session(
                model, off, {"token_count": 1, "seed": 0}
            )
        assert resident_binding.model_stats(model)["open_sessions"] == 3
        turboquant_child = resident_binding.fork_session(
            model, turboquant, {"token_count": 1, "seed": 0}
        )
        assert resident_binding.model_stats(model)["open_sessions"] == 4
        resident_binding.close_session(model, turboquant_child)
        assert resident_binding.model_stats(model)["open_sessions"] == 3
        resident_binding.close_session(model, turboquant_child)
        assert resident_binding.model_stats(model)["open_sessions"] == 3
        with pytest.raises(ValueError, match="does not belong"):
            resident_binding.fork_session(
                other_model, full, {"token_count": 1, "seed": 0}
            )
        assert resident_binding.model_stats(model)["open_sessions"] == 3
        assert resident_binding.model_stats(other_model)["open_sessions"] == 0
        resident_binding.close_session(model, full)
        assert resident_binding.model_stats(model)["open_sessions"] == 2
        with pytest.raises(RuntimeError, match="handle is closed"):
            resident_binding.fork_session(
                model, full, {"token_count": 1, "seed": 0}
            )
        assert resident_binding.model_stats(model)["open_sessions"] == 2
    finally:
        resident_binding.close_session(model, full)
        resident_binding.close_session(model, off)
        resident_binding.close_session(model, turboquant)
        resident_binding.close_model(other_model)
        resident_binding.close_model(model)


def test_sdk_fork_session_tracks_independent_dense_prefix_state(
    resident_binding: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model_prefix_cow_sdk.bin"
    _write_tiny_dense_v5(checkpoint, nontrivial=True)
    manifest = _manifest(checkpoint, turboquant=True)
    (tmp_path / "native-execution-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with NativeInferenceModel.load(
        tmp_path,
        binding=resident_binding,
        kv_cache=KVCacheConfig(mode="full"),
    ) as model:
        assert model.capabilities.session_prefix_cow is True
        assert model.stats()["session_prefix_cow"] is True
        parent = model.create_session(seed=1)
        parent.prefill([1, 2, 1])
        parent_logits = parent.current_logits()
        left = model.fork_session(parent, seed=11)
        right = model.fork_session(parent, token_count=2, seed=22)
        assert parent.token_ids == (1, 2, 1)
        assert left.token_ids == (1, 2, 1)
        assert right.token_ids == (1, 2)
        assert left.current_logits() == pytest.approx(parent_logits, abs=1.0e-7)
        assert left.stats()["seed"] == 11
        assert right.stats()["seed"] == 22

        logical_only = model.fork_session(parent, seed=23)
        logical_only.truncate(1)
        assert logical_only.stats()["prefix_cow_shared_storage"] is True
        assert logical_only.stats()["prefix_cow_detach_count"] == 0
        assert parent.token_ids == (1, 2, 1)
        logical_only.reset()
        assert logical_only.token_ids == ()
        assert logical_only.stats()["cached_tokens"] == 0
        assert logical_only.stats()["prefix_cow_detach_count"] == 0
        assert logical_only.stats()["prefix_cow_shared_storage"] is True
        assert parent.current_logits() == pytest.approx(parent_logits, abs=1.0e-7)
        logical_only.close()

        left.prefill([1, 2, 1, 2])
        right.prefill([1, 2, 3])
        assert left.token_ids == (1, 2, 1, 2)
        assert right.token_ids == (1, 2, 3)
        assert parent.token_ids == (1, 2, 1)
        assert parent.current_logits() == pytest.approx(parent_logits, abs=1.0e-7)
        assert left.stats()["prefix_cow_detach_count"] == 1
        assert right.stats()["prefix_cow_detach_count"] == 1

        left.cancel()
        assert left.cancelled is True
        assert right.cancelled is False
        assert parent.cancelled is False
        left.reset()
        assert left.cancelled is False
        assert left.token_ids == ()
        assert right.token_ids == (1, 2, 3)
        right.truncate(1)
        assert right.token_ids == (1,)
        assert parent.token_ids == (1, 2, 1)

        # Closing an owner releases only its reference to still-shared native
        # storage; the child retains the complete prefix and valid logits.
        survivor = model.fork_session(parent, seed=33)
        close_peer = model.fork_session(parent, seed=44)
        assert survivor.stats()["prefix_cow_storage_use_count"] == 3
        close_peer.close()
        assert survivor.stats()["prefix_cow_storage_use_count"] == 2
        assert survivor.current_logits() == pytest.approx(parent_logits, abs=1.0e-7)

        # The original parent is also a writer: decode must detach it before
        # committing a new row, leaving the shorter/shared child untouched.
        parent.decode(GenerationConfig(max_new_tokens=1, temperature=0.0))
        assert parent.stats()["prefix_cow_detach_count"] == 1
        assert parent.stats()["prefix_cow_storage_use_count"] == 1
        assert survivor.token_ids == (1, 2, 1)
        assert survivor.current_logits() == pytest.approx(parent_logits, abs=1.0e-7)
        parent.close()
        assert survivor.current_logits() == pytest.approx(parent_logits, abs=1.0e-7)
        assert survivor.stats()["prefix_cow_storage_use_count"] == 1
        assert survivor.stats()["prefix_cow_detach_count"] == 0
        assert right.current_logits()
        with pytest.raises(NativeInferenceClosedError, match="closed"):
            model.fork_session(parent)

        off = model.create_session(kv_cache=KVCacheConfig(mode="off"))
        off.prefill([1])
        with pytest.raises(NativeInferenceCapabilityError, match="full-cache"):
            model.fork_session(off)
        turboquant = model.create_session(
            kv_cache=KVCacheConfig(mode="turboquant", turboquant_profile="mse-3.5")
        )
        turboquant.prefill([1])
        turboquant_child = model.fork_session(turboquant)
        assert turboquant_child.token_ids == (1,)
        assert turboquant_child.stats()["effective_cache"] == "turboquant"
        turboquant_child.close()
        with pytest.raises(TypeError, match="token_count"):
            model.fork_session(right, token_count=True)
        with pytest.raises(ValueError, match="non-empty prefix"):
            model.fork_session(right, token_count=0)


def test_resident_binding_temperature_and_cache_contracts_fail_closed(
    resident_binding: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model_3.bin"
    _write_tiny_dense_v5(checkpoint)
    manifest = _manifest(checkpoint)
    assert resident_binding.resident_inference_abi_version() == 1
    capabilities = resident_binding.resident_inference_capabilities()
    assert capabilities["resident_inference"] is True
    assert capabilities["lossless_kv_cache"] is True
    assert capabilities["turboquant_kv_cache"] is True
    assert capabilities["current_logits_exact_prefill"] is True
    assert capabilities["function_tools"] is False
    assert capabilities["structured_output"] is False
    assert capabilities["session_prefix_cow_abi"] == {
        "version": 1,
        "operation": "fork_session",
        "profiles": [
            "dense-full-cache-kv-final-hidden-v1",
            "dense-cpu-turboquant-mse-qjl-packed-kv-final-hidden-v1",
            "llama-full-cache-gqa-kv-final-hidden-v1",
            "standard-moe-full-cache-gqa-kv-final-hidden-v1",
        ],
    }

    model = resident_binding.load_model(str(tmp_path), manifest)
    try:
        with pytest.raises(ValueError, match="requires deterministic codec tables"):
            resident_binding.create_session(
                model, {"seed": 0, "kv_cache": {
                    "effective_mode": "turboquant",
                    "turboquant_profile": "mse-3.5",
                }},
            )
        session = resident_binding.create_session(
            model, {"seed": 0, "kv_cache": {"effective_mode": "off"}}
        )
        try:
            resident_binding.prefill(model, session, [1], 0)
            base = {
                "top_k": 1,
                "top_p": 1.0,
                "seed": None,
                "stop_token_ids": [],
                "strict_model_compute": False,
            }
            with pytest.raises(RuntimeError, match="finite and non-negative"):
                resident_binding.decode_one(model, session, {**base, "temperature": -1.0})
            with pytest.raises(RuntimeError, match="finite and non-negative"):
                resident_binding.decode_one(model, session, {**base, "temperature": float("nan")})
            tiny_positive = resident_binding.decode_one(
                model, session, {**base, "temperature": 1.0e-50}
            )
            assert tiny_positive["token_id"] == 1
        finally:
            resident_binding.close_session(model, session)
    finally:
        resident_binding.close_model(model)

    (tmp_path / "native-execution-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(NativeInferenceCapabilityError, match="TurboQuant"):
        NativeInferenceModel.load(
            tmp_path,
            binding=resident_binding,
            kv_cache=KVCacheConfig(mode="turboquant"),
        )


def test_lossless_full_cache_matches_recompute_and_reports_real_bytes(
    resident_binding: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model_cache_parity.bin"
    _write_tiny_dense_v5(checkpoint, nontrivial=True)
    manifest = _manifest(checkpoint)
    model = resident_binding.load_model(str(tmp_path), manifest)
    off = resident_binding.create_session(
        model, {"seed": 19, "kv_cache": {"effective_mode": "off"}}
    )
    full = resident_binding.create_session(
        model, {"seed": 19, "kv_cache": {"effective_mode": "full"}}
    )
    full_repeat = resident_binding.create_session(
        model, {"seed": 19, "kv_cache": {"effective_mode": "full"}}
    )
    strict = {
        "temperature": 0.0,
        "top_k": None,
        "top_p": 1.0,
        "seed": None,
        "stop_token_ids": [],
        "strict_model_compute": True,
    }
    try:
        prompt = [1, 2, 3]
        resident_binding.prefill(model, off, prompt, 0)
        resident_binding.prefill(model, full, prompt, 0)
        resident_binding.prefill(model, full_repeat, prompt, 0)
        assert resident_binding.current_logits(model, full) == pytest.approx(
            resident_binding.current_logits(model, off), abs=1.0e-7
        )
        for _ in range(3):
            recomputed = resident_binding.decode_one(model, off, strict)
            cached = resident_binding.decode_one(model, full, strict)
            repeated = resident_binding.decode_one(model, full_repeat, strict)
            assert cached["token_id"] == recomputed["token_id"]
            assert repeated == cached
            assert cached["selected_logit"] == pytest.approx(
                recomputed["selected_logit"], abs=1.0e-7
            )
            assert resident_binding.current_logits(model, full) == pytest.approx(
                resident_binding.current_logits(model, off), abs=1.0e-7
            )

        off_stats = resident_binding.session_stats(model, off)
        full_stats = resident_binding.session_stats(model, full)
        assert off_stats["recompute_full_prefix"] is True
        assert off_stats["cached_tokens"] == 0
        assert off_stats["decode_rows_processed"] == 3 + 4 + 5
        assert full_stats["effective_cache"] == "full"
        assert full_stats["recompute_full_prefix"] is False
        assert full_stats["cached_tokens"] == 6
        # One layer: K, V, and final hidden; each row has 2 fp32 channels.
        assert full_stats["cache_bytes"] == 6 * 3 * 2 * 4
        assert full_stats["cache_capacity_bytes"] == 8 * 3 * 2 * 4
        assert full_stats["uncompressed_cache_bytes"] == full_stats["cache_bytes"]
        assert full_stats["compression_ratio"] == 1.0
        assert full_stats["decode_rows_processed"] == 3
        assert full_stats["strict_model_compute"] is True
        assert full_stats["lossy_cache"] is False
        assert full_stats["fallback_reason"] is None

        resident_binding.truncate_session(model, off, 2)
        resident_binding.truncate_session(model, full, 2)
        resident_binding.truncate_session(model, full_repeat, 2)
        assert resident_binding.current_logits(model, full) == pytest.approx(
            resident_binding.current_logits(model, off), abs=1.0e-7
        )
        resident_binding.reset_session(model, off)
        resident_binding.reset_session(model, full)
        resident_binding.reset_session(model, full_repeat)
        resident_binding.prefill(model, off, [2, 1], 0)
        resident_binding.prefill(model, full, [2, 1], 0)
        resident_binding.prefill(model, full_repeat, [2, 1], 0)
        assert resident_binding.current_logits(model, full) == pytest.approx(
            resident_binding.current_logits(model, off), abs=1.0e-7
        )
    finally:
        resident_binding.close_session(model, off)
        resident_binding.close_session(model, full)
        resident_binding.close_session(model, full_repeat)
        resident_binding.close_model(model)


@pytest.mark.parametrize("profile", ["mse-3.5", "qjl-3.5"])
def test_native_turboquant_codec_matches_portable_reference(
    resident_binding: ModuleType,
    profile: str,
) -> None:
    dimension = 8
    tables = _turboquant_binding_tables(
        channels=dimension,
        num_heads=1,
        profile=profile,
    )
    key = [0.5, -0.2, 0.8, -0.9, 0.1, 0.6, -0.4, 0.3]
    value = [0.25, -0.5, 1.0, 0.75, -0.125, 0.4, -0.9, 0.2]
    query = [-0.1, 0.7, 0.2, -0.3, 0.9, 0.1, -0.5, 0.4]
    codec = TurboQuantReferenceCodec(
        dimension,
        profile=profile,
        seed=0,
        outlier_indices=range(0, dimension, 2),
    )
    reference_key = codec.encode_key(key)
    reference_value = codec.encode_value(value)
    native = resident_binding.turboquant_codec_probe(
        {"turboquant_profile": profile, "tables": tables},
        key,
        value,
        query,
    )

    assert native["key_indices"] == reference_key.packed_indices
    assert native["value_indices"] == reference_value.packed_indices
    assert native["qjl_signs"] == (reference_key.qjl_signs or b"")
    assert native["key_norm"] == pytest.approx(reference_key.norm, abs=1.0e-7)
    assert native["value_norm"] == pytest.approx(reference_value.norm, abs=1.0e-7)
    assert native["residual_norm"] == pytest.approx(
        reference_key.residual_norm or 0.0,
        abs=2.0e-7,
    )
    assert native["key_inner_product"] == pytest.approx(
        codec.key_inner_product(query, reference_key),
        abs=2.0e-6,
    )
    assert native["decoded_value"] == pytest.approx(
        codec.decode_value(reference_value),
        abs=2.0e-6,
    )


def test_native_turboquant_codec_rejects_degenerate_or_noncanonical_tables(
    resident_binding: ModuleType,
) -> None:
    key = [0.1] * 8
    value = [0.2] * 8
    query = [0.3] * 8

    def probe(profile: str, tables: dict[str, Any]) -> None:
        resident_binding.turboquant_codec_probe(
            {"turboquant_profile": profile, "tables": tables},
            key,
            value,
            query,
        )

    base = _turboquant_binding_tables(channels=8, num_heads=1, profile="mse-3.5")
    zero_rotation = json.loads(json.dumps(base))
    zero_rotation["rotation"] = [0.0] * 64
    with pytest.raises(RuntimeError, match="orthonormal"):
        probe("mse-3.5", zero_rotation)

    scaled_rotation = json.loads(json.dumps(base))
    scaled_rotation["rotation"][:8] = [
        value * 2.0 for value in scaled_rotation["rotation"][:8]
    ]
    with pytest.raises(RuntimeError, match="orthonormal"):
        probe("mse-3.5", scaled_rotation)

    wrong_outliers = json.loads(json.dumps(base))
    wrong_outliers["value_bit_widths"][0] = 3
    wrong_outliers["value_bit_widths"][1] = 4
    wrong_outliers["key_bit_widths"] = list(wrong_outliers["value_bit_widths"])
    with pytest.raises(RuntimeError, match="even-channel outlier"):
        probe("mse-3.5", wrong_outliers)

    duplicate_centroid = json.loads(json.dumps(base))
    duplicate_centroid["centroids"][3][1] = duplicate_centroid["centroids"][3][0]
    with pytest.raises(RuntimeError, match="ordering"):
        probe("mse-3.5", duplicate_centroid)

    qjl = _turboquant_binding_tables(channels=8, num_heads=1, profile="qjl-3.5")
    qjl["qjl_projection"] = [0.0] * 64
    with pytest.raises(RuntimeError, match="nondegenerate"):
        probe("qjl-3.5", qjl)


def test_model_stats_report_effective_and_geometry_gated_turboquant_support(
    resident_binding: ModuleType,
    tmp_path: Path,
) -> None:
    even_checkpoint = tmp_path / "even.bin"
    _write_tiny_dense_v5(even_checkpoint)
    even_manifest = _manifest(even_checkpoint)
    even_model = resident_binding.load_model(str(tmp_path), even_manifest)
    try:
        assert resident_binding.model_stats(even_model)["turboquant_kv_cache"] is True
    finally:
        resident_binding.close_model(even_model)

    odd_checkpoint = tmp_path / "odd.bin"
    _write_tiny_dense_v5(odd_checkpoint, channels=3)
    odd_manifest = _manifest(odd_checkpoint, turboquant=True)
    odd_model = resident_binding.load_model(str(tmp_path), odd_manifest)
    try:
        assert resident_binding.model_stats(odd_model)["turboquant_kv_cache"] is False
        even_tables = _turboquant_binding_tables(
            channels=2,
            num_heads=1,
            profile="mse-3.5",
        )
        with pytest.raises(RuntimeError, match="geometry"):
            resident_binding.create_session(
                odd_model,
                {
                    "seed": 0,
                    "kv_cache": {
                        "effective_mode": "turboquant",
                        "turboquant_profile": "mse-3.5",
                        "tables": even_tables,
                    },
                },
            )
    finally:
        resident_binding.close_model(odd_model)


def test_inflight_turboquant_cancellation_is_recoverable_and_rng_transactional(
    resident_binding: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "cancellable.bin"
    _write_tiny_dense_v5(
        checkpoint,
        max_seq_len=24,
        vocab_size=16,
        num_layers=4,
        num_heads=16,
        channels=192,
    )
    manifest = _manifest(checkpoint, turboquant=True)
    (tmp_path / "native-execution-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    cache = KVCacheConfig(mode="turboquant", turboquant_profile="qjl-3.5")

    with NativeInferenceModel.load(tmp_path, binding=resident_binding, kv_cache=cache) as model:
        with model.create_session(seed=23) as session:
            prefill_errors: list[BaseException] = []

            def run_prefill() -> None:
                try:
                    session.prefill(list(range(12)))
                except BaseException as exc:  # captured for the parent-thread assertion
                    prefill_errors.append(exc)

            prefill_worker = threading.Thread(target=run_prefill)
            prefill_worker.start()
            deadline = time.monotonic() + 5.0
            while model.stats()["forward_calls"] == 0 and time.monotonic() < deadline:
                time.sleep(0.001)
            assert prefill_worker.is_alive(), "fixture completed before cancellation could be observed"
            session.cancel()
            prefill_worker.join(timeout=5.0)
            assert not prefill_worker.is_alive()
            assert len(prefill_errors) == 1
            assert isinstance(prefill_errors[0], NativeInferenceCancelledError)
            cancelled_prefill_stats = session.stats()
            assert cancelled_prefill_stats["token_count"] == 0
            assert cancelled_prefill_stats["cached_tokens"] == 0

            session.reset()
            session.prefill([1])
            baseline_forwards = model.stats()["forward_calls"]
            decode_results: list[Any] = []

            def run_decode() -> None:
                decode_results.append(
                    session.decode(GenerationConfig(max_new_tokens=1, temperature=0.8))
                )

            decode_worker = threading.Thread(target=run_decode)
            decode_worker.start()
            deadline = time.monotonic() + 5.0
            while (
                model.stats()["forward_calls"] == baseline_forwards
                and time.monotonic() < deadline
            ):
                # `forward_calls` increments before native layer work starts.
                # Poll without a fixed sleep so a sub-millisecond fixture
                # cannot complete entirely between two observations.
                pass
            assert decode_worker.is_alive(), "fixture completed before decode cancellation"
            session.cancel()
            decode_worker.join(timeout=5.0)
            assert not decode_worker.is_alive()
            assert len(decode_results) == 1
            cancelled_decode = decode_results[0]
            assert cancelled_decode.finish_reason == "cancelled"
            assert cancelled_decode.cancelled is True
            assert cancelled_decode.token_ids == ()
            assert session.stats()["token_count"] == 1

            session.reset()
            session.prefill([1])
            retry = session.decode(GenerationConfig(max_new_tokens=1, temperature=0.8))
            with model.create_session(seed=23) as fresh:
                fresh.prefill([1])
                expected = fresh.decode(
                    GenerationConfig(max_new_tokens=1, temperature=0.8)
                )
            assert retry.token_ids == expected.token_ids

            # A forked writer publishes private packed K/V and final-hidden
            # storage before computation. Cancellation after native forward
            # begins must restore the original shared allocation and detach
            # telemetry for both prefill and decode paths.
            session.reset()
            session.prefill([1])
            assert model.capabilities.session_prefix_cow_cpu_turboquant is True
            parent_calls = session.stats()[
                "turboquant_cpu_compressed_attention_calls"
            ]

            prefill_child = model.fork_session(session, seed=41)
            prefill_errors = []
            prefill_forward_baseline = model.stats()["forward_calls"]

            def run_forked_prefill() -> None:
                try:
                    prefill_child.prefill([1, *range(2, 13)])
                except BaseException as exc:
                    prefill_errors.append(exc)

            prefill_worker = threading.Thread(target=run_forked_prefill)
            prefill_worker.start()
            deadline = time.monotonic() + 5.0
            while (
                model.stats()["forward_calls"] < prefill_forward_baseline + 2
                and time.monotonic() < deadline
            ):
                pass
            assert prefill_worker.is_alive()
            prefill_child.cancel()
            prefill_worker.join(timeout=5.0)
            assert not prefill_worker.is_alive()
            assert len(prefill_errors) == 1
            assert isinstance(prefill_errors[0], NativeInferenceCancelledError)
            prefill_rolled_back = prefill_child.stats()
            assert prefill_child.token_ids == (1,)
            assert prefill_rolled_back["prefix_cow_storage_use_count"] == 2
            assert prefill_rolled_back["prefix_cow_shared_storage"] is True
            assert prefill_rolled_back["prefix_cow_detach_count"] == 0
            assert prefill_rolled_back["prefix_cow_detached_capacity_bytes"] == 0
            # Compressed-attention calls are attempted-work telemetry, not
            # committed-state telemetry. The cancelled child's real work is
            # retained even though its cache ownership/detach is rolled back.
            assert prefill_rolled_back[
                "turboquant_cpu_compressed_attention_calls"
            ] > 0
            assert session.stats()["turboquant_cpu_compressed_attention_calls"] == parent_calls
            prefill_child.close()

            decode_child = model.fork_session(session, seed=43)
            decode_results = []
            decode_forward_baseline = model.stats()["forward_calls"]

            def run_forked_decode() -> None:
                decode_results.append(
                    decode_child.decode(
                        GenerationConfig(max_new_tokens=1, temperature=0.8)
                    )
                )

            decode_worker = threading.Thread(target=run_forked_decode)
            decode_worker.start()
            deadline = time.monotonic() + 5.0
            while (
                model.stats()["forward_calls"] == decode_forward_baseline
                and time.monotonic() < deadline
            ):
                pass
            assert decode_worker.is_alive()
            decode_child.cancel()
            decode_worker.join(timeout=5.0)
            assert not decode_worker.is_alive()
            assert len(decode_results) == 1
            assert decode_results[0].cancelled is True
            assert decode_results[0].token_ids == ()
            decode_rolled_back = decode_child.stats()
            assert decode_child.token_ids == (1,)
            assert decode_rolled_back["prefix_cow_storage_use_count"] == 2
            assert decode_rolled_back["prefix_cow_shared_storage"] is True
            assert decode_rolled_back["prefix_cow_detach_count"] == 0
            assert decode_rolled_back["prefix_cow_detached_capacity_bytes"] == 0
            assert session.stats()["turboquant_cpu_compressed_attention_calls"] == parent_calls


@pytest.mark.parametrize(
    ("use_qk_norm", "logit_softcap"),
    [(True, 0.0), (False, 0.05), (True, 0.05)],
)
def test_parameter_free_dense_variants_match_formula_and_all_cache_paths(
    resident_binding: ModuleType,
    tmp_path: Path,
    use_qk_norm: bool,
    logit_softcap: float,
) -> None:
    checkpoint = tmp_path / f"variant-{use_qk_norm}-{logit_softcap}.bin"
    _write_tiny_dense_v5(checkpoint, nontrivial=True)
    manifest = _manifest(
        checkpoint,
        turboquant=True,
        use_qk_norm=use_qk_norm,
        logit_softcap=logit_softcap,
    )
    model = resident_binding.load_model(str(tmp_path), manifest)
    tables = _turboquant_binding_tables(
        channels=2,
        num_heads=1,
        profile="qjl-3.5",
    )
    off = resident_binding.create_session(
        model, {"seed": 7, "kv_cache": {"effective_mode": "off"}}
    )
    full = resident_binding.create_session(
        model, {"seed": 7, "kv_cache": {"effective_mode": "full"}}
    )
    first_turboquant = resident_binding.create_session(
        model,
        {
            "seed": 7,
            "kv_cache": {
                "effective_mode": "turboquant",
                "turboquant_profile": "qjl-3.5",
                "tables": tables,
            },
        },
    )
    second_turboquant = resident_binding.create_session(
        model,
        {
            "seed": 7,
            "kv_cache": {
                "effective_mode": "turboquant",
                "turboquant_profile": "qjl-3.5",
                "tables": tables,
            },
        },
    )
    sessions = (off, full, first_turboquant, second_turboquant)
    try:
        for session in sessions:
            resident_binding.prefill(model, session, [1, 2, 3], 0)
        expected = _python_dense_logits(
            checkpoint,
            [1, 2, 3],
            use_qk_norm=use_qk_norm,
            logit_softcap=logit_softcap,
        )
        off_logits = resident_binding.current_logits(model, off)
        full_logits = resident_binding.current_logits(model, full)
        first_turboquant_logits = resident_binding.current_logits(
            model, first_turboquant
        )
        second_turboquant_logits = resident_binding.current_logits(
            model, second_turboquant
        )
        assert off_logits == pytest.approx(expected, abs=2.0e-6)
        assert full_logits == pytest.approx(off_logits, abs=2.0e-6)
        assert first_turboquant_logits == second_turboquant_logits
        assert all(math.isfinite(value) for value in first_turboquant_logits)
        stats = resident_binding.model_stats(model)
        assert stats["use_qk_norm"] is use_qk_norm
        assert stats["qk_norm_eps"] == pytest.approx(1.0e-6)
        assert stats["logit_softcap"] == pytest.approx(logit_softcap)
    finally:
        for session in sessions:
            resident_binding.close_session(model, session)
        resident_binding.close_model(model)


def test_multilayer_multihead_qk_norm_matches_formula_and_lossless_cache(
    resident_binding: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "qknorm-multilayer-multihead.bin"
    _write_tiny_dense_v5(
        checkpoint,
        nontrivial=True,
        num_layers=2,
        num_heads=2,
        channels=8,
    )
    manifest = _manifest(checkpoint, use_qk_norm=True)
    model = resident_binding.load_model(str(tmp_path), manifest)
    off = resident_binding.create_session(
        model, {"seed": 17, "kv_cache": {"effective_mode": "off"}}
    )
    full = resident_binding.create_session(
        model, {"seed": 17, "kv_cache": {"effective_mode": "full"}}
    )
    try:
        prompt = [1, 2, 3]
        resident_binding.prefill(model, off, prompt, 0)
        resident_binding.prefill(model, full, prompt, 0)
        expected = _python_dense_logits(
            checkpoint,
            prompt,
            use_qk_norm=True,
            logit_softcap=0.0,
        )
        off_logits = resident_binding.current_logits(model, off)
        full_logits = resident_binding.current_logits(model, full)
        assert off_logits == pytest.approx(expected, abs=3.0e-6)
        assert full_logits == pytest.approx(off_logits, abs=3.0e-6)

        generation = {
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "seed": None,
            "stop_token_ids": [],
            "strict_model_compute": True,
        }
        off_token = resident_binding.decode_one(model, off, generation)
        full_token = resident_binding.decode_one(model, full, generation)
        assert full_token["token_id"] == off_token["token_id"]
        assert full_token["selected_logit"] == pytest.approx(
            off_token["selected_logit"], abs=3.0e-6
        )
        assert resident_binding.current_logits(model, full) == pytest.approx(
            resident_binding.current_logits(model, off),
            abs=3.0e-6,
        )
    finally:
        resident_binding.close_session(model, off)
        resident_binding.close_session(model, full)
        resident_binding.close_model(model)


@pytest.mark.parametrize("selected_activation", ["gelu", "relu", "silu", "relu2"])
def test_moa_selected_activation_matches_formula_and_all_cache_paths(
    resident_binding: ModuleType,
    tmp_path: Path,
    selected_activation: str,
) -> None:
    checkpoint = tmp_path / f"moa-{selected_activation}.bin"
    _write_tiny_dense_v5(
        checkpoint,
        nontrivial=True,
        max_seq_len=8,
        vocab_size=4,
        num_layers=2,
        num_heads=1,
        channels=4,
    )
    manifest = _manifest(
        checkpoint,
        turboquant=True,
        moa_activation=selected_activation,
        moa_interval=7,
    )
    model = resident_binding.load_model(str(tmp_path), manifest)
    tables = _turboquant_binding_tables(
        channels=4,
        num_heads=1,
        profile="mse-3.5",
    )
    off = resident_binding.create_session(
        model, {"seed": 11, "kv_cache": {"effective_mode": "off"}}
    )
    full = resident_binding.create_session(
        model, {"seed": 11, "kv_cache": {"effective_mode": "full"}}
    )
    first_turboquant = resident_binding.create_session(
        model,
        {
            "seed": 11,
            "kv_cache": {
                "effective_mode": "turboquant",
                "turboquant_profile": "mse-3.5",
                "tables": tables,
            },
        },
    )
    second_turboquant = resident_binding.create_session(
        model,
        {
            "seed": 11,
            "kv_cache": {
                "effective_mode": "turboquant",
                "turboquant_profile": "mse-3.5",
                "tables": tables,
            },
        },
    )
    sessions = (off, full, first_turboquant, second_turboquant)
    try:
        prompt = [1, 2, 3]
        for session in sessions:
            resident_binding.prefill(model, session, prompt, 0)
        expected = _python_dense_logits(
            checkpoint,
            prompt,
            use_qk_norm=False,
            logit_softcap=0.0,
            mlp_activation=selected_activation,
        )
        off_logits = resident_binding.current_logits(model, off)
        full_logits = resident_binding.current_logits(model, full)
        first_turboquant_logits = resident_binding.current_logits(
            model, first_turboquant
        )
        second_turboquant_logits = resident_binding.current_logits(
            model, second_turboquant
        )
        assert off_logits == pytest.approx(expected, abs=4.0e-6)
        assert full_logits == pytest.approx(off_logits, abs=4.0e-6)
        assert first_turboquant_logits == second_turboquant_logits
        assert all(math.isfinite(value) for value in first_turboquant_logits)
        stats = resident_binding.model_stats(model)
        assert stats["activation_mode"] == "moa"
        assert stats["mlp_activation"] == selected_activation
        assert stats["moa_interval"] == 7

        generation = {
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "seed": None,
            "stop_token_ids": [],
            "strict_model_compute": True,
        }
        off_token = resident_binding.decode_one(model, off, generation)
        full_token = resident_binding.decode_one(model, full, generation)
        assert full_token["token_id"] == off_token["token_id"]
        assert full_token["selected_logit"] == pytest.approx(
            off_token["selected_logit"], abs=4.0e-6
        )
        assert resident_binding.current_logits(model, full) == pytest.approx(
            resident_binding.current_logits(model, off), abs=4.0e-6
        )
    finally:
        for session in sessions:
            resident_binding.close_session(model, session)
        resident_binding.close_model(model)


def test_moa_checkpoint_contract_fails_closed(
    resident_binding: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "moa-contract.bin"
    _write_tiny_dense_v5(checkpoint, nontrivial=True, channels=4)

    missing = _manifest(checkpoint, moa_activation="gelu")
    del missing["checkpoint"]["moa"]
    with pytest.raises(RuntimeError, match="source-bound checkpoint.moa"):
        resident_binding.load_model(str(tmp_path), missing)

    wrong_source = _manifest(checkpoint, moa_activation="relu")
    wrong_source["checkpoint"]["moa"]["source_graph_sha256"] = "ef" * 32
    with pytest.raises(RuntimeError, match="source graph SHA-256"):
        resident_binding.load_model(str(tmp_path), wrong_source)

    bad_activation = _manifest(checkpoint, moa_activation="silu")
    bad_activation["checkpoint"]["moa"]["selected_activation"] = "prelu"
    with pytest.raises(RuntimeError, match="selected activation"):
        resident_binding.load_model(str(tmp_path), bad_activation)

    bad_candidates = _manifest(checkpoint, moa_activation="relu2")
    bad_candidates["model"]["template_spec"]["block_spec"]["moa_activations"] = [
        "gelu",
        "relu",
    ]
    with pytest.raises(RuntimeError, match="candidates"):
        resident_binding.load_model(str(tmp_path), bad_candidates)


def test_softcap_is_applied_after_the_tied_output_projection(
    resident_binding: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "softcap-order.bin"
    _write_tiny_dense_v5(checkpoint, nontrivial=True)
    plain_model = resident_binding.load_model(
        str(tmp_path), _manifest(checkpoint, logit_softcap=0.0)
    )
    softcap = 0.05
    capped_model = resident_binding.load_model(
        str(tmp_path), _manifest(checkpoint, logit_softcap=softcap)
    )
    plain = resident_binding.create_session(
        plain_model, {"seed": 0, "kv_cache": {"effective_mode": "off"}}
    )
    capped = resident_binding.create_session(
        capped_model, {"seed": 0, "kv_cache": {"effective_mode": "off"}}
    )
    try:
        resident_binding.prefill(plain_model, plain, [1, 2, 3], 0)
        resident_binding.prefill(capped_model, capped, [1, 2, 3], 0)
        raw = resident_binding.current_logits(plain_model, plain)
        transformed = resident_binding.current_logits(capped_model, capped)
        assert transformed == pytest.approx(
            [softcap * math.tanh(value / softcap) for value in raw],
            abs=2.0e-7,
        )
    finally:
        resident_binding.close_session(plain_model, plain)
        resident_binding.close_session(capped_model, capped)
        resident_binding.close_model(plain_model)
        resident_binding.close_model(capped_model)


@pytest.mark.parametrize(
    ("profile", "bytes_per_token"),
    [("mse-3.5", 18), ("qjl-3.5", 23)],
)
def test_native_turboquant_cache_is_packed_direct_deterministic_and_lossy(
    resident_binding: ModuleType,
    tmp_path: Path,
    profile: str,
    bytes_per_token: int,
) -> None:
    checkpoint = tmp_path / f"model_turboquant_{profile}.bin"
    _write_tiny_dense_v5(checkpoint, nontrivial=True)
    manifest = _manifest(checkpoint, turboquant=True)
    (tmp_path / "native-execution-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    cache = KVCacheConfig(mode="turboquant", turboquant_profile=profile)
    strict = GenerationConfig(max_new_tokens=2, temperature=0.0)

    with NativeInferenceModel.load(
        tmp_path,
        binding=resident_binding,
        kv_cache=cache,
    ) as model:
        with model.create_session(seed=41) as first, model.create_session(seed=41) as second:
            first.prefill([1, 2, 3])
            second.prefill([1, 2, 3])
            first_result = first.decode(strict)
            second_result = second.decode(strict)
            assert first_result.token_ids == second_result.token_ids
            assert first_result.events == second_result.events
            stats = first.stats()
            model_stats = model.stats()

            first.truncate(2)
            second.truncate(2)
            first.prefill([1, 2, 1])
            second.prefill([1, 2, 1])
            assert first.decode(GenerationConfig(temperature=0.0)).token_ids == (
                second.decode(GenerationConfig(temperature=0.0)).token_ids
            )

            first.reset()
            second.reset()
            first.prefill([2, 1])
            second.prefill([2, 1])
            assert first.decode(GenerationConfig(temperature=0.0)).token_ids == (
                second.decode(GenerationConfig(temperature=0.0)).token_ids
            )
            first.cancel()
            cancelled = first.decode(GenerationConfig(temperature=0.0))
            assert cancelled.cancelled is True
            assert cancelled.token_ids == ()
            first.reset()
            first.prefill([2, 1])
            assert first.decode(GenerationConfig(temperature=0.0)).completion_tokens == 1

    assert stats["effective_cache"] == "turboquant"
    assert stats["turboquant_profile"] == profile
    assert stats["cached_tokens"] == 5
    assert stats["cache_bytes"] == 5 * bytes_per_token
    assert stats["uncompressed_cache_bytes"] == 5 * 24
    assert stats["cache_bytes"] < stats["uncompressed_cache_bytes"]
    assert stats["compression_ratio"] == pytest.approx(24 / bytes_per_token)
    assert stats["decode_rows_processed"] == 2
    assert stats["strict_model_compute"] is True
    assert stats["lossy_cache"] is True
    assert stats["recompute_full_prefix"] is False
    assert stats["fallback_reason"] is None
    assert model_stats["turboquant_table_load_count"] == 1


def test_sdk_auto_cache_reuses_exact_prefix_and_rebuilds_after_front_trim(
    resident_binding: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model_sdk_full.bin"
    _write_tiny_dense_v5(checkpoint, nontrivial=True)
    manifest = _manifest(checkpoint)
    (tmp_path / "native-execution-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with NativeInferenceModel.load(tmp_path, binding=resident_binding) as model:
        with model.create_session() as session:
            assert session.prefill([1, 2, 3])["prefilled_tokens"] == 3
            assert session.prefill([1, 2, 0])["prefix_reused"] == 2
            assert session.prefill([2, 0])["prefix_reused"] == 0
            stats = session.stats()
            assert stats["effective_cache"] == "full"
            assert stats["cached_tokens"] == 2
            assert stats["prefix_tokens_reused"] == 2
            assert stats["cache_bytes"] == 2 * 3 * 2 * 4

            session.prefill([0] * 7)
            session.decode(GenerationConfig(max_new_tokens=1, temperature=0.0))
            assert session.stats()["cached_tokens"] == 8
            with pytest.raises(RuntimeError, match="context window"):
                session.decode(GenerationConfig(max_new_tokens=1, temperature=0.0))


def test_native_v5_graph_migration_emits_a_directly_loadable_resident_artifact(
    resident_binding: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = build_model_spec_from_config(
        {
            "preset": "gpt2",
            "num_layers": 1,
            "model_dim": 2,
            "num_heads": 1,
            "num_kv_heads": 1,
            "multiple_of": 1,
            "vocab_size": 4,
        },
        preview_defaults=True,
    )
    graph = build_gpt_root_graph(name="resident_migration", model_spec=spec)
    graph_path = tmp_path / "graph.json"
    graph_bytes = json.dumps(graph.to_dict(), indent=2).encode("utf-8")
    graph_path.write_bytes(graph_bytes)
    checkpoint = tmp_path / "model_00000005.bin"
    _write_tiny_dense_v5(checkpoint, nontrivial=True)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    artifact = tmp_path / "artifact"
    migration = migrate_graph_to_native(
        graph_path,
        weights_path=checkpoint,
        output_dir=artifact,
    )
    assert migration.report.compatible is True
    assert migration.manifest is not None
    assert migration.manifest.capabilities["resident_inference"] is True
    assert migration.manifest.capabilities["lossless_kv_cache"] is True
    assert migration.manifest.capabilities["turboquant_kv_cache"] is True
    assert migration.manifest.capabilities["turboquant_tile_attention"] is True
    assert migration.manifest.capabilities["serve"] is True
    assert migration.manifest.kernel_abi["resident_inference"] == {
        "version": 1,
        "status": "ready",
    }
    assert migration.manifest.kernel_abi["turboquant_cache"] == {
        "version": 1,
        "status": "ready",
        "backend": "cpu-reference-packed",
    }
    assert migration.manifest.kernel_abi["turboquant_tile_attention"] == {
        "symbol": "nfn_native_tile_turboquant_attention_forward_v1",
        "version": 1,
        "status": "ready",
        "backend": "tile-cuda-hybrid",
    }
    assert migration.manifest.checkpoint is not None
    assert migration.manifest.checkpoint["artifact_path"] == "model.bin"
    assert migration.manifest.checkpoint["target_sha256"] == checkpoint_sha
    assert migration.manifest.context_limits["max_context_tokens"] == 8
    assert (artifact / "model.bin").read_bytes() == checkpoint.read_bytes()
    assert graph_path.read_bytes() == graph_bytes
    assert checkpoint.exists()

    with NativeInferenceModel.load(artifact, binding=resident_binding) as model:
        with model.create_session() as session:
            session.prefill([1, 2])
            result = session.decode(GenerationConfig(max_new_tokens=2, temperature=0.0))
            assert len(result.token_ids) == 2
            assert session.stats()["effective_cache"] == "full"

    # Prove the capability is consumable by the production serving lifecycle,
    # not merely a manifest bit.  Presentation is injected because this tiny
    # numerical fixture deliberately has no real tokenizer asset.
    from neuralfn import native_serve

    class TinyCodec:
        name = "tiny-fixture"

        @staticmethod
        def encode(_text: str) -> tuple[int, ...]:
            return (1,)

        @staticmethod
        def decode(token_ids: Sequence[int]) -> str:
            return "".join(str(int(token)) for token in token_ids)

    monkeypatch.setattr(native_serve, "_load_text_codec", lambda _manifest: TinyCodec())
    runtime = native_serve.NativeServingRuntime.load(
        native_serve.NativeServeConfig(
            artifact=artifact,
            served_model_name="tiny-resident",
            max_output_tokens=2,
            chat_template="plain_roles",
        ),
        binding=resident_binding,
    )
    try:
        prepared = runtime.prepare_chat(
            {
                "model": "tiny-resident",
                "messages": [{"role": "user", "content": "hello"}],
                "max_completion_tokens": 2,
                "temperature": 0.0,
            }
        )
        completed = runtime.complete(prepared)
        assert completed.native.completion_tokens == 2
        assert completed.text
    finally:
        runtime.close()


def test_source_bound_moa_migration_is_portable_and_preserves_session_lifecycle(
    resident_binding: ModuleType,
    tmp_path: Path,
) -> None:
    from tests.test_native_moa_checkpoint import _write_bundle

    metadata, graph_path, _model, _payload = _write_bundle(
        tmp_path / "source",
        activation="relu2",
    )
    source_checkpoint = metadata.with_suffix("").with_suffix(".bin")
    source_done = metadata.with_name("DONE_00000007")
    artifact = tmp_path / "artifact"

    migration = migrate_graph_to_native(
        graph_path,
        weights_path=metadata,
        output_dir=artifact,
    )

    assert migration.report.compatible is True
    assert migration.manifest is not None
    assert migration.manifest.capabilities["resident_inference"] is True
    assert migration.manifest.capabilities["lossless_kv_cache"] is True
    assert migration.manifest.capabilities["turboquant_kv_cache"] is True
    assert migration.manifest.capabilities["turboquant_tile_attention"] is True
    assert migration.manifest.capabilities["serve"] is True
    assert migration.manifest.checkpoint is not None
    assert migration.manifest.checkpoint["moa"]["selected_activation"] == "relu2"
    assert migration.manifest.checkpoint["moa"]["interval"] == 50
    assert migration.manifest.checkpoint["moa"]["source_graph_sha256"] == hashlib.sha256(
        graph_path.read_bytes()
    ).hexdigest()
    assert (artifact / "model.bin").read_bytes() == source_checkpoint.read_bytes()

    # The migrated artifact is self-contained. Provenance paths may disappear
    # after migration without causing the resident engine to reopen them.
    metadata.unlink()
    source_checkpoint.unlink()
    source_done.unlink()

    with NativeInferenceModel.load(
        artifact,
        binding=resident_binding,
        kv_cache=KVCacheConfig(mode="full"),
    ) as model:
        stats = model.stats()
        assert stats["activation_mode"] == "moa"
        assert stats["mlp_activation"] == "relu2"
        assert stats["moa_interval"] == 50
        with model.create_session(seed=31) as left, model.create_session(seed=37) as right:
            left.prefill([1, 2, 3])
            right.prefill([3, 2])
            assert left.stats()["cached_tokens"] == 3
            assert right.stats()["cached_tokens"] == 2

            left.truncate(2)
            assert left.stats()["token_count"] == 2
            assert right.stats()["token_count"] == 2
            result = left.decode(
                GenerationConfig(max_new_tokens=1, temperature=0.0)
            )
            assert result.completion_tokens == 1
            assert left.stats()["cached_tokens"] == 3
            assert right.stats()["cached_tokens"] == 2

            left.reset()
            assert left.stats()["token_count"] == 0
            assert left.stats()["cached_tokens"] == 0
            assert right.stats()["token_count"] == 2

    stdout = io.StringIO()
    stderr = io.StringIO()
    assert run_native_artifact_cli(
        NativeArtifactCLIConfig(
            artifact=artifact,
            prompt_token_ids=(1, 2),
            max_new_tokens=1,
            temperature=0.0,
            kv_cache=KVCacheConfig(mode="full"),
        ),
        interactive=False,
        binding=resident_binding,
        stdout=stdout,
        stderr=stderr,
    ) == 0
    assert stdout.getvalue() == "0\n"
    assert "rendering generated token IDs" in stderr.getvalue()


@pytest.mark.parametrize(
    ("preset", "use_qk_norm", "logit_softcap"),
    [
        ("gpt2_qknorm", True, 0.0),
        ("gpt2_stable", True, 0.0),
        ("gpt2_softcap", False, 30.0),
    ],
)
def test_parameter_free_dense_variant_migrations_are_resident_loadable(
    resident_binding: ModuleType,
    tmp_path: Path,
    preset: str,
    use_qk_norm: bool,
    logit_softcap: float,
) -> None:
    spec = build_model_spec_from_config(
        {
            "preset": preset,
            "num_layers": 1,
            "model_dim": 2,
            "num_heads": 1,
            "num_kv_heads": 1,
            "multiple_of": 1,
            "vocab_size": 4,
        },
        preview_defaults=True,
    )
    graph = build_gpt_root_graph(name=f"resident_{preset}", model_spec=spec)
    graph_path = tmp_path / f"{preset}.json"
    graph_path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")
    checkpoint = tmp_path / f"{preset}.bin"
    _write_tiny_dense_v5(checkpoint, nontrivial=True)
    artifact = tmp_path / f"{preset}-artifact"

    migration = migrate_graph_to_native(
        graph_path,
        weights_path=checkpoint,
        output_dir=artifact,
    )
    assert migration.report.compatible is True
    assert migration.manifest is not None
    assert migration.manifest.capabilities["resident_inference"] is True
    assert migration.manifest.capabilities["lossless_kv_cache"] is True
    assert migration.manifest.capabilities["turboquant_kv_cache"] is True
    assert migration.manifest.capabilities["serve"] is True

    with NativeInferenceModel.load(
        artifact,
        binding=resident_binding,
        kv_cache=KVCacheConfig(mode="turboquant", turboquant_profile="mse-3.5"),
    ) as model:
        stats = model.stats()
        assert stats["use_qk_norm"] is use_qk_norm
        assert stats["logit_softcap"] == pytest.approx(logit_softcap)
        with model.create_session(seed=9) as session:
            session.prefill([1, 2])
            result = session.decode(
                GenerationConfig(max_new_tokens=1, temperature=0.0)
            )
            assert result.completion_tokens == 1
            assert session.stats()["effective_cache"] == "turboquant"


@pytest.mark.parametrize("preset", ["gpt2_megakernel", "gpt2_zloss"])
def test_named_plain_dense_variant_migrations_are_materialized_and_loadable(
    resident_binding: ModuleType,
    tmp_path: Path,
    preset: str,
) -> None:
    spec = build_model_spec_from_config(
        {
            "preset": preset,
            "num_layers": 1,
            "model_dim": 2,
            "num_heads": 1,
            "num_kv_heads": 1,
            "multiple_of": 1,
            "vocab_size": 4,
        },
        preview_defaults=True,
    )
    graph = build_gpt_root_graph(name=f"resident_{preset}", model_spec=spec)
    graph_path = tmp_path / f"{preset}.json"
    graph_path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")
    checkpoint = tmp_path / f"{preset}.bin"
    _write_tiny_dense_v5(checkpoint, nontrivial=True)
    artifact = tmp_path / f"{preset}-artifact"

    migration = migrate_graph_to_native(
        graph_path,
        weights_path=checkpoint,
        output_dir=artifact,
    )
    assert migration.report.compatible is True
    assert migration.manifest is not None
    assert migration.output_dir == artifact.resolve()
    assert (artifact / "native-execution-manifest.json").is_file()
    assert (artifact / "model.bin").is_file()
    assert migration.manifest.capabilities["resident_inference"] is True
    assert migration.manifest.capabilities["lossless_kv_cache"] is True
    assert migration.manifest.capabilities["turboquant_kv_cache"] is True
    assert migration.manifest.capabilities["serve"] is True

    with NativeInferenceModel.load(
        artifact,
        binding=resident_binding,
        kv_cache=KVCacheConfig(mode="full"),
    ) as model:
        with model.create_session(seed=29) as session:
            session.prefill([1, 2])
            result = session.decode(
                GenerationConfig(max_new_tokens=1, temperature=0.0)
            )
            assert result.completion_tokens == 1
            assert session.stats()["effective_cache"] == "full"


@pytest.mark.parametrize(
    ("container", "field", "expected_error"),
    [
        ("block_spec", "use_qk_norm", "use_qk_norm"),
        ("template_spec", "logit_softcap", "logit_softcap"),
        ("template_spec", "tie_embeddings", "tied token/output"),
        ("block_spec", "linear_bias", "biased linear"),
        ("block_spec", "dropout_p", "dropout_p"),
        ("block_spec", "activation_mode", "activation_mode"),
    ],
)
def test_missing_resident_contract_fields_cannot_materialize_ready_checkpoint(
    tmp_path: Path,
    container: str,
    field: str,
    expected_error: str,
) -> None:
    spec = build_model_spec_from_config(
        {
            "preset": "gpt2",
            "num_layers": 1,
            "model_dim": 2,
            "num_heads": 1,
            "num_kv_heads": 1,
            "multiple_of": 1,
            "vocab_size": 4,
        },
        preview_defaults=True,
    )
    payload = build_gpt_root_graph(
        name=f"resident_missing_{field}", model_spec=spec
    ).to_dict()
    template_spec = payload["torch_config"]["template_spec"]
    target = template_spec if container == "template_spec" else template_spec["block_spec"]
    del target[field]
    graph_path = tmp_path / f"missing-{field}.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    checkpoint = tmp_path / f"missing-{field}.bin"
    _write_tiny_dense_v5(checkpoint, nontrivial=True)
    artifact = tmp_path / f"missing-{field}-artifact"

    with pytest.raises(ValueError, match=expected_error):
        migrate_graph_to_native(
            graph_path,
            weights_path=checkpoint,
            output_dir=artifact,
        )
    assert not artifact.exists()


def test_resident_binding_rejects_checkpoint_path_escape(
    resident_binding: ModuleType,
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "outside-model.bin"
    _write_tiny_dense_v5(outside)
    manifest = _manifest(outside, artifact_path="../outside-model.bin")
    with pytest.raises(RuntimeError, match="escapes the artifact root"):
        resident_binding.load_model(str(tmp_path), manifest)


def test_resident_binding_revalidates_graph_contract_and_checkpoint_geometry(
    resident_binding: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model_contract.bin"
    _write_tiny_dense_v5(checkpoint)
    manifest = _manifest(checkpoint)

    qk_norm = json.loads(json.dumps(manifest))
    qk_norm["model"]["template_spec"]["block_spec"]["use_qk_norm"] = True
    with pytest.raises(RuntimeError, match="active topology"):
        resident_binding.load_model(str(tmp_path), qk_norm)

    wrong_qk_epsilon = _manifest(checkpoint, use_qk_norm=True)
    _manifest_topology_node(wrong_qk_epsilon, "qk_norm")["module_config"]["eps"] = 1.0e-5
    with pytest.raises(RuntimeError, match="eps=1e-6"):
        resident_binding.load_model(str(tmp_path), wrong_qk_epsilon)

    wrong_softcap = _manifest(checkpoint, logit_softcap=30.0)
    _manifest_topology_node(wrong_softcap, "logit_softcap")["module_config"]["softcap"] = 20.0
    with pytest.raises(RuntimeError, match="does not match"):
        resident_binding.load_model(str(tmp_path), wrong_softcap)

    wrong_geometry = json.loads(json.dumps(manifest))
    wrong_geometry["model"]["template_spec"]["model_dim"] = 4
    with pytest.raises(RuntimeError, match="geometry does not match"):
        resident_binding.load_model(str(tmp_path), wrong_geometry)

    wrong_context = json.loads(json.dumps(manifest))
    wrong_context["context_limits"]["max_context_tokens"] = 16
    with pytest.raises(RuntimeError, match="max_context_tokens does not match"):
        resident_binding.load_model(str(tmp_path), wrong_context)


@pytest.mark.parametrize("variant", ["qk_norm", "logit_softcap"])
def test_resident_binding_rejects_parameter_free_variant_dataflow_bypass(
    resident_binding: ModuleType,
    tmp_path: Path,
    variant: str,
) -> None:
    checkpoint = tmp_path / f"{variant}-bypass.bin"
    _write_tiny_dense_v5(checkpoint, nontrivial=True)
    manifest = _manifest(
        checkpoint,
        use_qk_norm=variant == "qk_norm",
        logit_softcap=30.0 if variant == "logit_softcap" else 0.0,
    )
    rewired = 0
    for graph in manifest["topology"]["graphs"]:
        for edge in graph.get("edges", []):
            if variant == "qk_norm" and edge["src_node"].endswith("/nodes/qk_norm"):
                if not edge["dst_node"].endswith("/nodes/sdpa"):
                    continue
                replacement = "q_heads" if edge["dst_port"] == 0 else "k_heads"
                edge["src_node"] = f"{graph['path']}/nodes/{replacement}"
                edge["src_port"] = 0
                rewired += 1
            elif (
                variant == "logit_softcap"
                and edge["src_node"].endswith("/nodes/softcap")
                and edge["dst_node"].endswith("/nodes/ce")
            ):
                edge["src_node"] = f"{graph['path']}/nodes/tied_lm_head"
                edge["src_port"] = 0
                rewired += 1
    assert rewired > 0

    with pytest.raises(RuntimeError, match="topology|dataflow"):
        resident_binding.load_model(str(tmp_path), manifest)


def test_training_binding_one_shot_surface_remains_available() -> None:
    source = (ROOT / "neuralfn" / "csrc" / "native_gpt2" / "binding.cpp").read_text(
        encoding="utf-8"
    )
    assert '{"run_gpt", run_gpt' in source
    assert '{"run_infer", run_gpt_capture' in source
    build_all = (ROOT / "tools" / "build_native_gpt2_all.sh").read_text(encoding="utf-8")
    assert "build_native_inference_binding.sh" in build_all
    assert "NFN_NATIVE_INFERENCE_BINDING_OUT" in build_all
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"csrc/native_gpt2/*.h"' in pyproject
