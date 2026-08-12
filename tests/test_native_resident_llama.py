from __future__ import annotations

import hashlib
import importlib.util
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
from typing import Any, Callable, Sequence

import pytest
import torch
import torch.nn.functional as F

from neuralfn.graph import NeuronGraph
from neuralfn.native_family_checkpoint import (
    NATIVE_FAMILY_LLAMA_CHECKPOINT_FORMAT,
    NATIVE_FAMILY_LLAMA_INFERENCE_SCHEMA,
)
from neuralfn.native_inference import GenerationConfig, KVCacheConfig, NativeInferenceModel
from neuralfn.native_cli import NativeArtifactCLIConfig, run_native_artifact_cli
from neuralfn.native_ir import migrate_graph_to_native
from neuralfn.torch_backend import CompiledTorchGraph
from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY: dict[str, int | float] = {
    "max_seq_len": 8,
    "vocab_size": 11,
    "padded_vocab_size": 16,
    "num_layers": 2,
    "model_dim": 8,
    "hidden_dim": 24,
    "num_heads": 4,
    "num_kv_heads": 2,
    "head_dim": 2,
    "rope_theta": 10_000.0,
    "rope_scaling_factor": 1.0,
    "rms_norm_eps": 1.0e-6,
}
SEMANTICS: dict[str, Any] = {
    "norm_type": "rmsnorm",
    "mlp_type": "swiglu",
    "pos_encoding": "rope",
    "attention_variant": "dense",
    "residual_type": "add",
    "linear_bias": False,
    "dropout_p": 0.0,
    "tie_embeddings": False,
}


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _elements(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _llama_layout() -> list[tuple[str, tuple[int, ...]]]:
    d = int(GEOMETRY["model_dim"])
    f = int(GEOMETRY["hidden_dim"])
    vp = int(GEOMETRY["padded_vocab_size"])
    kv = int(GEOMETRY["num_kv_heads"]) * int(GEOMETRY["head_dim"])
    rows: list[tuple[str, tuple[int, ...]]] = [
        ("token_embedding.weight", (vp, d)),
        ("final_norm.weight", (d,)),
        ("lm_head.weight", (vp, d)),
    ]
    for layer in range(int(GEOMETRY["num_layers"])):
        prefix = f"layers.{layer}."
        rows.extend(
            (
                (f"{prefix}attention_norm.weight", (d,)),
                (f"{prefix}q_proj.weight", (d, d)),
                (f"{prefix}k_proj.weight", (kv, d)),
                (f"{prefix}v_proj.weight", (kv, d)),
                (f"{prefix}attention_out.weight", (d, d)),
                (f"{prefix}ffn_norm.weight", (d,)),
                (f"{prefix}ffn_gate_up.weight", (2, f, d)),
                (f"{prefix}ffn_down.weight", (d, f)),
            )
        )
    return rows


def _fixture_value(name: str, global_index: int, local_index: int) -> float:
    if "norm.weight" in name:
        return _f32(0.92 + 0.06 * math.sin(0.17 * global_index + 0.11 * local_index))
    scale = 0.14 if name in {"token_embedding.weight", "lm_head.weight"} else 0.075
    return _f32(
        scale * math.sin(0.31 * global_index + 0.07 * local_index)
        + 0.025 * math.cos(0.13 * global_index - 0.19 * local_index)
    )


def _write_llama_checkpoint(root: Path) -> tuple[Path, Path]:
    metadata = root / "llama_native_family_model_00000000.json"
    sidecar = root / "llama_native_family_parameters_00000000.f32"
    done = root / "llama_native_family_model_DONE"
    tensor_rows: list[dict[str, Any]] = []
    legacy_buffers: list[dict[str, Any]] = []
    values: list[float] = []
    byte_offset = 0
    for name, shape in _llama_layout():
        count = _elements(shape)
        start = len(values)
        values.extend(_fixture_value(name, start + index, index) for index in range(count))
        nbytes = count * 4
        tensor_rows.append(
            {
                "name": name,
                "shape": list(shape),
                "offset": byte_offset,
                "nbytes": nbytes,
                "dtype": "float32",
                "byte_order": "little",
                "layout": "row_major",
            }
        )
        legacy_buffers.append(
            {
                "name": name,
                "offset": byte_offset // 4,
                "byte_offset": byte_offset,
                "elements": count,
                "bytes": nbytes,
                "trainable": True,
            }
        )
        byte_offset += nbytes
    sidecar.write_bytes(struct.pack(f"<{len(values)}f", *values))
    digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    payload = {
        "format": "nfn-native-family-optimizer-checkpoint-v1",
        "model_family": "llama",
        "native_target": "nfn_llama_native_train",
        "template_name": "llama",
        "dataset_alias": "resident-oracle-fixture",
        "checkpoint_kind": "native_family_optimizer_trained_model",
        "inference_supported": True,
        "steps_completed": 2,
        "train_batches_sampled": 2,
        "validation_batches_sampled": 1,
        "vocab_size": GEOMETRY["vocab_size"],
        "parameter_data": {
            "format": "nfn-native-family-float32-parameter-state-v1",
            "path": str(sidecar),
            "parameter_dtype": "float32",
            "parameter_elements": len(values),
            "persisted_parameter_elements": len(values),
            "bytes": byte_offset,
            "storage": "live_family_device_parameter_store_float32_state",
            "trained_parameter_elements": len(values),
            "parameter_update_checksum": 123,
        },
        "architecture_parameter_layout": {
            "layout_resolved": True,
            "parameter_dtype": "float32",
            "model_dim": GEOMETRY["model_dim"],
            "hidden_dim": GEOMETRY["hidden_dim"],
            "vocab_size": GEOMETRY["vocab_size"],
            "padded_vocab_size": GEOMETRY["padded_vocab_size"],
            "num_layers": GEOMETRY["num_layers"],
            "parameter_buffer_count": len(legacy_buffers),
            "parameter_elements": len(values),
            "parameter_bytes": byte_offset,
            "contiguous_parameter_state": True,
            "buffers": legacy_buffers,
        },
        "writer_verification": {
            "passed": True,
            "parameter_sidecar_exists": True,
            "parameter_sidecar_size_matches": True,
        },
        "native_parameter_state": {
            "full_template_parameter_state": True,
            "parameter_buffer_count": len(legacy_buffers),
            "parameter_elements": len(values),
            "persisted_parameter_elements": len(values),
            "trained_parameter_elements": len(values),
            "parameter_data_path": str(sidecar),
            "architecture_forward_inference_supported": True,
        },
        "inference_contract": {
            "schema": NATIVE_FAMILY_LLAMA_INFERENCE_SCHEMA,
            "version": 2,
            "family": "llama",
            "preset": "llama",
            "checkpoint_kind": "live_full_architecture",
            "done_marker": done.name,
            "geometry": GEOMETRY,
            "semantics": SEMANTICS,
            "training": {
                "train_seq_len": GEOMETRY["max_seq_len"],
                "steps_completed": 2,
                "train_batches_sampled": 2,
                "validation_batches_sampled": 1,
            },
            "artifact": {
                "format": NATIVE_FAMILY_LLAMA_CHECKPOINT_FORMAT,
                "path": sidecar.name,
                "dtype": "float32",
                "byte_order": "little",
                "layout": "contiguous_row_major",
                "nbytes": byte_offset,
                "sha256": digest,
            },
            "tensors": tensor_rows,
        },
    }
    metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    done.write_text("done\n", encoding="utf-8")
    return metadata, sidecar


def _build_llama_graph(preset: str = "llama") -> NeuronGraph:
    spec = build_model_spec_from_config(
        {
            "preset": preset,
            "num_layers": GEOMETRY["num_layers"],
            "model_dim": GEOMETRY["model_dim"],
            "num_heads": GEOMETRY["num_heads"],
            "num_kv_heads": GEOMETRY["num_kv_heads"],
            "multiple_of": 8,
            "vocab_size": GEOMETRY["vocab_size"],
        },
        preview_defaults=True,
    )
    assert spec.block_spec.mlp_multiplier == 8 / 3
    return build_gpt_root_graph(name="resident_llama_oracle", model_spec=spec)


def _write_llama_graph(path: Path, preset: str = "llama") -> None:
    path.write_text(
        json.dumps(_build_llama_graph(preset).to_dict(), indent=2),
        encoding="utf-8",
    )


def _read_llama_weights(checkpoint: Path) -> dict[str, list[float]]:
    raw = checkpoint.read_bytes()
    flat = list(struct.unpack(f"<{len(raw) // 4}f", raw))
    result: dict[str, list[float]] = {}
    offset = 0
    for name, shape in _llama_layout():
        count = _elements(shape)
        result[name] = flat[offset : offset + count]
        offset += count
    assert offset == len(flat)
    return result


def _python_llama_logits(checkpoint: Path, token_ids: Sequence[int]) -> list[float]:
    """Independent float32-boundary oracle for canonical RoPE/GQA LLaMA."""

    weights = _read_llama_weights(checkpoint)
    d = int(GEOMETRY["model_dim"])
    f = int(GEOMETRY["hidden_dim"])
    heads = int(GEOMETRY["num_heads"])
    kv_heads = int(GEOMETRY["num_kv_heads"])
    head_dim = int(GEOMETRY["head_dim"])
    kv_width = kv_heads * head_dim
    epsilon = float(GEOMETRY["rms_norm_eps"])
    theta = float(GEOMETRY["rope_theta"])

    def rms_norm(row: Sequence[float], weight: Sequence[float]) -> list[float]:
        inverse = 1.0 / math.sqrt(sum(value * value for value in row) / len(row) + epsilon)
        return [_f32(value * inverse * weight[index]) for index, value in enumerate(row)]

    def linear(row: Sequence[float], weight: Sequence[float], output_dim: int) -> list[float]:
        input_dim = len(row)
        return [
            _f32(
                sum(
                    row[index] * weight[output * input_dim + index]
                    for index in range(input_dim)
                )
            )
            for output in range(output_dim)
        ]

    def rope(row: Sequence[float], row_heads: int, position: int) -> list[float]:
        result = list(row)
        half = head_dim // 2
        for head in range(row_heads):
            start = head * head_dim
            for dimension in range(half):
                angle = position / (theta ** (2 * dimension / head_dim))
                first = row[start + dimension]
                second = row[start + dimension + half]
                result[start + dimension] = _f32(
                    first * math.cos(angle) + second * math.sin(angle)
                )
                result[start + dimension + half] = _f32(
                    -first * math.sin(angle) + second * math.cos(angle)
                )
        return result

    keys = [[[] for _layer in range(int(GEOMETRY["num_layers"]))] for _ in token_ids]
    values = [[[] for _layer in range(int(GEOMETRY["num_layers"]))] for _ in token_ids]
    final_hidden: list[float] = []
    embedding = weights["token_embedding.weight"]
    scale = 1.0 / math.sqrt(head_dim)
    for position, token in enumerate(token_ids):
        hidden = list(embedding[token * d : (token + 1) * d])
        for layer in range(int(GEOMETRY["num_layers"])):
            prefix = f"layers.{layer}."
            normalized = rms_norm(hidden, weights[f"{prefix}attention_norm.weight"])
            query = rope(linear(normalized, weights[f"{prefix}q_proj.weight"], d), heads, position)
            key = rope(
                linear(normalized, weights[f"{prefix}k_proj.weight"], kv_width),
                kv_heads,
                position,
            )
            value = linear(normalized, weights[f"{prefix}v_proj.weight"], kv_width)
            keys[position][layer] = key
            values[position][layer] = value
            attention = [0.0] * d
            for query_head in range(heads):
                kv_head = query_head * kv_heads // heads
                query_start = query_head * head_dim
                kv_start = kv_head * head_dim
                scores = [
                    sum(
                        query[query_start + dimension]
                        * keys[key_position][layer][kv_start + dimension]
                        for dimension in range(head_dim)
                    )
                    * scale
                    for key_position in range(position + 1)
                ]
                maximum = max(scores)
                probabilities = [math.exp(score - maximum) for score in scores]
                denominator = sum(probabilities)
                for dimension in range(head_dim):
                    attention[query_start + dimension] = _f32(
                        sum(
                            probabilities[key_position]
                            / denominator
                            * values[key_position][layer][kv_start + dimension]
                            for key_position in range(position + 1)
                        )
                    )
            projected = linear(attention, weights[f"{prefix}attention_out.weight"], d)
            residual = [_f32(left + right) for left, right in zip(hidden, projected)]
            normalized = rms_norm(residual, weights[f"{prefix}ffn_norm.weight"])
            packed = weights[f"{prefix}ffn_gate_up.weight"]
            gate = linear(normalized, packed[: f * d], f)
            up = linear(normalized, packed[f * d :], f)
            activated = [
                _f32(_f32(source / (1.0 + math.exp(-source))) * up[index])
                for index, source in enumerate(gate)
            ]
            down = linear(activated, weights[f"{prefix}ffn_down.weight"], d)
            hidden = [_f32(left + right) for left, right in zip(residual, down)]
        final_hidden = rms_norm(hidden, weights["final_norm.weight"])
    head = weights["lm_head.weight"]
    return [
        _f32(sum(final_hidden[index] * head[token * d + index] for index in range(d)))
        for token in range(int(GEOMETRY["vocab_size"]))
    ]


def _compiled_llama_with_native_weights(
    checkpoint: Path,
    *,
    graph: NeuronGraph | None = None,
) -> CompiledTorchGraph:
    """Compile the migrated source graph and import the canonical native tensor ABI."""

    compiled = CompiledTorchGraph(graph or _build_llama_graph(), kernel_backend="torch")
    weights = _read_llama_weights(checkpoint)
    d = int(GEOMETRY["model_dim"])
    f = int(GEOMETRY["hidden_dim"])
    vocab = int(GEOMETRY["vocab_size"])

    def copy_parameter(parameter: torch.nn.Parameter, values: Sequence[float]) -> None:
        source = torch.tensor(values, dtype=parameter.dtype).reshape(parameter.shape)
        parameter.copy_(source)

    model = compiled.node_modules["model"]
    with torch.no_grad():
        token_embedding = model.node_modules["token_embed"].embedding.weight
        copy_parameter(
            token_embedding,
            weights["token_embedding.weight"][: vocab * d],
        )
        copy_parameter(model.node_modules["final_norm"].weight, weights["final_norm.weight"])
        copy_parameter(
            model.node_modules["lm_head"].proj.weight,
            weights["lm_head.weight"][: vocab * d],
        )

        for layer in range(int(GEOMETRY["num_layers"])):
            prefix = f"layers.{layer}."
            block = model.node_modules[f"block_{layer}"]
            attention = block.node_modules["attention"]
            mlp = block.node_modules["mlp"].node_modules["swiglu"]

            copy_parameter(
                block.node_modules["attn_norm"].weight,
                weights[f"{prefix}attention_norm.weight"],
            )
            copy_parameter(
                attention.node_modules["q_proj"].proj.weight,
                weights[f"{prefix}q_proj.weight"],
            )
            copy_parameter(
                attention.node_modules["k_proj"].proj.weight,
                weights[f"{prefix}k_proj.weight"],
            )
            copy_parameter(
                attention.node_modules["v_proj"].proj.weight,
                weights[f"{prefix}v_proj.weight"],
            )
            copy_parameter(
                attention.node_modules["out_proj"].proj.weight,
                weights[f"{prefix}attention_out.weight"],
            )
            copy_parameter(
                block.node_modules["mlp_norm"].weight,
                weights[f"{prefix}ffn_norm.weight"],
            )
            packed_gate_up = weights[f"{prefix}ffn_gate_up.weight"]
            copy_parameter(mlp.w1.weight, packed_gate_up[: f * d])
            copy_parameter(mlp.w3.weight, packed_gate_up[f * d :])
            copy_parameter(mlp.w2.weight, weights[f"{prefix}ffn_down.weight"])

    compiled.eval()
    return compiled


@pytest.fixture(scope="session")
def resident_binding(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    output = tmp_path_factory.mktemp("native-resident-llama-binding") / (
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


@pytest.fixture()
def llama_artifact(tmp_path: Path) -> Path:
    metadata, _sidecar = _write_llama_checkpoint(tmp_path)
    graph_path = tmp_path / "llama.json"
    _write_llama_graph(graph_path)
    artifact = tmp_path / "artifact"
    migration = migrate_graph_to_native(
        graph_path,
        weights_path=metadata,
        output_dir=artifact,
    )
    assert migration.manifest is not None
    assert migration.report.compatible is True
    assert migration.manifest.capabilities["native_inference"] is True
    assert migration.manifest.capabilities["resident_inference"] is True
    assert migration.manifest.capabilities["lossless_kv_cache"] is True
    assert migration.manifest.capabilities["turboquant_kv_cache"] is False
    assert migration.manifest.capabilities["serve"] is True
    assert migration.manifest.kernel_abi["resident_inference"] == {
        "version": 1,
        "status": "ready",
    }
    return artifact


def _generation_config() -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "seed": None,
        "stop_token_ids": [],
        "strict_model_compute": True,
    }


def test_llama_recompute_and_lossless_gqa_cache_match_independent_rope_oracle(
    resident_binding: ModuleType,
    llama_artifact: Path,
) -> None:
    manifest = json.loads((llama_artifact / "native-execution-manifest.json").read_text())
    checkpoint = llama_artifact / "model.f32"
    model = resident_binding.load_model(str(llama_artifact), manifest)
    off = resident_binding.create_session(
        model, {"seed": 19, "kv_cache": {"effective_mode": "off"}}
    )
    full = resident_binding.create_session(
        model, {"seed": 19, "kv_cache": {"effective_mode": "full"}}
    )
    try:
        prompt = [1, 7, 3, 9]
        for session in (off, full):
            resident_binding.prefill(model, session, prompt, 0)
        expected = _python_llama_logits(checkpoint, prompt)
        off_logits = resident_binding.current_logits(model, off)
        full_logits = resident_binding.current_logits(model, full)
        assert off_logits == pytest.approx(expected, abs=4.0e-7)
        assert full_logits == pytest.approx(expected, abs=4.0e-7)

        off_token = resident_binding.decode_one(model, off, _generation_config())
        full_token = resident_binding.decode_one(model, full, _generation_config())
        assert off_token == pytest.approx(full_token, abs=4.0e-7)
        next_prompt = [*prompt, int(off_token["token_id"])]
        assert resident_binding.current_logits(model, off) == pytest.approx(
            _python_llama_logits(checkpoint, next_prompt), abs=4.0e-7
        )
        assert resident_binding.current_logits(model, full) == pytest.approx(
            _python_llama_logits(checkpoint, next_prompt), abs=4.0e-7
        )

        for session in (off, full):
            resident_binding.truncate_session(model, session, 2)
            resident_binding.prefill(model, session, [5, 2], 2)
        replacement = [1, 7, 5, 2]
        assert resident_binding.current_logits(model, off) == pytest.approx(
            _python_llama_logits(checkpoint, replacement), abs=4.0e-7
        )
        assert resident_binding.current_logits(model, full) == pytest.approx(
            _python_llama_logits(checkpoint, replacement), abs=4.0e-7
        )

        stats = resident_binding.model_stats(model)
        assert stats["model_family"] == "llama"
        assert stats["weights_load_count"] == 1
        assert stats["num_heads"] == 4
        assert stats["num_kv_heads"] == 2
        assert stats["head_dim"] == 2
        assert stats["hidden_dim"] == 24
        assert stats["rope_theta"] == 10_000.0
        assert stats["rms_norm_eps"] == pytest.approx(1.0e-6)
        assert stats["turboquant_kv_cache"] is False
        cache_stats = resident_binding.session_stats(model, full)
        assert cache_stats["cached_tokens"] == 4
        assert cache_stats["cache_bytes"] == 4 * (2 * 2 * 4 + 8) * 4
        assert cache_stats["cache_capacity_bytes"] == 8 * (2 * 2 * 4 + 8) * 4
        assert cache_stats["recompute_full_prefix"] is False
    finally:
        resident_binding.close_session(model, off)
        resident_binding.close_session(model, full)
        resident_binding.close_model(model)


def test_llama_full_cache_prefix_fork_is_copy_on_write_and_tail_isolated(
    resident_binding: ModuleType,
    llama_artifact: Path,
) -> None:
    manifest = json.loads((llama_artifact / "native-execution-manifest.json").read_text())
    assert manifest["capabilities"]["session_prefix_cow"] is True
    assert manifest["kernel_abi"]["session_prefix_cow"] == {
        "version": 1,
        "status": "ready",
        "profile": "llama-full-cache-gqa-kv-final-hidden-v1",
        "operation": "fork_session",
    }
    checkpoint = llama_artifact / "model.f32"
    model = resident_binding.load_model(str(llama_artifact), manifest)
    parent = resident_binding.create_session(
        model, {"seed": 1, "kv_cache": {"effective_mode": "full"}}
    )
    off = resident_binding.create_session(
        model, {"seed": 2, "kv_cache": {"effective_mode": "off"}}
    )
    children: list[Any] = []
    try:
        prompt = [1, 7, 3, 9]
        resident_binding.prefill(model, parent, prompt, 0)
        resident_binding.prefill(model, off, prompt[:1], 0)
        with pytest.raises(RuntimeError, match="full-cache"):
            resident_binding.fork_session(
                model, off, {"token_count": 1, "seed": 3}
            )
        parent_logits = resident_binding.current_logits(model, parent)
        assert parent_logits == pytest.approx(
            _python_llama_logits(checkpoint, prompt), abs=4.0e-7
        )

        left = resident_binding.fork_session(
            model, parent, {"token_count": 4, "seed": 11}
        )
        right = resident_binding.fork_session(
            model, parent, {"token_count": 2, "seed": 22}
        )
        survivor = resident_binding.fork_session(
            model, parent, {"token_count": 4, "seed": 33}
        )
        logical_only = resident_binding.fork_session(
            model, parent, {"token_count": 3, "seed": 44}
        )
        children.extend((left, right, survivor, logical_only))

        parent_shared = resident_binding.session_stats(model, parent)
        assert parent_shared["prefix_cow_forks_created"] == 4
        assert parent_shared["prefix_cow_storage_use_count"] == 5
        assert parent_shared["prefix_cow_shared_cached_tokens"] == 4
        assert parent_shared["prefix_cow_shared_capacity_bytes"] == parent_shared[
            "cache_capacity_bytes"
        ]
        assert resident_binding.session_stats(model, right)[
            "prefix_cow_forked_from_tokens"
        ] == 2
        assert resident_binding.current_logits(model, right) == pytest.approx(
            _python_llama_logits(checkpoint, prompt[:2]), abs=4.0e-7
        )

        # Truncate/reset are logical operations. They neither detach nor alter
        # any peer's immutable cached rows.
        resident_binding.truncate_session(model, logical_only, 1)
        assert resident_binding.session_stats(model, logical_only)[
            "prefix_cow_detach_count"
        ] == 0
        resident_binding.reset_session(model, logical_only)
        logical_stats = resident_binding.session_stats(model, logical_only)
        assert logical_stats["token_count"] == 0
        assert logical_stats["cached_tokens"] == 0
        assert logical_stats["prefix_cow_shared_storage"] is True
        assert resident_binding.current_logits(model, parent) == pytest.approx(
            parent_logits, abs=4.0e-7
        )
        resident_binding.close_session(model, logical_only)
        children.remove(logical_only)

        # Two children diverge at different tail positions. Each writer
        # detaches the complete GQA K/V plus final-hidden allocation first.
        resident_binding.prefill(model, left, [5], 4)
        resident_binding.prefill(model, right, [6], 2)
        for child in (left, right):
            stats = resident_binding.session_stats(model, child)
            assert stats["prefix_cow_storage_use_count"] == 1
            assert stats["prefix_cow_detach_count"] == 1
            assert stats["prefix_cow_detached_capacity_bytes"] == stats[
                "cache_capacity_bytes"
            ]
        assert resident_binding.current_logits(model, left) == pytest.approx(
            _python_llama_logits(checkpoint, [*prompt, 5]), abs=4.0e-7
        )
        assert resident_binding.current_logits(model, right) == pytest.approx(
            _python_llama_logits(checkpoint, [1, 7, 6]), abs=4.0e-7
        )
        assert resident_binding.current_logits(model, parent) == pytest.approx(
            parent_logits, abs=4.0e-7
        )

        # Cancellation and reset remain local to the child.
        resident_binding.cancel_session(model, left)
        with pytest.raises(InterruptedError, match="cancelled"):
            resident_binding.current_logits(model, left)
        assert resident_binding.current_logits(model, right)
        resident_binding.reset_session(model, left)
        assert resident_binding.session_stats(model, left)["cancelled"] is False
        assert resident_binding.session_stats(model, right)["token_count"] == 3

        # The parent is also a writer. It detaches from its still-shared child
        # before decode, and the child remains a byte-identical prefix owner.
        result = resident_binding.decode_one(model, parent, _generation_config())
        extended = [*prompt, int(result["token_id"])]
        assert resident_binding.session_stats(model, parent)[
            "prefix_cow_detach_count"
        ] == 1
        assert resident_binding.current_logits(model, parent) == pytest.approx(
            _python_llama_logits(checkpoint, extended), abs=4.0e-7
        )
        assert resident_binding.current_logits(model, survivor) == pytest.approx(
            parent_logits, abs=4.0e-7
        )

        # Closing either owner releases only its reference. The surviving
        # session retains valid logits and can later rewrite a truncated tail.
        resident_binding.close_session(model, parent)
        parent = None
        assert resident_binding.session_stats(model, survivor)[
            "prefix_cow_storage_use_count"
        ] == 1
        assert resident_binding.current_logits(model, survivor) == pytest.approx(
            parent_logits, abs=4.0e-7
        )
        resident_binding.truncate_session(model, survivor, 2)
        resident_binding.prefill(model, survivor, [8], 2)
        assert resident_binding.current_logits(model, survivor) == pytest.approx(
            _python_llama_logits(checkpoint, [1, 7, 8]), abs=4.0e-7
        )
    finally:
        for child in children:
            resident_binding.close_session(model, child)
        if parent is not None:
            resident_binding.close_session(model, parent)
        resident_binding.close_session(model, off)
        resident_binding.close_model(model)


def test_llama_prefix_fork_rng_cancel_and_counters_are_session_local(
    resident_binding: ModuleType,
    llama_artifact: Path,
) -> None:
    manifest = json.loads((llama_artifact / "native-execution-manifest.json").read_text())
    model = resident_binding.load_model(str(llama_artifact), manifest)
    parent = resident_binding.create_session(
        model, {"seed": 1, "kv_cache": {"effective_mode": "full"}}
    )
    left = None
    right = None
    try:
        resident_binding.prefill(model, parent, [1, 7], 0)
        left = resident_binding.fork_session(
            model, parent, {"token_count": 2, "seed": 91}
        )
        right = resident_binding.fork_session(
            model, parent, {"token_count": 2, "seed": 91}
        )
        sampled = {
            "temperature": 1.7,
            "top_k": 0,
            "top_p": 1.0,
            "seed": None,
            "stop_token_ids": [],
            "strict_model_compute": False,
        }
        left_tokens = [
            resident_binding.decode_one(model, left, sampled)["token_id"]
            for _ in range(3)
        ]
        right_tokens = [
            resident_binding.decode_one(model, right, sampled)["token_id"]
            for _ in range(3)
        ]
        assert right_tokens == left_tokens
        assert resident_binding.session_stats(model, left)["decode_calls"] == 3
        assert resident_binding.session_stats(model, right)["decode_calls"] == 3
        assert resident_binding.session_stats(model, parent)["decode_calls"] == 0
        assert resident_binding.session_stats(model, parent)["token_count"] == 2

        resident_binding.cancel_session(model, left)
        with pytest.raises(InterruptedError, match="cancelled"):
            resident_binding.current_logits(model, left)
        assert resident_binding.current_logits(model, right)
        assert resident_binding.session_stats(model, parent)["cancelled"] is False
        assert resident_binding.session_stats(model, right)["cancelled"] is False
    finally:
        if left is not None:
            resident_binding.close_session(model, left)
        if right is not None:
            resident_binding.close_session(model, right)
        resident_binding.close_session(model, parent)
        resident_binding.close_model(model)


def test_llama_cancelled_prefix_writes_restore_cow_storage_and_telemetry(
    resident_binding: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancellation_geometry: dict[str, int] = {
        "max_seq_len": 8,
        "vocab_size": 32,
        "padded_vocab_size": 32,
        "num_layers": 4,
        "model_dim": 192,
        "hidden_dim": 512,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 12,
    }
    for field, value in cancellation_geometry.items():
        monkeypatch.setitem(GEOMETRY, field, value)

    metadata, _sidecar = _write_llama_checkpoint(tmp_path)
    graph_path = tmp_path / "llama.json"
    _write_llama_graph(graph_path)
    artifact = tmp_path / "artifact"
    migration = migrate_graph_to_native(
        graph_path,
        weights_path=metadata,
        output_dir=artifact,
    )
    assert migration.manifest is not None
    manifest = migration.manifest.to_dict()

    model = resident_binding.load_model(str(artifact), manifest)
    parent = resident_binding.create_session(
        model, {"seed": 1, "kv_cache": {"effective_mode": "full"}}
    )
    prefill_child = None
    decode_child = None
    try:
        prompt = [1, 2]
        resident_binding.prefill(model, parent, prompt, 0)
        parent_logits = resident_binding.current_logits(model, parent)
        prefill_child = resident_binding.fork_session(
            model, parent, {"token_count": len(prompt), "seed": 11}
        )
        decode_child = resident_binding.fork_session(
            model, parent, {"token_count": len(prompt), "seed": 22}
        )
        capacity_bytes = resident_binding.session_stats(model, parent)[
            "cache_capacity_bytes"
        ]

        def cancel_after_native_write_starts(
            operation: Callable[[], None],
            session: Any,
        ) -> None:
            baseline = resident_binding.model_stats(model)["forward_calls"]
            errors: list[BaseException] = []

            def run() -> None:
                try:
                    operation()
                except BaseException as exc:  # asserted by the parent thread
                    errors.append(exc)

            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            deadline = time.monotonic() + 5.0
            while (
                resident_binding.model_stats(model)["forward_calls"] == baseline
                and time.monotonic() < deadline
            ):
                pass
            assert resident_binding.model_stats(model)["forward_calls"] > baseline
            assert worker.is_alive(), "fixture completed before cancellation could be observed"
            resident_binding.cancel_session(model, session)
            worker.join(timeout=5.0)
            assert not worker.is_alive()
            assert len(errors) == 1
            assert isinstance(errors[0], InterruptedError)

        def cancelled_prefill() -> None:
            resident_binding.prefill(model, prefill_child, [5], len(prompt))

        def cancelled_decode() -> None:
            resident_binding.decode_one(model, decode_child, _generation_config())

        cancel_after_native_write_starts(cancelled_prefill, prefill_child)
        cancel_after_native_write_starts(cancelled_decode, decode_child)

        for child in (prefill_child, decode_child):
            stats = resident_binding.session_stats(model, child)
            assert stats["token_count"] == len(prompt)
            assert stats["cached_tokens"] == len(prompt)
            assert stats["prefix_cow_storage_use_count"] == 3
            assert stats["prefix_cow_shared_storage"] is True
            assert stats["prefix_cow_shared_cached_tokens"] == len(prompt)
            assert stats["prefix_cow_shared_capacity_bytes"] == capacity_bytes
            assert stats["prefix_cow_detach_count"] == 0
            assert stats["prefix_cow_detached_capacity_bytes"] == 0
        assert resident_binding.session_stats(model, prefill_child)["prefill_calls"] == 0
        assert resident_binding.session_stats(model, prefill_child)["prefill_tokens"] == 0
        assert resident_binding.session_stats(model, decode_child)["decode_calls"] == 0
        assert resident_binding.session_stats(model, decode_child)[
            "decode_rows_processed"
        ] == 0
        assert resident_binding.current_logits(model, parent) == pytest.approx(
            parent_logits, abs=4.0e-7
        )

        # Cancellation requires reset before reuse. Each successful retry then
        # performs exactly one real detach from the restored parent allocation.
        resident_binding.reset_session(model, prefill_child)
        resident_binding.prefill(model, prefill_child, [*prompt, 5], 0)
        prefill_retry = resident_binding.session_stats(model, prefill_child)
        assert prefill_retry["token_count"] == len(prompt) + 1
        assert prefill_retry["prefix_cow_detach_count"] == 1
        assert prefill_retry["prefix_cow_detached_capacity_bytes"] == capacity_bytes

        resident_binding.reset_session(model, decode_child)
        resident_binding.prefill(model, decode_child, prompt, 0)
        decoded = resident_binding.decode_one(model, decode_child, _generation_config())
        decode_retry = resident_binding.session_stats(model, decode_child)
        assert decode_retry["token_count"] == len(prompt) + 1
        assert decode_retry["decode_calls"] == 1
        assert decode_retry["prefix_cow_detach_count"] == 1
        assert decode_retry["prefix_cow_detached_capacity_bytes"] == capacity_bytes
        assert decoded["token_id"] >= 0
        assert resident_binding.current_logits(model, parent) == pytest.approx(
            parent_logits, abs=4.0e-7
        )
        parent_stats = resident_binding.session_stats(model, parent)
        assert parent_stats["prefix_cow_detach_count"] == 0
        assert parent_stats["prefix_cow_storage_use_count"] == 1
    finally:
        if prefill_child is not None:
            resident_binding.close_session(model, prefill_child)
        if decode_child is not None:
            resident_binding.close_session(model, decode_child)
        resident_binding.close_session(model, parent)
        resident_binding.close_model(model)


def test_migrated_canonical_llama_graph_matches_resident_logits_and_loss(
    resident_binding: ModuleType,
    llama_artifact: Path,
) -> None:
    """Prove the migrated source graph and resident ABI compute the same model."""

    manifest = json.loads((llama_artifact / "native-execution-manifest.json").read_text())
    checkpoint = llama_artifact / "model.f32"
    graph_path = llama_artifact.parent / "llama.json"
    graph_bytes = graph_path.read_bytes()
    assert hashlib.sha256(graph_bytes).hexdigest() == manifest["source_graph"]["sha256"]
    graph = NeuronGraph.from_dict(json.loads(graph_bytes))

    compiled = _compiled_llama_with_native_weights(checkpoint, graph=graph)
    prompt = [1, 7, 3, 9]
    targets = [7, 3, 9, 2]
    tokens_tensor = torch.tensor([prompt], dtype=torch.long)
    targets_tensor = torch.tensor([targets], dtype=torch.long)
    with torch.no_grad():
        (graph_loss,), trace = compiled.trace(tokens_tensor, targets_tensor)
    graph_logits = trace["model/lm_head"][0].squeeze(0)

    model = resident_binding.load_model(str(llama_artifact), manifest)
    session = resident_binding.create_session(
        model, {"seed": 31, "kv_cache": {"effective_mode": "off"}}
    )
    try:
        resident_rows: list[list[float]] = []
        for position, token_id in enumerate(prompt):
            resident_binding.prefill(model, session, [token_id], position)
            resident_rows.append(resident_binding.current_logits(model, session))
    finally:
        resident_binding.close_session(model, session)
        resident_binding.close_model(model)

    resident_logits = torch.tensor(resident_rows, dtype=torch.float32)
    oracle_logits = torch.tensor(
        [_python_llama_logits(checkpoint, prompt[: position + 1]) for position in range(len(prompt))],
        dtype=torch.float32,
    )
    torch.testing.assert_close(resident_logits, oracle_logits, rtol=0.0, atol=4.0e-7)
    torch.testing.assert_close(graph_logits, resident_logits, rtol=1.0e-5, atol=2.0e-6)

    resident_loss = F.cross_entropy(resident_logits, targets_tensor.reshape(-1))
    torch.testing.assert_close(graph_loss, resident_loss, rtol=1.0e-6, atol=1.0e-6)


def test_llama_fast_migration_preserves_compile_profile_and_loads_canonical_abi(
    resident_binding: ModuleType,
    tmp_path: Path,
) -> None:
    metadata, _sidecar = _write_llama_checkpoint(tmp_path)
    graph_path = tmp_path / "llama-fast.json"
    _write_llama_graph(graph_path, "llama_fast")
    graph_sha256 = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    checkpoint_payload = json.loads(metadata.read_text(encoding="utf-8"))
    checkpoint_payload["inference_contract"]["training"]["source_graph"] = {
        "filename": graph_path.name,
        "sha256": graph_sha256,
        "byte_identity_verified": True,
    }
    metadata.write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    artifact = tmp_path / "llama-fast-artifact"

    migration = migrate_graph_to_native(
        graph_path,
        weights_path=metadata,
        output_dir=artifact,
    )

    assert migration.report.compatible is True
    assert migration.manifest is not None
    manifest = migration.manifest.to_dict()
    assert manifest["model"]["template_spec"]["template"]["runtime"] == "compile"
    assert manifest["capabilities"]["resident_inference"] is True
    assert manifest["capabilities"]["lossless_kv_cache"] is True
    assert manifest["capabilities"]["turboquant_kv_cache"] is False
    assert manifest["checkpoint"]["preset"] == "llama"
    assert manifest["source_graph"]["sha256"] == graph_sha256
    assert manifest["checkpoint"]["source_graph"]["sha256"] == graph_sha256

    model = resident_binding.load_model(str(artifact), manifest)
    session = resident_binding.create_session(
        model,
        {"seed": 37, "kv_cache": {"effective_mode": "full"}},
    )
    try:
        resident_binding.prefill(model, session, [1, 7, 3], 0)
        assert resident_binding.current_logits(model, session) == pytest.approx(
            _python_llama_logits(artifact / "model.f32", [1, 7, 3]),
            abs=4.0e-7,
        )
    finally:
        resident_binding.close_session(model, session)
        resident_binding.close_model(model)


def test_llama_sessions_are_isolated_resettable_cancellable_and_turboquant_closed(
    resident_binding: ModuleType,
    llama_artifact: Path,
) -> None:
    manifest = json.loads((llama_artifact / "native-execution-manifest.json").read_text())
    checkpoint = llama_artifact / "model.f32"
    model = resident_binding.load_model(str(llama_artifact), manifest)
    first = resident_binding.create_session(
        model, {"seed": 23, "kv_cache": {"effective_mode": "full"}}
    )
    second = resident_binding.create_session(
        model, {"seed": 29, "kv_cache": {"effective_mode": "full"}}
    )
    try:
        resident_binding.prefill(model, first, [1, 2, 3], 0)
        resident_binding.prefill(model, second, [4, 5], 0)
        assert resident_binding.current_logits(model, first) == pytest.approx(
            _python_llama_logits(checkpoint, [1, 2, 3]), abs=4.0e-7
        )
        assert resident_binding.current_logits(model, second) == pytest.approx(
            _python_llama_logits(checkpoint, [4, 5]), abs=4.0e-7
        )

        resident_binding.cancel_session(model, first)
        with pytest.raises(InterruptedError, match="cancelled"):
            resident_binding.current_logits(model, first)
        assert resident_binding.session_stats(model, first)["token_count"] == 3
        resident_binding.reset_session(model, first)
        resident_binding.prefill(model, first, [6, 7], 0)
        assert resident_binding.current_logits(model, first) == pytest.approx(
            _python_llama_logits(checkpoint, [6, 7]), abs=4.0e-7
        )
        assert resident_binding.current_logits(model, second) == pytest.approx(
            _python_llama_logits(checkpoint, [4, 5]), abs=4.0e-7
        )

        open_sessions_before = resident_binding.model_stats(model)["open_sessions"]
        with pytest.raises(ValueError, match="canonical LLaMA.*TurboQuant GQA"):
            resident_binding.create_session(
                model,
                {"seed": 0, "kv_cache": {"effective_mode": "turboquant"}},
            )
        assert resident_binding.model_stats(model)["open_sessions"] == open_sessions_before
    finally:
        resident_binding.close_session(model, first)
        resident_binding.close_session(model, second)
        resident_binding.close_model(model)


def test_llama_sdk_loads_migrated_artifact_and_reuses_exact_prefix(
    resident_binding: ModuleType,
    llama_artifact: Path,
) -> None:
    with NativeInferenceModel.load(
        llama_artifact,
        binding=resident_binding,
        kv_cache=KVCacheConfig(mode="auto"),
    ) as model:
        assert model.capabilities.resident_inference is True
        assert model.capabilities.lossless_kv_cache is True
        assert model.capabilities.turboquant_kv_cache is False
        assert model.capabilities.session_prefix_cow is True
        with model.create_session(seed=31) as session:
            assert session.prefill([1, 2, 3])["prefilled_tokens"] == 3
            synchronized = session.prefill([1, 2, 5])
            assert synchronized["prefix_reused"] == 2
            assert synchronized["prefilled_tokens"] == 1
            with model.fork_session(session, token_count=2, seed=47) as child:
                assert child.token_ids == (1, 2)
                assert child.stats()["seed"] == 47
                child.prefill([1, 2, 6])
                assert child.current_logits() == pytest.approx(
                    _python_llama_logits(llama_artifact / "model.f32", [1, 2, 6]),
                    abs=4.0e-7,
                )
                assert session.token_ids == (1, 2, 5)
            result = session.decode(GenerationConfig(max_new_tokens=2, temperature=0.0))
            assert len(result.token_ids) == 2
            stats = session.stats()
            assert stats["effective_cache"] == "full"
            assert stats["lossy_cache"] is False
            assert stats["prefix_tokens_reused"] == 2
            assert stats["decode_rows_processed"] == 2

        with pytest.raises(Exception, match="TurboQuant"):
            model.create_session(kv_cache=KVCacheConfig(mode="turboquant"))


def test_llama_raw_token_cli_uses_resident_logits_without_transition_sampling(
    resident_binding: ModuleType,
    llama_artifact: Path,
) -> None:
    prompt = (1, 2, 7)
    expected_logits = _python_llama_logits(llama_artifact / "model.f32", prompt)
    expected_token = max(range(len(expected_logits)), key=expected_logits.__getitem__)
    stdout = io.StringIO()
    stderr = io.StringIO()

    result = run_native_artifact_cli(
        NativeArtifactCLIConfig(
            artifact=llama_artifact,
            prompt_token_ids=prompt,
            max_new_tokens=1,
            temperature=0.0,
            top_k=0,
            kv_cache=KVCacheConfig(mode="full"),
        ),
        interactive=False,
        binding=resident_binding,
        stdout=stdout,
        stderr=stderr,
    )

    assert result == 0
    assert stdout.getvalue() == f"{expected_token}\n"
    assert "rendering generated token IDs" in stderr.getvalue()


def test_llama_binding_rejects_same_size_tampering_and_topology_bypass(
    resident_binding: ModuleType,
    llama_artifact: Path,
) -> None:
    manifest = json.loads((llama_artifact / "native-execution-manifest.json").read_text())
    checkpoint = llama_artifact / "model.f32"
    original = checkpoint.read_bytes()
    tampered = bytearray(original)
    tampered[-1] ^= 0x01
    checkpoint.write_bytes(tampered)
    with pytest.raises(RuntimeError, match="SHA-256"):
        resident_binding.load_model(str(llama_artifact), manifest)
    checkpoint.write_bytes(original)

    wrong_hidden = json.loads(json.dumps(manifest))
    wrong_hidden["checkpoint"]["geometry"]["hidden_dim"] = 16
    with pytest.raises(RuntimeError, match="SwiGLU width"):
        resident_binding.load_model(str(llama_artifact), wrong_hidden)

    stale_tensors = json.loads(json.dumps(manifest))
    stale_tensors["tensors"][-1]["shape"][-1] -= 1
    with pytest.raises(RuntimeError, match="tensor shape"):
        resident_binding.load_model(str(llama_artifact), stale_tensors)

    bypassed = json.loads(json.dumps(manifest))
    attention = next(
        graph
        for graph in bypassed["topology"]["graphs"]
        if graph["path"].endswith("nodes/attention/subgraph")
    )
    edge = next(
        item
        for item in attention["edges"]
        if item["dst_node"].endswith("/nodes/rope") and item["dst_port"] == 0
    )
    edge["dst_node"] = next(
        node["path"] for node in attention["nodes"] if node["instance_id"] == "sdpa"
    )
    with pytest.raises(RuntimeError, match="registry proof"):
        resident_binding.load_model(str(llama_artifact), bypassed)


def test_closed_model_identity_cannot_close_an_unrelated_llama_session(
    resident_binding: ModuleType,
    llama_artifact: Path,
) -> None:
    manifest = json.loads((llama_artifact / "native-execution-manifest.json").read_text())
    first_model = resident_binding.load_model(str(llama_artifact), manifest)
    second_model = resident_binding.load_model(str(llama_artifact), manifest)
    session = resident_binding.create_session(
        first_model, {"seed": 0, "kv_cache": {"effective_mode": "full"}}
    )
    resident_binding.close_model(first_model)
    try:
        with pytest.raises(ValueError, match="does not belong"):
            resident_binding.close_session(second_model, session)
        resident_binding.close_session(first_model, session)
    finally:
        resident_binding.close_model(second_model)
