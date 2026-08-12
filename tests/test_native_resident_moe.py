from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import sysconfig
import threading
import time
from types import ModuleType
from typing import Any, Callable, Sequence

import pytest
import torch
import torch.nn.functional as F

from neuralfn.graph import NeuronGraph
from neuralfn.native_cli import NativeArtifactCLIConfig, run_native_artifact_cli
from neuralfn.native_inference import GenerationConfig, KVCacheConfig, NativeInferenceModel
from neuralfn.native_ir import migrate_graph_to_native
from neuralfn.native_moe_checkpoint import (
    NATIVE_FAMILY_STANDARD_MOE_CHECKPOINT_FORMAT,
    NATIVE_FAMILY_STANDARD_MOE_INFERENCE_SCHEMA,
    inspect_native_family_standard_moe_checkpoint,
)
from neuralfn.torch_backend import CompiledTorchGraph
from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY: dict[str, int | float] = {
    "max_seq_len": 8,
    "vocab_size": 11,
    "padded_vocab_size": 16,
    "num_layers": 2,
    "model_dim": 8,
    "hidden_dim": 21,
    "num_heads": 4,
    "num_kv_heads": 2,
    "head_dim": 2,
    "experts": 3,
    "top_k": 2,
    "rope_theta": 10_000.0,
    "rope_scaling_factor": 1.0,
    "rms_norm_eps": 1.0e-6,
    "mlp_multiplier": 8.0 / 3.0,
    "multiple_of": 0,
    "router_aux_loss_coef": 0.01,
}
SEMANTICS: dict[str, Any] = {
    "norm_type": "rmsnorm",
    "mlp_type": "moe",
    "pos_encoding": "rope",
    "attention_variant": "dense",
    "residual_type": "add",
    "router_score_fn": "softmax",
    "router_selection": "topk_renormalized",
    "moe_balance_mode": "aux_loss",
    "linear_bias": False,
    "dropout_p": 0.0,
    "tie_embeddings": False,
    "use_qk_norm": False,
    "shared_experts": 0,
}


def _elements(shape: Sequence[int]) -> int:
    return math.prod(shape)


def _moe_layout() -> list[tuple[str, tuple[int, ...]]]:
    d = int(GEOMETRY["model_dim"])
    h = int(GEOMETRY["hidden_dim"])
    e = int(GEOMETRY["experts"])
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
                (f"{prefix}router.weight", (e, d)),
                (f"{prefix}experts.gate_up.weight", (2, e, d, h)),
                (f"{prefix}experts.down.weight", (e, h, d)),
            )
        )
    return rows


def _fixture_value(name: str, global_index: int, local_index: int) -> float:
    if "norm.weight" in name:
        value = 0.91 + 0.07 * math.sin(0.13 * global_index + 0.17 * local_index)
    elif "router.weight" in name:
        value = 0.19 * math.sin(0.23 * global_index + 0.31 * local_index) + 0.011 * local_index
    else:
        scale = 0.13 if name in {"token_embedding.weight", "lm_head.weight"} else 0.068
        value = (
            scale * math.sin(0.29 * global_index + 0.071 * local_index)
            + 0.023 * math.cos(0.11 * global_index - 0.193 * local_index)
        )
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _write_moe_checkpoint(root: Path, *, preset: str = "mixllama") -> tuple[Path, Path]:
    metadata = root / "mixllama_native_family_model_00000000.json"
    sidecar = root / "mixllama_native_family_parameters_00000000.f32"
    done = root / "mixllama_native_family_model_DONE"
    values: list[float] = []
    tensor_rows: list[dict[str, Any]] = []
    legacy_rows: list[dict[str, Any]] = []
    offset = 0
    for name, shape in _moe_layout():
        count = _elements(shape)
        start = len(values)
        values.extend(_fixture_value(name, start + index, index) for index in range(count))
        nbytes = count * 4
        tensor_rows.append(
            {
                "name": name,
                "shape": list(shape),
                "offset": offset,
                "nbytes": nbytes,
                "dtype": "float32",
                "byte_order": "little",
                "layout": "row_major",
            }
        )
        legacy_rows.append(
            {
                "name": name,
                "offset": offset // 4,
                "byte_offset": offset,
                "elements": count,
                "bytes": nbytes,
                "trainable": True,
            }
        )
        offset += nbytes
    sidecar.write_bytes(struct.pack(f"<{len(values)}f", *values))
    digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    payload = {
        "format": "nfn-native-family-optimizer-checkpoint-v1",
        "model_family": "mixllama",
        "template_name": preset,
        "router_aux_loss_coef": GEOMETRY["router_aux_loss_coef"],
        "parameter_data": {
            "path": str(sidecar),
            "parameter_dtype": "float32",
            "parameter_elements": len(values),
            "bytes": offset,
        },
        "architecture_parameter_layout": {
            "parameter_buffer_count": len(legacy_rows),
            "parameter_elements": len(values),
            "parameter_bytes": offset,
            "buffers": legacy_rows,
        },
        "writer_verification": {
            "passed": True,
            "parameter_sidecar_exists": True,
            "parameter_sidecar_size_matches": True,
        },
        "native_parameter_state": {
            "full_template_parameter_state": True,
            "parameter_elements": len(values),
            "persisted_parameter_elements": len(values),
            "trained_parameter_elements": len(values),
            "parameter_data_path": str(sidecar),
            "architecture_forward_inference_supported": True,
        },
        "inference_contract": {
            "schema": NATIVE_FAMILY_STANDARD_MOE_INFERENCE_SCHEMA,
            "version": 1,
            "family": "mixllama",
            "preset": preset,
            "checkpoint_kind": "live_full_architecture",
            "done_marker": done.name,
            "geometry": GEOMETRY,
            "semantics": SEMANTICS,
            "training": {
                "train_seq_len": GEOMETRY["max_seq_len"],
                "steps_completed": 2,
                "source_graph": {
                    "filename": "mixllama.json",
                    "sha256": "a" * 64,
                    "byte_identity_verified": True,
                },
            },
            "artifact": {
                "format": NATIVE_FAMILY_STANDARD_MOE_CHECKPOINT_FORMAT,
                "path": sidecar.name,
                "dtype": "float32",
                "byte_order": "little",
                "layout": "contiguous_row_major",
                "nbytes": offset,
                "sha256": digest,
            },
            "tensors": tensor_rows,
        },
    }
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    done.write_text("done\n", encoding="utf-8")
    return metadata, sidecar


def _model_spec(*, runtime: str = "eager") -> dict[str, Any]:
    return {
        "model_dim": GEOMETRY["model_dim"],
        "num_layers": GEOMETRY["num_layers"],
        "vocab_size": GEOMETRY["vocab_size"],
        "tie_embeddings": False,
        "logit_softcap": 0.0,
        "block_spec": {
            "family": "mixllama",
            "norm_type": "rmsnorm",
            "mlp_type": "moe",
            "pos_encoding": "rope",
            "attention_backend": "sdpa",
            "num_heads": GEOMETRY["num_heads"],
            "num_kv_heads": GEOMETRY["num_kv_heads"],
            "is_causal": True,
            "linear_bias": False,
            "dropout_p": 0.0,
            "rope_theta": GEOMETRY["rope_theta"],
            "rope_scaling": None,
            "mlp_multiplier": GEOMETRY["mlp_multiplier"],
            "multiple_of": None,
            "experts": GEOMETRY["experts"],
            "top_k": GEOMETRY["top_k"],
            "shared_experts": 0,
            "router_aux_loss_coef": GEOMETRY["router_aux_loss_coef"],
            "compression": "none",
            "adapter_type": "none",
            "attention_variant": "dense",
            "use_qk_norm": False,
            "moe_balance_mode": "aux_loss",
            "router_score_fn": "softmax",
            "residual_type": "add",
            "activation_mode": "single",
        },
        "template": {
            "objective": "ar",
            "backbone": "mixllama",
            "sparsity": "moe",
            "router_mode": "none",
            "compression": "none",
            "adapter": "none",
            "runtime": runtime,
        },
    }


def _write_moe_graph(path: Path, preset: str = "mixllama") -> None:
    spec = build_model_spec_from_config(
        {
            "preset": preset,
            "num_layers": GEOMETRY["num_layers"],
            "model_dim": GEOMETRY["model_dim"],
            "num_heads": GEOMETRY["num_heads"],
            "num_kv_heads": GEOMETRY["num_kv_heads"],
            "mlp_multiplier": GEOMETRY["mlp_multiplier"],
            "multiple_of": None,
            "experts": GEOMETRY["experts"],
            "top_k": GEOMETRY["top_k"],
            "router_aux_loss_coef": GEOMETRY["router_aux_loss_coef"],
            "vocab_size": GEOMETRY["vocab_size"],
        },
        preview_defaults=True,
    )
    assert spec.block_spec.multiple_of is None
    assert int(spec.model_dim * spec.block_spec.mlp_multiplier) == GEOMETRY["hidden_dim"]
    path.write_text(
        json.dumps(
            build_gpt_root_graph(name="resident_moe_oracle", model_spec=spec).to_dict(),
            indent=2,
        ),
        encoding="utf-8",
    )


def _manifest(metadata: Path, artifact: Path) -> dict[str, Any]:
    info = inspect_native_family_standard_moe_checkpoint(metadata)
    checkpoint = info.checkpoint_descriptor()
    runtime = "compile" if info.preset == "mixllama-fast" else "eager"
    return {
        "schema": "neuralfn.native_execution_manifest",
        "version": 1,
        "capabilities": {
            "native_inference": True,
            "resident_inference": True,
            "lossless_kv_cache": True,
            "turboquant_kv_cache": False,
        },
        "kernel_abi": {"resident_inference": {"version": 1, "status": "ready"}},
        "model": {
            "family": "mixllama",
            "family_class": "autoregressive_transformer",
            "template_spec": _model_spec(runtime=runtime),
        },
        "checkpoint": checkpoint,
        "tensors": [tensor.to_dict() for tensor in info.tensors],
        "context_limits": {"max_context_tokens": GEOMETRY["max_seq_len"]},
    }


def _tensor_values(sidecar: Path) -> dict[str, torch.Tensor]:
    payload = sidecar.read_bytes()
    values = struct.unpack(f"<{len(payload) // 4}f", payload)
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for name, shape in _moe_layout():
        count = _elements(shape)
        result[name] = torch.tensor(values[offset : offset + count], dtype=torch.float32).reshape(shape)
        offset += count
    return result


def _compiled_moe_with_native_weights(graph_path: Path, sidecar: Path) -> CompiledTorchGraph:
    graph = NeuronGraph.from_dict(json.loads(graph_path.read_text(encoding="utf-8")))
    compiled = CompiledTorchGraph(graph, kernel_backend="torch")
    weights = _tensor_values(sidecar)
    vocab = int(GEOMETRY["vocab_size"])

    def copy_parameter(parameter: torch.nn.Parameter, source: torch.Tensor) -> None:
        parameter.copy_(source.to(dtype=parameter.dtype).reshape(parameter.shape))

    model = compiled.node_modules["model"]
    with torch.no_grad():
        copy_parameter(
            model.node_modules["token_embed"].embedding.weight,
            weights["token_embedding.weight"][:vocab],
        )
        copy_parameter(model.node_modules["final_norm"].weight, weights["final_norm.weight"])
        copy_parameter(
            model.node_modules["lm_head"].proj.weight,
            weights["lm_head.weight"][:vocab],
        )
        for layer in range(int(GEOMETRY["num_layers"])):
            prefix = f"layers.{layer}."
            block = model.node_modules[f"block_{layer}"]
            attention = block.node_modules["attention"]
            mlp = block.node_modules["mlp"]
            dispatch = mlp.node_modules["dispatch"]
            gate_up = weights[f"{prefix}experts.gate_up.weight"]
            copy_parameter(
                block.node_modules["attn_norm"].weight,
                weights[f"{prefix}attention_norm.weight"],
            )
            for native_name, graph_name in (
                ("q_proj", "q_proj"),
                ("k_proj", "k_proj"),
                ("v_proj", "v_proj"),
                ("attention_out", "out_proj"),
            ):
                copy_parameter(
                    attention.node_modules[graph_name].proj.weight,
                    weights[f"{prefix}{native_name}.weight"],
                )
            copy_parameter(
                block.node_modules["mlp_norm"].weight,
                weights[f"{prefix}ffn_norm.weight"],
            )
            copy_parameter(
                mlp.node_modules["router"].gate.weight,
                weights[f"{prefix}router.weight"],
            )
            copy_parameter(dispatch.w1, gate_up[0])
            copy_parameter(dispatch.w3, gate_up[1])
            copy_parameter(dispatch.w2, weights[f"{prefix}experts.down.weight"])
    compiled.eval()
    return compiled


def _rope(values: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    head_dim = values.shape[-1]
    half = head_dim // 2
    index = torch.arange(half, dtype=torch.float32)
    inverse = 1.0 / (float(GEOMETRY["rope_theta"]) ** (2.0 * index / head_dim))
    angles = position.to(torch.float32).unsqueeze(-1) * inverse
    cosine = torch.cos(angles).unsqueeze(1)
    sine = torch.sin(angles).unsqueeze(1)
    first = values[..., :half]
    second = values[..., half:]
    return torch.cat((first * cosine + second * sine, -first * sine + second * cosine), dim=-1)


def _rms_norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1.0e-6) * weight


def _torch_moe_logits(sidecar: Path, tokens: Sequence[int]) -> list[float]:
    weights = _tensor_values(sidecar)
    d = int(GEOMETRY["model_dim"])
    heads = int(GEOMETRY["num_heads"])
    kv_heads = int(GEOMETRY["num_kv_heads"])
    head_dim = int(GEOMETRY["head_dim"])
    hidden = weights["token_embedding.weight"][torch.tensor(tokens)]
    positions = torch.arange(len(tokens))
    causal = torch.triu(torch.ones(len(tokens), len(tokens), dtype=torch.bool), diagonal=1)
    for layer in range(int(GEOMETRY["num_layers"])):
        prefix = f"layers.{layer}."
        normalized = _rms_norm(hidden, weights[f"{prefix}attention_norm.weight"])
        query = F.linear(normalized, weights[f"{prefix}q_proj.weight"]).reshape(-1, heads, head_dim)
        key = F.linear(normalized, weights[f"{prefix}k_proj.weight"]).reshape(-1, kv_heads, head_dim)
        value = F.linear(normalized, weights[f"{prefix}v_proj.weight"]).reshape(-1, kv_heads, head_dim)
        query = _rope(query, positions)
        key = _rope(key, positions)
        key = key.repeat_interleave(heads // kv_heads, dim=1)
        value = value.repeat_interleave(heads // kv_heads, dim=1)
        scores = torch.einsum("thd,shd->hts", query, key) / math.sqrt(head_dim)
        scores = scores.masked_fill(causal.unsqueeze(0), float("-inf"))
        attention = torch.einsum("hts,shd->thd", torch.softmax(scores, dim=-1), value).reshape(-1, d)
        residual = hidden + F.linear(attention, weights[f"{prefix}attention_out.weight"])
        normalized = _rms_norm(residual, weights[f"{prefix}ffn_norm.weight"])
        router = F.linear(normalized, weights[f"{prefix}router.weight"])
        probabilities = torch.softmax(router, dim=-1)
        route_weights, route_indices = torch.topk(probabilities, int(GEOMETRY["top_k"]), dim=-1)
        route_weights = route_weights / route_weights.sum(dim=-1, keepdim=True)
        gate_up = weights[f"{prefix}experts.gate_up.weight"]
        expert_down = weights[f"{prefix}experts.down.weight"]
        routed = torch.zeros_like(normalized)
        for token in range(len(tokens)):
            for route in range(int(GEOMETRY["top_k"])):
                expert = int(route_indices[token, route])
                gate = normalized[token] @ gate_up[0, expert]
                up = normalized[token] @ gate_up[1, expert]
                routed[token] += (
                    (F.silu(gate) * up) @ expert_down[expert]
                ) * route_weights[token, route]
        hidden = residual + routed
    hidden = _rms_norm(hidden, weights["final_norm.weight"])
    logits = F.linear(hidden, weights["lm_head.weight"][: int(GEOMETRY["vocab_size"])])
    return logits[-1].tolist()


@pytest.fixture(scope="session")
def resident_binding(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    output = tmp_path_factory.mktemp("native-resident-moe-binding") / (
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
def moe_bundle(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    metadata, sidecar = _write_moe_checkpoint(tmp_path)
    graph_path = tmp_path / "mixllama.json"
    _write_moe_graph(graph_path)
    graph_sha256 = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    checkpoint_payload = json.loads(metadata.read_text(encoding="utf-8"))
    checkpoint_payload["inference_contract"]["training"]["source_graph"] = {
        "filename": graph_path.name,
        "sha256": graph_sha256,
        "byte_identity_verified": True,
    }
    metadata.write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    artifact = tmp_path / "artifact"
    migration = migrate_graph_to_native(
        graph_path,
        weights_path=metadata,
        output_dir=artifact,
    )
    assert migration.report.compatible is True
    assert migration.manifest is not None
    manifest = migration.manifest.to_dict()
    assert manifest["source_graph"]["sha256"] == graph_sha256
    assert manifest["checkpoint"]["source_graph"]["sha256"] == graph_sha256
    assert manifest["capabilities"]["resident_inference"] is True
    assert manifest["capabilities"]["lossless_kv_cache"] is True
    assert manifest["capabilities"]["turboquant_kv_cache"] is False
    assert manifest["capabilities"]["serve"] is True
    return metadata, sidecar, artifact, manifest


def _generation_config() -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "seed": None,
        "stop_token_ids": [],
        "strict_model_compute": True,
    }


def test_standard_moe_inspector_imports_when_torch_is_blocked() -> None:
    probe = """
import importlib.abc
import importlib.util
import pathlib
import sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('torch blocked')
        return None
sys.meta_path.insert(0, BlockTorch())
path = pathlib.Path('neuralfn/native_moe_checkpoint.py')
spec = importlib.util.spec_from_file_location('_native_moe_checkpoint_probe', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.NATIVE_FAMILY_STANDARD_MOE_CHECKPOINT_FORMAT.endswith('.f32.v1')
"""
    subprocess.run([sys.executable, "-c", probe], cwd=ROOT, check=True)


def test_standard_moe_inspector_proves_exact_layout_geometry_and_source_graph(tmp_path: Path) -> None:
    metadata, sidecar = _write_moe_checkpoint(tmp_path)
    info = inspect_native_family_standard_moe_checkpoint(metadata)
    assert info.path == sidecar.resolve()
    assert info.geometry == GEOMETRY
    assert info.training["source_graph"]["sha256"] == "a" * 64
    assert info.sha256 == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert len(info.tensors) == 3 + 9 * int(GEOMETRY["num_layers"])
    assert info.tensors[10].name == "layers.0.experts.gate_up.weight"
    assert info.tensors[10].shape == (2, 3, 8, 21)
    assert info.tensors[11].shape == (3, 21, 8)
    info.validate_model(
        {
            "family": "mixllama",
            "family_class": "autoregressive_transformer",
            "template_spec": _model_spec(),
        }
    )


@pytest.mark.parametrize(
    ("preset", "runtime"),
    (("mixllama", "eager"), ("moe", "eager"), ("mixllama_fast", "compile")),
)
def test_standard_moe_inspector_accepts_only_exact_alias_cluster_and_runtime(
    tmp_path: Path,
    preset: str,
    runtime: str,
) -> None:
    metadata, _sidecar = _write_moe_checkpoint(tmp_path, preset=preset)
    info = inspect_native_family_standard_moe_checkpoint(metadata)
    model = {
        "family": "mixllama",
        "family_class": "autoregressive_transformer",
        "template_spec": _model_spec(runtime=runtime),
    }
    info.validate_model(model)
    wrong_runtime = copy.deepcopy(model)
    wrong_runtime["template_spec"]["template"]["runtime"] = (
        "compile" if runtime == "eager" else "eager"
    )
    with pytest.raises(ValueError, match="runtime"):
        info.validate_model(wrong_runtime)


def test_standard_moe_inspector_rejects_incomplete_tampered_and_neighbor_profiles(tmp_path: Path) -> None:
    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("source.graph", lambda payload: payload["inference_contract"]["training"].pop("source_graph")),
        ("tensor", lambda payload: payload["inference_contract"]["tensors"].__setitem__(0, payload["inference_contract"]["tensors"][1])),
        ("hidden_dim", lambda payload: payload["inference_contract"]["geometry"].__setitem__("hidden_dim", 22)),
        ("semantics", lambda payload: payload["inference_contract"]["semantics"].__setitem__("router_score_fn", "sigmoid")),
        ("representable", lambda payload: payload["inference_contract"]["geometry"].__setitem__("router_aux_loss_coef", 1.0e300)),
        ("cluster", lambda payload: payload.__setitem__("template_name", "deepseek_v3")),
    ]
    for index, (label, mutate) in enumerate(mutations):
        root = tmp_path / str(index)
        root.mkdir()
        metadata, _ = _write_moe_checkpoint(root)
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        mutate(payload)
        metadata.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=label):
            inspect_native_family_standard_moe_checkpoint(metadata)

    missing_root = tmp_path / "missing-done"
    missing_root.mkdir()
    metadata, _ = _write_moe_checkpoint(missing_root)
    (missing_root / "mixllama_native_family_model_DONE").unlink()
    with pytest.raises(ValueError, match="does not exist"):
        inspect_native_family_standard_moe_checkpoint(metadata)

    checksum_root = tmp_path / "checksum"
    checksum_root.mkdir()
    metadata, sidecar = _write_moe_checkpoint(checksum_root)
    payload = bytearray(sidecar.read_bytes())
    payload[-1] ^= 1
    sidecar.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA-256"):
        inspect_native_family_standard_moe_checkpoint(metadata)


def test_standard_moe_resident_forward_matches_independent_torch_route_oracle(
    resident_binding: ModuleType,
    moe_bundle: tuple[Path, Path, Path, dict[str, Any]],
) -> None:
    _metadata, sidecar, artifact, manifest = moe_bundle
    model = resident_binding.load_model(str(artifact), manifest)
    session = resident_binding.create_session(
        model, {"seed": 19, "kv_cache": {"effective_mode": "off"}}
    )
    try:
        prompt = [1, 7, 3, 9]
        resident_binding.prefill(model, session, prompt, 0)
        assert resident_binding.current_logits(model, session) == pytest.approx(
            _torch_moe_logits(sidecar, prompt), abs=2.5e-6
        )
        stats = resident_binding.model_stats(model)
        assert stats["model_family"] == "mixllama"
        assert stats["experts"] == 3
        assert stats["top_k"] == 2
        assert stats["hidden_dim"] == 21
        assert stats["multiple_of"] == 0
        assert stats["mlp_multiplier"] == pytest.approx(8.0 / 3.0)
        assert stats["router_aux_loss_coef"] == pytest.approx(0.01)
        assert stats["turboquant_kv_cache"] is False
        open_sessions_before = stats["open_sessions"]
        with pytest.raises(ValueError, match="canonical standard-MoE.*TurboQuant GQA"):
            resident_binding.create_session(
                model,
                {"seed": 0, "kv_cache": {"effective_mode": "turboquant"}},
            )
        assert resident_binding.model_stats(model)["open_sessions"] == open_sessions_before
    finally:
        resident_binding.close_session(model, session)
        resident_binding.close_model(model)


def test_migrated_standard_moe_graph_matches_resident_logits_and_exact_aux_loss(
    resident_binding: ModuleType,
    moe_bundle: tuple[Path, Path, Path, dict[str, Any]],
) -> None:
    _metadata, sidecar, artifact, manifest = moe_bundle
    graph_path = artifact.parent / "mixllama.json"
    compiled = _compiled_moe_with_native_weights(graph_path, sidecar)
    prompt = [1, 7, 3, 9]
    targets = torch.tensor([[7, 3, 9, 2]], dtype=torch.long)
    (graph_loss,), trace = compiled.trace(
        torch.tensor([prompt], dtype=torch.long),
        targets,
    )
    graph_logits = trace["model/lm_head"][0].squeeze(0)

    model = resident_binding.load_model(str(artifact), manifest)
    session = resident_binding.create_session(
        model, {"seed": 41, "kv_cache": {"effective_mode": "off"}}
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
    torch.testing.assert_close(graph_logits, resident_logits, rtol=1.0e-5, atol=2.5e-6)

    raw_auxiliary = graph_loss.new_zeros(())
    for layer in range(int(GEOMETRY["num_layers"])):
        router_logits = trace[f"model/block_{layer}/mlp/router"][0]
        probabilities = torch.softmax(router_logits, dim=-1).reshape(
            -1, int(GEOMETRY["experts"])
        )
        density = probabilities.mean(dim=0)
        raw_auxiliary = raw_auxiliary + int(GEOMETRY["experts"]) * density.square().sum()
    expected_loss = F.cross_entropy(graph_logits, targets.reshape(-1)) + float(
        GEOMETRY["router_aux_loss_coef"]
    ) * raw_auxiliary
    torch.testing.assert_close(graph_loss, expected_loss, rtol=1.0e-6, atol=1.0e-6)

    graph_loss.backward()
    for layer in range(int(GEOMETRY["num_layers"])):
        router = compiled.node_modules["model"].node_modules[f"block_{layer}"].node_modules[
            "mlp"
        ].node_modules["router"].gate.weight
        assert router.grad is not None
        assert torch.isfinite(router.grad).all()
        assert torch.count_nonzero(router.grad).item() > 0


def test_standard_moe_full_and_off_cache_match_through_decode_truncate_and_reset(
    resident_binding: ModuleType,
    moe_bundle: tuple[Path, Path, Path, dict[str, Any]],
) -> None:
    _metadata, sidecar, artifact, manifest = moe_bundle
    model = resident_binding.load_model(str(artifact), manifest)
    off = resident_binding.create_session(model, {"seed": 7, "kv_cache": {"effective_mode": "off"}})
    full = resident_binding.create_session(model, {"seed": 7, "kv_cache": {"effective_mode": "full"}})
    try:
        prompt = [2, 5, 1]
        for session in (off, full):
            resident_binding.prefill(model, session, prompt, 0)
        assert resident_binding.current_logits(model, full) == pytest.approx(
            resident_binding.current_logits(model, off), abs=2.5e-6
        )
        off_result = resident_binding.decode_one(model, off, _generation_config())
        full_result = resident_binding.decode_one(model, full, _generation_config())
        assert off_result == pytest.approx(full_result, abs=2.5e-6)
        extended = [*prompt, int(off_result["token_id"])]
        assert resident_binding.current_logits(model, full) == pytest.approx(
            _torch_moe_logits(sidecar, extended), abs=2.5e-6
        )
        assert resident_binding.current_logits(model, off) == pytest.approx(
            resident_binding.current_logits(model, full), abs=2.5e-6
        )
        for session in (off, full):
            resident_binding.truncate_session(model, session, 2)
            resident_binding.prefill(model, session, [8], 2)
        assert resident_binding.current_logits(model, full) == pytest.approx(
            resident_binding.current_logits(model, off), abs=2.5e-6
        )
        for session in (off, full):
            resident_binding.reset_session(model, session)
            resident_binding.prefill(model, session, [4, 6], 0)
        assert resident_binding.current_logits(model, full) == pytest.approx(
            resident_binding.current_logits(model, off), abs=2.5e-6
        )
        full_stats = resident_binding.session_stats(model, full)
        off_stats = resident_binding.session_stats(model, off)
        assert full_stats["effective_cache"] == "full"
        assert full_stats["cached_tokens"] == 2
        assert full_stats["lossy_cache"] is False
        assert off_stats["effective_cache"] == "off"
        assert off_stats["recompute_full_prefix"] is True
    finally:
        resident_binding.close_session(model, off)
        resident_binding.close_session(model, full)
        resident_binding.close_model(model)


def test_standard_moe_prefix_fork_reuses_llama_cache_without_router_owner_leakage(
    resident_binding: ModuleType,
    moe_bundle: tuple[Path, Path, Path, dict[str, Any]],
) -> None:
    _metadata, sidecar, artifact, manifest = moe_bundle
    assert manifest["capabilities"]["session_prefix_cow"] is True
    assert manifest["kernel_abi"]["session_prefix_cow"] == {
        "version": 1,
        "status": "ready",
        "profile": "standard-moe-full-cache-gqa-kv-final-hidden-v1",
        "operation": "fork_session",
    }
    model = resident_binding.load_model(str(artifact), manifest)
    parent = resident_binding.create_session(
        model, {"seed": 5, "kv_cache": {"effective_mode": "full"}}
    )
    off = resident_binding.create_session(
        model, {"seed": 6, "kv_cache": {"effective_mode": "off"}}
    )
    children: list[Any] = []
    try:
        prompt = [2, 5, 1]
        resident_binding.prefill(model, parent, prompt, 0)
        resident_binding.prefill(model, off, prompt[:1], 0)
        with pytest.raises(RuntimeError, match="full-cache"):
            resident_binding.fork_session(
                model, off, {"token_count": 1, "seed": 7}
            )
        parent_logits = resident_binding.current_logits(model, parent)
        left = resident_binding.fork_session(
            model, parent, {"token_count": 3, "seed": 11}
        )
        right = resident_binding.fork_session(
            model, parent, {"token_count": 2, "seed": 22}
        )
        survivor = resident_binding.fork_session(
            model, parent, {"token_count": 3, "seed": 33}
        )
        children.extend((left, right, survivor))
        assert resident_binding.session_stats(model, parent)[
            "prefix_cow_storage_use_count"
        ] == 4

        resident_binding.prefill(model, left, [8], 3)
        resident_binding.prefill(model, right, [7], 2)
        assert resident_binding.current_logits(model, left) == pytest.approx(
            _torch_moe_logits(sidecar, [2, 5, 1, 8]), abs=2.5e-6
        )
        assert resident_binding.current_logits(model, right) == pytest.approx(
            _torch_moe_logits(sidecar, [2, 5, 7]), abs=2.5e-6
        )
        assert resident_binding.current_logits(model, parent) == pytest.approx(
            parent_logits, abs=2.5e-6
        )
        assert resident_binding.session_stats(model, left)[
            "prefix_cow_detach_count"
        ] == 1
        assert resident_binding.session_stats(model, right)[
            "prefix_cow_detach_count"
        ] == 1

        # MoE keeps the same immutable router/expert model owner while the
        # delegated LLaMA cache detaches for the parent write.
        result = resident_binding.decode_one(model, parent, _generation_config())
        assert resident_binding.session_stats(model, parent)[
            "prefix_cow_detach_count"
        ] == 1
        assert resident_binding.current_logits(model, parent) == pytest.approx(
            _torch_moe_logits(sidecar, [*prompt, int(result["token_id"])]),
            abs=2.5e-6,
        )
        assert resident_binding.current_logits(model, survivor) == pytest.approx(
            parent_logits, abs=2.5e-6
        )
        model_stats = resident_binding.model_stats(model)
        assert model_stats["model_family"] == "mixllama"
        assert model_stats["experts"] == GEOMETRY["experts"]
        assert model_stats["top_k"] == GEOMETRY["top_k"]

        resident_binding.close_session(model, parent)
        parent = None
        assert resident_binding.session_stats(model, survivor)[
            "prefix_cow_storage_use_count"
        ] == 1
        assert resident_binding.current_logits(model, survivor) == pytest.approx(
            parent_logits, abs=2.5e-6
        )
    finally:
        for child in children:
            resident_binding.close_session(model, child)
        if parent is not None:
            resident_binding.close_session(model, parent)
        resident_binding.close_session(model, off)
        resident_binding.close_model(model)


def test_standard_moe_cancelled_prefix_writes_restore_cow_storage_and_telemetry(
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
        "experts": 3,
        "top_k": 2,
    }
    for field, value in cancellation_geometry.items():
        monkeypatch.setitem(GEOMETRY, field, value)

    metadata, _sidecar = _write_moe_checkpoint(tmp_path)
    graph_path = tmp_path / "mixllama.json"
    _write_moe_graph(graph_path)
    graph_sha256 = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    checkpoint_payload = json.loads(metadata.read_text(encoding="utf-8"))
    checkpoint_payload["inference_contract"]["training"]["source_graph"] = {
        "filename": graph_path.name,
        "sha256": graph_sha256,
        "byte_identity_verified": True,
    }
    metadata.write_text(json.dumps(checkpoint_payload), encoding="utf-8")
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

        def cancelled_prefill() -> None:
            resident_binding.prefill(model, prefill_child, [5], len(prompt))

        def cancelled_decode() -> None:
            resident_binding.decode_one(model, decode_child, _generation_config())

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
            parent_logits, abs=2.5e-6
        )

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
            parent_logits, abs=2.5e-6
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


def test_standard_moe_and_bare_llama_sessions_cannot_cross_fork(
    resident_binding: ModuleType,
    moe_bundle: tuple[Path, Path, Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    from tests.test_native_resident_llama import (
        _write_llama_checkpoint,
        _write_llama_graph,
    )

    _metadata, _sidecar, moe_artifact, moe_manifest = moe_bundle
    llama_root = tmp_path / "llama-source"
    llama_root.mkdir()
    llama_metadata, _llama_sidecar = _write_llama_checkpoint(llama_root)
    llama_graph = llama_root / "llama.json"
    _write_llama_graph(llama_graph)
    llama_artifact = tmp_path / "llama-artifact"
    llama_migration = migrate_graph_to_native(
        llama_graph,
        weights_path=llama_metadata,
        output_dir=llama_artifact,
    )
    assert llama_migration.manifest is not None
    llama_manifest = llama_migration.manifest.to_dict()

    moe_model = resident_binding.load_model(str(moe_artifact), moe_manifest)
    llama_model = resident_binding.load_model(str(llama_artifact), llama_manifest)
    moe_session = resident_binding.create_session(
        moe_model, {"seed": 1, "kv_cache": {"effective_mode": "full"}}
    )
    llama_session = resident_binding.create_session(
        llama_model, {"seed": 2, "kv_cache": {"effective_mode": "full"}}
    )
    try:
        resident_binding.prefill(moe_model, moe_session, [1], 0)
        resident_binding.prefill(llama_model, llama_session, [1], 0)
        with pytest.raises(ValueError, match="does not belong"):
            resident_binding.fork_session(
                moe_model, llama_session, {"token_count": 1, "seed": 3}
            )
        with pytest.raises(ValueError, match="does not belong"):
            resident_binding.fork_session(
                llama_model, moe_session, {"token_count": 1, "seed": 4}
            )
    finally:
        resident_binding.close_session(moe_model, moe_session)
        resident_binding.close_session(llama_model, llama_session)
        resident_binding.close_model(moe_model)
        resident_binding.close_model(llama_model)


def test_migrated_standard_moe_sdk_uses_resident_inference_and_prefix_reuse(
    resident_binding: ModuleType,
    moe_bundle: tuple[Path, Path, Path, dict[str, Any]],
) -> None:
    _metadata, _sidecar, artifact, _manifest_payload = moe_bundle
    with NativeInferenceModel.load(
        artifact,
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
            with model.fork_session(session, token_count=2, seed=43) as child:
                child.prefill([1, 2, 7])
                assert child.token_ids == (1, 2, 7)
                assert session.token_ids == (1, 2, 5)
            result = session.decode(GenerationConfig(max_new_tokens=2, temperature=0.0))
            assert len(result.token_ids) == 2
            stats = session.stats()
            assert stats["effective_cache"] == "full"
            assert stats["lossy_cache"] is False
            assert stats["prefix_tokens_reused"] == 2

        with pytest.raises(Exception, match="TurboQuant"):
            model.create_session(kv_cache=KVCacheConfig(mode="turboquant"))


def test_migrated_standard_moe_raw_token_cli_uses_resident_logits(
    resident_binding: ModuleType,
    moe_bundle: tuple[Path, Path, Path, dict[str, Any]],
) -> None:
    _metadata, sidecar, artifact, _manifest_payload = moe_bundle
    prompt = (1, 2, 7)
    expected_logits = _torch_moe_logits(sidecar, prompt)
    expected_token = max(range(len(expected_logits)), key=expected_logits.__getitem__)
    stdout = io.StringIO()
    stderr = io.StringIO()

    result = run_native_artifact_cli(
        NativeArtifactCLIConfig(
            artifact=artifact,
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


def test_standard_moe_resident_session_lifecycle_and_ownership_are_isolated(
    resident_binding: ModuleType,
    moe_bundle: tuple[Path, Path, Path, dict[str, Any]],
) -> None:
    _metadata, _sidecar, artifact, manifest = moe_bundle
    first = resident_binding.load_model(str(artifact), manifest)
    second = resident_binding.load_model(str(artifact), manifest)
    session = resident_binding.create_session(first, {"seed": 1, "kv_cache": {"effective_mode": "full"}})
    try:
        with pytest.raises(ValueError, match="does not belong"):
            resident_binding.prefill(second, session, [1], 0)
        resident_binding.cancel_session(first, session)
        with pytest.raises(InterruptedError, match="cancelled"):
            resident_binding.prefill(first, session, [1], 0)
        resident_binding.reset_session(first, session)
        resident_binding.prefill(first, session, [1], 0)
        resident_binding.close_session(first, session)
        with pytest.raises(RuntimeError, match="closed"):
            resident_binding.current_logits(first, session)
    finally:
        resident_binding.close_session(first, session)
        resident_binding.close_model(first)
        resident_binding.close_model(second)


def test_standard_moe_binding_revalidates_manifest_and_rejects_closed_profiles(
    resident_binding: ModuleType,
    moe_bundle: tuple[Path, Path, Path, dict[str, Any]],
) -> None:
    _metadata, _sidecar, artifact, manifest = moe_bundle
    mutations = [
        lambda value: value["model"]["template_spec"]["block_spec"].__setitem__("moe_balance_mode", "auxfree"),
        lambda value: value["model"]["template_spec"]["block_spec"].__setitem__("shared_experts", 1),
        lambda value: value["model"]["template_spec"]["block_spec"].__setitem__("router_score_fn", "sigmoid"),
        lambda value: value["model"]["template_spec"]["template"].__setitem__("runtime", "megakernel"),
        lambda value: value["checkpoint"]["source_graph"].__setitem__("sha256", "bad"),
        lambda value: value["tensors"].reverse(),
    ]
    for mutate in mutations:
        invalid = copy.deepcopy(manifest)
        mutate(invalid)
        with pytest.raises(RuntimeError):
            resident_binding.load_model(str(artifact), invalid)
