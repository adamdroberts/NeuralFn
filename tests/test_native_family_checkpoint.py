from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from neuralfn.native_family_checkpoint import (
    NATIVE_FAMILY_LLAMA_CHECKPOINT_FORMAT,
    NATIVE_FAMILY_LLAMA_INFERENCE_SCHEMA,
    inspect_native_family_llama_checkpoint,
)
from neuralfn.native_ir import migrate_graph_to_native
from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config


GEOMETRY = {
    "max_seq_len": 8,
    "vocab_size": 7,
    "padded_vocab_size": 8,
    "num_layers": 2,
    "model_dim": 4,
    "hidden_dim": 16,
    "num_heads": 2,
    "num_kv_heads": 1,
    "head_dim": 2,
    "rope_theta": 10_000.0,
    "rope_scaling_factor": 1.0,
    "rms_norm_eps": 1.0e-6,
}

SEMANTICS = {
    "norm_type": "rmsnorm",
    "mlp_type": "swiglu",
    "pos_encoding": "rope",
    "attention_variant": "dense",
    "residual_type": "add",
    "linear_bias": False,
    "dropout_p": 0.0,
    "tie_embeddings": False,
}


def _layout() -> list[tuple[str, tuple[int, ...]]]:
    d = GEOMETRY["model_dim"]
    f = GEOMETRY["hidden_dim"]
    vp = GEOMETRY["padded_vocab_size"]
    kv = GEOMETRY["num_kv_heads"] * GEOMETRY["head_dim"]
    rows: list[tuple[str, tuple[int, ...]]] = [
        ("token_embedding.weight", (vp, d)),
        ("final_norm.weight", (d,)),
        ("lm_head.weight", (vp, d)),
    ]
    for layer in range(GEOMETRY["num_layers"]):
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


def _elements(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _write_checkpoint(root: Path) -> tuple[Path, Path, dict]:
    metadata = root / "llama_native_family_model_00000000.json"
    sidecar = root / "llama_native_family_parameters_00000000.f32"
    done = root / "llama_native_family_model_DONE"
    tensors = []
    legacy_buffers = []
    offset = 0
    for name, shape in _layout():
        nbytes = _elements(shape) * 4
        tensors.append(
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
        legacy_buffers.append(
            {
                "name": name,
                "offset": offset // 4,
                "byte_offset": offset,
                "elements": nbytes // 4,
                "bytes": nbytes,
                "trainable": True,
            }
        )
        offset += nbytes
    parameter_elements = offset // 4
    values = [((index * 13) % 97 - 48) / 100.0 for index in range(parameter_elements)]
    sidecar.write_bytes(struct.pack(f"<{parameter_elements}f", *values))
    sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    payload = {
        "format": "nfn-native-family-optimizer-checkpoint-v1",
        "model_family": "llama",
        "native_target": "nfn_llama_native_train",
        "template_name": "llama",
        "dataset_alias": "fixture",
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
            "parameter_elements": parameter_elements,
            "persisted_parameter_elements": parameter_elements,
            "bytes": offset,
            "storage": "live_family_device_parameter_store_float32_state",
            "trained_parameter_elements": parameter_elements,
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
            "parameter_elements": parameter_elements,
            "parameter_bytes": offset,
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
            "parameter_elements": parameter_elements,
            "persisted_parameter_elements": parameter_elements,
            "trained_parameter_elements": parameter_elements,
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
                "nbytes": offset,
                "sha256": sha256,
            },
            "tensors": tensors,
        },
    }
    metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    done.write_text("done\n", encoding="utf-8")
    return metadata, sidecar, payload


def _write_graph(path: Path, **overrides) -> None:
    config = {
        "preset": "llama",
        "num_layers": GEOMETRY["num_layers"],
        "model_dim": GEOMETRY["model_dim"],
        "num_heads": GEOMETRY["num_heads"],
        "num_kv_heads": GEOMETRY["num_kv_heads"],
        "multiple_of": 8,
        "vocab_size": GEOMETRY["vocab_size"],
    }
    config.update(overrides)
    spec = build_model_spec_from_config(config, preview_defaults=True)
    graph = build_gpt_root_graph(name="llama_checkpoint_fixture", model_spec=spec)
    path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")


def test_inspector_validates_geometry_contiguous_layout_and_hashes(tmp_path: Path) -> None:
    metadata, sidecar, _payload = _write_checkpoint(tmp_path)
    info = inspect_native_family_llama_checkpoint(metadata)

    assert info.artifact_path == sidecar.resolve()
    assert info.geometry == GEOMETRY
    assert info.semantics == SEMANTICS
    assert info.sha256 == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert info.tensors[0].name == "token_embedding.weight"
    assert info.tensors[0].shape == (8, 4)
    assert info.tensors[-1].name == "layers.1.ffn_down.weight"
    assert info.tensors[-1].offset + info.tensors[-1].nbytes == sidecar.stat().st_size
    descriptor = info.checkpoint_descriptor()
    assert descriptor["format"] == NATIVE_FAMILY_LLAMA_CHECKPOINT_FORMAT
    assert descriptor["artifact_path"] == "model.f32"
    assert set(descriptor["geometry"]) == set(GEOMETRY)
    assert descriptor["semantics"] == SEMANTICS


def test_inspector_preserves_and_validates_graph_byte_identity_provenance(tmp_path: Path) -> None:
    metadata, _sidecar, payload = _write_checkpoint(tmp_path)
    provenance = {
        "filename": "source-graph.json",
        "sha256": "a" * 64,
        "byte_identity_verified": True,
    }
    payload["inference_contract"]["training"]["source_graph"] = provenance
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    info = inspect_native_family_llama_checkpoint(metadata)

    assert info.training["source_graph"] == provenance
    assert info.checkpoint_descriptor()["source_graph"] == provenance

    for invalid in (
        {**provenance, "filename": "../source-graph.json"},
        {**provenance, "sha256": "A" * 64},
        {**provenance, "byte_identity_verified": False},
    ):
        payload["inference_contract"]["training"]["source_graph"] = invalid
        metadata.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="source.graph|source-graph"):
            inspect_native_family_llama_checkpoint(metadata)


def test_inspector_rejects_diagnostic_v1_without_v2_contract(tmp_path: Path) -> None:
    metadata, _sidecar, payload = _write_checkpoint(tmp_path)
    del payload["inference_contract"]
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="inference_contract"):
        inspect_native_family_llama_checkpoint(metadata)


def test_inspector_rejects_unsafe_paths_missing_done_layout_drift_and_tampering(
    tmp_path: Path,
) -> None:
    metadata, sidecar, payload = _write_checkpoint(tmp_path)
    payload["inference_contract"]["artifact"]["path"] = str(sidecar.resolve())
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="safe relative path"):
        inspect_native_family_llama_checkpoint(metadata)

    metadata, sidecar, payload = _write_checkpoint(tmp_path)
    metadata.with_name("llama_native_family_model_DONE").unlink()
    with pytest.raises(ValueError, match="does not exist"):
        inspect_native_family_llama_checkpoint(metadata)

    metadata, sidecar, payload = _write_checkpoint(tmp_path)
    payload["inference_contract"]["tensors"][1]["offset"] += 4
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly contiguous"):
        inspect_native_family_llama_checkpoint(metadata)

    metadata, sidecar, _payload = _write_checkpoint(tmp_path)
    raw = bytearray(sidecar.read_bytes())
    raw[-1] ^= 0xFF
    sidecar.write_bytes(raw)
    with pytest.raises(ValueError, match="SHA-256"):
        inspect_native_family_llama_checkpoint(metadata)


def test_inspector_validates_canonical_llama_graph_identity_and_geometry(tmp_path: Path) -> None:
    metadata, _sidecar, _payload = _write_checkpoint(tmp_path)
    info = inspect_native_family_llama_checkpoint(metadata)
    graph_path = tmp_path / "graph.json"
    _write_graph(graph_path)
    dry_run = migrate_graph_to_native(
        graph_path,
        weights_path=metadata,
        output_dir=tmp_path / "dry-run-output",
        dry_run=True,
    )
    assert dry_run.manifest is not None
    info.validate_model(dry_run.manifest.model)

    model = json.loads(json.dumps(dry_run.manifest.model))
    model["template_spec"]["block_spec"]["num_kv_heads"] = 2
    with pytest.raises(ValueError, match="num_kv_heads"):
        info.validate_model(model)


def test_native_ir_migrates_metadata_and_copies_raw_sidecar_as_model_f32(tmp_path: Path) -> None:
    metadata, sidecar, _payload = _write_checkpoint(tmp_path)
    graph_path = tmp_path / "graph.json"
    _write_graph(graph_path)
    output = tmp_path / "native-artifact"

    result = migrate_graph_to_native(
        graph_path,
        weights_path=metadata,
        output_dir=output,
    )

    assert result.manifest is not None
    assert result.manifest.checkpoint is not None
    assert result.manifest.checkpoint["format"] == NATIVE_FAMILY_LLAMA_CHECKPOINT_FORMAT
    assert result.manifest.checkpoint["artifact_path"] == "model.f32"
    assert result.manifest.checkpoint["geometry"] == GEOMETRY
    assert result.manifest.checkpoint["semantics"] == SEMANTICS
    assert result.manifest.context_limits["max_context_tokens"] == GEOMETRY["max_seq_len"]
    assert result.manifest.capabilities["native_inference"] is True
    assert result.manifest.capabilities["resident_inference"] is True
    assert result.manifest.capabilities["lossless_kv_cache"] is True
    assert result.manifest.capabilities["turboquant_kv_cache"] is False
    assert result.manifest.capabilities["serve"] is True
    assert result.manifest.kernel_abi["resident_inference"] == {
        "version": 1,
        "status": "ready",
    }
    assert (output / "model.f32").read_bytes() == sidecar.read_bytes()
    assert not (output / "model.bin").exists()
    assert len(result.manifest.tensors) == len(_layout())
    assert result.manifest.tensors[-1].offset + result.manifest.tensors[-1].nbytes == sidecar.stat().st_size


def test_native_ir_migration_binds_checkpoint_to_source_graph_fingerprint(
    tmp_path: Path,
) -> None:
    metadata, _sidecar, payload = _write_checkpoint(tmp_path)
    graph_path = tmp_path / "graph.json"
    _write_graph(graph_path)
    graph_sha256 = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    payload["inference_contract"]["training"]["source_graph"] = {
        "filename": "source-graph.json",
        "sha256": graph_sha256,
        "byte_identity_verified": True,
    }
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    result = migrate_graph_to_native(
        graph_path,
        weights_path=metadata,
        output_dir=tmp_path / "matching",
        dry_run=True,
    )
    assert result.manifest is not None
    assert result.manifest.checkpoint["source_graph"]["sha256"] == graph_sha256

    payload["inference_contract"]["training"]["source_graph"]["sha256"] = "b" * 64
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the graph supplied"):
        migrate_graph_to_native(
            graph_path,
            weights_path=metadata,
            output_dir=tmp_path / "mismatching",
            dry_run=True,
        )


def test_native_ir_rejects_direct_raw_f32_sidecar(tmp_path: Path) -> None:
    _metadata, sidecar, _payload = _write_checkpoint(tmp_path)
    graph_path = tmp_path / "graph.json"
    _write_graph(graph_path)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match=r"raw \.f32 sidecars are rejected"):
        migrate_graph_to_native(graph_path, weights_path=sidecar, output_dir=output)
    assert not output.exists()


def test_inspector_is_dependency_light(tmp_path: Path) -> None:
    metadata, _sidecar, _payload = _write_checkpoint(tmp_path)
    code = (
        "import json,sys; "
        "from neuralfn.native_family_checkpoint import inspect_native_family_llama_checkpoint; "
        f"info=inspect_native_family_llama_checkpoint({str(metadata)!r}); "
        "print(json.dumps({'sha256':info.sha256,'heavy':[n for n in ('torch','numpy','networkx') if n in sys.modules]}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert len(payload["sha256"]) == 64
    assert payload["heavy"] == []
