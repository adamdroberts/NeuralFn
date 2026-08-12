from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import struct

import pytest

from neuralfn.native_dense_checkpoint import NATIVE_DENSE_GPT_CHECKPOINT_FORMAT
from neuralfn.native_graph_train import plan_native_graph_training
from neuralfn.native_moa_checkpoint import (
    NATIVE_MOA_CANDIDATE_ACTIVATIONS,
    NATIVE_MOA_INFERENCE_SCHEMA,
    inspect_native_moa_checkpoint,
)
from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config


def _write_moa_graph(path: Path) -> dict:
    spec = build_model_spec_from_config(
        {
            "preset": "gpt2_moa",
            "num_layers": 1,
            "model_dim": 8,
            "num_heads": 2,
            "num_kv_heads": 2,
            "multiple_of": None,
            "vocab_size": 50257,
        },
        preview_defaults=True,
    )
    graph = build_gpt_root_graph(name="gpt2_moa_checkpoint_test", model_spec=spec)
    path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")
    plan = plan_native_graph_training(path)
    assert plan.manifest is not None
    return dict(plan.manifest.model)


def _dense_parameter_count(
    *, max_seq_len: int, padded_vocab_size: int, num_layers: int, channels: int
) -> int:
    shapes: list[tuple[int, ...]] = [
        (padded_vocab_size, channels),
        (max_seq_len, channels),
    ]
    for _ in range(num_layers):
        shapes.extend(
            (
                (channels,),
                (channels,),
                (3 * channels, channels),
                (3 * channels,),
                (channels, channels),
                (channels,),
                (channels,),
                (channels,),
                (4 * channels, channels),
                (4 * channels,),
                (channels, 4 * channels),
                (channels,),
            )
        )
    shapes.extend(((channels,), (channels,)))
    return sum(math.prod(shape) for shape in shapes)


def _write_dense_v5(path: Path) -> None:
    max_seq_len = 16
    vocab_size = 50257
    padded_vocab_size = 50304
    num_layers = 1
    num_heads = 2
    channels = 8
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
    parameter_count = _dense_parameter_count(
        max_seq_len=max_seq_len,
        padded_vocab_size=padded_vocab_size,
        num_layers=num_layers,
        channels=channels,
    )
    path.write_bytes(struct.pack("<256i", *header) + bytes(parameter_count * 2))


def _write_bundle(root: Path, *, activation: str = "relu2") -> tuple[Path, Path, dict, dict]:
    root.mkdir(parents=True, exist_ok=True)
    graph_path = root / "source-graph.json"
    model = _write_moa_graph(graph_path)
    checkpoint = root / "model_00000007.bin"
    _write_dense_v5(checkpoint)
    done = root / "DONE_00000007"
    done.write_bytes(b"")
    payload = {
        "schema": NATIVE_MOA_INFERENCE_SCHEMA,
        "version": 1,
        "preset": "gpt2_moa",
        "checkpoint_kind": "trained_dense_v5",
        "model": {
            "path": checkpoint.name,
            "format": NATIVE_DENSE_GPT_CHECKPOINT_FORMAT,
            "nbytes": checkpoint.stat().st_size,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "done_marker": done.name,
        "source_graph": {
            "filename": graph_path.name,
            "sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
            "byte_identity_verified": True,
        },
        "selection": {
            "activation": activation,
            "candidates": list(NATIVE_MOA_CANDIDATE_ACTIVATIONS),
            "interval": 50,
        },
        "geometry": {
            "max_seq_len": 16,
            "vocab_size": 50257,
            "padded_vocab_size": 50304,
            "num_layers": 1,
            "model_dim": 8,
            "num_heads": 2,
            "head_dim": 4,
            "mlp_hidden_dim": 32,
        },
    }
    metadata = root / "model_00000007.moa.json"
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    return metadata, graph_path, model, payload


def test_native_moa_checkpoint_validates_and_extends_dense_descriptor(tmp_path: Path) -> None:
    metadata, graph_path, model, _payload = _write_bundle(tmp_path)

    info = inspect_native_moa_checkpoint(
        metadata,
        source_graph_path=graph_path,
        model=model,
    )
    descriptor = info.checkpoint_descriptor()

    assert info.selected_activation == "relu2"
    assert info.candidate_activations == NATIVE_MOA_CANDIDATE_ACTIVATIONS
    assert info.interval == 50
    assert info.source_graph_sha256 == hashlib.sha256(graph_path.read_bytes()).hexdigest()
    assert info.metadata_sha256 == hashlib.sha256(metadata.read_bytes()).hexdigest()
    assert descriptor["format"] == NATIVE_DENSE_GPT_CHECKPOINT_FORMAT
    assert descriptor["artifact_path"] == "model.bin"
    assert descriptor["target_file"] == "model.bin"
    assert descriptor["target_nbytes"] == info.file_size
    assert descriptor["target_sha256"] == info.sha256
    assert descriptor["source_graph"] == {
        "filename": graph_path.name,
        "sha256": info.source_graph_sha256,
        "byte_identity_verified": True,
    }
    assert descriptor["moa"] == {
        "schema": NATIVE_MOA_INFERENCE_SCHEMA,
        "version": 1,
        "preset": "gpt2_moa",
        "selected_activation": "relu2",
        "candidate_activations": ["gelu", "relu", "silu", "relu2"],
        "interval": 50,
        "source_graph_sha256": info.source_graph_sha256,
        "metadata_artifact_path": "model.moa.json",
        "metadata_nbytes": metadata.stat().st_size,
        "metadata_sha256": info.metadata_sha256,
    }

    byte_identical_original = tmp_path / "original-gpt2-moa.json"
    byte_identical_original.write_bytes(graph_path.read_bytes())
    renamed = inspect_native_moa_checkpoint(
        metadata,
        source_graph_path=byte_identical_original,
        model=model,
    )
    assert renamed.source_graph_filename == "source-graph.json"


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("missing_done", "done_marker does not exist"),
        ("nonempty_done", "DONE marker must be empty"),
        ("path_name", "wrong model filename"),
        ("symlink_escape", "escapes its artifact root"),
        ("model_sha", "model SHA-256 does not match"),
        ("source_sha", "source graph SHA-256 does not match"),
        ("candidates", "candidate list is not canonical"),
        ("activation", "selected activation is not canonical"),
        ("interval", "Graph MoA interval does not match checkpoint metadata"),
        ("geometry", "head geometry does not reconstruct"),
    ),
)
def test_native_moa_checkpoint_rejects_tampered_contract(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    metadata, graph_path, model, payload = _write_bundle(tmp_path)
    if case == "missing_done":
        (tmp_path / "DONE_00000007").unlink()
    elif case == "nonempty_done":
        (tmp_path / "DONE_00000007").write_text("done", encoding="utf-8")
    elif case == "path_name":
        payload["model"]["path"] = "../model_00000007.bin"
    elif case == "symlink_escape":
        checkpoint = tmp_path / "model_00000007.bin"
        outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
        outside.write_bytes(checkpoint.read_bytes())
        checkpoint.unlink()
        checkpoint.symlink_to(outside)
    elif case == "model_sha":
        payload["model"]["sha256"] = "0" * 64
    elif case == "source_sha":
        payload["source_graph"]["sha256"] = "0" * 64
    elif case == "candidates":
        payload["selection"]["candidates"] = ["relu", "gelu", "silu", "relu2"]
    elif case == "activation":
        payload["selection"]["activation"] = "tanh"
    elif case == "interval":
        payload["selection"]["interval"] = 51
    elif case == "geometry":
        payload["geometry"]["head_dim"] = 5
    if case not in {"missing_done", "nonempty_done"}:
        metadata.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        inspect_native_moa_checkpoint(
            metadata,
            source_graph_path=graph_path,
            model=model,
        )


def test_native_moa_checkpoint_rejects_dense_layout_and_graph_semantic_tampering(
    tmp_path: Path,
) -> None:
    layout_root = tmp_path / "layout"
    metadata, graph_path, model, payload = _write_bundle(layout_root)
    checkpoint = layout_root / "model_00000007.bin"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"x")
    payload["model"]["nbytes"] = checkpoint.stat().st_size
    payload["model"]["sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="version-5 geometry"):
        inspect_native_moa_checkpoint(
            metadata,
            source_graph_path=graph_path,
            model=model,
        )

    semantics_root = tmp_path / "semantics"
    metadata, graph_path, model, payload = _write_bundle(semantics_root)
    mutated_model = copy.deepcopy(model)
    mutated_model["template_spec"]["block_spec"]["activation_mode"] = "single"
    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    graph_payload["torch_config"]["template_spec"]["block_spec"]["activation_mode"] = "single"
    graph_path.write_text(json.dumps(graph_payload), encoding="utf-8")
    payload["source_graph"]["sha256"] = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="block_spec.activation_mode='moa'"):
        inspect_native_moa_checkpoint(
            metadata,
            source_graph_path=graph_path,
            model=mutated_model,
        )


def test_native_moa_checkpoint_accepts_positive_graph_bound_nondefault_interval(
    tmp_path: Path,
) -> None:
    metadata, graph_path, model, payload = _write_bundle(tmp_path)
    model = copy.deepcopy(model)
    model["template_spec"]["block_spec"]["moa_interval"] = 17
    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    graph_payload["torch_config"]["template_spec"]["block_spec"]["moa_interval"] = 17
    graph_path.write_text(json.dumps(graph_payload), encoding="utf-8")
    payload["source_graph"]["sha256"] = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    payload["selection"]["interval"] = 17
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    info = inspect_native_moa_checkpoint(
        metadata,
        source_graph_path=graph_path,
        model=model,
    )

    assert info.interval == 17
    assert info.checkpoint_descriptor()["moa"]["interval"] == 17


def test_native_moa_checkpoint_rejects_boolean_graph_interval_and_duplicate_keys(
    tmp_path: Path,
) -> None:
    bool_root = tmp_path / "bool-interval"
    metadata, graph_path, model, payload = _write_bundle(bool_root)
    model = copy.deepcopy(model)
    model["template_spec"]["block_spec"]["moa_interval"] = True
    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    graph_payload["torch_config"]["template_spec"]["block_spec"][
        "moa_interval"
    ] = True
    graph_path.write_text(json.dumps(graph_payload), encoding="utf-8")
    payload["source_graph"]["sha256"] = hashlib.sha256(
        graph_path.read_bytes()
    ).hexdigest()
    payload["selection"]["interval"] = 1
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Graph MoA interval does not match"):
        inspect_native_moa_checkpoint(
            metadata,
            source_graph_path=graph_path,
            model=model,
        )

    duplicate_root = tmp_path / "duplicate-key"
    metadata, graph_path, model, payload = _write_bundle(duplicate_root)
    encoded = json.dumps(payload)
    metadata.write_text(
        '{"schema":"untrusted-duplicate",' + encoded[1:],
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not valid UTF-8 JSON"):
        inspect_native_moa_checkpoint(
            metadata,
            source_graph_path=graph_path,
            model=model,
        )


def test_gpt2_moa_training_plan_passes_bound_fingerprint_and_rejects_candidate_drift(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "gpt2-moa.json"
    _write_moa_graph(graph_path)
    plan = plan_native_graph_training(graph_path)
    fingerprint = hashlib.sha256(graph_path.read_bytes()).hexdigest()

    assert plan.execution_ready is True
    assert plan.trainer_arguments[-4:] == (
        "--moa-interval",
        "50",
        "--graph-fingerprint",
        fingerprint,
    )

    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    payload["torch_config"]["template_spec"]["block_spec"]["moa_interval"] = 17
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    nondefault_interval = plan_native_graph_training(graph_path)
    assert nondefault_interval.execution_ready is True
    assert nondefault_interval.trainer_arguments[-4:-2] == ("--moa-interval", "17")

    payload["torch_config"]["template_spec"]["block_spec"]["moa_activations"] = [
        "gelu",
        "relu",
        "relu2",
        "silu",
    ]
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    drifted = plan_native_graph_training(graph_path)
    assert drifted.execution_ready is False
    assert any(
        "block_spec.moa_activations" in issue.message
        for issue in drifted.training_issues
    )


def test_dense_trainer_source_emits_moa_metadata_before_done_and_verifies_graph_bytes() -> None:
    source = Path(
        "neuralfn/csrc/native_gpt2/nfn_gpt2_native_train.cpp"
    ).read_text(encoding="utf-8")
    writer = source.index("bool write_native_gpt_moa_inference_metadata(")
    checkpoint_writer = source.index("auto write_trained_checkpoint = [&]()")
    metadata_call = source.index(
        "write_native_gpt_moa_inference_metadata(", checkpoint_writer
    )
    done_write = source.index("ExclusiveOutputFile done(done_marker);", metadata_call)

    assert '#include "resident_sha256.h"' in source
    assert 'cfg->activation == "moa"' in source
    assert "!cfg->graph_file.empty() && graph_bound_moa" in source
    assert "sha256_file_hex(fs::path(cfg->graph_file), &actual" in source
    assert "actual != cfg->graph_fingerprint" in source
    assert "const bool moa_metadata_required" in source
    assert (
        "native_moa_enabled &&\n"
        "            resolved_native_template_name(cfg.template_name) == \"gpt2_moa\" &&\n"
        "            !cfg.graph_file.empty();"
        in source
    )
    assert writer < checkpoint_writer < metadata_call < done_write
    assert "neuralfn.native_dense_moa.inference_checkpoint" in source
    assert "trained_dense_v5" in source
    assert '"model_"' in source and '".moa.json"' in source


def test_dense_trainer_resume_restores_moa_selection_before_loading_weights() -> None:
    source = Path(
        "neuralfn/csrc/native_gpt2/nfn_gpt2_native_train.cpp"
    ).read_text(encoding="utf-8")
    reader = source.index("bool read_native_gpt_moa_resume_metadata(")
    resume_loader = source.index("auto load_resume_checkpoint = [&]()")
    reader_call = source.index("read_native_gpt_moa_resume_metadata(", resume_loader)
    tensor_load = source.index("struct ResumeTensor", reader_call)
    loaded = source.index("resume_checkpoint_loaded = true", reader_call)

    assert reader < resume_loader < reader_call < tensor_load < loaded
    assert "resume requires a verified source graph SHA-256" in source
    assert "model SHA-256 does not match the checkpoint bytes" in source
    assert "selection.candidates are not canonical" in source
    assert "selection.activation is not a canonical candidate" in source
    assert "selection.interval does not match --moa-interval" in source
    assert "DONE marker must be empty" in source
    assert "does not match the dense-v5 checkpoint" in source
    assert (
        'std::string active_moa_activation = resume_moa_selected_activation;'
        in source
    )
    assert (
        'std::string moa_last_selected_activation = resume_moa_selected_activation;'
        in source
    )
    assert (
        "const std::int64_t optimizer_step = resume_checkpoint_step + step;"
        in source
    )
    assert "forward_backward_update(optimizer_step" in source
    for diagnostic in (
        "resume_moa_metadata_path",
        "resume_moa_metadata_validated",
        "resume_moa_activation_restored",
        "resume_moa_selected_activation",
    ):
        assert f'\\"{diagnostic}\\"' in source
