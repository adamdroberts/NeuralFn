from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import struct
import tracemalloc

import pytest

from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config
from server.models import GraphModel, TrainRequest
from server.services import native_training


def _preset_payload(preset: str = "gpt2") -> dict:
    spec = build_model_spec_from_config(
        {
            "preset": preset,
            "model_dim": 32,
            "num_layers": 1,
            "num_heads": 4,
            "num_kv_heads": 4,
            "multiple_of": (
                None
                if preset in {"moe", "mixllama", "mixllama_fast"}
                else 16
            ),
            "vocab_size": 50257,
        },
        preview_defaults=True,
    )
    graph = build_gpt_root_graph(name=f"{preset}_editor", model_spec=spec)
    graph.runtime = "native-cuda"
    return graph.to_dict()


def _write_gpt2_diff_bundle(
    checkpoint_dir: Path,
    *,
    source_graph_sha256: str,
    proof_contract_sha256: str,
    geometry: dict[str, int],
    step: int = 1,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    step_text = f"{step:08d}"
    max_seq_len = geometry["max_seq_len"]
    vocab_size = geometry["vocab_size"]
    padded_vocab_size = geometry["padded_vocab_size"]
    num_layers = geometry["num_layers"]
    model_dim = geometry["model_dim"]
    num_heads = geometry["num_heads"]
    tensor_elements = [padded_vocab_size * model_dim, max_seq_len * model_dim]
    for _layer in range(num_layers):
        tensor_elements.extend(
            (
                model_dim,
                model_dim,
                3 * model_dim * model_dim,
                3 * model_dim,
                model_dim * model_dim,
                model_dim,
                model_dim,
                model_dim,
                4 * model_dim * model_dim,
                4 * model_dim,
                4 * model_dim * model_dim,
                model_dim,
            )
        )
    tensor_elements.extend((model_dim, model_dim))
    tensor_count = 4 + 12 * num_layers
    parameter_count = sum(tensor_elements)
    grad_accum_steps = 1
    consumed_microbatches = step * grad_accum_steps

    model_header = [0] * 256
    model_header[:8] = [
        20240326,
        5,
        max_seq_len,
        vocab_size,
        num_layers,
        num_heads,
        model_dim,
        padded_vocab_size,
    ]
    model = struct.pack("<256i", *model_header) + b"\0" * (parameter_count * 2)

    def dense_state(magic: int, payload_elements: int) -> bytes:
        header = [0] * 32
        header[:13] = [
            magic,
            1,
            step,
            max_seq_len,
            vocab_size,
            num_layers,
            num_heads,
            model_dim,
            padded_vocab_size,
            tensor_count,
            payload_elements,
            grad_accum_steps,
            consumed_microbatches,
        ]
        return struct.pack("<32q", *header) + b"\0" * (payload_elements * 4)

    def diff_state(magic: int, count: int) -> bytes:
        header = [0] * 16
        header[:9] = [
            magic,
            1,
            step,
            num_layers,
            num_layers,
            count,
            max_seq_len,
            num_heads,
            model_dim,
        ]
        values = ([0.8] * num_layers) if count == 1 else ([0.0] * (2 * num_layers))
        return struct.pack("<16q", *header) + struct.pack(f"<{len(values)}f", *values)

    artifact_payloads = {
        "model": model,
        "dense_parameters": dense_state(20260711, parameter_count),
        "dense_optimizer": dense_state(20260710, parameter_count * 2),
        "diff_parameters": diff_state(20260808, 1),
        "diff_optimizer": diff_state(20260809, 2),
    }
    artifact_contracts = {
        "model": (f"model_{step_text}.bin", "neuralfn.native_dense_gpt.v5"),
        "dense_parameters": (
            f"parameters_{step_text}.bin",
            "neuralfn.native_dense_gpt.parameters.v1",
        ),
        "dense_optimizer": (
            f"optimizer_{step_text}.bin",
            "neuralfn.native_dense_gpt.optimizer.v1",
        ),
        "diff_parameters": (
            f"diff_parameters_{step_text}.bin",
            "neuralfn.native_gpt2_diff.parameters.v1",
        ),
        "diff_optimizer": (
            f"diff_optimizer_{step_text}.bin",
            "neuralfn.native_gpt2_diff.optimizer.v1",
        ),
    }
    artifacts: dict[str, dict[str, object]] = {}
    for key, payload in artifact_payloads.items():
        filename, artifact_format = artifact_contracts[key]
        (checkpoint_dir / filename).write_bytes(payload)
        artifacts[key] = {
            "path": filename,
            "format": artifact_format,
            "nbytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    done_name = f"DONE_{step_text}"
    (checkpoint_dir / done_name).write_bytes(b"")
    metadata_geometry = {
        key: geometry[key]
        for key in (
            "max_seq_len",
            "vocab_size",
            "padded_vocab_size",
            "num_layers",
            "model_dim",
            "num_heads",
            "head_dim",
            "mlp_hidden_dim",
        )
    }
    microbatch_tokens = max_seq_len
    continuation = {
        "optimizer_steps_completed": step,
        "train_microbatches_completed": consumed_microbatches,
        "microbatch_in_optimizer_step": 0,
        "batch_size": 1,
        "seq_len": max_seq_len,
        "microbatch_tokens": microbatch_tokens,
        "requested_train_batch_tokens": microbatch_tokens,
        "effective_train_batch_tokens": microbatch_tokens * grad_accum_steps,
        "grad_accum_steps": grad_accum_steps,
        "train_seed_explicit": False,
        "train_seed": 0,
        "sampler_start_batch": 0,
        "sampler_total_batches": 1,
        "train_shard_count": 1,
        "train_shard_total_bytes": 2,
        "train_shards_sha256": "1" * 64,
        "learning_rate": 0.001,
        "lr_schedule": "constant",
        "lr_schedule_total_steps": step,
        "warmup_steps": 0,
        "final_lr_fraction": 1.0,
        "weight_decay": 0.0,
        "beta1": 0.9,
        "beta2": 0.95,
        "adam_eps": 1.0e-8,
        "grad_clip_norm": 1.0,
        "lm_head_row_chunk_size": 256,
        "bf16_block_weight_params": True,
        "bf16_block_dweight_staging": False,
        "dweight_first_microbatch_beta_zero": True,
        "numerics_profile_sha256": "2" * 64,
    }
    metadata = {
        "schema": "neuralfn.native_gpt2_diff.training_checkpoint",
        "version": 2,
        "preset": "gpt2_diff",
        "checkpoint_kind": "trained_dense_v5_plus_diff_v1",
        "step": step,
        **artifacts,
        "done_marker": done_name,
        "source_graph": {
            "filename": "source-graph.json",
            "sha256": source_graph_sha256,
            "byte_identity_verified": True,
        },
        "lambda": {
            "count": num_layers,
            "dtype": "float32",
            "initial_value": 0.8,
            "output_scale": 0.2,
        },
        "geometry": metadata_geometry,
        "graph_preflight_proof": {
            "schema": "neuralfn.native_graph_training_proof",
            "version": 1,
            "contract_sha256": proof_contract_sha256,
        },
        "continuation": continuation,
    }
    metadata_path = checkpoint_dir / f"model_{step_text}.diff.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata_path


def test_runtime_models_accept_native_cuda_and_reject_unknown_values() -> None:
    assert GraphModel(runtime="native-cuda").runtime == "native-cuda"
    assert TrainRequest(runtime="native-cuda").runtime == "native-cuda"
    with pytest.raises(ValueError):
        GraphModel(runtime="native-ish")
    with pytest.raises(ValueError):
        TrainRequest(runtime="native-ish")


def test_editor_preflight_returns_node_specific_failures_without_artifacts() -> None:
    payload = _preset_payload()
    token_embed = payload["nodes"]["model"]["neuron_def"]["subgraph"]["nodes"]["token_embed"]
    token_embed["neuron_def"]["module_type"] = "unregistered_future_op"

    result = native_training.preflight_native_training(payload)

    assert result["runtime"] == "native-cuda"
    assert result["execution_ready"] is False
    issue = result["issues"][0]
    assert issue["code"] == "unsupported_module"
    assert issue["operation"] == "unregistered_future_op"
    assert issue["path"].endswith("/nodes/token_embed")
    assert result["artifact_metadata"]["materialized"] is False
    assert result["artifact_metadata"]["manifest_path"] is None


def test_supported_editor_preflight_uses_reviewed_cli_adapter() -> None:
    result = native_training.preflight_native_training(_preset_payload("gpt2"))

    assert result["execution_ready"] is True
    assert result["trainer_family"] == "gpt2"
    assert result["training_selector"] == "gpt2"
    assert result["native_target"] == "nfn_gpt_native_train"
    assert result["graph_preflight_enforced"] is True
    assert result["trainer_consumes_native_ir"] is False
    assert result["issues"] == []


def test_structurally_lowerable_but_unreviewed_adapter_fails_closed() -> None:
    result = native_training.preflight_native_training(_preset_payload("nanogpt"))

    assert result["compatibility_report"]["compatible"] is True
    assert result["training_selector"] == "nanogpt"
    assert result["execution_ready"] is False
    assert result["compatible"] is False
    issue = result["issues"][0]
    assert issue["code"] == "unsupported_training_adapter"
    assert issue["path"].endswith("/nodes/embed_dropout")


def test_prepare_run_materializes_native_ir_and_canonical_native_command(tmp_path: Path) -> None:
    prepared = native_training.prepare_native_training_run(
        _preset_payload(),
        run_id="run-one",
        artifacts_dir=tmp_path,
        dataset_names=["tiny_tokens"],
        max_steps=3,
        learning_rate=0.002,
        batch_size=2,
        weight_decay=0.1,
    )

    metadata = prepared.artifact_metadata
    assert (prepared.run_root / "editor-graph.json").stat().st_mode & 0o777 == 0o600
    assert Path(metadata["manifest_path"]).is_file()
    assert Path(metadata["compatibility_report_path"]).is_file()
    assert Path(metadata["training_plan_path"]).is_file()
    assert Path(metadata["source_graph_snapshot_path"]).is_file()
    command = prepared.config.argv()
    assert command[command.index("--graph-file") + 1] == str(prepared.plan.launch_graph)
    assert command[command.index("--template-name") + 1] == "gpt2"
    assert command[command.index("--dataset-alias") + 1] == "tiny_tokens"
    assert command[command.index("--output-dir") + 1] == str(prepared.checkpoint_dir)
    assert command[command.index("--max-steps") + 1] == "3"
    persisted = json.loads(Path(metadata["training_plan_path"]).read_text(encoding="utf-8"))
    assert persisted["execution_ready"] is True
    assert persisted["compatibility_report"]["compatible"] is True


def test_gpt2_diff_server_completion_requires_and_returns_strict_v2_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = native_training.prepare_native_training_run(
        _preset_payload("gpt2_diff"),
        run_id="gpt2-diff-run",
        artifacts_dir=tmp_path,
        dataset_names=["tiny_tokens"],
        max_steps=1,
        learning_rate=0.001,
        batch_size=1,
        weight_decay=0.0,
    )
    proof_path = prepared.plan.graph_preflight_proof
    assert proof_path is not None and proof_path.is_file()
    command = prepared.config.argv()
    assert command.count("--graph-preflight-proof") == 1
    assert command[command.index("--graph-preflight-proof") + 1] == str(proof_path)
    assert prepared.artifact_metadata["graph_preflight_proof_path"] == str(proof_path)
    proof = json.loads(proof_path.read_text(encoding="utf-8"))

    def fake_run(config, *, runner: str) -> int:
        assert config is prepared.config
        assert runner == "auto"
        _write_gpt2_diff_bundle(
            prepared.checkpoint_dir,
            source_graph_sha256=prepared.plan.compatibility_report.graph_fingerprint,
            proof_contract_sha256=proof["contract_sha256"],
            geometry=proof["contract"]["geometry"],
        )
        return 0

    monkeypatch.setattr(native_training, "run_native_train", fake_run)
    checkpoint = native_training.execute_native_training(prepared)

    assert checkpoint == (
        prepared.checkpoint_dir / "model_00000001.diff.json"
    ).resolve()
    assert checkpoint.suffixes == [".diff", ".json"]
    assert prepared.plan.architecture_persistence_proven is True
    assert prepared.plan.trainer_consumes_native_ir is False


@pytest.mark.parametrize("tamper", ("diff_header", "negative_second_moment"))
def test_gpt2_diff_discovery_rejects_self_consistent_invalid_payloads(
    tmp_path: Path,
    tamper: str,
) -> None:
    source_sha256 = "a" * 64
    proof_sha256 = "b" * 64
    geometry = {
        "max_seq_len": 16,
        "vocab_size": 64,
        "padded_vocab_size": 64,
        "num_layers": 1,
        "model_dim": 8,
        "num_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 2,
        "mlp_hidden_dim": 32,
    }
    metadata_path = _write_gpt2_diff_bundle(
        tmp_path,
        source_graph_sha256=source_sha256,
        proof_contract_sha256=proof_sha256,
        geometry=geometry,
    )
    artifact_key = "diff_parameters"
    artifact_path = tmp_path / "diff_parameters_00000001.bin"
    payload = bytearray(artifact_path.read_bytes())
    if tamper == "diff_header":
        struct.pack_into("<q", payload, 0, 0)
    else:
        artifact_key = "diff_optimizer"
        artifact_path = tmp_path / "diff_optimizer_00000001.bin"
        payload = bytearray(artifact_path.read_bytes())
        struct.pack_into("<f", payload, 16 * 8 + 4, -1.0)
    artifact_path.write_bytes(payload)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[artifact_key]["sha256"] = hashlib.sha256(payload).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid graph-bound gpt2_diff checkpoint"):
        native_training._discover_checkpoint(
            tmp_path,
            expected_gpt2_diff_graph_sha256=source_sha256,
            expected_gpt2_diff_proof_contract_sha256=proof_sha256,
            expected_gpt2_diff_geometry=geometry,
        )


def test_gpt2_diff_discovery_rejects_sparse_oversize_before_artifact_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_sha256 = "a" * 64
    proof_sha256 = "b" * 64
    geometry = {
        "max_seq_len": 16,
        "vocab_size": 64,
        "padded_vocab_size": 64,
        "num_layers": 1,
        "model_dim": 8,
        "num_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 2,
        "mlp_hidden_dim": 32,
    }
    metadata_path = _write_gpt2_diff_bundle(
        tmp_path,
        source_graph_sha256=source_sha256,
        proof_contract_sha256=proof_sha256,
        geometry=geometry,
    )
    model_path = tmp_path / "model_00000001.bin"
    with model_path.open("r+b") as handle:
        handle.truncate(1 << 34)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model"]["nbytes"] = 1 << 34
    metadata["model"]["sha256"] = "c" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    original_read = native_training._read_regular_nofollow
    read_paths: list[Path] = []

    def observe_read(path: Path, **kwargs: object) -> bytes:
        read_paths.append(Path(path))
        return original_read(path, **kwargs)

    monkeypatch.setattr(native_training, "_read_regular_nofollow", observe_read)
    with pytest.raises(RuntimeError, match="invalid graph-bound gpt2_diff checkpoint"):
        native_training._discover_checkpoint(
            tmp_path,
            expected_gpt2_diff_graph_sha256=source_sha256,
            expected_gpt2_diff_proof_contract_sha256=proof_sha256,
            expected_gpt2_diff_geometry=geometry,
        )

    assert metadata_path in read_paths
    assert model_path not in read_paths


def test_gpt2_diff_artifact_validation_streams_with_bounded_python_memory(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "streamed-state.bin"
    header = b"h" * 256
    payload_chunk = b"\0" * (1024 * 1024)
    payload_chunks = 32
    digest = hashlib.sha256(header)
    with artifact.open("wb") as handle:
        handle.write(header)
        for _ in range(payload_chunks):
            handle.write(payload_chunk)
            digest.update(payload_chunk)
    expected_bytes = len(header) + payload_chunks * len(payload_chunk)
    del payload_chunk
    gc.collect()

    largest_validation_chunk = 0

    def validate_payload(chunk: bytes, _element_index: int) -> None:
        nonlocal largest_validation_chunk
        largest_validation_chunk = max(largest_validation_chunk, len(chunk))

    tracemalloc.start()
    try:
        actual_digest = native_training._stream_regular_artifact_nofollow(
            artifact,
            expected_bytes=expected_bytes,
            header_bytes=len(header),
            payload_element_size=4,
            validate_header=lambda observed: observed == header or pytest.fail(
                "streamed artifact header changed"
            ),
            validate_payload=validate_payload,
        )
        _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert actual_digest == digest.hexdigest()
    assert largest_validation_chunk <= 1024 * 1024
    assert peak_bytes < 8 * 1024 * 1024


def test_gpt2_diff_discovery_rejects_overflowing_continuation_number(
    tmp_path: Path,
) -> None:
    source_sha256 = "a" * 64
    proof_sha256 = "b" * 64
    geometry = {
        "max_seq_len": 16,
        "vocab_size": 64,
        "padded_vocab_size": 64,
        "num_layers": 1,
        "model_dim": 8,
        "num_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 2,
        "mlp_hidden_dim": 32,
    }
    metadata_path = _write_gpt2_diff_bundle(
        tmp_path,
        source_graph_sha256=source_sha256,
        proof_contract_sha256=proof_sha256,
        geometry=geometry,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["continuation"]["learning_rate"] = 10**4000
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid graph-bound gpt2_diff checkpoint"):
        native_training._discover_checkpoint(
            tmp_path,
            expected_gpt2_diff_graph_sha256=source_sha256,
            expected_gpt2_diff_proof_contract_sha256=proof_sha256,
            expected_gpt2_diff_geometry=geometry,
        )


@pytest.mark.parametrize(
    ("preset", "selector", "runtime"),
    [("llama", "llama", "eager"), ("llama_fast", "llama_fast", "compile")],
)
def test_prepare_llama_run_uses_plan_backed_sdk_and_forwards_weight_decay(
    tmp_path: Path,
    preset: str,
    selector: str,
    runtime: str,
) -> None:
    prepared = native_training.prepare_native_training_run(
        _preset_payload(preset),
        run_id=f"{preset}-run",
        artifacts_dir=tmp_path,
        dataset_names=["tiny_tokens"],
        max_steps=1,
        learning_rate=0.001,
        batch_size=1,
        weight_decay=0.07,
    )

    command = prepared.config.argv()

    assert prepared.plan.trainer_family == "llama"
    assert prepared.plan.training_selector == selector
    assert command.count("--train-llama-dataset-loop") == 1
    assert command[command.index("--template-name") + 1] == "llama"
    assert command[command.index("--graph-fingerprint") + 1] == (
        prepared.plan.compatibility_report.graph_fingerprint
    )
    assert command[command.index("--weight-decay") + 1] == "0.07"
    provenance = prepared.plan.artifact_metadata["architecture_provenance"]
    assert provenance["source_preset"] == selector
    assert provenance["source_runtime"] == runtime
    assert provenance["native_template_name"] == "llama"


@pytest.mark.parametrize(
    ("preset", "selector", "native_template"),
    (
        ("moe", "moe", "mixllama"),
        ("mixllama", "mixllama", "mixllama"),
        ("mixllama_fast", "mixllama_fast", "mixllama-fast"),
    ),
)
def test_prepare_standard_moe_run_uses_plan_backed_action_and_weight_decay(
    tmp_path: Path,
    preset: str,
    selector: str,
    native_template: str,
) -> None:
    prepared = native_training.prepare_native_training_run(
        _preset_payload(preset),
        run_id=f"{preset}-run",
        artifacts_dir=tmp_path,
        dataset_names=["tiny_tokens"],
        max_steps=1,
        learning_rate=0.001,
        batch_size=1,
        weight_decay=0.07,
    )

    command = prepared.config.argv()

    assert prepared.plan.trainer_family == "mixllama"
    assert prepared.plan.training_selector == selector
    assert command.count("--train-moe-dataset-loop") == 1
    assert "--train-transformer-lm" not in command
    assert command[command.index("--template-name") + 1] == native_template
    assert command[command.index("--multiple-of") + 1] == "0"
    assert command[command.index("--layers-per-expert") + 1] == "1"
    assert command[command.index("--router-aux-loss-coef") + 1] == "0.01"
    assert command[command.index("--weight-decay") + 1] == "0.07"
    assert command[command.index("--graph-fingerprint") + 1] == (
        prepared.plan.compatibility_report.graph_fingerprint
    )


@pytest.mark.parametrize("datasets", [[], ["one", "two"], ["__semantic_builtin__"]])
def test_prepare_run_rejects_unroutable_datasets_before_writing(
    tmp_path: Path,
    datasets: list[str],
) -> None:
    with pytest.raises(ValueError, match="dataset"):
        native_training.prepare_native_training_run(
            _preset_payload(),
            run_id="rejected",
            artifacts_dir=tmp_path,
            dataset_names=datasets,
            max_steps=1,
            learning_rate=0.001,
            batch_size=1,
            weight_decay=0.0,
        )
    assert not (tmp_path / "runs" / "rejected").exists()


def test_execute_native_run_requires_and_returns_real_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = native_training.prepare_native_training_run(
        _preset_payload(),
        run_id="run-two",
        artifacts_dir=tmp_path,
        dataset_names=["tiny_tokens"],
        max_steps=1,
        learning_rate=0.001,
        batch_size=1,
        weight_decay=0.0,
    )

    def fake_run(config, *, runner: str) -> int:
        assert config is prepared.config
        assert runner == "auto"
        prepared.checkpoint_dir.mkdir()
        (prepared.checkpoint_dir / "model_00000001.bin").write_bytes(b"native-checkpoint")
        return 0

    monkeypatch.setattr(native_training, "run_native_train", fake_run)
    checkpoint = native_training.execute_native_training(prepared)
    assert checkpoint == (prepared.checkpoint_dir / "model_00000001.bin").resolve()


def test_discover_checkpoint_returns_validated_llama_metadata_not_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_native_family_checkpoint import _write_checkpoint

    metadata, sidecar, _payload = _write_checkpoint(tmp_path)
    inspected: list[Path] = []
    inspect_checkpoint = native_training.inspect_native_family_llama_checkpoint

    def inspect(path: str | Path):
        inspected.append(Path(path))
        return inspect_checkpoint(path)

    monkeypatch.setattr(native_training, "inspect_native_family_llama_checkpoint", inspect)

    checkpoint = native_training._discover_checkpoint(tmp_path)

    assert inspected == [metadata.resolve()]
    assert checkpoint == metadata.resolve()
    assert checkpoint != sidecar.resolve()


def test_discover_checkpoint_binds_llama_metadata_to_prepared_graph_fingerprint(
    tmp_path: Path,
) -> None:
    from tests.test_native_family_checkpoint import _write_checkpoint

    metadata, _sidecar, payload = _write_checkpoint(tmp_path)
    source_sha256 = "a" * 64
    payload["inference_contract"]["training"]["source_graph"] = {
        "filename": "source-graph.json",
        "sha256": source_sha256,
        "byte_identity_verified": True,
    }
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    assert native_training._discover_checkpoint(
        tmp_path,
        expected_llama_graph_sha256=source_sha256,
    ) == metadata.resolve()
    with pytest.raises(RuntimeError, match="does not match the prepared training plan"):
        native_training._discover_checkpoint(
            tmp_path,
            expected_llama_graph_sha256="b" * 64,
        )


@pytest.mark.parametrize("damage", ["sidecar", "done"])
def test_discover_checkpoint_rejects_incomplete_or_tampered_llama_bundle(
    tmp_path: Path,
    damage: str,
) -> None:
    from tests.test_native_family_checkpoint import _write_checkpoint

    metadata, sidecar, _payload = _write_checkpoint(tmp_path)
    if damage == "sidecar":
        raw = bytearray(sidecar.read_bytes())
        raw[-1] ^= 0xFF
        sidecar.write_bytes(raw)
    else:
        metadata.with_name("llama_native_family_model_DONE").unlink()

    with pytest.raises(RuntimeError, match="invalid canonical LLaMA v2 checkpoint"):
        native_training._discover_checkpoint(tmp_path)


def test_discover_checkpoint_accepts_only_graph_bound_standard_moe_bundle(
    tmp_path: Path,
) -> None:
    from tests.test_native_resident_moe import _write_moe_checkpoint

    metadata, _sidecar = _write_moe_checkpoint(tmp_path)

    assert native_training._discover_checkpoint(
        tmp_path,
        expected_standard_moe_graph_sha256="a" * 64,
    ) == metadata.resolve()
    with pytest.raises(RuntimeError, match="source graph does not match"):
        native_training._discover_checkpoint(
            tmp_path,
            expected_standard_moe_graph_sha256="b" * 64,
        )


def test_discover_checkpoint_selects_strict_graph_bound_moa_metadata(
    tmp_path: Path,
) -> None:
    from tests.test_native_moa_checkpoint import _write_bundle

    metadata, graph_path, model, _payload = _write_bundle(tmp_path)
    graph_fingerprint = hashlib.sha256(graph_path.read_bytes()).hexdigest()

    checkpoint = native_training._discover_checkpoint(
        tmp_path,
        expected_moa_graph_sha256=graph_fingerprint,
        moa_source_graph_path=graph_path,
        moa_model=model,
    )

    assert checkpoint == metadata.resolve()


def test_discover_checkpoint_does_not_fall_back_from_tampered_moa_metadata(
    tmp_path: Path,
) -> None:
    from tests.test_native_moa_checkpoint import _write_bundle

    metadata, graph_path, model, payload = _write_bundle(tmp_path)
    graph_fingerprint = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    payload["model"]["sha256"] = "0" * 64
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    assert list(tmp_path.glob("*.bin"))

    with pytest.raises(RuntimeError, match="invalid graph-bound MoA checkpoint"):
        native_training._discover_checkpoint(
            tmp_path,
            expected_moa_graph_sha256=graph_fingerprint,
            moa_source_graph_path=graph_path,
            moa_model=model,
        )


def test_discover_checkpoint_does_not_fall_back_when_moa_metadata_is_missing(
    tmp_path: Path,
) -> None:
    from tests.test_native_moa_checkpoint import _write_bundle

    metadata, graph_path, model, _payload = _write_bundle(tmp_path)
    graph_fingerprint = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    metadata.unlink()
    assert list(tmp_path.glob("*.bin"))

    with pytest.raises(
        RuntimeError,
        match=r"without a strict model_XXXXXXXX\.moa\.json checkpoint",
    ):
        native_training._discover_checkpoint(
            tmp_path,
            expected_moa_graph_sha256=graph_fingerprint,
            moa_source_graph_path=graph_path,
            moa_model=model,
        )


def test_discover_checkpoint_does_not_treat_other_json_as_a_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "training-metrics.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )

    def unexpected_inspection(_path: str | Path):
        raise AssertionError("unrelated JSON must not enter LLaMA checkpoint inspection")

    monkeypatch.setattr(
        native_training,
        "inspect_native_family_llama_checkpoint",
        unexpected_inspection,
    )

    with pytest.raises(RuntimeError, match="without a persisted .bin checkpoint or canonical LLaMA"):
        native_training._discover_checkpoint(tmp_path)


def test_discover_checkpoint_does_not_fall_back_from_invalid_llama_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / "model.bin").write_bytes(b"valid-dense-checkpoint")
    (tmp_path / "llama_native_family_model_00000000.json").write_text(
        json.dumps({"format": "not-a-checkpoint"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="invalid canonical LLaMA v2 checkpoint"):
        native_training._discover_checkpoint(tmp_path)


def test_execute_native_run_fails_closed_on_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = native_training.prepare_native_training_run(
        _preset_payload(),
        run_id="run-three",
        artifacts_dir=tmp_path,
        dataset_names=["tiny_tokens"],
        max_steps=1,
        learning_rate=0.001,
        batch_size=1,
        weight_decay=0.0,
    )
    monkeypatch.setattr(native_training, "run_native_train", lambda *_args, **_kwargs: 17)
    with pytest.raises(RuntimeError, match="status 17"):
        native_training.execute_native_training(prepared)


def test_execute_gpt2_moa_forwards_strict_checkpoint_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = native_training.prepare_native_training_run(
        _preset_payload("gpt2_moa"),
        run_id="run-gpt2-moa",
        artifacts_dir=tmp_path,
        dataset_names=["tiny_tokens"],
        max_steps=1,
        learning_rate=0.001,
        batch_size=1,
        weight_decay=0.0,
    )
    assert prepared.plan.manifest is not None
    selected = prepared.checkpoint_dir / "model_00000001.moa.json"
    captured: dict[str, object] = {}

    def fake_run(config, *, runner: str) -> int:
        assert config is prepared.config
        assert runner == "auto"
        prepared.checkpoint_dir.mkdir()
        selected.write_text("{}", encoding="utf-8")
        return 0

    def fake_discover(checkpoint_dir: Path, **kwargs: object) -> Path:
        captured["checkpoint_dir"] = checkpoint_dir
        captured.update(kwargs)
        return selected.resolve()

    monkeypatch.setattr(native_training, "run_native_train", fake_run)
    monkeypatch.setattr(native_training, "_discover_checkpoint", fake_discover)

    checkpoint = native_training.execute_native_training(prepared)

    assert checkpoint == selected.resolve()
    assert captured == {
        "checkpoint_dir": prepared.checkpoint_dir,
        "expected_gpt2_diff_graph_sha256": None,
        "expected_gpt2_diff_proof_contract_sha256": None,
        "expected_gpt2_diff_geometry": None,
        "expected_llama_graph_sha256": None,
        "expected_standard_moe_graph_sha256": None,
        "expected_moa_graph_sha256": (
            prepared.plan.compatibility_report.graph_fingerprint
        ),
        "moa_source_graph_path": prepared.plan.launch_graph,
        "moa_model": prepared.plan.manifest.model,
    }
