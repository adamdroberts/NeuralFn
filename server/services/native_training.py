"""Editor/server bridge for fail-closed graph-authored native training.

This module deliberately delegates graph lowering and trainer selection to the
same public Native IR planner and native trainer frontend used by the CLI.  It
does not interpret graph source code and it never promotes a structurally
lowerable graph into an executable adapter on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import tempfile
from typing import Any, Callable, Mapping, Sequence

from neuralfn.native_family_checkpoint import inspect_native_family_llama_checkpoint
from neuralfn.native_moe_checkpoint import inspect_native_family_standard_moe_checkpoint
from neuralfn.native_moa_checkpoint import inspect_native_moa_checkpoint
from neuralfn.native_graph_train import NativeGraphTrainPlan, plan_native_graph_training
from neuralfn.native_train import (
    NativeTrainRunConfig,
    build_native_train_run_config,
    run_native_train,
)


NATIVE_RUNTIME = "native-cuda"


class NativeTrainingIncompatibleError(ValueError):
    """Raised before launch when Native IR or its trainer adapter is unsupported."""

    def __init__(self, metadata: Mapping[str, Any]) -> None:
        self.metadata = dict(metadata)
        issues = list(self.metadata.get("issues") or [])
        if issues:
            first = issues[0]
            message = (
                f"Native training is incompatible at {first.get('path', 'root')}: "
                f"{first.get('message', first.get('code', 'unsupported graph'))}"
            )
        else:
            message = "Native training is incompatible with the current graph."
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class NativeTrainingPreparation:
    plan: NativeGraphTrainPlan
    run_root: Path
    checkpoint_dir: Path
    config: NativeTrainRunConfig
    compatibility_report: dict[str, Any]
    artifact_metadata: dict[str, Any]


def _graph_bytes(graph_payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(graph_payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_private_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        stat.S_IRUSR | stat.S_IWUSR,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def _combined_issues(plan: NativeGraphTrainPlan) -> list[dict[str, Any]]:
    return [
        issue.to_dict()
        for issue in (*plan.compatibility_report.issues, *plan.training_issues)
        if issue.severity == "error"
    ]


def native_training_metadata(plan: NativeGraphTrainPlan) -> dict[str, Any]:
    """Return the stable compatibility/artifact shape shared by REST and MCP."""

    issues = _combined_issues(plan)
    return {
        "runtime": NATIVE_RUNTIME,
        "compatible": bool(plan.execution_ready),
        "execution_ready": bool(plan.execution_ready),
        "trainer_family": plan.trainer_family,
        "training_selector": plan.training_selector,
        "native_target": plan.native_target,
        "adapter_mode": plan.adapter_mode,
        "trainer_registered": bool(plan.trainer_registered),
        "architecture_persistence_proven": bool(plan.architecture_persistence_proven),
        "trainer_consumes_native_ir": bool(plan.trainer_consumes_native_ir),
        "graph_preflight_enforced": bool(plan.graph_preflight_enforced),
        "graph_preflight_proof": (
            str(plan.graph_preflight_proof)
            if plan.graph_preflight_proof is not None
            else None
        ),
        "blockers": list(plan.blockers),
        "issues": issues,
        "compatibility_report": plan.compatibility_report.to_dict(),
        "training_compatibility": {
            "compatible": bool(plan.training_compatible),
            "issues": [issue.to_dict() for issue in plan.training_issues],
        },
        "artifact_metadata": dict(plan.artifact_metadata),
    }


def preflight_native_training(graph_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Compile an editor graph without writing persistent run artifacts."""

    with tempfile.TemporaryDirectory(prefix="neuralfn-editor-native-preflight-") as tmp:
        graph_path = Path(tmp) / "editor-graph.json"
        _write_private_exclusive(graph_path, _graph_bytes(graph_payload))
        plan = plan_native_graph_training(graph_path)
        metadata = native_training_metadata(plan)
    # Dry-run paths point into a deleted temporary directory.  Keep the
    # fingerprint and compatibility facts, but do not expose stale paths.
    artifact_metadata = dict(metadata["artifact_metadata"])
    artifact_metadata.update(
        {
            "source_graph": "editor-session",
            "requested_artifact_dir": None,
            "artifact_dir": None,
            "manifest_path": None,
            "compatibility_report_path": None,
            "training_plan_path": None,
            "source_graph_snapshot_path": None,
            "materialized": False,
        }
    )
    metadata["artifact_metadata"] = artifact_metadata
    return metadata


def _validate_native_options(
    dataset_names: Sequence[str], *, max_steps: int, batch_size: int
) -> tuple[str, ...]:
    datasets = tuple(str(name).strip() for name in dataset_names if str(name).strip())
    if len(datasets) != 1:
        raise ValueError(
            "Native CUDA editor training requires exactly one project-accessible cached dataset alias; "
            f"received {len(datasets)}."
        )
    if datasets[0] == "__semantic_builtin__":
        raise ValueError(
            "Native CUDA editor training requires a persisted cached dataset alias; "
            "the in-memory semantic builtin is not supported."
        )
    if max_steps <= 0:
        raise ValueError("Native CUDA editor training requires epochs/max steps greater than zero.")
    if batch_size <= 0:
        raise ValueError("Native CUDA editor training requires batch_size greater than zero.")
    return datasets


def _native_args(
    plan: NativeGraphTrainPlan,
    *,
    dataset_names: Sequence[str],
    checkpoint_dir: Path,
    max_steps: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
) -> tuple[str, ...]:
    datasets = _validate_native_options(
        dataset_names,
        max_steps=max_steps,
        batch_size=batch_size,
    )
    args = [
        *plan.trainer_arguments,
        "--dataset-alias",
        datasets[0],
        "--output-dir",
        str(checkpoint_dir),
        "--max-steps",
        str(int(max_steps)),
        "--learning-rate",
        str(float(learning_rate)),
        "--batch-size",
        str(int(batch_size)),
        "--weight-decay",
        str(float(weight_decay)),
        "--checkpoint-every-steps",
        str(int(max_steps)),
    ]
    if not any(
        argument
        in {
            "--train-transformer-lm",
            "--train-llama-dataset-loop",
            "--train-moe-dataset-loop",
        }
        for argument in args
    ):
        args.append(
            "--train-llama-dataset-loop"
            if plan.trainer_family == "llama"
            else (
                "--train-moe-dataset-loop"
                if plan.trainer_family == "mixllama"
                else "--train-transformer-lm"
            )
        )
    return tuple(args)


def prepare_native_training_run(
    graph_payload: Mapping[str, Any],
    *,
    run_id: str,
    artifacts_dir: str | Path,
    dataset_names: Sequence[str],
    max_steps: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
) -> NativeTrainingPreparation:
    """Materialize immutable run IR and build the canonical native command."""

    # Validate request-only constraints before creating a persistent run tree.
    _validate_native_options(
        dataset_names,
        max_steps=max_steps,
        batch_size=batch_size,
    )

    normalized_run_id = str(run_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", normalized_run_id) or normalized_run_id in {".", ".."}:
        raise ValueError("Native training run_id must be one safe path component.")
    root = Path(artifacts_dir).expanduser().resolve()
    runs_root = root / "runs"
    runs_root.mkdir(mode=stat.S_IRWXU, parents=True, exist_ok=True)
    run_root = runs_root / normalized_run_id
    run_root.mkdir(mode=stat.S_IRWXU, exist_ok=False)
    source_graph = run_root / "editor-graph.json"
    _write_private_exclusive(source_graph, _graph_bytes(graph_payload))

    native_ir_dir = run_root / "native-ir"
    plan = plan_native_graph_training(
        source_graph,
        artifact_dir=native_ir_dir,
        materialize=True,
    )
    metadata = native_training_metadata(plan)
    if not plan.execution_ready:
        raise NativeTrainingIncompatibleError(metadata)

    checkpoint_dir = run_root / "checkpoints"
    args = _native_args(
        plan,
        dataset_names=dataset_names,
        checkpoint_dir=checkpoint_dir,
        max_steps=max_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        weight_decay=weight_decay,
    )
    config = build_native_train_run_config(
        model_family=plan.trainer_family,
        args=args,
        graph_file=str(plan.launch_graph),
    )
    artifact_metadata = {
        **dict(plan.artifact_metadata),
        "run_id": str(run_id),
        "run_root": str(run_root),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_path": None,
        "native_command": config.argv(),
        "native_runner": "auto",
    }
    return NativeTrainingPreparation(
        plan=plan,
        run_root=run_root,
        checkpoint_dir=checkpoint_dir,
        config=config,
        compatibility_report=dict(metadata["compatibility_report"]),
        artifact_metadata=artifact_metadata,
    )


def _read_regular_nofollow(
    path: Path,
    *,
    max_bytes: int | None = None,
    expected_bytes: int | None = None,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(f"Native checkpoint artifact is not a regular file: {path}")
        if max_bytes is not None and file_stat.st_size > max_bytes:
            raise RuntimeError(
                f"Native checkpoint artifact exceeds {max_bytes} bytes: {path}"
            )
        if expected_bytes is not None and file_stat.st_size != expected_bytes:
            raise RuntimeError(
                "Native checkpoint artifact size does not match its exact layout: "
                f"expected {expected_bytes}, got {file_stat.st_size}: {path}"
            )
        chunks: list[bytes] = []
        remaining = file_stat.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"Native checkpoint artifact was truncated: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"Native checkpoint artifact grew while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _stream_regular_artifact_nofollow(
    path: Path,
    *,
    expected_bytes: int,
    header_bytes: int,
    payload_element_size: int,
    validate_header: Callable[[bytes], None],
    validate_payload: Callable[[bytes, int], None],
) -> str:
    """Hash and validate one exact-layout artifact with bounded memory.

    The fixed header is retained, while the potentially multi-gigabyte payload
    is handed to ``validate_payload`` in aligned chunks.  The returned digest
    covers the exact bytes read from the same no-follow descriptor.
    """

    if (
        expected_bytes < header_bytes
        or header_bytes < 0
        or payload_element_size <= 0
        or (expected_bytes - header_bytes) % payload_element_size != 0
    ):
        raise RuntimeError(f"Native checkpoint artifact layout is invalid: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(f"Native checkpoint artifact is not a regular file: {path}")
        if file_stat.st_size != expected_bytes:
            raise RuntimeError(
                "Native checkpoint artifact size does not match its exact layout: "
                f"expected {expected_bytes}, got {file_stat.st_size}: {path}"
            )

        header_parts: list[bytes] = []
        header_remaining = header_bytes
        while header_remaining:
            chunk = os.read(descriptor, header_remaining)
            if not chunk:
                raise RuntimeError(f"Native checkpoint artifact was truncated: {path}")
            header_parts.append(chunk)
            header_remaining -= len(chunk)
        header = b"".join(header_parts)
        validate_header(header)

        digest = hashlib.sha256()
        digest.update(header)
        payload_remaining = expected_bytes - header_bytes
        pending = b""
        element_index = 0
        while payload_remaining:
            chunk = os.read(descriptor, min(1024 * 1024, payload_remaining))
            if not chunk:
                raise RuntimeError(f"Native checkpoint artifact was truncated: {path}")
            digest.update(chunk)
            payload_remaining -= len(chunk)
            data = pending + chunk
            aligned_bytes = len(data) - (len(data) % payload_element_size)
            if aligned_bytes:
                aligned = data[:aligned_bytes]
                validate_payload(aligned, element_index)
                element_index += aligned_bytes // payload_element_size
            pending = data[aligned_bytes:]
        if pending:
            raise RuntimeError(f"Native checkpoint artifact payload is misaligned: {path}")
        if os.read(descriptor, 1):
            raise RuntimeError(f"Native checkpoint artifact grew while reading: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate object key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(raw_value: str) -> None:
        raise ValueError(f"non-finite numeric constant {raw_value!r}")

    def parse_finite_float(raw_value: str) -> float:
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number {raw_value!r}")
        return value

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_float=parse_finite_float,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Native trainer produced invalid {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Native trainer produced non-object {label}.")
    return payload


def _is_exact_positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_finite_json_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _inspect_gpt2_diff_checkpoint(
    metadata_path: Path,
    *,
    checkpoint_root: Path,
    expected_source_graph_sha256: str,
    expected_proof_contract_sha256: str,
    expected_geometry: Mapping[str, Any],
) -> Path:
    if metadata_path.parent.resolve() != checkpoint_root:
        raise RuntimeError("Native gpt2_diff metadata is not directly contained in its checkpoint directory.")
    match = re.fullmatch(r"model_([0-9]{8})\.diff\.json", metadata_path.name)
    if match is None:
        raise RuntimeError("Native gpt2_diff metadata filename is not canonical.")
    step_text = match.group(1)
    step = int(step_text)
    if step <= 0:
        raise RuntimeError("Native gpt2_diff checkpoint step must be positive.")
    raw_metadata = _read_regular_nofollow(metadata_path, max_bytes=1024 * 1024)
    if not raw_metadata:
        raise RuntimeError("Native gpt2_diff metadata is empty.")
    metadata = _strict_json_object(raw_metadata, label="gpt2_diff checkpoint metadata")
    expected_root_fields = {
        "schema",
        "version",
        "preset",
        "checkpoint_kind",
        "step",
        "model",
        "dense_parameters",
        "dense_optimizer",
        "diff_parameters",
        "diff_optimizer",
        "done_marker",
        "source_graph",
        "lambda",
        "geometry",
        "graph_preflight_proof",
        "continuation",
    }
    if set(metadata) != expected_root_fields:
        raise RuntimeError("Native gpt2_diff checkpoint metadata fields are not canonical.")
    if (
        metadata["schema"] != "neuralfn.native_gpt2_diff.training_checkpoint"
        or type(metadata["version"]) is not int
        or metadata["version"] != 2
        or metadata["preset"] != "gpt2_diff"
        or metadata["checkpoint_kind"] != "trained_dense_v5_plus_diff_v1"
        or type(metadata["step"]) is not int
        or metadata["step"] != step
    ):
        raise RuntimeError("Native gpt2_diff checkpoint identity is not canonical.")

    source_graph = metadata["source_graph"]
    if not isinstance(source_graph, dict) or set(source_graph) != {
        "filename",
        "sha256",
        "byte_identity_verified",
    }:
        raise RuntimeError("Native gpt2_diff source-graph binding is not canonical.")
    source_filename = source_graph["filename"]
    if (
        not isinstance(source_filename, str)
        or not source_filename
        or source_filename in {".", ".."}
        or "/" in source_filename
        or "\\" in source_filename
        or source_graph["sha256"] != expected_source_graph_sha256
        or source_graph["byte_identity_verified"] is not True
    ):
        raise RuntimeError(
            "Native gpt2_diff checkpoint source graph does not match the prepared training plan fingerprint."
        )

    proof = metadata["graph_preflight_proof"]
    if (
        not isinstance(proof, dict)
        or set(proof) != {"schema", "version", "contract_sha256"}
        or proof["schema"] != "neuralfn.native_graph_training_proof"
        or type(proof["version"]) is not int
        or proof["version"] != 1
        or proof["contract_sha256"] != expected_proof_contract_sha256
    ):
        raise RuntimeError(
            "Native gpt2_diff checkpoint proof does not match the prepared training plan contract."
        )

    geometry = metadata["geometry"]
    geometry_fields = {
        "max_seq_len",
        "vocab_size",
        "padded_vocab_size",
        "num_layers",
        "model_dim",
        "num_heads",
        "head_dim",
        "mlp_hidden_dim",
    }
    if (
        not isinstance(geometry, dict)
        or set(geometry) != geometry_fields
        or not all(_is_exact_positive_int(geometry[field]) for field in geometry_fields)
        or geometry["model_dim"] != geometry["num_heads"] * geometry["head_dim"]
        or geometry["mlp_hidden_dim"] != 4 * geometry["model_dim"]
        or any(
            geometry[field] != expected_geometry.get(field)
            for field in geometry_fields
        )
    ):
        raise RuntimeError("Native gpt2_diff checkpoint geometry is not canonical.")
    lambda_contract = metadata["lambda"]
    if (
        not isinstance(lambda_contract, dict)
        or set(lambda_contract) != {"count", "dtype", "initial_value", "output_scale"}
        or type(lambda_contract["count"]) is not int
        or lambda_contract["count"] != geometry["num_layers"]
        or lambda_contract["dtype"] != "float32"
        or type(lambda_contract["initial_value"]) is not float
        or lambda_contract["initial_value"] != 0.8
        or type(lambda_contract["output_scale"]) is not float
        or lambda_contract["output_scale"] != 0.2
    ):
        raise RuntimeError("Native gpt2_diff lambda contract is not canonical.")
    continuation = metadata["continuation"]
    continuation_fields = {
        "optimizer_steps_completed",
        "train_microbatches_completed",
        "microbatch_in_optimizer_step",
        "batch_size",
        "seq_len",
        "microbatch_tokens",
        "requested_train_batch_tokens",
        "effective_train_batch_tokens",
        "grad_accum_steps",
        "train_seed_explicit",
        "train_seed",
        "sampler_start_batch",
        "sampler_total_batches",
        "train_shard_count",
        "train_shard_total_bytes",
        "train_shards_sha256",
        "learning_rate",
        "lr_schedule",
        "lr_schedule_total_steps",
        "warmup_steps",
        "final_lr_fraction",
        "weight_decay",
        "beta1",
        "beta2",
        "adam_eps",
        "grad_clip_norm",
        "lm_head_row_chunk_size",
        "bf16_block_weight_params",
        "bf16_block_dweight_staging",
        "dweight_first_microbatch_beta_zero",
        "numerics_profile_sha256",
    }
    positive_continuation_ints = {
        "optimizer_steps_completed",
        "train_microbatches_completed",
        "batch_size",
        "seq_len",
        "microbatch_tokens",
        "requested_train_batch_tokens",
        "effective_train_batch_tokens",
        "grad_accum_steps",
        "sampler_total_batches",
        "train_shard_count",
        "train_shard_total_bytes",
        "lr_schedule_total_steps",
        "lm_head_row_chunk_size",
    }
    nonnegative_continuation_ints = {
        "microbatch_in_optimizer_step",
        "sampler_start_batch",
        "warmup_steps",
    }
    finite_continuation_numbers = {
        "learning_rate",
        "final_lr_fraction",
        "weight_decay",
        "beta1",
        "beta2",
        "adam_eps",
        "grad_clip_norm",
    }
    boolean_continuation_fields = {
        "train_seed_explicit",
        "bf16_block_weight_params",
        "bf16_block_dweight_staging",
        "dweight_first_microbatch_beta_zero",
    }
    if not isinstance(continuation, dict) or set(continuation) != continuation_fields:
        raise RuntimeError("Native gpt2_diff continuation fields are not canonical.")
    if (
        not all(
            _is_exact_positive_int(continuation[field])
            for field in positive_continuation_ints
        )
        or not all(
            type(continuation[field]) is int and continuation[field] >= 0
            for field in nonnegative_continuation_ints
        )
        or type(continuation["train_seed"]) is not int
        or not all(
            _is_finite_json_number(continuation[field])
            for field in finite_continuation_numbers
        )
        or not all(
            type(continuation[field]) is bool
            for field in boolean_continuation_fields
        )
        or continuation["lr_schedule"] not in {"constant", "cosine"}
        or not _is_sha256(continuation["train_shards_sha256"])
        or not _is_sha256(continuation["numerics_profile_sha256"])
        or continuation["optimizer_steps_completed"] != step
        or continuation["microbatch_in_optimizer_step"] != 0
        or continuation["seq_len"] != geometry["max_seq_len"]
        or continuation["microbatch_tokens"]
        != continuation["batch_size"] * continuation["seq_len"]
        or continuation["requested_train_batch_tokens"]
        < continuation["microbatch_tokens"]
        or continuation["effective_train_batch_tokens"]
        != continuation["microbatch_tokens"] * continuation["grad_accum_steps"]
        or continuation["train_microbatches_completed"]
        != step * continuation["grad_accum_steps"]
    ):
        raise RuntimeError("Native gpt2_diff continuation contract is not canonical.")

    max_seq_len = geometry["max_seq_len"]
    vocab_size = geometry["vocab_size"]
    padded_vocab_size = geometry["padded_vocab_size"]
    num_layers = geometry["num_layers"]
    model_dim = geometry["model_dim"]
    num_heads = geometry["num_heads"]
    dense_tensor_elements = [padded_vocab_size * model_dim, max_seq_len * model_dim]
    for _layer in range(num_layers):
        dense_tensor_elements.extend(
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
    dense_tensor_elements.extend((model_dim, model_dim))
    dense_tensor_count = 4 + 12 * num_layers
    dense_parameter_count = sum(dense_tensor_elements)
    if len(dense_tensor_elements) != dense_tensor_count:
        raise RuntimeError("Native gpt2_diff dense tensor layout is inconsistent.")
    expected_sizes = {
        "model": 256 * 4 + dense_parameter_count * 2,
        "dense_parameters": 32 * 8 + dense_parameter_count * 4,
        "dense_optimizer": 32 * 8 + dense_parameter_count * 2 * 4,
        "diff_parameters": 16 * 8 + num_layers * 4,
        "diff_optimizer": 16 * 8 + num_layers * 2 * 4,
    }

    artifacts = {
        "model": (
            f"model_{step_text}.bin",
            "neuralfn.native_dense_gpt.v5",
        ),
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
    artifact_declarations: dict[str, Mapping[str, Any]] = {}
    for key, (filename, artifact_format) in artifacts.items():
        declaration = metadata[key]
        if (
            not isinstance(declaration, dict)
            or set(declaration) != {"path", "format", "nbytes", "sha256"}
            or declaration["path"] != filename
            or declaration["format"] != artifact_format
            or type(declaration["nbytes"]) is not int
            or declaration["nbytes"] != expected_sizes[key]
            or not _is_sha256(declaration["sha256"])
        ):
            raise RuntimeError(f"Native gpt2_diff {key} declaration is not canonical.")
        artifact_declarations[key] = declaration

    continuation_grad_accum = continuation.get("grad_accum_steps")
    continuation_microbatches = continuation.get("train_microbatches_completed")
    if (
        not _is_exact_positive_int(continuation_grad_accum)
        or not _is_exact_positive_int(continuation_microbatches)
        or continuation_microbatches != step * continuation_grad_accum
    ):
        raise RuntimeError("Native gpt2_diff continuation accumulation state is invalid.")

    def require_artifact_digest(key: str, digest: str) -> None:
        if digest != artifact_declarations[key]["sha256"]:
            raise RuntimeError(f"Native gpt2_diff {key} SHA-256 does not match metadata.")

    def validate_model_header(header_bytes: bytes) -> None:
        model_header = struct.unpack("<256i", header_bytes)
        if model_header[:8] != (
            20240326,
            5,
            max_seq_len,
            vocab_size,
            num_layers,
            num_heads,
            model_dim,
            padded_vocab_size,
        ) or any(model_header[8:]):
            raise RuntimeError("Native gpt2_diff dense-v5 model header is invalid.")

    def validate_model_payload(chunk: bytes, _element_index: int) -> None:
        if any(
            (value & 0x7F80) == 0x7F80
            for (value,) in struct.iter_unpack("<H", chunk)
        ):
            raise RuntimeError("Native gpt2_diff dense-v5 BF16 parameters must be finite.")

    require_artifact_digest(
        "model",
        _stream_regular_artifact_nofollow(
            checkpoint_root / artifacts["model"][0],
            expected_bytes=expected_sizes["model"],
            header_bytes=256 * 4,
            payload_element_size=2,
            validate_header=validate_model_header,
            validate_payload=validate_model_payload,
        ),
    )

    def dense_state_header_validator(
        key: str,
        *,
        magic: int,
        payload_elements: int,
    ) -> Callable[[bytes], None]:
        def validate(header_bytes: bytes) -> None:
            header = struct.unpack("<32q", header_bytes)
            if header[:13] != (
                magic,
                1,
                step,
                max_seq_len,
                vocab_size,
                num_layers,
                num_heads,
                model_dim,
                padded_vocab_size,
                dense_tensor_count,
                payload_elements,
                continuation_grad_accum,
                continuation_microbatches,
            ) or any(header[13:]):
                raise RuntimeError(f"Native gpt2_diff {key} header is invalid.")

        return validate

    def validate_dense_parameters(chunk: bytes, _element_index: int) -> None:
        if any(
            not math.isfinite(value)
            for (value,) in struct.iter_unpack("<f", chunk)
        ):
            raise RuntimeError("Native gpt2_diff dense FP32 parameters must be finite.")

    require_artifact_digest(
        "dense_parameters",
        _stream_regular_artifact_nofollow(
            checkpoint_root / artifacts["dense_parameters"][0],
            expected_bytes=expected_sizes["dense_parameters"],
            header_bytes=32 * 8,
            payload_element_size=4,
            validate_header=dense_state_header_validator(
                "dense_parameters",
                magic=20260711,
                payload_elements=dense_parameter_count,
            ),
            validate_payload=validate_dense_parameters,
        ),
    )

    optimizer_tensor_index = 0
    optimizer_second_moment = False
    optimizer_tensor_remaining = dense_tensor_elements[0]

    def validate_dense_optimizer(chunk: bytes, _element_index: int) -> None:
        nonlocal optimizer_tensor_index
        nonlocal optimizer_second_moment
        nonlocal optimizer_tensor_remaining
        for (value,) in struct.iter_unpack("<f", chunk):
            if not math.isfinite(value) or (optimizer_second_moment and value < 0.0):
                raise RuntimeError("Native gpt2_diff dense optimizer moments are invalid.")
            optimizer_tensor_remaining -= 1
            if optimizer_tensor_remaining == 0:
                if not optimizer_second_moment:
                    optimizer_second_moment = True
                    optimizer_tensor_remaining = dense_tensor_elements[
                        optimizer_tensor_index
                    ]
                else:
                    optimizer_second_moment = False
                    optimizer_tensor_index += 1
                    optimizer_tensor_remaining = (
                        dense_tensor_elements[optimizer_tensor_index]
                        if optimizer_tensor_index < len(dense_tensor_elements)
                        else 0
                    )

    require_artifact_digest(
        "dense_optimizer",
        _stream_regular_artifact_nofollow(
            checkpoint_root / artifacts["dense_optimizer"][0],
            expected_bytes=expected_sizes["dense_optimizer"],
            header_bytes=32 * 8,
            payload_element_size=4,
            validate_header=dense_state_header_validator(
                "dense_optimizer",
                magic=20260710,
                payload_elements=dense_parameter_count * 2,
            ),
            validate_payload=validate_dense_optimizer,
        ),
    )
    if (
        optimizer_tensor_index != len(dense_tensor_elements)
        or optimizer_second_moment
        or optimizer_tensor_remaining != 0
    ):
        raise RuntimeError("Native gpt2_diff dense optimizer layout is inconsistent.")

    def diff_state_header_validator(
        key: str,
        *,
        magic: int,
        tensor_count: int,
    ) -> Callable[[bytes], None]:
        def validate(header_bytes: bytes) -> None:
            header = struct.unpack("<16q", header_bytes)
            if header[:9] != (
                magic,
                1,
                step,
                num_layers,
                num_layers,
                tensor_count,
                max_seq_len,
                num_heads,
                model_dim,
            ) or any(header[9:]):
                raise RuntimeError(f"Native gpt2_diff {key} header is invalid.")

        return validate

    def validate_diff_parameters(chunk: bytes, _element_index: int) -> None:
        if any(
            not math.isfinite(value)
            for (value,) in struct.iter_unpack("<f", chunk)
        ):
            raise RuntimeError("Native gpt2_diff learned differential state is invalid.")

    def validate_diff_optimizer(chunk: bytes, element_index: int) -> None:
        for offset, (value,) in enumerate(struct.iter_unpack("<f", chunk)):
            if not math.isfinite(value) or (
                element_index + offset >= num_layers and value < 0.0
            ):
                raise RuntimeError("Native gpt2_diff learned differential state is invalid.")

    require_artifact_digest(
        "diff_parameters",
        _stream_regular_artifact_nofollow(
            checkpoint_root / artifacts["diff_parameters"][0],
            expected_bytes=expected_sizes["diff_parameters"],
            header_bytes=16 * 8,
            payload_element_size=4,
            validate_header=diff_state_header_validator(
                "diff_parameters", magic=20260808, tensor_count=1
            ),
            validate_payload=validate_diff_parameters,
        ),
    )
    require_artifact_digest(
        "diff_optimizer",
        _stream_regular_artifact_nofollow(
            checkpoint_root / artifacts["diff_optimizer"][0],
            expected_bytes=expected_sizes["diff_optimizer"],
            header_bytes=16 * 8,
            payload_element_size=4,
            validate_header=diff_state_header_validator(
                "diff_optimizer", magic=20260809, tensor_count=2
            ),
            validate_payload=validate_diff_optimizer,
        ),
    )

    expected_done = f"DONE_{step_text}"
    if metadata["done_marker"] != expected_done:
        raise RuntimeError("Native gpt2_diff DONE marker name is not canonical.")
    if _read_regular_nofollow(checkpoint_root / expected_done):
        raise RuntimeError("Native gpt2_diff DONE marker must be empty.")
    return metadata_path.resolve()


def _discover_checkpoint(
    checkpoint_dir: Path,
    *,
    expected_gpt2_diff_graph_sha256: str | None = None,
    expected_gpt2_diff_proof_contract_sha256: str | None = None,
    expected_gpt2_diff_geometry: Mapping[str, Any] | None = None,
    expected_llama_graph_sha256: str | None = None,
    expected_standard_moe_graph_sha256: str | None = None,
    expected_moa_graph_sha256: str | None = None,
    moa_source_graph_path: Path | None = None,
    moa_model: Mapping[str, Any] | None = None,
) -> Path:
    if not checkpoint_dir.is_dir():
        raise RuntimeError(
            f"Native trainer completed without creating its checkpoint directory: {checkpoint_dir}"
        )
    checkpoint_root = checkpoint_dir.resolve()
    if expected_gpt2_diff_graph_sha256 is not None:
        if (
            not _is_sha256(expected_gpt2_diff_graph_sha256)
            or not _is_sha256(expected_gpt2_diff_proof_contract_sha256)
            or not isinstance(expected_gpt2_diff_geometry, Mapping)
        ):
            raise RuntimeError(
                "Native gpt2_diff checkpoint discovery requires canonical prepared graph and proof digests."
            )
        diff_candidates: list[Path] = []
        for path in sorted(checkpoint_dir.glob("model_????????.diff.json")):
            try:
                diff_candidates.append(
                    _inspect_gpt2_diff_checkpoint(
                        path,
                        checkpoint_root=checkpoint_root,
                        expected_source_graph_sha256=expected_gpt2_diff_graph_sha256,
                        expected_proof_contract_sha256=(
                            expected_gpt2_diff_proof_contract_sha256
                        ),
                        expected_geometry=expected_gpt2_diff_geometry,
                    )
                )
            except (OSError, RuntimeError) as exc:
                raise RuntimeError(
                    f"Native trainer produced an invalid graph-bound gpt2_diff checkpoint at {path}: {exc}"
                ) from exc
        if diff_candidates:
            return max(
                diff_candidates,
                key=lambda path: (path.stat().st_mtime_ns, path.name),
            )
        raise RuntimeError(
            "Native gpt2_diff trainer completed without a strict model_XXXXXXXX.diff.json checkpoint bundle."
        )
    if expected_moa_graph_sha256 is not None:
        if moa_source_graph_path is None or moa_model is None:
            raise RuntimeError(
                "Native MoA checkpoint discovery requires the prepared source graph and model contract."
            )
        moa_candidates: list[Path] = []
        for path in sorted(checkpoint_dir.rglob("model_????????.moa.json")):
            if path.is_symlink() or not path.is_file():
                continue
            resolved = path.resolve()
            try:
                resolved.relative_to(checkpoint_root)
            except ValueError:
                continue
            try:
                checkpoint = inspect_native_moa_checkpoint(
                    resolved,
                    source_graph_path=moa_source_graph_path,
                    model=moa_model,
                )
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Native trainer produced an invalid graph-bound MoA checkpoint at {resolved}: {exc}"
                ) from exc
            if checkpoint.metadata_path != resolved:
                raise RuntimeError(
                    "Native MoA checkpoint inspection returned a different metadata path."
                )
            if checkpoint.source_graph_sha256 != expected_moa_graph_sha256:
                raise RuntimeError(
                    "Native MoA checkpoint source graph does not match the prepared training plan fingerprint."
                )
            moa_candidates.append(resolved)
        if moa_candidates:
            return max(moa_candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
        raise RuntimeError(
            "Native MoA trainer completed without a strict model_XXXXXXXX.moa.json checkpoint."
        )

    family_candidates: list[Path] = []
    for path in sorted(checkpoint_dir.rglob("*_native_family_model_00000000.json")):
        if path.is_symlink() or not path.is_file():
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(checkpoint_root)
        except ValueError:
            continue
        try:
            raw_metadata = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Native trainer produced invalid family checkpoint metadata at {resolved}: {exc}"
            ) from exc
        model_family = (
            str(raw_metadata.get("model_family") or "").strip().lower().replace("_", "-")
            if isinstance(raw_metadata, Mapping)
            else ""
        )
        if not model_family:
            if resolved.name.startswith("llama_"):
                model_family = "llama"
            elif resolved.name.startswith("mixllama_"):
                model_family = "mixllama"
        try:
            if model_family == "llama":
                expected_graph_sha256 = expected_llama_graph_sha256
                profile_label = "canonical LLaMA v2"
                checkpoint = inspect_native_family_llama_checkpoint(resolved)
            elif model_family == "mixllama":
                expected_graph_sha256 = expected_standard_moe_graph_sha256
                profile_label = "canonical standard-MoE v1"
                checkpoint = inspect_native_family_standard_moe_checkpoint(resolved)
            else:
                continue
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Native trainer produced an invalid {profile_label} checkpoint at {resolved}: {exc}"
            ) from exc
        if checkpoint.metadata_path != resolved:
            raise RuntimeError(
                "Native family checkpoint inspection returned a different metadata path."
            )
        if expected_graph_sha256 is not None:
            source_graph = checkpoint.training.get("source_graph")
            actual_sha256 = (
                source_graph.get("sha256") if isinstance(source_graph, Mapping) else None
            )
            if actual_sha256 != expected_graph_sha256:
                raise RuntimeError(
                    "Native family checkpoint source graph does not match the "
                    "prepared training plan fingerprint."
                )
        family_candidates.append(resolved)
    if family_candidates:
        preferred = [
            path
            for path in family_candidates
            if path.name
            in {
                "llama_native_family_model_00000000.json",
                "mixllama_native_family_model_00000000.json",
            }
        ]
        pool = preferred or family_candidates
        return max(pool, key=lambda path: (path.stat().st_mtime_ns, path.name))

    candidates: list[Path] = []
    for path in checkpoint_dir.rglob("*.bin"):
        if path.is_symlink() or not path.is_file():
            continue
        if not (
            path.name == "model.bin"
            or re.fullmatch(r"model_[0-9]+\.bin", path.name)
            or re.fullmatch(r"checkpoint[_-]?[0-9]*\.bin", path.name)
        ):
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(checkpoint_root)
        except ValueError:
            continue
        if resolved.stat().st_size > 0:
            candidates.append(resolved)
    if not candidates:
        raise RuntimeError(
            "Native trainer completed without a persisted .bin checkpoint or canonical "
            f"LLaMA v2 or standard-MoE v1 metadata checkpoint under {checkpoint_dir}"
        )
    preferred = [path for path in candidates if path.name == "model.bin"]
    pool = preferred or candidates
    return max(pool, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _prepared_gpt2_diff_proof_contract(
    plan: NativeGraphTrainPlan,
) -> tuple[str, Mapping[str, Any]]:
    proof_path = plan.graph_preflight_proof
    graph_sha256 = plan.compatibility_report.graph_fingerprint
    if proof_path is None or not _is_sha256(graph_sha256):
        raise RuntimeError(
            "Native gpt2_diff execution requires its materialized graph preflight proof."
        )
    raw = _read_regular_nofollow(proof_path, max_bytes=1024 * 1024)
    envelope = _strict_json_object(raw, label="gpt2_diff graph preflight proof")
    if set(envelope) != {"contract", "contract_sha256"}:
        raise RuntimeError("Native gpt2_diff graph preflight proof envelope is not canonical.")
    contract = envelope["contract"]
    contract_sha256 = envelope["contract_sha256"]
    if not isinstance(contract, dict) or not _is_sha256(contract_sha256):
        raise RuntimeError("Native gpt2_diff graph preflight proof contract is invalid.")
    contract_bytes = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    canonical_envelope = (
        b'{"contract":'
        + contract_bytes
        + b',"contract_sha256":"'
        + contract_sha256.encode("ascii")
        + b'"}\n'
    )
    if raw != canonical_envelope or hashlib.sha256(contract_bytes).hexdigest() != contract_sha256:
        raise RuntimeError("Native gpt2_diff graph preflight proof bytes are not canonical.")
    if (
        contract.get("schema") != "neuralfn.native_graph_training_proof"
        or type(contract.get("version")) is not int
        or contract["version"] != 1
        or contract.get("training_selector") != "gpt2_diff"
        or contract.get("source_graph_sha256") != graph_sha256
        or contract.get("passed") is not True
        or contract_sha256
        != plan.artifact_metadata.get("graph_preflight_proof_contract_sha256")
    ):
        raise RuntimeError("Native gpt2_diff graph preflight proof does not match its plan.")
    geometry = contract.get("geometry")
    if not isinstance(geometry, dict):
        raise RuntimeError("Native gpt2_diff graph preflight proof geometry is missing.")
    return contract_sha256, geometry


def execute_native_training(preparation: NativeTrainingPreparation) -> Path:
    """Run the canonical native trainer and return its verified checkpoint path."""

    return_code = run_native_train(preparation.config, runner="auto")
    if return_code != 0:
        raise RuntimeError(f"Native trainer exited with status {return_code}.")
    expected_llama_graph_sha256 = (
        preparation.plan.compatibility_report.graph_fingerprint
        if preparation.plan.trainer_family == "llama"
        else None
    )
    expected_standard_moe_graph_sha256 = (
        preparation.plan.compatibility_report.graph_fingerprint
        if preparation.plan.trainer_family == "mixllama"
        else None
    )
    expected_moa_graph_sha256 = (
        preparation.plan.compatibility_report.graph_fingerprint
        if preparation.plan.training_selector == "gpt2_moa"
        else None
    )
    expected_gpt2_diff_graph_sha256 = (
        preparation.plan.compatibility_report.graph_fingerprint
        if preparation.plan.training_selector == "gpt2_diff"
        else None
    )
    expected_gpt2_diff_proof_contract_sha256: str | None = None
    expected_gpt2_diff_geometry: Mapping[str, Any] | None = None
    if expected_gpt2_diff_graph_sha256 is not None:
        (
            expected_gpt2_diff_proof_contract_sha256,
            expected_gpt2_diff_geometry,
        ) = _prepared_gpt2_diff_proof_contract(preparation.plan)
    return _discover_checkpoint(
        preparation.checkpoint_dir,
        expected_gpt2_diff_graph_sha256=expected_gpt2_diff_graph_sha256,
        expected_gpt2_diff_proof_contract_sha256=(
            expected_gpt2_diff_proof_contract_sha256
        ),
        expected_gpt2_diff_geometry=expected_gpt2_diff_geometry,
        expected_llama_graph_sha256=expected_llama_graph_sha256,
        expected_standard_moe_graph_sha256=expected_standard_moe_graph_sha256,
        expected_moa_graph_sha256=expected_moa_graph_sha256,
        moa_source_graph_path=(
            preparation.plan.launch_graph
            if expected_moa_graph_sha256 is not None
            else None
        ),
        moa_model=(
            preparation.plan.manifest.model
            if expected_moa_graph_sha256 is not None
            and preparation.plan.manifest is not None
            else None
        ),
    )


__all__ = [
    "NATIVE_RUNTIME",
    "NativeTrainingIncompatibleError",
    "NativeTrainingPreparation",
    "execute_native_training",
    "native_training_metadata",
    "preflight_native_training",
    "prepare_native_training_run",
]
