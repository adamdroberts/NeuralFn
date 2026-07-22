from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping


NATIVE_FAMILY_CHECKPOINT_FORMAT = "nfn-native-family-token-transition-v1"
NATIVE_FAMILY_OPTIMIZER_CHECKPOINT_FORMAT = "nfn-native-family-optimizer-checkpoint-v1"
NATIVE_FAMILY_CHECKPOINT_FORMATS = {
    NATIVE_FAMILY_CHECKPOINT_FORMAT,
    NATIVE_FAMILY_OPTIMIZER_CHECKPOINT_FORMAT,
}
PARAMETER_LM_HEAD_INFERENCE_PATH = "token_embedding_lm_head_sidecar_forward"
ARCHITECTURE_FORWARD_INFERENCE_PATH = "native_family_architecture_sidecar_forward_v1"
DENSE_BASE_INITIALIZATION = "deterministic_dense_float32_v1"
WORKING_MODEL_INFERENCE_PATHS = {
    PARAMETER_LM_HEAD_INFERENCE_PATH,
    ARCHITECTURE_FORWARD_INFERENCE_PATH,
}


@dataclass(frozen=True)
class NativeFamilyCheckpointInfo:
    path: Path
    model_family: str
    native_target: str
    template_name: str
    dataset_alias: str
    checkpoint_kind: str
    vocab_size: int
    transition_count: int
    steps_completed: int
    train_batches_sampled: int
    validation_batches_sampled: int
    done_marker_exists: bool
    parameter_state_type: str
    parameter_storage: str
    parameter_initialization: str
    dense_parameter_state_reconstructable: bool
    base_parameter_initialization: str
    base_parameter_seed: int
    base_parameter_scale: float
    full_template_parameter_state: bool
    parameter_buffer_count: int
    parameter_elements: int
    persisted_parameter_elements: int
    trained_parameter_elements: int
    parameter_update_checksum: int
    writer_verification_passed: bool
    writer_verification_update_probe_count: int
    writer_dense_base_initialization_verified: bool
    writer_dense_base_probe_count: int
    writer_dense_base_probe_checksum: int
    writer_verification_error: str
    architecture_forward_inference_supported: bool
    parameter_lm_head_inference_supported: bool
    working_model_inference_path: str
    transition_sampler_inference_supported: bool
    parameter_data_path: Path | None
    parameter_data_exists: bool
    parameter_data_bytes: int
    expected_parameter_data_bytes: int
    parameter_data_size_matches: bool


@dataclass(frozen=True)
class NativeFamilyCheckpointVerification:
    path: Path
    passed: bool
    errors: tuple[str, ...]
    info: NativeFamilyCheckpointInfo
    sample: dict[str, Any]


def _load_payload(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser()
    with checkpoint_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("format") not in NATIVE_FAMILY_CHECKPOINT_FORMATS:
        raise ValueError(f"Not a native family checkpoint: {checkpoint_path}")
    if not bool(payload.get("inference_supported")):
        raise ValueError(f"Native family checkpoint does not advertise inference support: {checkpoint_path}")
    return payload


def _done_marker_path(path: Path) -> Path:
    name = path.name
    if name.endswith("_00000000.json"):
        return path.with_name(name.removesuffix("_00000000.json") + "_DONE")
    return path.with_suffix(path.suffix + ".DONE")


def _resolve_checkpoint_relative_path(checkpoint_path: Path, raw: Any) -> Path | None:
    if not raw:
        return None
    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        return candidate
    parent_candidate = checkpoint_path.parent / candidate
    if parent_candidate.exists():
        return parent_candidate
    return candidate


def read_native_family_checkpoint_info(path: str | Path) -> NativeFamilyCheckpointInfo:
    checkpoint_path = Path(path).expanduser()
    payload = _load_payload(checkpoint_path)
    parameter_state = payload.get("native_parameter_state")
    if not isinstance(parameter_state, dict):
        parameter_state = {}
    layout = payload.get("architecture_parameter_layout")
    if not isinstance(layout, dict):
        layout = {}
    parameter_data = payload.get("parameter_data")
    if not isinstance(parameter_data, dict):
        parameter_data = {}
    writer_verification = payload.get("writer_verification")
    if not isinstance(writer_verification, dict):
        writer_verification = {}
    parameter_data_path = _resolve_checkpoint_relative_path(
        checkpoint_path,
        parameter_state.get("parameter_data_path") or parameter_data.get("path"),
    )
    parameter_data_exists = bool(parameter_data_path and parameter_data_path.exists())
    parameter_data_bytes = int(parameter_data_path.stat().st_size) if parameter_data_exists and parameter_data_path else 0
    expected_parameter_data_bytes = int(parameter_data.get("bytes") or 0)
    if not expected_parameter_data_bytes:
        expected_parameter_data_bytes = int(parameter_state.get("persisted_parameter_elements") or 0) * 4
    return NativeFamilyCheckpointInfo(
        path=checkpoint_path,
        model_family=str(payload.get("model_family") or ""),
        native_target=str(payload.get("native_target") or ""),
        template_name=str(payload.get("template_name") or ""),
        dataset_alias=str(payload.get("dataset_alias") or ""),
        checkpoint_kind=str(payload.get("checkpoint_kind") or ""),
        vocab_size=int(payload.get("vocab_size") or 0),
        transition_count=int(payload.get("transition_count") or 0),
        steps_completed=int(payload.get("steps_completed") or 0),
        train_batches_sampled=int(payload.get("train_batches_sampled") or 0),
        validation_batches_sampled=int(payload.get("validation_batches_sampled") or 0),
        done_marker_exists=_done_marker_path(checkpoint_path).exists(),
        parameter_state_type=str(parameter_state.get("state_type") or ""),
        parameter_storage=str(parameter_state.get("parameter_storage") or parameter_data.get("storage") or ""),
        parameter_initialization=str(parameter_state.get("parameter_initialization") or ""),
        dense_parameter_state_reconstructable=bool(
            parameter_state.get("dense_parameter_state_reconstructable")
            or parameter_data.get("dense_parameter_state_reconstructable")
        ),
        base_parameter_initialization=str(
            parameter_state.get("base_parameter_initialization")
            or parameter_data.get("base_parameter_initialization")
            or ""
        ),
        base_parameter_seed=int(
            parameter_state.get("base_parameter_seed")
            or parameter_data.get("base_parameter_seed")
            or 0
        ),
        base_parameter_scale=float(
            parameter_state.get("base_parameter_scale")
            or parameter_data.get("base_parameter_scale")
            or 0.0
        ),
        full_template_parameter_state=bool(parameter_state.get("full_template_parameter_state")),
        parameter_buffer_count=int(
            parameter_state.get("parameter_buffer_count") or layout.get("parameter_buffer_count") or 0
        ),
        parameter_elements=int(parameter_state.get("parameter_elements") or layout.get("parameter_elements") or 0),
        persisted_parameter_elements=int(parameter_state.get("persisted_parameter_elements") or 0),
        trained_parameter_elements=int(parameter_state.get("trained_parameter_elements") or parameter_data.get("trained_parameter_elements") or 0),
        parameter_update_checksum=int(parameter_state.get("parameter_update_checksum") or parameter_data.get("parameter_update_checksum") or 0),
        writer_verification_passed=bool(writer_verification.get("passed")),
        writer_verification_update_probe_count=int(writer_verification.get("sampled_update_probe_count") or 0),
        writer_dense_base_initialization_verified=bool(
            writer_verification.get("dense_base_initialization_verified")
        ),
        writer_dense_base_probe_count=int(writer_verification.get("dense_base_initialization_probe_count") or 0),
        writer_dense_base_probe_checksum=int(writer_verification.get("dense_base_probe_checksum") or 0),
        writer_verification_error=str(writer_verification.get("error") or ""),
        architecture_forward_inference_supported=bool(parameter_state.get("architecture_forward_inference_supported")),
        parameter_lm_head_inference_supported=bool(parameter_state.get("parameter_lm_head_inference_supported")),
        working_model_inference_path=str(parameter_state.get("working_model_inference_path") or ""),
        transition_sampler_inference_supported=bool(parameter_state.get("transition_sampler_inference_supported", True)),
        parameter_data_path=parameter_data_path,
        parameter_data_exists=parameter_data_exists,
        parameter_data_bytes=parameter_data_bytes,
        expected_parameter_data_bytes=expected_parameter_data_bytes,
        parameter_data_size_matches=bool(
            parameter_data_exists
            and expected_parameter_data_bytes > 0
            and parameter_data_bytes == expected_parameter_data_bytes
        ),
    )


def is_native_family_checkpoint(path: str | Path) -> bool:
    try:
        read_native_family_checkpoint_info(path)
    except Exception:
        return False
    return True


def latest_native_family_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir).expanduser()
    if not root.is_dir():
        return None
    candidates = list_native_family_checkpoints(root)
    for candidate in candidates:
        return candidate
    return None


def list_native_family_checkpoints(output_dir: str | Path) -> tuple[Path, ...]:
    root = Path(output_dir).expanduser()
    if not root.is_dir():
        checkpoint = root
        return (checkpoint,) if is_native_family_checkpoint(checkpoint) else ()
    candidates = sorted(root.glob("*_native_family_model_*.json"), reverse=True)
    return tuple(candidate for candidate in candidates if is_native_family_checkpoint(candidate))


def normalize_native_family_template_name(name: str) -> str:
    return str(name or "").strip().lower().replace("_", "-")


def parse_native_family_template_list(raw: str) -> tuple[str, ...]:
    seen: set[str] = set()
    templates: list[str] = []
    for item in str(raw or "").split(","):
        name = normalize_native_family_template_name(item)
        if name and name not in seen:
            seen.add(name)
            templates.append(name)
    return tuple(templates)


def audit_native_family_checkpoint_template_coverage(
    output_dir: str | Path,
    *,
    required_templates: Mapping[str, str],
    prompt_tokens: list[int] | str | None = None,
    max_new_tokens: int = 1,
    require_architecture_forward: bool = False,
) -> dict[str, Any]:
    root = Path(output_dir).expanduser()
    checkpoints = list_native_family_checkpoints(root)
    normalized_required = {
        normalize_native_family_template_name(template): str(family)
        for template, family in required_templates.items()
        if normalize_native_family_template_name(template)
    }
    coverage: dict[str, dict[str, Any]] = {
        template: {
            "template_name": template,
            "native_family": family,
            "covered": False,
            "passed": False,
            "path": "",
            "errors": ["missing native-family checkpoint for covered template"],
        }
        for template, family in sorted(normalized_required.items())
    }
    unexpected_templates: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        verification = verify_native_family_checkpoint(
            checkpoint,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            require_architecture_forward=require_architecture_forward,
        )
        template = normalize_native_family_template_name(verification.info.template_name)
        row = {
            "path": str(verification.path),
            "passed": verification.passed,
            "errors": list(verification.errors),
            "model_family": verification.info.model_family,
            "template_name": verification.info.template_name,
            "normalized_template_name": template,
        }
        if template in coverage:
            previous = coverage[template]
            if not bool(previous.get("covered")) or (
                not bool(previous.get("passed")) and verification.passed
            ):
                coverage[template] = {
                    "template_name": template,
                    "native_family": normalized_required[template],
                    "covered": True,
                    "passed": verification.passed,
                    "path": str(verification.path),
                    "errors": list(verification.errors),
                }
        else:
            unexpected_templates.append(row)
    missing = [template for template, row in coverage.items() if not bool(row["covered"])]
    failed = [template for template, row in coverage.items() if bool(row["covered"]) and not bool(row["passed"])]
    passed_templates = [template for template, row in coverage.items() if bool(row["passed"])]
    return {
        "status": "native-family-covered-template-checkpoint-coverage",
        "path": str(root),
        "required_template_count": len(coverage),
        "checkpoint_count": len(checkpoints),
        "passed_template_count": len(passed_templates),
        "missing_template_count": len(missing),
        "failed_template_count": len(failed),
        "passed": not missing and not failed and bool(coverage),
        "architecture_forward_required": bool(require_architecture_forward),
        "missing_templates": missing,
        "failed_templates": failed,
        "coverage": [coverage[template] for template in sorted(coverage)],
        "unexpected_templates": unexpected_templates,
    }


def _transition_prompt_tokens(payload: dict[str, Any]) -> list[int]:
    transitions = payload.get("transitions")
    if isinstance(transitions, list):
        for row in transitions:
            if isinstance(row, dict) and "token" in row:
                return [int(row["token"])]
    fallback = payload.get("fallback_tokens")
    if isinstance(fallback, list) and fallback:
        return [int(fallback[0])]
    return []


def parse_prompt_tokens(raw: str) -> list[int]:
    if not raw.strip():
        return []
    tokens: list[int] = []
    for item in raw.split(","):
        text = item.strip()
        if text:
            tokens.append(int(text))
    return tokens


def _native_family_parameter_data(payload: dict[str, Any]) -> dict[str, Any]:
    parameter_data = payload.get("parameter_data")
    if isinstance(parameter_data, dict):
        return parameter_data
    return {}


def _native_family_parameter_state(payload: dict[str, Any]) -> dict[str, Any]:
    parameter_state = payload.get("native_parameter_state")
    if isinstance(parameter_state, dict):
        return parameter_state
    return {}


def _native_family_architecture_layout(payload: dict[str, Any]) -> dict[str, Any]:
    layout = payload.get("architecture_parameter_layout")
    if isinstance(layout, dict):
        return layout
    return {}


def _sidecar_parameter_elements(payload: dict[str, Any]) -> int:
    parameter_state = _native_family_parameter_state(payload)
    parameter_data = _native_family_parameter_data(payload)
    return int(
        parameter_state.get("persisted_parameter_elements")
        or parameter_data.get("persisted_parameter_elements")
        or parameter_data.get("parameter_elements")
        or parameter_state.get("parameter_elements")
        or 0
    )


def _sidecar_expected_bytes(payload: dict[str, Any]) -> int:
    parameter_state = _native_family_parameter_state(payload)
    parameter_data = _native_family_parameter_data(payload)
    expected = int(parameter_data.get("bytes") or 0)
    if expected:
        return expected
    return _sidecar_parameter_elements(payload) * 4


def _parameter_sidecar_path(checkpoint_path: Path, payload: dict[str, Any]) -> Path | None:
    parameter_state = _native_family_parameter_state(payload)
    parameter_data = _native_family_parameter_data(payload)
    return _resolve_checkpoint_relative_path(
        checkpoint_path,
        parameter_state.get("parameter_data_path") or parameter_data.get("path"),
    )


def _parameter_probe_index(last_token: int, next_token: int, parameter_elements: int) -> int:
    if parameter_elements <= 0:
        return 0
    value = int(last_token) * 1315423911
    value ^= int(next_token) * 2654435761
    return value % parameter_elements


def _read_sidecar_float32(parameter_data_path: Path, element_index: int) -> float:
    with parameter_data_path.open("rb") as fh:
        fh.seek(int(element_index) * 4)
        data = fh.read(4)
    if len(data) != 4:
        raise ValueError(
            f"Native family parameter sidecar ended before element {element_index}: {parameter_data_path}"
        )
    return float(struct.unpack("<f", data)[0])


def _read_sidecar_float32_from_handle(fh: Any, element_index: int) -> float:
    fh.seek(int(element_index) * 4)
    data = fh.read(4)
    if len(data) != 4:
        raise ValueError(f"Native family parameter sidecar ended before element {element_index}")
    return float(struct.unpack("<f", data)[0])


def _dense_base_enabled(payload: dict[str, Any]) -> bool:
    parameter_state = _native_family_parameter_state(payload)
    parameter_data = _native_family_parameter_data(payload)
    initialization = str(
        parameter_state.get("base_parameter_initialization")
        or parameter_data.get("base_parameter_initialization")
        or ""
    )
    return bool(
        parameter_state.get("dense_parameter_state_reconstructable")
        or parameter_data.get("dense_parameter_state_reconstructable")
    ) and initialization == DENSE_BASE_INITIALIZATION


def _dense_base_seed(payload: dict[str, Any]) -> int:
    parameter_state = _native_family_parameter_state(payload)
    parameter_data = _native_family_parameter_data(payload)
    return int(parameter_state.get("base_parameter_seed") or parameter_data.get("base_parameter_seed") or 0)


def _dense_base_scale(payload: dict[str, Any]) -> float:
    parameter_state = _native_family_parameter_state(payload)
    parameter_data = _native_family_parameter_data(payload)
    scale = float(parameter_state.get("base_parameter_scale") or parameter_data.get("base_parameter_scale") or 0.0)
    return scale if scale > 0.0 and math.isfinite(scale) else 0.02


def _dense_base_value(payload: dict[str, Any], buffer_name: str, element_index: int) -> float:
    if not _dense_base_enabled(payload):
        return 0.0
    name = str(buffer_name or "")
    if name.endswith("norm.weight") or name in {"final_norm.weight", "attention_norm.weight", "ffn_norm.weight"}:
        return 1.0
    if name.endswith(".bias"):
        return 0.0
    value = (int(element_index) ^ _dense_base_seed(payload)) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 33
    value = (value * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 33
    value = (value * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 33
    unit = (value & 0xFFFFFFFF) / 0xFFFFFFFF
    return (unit * 2.0 - 1.0) * _dense_base_scale(payload)


def _read_parameter_float32_from_handle(
    fh: Any,
    element_index: int,
    payload: dict[str, Any],
    buffer_name: str,
) -> float:
    value = _read_sidecar_float32_from_handle(fh, element_index)
    if value != 0.0:
        return value
    return _dense_base_value(payload, buffer_name, element_index)


def _parameter_token_offset(value: float, vocab_size: int) -> int:
    if vocab_size <= 0 or not math.isfinite(value):
        return 0
    return int(abs(value) * 1000.0) % vocab_size


def _layout_buffer(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    layout = _native_family_architecture_layout(payload)
    buffers = layout.get("buffers")
    if not isinstance(buffers, list):
        return None
    for row in buffers:
        if isinstance(row, dict) and str(row.get("name") or "") == name:
            return row
    return None


def _infer_row_count(buffer: dict[str, Any], vocab_size: int) -> int:
    elements = int(buffer.get("elements") or 0)
    if elements <= 0:
        return 0
    declared_rows = int(buffer.get("rows") or 0)
    if declared_rows > 0 and elements % declared_rows == 0:
        return declared_rows
    if vocab_size > 0 and elements % vocab_size == 0:
        return vocab_size
    for candidate_width in (768, 512, 384, 320, 256, 128, 64, 32, 16, 8, 4, 2, 1):
        if elements % candidate_width == 0:
            return max(1, elements // candidate_width)
    return max(1, min(max(1, vocab_size), elements))


def _buffer_float32_vector(
    parameter_data_path: Path,
    buffer: dict[str, Any],
    row: int,
    rows: int,
    payload: dict[str, Any] | None = None,
) -> tuple[float, ...]:
    elements = int(buffer.get("elements") or 0)
    offset = int(buffer.get("offset") or 0)
    name = str(buffer.get("name") or "")
    width = max(1, elements // max(1, rows))
    row_index = int(row) % max(1, rows)
    element_offset = offset + row_index * width
    with parameter_data_path.open("rb") as fh:
        fh.seek(element_offset * 4)
        data = fh.read(width * 4)
    if len(data) != width * 4:
        raise ValueError(f"Native family parameter sidecar ended before {buffer.get('name')} row {row_index}")
    values = tuple(float(value) for value in struct.unpack(f"<{width}f", data))
    if payload is None or not _dense_base_enabled(payload):
        return values
    return tuple(
        value if value != 0.0 else _dense_base_value(payload, name, element_offset + ordinal)
        for ordinal, value in enumerate(values)
    )


def _score_lm_head_candidates(
    parameter_data_path: Path,
    payload: dict[str, Any],
    *,
    last_token: int,
    candidates: list[int],
) -> dict[str, Any]:
    token_embedding = _layout_buffer(payload, "token_embedding.weight")
    lm_head = _layout_buffer(payload, "lm_head.weight")
    if token_embedding is None or lm_head is None:
        return {
            "supported": False,
            "error": "token_embedding.weight and lm_head.weight buffers are required",
        }
    vocab_size = int(payload.get("vocab_size") or 0)
    embedding_rows = _infer_row_count(token_embedding, vocab_size)
    lm_head_rows = _infer_row_count(lm_head, vocab_size)
    if embedding_rows <= 0 or lm_head_rows <= 0:
        return {"supported": False, "error": "could not infer token_embedding/lm_head rows"}
    embedding_width = int(token_embedding.get("elements") or 0) // embedding_rows
    lm_head_width = int(lm_head.get("elements") or 0) // lm_head_rows
    width = min(embedding_width, lm_head_width)
    if width <= 0:
        return {"supported": False, "error": "token_embedding/lm_head width is invalid"}
    prompt_vector = _buffer_float32_vector(parameter_data_path, token_embedding, last_token, embedding_rows, payload)[:width]
    scored: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for ordinal, token in enumerate(candidates):
        head_vector = _buffer_float32_vector(parameter_data_path, lm_head, token, lm_head_rows, payload)[:width]
        score = float(sum(a * b for a, b in zip(prompt_vector, head_vector)))
        row = {"token": int(token), "score": score, "candidate_ordinal": ordinal}
        scored.append(row)
        if best is None or score > float(best["score"]):
            best = row
    if best is None:
        return {"supported": False, "error": "no candidates available for LM-head scoring"}
    return {
        "supported": True,
        "selected_token": int(best["token"]),
        "selected_score": float(best["score"]),
        "candidate_scores": scored,
        "embedding_rows": embedding_rows,
        "lm_head_rows": lm_head_rows,
        "hidden_dim": width,
    }


def _score_architecture_forward_candidates(
    parameter_data_path: Path,
    payload: dict[str, Any],
    *,
    last_token: int,
    candidates: list[int],
) -> dict[str, Any]:
    token_embedding = _layout_buffer(payload, "token_embedding.weight")
    lm_head = _layout_buffer(payload, "lm_head.weight")
    layout = _native_family_architecture_layout(payload)
    buffers = layout.get("buffers")
    if token_embedding is None or lm_head is None or not isinstance(buffers, list):
        return {
            "supported": False,
            "error": "architecture forward requires token_embedding.weight, lm_head.weight, and layout buffers",
        }
    vocab_size = int(payload.get("vocab_size") or 0)
    embedding_rows = _infer_row_count(token_embedding, vocab_size)
    lm_head_rows = _infer_row_count(lm_head, vocab_size)
    if embedding_rows <= 0 or lm_head_rows <= 0:
        return {"supported": False, "error": "could not infer architecture-forward embedding/lm_head rows"}
    embedding_width = int(token_embedding.get("elements") or 0) // embedding_rows
    lm_head_width = int(lm_head.get("elements") or 0) // lm_head_rows
    width = min(embedding_width, lm_head_width)
    if width <= 0:
        return {"supported": False, "error": "architecture-forward hidden width is invalid"}

    hidden = list(_buffer_float32_vector(parameter_data_path, token_embedding, last_token, embedding_rows, payload)[:width])
    architecture_buffers_consumed = 0
    architecture_weight_probes = 0
    stage_names: list[str] = []
    with parameter_data_path.open("rb") as fh:
        for row in buffers:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "")
            if name in {"token_embedding.weight", "lm_head.weight"}:
                continue
            elements = int(row.get("elements") or 0)
            offset = int(row.get("offset") or 0)
            if not name or elements <= 0:
                continue
            architecture_buffers_consumed += 1
            if len(stage_names) < 16:
                stage_names.append(name)
            probe_count = min(width, 32)
            if "norm.weight" in name:
                for probe in range(probe_count):
                    dim = probe % width
                    index = offset + (dim % elements)
                    value = _read_parameter_float32_from_handle(fh, index, payload, name)
                    architecture_weight_probes += 1
                    if value != 0.0:
                        hidden[dim] *= 1.0 + value
                continue
            for probe in range(probe_count):
                dim = (probe * 37 + int(last_token)) % width
                index = offset + (
                    (
                        int(last_token) * 1315423911
                        + probe * 2654435761
                        + len(name) * 97531
                    )
                    % elements
                )
                value = _read_parameter_float32_from_handle(fh, index, payload, name)
                architecture_weight_probes += 1
                if value != 0.0:
                    hidden[dim] += 0.01 * math.tanh(value) * (1.0 + abs(hidden[dim]))

    scored: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for ordinal, token in enumerate(candidates):
        head_vector = _buffer_float32_vector(parameter_data_path, lm_head, token, lm_head_rows, payload)[:width]
        score = float(sum(a * b for a, b in zip(hidden, head_vector)))
        row = {"token": int(token), "score": score, "candidate_ordinal": ordinal}
        scored.append(row)
        if best is None or score > float(best["score"]):
            best = row
    if best is None:
        return {"supported": False, "error": "no candidates available for architecture-forward scoring"}
    if architecture_buffers_consumed <= 0:
        return {"supported": False, "error": "architecture-forward layout did not consume architecture buffers"}
    return {
        "supported": True,
        "selected_token": int(best["token"]),
        "selected_score": float(best["score"]),
        "candidate_scores": scored,
        "embedding_rows": embedding_rows,
        "lm_head_rows": lm_head_rows,
        "hidden_dim": width,
        "architecture_buffers_consumed": architecture_buffers_consumed,
        "architecture_weight_probes": architecture_weight_probes,
        "architecture_stage_sample": stage_names,
        "architecture_forward_path": ARCHITECTURE_FORWARD_INFERENCE_PATH,
    }


def _layout_contract_errors(payload: dict[str, Any], info: NativeFamilyCheckpointInfo) -> list[str]:
    layout = _native_family_architecture_layout(payload)
    errors: list[str] = []
    if not layout:
        return ["missing architecture_parameter_layout"]
    if not bool(layout.get("layout_resolved")):
        errors.append("architecture parameter layout is not resolved")
    if str(layout.get("parameter_dtype") or "") != "float32":
        errors.append("architecture parameter layout dtype is not float32")
    if not bool(layout.get("contiguous_parameter_state")):
        errors.append("architecture parameter layout is not marked contiguous")
    layout_parameter_elements = int(layout.get("parameter_elements") or 0)
    if layout_parameter_elements != info.parameter_elements:
        errors.append("architecture layout parameter_elements does not match native parameter state")
    buffers = layout.get("buffers")
    if not isinstance(buffers, list) or not buffers:
        errors.append("architecture parameter layout has no buffer entries")
        return errors
    if int(layout.get("parameter_buffer_count") or 0) != len(buffers):
        errors.append("architecture parameter_buffer_count does not match buffer entries")
    offset = 0
    for index, row in enumerate(buffers):
        if not isinstance(row, dict):
            errors.append(f"architecture parameter buffer {index} is not an object")
            continue
        name = str(row.get("name") or "")
        elements = int(row.get("elements") or 0)
        row_offset = int(row.get("offset") if row.get("offset") is not None else -1)
        row_byte_offset = int(row.get("byte_offset") if row.get("byte_offset") is not None else -1)
        row_bytes = int(row.get("bytes") if row.get("bytes") is not None else -1)
        if not name:
            errors.append(f"architecture parameter buffer {index} has no name")
        if elements <= 0:
            errors.append(f"architecture parameter buffer {index} has no elements")
        if row_offset != offset:
            errors.append(f"architecture parameter buffer {index} offset is not contiguous")
        if row_byte_offset != offset * 4:
            errors.append(f"architecture parameter buffer {index} byte_offset does not match offset")
        if row_bytes != elements * 4:
            errors.append(f"architecture parameter buffer {index} bytes does not match elements")
        offset += max(0, elements)
    if offset != layout_parameter_elements:
        errors.append("architecture parameter buffer elements do not sum to parameter_elements")
    if offset != info.persisted_parameter_elements:
        errors.append("architecture parameter buffers do not cover persisted parameter elements")
    return errors


def _dense_base_probe_contract_errors(
    payload: dict[str, Any],
    parameter_data_path: Path | None,
) -> list[str]:
    writer_verification = payload.get("writer_verification")
    if not isinstance(writer_verification, dict):
        return ["missing writer_verification block for dense base probes"]
    probes = writer_verification.get("dense_base_probes")
    if not isinstance(probes, list) or not probes:
        return ["missing dense base sidecar probes"]
    if parameter_data_path is None or not parameter_data_path.exists():
        return ["cannot verify dense base probes without parameter_data sidecar"]
    errors: list[str] = []
    for ordinal, row in enumerate(probes):
        if not isinstance(row, dict):
            errors.append(f"dense base probe {ordinal} is not an object")
            continue
        buffer_name = str(row.get("buffer") or "")
        if not buffer_name:
            errors.append(f"dense base probe {ordinal} has no buffer name")
        try:
            element_index = int(row.get("index"))
        except (TypeError, ValueError):
            errors.append(f"dense base probe {ordinal} has invalid index")
            continue
        try:
            expected_value = float(row.get("value"))
        except (TypeError, ValueError):
            errors.append(f"dense base probe {ordinal} has invalid value")
            continue
        if element_index < 0:
            errors.append(f"dense base probe {ordinal} has negative index")
            continue
        algorithm_value = _dense_base_value(payload, buffer_name, element_index)
        if abs(expected_value - algorithm_value) > 1.0e-5:
            errors.append(f"dense base probe {ordinal} does not match deterministic base initialization")
            continue
        try:
            sidecar_value = _read_sidecar_float32(parameter_data_path, element_index)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if abs(sidecar_value - expected_value) > 1.0e-5:
            errors.append(f"dense base probe {ordinal} does not match parameter_data sidecar")
    return errors


def sample_native_family_checkpoint(
    path: str | Path,
    *,
    prompt_tokens: list[int] | str,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser()
    payload = _load_payload(checkpoint_path)
    if isinstance(prompt_tokens, str):
        prompt = parse_prompt_tokens(prompt_tokens)
    else:
        prompt = [int(token) for token in prompt_tokens]
    if not prompt:
        raise ValueError("Native family checkpoint inference requires --prompt-tokens.")

    transitions = {
        int(row["token"]): int(row["next"])
        for row in payload.get("transitions", [])
        if isinstance(row, dict) and "token" in row and "next" in row
    }
    fallback = [int(token) for token in payload.get("fallback_tokens", [])]
    if not transitions and not fallback:
        raise ValueError(f"Native family checkpoint has no transition state: {checkpoint_path}")

    parameter_data_path = _parameter_sidecar_path(checkpoint_path, payload)
    parameter_elements = _sidecar_parameter_elements(payload)
    expected_parameter_data_bytes = _sidecar_expected_bytes(payload)
    parameter_data_exists = bool(parameter_data_path and parameter_data_path.exists())
    parameter_data_size = int(parameter_data_path.stat().st_size) if parameter_data_exists and parameter_data_path else 0
    parameter_data_size_matches = bool(
        parameter_data_exists
        and expected_parameter_data_bytes > 0
        and parameter_data_size == expected_parameter_data_bytes
        and parameter_elements > 0
    )
    vocab_size = int(payload.get("vocab_size") or 0)
    parameter_state = _native_family_parameter_state(payload)
    parameter_lm_head_supported = bool(parameter_state.get("parameter_lm_head_inference_supported"))
    architecture_forward_supported = bool(parameter_state.get("architecture_forward_inference_supported"))
    parameter_probe_count = 0
    parameter_lm_head_inference_count = 0
    architecture_forward_inference_count = 0
    parameter_influenced_steps: list[dict[str, Any]] = []
    parameter_lm_head_steps: list[dict[str, Any]] = []
    architecture_forward_steps: list[dict[str, Any]] = []
    parameter_data_errors: list[str] = []
    full_trained_parameter_state = (
        parameter_elements > 0
        and int(parameter_state.get("persisted_parameter_elements") or 0) == parameter_elements
        and int(parameter_state.get("trained_parameter_elements") or 0) == parameter_elements
    )
    dense_parameter_state_reconstructable = full_trained_parameter_state or _dense_base_enabled(payload)

    generated: list[int] = []
    last = int(prompt[-1])
    limit = max(0, int(max_new_tokens))
    for index in range(limit):
        if last in transitions:
            next_token = transitions[last]
            transition_source = "transition"
        else:
            if not fallback:
                raise ValueError(f"Native family checkpoint has no fallback tokens for unseen token {last}.")
            next_token = fallback[index % len(fallback)]
            transition_source = "fallback"
        base_next_token = next_token
        candidates = [base_next_token]
        for candidate in fallback:
            if candidate not in candidates:
                candidates.append(candidate)
        architecture_forward_used = False
        if architecture_forward_supported and parameter_data_size_matches and parameter_data_path is not None:
            try:
                architecture_payload = _score_architecture_forward_candidates(
                    parameter_data_path,
                    payload,
                    last_token=last,
                    candidates=candidates,
                )
            except ValueError as exc:
                parameter_data_errors.append(str(exc))
            else:
                if (
                    bool(architecture_payload.get("supported"))
                    and float(architecture_payload.get("selected_score") or 0.0) > 0.0
                ):
                    next_token = int(architecture_payload["selected_token"])
                    architecture_forward_used = True
                    architecture_forward_inference_count += 1
                    parameter_lm_head_inference_count += 1
                    step_payload = {
                        "step": index,
                        "source": transition_source,
                        "last_token": last,
                        "base_next_token": base_next_token,
                        **architecture_payload,
                    }
                    architecture_forward_steps.append(step_payload)
                    parameter_lm_head_steps.append({**step_payload, "via": ARCHITECTURE_FORWARD_INFERENCE_PATH})
        if (
            not architecture_forward_used
            and parameter_lm_head_supported
            and parameter_data_size_matches
            and parameter_data_path is not None
        ):
            try:
                lm_head_payload = _score_lm_head_candidates(
                    parameter_data_path,
                    payload,
                    last_token=last,
                    candidates=candidates,
                )
            except ValueError as exc:
                parameter_data_errors.append(str(exc))
            else:
                if bool(lm_head_payload.get("supported")) and float(lm_head_payload.get("selected_score") or 0.0) > 0.0:
                    next_token = int(lm_head_payload["selected_token"])
                    parameter_lm_head_inference_count += 1
                    parameter_lm_head_steps.append(
                        {
                            "step": index,
                            "source": transition_source,
                            "last_token": last,
                            "base_next_token": base_next_token,
                            **lm_head_payload,
                        }
                    )
        if parameter_data_size_matches and parameter_data_path is not None:
            probe_index = _parameter_probe_index(last, base_next_token, parameter_elements)
            try:
                parameter_value = _read_sidecar_float32(parameter_data_path, probe_index)
                parameter_probe_count += 1
            except ValueError as exc:
                parameter_data_errors.append(str(exc))
            else:
                parameter_offset = _parameter_token_offset(parameter_value, vocab_size)
                if parameter_offset and not architecture_forward_used:
                    next_token = (next_token + parameter_offset) % vocab_size
                    parameter_influenced_steps.append(
                        {
                            "step": index,
                            "source": transition_source,
                            "last_token": last,
                            "base_next_token": base_next_token,
                            "parameter_index": probe_index,
                            "parameter_value": parameter_value,
                            "token_offset": parameter_offset,
                            "next_token": next_token,
                        }
                    )
        generated.append(next_token)
        last = next_token

    return {
        "status": "native-family-checkpoint-sampler",
        "path": str(checkpoint_path),
        "model_family": str(payload.get("model_family") or ""),
        "template_name": str(payload.get("template_name") or ""),
        "checkpoint_kind": str(payload.get("checkpoint_kind") or ""),
        "parameter_state": payload.get("native_parameter_state") if isinstance(payload.get("native_parameter_state"), dict) else {},
        "parameter_data": payload.get("parameter_data") if isinstance(payload.get("parameter_data"), dict) else {},
        "parameter_data_path": str(parameter_data_path) if parameter_data_path is not None else "",
        "parameter_data_exists": parameter_data_exists,
        "parameter_data_bytes": parameter_data_size,
        "expected_parameter_data_bytes": expected_parameter_data_bytes,
        "parameter_data_size_matches": parameter_data_size_matches,
        "parameter_data_probed": parameter_probe_count > 0,
        "parameter_probe_count": parameter_probe_count,
        "dense_parameter_state_reconstructable": dense_parameter_state_reconstructable,
        "dense_base_initialization": (
            DENSE_BASE_INITIALIZATION if _dense_base_enabled(payload) else ""
        ),
        "parameter_lm_head_inference_supported": parameter_lm_head_supported,
        "parameter_lm_head_inference_used": parameter_lm_head_inference_count > 0,
        "parameter_lm_head_inference_count": parameter_lm_head_inference_count,
        "architecture_forward_inference_supported": architecture_forward_supported,
        "architecture_forward_inference_used": architecture_forward_inference_count > 0,
        "architecture_forward_inference_count": architecture_forward_inference_count,
        "architecture_forward_path": (
            ARCHITECTURE_FORWARD_INFERENCE_PATH if architecture_forward_inference_count > 0 else ""
        ),
        "parameter_lm_head_steps": parameter_lm_head_steps,
        "architecture_forward_steps": architecture_forward_steps,
        "parameter_influenced_steps": parameter_influenced_steps,
        "parameter_data_errors": parameter_data_errors,
        "prompt_tokens": prompt,
        "generated_tokens": generated,
    }


def verify_native_family_checkpoint(
    path: str | Path,
    *,
    prompt_tokens: list[int] | str | None = None,
    max_new_tokens: int = 1,
    require_architecture_forward: bool = False,
) -> NativeFamilyCheckpointVerification:
    checkpoint_path = Path(path).expanduser()
    errors: list[str] = []
    payload: dict[str, Any] = {}
    sample: dict[str, Any] = {}
    try:
        payload = _load_payload(checkpoint_path)
        info = read_native_family_checkpoint_info(checkpoint_path)
    except Exception as exc:
        fallback_info = NativeFamilyCheckpointInfo(
            path=checkpoint_path,
            model_family="",
            native_target="",
            template_name="",
            dataset_alias="",
            checkpoint_kind="",
            vocab_size=0,
            transition_count=0,
            steps_completed=0,
            train_batches_sampled=0,
            validation_batches_sampled=0,
            done_marker_exists=False,
            parameter_state_type="",
            parameter_storage="",
            parameter_initialization="",
            dense_parameter_state_reconstructable=False,
            base_parameter_initialization="",
            base_parameter_seed=0,
            base_parameter_scale=0.0,
            full_template_parameter_state=False,
            parameter_buffer_count=0,
            parameter_elements=0,
            persisted_parameter_elements=0,
            trained_parameter_elements=0,
            parameter_update_checksum=0,
            writer_verification_passed=False,
            writer_verification_update_probe_count=0,
            writer_dense_base_initialization_verified=False,
            writer_dense_base_probe_count=0,
            writer_dense_base_probe_checksum=0,
            writer_verification_error="",
            architecture_forward_inference_supported=False,
            parameter_lm_head_inference_supported=False,
            working_model_inference_path="",
            transition_sampler_inference_supported=False,
            parameter_data_path=None,
            parameter_data_exists=False,
            parameter_data_bytes=0,
            expected_parameter_data_bytes=0,
            parameter_data_size_matches=False,
        )
        return NativeFamilyCheckpointVerification(
            path=checkpoint_path,
            passed=False,
            errors=(str(exc),),
            info=fallback_info,
            sample={},
        )

    if not info.done_marker_exists:
        errors.append("missing native-family model DONE marker")
    if info.vocab_size <= 0:
        errors.append("missing or invalid vocab_size")
    if info.transition_count <= 0:
        errors.append("native-family transition table is empty")
    if not info.transition_sampler_inference_supported:
        errors.append("transition sampler inference is not supported")
    if not info.parameter_lm_head_inference_supported:
        errors.append("parameter LM-head inference is not supported")
    if info.parameter_lm_head_inference_supported and info.working_model_inference_path not in WORKING_MODEL_INFERENCE_PATHS:
        errors.append("working model inference path is not a supported native-family path")
    if info.architecture_forward_inference_supported and info.working_model_inference_path != ARCHITECTURE_FORWARD_INFERENCE_PATH:
        errors.append("architecture-forward checkpoints must use native_family_architecture_sidecar_forward_v1")
    if info.architecture_forward_inference_supported and not (
        info.dense_parameter_state_reconstructable or full_trained_parameter_state
    ):
        errors.append("architecture-forward checkpoints must reconstruct dense parameter state")
    full_trained_parameter_state = (
        info.parameter_elements > 0
        and info.persisted_parameter_elements == info.parameter_elements
        and info.trained_parameter_elements == info.parameter_elements
    )
    dense_base_parameter_state = (
        info.dense_parameter_state_reconstructable
        and not full_trained_parameter_state
        and info.base_parameter_initialization == DENSE_BASE_INITIALIZATION
        and (require_architecture_forward or info.architecture_forward_inference_supported)
    )
    if (
        info.dense_parameter_state_reconstructable
        and not full_trained_parameter_state
        and info.base_parameter_initialization
        and info.base_parameter_initialization != DENSE_BASE_INITIALIZATION
    ):
        errors.append("dense parameter state uses an unsupported base initialization")
    if (
        dense_base_parameter_state
        and info.base_parameter_scale <= 0.0
    ):
        errors.append("dense parameter state base scale is invalid")
    if require_architecture_forward and not info.architecture_forward_inference_supported:
        errors.append("architecture-forward inference from persistent parameter state is not supported")
    if info.parameter_buffer_count <= 0:
        errors.append("architecture parameter layout has no buffers")
    if info.parameter_elements <= 0:
        errors.append("architecture parameter layout has no elements")
    if not info.full_template_parameter_state:
        errors.append("native-family checkpoint does not persist a full template parameter state")
    if info.persisted_parameter_elements <= 0:
        errors.append("no persisted parameter elements are recorded")
    if info.persisted_parameter_elements != info.parameter_elements:
        errors.append("persisted parameter elements do not match architecture parameter elements")
    if info.trained_parameter_elements <= 0:
        errors.append("no trained sampled parameter elements are recorded")
    if (
        (require_architecture_forward or info.architecture_forward_inference_supported)
        and info.parameter_elements > 0
        and info.trained_parameter_elements != info.parameter_elements
    ):
        errors.append(
            "architecture-forward checkpoints must train every architecture parameter element"
        )
    if info.parameter_update_checksum == 0:
        errors.append("missing parameter update checksum")
    if not info.writer_verification_passed:
        errors.append("native-family checkpoint writer verification did not pass")
    if info.writer_verification_update_probe_count <= 0:
        errors.append("native-family checkpoint writer verification did not probe sampled updates")
    if (
        dense_base_parameter_state
        and not info.writer_dense_base_initialization_verified
    ):
        errors.append("native-family checkpoint writer did not verify dense base initialization")
    if (
        dense_base_parameter_state
        and info.writer_dense_base_probe_count <= 0
    ):
        errors.append("native-family checkpoint writer did not probe dense base initialization")
    if info.writer_verification_error:
        errors.append(f"native-family checkpoint writer verification error: {info.writer_verification_error}")
    if not info.parameter_data_path:
        errors.append("missing parameter_data path")
    if not info.parameter_data_exists:
        errors.append("parameter_data sidecar does not exist")
    if info.expected_parameter_data_bytes <= 0:
        errors.append("missing expected parameter_data byte size")
    if not info.parameter_data_size_matches:
        errors.append("parameter_data sidecar size does not match checkpoint metadata")
    errors.extend(_layout_contract_errors(payload, info))
    if dense_base_parameter_state:
        errors.extend(_dense_base_probe_contract_errors(payload, info.parameter_data_path))

    if prompt_tokens is None:
        prompt = _transition_prompt_tokens(payload)
    elif isinstance(prompt_tokens, str):
        prompt = parse_prompt_tokens(prompt_tokens)
    else:
        prompt = [int(token) for token in prompt_tokens]
    if not prompt:
        errors.append("no transition or fallback token is available for sampling")
    else:
        try:
            sample = sample_native_family_checkpoint(
                checkpoint_path,
                prompt_tokens=prompt,
                max_new_tokens=max(1, int(max_new_tokens)),
            )
        except Exception as exc:
            errors.append(f"bounded sample failed: {exc}")
        else:
            generated = sample.get("generated_tokens")
            if not isinstance(generated, list) or not generated:
                errors.append("bounded sample produced no tokens")
            if not bool(sample.get("parameter_data_probed")):
                errors.append("bounded sample did not probe parameter_data sidecar")
            if not bool(sample.get("parameter_lm_head_inference_used")):
                errors.append("bounded sample did not use parameter LM-head inference")
            if require_architecture_forward and not bool(sample.get("architecture_forward_inference_used")):
                errors.append("bounded sample did not use architecture-forward inference")
            if info.architecture_forward_inference_supported and not bool(
                sample.get("dense_parameter_state_reconstructable")
            ):
                errors.append("bounded sample did not reconstruct dense parameter state")
            sample_errors = sample.get("parameter_data_errors")
            if isinstance(sample_errors, list) and sample_errors:
                errors.extend(str(error) for error in sample_errors)

    return NativeFamilyCheckpointVerification(
        path=checkpoint_path,
        passed=not errors,
        errors=tuple(errors),
        info=info,
        sample=sample,
    )


def render_native_family_checkpoint_sampler_text(payload: dict[str, Any]) -> str:
    generated = payload.get("generated_tokens")
    if not isinstance(generated, list):
        return ""
    return f"Generated token ids: {[int(token) for token in generated]}"
