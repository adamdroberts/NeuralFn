"""Lean, fail-closed SDK contract for resident native inference.

This module is only the Python coordination seam for a future in-process C++
engine.  It does not contain a model adapter and it never falls back to the
existing subprocess-based native commands.  A model can be opened only when
both its Native Execution manifest and an injected/installed binding prove the
resident ABI and requested cache capabilities.

The binding contract is intentionally small and duck-typed so a compiled
extension can implement it without a Python framework dependency.  Binding
operations must commit atomically before returning.  In particular,
``decode_one`` must commit the returned token to native session state before it
returns; Python token callbacks are invoked only after the matching Python
history has also been committed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
import os
import hashlib
import importlib
import json
import math
from pathlib import Path
import threading
from typing import Any, Callable, Mapping, Sequence


NATIVE_EXECUTION_MANIFEST_SCHEMA = "neuralfn.native_execution_manifest"
NATIVE_EXECUTION_MANIFEST_VERSION = 1
RESIDENT_INFERENCE_ABI_VERSION = 1
STRUCTURED_OUTPUT_PROFILE = "json-schema-ascii-byte-greedy-v1"
STRUCTURED_OUTPUT_TOKEN_SELECTION = "current_logits_exact_prefill"
FUNCTION_TOOL_TEMPLATE_PROFILE = "responses-forced-function-call-v1"
SESSION_PREFIX_COW_PROFILE = "dense-full-cache-kv-final-hidden-v1"
CPU_TURBOQUANT_SESSION_PREFIX_COW_PROFILE = (
    "dense-cpu-turboquant-mse-qjl-packed-kv-final-hidden-v1"
)
LLAMA_SESSION_PREFIX_COW_PROFILE = "llama-full-cache-gqa-kv-final-hidden-v1"
STANDARD_MOE_SESSION_PREFIX_COW_PROFILE = (
    "standard-moe-full-cache-gqa-kv-final-hidden-v1"
)
_SESSION_PREFIX_COW_PROFILES_BY_CHECKPOINT_FORMAT = {
    "neuralfn.native_dense_gpt.v5": SESSION_PREFIX_COW_PROFILE,
    "neuralfn.native_family_llama.f32.v1": LLAMA_SESSION_PREFIX_COW_PROFILE,
    "neuralfn.native_family_standard_moe.f32.v1": (
        STANDARD_MOE_SESSION_PREFIX_COW_PROFILE
    ),
}
_DEFAULT_BINDING_MODULE = "neuralfn._native_inference"
_CACHE_MODES = frozenset({"auto", "off", "full", "turboquant"})
_TURBOQUANT_PROFILES = frozenset({"mse-3.5", "qjl-3.5"})
_TURBOQUANT_ATTENTION_BACKENDS = frozenset({"cpu", "tile-cuda"})
_WEIGHT_PRECISIONS = (
    "auto",
    "bf16",
    "k-quant-dynamic",
    "k-quant-17gb",
)
_WEIGHT_FIDELITY_ORDER = (
    "bf16",
    "k-quant-dynamic",
    "k-quant-17gb",
)
_MODEL_RUNTIMES = frozenset({"auto", "cpu", "native-cuda"})
_SPECULATIVE_MODES = frozenset({"off", "auto", "required"})
_GIB = 1 << 30


class NativeInferenceError(RuntimeError):
    """Base error for the resident native inference SDK."""


class NativeInferenceCapabilityError(NativeInferenceError):
    """Raised when an artifact or binding cannot prove a requested feature."""


class NativeInferenceClosedError(NativeInferenceError):
    """Raised when an operation targets a closed model or session."""


class NativeInferenceCancelledError(NativeInferenceError):
    """Raised when an in-flight non-generation operation is cancelled."""


@dataclass(frozen=True, slots=True)
class NativeInferenceCapabilities:
    """Effective capabilities proven by both artifact and native binding."""

    native_inference: bool
    resident_inference: bool
    lossless_kv_cache: bool
    turboquant_kv_cache: bool
    function_tools: bool = False
    structured_output: bool = False
    session_state_kinds: tuple[str, ...] = ()
    resident_inference_abi: int = RESIDENT_INFERENCE_ABI_VERSION
    turboquant_tile_attention: bool = False
    session_prefix_cow: bool = False
    session_prefix_cow_cpu_turboquant: bool = False
    speculative_decoding: bool = False
    dflash_cpu: bool = False
    dflash_cuda: bool = False
    vision: bool = False
    video: bool = False
    vision_cpu: bool = False
    vision_cuda: bool = False
    native_lora: bool = False
    native_lora_cpu: bool = False
    native_lora_cuda: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["session_state_kinds"] = list(self.session_state_kinds)
        return payload


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """One decode request.

    Exact ``0.0`` (including ``-0.0``) selects strict model computation.
    Every positive value, including a positive subnormal, remains standard
    sampling.  Negative and non-finite values are invalid.
    """

    max_new_tokens: int = 1
    temperature: float = 0.8
    top_k: int | None = None
    top_p: float = 1.0
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be greater than or equal to 0")

        try:
            temperature = float(self.temperature)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "temperature must be a finite number greater than or equal to 0"
            ) from exc
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be a finite number greater than or equal to 0")
        object.__setattr__(self, "temperature", temperature)

        if self.top_k is not None:
            if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
                raise TypeError("top_k must be an integer or None")
            if self.top_k < 0:
                raise ValueError("top_k must be greater than or equal to 0")

        try:
            top_p = float(self.top_p)
        except (TypeError, ValueError) as exc:
            raise ValueError("top_p must be finite and in the interval (0, 1]") from exc
        if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be finite and in the interval (0, 1]")
        object.__setattr__(self, "top_p", top_p)

        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise TypeError("seed must be an integer or None")
        object.__setattr__(self, "stop_token_ids", _normalize_token_ids(self.stop_token_ids))

    @property
    def strict_model_compute(self) -> bool:
        return self.temperature == 0.0

    def to_binding_payload(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "seed": self.seed,
            "stop_token_ids": list(self.stop_token_ids),
            "strict_model_compute": self.strict_model_compute,
        }


@dataclass(frozen=True, slots=True)
class KVCacheConfig:
    """Resident cache request; TurboQuant is always explicit and fail-closed."""

    mode: str = "auto"
    turboquant_profile: str = "mse-3.5"
    turboquant_attention_backend: str = "cpu"
    tile_ops_lib: str | None = None
    cuda_runtime_lib: str | None = None
    cuda_device: int = 0

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        profile = str(self.turboquant_profile).strip().lower()
        backend = (
            str(self.turboquant_attention_backend)
            .strip()
            .lower()
            .replace("_", "-")
        )
        if mode not in _CACHE_MODES:
            raise ValueError("kv cache mode must be one of: auto, off, full, turboquant")
        if profile not in _TURBOQUANT_PROFILES:
            raise ValueError("turboquant_profile must be one of: mse-3.5, qjl-3.5")
        if backend not in _TURBOQUANT_ATTENTION_BACKENDS:
            raise ValueError(
                "turboquant_attention_backend must be one of: cpu, tile-cuda"
            )
        if isinstance(self.cuda_device, bool) or not isinstance(self.cuda_device, int):
            raise TypeError("cuda_device must be an integer")
        if self.cuda_device < 0:
            raise ValueError("cuda_device must be greater than or equal to 0")
        tile_ops_lib = (
            str(self.tile_ops_lib).strip() if self.tile_ops_lib is not None else None
        )
        cuda_runtime_lib = (
            str(self.cuda_runtime_lib).strip()
            if self.cuda_runtime_lib is not None
            else None
        )
        if tile_ops_lib == "":
            raise ValueError("tile_ops_lib must not be empty")
        if cuda_runtime_lib == "":
            raise ValueError("cuda_runtime_lib must not be empty")
        if backend == "tile-cuda":
            if mode != "turboquant":
                raise ValueError(
                    "tile-cuda TurboQuant attention requires mode='turboquant'"
                )
            if tile_ops_lib is None:
                raise ValueError(
                    "tile-cuda TurboQuant attention requires tile_ops_lib"
                )
        elif tile_ops_lib is not None or cuda_runtime_lib is not None or self.cuda_device != 0:
            raise ValueError(
                "Tile-CUDA library/device options require "
                "turboquant_attention_backend='tile-cuda'"
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "turboquant_profile", profile)
        object.__setattr__(self, "turboquant_attention_backend", backend)
        object.__setattr__(self, "tile_ops_lib", tile_ops_lib)
        object.__setattr__(self, "cuda_runtime_lib", cuda_runtime_lib)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NativeModelLoadConfig:
    """Immutable model-weight selection and device-load policy.

    Weight precision is independent of activation dtype and KV-cache format.
    ``auto`` is quality-first for a proved whole-model CUDA runner and resolves
    to the authenticated primary variant on CPU.
    """

    weight_precision: str = "auto"
    runtime: str = "auto"
    cuda_device: int = 0
    tile_ops_lib: str | None = None
    cuda_runtime_lib: str | None = None
    vram_reserve_bytes: int | None = None
    context_tokens: int | None = None
    session_count: int = 1
    companion_checkpoints: tuple[str, ...] = ()
    speculative_decoding: str = "auto"

    def __post_init__(self) -> None:
        precision = str(self.weight_precision).strip().lower().replace("_", "-")
        runtime = str(self.runtime).strip().lower().replace("_", "-")
        speculation = str(self.speculative_decoding).strip().lower().replace("_", "-")
        if precision not in _WEIGHT_PRECISIONS:
            raise ValueError(
                "weight_precision must be one of: " + ", ".join(_WEIGHT_PRECISIONS)
            )
        if runtime not in _MODEL_RUNTIMES:
            raise ValueError("runtime must be one of: auto, cpu, native-cuda")
        if speculation not in _SPECULATIVE_MODES:
            raise ValueError("speculative_decoding must be one of: off, auto, required")
        if isinstance(self.cuda_device, bool) or not isinstance(self.cuda_device, int):
            raise TypeError("cuda_device must be an integer")
        if self.cuda_device < 0:
            raise ValueError("cuda_device must be greater than or equal to 0")
        tile_ops_lib = (
            str(self.tile_ops_lib).strip()
            if self.tile_ops_lib is not None
            else None
        )
        runtime_lib = (
            str(self.cuda_runtime_lib).strip()
            if self.cuda_runtime_lib is not None
            else None
        )
        if tile_ops_lib == "":
            raise ValueError("tile_ops_lib must not be empty")
        if runtime_lib == "":
            raise ValueError("cuda_runtime_lib must not be empty")
        for name, value, allow_none in (
            ("vram_reserve_bytes", self.vram_reserve_bytes, True),
            ("context_tokens", self.context_tokens, True),
            ("session_count", self.session_count, False),
        ):
            if value is None and allow_none:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer" + (" or None" if allow_none else ""))
            if value < (1 if name == "session_count" else 0):
                raise ValueError(
                    f"{name} must be "
                    + ("positive" if name == "session_count" else "greater than or equal to 0")
                )
        companions: list[str] = []
        for name in self.companion_checkpoints:
            normalized = str(name).strip()
            if not normalized or normalized in companions:
                raise ValueError("companion_checkpoints must contain unique non-empty names")
            companions.append(normalized)
        object.__setattr__(self, "weight_precision", precision)
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "speculative_decoding", speculation)
        object.__setattr__(self, "tile_ops_lib", tile_ops_lib)
        object.__setattr__(self, "cuda_runtime_lib", runtime_lib)
        object.__setattr__(self, "companion_checkpoints", tuple(companions))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["companion_checkpoints"] = list(self.companion_checkpoints)
        return payload


def _resolve_model_tile_ops_library(config: NativeModelLoadConfig) -> str:
    """Resolve the strict raw CUDA sidecar without mutating load policy.

    Explicit paths are authoritative.  The fallback locations cover an
    installed package sidecar and this repository's documented build output;
    failure is reported before the resident handle is created.
    """

    raw_candidates: list[str | Path] = []
    if config.tile_ops_lib is not None:
        raw_candidates.append(config.tile_ops_lib)
    else:
        environment = os.environ.get("NFN_NATIVE_TILE_OPS_LIB")
        if environment:
            raw_candidates.append(environment)
        package_dir = Path(__file__).resolve().parent
        raw_candidates.extend(
            (
                package_dir / "libnfn_native_train_tile_ops.so",
                package_dir.parent / "build" / "libnfn_native_train_tile_ops.so",
            )
        )
    checked: list[str] = []
    for raw in raw_candidates:
        candidate = Path(raw).expanduser()
        checked.append(str(candidate))
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        if resolved.is_file():
            return str(resolved)
    raise NativeInferenceCapabilityError(
        "Whole-model CUDA requires the strict Tile-CUDA sidecar; checked: "
        + ", ".join(checked)
    )


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    """A token event emitted after native and Python state are committed."""

    token_id: int
    index: int
    position: int
    text: str = ""
    finish_reason: str | None = None
    committed: bool = True
    kind: str = "token"


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Immutable result of one session decode call."""

    token_ids: tuple[int, ...]
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    events: tuple[GenerationEvent, ...] = ()
    cancelled: bool = False
    speculative_proposed_tokens: int = 0
    speculative_accepted_tokens: int = 0
    speculative_rejected_tokens: int = 0
    speculative_target_rows: int = 0
    speculative_assistant_blocks: int = 0


TokenCallback = Callable[[GenerationEvent], None]


def _normalize_token_ids(token_ids: Sequence[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for index, token_id in enumerate(token_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"token_ids[{index}] must be an integer")
        if token_id < 0:
            raise ValueError(f"token_ids[{index}] must be greater than or equal to 0")
        normalized.append(token_id)
    return tuple(normalized)


def _longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _require_bool(mapping: Mapping[str, Any], key: str, *, source: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise NativeInferenceCapabilityError(
            f"{source} must declare boolean capability {key!r}"
        )
    return value


def _is_exact_version(value: Any, expected: int) -> bool:
    """Require an integer ABI/schema version without accepting bool as int."""

    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _load_manifest(artifact: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    requested = Path(artifact).expanduser().resolve()
    if requested.is_dir():
        artifact_root = requested
        manifest_path = requested / "native-execution-manifest.json"
    else:
        artifact_root = requested.parent
        manifest_path = requested
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Native Execution manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NativeInferenceCapabilityError(
            f"Native Execution manifest is not valid JSON: {manifest_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise NativeInferenceCapabilityError("Native Execution manifest root must be an object")
    if payload.get("schema") != NATIVE_EXECUTION_MANIFEST_SCHEMA:
        raise NativeInferenceCapabilityError(
            f"Unsupported Native Execution manifest schema {payload.get('schema')!r}"
        )
    if not _is_exact_version(
        payload.get("version"), NATIVE_EXECUTION_MANIFEST_VERSION
    ):
        raise NativeInferenceCapabilityError(
            f"Unsupported Native Execution manifest version {payload.get('version')!r}"
        )
    return artifact_root, manifest_path, payload


def _validate_checkpoint_artifact(
    artifact_root: Path,
    manifest: Mapping[str, Any],
) -> None:
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise NativeInferenceCapabilityError(
            "Resident inference artifact must declare a checkpoint object"
        )
    relative = checkpoint.get("artifact_path")
    if not isinstance(relative, str) or not relative.strip():
        raise NativeInferenceCapabilityError(
            "Resident inference checkpoint must declare a relative artifact_path"
        )
    requested = Path(relative)
    if requested.is_absolute():
        raise NativeInferenceCapabilityError(
            "Resident inference checkpoint artifact_path must be relative"
        )
    candidate = (artifact_root / requested).resolve()
    try:
        candidate.relative_to(artifact_root)
    except ValueError as exc:
        raise NativeInferenceCapabilityError(
            "Resident inference checkpoint artifact_path escapes the artifact root"
        ) from exc
    if not candidate.is_file():
        raise NativeInferenceCapabilityError(
            f"Resident inference checkpoint does not exist: {candidate}"
        )
    target_nbytes = checkpoint.get("target_nbytes")
    if isinstance(target_nbytes, bool) or not isinstance(target_nbytes, int) or target_nbytes < 0:
        raise NativeInferenceCapabilityError(
            "Resident inference checkpoint must declare a non-negative target_nbytes"
        )
    if candidate.stat().st_size != target_nbytes:
        raise NativeInferenceCapabilityError(
            "Resident inference checkpoint length does not match its manifest fingerprint"
        )
    expected_sha256 = checkpoint.get("target_sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256.lower())
    ):
        raise NativeInferenceCapabilityError(
            "Resident inference checkpoint must declare a hexadecimal target_sha256"
        )
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256.lower():
        raise NativeInferenceCapabilityError(
            "Resident inference checkpoint checksum does not match its manifest fingerprint"
        )


def _load_manifest_with_path_pin(
    artifact: str | Path,
) -> tuple[Path, Path, dict[str, Any], str | None]:
    requested = Path(artifact).expanduser().resolve()
    if requested.is_dir() or requested.name == "native-execution-manifest.json":
        artifact_root, manifest_path, manifest = _load_manifest(requested)
        return artifact_root, manifest_path, manifest, None
    if not requested.is_file():
        raise FileNotFoundError(f"Native artifact does not exist: {requested}")
    artifact_root, manifest_path, manifest = _load_manifest(requested.parent)
    variants = manifest.get("checkpoint_variants")
    if not isinstance(variants, Mapping):
        raise NativeInferenceCapabilityError(
            "An exact checkpoint-file path can be used only with a multi-variant manifest"
        )
    matches: list[str] = []
    for profile, descriptor in variants.items():
        if not isinstance(profile, str) or not isinstance(descriptor, Mapping):
            continue
        relative = descriptor.get("artifact_path")
        if not isinstance(relative, str):
            continue
        candidate = (artifact_root / relative).resolve()
        try:
            candidate.relative_to(artifact_root)
        except ValueError:
            continue
        if candidate == requested:
            matches.append(profile)
    if len(matches) != 1:
        raise NativeInferenceCapabilityError(
            "The supplied checkpoint file is not exactly one contained checkpoint variant"
        )
    return artifact_root, manifest_path, manifest, matches[0]


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NativeInferenceCapabilityError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: Any, field: str) -> int:
    parsed = _nonnegative_int(value, field)
    if parsed == 0:
        raise NativeInferenceCapabilityError(f"{field} must be positive")
    return parsed


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _contained_checkpoint_path(
    artifact_root: Path,
    descriptor: Mapping[str, Any],
    *,
    field: str,
) -> Path:
    relative = descriptor.get("artifact_path")
    if not isinstance(relative, str) or not relative.strip():
        raise NativeInferenceCapabilityError(f"{field}.artifact_path must be a non-empty string")
    path = Path(relative)
    if path.is_absolute():
        raise NativeInferenceCapabilityError(f"{field}.artifact_path must be relative")
    resolved = (artifact_root / path).resolve()
    try:
        resolved.relative_to(artifact_root)
    except ValueError as exc:
        raise NativeInferenceCapabilityError(f"{field}.artifact_path escapes the artifact root") from exc
    return resolved


def _validate_variant_catalog(
    artifact_root: Path,
    manifest: Mapping[str, Any],
) -> tuple[str | None, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    raw_variants = manifest.get("checkpoint_variants")
    if raw_variants is None:
        return None, {}, {}
    if not isinstance(raw_variants, Mapping) or not raw_variants:
        raise NativeInferenceCapabilityError("checkpoint_variants must be a non-empty object")
    primary = manifest.get("primary_checkpoint_variant")
    if not isinstance(primary, str) or primary not in raw_variants:
        raise NativeInferenceCapabilityError(
            "primary_checkpoint_variant must name one declared checkpoint variant"
        )
    unknown = sorted(set(raw_variants) - set(_WEIGHT_FIDELITY_ORDER))
    if unknown:
        raise NativeInferenceCapabilityError(
            "Unsupported checkpoint variant IDs: " + ", ".join(unknown)
        )
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise NativeInferenceCapabilityError("Multi-variant manifests require checkpoint")

    variants: dict[str, dict[str, Any]] = {}
    executable_fields = ("format", "artifact_path", "target_nbytes", "target_sha256")
    for profile, raw in raw_variants.items():
        if not isinstance(profile, str) or not isinstance(raw, Mapping):
            raise NativeInferenceCapabilityError("checkpoint_variants entries must be named objects")
        descriptor = dict(raw)
        for field in executable_fields:
            if field not in descriptor:
                raise NativeInferenceCapabilityError(
                    f"checkpoint_variants.{profile} is missing executable field {field}"
                )
        if not isinstance(descriptor["format"], str) or not descriptor["format"]:
            raise NativeInferenceCapabilityError(
                f"checkpoint_variants.{profile}.format must be a non-empty string"
            )
        _contained_checkpoint_path(
            artifact_root,
            descriptor,
            field=f"checkpoint_variants.{profile}",
        )
        _nonnegative_int(
            descriptor["target_nbytes"],
            f"checkpoint_variants.{profile}.target_nbytes",
        )
        digest = descriptor["target_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            raise NativeInferenceCapabilityError(
                f"checkpoint_variants.{profile}.target_sha256 must be a hexadecimal digest"
            )
        kernel_profile = descriptor.get("required_kernel_profile")
        if not isinstance(kernel_profile, str) or not kernel_profile.strip():
            raise NativeInferenceCapabilityError(
                f"checkpoint_variants.{profile}.required_kernel_profile must be a string"
            )
        for field in (
            "resident_weight_bytes",
            "peak_load_staging_bytes",
            "max_workspace_bytes",
        ):
            _nonnegative_int(descriptor.get(field), f"checkpoint_variants.{profile}.{field}")
        memory = descriptor.get("memory_profile")
        if not isinstance(memory, Mapping) or not _is_exact_version(memory.get("version"), 1):
            raise NativeInferenceCapabilityError(
                f"checkpoint_variants.{profile}.memory_profile must use version 1"
            )
        _positive_int(
            memory.get("minimum_total_vram_bytes"),
            f"checkpoint_variants.{profile}.memory_profile.minimum_total_vram_bytes",
        )
        for field in (
            "fixed_runtime_bytes",
            "kv_cache_bytes_per_context_token_per_session",
            "session_bytes",
        ):
            _nonnegative_int(
                memory.get(field, 0),
                f"checkpoint_variants.{profile}.memory_profile.{field}",
            )
        hybrid = memory.get("hybrid_kv_cache")
        if hybrid is not None:
            if not isinstance(hybrid, Mapping):
                raise NativeInferenceCapabilityError(
                    f"checkpoint_variants.{profile}.memory_profile.hybrid_kv_cache must be an object"
                )
            for field in (
                "local_layers",
                "global_layers",
                "local_window",
                "kv_heads",
                "head_dim",
                "key_value_components",
                "bytes_per_element",
                "final_hidden_elements",
            ):
                _positive_int(
                    hybrid.get(field),
                    f"checkpoint_variants.{profile}.memory_profile.hybrid_kv_cache.{field}",
                )
            if int(memory.get("kv_cache_bytes_per_context_token_per_session", 0)) != 0:
                raise NativeInferenceCapabilityError(
                    f"checkpoint_variants.{profile} cannot declare both linear and hybrid KV accounting"
                )
        fingerprint = memory.get("backend_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise NativeInferenceCapabilityError(
                f"checkpoint_variants.{profile}.memory_profile.backend_fingerprint must be a string"
            )
        variants[profile] = descriptor
    if any(checkpoint.get(field) != variants[primary].get(field) for field in executable_fields):
        raise NativeInferenceCapabilityError(
            "checkpoint must equal the primary checkpoint variant's executable fields"
        )

    raw_companions = manifest.get("companion_checkpoints", {})
    if not isinstance(raw_companions, Mapping):
        raise NativeInferenceCapabilityError("companion_checkpoints must be an object")
    companions: dict[str, dict[str, Any]] = {}
    for name, raw in raw_companions.items():
        if not isinstance(name, str) or not name or not isinstance(raw, Mapping):
            raise NativeInferenceCapabilityError("companion_checkpoints entries must be named objects")
        descriptor = dict(raw)
        _contained_checkpoint_path(
            artifact_root,
            descriptor,
            field=f"companion_checkpoints.{name}",
        )
        _nonnegative_int(
            descriptor.get("target_nbytes"),
            f"companion_checkpoints.{name}.target_nbytes",
        )
        _nonnegative_int(
            descriptor.get("resident_weight_bytes"),
            f"companion_checkpoints.{name}.resident_weight_bytes",
        )
        companion_format = descriptor.get("format")
        companion_sha = descriptor.get("target_sha256")
        if not isinstance(companion_format, str) or not companion_format:
            raise NativeInferenceCapabilityError(
                f"companion_checkpoints.{name}.format must be a non-empty string"
            )
        if (
            not isinstance(companion_sha, str)
            or len(companion_sha) != 64
            or any(character not in "0123456789abcdef" for character in companion_sha.lower())
        ):
            raise NativeInferenceCapabilityError(
                f"companion_checkpoints.{name}.target_sha256 must be hexadecimal"
            )
        if name == "dflash":
            compatibility = descriptor.get("target_compatibility")
            allowed = (
                compatibility.get("allowed_target_checkpoint_sha256")
                if isinstance(compatibility, Mapping)
                else None
            )
            if (
                companion_format
                not in {
                    "neuralfn.native_family_muse_glimmer_dflash.bf16.v1",
                    "neuralfn.native_family_muse_glimmer_dflash.gguf.kquant.v1",
                }
                or not isinstance(allowed, Sequence)
                or isinstance(allowed, (str, bytes))
                or not allowed
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(ch not in "0123456789abcdef" for ch in value.lower())
                    for value in allowed
                )
                or compatibility.get("target_layer_ids_zero_based")
                != [1, 13, 25, 37, 49]
                or compatibility.get("block_size") != 16
                or compatibility.get("proposal_tokens") != 15
                or compatibility.get("mask_token_id") != 201_818
                or compatibility.get("shared_embedding") is not True
                or compatibility.get("shared_lm_head") is not True
            ):
                raise NativeInferenceCapabilityError(
                    "companion_checkpoints.dflash compatibility contract is not canonical"
                )
        elif name == "mmproj":
            from .native_chat import (
                MUSE_GLIMMER_ATEM_TEMPLATE_SHA256,
                MUSE_GLIMMER_SPECIAL_TOKEN_IDS,
                MUSE_GLIMMER_TOKENIZER_SHA256,
            )
            from .native_muse_glimmer_checkpoint import (
                MAIN_CONFIG_SHA256,
                MAIN_PROCESSOR_CONFIG_SHA256,
            )

            compatibility = descriptor.get("target_compatibility")
            allowed = (
                compatibility.get("allowed_target_checkpoint_sha256")
                if isinstance(compatibility, Mapping)
                else None
            )
            if (
                companion_format
                != "neuralfn.native_family_muse_glimmer_mmproj.gguf.kquant.v1"
                or not isinstance(allowed, Sequence)
                or isinstance(allowed, (str, bytes))
                or not allowed
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(ch not in "0123456789abcdef" for ch in value.lower())
                    for value in allowed
                )
                or compatibility.get("target_config_sha256") != MAIN_CONFIG_SHA256
                or compatibility.get("processor_config_sha256")
                != MAIN_PROCESSOR_CONFIG_SHA256
                or compatibility.get("tokenizer_sha256")
                != MUSE_GLIMMER_TOKENIZER_SHA256
                or compatibility.get("chat_template_sha256")
                != MUSE_GLIMMER_ATEM_TEMPLATE_SHA256
                or compatibility.get("projection_width") != 6_656
                or compatibility.get("patch_size") != 14
                or compatibility.get("temporal_patch_size") != 2
                or compatibility.get("merge_size") != 2
                or compatibility.get("packed_patch_width") != 588
                or compatibility.get("temporal_patch_reduction") != "sum"
                or compatibility.get("media_token_ids")
                != {
                    "image": MUSE_GLIMMER_SPECIAL_TOKEN_IDS["image"],
                    "video": MUSE_GLIMMER_SPECIAL_TOKEN_IDS["video"],
                    "patch": MUSE_GLIMMER_SPECIAL_TOKEN_IDS["patch"],
                }
            ):
                raise NativeInferenceCapabilityError(
                    "companion_checkpoints.mmproj compatibility contract is not canonical"
                )
        elif name == "lora":
            compatibility = descriptor.get("target_compatibility")
            allowed = (
                compatibility.get("allowed_target_checkpoint_sha256")
                if isinstance(compatibility, Mapping)
                else None
            )
            targets = descriptor.get("targets")
            allowed_targets = {
                "q_proj", "k_proj", "v_proj", "o_proj", "attn_gate_proj",
                "gate_proj", "up_proj", "down_proj",
            }
            try:
                rank = int(descriptor.get("rank"))
                alpha = float(descriptor.get("alpha"))
                scaling = float(descriptor.get("scaling"))
                dropout = float(descriptor.get("dropout"))
            except (TypeError, ValueError):
                rank, alpha, scaling, dropout = 0, 0.0, 0.0, -1.0
            parsed_targets = (
                tuple(targets)
                if isinstance(targets, Sequence)
                and not isinstance(targets, (str, bytes))
                and all(isinstance(target, str) for target in targets)
                else ()
            )
            per_layer = 0
            if parsed_targets:
                shapes = {
                    "q_proj": (4_096, 6_656),
                    "k_proj": (256, 6_656),
                    "v_proj": (256, 6_656),
                    "o_proj": (6_656, 4_096),
                    "attn_gate_proj": (4_096, 6_656),
                    "gate_proj": (19_968, 6_656),
                    "up_proj": (19_968, 6_656),
                    "down_proj": (6_656, 19_968),
                }
                for target in parsed_targets:
                    if target in shapes:
                        rows, cols = shapes[target]
                        per_layer += rank * (rows + cols)
            expected_nbytes = 52 * per_layer * 2
            if (
                companion_format != "neuralfn.native_muse_glimmer_lora.bf16.v1"
                or descriptor.get("artifact_path") != "muse-glimmer-lora.bf16"
                or not isinstance(allowed, Sequence)
                or isinstance(allowed, (str, bytes))
                or len(allowed) != 1
                or any(not isinstance(value, str) or not _is_sha256(value) for value in allowed)
                or compatibility.get("base_weight_precision") != "bf16"
                or not parsed_targets
                or len(parsed_targets) != len(set(parsed_targets))
                or not set(parsed_targets) <= allowed_targets
                or rank <= 0
                or alpha <= 0.0
                or scaling != alpha / rank
                or not 0.0 <= dropout < 1.0
                or descriptor.get("target_nbytes") != expected_nbytes
                or descriptor.get("resident_weight_bytes") != expected_nbytes
                or descriptor.get("graph_topology_sha256")
                != "4e3c741890fefc43e1027c61ffcfe9ec09c8f6448ee27dd548ab3fad3c172a56"
                or not _is_sha256(descriptor.get("graph_fingerprint"))
                or compatibility.get("tokenizer_sha256")
                != MUSE_GLIMMER_TOKENIZER_SHA256
                or compatibility.get("chat_template_sha256")
                != MUSE_GLIMMER_ATEM_TEMPLATE_SHA256
            ):
                raise NativeInferenceCapabilityError(
                    "companion_checkpoints.lora contract is not canonical"
                )
        else:
            raise NativeInferenceCapabilityError(
                f"Unsupported companion checkpoint {name!r}"
            )
        companions[name] = descriptor
    return primary, variants, companions


def _binding_weight_kernel_profiles(binding: Any) -> frozenset[str]:
    raw = binding.resident_inference_capabilities()
    profiles = raw.get("weight_kernel_profiles", ()) if isinstance(raw, Mapping) else ()
    if not isinstance(profiles, Sequence) or isinstance(profiles, (str, bytes)):
        raise NativeInferenceCapabilityError("Binding weight_kernel_profiles must be an array")
    if any(not isinstance(profile, str) or not profile for profile in profiles):
        raise NativeInferenceCapabilityError(
            "Binding weight_kernel_profiles must contain non-empty strings"
        )
    return frozenset(profiles)


def _binding_whole_model_cuda(manifest: Mapping[str, Any], binding: Any) -> bool:
    artifact_caps = manifest.get("capabilities")
    binding_caps = binding.resident_inference_capabilities()
    artifact_value = (
        artifact_caps.get("whole_model_cuda", False)
        if isinstance(artifact_caps, Mapping)
        else False
    )
    binding_value = (
        binding_caps.get("whole_model_cuda", False)
        if isinstance(binding_caps, Mapping)
        else False
    )
    if not isinstance(artifact_value, bool) or not isinstance(binding_value, bool):
        raise NativeInferenceCapabilityError(
            "Artifact and binding whole_model_cuda capabilities must be boolean"
        )
    return artifact_value and binding_value


def _query_cuda_memory(config: NativeModelLoadConfig) -> tuple[int, int]:
    candidates = (
        (config.cuda_runtime_lib,)
        if config.cuda_runtime_lib is not None
        else ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0")
    )
    library = None
    last_error: OSError | None = None
    for library_name in candidates:
        try:
            library = ctypes.CDLL(library_name)
            break
        except OSError as exc:
            last_error = exc
    if library is None:
        raise NativeInferenceCapabilityError(
            "CUDA runtime could not be loaded for free-memory selection: "
            + ", ".join(candidates)
        ) from last_error
    try:
        set_device = library.cudaSetDevice
        mem_get_info = library.cudaMemGetInfo
    except AttributeError as exc:
        raise NativeInferenceCapabilityError(
            "CUDA runtime lacks cudaSetDevice/cudaMemGetInfo"
        ) from exc
    set_device.argtypes = [ctypes.c_int]
    set_device.restype = ctypes.c_int
    mem_get_info.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    mem_get_info.restype = ctypes.c_int
    status = int(set_device(config.cuda_device))
    if status != 0:
        raise NativeInferenceCapabilityError(
            f"cudaSetDevice({config.cuda_device}) failed with CUDA status {status}"
        )
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    status = int(mem_get_info(ctypes.byref(free), ctypes.byref(total)))
    if status != 0 or free.value > total.value or total.value == 0:
        raise NativeInferenceCapabilityError(
            f"cudaMemGetInfo failed with CUDA status {status}"
        )
    return int(free.value), int(total.value)


def _variant_runtime_bytes(
    descriptor: Mapping[str, Any],
    *,
    context_tokens: int,
    session_count: int,
    companion_bytes: int,
) -> int:
    memory = descriptor["memory_profile"]
    per_token = int(memory.get("kv_cache_bytes_per_context_token_per_session", 0))
    per_session = int(memory.get("session_bytes", 0))
    hybrid = memory.get("hybrid_kv_cache")
    if isinstance(hybrid, Mapping):
        retained_rows = (
            int(hybrid["local_layers"])
            * min(context_tokens, int(hybrid["local_window"]))
            + int(hybrid["global_layers"]) * context_tokens
        )
        cache_per_session = (
            retained_rows
            * int(hybrid["kv_heads"])
            * int(hybrid["head_dim"])
            * int(hybrid["key_value_components"])
            * int(hybrid["bytes_per_element"])
            + int(hybrid["final_hidden_elements"])
            * int(hybrid["bytes_per_element"])
        )
    else:
        cache_per_session = per_token * context_tokens
    return (
        int(descriptor["resident_weight_bytes"])
        + companion_bytes
        + int(descriptor["peak_load_staging_bytes"])
        + int(descriptor["max_workspace_bytes"])
        + int(memory.get("fixed_runtime_bytes", 0))
        + cache_per_session * session_count
        + per_session * session_count
    )


def _select_checkpoint_variant(
    artifact_root: Path,
    manifest: Mapping[str, Any],
    binding: Any,
    config: NativeModelLoadConfig,
    *,
    path_pin: str | None = None,
    memory_probe: Callable[[NativeModelLoadConfig], tuple[int, int]] = _query_cuda_memory,
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary, variants, companions = _validate_variant_catalog(artifact_root, manifest)
    requested = config.weight_precision
    if not variants:
        if path_pin is not None or requested != "auto":
            raise NativeInferenceCapabilityError(
                "The artifact has no checkpoint variant matching the requested weight precision"
            )
        if config.speculative_decoding == "required":
            raise NativeInferenceCapabilityError(
                "speculative_decoding='required' needs a multi-variant artifact with DFlash"
            )
        return dict(manifest), {
            "requested_weight_precision": "auto",
            "effective_weight_precision": None,
            "weight_precision_selection": "legacy-single",
            "companion_checkpoints": [],
            "requested_speculative_decoding": config.speculative_decoding,
            "effective_speculative_decoding": "off",
        }
    assert primary is not None
    if path_pin is not None and requested != "auto" and requested != path_pin:
        raise NativeInferenceCapabilityError(
            f"Checkpoint path pins {path_pin!r}, conflicting with explicit {requested!r}"
        )
    requested_companions = list(config.companion_checkpoints)
    missing_companions = sorted(set(requested_companions) - set(companions))
    if missing_companions:
        raise NativeInferenceCapabilityError(
            "Unknown companion checkpoints: " + ", ".join(missing_companions)
        )
    dflash_available = "dflash" in companions
    if "lora" in requested_companions and (
        "dflash" in requested_companions or config.speculative_decoding == "required"
    ):
        raise NativeInferenceCapabilityError(
            "Native LoRA cannot use the stock DFlash companion; select speculative_decoding='off'"
        )
    if config.speculative_decoding == "off" and "dflash" in requested_companions:
        raise NativeInferenceCapabilityError(
            "The dflash companion conflicts with speculative_decoding='off'"
        )
    if config.speculative_decoding == "required":
        if not dflash_available:
            raise NativeInferenceCapabilityError(
                "speculative_decoding='required' but the artifact has no dflash companion"
            )
        if "dflash" not in requested_companions:
            requested_companions.append("dflash")
    for companion_name in requested_companions:
        companion_path = _contained_checkpoint_path(
            artifact_root,
            companions[companion_name],
            field=f"companion_checkpoints.{companion_name}",
        )
        if not companion_path.is_file():
            raise NativeInferenceCapabilityError(
                f"Requested companion checkpoint does not exist: {companion_path}"
            )
    optional_dflash = bool(
        config.speculative_decoding == "auto"
        and dflash_available
        and "dflash" not in requested_companions
        and "lora" not in requested_companions
    )
    companion_bytes = sum(
        int(companions[name]["resident_weight_bytes"]) for name in requested_companions
    )
    kernel_profiles = _binding_weight_kernel_profiles(binding)
    whole_cuda = _binding_whole_model_cuda(manifest, binding)
    runtime = config.runtime
    if runtime == "auto":
        runtime = "native-cuda" if whole_cuda else "cpu"
    if runtime == "native-cuda" and not whole_cuda:
        raise NativeInferenceCapabilityError(
            "Whole-model CUDA weight loading was requested but is not jointly proven"
        )

    context_limits = manifest.get("context_limits")
    declared_context = (
        context_limits.get("max_context_tokens", 0)
        if isinstance(context_limits, Mapping)
        else 0
    )
    context_tokens = (
        config.context_tokens
        if config.context_tokens is not None
        else _positive_int(declared_context, "context_limits.max_context_tokens")
    )
    candidates = (
        (path_pin,)
        if path_pin is not None
        else ((requested,) if requested != "auto" else _WEIGHT_FIDELITY_ORDER)
    )
    if runtime == "cpu" and requested == "auto" and path_pin is None:
        candidates = (primary,)

    free_bytes: int | None = None
    total_bytes: int | None = None
    if runtime == "native-cuda":
        free_bytes, total_bytes = memory_probe(config)
    reasons: dict[str, dict[str, Any]] = {}
    selected: str | None = None
    selected_required = 0
    selected_reserve = 0
    for profile in candidates:
        if profile not in variants:
            reasons[str(profile)] = {"status": "missing-variant"}
            continue
        descriptor = variants[str(profile)]
        path = _contained_checkpoint_path(
            artifact_root,
            descriptor,
            field=f"checkpoint_variants.{profile}",
        )
        if not path.is_file():
            reasons[str(profile)] = {"status": "missing-file", "path": str(path)}
            continue
        incompatible_companion = next(
            (
                name
                for name in requested_companions
                if descriptor["target_sha256"]
                not in companions[name]["target_compatibility"][
                    "allowed_target_checkpoint_sha256"
                ]
            ),
            None,
        )
        if incompatible_companion is not None:
            reasons[str(profile)] = {
                "status": f"{incompatible_companion}-target-digest-mismatch",
                "target_sha256": descriptor["target_sha256"],
            }
            continue
        kernel_profile = descriptor["required_kernel_profile"]
        if kernel_profile not in kernel_profiles:
            reasons[str(profile)] = {
                "status": "missing-kernel-profile",
                "required_kernel_profile": kernel_profile,
            }
            continue
        if runtime == "native-cuda":
            assert free_bytes is not None and total_bytes is not None
            reserve = (
                config.vram_reserve_bytes
                if config.vram_reserve_bytes is not None
                else max(2 * _GIB, (total_bytes + 9) // 10)
            )
            runtime_bytes = _variant_runtime_bytes(
                descriptor,
                context_tokens=context_tokens,
                session_count=config.session_count,
                companion_bytes=companion_bytes,
            )
            required = runtime_bytes + reserve
            minimum_total = int(descriptor["memory_profile"]["minimum_total_vram_bytes"])
            if total_bytes < minimum_total or free_bytes < required:
                reasons[str(profile)] = {
                    "status": "insufficient-vram",
                    "runtime_bytes": runtime_bytes,
                    "reserve_bytes": reserve,
                    "required_free_vram_bytes": required,
                    "minimum_total_vram_bytes": minimum_total,
                }
                continue
            selected_required = required
            selected_reserve = reserve
        selected = str(profile)
        reasons[selected] = {"status": "selected"}
        break
    if selected is None:
        explicit = requested != "auto" or path_pin is not None
        prefix = "Explicit weight precision cannot be loaded" if explicit else "No weight precision fits"
        raise NativeInferenceCapabilityError(
            f"{prefix}; free={free_bytes}, total={total_bytes}, candidates={reasons}"
        )

    descriptor = variants[selected]
    speculation_reason = "disabled"
    if "dflash" in requested_companions:
        speculation_reason = "required-or-explicit"
    elif optional_dflash:
        dflash_path = _contained_checkpoint_path(
            artifact_root,
            companions["dflash"],
            field="companion_checkpoints.dflash",
        )
        allowed_targets = companions["dflash"]["target_compatibility"][
            "allowed_target_checkpoint_sha256"
        ]
        if descriptor["target_sha256"] not in allowed_targets:
            speculation_reason = "auto-target-digest-mismatch"
        elif dflash_path.is_file():
            if runtime == "cpu":
                requested_companions.append("dflash")
                speculation_reason = "auto-available-cpu"
            else:
                assert free_bytes is not None and total_bytes is not None
                optional_bytes = int(companions["dflash"]["resident_weight_bytes"])
                runtime_bytes = _variant_runtime_bytes(
                    descriptor,
                    context_tokens=context_tokens,
                    session_count=config.session_count,
                    companion_bytes=companion_bytes + optional_bytes,
                )
                required_with_dflash = runtime_bytes + selected_reserve
                if free_bytes >= required_with_dflash:
                    requested_companions.append("dflash")
                    selected_required = required_with_dflash
                    speculation_reason = "auto-fits-without-target-downgrade"
                else:
                    speculation_reason = "auto-skipped-to-preserve-target-precision"
        else:
            speculation_reason = "auto-companion-file-missing"
    effective = dict(manifest)
    effective["checkpoint"] = {
        field: descriptor[field]
        for field in ("format", "artifact_path", "target_nbytes", "target_sha256")
    }
    selection_kind = (
        "path-pin"
        if path_pin is not None
        else ("explicit" if requested != "auto" else ("auto-vram" if runtime == "native-cuda" else "auto-primary"))
    )
    proof = {
        "requested_weight_precision": requested,
        "effective_weight_precision": selected,
        "weight_precision_selection": selection_kind,
        "selected_artifact_sha256": descriptor["target_sha256"],
        "required_kernel_profile": descriptor["required_kernel_profile"],
        "cuda_device": config.cuda_device if runtime == "native-cuda" else None,
        "cuda_free_bytes_at_selection": free_bytes,
        "cuda_total_bytes_at_selection": total_bytes,
        "required_free_vram_bytes": selected_required if runtime == "native-cuda" else None,
        "vram_reserve_bytes": selected_reserve if runtime == "native-cuda" else None,
        "context_tokens": context_tokens,
        "session_count": config.session_count,
        "companion_checkpoints": list(requested_companions),
        "requested_speculative_decoding": config.speculative_decoding,
        "effective_speculative_decoding": (
            "dflash" if "dflash" in requested_companions else "off"
        ),
        "speculative_decoding_selection": speculation_reason,
        "candidates": reasons,
        "runtime": runtime,
    }
    effective["selected_weight_precision"] = selected
    effective["weight_precision_selection_proof"] = proof
    return effective, proof


def _resolve_binding(binding: Any | None) -> Any:
    if binding is None:
        try:
            binding = importlib.import_module(_DEFAULT_BINDING_MODULE)
        except ModuleNotFoundError as exc:
            if exc.name != _DEFAULT_BINDING_MODULE:
                raise
            raise NativeInferenceCapabilityError(
                "The in-process resident inference binding is unavailable. "
                "No subprocess fallback is permitted."
            ) from exc

    required = (
        "resident_inference_abi_version",
        "resident_inference_capabilities",
        "load_model",
        "close_model",
        "create_session",
        "close_session",
        "prefill",
        "decode_one",
        "truncate_session",
        "reset_session",
        "cancel_session",
        "model_stats",
        "session_stats",
    )
    missing = [name for name in required if not callable(getattr(binding, name, None))]
    if missing:
        raise NativeInferenceCapabilityError(
            "Resident inference binding is missing required in-process operations: "
            + ", ".join(missing)
        )
    abi = binding.resident_inference_abi_version()
    if not _is_exact_version(abi, RESIDENT_INFERENCE_ABI_VERSION):
        raise NativeInferenceCapabilityError(
            "Resident inference binding ABI mismatch: "
            f"expected {RESIDENT_INFERENCE_ABI_VERSION}, got {abi!r}"
        )
    return binding


def _prove_capabilities(
    manifest: Mapping[str, Any],
    binding: Any,
) -> NativeInferenceCapabilities:
    raw_artifact_caps = manifest.get("capabilities")
    if not isinstance(raw_artifact_caps, Mapping):
        raise NativeInferenceCapabilityError("Artifact capabilities must be an object")
    artifact_native = _require_bool(raw_artifact_caps, "native_inference", source="Artifact")
    artifact_resident = _require_bool(raw_artifact_caps, "resident_inference", source="Artifact")
    if not artifact_native or not artifact_resident:
        raise NativeInferenceCapabilityError(
            "Artifact does not prove native resident inference; migrate/rebuild it with a real adapter"
        )

    kernel_abi = manifest.get("kernel_abi")
    resident_abi = kernel_abi.get("resident_inference") if isinstance(kernel_abi, Mapping) else None
    if not isinstance(resident_abi, Mapping):
        raise NativeInferenceCapabilityError("Artifact does not declare the resident inference ABI")
    if (
        not _is_exact_version(
            resident_abi.get("version"), RESIDENT_INFERENCE_ABI_VERSION
        )
        or resident_abi.get("status") != "ready"
    ):
        raise NativeInferenceCapabilityError(
            "Artifact resident inference ABI is not proven ready at version "
            f"{RESIDENT_INFERENCE_ABI_VERSION}"
        )

    artifact_turboquant = raw_artifact_caps.get("turboquant_kv_cache", False)
    if not isinstance(artifact_turboquant, bool):
        raise NativeInferenceCapabilityError(
            "Artifact must declare boolean capability 'turboquant_kv_cache'"
        )
    if artifact_turboquant:
        turboquant_abi = kernel_abi.get("turboquant_cache")
        if not isinstance(turboquant_abi, Mapping) or (
            not _is_exact_version(turboquant_abi.get("version"), 1)
            or turboquant_abi.get("status") != "ready"
        ):
            raise NativeInferenceCapabilityError(
                "Artifact TurboQuant capability is not backed by cache ABI version 1"
            )

    raw_binding_caps = binding.resident_inference_capabilities()
    if not isinstance(raw_binding_caps, Mapping):
        raise NativeInferenceCapabilityError("Binding capabilities must be an object")
    binding_native = _require_bool(raw_binding_caps, "native_inference", source="Binding")
    binding_resident = _require_bool(raw_binding_caps, "resident_inference", source="Binding")
    if not binding_native or not binding_resident:
        raise NativeInferenceCapabilityError(
            "Binding does not prove an in-process resident inference implementation"
        )

    def jointly_proven(key: str) -> bool:
        artifact_value = raw_artifact_caps.get(key, False)
        binding_value = raw_binding_caps.get(key, False)
        if not isinstance(artifact_value, bool) or not isinstance(binding_value, bool):
            raise NativeInferenceCapabilityError(
                f"Artifact and binding must declare boolean capability {key!r}"
            )
        return artifact_value and binding_value

    tile_attention = jointly_proven("turboquant_tile_attention")
    tile_feature_abi = (
        kernel_abi.get("turboquant_tile_attention")
        if isinstance(kernel_abi, Mapping)
        else None
    )
    tile_attention = bool(
        tile_attention
        and isinstance(tile_feature_abi, Mapping)
        and _is_exact_version(tile_feature_abi.get("version"), 1)
        and tile_feature_abi.get("status") == "ready"
        and tile_feature_abi.get("symbol")
        == "nfn_native_tile_turboquant_attention_forward_v1"
        and callable(getattr(binding, "configure_model_turboquant_attention", None))
    )

    artifact_structured = raw_artifact_caps.get("structured_output", False)
    artifact_function_tools = raw_artifact_caps.get("function_tools", False)
    if not isinstance(artifact_structured, bool) or not isinstance(
        artifact_function_tools, bool
    ):
        raise NativeInferenceCapabilityError(
            "Artifact must declare boolean function_tools and structured_output capabilities"
        )
    if artifact_function_tools and not artifact_structured:
        raise NativeInferenceCapabilityError(
            "Artifact function_tools requires the structured_output capability"
        )
    binding_constrained = raw_binding_caps.get(
        STRUCTURED_OUTPUT_TOKEN_SELECTION,
        False,
    )
    if not isinstance(binding_constrained, bool):
        raise NativeInferenceCapabilityError(
            "Binding must declare boolean current_logits_exact_prefill capability"
        )
    structured_abi = kernel_abi.get("structured_output")
    structured_output = bool(
        artifact_structured
        and binding_constrained
        and callable(getattr(binding, "current_logits", None))
        and isinstance(structured_abi, Mapping)
        and _is_exact_version(structured_abi.get("version"), 1)
        and structured_abi.get("status") == "ready"
        and structured_abi.get("profile") == STRUCTURED_OUTPUT_PROFILE
        and structured_abi.get("token_selection")
        == STRUCTURED_OUTPUT_TOKEN_SELECTION
    )
    function_abi = kernel_abi.get("function_tools")
    function_tools = bool(
        artifact_function_tools
        and structured_output
        and isinstance(function_abi, Mapping)
        and _is_exact_version(function_abi.get("version"), 1)
        and function_abi.get("status") == "ready"
        and function_abi.get("profile") == FUNCTION_TOOL_TEMPLATE_PROFILE
        and function_abi.get("structured_output_profile")
        == STRUCTURED_OUTPUT_PROFILE
    )

    prefix_cow_abi = kernel_abi.get("session_prefix_cow")
    checkpoint = manifest.get("checkpoint")
    checkpoint_format = (
        checkpoint.get("format") if isinstance(checkpoint, Mapping) else None
    )
    expected_prefix_cow_profile = (
        _SESSION_PREFIX_COW_PROFILES_BY_CHECKPOINT_FORMAT.get(checkpoint_format)
        if isinstance(checkpoint_format, str)
        else None
    )
    binding_prefix_cow_abi = raw_binding_caps.get("session_prefix_cow_abi")
    binding_prefix_cow_profiles = (
        binding_prefix_cow_abi.get("profiles")
        if isinstance(binding_prefix_cow_abi, Mapping)
        else None
    )
    binding_prefix_cow_base_ready = bool(
        isinstance(binding_prefix_cow_abi, Mapping)
        and _is_exact_version(binding_prefix_cow_abi.get("version"), 1)
        and binding_prefix_cow_abi.get("operation") == "fork_session"
        and isinstance(binding_prefix_cow_profiles, Sequence)
        and not isinstance(binding_prefix_cow_profiles, (str, bytes))
        and all(isinstance(profile, str) for profile in binding_prefix_cow_profiles)
        and len(set(binding_prefix_cow_profiles)) == len(binding_prefix_cow_profiles)
    )
    binding_prefix_cow_ready = bool(
        binding_prefix_cow_base_ready
        and expected_prefix_cow_profile in binding_prefix_cow_profiles
    )
    session_prefix_cow = bool(
        jointly_proven("session_prefix_cow")
        and binding_prefix_cow_ready
        and callable(getattr(binding, "fork_session", None))
        and isinstance(prefix_cow_abi, Mapping)
        and _is_exact_version(prefix_cow_abi.get("version"), 1)
        and prefix_cow_abi.get("status") == "ready"
        and expected_prefix_cow_profile is not None
        and prefix_cow_abi.get("profile") == expected_prefix_cow_profile
        and prefix_cow_abi.get("operation") == "fork_session"
    )
    turboquant_prefix_cow_abi = kernel_abi.get(
        "session_prefix_cow_cpu_turboquant"
    )
    expected_turboquant_prefix_cow_profile = (
        CPU_TURBOQUANT_SESSION_PREFIX_COW_PROFILE
        if checkpoint_format == "neuralfn.native_dense_gpt.v5"
        else None
    )
    session_prefix_cow_cpu_turboquant = bool(
        jointly_proven("session_prefix_cow_cpu_turboquant")
        and binding_prefix_cow_base_ready
        and expected_turboquant_prefix_cow_profile in binding_prefix_cow_profiles
        and callable(getattr(binding, "fork_session", None))
        and isinstance(turboquant_prefix_cow_abi, Mapping)
        and _is_exact_version(turboquant_prefix_cow_abi.get("version"), 1)
        and turboquant_prefix_cow_abi.get("status") == "ready"
        and turboquant_prefix_cow_abi.get("profile")
        == expected_turboquant_prefix_cow_profile
        and turboquant_prefix_cow_abi.get("operation") == "fork_session"
        and turboquant_prefix_cow_abi.get("backend") == "cpu-reference-packed"
    )

    speculative_abi = kernel_abi.get("speculative_decoding")
    binding_speculative_abi = raw_binding_caps.get("speculative_decoding_abi")
    speculative_decoding = bool(
        jointly_proven("speculative_decoding")
        and isinstance(speculative_abi, Mapping)
        and _is_exact_version(speculative_abi.get("version"), 1)
        and speculative_abi.get("status") == "ready"
        and speculative_abi.get("profile") == "muse-glimmer-dflash-block16-v1"
        and speculative_abi.get("load_operation") == "load_companion"
        and speculative_abi.get("decode_operation") == "decode_speculative_block"
        and isinstance(binding_speculative_abi, Mapping)
        and _is_exact_version(binding_speculative_abi.get("version"), 1)
        and binding_speculative_abi.get("load_operation") == "load_companion"
        and binding_speculative_abi.get("decode_operation") == "decode_speculative_block"
        and binding_speculative_abi.get("block_size") == 16
        and binding_speculative_abi.get("proposal_tokens") == 15
        and callable(getattr(binding, "load_companion", None))
        and callable(getattr(binding, "decode_speculative_block", None))
    )
    binding_dflash_cpu = raw_binding_caps.get("dflash_cpu", False)
    binding_dflash_cuda = raw_binding_caps.get("dflash_cuda", False)
    if not isinstance(binding_dflash_cpu, bool) or not isinstance(binding_dflash_cuda, bool):
        raise NativeInferenceCapabilityError(
            "Binding dflash_cpu/dflash_cuda capabilities must be booleans"
        )

    media_abi = kernel_abi.get("media_encoder")
    binding_media_abi = raw_binding_caps.get("media_encoder_abi")
    vision = bool(
        jointly_proven("vision")
        and isinstance(media_abi, Mapping)
        and _is_exact_version(media_abi.get("version"), 1)
        and media_abi.get("status") == "ready"
        and media_abi.get("profile") == "muse-glimmer-vision-packed-patches-v1"
        and media_abi.get("encode_operation") == "encode_media"
        and media_abi.get("prefill_operation") == "prefill_with_embeddings"
        and media_abi.get("projection_width") == 6_656
        and isinstance(binding_media_abi, Mapping)
        and _is_exact_version(binding_media_abi.get("version"), 1)
        and binding_media_abi.get("encode_operation") == "encode_media"
        and binding_media_abi.get("prefill_operation") == "prefill_with_embeddings"
        and binding_media_abi.get("projection_width") == 6_656
        and callable(getattr(binding, "encode_media", None))
        and callable(getattr(binding, "prefill_with_embeddings", None))
    )
    video = bool(vision and jointly_proven("video"))
    binding_vision_cpu = raw_binding_caps.get("vision_cpu", False)
    binding_vision_cuda = raw_binding_caps.get("vision_cuda", False)
    if not isinstance(binding_vision_cpu, bool) or not isinstance(binding_vision_cuda, bool):
        raise NativeInferenceCapabilityError(
            "Binding vision_cpu/vision_cuda capabilities must be booleans"
        )
    binding_lora_abi = raw_binding_caps.get("native_lora_abi")
    expected_lora_targets = [
        "q_proj", "k_proj", "v_proj", "o_proj", "attn_gate_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    native_lora = bool(
        raw_binding_caps.get("native_lora") is True
        and isinstance(binding_lora_abi, Mapping)
        and _is_exact_version(binding_lora_abi.get("version"), 1)
        and binding_lora_abi.get("load_operation") == "load_companion"
        and binding_lora_abi.get("format")
        == "neuralfn.native_muse_glimmer_lora.bf16.v1"
        and binding_lora_abi.get("targets") == expected_lora_targets
        and callable(getattr(binding, "load_companion", None))
    )
    binding_lora_cpu = raw_binding_caps.get("native_lora_cpu", False)
    binding_lora_cuda = raw_binding_caps.get("native_lora_cuda", False)
    if not isinstance(binding_lora_cpu, bool) or not isinstance(binding_lora_cuda, bool):
        raise NativeInferenceCapabilityError(
            "Binding native_lora_cpu/native_lora_cuda capabilities must be booleans"
        )

    raw_state_kinds = manifest.get("session_state_kinds", ())
    if not isinstance(raw_state_kinds, Sequence) or isinstance(raw_state_kinds, (str, bytes)):
        raise NativeInferenceCapabilityError("Artifact session_state_kinds must be an array")
    state_kinds = tuple(str(kind) for kind in raw_state_kinds)
    return NativeInferenceCapabilities(
        native_inference=True,
        resident_inference=True,
        lossless_kv_cache=jointly_proven("lossless_kv_cache"),
        turboquant_kv_cache=jointly_proven("turboquant_kv_cache"),
        turboquant_tile_attention=tile_attention,
        session_prefix_cow=session_prefix_cow,
        session_prefix_cow_cpu_turboquant=session_prefix_cow_cpu_turboquant,
        speculative_decoding=speculative_decoding,
        dflash_cpu=speculative_decoding and binding_dflash_cpu,
        dflash_cuda=speculative_decoding and binding_dflash_cuda,
        vision=vision,
        video=video,
        vision_cpu=vision and binding_vision_cpu,
        vision_cuda=vision and binding_vision_cuda,
        native_lora=native_lora,
        native_lora_cpu=native_lora and binding_lora_cpu,
        native_lora_cuda=native_lora and binding_lora_cuda,
        function_tools=function_tools,
        structured_output=structured_output,
        session_state_kinds=state_kinds,
    )


def _effective_cache_mode(
    config: KVCacheConfig,
    capabilities: NativeInferenceCapabilities,
) -> str:
    if config.mode == "off":
        return "off"
    if config.mode in {"auto", "full"}:
        if not capabilities.lossless_kv_cache:
            raise NativeInferenceCapabilityError(
                f"KV cache mode {config.mode!r} requires a proven lossless resident cache"
            )
        return "full"
    if not capabilities.lossless_kv_cache or not capabilities.turboquant_kv_cache:
        raise NativeInferenceCapabilityError(
            "Explicit TurboQuant was requested, but the artifact and binding do not both prove "
            "lossless-cache and TurboQuant support; no full-cache fallback was selected"
        )
    return "turboquant"


def _tile_attention_config_payload(config: KVCacheConfig) -> dict[str, Any] | None:
    if config.turboquant_attention_backend == "cpu":
        return None
    assert config.tile_ops_lib is not None
    try:
        tile_ops_lib = Path(config.tile_ops_lib).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise NativeInferenceCapabilityError(
            f"Tile-CUDA TurboQuant sidecar does not exist: {config.tile_ops_lib}"
        ) from exc
    if not tile_ops_lib.is_file():
        raise NativeInferenceCapabilityError(
            f"Tile-CUDA TurboQuant sidecar is not a regular file: {tile_ops_lib}"
        )
    payload: dict[str, Any] = {
        "backend": "tile-cuda",
        "tile_ops_lib": str(tile_ops_lib),
        "device": config.cuda_device,
    }
    if config.cuda_runtime_lib is not None:
        payload["cuda_runtime_lib"] = config.cuda_runtime_lib
    return payload


def _validate_tile_attention_contract(
    manifest: Mapping[str, Any],
    binding: Any,
    capabilities: NativeInferenceCapabilities,
    config: KVCacheConfig,
) -> dict[str, Any] | None:
    payload = _tile_attention_config_payload(config)
    if payload is None:
        return None
    if not capabilities.turboquant_tile_attention:
        raise NativeInferenceCapabilityError(
            "Tile-CUDA TurboQuant attention was requested, but the artifact and "
            "binding do not both prove that feature; no CPU fallback was selected"
        )
    kernel_abi = manifest.get("kernel_abi")
    feature_abi = (
        kernel_abi.get("turboquant_tile_attention")
        if isinstance(kernel_abi, Mapping)
        else None
    )
    if not isinstance(feature_abi, Mapping) or (
        not _is_exact_version(feature_abi.get("version"), 1)
        or feature_abi.get("status") != "ready"
        or feature_abi.get("symbol")
        != "nfn_native_tile_turboquant_attention_forward_v1"
    ):
        raise NativeInferenceCapabilityError(
            "Artifact Tile-CUDA TurboQuant attention ABI is not proven ready at version 1"
        )
    if not callable(getattr(binding, "configure_model_turboquant_attention", None)):
        raise NativeInferenceCapabilityError(
            "Resident binding does not expose Tile-CUDA TurboQuant model configuration"
        )
    return payload


def _turboquant_head_dimension(*, channels: Any, num_heads: Any) -> int:
    """Validate model-specific 3.5-bit geometry before advertising it."""

    if (
        isinstance(channels, bool)
        or not isinstance(channels, int)
        or isinstance(num_heads, bool)
        or not isinstance(num_heads, int)
        or channels <= 0
        or num_heads <= 0
        or channels % num_heads != 0
    ):
        raise NativeInferenceCapabilityError(
            "TurboQuant requires valid resident channels/num_heads geometry"
        )
    dimension = channels // num_heads
    if dimension < 2 or dimension % 2:
        raise NativeInferenceCapabilityError(
            "TurboQuant 3.5-bit profiles require an even attention head dimension >= 2"
        )
    return dimension


def _turboquant_binding_tables(
    *,
    channels: int,
    num_heads: int,
    profile: str,
) -> dict[str, Any]:
    """Build deterministic shared tables for the native compressed cache.

    The resident binding owns all row encoding and attention work. Python only
    materializes the same model-level rotation/codebook metadata used by the
    portable correctness oracle so the two implementations have one format
    contract and no duplicated random-number implementation.
    """

    dimension = _turboquant_head_dimension(channels=channels, num_heads=num_heads)
    from .turboquant import TurboQuantReferenceCodec, lloyd_max_centroids

    codec = TurboQuantReferenceCodec(
        dimension,
        profile=profile,
        seed=0,
        outlier_indices=range(0, dimension, 2),
    )
    used_widths = sorted(set(codec.value_bit_widths) | set(codec.key_bit_widths))
    centroids: list[list[float]] = [[] for _ in range(5)]
    for width in used_widths:
        centroids[width] = list(lloyd_max_centroids(dimension, width))
    return {
        "dimension": dimension,
        "rotation": [value for row in codec.rotation for value in row],
        "qjl_projection": (
            [value for row in codec.qjl_projection for value in row]
            if codec.qjl_projection is not None
            else []
        ),
        "value_bit_widths": list(codec.value_bit_widths),
        "key_bit_widths": list(codec.key_bit_widths),
        "centroids": centroids,
    }


class NativeInferenceModel:
    """One immutable model loaded once into a proven in-process binding."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        manifest_path: Path,
        manifest: dict[str, Any],
        binding: Any,
        handle: Any,
        capabilities: NativeInferenceCapabilities,
        kv_cache: KVCacheConfig,
        model_load: NativeModelLoadConfig,
        weight_selection: Mapping[str, Any],
        effective_cache_mode: str,
        turboquant_tables: Mapping[str, dict[str, Any]] | None = None,
        tile_attention_config: Mapping[str, Any] | None = None,
    ) -> None:
        self._artifact_root = artifact_root
        self._manifest_path = manifest_path
        self._manifest = manifest
        self._binding = binding
        self._handle = handle
        self._capabilities = capabilities
        self._kv_cache = kv_cache
        self._model_load = model_load
        self._weight_selection = dict(weight_selection)
        self._effective_cache_mode = effective_cache_mode
        self._sessions: set[NativeInferenceSession] = set()
        self._turboquant_tables = dict(turboquant_tables or {})
        self._tile_attention_config = (
            dict(tile_attention_config) if tile_attention_config is not None else None
        )
        self._closing = False
        self._closed = False
        self._lock = threading.RLock()
        self._compute_lock = threading.Lock()

    @classmethod
    def load(
        cls,
        artifact: str | Path,
        *,
        binding: Any | None = None,
        kv_cache: KVCacheConfig | None = None,
        load_config: NativeModelLoadConfig | None = None,
    ) -> "NativeInferenceModel":
        """Load one artifact after all manifest and binding gates pass."""

        artifact_root, manifest_path, manifest, path_pin = _load_manifest_with_path_pin(artifact)
        resolved_binding = _resolve_binding(binding)
        model_load = load_config or NativeModelLoadConfig()
        effective_manifest, weight_selection = _select_checkpoint_variant(
            artifact_root,
            manifest,
            resolved_binding,
            model_load,
            path_pin=path_pin,
        )
        capabilities = _prove_capabilities(effective_manifest, resolved_binding)
        cache_config = kv_cache or KVCacheConfig()
        effective_cache_mode = _effective_cache_mode(cache_config, capabilities)
        selected_companions = tuple(weight_selection.get("companion_checkpoints", ()))
        if "dflash" in selected_companions:
            if not capabilities.speculative_decoding:
                raise NativeInferenceCapabilityError(
                    "DFlash was selected but artifact/binding feature ABI 1 is not jointly proven"
                )
            if effective_cache_mode != "full":
                raise NativeInferenceCapabilityError(
                    "DFlash speculative decoding requires the full lossless cache mode"
                )
            if (
                weight_selection.get("runtime") == "native-cuda"
                and not capabilities.dflash_cuda
            ):
                raise NativeInferenceCapabilityError(
                    "Whole-model CUDA DFlash was selected but the binding does not "
                    "prove device-resident assistant execution"
                )
        if "mmproj" in selected_companions:
            if not capabilities.vision:
                raise NativeInferenceCapabilityError(
                    "mmproj was selected but artifact/binding media feature ABI 1 is not jointly proven"
                )
            if weight_selection.get("runtime") == "native-cuda" and not capabilities.vision_cuda:
                raise NativeInferenceCapabilityError(
                    "Whole-model CUDA mmproj was selected but the binding does not "
                    "prove device-resident vision execution"
                )
        if "lora" in selected_companions:
            if not capabilities.native_lora:
                raise NativeInferenceCapabilityError(
                    "Native LoRA was selected but the binding adapter ABI 1 is not proven"
                )
            if weight_selection.get("runtime") == "native-cuda" and not capabilities.native_lora_cuda:
                raise NativeInferenceCapabilityError(
                    "Whole-model CUDA native LoRA was selected but the binding does not "
                    "prove device-resident adapter execution"
                )
            if weight_selection.get("runtime") == "cpu" and not capabilities.native_lora_cpu:
                raise NativeInferenceCapabilityError(
                    "CPU native LoRA was selected but the binding does not prove adapter execution"
                )
        tile_attention_config = _validate_tile_attention_contract(
            effective_manifest,
            resolved_binding,
            capabilities,
            cache_config,
        )
        _validate_checkpoint_artifact(artifact_root, effective_manifest)
        if weight_selection.get("runtime") == "native-cuda":
            load_with_options = getattr(resolved_binding, "load_model_with_options", None)
            if not callable(load_with_options):
                raise NativeInferenceCapabilityError(
                    "Whole-model CUDA selection requires load_model_with_options"
                )
            tile_ops_lib = _resolve_model_tile_ops_library(model_load)
            handle = load_with_options(
                str(artifact_root),
                effective_manifest,
                {
                    "cuda_device": model_load.cuda_device,
                    "tile_ops_lib": tile_ops_lib,
                    "cuda_runtime_lib": model_load.cuda_runtime_lib,
                    "weight_precision": weight_selection["effective_weight_precision"],
                    "selection_proof": weight_selection,
                },
            )
        else:
            handle = resolved_binding.load_model(
                str(artifact_root),
                effective_manifest,
            )
        if handle is None:
            raise NativeInferenceError("Resident binding returned no model handle")
        initial_turboquant_tables: dict[str, dict[str, Any]] = {}
        try:
            for companion_name in selected_companions:
                descriptor = effective_manifest.get("companion_checkpoints", {}).get(
                    companion_name
                )
                if not isinstance(descriptor, Mapping):
                    raise NativeInferenceCapabilityError(
                        f"Selected companion descriptor is missing: {companion_name}"
                    )
                confirmation = resolved_binding.load_companion(
                    handle,
                    str(artifact_root),
                    dict(descriptor),
                )
                if (
                    not isinstance(confirmation, Mapping)
                    or confirmation.get("loaded") is not True
                    or confirmation.get("component") != companion_name
                ):
                    raise NativeInferenceCapabilityError(
                        f"Resident binding did not confirm {companion_name} companion loading"
                    )
                if (
                    weight_selection.get("runtime") == "native-cuda"
                    and confirmation.get("whole_model_cuda") is not True
                ):
                    raise NativeInferenceCapabilityError(
                        f"Resident binding loaded {companion_name} without whole-model CUDA execution"
                    )
            if tile_attention_config is not None:
                configured = resolved_binding.configure_model_turboquant_attention(
                    handle,
                    tile_attention_config,
                )
                if (
                    not isinstance(configured, Mapping)
                    or configured.get("configured") is not True
                    or configured.get("backend") != "tile-cuda"
                ):
                    raise NativeInferenceCapabilityError(
                        "Resident binding did not confirm Tile-CUDA TurboQuant configuration"
                    )
            if capabilities.turboquant_kv_cache:
                raw_stats = resolved_binding.model_stats(handle)
                if not isinstance(raw_stats, Mapping):
                    raise NativeInferenceCapabilityError(
                        "Resident binding must expose model geometry for TurboQuant"
                    )
                _turboquant_head_dimension(
                    channels=raw_stats.get("channels"),
                    num_heads=raw_stats.get("num_heads"),
                )
                if effective_cache_mode == "turboquant":
                    initial_turboquant_tables[cache_config.turboquant_profile] = (
                        _turboquant_binding_tables(
                            channels=raw_stats.get("channels"),
                            num_heads=raw_stats.get("num_heads"),
                            profile=cache_config.turboquant_profile,
                        )
                    )
        except BaseException:
            resolved_binding.close_model(handle)
            raise
        return cls(
            artifact_root=artifact_root,
            manifest_path=manifest_path,
            manifest=effective_manifest,
            binding=resolved_binding,
            handle=handle,
            capabilities=capabilities,
            kv_cache=cache_config,
            model_load=model_load,
            weight_selection=weight_selection,
            effective_cache_mode=effective_cache_mode,
            turboquant_tables=initial_turboquant_tables,
            tile_attention_config=tile_attention_config,
        )

    @property
    def capabilities(self) -> NativeInferenceCapabilities:
        return self._capabilities

    @property
    def artifact_root(self) -> Path:
        return self._artifact_root

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def _ensure_open(self) -> None:
        if self._closing or self._closed:
            raise NativeInferenceClosedError("Native inference model is closed")

    def _ensure_session_operation_admitted(self) -> None:
        """Linearize new session work against the model close boundary."""

        with self._lock:
            self._ensure_open()

    def _register_session_handle(
        self,
        *,
        session_handle: Any,
        seed: int,
        kv_cache: KVCacheConfig,
        effective_cache_mode: str,
        initial_tokens: Sequence[int] = (),
    ) -> "NativeInferenceSession":
        """Construct and register one native handle while ``self._lock`` is held."""

        session: NativeInferenceSession | None = None
        try:
            session = NativeInferenceSession(
                model=self,
                handle=session_handle,
                seed=seed,
                kv_cache=kv_cache,
                effective_cache_mode=effective_cache_mode,
                initial_tokens=initial_tokens,
            )
            self._sessions.add(session)
        except BaseException:
            if session is not None:
                self._sessions.discard(session)
            self._binding.close_session(self._handle, session_handle)
            raise
        return session

    def create_session(
        self,
        *,
        seed: int = 0,
        kv_cache: KVCacheConfig | None = None,
    ) -> "NativeInferenceSession":
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        cache_config = kv_cache or self._kv_cache
        effective_cache_mode = _effective_cache_mode(cache_config, self._capabilities)
        with self._lock:
            self._ensure_open()
            requested_tile_config = _validate_tile_attention_contract(
                self._manifest,
                self._binding,
                self._capabilities,
                cache_config,
            )
            if requested_tile_config is not None:
                if self._tile_attention_config is None:
                    configured = self._binding.configure_model_turboquant_attention(
                        self._handle,
                        requested_tile_config,
                    )
                    if (
                        not isinstance(configured, Mapping)
                        or configured.get("configured") is not True
                        or configured.get("backend") != "tile-cuda"
                    ):
                        raise NativeInferenceCapabilityError(
                            "Resident binding did not confirm Tile-CUDA TurboQuant configuration"
                        )
                    self._tile_attention_config = dict(requested_tile_config)
                elif self._tile_attention_config != requested_tile_config:
                    raise NativeInferenceCapabilityError(
                        "A resident model cannot mix different Tile-CUDA sidecar/runtime/device "
                        "configurations across sessions"
                    )
            cache_payload: dict[str, Any] = {
                "mode": cache_config.mode,
                "turboquant_profile": cache_config.turboquant_profile,
                "effective_mode": effective_cache_mode,
            }
            if requested_tile_config is not None:
                cache_payload["turboquant_attention_backend"] = "tile-cuda"
            if effective_cache_mode == "turboquant":
                raw_stats = self._binding.model_stats(self._handle)
                if not isinstance(raw_stats, Mapping):
                    raise NativeInferenceCapabilityError(
                        "Resident binding must expose model geometry for TurboQuant"
                    )
                tables = self._turboquant_tables.get(cache_config.turboquant_profile)
                if tables is None:
                    tables = _turboquant_binding_tables(
                        channels=raw_stats.get("channels"),
                        num_heads=raw_stats.get("num_heads"),
                        profile=cache_config.turboquant_profile,
                    )
                    self._turboquant_tables[cache_config.turboquant_profile] = tables
                cache_payload["tables"] = tables
            session_handle = self._binding.create_session(
                self._handle,
                {
                    "seed": seed,
                    "kv_cache": cache_payload,
                },
            )
            if session_handle is None:
                raise NativeInferenceError("Resident binding returned no session handle")
            return self._register_session_handle(
                session_handle=session_handle,
                seed=seed,
                kv_cache=cache_config,
                effective_cache_mode=effective_cache_mode,
            )

    def fork_session(
        self,
        source: "NativeInferenceSession",
        *,
        token_count: int | None = None,
        seed: int = 0,
    ) -> "NativeInferenceSession":
        """Fork one non-empty supported native-cache prefix with native COW storage.

        The child receives independent token, RNG, cancellation, and lifecycle
        state.  Native K/V and final-hidden storage remains shared until either
        owner next appends a token, at which point that writer detaches first.
        """

        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if token_count is not None and (
            isinstance(token_count, bool) or not isinstance(token_count, int)
        ):
            raise TypeError("token_count must be an integer or None")
        self._require_owned_session(source)
        with source._operation_lock:
            with source._lock:
                source._ensure_usable()
                source_cache_mode = source._effective_cache_mode
                if source_cache_mode == "full":
                    if not self._capabilities.session_prefix_cow:
                        raise NativeInferenceCapabilityError(
                            "Artifact and binding do not prove a jointly compatible exact full-cache session prefix COW ABI v1 profile"
                        )
                elif source_cache_mode == "turboquant":
                    if source._kv_cache.turboquant_attention_backend != "cpu":
                        raise NativeInferenceCapabilityError(
                            "Session prefix COW rejects Tile-CUDA TurboQuant session storage"
                        )
                    if not self._capabilities.session_prefix_cow_cpu_turboquant:
                        raise NativeInferenceCapabilityError(
                            "Artifact and binding do not prove the exact dense CPU packed-TurboQuant session prefix COW ABI v1 profile"
                        )
                else:
                    raise NativeInferenceCapabilityError(
                        "Session prefix COW requires a lossless full-cache or dense CPU TurboQuant source session"
                    )
                if self._tile_attention_config is not None:
                    raise NativeInferenceCapabilityError(
                        "Session prefix COW rejects models configured for Tile-CUDA TurboQuant attention"
                    )
                source_tokens = tuple(source._tokens)
                selected_count = len(source_tokens) if token_count is None else token_count
                if selected_count <= 0 or selected_count > len(source_tokens):
                    raise ValueError(
                        "token_count must select a non-empty prefix no longer than the source history"
                    )
                cache_config = source._kv_cache

            with self._lock:
                self._ensure_open()
                with self._compute_lock:
                    session_handle = self._binding.fork_session(
                        self._handle,
                        source._handle,
                        {"token_count": selected_count, "seed": seed},
                    )
                if session_handle is None:
                    raise NativeInferenceError(
                        "Resident binding returned no forked session handle"
                    )
                return self._register_session_handle(
                    session_handle=session_handle,
                    seed=seed,
                    kv_cache=cache_config,
                    effective_cache_mode=source_cache_mode,
                    initial_tokens=source_tokens[:selected_count],
                )

    def prefill(self, session: "NativeInferenceSession", token_ids: Sequence[int]) -> dict[str, int]:
        self._require_owned_session(session)
        return session.prefill(token_ids)

    def prefill_with_embeddings(
        self,
        session: "NativeInferenceSession",
        token_ids: Sequence[int],
        *,
        replacement_positions: Sequence[int],
        replacement_embeddings: Sequence[Sequence[float]],
    ) -> dict[str, int]:
        """Prefill Glimmer with exact image/video-token embedding replacements."""

        self._require_owned_session(session)
        return session.prefill_with_embeddings(
            token_ids,
            replacement_positions=replacement_positions,
            replacement_embeddings=replacement_embeddings,
        )

    def encode_media(
        self,
        packed_patches: Sequence[Sequence[float]],
        grid_thw: Sequence[Sequence[int]],
    ) -> tuple[tuple[float, ...], ...]:
        """Encode processor-packed image/video patches into decoder-width rows.

        This is the strict native vision-tower seam. Raw image decoding and
        resizing stay in the pinned processor layer; the resident binding
        consumes only canonical patch rows and ``(temporal, height, width)``
        grids, then returns the rows passed to ``prefill_with_embeddings``.
        """

        if not self._capabilities.vision:
            raise NativeInferenceCapabilityError(
                "Artifact and binding do not jointly prove Muse Glimmer media feature ABI 1"
            )

        encode = getattr(self._binding, "encode_media", None)
        if not callable(encode):
            raise NativeInferenceCapabilityError(
                "Resident binding does not expose the Muse Glimmer media encoder"
            )
        rows: list[tuple[float, ...]] = []
        for index, row in enumerate(packed_patches):
            values = tuple(float(value) for value in row)
            if not values or any(not math.isfinite(value) for value in values):
                raise ValueError(f"packed_patches[{index}] must be a finite nonempty row")
            rows.append(values)
        if not rows or len({len(row) for row in rows}) != 1:
            raise ValueError("packed_patches must be nonempty equal-width rows")
        grids: list[tuple[int, int, int]] = []
        for index, grid in enumerate(grid_thw):
            values = tuple(int(value) for value in grid)
            if len(values) != 3 or any(value <= 0 for value in values):
                raise ValueError(f"grid_thw[{index}] must contain three positive integers")
            grids.append(values)
        if not grids:
            raise ValueError("grid_thw must be nonempty")
        with self._lock:
            self._ensure_open()
            with self._compute_lock:
                payload = encode(
                    self._handle,
                    [value for row in rows for value in row],
                    [value for grid in grids for value in grid],
                )
        if not isinstance(payload, Mapping):
            raise NativeInferenceError("Binding encode_media must return an object")
        output_rows = payload.get("rows")
        width = payload.get("width")
        values = payload.get("values")
        if (
            isinstance(output_rows, bool)
            or not isinstance(output_rows, int)
            or output_rows <= 0
            or isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or not isinstance(values, Sequence)
            or isinstance(values, (str, bytes, bytearray))
            or len(values) != output_rows * width
        ):
            raise NativeInferenceError("Binding encode_media returned malformed geometry")
        converted = tuple(float(value) for value in values)
        if any(not math.isfinite(value) for value in converted):
            raise NativeInferenceError("Binding encode_media returned non-finite values")
        return tuple(
            converted[row * width : (row + 1) * width]
            for row in range(output_rows)
        )

    def encode_images(
        self,
        images: Sequence[Any],
        *,
        max_image_tokens: int = 4_096,
    ) -> Any:
        """Run pinned still-image preprocessing and the resident vision tower."""

        from .native_glimmer_media import prepare_and_encode_images

        return prepare_and_encode_images(
            self,
            images,
            max_image_tokens=max_image_tokens,
        )

    def encode_videos(
        self,
        videos: Sequence[Sequence[Any]],
        *,
        frame_timestamps: Sequence[Sequence[float]] | None = None,
        sampled_fps: float = 2.0,
        max_video_frame_tokens: int = 144,
    ) -> Any:
        """Encode caller-decoded video frames with the pinned temporal ABI."""

        if not self._capabilities.video:
            raise NativeInferenceCapabilityError(
                "Artifact and binding do not jointly prove Muse Glimmer video support"
            )
        from .native_glimmer_media import prepare_and_encode_videos

        return prepare_and_encode_videos(
            self,
            videos,
            frame_timestamps=frame_timestamps,
            sampled_fps=sampled_fps,
            max_video_frame_tokens=max_video_frame_tokens,
        )

    def decode(
        self,
        session: "NativeInferenceSession",
        config: GenerationConfig | None = None,
        *,
        on_token: TokenCallback | None = None,
    ) -> GenerationResult:
        self._require_owned_session(session)
        return session.decode(config, on_token=on_token)

    def current_logits(self, session: "NativeInferenceSession") -> tuple[float, ...]:
        """Return the logits for the session's current non-empty token prefix."""

        self._require_owned_session(session)
        return session.current_logits()

    def truncate(self, session: "NativeInferenceSession", token_count: int) -> None:
        self._require_owned_session(session)
        session.truncate(token_count)

    def reset(self, session: "NativeInferenceSession") -> None:
        self._require_owned_session(session)
        session.reset()

    def cancel(self, session: "NativeInferenceSession") -> None:
        self._require_owned_session(session)
        session.cancel()

    def _require_owned_session(self, session: "NativeInferenceSession") -> None:
        if not isinstance(session, NativeInferenceSession) or session._model is not self:
            raise ValueError("Session does not belong to this native inference model")

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            binding_stats = self._binding.model_stats(self._handle)
            if not isinstance(binding_stats, Mapping):
                raise NativeInferenceError("Binding model_stats must return an object")
            return {
                **dict(binding_stats),
                "artifact": str(self._artifact_root),
                "resident_inference_abi": RESIDENT_INFERENCE_ABI_VERSION,
                "resident_inference": self._capabilities.resident_inference,
                "lossless_kv_cache": self._capabilities.lossless_kv_cache,
                "turboquant_kv_cache": self._capabilities.turboquant_kv_cache,
                "session_prefix_cow": self._capabilities.session_prefix_cow,
                "session_prefix_cow_cpu_turboquant": (
                    self._capabilities.session_prefix_cow_cpu_turboquant
                ),
                "speculative_decoding": self._capabilities.speculative_decoding,
                "dflash_cpu": self._capabilities.dflash_cpu,
                "dflash_cuda": self._capabilities.dflash_cuda,
                "vision": self._capabilities.vision,
                "video": self._capabilities.video,
                "vision_cpu": self._capabilities.vision_cpu,
                "vision_cuda": self._capabilities.vision_cuda,
                "native_lora": self._capabilities.native_lora,
                "native_lora_cpu": self._capabilities.native_lora_cpu,
                "native_lora_cuda": self._capabilities.native_lora_cuda,
                "open_sessions": sum(not session.closed for session in self._sessions),
                "requested_cache": self._kv_cache.mode,
                "effective_cache": self._effective_cache_mode,
                **self._weight_selection,
            }

    def _discard_session(self, session: "NativeInferenceSession") -> None:
        with self._lock:
            self._sessions.discard(session)

    def close(self) -> None:
        """Close registered sessions and then the model exactly once.

        The caller that publishes ``_closing`` owns teardown and receives its
        first error. Concurrent or reentrant callers return without waiting:
        they may hold a session operation lock that the owner needs, so waiting
        for the owner here would introduce a close-vs-operation deadlock.
        """

        with self._lock:
            if self._closed:
                return
            if self._closing:
                return
            # Creation/forking holds this same lock through native handle
            # creation and Python registration.  Publish the closing state
            # before taking the snapshot so no later session can miss it.
            self._closing = True
            sessions = tuple(self._sessions)
        first_error: BaseException | None = None
        for session in sessions:
            try:
                session.close()
            except BaseException as exc:  # close the model even if one session close fails
                if first_error is None:
                    first_error = exc
        with self._lock:
            if not self._closed:
                try:
                    self._binding.close_model(self._handle)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                finally:
                    self._closed = True
                    self._closing = False
                    self._sessions.clear()
                    self._turboquant_tables.clear()
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "NativeInferenceModel":
        self._ensure_open()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


class NativeInferenceSession:
    """Isolated mutable token/cache/RNG state owned by one resident model."""

    def __init__(
        self,
        *,
        model: NativeInferenceModel,
        handle: Any,
        seed: int,
        kv_cache: KVCacheConfig,
        effective_cache_mode: str,
        initial_tokens: Sequence[int] = (),
    ) -> None:
        self._model = model
        self._handle = handle
        self._seed = seed
        self._kv_cache = kv_cache
        self._effective_cache_mode = effective_cache_mode
        self._tokens: list[int] = list(initial_tokens)
        self._cancelled = False
        self._closed = False
        self._poisoned = False
        self._prefill_calls = 0
        self._prefix_tokens_reused = 0
        self._prefill_tokens_appended = 0
        self._decode_tokens = 0
        self._truncate_calls = 0
        self._reset_calls = 0
        self._lock = threading.RLock()
        self._operation_lock = threading.RLock()

    @property
    def token_ids(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._tokens)

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def _ensure_usable(self) -> None:
        if self._closed:
            raise NativeInferenceClosedError("Native inference session is closed")
        self._model._ensure_session_operation_admitted()
        if self._poisoned:
            raise NativeInferenceError(
                "Native inference session is poisoned after a binding failure; close and recreate it"
            )

    def prefill(self, token_ids: Sequence[int]) -> dict[str, int]:
        """Synchronize to an exact token prefix and prefill only its suffix.

        A zero-length common prefix resets native state.  This deliberately
        rebuilds from zero after front-of-context trimming instead of treating
        a matching old suffix as reusable absolute-position/RoPE state.
        """

        target = _normalize_token_ids(token_ids)
        with self._operation_lock:
            with self._lock:
                self._ensure_usable()
                current = tuple(self._tokens)
            common = _longest_common_prefix(current, target)
            try:
                if len(current) and common == 0:
                    with self._model._compute_lock:
                        self._model._binding.reset_session(
                            self._model._handle,
                            self._handle,
                        )
                    with self._lock:
                        self._tokens.clear()
                        self._cancelled = False
                        self._reset_calls += 1
                elif common < len(current):
                    with self._model._compute_lock:
                        self._model._binding.truncate_session(
                            self._model._handle,
                            self._handle,
                            common,
                        )
                    with self._lock:
                        del self._tokens[common:]
                        self._truncate_calls += 1

                suffix = target[common:]
                if suffix:
                    with self._model._compute_lock:
                        self._model._binding.prefill(
                            self._model._handle,
                            self._handle,
                            list(suffix),
                            common,
                        )
                    with self._lock:
                        self._tokens.extend(suffix)
            except InterruptedError as exc:
                with self._lock:
                    self._cancelled = True
                raise NativeInferenceCancelledError(
                    "Native inference prefill was cancelled; reset the session before reuse"
                ) from exc
            except BaseException:
                with self._lock:
                    self._poisoned = True
                raise

            with self._lock:
                self._prefill_calls += 1
                self._prefix_tokens_reused += common
                self._prefill_tokens_appended += len(suffix)
                return {
                    "prefix_tokens": len(target),
                    "prefix_reused": common,
                    "prefilled_tokens": len(suffix),
                }

    def prefill_with_embeddings(
        self,
        token_ids: Sequence[int],
        *,
        replacement_positions: Sequence[int],
        replacement_embeddings: Sequence[Sequence[float]],
    ) -> dict[str, int]:
        """Prefill an empty Glimmer session with externally encoded media rows.

        Positions are absolute indexes into ``token_ids`` and must point to an
        image/video placeholder.  This operation intentionally forbids prefix
        reuse: media bytes/processor lineage are not represented by token IDs.
        """

        target = _normalize_token_ids(token_ids)
        if not callable(getattr(self._model._binding, "prefill_with_embeddings", None)):
            raise NativeInferenceCapabilityError(
                "Resident binding does not expose Glimmer multimodal embedding prefill"
            )
        positions = tuple(int(value) for value in replacement_positions)
        if len(set(positions)) != len(positions) or any(
            value < 0 or value >= len(target) for value in positions
        ):
            raise ValueError("replacement_positions must be unique indexes into token_ids")
        rows: list[tuple[float, ...]] = []
        for row in replacement_embeddings:
            values = tuple(float(value) for value in row)
            if not values or any(not math.isfinite(value) for value in values):
                raise ValueError("replacement_embeddings must contain finite nonempty rows")
            rows.append(values)
        if len(rows) != len(positions) or (rows and len({len(row) for row in rows}) != 1):
            raise ValueError("replacement_embeddings must have one equal-width row per position")
        flattened = [value for row in rows for value in row]
        with self._operation_lock:
            with self._lock:
                self._ensure_usable()
                if self._tokens:
                    raise NativeInferenceError(
                        "multimodal embedding prefill requires a new or reset session"
                    )
            try:
                with self._model._compute_lock:
                    self._model._binding.prefill_with_embeddings(
                        self._model._handle,
                        self._handle,
                        list(target),
                        0,
                        list(positions),
                        flattened,
                    )
                with self._lock:
                    self._tokens.extend(target)
                    self._prefill_calls += 1
                    self._prefill_tokens_appended += len(target)
                    return {
                        "prefix_tokens": len(target),
                        "prefix_reused": 0,
                        "prefilled_tokens": len(target),
                        "replacement_embeddings": len(positions),
                    }
            except InterruptedError as exc:
                with self._lock:
                    self._cancelled = True
                raise NativeInferenceCancelledError(
                    "Native multimodal prefill was cancelled; reset the session before reuse"
                ) from exc
            except BaseException:
                with self._lock:
                    self._poisoned = True
                raise

    def decode(
        self,
        config: GenerationConfig | None = None,
        *,
        on_token: TokenCallback | None = None,
    ) -> GenerationResult:
        generation = config or GenerationConfig()
        if not isinstance(generation, GenerationConfig):
            raise TypeError("config must be a GenerationConfig or None")
        if on_token is not None and not callable(on_token):
            raise TypeError("on_token must be callable or None")

        if self._model._weight_selection.get("effective_speculative_decoding") == "dflash":
            return self._decode_speculative(generation, on_token=on_token)

        with self._operation_lock:
            with self._lock:
                self._ensure_usable()
                prompt_tokens = len(self._tokens)
            events: list[GenerationEvent] = []
            finish_reason: str | None = None
            stop_tokens = frozenset(
                generation.stop_token_ids
                or _normalize_manifest_stop_tokens(self._model._manifest)
            )
            # The native decoder consumes an immutable request contract. Build
            # it once instead of allocating and validating an identical Python
            # mapping at every generated token.
            binding_payload = generation.to_binding_payload()

            for index in range(generation.max_new_tokens):
                with self._lock:
                    self._ensure_usable()
                    if self._cancelled:
                        finish_reason = "cancelled"
                        break
                try:
                    with self._model._compute_lock:
                        raw = self._model._binding.decode_one(
                            self._model._handle,
                            self._handle,
                            binding_payload,
                        )
                except InterruptedError:
                    with self._lock:
                        self._cancelled = True
                    finish_reason = "cancelled"
                    break
                except BaseException:
                    with self._lock:
                        self._poisoned = True
                    raise

                if not isinstance(raw, Mapping):
                    with self._lock:
                        self._poisoned = True
                    raise NativeInferenceError("Binding decode_one must return an object")
                token_id = raw.get("token_id")
                if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                    with self._lock:
                        self._poisoned = True
                    raise NativeInferenceError(
                        "Binding decode_one returned an invalid committed token_id"
                    )
                raw_finish = raw.get("finish_reason")
                if raw_finish is not None and not isinstance(raw_finish, str):
                    with self._lock:
                        self._poisoned = True
                    raise NativeInferenceError("Binding finish_reason must be a string or None")

                with self._lock:
                    self._tokens.append(token_id)
                    self._decode_tokens += 1
                    position = len(self._tokens) - 1
                    event = GenerationEvent(
                        token_id=token_id,
                        index=index,
                        position=position,
                        text=str(raw.get("text", "")),
                        finish_reason=raw_finish,
                    )
                    events.append(event)

                # Both native state and the Python mirror are committed before
                # user code can observe this token.
                if on_token is not None:
                    on_token(event)

                if raw_finish is not None:
                    finish_reason = raw_finish
                    break
                if token_id in stop_tokens:
                    finish_reason = "stop"
                    break

            with self._lock:
                cancelled = self._cancelled or finish_reason == "cancelled"
            if finish_reason is None:
                finish_reason = "length"
            return GenerationResult(
                token_ids=tuple(event.token_id for event in events),
                text="".join(event.text for event in events),
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=len(events),
                events=tuple(events),
                cancelled=cancelled,
            )

    def _decode_speculative(
        self,
        generation: GenerationConfig,
        *,
        on_token: TokenCallback | None,
    ) -> GenerationResult:
        decode_block = getattr(self._model._binding, "decode_speculative_block", None)
        if not callable(decode_block):
            raise NativeInferenceCapabilityError(
                "The selected DFlash runtime does not expose decode_speculative_block"
            )
        with self._operation_lock:
            with self._lock:
                self._ensure_usable()
                prompt_tokens = len(self._tokens)
            events: list[GenerationEvent] = []
            finish_reason: str | None = None
            stop_tokens = frozenset(
                generation.stop_token_ids
                or _normalize_manifest_stop_tokens(self._model._manifest)
            )
            proposed = 0
            accepted = 0
            rejected = 0
            target_rows = 0
            assistant_blocks = 0
            while len(events) < generation.max_new_tokens:
                with self._lock:
                    self._ensure_usable()
                    if self._cancelled:
                        finish_reason = "cancelled"
                        break
                payload = generation.to_binding_payload()
                payload["stop_token_ids"] = list(stop_tokens)
                payload["max_tokens_remaining"] = generation.max_new_tokens - len(events)
                try:
                    with self._model._compute_lock:
                        raw = decode_block(
                            self._model._handle,
                            self._handle,
                            payload,
                        )
                except InterruptedError:
                    with self._lock:
                        self._cancelled = True
                    finish_reason = "cancelled"
                    break
                except BaseException:
                    with self._lock:
                        self._poisoned = True
                    raise
                if not isinstance(raw, Mapping):
                    with self._lock:
                        self._poisoned = True
                    raise NativeInferenceError(
                        "Binding decode_speculative_block must return an object"
                    )
                raw_tokens = raw.get("tokens")
                if (
                    not isinstance(raw_tokens, Sequence)
                    or isinstance(raw_tokens, (str, bytes))
                    or not raw_tokens
                    or len(raw_tokens) > generation.max_new_tokens - len(events)
                ):
                    with self._lock:
                        self._poisoned = True
                    raise NativeInferenceError(
                        "Binding DFlash block returned an invalid committed token list"
                    )
                block_rows: list[tuple[int, str, str | None]] = []
                saw_finish = False
                for raw_token in raw_tokens:
                    if not isinstance(raw_token, Mapping):
                        raise NativeInferenceError("Binding DFlash token row must be an object")
                    token_id = raw_token.get("token_id")
                    raw_finish = raw_token.get("finish_reason")
                    if (
                        isinstance(token_id, bool)
                        or not isinstance(token_id, int)
                        or token_id < 0
                        or (raw_finish is not None and not isinstance(raw_finish, str))
                        or saw_finish
                    ):
                        with self._lock:
                            self._poisoned = True
                        raise NativeInferenceError(
                            "Binding DFlash token/termination contract is invalid"
                        )
                    if raw_finish is not None or token_id in stop_tokens:
                        raw_finish = raw_finish or "stop"
                        saw_finish = True
                    block_rows.append((token_id, str(raw_token.get("text", "")), raw_finish))

                block_events: list[GenerationEvent] = []
                # Native commits the block atomically. Mirror every token before
                # invoking user callbacks so callback failure cannot desync state.
                with self._lock:
                    for token_id, text, raw_finish in block_rows:
                        self._tokens.append(token_id)
                        self._decode_tokens += 1
                        event = GenerationEvent(
                            token_id=token_id,
                            index=len(events) + len(block_events),
                            position=len(self._tokens) - 1,
                            text=text,
                            finish_reason=raw_finish,
                        )
                        block_events.append(event)
                    events.extend(block_events)
                if on_token is not None:
                    for event in block_events:
                        on_token(event)
                if block_events[-1].finish_reason is not None:
                    finish_reason = block_events[-1].finish_reason

                for field, accumulator_name in (
                    ("proposed_tokens", "proposed"),
                    ("accepted_tokens", "accepted"),
                    ("rejected_tokens", "rejected"),
                    ("target_rows", "target_rows"),
                    ("assistant_blocks", "assistant_blocks"),
                ):
                    value = raw.get(field, 0)
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        with self._lock:
                            self._poisoned = True
                        raise NativeInferenceError(
                            f"Binding DFlash counter {field!r} is invalid"
                        )
                    if accumulator_name == "proposed":
                        proposed += value
                    elif accumulator_name == "accepted":
                        accepted += value
                    elif accumulator_name == "rejected":
                        rejected += value
                    elif accumulator_name == "target_rows":
                        target_rows += value
                    else:
                        assistant_blocks += value
                if finish_reason is not None:
                    break

            with self._lock:
                cancelled = self._cancelled or finish_reason == "cancelled"
            if finish_reason is None:
                finish_reason = "length"
            return GenerationResult(
                token_ids=tuple(event.token_id for event in events),
                text="".join(event.text for event in events),
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=len(events),
                events=tuple(events),
                cancelled=cancelled,
                speculative_proposed_tokens=proposed,
                speculative_accepted_tokens=accepted,
                speculative_rejected_tokens=rejected,
                speculative_target_rows=target_rows,
                speculative_assistant_blocks=assistant_blocks,
            )

    def current_logits(self) -> tuple[float, ...]:
        """Return finite logits for the current non-empty token prefix.

        This is a read-only diagnostic used for parity and quality evaluation;
        it does not append a token or change native cache state.
        """

        with self._operation_lock:
            with self._lock:
                self._ensure_usable()
            current_logits = getattr(self._model._binding, "current_logits", None)
            if not callable(current_logits):
                raise NativeInferenceCapabilityError(
                    "Resident binding does not expose current_logits diagnostics"
                )
            with self._model._compute_lock:
                raw = current_logits(self._model._handle, self._handle)
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
                raise NativeInferenceError(
                    "Binding current_logits must return a non-empty numeric sequence"
                )
            logits: list[float] = []
            for value in raw:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise NativeInferenceError(
                        "Binding current_logits returned a non-numeric value"
                    )
                normalized = float(value)
                if not math.isfinite(normalized):
                    raise NativeInferenceError(
                        "Binding current_logits returned a non-finite value"
                    )
                logits.append(normalized)
            return tuple(logits)

    def truncate(self, token_count: int) -> None:
        if isinstance(token_count, bool) or not isinstance(token_count, int):
            raise TypeError("token_count must be an integer")
        with self._operation_lock:
            with self._lock:
                self._ensure_usable()
                if token_count < 0 or token_count > len(self._tokens):
                    raise ValueError(
                        f"token_count must be between 0 and {len(self._tokens)} inclusive"
                    )
                if token_count == len(self._tokens):
                    return
            try:
                with self._model._compute_lock:
                    self._model._binding.truncate_session(
                        self._model._handle,
                        self._handle,
                        token_count,
                    )
            except BaseException:
                with self._lock:
                    self._poisoned = True
                raise
            with self._lock:
                del self._tokens[token_count:]
                self._truncate_calls += 1

    def reset(self) -> None:
        with self._operation_lock:
            with self._lock:
                self._ensure_usable()
            try:
                with self._model._compute_lock:
                    self._model._binding.reset_session(
                        self._model._handle,
                        self._handle,
                    )
            except BaseException:
                with self._lock:
                    self._poisoned = True
                raise
            with self._lock:
                self._tokens.clear()
                self._cancelled = False
                self._reset_calls += 1

    def cancel(self) -> None:
        # Do not take the compute lock: the in-process engine must be able to
        # observe cancellation while a layer/token is executing.
        with self._lock:
            self._ensure_usable()
            self._cancelled = True
        self._model._binding.cancel_session(self._model._handle, self._handle)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_usable()
            binding_stats = self._model._binding.session_stats(
                self._model._handle,
                self._handle,
            )
            if not isinstance(binding_stats, Mapping):
                raise NativeInferenceError("Binding session_stats must return an object")
            return {
                **dict(binding_stats),
                "token_count": len(self._tokens),
                "cancelled": self._cancelled,
                "seed": self._seed,
                "requested_cache": self._kv_cache.mode,
                "effective_cache": self._effective_cache_mode,
                "turboquant_profile": (
                    self._kv_cache.turboquant_profile
                    if self._effective_cache_mode == "turboquant"
                    else None
                ),
                "prefill_calls": self._prefill_calls,
                "prefix_tokens_reused": self._prefix_tokens_reused,
                "prefill_tokens_appended": self._prefill_tokens_appended,
                "decode_tokens": self._decode_tokens,
                "truncate_calls": self._truncate_calls,
                "reset_calls": self._reset_calls,
            }

    def close(self) -> None:
        with self._operation_lock:
            with self._lock:
                if self._closed:
                    return
            try:
                if not self._model.closed:
                    self._model._binding.close_session(
                        self._model._handle,
                        self._handle,
                    )
            finally:
                with self._lock:
                    self._closed = True
                self._model._discard_session(self)

    def __enter__(self) -> "NativeInferenceSession":
        with self._lock:
            self._ensure_usable()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


def _normalize_manifest_stop_tokens(manifest: Mapping[str, Any]) -> tuple[int, ...]:
    raw = manifest.get("stop_tokens", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise NativeInferenceCapabilityError("Artifact stop_tokens must be an array")
    return _normalize_token_ids(raw)


__all__ = [
    "GenerationConfig",
    "GenerationEvent",
    "GenerationResult",
    "KVCacheConfig",
    "NativeInferenceCapabilities",
    "NativeInferenceCancelledError",
    "NativeInferenceCapabilityError",
    "NativeInferenceClosedError",
    "NativeInferenceError",
    "NativeInferenceModel",
    "NativeInferenceSession",
    "NativeModelLoadConfig",
    "LLAMA_SESSION_PREFIX_COW_PROFILE",
    "RESIDENT_INFERENCE_ABI_VERSION",
    "SESSION_PREFIX_COW_PROFILE",
    "STANDARD_MOE_SESSION_PREFIX_COW_PROFILE",
    "CPU_TURBOQUANT_SESSION_PREFIX_COW_PROFILE",
]
