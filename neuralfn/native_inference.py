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
    ) -> "NativeInferenceModel":
        """Load one artifact after all manifest and binding gates pass."""

        artifact_root, manifest_path, manifest = _load_manifest(artifact)
        resolved_binding = _resolve_binding(binding)
        capabilities = _prove_capabilities(manifest, resolved_binding)
        cache_config = kv_cache or KVCacheConfig()
        effective_cache_mode = _effective_cache_mode(cache_config, capabilities)
        tile_attention_config = _validate_tile_attention_contract(
            manifest,
            resolved_binding,
            capabilities,
            cache_config,
        )
        _validate_checkpoint_artifact(artifact_root, manifest)
        handle = resolved_binding.load_model(
            str(artifact_root),
            manifest,
        )
        if handle is None:
            raise NativeInferenceError("Resident binding returned no model handle")
        initial_turboquant_tables: dict[str, dict[str, Any]] = {}
        try:
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
            manifest=manifest,
            binding=resolved_binding,
            handle=handle,
            capabilities=capabilities,
            kv_cache=cache_config,
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
                "open_sessions": sum(not session.closed for session in self._sessions),
                "requested_cache": self._kv_cache.mode,
                "effective_cache": self._effective_cache_mode,
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
                            generation.to_binding_payload(),
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
    "LLAMA_SESSION_PREFIX_COW_PROFILE",
    "RESIDENT_INFERENCE_ABI_VERSION",
    "SESSION_PREFIX_COW_PROFILE",
    "STANDARD_MOE_SESSION_PREFIX_COW_PROFILE",
    "CPU_TURBOQUANT_SESSION_PREFIX_COW_PROFILE",
]
