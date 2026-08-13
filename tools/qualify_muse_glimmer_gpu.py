#!/usr/bin/env python3
"""Fail-closed real-GPU qualification for full-size Muse Glimmer artifacts.

The parent process probes the selected CUDA device, rebuilds the native Tile
libraries and resident binding from the current source tree with NVCC, runs a
short full-artifact worker under compute-sanitizer, then runs the timing worker
without instrumentation. Both workers load the same official full-size
profile and prove that model/assistant/vision compute remains on CUDA. The
timing worker records load, TTFT, decode, DFlash, vision, and sampled
device-memory data without sanitizer overhead contaminating those numbers.

This tool intentionally refuses fake runtimes, tiny geometry, prebuilt-only
"release" claims, profile/GPU-class mismatches, and missing sanitizer evidence.
It reports measurements; it does not invent throughput targets or speedups.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuralfn.native_inference import (  # noqa: E402
    GenerationConfig,
    KVCacheConfig,
    NativeInferenceModel,
    NativeModelLoadConfig,
)


SCHEMA = "neuralfn.muse_glimmer_gpu_qualification"
VERSION = 1
GIB = 1024**3
PROFILE_BY_CLASS = {
    "24": "k-quant-17gb",
    "32": "k-quant-dynamic",
    "80": "bf16",
}
GPU_CLASS_MINIMUM_BYTES = {
    "24": 24_000_000_000,
    "32": 32_000_000_000,
    "80": 80_000_000_000,
}
MINIMUM_CUDA_WEIGHT_BYTES = {
    "k-quant-17gb": 15_000_000_000,
    "k-quant-dynamic": 18_000_000_000,
    "bf16": 50_000_000_000,
}
CANONICAL_KQUANT_SHA256 = {
    "k-quant-17gb": "4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e",
    "k-quant-dynamic": "ac7023d6a4c704eb9af54ab53e476a66b7f5b6c0ef2fc4a8dde5253c291a6c38",
}
MAIN_ARTIFACT_REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
GGUF_ARTIFACT_REVISION = "43c7eadd41352a299ea8e0a36b3157978dd63596"
FULL_GEOMETRY = {
    "model_family": "muse_glimmer",
    "vocab_size": 202_048,
    "num_layers": 52,
    "num_heads": 32,
    "num_kv_heads": 2,
    "channels": 6_656,
    "head_dim": 128,
    "max_seq_len": 131_072,
}
FULL_TEXT_PARAMETER_COUNT = 27_854_780_928
KERNEL_PROBE_NAMES = (
    "q4_k_dequant_linear_dx",
    "sigmoid_gate",
    "rms_norm_6656",
    "positioned_rope_q32_kv2_h128",
    "gqa_decode_q32_kv2_h128_window2048",
    "cache_commit_bf16",
    "dflash_block_attention_q16_q32_kv8_h128",
    "masked_ce_vocab202048_i32",
    "dpo_forward_backward",
    "reward_head_6656_forward_backward",
    "preference_bce_forward_backward",
    "ppo_forward_backward",
    "vision_prepare_1536",
    "vision_layer_norm_1536",
    "vision_attention_12x128",
    "vision_pixel_shuffle_6144",
)
SANITIZER_TOOLS = ("memcheck", "synccheck", "initcheck", "racecheck")
SOURCE_PROOF_FILES = (
    "neuralfn/native_inference.py",
    "neuralfn/native_muse_glimmer_checkpoint.py",
    "neuralfn/native_gguf.py",
    "neuralfn/native_glimmer_media.py",
    "neuralfn/csrc/native_train/tile_ops.h",
    "neuralfn/csrc/native_train/tile_ops.cu",
    "neuralfn/csrc/tile_cuda/kernels.cu",
    "neuralfn/csrc/native_gpt2/resident_binding.cpp",
    "neuralfn/csrc/native_gpt2/resident_dense.cpp",
    "neuralfn/csrc/native_gpt2/resident_dense.h",
    "neuralfn/csrc/native_gpt2/resident_glimmer.cpp",
    "neuralfn/csrc/native_gpt2/resident_glimmer.h",
    "neuralfn/csrc/native_gpt2/resident_glimmer_cuda.cpp",
    "neuralfn/csrc/native_gpt2/resident_glimmer_cuda.h",
    "neuralfn/csrc/native_gpt2/resident_glimmer_vision.cpp",
    "neuralfn/csrc/native_gpt2/resident_glimmer_vision.h",
    "neuralfn/csrc/native_gpt2/resident_glimmer_assistant.cpp",
    "neuralfn/csrc/native_gpt2/resident_glimmer_assistant.h",
    "neuralfn/csrc/native_train/muse_glimmer_native_train.cpp",
    "tests/cpp/muse_glimmer_cuda_kernel_probe.cpp",
    "tools/build_native_train_tile_ops.sh",
    "tools/build_native_inference_binding.sh",
    "tools/build_native_missing_trainers.sh",
    "tools/qualify_muse_glimmer_gpu.py",
)


class QualificationError(RuntimeError):
    """A release-evidence invariant failed."""


class FailClosedParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise QualificationError(message)


@dataclass(frozen=True, slots=True)
class RunConfig:
    artifact: str
    profile: str
    gpu_class: str
    cuda_runtime_lib: str
    cuda_device: int
    binding_lib: str
    tile_ops_lib: str
    contexts: tuple[int, ...]
    decode_tokens: int
    warmups: int
    repetitions: int
    prompt_token_id: int
    stop_token_id: int
    companions: tuple[str, ...]
    require_dflash: bool
    run_vision: bool
    vision_patch_width: int
    vision_grid: tuple[int, int, int]
    source_tree_sha256: str
    weight_precision_request: str = "auto"
    progress_path: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_proof() -> tuple[str, list[dict[str, Any]]]:
    combined = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    for relative in SOURCE_PROOF_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise QualificationError(f"source proof file is missing: {path}")
        digest = _sha256(path)
        size = path.stat().st_size
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\0")
        rows.append({"path": relative, "bytes": size, "sha256": digest})
    return combined.hexdigest(), rows


def _require_int(mapping: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise QualificationError(
            f"telemetry {key!r} must be an integer >= {minimum}; got {value!r}"
        )
    return value


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise QualificationError("cannot calculate a percentile without samples")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples or any(not math.isfinite(value) or value <= 0.0 for value in samples):
        raise QualificationError("timing samples must be finite positive values")
    return {
        "samples": samples,
        "count": len(samples),
        "mean": statistics.fmean(samples),
        "p50": _percentile(samples, 0.50),
        "p95": _percentile(samples, 0.95),
        "minimum": min(samples),
        "maximum": max(samples),
    }


class CudaProbe:
    """Authoritative runtime/device probe plus sampled global-memory tracking."""

    COMPUTE_CAPABILITY_MAJOR = 75
    COMPUTE_CAPABILITY_MINOR = 76

    def __init__(self, runtime: str, device: int) -> None:
        try:
            self.library = ctypes.CDLL(runtime)
        except OSError as exc:
            raise QualificationError(f"cannot load CUDA runtime {runtime!r}: {exc}") from exc
        self.runtime = runtime
        self.device = device
        self.get_count = self._function("cudaGetDeviceCount", [ctypes.POINTER(ctypes.c_int)])
        self.set_device = self._function("cudaSetDevice", [ctypes.c_int])
        self.mem_info = self._function(
            "cudaMemGetInfo",
            [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)],
        )
        self.device_attribute = self._function(
            "cudaDeviceGetAttribute",
            [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
        )
        self.runtime_version = self._function(
            "cudaRuntimeGetVersion", [ctypes.POINTER(ctypes.c_int)]
        )
        self.driver_version = self._function(
            "cudaDriverGetVersion", [ctypes.POINTER(ctypes.c_int)]
        )
        self.synchronize = self._function("cudaDeviceSynchronize", [])
        count = ctypes.c_int()
        self._check(self.get_count(ctypes.byref(count)), "cudaGetDeviceCount")
        if count.value <= 0:
            raise QualificationError("CUDA runtime reports no devices")
        if device < 0 or device >= count.value:
            raise QualificationError(
                f"CUDA device {device} is outside the runtime count {count.value}"
            )
        self._check(self.set_device(device), f"cudaSetDevice({device})")
        self.device_count = int(count.value)
        self.compute_major = self._attribute(self.COMPUTE_CAPABILITY_MAJOR)
        self.compute_minor = self._attribute(self.COMPUTE_CAPABILITY_MINOR)
        self.cuda_runtime_version = self._version(self.runtime_version, "cudaRuntimeGetVersion")
        self.cuda_driver_version = self._version(self.driver_version, "cudaDriverGetVersion")
        self._check(self.synchronize(), "cudaDeviceSynchronize")
        self.baseline_free_bytes, self.total_bytes = self._raw_memory()
        self.minimum_free_bytes = self.baseline_free_bytes
        self.samples: list[dict[str, Any]] = []
        self.checkpoint("fresh_worker_baseline")

    def _function(self, name: str, argtypes: list[Any]) -> Any:
        try:
            function = getattr(self.library, name)
        except AttributeError as exc:
            raise QualificationError(
                f"CUDA runtime {self.runtime!r} is missing {name}"
            ) from exc
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        return function

    @staticmethod
    def _check(status: int, action: str) -> None:
        if int(status) != 0:
            raise QualificationError(f"{action} failed with CUDA status {int(status)}")

    def _attribute(self, attribute: int) -> int:
        value = ctypes.c_int()
        self._check(
            self.device_attribute(ctypes.byref(value), attribute, self.device),
            f"cudaDeviceGetAttribute({attribute})",
        )
        return int(value.value)

    def _version(self, function: Any, action: str) -> int:
        value = ctypes.c_int()
        self._check(function(ctypes.byref(value)), action)
        if value.value <= 0:
            raise QualificationError(f"{action} returned an invalid version")
        return int(value.value)

    def _raw_memory(self) -> tuple[int, int]:
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        self._check(self.mem_info(ctypes.byref(free), ctypes.byref(total)), "cudaMemGetInfo")
        if total.value <= 0 or free.value <= 0 or free.value > total.value:
            raise QualificationError("cudaMemGetInfo returned invalid free/total bytes")
        return int(free.value), int(total.value)

    def checkpoint(self, label: str) -> None:
        self._check(self.synchronize(), "cudaDeviceSynchronize")
        free, total = self._raw_memory()
        if total != self.total_bytes:
            raise QualificationError("CUDA total bytes changed during qualification")
        self.minimum_free_bytes = min(self.minimum_free_bytes, free)
        self.samples.append(
            {
                "label": label,
                "free_bytes": free,
                "allocated_delta_bytes": max(0, self.baseline_free_bytes - free),
            }
        )

    def hardware(self) -> dict[str, Any]:
        return {
            "cuda_runtime_lib": self.runtime,
            "cuda_device": self.device,
            "cuda_device_count": self.device_count,
            "compute_capability": f"{self.compute_major}.{self.compute_minor}",
            "cuda_runtime_version": self.cuda_runtime_version,
            "cuda_driver_version": self.cuda_driver_version,
            "total_bytes": self.total_bytes,
            "baseline_free_bytes": self.baseline_free_bytes,
        }

    def memory_result(self) -> dict[str, Any]:
        peak = max(0, self.baseline_free_bytes - self.minimum_free_bytes)
        if peak <= 0:
            raise QualificationError("real CUDA run produced no measurable VRAM delta")
        return {
            "provider": "cudaMemGetInfo",
            "scope": "sampled-device-global-baseline-subtracted",
            "peak_sampled_delta_bytes": peak,
            "minimum_free_bytes": self.minimum_free_bytes,
            "samples": self.samples,
        }


def _validate_gpu_class(total_bytes: int, gpu_class: str) -> None:
    try:
        minimum = GPU_CLASS_MINIMUM_BYTES[gpu_class]
    except KeyError as exc:
        raise QualificationError(f"unknown GPU class {gpu_class!r}") from exc
    if total_bytes < minimum:
        raise QualificationError(
            f"the {gpu_class}-GB profile tier requires at least {minimum} bytes, "
            f"but cudaMemGetInfo reports {total_bytes} bytes"
        )


def _weight_precision_request_for_tier(total_bytes: int, gpu_class: str) -> str:
    """Use auto for the best tier this device qualifies for, otherwise pin.

    A larger GPU is valid evidence for a lower-memory profile.  In that case
    the profile must be selected explicitly because the production auto policy
    should continue choosing the highest-fidelity tier that fits.
    """

    _validate_gpu_class(total_bytes, gpu_class)
    eligible = [
        tier
        for tier, minimum in GPU_CLASS_MINIMUM_BYTES.items()
        if total_bytes >= minimum
    ]
    highest_tier = max(eligible, key=lambda tier: GPU_CLASS_MINIMUM_BYTES[tier])
    return "auto" if gpu_class == highest_tier else PROFILE_BY_CLASS[gpu_class]


def _load_binding(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_native_inference", path)
    if spec is None or spec.loader is None:
        raise QualificationError(f"cannot import resident binding: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except BaseException as exc:
        raise QualificationError(f"resident binding import failed: {exc}") from exc
    return module


def _manifest_path(artifact: Path) -> Path:
    path = artifact / "native-execution-manifest.json" if artifact.is_dir() else artifact
    if not path.is_file():
        raise QualificationError(f"native execution manifest does not exist: {path}")
    return path.resolve()


def _validate_artifact_manifest(config: RunConfig) -> tuple[Path, dict[str, Any]]:
    path = _manifest_path(Path(config.artifact))
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError(f"cannot read native execution manifest: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "neuralfn.native_execution_manifest"
        or manifest.get("version") != 1
    ):
        raise QualificationError("artifact is not a Native Execution Manifest v1")
    model = manifest.get("model")
    source = manifest.get("source_graph")
    variants = manifest.get("checkpoint_variants")
    if not isinstance(model, Mapping) or model.get("family") != "muse_glimmer":
        raise QualificationError("artifact model family is not muse_glimmer")
    if not isinstance(source, Mapping) or not isinstance(variants, Mapping):
        raise QualificationError("artifact source/variant provenance is missing")
    descriptor = variants.get(config.profile)
    if not isinstance(descriptor, Mapping):
        raise QualificationError(f"artifact lacks profile {config.profile}")
    if config.profile in CANONICAL_KQUANT_SHA256:
        if (
            source.get("repository") != "meta-models/Muse-Glimmer-30B-GGUF"
            or source.get("revision") != GGUF_ARTIFACT_REVISION
            or descriptor.get("target_sha256") != CANONICAL_KQUANT_SHA256[config.profile]
        ):
            raise QualificationError("K-Quant profile is not the canonical pinned artifact")
    else:
        if (
            source.get("repository") != "meta-models/Muse-Glimmer-30B"
            or source.get("revision") != MAIN_ARTIFACT_REVISION
        ):
            raise QualificationError("BF16 profile is not derived from the pinned main artifact")
        if config.run_vision and descriptor.get("component") != "full":
            raise QualificationError(
                "BF16 CUDA vision qualification requires the full embedded-vision checkpoint"
            )
    target_nbytes = descriptor.get("target_nbytes")
    if (
        isinstance(target_nbytes, bool)
        or not isinstance(target_nbytes, int)
        or target_nbytes < MINIMUM_CUDA_WEIGHT_BYTES[config.profile]
    ):
        raise QualificationError("selected profile is not a full-size checkpoint")
    companions = manifest.get("companion_checkpoints")
    if not isinstance(companions, Mapping):
        raise QualificationError("artifact companion catalog is missing")
    for name in config.companions:
        row = companions.get(name)
        if not isinstance(row, Mapping):
            raise QualificationError(f"artifact lacks required companion {name}")
        capabilities = row.get("capabilities")
        if not isinstance(capabilities, Mapping) or capabilities.get("resident_cuda") is not True:
            raise QualificationError(
                f"companion {name} does not declare resident CUDA capability"
            )
    return path, manifest


def _validate_full_size_stats(
    stats: Mapping[str, Any], config: RunConfig, *, after_compute: bool
) -> None:
    for key, expected in FULL_GEOMETRY.items():
        if stats.get(key) != expected:
            raise QualificationError(
                f"full-size geometry mismatch for {key}: {stats.get(key)!r} != {expected!r}"
            )
    parameter_count = _require_int(stats, "parameter_count", minimum=1)
    if parameter_count != FULL_TEXT_PARAMETER_COUNT:
        raise QualificationError(
            "full-size text parameter count mismatch: "
            f"{parameter_count} != {FULL_TEXT_PARAMETER_COUNT}"
        )
    if stats.get("whole_model_cuda") is not True:
        raise QualificationError("model did not report whole_model_cuda=true")
    if stats.get("cuda_model_compute_only") is not True:
        raise QualificationError("model did not report cuda_model_compute_only=true")
    if _require_int(stats, "cpu_model_compute_rows") != 0:
        raise QualificationError("CPU model-compute rows are nonzero")
    if stats.get("requested_weight_precision") != config.weight_precision_request:
        raise QualificationError(
            "hardware qualification used a different weight-precision request"
        )
    if stats.get("effective_weight_precision") != config.profile:
        raise QualificationError("effective weight precision differs from the expected profile")
    expected_selection = (
        "auto-vram" if config.weight_precision_request == "auto" else "explicit"
    )
    if stats.get("weight_precision_selection") != expected_selection:
        raise QualificationError(
            "benchmark profile selection does not match its auto/explicit request"
        )
    selected_sha = stats.get("selected_artifact_sha256")
    if not isinstance(selected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", selected_sha) is None:
        raise QualificationError("selected artifact SHA-256 is missing or malformed")
    resident = _require_int(stats, "cuda_resident_weight_bytes", minimum=1)
    if resident < MINIMUM_CUDA_WEIGHT_BYTES[config.profile]:
        raise QualificationError(
            f"CUDA resident bytes {resident} are below the full-size {config.profile} floor"
        )
    if stats.get("cuda_device") != config.cuda_device:
        raise QualificationError("resident model loaded on the wrong CUDA device")
    if str(Path(str(stats.get("cuda_tile_ops_lib", ""))).resolve()) != str(
        Path(config.tile_ops_lib).resolve()
    ):
        raise QualificationError("resident model did not load the qualified Tile library")
    if config.require_dflash:
        if stats.get("dflash_loaded") is not True or stats.get("dflash_cuda") is not True:
            raise QualificationError("DFlash was required but is not resident on CUDA")
        _require_int(stats, "dflash_cuda_resident_weight_bytes", minimum=1)
    if config.run_vision:
        if stats.get("vision_loaded") is not True or stats.get("vision_cuda") is not True:
            raise QualificationError("vision was required but is not resident on CUDA")
        _require_int(stats, "vision_resident_weight_bytes", minimum=1)
    if after_compute and _require_int(stats, "cuda_kernel_launches", minimum=1) <= 0:
        raise QualificationError("CUDA kernel launch counter did not advance")


def _generation(config: RunConfig, count: int, seed: int) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=count,
        temperature=0.0,
        top_k=None,
        top_p=1.0,
        seed=seed,
        stop_token_ids=(config.stop_token_id,),
    )


def _require_token_count(result: Any, expected: int, label: str) -> tuple[int, ...]:
    tokens = tuple(getattr(result, "token_ids", ()))
    if len(tokens) != expected:
        raise QualificationError(
            f"{label} generated {len(tokens)} tokens; expected exactly {expected}"
        )
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens):
        raise QualificationError(f"{label} returned invalid token IDs")
    return tokens


def _one_timing_trial(
    model: Any,
    config: RunConfig,
    context: int,
    repetition: int,
    clock: Callable[[], float],
) -> dict[str, Any]:
    prompt = (config.prompt_token_id,) * context
    trial_start = clock()
    with model.create_session(seed=10_000 + repetition) as session:
        prefill_start = clock()
        session.prefill(prompt)
        prefill_elapsed = clock() - prefill_start
        first_decode_start = clock()
        first = session.decode(_generation(config, 1, 10_000 + repetition))
        first_decode_elapsed = clock() - first_decode_start
        first_tokens = _require_token_count(first, 1, "TTFT")
        ttft = clock() - trial_start
        decode_start = clock()
        decoded = session.decode(
            _generation(config, config.decode_tokens, 20_000 + repetition)
        )
        decode_elapsed = clock() - decode_start
        decode_ids = _require_token_count(decoded, config.decode_tokens, "decode")
        session_stats = session.stats()
    if (
        ttft <= 0.0
        or prefill_elapsed <= 0.0
        or first_decode_elapsed <= 0.0
        or decode_elapsed <= 0.0
    ):
        raise QualificationError("benchmark clock did not advance")
    if session_stats.get("effective_cache") != "full":
        raise QualificationError("full lossless cache was not used")
    result = {
        "ttft_seconds": ttft,
        "prefill_seconds": prefill_elapsed,
        "prefill_tokens_per_second": context / prefill_elapsed,
        "first_decode_seconds": first_decode_elapsed,
        "decode_elapsed_seconds": decode_elapsed,
        "decode_tokens_per_second": config.decode_tokens / decode_elapsed,
        "first_token_id": first_tokens[0],
        "decode_token_ids": list(decode_ids),
        "session": {
            key: session_stats.get(key)
            for key in (
                "token_count",
                "cache_bytes",
                "cache_capacity_bytes",
                "speculative_proposed_tokens",
                "speculative_accepted_tokens",
                "speculative_rejected_tokens",
                "speculative_target_rows",
                "speculative_assistant_blocks",
            )
            if key in session_stats
        },
        "generation": {
            "speculative_proposed_tokens": int(
                getattr(decoded, "speculative_proposed_tokens", 0)
            ),
            "speculative_accepted_tokens": int(
                getattr(decoded, "speculative_accepted_tokens", 0)
            ),
            "speculative_rejected_tokens": int(
                getattr(decoded, "speculative_rejected_tokens", 0)
            ),
            "speculative_target_rows": int(
                getattr(decoded, "speculative_target_rows", 0)
            ),
            "speculative_assistant_blocks": int(
                getattr(decoded, "speculative_assistant_blocks", 0)
            ),
        },
    }
    if config.require_dflash:
        speculative = result["generation"]
        if speculative["speculative_proposed_tokens"] <= 0:
            raise QualificationError("DFlash produced no proposals")
        if speculative["speculative_assistant_blocks"] <= 0:
            raise QualificationError("DFlash assistant executed no proposal blocks")
        proposed = speculative["speculative_proposed_tokens"]
        accepted = speculative["speculative_accepted_tokens"]
        rejected = speculative["speculative_rejected_tokens"]
        if accepted < 0 or rejected < 0 or accepted + rejected > proposed:
            raise QualificationError("DFlash acceptance counters are inconsistent")
    return result


def _summarize_trials(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = tuple(trials[0]["decode_token_ids"])
    if any(tuple(trial["decode_token_ids"]) != first for trial in trials[1:]):
        raise QualificationError("strict greedy output changed across repetitions")
    summary: dict[str, Any] = {
        "ttft_seconds": _distribution([float(row["ttft_seconds"]) for row in trials]),
        "prefill_seconds": _distribution(
            [float(row["prefill_seconds"]) for row in trials]
        ),
        "prefill_tokens_per_second": _distribution(
            [float(row["prefill_tokens_per_second"]) for row in trials]
        ),
        "first_decode_seconds": _distribution(
            [float(row["first_decode_seconds"]) for row in trials]
        ),
        "decode_elapsed_seconds": _distribution(
            [float(row["decode_elapsed_seconds"]) for row in trials]
        ),
        "decode_tokens_per_second": _distribution(
            [float(row["decode_tokens_per_second"]) for row in trials]
        ),
        "greedy_token_ids": list(first),
    }
    proposed = sum(
        int(row["generation"]["speculative_proposed_tokens"]) for row in trials
    )
    accepted = sum(
        int(row["generation"]["speculative_accepted_tokens"]) for row in trials
    )
    rejected = sum(
        int(row["generation"]["speculative_rejected_tokens"]) for row in trials
    )
    blocks = sum(
        int(row["generation"]["speculative_assistant_blocks"]) for row in trials
    )
    if proposed > 0:
        summary["dflash"] = {
            "proposed_tokens": proposed,
            "accepted_tokens": accepted,
            "rejected_tokens": rejected,
            "assistant_blocks": blocks,
            "acceptance_rate": accepted / proposed,
            "mean_accepted_per_block": accepted / max(blocks, 1),
        }
    return summary


def _write_worker_progress(
    config: RunConfig,
    stage: str,
    **details: Any,
) -> None:
    if config.progress_path is None:
        return
    destination = Path(config.progress_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "neuralfn.muse_glimmer_gpu_qualification.progress",
        "version": 1,
        "stage": stage,
        "profile": config.profile,
        "gpu_class": config.gpu_class,
        "source_tree_sha256": config.source_tree_sha256,
        "updated_unix_seconds": time.time(),
        **details,
    }
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _run_worker(
    config: RunConfig,
    *,
    model_loader: Any = NativeInferenceModel,
    binding_loader: Callable[[Path], Any] = _load_binding,
    probe_factory: Callable[[str, int], CudaProbe] = CudaProbe,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    probe = probe_factory(config.cuda_runtime_lib, config.cuda_device)
    _write_worker_progress(config, "worker_started")
    _validate_gpu_class(probe.total_bytes, config.gpu_class)
    expected_profile = PROFILE_BY_CLASS[config.gpu_class]
    if config.profile != expected_profile:
        raise QualificationError(
            f"{config.gpu_class}-GB qualification requires {expected_profile}, not {config.profile}"
        )
    expected_request = _weight_precision_request_for_tier(
        probe.total_bytes, config.gpu_class
    )
    if config.weight_precision_request != expected_request:
        raise QualificationError(
            "weight-precision request does not match the physical-device/profile-tier policy"
        )
    if config.source_tree_sha256 != _source_tree_proof()[0]:
        raise QualificationError("source tree changed between NVCC build and benchmark worker")
    manifest_path, _manifest = _validate_artifact_manifest(config)
    binding = binding_loader(Path(config.binding_lib))
    maximum_context = max(config.contexts)
    load_config = NativeModelLoadConfig(
        weight_precision=config.weight_precision_request,
        runtime="native-cuda",
        cuda_device=config.cuda_device,
        tile_ops_lib=config.tile_ops_lib,
        cuda_runtime_lib=config.cuda_runtime_lib,
        context_tokens=maximum_context + config.decode_tokens,
        session_count=1,
        companion_checkpoints=config.companions,
        speculative_decoding="required" if config.require_dflash else "off",
    )
    load_start = clock()
    with model_loader.load(
        Path(config.artifact),
        binding=binding,
        kv_cache=KVCacheConfig(mode="full"),
        load_config=load_config,
    ) as model:
        load_seconds = clock() - load_start
        if load_seconds <= 0.0:
            raise QualificationError("model load timer did not advance")
        probe.checkpoint("model_and_companions_loaded")
        initial_stats = model.stats()
        _validate_full_size_stats(initial_stats, config, after_compute=False)
        _write_worker_progress(
            config,
            "model_and_companions_loaded",
            resident_weight_bytes=initial_stats.get("cuda_resident_weight_bytes"),
            dflash_resident_weight_bytes=initial_stats.get(
                "dflash_cuda_resident_weight_bytes"
            ),
            vision_resident_weight_bytes=initial_stats.get(
                "vision_resident_weight_bytes"
            ),
        )
        vocab = _require_int(initial_stats, "vocab_size", minimum=1)
        if config.prompt_token_id >= vocab or config.stop_token_id >= vocab:
            raise QualificationError("prompt/stop benchmark token exceeds the vocabulary")
        max_seq_len = _require_int(initial_stats, "max_seq_len", minimum=1)
        if maximum_context + config.decode_tokens > max_seq_len:
            raise QualificationError(
                "largest prompt context plus decode tokens exceeds max_seq_len"
            )

        if config.warmups:
            for warmup in range(config.warmups):
                _write_worker_progress(
                    config, "warmup_started", warmup=warmup + 1, total=config.warmups
                )
                _one_timing_trial(
                    model, config, min(config.contexts), -1 - warmup, clock
                )
            probe.checkpoint("warmups_complete")
            _write_worker_progress(config, "warmups_complete", total=config.warmups)

        context_results: list[dict[str, Any]] = []
        for context in config.contexts:
            _write_worker_progress(
                config,
                "context_started",
                context=context,
                repetitions=config.repetitions,
                completed_contexts=[
                    row["prompt_context_tokens"] for row in context_results
                ],
            )
            trials = [
                _one_timing_trial(model, config, context, repetition, clock)
                for repetition in range(config.repetitions)
            ]
            probe.checkpoint(f"context_{context}_complete")
            context_results.append(
                {
                    "prompt_context_tokens": context,
                    "decode_tokens": config.decode_tokens,
                    "repetitions": config.repetitions,
                    "summary": _summarize_trials(trials),
                }
            )
            _write_worker_progress(
                config,
                "context_complete",
                context=context,
                result=context_results[-1],
                completed_contexts=[
                    row["prompt_context_tokens"] for row in context_results
                ],
            )

        vision: dict[str, Any] | None = None
        if config.run_vision:
            _write_worker_progress(config, "vision_started")
            t, h, w = config.vision_grid
            patch_rows = t * h * w
            row = tuple(0.0 for _ in range(config.vision_patch_width))
            patches = tuple(row for _ in range(patch_rows))
            vision_times: list[float] = []
            output_rows = 0
            output_width = 0
            for _ in range(config.repetitions):
                started = clock()
                output = model.encode_media(patches, (config.vision_grid,))
                elapsed = clock() - started
                if not output or any(
                    not values or any(not math.isfinite(value) for value in values)
                    for values in output
                ):
                    raise QualificationError("CUDA vision returned malformed/non-finite rows")
                widths = {len(values) for values in output}
                if widths != {6_656}:
                    raise QualificationError("CUDA vision output width is not 6656")
                output_rows = len(output)
                output_width = next(iter(widths))
                vision_times.append(elapsed)
            probe.checkpoint("vision_complete")
            vision = {
                "grid_thw": list(config.vision_grid),
                "input_rows": patch_rows,
                "patch_width": config.vision_patch_width,
                "output_rows": output_rows,
                "output_width": output_width,
                "elapsed_seconds": _distribution(vision_times),
            }
            _write_worker_progress(config, "vision_complete", result=vision)

        final_stats = model.stats()
        _validate_full_size_stats(final_stats, config, after_compute=True)
        if _require_int(final_stats, "cpu_model_compute_rows") != 0:
            raise QualificationError("CPU model compute occurred during CUDA qualification")
        probe.checkpoint("before_model_close")

    probe.checkpoint("after_model_close")
    _write_worker_progress(config, "worker_complete")
    if not manifest_path.is_file():
        raise QualificationError("qualified artifact manifest disappeared")
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "worker-complete",
        "profile": config.profile,
        "gpu_class": config.gpu_class,
        "profile_tier": {
            "minimum_total_vram_bytes": GPU_CLASS_MINIMUM_BYTES[config.gpu_class],
            "physical_total_vram_bytes": probe.total_bytes,
            "larger_device": config.weight_precision_request != "auto",
        },
        "artifact": {
            "path": str(Path(config.artifact).resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_path),
            "selected_artifact_sha256": final_stats["selected_artifact_sha256"],
        },
        "hardware": probe.hardware(),
        "load": {
            "seconds": load_seconds,
            "requested_weight_precision": final_stats["requested_weight_precision"],
            "effective_weight_precision": final_stats["effective_weight_precision"],
            "weight_precision_selection": final_stats[
                "weight_precision_selection"
            ],
            "companions": list(config.companions),
            "speculative_decoding": final_stats.get("effective_speculative_decoding"),
        },
        "model_stats": {
            key: final_stats.get(key)
            for key in (
                *FULL_GEOMETRY.keys(),
                "parameter_count",
                "resident_weight_bytes",
                "cuda_resident_weight_bytes",
                "cuda_workspace_bytes",
                "cuda_kernel_launches",
                "cpu_model_compute_rows",
                "whole_model_cuda",
                "cuda_model_compute_only",
                "dflash_loaded",
                "dflash_cuda",
                "dflash_cuda_resident_weight_bytes",
                "dflash_cuda_workspace_bytes",
                "dflash_cuda_kernel_launches",
                "vision_loaded",
                "vision_cuda",
                "vision_resident_weight_bytes",
            )
        },
        "contexts": context_results,
        "vision": vision,
        "memory": probe.memory_result(),
        "progress_path": config.progress_path,
        "source_tree_sha256": config.source_tree_sha256,
    }


def _run_checked(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> str:
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        env=dict(env) if env is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise QualificationError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}"
        )
    return completed.stdout


def _resolve_executable(requested: str | None, default: str) -> Path:
    raw = requested or shutil.which(default)
    if not raw:
        raise QualificationError(f"required executable is unavailable: {default}")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        resolved = shutil.which(str(candidate))
        candidate = Path(resolved) if resolved else candidate
    try:
        candidate = candidate.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(f"required executable is unavailable: {raw}") from exc
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise QualificationError(f"required executable is not executable: {candidate}")
    return candidate


def _build_cuda_kernel_probe(
    build: Mapping[str, Any],
    strict_tile: Path,
    cuda_runtime_lib: str,
    build_dir: Path,
) -> dict[str, Any]:
    source = ROOT / "tests/cpp/muse_glimmer_cuda_kernel_probe.cpp"
    runtime = Path(cuda_runtime_lib).expanduser().resolve(strict=True)
    nvcc = Path(str(build["nvcc_path"])).resolve(strict=True)
    include_candidates = (
        nvcc.parent.parent / "include",
        runtime.parent.parent / "include",
    )
    cuda_include = next(
        (
            candidate
            for candidate in include_candidates
            if (candidate / "cuda_runtime_api.h").is_file()
        ),
        None,
    )
    if cuda_include is None:
        raise QualificationError(
            "CUDA runtime headers are unavailable beside NVCC/runtime library"
        )
    executable = build_dir / "muse_glimmer_cuda_kernel_probe"
    cxx = str(build["host_cxx_path"])
    _run_checked(
        [
            cxx,
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "neuralfn/csrc/native_train"),
            "-I",
            str(cuda_include),
            str(source),
            str(runtime),
            f"-Wl,-rpath,{runtime.parent}",
            "-ldl",
            "-o",
            str(executable),
        ]
    )
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise QualificationError("CUDA kernel probe build produced no executable")
    return {
        "path": str(executable),
        "bytes": executable.stat().st_size,
        "sha256": _sha256(executable),
        "source_path": str(source.relative_to(ROOT)),
        "source_sha256": _sha256(source),
        "strict_tile_path": str(strict_tile),
        "strict_tile_sha256": _sha256(strict_tile),
        "cuda_runtime_lib": str(runtime),
    }


def _parse_kernel_probe_stdout(stdout: str, *, cuda_device: int) -> dict[str, Any]:
    for row in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(row)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("status") != "passed":
            continue
        if payload.get("device") != cuda_device:
            raise QualificationError("CUDA kernel probe ran on the wrong device")
        kernels = payload.get("kernels")
        if kernels != list(KERNEL_PROBE_NAMES):
            raise QualificationError("CUDA kernel probe coverage list is incomplete")
        return payload
    raise QualificationError("CUDA kernel probe emitted no passing JSON payload")


def _run_kernel_probe_sanitizers(
    probe: Mapping[str, Any],
    sanitizer: Path,
    *,
    cuda_device: int,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    tools: dict[str, Any] = {}
    for tool in SANITIZER_TOOLS:
        command = [
            str(sanitizer),
            "--tool",
            tool,
            "--error-exitcode",
            "97",
            "--target-processes",
            "all",
            str(probe["path"]),
            str(probe["strict_tile_path"]),
            str(cuda_device),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=dict(environment),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise QualificationError(
                f"compute-sanitizer {tool} kernel probe failed "
                f"({completed.returncode})\nstdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        _parse_kernel_probe_stdout(completed.stdout, cuda_device=cuda_device)
        log = completed.stderr + "\n" + completed.stdout
        zero_summary = (
            re.search(r"ERROR SUMMARY:\s*0 errors", log) is not None
            if tool != "racecheck"
            else re.search(
                r"RACECHECK SUMMARY:\s*0 hazards displayed\s*"
                r"\(0 errors, 0 warnings\)",
                log,
            )
            is not None
        )
        if not zero_summary:
            raise QualificationError(
                f"compute-sanitizer {tool} did not report a zero-error summary"
            )
        tools[tool] = {
            "status": "passed",
            "zero_error_summary": True,
            "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
        }
    return {
        "mode": "required",
        "status": "passed",
        "scope": "real-device-kernel-probe-before-full-artifact-benchmark",
        "path": str(sanitizer),
        "cuda_device": cuda_device,
        "probe": dict(probe),
        "kernels": list(KERNEL_PROBE_NAMES),
        "tools": tools,
        "zero_error_summary": True,
    }


def _validate_tile_abis(inference_tile: Path, training_tile: Path) -> dict[str, int]:
    try:
        inference = ctypes.CDLL(str(inference_tile))
        training = ctypes.CDLL(str(training_tile))
    except OSError as exc:
        raise QualificationError(f"built Tile library cannot be loaded: {exc}") from exc
    symbols = {
        "strict_math": (inference, "nfn_native_tile_strict_math_abi_version", 1),
        "packed_weight": (inference, "nfn_native_tile_packed_weight_abi_version", 1),
        "glimmer_inference": (
            inference,
            "nfn_native_tile_glimmer_inference_abi_version",
            1,
        ),
        "glimmer_vision": (inference, "nfn_native_tile_glimmer_vision_abi_version", 1),
        "glimmer_training": (
            training,
            "nfn_native_tile_glimmer_training_abi_version",
            1,
        ),
    }
    values: dict[str, int] = {}
    for label, (library, symbol, expected) in symbols.items():
        try:
            function = getattr(library, symbol)
        except AttributeError as exc:
            raise QualificationError(f"built Tile library is missing {symbol}") from exc
        function.argtypes = []
        function.restype = ctypes.c_int
        value = int(function())
        if value != expected:
            raise QualificationError(f"{symbol} returned ABI {value}; expected {expected}")
        values[label] = value
    return values


def _validated_cuda_arch(value: str) -> str:
    match = re.fullmatch(r"sm_([0-9]{2,3})", str(value))
    if match is None or int(match.group(1)) < 80:
        raise QualificationError(
            "--cuda-arch must be an sm_80-or-newer concrete architecture"
        )
    return str(value)


def _build_native_binaries(
    args: argparse.Namespace,
    arch: str,
) -> tuple[dict[str, Any], Path, Path]:
    arch = _validated_cuda_arch(arch)
    nvcc = _resolve_executable(args.nvcc, "nvcc")
    nvcc_version = _run_checked([str(nvcc), "--version"])
    if "Cuda compilation tools" not in nvcc_version:
        raise QualificationError("NVCC version output is not recognizable")
    build_dir = args.build_dir.expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    normal_tile = build_dir / "libnfn_muse_glimmer_train_tile_ops.so"
    strict_tile = build_dir / "libnfn_muse_glimmer_train_tile_ops_strict.so"
    binding = build_dir / (
        "_native_inference" + (sysconfig_suffix() or ".so")
    )
    trainer_dir = build_dir / "trainers"
    environment = os.environ.copy()
    cxx = str(_resolve_executable(environment.get("CXX"), "c++"))
    environment.update(
        {
            "NVCC": str(nvcc.resolve()),
            "CXX": cxx,
            "NFN_TILE_CUDA_ARCH": arch,
            "NFN_TILE_CUDA_STRICT_ARCH": arch,
            "NFN_TILE_CUDA_USE_TK_ATTENTION": "0",
            "NFN_NATIVE_BUILD_STRICT_TILE_OPS": "1",
        }
    )
    tile_output = _run_checked(
        ["bash", str(ROOT / "tools/build_native_train_tile_ops.sh"), str(normal_tile)],
        env=environment,
    )
    if not normal_tile.is_file() or not strict_tile.is_file():
        raise QualificationError(
            f"NVCC build did not produce both Tile libraries:\n{tile_output}"
        )
    binding_output = _run_checked(
        ["bash", str(ROOT / "tools/build_native_inference_binding.sh"), str(binding)],
        env={**environment, "PYTHON": sys.executable},
    )
    trainer_output = _run_checked(
        ["bash", str(ROOT / "tools/build_native_missing_trainers.sh"), str(trainer_dir)],
        env={**environment, "NFN_NATIVE_MISSING_BUILD_TARGETS": "muse-glimmer"},
    )
    trainer = trainer_dir / "nfn_muse_glimmer_native_train"
    if not binding.is_file() or not trainer.is_file():
        raise QualificationError(
            "native binding/trainer build did not produce expected outputs:\n"
            + binding_output
            + trainer_output
        )
    abis = _validate_tile_abis(strict_tile, normal_tile)
    source_sha, sources = _source_tree_proof()
    build = {
        "nvcc_path": str(nvcc.resolve()),
        "nvcc_version": nvcc_version.strip(),
        "nvcc_prepend_flags": environment.get("NVCC_PREPEND_FLAGS", ""),
        "unsupported_host_compiler_override": "-allow-unsupported-compiler"
        in environment.get("NVCC_PREPEND_FLAGS", "").split(),
        "host_cxx_path": cxx,
        "host_cxx_version": _run_checked([cxx, "--version"]).strip(),
        "python_path": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "cuda_arch": arch,
        "tile_compiler_extension": True,
        "training_tile_ops": {
            "path": str(normal_tile),
            "bytes": normal_tile.stat().st_size,
            "sha256": _sha256(normal_tile),
        },
        "strict_inference_tile_ops": {
            "path": str(strict_tile),
            "bytes": strict_tile.stat().st_size,
            "sha256": _sha256(strict_tile),
        },
        "resident_binding": {
            "path": str(binding),
            "bytes": binding.stat().st_size,
            "sha256": _sha256(binding),
        },
        "native_trainer": {
            "path": str(trainer),
            "bytes": trainer.stat().st_size,
            "sha256": _sha256(trainer),
        },
        "abi_versions": abis,
        "source_tree_sha256": source_sha,
        "sources": sources,
    }
    return build, binding, strict_tile


def _prepare_build(
    args: argparse.Namespace,
    probe: CudaProbe,
) -> tuple[RunConfig, dict[str, Any]]:
    arch = f"sm_{probe.compute_major}{probe.compute_minor}"
    build, binding, strict_tile = _build_native_binaries(args, arch)
    source_sha = str(build["source_tree_sha256"])
    contexts = _parse_csv_ints(args.contexts, "--contexts")
    companions = tuple(
        item.strip() for item in args.companions.split(",") if item.strip()
    )
    if args.require_dflash and "dflash" not in companions:
        companions += ("dflash",)
    if args.run_vision and args.profile != "bf16" and "mmproj" not in companions:
        companions += ("mmproj",)
    if len(set(companions)) != len(companions):
        raise QualificationError("companion list contains duplicates")
    grid = _parse_csv_ints(args.vision_grid, "--vision-grid", unique=False)
    if len(grid) != 3:
        raise QualificationError("--vision-grid must contain temporal,height,width")
    config = RunConfig(
        artifact=str(args.artifact.expanduser().resolve()),
        profile=args.profile,
        gpu_class=args.gpu_class,
        cuda_runtime_lib=args.cuda_runtime_lib,
        cuda_device=args.cuda_device,
        binding_lib=str(binding),
        tile_ops_lib=str(strict_tile),
        contexts=contexts,
        decode_tokens=args.decode_tokens,
        warmups=args.warmups,
        repetitions=args.repetitions,
        prompt_token_id=args.prompt_token_id,
        stop_token_id=args.stop_token_id,
        companions=companions,
        require_dflash=args.require_dflash,
        run_vision=args.run_vision,
        vision_patch_width=args.vision_patch_width,
        vision_grid=(grid[0], grid[1], grid[2]),
        source_tree_sha256=source_sha,
        weight_precision_request=_weight_precision_request_for_tier(
            probe.total_bytes, args.gpu_class
        ),
    )
    return config, build


def _run_build_only(args: argparse.Namespace) -> dict[str, Any]:
    build, _binding, _strict_tile = _build_native_binaries(
        args, _validated_cuda_arch(args.cuda_arch)
    )
    return {
        "schema": "neuralfn.muse_glimmer_cuda_build_qualification",
        "version": 1,
        "status": "source-built",
        "release_qualified": False,
        "reason": "compiler-only proof; GPU execution and sanitizer are separate gates",
        "source_tree_sha256": build["source_tree_sha256"],
        "build": build,
    }


def sysconfig_suffix() -> str:
    import sysconfig

    return str(sysconfig.get_config_var("EXT_SUFFIX") or ".so")


def _parse_csv_ints(
    raw: str,
    label: str,
    *,
    unique: bool = True,
) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise QualificationError(f"{label} must contain comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise QualificationError(f"{label} must contain positive integers")
    if unique and len(set(values)) != len(values):
        raise QualificationError(f"{label} must not contain duplicates")
    return values


def _validate_run_args(args: argparse.Namespace) -> None:
    if args.profile != PROFILE_BY_CLASS[args.gpu_class]:
        raise QualificationError(
            f"--gpu-class {args.gpu_class} requires --profile "
            f"{PROFILE_BY_CLASS[args.gpu_class]}"
        )
    if args.cuda_device < 0:
        raise QualificationError("--cuda-device must be non-negative")
    for label in ("decode_tokens", "repetitions", "vision_patch_width"):
        if getattr(args, label) <= 0:
            raise QualificationError(f"--{label.replace('_', '-')} must be positive")
    if args.warmups < 0:
        raise QualificationError("--warmups must be non-negative")
    contexts = _parse_csv_ints(args.contexts, "--contexts")
    if max(contexts) + args.decode_tokens > FULL_GEOMETRY["max_seq_len"]:
        raise QualificationError("contexts plus decode tokens exceed 131072")
    if args.profile == "bf16" and args.vision_patch_width != 1176:
        raise QualificationError("BF16 embedded vision requires patch width 1176")
    if args.profile != "bf16" and args.vision_patch_width != 588:
        raise QualificationError("packed mmproj vision requires patch width 588")
    companions = {
        item.strip() for item in args.companions.split(",") if item.strip()
    }
    if args.profile == "bf16" and "mmproj" in companions:
        raise QualificationError(
            "the full BF16 checkpoint embeds vision and must not attach mmproj"
        )
    if not args.artifact.exists():
        raise QualificationError(f"artifact does not exist: {args.artifact}")


def _worker_command(config_path: Path) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "_worker", str(config_path)]


def _parse_worker_stdout(stdout: str) -> dict[str, Any]:
    rows = [line.strip() for line in stdout.splitlines() if line.strip()]
    for row in reversed(rows):
        try:
            payload = json.loads(row)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("schema") == SCHEMA:
            return payload
    raise QualificationError("qualification worker did not emit its JSON payload")


def _invoke_worker(
    config: RunConfig,
    config_path: Path,
    *,
    environment: Mapping[str, str],
    sanitizer_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    config_path.write_text(json.dumps(asdict(config), sort_keys=True), encoding="utf-8")
    command = _worker_command(config_path)
    if sanitizer_path is not None:
        command = [
            str(sanitizer_path),
            "--tool",
            "memcheck",
            "--error-exitcode",
            "97",
            "--target-processes",
            "all",
            *command,
        ]
    return subprocess.run(
        command,
        cwd=ROOT,
        env=dict(environment),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _run_parent(args: argparse.Namespace) -> dict[str, Any]:
    _validate_run_args(args)
    probe = CudaProbe(args.cuda_runtime_lib, args.cuda_device)
    _validate_gpu_class(probe.total_bytes, args.gpu_class)
    config, build = _prepare_build(args, probe)
    build_dir = args.build_dir.expanduser().resolve()
    config = replace(
        config,
        progress_path=str(build_dir / "worker-progress.json"),
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )

    sanitizer_path: Path | None = None
    sanitizer_proof: dict[str, Any] | None = None
    if args.compute_sanitizer == "required":
        sanitizer_path = _resolve_executable(
            args.compute_sanitizer_bin, "compute-sanitizer"
        )
        kernel_probe = _build_cuda_kernel_probe(
            build,
            Path(config.tile_ops_lib),
            config.cuda_runtime_lib,
            build_dir,
        )
        sanitizer_proof = _run_kernel_probe_sanitizers(
            kernel_probe,
            sanitizer_path,
            cuda_device=config.cuda_device,
            environment=environment,
        )

    completed = _invoke_worker(
        config,
        build_dir / "worker-config.json",
        environment=environment,
    )
    if completed.returncode != 0:
        raise QualificationError(
            f"GPU benchmark worker failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    payload = _parse_worker_stdout(completed.stdout)
    sanitizer_status = "passed" if sanitizer_proof is not None else "skipped"
    payload["status"] = "qualified" if sanitizer_status == "passed" else "measured"
    payload["build"] = build
    if sanitizer_proof is not None:
        sanitizer_proof["source_tree_sha256"] = payload["source_tree_sha256"]
        sanitizer_proof["benchmark_artifact_sha256"] = payload["artifact"][
            "selected_artifact_sha256"
        ]
        payload["compute_sanitizer"] = sanitizer_proof
    else:
        payload["compute_sanitizer"] = {
            "mode": args.compute_sanitizer,
            "status": "skipped",
            "scope": "real-device-kernel-probe-before-full-artifact-benchmark",
            "path": None,
            "zero_error_summary": False,
        }
    payload["release_qualified"] = sanitizer_status == "passed"
    return payload


def _load_result(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError(f"cannot read qualification result {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise QualificationError(f"qualification result {path} is not an object")
    return payload


def _verify_result(payload: Mapping[str, Any], *, minimum_long_context: int) -> None:
    if payload.get("schema") != SCHEMA or payload.get("version") != VERSION:
        raise QualificationError("qualification result schema/version mismatch")
    if payload.get("status") != "qualified" or payload.get("release_qualified") is not True:
        raise QualificationError("result was not source-built and sanitizer-qualified")
    gpu_class = str(payload.get("gpu_class"))
    profile = str(payload.get("profile"))
    if PROFILE_BY_CLASS.get(gpu_class) != profile:
        raise QualificationError("result profile/GPU-class mapping is invalid")
    hardware = payload.get("hardware")
    if not isinstance(hardware, Mapping):
        raise QualificationError("result hardware proof is missing")
    total = _require_int(hardware, "total_bytes", minimum=1)
    _validate_gpu_class(total, gpu_class)
    expected_request = _weight_precision_request_for_tier(total, gpu_class)
    build = payload.get("build")
    if not isinstance(build, Mapping):
        raise QualificationError("result NVCC build proof is missing")
    if build.get("source_tree_sha256") != payload.get("source_tree_sha256"):
        raise QualificationError("result source/build hashes disagree")
    sanitizer = payload.get("compute_sanitizer")
    if (
        not isinstance(sanitizer, Mapping)
        or sanitizer.get("status") != "passed"
        or sanitizer.get("zero_error_summary") is not True
        or sanitizer.get("scope")
        != "real-device-kernel-probe-before-full-artifact-benchmark"
    ):
        raise QualificationError("result lacks a passing compute-sanitizer proof")
    sanitizer_tools = sanitizer.get("tools")
    if not isinstance(sanitizer_tools, Mapping) or set(sanitizer_tools) != set(
        SANITIZER_TOOLS
    ):
        raise QualificationError("result sanitizer tool matrix is incomplete")
    for tool in SANITIZER_TOOLS:
        row = sanitizer_tools[tool]
        if (
            not isinstance(row, Mapping)
            or row.get("status") != "passed"
            or row.get("zero_error_summary") is not True
        ):
            raise QualificationError(f"result lacks a passing {tool} proof")
    sanitizer_probe = sanitizer.get("probe")
    strict_tile = build.get("strict_inference_tile_ops")
    if (
        not isinstance(sanitizer_probe, Mapping)
        or not isinstance(strict_tile, Mapping)
        or sanitizer_probe.get("strict_tile_sha256") != strict_tile.get("sha256")
        or sanitizer.get("source_tree_sha256") != payload.get("source_tree_sha256")
        or sanitizer.get("kernels") != list(KERNEL_PROBE_NAMES)
    ):
        raise QualificationError("result sanitizer probe is not bound to the built source")
    stats = payload.get("model_stats")
    if not isinstance(stats, Mapping):
        raise QualificationError("result model telemetry is missing")
    for key, expected in FULL_GEOMETRY.items():
        if stats.get(key) != expected:
            raise QualificationError(f"result geometry {key} is not full-size")
    load = payload.get("load")
    expected_selection = "auto-vram" if expected_request == "auto" else "explicit"
    if (
        not isinstance(load, Mapping)
        or load.get("requested_weight_precision") != expected_request
        or load.get("effective_weight_precision") != profile
        or load.get("weight_precision_selection") != expected_selection
        or load.get("speculative_decoding") != "dflash"
    ):
        raise QualificationError("result load selection/speculation proof is invalid")
    tier = payload.get("profile_tier")
    if tier is not None and (
        not isinstance(tier, Mapping)
        or tier.get("minimum_total_vram_bytes")
        != GPU_CLASS_MINIMUM_BYTES[gpu_class]
        or tier.get("physical_total_vram_bytes") != total
        or tier.get("larger_device") != (expected_request != "auto")
    ):
        raise QualificationError("result profile-tier capacity proof is invalid")
    resident = _require_int(stats, "cuda_resident_weight_bytes", minimum=1)
    if resident < MINIMUM_CUDA_WEIGHT_BYTES[profile]:
        raise QualificationError("result CUDA resident bytes are below the full-size floor")
    if stats.get("whole_model_cuda") is not True or stats.get("vision_cuda") is not True:
        raise QualificationError("result does not prove target+vision CUDA execution")
    if stats.get("dflash_cuda") is not True or stats.get("dflash_loaded") is not True:
        raise QualificationError("result does not prove DFlash CUDA execution")
    if _require_int(stats, "cpu_model_compute_rows") != 0:
        raise QualificationError("result reports CPU model-compute rows")
    contexts = payload.get("contexts")
    if not isinstance(contexts, list) or not contexts:
        raise QualificationError("result contains no context benchmarks")
    context_values = [
        _require_int(row, "prompt_context_tokens", minimum=1)
        for row in contexts
        if isinstance(row, Mapping)
    ]
    if len(context_values) != len(contexts) or max(context_values) < minimum_long_context:
        raise QualificationError(
            f"result does not include a context >= {minimum_long_context} tokens"
        )
    if payload.get("vision") is None:
        raise QualificationError("result contains no CUDA vision measurement")
    artifact = payload.get("artifact")
    selected_sha = artifact.get("selected_artifact_sha256") if isinstance(artifact, Mapping) else None
    if not isinstance(selected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", selected_sha) is None:
        raise QualificationError("result selected artifact digest is missing")
    expected_kquant_sha = CANONICAL_KQUANT_SHA256.get(profile)
    if expected_kquant_sha is not None and selected_sha != expected_kquant_sha:
        raise QualificationError("result used a noncanonical K-Quant artifact")
    if sanitizer.get("benchmark_artifact_sha256") != selected_sha:
        raise QualificationError(
            "sanitizer proof is not bound to the benchmark artifact result"
        )
    memory = payload.get("memory")
    if not isinstance(memory, Mapping) or _require_int(
        memory, "peak_sampled_delta_bytes", minimum=1
    ) <= 0:
        raise QualificationError("result contains no positive VRAM measurement")


def _verify_matrix(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.result) != 3:
        raise QualificationError("matrix verification requires exactly three result files")
    rows: dict[str, dict[str, Any]] = {}
    source_hash: str | None = None
    for path in args.result:
        payload = _load_result(path)
        _verify_result(payload, minimum_long_context=args.minimum_long_context)
        gpu_class = str(payload["gpu_class"])
        if gpu_class in rows:
            raise QualificationError(f"duplicate {gpu_class}-GB result")
        if source_hash is None:
            source_hash = str(payload["source_tree_sha256"])
        elif payload["source_tree_sha256"] != source_hash:
            raise QualificationError("matrix results were built from different source trees")
        rows[gpu_class] = payload
    if set(rows) != set(PROFILE_BY_CLASS):
        raise QualificationError("matrix must contain 24-, 32-, and 80-GB results")
    return {
        "schema": "neuralfn.muse_glimmer_gpu_matrix_qualification",
        "version": 1,
        "status": "qualified",
        "source_tree_sha256": source_hash,
        "minimum_long_context": args.minimum_long_context,
        "results": {
            gpu_class: {
                "profile": rows[gpu_class]["profile"],
                "artifact_sha256": rows[gpu_class]["artifact"][
                    "selected_artifact_sha256"
                ],
                "total_bytes": rows[gpu_class]["hardware"]["total_bytes"],
                "peak_sampled_delta_bytes": rows[gpu_class]["memory"][
                    "peak_sampled_delta_bytes"
                ],
                "max_prompt_context": max(
                    row["prompt_context_tokens"] for row in rows[gpu_class]["contexts"]
                ),
            }
            for gpu_class in ("24", "32", "80")
        },
    }


def _parser() -> FailClosedParser:
    parser = FailClosedParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", allow_abbrev=False)
    build.add_argument("--cuda-arch", required=True)
    build.add_argument("--nvcc")
    build.add_argument("--build-dir", type=Path, required=True)
    build.add_argument("--json-out", type=Path)

    run = subparsers.add_parser("run", allow_abbrev=False)
    run.add_argument("--artifact", type=Path, required=True)
    run.add_argument("--profile", choices=tuple(MINIMUM_CUDA_WEIGHT_BYTES), required=True)
    run.add_argument(
        "--gpu-class",
        choices=("24", "32", "80"),
        required=True,
        help=(
            "profile qualification tier in decimal GB; the physical GPU may be "
            "larger, and its exact total/peak usage is recorded"
        ),
    )
    run.add_argument("--cuda-runtime-lib", required=True)
    run.add_argument("--cuda-device", type=int, default=0)
    run.add_argument("--nvcc")
    run.add_argument("--build-dir", type=Path, required=True)
    run.add_argument("--contexts", default="128,2048,8192")
    run.add_argument("--decode-tokens", type=int, default=16)
    run.add_argument("--warmups", type=int, default=1)
    run.add_argument("--repetitions", type=int, default=3)
    run.add_argument("--prompt-token-id", type=int, default=200000)
    run.add_argument("--stop-token-id", type=int, default=200018)
    run.add_argument("--companions", default="")
    run.add_argument("--require-dflash", action="store_true")
    run.add_argument("--run-vision", action="store_true")
    run.add_argument("--vision-patch-width", type=int, required=True)
    run.add_argument("--vision-grid", default="1,2,2")
    run.add_argument(
        "--compute-sanitizer", choices=("required", "off"), default="required"
    )
    run.add_argument("--compute-sanitizer-bin")
    run.add_argument("--json-out", type=Path)

    verify = subparsers.add_parser("verify", allow_abbrev=False)
    verify.add_argument("--result", type=Path, action="append", required=True)
    verify.add_argument("--minimum-long-context", type=int, default=8192)
    verify.add_argument("--json-out", type=Path)

    worker = subparsers.add_parser(
        "_worker", help="internal sanitizer worker; do not invoke directly"
    )
    worker.add_argument("config", type=Path)
    return parser


def _write_payload(payload: Mapping[str, Any], output: Path | None) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is not None:
        destination = output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.command == "_worker":
            raw = json.loads(args.config.read_text(encoding="utf-8"))
            raw["contexts"] = tuple(raw["contexts"])
            raw["companions"] = tuple(raw["companions"])
            raw["vision_grid"] = tuple(raw["vision_grid"])
            config = RunConfig(**raw)
            # One-line output keeps parent parsing independent of sanitizer logs.
            sys.stdout.write(json.dumps(_run_worker(config), sort_keys=True) + "\n")
            return 0
        if args.command == "build":
            payload = _run_build_only(args)
        elif args.command == "run":
            payload = _run_parent(args)
        else:
            if args.minimum_long_context <= 0:
                raise QualificationError("--minimum-long-context must be positive")
            payload = _verify_matrix(args)
        _write_payload(payload, args.json_out)
        return 0
    except (QualificationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Muse Glimmer GPU qualification failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
