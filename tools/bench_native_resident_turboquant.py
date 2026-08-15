#!/usr/bin/env python3
"""Transfer-inclusive resident TurboQuant benchmark.

Each mode/context cell runs in a fresh child process.  The harness reports raw
measurements only; it deliberately does not calculate or claim speedups.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuralfn.native_inference import (  # noqa: E402
    GenerationConfig,
    KVCacheConfig,
    NativeInferenceModel,
)


SCHEMA = "neuralfn.native_resident_turboquant_benchmark"
VERSION = 1
DEFAULT_CONTEXTS = (1024, 4096, 16384)
MODE_ORDER = ("full", "mse-cpu", "qjl-cpu", "mse-tile", "qjl-tile")


class BenchmarkError(RuntimeError):
    """A fail-closed benchmark contract error."""


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise BenchmarkError(message)


@dataclass(frozen=True, slots=True)
class Mode:
    name: str
    cache_mode: str
    profile: str | None
    attention_backend: str


MODES = {
    "full": Mode("full", "full", None, "cpu"),
    "mse-cpu": Mode("mse-cpu", "turboquant", "mse-3.5", "cpu"),
    "qjl-cpu": Mode("qjl-cpu", "turboquant", "qjl-3.5", "cpu"),
    "mse-tile": Mode("mse-tile", "turboquant", "mse-3.5", "tile-cuda"),
    "qjl-tile": Mode("qjl-tile", "turboquant", "qjl-3.5", "tile-cuda"),
}


@dataclass(frozen=True, slots=True)
class Config:
    artifact: Path
    binding_lib: Path
    tile_ops_lib: Path | None
    cuda_runtime_lib: str | None
    cuda_device: int
    contexts: tuple[int, ...]
    decode_tokens: int
    quality_window: int
    warmups: int
    repetitions: int
    tokens: tuple[int, ...]
    json_out: Path | None
    worker: bool = False
    worker_mode: str | None = None
    worker_pass: str | None = None


def _parser() -> JsonArgumentParser:
    parser = JsonArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--binding-lib", required=True, type=Path)
    parser.add_argument("--tile-ops-lib", type=Path)
    parser.add_argument("--cuda-runtime-lib")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--contexts", default=",".join(map(str, DEFAULT_CONTEXTS)))
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--quality-window", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--tokens-file",
        type=Path,
        help=(
            "Optional JSON array or comma/whitespace-separated token corpus. "
            "When omitted, deterministic token 0 is repeated; this is a mechanics "
            "benchmark and the JSON labels the quality corpus synthetic."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=MODE_ORDER, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-pass",
        choices=("timing", "quality-vram"),
        help=argparse.SUPPRESS,
    )
    return parser


def _parse_contexts(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise BenchmarkError("--contexts must be a CSV of positive integers") from exc
    if not values or any(value <= 0 for value in values) or len(set(values)) != len(values):
        raise BenchmarkError("--contexts must contain unique positive integers")
    return values


def _load_tokens(path: Path | None, required: int) -> tuple[tuple[int, ...], str]:
    if path is None:
        return (0,) * required, "synthetic-repeated-token-0"
    if not path.is_file():
        raise BenchmarkError(f"token corpus does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        if text.lstrip().startswith("["):
            raw: Any = json.loads(text)
            if not isinstance(raw, list):
                raise BenchmarkError("token corpus JSON root must be an array")
        else:
            raw = [int(item) for item in text.replace(",", " ").split()]
    except (json.JSONDecodeError, ValueError) as exc:
        raise BenchmarkError("token corpus must contain only integer token IDs") from exc
    if len(raw) < required:
        raise BenchmarkError(
            f"token corpus has {len(raw)} tokens but {required} are required"
        )
    tokens: list[int] = []
    for index, value in enumerate(raw[:required]):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BenchmarkError(f"token corpus entry {index} is not a non-negative integer")
        tokens.append(value)
    return tuple(tokens), "provided-token-corpus"


def _config_from_args(args: argparse.Namespace) -> tuple[Config, str]:
    contexts = _parse_contexts(args.contexts)
    for label in ("decode_tokens", "quality_window", "repetitions"):
        value = getattr(args, label)
        if isinstance(value, bool) or value <= 0:
            raise BenchmarkError(f"--{label.replace('_', '-')} must be positive")
    if isinstance(args.warmups, bool) or args.warmups < 0:
        raise BenchmarkError("--warmups must be non-negative")
    if isinstance(args.cuda_device, bool) or args.cuda_device < 0:
        raise BenchmarkError("--cuda-device must be non-negative")
    largest_window = max(args.decode_tokens, args.quality_window)
    if any(context <= largest_window for context in contexts):
        raise BenchmarkError(
            "each context must be greater than decode-tokens and quality-window"
        )
    artifact = args.artifact.expanduser().resolve()
    binding_lib = args.binding_lib.expanduser().resolve()
    tile_ops_lib = (
        args.tile_ops_lib.expanduser().resolve() if args.tile_ops_lib is not None else None
    )
    if not artifact.exists():
        raise BenchmarkError(f"artifact does not exist: {artifact}")
    if not binding_lib.is_file():
        raise BenchmarkError(f"resident binding library does not exist: {binding_lib}")
    if tile_ops_lib is not None and not tile_ops_lib.is_file():
        raise BenchmarkError(f"Tile-CUDA library does not exist: {tile_ops_lib}")
    if tile_ops_lib is not None and not args.cuda_runtime_lib:
        raise BenchmarkError(
            "Tile modes require --cuda-runtime-lib so VRAM uses the same explicit runtime"
        )
    required_tokens = max(contexts)
    tokens, corpus_kind = _load_tokens(args.tokens_file, required_tokens)
    config = Config(
        artifact=artifact,
        binding_lib=binding_lib,
        tile_ops_lib=tile_ops_lib,
        cuda_runtime_lib=args.cuda_runtime_lib,
        cuda_device=args.cuda_device,
        contexts=contexts,
        decode_tokens=args.decode_tokens,
        quality_window=args.quality_window,
        warmups=args.warmups,
        repetitions=args.repetitions,
        tokens=tokens,
        json_out=(args.json_out.expanduser().resolve() if args.json_out else None),
        worker=bool(args.worker),
        worker_mode=args.worker_mode,
        worker_pass=args.worker_pass,
    )
    if config.worker and (
        config.worker_mode is None
        or config.worker_pass is None
        or len(config.contexts) != 1
    ):
        raise BenchmarkError(
            "worker mode requires --worker-mode, --worker-pass, and exactly one context"
        )
    if config.worker_mode and config.worker_mode.endswith("-tile") and tile_ops_lib is None:
        raise BenchmarkError("Tile worker mode requires --tile-ops-lib")
    return config, corpus_kind


def _load_binding(path: Path) -> Any:
    """Load the exact compiled extension supplied by the caller."""

    spec = importlib.util.spec_from_file_location("_native_inference", path)
    if spec is None or spec.loader is None:
        raise BenchmarkError(f"cannot create an import spec for resident binding: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except BaseException as exc:
        raise BenchmarkError(f"failed to load resident binding {path}: {exc}") from exc
    return module


class CudaMemInfoTracker:
    """Sample device-global CUDA usage relative to a fresh-worker baseline.

    This is intentionally not described as a process allocator high-water mark.
    Public resident calls are synchronous and Tile cache allocations persist for
    the session lifetime, so checkpoints immediately after those calls retain
    the allocation state long enough to sample it.
    """

    provider = "cudaMemGetInfo"
    scope = "sampled-device-global-baseline-subtracted"

    def __init__(self, runtime: str, device: int) -> None:
        try:
            self._library = ctypes.CDLL(runtime)
        except OSError as exc:
            raise BenchmarkError(f"cannot load CUDA runtime {runtime!r}: {exc}") from exc
        try:
            self._set_device = self._library.cudaSetDevice
            self._mem_get_info = self._library.cudaMemGetInfo
        except AttributeError as exc:
            raise BenchmarkError(
                f"CUDA runtime {runtime!r} does not expose cudaSetDevice/cudaMemGetInfo"
            ) from exc
        self._set_device.argtypes = [ctypes.c_int]
        self._set_device.restype = ctypes.c_int
        self._mem_get_info.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._mem_get_info.restype = ctypes.c_int
        status = int(self._set_device(device))
        if status != 0:
            raise BenchmarkError(f"cudaSetDevice({device}) failed with status {status}")
        self.runtime = runtime
        self.device = device
        self._baseline_free, self._total = self._sample_raw()
        self._minimum_free = self._baseline_free
        self._samples: list[dict[str, Any]] = [
            {
                "label": "fresh_worker_baseline",
                "free_bytes": self._baseline_free,
                "allocated_delta_bytes": 0,
            }
        ]

    def _sample_raw(self) -> tuple[int, int]:
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        status = int(self._mem_get_info(ctypes.byref(free), ctypes.byref(total)))
        if status != 0:
            raise BenchmarkError(f"cudaMemGetInfo failed with status {status}")
        if free.value > total.value or total.value <= 0:
            raise BenchmarkError("cudaMemGetInfo returned invalid free/total memory")
        return int(free.value), int(total.value)

    def checkpoint(self, label: str) -> None:
        free, total = self._sample_raw()
        if total != self._total:
            raise BenchmarkError("CUDA total memory changed during isolated worker measurement")
        # A free-memory increase can be benign lazy runtime cleanup.  It never
        # becomes a negative allocation claim; the baseline remains fixed.
        delta = max(0, self._baseline_free - free)
        self._minimum_free = min(self._minimum_free, free)
        self._samples.append(
            {"label": label, "free_bytes": free, "allocated_delta_bytes": delta}
        )

    def result(self, *, required: bool) -> dict[str, Any]:
        peak = max(0, self._baseline_free - self._minimum_free)
        if required and peak <= 0:
            raise BenchmarkError(
                "Tile-CUDA VRAM was not measurable as a positive cudaMemGetInfo delta"
            )
        return {
            "status": "measured" if peak > 0 else "measured-no-positive-delta",
            "required": required,
            "provider": self.provider,
            "scope": self.scope,
            "is_process_allocator_high_water_mark": False,
            "contamination_scope": (
                "device-global free-memory changes; run on an otherwise idle selected device"
            ),
            "runtime": self.runtime,
            "device": self.device,
            "baseline_free_bytes": self._baseline_free,
            "device_total_bytes": self._total,
            "peak_sampled_delta_bytes": peak,
            "sample_count": len(self._samples),
            "samples": self._samples,
        }


def _mode_cache_config(config: Config, mode: Mode) -> KVCacheConfig:
    if mode.cache_mode == "full":
        return KVCacheConfig(mode="full")
    kwargs: dict[str, Any] = {
        "mode": "turboquant",
        "turboquant_profile": mode.profile,
        "turboquant_attention_backend": mode.attention_backend,
    }
    if mode.attention_backend == "tile-cuda":
        assert config.tile_ops_lib is not None
        kwargs.update(
            tile_ops_lib=str(config.tile_ops_lib),
            cuda_runtime_lib=config.cuda_runtime_lib,
            cuda_device=config.cuda_device,
        )
    return KVCacheConfig(**kwargs)


def _nonnegative_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkError(f"session telemetry {key!r} must be a non-negative integer")
    return value


def _positive_int(mapping: Mapping[str, Any], key: str) -> int:
    value = _nonnegative_int(mapping, key)
    if value <= 0:
        raise BenchmarkError(f"model/session telemetry {key!r} must be positive")
    return value


def _distribution(samples: Sequence[float]) -> dict[str, Any]:
    if not samples or any(not math.isfinite(value) or value <= 0.0 for value in samples):
        raise BenchmarkError("timing samples must be finite positive numbers")
    values = [float(value) for value in samples]
    return {
        "samples": values,
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _integer_samples(samples: Sequence[int]) -> dict[str, Any]:
    if not samples or any(isinstance(value, bool) or value < 0 for value in samples):
        raise BenchmarkError("telemetry samples must be non-negative integers")
    values = list(samples)
    return {
        "samples": values,
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
    }


def _generation(count: int, *, seed: int, vocab_size: int) -> GenerationConfig:
    del vocab_size  # token bounds are proved from model telemetry by the worker
    return GenerationConfig(
        max_new_tokens=count,
        temperature=0.0,
        top_k=None,
        top_p=1.0,
        seed=seed,
        # The public session applies artifact stop tokens when this is empty.
        # A resulting short generation fails the fixed-size trial explicitly;
        # the harness never injects an out-of-vocabulary sentinel.
        stop_token_ids=(),
    )


def _require_generation(result: Any, expected: int, label: str) -> tuple[int, ...]:
    token_ids = getattr(result, "token_ids", None)
    if not isinstance(token_ids, tuple):
        try:
            token_ids = tuple(token_ids)
        except (TypeError, ValueError) as exc:
            raise BenchmarkError(f"{label} did not return token IDs") from exc
    if len(token_ids) != expected:
        raise BenchmarkError(f"{label} returned {len(token_ids)} tokens; expected {expected}")
    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in token_ids
    ):
        raise BenchmarkError(f"{label} returned an invalid token ID")
    return token_ids


def _timing_trial(
    model: Any,
    token_stream: Sequence[int],
    *,
    context: int,
    decode_tokens: int,
    seed: int,
    vocab_size: int,
    clock: Callable[[], float],
    tracker: CudaMemInfoTracker | None,
    label: str,
) -> dict[str, Any]:
    ttft_start = clock()
    with model.create_session(seed=seed) as session:
        session.prefill(token_stream[: context - 1])
        first = session.decode(_generation(1, seed=seed, vocab_size=vocab_size))
        _require_generation(first, 1, "TTFT decode")
        ttft = clock() - ttft_start
        if tracker is not None:
            tracker.checkpoint(f"{label}:ttft_session_live")

    with model.create_session(seed=seed) as session:
        session.prefill(token_stream[: context - decode_tokens])
        if tracker is not None:
            tracker.checkpoint(f"{label}:decode_prefill_complete")
        decode_start = clock()
        generated = session.decode(
            _generation(decode_tokens, seed=seed, vocab_size=vocab_size)
        )
        elapsed = clock() - decode_start
        greedy = _require_generation(generated, decode_tokens, "throughput decode")
        if elapsed <= 0.0:
            raise BenchmarkError("decode timer did not advance")
        stats = session.stats()
        if tracker is not None:
            tracker.checkpoint(f"{label}:decode_complete")
    return {
        "ttft_seconds": ttft,
        "decode_elapsed_seconds": elapsed,
        "decode_tokens_per_second": decode_tokens / elapsed,
        "greedy_token_ids": greedy,
        "stats": stats,
    }


def _logsumexp(logits: Sequence[float]) -> float:
    maximum = max(logits)
    total = math.fsum(math.exp(value - maximum) for value in logits)
    result = maximum + math.log(total)
    if not math.isfinite(result):
        raise BenchmarkError("quality logits produced a non-finite log-sum-exp")
    return result


def _quality_pass(
    model: Any,
    token_stream: Sequence[int],
    *,
    context: int,
    quality_window: int,
    seed: int,
    vocab_size: int,
    tracker: CudaMemInfoTracker | None,
) -> dict[str, Any]:
    complete = list(token_stream[:context])
    total_nll = 0.0
    greedy_by_position: list[tuple[int, int]] = []
    with model.create_session(seed=seed) as session:
        session.prefill(complete)
        stats = session.stats()
        if tracker is not None:
            tracker.checkpoint("quality_full_context_prefill_complete")
        for target_position in range(context - 1, context - quality_window - 1, -1):
            session.truncate(target_position)
            logits = session.current_logits()
            if len(logits) != vocab_size:
                raise BenchmarkError(
                    f"current_logits returned {len(logits)} values; expected {vocab_size}"
                )
            target = token_stream[target_position]
            if target >= vocab_size:
                raise BenchmarkError(f"quality target token {target} exceeds vocabulary")
            total_nll += _logsumexp(logits) - float(logits[target])
            greedy = max(range(vocab_size), key=lambda token: logits[token])
            greedy_by_position.append((target_position, greedy))
    mean_nll = total_nll / quality_window
    if mean_nll > math.log(sys.float_info.max):
        raise BenchmarkError("quality perplexity overflowed float64")
    perplexity = math.exp(mean_nll)
    if not math.isfinite(perplexity):
        raise BenchmarkError("quality perplexity is non-finite")
    return {
        "tokens_scored": quality_window,
        "negative_log_likelihood": total_nll,
        "mean_negative_log_likelihood": mean_nll,
        "perplexity": perplexity,
        "teacher_forced_greedy_token_ids": [
            token for _position, token in sorted(greedy_by_position)
        ],
        "stats": stats,
    }


def _validate_session_stats(stats: Mapping[str, Any], mode: Mode, context: int) -> None:
    if stats.get("effective_cache") != mode.cache_mode:
        raise BenchmarkError(
            f"session effective_cache={stats.get('effective_cache')!r}; "
            f"expected {mode.cache_mode!r}"
        )
    if _positive_int(stats, "token_count") != context:
        raise BenchmarkError("session telemetry token_count does not match benchmark context")
    _positive_int(stats, "cache_bytes")
    _positive_int(stats, "uncompressed_cache_bytes")
    _positive_int(stats, "cache_capacity_bytes")
    if mode.attention_backend == "tile-cuda":
        if stats.get("turboquant_attention_backend") != "tile-cuda":
            raise BenchmarkError("Tile mode did not report the Tile-CUDA attention backend")
        for key in (
            "turboquant_gpu_launches",
            "turboquant_row_uploads",
            "turboquant_h2d_bytes",
            "turboquant_d2h_bytes",
        ):
            _positive_int(stats, key)


def _validate_model_inputs(
    model: Any,
    config: Config,
    context: int,
) -> tuple[int, int]:
    model_stats = model.stats()
    max_seq_len = _positive_int(model_stats, "max_seq_len")
    vocab_size = _positive_int(model_stats, "vocab_size")
    if context > max_seq_len:
        raise BenchmarkError(
            f"requested context {context} exceeds artifact max_seq_len {max_seq_len}"
        )
    for index, token in enumerate(config.tokens[:context]):
        if token >= vocab_size:
            raise BenchmarkError(
                f"token corpus entry {index}={token} exceeds vocabulary {vocab_size}"
            )
    return max_seq_len, vocab_size


def _run_worker_timing(
    config: Config,
    mode: Mode,
    context: int,
    *,
    binding_loader: Callable[[Path], Any] = _load_binding,
    model_loader: Any = NativeInferenceModel,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    binding = binding_loader(config.binding_lib)
    cache_config = _mode_cache_config(config, mode)
    with model_loader.load(config.artifact, binding=binding, kv_cache=cache_config) as model:
        _max_seq_len, vocab_size = _validate_model_inputs(model, config, context)

        for warmup in range(config.warmups):
            _timing_trial(
                model,
                config.tokens,
                context=context,
                decode_tokens=config.decode_tokens,
                seed=1337 + warmup,
                vocab_size=vocab_size,
                clock=clock,
                tracker=None,
                label=f"warmup-{warmup}",
            )

        trials = [
            _timing_trial(
                model,
                config.tokens,
                context=context,
                decode_tokens=config.decode_tokens,
                seed=7331,
                vocab_size=vocab_size,
                clock=clock,
                tracker=None,
                label=f"repetition-{repetition}",
            )
            for repetition in range(config.repetitions)
        ]
        greedy = trials[0]["greedy_token_ids"]
        if any(trial["greedy_token_ids"] != greedy for trial in trials[1:]):
            raise BenchmarkError("strict greedy decode was not deterministic across repetitions")
        for trial in trials:
            _validate_session_stats(trial["stats"], mode, context)

    physical = [_positive_int(trial["stats"], "cache_bytes") for trial in trials]
    logical = [
        _positive_int(trial["stats"], "uncompressed_cache_bytes") for trial in trials
    ]
    capacity = [
        _positive_int(trial["stats"], "cache_capacity_bytes") for trial in trials
    ]
    transfers = {
        key: _integer_samples([_nonnegative_int(trial["stats"], key) for trial in trials])
        for key in (
            "turboquant_gpu_launches",
            "turboquant_row_uploads",
            "turboquant_h2d_bytes",
            "turboquant_d2h_bytes",
            "turboquant_cpu_compressed_attention_calls",
        )
        if all(key in trial["stats"] for trial in trials)
    }
    return {
        "status": "complete",
        "pass": "timing",
        "mode": {
            "name": mode.name,
            "cache_mode": mode.cache_mode,
            "turboquant_profile": mode.profile,
            "attention_backend": mode.attention_backend,
        },
        "context_tokens": context,
        "ttft_prompt_tokens": context - 1,
        "decode_prompt_tokens": context - config.decode_tokens,
        "timing": {
            "ttft_seconds": _distribution([trial["ttft_seconds"] for trial in trials]),
            "decode_elapsed_seconds": _distribution(
                [trial["decode_elapsed_seconds"] for trial in trials]
            ),
            "decode_tokens_per_second": _distribution(
                [trial["decode_tokens_per_second"] for trial in trials]
            ),
        },
        "cache": {
            "live_cache_bytes": _integer_samples(physical),
            "live_uncompressed_equivalent_bytes": _integer_samples(logical),
            "allocated_cache_capacity_bytes": _integer_samples(capacity),
            "telemetry_source": "NativeInferenceSession.stats",
            "scope_note": "device-global CUDA allocation is reported separately",
        },
        "transfers": transfers,
        "free_running_greedy": {
            "token_ids": list(greedy),
            "agreement_vs_full": None,
            "agreement_vs_cpu_same_profile": None,
        },
    }


def _run_worker_quality_vram(
    config: Config,
    mode: Mode,
    context: int,
    *,
    binding_loader: Callable[[Path], Any] = _load_binding,
    model_loader: Any = NativeInferenceModel,
    tracker_factory: Callable[[str, int], CudaMemInfoTracker] = CudaMemInfoTracker,
) -> dict[str, Any]:
    binding = binding_loader(config.binding_lib)
    tracker: CudaMemInfoTracker | None = None
    if mode.attention_backend == "tile-cuda" and config.cuda_runtime_lib is None:
        raise BenchmarkError("Tile mode cannot measure VRAM without --cuda-runtime-lib")
    if config.cuda_runtime_lib is not None:
        tracker = tracker_factory(config.cuda_runtime_lib, config.cuda_device)
    cache_config = _mode_cache_config(config, mode)
    with model_loader.load(config.artifact, binding=binding, kv_cache=cache_config) as model:
        _max_seq_len, vocab_size = _validate_model_inputs(model, config, context)
        if tracker is not None:
            tracker.checkpoint("model_loaded")
        quality = _quality_pass(
            model,
            config.tokens,
            context=context,
            quality_window=config.quality_window,
            seed=991,
            vocab_size=vocab_size,
            tracker=tracker,
        )
        _validate_session_stats(quality["stats"], mode, context)
    vram = (
        tracker.result(required=mode.attention_backend == "tile-cuda")
        if tracker is not None
        else {
            "status": "not-applicable-cpu-mode",
            "required": False,
            "provider": None,
            "scope": "no-cuda-backend-requested",
            "is_process_allocator_high_water_mark": False,
            "peak_sampled_delta_bytes": None,
        }
    )
    return {
        "status": "complete",
        "pass": "quality-vram",
        "mode": {
            "name": mode.name,
            "cache_mode": mode.cache_mode,
            "turboquant_profile": mode.profile,
            "attention_backend": mode.attention_backend,
        },
        "context_tokens": context,
        "quality": {
            "tokens_scored": quality["tokens_scored"],
            "negative_log_likelihood": quality["negative_log_likelihood"],
            "mean_negative_log_likelihood": quality["mean_negative_log_likelihood"],
            "perplexity": quality["perplexity"],
            "perplexity_delta_vs_full": None,
            "negative_log_likelihood_delta_vs_full": None,
            "mean_nll_delta_vs_full": None,
            "teacher_forced_greedy_token_ids": quality[
                "teacher_forced_greedy_token_ids"
            ],
            "teacher_forced_greedy_agreement_vs_full": None,
            "teacher_forced_greedy_agreement_vs_cpu_same_profile": None,
        },
        "vram": vram,
    }


def _file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _artifact_manifest_path(artifact: Path) -> Path:
    return artifact / "native-execution-manifest.json" if artifact.is_dir() else artifact


def _worker_command(
    config: Config,
    mode: Mode,
    context: int,
    tokens_file: Path,
    worker_pass: str,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--artifact",
        str(config.artifact),
        "--binding-lib",
        str(config.binding_lib),
        "--contexts",
        str(context),
        "--decode-tokens",
        str(config.decode_tokens),
        "--quality-window",
        str(config.quality_window),
        "--warmups",
        str(config.warmups),
        "--repetitions",
        str(config.repetitions),
        "--tokens-file",
        str(tokens_file),
        "--cuda-device",
        str(config.cuda_device),
        "--worker",
        "--worker-mode",
        mode.name,
        "--worker-pass",
        worker_pass,
    ]
    if config.tile_ops_lib is not None:
        command.extend(("--tile-ops-lib", str(config.tile_ops_lib)))
    if config.cuda_runtime_lib is not None:
        command.extend(("--cuda-runtime-lib", config.cuda_runtime_lib))
    return command


def _run_child_pass(
    config: Config,
    mode: Mode,
    context: int,
    tokens_file: Path,
    worker_pass: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    command = _worker_command(config, mode, context, tokens_file, worker_pass)
    completed = runner(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no child output"
        raise BenchmarkError(
            f"worker {mode.name}/{context}/{worker_pass} did not emit valid JSON: {detail}"
        ) from exc
    if (
        completed.returncode != 0
        or not isinstance(payload, dict)
        or payload.get("status") != "complete"
    ):
        error = payload.get("error") if isinstance(payload, dict) else None
        message = error.get("message") if isinstance(error, Mapping) else completed.stderr.strip()
        raise BenchmarkError(
            f"worker {mode.name}/{context}/{worker_pass} failed: "
            f"{message or 'unspecified worker error'}"
        )
    if payload.get("context_tokens") != context:
        raise BenchmarkError("worker returned a mismatched context")
    raw_mode = payload.get("mode")
    if not isinstance(raw_mode, Mapping) or raw_mode.get("name") != mode.name:
        raise BenchmarkError("worker returned a mismatched mode")
    if payload.get("pass") != worker_pass:
        raise BenchmarkError("worker returned a mismatched pass")
    return payload


def _run_child_cell(
    config: Config,
    mode: Mode,
    context: int,
    tokens_file: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    def run(worker_pass: str) -> dict[str, Any]:
        return _run_child_pass(
            config,
            mode,
            context,
            tokens_file,
            worker_pass,
            runner=runner,
        )

    if mode.attention_backend == "tile-cuda":
        # Prove the required provider/live Tile path before spending time on
        # latency measurements whose result could not be accepted without it.
        quality = run("quality-vram")
        timing = run("timing")
    else:
        timing = run("timing")
        quality = run("quality-vram")
    return {
        **timing,
        "pass": "merged-isolated-workers",
        "quality": quality["quality"],
        "vram": quality["vram"],
        "worker_isolation": {
            "timing_worker": True,
            "quality_vram_worker": True,
        },
    }


def _attach_full_reference(cells: Sequence[dict[str, Any]]) -> None:
    by_key = {
        (cell["mode"]["name"], cell["context_tokens"]): cell for cell in cells
    }

    def agreement(generated: Any, expected: Any) -> dict[str, Any]:
        if (
            not isinstance(generated, list)
            or not isinstance(expected, list)
            or not expected
            or len(generated) != len(expected)
        ):
            raise BenchmarkError("greedy comparison requires equal non-empty token sequences")
        matched = sum(
            left == right for left, right in zip(generated, expected, strict=True)
        )
        return {
            "matched_tokens": matched,
            "compared_tokens": len(expected),
            "rate": matched / len(expected),
        }

    for cell in cells:
        context = cell["context_tokens"]
        baseline = by_key.get(("full", context))
        if baseline is None:
            raise BenchmarkError(f"missing full-cache baseline for context {context}")
        quality = cell["quality"]
        reference = baseline["quality"]
        perplexity = float(quality["perplexity"])
        reference_perplexity = float(reference["perplexity"])
        if not math.isfinite(perplexity) or not math.isfinite(reference_perplexity):
            raise BenchmarkError("perplexity comparison requires finite values")
        signed_delta = perplexity - reference_perplexity
        quality["perplexity_delta_vs_full"] = {
            "signed": signed_delta,
            "absolute": abs(signed_delta),
            "relative": signed_delta / reference_perplexity,
        }
        quality["mean_nll_delta_vs_full"] = (
            float(quality["mean_negative_log_likelihood"])
            - float(reference["mean_negative_log_likelihood"])
        )
        quality["negative_log_likelihood_delta_vs_full"] = (
            float(quality["negative_log_likelihood"])
            - float(reference["negative_log_likelihood"])
        )
        quality["teacher_forced_greedy_agreement_vs_full"] = agreement(
            quality["teacher_forced_greedy_token_ids"],
            reference["teacher_forced_greedy_token_ids"],
        )
        cell["free_running_greedy"]["agreement_vs_full"] = agreement(
            cell["free_running_greedy"]["token_ids"],
            baseline["free_running_greedy"]["token_ids"],
        )

        mode_name = cell["mode"]["name"]
        if mode_name in {"mse-tile", "qjl-tile"}:
            cpu_name = mode_name.replace("-tile", "-cpu")
            cpu_reference = by_key.get((cpu_name, context))
            if cpu_reference is None:
                raise BenchmarkError(
                    f"missing same-profile CPU reference {cpu_name!r} at context {context}"
                )
            quality["teacher_forced_greedy_agreement_vs_cpu_same_profile"] = agreement(
                quality["teacher_forced_greedy_token_ids"],
                cpu_reference["quality"]["teacher_forced_greedy_token_ids"],
            )
            cell["free_running_greedy"]["agreement_vs_cpu_same_profile"] = agreement(
                cell["free_running_greedy"]["token_ids"],
                cpu_reference["free_running_greedy"]["token_ids"],
            )


def _run_parent(
    config: Config,
    corpus_kind: str,
    *,
    child_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    selected_names = list(MODE_ORDER[:3])
    if config.tile_ops_lib is not None:
        selected_names.extend(MODE_ORDER[3:])
    # Exercise required live Tile dependencies before spending time on CPU
    # cells. No partial result is emitted if any child fails.
    execution_names = (
        [name for name in selected_names if name.endswith("-tile")]
        + [name for name in selected_names if not name.endswith("-tile")]
    )
    cells_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="nfn-resident-turboquant-bench-") as temp:
        tokens_file = Path(temp) / "tokens.json"
        tokens_file.write_text(json.dumps(list(config.tokens)), encoding="utf-8")
        for mode_name in execution_names:
            mode = MODES[mode_name]
            for context in config.contexts:
                cells_by_key[(mode_name, context)] = _run_child_cell(
                    config,
                    mode,
                    context,
                    tokens_file,
                    runner=child_runner,
                )
    cells = [
        cells_by_key[(mode_name, context)]
        for mode_name in selected_names
        for context in config.contexts
    ]
    _attach_full_reference(cells)
    manifest_path = _artifact_manifest_path(config.artifact)
    if not manifest_path.is_file():
        raise BenchmarkError(f"artifact manifest does not exist: {manifest_path}")
    token_digest = hashlib.sha256(
        json.dumps(list(config.tokens), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "complete",
        "speedup_claimed": False,
        "benchmark_scope": {
            "fresh_process_per_mode_context": True,
            "separate_timing_and_quality_vram_workers": True,
            "public_native_inference_model_session_path": True,
            "timing_and_quality_separate": True,
            "timing_includes_public_prefill_decode_and_synchronous_transfers": True,
            "ttft_includes_session_creation": True,
            "model_load_included_in_ttft": False,
            "quality_method": "teacher-forced-current-logits-autoregressive-nll",
            "vram_method": "sampled-device-global-baseline-subtracted-cudaMemGetInfo",
            "vram_is_process_allocator_high_water_mark": False,
        },
        "config": {
            "contexts": list(config.contexts),
            "decode_tokens": config.decode_tokens,
            "quality_window": config.quality_window,
            "warmups": config.warmups,
            "repetitions": config.repetitions,
            "cuda_device": config.cuda_device,
            "cuda_runtime_lib": config.cuda_runtime_lib,
            "corpus_kind": corpus_kind,
            "corpus_token_count": len(config.tokens),
            "corpus_sha256": token_digest,
        },
        "inputs": {
            "artifact_manifest": _file_fingerprint(manifest_path),
            "binding_library": _file_fingerprint(config.binding_lib),
            "tile_ops_library": (
                _file_fingerprint(config.tile_ops_lib)
                if config.tile_ops_lib is not None
                else None
            ),
        },
        "modes": [
            {
                "name": mode_name,
                "cells": [cells_by_key[(mode_name, context)] for context in config.contexts],
            }
            for mode_name in selected_names
        ],
    }


def _failure_payload(error: BaseException, *, worker: bool) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "failed",
        "speedup_claimed": False,
        "scope": "worker" if worker else "parent",
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def _emit(payload: Mapping[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)


def main(argv: Sequence[str] | None = None) -> int:
    args: argparse.Namespace | None = None
    config: Config | None = None
    try:
        args = _parser().parse_args(list(argv) if argv is not None else None)
        config, corpus_kind = _config_from_args(args)
        if config.worker:
            assert config.worker_mode is not None
            assert config.worker_pass is not None
            worker = (
                _run_worker_timing
                if config.worker_pass == "timing"
                else _run_worker_quality_vram
            )
            payload = worker(config, MODES[config.worker_mode], config.contexts[0])
        else:
            payload = _run_parent(config, corpus_kind)
    except SystemExit as exc:
        return int(exc.code)
    except KeyboardInterrupt as exc:
        payload = _failure_payload(exc, worker=bool(config and config.worker))
        output = config.json_out if config is not None and not config.worker else None
        _emit(payload, output)
        return 130
    except BaseException as exc:
        payload = _failure_payload(
            exc,
            worker=bool(config and config.worker) or bool(args and args.worker),
        )
        output = config.json_out if config is not None and not config.worker else None
        _emit(payload, output)
        return 2
    output = config.json_out if config is not None and not config.worker else None
    _emit(payload, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
