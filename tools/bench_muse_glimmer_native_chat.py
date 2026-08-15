#!/usr/bin/env python3
"""Benchmark a real Muse Glimmer chat turn in the native resident runtime.

The report is deliberately self-describing: it records the exact prompt-token
digest, artifact/profile, native binding and Tile libraries, CUDA memory
samples, target/DFlash counters, and separate prefill/decode timings.  It is
intended for same-machine comparisons with a pinned external runtime; it does
not manufacture a comparison from unmatched published numbers.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import struct
import threading
import time
from typing import Any, Sequence

from neuralfn.native_chat import (
    NativeChatMessage,
    load_native_text_codec,
    native_context_limit,
    native_stop_token_ids,
    native_text_stop_delimiters,
    parse_native_assistant_response,
    read_native_execution_manifest,
    resolve_native_chat_prompt,
    resolve_native_chat_renderer,
)
from neuralfn.native_inference import (
    GenerationConfig,
    KVCacheConfig,
    NativeInferenceModel,
    NativeModelLoadConfig,
)


def _load_binding(path: Path) -> Any:
    resolved = path.expanduser().resolve(strict=True)
    spec = importlib.util.spec_from_file_location("_native_inference", resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import resident binding: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CudaMemorySampler:
    """Sample process-global device free memory around one isolated run."""

    def __init__(self, runtime_path: Path, device: int, interval_seconds: float) -> None:
        self._runtime = ctypes.CDLL(str(runtime_path.expanduser().resolve(strict=True)))
        self._runtime.cudaSetDevice.argtypes = [ctypes.c_int]
        self._runtime.cudaSetDevice.restype = ctypes.c_int
        self._runtime.cudaMemGetInfo.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._runtime.cudaMemGetInfo.restype = ctypes.c_int
        self._device = int(device)
        self._interval = float(interval_seconds)
        self._samples: list[tuple[float, int, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _check(self, status: int, operation: str) -> None:
        if int(status) != 0:
            raise RuntimeError(f"{operation} failed with CUDA status {int(status)}")

    def sample(self) -> tuple[int, int]:
        self._check(self._runtime.cudaSetDevice(self._device), "cudaSetDevice")
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        self._check(
            self._runtime.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)),
            "cudaMemGetInfo",
        )
        row = (time.time(), int(free.value), int(total.value))
        self._samples.append(row)
        return row[1], row[2]

    def __enter__(self) -> "CudaMemorySampler":
        self.sample()

        def poll() -> None:
            while not self._stop.wait(self._interval):
                self.sample()

        self._thread = threading.Thread(target=poll, name="glimmer-vram-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval * 4.0))
        self.sample()

    def report(self) -> dict[str, Any]:
        if not self._samples:
            raise RuntimeError("CUDA memory sampler collected no samples")
        baseline = self._samples[0][1]
        minimum = min(row[1] for row in self._samples)
        totals = {row[2] for row in self._samples}
        if len(totals) != 1:
            raise RuntimeError("CUDA total memory changed during benchmark")
        return {
            "provider": "cudaMemGetInfo",
            "sample_interval_seconds": self._interval,
            "sample_count": len(self._samples),
            "total_bytes": totals.pop(),
            "baseline_free_bytes": baseline,
            "minimum_free_bytes": minimum,
            "peak_sampled_delta_bytes": max(0, baseline - minimum),
        }


def _distribution(values: Sequence[float]) -> dict[str, float | int | list[float]]:
    samples = [float(value) for value in values]
    if not samples:
        raise RuntimeError("timing distribution is empty")
    return {
        "samples": samples,
        "count": len(samples),
        "minimum": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "maximum": max(samples),
    }


def _token_digest(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(struct.pack("<I", int(token_id)))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--binding-lib", type=Path, required=True)
    parser.add_argument("--tile-ops-lib", type=Path, required=True)
    parser.add_argument("--cuda-runtime-lib", type=Path, required=True)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--weight-precision",
        choices=("auto", "k-quant-dynamic", "k-quant-17gb", "bf16"),
        default="k-quant-17gb",
    )
    parser.add_argument("--dflash", action="store_true")
    parser.add_argument(
        "--compute-mode",
        choices=("strict", "throughput", "model-card"),
        default="strict",
        help=(
            "strict uses exact-zero temperature; throughput uses temperature=1/top-k=1 "
            "for a deterministic K-quant control; model-card uses temperature=1, "
            "top-p=.95, and top-k=64"
        ),
    )
    parser.add_argument(
        "--prompt",
        default="Explain why the sky is blue in two concise paragraphs.",
    )
    parser.add_argument("--system-prompt", default="Be concise.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--vram-sample-ms", type=float, default=5.0)
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Also write the complete benchmark report to this path.",
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.warmups < 0 or args.repetitions <= 0:
        parser.error("token count/repetitions must be positive and warmups non-negative")
    if args.vram_sample_ms <= 0.0:
        parser.error("--vram-sample-ms must be positive")

    artifact_root, manifest_path, manifest = read_native_execution_manifest(args.artifact)
    codec = load_native_text_codec(manifest, artifact_root=artifact_root)
    renderer = resolve_native_chat_renderer(
        manifest,
        "auto",
        allow_auto_fallback=False,
        artifact_root=artifact_root,
    ).renderer
    history = (
        (NativeChatMessage("system", args.system_prompt),)
        if args.system_prompt.strip()
        else ()
    )
    prompt = resolve_native_chat_prompt(
        codec=codec,
        renderer=renderer,
        mode="stateless",
        history=history,
        draft=args.prompt,
        context_limit=native_context_limit(manifest),
        reserved_output_tokens=args.max_new_tokens,
    )
    strict_compute = args.compute_mode == "strict"
    model_card_sampling = args.compute_mode == "model-card"
    generation = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0 if strict_compute else 1.0,
        top_k=64 if model_card_sampling else (None if strict_compute else 1),
        top_p=0.95 if model_card_sampling else 1.0,
        seed=1337,
        stop_token_ids=native_stop_token_ids(manifest),
    )
    companions = ("dflash",) if args.dflash else ()
    load_config = NativeModelLoadConfig(
        weight_precision=args.weight_precision,
        runtime="native-cuda",
        cuda_device=args.cuda_device,
        tile_ops_lib=str(args.tile_ops_lib.expanduser().resolve(strict=True)),
        cuda_runtime_lib=str(args.cuda_runtime_lib.expanduser().resolve(strict=True)),
        context_tokens=len(prompt.token_ids) + args.max_new_tokens,
        session_count=1,
        companion_checkpoints=companions,
        speculative_decoding="required" if args.dflash else "off",
    )
    binding = _load_binding(args.binding_lib)
    sampler = CudaMemorySampler(
        args.cuda_runtime_lib,
        args.cuda_device,
        args.vram_sample_ms / 1000.0,
    )
    prefill_times: list[float] = []
    decode_times: list[float] = []
    rates: list[float] = []
    outputs: list[tuple[int, ...]] = []
    trials: list[dict[str, Any]] = []
    load_started = time.perf_counter()
    with sampler:
        with NativeInferenceModel.load(
            args.artifact,
            binding=binding,
            kv_cache=KVCacheConfig(mode="full"),
            load_config=load_config,
        ) as model:
            load_seconds = time.perf_counter() - load_started
            model_stats_loaded = model.stats()
            for trial_index in range(args.warmups + args.repetitions):
                with model.create_session(seed=1337) as session:
                    started = time.perf_counter()
                    prefill = session.prefill(prompt.token_ids)
                    decode_started = time.perf_counter()
                    result = session.decode(generation)
                    finished = time.perf_counter()
                    session_stats = session.stats()
                if trial_index < args.warmups:
                    continue
                prefill_seconds = decode_started - started
                decode_seconds = finished - decode_started
                completion_tokens = len(result.token_ids)
                if completion_tokens <= 0 or decode_seconds <= 0.0:
                    raise RuntimeError("native benchmark produced no timed completion tokens")
                decoded = codec.decode(result.token_ids)
                response = parse_native_assistant_response(
                    decoded,
                    renderer,
                    delimiters=native_text_stop_delimiters(manifest, renderer),
                    token_ids=result.token_ids,
                    codec=codec,
                )
                prefill_times.append(prefill_seconds)
                decode_times.append(decode_seconds)
                rates.append(completion_tokens / decode_seconds)
                outputs.append(tuple(result.token_ids))
                trials.append(
                    {
                        "prefill_seconds": prefill_seconds,
                        "decode_seconds": decode_seconds,
                        "completion_tokens": completion_tokens,
                        "tokens_per_second": completion_tokens / decode_seconds,
                        "finish_reason": result.finish_reason,
                        "visible_text": response.visible_text,
                        "atem_reasoning_hidden": bool(response.reasoning_text),
                        "reasoning_tokens": response.reasoning_tokens,
                        "prefill": prefill,
                        "speculative_proposed_tokens": result.speculative_proposed_tokens,
                        "speculative_accepted_tokens": result.speculative_accepted_tokens,
                        "speculative_rejected_tokens": result.speculative_rejected_tokens,
                        "speculative_target_rows": result.speculative_target_rows,
                        "speculative_assistant_blocks": result.speculative_assistant_blocks,
                        "session_stats": session_stats,
                    }
                )
            model_stats_final = model.stats()
    if args.compute_mode != "model-card" and any(output != outputs[0] for output in outputs[1:]):
        raise RuntimeError("deterministic greedy output changed across benchmark repetitions")
    report = {
        "schema": "neuralfn.muse_glimmer_native_chat_benchmark",
        "version": 1,
        "manifest": str(manifest_path),
        "binding_lib": str(args.binding_lib.expanduser().resolve(strict=True)),
        "tile_ops_lib": str(args.tile_ops_lib.expanduser().resolve(strict=True)),
        "cuda_runtime_lib": str(args.cuda_runtime_lib.expanduser().resolve(strict=True)),
        "cuda_device": args.cuda_device,
        "requested_weight_precision": args.weight_precision,
        "compute_mode": args.compute_mode,
        "dflash": bool(args.dflash),
        "prompt_tokens": len(prompt.token_ids),
        "prompt_token_ids_sha256_le_u32": _token_digest(prompt.token_ids),
        "max_new_tokens": args.max_new_tokens,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "load_seconds": load_seconds,
        "prefill_seconds": _distribution(prefill_times),
        "decode_seconds": _distribution(decode_times),
        "decode_tokens_per_second": _distribution(rates),
        "output_token_ids": list(outputs[0]),
        "output_token_ids_sha256_le_u32": _token_digest(outputs[0]),
        "memory": sampler.report(),
        "model_stats_loaded": model_stats_loaded,
        "model_stats_final": model_stats_final,
        "trials": trials,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        destination = args.json_out.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
