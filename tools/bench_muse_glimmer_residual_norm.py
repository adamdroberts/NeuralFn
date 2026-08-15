#!/usr/bin/env python3
"""Measure Glimmer fused residual-capture/RMSNorm CUDA primitives."""

from __future__ import annotations

import argparse
from array import array
import ctypes
from pathlib import Path
import statistics

from bench_muse_glimmer_packed_linear import CudaRuntime


def _symbol(library: ctypes.CDLL, name: str, argument_types: list[object]):
    function = getattr(library, name)
    function.argtypes = argument_types
    function.restype = ctypes.c_int
    return function


def _median(runtime: CudaRuntime, launch, warmups: int, repetitions: int) -> float:
    for _ in range(warmups):
        launch()
    runtime.synchronize()
    return statistics.median(runtime.elapsed_ms(launch) for _ in range(repetitions))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--cuda-runtime-lib", type=Path, required=True)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--width", type=int, default=6656)
    parser.add_argument("--eps", type=float, default=1.0e-8)
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args()
    if args.rows <= 0 or args.width <= 0 or args.repetitions <= 0:
        parser.error("rows, width, and repetitions must be positive")

    runtime = CudaRuntime(args.cuda_runtime_lib.resolve(strict=True), args.cuda_device)
    library = ctypes.CDLL(str(args.sidecar.resolve(strict=True)), mode=ctypes.RTLD_LOCAL)
    rms_args = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_float, ctypes.c_bool, ctypes.c_void_p,
    ]
    rms = _symbol(
        library, "nfn_native_tile_glimmer_rms_norm_affine_float32_v1", rms_args
    )
    capture = _symbol(
        library,
        "nfn_native_tile_glimmer_rms_norm_affine_capture_residual_float32_v1",
        [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_float, ctypes.c_bool,
            ctypes.c_void_p,
        ],
    )
    fused_add = _symbol(
        library,
        "nfn_native_tile_glimmer_rms_norm_affine_add_residual_float32_v1",
        [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_float, ctypes.c_bool,
            ctypes.c_void_p,
        ],
    )
    add = _symbol(
        library,
        "nfn_native_tile_add_float32",
        [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_void_p,
        ],
    )

    count = args.rows * args.width
    values = array("f", (((index * 17) % 101 - 50) / 31.0 for index in range(count)))
    host = (ctypes.c_float * count).from_buffer_copy(values)
    nbytes = count * ctypes.sizeof(ctypes.c_float)
    source = runtime.malloc(nbytes)
    normalized = runtime.malloc(nbytes)
    residual = runtime.malloc(nbytes)
    output = runtime.malloc(nbytes)
    baseline = runtime.malloc(nbytes)
    try:
        runtime.upload(source, host, nbytes)

        def baseline_capture() -> None:
            runtime.copy_d2d_async(residual, source, nbytes)
            runtime.check(
                rms(source, None, normalized, args.rows, args.width, args.eps, False, None),
                "baseline RMSNorm capture",
            )

        def candidate_capture() -> None:
            runtime.check(
                capture(
                    source, None, normalized, residual, args.rows, args.width,
                    args.eps, False, None,
                ),
                "fused RMSNorm capture",
            )

        def baseline_add() -> None:
            runtime.check(
                rms(source, None, normalized, args.rows, args.width, args.eps, False, None),
                "baseline RMSNorm add",
            )
            runtime.check(add(residual, normalized, baseline, count, None), "baseline add")

        def candidate_add() -> None:
            runtime.check(
                fused_add(
                    source, None, residual, output, args.rows, args.width,
                    args.eps, False, None,
                ),
                "fused RMSNorm add",
            )

        baseline_capture_ms = _median(
            runtime, baseline_capture, args.warmups, args.repetitions
        )
        candidate_capture_ms = _median(
            runtime, candidate_capture, args.warmups, args.repetitions
        )
        baseline_add()
        candidate_add()
        runtime.synchronize()
        baseline_host = (ctypes.c_float * count)()
        output_host = (ctypes.c_float * count)()
        runtime.download(baseline_host, baseline, nbytes)
        runtime.download(output_host, output, nbytes)
        max_error = max(
            abs(float(left) - float(right))
            for left, right in zip(baseline_host, output_host, strict=True)
        )
        if max_error != 0.0:
            raise RuntimeError(f"fused residual add is not bit-identical: {max_error}")
        baseline_add_ms = _median(runtime, baseline_add, args.warmups, args.repetitions)
        candidate_add_ms = _median(runtime, candidate_add, args.warmups, args.repetitions)
        print(
            f"rows={args.rows} width={args.width}: "
            f"capture={baseline_capture_ms:.4f}->{candidate_capture_ms:.4f} ms "
            f"({baseline_capture_ms / candidate_capture_ms:.3f}x), "
            f"add={baseline_add_ms:.4f}->{candidate_add_ms:.4f} ms "
            f"({baseline_add_ms / candidate_add_ms:.3f}x), max_abs_error={max_error:g}"
        )
    finally:
        runtime.free(baseline)
        runtime.free(output)
        runtime.free(residual)
        runtime.free(normalized)
        runtime.free(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
