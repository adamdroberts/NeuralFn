#!/usr/bin/env python3
"""Compare two Glimmer packed-linear sidecars on one CUDA device.

This is a focused kernel benchmark, not a whole-model throughput claim.  It
uses a deterministic synthetic packed tensor at real Glimmer projection
geometry and verifies the candidate output against the baseline before timing.
"""

from __future__ import annotations

import argparse
from array import array
import ctypes
from dataclasses import dataclass
from pathlib import Path
import statistics
import struct
import time


ENCODINGS = {
    "q4_k": (12, 144),
    "q5_k": (13, 176),
    "q6_k": (14, 210),
}


class PackedWeightDescriptor(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("encoding", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("data_nbytes", ctypes.c_int64),
        ("output_dim", ctypes.c_int64),
        ("input_dim", ctypes.c_int64),
        ("row_stride_bytes", ctypes.c_int64),
        ("reserved0", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32),
        ("cuda_stream", ctypes.c_void_p),
    ]


@dataclass(frozen=True)
class Result:
    label: str
    samples_ms: tuple[float, ...]

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)


class CudaRuntime:
    def __init__(self, path: Path, device: int) -> None:
        self.lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
        self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self.lib.cudaSetDevice.restype = ctypes.c_int
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.cudaMemcpy.restype = ctypes.c_int
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int
        self.lib.cudaDeviceSynchronize.argtypes = []
        self.lib.cudaDeviceSynchronize.restype = ctypes.c_int
        self.lib.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaEventCreate.restype = ctypes.c_int
        self.lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaEventDestroy.restype = ctypes.c_int
        self.lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cudaEventRecord.restype = ctypes.c_int
        self.lib.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaEventSynchronize.restype = ctypes.c_int
        self.lib.cudaEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.cudaEventElapsedTime.restype = ctypes.c_int
        self.check(self.lib.cudaSetDevice(device), f"cudaSetDevice({device})")

    @staticmethod
    def check(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"{operation} failed with CUDA status {status}")

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(pointer), nbytes), "cudaMalloc")
        return pointer

    def upload(self, pointer: ctypes.c_void_p, host: ctypes.Array, nbytes: int) -> None:
        self.check(
            self.lib.cudaMemcpy(pointer, ctypes.cast(host, ctypes.c_void_p), nbytes, 1),
            "cudaMemcpy H2D",
        )

    def download(self, host: ctypes.Array, pointer: ctypes.c_void_p, nbytes: int) -> None:
        self.check(
            self.lib.cudaMemcpy(ctypes.cast(host, ctypes.c_void_p), pointer, nbytes, 2),
            "cudaMemcpy D2H",
        )

    def copy_d2d_async(
        self,
        destination: ctypes.c_void_p,
        source: ctypes.c_void_p,
        nbytes: int,
    ) -> None:
        self.check(
            self.lib.cudaMemcpyAsync(destination, source, nbytes, 3, None),
            "cudaMemcpyAsync D2D",
        )

    def synchronize(self) -> None:
        self.check(self.lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    def free(self, pointer: ctypes.c_void_p) -> None:
        self.check(self.lib.cudaFree(pointer), "cudaFree")

    def elapsed_ms(self, launch) -> float:
        start = ctypes.c_void_p()
        stop = ctypes.c_void_p()
        self.check(self.lib.cudaEventCreate(ctypes.byref(start)), "cudaEventCreate(start)")
        try:
            self.check(self.lib.cudaEventCreate(ctypes.byref(stop)), "cudaEventCreate(stop)")
            try:
                self.check(self.lib.cudaEventRecord(start, None), "cudaEventRecord(start)")
                launch()
                self.check(self.lib.cudaEventRecord(stop, None), "cudaEventRecord(stop)")
                self.check(self.lib.cudaEventSynchronize(stop), "cudaEventSynchronize")
                elapsed = ctypes.c_float()
                self.check(
                    self.lib.cudaEventElapsedTime(
                        ctypes.byref(elapsed), start, stop
                    ),
                    "cudaEventElapsedTime",
                )
                return float(elapsed.value)
            finally:
                self.check(self.lib.cudaEventDestroy(stop), "cudaEventDestroy(stop)")
        finally:
            self.check(self.lib.cudaEventDestroy(start), "cudaEventDestroy(start)")


def _packed_bytes(encoding: str, output_dim: int, input_dim: int) -> tuple[bytes, int]:
    _encoding_id, block_bytes = ENCODINGS[encoding]
    if input_dim % 256:
        raise ValueError("K-quant input dimension must be divisible by 256")
    blocks_per_row = input_dim // 256
    row_stride = blocks_per_row * block_bytes
    total = output_dim * row_stride
    values = bytearray((index * 29 + 17) & 0xFF for index in range(total))
    for row in range(output_dim):
        for block_index in range(blocks_per_row):
            offset = row * row_stride + block_index * block_bytes
            if encoding in {"q4_k", "q5_k"}:
                values[offset : offset + 2] = struct.pack("<e", 0.03125)
                values[offset + 2 : offset + 4] = struct.pack("<e", 0.015625)
            else:
                values[offset + 208 : offset + 210] = struct.pack("<e", 0.03125)
    return bytes(values), row_stride


def _linear(library_path: Path):
    library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_LOCAL)
    function = library.nfn_native_tile_linear_packed_weight_float32_v1
    function.argtypes = [
        ctypes.POINTER(PackedWeightDescriptor),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_bool,
    ]
    function.restype = ctypes.c_int
    return library, function


def _q8_linear(library_path: Path):
    library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_LOCAL)
    quantize = library.nfn_native_tile_quantize_q8_1_float32_v1
    quantize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    quantize.restype = ctypes.c_int
    linear = library.nfn_native_tile_linear_packed_weight_q8_1_float32_v1
    linear.argtypes = [
        ctypes.POINTER(PackedWeightDescriptor),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_bool,
    ]
    linear.restype = ctypes.c_int
    return library, quantize, linear


def _llama_mmq_provider(library_path: Path, device: int):
    library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_LOCAL)
    abi = library.nfn_experimental_llama_mmq_provider_abi
    abi.argtypes = []
    abi.restype = ctypes.c_uint32
    if abi() != 1:
        raise RuntimeError("unsupported experimental llama MMQ provider ABI")
    create = library.nfn_experimental_llama_mmq_provider_create
    create.argtypes = [ctypes.c_int]
    create.restype = ctypes.c_void_p
    destroy = library.nfn_experimental_llama_mmq_provider_destroy
    destroy.argtypes = [ctypes.c_void_p]
    destroy.restype = None
    linear = library.nfn_experimental_llama_mmq_provider_linear_f32
    linear.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    linear.restype = ctypes.c_int
    context = create(device)
    if not context:
        raise RuntimeError("experimental llama MMQ provider initialization failed")
    return library, context, destroy, linear


def _standalone_mmq(library_path: Path):
    library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_LOCAL)
    workspace_bytes = library.nfn_candidate_k_mmq_workspace_bytes
    workspace_bytes.argtypes = [ctypes.c_int64, ctypes.c_int64]
    workspace_bytes.restype = ctypes.c_size_t
    linear = library.nfn_candidate_k_mmq
    linear.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    linear.restype = ctypes.c_int
    return library, workspace_bytes, linear


def _native_mmq(library_path: Path):
    library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_LOCAL)
    abi = library.nfn_native_tile_k_quant_mmq_abi_version
    abi.argtypes = []
    abi.restype = ctypes.c_int
    if abi() != 1:
        raise RuntimeError("unsupported NeuralFn K-quant MMQ ABI")
    workspace_bytes = library.nfn_native_tile_k_quant_mmq_workspace_bytes_v1
    workspace_bytes.argtypes = [ctypes.c_int64, ctypes.c_int64]
    workspace_bytes.restype = ctypes.c_int64
    multi = library.nfn_native_tile_k_quant_mmq_multi_linear_float32_v1
    multi.argtypes = [
        ctypes.POINTER(ctypes.POINTER(PackedWeightDescriptor)),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    multi.restype = ctypes.c_int
    return library, workspace_bytes, multi


def _native_mmvq(library_path: Path):
    library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_LOCAL)
    workspace_bytes = library.nfn_native_tile_k_quant_mmq_workspace_bytes_v1
    workspace_bytes.argtypes = [ctypes.c_int64, ctypes.c_int64]
    workspace_bytes.restype = ctypes.c_int64
    linear = library.nfn_native_tile_k_quant_mmvq_linear_float32_v1
    linear.argtypes = [
        ctypes.POINTER(PackedWeightDescriptor),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    linear.restype = ctypes.c_int
    return library, workspace_bytes, linear


def _time(
    label: str,
    function,
    descriptor: PackedWeightDescriptor,
    device_input: ctypes.c_void_p,
    device_output: ctypes.c_void_p,
    rows: int,
    runtime: CudaRuntime,
    warmups: int,
    repetitions: int,
) -> Result:
    for _ in range(warmups):
        runtime.check(
            function(
                ctypes.byref(descriptor), device_input, None, device_output, rows, False
            ),
            f"{label} packed linear",
        )
    runtime.synchronize()
    samples: list[float] = []
    for _ in range(repetitions):
        samples.append(
            runtime.elapsed_ms(
                lambda: runtime.check(
                    function(
                        ctypes.byref(descriptor),
                        device_input,
                        None,
                        device_output,
                        rows,
                        False,
                    ),
                    f"{label} packed linear",
                )
            )
        )
    return Result(label, tuple(samples))


def _time_q8(
    label: str,
    quantize,
    linear,
    descriptor: PackedWeightDescriptor,
    device_input: ctypes.c_void_p,
    q8_values: ctypes.c_void_p,
    q8_scales: ctypes.c_void_p,
    q8_sums: ctypes.c_void_p,
    device_output: ctypes.c_void_p,
    rows: int,
    input_dim: int,
    runtime: CudaRuntime,
    warmups: int,
    repetitions: int,
    include_quantize: bool,
) -> Result:
    def launch() -> None:
        if include_quantize:
            runtime.check(
                quantize(
                    device_input,
                    q8_values,
                    q8_scales,
                    q8_sums,
                    rows,
                    input_dim,
                    None,
                ),
                f"{label} Q8 activation quantize",
            )
        runtime.check(
            linear(
                ctypes.byref(descriptor),
                q8_values,
                q8_scales,
                q8_sums,
                None,
                device_output,
                rows,
                False,
            ),
            f"{label} Q8 packed linear",
        )

    for _ in range(warmups):
        launch()
    runtime.synchronize()
    samples: list[float] = []
    for _ in range(repetitions):
        samples.append(runtime.elapsed_ms(launch))
    return Result(label, tuple(samples))


def _time_llama_provider(
    label: str,
    context,
    linear,
    descriptor: PackedWeightDescriptor,
    device_input: ctypes.c_void_p,
    device_output: ctypes.c_void_p,
    rows: int,
    runtime: CudaRuntime,
    warmups: int,
    repetitions: int,
) -> Result:
    def launch() -> None:
        runtime.check(
            linear(
                context,
                descriptor.encoding,
                descriptor.data,
                descriptor.data_nbytes,
                descriptor.row_stride_bytes,
                device_input,
                device_output,
                rows,
                descriptor.input_dim,
                descriptor.output_dim,
                descriptor.cuda_stream,
            ),
            f"{label} packed linear",
        )

    for _ in range(warmups):
        launch()
    runtime.synchronize()
    samples: list[float] = []
    for _ in range(repetitions):
        samples.append(runtime.elapsed_ms(launch))
    return Result(label, tuple(samples))


def _time_standalone_mmq(
    label: str,
    linear,
    descriptor: PackedWeightDescriptor,
    device_input: ctypes.c_void_p,
    device_output: ctypes.c_void_p,
    workspace: ctypes.c_void_p,
    workspace_bytes: int,
    rows: int,
    runtime: CudaRuntime,
    warmups: int,
    repetitions: int,
) -> Result:
    def launch() -> None:
        runtime.check(
            linear(
                descriptor.encoding,
                descriptor.data,
                descriptor.row_stride_bytes,
                device_input,
                device_output,
                workspace,
                workspace_bytes,
                rows,
                descriptor.input_dim,
                descriptor.output_dim,
                descriptor.cuda_stream,
            ),
            f"{label} packed linear",
        )

    for _ in range(warmups):
        launch()
    runtime.synchronize()
    return Result(
        label,
        tuple(runtime.elapsed_ms(launch) for _ in range(repetitions)),
    )


def _time_native_mmq(
    label: str,
    multi,
    descriptor: PackedWeightDescriptor,
    device_input: ctypes.c_void_p,
    device_output: ctypes.c_void_p,
    workspace: ctypes.c_void_p,
    workspace_bytes: int,
    rows: int,
    runtime: CudaRuntime,
    warmups: int,
    repetitions: int,
) -> Result:
    descriptor_pointer = ctypes.pointer(descriptor)
    descriptors = (ctypes.POINTER(PackedWeightDescriptor) * 1)(descriptor_pointer)
    outputs = (ctypes.c_void_p * 1)(device_output)

    def launch() -> None:
        runtime.check(
            multi(
                descriptors,
                device_input,
                outputs,
                1,
                rows,
                workspace,
                workspace_bytes,
                descriptor.cuda_stream,
            ),
            f"{label} packed linear",
        )

    for _ in range(warmups):
        launch()
    runtime.synchronize()
    return Result(label, tuple(runtime.elapsed_ms(launch) for _ in range(repetitions)))


def _time_native_mmvq(
    label: str,
    linear,
    descriptor: PackedWeightDescriptor,
    device_input: ctypes.c_void_p,
    device_output: ctypes.c_void_p,
    workspace: ctypes.c_void_p,
    workspace_bytes: int,
    runtime: CudaRuntime,
    warmups: int,
    repetitions: int,
) -> Result:
    def launch() -> None:
        runtime.check(
            linear(
                ctypes.byref(descriptor),
                device_input,
                device_output,
                workspace,
                workspace_bytes,
                descriptor.cuda_stream,
            ),
            f"{label} packed linear",
        )

    for _ in range(warmups):
        launch()
    runtime.synchronize()
    return Result(label, tuple(runtime.elapsed_ms(launch) for _ in range(repetitions)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cuda-runtime-lib", type=Path, required=True)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--encoding", choices=tuple(ENCODINGS), default="q4_k")
    parser.add_argument("--input-dim", type=int, default=6656)
    parser.add_argument("--output-dim", type=int, default=6656)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument(
        "--candidate-q8",
        action="store_true",
        help="Use the candidate's optional Q8-activation K-quant fast path.",
    )
    parser.add_argument(
        "--candidate-q8-prequantized",
        action="store_true",
        help=(
            "Time only the Q8 linear after one activation quantization, modeling "
            "reuse across projections with a shared input."
        ),
    )
    parser.add_argument(
        "--baseline-llama-provider",
        action="store_true",
        help="Treat --baseline as the pinned experimental llama MMQ provider.",
    )
    parser.add_argument(
        "--baseline-native-mmq",
        action="store_true",
        help=(
            "Treat --baseline as the versioned NeuralFn K-quant MMQ/MMVQ ABI. "
            "This enables exact multi-row kernel-to-kernel tuning."
        ),
    )
    parser.add_argument(
        "--candidate-llama-provider",
        action="store_true",
        help=(
            "Treat --candidate as the pinned experimental llama MMQ provider. "
            "This benchmark-only path consumes the original FP32 activations."
        ),
    )
    parser.add_argument(
        "--candidate-standalone-mmq",
        action="store_true",
        help="Treat --candidate as the direct workspace-based MMQ prototype.",
    )
    parser.add_argument(
        "--candidate-native-mmq",
        action="store_true",
        help="Treat --candidate as the versioned NeuralFn K-quant MMQ ABI.",
    )
    parser.add_argument(
        "--candidate-native-mmvq",
        action="store_true",
        help="Treat --candidate as the exact NeuralFn one-row MMVQ ABI.",
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument(
        "--max-abs-error",
        type=float,
        default=1.0e-3,
        help="Maximum permitted accumulation-order difference from the baseline.",
    )
    args = parser.parse_args()
    if args.input_dim <= 0 or args.output_dim <= 0 or args.rows <= 0:
        parser.error("dimensions and rows must be positive")
    if args.warmups < 0 or args.repetitions <= 0:
        parser.error("warmups must be non-negative and repetitions must be positive")
    if args.candidate_q8_prequantized and not args.candidate_q8:
        parser.error("--candidate-q8-prequantized requires --candidate-q8")
    if sum((args.candidate_llama_provider, args.candidate_standalone_mmq, args.candidate_native_mmq, args.candidate_native_mmvq, args.candidate_q8)) > 1:
        parser.error("candidate MMQ/Q8 modes are mutually exclusive")
    if args.candidate_native_mmvq and args.rows != 1:
        parser.error("the native MMVQ candidate requires exactly one row")
    if args.candidate_llama_provider and args.rows < 2:
        parser.error("the experimental llama MMQ provider requires at least two rows")
    if args.baseline_llama_provider and args.baseline_native_mmq:
        parser.error("baseline provider modes are mutually exclusive")

    encoding_id, _block_bytes = ENCODINGS[args.encoding]
    packed, row_stride = _packed_bytes(args.encoding, args.output_dim, args.input_dim)
    inputs = array(
        "f",
        (
            ((row * 37 + column * 13) % 257 - 128) / 257.0
            for row in range(args.rows)
            for column in range(args.input_dim)
        ),
    )
    packed_host = (ctypes.c_uint8 * len(packed)).from_buffer_copy(packed)
    input_host = (ctypes.c_float * len(inputs)).from_buffer_copy(inputs)
    output_count = args.rows * args.output_dim
    output_bytes = output_count * ctypes.sizeof(ctypes.c_float)
    runtime = CudaRuntime(args.cuda_runtime_lib.resolve(strict=True), args.cuda_device)
    device_packed = runtime.malloc(len(packed))
    device_input = runtime.malloc(len(inputs) * ctypes.sizeof(ctypes.c_float))
    device_output = runtime.malloc(output_bytes)
    q8_values = q8_scales = q8_sums = None
    if args.candidate_q8:
        q8_values = runtime.malloc(len(inputs))
        q8_block_count = args.rows * args.input_dim // 32
        q8_scales = runtime.malloc(q8_block_count * ctypes.sizeof(ctypes.c_float))
        q8_sums = runtime.malloc(q8_block_count * ctypes.sizeof(ctypes.c_float))
    baseline_library = candidate_library = None
    provider_context = provider_destroy = None
    baseline_provider_context = baseline_provider_destroy = None
    standalone_workspace = None
    standalone_workspace_bytes = 0
    baseline_workspace = None
    baseline_workspace_bytes = 0
    try:
        runtime.upload(device_packed, packed_host, len(packed))
        runtime.upload(
            device_input,
            input_host,
            len(inputs) * ctypes.sizeof(ctypes.c_float),
        )
        descriptor = PackedWeightDescriptor(
            struct_size=ctypes.sizeof(PackedWeightDescriptor),
            version=1,
            encoding=encoding_id,
            flags=0,
            data=device_packed.value,
            data_nbytes=len(packed),
            output_dim=args.output_dim,
            input_dim=args.input_dim,
            row_stride_bytes=row_stride,
            reserved0=0,
            reserved1=0,
            cuda_stream=None,
        )
        if args.baseline_llama_provider:
            (
                baseline_library,
                baseline_provider_context,
                baseline_provider_destroy,
                baseline,
            ) = _llama_mmq_provider(args.baseline.resolve(strict=True), args.cuda_device)
        elif args.baseline_native_mmq:
            baseline_library, baseline_workspace_size_fn, baseline = _native_mmq(
                args.baseline.resolve(strict=True)
            )
            baseline_workspace_bytes = baseline_workspace_size_fn(
                args.rows, args.input_dim
            )
            baseline_workspace = runtime.malloc(baseline_workspace_bytes)
        else:
            baseline_library, baseline = _linear(args.baseline.resolve(strict=True))
        if args.candidate_llama_provider:
            (
                candidate_library,
                provider_context,
                provider_destroy,
                candidate,
            ) = _llama_mmq_provider(
                args.candidate.resolve(strict=True), args.cuda_device
            )
            candidate_quantize = None
        elif args.candidate_standalone_mmq:
            candidate_library, workspace_size_fn, candidate = _standalone_mmq(
                args.candidate.resolve(strict=True)
            )
            standalone_workspace_bytes = workspace_size_fn(args.rows, args.input_dim)
            standalone_workspace = runtime.malloc(standalone_workspace_bytes)
            candidate_quantize = None
        elif args.candidate_native_mmq:
            candidate_library, workspace_size_fn, candidate = _native_mmq(
                args.candidate.resolve(strict=True)
            )
            standalone_workspace_bytes = workspace_size_fn(args.rows, args.input_dim)
            standalone_workspace = runtime.malloc(standalone_workspace_bytes)
            candidate_quantize = None
        elif args.candidate_native_mmvq:
            candidate_library, workspace_size_fn, candidate = _native_mmvq(
                args.candidate.resolve(strict=True)
            )
            standalone_workspace_bytes = workspace_size_fn(1, args.input_dim)
            standalone_workspace = runtime.malloc(standalone_workspace_bytes)
            candidate_quantize = None
        elif args.candidate_q8:
            candidate_library, candidate_quantize, candidate = _q8_linear(
                args.candidate.resolve(strict=True)
            )
        else:
            candidate_library, candidate = _linear(args.candidate.resolve(strict=True))
            candidate_quantize = None

        if args.baseline_llama_provider:
            runtime.check(
                baseline(
                    baseline_provider_context,
                    descriptor.encoding,
                    descriptor.data,
                    descriptor.data_nbytes,
                    descriptor.row_stride_bytes,
                    device_input,
                    device_output,
                    args.rows,
                    descriptor.input_dim,
                    descriptor.output_dim,
                    descriptor.cuda_stream,
                ),
                "baseline llama MMQ parity launch",
            )
        elif args.baseline_native_mmq:
            descriptor_pointer = ctypes.pointer(descriptor)
            descriptors = (ctypes.POINTER(PackedWeightDescriptor) * 1)(
                descriptor_pointer
            )
            outputs = (ctypes.c_void_p * 1)(device_output)
            runtime.check(
                baseline(
                    descriptors,
                    device_input,
                    outputs,
                    1,
                    args.rows,
                    baseline_workspace,
                    baseline_workspace_bytes,
                    descriptor.cuda_stream,
                ),
                "baseline native MMVQ parity launch",
            )
        else:
            runtime.check(
                baseline(
                    ctypes.byref(descriptor), device_input, None, device_output, args.rows, False
                ),
                "baseline parity launch",
            )
        runtime.synchronize()
        baseline_output = (ctypes.c_float * output_count)()
        runtime.download(baseline_output, device_output, output_bytes)
        if args.candidate_llama_provider:
            runtime.check(
                candidate(
                    provider_context,
                    descriptor.encoding,
                    descriptor.data,
                    descriptor.data_nbytes,
                    descriptor.row_stride_bytes,
                    device_input,
                    device_output,
                    args.rows,
                    args.input_dim,
                    args.output_dim,
                    descriptor.cuda_stream,
                ),
                "candidate llama MMQ parity launch",
            )
        elif args.candidate_standalone_mmq:
            runtime.check(
                candidate(
                    descriptor.encoding,
                    descriptor.data,
                    descriptor.row_stride_bytes,
                    device_input,
                    device_output,
                    standalone_workspace,
                    standalone_workspace_bytes,
                    args.rows,
                    args.input_dim,
                    args.output_dim,
                    descriptor.cuda_stream,
                ),
                "candidate standalone MMQ parity launch",
            )
        elif args.candidate_native_mmq:
            descriptor_pointer = ctypes.pointer(descriptor)
            descriptors = (ctypes.POINTER(PackedWeightDescriptor) * 1)(
                descriptor_pointer
            )
            outputs = (ctypes.c_void_p * 1)(device_output)
            runtime.check(
                candidate(
                    descriptors,
                    device_input,
                    outputs,
                    1,
                    args.rows,
                    standalone_workspace,
                    standalone_workspace_bytes,
                    descriptor.cuda_stream,
                ),
                "candidate native MMQ parity launch",
            )
        elif args.candidate_native_mmvq:
            runtime.check(
                candidate(
                    ctypes.byref(descriptor),
                    device_input,
                    device_output,
                    standalone_workspace,
                    standalone_workspace_bytes,
                    descriptor.cuda_stream,
                ),
                "candidate native MMVQ parity launch",
            )
        elif args.candidate_q8:
            runtime.check(
                candidate_quantize(
                    device_input,
                    q8_values,
                    q8_scales,
                    q8_sums,
                    args.rows,
                    args.input_dim,
                    None,
                ),
                "candidate Q8 parity quantize",
            )
            runtime.check(
                candidate(
                    ctypes.byref(descriptor),
                    q8_values,
                    q8_scales,
                    q8_sums,
                    None,
                    device_output,
                    args.rows,
                    False,
                ),
                "candidate Q8 parity launch",
            )
        else:
            runtime.check(
                candidate(
                    ctypes.byref(descriptor), device_input, None, device_output, args.rows, False
                ),
                "candidate parity launch",
            )
        runtime.synchronize()
        candidate_output = (ctypes.c_float * output_count)()
        runtime.download(candidate_output, device_output, output_bytes)
        max_abs_error = max(
            abs(float(left) - float(right))
            for left, right in zip(baseline_output, candidate_output, strict=True)
        )
        if max_abs_error > args.max_abs_error:
            raise RuntimeError(
                f"candidate output differs from baseline (max abs error {max_abs_error})"
            )

        if args.baseline_llama_provider:
            baseline_result = _time_llama_provider(
                "baseline-llama-mmq",
                baseline_provider_context,
                baseline,
                descriptor,
                device_input,
                device_output,
                args.rows,
                runtime,
                args.warmups,
                args.repetitions,
            )
        elif args.baseline_native_mmq:
            baseline_result = _time_native_mmq(
                "baseline-native-mmvq",
                baseline,
                descriptor,
                device_input,
                device_output,
                baseline_workspace,
                baseline_workspace_bytes,
                args.rows,
                runtime,
                args.warmups,
                args.repetitions,
            )
        else:
            baseline_result = _time(
                "baseline",
                baseline,
                descriptor,
                device_input,
                device_output,
                args.rows,
                runtime,
                args.warmups,
                args.repetitions,
            )
        if args.candidate_llama_provider:
            candidate_result = _time_llama_provider(
                "candidate-llama-mmq",
                provider_context,
                candidate,
                descriptor,
                device_input,
                device_output,
                args.rows,
                runtime,
                args.warmups,
                args.repetitions,
            )
        elif args.candidate_standalone_mmq:
            candidate_result = _time_standalone_mmq(
                "candidate-standalone-mmq",
                candidate,
                descriptor,
                device_input,
                device_output,
                standalone_workspace,
                standalone_workspace_bytes,
                args.rows,
                runtime,
                args.warmups,
                args.repetitions,
            )
        elif args.candidate_native_mmq:
            candidate_result = _time_native_mmq(
                "candidate-native-mmq",
                candidate,
                descriptor,
                device_input,
                device_output,
                standalone_workspace,
                standalone_workspace_bytes,
                args.rows,
                runtime,
                args.warmups,
                args.repetitions,
            )
        elif args.candidate_native_mmvq:
            candidate_result = _time_native_mmvq(
                "candidate-native-mmvq",
                candidate,
                descriptor,
                device_input,
                device_output,
                standalone_workspace,
                standalone_workspace_bytes,
                runtime,
                args.warmups,
                args.repetitions,
            )
        elif args.candidate_q8:
            candidate_result = _time_q8(
                "candidate-q8",
                candidate_quantize,
                candidate,
                descriptor,
                device_input,
                q8_values,
                q8_scales,
                q8_sums,
                device_output,
                args.rows,
                args.input_dim,
                runtime,
                args.warmups,
                args.repetitions,
                not args.candidate_q8_prequantized,
            )
        else:
            candidate_result = _time(
                "candidate",
                candidate,
                descriptor,
                device_input,
                device_output,
                args.rows,
                runtime,
                args.warmups,
                args.repetitions,
            )
        speedup = baseline_result.median_ms / candidate_result.median_ms
        print(
            f"{args.encoding} {args.rows}x{args.input_dim}->{args.output_dim}: "
            f"baseline={baseline_result.median_ms:.6f} ms, "
            f"candidate={candidate_result.median_ms:.6f} ms, "
            f"speedup={speedup:.6f}x, max_abs_error={max_abs_error:.3g}"
        )
    finally:
        # Retain CDLL references until every launch has synchronized.
        if provider_context is not None and provider_destroy is not None:
            runtime.synchronize()
            provider_destroy(provider_context)
        if baseline_provider_context is not None and baseline_provider_destroy is not None:
            runtime.synchronize()
            baseline_provider_destroy(baseline_provider_context)
        if standalone_workspace is not None:
            runtime.free(standalone_workspace)
        if baseline_workspace is not None:
            runtime.free(baseline_workspace)
        del baseline_library, candidate_library
        if q8_sums is not None:
            runtime.free(q8_sums)
        if q8_scales is not None:
            runtime.free(q8_scales)
        if q8_values is not None:
            runtime.free(q8_values)
        runtime.free(device_output)
        runtime.free(device_input)
        runtime.free(device_packed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
