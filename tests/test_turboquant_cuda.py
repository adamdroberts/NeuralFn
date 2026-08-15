from __future__ import annotations

import ctypes
import importlib.util
import math
import os
from pathlib import Path
import struct
import subprocess
import sysconfig
from types import ModuleType
from typing import Iterable, Sequence

import pytest

from neuralfn.native_inference import _turboquant_binding_tables
from neuralfn.turboquant import (
    TurboQuantEncodedVector,
    TurboQuantReferenceCodec,
    lloyd_max_centroids,
)


ROOT = Path(__file__).resolve().parents[1]
LIVE_CUDA = os.environ.get("NFN_NATIVE_TURBOQUANT_CUDA_TEST", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


class TurboQuantAttentionDescriptorV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("profile", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("query", ctypes.c_void_p),
        ("key_records", ctypes.c_void_p),
        ("value_records", ctypes.c_void_p),
        ("current_key", ctypes.c_void_p),
        ("current_value", ctypes.c_void_p),
        ("output", ctypes.c_void_p),
        ("rotation", ctypes.c_void_p),
        ("qjl_projection", ctypes.c_void_p),
        ("centroids_2bit", ctypes.c_void_p),
        ("centroids_3bit", ctypes.c_void_p),
        ("centroids_4bit", ctypes.c_void_p),
        ("batch_size", ctypes.c_int64),
        ("layer_index", ctypes.c_int64),
        ("num_layers", ctypes.c_int64),
        ("query_heads", ctypes.c_int64),
        ("kv_heads", ctypes.c_int64),
        ("head_dim", ctypes.c_int64),
        ("past_sequence_length", ctypes.c_int64),
        ("cache_capacity", ctypes.c_int64),
        ("key_record_bytes", ctypes.c_int64),
        ("value_record_bytes", ctypes.c_int64),
        ("key_cache_batch_stride_bytes", ctypes.c_int64),
        ("value_cache_batch_stride_bytes", ctypes.c_int64),
        ("query_batch_stride", ctypes.c_int64),
        ("current_key_batch_stride", ctypes.c_int64),
        ("current_value_batch_stride", ctypes.c_int64),
        ("output_batch_stride", ctypes.c_int64),
        ("scale", ctypes.c_float),
        ("reserved0", ctypes.c_uint32),
        ("cuda_stream", ctypes.c_void_p),
    ]


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _vector(seed: int, dimension: int) -> list[float]:
    return [
        _f32(
            math.sin((seed + 1) * (index + 1) * 0.173)
            + 0.35 * math.cos((seed + 3) * (index + 2) * 0.119)
        )
        for index in range(dimension)
    ]


def _flatten(rows: Sequence[Sequence[float]]) -> list[float]:
    return [float(value) for row in rows for value in row]


def _key_record(encoded: TurboQuantEncodedVector) -> bytes:
    payload = bytearray(struct.pack("<f", encoded.norm))
    if encoded.profile == "qjl-3.5":
        assert encoded.residual_norm is not None
        assert encoded.qjl_signs is not None
        payload.extend(struct.pack("<f", encoded.residual_norm))
    payload.extend(encoded.packed_indices)
    if encoded.profile == "qjl-3.5":
        assert encoded.qjl_signs is not None
        payload.extend(encoded.qjl_signs)
    return bytes(payload)


def _value_record(encoded: TurboQuantEncodedVector) -> bytes:
    return struct.pack("<f", encoded.norm) + encoded.packed_indices


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = math.fsum(exponentials)
    return [value / denominator for value in exponentials]


class _CudaRuntime:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcudart.so.13")
        self.lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.cudaGetDeviceCount.restype = ctypes.c_int
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
        self.lib.cudaDeviceSynchronize.argtypes = []
        self.lib.cudaDeviceSynchronize.restype = ctypes.c_int
        count = ctypes.c_int()
        self._check(self.lib.cudaGetDeviceCount(ctypes.byref(count)), "cudaGetDeviceCount")
        if count.value <= 0:
            raise RuntimeError("TurboQuant CUDA acceptance requires a visible CUDA device")
        self.allocations: list[ctypes.c_void_p] = []

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"{operation} failed with CUDA status {status}")

    def allocate(self, nbytes: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        self._check(self.lib.cudaMalloc(ctypes.byref(pointer), max(1, nbytes)), "cudaMalloc")
        self.allocations.append(pointer)
        return pointer

    def upload_bytes(self, data: bytes) -> ctypes.c_void_p:
        raw = bytes(data) or b"\0"
        host = ctypes.create_string_buffer(raw, len(raw))
        device = self.allocate(len(raw))
        self._check(
            self.lib.cudaMemcpy(
                device,
                ctypes.cast(host, ctypes.c_void_p),
                len(raw),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            "cudaMemcpy H2D bytes",
        )
        return device

    def upload_floats(self, values: Iterable[float]) -> ctypes.c_void_p:
        converted = tuple(_f32(value) for value in values)
        host = (ctypes.c_float * len(converted))(*converted)
        device = self.allocate(ctypes.sizeof(host))
        self._check(
            self.lib.cudaMemcpy(
                device,
                ctypes.cast(host, ctypes.c_void_p),
                ctypes.sizeof(host),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            "cudaMemcpy H2D float32",
        )
        return device

    def upload_doubles(self, values: Iterable[float]) -> ctypes.c_void_p:
        converted = tuple(float(value) for value in values)
        host = (ctypes.c_double * len(converted))(*converted)
        device = self.allocate(ctypes.sizeof(host))
        self._check(
            self.lib.cudaMemcpy(
                device,
                ctypes.cast(host, ctypes.c_void_p),
                ctypes.sizeof(host),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            "cudaMemcpy H2D float64",
        )
        return device

    def download_floats(self, device: ctypes.c_void_p, count: int) -> tuple[float, ...]:
        host = (ctypes.c_float * count)()
        self._check(
            self.lib.cudaMemcpy(
                ctypes.cast(host, ctypes.c_void_p),
                device,
                ctypes.sizeof(host),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ),
            "cudaMemcpy D2H float32",
        )
        return tuple(float(value) for value in host)

    def synchronize(self) -> None:
        self._check(self.lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    def close(self) -> None:
        while self.allocations:
            pointer = self.allocations.pop()
            self._check(self.lib.cudaFree(pointer), "cudaFree")


def _configure_tile_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nfn_native_tile_turboquant_attention_abi_version.argtypes = []
    library.nfn_native_tile_turboquant_attention_abi_version.restype = ctypes.c_int
    library.nfn_native_tile_turboquant_attention_forward_v1.argtypes = [
        ctypes.POINTER(TurboQuantAttentionDescriptorV1)
    ]
    library.nfn_native_tile_turboquant_attention_forward_v1.restype = ctypes.c_int
    library.nfn_native_tile_turboquant_attention_stats_reset.argtypes = []
    library.nfn_native_tile_turboquant_attention_stats_reset.restype = None
    library.nfn_native_tile_turboquant_attention_launch_count.argtypes = []
    library.nfn_native_tile_turboquant_attention_launch_count.restype = ctypes.c_int64
    assert library.nfn_native_tile_turboquant_attention_abi_version() == 1
    return library


@pytest.fixture(scope="session")
def turboquant_tile_libraries(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[tuple[str, ctypes.CDLL], ...]:
    if not LIVE_CUDA:
        pytest.skip("set NFN_NATIVE_TURBOQUANT_CUDA_TEST=1 for fail-honest live CUDA coverage")
    output_dir = tmp_path_factory.mktemp("turboquant-tile")
    default_path = output_dir / "libnfn_native_train_tile_ops.so"
    strict_path = output_dir / "libnfn_native_train_tile_ops_strict.so"
    environment = os.environ.copy()
    environment["NFN_NATIVE_BUILD_STRICT_TILE_OPS"] = "1"
    subprocess.run(
        ["bash", str(ROOT / "tools" / "build_native_train_tile_ops.sh"), str(default_path)],
        cwd=ROOT,
        env=environment,
        check=True,
        text=True,
    )
    assert default_path.is_file()
    assert strict_path.is_file()
    return (
        ("default", _configure_tile_library(default_path)),
        ("strict", _configure_tile_library(strict_path)),
    )


@pytest.fixture(scope="session")
def resident_binding(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    if not LIVE_CUDA:
        pytest.skip("native CPU agreement is paired with the live CUDA gate")
    output = tmp_path_factory.mktemp("turboquant-resident-binding") / (
        "_native_inference" + (sysconfig.get_config_var("EXT_SUFFIX") or ".so")
    )
    subprocess.run(
        ["bash", str(ROOT / "tools" / "build_native_inference_binding.sh"), str(output)],
        cwd=ROOT,
        check=True,
        text=True,
    )
    spec = importlib.util.spec_from_file_location("_native_inference", output)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _descriptor_and_expected(
    cuda: _CudaRuntime,
    *,
    profile: str,
    dimension: int,
    query_heads: int,
    kv_heads: int,
    past: int,
    scale: float,
) -> tuple[TurboQuantAttentionDescriptorV1, list[float], list[bytes], list[bytes]]:
    codec = TurboQuantReferenceCodec(
        dimension,
        profile=profile,
        seed=0,
        outlier_indices=range(0, dimension, 2),
    )
    queries = [_vector(10 + head, dimension) for head in range(query_heads)]
    current_keys = [_vector(30 + head, dimension) for head in range(kv_heads)]
    current_values = [_vector(50 + head, dimension) for head in range(kv_heads)]
    encoded_keys: list[list[TurboQuantEncodedVector]] = []
    encoded_values: list[list[TurboQuantEncodedVector]] = []
    key_records: list[bytes] = []
    value_records: list[bytes] = []
    for position in range(past):
        position_keys: list[TurboQuantEncodedVector] = []
        position_values: list[TurboQuantEncodedVector] = []
        for head in range(kv_heads):
            key = codec.encode_key(_vector(100 + position * kv_heads + head, dimension))
            value = codec.encode_value(_vector(300 + position * kv_heads + head, dimension))
            position_keys.append(key)
            position_values.append(value)
            key_records.append(_key_record(key))
            value_records.append(_value_record(value))
        encoded_keys.append(position_keys)
        encoded_values.append(position_values)

    expected: list[float] = []
    normalized_scale = _f32(scale)
    heads_per_kv = query_heads // kv_heads
    for query_head, query in enumerate(queries):
        kv_head = query_head // heads_per_kv
        scores = [
            codec.key_inner_product(query, encoded_keys[position][kv_head]) * normalized_scale
            for position in range(past)
        ]
        scores.append(
            math.fsum(a * b for a, b in zip(query, current_keys[kv_head]))
            * normalized_scale
        )
        weights = _softmax(scores)
        output = [0.0] * dimension
        for position in range(past):
            codec.accumulate_value(
                output,
                weights[position],
                encoded_values[position][kv_head],
            )
        for coordinate in range(dimension):
            output[coordinate] += weights[-1] * current_values[kv_head][coordinate]
        expected.extend(output)

    key_record_bytes = len(key_records[0]) if key_records else (
        8 + ((dimension // 2 * 5 + 7) // 8) + ((dimension + 7) // 8)
        if profile == "qjl-3.5"
        else 4 + ((dimension // 2 * 7 + 7) // 8)
    )
    value_record_bytes = len(value_records[0]) if value_records else 4 + (
        (dimension // 2 * 7 + 7) // 8
    )
    cache_capacity = max(1, past)
    # Empty-history calls still pass one unused, correctly sized record because
    # v1 deliberately rejects null cache pointers.
    key_bytes = b"".join(key_records) or bytes(key_record_bytes)
    value_bytes = b"".join(value_records) or bytes(value_record_bytes)
    rotation = _flatten(codec.rotation)
    projection = _flatten(codec.qjl_projection or ())
    output_device = cuda.allocate(query_heads * dimension * ctypes.sizeof(ctypes.c_float))
    descriptor = TurboQuantAttentionDescriptorV1(
        struct_size=ctypes.sizeof(TurboQuantAttentionDescriptorV1),
        version=1,
        profile=2 if profile == "qjl-3.5" else 1,
        flags=0,
        query=cuda.upload_floats(_flatten(queries)),
        key_records=cuda.upload_bytes(key_bytes),
        value_records=cuda.upload_bytes(value_bytes),
        current_key=cuda.upload_floats(_flatten(current_keys)),
        current_value=cuda.upload_floats(_flatten(current_values)),
        output=output_device,
        rotation=cuda.upload_doubles(rotation),
        qjl_projection=cuda.upload_doubles(projection) if projection else None,
        centroids_2bit=cuda.upload_doubles(lloyd_max_centroids(dimension, 2)),
        centroids_3bit=cuda.upload_doubles(lloyd_max_centroids(dimension, 3)),
        centroids_4bit=cuda.upload_doubles(lloyd_max_centroids(dimension, 4)),
        batch_size=1,
        layer_index=0,
        num_layers=1,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=dimension,
        past_sequence_length=past,
        cache_capacity=cache_capacity,
        key_record_bytes=key_record_bytes,
        value_record_bytes=value_record_bytes,
        key_cache_batch_stride_bytes=0,
        value_cache_batch_stride_bytes=0,
        query_batch_stride=0,
        current_key_batch_stride=0,
        current_value_batch_stride=0,
        output_batch_stride=0,
        scale=normalized_scale,
        reserved0=0,
        cuda_stream=None,
    )
    return descriptor, expected, key_records, value_records


def _run_descriptor(
    library: ctypes.CDLL,
    descriptor: TurboQuantAttentionDescriptorV1,
    cuda: _CudaRuntime,
) -> tuple[float, ...]:
    status = library.nfn_native_tile_turboquant_attention_forward_v1(
        ctypes.byref(descriptor)
    )
    assert status == 0
    cuda.synchronize()
    return cuda.download_floats(
        ctypes.c_void_p(descriptor.output),
        descriptor.batch_size * descriptor.query_heads * descriptor.head_dim,
    )


def test_turboquant_tile_feature_abi_is_additive_and_direct() -> None:
    header = (ROOT / "neuralfn/csrc/native_train/tile_ops.h").read_text(encoding="utf-8")
    tile_ops = (ROOT / "neuralfn/csrc/native_train/tile_ops.cu").read_text(encoding="utf-8")
    kernels = (ROOT / "neuralfn/csrc/tile_cuda/kernels.cu").read_text(encoding="utf-8")
    assert "int nfn_native_tile_ops_abi_version()" in header
    assert "NfnNativeTileTurboQuantAttentionDescriptorV1" in header
    assert "nfn_native_tile_turboquant_attention_abi_version" in header
    assert "nfn_native_tile_turboquant_attention_forward_v1" in header
    assert "nfn_native_tile_turboquant_attention_launch_count" in header
    assert "source->struct_size < sizeof" in tile_ops
    assert "source->past_sequence_length >= kTurboQuantAttentionMaxSequenceLength" in tile_ops
    assert "turboquant_attention_forward_v1_kernel" in kernels
    assert "turboquant_historical_key_score" in kernels
    assert "turboquant_historical_value_coordinate" in kernels
    assert "for (std::int64_t chunk_start = 0; chunk_start < total_rows;" in kernels
    assert "dequantized_cache" not in kernels


@pytest.mark.parametrize("profile", ["mse-3.5", "qjl-3.5"])
@pytest.mark.parametrize(
    ("dimension", "query_heads", "kv_heads", "past"),
    [(8, 2, 1, 3), (64, 2, 2, 2), (128, 4, 2, 1)],
)
def test_cuda_attention_matches_portable_and_native_cpu_oracles(
    turboquant_tile_libraries: tuple[tuple[str, ctypes.CDLL], ...],
    resident_binding: ModuleType,
    profile: str,
    dimension: int,
    query_heads: int,
    kv_heads: int,
    past: int,
) -> None:
    cuda = _CudaRuntime()
    try:
        descriptor, expected, key_records, value_records = _descriptor_and_expected(
            cuda,
            profile=profile,
            dimension=dimension,
            query_heads=query_heads,
            kv_heads=kv_heads,
            past=past,
            scale=1.0 / math.sqrt(dimension),
        )
        # The first packed row is independently produced and consumed by the
        # existing native CPU codec probe, closing portable -> CPU -> CUDA.
        key = _vector(100, dimension)
        value = _vector(300, dimension)
        query = _vector(10, dimension)
        native = resident_binding.turboquant_codec_probe(
            {
                "turboquant_profile": profile,
                "tables": _turboquant_binding_tables(
                    channels=dimension,
                    num_heads=1,
                    profile=profile,
                ),
            },
            key,
            value,
            query,
        )
        cpu_oracle = TurboQuantReferenceCodec(
            dimension,
            profile=profile,
            seed=0,
            outlier_indices=range(0, dimension, 2),
        )
        portable_key = cpu_oracle.encode_key(key)
        portable_value = cpu_oracle.encode_value(value)
        assert native["key_inner_product"] == pytest.approx(
            cpu_oracle.key_inner_product(query, portable_key), abs=3.0e-6
        )
        assert native["decoded_value"] == pytest.approx(
            cpu_oracle.decode_value(portable_value), abs=3.0e-6
        )
        native_key = bytearray(struct.pack("<f", native["key_norm"]))
        if profile == "qjl-3.5":
            native_key.extend(struct.pack("<f", native["residual_norm"]))
        native_key.extend(native["key_indices"])
        if profile == "qjl-3.5":
            native_key.extend(native["qjl_signs"])
        native_value = struct.pack("<f", native["value_norm"]) + native["value_indices"]
        scalar_bytes = 8 if profile == "qjl-3.5" else 4
        assert bytes(native_key)[scalar_bytes:] == key_records[0][scalar_bytes:]
        assert struct.unpack_from("<f", native_key, 0)[0] == pytest.approx(
            struct.unpack_from("<f", key_records[0], 0)[0], abs=1.0e-7
        )
        if profile == "qjl-3.5":
            assert struct.unpack_from("<f", native_key, 4)[0] == pytest.approx(
                struct.unpack_from("<f", key_records[0], 4)[0], abs=2.0e-7
            )
        assert native_value[4:] == value_records[0][4:]
        assert struct.unpack_from("<f", native_value, 0)[0] == pytest.approx(
            struct.unpack_from("<f", value_records[0], 0)[0], abs=1.0e-7
        )

        for build_name, library in turboquant_tile_libraries:
            library.nfn_native_tile_turboquant_attention_stats_reset()
            actual = _run_descriptor(library, descriptor, cuda)
            repeated = _run_descriptor(library, descriptor, cuda)
            assert struct.pack(f"<{len(actual)}f", *actual) == struct.pack(
                f"<{len(repeated)}f", *repeated
            )
            assert library.nfn_native_tile_turboquant_attention_launch_count() == 2
            tolerance = 3.0e-5 if build_name == "default" else 1.5e-5
            assert actual == pytest.approx(expected, abs=tolerance, rel=tolerance)
    finally:
        cuda.close()


def test_cuda_attention_includes_exact_current_row_with_no_history(
    turboquant_tile_libraries: tuple[tuple[str, ctypes.CDLL], ...],
) -> None:
    cuda = _CudaRuntime()
    try:
        descriptor, expected, _keys, _values = _descriptor_and_expected(
            cuda,
            profile="mse-3.5",
            dimension=8,
            query_heads=1,
            kv_heads=1,
            past=0,
            scale=0.5,
        )
        for _build_name, library in turboquant_tile_libraries:
            assert _run_descriptor(library, descriptor, cuda) == pytest.approx(
                expected, abs=1.0e-6, rel=1.0e-6
            )
    finally:
        cuda.close()


@pytest.mark.parametrize("profile", ["mse-3.5", "qjl-3.5"])
@pytest.mark.parametrize("past", [1023, 1024, 1025, 4096, 16383])
def test_cuda_attention_chunked_softmax_matches_long_context_oracle(
    turboquant_tile_libraries: tuple[tuple[str, ctypes.CDLL], ...],
    profile: str,
    past: int,
) -> None:
    dimension = 8
    scale = _f32(1.0 / math.sqrt(dimension))
    codec = TurboQuantReferenceCodec(
        dimension,
        profile=profile,
        seed=0,
        outlier_indices=range(0, dimension, 2),
    )
    query = _vector(7, dimension)
    historical_key = codec.encode_key(_vector(11, dimension))
    historical_value = codec.encode_value(_vector(13, dimension))
    current_key = _vector(17, dimension)
    current_value = _vector(19, dimension)
    historical_score = codec.key_inner_product(query, historical_key) * scale
    current_score = math.fsum(a * b for a, b in zip(query, current_key)) * scale
    maximum = max(historical_score, current_score)
    historical_mass = past * math.exp(historical_score - maximum)
    current_mass = math.exp(current_score - maximum)
    denominator = historical_mass + current_mass
    historical_weight = historical_mass / denominator
    current_weight = current_mass / denominator
    decoded = codec.decode_value(historical_value)
    expected = [
        historical_weight * decoded[index] + current_weight * current_value[index]
        for index in range(dimension)
    ]
    key_record = _key_record(historical_key)
    value_record = _value_record(historical_value)

    cuda = _CudaRuntime()
    try:
        output_device = cuda.allocate(dimension * ctypes.sizeof(ctypes.c_float))
        descriptor = TurboQuantAttentionDescriptorV1(
            struct_size=ctypes.sizeof(TurboQuantAttentionDescriptorV1),
            version=1,
            profile=2 if profile == "qjl-3.5" else 1,
            flags=0,
            query=cuda.upload_floats(query),
            key_records=cuda.upload_bytes(key_record * past),
            value_records=cuda.upload_bytes(value_record * past),
            current_key=cuda.upload_floats(current_key),
            current_value=cuda.upload_floats(current_value),
            output=output_device,
            rotation=cuda.upload_doubles(_flatten(codec.rotation)),
            qjl_projection=(
                cuda.upload_doubles(_flatten(codec.qjl_projection or ()))
                if codec.qjl_projection is not None
                else None
            ),
            centroids_2bit=cuda.upload_doubles(lloyd_max_centroids(dimension, 2)),
            centroids_3bit=cuda.upload_doubles(lloyd_max_centroids(dimension, 3)),
            centroids_4bit=cuda.upload_doubles(lloyd_max_centroids(dimension, 4)),
            batch_size=1,
            layer_index=0,
            num_layers=1,
            query_heads=1,
            kv_heads=1,
            head_dim=dimension,
            past_sequence_length=past,
            cache_capacity=past,
            key_record_bytes=len(key_record),
            value_record_bytes=len(value_record),
            key_cache_batch_stride_bytes=0,
            value_cache_batch_stride_bytes=0,
            query_batch_stride=0,
            current_key_batch_stride=0,
            current_value_batch_stride=0,
            output_batch_stride=0,
            scale=scale,
            reserved0=0,
            cuda_stream=None,
        )
        for build_name, library in turboquant_tile_libraries:
            actual = _run_descriptor(library, descriptor, cuda)
            tolerance = 3.0e-5 if build_name == "default" else 1.5e-5
            assert actual == pytest.approx(expected, abs=tolerance, rel=tolerance)
    finally:
        cuda.close()


def test_feature_abi_rejects_invalid_descriptors_without_launch(
    turboquant_tile_libraries: tuple[tuple[str, ctypes.CDLL], ...],
) -> None:
    for _name, library in turboquant_tile_libraries:
        library.nfn_native_tile_turboquant_attention_stats_reset()
        descriptor = TurboQuantAttentionDescriptorV1()
        descriptor.struct_size = ctypes.sizeof(TurboQuantAttentionDescriptorV1) - 1
        descriptor.version = 1
        assert library.nfn_native_tile_turboquant_attention_forward_v1(
            ctypes.byref(descriptor)
        ) != 0
        assert library.nfn_native_tile_turboquant_attention_launch_count() == 0
