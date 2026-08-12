"""Portable, dependency-free TurboQuant reference codec.

This module is a correctness oracle and artifact-format prototype, not the
resident CUDA implementation.  It follows the two paper algorithms:

* MSE: normalize, deterministically rotate, Lloyd-Max scalar quantize, store
  the vector norm, and inverse-rotate during reconstruction.
* QJL: spend one bit per key coordinate on the sign of a Gaussian projection
  of the MSE residual and use that residual only for key/query inner products.

Values always use the MSE reconstruction path.  The existing graph
``kv_quant_pack``/``kv_quant_unpack`` format is unrelated and unchanged.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import struct
from typing import Iterable, Sequence


TURBOQUANT_REFERENCE_VERSION = 1
TURBOQUANT_PROFILES = ("mse-3.5", "qjl-3.5")


class TurboQuantError(ValueError):
    """Raised when a vector or codec configuration violates the contract."""


def _float32(value: float) -> float:
    try:
        converted = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except (OverflowError, struct.error) as exc:
        raise TurboQuantError("vector norm is not representable as finite float32") from exc
    if not math.isfinite(converted):
        raise TurboQuantError("vector norm is not representable as finite float32")
    return converted


def _validate_vector(vector: Sequence[float], dimension: int, *, name: str) -> tuple[float, ...]:
    if len(vector) != dimension:
        raise TurboQuantError(f"{name} must contain exactly {dimension} coordinates")
    result: list[float] = []
    for index, value in enumerate(vector):
        try:
            coordinate = float(value)
        except (TypeError, ValueError) as exc:
            raise TurboQuantError(f"{name}[{index}] must be a finite number") from exc
        if not math.isfinite(coordinate):
            raise TurboQuantError(f"{name}[{index}] must be a finite number")
        result.append(coordinate)
    return tuple(result)


class _DeterministicGaussian:
    """SHA-256 counter stream with a stable Box-Muller transform."""

    def __init__(self, seed: int, domain: str) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self._key = f"neuralfn-turboquant-v1:{seed}:{domain}".encode("utf-8")
        self._counter = 0
        self._spare: float | None = None

    def _uniform(self) -> float:
        counter = self._counter.to_bytes(16, "little", signed=False)
        self._counter += 1
        digest = hashlib.sha256(self._key + counter).digest()
        integer = int.from_bytes(digest[:8], "little")
        return (integer + 0.5) / float(1 << 64)

    def gaussian(self) -> float:
        if self._spare is not None:
            value = self._spare
            self._spare = None
            return value
        first = self._uniform()
        second = self._uniform()
        radius = math.sqrt(-2.0 * math.log(first))
        angle = 2.0 * math.pi * second
        self._spare = radius * math.sin(angle)
        return radius * math.cos(angle)


@lru_cache(maxsize=64)
def deterministic_random_rotation(dimension: int, seed: int) -> tuple[tuple[float, ...], ...]:
    """Generate a deterministic Haar-style orthogonal matrix via Gaussian QR."""

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 2:
        raise TurboQuantError("dimension must be an integer greater than or equal to 2")
    stream = _DeterministicGaussian(seed, f"rotation:{dimension}")
    columns: list[list[float]] = []
    for column_index in range(dimension):
        vector = [stream.gaussian() for _ in range(dimension)]
        # Two modified Gram-Schmidt passes keep the small reference matrices
        # stable enough for golden tests without depending on BLAS/NumPy.
        for _pass in range(2):
            for basis in columns:
                projection = math.fsum(a * b for a, b in zip(vector, basis))
                for row in range(dimension):
                    vector[row] -= projection * basis[row]
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if norm <= 1.0e-14:
            # A Gaussian matrix is nonsingular almost surely. Keep an explicit
            # deterministic failure rather than silently changing the seed.
            raise TurboQuantError(
                f"deterministic rotation became singular at column {column_index}"
            )
        columns.append([value / norm for value in vector])
    return tuple(
        tuple(columns[column][row] for column in range(dimension))
        for row in range(dimension)
    )


@lru_cache(maxsize=64)
def deterministic_qjl_projection(dimension: int, seed: int) -> tuple[tuple[float, ...], ...]:
    """Return the paper's deterministic-seed Gaussian QJL matrix."""

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 2:
        raise TurboQuantError("dimension must be an integer greater than or equal to 2")
    stream = _DeterministicGaussian(seed, f"qjl:{dimension}")
    return tuple(
        tuple(stream.gaussian() for _ in range(dimension))
        for _ in range(dimension)
    )


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [math.fsum(value * coordinate for value, coordinate in zip(row, vector)) for row in matrix]


def _transpose_matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    dimension = len(matrix)
    return [
        math.fsum(matrix[row][column] * vector[row] for row in range(dimension))
        for column in range(dimension)
    ]


@lru_cache(maxsize=64)
def lloyd_max_centroids(
    dimension: int,
    bit_width: int,
    *,
    quadrature_points: int = 16_384,
    iterations: int = 80,
) -> tuple[float, ...]:
    """Numerically solve the sphere-coordinate Lloyd-Max codebook.

    Midpoint quadrature samples the density proportional to
    ``(1 - x*x)**((dimension - 3)/2)`` on ``[-1, 1]``.  The result is
    symmetrized so the serialized reference is deterministic.
    """

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 2:
        raise TurboQuantError("dimension must be an integer greater than or equal to 2")
    if isinstance(bit_width, bool) or not isinstance(bit_width, int) or not 1 <= bit_width <= 8:
        raise TurboQuantError("bit_width must be an integer in 1..8")
    if quadrature_points < 1024:
        raise TurboQuantError("quadrature_points must be at least 1024")
    levels = 1 << bit_width
    exponent = (dimension - 3.0) / 2.0
    points = [-1.0 + (index + 0.5) * (2.0 / quadrature_points) for index in range(quadrature_points)]
    log_weights = [exponent * math.log1p(-(point * point)) for point in points]
    maximum = max(log_weights)
    weights = [math.exp(value - maximum) for value in log_weights]

    total = math.fsum(weights)
    centroids: list[float] = []
    cumulative = 0.0
    target_index = 0
    targets = [(index + 0.5) * total / levels for index in range(levels)]
    for point, weight in zip(points, weights):
        cumulative += weight
        while target_index < levels and cumulative >= targets[target_index]:
            centroids.append(point)
            target_index += 1
    while len(centroids) < levels:
        centroids.append(points[-1])

    for _ in range(iterations):
        boundaries = [(centroids[index] + centroids[index + 1]) * 0.5 for index in range(levels - 1)]
        sums = [0.0] * levels
        masses = [0.0] * levels
        for point, weight in zip(points, weights):
            bucket = bisect_right(boundaries, point)
            sums[bucket] += point * weight
            masses[bucket] += weight
        updated = [
            sums[index] / masses[index] if masses[index] > 0.0 else centroids[index]
            for index in range(levels)
        ]
        delta = max(abs(left - right) for left, right in zip(updated, centroids))
        centroids = updated
        if delta < 1.0e-13:
            break

    for index in range(levels // 2):
        magnitude = (abs(centroids[index]) + abs(centroids[-1 - index])) * 0.5
        centroids[index] = -magnitude
        centroids[-1 - index] = magnitude
    return tuple(centroids)


def pack_mixed_bit_indices(indices: Sequence[int], bit_widths: Sequence[int]) -> bytes:
    """Pack consecutive little-endian fields without byte padding."""

    if len(indices) != len(bit_widths):
        raise TurboQuantError("indices and bit_widths must have the same length")
    accumulator = 0
    available = 0
    output = bytearray()
    for position, (index, width) in enumerate(zip(indices, bit_widths)):
        if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= 8:
            raise TurboQuantError(f"bit_widths[{position}] must be an integer in 1..8")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < (1 << width):
            raise TurboQuantError(
                f"indices[{position}] must fit its unsigned {width}-bit field"
            )
        accumulator |= index << available
        available += width
        while available >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            available -= 8
    if available:
        output.append(accumulator & 0xFF)
    return bytes(output)


def unpack_mixed_bit_indices(
    packed: bytes | bytearray | memoryview,
    bit_widths: Sequence[int],
) -> tuple[int, ...]:
    expected = (sum(int(width) for width in bit_widths) + 7) // 8
    raw = bytes(packed)
    if len(raw) != expected:
        raise TurboQuantError(f"packed index payload must contain exactly {expected} bytes")
    accumulator = 0
    available = 0
    offset = 0
    result: list[int] = []
    for position, width in enumerate(bit_widths):
        if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= 8:
            raise TurboQuantError(f"bit_widths[{position}] must be an integer in 1..8")
        while available < width:
            accumulator |= raw[offset] << available
            offset += 1
            available += 8
        result.append(accumulator & ((1 << width) - 1))
        accumulator >>= width
        available -= width
    if accumulator != 0:
        raise TurboQuantError("packed index payload has non-zero trailing padding bits")
    return tuple(result)


def _pack_signs(signs: Sequence[int]) -> bytes:
    indices = [1 if sign > 0 else 0 for sign in signs]
    return pack_mixed_bit_indices(indices, [1] * len(indices))


def _unpack_signs(packed: bytes, dimension: int) -> tuple[int, ...]:
    return tuple(1 if value else -1 for value in unpack_mixed_bit_indices(packed, [1] * dimension))


@dataclass(frozen=True, slots=True)
class TurboQuantEncodedVector:
    """One compressed cache row; codec tables are shared model metadata."""

    dimension: int
    profile: str
    norm: float
    packed_indices: bytes
    qjl_signs: bytes | None = None
    residual_norm: float | None = None

    @property
    def data_bytes(self) -> int:
        return 4 + len(self.packed_indices) + (
            4 + len(self.qjl_signs) if self.qjl_signs is not None else 0
        )

    @property
    def uncompressed_bytes(self) -> int:
        return self.dimension * 4

    @property
    def compression_ratio(self) -> float:
        return self.uncompressed_bytes / self.data_bytes


class TurboQuantReferenceCodec:
    """Paper-aligned portable oracle for one fixed head dimension/profile."""

    def __init__(
        self,
        dimension: int,
        *,
        profile: str = "mse-3.5",
        seed: int = 0,
        outlier_indices: Iterable[int] | None = None,
    ) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 2:
            raise TurboQuantError("dimension must be an integer greater than or equal to 2")
        if dimension % 2:
            raise TurboQuantError("3.5-bit profiles require an even dimension")
        normalized_profile = str(profile).strip().lower()
        if normalized_profile not in TURBOQUANT_PROFILES:
            raise TurboQuantError("profile must be mse-3.5 or qjl-3.5")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if outlier_indices is None:
            raise TurboQuantError(
                "3.5-bit profiles require a fixed model-level outlier_indices set"
            )
        outliers = tuple(sorted(set(outlier_indices)))
        if len(outliers) != dimension // 2:
            raise TurboQuantError(
                f"3.5-bit profiles require exactly {dimension // 2} outlier channels"
            )
        if any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < dimension for index in outliers):
            raise TurboQuantError("outlier_indices must be unique integers inside the head dimension")

        self.dimension = dimension
        self.profile = normalized_profile
        self.seed = seed
        self.outlier_indices = outliers
        outlier_set = frozenset(outliers)
        self.value_bit_widths = tuple(4 if index in outlier_set else 3 for index in range(dimension))
        self.key_bit_widths = tuple(
            width - 1 if normalized_profile == "qjl-3.5" else width
            for width in self.value_bit_widths
        )
        self.rotation = deterministic_random_rotation(dimension, seed)
        self.qjl_projection = (
            deterministic_qjl_projection(dimension, seed)
            if normalized_profile == "qjl-3.5"
            else None
        )

    def _codebooks(self, bit_widths: Sequence[int]) -> dict[int, tuple[float, ...]]:
        return {
            width: lloyd_max_centroids(self.dimension, width)
            for width in sorted(set(bit_widths))
        }

    def _encode_mse(
        self,
        vector: Sequence[float],
        bit_widths: Sequence[int],
    ) -> tuple[float, bytes, list[float]]:
        validated = _validate_vector(vector, self.dimension, name="vector")
        norm = math.hypot(*validated)
        stored_norm = _float32(norm)
        if norm == 0.0:
            unit = [0.0] * self.dimension
        else:
            unit = [value / norm for value in validated]
        rotated = _matvec(self.rotation, unit)
        codebooks = self._codebooks(bit_widths)
        indices: list[int] = []
        quantized_rotated: list[float] = []
        for value, width in zip(rotated, bit_widths):
            codebook = codebooks[width]
            index = min(range(len(codebook)), key=lambda candidate: abs(value - codebook[candidate]))
            indices.append(index)
            quantized_rotated.append(codebook[index])
        return stored_norm, pack_mixed_bit_indices(indices, bit_widths), quantized_rotated

    def _decode_mse_unit(
        self,
        encoded: TurboQuantEncodedVector,
        bit_widths: Sequence[int],
    ) -> list[float]:
        self._validate_encoded(encoded)
        indices = unpack_mixed_bit_indices(encoded.packed_indices, bit_widths)
        codebooks = self._codebooks(bit_widths)
        rotated = [codebooks[width][index] for index, width in zip(indices, bit_widths)]
        return _transpose_matvec(self.rotation, rotated)

    def encode_value(self, vector: Sequence[float]) -> TurboQuantEncodedVector:
        norm, packed, _quantized_rotated = self._encode_mse(vector, self.value_bit_widths)
        return TurboQuantEncodedVector(
            dimension=self.dimension,
            profile="mse-3.5",
            norm=norm,
            packed_indices=packed,
        )

    def encode_key(self, vector: Sequence[float]) -> TurboQuantEncodedVector:
        if self.profile == "mse-3.5":
            return self.encode_value(vector)
        validated = _validate_vector(vector, self.dimension, name="vector")
        norm, packed, quantized_rotated = self._encode_mse(validated, self.key_bit_widths)
        mse_unit = _transpose_matvec(self.rotation, quantized_rotated)
        source_norm = math.hypot(*validated)
        unit = [0.0] * self.dimension if source_norm == 0.0 else [value / source_norm for value in validated]
        residual = [value - approximation for value, approximation in zip(unit, mse_unit)]
        residual_norm = math.sqrt(math.fsum(value * value for value in residual))
        assert self.qjl_projection is not None
        projected = _matvec(self.qjl_projection, residual)
        signs = [1 if value >= 0.0 else -1 for value in projected]
        return TurboQuantEncodedVector(
            dimension=self.dimension,
            profile="qjl-3.5",
            norm=norm,
            packed_indices=packed,
            qjl_signs=_pack_signs(signs),
            residual_norm=_float32(residual_norm),
        )

    def decode_value(self, encoded: TurboQuantEncodedVector) -> tuple[float, ...]:
        unit = self._decode_mse_unit(encoded, self.value_bit_widths)
        return tuple(encoded.norm * value for value in unit)

    def decode_key_mse(self, encoded: TurboQuantEncodedVector) -> tuple[float, ...]:
        widths = self.key_bit_widths if encoded.profile == "qjl-3.5" else self.value_bit_widths
        unit = self._decode_mse_unit(encoded, widths)
        return tuple(encoded.norm * value for value in unit)

    def key_inner_product(
        self,
        query: Sequence[float],
        encoded: TurboQuantEncodedVector,
    ) -> float:
        """Estimate ``query dot key`` without reconstructing a cache matrix."""

        validated_query = _validate_vector(query, self.dimension, name="query")
        self._validate_encoded(encoded)
        widths = self.key_bit_widths if encoded.profile == "qjl-3.5" else self.value_bit_widths
        indices = unpack_mixed_bit_indices(encoded.packed_indices, widths)
        codebooks = self._codebooks(widths)
        rotated_query = _matvec(self.rotation, validated_query)
        base = math.fsum(
            query_value * codebooks[width][index]
            for query_value, index, width in zip(rotated_query, indices, widths)
        )
        estimate = encoded.norm * base
        if encoded.profile != "qjl-3.5":
            return estimate
        if encoded.qjl_signs is None or encoded.residual_norm is None or self.qjl_projection is None:
            raise TurboQuantError("qjl-3.5 key is missing residual signs or norm")
        signs = _unpack_signs(encoded.qjl_signs, self.dimension)
        projected_query = _matvec(self.qjl_projection, validated_query)
        correction = (
            math.sqrt(math.pi / 2.0)
            / self.dimension
            * encoded.residual_norm
            * math.fsum(value * sign for value, sign in zip(projected_query, signs))
        )
        return encoded.norm * (base + correction)

    def accumulate_value(
        self,
        output: list[float],
        weight: float,
        encoded: TurboQuantEncodedVector,
    ) -> None:
        """Accumulate one value row; QJL is intentionally never used here."""

        if len(output) != self.dimension:
            raise TurboQuantError(f"output must contain exactly {self.dimension} coordinates")
        scalar = float(weight)
        if not math.isfinite(scalar):
            raise TurboQuantError("weight must be finite")
        decoded = self.decode_value(encoded)
        for index, value in enumerate(decoded):
            output[index] += scalar * value

    def _validate_encoded(self, encoded: TurboQuantEncodedVector) -> None:
        if not isinstance(encoded, TurboQuantEncodedVector):
            raise TypeError("encoded must be a TurboQuantEncodedVector")
        if encoded.dimension != self.dimension:
            raise TurboQuantError("encoded vector dimension does not match this codec")
        if encoded.profile not in TURBOQUANT_PROFILES:
            raise TurboQuantError("encoded vector has an unknown profile")
        if encoded.profile == "qjl-3.5" and self.profile != "qjl-3.5":
            raise TurboQuantError("qjl-3.5 data requires a qjl-3.5 codec")


__all__ = [
    "TURBOQUANT_PROFILES",
    "TURBOQUANT_REFERENCE_VERSION",
    "TurboQuantEncodedVector",
    "TurboQuantError",
    "TurboQuantReferenceCodec",
    "deterministic_qjl_projection",
    "deterministic_random_rotation",
    "lloyd_max_centroids",
    "pack_mixed_bit_indices",
    "unpack_mixed_bit_indices",
]
