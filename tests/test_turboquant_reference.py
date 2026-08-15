from __future__ import annotations

import math
import subprocess
import sys

import pytest

from neuralfn.turboquant import (
    TurboQuantError,
    TurboQuantReferenceCodec,
    deterministic_random_rotation,
    lloyd_max_centroids,
    pack_mixed_bit_indices,
    unpack_mixed_bit_indices,
)


def _dot(left, right) -> float:
    return math.fsum(a * b for a, b in zip(left, right))


def test_deterministic_rotation_is_orthogonal_and_seeded() -> None:
    first = deterministic_random_rotation(8, 17)
    repeated = deterministic_random_rotation(8, 17)
    different = deterministic_random_rotation(8, 18)
    assert first == repeated
    assert first != different
    for row_index, row in enumerate(first):
        for other_index, other in enumerate(first):
            expected = 1.0 if row_index == other_index else 0.0
            assert _dot(row, other) == pytest.approx(expected, abs=2.0e-12)


def test_lloyd_max_codebooks_are_symmetric_and_match_known_one_bit_limit() -> None:
    dimension = 64
    centroids = lloyd_max_centroids(dimension, 1)
    assert centroids[0] == pytest.approx(-centroids[1], abs=1.0e-15)
    expected = math.sqrt(2.0 / math.pi) / math.sqrt(dimension)
    assert centroids[1] == pytest.approx(expected, rel=0.04)

    four_bit = lloyd_max_centroids(dimension, 4)
    assert tuple(sorted(four_bit)) == four_bit
    for left, right in zip(four_bit, reversed(four_bit)):
        assert left == pytest.approx(-right, abs=1.0e-15)


def test_three_bit_fields_cross_byte_boundaries_without_padding() -> None:
    packed = pack_mixed_bit_indices([0, 7, 5, 2], [3, 3, 3, 3])
    assert packed == bytes.fromhex("7805")
    assert unpack_mixed_bit_indices(packed, [3, 3, 3, 3]) == (0, 7, 5, 2)


def test_mixed_width_packing_round_trips_and_rejects_nonzero_padding() -> None:
    widths = [3, 4, 3, 4, 3, 4]
    indices = [7, 15, 5, 9, 3, 12]
    packed = pack_mixed_bit_indices(indices, widths)
    assert len(packed) == math.ceil(sum(widths) / 8)
    assert unpack_mixed_bit_indices(packed, widths) == tuple(indices)
    with pytest.raises(TurboQuantError, match="non-zero trailing padding"):
        unpack_mixed_bit_indices(packed[:-1] + bytes([packed[-1] | 0x80]), widths)


def test_mse_35_stores_norm_and_has_exact_payload_accounting() -> None:
    codec = TurboQuantReferenceCodec(
        8,
        profile="mse-3.5",
        seed=123,
        outlier_indices=(0, 2, 4, 6),
    )
    vector = (0.25, -0.5, 1.0, 0.75, -0.125, 0.4, -0.9, 0.2)
    encoded = codec.encode_value(vector)
    repeated = codec.encode_value(vector)
    decoded = codec.decode_value(encoded)

    assert encoded == repeated
    assert encoded.norm == pytest.approx(math.sqrt(_dot(vector, vector)), rel=1.0e-7)
    assert len(encoded.packed_indices) == 4  # ceil((4*4 + 4*3) / 8)
    assert encoded.data_bytes == 8  # fp32 norm + packed indices
    assert encoded.uncompressed_bytes == 32
    assert encoded.compression_ratio == 4.0
    mse = _dot(
        [source - approximation for source, approximation in zip(vector, decoded)],
        [source - approximation for source, approximation in zip(vector, decoded)],
    )
    assert mse < _dot(vector, vector) * 0.08


def test_qjl_35_key_payload_and_direct_inner_product_are_deterministic() -> None:
    codec = TurboQuantReferenceCodec(
        8,
        profile="qjl-3.5",
        seed=77,
        outlier_indices=(0, 1, 4, 7),
    )
    key = (0.5, -0.2, 0.8, -0.9, 0.1, 0.6, -0.4, 0.3)
    query = (-0.1, 0.7, 0.2, -0.3, 0.9, 0.1, -0.5, 0.4)
    encoded = codec.encode_key(key)

    assert encoded == codec.encode_key(key)
    assert len(encoded.packed_indices) == 3  # ceil((4*3 + 4*2) / 8)
    assert len(encoded.qjl_signs or b"") == 1
    assert encoded.data_bytes == 12  # norm + indices + residual norm + QJL signs
    assert math.isfinite(codec.key_inner_product(query, encoded))
    assert codec.key_inner_product(query, encoded) == codec.key_inner_product(query, encoded)


def test_qjl_residual_is_empirically_unbiased_over_deterministic_seed_ensemble() -> None:
    dimension = 16
    outliers = tuple(range(0, dimension, 2))
    key = tuple(math.sin(index + 0.25) for index in range(dimension))
    query = tuple(math.cos(index * 0.7 + 0.1) for index in range(dimension))
    exact = _dot(query, key)
    estimates = []
    for seed in range(96):
        codec = TurboQuantReferenceCodec(
            dimension,
            profile="qjl-3.5",
            seed=seed,
            outlier_indices=outliers,
        )
        estimates.append(codec.key_inner_product(query, codec.encode_key(key)))
    mean = math.fsum(estimates) / len(estimates)
    assert mean == pytest.approx(exact, abs=0.16)


def test_values_use_mse_only_even_with_qjl_profile() -> None:
    codec = TurboQuantReferenceCodec(
        8,
        profile="qjl-3.5",
        seed=11,
        outlier_indices=(1, 3, 5, 7),
    )
    value = tuple(index / 10.0 - 0.3 for index in range(8))
    encoded = codec.encode_value(value)
    assert encoded.profile == "mse-3.5"
    assert encoded.qjl_signs is None
    assert encoded.residual_norm is None

    output = [0.0] * 8
    codec.accumulate_value(output, 0.25, encoded)
    assert output == pytest.approx([coordinate * 0.25 for coordinate in codec.decode_value(encoded)])


def test_35_profile_requires_fixed_half_outlier_channels() -> None:
    with pytest.raises(TurboQuantError, match="fixed model-level"):
        TurboQuantReferenceCodec(8)
    with pytest.raises(TurboQuantError, match="exactly 4"):
        TurboQuantReferenceCodec(8, outlier_indices=(0, 1))
    with pytest.raises(TurboQuantError, match="even dimension"):
        TurboQuantReferenceCodec(7, outlier_indices=(0, 1, 2))


def test_nonrepresentable_float32_norm_fails_closed() -> None:
    codec = TurboQuantReferenceCodec(
        8,
        outlier_indices=(0, 2, 4, 6),
    )
    with pytest.raises(TurboQuantError, match="finite float32"):
        codec.encode_value((1.0e39,) * 8)


def test_public_reference_types_are_lazy_root_exports() -> None:
    code = (
        "import sys; import neuralfn; "
        "assert neuralfn.TurboQuantReferenceCodec.__module__ == 'neuralfn.turboquant'; "
        "print(','.join(name for name in ('torch','numpy','networkx') if name in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == ""


def test_reference_module_import_is_dependency_light() -> None:
    code = (
        "import sys; import neuralfn.turboquant; "
        "print(','.join(name for name in ('torch','numpy','networkx') if name in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == ""
