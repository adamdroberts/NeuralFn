from __future__ import annotations

import torch

from neuralfn.tile_cuda import (
    NVFP4_BLOCK_SIZE,
    dequantize_fp8_reference,
    dequantize_nvfp4_reference,
    quantize_dequantize_fp8_reference,
    quantize_dequantize_nvfp4_reference,
    quantize_fp8_reference,
    quantize_nvfp4_reference,
)


def test_fp8_reference_quantize_dequantize_is_deterministic() -> None:
    x = torch.linspace(-4.0, 4.0, 65, dtype=torch.float32)

    for fmt, dtype in (("float8_e4m3fn", torch.float8_e4m3fn), ("float8_e5m2", torch.float8_e5m2)):
        q0 = quantize_fp8_reference(x, fmt)
        q1 = quantize_fp8_reference(x, fmt)
        assert q0.dtype == dtype
        torch.testing.assert_close(dequantize_fp8_reference(q0), dequantize_fp8_reference(q1))
        torch.testing.assert_close(quantize_dequantize_fp8_reference(x, fmt), dequantize_fp8_reference(q0))


def test_fp8_reference_records_boundary_overflow_semantics() -> None:
    for fmt, dtype in (("float8_e4m3fn", torch.float8_e4m3fn), ("float8_e5m2", torch.float8_e5m2)):
        max_value = torch.finfo(dtype).max
        x = torch.tensor([-2.0 * max_value, -max_value, 0.0, max_value, 2.0 * max_value], dtype=torch.float32)
        out = dequantize_fp8_reference(quantize_fp8_reference(x, fmt))

        torch.testing.assert_close(out[1:4], torch.tensor([-max_value, 0.0, max_value], dtype=torch.float32))
        if dtype == torch.float8_e4m3fn:
            assert torch.isnan(out[0])
            assert torch.isnan(out[-1])
        else:
            assert torch.isneginf(out[0])
            assert torch.isposinf(out[-1])


def test_nvfp4_reference_packs_two_values_per_byte_and_roundtrips_shape() -> None:
    x = torch.linspace(-3.0, 3.0, 2 * 3 * 17, dtype=torch.float32).reshape(2, 3, 17)
    encoded = quantize_nvfp4_reference(x)
    out = dequantize_nvfp4_reference(encoded)

    padded = ((x.numel() + NVFP4_BLOCK_SIZE - 1) // NVFP4_BLOCK_SIZE) * NVFP4_BLOCK_SIZE
    assert encoded.packed.dtype == torch.uint8
    assert encoded.packed.numel() == padded // 2
    assert encoded.block_scales.dtype == torch.float8_e4m3fn
    assert encoded.block_scales.numel() == padded // NVFP4_BLOCK_SIZE
    assert encoded.tensor_scale.dtype == torch.float32
    assert encoded.shape == tuple(x.shape)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    torch.testing.assert_close(quantize_dequantize_nvfp4_reference(x), out)


def test_nvfp4_reference_is_deterministic_and_handles_empty_tensors() -> None:
    x = torch.tensor([-7.0, -2.1, -0.2, 0.0, 0.2, 2.1, 7.0], dtype=torch.float32)
    first = quantize_nvfp4_reference(x)
    second = quantize_nvfp4_reference(x)

    torch.testing.assert_close(first.packed, second.packed)
    torch.testing.assert_close(first.block_scales.to(torch.float32), second.block_scales.to(torch.float32))
    torch.testing.assert_close(first.tensor_scale, second.tensor_scale)
    torch.testing.assert_close(dequantize_nvfp4_reference(first), dequantize_nvfp4_reference(second))

    empty = quantize_nvfp4_reference(torch.empty(0, 3))
    assert dequantize_nvfp4_reference(empty).shape == (0, 3)

    empty_source = torch.empty(0, 3, requires_grad=True)
    empty_encoded = quantize_nvfp4_reference(empty_source, preserve_grad=True)
    assert dequantize_nvfp4_reference(empty_encoded).requires_grad


def test_nvfp4_reference_boundary_values_remain_finite_and_scaled() -> None:
    x = torch.tensor([-1.0e9, -448.0, -6.0, -0.5, 0.0, 0.5, 6.0, 448.0, 1.0e9], dtype=torch.float32)
    out = dequantize_nvfp4_reference(quantize_nvfp4_reference(x))

    assert torch.isfinite(out).all()
    assert out[0] < 0
    assert out[-1] > 0
    assert out.abs().max() <= x.abs().max()


def test_nvfp4_reference_can_preserve_source_gradients_with_ste() -> None:
    x = torch.linspace(-2.0, 2.0, 2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    x.requires_grad_(True)
    encoded = quantize_nvfp4_reference(x, preserve_grad=True)

    out = dequantize_nvfp4_reference(encoded)
    out.square().mean().backward()

    assert encoded.source is x
    assert x.grad is not None
    expected_grad = 2.0 * out.detach() / out.numel()
    torch.testing.assert_close(x.grad, expected_grad, rtol=1e-6, atol=1e-6)
