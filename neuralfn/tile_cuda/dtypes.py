from __future__ import annotations

from dataclasses import dataclass

import torch


FP8_FORMATS: dict[str, torch.dtype] = {
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}

NVFP4_BLOCK_SIZE = 16
NVFP4_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class NVFP4Tensor:
    packed: torch.Tensor
    block_scales: torch.Tensor
    tensor_scale: torch.Tensor
    shape: tuple[int, ...]
    block_size: int = NVFP4_BLOCK_SIZE
    source: torch.Tensor | None = None


def quantize_fp8_reference(x: torch.Tensor, fmt: str) -> torch.Tensor:
    if fmt not in FP8_FORMATS:
        raise ValueError(f"Unsupported fp8 format '{fmt}'")
    return x.detach().to(dtype=torch.float32).to(dtype=FP8_FORMATS[fmt])


def dequantize_fp8_reference(x: torch.Tensor) -> torch.Tensor:
    if x.dtype not in set(FP8_FORMATS.values()):
        raise TypeError(f"Expected fp8 tensor, got {x.dtype}")
    return x.to(dtype=torch.float32)


def quantize_dequantize_fp8_reference(x: torch.Tensor, fmt: str) -> torch.Tensor:
    return dequantize_fp8_reference(quantize_fp8_reference(x, fmt))


def _nvfp4_codebook(device: torch.device) -> torch.Tensor:
    return NVFP4_E2M1_VALUES.to(device=device)


def _pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    if codes.dtype != torch.uint8:
        codes = codes.to(dtype=torch.uint8)
    if codes.numel() % 2 != 0:
        codes = torch.cat([codes, codes.new_zeros(1)])
    low = codes[0::2] & 0x0F
    high = (codes[1::2] & 0x0F) << 4
    return (low | high).contiguous()


def _unpack_nibbles(packed: torch.Tensor, count: int) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 NVFP4 packed data, got {packed.dtype}")
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.empty(packed.numel() * 2, device=packed.device, dtype=torch.uint8)
    codes[0::2] = low
    codes[1::2] = high
    return codes[:count].contiguous()


def quantize_nvfp4_reference(
    x: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
    *,
    preserve_grad: bool = False,
) -> NVFP4Tensor:
    if block_size != NVFP4_BLOCK_SIZE:
        raise ValueError("NeuralFn NVFP4 reference currently uses block_size=16")
    x_float = x.detach().to(dtype=torch.float32).contiguous()
    shape = tuple(int(dim) for dim in x_float.shape)
    flat = x_float.reshape(-1)
    if flat.numel() == 0:
        return NVFP4Tensor(
            packed=torch.empty(0, device=x.device, dtype=torch.uint8),
            block_scales=torch.empty(0, device=x.device, dtype=torch.float8_e4m3fn),
            tensor_scale=torch.tensor(1.0, device=x.device, dtype=torch.float32),
            shape=shape,
            block_size=block_size,
            source=x if preserve_grad and torch.is_floating_point(x) else None,
        )

    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, block_size)
    max_code = NVFP4_E2M1_VALUES.abs().max().item()
    raw_scales = blocks.abs().amax(dim=1) / float(max_code)
    max_scale = raw_scales.amax()
    tensor_scale = torch.where(max_scale > 0.0, max_scale / torch.finfo(torch.float8_e4m3fn).max, max_scale.new_tensor(1.0))
    block_scale_values = torch.where(raw_scales > 0.0, raw_scales / tensor_scale, raw_scales)
    block_scales = block_scale_values.clamp(0.0, torch.finfo(torch.float8_e4m3fn).max).to(dtype=torch.float8_e4m3fn)
    block_scales_float = block_scales.to(dtype=torch.float32)
    denom = (block_scales_float * tensor_scale).unsqueeze(1)
    normalized = torch.where(denom > 0.0, blocks / denom, blocks.new_zeros(blocks.shape))

    codebook = _nvfp4_codebook(x.device)
    distances = (normalized.reshape(-1, 1) - codebook.reshape(1, -1)).abs()
    codes = distances.argmin(dim=1).to(dtype=torch.uint8)
    return NVFP4Tensor(
        packed=_pack_nibbles(codes),
        block_scales=block_scales.contiguous(),
        tensor_scale=tensor_scale.reshape(()).to(dtype=torch.float32),
        shape=shape,
        block_size=block_size,
        source=x if preserve_grad and torch.is_floating_point(x) else None,
    )


def dequantize_nvfp4_reference(encoded: NVFP4Tensor) -> torch.Tensor:
    if encoded.block_size != NVFP4_BLOCK_SIZE:
        raise ValueError("NeuralFn NVFP4 reference currently uses block_size=16")
    total = 1
    for dim in encoded.shape:
        total *= int(dim)
    if total == 0:
        out = torch.empty(encoded.shape, device=encoded.packed.device, dtype=torch.float32)
    else:
        padded_total = ((total + encoded.block_size - 1) // encoded.block_size) * encoded.block_size
        codes = _unpack_nibbles(encoded.packed, padded_total).to(dtype=torch.long)
        values = _nvfp4_codebook(encoded.packed.device)[codes].reshape(-1, encoded.block_size)
        block_scales = encoded.block_scales.to(dtype=torch.float32).reshape(-1, 1)
        out = (values * block_scales * encoded.tensor_scale.to(dtype=torch.float32)).reshape(-1)[:total]
        out = out.reshape(encoded.shape).contiguous()
    if encoded.source is not None:
        source = encoded.source.to(device=out.device, dtype=torch.float32)
        if tuple(source.shape) != encoded.shape:
            raise ValueError(f"NVFP4 STE source shape {tuple(source.shape)} does not match encoded shape {encoded.shape}")
        out = source + (out - source).detach()
    return out


def quantize_dequantize_nvfp4_reference(x: torch.Tensor, block_size: int = NVFP4_BLOCK_SIZE) -> torch.Tensor:
    return dequantize_nvfp4_reference(quantize_nvfp4_reference(x, block_size=block_size))
