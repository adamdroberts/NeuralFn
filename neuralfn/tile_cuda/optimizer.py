from __future__ import annotations

import math

import torch

from .config import TileCudaConfig
from .runtime import load_tile_cuda_extension


def tile_ema_update_reference(target: torch.Tensor, source: torch.Tensor, decay: float) -> torch.Tensor:
    with torch.no_grad():
        target.mul_(float(decay)).add_(source, alpha=1.0 - float(decay))
    return target


def tile_ema_update(target: torch.Tensor, source: torch.Tensor, decay: float, config: TileCudaConfig | None = None) -> torch.Tensor:
    cfg = config or TileCudaConfig()
    can_use = (
        target.is_cuda
        and source.is_cuda
        and target.dtype == torch.float32
        and source.dtype == torch.float32
        and target.is_contiguous()
        and source.is_contiguous()
        and target.shape == source.shape
        and target.numel() > 0
    )
    if not can_use:
        if cfg.strict and (target.is_cuda or source.is_cuda):
            raise RuntimeError("CUDA Tile optimizer 'ema_update' requires same-shape contiguous CUDA float32 tensors")
        return tile_ema_update_reference(target, source, float(decay))
    ext = load_tile_cuda_extension(cfg)
    if ext is None:
        if cfg.strict:
            raise RuntimeError("CUDA Tile extension is unavailable for optimizer 'ema_update'")
        return tile_ema_update_reference(target, source, float(decay))
    with torch.no_grad():
        return ext.tile_ema_update(target, source, float(decay))


def tile_gradient_accumulate_reference(buffer: torch.Tensor, grad: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    with torch.no_grad():
        buffer.add_(grad, alpha=float(scale))
    return buffer


def tile_gradient_accumulate(
    buffer: torch.Tensor,
    grad: torch.Tensor,
    scale: float = 1.0,
    config: TileCudaConfig | None = None,
) -> torch.Tensor:
    cfg = config or TileCudaConfig()
    can_use = (
        buffer.is_cuda
        and grad.is_cuda
        and buffer.dtype == torch.float32
        and grad.dtype == torch.float32
        and buffer.is_contiguous()
        and grad.is_contiguous()
        and buffer.shape == grad.shape
        and buffer.numel() > 0
    )
    if not can_use:
        if cfg.strict and (buffer.is_cuda or grad.is_cuda):
            raise RuntimeError("CUDA Tile optimizer 'gradient_accumulate' requires same-shape contiguous CUDA float32 tensors")
        return tile_gradient_accumulate_reference(buffer, grad, float(scale))
    ext = load_tile_cuda_extension(cfg)
    if ext is None:
        if cfg.strict:
            raise RuntimeError("CUDA Tile extension is unavailable for optimizer 'gradient_accumulate'")
        return tile_gradient_accumulate_reference(buffer, grad, float(scale))
    with torch.no_grad():
        return ext.tile_gradient_accumulate(buffer, grad, float(scale))


def tile_gradient_clip_norm_reference(
    grads: list[torch.Tensor] | tuple[torch.Tensor, ...],
    max_norm: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    tensors = [grad for grad in grads if grad is not None and grad.numel() > 0]
    if not tensors:
        return torch.tensor(0.0)
    with torch.no_grad():
        total = sum(grad.detach().float().square().sum() for grad in tensors)
        total_norm = torch.sqrt(total)
        scale = min(1.0, float(max_norm) / (float(total_norm.item()) + float(eps)))
        if scale < 1.0:
            for grad in tensors:
                grad.mul_(scale)
    return total_norm


def tile_gradient_clip_norm(
    grads: list[torch.Tensor] | tuple[torch.Tensor, ...],
    max_norm: float,
    eps: float = 1e-6,
    config: TileCudaConfig | None = None,
) -> torch.Tensor:
    cfg = config or TileCudaConfig()
    tensors = [grad for grad in grads if grad is not None and grad.numel() > 0]
    if not tensors:
        return torch.tensor(0.0)
    can_use = all(
        grad.is_cuda and grad.dtype == torch.float32 and grad.is_contiguous() and grad.numel() > 0
        for grad in tensors
    )
    if not can_use:
        if cfg.strict and any(grad.is_cuda for grad in tensors):
            raise RuntimeError("CUDA Tile optimizer 'gradient_clip_norm' requires contiguous CUDA float32 tensors")
        return tile_gradient_clip_norm_reference(tensors, float(max_norm), float(eps))
    ext = load_tile_cuda_extension(cfg)
    if ext is None:
        if cfg.strict:
            raise RuntimeError("CUDA Tile extension is unavailable for optimizer 'gradient_clip_norm'")
        return tile_gradient_clip_norm_reference(tensors, float(max_norm), float(eps))
    with torch.no_grad():
        return ext.tile_gradient_clip_norm(tensors, float(max_norm), float(eps))


def tile_adamw_step_reference(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    step: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(step) <= 0:
        raise ValueError("step must be positive")
    with torch.no_grad():
        exp_avg.mul_(float(beta1)).add_(grad, alpha=1.0 - float(beta1))
        exp_avg_sq.mul_(float(beta2)).addcmul_(grad, grad, value=1.0 - float(beta2))
        param.mul_(1.0 - float(lr) * float(weight_decay))
        bias_correction1 = 1.0 - math.pow(float(beta1), int(step))
        bias_correction2 = 1.0 - math.pow(float(beta2), int(step))
        denom = exp_avg_sq.sqrt().div(math.sqrt(bias_correction2)).add_(float(eps))
        param.addcdiv_(exp_avg, denom, value=-(float(lr) / bias_correction1))
    return param, exp_avg, exp_avg_sq


def tile_adamw_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    step: int = 1,
    config: TileCudaConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = config or TileCudaConfig()
    tensors = (param, grad, exp_avg, exp_avg_sq)
    can_use = (
        all(tensor.is_cuda and tensor.dtype == torch.float32 and tensor.is_contiguous() for tensor in tensors)
        and param.numel() > 0
        and param.shape == grad.shape == exp_avg.shape == exp_avg_sq.shape
        and int(step) > 0
    )
    if not can_use:
        if cfg.strict and any(tensor.is_cuda for tensor in tensors):
            raise RuntimeError("CUDA Tile optimizer 'adamw_step' requires same-shape contiguous CUDA float32 tensors")
        return tile_adamw_step_reference(
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            lr=float(lr),
            beta1=float(beta1),
            beta2=float(beta2),
            eps=float(eps),
            weight_decay=float(weight_decay),
            step=int(step),
        )
    ext = load_tile_cuda_extension(cfg)
    if ext is None:
        if cfg.strict:
            raise RuntimeError("CUDA Tile extension is unavailable for optimizer 'adamw_step'")
        return tile_adamw_step_reference(
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            lr=float(lr),
            beta1=float(beta1),
            beta2=float(beta2),
            eps=float(eps),
            weight_decay=float(weight_decay),
            step=int(step),
        )
    with torch.no_grad():
        out_param, out_exp_avg, out_exp_avg_sq = ext.tile_adamw_step(
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            float(lr),
            float(beta1),
            float(beta2),
            float(eps),
            float(weight_decay),
            int(step),
        )
    return out_param, out_exp_avg, out_exp_avg_sq


def tile_muon_newton_schulz_reference(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + float(eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(int(steps)):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


def tile_muon_newton_schulz(
    update: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    config: TileCudaConfig | None = None,
) -> torch.Tensor:
    cfg = config or TileCudaConfig()
    can_use = update.dtype == torch.float32 and update.ndim == 2 and update.numel() > 0
    if not can_use:
        if cfg.strict and update.is_cuda:
            raise RuntimeError("CUDA Tile optimizer 'muon_newton_schulz' requires a non-empty float32 matrix")
    return tile_muon_newton_schulz_reference(update, int(steps), float(eps))


def tile_muon_step_reference(
    param: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    *,
    lr: float,
    momentum: float,
    backend_steps: int,
    nesterov: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        momentum_buffer.mul_(float(momentum)).add_(grad)
        update = grad.add(momentum_buffer, alpha=float(momentum)) if bool(nesterov) else momentum_buffer
        if update.ndim == 2:
            update = tile_muon_newton_schulz_reference(update, steps=int(backend_steps))
            update *= max(1.0, update.size(0) / max(update.size(1), 1)) ** 0.5
        param.add_(update.to(dtype=param.dtype), alpha=-float(lr))
    return param, momentum_buffer


def tile_muon_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    *,
    lr: float,
    momentum: float,
    backend_steps: int,
    nesterov: bool = True,
    config: TileCudaConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = config or TileCudaConfig()
    tensors = (param, grad, momentum_buffer)
    can_use = (
        all(tensor.dtype == torch.float32 and tensor.is_contiguous() for tensor in tensors)
        and param.shape == grad.shape == momentum_buffer.shape
        and param.numel() > 0
    )
    if not can_use and cfg.strict and any(tensor.is_cuda for tensor in tensors):
        raise RuntimeError("CUDA Tile optimizer 'muon_step' requires same-shape contiguous float32 tensors")
    return tile_muon_step_reference(
        param,
        grad,
        momentum_buffer,
        lr=float(lr),
        momentum=float(momentum),
        backend_steps=int(backend_steps),
        nesterov=bool(nesterov),
    )


def tile_split_optimizer_step_reference(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    momentum_buffer: torch.Tensor,
    *,
    lr: float,
    matrix_lr: float | None = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    step: int = 1,
    muon_momentum: float = 0.95,
    muon_backend_steps: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if param.ndim == 2:
        tile_muon_step_reference(
            param,
            grad,
            momentum_buffer,
            lr=float(matrix_lr if matrix_lr is not None else lr),
            momentum=float(muon_momentum),
            backend_steps=int(muon_backend_steps),
            nesterov=True,
        )
    else:
        tile_adamw_step_reference(
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            lr=float(lr),
            beta1=float(beta1),
            beta2=float(beta2),
            eps=float(eps),
            weight_decay=float(weight_decay),
            step=int(step),
        )
    return param, exp_avg, exp_avg_sq, momentum_buffer


def tile_split_optimizer_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    momentum_buffer: torch.Tensor,
    *,
    lr: float,
    matrix_lr: float | None = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    step: int = 1,
    muon_momentum: float = 0.95,
    muon_backend_steps: int = 5,
    config: TileCudaConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = config or TileCudaConfig()
    tensors = (param, grad, exp_avg, exp_avg_sq, momentum_buffer)
    can_use = all(tensor.dtype == torch.float32 and tensor.is_contiguous() for tensor in tensors)
    if not can_use and cfg.strict and any(tensor.is_cuda for tensor in tensors):
        raise RuntimeError("CUDA Tile optimizer 'split_optimizer_step' requires contiguous float32 tensors")
    return tile_split_optimizer_step_reference(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        momentum_buffer,
        lr=float(lr),
        matrix_lr=matrix_lr,
        beta1=float(beta1),
        beta2=float(beta2),
        eps=float(eps),
        weight_decay=float(weight_decay),
        step=int(step),
        muon_momentum=float(muon_momentum),
        muon_backend_steps=int(muon_backend_steps),
    )
