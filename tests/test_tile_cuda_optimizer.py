from __future__ import annotations

import pytest
import torch

from neuralfn.tile_cuda import (
    TileCudaConfig,
    load_tile_cuda_extension,
    tile_adamw_step,
    tile_ema_update,
    tile_gradient_accumulate,
    tile_gradient_clip_norm,
    tile_muon_newton_schulz,
    tile_muon_newton_schulz_reference,
    tile_muon_step,
    tile_muon_step_reference,
    tile_split_optimizer_step,
    tile_split_optimizer_step_reference,
)


def _ema_inputs(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.linspace(-2.0, 2.0, 2051, dtype=torch.float32, device=device)
    source = torch.linspace(1.5, -1.5, 2051, dtype=torch.float32, device=device)
    expected = target.detach().clone()
    return target, source, expected


def test_tile_ema_update_cpu_fallback_matches_torch() -> None:
    target, source, expected = _ema_inputs("cpu")
    decay = 0.875

    actual = tile_ema_update(target, source, decay, TileCudaConfig(backend="tile_cuda", strict=False))
    expected.mul_(decay).add_(source, alpha=1.0 - decay)

    assert actual is target
    torch.testing.assert_close(target, expected)


def test_tile_ema_update_gpu_kernel_matches_torch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda is not available")
    config = TileCudaConfig(backend="tile_cuda", strict=True, build_enabled=True)
    if load_tile_cuda_extension(config) is None:
        pytest.skip("CUDA Tile extension could not be built or loaded in this environment")

    target, source, expected = _ema_inputs("cuda")
    decay = 0.875

    actual = tile_ema_update(target, source, decay, config)
    expected.mul_(decay).add_(source, alpha=1.0 - decay)

    assert actual is target
    torch.testing.assert_close(target, expected, rtol=1e-5, atol=1e-6)


def _accumulate_inputs(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    buffer = torch.linspace(-3.0, 3.0, 2051, dtype=torch.float32, device=device)
    grad = torch.linspace(2.0, -2.0, 2051, dtype=torch.float32, device=device)
    expected = buffer.detach().clone()
    return buffer, grad, expected


def test_tile_gradient_accumulate_cpu_fallback_matches_torch() -> None:
    buffer, grad, expected = _accumulate_inputs("cpu")
    scale = 0.25

    actual = tile_gradient_accumulate(buffer, grad, scale, TileCudaConfig(backend="tile_cuda", strict=False))
    expected.add_(grad, alpha=scale)

    assert actual is buffer
    torch.testing.assert_close(buffer, expected)


def test_tile_gradient_accumulate_gpu_kernel_matches_torch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda is not available")
    config = TileCudaConfig(backend="tile_cuda", strict=True, build_enabled=True)
    if load_tile_cuda_extension(config) is None:
        pytest.skip("CUDA Tile extension could not be built or loaded in this environment")

    buffer, grad, expected = _accumulate_inputs("cuda")
    scale = 0.25

    actual = tile_gradient_accumulate(buffer, grad, scale, config)
    expected.add_(grad, alpha=scale)

    assert actual is buffer
    torch.testing.assert_close(buffer, expected, rtol=1e-5, atol=1e-6)


def _clip_inputs(device: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    grads = [
        torch.linspace(-2.0, 2.0, 1025, dtype=torch.float32, device=device),
        torch.linspace(1.5, -1.5, 2051, dtype=torch.float32, device=device),
    ]
    expected = [grad.detach().clone() for grad in grads]
    return grads, expected


def _clip_expected(grads: list[torch.Tensor], max_norm: float, eps: float = 1e-6) -> torch.Tensor:
    total_norm = torch.sqrt(sum(grad.float().square().sum() for grad in grads))
    scale = min(1.0, float(max_norm) / (float(total_norm.item()) + eps))
    if scale < 1.0:
        for grad in grads:
            grad.mul_(scale)
    return total_norm


def test_tile_gradient_clip_norm_cpu_fallback_matches_torch() -> None:
    grads, expected = _clip_inputs("cpu")
    max_norm = 7.5

    actual_norm = tile_gradient_clip_norm(grads, max_norm, config=TileCudaConfig(backend="tile_cuda", strict=False))
    expected_norm = _clip_expected(expected, max_norm)

    torch.testing.assert_close(actual_norm, expected_norm)
    for actual, expected_tensor in zip(grads, expected, strict=True):
        torch.testing.assert_close(actual, expected_tensor)


def test_tile_gradient_clip_norm_gpu_kernel_matches_torch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda is not available")
    config = TileCudaConfig(backend="tile_cuda", strict=True, build_enabled=True)
    if load_tile_cuda_extension(config) is None:
        pytest.skip("CUDA Tile extension could not be built or loaded in this environment")

    grads, expected = _clip_inputs("cuda")
    max_norm = 7.5

    actual_norm = tile_gradient_clip_norm(grads, max_norm, config=config)
    expected_norm = _clip_expected(expected, max_norm)

    torch.testing.assert_close(actual_norm, expected_norm, rtol=1e-5, atol=1e-6)
    for actual, expected_tensor in zip(grads, expected, strict=True):
        torch.testing.assert_close(actual, expected_tensor, rtol=1e-5, atol=1e-6)


def _adamw_inputs(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    param = torch.linspace(-1.0, 1.0, 2051, dtype=torch.float32, device=device)
    grad = torch.linspace(0.75, -0.5, 2051, dtype=torch.float32, device=device)
    exp_avg = torch.linspace(-0.2, 0.2, 2051, dtype=torch.float32, device=device)
    exp_avg_sq = torch.linspace(0.1, 0.3, 2051, dtype=torch.float32, device=device)
    expected = (param.detach().clone(), exp_avg.detach().clone(), exp_avg_sq.detach().clone())
    return param, grad, exp_avg, exp_avg_sq, expected


def _adamw_expected(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    param.mul_(1.0 - lr * weight_decay)
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    denom = exp_avg_sq.sqrt().div(bias_correction2**0.5).add_(eps)
    param.addcdiv_(exp_avg, denom, value=-(lr / bias_correction1))
    return param, exp_avg, exp_avg_sq


def test_tile_adamw_step_cpu_fallback_matches_torch() -> None:
    param, grad, exp_avg, exp_avg_sq, expected = _adamw_inputs("cpu")
    kwargs = dict(lr=0.003, beta1=0.8, beta2=0.95, eps=1e-6, weight_decay=0.01, step=4)

    actual_param, actual_exp_avg, actual_exp_avg_sq = tile_adamw_step(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        config=TileCudaConfig(backend="tile_cuda", strict=False),
        **kwargs,
    )
    expected_param, expected_exp_avg, expected_exp_avg_sq = _adamw_expected(
        expected[0],
        grad,
        expected[1],
        expected[2],
        **kwargs,
    )

    assert actual_param is param
    assert actual_exp_avg is exp_avg
    assert actual_exp_avg_sq is exp_avg_sq
    torch.testing.assert_close(param, expected_param)
    torch.testing.assert_close(exp_avg, expected_exp_avg)
    torch.testing.assert_close(exp_avg_sq, expected_exp_avg_sq)


def test_tile_adamw_step_gpu_kernel_matches_torch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda is not available")
    config = TileCudaConfig(backend="tile_cuda", strict=True, build_enabled=True)
    if load_tile_cuda_extension(config) is None:
        pytest.skip("CUDA Tile extension could not be built or loaded in this environment")

    param, grad, exp_avg, exp_avg_sq, expected = _adamw_inputs("cuda")
    kwargs = dict(lr=0.003, beta1=0.8, beta2=0.95, eps=1e-6, weight_decay=0.01, step=4)

    actual_param, actual_exp_avg, actual_exp_avg_sq = tile_adamw_step(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        config=config,
        **kwargs,
    )
    expected_param, expected_exp_avg, expected_exp_avg_sq = _adamw_expected(
        expected[0],
        grad,
        expected[1],
        expected[2],
        **kwargs,
    )

    assert actual_param is param
    assert actual_exp_avg is exp_avg
    assert actual_exp_avg_sq is exp_avg_sq
    torch.testing.assert_close(param, expected_param, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(exp_avg, expected_exp_avg, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(exp_avg_sq, expected_exp_avg_sq, rtol=1e-5, atol=1e-6)


def _muon_inputs(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    param = torch.linspace(-1.0, 1.0, 4 * 6, dtype=torch.float32, device=device).reshape(4, 6)
    grad = torch.linspace(0.5, -0.75, 4 * 6, dtype=torch.float32, device=device).reshape(4, 6)
    momentum_buffer = torch.linspace(-0.1, 0.2, 4 * 6, dtype=torch.float32, device=device).reshape(4, 6)
    expected = (param.detach().clone(), momentum_buffer.detach().clone())
    return param, grad, momentum_buffer, expected


def test_tile_muon_newton_schulz_cpu_matches_reference() -> None:
    update = torch.linspace(-0.75, 0.5, 4 * 6, dtype=torch.float32).reshape(4, 6)
    actual = tile_muon_newton_schulz(update, steps=3, config=TileCudaConfig(backend="tile_cuda", strict=False))
    expected = tile_muon_newton_schulz_reference(update, steps=3)
    torch.testing.assert_close(actual.float(), expected.float())


def test_tile_muon_newton_schulz_gpu_matches_reference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda is not available")
    config = TileCudaConfig(backend="tile_cuda", strict=True, build_enabled=True)
    if load_tile_cuda_extension(config) is None:
        pytest.skip("CUDA Tile extension could not be built or loaded in this environment")
    update = torch.linspace(-0.75, 0.5, 4 * 6, dtype=torch.float32, device="cuda").reshape(4, 6)
    actual = tile_muon_newton_schulz(update, steps=3, config=config)
    expected = tile_muon_newton_schulz_reference(update, steps=3)
    torch.testing.assert_close(actual.float(), expected.float())


def test_tile_muon_step_cpu_matches_reference() -> None:
    param, grad, momentum_buffer, expected = _muon_inputs("cpu")
    kwargs = dict(lr=0.02, momentum=0.9, backend_steps=3, nesterov=True)
    actual_param, actual_momentum = tile_muon_step(
        param,
        grad,
        momentum_buffer,
        config=TileCudaConfig(backend="tile_cuda", strict=False),
        **kwargs,
    )
    expected_param, expected_momentum = tile_muon_step_reference(expected[0], grad, expected[1], **kwargs)
    assert actual_param is param
    assert actual_momentum is momentum_buffer
    torch.testing.assert_close(param, expected_param)
    torch.testing.assert_close(momentum_buffer, expected_momentum)


def test_tile_muon_step_gpu_matches_reference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda is not available")
    config = TileCudaConfig(backend="tile_cuda", strict=True, build_enabled=True)
    if load_tile_cuda_extension(config) is None:
        pytest.skip("CUDA Tile extension could not be built or loaded in this environment")
    param, grad, momentum_buffer, expected = _muon_inputs("cuda")
    kwargs = dict(lr=0.02, momentum=0.9, backend_steps=3, nesterov=True)
    actual_param, actual_momentum = tile_muon_step(param, grad, momentum_buffer, config=config, **kwargs)
    expected_param, expected_momentum = tile_muon_step_reference(expected[0], grad, expected[1], **kwargs)
    assert actual_param is param
    assert actual_momentum is momentum_buffer
    torch.testing.assert_close(param, expected_param, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(momentum_buffer, expected_momentum, rtol=1e-5, atol=1e-6)


def test_tile_split_optimizer_step_cpu_matches_reference() -> None:
    param, grad, momentum_buffer, _ = _muon_inputs("cpu")
    exp_avg = torch.zeros_like(param)
    exp_avg_sq = torch.ones_like(param) * 0.1
    expected = (param.detach().clone(), exp_avg.detach().clone(), exp_avg_sq.detach().clone(), momentum_buffer.detach().clone())
    kwargs = dict(lr=0.003, matrix_lr=0.02, beta1=0.8, beta2=0.95, eps=1e-6, weight_decay=0.01, step=4, muon_momentum=0.9, muon_backend_steps=3)
    actual = tile_split_optimizer_step(param, grad, exp_avg, exp_avg_sq, momentum_buffer, config=TileCudaConfig(backend="tile_cuda", strict=False), **kwargs)
    expected_out = tile_split_optimizer_step_reference(expected[0], grad, expected[1], expected[2], expected[3], **kwargs)
    for actual_tensor, expected_tensor in zip(actual, expected_out, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_tile_split_optimizer_step_gpu_matches_reference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda is not available")
    config = TileCudaConfig(backend="tile_cuda", strict=True, build_enabled=True)
    if load_tile_cuda_extension(config) is None:
        pytest.skip("CUDA Tile extension could not be built or loaded in this environment")
    param, grad, momentum_buffer, _ = _muon_inputs("cuda")
    exp_avg = torch.zeros_like(param)
    exp_avg_sq = torch.ones_like(param) * 0.1
    expected = (param.detach().clone(), exp_avg.detach().clone(), exp_avg_sq.detach().clone(), momentum_buffer.detach().clone())
    kwargs = dict(lr=0.003, matrix_lr=0.02, beta1=0.8, beta2=0.95, eps=1e-6, weight_decay=0.01, step=4, muon_momentum=0.9, muon_backend_steps=3)
    actual = tile_split_optimizer_step(param, grad, exp_avg, exp_avg_sq, momentum_buffer, config=config, **kwargs)
    expected_out = tile_split_optimizer_step_reference(expected[0], grad, expected[1], expected[2], expected[3], **kwargs)
    for actual_tensor, expected_tensor in zip(actual, expected_out, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-6)
