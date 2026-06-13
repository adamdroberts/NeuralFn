from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F

from .config import TileCudaConfig
from .dtypes import NVFP4Tensor, dequantize_nvfp4_reference
from .runtime import load_tile_cuda_extension


UNARY_OPS: dict[str, int] = {
    "identity": 0,
    "negate": 1,
    "relu": 2,
    "sigmoid": 3,
    "tanh_neuron": 4,
    "leaky_relu": 5,
    "silu": 6,
    "softplus": 7,
    "hard_tanh": 8,
    "gaussian": 9,
    "log": 10,
    "prelu": 11,
    "relu6": 12,
    "elu": 13,
    "selu": 14,
    "mish": 15,
    "softsign": 16,
    "hard_sigmoid": 17,
    "hard_swish": 18,
    "threshold": 19,
    "gelu": 20,
}

BINARY_OPS: dict[str, int] = {
    "add": 0,
    "multiply": 1,
}

BINARY_PAIR_OPS: dict[str, int] = {
    "softmax_2": 0,
    "logsoftmax_2": 1,
}

TILE_FUNCTION_NAMES = frozenset((*UNARY_OPS.keys(), *BINARY_OPS.keys(), *BINARY_PAIR_OPS.keys()))

SCALAR_UNARY_MODULE_OPS: dict[str, int] = {
    "loss_scale": 0,
    "logit_softcap": 1,
}

SCALAR_BINARY_MODULE_OPS: dict[str, int] = {
    "aux_loss_add": 0,
}

SCALAR_TERNARY_MODULE_OPS: dict[str, int] = {
    "kl_penalty": 0,
}

VECTOR_BINARY_MODULE_OPS: dict[str, int] = {
    "residual_add": 0,
    "residual_mix": 1,
    "manifold_hyper_connection": 2,
}

TILE_MODULE_NAMES = frozenset(
    (
        *SCALAR_UNARY_MODULE_OPS.keys(),
        *SCALAR_BINARY_MODULE_OPS.keys(),
        *SCALAR_TERNARY_MODULE_OPS.keys(),
        *VECTOR_BINARY_MODULE_OPS.keys(),
        "act_halt_gate",
        "attentionless_decoder",
        "auxfree_load_balancing",
        "expert_combine",
        "expert_dispatch",
        "gae_compute",
        "gelu",
        "jepa_projector",
        "jepa_predictor",
        "kv_pca_encode",
        "kv_pca_decode",
        "kv_quant_pack",
        "kv_quant_unpack",
        "kv_cache_write",
        "kv_cache_read",
        "latent_pool",
        "absolute_position_embedding",
        "token_embedding",
        "causal_chunk_state",
        "qk_gain",
        "dyt",
        "dropout",
        "reshape_heads",
        "merge_heads",
        "repeat_kv",
        "rotary_embedding",
        "rms_norm",
        "layer_norm",
        "group_norm",
        "qk_norm",
        "linear",
        "bitlinear_ternary",
        "fp8_linear",
        "mx_linear",
        "nf4_linear",
        "randmap_adapter",
        "mlp_relu2",
        "swiglu",
        "geglu",
        "reglu",
        "solu",
        "load_balance_loss",
        "lm_head",
        "tied_lm_head",
        "ttt_linear",
        "lora_linear",
        "act_weighted_sum",
        "dpo_pairwise_loss",
        "masked_token_cross_entropy",
        "ppo_clipped_loss",
        "preference_bce_loss",
        "route_balance_loss",
        "route_distillation_loss",
        "route_selection_loss",
        "semantic_alignment_loss",
        "semantic_chunk_hasher",
        "semantic_chunk_projector",
        "semantic_hash_router",
        "semantic_hasher",
        "semantic_moe_jepa_evo_router",
        "semantic_moe_router",
        "semantic_projector",
        "sequence_logp",
        "softmax_distillation_loss",
        "scaled_dot_product_attention",
        "sliding_window_attention",
        "block_sparse_attention",
        "streaming_attention_sinks",
        "native_sparse_attention",
        "token_cross_entropy",
        "topk_route",
        "broadcast_expert_routes",
        "broadcast_chunk_routes",
        "byte_patch_embed",
        "byte_patch_merge",
        "latent_mse_loss",
    )
)


class TileCudaAutogradNotImplemented(torch.autograd.Function):
    """Sentinel autograd wrapper used until real CUDA Tile kernels land."""

    @staticmethod
    def forward(ctx, *args):  # type: ignore[override]
        del ctx, args
        raise NotImplementedError("CUDA Tile autograd kernels have not been implemented yet")


def require_tile_cuda_kernel(name: str) -> Callable[..., torch.Tensor]:
    def _missing(*_args, **_kwargs):
        raise NotImplementedError(f"CUDA Tile kernel '{name}' has not been implemented yet")

    return _missing


TILE_FLOAT_DTYPES = frozenset((torch.float32, torch.float16))
TILE_FP8_DTYPES = frozenset((torch.float8_e4m3fn, torch.float8_e5m2))
TILE_ELEMENTWISE_INPUT_DTYPES = frozenset((*TILE_FLOAT_DTYPES, *TILE_FP8_DTYPES))
TILE_LINEAR_INPUT_DTYPES = frozenset((*TILE_FLOAT_DTYPES, *TILE_FP8_DTYPES))


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _supported_dtype_names(allow_fp16: bool = False, allow_fp8: bool = False) -> str:
    dtypes = [torch.float32]
    if allow_fp16:
        dtypes.append(torch.float16)
    if allow_fp8:
        dtypes.extend((torch.float8_e4m3fn, torch.float8_e5m2))
    return ", ".join(_dtype_name(dtype) for dtype in dtypes)


def _tensor_contract_summary(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}: device={tensor.device.type}, dtype={_dtype_name(tensor.dtype)}, "
        f"contiguous={tensor.is_contiguous()}, shape={tuple(tensor.shape)}"
    )


def _linear_contract_summary(name: str, x: torch.Tensor | NVFP4Tensor) -> str:
    if isinstance(x, NVFP4Tensor):
        return (
            f"{name}: device={x.packed.device.type}, dtype=nvfp4, "
            f"contiguous={x.packed.is_contiguous()}, shape={x.shape}, block_size={x.block_size}"
        )
    return _tensor_contract_summary(name, x)


def _strict_unary_contract_error(
    kind: str,
    name: str,
    x: torch.Tensor,
    *,
    allow_fp16: bool = False,
    allow_fp8: bool = False,
) -> RuntimeError:
    supported = _supported_dtype_names(allow_fp16=allow_fp16, allow_fp8=allow_fp8)
    return RuntimeError(
        f"CUDA Tile {kind} '{name}' requires contiguous CUDA input with supported dtypes {{{supported}}}; "
        f"got {_tensor_contract_summary('x', x)}"
    )


def _strict_binary_contract_error(
    kind: str,
    name: str,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    allow_fp16: bool = False,
    allow_fp8: bool = False,
) -> RuntimeError:
    supported = _supported_dtype_names(allow_fp16=allow_fp16, allow_fp8=allow_fp8)
    return RuntimeError(
        f"CUDA Tile {kind} '{name}' requires same-shape contiguous CUDA inputs with matching supported dtypes "
        f"{{{supported}}}; got {_tensor_contract_summary('lhs', lhs)}; {_tensor_contract_summary('rhs', rhs)}"
    )


def _strict_ternary_contract_error(
    kind: str,
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    allow_fp16: bool = False,
    allow_fp8: bool = False,
) -> RuntimeError:
    supported = _supported_dtype_names(allow_fp16=allow_fp16, allow_fp8=allow_fp8)
    return RuntimeError(
        f"CUDA Tile {kind} '{name}' requires same-shape contiguous CUDA inputs with matching supported dtypes "
        f"{{{supported}}}; got {_tensor_contract_summary('a', a)}; "
        f"{_tensor_contract_summary('b', b)}; {_tensor_contract_summary('c', c)}"
    )


def _tile_kernel_input(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.float32 else x.to(dtype=torch.float32)


def _linear_input_tensor(x: torch.Tensor | NVFP4Tensor) -> torch.Tensor:
    if isinstance(x, NVFP4Tensor):
        return dequantize_nvfp4_reference(x)
    return x


def _tile_kernel_output(out: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return out if out.dtype == dtype else out.to(dtype=dtype)


def _tile_linear_output(out: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
    if input_dtype in TILE_FP8_DTYPES:
        return out
    return _tile_kernel_output(out, input_dtype)


def _tile_attention_output(out: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
    if input_dtype in TILE_FP8_DTYPES:
        return out
    return _tile_kernel_output(out, input_dtype)


def _can_use_tile_unary(x: torch.Tensor, *, allow_fp16: bool = False, allow_fp8: bool = False) -> bool:
    allowed = (
        TILE_ELEMENTWISE_INPUT_DTYPES
        if allow_fp16 and allow_fp8
        else TILE_FLOAT_DTYPES
        if allow_fp16
        else frozenset((torch.float32,))
    )
    return x.is_cuda and x.dtype in allowed and x.is_contiguous()


def _can_use_tile_binary(lhs: torch.Tensor, rhs: torch.Tensor, *, allow_fp16: bool = False, allow_fp8: bool = False) -> bool:
    allowed = (
        TILE_ELEMENTWISE_INPUT_DTYPES
        if allow_fp16 and allow_fp8
        else TILE_FLOAT_DTYPES
        if allow_fp16
        else frozenset((torch.float32,))
    )
    return (
        lhs.is_cuda
        and rhs.is_cuda
        and lhs.dtype in allowed
        and rhs.dtype == lhs.dtype
        and lhs.is_contiguous()
        and rhs.is_contiguous()
        and lhs.shape == rhs.shape
    )


def _can_use_tile_ternary(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    allow_fp16: bool = False,
    allow_fp8: bool = False,
) -> bool:
    return (
        _can_use_tile_binary(a, b, allow_fp16=allow_fp16, allow_fp8=allow_fp8)
        and c.is_cuda
        and c.dtype == a.dtype
        and c.is_contiguous()
        and c.shape == a.shape
    )


def _can_use_tile_identity(x: torch.Tensor) -> bool:
    return x.is_cuda and x.dtype in TILE_ELEMENTWISE_INPUT_DTYPES and x.is_contiguous()


def _can_use_tile_vector_binary(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    scale0: torch.Tensor,
    scale1: torch.Tensor | None = None,
    *,
    allow_fp16: bool = False,
    allow_fp8: bool = False,
) -> bool:
    if not _can_use_tile_binary(lhs, rhs, allow_fp16=allow_fp16, allow_fp8=allow_fp8):
        return False
    if lhs.ndim == 0:
        return False
    if not (scale0.is_cuda and scale0.dtype == torch.float32 and scale0.is_contiguous() and scale0.ndim == 1):
        return False
    if lhs.shape[-1] != scale0.numel():
        return False
    if scale1 is not None and not (
        scale1.is_cuda
        and scale1.dtype == torch.float32
        and scale1.is_contiguous()
        and scale1.shape == scale0.shape
    ):
        return False
    return True


def _sum_to_last_dim(x: torch.Tensor) -> torch.Tensor:
    if x.ndim <= 1:
        return x
    return x.reshape(-1, x.shape[-1]).sum(dim=0)


def _sum_to_head_dim(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        return x
    reduce_dims = tuple(dim for dim in range(x.ndim) if dim != 1)
    return x.sum(dim=reduce_dims)


def _float_outputs(value: torch.Tensor | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return tuple(item for item in value if torch.is_floating_point(item))
    return (value,) if torch.is_floating_point(value) else ()


def _fallback_unary(name: str, x: torch.Tensor) -> torch.Tensor:
    x_float = _tile_kernel_input(x) if torch.is_floating_point(x) else x.float()
    if name == "identity":
        return x
    if name == "negate":
        return _tile_kernel_output(-x_float, x.dtype)
    if name == "relu":
        return _tile_kernel_output(torch.relu(x_float), x.dtype)
    if name == "sigmoid":
        return _tile_kernel_output(torch.sigmoid(x_float), x.dtype)
    if name == "tanh_neuron":
        return _tile_kernel_output(torch.tanh(x_float), x.dtype)
    if name == "leaky_relu":
        return _tile_kernel_output(F.leaky_relu(x_float, negative_slope=0.01), x.dtype)
    if name == "silu":
        return _tile_kernel_output(F.silu(x_float), x.dtype)
    if name == "softplus":
        return _tile_kernel_output(F.softplus(x_float), x.dtype)
    if name == "hard_tanh":
        return _tile_kernel_output(F.hardtanh(x_float), x.dtype)
    if name == "gaussian":
        return _tile_kernel_output(torch.exp(-(x_float * x_float)), x.dtype)
    if name == "log":
        return _tile_kernel_output(torch.log(torch.clamp_min(x_float, 1e-7)), x.dtype)
    if name == "prelu":
        return _tile_kernel_output(torch.where(x_float >= 0, x_float, x_float * 0.25), x.dtype)
    if name == "relu6":
        return _tile_kernel_output(torch.clamp(x_float, min=0.0, max=6.0), x.dtype)
    if name == "elu":
        return _tile_kernel_output(F.elu(x_float), x.dtype)
    if name == "selu":
        return _tile_kernel_output(F.selu(x_float), x.dtype)
    if name == "mish":
        return _tile_kernel_output(F.mish(x_float), x.dtype)
    if name == "softsign":
        return _tile_kernel_output(F.softsign(x_float), x.dtype)
    if name == "hard_sigmoid":
        return _tile_kernel_output(F.hardsigmoid(x_float), x.dtype)
    if name == "hard_swish":
        return _tile_kernel_output(F.hardswish(x_float), x.dtype)
    if name == "threshold":
        return _tile_kernel_output(x_float * 0.0 + (x_float >= 0.0).to(dtype=x_float.dtype), x.dtype)
    if name == "gelu":
        return _tile_kernel_output(F.gelu(x_float), x.dtype)
    raise KeyError(f"Unsupported CUDA Tile unary function: {name}")


class _TileUnaryFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, op_name: str, op_code: int, config: TileCudaConfig):  # type: ignore[override]
        ctx.op_name = op_name
        ctx.save_for_backward(x)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError(f"CUDA Tile extension is unavailable for function '{op_name}'")
            return _fallback_unary(op_name, x)
        out = ext.tile_unary(_tile_kernel_input(x), int(op_code))
        return _tile_kernel_output(out, x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (saved_x,) = ctx.saved_tensors
        x = _tile_kernel_input(saved_x)
        grad_output = _tile_kernel_input(grad_output)
        op_name = ctx.op_name
        if op_name == "identity":
            grad = grad_output
        elif op_name == "negate":
            grad = -grad_output
        elif op_name == "relu":
            grad = grad_output * (x > 0).to(dtype=grad_output.dtype)
        elif op_name == "sigmoid":
            y = torch.sigmoid(x)
            grad = grad_output * y * (1.0 - y)
        elif op_name == "tanh_neuron":
            y = torch.tanh(x)
            grad = grad_output * (1.0 - y * y)
        elif op_name == "leaky_relu":
            slope = torch.where(x >= 0, torch.ones_like(x), torch.full_like(x, 0.01))
            grad = grad_output * slope
        elif op_name == "silu":
            s = torch.sigmoid(x)
            grad = grad_output * s * (1.0 + x * (1.0 - s))
        elif op_name == "softplus":
            grad = grad_output * torch.sigmoid(x)
        elif op_name == "hard_tanh":
            grad = grad_output * ((x > -1.0) & (x < 1.0)).to(dtype=grad_output.dtype)
        elif op_name == "gaussian":
            y = torch.exp(-(x * x))
            grad = grad_output * (-2.0 * x * y)
        elif op_name == "log":
            grad = grad_output * torch.where(x >= 1e-7, 1.0 / x, torch.zeros_like(x))
        elif op_name == "prelu":
            grad = grad_output * torch.where(x >= 0, torch.ones_like(x), torch.full_like(x, 0.25))
        elif op_name == "relu6":
            grad = grad_output * ((x > 0.0) & (x < 6.0)).to(dtype=grad_output.dtype)
        elif op_name == "elu":
            grad = grad_output * torch.where(x >= 0, torch.ones_like(x), torch.exp(x))
        elif op_name == "selu":
            alpha = 1.6732632423543772
            scale = 1.0507009873554805
            grad = grad_output * torch.where(x >= 0, torch.full_like(x, scale), torch.exp(x) * (scale * alpha))
        elif op_name == "mish":
            sp = F.softplus(x)
            tsp = torch.tanh(sp)
            grad = grad_output * (tsp + x * torch.sigmoid(x) * (1.0 - tsp * tsp))
        elif op_name == "softsign":
            grad = grad_output / torch.square(1.0 + torch.abs(x))
        elif op_name == "hard_sigmoid":
            grad = grad_output * ((x > -3.0) & (x < 3.0)).to(dtype=grad_output.dtype) / 6.0
        elif op_name == "hard_swish":
            h = torch.clamp(x / 6.0 + 0.5, min=0.0, max=1.0)
            dh = ((x > -3.0) & (x < 3.0)).to(dtype=grad_output.dtype) / 6.0
            grad = grad_output * (h + x * dh)
        elif op_name == "threshold":
            grad = torch.zeros_like(x)
        elif op_name == "gelu":
            inv_sqrt2 = 0.7071067811865476
            inv_sqrt2pi = 0.3989422804014327
            grad = grad_output * (
                0.5 * (1.0 + torch.erf(x * inv_sqrt2)) + x * torch.exp(-0.5 * x * x) * inv_sqrt2pi
            )
        else:
            raise KeyError(f"Unsupported CUDA Tile unary backward function: {op_name}")
        return _tile_kernel_output(grad, saved_x.dtype), None, None, None


class _TileBinaryFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lhs: torch.Tensor, rhs: torch.Tensor, op_name: str, op_code: int, config: TileCudaConfig):  # type: ignore[override]
        ctx.op_name = op_name
        ctx.save_for_backward(lhs, rhs)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError(f"CUDA Tile extension is unavailable for function '{op_name}'")
            return lhs + rhs if op_name == "add" else lhs * rhs
        out = ext.tile_binary(_tile_kernel_input(lhs), _tile_kernel_input(rhs), int(op_code))
        return _tile_kernel_output(out, lhs.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        saved_lhs, saved_rhs = ctx.saved_tensors
        lhs = _tile_kernel_input(saved_lhs)
        rhs = _tile_kernel_input(saved_rhs)
        grad_output = _tile_kernel_input(grad_output)
        op_name = ctx.op_name
        if op_name == "add":
            return _tile_kernel_output(grad_output, saved_lhs.dtype), _tile_kernel_output(grad_output, saved_rhs.dtype), None, None, None
        if op_name == "multiply":
            return (
                _tile_kernel_output(grad_output * rhs, saved_lhs.dtype),
                _tile_kernel_output(grad_output * lhs, saved_rhs.dtype),
                None,
                None,
                None,
            )
        raise KeyError(f"Unsupported CUDA Tile binary backward function: {op_name}")


class _TileBinaryPairFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lhs: torch.Tensor, rhs: torch.Tensor, op_name: str, op_code: int, config: TileCudaConfig):  # type: ignore[override]
        ctx.op_name = op_name
        ctx.save_for_backward(lhs, rhs)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError(f"CUDA Tile extension is unavailable for function '{op_name}'")
            return _fallback_binary_pair(op_name, lhs, rhs)
        out0, out1 = ext.tile_binary_pair(_tile_kernel_input(lhs), _tile_kernel_input(rhs), int(op_code))
        return _tile_kernel_output(out0, lhs.dtype), _tile_kernel_output(out1, lhs.dtype)

    @staticmethod
    def backward(ctx, grad_output0: torch.Tensor, grad_output1: torch.Tensor):  # type: ignore[override]
        saved_lhs, saved_rhs = ctx.saved_tensors
        lhs = _tile_kernel_input(saved_lhs)
        rhs = _tile_kernel_input(saved_rhs)
        grad_output0 = _tile_kernel_input(grad_output0)
        grad_output1 = _tile_kernel_input(grad_output1)
        op_name = ctx.op_name
        stacked = torch.stack((lhs, rhs), dim=0)
        probs = torch.softmax(stacked, dim=0)
        p0, p1 = probs[0], probs[1]
        if op_name == "softmax_2":
            dot = grad_output0 * p0 + grad_output1 * p1
            return (
                _tile_kernel_output(p0 * (grad_output0 - dot), saved_lhs.dtype),
                _tile_kernel_output(p1 * (grad_output1 - dot), saved_rhs.dtype),
                None,
                None,
                None,
            )
        if op_name == "logsoftmax_2":
            grad_sum = grad_output0 + grad_output1
            return (
                _tile_kernel_output(grad_output0 - p0 * grad_sum, saved_lhs.dtype),
                _tile_kernel_output(grad_output1 - p1 * grad_sum, saved_rhs.dtype),
                None,
                None,
                None,
            )
        raise KeyError(f"Unsupported CUDA Tile binary-pair backward function: {op_name}")


def tile_unary(name: str, x: torch.Tensor, config: TileCudaConfig) -> torch.Tensor:
    if not _can_use_tile_unary(x, allow_fp16=True, allow_fp8=True):
        if config.strict and x.is_cuda:
            raise _strict_unary_contract_error("function", name, x, allow_fp16=True, allow_fp8=True)
        return _fallback_unary(name, x)
    return _TileUnaryFunction.apply(x, name, UNARY_OPS[name], config)


def tile_binary(name: str, lhs: torch.Tensor, rhs: torch.Tensor, config: TileCudaConfig) -> torch.Tensor:
    if not _can_use_tile_binary(lhs, rhs, allow_fp16=True, allow_fp8=True):
        if config.strict and (lhs.is_cuda or rhs.is_cuda):
            raise _strict_binary_contract_error("function", name, lhs, rhs, allow_fp16=True, allow_fp8=True)
        return lhs + rhs if name == "add" else lhs * rhs
    return _TileBinaryFunction.apply(lhs, rhs, name, BINARY_OPS[name], config)


def _fallback_binary_pair(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    stacked = torch.stack((_tile_kernel_input(lhs), _tile_kernel_input(rhs)), dim=0)
    if name == "softmax_2":
        out = torch.softmax(stacked, dim=0)
    elif name == "logsoftmax_2":
        out = torch.log_softmax(stacked, dim=0)
    else:
        raise KeyError(f"Unsupported CUDA Tile binary-pair function: {name}")
    return _tile_kernel_output(out[0], lhs.dtype), _tile_kernel_output(out[1], rhs.dtype)


def tile_binary_pair(name: str, lhs: torch.Tensor, rhs: torch.Tensor, config: TileCudaConfig) -> tuple[torch.Tensor, torch.Tensor]:
    if not _can_use_tile_binary(lhs, rhs, allow_fp16=True, allow_fp8=True):
        if config.strict and (lhs.is_cuda or rhs.is_cuda):
            raise _strict_binary_contract_error("function", name, lhs, rhs, allow_fp16=True, allow_fp8=True)
        return _fallback_binary_pair(name, lhs, rhs)
    return _TileBinaryPairFunction.apply(lhs, rhs, name, BINARY_PAIR_OPS[name], config)


class _TileScalarUnaryModuleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, op_name: str, op_code: int, value: float, config: TileCudaConfig):  # type: ignore[override]
        ctx.op_name = op_name
        ctx.value = float(value)
        ctx.save_for_backward(x)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError(f"CUDA Tile extension is unavailable for module '{op_name}'")
            return tile_scalar_unary_module_reference(op_name, x, float(value))
        out = ext.tile_scalar_unary(_tile_kernel_input(x), float(value), int(op_code))
        return _tile_kernel_output(out, x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (x,) = ctx.saved_tensors
        op_name = ctx.op_name
        value = ctx.value
        grad = grad_output if grad_output.dtype == torch.float32 else grad_output.to(dtype=torch.float32)
        x_float = _tile_kernel_input(x)
        if op_name == "loss_scale":
            grad_x = grad * value
        elif op_name == "logit_softcap":
            t = torch.tanh(x_float / value)
            grad_x = grad * (1.0 - t * t)
        else:
            raise KeyError(f"Unsupported CUDA Tile scalar-unary module backward: {op_name}")
        return _tile_kernel_output(grad_x, x.dtype), None, None, None, None


class _TileScalarBinaryModuleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lhs: torch.Tensor, rhs: torch.Tensor, op_name: str, op_code: int, value: float, config: TileCudaConfig):  # type: ignore[override]
        ctx.op_name = op_name
        ctx.value = float(value)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError(f"CUDA Tile extension is unavailable for module '{op_name}'")
            return tile_scalar_binary_module_reference(op_name, lhs, rhs, float(value))
        out = ext.tile_scalar_binary(_tile_kernel_input(lhs), _tile_kernel_input(rhs), float(value), int(op_code))
        return _tile_kernel_output(out, lhs.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        op_name = ctx.op_name
        value = ctx.value
        grad_output = _tile_kernel_input(grad_output)
        if op_name == "aux_loss_add":
            return grad_output, grad_output * value, None, None, None, None
        raise KeyError(f"Unsupported CUDA Tile scalar-binary module backward: {op_name}")


class _TileScalarTernaryModuleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, op_name: str, op_code: int, value: float, config: TileCudaConfig):  # type: ignore[override]
        ctx.op_name = op_name
        ctx.value = float(value)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError(f"CUDA Tile extension is unavailable for module '{op_name}'")
            return tile_scalar_ternary_module_reference(op_name, a, b, c, float(value))
        out = ext.tile_scalar_ternary(
            _tile_kernel_input(a),
            _tile_kernel_input(b),
            _tile_kernel_input(c),
            float(value),
            int(op_code),
        )
        return _tile_kernel_output(out, a.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        op_name = ctx.op_name
        value = ctx.value
        grad_output = _tile_kernel_input(grad_output)
        if op_name == "kl_penalty":
            return -value * grad_output, value * grad_output, grad_output, None, None, None, None
        raise KeyError(f"Unsupported CUDA Tile scalar-ternary module backward: {op_name}")


class _TileVectorBinaryModuleFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        scale0: torch.Tensor,
        scale1: torch.Tensor | None,
        op_name: str,
        op_code: int,
        config: TileCudaConfig,
    ):  # type: ignore[override]
        ctx.op_name = op_name
        empty_scale = torch.empty(0, device=lhs.device, dtype=scale0.dtype)
        ctx.save_for_backward(lhs, rhs, scale0, scale1 if scale1 is not None else empty_scale)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError(f"CUDA Tile extension is unavailable for module '{op_name}'")
            return tile_vector_binary_module_reference(op_name, lhs, rhs, scale0, scale1)
        scale1_arg = scale1 if scale1 is not None else empty_scale
        out = ext.tile_vector_binary(_tile_kernel_input(lhs), _tile_kernel_input(rhs), scale0, scale1_arg, int(op_code))
        return _tile_kernel_output(out, lhs.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        lhs, rhs, scale0, scale1 = ctx.saved_tensors
        op_name = ctx.op_name
        view_shape = (1,) * (lhs.ndim - 1) + (-1,)
        grad = _tile_kernel_input(grad_output)
        lhs_float = _tile_kernel_input(lhs)
        rhs_float = _tile_kernel_input(rhs)
        if op_name == "residual_add":
            s = scale0.reshape(view_shape)
            return (
                _tile_kernel_output(grad, lhs.dtype),
                _tile_kernel_output(grad * s, rhs.dtype),
                _sum_to_last_dim(grad * rhs_float),
                None,
                None,
                None,
                None,
            )
        if op_name == "residual_mix":
            p = scale0.reshape(view_shape)
            s = scale1.reshape(view_shape)
            return (
                _tile_kernel_output(grad * p, lhs.dtype),
                _tile_kernel_output(grad * s, rhs.dtype),
                _sum_to_last_dim(grad * lhs_float),
                _sum_to_last_dim(grad * rhs_float),
                None,
                None,
                None,
            )
        if op_name == "manifold_hyper_connection":
            beta = torch.sigmoid(scale0)
            alpha = torch.sqrt((1.0 - beta * beta).clamp(min=0.0))
            beta_view = beta.reshape(view_shape)
            alpha_view = alpha.reshape(view_shape)
            grad_beta = _sum_to_last_dim(grad * (rhs_float - (beta_view / alpha_view.clamp_min(1e-12)) * lhs_float))
            grad_logit = grad_beta * beta * (1.0 - beta)
            return (
                _tile_kernel_output(grad * alpha_view, lhs.dtype),
                _tile_kernel_output(grad * beta_view, rhs.dtype),
                grad_logit,
                None,
                None,
                None,
                None,
            )
        raise KeyError(f"Unsupported CUDA Tile vector-binary module backward: {op_name}")


class _TileQKGainFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, gain: torch.Tensor, config: TileCudaConfig):  # type: ignore[override]
        ctx.save_for_backward(q, gain)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'qk_gain'")
            return tile_qk_gain_reference(q, gain)
        out = ext.tile_qk_gain(_tile_kernel_input(q), gain)
        return _tile_kernel_output(out, q.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        q, gain = ctx.saved_tensors
        gain_view = gain.reshape((1, -1) + (1,) * (q.ndim - 2))
        grad = _tile_kernel_input(grad_output)
        q_float = _tile_kernel_input(q)
        return _tile_kernel_output(grad * gain_view, q.dtype), _sum_to_head_dim(grad * q_float), None


class _TileDyTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, config: TileCudaConfig):  # type: ignore[override]
        ctx.save_for_backward(x, alpha, weight)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'dyt'")
            return tile_dyt_reference(x, alpha, weight, bias)
        out = ext.tile_dyt(_tile_kernel_input(x), weight, bias, alpha)
        return _tile_kernel_output(out, x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        x, alpha, weight = ctx.saved_tensors
        view_shape = (1,) * (x.ndim - 1) + (-1,)
        grad = _tile_kernel_input(grad_output)
        x_float = _tile_kernel_input(x)
        t = torch.tanh(alpha * x_float)
        dt = 1.0 - t * t
        weight_view = weight.reshape(view_shape)
        grad_x = grad * weight_view * alpha * dt
        grad_alpha = (grad * weight_view * x_float * dt).sum().reshape_as(alpha)
        grad_weight = _sum_to_last_dim(grad * t)
        grad_bias = _sum_to_last_dim(grad)
        return _tile_kernel_output(grad_x, x.dtype), grad_alpha, grad_weight, grad_bias, None


def tile_scalar_unary_module_reference(name: str, x: torch.Tensor, value: float) -> torch.Tensor:
    x_float = _tile_kernel_input(x) if torch.is_floating_point(x) else x.float()
    if name == "loss_scale":
        return _tile_kernel_output(x_float * value, x.dtype)
    if name == "logit_softcap":
        return _tile_kernel_output(value * torch.tanh(x_float / value), x.dtype)
    raise KeyError(f"Unsupported scalar-unary Tile module: {name}")


def tile_scalar_binary_module_reference(name: str, lhs: torch.Tensor, rhs: torch.Tensor, value: float) -> torch.Tensor:
    lhs_float = _tile_kernel_input(lhs)
    rhs_float = _tile_kernel_input(rhs)
    if name == "aux_loss_add":
        return _tile_kernel_output(lhs_float + value * rhs_float, lhs.dtype)
    raise KeyError(f"Unsupported scalar-binary Tile module: {name}")


def tile_scalar_ternary_module_reference(name: str, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, value: float) -> torch.Tensor:
    a_float = _tile_kernel_input(a)
    b_float = _tile_kernel_input(b)
    c_float = _tile_kernel_input(c)
    if name == "kl_penalty":
        return _tile_kernel_output(c_float - value * (a_float - b_float), a.dtype)
    raise KeyError(f"Unsupported scalar-ternary Tile module: {name}")


def tile_vector_binary_module_reference(
    name: str,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    scale0: torch.Tensor,
    scale1: torch.Tensor | None = None,
) -> torch.Tensor:
    view_shape = (1,) * (lhs.ndim - 1) + (-1,)
    lhs_float = _tile_kernel_input(lhs)
    rhs_float = _tile_kernel_input(rhs)
    if name == "residual_add":
        return _tile_kernel_output(lhs_float + scale0.reshape(view_shape) * rhs_float, lhs.dtype)
    if name == "residual_mix":
        if scale1 is None:
            raise ValueError("residual_mix requires scale1")
        return _tile_kernel_output(scale0.reshape(view_shape) * lhs_float + scale1.reshape(view_shape) * rhs_float, lhs.dtype)
    if name == "manifold_hyper_connection":
        beta = torch.sigmoid(scale0)
        alpha = torch.sqrt((1.0 - beta * beta).clamp(min=0.0))
        return _tile_kernel_output(alpha.reshape(view_shape) * lhs_float + beta.reshape(view_shape) * rhs_float, lhs.dtype)
    raise KeyError(f"Unsupported vector-binary Tile module: {name}")


def tile_qk_gain_reference(q: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    return _tile_kernel_output(_tile_kernel_input(q) * gain.reshape((1, -1) + (1,) * (q.ndim - 2)), q.dtype)


def tile_dyt_reference(x: torch.Tensor, alpha: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    view_shape = (1,) * (x.ndim - 1) + (-1,)
    x_float = _tile_kernel_input(x)
    t = torch.tanh(alpha * x_float)
    return _tile_kernel_output(weight.reshape(view_shape) * t + bias.reshape(view_shape), x.dtype)


def tile_scalar_unary_module(name: str, x: torch.Tensor, value: float, config: TileCudaConfig) -> torch.Tensor:
    if not _can_use_tile_unary(x, allow_fp16=True, allow_fp8=True):
        if config.strict and x.is_cuda:
            raise _strict_unary_contract_error("module", name, x, allow_fp16=True, allow_fp8=True)
        return tile_scalar_unary_module_reference(name, x, value)
    return _TileScalarUnaryModuleFunction.apply(x, name, SCALAR_UNARY_MODULE_OPS[name], float(value), config)


def tile_scalar_binary_module(name: str, lhs: torch.Tensor, rhs: torch.Tensor, value: float, config: TileCudaConfig) -> torch.Tensor:
    if not _can_use_tile_binary(lhs, rhs, allow_fp16=True, allow_fp8=True):
        if config.strict and (lhs.is_cuda or rhs.is_cuda):
            raise _strict_binary_contract_error("module", name, lhs, rhs, allow_fp16=True, allow_fp8=True)
        return tile_scalar_binary_module_reference(name, lhs, rhs, value)
    return _TileScalarBinaryModuleFunction.apply(lhs, rhs, name, SCALAR_BINARY_MODULE_OPS[name], float(value), config)


def tile_scalar_ternary_module(name: str, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, value: float, config: TileCudaConfig) -> torch.Tensor:
    if not _can_use_tile_ternary(a, b, c, allow_fp16=True, allow_fp8=True):
        if config.strict and (a.is_cuda or b.is_cuda or c.is_cuda):
            raise _strict_ternary_contract_error("module", name, a, b, c, allow_fp16=True, allow_fp8=True)
        return tile_scalar_ternary_module_reference(name, a, b, c, value)
    return _TileScalarTernaryModuleFunction.apply(a, b, c, name, SCALAR_TERNARY_MODULE_OPS[name], float(value), config)


def tile_vector_binary_module(
    name: str,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    scale0: torch.Tensor,
    scale1: torch.Tensor | None,
    config: TileCudaConfig,
) -> torch.Tensor:
    if not _can_use_tile_vector_binary(lhs, rhs, scale0, scale1, allow_fp16=True, allow_fp8=True):
        if config.strict and (lhs.is_cuda or rhs.is_cuda or scale0.is_cuda or (scale1 is not None and scale1.is_cuda)):
            scale_summary = _tensor_contract_summary("scale0", scale0)
            if scale1 is not None:
                scale_summary = f"{scale_summary}; {_tensor_contract_summary('scale1', scale1)}"
            raise RuntimeError(
                f"CUDA Tile module '{name}' requires same-shape contiguous CUDA activations with matching supported dtypes "
                f"{{{_supported_dtype_names(allow_fp16=True, allow_fp8=True)}}} and float32 contiguous 1D scales matching the last activation dimension; "
                f"got {_tensor_contract_summary('lhs', lhs)}; {_tensor_contract_summary('rhs', rhs)}; {scale_summary}"
            )
        return tile_vector_binary_module_reference(name, lhs, rhs, scale0, scale1)
    return _TileVectorBinaryModuleFunction.apply(lhs, rhs, scale0, scale1, name, VECTOR_BINARY_MODULE_OPS[name], config)


def tile_qk_gain_module(q: torch.Tensor, gain: torch.Tensor, config: TileCudaConfig) -> torch.Tensor:
    can_use = (
        q.is_cuda
        and gain.is_cuda
        and q.dtype in TILE_ELEMENTWISE_INPUT_DTYPES
        and gain.dtype == torch.float32
        and q.is_contiguous()
        and gain.is_contiguous()
        and q.ndim >= 3
        and gain.ndim == 1
        and q.shape[1] == gain.numel()
    )
    if not can_use:
        if config.strict and (q.is_cuda or gain.is_cuda):
            raise RuntimeError(
                f"CUDA Tile module 'qk_gain' requires contiguous CUDA q shaped [B,H,...] with supported dtypes "
                f"{{{_supported_dtype_names(allow_fp16=True, allow_fp8=True)}}} and float32 contiguous gain shaped [H]; "
                f"got {_tensor_contract_summary('q', q)}; {_tensor_contract_summary('gain', gain)}"
            )
        return tile_qk_gain_reference(q, gain)
    return _TileQKGainFunction.apply(q, gain, config)


def tile_dyt_module(x: torch.Tensor, alpha: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, config: TileCudaConfig) -> torch.Tensor:
    can_use = (
        x.is_cuda
        and alpha.is_cuda
        and weight.is_cuda
        and bias.is_cuda
        and x.dtype in TILE_ELEMENTWISE_INPUT_DTYPES
        and alpha.dtype == torch.float32
        and weight.dtype == torch.float32
        and bias.dtype == torch.float32
        and x.is_contiguous()
        and weight.is_contiguous()
        and bias.is_contiguous()
        and alpha.numel() == 1
        and weight.ndim == 1
        and bias.shape == weight.shape
        and x.ndim >= 1
        and x.shape[-1] == weight.numel()
    )
    if not can_use:
        if config.strict and (x.is_cuda or alpha.is_cuda or weight.is_cuda or bias.is_cuda):
            raise RuntimeError(
                f"CUDA Tile module 'dyt' requires contiguous CUDA activations with supported dtypes "
                f"{{{_supported_dtype_names(allow_fp16=True, allow_fp8=True)}}}, scalar float32 alpha, and float32 contiguous last-dim weight/bias; "
                f"got {_tensor_contract_summary('x', x)}; {_tensor_contract_summary('alpha', alpha)}; "
                f"{_tensor_contract_summary('weight', weight)}; {_tensor_contract_summary('bias', bias)}"
            )
        return tile_dyt_reference(x, alpha, weight, bias)
    return _TileDyTFunction.apply(x, alpha, weight, bias, config)


class _TileReshapeHeadsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, num_heads: int, config: TileCudaConfig):  # type: ignore[override]
        ctx.input_shape = tuple(x.shape)
        ctx.num_heads = int(num_heads)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'reshape_heads'")
            return tile_reshape_heads_reference(x, int(num_heads))
        return ext.tile_reshape_heads(x, int(num_heads))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        batch, seq_len, width = ctx.input_shape
        grad = grad_output.transpose(1, 2).contiguous().reshape(batch, seq_len, width)
        return grad, None, None


class _TileMergeHeadsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, config: TileCudaConfig):  # type: ignore[override]
        ctx.input_shape = tuple(x.shape)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'merge_heads'")
            return tile_merge_heads_reference(x)
        return ext.tile_merge_heads(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        batch, heads, seq_len, head_dim = ctx.input_shape
        grad = grad_output.reshape(batch, seq_len, heads, head_dim).transpose(1, 2).contiguous()
        return grad, None


class _TileRepeatKVFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, repeats: int, config: TileCudaConfig):  # type: ignore[override]
        ctx.input_shape = tuple(x.shape)
        ctx.repeats = int(repeats)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'repeat_kv'")
            return tile_repeat_kv_reference(x, int(repeats))
        return ext.tile_repeat_kv(x, int(repeats))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        batch, kv_heads, seq_len, head_dim = ctx.input_shape
        repeats = ctx.repeats
        grad = grad_output.reshape(batch, kv_heads, repeats, seq_len, head_dim).sum(dim=2)
        return grad, None, None


def tile_reshape_heads_reference(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, seq_len, width = x.shape
    if width % num_heads != 0:
        raise ValueError("Last tensor dimension must be divisible by num_heads")
    head_dim = width // num_heads
    return x.reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2)


def tile_merge_heads_reference(x: torch.Tensor) -> torch.Tensor:
    batch, heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().reshape(batch, seq_len, heads * head_dim)


def tile_repeat_kv_reference(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    return x.repeat_interleave(repeats, dim=1)


def tile_broadcast_expert_routes_reference(
    hidden: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden.ndim != 3:
        raise ValueError("broadcast_expert_routes expects hidden shaped [batch, seq, dim]")
    batch, seq_len, _ = hidden.shape

    weights = expert_weights
    indices = expert_indices.long()
    if weights.ndim == 1:
        weights = weights.unsqueeze(0)
    if indices.ndim == 1:
        indices = indices.unsqueeze(0)

    if weights.ndim == 2:
        weights = weights.unsqueeze(1)
    if indices.ndim == 2:
        indices = indices.unsqueeze(1)

    if weights.ndim != 3 or indices.ndim != 3:
        raise ValueError("broadcast_expert_routes expects routes shaped [batch, top_k] or [batch, seq, top_k]")
    if weights.size(0) != batch or indices.size(0) != batch:
        raise ValueError("broadcast_expert_routes batch size must match hidden batch size")

    if weights.size(1) == 1:
        weights = weights.expand(batch, seq_len, weights.size(-1))
    elif weights.size(1) != seq_len:
        raise ValueError("broadcast_expert_routes weights sequence axis must be 1 or match hidden seq_len")

    if indices.size(1) == 1:
        indices = indices.expand(batch, seq_len, indices.size(-1))
    elif indices.size(1) != seq_len:
        raise ValueError("broadcast_expert_routes indices sequence axis must be 1 or match hidden seq_len")

    return weights.to(dtype=hidden.dtype).contiguous(), indices.contiguous()


def tile_broadcast_chunk_routes_reference(
    hidden: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden.ndim != 3:
        raise ValueError("broadcast_chunk_routes expects hidden shaped [batch, seq, dim]")
    batch, seq_len, _ = hidden.shape
    if expert_weights.ndim != 3 or expert_indices.ndim != 3:
        raise ValueError("broadcast_chunk_routes expects routes shaped [batch, chunks, route_width]")
    if expert_weights.size(0) != batch or expert_indices.size(0) != batch:
        raise ValueError("broadcast_chunk_routes batch size must match hidden batch size")
    token_chunks = (
        torch.arange(seq_len, device=hidden.device, dtype=torch.long) // max(int(chunk_size), 1)
    ).clamp_max(expert_weights.size(1) - 1)
    weights = expert_weights[:, token_chunks, :]
    indices = expert_indices.long()[:, token_chunks, :]
    return weights.to(dtype=hidden.dtype).contiguous(), indices.contiguous()


def tile_byte_patch_merge_reference(x: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    target_len = target_tokens.size(1)
    return F.interpolate(x.transpose(1, 2), size=target_len, mode="nearest").transpose(1, 2)


def tile_byte_patch_embed_reference(
    tokens: torch.Tensor,
    embedding_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    patch_size: int,
    stride: int,
    vocab_size: int,
) -> torch.Tensor:
    byte_ids = tokens.clamp(0, int(vocab_size) - 1)
    x = F.embedding(byte_ids, embedding_weight).transpose(1, 2)
    seq_len = x.size(-1)
    if seq_len < int(patch_size):
        pad_right = int(patch_size) - seq_len
    else:
        pad_right = (int(stride) - ((seq_len - int(patch_size)) % int(stride))) % int(stride)
    if pad_right:
        x = F.pad(x, (0, pad_right))
    return F.conv1d(x, proj_weight, bias=None, stride=int(stride)).transpose(1, 2)


def tile_causal_chunk_state_reference(hidden: torch.Tensor, chunk_size: int = 32, mode: str = "prefix") -> torch.Tensor:
    if hidden.ndim != 3:
        raise ValueError("causal_chunk_state expects hidden shaped [batch, seq, dim]")
    batch, seq_len, dim = hidden.shape
    chunks = max(math.ceil(seq_len / max(int(chunk_size), 1)), 1)
    if str(mode) == "mean":
        padded_len = chunks * max(int(chunk_size), 1)
        if padded_len != seq_len:
            pad = hidden.new_zeros(batch, padded_len - seq_len, dim)
            work = torch.cat([hidden, pad], dim=1)
            valid = torch.cat(
                [
                    torch.ones(batch, seq_len, device=hidden.device, dtype=hidden.dtype),
                    torch.zeros(batch, padded_len - seq_len, device=hidden.device, dtype=hidden.dtype),
                ],
                dim=1,
            )
        else:
            work = hidden
            valid = torch.ones(batch, seq_len, device=hidden.device, dtype=hidden.dtype)
        work = work.reshape(batch, chunks, max(int(chunk_size), 1), dim)
        weights = valid.reshape(batch, chunks, max(int(chunk_size), 1)).unsqueeze(-1)
        denom = weights.sum(dim=2).clamp_min(1.0)
        return (work * weights).sum(dim=2) / denom

    cumulative = hidden.cumsum(dim=1)
    boundary_positions = torch.arange(chunks, device=hidden.device, dtype=torch.long) * max(int(chunk_size), 1) - 1
    boundary_positions = boundary_positions.clamp(min=0, max=seq_len - 1)
    gathered = cumulative[:, boundary_positions, :]
    denom = (boundary_positions + 1).to(device=hidden.device, dtype=hidden.dtype).view(1, chunks, 1)
    return gathered / denom


def tile_latent_mse_loss_reference(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred.float(), target.detach().float())


def tile_kv_cache_read_reference(
    k: torch.Tensor,
    v: torch.Tensor,
    cache_k: torch.Tensor | None = None,
    cache_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_k is not None and cache_v is not None:
        k = torch.cat([cache_k, k], dim=2)
        v = torch.cat([cache_v, v], dim=2)
    return k, v


def tile_kv_quant_pack_reference(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    kv = torch.cat([k, v], dim=-1)
    amax = kv.abs().amax(dim=-1, keepdim=True).clamp(min=1e-7)
    scale = amax / 127.0
    quantized = torch.round(kv / scale).clamp(-128, 127)
    return torch.cat([quantized, scale], dim=-1)


def tile_kv_quant_unpack_reference(packed: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    scale = packed[..., -1:]
    quantized = packed[..., :-1]
    dequantized = quantized * scale
    k, v = torch.split(dequantized, [int(head_dim), dequantized.size(-1) - int(head_dim)], dim=-1)
    return k, v


def tile_absolute_position_embedding_reference(weight: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
    pos = torch.arange(seq_len, device=weight.device, dtype=torch.long)
    return weight[pos].unsqueeze(0).expand(batch, -1, -1)


def tile_token_embedding_reference(weight: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    return F.embedding(token_ids.long(), weight)


def tile_rotary_embedding_reference(q: torch.Tensor, k: torch.Tensor, inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = q.size(2)
    freqs = torch.outer(torch.arange(seq_len, device=q.device, dtype=inv_freq.dtype), inv_freq.to(q.device))
    cos = freqs.cos()[None, None, :, :].to(dtype=q.dtype)
    sin = freqs.sin()[None, None, :, :].to(dtype=q.dtype)

    def _apply(x: torch.Tensor) -> torch.Tensor:
        half = x.size(-1) // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

    return _apply(q), _apply(k)


def tile_rms_norm_reference(x: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),), eps=float(eps))


def tile_layer_norm_reference(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    return F.layer_norm(x, (x.size(-1),), weight, bias, eps=float(eps))


def tile_group_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    return F.group_norm(x.transpose(1, 2), int(num_groups), weight, bias, eps=float(eps)).transpose(1, 2)


def tile_linear_reference(x: torch.Tensor | NVFP4Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    x_tensor = _linear_input_tensor(x)
    work_x = x_tensor.to(dtype=torch.float32) if x_tensor.dtype in TILE_FP8_DTYPES else x_tensor
    return F.linear(work_x, weight, bias)


def tile_scaled_residual_add_reference(lhs: torch.Tensor, rhs: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return lhs + scale.reshape(()) * rhs


def tile_act_weighted_sum_reference(states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (states * weights.to(dtype=states.dtype).unsqueeze(-1).unsqueeze(-1)).sum(dim=1)


def tile_latent_pool_reference(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=x.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    pooled = (x * weights).sum(dim=1) / denom
    fallback = x.mean(dim=1)
    has_mask = (mask.sum(dim=1, keepdim=True) > 0).to(dtype=x.dtype)
    return pooled * has_mask + fallback * (1.0 - has_mask)


def tile_token_cross_entropy_reference(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    return F.cross_entropy(flat_logits.float(), target_ids.reshape(-1), reduction="mean")


def tile_masked_token_cross_entropy_reference(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1)).float()
    flat_targets = target_ids.reshape(-1)
    flat_mask = loss_mask.reshape(-1).to(flat_logits.dtype)
    per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none", ignore_index=int(ignore_index))
    denom = flat_mask.sum().clamp(min=1.0)
    return (per_token * flat_mask).sum() / denom


def tile_sequence_logp_reference(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    target_safe = targets.clamp(min=0)
    gathered = log_probs.gather(-1, target_safe.unsqueeze(-1)).squeeze(-1)
    mask = loss_mask.to(gathered.dtype)
    valid = (targets != int(ignore_index)).to(gathered.dtype)
    return (gathered * mask * valid).sum(dim=-1)


def tile_preference_bce_loss_reference(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()


def tile_ppo_clipped_loss_reference(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    value_new: torch.Tensor,
    value_old: torch.Tensor,
    returns: torch.Tensor,
    clip_range: float,
    vf_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ratio = (logp_new.float() - logp_old.float()).exp()
    adv = advantages.float()
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - float(clip_range), 1.0 + float(clip_range)) * adv
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_new_f = value_new.float()
    value_old_f = value_old.float()
    returns_f = returns.float()
    value_clipped = value_old_f + (value_new_f - value_old_f).clamp(-float(clip_range), float(clip_range))
    vf_sq1 = (value_new_f - returns_f) ** 2
    vf_sq2 = (value_clipped - returns_f) ** 2
    value_loss = 0.5 * torch.maximum(vf_sq1, vf_sq2).mean()
    return policy_loss, value_loss, policy_loss + float(vf_coef) * value_loss


def tile_gae_compute_reference(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    next_adv = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)
    next_value = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)
    gamma_f = float(gamma)
    lambda_f = float(lambda_)
    for t in range(seq_len - 1, -1, -1):
        delta = rewards[:, t] + gamma_f * next_value - values[:, t]
        next_adv = delta + gamma_f * lambda_f * next_adv
        advantages[:, t] = next_adv
        next_value = values[:, t]
    return advantages, advantages + values


def tile_route_selection_loss_reference(
    route_logits: torch.Tensor,
    sem_targets: torch.Tensor,
    num_vocab_dims: int,
    shared_experts: int,
    ignore_index: int,
) -> torch.Tensor:
    logits = route_logits.float()
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)
    targets = sem_targets.long()
    if targets.ndim == 1:
        targets = targets.unsqueeze(0)
    if targets.size(1) < int(num_vocab_dims):
        pad = targets.new_full((targets.size(0), int(num_vocab_dims) - targets.size(1)), int(ignore_index))
        targets = torch.cat([targets, pad], dim=1)
    active = targets[:, : int(num_vocab_dims)] != int(ignore_index)
    if not bool(active.any()):
        return logits.sum() * 0.0
    semantic_logits = logits[..., int(shared_experts) : int(shared_experts) + int(num_vocab_dims)]
    target = active.to(dtype=semantic_logits.dtype).unsqueeze(1).expand_as(semantic_logits)
    valid = active.unsqueeze(1).expand_as(semantic_logits)
    losses = F.binary_cross_entropy_with_logits(semantic_logits, target, reduction="none")
    return losses[valid].mean()


def tile_semantic_alignment_loss_reference(
    pred: torch.Tensor,
    target: torch.Tensor,
    term_counts: tuple[int, ...] | list[int],
    ignore_index: int,
) -> torch.Tensor:
    logits = pred.float()
    targets = target.detach().long()
    if logits.ndim == 4:
        batch, chunks, dims, terms = logits.shape
        logits = logits.reshape(batch * chunks, dims, terms)
        if targets.ndim == 1:
            targets = targets.unsqueeze(0)
        targets = targets.unsqueeze(1).expand(batch, chunks, targets.size(-1)).reshape(batch * chunks, targets.size(-1))
    if targets.ndim == 1:
        targets = targets.unsqueeze(0)
    losses: list[torch.Tensor] = []
    n_dims = min(len(term_counts), logits.size(1), targets.size(1))
    for dim_idx in range(n_dims):
        term_count = min(int(term_counts[dim_idx]), logits.size(-1))
        if term_count <= 0:
            continue
        dim_targets = targets[:, dim_idx]
        valid = (dim_targets != int(ignore_index)) & (dim_targets >= 0) & (dim_targets < term_count)
        safe_targets = torch.where(valid, dim_targets, dim_targets.new_full(dim_targets.shape, int(ignore_index)))
        dim_logits = logits[:, dim_idx, :term_count]
        loss_sum = F.cross_entropy(
            dim_logits,
            safe_targets,
            ignore_index=int(ignore_index),
            reduction="sum",
        )
        valid_count = valid.to(dtype=loss_sum.dtype).sum()
        losses.append(torch.where(valid_count > 0, loss_sum / valid_count, loss_sum * 0.0))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def tile_dpo_pairwise_loss_reference(
    policy_logp_chosen: torch.Tensor,
    policy_logp_rejected: torch.Tensor,
    ref_logp_chosen: torch.Tensor,
    ref_logp_rejected: torch.Tensor,
    beta: float,
    label_smoothing: float,
    loss_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chosen_logratios = policy_logp_chosen - ref_logp_chosen
    rejected_logratios = policy_logp_rejected - ref_logp_rejected
    logits = float(beta) * (chosen_logratios - rejected_logratios)
    if loss_type == "hinge":
        per_example = F.relu(1.0 - logits)
    elif loss_type == "ipo":
        per_example = (logits - 1.0 / (2.0 * max(float(beta), 1e-8))) ** 2
    else:
        smoothing = float(label_smoothing)
        if smoothing > 0.0:
            per_example = -F.logsigmoid(logits) * (1.0 - smoothing) - F.logsigmoid(-logits) * smoothing
        else:
            per_example = -F.logsigmoid(logits)
    loss = per_example.mean()
    chosen_reward = (float(beta) * chosen_logratios).detach()
    rejected_reward = (float(beta) * rejected_logratios).detach()
    return loss, chosen_reward, rejected_reward


def tile_route_balance_loss_reference(route_logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(route_logits.float().reshape(-1, route_logits.size(-1)), dim=-1)
    density = probs.mean(dim=0)
    return route_logits.size(-1) * (density * density).sum()


def tile_load_balance_loss_reference(
    router_logits: torch.Tensor,
    routing_weights: torch.Tensor,
    routing_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del routing_weights, routing_indices
    return tile_route_balance_loss_reference(router_logits), router_logits


def tile_softmax_distillation_loss_reference(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    teacher = F.log_softmax(teacher_logits.float().detach(), dim=-1)
    student = F.log_softmax(student_logits.float(), dim=-1)
    return F.kl_div(student, teacher.exp(), reduction="batchmean")


def tile_reshape_heads_module(x: torch.Tensor, num_heads: int, config: TileCudaConfig) -> torch.Tensor:
    can_use = x.is_cuda and x.dtype == torch.float32 and x.is_contiguous() and x.ndim == 3 and num_heads > 0 and x.shape[2] % num_heads == 0
    if not can_use:
        if config.strict and x.is_cuda:
            raise RuntimeError("CUDA Tile module 'reshape_heads' requires contiguous CUDA float32 input shaped [B,S,D]")
        return tile_reshape_heads_reference(x, num_heads)
    return _TileReshapeHeadsFunction.apply(x, int(num_heads), config)


def tile_merge_heads_module(x: torch.Tensor, config: TileCudaConfig) -> torch.Tensor:
    can_use = x.is_cuda and x.dtype == torch.float32 and x.is_contiguous() and x.ndim == 4
    if not can_use:
        if config.strict and x.is_cuda:
            raise RuntimeError("CUDA Tile module 'merge_heads' requires contiguous CUDA float32 input shaped [B,H,S,D]")
        return tile_merge_heads_reference(x)
    return _TileMergeHeadsFunction.apply(x, config)


def tile_repeat_kv_module(x: torch.Tensor, repeats: int, config: TileCudaConfig) -> torch.Tensor:
    can_use = x.is_cuda and x.dtype == torch.float32 and x.is_contiguous() and x.ndim == 4 and repeats >= 1
    if not can_use:
        if config.strict and x.is_cuda:
            raise RuntimeError("CUDA Tile module 'repeat_kv' requires contiguous CUDA float32 input shaped [B,Hkv,S,D]")
        return tile_repeat_kv_reference(x, repeats)
    return _TileRepeatKVFunction.apply(x, int(repeats), config)


def tile_identity_module(name: str, x: torch.Tensor, config: TileCudaConfig) -> torch.Tensor:
    if not _can_use_tile_identity(x):
        if config.strict and x.is_cuda:
            raise RuntimeError(f"CUDA Tile module '{name}' requires contiguous CUDA float32 or float16 input")
        return x
    return _TileUnaryFunction.apply(x, "identity", UNARY_OPS["identity"], config)


def tile_kv_cache_write_module(k: torch.Tensor, v: torch.Tensor, config: TileCudaConfig) -> tuple[torch.Tensor, torch.Tensor]:
    return tile_identity_module("kv_cache_write", k, config), tile_identity_module("kv_cache_write", v, config)


class _TileKVCacheReadFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        config: TileCudaConfig,
    ):  # type: ignore[override]
        ctx.current_seq = int(k.size(2))
        ctx.cache_seq = int(cache_k.size(2))
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'kv_cache_read'")
            return tile_kv_cache_read_reference(k, v, cache_k, cache_v)
        out_k, out_v = ext.tile_kv_cache_read(k, v, cache_k, cache_v)
        return out_k, out_v

    @staticmethod
    def backward(ctx, grad_k_out: torch.Tensor, grad_v_out: torch.Tensor):  # type: ignore[override]
        cache_seq = ctx.cache_seq
        grad_cache_k = grad_k_out[:, :, :cache_seq, :].contiguous()
        grad_k = grad_k_out[:, :, cache_seq:, :].contiguous()
        grad_cache_v = grad_v_out[:, :, :cache_seq, :].contiguous()
        grad_v = grad_v_out[:, :, cache_seq:, :].contiguous()
        return grad_k, grad_v, grad_cache_k, grad_cache_v, None


def tile_kv_cache_read_module(
    k: torch.Tensor,
    v: torch.Tensor,
    cache_k: torch.Tensor | None,
    cache_v: torch.Tensor | None,
    config: TileCudaConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_k is None or cache_v is None:
        return tile_identity_module("kv_cache_read", k, config), tile_identity_module("kv_cache_read", v, config)
    can_use = (
        k.is_cuda
        and v.is_cuda
        and cache_k.is_cuda
        and cache_v.is_cuda
        and k.dtype == torch.float32
        and v.dtype == torch.float32
        and cache_k.dtype == torch.float32
        and cache_v.dtype == torch.float32
        and k.is_contiguous()
        and v.is_contiguous()
        and cache_k.is_contiguous()
        and cache_v.is_contiguous()
        and k.ndim == 4
        and v.shape == k.shape
        and cache_k.ndim == 4
        and cache_v.shape == cache_k.shape
        and cache_k.shape[:2] == k.shape[:2]
        and cache_k.shape[3] == k.shape[3]
    )
    if not can_use:
        if config.strict and (k.is_cuda or v.is_cuda or cache_k.is_cuda or cache_v.is_cuda):
            raise RuntimeError("CUDA Tile module 'kv_cache_read' requires contiguous CUDA float32 K/V/cache tensors shaped [B,H,S,D]")
        return tile_kv_cache_read_reference(k, v, cache_k, cache_v)
    return _TileKVCacheReadFunction.apply(k, v, cache_k, cache_v, config)


class _TileKVQuantPackFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k: torch.Tensor, v: torch.Tensor, config: TileCudaConfig):  # type: ignore[override]
        ctx.save_for_backward(k, v)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'kv_quant_pack'")
            return tile_kv_quant_pack_reference(k, v)
        return ext.tile_kv_quant_pack(k, v)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        k, v = ctx.saved_tensors
        with torch.enable_grad():
            k_ref = k.detach().requires_grad_(True)
            v_ref = v.detach().requires_grad_(True)
            out = tile_kv_quant_pack_reference(k_ref, v_ref)
            grad_k, grad_v = torch.autograd.grad(out, (k_ref, v_ref), grad_output, allow_unused=True)
        return grad_k, grad_v, None


class _TileKVQuantUnpackFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed: torch.Tensor, head_dim: int, config: TileCudaConfig):  # type: ignore[override]
        ctx.head_dim = int(head_dim)
        ctx.save_for_backward(packed)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'kv_quant_unpack'")
            return tile_kv_quant_unpack_reference(packed, int(head_dim))
        out_k, out_v = ext.tile_kv_quant_unpack(packed, int(head_dim))
        return out_k, out_v

    @staticmethod
    def backward(ctx, grad_k: torch.Tensor, grad_v: torch.Tensor):  # type: ignore[override]
        (packed,) = ctx.saved_tensors
        with torch.enable_grad():
            packed_ref = packed.detach().requires_grad_(True)
            out_k, out_v = tile_kv_quant_unpack_reference(packed_ref, ctx.head_dim)
            (grad_packed,) = torch.autograd.grad((out_k, out_v), (packed_ref,), (grad_k, grad_v), allow_unused=True)
        return grad_packed, None, None


def tile_kv_quant_pack_module(k: torch.Tensor, v: torch.Tensor, config: TileCudaConfig) -> torch.Tensor:
    can_use = (
        k.is_cuda
        and v.is_cuda
        and k.dtype == torch.float32
        and v.dtype == torch.float32
        and k.is_contiguous()
        and v.is_contiguous()
        and k.shape == v.shape
        and k.ndim >= 1
        and 0 < k.shape[-1] <= 512
    )
    if not can_use:
        if config.strict and (k.is_cuda or v.is_cuda):
            raise RuntimeError(
                "CUDA Tile module 'kv_quant_pack' requires same-shape contiguous CUDA float32 K/V tensors with head_dim <= 512"
            )
        return tile_kv_quant_pack_reference(k, v)
    return _TileKVQuantPackFunction.apply(k, v, config)


def tile_kv_quant_unpack_module(
    packed: torch.Tensor,
    head_dim: int,
    config: TileCudaConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed_dim = int(head_dim) * 2 + 1
    can_use = (
        packed.is_cuda
        and packed.dtype == torch.float32
        and packed.is_contiguous()
        and packed.ndim >= 1
        and int(head_dim) > 0
        and int(head_dim) <= 512
        and packed.shape[-1] == packed_dim
    )
    if not can_use:
        if config.strict and packed.is_cuda:
            raise RuntimeError(
                "CUDA Tile module 'kv_quant_unpack' requires contiguous CUDA float32 packed tensors shaped [..., 2*head_dim+1]"
            )
        return tile_kv_quant_unpack_reference(packed, int(head_dim))
    return _TileKVQuantUnpackFunction.apply(packed, int(head_dim), config)


class _TileBroadcastExpertRoutesFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights: torch.Tensor, indices: torch.Tensor, seq_len: int, config: TileCudaConfig):  # type: ignore[override]
        ctx.input_shape = tuple(weights.shape)
        ctx.seq_len = int(seq_len)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'broadcast_expert_routes'")
            if weights.size(1) == 1:
                out_weights = weights.expand(weights.size(0), int(seq_len), weights.size(-1))
            else:
                out_weights = weights
            if indices.size(1) == 1:
                out_indices = indices.expand(indices.size(0), int(seq_len), indices.size(-1))
            else:
                out_indices = indices
            return out_weights.contiguous(), out_indices.contiguous()
        out_weights, out_indices = ext.tile_broadcast_expert_routes(weights, indices, int(seq_len))
        return out_weights, out_indices

    @staticmethod
    def backward(ctx, grad_weights: torch.Tensor, grad_indices: torch.Tensor | None):  # type: ignore[override]
        del grad_indices
        _batch, route_seq, _route_width = ctx.input_shape
        if route_seq == 1:
            grad_input = grad_weights.sum(dim=1, keepdim=True)
        else:
            grad_input = grad_weights
        return grad_input, None, None, None


def tile_broadcast_expert_routes_module(
    hidden: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    config: TileCudaConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    ref_weights, ref_indices = tile_broadcast_expert_routes_reference(hidden, expert_weights, expert_indices)
    can_use = (
        ref_weights.is_cuda
        and ref_indices.is_cuda
        and ref_weights.dtype == torch.float32
        and ref_indices.dtype == torch.long
        and ref_weights.is_contiguous()
        and ref_indices.is_contiguous()
        and ref_weights.ndim == 3
        and ref_indices.shape == ref_weights.shape
    )
    if not can_use:
        if config.strict and (ref_weights.is_cuda or ref_indices.is_cuda):
            raise RuntimeError("CUDA Tile module 'broadcast_expert_routes' requires contiguous CUDA float32 weights and int64 indices")
        return ref_weights, ref_indices
    input_route_seq = 1 if expert_weights.ndim <= 2 else int(expert_weights.size(1))
    weights_for_kernel = ref_weights[:, :1, :].contiguous() if input_route_seq == 1 else ref_weights
    indices_for_kernel = ref_indices[:, :1, :].contiguous() if input_route_seq == 1 else ref_indices
    return _TileBroadcastExpertRoutesFunction.apply(weights_for_kernel, indices_for_kernel, int(hidden.size(1)), config)


class _TileBroadcastChunkRoutesFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        weights: torch.Tensor,
        indices: torch.Tensor,
        seq_len: int,
        chunk_size: int,
        config: TileCudaConfig,
    ):  # type: ignore[override]
        ctx.input_shape = tuple(weights.shape)
        ctx.seq_len = int(seq_len)
        ctx.chunk_size = int(chunk_size)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'broadcast_chunk_routes'")
            token_chunks = (
                torch.arange(int(seq_len), device=weights.device, dtype=torch.long) // max(int(chunk_size), 1)
            ).clamp_max(weights.size(1) - 1)
            return weights[:, token_chunks, :].contiguous(), indices[:, token_chunks, :].contiguous()
        out_weights, out_indices = ext.tile_broadcast_chunk_routes(weights, indices, int(seq_len), int(chunk_size))
        return out_weights, out_indices

    @staticmethod
    def backward(ctx, grad_weights: torch.Tensor, grad_indices: torch.Tensor | None):  # type: ignore[override]
        del grad_indices
        batch, chunks, route_width = ctx.input_shape
        token_chunks = (
            torch.arange(ctx.seq_len, device=grad_weights.device, dtype=torch.long) // max(int(ctx.chunk_size), 1)
        ).clamp_max(chunks - 1)
        grad_input = torch.zeros((batch, chunks, route_width), device=grad_weights.device, dtype=grad_weights.dtype)
        for batch_idx in range(batch):
            grad_input[batch_idx].index_add_(0, token_chunks, grad_weights[batch_idx])
        return grad_input, None, None, None, None


def tile_broadcast_chunk_routes_module(
    hidden: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    chunk_size: int,
    config: TileCudaConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    can_use = (
        hidden.is_cuda
        and expert_weights.is_cuda
        and expert_indices.is_cuda
        and expert_weights.dtype == torch.float32
        and expert_indices.dtype == torch.long
        and expert_weights.is_contiguous()
        and expert_indices.is_contiguous()
        and hidden.ndim == 3
        and expert_weights.ndim == 3
        and expert_indices.shape == expert_weights.shape
        and expert_weights.size(0) == hidden.size(0)
    )
    if not can_use:
        if config.strict and (hidden.is_cuda or expert_weights.is_cuda or expert_indices.is_cuda):
            raise RuntimeError("CUDA Tile module 'broadcast_chunk_routes' requires hidden [B,S,D], contiguous CUDA float32 weights, and int64 indices")
        return tile_broadcast_chunk_routes_reference(hidden, expert_weights, expert_indices, chunk_size)
    return _TileBroadcastChunkRoutesFunction.apply(expert_weights, expert_indices.long().contiguous(), int(hidden.size(1)), max(int(chunk_size), 1), config)


class _TileBytePatchEmbedFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tokens: torch.Tensor,
        embedding_weight: torch.Tensor,
        proj_weight: torch.Tensor,
        patch_size: int,
        stride: int,
        vocab_size: int,
        config: TileCudaConfig,
    ):  # type: ignore[override]
        ctx.patch_size = int(patch_size)
        ctx.stride = int(stride)
        ctx.vocab_size = int(vocab_size)
        ctx.save_for_backward(tokens, embedding_weight, proj_weight)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'byte_patch_embed'")
            return tile_byte_patch_embed_reference(tokens, embedding_weight, proj_weight, ctx.patch_size, ctx.stride, ctx.vocab_size)
        return ext.tile_byte_patch_embed(tokens, embedding_weight, proj_weight, ctx.patch_size, ctx.stride)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        tokens, embedding_weight, proj_weight = ctx.saved_tensors
        with torch.enable_grad():
            work_embedding = embedding_weight.detach().requires_grad_(True)
            work_proj = proj_weight.detach().requires_grad_(True)
            out = tile_byte_patch_embed_reference(tokens, work_embedding, work_proj, ctx.patch_size, ctx.stride, ctx.vocab_size)
            grad_embedding, grad_proj = torch.autograd.grad(
                out,
                (work_embedding, work_proj),
                grad_output.to(dtype=out.dtype),
                allow_unused=False,
            )
        return None, grad_embedding.to(dtype=embedding_weight.dtype), grad_proj.to(dtype=proj_weight.dtype), None, None, None, None


def tile_byte_patch_embed_module(
    tokens: torch.Tensor,
    embedding_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    patch_size: int,
    stride: int,
    vocab_size: int,
    config: TileCudaConfig,
) -> torch.Tensor:
    can_use = (
        tokens.is_cuda
        and embedding_weight.is_cuda
        and proj_weight.is_cuda
        and tokens.dtype == torch.long
        and embedding_weight.dtype == torch.float32
        and proj_weight.dtype == torch.float32
        and tokens.is_contiguous()
        and embedding_weight.is_contiguous()
        and proj_weight.is_contiguous()
        and tokens.ndim == 2
        and embedding_weight.ndim == 2
        and proj_weight.ndim == 3
        and int(patch_size) >= 1
        and int(stride) >= 1
        and embedding_weight.size(0) == int(vocab_size)
        and proj_weight.size(0) == embedding_weight.size(1)
        and proj_weight.size(1) == embedding_weight.size(1)
        and proj_weight.size(2) == int(patch_size)
        and tokens.size(1) > 0
    )
    if not can_use:
        if config.strict and (tokens.is_cuda or embedding_weight.is_cuda or proj_weight.is_cuda):
            raise RuntimeError("CUDA Tile module 'byte_patch_embed' requires contiguous CUDA int64 tokens [B,S], float32 embedding [V,D], and projection [D,D,K]")
        return tile_byte_patch_embed_reference(tokens, embedding_weight, proj_weight, patch_size, stride, vocab_size)
    return _TileBytePatchEmbedFunction.apply(tokens, embedding_weight, proj_weight, int(patch_size), int(stride), int(vocab_size), config)


class _TileBytePatchMergeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, target_len: int, config: TileCudaConfig):  # type: ignore[override]
        ctx.input_shape = tuple(x.shape)
        ctx.target_len = int(target_len)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'byte_patch_merge'")
            dummy_target = torch.empty((x.size(0), int(target_len)), device=x.device, dtype=torch.long)
            return tile_byte_patch_merge_reference(x, dummy_target)
        return ext.tile_byte_patch_merge(x, int(target_len))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        batch, source_len, dim = ctx.input_shape
        target_len = ctx.target_len
        src = torch.div(
            torch.arange(target_len, device=grad_output.device, dtype=torch.long) * source_len,
            target_len,
            rounding_mode="floor",
        ).clamp_max(source_len - 1)
        grad_x = torch.zeros((batch, source_len, dim), device=grad_output.device, dtype=grad_output.dtype)
        for batch_idx in range(batch):
            grad_x[batch_idx].index_add_(0, src, grad_output[batch_idx])
        return grad_x, None, None


def tile_byte_patch_merge_module(x: torch.Tensor, target_tokens: torch.Tensor, config: TileCudaConfig) -> torch.Tensor:
    can_use = x.is_cuda and x.dtype == torch.float32 and x.is_contiguous() and x.ndim == 3 and target_tokens.ndim >= 2
    if not can_use:
        if config.strict and x.is_cuda:
            raise RuntimeError("CUDA Tile module 'byte_patch_merge' requires contiguous CUDA float32 input shaped [B,S,D]")
        return tile_byte_patch_merge_reference(x, target_tokens)
    return _TileBytePatchMergeFunction.apply(x, int(target_tokens.size(1)), config)


class _TileCausalChunkStateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden: torch.Tensor, chunk_size: int, mode: str, config: TileCudaConfig):  # type: ignore[override]
        ctx.save_for_backward(hidden)
        ctx.chunk_size = max(int(chunk_size), 1)
        ctx.mode = str(mode)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'causal_chunk_state'")
            return tile_causal_chunk_state_reference(hidden, ctx.chunk_size, ctx.mode)
        return ext.tile_causal_chunk_state(hidden, ctx.chunk_size, ctx.mode)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (hidden,) = ctx.saved_tensors
        with torch.enable_grad():
            work = hidden.detach().requires_grad_(True)
            out = tile_causal_chunk_state_reference(work, ctx.chunk_size, ctx.mode)
            (grad_hidden,) = torch.autograd.grad(out, work, grad_output.to(dtype=out.dtype), allow_unused=False)
        return grad_hidden.to(dtype=hidden.dtype), None, None, None


def tile_causal_chunk_state_module(
    hidden: torch.Tensor,
    chunk_size: int,
    mode: str,
    config: TileCudaConfig,
) -> torch.Tensor:
    can_use = (
        hidden.is_cuda
        and hidden.dtype == torch.float32
        and hidden.is_contiguous()
        and hidden.ndim == 3
        and hidden.size(1) > 0
        and hidden.size(2) > 0
        and str(mode) in {"prefix", "mean"}
    )
    if not can_use:
        if config.strict and hidden.is_cuda:
            raise RuntimeError("CUDA Tile module 'causal_chunk_state' requires contiguous CUDA float32 hidden [B,S,D] and mode prefix/mean")
        return tile_causal_chunk_state_reference(hidden, chunk_size, mode)
    return _TileCausalChunkStateFunction.apply(hidden, max(int(chunk_size), 1), str(mode), config)


class _TileLatentMSELossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred: torch.Tensor, target: torch.Tensor, config: TileCudaConfig):  # type: ignore[override]
        ctx.save_for_backward(pred, target)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'latent_mse_loss'")
            return tile_latent_mse_loss_reference(pred, target)
        return ext.tile_latent_mse_loss(_tile_kernel_input(pred), _tile_kernel_input(target))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        pred, target = ctx.saved_tensors
        pred_float = _tile_kernel_input(pred)
        target_float = _tile_kernel_input(target)
        grad_pred = _tile_kernel_input(grad_output) * (2.0 / pred.numel()) * (pred_float - target_float.detach())
        return _tile_kernel_output(grad_pred, pred.dtype), None, None


def tile_latent_mse_loss_module(pred: torch.Tensor, target: torch.Tensor, config: TileCudaConfig) -> torch.Tensor:
    can_use = (
        pred.is_cuda
        and target.is_cuda
        and pred.dtype in TILE_FLOAT_DTYPES
        and target.dtype == pred.dtype
        and pred.is_contiguous()
        and target.is_contiguous()
        and pred.shape == target.shape
        and pred.numel() > 0
    )
    if not can_use:
        if config.strict and (pred.is_cuda or target.is_cuda):
            raise RuntimeError("CUDA Tile module 'latent_mse_loss' requires same-shape contiguous CUDA float32 or float16 inputs")
        return tile_latent_mse_loss_reference(pred, target)
    return _TileLatentMSELossFunction.apply(pred, target, config)


class _TileAbsolutePositionEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, batch: int, seq_len: int, config: TileCudaConfig):  # type: ignore[override]
        ctx.weight_shape = tuple(weight.shape)
        ctx.batch = int(batch)
        ctx.seq_len = int(seq_len)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'absolute_position_embedding'")
            return tile_absolute_position_embedding_reference(weight, int(batch), int(seq_len))
        return ext.tile_absolute_position_embedding(weight, int(batch), int(seq_len))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        max_seq_len, model_dim = ctx.weight_shape
        seq_len = ctx.seq_len
        grad_weight = torch.zeros((max_seq_len, model_dim), device=grad_output.device, dtype=grad_output.dtype)
        grad_weight[:seq_len] = grad_output.sum(dim=0)
        return grad_weight, None, None, None


def tile_absolute_position_embedding_module(weight: torch.Tensor, batch: int, seq_len: int, config: TileCudaConfig) -> torch.Tensor:
    can_use = (
        weight.is_cuda
        and weight.dtype == torch.float32
        and weight.is_contiguous()
        and weight.ndim == 2
        and batch >= 0
        and seq_len >= 0
        and seq_len <= weight.size(0)
    )
    if not can_use:
        if config.strict and weight.is_cuda:
            raise RuntimeError("CUDA Tile module 'absolute_position_embedding' requires contiguous CUDA float32 embedding weight")
        return tile_absolute_position_embedding_reference(weight, batch, seq_len)
    return _TileAbsolutePositionEmbeddingFunction.apply(weight, int(batch), int(seq_len), config)


class _TileTokenEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, token_ids: torch.Tensor, config: TileCudaConfig):  # type: ignore[override]
        ctx.save_for_backward(token_ids)
        ctx.weight_shape = tuple(weight.shape)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'token_embedding'")
            return tile_token_embedding_reference(weight, token_ids)
        return ext.tile_token_embedding(weight, token_ids.long().contiguous())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (token_ids,) = ctx.saved_tensors
        vocab_size, model_dim = ctx.weight_shape
        grad_weight = torch.zeros((vocab_size, model_dim), device=grad_output.device, dtype=grad_output.dtype)
        grad_weight.index_add_(0, token_ids.reshape(-1).long(), grad_output.reshape(-1, model_dim))
        return grad_weight, None, None


def tile_token_embedding_module(weight: torch.Tensor, token_ids: torch.Tensor, config: TileCudaConfig) -> torch.Tensor:
    token_ids = token_ids.long()
    can_use = (
        weight.is_cuda
        and token_ids.is_cuda
        and weight.dtype == torch.float32
        and token_ids.dtype == torch.long
        and weight.is_contiguous()
        and token_ids.is_contiguous()
        and weight.ndim == 2
    )
    if not can_use:
        if config.strict and (weight.is_cuda or token_ids.is_cuda):
            raise RuntimeError("CUDA Tile module 'token_embedding' requires contiguous CUDA float32 weight and int64 token ids")
        return tile_token_embedding_reference(weight, token_ids)
    return _TileTokenEmbeddingFunction.apply(weight, token_ids, config)


class _TileRotaryEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, inv_freq: torch.Tensor, config: TileCudaConfig):  # type: ignore[override]
        ctx.save_for_backward(inv_freq)
        ctx.q_shape = tuple(q.shape)
        ctx.k_shape = tuple(k.shape)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'rotary_embedding'")
            return tile_rotary_embedding_reference(q, k, inv_freq)
        out_q, out_k = ext.tile_rotary_embedding(_tile_kernel_input(q), _tile_kernel_input(k), inv_freq)
        return _tile_kernel_output(out_q, q.dtype), _tile_kernel_output(out_k, k.dtype)

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_k: torch.Tensor):  # type: ignore[override]
        (inv_freq,) = ctx.saved_tensors

        def _inverse(grad: torch.Tensor) -> torch.Tensor:
            seq_len = grad.size(2)
            half = grad.size(-1) // 2
            freqs = torch.outer(torch.arange(seq_len, device=grad.device, dtype=inv_freq.dtype), inv_freq.to(grad.device))
            cos = freqs.cos()[None, None, :, :].to(dtype=grad.dtype)
            sin = freqs.sin()[None, None, :, :].to(dtype=grad.dtype)
            g1, g2 = grad[..., :half], grad[..., half:]
            return torch.cat((g1 * cos - g2 * sin, g1 * sin + g2 * cos), dim=-1)

        return _inverse(grad_q), _inverse(grad_k), None, None


def tile_rotary_embedding_module(q: torch.Tensor, k: torch.Tensor, inv_freq: torch.Tensor, config: TileCudaConfig) -> tuple[torch.Tensor, torch.Tensor]:
    can_use = (
        q.is_cuda
        and k.is_cuda
        and inv_freq.is_cuda
        and q.dtype in TILE_FLOAT_DTYPES
        and k.dtype == q.dtype
        and inv_freq.dtype == torch.float32
        and q.is_contiguous()
        and k.is_contiguous()
        and inv_freq.is_contiguous()
        and q.ndim == 4
        and k.ndim == 4
        and q.size(0) == k.size(0)
        and q.size(2) == k.size(2)
        and q.size(3) == k.size(3)
        and q.size(3) % 2 == 0
        and inv_freq.ndim == 1
        and inv_freq.numel() == q.size(3) // 2
    )
    if not can_use:
        if config.strict and (q.is_cuda or k.is_cuda or inv_freq.is_cuda):
            raise RuntimeError("CUDA Tile module 'rotary_embedding' requires contiguous CUDA float32 or float16 Q/K shaped [B,H,S,D] and float32 inv_freq [D/2]")
        return tile_rotary_embedding_reference(q, k, inv_freq)
    return _TileRotaryEmbeddingFunction.apply(q, k, inv_freq, config)


class _TileRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float, config: TileCudaConfig):  # type: ignore[override]
        ctx.eps = float(eps)
        ctx.save_for_backward(x)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'rms_norm'")
            return tile_rms_norm_reference(x, float(eps))
        out = ext.tile_rms_norm(_tile_kernel_input(x), float(eps))
        return _tile_kernel_output(out, x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (x,) = ctx.saved_tensors
        eps = ctx.eps
        x_float = _tile_kernel_input(x)
        grad = _tile_kernel_input(grad_output)
        mean_sq = x_float.square().mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(mean_sq + eps)
        dot_mean = (grad * x_float).mean(dim=-1, keepdim=True)
        grad_x = grad * rstd - x_float * (rstd ** 3) * dot_mean
        return _tile_kernel_output(grad_x, x.dtype), None, None


def tile_rms_norm_module(x: torch.Tensor, eps: float, config: TileCudaConfig) -> torch.Tensor:
    can_use = (
        x.is_cuda
        and x.dtype in TILE_FLOAT_DTYPES
        and x.is_contiguous()
        and x.ndim >= 1
        and 0 < x.size(-1) <= 1024
    )
    if not can_use:
        if config.strict and x.is_cuda:
            raise RuntimeError("CUDA Tile module 'rms_norm' requires contiguous CUDA float32 or float16 input with last dim <= 1024")
        return tile_rms_norm_reference(x, eps)
    return _TileRMSNormFunction.apply(x, float(eps), config)


class _TileLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        config: TileCudaConfig,
    ):  # type: ignore[override]
        ctx.eps = float(eps)
        ctx.save_for_backward(x, weight, bias)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'layer_norm'")
            return tile_layer_norm_reference(x, weight, bias, float(eps))
        out = ext.tile_layer_norm(_tile_kernel_input(x), weight, bias, float(eps))
        return _tile_kernel_output(out, x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        x, weight, bias = ctx.saved_tensors
        del bias
        eps = ctx.eps
        reduce_dims = tuple(range(x.ndim - 1))
        mean = x.float().mean(dim=-1, keepdim=True)
        centered = x.float() - mean
        rstd = torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + eps)
        xhat = centered * rstd
        view_shape = (1,) * (x.ndim - 1) + (-1,)
        g = grad_output.float() * weight.reshape(view_shape).float()
        mean_g = g.mean(dim=-1, keepdim=True)
        mean_g_xhat = (g * xhat).mean(dim=-1, keepdim=True)
        grad_x = (rstd * (g - mean_g - xhat * mean_g_xhat)).to(dtype=x.dtype)
        grad_weight = (grad_output.float() * xhat).sum(dim=reduce_dims).to(dtype=weight.dtype)
        grad_bias = grad_output.sum(dim=reduce_dims).to(dtype=weight.dtype)
        return grad_x, grad_weight, grad_bias, None, None


def tile_layer_norm_module(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    config: TileCudaConfig,
) -> torch.Tensor:
    can_use = (
        x.is_cuda
        and weight.is_cuda
        and bias.is_cuda
        and x.dtype in TILE_FLOAT_DTYPES
        and weight.dtype == torch.float32
        and bias.dtype == torch.float32
        and x.is_contiguous()
        and weight.is_contiguous()
        and bias.is_contiguous()
        and x.ndim >= 1
        and 0 < x.size(-1) <= 1024
        and weight.ndim == 1
        and bias.shape == weight.shape
        and weight.numel() == x.size(-1)
    )
    if not can_use:
        if config.strict and (x.is_cuda or weight.is_cuda or bias.is_cuda):
            raise RuntimeError("CUDA Tile module 'layer_norm' requires contiguous CUDA float32 or float16 input and float32 1D affine parameters with last dim <= 1024")
        return tile_layer_norm_reference(x, weight, bias, eps)
    return _TileLayerNormFunction.apply(x, weight, bias, float(eps), config)


class _TileGroupNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        num_groups: int,
        eps: float,
        config: TileCudaConfig,
    ):  # type: ignore[override]
        ctx.num_groups = int(num_groups)
        ctx.eps = float(eps)
        ctx.save_for_backward(x, weight, bias)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'group_norm'")
            return tile_group_norm_reference(x, weight, bias, int(num_groups), float(eps))
        out = ext.tile_group_norm(_tile_kernel_input(x), weight, bias, int(num_groups), float(eps))
        return _tile_kernel_output(out, x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        x, weight, bias = ctx.saved_tensors
        with torch.enable_grad():
            work_x = _tile_kernel_input(x).detach().requires_grad_(True)
            work_weight = weight.detach().requires_grad_(True)
            work_bias = bias.detach().requires_grad_(True)
            out = tile_group_norm_reference(work_x, work_weight, work_bias, ctx.num_groups, ctx.eps)
            grad_x, grad_weight, grad_bias = torch.autograd.grad(
                out,
                (work_x, work_weight, work_bias),
                grad_output.to(dtype=out.dtype),
                allow_unused=False,
            )
        return grad_x.to(dtype=x.dtype), grad_weight.to(dtype=weight.dtype), grad_bias.to(dtype=bias.dtype), None, None, None


def tile_group_norm_module(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
    config: TileCudaConfig,
) -> torch.Tensor:
    groups = max(int(num_groups), 1)
    can_use = (
        x.is_cuda
        and weight.is_cuda
        and bias.is_cuda
        and x.dtype in TILE_FLOAT_DTYPES
        and weight.dtype == torch.float32
        and bias.dtype == torch.float32
        and x.is_contiguous()
        and weight.is_contiguous()
        and bias.is_contiguous()
        and x.ndim == 3
        and x.size(1) > 0
        and x.size(2) > 0
        and x.size(2) % groups == 0
        and x.size(1) * (x.size(2) // groups) <= 1024
        and weight.ndim == 1
        and bias.shape == weight.shape
        and weight.numel() == x.size(2)
    )
    if not can_use:
        if config.strict and (x.is_cuda or weight.is_cuda or bias.is_cuda):
            raise RuntimeError("CUDA Tile module 'group_norm' requires contiguous CUDA float32 or float16 [B,S,D], float32 affine [D], D divisible by groups, and S*group_dim <= 1024")
        return tile_group_norm_reference(x, weight, bias, groups, eps)
    return _TileGroupNormFunction.apply(x, weight, bias, groups, float(eps), config)


def tile_softmax_lastdim_reference(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x.float(), dim=-1).to(dtype=x.dtype)


class _TileSoftmaxLastdimFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, config: TileCudaConfig):  # type: ignore[override]
        ctx.save_for_backward(x)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'solu'")
            return tile_softmax_lastdim_reference(x)
        out = ext.tile_softmax_lastdim(_tile_kernel_input(x))
        return _tile_kernel_output(out, x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (x,) = ctx.saved_tensors
        with torch.enable_grad():
            work_x = _tile_kernel_input(x).detach().requires_grad_(True)
            out = tile_softmax_lastdim_reference(work_x)
            (grad_x,) = torch.autograd.grad(out, (work_x,), grad_output.to(dtype=out.dtype), allow_unused=False)
        return grad_x.to(dtype=x.dtype), None


def tile_softmax_lastdim_module(x: torch.Tensor, config: TileCudaConfig) -> torch.Tensor:
    can_use = (
        x.is_cuda
        and x.dtype in TILE_FLOAT_DTYPES
        and x.is_contiguous()
        and x.ndim >= 1
        and x.size(-1) > 0
        and x.size(-1) <= 1024
    )
    if not can_use:
        if config.strict and x.is_cuda:
            raise RuntimeError("CUDA Tile module 'solu' requires contiguous CUDA float32 or float16 gate logits with last dim <= 1024")
        return tile_softmax_lastdim_reference(x)
    return _TileSoftmaxLastdimFunction.apply(x, config)


def tile_semantic_hash_reference(sem_vec: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    bits = torch.einsum("tpd,bd->btp", proj.to(sem_vec.dtype), sem_vec) > 0
    powers = (2 ** torch.arange(bits.shape[-1], device=bits.device, dtype=torch.long)).unsqueeze(0).unsqueeze(0)
    return (bits.long() * powers).sum(dim=-1)


def tile_semantic_hash_module(sem_vec: torch.Tensor, proj: torch.Tensor, config: TileCudaConfig) -> torch.Tensor:
    can_use = (
        sem_vec.is_cuda
        and proj.is_cuda
        and sem_vec.dtype == torch.float32
        and proj.dtype == torch.float32
        and sem_vec.is_contiguous()
        and proj.is_contiguous()
        and sem_vec.ndim == 2
        and proj.ndim == 3
        and sem_vec.size(0) > 0
        and sem_vec.size(1) > 0
        and proj.size(0) > 0
        and 0 < proj.size(1) <= 62
        and proj.size(2) == sem_vec.size(1)
    )
    if not can_use:
        if config.strict and (sem_vec.is_cuda or proj.is_cuda):
            raise RuntimeError(
                "CUDA Tile module 'semantic_hasher' requires contiguous CUDA float32 semantic vectors [B,D] and projection [tables,planes,D] with planes<=62"
            )
        return tile_semantic_hash_reference(sem_vec, proj)
    ext = load_tile_cuda_extension(config)
    if ext is None:
        if config.strict:
            raise RuntimeError("CUDA Tile extension is unavailable for module 'semantic_hasher'")
        return tile_semantic_hash_reference(sem_vec, proj)
    return ext.tile_semantic_hash(sem_vec, proj)


def tile_topk_route_reference(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    scores = F.softmax(logits.float(), dim=-1)
    weights, indices = torch.topk(scores, int(top_k), dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.to(dtype=logits.dtype), indices


class _TileTopKRouteFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, top_k: int, config: TileCudaConfig):  # type: ignore[override]
        ctx.top_k = int(top_k)
        ctx.save_for_backward(logits)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'topk_route'")
            return tile_topk_route_reference(logits, int(top_k))
        return tuple(ext.tile_topk_route(logits, int(top_k)))

    @staticmethod
    def backward(ctx, grad_weights: torch.Tensor, grad_indices: torch.Tensor | None = None):  # type: ignore[override]
        del grad_indices
        (logits,) = ctx.saved_tensors
        with torch.enable_grad():
            work_logits = logits.detach().requires_grad_(True)
            weights, _ = tile_topk_route_reference(work_logits, ctx.top_k)
            (grad_logits,) = torch.autograd.grad(
                weights,
                (work_logits,),
                grad_weights.to(dtype=weights.dtype),
                allow_unused=False,
            )
        return grad_logits.to(dtype=logits.dtype), None, None


def tile_topk_route_module(logits: torch.Tensor, top_k: int, config: TileCudaConfig) -> tuple[torch.Tensor, torch.Tensor]:
    k = int(top_k)
    can_use = (
        logits.is_cuda
        and logits.dtype == torch.float32
        and logits.is_contiguous()
        and logits.ndim >= 1
        and logits.size(-1) > 0
        and 1 <= k <= logits.size(-1)
        and k <= 64
    )
    if not can_use:
        if config.strict and logits.is_cuda:
            raise RuntimeError("CUDA Tile module 'topk_route' requires contiguous CUDA float32 logits with 1 <= top_k <= experts and top_k <= 64")
        return tile_topk_route_reference(logits, k)
    return _TileTopKRouteFunction.apply(logits, k, config)


def tile_scaled_dot_product_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    window: int | None = None,
    num_sinks: int = 0,
    block_size: int | None = None,
    compress_stride: int | None = None,
    right_align_causal: bool = False,
) -> torch.Tensor:
    input_dtype = q.dtype
    q_work = _tile_kernel_input(q)
    k_work = _tile_kernel_input(k)
    v_work = _tile_kernel_input(v)
    attn_mask = None
    ref_is_causal = bool(is_causal)
    use_sparse = (
        (window is not None and int(window) > 0)
        or int(num_sinks) > 0
        or (block_size is not None and int(block_size) > 0)
        or (compress_stride is not None and int(compress_stride) > 1)
    )
    if use_sparse:
        seq_q, seq_k = q_work.size(-2), k_work.size(-2)
        i = torch.arange(seq_q, device=q_work.device).unsqueeze(1)
        j = torch.arange(seq_k, device=q_work.device).unsqueeze(0)
        offset = seq_k - seq_q if right_align_causal else 0
        causal_ok = (j <= i + offset) if is_causal else torch.ones(seq_q, seq_k, dtype=torch.bool, device=q_work.device)
        keep = torch.zeros(seq_q, seq_k, dtype=torch.bool, device=q_work.device)
        any_rule = False
        if window is not None and int(window) > 0:
            keep = keep | (j > (i + offset) - int(window))
            any_rule = True
        if int(num_sinks) > 0:
            keep = keep | (j < int(num_sinks))
            any_rule = True
        if block_size is not None and int(block_size) > 0:
            keep = keep | ((i + offset) // int(block_size) == j // int(block_size))
            any_rule = True
        if compress_stride is not None and int(compress_stride) > 1:
            keep = keep | (j % int(compress_stride) == 0)
            any_rule = True
        if not any_rule:
            keep = torch.ones(seq_q, seq_k, dtype=torch.bool, device=q.device)
        attn_mask = torch.zeros(seq_q, seq_k, dtype=q_work.dtype, device=q_work.device).masked_fill(~(causal_ok & keep), float("-inf"))
        ref_is_causal = False
    out = F.scaled_dot_product_attention(
        q_work,
        k_work,
        v_work,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=ref_is_causal,
        enable_gqa=q_work.size(1) != k_work.size(1),
    )
    return _tile_attention_output(out, input_dtype)


class _TileScaledDotProductAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
        right_align_causal: bool,
        use_sparse_rules: bool,
        window: int,
        num_sinks: int,
        block_size: int,
        compress_stride: int,
        config: TileCudaConfig,
    ):
        ctx.is_causal = bool(is_causal)
        ctx.right_align_causal = bool(right_align_causal)
        ctx.use_sparse_rules = bool(use_sparse_rules)
        ctx.window = int(window)
        ctx.num_sinks = int(num_sinks)
        ctx.block_size = int(block_size)
        ctx.compress_stride = int(compress_stride)
        ctx.save_for_backward(q, k, v)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'scaled_dot_product_attention'")
            return tile_scaled_dot_product_attention_reference(
                q,
                k,
                v,
                is_causal=ctx.is_causal,
                window=ctx.window if ctx.use_sparse_rules and ctx.window > 0 else None,
                num_sinks=ctx.num_sinks,
                block_size=ctx.block_size if ctx.use_sparse_rules and ctx.block_size > 0 else None,
                compress_stride=ctx.compress_stride if ctx.use_sparse_rules and ctx.compress_stride > 1 else None,
                right_align_causal=ctx.right_align_causal,
            )
        out = ext.tile_scaled_dot_product_attention(
            _tile_kernel_input(q),
            _tile_kernel_input(k),
            _tile_kernel_input(v),
            ctx.is_causal,
            ctx.right_align_causal,
            ctx.use_sparse_rules,
            ctx.window,
            ctx.num_sinks,
            ctx.block_size,
            ctx.compress_stride,
        )
        return _tile_attention_output(out, q.dtype)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        q, k, v = ctx.saved_tensors
        with torch.enable_grad():
            work_q = _tile_kernel_input(q).detach().requires_grad_(q.requires_grad)
            work_k = _tile_kernel_input(k).detach().requires_grad_(k.requires_grad)
            work_v = _tile_kernel_input(v).detach().requires_grad_(v.requires_grad)
            out = tile_scaled_dot_product_attention_reference(
                work_q,
                work_k,
                work_v,
                is_causal=ctx.is_causal,
                window=ctx.window if ctx.use_sparse_rules and ctx.window > 0 else None,
                num_sinks=ctx.num_sinks,
                block_size=ctx.block_size if ctx.use_sparse_rules and ctx.block_size > 0 else None,
                compress_stride=ctx.compress_stride if ctx.use_sparse_rules and ctx.compress_stride > 1 else None,
                right_align_causal=ctx.right_align_causal,
            )
            grads = torch.autograd.grad(
                out,
                (work_q, work_k, work_v),
                grad.to(dtype=out.dtype),
                allow_unused=True,
            )
        grad_q, grad_k, grad_v = grads
        return (
            None if grad_q is None else _tile_kernel_output(grad_q, q.dtype),
            None if grad_k is None else _tile_kernel_output(grad_k, k.dtype),
            None if grad_v is None else _tile_kernel_output(grad_v, v.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def tile_scaled_dot_product_attention_module(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    config: TileCudaConfig,
    *,
    right_align_causal: bool = False,
    window: int | None = None,
    num_sinks: int = 0,
    block_size: int | None = None,
    compress_stride: int | None = None,
) -> torch.Tensor:
    use_sparse_rules = (
        (window is not None and int(window) > 0)
        or int(num_sinks) > 0
        or (block_size is not None and int(block_size) > 0)
        or (compress_stride is not None and int(compress_stride) > 1)
    )
    can_use = (
        q.is_cuda
        and k.is_cuda
        and v.is_cuda
        and q.dtype in TILE_ELEMENTWISE_INPUT_DTYPES
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and q.ndim == 4
        and k.ndim == 4
        and v.ndim == 4
        and q.size(0) == k.size(0)
        and q.size(0) == v.size(0)
        and k.size(1) == v.size(1)
        and q.size(1) % k.size(1) == 0
        and k.size(2) == v.size(2)
        and q.size(3) == k.size(3)
        and 0 < k.size(2) <= 1024
    )
    if not can_use:
        if config.strict and (q.is_cuda or k.is_cuda or v.is_cuda):
            raise RuntimeError(
                "CUDA Tile module 'scaled_dot_product_attention' requires contiguous CUDA q/k/v [B,H,S,D] with supported dtypes {float32, float16, float8_e4m3fn, float8_e5m2} and key sequence <= 1024"
            )
        return tile_scaled_dot_product_attention_reference(
            q,
            k,
            v,
            is_causal=bool(is_causal),
            window=window,
            num_sinks=int(num_sinks),
            block_size=block_size,
            compress_stride=compress_stride,
            right_align_causal=bool(right_align_causal),
        )
    return _TileScaledDotProductAttentionFunction.apply(
        q,
        k,
        v,
        bool(is_causal),
        bool(right_align_causal),
        bool(use_sparse_rules),
        int(window or 0),
        int(num_sinks),
        int(block_size or 0),
        int(compress_stride or 0),
        config,
    )


def tile_attentionless_decoder_reference(
    bucket_indices: torch.Tensor,
    expert_output: torch.Tensor,
    bucket_embed_weight: torch.Tensor,
    out_weight: torch.Tensor,
) -> torch.Tensor:
    n_buckets = bucket_embed_weight.size(0)
    if bucket_indices.ndim == 2:
        primary_bucket = bucket_indices[:, 0] % n_buckets
    else:
        primary_bucket = bucket_indices % n_buckets
    target_dtype = expert_output.dtype if torch.is_floating_point(expert_output) else bucket_embed_weight.dtype
    bucket_bias = F.embedding(primary_bucket.long(), bucket_embed_weight).to(dtype=target_dtype)
    if expert_output.ndim == 3:
        expert_output = expert_output.squeeze(1)
    combined = expert_output + bucket_bias
    return F.linear(combined, out_weight).unsqueeze(1)


class _TileAttentionlessDecoderFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        bucket_indices: torch.Tensor,
        expert_output: torch.Tensor,
        bucket_embed_weight: torch.Tensor,
        out_weight: torch.Tensor,
        config: TileCudaConfig,
    ):
        ctx.save_for_backward(bucket_indices, expert_output, bucket_embed_weight, out_weight)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'attentionless_decoder'")
            return tile_attentionless_decoder_reference(bucket_indices, expert_output, bucket_embed_weight, out_weight)
        primary_bucket = bucket_indices[:, 0].contiguous() if bucket_indices.ndim == 2 else bucket_indices.contiguous()
        expert_flat = expert_output.squeeze(1).contiguous() if expert_output.ndim == 3 else expert_output
        return ext.tile_attentionless_decoder(primary_bucket, expert_flat, bucket_embed_weight, out_weight)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        bucket_indices, expert_output, bucket_embed_weight, out_weight = ctx.saved_tensors
        with torch.enable_grad():
            work_expert = expert_output.detach().requires_grad_(True)
            work_embed = bucket_embed_weight.detach().requires_grad_(True)
            work_weight = out_weight.detach().requires_grad_(True)
            out = tile_attentionless_decoder_reference(bucket_indices, work_expert, work_embed, work_weight)
            grad_expert, grad_embed, grad_weight = torch.autograd.grad(
                out,
                (work_expert, work_embed, work_weight),
                grad_output.to(dtype=out.dtype),
                allow_unused=False,
            )
        return None, grad_expert.to(dtype=expert_output.dtype), grad_embed.to(dtype=bucket_embed_weight.dtype), grad_weight.to(dtype=out_weight.dtype), None


def tile_attentionless_decoder_module(
    bucket_indices: torch.Tensor,
    expert_output: torch.Tensor,
    bucket_embed_weight: torch.Tensor,
    out_weight: torch.Tensor,
    config: TileCudaConfig,
) -> torch.Tensor:
    expert_flat = expert_output.squeeze(1).contiguous() if expert_output.ndim == 3 else expert_output
    primary_bucket = bucket_indices[:, 0].contiguous() if bucket_indices.ndim == 2 else bucket_indices
    can_use = (
        primary_bucket.is_cuda
        and expert_flat.is_cuda
        and bucket_embed_weight.is_cuda
        and out_weight.is_cuda
        and primary_bucket.dtype == torch.long
        and expert_flat.dtype == torch.float32
        and bucket_embed_weight.dtype == torch.float32
        and out_weight.dtype == torch.float32
        and primary_bucket.is_contiguous()
        and expert_flat.is_contiguous()
        and bucket_embed_weight.is_contiguous()
        and out_weight.is_contiguous()
        and primary_bucket.ndim == 1
        and expert_flat.ndim == 2
        and bucket_embed_weight.ndim == 2
        and out_weight.ndim == 2
        and primary_bucket.size(0) == expert_flat.size(0)
        and bucket_embed_weight.size(0) > 0
        and bucket_embed_weight.size(1) > 0
        and expert_flat.size(1) == bucket_embed_weight.size(1)
        and out_weight.size(1) == bucket_embed_weight.size(1)
        and out_weight.size(0) > 0
    )
    if not can_use:
        if config.strict and (bucket_indices.is_cuda or expert_output.is_cuda or bucket_embed_weight.is_cuda or out_weight.is_cuda):
            raise RuntimeError(
                "CUDA Tile module 'attentionless_decoder' requires contiguous CUDA int64 buckets plus float32 expert output [B,R], embedding [N,R], and output weight [V,R]"
            )
        return tile_attentionless_decoder_reference(bucket_indices, expert_output, bucket_embed_weight, out_weight)
    return _TileAttentionlessDecoderFunction.apply(primary_bucket, expert_flat, bucket_embed_weight, out_weight, config)


def tile_expert_bias_add_reference(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return logits + bias.to(dtype=logits.dtype)


class _TileExpertBiasAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, bias: torch.Tensor, config: TileCudaConfig):  # type: ignore[override]
        ctx.save_for_backward(logits)
        ctx.bias_dtype = bias.dtype
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'auxfree_load_balancing'")
            return tile_expert_bias_add_reference(logits, bias)
        return ext.tile_expert_bias_add(logits, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (logits,) = ctx.saved_tensors
        reduce_dims = tuple(range(grad_output.ndim - 1))
        grad_logits = grad_output.to(dtype=logits.dtype)
        grad_bias = grad_output.sum(dim=reduce_dims).to(dtype=ctx.bias_dtype)
        return grad_logits, grad_bias, None


def tile_expert_bias_add_module(logits: torch.Tensor, bias: torch.Tensor, config: TileCudaConfig) -> torch.Tensor:
    can_use = (
        logits.is_cuda
        and bias.is_cuda
        and logits.dtype == torch.float32
        and bias.dtype == torch.float32
        and logits.is_contiguous()
        and bias.is_contiguous()
        and logits.ndim >= 1
        and bias.ndim == 1
        and logits.size(-1) == bias.numel()
        and bias.numel() > 0
    )
    if not can_use:
        if config.strict and (logits.is_cuda or bias.is_cuda):
            raise RuntimeError("CUDA Tile module 'auxfree_load_balancing' requires contiguous CUDA float32 logits [...,E] and bias [E]")
        return tile_expert_bias_add_reference(logits, bias)
    return _TileExpertBiasAddFunction.apply(logits, bias, config)


class _TileLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        config: TileCudaConfig,
    ):  # type: ignore[override]
        has_bias = bias is not None
        bias_for_save = bias if bias is not None else torch.empty(0, device=x.device, dtype=weight.dtype)
        ctx.has_bias = has_bias
        ctx.input_shape = tuple(x.shape)
        ctx.save_for_backward(x, weight, bias_for_save)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'linear'")
            return tile_linear_reference(x, weight, bias)
        out = ext.tile_linear(_tile_kernel_input(x), weight, bias_for_save, bool(has_bias))
        return _tile_linear_output(out, x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        x, weight, _bias = ctx.saved_tensors
        grad = _tile_kernel_input(grad_output)
        x_float = _tile_kernel_input(x)
        grad_flat = grad.reshape(-1, grad.size(-1))
        x_flat = x_float.reshape(-1, x_float.size(-1))
        grad_x = (grad_flat @ weight).reshape(ctx.input_shape)
        grad_weight = grad_flat.t() @ x_flat
        grad_bias = grad_flat.sum(dim=0) if ctx.has_bias else None
        return _tile_kernel_output(grad_x, x.dtype), grad_weight, grad_bias, None


def tile_linear_module(
    x: torch.Tensor | NVFP4Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    config: TileCudaConfig,
    *,
    name: str = "linear",
) -> torch.Tensor:
    x_tensor = _linear_input_tensor(x)
    is_nvfp4 = isinstance(x, NVFP4Tensor)
    can_use = (
        x_tensor.is_cuda
        and weight.is_cuda
        and (is_nvfp4 or x_tensor.dtype in TILE_LINEAR_INPUT_DTYPES)
        and weight.dtype == torch.float32
        and x_tensor.is_contiguous()
        and weight.is_contiguous()
        and x_tensor.ndim >= 1
        and weight.ndim == 2
        and x_tensor.size(-1) == weight.size(1)
        and (bias is None or (bias.is_cuda and bias.dtype == torch.float32 and bias.is_contiguous() and bias.ndim == 1 and bias.numel() == weight.size(0)))
    )
    if not can_use:
        x_is_cuda = x.packed.is_cuda if is_nvfp4 else x_tensor.is_cuda
        if config.strict and (x_is_cuda or weight.is_cuda or (bias is not None and bias.is_cuda)):
            raise RuntimeError(
                f"CUDA Tile module '{name}' requires contiguous CUDA input with supported dtypes "
                "{float32, float16, float8_e4m3fn, float8_e5m2, nvfp4}, float32 weight, and optional float32 bias; "
                f"got {_linear_contract_summary('x', x)}; {_tensor_contract_summary('weight', weight)}"
                + (f"; {_tensor_contract_summary('bias', bias)}" if bias is not None else "")
            )
        return tile_linear_reference(x, weight, bias)
    return _TileLinearFunction.apply(x_tensor, weight, bias, config)


class _TileScaledResidualAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        scale: torch.Tensor,
        config: TileCudaConfig,
    ):  # type: ignore[override]
        ctx.save_for_backward(rhs, scale)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'scaled_residual_add'")
            return tile_scaled_residual_add_reference(lhs, rhs, scale)
        out = ext.tile_scaled_residual_add(_tile_kernel_input(lhs), _tile_kernel_input(rhs), scale)
        return _tile_kernel_output(out, lhs.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        rhs, scale = ctx.saved_tensors
        grad = _tile_kernel_input(grad_output)
        rhs_float = _tile_kernel_input(rhs)
        grad_lhs = grad
        grad_rhs = grad * scale.reshape(())
        grad_scale = (grad * rhs_float).sum().reshape_as(scale)
        return _tile_kernel_output(grad_lhs, rhs.dtype), _tile_kernel_output(grad_rhs, rhs.dtype), grad_scale.to(dtype=scale.dtype), None


def tile_scaled_residual_add_module(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    scale: torch.Tensor,
    config: TileCudaConfig,
) -> torch.Tensor:
    can_use = (
        lhs.is_cuda
        and rhs.is_cuda
        and scale.is_cuda
        and lhs.dtype in TILE_FLOAT_DTYPES
        and rhs.dtype == lhs.dtype
        and scale.dtype == torch.float32
        and lhs.is_contiguous()
        and rhs.is_contiguous()
        and scale.is_contiguous()
        and lhs.shape == rhs.shape
        and scale.numel() == 1
    )
    if not can_use:
        if config.strict and (lhs.is_cuda or rhs.is_cuda or scale.is_cuda):
            raise RuntimeError("CUDA Tile module 'scaled_residual_add' requires same-shape contiguous CUDA float32 or float16 inputs and scalar float32 scale")
        return tile_scaled_residual_add_reference(lhs, rhs, scale)
    return _TileScaledResidualAddFunction.apply(lhs, rhs, scale, config)


class _TileACTWeightedSumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, states: torch.Tensor, weights: torch.Tensor, config: TileCudaConfig):  # type: ignore[override]
        ctx.save_for_backward(states, weights)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'act_weighted_sum'")
            return tile_act_weighted_sum_reference(states, weights)
        out = ext.tile_act_weighted_sum(_tile_kernel_input(states), weights)
        return _tile_kernel_output(out, states.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        states, weights = ctx.saved_tensors
        grad = _tile_kernel_input(grad_output)
        states_float = _tile_kernel_input(states)
        grad_states = grad.unsqueeze(1) * weights.unsqueeze(-1).unsqueeze(-1)
        reduce_dims = tuple(range(2, states.ndim))
        grad_weights = (grad.unsqueeze(1) * states_float).sum(dim=reduce_dims).to(dtype=weights.dtype)
        return _tile_kernel_output(grad_states, states.dtype), grad_weights, None


def tile_act_weighted_sum_module(states: torch.Tensor, weights: torch.Tensor, config: TileCudaConfig) -> torch.Tensor:
    can_use = (
        states.is_cuda
        and weights.is_cuda
        and states.dtype in TILE_FLOAT_DTYPES
        and weights.dtype == torch.float32
        and states.is_contiguous()
        and weights.is_contiguous()
        and states.ndim >= 3
        and weights.ndim == 2
        and states.size(0) == weights.size(0)
        and states.size(1) == weights.size(1)
    )
    if not can_use:
        if config.strict and (states.is_cuda or weights.is_cuda):
            raise RuntimeError("CUDA Tile module 'act_weighted_sum' requires contiguous CUDA float32 or float16 states [B,steps,...] and float32 weights [B,steps]")
        return tile_act_weighted_sum_reference(states, weights)
    return _TileACTWeightedSumFunction.apply(states, weights, config)


class _TileLatentPoolFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, mask: torch.Tensor, config: TileCudaConfig):  # type: ignore[override]
        ctx.save_for_backward(x, mask)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'latent_pool'")
            return tile_latent_pool_reference(x, mask)
        out = ext.tile_latent_pool(_tile_kernel_input(x), mask)
        return _tile_kernel_output(out, x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        x, mask = ctx.saved_tensors
        x_float = _tile_kernel_input(x)
        weights = mask.to(dtype=torch.float32)
        denom_raw = weights.sum(dim=1, keepdim=True)
        has_mask = (denom_raw > 0).to(dtype=torch.float32)
        denom = denom_raw.clamp_min(1.0)
        sum_wx = (x_float * weights.unsqueeze(-1)).sum(dim=1)
        seq_len = x.size(1)
        grad = _tile_kernel_input(grad_output)
        grad_x_masked = grad.unsqueeze(1) * weights.unsqueeze(-1) / denom.unsqueeze(-1)
        grad_x_fallback = grad.unsqueeze(1).expand_as(x_float) / float(seq_len)
        grad_x = grad_x_masked * has_mask.unsqueeze(-1) + grad_x_fallback * (1.0 - has_mask.unsqueeze(-1))
        grad_mask = ((grad.unsqueeze(1) * (x_float * denom.unsqueeze(-1) - sum_wx.unsqueeze(1))).sum(dim=-1) / (denom * denom))
        grad_mask = grad_mask * has_mask
        return _tile_kernel_output(grad_x, x.dtype), grad_mask.to(dtype=mask.dtype), None


def tile_latent_pool_module(x: torch.Tensor, mask: torch.Tensor, config: TileCudaConfig) -> torch.Tensor:
    can_use = (
        x.is_cuda
        and mask.is_cuda
        and x.dtype in TILE_FLOAT_DTYPES
        and mask.dtype == torch.float32
        and x.is_contiguous()
        and mask.is_contiguous()
        and x.ndim == 3
        and mask.ndim == 2
        and x.size(0) == mask.size(0)
        and x.size(1) == mask.size(1)
        and x.numel() > 0
    )
    if not can_use:
        if config.strict and (x.is_cuda or mask.is_cuda):
            raise RuntimeError("CUDA Tile module 'latent_pool' requires contiguous CUDA float32 or float16 x [B,S,D] and float32 mask [B,S]")
        return tile_latent_pool_reference(x, mask)
    return _TileLatentPoolFunction.apply(x, mask, config)


class _TileTokenCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, target_ids: torch.Tensor, config: TileCudaConfig):  # type: ignore[override]
        ctx.save_for_backward(logits, target_ids)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'token_cross_entropy'")
            return tile_token_cross_entropy_reference(logits, target_ids)
        return ext.tile_token_cross_entropy(_tile_kernel_input(logits), target_ids)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        logits, target_ids = ctx.saved_tensors
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = target_ids.reshape(-1).long()
        grad = F.softmax(flat_logits.float(), dim=-1)
        grad.scatter_add_(1, flat_targets.unsqueeze(1), torch.full_like(flat_targets.unsqueeze(1), -1.0, dtype=grad.dtype))
        grad = grad / max(flat_targets.numel(), 1)
        return grad_output.to(dtype=logits.dtype) * grad.reshape_as(logits).to(dtype=logits.dtype), None, None


def tile_token_cross_entropy_module(logits: torch.Tensor, target_ids: torch.Tensor, config: TileCudaConfig) -> torch.Tensor:
    can_use = (
        logits.is_cuda
        and target_ids.is_cuda
        and logits.dtype in TILE_FLOAT_DTYPES
        and target_ids.dtype == torch.long
        and logits.is_contiguous()
        and target_ids.is_contiguous()
        and logits.ndim >= 2
        and target_ids.shape == logits.shape[:-1]
        and logits.numel() > 0
    )
    if not can_use:
        if config.strict and (logits.is_cuda or target_ids.is_cuda):
            raise RuntimeError("CUDA Tile module 'token_cross_entropy' requires contiguous CUDA float32 or float16 logits and int64 targets")
        return tile_token_cross_entropy_reference(logits, target_ids)
    return _TileTokenCrossEntropyFunction.apply(logits, target_ids, config)


class _TileMaskedTokenCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        ignore_index: int,
        config: TileCudaConfig,
    ):
        ctx.save_for_backward(logits, target_ids, loss_mask)
        ctx.ignore_index = int(ignore_index)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'masked_token_cross_entropy'")
            return tile_masked_token_cross_entropy_reference(logits, target_ids, loss_mask, ignore_index)
        return ext.tile_masked_token_cross_entropy(_tile_kernel_input(logits), target_ids, loss_mask, int(ignore_index))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        logits, target_ids, loss_mask = ctx.saved_tensors
        ignore_index = int(ctx.ignore_index)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = target_ids.reshape(-1).long()
        flat_mask = loss_mask.reshape(-1).to(dtype=flat_logits.dtype)
        valid = flat_targets != ignore_index
        safe_targets = flat_targets.clamp(min=0)
        softmax = F.softmax(flat_logits.float(), dim=-1)
        grad_logits = softmax.clone()
        grad_logits.scatter_add_(1, safe_targets.unsqueeze(1), torch.full_like(safe_targets.unsqueeze(1), -1.0, dtype=grad_logits.dtype))
        grad_logits = grad_logits * valid.to(dtype=grad_logits.dtype).unsqueeze(1)
        per_token = F.cross_entropy(flat_logits.float(), flat_targets, reduction="none", ignore_index=ignore_index)
        denom_raw = flat_mask.sum()
        denom = denom_raw.clamp(min=1.0)
        numerator = (per_token * flat_mask).sum()
        grad_logits = grad_logits * (flat_mask / denom).unsqueeze(1)
        if denom_raw >= 1.0:
            grad_mask = (per_token * denom - numerator) / (denom * denom)
        else:
            grad_mask = per_token
        grad = grad_output.to(dtype=logits.dtype)
        return (
            grad * grad_logits.reshape_as(logits).to(dtype=logits.dtype),
            None,
            grad.to(dtype=loss_mask.dtype) * grad_mask.reshape_as(loss_mask).to(dtype=loss_mask.dtype),
            None,
            None,
        )


def tile_masked_token_cross_entropy_module(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    ignore_index: int,
    config: TileCudaConfig,
) -> torch.Tensor:
    can_use = (
        logits.is_cuda
        and target_ids.is_cuda
        and loss_mask.is_cuda
        and logits.dtype in TILE_FLOAT_DTYPES
        and target_ids.dtype == torch.long
        and loss_mask.dtype == torch.float32
        and logits.is_contiguous()
        and target_ids.is_contiguous()
        and loss_mask.is_contiguous()
        and logits.ndim >= 2
        and target_ids.shape == logits.shape[:-1]
        and loss_mask.shape == target_ids.shape
        and logits.numel() > 0
    )
    if not can_use:
        if config.strict and (logits.is_cuda or target_ids.is_cuda or loss_mask.is_cuda):
            raise RuntimeError(
                "CUDA Tile module 'masked_token_cross_entropy' requires contiguous CUDA float32 or float16 logits, int64 targets, and float32 mask"
            )
        return tile_masked_token_cross_entropy_reference(logits, target_ids, loss_mask, ignore_index)
    return _TileMaskedTokenCrossEntropyFunction.apply(logits, target_ids, loss_mask, int(ignore_index), config)


class _TileSequenceLogpFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
        ignore_index: int,
        config: TileCudaConfig,
    ):
        ctx.save_for_backward(logits, targets, loss_mask)
        ctx.ignore_index = int(ignore_index)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'sequence_logp'")
            return tile_sequence_logp_reference(logits, targets, loss_mask, ignore_index)
        return ext.tile_sequence_logp(_tile_kernel_input(logits), targets, loss_mask, int(ignore_index))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        logits, targets, loss_mask = ctx.saved_tensors
        ignore_index = int(ctx.ignore_index)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        safe_targets = targets.clamp(min=0).long()
        gathered = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        effective = loss_mask.to(dtype=log_probs.dtype) * (targets != ignore_index).to(dtype=log_probs.dtype)
        per_row_grad = grad_output.to(dtype=log_probs.dtype).unsqueeze(-1)
        grad_logits = -probs * effective.unsqueeze(-1) * per_row_grad.unsqueeze(-1)
        grad_logits.scatter_add_(
            -1,
            safe_targets.unsqueeze(-1),
            (effective * per_row_grad).unsqueeze(-1),
        )
        grad_mask = gathered * (targets != ignore_index).to(dtype=gathered.dtype) * per_row_grad
        return grad_logits.to(dtype=logits.dtype), None, grad_mask.to(dtype=loss_mask.dtype), None, None


def tile_sequence_logp_module(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    ignore_index: int,
    config: TileCudaConfig,
) -> torch.Tensor:
    can_use = (
        logits.is_cuda
        and targets.is_cuda
        and loss_mask.is_cuda
        and logits.dtype in TILE_FLOAT_DTYPES
        and targets.dtype == torch.long
        and loss_mask.dtype == torch.float32
        and logits.is_contiguous()
        and targets.is_contiguous()
        and loss_mask.is_contiguous()
        and logits.ndim == 3
        and targets.shape == logits.shape[:2]
        and loss_mask.shape == targets.shape
        and logits.numel() > 0
        and logits.size(0) <= 1024
    )
    if not can_use:
        if config.strict and (logits.is_cuda or targets.is_cuda or loss_mask.is_cuda):
            raise RuntimeError(
                "CUDA Tile module 'sequence_logp' requires contiguous CUDA float32 or float16 logits [B,S,V], int64 targets, and float32 mask with B<=1024"
            )
        return tile_sequence_logp_reference(logits, targets, loss_mask, ignore_index)
    return _TileSequenceLogpFunction.apply(logits, targets, loss_mask, int(ignore_index), config)


class _TilePreferenceBCELossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reward_chosen: torch.Tensor, reward_rejected: torch.Tensor, config: TileCudaConfig):  # type: ignore[override]
        ctx.save_for_backward(reward_chosen, reward_rejected)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'preference_bce_loss'")
            return tile_preference_bce_loss_reference(reward_chosen, reward_rejected)
        return ext.tile_preference_bce_loss(_tile_kernel_input(reward_chosen), _tile_kernel_input(reward_rejected))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        reward_chosen, reward_rejected = ctx.saved_tensors
        diff = reward_chosen - reward_rejected
        grad = -torch.sigmoid(-diff) / reward_chosen.numel()
        grad = grad_output.to(dtype=reward_chosen.dtype) * grad
        return grad, -grad, None


def tile_preference_bce_loss_module(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
    config: TileCudaConfig,
) -> torch.Tensor:
    can_use = (
        reward_chosen.is_cuda
        and reward_rejected.is_cuda
        and reward_chosen.dtype in TILE_FLOAT_DTYPES
        and reward_rejected.dtype == reward_chosen.dtype
        and reward_chosen.is_contiguous()
        and reward_rejected.is_contiguous()
        and reward_chosen.shape == reward_rejected.shape
        and reward_chosen.numel() > 0
    )
    if not can_use:
        if config.strict and (reward_chosen.is_cuda or reward_rejected.is_cuda):
            raise RuntimeError("CUDA Tile module 'preference_bce_loss' requires same-shape contiguous CUDA float32 or float16 rewards")
        return tile_preference_bce_loss_reference(reward_chosen, reward_rejected)
    return _TilePreferenceBCELossFunction.apply(reward_chosen, reward_rejected, config)


class _TilePPOClippedLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        logp_new: torch.Tensor,
        logp_old: torch.Tensor,
        advantages: torch.Tensor,
        value_new: torch.Tensor,
        value_old: torch.Tensor,
        returns: torch.Tensor,
        clip_range: float,
        vf_coef: float,
        config: TileCudaConfig,
    ):
        ctx.clip_range = float(clip_range)
        ctx.vf_coef = float(vf_coef)
        ctx.save_for_backward(logp_new, logp_old, advantages, value_new, value_old, returns)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'ppo_clipped_loss'")
            return tile_ppo_clipped_loss_reference(
                logp_new, logp_old, advantages, value_new, value_old, returns, float(clip_range), float(vf_coef)
            )
        policy_loss, value_loss, loss = ext.tile_ppo_clipped_loss(
            _tile_kernel_input(logp_new),
            _tile_kernel_input(logp_old),
            _tile_kernel_input(advantages),
            _tile_kernel_input(value_new),
            _tile_kernel_input(value_old),
            _tile_kernel_input(returns),
            float(clip_range),
            float(vf_coef),
        )
        return policy_loss, value_loss, loss

    @staticmethod
    def backward(ctx, grad_policy: torch.Tensor, grad_value: torch.Tensor, grad_loss: torch.Tensor):  # type: ignore[override]
        logp_new, logp_old, advantages, value_new, value_old, returns = ctx.saved_tensors
        with torch.enable_grad():
            refs = tuple(
                t.detach().requires_grad_(True)
                for t in (logp_new, logp_old, advantages, value_new, value_old, returns)
            )
            policy_loss, value_loss, loss = tile_ppo_clipped_loss_reference(*refs, ctx.clip_range, ctx.vf_coef)
            grads = torch.autograd.grad(
                (policy_loss, value_loss, loss),
                refs,
                (grad_policy, grad_value, grad_loss),
                allow_unused=True,
            )
        return (*grads, None, None, None)


def tile_ppo_clipped_loss_module(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    value_new: torch.Tensor,
    value_old: torch.Tensor,
    returns: torch.Tensor,
    clip_range: float,
    vf_coef: float,
    config: TileCudaConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = (logp_new, logp_old, advantages, value_new, value_old, returns)
    can_use = (
        all(t.is_cuda and t.dtype in TILE_FLOAT_DTYPES and t.dtype == logp_new.dtype and t.is_contiguous() for t in tensors)
        and all(t.shape == logp_new.shape for t in tensors[1:])
        and logp_new.numel() > 0
    )
    if not can_use:
        if config.strict and any(t.is_cuda for t in tensors):
            raise RuntimeError("CUDA Tile module 'ppo_clipped_loss' requires same-shape contiguous CUDA float32 or float16 tensors")
        return tile_ppo_clipped_loss_reference(
            logp_new, logp_old, advantages, value_new, value_old, returns, clip_range, vf_coef
        )
    return _TilePPOClippedLossFunction.apply(
        logp_new,
        logp_old,
        advantages,
        value_new,
        value_old,
        returns,
        float(clip_range),
        float(vf_coef),
        config,
    )


class _TileGAEComputeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rewards: torch.Tensor, values: torch.Tensor, gamma: float, lambda_: float, config: TileCudaConfig):  # type: ignore[override]
        ctx.gamma = float(gamma)
        ctx.lambda_ = float(lambda_)
        ctx.save_for_backward(rewards, values)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'gae_compute'")
            return tile_gae_compute_reference(rewards, values, float(gamma), float(lambda_))
        advantages, returns = ext.tile_gae_compute(_tile_kernel_input(rewards), _tile_kernel_input(values), float(gamma), float(lambda_))
        return _tile_kernel_output(advantages, rewards.dtype), _tile_kernel_output(returns, rewards.dtype)

    @staticmethod
    def backward(ctx, grad_advantages: torch.Tensor, grad_returns: torch.Tensor):  # type: ignore[override]
        rewards, values = ctx.saved_tensors
        with torch.enable_grad():
            rewards_ref = _tile_kernel_input(rewards).detach().requires_grad_(True)
            values_ref = _tile_kernel_input(values).detach().requires_grad_(True)
            advantages, returns = tile_gae_compute_reference(rewards_ref, values_ref, ctx.gamma, ctx.lambda_)
            grad_rewards, grad_values = torch.autograd.grad(
                (advantages, returns),
                (rewards_ref, values_ref),
                (grad_advantages, grad_returns),
                allow_unused=True,
            )
        return (
            None if grad_rewards is None else _tile_kernel_output(grad_rewards, rewards.dtype),
            None if grad_values is None else _tile_kernel_output(grad_values, values.dtype),
            None,
            None,
            None,
        )


def tile_gae_compute_module(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambda_: float,
    config: TileCudaConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    can_use = (
        rewards.is_cuda
        and values.is_cuda
        and rewards.dtype in TILE_FLOAT_DTYPES
        and values.dtype == rewards.dtype
        and rewards.is_contiguous()
        and values.is_contiguous()
        and rewards.shape == values.shape
        and rewards.ndim == 2
        and rewards.numel() > 0
    )
    if not can_use:
        if config.strict and (rewards.is_cuda or values.is_cuda):
            raise RuntimeError("CUDA Tile module 'gae_compute' requires same-shape contiguous CUDA float32 or float16 tensors shaped [B,S]")
        return tile_gae_compute_reference(rewards, values, gamma, lambda_)
    return _TileGAEComputeFunction.apply(rewards, values, float(gamma), float(lambda_), config)


class _TileRouteSelectionLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        route_logits: torch.Tensor,
        sem_targets: torch.Tensor,
        num_vocab_dims: int,
        shared_experts: int,
        ignore_index: int,
        config: TileCudaConfig,
    ):
        ctx.num_vocab_dims = int(num_vocab_dims)
        ctx.shared_experts = int(shared_experts)
        ctx.ignore_index = int(ignore_index)
        ctx.save_for_backward(route_logits, sem_targets)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'route_selection_loss'")
            return tile_route_selection_loss_reference(
                route_logits, sem_targets, int(num_vocab_dims), int(shared_experts), int(ignore_index)
            )
        return ext.tile_route_selection_loss(
            _tile_kernel_input(route_logits),
            sem_targets,
            int(num_vocab_dims),
            int(shared_experts),
            int(ignore_index),
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        route_logits, sem_targets = ctx.saved_tensors
        with torch.enable_grad():
            logits_ref = route_logits.detach().requires_grad_(True)
            loss = tile_route_selection_loss_reference(
                logits_ref,
                sem_targets,
                ctx.num_vocab_dims,
                ctx.shared_experts,
                ctx.ignore_index,
            )
            (grad_logits,) = torch.autograd.grad(loss, (logits_ref,), grad_output, allow_unused=True)
        return grad_logits, None, None, None, None, None


def tile_route_selection_loss_module(
    route_logits: torch.Tensor,
    sem_targets: torch.Tensor,
    num_vocab_dims: int,
    shared_experts: int,
    ignore_index: int,
    config: TileCudaConfig,
) -> torch.Tensor:
    can_use = (
        route_logits.is_cuda
        and sem_targets.is_cuda
        and route_logits.dtype in TILE_FLOAT_DTYPES
        and sem_targets.dtype == torch.long
        and route_logits.is_contiguous()
        and sem_targets.is_contiguous()
        and route_logits.ndim == 3
        and sem_targets.ndim == 2
        and route_logits.shape[0] == sem_targets.shape[0]
        and int(num_vocab_dims) > 0
        and sem_targets.shape[1] >= int(num_vocab_dims)
        and route_logits.shape[-1] >= int(shared_experts) + int(num_vocab_dims)
        and route_logits.numel() > 0
    )
    if not can_use:
        if config.strict and (route_logits.is_cuda or sem_targets.is_cuda):
            raise RuntimeError(
                "CUDA Tile module 'route_selection_loss' requires CUDA float32 or float16 route_logits [B,S,E] and int64 sem_targets [B,D]"
            )
        return tile_route_selection_loss_reference(
            route_logits, sem_targets, int(num_vocab_dims), int(shared_experts), int(ignore_index)
        )
    return _TileRouteSelectionLossFunction.apply(
        route_logits,
        sem_targets,
        int(num_vocab_dims),
        int(shared_experts),
        int(ignore_index),
        config,
    )


def _dpo_loss_type_code(loss_type: str) -> int:
    if loss_type == "hinge":
        return 1
    if loss_type == "ipo":
        return 2
    return 0


class _TileDPOPairwiseLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        policy_logp_chosen: torch.Tensor,
        policy_logp_rejected: torch.Tensor,
        ref_logp_chosen: torch.Tensor,
        ref_logp_rejected: torch.Tensor,
        beta: float,
        label_smoothing: float,
        loss_type_code: int,
        config: TileCudaConfig,
    ):
        ctx.save_for_backward(policy_logp_chosen, policy_logp_rejected, ref_logp_chosen, ref_logp_rejected)
        ctx.beta = float(beta)
        ctx.label_smoothing = float(label_smoothing)
        ctx.loss_type_code = int(loss_type_code)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'dpo_pairwise_loss'")
            loss_type = "hinge" if loss_type_code == 1 else "ipo" if loss_type_code == 2 else "sigmoid"
            loss, chosen_reward, rejected_reward = tile_dpo_pairwise_loss_reference(
                policy_logp_chosen,
                policy_logp_rejected,
                ref_logp_chosen,
                ref_logp_rejected,
                beta,
                label_smoothing,
                loss_type,
            )
        else:
            loss, chosen_reward, rejected_reward = ext.tile_dpo_pairwise_loss(
                _tile_kernel_input(policy_logp_chosen),
                _tile_kernel_input(policy_logp_rejected),
                _tile_kernel_input(ref_logp_chosen),
                _tile_kernel_input(ref_logp_rejected),
                float(beta),
                float(label_smoothing),
                int(loss_type_code),
            )
        ctx.mark_non_differentiable(chosen_reward, rejected_reward)
        return loss, _tile_kernel_output(chosen_reward, policy_logp_chosen.dtype), _tile_kernel_output(rejected_reward, policy_logp_chosen.dtype)

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor, grad_chosen_reward: torch.Tensor, grad_rejected_reward: torch.Tensor):  # type: ignore[override]
        del grad_chosen_reward, grad_rejected_reward
        policy_logp_chosen, policy_logp_rejected, ref_logp_chosen, ref_logp_rejected = ctx.saved_tensors
        beta = float(ctx.beta)
        logits = beta * ((policy_logp_chosen - ref_logp_chosen) - (policy_logp_rejected - ref_logp_rejected))
        if int(ctx.loss_type_code) == 1:
            dloss_dlogits = torch.where(logits < 1.0, torch.full_like(logits, -1.0), torch.zeros_like(logits))
        elif int(ctx.loss_type_code) == 2:
            target = 1.0 / (2.0 * max(beta, 1e-8))
            dloss_dlogits = 2.0 * (logits - target)
        else:
            dloss_dlogits = torch.sigmoid(logits) - (1.0 - float(ctx.label_smoothing))
        scale = grad_loss.to(dtype=policy_logp_chosen.dtype) * dloss_dlogits / policy_logp_chosen.numel()
        grad_policy_chosen = scale * beta
        grad_policy_rejected = -scale * beta
        grad_ref_chosen = -grad_policy_chosen
        grad_ref_rejected = -grad_policy_rejected
        return grad_policy_chosen, grad_policy_rejected, grad_ref_chosen, grad_ref_rejected, None, None, None, None


def tile_dpo_pairwise_loss_module(
    policy_logp_chosen: torch.Tensor,
    policy_logp_rejected: torch.Tensor,
    ref_logp_chosen: torch.Tensor,
    ref_logp_rejected: torch.Tensor,
    beta: float,
    label_smoothing: float,
    loss_type: str,
    config: TileCudaConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = (policy_logp_chosen, policy_logp_rejected, ref_logp_chosen, ref_logp_rejected)
    can_use = (
        all(tensor.is_cuda and tensor.dtype in TILE_FLOAT_DTYPES and tensor.dtype == policy_logp_chosen.dtype and tensor.is_contiguous() for tensor in tensors)
        and policy_logp_chosen.numel() > 0
        and policy_logp_chosen.shape == policy_logp_rejected.shape == ref_logp_chosen.shape == ref_logp_rejected.shape
        and loss_type in {"sigmoid", "hinge", "ipo"}
    )
    if not can_use:
        if config.strict and any(tensor.is_cuda for tensor in tensors):
            raise RuntimeError(
                "CUDA Tile module 'dpo_pairwise_loss' requires same-shape contiguous CUDA float32 or float16 log-prob tensors"
            )
        return tile_dpo_pairwise_loss_reference(
            policy_logp_chosen,
            policy_logp_rejected,
            ref_logp_chosen,
            ref_logp_rejected,
            beta,
            label_smoothing,
            loss_type,
        )
    return _TileDPOPairwiseLossFunction.apply(
        policy_logp_chosen,
        policy_logp_rejected,
        ref_logp_chosen,
        ref_logp_rejected,
        float(beta),
        float(label_smoothing),
        _dpo_loss_type_code(loss_type),
        config,
    )


def _semantic_alignment_canonical_tensors(
    pred: torch.Tensor,
    target: torch.Tensor,
    term_counts: tuple[int, ...] | list[int],
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    logits = pred
    targets = target.long()
    if logits.ndim == 4:
        batch, chunks, dims, terms = logits.shape
        logits = logits.reshape(batch * chunks, dims, terms)
        if targets.ndim == 1:
            targets = targets.unsqueeze(0)
        targets = targets.unsqueeze(1).expand(batch, chunks, targets.size(-1)).reshape(batch * chunks, targets.size(-1))
    if targets.ndim == 1:
        targets = targets.unsqueeze(0)
    n_dims = min(len(term_counts), logits.size(1), targets.size(1))
    return logits[:, :n_dims, :].contiguous(), targets[:, :n_dims].contiguous(), tuple(int(v) for v in term_counts[:n_dims])


class _TileSemanticAlignmentLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        pred: torch.Tensor,
        target: torch.Tensor,
        term_counts_tensor: torch.Tensor,
        ignore_index: int,
        term_counts: tuple[int, ...],
        config: TileCudaConfig,
    ):
        ctx.ignore_index = int(ignore_index)
        ctx.term_counts = tuple(int(v) for v in term_counts)
        ctx.save_for_backward(pred, target)
        logits, targets, counts = _semantic_alignment_canonical_tensors(pred, target, ctx.term_counts)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'semantic_alignment_loss'")
            return tile_semantic_alignment_loss_reference(pred, target, ctx.term_counts, int(ignore_index))
        return ext.tile_semantic_alignment_loss(
            _tile_kernel_input(logits),
            targets,
            term_counts_tensor[: len(counts)].contiguous(),
            int(ignore_index),
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        pred, target = ctx.saved_tensors
        with torch.enable_grad():
            pred_ref = pred.detach().requires_grad_(True)
            loss = tile_semantic_alignment_loss_reference(pred_ref, target, ctx.term_counts, ctx.ignore_index)
            (grad_pred,) = torch.autograd.grad(loss, (pred_ref,), grad_output, allow_unused=True)
        return grad_pred, None, None, None, None, None


def tile_semantic_alignment_loss_module(
    pred: torch.Tensor,
    target: torch.Tensor,
    term_counts: tuple[int, ...] | list[int],
    ignore_index: int,
    config: TileCudaConfig,
) -> torch.Tensor:
    term_counts_tuple = tuple(int(v) for v in term_counts)
    logits, targets, counts = _semantic_alignment_canonical_tensors(pred, target, term_counts_tuple)
    term_counts_tensor = torch.tensor(counts, device=pred.device, dtype=torch.long)
    can_use = (
        pred.is_cuda
        and target.is_cuda
        and pred.dtype in TILE_FLOAT_DTYPES
        and target.dtype == torch.long
        and pred.is_contiguous()
        and target.is_contiguous()
        and pred.ndim in {3, 4}
        and target.ndim in {1, 2}
        and logits.ndim == 3
        and targets.ndim == 2
        and logits.is_contiguous()
        and targets.is_contiguous()
        and logits.size(0) == targets.size(0)
        and logits.size(1) == targets.size(1)
        and logits.size(1) == len(counts)
        and logits.size(0) > 0
        and logits.size(1) > 0
        and logits.size(2) > 0
    )
    if not can_use:
        if config.strict and (pred.is_cuda or target.is_cuda):
            raise RuntimeError(
                "CUDA Tile module 'semantic_alignment_loss' requires contiguous CUDA float32 or float16 logits [R,D,T] or [B,C,D,T] and int64 targets"
            )
        return tile_semantic_alignment_loss_reference(pred, target, term_counts_tuple, int(ignore_index))
    return _TileSemanticAlignmentLossFunction.apply(
        pred,
        target,
        term_counts_tensor,
        int(ignore_index),
        counts,
        config,
    )


class _TileRouteBalanceLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, route_logits: torch.Tensor, config: TileCudaConfig):  # type: ignore[override]
        ctx.save_for_backward(route_logits)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'route_balance_loss'")
            return tile_route_balance_loss_reference(route_logits)
        return ext.tile_route_balance_loss(_tile_kernel_input(route_logits))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (route_logits,) = ctx.saved_tensors
        rows = route_logits.reshape(-1, route_logits.size(-1))
        probs = F.softmax(rows.float(), dim=-1)
        density = probs.mean(dim=0)
        weighted = (probs * density.unsqueeze(0)).sum(dim=-1, keepdim=True)
        grad_rows = (2.0 * rows.size(-1) / rows.size(0)) * probs * (density.unsqueeze(0) - weighted)
        return grad_output.to(dtype=route_logits.dtype) * grad_rows.reshape_as(route_logits).to(dtype=route_logits.dtype), None


def tile_route_balance_loss_module(route_logits: torch.Tensor, config: TileCudaConfig) -> torch.Tensor:
    can_use = (
        route_logits.is_cuda
        and route_logits.dtype in TILE_FLOAT_DTYPES
        and route_logits.is_contiguous()
        and route_logits.ndim >= 1
        and route_logits.numel() > 0
        and route_logits.size(-1) > 0
        and route_logits.size(-1) <= 1024
        and route_logits.numel() // route_logits.size(-1) <= 1024
    )
    if not can_use:
        if config.strict and route_logits.is_cuda:
            raise RuntimeError(
                "CUDA Tile module 'route_balance_loss' requires contiguous CUDA float32 or float16 logits with rows<=1024 and experts<=1024"
            )
        return tile_route_balance_loss_reference(route_logits)
    return _TileRouteBalanceLossFunction.apply(route_logits, config)


class _TileLoadBalanceLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        router_logits: torch.Tensor,
        routing_weights: torch.Tensor,
        routing_indices: torch.Tensor,
        config: TileCudaConfig,
    ):
        del routing_weights, routing_indices
        ctx.save_for_backward(router_logits)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'load_balance_loss'")
            return tile_route_balance_loss_reference(router_logits), router_logits
        return ext.tile_route_balance_loss(_tile_kernel_input(router_logits)), router_logits

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor, grad_router_passthrough: torch.Tensor):  # type: ignore[override]
        (router_logits,) = ctx.saved_tensors
        rows = router_logits.reshape(-1, router_logits.size(-1))
        probs = F.softmax(rows.float(), dim=-1)
        density = probs.mean(dim=0)
        weighted = (probs * density.unsqueeze(0)).sum(dim=-1, keepdim=True)
        grad_rows = (2.0 * rows.size(-1) / rows.size(0)) * probs * (density.unsqueeze(0) - weighted)
        grad_router = grad_loss.to(dtype=router_logits.dtype) * grad_rows.reshape_as(router_logits).to(dtype=router_logits.dtype)
        if grad_router_passthrough is not None:
            grad_router = grad_router + grad_router_passthrough.to(dtype=router_logits.dtype)
        return grad_router, None, None, None


def tile_load_balance_loss_module(
    router_logits: torch.Tensor,
    routing_weights: torch.Tensor,
    routing_indices: torch.Tensor,
    config: TileCudaConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    can_use = (
        router_logits.is_cuda
        and router_logits.dtype in TILE_FLOAT_DTYPES
        and router_logits.is_contiguous()
        and router_logits.ndim >= 1
        and router_logits.numel() > 0
        and router_logits.size(-1) > 0
        and router_logits.size(-1) <= 1024
        and router_logits.numel() // router_logits.size(-1) <= 1024
    )
    if not can_use:
        if config.strict and router_logits.is_cuda:
            raise RuntimeError(
                "CUDA Tile module 'load_balance_loss' requires contiguous CUDA float32 or float16 logits with rows<=1024 and experts<=1024"
            )
        return tile_load_balance_loss_reference(router_logits, routing_weights, routing_indices)
    return _TileLoadBalanceLossFunction.apply(router_logits, routing_weights, routing_indices, config)


class _TileSoftmaxDistillationLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, teacher_logits: torch.Tensor, student_logits: torch.Tensor, config: TileCudaConfig):  # type: ignore[override]
        ctx.save_for_backward(teacher_logits, student_logits)
        ext = load_tile_cuda_extension(config)
        if ext is None:
            if config.strict:
                raise RuntimeError("CUDA Tile extension is unavailable for module 'softmax_distillation_loss'")
            return tile_softmax_distillation_loss_reference(teacher_logits, student_logits)
        return ext.tile_softmax_distillation_loss(_tile_kernel_input(teacher_logits), _tile_kernel_input(student_logits))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        teacher_logits, student_logits = ctx.saved_tensors
        teacher_prob = F.softmax(teacher_logits.float().detach(), dim=-1)
        student_prob = F.softmax(student_logits.float(), dim=-1)
        grad_student = (student_prob - teacher_prob) / max(student_logits.size(0), 1)
        return None, grad_output.to(dtype=student_logits.dtype) * grad_student.to(dtype=student_logits.dtype), None


def tile_softmax_distillation_loss_module(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    config: TileCudaConfig,
) -> torch.Tensor:
    can_use = (
        teacher_logits.is_cuda
        and student_logits.is_cuda
        and teacher_logits.dtype in TILE_FLOAT_DTYPES
        and student_logits.dtype == teacher_logits.dtype
        and teacher_logits.is_contiguous()
        and student_logits.is_contiguous()
        and teacher_logits.shape == student_logits.shape
        and teacher_logits.ndim >= 2
        and teacher_logits.numel() > 0
        and teacher_logits.size(-1) <= 1024
    )
    if not can_use:
        if config.strict and (teacher_logits.is_cuda or student_logits.is_cuda):
            raise RuntimeError(
                "CUDA Tile module 'softmax_distillation_loss' requires same-shape contiguous CUDA float32 or float16 logits with vocab<=1024"
            )
        return tile_softmax_distillation_loss_reference(teacher_logits, student_logits)
    return _TileSoftmaxDistillationLossFunction.apply(teacher_logits, student_logits, config)
