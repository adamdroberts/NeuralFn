from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..torch_backend import RoutingStatsMixin
from .autograd import (
    BINARY_OPS,
    BINARY_PAIR_OPS,
    SCALAR_BINARY_MODULE_OPS,
    SCALAR_TERNARY_MODULE_OPS,
    SCALAR_UNARY_MODULE_OPS,
    TILE_FUNCTION_NAMES,
    TILE_MODULE_NAMES,
    UNARY_OPS,
    VECTOR_BINARY_MODULE_OPS,
    tile_absolute_position_embedding_module,
    tile_attentionless_decoder_module,
    tile_act_weighted_sum_module,
    tile_binary,
    tile_binary_pair,
    tile_broadcast_chunk_routes_module,
    tile_broadcast_expert_routes_module,
    tile_byte_patch_embed_module,
    tile_byte_patch_merge_module,
    tile_causal_chunk_state_module,
    tile_expert_bias_add_module,
    tile_identity_module,
    tile_gae_compute_module,
    tile_group_norm_module,
    tile_kv_cache_read_module,
    tile_kv_cache_write_module,
    tile_kv_quant_pack_module,
    tile_kv_quant_unpack_module,
    tile_latent_pool_module,
    tile_latent_mse_loss_module,
    tile_layer_norm_module,
    tile_linear_module,
    tile_load_balance_loss_module,
    tile_masked_token_cross_entropy_module,
    tile_scalar_binary_module,
    tile_scalar_ternary_module,
    tile_scalar_unary_module,
    tile_sequence_logp_module,
    tile_semantic_alignment_loss_module,
    tile_semantic_hash_module,
    tile_scaled_residual_add_module,
    tile_softmax_lastdim_module,
    tile_token_embedding_module,
    tile_dyt_module,
    tile_dpo_pairwise_loss_module,
    tile_qk_gain_module,
    tile_ppo_clipped_loss_module,
    tile_preference_bce_loss_module,
    tile_merge_heads_module,
    tile_repeat_kv_module,
    tile_route_balance_loss_module,
    tile_route_selection_loss_module,
    tile_rotary_embedding_module,
    tile_rms_norm_module,
    tile_reshape_heads_module,
    tile_scaled_dot_product_attention_module,
    tile_softmax_distillation_loss_module,
    tile_token_cross_entropy_module,
    tile_topk_route_module,
    tile_unary,
    tile_vector_binary_module,
)
from .config import TileCudaConfig
from .runtime import load_tile_cuda_extension


def _deterministic_uniform(shape: tuple[int, ...], device: torch.device, counter: Tensor, salt: int) -> Tensor:
    n = math.prod(shape)
    idx = torch.arange(n, device=device, dtype=torch.long)
    value = (idx * 1_103_515_245 + counter.to(device=device, dtype=torch.long) * 12_345 + int(salt)) % 16_777_216
    return (value.to(dtype=torch.float32) / 16_777_216.0).reshape(shape)


class TileCudaUnaryFunctionStage(nn.Module):
    def __init__(self, name: str, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        if name not in UNARY_OPS:
            raise KeyError(f"Unsupported CUDA Tile unary function stage: {name}")
        self.name = name
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        return tile_unary(self.name, x, self.config)


class TileCudaBinaryFunctionStage(nn.Module):
    def __init__(self, name: str, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        if name not in BINARY_OPS:
            raise KeyError(f"Unsupported CUDA Tile binary function stage: {name}")
        self.name = name
        self.config = config or TileCudaConfig()

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return tile_binary(self.name, lhs, rhs, self.config)


class TileCudaBinaryPairFunctionStage(nn.Module):
    def __init__(self, name: str, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        if name not in BINARY_PAIR_OPS:
            raise KeyError(f"Unsupported CUDA Tile binary-pair function stage: {name}")
        self.name = name
        self.config = config or TileCudaConfig()

    def forward(self, lhs: Tensor, rhs: Tensor) -> tuple[Tensor, Tensor]:
        return tile_binary_pair(self.name, lhs, rhs, self.config)


def build_tile_function_module(name: str, config: TileCudaConfig | None = None) -> nn.Module | None:
    if name in UNARY_OPS:
        return TileCudaUnaryFunctionStage(name, config)
    if name in BINARY_OPS:
        return TileCudaBinaryFunctionStage(name, config)
    if name in BINARY_PAIR_OPS:
        return TileCudaBinaryPairFunctionStage(name, config)
    return None


class TileCudaLogitSoftcapStage(nn.Module):
    def __init__(self, softcap: float, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        if softcap <= 0.0:
            raise ValueError("softcap must be positive")
        self.softcap = float(softcap)
        self.config = config or TileCudaConfig()

    def forward(self, logits: Tensor) -> Tensor:
        return tile_scalar_unary_module("logit_softcap", logits, self.softcap, self.config)


class TileCudaLossScaleStage(nn.Module):
    def __init__(self, coef: float, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.coef = float(coef)
        self.config = config or TileCudaConfig()

    def forward(self, loss: Tensor) -> Tensor:
        return tile_scalar_unary_module("loss_scale", loss, self.coef, self.config)


class TileCudaAuxLossAddStage(nn.Module):
    def __init__(self, coef: float, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.coef = float(coef)
        self.config = config or TileCudaConfig()

    def forward(self, main_loss: Tensor, aux_loss: Tensor) -> Tensor:
        return tile_scalar_binary_module("aux_loss_add", main_loss, aux_loss, self.coef, self.config)


class TileCudaKLPenaltyStage(nn.Module):
    def __init__(self, kl_coef: float = 0.1, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.kl_coef = float(kl_coef)
        self.config = config or TileCudaConfig()

    def forward(self, logp_policy: Tensor, logp_ref: Tensor, rewards: Tensor) -> Tensor:
        return tile_scalar_ternary_module("kl_penalty", logp_policy, logp_ref, rewards, self.kl_coef, self.config)


class TileCudaResidualAddStage(nn.Module):
    def __init__(self, dim: int, init_scale: float = 1.0, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), init_scale, dtype=torch.float32))
        self.config = config or TileCudaConfig()

    def forward(self, residual: Tensor, delta: Tensor) -> Tensor:
        return tile_vector_binary_module("residual_add", residual, delta, self.scale, None, self.config)


class TileCudaResidualMixStage(nn.Module):
    def __init__(
        self,
        dim: int,
        primary_init: float = 1.0,
        skip_init: float = 0.0,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        self.primary_scale = nn.Parameter(torch.full((dim,), primary_init, dtype=torch.float32))
        self.skip_scale = nn.Parameter(torch.full((dim,), skip_init, dtype=torch.float32))
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        return tile_vector_binary_module("residual_mix", x, x0, self.primary_scale, self.skip_scale, self.config)


class TileCudaManifoldHyperConnectionStage(nn.Module):
    def __init__(self, dim: int, beta_init: float = 0.1, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        b = min(max(float(beta_init), 1e-3), 1.0 - 1e-3)
        logit = torch.logit(torch.tensor(b, dtype=torch.float32)).item()
        self.beta_logit = nn.Parameter(torch.full((dim,), float(logit), dtype=torch.float32))
        self.config = config or TileCudaConfig()

    def forward(self, residual: Tensor, delta: Tensor) -> Tensor:
        return tile_vector_binary_module(
            "manifold_hyper_connection", residual, delta, self.beta_logit, None, self.config
        )


class TileCudaQKGainStage(nn.Module):
    def __init__(self, num_heads: int, qk_gain_init: float, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.config = config or TileCudaConfig()

    def forward(self, q: Tensor) -> Tensor:
        return tile_qk_gain_module(q, self.q_gain, self.config)


class TileCudaDyTStage(nn.Module):
    def __init__(self, model_dim: int, alpha_init: float = 1.0, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.weight = nn.Parameter(torch.ones(model_dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32))
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        return tile_dyt_module(x, self.alpha, self.weight, self.bias, self.config)


class TileCudaDropoutStage(nn.Module):
    def __init__(self, p: float, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.p = float(p)
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p <= 0.0:
            return tile_identity_module("dropout", x, self.config)
        if self.p >= 1.0:
            return x * 0.0
        return F.dropout(x, p=self.p, training=True)


class TileCudaRandomTimestepsStage(nn.Module):
    def __init__(self, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.register_buffer("_rng_counter", torch.zeros((), dtype=torch.long), persistent=False)
        self.config = config or TileCudaConfig()

    def forward(self, tokens: Tensor) -> Tensor:
        counter = int(self._rng_counter.item())
        if tokens.is_cuda:
            ext = load_tile_cuda_extension(self.config)
            if ext is None:
                if self.config.strict:
                    raise RuntimeError("CUDA Tile extension is unavailable for module 'random_timesteps'")
                out = _deterministic_uniform((tokens.size(0),), tokens.device, self._rng_counter, 17)
            else:
                out = ext.tile_random_timesteps(tokens, counter)
        else:
            out = _deterministic_uniform((tokens.size(0),), tokens.device, self._rng_counter, 17)
        self._rng_counter.add_(1)
        return out


class TileCudaMaskSchedulerStage(nn.Module):
    def __init__(self, vocab_size: int, mask_token_id: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.mask_token_id = int(mask_token_id)
        self.register_buffer("_rng_counter", torch.zeros((), dtype=torch.long), persistent=False)
        self.config = config or TileCudaConfig()

    def forward(self, tokens: Tensor, timesteps: Tensor) -> Tensor:
        counter = int(self._rng_counter.item())
        can_use = (
            tokens.is_cuda
            and timesteps.is_cuda
            and tokens.dtype == torch.long
            and timesteps.dtype == torch.float32
            and tokens.is_contiguous()
            and timesteps.is_contiguous()
            and tokens.ndim == 2
            and timesteps.ndim == 1
            and tokens.size(0) == timesteps.size(0)
        )
        if can_use:
            ext = load_tile_cuda_extension(self.config)
            if ext is None:
                if self.config.strict:
                    raise RuntimeError("CUDA Tile extension is unavailable for module 'mask_scheduler'")
                noise = _deterministic_uniform(tuple(tokens.shape), tokens.device, self._rng_counter, 53)
                out = torch.where(noise < timesteps.view(-1, 1), tokens.new_full(tokens.shape, self.mask_token_id), tokens)
            else:
                out = ext.tile_mask_scheduler(tokens, timesteps, self.mask_token_id, counter)
        else:
            if self.config.strict and (tokens.is_cuda or timesteps.is_cuda):
                raise RuntimeError("CUDA Tile module 'mask_scheduler' requires contiguous CUDA int64 tokens [B,S] and float32 timesteps [B]")
            noise = _deterministic_uniform(tuple(tokens.shape), tokens.device, self._rng_counter, 53)
            out = torch.where(noise < timesteps.view(-1, 1), tokens.new_full(tokens.shape, self.mask_token_id), tokens)
        self._rng_counter.add_(1)
        return out


class TileCudaJEPAMaskStage(nn.Module):
    def __init__(
        self,
        mask_ratio: float,
        mask_token_id: int = 0,
        mask_strategy: str = "random",
        num_blocks: int = 4,
        min_block_ratio: float = 0.1,
        max_block_ratio: float = 0.25,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        self.mask_ratio = float(mask_ratio)
        self.mask_token_id = int(mask_token_id)
        self.mask_strategy = str(mask_strategy)
        self.num_blocks = int(num_blocks)
        self.min_block_ratio = float(min_block_ratio)
        self.max_block_ratio = float(max_block_ratio)
        self.register_buffer("_rng_counter", torch.zeros((), dtype=torch.long), persistent=False)
        self.config = config or TileCudaConfig()

    def _fallback(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        if self.mask_strategy == "block":
            batch, seq_len = tokens.shape
            mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=tokens.device)
            min_len = max(1, int(self.min_block_ratio * seq_len))
            max_len = max(min_len, int(self.max_block_ratio * seq_len))
            len_span = max(max_len - min_len + 1, 1)
            positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
            for block_idx in range(self.num_blocks):
                block_noise = _deterministic_uniform((batch,), tokens.device, self._rng_counter, 101 + block_idx * 2)
                start_noise = _deterministic_uniform((batch,), tokens.device, self._rng_counter, 102 + block_idx * 2)
                block_len = (block_noise * len_span).long().clamp(max=len_span - 1) + min_len
                max_start = (seq_len - block_len).clamp_min(0)
                start = (start_noise * (max_start.float() + 1.0)).long().clamp_max(max_start)
                mask = mask | ((positions >= start.unsqueeze(1)) & (positions < (start + block_len).unsqueeze(1)))
        else:
            noise = _deterministic_uniform(tuple(tokens.shape), tokens.device, self._rng_counter, 29)
            mask = noise < self.mask_ratio
        masked = tokens.clone()
        masked[mask] = self.mask_token_id
        return masked, mask.to(dtype=torch.float32)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        counter = int(self._rng_counter.item())
        can_use = tokens.is_cuda and tokens.dtype == torch.long and tokens.is_contiguous() and tokens.ndim == 2
        if can_use:
            ext = load_tile_cuda_extension(self.config)
            if ext is None:
                if self.config.strict:
                    raise RuntimeError("CUDA Tile extension is unavailable for module 'jepa_mask'")
                out = self._fallback(tokens)
            else:
                out = tuple(
                    ext.tile_jepa_mask(
                        tokens,
                        self.mask_ratio,
                        self.mask_token_id,
                        self.mask_strategy,
                        self.num_blocks,
                        self.min_block_ratio,
                        self.max_block_ratio,
                        counter,
                    )
                )
        else:
            if self.config.strict and tokens.is_cuda:
                raise RuntimeError("CUDA Tile module 'jepa_mask' requires contiguous CUDA int64 tokens [B,S]")
            out = self._fallback(tokens)
        self._rng_counter.add_(1)
        return out


class TileCudaReshapeHeadsStage(nn.Module):
    def __init__(self, num_heads: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        return tile_reshape_heads_module(x, self.num_heads, self.config)


class TileCudaMergeHeadsStage(nn.Module):
    def __init__(self, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        return tile_merge_heads_module(x, self.config)


class TileCudaRepeatKVStage(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.repeats = num_heads // num_kv_heads
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        return tile_repeat_kv_module(x, self.repeats, self.config)


class TileCudaScaledDotProductAttentionStage(nn.Module):
    def __init__(
        self,
        is_causal: bool = True,
        backend: str = "sdpa",
        dropout_p: float = 0.0,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        self.is_causal = bool(is_causal)
        self.backend = str(backend)
        self.dropout_p = float(dropout_p)
        self.config = config or TileCudaConfig()

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        drop = self.dropout_p if self.training else 0.0
        if drop != 0.0 or self.backend == "flex":
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=drop,
                is_causal=self.is_causal,
                enable_gqa=q.size(1) != k.size(1),
            )
        return tile_scaled_dot_product_attention_module(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            self.is_causal,
            self.config,
        )


class TileCudaSparseAttentionStage(nn.Module):
    def __init__(
        self,
        *,
        window: int | None = None,
        num_sinks: int = 0,
        block_size: int | None = None,
        compress_stride: int | None = None,
        is_causal: bool = True,
        dropout_p: float = 0.0,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        self.window = int(window) if window is not None else None
        self.num_sinks = int(num_sinks)
        self.block_size = int(block_size) if block_size is not None else None
        self.compress_stride = int(compress_stride) if compress_stride is not None else None
        self.is_causal = bool(is_causal)
        self.dropout_p = float(dropout_p)
        self.config = config or TileCudaConfig()

    def _reference(self, q: Tensor, k: Tensor, v: Tensor, drop: float) -> Tensor:
        seq_q, seq_k = q.size(-2), k.size(-2)
        i = torch.arange(seq_q, device=q.device).unsqueeze(1)
        j = torch.arange(seq_k, device=q.device).unsqueeze(0)
        offset = seq_k - seq_q
        causal_ok = (j <= i + offset) if self.is_causal else torch.ones(seq_q, seq_k, dtype=torch.bool, device=q.device)
        keep = torch.zeros(seq_q, seq_k, dtype=torch.bool, device=q.device)
        any_rule = False
        if self.window is not None and self.window > 0:
            keep = keep | (j > (i + offset) - self.window)
            any_rule = True
        if self.num_sinks > 0:
            keep = keep | (j < self.num_sinks)
            any_rule = True
        if self.block_size is not None and self.block_size > 0:
            keep = keep | ((i + offset) // self.block_size == j // self.block_size)
            any_rule = True
        if self.compress_stride is not None and self.compress_stride > 1:
            keep = keep | (j % self.compress_stride == 0)
            any_rule = True
        if not any_rule:
            keep = torch.ones(seq_q, seq_k, dtype=torch.bool, device=q.device)
        mask = torch.zeros(seq_q, seq_k, dtype=q.dtype, device=q.device).masked_fill(~(causal_ok & keep), float("-inf"))
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=drop,
            is_causal=False,
            enable_gqa=q.size(1) != k.size(1),
        )

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        drop = self.dropout_p if self.training else 0.0
        if drop != 0.0:
            return self._reference(q, k, v, drop)
        return tile_scaled_dot_product_attention_module(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            self.is_causal,
            self.config,
            right_align_causal=True,
            window=self.window,
            num_sinks=self.num_sinks,
            block_size=self.block_size,
            compress_stride=self.compress_stride,
        )


class TileCudaDifferentialAttentionStage(nn.Module):
    def __init__(
        self,
        lambda_init: float = 0.8,
        is_causal: bool = True,
        dropout_p: float = 0.0,
        eps: float = 1e-5,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        self.is_causal = bool(is_causal)
        self.dropout_p = float(dropout_p)
        self.eps = float(eps)
        self.lambda_param = nn.Parameter(torch.tensor(float(lambda_init), dtype=torch.float32))
        self.lambda_init = float(lambda_init)
        self.config = config or TileCudaConfig()

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        head_dim = q.size(-1)
        if head_dim % 2 != 0:
            raise ValueError("differential attention requires an even head_dim")
        half = head_dim // 2
        q1, q2 = q[..., :half].contiguous(), q[..., half:].contiguous()
        k1, k2 = k[..., :half].contiguous(), k[..., half:].contiguous()
        drop = self.dropout_p if self.training else 0.0
        if drop != 0.0:
            gqa = q.size(1) != k.size(1)
            a1 = F.scaled_dot_product_attention(q1, k1, v, dropout_p=drop, is_causal=self.is_causal, enable_gqa=gqa)
            a2 = F.scaled_dot_product_attention(q2, k2, v, dropout_p=drop, is_causal=self.is_causal, enable_gqa=gqa)
        else:
            a1 = tile_scaled_dot_product_attention_module(q1, k1, v.contiguous(), self.is_causal, self.config)
            a2 = tile_scaled_dot_product_attention_module(q2, k2, v.contiguous(), self.is_causal, self.config)
        out = a1 - self.lambda_param.to(dtype=a1.dtype) * a2
        out = tile_rms_norm_module(out.contiguous(), self.eps, self.config)
        return out * (1.0 - self.lambda_init)


class TileCudaCausalSelfAttentionStage(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        head_dim = model_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        kv_dim = int(num_kv_heads) * head_dim
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.q_proj = nn.Linear(int(model_dim), int(model_dim), bias=False)
        self.k_proj = nn.Linear(int(model_dim), kv_dim, bias=False)
        self.v_proj = nn.Linear(int(model_dim), kv_dim, bias=False)
        self.out_proj = nn.Linear(int(model_dim), int(model_dim), bias=False)
        self.q_gain = nn.Parameter(torch.full((self.num_heads,), float(qk_gain_init), dtype=torch.float32))
        inv_freq = TileCudaRotaryEmbeddingStage._compute_inv_freq(self.head_dim, float(rope_base), None)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, model_dim = x.shape
        q = tile_linear_module(x, self.q_proj.weight, None, self.config, name="causal_self_attention.q_proj")
        k = tile_linear_module(x, self.k_proj.weight, None, self.config, name="causal_self_attention.k_proj")
        v = tile_linear_module(x, self.v_proj.weight, None, self.config, name="causal_self_attention.v_proj")
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        eps = torch.finfo(torch.float32).eps
        q = tile_rms_norm_module(q, eps, self.config)
        k = tile_rms_norm_module(k, eps, self.config)
        q, k = tile_rotary_embedding_module(q.contiguous(), k.contiguous(), self.inv_freq.to(device=x.device), self.config)
        q = tile_qk_gain_module(q.contiguous(), self.q_gain, self.config)
        y = tile_scaled_dot_product_attention_module(q.contiguous(), k.contiguous(), v.contiguous(), True, self.config)
        y = y.transpose(1, 2).contiguous().reshape(batch, seq_len, model_dim)
        return tile_linear_module(y, self.out_proj.weight, None, self.config, name="causal_self_attention.out_proj")


class TileCudaFusedCausalAttentionStage(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float = 10000.0,
        dropout_p: float = 0.0,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        head_dim = model_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        kv_dim = int(num_kv_heads) * head_dim
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.dropout_p = float(dropout_p)
        self.q_proj = nn.Linear(int(model_dim), int(model_dim), bias=False)
        self.k_proj = nn.Linear(int(model_dim), kv_dim, bias=False)
        self.v_proj = nn.Linear(int(model_dim), kv_dim, bias=False)
        self.out_proj = nn.Linear(int(model_dim), int(model_dim), bias=False)
        inv_freq = TileCudaRotaryEmbeddingStage._compute_inv_freq(self.head_dim, float(rope_base), None)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, model_dim = x.shape
        q = tile_linear_module(x, self.q_proj.weight, None, self.config, name="fused_causal_attention.q_proj")
        k = tile_linear_module(x, self.k_proj.weight, None, self.config, name="fused_causal_attention.k_proj")
        v = tile_linear_module(x, self.v_proj.weight, None, self.config, name="fused_causal_attention.v_proj")
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        q, k = tile_rotary_embedding_module(q, k, self.inv_freq.to(device=x.device), self.config)
        drop = self.dropout_p if self.training else 0.0
        if drop != 0.0:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=drop,
                is_causal=True,
                enable_gqa=self.num_heads != self.num_kv_heads,
            )
        else:
            y = tile_scaled_dot_product_attention_module(q.contiguous(), k.contiguous(), v.contiguous(), True, self.config)
        y = y.transpose(1, 2).contiguous().reshape(batch, seq_len, model_dim)
        return tile_linear_module(y, self.out_proj.weight, None, self.config, name="fused_causal_attention.out_proj")


class TileCudaMLAStage(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        kv_lora_rank: int | None = None,
        qk_rope_dim: int | None = None,
        rope_base: float = 10000.0,
        dropout_p: float = 0.0,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        head_dim = model_dim // num_heads
        rope_dim = int(qk_rope_dim) if qk_rope_dim is not None else max(head_dim // 2, 2)
        if rope_dim % 2 != 0:
            rope_dim -= 1
        rope_dim = max(rope_dim, 2)
        nope_dim = head_dim - rope_dim
        if nope_dim <= 0:
            raise ValueError("qk_rope_dim must be smaller than head_dim")
        lora_rank = int(kv_lora_rank) if kv_lora_rank is not None else max(2 * head_dim, model_dim // 2)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rope_dim = int(rope_dim)
        self.nope_dim = int(nope_dim)
        self.v_head_dim = int(head_dim)
        self.kv_lora_rank = int(lora_rank)
        self.dropout_p = float(dropout_p)
        self.q_proj = nn.Linear(int(model_dim), self.num_heads * self.head_dim, bias=False)
        self.kv_a = nn.Linear(int(model_dim), self.kv_lora_rank + self.rope_dim, bias=False)
        self.kv_b = nn.Linear(self.kv_lora_rank, self.num_heads * (self.nope_dim + self.v_head_dim), bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.v_head_dim, int(model_dim), bias=False)
        inv_freq = TileCudaRotaryEmbeddingStage._compute_inv_freq(self.rope_dim, float(rope_base), None)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        h = self.num_heads
        q = tile_linear_module(x, self.q_proj.weight, None, self.config, name="multi_latent_attention.q_proj")
        q = q.view(batch, seq_len, h, self.head_dim)
        q_nope, q_rope = q.split([self.nope_dim, self.rope_dim], dim=-1)
        kv = tile_linear_module(x, self.kv_a.weight, None, self.config, name="multi_latent_attention.kv_a")
        kv_c, k_rope = kv.split([self.kv_lora_rank, self.rope_dim], dim=-1)
        eps = torch.finfo(torch.float32).eps
        kv_c = tile_rms_norm_module(kv_c.contiguous(), eps, self.config)
        kv = tile_linear_module(kv_c, self.kv_b.weight, None, self.config, name="multi_latent_attention.kv_b")
        kv = kv.view(batch, seq_len, h, self.nope_dim + self.v_head_dim)
        k_nope, v = kv.split([self.nope_dim, self.v_head_dim], dim=-1)
        q_rope = q_rope.transpose(1, 2).contiguous()
        k_rope_h = k_rope.unsqueeze(2).expand(batch, seq_len, h, self.rope_dim).transpose(1, 2).contiguous()
        q_rope, k_rope_h = tile_rotary_embedding_module(q_rope, k_rope_h, self.inv_freq.to(device=x.device), self.config)
        q_full = torch.cat([q_nope.transpose(1, 2).contiguous(), q_rope], dim=-1)
        k_full = torch.cat([k_nope.transpose(1, 2).contiguous(), k_rope_h], dim=-1)
        v = v.transpose(1, 2).contiguous()
        drop = self.dropout_p if self.training else 0.0
        if drop != 0.0:
            attn = F.scaled_dot_product_attention(q_full, k_full, v, dropout_p=drop, is_causal=True)
        else:
            attn = tile_scaled_dot_product_attention_module(q_full.contiguous(), k_full.contiguous(), v, True, self.config)
        out = attn.transpose(1, 2).contiguous().reshape(batch, seq_len, h * self.v_head_dim)
        return tile_linear_module(out, self.out_proj.weight, None, self.config, name="multi_latent_attention.out_proj")


class TileCudaRoutedAttentionExpertsStage(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        experts: int,
        top_k: int = 2,
        is_causal: bool = True,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        head_dim = model_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        kv_dim = int(num_kv_heads) * head_dim
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.experts = int(experts)
        self.top_k = int(top_k)
        self.is_causal = bool(is_causal)
        self.q_proj = nn.Parameter(torch.empty(self.experts, self.model_dim, self.model_dim))
        self.k_proj = nn.Parameter(torch.empty(self.experts, self.model_dim, kv_dim))
        self.v_proj = nn.Parameter(torch.empty(self.experts, self.model_dim, kv_dim))
        self.out_proj = nn.Parameter(torch.empty(self.experts, self.model_dim, self.model_dim))
        self.q_gain = nn.Parameter(torch.full((self.experts, self.num_heads), float(qk_gain_init), dtype=torch.float32))
        inv_freq = TileCudaRotaryEmbeddingStage._compute_inv_freq(self.head_dim, float(rope_base), None)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        nn.init.normal_(self.q_proj, std=0.02)
        nn.init.normal_(self.k_proj, std=0.02)
        nn.init.normal_(self.v_proj, std=0.02)
        nn.init.normal_(self.out_proj, std=0.02)
        self.config = config or TileCudaConfig()

    def _expert_attention(self, x: Tensor, expert_idx: int) -> Tensor:
        batch, seq_len, model_dim = x.shape
        q = tile_linear_module(x, self.q_proj[expert_idx].t().contiguous(), None, self.config, name="routed_attention_experts.q_proj")
        k = tile_linear_module(x, self.k_proj[expert_idx].t().contiguous(), None, self.config, name="routed_attention_experts.k_proj")
        v = tile_linear_module(x, self.v_proj[expert_idx].t().contiguous(), None, self.config, name="routed_attention_experts.v_proj")
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        eps = torch.finfo(torch.float32).eps
        q = tile_rms_norm_module(q, eps, self.config)
        k = tile_rms_norm_module(k, eps, self.config)
        q, k = tile_rotary_embedding_module(q.contiguous(), k.contiguous(), self.inv_freq.to(device=x.device), self.config)
        q = tile_qk_gain_module(q.contiguous(), self.q_gain[expert_idx].contiguous(), self.config)
        y = tile_scaled_dot_product_attention_module(q.contiguous(), k.contiguous(), v.contiguous(), self.is_causal, self.config)
        y = y.transpose(1, 2).contiguous().reshape(batch, seq_len, model_dim)
        return tile_linear_module(y, self.out_proj[expert_idx].t().contiguous(), None, self.config, name="routed_attention_experts.out_proj")

    def forward(self, x: Tensor, routing_weights: Tensor, routing_indices: Tensor) -> Tensor:
        if routing_weights.ndim == 3:
            routing_weights = routing_weights.squeeze(1)
        if routing_indices.ndim == 3:
            routing_indices = routing_indices.squeeze(1)
        out = torch.zeros_like(x)
        for expert_idx in range(self.experts):
            mask = routing_indices == expert_idx
            batch_idx, slot_idx = torch.where(mask)
            if batch_idx.numel() == 0:
                continue
            expert_inputs = x[batch_idx].contiguous()
            expert_out = self._expert_attention(expert_inputs, expert_idx)
            weights = routing_weights[batch_idx, slot_idx].to(dtype=x.dtype).view(-1, 1, 1)
            out[batch_idx] += expert_out * weights
        return out


class TileCudaMambaStage(nn.Module):
    def __init__(
        self,
        model_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.d_inner = int(model_dim) * int(expand)
        self.in_proj = nn.Linear(self.model_dim, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=int(d_conv),
            groups=self.d_inner,
            padding=int(d_conv) - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, int(d_state) + 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, self.model_dim, bias=False)
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        xz = tile_linear_module(x, self.in_proj.weight, None, self.config, name="mamba.in_proj")
        x_part, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_part.transpose(1, 2))[:, :, :seq_len].transpose(1, 2).contiguous()
        x_conv = tile_unary("silu", x_conv, self.config)
        gate = tile_unary("sigmoid", z.contiguous(), self.config)
        y = tile_binary("multiply", x_conv, gate, self.config)
        return tile_linear_module(y, self.out_proj.weight, None, self.config, name="mamba.out_proj")


class TileCudaUniversalTransformerStage(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        mlp_mult: float,
        max_steps: int,
        halt_epsilon: float,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        hidden_dim = max(int(model_dim * mlp_mult), int(model_dim))
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.model_dim // self.num_heads
        self.max_steps = int(max_steps)
        self.halt_epsilon = float(halt_epsilon)
        self.attn_norm = nn.LayerNorm(self.model_dim)
        self.attn = nn.MultiheadAttention(self.model_dim, self.num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(self.model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.model_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.model_dim, bias=False),
        )
        self.halt_gate = nn.Module()
        self.halt_gate.proj = nn.Linear(self.model_dim, 1)
        self.config = config or TileCudaConfig()

    def _self_attention(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        in_weight = self.attn.in_proj_weight
        in_bias = self.attn.in_proj_bias
        q_w, k_w, v_w = in_weight.chunk(3, dim=0)
        q_b, k_b, v_b = in_bias.chunk(3, dim=0) if in_bias is not None else (None, None, None)
        q = tile_linear_module(x, q_w.contiguous(), q_b.contiguous() if q_b is not None else None, self.config, name="universal_transformer.q_proj")
        k = tile_linear_module(x, k_w.contiguous(), k_b.contiguous() if k_b is not None else None, self.config, name="universal_transformer.k_proj")
        v = tile_linear_module(x, v_w.contiguous(), v_b.contiguous() if v_b is not None else None, self.config, name="universal_transformer.v_proj")
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        y = tile_scaled_dot_product_attention_module(q, k, v, False, self.config)
        y = y.transpose(1, 2).contiguous().reshape(batch, seq_len, self.model_dim)
        return tile_linear_module(
            y,
            self.attn.out_proj.weight,
            self.attn.out_proj.bias,
            self.config,
            name="universal_transformer.out_proj",
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        state = x
        batch = x.size(0)
        remaining = torch.ones(batch, 1, device=x.device, dtype=torch.float32)
        accum = torch.zeros_like(x)
        weights: list[Tensor] = []

        for step in range(self.max_steps):
            attn_in = tile_layer_norm_module(
                state.contiguous(),
                self.attn_norm.weight,
                self.attn_norm.bias,
                float(self.attn_norm.eps),
                self.config,
            )
            state = state + self._self_attention(attn_in.contiguous())
            mlp_in = tile_layer_norm_module(
                state.contiguous(),
                self.mlp_norm.weight,
                self.mlp_norm.bias,
                float(self.mlp_norm.eps),
                self.config,
            )
            hidden = tile_linear_module(mlp_in, self.mlp[0].weight, None, self.config, name="universal_transformer.mlp_in")
            hidden = tile_unary("silu", hidden.contiguous(), self.config)
            state = state + tile_linear_module(hidden, self.mlp[2].weight, None, self.config, name="universal_transformer.mlp_out")

            pooled = state.mean(dim=1).contiguous()
            raw_p = tile_linear_module(
                pooled,
                self.halt_gate.proj.weight,
                self.halt_gate.proj.bias,
                self.config,
                name="universal_transformer.halt_gate",
            )
            raw_p = tile_unary("sigmoid", raw_p.contiguous(), self.config).float()
            step_p = torch.minimum(raw_p, remaining)
            if step == self.max_steps - 1:
                step_p = remaining
            if self.halt_epsilon > 0.0:
                step_p = torch.where(remaining <= self.halt_epsilon, remaining, step_p)

            accum = accum + step_p.to(dtype=state.dtype).unsqueeze(-1) * state
            remaining = (remaining - step_p).clamp_min(0.0)
            weights.append(step_p.squeeze(-1))

        return accum, torch.stack(weights, dim=1)


class TileCudaRotaryEmbeddingStage(nn.Module):
    def __init__(
        self,
        head_dim: int,
        rope_base: float,
        rope_scaling: dict[str, object] | None = None,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        inv_freq = self._compute_inv_freq(int(head_dim), float(rope_base), rope_scaling)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.config = config or TileCudaConfig()

    @staticmethod
    def _compute_inv_freq(dim: int, base: float, scaling: dict[str, object] | None) -> Tensor:
        idx = torch.arange(0, dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (idx / dim))
        if not scaling:
            return inv_freq
        stype = str(scaling.get("type", scaling.get("rope_type", "linear"))).lower()
        factor = float(scaling.get("factor", 1.0)) or 1.0
        if stype in ("linear", "pi"):
            return inv_freq / factor
        if stype in ("ntk", "ntk-aware", "dynamic"):
            base2 = base * (factor ** (dim / max(dim - 2, 1)))
            return 1.0 / (base2 ** (idx / dim))
        if stype == "yarn":
            orig = float(scaling.get("original_max_position", scaling.get("original_max_position_embeddings", 2048)))
            beta_fast = float(scaling.get("beta_fast", 32.0))
            beta_slow = float(scaling.get("beta_slow", 1.0))

            def _corr_dim(num_rot: float) -> float:
                return (dim * math.log(orig / (num_rot * 2 * math.pi))) / (2 * math.log(base))

            low = max(math.floor(_corr_dim(beta_fast)), 0)
            high = min(math.ceil(_corr_dim(beta_slow)), dim // 2 - 1)
            if high <= low:
                high = low + 1
            ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0.0, 1.0)
            return inv_freq * (1.0 - ramp) + (inv_freq / factor) * ramp
        return inv_freq

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        return tile_rotary_embedding_module(q, k, self.inv_freq.to(device=q.device), self.config)


class TileCudaRMSNormStage(nn.Module):
    def __init__(self, eps: float = 1e-6, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.eps = float(eps)
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        return tile_rms_norm_module(x, self.eps, self.config)


class TileCudaQKNormStage(nn.Module):
    def __init__(self, eps: float = 1e-6, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.eps = float(eps)
        self.config = config or TileCudaConfig()

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        return tile_rms_norm_module(q, self.eps, self.config), tile_rms_norm_module(k, self.eps, self.config)


class TileCudaLayerNormStage(nn.Module):
    def __init__(self, model_dim: int, eps: float = 1e-5, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(model_dim, eps=eps)
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        return tile_layer_norm_module(x, self.norm.weight, self.norm.bias, float(self.norm.eps), self.config)


class TileCudaGroupNormStage(nn.Module):
    def __init__(self, model_dim: int, num_groups: int = 1, eps: float = 1e-5, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        groups = max(int(num_groups), 1)
        while int(model_dim) % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, int(model_dim), eps=float(eps))
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        return tile_group_norm_module(x, self.norm.weight, self.norm.bias, int(self.norm.num_groups), float(self.norm.eps), self.config)


class TileCudaLinearStage(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = False, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=bias)
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        return tile_linear_module(x, self.proj.weight, self.proj.bias, self.config, name="linear")


class TileCudaLMHeadStage(nn.Module):
    def __init__(self, model_dim: int, vocab_size: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.proj = nn.Linear(model_dim, vocab_size, bias=False)
        self.config = config or TileCudaConfig()

    def forward(self, hidden: Tensor) -> Tensor:
        return tile_linear_module(hidden, self.proj.weight, None, self.config, name="lm_head")


class TileCudaTiedLMHeadStage(nn.Module):
    def __init__(self, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.config = config or TileCudaConfig()

    def forward(self, hidden: Tensor, tied_weight: Tensor) -> Tensor:
        return tile_linear_module(hidden, tied_weight, None, self.config, name="tied_lm_head")


class TileCudaRouterLogitsStage(nn.Module):
    def __init__(self, model_dim: int, experts: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.gate = nn.Linear(model_dim, experts, bias=False)
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        return tile_linear_module(x, self.gate.weight, None, self.config, name="router_logits")


class TileCudaValueHeadStage(nn.Module):
    def __init__(self, model_dim: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.proj = nn.Linear(int(model_dim), 1, bias=False)
        self.config = config or TileCudaConfig()

    def forward(self, hidden: Tensor) -> Tensor:
        return tile_linear_module(hidden, self.proj.weight, None, self.config, name="value_head").squeeze(-1)


class TileCudaRewardHeadStage(nn.Module):
    def __init__(self, model_dim: int, pool: str = "last", config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.pool = str(pool)
        self.proj = nn.Linear(int(model_dim), 1, bias=False)
        self.config = config or TileCudaConfig()

    def forward(self, hidden: Tensor) -> Tensor:
        pooled = hidden.mean(dim=1) if self.pool == "mean" else hidden[:, -1, :]
        return tile_linear_module(pooled.contiguous(), self.proj.weight, None, self.config, name="reward_head").squeeze(-1)


class TileCudaDenoiseHeadStage(nn.Module):
    def __init__(self, model_dim: int, vocab_size: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.proj = nn.Linear(model_dim, vocab_size, bias=False)
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        return tile_linear_module(x, self.proj.weight, None, self.config, name="denoise_head")


class TileCudaKVPCAEncodeStage(nn.Module):
    def __init__(self, head_dim: int, compressed_dim: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.k_proj = nn.Linear(int(head_dim), int(compressed_dim), bias=False)
        self.v_proj = nn.Linear(int(head_dim), int(compressed_dim), bias=False)
        self.config = config or TileCudaConfig()

    def forward(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return (
            tile_linear_module(k, self.k_proj.weight, None, self.config, name="kv_pca_encode.k"),
            tile_linear_module(v, self.v_proj.weight, None, self.config, name="kv_pca_encode.v"),
        )


class TileCudaKVPCADecodeStage(nn.Module):
    def __init__(self, head_dim: int, compressed_dim: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.k_unproj = nn.Linear(int(compressed_dim), int(head_dim), bias=False)
        self.v_unproj = nn.Linear(int(compressed_dim), int(head_dim), bias=False)
        self.config = config or TileCudaConfig()

    def forward(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return (
            tile_linear_module(k, self.k_unproj.weight, None, self.config, name="kv_pca_decode.k"),
            tile_linear_module(v, self.v_unproj.weight, None, self.config, name="kv_pca_decode.v"),
        )


class TileCudaJEPAProjectorStage(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(latent_dim), bias=False),
            nn.LayerNorm(int(latent_dim)),
            nn.GELU(),
            nn.Linear(int(latent_dim), int(latent_dim), bias=False),
        )
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        h = tile_linear_module(x, self.net[0].weight, None, self.config, name="jepa_projector.in")
        h = tile_layer_norm_module(h, self.net[1].weight, self.net[1].bias, float(self.net[1].eps), self.config)
        h = tile_unary("gelu", h, self.config)
        return tile_linear_module(h, self.net[3].weight, None, self.config, name="jepa_projector.out")


class TileCudaJEPAPredictorStage(nn.Module):
    def __init__(self, latent_dim: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        hidden_dim = max(int(latent_dim) // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(int(latent_dim), hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, int(latent_dim), bias=False),
        )
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        h = tile_linear_module(x, self.net[0].weight, None, self.config, name="jepa_predictor.in")
        h = tile_unary("gelu", h, self.config)
        return tile_linear_module(h, self.net[2].weight, None, self.config, name="jepa_predictor.out")


class TileCudaTTTLinearStage(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 16, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.weight = nn.Parameter(torch.randn(self.output_dim, self.input_dim) / math.sqrt(self.input_dim))
        self.ttt_down = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.ttt_up = nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        base = tile_linear_module(x, self.weight, None, self.config, name="ttt_linear.base")
        ttt = tile_linear_module(x, self.ttt_down.weight, None, self.config, name="ttt_linear.down")
        ttt = tile_unary("tanh_neuron", ttt, self.config)
        ttt = tile_linear_module(ttt, self.ttt_up.weight, None, self.config, name="ttt_linear.up")
        return tile_binary("add", base, ttt, self.config)


class TileCudaLoRALinearStage(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = False,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rank = max(int(rank), 1)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.base = nn.Linear(self.input_dim, self.output_dim, bias=bool(bias))
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.input_dim))
        self.lora_B = nn.Parameter(torch.zeros(self.output_dim, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        base = tile_linear_module(x, self.base.weight, self.base.bias, self.config, name="lora_linear.base")
        lora = tile_linear_module(x, self.lora_A, None, self.config, name="lora_linear.A")
        lora = tile_linear_module(lora, self.lora_B, None, self.config, name="lora_linear.B")
        return tile_scalar_binary_module("aux_loss_add", base, lora, self.scaling, self.config)


class TileCudaBitLinearTernaryStage(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.weight = nn.Parameter(torch.randn(self.output_dim, self.input_dim))
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        scale = self.weight.abs().mean()
        w_quant = torch.round(self.weight / (scale + 1e-7)).clamp(-1, 1)
        w_quant = self.weight + (w_quant - self.weight).detach()

        x_max = x.abs().max(dim=-1, keepdim=True).values
        x_quant = torch.round(x * 127 / (x_max + 1e-7)).clamp(-128, 127)
        x_quant = x + (x_quant * x_max / 127 - x).detach()
        return tile_linear_module(x_quant, w_quant, None, self.config, name="bitlinear_ternary")


class TileCudaFP8LinearStage(nn.Module):
    _FMAX = {"e4m3": 448.0, "e5m2": 57344.0}

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = False,
        fp8_format: str = "e4m3",
        amax_history_len: int = 16,
        use_stochastic_rounding: bool = True,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.fp8_format = "e5m2" if str(fp8_format) == "e5m2" else "e4m3"
        self._fp8_dtype = torch.float8_e5m2 if self.fp8_format == "e5m2" else torch.float8_e4m3fn
        self._fmax = self._FMAX[self.fp8_format]
        self.use_stochastic_rounding = bool(use_stochastic_rounding)
        self.weight = nn.Parameter(torch.empty(self.output_dim, self.input_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(self.output_dim)) if bias else None
        self.register_buffer(
            "amax_history", torch.zeros(max(int(amax_history_len), 1), dtype=torch.float32), persistent=False
        )
        self.config = config or TileCudaConfig()

    def _quant_weight(self, weight: Tensor) -> Tensor:
        amax = weight.detach().abs().amax().clamp(min=1e-12)
        if self.training:
            self.amax_history = torch.roll(self.amax_history, 1)
            self.amax_history[0] = amax.to(self.amax_history.dtype)
        scale = amax / self._fmax
        weight_q = (weight / scale).to(self._fp8_dtype).to(weight.dtype) * scale
        return weight + (weight_q - weight).detach()

    def forward(self, x: Tensor) -> Tensor:
        return tile_linear_module(x, self._quant_weight(self.weight), self.bias, self.config, name="fp8_linear")


class TileCudaMXLinearStage(nn.Module):
    _FP4_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    _EMAX = {"mxfp4": 6.0, "mxfp8": 448.0}

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = False,
        mx_format: str = "mxfp4",
        mx_block_size: int = 32,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.mx_format = "mxfp8" if str(mx_format) == "mxfp8" else "mxfp4"
        self.block = max(int(mx_block_size), 1)
        self.weight = nn.Parameter(torch.empty(self.output_dim, self.input_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(self.output_dim)) if bias else None
        mag = torch.tensor(self._FP4_MAG, dtype=torch.float32)
        grid = torch.cat([-mag.flip(0), mag[1:]])
        self.register_buffer("fp4_grid", grid, persistent=False)
        self.config = config or TileCudaConfig()

    def _quant_weight(self, weight: Tensor) -> Tensor:
        out_dim, in_dim = weight.shape
        pad = (self.block - in_dim % self.block) % self.block
        padded = F.pad(weight, (0, pad))
        blocked = padded.reshape(out_dim, -1, self.block)
        amax = blocked.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        emax = self._EMAX[self.mx_format]
        exp = torch.floor(torch.log2(amax / emax)).clamp(-127, 127)
        scale = torch.exp2(exp)
        normalized = blocked / scale
        if self.mx_format == "mxfp8":
            quantized = normalized.to(torch.float8_e4m3fn).to(weight.dtype)
        else:
            grid = self.fp4_grid.to(weight.dtype)
            idx = (normalized.unsqueeze(-1) - grid).abs().argmin(dim=-1)
            quantized = grid[idx]
        weight_q = (quantized * scale).reshape(out_dim, -1)[:, :in_dim]
        return weight + (weight_q - weight).detach()

    def forward(self, x: Tensor) -> Tensor:
        return tile_linear_module(x, self._quant_weight(self.weight), self.bias, self.config, name="mx_linear")


class TileCudaNF4LinearStage(nn.Module):
    _NF4_CODEBOOK = (
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    )

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = False,
        group_size: int = 64,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rank = max(int(rank), 1)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.group_size = max(int(group_size), 1)
        self.compute_dtype = "fp32"
        self._compute_dtype = torch.float32

        packed_cols = (self.input_dim + 1) // 2
        self.register_buffer("qweight", torch.zeros(self.output_dim, packed_cols, dtype=torch.uint8))
        num_groups = (self.input_dim + self.group_size - 1) // self.group_size
        self.register_buffer("absmax", torch.ones(self.output_dim, num_groups, dtype=torch.float32))
        self.register_buffer("nf4_codebook", torch.tensor(self._NF4_CODEBOOK, dtype=torch.float32))

        self.bias = nn.Parameter(torch.zeros(self.output_dim)) if bias else None
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.input_dim))
        self.lora_B = nn.Parameter(torch.zeros(self.output_dim, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.config = config or TileCudaConfig()

    @classmethod
    def _quantize_nf4(cls, weight: Tensor, group_size: int) -> tuple[Tensor, Tensor]:
        codebook = torch.tensor(cls._NF4_CODEBOOK, dtype=torch.float32, device=weight.device)
        out_dim, in_dim = weight.shape
        num_groups = (in_dim + group_size - 1) // group_size
        absmax = torch.zeros(out_dim, num_groups, dtype=torch.float32, device=weight.device)
        indices = torch.zeros(out_dim, in_dim, dtype=torch.uint8, device=weight.device)
        for group_idx in range(num_groups):
            start = group_idx * group_size
            end = min(start + group_size, in_dim)
            block = weight[:, start:end].float()
            block_absmax = block.abs().amax(dim=-1).clamp(min=1e-8)
            absmax[:, group_idx] = block_absmax
            normalized = block / block_absmax.unsqueeze(-1)
            diffs = (normalized.unsqueeze(-1) - codebook.view(1, 1, -1)).abs()
            indices[:, start:end] = diffs.argmin(dim=-1).to(torch.uint8)
        packed_cols = (in_dim + 1) // 2
        packed = torch.zeros(out_dim, packed_cols, dtype=torch.uint8, device=weight.device)
        even = indices[:, 0::2]
        odd = indices[:, 1::2] if indices.shape[1] > 1 else torch.zeros_like(even[:, :0])
        packed[:, : even.shape[1]] = even & 0x0F
        if odd.shape[1] > 0:
            packed[:, : odd.shape[1]] |= (odd & 0x0F) << 4
        return packed, absmax

    def _dequantize_weight(self) -> Tensor:
        even_codes = (self.qweight & 0x0F).long()
        odd_codes = ((self.qweight >> 4) & 0x0F).long()
        interleaved = torch.empty(
            self.output_dim,
            self.qweight.shape[1] * 2,
            dtype=torch.long,
            device=self.qweight.device,
        )
        interleaved[:, 0::2] = even_codes
        if self.qweight.shape[1] * 2 > 1:
            interleaved[:, 1::2] = odd_codes
        codes = interleaved[:, : self.input_dim]
        normalized = self.nf4_codebook[codes]
        scales_per_col = self.absmax.repeat_interleave(self.group_size, dim=-1)[:, : self.input_dim]
        return (normalized * scales_per_col).to(self._compute_dtype)

    def load_base_weight(self, weight: Tensor) -> None:
        weight = weight.to(self.qweight.device).detach()
        if weight.shape != (self.output_dim, self.input_dim):
            raise ValueError(
                f"NF4LinearStage expected base weight shape {(self.output_dim, self.input_dim)}, got {tuple(weight.shape)}"
            )
        packed, absmax = self._quantize_nf4(weight, self.group_size)
        self.qweight.copy_(packed)
        self.absmax.copy_(absmax)

    def forward(self, x: Tensor) -> Tensor:
        base = tile_linear_module(x.to(torch.float32), self._dequantize_weight(), self.bias, self.config, name="nf4_linear.base")
        lora = tile_linear_module(x, self.lora_A, None, self.config, name="nf4_linear.A")
        lora = tile_linear_module(lora, self.lora_B, None, self.config, name="nf4_linear.B")
        return tile_scalar_binary_module("aux_loss_add", base, lora.to(base.dtype), self.scaling, self.config)


class TileCudaRandMapAdapterStage(nn.Module):
    def __init__(self, model_dim: int, adapter_dim: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.adapter_dim = int(adapter_dim)
        self.down_proj = nn.Linear(self.model_dim, self.adapter_dim, bias=False)
        self.up_proj = nn.Linear(self.adapter_dim, self.model_dim, bias=False)
        for param in self.down_proj.parameters():
            param.requires_grad = False
        for param in self.up_proj.parameters():
            param.requires_grad = False
        self.middle = nn.Linear(self.adapter_dim, self.adapter_dim, bias=False)
        self.scale = nn.Parameter(torch.zeros(1))
        nn.init.orthogonal_(self.down_proj.weight)
        nn.init.orthogonal_(self.up_proj.weight)
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        adapter = tile_linear_module(x, self.down_proj.weight, None, self.config, name="randmap_adapter.down")
        adapter = tile_linear_module(adapter, self.middle.weight, None, self.config, name="randmap_adapter.middle")
        adapter = tile_linear_module(adapter, self.up_proj.weight, None, self.config, name="randmap_adapter.up")
        return tile_scaled_residual_add_module(x, adapter, self.scale, self.config)


class TileCudaMLPReluSquaredStage(nn.Module):
    def __init__(self, model_dim: int, mlp_mult: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        hidden = int(model_dim) * int(mlp_mult)
        self.fc = nn.Linear(int(model_dim), hidden, bias=False)
        self.proj = nn.Linear(hidden, int(model_dim), bias=False)
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        h = tile_linear_module(x, self.fc.weight, None, self.config, name="mlp_relu2.fc")
        h = tile_unary("relu", h, self.config)
        h = tile_binary("multiply", h, h, self.config)
        return tile_linear_module(h, self.proj.weight, None, self.config, name="mlp_relu2.proj")


class TileCudaSwiGLUStage(nn.Module):
    def __init__(self, model_dim: int, mlp_mult: int, multiple_of: int | None = None, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        hidden = int(8.0 * int(model_dim) / 3.0)
        if multiple_of is not None:
            multiple = int(multiple_of)
            hidden = multiple * ((hidden + multiple - 1) // multiple)
        self.w1 = nn.Linear(int(model_dim), hidden, bias=False)
        self.w2 = nn.Linear(hidden, int(model_dim), bias=False)
        self.w3 = nn.Linear(int(model_dim), hidden, bias=False)
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        gate = tile_unary("silu", tile_linear_module(x, self.w1.weight, None, self.config, name="swiglu.w1"), self.config)
        value = tile_linear_module(x, self.w3.weight, None, self.config, name="swiglu.w3")
        hidden = tile_binary("multiply", gate, value, self.config)
        return tile_linear_module(hidden, self.w2.weight, None, self.config, name="swiglu.w2")


class TileCudaGLUStage(nn.Module):
    def __init__(
        self,
        model_dim: int,
        mlp_mult: int,
        multiple_of: int | None = None,
        activation: str = "gelu",
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        hidden = int(8.0 * int(model_dim) / 3.0)
        if multiple_of is not None:
            multiple = int(multiple_of)
            hidden = multiple * ((hidden + multiple - 1) // multiple)
        self.activation = str(activation)
        self.w1 = nn.Linear(int(model_dim), hidden, bias=False)
        self.w2 = nn.Linear(hidden, int(model_dim), bias=False)
        self.w3 = nn.Linear(int(model_dim), hidden, bias=False)
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        gate = tile_linear_module(x, self.w1.weight, None, self.config, name=f"{self.activation}.w1")
        if self.activation == "relu":
            gate = tile_unary("relu", gate, self.config)
        elif self.activation == "gelu":
            gate = tile_unary("gelu", gate, self.config)
        else:
            gate = tile_softmax_lastdim_module(gate, self.config)
        value = tile_linear_module(x, self.w3.weight, None, self.config, name=f"{self.activation}.w3")
        hidden = tile_binary("multiply", gate, value, self.config)
        return tile_linear_module(hidden, self.w2.weight, None, self.config, name=f"{self.activation}.w2")


class TileCudaACTWeightedSumStage(nn.Module):
    def __init__(self, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.config = config or TileCudaConfig()

    def forward(self, states: Tensor, weights: Tensor) -> Tensor:
        return tile_act_weighted_sum_module(states, weights, self.config)


class TileCudaACTHaltGateStage(nn.Module):
    def __init__(self, model_dim: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.proj = nn.Linear(int(model_dim), 1)
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        pooled = x.mean(dim=1).contiguous()
        logits = tile_linear_module(pooled, self.proj.weight, self.proj.bias, self.config, name="act_halt_gate")
        return tile_unary("sigmoid", logits, self.config)


class TileCudaPreferenceBCELossStage(nn.Module):
    def __init__(self, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.config = config or TileCudaConfig()

    def forward(self, reward_chosen: Tensor, reward_rejected: Tensor) -> Tensor:
        return tile_preference_bce_loss_module(reward_chosen, reward_rejected, self.config)


class TileCudaPPOClippedLossStage(nn.Module):
    def __init__(
        self,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        self.clip_range = float(clip_range)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)
        self.config = config or TileCudaConfig()

    def forward(
        self,
        logp_new: Tensor,
        logp_old: Tensor,
        advantages: Tensor,
        value_new: Tensor,
        value_old: Tensor,
        returns: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return tile_ppo_clipped_loss_module(
            logp_new,
            logp_old,
            advantages,
            value_new,
            value_old,
            returns,
            self.clip_range,
            self.vf_coef,
            self.config,
        )


class TileCudaGAEComputeStage(nn.Module):
    def __init__(self, gamma: float = 1.0, lambda_: float = 0.95, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.lam = float(lambda_)
        self.config = config or TileCudaConfig()

    def forward(self, rewards: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        return tile_gae_compute_module(rewards, values, self.gamma, self.lam, self.config)


class TileCudaDPOPairwiseLossStage(nn.Module):
    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.label_smoothing = float(label_smoothing)
        self.loss_type = str(loss_type)
        self.config = config or TileCudaConfig()

    def forward(
        self,
        policy_logp_chosen: Tensor,
        policy_logp_rejected: Tensor,
        ref_logp_chosen: Tensor,
        ref_logp_rejected: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return tile_dpo_pairwise_loss_module(
            policy_logp_chosen,
            policy_logp_rejected,
            ref_logp_chosen,
            ref_logp_rejected,
            self.beta,
            self.label_smoothing,
            self.loss_type,
            self.config,
        )


class TileCudaRouteBalanceLossStage(nn.Module):
    def __init__(self, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.config = config or TileCudaConfig()

    def forward(self, route_logits: Tensor) -> Tensor:
        return tile_route_balance_loss_module(route_logits, self.config)


class TileCudaRouteSelectionLossStage(nn.Module):
    def __init__(
        self,
        semantic_vocab_ref: str = "",
        shared_experts: int = 2,
        free_experts: int = 8,
        ignore_index: int = -100,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        from ..semantic import ConversationalVocabulary, NUM_SEMANTIC_DIMS, semantic_vocab_ref_for_dim

        vocab = ConversationalVocabulary(semantic_vocab_ref or semantic_vocab_ref_for_dim(NUM_SEMANTIC_DIMS))
        self.num_vocab_dims = int(vocab.num_vocab_dims)
        self.shared_experts = max(int(shared_experts), 0)
        self.free_experts = max(int(free_experts), 0)
        self.ignore_index = int(ignore_index)
        self.config = config or TileCudaConfig()

    def forward(self, route_logits: Tensor, sem_targets: Tensor) -> Tensor:
        return tile_route_selection_loss_module(
            route_logits,
            sem_targets,
            self.num_vocab_dims,
            self.shared_experts,
            self.ignore_index,
            self.config,
        )


class TileCudaRouteDistillationLossStage(nn.Module):
    def __init__(
        self,
        semantic_vocab_ref: str = "",
        shared_experts: int = 2,
        free_experts: int = 8,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        from ..semantic import ConversationalVocabulary, NUM_SEMANTIC_DIMS, semantic_vocab_ref_for_dim

        vocab = ConversationalVocabulary(semantic_vocab_ref or semantic_vocab_ref_for_dim(NUM_SEMANTIC_DIMS))
        self.num_vocab_dims = int(vocab.num_vocab_dims)
        self.shared_experts = max(int(shared_experts), 0)
        self.free_experts = max(int(free_experts), 0)
        self.total_experts = self.shared_experts + self.num_vocab_dims + self.free_experts
        self.term_counts = [len(vocab.terms(dim_name)) for dim_name in vocab.dim_names]
        self.config = config or TileCudaConfig()

    def forward(self, student_route_logits: Tensor, target_topic_logits: Tensor) -> Tensor:
        student = student_route_logits.float()
        if student.ndim == 2:
            student = student.unsqueeze(1)
        target_logits = target_topic_logits.float()
        if target_logits.ndim == 3:
            target_logits = target_logits.unsqueeze(1)
        target_scores = torch.zeros(
            *target_logits.shape[:2],
            self.num_vocab_dims,
            device=target_logits.device,
            dtype=torch.float32,
        )
        for dim_idx, term_count in enumerate(self.term_counts):
            count = min(int(term_count), target_logits.size(-1))
            if count <= 0 or dim_idx >= target_logits.size(-2):
                continue
            dim_logits = target_logits[..., dim_idx, :count].contiguous()
            if dim_logits.is_cuda and dim_logits.dtype == torch.float32 and dim_logits.size(-1) <= 1024:
                probs = tile_softmax_lastdim_module(dim_logits, self.config)
            else:
                probs = F.softmax(dim_logits.float(), dim=-1)
            target_scores[..., dim_idx] = probs.max(dim=-1).values
        teacher_logits = student.new_full(student.shape, -10.0)
        if self.shared_experts > 0:
            teacher_logits[..., : self.shared_experts] = 0.0
        semantic_end = self.shared_experts + self.num_vocab_dims
        teacher_logits[..., self.shared_experts : semantic_end] = target_scores
        return tile_softmax_distillation_loss_module(
            teacher_logits.detach().reshape(-1, teacher_logits.size(-1)).contiguous(),
            student.reshape(-1, student.size(-1)).contiguous(),
            self.config,
        )


class TileCudaSemanticAlignmentLossStage(nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        semantic_vocab_ref: str = "",
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        from ..semantic import ConversationalVocabulary, semantic_vocab_ref_for_dim

        vocab_ref = semantic_vocab_ref or semantic_vocab_ref_for_dim()
        vocab = ConversationalVocabulary(vocab_ref or None)
        self.term_counts = tuple(len(vocab.terms(dim_name)) for dim_name in vocab.dim_names)
        self.ignore_index = int(ignore_index)
        self.config = config or TileCudaConfig()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return tile_semantic_alignment_loss_module(pred, target, self.term_counts, self.ignore_index, self.config)


class TileCudaSemanticHasherStage(nn.Module):
    def __init__(self, dim: int = 9, tables: int = 8, planes: int = 12, seed: int = 42, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        rng = np.random.RandomState(int(seed))
        proj = torch.from_numpy(rng.randn(int(tables), int(planes), int(dim)).astype("float32"))
        self.register_buffer("proj", proj)
        self.config = config or TileCudaConfig()

    def forward(self, sem_vec: Tensor) -> Tensor:
        return tile_semantic_hash_module(sem_vec.contiguous(), self.proj, self.config)


class TileCudaSemanticChunkHasherStage(nn.Module):
    def __init__(self, dim: int = 9, tables: int = 8, planes: int = 12, seed: int = 42, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        rng = np.random.RandomState(int(seed))
        proj = torch.from_numpy(rng.randn(int(tables), int(planes), int(dim)).astype("float32"))
        self.register_buffer("proj", proj)
        self.config = config or TileCudaConfig()

    def forward(self, sem_vec: Tensor) -> Tensor:
        if sem_vec.ndim == 2:
            sem_vec = sem_vec.unsqueeze(1)
        if sem_vec.ndim != 3:
            raise ValueError("semantic_chunk_hasher expects [batch, chunks, semantic_dim]")
        batch, chunks, dim = sem_vec.shape
        flat = sem_vec.reshape(batch * chunks, dim).contiguous()
        buckets = tile_semantic_hash_module(flat, self.proj, self.config)
        return buckets.reshape(batch, chunks, -1)


class TileCudaSemanticMoERouterStage(RoutingStatsMixin, nn.Module):
    def __init__(self, n_experts: int, semantic_dim: int = 9, top_k: int = 2, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.centroids = nn.Parameter(torch.randn(int(n_experts), int(semantic_dim)))
        self.top_k = int(top_k)
        self.config = config or TileCudaConfig()
        self._init_routing_stats(num_experts=int(n_experts), top_k=self.top_k)

    def forward(self, sem_vec: Tensor) -> tuple[Tensor, Tensor]:
        work_dtype = sem_vec.dtype if torch.is_floating_point(sem_vec) else self.centroids.dtype
        c = F.normalize(self.centroids.float(), dim=-1).to(dtype=work_dtype)
        s = sem_vec
        if s.ndim == 3:
            s = s.mean(dim=1)
        s = F.normalize(s.float(), dim=-1).to(dtype=work_dtype)
        sim = (s @ c.T).contiguous()
        weights, indices = tile_topk_route_module(sim, self.top_k, self.config)
        self._update_routing_stats(
            scores=torch.softmax(sim.detach().float(), dim=-1),
            routing_weights=weights.detach(),
            routing_indices=indices.detach(),
        )
        return weights.unsqueeze(1).to(dtype=work_dtype), indices.unsqueeze(1)


class TileCudaSemanticHashRouterStage(RoutingStatsMixin, nn.Module):
    def __init__(
        self,
        n_experts: int,
        semantic_dim: int = 9,
        top_k: int = 2,
        tables: int = 8,
        n_buckets: int = 4096,
        ignore_index: int = -100,
        semantic_vocab_ref: str = "",
        routing_source: str = "topic_logits",
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        from ..semantic import ConversationalVocabulary, semantic_vocab_ref_for_dim

        vocab_ref = semantic_vocab_ref or semantic_vocab_ref_for_dim(semantic_dim)
        vocab = ConversationalVocabulary(vocab_ref or None)
        self.num_vocab_dims = vocab.num_vocab_dims
        if int(n_experts) != self.num_vocab_dims:
            raise ValueError(
                f"semantic_hash_router requires exactly {self.num_vocab_dims} experts, got {n_experts}"
            )
        self.semantic_dim = int(semantic_dim)
        self.top_k = max(1, min(int(top_k), self.num_vocab_dims))
        self.n_buckets = max(int(n_buckets), 1)
        self.ignore_index = int(ignore_index)
        self.routing_source = str(routing_source)
        self.term_counts = [len(vocab.terms(dim_name)) for dim_name in vocab.dim_names]
        self.hash_embed = nn.Embedding(self.n_buckets, self.num_vocab_dims)
        self.table_gate = nn.Parameter(torch.zeros(int(tables), dtype=torch.float32))
        self.dimension_bias = nn.Parameter(torch.zeros(self.num_vocab_dims, dtype=torch.float32))
        self.register_buffer("expert_map", torch.arange(self.num_vocab_dims, dtype=torch.long))
        nn.init.zeros_(self.hash_embed.weight)
        self.config = config or TileCudaConfig()
        self._init_routing_stats(num_experts=self.num_vocab_dims, top_k=self.top_k)

    def _auto_scores(self, sem_vec: Tensor, topic_logits: Tensor, bucket_indices: Tensor) -> Tensor:
        if self.routing_source == "semantic_vec":
            vec = sem_vec.float()
            if vec.ndim == 3:
                vec = vec.mean(dim=1)
            if vec.size(-1) < self.num_vocab_dims:
                raise ValueError(
                    f"semantic_hash_router expected semantic_vec width >= {self.num_vocab_dims}, got {vec.size(-1)}"
                )
            return vec[:, : self.num_vocab_dims] + self.dimension_bias

        logits = topic_logits.float()
        batch = logits.size(0)
        scores = torch.zeros(batch, len(self.term_counts), device=logits.device, dtype=torch.float32)
        for dim_idx, term_count in enumerate(self.term_counts):
            count = min(int(term_count), logits.size(-1))
            if count <= 0 or dim_idx >= logits.size(-2):
                continue
            dim_logits = logits[:, dim_idx, :count].contiguous()
            if dim_logits.is_cuda and dim_logits.dtype == torch.float32 and dim_logits.size(-1) <= 1024:
                probs = tile_softmax_lastdim_module(dim_logits, self.config)
            else:
                probs = F.softmax(dim_logits.float(), dim=-1)
            scores[:, dim_idx] = probs.max(dim=-1).values
        if bucket_indices.ndim == 1:
            bucket_indices = bucket_indices.unsqueeze(-1)
        bucket_features = self.hash_embed(bucket_indices.long() % self.n_buckets).float()
        gate_logits = self.table_gate[: bucket_features.size(1)].contiguous()
        if gate_logits.is_cuda and gate_logits.dtype == torch.float32 and gate_logits.size(-1) <= 1024:
            gate = tile_softmax_lastdim_module(gate_logits, self.config)
        else:
            gate = F.softmax(gate_logits.float(), dim=0)
        hash_bias = (bucket_features * gate.view(1, -1, 1)).sum(dim=1)
        return scores + hash_bias + self.dimension_bias

    def forward(
        self,
        sem_vec: Tensor,
        bucket_indices: Tensor,
        topic_logits: Tensor,
        sem_targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        work_dtype = topic_logits.dtype if torch.is_floating_point(topic_logits) else torch.float32
        scores = self._auto_scores(sem_vec, topic_logits, bucket_indices)
        targets = sem_targets.long()
        if targets.ndim == 1:
            targets = targets.unsqueeze(0)
        if targets.size(1) < len(self.term_counts):
            pad = targets.new_full((targets.size(0), len(self.term_counts) - targets.size(1)), self.ignore_index)
            targets = torch.cat([targets, pad], dim=1)
        forced_mask = targets[:, : len(self.term_counts)] != self.ignore_index
        has_forced = forced_mask.any(dim=-1)
        if not bool(has_forced.any()):
            weights, indices = tile_topk_route_module(scores.contiguous(), self.top_k, self.config)
            self._update_routing_stats(
                scores=F.softmax(scores.detach().float(), dim=-1),
                routing_weights=weights.detach(),
                routing_indices=indices.detach(),
            )
            return weights.to(dtype=work_dtype), indices

        neg_inf = scores.new_full((), float("-inf"))
        ordered_forced = torch.argsort(torch.where(forced_mask, scores, neg_inf), dim=-1, descending=True)
        ordered_all = torch.argsort(scores, dim=-1, descending=True)
        ordered = torch.where(has_forced.unsqueeze(-1), ordered_forced, ordered_all)
        k_per_row = torch.where(
            has_forced,
            forced_mask.sum(dim=-1).clamp(min=1, max=self.top_k),
            torch.full((targets.size(0),), self.top_k, device=targets.device, dtype=torch.long),
        )
        chosen_dims = ordered[:, : self.top_k]
        chosen_experts = self.expert_map[chosen_dims]
        chosen_scores = torch.gather(scores, dim=1, index=chosen_dims)
        slot_ids = torch.arange(self.top_k, device=targets.device, dtype=torch.long).unsqueeze(0)
        valid_slots = slot_ids < k_per_row.unsqueeze(-1)
        fallback_experts = chosen_experts[:, :1].expand(-1, self.top_k)
        masked_scores = torch.where(valid_slots, chosen_scores, neg_inf)
        weights = F.softmax(masked_scores, dim=-1).to(dtype=work_dtype)
        weights = torch.where(valid_slots, weights, weights.new_zeros(weights.shape))
        indices = torch.where(valid_slots, chosen_experts, fallback_experts)
        self._update_routing_stats(
            scores=F.softmax(scores.detach().float(), dim=-1),
            routing_weights=weights.detach(),
            routing_indices=indices.detach(),
        )
        return weights, indices


class TileCudaSemanticMoeJepaEvoRouterStage(RoutingStatsMixin, nn.Module):
    def __init__(
        self,
        semantic_dim: int = 9,
        top_k: int = 2,
        shared_experts: int = 2,
        free_experts: int = 8,
        tables: int = 8,
        n_buckets: int = 4096,
        ignore_index: int = -100,
        semantic_vocab_ref: str = "",
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        from ..semantic import ConversationalVocabulary, semantic_vocab_ref_for_dim

        vocab_ref = semantic_vocab_ref or semantic_vocab_ref_for_dim(semantic_dim)
        vocab = ConversationalVocabulary(vocab_ref or None)
        self.num_vocab_dims = vocab.num_vocab_dims
        self.semantic_dim = int(semantic_dim)
        self.shared_experts = max(int(shared_experts), 0)
        self.free_experts = max(int(free_experts), 0)
        self.total_experts = self.shared_experts + self.num_vocab_dims + self.free_experts
        self.top_k = max(1, min(int(top_k), self.num_vocab_dims + self.free_experts))
        self.route_width = self.shared_experts + self.top_k
        self.n_buckets = max(int(n_buckets), 1)
        self.ignore_index = int(ignore_index)
        self.term_counts = [len(vocab.terms(dim_name)) for dim_name in vocab.dim_names]
        hash_width = self.num_vocab_dims + self.free_experts
        self.hash_embed = nn.Embedding(self.n_buckets, hash_width)
        self.table_gate = nn.Parameter(torch.zeros(int(tables), dtype=torch.float32))
        self.dimension_bias = nn.Parameter(torch.zeros(self.num_vocab_dims, dtype=torch.float32))
        self.shared_logits = nn.Parameter(torch.zeros(self.shared_experts, dtype=torch.float32))
        self.free_head = nn.Linear(self.semantic_dim, self.free_experts, bias=True) if self.free_experts > 0 else None
        nn.init.zeros_(self.hash_embed.weight)
        if self.free_head is not None:
            nn.init.zeros_(self.free_head.weight)
            nn.init.zeros_(self.free_head.bias)
        self.config = config or TileCudaConfig()
        self._init_routing_stats(num_experts=self.total_experts, top_k=max(self.route_width, 1))

    def route_evo_parameters(self) -> list[Tensor]:
        params: list[Tensor] = [self.hash_embed.weight, self.table_gate, self.dimension_bias, self.shared_logits]
        if self.free_head is not None:
            params.extend([self.free_head.weight, self.free_head.bias])
        return [param for param in params if param.requires_grad]

    def _topic_scores(self, topic_logits: Tensor) -> Tensor:
        scores = torch.zeros(
            *topic_logits.shape[:2],
            len(self.term_counts),
            device=topic_logits.device,
            dtype=torch.float32,
        )
        for dim_idx, term_count in enumerate(self.term_counts):
            count = min(int(term_count), topic_logits.size(-1))
            if count <= 0 or dim_idx >= topic_logits.size(-2):
                continue
            dim_logits = topic_logits[..., dim_idx, :count].contiguous()
            if dim_logits.is_cuda and dim_logits.dtype == torch.float32 and dim_logits.size(-1) <= 1024:
                probs = tile_softmax_lastdim_module(dim_logits, self.config)
            else:
                probs = F.softmax(dim_logits.float(), dim=-1)
            scores[..., dim_idx] = probs.max(dim=-1).values
        return scores

    def forward(
        self,
        sem_vec: Tensor,
        bucket_indices: Tensor,
        topic_logits: Tensor,
        sem_targets: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if sem_vec.ndim == 2:
            sem_vec = sem_vec.unsqueeze(1)
        if topic_logits.ndim == 3:
            topic_logits = topic_logits.unsqueeze(1)
        if bucket_indices.ndim == 2:
            bucket_indices = bucket_indices.unsqueeze(1)
        if sem_vec.ndim != 3 or topic_logits.ndim != 4:
            raise ValueError("semantic_moe_jepa_evo_router expects chunked semantic inputs")
        batch, chunks, _ = sem_vec.shape
        work_dtype = topic_logits.dtype if torch.is_floating_point(topic_logits) else torch.float32

        topic_scores = self._topic_scores(topic_logits)
        bucket_features = self.hash_embed(bucket_indices.long() % self.n_buckets).float()
        gate_logits = self.table_gate[: bucket_features.size(2)].contiguous()
        if gate_logits.is_cuda and gate_logits.dtype == torch.float32 and gate_logits.size(-1) <= 1024:
            gate = tile_softmax_lastdim_module(gate_logits, self.config)
        else:
            gate = F.softmax(gate_logits.float(), dim=0)
        hash_bias = (bucket_features * gate.view(1, 1, -1, 1)).sum(dim=2)
        semantic_scores = topic_scores + hash_bias[..., : self.num_vocab_dims] + self.dimension_bias
        if self.free_experts > 0 and self.free_head is not None:
            free_scores = tile_linear_module(
                sem_vec.float(),
                self.free_head.weight,
                self.free_head.bias,
                self.config,
                name="semantic_moe_jepa_evo_router.free_head",
            )
            free_scores = free_scores + hash_bias[..., self.num_vocab_dims :]
            candidate_scores = torch.cat([semantic_scores, free_scores], dim=-1)
        else:
            candidate_scores = semantic_scores

        route_logits = sem_vec.new_zeros((batch, chunks, self.total_experts), dtype=torch.float32)
        if self.shared_experts > 0:
            route_logits[..., : self.shared_experts] = self.shared_logits.view(1, 1, -1)
        route_logits[..., self.shared_experts : self.shared_experts + self.num_vocab_dims] = semantic_scores
        if self.free_experts > 0:
            route_logits[..., self.shared_experts + self.num_vocab_dims :] = candidate_scores[..., self.num_vocab_dims :]

        targets = sem_targets.long()
        if targets.ndim == 1:
            targets = targets.unsqueeze(0)
        if targets.size(1) < self.num_vocab_dims:
            pad = targets.new_full((targets.size(0), self.num_vocab_dims - targets.size(1)), self.ignore_index)
            targets = torch.cat([targets, pad], dim=1)
        forced_mask = targets[:, : self.num_vocab_dims] != self.ignore_index
        has_forced = forced_mask.any(dim=-1)
        neg_inf = candidate_scores.new_full((), float("-inf"))
        forced_scores = candidate_scores.new_full(candidate_scores.shape, float("-inf"))
        forced_scores[..., : self.num_vocab_dims] = torch.where(
            forced_mask.unsqueeze(1),
            semantic_scores,
            neg_inf,
        )
        ordered_forced = torch.argsort(forced_scores, dim=-1, descending=True)
        ordered_all = torch.argsort(candidate_scores, dim=-1, descending=True)
        ordered = torch.where(has_forced.view(batch, 1, 1), ordered_forced, ordered_all)
        chosen = ordered[..., : self.top_k]
        k_per_row = torch.where(
            has_forced,
            forced_mask.sum(dim=-1).clamp(min=1, max=self.top_k),
            torch.full((batch,), self.top_k, device=sem_vec.device, dtype=torch.long),
        )
        chosen_scores = torch.gather(candidate_scores, dim=-1, index=chosen)
        slot_ids = torch.arange(self.top_k, device=sem_vec.device, dtype=torch.long).view(1, 1, -1)
        valid_slots = slot_ids < k_per_row.view(batch, 1, 1)
        masked_scores = torch.where(valid_slots, chosen_scores, neg_inf)
        chosen_experts = torch.where(
            chosen < self.num_vocab_dims,
            chosen + self.shared_experts,
            chosen - self.num_vocab_dims + self.shared_experts + self.num_vocab_dims,
        )
        fallback_experts = chosen_experts[..., :1].expand(-1, -1, self.top_k)
        chosen_experts = torch.where(valid_slots, chosen_experts, fallback_experts)

        if self.shared_experts > 0:
            shared_scores = self.shared_logits.to(device=sem_vec.device, dtype=masked_scores.dtype).view(1, 1, -1).expand(batch, chunks, -1)
            shared_indices = torch.arange(self.shared_experts, device=sem_vec.device, dtype=torch.long).view(1, 1, -1).expand(batch, chunks, -1)
            combined_scores = torch.cat([shared_scores, masked_scores], dim=-1)
            combined_indices = torch.cat([shared_indices, chosen_experts], dim=-1)
        else:
            combined_scores = masked_scores
            combined_indices = chosen_experts
        weights = F.softmax(combined_scores.float(), dim=-1).to(dtype=work_dtype)
        self._update_routing_stats(
            scores=F.softmax(route_logits.detach().float(), dim=-1),
            routing_weights=weights.detach(),
            routing_indices=combined_indices.detach(),
        )
        return weights, combined_indices, route_logits.to(dtype=work_dtype)


class TileCudaSemanticProjectorStage(nn.Module):
    def __init__(
        self,
        input_dim: int,
        semantic_dim: int = 9,
        residual_dim: int = 64,
        n_sig_buckets: int = 4096,
        semantic_vocab_ref: str = "",
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        from ..semantic import ConversationalVocabulary, semantic_vocab_ref_for_dim

        vocab_ref = semantic_vocab_ref or semantic_vocab_ref_for_dim(semantic_dim)
        vocab = ConversationalVocabulary(vocab_ref or None)
        self.semantic_dim = int(semantic_dim)
        self.num_vocab_dims = vocab.num_vocab_dims
        self.term_counts = [len(vocab.terms(dim_name)) for dim_name in vocab.dim_names]
        self.max_terms = max(self.term_counts) if self.term_counts else 0
        self.topic_heads = nn.ModuleList(
            nn.Linear(int(input_dim), max(count, 1), bias=False) for count in self.term_counts
        )
        self.sig_head = nn.Linear(int(input_dim), int(n_sig_buckets), bias=False)
        self.residual_head = nn.Sequential(
            nn.Linear(int(input_dim), int(residual_dim), bias=False),
            nn.GELU(),
            nn.Linear(int(residual_dim), int(residual_dim), bias=False),
        )
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        target_dtype = x.dtype if torch.is_floating_point(x) else self.sig_head.weight.dtype
        if x.ndim == 2:
            x = x.unsqueeze(1)
        pooled = x.mean(dim=1)
        batch = pooled.size(0)
        topic_logits = pooled.new_zeros((batch, self.num_vocab_dims, self.max_terms))
        semantic_dims: list[Tensor] = []
        for dim_idx, head in enumerate(self.topic_heads):
            count = self.term_counts[dim_idx]
            logits = tile_linear_module(pooled, head.weight, None, self.config, name=f"semantic_projector.topic_heads.{dim_idx}")
            topic_logits[:, dim_idx, :count] = logits
            topic_idx = torch.argmax(logits.float(), dim=-1)
            if count > 1:
                semantic_dims.append(2.0 * topic_idx.to(dtype=target_dtype) / float(count - 1) - 1.0)
            else:
                semantic_dims.append(torch.zeros(batch, device=pooled.device, dtype=target_dtype))
        sig_logits = tile_linear_module(pooled, self.sig_head.weight, None, self.config, name="semantic_projector.sig_head")
        sig_probs = tile_softmax_lastdim_module(sig_logits.contiguous(), self.config)
        bucket_axis = torch.linspace(0.0, 1.0, sig_probs.size(-1), device=sig_probs.device, dtype=torch.float32)
        sig_scalar = (sig_probs.float() * bucket_axis).sum(dim=-1, keepdim=True).to(dtype=target_dtype)
        sem = torch.stack(semantic_dims, dim=-1)
        sem = torch.cat([sem, sig_scalar], dim=-1)
        res = tile_linear_module(x, self.residual_head[0].weight, None, self.config, name="semantic_projector.residual_head.0")
        res = tile_unary("gelu", res, self.config)
        res = tile_linear_module(res, self.residual_head[2].weight, None, self.config, name="semantic_projector.residual_head.2")
        return sem, res.to(dtype=target_dtype), topic_logits.to(dtype=target_dtype)


class TileCudaSemanticChunkProjectorStage(nn.Module):
    def __init__(
        self,
        input_dim: int,
        semantic_dim: int = 9,
        residual_dim: int = 64,
        n_sig_buckets: int = 4096,
        semantic_vocab_ref: str = "",
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        from ..semantic import ConversationalVocabulary, semantic_vocab_ref_for_dim

        vocab_ref = semantic_vocab_ref or semantic_vocab_ref_for_dim(semantic_dim)
        vocab = ConversationalVocabulary(vocab_ref or None)
        self.semantic_dim = int(semantic_dim)
        self.num_vocab_dims = vocab.num_vocab_dims
        self.term_counts = [len(vocab.terms(dim_name)) for dim_name in vocab.dim_names]
        self.max_terms = max(self.term_counts) if self.term_counts else 0
        self.topic_heads = nn.ModuleList(
            nn.Linear(int(input_dim), max(count, 1), bias=False) for count in self.term_counts
        )
        self.sig_head = nn.Linear(int(input_dim), int(n_sig_buckets), bias=False)
        self.residual_head = nn.Sequential(
            nn.Linear(int(input_dim), int(residual_dim), bias=False),
            nn.GELU(),
            nn.Linear(int(residual_dim), int(residual_dim), bias=False),
        )
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError("semantic_chunk_projector expects [batch, chunks, dim]")
        batch, chunks, dim = x.shape
        flat = x.reshape(batch * chunks, dim).contiguous()
        target_dtype = flat.dtype if torch.is_floating_point(flat) else self.sig_head.weight.dtype
        topic_logits = flat.new_zeros((batch * chunks, self.num_vocab_dims, self.max_terms))
        semantic_dims: list[Tensor] = []
        for dim_idx, head in enumerate(self.topic_heads):
            count = self.term_counts[dim_idx]
            logits = tile_linear_module(
                flat,
                head.weight,
                None,
                self.config,
                name=f"semantic_chunk_projector.topic_heads.{dim_idx}",
            )
            topic_logits[:, dim_idx, :count] = logits
            topic_idx = torch.argmax(logits.float(), dim=-1)
            if count > 1:
                semantic_dims.append(2.0 * topic_idx.to(dtype=target_dtype) / float(count - 1) - 1.0)
            else:
                semantic_dims.append(torch.zeros(batch * chunks, device=flat.device, dtype=target_dtype))
        sig_logits = tile_linear_module(
            flat,
            self.sig_head.weight,
            None,
            self.config,
            name="semantic_chunk_projector.sig_head",
        )
        sig_probs = tile_softmax_lastdim_module(sig_logits.contiguous(), self.config)
        bucket_axis = torch.linspace(0.0, 1.0, sig_probs.size(-1), device=sig_probs.device, dtype=torch.float32)
        sig_scalar = (sig_probs.float() * bucket_axis).sum(dim=-1, keepdim=True).to(dtype=target_dtype)
        sem = torch.stack(semantic_dims, dim=-1)
        sem = torch.cat([sem, sig_scalar], dim=-1)
        res = tile_linear_module(
            flat,
            self.residual_head[0].weight,
            None,
            self.config,
            name="semantic_chunk_projector.residual_head.0",
        )
        res = tile_unary("gelu", res, self.config)
        res = tile_linear_module(
            res,
            self.residual_head[2].weight,
            None,
            self.config,
            name="semantic_chunk_projector.residual_head.2",
        )
        return (
            sem.reshape(batch, chunks, -1),
            res.to(dtype=target_dtype).reshape(batch, chunks, -1),
            topic_logits.to(dtype=target_dtype).reshape(batch, chunks, self.num_vocab_dims, self.max_terms),
        )


class TileCudaTopKRouteStage(RoutingStatsMixin, nn.Module):
    def __init__(self, top_k: int, experts: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.experts = int(experts)
        self.config = config or TileCudaConfig()
        self._init_routing_stats(num_experts=self.experts, top_k=self.top_k)

    def forward(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        weights, indices = tile_topk_route_module(logits, self.top_k, self.config)
        scores = torch.softmax(logits.detach().float(), dim=-1)
        self._update_routing_stats(scores=scores, routing_weights=weights.detach(), routing_indices=indices.detach())
        return weights.to(dtype=logits.dtype), indices


class TileCudaAttentionlessDecoderStage(nn.Module):
    def __init__(
        self,
        semantic_dim: int = 9,
        residual_dim: int = 64,
        vocab_size: int = 256,
        n_buckets: int = 256,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        del semantic_dim
        self.bucket_embed = nn.Embedding(int(n_buckets), int(residual_dim))
        self.out_proj = nn.Linear(int(residual_dim), int(vocab_size), bias=False)
        self.n_buckets = int(n_buckets)
        self.config = config or TileCudaConfig()

    def forward(self, bucket_indices: Tensor, expert_output: Tensor) -> Tensor:
        return tile_attentionless_decoder_module(
            bucket_indices,
            expert_output,
            self.bucket_embed.weight,
            self.out_proj.weight,
            self.config,
        )


class TileCudaAuxFreeBalancingStage(nn.Module):
    def __init__(self, experts: int, top_k: int, bias_lr: float = 0.001, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.experts = int(experts)
        self.top_k = max(min(int(top_k), self.experts), 1)
        self.bias_lr = float(bias_lr)
        self.register_buffer("expert_bias", torch.zeros(self.experts, dtype=torch.float32))
        self.config = config or TileCudaConfig()

    def forward(self, logits: Tensor) -> Tensor:
        biased = tile_expert_bias_add_module(logits, self.expert_bias.to(device=logits.device), self.config)
        if self.training:
            with torch.no_grad():
                flat = biased.reshape(-1, self.experts)
                _, idx = torch.topk(flat, self.top_k, dim=-1)
                counts = torch.zeros(self.experts, device=logits.device, dtype=torch.float32)
                counts.scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel(), device=logits.device, dtype=torch.float32))
                load = counts / counts.sum().clamp(min=1.0)
                target = 1.0 / self.experts
                self.expert_bias += self.bias_lr * torch.sign(target - load).to(device=self.expert_bias.device)
        return biased


class TileCudaSoftmaxDistillationLossStage(nn.Module):
    def __init__(self, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.config = config or TileCudaConfig()

    def forward(self, teacher_logits: Tensor, student_logits: Tensor) -> Tensor:
        return tile_softmax_distillation_loss_module(teacher_logits, student_logits, self.config)


class TileCudaExpertDispatchStage(nn.Module):
    def __init__(self, model_dim: int, experts: int, mlp_mult: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        hidden_dim = int(model_dim) * int(mlp_mult)
        self.w1 = nn.Parameter(torch.empty(int(experts), int(model_dim), hidden_dim))
        self.w2 = nn.Parameter(torch.empty(int(experts), hidden_dim, int(model_dim)))
        self.w3 = nn.Parameter(torch.empty(int(experts), int(model_dim), hidden_dim))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)
        nn.init.normal_(self.w3, std=0.02)
        self.experts = int(experts)
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor, routing_weights: Tensor, routing_indices: Tensor) -> Tensor:
        batch, seq_len, d = x.shape
        top_k = routing_indices.shape[-1]
        x_flat = x.reshape(-1, d)
        out = torch.zeros_like(x_flat)
        routing_weights_flat = routing_weights.reshape(-1, top_k)
        routing_indices_flat = routing_indices.reshape(-1, top_k)

        for expert_idx in range(self.experts):
            mask = routing_indices_flat == expert_idx
            idx = torch.where(mask)[0]
            if idx.numel() == 0:
                continue
            expert_inputs = x_flat[idx].contiguous()
            h1 = tile_linear_module(
                expert_inputs,
                self.w1[expert_idx].t().contiguous(),
                None,
                self.config,
                name="expert_dispatch.w1",
            )
            h3 = tile_linear_module(
                expert_inputs,
                self.w3[expert_idx].t().contiguous(),
                None,
                self.config,
                name="expert_dispatch.w3",
            )
            h = tile_unary("silu", h1.contiguous(), self.config) * h3
            expert_out = tile_linear_module(
                h.contiguous(),
                self.w2[expert_idx].t().contiguous(),
                None,
                self.config,
                name="expert_dispatch.w2",
            )
            weights = routing_weights_flat[mask]
            out[idx] += expert_out * weights.unsqueeze(-1)
        return out.reshape(batch, seq_len, d)


class TileCudaExpertCombineStage(nn.Module):
    def __init__(self, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        return tile_identity_module("expert_combine", x, self.config)


class TileCudaKVCacheWriteStage(nn.Module):
    def __init__(self, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.config = config or TileCudaConfig()

    def forward(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return tile_kv_cache_write_module(k, v, self.config)


class TileCudaKVCacheReadStage(nn.Module):
    def __init__(self, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.config = config or TileCudaConfig()

    def forward(
        self,
        k: Tensor,
        v: Tensor,
        cache_k: Tensor | None = None,
        cache_v: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return tile_kv_cache_read_module(k, v, cache_k, cache_v, self.config)


class TileCudaKVQuantPackStage(nn.Module):
    def __init__(self, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.config = config or TileCudaConfig()

    def forward(self, k: Tensor, v: Tensor) -> Tensor:
        return tile_kv_quant_pack_module(k, v, self.config)


class TileCudaKVQuantUnpackStage(nn.Module):
    def __init__(self, head_dim: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.config = config or TileCudaConfig()

    def forward(self, packed: Tensor) -> tuple[Tensor, Tensor]:
        return tile_kv_quant_unpack_module(packed, self.head_dim, self.config)


class TileCudaAbsolutePositionEmbeddingStage(nn.Module):
    def __init__(self, max_seq_len: int, model_dim: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, model_dim)
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len = x.shape[:2]
        return tile_absolute_position_embedding_module(self.embedding.weight, int(batch), int(seq_len), self.config)


class TileCudaTokenEmbeddingStage(nn.Module):
    def __init__(self, vocab_size: int, model_dim: int, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.config = config or TileCudaConfig()

    def forward(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        return tile_token_embedding_module(self.embedding.weight, token_ids, self.config), self.embedding.weight


class TileCudaBroadcastExpertRoutesStage(nn.Module):
    def __init__(self, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.config = config or TileCudaConfig()

    def forward(self, hidden: Tensor, expert_weights: Tensor, expert_indices: Tensor) -> tuple[Tensor, Tensor]:
        return tile_broadcast_expert_routes_module(hidden, expert_weights, expert_indices, self.config)


class TileCudaBroadcastChunkRoutesStage(nn.Module):
    def __init__(self, chunk_size: int = 32, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.chunk_size = max(int(chunk_size), 1)
        self.config = config or TileCudaConfig()

    def forward(self, hidden: Tensor, expert_weights: Tensor, expert_indices: Tensor) -> tuple[Tensor, Tensor]:
        return tile_broadcast_chunk_routes_module(hidden, expert_weights, expert_indices, self.chunk_size, self.config)


class TileCudaBytePatchMergeStage(nn.Module):
    def __init__(self, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor, target_tokens: Tensor) -> Tensor:
        return tile_byte_patch_merge_module(x, target_tokens, self.config)


class TileCudaBytePatchEmbedStage(nn.Module):
    def __init__(
        self,
        model_dim: int,
        patch_size: int,
        stride: int,
        vocab_size: int = 256,
        config: TileCudaConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.vocab_size = int(vocab_size)
        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
        self.proj = nn.Conv1d(self.model_dim, self.model_dim, kernel_size=self.patch_size, stride=self.stride, bias=False)
        self.config = config or TileCudaConfig()

    def forward(self, tokens: Tensor) -> Tensor:
        return tile_byte_patch_embed_module(
            tokens,
            self.embedding.weight,
            self.proj.weight,
            self.patch_size,
            self.stride,
            self.vocab_size,
            self.config,
        )


class TileCudaCausalChunkStateStage(nn.Module):
    def __init__(self, chunk_size: int = 32, mode: str = "prefix", config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.chunk_size = max(int(chunk_size), 1)
        self.mode = str(mode)
        self.config = config or TileCudaConfig()

    def forward(self, hidden: Tensor) -> Tensor:
        return tile_causal_chunk_state_module(hidden, self.chunk_size, self.mode, self.config)


class TileCudaLatentMSELossStage(nn.Module):
    def __init__(self, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.config = config or TileCudaConfig()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return tile_latent_mse_loss_module(pred, target, self.config)


class TileCudaLatentPoolStage(nn.Module):
    def __init__(self, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        return tile_latent_pool_module(x, mask, self.config)


class TileCudaTokenCrossEntropyStage(nn.Module):
    def __init__(self, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.config = config or TileCudaConfig()

    def forward(self, logits: Tensor, target_ids: Tensor) -> Tensor:
        return tile_token_cross_entropy_module(logits, target_ids, self.config)


class TileCudaGELUStage(nn.Module):
    def __init__(self, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.config = config or TileCudaConfig()

    def forward(self, x: Tensor) -> Tensor:
        return tile_unary("gelu", x, self.config)


class TileCudaMaskedTokenCrossEntropyStage(nn.Module):
    def __init__(self, ignore_index: int = -100, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.config = config or TileCudaConfig()

    def forward(self, logits: Tensor, target_ids: Tensor, loss_mask: Tensor) -> Tensor:
        return tile_masked_token_cross_entropy_module(logits, target_ids, loss_mask, self.ignore_index, self.config)


class TileCudaSequenceLogpStage(nn.Module):
    def __init__(self, ignore_index: int = -100, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.config = config or TileCudaConfig()

    def forward(self, logits: Tensor, targets: Tensor, loss_mask: Tensor) -> Tensor:
        return tile_sequence_logp_module(logits, targets, loss_mask, self.ignore_index, self.config)


class TileCudaLoadBalanceLossStage(nn.Module):
    def __init__(self, config: TileCudaConfig | None = None) -> None:
        super().__init__()
        self.config = config or TileCudaConfig()

    def forward(
        self,
        router_logits: Tensor,
        routing_weights: Tensor,
        routing_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return tile_load_balance_loss_module(router_logits, routing_weights, routing_indices, self.config)


def build_tile_module(module_type: str, module_config: dict[str, object], config: TileCudaConfig | None = None) -> nn.Module | None:
    cfg = dict(module_config)
    if module_type == "logit_softcap":
        return TileCudaLogitSoftcapStage(float(cfg.get("softcap", 30.0)), config)
    if module_type == "loss_scale":
        return TileCudaLossScaleStage(float(cfg.get("coef", 1.0)), config)
    if module_type == "aux_loss_add":
        return TileCudaAuxLossAddStage(float(cfg.get("coef", 1.0)), config)
    if module_type == "kl_penalty":
        return TileCudaKLPenaltyStage(float(cfg.get("kl_coef", 0.1)), config)
    if module_type == "residual_add":
        return TileCudaResidualAddStage(int(cfg["dim"]), float(cfg.get("init_scale", 1.0)), config)
    if module_type == "residual_mix":
        return TileCudaResidualMixStage(
            int(cfg["dim"]),
            float(cfg.get("primary_init", 1.0)),
            float(cfg.get("skip_init", 0.0)),
            config,
        )
    if module_type == "manifold_hyper_connection":
        return TileCudaManifoldHyperConnectionStage(int(cfg["dim"]), float(cfg.get("beta_init", 0.1)), config)
    if module_type == "qk_gain":
        return TileCudaQKGainStage(int(cfg["num_heads"]), float(cfg.get("qk_gain_init", 1.0)), config)
    if module_type == "dyt":
        return TileCudaDyTStage(int(cfg["model_dim"]), float(cfg.get("alpha_init", 1.0)), config)
    if module_type == "dropout":
        return TileCudaDropoutStage(float(cfg.get("p", cfg.get("dropout", 0.0))), config)
    if module_type == "reshape_heads":
        return TileCudaReshapeHeadsStage(int(cfg["num_heads"]), config)
    if module_type == "merge_heads":
        return TileCudaMergeHeadsStage(config)
    if module_type == "repeat_kv":
        return TileCudaRepeatKVStage(
            int(cfg.get("num_heads", 4)),
            int(cfg.get("num_kv_heads", cfg.get("num_heads", 4))),
            config,
        )
    if module_type == "rotary_embedding":
        return TileCudaRotaryEmbeddingStage(
            int(cfg["head_dim"]),
            float(cfg.get("rope_base", 10000.0)),
            cfg.get("rope_scaling"),
            config,
        )
    if module_type == "rms_norm":
        return TileCudaRMSNormStage(float(cfg.get("eps", 1e-6)), config)
    if module_type == "layer_norm":
        return TileCudaLayerNormStage(int(cfg["model_dim"]), float(cfg.get("eps", 1e-5)), config)
    if module_type == "group_norm":
        return TileCudaGroupNormStage(int(cfg["model_dim"]), int(cfg.get("num_groups", 1)), float(cfg.get("eps", 1e-5)), config)
    if module_type == "qk_norm":
        return TileCudaQKNormStage(float(cfg.get("eps", 1e-6)), config)
    if module_type == "linear":
        return TileCudaLinearStage(int(cfg["input_dim"]), int(cfg["output_dim"]), bool(cfg.get("bias", False)), config)
    if module_type == "lm_head":
        return TileCudaLMHeadStage(int(cfg["model_dim"]), int(cfg["vocab_size"]), config)
    if module_type == "tied_lm_head":
        return TileCudaTiedLMHeadStage(config)
    if module_type == "router_logits":
        return TileCudaRouterLogitsStage(int(cfg["model_dim"]), int(cfg["experts"]), config)
    if module_type == "value_head":
        return TileCudaValueHeadStage(int(cfg.get("model_dim", 128)), config)
    if module_type == "reward_head":
        return TileCudaRewardHeadStage(int(cfg.get("model_dim", 128)), str(cfg.get("pool", "last")), config)
    if module_type == "denoise_head":
        return TileCudaDenoiseHeadStage(int(cfg["model_dim"]), int(cfg["vocab_size"]), config)
    if module_type == "kv_pca_encode":
        return TileCudaKVPCAEncodeStage(int(cfg["head_dim"]), int(cfg["compressed_dim"]), config)
    if module_type == "kv_pca_decode":
        return TileCudaKVPCADecodeStage(int(cfg["head_dim"]), int(cfg["compressed_dim"]), config)
    if module_type == "jepa_projector":
        return TileCudaJEPAProjectorStage(int(cfg["input_dim"]), int(cfg["latent_dim"]), config)
    if module_type == "jepa_predictor":
        return TileCudaJEPAPredictorStage(int(cfg["latent_dim"]), config)
    if module_type == "ttt_linear":
        return TileCudaTTTLinearStage(
            int(cfg["input_dim"]),
            int(cfg["output_dim"]),
            int(cfg.get("hidden_dim", 16)),
            config,
        )
    if module_type == "lora_linear":
        if float(cfg.get("dropout", 0.0)) != 0.0:
            return None
        return TileCudaLoRALinearStage(
            int(cfg["input_dim"]),
            int(cfg["output_dim"]),
            int(cfg.get("rank", 8)),
            float(cfg.get("alpha", 16.0)),
            bool(cfg.get("bias", False)),
            config,
        )
    if module_type == "bitlinear_ternary":
        return TileCudaBitLinearTernaryStage(int(cfg["input_dim"]), int(cfg["output_dim"]), config)
    if module_type == "fp8_linear":
        return TileCudaFP8LinearStage(
            int(cfg["input_dim"]),
            int(cfg["output_dim"]),
            bool(cfg.get("bias", False)),
            str(cfg.get("fp8_format", "e4m3")),
            int(cfg.get("amax_history_len", 16)),
            bool(cfg.get("use_stochastic_rounding", True)),
            config,
        )
    if module_type == "mx_linear":
        return TileCudaMXLinearStage(
            int(cfg["input_dim"]),
            int(cfg["output_dim"]),
            bool(cfg.get("bias", False)),
            str(cfg.get("mx_format", "mxfp4")),
            int(cfg.get("mx_block_size", 32)),
            config,
        )
    if module_type == "nf4_linear":
        if float(cfg.get("dropout", 0.0)) != 0.0:
            return None
        if str(cfg.get("compute_dtype", "bf16")) not in {"fp32", "float32"}:
            return None
        return TileCudaNF4LinearStage(
            int(cfg["input_dim"]),
            int(cfg["output_dim"]),
            int(cfg.get("rank", 8)),
            float(cfg.get("alpha", 16.0)),
            bool(cfg.get("bias", False)),
            int(cfg.get("group_size", 64)),
            config,
        )
    if module_type == "randmap_adapter":
        return TileCudaRandMapAdapterStage(int(cfg["model_dim"]), int(cfg["adapter_dim"]), config)
    if module_type == "mlp_relu2":
        return TileCudaMLPReluSquaredStage(int(cfg["model_dim"]), int(cfg["mlp_mult"]), config)
    if module_type == "swiglu":
        return TileCudaSwiGLUStage(
            int(cfg["model_dim"]),
            int(cfg["mlp_mult"]),
            int(cfg["multiple_of"]) if cfg.get("multiple_of") is not None else None,
            config,
        )
    if module_type in {"geglu", "reglu", "solu"}:
        return TileCudaGLUStage(
            int(cfg["model_dim"]),
            int(cfg["mlp_mult"]),
            int(cfg["multiple_of"]) if cfg.get("multiple_of") is not None else None,
            {"geglu": "gelu", "reglu": "relu", "solu": "softmax"}[module_type],
            config,
        )
    if module_type == "act_weighted_sum":
        return TileCudaACTWeightedSumStage(config)
    if module_type == "act_halt_gate":
        return TileCudaACTHaltGateStage(int(cfg.get("model_dim", 128)), config)
    if module_type == "random_timesteps":
        return TileCudaRandomTimestepsStage(config)
    if module_type == "mask_scheduler":
        return TileCudaMaskSchedulerStage(
            int(cfg["vocab_size"]),
            int(cfg["mask_token_id"]),
            config,
        )
    if module_type == "jepa_mask":
        return TileCudaJEPAMaskStage(
            float(cfg.get("mask_ratio", 0.5)),
            int(cfg.get("mask_token_id", 0)),
            str(cfg.get("mask_strategy", "random")),
            int(cfg.get("num_blocks", 4)),
            float(cfg.get("min_block_ratio", 0.1)),
            float(cfg.get("max_block_ratio", 0.25)),
            config,
        )
    if module_type == "dpo_pairwise_loss":
        return TileCudaDPOPairwiseLossStage(
            float(cfg.get("beta", 0.1)),
            float(cfg.get("label_smoothing", 0.0)),
            str(cfg.get("loss_type", "sigmoid")),
            config,
        )
    if module_type == "preference_bce_loss":
        return TileCudaPreferenceBCELossStage(config)
    if module_type == "ppo_clipped_loss":
        return TileCudaPPOClippedLossStage(
            float(cfg.get("clip_range", 0.2)),
            float(cfg.get("vf_coef", 0.5)),
            float(cfg.get("ent_coef", 0.0)),
            config,
        )
    if module_type == "gae_compute":
        return TileCudaGAEComputeStage(
            float(cfg.get("gamma", 1.0)),
            float(cfg.get("lambda_", cfg.get("lambda", 0.95))),
            config,
        )
    if module_type == "route_balance_loss":
        return TileCudaRouteBalanceLossStage(config)
    if module_type == "route_selection_loss":
        return TileCudaRouteSelectionLossStage(
            str(cfg.get("semantic_vocab_ref", "")),
            int(cfg.get("shared_experts", 2)),
            int(cfg.get("free_experts", 8)),
            int(cfg.get("ignore_index", -100)),
            config,
        )
    if module_type == "route_distillation_loss":
        return TileCudaRouteDistillationLossStage(
            str(cfg.get("semantic_vocab_ref", "")),
            int(cfg.get("shared_experts", 2)),
            int(cfg.get("free_experts", 8)),
            config,
        )
    if module_type == "semantic_alignment_loss":
        return TileCudaSemanticAlignmentLossStage(
            int(cfg.get("ignore_index", -100)),
            str(cfg.get("semantic_vocab_ref", "")),
            config,
        )
    if module_type == "semantic_hasher":
        return TileCudaSemanticHasherStage(
            int(cfg.get("dim", 9)),
            int(cfg.get("tables", 8)),
            int(cfg.get("planes", 12)),
            int(cfg.get("seed", 42)),
            config,
        )
    if module_type == "semantic_chunk_hasher":
        return TileCudaSemanticChunkHasherStage(
            int(cfg.get("dim", 9)),
            int(cfg.get("tables", 8)),
            int(cfg.get("planes", 12)),
            int(cfg.get("seed", 42)),
            config,
        )
    if module_type == "semantic_moe_router":
        return TileCudaSemanticMoERouterStage(
            int(cfg["n_experts"]),
            int(cfg.get("semantic_dim", 9)),
            int(cfg.get("top_k", 2)),
            config,
        )
    if module_type == "semantic_hash_router":
        return TileCudaSemanticHashRouterStage(
            int(cfg["n_experts"]),
            int(cfg.get("semantic_dim", 9)),
            int(cfg.get("top_k", 2)),
            int(cfg.get("tables", 8)),
            int(cfg.get("n_buckets", 4096)),
            int(cfg.get("ignore_index", -100)),
            str(cfg.get("semantic_vocab_ref", "")),
            str(cfg.get("routing_source", "topic_logits")),
            config,
        )
    if module_type == "semantic_moe_jepa_evo_router":
        return TileCudaSemanticMoeJepaEvoRouterStage(
            int(cfg.get("semantic_dim", 9)),
            int(cfg.get("top_k", 2)),
            int(cfg.get("shared_experts", 2)),
            int(cfg.get("free_experts", 8)),
            int(cfg.get("tables", 8)),
            int(cfg.get("n_buckets", 4096)),
            int(cfg.get("ignore_index", -100)),
            str(cfg.get("semantic_vocab_ref", "")),
            config,
        )
    if module_type == "semantic_projector":
        return TileCudaSemanticProjectorStage(
            int(cfg["input_dim"]),
            int(cfg.get("semantic_dim", 9)),
            int(cfg.get("residual_dim", 64)),
            int(cfg.get("n_sig_buckets", 4096)),
            str(cfg.get("semantic_vocab_ref", "")),
            config,
        )
    if module_type == "semantic_chunk_projector":
        return TileCudaSemanticChunkProjectorStage(
            int(cfg["input_dim"]),
            int(cfg.get("semantic_dim", 9)),
            int(cfg.get("residual_dim", 64)),
            int(cfg.get("n_sig_buckets", 4096)),
            str(cfg.get("semantic_vocab_ref", "")),
            config,
        )
    if module_type == "topk_route":
        return TileCudaTopKRouteStage(
            int(cfg["top_k"]),
            int(cfg.get("experts", 8)),
            config,
        )
    if module_type == "scaled_dot_product_attention":
        return TileCudaScaledDotProductAttentionStage(
            bool(cfg.get("is_causal", True)),
            str(cfg.get("backend", "sdpa")),
            float(cfg.get("dropout_p", 0.0)),
            config,
        )
    if module_type == "sliding_window_attention":
        return TileCudaSparseAttentionStage(
            window=int(cfg.get("window_size", 256)),
            is_causal=bool(cfg.get("is_causal", True)),
            dropout_p=float(cfg.get("dropout_p", 0.0)),
            config=config,
        )
    if module_type == "block_sparse_attention":
        return TileCudaSparseAttentionStage(
            block_size=int(cfg.get("sparse_block_size", 64)),
            num_sinks=int(cfg.get("num_sinks", 0)),
            is_causal=bool(cfg.get("is_causal", True)),
            dropout_p=float(cfg.get("dropout_p", 0.0)),
            config=config,
        )
    if module_type == "streaming_attention_sinks":
        return TileCudaSparseAttentionStage(
            window=int(cfg.get("window_size", 256)),
            num_sinks=int(cfg.get("num_sinks", 4)),
            is_causal=bool(cfg.get("is_causal", True)),
            dropout_p=float(cfg.get("dropout_p", 0.0)),
            config=config,
        )
    if module_type == "native_sparse_attention":
        return TileCudaSparseAttentionStage(
            window=int(cfg.get("window_size", 128)),
            num_sinks=int(cfg.get("num_sinks", 0)),
            compress_stride=int(cfg.get("compress_stride", 16)),
            is_causal=bool(cfg.get("is_causal", True)),
            dropout_p=float(cfg.get("dropout_p", 0.0)),
            config=config,
        )
    if module_type == "differential_attention":
        return TileCudaDifferentialAttentionStage(
            float(cfg.get("lambda_init", 0.8)),
            bool(cfg.get("is_causal", True)),
            float(cfg.get("dropout_p", 0.0)),
            float(cfg.get("eps", 1e-5)),
            config,
        )
    if module_type == "causal_self_attention":
        return TileCudaCausalSelfAttentionStage(
            int(cfg["model_dim"]),
            int(cfg["num_heads"]),
            int(cfg["num_kv_heads"]),
            float(cfg["rope_base"]),
            float(cfg["qk_gain_init"]),
            config,
        )
    if module_type == "fused_causal_attention":
        return TileCudaFusedCausalAttentionStage(
            int(cfg["model_dim"]),
            int(cfg["num_heads"]),
            int(cfg["num_kv_heads"]),
            float(cfg.get("rope_base", 10000.0)),
            float(cfg.get("dropout_p", 0.0)),
            config,
        )
    if module_type == "multi_latent_attention":
        return TileCudaMLAStage(
            int(cfg["model_dim"]),
            int(cfg["num_heads"]),
            int(cfg["kv_lora_rank"]) if cfg.get("kv_lora_rank") is not None else None,
            int(cfg["qk_rope_dim"]) if cfg.get("qk_rope_dim") is not None else None,
            float(cfg.get("rope_base", 10000.0)),
            float(cfg.get("dropout_p", 0.0)),
            config,
        )
    if module_type == "routed_attention_experts":
        return TileCudaRoutedAttentionExpertsStage(
            int(cfg["model_dim"]),
            int(cfg["num_heads"]),
            int(cfg["num_kv_heads"]),
            float(cfg["rope_base"]),
            float(cfg["qk_gain_init"]),
            int(cfg["experts"]),
            int(cfg.get("top_k", 2)),
            bool(cfg.get("is_causal", True)),
            config,
        )
    if module_type == "mamba":
        return TileCudaMambaStage(
            int(cfg["model_dim"]),
            int(cfg.get("d_state", 16)),
            int(cfg.get("d_conv", 4)),
            int(cfg.get("expand", 2)),
            config,
        )
    if module_type == "universal_transformer":
        return TileCudaUniversalTransformerStage(
            int(cfg["model_dim"]),
            int(cfg["num_heads"]),
            float(cfg.get("mlp_mult", 4.0)),
            int(cfg.get("max_steps", 4)),
            float(cfg.get("halt_epsilon", 0.01)),
            config,
        )
    if module_type == "attentionless_decoder":
        return TileCudaAttentionlessDecoderStage(
            int(cfg.get("semantic_dim", 9)),
            int(cfg.get("residual_dim", 64)),
            int(cfg.get("vocab_size", 256)),
            int(cfg.get("n_buckets", 256)),
            config,
        )
    if module_type == "auxfree_load_balancing":
        return TileCudaAuxFreeBalancingStage(
            int(cfg["experts"]),
            int(cfg.get("top_k", 2)),
            float(cfg.get("bias_lr", 0.001)),
            config,
        )
    if module_type == "softmax_distillation_loss":
        return TileCudaSoftmaxDistillationLossStage(config)
    if module_type == "expert_dispatch":
        return TileCudaExpertDispatchStage(
            int(cfg["model_dim"]),
            int(cfg["experts"]),
            int(cfg.get("mlp_mult", 4)),
            config,
        )
    if module_type == "expert_combine":
        return TileCudaExpertCombineStage(config)
    if module_type == "kv_cache_write":
        return TileCudaKVCacheWriteStage(config)
    if module_type == "kv_cache_read":
        return TileCudaKVCacheReadStage(config)
    if module_type == "kv_quant_pack":
        return TileCudaKVQuantPackStage(config)
    if module_type == "kv_quant_unpack":
        return TileCudaKVQuantUnpackStage(int(cfg["head_dim"]), config)
    if module_type == "absolute_position_embedding":
        return TileCudaAbsolutePositionEmbeddingStage(
            int(cfg.get("max_seq_len", 1024)),
            int(cfg["model_dim"]),
            config,
        )
    if module_type == "token_embedding":
        return TileCudaTokenEmbeddingStage(int(cfg["vocab_size"]), int(cfg["model_dim"]), config)
    if module_type == "broadcast_expert_routes":
        return TileCudaBroadcastExpertRoutesStage(config)
    if module_type == "broadcast_chunk_routes":
        return TileCudaBroadcastChunkRoutesStage(int(cfg.get("chunk_size", 32)), config)
    if module_type == "byte_patch_merge":
        return TileCudaBytePatchMergeStage(config)
    if module_type == "byte_patch_embed":
        return TileCudaBytePatchEmbedStage(
            int(cfg["model_dim"]),
            int(cfg["patch_size"]),
            int(cfg["stride"]),
            int(cfg.get("vocab_size", 256)),
            config,
        )
    if module_type == "causal_chunk_state":
        return TileCudaCausalChunkStateStage(int(cfg.get("chunk_size", 32)), str(cfg.get("mode", "prefix")), config)
    if module_type == "latent_mse_loss":
        return TileCudaLatentMSELossStage(config)
    if module_type == "latent_pool":
        return TileCudaLatentPoolStage(config)
    if module_type == "token_cross_entropy":
        return TileCudaTokenCrossEntropyStage(config)
    if module_type == "gelu":
        return TileCudaGELUStage(config)
    if module_type == "masked_token_cross_entropy":
        return TileCudaMaskedTokenCrossEntropyStage(int(cfg.get("ignore_index", -100)), config)
    if module_type == "sequence_logp":
        return TileCudaSequenceLogpStage(int(cfg.get("ignore_index", -100)), config)
    if module_type == "load_balance_loss":
        return TileCudaLoadBalanceLossStage(config)
    return None


def tile_function_reference(name: str, *args: Tensor) -> Tensor:
    if name == "identity":
        return args[0]
    if name == "negate":
        return -args[0]
    if name == "relu":
        return torch.relu(args[0])
    if name == "sigmoid":
        return torch.sigmoid(args[0])
    if name == "tanh_neuron":
        return torch.tanh(args[0])
    if name == "leaky_relu":
        return torch.nn.functional.leaky_relu(args[0], negative_slope=0.01)
    if name == "silu":
        return torch.nn.functional.silu(args[0])
    if name == "softplus":
        return torch.nn.functional.softplus(args[0])
    if name == "hard_tanh":
        return torch.nn.functional.hardtanh(args[0])
    if name == "gaussian":
        return torch.exp(-(args[0] * args[0]))
    if name == "log":
        return torch.log(torch.clamp_min(args[0], 1e-7))
    if name == "prelu":
        return torch.where(args[0] >= 0, args[0], args[0] * 0.25)
    if name == "relu6":
        return torch.clamp(args[0], min=0.0, max=6.0)
    if name == "elu":
        return torch.nn.functional.elu(args[0])
    if name == "selu":
        return torch.nn.functional.selu(args[0])
    if name == "mish":
        return torch.nn.functional.mish(args[0])
    if name == "softsign":
        return torch.nn.functional.softsign(args[0])
    if name == "hard_sigmoid":
        return torch.nn.functional.hardsigmoid(args[0])
    if name == "hard_swish":
        return torch.nn.functional.hardswish(args[0])
    if name == "threshold":
        return args[0] * 0.0 + (args[0] >= 0).to(dtype=args[0].dtype)
    if name == "gelu":
        return torch.nn.functional.gelu(args[0])
    if name == "add":
        return args[0] + args[1]
    if name == "multiply":
        return args[0] * args[1]
    if name == "softmax_2":
        out = torch.softmax(torch.stack((args[0], args[1]), dim=0), dim=0)
        return out[0], out[1]
    if name == "logsoftmax_2":
        out = torch.log_softmax(torch.stack((args[0], args[1]), dim=0), dim=0)
        return out[0], out[1]
    raise KeyError(f"No CUDA Tile reference function registered for {name!r}")


__all__ = [
    "TILE_FUNCTION_NAMES",
    "TILE_MODULE_NAMES",
    "TileCudaBinaryFunctionStage",
    "TileCudaBinaryPairFunctionStage",
    "TileCudaAuxLossAddStage",
    "TileCudaKLPenaltyStage",
    "TileCudaBroadcastChunkRoutesStage",
    "TileCudaBroadcastExpertRoutesStage",
    "TileCudaBytePatchMergeStage",
    "TileCudaExpertCombineStage",
    "TileCudaExpertDispatchStage",
    "TileCudaAbsolutePositionEmbeddingStage",
    "TileCudaKVCacheReadStage",
    "TileCudaKVCacheWriteStage",
    "TileCudaLatentMSELossStage",
    "TileCudaTokenEmbeddingStage",
    "TileCudaLogitSoftcapStage",
    "TileCudaLossScaleStage",
    "TileCudaManifoldHyperConnectionStage",
    "TileCudaDyTStage",
    "TileCudaQKGainStage",
    "TileCudaMergeHeadsStage",
    "TileCudaRepeatKVStage",
    "TileCudaRotaryEmbeddingStage",
    "TileCudaRMSNormStage",
    "TileCudaLayerNormStage",
    "TileCudaQKNormStage",
    "TileCudaLinearStage",
    "TileCudaLMHeadStage",
    "TileCudaTiedLMHeadStage",
    "TileCudaRouterLogitsStage",
    "TileCudaValueHeadStage",
    "TileCudaRewardHeadStage",
    "TileCudaDenoiseHeadStage",
    "TileCudaACTWeightedSumStage",
    "TileCudaPreferenceBCELossStage",
    "TileCudaReshapeHeadsStage",
    "TileCudaResidualAddStage",
    "TileCudaResidualMixStage",
    "TileCudaUnaryFunctionStage",
    "build_tile_function_module",
    "build_tile_module",
    "tile_function_reference",
]
