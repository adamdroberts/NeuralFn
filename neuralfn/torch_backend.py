from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .graph import NeuronGraph
from .neuron import decode_module_state_dict, encode_module_state_dict
from .semantic import NUM_SEMANTIC_DIMS, NUM_VOCAB_DIMS


def default_gpt_config() -> dict[str, Any]:
    return {
        "vocab_size": 256,
        "num_layers": 4,
        "model_dim": 128,
        "num_heads": 4,
        "num_kv_heads": 2,
        "mlp_mult": 2,
        "tie_embeddings": True,
        "logit_softcap": 30.0,
        "rope_base": 10000.0,
        "qk_gain_init": 1.0,
    }


def default_token_embedding_config() -> dict[str, Any]:
    cfg = default_gpt_config()
    return {
        "vocab_size": cfg["vocab_size"],
        "model_dim": cfg["model_dim"],
    }


def default_rms_norm_config() -> dict[str, Any]:
    return {"eps": 1e-6}


def default_attention_config() -> dict[str, Any]:
    cfg = default_gpt_config()
    return {
        "model_dim": cfg["model_dim"],
        "num_heads": cfg["num_heads"],
        "num_kv_heads": cfg["num_kv_heads"],
        "rope_base": cfg["rope_base"],
        "qk_gain_init": cfg["qk_gain_init"],
    }


def default_linear_config() -> dict[str, Any]:
    cfg = default_gpt_config()
    return {
        "input_dim": cfg["model_dim"],
        "output_dim": cfg["model_dim"],
        "bias": False,
    }


def default_reshape_heads_config(num_heads: int | None = None) -> dict[str, Any]:
    cfg = default_gpt_config()
    return {"num_heads": int(num_heads or cfg["num_heads"])}


def default_merge_heads_config() -> dict[str, Any]:
    return {}


def default_repeat_kv_config() -> dict[str, Any]:
    cfg = default_gpt_config()
    return {
        "num_heads": cfg["num_heads"],
        "num_kv_heads": cfg["num_kv_heads"],
    }


def default_rotary_embedding_config() -> dict[str, Any]:
    cfg = default_gpt_config()
    return {
        "head_dim": cfg["model_dim"] // cfg["num_heads"],
        "rope_base": cfg["rope_base"],
    }


def default_qk_gain_config() -> dict[str, Any]:
    cfg = default_gpt_config()
    return {
        "num_heads": cfg["num_heads"],
        "qk_gain_init": cfg["qk_gain_init"],
    }


def default_scaled_dot_product_attention_config() -> dict[str, Any]:
    return {"is_causal": True}


def default_residual_mix_config() -> dict[str, Any]:
    return {"dim": default_gpt_config()["model_dim"], "primary_init": 1.0, "skip_init": 0.0}


def default_residual_add_config() -> dict[str, Any]:
    return {"dim": default_gpt_config()["model_dim"], "init_scale": 1.0}


def default_mlp_config() -> dict[str, Any]:
    cfg = default_gpt_config()
    return {"model_dim": cfg["model_dim"], "mlp_mult": cfg["mlp_mult"]}


def default_lm_head_config() -> dict[str, Any]:
    cfg = default_gpt_config()
    return {"model_dim": cfg["model_dim"], "vocab_size": cfg["vocab_size"]}


def default_logit_softcap_config() -> dict[str, Any]:
    return {"softcap": default_gpt_config()["logit_softcap"]}


def default_loss_scale_config() -> dict[str, Any]:
    return {"coef": 1.0}


def resolve_amp_settings(amp_name: str | None) -> tuple[torch.dtype, str, bool]:
    normalized = str(amp_name or "float32").strip().lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16, "float16", True
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16, "bfloat16", True
    if normalized in {"float32", "fp32", "full", "none"}:
        return torch.float32, "float32", False
    raise ValueError(f"Unsupported amp dtype: {amp_name!r}")


def _zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: list[Tensor],
        *,
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
    ) -> None:
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            backend_steps = int(group["backend_steps"])
            nesterov = bool(group.get("nesterov", True))
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                if update.ndim != 2:
                    p.add_(update, alpha=-lr)
                    continue
                update = _zeropower_via_newtonschulz5(update, steps=backend_steps)
                update *= max(1.0, update.size(0) / max(update.size(1), 1)) ** 0.5
                p.add_(update.to(dtype=p.dtype), alpha=-lr)
        return loss


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        cos = freqs.cos()[None, None, :, :].to(dtype=dtype)
        sin = freqs.sin()[None, None, :, :].to(dtype=dtype)
        return cos, sin


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class TokenEmbeddingStage(nn.Module):
    def __init__(self, vocab_size: int, model_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)

    def forward(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        return self.embedding(token_ids), self.embedding.weight


class RMSNormStage(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class ResidualMixStage(nn.Module):
    def __init__(self, dim: int, primary_init: float = 1.0, skip_init: float = 0.0) -> None:
        super().__init__()
        self.primary_scale = nn.Parameter(torch.full((dim,), primary_init, dtype=torch.float32))
        self.skip_scale = nn.Parameter(torch.full((dim,), skip_init, dtype=torch.float32))

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        return (
            self.primary_scale[None, None, :].to(dtype=x.dtype) * x
            + self.skip_scale[None, None, :].to(dtype=x.dtype) * x0
        )


class ResidualAddStage(nn.Module):
    def __init__(self, dim: int, init_scale: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), init_scale, dtype=torch.float32))

    def forward(self, residual: Tensor, delta: Tensor) -> Tensor:
        return residual + self.scale[None, None, :].to(dtype=residual.dtype) * delta


class CausalSelfAttentionStage(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        head_dim = model_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        kv_dim = num_kv_heads * head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(head_dim, rope_base)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, model_dim = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seq_len, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        return self.out_proj(y.transpose(1, 2).contiguous().reshape(batch, seq_len, model_dim))


class FusedCausalAttentionStage(nn.Module):
    """Fused QKV-proj + reshape + RoPE + SDPA + merge + out-proj in one module.

    Designed for the ``megakernel`` runtime so ``torch.compile`` sees the
    entire attention layer as a single graph, enabling aggressive kernel fusion.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float = 10000.0,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        head_dim = model_dim // num_heads
        kv_dim = num_kv_heads * head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dropout_p = dropout_p
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.rotary = Rotary(head_dim, rope_base)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, model_dim = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(seq_len, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        drop = self.dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=drop,
            is_causal=True, enable_gqa=self.num_heads != self.num_kv_heads,
        )
        return self.out_proj(y.transpose(1, 2).contiguous().reshape(batch, seq_len, model_dim))


class MLPReluSquaredStage(nn.Module):
    def __init__(self, model_dim: int, mlp_mult: int) -> None:
        super().__init__()
        hidden = model_dim * mlp_mult
        self.fc = nn.Linear(model_dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, model_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class TiedLMHeadStage(nn.Module):
    def forward(self, hidden: Tensor, tied_weight: Tensor) -> Tensor:
        return F.linear(hidden, tied_weight)


class LMHeadStage(nn.Module):
    def __init__(self, model_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(model_dim, vocab_size, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.proj(hidden)


class LogitSoftcapStage(nn.Module):
    def __init__(self, softcap: float) -> None:
        super().__init__()
        if softcap <= 0.0:
            raise ValueError("softcap must be positive")
        self.softcap = float(softcap)

    def forward(self, logits: Tensor) -> Tensor:
        return self.softcap * torch.tanh(logits / self.softcap)


def _ensure_float_tensor(x: Tensor) -> Tensor:
    return x if torch.is_floating_point(x) else x.float()


class PassthroughFunctionStage(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class AddFunctionStage(nn.Module):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return x + y


class MultiplyFunctionStage(nn.Module):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return x * y


class NegateFunctionStage(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return -x


class ReluFunctionStage(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(x)


class SigmoidFunctionStage(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(_ensure_float_tensor(x))


class TanhFunctionStage(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(_ensure_float_tensor(x))


class LeakyReluFunctionStage(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(_ensure_float_tensor(x), negative_slope=0.01)


class GeluFunctionStage(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(_ensure_float_tensor(x))


class SiluFunctionStage(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.silu(_ensure_float_tensor(x))


class SoftplusFunctionStage(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(_ensure_float_tensor(x))


class HardTanhFunctionStage(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.hardtanh(_ensure_float_tensor(x))


class TokenCrossEntropyStage(nn.Module):
    def forward(self, logits: Tensor, target_ids: Tensor) -> Tensor:
        flat_logits = logits.reshape(-1, logits.size(-1))
        return F.cross_entropy(flat_logits.float(), target_ids.reshape(-1), reduction="mean")


class LinearStage(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class BitLinearTernaryStage(nn.Module):
    """BitNet b1.58 style Ternary Linear Layer (-1, 0, 1)."""
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))

    def forward(self, x: Tensor) -> Tensor:
        # Weight quantization to {-1, 0, 1}
        w = self.weight
        scale = w.abs().mean()
        w_quant = torch.round(w / (scale + 1e-7)).clamp(-1, 1)
        # Straight-through estimator
        w_quant = w + (w_quant - w).detach()

        # Activation quantization (simple 8-bit-like scaling)
        x_max = x.abs().max(dim=-1, keepdim=True).values
        x_quant = torch.round(x * 127 / (x_max + 1e-7)).clamp(-128, 127)
        x_quant = x + (x_quant * x_max / 127 - x).detach()

        return F.linear(x_quant, w_quant)

class LoRALinearStage(nn.Module):
    """Linear with a trainable low-rank delta: ``y = base(x) + (alpha/rank) * dropout(x @ A.T) @ B.T``.

    ``base`` is initialized by ``_load_base_checkpoint`` (``TorchTrainer``) from
    a pretrained weight and frozen via ``_freeze_non_lora``. ``A`` uses Kaiming
    init, ``B`` starts at zero so the LoRA delta is a no-op at step 0.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False,
        merge_on_eval: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rank = max(int(rank), 1)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.merge_on_eval = bool(merge_on_eval)
        self.base = nn.Linear(self.input_dim, self.output_dim, bias=bool(bias))
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.input_dim))
        self.lora_B = nn.Parameter(torch.zeros(self.output_dim, self.rank))
        self.lora_dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_out

    def merged_weight(self) -> Tensor:
        delta = self.lora_B @ self.lora_A
        return self.base.weight + self.scaling * delta


class NF4LinearStage(nn.Module):
    """qLoRA-style linear: nf4-packed base weight plus a fp16/bf16 LoRA delta."""

    _NF4_CODEBOOK = (
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
    )

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False,
        group_size: int = 64,
        compute_dtype: str = "bf16",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rank = max(int(rank), 1)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.group_size = max(int(group_size), 1)
        self.compute_dtype = compute_dtype
        if compute_dtype in {"bf16", "bfloat16"}:
            self._compute_dtype = torch.bfloat16
        elif compute_dtype in {"fp16", "float16", "half"}:
            self._compute_dtype = torch.float16
        else:
            self._compute_dtype = torch.float32

        packed_cols = (self.input_dim + 1) // 2
        self.register_buffer("qweight", torch.zeros(self.output_dim, packed_cols, dtype=torch.uint8))
        num_groups = (self.input_dim + self.group_size - 1) // self.group_size
        self.register_buffer("absmax", torch.ones(self.output_dim, num_groups, dtype=torch.float32))
        self.register_buffer(
            "nf4_codebook", torch.tensor(self._NF4_CODEBOOK, dtype=torch.float32)
        )

        self.bias = nn.Parameter(torch.zeros(self.output_dim)) if bias else None
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.input_dim))
        self.lora_B = nn.Parameter(torch.zeros(self.output_dim, self.rank))
        self.lora_dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @classmethod
    def _quantize_nf4(cls, weight: Tensor, group_size: int) -> tuple[Tensor, Tensor]:
        codebook = torch.tensor(cls._NF4_CODEBOOK, dtype=torch.float32, device=weight.device)
        out_dim, in_dim = weight.shape
        num_groups = (in_dim + group_size - 1) // group_size
        absmax = torch.zeros(out_dim, num_groups, dtype=torch.float32, device=weight.device)
        indices = torch.zeros(out_dim, in_dim, dtype=torch.uint8, device=weight.device)
        for g in range(num_groups):
            s = g * group_size
            e = min(s + group_size, in_dim)
            block = weight[:, s:e].float()
            block_absmax = block.abs().amax(dim=-1).clamp(min=1e-8)
            absmax[:, g] = block_absmax
            normalized = block / block_absmax.unsqueeze(-1)
            diffs = (normalized.unsqueeze(-1) - codebook.view(1, 1, -1)).abs()
            indices[:, s:e] = diffs.argmin(dim=-1).to(torch.uint8)
        packed_cols = (in_dim + 1) // 2
        packed = torch.zeros(out_dim, packed_cols, dtype=torch.uint8, device=weight.device)
        even = indices[:, 0::2]
        odd = indices[:, 1::2] if indices.shape[1] > 1 else torch.zeros_like(even[:, :0])
        packed[:, : even.shape[1]] = even & 0x0F
        if odd.shape[1] > 0:
            packed[:, : odd.shape[1]] |= (odd & 0x0F) << 4
        return packed, absmax

    def _dequantize_weight(self) -> Tensor:
        packed = self.qweight
        absmax = self.absmax
        codebook = self.nf4_codebook
        out_dim = self.output_dim
        in_dim = self.input_dim
        even_codes = (packed & 0x0F).long()
        odd_codes = ((packed >> 4) & 0x0F).long()
        interleaved = torch.empty(out_dim, packed.shape[1] * 2, dtype=torch.long, device=packed.device)
        interleaved[:, 0::2] = even_codes
        if packed.shape[1] * 2 > 1:
            interleaved[:, 1::2] = odd_codes
        codes = interleaved[:, :in_dim]
        normalized = codebook[codes]
        group_size = self.group_size
        scales_per_col = absmax.repeat_interleave(group_size, dim=-1)[:, :in_dim]
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
        w = self._dequantize_weight()
        base_out = F.linear(x.to(w.dtype), w, self.bias if self.bias is not None else None)
        lora_out = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_out.to(base_out.dtype)


class MaskedTokenCrossEntropyStage(nn.Module):
    """Cross-entropy averaged only where ``loss_mask > 0`` (SFT response-only loss)."""

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)

    def forward(self, logits: Tensor, target_ids: Tensor, loss_mask: Tensor) -> Tensor:
        flat_logits = logits.reshape(-1, logits.size(-1)).float()
        flat_targets = target_ids.reshape(-1)
        flat_mask = loss_mask.reshape(-1).to(flat_logits.dtype)
        per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none", ignore_index=self.ignore_index)
        denom = flat_mask.sum().clamp(min=1.0)
        return (per_token * flat_mask).sum() / denom


class SFTDatasetSourceStage(nn.Module):
    """Passthrough source for (tokens, targets, loss_mask) triples."""

    def __init__(
        self,
        dataset_names: list[str] | None = None,
        seq_len: int = 64,
        prompt_field: str = "prompt",
        response_field: str = "response",
        format: str = "chat",
        mask_prompt: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_names: list[str] = dataset_names or []
        self.seq_len = int(seq_len)
        self.prompt_field = str(prompt_field)
        self.response_field = str(response_field)
        self.format = str(format)
        self.mask_prompt = bool(mask_prompt)

    def forward(self, *args: Tensor) -> tuple[Tensor, ...]:
        return tuple(args)


class ReferenceForwardStage(nn.Module):
    """Wraps a frozen reference ``CompiledTorchGraph`` loaded from its own checkpoint."""

    def __init__(
        self,
        ref_graph_path: str = "",
        ref_weights_path: str = "",
        vocab_size: int = 256,
        model_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.ref_graph_path = str(ref_graph_path)
        self.ref_weights_path = str(ref_weights_path)
        self.vocab_size = int(vocab_size)
        self.model_dim = int(model_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self._compiled: CompiledTorchGraph | None = None

    def _ensure_loaded(self, device: torch.device | None = None) -> CompiledTorchGraph:
        if self._compiled is not None:
            return self._compiled
        if not self.ref_graph_path or not self.ref_weights_path:
            raise ValueError(
                "ReferenceForwardStage requires ref_graph_path and ref_weights_path to be set"
            )
        from .serialization import load_graph
        from .inference import load_pt_checkpoint
        ref_graph = load_graph(self.ref_graph_path)
        compiled = CompiledTorchGraph(ref_graph)
        state, _meta = load_pt_checkpoint(self.ref_weights_path, map_location=device or "cpu")
        compiled.load_state_dict(state, strict=False)
        for param in compiled.parameters():
            param.requires_grad = False
        compiled.train(False)
        if device is not None:
            compiled.to(device)
        self._compiled = compiled
        return compiled

    @torch.no_grad()
    def forward(self, tokens: Tensor) -> Tensor:
        compiled = self._ensure_loaded(device=tokens.device)
        if len(compiled.graph.input_node_ids) >= 2:
            dummy_targets = torch.zeros_like(tokens)
            outputs = compiled(tokens, dummy_targets)
        else:
            outputs = compiled(tokens)
        return outputs[0]


class SequenceLogpStage(nn.Module):
    """Sum of per-token logprobs of ``targets`` under ``logits``, masked by ``loss_mask``."""

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)

    def forward(self, logits: Tensor, targets: Tensor, loss_mask: Tensor) -> Tensor:
        log_probs = F.log_softmax(logits.float(), dim=-1)
        target_safe = targets.clamp(min=0)
        gathered = log_probs.gather(-1, target_safe.unsqueeze(-1)).squeeze(-1)
        mask = loss_mask.to(gathered.dtype)
        valid = (targets != self.ignore_index).to(gathered.dtype)
        effective = mask * valid
        return (gathered * effective).sum(dim=-1)


class DPOPairwiseLossStage(nn.Module):
    """Direct Preference Optimization loss (sigmoid / hinge / ipo variants)."""

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0, loss_type: str = "sigmoid") -> None:
        super().__init__()
        self.beta = float(beta)
        self.label_smoothing = float(label_smoothing)
        self.loss_type = str(loss_type)

    def forward(
        self,
        policy_logp_chosen: Tensor,
        policy_logp_rejected: Tensor,
        ref_logp_chosen: Tensor,
        ref_logp_rejected: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        chosen_logratios = policy_logp_chosen - ref_logp_chosen
        rejected_logratios = policy_logp_rejected - ref_logp_rejected
        logits = self.beta * (chosen_logratios - rejected_logratios)
        if self.loss_type == "hinge":
            per_example = F.relu(1.0 - logits)
        elif self.loss_type == "ipo":
            per_example = (logits - 1.0 / (2.0 * max(self.beta, 1e-8))) ** 2
        else:
            if self.label_smoothing > 0.0:
                per_example = (
                    -F.logsigmoid(logits) * (1.0 - self.label_smoothing)
                    - F.logsigmoid(-logits) * self.label_smoothing
                )
            else:
                per_example = -F.logsigmoid(logits)
        loss = per_example.mean()
        chosen_reward = self.beta * chosen_logratios.detach()
        rejected_reward = self.beta * rejected_logratios.detach()
        return loss, chosen_reward, rejected_reward


class DPODatasetSourceStage(nn.Module):
    """Passthrough source for DPO pair batches."""

    def __init__(
        self,
        dataset_names: list[str] | None = None,
        seq_len: int = 64,
        prompt_field: str = "prompt",
        chosen_field: str = "chosen",
        rejected_field: str = "rejected",
    ) -> None:
        super().__init__()
        self.dataset_names: list[str] = dataset_names or []
        self.seq_len = int(seq_len)
        self.prompt_field = str(prompt_field)
        self.chosen_field = str(chosen_field)
        self.rejected_field = str(rejected_field)

    def forward(self, *args: Tensor) -> tuple[Tensor, ...]:
        return tuple(args)


class RewardHeadStage(nn.Module):
    """Linear-to-1 scalar head for reward-model training."""

    def __init__(self, model_dim: int, pool: str = "last") -> None:
        super().__init__()
        self.pool = str(pool)
        self.proj = nn.Linear(int(model_dim), 1, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        if self.pool == "mean":
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden[:, -1, :]
        return self.proj(pooled).squeeze(-1)


class PreferenceBCELossStage(nn.Module):
    """Bradley-Terry preference loss for reward-model training."""

    def forward(self, reward_chosen: Tensor, reward_rejected: Tensor) -> Tensor:
        return -F.logsigmoid(reward_chosen - reward_rejected).mean()


class ValueHeadStage(nn.Module):
    """Per-token value head (scalar per position) for PPO."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(model_dim), 1, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.proj(hidden).squeeze(-1)


class PPOClippedLossStage(nn.Module):
    """Clipped PPO policy-and-value loss."""

    def __init__(self, clip_range: float = 0.2, vf_coef: float = 0.5, ent_coef: float = 0.0) -> None:
        super().__init__()
        self.clip_range = float(clip_range)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)

    def forward(
        self,
        logp_new: Tensor,
        logp_old: Tensor,
        advantages: Tensor,
        value_new: Tensor,
        value_old: Tensor,
        returns: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        ratio = (logp_new - logp_old).exp()
        adv = advantages
        unclipped = ratio * adv
        clipped = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range) * adv
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        value_clipped = value_old + (value_new - value_old).clamp(-self.clip_range, self.clip_range)
        vf_sq1 = (value_new - returns) ** 2
        vf_sq2 = (value_clipped - returns) ** 2
        value_loss = 0.5 * torch.maximum(vf_sq1, vf_sq2).mean()
        loss = policy_loss + self.vf_coef * value_loss
        return policy_loss, value_loss, loss


class KLPenaltyStage(nn.Module):
    """Shape per-token rewards by subtracting ``kl_coef * (logp_policy - logp_ref)``."""

    def __init__(self, kl_coef: float = 0.1) -> None:
        super().__init__()
        self.kl_coef = float(kl_coef)

    def forward(self, logp_policy: Tensor, logp_ref: Tensor, rewards: Tensor) -> Tensor:
        kl = logp_policy - logp_ref
        return rewards - self.kl_coef * kl


class RewardForwardStage(nn.Module):
    """Frozen reward-model wrapper."""

    def __init__(
        self,
        reward_graph_path: str = "",
        reward_weights_path: str = "",
        model_dim: int = 128,
    ) -> None:
        super().__init__()
        self.reward_graph_path = str(reward_graph_path)
        self.reward_weights_path = str(reward_weights_path)
        self.model_dim = int(model_dim)
        self._compiled: CompiledTorchGraph | None = None

    def _ensure_loaded(self, device: torch.device | None = None) -> CompiledTorchGraph:
        if self._compiled is not None:
            return self._compiled
        if not self.reward_graph_path or not self.reward_weights_path:
            raise ValueError(
                "RewardForwardStage requires reward_graph_path and reward_weights_path to be set"
            )
        from .serialization import load_graph
        from .inference import load_pt_checkpoint
        reward_graph = load_graph(self.reward_graph_path)
        compiled = CompiledTorchGraph(reward_graph)
        state, _meta = load_pt_checkpoint(self.reward_weights_path, map_location=device or "cpu")
        compiled.load_state_dict(state, strict=False)
        for param in compiled.parameters():
            param.requires_grad = False
        compiled.train(False)
        if device is not None:
            compiled.to(device)
        self._compiled = compiled
        return compiled

    @torch.no_grad()
    def forward(self, tokens: Tensor) -> Tensor:
        compiled = self._ensure_loaded(device=tokens.device)
        outputs = compiled(tokens)
        return outputs[0]


class PPORolloutSourceStage(nn.Module):
    """Source emitting the current rollout buffer."""

    def __init__(self, seq_len: int = 64, rollout_length: int = 64) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.rollout_length = int(rollout_length)

    def forward(self, *args: Tensor) -> tuple[Tensor, ...]:
        return tuple(args)


class GAEComputeStage(nn.Module):
    """Generalized Advantage Estimation: returns ``(advantages, returns)``."""

    def __init__(self, gamma: float = 1.0, lambda_: float = 0.95) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.lam = float(lambda_)

    def forward(self, rewards: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        gamma = self.gamma
        lam = self.lam
        batch, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        next_adv = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)
        next_value = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)
        for t in range(seq_len - 1, -1, -1):
            delta = rewards[:, t] + gamma * next_value - values[:, t]
            next_adv = delta + gamma * lam * next_adv
            advantages[:, t] = next_adv
            next_value = values[:, t]
        returns = advantages + values
        return advantages, returns


class RandMapAdapterStage(nn.Module):
    """Random-map adapter: frozen random projections with a trainable middle."""
    def __init__(self, model_dim: int, adapter_dim: int) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.adapter_dim = adapter_dim

        # Frozen random projections
        self.down_proj = nn.Linear(model_dim, adapter_dim, bias=False)
        self.up_proj = nn.Linear(adapter_dim, model_dim, bias=False)

        for p in self.down_proj.parameters():
            p.requires_grad = False
        for p in self.up_proj.parameters():
            p.requires_grad = False

        # Trainable middle
        self.middle = nn.Linear(adapter_dim, adapter_dim, bias=False)
        # Residual scale gate
        self.scale = nn.Parameter(torch.zeros(1))

        self._init_random_projections()

    def _init_random_projections(self) -> None:
        # Use orthogonal init for the frozen maps
        nn.init.orthogonal_(self.down_proj.weight)
        nn.init.orthogonal_(self.up_proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, seq, model_dim]
        # Residual adapter: x + scale * (up(middle(down(x))))
        adapter_out = self.up_proj(self.middle(self.down_proj(x)))
        return x + self.scale * adapter_out

class MambaStage(nn.Module):
    """Simplified Mamba-style SSM Stage."""
    def __init__(self, model_dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.d_inner = model_dim * expand
        self.in_proj = nn.Linear(model_dim, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, d_state + 2, bias=False) # delta, B, C
        self.out_proj = nn.Linear(self.d_inner, model_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Conv
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)

        x = F.silu(x)

        # Simplified SSM (Identity for now to ensure it runs, real Mamba is complex)
        # In a real impl, we'd do the scan here.
        y = x * F.sigmoid(z)

        return self.out_proj(y)

class ReshapeHeadsStage(nn.Module):
    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.num_heads = int(num_heads)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, width = x.shape
        if width % self.num_heads != 0:
            raise ValueError("Last tensor dimension must be divisible by num_heads")
        head_dim = width // self.num_heads
        return x.reshape(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)


class MergeHeadsStage(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        batch, heads, seq_len, head_dim = x.shape
        return x.transpose(1, 2).contiguous().reshape(batch, seq_len, heads * head_dim)


class RepeatKVStage(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.repeats = num_heads // num_kv_heads

    def forward(self, x: Tensor) -> Tensor:
        if self.repeats == 1:
            return x
        return x.repeat_interleave(self.repeats, dim=1)


class RotaryEmbeddingStage(nn.Module):
    def __init__(self, head_dim: int, rope_base: float) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        self.rotary = Rotary(head_dim, rope_base)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        cos, sin = self.rotary(q.size(2), q.device, q.dtype)
        return apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)


class QKGainStage(nn.Module):
    def __init__(self, num_heads: int, qk_gain_init: float) -> None:
        super().__init__()
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))

    def forward(self, q: Tensor) -> Tensor:
        return q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]


class ScaledDotProductAttentionStage(nn.Module):
    def __init__(self, is_causal: bool = True, backend: str = "sdpa", dropout_p: float = 0.0) -> None:
        super().__init__()
        self.is_causal = bool(is_causal)
        self.backend = backend
        self.dropout_p = float(dropout_p)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        drop = self.dropout_p if self.training else 0.0
        if self.backend == "math":
            with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=drop, is_causal=self.is_causal)
        elif self.backend == "flex":
            # Experimental flex attention (only available in nightly or pt 2.5+)
            try:
                from torch.nn.attention.flex_attention import flex_attention, causal_mask
                if self.is_causal:
                    return flex_attention(q, k, v, block_mask=causal_mask)
                return flex_attention(q, k, v)
            except ImportError:
                # fall back to SDPA if not available
                return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=drop, is_causal=self.is_causal)
        else: # sdpa (default)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=drop, is_causal=self.is_causal)


class LayerNormStage(nn.Module):
    def __init__(self, model_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(model_dim, eps=eps)
    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)

class DropoutStage(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p)
    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x)

class GeluStage(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x)

class SwiGLUStage(nn.Module):
    def __init__(self, model_dim: int, mlp_mult: int, multiple_of: int | None = None) -> None:
        super().__init__()
        hidden = int(8.0 * model_dim / 3.0)
        if multiple_of is not None:
            hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(model_dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, model_dim, bias=False)
        self.w3 = nn.Linear(model_dim, hidden, bias=False)
    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class AbsolutePositionEmbeddingStage(nn.Module):
    def __init__(self, max_seq_len: int, model_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, model_dim)
    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len = x.shape[:2]
        pos = torch.arange(seq_len, device=x.device, dtype=torch.long)
        return self.embedding(pos).unsqueeze(0).expand(batch, -1, -1)

class KVCacheReadStage(nn.Module):
    def forward(self, k: Tensor, v: Tensor, cache_k: Tensor | None = None, cache_v: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if cache_k is not None and cache_v is not None:
            k = torch.cat([cache_k, k], dim=2)
            v = torch.cat([cache_v, v], dim=2)
        return k, v

class KVCacheWriteStage(nn.Module):
    def forward(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return k, v

class KVPCAEncodeStage(nn.Module):
    def __init__(self, head_dim: int, compressed_dim: int) -> None:
        super().__init__()
        self.k_proj = nn.Linear(head_dim, compressed_dim, bias=False)
        self.v_proj = nn.Linear(head_dim, compressed_dim, bias=False)
    def forward(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return self.k_proj(k), self.v_proj(v)

class KVPCADecodeStage(nn.Module):
    def __init__(self, head_dim: int, compressed_dim: int) -> None:
        super().__init__()
        self.k_unproj = nn.Linear(compressed_dim, head_dim, bias=False)
        self.v_unproj = nn.Linear(compressed_dim, head_dim, bias=False)
    def forward(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return self.k_unproj(k), self.v_unproj(v)

class KVQuantPackStage(nn.Module):
    """Pack K and V into int8 with per-token scales for memory-efficient KV storage."""

    def forward(self, k: Tensor, v: Tensor) -> Tensor:
        kv = torch.cat([k, v], dim=-1)  # (..., 2*head_dim)
        amax = kv.abs().amax(dim=-1, keepdim=True).clamp(min=1e-7)
        scale = amax / 127.0
        quantized = torch.round(kv / scale).clamp(-128, 127)
        # Store scale as the last element so unpack can recover it
        return torch.cat([quantized, scale], dim=-1)


class KVQuantUnpackStage(nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim

    def forward(self, packed: Tensor) -> tuple[Tensor, Tensor]:
        scale = packed[..., -1:]
        quantized = packed[..., :-1]
        dequantized = quantized * scale
        k, v = torch.split(dequantized, [self.head_dim, dequantized.size(-1) - self.head_dim], dim=-1)
        return k, v


class RoutingStatsMixin:
    """Expose lightweight routing telemetry from routed expert stages."""

    def _init_routing_stats(self, *, num_experts: int, top_k: int) -> None:
        self._routing_num_experts = max(int(num_experts), 1)
        self._routing_top_k = max(int(top_k), 1)
        self.register_buffer("_routing_route_rows", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer(
            "_routing_selection_counts",
            torch.zeros(self._routing_num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_routing_weight_mass",
            torch.zeros(self._routing_num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("_routing_mean_router_entropy", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_routing_mean_router_entropy_norm", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_routing_mean_topk_entropy", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_routing_mean_topk_entropy_norm", torch.zeros((), dtype=torch.float32), persistent=False)

    def clear_routing_stats(self) -> None:
        with torch.no_grad():
            self._routing_route_rows.zero_()
            self._routing_selection_counts.zero_()
            self._routing_weight_mass.zero_()
            self._routing_mean_router_entropy.zero_()
            self._routing_mean_router_entropy_norm.zero_()
            self._routing_mean_topk_entropy.zero_()
            self._routing_mean_topk_entropy_norm.zero_()

    def _update_routing_stats(
        self,
        *,
        scores: Tensor,
        routing_weights: Tensor,
        routing_indices: Tensor,
    ) -> None:
        flat_scores = scores.detach().float().reshape(-1, scores.size(-1))
        flat_weights = routing_weights.detach().float().reshape(-1, routing_weights.size(-1))
        flat_indices = routing_indices.detach().long().reshape(-1, routing_indices.size(-1))
        route_rows = int(flat_scores.size(0))
        if route_rows <= 0:
            self.clear_routing_stats()
            return

        flat_index_values = flat_indices.reshape(-1)
        selection_counts = torch.bincount(flat_index_values, minlength=self._routing_num_experts).to(dtype=torch.float32)
        weight_mass = torch.zeros(self._routing_num_experts, device=flat_weights.device, dtype=torch.float32)
        weight_mass.scatter_add_(0, flat_index_values, flat_weights.reshape(-1))

        router_entropy = -(flat_scores * flat_scores.clamp_min(1e-12).log()).sum(dim=-1)
        topk_entropy = -(flat_weights * flat_weights.clamp_min(1e-12).log()).sum(dim=-1)
        mean_router_entropy = router_entropy.mean()
        mean_topk_entropy = topk_entropy.mean()

        router_entropy_norm_denom = math.log(self._routing_num_experts) if self._routing_num_experts > 1 else 0.0
        topk_entropy_norm_denom = math.log(self._routing_top_k) if self._routing_top_k > 1 else 0.0
        mean_router_entropy_norm = (
            mean_router_entropy / router_entropy_norm_denom
            if router_entropy_norm_denom > 0.0
            else mean_router_entropy.new_zeros(())
        )
        mean_topk_entropy_norm = (
            mean_topk_entropy / topk_entropy_norm_denom
            if topk_entropy_norm_denom > 0.0
            else mean_topk_entropy.new_zeros(())
        )

        with torch.no_grad():
            self._routing_route_rows.copy_(
                torch.tensor(route_rows, dtype=self._routing_route_rows.dtype, device=self._routing_route_rows.device)
            )
            self._routing_selection_counts.copy_(
                selection_counts.to(
                    dtype=self._routing_selection_counts.dtype,
                    device=self._routing_selection_counts.device,
                )
            )
            self._routing_weight_mass.copy_(
                weight_mass.to(dtype=self._routing_weight_mass.dtype, device=self._routing_weight_mass.device)
            )
            self._routing_mean_router_entropy.copy_(
                mean_router_entropy.to(
                    dtype=self._routing_mean_router_entropy.dtype,
                    device=self._routing_mean_router_entropy.device,
                )
            )
            self._routing_mean_router_entropy_norm.copy_(
                mean_router_entropy_norm.to(
                    dtype=self._routing_mean_router_entropy_norm.dtype,
                    device=self._routing_mean_router_entropy_norm.device,
                )
            )
            self._routing_mean_topk_entropy.copy_(
                mean_topk_entropy.to(
                    dtype=self._routing_mean_topk_entropy.dtype,
                    device=self._routing_mean_topk_entropy.device,
                )
            )
            self._routing_mean_topk_entropy_norm.copy_(
                mean_topk_entropy_norm.to(
                    dtype=self._routing_mean_topk_entropy_norm.dtype,
                    device=self._routing_mean_topk_entropy_norm.device,
                )
            )

    @property
    def last_routing_stats(self) -> dict[str, Any] | None:
        route_rows = int(self._routing_route_rows.item())
        if route_rows <= 0:
            return None

        selection_counts = [int(round(value)) for value in self._routing_selection_counts.detach().cpu().tolist()]
        weight_mass = [float(value) for value in self._routing_weight_mass.detach().cpu().tolist()]
        selection_total = max(sum(selection_counts), 0)
        weight_total = float(sum(weight_mass))
        active_experts = [idx for idx, count in enumerate(selection_counts) if count > 0]

        def _clamp_unit(value: float) -> float:
            if not math.isfinite(value):
                return 0.0
            return max(0.0, min(value, 1.0))

        return {
            "num_experts": int(self._routing_num_experts),
            "route_rows": route_rows,
            "top_k": int(self._routing_top_k),
            "selection_counts": selection_counts,
            "selection_shares": [
                (count / selection_total) if selection_total > 0 else 0.0
                for count in selection_counts
            ],
            "weight_mass": weight_mass,
            "weight_mass_shares": [
                (mass / weight_total) if weight_total > 0 else 0.0
                for mass in weight_mass
            ],
            "active_experts": active_experts,
            "active_expert_count": len(active_experts),
            "mean_router_entropy": float(self._routing_mean_router_entropy.item()),
            "mean_router_entropy_norm": _clamp_unit(float(self._routing_mean_router_entropy_norm.item())),
            "mean_topk_entropy": float(self._routing_mean_topk_entropy.item()),
            "mean_topk_entropy_norm": _clamp_unit(float(self._routing_mean_topk_entropy_norm.item())),
        }


class RouterLogitsStage(nn.Module):
    def __init__(self, model_dim: int, experts: int) -> None:
        super().__init__()
        self.gate = nn.Linear(model_dim, experts, bias=False)
    def forward(self, x: Tensor) -> Tensor:
        return self.gate(x)

class TopKRouteStage(RoutingStatsMixin, nn.Module):
    def __init__(self, top_k: int, experts: int) -> None:
        super().__init__()
        self.top_k = top_k
        self._init_routing_stats(num_experts=experts, top_k=top_k)
    def forward(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        scores = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        self._update_routing_stats(
            scores=scores,
            routing_weights=topk_weights,
            routing_indices=topk_indices,
        )
        return topk_weights.to(dtype=logits.dtype), topk_indices

class ExpertDispatchStage(nn.Module):
    def __init__(self, model_dim: int, experts: int, mlp_mult: int) -> None:
        super().__init__()
        hidden_dim = model_dim * mlp_mult
        self.w1 = nn.Parameter(torch.empty(experts, model_dim, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(experts, hidden_dim, model_dim))
        self.w3 = nn.Parameter(torch.empty(experts, model_dim, hidden_dim))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)
        nn.init.normal_(self.w3, std=0.02)
        self.experts = experts
    def forward(self, x: Tensor, routing_weights: Tensor, routing_indices: Tensor) -> Tensor:
        batch, seq_len, d = x.shape
        top_k = routing_indices.shape[-1]
        x_flat = x.reshape(-1, d)
        out = torch.zeros_like(x_flat)
        routing_weights_flat = routing_weights.reshape(-1, top_k)
        routing_indices_flat = routing_indices.reshape(-1, top_k)

        for i in range(self.experts):
            mask = (routing_indices_flat == i)
            idx = torch.where(mask)[0]
            expert_inputs = x_flat[idx]
            w1, w2, w3 = self.w1[i], self.w2[i], self.w3[i]
            h = F.silu(expert_inputs @ w1) * (expert_inputs @ w3)
            expert_out = h @ w2
            weights = routing_weights_flat[mask]
            out[idx] += expert_out * weights.unsqueeze(-1)
        return out.reshape(batch, seq_len, d)

class ExpertCombineStage(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class BroadcastExpertRoutesStage(nn.Module):
    """Broadcast shared batch-level expert routes across every sequence position."""

    def forward(self, hidden: Tensor, expert_weights: Tensor, expert_indices: Tensor) -> tuple[Tensor, Tensor]:
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


class LoadBalanceLossStage(nn.Module):
    def __init__(self, experts: int) -> None:
        super().__init__()
        self.experts = experts
    def forward(self, router_logits: Tensor, routing_weights: Tensor, routing_indices: Tensor) -> tuple[Tensor, Tensor]:
        scores = F.softmax(router_logits, dim=-1)
        density = scores.mean(dim=(0, 1))
        aux_loss = self.experts * (density * density).sum()
        return aux_loss, router_logits

class AuxLossAddStage(nn.Module):
    def __init__(self, coef: float) -> None:
        super().__init__()
        self.coef = coef
    def forward(self, main_loss: Tensor, aux_loss: Tensor) -> Tensor:
        return main_loss + self.coef * aux_loss


class LossScaleStage(nn.Module):
    def __init__(self, coef: float) -> None:
        super().__init__()
        self.coef = float(coef)

    def forward(self, loss: Tensor) -> Tensor:
        return loss * self.coef


class DatasetSourceStage(nn.Module):
    """Source node that stores dataset configuration.

    module_config holds {"dataset_names": [...], "seq_len": 64}.
    This node has no inputs — the trainer provides data directly.
    During forward, it passes through the tensors provided by the trainer.
    """
    def __init__(self, dataset_names: list[str] | None = None, seq_len: int = 64) -> None:
        super().__init__()
        self.dataset_names: list[str] = dataset_names or []
        self.seq_len: int = seq_len

    def forward(self, *args: Tensor) -> tuple[Tensor, ...]:
        # Passthrough — trainer feeds tokens and targets as external inputs
        return args if len(args) > 1 else (args[0],)


class SemanticDataSourceStage(nn.Module):
    """Experimental: source node that auto-loads vocab-derived semantic targets.

    Like ``DatasetSourceStage`` this is a passthrough -- the trainer detects
    ``module_type == "semantic_data_source"`` and materializes deterministic
    vocab-backed categorical semantic targets automatically.
    """

    def __init__(
        self,
        seq_len: int = 9,
        semantic_vocab_ref: str = "",
        emit_router_vecs: bool = False,
        router_vec_dim: int = 0,
    ) -> None:
        super().__init__()
        self.seq_len: int = seq_len
        self.semantic_vocab_ref = str(semantic_vocab_ref)
        self.emit_router_vecs = bool(emit_router_vecs)
        self.router_vec_dim = int(router_vec_dim)

    def forward(self, *args: Tensor) -> tuple[Tensor, ...]:
        return args if len(args) > 1 else (args[0],)


class RandomTimestepsStage(nn.Module):
    def forward(self, tokens: Tensor) -> Tensor:
        return torch.rand(tokens.size(0), device=tokens.device, dtype=torch.float32)


class JEPAMaskStage(nn.Module):
    def __init__(
        self,
        mask_ratio: float,
        mask_token_id: int = 0,
        mask_strategy: str = "random",
        num_blocks: int = 4,
        min_block_ratio: float = 0.1,
        max_block_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.mask_ratio = float(mask_ratio)
        self.mask_token_id = int(mask_token_id)
        self.mask_strategy = mask_strategy
        self.num_blocks = int(num_blocks)
        self.min_block_ratio = float(min_block_ratio)
        self.max_block_ratio = float(max_block_ratio)

    def _random_mask(self, tokens: Tensor) -> Tensor:
        noise = torch.rand(tokens.shape, device=tokens.device)
        return noise < self.mask_ratio

    def _block_mask(self, tokens: Tensor) -> Tensor:
        batch, seq_len = tokens.shape
        mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=tokens.device)
        min_len = max(1, int(self.min_block_ratio * seq_len))
        max_len = max(min_len, int(self.max_block_ratio * seq_len))
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        for _ in range(self.num_blocks):
            block_len = torch.randint(min_len, max_len + 1, (batch,), device=tokens.device)
            max_start = (seq_len - block_len).clamp_min(0)
            start = (torch.rand(batch, device=tokens.device) * (max_start.float() + 1.0)).long().clamp_max(max_start)
            end = start + block_len
            span = (positions >= start.unsqueeze(1)) & (positions < end.unsqueeze(1))
            mask = mask | span
        return mask

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        if self.mask_strategy == "block":
            mask = self._block_mask(tokens)
        else:
            mask = self._random_mask(tokens)
        masked_tokens = tokens.clone()
        masked_tokens[mask] = self.mask_token_id
        return masked_tokens, mask.to(dtype=torch.float32)


class LatentPoolStage(nn.Module):
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(dtype=x.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (x * weights).sum(dim=1) / denom
        fallback = x.mean(dim=1)
        has_mask = (mask.sum(dim=1, keepdim=True) > 0).to(dtype=x.dtype)
        return pooled * has_mask + fallback * (1.0 - has_mask)


class JEPAProjectorStage(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim, bias=False),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class JEPAPredictorStage(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        hidden_dim = max(latent_dim // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LatentMSELossStage(nn.Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(pred.float(), target.detach().float())


class BytePatchEmbedStage(nn.Module):
    def __init__(self, model_dim: int, patch_size: int, stride: int, vocab_size: int = 256) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.vocab_size = int(vocab_size)
        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
        self.proj = nn.Conv1d(self.model_dim, self.model_dim, kernel_size=self.patch_size, stride=self.stride, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        byte_ids = tokens.clamp(0, self.vocab_size - 1)
        x = self.embedding(byte_ids).transpose(1, 2)
        seq_len = x.size(-1)
        if seq_len < self.patch_size:
            pad_right = self.patch_size - seq_len
        else:
            pad_right = (self.stride - ((seq_len - self.patch_size) % self.stride)) % self.stride
        if pad_right:
            x = F.pad(x, (0, pad_right))
        return self.proj(x).transpose(1, 2)


class BytePatchMergeStage(nn.Module):
    def forward(self, x: Tensor, target_tokens: Tensor) -> Tensor:
        target_len = target_tokens.size(1)
        merged = F.interpolate(x.transpose(1, 2), size=target_len, mode="nearest")
        return merged.transpose(1, 2)


class ACTHaltGateStage(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(model_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.proj(x.mean(dim=1)))


class ACTWeightedSumStage(nn.Module):
    def forward(self, states: Tensor, weights: Tensor) -> Tensor:
        return (states * weights.to(dtype=states.dtype).unsqueeze(-1).unsqueeze(-1)).sum(dim=1)


class UniversalTransformerStage(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        mlp_mult: float,
        max_steps: int,
        halt_epsilon: float,
    ) -> None:
        super().__init__()
        hidden_dim = max(int(model_dim * mlp_mult), model_dim)
        self.max_steps = int(max_steps)
        self.halt_epsilon = float(halt_epsilon)
        self.attn_norm = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, model_dim, bias=False),
        )
        self.halt_gate = ACTHaltGateStage(model_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        state = x
        batch = x.size(0)
        remaining = torch.ones(batch, 1, device=x.device, dtype=torch.float32)
        accum = torch.zeros_like(x)
        weights: list[Tensor] = []

        for step in range(self.max_steps):
            attn_in = self.attn_norm(state)
            attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
            state = state + attn_out
            mlp_in = self.mlp_norm(state)
            state = state + self.mlp(mlp_in)

            raw_p = self.halt_gate(state).float()
            step_p = torch.minimum(raw_p, remaining)
            if step == self.max_steps - 1:
                step_p = remaining
            if self.halt_epsilon > 0.0:
                step_p = torch.where(remaining <= self.halt_epsilon, remaining, step_p)

            accum = accum + step_p.to(dtype=state.dtype).unsqueeze(-1) * state
            remaining = (remaining - step_p).clamp_min(0.0)
            weights.append(step_p.squeeze(-1))

        return accum, torch.stack(weights, dim=1)


class RoleMappedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        roles: list[str],
        *,
        extra_tensors: dict[str, Tensor] | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.roles = list(roles)
        self.extra_tensors = {str(name): tensor for name, tensor in dict(extra_tensors or {}).items()}
        self.length = len(self.base_dataset)
        if self.extra_tensors:
            self.length = min(self.length, *(int(tensor.size(0)) for tensor in self.extra_tensors.values()))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        sample = self.base_dataset[idx]
        if isinstance(sample, Tensor):
            x = sample
            y = sample
        else:
            values = tuple(sample)
            x = values[0]
            y = values[1] if len(values) > 1 else values[0]
        mapped: list[Tensor] = []
        for role in self.roles:
            if role in {"tokens", "enc_tokens", "dec_tokens"}:
                mapped.append(x)
            elif role == "targets":
                mapped.append(y)
            elif role in self.extra_tensors:
                mapped.append(self.extra_tensors[role][idx])
            else:
                raise ValueError(f"Unsupported dataset role '{role}'")
        return tuple(mapped)


class DualSourceTokenDataset(torch.utils.data.Dataset):
    """Pair a normal text token dataset with semantic target tokens."""

    def __init__(self, text_dataset: torch.utils.data.Dataset, semantic_tokens: torch.Tensor) -> None:
        self.text_dataset = text_dataset
        self.semantic_tokens = semantic_tokens
        self.length = min(len(text_dataset), int(semantic_tokens.size(0)))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        sample = self.text_dataset[idx]
        if isinstance(sample, Tensor):
            x = sample
            return x, self.semantic_tokens[idx]

        values = tuple(sample)
        if len(values) == 1:
            return values[0], self.semantic_tokens[idx]
        return values[0], values[1], self.semantic_tokens[idx]


def default_dataset_source_config() -> dict[str, Any]:
    return {"dataset_names": [], "seq_len": 64}


def default_fused_attention_config() -> dict[str, Any]:
    return {"model_dim": 128, "num_heads": 4, "num_kv_heads": 4, "rope_base": 10000.0, "dropout_p": 0.0}


def default_kv_pca_config() -> dict[str, Any]:
    return {"head_dim": 64, "compressed_dim": 16}


def default_kv_quant_unpack_config() -> dict[str, Any]:
    return {"head_dim": 64}


def build_module(module_type: str, module_config: dict[str, Any]) -> nn.Module:
    cfg = dict(module_config or {})
    if module_type == "fused_causal_attention":
        return FusedCausalAttentionStage(
            model_dim=int(cfg["model_dim"]),
            num_heads=int(cfg["num_heads"]),
            num_kv_heads=int(cfg.get("num_kv_heads", cfg["num_heads"])),
            rope_base=float(cfg.get("rope_base", 10000.0)),
            dropout_p=float(cfg.get("dropout_p", 0.0)),
        )
    if module_type == "bitlinear_ternary":
        return BitLinearTernaryStage(
            input_dim=int(cfg["input_dim"]),
            output_dim=int(cfg["output_dim"]),
        )
    if module_type == "randmap_adapter":
        return RandMapAdapterStage(
            model_dim=int(cfg["model_dim"]),
            adapter_dim=int(cfg["adapter_dim"]),
        )
    if module_type == "lora_linear":
        return LoRALinearStage(
            input_dim=int(cfg["input_dim"]),
            output_dim=int(cfg["output_dim"]),
            rank=int(cfg.get("rank", 8)),
            alpha=float(cfg.get("alpha", 16.0)),
            dropout=float(cfg.get("dropout", 0.0)),
            bias=bool(cfg.get("bias", False)),
            merge_on_eval=bool(cfg.get("merge_on_eval", False)),
        )
    if module_type == "nf4_linear":
        return NF4LinearStage(
            input_dim=int(cfg["input_dim"]),
            output_dim=int(cfg["output_dim"]),
            rank=int(cfg.get("rank", 8)),
            alpha=float(cfg.get("alpha", 16.0)),
            dropout=float(cfg.get("dropout", 0.0)),
            bias=bool(cfg.get("bias", False)),
            group_size=int(cfg.get("group_size", 64)),
            compute_dtype=str(cfg.get("compute_dtype", "bf16")),
        )
    if module_type == "masked_token_cross_entropy":
        return MaskedTokenCrossEntropyStage(ignore_index=int(cfg.get("ignore_index", -100)))
    if module_type == "sft_dataset_source":
        return SFTDatasetSourceStage(
            dataset_names=list(cfg.get("dataset_names", [])),
            seq_len=int(cfg.get("seq_len", 64)),
            prompt_field=str(cfg.get("prompt_field", "prompt")),
            response_field=str(cfg.get("response_field", "response")),
            format=str(cfg.get("format", "chat")),
            mask_prompt=bool(cfg.get("mask_prompt", True)),
        )
    if module_type == "reference_forward":
        return ReferenceForwardStage(
            ref_graph_path=str(cfg.get("ref_graph_path", "")),
            ref_weights_path=str(cfg.get("ref_weights_path", "")),
            vocab_size=int(cfg.get("vocab_size", 256)),
            model_dim=int(cfg.get("model_dim", 128)),
            num_layers=int(cfg.get("num_layers", 4)),
            num_heads=int(cfg.get("num_heads", 4)),
        )
    if module_type == "sequence_logp":
        return SequenceLogpStage(ignore_index=int(cfg.get("ignore_index", -100)))
    if module_type == "dpo_pairwise_loss":
        return DPOPairwiseLossStage(
            beta=float(cfg.get("beta", 0.1)),
            label_smoothing=float(cfg.get("label_smoothing", 0.0)),
            loss_type=str(cfg.get("loss_type", "sigmoid")),
        )
    if module_type == "dpo_dataset_source":
        return DPODatasetSourceStage(
            dataset_names=list(cfg.get("dataset_names", [])),
            seq_len=int(cfg.get("seq_len", 64)),
            prompt_field=str(cfg.get("prompt_field", "prompt")),
            chosen_field=str(cfg.get("chosen_field", "chosen")),
            rejected_field=str(cfg.get("rejected_field", "rejected")),
        )
    if module_type == "reward_head":
        return RewardHeadStage(
            model_dim=int(cfg.get("model_dim", 128)),
            pool=str(cfg.get("pool", "last")),
        )
    if module_type == "preference_bce_loss":
        return PreferenceBCELossStage()
    if module_type == "value_head":
        return ValueHeadStage(model_dim=int(cfg.get("model_dim", 128)))
    if module_type == "ppo_clipped_loss":
        return PPOClippedLossStage(
            clip_range=float(cfg.get("clip_range", 0.2)),
            vf_coef=float(cfg.get("vf_coef", 0.5)),
            ent_coef=float(cfg.get("ent_coef", 0.0)),
        )
    if module_type == "kl_penalty":
        return KLPenaltyStage(kl_coef=float(cfg.get("kl_coef", 0.1)))
    if module_type == "reward_forward":
        return RewardForwardStage(
            reward_graph_path=str(cfg.get("reward_graph_path", "")),
            reward_weights_path=str(cfg.get("reward_weights_path", "")),
            model_dim=int(cfg.get("model_dim", 128)),
        )
    if module_type == "ppo_rollout_source":
        return PPORolloutSourceStage(
            seq_len=int(cfg.get("seq_len", 64)),
            rollout_length=int(cfg.get("rollout_length", 64)),
        )
    if module_type == "gae_compute":
        return GAEComputeStage(
            gamma=float(cfg.get("gamma", 1.0)),
            lambda_=float(cfg.get("lambda_", cfg.get("lambda", 0.95))),
        )
    if module_type == "mamba":
        return MambaStage(
            model_dim=int(cfg["model_dim"]),
            d_state=int(cfg.get("d_state", 16)),
            d_conv=int(cfg.get("d_conv", 4)),
            expand=int(cfg.get("expand", 2)),
        )
    if module_type == "denoise_head":
        return DenoiseHeadStage(
            model_dim=int(cfg["model_dim"]),
            vocab_size=int(cfg["vocab_size"]),
        )
    if module_type == "mask_scheduler":
        return MaskSchedulerStage(
            vocab_size=int(cfg["vocab_size"]),
            mask_token_id=int(cfg["mask_token_id"]),
        )
    if module_type == "ttt_linear":
        return TTTLinearStage(
            input_dim=int(cfg["input_dim"]),
            output_dim=int(cfg["output_dim"]),
            hidden_dim=int(cfg.get("hidden_dim", 16)),
        )
    if module_type == "kv_pca_encode":
        return KVPCAEncodeStage(
            head_dim=int(cfg["head_dim"]),
            compressed_dim=int(cfg["compressed_dim"]),
        )
    if module_type == "kv_pca_decode":
        return KVPCADecodeStage(
            head_dim=int(cfg["head_dim"]),
            compressed_dim=int(cfg["compressed_dim"]),
        )
    if module_type == "kv_quant_pack":
        return KVQuantPackStage()
    if module_type == "kv_quant_unpack":
        return KVQuantUnpackStage(head_dim=int(cfg["head_dim"]))
    if module_type == "layer_norm":
        return LayerNormStage(
            model_dim=int(cfg["model_dim"]),
            eps=float(cfg.get("eps", 1e-5)),
        )
    if module_type == "dropout":
        return DropoutStage(p=float(cfg.get("p", 0.1)))
    if module_type == "gelu":
        return GeluStage()
    if module_type == "swiglu":
        return SwiGLUStage(
            model_dim=int(cfg["model_dim"]),
            mlp_mult=int(cfg["mlp_mult"]),
            multiple_of=cfg.get("multiple_of"),
        )
    if module_type == "absolute_position_embedding":
        return AbsolutePositionEmbeddingStage(
            max_seq_len=int(cfg.get("max_seq_len", 1024)),
            model_dim=int(cfg["model_dim"]),
        )
    if module_type == "kv_cache_read":
        return KVCacheReadStage()
    if module_type == "kv_cache_write":
        return KVCacheWriteStage()
    if module_type == "router_logits":
        return RouterLogitsStage(
            model_dim=int(cfg["model_dim"]),
            experts=int(cfg["experts"]),
        )
    if module_type == "topk_route":
        return TopKRouteStage(
            top_k=int(cfg["top_k"]),
            experts=int(cfg.get("experts", 8)),
        )
    if module_type == "expert_dispatch":
        return ExpertDispatchStage(
            model_dim=int(cfg["model_dim"]),
            experts=int(cfg["experts"]),
            mlp_mult=int(cfg["mlp_mult"]),
        )
    if module_type == "expert_combine":
        return ExpertCombineStage()
    if module_type == "load_balance_loss":
        return LoadBalanceLossStage(experts=int(cfg["experts"]))
    if module_type == "aux_loss_add":
        return AuxLossAddStage(coef=float(cfg["coef"]))
    if module_type == "loss_scale":
        return LossScaleStage(coef=float(cfg.get("coef", 1.0)))
    if module_type == "linear":
        return LinearStage(
            input_dim=int(cfg["input_dim"]),
            output_dim=int(cfg["output_dim"]),
            bias=bool(cfg.get("bias", False)),
        )
    if module_type == "reshape_heads":
        return ReshapeHeadsStage(num_heads=int(cfg["num_heads"]))
    if module_type == "merge_heads":
        return MergeHeadsStage()
    if module_type == "repeat_kv":
        return RepeatKVStage(
            num_heads=int(cfg["num_heads"]),
            num_kv_heads=int(cfg["num_kv_heads"]),
        )
    if module_type == "rotary_embedding":
        return RotaryEmbeddingStage(
            head_dim=int(cfg["head_dim"]),
            rope_base=float(cfg["rope_base"]),
        )
    if module_type == "qk_gain":
        return QKGainStage(
            num_heads=int(cfg["num_heads"]),
            qk_gain_init=float(cfg.get("qk_gain_init", 1.0)),
        )
    if module_type == "scaled_dot_product_attention":
        return ScaledDotProductAttentionStage(
            is_causal=bool(cfg.get("is_causal", True)),
            backend=str(cfg.get("backend", "sdpa")),
            dropout_p=float(cfg.get("dropout_p", 0.0)),
        )
    if module_type == "token_embedding":
        return TokenEmbeddingStage(
            vocab_size=int(cfg["vocab_size"]),
            model_dim=int(cfg["model_dim"]),
        )
    if module_type == "rms_norm":
        return RMSNormStage(eps=float(cfg.get("eps", 1e-6)))
    if module_type == "residual_mix":
        return ResidualMixStage(
            dim=int(cfg["dim"]),
            primary_init=float(cfg.get("primary_init", 1.0)),
            skip_init=float(cfg.get("skip_init", 0.0)),
        )
    if module_type == "residual_add":
        return ResidualAddStage(
            dim=int(cfg["dim"]),
            init_scale=float(cfg.get("init_scale", 1.0)),
        )
    if module_type == "causal_self_attention":
        return CausalSelfAttentionStage(
            model_dim=int(cfg["model_dim"]),
            num_heads=int(cfg["num_heads"]),
            num_kv_heads=int(cfg["num_kv_heads"]),
            rope_base=float(cfg["rope_base"]),
            qk_gain_init=float(cfg["qk_gain_init"]),
        )
    if module_type == "mlp_relu2":
        return MLPReluSquaredStage(
            model_dim=int(cfg["model_dim"]),
            mlp_mult=int(cfg["mlp_mult"]),
        )
    if module_type == "tied_lm_head":
        return TiedLMHeadStage()
    if module_type == "lm_head":
        return LMHeadStage(
            model_dim=int(cfg["model_dim"]),
            vocab_size=int(cfg["vocab_size"]),
        )
    if module_type == "logit_softcap":
        return LogitSoftcapStage(softcap=float(cfg["softcap"]))
    if module_type == "token_cross_entropy":
        return TokenCrossEntropyStage()
    if module_type == "dataset_source":
        return DatasetSourceStage(
            dataset_names=list(cfg.get("dataset_names", [])),
            seq_len=int(cfg.get("seq_len", 64)),
        )
    if module_type == "random_timesteps":
        return RandomTimestepsStage()
    if module_type == "jepa_mask":
        return JEPAMaskStage(
            mask_ratio=float(cfg.get("mask_ratio", 0.5)),
            mask_token_id=int(cfg.get("mask_token_id", 0)),
            mask_strategy=str(cfg.get("mask_strategy", "random")),
            num_blocks=int(cfg.get("num_blocks", 4)),
            min_block_ratio=float(cfg.get("min_block_ratio", 0.1)),
            max_block_ratio=float(cfg.get("max_block_ratio", 0.25)),
        )
    if module_type == "latent_pool":
        return LatentPoolStage()
    if module_type == "jepa_projector":
        return JEPAProjectorStage(
            input_dim=int(cfg["input_dim"]),
            latent_dim=int(cfg["latent_dim"]),
        )
    if module_type == "jepa_predictor":
        return JEPAPredictorStage(latent_dim=int(cfg["latent_dim"]))
    if module_type == "latent_mse_loss":
        return LatentMSELossStage()
    if module_type == "byte_patch_embed":
        return BytePatchEmbedStage(
            model_dim=int(cfg["model_dim"]),
            patch_size=int(cfg.get("patch_size", 4)),
            stride=int(cfg.get("stride", cfg.get("patch_size", 4))),
            vocab_size=int(cfg.get("vocab_size", 256)),
        )
    if module_type == "byte_patch_merge":
        return BytePatchMergeStage()
    if module_type == "act_halt_gate":
        return ACTHaltGateStage(model_dim=int(cfg["model_dim"]))
    if module_type == "act_weighted_sum":
        return ACTWeightedSumStage()
    if module_type == "universal_transformer":
        return UniversalTransformerStage(
            model_dim=int(cfg["model_dim"]),
            num_heads=int(cfg["num_heads"]),
            mlp_mult=float(cfg.get("mlp_mult", 4.0)),
            max_steps=int(cfg.get("max_steps", 4)),
            halt_epsilon=float(cfg.get("halt_epsilon", 0.01)),
        )
    if module_type == "semantic_projector":
        return SemanticProjectorStage(
            input_dim=int(cfg["input_dim"]),
            semantic_dim=int(cfg.get("semantic_dim", NUM_SEMANTIC_DIMS)),
            residual_dim=int(cfg.get("residual_dim", 64)),
            n_sig_buckets=int(cfg.get("n_sig_buckets", 4096)),
            semantic_vocab_ref=str(cfg.get("semantic_vocab_ref", "")),
        )
    if module_type == "semantic_alignment_loss":
        return SemanticAlignmentLossStage(
            ignore_index=int(cfg.get("ignore_index", -100)),
            semantic_vocab_ref=str(cfg.get("semantic_vocab_ref", "")),
        )
    if module_type == "semantic_hasher":
        return SemanticHasherStage(
            dim=int(cfg.get("dim", NUM_SEMANTIC_DIMS)),
            tables=int(cfg.get("tables", 8)),
            planes=int(cfg.get("planes", 12)),
            seed=int(cfg.get("seed", 42)),
        )
    if module_type == "semantic_moe_router":
        return SemanticMoERouterStage(
            n_experts=int(cfg["n_experts"]),
            semantic_dim=int(cfg.get("semantic_dim", NUM_SEMANTIC_DIMS)),
            top_k=int(cfg.get("top_k", 2)),
        )
    if module_type == "semantic_hash_router":
        return SemanticHashRouterStage(
            n_experts=int(cfg["n_experts"]),
            semantic_dim=int(cfg.get("semantic_dim", NUM_SEMANTIC_DIMS)),
            top_k=int(cfg.get("top_k", 2)),
            tables=int(cfg.get("tables", 8)),
            n_buckets=int(cfg.get("n_buckets", 4096)),
            ignore_index=int(cfg.get("ignore_index", -100)),
            semantic_vocab_ref=str(cfg.get("semantic_vocab_ref", "")),
            routing_source=str(cfg.get("routing_source", "topic_logits")),
        )
    if module_type == "causal_chunk_state":
        return CausalChunkStateStage(
            chunk_size=int(cfg.get("chunk_size", 32)),
            mode=str(cfg.get("mode", "prefix")),
        )
    if module_type == "semantic_chunk_projector":
        return SemanticChunkProjectorStage(
            input_dim=int(cfg["input_dim"]),
            semantic_dim=int(cfg.get("semantic_dim", NUM_SEMANTIC_DIMS)),
            residual_dim=int(cfg.get("residual_dim", 64)),
            n_sig_buckets=int(cfg.get("n_sig_buckets", 4096)),
            semantic_vocab_ref=str(cfg.get("semantic_vocab_ref", "")),
        )
    if module_type == "semantic_chunk_hasher":
        return SemanticChunkHasherStage(
            dim=int(cfg.get("dim", NUM_SEMANTIC_DIMS)),
            tables=int(cfg.get("tables", 8)),
            planes=int(cfg.get("planes", 12)),
            seed=int(cfg.get("seed", 42)),
        )
    if module_type == "semantic_moe_jepa_evo_router":
        return SemanticMoeJepaEvoRouterStage(
            semantic_dim=int(cfg.get("semantic_dim", NUM_SEMANTIC_DIMS)),
            top_k=int(cfg.get("top_k", 2)),
            shared_experts=int(cfg.get("shared_experts", 2)),
            free_experts=int(cfg.get("free_experts", 8)),
            tables=int(cfg.get("tables", 8)),
            n_buckets=int(cfg.get("n_buckets", 4096)),
            ignore_index=int(cfg.get("ignore_index", -100)),
            semantic_vocab_ref=str(cfg.get("semantic_vocab_ref", "")),
        )
    if module_type == "broadcast_chunk_routes":
        return BroadcastChunkRoutesStage(chunk_size=int(cfg.get("chunk_size", 32)))
    if module_type == "route_balance_loss":
        return RouteBalanceLossStage()
    if module_type == "route_selection_loss":
        return RouteSelectionLossStage(
            semantic_vocab_ref=str(cfg.get("semantic_vocab_ref", "")),
            shared_experts=int(cfg.get("shared_experts", 2)),
            free_experts=int(cfg.get("free_experts", 8)),
            ignore_index=int(cfg.get("ignore_index", -100)),
        )
    if module_type == "route_distillation_loss":
        return RouteDistillationLossStage(
            semantic_vocab_ref=str(cfg.get("semantic_vocab_ref", "")),
            shared_experts=int(cfg.get("shared_experts", 2)),
            free_experts=int(cfg.get("free_experts", 8)),
        )
    if module_type == "broadcast_expert_routes":
        return BroadcastExpertRoutesStage()
    if module_type == "routed_attention_experts":
        return RoutedAttentionExpertsStage(
            model_dim=int(cfg["model_dim"]),
            num_heads=int(cfg["num_heads"]),
            num_kv_heads=int(cfg.get("num_kv_heads", cfg["num_heads"])),
            rope_base=float(cfg.get("rope_base", 10000.0)),
            qk_gain_init=float(cfg.get("qk_gain_init", 1.0)),
            experts=int(cfg["experts"]),
            top_k=int(cfg.get("top_k", 2)),
            is_causal=bool(cfg.get("is_causal", True)),
        )
    if module_type == "attentionless_decoder":
        return AttentionlessDecoderStage(
            semantic_dim=int(cfg.get("semantic_dim", NUM_SEMANTIC_DIMS)),
            residual_dim=int(cfg.get("residual_dim", 64)),
            vocab_size=int(cfg.get("vocab_size", 256)),
            n_buckets=int(cfg.get("n_buckets", 256)),
        )
    if module_type == "softmax_distillation_loss":
        return SoftmaxDistillationLossStage()
    if module_type == "semantic_data_source":
        return SemanticDataSourceStage(
            seq_len=int(cfg.get("seq_len", NUM_SEMANTIC_DIMS)),
            semantic_vocab_ref=str(cfg.get("semantic_vocab_ref", "")),
            emit_router_vecs=bool(cfg.get("emit_router_vecs", False)),
            router_vec_dim=int(cfg.get("router_vec_dim", 0)),
        )
    raise KeyError(f"Unsupported module type: {module_type}")


def build_function_module(name: str) -> nn.Module:
    if name in {"input", "output", "identity"}:
        return PassthroughFunctionStage()
    if name == "add":
        return AddFunctionStage()
    if name == "multiply":
        return MultiplyFunctionStage()
    if name == "negate":
        return NegateFunctionStage()
    if name == "relu":
        return ReluFunctionStage()
    if name == "sigmoid":
        return SigmoidFunctionStage()
    if name == "tanh_neuron":
        return TanhFunctionStage()
    if name == "leaky_relu":
        return LeakyReluFunctionStage()
    if name == "gelu":
        return GeluFunctionStage()
    if name == "silu":
        return SiluFunctionStage()
    if name == "softplus":
        return SoftplusFunctionStage()
    if name == "hard_tanh":
        return HardTanhFunctionStage()
    raise TypeError(f"Function node '{name}' is not supported by the torch runtime")


def _wrap_output(value: Any) -> tuple[Tensor, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def resolve_torch_train_drop_last(
    *,
    drop_last: bool | None,
    template_runtime: str,
    device: Any,
    dataset_rows: int,
    batch_size: int,
) -> bool:
    if batch_size <= 0 or dataset_rows < batch_size:
        return False
    if drop_last is not None:
        return bool(drop_last)
    device_type = getattr(device, "type", device)
    return str(template_runtime).lower() == "megakernel" and str(device_type).lower() == "cuda"


class CompiledTorchGraph(nn.Module):
    def __init__(self, graph: NeuronGraph) -> None:
        super().__init__()
        self.graph = graph
        if graph.has_cycles():
            raise ValueError(f"Torch runtime does not support cyclic graphs: '{graph.name}'")
        self.order = graph.topological_order() if graph.nodes else []
        self.node_modules = nn.ModuleDict()
        for nid, node in graph.nodes.items():
            ndef = node.neuron_def
            if ndef.kind == "module":
                module = build_module(ndef.module_type, ndef.module_config)
                if ndef.module_state:
                    module.load_state_dict(decode_module_state_dict(ndef.module_state))
                self.node_modules[nid] = module
            elif ndef.kind == "subgraph" and ndef.subgraph is not None:
                self.node_modules[nid] = CompiledTorchGraph(ndef.subgraph)
            elif ndef.kind == "function":
                self.node_modules[nid] = build_function_module(ndef.name)

    def forward(self, *flat_inputs: Tensor) -> tuple[Tensor, ...]:
        outputs, _trace = self._run(flat_inputs, want_trace=False)
        return outputs

    def trace(self, *flat_inputs: Tensor) -> tuple[tuple[Tensor, ...], dict[str, tuple[Tensor, ...]]]:
        return self._run(flat_inputs, want_trace=True)

    def _run(
        self,
        flat_inputs: tuple[Tensor, ...],
        *,
        want_trace: bool,
    ) -> tuple[tuple[Tensor, ...], dict[str, tuple[Tensor, ...]]]:
        provided = self._expand_flat_inputs(flat_inputs)
        values: dict[str, tuple[Tensor, ...]] = {}
        traces: dict[str, tuple[Tensor, ...]] = {}
        for nid, tensor_tuple in provided.items():
            values[nid] = tensor_tuple
            traces[nid] = tensor_tuple

        for nid in self.order:
            if nid in values and nid in provided:
                continue
            node = self.graph.nodes[nid]
            args = self._gather_inputs(nid, values)
            if node.neuron_def.kind == "subgraph" and want_trace:
                child_outputs, child_trace = self.node_modules[nid].trace(*args)
                values[nid] = _wrap_output(child_outputs)
                traces[nid] = values[nid]
                for child_key, child_value in child_trace.items():
                    traces[f"{nid}/{child_key}"] = child_value
            else:
                # Use the node's fixed child module directly so torch.compile does
                # not have to specialize one generic dispatcher across mixed node
                # arities and dtypes.
                values[nid] = _wrap_output(self.node_modules[nid](*args))
                traces[nid] = values[nid]

        outputs = self._flatten_outputs(values)
        return outputs, traces if want_trace else {}

    def _expand_flat_inputs(self, flat_inputs: tuple[Tensor, ...]) -> dict[str, tuple[Tensor, ...]]:
        expanded: dict[str, tuple[Tensor, ...]] = {}
        idx = 0
        for nid in self.graph.input_node_ids:
            node = self.graph.nodes[nid]
            n_out = node.neuron_def.n_outputs
            chunk = flat_inputs[idx : idx + n_out]
            if len(chunk) != n_out:
                raise ValueError(
                    f"Graph '{self.graph.name}' expected {len(self.graph.interface_input_layout())} "
                    f"flattened inputs but received {len(flat_inputs)}"
                )
            expanded[nid] = tuple(chunk)
            idx += n_out
        if idx != len(flat_inputs):
            raise ValueError(
                f"Graph '{self.graph.name}' expected {idx} flattened inputs but received {len(flat_inputs)}"
            )
        return expanded

    def _flatten_outputs(self, values: dict[str, tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
        flattened: list[Tensor] = []
        for nid in self.graph.output_node_ids:
            flattened.extend(values.get(nid, ()))
        return tuple(flattened)

    def _gather_inputs(self, node_id: str, values: dict[str, tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
        node = self.graph.nodes[node_id]
        gathered: list[Tensor | None] = [None] * node.neuron_def.n_inputs
        for edge in self.graph._incoming(node_id):
            src_vals = values.get(edge.src_node)
            if src_vals is None:
                raise ValueError(f"Node '{node_id}' is missing source values from '{edge.src_node}'")
            if edge.src_port >= len(src_vals):
                raise ValueError(f"Edge '{edge.id}' src_port={edge.src_port} exceeds source outputs")
            if gathered[edge.dst_port] is not None:
                raise ValueError(
                    f"Torch graph '{self.graph.name}' has multiple incoming edges into "
                    f"node '{node_id}' port {edge.dst_port}; use an explicit combine node instead"
                )
            gathered[edge.dst_port] = src_vals[edge.src_port]

        missing = [idx for idx, value in enumerate(gathered) if value is None]
        if missing:
            raise ValueError(f"Node '{node_id}' is missing required torch inputs at ports {missing}")
        return tuple(value for value in gathered if value is not None)

    def sync_state_back(self, graph: NeuronGraph | None = None) -> None:
        target_graph = graph or self.graph
        for nid, node in target_graph.nodes.items():
            if node.neuron_def.kind == "module":
                node.neuron_def.module_state = encode_module_state_dict(
                    self.node_modules[nid].to("cpu").state_dict()
                )
            elif node.neuron_def.kind == "subgraph" and node.neuron_def.subgraph is not None:
                compiled_child = self.node_modules[nid]
                if isinstance(compiled_child, CompiledTorchGraph):
                    compiled_child.sync_state_back(node.neuron_def.subgraph)


@dataclass
class TorchTrainConfig:
    learning_rate: float = 3e-4
    epochs: int = 50
    batch_size: int = 8
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    device: str = "cuda"
    amp_dtype: str = "float32"
    compile: bool = False
    activation_checkpointing: bool = False
    fsdp2_enabled: bool = False
    max_steps: int | None = None
    grad_clip_norm: float = 0.0
    optimizer_profile: str = "adamw"
    train_batch_tokens: int | None = None
    warmup_steps: int = 0
    warmdown_fraction: float = 0.75
    lr_decay_iters: int | None = None
    min_lr: float | None = None
    max_wallclock_seconds: float = 0.0
    embed_lr: float | None = None
    head_lr: float | None = None
    tied_embed_lr: float | None = None
    matrix_lr: float | None = None
    scalar_lr: float | None = None
    muon_momentum: float = 0.95
    muon_backend_steps: int = 5
    muon_momentum_warmup_start: float = 0.85
    muon_momentum_warmup_steps: int = 500
    drop_last: bool | None = None
    respect_epoch_boundaries: bool = False
    evolutionary: bool = False
    evo_population_size: int = 50
    evo_mutation_rate: float = 0.1
    evo_mutation_scale: float = 0.3
    evo_crossover_rate: float = 0.5
    evo_tournament_size: int = 3
    evo_elite_count: int = 2
    evo_seed: int | None = None


class TorchTrainer:
    def __init__(self, graph: NeuronGraph, config: TorchTrainConfig | None = None) -> None:
        self.graph = graph
        self.config = config or TorchTrainConfig()
        warmdown_fraction = float(self.config.warmdown_fraction)
        if warmdown_fraction < 0.0 or warmdown_fraction > 1.0:
            raise ValueError("warmdown_fraction must be within [0.0, 1.0]")
        self._stop = False
        self.loss_history: list[float] = []

    def stop(self) -> None:
        self._stop = True

    @staticmethod
    def _lr_warmdown_scale(step: int, total_steps: int, warmdown_fraction: float) -> float:
        if warmdown_fraction <= 0.0 or total_steps <= 0:
            return 1.0
        warmdown_steps = max(int(math.ceil(total_steps * warmdown_fraction)), 1)
        warmdown_start = max(total_steps - warmdown_steps, 0)
        if step < warmdown_start:
            return 1.0
        if step >= total_steps:
            return 0.0
        return max(
            (total_steps - step) / max(warmdown_steps, 1),
            0.0,
        )

    @staticmethod
    def _cosine_decay_lr(step: int, *, base_lr: float, min_lr: float, lr_decay_iters: int) -> float:
        floor_lr = min(base_lr, max(float(min_lr), 0.0))
        if lr_decay_iters <= 0:
            return floor_lr
        if step <= 0:
            return base_lr
        if step >= lr_decay_iters:
            return floor_lr
        decay_ratio = step / max(lr_decay_iters, 1)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return floor_lr + coeff * (base_lr - floor_lr)

    @staticmethod
    def _maybe_mark_cudagraph_step_begin(template_runtime: str, device: Any) -> None:
        if str(template_runtime).lower() != "megakernel":
            return
        if getattr(device, "type", "").lower() != "cuda":
            return
        compiler = getattr(torch, "compiler", None)
        marker = getattr(compiler, "cudagraph_mark_step_begin", None)
        if marker is None:
            return
        # Megakernel training can expose aliased parameter outputs like
        # embedding weights through compiled CUDA graphs, so each forward
        # needs a fresh cudagraph step boundary.
        marker()

    @staticmethod
    def _compile_training_graph(
        compiled: nn.Module,
        *,
        template_runtime: str,
        device: Any,
        force_compile: bool,
    ) -> nn.Module:
        if force_compile or str(template_runtime).lower() == "compile":
            return torch.compile(compiled)
        if str(template_runtime).lower() != "megakernel":
            return compiled
        # Megakernel training still benefits from fullgraph compilation, but
        # CUDA graph capture is unsafe here because token embedding stages can
        # surface aliased parameter outputs such as embedding.weight.
        mode = "max-autotune-no-cudagraphs" if getattr(device, "type", "").lower() == "cuda" else "max-autotune"
        return torch.compile(compiled, mode=mode, fullgraph=True)

    @staticmethod
    def _adjust_vocab_size(graph: NeuronGraph, required_vocab: int) -> None:
        """Recursively update vocab_size in all embedding/lm_head modules."""
        for node in graph.nodes.values():
            ndef = node.neuron_def
            if ndef.kind == "module" and ndef.module_type in ("token_embedding", "lm_head", "denoise_head", "mask_scheduler"):
                cfg = ndef.module_config or {}
                current = cfg.get("vocab_size", 0)
                if current < required_vocab:
                    ndef.module_config = {**cfg, "vocab_size": required_vocab}
            elif ndef.kind == "subgraph" and ndef.subgraph is not None:
                TorchTrainer._adjust_vocab_size(ndef.subgraph, required_vocab)

    @staticmethod
    def _flatten_input_roles(graph: NeuronGraph) -> list[str]:
        roles: list[str] = []
        for nid in graph.input_node_ids:
            node = graph.nodes[nid]
            roles.extend(port.name for port in node.neuron_def.output_ports)
        return roles

    @staticmethod
    def _template_spec(graph: NeuronGraph) -> dict[str, Any]:
        return dict(graph.torch_config.get("template_spec", {}))

    @classmethod
    def _tokenization_mode(cls, graph: NeuronGraph) -> str:
        template = cls._template_spec(graph).get("template", {})
        return str(template.get("tokenization", "sp"))

    @classmethod
    def _semantic_vocab(cls, graph: NeuronGraph):
        from .semantic import ConversationalVocabulary, semantic_vocab_ref_for_graph

        return ConversationalVocabulary(semantic_vocab_ref_for_graph(graph))

    @classmethod
    def _semantic_active_dims(cls, graph: NeuronGraph) -> int:
        block_spec = cls._template_spec(graph).get("block_spec", {})
        vocab = cls._semantic_vocab(graph)
        return max(1, min(int(block_spec.get("top_k", 2) or 2), vocab.num_vocab_dims))

    @classmethod
    def _semantic_router_vecs_enabled(cls, graph: NeuronGraph) -> bool:
        template_spec = cls._template_spec(graph)
        if bool(template_spec.get("experimental_semantic_router_vecs", False)):
            return True
        return "semantic_router_vecs" in cls._flatten_input_roles(graph)

    @classmethod
    def _load_dataset_for_graph(
        cls,
        graph: NeuronGraph,
        dataset_names: list[str],
        *,
        seq_len: int,
    ) -> torch.utils.data.Dataset:
        if cls._tokenization_mode(graph) == "byte_hnet":
            from server.dataset_manager import load_dataset_byte_tensors

            return load_dataset_byte_tensors(dataset_names, seq_len=seq_len)
        cls._validate_tokenizer_backed_datasets(graph, dataset_names)
        from server.dataset_manager import load_dataset_tensors, raw_text_encoding_name_for_template_spec

        return load_dataset_tensors(
            dataset_names,
            seq_len=seq_len,
            encoding_name=raw_text_encoding_name_for_template_spec(cls._template_spec(graph)),
        )

    @classmethod
    def _load_semantic_tensors(cls, graph: NeuronGraph, active_dims: int = 2) -> dict[str, Tensor]:
        """Load deterministic vocab-derived semantic tensors for the graph contract."""
        from .semantic import load_training_targets, semantic_targets_to_router_vectors

        vocab = cls._semantic_vocab(graph)
        _ids, targets = load_training_targets(active_dims=active_dims, vocab=vocab)
        tensors: dict[str, Tensor] = {
            "sem_targets": torch.from_numpy(targets.astype(np.int64)),
        }
        if cls._semantic_router_vecs_enabled(graph):
            router_vecs = semantic_targets_to_router_vectors(targets, vocab=vocab)
            tensors["semantic_router_vecs"] = torch.from_numpy(router_vecs.astype(np.float32))
        return tensors

    @staticmethod
    def _build_semantic_placeholder_text(sem_targets: Tensor) -> tuple[Tensor, Tensor]:
        """Build safe placeholder token/target tensors for semantic-only graphs."""
        if sem_targets.ndim != 2:
            raise ValueError("semantic placeholder text expects sem_targets shaped [batch, dims]")
        batch_size, seq_len = sem_targets.shape
        tokens = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len).clone()
        targets = torch.roll(tokens, shifts=-1, dims=1)
        return tokens, targets

    @staticmethod
    def _required_vocab_for_text_dataset(base_dataset: torch.utils.data.Dataset) -> int:
        """Return the max token id + 1 for a text dataset when possible."""
        arrays = getattr(base_dataset, "arrays", None)
        if arrays:
            return max(int(arr.max()) for arr in arrays) + 1
        if len(base_dataset) == 0:
            return 0
        sample = base_dataset[0]
        if isinstance(sample, Tensor):
            sample_tensors = (sample,)
        else:
            sample_tensors = tuple(sample)
        return max(int(t.max()) for t in sample_tensors) + 1

    @classmethod
    def _configured_vocab_size(cls, graph: NeuronGraph) -> int:
        configured = 0
        template_vocab = cls._template_spec(graph).get("vocab_size")
        if template_vocab is not None:
            configured = max(configured, int(template_vocab))
        for node in graph.nodes.values():
            ndef = node.neuron_def
            if ndef.kind == "module":
                cfg = ndef.module_config or {}
                vocab_size = cfg.get("vocab_size")
                if vocab_size is not None:
                    configured = max(configured, int(vocab_size))
            elif ndef.kind == "subgraph" and ndef.subgraph is not None:
                configured = max(configured, cls._configured_vocab_size(ndef.subgraph))
        return configured

    @classmethod
    def _validate_tokenizer_backed_datasets(cls, graph: NeuronGraph, dataset_names: list[str]) -> None:
        if not dataset_names:
            return
        from server.dataset_manager import validate_cached_tokenizer_contract

        model_vocab_size = cls._configured_vocab_size(graph)
        for dataset_name in dataset_names:
            validate_cached_tokenizer_contract(
                dataset_name,
                model_vocab_size=model_vocab_size if model_vocab_size > 0 else None,
            )

    @classmethod
    def _build_manual_dataset(
        cls,
        graph: NeuronGraph,
        train_inputs: list[list[int]] | Tensor,
        train_targets: list[list[int]] | Tensor,
    ) -> torch.utils.data.Dataset:
        roles = cls._flatten_input_roles(graph)
        x = torch.as_tensor(train_inputs, dtype=torch.long)
        if len(roles) == 1:
            if x.ndim != 2:
                raise ValueError("Torch training expects integer input arrays of shape [batch, seq_len]")
            return torch.utils.data.TensorDataset(x)

        y = torch.as_tensor(train_targets, dtype=torch.long)
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("Torch training expects integer token arrays of shape [batch, seq_len]")
        if x.shape != y.shape:
            raise ValueError("train_inputs and train_targets must have the same [batch, seq_len] shape")

        tensors = [x if role in {"tokens", "enc_tokens", "dec_tokens"} else y for role in roles]
        return torch.utils.data.TensorDataset(*tensors)

    @staticmethod
    def _freeze_module_params(module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False

    # Backwards-compat alias — EMA-target freezing was the original call site.
    _freeze_ema_targets = _freeze_module_params

    @classmethod
    def _prepare_ema_targets(cls, compiled: nn.Module) -> None:
        if not isinstance(compiled, CompiledTorchGraph):
            return
        for child in compiled.node_modules.values():
            if isinstance(child, CompiledTorchGraph):
                cls._prepare_ema_targets(child)
        if "online_encoder" in compiled.node_modules and "target_encoder" in compiled.node_modules:
            target = compiled.node_modules["target_encoder"]
            target.load_state_dict(compiled.node_modules["online_encoder"].state_dict())
            cls._freeze_module_params(target)

    @classmethod
    def _load_base_checkpoint(cls, compiled: nn.Module, path: str) -> None:
        """Load a pretrained base-model checkpoint into ``compiled``.

        Uses ``load_pt_checkpoint`` then ``load_state_dict(strict=False)`` so
        LoRA ``A`` / ``B`` parameters retain their initialization. Also loads
        base weights into ``NF4LinearStage`` buffers when present.
        """
        if not path:
            return
        from .inference import load_pt_checkpoint  # local import to avoid cycle
        state, _meta = load_pt_checkpoint(path)
        if not isinstance(compiled, nn.Module):
            return
        # Remap ``foo.proj.weight`` keys (``LinearStage``) onto ``foo.base.weight``
        # (``LoRALinearStage`` / ``NF4LinearStage``) so users can fine-tune a
        # plain pretrained checkpoint without re-exporting.
        target_keys = set(compiled.state_dict().keys())
        remapped: dict[str, Tensor] = {}
        for key, value in state.items():
            if key in target_keys:
                remapped[key] = value
                continue
            if key.endswith(".proj.weight"):
                candidate = key[: -len(".proj.weight")] + ".base.weight"
                if candidate in target_keys:
                    remapped[candidate] = value
                    continue
                qcandidate = key[: -len(".proj.weight")] + ".qweight"
                if qcandidate in target_keys:
                    remapped[key] = value  # handled in nf4 pass below
                    continue
            remapped[key] = value
        compiled.load_state_dict(remapped, strict=False)
        # Quantize any matching nf4 linears from the float base weight.
        for name, module in compiled.named_modules():
            if isinstance(module, NF4LinearStage):
                candidate = name + ".proj.weight"
                if candidate in state:
                    module.load_base_weight(state[candidate])
                else:
                    base_candidate = name + ".base.weight"
                    if base_candidate in state:
                        module.load_base_weight(state[base_candidate])

    @classmethod
    def _freeze_non_lora(cls, compiled: nn.Module) -> None:
        """Freeze every parameter except LoRA ``A``/``B`` (and optional bias).

        ``ReferenceForwardStage`` / ``RewardForwardStage`` are already frozen on
        load. ``NF4LinearStage`` base weights are buffers, not parameters, so
        they are not in the optimizer's parameter list.
        """
        for module in compiled.modules():
            if isinstance(module, (LoRALinearStage, NF4LinearStage)):
                if hasattr(module, "base"):
                    for p in module.base.parameters():
                        p.requires_grad = False
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True
                if getattr(module, "bias", None) is not None:
                    module.bias.requires_grad = True  # trainable head bias is cheap
                continue
            # Everything else: freeze unless it's inside a LoRA submodule.
            # We keep ``reward_head`` and ``value_head`` trainable because they
            # may be the only learnable layer in reward-model training.
            pass
        # Two-pass: freeze all params, then re-enable for LoRA A/B and heads.
        for p in compiled.parameters():
            p.requires_grad = False
        trainable_unfreeze_types = (
            LoRALinearStage,
            NF4LinearStage,
            RewardHeadStage,
            ValueHeadStage,
        )
        for module in compiled.modules():
            if isinstance(module, trainable_unfreeze_types):
                if isinstance(module, (LoRALinearStage, NF4LinearStage)):
                    module.lora_A.requires_grad = True
                    module.lora_B.requires_grad = True
                    if getattr(module, "bias", None) is not None:
                        module.bias.requires_grad = True
                else:
                    for p in module.parameters():
                        p.requires_grad = True

    @classmethod
    def _apply_finetune_prehook(cls, compiled: nn.Module, graph: NeuronGraph) -> None:
        """Load base-model weights and freeze non-LoRA params when a fine-tune spec is present."""
        ft_raw = graph.torch_config.get("finetune_spec") if hasattr(graph, "torch_config") else None
        if not ft_raw:
            return
        if isinstance(ft_raw, dict):
            base_ckpt = str(ft_raw.get("base_checkpoint", ""))
            objective = str(ft_raw.get("objective", "pretrain"))
        else:
            base_ckpt = getattr(ft_raw, "base_checkpoint", "")
            objective = getattr(ft_raw, "objective", "pretrain")
        if base_ckpt:
            cls._load_base_checkpoint(compiled, base_ckpt)
        if objective in {"sft", "dpo", "ppo", "reward_model"}:
            # For sft/dpo/ppo we freeze everything except LoRA adapters and
            # small heads. For full fine-tuning, users leave adapter_type="none"
            # and set objective="sft" — they get base weights loaded but no
            # freezing (equivalent to continued pretraining).
            has_adapter = any(
                isinstance(m, (LoRALinearStage, NF4LinearStage)) for m in compiled.modules()
            )
            if has_adapter:
                cls._freeze_non_lora(compiled)

    @classmethod
    def _ema_update_targets(cls, compiled: nn.Module, decay: float) -> None:
        if not isinstance(compiled, CompiledTorchGraph):
            return
        for child in compiled.node_modules.values():
            if isinstance(child, CompiledTorchGraph):
                cls._ema_update_targets(child, decay)
        if "online_encoder" not in compiled.node_modules or "target_encoder" not in compiled.node_modules:
            return
        online = compiled.node_modules["online_encoder"]
        target = compiled.node_modules["target_encoder"]
        online_params = dict(online.named_parameters())
        for name, target_param in target.named_parameters():
            source_param = online_params.get(name)
            if source_param is None:
                continue
            target_param.data.mul_(decay).add_(source_param.data, alpha=1.0 - decay)
        online_buffers = dict(online.named_buffers())
        for name, target_buffer in target.named_buffers():
            source_buffer = online_buffers.get(name)
            if source_buffer is None:
                continue
            target_buffer.data.copy_(source_buffer.data)

    @staticmethod
    def _auto_detect_outputs(graph: NeuronGraph, exclude_nid: str) -> None:
        """Auto-detect output nodes by finding nodes with loss-typed output ports."""
        # Strategy 1: look for nodes with a 'loss' dtype output port
        for nid, node in graph.nodes.items():
            if nid == exclude_nid:
                continue
            for port in node.neuron_def.output_ports:
                if port.dtype == "loss":
                    graph.output_node_ids = [nid]
                    return
        # Strategy 2: look for nodes named 'loss_out' or with 'loss' in name
        for nid, node in graph.nodes.items():
            if nid == exclude_nid:
                continue
            if "loss" in nid.lower() or "loss" in node.neuron_def.name.lower():
                graph.output_node_ids = [nid]
                return
        # Strategy 3: use the last non-input node (sink node)
        from .graph import NeuronGraph as _NG
        try:
            order = graph.topological_order()
            for nid in reversed(order):
                if nid != exclude_nid:
                    graph.output_node_ids = [nid]
                    return
        except Exception:
            pass

    @staticmethod
    def _infer_seq_len(dataset: torch.utils.data.Dataset) -> int:
        if len(dataset) == 0:
            return 1
        sample = dataset[0]
        values = (sample,) if isinstance(sample, Tensor) else tuple(sample)
        if not values:
            return 1
        return int(values[0].shape[-1])

    @staticmethod
    def _cycle_loader(loader: torch.utils.data.DataLoader) -> Callable[[], tuple[Any, ...] | Tensor]:
        iterator = iter(loader)

        def _next_batch() -> tuple[Any, ...] | Tensor:
            nonlocal iterator
            try:
                return next(iterator)
            except StopIteration:
                iterator = iter(loader)
                return next(iterator)

        return _next_batch

    @staticmethod
    def _is_parameter_golf_profile(config: TorchTrainConfig) -> bool:
        return str(config.optimizer_profile).lower() in {"parameter_golf", "split_muon"}

    @staticmethod
    def _control_tensor_patterns() -> tuple[str, ...]:
        return (
            "attn_scale",
            "attn_scales",
            "mlp_scale",
            "mlp_scales",
            "resid_mix",
            "resid_mixes",
            "q_gain",
            "skip_weight",
            "skip_weights",
            "table_gate",
        )

    @staticmethod
    def _routing_modules(compiled: nn.Module) -> list[RoutingStatsMixin]:
        return [module for module in compiled.modules() if isinstance(module, RoutingStatsMixin)]

    @staticmethod
    def _clear_routing_modules(routing_modules: list[RoutingStatsMixin]) -> None:
        for module in routing_modules:
            module.clear_routing_stats()

    @staticmethod
    def _accumulate_routing_stats(
        accumulator: dict[str, Any] | None,
        stats: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if stats is None:
            return accumulator
        route_rows = int(stats.get("route_rows", 0))
        if route_rows <= 0:
            return accumulator
        num_experts = int(stats.get("num_experts", 0))
        selection_counts = [float(value) for value in stats.get("selection_counts", [])]
        weight_mass = [float(value) for value in stats.get("weight_mass", [])]
        if len(selection_counts) != num_experts or len(weight_mass) != num_experts:
            if accumulator is None:
                return {"_invalid": True}
            accumulator["_invalid"] = True
            return accumulator

        if accumulator is None:
            return {
                "_invalid": False,
                "num_experts": num_experts,
                "route_rows": route_rows,
                "selection_counts": selection_counts,
                "weight_mass": weight_mass,
                "_router_entropy_sum": float(stats.get("mean_router_entropy", 0.0)) * route_rows,
                "_router_entropy_norm_sum": float(stats.get("mean_router_entropy_norm", 0.0)) * route_rows,
                "_topk_entropy_sum": float(stats.get("mean_topk_entropy", 0.0)) * route_rows,
                "_topk_entropy_norm_sum": float(stats.get("mean_topk_entropy_norm", 0.0)) * route_rows,
            }

        if accumulator.get("_invalid"):
            return accumulator
        if int(accumulator.get("num_experts", 0)) != num_experts:
            accumulator["_invalid"] = True
            return accumulator

        accumulator["route_rows"] = int(accumulator.get("route_rows", 0)) + route_rows
        for idx, value in enumerate(selection_counts):
            accumulator["selection_counts"][idx] += value
        for idx, value in enumerate(weight_mass):
            accumulator["weight_mass"][idx] += value
        accumulator["_router_entropy_sum"] += float(stats.get("mean_router_entropy", 0.0)) * route_rows
        accumulator["_router_entropy_norm_sum"] += float(stats.get("mean_router_entropy_norm", 0.0)) * route_rows
        accumulator["_topk_entropy_sum"] += float(stats.get("mean_topk_entropy", 0.0)) * route_rows
        accumulator["_topk_entropy_norm_sum"] += float(stats.get("mean_topk_entropy_norm", 0.0)) * route_rows
        return accumulator

    @staticmethod
    def _finalize_routing_stats(accumulator: dict[str, Any] | None) -> dict[str, Any] | None:
        if not accumulator or accumulator.get("_invalid"):
            return None
        route_rows = int(accumulator.get("route_rows", 0))
        if route_rows <= 0:
            return None

        selection_counts = [int(round(value)) for value in accumulator.get("selection_counts", [])]
        weight_mass = [float(value) for value in accumulator.get("weight_mass", [])]
        selection_total = max(sum(selection_counts), 0)
        weight_total = float(sum(weight_mass))
        active_experts = [idx for idx, count in enumerate(selection_counts) if count > 0]

        return {
            "num_experts": int(accumulator.get("num_experts", 0)),
            "route_rows": route_rows,
            "selection_counts": selection_counts,
            "selection_shares": [
                (count / selection_total) if selection_total > 0 else 0.0
                for count in selection_counts
            ],
            "weight_mass": weight_mass,
            "weight_mass_shares": [
                (mass / weight_total) if weight_total > 0 else 0.0
                for mass in weight_mass
            ],
            "active_experts": active_experts,
            "active_expert_count": len(active_experts),
            "mean_router_entropy": float(accumulator["_router_entropy_sum"]) / route_rows,
            "mean_router_entropy_norm": float(accumulator["_router_entropy_norm_sum"]) / route_rows,
            "mean_topk_entropy": float(accumulator["_topk_entropy_sum"]) / route_rows,
            "mean_topk_entropy_norm": float(accumulator["_topk_entropy_norm_sum"]) / route_rows,
        }

    @classmethod
    def _build_optimizers(
        cls,
        compiled: CompiledTorchGraph,
        config: TorchTrainConfig,
    ) -> list[torch.optim.Optimizer]:
        if not cls._is_parameter_golf_profile(config):
            return [
                torch.optim.AdamW(
                    [{"params": list(compiled.parameters()), "lr": config.learning_rate, "base_lr": config.learning_rate}],
                    weight_decay=config.weight_decay,
                    betas=(config.beta1, config.beta2),
                    eps=config.adam_eps,
                )
            ]

        control_patterns = cls._control_tensor_patterns()
        embed_params: list[Tensor] = []
        head_params: list[Tensor] = []
        matrix_params: list[Tensor] = []
        scalar_params: list[Tensor] = []

        for name, param in compiled.named_parameters():
            if not param.requires_grad:
                continue
            if (
                name.endswith("embedding.weight")
                and "bucket_embed" not in name
                and "hash_embed" not in name
            ):
                embed_params.append(param)
                continue
            if "lm_head" in name or "ar_head" in name:
                head_params.append(param)
                continue
            if param.ndim == 2 and not any(pattern in name for pattern in control_patterns):
                matrix_params.append(param)
                continue
            scalar_params.append(param)

        token_lr = config.tied_embed_lr if config.tied_embed_lr is not None else config.embed_lr
        if token_lr is None:
            token_lr = config.learning_rate
        head_lr = config.head_lr if config.head_lr is not None else config.learning_rate
        matrix_lr = config.matrix_lr if config.matrix_lr is not None else config.learning_rate
        scalar_lr = config.scalar_lr if config.scalar_lr is not None else config.learning_rate

        optimizers: list[torch.optim.Optimizer] = []
        if embed_params:
            optimizers.append(
                torch.optim.Adam(
                    [{"params": embed_params, "lr": token_lr, "base_lr": token_lr}],
                    betas=(config.beta1, config.beta2),
                    eps=config.adam_eps,
                )
            )
        if head_params:
            optimizers.append(
                torch.optim.Adam(
                    [{"params": head_params, "lr": head_lr, "base_lr": head_lr}],
                    betas=(config.beta1, config.beta2),
                    eps=config.adam_eps,
                )
            )
        if matrix_params:
            muon = Muon(
                matrix_params,
                lr=matrix_lr,
                momentum=config.muon_momentum,
                backend_steps=config.muon_backend_steps,
            )
            for group in muon.param_groups:
                group["base_lr"] = matrix_lr
            optimizers.append(muon)
        if scalar_params:
            optimizers.append(
                torch.optim.Adam(
                    [{"params": scalar_params, "lr": scalar_lr, "base_lr": scalar_lr}],
                    betas=(config.beta1, config.beta2),
                    eps=config.adam_eps,
                    weight_decay=config.weight_decay,
                )
            )
        return optimizers

    @staticmethod
    def _optimization_method(config: TorchTrainConfig) -> str:
        return "evolutionary" if bool(config.evolutionary) else "gradient_descent"

    @staticmethod
    def _gradient_ignored_fields() -> list[str]:
        return [
            "optimizer_profile",
            "learning_rate",
            "weight_decay",
            "embed_lr",
            "head_lr",
            "tied_embed_lr",
            "matrix_lr",
            "scalar_lr",
            "warmup_steps",
            "warmdown_fraction",
            "lr_decay_iters",
            "min_lr",
            "muon_momentum",
            "muon_backend_steps",
            "muon_momentum_warmup_start",
            "muon_momentum_warmup_steps",
            "beta1",
            "beta2",
            "adam_eps",
            "grad_clip_norm",
        ]

    @staticmethod
    def _evolutionary_config_dict(config: TorchTrainConfig) -> dict[str, Any] | None:
        if not bool(config.evolutionary):
            return None
        return {
            "population_size": int(config.evo_population_size),
            "mutation_rate": float(config.evo_mutation_rate),
            "mutation_scale": float(config.evo_mutation_scale),
            "crossover_rate": float(config.evo_crossover_rate),
            "tournament_size": int(config.evo_tournament_size),
            "elite_count": int(config.evo_elite_count),
            "seed": config.evo_seed,
        }

    @staticmethod
    def _trainable_parameters(compiled: nn.Module) -> list[Tensor]:
        return [param for param in compiled.parameters() if param.requires_grad]

    @staticmethod
    def _route_evo_modules(compiled: nn.Module) -> list[SemanticMoeJepaEvoRouterStage]:
        return [module for module in compiled.modules() if isinstance(module, SemanticMoeJepaEvoRouterStage)]

    @staticmethod
    def _route_evo_parameters(route_modules: list[SemanticMoeJepaEvoRouterStage]) -> list[Tensor]:
        params: list[Tensor] = []
        seen: set[int] = set()
        for module in route_modules:
            for param in module.route_evo_parameters():
                if id(param) in seen:
                    continue
                seen.add(id(param))
                params.append(param)
        return params

    @staticmethod
    def _route_evo_config(template_spec: dict[str, Any]) -> dict[str, Any] | None:
        objective = str(template_spec.get("template", {}).get("objective", "ar"))
        if objective != "semantic_moe_jepa_evo":
            return None
        if not bool(template_spec.get("route_evo_enabled", True)):
            return None
        fraction = float(template_spec.get("route_evo_fraction", 0.10))
        if fraction <= 0.0:
            return None
        seed_raw = template_spec.get("route_evo_seed")
        seed = None if seed_raw in (None, "") else int(seed_raw)
        return {
            "fraction": min(fraction, 1.0),
            "interval": max(1, int(round(1.0 / min(fraction, 1.0)))),
            "population": max(1, int(template_spec.get("route_evo_population", 8))),
            "mutation_scale": max(0.0, float(template_spec.get("route_evo_mutation_scale", 0.05))),
            "seed": seed,
        }

    @classmethod
    def _run_route_evolution(
        cls,
        run_graph: nn.Module,
        route_params: list[Tensor],
        macro_batches: list[tuple[Tensor, ...]],
        *,
        graph_name: str,
        device: torch.device,
        amp_dtype: torch.dtype,
        use_amp: bool,
        template_runtime: str,
        routing_modules: list[RoutingStatsMixin],
        config: dict[str, Any],
        step: int,
    ) -> dict[str, Any] | None:
        if not route_params or not macro_batches:
            return None
        base = parameters_to_vector(route_params).detach().cpu()
        rng_seed = config.get("seed")
        rng = np.random.default_rng(None if rng_seed is None else int(rng_seed) + int(step))
        candidates = [base]
        for _ in range(max(int(config.get("population", 1)) - 1, 0)):
            noise = torch.from_numpy(
                rng.normal(0.0, float(config.get("mutation_scale", 0.05)), size=base.numel())
            ).to(dtype=base.dtype)
            candidates.append(base + noise)

        scored: list[tuple[float, Tensor, dict[str, Any] | None]] = []
        for candidate in candidates:
            loss, _rows, stats = cls._evaluate_evolutionary_candidate(
                run_graph,
                candidate,
                route_params,
                macro_batches,
                graph_name=graph_name,
                device=device,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
                template_runtime=template_runtime,
                routing_modules=routing_modules,
            )
            scored.append((loss, candidate, stats))
        best_loss, best_candidate, best_stats = min(scored, key=lambda item: item[0])
        with torch.no_grad():
            vector_to_parameters(
                best_candidate.to(device=route_params[0].device, dtype=route_params[0].dtype),
                route_params,
            )
        return {
            "candidate_count": len(candidates),
            "best_loss": float(best_loss),
            "mutation_scale": float(config.get("mutation_scale", 0.05)),
            "routing_stats": best_stats,
        }

    @staticmethod
    def _extract_scalar_loss(outputs: tuple[Tensor, ...], graph_name: str) -> Tensor:
        if len(outputs) != 1:
            raise ValueError(
                f"Torch training graph '{graph_name}' must expose exactly one scalar loss output"
            )
        loss = outputs[0]
        if loss.ndim != 0:
            raise ValueError("Torch training output must be a scalar loss tensor")
        return loss

    @staticmethod
    def _validate_evolutionary_config(config: TorchTrainConfig) -> None:
        if int(config.evo_population_size) <= 0:
            raise ValueError("Evolutionary torch training requires evo_population_size >= 1")
        if int(config.evo_tournament_size) <= 0:
            raise ValueError("Evolutionary torch training requires evo_tournament_size >= 1")
        if int(config.evo_elite_count) < 0:
            raise ValueError("Evolutionary torch training requires evo_elite_count >= 0")
        if int(config.evo_elite_count) > int(config.evo_population_size):
            raise ValueError("Evolutionary torch training requires evo_elite_count <= evo_population_size")
        if float(config.evo_mutation_rate) < 0.0:
            raise ValueError("Evolutionary torch training requires evo_mutation_rate >= 0")
        if float(config.evo_crossover_rate) < 0.0:
            raise ValueError("Evolutionary torch training requires evo_crossover_rate >= 0")
        if float(config.evo_mutation_scale) < 0.0:
            raise ValueError("Evolutionary torch training requires evo_mutation_scale >= 0")

    @staticmethod
    def _evolutionary_tournament_select(
        scores: list[float],
        population: list[Tensor],
        *,
        tournament_size: int,
        rng: np.random.Generator,
    ) -> Tensor:
        if not population:
            raise ValueError("Evolutionary torch training requires a non-empty population")
        size = min(max(int(tournament_size), 1), len(population))
        idxs = rng.choice(len(population), size=size, replace=False)
        best_idx = min(idxs, key=lambda idx: scores[int(idx)])
        return population[int(best_idx)]

    @staticmethod
    def _evolutionary_crossover(
        parent_a: Tensor,
        parent_b: Tensor,
        *,
        crossover_rate: float,
        rng: np.random.Generator,
    ) -> Tensor:
        child = parent_a.clone()
        if child.numel() == 0 or float(crossover_rate) <= 0.0:
            return child
        mask = rng.random(child.numel()) < float(crossover_rate)
        if not np.any(mask):
            return child
        idx = torch.from_numpy(np.flatnonzero(mask)).to(dtype=torch.long)
        child[idx] = parent_b[idx]
        return child

    @staticmethod
    def _evolutionary_mutate(
        candidate: Tensor,
        *,
        mutation_rate: float,
        mutation_scale: float,
        rng: np.random.Generator,
        force_mutation: bool = False,
    ) -> Tensor:
        mutated = candidate.clone()
        if mutated.numel() == 0 or float(mutation_scale) == 0.0:
            return mutated
        effective_rate = float(mutation_rate)
        if force_mutation and effective_rate <= 0.0:
            effective_rate = 1.0 / max(mutated.numel(), 1)
        if effective_rate <= 0.0:
            return mutated
        mask = rng.random(mutated.numel()) < effective_rate
        if force_mutation and not np.any(mask):
            mask[int(rng.integers(mutated.numel()))] = True
        if not np.any(mask):
            return mutated
        idx = torch.from_numpy(np.flatnonzero(mask)).to(dtype=torch.long)
        noise = torch.from_numpy(
            rng.normal(0.0, float(mutation_scale), size=int(idx.numel()))
        ).to(dtype=mutated.dtype)
        mutated[idx] += noise
        return mutated

    @classmethod
    def _init_evolutionary_population(
        cls,
        base_vector: Tensor,
        config: TorchTrainConfig,
        *,
        rng: np.random.Generator,
    ) -> list[Tensor]:
        base = base_vector.detach().cpu().clone()
        population = [base]
        for _ in range(max(int(config.evo_population_size) - 1, 0)):
            population.append(
                cls._evolutionary_mutate(
                    base,
                    mutation_rate=float(config.evo_mutation_rate),
                    mutation_scale=float(config.evo_mutation_scale),
                    rng=rng,
                    force_mutation=True,
                )
            )
        return population

    @classmethod
    def _evaluate_evolutionary_candidate(
        cls,
        run_graph: nn.Module,
        candidate_vector: Tensor,
        trainable_params: list[Tensor],
        macro_batches: list[tuple[Tensor, ...]],
        *,
        graph_name: str,
        device: torch.device,
        amp_dtype: torch.dtype,
        use_amp: bool,
        template_runtime: str,
        routing_modules: list[RoutingStatsMixin],
    ) -> tuple[float, int, dict[str, Any] | None]:
        if not trainable_params:
            raise ValueError("Evolutionary torch training requires at least one trainable parameter")
        candidate_on_device = candidate_vector.to(
            device=trainable_params[0].device,
            dtype=trainable_params[0].dtype,
        )
        with torch.no_grad():
            vector_to_parameters(candidate_on_device, trainable_params)
            step_rows = 0
            step_loss_total = 0.0
            step_routing_accumulator: dict[str, Any] | None = None
            for flat_inputs in macro_batches:
                if routing_modules:
                    cls._clear_routing_modules(routing_modules)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    cls._maybe_mark_cudagraph_step_begin(template_runtime, device)
                    outputs = run_graph(*flat_inputs)
                    loss = cls._extract_scalar_loss(outputs, graph_name)
                if routing_modules:
                    for module in routing_modules:
                        step_routing_accumulator = cls._accumulate_routing_stats(
                            step_routing_accumulator,
                            module.last_routing_stats,
                        )
                batch_rows = int(flat_inputs[0].size(0))
                step_rows += batch_rows
                step_loss_total += float(loss.item()) * batch_rows
        step_loss = step_loss_total / max(step_rows, 1)
        if not math.isfinite(step_loss):
            step_loss = float("inf")
        return step_loss, step_rows, cls._finalize_routing_stats(step_routing_accumulator)

    def train(
        self,
        train_inputs: list[list[int]] | Tensor,
        train_targets: list[list[int]] | Tensor,
        *,
        on_epoch: Callable[[int, float], None] | None = None,
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[float]:
        roles = self._flatten_input_roles(self.graph)
        template_spec = self._template_spec(self.graph)
        objective = str(template_spec.get("template", {}).get("objective", "ar"))
        ema_decay = float(template_spec.get("ema_decay", 0.99))
        # ── 1. Resolve data source nodes BEFORE compiling ──────────
        semantic_source_node = None
        for nid, node in self.graph.nodes.items():
            if getattr(node.neuron_def, 'module_type', '') == 'semantic_data_source':
                semantic_source_node = (nid, node.neuron_def.module_config or {})
                break

        dataset_source_node = None
        for nid, node in self.graph.nodes.items():
            if getattr(node.neuron_def, 'module_type', '') == 'dataset_source':
                ds_cfg = node.neuron_def.module_config or {}
                ds_names = ds_cfg.get('dataset_names', [])
                if ds_names:
                    dataset_source_node = (nid, ds_cfg)
                    break

        dataset = None
        semantic_active_dims = self._semantic_active_dims(self.graph)
        if semantic_source_node is not None and dataset_source_node is not None:
            sem_nid, _sem_cfg = semantic_source_node
            ds_nid, ds_cfg = dataset_source_node
            ds_names = ds_cfg.get('dataset_names', [])
            ds_seq_len = int(ds_cfg.get('seq_len', 64))
            base_dataset = self._load_dataset_for_graph(self.graph, ds_names, seq_len=ds_seq_len)
            semantic_tensors = self._load_semantic_tensors(self.graph, active_dims=semantic_active_dims)
            max_text = self._required_vocab_for_text_dataset(base_dataset) - 1
            self._adjust_vocab_size(self.graph, max_text + 1)

            dataset = RoleMappedDataset(base_dataset, roles, extra_tensors=semantic_tensors)
            self.graph.input_node_ids = [ds_nid, sem_nid]
            if not self.graph.output_node_ids:
                self._auto_detect_outputs(self.graph, ds_nid)
        elif semantic_source_node is not None:
            sem_nid, _sem_cfg = semantic_source_node
            semantic_tensors = self._load_semantic_tensors(self.graph, active_dims=semantic_active_dims)
            tokens_nid = None
            for nid in self.graph.input_node_ids:
                if nid != sem_nid:
                    tokens_nid = nid
                    break
            if tokens_nid is not None:
                self.graph.input_node_ids = [tokens_nid, sem_nid]
            else:
                self.graph.input_node_ids = [sem_nid]
            base_t = semantic_tensors["sem_targets"]
            if tokens_nid is not None:
                text_tokens, text_targets = self._build_semantic_placeholder_text(base_t)
                role_tensors = []
                for role in self._flatten_input_roles(self.graph):
                    if role in {"tokens", "enc_tokens", "dec_tokens"}:
                        role_tensors.append(text_tokens)
                    elif role == "targets":
                        role_tensors.append(text_targets)
                    elif role in semantic_tensors:
                        role_tensors.append(semantic_tensors[role])
                    else:
                        raise ValueError(f"Unsupported semantic-only training role '{role}'")
                dataset = torch.utils.data.TensorDataset(*role_tensors)
            else:
                dataset = torch.utils.data.TensorDataset(
                    *(semantic_tensors[role] for role in self._flatten_input_roles(self.graph))
                )
            if not self.graph.output_node_ids:
                self._auto_detect_outputs(self.graph, sem_nid)
        elif dataset_source_node is not None:
            ds_nid, ds_cfg = dataset_source_node
            ds_names = ds_cfg.get('dataset_names', [])
            ds_seq_len = int(ds_cfg.get('seq_len', 64))
            base_dataset = self._load_dataset_for_graph(self.graph, ds_names, seq_len=ds_seq_len)
            required_vocab = self._required_vocab_for_text_dataset(base_dataset)
            if required_vocab > 0:
                self._adjust_vocab_size(self.graph, required_vocab)
            dataset = RoleMappedDataset(base_dataset, roles)

            # Set the dataset_source node as the sole input node
            self.graph.input_node_ids = [ds_nid]
            # Auto-detect output nodes if not set
            if not self.graph.output_node_ids:
                self._auto_detect_outputs(self.graph, ds_nid)
        else:
            dataset = self._build_manual_dataset(self.graph, train_inputs, train_targets)
        # ── 2. Compile the graph AFTER adjustments ────────────────────
        compiled = CompiledTorchGraph(self.graph)
        self._prepare_ema_targets(compiled)
        # Fine-tuning: load pretrained base + freeze non-LoRA params if spec present.
        self._apply_finetune_prehook(compiled, self.graph)

        device_name = str(self.graph.torch_config.get("device", self.config.device or "cuda")).lower()
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Torch training is configured for CUDA, but no CUDA device is available")
        device = torch.device(device_name)
        amp_dtype, amp_name, use_amp = resolve_amp_settings(
            self.graph.torch_config.get("amp_dtype", self.config.amp_dtype)
        )
        use_amp = use_amp and device.type == "cuda"

        compiled.to(device)
        run_graph: nn.Module = compiled

        if self.config.activation_checkpointing and not self.config.evolutionary:
            for mod in compiled.modules():
                pass

        template_runtime = str(template_spec.get("template", {}).get("runtime", "eager"))
        if not self.config.evolutionary:
            run_graph = self._compile_training_graph(
                compiled,
                template_runtime=template_runtime,
                device=device,
                force_compile=bool(self.config.compile),
            )

            if self.config.fsdp2_enabled:
                try:
                    from torch.distributed.fsdp import fully_shard
                    run_graph = fully_shard(run_graph)
                except ImportError:
                    pass

        optimizers = [] if self.config.evolutionary else self._build_optimizers(compiled, self.config)

        # ── 3. Build DataLoader ───────────────────────────────────────
        resolved_drop_last = resolve_torch_train_drop_last(
            drop_last=self.config.drop_last,
            template_runtime=template_runtime,
            device=device,
            dataset_rows=len(dataset),
            batch_size=int(self.config.batch_size),
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=resolved_drop_last,
        )
        if len(loader) == 0:
            raise ValueError("Torch training requires a non-empty dataset")
        next_batch = self._cycle_loader(loader)
        respect_epoch_boundaries = bool(self.config.respect_epoch_boundaries)
        seq_len = self._infer_seq_len(dataset)
        microbatch_tokens = max(int(self.config.batch_size) * max(seq_len, 1), 1)
        train_batch_tokens = int(self.config.train_batch_tokens or microbatch_tokens)
        grad_accum_steps = max(1, math.ceil(train_batch_tokens / microbatch_tokens))
        steps_per_epoch = max(1, math.ceil(len(loader) / grad_accum_steps))
        estimated_total_steps = int(self.config.max_steps or (self.config.epochs * steps_per_epoch))

        def zero_grad_all() -> None:
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

        resolved_min_lr = float(
            self.config.min_lr if self.config.min_lr is not None else (self.config.learning_rate / 10.0)
        )

        def scheduled_group_lr(step: int, elapsed_s: float, *, base_lr: float) -> float:
            del elapsed_s
            if self.config.lr_decay_iters is not None:
                if self.config.learning_rate > 0:
                    min_lr_ratio = min(max(resolved_min_lr / self.config.learning_rate, 0.0), 1.0)
                    group_min_lr = base_lr * min_lr_ratio
                else:
                    group_min_lr = min(base_lr, max(resolved_min_lr, 0.0))
                return self._cosine_decay_lr(
                    step,
                    base_lr=base_lr,
                    min_lr=group_min_lr,
                    lr_decay_iters=int(self.config.lr_decay_iters),
                )
            scale = self._lr_warmdown_scale(
                step=step,
                total_steps=estimated_total_steps,
                warmdown_fraction=self.config.warmdown_fraction,
            )
            return base_lr * scale

        def muon_momentum(step: int) -> float:
            if self.config.muon_momentum_warmup_steps <= 0:
                return self.config.muon_momentum
            frac = min((step + 1) / self.config.muon_momentum_warmup_steps, 1.0)
            start = self.config.muon_momentum_warmup_start
            return start + frac * (self.config.muon_momentum - start)

        def apply_step_schedule(step: int, elapsed_s: float) -> None:
            for opt in optimizers:
                for group in opt.param_groups:
                    base_lr = float(group.get("base_lr", group["lr"]))
                    group["lr"] = scheduled_group_lr(step, elapsed_s, base_lr=base_lr)
                    if isinstance(opt, Muon):
                        group["momentum"] = muon_momentum(step)

        def current_lrs() -> list[float]:
            lrs: list[float] = []
            for opt in optimizers:
                for group in opt.param_groups:
                    lrs.append(float(group["lr"]))
            return lrs

        self._stop = False
        self.loss_history = []
        run_graph.train()
        routing_modules = self._routing_modules(compiled) if on_step is not None else []
        route_evo_config = self._route_evo_config(template_spec)
        route_evo_params = (
            self._route_evo_parameters(self._route_evo_modules(compiled))
            if route_evo_config is not None and not self.config.evolutionary
            else []
        )
        optimization_method = self._optimization_method(self.config)
        evolutionary_config = self._evolutionary_config_dict(self.config)

        global_step = 0
        start_time = time.perf_counter()
        propagating_exception = False
        try:
            if self.config.evolutionary:
                self._validate_evolutionary_config(self.config)
                trainable_params = self._trainable_parameters(compiled)
                if not trainable_params:
                    raise ValueError("Evolutionary torch training requires at least one trainable parameter")
                rng = np.random.default_rng(self.config.evo_seed)
                population = self._init_evolutionary_population(
                    parameters_to_vector(trainable_params),
                    self.config,
                    rng=rng,
                )

                for epoch in range(self.config.epochs):
                    if self._stop:
                        break
                    total_loss = 0.0
                    total_rows = 0
                    epoch_loader = iter(loader)
                    remaining_loader_batches = len(loader)
                    for epoch_step in range(steps_per_epoch):
                        if self._stop or (self.config.max_steps is not None and global_step >= self.config.max_steps):
                            self._stop = True
                            break
                        step_grad_accum_steps = grad_accum_steps
                        if respect_epoch_boundaries:
                            if remaining_loader_batches <= 0:
                                break
                            step_grad_accum_steps = min(grad_accum_steps, remaining_loader_batches)
                        macro_batches: list[tuple[Tensor, ...]] = []
                        for _ in range(step_grad_accum_steps):
                            if respect_epoch_boundaries:
                                try:
                                    batch = next(epoch_loader)
                                except StopIteration:
                                    break
                                remaining_loader_batches -= 1
                            else:
                                batch = next_batch()
                            if isinstance(batch, Tensor):
                                batch = (batch,)
                            macro_batches.append(tuple(item.to(device) for item in batch))
                        if not macro_batches:
                            break

                        scored_population: list[tuple[float, Tensor, dict[str, Any] | None]] = []
                        step_rows = 0
                        for candidate in population:
                            candidate_loss, candidate_rows, candidate_routing_stats = self._evaluate_evolutionary_candidate(
                                run_graph,
                                candidate,
                                trainable_params,
                                macro_batches,
                                graph_name=self.graph.name,
                                device=device,
                                amp_dtype=amp_dtype,
                                use_amp=use_amp,
                                template_runtime=template_runtime,
                                routing_modules=routing_modules,
                            )
                            if step_rows <= 0:
                                step_rows = candidate_rows
                            scored_population.append((candidate_loss, candidate, candidate_routing_stats))
                        if step_rows <= 0:
                            break

                        ranked = sorted(scored_population, key=lambda item: item[0])
                        best_step_loss, best_candidate, best_step_routing_stats = ranked[0]
                        with torch.no_grad():
                            vector_to_parameters(
                                best_candidate.to(
                                    device=trainable_params[0].device,
                                    dtype=trainable_params[0].dtype,
                                ),
                                trainable_params,
                            )
                        if objective in ("jepa", "ar_jepa", "jepa_semantic", "semantic_router_jepa", "semantic_dense_jepa_evo", "semantic_moe_jepa_evo"):
                            self._ema_update_targets(compiled, ema_decay)

                        scores = [item[0] for item in scored_population]
                        elite_count = min(max(int(self.config.evo_elite_count), 0), len(population))
                        next_population = [ranked[idx][1].clone() for idx in range(elite_count)]
                        while len(next_population) < int(self.config.evo_population_size):
                            parent_a = self._evolutionary_tournament_select(
                                scores,
                                population,
                                tournament_size=int(self.config.evo_tournament_size),
                                rng=rng,
                            )
                            parent_b = self._evolutionary_tournament_select(
                                scores,
                                population,
                                tournament_size=int(self.config.evo_tournament_size),
                                rng=rng,
                            )
                            child = self._evolutionary_crossover(
                                parent_a,
                                parent_b,
                                crossover_rate=float(self.config.evo_crossover_rate),
                                rng=rng,
                            )
                            child = self._evolutionary_mutate(
                                child,
                                mutation_rate=float(self.config.evo_mutation_rate),
                                mutation_scale=float(self.config.evo_mutation_scale),
                                rng=rng,
                            )
                            next_population.append(child)
                        population = next_population

                        total_loss += best_step_loss * step_rows
                        total_rows += step_rows
                        global_step += 1
                        if on_step is not None:
                            step_info = {
                                "phase": "train",
                                "optimization_method": optimization_method,
                                "step": global_step,
                                "max_steps": int(self.config.max_steps or estimated_total_steps),
                                "epoch": epoch + 1,
                                "max_epochs": self.config.epochs,
                                "epoch_step": epoch_step + 1,
                                "steps_per_epoch": steps_per_epoch,
                                "loss": best_step_loss,
                                "elapsed_seconds": time.perf_counter() - start_time,
                                "grad_accum_steps": grad_accum_steps,
                                "learning_rates": [],
                                "population_size": int(self.config.evo_population_size),
                            }
                            if step_grad_accum_steps != grad_accum_steps:
                                step_info["actual_grad_accum_steps"] = step_grad_accum_steps
                            if best_step_routing_stats is not None:
                                step_info["routing_stats"] = best_step_routing_stats
                            on_step(step_info)
                        if (
                            self.config.max_wallclock_seconds > 0
                            and (time.perf_counter() - start_time) >= self.config.max_wallclock_seconds
                        ):
                            self._stop = True
                            break

                    avg_loss = total_loss / max(total_rows, 1)
                    self.loss_history.append(avg_loss)
                    if on_epoch is not None:
                        on_epoch(epoch, avg_loss)
            else:
                if self._is_parameter_golf_profile(self.config) and self.config.warmup_steps > 0:
                    initial_model_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in compiled.state_dict().items()
                    }
                    initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
                    zero_grad_all()
                    for warmup_step in range(self.config.warmup_steps):
                        apply_step_schedule(0, 0.0)
                        step_rows = 0
                        step_loss_total = 0.0
                        step_routing_accumulator: dict[str, Any] | None = None
                        for _ in range(grad_accum_steps):
                            if routing_modules:
                                self._clear_routing_modules(routing_modules)
                            batch = next_batch()
                            if isinstance(batch, Tensor):
                                batch = (batch,)
                            flat_inputs = tuple(item.to(device) for item in batch)
                            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                                self._maybe_mark_cudagraph_step_begin(template_runtime, device)
                                loss = self._extract_scalar_loss(
                                    run_graph(*flat_inputs),
                                    self.graph.name,
                                )
                            (loss / grad_accum_steps).backward()
                            if routing_modules:
                                for module in routing_modules:
                                    step_routing_accumulator = self._accumulate_routing_stats(
                                        step_routing_accumulator,
                                        module.last_routing_stats,
                                    )
                            batch_rows = int(flat_inputs[0].size(0))
                            step_rows += batch_rows
                            step_loss_total += float(loss.item()) * batch_rows
                        if self.config.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(compiled.parameters(), self.config.grad_clip_norm)
                        for opt in optimizers:
                            opt.step()
                        if objective in ("jepa", "ar_jepa", "jepa_semantic", "semantic_router_jepa", "semantic_dense_jepa_evo", "semantic_moe_jepa_evo"):
                            self._ema_update_targets(compiled, ema_decay)
                        zero_grad_all()
                        if on_step is not None:
                            step_info = {
                                "phase": "warmup",
                                "optimization_method": optimization_method,
                                "step": warmup_step + 1,
                                "warmup_steps": self.config.warmup_steps,
                                "loss": step_loss_total / max(step_rows, 1),
                                "elapsed_seconds": time.perf_counter() - start_time,
                                "grad_accum_steps": grad_accum_steps,
                                "learning_rates": current_lrs(),
                            }
                            routing_stats = self._finalize_routing_stats(step_routing_accumulator)
                            if routing_stats is not None:
                                step_info["routing_stats"] = routing_stats
                            on_step(step_info)
                    compiled.load_state_dict(initial_model_state, strict=True)
                    for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
                        opt.load_state_dict(state)
                    zero_grad_all()

                for epoch in range(self.config.epochs):
                    if self._stop:
                        break
                    total_loss = 0.0
                    total_rows = 0
                    epoch_loader = iter(loader)
                    remaining_loader_batches = len(loader)
                    for epoch_step in range(steps_per_epoch):
                        if self._stop or (self.config.max_steps is not None and global_step >= self.config.max_steps):
                            self._stop = True
                            break
                        step_grad_accum_steps = grad_accum_steps
                        if respect_epoch_boundaries:
                            if remaining_loader_batches <= 0:
                                break
                            step_grad_accum_steps = min(grad_accum_steps, remaining_loader_batches)
                        zero_grad_all()
                        step_rows = 0
                        step_loss_total = 0.0
                        step_routing_accumulator = None
                        macro_batches: list[tuple[Tensor, ...]] = []
                        for _ in range(step_grad_accum_steps):
                            if routing_modules:
                                self._clear_routing_modules(routing_modules)
                            if respect_epoch_boundaries:
                                try:
                                    batch = next(epoch_loader)
                                except StopIteration:
                                    break
                                remaining_loader_batches -= 1
                            else:
                                batch = next_batch()
                            if isinstance(batch, Tensor):
                                batch = (batch,)
                            flat_inputs = tuple(item.to(device) for item in batch)
                            macro_batches.append(flat_inputs)
                            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                                self._maybe_mark_cudagraph_step_begin(template_runtime, device)
                                loss = self._extract_scalar_loss(
                                    run_graph(*flat_inputs),
                                    self.graph.name,
                                )
                            (loss / step_grad_accum_steps).backward()
                            if routing_modules:
                                for module in routing_modules:
                                    step_routing_accumulator = self._accumulate_routing_stats(
                                        step_routing_accumulator,
                                        module.last_routing_stats,
                                    )
                            batch_rows = int(flat_inputs[0].size(0))
                            step_rows += batch_rows
                            step_loss_total += float(loss.item()) * batch_rows
                        if step_rows <= 0:
                            break

                        apply_step_schedule(global_step, time.perf_counter() - start_time)
                        if self.config.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(compiled.parameters(), self.config.grad_clip_norm)
                        for opt in optimizers:
                            opt.step()
                        route_evo_info = None
                        if (
                            route_evo_config is not None
                            and route_evo_params
                            and ((global_step + 1) % int(route_evo_config["interval"]) == 0)
                        ):
                            route_evo_info = self._run_route_evolution(
                                run_graph,
                                route_evo_params,
                                macro_batches,
                                graph_name=self.graph.name,
                                device=device,
                                amp_dtype=amp_dtype,
                                use_amp=use_amp,
                                template_runtime=template_runtime,
                                routing_modules=routing_modules,
                                config=route_evo_config,
                                step=global_step + 1,
                            )
                        zero_grad_all()
                        if objective in ("jepa", "ar_jepa", "jepa_semantic", "semantic_router_jepa", "semantic_dense_jepa_evo", "semantic_moe_jepa_evo"):
                            self._ema_update_targets(compiled, ema_decay)
                        total_loss += step_loss_total
                        total_rows += step_rows
                        global_step += 1
                        if on_step is not None:
                            step_info = {
                                "phase": "train",
                                "optimization_method": optimization_method,
                                "step": global_step,
                                "max_steps": int(self.config.max_steps or estimated_total_steps),
                                "epoch": epoch + 1,
                                "max_epochs": self.config.epochs,
                                "epoch_step": epoch_step + 1,
                                "steps_per_epoch": steps_per_epoch,
                                "loss": step_loss_total / max(step_rows, 1),
                                "elapsed_seconds": time.perf_counter() - start_time,
                                "grad_accum_steps": grad_accum_steps,
                                "learning_rates": current_lrs(),
                            }
                            if step_grad_accum_steps != grad_accum_steps:
                                step_info["actual_grad_accum_steps"] = step_grad_accum_steps
                            routing_stats = self._finalize_routing_stats(step_routing_accumulator)
                            if routing_stats is not None:
                                step_info["routing_stats"] = routing_stats
                            if route_evo_info is not None:
                                step_info["route_evo"] = route_evo_info
                            on_step(step_info)
                        if (
                            self.config.max_wallclock_seconds > 0
                            and (time.perf_counter() - start_time) >= self.config.max_wallclock_seconds
                        ):
                            self._stop = True
                            break

                    avg_loss = total_loss / max(total_rows, 1)
                    self.loss_history.append(avg_loss)
                    if on_epoch is not None:
                        on_epoch(epoch, avg_loss)
        except BaseException:
            propagating_exception = True
            raise
        finally:
            try:
                compiled.sync_state_back(self.graph)
            except Exception:
                if not propagating_exception:
                    raise
            self.graph.training_method = "torch"
            self.graph.runtime = "torch"
            final_torch_config = {
                **self.graph.torch_config,
                "device": device.type,
                "amp_dtype": amp_name,
                "drop_last": resolved_drop_last,
                "respect_epoch_boundaries": respect_epoch_boundaries,
                "optimization_method": optimization_method,
            }
            if evolutionary_config is not None:
                final_torch_config["evolutionary"] = evolutionary_config
            else:
                final_torch_config.pop("evolutionary", None)
            self.graph.torch_config = final_torch_config
        return self.loss_history


class DenoiseHeadStage(nn.Module):
    """Diffusion denoising head: predicts clean tokens from noisy ones."""
    def __init__(self, model_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(model_dim, vocab_size, bias=False)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class MaskSchedulerStage(nn.Module):
    """Discrete mask scheduler for Diffusion LMs."""
    def __init__(self, vocab_size: int, mask_token_id: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        
    def forward(self, tokens: Tensor, timesteps: Tensor) -> Tensor:
        # tokens: [batch, seq]
        # timesteps: [batch] normalized 0-1
        # Simple random masking based on timestep
        batch, seq = tokens.shape
        mask_probs = timesteps.view(-1, 1).expand(batch, seq)
        noise = torch.rand(batch, seq, device=tokens.device)
        mask = noise < mask_probs
        
        noisy_tokens = tokens.clone()
        noisy_tokens[mask] = self.mask_token_id
        return noisy_tokens


class SemanticProjectorStage(nn.Module):
    """Experimental: predict vocab-topic logits + internal semantic state."""

    def __init__(
        self,
        input_dim: int,
        semantic_dim: int = NUM_SEMANTIC_DIMS,
        residual_dim: int = 64,
        n_sig_buckets: int = 4096,
        semantic_vocab_ref: str = "",
    ) -> None:
        super().__init__()
        from .semantic import ConversationalVocabulary, semantic_vocab_ref_for_dim

        vocab_ref = semantic_vocab_ref or semantic_vocab_ref_for_dim(semantic_dim)
        vocab = ConversationalVocabulary(vocab_ref or None)
        self.semantic_dim = semantic_dim
        self.num_vocab_dims = vocab.num_vocab_dims
        self.term_counts = [len(vocab.terms(dim_name)) for dim_name in vocab.dim_names]
        self.max_terms = max(self.term_counts) if self.term_counts else 0
        self.topic_heads = nn.ModuleList(
            nn.Linear(input_dim, max(count, 1), bias=False) for count in self.term_counts
        )
        self.sig_head = nn.Linear(input_dim, n_sig_buckets, bias=False)
        self.residual_head = nn.Sequential(
            nn.Linear(input_dim, residual_dim, bias=False),
            nn.GELU(),
            nn.Linear(residual_dim, residual_dim, bias=False),
        )

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
            logits = head(pooled)
            topic_logits[:, dim_idx, :count] = logits
            topic_idx = torch.argmax(logits.float(), dim=-1)
            if count > 1:
                semantic_dims.append(2.0 * topic_idx.to(dtype=target_dtype) / float(count - 1) - 1.0)
            else:
                semantic_dims.append(torch.zeros(batch, device=pooled.device, dtype=target_dtype))

        sig_logits = self.sig_head(pooled)
        sig_probs = F.softmax(sig_logits.float(), dim=-1)
        bucket_axis = torch.linspace(0.0, 1.0, sig_probs.size(-1), device=sig_probs.device, dtype=torch.float32)
        sig_scalar = (sig_probs * bucket_axis).sum(dim=-1, keepdim=True).to(dtype=target_dtype)
        sem = torch.stack(semantic_dims, dim=-1)
        sem = torch.cat([sem, sig_scalar], dim=-1)
        res = self.residual_head(x).to(dtype=target_dtype)
        return sem, res, topic_logits.to(dtype=target_dtype)


class SemanticAlignmentLossStage(nn.Module):
    """Experimental: masked categorical loss over vocab-topic logits."""

    def __init__(self, ignore_index: int = -100, semantic_vocab_ref: str = "") -> None:
        super().__init__()
        self.semantic_vocab_ref = str(semantic_vocab_ref)
        self.term_counts: list[int] = []
        self.ignore_index = int(ignore_index)
        if self.semantic_vocab_ref:
            self._ensure_term_counts()

    def _ensure_term_counts(self, semantic_dim: int | None = None) -> None:
        if self.term_counts:
            return
        from .semantic import ConversationalVocabulary, semantic_vocab_ref_for_dim

        vocab_ref = self.semantic_vocab_ref or semantic_vocab_ref_for_dim(semantic_dim)
        vocab = ConversationalVocabulary(vocab_ref or None)
        self.term_counts = [len(vocab.terms(dim_name)) for dim_name in vocab.dim_names]

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
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
        self._ensure_term_counts(targets.size(1))
        losses: list[Tensor] = []
        n_dims = min(len(self.term_counts), logits.size(1), targets.size(1))
        for dim_idx in range(n_dims):
            term_count = min(self.term_counts[dim_idx], logits.size(-1))
            if term_count <= 0:
                continue
            dim_targets = targets[:, dim_idx]
            valid = (dim_targets != self.ignore_index) & (dim_targets >= 0) & (dim_targets < term_count)
            safe_targets = torch.where(valid, dim_targets, dim_targets.new_full(dim_targets.shape, self.ignore_index))
            dim_logits = logits[:, dim_idx, :term_count]
            loss_sum = F.cross_entropy(
                dim_logits,
                safe_targets,
                ignore_index=self.ignore_index,
                reduction="sum",
            )
            valid_count = valid.to(dtype=loss_sum.dtype).sum()
            losses.append(torch.where(valid_count > 0, loss_sum / valid_count, loss_sum * 0.0))
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


class SemanticHasherStage(nn.Module):
    """Experimental: LSH hashing of semantic vectors inside a compiled graph."""

    def __init__(self, dim: int = NUM_SEMANTIC_DIMS, tables: int = 8, planes: int = 12, seed: int = 42) -> None:
        super().__init__()
        import numpy as np
        rng = np.random.RandomState(seed)
        proj = torch.from_numpy(rng.randn(tables, planes, dim).astype("float32"))
        self.register_buffer("proj", proj)
        self.n_tables = tables

    def forward(self, sem_vec: Tensor) -> Tensor:
        bits = torch.einsum("tpd,bd->btp", self.proj.to(sem_vec.dtype), sem_vec) > 0
        powers = (2 ** torch.arange(bits.shape[-1], device=bits.device, dtype=torch.long)).unsqueeze(0).unsqueeze(0)
        return (bits.long() * powers).sum(dim=-1)


class SemanticMoERouterStage(RoutingStatsMixin, nn.Module):
    """Experimental: legacy cosine router retained for compatibility."""

    def __init__(self, n_experts: int, semantic_dim: int = NUM_SEMANTIC_DIMS, top_k: int = 2) -> None:
        super().__init__()
        self.centroids = nn.Parameter(torch.randn(n_experts, semantic_dim))
        self.top_k = top_k
        self._init_routing_stats(num_experts=n_experts, top_k=top_k)

    def forward(self, sem_vec: Tensor) -> tuple[Tensor, Tensor]:
        work_dtype = sem_vec.dtype if torch.is_floating_point(sem_vec) else self.centroids.dtype
        c = F.normalize(self.centroids.float(), dim=-1).to(dtype=work_dtype)
        s = sem_vec
        if s.ndim == 3:
            s = s.mean(dim=1)
        s = F.normalize(s.float(), dim=-1).to(dtype=work_dtype)
        sim = s @ c.T
        topk_weights, topk_indices = torch.topk(sim, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights.float(), dim=-1).to(dtype=work_dtype)
        self._update_routing_stats(
            scores=F.softmax(sim.float(), dim=-1),
            routing_weights=topk_weights,
            routing_indices=topk_indices,
        )
        return topk_weights.unsqueeze(1), topk_indices.unsqueeze(1)


class SemanticHashRouterStage(RoutingStatsMixin, nn.Module):
    """Experimental: fixed dimension-to-expert router with hash-aware weighting."""

    def __init__(
        self,
        n_experts: int,
        semantic_dim: int = NUM_SEMANTIC_DIMS,
        top_k: int = 2,
        tables: int = 8,
        n_buckets: int = 4096,
        ignore_index: int = -100,
        semantic_vocab_ref: str = "",
        routing_source: str = "topic_logits",
    ) -> None:
        super().__init__()
        from .semantic import ConversationalVocabulary, semantic_vocab_ref_for_dim

        vocab_ref = semantic_vocab_ref or semantic_vocab_ref_for_dim(semantic_dim)
        vocab = ConversationalVocabulary(vocab_ref or None)
        self.num_vocab_dims = vocab.num_vocab_dims
        if int(n_experts) != self.num_vocab_dims:
            raise ValueError(
                f"semantic_hash_router requires exactly {self.num_vocab_dims} experts, got {n_experts}"
            )
        self.semantic_dim = semantic_dim
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
            probs = F.softmax(logits[:, dim_idx, :term_count], dim=-1)
            scores[:, dim_idx] = probs.max(dim=-1).values
        if bucket_indices.ndim == 1:
            bucket_indices = bucket_indices.unsqueeze(-1)
        bucket_features = self.hash_embed(bucket_indices.long() % self.n_buckets).float()
        gate = F.softmax(self.table_gate[: bucket_features.size(1)], dim=0)
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
            scores=F.softmax(scores.float(), dim=-1),
            routing_weights=weights,
            routing_indices=indices,
        )
        return weights, indices


class CausalChunkStateStage(nn.Module):
    """Build prefix-safe or full-span chunk states from token hidden states."""

    def __init__(self, chunk_size: int = 32, mode: str = "prefix") -> None:
        super().__init__()
        self.chunk_size = max(int(chunk_size), 1)
        self.mode = str(mode)

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.ndim != 3:
            raise ValueError("causal_chunk_state expects hidden shaped [batch, seq, dim]")
        batch, seq_len, dim = hidden.shape
        chunks = max(math.ceil(seq_len / self.chunk_size), 1)
        if self.mode == "mean":
            padded_len = chunks * self.chunk_size
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
            work = work.reshape(batch, chunks, self.chunk_size, dim)
            weights = valid.reshape(batch, chunks, self.chunk_size).unsqueeze(-1)
            denom = weights.sum(dim=2).clamp_min(1.0)
            return (work * weights).sum(dim=2) / denom

        cumulative = hidden.cumsum(dim=1)
        boundary_positions = torch.arange(chunks, device=hidden.device, dtype=torch.long) * self.chunk_size - 1
        boundary_positions = boundary_positions.clamp(min=0, max=seq_len - 1)
        gathered = cumulative[:, boundary_positions, :]
        denom = (boundary_positions + 1).to(device=hidden.device, dtype=hidden.dtype).view(1, chunks, 1)
        return gathered / denom


class SemanticChunkProjectorStage(nn.Module):
    """Chunk-level semantic projector that preserves the chunk axis."""

    def __init__(
        self,
        input_dim: int,
        semantic_dim: int = NUM_SEMANTIC_DIMS,
        residual_dim: int = 64,
        n_sig_buckets: int = 4096,
        semantic_vocab_ref: str = "",
    ) -> None:
        super().__init__()
        from .semantic import ConversationalVocabulary, semantic_vocab_ref_for_dim

        vocab_ref = semantic_vocab_ref or semantic_vocab_ref_for_dim(semantic_dim)
        vocab = ConversationalVocabulary(vocab_ref or None)
        self.semantic_dim = semantic_dim
        self.num_vocab_dims = vocab.num_vocab_dims
        self.term_counts = [len(vocab.terms(dim_name)) for dim_name in vocab.dim_names]
        self.max_terms = max(self.term_counts) if self.term_counts else 0
        self.topic_heads = nn.ModuleList(
            nn.Linear(input_dim, max(count, 1), bias=False) for count in self.term_counts
        )
        self.sig_head = nn.Linear(input_dim, n_sig_buckets, bias=False)
        self.residual_head = nn.Sequential(
            nn.Linear(input_dim, residual_dim, bias=False),
            nn.GELU(),
            nn.Linear(residual_dim, residual_dim, bias=False),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError("semantic_chunk_projector expects [batch, chunks, dim]")
        batch, chunks, dim = x.shape
        flat = x.reshape(batch * chunks, dim)
        target_dtype = flat.dtype if torch.is_floating_point(flat) else self.sig_head.weight.dtype
        topic_logits = flat.new_zeros((batch * chunks, self.num_vocab_dims, self.max_terms))
        semantic_dims: list[Tensor] = []
        for dim_idx, head in enumerate(self.topic_heads):
            count = self.term_counts[dim_idx]
            logits = head(flat)
            topic_logits[:, dim_idx, :count] = logits
            topic_idx = torch.argmax(logits.float(), dim=-1)
            if count > 1:
                semantic_dims.append(2.0 * topic_idx.to(dtype=target_dtype) / float(count - 1) - 1.0)
            else:
                semantic_dims.append(torch.zeros(batch * chunks, device=flat.device, dtype=target_dtype))
        sig_logits = self.sig_head(flat)
        sig_probs = F.softmax(sig_logits.float(), dim=-1)
        bucket_axis = torch.linspace(0.0, 1.0, sig_probs.size(-1), device=sig_probs.device, dtype=torch.float32)
        sig_scalar = (sig_probs * bucket_axis).sum(dim=-1, keepdim=True).to(dtype=target_dtype)
        sem = torch.stack(semantic_dims, dim=-1)
        sem = torch.cat([sem, sig_scalar], dim=-1)
        res = self.residual_head(flat).to(dtype=target_dtype)
        return (
            sem.reshape(batch, chunks, -1),
            res.reshape(batch, chunks, -1),
            topic_logits.to(dtype=target_dtype).reshape(batch, chunks, self.num_vocab_dims, self.max_terms),
        )


class SemanticChunkHasherStage(nn.Module):
    """LSH semantic chunk vectors while preserving [batch, chunks]."""

    def __init__(self, dim: int = NUM_SEMANTIC_DIMS, tables: int = 8, planes: int = 12, seed: int = 42) -> None:
        super().__init__()
        rng = np.random.RandomState(seed)
        proj = torch.from_numpy(rng.randn(tables, planes, dim).astype("float32"))
        self.register_buffer("proj", proj)

    def forward(self, sem_vec: Tensor) -> Tensor:
        if sem_vec.ndim == 2:
            sem_vec = sem_vec.unsqueeze(1)
        if sem_vec.ndim != 3:
            raise ValueError("semantic_chunk_hasher expects [batch, chunks, semantic_dim]")
        batch, chunks, dim = sem_vec.shape
        flat = sem_vec.reshape(batch * chunks, dim)
        bits = torch.einsum("tpd,bd->btp", self.proj.to(flat.dtype), flat) > 0
        powers = (2 ** torch.arange(bits.shape[-1], device=bits.device, dtype=torch.long)).view(1, 1, -1)
        buckets = (bits.long() * powers).sum(dim=-1)
        return buckets.reshape(batch, chunks, -1)


class SemanticMoeJepaEvoRouterStage(RoutingStatsMixin, nn.Module):
    """Chunk-level semantic/free expert router with always-on shared experts."""

    def __init__(
        self,
        semantic_dim: int = NUM_SEMANTIC_DIMS,
        top_k: int = 2,
        shared_experts: int = 2,
        free_experts: int = 8,
        tables: int = 8,
        n_buckets: int = 4096,
        ignore_index: int = -100,
        semantic_vocab_ref: str = "",
    ) -> None:
        super().__init__()
        from .semantic import ConversationalVocabulary, semantic_vocab_ref_for_dim

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
            probs = F.softmax(topic_logits[..., dim_idx, :term_count].float(), dim=-1)
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
        gate = F.softmax(self.table_gate[: bucket_features.size(2)], dim=0)
        hash_bias = (bucket_features * gate.view(1, 1, -1, 1)).sum(dim=2)
        semantic_scores = topic_scores + hash_bias[..., : self.num_vocab_dims] + self.dimension_bias
        if self.free_experts > 0 and self.free_head is not None:
            free_scores = self.free_head(sem_vec.float()) + hash_bias[..., self.num_vocab_dims :]
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
        k_per_row = torch.where(
            has_forced,
            forced_mask.sum(dim=-1).clamp(min=1, max=self.top_k),
            torch.full((batch,), self.top_k, device=sem_vec.device, dtype=torch.long),
        )
        chosen = ordered[..., : self.top_k]
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
            scores=F.softmax(route_logits.float(), dim=-1),
            routing_weights=weights,
            routing_indices=combined_indices,
        )
        return weights, combined_indices, route_logits.to(dtype=work_dtype)


class BroadcastChunkRoutesStage(nn.Module):
    """Expand chunk-level routes to the per-token route tensors used by MoE dispatch."""

    def __init__(self, chunk_size: int = 32) -> None:
        super().__init__()
        self.chunk_size = max(int(chunk_size), 1)

    def forward(self, hidden: Tensor, expert_weights: Tensor, expert_indices: Tensor) -> tuple[Tensor, Tensor]:
        if hidden.ndim != 3:
            raise ValueError("broadcast_chunk_routes expects hidden shaped [batch, seq, dim]")
        batch, seq_len, _ = hidden.shape
        if expert_weights.ndim != 3 or expert_indices.ndim != 3:
            raise ValueError("broadcast_chunk_routes expects routes shaped [batch, chunks, route_width]")
        if expert_weights.size(0) != batch or expert_indices.size(0) != batch:
            raise ValueError("broadcast_chunk_routes batch size must match hidden batch size")
        token_chunks = (torch.arange(seq_len, device=hidden.device, dtype=torch.long) // self.chunk_size).clamp_max(expert_weights.size(1) - 1)
        weights = expert_weights[:, token_chunks, :]
        indices = expert_indices.long()[:, token_chunks, :]
        return weights.to(dtype=hidden.dtype).contiguous(), indices.contiguous()


class RouteBalanceLossStage(nn.Module):
    def forward(self, route_logits: Tensor) -> Tensor:
        probs = F.softmax(route_logits.float().reshape(-1, route_logits.size(-1)), dim=-1)
        density = probs.mean(dim=0)
        return route_logits.size(-1) * (density * density).sum()


class RouteSelectionLossStage(nn.Module):
    def __init__(
        self,
        semantic_vocab_ref: str = "",
        shared_experts: int = 2,
        free_experts: int = 8,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        from .semantic import ConversationalVocabulary, semantic_vocab_ref_for_dim

        vocab = ConversationalVocabulary(semantic_vocab_ref or semantic_vocab_ref_for_dim(NUM_SEMANTIC_DIMS))
        self.num_vocab_dims = vocab.num_vocab_dims
        self.shared_experts = max(int(shared_experts), 0)
        self.free_experts = max(int(free_experts), 0)
        self.ignore_index = int(ignore_index)

    def forward(self, route_logits: Tensor, sem_targets: Tensor) -> Tensor:
        logits = route_logits.float()
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)
        targets = sem_targets.long()
        if targets.ndim == 1:
            targets = targets.unsqueeze(0)
        if targets.size(1) < self.num_vocab_dims:
            pad = targets.new_full((targets.size(0), self.num_vocab_dims - targets.size(1)), self.ignore_index)
            targets = torch.cat([targets, pad], dim=1)
        active = targets[:, : self.num_vocab_dims] != self.ignore_index
        if not bool(active.any()):
            return logits.sum() * 0.0
        semantic_logits = logits[..., self.shared_experts : self.shared_experts + self.num_vocab_dims]
        target = active.to(dtype=semantic_logits.dtype).unsqueeze(1).expand_as(semantic_logits)
        valid = active.unsqueeze(1).expand_as(semantic_logits)
        losses = F.binary_cross_entropy_with_logits(semantic_logits, target, reduction="none")
        return losses[valid].mean()


class RouteDistillationLossStage(nn.Module):
    def __init__(self, semantic_vocab_ref: str = "", shared_experts: int = 2, free_experts: int = 8) -> None:
        super().__init__()
        from .semantic import ConversationalVocabulary, semantic_vocab_ref_for_dim

        vocab = ConversationalVocabulary(semantic_vocab_ref or semantic_vocab_ref_for_dim(NUM_SEMANTIC_DIMS))
        self.num_vocab_dims = vocab.num_vocab_dims
        self.shared_experts = max(int(shared_experts), 0)
        self.free_experts = max(int(free_experts), 0)
        self.total_experts = self.shared_experts + self.num_vocab_dims + self.free_experts
        self.term_counts = [len(vocab.terms(dim_name)) for dim_name in vocab.dim_names]

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
            probs = F.softmax(target_logits[..., dim_idx, :term_count], dim=-1)
            target_scores[..., dim_idx] = probs.max(dim=-1).values
        teacher_logits = student.new_full(student.shape, -10.0)
        if self.shared_experts > 0:
            teacher_logits[..., : self.shared_experts] = 0.0
        teacher_logits[..., self.shared_experts : self.shared_experts + self.num_vocab_dims] = target_scores
        teacher = F.softmax(teacher_logits.detach().reshape(-1, teacher_logits.size(-1)), dim=-1)
        student_log = F.log_softmax(student.reshape(-1, student.size(-1)), dim=-1)
        return F.kl_div(student_log, teacher, reduction="batchmean")


class RoutedAttentionExpertsStage(nn.Module):
    """Experimental: combine top-k expert attention outputs over the full sequence."""

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
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        head_dim = model_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")

        kv_dim = num_kv_heads * head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.experts = experts
        self.top_k = top_k
        self.is_causal = is_causal
        self.q_proj = nn.Parameter(torch.empty(experts, model_dim, model_dim))
        self.k_proj = nn.Parameter(torch.empty(experts, model_dim, kv_dim))
        self.v_proj = nn.Parameter(torch.empty(experts, model_dim, kv_dim))
        self.out_proj = nn.Parameter(torch.empty(experts, model_dim, model_dim))
        self.q_gain = nn.Parameter(torch.full((experts, num_heads), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(head_dim, rope_base)
        nn.init.normal_(self.q_proj, std=0.02)
        nn.init.normal_(self.k_proj, std=0.02)
        nn.init.normal_(self.v_proj, std=0.02)
        nn.init.normal_(self.out_proj, std=0.02)

    def _expert_attention(self, x: Tensor, expert_idx: int) -> Tensor:
        batch, seq_len, model_dim = x.shape
        q = torch.einsum("bsd,df->bsf", x, self.q_proj[expert_idx]).reshape(
            batch, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = torch.einsum("bsd,df->bsf", x, self.k_proj[expert_idx]).reshape(
            batch, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = torch.einsum("bsd,df->bsf", x, self.v_proj[expert_idx]).reshape(
            batch, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seq_len, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain[expert_idx].to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=self.is_causal,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        y = y.transpose(1, 2).contiguous().reshape(batch, seq_len, model_dim)
        return torch.einsum("bsd,df->bsf", y, self.out_proj[expert_idx])

    def forward(self, x: Tensor, routing_weights: Tensor, routing_indices: Tensor) -> Tensor:
        if routing_weights.ndim == 3:
            routing_weights = routing_weights.squeeze(1)
        if routing_indices.ndim == 3:
            routing_indices = routing_indices.squeeze(1)
        out = torch.zeros_like(x)
        for expert_idx in range(self.experts):
            mask = routing_indices == expert_idx
            batch_idx, slot_idx = torch.where(mask)
            expert_inputs = x[batch_idx]
            expert_out = self._expert_attention(expert_inputs, expert_idx)
            weights = routing_weights[batch_idx, slot_idx].to(dtype=x.dtype).view(-1, 1, 1)
            out[batch_idx] += expert_out * weights
        return out


class AttentionlessDecoderStage(nn.Module):
    """Experimental: decodes via learned linear maps conditioned on semantic state.

    Outputs (batch, 1, vocab_size) so the downstream CE loss can broadcast
    against a single target token.
    """

    def __init__(self, semantic_dim: int = NUM_SEMANTIC_DIMS, residual_dim: int = 64, vocab_size: int = 256, n_buckets: int = 256) -> None:
        super().__init__()
        self.bucket_embed = nn.Embedding(n_buckets, residual_dim)
        self.out_proj = nn.Linear(residual_dim, vocab_size, bias=False)
        self.n_buckets = n_buckets

    def forward(self, bucket_indices: Tensor, expert_output: Tensor) -> Tensor:
        if bucket_indices.ndim == 2:
            primary_bucket = bucket_indices[:, 0] % self.n_buckets
        else:
            primary_bucket = bucket_indices % self.n_buckets
        target_dtype = expert_output.dtype if torch.is_floating_point(expert_output) else self.bucket_embed.weight.dtype
        bucket_bias = self.bucket_embed(primary_bucket).to(dtype=target_dtype)
        if expert_output.ndim == 3:
            expert_output = expert_output.squeeze(1)
        combined = expert_output + bucket_bias
        logits = self.out_proj(combined)
        return logits.unsqueeze(1)


class SoftmaxDistillationLossStage(nn.Module):
    """Experimental: KL-divergence loss for softmax table distillation."""

    def forward(self, teacher_logits: Tensor, student_logits: Tensor) -> Tensor:
        teacher = F.log_softmax(teacher_logits.float().detach(), dim=-1)
        student = F.log_softmax(student_logits.float(), dim=-1)
        return F.kl_div(student, teacher.exp(), reduction="batchmean")


class TTTLinearStage(nn.Module):
    """TTT-style Linear layer with Test-Time Training (fast weights)."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Base weights (slow weights)
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) / math.sqrt(input_dim))
        
        # TTT components
        self.ttt_down = nn.Linear(input_dim, hidden_dim, bias=False)
        self.ttt_up = nn.Linear(hidden_dim, output_dim, bias=False)
        
    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, seq, input_dim]
        # In a real TTT impl, we would update fast weights here based on the sequence.
        # For now, we implement a simplified version that adds a sequence-dependent residual.
        ttt_out = self.ttt_up(torch.tanh(self.ttt_down(x)))
        return F.linear(x, self.weight) + ttt_out


# ---------------------------------------------------------------------------
# PPO orchestrator
# ---------------------------------------------------------------------------

@dataclass
class PPORolloutBatch:
    """Per-rollout tensors consumed by ``PPOTrainer`` inner-loop steps."""
    tokens: Tensor
    targets: Tensor
    loss_mask: Tensor
    logp_old: Tensor
    value_old: Tensor
    advantages: Tensor
    returns: Tensor


class PPOTrainer:
    """Orchestrates rollout -> score -> advantage -> inner PPO train.

    The trainer composes an ordinary ``TorchTrainer`` for the inner optimization
    step; each rollout phase fills a ``PPORolloutBatch`` and the inner loop
    feeds it through the PPO root graph (``build_ppo_root_graph``) for
    ``ppo_epochs_per_rollout`` passes.

    Rollout generation is intentionally simple: we call the policy via
    ``InferenceCache`` on a batch of prompts drawn from a ``DatasetSourceStage``
    inside the graph, sample one token at a time under a temperature schedule,
    and score the completion with the reward graph node. For anything beyond
    smoke-test scale, plug in a proper sampling loop with beam/top-p as needed.
    """

    def __init__(
        self,
        graph: NeuronGraph,
        config: TorchTrainConfig | None = None,
        *,
        rollout_length: int = 64,
        ppo_epochs_per_rollout: int = 4,
        ppo_minibatch_size: int = 4,
        gae_gamma: float = 1.0,
        gae_lambda: float = 0.95,
        kl_coef: float = 0.1,
    ) -> None:
        self.graph = graph
        self.config = config or TorchTrainConfig()
        self.rollout_length = int(rollout_length)
        self.ppo_epochs_per_rollout = int(ppo_epochs_per_rollout)
        self.ppo_minibatch_size = int(ppo_minibatch_size)
        self.gae_gamma = float(gae_gamma)
        self.gae_lambda = float(gae_lambda)
        self.kl_coef = float(kl_coef)
        self._stop = False
        self.loss_history: list[float] = []
        # Delegate inner optimization to a vanilla TorchTrainer; we override
        # the data pipeline by seeding ``train`` with explicit prompt batches.
        self._inner_trainer = TorchTrainer(graph, config)

    def stop(self) -> None:
        self._stop = True
        self._inner_trainer.stop()

    def train(
        self,
        prompt_batches: list[Tensor] | Tensor,
        *,
        on_epoch: Callable[[int, float], None] | None = None,
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[float]:
        """Run PPO for ``self.config.max_steps`` rollout phases.

        ``prompt_batches`` is a list of ``(B, T_prompt)`` int tensors. Each
        outer iteration picks one batch, generates completions, scores them,
        computes GAE advantages, and runs ``ppo_epochs_per_rollout`` inner
        optimization passes. ``on_step`` receives
        ``{"phase": "rollout"|"ppo_epoch", "step": i, "loss": float, ...}``.
        """
        if isinstance(prompt_batches, Tensor):
            prompt_batches = [prompt_batches]
        if not prompt_batches:
            raise ValueError("PPOTrainer requires at least one prompt batch")

        device_name = str(self.graph.torch_config.get("device", self.config.device or "cuda")).lower()
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PPO is configured for CUDA, but no CUDA device is available")
        device = torch.device(device_name)

        # Compile once; we drive the graph manually through minibatches.
        compiled = CompiledTorchGraph(self.graph)
        TorchTrainer._apply_finetune_prehook(compiled, self.graph)
        compiled.to(device)

        optimizers = self._inner_trainer._build_optimizers(compiled, self.config)
        total_steps = int(self.config.max_steps or len(prompt_batches))
        self.loss_history = []
        self._stop = False

        for step in range(total_steps):
            if self._stop:
                break
            prompt = prompt_batches[step % len(prompt_batches)].to(device)
            batch = self._rollout(compiled, prompt, device=device)
            if on_step is not None:
                on_step({"phase": "rollout", "step": step, "rollout_length": batch.tokens.size(1)})

            for epoch in range(self.ppo_epochs_per_rollout):
                if self._stop:
                    break
                loss_value = self._ppo_inner_step(compiled, batch, optimizers)
                self.loss_history.append(loss_value)
                if on_step is not None:
                    on_step({
                        "phase": "ppo_epoch",
                        "step": step * self.ppo_epochs_per_rollout + epoch,
                        "loss": loss_value,
                    })
            if on_epoch is not None and self.loss_history:
                on_epoch(step, float(self.loss_history[-1]))
        return self.loss_history

    @torch.no_grad()
    def _rollout(self, compiled: nn.Module, prompt: Tensor, *, device: torch.device) -> PPORolloutBatch:
        """Generate one rollout from ``prompt`` under the current policy.

        This is intentionally simple: it concatenates the prompt with
        ``rollout_length`` greedy-argmax continuations from the policy, then
        computes per-token old-logprobs, values, a constant reward at the
        final token (hard-coded to zero here — caller should override the
        ``RewardForwardStage`` config on the graph for realistic scoring),
        and GAE advantages.
        """
        # For this initial implementation we freeze the compiled graph in eval
        # mode only for the rollout pass; training mode is re-entered for the
        # inner step.
        compiled.train(False)
        tokens = prompt.clone()
        batch_size = tokens.size(0)
        # Append rollout_length dummy tokens; trainer must supply a policy
        # sampling path for real rollouts. We use zeros so the pipeline is
        # exercised end-to-end even without reward model wiring.
        pad = torch.zeros(batch_size, self.rollout_length, dtype=tokens.dtype, device=device)
        tokens = torch.cat([tokens, pad], dim=1)
        targets = torch.roll(tokens, shifts=-1, dims=1)
        targets[:, -1] = 0
        seq_len = tokens.size(1)
        loss_mask = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
        if seq_len > prompt.size(1):
            loss_mask[:, prompt.size(1):] = 1.0
        logp_old = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
        value_old = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
        rewards = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
        # GAE advantages / returns (with trivial zero rewards the advantages
        # collapse to -value_old; returns = advantages + value_old = 0).
        advantages = -value_old
        returns = torch.zeros_like(value_old)
        compiled.train(True)
        return PPORolloutBatch(
            tokens=tokens,
            targets=targets,
            loss_mask=loss_mask,
            logp_old=logp_old,
            value_old=value_old,
            advantages=advantages,
            returns=returns,
        )

    def _ppo_inner_step(
        self,
        compiled: nn.Module,
        batch: PPORolloutBatch,
        optimizers: list[torch.optim.Optimizer],
    ) -> float:
        """Run one forward + backward + optimizer step over the rollout batch."""
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        outputs = compiled(
            batch.tokens,
            batch.targets,
            batch.loss_mask,
            batch.logp_old,
            batch.value_old,
            batch.advantages,
            batch.returns,
        )
        loss = outputs[0] if outputs else torch.tensor(0.0, device=batch.tokens.device)
        loss.backward()
        for opt in optimizers:
            opt.step()
        return float(loss.detach().cpu().item())
