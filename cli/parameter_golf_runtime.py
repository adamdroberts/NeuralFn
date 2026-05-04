from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


PARAMETER_GOLF_CHECKPOINT_FORMAT = "parameter_golf_root_gpt_flat_v1"
DEFAULT_PARAMETER_GOLF_CONTEXT_WINDOW = 2048
DEFAULT_PARAMETER_GOLF_TOKENIZER_CANDIDATES = (
    Path.home() / "Downloads" / "fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model",
    Path.home() / ".cache" / "nfn" / "tokenizers" / "fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model",
    Path.home() / ".cache" / "nfn" / "tokenizers" / "fineweb_8192_bpe.model",
)
CASEOPS_SUPPRESSED_PIECES = frozenset({"▁", "...", "▁...", "....", "....."})


def _is_private_use_char(ch: str) -> bool:
    codepoint = ord(ch)
    return 0xE000 <= codepoint <= 0xF8FF


def _collapse_repeated_spaces(text: str) -> str:
    out: list[str] = []
    previous_space = False
    for ch in text:
        if ch == " ":
            if not previous_space:
                out.append(ch)
            previous_space = True
        else:
            out.append(ch)
            previous_space = False
    return "".join(out)


def sanitize_caseops_decoded_text(text: str) -> str:
    cleaned = "".join(ch for ch in text if not _is_private_use_char(ch) and ch != "\ufffd")
    return _collapse_repeated_spaces(cleaned)


def _processor_is_byte(processor: Any, token_id: int) -> bool:
    is_byte = getattr(processor, "is_byte", None)
    return bool(is_byte(token_id)) if callable(is_byte) else False


class ParameterGolfSentencePieceTokenizer:
    def __init__(self, processor: Any, *, tokenizer_path: Path, caseops: bool):
        self.processor = processor
        self.tokenizer_path = tokenizer_path
        self.caseops = bool(caseops)
        self.lossless_fallback_token_ids = tuple(self._build_lossless_fallback_token_ids())
        self.suppressed_token_ids = tuple(self._build_suppressed_token_ids())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.processor, name)

    def encode(self, text: str, *args, **kwargs):
        return self.processor.encode(text, *args, **kwargs)

    def decode(self, token_ids):
        text = str(self.processor.decode(token_ids))
        return sanitize_caseops_decoded_text(text) if self.caseops else text

    def _build_suppressed_token_ids(self) -> list[int]:
        suppressed: set[int] = set()
        vocab_size = int(self.processor.vocab_size())
        for token_id in range(vocab_size):
            if self.processor.is_unknown(token_id) or self.processor.is_unused(token_id):
                suppressed.add(token_id)
                continue
            piece = str(self.processor.id_to_piece(token_id))
            if self.caseops and (
                _processor_is_byte(self.processor, token_id)
                or piece in CASEOPS_SUPPRESSED_PIECES
                or any(_is_private_use_char(ch) for ch in piece)
            ):
                suppressed.add(token_id)
        pad_id = int(self.processor.pad_id())
        bos_id = int(self.processor.bos_id())
        for token_id in (pad_id, bos_id):
            if token_id >= 0:
                suppressed.add(token_id)
        return sorted(suppressed)

    def _build_lossless_fallback_token_ids(self) -> list[int]:
        if not self.caseops:
            return []
        vocab_size = int(self.processor.vocab_size())
        high_id_floor = max(0, vocab_size - 128)
        fallback: set[int] = set()
        for token_id in range(high_id_floor, vocab_size):
            piece = str(self.processor.id_to_piece(token_id))
            if len(piece) == 1 and not _is_private_use_char(piece):
                fallback.add(token_id)
        return sorted(fallback)


@dataclass(frozen=True)
class ParameterGolfConfig:
    vocab_size: int
    num_layers: int
    model_dim: int
    num_heads: int
    num_kv_heads: int
    mlp_mult: int
    tie_embeddings: bool
    logit_softcap: float = 30.0
    rope_base: float = 10000.0
    qk_gain_init: float = 1.5
    tied_embed_init_std: float = 0.005
    context_window: int = DEFAULT_PARAMETER_GOLF_CONTEXT_WINDOW
    rope_dims: int = 0
    ln_scale: bool = False


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, *, base: float = 10000.0, rope_dims: int = 0):
        super().__init__()
        self.dim = int(dim)
        self.rope_dims = int(rope_dims) if int(rope_dims or 0) > 0 else int(dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached[:, :, :seq_len].to(dtype=dtype), self._sin_cached[:, :, :seq_len].to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ParameterGolfConfig):
        super().__init__()
        dim = int(config.model_dim)
        num_heads = int(config.num_heads)
        num_kv_heads = int(config.num_kv_heads)
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), float(config.qk_gain_init), dtype=torch.float32))
        self.rope_dims = int(config.rope_dims or 0)
        self.rotary = Rotary(self.head_dim, base=float(config.rope_base), rope_dims=self.rope_dims)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))


class MLP(nn.Module):
    def __init__(self, config: ParameterGolfConfig):
        super().__init__()
        dim = int(config.model_dim)
        hidden = int(config.mlp_mult) * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, config: ParameterGolfConfig, *, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = nn.Parameter(torch.ones(config.model_dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(config.model_dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(config.model_dim), torch.zeros(config.model_dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if config.ln_scale else 1.0

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        return x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(
            self.mlp_norm(x_out) * self.ln_scale_factor
        )


class ParameterGolfGPT(nn.Module):
    def __init__(self, config: ParameterGolfConfig):
        super().__init__()
        self.config = config
        self.tie_embeddings = bool(config.tie_embeddings)
        self.logit_softcap = float(config.logit_softcap)
        self.tok_emb = nn.Embedding(int(config.vocab_size), int(config.model_dim))
        self.num_encoder_layers = int(config.num_layers) // 2
        self.num_decoder_layers = int(config.num_layers) - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, int(config.model_dim), dtype=torch.float32))
        self.blocks = nn.ModuleList([Block(config, layer_idx=i) for i in range(int(config.num_layers))])
        self.final_norm = RMSNorm()
        self.lm_head = None if self.tie_embeddings else CastedLinear(int(config.model_dim), int(config.vocab_size), bias=False)

    def forward_hidden(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        hidden = self.forward_hidden(input_ids)
        if self.tie_embeddings:
            logits_proj = F.linear(hidden, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False.")
            logits_proj = self.lm_head(hidden)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


def _coerce_hparam(raw: str) -> object:
    stripped = raw.strip()
    if stripped == "":
        return ""
    if stripped in {"True", "False"}:
        return stripped == "True"
    try:
        if any(ch in stripped for ch in (".", "e", "E")):
            return float(stripped)
        return int(stripped)
    except ValueError:
        return stripped


def load_training_log_hparams(path: str | Path | None) -> dict[str, object]:
    if not path:
        return {}
    log_path = Path(path).expanduser()
    if not log_path.exists():
        raise FileNotFoundError(f"Parameter Golf training log not found: {log_path}")
    hparams: dict[str, object] = {}
    in_hparams = False
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip() == "Hyperparameters:":
            in_hparams = True
            continue
        if in_hparams and line.startswith("="):
            break
        if not in_hparams or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key:
            hparams[key] = _coerce_hparam(value)
    return hparams


def checkpoint_metadata_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.name.endswith(".int8.ptz") or checkpoint_path.name.endswith(".int6.ptz"):
        stem = checkpoint_path.name.rsplit(".", 2)[0]
    else:
        stem = checkpoint_path.stem
    return checkpoint_path.with_name(f"{stem}.meta.json")


def load_checkpoint_metadata(checkpoint_path: Path) -> dict[str, object]:
    meta_path = checkpoint_metadata_path(checkpoint_path)
    if not meta_path.exists():
        return {}
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def load_parameter_golf_state_dict(checkpoint_path: str | Path) -> dict[str, Tensor]:
    path = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Parameter Golf checkpoint must contain a state_dict dictionary: {path}")
    return dict(checkpoint)


def is_parameter_golf_flat_state_dict(state_dict: dict[str, Any]) -> bool:
    return (
        isinstance(state_dict, dict)
        and "tok_emb.weight" in state_dict
        and "skip_weights" in state_dict
        and any(str(key).startswith("blocks.0.attn.c_q.") for key in state_dict)
    )


def infer_config_from_state_dict(
    state_dict: dict[str, Tensor],
    *,
    metadata: dict[str, object] | None = None,
    training_hparams: dict[str, object] | None = None,
    context_window: int | None = None,
) -> ParameterGolfConfig:
    if "tok_emb.weight" not in state_dict:
        raise KeyError("Missing tok_emb.weight in Parameter Golf checkpoint.")
    block_ids = sorted(
        {
            int(str(name).split(".")[1])
            for name in state_dict
            if str(name).startswith("blocks.") and str(name).endswith(".attn_scale")
        }
    )
    if not block_ids:
        raise ValueError("Parameter Golf checkpoint does not contain any transformer blocks.")
    num_layers = block_ids[-1] + 1
    if block_ids != list(range(num_layers)):
        raise ValueError(f"Checkpoint block indices are not contiguous: {block_ids}")

    vocab_size, model_dim = state_dict["tok_emb.weight"].shape
    num_heads = int(state_dict["blocks.0.attn.q_gain"].numel())
    if num_heads <= 0 or int(model_dim) % num_heads != 0:
        raise ValueError(f"Cannot infer valid num_heads from model_dim={model_dim}, num_heads={num_heads}.")
    head_dim = int(model_dim) // num_heads
    c_k_shape = state_dict["blocks.0.attn.c_k.weight"].shape
    if c_k_shape[1] != model_dim or c_k_shape[0] % head_dim != 0:
        raise ValueError(f"Unexpected c_k shape for Parameter Golf checkpoint: {tuple(c_k_shape)}")
    num_kv_heads = int(c_k_shape[0]) // head_dim
    fc_shape = state_dict["blocks.0.mlp.fc.weight"].shape
    if fc_shape[1] != model_dim or fc_shape[0] % model_dim != 0:
        raise ValueError(f"Unexpected MLP fc shape for Parameter Golf checkpoint: {tuple(fc_shape)}")
    mlp_mult = int(fc_shape[0]) // int(model_dim)
    tie_embeddings = "lm_head.weight" not in state_dict

    expected_skip_shape = (min(num_layers // 2, num_layers - (num_layers // 2)), int(model_dim))
    if tuple(state_dict["skip_weights"].shape) != expected_skip_shape:
        raise ValueError(
            f"Unexpected skip_weights shape {tuple(state_dict['skip_weights'].shape)}; expected {expected_skip_shape}."
        )

    hints = {**(metadata or {}), **(training_hparams or {})}
    resolved_context_window = int(
        context_window
        or hints.get("eval_seq_len")
        or hints.get("train_seq_len")
        or DEFAULT_PARAMETER_GOLF_CONTEXT_WINDOW
    )
    rope_dims = int((metadata or {}).get("rope_dims") or 0)
    if rope_dims < 0 or rope_dims > head_dim or rope_dims % 2 != 0:
        rope_dims = 0
    return ParameterGolfConfig(
        vocab_size=int(vocab_size),
        num_layers=int(num_layers),
        model_dim=int(model_dim),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        mlp_mult=int(mlp_mult),
        tie_embeddings=bool(tie_embeddings),
        logit_softcap=float(hints.get("logit_softcap") or 30.0),
        rope_base=float(hints.get("rope_base") or 10000.0),
        qk_gain_init=float(hints.get("qk_gain_init") or 1.5),
        tied_embed_init_std=float(hints.get("tied_embed_init_std") or 0.005),
        context_window=resolved_context_window,
        rope_dims=rope_dims,
        ln_scale=bool((metadata or {}).get("ln_scale", False)),
    )


def build_parameter_golf_model(
    state_dict: dict[str, Tensor],
    config: ParameterGolfConfig,
    *,
    device: torch.device,
) -> ParameterGolfGPT:
    model = ParameterGolfGPT(config)
    model.load_state_dict(state_dict, strict=True)
    if device.type == "cuda":
        model = model.bfloat16()
        for module in model.modules():
            if isinstance(module, CastedLinear):
                module.float()
        for name, param in model.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weights"))) and param.dtype != torch.float32:
                param.data = param.data.float()
    model.to(device)
    model.eval()
    return model


def resolve_parameter_golf_tokenizer_path(
    *,
    checkpoint_path: Path,
    tokenizer_path: str | Path | None = None,
    metadata: dict[str, object] | None = None,
    training_hparams: dict[str, object] | None = None,
) -> Path:
    raw_candidates: list[str | Path] = []
    if tokenizer_path:
        raw_candidates.append(tokenizer_path)
    for source in (metadata or {}, training_hparams or {}):
        raw_value = source.get("tokenizer_path")
        if isinstance(raw_value, str) and raw_value.strip():
            raw_candidates.append(raw_value)
    raw_candidates.extend(DEFAULT_PARAMETER_GOLF_TOKENIZER_CANDIDATES)

    candidates: list[Path] = []
    for raw in raw_candidates:
        candidate = Path(raw).expanduser()
        candidates.append(candidate)
        if not candidate.is_absolute():
            candidates.append((checkpoint_path.parent / candidate).expanduser())
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    formatted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not resolve a Parameter Golf SentencePiece tokenizer. "
        f"Pass --checkpoint-tokenizer with a .model path. Tried: {formatted}"
    )


def load_sentencepiece_tokenizer(tokenizer_path: Path, *, expected_vocab_size: int):
    try:
        import sentencepiece as spm  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Parameter Golf checkpoint inference requires the sentencepiece package.") from exc
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    actual_vocab_size = int(tokenizer.vocab_size())
    if actual_vocab_size != int(expected_vocab_size):
        raise ValueError(
            f"Tokenizer vocab_size={actual_vocab_size} does not match checkpoint vocab_size={expected_vocab_size}."
        )
    caseops = "caseops" in tokenizer_path.name.lower()
    return ParameterGolfSentencePieceTokenizer(
        tokenizer,
        tokenizer_path=tokenizer_path,
        caseops=caseops,
    )
