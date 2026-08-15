from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import struct
import subprocess
from typing import Sequence

import pytest
import torch
import torch.nn.functional as F

import neuralfn.native_gguf as gguf
import neuralfn.native_muse_glimmer_checkpoint as glimmer_checkpoint


ROOT = Path(__file__).resolve().parents[1]
D = 8
Q = 4
KV = 2
FF = 16
VOCAB = 13
LAYERS = 4
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 2
WINDOW = 3
EPS = 1.0e-5
POST_EPS = 1.0e-8
Q_SCALE = 3.87
OUTPUT_MULTIPLIER = 0.19611613513818404
SOFTCAP = 20.0


def _gguf_string(value: bytes) -> bytes:
    return struct.pack("<Q", len(value)) + value


def _gguf_metadata(key: str, value_type: int, payload: bytes) -> bytes:
    return _gguf_string(key.encode()) + struct.pack("<I", value_type) + payload


def _write_sparse_canonical_dflash_gguf(path: Path) -> int:
    u32 = lambda value: struct.pack("<I", value)
    f32 = lambda value: struct.pack("<f", value)
    string = lambda value: _gguf_string(value.encode())
    entries = [
        _gguf_metadata("general.architecture", 8, string("dflash")),
        _gguf_metadata("general.type", 8, string("model")),
        _gguf_metadata("general.name", 8, string("Hf_Museglimmer")),
        _gguf_metadata("general.size_label", 8, string("2.6B")),
    ]
    scalar_values = (
        ("dflash.block_count", 5),
        ("dflash.context_length", 131_072),
        ("dflash.embedding_length", 6_656),
        ("dflash.feed_forward_length", 19_968),
        ("dflash.attention.head_count", 32),
        ("dflash.attention.head_count_kv", 8),
    )
    entries.extend(_gguf_metadata(key, 4, u32(value)) for key, value in scalar_values)
    entries.extend(
        (
            _gguf_metadata("dflash.rope.freq_base", 6, f32(500_000.0)),
            _gguf_metadata(
                "dflash.attention.layer_norm_rms_epsilon", 6, f32(1.0e-5)
            ),
        )
    )
    entries.extend(
        _gguf_metadata(key, 4, u32(value))
        for key, value in (
            ("dflash.attention.key_length", 128),
            ("dflash.attention.value_length", 128),
            ("dflash.block_size", 16),
        )
    )
    entries.append(
        _gguf_metadata(
            "dflash.target_layers",
            9,
            u32(5) + struct.pack("<Q5i", 5, 2, 14, 26, 38, 50),
        )
    )
    entries.append(
        _gguf_metadata("dflash.attention.sliding_window", 4, u32(2_048))
    )
    entries.append(
        _gguf_metadata(
            "dflash.attention.sliding_window_pattern",
            9,
            u32(7) + struct.pack("<Q", 5) + bytes((1,) * 5),
        )
    )
    entries.append(_gguf_metadata("general.quantization_version", 4, u32(2)))
    entries.extend(
        (
            _gguf_metadata("tokenizer.ggml.model", 8, string("gpt2")),
            _gguf_metadata("tokenizer.ggml.pre", 8, string("llama4")),
        )
    )
    repeated_string = _gguf_string(b"x")
    entries.append(
        _gguf_metadata(
            "tokenizer.ggml.tokens",
            9,
            u32(8) + struct.pack("<Q", 202_048) + repeated_string * 202_048,
        )
    )
    entries.append(
        _gguf_metadata(
            "tokenizer.ggml.token_type",
            9,
            u32(5) + struct.pack("<Q", 202_048) + struct.pack("<i", 1) * 202_048,
        )
    )
    entries.append(
        _gguf_metadata(
            "tokenizer.ggml.merges",
            9,
            u32(8) + struct.pack("<Q", 439_802) + repeated_string * 439_802,
        )
    )
    entries.extend(
        _gguf_metadata(key, 4, u32(value))
        for key, value in (
            ("tokenizer.ggml.bos_token_id", 200_000),
            ("tokenizer.ggml.eos_token_id", 200_001),
            ("tokenizer.ggml.padding_token_id", 200_018),
        )
    )
    entries.extend(
        (
            _gguf_metadata("tokenizer.ggml.add_bos_token", 7, b"\x01"),
            _gguf_metadata("tokenizer.ggml.add_sep_token", 7, b"\x00"),
            _gguf_metadata("tokenizer.ggml.eot_token_id", 4, u32(200_008)),
            _gguf_metadata("tokenizer.ggml.mask_token_id", 5, struct.pack("<i", 201_818)),
            _gguf_metadata("general.file_type", 4, u32(15)),
        )
    )
    prefix = b"GGUF" + struct.pack("<IQQ", 3, 58, 33) + b"".join(entries)
    chat_prefix = _gguf_string(b"tokenizer.chat_template") + struct.pack("<I", 8)
    chat_size = 13_072_791 - len(prefix) - len(chat_prefix) - 8
    assert chat_size > 0
    metadata = prefix + chat_prefix + _gguf_string(b"x" * chat_size)
    assert len(metadata) == 13_072_791

    f32_names = {"enc.output_norm.weight", "output_norm.weight"}
    q6_names: set[str] = set()
    for layer in range(5):
        f32_names.update(
            {
                f"blk.{layer}.attn_norm.weight",
                f"blk.{layer}.ffn_norm.weight",
                f"blk.{layer}.attn_k_norm.weight",
                f"blk.{layer}.attn_q_norm.weight",
            }
        )
        q6_names.update(
            {f"blk.{layer}.ffn_down.weight", f"blk.{layer}.attn_v.weight"}
        )
    table = bytearray()
    offset = 0
    resident_bytes = 0
    for name in gguf.expected_muse_glimmer_dflash_gguf_tensor_names():
        shape, _native_name = gguf._dflash_tensor_contract(name)
        dimensions = tuple(reversed(shape))
        tensor_type = (
            gguf.GGML_TYPE_F32
            if name in f32_names
            else gguf.GGML_TYPE_Q6_K
            if name in q6_names
            else gguf.GGML_TYPE_Q4_K
        )
        table.extend(_gguf_string(name.encode()))
        table.extend(struct.pack("<I", len(dimensions)))
        table.extend(struct.pack("<" + "Q" * len(dimensions), *dimensions))
        table.extend(struct.pack("<IQ", tensor_type, offset))
        _encoding, block_elements, block_bytes = gguf.GGML_TYPE_LAYOUTS[tensor_type]
        nbytes = dimensions[0] // block_elements * block_bytes
        for dimension in dimensions[1:]:
            nbytes *= dimension
        resident_bytes += nbytes
        offset = gguf._align(offset + nbytes, 32)
    assert len(metadata) + len(table) == 13_076_149
    data_offset = gguf._align(len(metadata) + len(table), 32)
    with path.open("wb") as stream:
        stream.write(metadata)
        stream.write(table)
        stream.write(bytes(data_offset - stream.tell()))
        stream.truncate(gguf.DFLASH_KQUANT_PROFILE["nbytes"])
    assert data_offset + offset == path.stat().st_size
    return resident_bytes


def _bf16_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if bits & 0x7F800000 == 0x7F800000:
        return (bits >> 16) & 0xFFFF
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def _bf16_value(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _layout() -> list[tuple[str, tuple[int, ...], bool]]:
    rows: list[tuple[str, tuple[int, ...], bool]] = [
        ("embedding", (VOCAB, D), False),
    ]
    for layer in range(LAYERS):
        prefix = f"layers.{layer}."
        rows.extend(
            [
                (prefix + "input_norm", (D,), True),
                (prefix + "post_attention_norm", (D,), True),
                (prefix + "pre_feedforward_norm", (D,), True),
                (prefix + "post_feedforward_norm", (D,), True),
                (prefix + "q", (Q, D), False),
                (prefix + "k", (KV, D), False),
                (prefix + "v", (KV, D), False),
                (prefix + "attn_gate", (Q, D), False),
                (prefix + "o", (D, Q), False),
                (prefix + "mlp_gate", (FF, D), False),
                (prefix + "mlp_up", (FF, D), False),
                (prefix + "mlp_down", (D, FF), False),
            ]
        )
    rows.extend(
        [
            ("final_norm", (D,), False),
            ("lm_head", (VOCAB, D), False),
        ]
    )
    return rows


def _product(shape: Sequence[int]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result


def _fixture_value(name: str, global_index: int, local_index: int, centered: bool) -> float:
    if centered:
        return 0.045 * math.sin(global_index * 0.07 + local_index * 0.13)
    if name == "final_norm":
        return 0.94 + 0.08 * math.cos(local_index * 0.19)
    scale = 0.16 if name in {"embedding", "lm_head"} else 0.075
    return (
        scale * math.sin(global_index * 0.031 + local_index * 0.17)
        + 0.027 * math.cos(global_index * 0.019 - local_index * 0.11)
    )


def _write_checkpoint(path: Path) -> tuple[str, dict[str, torch.Tensor]]:
    payload = bytearray()
    weights: dict[str, torch.Tensor] = {}
    global_index = 0
    for name, shape, centered in _layout():
        values: list[float] = []
        bits: list[int] = []
        for local_index in range(_product(shape)):
            encoded = _bf16_bits(
                _fixture_value(name, global_index, local_index, centered)
            )
            bits.append(encoded)
            values.append(_bf16_value(encoded))
            global_index += 1
        payload.extend(struct.pack(f"<{len(bits)}H", *bits))
        weights[name] = torch.tensor(values, dtype=torch.float32).reshape(shape)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), weights


def _write_q_lora(
    path: Path,
    weights: dict[str, torch.Tensor],
    *,
    rank: int = 2,
    alpha: float = 4.0,
) -> tuple[str, dict[str, torch.Tensor]]:
    payload = bytearray()
    adapted = {name: value.clone() for name, value in weights.items()}
    for layer in range(LAYERS):
        a = torch.empty((rank, D), dtype=torch.float32)
        b = torch.empty((Q, rank), dtype=torch.float32)
        for row in range(rank):
            for col in range(D):
                a[row, col] = 0.031 * math.sin(1.7 + layer + row * 3 + col)
        for row in range(Q):
            for col in range(rank):
                b[row, col] = 0.047 * math.cos(0.9 + layer + row * 2 + col)
        a = a.to(torch.bfloat16).float()
        b = b.to(torch.bfloat16).float()
        payload.extend(a.to(torch.bfloat16).view(torch.uint16).numpy().tobytes())
        payload.extend(b.to(torch.bfloat16).view(torch.uint16).numpy().tobytes())
        adapted[f"layers.{layer}.q"] += (alpha / rank) * (b @ a)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), adapted


def _rms(
    value: torch.Tensor,
    *,
    eps: float,
    weight: torch.Tensor | None = None,
    centered: bool = False,
) -> torch.Tensor:
    result = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        result = result * ((1.0 + weight) if centered else weight)
    return result


def _rope(value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    half = HEAD_DIM // 2
    inverse = 1.0 / (
        500000.0
        ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    angle = positions.float().unsqueeze(1) * inverse.unsqueeze(0)
    cosine = angle.cos().unsqueeze(1)
    sine = angle.sin().unsqueeze(1)
    first, second = value[..., :half], value[..., half:]
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1)


def _oracle(weights: dict[str, torch.Tensor], token_ids: Sequence[int]) -> list[float]:
    hidden = F.embedding(torch.tensor(token_ids), weights["embedding"])
    hidden = _rms(hidden, eps=EPS)
    seq = len(token_ids)
    positions = torch.arange(seq)
    row = positions.unsqueeze(1)
    col = positions.unsqueeze(0)
    for layer in range(LAYERS):
        prefix = f"layers.{layer}."
        residual = hidden
        normed = _rms(
            hidden,
            eps=EPS,
            weight=weights[prefix + "input_norm"],
            centered=True,
        )
        query = F.linear(normed, weights[prefix + "q"]).view(seq, HEADS, HEAD_DIM)
        key = F.linear(normed, weights[prefix + "k"]).view(seq, KV_HEADS, HEAD_DIM)
        value = F.linear(normed, weights[prefix + "v"]).view(seq, KV_HEADS, HEAD_DIM)
        query = _rms(query, eps=EPS) * Q_SCALE
        key = _rms(key, eps=EPS)
        if layer % 4 != 3:
            query = _rope(query, positions)
            key = _rope(key, positions)
        key = key.repeat_interleave(HEADS // KV_HEADS, dim=1)
        value = value.repeat_interleave(HEADS // KV_HEADS, dim=1)
        scores = torch.einsum("qhd,khd->hqk", query, key) / math.sqrt(HEAD_DIM)
        allowed = col <= row
        if layer % 4 != 3:
            allowed &= col > row - WINDOW
        scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        attention = torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), value)
        attention = attention.reshape(seq, Q)
        gate = torch.sigmoid(F.linear(normed, weights[prefix + "attn_gate"]))
        projected = F.linear(attention * gate, weights[prefix + "o"])
        projected = _rms(
            projected,
            eps=POST_EPS,
            weight=weights[prefix + "post_attention_norm"],
            centered=True,
        )
        hidden = residual + projected
        residual = hidden
        normed = _rms(
            hidden,
            eps=EPS,
            weight=weights[prefix + "pre_feedforward_norm"],
            centered=True,
        )
        mlp = F.silu(F.linear(normed, weights[prefix + "mlp_gate"]))
        mlp = mlp * F.linear(normed, weights[prefix + "mlp_up"])
        mlp = F.linear(mlp, weights[prefix + "mlp_down"])
        mlp = _rms(
            mlp,
            eps=POST_EPS,
            weight=weights[prefix + "post_feedforward_norm"],
            centered=True,
        )
        hidden = residual + mlp
    hidden = _rms(hidden, eps=EPS, weight=weights["final_norm"])
    logits = F.linear(hidden[-1], weights["lm_head"])
    logits = SOFTCAP * torch.tanh(logits * OUTPUT_MULTIPLIER / SOFTCAP)
    return logits.tolist()


def _target_rows_and_taps(
    weights: dict[str, torch.Tensor], token_ids: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = F.embedding(torch.tensor(token_ids), weights["embedding"])
    hidden = _rms(hidden, eps=EPS)
    seq = len(token_ids)
    positions = torch.arange(seq)
    row = positions.unsqueeze(1)
    col = positions.unsqueeze(0)
    taps: list[torch.Tensor] = []
    for layer in range(LAYERS):
        prefix = f"layers.{layer}."
        residual = hidden
        normed = _rms(
            hidden, eps=EPS, weight=weights[prefix + "input_norm"], centered=True
        )
        query = F.linear(normed, weights[prefix + "q"]).view(seq, HEADS, HEAD_DIM)
        key = F.linear(normed, weights[prefix + "k"]).view(seq, KV_HEADS, HEAD_DIM)
        value = F.linear(normed, weights[prefix + "v"]).view(seq, KV_HEADS, HEAD_DIM)
        query = _rms(query, eps=EPS) * Q_SCALE
        key = _rms(key, eps=EPS)
        if layer % 4 != 3:
            query = _rope(query, positions)
            key = _rope(key, positions)
        repeated_key = key.repeat_interleave(HEADS // KV_HEADS, dim=1)
        repeated_value = value.repeat_interleave(HEADS // KV_HEADS, dim=1)
        scores = torch.einsum("qhd,khd->hqk", query, repeated_key) / math.sqrt(HEAD_DIM)
        allowed = col <= row
        if layer % 4 != 3:
            allowed &= col > row - WINDOW
        attention = torch.einsum(
            "hqk,khd->qhd",
            scores.masked_fill(~allowed.unsqueeze(0), float("-inf")).softmax(dim=-1),
            repeated_value,
        ).reshape(seq, Q)
        gate = torch.sigmoid(F.linear(normed, weights[prefix + "attn_gate"]))
        projected = _rms(
            F.linear(attention * gate, weights[prefix + "o"]),
            eps=POST_EPS,
            weight=weights[prefix + "post_attention_norm"],
            centered=True,
        )
        hidden = residual + projected
        residual = hidden
        normed = _rms(
            hidden,
            eps=EPS,
            weight=weights[prefix + "pre_feedforward_norm"],
            centered=True,
        )
        mlp = F.silu(F.linear(normed, weights[prefix + "mlp_gate"]))
        mlp *= F.linear(normed, weights[prefix + "mlp_up"])
        mlp = _rms(
            F.linear(mlp, weights[prefix + "mlp_down"]),
            eps=POST_EPS,
            weight=weights[prefix + "post_feedforward_norm"],
            centered=True,
        )
        hidden = residual + mlp
        if layer in {0, 2}:
            taps.append(hidden.clone())
    final = _rms(hidden, eps=EPS, weight=weights["final_norm"])
    logits = F.linear(final, weights["lm_head"])
    logits = SOFTCAP * torch.tanh(logits * OUTPUT_MULTIPLIER / SOFTCAP)
    return logits, torch.cat(taps, dim=-1)


def _assistant_layout() -> list[tuple[str, tuple[int, ...]]]:
    rows: list[tuple[str, tuple[int, ...]]] = []
    for layer in range(2):
        prefix = f"assistant.{layer}."
        rows.extend(
            [
                (prefix + "q", (Q, D)),
                (prefix + "k", (KV, D)),
                (prefix + "v", (KV, D)),
                (prefix + "o", (D, Q)),
                (prefix + "q_norm", (HEAD_DIM,)),
                (prefix + "k_norm", (HEAD_DIM,)),
                (prefix + "mlp_gate", (FF, D)),
                (prefix + "mlp_up", (FF, D)),
                (prefix + "mlp_down", (D, FF)),
                (prefix + "post_attention_norm", (D,)),
                (prefix + "input_norm", (D,)),
            ]
        )
    rows.extend(
        [
            ("assistant.final_norm", (D,)),
            ("assistant.context_projection", (D, 2 * D)),
            ("assistant.context_norm", (D,)),
        ]
    )
    return rows


def _write_assistant_checkpoint(path: Path) -> tuple[str, dict[str, torch.Tensor]]:
    payload = bytearray()
    weights: dict[str, torch.Tensor] = {}
    global_index = 0
    for name, shape in _assistant_layout():
        values: list[float] = []
        bits: list[int] = []
        norm = name.endswith("norm") or "_norm" in name
        for local_index in range(_product(shape)):
            value = (
                0.91 + 0.12 * math.cos(global_index * 0.029 + local_index * 0.17)
                if norm
                else 0.068 * math.sin(global_index * 0.037 + local_index * 0.13)
                + 0.021 * math.cos(global_index * 0.023 - local_index * 0.09)
            )
            encoded = _bf16_bits(value)
            bits.append(encoded)
            values.append(_bf16_value(encoded))
            global_index += 1
        payload.extend(struct.pack(f"<{len(bits)}H", *bits))
        weights[name] = torch.tensor(values, dtype=torch.float32).reshape(shape)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), weights


def _assistant_candidates(
    target_weights: dict[str, torch.Tensor],
    assistant_weights: dict[str, torch.Tensor],
    context_taps: torch.Tensor,
    anchor: int,
) -> list[int]:
    context = F.linear(context_taps, assistant_weights["assistant.context_projection"])
    context = _rms(
        context,
        eps=EPS,
        weight=assistant_weights["assistant.context_norm"],
    )
    noise_ids = torch.tensor([anchor, 12, 12, 12])
    hidden = F.embedding(noise_ids, target_weights["embedding"])
    context_length = context.shape[0]
    block = hidden.shape[0]
    q_positions = torch.arange(context_length, context_length + block)
    all_positions = torch.arange(context_length + block)
    for layer in range(2):
        prefix = f"assistant.{layer}."
        residual = hidden
        normalized = _rms(
            hidden,
            eps=EPS,
            weight=assistant_weights[prefix + "input_norm"],
        )
        query = F.linear(normalized, assistant_weights[prefix + "q"]).view(
            block, HEADS, HEAD_DIM
        )
        kv_input = torch.cat((context, normalized), dim=0)
        key = F.linear(kv_input, assistant_weights[prefix + "k"]).view(
            context_length + block, KV_HEADS, HEAD_DIM
        )
        value = F.linear(kv_input, assistant_weights[prefix + "v"]).view(
            context_length + block, KV_HEADS, HEAD_DIM
        )
        query = _rms(
            query, eps=EPS, weight=assistant_weights[prefix + "q_norm"]
        )
        key = _rms(key, eps=EPS, weight=assistant_weights[prefix + "k_norm"])
        query = _rope(query, q_positions)
        key = _rope(key, all_positions)
        key = key.repeat_interleave(HEADS // KV_HEADS, dim=1)
        value = value.repeat_interleave(HEADS // KV_HEADS, dim=1)
        scores = torch.einsum("qhd,khd->hqk", query, key) / math.sqrt(HEAD_DIM)
        allowed = (
            q_positions.unsqueeze(1) - all_positions.unsqueeze(0)
        ).abs() <= WINDOW
        attention = torch.einsum(
            "hqk,khd->qhd",
            scores.masked_fill(~allowed.unsqueeze(0), float("-inf")).softmax(dim=-1),
            value,
        ).reshape(block, Q)
        hidden = residual + F.linear(attention, assistant_weights[prefix + "o"])
        residual = hidden
        normalized = _rms(
            hidden,
            eps=EPS,
            weight=assistant_weights[prefix + "post_attention_norm"],
        )
        mlp = F.silu(F.linear(normalized, assistant_weights[prefix + "mlp_gate"]))
        mlp *= F.linear(normalized, assistant_weights[prefix + "mlp_up"])
        hidden = residual + F.linear(mlp, assistant_weights[prefix + "mlp_down"])
    hidden = _rms(
        hidden, eps=EPS, weight=assistant_weights["assistant.final_norm"]
    )
    logits = F.linear(hidden[1:], target_weights["lm_head"])
    return logits.argmax(dim=-1).tolist()


@pytest.fixture(scope="session")
def glimmer_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("native-glimmer-probe") / "probe"
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O2",
            "-pthread",
            "-I",
            str(ROOT / "neuralfn" / "csrc" / "native_gpt2"),
            str(ROOT / "tests" / "cpp" / "resident_glimmer_probe.cpp"),
            str(ROOT / "neuralfn" / "csrc" / "native_gpt2" / "resident_glimmer.cpp"),
            str(ROOT / "neuralfn" / "csrc" / "native_gpt2" / "resident_glimmer_vision.cpp"),
            str(
                ROOT
                / "neuralfn"
                / "csrc"
                / "native_gpt2"
                / "resident_glimmer_assistant.cpp"
            ),
            str(ROOT / "neuralfn" / "csrc" / "native_gpt2" / "resident_glimmer_cuda.cpp"),
            "-ldl",
            "-o",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return output


@pytest.fixture(scope="session")
def glimmer_dflash_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("native-glimmer-dflash-probe") / "probe"
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O2",
            "-pthread",
            "-I",
            str(ROOT / "neuralfn" / "csrc" / "native_gpt2"),
            str(ROOT / "tests" / "cpp" / "resident_glimmer_dflash_probe.cpp"),
            str(ROOT / "neuralfn" / "csrc" / "native_gpt2" / "resident_glimmer.cpp"),
            str(ROOT / "neuralfn" / "csrc" / "native_gpt2" / "resident_glimmer_vision.cpp"),
            str(
                ROOT
                / "neuralfn"
                / "csrc"
                / "native_gpt2"
                / "resident_glimmer_assistant.cpp"
            ),
            str(ROOT / "neuralfn" / "csrc" / "native_gpt2" / "resident_glimmer_cuda.cpp"),
            "-ldl",
            "-o",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return output


@pytest.fixture(scope="session")
def glimmer_dflash_gguf_load_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("native-glimmer-dflash-gguf-probe") / "probe"
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O2",
            "-pthread",
            "-I",
            str(ROOT / "neuralfn" / "csrc" / "native_gpt2"),
            str(
                ROOT
                / "tests"
                / "cpp"
                / "resident_glimmer_dflash_gguf_load_probe.cpp"
            ),
            str(ROOT / "neuralfn" / "csrc" / "native_gpt2" / "resident_glimmer.cpp"),
            str(ROOT / "neuralfn" / "csrc" / "native_gpt2" / "resident_glimmer_vision.cpp"),
            str(
                ROOT
                / "neuralfn"
                / "csrc"
                / "native_gpt2"
                / "resident_glimmer_assistant.cpp"
            ),
            str(ROOT / "neuralfn" / "csrc" / "native_gpt2" / "resident_glimmer_cuda.cpp"),
            "-ldl",
            "-o",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return output


@pytest.fixture(scope="session")
def fake_glimmer_cuda(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    output = tmp_path_factory.mktemp("fake-glimmer-cuda")
    runtime = output / "libcudart-fake.so"
    tile = output / "libnfn-glimmer-tile-fake.so"
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-shared",
            str(ROOT / "tests" / "cpp" / "fake_cuda_runtime.cpp"),
            "-o",
            str(runtime),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-shared",
            "-I",
            str(ROOT / "neuralfn" / "csrc" / "native_train"),
            str(ROOT / "tests" / "cpp" / "fake_glimmer_tile_ops.cpp"),
            "-o",
            str(tile),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return runtime, tile


def _run(
    probe: Path,
    checkpoint: Path,
    digest: str,
    mode: str,
    prompt: Sequence[int],
    *,
    truncate: int | None = None,
    append: Sequence[int] = (),
    cuda_libraries: tuple[Path, Path] | None = None,
    lora: tuple[Path, str, int, float, int] | None = None,
) -> tuple[list[float], list[str], list[str], str, list[str]]:
    arguments = [
        str(probe),
        str(checkpoint),
        digest,
        mode,
        ",".join(map(str, prompt)),
    ]
    if truncate is not None:
        arguments.extend((str(truncate), ",".join(map(str, append))))
    environment = os.environ.copy()
    if cuda_libraries is not None:
        environment["NFN_TEST_GLIMMER_CUDA_RUNTIME"] = str(cuda_libraries[0])
        environment["NFN_TEST_GLIMMER_TILE_OPS"] = str(cuda_libraries[1])
    if lora is not None:
        path, sha256, rank, alpha, target_mask = lora
        environment["NFN_TEST_GLIMMER_LORA"] = str(path)
        environment["NFN_TEST_GLIMMER_LORA_SHA256"] = sha256
        environment["NFN_TEST_GLIMMER_LORA_RANK"] = str(rank)
        environment["NFN_TEST_GLIMMER_LORA_ALPHA"] = str(alpha)
        environment["NFN_TEST_GLIMMER_LORA_TARGET_MASK"] = str(target_mask)
    completed = subprocess.run(
        arguments, check=True, capture_output=True, text=True, env=environment
    )
    lines = completed.stdout.splitlines()
    return (
        [float(value) for value in lines[0].split(",")],
        lines[1].split(","),
        lines[2].split(","),
        lines[3],
        lines[4].split(","),
    )


def test_resident_glimmer_bf16_schedule_hybrid_cache_and_transactions(
    glimmer_probe: Path,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny-glimmer.bf16"
    digest, weights = _write_checkpoint(checkpoint)
    prompt = [2, 5, 1, 9, 4]
    expected = _oracle(weights, prompt)
    off = _run(glimmer_probe, checkpoint, digest, "off", prompt)
    full = _run(glimmer_probe, checkpoint, digest, "full", prompt)
    assert off[0] == pytest.approx(expected, abs=2.5e-5)
    assert full[0] == pytest.approx(expected, abs=2.5e-5)
    assert full[0] == pytest.approx(off[0], abs=2.0e-7)
    # Three local layers retain W=3 rows, one global layer retains all five:
    # (3*3 + 5) rows * (K+V)*KVdim * sizeof(float), plus one final hidden row.
    assert [int(value) for value in full[1]] == [5, 5, 256, 256]
    assert [int(value) for value in off[1]] == [5, 0, 0, 0]
    assert int(full[2][0]) == max(range(VOCAB), key=expected.__getitem__)
    assert int(off[2][0]) == int(full[2][0])
    assert full[3] == off[3] == "cancelled"

    replacement = [2, 5, 8, 7]
    replaced = _run(
        glimmer_probe,
        checkpoint,
        digest,
        "full",
        prompt,
        truncate=2,
        append=[8, 7],
    )
    assert replaced[0] == pytest.approx(_oracle(weights, replacement), abs=2.5e-5)
    assert [int(value) for value in replaced[1]] == [4, 4, 240, 240]


def test_resident_glimmer_applies_strict_bf16_lora_without_mutating_base(
    glimmer_probe: Path,
    fake_glimmer_cuda: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny-glimmer.bf16"
    digest, weights = _write_checkpoint(checkpoint)
    adapter = tmp_path / "tiny-q-lora.bf16"
    adapter_sha, adapted_weights = _write_q_lora(adapter, weights)
    prompt = [2, 5, 1, 9]

    base = _run(glimmer_probe, checkpoint, digest, "full", prompt)
    adapted = _run(
        glimmer_probe,
        checkpoint,
        digest,
        "full",
        prompt,
        lora=(adapter, adapter_sha, 2, 4.0, 1 << 0),
    )
    base_cuda = _run(
        glimmer_probe,
        checkpoint,
        digest,
        "full",
        prompt,
        cuda_libraries=fake_glimmer_cuda,
    )
    adapted_cuda = _run(
        glimmer_probe,
        checkpoint,
        digest,
        "full",
        prompt,
        cuda_libraries=fake_glimmer_cuda,
        lora=(adapter, adapter_sha, 2, 4.0, 1 << 0),
    )
    assert base[0] == pytest.approx(_oracle(weights, prompt), abs=2.5e-5)
    assert adapted[0] == pytest.approx(_oracle(adapted_weights, prompt), abs=2.5e-5)
    # The fake CUDA runner stores the same BF16 KV rows as production; its
    # accumulated decode tolerance is therefore wider than the all-FP32 CPU cache.
    assert adapted_cuda[0] == pytest.approx(adapted[0], abs=2.5e-3)
    assert adapted[0] != pytest.approx(base[0], abs=1.0e-7)
    assert adapted_cuda[0] != pytest.approx(base_cuda[0], abs=1.0e-7)

    corrupted = subprocess.run(
        [str(glimmer_probe), str(checkpoint), digest, "full", "2,5"],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "NFN_TEST_GLIMMER_LORA": str(adapter),
            "NFN_TEST_GLIMMER_LORA_SHA256": "0" * 64,
            "NFN_TEST_GLIMMER_LORA_RANK": "2",
            "NFN_TEST_GLIMMER_LORA_ALPHA": "4",
            "NFN_TEST_GLIMMER_LORA_TARGET_MASK": "1",
        },
    )
    assert corrupted.returncode == 1
    assert "SHA-256" in corrupted.stderr


def test_resident_glimmer_rejects_wrong_fingerprint(
    glimmer_probe: Path,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny-glimmer.bf16"
    _digest, _weights = _write_checkpoint(checkpoint)
    completed = subprocess.run(
        [str(glimmer_probe), str(checkpoint), "0" * 64, "full", "1"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "SHA-256 fingerprint" in completed.stderr


def test_resident_glimmer_cuda_orchestration_matches_bf16_oracle(
    glimmer_probe: Path,
    fake_glimmer_cuda: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny-glimmer-cuda.bf16"
    digest, weights = _write_checkpoint(checkpoint)
    prompt = [2, 5, 1, 9, 4]
    expected = _oracle(weights, prompt)
    result = _run(
        glimmer_probe,
        checkpoint,
        digest,
        "full",
        prompt,
        cuda_libraries=fake_glimmer_cuda,
    )
    # The CUDA contract stores K/V as BF16, whereas the portable CPU oracle
    # intentionally retains float32 cache rows.
    assert result[0] == pytest.approx(expected, abs=2.5e-3)
    assert [int(value) for value in result[1]] == [5, 5, 144, 144]
    assert int(result[2][0]) == max(range(VOCAB), key=result[0].__getitem__)
    assert result[3] == "cancelled"
    assert [int(value) for value in result[4]] == [1, 1]


def test_resident_dflash_matches_independent_torch_block_oracle(
    glimmer_dflash_probe: Path,
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "tiny-glimmer-target.bf16"
    assistant_path = tmp_path / "tiny-glimmer-assistant.bf16"
    target_digest, target_weights = _write_checkpoint(target_path)
    assistant_digest, assistant_weights = _write_assistant_checkpoint(assistant_path)
    prompt = [2, 5, 1]

    prompt_logits, prompt_taps = _target_rows_and_taps(target_weights, prompt)
    anchor = int(prompt_logits[-1].argmax())
    proposals = _assistant_candidates(
        target_weights,
        assistant_weights,
        prompt_taps,
        anchor,
    )
    expected: list[int] = []
    accepted = 0
    history = [*prompt, anchor]
    for candidate in proposals:
        target_logits, _taps = _target_rows_and_taps(target_weights, history)
        target_token = int(target_logits[-1].argmax())
        if target_token != candidate:
            expected.append(target_token)
            history.append(target_token)
            break
        expected.append(candidate)
        history.append(candidate)
        accepted += 1
    else:
        bonus_logits, _taps = _target_rows_and_taps(target_weights, history)
        bonus = int(bonus_logits[-1].argmax())
        expected.append(bonus)
        history.append(bonus)

    completed = subprocess.run(
        [
            str(glimmer_dflash_probe),
            str(target_path),
            target_digest,
            str(assistant_path),
            assistant_digest,
            ",".join(map(str, prompt)),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.splitlines()
    assert [int(value) for value in lines[0].split(",")] == [anchor]
    assert [int(value) for value in lines[1].split(",")] == expected
    counters = [int(value) for value in lines[2].split(",")]
    assert counters[:3] == [3, accepted, 3 - accepted]
    assert counters[4] == 1
    # Verification first evaluates all three tentative target rows. A rejected
    # block is then rebuilt and its committed prefix/correction is evaluated.
    assert counters[3] == (4 if accepted == 3 else 3 + len(expected))
    stats = [int(value) for value in lines[3].split(",")]
    assert stats[:2] == [len(history), len(history)]
    assert stats[2:6] == [1, 3, accepted, 3 - accepted]
    assert stats[6] > 0


def test_resident_dflash_cuda_model_compute_matches_block_oracle(
    glimmer_dflash_probe: Path,
    fake_glimmer_cuda: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "tiny-glimmer-target-cuda.bf16"
    assistant_path = tmp_path / "tiny-glimmer-assistant-cuda.bf16"
    target_digest, target_weights = _write_checkpoint(target_path)
    assistant_digest, assistant_weights = _write_assistant_checkpoint(assistant_path)
    prompt = [2, 5, 1]
    prompt_logits, prompt_taps = _target_rows_and_taps(target_weights, prompt)
    anchor = int(prompt_logits[-1].argmax())
    proposals = _assistant_candidates(
        target_weights, assistant_weights, prompt_taps, anchor
    )
    expected: list[int] = []
    accepted = 0
    history = [*prompt, anchor]
    for candidate in proposals:
        target_logits, _taps = _target_rows_and_taps(target_weights, history)
        target_token = int(target_logits[-1].argmax())
        if target_token != candidate:
            expected.append(target_token)
            break
        expected.append(candidate)
        history.append(candidate)
        accepted += 1
    else:
        bonus_logits, _taps = _target_rows_and_taps(target_weights, history)
        expected.append(int(bonus_logits[-1].argmax()))

    environment = os.environ.copy()
    environment["NFN_TEST_GLIMMER_CUDA_RUNTIME"] = str(fake_glimmer_cuda[0])
    environment["NFN_TEST_GLIMMER_TILE_OPS"] = str(fake_glimmer_cuda[1])
    completed = subprocess.run(
        [
            str(glimmer_dflash_probe),
            str(target_path),
            target_digest,
            str(assistant_path),
            assistant_digest,
            ",".join(map(str, prompt)),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    lines = completed.stdout.splitlines()
    assert [int(value) for value in lines[0].split(",")] == [anchor]
    assert [int(value) for value in lines[1].split(",")] == expected
    counters = [int(value) for value in lines[2].split(",")]
    assert counters[:3] == [3, accepted, 3 - accepted]
    assert counters[4] == 1
    assert int(lines[3].split(",")[-1]) > 0


def test_resident_dflash_loads_exact_packed_gguf_without_dense_expansion(
    glimmer_dflash_gguf_load_probe: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "muse-glimmer-text.bf16"
    with target.open("wb") as stream:
        stream.truncate(glimmer_checkpoint.MAIN_TEXT_PAYLOAD_BYTES)
    assistant = tmp_path / gguf.DFLASH_KQUANT_PROFILE["filename"]
    resident_bytes = _write_sparse_canonical_dflash_gguf(assistant)
    assert assistant.stat().st_blocks * 512 < assistant.stat().st_size // 8

    completed = subprocess.run(
        [str(glimmer_dflash_gguf_load_probe), str(target), str(assistant)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    parameter_count, reported_bytes, first_layer, last_layer = map(
        int, completed.stdout.strip().split(",")
    )
    assert parameter_count == glimmer_checkpoint.ASSISTANT_PARAMETER_COUNT
    assert reported_bytes == resident_bytes
    assert reported_bytes < assistant.stat().st_size
    assert (first_layer, last_layer) == (1, 49)


def test_resident_dflash_cuda_pins_neox_half_split_rope() -> None:
    source = (
        ROOT
        / "neuralfn"
        / "csrc"
        / "native_gpt2"
        / "resident_glimmer_assistant.cpp"
    ).read_text(encoding="utf-8")
    assert "cuda.gguf_interleaved = false;" in source
    assert (
        "cuda.gguf_interleaved = config_.container == WeightContainer::GgufKQuant;"
        not in source
    )
