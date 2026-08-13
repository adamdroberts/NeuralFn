"""Fail-closed GGUF v3 inspection and K-quant reference decoding.

Only the tensor encodings required by the pinned Muse Glimmer artifacts are
accepted: F32, BF16, Q4_K, Q5_K, and Q6_K.  Unknown encodings never fall
through to a dense pointer interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, BinaryIO, Mapping, Sequence
import uuid

from .native_chat import (
    MUSE_GLIMMER_ATEM_TEMPLATE_SHA256,
    MUSE_GLIMMER_SPECIAL_TOKEN_IDS,
    MUSE_GLIMMER_TOKENIZER_SHA256,
)


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32
GGUF_MAX_HEADER_BYTES = 64 * 1024 * 1024
GGUF_MAX_STRING_BYTES = 32 * 1024 * 1024
GGUF_MAX_ARRAY_ELEMENTS = 1_000_000

GGML_TYPE_F32 = 0
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_BF16 = 30

MUSE_GLIMMER_GGUF_TOKENIZER_METADATA_SHA256 = (
    "40f03520b77ade69aff33e980b24612b98f79008c5f79784583cbf0153d03e2b"
)
MUSE_GLIMMER_GGUF_REVISION = "43c7eadd41352a299ea8e0a36b3157978dd63596"

GGML_TYPE_LAYOUTS = {
    GGML_TYPE_F32: ("F32", 1, 4),
    GGML_TYPE_Q4_K: ("Q4_K", 256, 144),
    GGML_TYPE_Q5_K: ("Q5_K", 256, 176),
    GGML_TYPE_Q6_K: ("Q6_K", 256, 210),
    GGML_TYPE_BF16: ("BF16", 1, 2),
}

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

KQUANT_PROFILES = {
    "k-quant-17gb": {
        "filename": "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
        "nbytes": 16_756_683_904,
        "sha256": "4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e",
        "tensor_table_sha256": "44cd4374970ce14d5f944a1e0627831615a482874b656f0eb7bb5753817cc8fa",
        "inventory": {"F32": 313, "Q4_K": 365, "Q5_K": 1, "Q6_K": 52},
    },
    "k-quant-dynamic": {
        "filename": "Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf",
        "nbytes": 19_653_960_832,
        "sha256": "ac7023d6a4c704eb9af54ab53e476a66b7f5b6c0ef2fc4a8dde5253c291a6c38",
        "tensor_table_sha256": "3d018b78e0dd073122086ca2ce0e6b5abd1714b7f3afb5dd8b78ee59c4f90aad",
        "inventory": {"F32": 313, "Q4_K": 51, "Q5_K": 130, "Q6_K": 237},
    },
}

DFLASH_KQUANT_PROFILE = {
    "filename": "dflash-Muse-Glimmer-30B-Q4_K_M.gguf",
    "nbytes": 1_631_208_128,
    "sha256": "b2e808bf656086fe86bd0d0bd990f01d33e377537a07c02d45371517c8b264ef",
    "tensor_table_sha256": "cfb4c50ed5e0e760f5601b84d5ddbbce03d08fedcee41c4c9ed10c298def0b30",
    "tokenizer_metadata_sha256": "f7b318c5ce1048bc5775efc7f7dbbd92c50e02177b4dba5763ceaa0f5874f7de",
    "inventory": {"F32": 22, "Q4_K": 26, "Q6_K": 10},
}

MMPROJ_KQUANT_PROFILE = {
    "filename": "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf",
    "nbytes": 1_400_328_928,
    "sha256": "f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6",
    "tensor_table_sha256": "47a880e1fde666694bf591879b3e8bbab6cff1a72ba883d959d3bf3cae4bea78",
    "inventory": {"F32": 506, "Q4_K": 200, "Q6_K": 100, "BF16": 3},
}


class GGUFError(ValueError):
    """A malformed, unsupported, or unauthenticated GGUF artifact."""


@dataclass(frozen=True, slots=True)
class GGUFArray:
    element_type: int
    values: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class GGUFTensorDescriptor:
    name: str
    native_name: str
    dimensions: tuple[int, ...]
    shape: tuple[int, ...]
    ggml_type: int
    encoding: str
    block_elements: int
    block_bytes: int
    row_elements: int
    rows: int
    row_stride_bytes: int
    relative_offset: int
    absolute_offset: int
    nbytes: int


@dataclass(frozen=True, slots=True)
class GGUFModel:
    path: Path
    version: int
    metadata: Mapping[str, Any]
    tensors: tuple[GGUFTensorDescriptor, ...]
    alignment: int
    data_offset: int
    file_nbytes: int
    file_sha256: str | None
    tensor_table_sha256: str
    tokenizer_metadata_sha256: str

    @property
    def encoding_inventory(self) -> dict[str, int]:
        inventory: dict[str, int] = {}
        for tensor in self.tensors:
            inventory[tensor.encoding] = inventory.get(tensor.encoding, 0) + 1
        return inventory


class _GGUFReader:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.consumed = 0

    def read(self, nbytes: int) -> bytes:
        if nbytes < 0 or self.consumed + nbytes > GGUF_MAX_HEADER_BYTES:
            raise GGUFError("GGUF metadata/tensor table exceeds the 64 MiB bound")
        value = self.stream.read(nbytes)
        if len(value) != nbytes:
            raise GGUFError("GGUF header is truncated")
        self.consumed += nbytes
        return value

    def unpack(self, fmt: str) -> Any:
        value = struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))
        return value[0] if len(value) == 1 else value

    def string(self, *, key: bool = False) -> bytes:
        length = self.unpack("Q")
        if length > GGUF_MAX_STRING_BYTES:
            raise GGUFError("GGUF string exceeds the 32 MiB bound")
        value = self.read(length)
        if key:
            try:
                value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise GGUFError("GGUF metadata/tensor name is not UTF-8") from exc
        return value


def _read_value(reader: _GGUFReader, value_type: int, *, depth: int = 0) -> Any:
    if depth > 1:
        raise GGUFError("Nested GGUF arrays are unsupported")
    scalar_formats = {
        GGUF_TYPE_UINT8: "B",
        GGUF_TYPE_INT8: "b",
        GGUF_TYPE_UINT16: "H",
        GGUF_TYPE_INT16: "h",
        GGUF_TYPE_UINT32: "I",
        GGUF_TYPE_INT32: "i",
        GGUF_TYPE_FLOAT32: "f",
        GGUF_TYPE_BOOL: "?",
        GGUF_TYPE_UINT64: "Q",
        GGUF_TYPE_INT64: "q",
        GGUF_TYPE_FLOAT64: "d",
    }
    if value_type in scalar_formats:
        value = reader.unpack(scalar_formats[value_type])
        if isinstance(value, float) and not math.isfinite(value):
            raise GGUFError("GGUF metadata contains a non-finite number")
        return value
    if value_type == GGUF_TYPE_STRING:
        return reader.string()
    if value_type == GGUF_TYPE_ARRAY:
        element_type = reader.unpack("I")
        if element_type == GGUF_TYPE_ARRAY or element_type not in {
            GGUF_TYPE_UINT8,
            GGUF_TYPE_INT8,
            GGUF_TYPE_UINT16,
            GGUF_TYPE_INT16,
            GGUF_TYPE_UINT32,
            GGUF_TYPE_INT32,
            GGUF_TYPE_FLOAT32,
            GGUF_TYPE_BOOL,
            GGUF_TYPE_STRING,
            GGUF_TYPE_UINT64,
            GGUF_TYPE_INT64,
            GGUF_TYPE_FLOAT64,
        }:
            raise GGUFError(f"Unsupported GGUF array element type {element_type}")
        length = reader.unpack("Q")
        if length > GGUF_MAX_ARRAY_ELEMENTS:
            raise GGUFError("GGUF array exceeds the 1,000,000-element bound")
        return GGUFArray(
            element_type=element_type,
            values=tuple(_read_value(reader, element_type, depth=depth + 1) for _ in range(length)),
        )
    raise GGUFError(f"Unsupported GGUF metadata value type {value_type}")


def _decode_key(raw: bytes, *, label: str) -> str:
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GGUFError(f"GGUF {label} is not UTF-8") from exc
    if not value or "\x00" in value:
        raise GGUFError(f"GGUF {label} is empty or contains NUL")
    return value


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _native_tensor_name(name: str) -> str:
    if name == "token_embd.weight":
        return "text.embedding.weight"
    if name == "output.weight":
        return "text.lm_head.weight"
    if name == "output_norm.weight":
        return "text.final_norm.weight"
    if name.startswith("blk."):
        parts = name.split(".")
        if len(parts) == 4 and parts[0] == "blk" and parts[1].isdigit() and parts[3] == "weight":
            layer = int(parts[1])
            mapping = {
                "attn_norm": "input_norm.centered_weight",
                "post_attention_norm": "post_attention_norm.centered_weight",
                "ffn_norm": "pre_feedforward_norm.centered_weight",
                "post_ffw_norm": "post_feedforward_norm.centered_weight",
                "ffn_down": "mlp.down.weight",
                "ffn_gate": "mlp.gate.weight",
                "ffn_up": "mlp.up.weight",
                "attn_gate": "attention.gate.weight",
                "attn_k": "attention.k.weight",
                "attn_output": "attention.o.weight",
                "attn_q": "attention.q.weight",
                "attn_v": "attention.v.weight",
                "attn_q_norm": "attention.q_norm.constant_weight",
                "attn_k_norm": "attention.k_norm.constant_weight",
            }
            if 0 <= layer < 52 and parts[2] in mapping:
                return f"text.layers.{layer}.{mapping[parts[2]]}"
    raise GGUFError(f"Unexpected Muse Glimmer GGUF tensor name {name!r}")


def _tensor_contract(name: str) -> tuple[tuple[int, ...], str]:
    if name == "token_embd.weight":
        return (202_048, 6_656), _native_tensor_name(name)
    if name == "output.weight":
        return (202_048, 6_656), _native_tensor_name(name)
    if name == "output_norm.weight":
        return (6_656,), _native_tensor_name(name)
    native = _native_tensor_name(name)
    suffix = name.split(".")[2]
    shapes = {
        "attn_norm": (6_656,),
        "post_attention_norm": (6_656,),
        "ffn_norm": (6_656,),
        "post_ffw_norm": (6_656,),
        "ffn_down": (6_656, 19_968),
        "ffn_gate": (19_968, 6_656),
        "ffn_up": (19_968, 6_656),
        "attn_gate": (4_096, 6_656),
        "attn_k": (256, 6_656),
        "attn_output": (6_656, 4_096),
        "attn_q": (4_096, 6_656),
        "attn_v": (256, 6_656),
        "attn_q_norm": (128,),
        "attn_k_norm": (128,),
    }
    return shapes[suffix], native


def expected_muse_glimmer_gguf_tensor_names() -> tuple[str, ...]:
    values = ["token_embd.weight"]
    suffixes = (
        "attn_norm",
        "ffn_down",
        "ffn_gate",
        "ffn_up",
        "post_attention_norm",
        "post_ffw_norm",
        "ffn_norm",
        "attn_gate",
        "attn_k",
        "attn_output",
        "attn_q_norm",
        "attn_k_norm",
        "attn_q",
        "attn_v",
    )
    for layer in range(52):
        values.extend(f"blk.{layer}.{suffix}.weight" for suffix in suffixes)
    values.extend(("output.weight", "output_norm.weight"))
    return tuple(values)


def _dflash_native_tensor_name(name: str) -> str:
    if name == "fc.weight":
        return "assistant.context_projection.weight"
    if name == "enc.output_norm.weight":
        return "assistant.context_norm.weight"
    if name == "output_norm.weight":
        return "assistant.final_norm.weight"
    parts = name.split(".")
    if len(parts) == 4 and parts[0] == "blk" and parts[1].isdigit() and parts[3] == "weight":
        layer = int(parts[1])
        mapping = {
            "attn_norm": "input_norm.weight",
            "ffn_down": "mlp.down.weight",
            "ffn_gate": "mlp.gate.weight",
            "ffn_up": "mlp.up.weight",
            "ffn_norm": "post_attention_norm.weight",
            "attn_k_norm": "attention.k_norm.weight",
            "attn_k": "attention.k.weight",
            "attn_output": "attention.o.weight",
            "attn_q_norm": "attention.q_norm.weight",
            "attn_q": "attention.q.weight",
            "attn_v": "attention.v.weight",
        }
        if 0 <= layer < 5 and parts[2] in mapping:
            return f"assistant.layers.{layer}.{mapping[parts[2]]}"
    raise GGUFError(f"Unexpected Muse Glimmer DFlash GGUF tensor name {name!r}")


def _dflash_tensor_contract(name: str) -> tuple[tuple[int, ...], str]:
    fixed = {
        "fc.weight": (6_656, 33_280),
        "enc.output_norm.weight": (6_656,),
        "output_norm.weight": (6_656,),
    }
    if name in fixed:
        return fixed[name], _dflash_native_tensor_name(name)
    suffix = name.split(".")[2]
    shapes = {
        "attn_norm": (6_656,),
        "ffn_down": (6_656, 19_968),
        "ffn_gate": (19_968, 6_656),
        "ffn_up": (19_968, 6_656),
        "ffn_norm": (6_656,),
        "attn_k_norm": (128,),
        "attn_k": (1_024, 6_656),
        "attn_output": (6_656, 4_096),
        "attn_q_norm": (128,),
        "attn_q": (4_096, 6_656),
        "attn_v": (1_024, 6_656),
    }
    return shapes[suffix], _dflash_native_tensor_name(name)


def expected_muse_glimmer_dflash_gguf_tensor_names() -> tuple[str, ...]:
    values = ["fc.weight", "enc.output_norm.weight"]
    suffixes = (
        "attn_norm",
        "ffn_down",
        "ffn_gate",
        "ffn_up",
        "ffn_norm",
        "attn_k_norm",
        "attn_k",
        "attn_output",
        "attn_q_norm",
        "attn_q",
        "attn_v",
    )
    for layer in range(5):
        values.extend(f"blk.{layer}.{suffix}.weight" for suffix in suffixes)
    values.append("output_norm.weight")
    return tuple(values)


def _mmproj_native_tensor_name(name: str) -> str:
    fixed = {
        "v.patch_embd.weight": "vision.patch_embedding.weight",
        "v.position_embd.weight": "vision.position_embedding.weight",
        "v.pre_ln.weight": "vision.pre_norm.weight",
        "v.pre_ln.bias": "vision.pre_norm.bias",
        "v.post_ln.weight": "vision.post_norm.weight",
        "v.post_ln.bias": "vision.post_norm.bias",
        "mm.0.weight": "vision.adapter.fc1.weight",
        "mm.1.weight": "vision.adapter.fc2.weight",
        "mm.2.weight": "vision.projection.weight",
    }
    if name in fixed:
        return fixed[name]
    parts = name.split(".")
    if (
        len(parts) == 5
        and parts[0] == "v"
        and parts[1] == "blk"
        and parts[2].isdigit()
        and 0 <= int(parts[2]) < 50
        and parts[4] in {"weight", "bias"}
    ):
        mapping = {
            "attn_q": "attention.q",
            "attn_k": "attention.k",
            "attn_v": "attention.v",
            "attn_out": "attention.proj",
            "ln1": "norm1",
            "ln2": "norm2",
            "ffn_up": "mlp.fc1",
            "ffn_down": "mlp.fc2",
        }
        if parts[3] in mapping:
            return (
                f"vision.layers.{int(parts[2])}.{mapping[parts[3]]}.{parts[4]}"
            )
    raise GGUFError(f"Unexpected Muse Glimmer mmproj tensor name {name!r}")


def _mmproj_tensor_contract(name: str) -> tuple[tuple[int, ...], str, int]:
    fixed: dict[str, tuple[tuple[int, ...], int]] = {
        # The official converter sums the two temporal patch slabs before its
        # conv2d-compatible GGUF export, so this packed companion consumes 588
        # values per spatial patch rather than the BF16 Transformer's 1176.
        "v.patch_embd.weight": ((1_536, 3, 14, 14), GGML_TYPE_F32),
        "v.position_embd.weight": ((1_024, 1_536), GGML_TYPE_F32),
        "v.pre_ln.weight": ((1_536,), GGML_TYPE_F32),
        "v.pre_ln.bias": ((1_536,), GGML_TYPE_F32),
        "v.post_ln.weight": ((1_536,), GGML_TYPE_F32),
        "v.post_ln.bias": ((1_536,), GGML_TYPE_F32),
        "mm.0.weight": ((4_096, 6_144), GGML_TYPE_BF16),
        "mm.1.weight": ((4_096, 4_096), GGML_TYPE_BF16),
        "mm.2.weight": ((6_656, 4_096), GGML_TYPE_BF16),
    }
    if name in fixed:
        shape, ggml_type = fixed[name]
        return shape, _mmproj_native_tensor_name(name), ggml_type
    parts = name.split(".")
    suffix, parameter = parts[3], parts[4]
    if parameter == "bias":
        width = 8_960 if suffix == "ffn_up" else 1_536
        return (width,), _mmproj_native_tensor_name(name), GGML_TYPE_F32
    if suffix in {"ln1", "ln2"}:
        return (1_536,), _mmproj_native_tensor_name(name), GGML_TYPE_F32
    shapes = {
        "attn_q": (1_536, 1_536),
        "attn_k": (1_536, 1_536),
        "attn_v": (1_536, 1_536),
        "attn_out": (1_536, 1_536),
        "ffn_up": (8_960, 1_536),
        "ffn_down": (1_536, 8_960),
    }
    ggml_type = (
        GGML_TYPE_Q6_K if suffix in {"attn_v", "ffn_down"} else GGML_TYPE_Q4_K
    )
    return shapes[suffix], _mmproj_native_tensor_name(name), ggml_type


def expected_muse_glimmer_mmproj_gguf_tensor_names() -> tuple[str, ...]:
    values: list[str] = []
    suffixes = (
        "attn_k",
        "attn_out",
        "attn_q",
        "attn_v",
        "ffn_up",
        "ffn_down",
        "ln1",
        "ln2",
    )
    for layer in range(50):
        for suffix in suffixes:
            values.extend(
                (
                    f"v.blk.{layer}.{suffix}.bias",
                    f"v.blk.{layer}.{suffix}.weight",
                )
            )
    values.extend(
        (
            "v.post_ln.bias",
            "v.post_ln.weight",
            "v.pre_ln.bias",
            "v.pre_ln.weight",
            "v.patch_embd.weight",
            "v.position_embd.weight",
            "mm.0.weight",
            "mm.1.weight",
            "mm.2.weight",
        )
    )
    return tuple(values)


_MUSE_GLIMMER_METADATA_KEYS = frozenset(
    {
        "general.architecture",
        "general.type",
        "general.name",
        "general.size_label",
        "muse-glimmer.block_count",
        "muse-glimmer.context_length",
        "muse-glimmer.embedding_length",
        "muse-glimmer.feed_forward_length",
        "muse-glimmer.attention.head_count",
        "muse-glimmer.attention.head_count_kv",
        "muse-glimmer.rope.freq_base",
        "muse-glimmer.attention.layer_norm_rms_epsilon",
        "muse-glimmer.attention.key_length",
        "muse-glimmer.attention.value_length",
        "muse-glimmer.final_logit_softcapping",
        "muse-glimmer.logit_scale",
        "muse-glimmer.attention.sliding_window",
        "muse-glimmer.attention.sliding_window_pattern",
        "general.quantization_version",
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_sep_token",
        "tokenizer.ggml.eot_token_id",
        "general.file_type",
        "tokenizer.chat_template",
    }
)

_MUSE_GLIMMER_SCALARS: Mapping[str, Any] = {
    "general.architecture": b"muse-glimmer",
    "general.type": b"model",
    "general.name": b"Muse Glimmer Hf",
    "general.size_label": b"28B",
    "muse-glimmer.block_count": 52,
    "muse-glimmer.context_length": 131_072,
    "muse-glimmer.embedding_length": 6_656,
    "muse-glimmer.feed_forward_length": 19_968,
    "muse-glimmer.attention.head_count": 32,
    "muse-glimmer.attention.head_count_kv": 2,
    "muse-glimmer.attention.key_length": 128,
    "muse-glimmer.attention.value_length": 128,
    "muse-glimmer.attention.sliding_window": 2_048,
    "general.quantization_version": 2,
    "tokenizer.ggml.model": b"gpt2",
    "tokenizer.ggml.pre": b"llama4",
    "tokenizer.ggml.bos_token_id": 200_000,
    "tokenizer.ggml.eos_token_id": 200_001,
    "tokenizer.ggml.padding_token_id": 200_018,
    "tokenizer.ggml.add_bos_token": True,
    "tokenizer.ggml.add_sep_token": False,
    "tokenizer.ggml.eot_token_id": 200_008,
    "general.file_type": 15,
}

_MUSE_GLIMMER_FLOATS = {
    "muse-glimmer.rope.freq_base": 500_000.0,
    "muse-glimmer.attention.layer_norm_rms_epsilon": 1.0e-5,
    "muse-glimmer.final_logit_softcapping": 20.0,
    "muse-glimmer.logit_scale": 0.19611613513818404,
}

_MUSE_GLIMMER_GGUF_SPECIAL_TOKENS = {
    200_000: b"<|begin_of_text|>",
    200_001: b"<|end_of_text|>",
    200_007: b"<|eom|>",
    200_008: b"<|eot|>",
    200_018: b"<|finetune_right_pad|>",
    200_022: b"<|start|>",
    200_023: b"<|message|>",
    200_090: b"<|image|>",
    200_091: b"<|video|>",
    200_092: b"<|patch|>",
    201_818: b"<|reserved_special_token_1818|>",
}


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_metadata_value(digest: Any, value: Any) -> None:
    if isinstance(value, GGUFArray):
        digest.update(b"a")
        digest.update(struct.pack("<IQ", value.element_type, len(value.values)))
        for item in value.values:
            _hash_metadata_value(digest, item)
        return
    if isinstance(value, bytes):
        digest.update(b"s")
        digest.update(struct.pack("<Q", len(value)))
        digest.update(value)
        return
    if isinstance(value, bool):
        digest.update(b"b\x01" if value else b"b\x00")
        return
    if isinstance(value, int):
        digest.update(b"i")
        digest.update(struct.pack("<q", value))
        return
    if isinstance(value, float):
        digest.update(b"f")
        digest.update(struct.pack("<d", value))
        return
    raise GGUFError(f"Cannot hash unsupported GGUF metadata value {type(value).__name__}")


def _tokenizer_metadata_digest(metadata: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(key for key in metadata if key.startswith("tokenizer.")):
        encoded = key.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        _hash_metadata_value(digest, metadata[key])
    return digest.hexdigest()


def _require_array(
    metadata: Mapping[str, Any],
    key: str,
    *,
    element_type: int,
    length: int,
) -> tuple[Any, ...]:
    value = metadata.get(key)
    if not isinstance(value, GGUFArray):
        raise GGUFError(f"Muse Glimmer GGUF metadata {key!r} must be an array")
    if value.element_type != element_type or len(value.values) != length:
        raise GGUFError(
            f"Muse Glimmer GGUF metadata {key!r} must have element type "
            f"{element_type} and length {length}"
        )
    return value.values


def _validate_muse_glimmer_metadata(metadata: Mapping[str, Any]) -> str:
    keys = frozenset(metadata)
    if keys != _MUSE_GLIMMER_METADATA_KEYS:
        missing = sorted(_MUSE_GLIMMER_METADATA_KEYS - keys)
        unexpected = sorted(keys - _MUSE_GLIMMER_METADATA_KEYS)
        raise GGUFError(
            "Muse Glimmer GGUF metadata allowlist mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key, expected in _MUSE_GLIMMER_SCALARS.items():
        if metadata[key] != expected:
            raise GGUFError(
                f"Muse Glimmer GGUF metadata {key!r} is {metadata[key]!r}; "
                f"expected {expected!r}"
            )
    for key, expected in _MUSE_GLIMMER_FLOATS.items():
        actual = metadata[key]
        if not isinstance(actual, float) or not math.isclose(
            actual, expected, rel_tol=1.0e-6, abs_tol=1.0e-9
        ):
            raise GGUFError(
                f"Muse Glimmer GGUF metadata {key!r} is {actual!r}; expected {expected!r}"
            )

    pattern = _require_array(
        metadata,
        "muse-glimmer.attention.sliding_window_pattern",
        element_type=GGUF_TYPE_BOOL,
        length=52,
    )
    if pattern != tuple((layer % 4) != 3 for layer in range(52)):
        raise GGUFError("Muse Glimmer GGUF has the wrong local/global attention schedule")

    tokens = _require_array(
        metadata,
        "tokenizer.ggml.tokens",
        element_type=GGUF_TYPE_STRING,
        length=202_048,
    )
    _require_array(
        metadata,
        "tokenizer.ggml.token_type",
        element_type=GGUF_TYPE_INT32,
        length=202_048,
    )
    _require_array(
        metadata,
        "tokenizer.ggml.merges",
        element_type=GGUF_TYPE_STRING,
        length=439_802,
    )
    for token_id, expected in _MUSE_GLIMMER_GGUF_SPECIAL_TOKENS.items():
        if tokens[token_id] != expected:
            raise GGUFError(
                f"Muse Glimmer GGUF token {token_id} is {tokens[token_id]!r}; "
                f"expected {expected!r}"
            )
    for name, token_id in MUSE_GLIMMER_SPECIAL_TOKEN_IDS.items():
        if name in {"eos", "eot"} and token_id not in {200_001, 200_008}:
            raise GGUFError("Internal Muse Glimmer stop-token contract is inconsistent")
    template = metadata["tokenizer.chat_template"]
    if not isinstance(template, bytes):
        raise GGUFError("Muse Glimmer GGUF tokenizer.chat_template must be a string")
    template_sha = hashlib.sha256(template).hexdigest()
    if template_sha != MUSE_GLIMMER_ATEM_TEMPLATE_SHA256:
        raise GGUFError(
            "Muse Glimmer GGUF embeds an unreviewed ATEM template: "
            f"SHA-256 {template_sha}"
        )
    tokenizer_digest = _tokenizer_metadata_digest(metadata)
    if tokenizer_digest != MUSE_GLIMMER_GGUF_TOKENIZER_METADATA_SHA256:
        raise GGUFError(
            "Muse Glimmer GGUF tokenizer metadata SHA-256 mismatch: "
            f"{tokenizer_digest}"
        )
    return tokenizer_digest


_DFLASH_METADATA_KEYS = frozenset(
    {
        "general.architecture",
        "general.type",
        "general.name",
        "general.size_label",
        "dflash.block_count",
        "dflash.context_length",
        "dflash.embedding_length",
        "dflash.feed_forward_length",
        "dflash.attention.head_count",
        "dflash.attention.head_count_kv",
        "dflash.rope.freq_base",
        "dflash.attention.layer_norm_rms_epsilon",
        "dflash.attention.key_length",
        "dflash.attention.value_length",
        "dflash.block_size",
        "dflash.target_layers",
        "dflash.attention.sliding_window",
        "dflash.attention.sliding_window_pattern",
        "general.quantization_version",
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_sep_token",
        "tokenizer.ggml.eot_token_id",
        "tokenizer.ggml.mask_token_id",
        "general.file_type",
        "tokenizer.chat_template",
    }
)


def _validate_muse_glimmer_dflash_metadata(metadata: Mapping[str, Any]) -> str:
    if frozenset(metadata) != _DFLASH_METADATA_KEYS:
        raise GGUFError("Muse Glimmer DFlash GGUF metadata allowlist mismatch")
    expected_scalars: Mapping[str, Any] = {
        "general.architecture": b"dflash",
        "general.type": b"model",
        "general.name": b"Hf_Museglimmer",
        "general.size_label": b"2.6B",
        "dflash.block_count": 5,
        "dflash.context_length": 131_072,
        "dflash.embedding_length": 6_656,
        "dflash.feed_forward_length": 19_968,
        "dflash.attention.head_count": 32,
        "dflash.attention.head_count_kv": 8,
        "dflash.attention.key_length": 128,
        "dflash.attention.value_length": 128,
        "dflash.block_size": 16,
        "dflash.attention.sliding_window": 2_048,
        "general.quantization_version": 2,
        "tokenizer.ggml.model": b"gpt2",
        "tokenizer.ggml.pre": b"llama4",
        "tokenizer.ggml.bos_token_id": 200_000,
        "tokenizer.ggml.eos_token_id": 200_001,
        "tokenizer.ggml.padding_token_id": 200_018,
        "tokenizer.ggml.add_bos_token": True,
        "tokenizer.ggml.add_sep_token": False,
        "tokenizer.ggml.eot_token_id": 200_008,
        "tokenizer.ggml.mask_token_id": 201_818,
        "general.file_type": 15,
    }
    for key, expected in expected_scalars.items():
        if metadata[key] != expected:
            raise GGUFError(
                f"Muse Glimmer DFlash GGUF metadata {key!r} is "
                f"{metadata[key]!r}; expected {expected!r}"
            )
    for key, expected in {
        "dflash.rope.freq_base": 500_000.0,
        "dflash.attention.layer_norm_rms_epsilon": 1.0e-5,
    }.items():
        value = metadata[key]
        if not isinstance(value, float) or not math.isclose(
            value, expected, rel_tol=1.0e-6, abs_tol=1.0e-9
        ):
            raise GGUFError(f"Muse Glimmer DFlash GGUF metadata {key!r} is invalid")
    target_layers = _require_array(
        metadata,
        "dflash.target_layers",
        element_type=GGUF_TYPE_INT32,
        length=5,
    )
    if target_layers != (2, 14, 26, 38, 50):
        raise GGUFError("Muse Glimmer DFlash target-layer metadata is not canonical")
    # GGUF stores one-based human layer numbers. The runtime uses zero-based
    # target decoder indices and performs this normalization exactly once.
    if tuple(value - 1 for value in target_layers) != (1, 13, 25, 37, 49):
        raise GGUFError("Muse Glimmer DFlash target-layer normalization failed")
    if _require_array(
        metadata,
        "dflash.attention.sliding_window_pattern",
        element_type=GGUF_TYPE_BOOL,
        length=5,
    ) != (True,) * 5:
        raise GGUFError("Muse Glimmer DFlash sliding-window pattern is invalid")
    tokens = _require_array(
        metadata,
        "tokenizer.ggml.tokens",
        element_type=GGUF_TYPE_STRING,
        length=202_048,
    )
    _require_array(
        metadata,
        "tokenizer.ggml.token_type",
        element_type=GGUF_TYPE_INT32,
        length=202_048,
    )
    _require_array(
        metadata,
        "tokenizer.ggml.merges",
        element_type=GGUF_TYPE_STRING,
        length=439_802,
    )
    for token_id, expected in _MUSE_GLIMMER_GGUF_SPECIAL_TOKENS.items():
        if tokens[token_id] != expected:
            raise GGUFError(f"Muse Glimmer DFlash token {token_id} is not canonical")
    template = metadata["tokenizer.chat_template"]
    if not isinstance(template, bytes) or hashlib.sha256(template).hexdigest() != (
        MUSE_GLIMMER_ATEM_TEMPLATE_SHA256
    ):
        raise GGUFError("Muse Glimmer DFlash embeds an unreviewed ATEM template")
    tokenizer_digest = _tokenizer_metadata_digest(metadata)
    if tokenizer_digest != DFLASH_KQUANT_PROFILE["tokenizer_metadata_sha256"]:
        raise GGUFError("Muse Glimmer DFlash tokenizer metadata SHA-256 mismatch")
    return tokenizer_digest


_MMPROJ_METADATA_KEYS = frozenset(
    {
        "general.architecture",
        "general.type",
        "general.name",
        "general.size_label",
        "clip.has_vision_encoder",
        "clip.vision.projection_dim",
        "clip.vision.image_size",
        "clip.vision.patch_size",
        "clip.vision.embedding_length",
        "clip.vision.feed_forward_length",
        "clip.vision.block_count",
        "clip.vision.attention.head_count",
        "clip.vision.image_mean",
        "clip.vision.image_std",
        "clip.projector_type",
        "clip.vision.attention.layer_norm_epsilon",
        "clip.vision.spatial_merge_size",
        "general.quantization_version",
        "general.file_type",
    }
)


def _validate_muse_glimmer_mmproj_metadata(metadata: Mapping[str, Any]) -> str:
    if frozenset(metadata) != _MMPROJ_METADATA_KEYS:
        missing = sorted(_MMPROJ_METADATA_KEYS - frozenset(metadata))
        unexpected = sorted(frozenset(metadata) - _MMPROJ_METADATA_KEYS)
        raise GGUFError(
            "Muse Glimmer mmproj metadata allowlist mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    expected_scalars: Mapping[str, Any] = {
        "general.architecture": b"clip",
        "general.type": b"mmproj",
        "general.name": b"Muse Glimmer Hf",
        "general.size_label": b"1.9B",
        "clip.has_vision_encoder": True,
        "clip.vision.projection_dim": 6_656,
        "clip.vision.image_size": 896,
        "clip.vision.patch_size": 14,
        "clip.vision.embedding_length": 1_536,
        "clip.vision.feed_forward_length": 8_960,
        "clip.vision.block_count": 50,
        "clip.vision.attention.head_count": 16,
        "clip.projector_type": b"muse-glimmer",
        "clip.vision.spatial_merge_size": 2,
        "general.quantization_version": 2,
        "general.file_type": 15,
    }
    for key, expected in expected_scalars.items():
        if metadata[key] != expected:
            raise GGUFError(
                f"Muse Glimmer mmproj metadata {key!r} is {metadata[key]!r}; "
                f"expected {expected!r}"
            )
    epsilon = metadata["clip.vision.attention.layer_norm_epsilon"]
    if not isinstance(epsilon, float) or not math.isclose(
        epsilon, 1.0e-5, rel_tol=1.0e-6, abs_tol=1.0e-9
    ):
        raise GGUFError("Muse Glimmer mmproj LayerNorm epsilon is invalid")
    for key in ("clip.vision.image_mean", "clip.vision.image_std"):
        values = _require_array(
            metadata, key, element_type=GGUF_TYPE_FLOAT32, length=3
        )
        if any(
            not isinstance(value, float)
            or not math.isclose(value, 0.5, rel_tol=0.0, abs_tol=1.0e-7)
            for value in values
        ):
            raise GGUFError(f"Muse Glimmer mmproj metadata {key!r} is invalid")
    digest = hashlib.sha256()
    for key in sorted(metadata):
        encoded = key.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        _hash_metadata_value(digest, metadata[key])
    return digest.hexdigest()


def _parse_gguf(
    path: Path,
    *,
    require_complete_file: bool,
    declared_file_nbytes: int | None = None,
    compute_file_sha256: bool,
) -> GGUFModel:
    path = Path(path).resolve()
    if not path.is_file():
        raise GGUFError(f"GGUF artifact does not exist: {path}")
    actual_file_nbytes = path.stat().st_size
    file_nbytes = actual_file_nbytes if declared_file_nbytes is None else int(declared_file_nbytes)
    if file_nbytes < actual_file_nbytes:
        raise GGUFError("Declared GGUF byte length is smaller than the supplied file")

    with path.open("rb") as stream:
        reader = _GGUFReader(stream)
        if reader.read(4) != GGUF_MAGIC:
            raise GGUFError("GGUF magic is invalid")
        version = reader.unpack("I")
        if version != GGUF_VERSION:
            raise GGUFError(f"Unsupported GGUF version {version}; expected version 3")
        tensor_count = reader.unpack("Q")
        metadata_count = reader.unpack("Q")
        if tensor_count != 731:
            raise GGUFError(f"Muse Glimmer GGUF must contain exactly 731 tensors, got {tensor_count}")
        if metadata_count != 32:
            raise GGUFError(
                f"Muse Glimmer GGUF must contain exactly 32 metadata entries, got {metadata_count}"
            )

        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = _decode_key(reader.string(key=True), label="metadata key")
            if key in metadata:
                raise GGUFError(f"Duplicate GGUF metadata key {key!r}")
            metadata[key] = _read_value(reader, reader.unpack("I"))
        tokenizer_digest = _validate_muse_glimmer_metadata(metadata)
        alignment_value = metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
        if not isinstance(alignment_value, int) or alignment_value < 1:
            raise GGUFError("GGUF alignment must be a positive integer")
        alignment = int(alignment_value)
        if alignment & (alignment - 1) or alignment > 4096:
            raise GGUFError("GGUF alignment must be a power of two no larger than 4096")

        tensor_table_start = reader.consumed
        raw_tensors: list[tuple[str, tuple[int, ...], int, int]] = []
        names: set[str] = set()
        for _ in range(tensor_count):
            name = _decode_key(reader.string(key=True), label="tensor name")
            if name in names:
                raise GGUFError(f"Duplicate GGUF tensor name {name!r}")
            names.add(name)
            dimension_count = reader.unpack("I")
            if not 1 <= dimension_count <= 4:
                raise GGUFError(
                    f"GGUF tensor {name!r} has unsupported rank {dimension_count}"
                )
            dimensions = tuple(reader.unpack("Q") for _ in range(dimension_count))
            if any(dimension <= 0 for dimension in dimensions):
                raise GGUFError(f"GGUF tensor {name!r} has an empty dimension")
            ggml_type = reader.unpack("I")
            relative_offset = reader.unpack("Q")
            raw_tensors.append((name, dimensions, ggml_type, relative_offset))
        tensor_table_end = reader.consumed
        stream.seek(tensor_table_start)
        tensor_table_sha256 = hashlib.sha256(
            stream.read(tensor_table_end - tensor_table_start)
        ).hexdigest()
        data_offset = _align(tensor_table_end, alignment)
        if require_complete_file:
            stream.seek(tensor_table_end)
            padding = stream.read(data_offset - tensor_table_end)
            if len(padding) != data_offset - tensor_table_end or any(padding):
                raise GGUFError("GGUF header padding is truncated or nonzero")

    expected_names = frozenset(expected_muse_glimmer_gguf_tensor_names())
    if names != expected_names:
        raise GGUFError(
            "Muse Glimmer GGUF tensor allowlist mismatch: "
            f"missing={sorted(expected_names - names)}, unexpected={sorted(names - expected_names)}"
        )

    tensors: list[GGUFTensorDescriptor] = []
    for name, dimensions, ggml_type, relative_offset in raw_tensors:
        layout = GGML_TYPE_LAYOUTS.get(ggml_type)
        if layout is None:
            raise GGUFError(
                f"GGUF tensor {name!r} uses unsupported ggml type id {ggml_type}; "
                "unknown packed bytes are never interpreted as float weights"
            )
        encoding, block_elements, block_bytes = layout
        expected_shape, native_name = _tensor_contract(name)
        shape = tuple(reversed(dimensions))
        if shape != expected_shape:
            raise GGUFError(
                f"GGUF tensor {name!r} has logical shape {shape}; expected {expected_shape}"
            )
        row_elements = dimensions[0]
        rows = _product(dimensions[1:])
        if row_elements % block_elements:
            raise GGUFError(
                f"GGUF tensor {name!r} row width {row_elements} is not divisible by "
                f"the {encoding} block width {block_elements}"
            )
        if relative_offset % alignment:
            raise GGUFError(f"GGUF tensor {name!r} offset is not {alignment}-byte aligned")
        row_stride_bytes = (row_elements // block_elements) * block_bytes
        nbytes = rows * row_stride_bytes
        absolute_offset = data_offset + relative_offset
        if absolute_offset > file_nbytes or nbytes > file_nbytes - absolute_offset:
            raise GGUFError(f"GGUF tensor {name!r} extends beyond the declared artifact")
        tensors.append(
            GGUFTensorDescriptor(
                name=name,
                native_name=native_name,
                dimensions=dimensions,
                shape=shape,
                ggml_type=ggml_type,
                encoding=encoding,
                block_elements=block_elements,
                block_bytes=block_bytes,
                row_elements=row_elements,
                rows=rows,
                row_stride_bytes=row_stride_bytes,
                relative_offset=relative_offset,
                absolute_offset=absolute_offset,
                nbytes=nbytes,
            )
        )

    by_offset = sorted(tensors, key=lambda tensor: tensor.relative_offset)
    expected_offset = 0
    for tensor in by_offset:
        if tensor.relative_offset != expected_offset:
            raise GGUFError(
                f"GGUF tensor {tensor.name!r} starts at {tensor.relative_offset}; "
                f"expected contiguous aligned offset {expected_offset}"
            )
        expected_offset = _align(tensor.relative_offset + tensor.nbytes, alignment)
    exact_end = by_offset[-1].relative_offset + by_offset[-1].nbytes
    if data_offset + exact_end != file_nbytes:
        raise GGUFError(
            f"GGUF byte extent is {data_offset + exact_end}; declared file size is {file_nbytes}"
        )
    if require_complete_file and actual_file_nbytes != file_nbytes:
        raise GGUFError("GGUF artifact is truncated")

    file_sha256 = _sha256_file(path) if compute_file_sha256 else None
    return GGUFModel(
        path=path,
        version=version,
        metadata=metadata,
        tensors=tuple(tensors),
        alignment=alignment,
        data_offset=data_offset,
        file_nbytes=file_nbytes,
        file_sha256=file_sha256,
        tensor_table_sha256=tensor_table_sha256,
        tokenizer_metadata_sha256=tokenizer_digest,
    )


def _parse_dflash_gguf(
    path: Path,
    *,
    require_complete_file: bool,
    declared_file_nbytes: int | None = None,
    compute_file_sha256: bool,
) -> GGUFModel:
    """Parse the distinct 58-tensor official DFlash GGUF contract."""

    path = Path(path).resolve()
    if not path.is_file():
        raise GGUFError(f"DFlash GGUF artifact does not exist: {path}")
    actual_file_nbytes = path.stat().st_size
    file_nbytes = actual_file_nbytes if declared_file_nbytes is None else int(
        declared_file_nbytes
    )
    if file_nbytes < actual_file_nbytes:
        raise GGUFError("Declared DFlash GGUF byte length is smaller than the supplied file")

    with path.open("rb") as stream:
        reader = _GGUFReader(stream)
        if reader.read(4) != GGUF_MAGIC:
            raise GGUFError("DFlash GGUF magic is invalid")
        version = reader.unpack("I")
        if version != GGUF_VERSION:
            raise GGUFError(f"Unsupported DFlash GGUF version {version}")
        tensor_count = reader.unpack("Q")
        metadata_count = reader.unpack("Q")
        if tensor_count != 58 or metadata_count != 33:
            raise GGUFError(
                "Muse Glimmer DFlash GGUF must contain exactly 58 tensors and "
                f"33 metadata entries, got {tensor_count}/{metadata_count}"
            )
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = _decode_key(reader.string(key=True), label="DFlash metadata key")
            if key in metadata:
                raise GGUFError(f"Duplicate DFlash GGUF metadata key {key!r}")
            metadata[key] = _read_value(reader, reader.unpack("I"))
        tokenizer_digest = _validate_muse_glimmer_dflash_metadata(metadata)
        alignment = GGUF_DEFAULT_ALIGNMENT
        tensor_table_start = reader.consumed
        raw_tensors: list[tuple[str, tuple[int, ...], int, int]] = []
        names: set[str] = set()
        for _ in range(tensor_count):
            name = _decode_key(reader.string(key=True), label="DFlash tensor name")
            if name in names:
                raise GGUFError(f"Duplicate DFlash GGUF tensor name {name!r}")
            names.add(name)
            dimension_count = reader.unpack("I")
            if not 1 <= dimension_count <= 4:
                raise GGUFError(
                    f"DFlash GGUF tensor {name!r} has unsupported rank {dimension_count}"
                )
            dimensions = tuple(reader.unpack("Q") for _ in range(dimension_count))
            if any(dimension <= 0 for dimension in dimensions):
                raise GGUFError(f"DFlash GGUF tensor {name!r} has an empty dimension")
            ggml_type = reader.unpack("I")
            relative_offset = reader.unpack("Q")
            raw_tensors.append((name, dimensions, ggml_type, relative_offset))
        tensor_table_end = reader.consumed
        stream.seek(tensor_table_start)
        tensor_table_sha256 = hashlib.sha256(
            stream.read(tensor_table_end - tensor_table_start)
        ).hexdigest()
        data_offset = _align(tensor_table_end, alignment)
        if require_complete_file:
            stream.seek(tensor_table_end)
            padding = stream.read(data_offset - tensor_table_end)
            if len(padding) != data_offset - tensor_table_end or any(padding):
                raise GGUFError("DFlash GGUF header padding is truncated or nonzero")

    expected_names = frozenset(expected_muse_glimmer_dflash_gguf_tensor_names())
    if names != expected_names:
        raise GGUFError(
            "Muse Glimmer DFlash tensor allowlist mismatch: "
            f"missing={sorted(expected_names - names)}, unexpected={sorted(names - expected_names)}"
        )
    tensors: list[GGUFTensorDescriptor] = []
    for name, dimensions, ggml_type, relative_offset in raw_tensors:
        layout = GGML_TYPE_LAYOUTS.get(ggml_type)
        if layout is None:
            raise GGUFError(
                f"DFlash GGUF tensor {name!r} uses unsupported ggml type id {ggml_type}"
            )
        encoding, block_elements, block_bytes = layout
        expected_shape, native_name = _dflash_tensor_contract(name)
        shape = tuple(reversed(dimensions))
        if shape != expected_shape:
            raise GGUFError(
                f"DFlash GGUF tensor {name!r} has logical shape {shape}; "
                f"expected {expected_shape}"
            )
        row_elements = dimensions[0]
        rows = _product(dimensions[1:])
        if row_elements % block_elements:
            raise GGUFError(
                f"DFlash GGUF tensor {name!r} row width {row_elements} is not "
                f"divisible by the {encoding} block width {block_elements}"
            )
        if relative_offset % alignment:
            raise GGUFError("DFlash GGUF tensor offset is not 32-byte aligned")
        row_stride_bytes = (row_elements // block_elements) * block_bytes
        nbytes = rows * row_stride_bytes
        absolute_offset = data_offset + relative_offset
        if absolute_offset > file_nbytes or nbytes > file_nbytes - absolute_offset:
            raise GGUFError(f"DFlash GGUF tensor {name!r} exceeds the artifact")
        tensors.append(
            GGUFTensorDescriptor(
                name=name,
                native_name=native_name,
                dimensions=dimensions,
                shape=shape,
                ggml_type=ggml_type,
                encoding=encoding,
                block_elements=block_elements,
                block_bytes=block_bytes,
                row_elements=row_elements,
                rows=rows,
                row_stride_bytes=row_stride_bytes,
                relative_offset=relative_offset,
                absolute_offset=absolute_offset,
                nbytes=nbytes,
            )
        )
    expected_offset = 0
    by_offset = sorted(tensors, key=lambda tensor: tensor.relative_offset)
    for tensor in by_offset:
        if tensor.relative_offset != expected_offset:
            raise GGUFError(
                f"DFlash GGUF tensor {tensor.name!r} starts at "
                f"{tensor.relative_offset}; expected {expected_offset}"
            )
        expected_offset = _align(tensor.relative_offset + tensor.nbytes, alignment)
    exact_end = by_offset[-1].relative_offset + by_offset[-1].nbytes
    if data_offset + exact_end != file_nbytes:
        raise GGUFError(
            f"DFlash GGUF byte extent is {data_offset + exact_end}; expected {file_nbytes}"
        )
    if require_complete_file and actual_file_nbytes != file_nbytes:
        raise GGUFError("DFlash GGUF artifact is truncated")
    return GGUFModel(
        path=path,
        version=version,
        metadata=metadata,
        tensors=tuple(tensors),
        alignment=alignment,
        data_offset=data_offset,
        file_nbytes=file_nbytes,
        file_sha256=_sha256_file(path) if compute_file_sha256 else None,
        tensor_table_sha256=tensor_table_sha256,
        tokenizer_metadata_sha256=tokenizer_digest,
    )


def _parse_mmproj_gguf(
    path: Path,
    *,
    require_complete_file: bool,
    declared_file_nbytes: int | None = None,
    compute_file_sha256: bool,
) -> GGUFModel:
    """Parse the exact official 809-tensor Muse Glimmer vision companion."""

    path = Path(path).resolve()
    if not path.is_file():
        raise GGUFError(f"mmproj GGUF artifact does not exist: {path}")
    actual_file_nbytes = path.stat().st_size
    file_nbytes = actual_file_nbytes if declared_file_nbytes is None else int(
        declared_file_nbytes
    )
    if file_nbytes < actual_file_nbytes:
        raise GGUFError("Declared mmproj byte length is smaller than the supplied file")
    with path.open("rb") as stream:
        reader = _GGUFReader(stream)
        if reader.read(4) != GGUF_MAGIC:
            raise GGUFError("mmproj GGUF magic is invalid")
        version = reader.unpack("I")
        if version != GGUF_VERSION:
            raise GGUFError(f"Unsupported mmproj GGUF version {version}")
        tensor_count = reader.unpack("Q")
        metadata_count = reader.unpack("Q")
        if tensor_count != 809 or metadata_count != 19:
            raise GGUFError(
                "Muse Glimmer mmproj must contain exactly 809 tensors and 19 "
                f"metadata entries, got {tensor_count}/{metadata_count}"
            )
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = _decode_key(reader.string(key=True), label="mmproj metadata key")
            if key in metadata:
                raise GGUFError(f"Duplicate mmproj metadata key {key!r}")
            metadata[key] = _read_value(reader, reader.unpack("I"))
        metadata_digest = _validate_muse_glimmer_mmproj_metadata(metadata)
        alignment = GGUF_DEFAULT_ALIGNMENT
        tensor_table_start = reader.consumed
        raw_tensors: list[tuple[str, tuple[int, ...], int, int]] = []
        names: set[str] = set()
        for _ in range(tensor_count):
            name = _decode_key(reader.string(key=True), label="mmproj tensor name")
            if name in names:
                raise GGUFError(f"Duplicate mmproj tensor name {name!r}")
            names.add(name)
            dimension_count = reader.unpack("I")
            if not 1 <= dimension_count <= 4:
                raise GGUFError(
                    f"mmproj tensor {name!r} has unsupported rank {dimension_count}"
                )
            dimensions = tuple(reader.unpack("Q") for _ in range(dimension_count))
            if any(dimension <= 0 for dimension in dimensions):
                raise GGUFError(f"mmproj tensor {name!r} has an empty dimension")
            ggml_type = reader.unpack("I")
            relative_offset = reader.unpack("Q")
            raw_tensors.append((name, dimensions, ggml_type, relative_offset))
        tensor_table_end = reader.consumed
        stream.seek(tensor_table_start)
        tensor_table_sha256 = hashlib.sha256(
            stream.read(tensor_table_end - tensor_table_start)
        ).hexdigest()
        data_offset = _align(tensor_table_end, alignment)
        if require_complete_file:
            stream.seek(tensor_table_end)
            padding = stream.read(data_offset - tensor_table_end)
            if len(padding) != data_offset - tensor_table_end or any(padding):
                raise GGUFError("mmproj header padding is truncated or nonzero")

    expected_names = frozenset(expected_muse_glimmer_mmproj_gguf_tensor_names())
    if names != expected_names:
        raise GGUFError(
            "Muse Glimmer mmproj tensor allowlist mismatch: "
            f"missing={sorted(expected_names - names)}, "
            f"unexpected={sorted(names - expected_names)}"
        )
    tensors: list[GGUFTensorDescriptor] = []
    for name, dimensions, ggml_type, relative_offset in raw_tensors:
        layout = GGML_TYPE_LAYOUTS.get(ggml_type)
        if layout is None:
            raise GGUFError(
                f"mmproj tensor {name!r} uses unsupported ggml type id {ggml_type}"
            )
        encoding, block_elements, block_bytes = layout
        expected_shape, native_name, expected_type = _mmproj_tensor_contract(name)
        shape = tuple(reversed(dimensions))
        if shape != expected_shape:
            raise GGUFError(
                f"mmproj tensor {name!r} has logical shape {shape}; "
                f"expected {expected_shape}"
            )
        if ggml_type != expected_type:
            raise GGUFError(
                f"mmproj tensor {name!r} uses {encoding}; exact profile requires "
                f"{GGML_TYPE_LAYOUTS[expected_type][0]}"
            )
        row_elements = dimensions[0]
        rows = _product(dimensions[1:])
        if row_elements % block_elements:
            raise GGUFError(
                f"mmproj tensor {name!r} row width {row_elements} is not divisible "
                f"by the {encoding} block width {block_elements}"
            )
        if relative_offset % alignment:
            raise GGUFError("mmproj tensor offset is not 32-byte aligned")
        row_stride_bytes = (row_elements // block_elements) * block_bytes
        nbytes = rows * row_stride_bytes
        absolute_offset = data_offset + relative_offset
        if absolute_offset > file_nbytes or nbytes > file_nbytes - absolute_offset:
            raise GGUFError(f"mmproj tensor {name!r} exceeds the artifact")
        tensors.append(
            GGUFTensorDescriptor(
                name=name,
                native_name=native_name,
                dimensions=dimensions,
                shape=shape,
                ggml_type=ggml_type,
                encoding=encoding,
                block_elements=block_elements,
                block_bytes=block_bytes,
                row_elements=row_elements,
                rows=rows,
                row_stride_bytes=row_stride_bytes,
                relative_offset=relative_offset,
                absolute_offset=absolute_offset,
                nbytes=nbytes,
            )
        )
    by_offset = sorted(tensors, key=lambda tensor: tensor.relative_offset)
    expected_offset = 0
    for tensor in by_offset:
        if tensor.relative_offset != expected_offset:
            raise GGUFError(
                f"mmproj tensor {tensor.name!r} starts at {tensor.relative_offset}; "
                f"expected {expected_offset}"
            )
        expected_offset = _align(tensor.relative_offset + tensor.nbytes, alignment)
    exact_end = by_offset[-1].relative_offset + by_offset[-1].nbytes
    if data_offset + exact_end != file_nbytes:
        raise GGUFError(
            f"mmproj byte extent is {data_offset + exact_end}; expected {file_nbytes}"
        )
    if require_complete_file and actual_file_nbytes != file_nbytes:
        raise GGUFError("mmproj artifact is truncated")
    return GGUFModel(
        path=path,
        version=version,
        metadata=metadata,
        tensors=tuple(tensors),
        alignment=alignment,
        data_offset=data_offset,
        file_nbytes=file_nbytes,
        file_sha256=_sha256_file(path) if compute_file_sha256 else None,
        tensor_table_sha256=tensor_table_sha256,
        tokenizer_metadata_sha256=metadata_digest,
    )


def inspect_muse_glimmer_gguf(
    path: str | Path,
    *,
    profile: str,
    verify_file_sha256: bool = True,
) -> GGUFModel:
    """Authenticate one canonical Muse Glimmer K-quant GGUF artifact.

    ``profile`` is an artifact profile, not a scalar dtype.  Dynamic is a
    mixed per-tensor Q4_K/Q5_K/Q6_K table and remains mixed at execution time.
    """

    if profile not in KQUANT_PROFILES:
        raise GGUFError(
            f"Unknown Muse Glimmer K-quant profile {profile!r}; expected one of "
            + ", ".join(sorted(KQUANT_PROFILES))
        )
    expected = KQUANT_PROFILES[profile]
    resolved = Path(path).resolve()
    if resolved.name != expected["filename"]:
        raise GGUFError(
            f"Profile {profile!r} requires canonical filename {expected['filename']!r}"
        )
    if resolved.stat().st_size != expected["nbytes"]:
        raise GGUFError(
            f"Profile {profile!r} requires {expected['nbytes']} bytes, got "
            f"{resolved.stat().st_size}"
        )
    model = _parse_gguf(
        resolved,
        require_complete_file=True,
        compute_file_sha256=verify_file_sha256,
    )
    if model.tensor_table_sha256 != expected["tensor_table_sha256"]:
        raise GGUFError(
            f"Profile {profile!r} tensor-table SHA-256 mismatch: "
            f"{model.tensor_table_sha256}"
        )
    if model.encoding_inventory != expected["inventory"]:
        raise GGUFError(
            f"Profile {profile!r} tensor inventory is {model.encoding_inventory}; "
            f"expected {expected['inventory']}"
        )
    if verify_file_sha256 and model.file_sha256 != expected["sha256"]:
        raise GGUFError(
            f"Profile {profile!r} file SHA-256 mismatch: {model.file_sha256}"
        )
    return model


def inspect_muse_glimmer_gguf_header_fixture(
    path: str | Path,
    *,
    profile: str,
) -> GGUFModel:
    """Inspect a bounded canonical-header fixture without authenticating payload.

    This deliberately separate test-only surface lets CI validate the 13 MiB
    metadata/table contract without pretending that a partial file is runnable.
    """

    if profile not in KQUANT_PROFILES:
        raise GGUFError(f"Unknown Muse Glimmer K-quant profile {profile!r}")
    expected = KQUANT_PROFILES[profile]
    model = _parse_gguf(
        Path(path),
        require_complete_file=False,
        declared_file_nbytes=int(expected["nbytes"]),
        compute_file_sha256=False,
    )
    if model.tensor_table_sha256 != expected["tensor_table_sha256"]:
        raise GGUFError("GGUF header fixture tensor-table SHA-256 mismatch")
    if model.encoding_inventory != expected["inventory"]:
        raise GGUFError("GGUF header fixture tensor inventory mismatch")
    return model


def inspect_muse_glimmer_dflash_gguf(
    path: str | Path,
    *,
    verify_file_sha256: bool = True,
) -> GGUFModel:
    """Authenticate the canonical packed DFlash companion."""

    expected = DFLASH_KQUANT_PROFILE
    resolved = Path(path).resolve()
    if resolved.name != expected["filename"]:
        raise GGUFError(
            f"DFlash requires canonical filename {expected['filename']!r}"
        )
    if resolved.stat().st_size != expected["nbytes"]:
        raise GGUFError(
            f"DFlash requires {expected['nbytes']} bytes, got {resolved.stat().st_size}"
        )
    model = _parse_dflash_gguf(
        resolved,
        require_complete_file=True,
        compute_file_sha256=verify_file_sha256,
    )
    if model.tensor_table_sha256 != expected["tensor_table_sha256"]:
        raise GGUFError("DFlash GGUF tensor-table SHA-256 mismatch")
    if model.encoding_inventory != expected["inventory"]:
        raise GGUFError(
            f"DFlash GGUF tensor inventory is {model.encoding_inventory}; "
            f"expected {expected['inventory']}"
        )
    if verify_file_sha256 and model.file_sha256 != expected["sha256"]:
        raise GGUFError("DFlash GGUF file SHA-256 mismatch")
    return model


def inspect_muse_glimmer_dflash_gguf_header_fixture(path: str | Path) -> GGUFModel:
    """Validate a bounded canonical DFlash header without claiming execution."""

    model = _parse_dflash_gguf(
        Path(path),
        require_complete_file=False,
        declared_file_nbytes=int(DFLASH_KQUANT_PROFILE["nbytes"]),
        compute_file_sha256=False,
    )
    if model.tensor_table_sha256 != DFLASH_KQUANT_PROFILE["tensor_table_sha256"]:
        raise GGUFError("DFlash header fixture tensor-table SHA-256 mismatch")
    if model.encoding_inventory != DFLASH_KQUANT_PROFILE["inventory"]:
        raise GGUFError("DFlash header fixture tensor inventory mismatch")
    return model


def inspect_muse_glimmer_mmproj_gguf(
    path: str | Path,
    *,
    verify_file_sha256: bool = True,
) -> GGUFModel:
    """Authenticate the canonical full vision encoder/projector companion."""

    source = Path(path).resolve()
    expected = MMPROJ_KQUANT_PROFILE
    if source.name != expected["filename"]:
        raise GGUFError(
            f"Expected canonical mmproj filename {expected['filename']!r}, got {source.name!r}"
        )
    model = _parse_mmproj_gguf(
        source,
        require_complete_file=True,
        compute_file_sha256=verify_file_sha256,
    )
    if model.file_nbytes != expected["nbytes"]:
        raise GGUFError("Muse Glimmer mmproj file byte length mismatch")
    if model.tensor_table_sha256 != expected["tensor_table_sha256"]:
        raise GGUFError("Muse Glimmer mmproj tensor-table SHA-256 mismatch")
    if model.encoding_inventory != expected["inventory"]:
        raise GGUFError(
            f"Muse Glimmer mmproj inventory is {model.encoding_inventory}; "
            f"expected {expected['inventory']}"
        )
    if verify_file_sha256 and model.file_sha256 != expected["sha256"]:
        raise GGUFError("Muse Glimmer mmproj file SHA-256 mismatch")
    return model


def inspect_muse_glimmer_mmproj_gguf_header_fixture(path: str | Path) -> GGUFModel:
    """Validate a bounded canonical mmproj header without claiming full bytes."""

    model = _parse_mmproj_gguf(
        Path(path),
        require_complete_file=False,
        declared_file_nbytes=int(MMPROJ_KQUANT_PROFILE["nbytes"]),
        compute_file_sha256=False,
    )
    if model.tensor_table_sha256 != MMPROJ_KQUANT_PROFILE["tensor_table_sha256"]:
        raise GGUFError("mmproj header fixture tensor-table SHA-256 mismatch")
    if model.encoding_inventory != MMPROJ_KQUANT_PROFILE["inventory"]:
        raise GGUFError("mmproj header fixture tensor inventory mismatch")
    return model


def _half(raw: bytes) -> float:
    return float(struct.unpack("<e", raw)[0])


def _bf16(raw: bytes) -> float:
    return float(struct.unpack("<f", b"\x00\x00" + raw)[0])


def _scale_min(scales: bytes, index: int) -> tuple[int, int]:
    if index < 4:
        return scales[index] & 63, scales[index + 4] & 63
    return (
        (scales[index + 4] & 0x0F) | ((scales[index - 4] >> 6) << 4),
        (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
    )


def _dequantize_q4_k_block(block: bytes) -> tuple[float, ...]:
    d = _half(block[0:2])
    dmin = _half(block[2:4])
    scales = block[4:16]
    quants = block[16:144]
    values: list[float] = []
    scale_index = 0
    for group in range(4):
        scale1, min1 = _scale_min(scales, scale_index)
        scale2, min2 = _scale_min(scales, scale_index + 1)
        q = quants[group * 32 : (group + 1) * 32]
        values.extend(d * scale1 * (value & 0x0F) - dmin * min1 for value in q)
        values.extend(d * scale2 * (value >> 4) - dmin * min2 for value in q)
        scale_index += 2
    return tuple(values)


def _dequantize_q5_k_block(block: bytes) -> tuple[float, ...]:
    d = _half(block[0:2])
    dmin = _half(block[2:4])
    scales = block[4:16]
    high = block[16:48]
    low = block[48:176]
    values: list[float] = []
    scale_index = 0
    high_mask1, high_mask2 = 1, 2
    for group in range(4):
        scale1, min1 = _scale_min(scales, scale_index)
        scale2, min2 = _scale_min(scales, scale_index + 1)
        q = low[group * 32 : (group + 1) * 32]
        values.extend(
            d * scale1 * ((value & 0x0F) + (16 if high[index] & high_mask1 else 0))
            - dmin * min1
            for index, value in enumerate(q)
        )
        values.extend(
            d * scale2 * ((value >> 4) + (16 if high[index] & high_mask2 else 0))
            - dmin * min2
            for index, value in enumerate(q)
        )
        scale_index += 2
        high_mask1 <<= 2
        high_mask2 <<= 2
    return tuple(values)


def _dequantize_q6_k_block(block: bytes) -> tuple[float, ...]:
    low = block[0:128]
    high = block[128:192]
    scales = struct.unpack("<16b", block[192:208])
    d = _half(block[208:210])
    values = [0.0] * 256
    for half_index in range(2):
        ql = low[half_index * 64 : (half_index + 1) * 64]
        qh = high[half_index * 32 : (half_index + 1) * 32]
        scale = scales[half_index * 8 : (half_index + 1) * 8]
        base = half_index * 128
        for lane in range(32):
            scale_index = lane // 16
            q1 = ((ql[lane] & 0x0F) | (((qh[lane] >> 0) & 3) << 4)) - 32
            q2 = ((ql[lane + 32] & 0x0F) | (((qh[lane] >> 2) & 3) << 4)) - 32
            q3 = ((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32
            q4 = ((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4)) - 32
            values[base + lane] = d * scale[scale_index] * q1
            values[base + lane + 32] = d * scale[scale_index + 2] * q2
            values[base + lane + 64] = d * scale[scale_index + 4] * q3
            values[base + lane + 96] = d * scale[scale_index + 6] * q4
    return tuple(values)


def dequantize_ggml_blocks(data: bytes, ggml_type: int) -> tuple[float, ...]:
    """Reference-decode complete GGML blocks without whole-tensor allocation."""

    layout = GGML_TYPE_LAYOUTS.get(int(ggml_type))
    if layout is None:
        raise GGUFError(f"Unsupported ggml type id {ggml_type}")
    _encoding, block_elements, block_bytes = layout
    if len(data) % block_bytes:
        raise GGUFError(
            f"Encoded data length {len(data)} is not a multiple of block size {block_bytes}"
        )
    output: list[float] = []
    for offset in range(0, len(data), block_bytes):
        block = data[offset : offset + block_bytes]
        if ggml_type == GGML_TYPE_F32:
            output.append(float(struct.unpack("<f", block)[0]))
        elif ggml_type == GGML_TYPE_BF16:
            output.append(_bf16(block))
        elif ggml_type == GGML_TYPE_Q4_K:
            output.extend(_dequantize_q4_k_block(block))
        elif ggml_type == GGML_TYPE_Q5_K:
            output.extend(_dequantize_q5_k_block(block))
        elif ggml_type == GGML_TYPE_Q6_K:
            output.extend(_dequantize_q6_k_block(block))
        else:  # pragma: no cover - guarded by the layout lookup
            raise GGUFError(f"Unsupported ggml type id {ggml_type}")
    if len(output) != (len(data) // block_bytes) * block_elements:
        raise AssertionError("GGUF reference dequantizer produced the wrong element count")
    return tuple(output)


def kquant_checkpoint_descriptor(model: GGUFModel, *, profile: str) -> dict[str, Any]:
    """Build an authenticated descriptor for packed CPU/CUDA execution.

    The typed tensor table remains packed in device/host resident memory and is
    never expanded into a second whole-model buffer.
    """

    if profile not in KQUANT_PROFILES:
        raise GGUFError(f"Unknown Muse Glimmer K-quant profile {profile!r}")
    expected = KQUANT_PROFILES[profile]
    if (
        model.file_sha256 != expected["sha256"]
        or model.file_nbytes != expected["nbytes"]
        or model.tensor_table_sha256 != expected["tensor_table_sha256"]
    ):
        raise GGUFError("Cannot describe a K-quant model that was not fully authenticated")
    resident_weight_bytes = sum(tensor.nbytes for tensor in model.tensors)
    minimum_total = (32 if profile == "k-quant-dynamic" else 24) * 1024**3
    return {
        "format": "neuralfn.native_family_muse_glimmer.gguf.kquant.v1",
        "artifact_path": model.path.name,
        "target_nbytes": model.file_nbytes,
        "target_sha256": model.file_sha256,
        "artifact_size_bytes": model.file_nbytes,
        "artifact_sha256": model.file_sha256,
        "weight_precision": profile,
        "required_kernel_profile": "muse-glimmer-gguf-kquant-mapped-v1",
        "resident_weight_bytes": resident_weight_bytes,
        "peak_load_staging_bytes": 0,
        "max_workspace_bytes": 64 * 1024**2,
        "memory_profile": {
            "version": 1,
            "minimum_total_vram_bytes": minimum_total,
            "backend_fingerprint": "muse-glimmer-gguf-kquant-cpu-cuda-v1",
            "fixed_runtime_bytes": 0,
            "kv_cache_bytes_per_context_token_per_session": 0,
            "session_bytes": 0,
            "hybrid_kv_cache": {
                "local_layers": 39,
                "global_layers": 13,
                "local_window": 2_048,
                "kv_heads": 2,
                "head_dim": 128,
                "key_value_components": 2,
                "bytes_per_element": 2,
                "final_hidden_elements": 6_656,
            },
        },
        "gguf_version": model.version,
        "quantization_version": 2,
        "tensor_table_sha256": model.tensor_table_sha256,
        "tokenizer_metadata_sha256": model.tokenizer_metadata_sha256,
        "encoding_inventory": dict(model.encoding_inventory),
        "tensor_encodings": [
            {
                "name": tensor.name,
                "shape": list(tensor.shape),
                "encoding": tensor.encoding,
                "ggml_type": tensor.ggml_type,
                "relative_offset": tensor.relative_offset,
                "nbytes": tensor.nbytes,
                "row_stride_bytes": tensor.row_stride_bytes,
            }
            for tensor in model.tensors
        ],
        "source_provenance": {
            "repository": "meta-models/Muse-Glimmer-30B-GGUF",
            "revision": MUSE_GLIMMER_GGUF_REVISION,
            "canonical_filename": expected["filename"],
            "sha256": expected["sha256"],
            "tensor_table_sha256": expected["tensor_table_sha256"],
        },
        "capabilities": {
            "resident_cpu": True,
            "whole_model_cuda": True,
            "post_training": False,
        },
    }


def dflash_kquant_checkpoint_descriptor(
    model: GGUFModel,
    *,
    target_checkpoint_sha256: Sequence[str],
) -> dict[str, Any]:
    """Describe the authenticated packed DFlash companion and target binding."""

    expected = DFLASH_KQUANT_PROFILE
    allowed = tuple(dict.fromkeys(str(value).lower() for value in target_checkpoint_sha256))
    if (
        model.file_sha256 != expected["sha256"]
        or model.file_nbytes != expected["nbytes"]
        or model.tensor_table_sha256 != expected["tensor_table_sha256"]
        or model.encoding_inventory != expected["inventory"]
        or not allowed
        or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in allowed
        )
    ):
        raise GGUFError("Cannot describe an unauthenticated or unbound DFlash GGUF")
    return {
        "format": "neuralfn.native_family_muse_glimmer_dflash.gguf.kquant.v1",
        "artifact_path": model.path.name,
        "target_nbytes": model.file_nbytes,
        "target_sha256": model.file_sha256,
        "component": "dflash",
        "weight_precision": "k-quant-17gb",
        "required_kernel_profile": "muse-glimmer-dflash-gguf-kquant-mapped-v1",
        "resident_weight_bytes": sum(tensor.nbytes for tensor in model.tensors),
        "max_workspace_bytes": 16 * 202_048 * 4 + 64 * 1024**2,
        "gguf_version": 3,
        "quantization_version": 2,
        "tensor_table_sha256": model.tensor_table_sha256,
        "tokenizer_metadata_sha256": model.tokenizer_metadata_sha256,
        "encoding_inventory": dict(model.encoding_inventory),
        "tensor_encodings": [
            {
                "name": tensor.name,
                "native_name": tensor.native_name,
                "shape": list(tensor.shape),
                "encoding": tensor.encoding,
                "ggml_type": tensor.ggml_type,
                "relative_offset": tensor.relative_offset,
                "nbytes": tensor.nbytes,
                "row_stride_bytes": tensor.row_stride_bytes,
            }
            for tensor in model.tensors
        ],
        "target_compatibility": {
            "allowed_target_checkpoint_sha256": list(allowed),
            "target_layer_ids_gguf_one_based": [2, 14, 26, 38, 50],
            "target_layer_ids_zero_based": [1, 13, 25, 37, 49],
            "block_size": 16,
            "proposal_tokens": 15,
            "mask_token_id": 201_818,
            "shared_embedding": True,
            "shared_lm_head": True,
        },
        "source": {
            "repository": "meta-models/Muse-Glimmer-30B-GGUF",
            "revision": MUSE_GLIMMER_GGUF_REVISION,
            "canonical_filename": expected["filename"],
            "sha256": expected["sha256"],
        },
        "capabilities": {
            "resident_cpu": True,
            "resident_cuda": True,
            "greedy": True,
            "lossless_sampling": True,
        },
    }


def mmproj_kquant_checkpoint_descriptor(
    model: GGUFModel,
    *,
    target_checkpoint_sha256: Sequence[str],
) -> dict[str, Any]:
    """Describe the authenticated packed vision companion and target binding."""

    from .native_muse_glimmer_checkpoint import (
        MAIN_CONFIG_SHA256,
        MAIN_PROCESSOR_CONFIG_SHA256,
    )

    expected = MMPROJ_KQUANT_PROFILE
    allowed = tuple(dict.fromkeys(str(value).lower() for value in target_checkpoint_sha256))
    if (
        model.file_sha256 != expected["sha256"]
        or model.file_nbytes != expected["nbytes"]
        or model.tensor_table_sha256 != expected["tensor_table_sha256"]
        or model.encoding_inventory != expected["inventory"]
        or not allowed
        or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in allowed
        )
    ):
        raise GGUFError("Cannot describe an unauthenticated or unbound mmproj GGUF")
    return {
        "format": "neuralfn.native_family_muse_glimmer_mmproj.gguf.kquant.v1",
        "artifact_path": model.path.name,
        "target_nbytes": model.file_nbytes,
        "target_sha256": model.file_sha256,
        "component": "vision",
        "weight_precision": "k-quant-17gb",
        "required_kernel_profile": "muse-glimmer-mmproj-gguf-kquant-mapped-v1",
        "resident_weight_bytes": sum(tensor.nbytes for tensor in model.tensors),
        "max_workspace_bytes": 256 * 1024**2,
        "gguf_version": 3,
        "quantization_version": 2,
        "tensor_table_sha256": model.tensor_table_sha256,
        "encoding_inventory": dict(model.encoding_inventory),
        "tensor_encodings": [
            {
                "name": tensor.name,
                "native_name": tensor.native_name,
                "shape": list(tensor.shape),
                "encoding": tensor.encoding,
                "ggml_type": tensor.ggml_type,
                "relative_offset": tensor.relative_offset,
                "nbytes": tensor.nbytes,
                "row_stride_bytes": tensor.row_stride_bytes,
            }
            for tensor in model.tensors
        ],
        "target_compatibility": {
            "allowed_target_checkpoint_sha256": list(allowed),
            "target_config_sha256": MAIN_CONFIG_SHA256,
            "processor_config_sha256": MAIN_PROCESSOR_CONFIG_SHA256,
            "tokenizer_sha256": MUSE_GLIMMER_TOKENIZER_SHA256,
            "chat_template_sha256": MUSE_GLIMMER_ATEM_TEMPLATE_SHA256,
            "projection_width": 6_656,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "packed_patch_width": 588,
            "temporal_patch_reduction": "sum",
            "media_token_ids": {
                "image": MUSE_GLIMMER_SPECIAL_TOKEN_IDS["image"],
                "video": MUSE_GLIMMER_SPECIAL_TOKEN_IDS["video"],
                "patch": MUSE_GLIMMER_SPECIAL_TOKEN_IDS["patch"],
            },
        },
        "source": {
            "repository": "meta-models/Muse-Glimmer-30B-GGUF",
            "revision": MUSE_GLIMMER_GGUF_REVISION,
            "canonical_filename": expected["filename"],
            "sha256": expected["sha256"],
        },
        "capabilities": {
            "resident_cpu": True,
            "resident_cuda": False,
            "image": True,
            # The official GGUF converter collapses the temporal-2 patch kernel
            # to one 588-wide spatial slab. That is exact for duplicated still
            # images, but it cannot preserve two distinct video frames.
            "video": False,
        },
    }


def build_muse_glimmer_kquant_execution_manifest_payload(
    models: Mapping[str, GGUFModel],
    *,
    primary_variant: str | None = None,
    dflash_model: GGUFModel | None = None,
    mmproj_model: GGUFModel | None = None,
) -> dict[str, Any]:
    if not models or any(profile not in KQUANT_PROFILES for profile in models):
        raise GGUFError(
            "K-Quant execution manifest requires one or both canonical profile IDs"
        )
    descriptors = {
        profile: kquant_checkpoint_descriptor(model, profile=profile)
        for profile, model in models.items()
    }
    if primary_variant is None:
        primary_variant = (
            "k-quant-dynamic"
            if "k-quant-dynamic" in descriptors
            else "k-quant-17gb"
        )
    if primary_variant not in descriptors:
        raise GGUFError("Primary K-Quant profile is absent from the supplied artifacts")
    from dataclasses import asdict
    from .config import build_muse_glimmer_spec
    from .native_muse_glimmer_checkpoint import _base_muse_glimmer_execution_manifest

    model_spec = json.loads(json.dumps(asdict(build_muse_glimmer_spec())))
    target_digests = [
        descriptor["target_sha256"] for descriptor in descriptors.values()
    ]
    companions: dict[str, dict[str, Any]] = {}
    if dflash_model is not None:
        companions["dflash"] = dflash_kquant_checkpoint_descriptor(
            dflash_model,
            target_checkpoint_sha256=target_digests,
        )
    if mmproj_model is not None:
        companions["mmproj"] = mmproj_kquant_checkpoint_descriptor(
            mmproj_model,
            target_checkpoint_sha256=target_digests,
        )
    manifest = _base_muse_glimmer_execution_manifest(
        model_spec=model_spec,
        tensors=(),
        primary_variant=primary_variant,
        variants=descriptors,
        companion_checkpoints=companions,
    )
    manifest["source_graph"].update(
        {
            "kind": "pinned_official_gguf",
            "repository": "meta-models/Muse-Glimmer-30B-GGUF",
            "revision": MUSE_GLIMMER_GGUF_REVISION,
        }
    )
    manifest["topology"]["checkpoint_tensor_order"] = (
        "gguf_v3_mixed_per_tensor_typed_table"
    )
    return manifest


def publish_muse_glimmer_kquant_execution_bundle(
    gguf_paths: Sequence[str | Path],
    *,
    tokenizer_source: str | Path,
    output_root: str | Path,
    primary_variant: str | None = None,
    dflash_path: str | Path | None = None,
    mmproj_path: str | Path | None = None,
) -> Path:
    """Authenticate, copy, and publish a self-contained K-Quant artifact.

    Missing profiles are not downloaded. Every supplied file must use its
    canonical filename and pinned digest; the manifest is published last.
    """

    if not gguf_paths:
        raise GGUFError("At least one canonical GGUF path is required")
    source_by_profile: dict[str, Path] = {}
    models: dict[str, GGUFModel] = {}
    filenames = {
        str(profile_data["filename"]): profile
        for profile, profile_data in KQUANT_PROFILES.items()
    }
    for raw_path in gguf_paths:
        path = Path(raw_path).expanduser().resolve()
        profile = filenames.get(path.name)
        if profile is None:
            raise GGUFError(f"Unrecognized canonical Muse Glimmer GGUF filename {path.name!r}")
        if profile in models:
            raise GGUFError(f"Duplicate K-Quant profile {profile!r}")
        models[profile] = inspect_muse_glimmer_gguf(path, profile=profile)
        source_by_profile[profile] = path
    dflash_model = (
        inspect_muse_glimmer_dflash_gguf(dflash_path)
        if dflash_path is not None
        else None
    )
    mmproj_model = (
        inspect_muse_glimmer_mmproj_gguf(mmproj_path)
        if mmproj_path is not None
        else None
    )

    tokenizer_root = Path(tokenizer_source).expanduser().resolve()
    if not tokenizer_root.is_dir():
        raise GGUFError("tokenizer_source must be the pinned official main-model directory")
    from .native_chat import (
        MUSE_GLIMMER_ATEM_TEMPLATE_SHA256,
        MUSE_GLIMMER_TOKENIZER_CONFIG_SHA256,
        MUSE_GLIMMER_TOKENIZER_SHA256,
    )
    required_assets = {
        "tokenizer.json": MUSE_GLIMMER_TOKENIZER_SHA256,
        "tokenizer_config.json": MUSE_GLIMMER_TOKENIZER_CONFIG_SHA256,
        "chat_template.jinja": MUSE_GLIMMER_ATEM_TEMPLATE_SHA256,
    }
    for name, expected_sha in required_assets.items():
        path = (tokenizer_root / name).resolve()
        try:
            path.relative_to(tokenizer_root)
        except ValueError as exc:
            raise GGUFError("Tokenizer asset path escapes its source root") from exc
        if not path.is_file() or _sha256_file(path) != expected_sha:
            raise GGUFError(f"Pinned tokenizer asset {name} is missing or has the wrong SHA-256")

    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "native-execution-manifest.json"
    done_path = output / "native-execution-DONE"
    destination_files = [
        *(output / model.path.name for model in models.values()),
        *([output / dflash_model.path.name] if dflash_model is not None else []),
        *([output / mmproj_model.path.name] if mmproj_model is not None else []),
        *(output / name for name in required_assets),
        manifest_path,
        done_path,
    ]
    if any(path.exists() for path in destination_files):
        raise FileExistsError("Refusing to overwrite an existing K-Quant execution artifact")
    nonce = uuid.uuid4().hex
    published: list[Path] = []
    try:
        for profile, source in source_by_profile.items():
            destination = output / source.name
            temporary = output / f".{source.name}.{nonce}.tmp"
            digest = hashlib.sha256()
            with source.open("rb") as reader, temporary.open("xb") as writer:
                while True:
                    chunk = reader.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    writer.write(chunk)
                    digest.update(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            if digest.hexdigest() != KQUANT_PROFILES[profile]["sha256"]:
                raise GGUFError("GGUF source changed while it was copied")
            os.replace(temporary, destination)
            published.append(destination)
        if dflash_model is not None:
            source = dflash_model.path
            destination = output / source.name
            temporary = output / f".{source.name}.{nonce}.tmp"
            digest = hashlib.sha256()
            with source.open("rb") as reader, temporary.open("xb") as writer:
                while True:
                    chunk = reader.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    writer.write(chunk)
                    digest.update(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            if digest.hexdigest() != DFLASH_KQUANT_PROFILE["sha256"]:
                raise GGUFError("DFlash GGUF source changed while it was copied")
            os.replace(temporary, destination)
            published.append(destination)
        if mmproj_model is not None:
            source = mmproj_model.path
            destination = output / source.name
            temporary = output / f".{source.name}.{nonce}.tmp"
            digest = hashlib.sha256()
            with source.open("rb") as reader, temporary.open("xb") as writer:
                while True:
                    chunk = reader.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    writer.write(chunk)
                    digest.update(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            if digest.hexdigest() != MMPROJ_KQUANT_PROFILE["sha256"]:
                raise GGUFError("mmproj GGUF source changed while it was copied")
            os.replace(temporary, destination)
            published.append(destination)
        for name in required_assets:
            source = tokenizer_root / name
            destination = output / name
            temporary = output / f".{name}.{nonce}.tmp"
            digest = hashlib.sha256()
            with source.open("rb") as reader, temporary.open("xb") as writer:
                while True:
                    chunk = reader.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    writer.write(chunk)
                    digest.update(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            if digest.hexdigest() != required_assets[name]:
                raise GGUFError(f"Tokenizer asset {name} changed while it was copied")
            os.replace(temporary, destination)
            published.append(destination)
        manifest = build_muse_glimmer_kquant_execution_manifest_payload(
            models,
            primary_variant=primary_variant,
            dflash_model=dflash_model,
            mmproj_model=mmproj_model,
        )
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        temporary = output / f".{manifest_path.name}.{nonce}.tmp"
        with temporary.open("xb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
        published.append(manifest_path)
        temporary = output / f".{done_path.name}.{nonce}.tmp"
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            done_payload = {
                    "schema": "neuralfn.native_execution_bundle.done",
                    "version": 1,
                    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "profiles": sorted(models),
                }
            if dflash_model is not None:
                done_payload["dflash_checkpoint_sha256"] = dflash_model.file_sha256
            if mmproj_model is not None:
                done_payload["mmproj_checkpoint_sha256"] = mmproj_model.file_sha256
            json.dump(
                done_payload,
                stream,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, done_path)
        published.append(done_path)
        return manifest_path
    except Exception:
        for path in published:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        for path in output.glob(".*.tmp"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


__all__ = [
    "DFLASH_KQUANT_PROFILE",
    "GGML_TYPE_BF16",
    "GGML_TYPE_F32",
    "GGML_TYPE_LAYOUTS",
    "GGML_TYPE_Q4_K",
    "GGML_TYPE_Q5_K",
    "GGML_TYPE_Q6_K",
    "GGUFArray",
    "GGUFError",
    "GGUFModel",
    "GGUFTensorDescriptor",
    "KQUANT_PROFILES",
    "MMPROJ_KQUANT_PROFILE",
    "MUSE_GLIMMER_GGUF_TOKENIZER_METADATA_SHA256",
    "MUSE_GLIMMER_GGUF_REVISION",
    "build_muse_glimmer_kquant_execution_manifest_payload",
    "dflash_kquant_checkpoint_descriptor",
    "dequantize_ggml_blocks",
    "expected_muse_glimmer_dflash_gguf_tensor_names",
    "expected_muse_glimmer_gguf_tensor_names",
    "expected_muse_glimmer_mmproj_gguf_tensor_names",
    "inspect_muse_glimmer_dflash_gguf",
    "inspect_muse_glimmer_dflash_gguf_header_fixture",
    "inspect_muse_glimmer_gguf",
    "inspect_muse_glimmer_gguf_header_fixture",
    "inspect_muse_glimmer_mmproj_gguf",
    "inspect_muse_glimmer_mmproj_gguf_header_fixture",
    "kquant_checkpoint_descriptor",
    "mmproj_kquant_checkpoint_descriptor",
    "publish_muse_glimmer_kquant_execution_bundle",
]
