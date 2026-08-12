"""Torch-free inspection for production dense GPT native checkpoints.

The resident dense ABI consumes the version-5 checkpoint written by the native
GPT trainer.  This module is the single Python-side description of that binary
layout.  It validates geometry and exact file length before a checkpoint is
bound into a Native Execution artifact, and emits checksummed tensor records
without deserializing the payload or allocating it in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct
from typing import Any, BinaryIO, Mapping


NATIVE_DENSE_GPT_CHECKPOINT_FORMAT = "neuralfn.native_dense_gpt.v5"
NATIVE_DENSE_GPT_CHECKPOINT_MAGIC = 20240326
NATIVE_DENSE_GPT_CHECKPOINT_VERSION = 5
NATIVE_DENSE_GPT_HEADER_BYTES = 256 * 4
_BF16_BYTES = 2
_SUPPORTED_FAMILIES = frozenset({"gpt", "gpt2", "gpt3", "nanogpt", "gpt2-evo"})


@dataclass(frozen=True, slots=True)
class NativeDenseCheckpointTensor:
    name: str
    source_name: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int
    sha256: str
    dtype: str = "bfloat16"
    role: str = "parameter"
    byte_order: str = "little"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_name": self.source_name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "offset": self.offset,
            "nbytes": self.nbytes,
            "sha256": self.sha256,
            "role": self.role,
            "byte_order": self.byte_order,
        }


@dataclass(frozen=True, slots=True)
class NativeDenseCheckpointInfo:
    path: Path
    max_seq_len: int
    vocab_size: int
    num_layers: int
    num_heads: int
    channels: int
    padded_vocab_size: int
    parameter_count: int
    file_size: int
    sha256: str
    tensors: tuple[NativeDenseCheckpointTensor, ...]

    def checkpoint_descriptor(self, *, artifact_path: str) -> dict[str, Any]:
        return {
            "format": NATIVE_DENSE_GPT_CHECKPOINT_FORMAT,
            "artifact_path": artifact_path,
            "source_path": str(self.path),
            "source_format": NATIVE_DENSE_GPT_CHECKPOINT_FORMAT,
            "source_sha256": self.sha256,
            "target_file": artifact_path,
            "target_sha256": self.sha256,
            "target_nbytes": self.file_size,
            "restricted_unpickler": False,
            "isolated_worker": False,
            "tensor_offsets_include_header": True,
            "geometry": {
                "max_seq_len": self.max_seq_len,
                "vocab_size": self.vocab_size,
                "padded_vocab_size": self.padded_vocab_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "channels": self.channels,
            },
        }

    def validate_model(self, model: Mapping[str, Any]) -> None:
        """Reject graph identities that the dense-v5 resident engine cannot run."""

        family = str(model.get("family") or "").strip().lower().replace("_", "-")
        family_class = str(model.get("family_class") or "").strip().lower()
        if family not in _SUPPORTED_FAMILIES or family_class != "autoregressive_transformer":
            raise ValueError(
                "Native dense v5 checkpoints require a proved dense GPT-family "
                "autoregressive graph."
            )
        spec = model.get("template_spec")
        spec = spec if isinstance(spec, Mapping) else {}
        block = spec.get("block_spec")
        block = block if isinstance(block, Mapping) else {}

        required_block_values = {
            "norm_type": "layernorm",
            "mlp_type": "gelu",
            "pos_encoding": "absolute",
            "attention_variant": "dense",
            "residual_type": "add",
            "compression": "none",
            "activation_mode": "single",
        }
        for field, expected in required_block_values.items():
            if field not in block:
                raise ValueError(
                    f"Native dense v5 resident inference requires explicit block_spec.{field}."
                )
            actual = str(block[field] or "").strip().lower().replace("_", "-")
            if actual != expected:
                raise ValueError(
                    f"Native dense v5 resident inference requires block_spec.{field}={expected!r}; "
                    f"the graph declares {actual!r}."
                )
        if block.get("linear_bias") is not True:
            raise ValueError("Native dense v5 resident inference requires biased linear layers.")
        if "use_qk_norm" not in block or not isinstance(block["use_qk_norm"], bool):
            raise ValueError("Native dense v5 resident inference requires boolean use_qk_norm.")
        if "dropout_p" not in block or isinstance(block["dropout_p"], bool):
            raise ValueError("Native dense v5 resident inference requires explicit dropout_p=0.")
        if float(block["dropout_p"]) != 0.0:
            raise ValueError("Native dense v5 resident inference requires dropout_p=0.")
        try:
            if "logit_softcap" not in spec or isinstance(spec["logit_softcap"], bool):
                raise ValueError
            logit_softcap = float(spec["logit_softcap"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Native dense v5 resident inference requires a finite non-negative logit_softcap."
            ) from exc
        if not math.isfinite(logit_softcap) or logit_softcap < 0.0:
            raise ValueError(
                "Native dense v5 resident inference requires a finite non-negative logit_softcap."
            )
        if spec.get("tie_embeddings") is not True:
            raise ValueError("Native dense v5 checkpoints require tied token/output embeddings.")

        expected_geometry = {
            "model_dim": self.channels,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
        }
        for field, checkpoint_value in expected_geometry.items():
            graph_value = spec.get(field)
            if (
                graph_value in (None, "")
                or isinstance(graph_value, bool)
                or int(graph_value) != checkpoint_value
            ):
                raise ValueError(
                    f"Graph {field}={graph_value!r} does not match native checkpoint "
                    f"{field}={checkpoint_value}; the field is required explicitly."
                )
        graph_heads = block.get("num_heads")
        if (
            graph_heads in (None, "")
            or isinstance(graph_heads, bool)
            or int(graph_heads) != self.num_heads
        ):
            raise ValueError(
                f"Graph num_heads={graph_heads!r} does not match native checkpoint "
                f"num_heads={self.num_heads}; the field is required explicitly."
            )
        graph_kv_heads = block.get("num_kv_heads")
        if graph_kv_heads not in (None, "") and int(graph_kv_heads) != self.num_heads:
            raise ValueError("Native dense v5 resident inference requires MHA, not grouped-query attention.")


def _tensor_layout(
    *,
    max_seq_len: int,
    padded_vocab_size: int,
    num_layers: int,
    channels: int,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    tensors: list[tuple[str, tuple[int, ...]]] = [
        ("transformer.wte.weight", (padded_vocab_size, channels)),
        ("transformer.wpe.weight", (max_seq_len, channels)),
    ]
    for layer in range(num_layers):
        prefix = f"transformer.h.{layer}"
        tensors.extend(
            (
                (f"{prefix}.ln_1.weight", (channels,)),
                (f"{prefix}.ln_1.bias", (channels,)),
                (f"{prefix}.attn.c_attn.weight", (3 * channels, channels)),
                (f"{prefix}.attn.c_attn.bias", (3 * channels,)),
                (f"{prefix}.attn.c_proj.weight", (channels, channels)),
                (f"{prefix}.attn.c_proj.bias", (channels,)),
                (f"{prefix}.ln_2.weight", (channels,)),
                (f"{prefix}.ln_2.bias", (channels,)),
                (f"{prefix}.mlp.c_fc.weight", (4 * channels, channels)),
                (f"{prefix}.mlp.c_fc.bias", (4 * channels,)),
                (f"{prefix}.mlp.c_proj.weight", (channels, 4 * channels)),
                (f"{prefix}.mlp.c_proj.bias", (channels,)),
            )
        )
    tensors.extend(
        (
            ("transformer.ln_f.weight", (channels,)),
            ("transformer.ln_f.bias", (channels,)),
        )
    )
    return tuple(tensors)


def _read_exact_and_hash(
    handle: BinaryIO,
    count: int,
    *,
    file_digest: Any,
) -> str:
    tensor_digest = hashlib.sha256()
    remaining = count
    while remaining:
        chunk = handle.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise ValueError("Native dense checkpoint payload is truncated.")
        remaining -= len(chunk)
        tensor_digest.update(chunk)
        file_digest.update(chunk)
    return tensor_digest.hexdigest()


def inspect_native_dense_checkpoint(path: str | Path) -> NativeDenseCheckpointInfo:
    """Validate and checksum a dense-v5 checkpoint without loading its tensors."""

    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Native dense checkpoint is not a regular file: {resolved}")
    file_size = resolved.stat().st_size
    with resolved.open("rb") as handle:
        header_bytes = handle.read(NATIVE_DENSE_GPT_HEADER_BYTES)
        if len(header_bytes) != NATIVE_DENSE_GPT_HEADER_BYTES:
            raise ValueError("Native dense checkpoint header is truncated.")
        header = struct.unpack("<256i", header_bytes)
        if header[0] != NATIVE_DENSE_GPT_CHECKPOINT_MAGIC or header[1] != NATIVE_DENSE_GPT_CHECKPOINT_VERSION:
            raise ValueError("Resident inference requires a native dense GPT checkpoint at version 5.")
        max_seq_len, vocab_size, num_layers, num_heads, channels, padded_vocab_size = header[2:8]
        if (
            max_seq_len <= 0
            or vocab_size <= 0
            or num_layers <= 0
            or num_heads <= 0
            or channels <= 0
            or padded_vocab_size < vocab_size
            or channels % num_heads
        ):
            raise ValueError("Native dense checkpoint has invalid model geometry.")

        layout = _tensor_layout(
            max_seq_len=max_seq_len,
            padded_vocab_size=padded_vocab_size,
            num_layers=num_layers,
            channels=channels,
        )
        parameter_count = sum(
            _shape_elements(shape)
            for _name, shape in layout
        )
        expected_size = NATIVE_DENSE_GPT_HEADER_BYTES + parameter_count * _BF16_BYTES
        if file_size != expected_size:
            raise ValueError(
                "Native dense checkpoint file size does not match its version-5 geometry."
            )

        file_digest = hashlib.sha256()
        file_digest.update(header_bytes)
        offset = NATIVE_DENSE_GPT_HEADER_BYTES
        tensors: list[NativeDenseCheckpointTensor] = []
        for name, shape in layout:
            nbytes = _shape_elements(shape) * _BF16_BYTES
            checksum = _read_exact_and_hash(handle, nbytes, file_digest=file_digest)
            tensors.append(
                NativeDenseCheckpointTensor(
                    name=name,
                    source_name=name,
                    shape=shape,
                    offset=offset,
                    nbytes=nbytes,
                    sha256=checksum,
                )
            )
            offset += nbytes
        if handle.read(1):
            raise ValueError("Native dense checkpoint contains trailing payload bytes.")

    return NativeDenseCheckpointInfo(
        path=resolved,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        channels=channels,
        padded_vocab_size=padded_vocab_size,
        parameter_count=parameter_count,
        file_size=file_size,
        sha256=file_digest.hexdigest(),
        tensors=tuple(tensors),
    )


def _shape_elements(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        if dimension <= 0:
            raise ValueError("Native dense checkpoint tensor dimensions must be positive.")
        result *= dimension
    return result


__all__ = [
    "NATIVE_DENSE_GPT_CHECKPOINT_FORMAT",
    "NATIVE_DENSE_GPT_CHECKPOINT_MAGIC",
    "NATIVE_DENSE_GPT_CHECKPOINT_VERSION",
    "NativeDenseCheckpointInfo",
    "NativeDenseCheckpointTensor",
    "inspect_native_dense_checkpoint",
]
