"""Dataset manager for loading HuggingFace and local datasets.

Handles downloading HuggingFace datasets into ~/.cache/nfn/datasets/,
listing available local datasets, and tokenizing text data into
integer sequences suitable for GPT-style training.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken

NFN_CACHE_DIR = Path.home() / ".cache" / "nfn"
DATASETS_DIR = NFN_CACHE_DIR / "datasets"
SENTENCEPIECE_TOKENIZERS_DIR = NFN_CACHE_DIR / "tokenizers"
TIKTOKEN_ENCODINGS_DIR = Path.home() / "tiktoken_encodings"
RAW_TEXT_GPT2_BACKBONES = frozenset({"gpt2", "nanogpt"})
RAW_TEXT_DEFAULT_ENCODING = "o200k_base"
RAW_TEXT_CL100K_ENCODING = "cl100k_base"
RAW_TEXT_EOT_TOKEN = "<|endoftext|>"
RAW_TEXT_FILE_SUFFIXES = frozenset({".txt", ".json", ".jsonl", ".csv"})
TOKEN_SHARD_DATA_FORMATS = frozenset({"uint16_shards", "uint32_shards"})
TOKEN_SHARD_V2_MAGIC = b"NFNTSH2\0"
TOKEN_SHARD_V2_VERSION = 2
TOKEN_SHARD_V2_HEADER_BYTES = 512
TOKEN_SHARD_V2_DTYPE_UINT32_LE = 2
TOKEN_SHARD_V2_ENDIAN_MARKER = 0x01020304
STRUCTURED_SFT_V1_MAGIC = b"NFNSFT1\0"
STRUCTURED_SFT_V1_VERSION = 1
STRUCTURED_SFT_V1_HEADER_BYTES = 512
SENTENCEPIECE_TOKENIZER_VARIANTS = (
    "sp1024",
    "sp2048",
    "sp4096",
    "sp8192",
)
RAW_TEXT_ENCODING_ALIASES = {
    "gpt2": "gpt2",
    "tokgpt2": "gpt2",
    "cl100k": RAW_TEXT_CL100K_ENCODING,
    RAW_TEXT_CL100K_ENCODING: RAW_TEXT_CL100K_ENCODING,
    "o200k": RAW_TEXT_DEFAULT_ENCODING,
    RAW_TEXT_DEFAULT_ENCODING: RAW_TEXT_DEFAULT_ENCODING,
    **{name: name for name in SENTENCEPIECE_TOKENIZER_VARIANTS},
}
_LOCAL_TIKTOKEN_FILES = {
    "cl100k_base": "cl100k_base.tiktoken",
    "o200k_base": "o200k_base.tiktoken",
}
_KNOWN_TIKTOKEN_VOCAB_SIZES = {
    "gpt2": 50257,
    "cl100k_base": 100277,
    "o200k_base": 200019,
}
_SHARED_SENTENCEPIECE_MODEL_FILENAMES = {
    "sp1024": ("sp1024.model", "fineweb_1024_bpe.model"),
    "sp2048": ("sp2048.model", "fineweb_2048_bpe.model"),
    "sp4096": ("sp4096.model", "fineweb_4096_bpe.model"),
    "sp8192": ("sp8192.model", "fineweb_8192_bpe.model"),
}
_SHARED_SENTENCEPIECE_VOCAB_FILENAMES = {
    "sp1024": ("sp1024.vocab", "fineweb_1024_bpe.vocab"),
    "sp2048": ("sp2048.vocab", "fineweb_2048_bpe.vocab"),
    "sp4096": ("sp4096.vocab", "fineweb_4096_bpe.vocab"),
    "sp8192": ("sp8192.vocab", "fineweb_8192_bpe.vocab"),
}
_LOCAL_TIKTOKEN_SPECS: dict[str, dict[str, Any]] = {
    "cl100k_base": {
        "name": "cl100k_base",
        "pat_str": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""",
        "special_tokens": {
            RAW_TEXT_EOT_TOKEN: 100257,
            "<|fim_prefix|>": 100258,
            "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260,
            "<|endofprompt|>": 100276,
        },
    },
    "o200k_base": {
        "name": "o200k_base",
        "pat_str": "|".join(
            [
                r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                r"""\p{N}{1,3}""",
                r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
                r"""\s*[\r\n]+""",
                r"""\s+(?!\S)""",
                r"""\s+""",
            ]
        ),
        "special_tokens": {
            RAW_TEXT_EOT_TOKEN: 199999,
            "<|endofprompt|>": 200018,
        },
    },
}


class DatasetTokenizerMismatchError(ValueError):
    """Raised when a tokenizer-backed cached dataset alias is internally inconsistent."""


def _write_fixed_ascii(header: bytearray, offset: int, width: int, value: str, *, field: str) -> None:
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"Token shard field {field} must be ASCII") from exc
    if not encoded or len(encoded) >= width or b"\0" in encoded:
        raise ValueError(f"Token shard field {field} must contain 1..{width - 1} non-NUL ASCII bytes")
    header[offset : offset + len(encoded)] = encoded


def build_token_shard_v2_header(
    *,
    token_count: int,
    tokenizer_vocab_size: int,
    tokenizer_sha256: str,
    tokenizer_revision: str,
    tokenizer_name: str,
    split: str,
    objective: str = "ar",
) -> bytes:
    """Build the fixed, little-endian header for a native uint32 token shard."""

    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if not 0 < tokenizer_vocab_size <= np.iinfo(np.uint32).max:
        raise ValueError("tokenizer_vocab_size must fit uint32 and be positive")
    if len(tokenizer_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in tokenizer_sha256):
        raise ValueError("tokenizer_sha256 must be 64 lowercase hexadecimal characters")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if objective not in {"ar", "pretrain"}:
        raise ValueError("flat token shard v2 supports only ar/pretrain objectives")
    header = bytearray(TOKEN_SHARD_V2_HEADER_BYTES)
    header[:8] = TOKEN_SHARD_V2_MAGIC
    struct.pack_into(
        "<IIIIQII",
        header,
        8,
        TOKEN_SHARD_V2_VERSION,
        TOKEN_SHARD_V2_HEADER_BYTES,
        TOKEN_SHARD_V2_DTYPE_UINT32_LE,
        TOKEN_SHARD_V2_ENDIAN_MARKER,
        int(token_count),
        int(tokenizer_vocab_size),
        0,
    )
    _write_fixed_ascii(header, 40, 65, tokenizer_sha256, field="tokenizer_sha256")
    _write_fixed_ascii(header, 105, 96, tokenizer_revision, field="tokenizer_revision")
    _write_fixed_ascii(header, 201, 32, split, field="split")
    _write_fixed_ascii(header, 233, 32, objective, field="objective")
    _write_fixed_ascii(header, 265, 128, tokenizer_name, field="tokenizer_name")
    return bytes(header)


def build_structured_sft_v1_header(
    *,
    record_count: int,
    sequence_length: int,
    tokenizer_vocab_size: int,
    pad_token_id: int,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    tokenizer_revision: str,
    split: str,
) -> bytes:
    """Build the authenticated fixed-width native SFT record header."""

    if record_count <= 0 or sequence_length <= 0:
        raise ValueError("record_count and sequence_length must be positive")
    if not 0 < tokenizer_vocab_size <= np.iinfo(np.uint32).max:
        raise ValueError("tokenizer_vocab_size must fit uint32 and be positive")
    if not 0 <= pad_token_id < tokenizer_vocab_size:
        raise ValueError("pad_token_id must be inside tokenizer_vocab_size")
    for label, digest in (
        ("tokenizer_sha256", tokenizer_sha256),
        ("chat_template_sha256", chat_template_sha256),
    ):
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    header = bytearray(STRUCTURED_SFT_V1_HEADER_BYTES)
    header[:8] = STRUCTURED_SFT_V1_MAGIC
    struct.pack_into(
        "<IIIIQIII",
        header,
        8,
        STRUCTURED_SFT_V1_VERSION,
        STRUCTURED_SFT_V1_HEADER_BYTES,
        TOKEN_SHARD_V2_ENDIAN_MARKER,
        0,
        int(record_count),
        int(sequence_length),
        int(tokenizer_vocab_size),
        int(pad_token_id),
    )
    _write_fixed_ascii(header, 48, 65, tokenizer_sha256, field="tokenizer_sha256")
    _write_fixed_ascii(
        header, 113, 65, chat_template_sha256, field="chat_template_sha256"
    )
    _write_fixed_ascii(
        header, 178, 96, tokenizer_revision, field="tokenizer_revision"
    )
    _write_fixed_ascii(header, 274, 32, split, field="split")
    _write_fixed_ascii(header, 306, 32, "sft", field="objective")
    return bytes(header)


def _validated_structured_sft_arrays(
    record: dict[str, Any],
    *,
    sequence_length: int,
    tokenizer_vocab_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {"input_ids", "targets", "loss_mask", "sequence_ids"}
    if set(record) != required:
        raise ValueError(
            "Each structured SFT record must contain exactly input_ids, targets, "
            "loss_mask, and sequence_ids"
        )
    input_ids = np.asarray(record["input_ids"], dtype=np.dtype("<u4"))
    targets = np.asarray(record["targets"], dtype=np.dtype("<i4"))
    loss_mask = np.asarray(record["loss_mask"], dtype=np.dtype("<f4"))
    sequence_ids = np.asarray(record["sequence_ids"], dtype=np.dtype("<i4"))
    arrays = (input_ids, targets, loss_mask, sequence_ids)
    if any(array.ndim != 1 or array.shape[0] != sequence_length for array in arrays):
        raise ValueError("Every structured SFT array must have exactly sequence_length items")
    if bool(np.any(input_ids >= tokenizer_vocab_size)):
        raise ValueError("Structured SFT input_ids contain an out-of-vocabulary ID")
    invalid_targets = (targets != -100) & (
        (targets < 0) | (targets >= tokenizer_vocab_size)
    )
    if bool(np.any(invalid_targets)):
        raise ValueError("Structured SFT targets contain an invalid ID")
    if not bool(np.all(np.isfinite(loss_mask))) or bool(np.any(loss_mask < 0)):
        raise ValueError("Structured SFT loss_mask must be finite and non-negative")
    if bool(np.any((targets == -100) & (loss_mask != 0))) or not float(loss_mask.sum()) > 0:
        raise ValueError("Ignored targets require zero mask and each SFT record needs positive loss")
    if sequence_ids[0] != 0 or bool(np.any(sequence_ids < 0)):
        raise ValueError("Structured SFT sequence_ids must start at zero and be non-negative")
    deltas = np.diff(sequence_ids.astype(np.int64, copy=False))
    if bool(np.any((deltas != 0) & (deltas != 1))):
        raise ValueError("Structured SFT sequence_ids must be contiguous packed segments")
    return arrays


def write_structured_sft_v1(
    path: str | Path,
    records: list[dict[str, Any]],
    *,
    sequence_length: int,
    tokenizer_vocab_size: int,
    pad_token_id: int,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    tokenizer_revision: str,
    split: str,
) -> Path:
    """Atomically publish exact masked/segmented records for native Glimmer SFT."""

    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite structured SFT file: {destination}")
    if not records:
        raise ValueError("records must not be empty")
    validated = [
        _validated_structured_sft_arrays(
            record,
            sequence_length=sequence_length,
            tokenizer_vocab_size=tokenizer_vocab_size,
        )
        for record in records
    ]
    header = build_structured_sft_v1_header(
        record_count=len(validated),
        sequence_length=sequence_length,
        tokenizer_vocab_size=tokenizer_vocab_size,
        pad_token_id=pad_token_id,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        tokenizer_revision=tokenizer_revision,
        split=split,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Structured SFT staging file already exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(header)
            for arrays in validated:
                for array in arrays:
                    stream.write(array.tobytes(order="C"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


def inspect_structured_sft_v1(
    path: str | Path,
    *,
    validate_records: bool = True,
) -> dict[str, Any]:
    """Strictly inspect a native structured-SFT file and its lineage."""

    source = Path(path).expanduser().resolve()
    size = source.stat().st_size
    with source.open("rb") as stream:
        header = stream.read(STRUCTURED_SFT_V1_HEADER_BYTES)
    if len(header) != STRUCTURED_SFT_V1_HEADER_BYTES or header[:8] != STRUCTURED_SFT_V1_MAGIC:
        raise DatasetTokenizerMismatchError(f"Invalid structured SFT header: {source}")
    version, header_bytes, endian, flags, records, sequence_length, vocab, pad = struct.unpack_from(
        "<IIIIQIII", header, 8
    )
    if (
        version != STRUCTURED_SFT_V1_VERSION
        or header_bytes != STRUCTURED_SFT_V1_HEADER_BYTES
        or endian != TOKEN_SHARD_V2_ENDIAN_MARKER
        or flags != 0
        or records <= 0
        or sequence_length <= 0
        or vocab <= 0
        or pad >= vocab
        or size != header_bytes + records * sequence_length * 16
        or any(header[338:])
    ):
        raise DatasetTokenizerMismatchError(f"Invalid structured SFT geometry/extent: {source}")
    tokenizer_sha = _read_fixed_ascii(header, 48, 65, field="tokenizer_sha256")
    template_sha = _read_fixed_ascii(header, 113, 65, field="chat_template_sha256")
    revision = _read_fixed_ascii(header, 178, 96, field="tokenizer_revision")
    split = _read_fixed_ascii(header, 274, 32, field="split")
    objective = _read_fixed_ascii(header, 306, 32, field="objective")
    if (
        any(len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
            for value in (tokenizer_sha, template_sha))
        or split not in {"train", "validation", "test"}
        or objective != "sft"
    ):
        raise DatasetTokenizerMismatchError(f"Invalid structured SFT lineage: {source}")
    if validate_records:
        record_dtype = np.dtype(
            [
                ("input_ids", "<u4", (sequence_length,)),
                ("targets", "<i4", (sequence_length,)),
                ("loss_mask", "<f4", (sequence_length,)),
                ("sequence_ids", "<i4", (sequence_length,)),
            ]
        )
        payload = np.memmap(
            source,
            dtype=record_dtype,
            mode="r",
            offset=STRUCTURED_SFT_V1_HEADER_BYTES,
            shape=(records,),
        )
        for record in payload:
            _validated_structured_sft_arrays(
                {name: record[name] for name in record_dtype.names or ()},
                sequence_length=sequence_length,
                tokenizer_vocab_size=vocab,
            )
    return {
        "schema": "neuralfn.native_structured_sft.v1",
        "record_count": int(records),
        "sequence_length": int(sequence_length),
        "tokenizer_vocab_size": int(vocab),
        "pad_token_id": int(pad),
        "tokenizer_sha256": tokenizer_sha,
        "chat_template_sha256": template_sha,
        "tokenizer_revision": revision,
        "split": split,
        "objective": objective,
    }


def _read_fixed_ascii(header: bytes, offset: int, width: int, *, field: str) -> str:
    raw = header[offset : offset + width]
    nul = raw.find(b"\0")
    if nul < 0:
        raise DatasetTokenizerMismatchError(f"Token shard v2 field {field} is not NUL-terminated")
    if any(raw[nul:]):
        raise DatasetTokenizerMismatchError(f"Token shard v2 field {field} has nonzero padding")
    try:
        value = raw[:nul].decode("ascii")
    except UnicodeDecodeError as exc:
        raise DatasetTokenizerMismatchError(f"Token shard v2 field {field} is not ASCII") from exc
    if not value:
        raise DatasetTokenizerMismatchError(f"Token shard v2 field {field} is empty")
    return value


def inspect_token_shard(shard_path: Path, *, validate_ids: bool = True) -> dict[str, Any]:
    """Inspect one legacy uint16 or versioned uint32 shard without ambiguity."""

    file_bytes = shard_path.stat().st_size
    with shard_path.open("rb") as handle:
        prefix = handle.read(8)
        if prefix == TOKEN_SHARD_V2_MAGIC:
            handle.seek(0)
            header = handle.read(TOKEN_SHARD_V2_HEADER_BYTES)
            if len(header) != TOKEN_SHARD_V2_HEADER_BYTES:
                raise DatasetTokenizerMismatchError(f"Token shard v2 header is truncated: {shard_path}")
            version, header_bytes, dtype_code, endian, token_count, vocab_size, flags = struct.unpack_from(
                "<IIIIQII", header, 8
            )
            if version != TOKEN_SHARD_V2_VERSION or header_bytes != TOKEN_SHARD_V2_HEADER_BYTES:
                raise DatasetTokenizerMismatchError(f"Unsupported token shard v2 version/header: {shard_path}")
            if dtype_code != TOKEN_SHARD_V2_DTYPE_UINT32_LE:
                raise DatasetTokenizerMismatchError(f"Unsupported token shard v2 dtype {dtype_code}: {shard_path}")
            if endian != TOKEN_SHARD_V2_ENDIAN_MARKER:
                raise DatasetTokenizerMismatchError(f"Token shard v2 endian marker mismatch: {shard_path}")
            if token_count <= 0 or vocab_size <= 0 or flags != 0:
                raise DatasetTokenizerMismatchError(f"Invalid token shard v2 count/vocab/flags: {shard_path}")
            if file_bytes != header_bytes + token_count * 4:
                raise DatasetTokenizerMismatchError(f"Token shard v2 byte size does not match token_count: {shard_path}")
            tokenizer_sha256 = _read_fixed_ascii(header, 40, 65, field="tokenizer_sha256")
            tokenizer_revision = _read_fixed_ascii(header, 105, 96, field="tokenizer_revision")
            split = _read_fixed_ascii(header, 201, 32, field="split")
            objective = _read_fixed_ascii(header, 233, 32, field="objective")
            tokenizer_name = _read_fixed_ascii(header, 265, 128, field="tokenizer_name")
            if len(tokenizer_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in tokenizer_sha256):
                raise DatasetTokenizerMismatchError(f"Invalid token shard v2 tokenizer SHA-256: {shard_path}")
            if split not in {"train", "validation", "test"} or objective not in {"ar", "pretrain"}:
                raise DatasetTokenizerMismatchError(f"Invalid token shard v2 split/objective: {shard_path}")
            if any(header[393:]):
                raise DatasetTokenizerMismatchError(f"Token shard v2 reserved header bytes are nonzero: {shard_path}")
            max_token_id = -1
            if validate_ids:
                payload = np.memmap(
                    shard_path,
                    dtype=np.dtype("<u4"),
                    mode="r",
                    offset=TOKEN_SHARD_V2_HEADER_BYTES,
                    shape=(int(token_count),),
                )
                max_token_id = int(np.max(payload))
                if max_token_id >= int(vocab_size):
                    raise DatasetTokenizerMismatchError(
                        f"Token shard {shard_path} contains token id {max_token_id} outside tokenizer vocab {vocab_size}"
                    )
            return {
                "schema": "neuralfn.native_token_shard.v2",
                "dtype": "uint32_le",
                "element_bytes": 4,
                "header_bytes": int(header_bytes),
                "token_count": int(token_count),
                "tokenizer_vocab_size": int(vocab_size),
                "tokenizer_sha256": tokenizer_sha256,
                "tokenizer_revision": tokenizer_revision,
                "tokenizer_name": tokenizer_name,
                "split": split,
                "objective": objective,
                "max_token_id": max_token_id,
            }

    if file_bytes % 2:
        raise DatasetTokenizerMismatchError(f"Legacy uint16 token shard has odd byte size: {shard_path}")
    header_values = _shard_header_offset_uint16(shard_path)
    token_count = file_bytes // 2 - header_values
    max_token_id = -1
    if validate_ids and token_count:
        payload = np.memmap(shard_path, dtype=np.dtype("<u2"), mode="r", offset=header_values * 2)
        max_token_id = int(np.max(payload))
    return {
        "schema": "legacy.uint16",
        "dtype": "uint16_le",
        "element_bytes": 2,
        "header_bytes": header_values * 2,
        "token_count": int(token_count),
        "tokenizer_vocab_size": 65_536,
        "tokenizer_sha256": "",
        "tokenizer_revision": "",
        "tokenizer_name": "",
        "split": "",
        "objective": "",
        "max_token_id": max_token_id,
    }


def _ensure_datasets_dir() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_sentencepiece_tokenizers_dir() -> None:
    SENTENCEPIECE_TOKENIZERS_DIR.mkdir(parents=True, exist_ok=True)


def is_sentencepiece_tokenizer_name(encoding_name: str | None) -> bool:
    normalized = str(encoding_name or "").strip().lower()
    return normalized in SENTENCEPIECE_TOKENIZER_VARIANTS


def _sentencepiece_vocab_size(encoding_name: str) -> int:
    normalized = normalize_raw_text_encoding_name(encoding_name)
    if normalized not in SENTENCEPIECE_TOKENIZER_VARIANTS:
        raise ValueError(f"Unsupported sentencepiece tokenizer {encoding_name!r}")
    return int(str(normalized).removeprefix("sp"))


def shared_sentencepiece_artifact_filenames(encoding_name: str | None) -> dict[str, tuple[str, ...]]:
    normalized = normalize_raw_text_encoding_name(encoding_name)
    if normalized not in SENTENCEPIECE_TOKENIZER_VARIANTS:
        return {"model": (), "vocab": ()}
    return {
        "model": _SHARED_SENTENCEPIECE_MODEL_FILENAMES[str(normalized)],
        "vocab": _SHARED_SENTENCEPIECE_VOCAB_FILENAMES[str(normalized)],
    }


def _shared_sentencepiece_artifact_path(
    encoding_name: str | None,
    *,
    filenames: dict[str, tuple[str, ...]],
) -> Path | None:
    normalized = normalize_raw_text_encoding_name(encoding_name)
    if normalized not in SENTENCEPIECE_TOKENIZER_VARIANTS:
        return None
    for filename in filenames[str(normalized)]:
        candidate = SENTENCEPIECE_TOKENIZERS_DIR / filename
        if candidate.exists():
            return candidate
    return None


def shared_sentencepiece_model_path(encoding_name: str | None) -> Path | None:
    return _shared_sentencepiece_artifact_path(
        encoding_name,
        filenames=_SHARED_SENTENCEPIECE_MODEL_FILENAMES,
    )


def shared_sentencepiece_vocab_path(encoding_name: str | None) -> Path | None:
    return _shared_sentencepiece_artifact_path(
        encoding_name,
        filenames=_SHARED_SENTENCEPIECE_VOCAB_FILENAMES,
    )


def shared_sentencepiece_remote_artifact_paths(encoding_name: str | None) -> dict[str, tuple[str, ...]]:
    encoding_filenames = shared_sentencepiece_artifact_filenames(encoding_name)
    remote_root = "datasets/tokenizers"
    return {
        "model": tuple(f"{remote_root}/{name}" for name in encoding_filenames["model"]),
        "vocab": tuple(f"{remote_root}/{name}" for name in encoding_filenames["vocab"]),
    }


def resolve_sentencepiece_model_path(encoding_name: str) -> Path:
    normalized = normalize_raw_text_encoding_name(encoding_name)
    if normalized not in SENTENCEPIECE_TOKENIZER_VARIANTS:
        raise ValueError(f"Unsupported sentencepiece tokenizer {encoding_name!r}")
    model_path = shared_sentencepiece_model_path(normalized)
    if model_path is not None:
        return model_path
    expected = ", ".join(str(SENTENCEPIECE_TOKENIZERS_DIR / name) for name in _SHARED_SENTENCEPIECE_MODEL_FILENAMES[str(normalized)])
    raise FileNotFoundError(
        f"Raw-text tokenizer {normalized!r} requires a shared sentencepiece model under "
        f"{SENTENCEPIECE_TOKENIZERS_DIR}. Looked for: {expected}."
    )


@lru_cache(maxsize=None)
def resolve_sentencepiece_encoding(encoding_name: str):
    normalized = normalize_raw_text_encoding_name(encoding_name)
    if normalized not in SENTENCEPIECE_TOKENIZER_VARIANTS:
        raise ValueError(f"Unsupported sentencepiece tokenizer {encoding_name!r}")
    try:
        import sentencepiece as spm  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"Raw-text tokenizer {normalized!r} requires the sentencepiece package to be installed."
        ) from exc
    processor = spm.SentencePieceProcessor()
    model_path = resolve_sentencepiece_model_path(normalized)
    processor.load(str(model_path))
    expected_vocab_size = _sentencepiece_vocab_size(normalized)
    actual_vocab_size = int(processor.get_piece_size())
    if actual_vocab_size != expected_vocab_size:
        raise ValueError(
            f"Sentencepiece tokenizer {normalized!r} loaded from {model_path} reports vocab size "
            f"{actual_vocab_size}, expected {expected_vocab_size}."
        )
    return processor


def _raw_text_tokenizer_metadata_fields(encoding_name: str) -> dict[str, Any]:
    normalized = normalize_raw_text_encoding_name(encoding_name)
    if normalized is None:
        return {}
    metadata: dict[str, Any] = {
        "tokenizer_vocab_size": raw_text_encoding_vocab_size(normalized),
    }
    if is_sentencepiece_tokenizer_name(normalized):
        metadata["tokenizer_name"] = normalized
        model_path = shared_sentencepiece_model_path(normalized)
        vocab_path = shared_sentencepiece_vocab_path(normalized)
        tokenizer_files = [
            path.name
            for path in (model_path, vocab_path)
            if path is not None
        ]
        if tokenizer_files:
            metadata["tokenizer_files"] = tokenizer_files
    else:
        metadata["tokenizer_encoding"] = normalized
    return metadata


def normalize_raw_text_encoding_name(encoding_name: str | None) -> str | None:
    normalized = str(encoding_name or "").strip().lower()
    if not normalized:
        return None
    resolved = RAW_TEXT_ENCODING_ALIASES.get(normalized)
    if resolved is None:
        allowed = ", ".join(sorted(RAW_TEXT_ENCODING_ALIASES))
        raise ValueError(
            f"Unsupported raw-text encoding override {encoding_name!r}. "
            f"Expected one of: {allowed}."
        )
    return resolved


def raw_text_encoding_name_for_backbone(
    backbone: str | None,
    *,
    prefer_cl100k: bool = False,
    encoding_override: str | None = None,
) -> str:
    resolved_override = normalize_raw_text_encoding_name(encoding_override)
    if resolved_override is not None:
        return resolved_override
    normalized = str(backbone or "").strip().lower()
    if not normalized or normalized in RAW_TEXT_GPT2_BACKBONES:
        return "gpt2"
    return RAW_TEXT_CL100K_ENCODING if prefer_cl100k else RAW_TEXT_DEFAULT_ENCODING


def raw_text_encoding_name_for_template_spec(
    template_spec: dict[str, Any] | None,
    *,
    prefer_cl100k: bool = False,
    encoding_override: str | None = None,
) -> str:
    resolved_override = normalize_raw_text_encoding_name(encoding_override)
    if resolved_override is not None:
        return resolved_override
    resolved_template_name = normalize_raw_text_encoding_name((template_spec or {}).get("raw_text_encoding_name"))
    if resolved_template_name is not None:
        return resolved_template_name
    template = dict((template_spec or {}).get("template", {}) or {})
    return raw_text_encoding_name_for_backbone(
        str(template.get("backbone", "")),
        prefer_cl100k=prefer_cl100k,
    )


def local_tiktoken_encoding_path(encoding_name: str) -> Path | None:
    filename = _LOCAL_TIKTOKEN_FILES.get(str(encoding_name))
    if not filename:
        return None
    path = TIKTOKEN_ENCODINGS_DIR / filename
    if path.exists():
        return path
    return None


@lru_cache(maxsize=None)
def resolve_tiktoken_encoding(encoding_name: str) -> tiktoken.Encoding:
    local_path = local_tiktoken_encoding_path(encoding_name)
    if local_path is not None:
        from tiktoken.load import load_tiktoken_bpe

        spec = dict(_LOCAL_TIKTOKEN_SPECS[str(encoding_name)])
        spec["mergeable_ranks"] = load_tiktoken_bpe(str(local_path))
        return tiktoken.Encoding(**spec)
    return tiktoken.get_encoding(str(encoding_name))


def raw_text_encoding_vocab_size(encoding_name: str) -> int:
    normalized = normalize_raw_text_encoding_name(encoding_name)
    if normalized is None:
        raise ValueError("encoding_name must be non-empty")
    if is_sentencepiece_tokenizer_name(normalized):
        return _sentencepiece_vocab_size(normalized)
    if local_tiktoken_encoding_path(normalized) is None and normalized in _KNOWN_TIKTOKEN_VOCAB_SIZES:
        return _KNOWN_TIKTOKEN_VOCAB_SIZES[normalized]
    return int(resolve_tiktoken_encoding(normalized).n_vocab)


def _raw_text_allowed_special_tokens(encoding: tiktoken.Encoding) -> set[str]:
    specials = getattr(encoding, "special_tokens_set", set()) or set()
    if RAW_TEXT_EOT_TOKEN in specials:
        return {RAW_TEXT_EOT_TOKEN}
    return set()


def encode_raw_text(
    text: str,
    *,
    encoding_name: str = "gpt2",
    encoding: Any | None = None,
) -> list[int]:
    normalized = normalize_raw_text_encoding_name(encoding_name) or "gpt2"
    if is_sentencepiece_tokenizer_name(normalized):
        resolved = encoding or resolve_sentencepiece_encoding(normalized)
        encode = getattr(resolved, "encode", None)
        if not callable(encode):
            raise RuntimeError(f"Sentencepiece tokenizer {normalized!r} does not expose encode().")
        try:
            return list(encode(text, out_type=int))
        except TypeError:
            return [int(token) for token in encode(text)]
    resolved = encoding or resolve_tiktoken_encoding(normalized)
    return resolved.encode(
        text,
        allowed_special=_raw_text_allowed_special_tokens(resolved),
    )


def _load_dataset_meta(ds_path: Path) -> dict[str, Any]:
    meta_file = ds_path / "meta.json"
    if not meta_file.exists():
        return {}
    return json.loads(meta_file.read_text(encoding="utf-8"))


def _tokenizer_backed_token_shards(dataset_meta: dict[str, Any]) -> bool:
    if dataset_meta.get("data_format") not in TOKEN_SHARD_DATA_FORMATS:
        return False
    tokenizer_files = dataset_meta.get("tokenizer_files")
    tokenizer_name = dataset_meta.get("tokenizer_name")
    tokenizer_encoding = dataset_meta.get("tokenizer_encoding")
    return bool(tokenizer_name or tokenizer_encoding) or (
        isinstance(tokenizer_files, list) and len(tokenizer_files) > 0
    )


def _tokenizer_backed_uint16_shards(dataset_meta: dict[str, Any]) -> bool:
    """Compatibility predicate retained for callers that require legacy shards."""

    return dataset_meta.get("data_format") == "uint16_shards" and _tokenizer_backed_token_shards(dataset_meta)


def _raw_text_metadata_matches_encoding(dataset_meta: dict[str, Any], encoding_name: str) -> bool:
    normalized_encoding = normalize_raw_text_encoding_name(encoding_name) or "gpt2"
    tokenizer_vocab_size = raw_text_encoding_vocab_size(normalized_encoding)
    if is_sentencepiece_tokenizer_name(normalized_encoding):
        return (
            str(dataset_meta.get("tokenizer_name") or "").strip().lower() == normalized_encoding
            and int(dataset_meta.get("tokenizer_vocab_size") or 0) == tokenizer_vocab_size
        )
    return (
        str(dataset_meta.get("tokenizer_encoding") or "").strip().lower() == normalized_encoding
        and int(dataset_meta.get("tokenizer_vocab_size") or 0) == tokenizer_vocab_size
    )


def resolve_cached_tokenizer_artifacts(
    dataset_path: Path,
    dataset_meta: dict[str, Any],
) -> tuple[Path | None, Path | None]:
    tokenizer_dir = dataset_path / "tokenizers"
    model_candidates: list[Path] = []
    vocab_candidates: list[Path] = []

    tokenizer_files = dataset_meta.get("tokenizer_files")
    if isinstance(tokenizer_files, list):
        for filename in tokenizer_files:
            if not isinstance(filename, str):
                continue
            candidate = tokenizer_dir / Path(filename).name
            if filename.endswith(".model"):
                model_candidates.append(candidate)
            elif filename.endswith(".vocab"):
                vocab_candidates.append(candidate)

    if tokenizer_dir.exists():
        model_candidates.extend(sorted(tokenizer_dir.glob("*.model")))
        vocab_candidates.extend(sorted(tokenizer_dir.glob("*.vocab")))

    model_path = next((path for path in model_candidates if path.exists()), None)
    vocab_path = next((path for path in vocab_candidates if path.exists()), None)
    return model_path, vocab_path


def _tokenizer_vocab_size_from_artifacts(model_path: Path | None, vocab_path: Path | None) -> int:
    if model_path is not None and model_path.exists():
        try:
            import sentencepiece as spm  # type: ignore
        except ImportError:
            pass
        else:
            try:
                processor = spm.SentencePieceProcessor()
                processor.load(str(model_path))
            except Exception:
                pass
            else:
                return int(processor.get_piece_size())

    if vocab_path is not None and vocab_path.exists():
        with vocab_path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    artifact = model_path or vocab_path
    if artifact is None:
        raise DatasetTokenizerMismatchError(
            "Tokenizer-backed cached dataset is missing tokenizer artifacts under its tokenizers/ directory."
        )
    raise DatasetTokenizerMismatchError(
        f"Could not determine tokenizer vocab size from {artifact}. "
        "Install sentencepiece or include the tokenizer .vocab file in the cached alias."
    )


def _shard_header_offset_uint16(shard_path: Path) -> int:
    """Return the uint16 element offset to skip a binary shard header, if present.

    The header is 1024 bytes (512 uint16 elements) and starts with magic
    ``0x0134D888`` stored little-endian (``b'\\x88\\xd8\\x34\\x01'``).
    """
    with open(shard_path, "rb") as f:
        magic = f.read(4)
    if magic == b'\x88\xd8\x34\x01':
        return 512  # 1024 bytes / 2 bytes per uint16
    return 0


def _max_token_id_in_token_shards(dataset_path: Path) -> int:
    shard_paths = sorted(dataset_path.glob("fineweb_*.bin"))
    if not shard_paths:
        raise DatasetTokenizerMismatchError(
            f"Tokenizer-backed cached dataset {dataset_path.name!r} has no .bin shard files to validate."
        )

    max_token_id = -1
    for shard_path in shard_paths:
        if shard_path.stat().st_size == 0:
            continue
        inspected = inspect_token_shard(shard_path, validate_ids=True)
        if inspected["token_count"] == 0:
            continue
        shard_max = int(inspected["max_token_id"])
        if shard_max > max_token_id:
            max_token_id = shard_max
    return max_token_id


def _uint16_shard_token_count(shard_path: Path) -> int:
    return max(0, (shard_path.stat().st_size // 2) - _shard_header_offset_uint16(shard_path))


def _uint16_shard_sequence_count(shard_paths: list[Path], seq_len: int) -> int:
    return sum(max(0, (_uint16_shard_token_count(path) - 1) // seq_len) for path in shard_paths)


def _token_shard_token_count(shard_path: Path) -> int:
    return int(inspect_token_shard(shard_path, validate_ids=False)["token_count"])


def _token_shard_sequence_count(shard_paths: list[Path], seq_len: int) -> int:
    return sum(max(0, (_token_shard_token_count(path) - 1) // seq_len) for path in shard_paths)


def _token_shard_memmap(shard_path: Path) -> np.memmap:
    inspected = inspect_token_shard(shard_path, validate_ids=False)
    dtype = np.dtype("<u4") if inspected["dtype"] == "uint32_le" else np.dtype("<u2")
    return np.memmap(
        shard_path,
        dtype=dtype,
        mode="r",
        offset=int(inspected["header_bytes"]),
        shape=(int(inspected["token_count"]),),
    )


def _tokenizer_mismatch_message(
    *,
    dataset_name: str,
    tokenizer_path: Path | None,
    tokenizer_vocab_size: int,
    max_token_id: int | None = None,
    model_vocab_size: int | None = None,
) -> str:
    tokenizer_label = str(tokenizer_path) if tokenizer_path is not None else "<missing tokenizer artifact>"
    lines = [
        f"Dataset alias {dataset_name!r} has an invalid tokenizer-backed cached token contract.",
        f"Tokenizer artifact: {tokenizer_label}",
        f"Tokenizer vocab size: {tokenizer_vocab_size}",
    ]
    if max_token_id is not None:
        lines.extend(
            [
                f"Observed max token id in cached shards: {max_token_id}",
                f"Expected every cached token id to be < {tokenizer_vocab_size}.",
            ]
        )
    if model_vocab_size is not None:
        lines.append(f"Model/checkpoint vocab size: {model_vocab_size}")
    lines.append(
        "Delete/rebuild or re-download this dataset alias with matching tokenizer artifacts before training or inference."
    )
    return " ".join(lines)


def validate_cached_tokenizer_contract(
    dataset_name: str,
    *,
    dataset_path: Path | None = None,
    dataset_meta: dict[str, Any] | None = None,
    model_vocab_size: int | None = None,
) -> dict[str, Any] | None:
    _ensure_datasets_dir()
    ds_path = dataset_path or (DATASETS_DIR / dataset_name)
    if not ds_path.is_dir():
        return None

    meta = dataset_meta if dataset_meta is not None else _load_dataset_meta(ds_path)
    if not _tokenizer_backed_token_shards(meta):
        return None

    model_path, vocab_path = resolve_cached_tokenizer_artifacts(ds_path, meta)
    tokenizer_encoding = str(meta.get("tokenizer_encoding") or "").strip().lower()
    if tokenizer_encoding:
        tokenizer_vocab_size = raw_text_encoding_vocab_size(tokenizer_encoding)
    else:
        tokenizer_vocab_size = _tokenizer_vocab_size_from_artifacts(model_path, vocab_path)
    max_token_id = _max_token_id_in_token_shards(ds_path)
    if max_token_id >= tokenizer_vocab_size:
        raise DatasetTokenizerMismatchError(
            _tokenizer_mismatch_message(
                dataset_name=dataset_name,
                tokenizer_path=model_path or vocab_path,
                tokenizer_vocab_size=tokenizer_vocab_size,
                max_token_id=max_token_id,
            )
        )
    if model_vocab_size is not None and int(model_vocab_size) != tokenizer_vocab_size:
        raise DatasetTokenizerMismatchError(
            _tokenizer_mismatch_message(
                dataset_name=dataset_name,
                tokenizer_path=model_path or vocab_path,
                tokenizer_vocab_size=tokenizer_vocab_size,
                max_token_id=max_token_id,
                model_vocab_size=int(model_vocab_size),
            )
        )
    if meta.get("data_format") == "uint32_shards":
        expected_sha = str(meta.get("tokenizer_sha256") or "")
        expected_revision = str(meta.get("tokenizer_revision") or "")
        for shard_path in sorted(ds_path.glob("fineweb_*.bin")):
            inspected = inspect_token_shard(shard_path, validate_ids=False)
            if inspected["dtype"] != "uint32_le":
                raise DatasetTokenizerMismatchError(
                    f"Dataset {dataset_name!r} declares uint32_shards but {shard_path.name} is {inspected['dtype']}"
                )
            if int(inspected["tokenizer_vocab_size"]) != tokenizer_vocab_size:
                raise DatasetTokenizerMismatchError(
                    f"Dataset {dataset_name!r} shard vocab does not match tokenizer metadata"
                )
            if expected_sha and inspected["tokenizer_sha256"] != expected_sha:
                raise DatasetTokenizerMismatchError(
                    f"Dataset {dataset_name!r} shard tokenizer SHA-256 does not match meta.json"
                )
            if expected_revision and inspected["tokenizer_revision"] != expected_revision:
                raise DatasetTokenizerMismatchError(
                    f"Dataset {dataset_name!r} shard tokenizer revision does not match meta.json"
                )
    return {
        "dataset_name": dataset_name,
        "dataset_path": ds_path,
        "dataset_meta": meta,
        "tokenizer_model_path": model_path,
        "tokenizer_vocab_path": vocab_path,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "max_token_id": max_token_id,
    }


# ── Listing ───────────────────────────────────────────────────────────

def _meta_to_summary(name: str, meta: dict[str, Any], *, default_source: str) -> dict[str, Any]:
    return {
        "name": name,
        "source": meta.get("source", default_source),
        "hf_path": meta.get("hf_path"),
        "hf_split": meta.get("hf_split"),
        "text_column": meta.get("text_column", "text"),
        "num_tokens": meta.get("num_tokens"),
        "num_rows": meta.get("num_rows"),
        "variant": meta.get("variant"),
        "train_shards": meta.get("train_shards"),
        "val_shards": meta.get("val_shards"),
        "data_format": meta.get("data_format"),
        "repo_id": meta.get("repo_id"),
        "remote_root_prefix": meta.get("remote_root_prefix"),
        "train_file": meta.get("train_file"),
        "val_file": meta.get("val_file"),
        "tokenizer_name": meta.get("tokenizer_name"),
        "tokenizer_encoding": meta.get("tokenizer_encoding"),
        "tokenizer_vocab_size": meta.get("tokenizer_vocab_size"),
        "tokenizer_sha256": meta.get("tokenizer_sha256"),
        "tokenizer_revision": meta.get("tokenizer_revision"),
        "token_shard_schema": meta.get("token_shard_schema"),
    }


def get_local_dataset_info(name: str) -> dict[str, Any] | None:
    """Return metadata for one dataset stored under ~/.cache/nfn/datasets/."""
    _ensure_datasets_dir()
    ds_dir = DATASETS_DIR / name
    if ds_dir.is_dir():
        meta_file = ds_dir / "meta.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        return _meta_to_summary(name, meta, default_source="local")
    for ext in (".txt", ".json", ".jsonl", ".csv", ".parquet"):
        file_path = DATASETS_DIR / f"{name}{ext}"
        if file_path.exists():
            return _meta_to_summary(name, {}, default_source="local_file")
    return None


def list_local_datasets() -> list[dict[str, Any]]:
    """Return metadata about locally available datasets in ~/.cache/nfn/datasets/."""
    _ensure_datasets_dir()
    results: list[dict[str, Any]] = []
    for entry in sorted(DATASETS_DIR.iterdir()):
        if entry.name.startswith("."):
            continue
        summary = get_local_dataset_info(entry.stem if entry.is_file() else entry.name)
        if summary is not None:
            results.append(summary)
    deduped = {dataset["name"]: dataset for dataset in results}
    return [deduped[name] for name in sorted(deduped)]


# ── Downloading / Importing ───────────────────────────────────────────

def download_hf_dataset(
    hf_path: str,
    *,
    hf_split: str = "train",
    text_column: str = "text",
    max_rows: int | None = None,
    alias: str | None = None,
    variant: str | None = None,
    train_shards: int | None = None,
    skip_manifest: bool = False,
    with_docs: bool = False,
    repo_id: str | None = None,
    remote_root_prefix: str = "datasets",
    train_file: str | None = None,
    val_file: str | None = None,
    encoding_name: str = "gpt2",
) -> dict[str, Any]:
    """Download a HuggingFace dataset and persist it locally as a .txt file.

    Falls back to a direct raw-file download for legacy script-based datasets
    (e.g. karpathy/tiny_shakespeare).

    Returns metadata about the downloaded dataset.
    """
    if variant is not None:
        return _download_cached_fineweb_variant(
            hf_path,
            variant=variant,
            train_shards=train_shards,
            alias=alias,
            skip_manifest=skip_manifest,
            with_docs=with_docs,
            repo_id=repo_id,
            remote_root_prefix=remote_root_prefix,
        )

    from datasets import load_dataset

    _ensure_datasets_dir()
    ds_name = alias or hf_path.replace("/", "__")
    ds_dir = DATASETS_DIR / ds_name
    created_now = not ds_dir.exists()
    ds_dir.mkdir(parents=True, exist_ok=True)

    text_path = ds_dir / "data.txt"
    val_path = ds_dir / "val.txt" if val_file else None
    try:
        if train_file is not None:
            num_rows = _download_explicit_raw_hf_text_file(
                hf_path,
                train_file,
                text_path,
                max_rows=max_rows,
            )
            val_rows = None
            if val_path is not None:
                val_rows = _download_explicit_raw_hf_text_file(
                    hf_path,
                    val_file,
                    val_path,
                    max_rows=max_rows,
                )
            full_text = text_path.read_text(encoding="utf-8")
            num_tokens = len(encode_raw_text(full_text, encoding_name=encoding_name))
            meta = {
                "source": "huggingface",
                "hf_path": hf_path,
                "hf_split": hf_split,
                "text_column": text_column,
                "num_rows": num_rows,
                "num_tokens": num_tokens,
                "train_file": train_file,
                "val_file": val_file,
                "val_rows": val_rows,
                **_raw_text_tokenizer_metadata_fields(encoding_name),
            }
            (ds_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            return {"name": ds_name, **meta}

        try:
            ds = load_dataset(hf_path, split=hf_split)
            available_cols = ds.column_names
            col = text_column if text_column in available_cols else available_cols[0]

            if max_rows is not None and len(ds) > max_rows:
                ds = ds.select(range(max_rows))

            num_rows = 0
            with open(text_path, "w", encoding="utf-8") as f:
                for row in ds:
                    line = str(row[col]).replace("\n", " ")
                    f.write(line + "\n")
                    num_rows += 1
        except Exception:
            # Fallback: try to download the raw text file directly from the repo
            num_rows = _download_raw_hf_text(hf_path, text_path, max_rows)

        full_text = text_path.read_text(encoding="utf-8")
        num_tokens = len(encode_raw_text(full_text, encoding_name=encoding_name))

        meta = {
            "source": "huggingface",
            "hf_path": hf_path,
            "hf_split": hf_split,
            "text_column": text_column,
            "num_rows": num_rows,
            "num_tokens": num_tokens,
            **_raw_text_tokenizer_metadata_fields(encoding_name),
        }
        (ds_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return {"name": ds_name, **meta}
    except Exception:
        if created_now:
            shutil.rmtree(ds_dir, ignore_errors=True)
        raise


def _dataset_dir_for_variant(name: str) -> str:
    if name == "byte260":
        return "fineweb10B_byte260"
    if name.startswith("sp") and name[2:].isdigit():
        return f"fineweb10B_{name}"
    raise ValueError(f"unsupported variant {name!r}; expected byte260 or sp<VOCAB_SIZE>")


def _download_hf_file(
    repo_id: str,
    relative_path: str,
    destination: Path,
    *,
    repo_type: str = "dataset",
) -> Path:
    from huggingface_hub import hf_hub_download

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    remote_path = Path(relative_path)
    cached_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=remote_path.name,
            subfolder=remote_path.parent.as_posix() if remote_path.parent != Path(".") else None,
            repo_type=repo_type,
        )
    )
    cached_source = cached_path.resolve(strict=True)
    try:
        os.link(cached_source, destination)
    except OSError:
        shutil.copy2(cached_source, destination)
    return destination


def _download_cached_fineweb_variant(
    hf_path: str,
    *,
    variant: str,
    train_shards: int | None,
    alias: str | None,
    skip_manifest: bool,
    with_docs: bool,
    repo_id: str | None,
    remote_root_prefix: str,
) -> dict[str, Any]:
    _ensure_datasets_dir()

    repo = repo_id or hf_path
    effective_train_shards = 80 if train_shards is None else train_shards
    if effective_train_shards < 0:
        raise ValueError("train_shards must be non-negative")

    ds_name = alias or f"{repo.replace('/', '__')}__{variant}__train{effective_train_shards}"
    ds_dir = DATASETS_DIR / ds_name
    ds_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = ds_dir / "manifest.json"
    manifest_remote = f"{remote_root_prefix}/manifest.json"
    if not manifest_path.exists():
        if skip_manifest:
            raise FileNotFoundError(
                f"manifest.json is required for variant downloads but skip_manifest=True and {manifest_path} is missing"
            )
        _download_hf_file(repo, manifest_remote, manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_dir = _dataset_dir_for_variant(variant)
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir), None)
    if dataset_entry is None:
        raise ValueError(f"dataset {dataset_dir} not found in {manifest_remote}")

    max_train_shards = int((dataset_entry.get("stats") or {}).get("files_train", 0))
    val_shards = int((dataset_entry.get("stats") or {}).get("files_val", 0))
    if effective_train_shards > max_train_shards:
        raise ValueError(
            f"{variant} only has {max_train_shards} training shards on {repo}, requested {effective_train_shards}"
        )

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
    if tokenizer_entry is None:
        raise ValueError(f"tokenizer {tokenizer_name} not found in {manifest_remote}")

    dataset_prefix = f"{remote_root_prefix}/datasets/{dataset_dir}"
    for i in range(val_shards):
        _download_hf_file(repo, f"{dataset_prefix}/fineweb_val_{i:06d}.bin", ds_dir / f"fineweb_val_{i:06d}.bin")
    for i in range(effective_train_shards):
        _download_hf_file(repo, f"{dataset_prefix}/fineweb_train_{i:06d}.bin", ds_dir / f"fineweb_train_{i:06d}.bin")

    tokenizer_artifacts: list[str] = []
    for key in ("model_path", "vocab_path", "path"):
        value = tokenizer_entry.get(key)
        if value:
            tokenizer_artifacts.append(str(value))
    if not tokenizer_artifacts:
        raise ValueError(f"tokenizer entry is missing downloadable artifacts: {tokenizer_entry}")

    for artifact_path in tokenizer_artifacts:
        filename = Path(artifact_path).name
        _download_hf_file(repo, f"{remote_root_prefix}/{artifact_path}", ds_dir / "tokenizers" / filename)

    if with_docs:
        _download_hf_file(repo, f"{remote_root_prefix}/docs_selected.jsonl", ds_dir / "docs_selected.jsonl")
        _download_hf_file(
            repo,
            f"{remote_root_prefix}/docs_selected.source_manifest.json",
            ds_dir / "docs_selected.source_manifest.json",
        )

    train_files = sorted(ds_dir.glob("fineweb_train_*.bin"))
    num_tokens = sum(path.stat().st_size for path in train_files) // 2
    meta = {
        "source": "huggingface_cached_tokens",
        "hf_path": hf_path,
        "hf_split": "train",
        "text_column": "tokens",
        "num_rows": effective_train_shards,
        "num_tokens": int(num_tokens),
        "variant": variant,
        "train_shards": effective_train_shards,
        "val_shards": val_shards,
        "repo_id": repo,
        "remote_root_prefix": remote_root_prefix,
        "tokenizer_name": tokenizer_name,
        "tokenizer_files": [Path(path).name for path in tokenizer_artifacts],
        "data_format": "uint16_shards",
    }
    try:
        validate_cached_tokenizer_contract(ds_name, dataset_path=ds_dir, dataset_meta=meta)
    except Exception:
        shutil.rmtree(ds_dir, ignore_errors=True)
        raise
    (ds_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"name": ds_name, **meta}


def _download_raw_hf_text(hf_path: str, dest: Path, max_rows: int | None) -> int:
    """Fallback: download data from a HuggingFace dataset repo.

    Uses huggingface_hub to list/download files. For legacy script-based repos
    (like karpathy/tiny_shakespeare), parses the .py script to extract the
    actual data URL and downloads from there.
    """
    import re
    import urllib.request
    from huggingface_hub import list_repo_files, hf_hub_download

    # List all files in the repo
    try:
        files = list_repo_files(hf_path, repo_type="dataset")
    except Exception:
        files = []

    # Strategy 1: look for data files directly in the repo
    data_extensions = {".txt", ".csv", ".json", ".jsonl", ".parquet"}
    data_files = [f for f in files if Path(f).suffix in data_extensions]
    for data_file in data_files:
        try:
            local = hf_hub_download(hf_path, data_file, repo_type="dataset")
            text = Path(local).read_text(encoding="utf-8")
            dest.write_text(text, encoding="utf-8")
            return _trim_rows(dest, text, max_rows)
        except Exception:
            continue

    # Strategy 2: find .py scripts and extract data URLs from them
    script_files = [f for f in files if f.endswith(".py")]
    for script_file in script_files:
        try:
            local = hf_hub_download(hf_path, script_file, repo_type="dataset")
            script_text = Path(local).read_text(encoding="utf-8")
            urls = re.findall(r'["\'](https?://[^"\']+)["\']', script_text)
            for data_url in urls:
                if any(ext in data_url for ext in [".txt", ".csv", ".json"]):
                    try:
                        urllib.request.urlretrieve(data_url, str(dest))
                        text = dest.read_text(encoding="utf-8")
                        return _trim_rows(dest, text, max_rows)
                    except Exception:
                        continue
        except Exception:
            continue

    raise FileNotFoundError(
        f"Could not load HuggingFace dataset '{hf_path}'. "
        f"No downloadable data files found. Try uploading the data manually."
    )


def _download_explicit_raw_hf_text_file(
    hf_path: str,
    filename: str,
    dest: Path,
    *,
    max_rows: int | None,
) -> int:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(hf_path, filename, repo_type="dataset")
    text = Path(local).read_text(encoding="utf-8")
    dest.write_text(text, encoding="utf-8")
    return _trim_rows(dest, text, max_rows)


def _trim_rows(dest: Path, text: str, max_rows: int | None) -> int:
    """Count rows and optionally trim a text file to max_rows lines."""
    num_rows = text.count("\n")
    if max_rows is not None and num_rows > max_rows:
        lines = text.split("\n")[:max_rows]
        dest.write_text("\n".join(lines), encoding="utf-8")
        return max_rows
    return num_rows


def upload_local_file(name: str, content: bytes, filename: str) -> dict[str, Any]:
    """Save an uploaded file into the datasets directory."""
    _ensure_datasets_dir()
    ds_dir = DATASETS_DIR / name
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Write the raw file
    ext = Path(filename).suffix or ".txt"
    data_path = ds_dir / f"data{ext}"
    data_path.write_bytes(content)

    # If it's a text file, count tokens
    num_tokens = None
    num_rows = None
    if ext in {".txt", ".json", ".jsonl", ".csv"}:
        try:
            text = data_path.read_text(encoding="utf-8")
            num_tokens = len(encode_raw_text(text))
            num_rows = text.count("\n")
        except Exception:
            pass

    meta = {
        "source": "local_upload",
        "hf_path": None,
        "hf_split": None,
        "text_column": "text",
        "num_rows": num_rows,
        "num_tokens": num_tokens,
    }
    (ds_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return {"name": name, **meta}


def _raw_text_data_file_for_path(ds_path: Path) -> Path | None:
    if ds_path.is_file():
        return ds_path if ds_path.suffix in RAW_TEXT_FILE_SUFFIXES else None
    if not ds_path.is_dir():
        return None
    data_file = ds_path / "data.txt"
    if data_file.exists():
        return data_file
    for candidate in sorted(ds_path.iterdir()):
        if candidate.is_file() and candidate.name != "meta.json" and candidate.suffix in RAW_TEXT_FILE_SUFFIXES:
            return candidate
    return None


def refresh_raw_text_dataset_metadata(
    dataset_name: str,
    *,
    dataset_path: Path | None = None,
    dataset_meta: dict[str, Any] | None = None,
    encoding_name: str = "gpt2",
) -> dict[str, Any]:
    _ensure_datasets_dir()
    ds_path = dataset_path or (DATASETS_DIR / dataset_name)
    meta = dict(dataset_meta or _load_dataset_meta(ds_path))
    if not ds_path.is_dir() or meta.get("data_format") in TOKEN_SHARD_DATA_FORMATS:
        return meta

    tokenizer_vocab_size = raw_text_encoding_vocab_size(encoding_name)
    normalized_encoding = normalize_raw_text_encoding_name(encoding_name) or "gpt2"
    tokenizer_matches = False
    if is_sentencepiece_tokenizer_name(normalized_encoding):
        tokenizer_matches = (
            str(meta.get("tokenizer_name") or "").strip().lower() == normalized_encoding
            and int(meta.get("tokenizer_vocab_size") or 0) == tokenizer_vocab_size
        )
    else:
        tokenizer_matches = (
            meta.get("tokenizer_encoding") == normalized_encoding
            and int(meta.get("tokenizer_vocab_size") or 0) == tokenizer_vocab_size
        )
    if tokenizer_matches and meta.get("num_tokens") is not None:
        return meta

    data_file = _raw_text_data_file_for_path(ds_path)
    if data_file is None:
        raise FileNotFoundError(f"No raw-text data file found in dataset {dataset_name!r}")

    full_text = data_file.read_text(encoding="utf-8")
    meta["num_tokens"] = len(encode_raw_text(full_text, encoding_name=normalized_encoding))
    meta["tokenizer_vocab_size"] = tokenizer_vocab_size
    if is_sentencepiece_tokenizer_name(normalized_encoding):
        meta["tokenizer_name"] = normalized_encoding
        meta.pop("tokenizer_encoding", None)
        tokenizer_files = _raw_text_tokenizer_metadata_fields(normalized_encoding).get("tokenizer_files")
        if tokenizer_files is not None:
            meta["tokenizer_files"] = tokenizer_files
    else:
        meta["tokenizer_encoding"] = normalized_encoding
        meta.pop("tokenizer_name", None)
    meta.setdefault("num_rows", full_text.count("\n"))

    val_path = ds_path / "val.txt"
    if val_path.exists():
        meta.setdefault("val_rows", val_path.read_text(encoding="utf-8").count("\n"))

    (ds_path / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_text_tokenizer_identity(encoding_name: str, encoding: Any) -> tuple[str, str, str]:
    normalized = normalize_raw_text_encoding_name(encoding_name) or "gpt2"
    artifact = (
        shared_sentencepiece_model_path(normalized)
        if is_sentencepiece_tokenizer_name(normalized)
        else local_tiktoken_encoding_path(normalized)
    )
    if artifact is not None:
        sha256 = _sha256_file(artifact)
        revision = f"artifact-sha256:{sha256[:64]}"
        return normalized, sha256, revision

    # Built-in tiktoken encodings have no standalone artifact. Fingerprint the
    # complete rank/special-token contract rather than trusting only its name.
    digest = hashlib.sha256()
    digest.update(b"neuralfn.tiktoken.contract.v1\0")
    digest.update(normalized.encode("utf-8") + b"\0")
    digest.update(str(getattr(encoding, "_pat_str", "")).encode("utf-8") + b"\0")
    mergeable = getattr(encoding, "_mergeable_ranks", {})
    for token, rank in sorted(mergeable.items(), key=lambda item: (int(item[1]), bytes(item[0]))):
        raw = bytes(token)
        digest.update(struct.pack("<II", len(raw), int(rank)))
        digest.update(raw)
    specials = getattr(encoding, "_special_tokens", {})
    for token, rank in sorted(specials.items()):
        raw = str(token).encode("utf-8")
        digest.update(struct.pack("<II", len(raw), int(rank)))
        digest.update(raw)
    sha256 = digest.hexdigest()
    revision = f"tiktoken-contract-v1:{getattr(tiktoken, '__version__', 'unknown')}"
    return normalized, sha256, revision


def _write_token_shard_from_text(
    source_path: Path,
    destination_path: Path,
    *,
    encoding_name: str,
    encoding: Any | None = None,
    dtype: str,
    tokenizer_vocab_size: int,
    tokenizer_sha256: str,
    tokenizer_revision: str,
    tokenizer_name: str,
    split: str,
) -> int:
    if dtype not in {"uint16", "uint32"}:
        raise ValueError("token shard dtype must be uint16 or uint32")
    token_count = 0
    tmp_path = destination_path.with_suffix(destination_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        with source_path.open("r", encoding="utf-8") as source, tmp_path.open("w+b") as output:
            if dtype == "uint32":
                output.write(bytes(TOKEN_SHARD_V2_HEADER_BYTES))
            for text in source:
                tokens = encode_raw_text(text, encoding_name=encoding_name, encoding=encoding)
                if not tokens:
                    continue
                if min(tokens) < 0 or max(tokens) >= tokenizer_vocab_size:
                    raise ValueError(
                        f"Raw-text token cache for {source_path} contains an id outside tokenizer vocab {tokenizer_vocab_size}."
                    )
                if dtype == "uint16" and max(tokens) > np.iinfo(np.uint16).max:
                    raise ValueError(
                        f"Raw-text token cache for {source_path} cannot be stored as uint16 with tokenizer {encoding_name!r}."
                    )
                output.write(np.asarray(tokens, dtype=np.dtype("<u2" if dtype == "uint16" else "<u4")).tobytes())
                token_count += len(tokens)
            if dtype == "uint32":
                output.seek(0)
                output.write(
                    build_token_shard_v2_header(
                        token_count=token_count,
                        tokenizer_vocab_size=tokenizer_vocab_size,
                        tokenizer_sha256=tokenizer_sha256,
                        tokenizer_revision=tokenizer_revision,
                        tokenizer_name=tokenizer_name,
                        split=split,
                        objective="ar",
                    )
                )
        tmp_path.replace(destination_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return token_count


def _write_uint16_token_shard_from_text(
    source_path: Path,
    destination_path: Path,
    *,
    encoding_name: str,
    encoding: Any | None = None,
) -> int:
    resolved = encoding or (
        resolve_sentencepiece_encoding(encoding_name)
        if is_sentencepiece_tokenizer_name(encoding_name)
        else resolve_tiktoken_encoding(encoding_name)
    )
    tokenizer_name, tokenizer_sha256, tokenizer_revision = _raw_text_tokenizer_identity(
        encoding_name, resolved
    )
    return _write_token_shard_from_text(
        source_path,
        destination_path,
        encoding_name=encoding_name,
        encoding=resolved,
        dtype="uint16",
        tokenizer_vocab_size=raw_text_encoding_vocab_size(encoding_name),
        tokenizer_sha256=tokenizer_sha256,
        tokenizer_revision=tokenizer_revision,
        tokenizer_name=tokenizer_name,
        split="validation" if "val" in destination_path.stem else "train",
    )


def ensure_raw_text_token_cache(
    dataset_name: str,
    *,
    dataset_path: Path | None = None,
    dataset_meta: dict[str, Any] | None = None,
    encoding_name: str = "gpt2",
) -> dict[str, Any]:
    """Materialize raw-text datasets as unambiguous uint16/uint32 shards.

    This keeps repeated training runs on large text corpora from re-reading and
    re-tokenizing multi-GB text files. Wide tokenizers use the versioned
    little-endian uint32 format; legacy small-vocabulary caches remain uint16.
    """

    _ensure_datasets_dir()
    ds_path = dataset_path or (DATASETS_DIR / dataset_name)
    meta = dict(dataset_meta or _load_dataset_meta(ds_path))
    if not ds_path.is_dir():
        return meta

    normalized_encoding = normalize_raw_text_encoding_name(encoding_name) or "gpt2"
    tokenizer_vocab_size = raw_text_encoding_vocab_size(normalized_encoding)
    dtype = "uint16" if tokenizer_vocab_size <= int(np.iinfo(np.uint16).max) + 1 else "uint32"
    data_format = f"{dtype}_shards"

    train_files = sorted(ds_path.glob("fineweb_train_*.bin"))
    if (
        meta.get("data_format") == data_format
        and train_files
        and _raw_text_metadata_matches_encoding(meta, normalized_encoding)
    ):
        validate_cached_tokenizer_contract(dataset_name, dataset_path=ds_path, dataset_meta=meta)
        return meta

    data_file = _raw_text_data_file_for_path(ds_path)
    if data_file is None:
        return meta

    if is_sentencepiece_tokenizer_name(normalized_encoding):
        encoding = resolve_sentencepiece_encoding(normalized_encoding)
    else:
        encoding = resolve_tiktoken_encoding(normalized_encoding)
    tokenizer_name, tokenizer_sha256, tokenizer_revision = _raw_text_tokenizer_identity(
        normalized_encoding, encoding
    )

    for stale_path in sorted(ds_path.glob("fineweb_train_*.bin")) + sorted(ds_path.glob("fineweb_val_*.bin")):
        stale_path.unlink()

    train_path = ds_path / "fineweb_train_000000.bin"
    train_tokens = _write_token_shard_from_text(
        data_file,
        train_path,
        encoding_name=normalized_encoding,
        encoding=encoding,
        dtype=dtype,
        tokenizer_vocab_size=tokenizer_vocab_size,
        tokenizer_sha256=tokenizer_sha256,
        tokenizer_revision=tokenizer_revision,
        tokenizer_name=tokenizer_name,
        split="train",
    )

    val_shards = 0
    val_tokens = 0
    val_path = ds_path / "val.txt"
    if val_path.exists():
        val_tokens = _write_token_shard_from_text(
            val_path,
            ds_path / "fineweb_val_000000.bin",
            encoding_name=normalized_encoding,
            encoding=encoding,
            dtype=dtype,
            tokenizer_vocab_size=tokenizer_vocab_size,
            tokenizer_sha256=tokenizer_sha256,
            tokenizer_revision=tokenizer_revision,
            tokenizer_name=tokenizer_name,
            split="validation",
        )
        val_shards = 1

    meta.update(
        {
            "text_column": "tokens",
            "num_tokens": int(train_tokens),
            "raw_text_train_file": data_file.name,
            "raw_text_val_file": val_path.name if val_path.exists() else None,
            "train_shards": 1,
            "val_shards": val_shards,
            "data_format": data_format,
            "token_cache_format": f"raw_text_{dtype}_shards",
            "token_cache_dtype": dtype,
            "token_shard_schema": (
                "legacy.uint16" if dtype == "uint16" else "neuralfn.native_token_shard.v2"
            ),
            "tokenizer_sha256": tokenizer_sha256,
            "tokenizer_revision": tokenizer_revision,
            "token_cache_train_tokens": int(train_tokens),
            "token_cache_val_tokens": int(val_tokens),
            **_raw_text_tokenizer_metadata_fields(normalized_encoding),
        }
    )
    (ds_path / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def estimate_dataset_sequence_count(
    dataset_name: str,
    *,
    seq_len: int,
    encoding_name: str = "gpt2",
) -> int | None:
    """Return dataset row count from metadata/shard sizes without tokenizing text."""

    _ensure_datasets_dir()
    ds_path = DATASETS_DIR / dataset_name
    if not ds_path.is_dir():
        return None
    meta = _load_dataset_meta(ds_path)
    if meta.get("data_format") in TOKEN_SHARD_DATA_FORMATS:
        train_files = sorted(ds_path.glob("fineweb_train_*.bin"))
        if train_files:
            return _token_shard_sequence_count(train_files, seq_len)
    if _raw_text_metadata_matches_encoding(meta, encoding_name) and meta.get("num_tokens") is not None:
        return max(0, (int(meta["num_tokens"]) - 1) // seq_len)
    return None


# ── Loading for Training ─────────────────────────────────────────────

def load_dataset_tokens(
    dataset_names: list[str],
    *,
    seq_len: int = 64,
    encoding_name: str = "gpt2",
) -> tuple[list[list[int]], list[list[int]]]:
    """Load one or more local datasets and tokenize into training sequences.

    Returns (inputs, targets) where each is a list of integer lists of
    length `seq_len`.  targets are inputs shifted by one token.
    """
    _ensure_datasets_dir()
    enc: Any | None = None

    all_tokens: list[int] = []
    for ds_name in dataset_names:
        tokens = _load_tokens_for(ds_name, enc, encoding_name=encoding_name)
        all_tokens.extend(tokens)

    if len(all_tokens) < seq_len + 1:
        raise ValueError(
            f"Combined dataset has only {len(all_tokens)} tokens but "
            f"need at least {seq_len + 1} for seq_len={seq_len}"
        )

    # Chunk into sequences of (seq_len + 1), inputs=chunk[:-1], targets=chunk[1:]
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for start in range(0, len(all_tokens) - seq_len, seq_len):
        chunk = all_tokens[start : start + seq_len + 1]
        if len(chunk) < seq_len + 1:
            break
        inputs.append(chunk[:-1])
        targets.append(chunk[1:])

    return inputs, targets


def load_dataset_bytes(
    dataset_names: list[str],
    *,
    seq_len: int = 64,
) -> tuple[list[list[int]], list[list[int]]]:
    """Load one or more datasets as raw-byte training sequences."""
    _ensure_datasets_dir()

    all_bytes: list[int] = []
    for ds_name in dataset_names:
        all_bytes.extend(_load_bytes_for(ds_name))

    if len(all_bytes) < seq_len + 1:
        raise ValueError(
            f"Combined dataset has only {len(all_bytes)} bytes but "
            f"need at least {seq_len + 1} for seq_len={seq_len}"
        )

    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for start in range(0, len(all_bytes) - seq_len, seq_len):
        chunk = all_bytes[start : start + seq_len + 1]
        if len(chunk) < seq_len + 1:
            break
        inputs.append(chunk[:-1])
        targets.append(chunk[1:])

    return inputs, targets

class MemmapTokenDataset:
    def __init__(self, token_arrays: list[np.ndarray], seq_len: int):
        self.seq_len = seq_len
        self.arrays = token_arrays
        self.array_lengths = [len(arr) for arr in self.arrays]
        
        self.chunk_counts = []
        for length in self.array_lengths:
            count = max(0, (length - 1) // seq_len)
            self.chunk_counts.append(count)
            
        self.cumulative_chunks = np.cumsum([0] + self.chunk_counts)
        self.total_chunks = self.cumulative_chunks[-1]
        
    def __len__(self) -> int:
        return self.total_chunks
        
    def __getitem__(self, idx: int):
        import torch

        if idx < 0 or idx >= self.total_chunks:
            raise IndexError("Index out of bounds")
            
        array_idx = np.searchsorted(self.cumulative_chunks[1:], idx, side='right')
        local_idx = idx - self.cumulative_chunks[array_idx]
        
        start_pos = local_idx * self.seq_len
        end_pos = start_pos + self.seq_len + 1
        
        chunk = self.arrays[array_idx][start_pos:end_pos].astype(np.int64)
        
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

def load_dataset_tensors(
    dataset_names: list[str],
    *,
    seq_len: int = 64,
    encoding_name: str = "gpt2",
) -> MemmapTokenDataset:
    """Load one or more local datasets efficiently using MemmapTokenDataset."""
    _ensure_datasets_dir()
    
    arrays = []
    for ds_name in dataset_names:
        ds_path = DATASETS_DIR / ds_name
        if ds_path.is_dir():
            meta_file = ds_path / "meta.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if meta.get("data_format") not in TOKEN_SHARD_DATA_FORMATS:
                    meta = ensure_raw_text_token_cache(
                        ds_name,
                        dataset_path=ds_path,
                        dataset_meta=meta,
                        encoding_name=encoding_name,
                    )
                if meta.get("data_format") in TOKEN_SHARD_DATA_FORMATS:
                    validate_cached_tokenizer_contract(ds_name, dataset_path=ds_path, dataset_meta=meta)
                    train_files = sorted(ds_path.glob("fineweb_train_*.bin"))
                    for path in train_files:
                        # Memmap to avoid loading entirely into memory at once
                        arrays.append(_token_shard_memmap(path))
                    continue
                
        # Fallback to in-memory load for text/json
        tokens = _load_tokens_for(ds_name, None, encoding_name=encoding_name)
        arrays.append(np.array(tokens, dtype=np.int32))
        
    if not arrays:
        raise ValueError(f"No tokens found for datasets {dataset_names}")
        
    return MemmapTokenDataset(arrays, seq_len)


def load_dataset_byte_tensors(
    dataset_names: list[str],
    *,
    seq_len: int = 64,
) -> Dataset:
    """Load one or more datasets as raw-byte tensors efficiently."""
    _ensure_datasets_dir()

    arrays: list[np.ndarray] = []
    for ds_name in dataset_names:
        data_file = _data_file_for(ds_name)
        if data_file is not None:
            arrays.append(np.memmap(data_file, dtype=np.uint8, mode='r'))
            continue
        raw_bytes = _load_bytes_for(ds_name)
        arrays.append(np.array(raw_bytes, dtype=np.uint8))

    if not arrays:
        raise ValueError(f"No bytes found for datasets {dataset_names}")

    return MemmapTokenDataset(arrays, seq_len)


def _data_file_for(ds_name: str) -> Path | None:
    ds_path = DATASETS_DIR / ds_name
    if ds_path.is_dir():
        meta_file = ds_path / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta.get("data_format") in TOKEN_SHARD_DATA_FORMATS:
                return None

        data_file = ds_path / "data.txt"
        if not data_file.exists():
            for candidate in ds_path.iterdir():
                if candidate.is_file() and candidate.name != "meta.json":
                    data_file = candidate
                    break
        return data_file if data_file.exists() else None

    for ext in (".txt", ".json", ".jsonl", ".csv", ".parquet", ".bin"):
        file_path = DATASETS_DIR / f"{ds_name}{ext}"
        if file_path.exists():
            return file_path
    return None

def _load_tokens_for(
    ds_name: str,
    enc: Any | None,
    *,
    encoding_name: str = "gpt2",
) -> list[int]:
    """Load tokenized data for a single dataset name."""
    ds_path = DATASETS_DIR / ds_name

    # Case 1: it's a directory with data.txt
    if ds_path.is_dir():
        meta_file = ds_path / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta.get("data_format") in TOKEN_SHARD_DATA_FORMATS:
                validate_cached_tokenizer_contract(ds_name, dataset_path=ds_path, dataset_meta=meta)
                train_files = sorted(ds_path.glob("fineweb_train_*.bin"))
                if not train_files:
                    raise FileNotFoundError(f"No training shards found in dataset '{ds_name}'")
                shards = [np.asarray(_token_shard_memmap(path)) for path in train_files]
                if not shards:
                    raise FileNotFoundError(f"No readable training shards found in dataset '{ds_name}'")
                return np.concatenate(shards).astype(int).tolist()

        data_file = ds_path / "data.txt"
        if not data_file.exists():
            # Try to find any data file
            for candidate in ds_path.iterdir():
                if candidate.suffix in {".txt", ".json", ".jsonl", ".csv"}:
                    data_file = candidate
                    break
        if not data_file.exists():
            raise FileNotFoundError(f"No data file found in dataset '{ds_name}'")
        text = data_file.read_text(encoding="utf-8")
        if enc is None:
            if is_sentencepiece_tokenizer_name(encoding_name):
                enc = resolve_sentencepiece_encoding(encoding_name)
            else:
                enc = resolve_tiktoken_encoding(encoding_name)
        return encode_raw_text(text, encoding_name=encoding_name, encoding=enc)

    # Case 2: it's a plain file in the datasets dir
    for ext in (".txt", ".json", ".jsonl", ".csv", ".parquet"):
        file_path = DATASETS_DIR / f"{ds_name}{ext}"
        if file_path.exists():
            text = file_path.read_text(encoding="utf-8")
            if enc is None:
                if is_sentencepiece_tokenizer_name(encoding_name):
                    enc = resolve_sentencepiece_encoding(encoding_name)
                else:
                    enc = resolve_tiktoken_encoding(encoding_name)
            return encode_raw_text(text, encoding_name=encoding_name, encoding=enc)

    raise FileNotFoundError(f"Dataset '{ds_name}' not found in {DATASETS_DIR}")


def _load_bytes_for(ds_name: str) -> list[int]:
    """Load raw bytes for a single dataset name."""
    ds_path = DATASETS_DIR / ds_name

    if ds_path.is_dir():
        meta_file = ds_path / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta.get("data_format") in TOKEN_SHARD_DATA_FORMATS:
                raise ValueError(
                    f"Dataset '{ds_name}' stores token shards and cannot be used for raw-byte H-Net training"
                )

    data_file = _data_file_for(ds_name)
    if data_file is None:
        raise FileNotFoundError(f"Dataset '{ds_name}' not found in {DATASETS_DIR}")
    return list(data_file.read_bytes())


def delete_dataset(ds_name: str) -> bool:
    """Delete a dataset from the local storage. Returns True if deleted."""
    ds_path = DATASETS_DIR / ds_name
    if ds_path.is_dir():
        shutil.rmtree(ds_path)
        return True
    for ext in (".txt", ".json", ".jsonl", ".csv", ".parquet"):
        file_path = DATASETS_DIR / f"{ds_name}{ext}"
        if file_path.exists():
            file_path.unlink()
            return True
    return False
