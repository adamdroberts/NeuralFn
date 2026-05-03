"""Dataset manager for loading HuggingFace and local datasets.

Handles downloading HuggingFace datasets into ~/.cache/nfn/datasets/,
listing available local datasets, and tokenizing text data into
integer sequences suitable for GPT-style training.
"""

from __future__ import annotations

from functools import lru_cache
import json
import os
import shutil
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
    return path if path.exists() else None


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


def _tokenizer_backed_uint16_shards(dataset_meta: dict[str, Any]) -> bool:
    if dataset_meta.get("data_format") != "uint16_shards":
        return False
    tokenizer_files = dataset_meta.get("tokenizer_files")
    tokenizer_name = dataset_meta.get("tokenizer_name")
    return bool(tokenizer_name) or (isinstance(tokenizer_files, list) and len(tokenizer_files) > 0)


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


def _max_token_id_in_uint16_shards(dataset_path: Path) -> int:
    shard_paths = sorted(dataset_path.glob("fineweb_*.bin"))
    if not shard_paths:
        raise DatasetTokenizerMismatchError(
            f"Tokenizer-backed cached dataset {dataset_path.name!r} has no .bin shard files to validate."
        )

    max_token_id = -1
    for shard_path in shard_paths:
        if shard_path.stat().st_size == 0:
            continue
        shard = np.memmap(shard_path, dtype=np.uint16, mode="r")
        offset = _shard_header_offset_uint16(shard_path)
        shard = shard[offset:]
        if shard.size == 0:
            continue
        shard_max = int(np.max(shard))
        if shard_max > max_token_id:
            max_token_id = shard_max
    return max_token_id


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
    if not _tokenizer_backed_uint16_shards(meta):
        return None

    model_path, vocab_path = resolve_cached_tokenizer_artifacts(ds_path, meta)
    tokenizer_vocab_size = _tokenizer_vocab_size_from_artifacts(model_path, vocab_path)
    max_token_id = _max_token_id_in_uint16_shards(ds_path)
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
    if not ds_path.is_dir() or meta.get("data_format") == "uint16_shards":
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

import torch
from torch.utils.data import Dataset

class MemmapTokenDataset(Dataset):
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
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
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
) -> Dataset:
    """Load one or more local datasets efficiently using MemmapTokenDataset."""
    _ensure_datasets_dir()
    
    arrays = []
    for ds_name in dataset_names:
        ds_path = DATASETS_DIR / ds_name
        if ds_path.is_dir():
            meta_file = ds_path / "meta.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if meta.get("data_format") == "uint16_shards":
                    validate_cached_tokenizer_contract(ds_name, dataset_path=ds_path, dataset_meta=meta)
                    train_files = sorted(ds_path.glob("fineweb_train_*.bin"))
                    for path in train_files:
                        # Memmap to avoid loading entirely into memory at once
                        arr = np.memmap(path, dtype=np.uint16, mode='r')
                        offset = _shard_header_offset_uint16(path)
                        arrays.append(arr[offset:])
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
            if meta.get("data_format") == "uint16_shards":
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
            if meta.get("data_format") == "uint16_shards":
                validate_cached_tokenizer_contract(ds_name, dataset_path=ds_path, dataset_meta=meta)
                train_files = sorted(ds_path.glob("fineweb_train_*.bin"))
                if not train_files:
                    raise FileNotFoundError(f"No training shards found in dataset '{ds_name}'")
                shards = [np.fromfile(path, dtype=np.uint16)[_shard_header_offset_uint16(path):] for path in train_files]
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
            if meta.get("data_format") == "uint16_shards":
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
