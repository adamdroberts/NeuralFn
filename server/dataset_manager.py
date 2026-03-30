"""Dataset manager for loading HuggingFace and local datasets.

Handles downloading HuggingFace datasets into server/datasets/,
listing available local datasets, and tokenizing text data into
integer sequences suitable for GPT-style training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import tiktoken

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"


def _ensure_datasets_dir() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)


# ── Listing ───────────────────────────────────────────────────────────

def list_local_datasets() -> list[dict[str, Any]]:
    """Return metadata about locally available datasets in server/datasets/."""
    _ensure_datasets_dir()
    results: list[dict[str, Any]] = []
    for entry in sorted(DATASETS_DIR.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            meta_file = entry / "meta.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
            else:
                meta = {}
            results.append({
                "name": entry.name,
                "source": meta.get("source", "local"),
                "hf_path": meta.get("hf_path"),
                "hf_split": meta.get("hf_split"),
                "text_column": meta.get("text_column", "text"),
                "num_tokens": meta.get("num_tokens"),
                "num_rows": meta.get("num_rows"),
            })
        elif entry.suffix in {".txt", ".json", ".jsonl", ".csv", ".parquet"}:
            results.append({
                "name": entry.stem,
                "source": "local_file",
                "hf_path": None,
                "hf_split": None,
                "text_column": "text",
                "num_tokens": None,
                "num_rows": None,
            })
    return results


# ── Downloading / Importing ───────────────────────────────────────────

def download_hf_dataset(
    hf_path: str,
    *,
    hf_split: str = "train",
    text_column: str = "text",
    max_rows: int | None = None,
    alias: str | None = None,
) -> dict[str, Any]:
    """Download a HuggingFace dataset and persist it locally as a .txt file.

    Falls back to a direct raw-file download for legacy script-based datasets
    (e.g. karpathy/tiny_shakespeare).

    Returns metadata about the downloaded dataset.
    """
    from datasets import load_dataset

    _ensure_datasets_dir()
    ds_name = alias or hf_path.replace("/", "__")
    ds_dir = DATASETS_DIR / ds_name
    ds_dir.mkdir(parents=True, exist_ok=True)

    text_path = ds_dir / "data.txt"

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

    # Tokenize and count tokens
    enc = tiktoken.get_encoding("gpt2")
    full_text = text_path.read_text(encoding="utf-8")
    token_ids = enc.encode(full_text)
    num_tokens = len(token_ids)

    meta = {
        "source": "huggingface",
        "hf_path": hf_path,
        "hf_split": hf_split,
        "text_column": text_column,
        "num_rows": num_rows,
        "num_tokens": num_tokens,
    }
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
            enc = tiktoken.get_encoding("gpt2")
            num_tokens = len(enc.encode(text))
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
    enc = tiktoken.get_encoding(encoding_name)

    all_tokens: list[int] = []
    for ds_name in dataset_names:
        tokens = _load_tokens_for(ds_name, enc)
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


def _load_tokens_for(ds_name: str, enc: tiktoken.Encoding) -> list[int]:
    """Load tokenized data for a single dataset name."""
    ds_path = DATASETS_DIR / ds_name

    # Case 1: it's a directory with data.txt
    if ds_path.is_dir():
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
        return enc.encode(text)

    # Case 2: it's a plain file in the datasets dir
    for ext in (".txt", ".json", ".jsonl", ".csv", ".parquet"):
        file_path = DATASETS_DIR / f"{ds_name}{ext}"
        if file_path.exists():
            text = file_path.read_text(encoding="utf-8")
            return enc.encode(text)

    raise FileNotFoundError(f"Dataset '{ds_name}' not found in {DATASETS_DIR}")


def delete_dataset(ds_name: str) -> bool:
    """Delete a dataset from the local storage. Returns True if deleted."""
    import shutil
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
