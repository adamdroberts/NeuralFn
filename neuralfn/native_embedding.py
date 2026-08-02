from __future__ import annotations

"""Dataset preparation and executable discovery for native text embeddings.

The training loop is C++ and has no Python/Torch dependency.  This module is the
small preparation boundary which turns user-facing tabular datasets into the
indexed, numeric stream consumed by that loop.
"""

import csv
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Iterable, Iterator, Mapping, Sequence


EMBEDDING_DATA_VERSION = 1
SUPPORTED_OBJECTIVES = {"raw", "retrieval", "similarity", "class"}
SUPPORTED_FILE_FORMATS = {"txt", "jsonl", "json", "csv", "parquet", "hf"}


def stable_token_id(token: str, vocab_size: int) -> int:
    """Return the native trainer's stable FNV-1a token id."""
    if vocab_size < 4:
        raise ValueError("embedding vocab size must be at least 4")
    value = 2166136261
    for byte in token.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return 3 + value % (vocab_size - 3)


def tokenize_embedding_text(text: object, *, vocab_size: int, max_tokens: int) -> list[int]:
    tokens = [stable_token_id(piece, vocab_size) for piece in str(text).lower().split()]
    return tokens[:max_tokens] or [2]


def _json_records(path: Path) -> Iterator[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("data", payload.get("records", [payload]))
    else:
        raise ValueError(f"{path}: JSON dataset must be an object or array")
    if not isinstance(records, list):
        raise ValueError(f"{path}: data/records must be an array")
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"{path}: record {index} is not an object")
        yield record


def _file_records(path: Path, fmt: str, split: str) -> Iterator[Mapping[str, Any]]:
    if fmt == "txt":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield {"text": line.strip()}
        return
    if fmt == "jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    raise ValueError(f"{path}:{line_number}: JSONL record must be an object")
                yield record
        return
    if fmt == "json":
        yield from _json_records(path)
        return
    if fmt == "csv":
        with path.open(encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    if fmt == "parquet":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Parquet embedding datasets require `pip install neuralfn[datasets]`") from exc
        yield from load_dataset("parquet", data_files=str(path), split="train")
        return
    if fmt == "hf":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Hugging Face embedding datasets require `pip install neuralfn[datasets]`") from exc
        yield from load_dataset(str(path), split=split)
        return
    raise ValueError(f"unsupported embedding dataset format: {fmt}")


def load_embedding_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = {"datasets": payload}
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), list):
        raise ValueError("embedding dataset manifest must contain a datasets array")
    payload = dict(payload)
    payload["_manifest_path"] = str(manifest_path)
    return payload


def _dataset_format(source: str, explicit: object) -> str:
    if explicit:
        return str(explicit).strip().lower()
    suffix = Path(source).suffix.lower().lstrip(".")
    return suffix if suffix in SUPPORTED_FILE_FORMATS else "hf"


def _field(record: Mapping[str, Any], columns: Mapping[str, Any], logical: str, aliases: Sequence[str]) -> Any:
    requested = str(columns.get(logical, "") or "").strip()
    candidates = [requested] if requested else list(aliases)
    for candidate in candidates:
        if candidate in record:
            return record[candidate]
    return None


def _ids(value: object, *, vocab_size: int, max_tokens: int) -> str:
    return ",".join(str(token) for token in tokenize_embedding_text(value, vocab_size=vocab_size, max_tokens=max_tokens))


def _float(value: object, *, field: str, context: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: {field} must be numeric") from exc


def compile_embedding_datasets(
    manifest: str | Path | Mapping[str, Any],
    output_path: str | Path,
    *,
    vocab_size: int = 32768,
    max_tokens: int = 128,
) -> dict[str, Any]:
    """Compile a multi-dataset manifest into the native numeric stream."""
    payload = load_embedding_manifest(manifest) if isinstance(manifest, (str, Path)) else dict(manifest)
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("embedding dataset manifest requires at least one dataset")
    manifest_path = Path(str(payload.get("_manifest_path", ""))) if payload.get("_manifest_path") else None
    base_dir = manifest_path.parent if manifest_path else Path.cwd()
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# nfn_embedding_indexed_v{EMBEDDING_DATA_VERSION}\tvocab={vocab_size}\tmax_tokens={max_tokens}"]
    counts: dict[str, int] = {}
    dataset_summaries: list[dict[str, Any]] = []
    for dataset_index, raw_spec in enumerate(datasets):
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"dataset entry {dataset_index} must be an object")
        spec = dict(raw_spec)
        source = str(spec.get("source", "")).strip()
        if not source:
            raise ValueError(f"dataset entry {dataset_index} is missing source")
        objective = str(spec.get("objective", "raw")).strip().lower()
        if objective not in SUPPORTED_OBJECTIVES:
            raise ValueError(f"dataset entry {dataset_index}: unsupported objective {objective!r}")
        fmt = _dataset_format(source, spec.get("format"))
        if fmt not in SUPPORTED_FILE_FORMATS:
            raise ValueError(f"dataset entry {dataset_index}: unsupported format {fmt!r}")
        source_path = Path(source).expanduser()
        if fmt != "hf" and not source_path.is_absolute():
            source_path = (base_dir / source_path).resolve()
        resolved_source = str(source_path) if fmt != "hf" else source
        columns = spec.get("columns") or {}
        if not isinstance(columns, Mapping):
            raise ValueError(f"dataset entry {dataset_index}: columns must be an object")
        weight = _float(spec.get("weight", 1.0), field="weight", context=f"dataset {dataset_index}")
        loss_weight = _float(spec.get("loss_weight", 1.0), field="loss_weight", context=f"dataset {dataset_index}")
        if weight <= 0 or loss_weight < 0:
            raise ValueError(f"dataset entry {dataset_index}: weight must be > 0 and loss_weight >= 0")
        score_min = _float(spec.get("score_min", 0.0), field="score_min", context=f"dataset {dataset_index}")
        score_max = _float(spec.get("score_max", 1.0), field="score_max", context=f"dataset {dataset_index}")
        if objective == "similarity" and score_max <= score_min:
            raise ValueError(f"dataset entry {dataset_index}: score_max must exceed score_min")
        count = 0
        label_ids: dict[str, int] = {}
        for row_number, record in enumerate(_file_records(Path(resolved_source), fmt, str(spec.get("split", "train"))), start=1):
            context = f"{source}:{row_number}"
            first = second = ""
            negatives = ""
            label = -1
            score = 0.0
            if objective == "raw":
                text = _field(record, columns, "text", ("text", "sentence", "document", "content"))
                if text is None:
                    raise ValueError(f"{context}: raw objective requires a text column")
                first = _ids(text, vocab_size=vocab_size, max_tokens=max_tokens)
            elif objective == "retrieval":
                query = _field(record, columns, "query", ("query", "anchor", "question"))
                positive = _field(record, columns, "positive", ("positive", "document", "passage", "answer"))
                if query is None or positive is None:
                    raise ValueError(f"{context}: retrieval objective requires query and positive columns")
                first = _ids(query, vocab_size=vocab_size, max_tokens=max_tokens)
                second = _ids(positive, vocab_size=vocab_size, max_tokens=max_tokens)
                raw_negatives = _field(record, columns, "negatives", ("negatives", "hard_negatives", "negative"))
                if raw_negatives is not None:
                    values = raw_negatives if isinstance(raw_negatives, list) else [raw_negatives]
                    negatives = ";".join(_ids(value, vocab_size=vocab_size, max_tokens=max_tokens) for value in values)
            elif objective == "similarity":
                left = _field(record, columns, "sentence1", ("sentence1", "text1", "anchor"))
                right = _field(record, columns, "sentence2", ("sentence2", "text2", "positive"))
                raw_score = _field(record, columns, "score", ("score", "similarity", "label"))
                if left is None or right is None or raw_score is None:
                    raise ValueError(f"{context}: similarity objective requires sentence1, sentence2, and score")
                first = _ids(left, vocab_size=vocab_size, max_tokens=max_tokens)
                second = _ids(right, vocab_size=vocab_size, max_tokens=max_tokens)
                score = 2.0 * (_float(raw_score, field="score", context=context) - score_min) / (score_max - score_min) - 1.0
                score = max(-1.0, min(1.0, score))
            else:
                text = _field(record, columns, "text", ("text", "sentence", "document"))
                raw_label = _field(record, columns, "label", ("label", "class", "category"))
                if text is None or raw_label is None:
                    raise ValueError(f"{context}: class objective requires text and label columns")
                first = _ids(text, vocab_size=vocab_size, max_tokens=max_tokens)
                label_key = str(raw_label)
                label = label_ids.setdefault(label_key, len(label_ids))
            lines.append(
                "\t".join(
                    (str(dataset_index), objective, f"{weight:.9g}", f"{loss_weight:.9g}", str(label), f"{score:.9g}", first, second, negatives)
                )
            )
            count += 1
        if count == 0:
            raise ValueError(f"dataset entry {dataset_index}: no usable records")
        counts[objective] = counts.get(objective, 0) + count
        dataset_summaries.append({
            "index": dataset_index,
            "name": str(spec.get("name") or spec.get("topic") or f"dataset-{dataset_index}"),
            "source": source,
            "format": fmt,
            "objective": objective,
            "weight": weight,
            "loss_weight": loss_weight,
            "records": count,
        })
    encoded = ("\n".join(lines) + "\n").encode("utf-8")
    destination.write_bytes(encoded)
    metadata = {
        "format": f"embedding_indexed_v{EMBEDDING_DATA_VERSION}",
        "path": str(destination),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "vocab_size": vocab_size,
        "max_tokens": max_tokens,
        "records": sum(counts.values()),
        "objectives": counts,
        "datasets": dataset_summaries,
    }
    destination.with_suffix(destination.suffix + ".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def inline_embedding_manifest(sources: Iterable[str], *, objective: str = "raw") -> dict[str, Any]:
    return {
        "datasets": [
            {"name": Path(source).stem or f"dataset-{index}", "source": source, "objective": objective, "weight": 1.0}
            for index, source in enumerate(sources)
        ]
    }


def resolve_native_embedding_cli(repo_root: str | Path | None = None) -> str:
    import os
    import shutil

    requested = os.environ.get("NFN_NATIVE_EMBEDDING_CLI", "").strip()
    if requested:
        return requested
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
    built = root / "build" / "nfn_embedding_native_train"
    if built.exists():
        return str(built)
    return shutil.which("nfn_embedding_native_train") or str(built)


def prepare_embedding_training_command(command: Sequence[str], *, repo_root: str | Path) -> tuple[list[str], dict[str, Any] | None]:
    """Replace user dataset arguments with a compiled stream for the C++ loop."""
    args = list(command)
    if "--embedding-data" in args or any(item.startswith("--embedding-data=") for item in args):
        return args, None

    def values(flag: str) -> list[str]:
        found: list[str] = []
        index = 1
        while index < len(args):
            item = args[index]
            if item == flag and index + 1 < len(args):
                found.append(args[index + 1])
                index += 2
            elif item.startswith(flag + "="):
                found.append(item.split("=", 1)[1])
                index += 1
            else:
                index += 1
        return found

    manifests = values("--embedding-datasets-manifest")
    sources = values("--embedding-dataset")
    if not manifests and not sources:
        raise ValueError("embedding training requires --embedding-datasets-manifest PATH or at least one --embedding-dataset PATH")
    if len(manifests) > 1 or (manifests and sources):
        raise ValueError("use one embedding dataset manifest or repeated inline datasets, not both")
    output_values = values("--output-dir")
    output_dir = Path(output_values[-1] if output_values else Path(repo_root) / "artifacts" / "embedding").expanduser().resolve()
    vocab_values = values("--embedding-vocab-size")
    max_values = values("--max-seq-len")
    vocab_size = int(vocab_values[-1]) if vocab_values else 32768
    max_tokens = int(max_values[-1]) if max_values else 128
    manifest: str | Path | Mapping[str, Any] = manifests[0] if manifests else inline_embedding_manifest(sources)
    metadata = compile_embedding_datasets(manifest, output_dir / "embedding_data.tsv", vocab_size=vocab_size, max_tokens=max_tokens)
    stripped: list[str] = [args[0]]
    index = 1
    removed = {"--embedding-datasets-manifest", "--embedding-dataset"}
    while index < len(args):
        item = args[index]
        if item in removed:
            index += 2
            continue
        if any(item.startswith(flag + "=") for flag in removed):
            index += 1
            continue
        stripped.append(item)
        index += 1
    stripped.extend(["--embedding-data", str(metadata["path"]), "--embedding-data-sha256", str(metadata["sha256"])])
    return stripped, metadata


def read_embedding_checkpoint_header(path: str | Path) -> dict[str, int]:
    checkpoint = Path(path).expanduser()
    if checkpoint.is_dir():
        checkpoint = checkpoint / "embedding_model.bin"
    with checkpoint.open("rb") as handle:
        header = handle.read(8 + 8 * 4)
    if len(header) < 40 or header[:8] != b"NFNEMB1\0":
        raise ValueError(f"{checkpoint} is not a NeuralFn embedding checkpoint")
    version, vocab_size, hidden_dim, output_dim, max_tokens, step, adapter = struct.unpack("<7I", header[8:36])
    return {
        "version": version,
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "max_tokens": max_tokens,
        "step": step,
        "adapter_type": adapter,
    }
