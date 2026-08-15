"""Isolated legacy PyTorch checkpoint reader for Native IR migration.

This module is intentionally executable as a script and has no imports from
``neuralfn``.  The parent migration process validates the graph first, then
starts this worker with Python isolated mode.  The worker uses PyTorch's
restricted ``weights_only`` unpickler and emits only JSON plus a raw tensor
bundle for the parent to validate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping


_BUNDLE_FORMAT = "neuralfn.raw_tensor_bundle.v1"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_tensor_name(source_name: str) -> str:
    components = [component for component in re.split(r"[./]+", source_name) if component]
    safe = [re.sub(r"[^A-Za-z0-9_-]+", "_", component) for component in components]
    return "parameters/" + "/".join(safe)


def _load_checkpoint(torch: Any, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        return dict(checkpoint["state_dict"]), dict(checkpoint.get("checkpoint_metadata", {}) or {})
    if isinstance(checkpoint, dict):
        return dict(checkpoint), {}
    raise TypeError(
        f"Unsupported checkpoint payload type {type(checkpoint).__name__!r}; "
        "expected a state_dict mapping."
    )


def convert(input_path: Path, output_dir: Path) -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised by parent diagnostics
        raise RuntimeError(
            "Migrating legacy .pt weights requires PyTorch in the migration environment."
        ) from exc

    state_dict, metadata = _load_checkpoint(torch, input_path)
    output_dir.mkdir(mode=stat.S_IRWXU, parents=False, exist_ok=False)
    bundle_path = output_dir / "weights.bin"
    descriptors: list[dict[str, Any]] = []
    native_names: set[str] = set()
    offset = 0
    with bundle_path.open("xb") as bundle:
        for source_name, value in sorted(state_dict.items()):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Checkpoint entry {source_name!r} is not a tensor.")
            tensor = value.detach().cpu().contiguous()
            if tensor.layout != torch.strided:
                raise TypeError(
                    f"Checkpoint tensor {source_name!r} has unsupported layout {tensor.layout}."
                )
            # Viewing through uint8 avoids NumPy dtype limitations (notably bfloat16).
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
            padding = (-offset) % 64
            if padding:
                bundle.write(b"\x00" * padding)
                offset += padding
            native_name = _native_tensor_name(str(source_name))
            if native_name in native_names:
                raise ValueError(
                    f"Tensor name collision after native normalization: {native_name}"
                )
            native_names.add(native_name)
            bundle.write(raw)
            descriptors.append(
                {
                    "name": native_name,
                    "source_name": str(source_name),
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "shape": [int(dim) for dim in tensor.shape],
                    "offset": offset,
                    "nbytes": len(raw),
                    "sha256": _sha256_bytes(raw),
                    "role": "parameter",
                    "byte_order": sys.byteorder,
                }
            )
            offset += len(raw)
    bundle_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    descriptor = {
        "schema": "neuralfn.pt_migration_worker_result",
        "version": 1,
        "tensors": descriptors,
        "checkpoint": {
            "source_path": str(input_path.resolve()),
            "source_format": "torch.pt",
            "source_sha256": _sha256_file(input_path),
            "target_format": _BUNDLE_FORMAT,
            "target_file": "weights.bin",
            "target_sha256": _sha256_file(bundle_path),
            "target_nbytes": bundle_path.stat().st_size,
            "metadata": _json_safe(metadata),
            "restricted_unpickler": True,
            "isolated_worker": True,
        },
    }
    descriptor_path = output_dir / "result.json"
    with descriptor_path.open("x", encoding="utf-8") as handle:
        json.dump(descriptor, handle, indent=2, sort_keys=True)
        handle.write("\n")
    descriptor_path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    convert(args.input.resolve(strict=True), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
