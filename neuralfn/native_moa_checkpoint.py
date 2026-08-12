"""Strict, Torch-free inspection for native dense GPT-2 MoA checkpoints.

The trainable tensors remain in the ordinary native dense-v5 ``model_*.bin``
file.  A sibling JSON document freezes the activation selected by the training
loop and binds those semantics to both the completed model file and the exact
source graph bytes.  This module treats every path and metadata field as
untrusted input and reuses :mod:`neuralfn.native_dense_checkpoint` as the
authoritative binary-layout validator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, Mapping

from .native_dense_checkpoint import (
    NATIVE_DENSE_GPT_CHECKPOINT_FORMAT,
    NativeDenseCheckpointInfo,
    NativeDenseCheckpointTensor,
    inspect_native_dense_checkpoint,
)


NATIVE_MOA_INFERENCE_SCHEMA = "neuralfn.native_dense_moa.inference_checkpoint"
NATIVE_MOA_INFERENCE_VERSION = 1
NATIVE_MOA_CHECKPOINT_KIND = "trained_dense_v5"
NATIVE_MOA_PRESET = "gpt2_moa"
NATIVE_MOA_CANDIDATE_ACTIVATIONS = ("gelu", "relu", "silu", "relu2")
NATIVE_MOA_INTERVAL = 50

_METADATA_NAME = re.compile(r"model_([0-9]{8})\.moa\.json\Z")
_MAX_METADATA_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class NativeMoaCheckpointInfo:
    """A validated dense-v5 model plus its immutable MoA inference semantics."""

    metadata_path: Path
    model_path: Path
    done_marker_path: Path
    source_graph_path: Path
    source_graph_filename: str
    source_graph_sha256: str
    metadata_sha256: str
    selected_activation: str
    candidate_activations: tuple[str, ...]
    interval: int
    geometry: dict[str, int]
    dense_checkpoint: NativeDenseCheckpointInfo

    @property
    def path(self) -> Path:
        return self.model_path

    @property
    def file_size(self) -> int:
        return self.dense_checkpoint.file_size

    @property
    def sha256(self) -> str:
        return self.dense_checkpoint.sha256

    @property
    def tensors(self) -> tuple[NativeDenseCheckpointTensor, ...]:
        return self.dense_checkpoint.tensors

    @property
    def max_seq_len(self) -> int:
        return self.dense_checkpoint.max_seq_len

    def checkpoint_descriptor(self, *, artifact_path: str = "model.bin") -> dict[str, Any]:
        """Return a dense-v5 descriptor extended with the strict MoA contract."""

        descriptor = self.dense_checkpoint.checkpoint_descriptor(
            artifact_path=artifact_path
        )
        metadata_artifact_path = f"{Path(artifact_path).stem}.moa.json"
        descriptor["source_metadata_path"] = str(self.metadata_path)
        descriptor["source_graph"] = {
            "filename": self.source_graph_filename,
            "sha256": self.source_graph_sha256,
            "byte_identity_verified": True,
        }
        descriptor["moa"] = {
            "schema": NATIVE_MOA_INFERENCE_SCHEMA,
            "version": NATIVE_MOA_INFERENCE_VERSION,
            "preset": NATIVE_MOA_PRESET,
            "selected_activation": self.selected_activation,
            "candidate_activations": list(self.candidate_activations),
            "interval": self.interval,
            "source_graph_sha256": self.source_graph_sha256,
            "metadata_artifact_path": metadata_artifact_path,
            "metadata_nbytes": self.metadata_path.stat().st_size,
            "metadata_sha256": self.metadata_sha256,
        }
        return descriptor


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Native MoA checkpoint field {field} must be an object.")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise ValueError(f"Native MoA checkpoint field {field} is not canonical.")


def _integer(value: Any, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"Native MoA checkpoint field {field} must be an integer >= {minimum}."
        )
    return value


def _lower_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"Native MoA checkpoint field {field} must be a lowercase SHA-256 digest."
        )
    return value


def _normalized_id(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _safe_relative_file(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(
            f"Native MoA checkpoint field {field} must be a safe relative path."
        )
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError(
            f"Native MoA checkpoint field {field} must be a safe relative path."
        )
    try:
        candidate = (root / Path(*posix.parts)).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"Native MoA checkpoint {field} does not exist.") from exc
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError(f"Native MoA checkpoint field {field} escapes its artifact root.")
    return candidate


def _validate_model_semantics(
    model: Mapping[str, Any],
    *,
    geometry: Mapping[str, int],
    selected_activation: str,
    candidate_activations: tuple[str, ...],
    interval: int,
) -> None:
    if (
        _normalized_id(model.get("family")) != "gpt2"
        or _normalized_id(model.get("backbone")) != "gpt2"
        or _normalized_id(model.get("family_class")) != "autoregressive_transformer"
        or _normalized_id(model.get("objective")) != "ar"
    ):
        raise ValueError(
            "Native MoA checkpoints require an exact GPT-2 autoregressive graph model."
        )

    spec = _mapping(model.get("template_spec"), "model.template_spec")
    template = _mapping(spec.get("template"), "model.template_spec.template")
    block = _mapping(spec.get("block_spec"), "model.template_spec.block_spec")
    exact_template = {
        "objective": "ar",
        "backbone": "gpt2",
        "sparsity": "dense",
        "router_mode": "none",
        "compression": "none",
        "adapter": "none",
        "runtime": "eager",
    }
    for field, expected in exact_template.items():
        if _normalized_id(template.get(field)) != expected:
            raise ValueError(
                f"Native MoA inference requires template.{field}={expected!r}."
            )

    exact_block = {
        "family": "gpt2",
        "norm_type": "layernorm",
        "mlp_type": "gelu",
        "pos_encoding": "absolute",
        "attention_backend": "sdpa",
        "attention_variant": "dense",
        "compression": "none",
        "adapter_type": "none",
        "residual_type": "add",
        "activation_mode": "moa",
    }
    for field, expected in exact_block.items():
        if _normalized_id(block.get(field)) != expected:
            raise ValueError(
                f"Native MoA inference requires block_spec.{field}={expected!r}."
            )
    for field in ("is_causal", "linear_bias"):
        if block.get(field) is not True:
            raise ValueError(f"Native MoA inference requires block_spec.{field}=true.")
    if block.get("use_qk_norm") is not False:
        raise ValueError("Native MoA inference requires block_spec.use_qk_norm=false.")

    numeric_exact = {
        "dropout_p": 0.0,
        "mlp_multiplier": 4.0,
    }
    for field, expected in numeric_exact.items():
        value = block.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Native MoA inference requires numeric block_spec.{field}.")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed != expected:
            raise ValueError(
                f"Native MoA inference requires block_spec.{field}={expected!r}."
            )
    for field in ("z_loss_coef", "logit_softcap"):
        value = spec.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Native MoA inference requires numeric template_spec.{field}.")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed != 0.0:
            raise ValueError(f"Native MoA inference requires template_spec.{field}=0.")
    if spec.get("tie_embeddings") is not True:
        raise ValueError("Native MoA inference requires tied token/output embeddings.")

    graph_candidates = block.get("moa_activations")
    if not isinstance(graph_candidates, list) or tuple(graph_candidates) != candidate_activations:
        raise ValueError("Graph MoA candidate activations do not match checkpoint metadata.")
    if candidate_activations != NATIVE_MOA_CANDIDATE_ACTIVATIONS:
        raise ValueError("Native MoA candidate activations are not canonical.")
    graph_interval = block.get("moa_interval")
    if (
        isinstance(graph_interval, bool)
        or not isinstance(graph_interval, int)
        or graph_interval <= 0
        or graph_interval != interval
    ):
        raise ValueError("Graph MoA interval does not match checkpoint metadata.")
    if selected_activation not in candidate_activations:
        raise ValueError("Native MoA selected activation is not a canonical candidate.")

    expected_geometry = {
        "model_dim": geometry["model_dim"],
        "num_layers": geometry["num_layers"],
        "vocab_size": geometry["vocab_size"],
    }
    for field, expected in expected_geometry.items():
        value = spec.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError(
                f"Graph {field} does not match native MoA checkpoint geometry."
            )
    graph_padded_vocab = spec.get("padded_vocab_size")
    if graph_padded_vocab not in (None, "") and graph_padded_vocab != geometry["padded_vocab_size"]:
        raise ValueError("Graph padded_vocab_size does not match native MoA checkpoint geometry.")
    graph_heads = block.get("num_heads")
    if isinstance(graph_heads, bool) or graph_heads != geometry["num_heads"]:
        raise ValueError("Graph num_heads does not match native MoA checkpoint geometry.")
    graph_kv_heads = block.get("num_kv_heads")
    if graph_kv_heads not in (None, "", geometry["num_heads"]):
        raise ValueError("Native MoA inference requires MHA, not grouped-query attention.")
    for field in ("max_seq_len", "seq_len", "context_window"):
        graph_context = spec.get(field)
        if graph_context not in (None, "") and graph_context != geometry["max_seq_len"]:
            raise ValueError("Graph context length does not match native MoA checkpoint geometry.")


def inspect_native_moa_checkpoint(
    path: str | Path,
    *,
    source_graph_path: str | Path,
    model: Mapping[str, Any],
) -> NativeMoaCheckpointInfo:
    """Validate a completed, graph-bound native dense GPT-2 MoA checkpoint."""

    try:
        metadata_path = Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("Native MoA checkpoint metadata does not exist.") from exc
    if not metadata_path.is_file():
        raise ValueError("Native MoA checkpoint metadata is not a regular file.")
    match = _METADATA_NAME.fullmatch(metadata_path.name)
    if match is None:
        raise ValueError("Native MoA checkpoint metadata must be named model_XXXXXXXX.moa.json.")
    if metadata_path.stat().st_size > _MAX_METADATA_BYTES:
        raise ValueError("Native MoA checkpoint metadata exceeds the size limit.")
    metadata_bytes = metadata_path.read_bytes()
    metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    try:
        payload = json.loads(
            metadata_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Native MoA checkpoint metadata is not valid UTF-8 JSON.") from exc
    payload = _mapping(payload, "root")
    _exact_keys(
        payload,
        {
            "schema",
            "version",
            "preset",
            "checkpoint_kind",
            "model",
            "done_marker",
            "source_graph",
            "selection",
            "geometry",
        },
        "root",
    )
    if payload.get("schema") != NATIVE_MOA_INFERENCE_SCHEMA:
        raise ValueError("Native MoA checkpoint has an unknown schema.")
    if (
        isinstance(payload.get("version"), bool)
        or payload.get("version") != NATIVE_MOA_INFERENCE_VERSION
    ):
        raise ValueError("Native MoA checkpoint has an unsupported version.")
    if payload.get("preset") != NATIVE_MOA_PRESET:
        raise ValueError("Native MoA checkpoint requires preset='gpt2_moa'.")
    if payload.get("checkpoint_kind") != NATIVE_MOA_CHECKPOINT_KIND:
        raise ValueError("Native MoA checkpoint kind is not a trained dense-v5 model.")

    root = metadata_path.parent.resolve()
    step = match.group(1)
    model_payload = _mapping(payload.get("model"), "model")
    _exact_keys(model_payload, {"path", "format", "nbytes", "sha256"}, "model")
    if model_payload.get("format") != NATIVE_DENSE_GPT_CHECKPOINT_FORMAT:
        raise ValueError("Native MoA checkpoint model is not native dense GPT v5.")
    if model_payload.get("path") != f"model_{step}.bin":
        raise ValueError("Native MoA checkpoint declares the wrong model filename.")
    model_path = _safe_relative_file(root, model_payload.get("path"), "model.path")
    declared_nbytes = _integer(model_payload.get("nbytes"), "model.nbytes")
    declared_sha256 = _lower_sha256(model_payload.get("sha256"), "model.sha256")

    if payload.get("done_marker") != f"DONE_{step}":
        raise ValueError("Native MoA checkpoint declares the wrong DONE marker.")
    done_marker_path = _safe_relative_file(root, payload.get("done_marker"), "done_marker")
    if done_marker_path.stat().st_size != 0:
        raise ValueError("Native MoA checkpoint DONE marker must be empty.")

    try:
        graph_path = Path(source_graph_path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("Native MoA source graph does not exist.") from exc
    if not graph_path.is_file():
        raise ValueError("Native MoA source graph is not a regular file.")
    graph_bytes = graph_path.read_bytes()
    actual_graph_sha256 = hashlib.sha256(graph_bytes).hexdigest()
    try:
        graph_payload = json.loads(
            graph_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Native MoA source graph is not valid UTF-8 JSON.") from exc
    graph_payload = _mapping(graph_payload, "source_graph.root")
    torch_config = _mapping(
        graph_payload.get("torch_config"), "source_graph.torch_config"
    )
    serialized_spec = _mapping(
        torch_config.get("template_spec"), "source_graph.torch_config.template_spec"
    )
    model_spec = _mapping(model.get("template_spec"), "model.template_spec")
    if dict(serialized_spec) != dict(model_spec):
        raise ValueError("Native MoA parsed model does not match the bound source graph bytes.")
    graph_name = graph_payload.get("name")
    if not isinstance(graph_name, str) or graph_name != model.get("name"):
        raise ValueError("Native MoA parsed model name does not match the bound source graph.")
    source_graph = _mapping(payload.get("source_graph"), "source_graph")
    _exact_keys(
        source_graph,
        {"filename", "sha256", "byte_identity_verified"},
        "source_graph",
    )
    declared_graph_filename = source_graph.get("filename")
    if (
        not isinstance(declared_graph_filename, str)
        or not declared_graph_filename
        or declared_graph_filename in {".", ".."}
        or "/" in declared_graph_filename
        or "\\" in declared_graph_filename
        or PurePosixPath(declared_graph_filename).name != declared_graph_filename
        or PureWindowsPath(declared_graph_filename).name != declared_graph_filename
    ):
        raise ValueError("Native MoA checkpoint source-graph filename is unsafe.")
    if source_graph.get("byte_identity_verified") is not True:
        raise ValueError("Native MoA checkpoint source graph is not byte-identity verified.")
    declared_graph_sha256 = _lower_sha256(
        source_graph.get("sha256"), "source_graph.sha256"
    )
    if declared_graph_sha256 != actual_graph_sha256:
        raise ValueError("Native MoA source graph SHA-256 does not match metadata.")

    selection = _mapping(payload.get("selection"), "selection")
    _exact_keys(selection, {"activation", "candidates", "interval"}, "selection")
    candidates_raw = selection.get("candidates")
    if not isinstance(candidates_raw, list) or any(
        not isinstance(candidate, str) for candidate in candidates_raw
    ):
        raise ValueError("Native MoA checkpoint candidates must be a JSON string array.")
    candidates = tuple(candidates_raw)
    if candidates != NATIVE_MOA_CANDIDATE_ACTIVATIONS:
        raise ValueError("Native MoA checkpoint candidate list is not canonical.")
    activation = selection.get("activation")
    if not isinstance(activation, str) or activation not in candidates:
        raise ValueError("Native MoA checkpoint selected activation is not canonical.")
    interval = _integer(selection.get("interval"), "selection.interval")

    geometry_payload = _mapping(payload.get("geometry"), "geometry")
    geometry_fields = {
        "max_seq_len",
        "vocab_size",
        "padded_vocab_size",
        "num_layers",
        "model_dim",
        "num_heads",
        "head_dim",
        "mlp_hidden_dim",
    }
    _exact_keys(geometry_payload, geometry_fields, "geometry")
    geometry = {
        field: _integer(geometry_payload.get(field), f"geometry.{field}")
        for field in geometry_fields
    }
    if geometry["padded_vocab_size"] < geometry["vocab_size"]:
        raise ValueError("Native MoA padded vocabulary is smaller than the vocabulary.")
    if geometry["model_dim"] != geometry["num_heads"] * geometry["head_dim"]:
        raise ValueError("Native MoA head geometry does not reconstruct model_dim.")
    if geometry["mlp_hidden_dim"] != 4 * geometry["model_dim"]:
        raise ValueError("Native MoA checkpoint requires an exact 4x dense MLP width.")

    dense = inspect_native_dense_checkpoint(model_path)
    if dense.file_size != declared_nbytes:
        raise ValueError("Native MoA model size does not match metadata.")
    if dense.sha256 != declared_sha256:
        raise ValueError("Native MoA model SHA-256 does not match metadata.")
    dense_geometry = {
        "max_seq_len": dense.max_seq_len,
        "vocab_size": dense.vocab_size,
        "padded_vocab_size": dense.padded_vocab_size,
        "num_layers": dense.num_layers,
        "model_dim": dense.channels,
        "num_heads": dense.num_heads,
        "head_dim": dense.channels // dense.num_heads,
        "mlp_hidden_dim": 4 * dense.channels,
    }
    if geometry != dense_geometry:
        raise ValueError("Native MoA metadata geometry does not match the dense-v5 model.")

    _validate_model_semantics(
        model,
        geometry=geometry,
        selected_activation=activation,
        candidate_activations=candidates,
        interval=interval,
    )

    return NativeMoaCheckpointInfo(
        metadata_path=metadata_path,
        model_path=model_path,
        done_marker_path=done_marker_path,
        source_graph_path=graph_path,
        source_graph_filename=declared_graph_filename,
        source_graph_sha256=actual_graph_sha256,
        metadata_sha256=metadata_sha256,
        selected_activation=activation,
        candidate_activations=candidates,
        interval=interval,
        geometry=geometry,
        dense_checkpoint=dense,
    )


__all__ = [
    "NATIVE_MOA_CANDIDATE_ACTIVATIONS",
    "NATIVE_MOA_CHECKPOINT_KIND",
    "NATIVE_MOA_INFERENCE_SCHEMA",
    "NATIVE_MOA_INFERENCE_VERSION",
    "NATIVE_MOA_INTERVAL",
    "NATIVE_MOA_PRESET",
    "NativeMoaCheckpointInfo",
    "inspect_native_moa_checkpoint",
]
