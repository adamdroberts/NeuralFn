"""Source-bound Torch runtime for migrated native dense GPT-2 MoA artifacts.

The authored ``gpt2_moa`` graph intentionally contains an ordinary GELU MLP.
Training probes four weight-compatible pointwise activations and commits the
winner only in the completed checkpoint's sibling metadata.  Migration carries
that selection into the Native Execution manifest without rewriting the source
graph.  This module applies the resulting semantic overlay at runtime while
requiring both the original graph bytes and the dense-v5 checkpoint bytes to
match their migration fingerprints.

This is a parity/debug runtime, not the resident serving path.  It imports the
native dense-v5 tensors into ``CompiledTorchGraph`` and replaces only the
canonical ``model/block_N/mlp/gelu`` stages.  The source ``NeuronGraph`` and its
serialized bytes remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .graph import NeuronGraph
from .native_dense_checkpoint import (
    NATIVE_DENSE_GPT_CHECKPOINT_FORMAT,
    NativeDenseCheckpointInfo,
    inspect_native_dense_checkpoint,
)
from .native_ir import (
    NATIVE_EXECUTION_MANIFEST_SCHEMA,
    NATIVE_EXECUTION_MANIFEST_VERSION,
    NativeExecutionManifest,
)
from .native_moa_checkpoint import (
    NATIVE_MOA_CANDIDATE_ACTIVATIONS,
    NATIVE_MOA_INFERENCE_SCHEMA,
    NATIVE_MOA_INFERENCE_VERSION,
    NATIVE_MOA_PRESET,
    _validate_model_semantics,
)
from .torch_backend import CompiledTorchGraph


NATIVE_MOA_GRAPH_RUNTIME_PROFILE = "source-bound-gpt2-moa-selected-activation-v1"
_SOURCE_MODEL_NAME = re.compile(r"model_([0-9]{8})\.bin\Z")


class NativeMoaGraphRuntimeError(ValueError):
    """Raised when graph/checkpoint bytes cannot prove the MoA runtime overlay."""


@dataclass(frozen=True, slots=True)
class NativeMoaGraphRuntimeBinding:
    """Immutable provenance for one selected-activation graph runtime."""

    source_graph_sha256: str
    checkpoint_sha256: str
    selected_activation: str
    candidate_activations: tuple[str, ...]
    interval: int
    activation_node_paths: tuple[str, ...]
    profile: str = NATIVE_MOA_GRAPH_RUNTIME_PROFILE


@dataclass(frozen=True, slots=True)
class NativeMoaGraphRuntime:
    """Compiled graph plus the exact artifact semantics bound into it."""

    graph: NeuronGraph
    compiled: CompiledTorchGraph
    manifest: NativeExecutionManifest
    checkpoint: NativeDenseCheckpointInfo
    binding: NativeMoaGraphRuntimeBinding

    def _validate_inputs(self, flat_inputs: tuple[Tensor, ...]) -> None:
        if not flat_inputs:
            return
        token_ids = flat_inputs[0]
        if isinstance(token_ids, Tensor) and token_ids.ndim >= 1:
            sequence_length = int(token_ids.shape[-1])
            if sequence_length > self.checkpoint.max_seq_len:
                raise NativeMoaGraphRuntimeError(
                    "Native MoA graph runtime input exceeds the dense-v5 checkpoint "
                    f"context limit of {self.checkpoint.max_seq_len} tokens."
                )

    def forward(self, *flat_inputs: Tensor) -> tuple[Tensor, ...]:
        """Execute the selected-activation graph within its native context limit."""

        self._validate_inputs(flat_inputs)
        return self.compiled(*flat_inputs)

    def trace(
        self, *flat_inputs: Tensor
    ) -> tuple[tuple[Tensor, ...], dict[str, tuple[Tensor, ...]]]:
        """Execute and return the normal ``CompiledTorchGraph`` node trace."""

        self._validate_inputs(flat_inputs)
        return self.compiled.trace(*flat_inputs)


class _SelectedMoaActivationStage(nn.Module):
    """Parameter-free pointwise activation selected by native MoA training."""

    def __init__(self, activation: str) -> None:
        super().__init__()
        if activation not in NATIVE_MOA_CANDIDATE_ACTIVATIONS:
            raise NativeMoaGraphRuntimeError(
                f"Unsupported native MoA selected activation {activation!r}."
            )
        self.activation = activation

    def forward(self, value: Tensor) -> Tensor:
        if self.activation == "gelu":
            # The native MoA kernel uses the canonical tanh GELU approximation;
            # ordinary dense-v5 graph execution continues to use exact GELU.
            return F.gelu(value, approximate="tanh")
        if self.activation == "relu":
            return F.relu(value)
        if self.activation == "silu":
            return F.silu(value)
        positive = F.relu(value)
        return positive * positive


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NativeMoaGraphRuntimeError(
                f"Native MoA graph runtime JSON contains duplicate object key {key!r}."
            )
        result[key] = value
    return result


def _strict_json_object(payload: bytes, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except NativeMoaGraphRuntimeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeMoaGraphRuntimeError(
            f"Native MoA graph runtime field {field} is not valid UTF-8 JSON."
        ) from exc
    if not isinstance(value, dict):
        raise NativeMoaGraphRuntimeError(
            f"Native MoA graph runtime field {field} must be a JSON object."
        )
    return value


def _lower_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NativeMoaGraphRuntimeError(
            f"Native MoA graph runtime field {field} must be a lowercase SHA-256 digest."
        )
    return value


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeMoaGraphRuntimeError(
            f"Native MoA graph runtime field {field} must be an object."
        )
    return value


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NativeMoaGraphRuntimeError(
            f"Native MoA graph runtime field {field} must be a positive integer."
        )
    return value


def _safe_artifact_file(root: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise NativeMoaGraphRuntimeError(
            f"Native MoA graph runtime {field} must be a safe relative file."
        )
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or len(posix.parts) != 1
        or posix.name in {"", ".", ".."}
    ):
        raise NativeMoaGraphRuntimeError(
            f"Native MoA graph runtime {field} must be a safe relative file."
        )
    try:
        candidate = (root / posix.name).resolve(strict=True)
    except FileNotFoundError as exc:
        raise NativeMoaGraphRuntimeError(
            f"Native MoA graph runtime {field} does not exist."
        ) from exc
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise NativeMoaGraphRuntimeError(
            f"Native MoA graph runtime {field} escapes its artifact root."
        )
    return candidate


def _load_manifest(artifact: str | Path) -> tuple[Path, NativeExecutionManifest, dict[str, Any]]:
    requested = Path(artifact).expanduser().resolve()
    if requested.is_dir():
        root = requested
        path = root / "native-execution-manifest.json"
    else:
        root = requested.parent
        path = requested
    if not path.is_file():
        raise NativeMoaGraphRuntimeError(
            f"Native Execution manifest does not exist: {path}"
        )
    raw = _strict_json_object(path.read_bytes(), "manifest")
    version = raw.get("version")
    if (
        raw.get("schema") != NATIVE_EXECUTION_MANIFEST_SCHEMA
        or isinstance(version, bool)
        or version != NATIVE_EXECUTION_MANIFEST_VERSION
    ):
        raise NativeMoaGraphRuntimeError(
            "Native MoA graph runtime requires Native Execution Manifest version 1."
        )
    try:
        manifest = NativeExecutionManifest.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeMoaGraphRuntimeError(
            "Native Execution manifest is incomplete or malformed."
        ) from exc
    return root, manifest, raw


def _validate_source_graph(
    source_graph_path: str | Path,
    manifest: NativeExecutionManifest,
) -> tuple[NeuronGraph, dict[str, Any], str]:
    try:
        path = Path(source_graph_path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise NativeMoaGraphRuntimeError("Bound native MoA source graph does not exist.") from exc
    if not path.is_file():
        raise NativeMoaGraphRuntimeError("Bound native MoA source graph is not a regular file.")
    payload = path.read_bytes()
    digest = _sha256_bytes(payload)
    manifest_digest = _lower_sha256(
        manifest.source_graph.get("sha256"), "source_graph.sha256"
    )
    if digest != manifest_digest:
        raise NativeMoaGraphRuntimeError(
            "Native MoA source graph SHA-256 does not match the migrated artifact."
        )
    if manifest.source_graph.get("serialization_changed") is not False:
        raise NativeMoaGraphRuntimeError(
            "Native MoA graph runtime requires byte-identical source serialization."
        )
    raw_graph = _strict_json_object(payload, "source graph")
    try:
        graph = NeuronGraph.from_dict(raw_graph)
        graph.validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeMoaGraphRuntimeError("Bound native MoA source graph is invalid.") from exc
    return graph, raw_graph, digest


def _validate_metadata_artifact(
    *,
    root: Path,
    moa: Mapping[str, Any],
    graph_digest: str,
    checkpoint: NativeDenseCheckpointInfo,
) -> tuple[str, int]:
    metadata_path = _safe_artifact_file(
        root,
        moa.get("metadata_artifact_path"),
        field="checkpoint.moa.metadata_artifact_path",
    )
    metadata_nbytes = _positive_integer(
        moa.get("metadata_nbytes"), "checkpoint.moa.metadata_nbytes"
    )
    metadata_sha256 = _lower_sha256(
        moa.get("metadata_sha256"), "checkpoint.moa.metadata_sha256"
    )
    metadata_bytes = metadata_path.read_bytes()
    if len(metadata_bytes) != metadata_nbytes or _sha256_bytes(metadata_bytes) != metadata_sha256:
        raise NativeMoaGraphRuntimeError(
            "Native MoA metadata artifact bytes do not match the migrated manifest."
        )
    metadata = _strict_json_object(metadata_bytes, "checkpoint MoA metadata")
    if set(metadata) != {
        "schema",
        "version",
        "preset",
        "checkpoint_kind",
        "model",
        "done_marker",
        "source_graph",
        "selection",
        "geometry",
    }:
        raise NativeMoaGraphRuntimeError(
            "Native MoA metadata artifact root is not canonical."
        )
    metadata_version = metadata.get("version")
    if (
        metadata.get("schema") != NATIVE_MOA_INFERENCE_SCHEMA
        or isinstance(metadata_version, bool)
        or metadata_version != NATIVE_MOA_INFERENCE_VERSION
        or metadata.get("preset") != NATIVE_MOA_PRESET
        or metadata.get("checkpoint_kind") != "trained_dense_v5"
    ):
        raise NativeMoaGraphRuntimeError(
            "Native MoA metadata artifact schema/profile is unsupported."
        )

    source_model = _mapping(metadata.get("model"), "checkpoint metadata.model")
    if set(source_model) != {"path", "format", "nbytes", "sha256"}:
        raise NativeMoaGraphRuntimeError(
            "Native MoA metadata artifact model descriptor is not canonical."
        )
    source_model_name = source_model.get("path")
    source_model_match = (
        _SOURCE_MODEL_NAME.fullmatch(source_model_name)
        if isinstance(source_model_name, str)
        else None
    )
    if (
        source_model_match is None
        or source_model.get("format") != NATIVE_DENSE_GPT_CHECKPOINT_FORMAT
        or source_model.get("nbytes") != checkpoint.file_size
        or _lower_sha256(
            source_model.get("sha256"), "checkpoint metadata.model.sha256"
        )
        != checkpoint.sha256
    ):
        raise NativeMoaGraphRuntimeError(
            "Native MoA metadata artifact does not describe the migrated dense-v5 bytes."
        )
    if metadata.get("done_marker") != f"DONE_{source_model_match.group(1)}":
        raise NativeMoaGraphRuntimeError(
            "Native MoA metadata artifact declares the wrong completion marker."
        )

    source_graph = _mapping(
        metadata.get("source_graph"), "checkpoint metadata.source_graph"
    )
    if set(source_graph) != {"filename", "sha256", "byte_identity_verified"}:
        raise NativeMoaGraphRuntimeError(
            "Native MoA metadata source-graph descriptor is not canonical."
        )
    source_filename = source_graph.get("filename")
    if (
        not isinstance(source_filename, str)
        or not source_filename
        or source_filename in {".", ".."}
        or PurePosixPath(source_filename).name != source_filename
        or PureWindowsPath(source_filename).name != source_filename
        or _lower_sha256(
            source_graph.get("sha256"), "checkpoint metadata.source_graph.sha256"
        )
        != graph_digest
        or source_graph.get("byte_identity_verified") is not True
    ):
        raise NativeMoaGraphRuntimeError(
            "Native MoA metadata artifact is not bound to the exact source graph bytes."
        )

    selection = _mapping(metadata.get("selection"), "checkpoint metadata.selection")
    if set(selection) != {"activation", "candidates", "interval"}:
        raise NativeMoaGraphRuntimeError(
            "Native MoA metadata artifact selection is not canonical."
        )
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or tuple(candidates) != NATIVE_MOA_CANDIDATE_ACTIVATIONS:
        raise NativeMoaGraphRuntimeError(
            "Native MoA metadata artifact candidate activations are not canonical."
        )
    selected = selection.get("activation")
    if not isinstance(selected, str) or selected not in NATIVE_MOA_CANDIDATE_ACTIVATIONS:
        raise NativeMoaGraphRuntimeError(
            "Native MoA metadata artifact selected activation is not canonical."
        )
    interval = _positive_integer(
        selection.get("interval"), "checkpoint metadata.selection.interval"
    )

    geometry = _mapping(metadata.get("geometry"), "checkpoint metadata.geometry")
    expected_geometry = {
        "max_seq_len": checkpoint.max_seq_len,
        "vocab_size": checkpoint.vocab_size,
        "padded_vocab_size": checkpoint.padded_vocab_size,
        "num_layers": checkpoint.num_layers,
        "model_dim": checkpoint.channels,
        "num_heads": checkpoint.num_heads,
        "head_dim": checkpoint.channels // checkpoint.num_heads,
        "mlp_hidden_dim": checkpoint.channels * 4,
    }
    if set(geometry) != set(expected_geometry):
        raise NativeMoaGraphRuntimeError(
            "Native MoA metadata artifact geometry is not canonical."
        )
    for field, expected in expected_geometry.items():
        value = geometry.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise NativeMoaGraphRuntimeError(
                f"Native MoA metadata artifact geometry.{field} does not match "
                "the dense-v5 checkpoint."
            )
    return selected, interval


def _validate_moa_contract(
    *,
    root: Path,
    raw_graph: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
    graph_digest: str,
    checkpoint: NativeDenseCheckpointInfo,
) -> tuple[str, int]:
    model = _mapping(raw_manifest.get("model"), "model")
    capabilities = _mapping(raw_manifest.get("capabilities"), "capabilities")
    if (
        capabilities.get("native_inference") is not True
        or capabilities.get("resident_inference") is not True
    ):
        raise NativeMoaGraphRuntimeError(
            "Native MoA graph runtime requires a migration artifact with proved "
            "native resident inference."
        )
    graph_torch_config = _mapping(raw_graph.get("torch_config"), "source_graph.torch_config")
    graph_spec = _mapping(
        graph_torch_config.get("template_spec"),
        "source_graph.torch_config.template_spec",
    )
    model_spec = _mapping(model.get("template_spec"), "model.template_spec")
    if dict(graph_spec) != dict(model_spec) or raw_graph.get("name") != model.get("name"):
        raise NativeMoaGraphRuntimeError(
            "Native MoA manifest model does not match the bound source graph."
        )

    checkpoint_payload = _mapping(raw_manifest.get("checkpoint"), "checkpoint")
    provenance = _mapping(checkpoint_payload.get("source_graph"), "checkpoint.source_graph")
    if set(provenance) != {"filename", "sha256", "byte_identity_verified"}:
        raise NativeMoaGraphRuntimeError(
            "Native MoA checkpoint source-graph provenance is not canonical."
        )
    if (
        _lower_sha256(provenance.get("sha256"), "checkpoint.source_graph.sha256")
        != graph_digest
        or provenance.get("byte_identity_verified") is not True
    ):
        raise NativeMoaGraphRuntimeError(
            "Native MoA checkpoint is not bound to the exact source graph bytes."
        )
    filename = provenance.get("filename")
    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or PurePosixPath(filename).name != filename
        or PureWindowsPath(filename).name != filename
    ):
        raise NativeMoaGraphRuntimeError(
            "Native MoA checkpoint source-graph filename is not canonical."
        )

    moa = _mapping(checkpoint_payload.get("moa"), "checkpoint.moa")
    if set(moa) != {
        "schema",
        "version",
        "preset",
        "selected_activation",
        "candidate_activations",
        "interval",
        "source_graph_sha256",
        "metadata_artifact_path",
        "metadata_nbytes",
        "metadata_sha256",
    }:
        raise NativeMoaGraphRuntimeError("Native MoA checkpoint selection is not canonical.")
    if moa.get("metadata_artifact_path") == checkpoint_payload.get("artifact_path"):
        raise NativeMoaGraphRuntimeError(
            "Native MoA metadata artifact must be distinct from the dense-v5 checkpoint."
        )
    moa_version = moa.get("version")
    if (
        moa.get("schema") != NATIVE_MOA_INFERENCE_SCHEMA
        or isinstance(moa_version, bool)
        or moa_version != NATIVE_MOA_INFERENCE_VERSION
        or moa.get("preset") != NATIVE_MOA_PRESET
    ):
        raise NativeMoaGraphRuntimeError(
            "Native MoA checkpoint selection schema/profile is unsupported."
        )
    candidates = moa.get("candidate_activations")
    if not isinstance(candidates, list) or tuple(candidates) != NATIVE_MOA_CANDIDATE_ACTIVATIONS:
        raise NativeMoaGraphRuntimeError(
            "Native MoA checkpoint candidate activations are not canonical."
        )
    selected = moa.get("selected_activation")
    if not isinstance(selected, str) or selected not in NATIVE_MOA_CANDIDATE_ACTIVATIONS:
        raise NativeMoaGraphRuntimeError(
            "Native MoA checkpoint selected activation is not canonical."
        )
    interval = _positive_integer(moa.get("interval"), "checkpoint.moa.interval")
    if (
        _lower_sha256(
            moa.get("source_graph_sha256"), "checkpoint.moa.source_graph_sha256"
        )
        != graph_digest
    ):
        raise NativeMoaGraphRuntimeError(
            "Native MoA checkpoint selection is not bound to the exact source graph bytes."
        )
    metadata_selected, metadata_interval = _validate_metadata_artifact(
        root=root,
        moa=moa,
        graph_digest=graph_digest,
        checkpoint=checkpoint,
    )
    if selected != metadata_selected or interval != metadata_interval:
        raise NativeMoaGraphRuntimeError(
            "Native MoA manifest selection does not match its bound metadata artifact."
        )
    geometry = {
        "max_seq_len": checkpoint.max_seq_len,
        "vocab_size": checkpoint.vocab_size,
        "padded_vocab_size": checkpoint.padded_vocab_size,
        "num_layers": checkpoint.num_layers,
        "model_dim": checkpoint.channels,
        "num_heads": checkpoint.num_heads,
        "head_dim": checkpoint.channels // checkpoint.num_heads,
        "mlp_hidden_dim": checkpoint.channels * 4,
    }
    try:
        _validate_model_semantics(
            model,
            geometry=geometry,
            selected_activation=selected,
            candidate_activations=NATIVE_MOA_CANDIDATE_ACTIVATIONS,
            interval=interval,
        )
    except ValueError as exc:
        raise NativeMoaGraphRuntimeError(
            f"Native MoA graph runtime model contract is invalid: {exc}"
        ) from exc
    return selected, interval


def _validate_checkpoint_and_tensors(
    root: Path,
    raw_manifest: Mapping[str, Any],
) -> NativeDenseCheckpointInfo:
    checkpoint_payload = _mapping(raw_manifest.get("checkpoint"), "checkpoint")
    if checkpoint_payload.get("format") != NATIVE_DENSE_GPT_CHECKPOINT_FORMAT:
        raise NativeMoaGraphRuntimeError(
            "Native MoA graph runtime requires a dense-v5 checkpoint."
        )
    path = _safe_artifact_file(
        root,
        checkpoint_payload.get("artifact_path"),
        field="checkpoint artifact_path",
    )
    expected_nbytes = checkpoint_payload.get("target_nbytes")
    if (
        isinstance(expected_nbytes, bool)
        or not isinstance(expected_nbytes, int)
        or expected_nbytes <= 0
    ):
        raise NativeMoaGraphRuntimeError(
            "Native MoA checkpoint target_nbytes must be a positive integer."
        )
    expected_sha256 = _lower_sha256(
        checkpoint_payload.get("target_sha256"), "checkpoint.target_sha256"
    )
    info = inspect_native_dense_checkpoint(path)
    if info.file_size != expected_nbytes or info.sha256 != expected_sha256:
        raise NativeMoaGraphRuntimeError(
            "Native MoA dense-v5 checkpoint bytes do not match the migrated artifact."
        )
    raw_tensors = raw_manifest.get("tensors")
    if not isinstance(raw_tensors, list) or len(raw_tensors) != len(info.tensors):
        raise NativeMoaGraphRuntimeError(
            "Native MoA manifest tensor table does not match the dense-v5 checkpoint."
        )
    for index, (declared, actual) in enumerate(zip(raw_tensors, info.tensors, strict=True)):
        declared = _mapping(declared, f"tensors[{index}]")
        expected = {
            "name": actual.name,
            "source_name": actual.source_name,
            "dtype": actual.dtype,
            "shape": list(actual.shape),
            "offset": actual.offset,
            "nbytes": actual.nbytes,
            "sha256": actual.sha256,
            "role": actual.role,
            "byte_order": actual.byte_order,
            "layout": "row_major",
        }
        if dict(declared) != expected:
            raise NativeMoaGraphRuntimeError(
                f"Native MoA manifest tensor table entry {index} is not canonical."
            )
    return info


def _activation_paths(graph: NeuronGraph, num_layers: int) -> tuple[str, ...]:
    found: list[str] = []

    def walk(current: NeuronGraph, prefix: str = "") -> None:
        for node_id, node in current.nodes.items():
            path = f"{prefix}/{node_id}" if prefix else node_id
            definition = node.neuron_def
            if definition.kind == "module" and definition.module_type == "gelu":
                found.append(path)
            elif definition.kind == "subgraph" and definition.subgraph is not None:
                walk(definition.subgraph, path)

    walk(graph)
    expected = tuple(f"model/block_{layer}/mlp/gelu" for layer in range(num_layers))
    if tuple(found) != expected:
        raise NativeMoaGraphRuntimeError(
            "Native MoA graph must contain exactly one canonical MLP GELU stage per layer."
        )
    return expected


def _module_at_path(compiled: CompiledTorchGraph, path: str) -> tuple[CompiledTorchGraph, str]:
    parts = path.split("/")
    current = compiled
    for part in parts[:-1]:
        child = current.node_modules[part] if part in current.node_modules else None
        if not isinstance(child, CompiledTorchGraph):
            raise NativeMoaGraphRuntimeError(
                f"Native MoA compiled graph is missing subgraph path {path!r}."
            )
        current = child
    leaf = parts[-1]
    if leaf not in current.node_modules:
        raise NativeMoaGraphRuntimeError(
            f"Native MoA compiled graph is missing activation path {path!r}."
        )
    return current, leaf


def _load_native_dense_weights(
    compiled: CompiledTorchGraph,
    checkpoint: NativeDenseCheckpointInfo,
) -> None:
    weights: dict[str, Tensor] = {}
    with checkpoint.path.open("rb") as handle:
        for tensor in checkpoint.tensors:
            handle.seek(tensor.offset)
            payload = bytearray(handle.read(tensor.nbytes))
            if len(payload) != tensor.nbytes:
                raise NativeMoaGraphRuntimeError(
                    f"Native MoA checkpoint tensor {tensor.name!r} is truncated."
                )
            weights[tensor.name] = torch.frombuffer(
                payload, dtype=torch.bfloat16
            ).float().reshape(tensor.shape)

    consumed: set[str] = set()

    def copy_parameter(parameter: nn.Parameter, source: Tensor, native_name: str) -> None:
        parameter.copy_(source.to(dtype=parameter.dtype).reshape(parameter.shape))
        consumed.add(native_name)

    try:
        model = compiled.node_modules["model"]
        if not isinstance(model, CompiledTorchGraph):
            raise KeyError("model")
        channels = checkpoint.channels
        vocab_size = checkpoint.vocab_size
        with torch.no_grad():
            copy_parameter(
                model.node_modules["token_embed"].embedding.weight,
                weights["transformer.wte.weight"][:vocab_size],
                "transformer.wte.weight",
            )
            position_embedding = model.node_modules["pos_embed"].embedding.weight
            position_embedding.zero_()
            position_embedding[: checkpoint.max_seq_len].copy_(
                weights["transformer.wpe.weight"]
            )
            consumed.add("transformer.wpe.weight")
            copy_parameter(
                model.node_modules["final_norm"].norm.weight,
                weights["transformer.ln_f.weight"],
                "transformer.ln_f.weight",
            )
            copy_parameter(
                model.node_modules["final_norm"].norm.bias,
                weights["transformer.ln_f.bias"],
                "transformer.ln_f.bias",
            )
            for layer in range(checkpoint.num_layers):
                prefix = f"transformer.h.{layer}"
                block = model.node_modules[f"block_{layer}"]
                attention = block.node_modules["attention"]
                mlp = block.node_modules["mlp"]
                for suffix, parameter in (
                    ("ln_1.weight", block.node_modules["attn_norm"].norm.weight),
                    ("ln_1.bias", block.node_modules["attn_norm"].norm.bias),
                    ("attn.c_proj.weight", attention.node_modules["out_proj"].proj.weight),
                    ("attn.c_proj.bias", attention.node_modules["out_proj"].proj.bias),
                    ("ln_2.weight", block.node_modules["mlp_norm"].norm.weight),
                    ("ln_2.bias", block.node_modules["mlp_norm"].norm.bias),
                    ("mlp.c_fc.weight", mlp.node_modules["fc1"].proj.weight),
                    ("mlp.c_fc.bias", mlp.node_modules["fc1"].proj.bias),
                    ("mlp.c_proj.weight", mlp.node_modules["fc2"].proj.weight),
                    ("mlp.c_proj.bias", mlp.node_modules["fc2"].proj.bias),
                ):
                    native_name = f"{prefix}.{suffix}"
                    copy_parameter(parameter, weights[native_name], native_name)
                packed_weight_name = f"{prefix}.attn.c_attn.weight"
                packed_bias_name = f"{prefix}.attn.c_attn.bias"
                packed_weight = weights[packed_weight_name]
                packed_bias = weights[packed_bias_name]
                for projection, projection_index in (
                    ("q_proj", 0),
                    ("k_proj", 1),
                    ("v_proj", 2),
                ):
                    stage = attention.node_modules[projection].proj
                    copy_parameter(
                        stage.weight,
                        packed_weight[
                            projection_index * channels : (projection_index + 1) * channels
                        ],
                        packed_weight_name,
                    )
                    copy_parameter(
                        stage.bias,
                        packed_bias[
                            projection_index * channels : (projection_index + 1) * channels
                        ],
                        packed_bias_name,
                    )
    except (AttributeError, KeyError, RuntimeError, TypeError) as exc:
        raise NativeMoaGraphRuntimeError(
            "Native MoA source graph does not expose the canonical dense-v5 parameter stages."
        ) from exc
    if consumed != set(weights):
        missing = sorted(set(weights) - consumed)
        raise NativeMoaGraphRuntimeError(
            "Native MoA graph runtime did not consume every dense-v5 tensor: "
            + ", ".join(missing)
        )


def load_native_moa_graph_runtime(
    source_graph_path: str | Path,
    artifact: str | Path,
    *,
    kernel_backend: str = "torch",
) -> NativeMoaGraphRuntime:
    """Load an exact migrated ``gpt2_moa`` artifact into the Torch graph runtime.

    ``source_graph_path`` may be a renamed byte-for-byte copy of the graph used
    for migration.  The loader never edits it.  It fails before compilation if
    graph, manifest, selection, tensor-table, or checkpoint fingerprints drift.
    """

    root, manifest, raw_manifest = _load_manifest(artifact)
    graph, raw_graph, graph_digest = _validate_source_graph(
        source_graph_path, manifest
    )
    checkpoint = _validate_checkpoint_and_tensors(root, raw_manifest)
    selected, interval = _validate_moa_contract(
        root=root,
        raw_graph=raw_graph,
        raw_manifest=raw_manifest,
        graph_digest=graph_digest,
        checkpoint=checkpoint,
    )
    paths = _activation_paths(graph, checkpoint.num_layers)
    compiled = CompiledTorchGraph(graph, kernel_backend=kernel_backend)
    for path in paths:
        parent, leaf = _module_at_path(compiled, path)
        parent.node_modules[leaf] = _SelectedMoaActivationStage(selected)
    _load_native_dense_weights(compiled, checkpoint)
    compiled.eval()
    binding = NativeMoaGraphRuntimeBinding(
        source_graph_sha256=graph_digest,
        checkpoint_sha256=checkpoint.sha256,
        selected_activation=selected,
        candidate_activations=NATIVE_MOA_CANDIDATE_ACTIVATIONS,
        interval=interval,
        activation_node_paths=paths,
    )
    return NativeMoaGraphRuntime(
        graph=graph,
        compiled=compiled,
        manifest=manifest,
        checkpoint=checkpoint,
        binding=binding,
    )


__all__ = [
    "NATIVE_MOA_GRAPH_RUNTIME_PROFILE",
    "NativeMoaGraphRuntime",
    "NativeMoaGraphRuntimeBinding",
    "NativeMoaGraphRuntimeError",
    "load_native_moa_graph_runtime",
]
