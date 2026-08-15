"""Torch-free inspection for canonical native-family LLaMA checkpoints.

The family trainer historically wrote a diagnostic JSON document next to a
raw float32 parameter sidecar.  Canonical LLaMA production checkpoints add a
self-describing inference contract to that JSON.  This module treats the JSON
as untrusted input: it validates the completion marker, contains the relative
sidecar path, re-derives the exact tensor layout from geometry, and streams the
sidecar through SHA-256 without loading model weights into memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Mapping


NATIVE_FAMILY_OPTIMIZER_CHECKPOINT_FORMAT = "nfn-native-family-optimizer-checkpoint-v1"
NATIVE_FAMILY_LLAMA_INFERENCE_SCHEMA = "neuralfn.native_family_llama.inference_checkpoint"
NATIVE_FAMILY_LLAMA_INFERENCE_VERSION = 2
NATIVE_FAMILY_LLAMA_CHECKPOINT_FORMAT = "neuralfn.native_family_llama.f32.v1"
_FLOAT32_BYTES = 4

_LLAMA_SEMANTICS: dict[str, Any] = {
    "norm_type": "rmsnorm",
    "mlp_type": "swiglu",
    "pos_encoding": "rope",
    "attention_variant": "dense",
    "residual_type": "add",
    "linear_bias": False,
    "dropout_p": 0.0,
    "tie_embeddings": False,
}


@dataclass(frozen=True, slots=True)
class NativeFamilyLlamaTensor:
    name: str
    source_name: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int
    sha256: str
    dtype: str = "float32"
    role: str = "parameter"
    byte_order: str = "little"
    layout: str = "row_major"

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
            "layout": self.layout,
        }


@dataclass(frozen=True, slots=True)
class NativeFamilyLlamaCheckpointInfo:
    metadata_path: Path
    artifact_path: Path
    done_marker_path: Path
    geometry: dict[str, int | float]
    semantics: dict[str, Any]
    training: dict[str, Any]
    file_size: int
    sha256: str
    tensors: tuple[NativeFamilyLlamaTensor, ...]

    @property
    def path(self) -> Path:
        """The raw float32 sidecar path, matching dense-inspector terminology."""

        return self.artifact_path

    @property
    def max_seq_len(self) -> int:
        return int(self.geometry["max_seq_len"])

    def checkpoint_descriptor(self, *, artifact_path: str = "model.f32") -> dict[str, Any]:
        if artifact_path != "model.f32":
            raise ValueError("Canonical LLaMA Native IR artifacts must be named model.f32.")
        descriptor = {
            "format": NATIVE_FAMILY_LLAMA_CHECKPOINT_FORMAT,
            "artifact_path": artifact_path,
            "source_path": str(self.metadata_path),
            "source_artifact_path": str(self.artifact_path),
            "source_format": NATIVE_FAMILY_LLAMA_INFERENCE_SCHEMA,
            "source_version": NATIVE_FAMILY_LLAMA_INFERENCE_VERSION,
            "source_sha256": self.sha256,
            "target_file": artifact_path,
            "target_sha256": self.sha256,
            "target_nbytes": self.file_size,
            "dtype": "float32",
            "byte_order": "little",
            "layout": "contiguous_row_major",
            "tensor_offsets_include_header": False,
            "restricted_unpickler": False,
            "isolated_worker": False,
            "family": "llama",
            "preset": "llama",
            "geometry": dict(self.geometry),
            "semantics": dict(self.semantics),
        }
        source_graph = self.training.get("source_graph")
        if isinstance(source_graph, Mapping):
            descriptor["source_graph"] = dict(source_graph)
        return descriptor

    def validate_model(self, model: Mapping[str, Any]) -> None:
        family = _normalized_id(model.get("family"))
        family_class = str(model.get("family_class") or "").strip().lower()
        if family != "llama" or family_class != "autoregressive_transformer":
            raise ValueError(
                "Native family LLaMA checkpoints require a canonical LLaMA "
                "autoregressive graph."
            )
        spec = _mapping(model.get("template_spec"), "model.template_spec")
        block = _mapping(spec.get("block_spec"), "model.template_spec.block_spec")
        for field, expected in _LLAMA_SEMANTICS.items():
            container = spec if field == "tie_embeddings" else block
            if field not in container:
                raise ValueError(f"Canonical LLaMA graph is missing explicit {field} metadata.")
            actual = container[field]
            if isinstance(expected, str):
                actual = _normalized_id(actual)
            elif isinstance(expected, bool):
                if not isinstance(actual, bool):
                    raise ValueError(f"Canonical LLaMA graph {field} must be boolean.")
            elif isinstance(actual, bool):
                raise ValueError(f"Canonical LLaMA graph {field} must be numeric.")
            if actual != expected:
                raise ValueError(
                    f"Canonical LLaMA graph requires {field}={expected!r}; got {actual!r}."
                )

        expected_geometry = {
            "model_dim": spec.get("model_dim"),
            "num_layers": spec.get("num_layers"),
            "vocab_size": spec.get("vocab_size"),
            "num_heads": block.get("num_heads"),
            "num_kv_heads": block.get("num_kv_heads"),
        }
        for field, graph_value in expected_geometry.items():
            if isinstance(graph_value, bool) or graph_value in (None, ""):
                raise ValueError(f"Canonical LLaMA graph is missing explicit {field} metadata.")
            if int(graph_value) != int(self.geometry[field]):
                raise ValueError(
                    f"Graph {field}={graph_value!r} does not match native LLaMA checkpoint "
                    f"{field}={self.geometry[field]}."
                )

        graph_rope_theta = block.get("rope_theta")
        if isinstance(graph_rope_theta, bool) or graph_rope_theta in (None, ""):
            raise ValueError("Canonical LLaMA graph is missing explicit rope_theta metadata.")
        if float(graph_rope_theta) != float(self.geometry["rope_theta"]):
            raise ValueError("Graph rope_theta does not match native LLaMA checkpoint geometry.")
        rope_scaling = block.get("rope_scaling")
        if rope_scaling in (None, ""):
            graph_rope_factor = 1.0
        elif isinstance(rope_scaling, Mapping):
            raw_factor = rope_scaling.get("factor", 1.0)
            if isinstance(raw_factor, bool):
                raise ValueError("Canonical LLaMA graph rope scaling factor must be numeric.")
            graph_rope_factor = float(raw_factor)
        else:
            raise ValueError("Canonical LLaMA graph rope_scaling must be a mapping or null.")
        if graph_rope_factor != float(self.geometry["rope_scaling_factor"]):
            raise ValueError("Graph rope scaling does not match native LLaMA checkpoint geometry.")

        model_dim = int(spec["model_dim"])
        hidden_dim = _derived_hidden_dim(block, model_dim)
        if hidden_dim != int(self.geometry["hidden_dim"]):
            raise ValueError(
                f"Graph hidden_dim={hidden_dim} does not match native LLaMA checkpoint "
                f"hidden_dim={self.geometry['hidden_dim']}."
            )


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Native family LLaMA checkpoint field {field} must be an object.")
    return value


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"Native family LLaMA checkpoint field {field} must be an integer >= {minimum}."
        )
    return value


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Native family LLaMA checkpoint field {field} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"Native family LLaMA checkpoint field {field} must be {qualifier}.")
    return result


def _normalized_id(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _safe_relative_path(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Native family LLaMA checkpoint field {field} must be a safe relative path.")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError(f"Native family LLaMA checkpoint field {field} must be a safe relative path.")
    try:
        resolved = (root / Path(*posix.parts)).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"Native family LLaMA checkpoint {field} does not exist.") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Native family LLaMA checkpoint field {field} escapes its artifact root.")
    return resolved


def _shape_elements(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        if dim <= 0:
            raise ValueError("Native family LLaMA tensor shapes must be positive.")
        result *= dim
    return result


def _expected_layout(geometry: Mapping[str, int | float]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    model_dim = int(geometry["model_dim"])
    hidden_dim = int(geometry["hidden_dim"])
    layers = int(geometry["num_layers"])
    padded_vocab = int(geometry["padded_vocab_size"])
    kv_dim = int(geometry["num_kv_heads"]) * int(geometry["head_dim"])
    tensors: list[tuple[str, tuple[int, ...]]] = [
        ("token_embedding.weight", (padded_vocab, model_dim)),
        ("final_norm.weight", (model_dim,)),
        ("lm_head.weight", (padded_vocab, model_dim)),
    ]
    for layer in range(layers):
        prefix = f"layers.{layer}."
        tensors.extend(
            (
                (f"{prefix}attention_norm.weight", (model_dim,)),
                (f"{prefix}q_proj.weight", (model_dim, model_dim)),
                (f"{prefix}k_proj.weight", (kv_dim, model_dim)),
                (f"{prefix}v_proj.weight", (kv_dim, model_dim)),
                (f"{prefix}attention_out.weight", (model_dim, model_dim)),
                (f"{prefix}ffn_norm.weight", (model_dim,)),
                (f"{prefix}ffn_gate_up.weight", (2, hidden_dim, model_dim)),
                (f"{prefix}ffn_down.weight", (model_dim, hidden_dim)),
            )
        )
    return tuple(tensors)


def _derived_hidden_dim(block: Mapping[str, Any], model_dim: int) -> int:
    multiplier = _number(block.get("mlp_multiplier"), "model.block_spec.mlp_multiplier", positive=True)
    multiple_of = _integer(block.get("multiple_of"), "model.block_spec.multiple_of", minimum=0)
    hidden = max(1, int(model_dim * multiplier))
    if multiple_of > 0:
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
    return hidden


def _read_tensor_and_hash(handle: BinaryIO, nbytes: int, file_digest: Any) -> str:
    digest = hashlib.sha256()
    remaining = nbytes
    while remaining:
        chunk = handle.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise ValueError("Native family LLaMA parameter sidecar is truncated.")
        remaining -= len(chunk)
        digest.update(chunk)
        file_digest.update(chunk)
    return digest.hexdigest()


def inspect_native_family_llama_checkpoint(
    path: str | Path,
) -> NativeFamilyLlamaCheckpointInfo:
    """Validate a canonical LLaMA metadata JSON and its float32 sidecar."""

    try:
        metadata_path = Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("Native family LLaMA checkpoint metadata does not exist.") from exc
    if not metadata_path.is_file() or metadata_path.suffix.lower() != ".json":
        raise ValueError("Native family LLaMA checkpoint input must be a metadata .json file.")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Native family LLaMA checkpoint metadata is not valid UTF-8 JSON.") from exc
    payload = _mapping(payload, "root")
    if payload.get("format") != NATIVE_FAMILY_OPTIMIZER_CHECKPOINT_FORMAT:
        raise ValueError("Native family LLaMA checkpoint requires optimizer checkpoint metadata v1.")
    if _normalized_id(payload.get("model_family")) != "llama":
        raise ValueError("Native family LLaMA checkpoint metadata has the wrong model family.")
    if _normalized_id(payload.get("template_name")) != "llama":
        raise ValueError("Native family LLaMA inference requires the canonical llama preset.")

    contract = _mapping(payload.get("inference_contract"), "inference_contract")
    if contract.get("schema") != NATIVE_FAMILY_LLAMA_INFERENCE_SCHEMA:
        raise ValueError("Native family LLaMA checkpoint has an unknown inference schema.")
    if _integer(contract.get("version"), "inference_contract.version", minimum=1) != 2:
        raise ValueError("Native family LLaMA checkpoint has an unsupported inference version.")
    if _normalized_id(contract.get("family")) != "llama" or _normalized_id(contract.get("preset")) != "llama":
        raise ValueError("Native family LLaMA inference contract identity is inconsistent.")
    if contract.get("checkpoint_kind") != "live_full_architecture":
        raise ValueError("Native family LLaMA inference contract is not a live full-architecture checkpoint.")

    root = metadata_path.parent.resolve()
    artifact = _mapping(contract.get("artifact"), "inference_contract.artifact")
    if artifact.get("format") != NATIVE_FAMILY_LLAMA_CHECKPOINT_FORMAT:
        raise ValueError("Native family LLaMA parameter sidecar has an unknown format.")
    artifact_path = _safe_relative_path(root, artifact.get("path"), "inference_contract.artifact.path")
    done_marker_path = _safe_relative_path(
        root, contract.get("done_marker"), "inference_contract.done_marker"
    )
    expected_done_name = (
        metadata_path.name.removesuffix("_00000000.json") + "_DONE"
        if metadata_path.name.endswith("_00000000.json")
        else metadata_path.name + ".DONE"
    )
    if done_marker_path != (root / expected_done_name).resolve(strict=True):
        raise ValueError("Native family LLaMA checkpoint declares the wrong DONE marker.")
    if done_marker_path.read_text(encoding="utf-8").strip() != "done":
        raise ValueError("Native family LLaMA checkpoint DONE marker is invalid.")

    if artifact.get("dtype") != "float32" or artifact.get("byte_order") != "little":
        raise ValueError("Native family LLaMA parameter sidecar must be little-endian float32.")
    if artifact.get("layout") != "contiguous_row_major":
        raise ValueError("Native family LLaMA parameter sidecar must use contiguous row-major layout.")
    expected_nbytes = _integer(artifact.get("nbytes"), "inference_contract.artifact.nbytes", minimum=1)
    expected_sha256 = artifact.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("Native family LLaMA parameter sidecar requires a lowercase SHA-256 digest.")
    if artifact_path.stat().st_size != expected_nbytes:
        raise ValueError("Native family LLaMA parameter sidecar size does not match metadata.")

    geometry_raw = _mapping(contract.get("geometry"), "inference_contract.geometry")
    required_geometry = {
        "max_seq_len",
        "vocab_size",
        "padded_vocab_size",
        "num_layers",
        "model_dim",
        "hidden_dim",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "rope_theta",
        "rope_scaling_factor",
        "rms_norm_eps",
    }
    if set(geometry_raw) != required_geometry:
        raise ValueError("Native family LLaMA checkpoint geometry fields are not canonical.")
    geometry: dict[str, int | float] = {
        field: _integer(geometry_raw.get(field), f"geometry.{field}", minimum=1)
        for field in (
            "max_seq_len",
            "vocab_size",
            "padded_vocab_size",
            "num_layers",
            "model_dim",
            "hidden_dim",
            "num_heads",
            "num_kv_heads",
            "head_dim",
        )
    }
    geometry["rope_theta"] = _number(geometry_raw.get("rope_theta"), "geometry.rope_theta", positive=True)
    geometry["rope_scaling_factor"] = _number(
        geometry_raw.get("rope_scaling_factor"), "geometry.rope_scaling_factor", positive=True
    )
    geometry["rms_norm_eps"] = _number(geometry_raw.get("rms_norm_eps"), "geometry.rms_norm_eps", positive=True)
    if int(geometry["padded_vocab_size"]) < int(geometry["vocab_size"]):
        raise ValueError("Native family LLaMA padded vocabulary is smaller than its vocabulary.")
    if int(geometry["model_dim"]) != int(geometry["num_heads"]) * int(geometry["head_dim"]):
        raise ValueError("Native family LLaMA head geometry does not reconstruct model_dim.")
    if int(geometry["num_heads"]) % int(geometry["num_kv_heads"]):
        raise ValueError("Native family LLaMA grouped-query head geometry is invalid.")
    if int(geometry["head_dim"]) % 2:
        raise ValueError("Native family LLaMA RoPE requires an even head_dim.")
    if float(geometry["rms_norm_eps"]) != 1.0e-6:
        raise ValueError("Native family LLaMA checkpoint requires rms_norm_eps=1e-6.")

    semantics = dict(_mapping(contract.get("semantics"), "inference_contract.semantics"))
    if semantics != _LLAMA_SEMANTICS:
        raise ValueError("Native family LLaMA checkpoint semantics are not canonical.")
    training = dict(_mapping(contract.get("training"), "inference_contract.training"))
    if _integer(training.get("train_seq_len"), "training.train_seq_len", minimum=1) != geometry["max_seq_len"]:
        raise ValueError("Native family LLaMA training and context lengths disagree.")
    source_graph_raw = training.get("source_graph")
    if source_graph_raw is not None:
        source_graph = _mapping(source_graph_raw, "training.source_graph")
        if set(source_graph) != {"filename", "sha256", "byte_identity_verified"}:
            raise ValueError("Native family LLaMA source-graph provenance fields are not canonical.")
        filename = source_graph.get("filename")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or PurePosixPath(filename).is_absolute()
            or PureWindowsPath(filename).is_absolute()
        ):
            raise ValueError("Native family LLaMA source-graph filename is unsafe.")
        graph_sha256 = source_graph.get("sha256")
        if (
            not isinstance(graph_sha256, str)
            or len(graph_sha256) != 64
            or any(character not in "0123456789abcdef" for character in graph_sha256)
        ):
            raise ValueError("Native family LLaMA source graph requires a lowercase SHA-256 digest.")
        if source_graph.get("byte_identity_verified") is not True:
            raise ValueError("Native family LLaMA source graph must prove byte-identity verification.")

    expected_layout = _expected_layout(geometry)
    tensor_rows = contract.get("tensors")
    if not isinstance(tensor_rows, list) or len(tensor_rows) != len(expected_layout):
        raise ValueError("Native family LLaMA tensor table does not match canonical layout.")
    legacy_layout = _mapping(payload.get("architecture_parameter_layout"), "architecture_parameter_layout")
    legacy_buffers = legacy_layout.get("buffers")
    if not isinstance(legacy_buffers, list) or len(legacy_buffers) != len(expected_layout):
        raise ValueError("Native family LLaMA legacy parameter layout is inconsistent.")

    parsed_rows: list[tuple[str, tuple[int, ...], int, int]] = []
    occupied_end = 0
    for index, ((expected_name, expected_shape), raw_row, legacy_row) in enumerate(
        zip(expected_layout, tensor_rows, legacy_buffers, strict=True)
    ):
        row = _mapping(raw_row, f"tensors[{index}]")
        legacy = _mapping(legacy_row, f"architecture_parameter_layout.buffers[{index}]")
        name = row.get("name")
        raw_shape = row.get("shape")
        if name != expected_name or not isinstance(raw_shape, list):
            raise ValueError("Native family LLaMA tensor names or shapes are not canonical.")
        shape = tuple(_integer(dim, f"tensors[{index}].shape", minimum=1) for dim in raw_shape)
        if shape != expected_shape:
            raise ValueError(f"Native family LLaMA tensor {expected_name!r} has the wrong shape.")
        offset = _integer(row.get("offset"), f"tensors[{index}].offset", minimum=0)
        nbytes = _integer(row.get("nbytes"), f"tensors[{index}].nbytes", minimum=1)
        expected_tensor_nbytes = _shape_elements(shape) * _FLOAT32_BYTES
        if offset != occupied_end or nbytes != expected_tensor_nbytes:
            raise ValueError("Native family LLaMA tensor table is not exactly contiguous.")
        if row.get("dtype") != "float32" or row.get("byte_order") != "little" or row.get("layout") != "row_major":
            raise ValueError("Native family LLaMA tensor encoding is not canonical.")
        if (
            legacy.get("name") != expected_name
            or legacy.get("byte_offset") != offset
            or legacy.get("bytes") != nbytes
            or legacy.get("elements") != nbytes // _FLOAT32_BYTES
        ):
            raise ValueError("Native family LLaMA v1 and v2 parameter layouts disagree.")
        parsed_rows.append((expected_name, shape, offset, nbytes))
        occupied_end += nbytes
    if occupied_end != expected_nbytes:
        raise ValueError("Native family LLaMA tensor layout does not cover the entire sidecar.")

    parameter_data = _mapping(payload.get("parameter_data"), "parameter_data")
    parameter_state = _mapping(payload.get("native_parameter_state"), "native_parameter_state")
    writer = _mapping(payload.get("writer_verification"), "writer_verification")
    if (
        parameter_data.get("parameter_dtype") != "float32"
        or parameter_data.get("bytes") != expected_nbytes
        or not writer.get("passed")
        or not writer.get("parameter_sidecar_exists")
        or not writer.get("parameter_sidecar_size_matches")
        or not parameter_state.get("full_template_parameter_state")
        or not parameter_state.get("architecture_forward_inference_supported")
    ):
        raise ValueError("Native family LLaMA v1 metadata does not prove a complete live checkpoint.")
    parameter_elements = expected_nbytes // _FLOAT32_BYTES
    if (
        parameter_data.get("parameter_elements") != parameter_elements
        or parameter_state.get("parameter_elements") != parameter_elements
        or parameter_state.get("persisted_parameter_elements") != parameter_elements
        or parameter_state.get("trained_parameter_elements") != parameter_elements
    ):
        raise ValueError("Native family LLaMA parameter counts are inconsistent.")
    for field, raw_path in (
        ("parameter_data.path", parameter_data.get("path")),
        ("native_parameter_state.parameter_data_path", parameter_state.get("parameter_data_path")),
    ):
        # The legacy v1 fields may contain the trainer's original absolute or
        # cwd-relative output path.  The v2 relative path is authoritative and
        # intentionally keeps a metadata/sidecar/DONE bundle relocatable; only
        # require the legacy fields to identify the same sidecar filename.
        if not isinstance(raw_path, str) or Path(raw_path).name != artifact_path.name:
            raise ValueError(f"Native family LLaMA {field} disagrees with the inference artifact.")

    file_digest = hashlib.sha256()
    tensors: list[NativeFamilyLlamaTensor] = []
    with artifact_path.open("rb") as handle:
        for name, shape, offset, nbytes in parsed_rows:
            if handle.tell() != offset:
                raise ValueError("Native family LLaMA tensor table is not exactly contiguous.")
            tensor_sha = _read_tensor_and_hash(handle, nbytes, file_digest)
            tensors.append(
                NativeFamilyLlamaTensor(
                    name=name,
                    source_name=name,
                    shape=shape,
                    offset=offset,
                    nbytes=nbytes,
                    sha256=tensor_sha,
                )
            )
        if handle.read(1):
            raise ValueError("Native family LLaMA parameter sidecar has trailing bytes.")
    actual_sha256 = file_digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("Native family LLaMA parameter sidecar failed SHA-256 validation.")

    return NativeFamilyLlamaCheckpointInfo(
        metadata_path=metadata_path,
        artifact_path=artifact_path,
        done_marker_path=done_marker_path,
        geometry=geometry,
        semantics=semantics,
        training=training,
        file_size=expected_nbytes,
        sha256=actual_sha256,
        tensors=tuple(tensors),
    )


__all__ = [
    "NATIVE_FAMILY_LLAMA_CHECKPOINT_FORMAT",
    "NATIVE_FAMILY_LLAMA_INFERENCE_SCHEMA",
    "NATIVE_FAMILY_LLAMA_INFERENCE_VERSION",
    "NativeFamilyLlamaCheckpointInfo",
    "NativeFamilyLlamaTensor",
    "inspect_native_family_llama_checkpoint",
]
