"""Strict, Torch-free inspection for canonical standard-MoE checkpoints.

This contract is deliberately narrower than the native family trainer's broad
MoE surface.  It accepts only the graph-equivalent ``mixllama``/``moe`` eager
profile and its reviewed ``mixllama_fast`` compile alias: RMSNorm, RoPE/GQA,
softmax top-k routing with renormalisation, no shared experts, and the ordinary
router auxiliary loss.  Metadata and sidecars are treated as untrusted input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Mapping


NATIVE_FAMILY_OPTIMIZER_CHECKPOINT_FORMAT = "nfn-native-family-optimizer-checkpoint-v1"
NATIVE_FAMILY_STANDARD_MOE_INFERENCE_SCHEMA = (
    "neuralfn.native_family_standard_moe.inference_checkpoint"
)
NATIVE_FAMILY_STANDARD_MOE_INFERENCE_VERSION = 1
NATIVE_FAMILY_STANDARD_MOE_CHECKPOINT_FORMAT = (
    "neuralfn.native_family_standard_moe.f32.v1"
)
_FLOAT32_BYTES = 4
_ALLOWED_PRESETS = {"mixllama", "moe", "mixllama-fast"}

_STANDARD_MOE_SEMANTICS: dict[str, Any] = {
    "norm_type": "rmsnorm",
    "mlp_type": "moe",
    "pos_encoding": "rope",
    "attention_variant": "dense",
    "residual_type": "add",
    "router_score_fn": "softmax",
    "router_selection": "topk_renormalized",
    "moe_balance_mode": "aux_loss",
    "linear_bias": False,
    "dropout_p": 0.0,
    "tie_embeddings": False,
    "use_qk_norm": False,
    "shared_experts": 0,
}


@dataclass(frozen=True, slots=True)
class NativeFamilyStandardMoeTensor:
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
class NativeFamilyStandardMoeCheckpointInfo:
    metadata_path: Path
    artifact_path: Path
    done_marker_path: Path
    preset: str
    geometry: dict[str, int | float]
    semantics: dict[str, Any]
    training: dict[str, Any]
    file_size: int
    sha256: str
    tensors: tuple[NativeFamilyStandardMoeTensor, ...]

    @property
    def path(self) -> Path:
        return self.artifact_path

    @property
    def max_seq_len(self) -> int:
        return int(self.geometry["max_seq_len"])

    def checkpoint_descriptor(self, *, artifact_path: str = "model.f32") -> dict[str, Any]:
        if artifact_path != "model.f32":
            raise ValueError("Canonical standard-MoE Native IR artifacts must be named model.f32.")
        return {
            "format": NATIVE_FAMILY_STANDARD_MOE_CHECKPOINT_FORMAT,
            "artifact_path": artifact_path,
            "source_path": str(self.metadata_path),
            "source_artifact_path": str(self.artifact_path),
            "source_format": NATIVE_FAMILY_STANDARD_MOE_INFERENCE_SCHEMA,
            "source_version": NATIVE_FAMILY_STANDARD_MOE_INFERENCE_VERSION,
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
            "family": "mixllama",
            "preset": self.preset,
            "geometry": dict(self.geometry),
            "semantics": dict(self.semantics),
            "source_graph": dict(self.training["source_graph"]),
        }

    def validate_model(self, model: Mapping[str, Any]) -> None:
        if _normalized_id(model.get("family")) != "mixllama" or str(
            model.get("family_class") or ""
        ).strip().lower() != "autoregressive_transformer":
            raise ValueError(
                "Native standard-MoE checkpoints require a canonical MixLLaMA autoregressive graph."
            )
        spec = _mapping(model.get("template_spec"), "model.template_spec")
        block = _mapping(spec.get("block_spec"), "model.template_spec.block_spec")
        template = _mapping(spec.get("template"), "model.template_spec.template")
        string_contract = {
            "norm_type": "rmsnorm",
            "mlp_type": "moe",
            "pos_encoding": "rope",
            "attention_variant": "dense",
            "attention_backend": "sdpa",
            "residual_type": "add",
            "compression": "none",
            "adapter_type": "none",
            "activation_mode": "single",
            "moe_balance_mode": "aux_loss",
            "router_score_fn": "softmax",
        }
        for field, expected in string_contract.items():
            if _normalized_id(block.get(field)) != _normalized_id(expected):
                raise ValueError(f"Canonical standard-MoE graph requires {field}={expected!r}.")
        template_contract = {
            "objective": "ar",
            "backbone": "mixllama",
            "sparsity": "moe",
            "router_mode": "none",
            "compression": "none",
            "adapter": "none",
        }
        for field, expected in template_contract.items():
            if _normalized_id(template.get(field)) != expected:
                raise ValueError(f"Canonical standard-MoE template requires {field}={expected!r}.")
        expected_runtime = "compile" if self.preset == "mixllama-fast" else "eager"
        if _normalized_id(template.get("runtime")) != expected_runtime:
            raise ValueError(
                f"Canonical standard-MoE preset {self.preset!r} requires runtime={expected_runtime!r}."
            )
        for field, expected in {
            "linear_bias": False,
            "use_qk_norm": False,
            "is_causal": True,
        }.items():
            if block.get(field) is not expected:
                raise ValueError(f"Canonical standard-MoE graph requires {field}={expected!r}.")
        for field, expected in {"dropout_p": 0.0, "logit_softcap": 0.0}.items():
            container = block if field == "dropout_p" else spec
            if isinstance(container.get(field), bool) or float(container.get(field, math.nan)) != expected:
                raise ValueError(f"Canonical standard-MoE graph requires {field}={expected!r}.")
        if spec.get("tie_embeddings") is not False or block.get("rope_scaling") is not None:
            raise ValueError("Canonical standard-MoE graph requires untied weights and unscaled RoPE.")

        expected_ints = {
            "model_dim": spec.get("model_dim"),
            "num_layers": spec.get("num_layers"),
            "vocab_size": spec.get("vocab_size"),
            "num_heads": block.get("num_heads"),
            "num_kv_heads": block.get("num_kv_heads"),
            "experts": block.get("experts"),
            "top_k": block.get("top_k"),
            "shared_experts": block.get("shared_experts"),
        }
        for field, value in expected_ints.items():
            if isinstance(value, bool) or value is None or int(value) != int(
                self.geometry[field] if field != "shared_experts" else 0
            ):
                raise ValueError(f"Graph {field} does not match the standard-MoE checkpoint.")
        for field in ("rope_theta", "mlp_multiplier", "router_aux_loss_coef"):
            value = block.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != float(
                self.geometry[field]
            ):
                raise ValueError(f"Graph {field} does not match the standard-MoE checkpoint.")
        graph_multiple = 0 if block.get("multiple_of") is None else int(block["multiple_of"])
        if graph_multiple != int(self.geometry["multiple_of"]):
            raise ValueError("Graph multiple_of does not match the standard-MoE checkpoint.")
        if _derived_hidden_dim(
            int(self.geometry["model_dim"]),
            float(self.geometry["mlp_multiplier"]),
            int(self.geometry["multiple_of"]),
        ) != int(self.geometry["hidden_dim"]):
            raise ValueError("Standard-MoE graph width does not match checkpoint hidden_dim.")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Native standard-MoE checkpoint field {field} must be an object.")
    return value


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Native standard-MoE checkpoint field {field} must be an integer >= {minimum}.")
    return value


def _number(value: Any, field: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Native standard-MoE checkpoint field {field} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0) or (nonnegative and result < 0.0):
        raise ValueError(f"Native standard-MoE checkpoint field {field} has an invalid value.")
    return result


def _normalized_id(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _lower_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"Native standard-MoE checkpoint field {field} requires lowercase SHA-256.")
    return value


def _safe_relative_path(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Native standard-MoE checkpoint field {field} must be a safe relative path.")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or any(
        part in {"", ".", ".."} for part in posix.parts
    ):
        raise ValueError(f"Native standard-MoE checkpoint field {field} must be a safe relative path.")
    try:
        result = (root / Path(*posix.parts)).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"Native standard-MoE checkpoint {field} does not exist.") from exc
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError(f"Native standard-MoE checkpoint field {field} escapes its artifact root.")
    return result


def _derived_hidden_dim(model_dim: int, multiplier: float, multiple_of: int) -> int:
    hidden = max(1, int(model_dim * multiplier))
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of) if multiple_of > 0 else hidden


def _expected_layout(geometry: Mapping[str, int | float]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    d = int(geometry["model_dim"])
    h = int(geometry["hidden_dim"])
    e = int(geometry["experts"])
    kv = int(geometry["num_kv_heads"]) * int(geometry["head_dim"])
    tensors: list[tuple[str, tuple[int, ...]]] = [
        ("token_embedding.weight", (int(geometry["padded_vocab_size"]), d)),
        ("final_norm.weight", (d,)),
        ("lm_head.weight", (int(geometry["padded_vocab_size"]), d)),
    ]
    for layer in range(int(geometry["num_layers"])):
        prefix = f"layers.{layer}."
        tensors.extend(
            (
                (f"{prefix}attention_norm.weight", (d,)),
                (f"{prefix}q_proj.weight", (d, d)),
                (f"{prefix}k_proj.weight", (kv, d)),
                (f"{prefix}v_proj.weight", (kv, d)),
                (f"{prefix}attention_out.weight", (d, d)),
                (f"{prefix}ffn_norm.weight", (d,)),
                (f"{prefix}router.weight", (e, d)),
                (f"{prefix}experts.gate_up.weight", (2, e, d, h)),
                (f"{prefix}experts.down.weight", (e, h, d)),
            )
        )
    return tuple(tensors)


def _read_tensor_and_hash(handle: BinaryIO, nbytes: int, file_digest: Any) -> str:
    digest = hashlib.sha256()
    remaining = nbytes
    while remaining:
        chunk = handle.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise ValueError("Native standard-MoE parameter sidecar is truncated.")
        remaining -= len(chunk)
        digest.update(chunk)
        file_digest.update(chunk)
    return digest.hexdigest()


def inspect_native_family_standard_moe_checkpoint(
    path: str | Path,
) -> NativeFamilyStandardMoeCheckpointInfo:
    """Validate one canonical standard-MoE metadata/sidecar/DONE bundle."""

    try:
        metadata_path = Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("Native standard-MoE checkpoint metadata does not exist.") from exc
    if not metadata_path.is_file() or metadata_path.suffix.lower() != ".json":
        raise ValueError("Native standard-MoE checkpoint input must be a metadata .json file.")
    try:
        payload = _mapping(json.loads(metadata_path.read_text(encoding="utf-8")), "root")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Native standard-MoE checkpoint metadata is not valid UTF-8 JSON.") from exc
    if payload.get("format") != NATIVE_FAMILY_OPTIMIZER_CHECKPOINT_FORMAT:
        raise ValueError("Native standard-MoE checkpoint requires optimizer checkpoint metadata v1.")
    if _normalized_id(payload.get("model_family")) != "mixllama":
        raise ValueError("Native standard-MoE checkpoint metadata has the wrong model family.")
    preset = _normalized_id(payload.get("template_name"))
    if preset not in _ALLOWED_PRESETS:
        raise ValueError("Native standard-MoE checkpoint preset is outside the exact supported cluster.")

    contract = _mapping(payload.get("inference_contract"), "inference_contract")
    if contract.get("schema") != NATIVE_FAMILY_STANDARD_MOE_INFERENCE_SCHEMA or _integer(
        contract.get("version"), "inference_contract.version", minimum=1
    ) != NATIVE_FAMILY_STANDARD_MOE_INFERENCE_VERSION:
        raise ValueError("Native standard-MoE checkpoint has an unsupported inference contract.")
    if _normalized_id(contract.get("family")) != "mixllama" or _normalized_id(
        contract.get("preset")
    ) != preset or contract.get("checkpoint_kind") != "live_full_architecture":
        raise ValueError("Native standard-MoE checkpoint identity is inconsistent.")

    root = metadata_path.parent.resolve()
    artifact = _mapping(contract.get("artifact"), "inference_contract.artifact")
    if artifact.get("format") != NATIVE_FAMILY_STANDARD_MOE_CHECKPOINT_FORMAT:
        raise ValueError("Native standard-MoE parameter sidecar has an unknown format.")
    artifact_path = _safe_relative_path(root, artifact.get("path"), "inference_contract.artifact.path")
    done_marker_path = _safe_relative_path(root, contract.get("done_marker"), "inference_contract.done_marker")
    expected_done_name = (
        metadata_path.name.removesuffix("_00000000.json") + "_DONE"
        if metadata_path.name.endswith("_00000000.json")
        else metadata_path.name + ".DONE"
    )
    if done_marker_path != (root / expected_done_name).resolve(strict=True) or done_marker_path.read_text(
        encoding="utf-8"
    ).strip() != "done":
        raise ValueError("Native standard-MoE checkpoint DONE marker is invalid.")
    if artifact.get("dtype") != "float32" or artifact.get("byte_order") != "little" or artifact.get(
        "layout"
    ) != "contiguous_row_major":
        raise ValueError("Native standard-MoE sidecar encoding is not canonical.")
    expected_nbytes = _integer(artifact.get("nbytes"), "artifact.nbytes", minimum=1)
    expected_sha256 = _lower_sha256(artifact.get("sha256"), "artifact.sha256")
    if artifact_path.stat().st_size != expected_nbytes:
        raise ValueError("Native standard-MoE parameter sidecar size does not match metadata.")

    raw_geometry = _mapping(contract.get("geometry"), "inference_contract.geometry")
    required_geometry = {
        "max_seq_len", "vocab_size", "padded_vocab_size", "num_layers", "model_dim",
        "hidden_dim", "num_heads", "num_kv_heads", "head_dim", "experts", "top_k",
        "rope_theta", "rope_scaling_factor", "rms_norm_eps", "mlp_multiplier",
        "multiple_of", "router_aux_loss_coef",
    }
    if set(raw_geometry) != required_geometry:
        raise ValueError("Native standard-MoE checkpoint geometry fields are not canonical.")
    geometry: dict[str, int | float] = {
        field: _integer(raw_geometry.get(field), f"geometry.{field}", minimum=1)
        for field in (
            "max_seq_len", "vocab_size", "padded_vocab_size", "num_layers", "model_dim",
            "hidden_dim", "num_heads", "num_kv_heads", "head_dim", "experts", "top_k",
        )
    }
    geometry["multiple_of"] = _integer(raw_geometry.get("multiple_of"), "geometry.multiple_of")
    for field in ("rope_theta", "rope_scaling_factor", "rms_norm_eps", "mlp_multiplier"):
        geometry[field] = _number(raw_geometry.get(field), f"geometry.{field}", positive=True)
    geometry["router_aux_loss_coef"] = _number(
        raw_geometry.get("router_aux_loss_coef"), "geometry.router_aux_loss_coef", nonnegative=True
    )
    router_aux_loss_coef = float(geometry["router_aux_loss_coef"])
    if router_aux_loss_coef != 0.0 and not (
        1.401298464324817e-45 <= router_aux_loss_coef <= 3.4028234663852886e38
    ):
        raise ValueError(
            "Native standard-MoE router_aux_loss_coef is not representable as finite nonzero float32."
        )
    if int(geometry["padded_vocab_size"]) < int(geometry["vocab_size"]):
        raise ValueError("Native standard-MoE padded vocabulary is too small.")
    if int(geometry["model_dim"]) != int(geometry["num_heads"]) * int(geometry["head_dim"]):
        raise ValueError("Native standard-MoE head geometry does not reconstruct model_dim.")
    if int(geometry["num_heads"]) % int(geometry["num_kv_heads"]) or int(geometry["head_dim"]) % 2:
        raise ValueError("Native standard-MoE GQA/RoPE head geometry is invalid.")
    if int(geometry["top_k"]) > int(geometry["experts"]):
        raise ValueError("Native standard-MoE top_k exceeds experts.")
    if float(geometry["rope_scaling_factor"]) != 1.0 or float(geometry["rms_norm_eps"]) != 1.0e-6:
        raise ValueError("Native standard-MoE requires unscaled RoPE and RMSNorm epsilon 1e-6.")
    if _derived_hidden_dim(
        int(geometry["model_dim"]), float(geometry["mlp_multiplier"]), int(geometry["multiple_of"])
    ) != int(geometry["hidden_dim"]):
        raise ValueError("Native standard-MoE hidden_dim is inconsistent with width metadata.")
    if _number(
        payload.get("router_aux_loss_coef"),
        "router_aux_loss_coef",
        nonnegative=True,
    ) != float(geometry["router_aux_loss_coef"]):
        raise ValueError("Native standard-MoE router auxiliary-loss coefficients disagree.")

    semantics = dict(_mapping(contract.get("semantics"), "inference_contract.semantics"))
    if semantics != _STANDARD_MOE_SEMANTICS:
        raise ValueError("Native standard-MoE checkpoint semantics are not canonical.")
    training = dict(_mapping(contract.get("training"), "inference_contract.training"))
    if _integer(training.get("train_seq_len"), "training.train_seq_len", minimum=1) != int(
        geometry["max_seq_len"]
    ):
        raise ValueError("Native standard-MoE training and context lengths disagree.")
    source_graph = _mapping(training.get("source_graph"), "training.source_graph")
    if set(source_graph) != {"filename", "sha256", "byte_identity_verified"}:
        raise ValueError("Native standard-MoE source-graph provenance fields are not canonical.")
    filename = source_graph.get("filename")
    if (
        not isinstance(filename, str)
        or not filename
        or "\\" in filename
        or Path(filename).name != filename
        or PurePosixPath(filename).is_absolute()
        or PureWindowsPath(filename).is_absolute()
        or bool(PureWindowsPath(filename).drive)
    ):
        raise ValueError("Native standard-MoE source-graph filename is unsafe.")
    _lower_sha256(source_graph.get("sha256"), "training.source_graph.sha256")
    if source_graph.get("byte_identity_verified") is not True:
        raise ValueError("Native standard-MoE source graph must prove byte identity.")

    expected_layout = _expected_layout(geometry)
    tensor_rows = contract.get("tensors")
    legacy = _mapping(payload.get("architecture_parameter_layout"), "architecture_parameter_layout")
    legacy_rows = legacy.get("buffers")
    if not isinstance(tensor_rows, list) or not isinstance(legacy_rows, list) or len(
        tensor_rows
    ) != len(expected_layout) or len(legacy_rows) != len(expected_layout):
        raise ValueError("Native standard-MoE tensor table does not match canonical layout.")
    parsed: list[tuple[str, tuple[int, ...], int, int]] = []
    occupied_end = 0
    for index, ((expected_name, expected_shape), raw_row, raw_legacy) in enumerate(
        zip(expected_layout, tensor_rows, legacy_rows, strict=True)
    ):
        row = _mapping(raw_row, f"tensors[{index}]")
        legacy_row = _mapping(raw_legacy, f"architecture_parameter_layout.buffers[{index}]")
        raw_shape = row.get("shape")
        shape = tuple(_integer(dim, f"tensors[{index}].shape", minimum=1) for dim in raw_shape) if isinstance(
            raw_shape, list
        ) else ()
        nbytes = math.prod(shape) * _FLOAT32_BYTES if shape else 0
        offset = _integer(row.get("offset"), f"tensors[{index}].offset")
        if row.get("name") != expected_name or shape != expected_shape or row.get("nbytes") != nbytes:
            raise ValueError("Native standard-MoE tensor names, shapes, or extents are not canonical.")
        if offset != occupied_end or row.get("dtype") != "float32" or row.get(
            "byte_order"
        ) != "little" or row.get("layout") != "row_major":
            raise ValueError("Native standard-MoE tensor table is not exactly contiguous and canonical.")
        if legacy_row.get("name") != expected_name or legacy_row.get("byte_offset") != offset or legacy_row.get(
            "bytes"
        ) != nbytes or legacy_row.get("elements") != nbytes // _FLOAT32_BYTES:
            raise ValueError("Native standard-MoE legacy and inference tensor tables disagree.")
        parsed.append((expected_name, shape, offset, nbytes))
        occupied_end += nbytes
    if occupied_end != expected_nbytes:
        raise ValueError("Native standard-MoE tensor layout does not cover the entire sidecar.")

    parameter_data = _mapping(payload.get("parameter_data"), "parameter_data")
    parameter_state = _mapping(payload.get("native_parameter_state"), "native_parameter_state")
    writer = _mapping(payload.get("writer_verification"), "writer_verification")
    elements = expected_nbytes // _FLOAT32_BYTES
    if parameter_data.get("parameter_dtype") != "float32" or parameter_data.get("bytes") != expected_nbytes or any(
        value != elements
        for value in (
            parameter_data.get("parameter_elements"), parameter_state.get("parameter_elements"),
            parameter_state.get("persisted_parameter_elements"), parameter_state.get("trained_parameter_elements"),
        )
    ) or writer.get("passed") is not True or writer.get("parameter_sidecar_exists") is not True or writer.get(
        "parameter_sidecar_size_matches"
    ) is not True or parameter_state.get("full_template_parameter_state") is not True or parameter_state.get(
        "architecture_forward_inference_supported"
    ) is not True:
        raise ValueError("Native standard-MoE v1 metadata does not prove a complete live checkpoint.")
    for field, raw_path in (
        ("parameter_data.path", parameter_data.get("path")),
        ("native_parameter_state.parameter_data_path", parameter_state.get("parameter_data_path")),
    ):
        if not isinstance(raw_path, str) or Path(raw_path).name != artifact_path.name:
            raise ValueError(f"Native standard-MoE {field} disagrees with the inference artifact.")

    file_digest = hashlib.sha256()
    tensors: list[NativeFamilyStandardMoeTensor] = []
    with artifact_path.open("rb") as handle:
        for name, shape, offset, nbytes in parsed:
            if handle.tell() != offset:
                raise ValueError("Native standard-MoE tensor table is not exactly contiguous.")
            tensors.append(
                NativeFamilyStandardMoeTensor(
                    name=name, source_name=name, shape=shape, offset=offset, nbytes=nbytes,
                    sha256=_read_tensor_and_hash(handle, nbytes, file_digest),
                )
            )
        if handle.read(1):
            raise ValueError("Native standard-MoE sidecar has trailing bytes.")
    actual_sha256 = file_digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("Native standard-MoE parameter sidecar failed SHA-256 validation.")
    return NativeFamilyStandardMoeCheckpointInfo(
        metadata_path=metadata_path,
        artifact_path=artifact_path,
        done_marker_path=done_marker_path,
        preset=preset,
        geometry=geometry,
        semantics=semantics,
        training=training,
        file_size=expected_nbytes,
        sha256=actual_sha256,
        tensors=tuple(tensors),
    )


__all__ = [
    "NATIVE_FAMILY_STANDARD_MOE_CHECKPOINT_FORMAT",
    "NATIVE_FAMILY_STANDARD_MOE_INFERENCE_SCHEMA",
    "NATIVE_FAMILY_STANDARD_MOE_INFERENCE_VERSION",
    "NativeFamilyStandardMoeCheckpointInfo",
    "NativeFamilyStandardMoeTensor",
    "inspect_native_family_standard_moe_checkpoint",
]
