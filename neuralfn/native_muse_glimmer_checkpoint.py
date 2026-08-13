"""Strict, bounded-memory Muse Glimmer BF16 checkpoint ingestion.

The importer intentionally supports only the pinned Meta Muse Glimmer main and
assistant revisions.  It validates complete tensor allowlists and safetensors
layout before copying any payload into a relocatable NeuralFn sidecar.  It does
not import Torch, NumPy, safetensors, or mmap the complete model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any, BinaryIO, Iterable, Mapping, Sequence
import uuid

from .native_chat import (
    MUSE_GLIMMER_ADDED_TOKENS_SHA256,
    MUSE_GLIMMER_ARTIFACT_REVISION,
    MUSE_GLIMMER_ATEM_TEMPLATE_SHA256,
    MUSE_GLIMMER_TOKENIZER_CONFIG_SHA256,
    MUSE_GLIMMER_TOKENIZER_SHA256,
)


MAIN_FORMAT = "neuralfn.native_family_muse_glimmer.bf16.v1"
VISION_FORMAT = "neuralfn.native_family_muse_glimmer_vision.bf16.v1"
ASSISTANT_FORMAT = "neuralfn.native_family_muse_glimmer_dflash.bf16.v1"
NATIVE_LORA_FORMAT = "neuralfn.native_muse_glimmer_lora.bf16.v1"
NATIVE_TRAIN_TOPOLOGY_SHA256 = (
    "4e3c741890fefc43e1027c61ffcfe9ec09c8f6448ee27dd548ab3fad3c172a56"
)
MAIN_CONFIG_SHA256 = "5a9df2d8a385b3d361ab6ae68d73586f4e775033933bd0cd863fb7f3820e6a14"
MAIN_INDEX_SHA256 = "7d817b4dccb1b123fc6c1939356c65cee3a0ad462a5b821ac88280990a27d1ba"
MAIN_PROCESSOR_CONFIG_SHA256 = "97e2a486dd9866b81f40cf4b8bc0c9ced9a7cd8a5bc65aa4cc2f4de0712dae77"
MAIN_GENERATION_CONFIG_SHA256 = "1fa51889b1f8d3659802dedaa27e005b81e5c58483f13ecf13f2d97306bc6e35"
MAIN_SHARDS = {
    "model-00001-of-00002.safetensors": (
        49_950_112_952,
        "8eef61530e1283642c77ce2e6721feb5c6f348fa055c00e90f2844a136372694",
    ),
    "model-00002-of-00002.safetensors": (
        9_603_322_320,
        "b58cc2144ba1ba1af4420f67f4ca3ced7f09298510b80464cc75018a0be14381",
    ),
}
MAIN_PARAMETER_COUNT = 29_776_626_688
MAIN_PAYLOAD_BYTES = MAIN_PARAMETER_COUNT * 2
MAIN_TEXT_PAYLOAD_BYTES = 55_709_561_856

ASSISTANT_REVISION = "e8192f3a8f617f74be2ce220360c89ef4789f39f"
ASSISTANT_CONFIG_SHA256 = "38915167b64b1e6405492aacae5b1b4511b6431163d2960b9bd25821df6fa30a"
ASSISTANT_SHARD = (
    "model.safetensors",
    5_111_976_608,
    "fd88d337eb84f8d0e6ba33a7684d7efa6722d4460ba4d6badca9699418392a84",
)
ASSISTANT_PARAMETER_COUNT = 2_555_985_152
ASSISTANT_PAYLOAD_BYTES = ASSISTANT_PARAMETER_COUNT * 2

_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
_COPY_CHUNK_BYTES = 8 * 1024 * 1024
_SHA256_HEX = frozenset("0123456789abcdef")


class MuseGlimmerCheckpointError(ValueError):
    """A strict source or converted-checkpoint contract violation."""


@dataclass(frozen=True, slots=True)
class MuseGlimmerTensorContract:
    source_name: str
    native_name: str
    shape: tuple[int, ...]
    component: str
    parameterization: str = "ordinary"

    @property
    def elements(self) -> int:
        value = 1
        for dimension in self.shape:
            value *= dimension
        return value

    @property
    def nbytes(self) -> int:
        return self.elements * 2


@dataclass(frozen=True, slots=True)
class SafeTensorEntry:
    contract: MuseGlimmerTensorContract
    shard_path: Path
    shard_name: str
    data_offset: int
    nbytes: int
    dtype: str


@dataclass(frozen=True, slots=True)
class SafeTensorBundle:
    root: Path
    entries: tuple[SafeTensorEntry, ...]
    shard_sha256: Mapping[str, str]
    shard_nbytes: Mapping[str, int]
    payload_bytes: int
    parameter_count: int


@dataclass(frozen=True, slots=True)
class ConvertedMuseGlimmerCheckpoint:
    metadata_path: Path
    artifact_path: Path
    done_path: Path
    format: str
    component: str
    nbytes: int
    sha256: str
    tensor_count: int
    execution_manifest_path: Path | None = None
    execution_done_path: Path | None = None


def _execution_variant_descriptor(
    converted: ConvertedMuseGlimmerCheckpoint,
) -> dict[str, Any]:
    minimum_total = converted.nbytes + 4 * 1024**3
    return {
        "format": converted.format,
        "artifact_path": converted.artifact_path.name,
        "target_nbytes": converted.nbytes,
        "target_sha256": converted.sha256,
        "component": converted.component,
        "weight_precision": "bf16",
        "required_kernel_profile": "muse-glimmer-bf16-mapped-v1",
        "resident_weight_bytes": converted.nbytes,
        "peak_load_staging_bytes": 0,
        "max_workspace_bytes": 64 * 1024**2,
        "memory_profile": {
            "version": 1,
            "minimum_total_vram_bytes": minimum_total,
            "backend_fingerprint": "muse-glimmer-bf16-cpu-cuda-v1",
            "fixed_runtime_bytes": 0,
            "kv_cache_bytes_per_context_token_per_session": 0,
            "session_bytes": 0,
            "hybrid_kv_cache": {
                "local_layers": 39,
                "global_layers": 13,
                "local_window": 2_048,
                "kv_heads": 2,
                "head_dim": 128,
                "key_value_components": 2,
                "bytes_per_element": 2,
                "final_hidden_elements": 6_656,
            },
        },
    }


def _base_muse_glimmer_execution_manifest(
    *,
    model_spec: Mapping[str, Any],
    tensors: Sequence[Mapping[str, Any]],
    primary_variant: str,
    variants: Mapping[str, Mapping[str, Any]],
    companion_checkpoints: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    companions = {
        str(name): dict(descriptor)
        for name, descriptor in (companion_checkpoints or {}).items()
    }
    dflash_ready = "dflash" in companions
    primary = variants.get(primary_variant)
    if not isinstance(primary, Mapping):
        raise MuseGlimmerCheckpointError("Primary Glimmer checkpoint variant is absent")
    embedded_vision = primary.get("component") == "full"
    vision_ready = embedded_vision or "mmproj" in companions
    mmproj_capabilities = (
        companions.get("mmproj", {}).get("capabilities", {})
        if isinstance(companions.get("mmproj", {}), Mapping)
        else {}
    )
    video_ready = embedded_vision or (
        isinstance(mmproj_capabilities, Mapping)
        and mmproj_capabilities.get("video") is True
    )
    checkpoint = {
        key: primary[key]
        for key in ("format", "artifact_path", "target_nbytes", "target_sha256")
    }
    return {
        "schema": "neuralfn.native_execution_manifest",
        "version": 1,
        "source_graph": {
            "kind": "pinned_transformers_model",
            "repository": "meta-models/Muse-Glimmer-30B",
            "revision": MUSE_GLIMMER_ARTIFACT_REVISION,
            "config_sha256": MAIN_CONFIG_SHA256,
            "transformers_revision": "d1123114da1ab4395198146f4f84dae7fe8b693e",
        },
        "model": {
            "family": "muse_glimmer",
            "model_type": "muse_glimmer",
            "architecture": "MuseGlimmerForConditionalGeneration",
            "family_class": "autoregressive_transformer",
            "text_generation": True,
            "template_preset": "muse_glimmer",
            "template_spec": dict(model_spec),
            "modalities": [
                "text",
                *(["image"] if vision_ready else []),
                *(["video"] if video_ready else []),
            ],
        },
        "topology": {
            "profile": "muse-glimmer-text-decoder-v1",
            "target_layers": 52,
            "local_global_pattern": ["local", "local", "local", "global"],
            "checkpoint_tensor_order": "canonical_contiguous_row_major",
        },
        "tensors": [dict(row) for row in tensors],
        "tokenizer": {
            "family": "hf_tokenizer_json",
            "backend": "tokenizers",
            "tokenization": "hf_tokenizer_json",
            "artifact_path": "tokenizer.json",
            "sha256": MUSE_GLIMMER_TOKENIZER_SHA256,
            "vocab_size": 202_048,
            "revision": MUSE_GLIMMER_ARTIFACT_REVISION,
            "config_artifact_path": "tokenizer_config.json",
            "config_sha256": MUSE_GLIMMER_TOKENIZER_CONFIG_SHA256,
            "added_tokens_sha256": MUSE_GLIMMER_ADDED_TOKENS_SHA256,
        },
        "chat_template": {
            "format": "muse_glimmer_atem_v1",
            "artifact_path": "chat_template.jinja",
            "sha256": MUSE_GLIMMER_ATEM_TEMPLATE_SHA256,
            "defaults": {
                "reasoning_strength": "high",
                "knowledge_cutoff": "2026-01-04",
            },
            "supported_content": ["system_text", "user_text", "assistant_text"],
        },
        "context_limits": {
            "max_context_tokens": 131_072,
            "sliding_window_tokens": 2_048,
        },
        "stop_tokens": [200_001, 200_008],
        "kernel_abi": {
            "resident_inference": {"version": 1, "status": "ready"},
            "weight_profiles": [
                str(variant["required_kernel_profile"])
                for variant in variants.values()
            ],
            "whole_model_cuda": {
                "version": 1,
                "status": "ready",
                "profile": "muse-glimmer-hybrid-gqa-bf16-cache-v1",
                "feature_abi_symbol": "nfn_native_tile_glimmer_inference_abi_version",
                "load_operation": "load_model_with_options",
            },
            "speculative_decoding": (
                {
                    "version": 1,
                    "status": "ready",
                    "profile": "muse-glimmer-dflash-block16-v1",
                    "load_operation": "load_companion",
                    "decode_operation": "decode_speculative_block",
                    "block_size": 16,
                    "proposal_tokens": 15,
                }
                if dflash_ready
                else {"version": 0, "status": "unavailable"}
            ),
            "media_encoder": (
                {
                    "version": 1,
                    "status": "ready",
                    "profile": "muse-glimmer-vision-packed-patches-v1",
                    "load_operation": (
                        "embedded" if embedded_vision else "load_companion"
                    ),
                    "encode_operation": "encode_media",
                    "prefill_operation": "prefill_with_embeddings",
                    "projection_width": 6_656,
                }
                if vision_ready
                else {"version": 0, "status": "unavailable"}
            ),
        },
        "checkpoint": checkpoint,
        "primary_checkpoint_variant": primary_variant,
        "checkpoint_variants": {
            str(name): dict(descriptor) for name, descriptor in variants.items()
        },
        "companion_checkpoints": companions,
        "speculative_decoding": (
            {
                "profile": "muse-glimmer-dflash-block16-v1",
                "assistant_checkpoint": "dflash",
                "block_size": 16,
                "proposal_tokens": 15,
                "target_layer_ids_zero_based": [1, 13, 25, 37, 49],
                "mask_token_id": 201_818,
                "modes": ["greedy", "lossless_sampling"],
                "cache_abi": "muse_glimmer_target_assistant_transactional_v1",
            }
            if dflash_ready
            else {}
        ),
        "session_state_kinds": [
            "muse_glimmer_hybrid_lossless_kv_v1",
            *(["muse_glimmer_dflash_context_rollback_v1"] if dflash_ready else []),
        ],
        "capabilities": {
            "native_inference": True,
            "resident_inference": True,
            "lossless_kv_cache": True,
            "turboquant_kv_cache": False,
            "serve": True,
            "text": True,
            "whole_model_cuda": True,
            "speculative_decoding": dflash_ready,
            "vision": vision_ready,
            "video": video_ready,
            "native_pretrain": False,
            "post_training": False,
        },
    }


def build_muse_glimmer_execution_manifest_payload(
    converted: ConvertedMuseGlimmerCheckpoint,
    checkpoint_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact text-decoder Native Execution Manifest.

    The full BF16 artifact contains embedded vision weights, but this manifest
    intentionally grants only the independently proved text capability.
    """

    if converted.component not in {"text", "full"} or converted.format != MAIN_FORMAT:
        raise MuseGlimmerCheckpointError(
            "A runnable target manifest requires a text or full main BF16 conversion"
        )
    expected_nbytes = (
        MAIN_TEXT_PAYLOAD_BYTES if converted.component == "text" else MAIN_PAYLOAD_BYTES
    )
    expected_tensors = 627 if converted.component == "text" else 1_436
    if converted.nbytes != expected_nbytes or converted.tensor_count != expected_tensors:
        raise MuseGlimmerCheckpointError(
            "Converted checkpoint does not have the canonical main target extent"
        )
    artifact = checkpoint_metadata.get("artifact")
    tensors = checkpoint_metadata.get("tensors")
    if (
        checkpoint_metadata.get("schema") != "neuralfn.native_muse_glimmer_checkpoint"
        or checkpoint_metadata.get("version") != 1
        or checkpoint_metadata.get("component") != converted.component
        or not isinstance(artifact, Mapping)
        or artifact.get("target_nbytes") != converted.nbytes
        or artifact.get("target_sha256") != converted.sha256
        or not isinstance(tensors, Sequence)
        or isinstance(tensors, (str, bytes))
        or len(tensors) != converted.tensor_count
    ):
        raise MuseGlimmerCheckpointError(
            "Converted checkpoint metadata does not bind the target artifact"
        )
    from .config import build_muse_glimmer_spec

    model_spec = json.loads(json.dumps(asdict(build_muse_glimmer_spec())))
    manifest_tensors: list[dict[str, Any]] = []
    for row in tensors:
        if not isinstance(row, Mapping):
            raise MuseGlimmerCheckpointError("Converted tensor metadata is malformed")
        manifest_tensors.append(
            {
                "name": row["name"],
                "source_name": row["source_name"],
                "dtype": row["dtype"],
                "shape": list(row["shape"]),
                "offset": row["offset"],
                "nbytes": row["nbytes"],
                "sha256": row["sha256"],
                "role": "parameter",
                "byte_order": row["byte_order"],
                "layout": row["layout"],
                "parameterization": row["parameterization"],
                "component": row["component"],
            }
        )
    variant = _execution_variant_descriptor(converted)
    return _base_muse_glimmer_execution_manifest(
        model_spec=model_spec,
        tensors=manifest_tensors,
        primary_variant="bf16",
        variants={"bf16": variant},
    )


def publish_muse_glimmer_execution_bundle(
    source_root: str | Path,
    converted: ConvertedMuseGlimmerCheckpoint,
) -> ConvertedMuseGlimmerCheckpoint:
    source = Path(source_root).expanduser().resolve()
    output = converted.artifact_path.parent.resolve()
    manifest_path = output / "native-execution-manifest.json"
    done_path = output / "native-execution-DONE"
    assets = (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
    )
    for path in (manifest_path, done_path, *(output / name for name in assets)):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite execution artifact file: {path}")
    try:
        checkpoint_metadata = _load_strict_json(
            converted.metadata_path, label=converted.metadata_path.name
        )
        manifest = build_muse_glimmer_execution_manifest_payload(
            converted, checkpoint_metadata
        )
        nonce = uuid.uuid4().hex
        for name in assets:
            source_path = _safe_shard_path(source, name)
            temporary = output / f".{name}.{nonce}.tmp"
            with source_path.open("rb") as source_stream, temporary.open("xb") as target:
                while True:
                    chunk = source_stream.read(_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, output / name)
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        manifest_tmp = output / f".{manifest_path.name}.{nonce}.tmp"
        with manifest_tmp.open("xb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(manifest_tmp, manifest_path)
        done_tmp = output / f".{done_path.name}.{nonce}.tmp"
        with done_tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(
                {
                    "schema": "neuralfn.native_execution_bundle.done",
                    "version": 1,
                    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "checkpoint_sha256": converted.sha256,
                },
                stream,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(done_tmp, done_path)
    except Exception:
        for path in (manifest_path, done_path, *(output / name for name in assets)):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        for path in output.glob(".*.tmp"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    return replace(
        converted,
        execution_manifest_path=manifest_path,
        execution_done_path=done_path,
    )


def _dflash_companion_descriptor(
    converted: ConvertedMuseGlimmerCheckpoint,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = metadata.get("artifact")
    compatibility = metadata.get("target_compatibility")
    tokenizer = metadata.get("tokenizer")
    if (
        converted.component != "assistant"
        or converted.format != ASSISTANT_FORMAT
        or converted.nbytes != ASSISTANT_PAYLOAD_BYTES
        or converted.tensor_count != 58
        or metadata.get("schema") != "neuralfn.native_muse_glimmer_checkpoint"
        or metadata.get("version") != 1
        or metadata.get("component") != "assistant"
        or not isinstance(artifact, Mapping)
        or artifact.get("target_nbytes") != converted.nbytes
        or artifact.get("target_sha256") != converted.sha256
        or not isinstance(compatibility, Mapping)
        or compatibility.get("target_layer_ids_zero_based") != [1, 13, 25, 37, 49]
        or compatibility.get("block_size") != 16
        or compatibility.get("mask_token_id") != 201_818
        or compatibility.get("shared_lm_head") is not True
        or not isinstance(tokenizer, Mapping)
        or tokenizer.get("tokenizer_sha256") != MUSE_GLIMMER_TOKENIZER_SHA256
    ):
        raise MuseGlimmerCheckpointError(
            "Converted DFlash metadata does not satisfy the resident companion contract"
        )
    allowed = compatibility.get("allowed_target_checkpoint_sha256")
    if (
        not isinstance(allowed, Sequence)
        or isinstance(allowed, (str, bytes))
        or not allowed
        or any(not isinstance(value, str) or not _valid_sha256(value) for value in allowed)
    ):
        raise MuseGlimmerCheckpointError(
            "Converted DFlash metadata has no valid target checkpoint allowlist"
        )
    return {
        "format": ASSISTANT_FORMAT,
        "artifact_path": "muse-glimmer-dflash.bf16",
        "target_nbytes": converted.nbytes,
        "target_sha256": converted.sha256,
        "component": "dflash",
        "weight_precision": "bf16",
        "required_kernel_profile": "muse-glimmer-dflash-bf16-mapped-v1",
        "resident_weight_bytes": converted.nbytes,
        "max_workspace_bytes": 16 * 202_048 * 4 + 64 * 1024**2,
        "target_compatibility": {
            "allowed_target_checkpoint_sha256": list(allowed),
            "target_layer_ids_zero_based": [1, 13, 25, 37, 49],
            "block_size": 16,
            "proposal_tokens": 15,
            "mask_token_id": 201_818,
            "shared_embedding": True,
            "shared_lm_head": True,
            "target_config_sha256": MAIN_CONFIG_SHA256,
            "tokenizer_sha256": MUSE_GLIMMER_TOKENIZER_SHA256,
            "chat_template_sha256": MUSE_GLIMMER_ATEM_TEMPLATE_SHA256,
        },
        "source": {
            "repository": "meta-models/Muse-Glimmer-30B-assistant",
            "revision": ASSISTANT_REVISION,
            "config_sha256": ASSISTANT_CONFIG_SHA256,
        },
        "capabilities": {
            "resident_cpu": True,
            "resident_cuda": True,
            "greedy": True,
            "lossless_sampling": True,
        },
    }


def _sha256_file_range(path: Path, offset: int, nbytes: int) -> str:
    digest = hashlib.sha256()
    remaining = nbytes
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            chunk = stream.read(min(_COPY_CHUNK_BYTES, remaining))
            if not chunk:
                raise MuseGlimmerCheckpointError(
                    f"Adapter tensor at byte {offset} is truncated"
                )
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def inspect_native_muse_glimmer_lora_checkpoint(
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Authenticate a production native LoRA checkpoint and return its companion descriptor."""

    root = Path(checkpoint_dir).expanduser().resolve()
    manifest_path = root / "adapter_manifest.json"
    adapter_path = root / "adapter.bf16"
    if not root.is_dir() or not manifest_path.is_file() or not adapter_path.is_file():
        raise MuseGlimmerCheckpointError(
            "Native LoRA checkpoint requires adapter_manifest.json and adapter.bf16"
        )
    manifest = _load_strict_json(manifest_path, label=manifest_path.name)
    if not isinstance(manifest, Mapping):
        raise MuseGlimmerCheckpointError("Native LoRA manifest root must be an object")
    targets = manifest.get("targets")
    training_adapter = str(manifest.get("training_adapter") or "lora")
    training_base_precision = str(
        manifest.get("training_base_precision") or "bf16"
    )
    allowed_targets = {
        "q_proj", "k_proj", "v_proj", "o_proj", "attn_gate_proj",
        "gate_proj", "up_proj", "down_proj",
    }


    if (
        manifest.get("format") != NATIVE_LORA_FORMAT
        or manifest.get("architecture") != "muse_glimmer"
        or manifest.get("base_weight_precision") != "bf16"
        or training_adapter not in {"lora", "qlora"}
        or training_base_precision
        != (
            "nf4-group64-fp32-scale"
            if training_adapter == "qlora"
            else "bf16"
        )
        or manifest.get("layers") != 52
        or manifest.get("hidden_size") != 6_656
        or manifest.get("attention_size") != 4_096
        or manifest.get("kv_size") != 256
        or manifest.get("intermediate_size") != 19_968
        or manifest.get("adapter_path") != "adapter.bf16"
        or manifest.get("dtype") != "bfloat16"
        or not _valid_sha256(manifest.get("adapter_sha256"))
        or not _valid_sha256(manifest.get("base_sha256"))
        or manifest.get("graph_topology_sha256") != NATIVE_TRAIN_TOPOLOGY_SHA256
        or not _valid_sha256(manifest.get("graph_fingerprint"))
        or not _valid_sha256(manifest.get("tokenizer_sha256"))
        or not _valid_sha256(manifest.get("chat_template_sha256"))
        or not isinstance(targets, Sequence)
        or isinstance(targets, (str, bytes))
        or not targets
        or any(not isinstance(target, str) for target in targets)
        or len(targets) != len(set(targets))
        or not set(targets) <= allowed_targets
    ):
        raise MuseGlimmerCheckpointError(
            "Native LoRA manifest architecture/provenance contract is invalid"
        )
    try:
        rank = int(manifest.get("rank"))
        alpha = float(manifest.get("alpha"))
        scaling = float(manifest.get("scaling"))
        dropout = float(manifest.get("dropout"))
        seed = int(manifest.get("seed"))
    except (TypeError, ValueError) as exc:
        raise MuseGlimmerCheckpointError(
            "Native LoRA rank/alpha/dropout/seed metadata is invalid"
        ) from exc
    if (
        rank <= 0
        or rank > 6_656
        or alpha <= 0.0
        or scaling != alpha / rank
        or not 0.0 <= dropout < 1.0
        or seed < 0
        or seed >= 2**64
    ):
        raise MuseGlimmerCheckpointError(
            "Native LoRA rank/alpha/scaling/dropout/seed contract is invalid"
        )
    shapes = {
        "q_proj": (4_096, 6_656),
        "k_proj": (256, 6_656),
        "v_proj": (256, 6_656),
        "attn_gate_proj": (4_096, 6_656),
        "o_proj": (6_656, 4_096),
        "gate_proj": (19_968, 6_656),
        "up_proj": (19_968, 6_656),
        "down_proj": (6_656, 19_968),
    }
    canonical_roles = (
        "q_proj", "k_proj", "v_proj", "attn_gate_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    expected: list[tuple[str, int, int]] = []
    target_set = set(targets)
    for layer in range(52):
        for role in canonical_roles:
            if role not in target_set:
                continue
            rows, cols = shapes[role]
            prefix = f"layers.{layer}.{role}.weight"
            expected.extend(
                [
                    (prefix + ".lora_A", rank, cols),
                    (prefix + ".lora_B", rows, rank),
                ]
            )
    tensors = manifest.get("tensors")
    if (
        not isinstance(tensors, Sequence)
        or isinstance(tensors, (str, bytes))
        or len(tensors) != len(expected)
    ):
        raise MuseGlimmerCheckpointError("Native LoRA tensor manifest is incomplete")
    offset = 0
    for index, ((name, rows, cols), raw) in enumerate(zip(expected, tensors, strict=True)):
        nbytes = rows * cols * 2
        if (
            not isinstance(raw, Mapping)
            or raw.get("name") != name
            or raw.get("rows") != rows
            or raw.get("cols") != cols
            or raw.get("byte_offset") != offset
            or raw.get("nbytes") != nbytes
            or not _valid_sha256(raw.get("sha256"))
        ):
            raise MuseGlimmerCheckpointError(
                f"Native LoRA tensor manifest mismatch at index {index}"
            )
        if _sha256_file_range(adapter_path, offset, nbytes) != raw["sha256"]:
            raise MuseGlimmerCheckpointError(
                f"Native LoRA tensor {name!r} failed SHA-256 validation"
            )
        offset += nbytes
    if adapter_path.stat().st_size != offset:
        raise MuseGlimmerCheckpointError(
            "Native LoRA adapter byte extent does not match its tensor table"
        )
    adapter_sha = _require_sha(adapter_path, manifest["adapter_sha256"], expected_nbytes=offset)
    return {
        "format": NATIVE_LORA_FORMAT,
        "artifact_path": "muse-glimmer-lora.bf16",
        "target_nbytes": offset,
        "target_sha256": adapter_sha,
        "component": "lora",
        "weight_precision": "bf16",
        "required_kernel_profile": "muse-glimmer-native-lora-bf16-v1",
        "resident_weight_bytes": offset,
        "max_workspace_bytes": rank * (19_968 + 6_656) * 4,
        "rank": rank,
        "alpha": alpha,
        "scaling": scaling,
        "dropout": dropout,
        "targets": list(targets),
        "graph_topology_sha256": manifest["graph_topology_sha256"],
        "graph_fingerprint": manifest["graph_fingerprint"],
        "target_compatibility": {
            "allowed_target_checkpoint_sha256": [manifest["base_sha256"]],
            "base_weight_precision": "bf16",
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "chat_template_sha256": manifest["chat_template_sha256"],
        },
        "source": {
            "training_format": NATIVE_LORA_FORMAT,
            "training_adapter": training_adapter,
            "training_base_precision": training_base_precision,
            "adapter_manifest_sha256": _require_sha(
                manifest_path, hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            ),
            "seed": seed,
        },
        "capabilities": {
            "resident_cpu": True,
            "resident_cuda": True,
            "adapter_only": True,
            "base_frozen": True,
            "qlora": training_adapter == "qlora",
        },
    }


def attach_native_muse_glimmer_lora(
    target_artifact: str | Path,
    checkpoint_dir: str | Path,
) -> Path:
    """Atomically add an authenticated native LoRA adapter to a target bundle."""

    target = Path(target_artifact).expanduser().resolve()
    manifest_path = (
        target / "native-execution-manifest.json" if target.is_dir() else target
    )
    if manifest_path.name != "native-execution-manifest.json" or not manifest_path.is_file():
        raise MuseGlimmerCheckpointError(
            "LoRA attachment requires a Native Execution target bundle"
        )
    source_root = Path(checkpoint_dir).expanduser().resolve()
    source_adapter = source_root / "adapter.bf16"
    source_metadata = source_root / "adapter_manifest.json"
    descriptor = inspect_native_muse_glimmer_lora_checkpoint(source_root)
    output = manifest_path.parent
    destination = output / "muse-glimmer-lora.bf16"
    metadata_destination = output / "lora-checkpoint.json"
    if destination.exists() or metadata_destination.exists():
        raise FileExistsError("Refusing to overwrite an existing native LoRA companion")
    manifest = _load_strict_json(manifest_path, label=manifest_path.name)
    if not isinstance(manifest, Mapping):
        raise MuseGlimmerCheckpointError("Target manifest root must be an object")
    variants = manifest.get("checkpoint_variants")
    tokenizer = manifest.get("tokenizer")
    chat_template = manifest.get("chat_template")
    companions = manifest.get("companion_checkpoints", {})
    base_digest = descriptor["target_compatibility"][
        "allowed_target_checkpoint_sha256"
    ][0]
    if (
        manifest.get("schema") != "neuralfn.native_execution_manifest"
        or manifest.get("version") != 1
        or not isinstance(variants, Mapping)
        or not isinstance(tokenizer, Mapping)
        or not isinstance(chat_template, Mapping)
        or not isinstance(companions, Mapping)
        or "lora" in companions
        or tokenizer.get("sha256")
        != descriptor["target_compatibility"]["tokenizer_sha256"]
        or chat_template.get("sha256")
        != descriptor["target_compatibility"]["chat_template_sha256"]
        or base_digest
        not in {
            row.get("target_sha256")
            for row in variants.values()
            if isinstance(row, Mapping)
        }
    ):
        raise MuseGlimmerCheckpointError(
            "Target bundle is incompatible with the native LoRA lineage"
        )
    updated = json.loads(json.dumps(manifest))
    updated.setdefault("companion_checkpoints", {})["lora"] = descriptor
    updated.setdefault("kernel_abi", {})["native_lora"] = {
        "version": 1,
        "status": "ready",
        "profile": "muse-glimmer-native-lora-bf16-v1",
        "load_operation": "load_companion",
        "format": NATIVE_LORA_FORMAT,
    }
    updated.setdefault("capabilities", {})["post_training"] = True
    updated["capabilities"]["native_lora"] = True

    done_path = output / "native-execution-DONE"
    if not done_path.is_file():
        raise MuseGlimmerCheckpointError(
            "LoRA attachment requires the target bundle DONE marker"
        )
    done = _load_strict_json(done_path, label=done_path.name)
    if (
        not isinstance(done, Mapping)
        or done.get("schema") != "neuralfn.native_execution_bundle.done"
        or done.get("version") != 1
    ):
        raise MuseGlimmerCheckpointError("Target bundle DONE marker is invalid")

    nonce = uuid.uuid4().hex
    artifact_tmp = output / f".{destination.name}.{nonce}.tmp"
    metadata_tmp = output / f".{metadata_destination.name}.{nonce}.tmp"
    manifest_tmp = output / f".{manifest_path.name}.{nonce}.tmp"
    done_tmp = output / f".{done_path.name}.{nonce}.tmp"
    manifest_rollback = output / f".{manifest_path.name}.{nonce}.rollback"
    done_rollback = output / f".{done_path.name}.{nonce}.rollback"
    published: list[Path] = []
    manifest_replaced = False
    done_replaced = False
    try:
        with manifest_rollback.open("xb") as stream:
            stream.write(manifest_path.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        with done_rollback.open("xb") as stream:
            stream.write(done_path.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256()
        copied = 0
        with source_adapter.open("rb") as source, artifact_tmp.open("xb") as sink:
            while True:
                chunk = source.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                sink.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            sink.flush()
            os.fsync(sink.fileno())
        if (
            copied != descriptor["target_nbytes"]
            or digest.hexdigest() != descriptor["target_sha256"]
        ):
            raise MuseGlimmerCheckpointError(
                "Native LoRA bytes changed while attaching the companion"
            )
        metadata_bytes = source_metadata.read_bytes()
        if hashlib.sha256(metadata_bytes).hexdigest() != descriptor["source"][
            "adapter_manifest_sha256"
        ]:
            raise MuseGlimmerCheckpointError(
                "Native LoRA manifest changed while attaching the companion"
            )
        with metadata_tmp.open("xb") as stream:
            stream.write(metadata_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        manifest_bytes = (
            json.dumps(updated, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        with manifest_tmp.open("xb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        done_payload = dict(done)
        done_payload["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        done_payload["lora_checkpoint_sha256"] = descriptor["target_sha256"]
        with done_tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(done_payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(artifact_tmp, destination)
        published.append(destination)
        os.replace(metadata_tmp, metadata_destination)
        published.append(metadata_destination)
        os.replace(manifest_tmp, manifest_path)
        manifest_replaced = True
        os.replace(done_tmp, done_path)
        done_replaced = True
        manifest_rollback.unlink()
        done_rollback.unlink()
        return manifest_path
    except Exception:
        restored = True
        try:
            if manifest_replaced:
                os.replace(manifest_rollback, manifest_path)
            if done_replaced:
                os.replace(done_rollback, done_path)
        except Exception:
            restored = False
        cleanup = [artifact_tmp, metadata_tmp, manifest_tmp, done_tmp]
        if restored:
            cleanup.extend(published)
        cleanup.extend((manifest_rollback, done_rollback))
        for path in cleanup:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def attach_converted_muse_glimmer_assistant(
    target_artifact: str | Path,
    converted: ConvertedMuseGlimmerCheckpoint,
) -> Path:
    """Atomically attach a strict BF16 DFlash conversion to a target bundle."""

    target = Path(target_artifact).expanduser().resolve()
    manifest_path = (
        target / "native-execution-manifest.json" if target.is_dir() else target
    )
    if manifest_path.name != "native-execution-manifest.json" or not manifest_path.is_file():
        raise MuseGlimmerCheckpointError(
            "DFlash attachment requires a Native Execution target bundle"
        )
    output = manifest_path.parent
    destination = output / "muse-glimmer-dflash.bf16"
    companion_metadata_path = output / "dflash-checkpoint.json"
    if destination.exists() or companion_metadata_path.exists():
        raise FileExistsError("Refusing to overwrite an existing DFlash companion")
    manifest = _load_strict_json(manifest_path, label=manifest_path.name)
    metadata = _load_strict_json(converted.metadata_path, label=converted.metadata_path.name)
    if not isinstance(manifest, Mapping) or not isinstance(metadata, Mapping):
        raise MuseGlimmerCheckpointError("Target/assistant metadata roots must be objects")
    descriptor = _dflash_companion_descriptor(converted, metadata)
    variants = manifest.get("checkpoint_variants")
    tokenizer = manifest.get("tokenizer")
    if (
        manifest.get("schema") != "neuralfn.native_execution_manifest"
        or manifest.get("version") != 1
        or not isinstance(variants, Mapping)
        or not isinstance(tokenizer, Mapping)
        or tokenizer.get("sha256") != MUSE_GLIMMER_TOKENIZER_SHA256
        or manifest.get("companion_checkpoints") not in ({}, None)
    ):
        raise MuseGlimmerCheckpointError(
            "Target bundle is not an unattached canonical Muse Glimmer artifact"
        )
    target_digests = {
        row.get("target_sha256")
        for row in variants.values()
        if isinstance(row, Mapping)
    }
    allowed = set(
        descriptor["target_compatibility"]["allowed_target_checkpoint_sha256"]
    )
    if not target_digests or not target_digests.issubset(allowed):
        raise MuseGlimmerCheckpointError(
            "DFlash conversion is not bound to every target variant in the bundle"
        )

    updated = json.loads(json.dumps(manifest))
    updated["companion_checkpoints"] = {"dflash": descriptor}
    updated["speculative_decoding"] = {
        "profile": "muse-glimmer-dflash-block16-v1",
        "assistant_checkpoint": "dflash",
        "block_size": 16,
        "proposal_tokens": 15,
        "target_layer_ids_zero_based": [1, 13, 25, 37, 49],
        "mask_token_id": 201_818,
        "modes": ["greedy", "lossless_sampling"],
        "cache_abi": "muse_glimmer_target_assistant_transactional_v1",
    }
    updated["kernel_abi"]["speculative_decoding"] = {
        "version": 1,
        "status": "ready",
        "profile": "muse-glimmer-dflash-block16-v1",
        "load_operation": "load_companion",
        "decode_operation": "decode_speculative_block",
        "block_size": 16,
        "proposal_tokens": 15,
    }
    updated["capabilities"]["speculative_decoding"] = True
    state_kinds = list(updated.get("session_state_kinds", []))
    if "muse_glimmer_dflash_context_rollback_v1" not in state_kinds:
        state_kinds.append("muse_glimmer_dflash_context_rollback_v1")
    updated["session_state_kinds"] = state_kinds

    nonce = uuid.uuid4().hex
    artifact_tmp = output / f".{destination.name}.{nonce}.tmp"
    metadata_tmp = output / f".{companion_metadata_path.name}.{nonce}.tmp"
    manifest_tmp = output / f".{manifest_path.name}.{nonce}.tmp"
    done_path = output / "native-execution-DONE"
    done_tmp = output / f".{done_path.name}.{nonce}.tmp"
    manifest_rollback = output / f".{manifest_path.name}.{nonce}.rollback"
    done_rollback = output / f".{done_path.name}.{nonce}.rollback"
    published: list[Path] = []
    manifest_replaced = False
    done_replaced = False
    try:
        if not done_path.is_file():
            raise MuseGlimmerCheckpointError(
                "DFlash attachment requires the target bundle DONE marker"
            )
        with manifest_rollback.open("xb") as stream:
            stream.write(manifest_path.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        with done_rollback.open("xb") as stream:
            stream.write(done_path.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256()
        copied = 0
        with converted.artifact_path.open("rb") as source, artifact_tmp.open("xb") as sink:
            while True:
                chunk = source.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                sink.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            sink.flush()
            os.fsync(sink.fileno())
        if copied != converted.nbytes or digest.hexdigest() != converted.sha256:
            raise MuseGlimmerCheckpointError(
                "DFlash bytes changed while attaching the companion"
            )
        metadata_bytes = (
            json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        with metadata_tmp.open("xb") as stream:
            stream.write(metadata_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        manifest_bytes = (
            json.dumps(updated, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        with manifest_tmp.open("xb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        done_payload = {
            "schema": "neuralfn.native_execution_bundle.done",
            "version": 1,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "checkpoint_sha256": updated["checkpoint"]["target_sha256"],
            "dflash_checkpoint_sha256": converted.sha256,
        }
        with done_tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(done_payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(artifact_tmp, destination)
        published.append(destination)
        os.replace(metadata_tmp, companion_metadata_path)
        published.append(companion_metadata_path)
        os.replace(manifest_tmp, manifest_path)
        manifest_replaced = True
        os.replace(done_tmp, done_path)
        done_replaced = True
        manifest_rollback.unlink()
        done_rollback.unlink()
        return manifest_path
    except Exception:
        # Restore both authoritative files before removing companion payloads.
        # This also covers a failure between the two atomic replacements.
        restored = True
        try:
            if manifest_replaced:
                os.replace(manifest_rollback, manifest_path)
            if done_replaced:
                os.replace(done_rollback, done_path)
        except Exception:
            restored = False
        cleanup = [artifact_tmp, metadata_tmp, manifest_tmp, done_tmp]
        if restored:
            cleanup.extend(published)
        cleanup.extend((manifest_rollback, done_rollback))
        for path in cleanup:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def _text_contracts() -> tuple[MuseGlimmerTensorContract, ...]:
    contracts = [
        MuseGlimmerTensorContract(
            "model.language_model.embed_tokens.weight",
            "text.embedding.weight",
            (202_048, 6_656),
            "text",
        )
    ]
    norm_mapping = {
        "input_layernorm": "input_norm.centered_weight",
        "post_attention_layernorm": "post_attention_norm.centered_weight",
        "pre_feedforward_layernorm": "pre_feedforward_norm.centered_weight",
        "post_feedforward_layernorm": "post_feedforward_norm.centered_weight",
    }
    projection_shapes = {
        "self_attn.q_proj": (4_096, 6_656),
        "self_attn.k_proj": (256, 6_656),
        "self_attn.v_proj": (256, 6_656),
        "self_attn.gate_proj": (4_096, 6_656),
        "self_attn.o_proj": (6_656, 4_096),
        "mlp.gate_proj": (19_968, 6_656),
        "mlp.up_proj": (19_968, 6_656),
        "mlp.down_proj": (6_656, 19_968),
    }
    native_projection = {
        "self_attn.q_proj": "attention.q.weight",
        "self_attn.k_proj": "attention.k.weight",
        "self_attn.v_proj": "attention.v.weight",
        "self_attn.gate_proj": "attention.gate.weight",
        "self_attn.o_proj": "attention.o.weight",
        "mlp.gate_proj": "mlp.gate.weight",
        "mlp.up_proj": "mlp.up.weight",
        "mlp.down_proj": "mlp.down.weight",
    }
    for layer in range(52):
        prefix = f"model.language_model.layers.{layer}."
        native_prefix = f"text.layers.{layer}."
        for source_suffix, native_suffix in norm_mapping.items():
            contracts.append(
                MuseGlimmerTensorContract(
                    prefix + source_suffix + ".weight",
                    native_prefix + native_suffix,
                    (6_656,),
                    "text",
                    "centered_delta",
                )
            )
        for source_suffix, shape in projection_shapes.items():
            contracts.append(
                MuseGlimmerTensorContract(
                    prefix + source_suffix + ".weight",
                    native_prefix + native_projection[source_suffix],
                    shape,
                    "text",
                )
            )
    contracts.extend(
        [
            MuseGlimmerTensorContract(
                "model.language_model.norm.weight",
                "text.final_norm.weight",
                (6_656,),
                "text",
            ),
            MuseGlimmerTensorContract(
                "lm_head.weight",
                "text.lm_head.weight",
                (202_048, 6_656),
                "text",
            ),
        ]
    )
    return tuple(contracts)


def _vision_contracts() -> tuple[MuseGlimmerTensorContract, ...]:
    contracts = [
        MuseGlimmerTensorContract(
            "model.vision_tower.patch_embedder.patch_embedding.weight",
            "vision.patch_embedding.weight",
            (1_536, 3, 2, 14, 14),
            "vision",
        ),
        MuseGlimmerTensorContract(
            "model.vision_tower.patch_embedder.position_embedding_table.weight",
            "vision.position_embedding.weight",
            (1_024, 1_536),
            "vision",
        ),
    ]
    for source_name, native_name in (
        ("ln_pre", "pre_norm"),
        ("ln_post", "post_norm"),
    ):
        for suffix in ("weight", "bias"):
            contracts.append(
                MuseGlimmerTensorContract(
                    f"model.vision_tower.{source_name}.{suffix}",
                    f"vision.{native_name}.{suffix}",
                    (1_536,),
                    "vision",
                )
            )
    for layer in range(50):
        source_prefix = f"model.vision_tower.layers.{layer}."
        native_prefix = f"vision.layers.{layer}."
        for projection in ("q_proj", "k_proj", "v_proj", "proj"):
            for suffix, shape in (
                ("weight", (1_536, 1_536)),
                ("bias", (1_536,)),
            ):
                contracts.append(
                    MuseGlimmerTensorContract(
                        f"{source_prefix}attn.{projection}.{suffix}",
                        f"{native_prefix}attention.{projection}.{suffix}",
                        shape,
                        "vision",
                    )
                )
        for norm in ("norm1", "norm2"):
            for suffix in ("weight", "bias"):
                contracts.append(
                    MuseGlimmerTensorContract(
                        f"{source_prefix}{norm}.{suffix}",
                        f"{native_prefix}{norm}.{suffix}",
                        (1_536,),
                        "vision",
                    )
                )
        for projection, weight_shape, bias_shape in (
            ("fc1", (8_960, 1_536), (8_960,)),
            ("fc2", (1_536, 8_960), (1_536,)),
        ):
            contracts.extend(
                [
                    MuseGlimmerTensorContract(
                        f"{source_prefix}mlp.{projection}.weight",
                        f"{native_prefix}mlp.{projection}.weight",
                        weight_shape,
                        "vision",
                    ),
                    MuseGlimmerTensorContract(
                        f"{source_prefix}mlp.{projection}.bias",
                        f"{native_prefix}mlp.{projection}.bias",
                        bias_shape,
                        "vision",
                    ),
                ]
            )
    contracts.extend(
        [
            MuseGlimmerTensorContract(
                "model.vision_adapter.fc1.weight",
                "vision.adapter.fc1.weight",
                (4_096, 6_144),
                "vision",
            ),
            MuseGlimmerTensorContract(
                "model.vision_adapter.fc2.weight",
                "vision.adapter.fc2.weight",
                (4_096, 4_096),
                "vision",
            ),
            MuseGlimmerTensorContract(
                "model.vision_projection.weight",
                "vision.projection.weight",
                (6_656, 4_096),
                "vision",
            ),
        ]
    )
    return tuple(contracts)


def muse_glimmer_main_tensor_contracts() -> tuple[MuseGlimmerTensorContract, ...]:
    return _text_contracts() + _vision_contracts()


def muse_glimmer_assistant_tensor_contracts() -> tuple[MuseGlimmerTensorContract, ...]:
    contracts: list[MuseGlimmerTensorContract] = []
    projection_shapes = {
        "self_attn.q_proj.weight": (4_096, 6_656),
        "self_attn.k_proj.weight": (1_024, 6_656),
        "self_attn.v_proj.weight": (1_024, 6_656),
        "self_attn.o_proj.weight": (6_656, 4_096),
        "self_attn.q_norm.weight": (128,),
        "self_attn.k_norm.weight": (128,),
        "mlp.gate_proj.weight": (19_968, 6_656),
        "mlp.up_proj.weight": (19_968, 6_656),
        "mlp.down_proj.weight": (6_656, 19_968),
        "post_attention_layernorm.weight": (6_656,),
        "input_layernorm.weight": (6_656,),
    }
    native_suffixes = {
        "self_attn.q_proj.weight": "attention.q.weight",
        "self_attn.k_proj.weight": "attention.k.weight",
        "self_attn.v_proj.weight": "attention.v.weight",
        "self_attn.o_proj.weight": "attention.o.weight",
        "self_attn.q_norm.weight": "attention.q_norm.weight",
        "self_attn.k_norm.weight": "attention.k_norm.weight",
        "mlp.gate_proj.weight": "mlp.gate.weight",
        "mlp.up_proj.weight": "mlp.up.weight",
        "mlp.down_proj.weight": "mlp.down.weight",
        "post_attention_layernorm.weight": "post_attention_norm.weight",
        "input_layernorm.weight": "input_norm.weight",
    }
    for layer in range(5):
        for suffix, shape in projection_shapes.items():
            contracts.append(
                MuseGlimmerTensorContract(
                    f"layers.{layer}.{suffix}",
                    f"assistant.layers.{layer}.{native_suffixes[suffix]}",
                    shape,
                    "assistant",
                )
            )
    contracts.extend(
        [
            MuseGlimmerTensorContract(
                "norm.weight", "assistant.final_norm.weight", (6_656,), "assistant"
            ),
            MuseGlimmerTensorContract(
                "encoder.fc.weight",
                "assistant.context_projection.weight",
                (6_656, 33_280),
                "assistant",
            ),
            MuseGlimmerTensorContract(
                "encoder.output_norm_enc.weight",
                "assistant.context_norm.weight",
                (6_656,),
                "assistant",
            ),
        ]
    )
    return tuple(contracts)


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MuseGlimmerCheckpointError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_strict_json_bytes(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                MuseGlimmerCheckpointError(f"{label} contains non-finite JSON {value}")
            ),
        )
    except MuseGlimmerCheckpointError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MuseGlimmerCheckpointError(f"{label} is not strict UTF-8 JSON") from exc


def _load_strict_json(path: Path, *, label: str) -> Any:
    try:
        return _load_strict_json_bytes(path.read_bytes(), label=label)
    except OSError as exc:
        raise MuseGlimmerCheckpointError(f"Unable to read {label}: {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(path: Path, expected_sha256: str, *, expected_nbytes: int | None = None) -> str:
    if not path.is_file():
        raise MuseGlimmerCheckpointError(f"Required source file does not exist: {path}")
    if expected_nbytes is not None and path.stat().st_size != expected_nbytes:
        raise MuseGlimmerCheckpointError(
            f"Source size mismatch for {path.name}: expected {expected_nbytes}, "
            f"got {path.stat().st_size}"
        )
    digest = _sha256(path)
    if digest != expected_sha256:
        raise MuseGlimmerCheckpointError(
            f"Source SHA-256 mismatch for {path.name}: expected {expected_sha256}, got {digest}"
        )
    return digest


def _safe_shard_path(root: Path, filename: Any) -> Path:
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or Path(filename).is_absolute()
        or "\\" in filename
    ):
        raise MuseGlimmerCheckpointError(f"Unsafe safetensors shard name {filename!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / filename).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise MuseGlimmerCheckpointError("Safetensors shard escapes its source directory") from exc
    if not resolved.is_file():
        raise MuseGlimmerCheckpointError(f"Safetensors shard does not exist: {resolved}")
    return resolved


def _shape(raw: Any, *, tensor_name: str) -> tuple[int, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise MuseGlimmerCheckpointError(f"Tensor {tensor_name!r} has an invalid shape")
    values: list[int] = []
    for dimension in raw:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise MuseGlimmerCheckpointError(f"Tensor {tensor_name!r} has an invalid shape")
        values.append(dimension)
    return tuple(values)


def _inspect_safetensors_shard(
    path: Path,
    *,
    expected_contracts: Mapping[str, MuseGlimmerTensorContract],
) -> dict[str, tuple[int, int, str, tuple[int, ...]]]:
    file_nbytes = path.stat().st_size
    if file_nbytes < 10:
        raise MuseGlimmerCheckpointError(f"Safetensors shard is truncated: {path}")
    with path.open("rb") as stream:
        raw_header_nbytes = stream.read(8)
        if len(raw_header_nbytes) != 8:
            raise MuseGlimmerCheckpointError(f"Safetensors header is truncated: {path}")
        header_nbytes = struct.unpack("<Q", raw_header_nbytes)[0]
        if header_nbytes < 2 or header_nbytes > _MAX_SAFETENSORS_HEADER_BYTES:
            raise MuseGlimmerCheckpointError(
                f"Safetensors header length is invalid in {path.name}: {header_nbytes}"
            )
        if 8 + header_nbytes >= file_nbytes:
            raise MuseGlimmerCheckpointError(f"Safetensors payload is missing: {path}")
        header_bytes = stream.read(header_nbytes)
        if len(header_bytes) != header_nbytes:
            raise MuseGlimmerCheckpointError(f"Safetensors header is truncated: {path}")
    header = _load_strict_json_bytes(header_bytes, label=f"{path.name} header")
    if not isinstance(header, Mapping):
        raise MuseGlimmerCheckpointError("Safetensors header root must be an object")
    metadata = header.get("__metadata__")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise MuseGlimmerCheckpointError("Safetensors __metadata__ must be an object")
    data_bytes = file_nbytes - 8 - header_nbytes
    intervals: list[tuple[int, int, str]] = []
    parsed: dict[str, tuple[int, int, str, tuple[int, ...]]] = {}
    for name, raw in header.items():
        if name == "__metadata__":
            continue
        if name not in expected_contracts:
            raise MuseGlimmerCheckpointError(
                f"Unexpected safetensors tensor {name!r} in {path.name}"
            )
        if not isinstance(raw, Mapping) or set(raw) != {"dtype", "shape", "data_offsets"}:
            raise MuseGlimmerCheckpointError(
                f"Safetensors tensor {name!r} metadata is not canonical"
            )
        dtype = raw.get("dtype")
        if dtype != "BF16":
            raise MuseGlimmerCheckpointError(
                f"Tensor {name!r} must be BF16, got {dtype!r}"
            )
        parsed_shape = _shape(raw.get("shape"), tensor_name=name)
        expected = expected_contracts[name]
        if parsed_shape != expected.shape:
            raise MuseGlimmerCheckpointError(
                f"Tensor {name!r} shape mismatch: expected {expected.shape}, got {parsed_shape}"
            )
        offsets = raw.get("data_offsets")
        if (
            not isinstance(offsets, Sequence)
            or isinstance(offsets, (str, bytes))
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
        ):
            raise MuseGlimmerCheckpointError(f"Tensor {name!r} has invalid data_offsets")
        start, end = int(offsets[0]), int(offsets[1])
        if start < 0 or end <= start or end > data_bytes or end - start != expected.nbytes:
            raise MuseGlimmerCheckpointError(
                f"Tensor {name!r} byte extent does not match its BF16 shape"
            )
        intervals.append((start, end, name))
        parsed[name] = (8 + header_nbytes + start, end - start, dtype, parsed_shape)
    intervals.sort()
    cursor = 0
    for start, end, name in intervals:
        if start != cursor:
            issue = "overlaps" if start < cursor else "has a gap before"
            raise MuseGlimmerCheckpointError(
                f"Safetensors tensor {name!r} {issue} the preceding payload"
            )
        cursor = end
    if cursor != data_bytes:
        raise MuseGlimmerCheckpointError(
            f"Safetensors tensor table does not cover {path.name}'s complete payload"
        )
    return parsed


def inspect_safetensors_bundle(
    root: str | Path,
    *,
    contracts: Sequence[MuseGlimmerTensorContract],
    shard_assignments: Mapping[str, str],
    shard_authentication: Mapping[str, tuple[int, str]],
) -> SafeTensorBundle:
    source_root = Path(root).expanduser().resolve()
    if not source_root.is_dir():
        raise MuseGlimmerCheckpointError(f"Source directory does not exist: {source_root}")
    contract_by_source = {contract.source_name: contract for contract in contracts}
    if len(contract_by_source) != len(contracts):
        raise MuseGlimmerCheckpointError("Tensor contract contains duplicate source names")
    native_names = {contract.native_name for contract in contracts}
    if len(native_names) != len(contracts):
        raise MuseGlimmerCheckpointError("Tensor contract contains duplicate native names")
    if set(shard_assignments) != set(contract_by_source):
        missing = sorted(set(contract_by_source) - set(shard_assignments))
        unexpected = sorted(set(shard_assignments) - set(contract_by_source))
        raise MuseGlimmerCheckpointError(
            f"Safetensors index tensor allowlist mismatch; missing={missing[:3]}, "
            f"unexpected={unexpected[:3]}"
        )
    assigned_shards = set(shard_assignments.values())
    if assigned_shards != set(shard_authentication):
        raise MuseGlimmerCheckpointError(
            "Safetensors shard authentication table does not match the index"
        )
    shard_sha256: dict[str, str] = {}
    shard_nbytes: dict[str, int] = {}
    parsed_by_shard: dict[str, dict[str, tuple[int, int, str, tuple[int, ...]]]] = {}
    for shard_name in sorted(assigned_shards):
        path = _safe_shard_path(source_root, shard_name)
        expected_nbytes, expected_sha = shard_authentication[shard_name]
        shard_sha256[shard_name] = _require_sha(
            path, expected_sha, expected_nbytes=expected_nbytes
        )
        shard_nbytes[shard_name] = path.stat().st_size
        expected_for_shard = {
            name: contract_by_source[name]
            for name, assigned in shard_assignments.items()
            if assigned == shard_name
        }
        parsed = _inspect_safetensors_shard(path, expected_contracts=expected_for_shard)
        if set(parsed) != set(expected_for_shard):
            missing = sorted(set(expected_for_shard) - set(parsed))
            raise MuseGlimmerCheckpointError(
                f"Safetensors shard {shard_name} is missing tensors {missing[:3]}"
            )
        parsed_by_shard[shard_name] = parsed
    entries: list[SafeTensorEntry] = []
    for contract in contracts:
        shard_name = shard_assignments[contract.source_name]
        data_offset, nbytes, dtype, _ = parsed_by_shard[shard_name][contract.source_name]
        entries.append(
            SafeTensorEntry(
                contract=contract,
                shard_path=_safe_shard_path(source_root, shard_name),
                shard_name=shard_name,
                data_offset=data_offset,
                nbytes=nbytes,
                dtype=dtype,
            )
        )
    return SafeTensorBundle(
        root=source_root,
        entries=tuple(entries),
        shard_sha256=shard_sha256,
        shard_nbytes=shard_nbytes,
        payload_bytes=sum(entry.nbytes for entry in entries),
        parameter_count=sum(entry.contract.elements for entry in entries),
    )


def inspect_safetensors_index(
    root: str | Path,
    *,
    index_filename: str,
    index_sha256: str,
    contracts: Sequence[MuseGlimmerTensorContract],
    shard_authentication: Mapping[str, tuple[int, str]],
    expected_metadata: Mapping[str, Any] | None = None,
) -> SafeTensorBundle:
    source_root = Path(root).expanduser().resolve()
    index_path = _safe_shard_path(source_root, index_filename)
    _require_sha(index_path, index_sha256)
    index = _load_strict_json(index_path, label=index_filename)
    if not isinstance(index, Mapping) or set(index) != {"metadata", "weight_map"}:
        raise MuseGlimmerCheckpointError("Safetensors index fields are not canonical")
    metadata = index.get("metadata")
    if not isinstance(metadata, Mapping):
        raise MuseGlimmerCheckpointError("Safetensors index metadata must be an object")
    if expected_metadata is not None and dict(metadata) != dict(expected_metadata):
        raise MuseGlimmerCheckpointError("Safetensors index totals are not canonical")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise MuseGlimmerCheckpointError("Safetensors index weight_map must be an object")
    assignments: dict[str, str] = {}
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise MuseGlimmerCheckpointError("Safetensors index weight_map is malformed")
        assignments[name] = shard
    return inspect_safetensors_bundle(
        source_root,
        contracts=contracts,
        shard_assignments=assignments,
        shard_authentication=shard_authentication,
    )


def _inspect_official_main(root: Path) -> SafeTensorBundle:
    for filename, expected_sha in (
        ("config.json", MAIN_CONFIG_SHA256),
        ("model.safetensors.index.json", MAIN_INDEX_SHA256),
        ("tokenizer.json", MUSE_GLIMMER_TOKENIZER_SHA256),
        ("tokenizer_config.json", MUSE_GLIMMER_TOKENIZER_CONFIG_SHA256),
        ("chat_template.jinja", MUSE_GLIMMER_ATEM_TEMPLATE_SHA256),
        ("processor_config.json", MAIN_PROCESSOR_CONFIG_SHA256),
        ("generation_config.json", MAIN_GENERATION_CONFIG_SHA256),
    ):
        _require_sha(root / filename, expected_sha)
    bundle = inspect_safetensors_index(
        root,
        index_filename="model.safetensors.index.json",
        index_sha256=MAIN_INDEX_SHA256,
        contracts=muse_glimmer_main_tensor_contracts(),
        shard_authentication=MAIN_SHARDS,
        expected_metadata={
            "total_parameters": MAIN_PARAMETER_COUNT,
            "total_size": MAIN_PAYLOAD_BYTES,
        },
    )
    if (
        bundle.parameter_count != MAIN_PARAMETER_COUNT
        or bundle.payload_bytes != MAIN_PAYLOAD_BYTES
    ):
        raise MuseGlimmerCheckpointError("Main tensor contracts do not match official totals")
    return bundle


def _inspect_official_assistant(root: Path) -> SafeTensorBundle:
    _require_sha(root / "config.json", ASSISTANT_CONFIG_SHA256)
    shard_name, shard_nbytes, shard_sha = ASSISTANT_SHARD
    contracts = muse_glimmer_assistant_tensor_contracts()
    bundle = inspect_safetensors_bundle(
        root,
        contracts=contracts,
        shard_assignments={contract.source_name: shard_name for contract in contracts},
        shard_authentication={shard_name: (shard_nbytes, shard_sha)},
    )
    if (
        bundle.parameter_count != ASSISTANT_PARAMETER_COUNT
        or bundle.payload_bytes != ASSISTANT_PAYLOAD_BYTES
    ):
        raise MuseGlimmerCheckpointError("Assistant tensor contracts do not match official totals")
    return bundle


def inspect_official_muse_glimmer_safetensors(
    source_root: str | Path,
    *,
    assistant: bool = False,
) -> SafeTensorBundle:
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise MuseGlimmerCheckpointError(f"Source directory does not exist: {root}")
    return _inspect_official_assistant(root) if assistant else _inspect_official_main(root)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def _geometry(component: str) -> dict[str, Any]:
    if component == "assistant":
        return {
            "block_size": 16,
            "context_length": 131_072,
            "hidden_size": 6_656,
            "intermediate_size": 19_968,
            "layers": 5,
            "head_dim": 128,
            "query_heads": 32,
            "kv_heads": 8,
            "sliding_window": 2_048,
            "rope_theta": 500_000.0,
            "target_layer_ids_zero_based": [1, 13, 25, 37, 49],
            "mask_token_id": 201_818,
        }
    geometry: dict[str, Any] = {
        "context_length": 131_072,
        "vocab_size": 202_048,
        "hidden_size": 6_656,
        "attention_size": 4_096,
        "intermediate_size": 19_968,
        "layers": 52,
        "head_dim": 128,
        "query_heads": 32,
        "kv_heads": 2,
        "sliding_window": 2_048,
        "layer_types": [
            "sliding_attention" if index % 4 != 3 else "full_attention"
            for index in range(52)
        ],
        "layer_rope_theta": [
            500_000.0 if index % 4 != 3 else 0.0 for index in range(52)
        ],
        "qk_scale_factor": 3.87,
        "norm_eps": 1.0e-5,
        "post_norm_eps": 1.0e-8,
        "output_multiplier": 0.19611613513818404,
        "logit_softcap": 20.0,
        "untied_lm_head": True,
    }
    if component in {"vision", "full"}:
        geometry["vision"] = {
            "hidden_size": 1_536,
            "intermediate_size": 8_960,
            "layers": 50,
            "heads": 16,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "position_grid": [32, 32],
            "adapter_hidden_size": 4_096,
            "projection_width": 6_656,
        }
    return geometry


def _copy_entry(
    source: BinaryIO,
    destination: BinaryIO,
    entry: SafeTensorEntry,
    output_digest: Any,
) -> str:
    source.seek(entry.data_offset)
    remaining = entry.nbytes
    tensor_digest = hashlib.sha256()
    while remaining:
        chunk = source.read(min(remaining, _COPY_CHUNK_BYTES))
        if not chunk:
            raise MuseGlimmerCheckpointError(
                f"Tensor {entry.contract.source_name!r} payload is truncated"
            )
        destination.write(chunk)
        tensor_digest.update(chunk)
        output_digest.update(chunk)
        remaining -= len(chunk)
    return tensor_digest.hexdigest()


def _publish_converted_checkpoint(
    bundle: SafeTensorBundle,
    output_root: Path,
    *,
    entries: Sequence[SafeTensorEntry],
    format_name: str,
    component: str,
    source_revision: str,
    source_files: Mapping[str, Mapping[str, Any]],
    compatible_target_sha256: str | None,
) -> ConvertedMuseGlimmerCheckpoint:
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_name = {
        "text": "muse-glimmer-text.bf16",
        "vision": "muse-glimmer-vision.bf16",
        "full": "muse-glimmer-full.bf16",
        "assistant": "muse-glimmer-dflash.bf16",
    }[component]
    artifact_path = output_root / artifact_name
    metadata_path = output_root / "checkpoint.json"
    done_path = output_root / "DONE"
    for path in (artifact_path, metadata_path, done_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite converted checkpoint file: {path}")
    nonce = uuid.uuid4().hex
    artifact_tmp = output_root / f".{artifact_name}.{nonce}.tmp"
    metadata_tmp = output_root / f".checkpoint.json.{nonce}.tmp"
    done_tmp = output_root / f".DONE.{nonce}.tmp"
    output_digest = hashlib.sha256()
    output_offset = 0
    tensor_rows: list[dict[str, Any]] = []
    handles: dict[Path, BinaryIO] = {}
    try:
        with artifact_tmp.open("xb") as destination:
            for entry in entries:
                source = handles.get(entry.shard_path)
                if source is None:
                    source = entry.shard_path.open("rb")
                    handles[entry.shard_path] = source
                tensor_sha = _copy_entry(source, destination, entry, output_digest)
                tensor_rows.append(
                    {
                        "name": entry.contract.native_name,
                        "source_name": entry.contract.source_name,
                        "component": entry.contract.component,
                        "dtype": "bfloat16",
                        "byte_order": "little",
                        "layout": "row_major",
                        "parameterization": entry.contract.parameterization,
                        "shape": list(entry.contract.shape),
                        "offset": output_offset,
                        "nbytes": entry.nbytes,
                        "sha256": tensor_sha,
                        "source_shard": entry.shard_name,
                        "source_data_offset": entry.data_offset,
                    }
                )
                output_offset += entry.nbytes
            destination.flush()
            os.fsync(destination.fileno())
        output_sha = output_digest.hexdigest()
        source_provenance: dict[str, Any] = {
            "repository": (
                "meta-models/Muse-Glimmer-30B-assistant"
                if component == "assistant"
                else "meta-models/Muse-Glimmer-30B"
            ),
            "revision": source_revision,
            "files": dict(source_files),
            "conversion": {
                "implementation": "neuralfn.native_muse_glimmer_checkpoint",
                "version": 1,
                "payload_transform": "canonical-order-copy-no-transpose",
                "max_copy_buffer_bytes": _COPY_CHUNK_BYTES,
            },
        }
        metadata: dict[str, Any] = {
            "schema": "neuralfn.native_muse_glimmer_checkpoint",
            "version": 1,
            "format": format_name,
            "component": component,
            "artifact": {
                "artifact_path": artifact_name,
                "target_nbytes": output_offset,
                "target_sha256": output_sha,
                "dtype": "bfloat16",
                "byte_order": "little",
                "layout": "canonical_contiguous_row_major",
            },
            "geometry": _geometry(component),
            "tensors": tensor_rows,
            "source_provenance": source_provenance,
            "tokenizer": {
                "revision": MUSE_GLIMMER_ARTIFACT_REVISION,
                "tokenizer_sha256": MUSE_GLIMMER_TOKENIZER_SHA256,
                "tokenizer_config_sha256": MUSE_GLIMMER_TOKENIZER_CONFIG_SHA256,
                "added_tokens_sha256": MUSE_GLIMMER_ADDED_TOKENS_SHA256,
                "chat_template_sha256": MUSE_GLIMMER_ATEM_TEMPLATE_SHA256,
                "vocab_size": 202_048,
            },
            "capabilities": {
                "checkpoint_import": True,
                "resident_cpu": component == "assistant",
                "resident_cuda": False,
                "speculative_decoding": component == "assistant",
                "vision": component == "full",
                "video": False,
            },
        }
        if component == "assistant":
            metadata["target_compatibility"] = {
                "allowed_target_checkpoint_sha256": [compatible_target_sha256],
                "target_layer_ids_zero_based": [1, 13, 25, 37, 49],
                "block_size": 16,
                "mask_token_id": 201_818,
                "shared_lm_head": True,
                "tokenizer_sha256": MUSE_GLIMMER_TOKENIZER_SHA256,
            }
        metadata_bytes = (
            json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        metadata_sha = hashlib.sha256(metadata_bytes).hexdigest()
        with metadata_tmp.open("xb") as stream:
            stream.write(metadata_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        done_payload = {
            "schema": "neuralfn.native_muse_glimmer_checkpoint.done",
            "version": 1,
            "metadata_sha256": metadata_sha,
            "artifact_sha256": output_sha,
            "artifact_nbytes": output_offset,
        }
        with done_tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(done_payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(artifact_tmp, artifact_path)
        os.replace(metadata_tmp, metadata_path)
        os.replace(done_tmp, done_path)
        return ConvertedMuseGlimmerCheckpoint(
            metadata_path=metadata_path,
            artifact_path=artifact_path,
            done_path=done_path,
            format=format_name,
            component=component,
            nbytes=output_offset,
            sha256=output_sha,
            tensor_count=len(entries),
        )
    finally:
        for handle in handles.values():
            handle.close()
        for temporary in (artifact_tmp, metadata_tmp, done_tmp):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def convert_official_muse_glimmer_safetensors(
    source_root: str | Path,
    output_root: str | Path,
    *,
    component: str = "text",
) -> ConvertedMuseGlimmerCheckpoint:
    if component not in {"text", "vision", "full"}:
        raise MuseGlimmerCheckpointError("component must be text, vision, or full")
    root = Path(source_root).expanduser().resolve()
    bundle = _inspect_official_main(root)
    entries = tuple(
        entry
        for entry in bundle.entries
        if component == "full" or entry.contract.component == component
    )
    format_name = VISION_FORMAT if component == "vision" else MAIN_FORMAT
    source_files = {
        "config.json": {"sha256": MAIN_CONFIG_SHA256, "nbytes": 5_109},
        "model.safetensors.index.json": {
            "sha256": MAIN_INDEX_SHA256,
            "nbytes": 132_674,
        },
        **{
            name: {"sha256": sha, "nbytes": nbytes}
            for name, (nbytes, sha) in MAIN_SHARDS.items()
        },
        "tokenizer.json": {
            "sha256": MUSE_GLIMMER_TOKENIZER_SHA256,
            "nbytes": 28_129_897,
        },
        "tokenizer_config.json": {
            "sha256": MUSE_GLIMMER_TOKENIZER_CONFIG_SHA256,
            "nbytes": 79_936,
        },
        "chat_template.jinja": {
            "sha256": MUSE_GLIMMER_ATEM_TEMPLATE_SHA256,
            "nbytes": 9_992,
        },
        "processor_config.json": {
            "sha256": MAIN_PROCESSOR_CONFIG_SHA256,
            "nbytes": 1_084,
        },
        "generation_config.json": {
            "sha256": MAIN_GENERATION_CONFIG_SHA256,
            "nbytes": 202,
        },
    }
    converted = _publish_converted_checkpoint(
        bundle,
        Path(output_root).expanduser().resolve(),
        entries=entries,
        format_name=format_name,
        component=component,
        source_revision=MUSE_GLIMMER_ARTIFACT_REVISION,
        source_files=source_files,
        compatible_target_sha256=None,
    )
    if component in {"text", "full"}:
        return publish_muse_glimmer_execution_bundle(root, converted)
    return converted


def convert_official_muse_glimmer_assistant_safetensors(
    source_root: str | Path,
    output_root: str | Path,
    *,
    target_checkpoint_sha256: str,
) -> ConvertedMuseGlimmerCheckpoint:
    if not _valid_sha256(target_checkpoint_sha256):
        raise MuseGlimmerCheckpointError(
            "Assistant conversion requires a lowercase target checkpoint SHA-256"
        )
    root = Path(source_root).expanduser().resolve()
    bundle = _inspect_official_assistant(root)
    shard_name, shard_nbytes, shard_sha = ASSISTANT_SHARD
    return _publish_converted_checkpoint(
        bundle,
        Path(output_root).expanduser().resolve(),
        entries=bundle.entries,
        format_name=ASSISTANT_FORMAT,
        component="assistant",
        source_revision=ASSISTANT_REVISION,
        source_files={
            "config.json": {"sha256": ASSISTANT_CONFIG_SHA256, "nbytes": 883},
            shard_name: {"sha256": shard_sha, "nbytes": shard_nbytes},
        },
        compatible_target_sha256=target_checkpoint_sha256,
    )


__all__ = [
    "ASSISTANT_FORMAT",
    "ASSISTANT_PARAMETER_COUNT",
    "ASSISTANT_PAYLOAD_BYTES",
    "ASSISTANT_REVISION",
    "ConvertedMuseGlimmerCheckpoint",
    "MAIN_FORMAT",
    "MAIN_PARAMETER_COUNT",
    "MAIN_PAYLOAD_BYTES",
    "MAIN_TEXT_PAYLOAD_BYTES",
    "NATIVE_LORA_FORMAT",
    "NATIVE_TRAIN_TOPOLOGY_SHA256",
    "MuseGlimmerCheckpointError",
    "MuseGlimmerTensorContract",
    "SafeTensorBundle",
    "SafeTensorEntry",
    "VISION_FORMAT",
    "attach_converted_muse_glimmer_assistant",
    "attach_native_muse_glimmer_lora",
    "build_muse_glimmer_execution_manifest_payload",
    "convert_official_muse_glimmer_assistant_safetensors",
    "convert_official_muse_glimmer_safetensors",
    "inspect_official_muse_glimmer_safetensors",
    "inspect_native_muse_glimmer_lora_checkpoint",
    "publish_muse_glimmer_execution_bundle",
    "inspect_safetensors_bundle",
    "inspect_safetensors_index",
    "muse_glimmer_assistant_tensor_contracts",
    "muse_glimmer_main_tensor_contracts",
]
