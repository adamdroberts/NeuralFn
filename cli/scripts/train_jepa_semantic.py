from __future__ import annotations

import argparse
import base64
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import tempfile
import time
import uuid
from typing import Any, Callable

from native_training_guard import reject_torch_training_by_default

if __name__ == "__main__":
    reject_torch_training_by_default("train_jepa_semantic.py", native_target="nfn train --base-model jepa", model_family="jepa")

import numpy as np
import torch

from cli_utils import artifact_path, create_argument_parser
import neuralfn.semantic as semantic_module
from neuralfn import TorchTrainConfig, TorchTrainer, save_graph
from neuralfn.config import (
    build_jepa_semantic_hybrid_megakernel_spec,
    build_jepa_semantic_hybrid_spec,
    model_spec_to_dict,
)
from neuralfn.inference import export_to_pt
from neuralfn.torch_backend import CompiledTorchGraph, resolve_torch_train_drop_last
from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config
import server.dataset_manager as dataset_manager_module
from server.dataset_manager import (
    DATASETS_DIR,
    DatasetTokenizerMismatchError,
    MemmapTokenDataset,
    _download_hf_file,
    _load_tokens_for,
    _shard_header_offset_uint16,
    download_hf_dataset,
    encode_raw_text,
    estimate_dataset_sequence_count,
    is_sentencepiece_tokenizer_name,
    load_dataset_tensors,
    normalize_raw_text_encoding_name,
    raw_text_encoding_name_for_backbone,
    raw_text_encoding_name_for_template_spec,
    raw_text_encoding_vocab_size,
    refresh_raw_text_dataset_metadata,
    resolve_sentencepiece_model_path,
    shared_sentencepiece_artifact_filenames,
    shared_sentencepiece_model_path,
    shared_sentencepiece_vocab_path,
    validate_cached_tokenizer_contract,
)

PARAMETER_GOLF_HF_PATH = "willdepueoai/parameter-golf"
DEFAULT_TOKENIZER_HF_PATH = "sproos/parameter-golf-tokenizers"
DEFAULT_TOKENIZER_REPO_TYPE = "model"
DEFAULT_TOKENIZER_REMOTE_ROOT_PREFIX = "tokenizers"
DEFAULT_CACHED_TOKENIZER_VARIANT = "sp1024"
SUPPORTED_CACHED_TOKENIZER_VARIANTS = (
    "sp1024",
    "sp2048",
    "sp4096",
    "sp8192",
)
SUPPORTED_TOKENIZER_CHOICES = (
    "gpt2",
    "cl100k_base",
    "o200k_base",
    *SUPPORTED_CACHED_TOKENIZER_VARIANTS,
)
DEFAULT_DATASET_ALIAS = f"willdepueoai__parameter-golf__{DEFAULT_CACHED_TOKENIZER_VARIANT}__train1"
TINYSTORIES_HF_PATH = "roneneldan/TinyStories"
TINYSTORIES_ALIAS = "roneneldan__TinyStories__TinyStoriesV2-GPT4"
TINYSTORIES_TRAIN_FILE = "TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_VAL_FILE = "TinyStoriesV2-GPT4-valid.txt"
CACHED_TOKEN_DATASETS = frozenset({"golf1", "golf10"})
VAL_HOLDOUT_FRACTION = 0.2
PRETRAINING_FILE_CONFLICT_FLAGS = (
    "--tinystories",
    "--dataset",
    "--dataset-alias",
    "--dataset-hf-path",
    "--dataset-variant",
    "--dataset-train-shards",
    "--dataset-repo-id",
    "--dataset-remote-root-prefix",
    "--dataset-train-file",
    "--dataset-val-file",
)
DATASET_DEFAULT_TOKENIZERS = {
    "golf1": DEFAULT_CACHED_TOKENIZER_VARIANT,
    "golf10": DEFAULT_CACHED_TOKENIZER_VARIANT,
    "shakespear": "cl100k_base",
    "shakespeare": "cl100k_base",
    "tinystories": "o200k_base",
}
RAW_TEXT_ENCODING_NAME_FIELD = "raw_text_encoding_name"
SANITIZED_COMPOSED_OBJECTIVES = frozenset({"ar", "ar_jepa", "jepa_semantic", "semantic_router", "semantic_router_jepa"})
JEPA_OBJECTIVES = frozenset({"ar_jepa", "jepa_semantic", "semantic_router_jepa"})
SEMANTIC_OBJECTIVES = frozenset({"jepa_semantic", "semantic_router", "semantic_router_jepa"})
AR_LOSS_OBJECTIVES = frozenset({"ar_jepa", "jepa_semantic", "semantic_router", "semantic_router_jepa"})
JEPA_ONLY_MODEL_SPEC_FIELDS = frozenset(
    {
        "jepa_latent_dim",
        "jepa_mask_ratio",
        "jepa_mask_strategy",
        "jepa_num_blocks",
        "jepa_min_block_ratio",
        "jepa_max_block_ratio",
        "ema_decay",
        "jepa_loss_coef",
    }
)
SEMANTIC_ONLY_MODEL_SPEC_FIELDS = frozenset(
    {
        "semantic_dim",
        "semantic_residual_dim",
        "semantic_n_lsh_tables",
        "semantic_n_lsh_planes",
        "semantic_table_path",
        "semantic_vocab_ref",
        "experimental_semantic_router_vecs",
        "semantic_align_loss_coef",
    }
)
UNUSED_COMPOSED_MODEL_SPEC_FIELDS = frozenset({"max_recurrence_steps", "halt_epsilon"})
SUPPORTED_HF_REPO_TYPES = ("dataset", "model", "space")


def _dataset_shortcut_contract(**kwargs: object) -> dict[str, object]:
    return dict(kwargs)


def default_tokenizer_for_dataset(dataset_name: str | None) -> str | None:
    normalized = str(dataset_name or "").strip().lower()
    if not normalized:
        return None
    return DATASET_DEFAULT_TOKENIZERS.get(normalized)


def raw_text_tokenizer_is_available(encoding_name: str | None) -> bool:
    normalized = normalize_raw_text_encoding_name(encoding_name)
    if normalized is None:
        return False
    if not is_sentencepiece_tokenizer_name(normalized):
        return True
    return shared_sentencepiece_model_path(normalized) is not None


def _shared_sentencepiece_tokenizer_artifacts(
    encoding_name: str,
    *,
    remote_root_prefix: str = DEFAULT_TOKENIZER_REMOTE_ROOT_PREFIX,
) -> dict[str, tuple[str, ...]]:
    normalized = normalize_raw_text_encoding_name(encoding_name)
    if normalized not in SUPPORTED_CACHED_TOKENIZER_VARIANTS:
        raise ValueError(f"Unsupported sentencepiece tokenizer {encoding_name!r}.")
    filenames = shared_sentencepiece_artifact_filenames(normalized)
    if not filenames.get("model") or not filenames.get("vocab"):
        raise FileNotFoundError(
            f"Raw-text tokenizer {normalized!r} does not have shared download paths configured."
        )
    remote_root = str(remote_root_prefix or "").strip().strip("/")
    return {
        "model": tuple(f"{remote_root}/{name}" if remote_root else name for name in filenames["model"]),
        "vocab": tuple(f"{remote_root}/{name}" if remote_root else name for name in filenames["vocab"]),
    }


def normalize_hf_repo_type(repo_type: str | None, *, default: str | None = None) -> str | None:
    normalized = str(repo_type or "").strip().lower()
    if not normalized:
        return default
    if normalized not in SUPPORTED_HF_REPO_TYPES:
        allowed = ", ".join(SUPPORTED_HF_REPO_TYPES)
        raise ValueError(f"Unsupported Hugging Face repo type {repo_type!r}. Expected one of: {allowed}.")
    return normalized


def resolve_tokenizer_download_contract(
    *,
    tokenizer_hf_path: str | None = None,
    tokenizer_repo_id: str | None = None,
    tokenizer_remote_root_prefix: str | None = None,
    tokenizer_repo_type: str | None = None,
) -> dict[str, str]:
    hf_path = str(tokenizer_hf_path or DEFAULT_TOKENIZER_HF_PATH).strip()
    repo_id = str(tokenizer_repo_id or hf_path).strip()
    remote_root_prefix = str(tokenizer_remote_root_prefix or DEFAULT_TOKENIZER_REMOTE_ROOT_PREFIX).strip().strip("/")
    repo_type = str(normalize_hf_repo_type(tokenizer_repo_type, default=DEFAULT_TOKENIZER_REPO_TYPE))
    return {
        "hf_path": hf_path,
        "repo_id": repo_id,
        "remote_root_prefix": remote_root_prefix,
        "repo_type": repo_type,
    }


def _copy_into_shared_tokenizer_cache(source_path: Path, destination_name: str) -> bool:
    tokenizers_dir = dataset_manager_module.SENTENCEPIECE_TOKENIZERS_DIR
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    destination = tokenizers_dir / destination_name
    if destination.exists():
        return False
    source = source_path.resolve(strict=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return True


def promote_dataset_tokenizer_to_shared_cache(
    encoding_name: str | None,
    *,
    dataset_alias: str | None = None,
    dataset_name: str | None = None,
    dataset_path: Path | None = None,
    dataset_meta: dict[str, object] | None = None,
) -> bool:
    normalized = normalize_raw_text_encoding_name(encoding_name)
    if not is_sentencepiece_tokenizer_name(normalized):
        return False

    resolved_name = dataset_name
    resolved_path = dataset_path
    if resolved_path is None:
        dataset_ref = dataset_alias or dataset_name
        if dataset_ref:
            try:
                resolved_name, resolved_path, _loaded_meta = resolve_existing_dataset(dataset_ref)
            except FileNotFoundError:
                return False
    if resolved_path is None or not resolved_path.is_dir():
        return False

    tokenizer_dir = resolved_path / "tokenizers"
    if not tokenizer_dir.exists():
        return False

    filenames = shared_sentencepiece_artifact_filenames(normalized)
    copied_any = False
    for artifact_kind in ("model", "vocab"):
        for filename in filenames[artifact_kind]:
            source_path = tokenizer_dir / filename
            if not source_path.exists():
                continue
            copied_any = _copy_into_shared_tokenizer_cache(source_path, source_path.name) or copied_any
            break

    if copied_any:
        log_stage(
            f"Promoted tokenizer assets for {normalized} from {resolved_name or resolved_path} into "
            f"{dataset_manager_module.SENTENCEPIECE_TOKENIZERS_DIR}"
        )
    return shared_sentencepiece_model_path(normalized) is not None and shared_sentencepiece_vocab_path(normalized) is not None


def ensure_raw_text_tokenizer_assets(
    encoding_name: str | None,
    *,
    download_if_missing: bool = False,
    dataset_alias: str | None = None,
    dataset_name: str | None = None,
    dataset_path: Path | None = None,
    dataset_meta: dict[str, object] | None = None,
    tokenizer_hf_path: str | None = None,
    tokenizer_repo_id: str | None = None,
    tokenizer_remote_root_prefix: str | None = None,
    tokenizer_repo_type: str | None = None,
) -> str:
    normalized = normalize_raw_text_encoding_name(encoding_name)
    if normalized is None:
        raise ValueError(f"Unsupported raw-text tokenizer {encoding_name!r}.")
    if not is_sentencepiece_tokenizer_name(normalized):
        return normalized

    model_path = shared_sentencepiece_model_path(normalized)
    vocab_path = shared_sentencepiece_vocab_path(normalized)
    if model_path is not None and vocab_path is not None:
        return normalized

    promote_dataset_tokenizer_to_shared_cache(
        normalized,
        dataset_alias=dataset_alias,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        dataset_meta=dataset_meta,
    )
    model_path = shared_sentencepiece_model_path(normalized)
    vocab_path = shared_sentencepiece_vocab_path(normalized)
    if model_path is not None and vocab_path is not None:
        return normalized

    if not download_if_missing:
        resolve_sentencepiece_model_path(normalized)
        return normalized

    contract = resolve_tokenizer_download_contract(
        tokenizer_hf_path=tokenizer_hf_path,
        tokenizer_repo_id=tokenizer_repo_id,
        tokenizer_remote_root_prefix=tokenizer_remote_root_prefix,
        tokenizer_repo_type=tokenizer_repo_type,
    )
    tokenizers_dir = dataset_manager_module.SENTENCEPIECE_TOKENIZERS_DIR
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    log_stage(
        f"Downloading shared sentencepiece tokenizer assets for {normalized} from "
        f"{contract['repo_id']} ({contract['repo_type']})"
    )
    remote_paths = _shared_sentencepiece_tokenizer_artifacts(
        normalized,
        remote_root_prefix=contract["remote_root_prefix"],
    )
    for artifact_kind in ("model", "vocab"):
        resolved = False
        last_error: Exception | None = None
        for remote_path in remote_paths[artifact_kind]:
            target_path = tokenizers_dir / Path(remote_path).name
            if target_path.exists():
                resolved = True
                break
            try:
                _download_hf_file(
                    contract["repo_id"],
                    remote_path,
                    target_path,
                    repo_type=contract["repo_type"],
                )
                resolved = True
                break
            except Exception as exc:
                last_error = exc
                try:
                    target_path.unlink()
                except FileNotFoundError:
                    pass
        if not resolved:
            expected = ", ".join(remote_paths[artifact_kind])
            raise FileNotFoundError(
                f"Raw-text tokenizer {normalized!r} could not download its shared sentencepiece {artifact_kind} "
                f"into {tokenizers_dir}. Tried: {expected}."
            ) from last_error
    resolve_sentencepiece_model_path(normalized)
    return normalized


def validate_raw_text_tokenizer_availability(
    encoding_name: str | None,
    *,
    download_if_missing: bool = False,
    dataset_alias: str | None = None,
    dataset_name: str | None = None,
    dataset_path: Path | None = None,
    dataset_meta: dict[str, object] | None = None,
    tokenizer_hf_path: str | None = None,
    tokenizer_repo_id: str | None = None,
    tokenizer_remote_root_prefix: str | None = None,
    tokenizer_repo_type: str | None = None,
) -> str:
    return ensure_raw_text_tokenizer_assets(
        encoding_name,
        download_if_missing=download_if_missing,
        dataset_alias=dataset_alias,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        dataset_meta=dataset_meta,
        tokenizer_hf_path=tokenizer_hf_path,
        tokenizer_repo_id=tokenizer_repo_id,
        tokenizer_remote_root_prefix=tokenizer_remote_root_prefix,
        tokenizer_repo_type=tokenizer_repo_type,
    )


def serialized_model_objective(model_spec: dict[str, Any] | None) -> str:
    template = dict((model_spec or {}).get("template", {}) or {})
    return str(template.get("objective", "") or "").strip().lower()


def sanitize_serialized_model_spec(model_spec: dict[str, Any] | None) -> dict[str, Any]:
    sanitized = dict(model_spec or {})
    resolved_encoding_name = normalize_raw_text_encoding_name(sanitized.get(RAW_TEXT_ENCODING_NAME_FIELD))
    if resolved_encoding_name is None:
        sanitized.pop(RAW_TEXT_ENCODING_NAME_FIELD, None)
    else:
        sanitized[RAW_TEXT_ENCODING_NAME_FIELD] = resolved_encoding_name
    objective = serialized_model_objective(sanitized)
    if objective not in SANITIZED_COMPOSED_OBJECTIVES:
        return sanitized

    for key in UNUSED_COMPOSED_MODEL_SPEC_FIELDS:
        sanitized.pop(key, None)
    if objective not in JEPA_OBJECTIVES:
        for key in JEPA_ONLY_MODEL_SPEC_FIELDS:
            sanitized.pop(key, None)
    if objective not in SEMANTIC_OBJECTIVES:
        for key in SEMANTIC_ONLY_MODEL_SPEC_FIELDS:
            sanitized.pop(key, None)
    if objective not in AR_LOSS_OBJECTIVES:
        sanitized.pop("ar_loss_coef", None)
    return sanitized


def sanitized_model_spec_dict(spec: Any, *, raw_text_encoding_name: str | None = None) -> dict[str, Any]:
    if isinstance(spec, dict):
        serialized = dict(spec)
    else:
        serialized = model_spec_to_dict(spec)
    resolved_encoding_name = normalize_raw_text_encoding_name(raw_text_encoding_name)
    if resolved_encoding_name is not None:
        serialized[RAW_TEXT_ENCODING_NAME_FIELD] = resolved_encoding_name
    return sanitize_serialized_model_spec(serialized)


def apply_sanitized_template_spec(graph, *, raw_text_encoding_name: str | None = None) -> dict[str, Any]:
    torch_config = dict(graph.torch_config or {})
    if "template_spec" not in torch_config:
        return {}
    template_spec = dict(torch_config.get("template_spec", {}) or {})
    resolved_encoding_name = normalize_raw_text_encoding_name(raw_text_encoding_name)
    if resolved_encoding_name is not None:
        template_spec[RAW_TEXT_ENCODING_NAME_FIELD] = resolved_encoding_name
    sanitized = sanitize_serialized_model_spec(template_spec)
    graph.torch_config = {
        **torch_config,
        "template_spec": sanitized,
    }
    return sanitized


def normalize_tokenizer_selection_args(args: argparse.Namespace) -> argparse.Namespace:
    dataset_choice = str(getattr(args, "dataset", "") or "").strip().lower()
    resolved_fields: dict[str, str] = {}
    for field_name in ("tokenizer", "raw_text_encoding_override"):
        raw_value = getattr(args, field_name, None)
        normalized = normalize_raw_text_encoding_name(raw_value)
        if normalized is not None:
            resolved_fields[field_name] = normalized

    raw_variant = getattr(args, "dataset_variant", None)
    if raw_variant not in (None, ""):
        normalized_variant = str(raw_variant).strip().lower()
        if normalized_variant in SUPPORTED_CACHED_TOKENIZER_VARIANTS:
            resolved_fields["dataset_variant"] = normalize_cached_tokenizer_variant(normalized_variant)
        else:
            resolved_fields["dataset_variant"] = normalize_raw_text_encoding_name(normalized_variant) or normalized_variant

    unique_values = {value for value in resolved_fields.values() if value}
    if len(unique_values) > 1:
        joined = ", ".join(f"{field}={value}" for field, value in sorted(resolved_fields.items()))
        raise ValueError(f"Conflicting tokenizer selections: {joined}")

    chosen = next(iter(unique_values), None)
    if dataset_choice in CACHED_TOKEN_DATASETS and chosen is not None and chosen not in SUPPORTED_CACHED_TOKENIZER_VARIANTS:
        allowed = ", ".join(SUPPORTED_CACHED_TOKENIZER_VARIANTS)
        raise ValueError(
            f"Cached-token dataset shortcut {dataset_choice!r} requires a sentencepiece tokenizer variant ({allowed}), "
            f"but received {chosen!r}."
        )

    args.tokenizer = chosen
    args.raw_text_encoding_override = chosen
    if dataset_choice in CACHED_TOKEN_DATASETS:
        if chosen is not None:
            args.dataset_variant = chosen
    elif chosen is not None:
        args.dataset_variant = None
    return args


def effective_tokenizer_name_for_args(args: argparse.Namespace) -> str | None:
    tokenizer_name = normalize_raw_text_encoding_name(
        getattr(args, "tokenizer", None)
        or getattr(args, "raw_text_encoding_override", None)
        or getattr(args, "dataset_variant", None)
        or getattr(args, "raw_text_encoding_name", None)
    )
    if tokenizer_name is not None:
        return tokenizer_name
    return default_tokenizer_for_dataset(getattr(args, "dataset", None))


def normalize_cached_tokenizer_variant(dataset_variant: str | None) -> str:
    normalized = str(dataset_variant or DEFAULT_CACHED_TOKENIZER_VARIANT).strip().lower()
    if normalized not in SUPPORTED_CACHED_TOKENIZER_VARIANTS:
        allowed = ", ".join(SUPPORTED_CACHED_TOKENIZER_VARIANTS)
        raise ValueError(
            f"Unsupported cached tokenizer variant {dataset_variant!r}. "
            f"Expected one of: {allowed}."
        )
    return normalized


def parameter_golf_dataset_alias(*, train_shards: int, variant: str | None = None) -> str:
    resolved_variant = normalize_cached_tokenizer_variant(variant)
    return f"willdepueoai__parameter-golf__{resolved_variant}__train{int(train_shards)}"


def parameter_golf_dataset_contract(*, train_shards: int, variant: str | None = None) -> dict[str, object]:
    resolved_variant = normalize_cached_tokenizer_variant(variant)
    return _dataset_shortcut_contract(
        dataset_alias=parameter_golf_dataset_alias(train_shards=train_shards, variant=resolved_variant),
        dataset_hf_path=PARAMETER_GOLF_HF_PATH,
        dataset_variant=resolved_variant,
        dataset_train_shards=int(train_shards),
    )


TINYSTORIES_DATASET_CONTRACT = _dataset_shortcut_contract(
    dataset_alias=TINYSTORIES_ALIAS,
    dataset_hf_path=TINYSTORIES_HF_PATH,
    dataset_train_file=TINYSTORIES_TRAIN_FILE,
    dataset_val_file=TINYSTORIES_VAL_FILE,
)

DATASET_SHORTCUT_CONTRACTS = {
    "golf1": parameter_golf_dataset_contract(train_shards=1),
    "golf10": parameter_golf_dataset_contract(train_shards=10),
    "shakespear": _dataset_shortcut_contract(dataset_alias="karpathy__tiny_shakespeare"),
    "shakespeare": _dataset_shortcut_contract(dataset_alias="karpathy__tiny_shakespeare"),
    "tinystories": TINYSTORIES_DATASET_CONTRACT,
}
DEFAULT_ARTIFACT = artifact_path("jepa_semantic_hybrid.pt")
DEFAULT_GRAPH_ARTIFACT = DEFAULT_ARTIFACT.with_suffix(".json")
INTERRUPTED_ARTIFACT = DEFAULT_ARTIFACT.with_name("jepa_semantic_hybrid.interrupted.pt")
INTERRUPTED_GRAPH_ARTIFACT = INTERRUPTED_ARTIFACT.with_suffix(".json")

REFERENCE_DEFAULTS = {
    "iterations": 20_000,
    "warmdown_fraction": 0.75,
    "warmup_steps": 20,
    "train_batch_tokens": 524_288,
    "train_seq_len": 1_024,
    "max_wallclock_seconds": 0.0,
    "qk_gain_init": 1.5,
    "vocab_size": 1_024,
    "num_layers": 9,
    "num_kv_heads": 4,
    "model_dim": 512,
    "num_heads": 8,
    "mlp_mult": 2.0,
    "tie_embeddings": True,
    "rope_base": 10_000.0,
    "logit_softcap": 30.0,
    "embed_lr": 0.6,
    "head_lr": 0.008,
    "tied_embed_lr": 0.05,
    "tied_embed_init_std": 0.005,
    "matrix_lr": 0.04,
    "scalar_lr": 0.04,
    "muon_momentum": 0.95,
    "muon_backend_steps": 5,
    "muon_momentum_warmup_start": 0.85,
    "muon_momentum_warmup_steps": 500,
    "beta1": 0.9,
    "beta2": 0.95,
    "adam_eps": 1e-8,
    "grad_clip_norm": 0.0,
    "val_batch_size": 524_288,
    "val_loss_every": 1_000,
    "train_log_every": 1,
}

JEPA_DEFAULTS = {
    "seed": 1337,
    "device": "cuda",
    "run_id": str(uuid.uuid4()),
    "dataset_alias": DEFAULT_DATASET_ALIAS,
    "output": str(DEFAULT_ARTIFACT),
    "max_steps": 20_000,
    "train_seq_len": 1_024,
    "batch_size": 64,
    "train_batch_tokens": 524_288,
    "eval_batches": 20,
    "eval_batch_size": 64,
    "train_log_every": 1,
    "val_loss_every": 250,
    "max_wallclock_seconds": 0.0,
    "warmup_steps": 60,
    "warmdown_fraction": 0.0,
    "vocab_size": 1_024,
    "num_layers": 4,
    "model_dim": 256,
    "num_heads": 4,
    "num_kv_heads": 4,
    "mlp_mult": 2.0,
    "multiple_of": 64,
    "experts": semantic_module.NUM_VOCAB_DIMS,
    "top_k": 2,
    "ema_decay": 0.99,
    "rope_base": 10_000.0,
    "qk_gain_init": 1.5,
    "logit_softcap": 30.0,
    "optimizer_profile": "adamw",
    "learning_rate": 6e-4,
    "weight_decay": 0.1,
    "embed_lr": 0.02,
    "head_lr": 0.005,
    "tied_embed_lr": 0.01,
    "matrix_lr": 0.008,
    "scalar_lr": 0.004,
    "muon_momentum": 0.95,
    "muon_backend_steps": 5,
    "muon_momentum_warmup_start": 0.85,
    "muon_momentum_warmup_steps": 64,
    "beta1": 0.9,
    "beta2": 0.95,
    "adam_eps": 1e-8,
    "grad_clip_norm": 1.0,
    "ar_loss_coef": 1.0,
    "jepa_loss_coef": 0.25,
    "semantic_align_loss_coef": 0.5,
}
ALL_TRAIN_ROWS_DEFAULT_EPOCHS = 2
EVOLUTIONARY_DEFAULTS = {
    "population_size": 50,
    "mutation_rate": 0.1,
    "mutation_scale": 0.3,
    "crossover_rate": 0.5,
    "tournament_size": 3,
    "elite_count": 2,
}

LOGGER = logging.getLogger("jepa_semantic_harness")
_CACHED_VARIANT_ALIAS_RE = re.compile(
    r"^(?P<owner>[^_][^/]*)__"
    r"(?P<repo>[^_].*?)__"
    r"(?P<variant>[^_].*?)__train(?P<train_shards>\d+)$"
)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return int(value)


def env_optional_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return float(value)


class _ExplicitValueAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"_{self.dest}_explicit", True)


def configure_console_logging() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def log_stage(message: str) -> None:
    LOGGER.info(message)


def format_elapsed(seconds: float) -> str:
    total = max(int(seconds), 0)
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def mode_name(*, megakernel: bool) -> str:
    return "jepa_semantic_hybrid_megakernel" if megakernel else "jepa_semantic_hybrid"


def default_output_path(*, megakernel: bool) -> Path:
    if megakernel:
        return DEFAULT_ARTIFACT.with_name("jepa_semantic_hybrid_megakernel.pt")
    return DEFAULT_ARTIFACT


def interrupted_output_path(*, megakernel: bool) -> Path:
    if megakernel:
        return DEFAULT_ARTIFACT.with_name("jepa_semantic_hybrid_megakernel.interrupted.pt")
    return INTERRUPTED_ARTIFACT


def graph_name(*, megakernel: bool) -> str:
    return f"{mode_name(megakernel=megakernel)}_sdk"


def resolve_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    args._max_steps_explicit = bool(getattr(args, "_max_steps_explicit", False) or ("ITERATIONS" in os.environ))
    args._max_wallclock_seconds_explicit = bool(
        getattr(args, "_max_wallclock_seconds_explicit", False) or ("MAX_WALLCLOCK_SECONDS" in os.environ)
    )
    if not getattr(args, "output", ""):
        args.output = str(default_output_path(megakernel=bool(args.megakernel)))
    resolve_lr_schedule_defaults(args)
    apply_raw_text_tokenizer_policy(
        args,
        preset_name=mode_name(megakernel=bool(args.megakernel)),
        default_vocab_size=int(JEPA_DEFAULTS["vocab_size"]),
    )
    return args


def add_lr_schedule_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lr-decay-iters",
        type=int,
        default=env_optional_int("LR_DECAY_ITERS"),
        help="Cosine LR decay horizon in optimizer steps; when set, cosine decay overrides warmdown_fraction.",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=env_optional_float("MIN_LR"),
        help="Cosine LR floor; defaults to --learning-rate / 10 when --lr-decay-iters is set and --min-lr is unset.",
    )


def resolve_lr_schedule_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "lr_decay_iters", None) is not None and getattr(args, "min_lr", None) is None:
        args.min_lr = float(args.learning_rate) / 10.0
    return args


def add_warmdown_fraction_argument(parser: argparse.ArgumentParser, *, default: float) -> None:
    parser.add_argument(
        "--warmdown-fraction",
        type=float,
        default=default,
        help="Fraction of optimizer steps reserved for linear tail warmdown.",
    )


def add_all_train_rows_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--all-train-rows",
        action="store_true",
        help=(
            "Use every train row in each epoch, keep partial final batches, "
            "and round max_steps up to the next full epoch boundary."
        ),
    )


def add_evolutionary_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--evolutionary",
        action="store_true",
        help="Use evolutionary search over torch parameters instead of gradient descent.",
    )
    parser.add_argument(
        "--evo-population-size",
        type=int,
        default=env_int("EVO_POPULATION_SIZE", EVOLUTIONARY_DEFAULTS["population_size"]),
    )
    parser.add_argument(
        "--evo-mutation-rate",
        type=float,
        default=env_float("EVO_MUTATION_RATE", EVOLUTIONARY_DEFAULTS["mutation_rate"]),
    )
    parser.add_argument(
        "--evo-mutation-scale",
        type=float,
        default=env_float("EVO_MUTATION_SCALE", EVOLUTIONARY_DEFAULTS["mutation_scale"]),
    )
    parser.add_argument(
        "--evo-crossover-rate",
        type=float,
        default=env_float("EVO_CROSSOVER_RATE", EVOLUTIONARY_DEFAULTS["crossover_rate"]),
    )
    parser.add_argument(
        "--evo-tournament-size",
        type=int,
        default=env_int("EVO_TOURNAMENT_SIZE", EVOLUTIONARY_DEFAULTS["tournament_size"]),
    )
    parser.add_argument(
        "--evo-elite-count",
        type=int,
        default=env_int("EVO_ELITE_COUNT", EVOLUTIONARY_DEFAULTS["elite_count"]),
    )
    parser.add_argument(
        "--evo-seed",
        type=int,
        default=env_optional_int("EVO_SEED"),
        help="Optional RNG seed for evolutionary search; defaults to --seed.",
    )


def add_max_steps_argument(parser: argparse.ArgumentParser, *, default: int) -> None:
    parser.set_defaults(_max_steps_explicit=("ITERATIONS" in os.environ))
    parser.add_argument(
        "--max-steps",
        action=_ExplicitValueAction,
        type=int,
        default=default,
    )


def add_max_wallclock_seconds_argument(parser: argparse.ArgumentParser, *, default: float) -> None:
    parser.set_defaults(_max_wallclock_seconds_explicit=("MAX_WALLCLOCK_SECONDS" in os.environ))
    parser.add_argument(
        "--max-wallclock-seconds",
        action=_ExplicitValueAction,
        type=float,
        default=default,
    )


def _resolve_schedule_layout(
    *,
    source_train_rows: int,
    batch_size: int,
    grad_accum_steps: int,
    template_runtime: str,
    device: str,
    drop_last: bool | None,
    all_train_rows: bool,
) -> dict[str, int | bool]:
    if all_train_rows:
        effective_drop_last = False
    else:
        effective_drop_last = resolve_torch_train_drop_last(
            drop_last=drop_last,
            template_runtime=template_runtime,
            device=device,
            dataset_rows=source_train_rows,
            batch_size=batch_size,
        )
    if effective_drop_last:
        loader_batches = max(1, source_train_rows // batch_size)
        train_rows = loader_batches * batch_size
    else:
        loader_batches = max(1, math.ceil(source_train_rows / batch_size))
        train_rows = source_train_rows
    steps_per_epoch = max(1, math.ceil(loader_batches / grad_accum_steps))
    tail_loader_batches = loader_batches % grad_accum_steps
    tail_grad_accum_steps = tail_loader_batches or grad_accum_steps
    return {
        "train_rows": train_rows,
        "loader_batches": loader_batches,
        "steps_per_epoch": steps_per_epoch,
        "drop_last": effective_drop_last,
        "dropped_train_rows": max(source_train_rows - train_rows, 0),
        "respect_epoch_boundaries": bool(all_train_rows),
        "has_short_epoch_tail_step": bool(all_train_rows and tail_grad_accum_steps != grad_accum_steps),
        "epoch_tail_grad_accum_steps": tail_grad_accum_steps,
    }


def resolve_effective_training_schedule(
    args: argparse.Namespace,
    derived: dict[str, int | bool],
) -> tuple[dict[str, int | bool | None], int, int, int | None, float]:
    steps_per_epoch = max(int(derived["steps_per_epoch"]), 1)
    requested_max_steps = int(args.max_steps)
    resolved_max_steps = requested_max_steps
    if bool(getattr(args, "all_train_rows", False)):
        requested_for_rounding = max(requested_max_steps, 1)
        if not getattr(args, "_max_steps_explicit", False) and "ITERATIONS" not in os.environ:
            requested_for_rounding = max(
                requested_for_rounding,
                steps_per_epoch * ALL_TRAIN_ROWS_DEFAULT_EPOCHS,
            )
        resolved_max_steps = max(
            steps_per_epoch,
            math.ceil(requested_for_rounding / steps_per_epoch) * steps_per_epoch,
        )
    resolved_epochs = max(1, math.ceil(max(resolved_max_steps, 1) / steps_per_epoch))
    requested_lr_decay_iters = (
        None if getattr(args, "lr_decay_iters", None) is None else int(args.lr_decay_iters)
    )
    resolved_lr_decay_iters = requested_lr_decay_iters
    requested_max_wallclock_seconds = float(args.max_wallclock_seconds)
    resolved_max_wallclock_seconds = requested_max_wallclock_seconds
    if bool(getattr(args, "all_train_rows", False)) and not getattr(args, "_max_wallclock_seconds_explicit", False):
        resolved_max_wallclock_seconds = 0.0
    derived = {
        **derived,
        "all_train_rows": bool(getattr(args, "all_train_rows", False)),
        "requested_max_steps": requested_max_steps,
        "resolved_max_steps": resolved_max_steps,
        "requested_lr_decay_iters": requested_lr_decay_iters,
        "resolved_lr_decay_iters": resolved_lr_decay_iters,
        "requested_max_wallclock_seconds": requested_max_wallclock_seconds,
        "resolved_max_wallclock_seconds": resolved_max_wallclock_seconds,
        "resolved_epochs": resolved_epochs,
        "default_all_train_rows_epochs": ALL_TRAIN_ROWS_DEFAULT_EPOCHS,
    }
    return derived, resolved_epochs, resolved_max_steps, resolved_lr_decay_iters, resolved_max_wallclock_seconds


def format_routing_stats_suffix(
    routing_stats: dict[str, Any] | None,
    *,
    semantic_labels: bool,
    max_entries: int = 8,
) -> str:
    if not routing_stats:
        return ""

    num_experts = int(routing_stats.get("num_experts", 0))
    active_expert_count = int(routing_stats.get("active_expert_count", 0))
    weight_mass_shares = [
        float(value)
        for value in routing_stats.get("weight_mass_shares", [])
    ]
    nonzero_usage = [
        (idx, share)
        for idx, share in enumerate(weight_mass_shares)
        if share > 0.0
    ]
    nonzero_usage.sort(key=lambda item: (-item[1], item[0]))
    truncated = len(nonzero_usage) > max_entries
    preview_items = nonzero_usage[:max_entries]

    usage_preview: list[str] = []
    for expert_idx, share in preview_items:
        if semantic_labels:
            label = semantic_module.EXPERT_TO_DIMENSION.get(expert_idx, f"expert_{expert_idx}")
        else:
            label = str(expert_idx)
        usage_preview.append(f"{label}:{share * 100:.0f}%")
    if truncated:
        usage_preview.append("...")

    parts = [f"route=active {active_expert_count}/{num_experts}"]
    router_entropy = routing_stats.get("mean_router_entropy_norm")
    topk_entropy = routing_stats.get("mean_topk_entropy_norm")
    if isinstance(router_entropy, (int, float)) and math.isfinite(float(router_entropy)):
        parts.append(f"entropy={float(router_entropy):.2f}")
    if isinstance(topk_entropy, (int, float)) and math.isfinite(float(topk_entropy)):
        parts.append(f"topk_h={float(topk_entropy):.2f}")
    if usage_preview:
        parts.append(f"usage=[{','.join(usage_preview)}]")
    return " " + " ".join(parts)


def resolve_existing_dataset(alias: str) -> tuple[str, Path, dict[str, object]]:
    alias_path = Path(str(alias)).expanduser()
    if alias_path.is_absolute():
        if alias_path.is_dir():
            meta_path = alias_path / "meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"Dataset path {alias_path} exists but is missing meta.json."
                )
            meta: dict[str, object] = json.loads(meta_path.read_text(encoding="utf-8"))
            return str(alias_path), alias_path, meta
        if alias_path.is_file():
            return str(alias_path), alias_path, {}
        raise FileNotFoundError(f"Dataset path {alias_path} does not exist.")
    ds_dir = DATASETS_DIR / alias
    ds_file = DATASETS_DIR / f"{alias}.txt"
    if ds_dir.is_dir():
        meta_path = ds_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Dataset alias {alias!r} exists under {DATASETS_DIR} but is missing meta.json. "
                "Treating it as an incomplete cache and forcing a fresh download."
            )
        meta: dict[str, object] = json.loads(meta_path.read_text(encoding="utf-8"))
        return alias, ds_dir, meta
    if ds_file.exists():
        return alias, ds_file, {}
    raise FileNotFoundError(
        f"Dataset alias {alias!r} was not found under {DATASETS_DIR}. "
        "This harness reuses an existing cached dataset and will not redownload it."
    )


def parse_cached_variant_alias(alias: str) -> dict[str, object] | None:
    match = _CACHED_VARIANT_ALIAS_RE.fullmatch(alias)
    if match is None:
        return None
    hf_path = f"{match.group('owner')}/{match.group('repo')}"
    train_shards = int(match.group("train_shards"))
    return {
        "hf_path": hf_path,
        "variant": match.group("variant"),
        "train_shards": train_shards,
        "repo_id": hf_path,
        "remote_root_prefix": "datasets",
    }


def resolve_dataset_download_contract(
    alias: str,
    *,
    dataset_hf_path: str | None = None,
    dataset_variant: str | None = None,
    dataset_train_shards: int | None = None,
    dataset_repo_id: str | None = None,
    dataset_remote_root_prefix: str | None = None,
    dataset_train_file: str | None = None,
    dataset_val_file: str | None = None,
) -> dict[str, object]:
    parsed = parse_cached_variant_alias(alias) or {}
    contract: dict[str, object] = dict(parsed)
    overrides = {
        "hf_path": dataset_hf_path,
        "variant": dataset_variant,
        "train_shards": dataset_train_shards,
        "repo_id": dataset_repo_id,
        "remote_root_prefix": dataset_remote_root_prefix,
        "train_file": dataset_train_file,
        "val_file": dataset_val_file,
    }
    for key, value in overrides.items():
        if value is not None:
            contract[key] = value

    if contract.get("train_file"):
        missing_flags: list[str] = []
        if not contract.get("hf_path"):
            missing_flags.append("--dataset-hf-path")
        if not contract.get("train_file"):
            missing_flags.append("--dataset-train-file")
        if missing_flags:
            missing = ", ".join(missing_flags)
            raise ValueError(
                f"Dataset alias {alias!r} is missing locally under {DATASETS_DIR}, and its raw download contract "
                f"could not be derived automatically. Pass {missing} to enable auto-download for this alias."
            )
        contract.setdefault("repo_id", str(contract["hf_path"]))
        contract.setdefault("remote_root_prefix", "datasets")
        return contract

    missing_flags: list[str] = []
    if not contract.get("hf_path"):
        missing_flags.append("--dataset-hf-path")
    if not contract.get("variant"):
        missing_flags.append("--dataset-variant")
    if contract.get("train_shards") is None:
        missing_flags.append("--dataset-train-shards")
    if missing_flags:
        missing = ", ".join(missing_flags)
        raise ValueError(
            f"Dataset alias {alias!r} is missing locally under {DATASETS_DIR}, and its download contract "
            f"could not be derived automatically. Pass {missing} to enable auto-download for this alias."
        )

    contract.setdefault("repo_id", str(contract["hf_path"]))
    contract.setdefault("remote_root_prefix", "datasets")
    return contract


def _prepare_dataset_from_text(
    alias: str,
    contract: dict[str, object],
) -> tuple[str, Path, dict[str, object]]:
    """Fallback: tokenize raw text into headerless uint16 shards when binary download fails."""
    import shutil

    repo = str(contract["repo_id"])
    variant = str(contract["variant"])
    remote_root_prefix = str(contract["remote_root_prefix"])
    train_shards_requested = int(contract["train_shards"])

    ds_dir = DATASETS_DIR / alias
    ds_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Download manifest & tokenizer artifacts ---
    manifest_path = ds_dir / "manifest.json"
    if not manifest_path.exists():
        _download_hf_file(repo, f"{remote_root_prefix}/manifest.json", manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Find tokenizer entry for this variant
    dataset_dir_name = f"fineweb10B_{variant}"
    dataset_entry = next(
        (x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir_name), None
    )
    if dataset_entry is None:
        raise FileNotFoundError(f"Dataset {dataset_dir_name} not in manifest for {repo}")

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next(
        (x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None
    )
    if tokenizer_entry is None:
        raise FileNotFoundError(f"Tokenizer {tokenizer_name} not in manifest for {repo}")

    tokenizer_artifacts: list[str] = []
    for key in ("model_path", "vocab_path", "path"):
        value = tokenizer_entry.get(key)
        if value:
            tokenizer_artifacts.append(str(value))

    for artifact_path in tokenizer_artifacts:
        filename = Path(artifact_path).name
        _download_hf_file(repo, f"{remote_root_prefix}/{artifact_path}", ds_dir / "tokenizers" / filename)

    # --- 2. Load SentencePiece model ---
    model_candidates = sorted((ds_dir / "tokenizers").glob("*.model"))
    if not model_candidates:
        raise FileNotFoundError(f"No .model tokenizer file found in {ds_dir / 'tokenizers'}")

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_candidates[0]))
    vocab_size = sp.get_piece_size()
    log_stage(f"Text fallback: loaded tokenizer {model_candidates[0].name} (vocab={vocab_size})")

    # --- 3. Stream raw text ---
    docs_jsonl_path = ds_dir / "docs_selected.jsonl"
    text_lines: list[str] = []
    try:
        _download_hf_file(repo, f"{remote_root_prefix}/docs_selected.jsonl", docs_jsonl_path)
        with open(docs_jsonl_path, encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                text = doc.get("text") or doc.get("content") or ""
                if text:
                    text_lines.append(text)
        log_stage(f"Text fallback: read {len(text_lines)} docs from docs_selected.jsonl")
    except Exception:
        log_stage("Text fallback: docs_selected.jsonl unavailable, streaming from HuggingFaceFW/fineweb")
        from datasets import load_dataset as hf_load_dataset
        ds = hf_load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
        target_tokens = train_shards_requested * 100_000_000  # ~100M tokens per shard
        collected_tokens = 0
        for sample in ds:
            text = sample.get("text", "")
            if text:
                text_lines.append(text)
                collected_tokens += len(text) // 4  # rough char-to-token estimate
                if collected_tokens >= target_tokens:
                    break
        log_stage(f"Text fallback: streamed {len(text_lines)} docs from fineweb")

    if not text_lines:
        raise FileNotFoundError("Text fallback: no text data available to tokenize")

    # --- 4. Tokenize and write headerless uint16 shards ---
    all_tokens: list[int] = []
    for text in text_lines:
        all_tokens.extend(sp.encode(text, out_type=int))
    log_stage(f"Text fallback: tokenized {len(all_tokens)} tokens total")

    total_tokens = len(all_tokens)
    tokens_per_shard = total_tokens // train_shards_requested if train_shards_requested > 0 else total_tokens
    # Reserve ~0.1% for validation
    val_token_count = max(tokens_per_shard // 10, min(total_tokens // 100, 1_000_000))
    val_tokens = all_tokens[:val_token_count]
    train_tokens = all_tokens[val_token_count:]

    # Write validation shard
    np.array(val_tokens, dtype=np.uint16).tofile(ds_dir / "fineweb_val_000000.bin")

    # Write training shards
    tokens_per_train_shard = len(train_tokens) // train_shards_requested if train_shards_requested > 0 else len(train_tokens)
    actual_train_shards = 0
    for i in range(train_shards_requested):
        start = i * tokens_per_train_shard
        end = (i + 1) * tokens_per_train_shard if i < train_shards_requested - 1 else len(train_tokens)
        if start >= len(train_tokens):
            break
        shard_arr = np.array(train_tokens[start:end], dtype=np.uint16)
        shard_arr.tofile(ds_dir / f"fineweb_train_{i:06d}.bin")
        actual_train_shards += 1

    # --- 5. Write meta.json ---
    meta: dict[str, object] = {
        "source": "text_fallback_tokenized",
        "hf_path": str(contract["hf_path"]),
        "hf_split": "train",
        "text_column": "tokens",
        "num_rows": actual_train_shards,
        "num_tokens": len(train_tokens),
        "variant": variant,
        "train_shards": actual_train_shards,
        "val_shards": 1,
        "repo_id": repo,
        "remote_root_prefix": remote_root_prefix,
        "tokenizer_name": tokenizer_name,
        "tokenizer_files": [Path(p).name for p in tokenizer_artifacts],
        "data_format": "uint16_shards",
    }

    # --- 6. Validate ---
    try:
        validate_cached_tokenizer_contract(alias, dataset_path=ds_dir, dataset_meta=meta)
    except Exception:
        shutil.rmtree(ds_dir, ignore_errors=True)
        raise

    (ds_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log_stage(f"Text fallback: wrote {actual_train_shards} train shards + 1 val shard to {ds_dir}")
    return alias, ds_dir, meta


def resolve_or_download_dataset(
    alias: str,
    *,
    download_if_missing: bool = True,
    dataset_hf_path: str | None = None,
    dataset_variant: str | None = None,
    dataset_train_shards: int | None = None,
    dataset_repo_id: str | None = None,
    dataset_remote_root_prefix: str | None = None,
    dataset_train_file: str | None = None,
    dataset_val_file: str | None = None,
    raw_text_encoding_name: str = "gpt2",
    tokenizer_hf_path: str | None = None,
    tokenizer_repo_id: str | None = None,
    tokenizer_remote_root_prefix: str | None = None,
    tokenizer_repo_type: str | None = None,
) -> tuple[str, Path, dict[str, object]]:
    try:
        dataset_name, dataset_path, dataset_meta = resolve_existing_dataset(alias)
    except FileNotFoundError:
        if not download_if_missing:
            raise FileNotFoundError(
                f"Dataset alias {alias!r} was not found under {DATASETS_DIR}. "
                "Auto-download is disabled; rerun with --download-if-missing or provide a cached alias."
            ) from None
    else:
        if is_sentencepiece_tokenizer_name(raw_text_encoding_name):
            promote_dataset_tokenizer_to_shared_cache(
                raw_text_encoding_name,
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                dataset_meta=dataset_meta,
            )
        if dataset_path.is_dir() and dataset_meta.get("data_format") != "uint16_shards":
            validate_raw_text_tokenizer_availability(
                raw_text_encoding_name,
                download_if_missing=download_if_missing,
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                dataset_meta=dataset_meta,
                tokenizer_hf_path=tokenizer_hf_path,
                tokenizer_repo_id=tokenizer_repo_id,
                tokenizer_remote_root_prefix=tokenizer_remote_root_prefix,
                tokenizer_repo_type=tokenizer_repo_type,
            )
            dataset_meta = refresh_raw_text_dataset_metadata(
                dataset_name,
                dataset_path=dataset_path,
                dataset_meta=dataset_meta,
                encoding_name=raw_text_encoding_name,
            )
        return dataset_name, dataset_path, dataset_meta

    contract = resolve_dataset_download_contract(
        alias,
        dataset_hf_path=dataset_hf_path,
        dataset_variant=dataset_variant,
        dataset_train_shards=dataset_train_shards,
        dataset_repo_id=dataset_repo_id,
        dataset_remote_root_prefix=dataset_remote_root_prefix,
        dataset_train_file=dataset_train_file,
        dataset_val_file=dataset_val_file,
    )
    if contract.get("train_file"):
        if is_sentencepiece_tokenizer_name(raw_text_encoding_name):
            validate_raw_text_tokenizer_availability(
                raw_text_encoding_name,
                download_if_missing=download_if_missing,
                tokenizer_hf_path=tokenizer_hf_path,
                tokenizer_repo_id=tokenizer_repo_id,
                tokenizer_remote_root_prefix=tokenizer_remote_root_prefix,
                tokenizer_repo_type=tokenizer_repo_type,
            )
        download_hf_dataset(
            str(contract["hf_path"]),
            alias=alias,
            repo_id=str(contract["repo_id"]),
            remote_root_prefix=str(contract["remote_root_prefix"]),
            train_file=str(contract["train_file"]),
            val_file=str(contract["val_file"]) if contract.get("val_file") is not None else None,
            encoding_name=raw_text_encoding_name,
        )
        return resolve_or_download_dataset(
            alias,
            download_if_missing=False,
            raw_text_encoding_name=raw_text_encoding_name,
            tokenizer_hf_path=tokenizer_hf_path,
            tokenizer_repo_id=tokenizer_repo_id,
            tokenizer_remote_root_prefix=tokenizer_remote_root_prefix,
            tokenizer_repo_type=tokenizer_repo_type,
        )
    try:
        download_hf_dataset(
            str(contract["hf_path"]),
            alias=alias,
            variant=str(contract["variant"]),
            train_shards=int(contract["train_shards"]),
            repo_id=str(contract["repo_id"]),
            remote_root_prefix=str(contract["remote_root_prefix"]),
            encoding_name=raw_text_encoding_name,
        )
    except DatasetTokenizerMismatchError:
        LOGGER.warning(
            "Binary shard download failed tokenizer validation; falling back to text-based preparation."
        )
        return _prepare_dataset_from_text(alias, contract)
    return resolve_or_download_dataset(
        alias,
        download_if_missing=False,
        raw_text_encoding_name=raw_text_encoding_name,
        tokenizer_hf_path=tokenizer_hf_path,
        tokenizer_repo_id=tokenizer_repo_id,
        tokenizer_remote_root_prefix=tokenizer_remote_root_prefix,
        tokenizer_repo_type=tokenizer_repo_type,
    )


def add_dataset_download_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--download-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the dataset alias automatically if it is missing from ~/.cache/nfn/datasets/.",
    )
    parser.add_argument("--dataset-hf-path", default=None)
    parser.add_argument("--dataset-variant", default=None)
    parser.add_argument("--dataset-train-shards", type=int, default=None)
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--dataset-remote-root-prefix", default=None)
    parser.add_argument("--dataset-train-file", default=None)
    parser.add_argument("--dataset-val-file", default=None)


def add_pretraining_file_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str = "Local .txt corpus to train on directly.",
) -> None:
    parser.add_argument("--pretraining-file", default=None, help=help_text)


def add_dataset_selector_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_alias: str,
) -> None:
    parser.add_argument(
        "--tinystories",
        action="store_true",
        help=(
            "Use the TinyStoriesV2 GPT-4 train/valid text files from "
            "roneneldan/TinyStories."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_SHORTCUT_CONTRACTS.keys()),
        default=None,
        help=(
            "Shortcut dataset selector. "
            "Use golf1, golf10, tinystories, shakespear, or shakespeare instead "
            "of the full cached alias."
        ),
    )
    parser.add_argument(
        "--dataset-alias",
        default=default_alias,
        help="Full cached dataset alias. Ignored when --dataset is provided.",
    )


def apply_tinystories_dataset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not getattr(args, "tinystories", False):
        return args

    conflicts: list[str] = []
    if getattr(args, "dataset", None):
        conflicts.append("--dataset")
    if str(getattr(args, "dataset_alias", DEFAULT_DATASET_ALIAS)) != DEFAULT_DATASET_ALIAS:
        conflicts.append("--dataset-alias")
    if getattr(args, "dataset_hf_path", None):
        conflicts.append("--dataset-hf-path")
    if getattr(args, "dataset_variant", None):
        conflicts.append("--dataset-variant")
    if getattr(args, "dataset_train_shards", None) is not None:
        conflicts.append("--dataset-train-shards")
    if getattr(args, "dataset_repo_id", None):
        conflicts.append("--dataset-repo-id")
    if getattr(args, "dataset_remote_root_prefix", None):
        conflicts.append("--dataset-remote-root-prefix")
    if getattr(args, "dataset_train_file", None):
        conflicts.append("--dataset-train-file")
    if getattr(args, "dataset_val_file", None):
        conflicts.append("--dataset-val-file")
    if getattr(args, "pretraining_file", None):
        conflicts.append("--pretraining-file")
    if conflicts:
        joined = ", ".join(conflicts)
        raise ValueError(f"--tinystories cannot be combined with {joined}")

    return apply_dataset_selector_contract(args, TINYSTORIES_DATASET_CONTRACT)


def apply_dataset_selector_contract(
    args: argparse.Namespace,
    contract: dict[str, object],
) -> argparse.Namespace:
    for field_name, value in contract.items():
        setattr(args, field_name, value)
    return args


def resolve_dataset_shortcut_contract(
    shortcut_name: str,
    *,
    dataset_variant: str | None = None,
) -> dict[str, object] | None:
    if shortcut_name == "golf1":
        return parameter_golf_dataset_contract(train_shards=1, variant=dataset_variant)
    if shortcut_name == "golf10":
        return parameter_golf_dataset_contract(train_shards=10, variant=dataset_variant)
    contract = DATASET_SHORTCUT_CONTRACTS.get(shortcut_name)
    return None if contract is None else dict(contract)


def resolve_dataset_selector_args(args: argparse.Namespace) -> str:
    normalize_tokenizer_selection_args(args)
    shortcut = getattr(args, "dataset", None)
    if shortcut:
        contract = resolve_dataset_shortcut_contract(
            str(shortcut),
            dataset_variant=getattr(args, "dataset_variant", None),
        )
        if contract is not None:
            apply_dataset_selector_contract(args, contract)
    normalize_tokenizer_selection_args(args)
    return str(args.dataset_alias)


def dataset_download_kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "download_if_missing": bool(getattr(args, "download_if_missing", False)),
        "dataset_hf_path": getattr(args, "dataset_hf_path", None),
        "dataset_variant": getattr(args, "dataset_variant", None),
        "dataset_train_shards": getattr(args, "dataset_train_shards", None),
        "dataset_repo_id": getattr(args, "dataset_repo_id", None),
        "dataset_remote_root_prefix": getattr(args, "dataset_remote_root_prefix", None),
        "dataset_train_file": getattr(args, "dataset_train_file", None),
        "dataset_val_file": getattr(args, "dataset_val_file", None),
        "tokenizer_hf_path": getattr(args, "tokenizer_hf_path", None),
        "tokenizer_repo_id": getattr(args, "tokenizer_repo_id", None),
        "tokenizer_remote_root_prefix": getattr(args, "tokenizer_remote_root_prefix", None),
        "tokenizer_repo_type": getattr(args, "tokenizer_repo_type", None),
    }


def tokenizer_download_kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "tokenizer_hf_path": getattr(args, "tokenizer_hf_path", None),
        "tokenizer_repo_id": getattr(args, "tokenizer_repo_id", None),
        "tokenizer_remote_root_prefix": getattr(args, "tokenizer_remote_root_prefix", None),
        "tokenizer_repo_type": getattr(args, "tokenizer_repo_type", None),
    }


def _pretraining_file_conflicts(args: argparse.Namespace) -> list[str]:
    conflicts: list[str] = []
    if getattr(args, "tinystories", False):
        conflicts.append("--tinystories")
    if getattr(args, "dataset", None):
        conflicts.append("--dataset")
    if str(getattr(args, "dataset_alias", DEFAULT_DATASET_ALIAS)) != DEFAULT_DATASET_ALIAS:
        conflicts.append("--dataset-alias")
    if getattr(args, "dataset_hf_path", None):
        conflicts.append("--dataset-hf-path")
    if getattr(args, "dataset_variant", None):
        conflicts.append("--dataset-variant")
    if getattr(args, "dataset_train_shards", None) is not None:
        conflicts.append("--dataset-train-shards")
    if getattr(args, "dataset_repo_id", None):
        conflicts.append("--dataset-repo-id")
    if getattr(args, "dataset_remote_root_prefix", None):
        conflicts.append("--dataset-remote-root-prefix")
    if getattr(args, "dataset_train_file", None):
        conflicts.append("--dataset-train-file")
    if getattr(args, "dataset_val_file", None):
        conflicts.append("--dataset-val-file")
    return conflicts


def resolve_pretraining_file_dataset(args: argparse.Namespace) -> argparse.Namespace:
    raw_value = getattr(args, "pretraining_file", None)
    if not raw_value:
        return args

    conflicts = _pretraining_file_conflicts(args)
    if conflicts:
        joined = ", ".join(conflicts)
        raise ValueError(f"--pretraining-file cannot be combined with {joined}")

    source_path = Path(str(raw_value)).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Pretraining file {source_path} does not exist.")
    if not source_path.is_file():
        raise ValueError(f"--pretraining-file must point to a file, got {source_path}.")
    if source_path.suffix.lower() != ".txt":
        raise ValueError(f"--pretraining-file requires a .txt file, got {source_path.name!r}.")
    source_path = source_path.resolve(strict=True)

    resolved_alias = getattr(args, "_resolved_pretraining_dataset_alias", None)
    if resolved_alias:
        args.dataset_alias = str(resolved_alias)
        args.pretraining_file = str(source_path)
        return args

    # Create a lightweight raw-text dataset adapter so the rest of the training
    # stack can reuse its existing alias-based loaders unchanged.
    adapter_dir = Path(
        tempfile.mkdtemp(prefix=f"neuralfn-pretraining-{source_path.stem[:32]}-")
    ).resolve(strict=True)
    data_path = adapter_dir / "data.txt"
    data_path.symlink_to(source_path)
    meta = {
        "source": "local_pretraining_file",
        "pretraining_file": str(source_path),
    }
    (adapter_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    args.pretraining_file = str(source_path)
    args.dataset_alias = str(adapter_dir)
    args._resolved_pretraining_dataset_alias = str(adapter_dir)
    return args


def add_raw_text_tokenizer_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--tokenizer",
        choices=SUPPORTED_TOKENIZER_CHOICES,
        dest="raw_text_encoding_override",
        default=None,
        help=(
            "Tokenizer to use for raw-text datasets and prompt encoding. "
            "Sentencepiece variants also select the cached golf dataset variant for --dataset golf1/golf10."
        ),
    )
    group.add_argument(
        "--tokgpt2",
        action="store_const",
        const="gpt2",
        dest="raw_text_encoding_override",
        help="Use the GPT-2 byte-level BPE tokenizer (~50k vocab).",
    )
    group.add_argument(
        "--cl100k",
        action="store_const",
        const="cl100k_base",
        dest="raw_text_encoding_override",
        help="Use cl100k_base for raw-text datasets and prompt encoding.",
    )
    group.add_argument(
        "--o200k",
        action="store_const",
        const="o200k_base",
        dest="raw_text_encoding_override",
        help="Use o200k_base for raw-text datasets and prompt encoding.",
    )
    parser.add_argument(
        "--tokenizer-hf-path",
        default=None,
        help=(
            "Optional Hugging Face repo path to use when downloading missing shared sentencepiece tokenizer assets. "
            f"Defaults to {DEFAULT_TOKENIZER_HF_PATH!r}."
        ),
    )
    parser.add_argument("--tokenizer-repo-id", default=None, help="Optional explicit repo id for tokenizer asset downloads.")
    parser.add_argument(
        "--tokenizer-remote-root-prefix",
        default=None,
        help=(
            "Optional remote directory prefix inside the tokenizer repo. "
            f"Defaults to {DEFAULT_TOKENIZER_REMOTE_ROOT_PREFIX!r}."
        ),
    )
    parser.add_argument(
        "--tokenizer-repo-type",
        choices=SUPPORTED_HF_REPO_TYPES,
        default=None,
        help=(
            "Hugging Face repo type for tokenizer asset downloads. "
            f"Defaults to {DEFAULT_TOKENIZER_REPO_TYPE!r}."
        ),
    )
    parser.set_defaults(tokenizer=None, raw_text_encoding_override=None)


def _existing_dataset_is_raw_text(alias: str) -> bool:
    ds_dir = DATASETS_DIR / alias
    if ds_dir.is_dir():
        meta_path = ds_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if meta.get("data_format") == "uint16_shards":
            return False
        if (ds_dir / "data.txt").exists():
            return True
        return any(
            candidate.is_file()
            and candidate.name != "meta.json"
            and candidate.suffix in {".txt", ".json", ".jsonl", ".csv"}
            for candidate in ds_dir.iterdir()
        )
    return (DATASETS_DIR / f"{alias}.txt").exists()


def dataset_selection_uses_raw_text(args: argparse.Namespace) -> bool:
    if getattr(args, "dataset_train_file", None):
        return True
    return _existing_dataset_is_raw_text(str(args.dataset_alias))


def resolve_raw_text_tokenizer_policy_for_preset(
    preset_name: str,
    *,
    encoding_override: str | None = None,
    prefer_cl100k: bool = False,
) -> dict[str, object]:
    spec = build_model_spec_from_config({"preset": preset_name}, preview_defaults=True)
    backbone = str(spec.template.backbone)
    encoding_name = raw_text_encoding_name_for_backbone(
        backbone,
        prefer_cl100k=prefer_cl100k,
        encoding_override=normalize_raw_text_encoding_name(encoding_override),
    )
    return {
        "preset_name": preset_name,
        "backbone": backbone,
        "encoding_name": encoding_name,
        "encoding_vocab_size": raw_text_encoding_vocab_size(encoding_name),
        "legacy_gpt2": encoding_name == "gpt2",
    }


def apply_raw_text_tokenizer_policy(
    args: argparse.Namespace,
    *,
    preset_name: str,
    default_vocab_size: int,
) -> argparse.Namespace:
    explicit_override = normalize_raw_text_encoding_name(
        getattr(args, "tokenizer", None) or getattr(args, "raw_text_encoding_override", None)
    )
    dataset_default = default_tokenizer_for_dataset(getattr(args, "dataset", None))
    policy = resolve_raw_text_tokenizer_policy_for_preset(
        preset_name,
        encoding_override=explicit_override or dataset_default,
        prefer_cl100k=dataset_default is None and str(getattr(args, "dataset", "") or "").strip().lower() in {"shakespear", "shakespeare"},
    )
    args.raw_text_encoding_name = str(policy["encoding_name"])
    args.tokenizer = args.raw_text_encoding_name
    args.raw_text_encoding_vocab_size = int(policy["encoding_vocab_size"])
    args.raw_text_legacy_gpt2 = bool(policy["legacy_gpt2"])
    args.raw_text_selected = dataset_selection_uses_raw_text(args)

    if args.raw_text_selected:
        validate_raw_text_tokenizer_availability(
            args.raw_text_encoding_name,
            download_if_missing=bool(getattr(args, "download_if_missing", False)),
            dataset_alias=getattr(args, "dataset_alias", None),
            **tokenizer_download_kwargs_from_args(args),
        )
        configured_vocab = int(getattr(args, "vocab_size", default_vocab_size))
        if configured_vocab == int(default_vocab_size):
            args.vocab_size = args.raw_text_encoding_vocab_size
        elif configured_vocab != args.raw_text_encoding_vocab_size:
            raise ValueError(
                f"Preset {preset_name!r} uses {args.raw_text_encoding_name} for raw-text datasets, "
                f"which requires vocab_size={args.raw_text_encoding_vocab_size}. "
                f"Received vocab_size={configured_vocab}. "
                "Use the matching vocab size or pass --tokenizer to switch tokenization."
            )
    return args


def apply_cached_tokenizer_vocab_policy(
    args: argparse.Namespace,
    *,
    dataset_name: str,
    dataset_path: Path,
    dataset_meta: dict[str, object],
    default_vocab_size: int,
) -> dict[str, object]:
    contract = validate_cached_tokenizer_contract(
        dataset_name,
        dataset_path=dataset_path,
        dataset_meta=dict(dataset_meta),
    )
    if contract is None:
        return dict(dataset_meta)

    tokenizer_vocab_size = int(contract["tokenizer_vocab_size"])
    configured_vocab = int(getattr(args, "vocab_size", default_vocab_size))
    if configured_vocab == int(default_vocab_size):
        args.vocab_size = tokenizer_vocab_size
    elif configured_vocab != tokenizer_vocab_size:
        raise ValueError(
            f"Dataset alias {dataset_name!r} requires vocab_size={tokenizer_vocab_size} "
            "from its tokenizer-backed cached token contract, "
            f"but received vocab_size={configured_vocab}."
        )

    resolved_meta = dict(dataset_meta)
    resolved_meta["tokenizer_vocab_size"] = tokenizer_vocab_size
    return resolved_meta


def build_parser() -> argparse.ArgumentParser:
    parser = create_argument_parser(description="Train jepa_semantic_hybrid with the NeuralFn CUDA harness.")
    parser.add_argument(
        "--megakernel",
        action="store_true",
        help="Use the jepa_semantic_hybrid_megakernel preset/runtime.",
    )
    parser.add_argument("--run-id", default=env_str("RUN_ID", JEPA_DEFAULTS["run_id"]))
    parser.add_argument("--seed", type=int, default=env_int("SEED", JEPA_DEFAULTS["seed"]))
    parser.add_argument("--device", default=env_str("DEVICE", JEPA_DEFAULTS["device"]))
    add_dataset_selector_arguments(
        parser,
        default_alias=env_str("DATASET_ALIAS", JEPA_DEFAULTS["dataset_alias"]),
    )
    add_dataset_download_arguments(parser)
    add_pretraining_file_argument(parser)
    add_raw_text_tokenizer_arguments(parser)
    parser.add_argument("--output", default=env_str("OUTPUT", ""))

    add_max_steps_argument(parser, default=env_int("ITERATIONS", JEPA_DEFAULTS["max_steps"]))
    parser.add_argument("--train-seq-len", type=int, default=env_int("TRAIN_SEQ_LEN", JEPA_DEFAULTS["train_seq_len"]))
    parser.add_argument("--batch-size", type=int, default=env_int("BATCH_SIZE", JEPA_DEFAULTS["batch_size"]))
    parser.add_argument(
        "--train-batch-tokens",
        type=int,
        default=env_int("TRAIN_BATCH_TOKENS", JEPA_DEFAULTS["train_batch_tokens"]),
    )
    parser.add_argument("--eval-batches", type=int, default=env_int("EVAL_BATCHES", JEPA_DEFAULTS["eval_batches"]))
    parser.add_argument("--eval-batch-size", type=int, default=env_int("EVAL_BATCH_SIZE", JEPA_DEFAULTS["eval_batch_size"]))
    parser.add_argument("--train-log-every", type=int, default=env_int("TRAIN_LOG_EVERY", JEPA_DEFAULTS["train_log_every"]))
    parser.add_argument("--val-loss-every", type=int, default=env_int("VAL_LOSS_EVERY", JEPA_DEFAULTS["val_loss_every"]))
    add_max_wallclock_seconds_argument(
        parser,
        default=env_float("MAX_WALLCLOCK_SECONDS", JEPA_DEFAULTS["max_wallclock_seconds"]),
    )
    parser.add_argument("--warmup-steps", type=int, default=env_int("WARMUP_STEPS", JEPA_DEFAULTS["warmup_steps"]))
    add_warmdown_fraction_argument(
        parser,
        default=env_float("WARMDOWN_FRACTION", JEPA_DEFAULTS["warmdown_fraction"]),
    )
    add_all_train_rows_argument(parser)
    add_evolutionary_training_arguments(parser)

    parser.add_argument("--vocab-size", type=int, default=env_int("VOCAB_SIZE", JEPA_DEFAULTS["vocab_size"]))
    parser.add_argument("--num-layers", type=int, default=env_int("NUM_LAYERS", JEPA_DEFAULTS["num_layers"]))
    parser.add_argument("--model-dim", type=int, default=env_int("MODEL_DIM", JEPA_DEFAULTS["model_dim"]))
    parser.add_argument("--num-heads", type=int, default=env_int("NUM_HEADS", JEPA_DEFAULTS["num_heads"]))
    parser.add_argument("--num-kv-heads", type=int, default=env_int("NUM_KV_HEADS", JEPA_DEFAULTS["num_kv_heads"]))
    parser.add_argument("--mlp-mult", type=float, default=env_float("MLP_MULT", JEPA_DEFAULTS["mlp_mult"]))
    parser.add_argument("--multiple-of", type=int, default=env_int("MULTIPLE_OF", JEPA_DEFAULTS["multiple_of"]))
    parser.add_argument("--experts", type=int, default=env_int("EXPERTS", JEPA_DEFAULTS["experts"]))
    parser.add_argument("--top-k", type=int, default=env_int("TOP_K", JEPA_DEFAULTS["top_k"]))
    parser.add_argument(
        "--experimental-semantic-router-vecs",
        action="store_true",
        help="Add semantic_router_vecs to the graph contract and route directly from the normalized semantic router vector.",
    )
    parser.add_argument("--ema-decay", type=float, default=env_float("EMA_DECAY", JEPA_DEFAULTS["ema_decay"]))
    parser.add_argument("--rope-base", type=float, default=env_float("ROPE_BASE", JEPA_DEFAULTS["rope_base"]))
    parser.add_argument("--qk-gain-init", type=float, default=env_float("QK_GAIN_INIT", JEPA_DEFAULTS["qk_gain_init"]))
    parser.add_argument("--logit-softcap", type=float, default=env_float("LOGIT_SOFTCAP", JEPA_DEFAULTS["logit_softcap"]))

    parser.add_argument("--optimizer-profile", default=env_str("OPTIMIZER_PROFILE", JEPA_DEFAULTS["optimizer_profile"]))
    parser.add_argument("--learning-rate", type=float, default=env_float("LEARNING_RATE", JEPA_DEFAULTS["learning_rate"]))
    add_lr_schedule_arguments(parser)
    parser.add_argument("--weight-decay", type=float, default=env_float("WEIGHT_DECAY", JEPA_DEFAULTS["weight_decay"]))
    parser.add_argument("--embed-lr", type=float, default=env_float("EMBED_LR", JEPA_DEFAULTS["embed_lr"]))
    parser.add_argument("--head-lr", type=float, default=env_float("HEAD_LR", JEPA_DEFAULTS["head_lr"]))
    parser.add_argument("--tied-embed-lr", type=float, default=env_float("TIED_EMBED_LR", JEPA_DEFAULTS["tied_embed_lr"]))
    parser.add_argument("--matrix-lr", type=float, default=env_float("MATRIX_LR", JEPA_DEFAULTS["matrix_lr"]))
    parser.add_argument("--scalar-lr", type=float, default=env_float("SCALAR_LR", JEPA_DEFAULTS["scalar_lr"]))
    parser.add_argument("--muon-momentum", type=float, default=env_float("MUON_MOMENTUM", JEPA_DEFAULTS["muon_momentum"]))
    parser.add_argument("--muon-backend-steps", type=int, default=env_int("MUON_BACKEND_STEPS", JEPA_DEFAULTS["muon_backend_steps"]))
    parser.add_argument(
        "--muon-momentum-warmup-start",
        type=float,
        default=env_float("MUON_MOMENTUM_WARMUP_START", JEPA_DEFAULTS["muon_momentum_warmup_start"]),
    )
    parser.add_argument(
        "--muon-momentum-warmup-steps",
        type=int,
        default=env_int("MUON_MOMENTUM_WARMUP_STEPS", JEPA_DEFAULTS["muon_momentum_warmup_steps"]),
    )
    parser.add_argument("--beta1", type=float, default=env_float("BETA1", JEPA_DEFAULTS["beta1"]))
    parser.add_argument("--beta2", type=float, default=env_float("BETA2", JEPA_DEFAULTS["beta2"]))
    parser.add_argument("--adam-eps", type=float, default=env_float("ADAM_EPS", JEPA_DEFAULTS["adam_eps"]))
    parser.add_argument("--grad-clip-norm", type=float, default=env_float("GRAD_CLIP_NORM", JEPA_DEFAULTS["grad_clip_norm"]))

    parser.add_argument("--ar-loss-coef", type=float, default=env_float("AR_LOSS_COEF", JEPA_DEFAULTS["ar_loss_coef"]))
    parser.add_argument("--jepa-loss-coef", type=float, default=env_float("JEPA_LOSS_COEF", JEPA_DEFAULTS["jepa_loss_coef"]))
    parser.add_argument(
        "--semantic-align-loss-coef",
        type=float,
        default=env_float("SEMANTIC_ALIGN_LOSS_COEF", JEPA_DEFAULTS["semantic_align_loss_coef"]),
    )
    return parser


def build_graph(args: argparse.Namespace, dataset_name: str):
    builder = build_jepa_semantic_hybrid_megakernel_spec if args.megakernel else build_jepa_semantic_hybrid_spec
    semantic_vocab_ref = semantic_module.semantic_vocab_ref_for_tokenizer(effective_tokenizer_name_for_args(args))
    spec = builder(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_multiplier=args.mlp_mult,
        multiple_of=args.multiple_of,
        experts=args.experts,
        top_k=args.top_k,
        ema_decay=args.ema_decay,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        logit_softcap=args.logit_softcap,
        semantic_vocab_ref=semantic_vocab_ref,
        experimental_semantic_router_vecs=bool(args.experimental_semantic_router_vecs),
        ar_loss_coef=args.ar_loss_coef,
        jepa_loss_coef=args.jepa_loss_coef,
        semantic_align_loss_coef=args.semantic_align_loss_coef,
    )
    graph = build_gpt_root_graph(name=graph_name(megakernel=bool(args.megakernel)), model_spec=spec)
    graph.torch_config = {**graph.torch_config, "device": args.device, "amp_dtype": "float32"}

    ds_node = graph.nodes["dataset_source"]
    ds_cfg = dict(ds_node.neuron_def.module_config or {})
    ds_node.neuron_def.module_config = {
        **ds_cfg,
        "dataset_names": [dataset_name],
        "seq_len": args.train_seq_len,
    }
    apply_sanitized_template_spec(
        graph,
        raw_text_encoding_name=str(getattr(args, "raw_text_encoding_name", "") or ""),
    )

    return graph, spec


def print_graph_summary(graph) -> None:
    print("Root graph nodes:")
    for node_id in graph.nodes:
        node = graph.nodes[node_id]
        ports = [port.name for port in node.neuron_def.output_ports]
        print(f"  - {node_id}: kind={node.neuron_def.kind} module_type={node.neuron_def.module_type!r} outputs={ports}")
    print("Input node ids:", graph.input_node_ids)
    print("Output node ids:", graph.output_node_ids)


def dataset_tokenizer_summary_lines(dataset_meta: dict[str, object]) -> list[str]:
    tokenizer_name = dataset_meta.get("tokenizer_name")
    tokenizer_encoding = dataset_meta.get("tokenizer_encoding")
    tokenizer_vocab_size = dataset_meta.get("tokenizer_vocab_size")

    lines: list[str] = []
    if tokenizer_name:
        lines.append("  - Tokenizer backend: sentencepiece")
        lines.append(f"  - Tokenizer: {tokenizer_name}")
    elif tokenizer_encoding:
        lines.append("  - Tokenizer backend: tiktoken")

    if tokenizer_encoding:
        lines.append(f"  - Tokenizer encoding: {tokenizer_encoding}")
    if tokenizer_vocab_size is not None:
        lines.append(f"  - Tokenizer vocab size: {tokenizer_vocab_size}")
    return lines


def build_dataset_manifest(dataset_name: str, dataset_meta: dict[str, object]) -> dict[str, object]:
    manifest: dict[str, object] = {
        "dataset_alias": dataset_name,
        "source": dataset_meta.get("source"),
        "hf_path": dataset_meta.get("hf_path"),
        "hf_split": dataset_meta.get("hf_split"),
        "variant": dataset_meta.get("variant"),
        "train_file": dataset_meta.get("train_file"),
        "val_file": dataset_meta.get("val_file"),
        "tokenizer_name": dataset_meta.get("tokenizer_name"),
        "tokenizer_encoding": dataset_meta.get("tokenizer_encoding"),
        "tokenizer_vocab_size": dataset_meta.get("tokenizer_vocab_size"),
    }
    return {key: value for key, value in manifest.items() if value is not None}


def build_training_manifest(
    summary: dict[str, Any] | None,
    *,
    dataset_name: str | None,
    dataset_meta: dict[str, object] | None,
    raw_text_encoding_name: str | None = None,
) -> dict[str, Any] | None:
    if summary is None:
        return None
    manifest = dict(summary)
    if isinstance(manifest.get("model_spec"), dict):
        manifest["model_spec"] = sanitized_model_spec_dict(
            manifest["model_spec"],
            raw_text_encoding_name=raw_text_encoding_name,
        )
    if dataset_name is not None and dataset_meta is not None:
        dataset_manifest = build_dataset_manifest(dataset_name, dataset_meta)
        if raw_text_encoding_name and "tokenizer_encoding" not in dataset_manifest:
            if is_sentencepiece_tokenizer_name(raw_text_encoding_name):
                dataset_manifest["tokenizer_name"] = raw_text_encoding_name
            else:
                dataset_manifest["tokenizer_encoding"] = raw_text_encoding_name
            dataset_manifest["tokenizer_vocab_size"] = raw_text_encoding_vocab_size(raw_text_encoding_name)
        manifest["dataset"] = dataset_manifest
    elif raw_text_encoding_name:
        dataset_manifest = {
            "tokenizer_vocab_size": raw_text_encoding_vocab_size(raw_text_encoding_name),
        }
        if is_sentencepiece_tokenizer_name(raw_text_encoding_name):
            dataset_manifest["tokenizer_name"] = raw_text_encoding_name
        else:
            dataset_manifest["tokenizer_encoding"] = raw_text_encoding_name
        manifest["dataset"] = dataset_manifest
    return manifest


def _resolve_sentencepiece_model_path(dataset_path: Path, dataset_meta: dict[str, object]) -> Path | None:
    model_candidates: list[Path] = []
    tokenizer_files = dataset_meta.get("tokenizer_files")
    if isinstance(tokenizer_files, list):
        for filename in tokenizer_files:
            if isinstance(filename, str) and filename.endswith(".model"):
                model_candidates.append(dataset_path / "tokenizers" / filename)
    if dataset_path.is_dir():
        model_candidates.extend(sorted((dataset_path / "tokenizers").glob("*.model")))
    return next((path for path in model_candidates if path.exists()), None)


def build_tokenizer_manifest(
    dataset_path: Path | None,
    dataset_meta: dict[str, object] | None,
    *,
    raw_text_encoding_name: str | None = None,
) -> dict[str, object] | None:
    metadata = dict(dataset_meta or {})
    tokenizer_name = str(metadata.get("tokenizer_name") or raw_text_encoding_name or "").strip()
    model_path: Path | None = None
    if dataset_path is not None:
        model_path = _resolve_sentencepiece_model_path(dataset_path, metadata)
    if model_path is None and tokenizer_name and is_sentencepiece_tokenizer_name(tokenizer_name):
        model_path = resolve_sentencepiece_model_path(tokenizer_name)
    if model_path is not None:
        vocab_size = metadata.get("tokenizer_vocab_size")
        if vocab_size is None and tokenizer_name:
            vocab_size = raw_text_encoding_vocab_size(tokenizer_name)
        return {
            "backend": "sentencepiece",
            "model_file": model_path.name,
            "tokenizer_name": tokenizer_name or metadata.get("tokenizer_name"),
            "tokenizer_vocab_size": None if vocab_size is None else int(vocab_size),
            "model_proto_b64": base64.b64encode(model_path.read_bytes()).decode("ascii"),
        }

    encoding_name = str(
        metadata.get("tokenizer_encoding")
        or raw_text_encoding_name
        or ""
    ).strip()
    if not encoding_name:
        return None
    vocab_size = metadata.get("tokenizer_vocab_size")
    if vocab_size is None:
        vocab_size = raw_text_encoding_vocab_size(encoding_name)
    return {
        "backend": "tiktoken",
        "encoding_name": encoding_name,
        "tokenizer_vocab_size": int(vocab_size),
    }


def print_data_source_summary(dataset_name: str, dataset_path: Path, dataset_meta: dict[str, object], graph) -> None:
    semantic_vocab_ref = semantic_module.semantic_vocab_ref_for_graph(graph)
    semantic_vocab = semantic_module.resolve_semantic_vocab_path(semantic_vocab_ref)
    val_files = sorted(dataset_path.glob("fineweb_val_*.bin")) if dataset_path.is_dir() else []
    explicit_val_path = dataset_path / "val.txt" if dataset_path.is_dir() and (dataset_path / "val.txt").exists() else None

    print("Text data source:")
    print(f"  - Local dataset name: {dataset_name}")
    print(f"  - Local storage path: {dataset_path}")
    if dataset_meta:
        print(f"  - Source: {dataset_meta.get('source')}")
        print(f"  - HF path: {dataset_meta.get('hf_path')}")
        print(f"  - Split: {dataset_meta.get('hf_split')}")
        print(f"  - Variant: {dataset_meta.get('variant')}")
        for line in dataset_tokenizer_summary_lines(dataset_meta):
            print(line)
        print(f"  - Validation shards: {dataset_meta.get('val_shards')}")
        print(f"  - Train file: {dataset_meta.get('train_file')}")
        print(f"  - Validation file: {dataset_meta.get('val_file')}")
    if val_files:
        print(f"  - Validation shard path: {val_files[0]}")
    if explicit_val_path is not None:
        print(f"  - Validation text path: {explicit_val_path}")

    print("Semantic data source:")
    print(f"  - Source: {semantic_vocab_ref}")
    print(f"  - Vocabulary JSON: {semantic_vocab}")

    ds_cfg = dict(graph.nodes["dataset_source"].neuron_def.module_config or {})
    sem_cfg = dict(graph.nodes["semantic_data_source"].neuron_def.module_config or {})
    print("dataset_source config:", ds_cfg)
    print("semantic_data_source config:", sem_cfg)


def save_artifacts(
    graph,
    weights_path: Path,
    graph_path: Path,
    *,
    training_manifest: dict[str, Any] | None = None,
    dataset_name: str | None = None,
    dataset_path: Path | None = None,
    dataset_meta: dict[str, object] | None = None,
    raw_text_encoding_name: str | None = None,
) -> None:
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    template_spec = apply_sanitized_template_spec(
        graph,
        raw_text_encoding_name=raw_text_encoding_name,
    )
    objective = serialized_model_objective(template_spec)
    export_to_pt(graph, weights_path)
    artifact_metadata = dict(graph.torch_config.get("artifact_metadata", {}) or {})
    artifact_metadata["artifact_format_version"] = 2
    artifact_metadata["weights_file"] = os.path.relpath(weights_path, graph_path.parent)
    template = dict(template_spec.get("template", {}) or {})
    runtime = str(template.get("runtime", "")).strip().lower()
    if runtime:
        artifact_metadata["checkpoint_runtime"] = runtime
    artifact_metadata["stores_embedded_module_state"] = False
    artifact_metadata.pop("semantic_vocab_ref", None)
    artifact_metadata.pop("experimental_semantic_router_vecs", None)
    if objective in SEMANTIC_OBJECTIVES:
        artifact_metadata["semantic_vocab_ref"] = semantic_module.semantic_vocab_ref_for_graph(graph)
        artifact_metadata["experimental_semantic_router_vecs"] = graph_uses_semantic_router_vecs(graph)
    torch_config = {
        **graph.torch_config,
        "artifact_metadata": artifact_metadata,
    }
    resolved_training_manifest = build_training_manifest(
        training_manifest,
        dataset_name=dataset_name,
        dataset_meta=dataset_meta,
        raw_text_encoding_name=raw_text_encoding_name,
    )
    if resolved_training_manifest is not None:
        torch_config["training_manifest"] = resolved_training_manifest
    tokenizer_manifest = build_tokenizer_manifest(
        dataset_path,
        dataset_meta,
        raw_text_encoding_name=raw_text_encoding_name,
    )
    if tokenizer_manifest is not None:
        torch_config["tokenizer_manifest"] = tokenizer_manifest
    graph.torch_config = torch_config
    save_graph(graph, graph_path, include_module_state=False)


class DualSourceEvalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        text_dataset: torch.utils.data.Dataset,
        semantic_tensors: dict[str, torch.Tensor],
        roles: list[str],
    ) -> None:
        self.text_dataset = text_dataset
        self.semantic_tensors = {str(name): tensor for name, tensor in semantic_tensors.items()}
        self.roles = list(roles)
        self.length = min(len(text_dataset), *(int(tensor.size(0)) for tensor in self.semantic_tensors.values()))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        sample = self.text_dataset[idx]
        if isinstance(sample, torch.Tensor):
            x = sample
            y = sample
        else:
            values = tuple(sample)
            x = values[0]
            y = values[1] if len(values) > 1 else values[0]
        mapped: list[torch.Tensor] = []
        for role in self.roles:
            if role in {"tokens", "enc_tokens", "dec_tokens"}:
                mapped.append(x)
            elif role == "targets":
                mapped.append(y)
            elif role in self.semantic_tensors:
                mapped.append(self.semantic_tensors[role][idx])
            else:
                raise ValueError(f"Unsupported eval dataset role {role!r}")
        return tuple(mapped)


def graph_uses_semantic_router_vecs(graph) -> bool:
    template_spec = dict(graph.torch_config.get("template_spec", {}) or {})
    if bool(template_spec.get("experimental_semantic_router_vecs", False)):
        return True
    sem_node = graph.nodes.get("semantic_data_source")
    if sem_node is None:
        return False
    return any(port.name == "semantic_router_vecs" for port in sem_node.neuron_def.output_ports)


def graph_input_roles(graph) -> list[str]:
    roles: list[str] = []
    for node_id in graph.input_node_ids:
        node = graph.nodes[node_id]
        roles.extend(port.name for port in node.neuron_def.output_ports)
    return roles


def load_semantic_inputs(graph, active_dims: int) -> dict[str, torch.Tensor]:
    vocab_ref = semantic_module.semantic_vocab_ref_for_graph(graph)
    vocab = semantic_module.ConversationalVocabulary(vocab_ref)
    _ids, targets = semantic_module.load_training_targets(active_dims=active_dims, vocab=vocab)
    semantic_inputs: dict[str, torch.Tensor] = {
        "sem_targets": torch.from_numpy(targets.astype(np.int64)),
    }
    if graph_uses_semantic_router_vecs(graph):
        router_vecs = semantic_module.semantic_targets_to_router_vectors(targets, vocab=vocab)
        semantic_inputs["semantic_router_vecs"] = torch.from_numpy(router_vecs.astype(np.float32))
    return semantic_inputs


def load_semantic_tokens(active_dims: int, *, vocab_ref: str | None = None) -> torch.Tensor:
    vocab = semantic_module.ConversationalVocabulary(vocab_ref or None)
    _ids, targets = semantic_module.load_training_targets(active_dims=active_dims, vocab=vocab)
    return torch.from_numpy(targets.astype(np.int64))


def _require_validation_window(tokens: np.ndarray, *, seq_len: int, source_label: str) -> None:
    min_tokens = seq_len + 1
    if tokens.size < min_tokens:
        raise FileNotFoundError(
            f"{source_label} does not contain enough tokens to build a validation window of length {seq_len}."
        )


def _raw_text_eval_file(dataset_path: Path) -> Path | None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    if dataset_path.is_dir():
        data_path = dataset_path / "data.txt"
        if data_path.is_symlink() and not data_path.exists():
            raise FileNotFoundError(f"Training text file {data_path} points to a missing source file.")
    return dataset_manager_module._raw_text_data_file_for_path(dataset_path)


def safe_evaluate_validation_loss(
    evaluate_fn: Callable[[], float],
    *,
    logger: logging.Logger | None = None,
) -> float:
    try:
        return float(evaluate_fn())
    except Exception as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        message = f"Validation skipped: {detail}"
        if logger is not None:
            logger.warning(message)
        else:
            print(message, file=sys.stderr)
        return float("nan")


def load_val_token_dataset(
    dataset_path: Path,
    seq_len: int,
    *,
    encoding_name: str = "gpt2",
) -> torch.utils.data.Dataset:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    if dataset_path.is_dir():
        val_files = sorted(dataset_path.glob("fineweb_val_*.bin"))
        if val_files:
            arrays = [np.memmap(path, dtype=np.uint16, mode="r")[_shard_header_offset_uint16(path):] for path in val_files]
            return MemmapTokenDataset(arrays, seq_len)
        explicit_val_path = dataset_path / "val.txt"
        if explicit_val_path.exists():
            text = explicit_val_path.read_text(encoding="utf-8")
            tokens = np.asarray(encode_raw_text(text, encoding_name=encoding_name), dtype=np.int64)
            _require_validation_window(tokens, seq_len=seq_len, source_label=f"Validation file {explicit_val_path}")
            return MemmapTokenDataset([tokens], seq_len)
    raw_text_file = _raw_text_eval_file(dataset_path)
    if raw_text_file is not None:
        text = raw_text_file.read_text(encoding="utf-8")
        tokens = np.asarray(encode_raw_text(text, encoding_name=encoding_name), dtype=np.int64)
        _require_validation_window(tokens, seq_len=seq_len, source_label=f"Training file {raw_text_file}")
        holdout_tokens = max(seq_len + 1, int(math.ceil(tokens.size * VAL_HOLDOUT_FRACTION)))
        val_tokens = tokens[-holdout_tokens:]
        return MemmapTokenDataset([val_tokens], seq_len)
    dataset_name = dataset_path.name if dataset_path.is_dir() else dataset_path.stem
    tokens = np.asarray(_load_tokens_for(dataset_name, None, encoding_name=encoding_name), dtype=np.int64)
    _require_validation_window(tokens, seq_len=seq_len, source_label=f"Dataset {dataset_name!r}")
    holdout_tokens = max(seq_len + 1, int(math.ceil(tokens.size * VAL_HOLDOUT_FRACTION)))
    val_tokens = tokens[-holdout_tokens:]
    return MemmapTokenDataset([val_tokens], seq_len)


def evaluate_model(
    graph,
    dataset_path: Path,
    *,
    device: str,
    seq_len: int,
    batch_size: int,
    eval_batches: int,
    encoding_name: str | None = None,
) -> float:
    if eval_batches <= 0:
        return float("nan")

    template_spec = dict(graph.torch_config.get("template_spec", {}))
    resolved_encoding_name = encoding_name or raw_text_encoding_name_for_template_spec(template_spec)
    val_dataset = load_val_token_dataset(
        dataset_path,
        seq_len=seq_len,
        encoding_name=resolved_encoding_name,
    )
    vocab = semantic_module.ConversationalVocabulary(semantic_module.semantic_vocab_ref_for_graph(graph))
    active_dims = max(1, min(int((template_spec.get("block_spec") or {}).get("top_k", 2) or 2), vocab.num_vocab_dims))
    semantic_inputs = load_semantic_inputs(graph, active_dims)
    dual_dataset = DualSourceEvalDataset(val_dataset, semantic_inputs, graph_input_roles(graph))
    loader = torch.utils.data.DataLoader(dual_dataset, batch_size=batch_size, shuffle=False)

    compiled = CompiledTorchGraph(graph)
    compiled.to(device)
    compiled.eval()

    amp_dtype = torch.float32
    use_amp = False
    total_loss = 0.0
    total_rows = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= eval_batches:
                break
            flat_inputs = tuple(item.to(device) for item in batch)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                outputs = compiled(*flat_inputs)
                loss = outputs[0]
            batch_rows = int(flat_inputs[0].size(0))
            total_loss += float(loss.item()) * batch_rows
            total_rows += batch_rows
    return total_loss / max(total_rows, 1)


def estimate_text_schedule(
    dataset_name: str,
    *,
    seq_len: int,
    batch_size: int,
    train_batch_tokens: int,
    template_runtime: str = "compile",
    device: str = "cuda",
    drop_last: bool | None = None,
    all_train_rows: bool = False,
    encoding_name: str = "gpt2",
) -> dict[str, int | bool]:
    estimated_rows = estimate_dataset_sequence_count(
        dataset_name,
        seq_len=seq_len,
        encoding_name=encoding_name,
    )
    if estimated_rows is None:
        text_dataset = load_dataset_tensors([dataset_name], seq_len=seq_len, encoding_name=encoding_name)
        source_train_rows = len(text_dataset)
    else:
        source_train_rows = int(estimated_rows)
    microbatch_tokens = max(batch_size * seq_len, 1)
    grad_accum_steps = max(1, math.ceil(train_batch_tokens / microbatch_tokens))
    layout = _resolve_schedule_layout(
        source_train_rows=source_train_rows,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        template_runtime=template_runtime,
        device=device,
        drop_last=drop_last,
        all_train_rows=all_train_rows,
    )
    return {
        "source_train_rows": source_train_rows,
        "microbatch_tokens": microbatch_tokens,
        "effective_train_batch_tokens": microbatch_tokens * grad_accum_steps,
        "grad_accum_steps": grad_accum_steps,
        **layout,
    }


def estimate_schedule(
    dataset_name: str,
    *,
    seq_len: int,
    batch_size: int,
    train_batch_tokens: int,
    top_k: int,
    template_runtime: str = "compile",
    device: str = "cuda",
    drop_last: bool | None = None,
    all_train_rows: bool = False,
    encoding_name: str = "gpt2",
) -> dict[str, int | bool]:
    base = estimate_text_schedule(
        dataset_name,
        seq_len=seq_len,
        batch_size=batch_size,
        train_batch_tokens=train_batch_tokens,
        template_runtime=template_runtime,
        device=device,
        drop_last=drop_last,
        all_train_rows=all_train_rows,
        encoding_name=encoding_name,
    )
    vocab = semantic_module.ConversationalVocabulary(semantic_module.DEFAULT_SEMANTIC_VOCAB_REF)
    semantic_rows = int(
        load_semantic_tokens(
            active_dims=max(1, min(int(top_k), vocab.num_vocab_dims)),
            vocab_ref=vocab.ref,
        ).size(0)
    )
    source_train_rows = min(int(base["source_train_rows"]), semantic_rows)
    microbatch_tokens = base["microbatch_tokens"]
    grad_accum_steps = base["grad_accum_steps"]
    layout = _resolve_schedule_layout(
        source_train_rows=source_train_rows,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        template_runtime=template_runtime,
        device=device,
        drop_last=drop_last,
        all_train_rows=all_train_rows,
    )
    return {
        "text_rows": int(base["source_train_rows"]),
        "semantic_rows": semantic_rows,
        "microbatch_tokens": microbatch_tokens,
        "effective_train_batch_tokens": microbatch_tokens * grad_accum_steps,
        "grad_accum_steps": grad_accum_steps,
        **layout,
    }


def build_trainer_config(
    args: argparse.Namespace,
    *,
    resolved_epochs: int,
    drop_last: bool | None = None,
    max_steps: int | None = None,
    lr_decay_iters: int | None = None,
    max_wallclock_seconds: float | None = None,
    respect_epoch_boundaries: bool | None = None,
) -> TorchTrainConfig:
    resolved_drop_last = drop_last
    if resolved_drop_last is None and bool(getattr(args, "all_train_rows", False)):
        resolved_drop_last = False
    return TorchTrainConfig(
        epochs=resolved_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        max_steps=args.max_steps if max_steps is None else max_steps,
        optimizer_profile=args.optimizer_profile,
        train_batch_tokens=args.train_batch_tokens,
        warmup_steps=args.warmup_steps,
        warmdown_fraction=args.warmdown_fraction,
        lr_decay_iters=args.lr_decay_iters if lr_decay_iters is None else lr_decay_iters,
        min_lr=args.min_lr,
        max_wallclock_seconds=(
            args.max_wallclock_seconds if max_wallclock_seconds is None else max_wallclock_seconds
        ),
        embed_lr=args.embed_lr,
        head_lr=args.head_lr,
        tied_embed_lr=args.tied_embed_lr,
        matrix_lr=args.matrix_lr,
        scalar_lr=args.scalar_lr,
        muon_momentum=args.muon_momentum,
        muon_backend_steps=args.muon_backend_steps,
        muon_momentum_warmup_start=args.muon_momentum_warmup_start,
        muon_momentum_warmup_steps=args.muon_momentum_warmup_steps,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=args.adam_eps,
        grad_clip_norm=args.grad_clip_norm,
        drop_last=resolved_drop_last,
        respect_epoch_boundaries=(
            bool(getattr(args, "all_train_rows", False))
            if respect_epoch_boundaries is None
            else bool(respect_epoch_boundaries)
        ),
        evolutionary=bool(getattr(args, "evolutionary", False)),
        evo_population_size=int(getattr(args, "evo_population_size", EVOLUTIONARY_DEFAULTS["population_size"])),
        evo_mutation_rate=float(getattr(args, "evo_mutation_rate", EVOLUTIONARY_DEFAULTS["mutation_rate"])),
        evo_mutation_scale=float(getattr(args, "evo_mutation_scale", EVOLUTIONARY_DEFAULTS["mutation_scale"])),
        evo_crossover_rate=float(getattr(args, "evo_crossover_rate", EVOLUTIONARY_DEFAULTS["crossover_rate"])),
        evo_tournament_size=int(getattr(args, "evo_tournament_size", EVOLUTIONARY_DEFAULTS["tournament_size"])),
        evo_elite_count=int(getattr(args, "evo_elite_count", EVOLUTIONARY_DEFAULTS["elite_count"])),
        evo_seed=(
            int(args.seed)
            if getattr(args, "evo_seed", None) is None
            else int(args.evo_seed)
        ),
    )


def build_trainer_summary(trainer_cfg: TorchTrainConfig) -> dict[str, Any]:
    summary = {
        "epochs": trainer_cfg.epochs,
        "max_steps": trainer_cfg.max_steps,
        "batch_size": trainer_cfg.batch_size,
        "train_batch_tokens": trainer_cfg.train_batch_tokens,
        "optimization_method": TorchTrainer._optimization_method(trainer_cfg),
        "optimizer_profile": trainer_cfg.optimizer_profile,
        "learning_rate": trainer_cfg.learning_rate,
        "lr_decay_iters": trainer_cfg.lr_decay_iters,
        "min_lr": trainer_cfg.min_lr,
        "embed_lr": trainer_cfg.embed_lr,
        "head_lr": trainer_cfg.head_lr,
        "tied_embed_lr": trainer_cfg.tied_embed_lr,
        "matrix_lr": trainer_cfg.matrix_lr,
        "scalar_lr": trainer_cfg.scalar_lr,
        "warmup_steps": trainer_cfg.warmup_steps,
        "warmdown_fraction": trainer_cfg.warmdown_fraction,
        "max_wallclock_seconds": trainer_cfg.max_wallclock_seconds,
        "beta1": trainer_cfg.beta1,
        "beta2": trainer_cfg.beta2,
        "adam_eps": trainer_cfg.adam_eps,
        "muon_momentum": trainer_cfg.muon_momentum,
        "muon_backend_steps": trainer_cfg.muon_backend_steps,
        "muon_momentum_warmup_start": trainer_cfg.muon_momentum_warmup_start,
        "muon_momentum_warmup_steps": trainer_cfg.muon_momentum_warmup_steps,
        "grad_clip_norm": trainer_cfg.grad_clip_norm,
        "drop_last": trainer_cfg.drop_last,
        "respect_epoch_boundaries": trainer_cfg.respect_epoch_boundaries,
    }
    if trainer_cfg.evolutionary:
        summary["evolutionary"] = TorchTrainer._evolutionary_config_dict(trainer_cfg)
        summary["ignored_gradient_optimizer_fields"] = TorchTrainer._gradient_ignored_fields()
    return summary


def build_reference_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "adapted": {
            "ITERATIONS->max_steps": args.max_steps,
            "TRAIN_SEQ_LEN->train_seq_len": args.train_seq_len,
            "TRAIN_BATCH_TOKENS": args.train_batch_tokens,
            "TRAIN_LOG_EVERY->harness_progress_logging": args.train_log_every,
            "WARMUP_STEPS": args.warmup_steps,
            "WARMDOWN_FRACTION": args.warmdown_fraction,
            "LR_DECAY_ITERS": args.lr_decay_iters,
            "MIN_LR": args.min_lr,
            "MAX_WALLCLOCK_SECONDS": args.max_wallclock_seconds,
            "QK_GAIN_INIT": args.qk_gain_init,
            "ROPE_BASE": args.rope_base,
            "LOGIT_SOFTCAP": args.logit_softcap,
            "NUM_LAYERS": args.num_layers,
            "MODEL_DIM": args.model_dim,
            "NUM_HEADS": args.num_heads,
            "NUM_KV_HEADS": args.num_kv_heads,
            "MLP_MULT": args.mlp_mult,
            "VOCAB_SIZE": args.vocab_size,
            "EMBED_LR": args.embed_lr,
            "HEAD_LR": args.head_lr,
            "TIED_EMBED_LR": args.tied_embed_lr,
            "MATRIX_LR": args.matrix_lr,
            "SCALAR_LR": args.scalar_lr,
            "MUON_MOMENTUM": args.muon_momentum,
            "MUON_BACKEND_STEPS": args.muon_backend_steps,
            "MUON_MOMENTUM_WARMUP_START": args.muon_momentum_warmup_start,
            "MUON_MOMENTUM_WARMUP_STEPS": args.muon_momentum_warmup_steps,
            "BETA1": args.beta1,
            "BETA2": args.beta2,
            "ADAM_EPS": args.adam_eps,
            "GRAD_CLIP_NORM": args.grad_clip_norm,
        },
        "jepa_tuned_overrides": {
            "reference_defaults": {
                key: REFERENCE_DEFAULTS[key]
                for key in (
                    "iterations",
                    "train_batch_tokens",
                    "train_seq_len",
                    "num_layers",
                    "model_dim",
                    "num_heads",
                    "num_kv_heads",
                    "embed_lr",
                    "matrix_lr",
                    "scalar_lr",
                )
            },
            "harness_defaults": {
                "max_steps": JEPA_DEFAULTS["max_steps"],
                "train_batch_tokens": JEPA_DEFAULTS["train_batch_tokens"],
                "train_seq_len": JEPA_DEFAULTS["train_seq_len"],
                "num_layers": JEPA_DEFAULTS["num_layers"],
                "model_dim": JEPA_DEFAULTS["model_dim"],
                "num_heads": JEPA_DEFAULTS["num_heads"],
                "num_kv_heads": JEPA_DEFAULTS["num_kv_heads"],
                "embed_lr": JEPA_DEFAULTS["embed_lr"],
                "matrix_lr": JEPA_DEFAULTS["matrix_lr"],
                "scalar_lr": JEPA_DEFAULTS["scalar_lr"],
            },
        },
        "logged_but_not_applied": {
            "DATA_PATH": "dataset_alias is used instead of raw shard globs",
            "TRAIN_FILES": "dataset_alias resolves cached shards",
            "VAL_FILES": "validation shards are discovered from the cached dataset alias",
            "TOKENIZER_PATH": "tokenization is already baked into the cached dataset alias",
            "VAL_BATCH_SIZE": args.eval_batch_size,
            "VAL_LOSS_EVERY": args.val_loss_every,
            "TIE_EMBEDDINGS": "ignored because the JEPA hybrid uses an untied LM head after routed experts",
            "TIED_EMBED_INIT_STD": "not exposed by the SDK trainer/profile",
        },
    }


def print_resolved_summary(
    args: argparse.Namespace,
    spec,
    trainer_cfg: TorchTrainConfig,
    derived: dict[str, int | bool],
) -> dict[str, Any]:
    summary = {
        "run_id": args.run_id,
        "seed": args.seed,
        "device": args.device,
        "dataset_alias": args.dataset_alias,
        "artifact_path": args.output,
        "graph_contract": ["tokens", "targets", "sem_targets"],
        "model_spec": sanitized_model_spec_dict(
            spec,
            raw_text_encoding_name=str(getattr(args, "raw_text_encoding_name", "") or ""),
        ),
        "trainer": build_trainer_summary(trainer_cfg),
        "derived_schedule": derived,
        "reference_mapping": build_reference_summary(args),
    }
    print("Resolved training configuration:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_progress_logger(
    *,
    train_log_every: int,
    resolved_epochs: int,
    max_steps: int,
) -> tuple[Callable[[dict[str, Any]], None], Callable[[int, float], None]]:
    interval = max(int(train_log_every), 1)
    run_start = time.perf_counter()

    def on_step(info: dict[str, Any]) -> None:
        phase = str(info.get("phase", "train"))
        elapsed = format_elapsed(float(info.get("elapsed_seconds", time.perf_counter() - run_start)))
        routing_suffix = format_routing_stats_suffix(
            info.get("routing_stats"),
            semantic_labels=True,
        )
        actual_grad_accum_steps = int(info.get("actual_grad_accum_steps", info.get("grad_accum_steps", 0)))
        grad_accum_suffix = ""
        if actual_grad_accum_steps and actual_grad_accum_steps != int(info.get("grad_accum_steps", 0)):
            grad_accum_suffix = f" actual_grad_accum_steps={actual_grad_accum_steps}"
        if phase == "warmup":
            log_stage(
                "Warmup step "
                f"{int(info.get('step', 0))}/{int(info.get('warmup_steps', 0))} "
                f"loss={float(info.get('loss', float('nan'))):.6f} "
                f"elapsed={elapsed}{routing_suffix}"
            )
            return

        step = int(info.get("step", 0))
        should_log = (
            step == 1
            or step % interval == 0
            or step >= max_steps
            or int(info.get("epoch_step", 0)) >= int(info.get("steps_per_epoch", 0))
        )
        if not should_log:
            return
        lrs = [float(lr) for lr in info.get("learning_rates", [])]
        lr_preview = ", ".join(f"{lr:.4g}" for lr in lrs[:3])
        if len(lrs) > 3:
            lr_preview += ", ..."
        lr_text = f" lr=[{lr_preview}]" if lr_preview else ""
        log_stage(
            "Train step "
            f"{step}/{int(info.get('max_steps', max_steps))} "
            f"(epoch {int(info.get('epoch', 0))}/{int(info.get('max_epochs', resolved_epochs))}, "
            f"epoch-step {int(info.get('epoch_step', 0))}/{int(info.get('steps_per_epoch', 0))}) "
            f"loss={float(info.get('loss', float('nan'))):.6f} "
            f"elapsed={elapsed}{lr_text}{grad_accum_suffix}{routing_suffix}"
        )

    def on_epoch(epoch_idx: int, loss: float) -> None:
        elapsed = format_elapsed(time.perf_counter() - run_start)
        log_stage(
            f"Epoch {epoch_idx + 1}/{resolved_epochs} complete: avg_loss={float(loss):.6f} elapsed={elapsed}"
        )

    return on_step, on_epoch


def main() -> int:
    configure_console_logging()
    parser = build_parser()
    args = parser.parse_args()
    apply_tinystories_dataset_defaults(args)
    resolve_dataset_selector_args(args)
    resolve_pretraining_file_dataset(args)
    resolve_mode_defaults(args)
    run_label = mode_name(megakernel=bool(args.megakernel))
    interrupted_weights_path = interrupted_output_path(megakernel=bool(args.megakernel))
    interrupted_graph_path = interrupted_weights_path.with_suffix(".json")

    log_stage(f"Starting {run_label} harness run {args.run_id}")
    log_stage(f"CLI started at {datetime.now().isoformat(timespec='seconds')}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    log_stage(f"Seeds set to {args.seed}")

    if args.device != "cuda":
        print("This harness is configured to run on CUDA only.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("CUDA device is not available in this environment.", file=sys.stderr)
        return 1
    log_stage("CUDA is available; resolving datasets and graph configuration")

    log_stage(f"Resolving dataset alias {args.dataset_alias}")
    dataset_name, dataset_path, dataset_meta = resolve_or_download_dataset(
        args.dataset_alias,
        raw_text_encoding_name=str(getattr(args, "raw_text_encoding_name", "gpt2")),
        **dataset_download_kwargs_from_args(args),
    )
    dataset_meta = apply_cached_tokenizer_vocab_policy(
        args,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        dataset_meta=dataset_meta,
        default_vocab_size=int(JEPA_DEFAULTS["vocab_size"]),
    )
    log_stage("Estimating training schedule from cached dataset and semantic rows")
    derived = estimate_schedule(
        dataset_name,
        seq_len=args.train_seq_len,
        batch_size=args.batch_size,
        train_batch_tokens=args.train_batch_tokens,
        top_k=args.top_k,
        template_runtime="megakernel" if args.megakernel else "compile",
        device=args.device,
        all_train_rows=bool(args.all_train_rows),
    )
    (
        derived,
        resolved_epochs,
        resolved_max_steps,
        resolved_lr_decay_iters,
        resolved_max_wallclock_seconds,
    ) = resolve_effective_training_schedule(
        args,
        derived,
    )
    trainer_cfg = build_trainer_config(
        args,
        resolved_epochs=resolved_epochs,
        max_steps=resolved_max_steps,
        lr_decay_iters=resolved_lr_decay_iters,
        max_wallclock_seconds=resolved_max_wallclock_seconds,
        drop_last=bool(derived["drop_last"]),
        respect_epoch_boundaries=bool(derived["respect_epoch_boundaries"]),
    )

    log_stage(f"Building {run_label} graph")
    graph, spec = build_graph(args, dataset_name)
    print(f"Using dataset: {dataset_name}")
    print_graph_summary(graph)
    print_data_source_summary(dataset_name, dataset_path, dataset_meta, graph)
    resolved_training_summary = print_resolved_summary(args, spec, trainer_cfg, derived)

    log_stage("Initializing TorchTrainer")
    trainer = TorchTrainer(graph, trainer_cfg)
    on_step, on_epoch = build_progress_logger(
        train_log_every=args.train_log_every,
        resolved_epochs=resolved_epochs,
        max_steps=resolved_max_steps,
    )
    interrupted = False
    force_abort = False
    previous_sigint = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum, frame):
        nonlocal interrupted, force_abort
        if interrupted:
            force_abort = True
            raise KeyboardInterrupt
        interrupted = True
        log_stage("Interrupt received. Stopping after the current safe boundary.")
        trainer.stop()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        log_stage(
            "Training about to start: "
            f"max_steps={resolved_max_steps}, warmup_steps={args.warmup_steps}, "
            f"train_log_every={args.train_log_every}, grad_accum_steps={derived['grad_accum_steps']}, "
            f"steps_per_epoch={derived['steps_per_epoch']}"
        )
        losses = trainer.train([], [], on_epoch=on_epoch, on_step=on_step)
        output_path = Path(args.output)
        graph_output_path = output_path.with_suffix(".json")

        if interrupted:
            log_stage("Training stopped after interrupt; saving interrupted artifacts")
            save_artifacts(
                graph,
                interrupted_weights_path,
                interrupted_graph_path,
                training_manifest=resolved_training_summary,
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                dataset_meta=dataset_meta,
                raw_text_encoding_name=str(getattr(args, "raw_text_encoding_name", "gpt2")),
            )
            print("Interrupted by user.")
            print("Saved checkpoint:")
            print(f"  weights: {interrupted_weights_path}")
            print(f"  graph:   {interrupted_graph_path}")
            print("Losses:", [round(float(loss), 6) for loss in losses])
            return 130

        if not losses:
            print("Trainer returned no losses.", file=sys.stderr)
            return 1
        if not all(math.isfinite(float(loss)) for loss in losses):
            print("Encountered non-finite loss", file=sys.stderr)
            return 1

        log_stage("Training finished. Saving exported artifacts")
        save_artifacts(
            graph,
            output_path,
            graph_output_path,
            training_manifest=resolved_training_summary,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_meta=dataset_meta,
            raw_text_encoding_name=str(getattr(args, "raw_text_encoding_name", "gpt2")),
        )
        log_stage("Artifacts saved. Starting validation pass")
        val_loss = safe_evaluate_validation_loss(
            lambda: evaluate_model(
                graph,
                dataset_path,
                device=args.device,
                seq_len=args.train_seq_len,
                batch_size=args.eval_batch_size,
                eval_batches=args.eval_batches,
                encoding_name=str(getattr(args, "raw_text_encoding_name", "gpt2")),
            ),
            logger=LOGGER,
        )
        log_stage("Validation finished.")
        print("Losses:", [round(float(loss), 6) for loss in losses])
        print(f"Final train loss: {float(losses[-1]):.6f}")
        if math.isfinite(val_loss):
            print(f"Validation loss: {val_loss:.6f}")
        else:
            print("Validation loss: skipped")
        print(f"Exported model: {output_path}")
        print(f"Exported graph: {graph_output_path}")
        print("Training completed successfully.")
        log_stage("Run completed successfully")
        return 0
    except KeyboardInterrupt:
        trainer.stop()
        log_stage("Saving interrupted artifacts after keyboard interrupt")
        save_artifacts(
            graph,
            interrupted_weights_path,
            interrupted_graph_path,
            training_manifest=resolved_training_summary,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_meta=dataset_meta,
            raw_text_encoding_name=str(getattr(args, "raw_text_encoding_name", "gpt2")),
        )
        if force_abort:
            print("\nForced abort received.")
        else:
            print("\nInterrupted by user.")
        print("Saved checkpoint:")
        print(f"  weights: {interrupted_weights_path}")
        print(f"  graph:   {interrupted_graph_path}")
        return 130
    finally:
        signal.signal(signal.SIGINT, previous_sigint)


if __name__ == "__main__":
    raise SystemExit(main())
