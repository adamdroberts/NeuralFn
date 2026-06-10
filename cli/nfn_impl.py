from __future__ import annotations

import argparse
import curses
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import sys
import termios
import time
import tty as tty_module
from types import SimpleNamespace
import uuid
from typing import Any, Callable, Sequence

import numpy as np
import torch

try:
    from rich.align import Align
    from rich.box import HEAVY, ROUNDED
    from rich.cells import cell_len
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.markup import escape as _rich_escape
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.table import Table
    from rich.text import Text
    from rich.theme import Theme
except ImportError as exc:
    raise SystemExit(
        "The 'rich' package is required for nfn infer. "
        "Reinstall with: pip install -e . (or pip install 'rich>=13')"
    ) from exc

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
NEURALFN_ROOT = ROOT.parent
for candidate in (SCRIPTS_DIR, NEURALFN_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from cli_utils import artifact_root, create_argument_parser
from eval_llama_fast import evaluate_validation_loss, resolve_prompt_suite
from infer_jepa_semantic import (
    autocast_enabled_for,
    artifact_metadata_for_graph,
    build_semantic_model_inputs,
    dataset_download_kwargs_from_args,
    decode_tokens,
    default_inference_graph_artifact,
    default_inference_weights_artifact,
    describe_token,
    describe_routing,
    find_logits_trace_key,
    graph_uses_semantic_router_vecs,
    infer_graph_template_runtime,
    load_compiled_inference_graph,
    log_tokenizer_status,
    repetition_penalty_arg,
    resolve_autocast_settings,
    resolve_inference_dataset_alias,
    resolve_inference_tokenizer_context,
    resolve_prompt_tokens,
    resolve_prompt_text,
    resolve_raw_text_encoding_name,
    resolve_graph_weights_path,
    resolve_semantic_router_vecs,
    resolve_semantic_targets,
    sample_next_token,
    tokenizer_manifest_for_graph,
    top_p_arg,
)
from infer_llama_fast import generate_sequence
from neuralfn import TorchTrainer, load_graph
from neuralfn.config import build_composed_lm_spec, model_spec_to_dict
from neuralfn.semantic import ConversationalVocabulary, semantic_vocab_ref_for_graph, semantic_vocab_ref_for_tokenizer
from neuralfn.torch_backend import TorchTrainConfig
from neuralfn.torch_templates import build_gpt_root_graph
from parameter_golf_runtime import (
    PARAMETER_GOLF_CHECKPOINT_FORMAT,
    build_parameter_golf_model,
    checkpoint_metadata_path,
    infer_config_from_state_dict,
    is_parameter_golf_flat_state_dict,
    load_checkpoint_metadata,
    load_parameter_golf_state_dict,
    load_sentencepiece_tokenizer,
    load_training_log_hparams,
    resolve_parameter_golf_tokenizer_path,
)
from server.dataset_manager import normalize_raw_text_encoding_name, raw_text_encoding_name_for_backbone, raw_text_encoding_vocab_size
from server.models import LoadDatasetRequest
from server.services.graph_ops import load_dataset_source_into_graph
from train_jepa_semantic import (
    DEFAULT_CACHED_TOKENIZER_VARIANT,
    DEFAULT_DATASET_ALIAS,
    DATASET_SHORTCUT_CONTRACTS,
    REFERENCE_DEFAULTS,
    SUPPORTED_CACHED_TOKENIZER_VARIANTS,
    SUPPORTED_TOKENIZER_CHOICES,
    TINYSTORIES_DATASET_CONTRACT,
    apply_sanitized_template_spec,
    add_pretraining_file_argument,
    add_dataset_download_arguments,
    add_dataset_selector_arguments,
    add_raw_text_tokenizer_arguments,
    apply_cached_tokenizer_vocab_policy,
    apply_tinystories_dataset_defaults,
    build_trainer_summary,
    default_tokenizer_for_dataset,
    estimate_schedule,
    estimate_text_schedule,
    env_optional_float,
    env_optional_int,
    evaluate_model as evaluate_semantic_model,
    format_elapsed,
    print_graph_summary,
    raw_text_tokenizer_is_available,
    resolve_dataset_shortcut_contract,
    resolve_dataset_selector_args,
    resolve_effective_training_schedule,
    resolve_lr_schedule_defaults,
    resolve_or_download_dataset,
    resolve_pretraining_file_dataset,
    safe_evaluate_validation_loss,
    sanitized_model_spec_dict,
    save_artifacts,
    tokenizer_download_kwargs_from_args,
    validate_raw_text_tokenizer_availability,
)
from train_llama_fast import (
    LLAMA_DEFAULTS,
    build_progress_logger,
    configure_console_logging,
    env_float,
    env_int,
    env_str,
    evaluate_model as evaluate_text_model,
)
from train_mixllama_fast import MIXLLAMA_DEFAULTS
from train_nanogpt import NANOGPT_DEFAULTS
from train_gpt2 import GPT2_DEFAULTS
from train_semantic_router_moe import ROUTER_DEFAULTS

HELP_STYLES = ("short", "long", "verbose")
COMMANDS = ("train", "infer", "eval", "kernels")
BASE_MODELS = ("llama", "gpt2", "nanogpt")
TOPOLOGIES = ("dense", "moe")
ROUTER_MODES = ("standard", "semantic")
DATASET_CHOICES = tuple(DATASET_SHORTCUT_CONTRACTS)
PRETRAINING_FILE_DATASET = "pretraining_file"
CACHED_TOKEN_DATASETS = frozenset({"golf1", "golf10"})
RAW_TEXT_DATASETS = frozenset({"shakespear", "shakespeare", "tinystories", PRETRAINING_FILE_DATASET})
SMALL_RAW_TEXT_DATASETS = frozenset({"shakespear", "shakespeare"})

RUN_PRESET_VALUES = {
    "smoke": {
        "max_steps": 40,
        "train_seq_len": 128,
        "batch_size": 4,
        "train_batch_tokens": 4_096,
        "eval_batches": 2,
        "eval_batch_size": 4,
        "train_log_every": 1,
        "max_wallclock_seconds": 120.0,
        "warmup_steps": 2,
        "warmdown_fraction": 0.75,
    },
    "default": {
        "max_steps": 400,
        "train_seq_len": 192,
        "batch_size": 8,
        "train_batch_tokens": 24_576,
        "eval_batches": 8,
        "eval_batch_size": 8,
        "train_log_every": 1,
        "max_wallclock_seconds": 900.0,
        "warmup_steps": 8,
        "warmdown_fraction": 0.75,
    },
    "overnight": {
        "max_steps": 4_000,
        "train_seq_len": 256,
        "batch_size": 8,
        "train_batch_tokens": 98_304,
        "eval_batches": 16,
        "eval_batch_size": 8,
        "train_log_every": 5,
        "max_wallclock_seconds": 28_800.0,
        "warmup_steps": 32,
        "warmdown_fraction": 0.75,
    },
    "parameter_golf_10min": {
        "max_steps": 2_500,
        "train_seq_len": 2_048,
        "batch_size": 4,
        "train_batch_tokens": 786_432,
        "eval_batches": 8,
        "eval_batch_size": 4,
        "train_log_every": 500,
        "max_wallclock_seconds": 600.0,
        "warmup_steps": 20,
        "warmdown_fraction": 0.75,
    },
}

OPTIMIZER_PRESET_VALUES = {
    "gradient_default": {
        "evolutionary": False,
        "optimizer_profile": "parameter_golf",
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
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
    },
    "evolutionary_lean": {
        "evolutionary": True,
        "evo_population_size": 24,
        "evo_mutation_rate": 0.08,
        "evo_mutation_scale": 0.2,
        "evo_crossover_rate": 0.4,
        "evo_tournament_size": 3,
        "evo_elite_count": 2,
    },
    "evolutionary_balanced": {
        "evolutionary": True,
        "evo_population_size": 50,
        "evo_mutation_rate": 0.1,
        "evo_mutation_scale": 0.3,
        "evo_crossover_rate": 0.5,
        "evo_tournament_size": 3,
        "evo_elite_count": 2,
    },
    "evolutionary_broad": {
        "evolutionary": True,
        "evo_population_size": 96,
        "evo_mutation_rate": 0.12,
        "evo_mutation_scale": 0.35,
        "evo_crossover_rate": 0.6,
        "evo_tournament_size": 5,
        "evo_elite_count": 4,
    },
    "parameter_golf_muon": {
        "evolutionary": False,
        "optimizer_profile": "parameter_golf",
        "learning_rate": 3e-4,
        "weight_decay": 0.02,
        "embed_lr": 0.6,
        "head_lr": 0.008,
        "tied_embed_lr": 0.03,
        "matrix_lr": 0.026,
        "scalar_lr": 0.02,
        "muon_momentum": 0.97,
        "muon_backend_steps": 5,
        "muon_momentum_warmup_start": 0.92,
        "muon_momentum_warmup_steps": 1_500,
        "beta1": 0.9,
        "beta2": 0.99,
        "adam_eps": 1e-8,
        "grad_clip_norm": 0.3,
    },
}

PARAMETER_GOLF_CASEOPS_MODEL_PRESET = "parameter_golf_caseops_8192"
PARAMETER_GOLF_CASEOPS_MODEL_VALUES = {
    "vocab_size": 8_192,
    "num_layers": 11,
    "model_dim": 512,
    "num_heads": 8,
    "num_kv_heads": 4,
    "mlp_mult": 4.0,
    "rope_base": 10_000.0,
    "qk_gain_init": 5.25,
    "logit_softcap": 30.0,
}

INFER_PROMPT_PRESETS = {
    "hello": {"prompt": "hello"},
    "story": {"prompt": "Once upon a time"},
    "code": {"prompt": "def hello_world():"},
}

INFER_GENERATION_PRESETS = {
    "balanced": {"max_new_tokens": 64, "temperature": 0.8, "top_k": 32, "top_p": 1.0, "repetition_penalty": 1.08},
    "focused": {"max_new_tokens": 64, "temperature": 0.4, "top_k": 16, "top_p": 0.9, "repetition_penalty": 1.12},
    "explore": {"max_new_tokens": 128, "temperature": 0.95, "top_k": 64, "top_p": 0.95, "repetition_penalty": 1.05},
}

INFER_CHAT_MODES = ("stateless", "transcript")
INFER_RECIPE_KEYS = ("base_model", "topology", "router_mode", "use_jepa", "megakernel")
DEFAULT_INFER_TOP_P = 1.0
DEFAULT_PARAMETER_GOLF_NO_REPEAT_NGRAM_SIZE = 4
DEFAULT_PARAMETER_GOLF_REPEAT_RUN_LIMIT = 3

EVAL_PRESET_VALUES = {
    "smoke": {"eval_batches": 2, "eval_batch_size": 4, "max_new_tokens": 32},
    "default": {"eval_batches": 8, "eval_batch_size": 8, "max_new_tokens": 64},
    "extended": {"eval_batches": 16, "eval_batch_size": 8, "max_new_tokens": 96},
}


@dataclass(frozen=True)
class OptionChoice:
    label: str
    description: str
    value: Any
    recommended: bool = False
    custom_prompt: str | None = None
    parser: Callable[[str], Any] | None = None


@dataclass(frozen=True)
class Question:
    key: str
    prompt: str
    options_factory: Callable[[dict[str, Any]], list[OptionChoice]]
    visible: Callable[[dict[str, Any], set[str]], bool]


@dataclass(frozen=True)
class ComposedRecipe:
    base_model: str
    topology: str
    router_mode: str
    use_jepa: bool
    runtime: str
    training_mode: str = "pretrain"  # "pretrain" | "sft" | "dpo" | "ppo" | "reward_model"
    adapter_type: str = "none"       # "none" | "lora" | "qlora" | "randmap"

    @property
    def uses_semantic(self) -> bool:
        return self.topology == "moe" and self.router_mode == "semantic"

    @property
    def template_runtime(self) -> str:
        if self.runtime == "megakernel":
            return "megakernel"
        return "compile" if self.base_model == "llama" else "eager"

    def mode_name(self) -> str:
        legacy = self.legacy_mode_name()
        if legacy is not None:
            return legacy
        topology_name = "dense"
        if self.topology == "moe":
            topology_name = "semantic_moe" if self.router_mode == "semantic" else "moe"
        mode = f"{self.base_model}_{topology_name}"
        if self.use_jepa:
            mode += "_jepa"
        if self.runtime == "megakernel":
            mode += "_megakernel"
        return mode

    def legacy_mode_name(self) -> str | None:
        if self.base_model == "llama" and self.topology == "dense" and not self.use_jepa:
            return "llama_fast_megakernel" if self.runtime == "megakernel" else "llama_fast"
        if self.base_model == "llama" and self.topology == "moe" and self.router_mode == "standard" and not self.use_jepa:
            return "mixllama_fast_megakernel" if self.runtime == "megakernel" else "mixllama_fast"
        if self.base_model == "llama" and self.topology == "moe" and self.router_mode == "semantic" and not self.use_jepa:
            return "semantic_router_moe_megakernel" if self.runtime == "megakernel" else "semantic_router_moe"
        if self.base_model == "llama" and self.topology == "moe" and self.router_mode == "semantic" and self.use_jepa:
            return "jepa_semantic_hybrid_megakernel" if self.runtime == "megakernel" else "jepa_semantic_hybrid"
        if self.base_model == "gpt2" and self.topology == "dense" and not self.use_jepa:
            return "gpt2_megakernel" if self.runtime == "megakernel" else "gpt2"
        if self.base_model == "nanogpt" and self.topology == "dense" and not self.use_jepa:
            return "nanogpt_megakernel" if self.runtime == "megakernel" else "nanogpt"
        return None

    def graph_name(self) -> str:
        return f"{self.mode_name()}_sdk"


@dataclass
class InferChatSettings:
    top_k: int
    top_p: float
    temperature: float
    max_new_tokens: int
    repetition_penalty: float = 1.0
    autocomplete_words: int = 0


@dataclass
class InferRuntimeContext:
    args: argparse.Namespace
    graph_path: Path
    resolved_weights_path: Path
    graph: Any
    compiled: Any
    state_dict: dict[str, torch.Tensor]
    tokenizer: Any
    tokenizer_path: Path | None
    tokenizer_name: str | None
    raw_text_encoding_name: str
    dataset_alias: str
    device: torch.device
    generator: torch.Generator
    amp_dtype: torch.dtype
    amp_name: str
    context_window: int
    generation_backend: str = "graph"


@dataclass(frozen=True)
class InferAutocompletePreview:
    token_id: int
    token_text: str
    display_text: str
    insertable: bool
    buffer_snapshot: str
    mode: str
    settings_signature: tuple[int, float, float, int, float, int]


@dataclass(frozen=True)
class InferInlineAutocomplete:
    text: str
    display_text: str
    insertable: bool
    buffer_snapshot: str
    mode: str
    settings_signature: tuple[int, float, float, int, float, int]


HELP_COPY: dict[str, tuple[str, str, str]] = {
    "root": (
        "Master NeuralFn CLI.",
        "Master NeuralFn CLI for train, infer, and eval.",
        "Master NeuralFn CLI for train, infer, and eval. Build recipes by choosing a base model, topology, router mode, JEPA, and runtime.",
    ),
    "plan": (
        "Open planner.",
        "Open the interactive planner.",
        "Open the interactive planner. In a TTY, use left/right arrows to move between questions and up/down to change the current choice.",
    ),
    "plan_auto": (
        "Auto-plan.",
        "Fill omitted planner-managed options with recommendations.",
        "Fill omitted planner-managed options with the same recommendation engine used by the interactive planner, print the resolved command, and run it.",
    ),
    "help_style": (
        "Help style.",
        "Help detail level.",
        "Help detail level. Short is compact, long is the default prose view, and verbose adds compatibility notes and preset guidance.",
    ),
    "base_model": (
        "Base model.",
        "Base model to start from.",
        "Base model to start from. The planner always begins here before it asks about topology, router mode, JEPA, or megakernel.",
    ),
    "topology": (
        "Dense or MoE.",
        "Top-level topology selection.",
        "Top-level topology selection. Choose dense or MoE, then select standard vs semantic routing inside the MoE flow.",
    ),
    "router_mode": (
        "MoE router.",
        "MoE router mode.",
        "MoE router mode. Standard MoE uses learned expert gating; semantic router ties the route choice to semantic targets or semantic-router vectors.",
    ),
    "jepa": (
        "Add JEPA.",
        "Enable the additive JEPA objective.",
        "Enable the additive JEPA objective. Dense and standard-MoE recipes use a shared-backbone AR+JEPA path; semantic-MoE recipes add JEPA on top of the semantic-router path.",
    ),
    "megakernel": (
        "Megakernel runtime.",
        "Enable the megakernel runtime variant.",
        "Enable the megakernel runtime variant. This is orthogonal to the base model and recipe composition.",
    ),
    "dataset": (
        "Dataset shortcut.",
        "Dataset shortcut to use.",
        "Dataset shortcut to use. `golf1` and `golf10` are cached-token datasets; `shakespeare` and `tinystories` are raw-text datasets that also trigger tokenizer selection.",
    ),
    "pretraining_file": (
        "Local corpus.",
        "Local .txt corpus to train on directly.",
        "Local .txt corpus to train on directly without downloading or importing a named dataset. This hides the dataset question but still keeps tokenizer selection active.",
    ),
    "model_preset": (
        "Model preset.",
        "Recommended model hyperparameter preset.",
        "Recommended model hyperparameter preset. Start with a sensible recipe-specific shape, then optionally open the advanced per-parameter questions.",
    ),
    "run_preset": (
        "Run preset.",
        "Training run preset.",
        "Training run preset. Smoke gives a fast sanity check, default matches the harness baseline, and overnight stretches both time and token budget.",
    ),
    "optimizer_preset": (
        "Search preset.",
        "Optimizer or evolutionary-search preset.",
        "Optimizer or evolutionary-search preset. Gradient uses the current parameter-golf baseline; the evolutionary presets switch the trainer into generation-based search with different population sizes.",
    ),
}


def help_text(key: str, style: str) -> str:
    short, long, verbose = HELP_COPY[key]
    if style == "short":
        return short
    if style == "verbose":
        return verbose
    return long


def recipe_from_state(state: dict[str, Any]) -> ComposedRecipe:
    base_model = str(state.get("base_model") or state.get("model") or "llama").strip().lower()
    raw_topology = str(state.get("topology") or "dense").strip().lower()
    raw_router_mode = str(state.get("router_mode") or "none").strip().lower()
    if raw_topology == "semantic_router":
        raw_topology = "moe"
        raw_router_mode = "semantic"
    if raw_topology == "dense":
        raw_router_mode = "none"
    elif raw_router_mode == "none":
        raw_router_mode = "standard"
    runtime = "megakernel" if bool(state.get("megakernel")) or str(state.get("runtime") or "") == "megakernel" else "default"
    training_mode = str(state.get("training_mode") or "pretrain").strip().lower()
    if training_mode not in {"pretrain", "sft", "dpo", "ppo", "reward_model"}:
        training_mode = "pretrain"
    adapter_type = str(state.get("adapter_type") or "none").strip().lower()
    if adapter_type not in {"none", "lora", "qlora", "randmap"}:
        adapter_type = "none"
    return ComposedRecipe(
        base_model=base_model,
        topology=raw_topology,
        router_mode=raw_router_mode,
        use_jepa=bool(state.get("use_jepa") or state.get("jepa")),
        runtime=runtime,
        training_mode=training_mode,
        adapter_type=adapter_type,
    )


def dataset_choice_from_state(state: dict[str, Any]) -> str:
    if state.get("tinystories"):
        return "tinystories"
    if state.get("pretraining_file"):
        return PRETRAINING_FILE_DATASET
    return str(state.get("dataset") or "golf1")


DATASET_SELECTOR_DERIVED_KEYS = frozenset(
    {
        "dataset_alias",
        "dataset_hf_path",
        "dataset_variant",
        "dataset_train_shards",
        "dataset_repo_id",
        "dataset_remote_root_prefix",
        "dataset_train_file",
        "dataset_val_file",
    }
)


def normalize_dataset_selector_state(state: dict[str, Any]) -> dict[str, Any]:
    shortcut = state.get("dataset")
    if shortcut:
        shortcut_name = str(shortcut)
        state["tinystories"] = shortcut_name == "tinystories"
    elif state.get("tinystories"):
        shortcut_name = "tinystories"
        state["dataset"] = shortcut_name
    else:
        return state

    explicit_values: dict[str, str] = {}
    for key in ("tokenizer", "raw_text_encoding_override", "dataset_variant"):
        normalized = normalize_raw_text_encoding_name(state.get(key))
        if normalized is not None:
            explicit_values[key] = normalized
    unique_tokenizers = {value for value in explicit_values.values() if value}
    if len(unique_tokenizers) > 1:
        joined = ", ".join(f"{key}={value}" for key, value in sorted(explicit_values.items()))
        raise ValueError(f"Conflicting tokenizer selections: {joined}")
    chosen_tokenizer = next(iter(unique_tokenizers), None)
    if chosen_tokenizer is not None:
        state["tokenizer"] = chosen_tokenizer
        state["raw_text_encoding_override"] = chosen_tokenizer
        if shortcut_name in CACHED_TOKEN_DATASETS:
            state["dataset_variant"] = chosen_tokenizer
        else:
            state.pop("dataset_variant", None)

    contract = resolve_dataset_shortcut_contract(
        shortcut_name,
        dataset_variant=state.get("dataset_variant"),
    )
    if contract is None:
        return state
    for key in DATASET_SELECTOR_DERIVED_KEYS:
        if key not in contract:
            state.pop(key, None)
    for key, value in contract.items():
        state[key] = value
    if shortcut_name in CACHED_TOKEN_DATASETS:
        state["tokenizer"] = str(state.get("dataset_variant") or DEFAULT_CACHED_TOKENIZER_VARIANT)
        state["raw_text_encoding_override"] = state["tokenizer"]
    elif chosen_tokenizer is not None:
        state["tokenizer"] = chosen_tokenizer
        state["raw_text_encoding_override"] = chosen_tokenizer
    return state


def uses_raw_text_dataset(state: dict[str, Any]) -> bool:
    return dataset_choice_from_state(state) in RAW_TEXT_DATASETS


def selected_tokenizer_name(recipe: ComposedRecipe, state: dict[str, Any]) -> str:
    normalize_dataset_selector_state(state)
    tokenizer_name = normalize_raw_text_encoding_name(
        state.get("tokenizer") or state.get("raw_text_encoding_override") or state.get("dataset_variant")
    )
    if tokenizer_name is not None:
        return tokenizer_name
    dataset = dataset_choice_from_state(state)
    dataset_default = default_tokenizer_for_dataset(dataset)
    if dataset_default is not None:
        return dataset_default
    return raw_text_encoding_name_for_backbone(
        recipe.base_model,
        prefer_cl100k=dataset in SMALL_RAW_TEXT_DATASETS,
    )


def tokenizer_choices(recipe: ComposedRecipe, state: dict[str, Any]) -> list[OptionChoice]:
    dataset = dataset_choice_from_state(state)
    default_name = selected_tokenizer_name(recipe, state)
    if dataset in CACHED_TOKEN_DATASETS:
        train_shards = 10 if dataset == "golf10" else 1
        choices: list[OptionChoice] = []
        for variant in SUPPORTED_CACHED_TOKENIZER_VARIANTS:
            vocab_size = int(variant.removeprefix("sp"))
            shard_text = "shards" if train_shards != 1 else "shard"
            choices.append(
                OptionChoice(
                    variant,
                    f"Use parameter-golf cached tokens with sentencepiece vocab size {vocab_size} and {train_shards} training {shard_text}.",
                    value={"tokenizer": variant},
                    recommended=variant == default_name,
                )
            )
        return choices
    if not uses_raw_text_dataset(state):
        return []
    choice_specs = (
        ("GPT-2 tokenizer", "gpt2", "Original OpenAI byte-level BPE tokenizer (~50k vocab)."),
        ("cl100k tokenizer", "cl100k_base", "Use cl100k_base for raw-text encoding."),
        ("o200k tokenizer", "o200k_base", "Use o200k_base for raw-text encoding."),
        ("sp1024 tokenizer", "sp1024", "Use the shared sp1024 sentencepiece model for raw-text encoding."),
        ("sp2048 tokenizer", "sp2048", "Use the shared sp2048 sentencepiece model for raw-text encoding."),
        ("sp4096 tokenizer", "sp4096", "Use the shared sp4096 sentencepiece model for raw-text encoding."),
        ("sp8192 tokenizer", "sp8192", "Use the shared sp8192 sentencepiece model for raw-text encoding."),
    )
    choices: list[OptionChoice] = []
    for label, tokenizer_name, description in choice_specs:
        available = raw_text_tokenizer_is_available(tokenizer_name)
        if str(tokenizer_name).startswith("sp") and not available:
            description = f"{description} Downloads shared tokenizer assets before training if missing."
        choices.append(
            OptionChoice(
                label=label,
                description=description,
                value={"tokenizer": tokenizer_name},
                recommended=tokenizer_name == default_name,
            )
        )
    return choices


def command_description(command: str, style: str) -> str:
    if command == "train":
        base = "Train a composed NeuralFn recipe."
        if style == "verbose":
            return f"{base} Choose a base model first, then topology, router mode, JEPA, megakernel, and preset stacks."
        return base
    if command == "infer":
        base = "Run generation from an exported graph artifact or supported graphless checkpoint."
        if style == "verbose":
            return (
                f"{base} Pass --graph for NeuralFn exports, --checkpoint for flat Parameter Golf .pt files, "
                "or pick a local graph artifact interactively and chat with it."
            )
        return base
    if command == "kernels":
        base = "Inspect NeuralFn CUDA Tile kernel coverage and runtime diagnostics."
        if style == "verbose":
            return f"{base} Use 'list' to print registry coverage or 'doctor' to inspect the local CUDA Tile toolchain."
        return base
    base = "Evaluate a composed NeuralFn recipe."
    if style == "verbose":
        return f"{base} Validation and prompt probes use the same composed recipe flow as training and inference."
    return base


def command_epilog(command: str, style: str) -> str:
    if style == "short":
        return "Use --plan for guided setup."
    if command == "train":
        return (
            "Examples:\n"
            "  nfn train --plan\n"
            "  nfn train --pretraining-file ./pretraining-data.txt\n"
            "  nfn train --base-model nanogpt --topology moe --router-mode semantic --jepa --megakernel\n"
            "  nfn train --base-model gpt2 --model-preset harness_default --run-preset overnight"
        )
    if command == "infer":
        return (
            "Examples:\n"
            "  nfn infer --graph ~/NeuralFn/artifacts/llama_fast.json\n"
            "  nfn infer --graph ~/NeuralFn/artifacts/llama_fast.json --prompt \"Once upon a time\"\n"
            "  nfn infer --graph ~/NeuralFn/artifacts/semantic_router_moe.json --weights ~/NeuralFn/artifacts/semantic_router_moe.pt\n"
            "  nfn infer --checkpoint ~/NeuralFn/artifacts/final_model.pt "
            "--checkpoint-tokenizer ~/Downloads/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
        )
    if command == "kernels":
        return (
            "Examples:\n"
            "  nfn kernels list\n"
            "  nfn kernels list --json\n"
            "  nfn kernels doctor --json"
        )
    return (
        "Examples:\n"
        "  nfn eval --plan\n"
        "  nfn eval --base-model gpt2\n"
        "  nfn eval --base-model nanogpt --topology moe --router-mode semantic --jepa --megakernel"
    )


def build_root_parser(style: str) -> argparse.ArgumentParser:
    parser = create_argument_parser(
        prog="nfn",
        add_help=False,
        description=help_text("root", style),
    )
    parser.add_argument("-h", "--help", action="store_true", help="Show help for the master CLI.")
    parser.add_argument("--help-style", choices=HELP_STYLES, default=None, help=help_text("help_style", style))
    return parser


def add_shared_control_arguments(parser: argparse.ArgumentParser, style: str) -> None:
    parser.add_argument("-h", "--help", action="store_true", help="Show help for this command.")
    parser.add_argument("--help-style", choices=HELP_STYLES, default=None, help=help_text("help_style", style))
    parser.add_argument("--plan", action="store_true", help=help_text("plan", style))
    parser.add_argument("--plan-auto", action="store_true", help=help_text("plan_auto", style))


def add_recipe_arguments(parser: argparse.ArgumentParser, style: str) -> None:
    group = parser.add_argument_group("Recipe")
    group.add_argument("--base-model", "--model", dest="base_model", choices=BASE_MODELS, default=None, help=help_text("base_model", style))
    group.add_argument("--topology", choices=("dense", "moe", "semantic_router"), default=None, help=help_text("topology", style))
    group.add_argument("--router-mode", choices=ROUTER_MODES, default=None, help=help_text("router_mode", style))
    group.add_argument("--jepa", action="store_true", dest="use_jepa", help=help_text("jepa", style))
    group.add_argument("--megakernel", action="store_true", help=help_text("megakernel", style))
    # ── Fine-tuning flags (Phase 1-4) ──────────────────────────────────
    ft_group = parser.add_argument_group("Fine-tuning")
    ft_group.add_argument(
        "--training-mode",
        choices=("pretrain", "sft", "dpo", "ppo", "reward_model"),
        default=None,
        help="Training objective. 'pretrain' (default) matches today's behaviour. "
             "'sft' runs supervised fine-tuning from --base-checkpoint. "
             "'dpo' / 'ppo' use --ref-checkpoint (and --reward-checkpoint for ppo). "
             "'reward_model' trains a reward head on preference pairs.",
    )
    ft_group.add_argument(
        "--adapter-type",
        choices=("none", "lora", "qlora", "randmap"),
        default=None,
        help="Parameter-efficient adapter. 'lora' = trainable rank-r delta on frozen base; "
             "'qlora' = nf4 base + LoRA delta; 'randmap' = frozen random projections w/ trainable middle; 'none' = full fine-tune.",
    )
    ft_group.add_argument("--lora-rank", type=int, default=None, help="LoRA rank r (default 8).")
    ft_group.add_argument("--lora-alpha", type=float, default=None, help="LoRA alpha scaling (default 16).")
    ft_group.add_argument("--lora-dropout", type=float, default=None, help="LoRA dropout on the input (default 0).")
    ft_group.add_argument("--lora-targets", default=None, help="Comma-separated projection names to adapt, e.g. 'q_proj,v_proj'.")
    ft_group.add_argument("--lora-bias", action="store_true", help="Enable bias in LoRA-wrapped linears.")
    ft_group.add_argument("--qlora-group-size", type=int, default=None, help="qLoRA nf4 group size along input dim (default 64).")
    ft_group.add_argument("--qlora-compute-dtype", default=None, help="qLoRA dequantize compute dtype: bf16 (default), fp16, fp32.")
    ft_group.add_argument("--base-checkpoint", default=None, help="Pretrained checkpoint .pt path used as frozen base.")
    ft_group.add_argument("--ref-checkpoint", default=None, help="Frozen reference model .pt for DPO/PPO.")
    ft_group.add_argument("--reward-checkpoint", default=None, help="Frozen reward model .pt for PPO.")
    ft_group.add_argument("--adapter-only-save", action="store_true", help="After training, save only the adapter/head parameters (small artifact).")
    ft_group.add_argument("--dpo-beta", type=float, default=None, help="DPO beta (reward temperature). Typical 0.1–0.5.")
    ft_group.add_argument("--dpo-loss-type", choices=("sigmoid", "hinge", "ipo"), default=None, help="DPO loss variant.")
    ft_group.add_argument("--kl-coef", type=float, default=None, help="PPO KL-to-ref coefficient (default 0.1).")
    ft_group.add_argument("--ppo-clip", type=float, default=None, help="PPO clip range (default 0.2).")
    ft_group.add_argument("--ppo-vf-coef", type=float, default=None, help="PPO value-function loss coefficient.")
    ft_group.add_argument("--ppo-ent-coef", type=float, default=None, help="PPO entropy bonus coefficient.")
    ft_group.add_argument("--rollout-length", type=int, default=None, help="PPO rollout length in tokens (default 64).")
    ft_group.add_argument("--ppo-epochs-per-rollout", type=int, default=None, help="PPO inner-loop epochs per rollout (default 4).")
    group.add_argument(
        "--model-preset",
        choices=("harness_default", "compact", "jepa_tuned", "jepa_reference", PARAMETER_GOLF_CASEOPS_MODEL_PRESET),
        default=None,
        help=help_text("model_preset", style),
    )
    group.add_argument(
        "--run-preset",
        choices=tuple(RUN_PRESET_VALUES),
        default=None,
        help=help_text("run_preset", style),
    )
    group.add_argument(
        "--optimizer-preset",
        choices=tuple(OPTIMIZER_PRESET_VALUES),
        default=None,
        help=help_text("optimizer_preset", style),
    )


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    add_dataset_selector_arguments(parser, default_alias=DEFAULT_DATASET_ALIAS)
    add_dataset_download_arguments(parser)
    add_raw_text_tokenizer_arguments(parser)


def build_command_parser(command: str, style: str) -> argparse.ArgumentParser:
    parser = create_argument_parser(
        prog=f"nfn {command}",
        add_help=False,
        description=command_description(command, style),
        epilog=command_epilog(command, style),
    )
    if command == "kernels":
        parser.add_argument("-h", "--help", action="store_true", help="Show help for this command.")
        parser.add_argument("--help-style", choices=HELP_STYLES, default=None, help=help_text("help_style", style))
        parser.add_argument("kernel_action", choices=("list", "doctor", "bench", "examples"), nargs="?", default="list")
        parser.add_argument("--json", action="store_true", dest="json_output", help="Print machine-readable JSON.")
        parser.add_argument("--iterations", type=int, default=200, help="Benchmark iterations for 'kernels bench'.")
        parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations for 'kernels bench'.")
        parser.add_argument("--device", default="auto", help="Benchmark device: auto, cpu, cuda, or cuda:N.")
        parser.add_argument("--output-dir", default=None, help="Directory for 'kernels examples --write'.")
        parser.add_argument("--write", action="store_true", help="Write CUDA Tile example files for 'kernels examples'.")
        return parser
    add_shared_control_arguments(parser, style)
    add_recipe_arguments(parser, style)
    parser.add_argument("--run-id", default=None, help="Optional run identifier for training.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--device", default="cuda", help="Torch device. The harness expects CUDA in practice.")
    parser.add_argument("--amp-dtype", choices=("float32", "bfloat16", "float16"), default=None, help="Autocast dtype. Defaults to float32.")
    parser.add_argument(
        "--kernel-backend",
        choices=("auto", "torch", "tile-cuda"),
        default=None,
        help="Kernel backend selector. auto keeps PyTorch fallback; tile-cuda requests CUDA Tile fast paths.",
    )
    parser.add_argument("--tile-cuda-strict", action="store_true", help="Fail instead of falling back when requested CUDA Tile coverage is missing.")
    parser.add_argument("--tile-cuda-report", default=None, help="Optional CUDA Tile diagnostics and coverage JSON report path.")
    add_dataset_arguments(parser)
    if command == "train":
        add_pretraining_file_argument(parser, help_text=help_text("pretraining_file", style))
        parser.add_argument("--output", default=None, help="Weights artifact path.")
        parser.add_argument("--max-steps", type=int, default=None, help="Maximum number of optimizer steps or evolutionary generations.")
        parser.add_argument("--train-seq-len", type=int, default=None, help="Training sequence length.")
        parser.add_argument("--batch-size", type=int, default=None, help="Per-loader batch size.")
        parser.add_argument("--train-batch-tokens", type=int, default=None, help="Training token budget per step.")
        parser.add_argument("--eval-batches", type=int, default=None, help="Validation batches to score after training.")
        parser.add_argument("--eval-batch-size", type=int, default=None, help="Validation batch size.")
        parser.add_argument("--train-log-every", type=int, default=None, help="Training progress logging interval.")
        parser.add_argument("--max-wallclock-seconds", type=float, default=None, help="Optional wallclock budget.")
        parser.add_argument("--warmup-steps", type=int, default=None, help="Warmup steps.")
        parser.add_argument(
            "--warmdown-fraction",
            type=float,
            default=None,
            help="Fraction of optimizer steps reserved for linear tail warmdown.",
        )
        parser.add_argument("--all-train-rows", action="store_true", help="Use every train row each epoch and respect epoch boundaries.")
        parser.add_argument("--val-loss-every", type=int, default=None, help="Reserved validation-loss interval setting.")
        parser.add_argument("--vocab-size", type=int, default=None, help="Vocabulary size.")
        parser.add_argument("--num-layers", type=int, default=None, help="Transformer layer count.")
        parser.add_argument("--model-dim", type=int, default=None, help="Model width.")
        parser.add_argument("--num-heads", type=int, default=None, help="Attention head count.")
        parser.add_argument("--num-kv-heads", type=int, default=None, help="Grouped-query or multi-query KV head count.")
        parser.add_argument("--mlp-mult", type=float, default=None, help="MLP multiplier for Llama-family blocks.")
        parser.add_argument("--multiple-of", type=int, default=None, help="Llama-family hidden-size rounding multiple.")
        parser.add_argument("--rope-base", type=float, default=None, help="RoPE base for Llama-family attention.")
        parser.add_argument("--qk-gain-init", type=float, default=None, help="Initial query/key gain for Llama-family attention.")
        parser.add_argument("--logit-softcap", type=float, default=None, help="Optional logit softcap.")
        parser.add_argument("--experts", type=int, default=None, help="Expert count for standard MoE recipes.")
        parser.add_argument("--top-k", type=int, default=None, help="Active expert count for MoE routing.")
        parser.add_argument("--bias", action="store_true", default=None, help="Enable NanoGPT linear bias terms.")
        parser.add_argument("--dropout-p", type=float, default=None, help="NanoGPT dropout probability.")
        parser.add_argument("--experimental-semantic-router-vecs", action="store_true", help="Emit semantic_router_vecs and use vector-driven semantic routing.")
        parser.add_argument("--ema-decay", type=float, default=None, help="EMA decay for JEPA target encoders.")
        parser.add_argument("--ar-loss-coef", type=float, default=None, help="Autoregressive loss scale.")
        parser.add_argument("--jepa-loss-coef", type=float, default=None, help="JEPA loss scale.")
        parser.add_argument("--semantic-align-loss-coef", type=float, default=None, help="Semantic alignment loss scale.")
        parser.add_argument("--optimizer-profile", default=None, help="Gradient optimizer profile.")
        parser.add_argument("--learning-rate", type=float, default=None, help="Global learning rate.")
        parser.add_argument("--lr-decay-iters", type=int, default=env_optional_int("LR_DECAY_ITERS"), help="Optional cosine LR decay horizon.")
        parser.add_argument("--min-lr", type=float, default=env_optional_float("MIN_LR"), help="Optional cosine LR floor.")
        parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay.")
        parser.add_argument("--embed-lr", type=float, default=None, help="Embedding learning rate.")
        parser.add_argument("--head-lr", type=float, default=None, help="LM-head learning rate.")
        parser.add_argument("--tied-embed-lr", type=float, default=None, help="Tied-embedding learning rate.")
        parser.add_argument("--matrix-lr", type=float, default=None, help="Matrix-parameter learning rate.")
        parser.add_argument("--scalar-lr", type=float, default=None, help="Scalar-parameter learning rate.")
        parser.add_argument("--muon-momentum", type=float, default=None, help="Muon momentum.")
        parser.add_argument("--muon-backend-steps", type=int, default=None, help="Muon backend lookahead steps.")
        parser.add_argument("--muon-momentum-warmup-start", type=float, default=None, help="Muon warmup starting momentum.")
        parser.add_argument("--muon-momentum-warmup-steps", type=int, default=None, help="Muon warmup steps.")
        parser.add_argument("--beta1", type=float, default=None, help="Adam beta1.")
        parser.add_argument("--beta2", type=float, default=None, help="Adam beta2.")
        parser.add_argument("--adam-eps", type=float, default=None, help="Adam epsilon.")
        parser.add_argument("--grad-clip-norm", type=float, default=None, help="Gradient clipping norm.")
        parser.add_argument("--evolutionary", action="store_true", help="Use evolutionary search instead of gradient descent.")
        parser.add_argument("--evo-population-size", type=int, default=None, help="Evolutionary population size.")
        parser.add_argument("--evo-mutation-rate", type=float, default=None, help="Evolutionary mutation rate.")
        parser.add_argument("--evo-mutation-scale", type=float, default=None, help="Evolutionary mutation scale.")
        parser.add_argument("--evo-crossover-rate", type=float, default=None, help="Evolutionary crossover rate.")
        parser.add_argument("--evo-tournament-size", type=int, default=None, help="Evolutionary tournament size.")
        parser.add_argument("--evo-elite-count", type=int, default=None, help="Evolutionary elite count.")
        parser.add_argument("--evo-seed", type=int, default=None, help="Optional RNG seed override for evolutionary search.")
    elif command == "infer":
        parser.add_argument("--graph", default=None, help="Graph artifact path.")
        parser.add_argument("--weights", default=None, help="Weights artifact path.")
        parser.add_argument(
            "--checkpoint",
            default=None,
            help="Graphless Parameter Golf .pt checkpoint path. Equivalent to --weights without --graph for supported checkpoints.",
        )
        parser.add_argument(
            "--checkpoint-tokenizer",
            default=None,
            help="SentencePiece .model path for graphless Parameter Golf checkpoints.",
        )
        parser.add_argument(
            "--checkpoint-log",
            default=None,
            help="Optional Parameter Golf training log whose Hyperparameters block supplies non-tensor runtime hints.",
        )
        parser.add_argument("--prompt", default=None, help="Text prompt.")
        parser.add_argument("--prompt-tokens", default=None, help="Comma-separated token ids that override --prompt.")
        parser.add_argument("--sem-targets", default=None, help="Optional comma-separated semantic target ids.")
        parser.add_argument("--semantic-topics", default=None, help="Optional semantic topic overrides in dimension=topic form.")
        parser.add_argument("--experimental-semantic-router-vecs", action="store_true", help="Generate semantic_router_vecs when the loaded graph expects them.")
        parser.add_argument("--max-new-tokens", type=int, default=None, help="Maximum new tokens to generate.")
        parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
        parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling cutoff.")
        parser.add_argument("--top-p", type=top_p_arg, default=None, help="Nucleus sampling cutoff.")
        parser.add_argument("--repetition-penalty", type=repetition_penalty_arg, default=None, help="Penalty applied to tokens already seen in the prompt or continuation.")
        parser.add_argument(
            "--no-repeat-ngram-size",
            type=lambda raw: _infer_nonnegative_int(raw, flag_name="no_repeat_ngram_size"),
            default=None,
            help="Graphless checkpoint repeat guard. Ban tokens that would repeat an n-gram of this size; 0 disables.",
        )
        parser.add_argument(
            "--repeat-run-limit",
            type=lambda raw: _infer_nonnegative_int(raw, flag_name="repeat_run_limit"),
            default=None,
            help="Graphless checkpoint repeat guard. Ban a token after this many consecutive repeats; 0 disables.",
        )
        parser.add_argument("--log-every", type=int, default=None, help="Generation logging interval.")
        parser.add_argument("--stop-token", type=int, default=None, help="Optional stop token id.")
        parser.add_argument("--logits-node", default=None, help="Optional traced logits node override.")
        parser.add_argument("--context-window", type=int, default=None, help="Optional context-window override.")
    else:
        parser.add_argument("--graph", default=None, help="Graph artifact path.")
        parser.add_argument("--weights", default=None, help="Weights artifact path.")
        parser.add_argument("--report-path", default=None, help="JSON report path.")
        parser.add_argument("--eval-batches", type=int, default=None, help="Validation batches to score.")
        parser.add_argument("--eval-batch-size", type=int, default=None, help="Validation batch size.")
        parser.add_argument("--prompt-suite", choices=("auto", "general", "shakespeare"), default=None, help="Prompt suite for prompt probes.")
        parser.add_argument("--prompt", default=None, help="Optional single prompt override for prompt probes.")
        parser.add_argument("--prompt-tokens", default=None, help="Comma-separated token ids that override --prompt.")
        parser.add_argument("--sem-targets", default=None, help="Optional comma-separated semantic target ids.")
        parser.add_argument("--semantic-topics", default=None, help="Optional semantic topic overrides.")
        parser.add_argument("--experimental-semantic-router-vecs", action="store_true", help="Generate semantic_router_vecs when the loaded graph expects them.")
        parser.add_argument("--max-new-tokens", type=int, default=None, help="Maximum new tokens to generate per prompt.")
        parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
        parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling cutoff.")
        parser.add_argument("--top-p", type=top_p_arg, default=None, help="Nucleus sampling cutoff.")
        parser.add_argument("--repetition-penalty", type=repetition_penalty_arg, default=None, help="Penalty applied to previously seen tokens.")
        parser.add_argument("--logits-node", default=None, help="Optional traced logits node override.")
        parser.add_argument("--context-window", type=int, default=None, help="Optional context-window override.")
        parser.add_argument("--stop-token", type=int, default=None, help="Optional stop token id.")
        parser.add_argument("--log-every", type=int, default=None, help="Generation logging interval.")
    return parser


def namespace_from_state(command: str, state: dict[str, Any]) -> argparse.Namespace:
    parser = build_command_parser(command, style="long")
    args = parser.parse_args([])
    for key, value in state.items():
        setattr(args, key, value)
    return args


def collect_explicit_dests(parser: argparse.ArgumentParser, argv: Sequence[str]) -> set[str]:
    option_map: dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_map[option] = action.dest
    explicit: set[str] = set()
    for token in argv:
        if token == "--":
            break
        option = token.split("=", 1)[0] if token.startswith("-") else token
        dest = option_map.get(option)
        if dest:
            explicit.add(dest)
    return explicit


def detect_command(argv: Sequence[str]) -> tuple[str | None, list[str]]:
    tokens = list(argv)
    if not tokens:
        return None, []
    if tokens[0] in COMMANDS:
        return str(tokens[0]), tokens[1:]
    return None, tokens


def is_tty(stdin_isatty: bool | None = None, stdout_isatty: bool | None = None) -> bool:
    in_tty = sys.stdin.isatty() if stdin_isatty is None else bool(stdin_isatty)
    out_tty = sys.stdout.isatty() if stdout_isatty is None else bool(stdout_isatty)
    return in_tty and out_tty


def render_help(command: str | None, *, style: str = "long") -> str:
    parser = build_root_parser(style) if command is None else build_command_parser(command, style)
    return parser.format_help()


def choose_help_style_interactively() -> str:
    options = [
        OptionChoice("Short", "Compact help with terse descriptions.", "short"),
        OptionChoice("Long", "Standard prose help with defaults and guidance.", "long", recommended=True),
        OptionChoice("Verbose", "Expanded help with planner and preset notes.", "verbose"),
    ]
    return str(run_single_choice_menu("Help Style", "Choose a help detail level.", options))


def default_help_style(argv: Sequence[str], *, tty: bool) -> str:
    tokens = list(argv)
    if "--help-style" in tokens:
        idx = tokens.index("--help-style")
        if idx + 1 < len(tokens) and tokens[idx + 1] in HELP_STYLES:
            return str(tokens[idx + 1])
    for token in tokens:
        if token.startswith("--help-style="):
            value = token.split("=", 1)[1]
            if value in HELP_STYLES:
                return value
    if tty:
        return choose_help_style_interactively()
    return "long"


def run_single_choice_menu(title: str, prompt: str, options: list[OptionChoice]) -> Any:
    result = run_curses_questionnaire(title, [Question("_choice", prompt, lambda _state: options, lambda _s, _e: True)], {})
    return result


def next_visible_question_key(
    questions: Sequence[Question],
    state: dict[str, Any],
    explicit: set[str],
    current_key: str,
) -> str | None:
    visible_keys = {question.key for question in questions if question.visible(state, explicit)}
    after_current = False
    for question in questions:
        if after_current and question.key in visible_keys:
            return question.key
        if question.key == current_key:
            after_current = True
    return None


def run_curses_questionnaire(title: str, questions: list[Question], state: dict[str, Any]) -> Any:
    def visible_questions(explicit: set[str]) -> list[Question]:
        return [question for question in questions if question.visible(state, explicit)]

    def initial_index(question: Question) -> int:
        options = question.options_factory(state)
        current_value = state.get(question.key)
        for idx, choice in enumerate(options):
            if isinstance(choice.value, dict):
                if all(state.get(key) == value for key, value in choice.value.items()):
                    return idx
            elif current_value == choice.value:
                return idx
        for idx, choice in enumerate(options):
            if choice.recommended:
                return idx
        return 0

    def apply_choice(stdscr, question: Question, choice: OptionChoice) -> None:
        value = choice.value
        if choice.custom_prompt is not None:
            value = prompt_curses_text(stdscr, choice.custom_prompt, parser=choice.parser)
            if isinstance(choice.value, dict) and isinstance(value, dict):
                value = {**choice.value, **value}
            elif isinstance(choice.value, dict):
                merged = dict(choice.value)
                merged[question.key] = value
                value = merged
        if isinstance(value, dict):
            state.update(value)
        else:
            state[question.key] = value
        normalize_dataset_selector_state(state)

    def render(stdscr, question: Question, question_idx: int, current_visible: list[Question], selected: int) -> None:
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()
        header = f"{title}  [{question_idx + 1}/{len(current_visible)}]"
        stdscr.addnstr(0, 0, header, max_x - 1, curses.A_BOLD)
        stdscr.addnstr(2, 0, question.prompt, max_x - 1)
        stdscr.addnstr(3, 0, "Up/Down: choice  Left/Right: questions  Enter: select  q: abort", max_x - 1)
        row = 5
        for idx, choice in enumerate(question.options_factory(state)):
            attrs = curses.A_REVERSE if idx == selected else curses.A_NORMAL
            prefix = ">" if idx == selected else " "
            stdscr.addnstr(row, 0, f"{prefix} {choice.label}", max_x - 1, attrs)
            row += 1
            stdscr.addnstr(row, 2, choice.description, max_x - 3, attrs)
            row += 2
        row = max(row, max_y - 6)
        stdscr.addnstr(row, 0, "Current choices:", max_x - 1, curses.A_BOLD)
        row += 1
        for key in sorted(k for k in state if not k.startswith("_")):
            stdscr.addnstr(row, 2, f"{key}={state[key]}", max_x - 3)
            row += 1
            if row >= max_y:
                break
        stdscr.refresh()

    def prompt_curses_text(stdscr, prompt: str, *, parser: Callable[[str], Any] | None = None) -> Any:
        while True:
            stdscr.erase()
            max_y, max_x = stdscr.getmaxyx()
            stdscr.addnstr(0, 0, prompt, max_x - 1, curses.A_BOLD)
            stdscr.addnstr(2, 0, "Enter a value and press Return. Esc cancels.", max_x - 1)
            stdscr.move(4, 0)
            curses.curs_set(1)
            curses.echo()
            text = stdscr.getstr(4, 0).decode("utf-8").strip()
            curses.noecho()
            curses.curs_set(0)
            if parser is None:
                return text
            try:
                return parser(text)
            except ValueError as exc:
                stdscr.addnstr(6, 0, f"Invalid value: {exc}", max_x - 1, curses.A_BOLD)
                stdscr.getch()

    def run(stdscr) -> Any:
        curses.curs_set(0)
        current_question = 0
        explicit: set[str] = set()
        while True:
            current_visible = visible_questions(explicit)
            if not current_visible:
                return state
            if len(current_visible) == 1 and current_visible[0].key == "_choice":
                question = current_visible[0]
                selected = initial_index(question)
                while True:
                    render(stdscr, question, 0, current_visible, selected)
                    key = stdscr.getch()
                    if key in (curses.KEY_UP, ord("k")):
                        selected = (selected - 1) % len(question.options_factory(state))
                    elif key in (curses.KEY_DOWN, ord("j")):
                        selected = (selected + 1) % len(question.options_factory(state))
                    elif key in (curses.KEY_ENTER, 10, 13, curses.KEY_RIGHT):
                        return question.options_factory(state)[selected].value
                    elif key in (ord("q"), 27):
                        raise KeyboardInterrupt
            current_question = max(0, min(current_question, len(current_visible) - 1))
            question = current_visible[current_question]
            selected = initial_index(question)
            while True:
                render(stdscr, question, current_question, current_visible, selected)
                key = stdscr.getch()
                if key in (curses.KEY_UP, ord("k")):
                    selected = (selected - 1) % len(question.options_factory(state))
                elif key in (curses.KEY_DOWN, ord("j")):
                    selected = (selected + 1) % len(question.options_factory(state))
                elif key == curses.KEY_LEFT:
                    if current_question > 0:
                        current_question -= 1
                    break
                elif key in (curses.KEY_ENTER, 10, 13, curses.KEY_RIGHT):
                    apply_choice(stdscr, question, question.options_factory(state)[selected])
                    next_key = next_visible_question_key(questions, state, explicit, question.key)
                    if next_key is None:
                        return state
                    current_visible = visible_questions(explicit)
                    for idx, candidate in enumerate(current_visible):
                        if candidate.key == next_key:
                            current_question = idx
                            break
                    break
                elif key in (ord("q"), 27):
                    raise KeyboardInterrupt

    try:
        return curses.wrapper(run)
    except Exception:
        return fallback_prompt(title, questions, state)


def fallback_prompt(title: str, questions: list[Question], state: dict[str, Any]) -> Any:
    print(title)
    for question in questions:
        if not question.visible(state, set()):
            continue
        options = question.options_factory(state)
        print(question.prompt)
        for idx, choice in enumerate(options, start=1):
            suffix = " (recommended)" if choice.recommended else ""
            print(f"  {idx}. {choice.label}{suffix} - {choice.description}")
        while True:
            raw = input("> ").strip() or "1"
            try:
                choice = options[int(raw) - 1]
            except (ValueError, IndexError):
                continue
            value = choice.value
            if choice.custom_prompt is not None:
                raw_value = input(f"{choice.custom_prompt}: ").strip()
                value = choice.parser(raw_value) if choice.parser is not None else raw_value
                if isinstance(choice.value, dict) and not isinstance(value, dict):
                    merged = dict(choice.value)
                    merged[question.key] = value
                    value = merged
            if isinstance(value, dict):
                state.update(value)
            else:
                state[question.key] = value
            break
    if len(questions) == 1 and questions[0].key == "_choice":
        return state["_choice"]
    return state


def pick_recommended_option(options: Sequence[OptionChoice]) -> OptionChoice:
    for option in options:
        if option.recommended:
            return option
    return options[0]


def _custom_int(prompt: str) -> OptionChoice:
    return OptionChoice("Custom...", prompt, {}, parser=lambda raw: int(raw), custom_prompt=prompt)


def _custom_float(prompt: str) -> OptionChoice:
    return OptionChoice("Custom...", prompt, {}, parser=lambda raw: float(raw), custom_prompt=prompt)


def infer_context_window_from_graph(graph) -> int | None:
    dataset_node = graph.nodes.get("dataset_source") if hasattr(graph, "nodes") else None
    if dataset_node is None:
        return None
    module_config = dict(dataset_node.neuron_def.module_config or {})
    seq_len = module_config.get("seq_len")
    if seq_len in (None, ""):
        return None
    return int(seq_len)


def infer_recipe_overrides(state: dict[str, Any]) -> list[str]:
    return [key for key in INFER_RECIPE_KEYS if key in state and state.get(key) not in (None, "", False)]


def warn_ignored_infer_recipe_overrides(state: dict[str, Any], graph_path: Path) -> None:
    ignored = infer_recipe_overrides(state)
    if not ignored:
        return
    flag_names = {
        "base_model": "--base-model",
        "topology": "--topology",
        "router_mode": "--router-mode",
        "use_jepa": "--jepa",
        "megakernel": "--megakernel",
    }
    joined = ", ".join(flag_names.get(key, f"--{key.replace('_', '-')}") for key in ignored)
    print(
        f"Infer is graph-first; ignoring {joined} because {graph_path} is authoritative.",
        file=sys.stderr,
    )


def infer_graph_picker_options() -> list[OptionChoice]:
    artifact_dir = artifact_root()
    if not artifact_dir.exists():
        return [OptionChoice("Custom path...", "Enter a graph artifact path.", "", recommended=True, custom_prompt="Graph artifact path")]

    options: list[OptionChoice] = []
    graph_paths = sorted(
        (
            path for path in artifact_dir.glob("*.json")
            if not path.name.endswith(".eval.json")
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for graph_path in graph_paths:
        try:
            graph = load_graph(graph_path)
            weights_path = resolve_graph_weights_path(graph, graph_path=graph_path, weights_path=None)
            if not weights_path.exists():
                continue
            runtime = infer_graph_template_runtime(graph) or "unknown"
            tokenizer_manifest = tokenizer_manifest_for_graph(graph) or {}
            tokenizer_name = str(
                tokenizer_manifest.get("tokenizer_name")
                or tokenizer_manifest.get("encoding_name")
                or ""
            ).strip() or "unknown"
            context_window = infer_context_window_from_graph(graph)
            context_text = f"ctx {context_window}" if context_window is not None else "ctx ?"
            description = f"{runtime}, {tokenizer_name}, {context_text}, weights {weights_path.name}"
            options.append(
                OptionChoice(
                    graph_path.name,
                    description,
                    str(graph_path),
                    recommended=not options,
                )
            )
        except Exception:
            continue
    options.append(
        OptionChoice(
            "Custom path...",
            "Enter a graph artifact path.",
            "",
            recommended=not options,
            custom_prompt="Graph artifact path",
        )
    )
    return options


def choose_infer_graph_path() -> Path:
    question = Question(
        "graph",
        "Choose a graph artifact.",
        lambda _state: infer_graph_picker_options(),
        lambda _state, _explicit: True,
    )
    state = run_curses_questionnaire("nfn infer", [question], {})
    graph_value = str(state.get("graph") or "").strip()
    if not graph_value:
        raise KeyboardInterrupt
    return Path(graph_value).expanduser().resolve()


def resolve_infer_graph_path(args: argparse.Namespace, *, interactive: bool) -> Path:
    graph_value = str(getattr(args, "graph", None) or "").strip()
    if graph_value:
        return Path(graph_value).expanduser().resolve()
    if not interactive:
        raise FileNotFoundError("Non-interactive infer requires --graph or --checkpoint/--weights.")
    graph_path = choose_infer_graph_path()
    args.graph = str(graph_path)
    return graph_path


def ensure_infer_defaults(args: argparse.Namespace, *, interactive: bool) -> argparse.Namespace:
    preset = INFER_GENERATION_PRESETS["balanced"]
    if getattr(args, "seed", None) is None:
        args.seed = 1337
    if getattr(args, "max_new_tokens", None) is None:
        args.max_new_tokens = int(preset["max_new_tokens"])
    if getattr(args, "temperature", None) is None:
        args.temperature = float(preset["temperature"])
    if getattr(args, "top_k", None) is None:
        args.top_k = int(preset["top_k"])
    if getattr(args, "top_p", None) is None:
        args.top_p = float(preset["top_p"])
    if getattr(args, "repetition_penalty", None) is None:
        args.repetition_penalty = float(preset["repetition_penalty"])
    if getattr(args, "log_every", None) is None:
        args.log_every = 0 if interactive else 1
    if getattr(args, "logits_node", None) is None:
        args.logits_node = "auto"
    return args


def infer_settings_from_args(args: argparse.Namespace) -> InferChatSettings:
    return InferChatSettings(
        top_k=int(getattr(args, "top_k", None) or 0),
        top_p=float(getattr(args, "top_p", None) or DEFAULT_INFER_TOP_P),
        temperature=float(getattr(args, "temperature", None) or 0.0),
        max_new_tokens=int(getattr(args, "max_new_tokens", None) or 1),
        repetition_penalty=float(getattr(args, "repetition_penalty", None) or 1.0),
        autocomplete_words=int(getattr(args, "autocomplete_words", None) or 0),
    )


def infer_settings_signature(settings: InferChatSettings) -> tuple[int, float, float, int, float, int]:
    return (
        int(settings.top_k),
        round(float(settings.top_p), 6),
        round(float(settings.temperature), 6),
        int(settings.max_new_tokens),
        round(float(settings.repetition_penalty), 6),
        int(settings.autocomplete_words),
    )


def infer_prompt_was_supplied(args: argparse.Namespace) -> bool:
    return bool(
        str(getattr(args, "prompt", None) or "").strip()
        or str(getattr(args, "prompt_tokens", None) or "").strip()
    )


def resolve_graphless_checkpoint_path(args: argparse.Namespace) -> Path | None:
    checkpoint_value = str(getattr(args, "checkpoint", None) or "").strip()
    if checkpoint_value:
        return Path(checkpoint_value).expanduser().resolve()
    weights_value = str(getattr(args, "weights", None) or "").strip()
    graph_value = str(getattr(args, "graph", None) or "").strip()
    if weights_value and not graph_value and Path(weights_value).suffix.lower() == ".pt":
        return Path(weights_value).expanduser().resolve()
    return None


def resolve_parameter_golf_device(device_arg: str) -> torch.device:
    raw = str(device_arg or "cuda").strip().lower()
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device is not available in this environment.")
    return device


def _amp_dtype_from_name(name: str | None, *, device: torch.device) -> tuple[torch.dtype, str]:
    raw = str(name or ("bfloat16" if device.type == "cuda" else "float32")).strip().lower()
    if raw == "bf16":
        raw = "bfloat16"
    if raw in {"float32", "fp32"}:
        return torch.float32, "float32"
    if raw in {"bfloat16", "bf16"}:
        return torch.bfloat16, "bfloat16"
    if raw in {"float16", "fp16"}:
        return torch.float16, "float16"
    raise ValueError(f"Unsupported AMP dtype for Parameter Golf checkpoint inference: {name!r}")


def _infer_nonnegative_int(raw: str, *, flag_name: str) -> int:
    value = int(raw)
    if value < 0:
        raise ValueError(f"{flag_name} must be greater than or equal to 0.")
    return value


def _infer_positive_int(raw: str, *, flag_name: str) -> int:
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{flag_name} must be greater than 0.")
    return value


def _infer_temperature_arg(raw: str) -> float:
    value = float(raw)
    if value < 0.0:
        raise ValueError("temperature must be greater than or equal to 0.")
    return value


def _infer_repetition_penalty_arg(raw: str) -> float:
    value = float(raw)
    if value < 1.0:
        raise ValueError("repetition_penalty must be greater than or equal to 1.0.")
    return value


def _infer_autocomplete_words_arg(raw: str) -> int:
    return _infer_nonnegative_int(raw, flag_name="autocomplete_words")


def edit_infer_settings(settings: InferChatSettings) -> InferChatSettings:
    state: dict[str, Any] = {
        "top_k": int(settings.top_k),
        "top_p": float(settings.top_p),
        "temperature": float(settings.temperature),
        "max_new_tokens": int(settings.max_new_tokens),
        "repetition_penalty": float(settings.repetition_penalty),
        "autocomplete_words": int(settings.autocomplete_words),
    }
    current = lambda key: state[key]
    questions = [
        Question(
            "top_k",
            "Choose the top-k sampling cutoff.",
            lambda _state: [
                _option_with_value(
                    "Keep current",
                    _format_menu_number(int(current("top_k"))),
                    "Leave top-k unchanged for this session.",
                    int(current("top_k")),
                    recommended=True,
                ),
                _option_with_value("Greedy", "0", "Disable top-k filtering.", 0),
                _option_with_value("Focused", "16", "Tighter candidate set.", 16),
                _option_with_value("Balanced", "32", "Current default balance.", 32),
                _option_with_value("Broad", "64", "Wider candidate set.", 64),
                OptionChoice(
                    "Custom...",
                    "Enter a non-negative top-k value.",
                    {},
                    custom_prompt="top_k",
                    parser=lambda raw: {"top_k": _infer_nonnegative_int(raw, flag_name="top_k")},
                ),
            ],
            lambda _state, _explicit: True,
        ),
        Question(
            "top_p",
            "Choose the nucleus sampling cutoff.",
            lambda _state: [
                _option_with_value(
                    "Keep current",
                    _format_menu_number(float(current("top_p"))),
                    "Leave top-p unchanged for this session.",
                    float(current("top_p")),
                    recommended=True,
                ),
                _option_with_value("Conservative", "0.9", "Keep only the tightest probability mass.", 0.9),
                _option_with_value("Balanced", "0.95", "Balanced nucleus cutoff.", 0.95),
                _option_with_value("Open", "1.0", "Disable nucleus filtering.", 1.0),
                OptionChoice(
                    "Custom...",
                    "Enter top_p in the range (0, 1].",
                    {},
                    custom_prompt="top_p",
                    parser=lambda raw: {"top_p": top_p_arg(raw)},
                ),
            ],
            lambda _state, _explicit: True,
        ),
        Question(
            "temperature",
            "Choose the sampling temperature.",
            lambda _state: [
                _option_with_value(
                    "Keep current",
                    _format_menu_number(float(current("temperature"))),
                    "Leave temperature unchanged for this session.",
                    float(current("temperature")),
                    recommended=True,
                ),
                _option_with_value("Greedy", "0", "Always pick the argmax token.", 0.0),
                _option_with_value("Focused", "0.4", "Sharper probability distribution.", 0.4),
                _option_with_value("Balanced", "0.8", "Current default balance.", 0.8),
                _option_with_value("Exploratory", "1.0", "Keep the logits distribution looser.", 1.0),
                OptionChoice(
                    "Custom...",
                    "Enter temperature >= 0.",
                    {},
                    custom_prompt="temperature",
                    parser=lambda raw: {"temperature": _infer_temperature_arg(raw)},
                ),
            ],
            lambda _state, _explicit: True,
        ),
        Question(
            "max_new_tokens",
            "Choose the maximum response length.",
            lambda _state: [
                _option_with_value(
                    "Keep current",
                    _format_menu_number(int(current("max_new_tokens"))),
                    "Leave max_new_tokens unchanged for this session.",
                    int(current("max_new_tokens")),
                    recommended=True,
                ),
                _option_with_value("Short", "32", "Compact responses.", 32),
                _option_with_value("Balanced", "64", "Current default balance.", 64),
                _option_with_value("Long", "128", "Longer responses.", 128),
                _option_with_value("Extended", "256", "Extended responses.", 256),
                OptionChoice(
                    "Custom...",
                    "Enter max_new_tokens > 0.",
                    {},
                    custom_prompt="max_new_tokens",
                    parser=lambda raw: {"max_new_tokens": _infer_positive_int(raw, flag_name="max_new_tokens")},
                ),
            ],
            lambda _state, _explicit: True,
        ),
        Question(
            "repetition_penalty",
            "Choose the repetition penalty.",
            lambda _state: [
                _option_with_value(
                    "Keep current",
                    _format_menu_number(float(current("repetition_penalty"))),
                    "Leave repetition penalty unchanged for this session.",
                    float(current("repetition_penalty")),
                    recommended=True,
                ),
                _option_with_value("Off", "1.0", "Disable repetition penalty.", 1.0),
                _option_with_value("Balanced", "1.08", "Lightly discourage repeated tokens.", 1.08),
                _option_with_value("Stronger", "1.15", "Push harder against loops.", 1.15),
                OptionChoice(
                    "Custom...",
                    "Enter repetition_penalty >= 1.0.",
                    {},
                    custom_prompt="repetition_penalty",
                    parser=lambda raw: {"repetition_penalty": _infer_repetition_penalty_arg(raw)},
                ),
            ],
            lambda _state, _explicit: True,
        ),
        Question(
            "autocomplete_words",
            "Choose the inline autocomplete word count.",
            lambda _state: [
                _option_with_value(
                    "Keep current",
                    _format_menu_number(int(current("autocomplete_words"))),
                    "Leave inline autocomplete unchanged for this session.",
                    int(current("autocomplete_words")),
                    recommended=True,
                ),
                _option_with_value("Off", "0", "Disable inline typing predictions.", 0),
                _option_with_value("Single word", "1", "Ghost one predicted word.", 1),
                _option_with_value("Short phrase", "3", "Ghost three predicted words.", 3),
                _option_with_value("Longer phrase", "5", "Ghost five predicted words.", 5),
                OptionChoice(
                    "Custom...",
                    "Enter autocomplete_words >= 0.",
                    {},
                    custom_prompt="autocomplete_words",
                    parser=lambda raw: {"autocomplete_words": _infer_autocomplete_words_arg(raw)},
                ),
            ],
            lambda _state, _explicit: True,
        ),
    ]
    resolved = run_curses_questionnaire("Infer settings", questions, state)
    return InferChatSettings(
        top_k=int(resolved["top_k"]),
        top_p=float(resolved["top_p"]),
        temperature=float(resolved["temperature"]),
        max_new_tokens=int(resolved["max_new_tokens"]),
        repetition_penalty=float(resolved["repetition_penalty"]),
        autocomplete_words=int(resolved["autocomplete_words"]),
    )


def build_infer_runtime_context(
    args: argparse.Namespace,
    *,
    state: dict[str, Any],
    interactive: bool,
) -> InferRuntimeContext:
    args.dataset_alias = str(getattr(args, "dataset_alias", None) or DEFAULT_DATASET_ALIAS)
    ensure_infer_defaults(args, interactive=interactive)
    checkpoint_path = resolve_graphless_checkpoint_path(args)
    if checkpoint_path is not None:
        return build_parameter_golf_infer_runtime_context(args, checkpoint_path=checkpoint_path)
    apply_tinystories_dataset_defaults(args)
    resolve_dataset_selector_args(args)

    graph_path = resolve_infer_graph_path(args, interactive=interactive)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph artifact not found: {graph_path}")
    warn_ignored_infer_recipe_overrides(state, graph_path)

    if args.device != "cuda":
        raise RuntimeError("This inference command is configured to run on CUDA only.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is not available in this environment.")

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(args.seed))

    graph, compiled, state_dict, resolved_weights_path = load_compiled_inference_graph(
        graph_path=graph_path,
        weights_path=Path(args.weights).expanduser().resolve() if getattr(args, "weights", None) else None,
        device=device,
    )
    raw_text_encoding_name = resolve_raw_text_encoding_name(graph, encoding_override=getattr(args, "raw_text_encoding_override", None))
    dataset_alias = resolve_inference_dataset_alias(args, graph, default_alias=DEFAULT_DATASET_ALIAS, log=log_stage)
    tokenizer, tokenizer_path, tokenizer_name, _dataset_name, _dataset_path, _dataset_meta = resolve_inference_tokenizer_context(
        graph=graph,
        state_dict=state_dict,
        dataset_alias=dataset_alias,
        raw_text_encoding_name=raw_text_encoding_name,
        dataset_download_kwargs=dataset_download_kwargs_from_args(args),
        require_dataset=False,
    )
    log_tokenizer_status(log_stage, tokenizer, tokenizer_path, tokenizer_name)

    amp_dtype, amp_name, _use_amp = resolve_autocast_settings(
        graph,
        amp_dtype_override=getattr(args, "amp_dtype", None),
    )
    if getattr(args, "amp_dtype", None):
        graph.torch_config = {**graph.torch_config, "amp_dtype": amp_name}

    context_window = int(getattr(args, "context_window", None) or infer_context_window_from_graph(graph) or 0)
    if context_window <= 0:
        raise RuntimeError("Could not resolve a positive context window from the graph.")

    return InferRuntimeContext(
        args=args,
        graph_path=graph_path,
        resolved_weights_path=resolved_weights_path,
        graph=graph,
        compiled=compiled,
        state_dict=state_dict,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        tokenizer_name=tokenizer_name,
        raw_text_encoding_name=raw_text_encoding_name,
        dataset_alias=dataset_alias,
        device=device,
        generator=generator,
        amp_dtype=amp_dtype,
        amp_name=amp_name,
        context_window=context_window,
    )


def build_parameter_golf_infer_runtime_context(
    args: argparse.Namespace,
    *,
    checkpoint_path: Path,
) -> InferRuntimeContext:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint artifact not found: {checkpoint_path}")
    device = resolve_parameter_golf_device(str(getattr(args, "device", "cuda") or "cuda"))
    torch.manual_seed(int(args.seed))
    generator = torch.Generator(device="cuda" if device.type == "cuda" else "cpu")
    generator.manual_seed(int(args.seed))

    state_dict = load_parameter_golf_state_dict(checkpoint_path)
    if not is_parameter_golf_flat_state_dict(state_dict):
        raise RuntimeError(
            "Graphless inference currently supports flat Parameter Golf root GPT checkpoints only. "
            "Pass NeuralFn exports with --graph, or pass a supported Parameter Golf .pt checkpoint with --checkpoint."
        )
    metadata = load_checkpoint_metadata(checkpoint_path)
    training_hparams = load_training_log_hparams(getattr(args, "checkpoint_log", None))
    config = infer_config_from_state_dict(
        state_dict,
        metadata=metadata,
        training_hparams=training_hparams,
        context_window=getattr(args, "context_window", None),
    )
    tokenizer_path = resolve_parameter_golf_tokenizer_path(
        checkpoint_path=checkpoint_path,
        tokenizer_path=getattr(args, "checkpoint_tokenizer", None),
        metadata=metadata,
        training_hparams=training_hparams,
    )
    tokenizer = load_sentencepiece_tokenizer(tokenizer_path, expected_vocab_size=config.vocab_size)
    amp_dtype, amp_name = _amp_dtype_from_name(getattr(args, "amp_dtype", None), device=device)
    model = build_parameter_golf_model(state_dict, config, device=device)

    graph_info = SimpleNamespace(
        name=PARAMETER_GOLF_CHECKPOINT_FORMAT,
        nodes={},
        torch_config={
            "artifact_metadata": {
                "checkpoint_format": PARAMETER_GOLF_CHECKPOINT_FORMAT,
                "metadata_file": str(checkpoint_metadata_path(checkpoint_path)),
            }
        },
    )
    return InferRuntimeContext(
        args=args,
        graph_path=checkpoint_path,
        resolved_weights_path=checkpoint_path,
        graph=graph_info,
        compiled=model,
        state_dict=state_dict,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        tokenizer_name=tokenizer_path.name,
        raw_text_encoding_name="sentencepiece",
        dataset_alias="parameter_golf_checkpoint",
        device=device,
        generator=generator,
        amp_dtype=amp_dtype,
        amp_name=amp_name,
        context_window=int(config.context_window),
        generation_backend="parameter_golf",
    )


def infer_prompt_source(
    *,
    prompt: str,
    prompt_tokens: str,
    tokenizer,
) -> tuple[str, list[int]]:
    prompt_ids = resolve_prompt_tokens(
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        tokenizer=tokenizer,
    )
    prompt_text = resolve_prompt_text(
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        prompt_ids=prompt_ids,
        tokenizer=tokenizer,
    )
    return prompt_text, prompt_ids


def render_infer_transcript(
    history: Sequence[tuple[str, str]],
    draft: str,
    *,
    include_assistant_prompt: bool,
) -> str:
    lines: list[str] = []
    for user_text, assistant_text in history:
        lines.append(f"User: {user_text}")
        lines.append(f"Assistant: {assistant_text}")
    lines.append(f"User: {draft}")
    if include_assistant_prompt:
        lines.append("Assistant:")
    return "\n".join(lines)


def resolve_infer_chat_prompt(
    context: InferRuntimeContext,
    *,
    mode: str,
    history: Sequence[tuple[str, str]],
    draft: str,
    include_assistant_prompt: bool,
) -> tuple[str, list[int], int]:
    if mode not in INFER_CHAT_MODES:
        raise ValueError(f"Unsupported infer chat mode: {mode}")
    if mode == "stateless":
        prompt_text, prompt_ids = infer_prompt_source(prompt=draft, prompt_tokens="", tokenizer=context.tokenizer)
        return prompt_text, prompt_ids, 0

    working_history = list(history)
    dropped_turns = 0
    while True:
        prompt_text = render_infer_transcript(
            working_history,
            draft,
            include_assistant_prompt=include_assistant_prompt,
        )
        prompt_ids = resolve_prompt_tokens(
            prompt=prompt_text,
            prompt_tokens="",
            tokenizer=context.tokenizer,
        )
        if len(prompt_ids) <= context.context_window or not working_history:
            return prompt_text, prompt_ids, dropped_turns
        working_history = working_history[1:]
        dropped_turns += 1


def infer_graph_uses_semantics(context: InferRuntimeContext) -> bool:
    if context.generation_backend != "graph":
        return False
    return "semantic_data_source" in context.graph.nodes


def resolve_parameter_golf_logits_key(requested: str) -> str:
    if requested == "auto":
        return "parameter_golf/softcap"
    allowed = {
        "parameter_golf/softcap",
        "softcap",
        "logits",
        "model/softcap",
        "model/lm_head",
        "model/tied_lm_head",
    }
    if requested in allowed:
        return requested
    raise KeyError(
        "Graphless Parameter Golf inference does not expose traced graph nodes. "
        "Use --logits-node auto, parameter_golf/softcap, softcap, or logits."
    )


def suppress_parameter_golf_token_ids(
    logits: torch.Tensor,
    *,
    token_ids: Sequence[int],
    stop_token: int | None,
) -> torch.Tensor:
    suppressed = {int(token_id) for token_id in token_ids if 0 <= int(token_id) < logits.size(-1)}
    if stop_token is not None:
        suppressed.discard(int(stop_token))
    if not suppressed:
        return logits
    masked = logits.clone()
    masked[:, sorted(suppressed)] = float("-inf")
    if not torch.isfinite(masked).any(dim=-1).all():
        return logits
    return masked


def parameter_golf_repeat_bans(
    generated: Sequence[int],
    *,
    repeat_run_limit: int,
    no_repeat_ngram_size: int,
) -> list[int]:
    bans: set[int] = set()
    if repeat_run_limit > 0 and generated:
        last_token = int(generated[-1])
        run_length = 1
        for token_id in reversed(generated[:-1]):
            if int(token_id) != last_token:
                break
            run_length += 1
        if run_length >= repeat_run_limit:
            bans.add(last_token)

    ngram_size = int(no_repeat_ngram_size)
    if ngram_size > 1 and len(generated) >= ngram_size - 1:
        prefix = tuple(int(token_id) for token_id in generated[-(ngram_size - 1):])
        for idx in range(0, len(generated) - ngram_size + 1):
            if tuple(int(token_id) for token_id in generated[idx : idx + ngram_size - 1]) == prefix:
                bans.add(int(generated[idx + ngram_size - 1]))
    return sorted(bans)


def build_parameter_golf_generation(
    context: InferRuntimeContext,
    *,
    prompt_ids: list[int],
    settings: InferChatSettings,
    generator: torch.Generator,
    log_every: int,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    generated = list(prompt_ids)
    resolved_logits_key = resolve_parameter_golf_logits_key(str(getattr(context.args, "logits_node", None) or "auto"))
    use_amp = autocast_enabled_for(context.device, context.amp_dtype)
    repeat_run_limit = int(
        getattr(context.args, "repeat_run_limit", None)
        if getattr(context.args, "repeat_run_limit", None) is not None
        else DEFAULT_PARAMETER_GOLF_REPEAT_RUN_LIMIT
    )
    no_repeat_ngram_size = int(
        getattr(context.args, "no_repeat_ngram_size", None)
        if getattr(context.args, "no_repeat_ngram_size", None) is not None
        else DEFAULT_PARAMETER_GOLF_NO_REPEAT_NGRAM_SIZE
    )
    with torch.no_grad():
        for step_idx in range(int(settings.max_new_tokens)):
            context_ids = generated[-context.context_window:]
            tokens = torch.tensor([context_ids], dtype=torch.long, device=context.device)
            with torch.autocast(device_type=context.device.type, dtype=context.amp_dtype, enabled=use_amp):
                logits = context.compiled.forward_logits(tokens)
            stop_token = getattr(context.args, "stop_token", None)
            next_logits = suppress_parameter_golf_token_ids(
                logits[:, -1, :],
                token_ids=[
                    *getattr(context.tokenizer, "suppressed_token_ids", ()),
                    *getattr(context.tokenizer, "lossless_fallback_token_ids", ()),
                    *parameter_golf_repeat_bans(
                        generated,
                        repeat_run_limit=repeat_run_limit,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                    ),
                ],
                stop_token=stop_token,
            )
            next_token = sample_next_token(
                next_logits,
                temperature=float(settings.temperature),
                top_k=int(settings.top_k),
                top_p=float(settings.top_p),
                token_history=generated,
                repetition_penalty=float(settings.repetition_penalty),
                generator=generator,
            )
            generated.append(next_token)
            should_log = (
                log is not None
                and log_every > 0
                and (
                    step_idx == 0
                    or (step_idx + 1) % max(log_every, 1) == 0
                    or step_idx + 1 >= int(settings.max_new_tokens)
                )
            )
            if should_log:
                token_piece = describe_token(context.tokenizer, next_token)
                log(
                    f"Generation step {step_idx + 1}/{int(settings.max_new_tokens)}: "
                    f"token={next_token} piece={token_piece!r}"
                )
            if stop_token is not None and next_token == int(stop_token):
                if log is not None:
                    log(f"Stop token {stop_token} reached; ending generation early")
                break

    generated_tail = generated[len(prompt_ids):]
    return {
        "generated_token_ids": generated_tail,
        "all_token_ids": generated,
        "generated_text": decode_tokens(context.tokenizer, generated_tail) if context.tokenizer is not None else "",
        "full_text": decode_tokens(context.tokenizer, generated) if context.tokenizer is not None else "",
        "resolved_logits_key": resolved_logits_key,
    }


def build_infer_generation(
    context: InferRuntimeContext,
    *,
    prompt_ids: list[int],
    prompt_text: str,
    settings: InferChatSettings,
    generator: torch.Generator | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    active_generator = context.generator if generator is None else generator
    log_every = int(getattr(context.args, "log_every", 0) or 0)
    if context.generation_backend == "parameter_golf":
        result = build_parameter_golf_generation(
            context,
            prompt_ids=prompt_ids,
            settings=settings,
            generator=active_generator,
            log_every=log_every,
            log=log,
        )
        return result, {}
    if infer_graph_uses_semantics(context):
        vocab = ConversationalVocabulary(semantic_vocab_ref_for_graph(context.graph))
        sem_cfg = dict(context.graph.nodes["semantic_data_source"].neuron_def.module_config or {})
        semantic_dim = int(sem_cfg.get("seq_len", vocab.vector_dim))
        sem_targets, semantic_overrides = resolve_semantic_targets(
            str(getattr(context.args, "sem_targets", None) or ""),
            str(getattr(context.args, "semantic_topics", None) or ""),
            semantic_dim,
            context.device,
            vocab,
            sequence_text=prompt_text,
        )
        semantic_router_vecs: torch.Tensor | None = None
        if graph_uses_semantic_router_vecs(context.graph) or bool(getattr(context.args, "experimental_semantic_router_vecs", False)):
            semantic_router_vecs = resolve_semantic_router_vecs(sem_targets, vocab=vocab, device=context.device)
        result = build_semantic_generation(
            graph=context.graph,
            compiled=context.compiled,
            tokenizer=context.tokenizer,
            prompt_ids=prompt_ids,
            device=context.device,
            amp_dtype=context.amp_dtype,
            generator=active_generator,
            max_new_tokens=int(settings.max_new_tokens),
            temperature=float(settings.temperature),
            top_k=int(settings.top_k),
            top_p=float(settings.top_p),
            repetition_penalty=float(settings.repetition_penalty),
            stop_token=getattr(context.args, "stop_token", None),
            logits_node=str(getattr(context.args, "logits_node", None) or "auto"),
            context_window=context.context_window,
            sem_targets=sem_targets,
            semantic_router_vecs=semantic_router_vecs,
            log_every=log_every,
            log=log,
        )
        extras = {
            "semantic_targets": sem_targets[0].tolist(),
            "semantic_overrides": semantic_overrides,
            "semantic_router_vecs": semantic_router_vecs[0].tolist() if semantic_router_vecs is not None else None,
        }
        return result, extras

    result = generate_sequence(
        context.compiled,
        tokenizer=context.tokenizer,
        prompt_ids=prompt_ids,
        device=context.device,
        amp_dtype=context.amp_dtype,
        generator=active_generator,
        max_new_tokens=int(settings.max_new_tokens),
        temperature=float(settings.temperature),
        top_k=int(settings.top_k),
        repetition_penalty=float(settings.repetition_penalty),
        top_p=float(settings.top_p),
        stop_token=getattr(context.args, "stop_token", None),
        logits_node=str(getattr(context.args, "logits_node", None) or "auto"),
        context_window=context.context_window,
        log_every=log_every,
        log=log,
    )
    return result, {}


INFER_THEME = Theme(
    {
        "infer.user": "bold bright_cyan",
        "infer.assistant": "bold bright_magenta",
        "infer.system": "dim italic",
        "infer.banner": "bold white on #2a004d",
        "infer.accent": "bright_yellow",
        "infer.status": "dim",
        "infer.preview": "italic bright_green",
        "infer.ghost": "italic #808080",
        "infer.error": "bold red",
    }
)

INFER_IDLE_STATUS = (
    ":sparkles: [infer.status]Tab[/] preview/insert  "
    "[infer.accent]/help[/]  :gear: /settings  :compass: /mode  "
    ":sparkles: /autocomplete  :eye: /show  :broom: /reset  "
    ":wastebasket: /clear  :door: /exit"
)

INFER_AUTOCOMPLETE_MAX_TOKENS = 128
INFER_AUTOCOMPLETE_WORD_RE = re.compile(r"\S+")
INFER_INPUT_CURSOR_MARK = "\u2060"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def make_infer_console() -> Console:
    return Console(theme=INFER_THEME, emoji=True, highlight=False, soft_wrap=False)


def render_infer_banner(
    console: Console,
    context: "InferRuntimeContext",
    settings: "InferChatSettings",
    mode: str,
) -> None:
    graph_name = str(getattr(context.graph, "name", context.graph_path.stem))
    runtime = infer_graph_template_runtime(context.graph) or "unknown"
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="infer.accent", justify="right", no_wrap=True)
    grid.add_column(overflow="fold")
    if context.generation_backend == "parameter_golf":
        grid.add_row(":brain: Checkpoint", f"{context.graph_path.name}  [dim]({context.graph_path})[/dim]")
        grid.add_row(":floppy_disk: Format", PARAMETER_GOLF_CHECKPOINT_FORMAT)
        grid.add_row(":gear: Runtime", f"{context.amp_name} on {context.device}")
    else:
        grid.add_row(":brain: Graph", f"{graph_name}  [dim]({context.graph_path})[/dim]")
        grid.add_row(":floppy_disk: Weights", context.resolved_weights_path.name)
        grid.add_row(":gear: Runtime", runtime)
    grid.add_row(":abc: Tokenizer", context.tokenizer_name or "unavailable")
    grid.add_row(":straight_ruler: Context", f"{context.context_window} tokens")
    grid.add_row(":compass: Mode", f"[infer.accent]{mode}[/]")
    grid.add_row(
        ":sparkles: Autocomplete",
        (
            "off"
            if int(settings.autocomplete_words) <= 0
            else f"{int(settings.autocomplete_words)} word{'s' if int(settings.autocomplete_words) != 1 else ''}"
        ),
    )
    grid.add_row(
        ":control_knobs: Sampling",
        (
            f"top_k={settings.top_k}  top_p={settings.top_p:g}  "
            f"temp={settings.temperature:g}  max_new={settings.max_new_tokens}  "
            f"repeat={settings.repetition_penalty:g}"
        ),
    )
    console.print(
        Panel(
            Align.center(grid),
            title="[infer.banner] :sparkles:  NeuralFn Infer Chat  :sparkles: [/]",
            subtitle="[infer.status]Tab: preview / insert   :keyboard: /help for commands[/]",
            box=HEAVY,
            border_style="bright_magenta",
            padding=(1, 2),
        )
    )


def render_infer_turn_panel(
    console: Console, role: str, index: int, text: str
) -> None:
    if role == "user":
        title = f":bust_in_silhouette: You  [dim]#{index}[/]"
        border = "infer.user"
        body: Any = Text.from_markup(_rich_escape(text), emoji=True)
    elif role == "assistant":
        title = f":robot: Assistant  [dim]#{index}[/]"
        border = "infer.assistant"
        body = Markdown(text, code_theme="monokai") if text else Text("(empty)")
    else:
        title = ":information_source: System"
        border = "infer.system"
        body = Text(text)
    console.print(
        Panel(
            body,
            title=title,
            title_align="left",
            border_style=border,
            box=ROUNDED,
            padding=(0, 1),
        )
    )


def render_infer_settings_table(settings: "InferChatSettings") -> Table:
    table = Table(title=":control_knobs: Sampling settings", box=ROUNDED, expand=False)
    table.add_column("Parameter", style="infer.accent", no_wrap=True)
    table.add_column("Value")
    table.add_row("top_k", str(settings.top_k))
    table.add_row("top_p", f"{settings.top_p:g}")
    table.add_row("temperature", f"{settings.temperature:g}")
    table.add_row("max_new_tokens", str(settings.max_new_tokens))
    table.add_row("repetition_penalty", f"{settings.repetition_penalty:g}")
    table.add_row("autocomplete_words", str(settings.autocomplete_words))
    return table


@dataclass(frozen=True)
class InferSlashCommand:
    name: str
    description: str
    aliases: tuple[str, ...] = ()
    value_hint: str = ""
    setting_field: str | None = None
    value_type: Callable[[str], Any] | None = None

    @property
    def takes_value(self) -> bool:
        return bool(self.value_hint)

    def all_names(self) -> tuple[str, ...]:
        return (self.name,) + self.aliases

    def display_name(self) -> str:
        head = self.name
        if self.aliases:
            head += ", " + ", ".join(self.aliases)
        if self.value_hint:
            head += f" {self.value_hint}"
        return head


INFER_SLASH_COMMANDS: tuple[InferSlashCommand, ...] = (
    InferSlashCommand("/help", "Show this commands table"),
    InferSlashCommand("/show", "Current model + settings snapshot"),
    InferSlashCommand("/settings", "Open interactive settings menu"),
    InferSlashCommand(
        "/mode",
        "Switch REPL mode",
        value_hint="stateless|transcript",
    ),
    InferSlashCommand("/reset", "Clear transcript history"),
    InferSlashCommand("/clear", "Clear the screen"),
    InferSlashCommand(
        "/temp",
        "Set sampling temperature",
        aliases=("/temperature",),
        value_hint="<float>",
        setting_field="temperature",
        value_type=float,
    ),
    InferSlashCommand(
        "/top_k",
        "Set top-k sampling",
        value_hint="<int>",
        setting_field="top_k",
        value_type=int,
    ),
    InferSlashCommand(
        "/top_p",
        "Set top-p (nucleus) sampling",
        value_hint="<float>",
        setting_field="top_p",
        value_type=float,
    ),
    InferSlashCommand(
        "/max_new",
        "Set max new tokens",
        aliases=("/max_new_tokens",),
        value_hint="<int>",
        setting_field="max_new_tokens",
        value_type=int,
    ),
    InferSlashCommand(
        "/repeat",
        "Set repetition penalty",
        aliases=("/repetition_penalty",),
        value_hint="<float>",
        setting_field="repetition_penalty",
        value_type=_infer_repetition_penalty_arg,
    ),
    InferSlashCommand(
        "/autocomplete",
        "Set inline autocomplete word count",
        value_hint="<words>",
        setting_field="autocomplete_words",
        value_type=_infer_autocomplete_words_arg,
    ),
    InferSlashCommand("/exit", "Leave the chat", aliases=("/quit",)),
)


def _infer_slash_lookup(head: str) -> InferSlashCommand | None:
    head = head.lower()
    for cmd in INFER_SLASH_COMMANDS:
        if head in cmd.all_names():
            return cmd
    return None


def _infer_slash_candidates(prefix: str) -> list[tuple[InferSlashCommand, str]]:
    prefix = prefix.lower()
    out: list[tuple[InferSlashCommand, str]] = []
    for cmd in INFER_SLASH_COMMANDS:
        for name in cmd.all_names():
            if name.startswith(prefix):
                out.append((cmd, name))
                break
    return out


def infer_slash_status_for_buffer(buffer_text: str) -> str | None:
    if not buffer_text.startswith("/"):
        return None
    if " " in buffer_text:
        head, _ = buffer_text.split(" ", 1)
        cmd = _infer_slash_lookup(head)
        if cmd is not None and cmd.takes_value:
            return (
                f":sparkles: [infer.preview]{cmd.name}[/] "
                f"[infer.status]{_rich_escape(cmd.value_hint)}[/]"
            )
        return None

    candidates = _infer_slash_candidates(buffer_text)
    if not candidates:
        return (
            f":warning: [infer.error]No command matches[/] "
            f"{_rich_escape(repr(buffer_text))}"
        )
    if len(candidates) == 1:
        cmd, match = candidates[0]
        hint = f" {_rich_escape(cmd.value_hint)}" if cmd.takes_value else ""
        if match == buffer_text:
            return (
                f":sparkles: [infer.preview]{match}{hint}[/] "
                f"[infer.status]{_rich_escape(cmd.description)}[/]"
            )
        return (
            f":sparkles: [infer.preview]Tab[/] "
            f"[infer.status]complete to[/] [infer.accent]{match}{hint}[/] "
            f"[infer.status]{_rich_escape(cmd.description)}[/]"
        )

    names = [match for _, match in candidates]
    common = os.path.commonprefix(names)
    listing = "  ".join(names)
    if len(common) > len(buffer_text):
        return (
            f":sparkles: [infer.preview]Tab[/] "
            f"[infer.status]complete to[/] [infer.accent]{common}[/]  "
            f"[infer.status]{_rich_escape(listing)}[/]"
        )
    return f":sparkles: [infer.preview]Options:[/] {_rich_escape(listing)}"


def complete_infer_slash_command(buffer_text: str) -> tuple[str | None, str]:
    """Attempt Tab-completion of a slash command.

    Returns a (new_buffer, status) pair. If ``new_buffer`` is ``None`` the caller
    should leave the buffer untouched and only update the status line.
    """
    if " " in buffer_text:
        head, _ = buffer_text.split(" ", 1)
        cmd = _infer_slash_lookup(head)
        if cmd is not None and cmd.takes_value:
            return None, (
                f":sparkles: [infer.preview]{cmd.name}[/] "
                f"[infer.status]{_rich_escape(cmd.value_hint)}[/]"
            )
        return None, ":warning: [infer.status]No completion available after value.[/]"

    candidates = _infer_slash_candidates(buffer_text)
    if not candidates:
        return None, (
            f":warning: [infer.error]No command matches[/] "
            f"{_rich_escape(repr(buffer_text))}"
        )
    if len(candidates) == 1:
        cmd, match = candidates[0]
        suffix = " " if cmd.takes_value else ""
        completion = match + suffix
        if completion == buffer_text:
            hint = (
                f" [infer.status]{_rich_escape(cmd.value_hint)}[/]"
                if cmd.value_hint
                else ""
            )
            return None, (
                f":sparkles: [infer.preview]{match}[/] "
                f"[infer.status]{_rich_escape(cmd.description)}[/]{hint}"
            )
        return completion, (
            f":sparkles: [infer.preview]Completed:[/] {match} "
            f"[infer.status]{_rich_escape(cmd.description)}[/]"
        )

    names = [match for _, match in candidates]
    common = os.path.commonprefix(names)
    listing = "  ".join(names)
    if len(common) > len(buffer_text):
        return common, f":sparkles: [infer.preview]Options:[/] {_rich_escape(listing)}"
    return None, f":sparkles: [infer.preview]Options:[/] {_rich_escape(listing)}"


def apply_infer_setting_command(
    message: str, settings: "InferChatSettings"
) -> tuple["InferChatSettings", str] | None:
    """Handle ``/temp 0.8``-style commands.

    Returns ``(new_settings, status_markup)`` or ``None`` if ``message`` is not a
    value-setter slash command. ``new_settings`` equals the input when the value
    was omitted (query form) or invalid.
    """
    if not message.startswith("/"):
        return None
    parts = message.split(None, 1)
    head = parts[0].lower()
    cmd = _infer_slash_lookup(head)
    if cmd is None or cmd.setting_field is None:
        return None
    raw_value = parts[1].strip() if len(parts) > 1 else ""
    field = cmd.setting_field
    if not raw_value:
        current = getattr(settings, field)
        return settings, (
            f":sparkles: [infer.system]{field}[/] = "
            f"[infer.accent]{_rich_escape(str(current))}[/]  "
            f"[infer.status](pass a value to update, e.g. {cmd.name} {cmd.value_hint})[/]"
        )
    parser = cmd.value_type or str
    try:
        new_value: Any = parser(raw_value)
    except (TypeError, ValueError):
        return settings, (
            f":warning: [infer.error]Invalid[/] {cmd.name} "
            f"[infer.error]value[/] {_rich_escape(repr(raw_value))}"
        )
    updated = replace(settings, **{field: new_value})
    return updated, (
        f":white_check_mark: [infer.system]{field} =[/] "
        f"[infer.accent]{_rich_escape(str(new_value))}[/]"
    )


def render_infer_help_table() -> Table:
    table = Table(title=":sparkles: Commands", box=ROUNDED, expand=False)
    table.add_column("Command", style="infer.accent", no_wrap=True)
    table.add_column("Description")
    for cmd in INFER_SLASH_COMMANDS:
        table.add_row(cmd.display_name(), cmd.description)
    table.add_row("Tab", "Complete /command, accept inline autocomplete, or preview/insert next token")
    return table


def render_infer_show_panel(
    console: Console,
    context: "InferRuntimeContext",
    settings: "InferChatSettings",
    *,
    mode: str,
) -> None:
    graph_name = str(getattr(context.graph, "name", context.graph_path.stem))
    runtime = infer_graph_template_runtime(context.graph) or "unknown"
    info = Table.grid(padding=(0, 2))
    info.add_column(style="infer.accent", justify="right", no_wrap=True)
    info.add_column(overflow="fold")
    if context.generation_backend == "parameter_golf":
        info.add_row(":brain: Checkpoint", context.graph_path.name)
        info.add_row(":page_facing_up: Checkpoint path", str(context.graph_path))
        info.add_row(":floppy_disk: Format", PARAMETER_GOLF_CHECKPOINT_FORMAT)
        info.add_row(":gear: Runtime", f"{context.amp_name} on {context.device}")
    else:
        info.add_row(":brain: Graph", f"{graph_name}")
        info.add_row(":page_facing_up: Graph path", str(context.graph_path))
        info.add_row(":floppy_disk: Weights", str(context.resolved_weights_path))
        info.add_row(":gear: Runtime", runtime)
    info.add_row(":abc: Tokenizer", context.tokenizer_name or "unavailable")
    info.add_row(":page_with_curl: Raw-text encoding", context.raw_text_encoding_name)
    info.add_row(":card_index: Dataset alias", context.dataset_alias or "—")
    info.add_row(":straight_ruler: Context window", str(context.context_window))
    info.add_row(":compass: Mode", f"[infer.accent]{mode}[/]")
    console.print(
        Panel(
            info,
            title=":eye: Current session",
            border_style="infer.accent",
            box=ROUNDED,
            padding=(1, 2),
        )
    )
    console.print(render_infer_settings_table(settings))


def _render_infer_input_line(
    console: Console,
    prompt_label: str,
    buffer_text: str,
    cursor: int,
    status: str,
    ghost_text: str = "",
    previous_rows: int = 0,
    previous_cursor_row: int = 0,
) -> tuple[int, int]:
    line1 = Text()
    line1.append(prompt_label, style="infer.user")
    line1.append(_rich_escape(buffer_text[:cursor]))
    line1.append(INFER_INPUT_CURSOR_MARK)
    line1.append(_rich_escape(buffer_text[cursor:]))
    if ghost_text:
        line1.append(_rich_escape(ghost_text), style="infer.ghost")
    with console.capture() as cap1:
        console.print(line1, end="")
    marked_line_ansi = cap1.get()
    line1_ansi, cursor_row, cursor_col = _resolve_infer_input_cursor(marked_line_ansi)
    status_text = Text.from_markup(status, emoji=True)
    with console.capture() as cap2:
        console.print(status_text, end="")
    status_ansi = cap2.get()
    rendered_rows = len((line1_ansi + "\n" + status_ansi).split("\n"))
    _clear_rendered_infer_input(previous_rows, previous_cursor_row)
    sys.stdout.write(_tty_newlines(line1_ansi) + "\r\n" + _tty_newlines(status_ansi))
    rows_below_cursor = max(0, rendered_rows - cursor_row - 1)
    sys.stdout.write("\r")
    if rows_below_cursor:
        sys.stdout.write(f"\033[{rows_below_cursor}A")
    if cursor_col > 0:
        sys.stdout.write(f"\033[{cursor_col}C")
    sys.stdout.flush()
    return rendered_rows, cursor_row


def _clear_rendered_infer_input(previous_rows: int, previous_cursor_row: int) -> None:
    rows = max(0, int(previous_rows))
    if rows <= 0:
        sys.stdout.write("\r")
        return
    cursor_row = max(0, min(int(previous_cursor_row), rows - 1))
    sys.stdout.write("\r")
    if cursor_row:
        sys.stdout.write(f"\033[{cursor_row}A")
    for idx in range(rows):
        sys.stdout.write("\033[2K")
        if idx < rows - 1:
            sys.stdout.write("\r\n")
    if rows > 1:
        sys.stdout.write(f"\033[{rows - 1}A")
    sys.stdout.write("\r")


def _tty_newlines(text: str) -> str:
    return text.replace("\n", "\r\n")


def _resolve_infer_input_cursor(marked_ansi: str) -> tuple[str, int, int]:
    lines = marked_ansi.split("\n")
    for row_idx, line in enumerate(lines):
        marker_idx = line.find(INFER_INPUT_CURSOR_MARK)
        if marker_idx >= 0:
            before = ANSI_ESCAPE_RE.sub("", line[:marker_idx])
            return (
                marked_ansi.replace(INFER_INPUT_CURSOR_MARK, ""),
                row_idx,
                cell_len(before),
            )
    return marked_ansi, 0, 0


def _commit_infer_input_line(
    console: Console,
    prompt_label: str,
    buffer_text: str,
    *,
    previous_rows: int = 0,
    previous_cursor_row: int = 0,
) -> None:
    line = Text()
    line.append(prompt_label, style="infer.user")
    line.append(_rich_escape(buffer_text))
    with console.capture() as cap:
        console.print(line, end="")
    _clear_rendered_infer_input(previous_rows, previous_cursor_row)
    sys.stdout.write(_tty_newlines(cap.get()) + "\r\n\033[2K\r")
    sys.stdout.flush()


def run_infer_generation_with_spinner(
    console: Console,
    context: "InferRuntimeContext",
    *,
    prompt_ids: Sequence[int],
    prompt_text: str,
    settings: "InferChatSettings",
) -> tuple[dict[str, Any], dict[str, Any]]:
    log_every = int(getattr(context.args, "log_every", 0) or 0)
    progress: list[str] = []
    spinner = Spinner(
        "dots",
        text=Text.from_markup(":brain: [infer.assistant]thinking...[/]", emoji=True),
    )

    def _log(msg: str) -> None:
        progress.append(str(msg))
        tail = "\n".join(progress[-3:])
        spinner.update(
            text=Text.from_markup(
                ":brain: [infer.assistant]thinking...[/]\n"
                f"[infer.status]{_rich_escape(tail)}[/]",
                emoji=True,
            )
        )

    log_cb = _log if log_every > 0 else None
    with Live(spinner, console=console, refresh_per_second=12, transient=True):
        return build_infer_generation(
            context,
            prompt_ids=prompt_ids,
            prompt_text=prompt_text,
            settings=settings,
            log=log_cb,
        )


def print_infer_show(context: InferRuntimeContext, settings: InferChatSettings, *, mode: str) -> None:
    runtime = infer_graph_template_runtime(context.graph) or "unknown"
    graph_name = str(getattr(context.graph, "name", context.graph_path.stem))
    if context.generation_backend == "parameter_golf":
        print(f"Checkpoint: {context.graph_path}")
        print(f"Checkpoint format: {PARAMETER_GOLF_CHECKPOINT_FORMAT}")
        print(f"Runtime: {context.amp_name} on {context.device}")
    else:
        print(f"Graph: {context.graph_path}")
        print(f"Weights: {context.resolved_weights_path}")
        print(f"Graph name: {graph_name}")
        print(f"Runtime: {runtime}")
    print(f"Tokenizer: {context.tokenizer_name or 'unavailable'}")
    print(f"Raw-text encoding: {context.raw_text_encoding_name}")
    print(f"Dataset alias: {context.dataset_alias}")
    print(f"Context window: {context.context_window}")
    print(f"Mode: {mode}")
    print(
        "Settings: "
        f"top_k={settings.top_k} top_p={settings.top_p:g} "
        f"temperature={settings.temperature:g} max_new_tokens={settings.max_new_tokens} "
        f"repetition_penalty={settings.repetition_penalty:g}"
    )


def print_infer_result(
    result: dict[str, Any],
    *,
    tokenizer,
    semantic_extras: dict[str, Any] | None = None,
) -> None:
    print(f"Generated token ids: {result['generated_token_ids']}")
    print(f"All token ids: {result['all_token_ids']}")
    if semantic_extras:
        print(f"Semantic targets: {semantic_extras['semantic_targets']}")
        if semantic_extras.get("semantic_router_vecs") is not None:
            print(f"Semantic router vecs: {semantic_extras['semantic_router_vecs']}")
        if semantic_extras.get("semantic_overrides"):
            print(f"Semantic topic overrides: {semantic_extras['semantic_overrides']}")
    if tokenizer is not None:
        print("Generated text:")
        print(result["generated_text"])
        print("Full text:")
        print(result["full_text"])


def infer_preview_seed(
    *,
    base_seed: int,
    prompt_ids: Sequence[int],
    settings: InferChatSettings,
    mode: str,
) -> int:
    digest = hashlib.sha256()
    digest.update(str(base_seed).encode("ascii"))
    digest.update(mode.encode("utf-8"))
    digest.update(str(infer_settings_signature(settings)).encode("ascii"))
    for token_id in prompt_ids:
        digest.update(f",{int(token_id)}".encode("ascii"))
    seed = int.from_bytes(digest.digest()[:8], "big") % (2**63 - 1)
    return seed if seed > 0 else 1


def infer_preview_display(tokenizer, token_id: int) -> tuple[str, str, bool]:
    token_text = ""
    if tokenizer is not None:
        try:
            token_text = decode_tokens(tokenizer, [int(token_id)])
        except ValueError:
            token_text = ""
    display_text = token_text or describe_token(tokenizer, int(token_id))
    display_text = display_text.encode("unicode_escape").decode("ascii")
    insertable = bool(token_text) and not any(ord(ch) < 32 or ch == "\x7f" for ch in token_text)
    return token_text, display_text, insertable


def trim_infer_autocomplete_words(text: str, word_count: int) -> str:
    if word_count <= 0:
        return ""
    matches = list(INFER_AUTOCOMPLETE_WORD_RE.finditer(text))
    if not matches:
        return ""
    end = matches[min(word_count, len(matches)) - 1].end()
    return text[:end]


def infer_inline_autocomplete_display(text: str) -> tuple[str, str, bool]:
    display_text = text.encode("unicode_escape").decode("ascii")
    insertable = bool(text) and not any(ord(ch) < 32 or ch == "\x7f" for ch in text)
    return text, display_text, insertable


def build_infer_inline_autocomplete(
    context: InferRuntimeContext,
    *,
    settings: InferChatSettings,
    mode: str,
    history: Sequence[tuple[str, str]],
    buffer_text: str,
) -> tuple[InferInlineAutocomplete | None, int]:
    word_count = int(settings.autocomplete_words)
    if word_count <= 0 or not buffer_text.strip() or buffer_text.startswith("/"):
        return None, 0
    prompt_text, prompt_ids, dropped_turns = resolve_infer_chat_prompt(
        context,
        mode=mode,
        history=history,
        draft=buffer_text,
        include_assistant_prompt=False,
    )
    preview_settings = replace(
        settings,
        max_new_tokens=max(1, min(INFER_AUTOCOMPLETE_MAX_TOKENS, word_count * 8)),
    )
    preview_generator = torch.Generator(device=context.device.type)
    preview_generator.manual_seed(
        infer_preview_seed(
            base_seed=int(getattr(context.args, "seed", 1337)),
            prompt_ids=prompt_ids,
            settings=preview_settings,
            mode=f"{mode}:inline",
        )
    )
    result, _extras = build_infer_generation(
        context,
        prompt_ids=prompt_ids,
        prompt_text=prompt_text,
        settings=preview_settings,
        generator=preview_generator,
        log=None,
    )
    prediction_text = trim_infer_autocomplete_words(
        str(result.get("generated_text", "") or ""),
        word_count,
    )
    token_text, display_text, insertable = infer_inline_autocomplete_display(prediction_text)
    if not insertable:
        return None, dropped_turns
    return (
        InferInlineAutocomplete(
            text=token_text,
            display_text=display_text,
            insertable=insertable,
            buffer_snapshot=buffer_text,
            mode=mode,
            settings_signature=infer_settings_signature(settings),
        ),
        dropped_turns,
    )


def build_infer_autocomplete_preview(
    context: InferRuntimeContext,
    *,
    settings: InferChatSettings,
    mode: str,
    history: Sequence[tuple[str, str]],
    buffer_text: str,
) -> tuple[InferAutocompletePreview, int]:
    prompt_text, prompt_ids, dropped_turns = resolve_infer_chat_prompt(
        context,
        mode=mode,
        history=history,
        draft=buffer_text,
        include_assistant_prompt=False,
    )
    preview_settings = InferChatSettings(
        top_k=settings.top_k,
        top_p=settings.top_p,
        temperature=settings.temperature,
        max_new_tokens=1,
        repetition_penalty=settings.repetition_penalty,
    )
    preview_generator = torch.Generator(device=context.device.type)
    preview_generator.manual_seed(
        infer_preview_seed(
            base_seed=int(getattr(context.args, "seed", 1337)),
            prompt_ids=prompt_ids,
            settings=preview_settings,
            mode=mode,
        )
    )
    result, _extras = build_infer_generation(
        context,
        prompt_ids=prompt_ids,
        prompt_text=prompt_text,
        settings=preview_settings,
        generator=preview_generator,
        log=None,
    )
    if not result["generated_token_ids"]:
        raise RuntimeError("Autocomplete preview produced no token.")
    token_id = int(result["generated_token_ids"][0])
    token_text, display_text, insertable = infer_preview_display(context.tokenizer, token_id)
    return (
        InferAutocompletePreview(
            token_id=token_id,
            token_text=token_text,
            display_text=display_text,
            insertable=insertable,
            buffer_snapshot=buffer_text,
            mode=mode,
            settings_signature=infer_settings_signature(settings),
        ),
        dropped_turns,
    )


def _read_infer_escape_sequence(fd: int) -> str:
    second = os.read(fd, 1)
    if not second:
        return "escape"
    if second == b"[":
        third = os.read(fd, 1)
        if third == b"3":
            fourth = os.read(fd, 1)
            if fourth == b"~":
                return "delete"
        mapping = {
            b"A": "up",
            b"B": "down",
            b"C": "right",
            b"D": "left",
            b"H": "home",
            b"F": "end",
        }
        return mapping.get(third, "escape")
    return "escape"


def _read_infer_tty_key(fd: int) -> str:
    first = os.read(fd, 1)
    if not first:
        return "eof"
    if first == b"\x1b":
        return _read_infer_escape_sequence(fd)
    b0 = first[0]
    if b0 < 0x80:
        try:
            return first.decode("utf-8")
        except UnicodeDecodeError:
            return ""
    if (b0 & 0xE0) == 0xC0:
        n_follow = 1
    elif (b0 & 0xF0) == 0xE0:
        n_follow = 2
    elif (b0 & 0xF8) == 0xF0:
        n_follow = 3
    else:
        return ""
    buf = bytearray(first)
    for _ in range(n_follow):
        nxt = os.read(fd, 1)
        if not nxt or (nxt[0] & 0xC0) != 0x80:
            return ""
        buf.append(nxt[0])
    try:
        return buf.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def read_infer_chat_line(
    context: InferRuntimeContext,
    *,
    settings: InferChatSettings,
    mode: str,
    history: Sequence[tuple[str, str]],
    console: Console | None = None,
) -> str | None:
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    buffer_text = ""
    cursor = 0
    pending_preview: InferAutocompletePreview | None = None
    pending_inline: InferInlineAutocomplete | None = None
    status = INFER_IDLE_STATUS
    prompt_label = "You "
    line_console = console if console is not None else make_infer_console()
    render_rows = 0
    render_cursor_row = 0

    def clear_pending_preview() -> None:
        nonlocal pending_preview
        pending_preview = None

    def clear_pending_inline() -> None:
        nonlocal pending_inline
        pending_inline = None

    def active_inline_autocomplete() -> InferInlineAutocomplete | None:
        if (
            pending_inline is not None
            and pending_inline.buffer_snapshot == buffer_text
            and pending_inline.mode == mode
            and pending_inline.settings_signature == infer_settings_signature(settings)
        ):
            return pending_inline
        return None

    def idle_status_for_buffer() -> str:
        if cursor == len(buffer_text):
            slash_status = infer_slash_status_for_buffer(buffer_text)
            if slash_status is not None:
                return slash_status
        return INFER_IDLE_STATUS

    def refresh_inline_autocomplete() -> None:
        nonlocal pending_inline, status
        clear_pending_inline()
        status = idle_status_for_buffer()
        if (
            int(settings.autocomplete_words) <= 0
            or cursor != len(buffer_text)
            or buffer_text.startswith("/")
            or not buffer_text.strip()
        ):
            return
        try:
            suggestion, dropped_turns = build_infer_inline_autocomplete(
                context,
                settings=settings,
                mode=mode,
                history=history,
                buffer_text=buffer_text,
            )
        except Exception as exc:
            status = f":warning: [infer.error]Autocomplete unavailable:[/] {_rich_escape(str(exc))}"
            return
        if suggestion is None:
            return
        pending_inline = suggestion
        display_repr = _rich_escape(repr(suggestion.display_text))
        trimmed = f" (dropped {dropped_turns} old turn{'s' if dropped_turns != 1 else ''})" if dropped_turns else ""
        status = (
            f":sparkles: [infer.preview]Inline autocomplete[/] "
            f"{display_repr} "
            f"[infer.status]\\[Tab accepts]{_rich_escape(trimmed)}[/]"
        )

    def render() -> None:
        nonlocal render_rows, render_cursor_row
        inline = active_inline_autocomplete()
        render_rows, render_cursor_row = _render_infer_input_line(
            line_console,
            prompt_label,
            buffer_text,
            cursor,
            status,
            ghost_text=inline.text if inline is not None else "",
            previous_rows=render_rows,
            previous_cursor_row=render_cursor_row,
        )

    def clear_rendered_input_and_break_line() -> None:
        nonlocal render_rows, render_cursor_row
        _clear_rendered_infer_input(render_rows, render_cursor_row)
        render_rows = 0
        render_cursor_row = 0
        sys.stdout.write("\r\n")
        sys.stdout.flush()

    tty_module.setraw(fd)
    try:
        render()
        while True:
            key = _read_infer_tty_key(fd)
            if key in {"\r", "\n"}:
                _commit_infer_input_line(
                    line_console,
                    prompt_label,
                    buffer_text,
                    previous_rows=render_rows,
                    previous_cursor_row=render_cursor_row,
                )
                return buffer_text
            if key == "\x03":
                raise KeyboardInterrupt
            if key == "eof":
                if not buffer_text:
                    clear_rendered_input_and_break_line()
                    return None
                continue
            if key == "\x04":
                if not buffer_text:
                    clear_rendered_input_and_break_line()
                    return None
                if cursor < len(buffer_text):
                    buffer_text = buffer_text[:cursor] + buffer_text[cursor + 1:]
                clear_pending_preview()
                refresh_inline_autocomplete()
                render()
                continue
            if key in {"\x7f", "\b"}:
                if cursor > 0:
                    buffer_text = buffer_text[:cursor - 1] + buffer_text[cursor:]
                    cursor -= 1
                    clear_pending_preview()
                refresh_inline_autocomplete()
                render()
                continue
            if key == "left":
                cursor = max(0, cursor - 1)
                clear_pending_preview()
                clear_pending_inline()
                status = idle_status_for_buffer()
                render()
                continue
            if key == "right":
                cursor = min(len(buffer_text), cursor + 1)
                clear_pending_preview()
                clear_pending_inline()
                status = idle_status_for_buffer()
                render()
                continue
            if key == "home" or key == "\x01":
                cursor = 0
                clear_pending_preview()
                clear_pending_inline()
                status = idle_status_for_buffer()
                render()
                continue
            if key == "end" or key == "\x05":
                cursor = len(buffer_text)
                clear_pending_preview()
                clear_pending_inline()
                status = idle_status_for_buffer()
                render()
                continue
            if key == "delete":
                if cursor < len(buffer_text):
                    buffer_text = buffer_text[:cursor] + buffer_text[cursor + 1:]
                clear_pending_preview()
                refresh_inline_autocomplete()
                render()
                continue
            if key == "\t":
                if cursor != len(buffer_text):
                    clear_pending_preview()
                    status = ":warning: [infer.status]Autocomplete preview only works at the end of the line.[/]"
                    render()
                    continue
                if buffer_text.startswith("/"):
                    clear_pending_preview()
                    clear_pending_inline()
                    new_buffer, slash_status = complete_infer_slash_command(buffer_text)
                    if new_buffer is not None:
                        buffer_text = new_buffer
                        cursor = len(buffer_text)
                    status = slash_status
                    render()
                    continue
                if int(settings.autocomplete_words) > 0:
                    inline = active_inline_autocomplete()
                    if inline is not None:
                        display_repr = _rich_escape(repr(inline.display_text))
                        buffer_text += inline.text
                        cursor = len(buffer_text)
                        clear_pending_inline()
                        clear_pending_preview()
                        status = f":sparkles: [infer.preview]Accepted inline autocomplete[/] {display_repr}"
                        render()
                        continue
                    refresh_inline_autocomplete()
                    render()
                    continue
                if pending_preview is not None and pending_preview.buffer_snapshot == buffer_text and pending_preview.mode == mode and pending_preview.settings_signature == infer_settings_signature(settings):
                    display_repr = _rich_escape(repr(pending_preview.display_text))
                    if pending_preview.insertable:
                        buffer_text += pending_preview.token_text
                        cursor = len(buffer_text)
                        status = f":sparkles: [infer.preview]Inserted preview token[/] {display_repr}"
                    else:
                        status = f":warning: [infer.status]Preview token {display_repr} cannot be inserted safely.[/]"
                    clear_pending_preview()
                    render()
                    continue
                if not buffer_text and mode == "stateless":
                    status = ":pencil2: [infer.status]Type a prompt before requesting an autocomplete preview.[/]"
                    render()
                    continue
                try:
                    preview, dropped_turns = build_infer_autocomplete_preview(
                        context,
                        settings=settings,
                        mode=mode,
                        history=history,
                        buffer_text=buffer_text,
                    )
                    pending_preview = preview
                    preview_action = "press Tab again to insert" if preview.insertable else "preview only"
                    trimmed = f" (dropped {dropped_turns} old turn{'s' if dropped_turns != 1 else ''})" if dropped_turns else ""
                    display_repr = _rich_escape(repr(preview.display_text))
                    trimmed_escaped = _rich_escape(trimmed)
                    status = (
                        f":sparkles: [infer.preview]Preview[/] "
                        f"{display_repr} "
                        f"[infer.status]\\[{preview_action}]{trimmed_escaped}[/]"
                    )
                except Exception as exc:
                    clear_pending_preview()
                    status = f":warning: [infer.error]Autocomplete unavailable:[/] {_rich_escape(str(exc))}"
                render()
                continue
            if not key or ord(key[0]) < 32:
                clear_pending_preview()
                clear_pending_inline()
                render()
                continue
            buffer_text = buffer_text[:cursor] + key + buffer_text[cursor:]
            cursor += len(key)
            clear_pending_preview()
            refresh_inline_autocomplete()
            render()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)


def run_infer_chat_session(
    context: InferRuntimeContext,
    *,
    initial_prompt_text: str = "",
    initial_prompt_tokens: str = "",
) -> int:
    if context.tokenizer is None:
        raise RuntimeError("Interactive infer chat requires a tokenizer-capable graph export.")

    settings = infer_settings_from_args(context.args)
    mode = "stateless"
    history: list[tuple[str, str]] = []
    turn_idx = 0

    console = make_infer_console()
    render_infer_banner(console, context, settings, mode)
    console.print(
        ":white_check_mark: [infer.system]Ready. Type [/]"
        "[infer.accent]/help[/][infer.system] for commands.[/]"
    )

    def respond_to_user(user_text: str, *, prompt_tokens: str = "") -> None:
        nonlocal history, turn_idx
        if mode == "transcript" and not prompt_tokens.strip():
            prompt_text, prompt_ids, dropped_turns = resolve_infer_chat_prompt(
                context,
                mode=mode,
                history=history,
                draft=user_text,
                include_assistant_prompt=True,
            )
        else:
            prompt_text, prompt_ids = infer_prompt_source(
                prompt=user_text,
                prompt_tokens=prompt_tokens,
                tokenizer=context.tokenizer,
            )
            dropped_turns = 0
        result, extras = run_infer_generation_with_spinner(
            console,
            context,
            prompt_ids=prompt_ids,
            prompt_text=prompt_text,
            settings=settings,
        )
        turn_idx += 1
        if dropped_turns:
            console.print(
                f":scissors: [infer.system]trimmed {dropped_turns} oldest turn"
                f"{'s' if dropped_turns != 1 else ''} to fit context[/]"
            )
        display_user = user_text if user_text else f"[tokens] {prompt_tokens}"
        render_infer_turn_panel(console, "user", turn_idx, display_user)
        response_text = str(result.get("generated_text", "") or "").strip("\n")
        if response_text:
            render_infer_turn_panel(console, "assistant", turn_idx, response_text)
        else:
            console.print(
                Panel(
                    Text(str(result["generated_token_ids"])),
                    title=f":robot: Assistant  [dim]#{turn_idx}[/]  [infer.status](tokens)[/]",
                    title_align="left",
                    border_style="infer.assistant",
                    box=ROUNDED,
                    padding=(0, 1),
                )
            )
        if extras.get("semantic_overrides"):
            console.print(
                Panel(
                    Text(str(extras["semantic_overrides"])),
                    title=":dna: semantic overrides",
                    border_style="infer.accent",
                    box=ROUNDED,
                    padding=(0, 1),
                )
            )
        if mode == "transcript" and not prompt_tokens.strip():
            history.append((user_text, str(result.get("generated_text", "") or "")))

    if initial_prompt_text or initial_prompt_tokens.strip():
        respond_to_user(initial_prompt_text, prompt_tokens=initial_prompt_tokens)

    while True:
        raw = read_infer_chat_line(
            context,
            settings=settings,
            mode=mode,
            history=history,
            console=console,
        )
        if raw is None:
            return 0
        message = raw.strip()
        if not message:
            continue
        if message in {"/exit", "/quit"}:
            console.print(":wave: [infer.system]Bye.[/]")
            return 0
        if message == "/help":
            console.print(render_infer_help_table())
            continue
        if message == "/settings":
            settings = edit_infer_settings(settings)
            console.clear()
            render_infer_banner(console, context, settings, mode)
            console.print(":white_check_mark: [infer.system]Updated settings:[/]")
            console.print(render_infer_settings_table(settings))
            continue
        if message.startswith("/mode "):
            requested_mode = message.split(None, 1)[1].strip().lower()
            if requested_mode not in INFER_CHAT_MODES:
                console.print(
                    ":warning: [infer.error]Usage:[/] /mode stateless|transcript"
                )
                continue
            mode = requested_mode
            console.print(
                f":compass: [infer.system]Switched to[/] [infer.accent]{mode}[/] [infer.system]mode.[/]"
            )
            continue
        if message == "/show":
            render_infer_show_panel(console, context, settings, mode=mode)
            continue
        if message == "/reset":
            history.clear()
            turn_idx = 0
            console.print(":broom: [infer.system]Transcript history cleared.[/]")
            continue
        if message == "/clear":
            console.clear()
            render_infer_banner(console, context, settings, mode)
            continue
        setter_result = apply_infer_setting_command(message, settings)
        if setter_result is not None:
            settings, setter_status = setter_result
            console.print(Text.from_markup(setter_status, emoji=True))
            continue
        if message.startswith("/"):
            console.print(
                Text.from_markup(
                    f":warning: [infer.error]Unknown command[/] "
                    f"{_rich_escape(message.split()[0])}"
                    f" [infer.status](try /help)[/]",
                    emoji=True,
                )
            )
            continue
        respond_to_user(message)


def base_model_defaults(base_model: str) -> dict[str, Any]:
    if base_model == "llama":
        return dict(LLAMA_DEFAULTS)
    if base_model == "gpt2":
        return dict(GPT2_DEFAULTS)
    return dict(NANOGPT_DEFAULTS)


def recipe_model_defaults(recipe: ComposedRecipe) -> dict[str, Any]:
    defaults = base_model_defaults(recipe.base_model)
    if recipe.base_model == "llama":
        defaults["num_kv_heads"] = defaults.get("num_kv_heads", defaults.get("num_heads", 4))
    if recipe.topology == "moe":
        if recipe.uses_semantic:
            defaults.update(
                {
                    "num_layers": ROUTER_DEFAULTS["num_layers"],
                    "model_dim": ROUTER_DEFAULTS["model_dim"],
                    "num_heads": ROUTER_DEFAULTS["num_heads"],
                    "num_kv_heads": ROUTER_DEFAULTS.get("num_kv_heads"),
                    "mlp_mult": ROUTER_DEFAULTS.get("mlp_mult"),
                    "multiple_of": ROUTER_DEFAULTS.get("multiple_of"),
                    "experts": ROUTER_DEFAULTS["experts"],
                    "top_k": ROUTER_DEFAULTS["top_k"],
                    "rope_base": ROUTER_DEFAULTS.get("rope_base"),
                    "qk_gain_init": ROUTER_DEFAULTS.get("qk_gain_init"),
                    "logit_softcap": ROUTER_DEFAULTS.get("logit_softcap"),
                    "ar_loss_coef": ROUTER_DEFAULTS.get("ar_loss_coef", 1.0),
                    "semantic_align_loss_coef": ROUTER_DEFAULTS.get("semantic_align_loss_coef", 0.5),
                }
            )
        else:
            defaults.update(
                {
                    "num_layers": MIXLLAMA_DEFAULTS["num_layers"],
                    "model_dim": MIXLLAMA_DEFAULTS["model_dim"],
                    "num_heads": MIXLLAMA_DEFAULTS["num_heads"],
                    "num_kv_heads": MIXLLAMA_DEFAULTS.get("num_kv_heads"),
                    "mlp_mult": MIXLLAMA_DEFAULTS.get("mlp_mult"),
                    "multiple_of": MIXLLAMA_DEFAULTS.get("multiple_of"),
                    "experts": MIXLLAMA_DEFAULTS["experts"],
                    "top_k": MIXLLAMA_DEFAULTS["top_k"],
                    "rope_base": MIXLLAMA_DEFAULTS.get("rope_base"),
                    "qk_gain_init": MIXLLAMA_DEFAULTS.get("qk_gain_init"),
                    "logit_softcap": MIXLLAMA_DEFAULTS.get("logit_softcap"),
                }
            )
    if recipe.use_jepa:
        defaults.update(
            {
                "ema_decay": 0.99,
                "ar_loss_coef": 1.0,
                "jepa_loss_coef": 0.25,
            }
        )
        if recipe.uses_semantic:
            defaults["semantic_align_loss_coef"] = defaults.get("semantic_align_loss_coef", 0.5)
    return defaults


def compact_model_defaults(recipe: ComposedRecipe) -> dict[str, Any]:
    defaults = recipe_model_defaults(recipe)
    defaults["num_layers"] = max(2, int(defaults.get("num_layers", 4)) - 1)
    defaults["model_dim"] = max(128, int(defaults.get("model_dim", 256)) - 64)
    defaults["num_heads"] = max(2, int(defaults.get("num_heads", 4)) - 1)
    if recipe.base_model == "llama":
        defaults["num_kv_heads"] = max(2, int(defaults.get("num_kv_heads", defaults["num_heads"])))
    if recipe.topology == "moe" and not recipe.uses_semantic:
        defaults["experts"] = max(4, int(defaults.get("experts", 8)) // 2)
    return defaults


def jepa_reference_defaults(recipe: ComposedRecipe) -> dict[str, Any]:
    defaults = recipe_model_defaults(recipe)
    defaults.update(
        {
            "max_steps": REFERENCE_DEFAULTS["iterations"],
            "warmdown_fraction": REFERENCE_DEFAULTS["warmdown_fraction"],
            "warmup_steps": REFERENCE_DEFAULTS["warmup_steps"],
            "train_batch_tokens": REFERENCE_DEFAULTS["train_batch_tokens"],
            "train_seq_len": REFERENCE_DEFAULTS["train_seq_len"],
            "max_wallclock_seconds": REFERENCE_DEFAULTS["max_wallclock_seconds"],
            "qk_gain_init": REFERENCE_DEFAULTS["qk_gain_init"],
            "vocab_size": REFERENCE_DEFAULTS["vocab_size"],
            "num_layers": REFERENCE_DEFAULTS["num_layers"],
            "num_kv_heads": REFERENCE_DEFAULTS["num_kv_heads"],
            "model_dim": REFERENCE_DEFAULTS["model_dim"],
            "num_heads": REFERENCE_DEFAULTS["num_heads"],
            "mlp_mult": REFERENCE_DEFAULTS["mlp_mult"],
            "rope_base": REFERENCE_DEFAULTS["rope_base"],
            "logit_softcap": REFERENCE_DEFAULTS["logit_softcap"],
            "embed_lr": REFERENCE_DEFAULTS["embed_lr"],
            "head_lr": REFERENCE_DEFAULTS["head_lr"],
            "tied_embed_lr": REFERENCE_DEFAULTS["tied_embed_lr"],
            "matrix_lr": REFERENCE_DEFAULTS["matrix_lr"],
            "scalar_lr": REFERENCE_DEFAULTS["scalar_lr"],
            "muon_momentum": REFERENCE_DEFAULTS["muon_momentum"],
            "muon_backend_steps": REFERENCE_DEFAULTS["muon_backend_steps"],
            "muon_momentum_warmup_start": REFERENCE_DEFAULTS["muon_momentum_warmup_start"],
            "muon_momentum_warmup_steps": REFERENCE_DEFAULTS["muon_momentum_warmup_steps"],
            "beta1": REFERENCE_DEFAULTS["beta1"],
            "beta2": REFERENCE_DEFAULTS["beta2"],
            "adam_eps": REFERENCE_DEFAULTS["adam_eps"],
            "grad_clip_norm": REFERENCE_DEFAULTS["grad_clip_norm"],
            "eval_batch_size": REFERENCE_DEFAULTS["val_batch_size"],
            "val_loss_every": REFERENCE_DEFAULTS["val_loss_every"],
            "train_log_every": REFERENCE_DEFAULTS["train_log_every"],
        }
    )
    return defaults


def model_preset_values(recipe: ComposedRecipe, preset_name: str) -> dict[str, Any]:
    if preset_name == PARAMETER_GOLF_CASEOPS_MODEL_PRESET:
        return dict(PARAMETER_GOLF_CASEOPS_MODEL_VALUES)
    if preset_name == "compact":
        return compact_model_defaults(recipe)
    if preset_name == "jepa_tuned":
        return recipe_model_defaults(recipe)
    if preset_name == "jepa_reference":
        return jepa_reference_defaults(recipe)
    return recipe_model_defaults(recipe)


def apply_recommended_presets(command: str, state: dict[str, Any], explicit: set[str]) -> dict[str, Any]:
    recipe = recipe_from_state(state)
    if "seed" not in explicit and state.get("seed") is None:
        state["seed"] = 1337
    if command == "train" and "run_id" not in explicit and not state.get("run_id"):
        state["run_id"] = str(uuid.uuid4())
    if (
        "dataset" not in explicit
        and "dataset_alias" not in explicit
        and not state.get("dataset")
        and not state.get("pretraining_file")
    ):
        state["dataset"] = "golf1"

    if command == "train":
        if "model_preset" not in explicit and not state.get("model_preset"):
            state["model_preset"] = "jepa_reference" if recipe.use_jepa and recipe.base_model == "llama" and recipe.uses_semantic else "harness_default"
        if state.get("model_preset") == PARAMETER_GOLF_CASEOPS_MODEL_PRESET:
            if "run_preset" not in explicit and not state.get("run_preset"):
                state["run_preset"] = "parameter_golf_10min"
            if "optimizer_preset" not in explicit and not state.get("optimizer_preset"):
                state["optimizer_preset"] = "parameter_golf_muon"
            if "tokenizer" not in explicit and "dataset_variant" not in explicit and not state.get("tokenizer"):
                state["tokenizer"] = "sp8192"
        if "run_preset" not in explicit and not state.get("run_preset"):
            state["run_preset"] = "default"
        if "optimizer_preset" not in explicit and not state.get("optimizer_preset"):
            state["optimizer_preset"] = "evolutionary_balanced" if bool(state.get("evolutionary")) else "gradient_default"

        for key, value in model_preset_values(recipe, str(state["model_preset"])).items():
            if key not in explicit and state.get(key) is None:
                state[key] = value
        for key, value in RUN_PRESET_VALUES[str(state["run_preset"])].items():
            if key not in explicit and state.get(key) is None:
                state[key] = value
        gradient_baseline = OPTIMIZER_PRESET_VALUES["gradient_default"]
        for key, value in gradient_baseline.items():
            if key not in explicit and state.get(key) is None:
                state[key] = value
        for key, value in OPTIMIZER_PRESET_VALUES[str(state["optimizer_preset"])].items():
            if key not in explicit and (state.get(key) is None or state.get(key) == gradient_baseline.get(key)):
                state[key] = value
        if state.get("evolutionary") and "evo_seed" not in explicit and state.get("evo_seed") is None:
            state["evo_seed"] = int(state["seed"])
        if not state.get("output"):
            state["output"] = str(default_inference_weights_artifact(recipe.mode_name()))
    elif command == "infer":
        if not state.get("prompt"):
            state.update(INFER_PROMPT_PRESETS["story"])
        for key, value in INFER_GENERATION_PRESETS["balanced"].items():
            if key not in explicit and state.get(key) is None:
                state[key] = value
        state["graph"] = state.get("graph") or str(default_inference_graph_artifact(recipe.mode_name()))
        if "log_every" not in explicit and state.get("log_every") is None:
            state["log_every"] = 1
        if "seed" not in explicit and state.get("seed") is None:
            state["seed"] = 1337
        if "logits_node" not in explicit and state.get("logits_node") is None:
            state["logits_node"] = "auto"
    else:
        for key, value in EVAL_PRESET_VALUES["default"].items():
            if key not in explicit and state.get(key) is None:
                state[key] = value
        if "seed" not in explicit and state.get("seed") is None:
            state["seed"] = 1337
        if "prompt_suite" not in explicit and state.get("prompt_suite") is None:
            state["prompt_suite"] = "auto"
        if "repetition_penalty" not in explicit and state.get("repetition_penalty") is None:
            state["repetition_penalty"] = 1.0
        if "temperature" not in explicit and state.get("temperature") is None:
            state["temperature"] = 0.8
        if "top_k" not in explicit and state.get("top_k") is None:
            state["top_k"] = 32
        if "log_every" not in explicit and state.get("log_every") is None:
            state["log_every"] = 0
        if "logits_node" not in explicit and state.get("logits_node") is None:
            state["logits_node"] = "auto"
        state["graph"] = state.get("graph") or str(default_inference_graph_artifact(recipe.mode_name()))
        if not state.get("report_path"):
            state["report_path"] = str(default_inference_weights_artifact(recipe.mode_name()).with_suffix(".eval.json"))
    return state


def _format_menu_number(value: int | float) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return f"{value:,}"
    if value.is_integer() and abs(value) >= 1.0:
        return f"{int(value):,}"
    if value != 0.0 and abs(value) < 1e-3:
        mantissa, exponent = f"{value:.0e}".split("e")
        return f"{mantissa}e{int(exponent)}"
    return f"{value:g}"


def _format_count(value: int, unit: str) -> str:
    suffix = unit if abs(value) == 1 else f"{unit}s"
    return f"{_format_menu_number(value)} {suffix}"


def _option_with_value(
    label: str,
    value_summary: str,
    description: str,
    value: Any,
    *,
    recommended: bool = False,
) -> OptionChoice:
    return OptionChoice(f"{label} ({value_summary})", description, value, recommended=recommended)


def _model_preset_summary(recipe: ComposedRecipe, preset_name: str) -> str:
    values = model_preset_values(recipe, preset_name)
    parts = [
        _format_count(int(values.get("num_layers", 4)), "layer"),
        f"d_model {_format_menu_number(int(values.get('model_dim', 256)))}",
        _format_count(int(values.get("num_heads", 4)), "head"),
    ]
    if recipe.topology == "moe":
        experts = values.get("experts")
        if experts is not None:
            parts.append(_format_count(int(experts), "expert"))
        top_k = values.get("top_k")
        if top_k is not None:
            parts.append(f"top-k {_format_menu_number(int(top_k))}")
    return ", ".join(parts)


def _run_preset_summary(preset_name: str) -> str:
    values = RUN_PRESET_VALUES[preset_name]
    parts = [
        _format_count(int(values["max_steps"]), "step"),
        f"{_format_menu_number(int(values['train_batch_tokens']))} tokens/step",
        f"batch {_format_menu_number(int(values['batch_size']))}",
        f"seq {_format_menu_number(int(values['train_seq_len']))}",
    ]
    return ", ".join(parts)


def _optimizer_preset_summary(preset_name: str) -> str:
    values = OPTIMIZER_PRESET_VALUES[preset_name]
    if not values.get("evolutionary"):
        parts = [
            f"profile {values['optimizer_profile']}",
            f"lr {_format_menu_number(float(values['learning_rate']))}",
            f"matrix {_format_menu_number(float(values['matrix_lr']))}",
            f"scalar {_format_menu_number(float(values['scalar_lr']))}",
        ]
        return ", ".join(parts)
    parts = [
        f"pop {_format_menu_number(int(values['evo_population_size']))}",
        f"mut {_format_menu_number(float(values['evo_mutation_rate']))}",
        f"scale {_format_menu_number(float(values['evo_mutation_scale']))}",
        f"xover {_format_menu_number(float(values['evo_crossover_rate']))}",
        f"elite {_format_menu_number(int(values['evo_elite_count']))}",
        f"tournament {_format_menu_number(int(values['evo_tournament_size']))}",
    ]
    return ", ".join(parts)


def training_questionnaire(explicit: set[str]) -> list[Question]:
    return [
        Question(
            "base_model",
            "Choose the base model to start from.",
            lambda _state: [
                OptionChoice("Llama", "Balanced default for rope-based recipes.", "llama", recommended=True),
                OptionChoice("GPT-2", "Absolute-position baseline with GPT-2 tokenizer defaults.", "gpt2"),
                OptionChoice("NanoGPT", "NanoGPT-style baseline with optional bias and dropout knobs.", "nanogpt"),
            ],
            lambda _state, _explicit: "base_model" not in explicit,
        ),
        Question(
            "topology",
            "Choose the top-level topology.",
            lambda _state: [
                OptionChoice("Dense", "Standard dense decoder stack.", "dense", recommended=True),
                OptionChoice("MoE", "Mixture-of-experts decoder stack.", "moe"),
            ],
            lambda _state, _explicit: "topology" not in explicit,
        ),
        Question(
            "router_mode",
            "Choose the MoE router mode.",
            lambda _state: [
                OptionChoice("Standard MoE", "Learned expert gating.", "standard", recommended=True),
                OptionChoice("Semantic MoE Router", "Semantic-target-driven routing.", "semantic"),
            ],
            lambda state, _explicit: state.get("topology") == "moe" and "router_mode" not in explicit,
        ),
        Question(
            "use_jepa",
            "Add JEPA on top of the selected recipe?",
            lambda _state: [
                OptionChoice("Off", "Pure autoregressive training.", False, recommended=True),
                OptionChoice("On", "Add the JEPA objective to the selected recipe.", True),
            ],
            lambda _state, _explicit: "use_jepa" not in explicit,
        ),
        Question(
            "megakernel",
            "Enable the megakernel runtime variant?",
            lambda _state: [
                OptionChoice("Off", "Use the default runtime for the chosen base model.", False, recommended=True),
                OptionChoice("On", "Switch to the megakernel runtime variant.", True),
            ],
            lambda _state, _explicit: "megakernel" not in explicit,
        ),
        Question(
            "dataset",
            "Choose a dataset.",
            lambda _state: [
                OptionChoice("golf1", "Small cached-token dataset for fast iteration.", "golf1", recommended=True),
                OptionChoice("golf10", "Larger cached-token parameter-golf dataset.", "golf10"),
                OptionChoice("shakespeare", "Small raw-text dataset.", "shakespeare"),
                OptionChoice("tinystories", "Larger raw-text TinyStories dataset.", "tinystories"),
            ],
            lambda state, _explicit: (
                "dataset" not in explicit
                and "dataset_alias" not in explicit
                and "tinystories" not in explicit
                and "pretraining_file" not in explicit
                and not state.get("pretraining_file")
            ),
        ),
        Question(
            "tokenizer",
            "Choose the tokenizer.",
            lambda state: tokenizer_choices(recipe_from_state(state), state),
            lambda state, _explicit: (
                state.get("dataset") in CACHED_TOKEN_DATASETS or uses_raw_text_dataset(state)
            ) and not any(key in explicit for key in {"tokenizer", "dataset_variant", "raw_text_encoding_override"}),
        ),
        Question(
            "model_preset",
            "Choose a model preset.",
            lambda state: [
                OptionChoice(
                    "Harness Default",
                    f"Current harness-shaped default for the selected recipe. Enforces {_model_preset_summary(recipe_from_state(state), 'harness_default')}.",
                    "harness_default",
                    recommended=not recipe_from_state(state).use_jepa,
                ),
                OptionChoice(
                    "Compact",
                    f"Smaller shape for faster experiments. Enforces {_model_preset_summary(recipe_from_state(state), 'compact')}.",
                    "compact",
                ),
                OptionChoice(
                    "JEPA Tuned",
                    f"JEPA-oriented defaults for additive JEPA recipes. Enforces {_model_preset_summary(recipe_from_state(state), 'jepa_tuned')}.",
                    "jepa_tuned",
                    recommended=recipe_from_state(state).use_jepa,
                ),
                OptionChoice(
                    "JEPA Reference",
                    f"Reference-scale JEPA semantic preset from the existing harness. Enforces {_model_preset_summary(recipe_from_state(state), 'jepa_reference')}.",
                    "jepa_reference",
                ),
                OptionChoice(
                    "Parameter Golf CaseOps 8192",
                    f"Shape from the supplied lossless-caps Parameter Golf run. Enforces {_model_preset_summary(recipe_from_state(state), PARAMETER_GOLF_CASEOPS_MODEL_PRESET)}.",
                    PARAMETER_GOLF_CASEOPS_MODEL_PRESET,
                ),
            ],
            lambda _state, _explicit: "model_preset" not in explicit,
        ),
        Question(
            "run_preset",
            "Choose a run preset.",
            lambda _state: [
                OptionChoice("Smoke", f"Short sanity run. Enforces {_run_preset_summary('smoke')}.", "smoke"),
                OptionChoice("Default", f"Balanced harness baseline. Enforces {_run_preset_summary('default')}.", "default", recommended=True),
                OptionChoice("Overnight", f"Longer run with a larger token budget. Enforces {_run_preset_summary('overnight')}.", "overnight"),
                OptionChoice("Parameter Golf 10min", f"10-minute Parameter Golf budget. Enforces {_run_preset_summary('parameter_golf_10min')}.", "parameter_golf_10min"),
            ],
            lambda _state, _explicit: "run_preset" not in explicit,
        ),
        Question(
            "optimizer_preset",
            "Choose an optimizer or search preset.",
            lambda state: [
                OptionChoice(
                    "Gradient Default",
                    f"Current parameter-golf gradient baseline. Enforces {_optimizer_preset_summary('gradient_default')}.",
                    "gradient_default",
                    recommended=not bool(state.get("evolutionary")),
                ),
                OptionChoice(
                    "Parameter Golf Muon",
                    f"Muon/Adam rates from the supplied Parameter Golf run. Enforces {_optimizer_preset_summary('parameter_golf_muon')}.",
                    "parameter_golf_muon",
                ),
                OptionChoice(
                    "Evolutionary Lean",
                    f"Smaller population evolutionary search. Enforces {_optimizer_preset_summary('evolutionary_lean')}.",
                    "evolutionary_lean",
                ),
                OptionChoice(
                    "Evolutionary Balanced",
                    f"Balanced population evolutionary search. Enforces {_optimizer_preset_summary('evolutionary_balanced')}.",
                    "evolutionary_balanced",
                    recommended=bool(state.get("evolutionary")),
                ),
                OptionChoice(
                    "Evolutionary Broad",
                    f"Broader population evolutionary search. Enforces {_optimizer_preset_summary('evolutionary_broad')}.",
                    "evolutionary_broad",
                ),
            ],
            lambda _state, _explicit: "optimizer_preset" not in explicit,
        ),
        Question(
            "_action",
            "Start now or open the advanced per-parameter questions?",
            lambda _state: [
                OptionChoice("Start Training", "Accept the recommended preset stack and start.", "start", recommended=True),
                OptionChoice("Advanced Hyperparameters", "Open the relevant per-parameter questions.", "advanced"),
            ],
            lambda _state, _explicit: True,
        ),
        Question(
            "num_layers",
            "Choose the layer count.",
            lambda state: [
                _option_with_value(
                    "Recommended",
                    _format_count(int(model_preset_values(recipe_from_state(state), str(state.get("model_preset") or "harness_default")).get("num_layers", 4)), "layer"),
                    "Use the selected model preset layer count.",
                    int(model_preset_values(recipe_from_state(state), str(state.get("model_preset") or "harness_default")).get("num_layers", 4)),
                    recommended=True,
                ),
                _option_with_value(
                    "Smaller",
                    _format_count(max(2, int(model_preset_values(recipe_from_state(state), str(state.get("model_preset") or "harness_default")).get("num_layers", 4)) - 1), "layer"),
                    "One step smaller for quicker runs.",
                    max(2, int(model_preset_values(recipe_from_state(state), str(state.get("model_preset") or "harness_default")).get("num_layers", 4)) - 1),
                ),
                _option_with_value(
                    "Larger",
                    _format_count(int(model_preset_values(recipe_from_state(state), str(state.get("model_preset") or "harness_default")).get("num_layers", 4)) + 1, "layer"),
                    "One step larger for more capacity.",
                    int(model_preset_values(recipe_from_state(state), str(state.get("model_preset") or "harness_default")).get("num_layers", 4)) + 1,
                ),
                _custom_int("Custom layer count"),
            ],
            lambda state, _explicit: state.get("_action") == "advanced" and "num_layers" not in explicit,
        ),
        Question(
            "model_dim",
            "Choose the model width.",
            lambda state: [
                _option_with_value(
                    "Recommended",
                    f"d_model {_format_menu_number(int(model_preset_values(recipe_from_state(state), str(state.get('model_preset') or 'harness_default')).get('model_dim', 256)))}",
                    "Use the selected model preset width.",
                    int(model_preset_values(recipe_from_state(state), str(state.get("model_preset") or "harness_default")).get("model_dim", 256)),
                    recommended=True,
                ),
                _option_with_value(
                    "Smaller",
                    f"d_model {_format_menu_number(max(128, int(model_preset_values(recipe_from_state(state), str(state.get('model_preset') or 'harness_default')).get('model_dim', 256)) - 64))}",
                    "Smaller width for faster runs.",
                    max(128, int(model_preset_values(recipe_from_state(state), str(state.get("model_preset") or "harness_default")).get("model_dim", 256)) - 64),
                ),
                _option_with_value(
                    "Larger",
                    f"d_model {_format_menu_number(int(model_preset_values(recipe_from_state(state), str(state.get('model_preset') or 'harness_default')).get('model_dim', 256)) + 64)}",
                    "Larger width for more capacity.",
                    int(model_preset_values(recipe_from_state(state), str(state.get("model_preset") or "harness_default")).get("model_dim", 256)) + 64,
                ),
                _custom_int("Custom model dimension"),
            ],
            lambda state, _explicit: state.get("_action") == "advanced" and "model_dim" not in explicit,
        ),
        Question(
            "num_heads",
            "Choose the attention head count.",
            lambda state: [
                _option_with_value(
                    "Recommended",
                    _format_count(int(model_preset_values(recipe_from_state(state), str(state.get("model_preset") or "harness_default")).get("num_heads", 4)), "head"),
                    "Use the selected model preset head count.",
                    int(model_preset_values(recipe_from_state(state), str(state.get("model_preset") or "harness_default")).get("num_heads", 4)),
                    recommended=True,
                ),
                _option_with_value(
                    "Smaller",
                    _format_count(max(2, int(model_preset_values(recipe_from_state(state), str(state.get("model_preset") or "harness_default")).get("num_heads", 4)) - 1), "head"),
                    "Fewer heads for a lighter model.",
                    max(2, int(model_preset_values(recipe_from_state(state), str(state.get("model_preset") or "harness_default")).get("num_heads", 4)) - 1),
                ),
                _option_with_value(
                    "Larger",
                    _format_count(int(model_preset_values(recipe_from_state(state), str(state.get("model_preset") or "harness_default")).get("num_heads", 4)) + 1, "head"),
                    "More heads for higher capacity.",
                    int(model_preset_values(recipe_from_state(state), str(state.get("model_preset") or "harness_default")).get("num_heads", 4)) + 1,
                ),
                _custom_int("Custom head count"),
            ],
            lambda state, _explicit: state.get("_action") == "advanced" and "num_heads" not in explicit,
        ),
        Question(
            "num_kv_heads",
            "Choose the KV head count.",
            lambda state: [
                _option_with_value(
                    "Match heads",
                    _format_count(int(state.get("num_heads", 4)), "KV head"),
                    "Use the same KV head count as num_heads.",
                    {"num_kv_heads": int(state.get("num_heads", 4))},
                    recommended=True,
                ),
                _option_with_value(
                    "Half heads",
                    _format_count(max(1, int(state.get("num_heads", 4)) // 2), "KV head"),
                    "Use grouped-query attention with fewer KV heads.",
                    {"num_kv_heads": max(1, int(state.get("num_heads", 4)) // 2)},
                ),
                _custom_int("Custom KV head count"),
            ],
            lambda state, _explicit: state.get("_action") == "advanced" and recipe_from_state(state).base_model == "llama" and "num_kv_heads" not in explicit,
        ),
        Question(
            "experts",
            "Choose the expert count.",
            lambda state: [
                _option_with_value(
                    "Recommended",
                    _format_count(int(model_preset_values(recipe_from_state(state), str(state.get("model_preset") or "harness_default")).get("experts", 8)), "expert"),
                    "Use the selected model preset expert count.",
                    int(model_preset_values(recipe_from_state(state), str(state.get("model_preset") or "harness_default")).get("experts", 8)),
                    recommended=True,
                ),
                _option_with_value(
                    "More experts",
                    _format_count(int(model_preset_values(recipe_from_state(state), str(state.get("model_preset") or "harness_default")).get("experts", 8)) * 2, "expert"),
                    "Increase expert capacity.",
                    int(model_preset_values(recipe_from_state(state), str(state.get("model_preset") or "harness_default")).get("experts", 8)) * 2,
                ),
                _custom_int("Custom expert count"),
            ],
            lambda state, _explicit: state.get("_action") == "advanced" and recipe_from_state(state).topology == "moe" and not recipe_from_state(state).uses_semantic and "experts" not in explicit,
        ),
        Question(
            "top_k",
            "Choose the active expert count.",
            lambda _state: [
                OptionChoice("2 experts", "Current harness baseline.", 2, recommended=True),
                OptionChoice("4 experts", "Broader routing per token.", 4),
                _custom_int("Custom top-k"),
            ],
            lambda state, _explicit: state.get("_action") == "advanced" and recipe_from_state(state).topology == "moe" and "top_k" not in explicit,
        ),
        Question(
            "ema_decay",
            "Choose the JEPA EMA decay.",
            lambda _state: [
                OptionChoice("0.99", "Current harness baseline.", 0.99, recommended=True),
                OptionChoice("0.995", "Slower EMA target updates.", 0.995),
                OptionChoice("0.999", "Very slow EMA target updates.", 0.999),
                _custom_float("Custom EMA decay"),
            ],
            lambda state, _explicit: state.get("_action") == "advanced" and recipe_from_state(state).use_jepa and "ema_decay" not in explicit,
        ),
        Question(
            "learning_rate",
            "Choose the learning rate.",
            lambda _state: [
                OptionChoice("3e-4", "Current harness baseline.", 3e-4, recommended=True),
                OptionChoice("1e-4", "Conservative training rate.", 1e-4),
                OptionChoice("6e-4", "More aggressive training rate.", 6e-4),
                _custom_float("Custom learning rate"),
            ],
            lambda state, _explicit: state.get("_action") == "advanced" and not bool(state.get("evolutionary")) and "learning_rate" not in explicit,
        ),
        Question(
            "evo_population_size",
            "Choose the evolutionary population size.",
            lambda _state: [
                OptionChoice("24", "Lean evolutionary search.", 24),
                OptionChoice("50", "Balanced evolutionary search.", 50, recommended=True),
                OptionChoice("96", "Broad evolutionary search.", 96),
                _custom_int("Custom population size"),
            ],
            lambda state, _explicit: state.get("_action") == "advanced" and bool(state.get("evolutionary")) and "evo_population_size" not in explicit,
        ),
        Question(
            "max_steps",
            "Choose the total step budget.",
            lambda state: [
                _option_with_value(
                    "Recommended",
                    _format_count(int(RUN_PRESET_VALUES[str(state.get("run_preset") or "default")]["max_steps"]), "step"),
                    "Use the selected run preset step budget.",
                    int(RUN_PRESET_VALUES[str(state.get("run_preset") or "default")]["max_steps"]),
                    recommended=True,
                ),
                _option_with_value(
                    "Shorter",
                    _format_count(max(20, int(RUN_PRESET_VALUES[str(state.get("run_preset") or "default")]["max_steps"]) // 2), "step"),
                    "Half the default step budget.",
                    max(20, int(RUN_PRESET_VALUES[str(state.get("run_preset") or "default")]["max_steps"]) // 2),
                ),
                _option_with_value(
                    "Longer",
                    _format_count(int(RUN_PRESET_VALUES[str(state.get("run_preset") or "default")]["max_steps"]) * 2, "step"),
                    "Double the default step budget.",
                    int(RUN_PRESET_VALUES[str(state.get("run_preset") or "default")]["max_steps"]) * 2,
                ),
                _custom_int("Custom max-steps"),
            ],
            lambda state, _explicit: state.get("_action") == "advanced" and "max_steps" not in explicit,
        ),
        Question(
            "train_batch_tokens",
            "Choose the training token budget per step.",
            lambda state: [
                _option_with_value(
                    "Recommended",
                    f"{_format_menu_number(int(RUN_PRESET_VALUES[str(state.get('run_preset') or 'default')]['train_batch_tokens']))} tokens/step",
                    "Use the selected run preset token budget.",
                    int(RUN_PRESET_VALUES[str(state.get("run_preset") or "default")]["train_batch_tokens"]),
                    recommended=True,
                ),
                _option_with_value(
                    "Smaller",
                    f"{_format_menu_number(max(1024, int(RUN_PRESET_VALUES[str(state.get('run_preset') or 'default')]['train_batch_tokens']) // 2))} tokens/step",
                    "Use half the recommended token budget.",
                    max(1024, int(RUN_PRESET_VALUES[str(state.get("run_preset") or "default")]["train_batch_tokens"]) // 2),
                ),
                _option_with_value(
                    "Larger",
                    f"{_format_menu_number(int(RUN_PRESET_VALUES[str(state.get('run_preset') or 'default')]['train_batch_tokens']) * 2)} tokens/step",
                    "Use double the recommended token budget.",
                    int(RUN_PRESET_VALUES[str(state.get("run_preset") or "default")]["train_batch_tokens"]) * 2,
                ),
                _custom_int("Custom train-batch-tokens"),
            ],
            lambda state, _explicit: state.get("_action") == "advanced" and "train_batch_tokens" not in explicit,
        ),
    ]


def infer_questionnaire(explicit: set[str]) -> list[Question]:
    return [
        question for question in training_questionnaire(explicit)
        if question.key in {"base_model", "topology", "router_mode", "use_jepa", "megakernel", "dataset", "tokenizer", "dataset_variant", "raw_text_encoding_override"}
    ] + [
        Question(
            "graph",
            "Use default composed artifacts or custom paths?",
            lambda _state: [
                OptionChoice("Default artifacts", "Use the recipe-derived graph path and the graph-linked checkpoint.", {"_artifact_mode": "default"}, recommended=True),
                OptionChoice("Custom artifacts", "Enter a custom graph path and an optional weights override.", {"_artifact_mode": "custom"}),
            ],
            lambda _state, _explicit: "graph" not in explicit and "weights" not in explicit,
        ),
        Question(
            "prompt",
            "Choose a prompt preset.",
            lambda _state: [
                OptionChoice("Story opener", "Once upon a time", INFER_PROMPT_PRESETS["story"], recommended=True),
                OptionChoice("Hello", "Simple greeting.", INFER_PROMPT_PRESETS["hello"]),
                OptionChoice("Code snippet", "Short code prompt.", INFER_PROMPT_PRESETS["code"]),
                OptionChoice("Custom...", "Enter your own prompt text.", {}, custom_prompt="Custom prompt text"),
            ],
            lambda _state, _explicit: "prompt" not in explicit and "prompt_tokens" not in explicit,
        ),
        Question(
            "generation_preset",
            "Choose a generation preset.",
            lambda _state: [
                OptionChoice("Balanced", "Current default generation settings.", "balanced", recommended=True),
                OptionChoice("Focused", "Lower-entropy generation.", "focused"),
                OptionChoice("Explore", "Longer, higher-entropy generation.", "explore"),
            ],
            lambda _state, _explicit: True,
        ),
    ]


def eval_questionnaire(explicit: set[str]) -> list[Question]:
    return [
        question for question in training_questionnaire(explicit)
        if question.key in {"base_model", "topology", "router_mode", "use_jepa", "megakernel", "dataset", "tokenizer", "dataset_variant", "raw_text_encoding_override"}
    ] + [
        Question(
            "eval_preset",
            "Choose an evaluation preset.",
            lambda _state: [
                OptionChoice("Default", "Balanced validation and prompt probes.", "default", recommended=True),
                OptionChoice("Smoke", "Quick validation and prompt probes.", "smoke"),
                OptionChoice("Extended", "Longer validation and prompt probes.", "extended"),
            ],
            lambda _state, _explicit: True,
        )
    ]


def maybe_plan(command: str, state: dict[str, Any], explicit: set[str], *, interactive: bool) -> dict[str, Any]:
    working = dict(state)
    normalize_dataset_selector_state(working)
    if command == "train":
        questions = training_questionnaire(explicit)
    elif command == "infer":
        questions = infer_questionnaire(explicit)
    else:
        questions = eval_questionnaire(explicit)

    if interactive:
        working = run_curses_questionnaire(f"nfn {command}", questions, working)
    else:
        for question in questions:
            if not question.visible(working, explicit):
                continue
            if question.key in explicit:
                continue
            choice = pick_recommended_option(question.options_factory(working))
            value = choice.value
            if isinstance(value, dict):
                working.update(value)
            else:
                working[question.key] = value
            normalize_dataset_selector_state(working)

    if command == "infer":
        preset_name = str(working.get("generation_preset") or "balanced")
        working.update({k: v for k, v in INFER_GENERATION_PRESETS[preset_name].items() if k not in explicit and working.get(k) is None})
    elif command == "eval":
        preset_name = str(working.get("eval_preset") or "default")
        working.update({k: v for k, v in EVAL_PRESET_VALUES[preset_name].items() if k not in explicit and working.get(k) is None})

    apply_recommended_presets(command, working, explicit)
    return working


def state_to_cli_args(command: str, state: dict[str, Any]) -> list[str]:
    working = dict(state)
    normalize_dataset_selector_state(working)
    if command != "infer" and not working.get("tokenizer"):
        working["tokenizer"] = selected_tokenizer_name(recipe_from_state(working), working)
    args = [command]
    ordered_keys = [
        "base_model",
        "topology",
        "router_mode",
        "use_jepa",
        "megakernel",
        "dataset",
        "pretraining_file",
        "tokenizer",
        "tokenizer_hf_path",
        "tokenizer_repo_id",
        "tokenizer_remote_root_prefix",
        "tokenizer_repo_type",
        "dataset_alias",
        "dataset_hf_path",
        "dataset_variant",
        "dataset_train_shards",
        "dataset_repo_id",
        "dataset_remote_root_prefix",
        "dataset_train_file",
        "dataset_val_file",
        "model_preset",
        "run_preset",
        "optimizer_preset",
        "run_id",
        "seed",
        "device",
        "amp_dtype",
        "output",
        "graph",
        "weights",
        "checkpoint",
        "checkpoint_tokenizer",
        "checkpoint_log",
        "report_path",
        "max_steps",
        "train_seq_len",
        "batch_size",
        "train_batch_tokens",
        "eval_batches",
        "eval_batch_size",
        "max_new_tokens",
        "temperature",
        "top_k",
        "top_p",
        "prompt",
        "prompt_tokens",
    ]
    for key in ordered_keys:
        if key == "dataset_alias" and working.get("pretraining_file"):
            continue
        value = working.get(key)
        if value in (None, "", False):
            continue
        if key == "use_jepa":
            args.append("--jepa")
        elif key == "megakernel":
            args.append("--megakernel")
        else:
            flag = "--" + key.replace("_", "-")
            args.extend([flag, str(value)])
    return args


def print_resolved_command(command: str, state: dict[str, Any]) -> None:
    argv = state_to_cli_args(command, state)
    print("Resolved command:")
    print("  " + shlex.join(["nfn", *argv]))


def raw_text_tokenizer_name(recipe: ComposedRecipe, state: dict[str, Any]) -> str:
    return selected_tokenizer_name(recipe, state)


def apply_raw_text_vocab_policy(args: argparse.Namespace, recipe: ComposedRecipe) -> None:
    encoding_name = raw_text_tokenizer_name(recipe, vars(args))
    args.tokenizer = encoding_name
    args.raw_text_encoding_override = encoding_name
    args.raw_text_encoding_name = encoding_name
    dataset = dataset_choice_from_state(vars(args))
    if dataset not in RAW_TEXT_DATASETS:
        return
    validate_raw_text_tokenizer_availability(
        encoding_name,
        download_if_missing=bool(getattr(args, "download_if_missing", False)),
        dataset_alias=getattr(args, "dataset_alias", None),
        **tokenizer_download_kwargs_from_args(args),
    )
    expected_vocab_size = raw_text_encoding_vocab_size(encoding_name)
    default_vocab_size = int(model_preset_values(recipe, str(getattr(args, "model_preset", None) or "harness_default")).get("vocab_size", 1024))
    if args.vocab_size is None or int(args.vocab_size) == default_vocab_size:
        args.vocab_size = expected_vocab_size
    elif int(args.vocab_size) != expected_vocab_size:
        raise ValueError(
            f"Raw-text dataset {dataset!r} with tokenizer {encoding_name} requires vocab_size={expected_vocab_size}, "
            f"but received vocab_size={args.vocab_size}."
        )


def build_spec_from_args(args: argparse.Namespace, recipe: ComposedRecipe):
    kwargs: dict[str, Any] = {
        "vocab_size": int(args.vocab_size),
        "num_layers": int(args.num_layers),
        "model_dim": int(args.model_dim),
        "num_heads": int(args.num_heads),
        "logit_softcap": float(args.logit_softcap if args.logit_softcap is not None else 0.0),
    }
    if recipe.use_jepa or recipe.uses_semantic:
        kwargs["ar_loss_coef"] = float(args.ar_loss_coef if args.ar_loss_coef is not None else 1.0)
    if recipe.use_jepa:
        kwargs["ema_decay"] = float(args.ema_decay if args.ema_decay is not None else 0.99)
        kwargs["jepa_loss_coef"] = float(args.jepa_loss_coef if args.jepa_loss_coef is not None else 0.25)
    if recipe.uses_semantic:
        kwargs["semantic_align_loss_coef"] = float(
            args.semantic_align_loss_coef if args.semantic_align_loss_coef is not None else 0.5
        )
        kwargs["semantic_vocab_ref"] = semantic_vocab_ref_for_tokenizer(raw_text_tokenizer_name(recipe, vars(args)))
        kwargs["experimental_semantic_router_vecs"] = bool(getattr(args, "experimental_semantic_router_vecs", False))
    if args.num_kv_heads is not None:
        kwargs["num_kv_heads"] = int(args.num_kv_heads)
    if args.mlp_mult is not None:
        kwargs["mlp_mult"] = float(args.mlp_mult)
    if args.multiple_of is not None:
        kwargs["multiple_of"] = int(args.multiple_of)
    if args.rope_base is not None:
        kwargs["rope_base"] = float(args.rope_base)
    if args.qk_gain_init is not None:
        kwargs["qk_gain_init"] = float(args.qk_gain_init)
    if recipe.topology == "moe":
        kwargs["experts"] = int(args.experts)
        kwargs["top_k"] = int(args.top_k)
    if recipe.base_model == "nanogpt":
        kwargs["bias"] = bool(args.bias)
        kwargs["dropout_p"] = float(args.dropout_p)
    # ── Fine-tuning ────────────────────────────────────────────────────
    if recipe.adapter_type != "none":
        kwargs["adapter_type"] = recipe.adapter_type
        if getattr(args, "lora_rank", None) is not None:
            kwargs["lora_rank"] = int(args.lora_rank)
        if getattr(args, "lora_alpha", None) is not None:
            kwargs["lora_alpha"] = float(args.lora_alpha)
        if getattr(args, "lora_dropout", None) is not None:
            kwargs["lora_dropout"] = float(args.lora_dropout)
        if getattr(args, "lora_targets", None) is not None:
            kwargs["lora_targets"] = str(args.lora_targets)
        if getattr(args, "lora_bias", False):
            kwargs["lora_bias"] = True
        if getattr(args, "qlora_group_size", None) is not None:
            kwargs["qlora_group_size"] = int(args.qlora_group_size)
        if getattr(args, "qlora_compute_dtype", None) is not None:
            kwargs["qlora_compute_dtype"] = str(args.qlora_compute_dtype)
    if recipe.training_mode != "pretrain":
        from neuralfn.config import FineTuneSpec as _FineTuneSpec
        kwargs["finetune"] = _FineTuneSpec(
            objective=recipe.training_mode,
            base_checkpoint=str(getattr(args, "base_checkpoint", "") or ""),
            ref_checkpoint=str(getattr(args, "ref_checkpoint", "") or ""),
            reward_checkpoint=str(getattr(args, "reward_checkpoint", "") or ""),
            adapter_only_save=bool(getattr(args, "adapter_only_save", False)),
            beta=float(getattr(args, "dpo_beta", None) if getattr(args, "dpo_beta", None) is not None else 0.1),
            dpo_loss_type=str(getattr(args, "dpo_loss_type", None) or "sigmoid"),
            kl_coef=float(getattr(args, "kl_coef", None) if getattr(args, "kl_coef", None) is not None else 0.1),
            ppo_clip=float(getattr(args, "ppo_clip", None) if getattr(args, "ppo_clip", None) is not None else 0.2),
            ppo_vf_coef=float(getattr(args, "ppo_vf_coef", None) if getattr(args, "ppo_vf_coef", None) is not None else 0.5),
            ppo_ent_coef=float(getattr(args, "ppo_ent_coef", None) if getattr(args, "ppo_ent_coef", None) is not None else 0.0),
            rollout_length=int(getattr(args, "rollout_length", None) if getattr(args, "rollout_length", None) is not None else 64),
            ppo_epochs_per_rollout=int(getattr(args, "ppo_epochs_per_rollout", None) if getattr(args, "ppo_epochs_per_rollout", None) is not None else 4),
        )
    spec = build_composed_lm_spec(
        base_model=recipe.base_model,
        topology=recipe.topology,
        router_mode=recipe.router_mode,
        use_jepa=recipe.use_jepa,
        runtime=recipe.runtime,
        **kwargs,
    )
    # Override template objective for fine-tuning so the root-graph dispatcher
    # routes into build_sft_root_graph / build_dpo_root_graph / etc.
    if recipe.training_mode != "pretrain":
        spec.template.objective = recipe.training_mode  # type: ignore[assignment]
    return spec


def build_graph_for_training(args: argparse.Namespace, recipe: ComposedRecipe, dataset_name: str):
    spec = build_spec_from_args(args, recipe)
    graph = build_gpt_root_graph(name=recipe.graph_name(), model_spec=spec)
    graph.torch_config = {
        **graph.torch_config,
        "device": args.device,
        "amp_dtype": str(getattr(args, "amp_dtype", None) or "float32"),
    }
    load_dataset_source_into_graph(graph, LoadDatasetRequest(dataset_names=[dataset_name], seq_len=int(args.train_seq_len)))
    apply_sanitized_template_spec(
        graph,
        raw_text_encoding_name=str(getattr(args, "raw_text_encoding_name", "") or ""),
    )
    return graph, spec


def build_trainer_config(args: argparse.Namespace, *, resolved_epochs: int, derived: dict[str, Any]) -> TorchTrainConfig:
    evo_defaults = OPTIMIZER_PRESET_VALUES["evolutionary_balanced"]
    return TorchTrainConfig(
        epochs=resolved_epochs,
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        device=str(args.device),
        amp_dtype=str(getattr(args, "amp_dtype", None) or "float32"),
        max_steps=int(args.max_steps),
        optimizer_profile=str(args.optimizer_profile),
        train_batch_tokens=int(args.train_batch_tokens),
        warmup_steps=int(args.warmup_steps),
        warmdown_fraction=float(args.warmdown_fraction),
        lr_decay_iters=None if args.lr_decay_iters is None else int(args.lr_decay_iters),
        min_lr=None if args.min_lr is None else float(args.min_lr),
        max_wallclock_seconds=float(args.max_wallclock_seconds),
        embed_lr=float(args.embed_lr),
        head_lr=float(args.head_lr),
        tied_embed_lr=float(args.tied_embed_lr),
        matrix_lr=float(args.matrix_lr),
        scalar_lr=float(args.scalar_lr),
        muon_momentum=float(args.muon_momentum),
        muon_backend_steps=int(args.muon_backend_steps),
        muon_momentum_warmup_start=float(args.muon_momentum_warmup_start),
        muon_momentum_warmup_steps=int(args.muon_momentum_warmup_steps),
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        adam_eps=float(args.adam_eps),
        grad_clip_norm=float(args.grad_clip_norm),
        drop_last=bool(derived["drop_last"]),
        respect_epoch_boundaries=bool(derived["respect_epoch_boundaries"]),
        kernel_backend=str(getattr(args, "kernel_backend", None) or "auto"),
        tile_cuda_strict=bool(getattr(args, "tile_cuda_strict", False)),
        tile_cuda_report_path=getattr(args, "tile_cuda_report", None),
        evolutionary=bool(args.evolutionary),
        evo_population_size=int(args.evo_population_size if args.evo_population_size is not None else evo_defaults["evo_population_size"]),
        evo_mutation_rate=float(args.evo_mutation_rate if args.evo_mutation_rate is not None else evo_defaults["evo_mutation_rate"]),
        evo_mutation_scale=float(args.evo_mutation_scale if args.evo_mutation_scale is not None else evo_defaults["evo_mutation_scale"]),
        evo_crossover_rate=float(args.evo_crossover_rate if args.evo_crossover_rate is not None else evo_defaults["evo_crossover_rate"]),
        evo_tournament_size=int(args.evo_tournament_size if args.evo_tournament_size is not None else evo_defaults["evo_tournament_size"]),
        evo_elite_count=int(args.evo_elite_count if args.evo_elite_count is not None else evo_defaults["evo_elite_count"]),
        evo_seed=int(args.seed if args.evo_seed is None else args.evo_seed),
    )


def ensure_train_defaults(args: argparse.Namespace, recipe: ComposedRecipe) -> argparse.Namespace:
    defaults = recipe_model_defaults(recipe)
    for key in (
        "vocab_size",
        "num_layers",
        "model_dim",
        "num_heads",
        "num_kv_heads",
        "mlp_mult",
        "multiple_of",
        "rope_base",
        "qk_gain_init",
        "logit_softcap",
        "experts",
        "top_k",
        "bias",
        "dropout_p",
        "ema_decay",
        "ar_loss_coef",
        "jepa_loss_coef",
        "semantic_align_loss_coef",
    ):
        if getattr(args, key, None) is None and key in defaults:
            setattr(args, key, defaults[key])
    resolve_lr_schedule_defaults(args)
    return args


def log_stage(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def run_train(state: dict[str, Any]) -> int:
    configure_console_logging()
    recipe = recipe_from_state(state)
    args = namespace_from_state("train", state)
    args.dataset_alias = str(getattr(args, "dataset_alias", None) or DEFAULT_DATASET_ALIAS)
    args.run_id = str(getattr(args, "run_id", None) or uuid.uuid4())
    apply_tinystories_dataset_defaults(args)
    resolve_dataset_selector_args(args)
    resolve_pretraining_file_dataset(args)
    ensure_train_defaults(args, recipe)
    apply_raw_text_vocab_policy(args, recipe)
    args.output = str(getattr(args, "output", None) or default_inference_weights_artifact(recipe.mode_name()))

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if args.device != "cuda":
        print("This harness is configured to run on CUDA only.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("CUDA device is not available in this environment.", file=sys.stderr)
        return 1

    dataset_name, dataset_path, dataset_meta = resolve_or_download_dataset(
        args.dataset_alias,
        raw_text_encoding_name=str(getattr(args, "raw_text_encoding_name", "gpt2")),
        **dataset_download_kwargs_from_args(args),
    )
    default_vocab_size = int(
        model_preset_values(recipe, str(getattr(args, "model_preset", None) or "harness_default")).get(
            "vocab_size",
            args.vocab_size,
        )
    )
    dataset_meta = apply_cached_tokenizer_vocab_policy(
        args,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        dataset_meta=dataset_meta,
        default_vocab_size=default_vocab_size,
    )
    estimate_fn = estimate_schedule if recipe.uses_semantic else estimate_text_schedule
    derived = estimate_fn(
        dataset_name,
        seq_len=int(args.train_seq_len),
        batch_size=int(args.batch_size),
        train_batch_tokens=int(args.train_batch_tokens),
        top_k=int(args.top_k) if recipe.uses_semantic else 2,
        template_runtime=recipe.template_runtime,
        device=str(args.device),
        all_train_rows=bool(getattr(args, "all_train_rows", False)),
    ) if recipe.uses_semantic else estimate_fn(
        dataset_name,
        seq_len=int(args.train_seq_len),
        batch_size=int(args.batch_size),
        train_batch_tokens=int(args.train_batch_tokens),
        template_runtime=recipe.template_runtime,
        device=str(args.device),
        all_train_rows=bool(getattr(args, "all_train_rows", False)),
    )
    derived = {**derived}
    derived, resolved_epochs, resolved_max_steps, resolved_lr_decay_iters, resolved_max_wallclock_seconds = resolve_effective_training_schedule(args, derived)
    args.max_steps = resolved_max_steps
    args.lr_decay_iters = resolved_lr_decay_iters
    args.max_wallclock_seconds = resolved_max_wallclock_seconds
    trainer_cfg = build_trainer_config(args, resolved_epochs=resolved_epochs, derived=derived)
    graph, spec = build_graph_for_training(args, recipe, dataset_name)

    print_graph_summary(graph)
    resolved_training_summary = {
        "recipe": {
            "base_model": recipe.base_model,
            "topology": recipe.topology,
            "router_mode": recipe.router_mode,
            "use_jepa": recipe.use_jepa,
            "runtime": recipe.runtime,
            "mode_name": recipe.mode_name(),
        },
        "dataset_alias": args.dataset_alias,
        "artifact_path": args.output,
        "graph_contract": list(graph.input_node_ids),
        "model_spec": sanitized_model_spec_dict(
            spec,
            raw_text_encoding_name=str(getattr(args, "raw_text_encoding_name", "") or ""),
        ),
        "trainer": build_trainer_summary(trainer_cfg),
        "derived_schedule": derived,
    }
    print("Resolved training configuration:")
    print(json.dumps(resolved_training_summary, indent=2, sort_keys=True))

    trainer = TorchTrainer(graph, trainer_cfg)
    on_step, on_epoch = build_progress_logger(
        train_log_every=int(args.train_log_every),
        resolved_epochs=resolved_epochs,
        max_steps=resolved_max_steps,
    )
    interrupted = False
    force_abort = False
    previous_sigint = signal.getsignal(signal.SIGINT)
    output_path = Path(args.output)
    graph_output_path = output_path.with_suffix(".json")
    interrupted_weights_path = output_path.with_name(output_path.stem + ".interrupted" + output_path.suffix)
    interrupted_graph_path = interrupted_weights_path.with_suffix(".json")

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
        losses = trainer.train([], [], on_epoch=on_epoch, on_step=on_step)
        if interrupted:
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
            print(f"Saved checkpoint: {interrupted_weights_path}")
            print(f"Saved graph: {interrupted_graph_path}")
            return 130
        if not losses:
            print("Trainer returned no losses.", file=sys.stderr)
            return 1
        if not all(math.isfinite(float(loss)) for loss in losses):
            print("Encountered non-finite loss.", file=sys.stderr)
            return 1
        log_stage("Training finished. Saving exported artifacts.")
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
        log_stage("Artifacts saved. Starting validation pass.")
        if recipe.uses_semantic:
            val_loss = safe_evaluate_validation_loss(
                lambda: evaluate_semantic_model(
                    graph,
                    dataset_path,
                    device=str(args.device),
                    seq_len=int(args.train_seq_len),
                    batch_size=int(args.eval_batch_size),
                    eval_batches=int(args.eval_batches),
                    encoding_name=str(getattr(args, "raw_text_encoding_name", "gpt2")),
                )
            )
        else:
            val_loss = safe_evaluate_validation_loss(
                lambda: evaluate_text_model(
                    graph,
                    dataset_path,
                    device=str(args.device),
                    seq_len=int(args.train_seq_len),
                    batch_size=int(args.eval_batch_size),
                    eval_batches=int(args.eval_batches),
                    encoding_name=str(getattr(args, "raw_text_encoding_name", "gpt2")),
                )
            )
        log_stage("Validation finished.")
        print("Losses:", [round(float(loss), 6) for loss in losses])
        print(f"Final train loss: {float(losses[-1]):.6f}")
        if math.isfinite(float(val_loss)):
            print(f"Validation loss: {float(val_loss):.6f}")
        else:
            print("Validation loss: skipped")
        print(f"Exported model: {output_path}")
        print(f"Exported graph: {graph_output_path}")
        return 0
    except KeyboardInterrupt:
        trainer.stop()
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
        print(f"Saved checkpoint: {interrupted_weights_path}")
        print(f"Saved graph: {interrupted_graph_path}")
        return 130
    finally:
        signal.signal(signal.SIGINT, previous_sigint)


def build_semantic_generation(
    *,
    graph,
    compiled: CompiledTorchGraph,
    tokenizer,
    prompt_ids: list[int],
    device: torch.device,
    amp_dtype: torch.dtype,
    generator: torch.Generator,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    stop_token: int | None,
    logits_node: str,
    context_window: int,
    sem_targets: torch.Tensor,
    semantic_router_vecs: torch.Tensor | None,
    log_every: int,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    generated = list(prompt_ids)
    resolved_logits_key: str | None = None
    use_amp = autocast_enabled_for(device, amp_dtype)
    with torch.no_grad():
        for step_idx in range(max_new_tokens):
            context_ids = generated[-context_window:]
            tokens = torch.tensor([context_ids], dtype=torch.long, device=device)
            targets = torch.zeros_like(tokens)
            model_inputs = build_semantic_model_inputs(
                graph,
                tokens,
                targets,
                sem_targets,
                semantic_router_vecs=semantic_router_vecs,
            )
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _outputs, trace = compiled.trace(*model_inputs)
            if resolved_logits_key is None:
                resolved_logits_key = find_logits_trace_key(trace, logits_node)
                if log is not None:
                    log(f"Using traced logits node {resolved_logits_key}")
            logits = trace[resolved_logits_key][0]
            next_token = sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                token_history=generated,
                repetition_penalty=repetition_penalty,
                generator=generator,
            )
            generated.append(next_token)
            if log is not None and log_every > 0 and (step_idx == 0 or (step_idx + 1) % log_every == 0 or step_idx + 1 >= max_new_tokens):
                routing_summary = describe_routing(trace)
                suffix = f" {routing_summary}" if routing_summary else ""
                log(f"Generation step {step_idx + 1}/{max_new_tokens}: token={next_token}{suffix}")
            if stop_token is not None and next_token == stop_token:
                break
    generated_tail = generated[len(prompt_ids):]
    return {
        "generated_token_ids": generated_tail,
        "all_token_ids": generated,
        "generated_text": decode_tokens(tokenizer, generated_tail) if tokenizer is not None else "",
        "full_text": decode_tokens(tokenizer, generated) if tokenizer is not None else "",
        "resolved_logits_key": resolved_logits_key,
    }


def run_infer(state: dict[str, Any]) -> int:
    configure_console_logging()
    interactive = bool(state.get("_tty", is_tty()))
    args = namespace_from_state("infer", state)
    try:
        context = build_infer_runtime_context(args, state=state, interactive=interactive)
        if interactive:
            return run_infer_chat_session(
                context,
                initial_prompt_text=str(getattr(args, "prompt", None) or ""),
                initial_prompt_tokens=str(getattr(args, "prompt_tokens", None) or ""),
            )
        if not infer_prompt_was_supplied(args):
            print("Non-interactive infer requires --prompt or --prompt-tokens.", file=sys.stderr)
            return 2
        prompt_text, prompt_ids = infer_prompt_source(
            prompt=str(getattr(args, "prompt", None) or ""),
            prompt_tokens=str(getattr(args, "prompt_tokens", None) or ""),
            tokenizer=context.tokenizer,
        )
        if not prompt_ids:
            print("Prompt resolved to an empty token list.", file=sys.stderr)
            return 1
        settings = infer_settings_from_args(args)
        result, extras = build_infer_generation(
            context,
            prompt_ids=prompt_ids,
            prompt_text=prompt_text,
            settings=settings,
            log=log_stage if int(getattr(args, "log_every", 0) or 0) > 0 else None,
        )
        print(f"Prompt token ids: {prompt_ids}")
        print_infer_result(
            result,
            tokenizer=context.tokenizer,
            semantic_extras=extras or None,
        )
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


def run_eval(state: dict[str, Any]) -> int:
    configure_console_logging()
    recipe = recipe_from_state(state)
    args = namespace_from_state("eval", state)
    args.dataset_alias = str(getattr(args, "dataset_alias", None) or DEFAULT_DATASET_ALIAS)
    resolve_inference_artifact_defaults(args, mode_name=recipe.mode_name())
    apply_tinystories_dataset_defaults(args)
    resolve_dataset_selector_args(args)

    if args.device != "cuda":
        print("This evaluation command is configured to run on CUDA only.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("CUDA device is not available in this environment.", file=sys.stderr)
        return 1
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))

    graph_path = Path(args.graph).expanduser().resolve()
    report_path = Path(args.report_path).expanduser().resolve()
    graph, compiled, state_dict, _resolved_weights_path = load_compiled_inference_graph(
        graph_path=graph_path,
        weights_path=Path(args.weights).expanduser().resolve() if getattr(args, "weights", None) else None,
        device=device,
    )
    raw_text_encoding_name = resolve_raw_text_encoding_name(graph, encoding_override=getattr(args, "raw_text_encoding_override", None))
    dataset_alias = resolve_inference_dataset_alias(args, graph, default_alias=DEFAULT_DATASET_ALIAS, log=log_stage)
    tokenizer, tokenizer_path, tokenizer_name, dataset_name, dataset_path, dataset_meta = resolve_inference_tokenizer_context(
        graph=graph,
        state_dict=state_dict,
        dataset_alias=dataset_alias,
        raw_text_encoding_name=raw_text_encoding_name,
        dataset_download_kwargs=dataset_download_kwargs_from_args(args),
        require_dataset=True,
    )
    if dataset_name is None or dataset_path is None or dataset_meta is None:
        print("Evaluation requires a resolved dataset context.", file=sys.stderr)
        return 1
    log_tokenizer_status(log_stage, tokenizer, tokenizer_path, tokenizer_name)

    dataset_cfg = dict(graph.nodes.get("dataset_source").neuron_def.module_config or {}) if "dataset_source" in graph.nodes else {}
    context_window = int(getattr(args, "context_window", None) or dataset_cfg.get("seq_len") or 0)
    if context_window <= 0:
        print("Could not resolve a positive context window from the graph.", file=sys.stderr)
        return 1

    amp_dtype, amp_name, _use_amp = resolve_autocast_settings(
        graph,
        amp_dtype_override=getattr(args, "amp_dtype", None),
    )
    if getattr(args, "amp_dtype", None):
        graph.torch_config = {**graph.torch_config, "amp_dtype": amp_name}
    if "semantic_data_source" in graph.nodes:
        validation_loss = evaluate_semantic_model(
            graph,
            dataset_path,
            device=str(args.device),
            seq_len=context_window,
            batch_size=int(args.eval_batch_size),
            eval_batches=int(args.eval_batches),
            encoding_name=raw_text_encoding_name,
        )
    else:
        validation_loss = evaluate_validation_loss(
            compiled,
            dataset_path,
            device=device,
            amp_dtype=amp_dtype,
            seq_len=context_window,
            batch_size=int(args.eval_batch_size),
            eval_batches=int(args.eval_batches),
        )

    prompt_suite_name, prompt_suite = resolve_prompt_suite(
        dataset_name=dataset_name,
        dataset_meta=dataset_meta,
        requested_suite=str(getattr(args, "prompt_suite", None) or "auto"),
    )
    if getattr(args, "prompt", None) or getattr(args, "prompt_tokens", None):
        prompt_suite = [str(getattr(args, "prompt", None) or "")]
        prompt_suite_name = "custom"

    prompt_results: list[dict[str, Any]] = []
    for prompt_idx, prompt in enumerate(prompt_suite):
        prompt_ids = resolve_prompt_tokens(
            prompt=prompt,
            prompt_tokens=str(getattr(args, "prompt_tokens", None) or ""),
            tokenizer=tokenizer,
        )
        prompt_text = resolve_prompt_text(
            prompt=prompt,
            prompt_tokens=str(getattr(args, "prompt_tokens", None) or ""),
            prompt_ids=prompt_ids,
            tokenizer=tokenizer,
        )
        generator = torch.Generator(device=device.type)
        generator.manual_seed(int(args.seed) + prompt_idx)
        if "semantic_data_source" in graph.nodes:
            vocab = ConversationalVocabulary(semantic_vocab_ref_for_graph(graph))
            sem_cfg = dict(graph.nodes["semantic_data_source"].neuron_def.module_config or {})
            semantic_dim = int(sem_cfg.get("seq_len", vocab.vector_dim))
            sem_targets, semantic_overrides = resolve_semantic_targets(
                str(getattr(args, "sem_targets", None) or ""),
                str(getattr(args, "semantic_topics", None) or ""),
                semantic_dim,
                device,
                vocab,
                sequence_text=prompt_text,
            )
            semantic_router_vecs: torch.Tensor | None = None
            if graph_uses_semantic_router_vecs(graph) or bool(getattr(args, "experimental_semantic_router_vecs", False)):
                semantic_router_vecs = resolve_semantic_router_vecs(sem_targets, vocab=vocab, device=device)
            result = build_semantic_generation(
                graph=graph,
                compiled=compiled,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                device=device,
                amp_dtype=amp_dtype,
                generator=generator,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                top_p=float(getattr(args, "top_p", None) or DEFAULT_INFER_TOP_P),
                repetition_penalty=float(args.repetition_penalty),
                stop_token=args.stop_token,
                logits_node=str(getattr(args, "logits_node", None) or "auto"),
                context_window=context_window,
                sem_targets=sem_targets,
                semantic_router_vecs=semantic_router_vecs,
                log_every=int(args.log_every),
                log=log_stage if int(getattr(args, "log_every", 0) or 0) > 0 else None,
            )
        else:
            result = generate_sequence(
                compiled,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                device=device,
                amp_dtype=amp_dtype,
                generator=generator,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                repetition_penalty=float(args.repetition_penalty),
                top_p=float(getattr(args, "top_p", None) or DEFAULT_INFER_TOP_P),
                stop_token=args.stop_token,
                logits_node=str(getattr(args, "logits_node", None) or "auto"),
                context_window=context_window,
                log_every=int(args.log_every),
                log=log_stage if int(getattr(args, "log_every", 0) or 0) > 0 else None,
            )
            semantic_overrides = {}
        prompt_results.append(
            {
                "prompt": prompt,
                "seed": int(args.seed) + prompt_idx,
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": result["generated_token_ids"],
                "all_token_ids": result["all_token_ids"],
                "generated_text": result["generated_text"],
                "full_text": result["full_text"],
                "resolved_logits_key": result["resolved_logits_key"],
                "semantic_overrides": semantic_overrides,
            }
        )

    report = {
        "recipe": {
            "base_model": recipe.base_model,
            "topology": recipe.topology,
            "router_mode": recipe.router_mode,
            "use_jepa": recipe.use_jepa,
            "runtime": recipe.runtime,
            "mode_name": recipe.mode_name(),
        },
        "dataset_alias": dataset_alias,
        "validation_loss": validation_loss,
        "prompt_suite": prompt_suite_name,
        "prompt_results": prompt_results,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report written to: {report_path}")
    return 0


def run_kernels(state: dict[str, Any]) -> int:
    from neuralfn.tile_cuda import coverage_report, tile_cuda_diagnostics

    action = str(state.get("kernel_action") or "list")
    json_output = bool(state.get("json_output", False))
    report = coverage_report()
    diagnostics = tile_cuda_diagnostics()
    if action == "bench":
        payload = run_kernel_benchmark(state)
        if json_output:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("CUDA Tile benchmark:")
            print(f"  device: {payload['device']}")
            print(f"  iterations: {payload['iterations']}")
            print(f"  warmup: {payload['warmup']}")
            print(f"  tile_backend_resolved: {payload['tile_backend_resolved']}")
            for name, value in payload["seconds"].items():
                print(f"  {name}: {value:.6f}s")
        return 0
    if action == "examples":
        payload = kernel_examples_payload(
            output_dir=state.get("output_dir"),
            write=bool(state.get("write", False)),
        )
        if json_output:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("CUDA Tile examples:")
            for path in payload["examples"]:
                print(f"  {path}")
            print(f"Generated registry examples: {payload['generated_count']}")
            if payload["written"]:
                print(f"Wrote examples to: {payload['output_dir']}")
        return 0
    if action == "doctor":
        payload = {
            "diagnostics": diagnostics.to_dict(),
            "coverage": {
                "total_inventory": report.total_inventory,
                "accounted": report.accounted,
                "missing": list(report.missing),
                "by_status": report.by_status,
                "complete": report.complete,
            },
        }
        if json_output:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("CUDA Tile diagnostics:")
            for key, value in payload["diagnostics"].items():
                print(f"  {key}: {value}")
            print("Kernel coverage:")
            print(f"  accounted: {report.accounted}/{report.total_inventory}")
            print(f"  missing: {len(report.missing)}")
            for status, count in sorted(report.by_status.items()):
                print(f"  {status}: {count}")
        return 0

    payload = report.to_dict()
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"NeuralFn CUDA Tile kernel coverage: {report.accounted}/{report.total_inventory} accounted")
        for status, count in sorted(report.by_status.items()):
            print(f"  {status}: {count}")
        if report.missing:
            print("Missing:")
            for name in report.missing:
                print(f"  {name}")
        else:
            print("Missing: none")
    return 0


def _tile_benchmark_graph() -> Any:
    from neuralfn import BuiltinNeurons, Edge, NeuronGraph, NeuronInstance

    graph = NeuronGraph(name="tile_cuda_benchmark", training_method="torch", runtime="torch")
    graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="x"))
    graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="y"))
    graph.add_node(NeuronInstance(BuiltinNeurons.add, instance_id="add"))
    graph.add_node(NeuronInstance(BuiltinNeurons.relu, instance_id="relu"))
    graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
    graph.add_edge(Edge(src_node="x", src_port=0, dst_node="add", dst_port=0))
    graph.add_edge(Edge(src_node="y", src_port=0, dst_node="add", dst_port=1))
    graph.add_edge(Edge(src_node="add", src_port=0, dst_node="relu", dst_port=0))
    graph.add_edge(Edge(src_node="relu", src_port=0, dst_node="out", dst_port=0))
    graph.input_node_ids = ["x", "y"]
    graph.output_node_ids = ["out"]
    return graph


def _wrap_benchmark_output(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def _run_graph_walk_for_benchmark(compiled: Any, *flat_inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    graph = compiled.graph
    values: dict[str, tuple[torch.Tensor, ...]] = {}
    input_idx = 0
    for node_id in graph.input_node_ids:
        output_count = graph.nodes[node_id].neuron_def.n_outputs
        values[node_id] = tuple(flat_inputs[input_idx : input_idx + output_count])
        input_idx += output_count

    for node_id in graph.topological_order():
        if node_id in graph.input_node_ids:
            continue
        incoming = sorted(graph._incoming(node_id), key=lambda edge: edge.dst_port)
        args = tuple(values[edge.src_node][edge.src_port] for edge in incoming)
        values[node_id] = _wrap_benchmark_output(compiled.node_modules[node_id](*args))

    outputs: list[torch.Tensor] = []
    for node_id in graph.output_node_ids:
        outputs.extend(values[node_id])
    return tuple(outputs)


def _benchmark_callable(fn: Callable[[], Any], *, iterations: int, warmup: int, device: torch.device) -> float:
    for _ in range(max(0, warmup)):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(max(1, iterations)):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


def run_kernel_benchmark(state: dict[str, Any]) -> dict[str, Any]:
    from neuralfn.torch_backend import CompiledTorchGraph

    requested_device = str(state.get("device") or "auto")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested, but CUDA is not available.")

    iterations_raw = state.get("iterations")
    warmup_raw = state.get("warmup")
    iterations = max(1, int(200 if iterations_raw is None else iterations_raw))
    warmup = max(0, int(20 if warmup_raw is None else warmup_raw))
    graph = _tile_benchmark_graph()
    x = torch.randn(8192, device=device)
    y = torch.randn(8192, device=device)
    compiled_torch = CompiledTorchGraph(graph, kernel_backend="torch").to(device)
    compiled_tile = CompiledTorchGraph(graph, kernel_backend="tile_cuda", tile_cuda_strict=False).to(device)

    with torch.no_grad():
        graph_walk_seconds = _benchmark_callable(
            lambda: _run_graph_walk_for_benchmark(compiled_torch, x, y),
            iterations=iterations,
            warmup=warmup,
            device=device,
        )
        compiled_torch_seconds = _benchmark_callable(
            lambda: compiled_torch(x, y),
            iterations=iterations,
            warmup=warmup,
            device=device,
        )
        compiled_tile_seconds = _benchmark_callable(
            lambda: compiled_tile(x, y),
            iterations=iterations,
            warmup=warmup,
            device=device,
        )

    return {
        "device": str(device),
        "iterations": iterations,
        "warmup": warmup,
        "tile_backend_resolved": compiled_tile.resolved_kernel_backend,
        "seconds": {
            "graph_walk_pytorch": graph_walk_seconds,
            "compiled_pytorch": compiled_torch_seconds,
            "compiled_tile_cuda_requested": compiled_tile_seconds,
        },
    }


_TILE_CUDA_EXAMPLE_NAMES = (
    "scalar_add_train.py",
    "dense_llm_smoke_train.py",
    "moe_router_smoke_train.py",
    "jepa_smoke_train.py",
    "strict_mode_report.py",
    "kernel_bench.py",
)


def _example_source(name: str) -> str:
    header = '''"""NeuralFn CUDA Tile example."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cli"))

'''
    if name == "scalar_add_train.py":
        return header + """import torch
from neuralfn import BuiltinNeurons, Edge, NeuronGraph, NeuronInstance
from neuralfn.torch_backend import CompiledTorchGraph


def build_graph() -> NeuronGraph:
    graph = NeuronGraph(name="tile_scalar_add")
    graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="x"))
    graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="y"))
    graph.add_node(NeuronInstance(BuiltinNeurons.add, instance_id="add"))
    graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
    graph.add_edge(Edge(src_node="x", src_port=0, dst_node="add", dst_port=0))
    graph.add_edge(Edge(src_node="y", src_port=0, dst_node="add", dst_port=1))
    graph.add_edge(Edge(src_node="add", src_port=0, dst_node="out", dst_port=0))
    graph.input_node_ids = ["x", "y"]
    graph.output_node_ids = ["out"]
    return graph


compiled = CompiledTorchGraph(build_graph(), kernel_backend="tile_cuda")
x = torch.tensor([1.0, 2.0])
y = torch.tensor([3.0, 4.0])
print(compiled(x, y)[0])
"""
    if name == "kernel_bench.py":
        return header + """from cli.nfn_impl import main


raise SystemExit(main(["kernels", "bench", "--iterations", "200"]))
"""
    if name == "strict_mode_report.py":
        return header + """from neuralfn import BuiltinNeurons, Edge, NeuronGraph, NeuronInstance
from neuralfn.torch_backend import CompiledTorchGraph


graph = NeuronGraph(name="tile_strict_report")
graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="x"))
graph.add_node(NeuronInstance(BuiltinNeurons.relu, instance_id="relu"))
graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
graph.add_edge(Edge(src_node="x", src_port=0, dst_node="relu", dst_port=0))
graph.add_edge(Edge(src_node="relu", src_port=0, dst_node="out", dst_port=0))
graph.input_node_ids = ["x"]
graph.output_node_ids = ["out"]
CompiledTorchGraph(
    graph,
    kernel_backend="tile_cuda",
    tile_cuda_strict=False,
    tile_cuda_report_path="tile_cuda_report.json",
)
print("wrote tile_cuda_report.json with fallback-safe diagnostics")
try:
    CompiledTorchGraph(graph, kernel_backend="tile_cuda", tile_cuda_strict=True)
except RuntimeError as exc:
    print(f"strict mode rejected unavailable or uncovered Tile backend: {exc}")
"""
    if name == "dense_llm_smoke_train.py":
        preset = "gpt2"
    elif name == "moe_router_smoke_train.py":
        preset = "moe"
    else:
        preset = "llm_jepa"
    return header + f"""from neuralfn.torch_backend import CompiledTorchGraph
from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config


spec = build_model_spec_from_config({{"preset": "{preset}", "vocab_size": 128, "num_layers": 1, "model_dim": 32, "num_heads": 4}}, preview_defaults=True)
graph = build_gpt_root_graph(name="{preset}_tile_cuda_smoke", model_spec=spec)
compiled = CompiledTorchGraph(graph, kernel_backend="tile_cuda")
print(f"prepared {{graph.name}} with {{len(graph.nodes)}} nodes via {{compiled.resolved_kernel_backend}}")
"""


def _generated_example_source(inventory_key: str) -> str:
    safe_name = inventory_key.replace(":", "_")
    return f'''"""Generated CUDA Tile SDK example for {inventory_key}."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("{inventory_key}")
if spec is None:
    raise SystemExit("missing registry spec: {inventory_key}")

print("{safe_name}", spec.status, spec.shape_contract)
'''


def kernel_examples_payload(*, output_dir: Any, write: bool) -> dict[str, Any]:
    from neuralfn.tile_cuda import coverage_report

    target_dir = Path(output_dir or "examples/tile_cuda").expanduser()
    examples = [str(target_dir / name) for name in _TILE_CUDA_EXAMPLE_NAMES]
    report = coverage_report()
    generated_names = [
        f"{spec.inventory_key.replace(':', '_').replace('/', '_')}.py"
        for spec in report.specs
    ]
    if write:
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in _TILE_CUDA_EXAMPLE_NAMES:
            (target_dir / name).write_text(_example_source(name), encoding="utf-8")
        generated_dir = target_dir / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        for spec, file_name in zip(report.specs, generated_names):
            (generated_dir / file_name).write_text(_generated_example_source(spec.inventory_key), encoding="utf-8")
    return {
        "output_dir": str(target_dir),
        "written": write,
        "examples": examples,
        "generated_count": len(generated_names),
        "generated_dir": str(target_dir / "generated"),
    }


def execute(command: str, state: dict[str, Any]) -> int:
    if command == "train":
        return run_train(state)
    if command == "infer":
        return run_infer(state)
    if command == "kernels":
        return run_kernels(state)
    return run_eval(state)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin_isatty: bool | None = None,
    stdout_isatty: bool | None = None,
) -> int:
    tokens = list(argv if argv is not None else sys.argv[1:])
    command, remainder = detect_command(tokens)
    tty = is_tty(stdin_isatty=stdin_isatty, stdout_isatty=stdout_isatty)
    if command is None:
        root_parser = build_root_parser(default_help_style(tokens, tty=tty) if any(token in {"-h", "--help"} for token in tokens) else "long")
        ns, _unknown = root_parser.parse_known_args(tokens)
        if ns.help or not tokens:
            style = default_help_style(tokens, tty=tty)
            print(render_help(None, style=style))
            return 0
        if tty:
            choice = run_single_choice_menu(
                "nfn",
                "Choose a command.",
                [
                    OptionChoice("Train", "Train a composed recipe.", "train", recommended=True),
                    OptionChoice("Infer", "Load an exported graph and start infer chat.", "infer"),
                    OptionChoice("Eval", "Run validation and prompt probes.", "eval"),
                    OptionChoice("Kernels", "Inspect CUDA Tile kernel coverage.", "kernels"),
                ],
            )
            command = str(choice)
            remainder = []
        else:
            root_parser.error("the following arguments are required: command")
            return 2

    parser = build_command_parser(command, style="long")
    explicit = collect_explicit_dests(parser, remainder)
    args = parser.parse_args(remainder)
    if args.help:
        style = default_help_style(remainder, tty=tty)
        print(render_help(command, style=style))
        return 0

    state = {key: value for key, value in vars(args).items() if value is not None}
    if command == "kernels":
        return execute(command, state)
    if command == "infer":
        if args.plan or args.plan_auto:
            print("nfn infer is artifact-first; ignoring --plan/--plan-auto.", file=sys.stderr)
        state["_tty"] = tty
        return execute(command, state)
    if "base_model" not in state or args.plan or args.plan_auto:
        if args.plan and not tty and not args.plan_auto:
            print("--plan requires a TTY. Use --plan-auto for non-interactive planning.", file=sys.stderr)
            return 2
        planned = maybe_plan(command, state, explicit, interactive=tty and not args.plan_auto)
        print_resolved_command(command, planned)
        state = planned
    else:
        apply_recommended_presets(command, state, explicit)
    state["_tty"] = tty
    return execute(command, state)


__all__ = [
    "ComposedRecipe",
    "OptionChoice",
    "Question",
    "execute",
    "main",
    "maybe_plan",
    "recipe_from_state",
    "render_help",
    "tokenizer_choices",
]
