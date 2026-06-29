from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import uuid
from typing import Any

from cli_utils import artifact_path, create_argument_parser
from neuralfn.native_gpt import (
    NativeGptRunConfig,
    build_native_gpt_compiled_cli_run_config,
    build_native_gpt_run_config,
    native_gpt_encoding_vocab_size,
    native_gpt_runner_status,
    normalize_native_gpt_encoding_name,
    run_native_gpt,
    write_native_gpt_run_config,
)


LOGGER = logging.getLogger("gpt_native_harness")
DATASETS_DIR = Path(os.environ.get("NFN_DATASETS_DIR", Path.home() / ".cache" / "nfn" / "datasets")).expanduser()
DEFAULT_ARTIFACT = artifact_path("gpt.pt")
TINYSTORIES_ALIAS = "roneneldan__TinyStories__TinyStoriesV2-GPT4"
TINYSTORIES_HF_PATH = "roneneldan/TinyStories"
TINYSTORIES_TRAIN_FILE = "TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_VAL_FILE = "TinyStoriesV2-GPT4-valid.txt"
DEFAULT_DATASET_ALIAS = TINYSTORIES_ALIAS
DEFAULT_MODEL_FAMILY = "gpt"

NATIVE_GPT_DEFAULTS = {
    "seed": 1337,
    "device": "cuda",
    "run_id": str(uuid.uuid4()),
    "dataset_alias": DEFAULT_DATASET_ALIAS,
    "dataset_hf_path": TINYSTORIES_HF_PATH,
    "dataset_train_file": TINYSTORIES_TRAIN_FILE,
    "dataset_val_file": TINYSTORIES_VAL_FILE,
    "output": str(DEFAULT_ARTIFACT),
    "max_steps": 20_000,
    "train_seq_len": 1_024,
    "batch_size": 64,
    "train_batch_tokens": 524_288,
    "warmup_steps": 1000,
    "learning_rate": 6e-4,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "adam_eps": 1e-8,
    "grad_clip_norm": 1.0,
    "eval_every_steps": 1000,
    "eval_batches": 20,
    "train_loss_every_steps": 0,
    "lm_head_row_chunk_size": 28_672,
    "num_layers": 12,
    "native_cuda_checkpoint_every": 200,
    "native_cuda_sample_every": 20_000,
    "native_cuda_generate_tokens": 144,
    "native_cuda_runner": "compiled-cli",
    "native_cuda_activation": "gelu",
    "native_cuda_moa_interval": 50,
    "native_cuda_kernel_backend": "tile-cuda",
    "native_cuda_tile_ops_lib": "",
    "native_cuda_cuda_runtime_lib": "",
    "model_family": DEFAULT_MODEL_FAMILY,
    "template_name": "gpt",
    "graph_file": "",
}
NATIVE_GPT2_DEFAULTS = NATIVE_GPT_DEFAULTS


def _compiled_cli_env(config: NativeGptRunConfig) -> dict[str, str]:
    env = os.environ.copy()
    if str(config.cuda_visible_devices or "").strip():
        env["CUDA_VISIBLE_DEVICES"] = str(config.cuda_visible_devices)
    if str(config.cuda_device_max_connections or "").strip():
        env["CUDA_DEVICE_MAX_CONNECTIONS"] = str(config.cuda_device_max_connections)
    _set_env_default_if_empty(env, "CUDA_MODULE_LOADING", "LAZY")
    return env


def _set_env_default_if_empty(env: dict[str, str], key: str, value: str) -> None:
    if value and not str(env.get(key, "")).strip():
        env[key] = value


def _exec_compiled_cli(command: list[str], config: NativeGptRunConfig) -> int:
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvpe(command[0], command, _compiled_cli_env(config))
    return 127


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def configure_console_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _read_meta(dataset_path: Path) -> dict[str, Any]:
    meta_path = dataset_path / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _resolve_existing_dataset(alias: str) -> tuple[str, Path, dict[str, Any]]:
    alias_path = Path(str(alias)).expanduser()
    if alias_path.is_absolute():
        if not alias_path.is_dir():
            raise FileNotFoundError(f"Native GPT dataset path {alias_path} is not a directory.")
        return str(alias_path), alias_path, _read_meta(alias_path)
    dataset_path = DATASETS_DIR / alias
    if dataset_path.is_dir():
        return alias, dataset_path, _read_meta(dataset_path)
    raise FileNotFoundError(f"Dataset alias {alias!r} was not found under {DATASETS_DIR}.")


def _resolve_existing_dataset_path(alias: str) -> tuple[str, Path]:
    alias_path = Path(str(alias)).expanduser()
    if alias_path.is_absolute():
        if not alias_path.is_dir():
            raise FileNotFoundError(f"Native GPT dataset path {alias_path} is not a directory.")
        return str(alias_path), alias_path
    dataset_path = DATASETS_DIR / alias
    if dataset_path.is_dir():
        return alias, dataset_path
    raise FileNotFoundError(f"Dataset alias {alias!r} was not found under {DATASETS_DIR}.")


def _apply_dataset_shortcuts(args: argparse.Namespace) -> None:
    if bool(args.tinystories):
        if args.dataset not in (None, "tinystories"):
            raise ValueError("--tinystories cannot be combined with another --dataset shortcut.")
        args.dataset = "tinystories"
    if args.dataset == "tinystories":
        args.dataset_alias = TINYSTORIES_ALIAS
        args.dataset_hf_path = TINYSTORIES_HF_PATH
        args.dataset_train_file = TINYSTORIES_TRAIN_FILE
        args.dataset_val_file = TINYSTORIES_VAL_FILE
    elif args.dataset in {"golf1", "golf10"}:
        shard_count = 1 if args.dataset == "golf1" else 10
        args.dataset_alias = f"willdepueoai__parameter-golf__sp1024__train{shard_count}"
        args.dataset_hf_path = "willdepueoai/parameter-golf"
        args.dataset_variant = "sp1024"
        args.dataset_train_shards = shard_count


def _resolve_tokenizer(args: argparse.Namespace) -> str:
    selected = str(args.tokenizer or "gpt2")
    legacy_flags = [bool(args.tokgpt2), bool(args.cl100k), bool(args.o200k)]
    if sum(1 for flag in legacy_flags if flag) > 1:
        raise ValueError("Tokenizer flags are mutually exclusive.")
    if args.tokgpt2:
        selected = "gpt2"
    if args.cl100k:
        selected = "cl100k_base"
    if args.o200k:
        selected = "o200k_base"
    normalized = normalize_native_gpt_encoding_name(selected) or selected
    native_gpt_encoding_vocab_size(normalized)
    return normalized


def _download_dataset_if_needed(args: argparse.Namespace, encoding_name: str) -> tuple[str, Path, dict[str, Any]]:
    try:
        return _resolve_existing_dataset(str(args.dataset_alias))
    except FileNotFoundError:
        if not bool(args.download_if_missing):
            raise
    if not args.dataset_hf_path:
        raise FileNotFoundError(
            f"Dataset alias {args.dataset_alias!r} is missing and no download contract is known."
        )
    from server.dataset_manager import download_hf_dataset

    download_hf_dataset(
        str(args.dataset_hf_path),
        alias=str(args.dataset_alias),
        variant=args.dataset_variant,
        train_shards=args.dataset_train_shards,
        train_file=args.dataset_train_file,
        val_file=args.dataset_val_file,
        encoding_name=encoding_name,
    )
    return _resolve_existing_dataset(str(args.dataset_alias))


def _download_dataset_path_if_needed(args: argparse.Namespace, encoding_name: str) -> tuple[str, Path]:
    try:
        return _resolve_existing_dataset_path(str(args.dataset_alias))
    except FileNotFoundError:
        if not bool(args.download_if_missing):
            raise
    if not args.dataset_hf_path:
        raise FileNotFoundError(
            f"Dataset alias {args.dataset_alias!r} is missing and no download contract is known."
        )
    from server.dataset_manager import download_hf_dataset

    download_hf_dataset(
        str(args.dataset_hf_path),
        alias=str(args.dataset_alias),
        variant=args.dataset_variant,
        train_shards=args.dataset_train_shards,
        train_file=args.dataset_train_file,
        val_file=args.dataset_val_file,
        encoding_name=encoding_name,
    )
    dataset_name, dataset_path, _meta = _resolve_existing_dataset(str(args.dataset_alias))
    return dataset_name, dataset_path


def _has_native_token_shards(dataset_path: Path, *, allow_train_as_val: bool) -> bool:
    train_files = sorted(dataset_path.glob("fineweb_train_*.bin"))
    if not train_files:
        return False
    val_files = sorted(dataset_path.glob("fineweb_val_*.bin"))
    return bool(val_files or allow_train_as_val)


def _build_compiled_cli_config(args: argparse.Namespace, dataset_arg: str | Path) -> NativeGptRunConfig:
    output_dir = (
        Path(args.native_cuda_output_dir)
        if str(args.native_cuda_output_dir or "").strip()
        else Path(args.output).with_suffix("")
    )
    return build_native_gpt_compiled_cli_run_config(
        dataset_alias=str(dataset_arg),
        executable=str(args.native_cuda_executable or "") or None,
        output_dir=output_dir,
        eval_every_steps=int(args.eval_every_steps),
        eval_batches=int(args.eval_batches),
        eval_batch_size=int(args.eval_batch_size),
        train_loss_every_steps=int(args.train_loss_every_steps),
        lm_head_row_chunk_size=int(args.lm_head_row_chunk_size),
        sample_every_steps=int(args.native_cuda_sample_every),
        generate_tokens=int(args.native_cuda_generate_tokens),
        checkpoint_every_steps=int(args.native_cuda_checkpoint_every),
        batch_size=int(args.batch_size),
        seq_len=int(args.train_seq_len),
        train_batch_tokens=int(args.train_batch_tokens),
        learning_rate=float(args.learning_rate),
        min_lr=args.min_lr,
        warmup_steps=int(args.warmup_steps),
        weight_decay=float(args.weight_decay),
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        adam_eps=float(args.adam_eps),
        grad_clip_norm=float(args.grad_clip_norm),
        max_steps=int(args.max_steps),
        num_layers=int(args.num_layers),
        activation=str(args.native_cuda_activation),
        moa_interval=int(args.native_cuda_moa_interval),
        kernel_backend=str(args.native_cuda_kernel_backend),
        tile_ops_lib=str(args.native_cuda_tile_ops_lib or ""),
        smoke_tile_ops=bool(args.native_cuda_smoke_tile_ops),
        smoke_nvfp4_pack=bool(args.native_cuda_smoke_nvfp4_pack),
        smoke_optimizer_step=bool(args.native_cuda_smoke_optimizer_step),
        smoke_lm_step=bool(args.native_cuda_smoke_lm_step),
        smoke_attention_step=bool(args.native_cuda_smoke_attention_step),
        smoke_mlp_step=bool(args.native_cuda_smoke_mlp_step),
        smoke_norm_residual_step=bool(args.native_cuda_smoke_norm_residual_step),
        smoke_transformer_block_step=bool(args.native_cuda_smoke_transformer_block_step),
        smoke_transformer_lm_step=bool(args.native_cuda_smoke_transformer_lm_step),
        smoke_embedding_lm_step=bool(args.native_cuda_smoke_embedding_lm_step),
        train_embedding_lm=bool(args.train_embedding_lm),
        train_transformer_lm=bool(args.train_transformer_lm),
        require_cooperative_lm_head_backward=bool(args.require_cooperative_lm_head_backward),
        cuda_runtime_lib=str(args.native_cuda_cuda_runtime_lib or ""),
        template_name=str(args.template_name or "gpt"),
        graph_file=str(args.graph_file or ""),
        model_family=str(args.model_family or DEFAULT_MODEL_FAMILY),
        write_checkpoint=not bool(args.native_cuda_no_checkpoint),
        batch_size_explicit=bool(getattr(args, "_batch_size_explicit", True)),
        seq_len_explicit=bool(getattr(args, "_seq_len_explicit", True)),
        num_layers_explicit=bool(getattr(args, "_num_layers_explicit", True)),
    )


def _compiled_cli_dataset_arg(args: argparse.Namespace) -> tuple[str, Path, str | Path]:
    dataset_name = str(args.dataset_alias)
    alias_path = Path(dataset_name).expanduser()
    if alias_path.is_absolute():
        return str(alias_path), alias_path, alias_path
    return dataset_name, DATASETS_DIR / dataset_name, dataset_name


def _compiled_cli_defer_dataset_resolution(args: argparse.Namespace) -> bool:
    return not bool(args.download_if_missing)


def _uint16_sequence_count(train_shard: Path, *, seq_len: int) -> int | None:
    if not train_shard.exists() or int(seq_len) <= 0:
        return None
    token_count = train_shard.stat().st_size // 2
    return max(0, (int(token_count) - 1) // int(seq_len))


def build_parser() -> argparse.ArgumentParser:
    parser = create_argument_parser(description="Train dense GPT templates with the NeuralFn native CUDA harness.")
    parser.add_argument("--runtime", choices=("native-cuda",), default="native-cuda")
    parser.add_argument(
        "--model-family",
        "--base-model",
        "--model",
        choices=("gpt", "gpt2", "gpt3", "nanogpt"),
        default=env_str("MODEL_FAMILY", env_str("BASE_MODEL", NATIVE_GPT_DEFAULTS["model_family"])),
        help=(
            "Dense GPT family label for native metadata and defaults. gpt3 uses the same native GPT "
            "kernel family and defaults to a 2048 context only when no template, graph, or seq len is supplied. "
            "nanogpt selects the shared dense GPT trainer with the nanogpt template."
        ),
    )
    parser.add_argument("--run-id", default=env_str("RUN_ID", NATIVE_GPT_DEFAULTS["run_id"]))
    parser.add_argument("--seed", type=int, default=env_int("SEED", NATIVE_GPT_DEFAULTS["seed"]))
    parser.add_argument("--device", default=env_str("DEVICE", NATIVE_GPT_DEFAULTS["device"]))
    parser.add_argument("--tinystories", action="store_true")
    parser.add_argument("--dataset", choices=("tinystories", "golf1", "golf10"), default=None)
    parser.add_argument("--dataset-alias", default=env_str("DATASET_ALIAS", NATIVE_GPT_DEFAULTS["dataset_alias"]))
    parser.add_argument("--download-if-missing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dataset-hf-path", default=env_str("DATASET_HF_PATH", NATIVE_GPT_DEFAULTS["dataset_hf_path"]))
    parser.add_argument("--dataset-variant", default=None)
    parser.add_argument("--dataset-train-shards", type=int, default=None)
    parser.add_argument("--dataset-train-file", default=env_str("DATASET_TRAIN_FILE", NATIVE_GPT_DEFAULTS["dataset_train_file"]))
    parser.add_argument("--dataset-val-file", default=env_str("DATASET_VAL_FILE", NATIVE_GPT_DEFAULTS["dataset_val_file"]))
    parser.add_argument(
        "--template-name",
        "--template",
        "--preset",
        default=env_str("TEMPLATE_NAME", NATIVE_GPT_DEFAULTS["template_name"]),
        help=(
            "GPT template preset or alias to select for native training metadata and dispatch. "
            "The default gpt alias resolves to the dense GPT native implementation."
        ),
    )
    parser.add_argument(
        "--graph-file",
        "--graph",
        default=env_str("GRAPH_FILE", NATIVE_GPT_DEFAULTS["graph_file"]),
        help="Custom NeuralFn graph JSON to select for native training metadata and dispatch.",
    )
    parser.add_argument("--tokenizer", default=env_str("TOKENIZER", "gpt2"))
    parser.add_argument("--tokgpt2", action="store_true", help="Use the GPT-2 byte-level BPE tokenizer.")
    parser.add_argument("--cl100k", action="store_true", help="Use cl100k_base. Not valid for native uint16 GPT shards.")
    parser.add_argument("--o200k", action="store_true", help="Use o200k_base. Not valid for native uint16 GPT shards.")
    parser.add_argument("--output", default=env_str("OUTPUT", ""))
    parser.add_argument("--max-steps", type=int, default=env_int("ITERATIONS", NATIVE_GPT_DEFAULTS["max_steps"]))
    parser.add_argument("--train-seq-len", type=int, default=env_int("TRAIN_SEQ_LEN", NATIVE_GPT_DEFAULTS["train_seq_len"]))
    parser.add_argument("--batch-size", type=int, default=env_int("BATCH_SIZE", NATIVE_GPT_DEFAULTS["batch_size"]))
    parser.add_argument(
        "--train-batch-tokens",
        type=int,
        default=env_int("TRAIN_BATCH_TOKENS", NATIVE_GPT_DEFAULTS["train_batch_tokens"]),
    )
    parser.add_argument("--learning-rate", type=float, default=env_float("LEARNING_RATE", NATIVE_GPT_DEFAULTS["learning_rate"]))
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=env_float("WEIGHT_DECAY", NATIVE_GPT_DEFAULTS["weight_decay"]))
    parser.add_argument("--beta1", type=float, default=env_float("BETA1", NATIVE_GPT_DEFAULTS["beta1"]))
    parser.add_argument("--beta2", type=float, default=env_float("BETA2", NATIVE_GPT_DEFAULTS["beta2"]))
    parser.add_argument("--adam-eps", type=float, default=env_float("ADAM_EPS", NATIVE_GPT_DEFAULTS["adam_eps"]))
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=env_float("GRAD_CLIP_NORM", NATIVE_GPT_DEFAULTS["grad_clip_norm"]),
    )
    parser.add_argument("--warmup-steps", type=int, default=env_int("WARMUP_STEPS", NATIVE_GPT_DEFAULTS["warmup_steps"]))
    parser.add_argument("--eval-every-steps", type=int, default=env_int("EVAL_EVERY_STEPS", NATIVE_GPT_DEFAULTS["eval_every_steps"]))
    parser.add_argument("--eval-batches", type=int, default=env_int("EVAL_BATCHES", NATIVE_GPT_DEFAULTS["eval_batches"]))
    parser.add_argument("--eval-batch-size", type=int, default=env_int("EVAL_BATCH_SIZE", 0))
    parser.add_argument(
        "--train-loss-every-steps",
        "--train-log-every",
        "--train-log-every-steps",
        type=int,
        default=env_int("TRAIN_LOSS_EVERY_STEPS", NATIVE_GPT_DEFAULTS["train_loss_every_steps"]),
        help="Record native training loss every N optimizer steps; 0 disables train-loss evaluation for timing runs.",
    )
    parser.add_argument(
        "--lm-head-row-chunk-size",
        "--native-cuda-lm-head-row-chunk-size",
        type=int,
        default=env_int(
            "NATIVE_CUDA_LM_HEAD_ROW_CHUNK_SIZE",
            env_int("LM_HEAD_ROW_CHUNK_SIZE", NATIVE_GPT_DEFAULTS["lm_head_row_chunk_size"]),
        ),
    )
    parser.add_argument("--num-layers", type=int, default=env_int("NUM_LAYERS", NATIVE_GPT_DEFAULTS["num_layers"]))
    parser.add_argument(
        "--native-cuda-executable",
        default=env_str("NFN_NATIVE_GPT_TRAIN_BIN", env_str("NFN_NATIVE_GPT2_TRAIN_BIN", "")),
    )
    parser.add_argument(
        "--native-cuda-runner",
        choices=("auto", "binding", "compiled-cli", "launcher"),
        default=env_str(
            "NFN_NATIVE_GPT_RUNNER",
            env_str("NFN_NATIVE_GPT2_RUNNER", NATIVE_GPT_DEFAULTS["native_cuda_runner"]),
        ),
        help=(
            "Native GPT launch mode. The default requires the compiled no-Python cached-shard CLI; "
            "use auto, binding, or launcher explicitly for alternate NeuralFn-native launch modes."
        ),
    )
    parser.add_argument("--native-cuda-output-dir", default=env_str("NATIVE_CUDA_OUTPUT_DIR", ""))
    parser.add_argument("--native-cuda-config-out", default=env_str("NATIVE_CUDA_CONFIG_OUT", ""))
    parser.add_argument("--native-cuda-print-command", action="store_true")
    parser.add_argument("--native-cuda-dry-run", action="store_true")
    parser.add_argument("--native-cuda-print-plan", action="store_true")
    parser.add_argument("--native-cuda-list-templates", "--list-templates", action="store_true")
    parser.add_argument("--native-cuda-check-tile-ops", action="store_true")
    parser.add_argument(
        "--native-cuda-no-checkpoint",
        "--no-checkpoint",
        action="store_true",
        help="Skip final trained checkpoint export for native benchmark/preflight runs.",
    )
    parser.add_argument("--native-cuda-smoke-tile-ops", "--smoke-tile-ops", action="store_true")
    parser.add_argument("--native-cuda-smoke-nvfp4-pack", "--smoke-nvfp4-pack", action="store_true")
    parser.add_argument("--native-cuda-smoke-optimizer-step", "--smoke-optimizer-step", action="store_true")
    parser.add_argument("--native-cuda-smoke-lm-step", "--smoke-lm-step", action="store_true")
    parser.add_argument("--native-cuda-smoke-attention-step", "--smoke-attention-step", action="store_true")
    parser.add_argument("--native-cuda-smoke-mlp-step", "--smoke-mlp-step", action="store_true")
    parser.add_argument("--native-cuda-smoke-norm-residual-step", "--smoke-norm-residual-step", action="store_true")
    parser.add_argument("--native-cuda-smoke-transformer-block-step", "--smoke-transformer-block-step", action="store_true")
    parser.add_argument("--native-cuda-smoke-transformer-lm-step", "--smoke-transformer-lm-step", action="store_true")
    parser.add_argument("--native-cuda-smoke-embedding-lm-step", "--smoke-embedding-lm-step", action="store_true")
    parser.add_argument("--train-embedding-lm", action="store_true")
    parser.add_argument("--train-transformer-lm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-cooperative-lm-head-backward",
        "--native-cuda-require-cooperative-lm-head-backward",
        action="store_true",
        help=(
            "Require the compiled true-fused cooperative LM-head backward Tile route; "
            "fails before training if only graph or sequence wrappers are available."
        ),
    )
    parser.add_argument("--native-cuda-allow-train-val-fallback", action="store_true")
    parser.add_argument(
        "--native-cuda-checkpoint-every",
        type=int,
        default=env_int("NATIVE_CUDA_CHECKPOINT_EVERY", NATIVE_GPT_DEFAULTS["native_cuda_checkpoint_every"]),
    )
    parser.add_argument(
        "--native-cuda-sample-every",
        type=int,
        default=env_int("NATIVE_CUDA_SAMPLE_EVERY", NATIVE_GPT_DEFAULTS["native_cuda_sample_every"]),
    )
    parser.add_argument(
        "--native-cuda-generate-tokens",
        type=int,
        default=env_int("NATIVE_CUDA_GENERATE_TOKENS", NATIVE_GPT_DEFAULTS["native_cuda_generate_tokens"]),
    )
    parser.add_argument("--native-cuda-activation", default=env_str("NATIVE_CUDA_ACTIVATION", "gelu"))
    parser.add_argument(
        "--native-cuda-kernel-backend",
        "--kernel-backend",
        choices=("tile-cuda",),
        default=env_str("NATIVE_CUDA_KERNEL_BACKEND", NATIVE_GPT_DEFAULTS["native_cuda_kernel_backend"]),
    )
    parser.add_argument(
        "--native-cuda-tile-ops-lib",
        "--tile-ops-lib",
        default=env_str("NATIVE_CUDA_TILE_OPS_LIB", NATIVE_GPT_DEFAULTS["native_cuda_tile_ops_lib"]),
    )
    parser.add_argument(
        "--native-cuda-cuda-runtime-lib",
        "--cuda-runtime-lib",
        default=env_str("NATIVE_CUDA_RUNTIME_LIB", NATIVE_GPT_DEFAULTS["native_cuda_cuda_runtime_lib"]),
    )
    parser.add_argument(
        "--native-cuda-moa-interval",
        type=int,
        default=env_int("NATIVE_CUDA_MOA_INTERVAL", NATIVE_GPT_DEFAULTS["native_cuda_moa_interval"]),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_console_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    explicit_tokens = list(argv if argv is not None else sys.argv[1:])
    explicit_seq_len = any(token == "--train-seq-len" or token.startswith("--train-seq-len=") for token in explicit_tokens)
    explicit_batch_size = any(token == "--batch-size" or token.startswith("--batch-size=") for token in explicit_tokens)
    explicit_num_layers = any(token == "--num-layers" or token.startswith("--num-layers=") for token in explicit_tokens)
    explicit_template = any(
        token in {"--template-name", "--template", "--preset"}
        or token.startswith("--template-name=")
        or token.startswith("--template=")
        or token.startswith("--preset=")
        for token in explicit_tokens
    )
    explicit_graph = any(
        token in {"--graph-file", "--graph"}
        or token.startswith("--graph-file=")
        or token.startswith("--graph=")
        for token in explicit_tokens
    )
    model_selector = str(args.model_family or DEFAULT_MODEL_FAMILY).strip().lower().replace("_", "-")
    if model_selector == "gpt3" and not explicit_seq_len and not explicit_template and not explicit_graph:
        args.train_seq_len = 2048
        args._seq_len_explicit = True
    else:
        args._seq_len_explicit = explicit_seq_len
    args._batch_size_explicit = explicit_batch_size
    args._num_layers_explicit = explicit_num_layers
    if model_selector == "nanogpt" and not explicit_template and not explicit_graph:
        args.template_name = "nanogpt"
    args.model_family = "gpt" if model_selector in {"gpt", "gpt2", "gpt3", "nanogpt"} else model_selector
    _apply_dataset_shortcuts(args)
    encoding_name = _resolve_tokenizer(args)
    if not str(args.output or "").strip():
        args.output = str(DEFAULT_ARTIFACT)
    if args.device != "cuda":
        print("This harness is configured to run on CUDA only.", file=sys.stderr)
        return 1

    LOGGER.info("Starting GPT native CUDA harness run %s", args.run_id)
    LOGGER.info("CLI started at %s", datetime.now().isoformat(timespec="seconds"))
    LOGGER.info("Resolving dataset alias %s", args.dataset_alias)

    runner_status = native_gpt_runner_status(str(args.native_cuda_runner))
    compiled_cli_deferred_dataset = False
    if runner_status.resolved == "compiled-cli" and _compiled_cli_defer_dataset_resolution(args):
        dataset_name, dataset_path, dataset_arg = _compiled_cli_dataset_arg(args)
        native_cfg = _build_compiled_cli_config(args, dataset_arg)
        use_compiled_resolver = True
        compiled_cli_deferred_dataset = True
    elif runner_status.resolved == "compiled-cli":
        dataset_name, dataset_path = _download_dataset_path_if_needed(args, encoding_name)
        use_compiled_resolver = _has_native_token_shards(
            dataset_path,
            allow_train_as_val=bool(args.native_cuda_allow_train_val_fallback),
        )
    else:
        use_compiled_resolver = False

    if use_compiled_resolver:
        if not compiled_cli_deferred_dataset:
            native_cfg = _build_compiled_cli_config(args, dataset_path)
    else:
        if runner_status.resolved != "compiled-cli":
            dataset_name, dataset_path, dataset_meta = _download_dataset_if_needed(args, encoding_name)
        else:
            dataset_meta = _read_meta(dataset_path)
        output_dir = (
            Path(args.native_cuda_output_dir)
            if str(args.native_cuda_output_dir or "").strip()
            else Path(args.output).with_suffix("")
        )
        native_cfg, dataset_meta = build_native_gpt_run_config(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_meta=dataset_meta,
            encoding_name=encoding_name,
            executable=str(args.native_cuda_executable or "") or None,
            output_dir=output_dir,
            eval_every_steps=int(args.eval_every_steps),
            eval_batches=int(args.eval_batches),
            eval_batch_size=int(args.eval_batch_size),
            lm_head_row_chunk_size=int(args.lm_head_row_chunk_size),
            sample_every_steps=int(args.native_cuda_sample_every),
            generate_tokens=int(args.native_cuda_generate_tokens),
            checkpoint_every_steps=int(args.native_cuda_checkpoint_every),
            batch_size=int(args.batch_size),
            seq_len=int(args.train_seq_len),
            train_batch_tokens=int(args.train_batch_tokens),
            learning_rate=float(args.learning_rate),
            min_lr=args.min_lr,
            warmup_steps=int(args.warmup_steps),
            weight_decay=float(args.weight_decay),
            beta1=float(args.beta1),
            beta2=float(args.beta2),
            adam_eps=float(args.adam_eps),
            grad_clip_norm=float(args.grad_clip_norm),
            max_steps=int(args.max_steps),
            num_layers=int(args.num_layers),
            activation=str(args.native_cuda_activation),
            moa_interval=int(args.native_cuda_moa_interval),
            kernel_backend=str(args.native_cuda_kernel_backend),
            tile_ops_lib=str(args.native_cuda_tile_ops_lib or ""),
            smoke_tile_ops=bool(args.native_cuda_smoke_tile_ops),
            smoke_nvfp4_pack=bool(args.native_cuda_smoke_nvfp4_pack),
            smoke_optimizer_step=bool(args.native_cuda_smoke_optimizer_step),
            smoke_lm_step=bool(args.native_cuda_smoke_lm_step),
            smoke_attention_step=bool(args.native_cuda_smoke_attention_step),
            smoke_mlp_step=bool(args.native_cuda_smoke_mlp_step),
            smoke_norm_residual_step=bool(args.native_cuda_smoke_norm_residual_step),
            smoke_transformer_block_step=bool(args.native_cuda_smoke_transformer_block_step),
            smoke_transformer_lm_step=bool(args.native_cuda_smoke_transformer_lm_step),
            smoke_embedding_lm_step=bool(args.native_cuda_smoke_embedding_lm_step),
            train_embedding_lm=bool(args.train_embedding_lm),
            train_transformer_lm=bool(args.train_transformer_lm),
            require_cooperative_lm_head_backward=bool(args.require_cooperative_lm_head_backward),
            cuda_runtime_lib=str(args.native_cuda_cuda_runtime_lib or ""),
            template_name=str(args.template_name or "gpt"),
            graph_file=str(args.graph_file or ""),
            allow_train_as_val=bool(args.native_cuda_allow_train_val_fallback),
            model_family=str(args.model_family or DEFAULT_MODEL_FAMILY),
            write_checkpoint=not bool(args.native_cuda_no_checkpoint),
        )
    print(f"Using dataset: {dataset_name}")
    print(f"Tokenizer: {encoding_name}")
    print(f"Native CUDA model family: {native_cfg.model_family}")
    print(f"Native CUDA template: {native_cfg.template_name}")
    if str(native_cfg.graph_file or "").strip():
        print(f"Native CUDA graph file: {native_cfg.graph_file}")
    if use_compiled_resolver:
        print(f"Native CUDA dataset path: {dataset_path}")
        if compiled_cli_deferred_dataset:
            print("Native CUDA shard resolution: compiled C++ frontend (deferred)")
        else:
            print("Native CUDA shard resolution: compiled C++ frontend")
    else:
        print(f"Native CUDA train shard: {native_cfg.train_data}")
        print(f"Native CUDA validation shard: {native_cfg.val_data}")
    print(f"Native CUDA validation eval: every {native_cfg.eval_every_steps} optimizer steps")
    print(f"Native CUDA LM-head row chunk size: {native_cfg.lm_head_row_chunk_size}")
    print(f"Native CUDA checkpoint export: {'enabled' if native_cfg.write_checkpoint else 'disabled'}")
    print(f"Native CUDA runner: {runner_status.resolved} (requested={runner_status.requested})")
    print(f"Native CUDA kernel backend: {native_cfg.kernel_backend}")
    if runner_status.reason:
        print(f"Native CUDA runner note: {runner_status.reason}")
    if not use_compiled_resolver:
        rows = _uint16_sequence_count(Path(native_cfg.train_data), seq_len=int(args.train_seq_len))
        if rows is not None:
            print(f"Estimated train rows: {rows}")
    print(f"Native CUDA output dir: {native_cfg.output_dir}")
    if str(args.native_cuda_config_out or "").strip():
        write_native_gpt_run_config(
            native_cfg,
            Path(args.native_cuda_config_out),
            runner=str(args.native_cuda_runner),
        )
        print(f"Native CUDA config: {args.native_cuda_config_out}")
    compiled_cli_args: list[str] | None = None
    if runner_status.resolved == "compiled-cli":
        compiled_cli_args = native_cfg.compiled_cli_argv()
        if bool(args.native_cuda_print_plan):
            compiled_cli_args.append("--print-plan")
        if bool(args.native_cuda_list_templates):
            compiled_cli_args.append("--list-templates")
        if bool(args.native_cuda_check_tile_ops):
            compiled_cli_args.append("--check-tile-ops")
    if bool(args.native_cuda_print_command) or bool(args.native_cuda_dry_run):
        if runner_status.resolved == "compiled-cli":
            import shlex

            print(shlex.join(compiled_cli_args or native_cfg.compiled_cli_argv()))
        elif runner_status.resolved == "launcher":
            print(native_cfg.launcher_command())
        else:
            print(native_cfg.command())
    if bool(args.native_cuda_dry_run):
        return 0
    if (
        bool(args.native_cuda_print_plan)
        or bool(args.native_cuda_list_templates)
        or bool(args.native_cuda_check_tile_ops)
        or bool(args.native_cuda_smoke_tile_ops)
        or bool(args.native_cuda_smoke_nvfp4_pack)
        or bool(args.native_cuda_smoke_optimizer_step)
        or bool(args.native_cuda_smoke_lm_step)
        or bool(args.native_cuda_smoke_attention_step)
        or bool(args.native_cuda_smoke_mlp_step)
        or bool(args.native_cuda_smoke_norm_residual_step)
        or bool(args.native_cuda_smoke_transformer_block_step)
        or bool(args.native_cuda_smoke_transformer_lm_step)
        or bool(args.native_cuda_smoke_embedding_lm_step)
        or bool(args.train_embedding_lm)
        or bool(args.train_transformer_lm)
    ):
        if runner_status.resolved != "compiled-cli":
            print(
                "--native-cuda-print-plan, --native-cuda-list-templates, --native-cuda-check-tile-ops, --native-cuda-smoke-tile-ops, "
                "--native-cuda-smoke-nvfp4-pack, --native-cuda-smoke-optimizer-step, --native-cuda-smoke-lm-step, and "
                "--native-cuda-smoke-attention-step, --native-cuda-smoke-mlp-step, and "
                "--native-cuda-smoke-norm-residual-step, and "
                "--native-cuda-smoke-transformer-block-step, and "
                "--native-cuda-smoke-transformer-lm-step, and "
                "--native-cuda-smoke-embedding-lm-step, and "
                "--train-embedding-lm, and --train-transformer-lm "
                "require --native-cuda-runner compiled-cli.",
                file=sys.stderr,
            )
            return 2
        return _exec_compiled_cli(compiled_cli_args or native_cfg.compiled_cli_argv(), native_cfg)
    LOGGER.info("Launching native CUDA GPT trainer")
    if runner_status.resolved == "compiled-cli":
        return _exec_compiled_cli(compiled_cli_args or native_cfg.compiled_cli_argv(), native_cfg)
    return run_native_gpt(native_cfg, runner=str(args.native_cuda_runner))


if __name__ == "__main__":
    raise SystemExit(main())
