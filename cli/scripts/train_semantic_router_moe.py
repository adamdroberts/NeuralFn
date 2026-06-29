from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import uuid


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cli_utils import artifact_path, create_argument_parser
from native_training_guard import reject_torch_training_by_default


MODE_NAME = "semantic_router_moe"
GRAPH_NAME = f"{MODE_NAME}_sdk"
DEFAULT_DATASET_ALIAS = "willdepueoai__parameter-golf__sp1024__train1"
DEFAULT_ARTIFACT = artifact_path("semantic_router_moe.pt")
INTERRUPTED_ARTIFACT = DEFAULT_ARTIFACT.with_name("semantic_router_moe.interrupted.pt")
DEFAULT_SEMANTIC_ROUTER_EXPERTS = 86

ROUTER_DEFAULTS = {
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
    "warmup_steps": 600,
    "warmdown_fraction": 0.0,
    "vocab_size": 1_024,
    "num_layers": 4,
    "model_dim": 256,
    "num_heads": 4,
    "num_kv_heads": 4,
    "experts": DEFAULT_SEMANTIC_ROUTER_EXPERTS,
    "top_k": 2,
    "optimizer_profile": "adamw",
    "learning_rate": 6e-4,
    "weight_decay": 0.1,
}


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def mode_name(*, megakernel: bool = False) -> str:
    return "semantic_router_moe_megakernel" if megakernel else MODE_NAME


def default_output_path(*, megakernel: bool = False) -> Path:
    if megakernel:
        return DEFAULT_ARTIFACT.with_name("semantic_router_moe_megakernel.pt")
    return DEFAULT_ARTIFACT


def interrupted_output_path(*, megakernel: bool = False) -> Path:
    if megakernel:
        return DEFAULT_ARTIFACT.with_name("semantic_router_moe_megakernel.interrupted.pt")
    return INTERRUPTED_ARTIFACT


def graph_name(*, megakernel: bool = False) -> str:
    return f"{mode_name(megakernel=megakernel)}_sdk"


def build_parser() -> argparse.ArgumentParser:
    parser = create_argument_parser(description="Train semantic_router_moe through the NeuralFn native CUDA harness.")
    parser.add_argument("--megakernel", action="store_true", help="Select the semantic_router_moe_megakernel template metadata.")
    parser.add_argument("--run-id", default=_env_str("RUN_ID", ROUTER_DEFAULTS["run_id"]))
    parser.add_argument("--seed", type=int, default=_env_int("SEED", ROUTER_DEFAULTS["seed"]))
    parser.add_argument("--device", default=_env_str("DEVICE", ROUTER_DEFAULTS["device"]))
    parser.add_argument("--dataset-alias", default=_env_str("DATASET_ALIAS", ROUTER_DEFAULTS["dataset_alias"]))
    parser.add_argument("--dataset", choices=["golf1", "tinystories"], default=None)
    parser.add_argument("--tinystories", action="store_true")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output", default=_env_str("OUTPUT", ""))
    parser.add_argument("--max-steps", type=int, default=_env_int("ITERATIONS", ROUTER_DEFAULTS["max_steps"]))
    parser.add_argument("--train-seq-len", type=int, default=_env_int("TRAIN_SEQ_LEN", ROUTER_DEFAULTS["train_seq_len"]))
    parser.add_argument("--batch-size", type=int, default=_env_int("BATCH_SIZE", ROUTER_DEFAULTS["batch_size"]))
    parser.add_argument(
        "--train-batch-tokens",
        type=int,
        default=_env_int("TRAIN_BATCH_TOKENS", ROUTER_DEFAULTS["train_batch_tokens"]),
    )
    parser.add_argument("--eval-batches", type=int, default=_env_int("EVAL_BATCHES", ROUTER_DEFAULTS["eval_batches"]))
    parser.add_argument("--eval-batch-size", type=int, default=_env_int("EVAL_BATCH_SIZE", ROUTER_DEFAULTS["eval_batch_size"]))
    parser.add_argument("--train-log-every", type=int, default=_env_int("TRAIN_LOG_EVERY", ROUTER_DEFAULTS["train_log_every"]))
    parser.add_argument("--val-loss-every", type=int, default=_env_int("VAL_LOSS_EVERY", ROUTER_DEFAULTS["val_loss_every"]))
    parser.add_argument("--warmup-steps", type=int, default=_env_int("WARMUP_STEPS", ROUTER_DEFAULTS["warmup_steps"]))
    parser.add_argument("--warmdown-fraction", type=float, default=_env_float("WARMDOWN_FRACTION", ROUTER_DEFAULTS["warmdown_fraction"]))
    parser.add_argument("--max-wallclock-seconds", type=float, default=_env_float("MAX_WALLCLOCK_SECONDS", ROUTER_DEFAULTS["max_wallclock_seconds"]))
    parser.add_argument("--vocab-size", type=int, default=_env_int("VOCAB_SIZE", ROUTER_DEFAULTS["vocab_size"]))
    parser.add_argument("--num-layers", type=int, default=_env_int("NUM_LAYERS", ROUTER_DEFAULTS["num_layers"]))
    parser.add_argument("--model-dim", type=int, default=_env_int("MODEL_DIM", ROUTER_DEFAULTS["model_dim"]))
    parser.add_argument("--num-heads", type=int, default=_env_int("NUM_HEADS", ROUTER_DEFAULTS["num_heads"]))
    parser.add_argument("--num-kv-heads", type=int, default=_env_int("NUM_KV_HEADS", ROUTER_DEFAULTS["num_kv_heads"]))
    parser.add_argument("--experts", type=int, default=_env_int("EXPERTS", ROUTER_DEFAULTS["experts"]))
    parser.add_argument("--top-k", type=int, default=_env_int("TOP_K", ROUTER_DEFAULTS["top_k"]))
    parser.add_argument("--optimizer-profile", default=_env_str("OPTIMIZER_PROFILE", ROUTER_DEFAULTS["optimizer_profile"]))
    parser.add_argument("--learning-rate", type=float, default=_env_float("LEARNING_RATE", ROUTER_DEFAULTS["learning_rate"]))
    parser.add_argument("--weight-decay", type=float, default=_env_float("WEIGHT_DECAY", ROUTER_DEFAULTS["weight_decay"]))
    parser.add_argument("--native-cuda-dry-run", "--dry-run", action="store_true")
    parser.add_argument("--native-cuda-print-command", "--print-command", action="store_true")
    parser.add_argument("--native-cuda-print-plan", "--print-plan", action="store_true")
    return parser


def resolve_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not str(getattr(args, "output", "") or "").strip():
        args.output = str(default_output_path(megakernel=bool(getattr(args, "megakernel", False))))
    return args


def main(argv: list[str] | None = None) -> int:
    """Dispatch semantic-router MoE training/preflight to the compiled native frontend."""

    original_argv = sys.argv
    if argv is not None:
        sys.argv = [str(Path(__file__).resolve()), *argv]
    try:
        reject_torch_training_by_default(
            "train_semantic_router_moe.py",
            native_target="nfn train --base-model semantic-router-moe",
            model_family="semantic-router-moe",
            family_native_cli_env="NFN_NATIVE_SEMANTIC_ROUTER_MOE_CLI",
            family_native_cli_name="nfn_semantic_router_moe_native_train",
        )
    finally:
        if argv is not None:
            sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
