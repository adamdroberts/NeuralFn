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


MODE_NAME = "llama_fast"
GRAPH_NAME = f"{MODE_NAME}_sdk"
DEFAULT_DATASET_ALIAS = "willdepueoai__parameter-golf__sp1024__train1"
DEFAULT_ARTIFACT = artifact_path("llama_fast.pt")
INTERRUPTED_ARTIFACT = DEFAULT_ARTIFACT.with_name("llama_fast.interrupted.pt")

LLAMA_DEFAULTS = {
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
    "eval_every_steps": 1000,
    "max_wallclock_seconds": 0.0,
    "warmup_steps": 60,
    "warmdown_fraction": 0.0,
    "vocab_size": 1_024,
    "num_layers": 5,
    "model_dim": 320,
    "num_heads": 5,
    "num_kv_heads": 5,
    "mlp_mult": 2.0,
    "multiple_of": 64,
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
}


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def mode_name(*, megakernel: bool = False) -> str:
    return "llama_fast_megakernel" if megakernel else MODE_NAME


def default_output_path(*, megakernel: bool = False) -> Path:
    if megakernel:
        return DEFAULT_ARTIFACT.with_name("llama_fast_megakernel.pt")
    return DEFAULT_ARTIFACT


def interrupted_output_path(*, megakernel: bool = False) -> Path:
    if megakernel:
        return DEFAULT_ARTIFACT.with_name("llama_fast_megakernel.interrupted.pt")
    return INTERRUPTED_ARTIFACT


def graph_name(*, megakernel: bool = False) -> str:
    return f"{mode_name(megakernel=megakernel)}_sdk"


def build_parser() -> argparse.ArgumentParser:
    parser = create_argument_parser(description="Train llama_fast through the NeuralFn native CUDA harness.")
    parser.add_argument("--megakernel", action="store_true", help="Select the llama_fast_megakernel template metadata.")
    parser.add_argument("--run-id", default=_env_str("RUN_ID", LLAMA_DEFAULTS["run_id"]))
    parser.add_argument("--seed", type=int, default=_env_int("SEED", LLAMA_DEFAULTS["seed"]))
    parser.add_argument("--device", default=_env_str("DEVICE", LLAMA_DEFAULTS["device"]))
    parser.add_argument("--dataset-alias", default=_env_str("DATASET_ALIAS", LLAMA_DEFAULTS["dataset_alias"]))
    parser.add_argument("--dataset", choices=["golf1", "tinystories"], default=None)
    parser.add_argument("--tinystories", action="store_true")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output", default=_env_str("OUTPUT", ""))
    parser.add_argument("--max-steps", type=int, default=_env_int("ITERATIONS", LLAMA_DEFAULTS["max_steps"]))
    parser.add_argument("--train-seq-len", type=int, default=_env_int("TRAIN_SEQ_LEN", LLAMA_DEFAULTS["train_seq_len"]))
    parser.add_argument("--batch-size", type=int, default=_env_int("BATCH_SIZE", LLAMA_DEFAULTS["batch_size"]))
    parser.add_argument(
        "--train-batch-tokens",
        type=int,
        default=_env_int("TRAIN_BATCH_TOKENS", LLAMA_DEFAULTS["train_batch_tokens"]),
    )
    parser.add_argument("--eval-batches", type=int, default=_env_int("EVAL_BATCHES", LLAMA_DEFAULTS["eval_batches"]))
    parser.add_argument("--eval-batch-size", type=int, default=_env_int("EVAL_BATCH_SIZE", LLAMA_DEFAULTS["eval_batch_size"]))
    parser.add_argument("--eval-every-steps", type=int, default=_env_int("EVAL_EVERY_STEPS", LLAMA_DEFAULTS["eval_every_steps"]))
    parser.add_argument("--train-log-every", type=int, default=_env_int("TRAIN_LOG_EVERY", LLAMA_DEFAULTS["train_log_every"]))
    parser.add_argument("--warmup-steps", type=int, default=_env_int("WARMUP_STEPS", LLAMA_DEFAULTS["warmup_steps"]))
    parser.add_argument("--warmdown-fraction", type=float, default=_env_float("WARMDOWN_FRACTION", LLAMA_DEFAULTS["warmdown_fraction"]))
    parser.add_argument("--max-wallclock-seconds", type=float, default=_env_float("MAX_WALLCLOCK_SECONDS", LLAMA_DEFAULTS["max_wallclock_seconds"]))
    parser.add_argument("--vocab-size", type=int, default=_env_int("VOCAB_SIZE", LLAMA_DEFAULTS["vocab_size"]))
    parser.add_argument("--num-layers", type=int, default=_env_int("NUM_LAYERS", LLAMA_DEFAULTS["num_layers"]))
    parser.add_argument("--model-dim", type=int, default=_env_int("MODEL_DIM", LLAMA_DEFAULTS["model_dim"]))
    parser.add_argument("--num-heads", type=int, default=_env_int("NUM_HEADS", LLAMA_DEFAULTS["num_heads"]))
    parser.add_argument("--num-kv-heads", type=int, default=_env_int("NUM_KV_HEADS", LLAMA_DEFAULTS["num_kv_heads"]))
    parser.add_argument("--optimizer-profile", default=_env_str("OPTIMIZER_PROFILE", LLAMA_DEFAULTS["optimizer_profile"]))
    parser.add_argument("--learning-rate", type=float, default=_env_float("LEARNING_RATE", LLAMA_DEFAULTS["learning_rate"]))
    parser.add_argument("--weight-decay", type=float, default=_env_float("WEIGHT_DECAY", LLAMA_DEFAULTS["weight_decay"]))
    parser.add_argument("--native-cuda-dry-run", "--dry-run", action="store_true")
    parser.add_argument("--native-cuda-print-command", "--print-command", action="store_true")
    parser.add_argument("--native-cuda-print-plan", "--print-plan", action="store_true")
    return parser


def resolve_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not str(getattr(args, "output", "") or "").strip():
        args.output = str(default_output_path(megakernel=bool(getattr(args, "megakernel", False))))
    return args


def main(argv: list[str] | None = None) -> int:
    """Dispatch LLaMA-fast training/preflight to the compiled native frontend."""

    original_argv = sys.argv
    if argv is not None:
        sys.argv = [str(Path(__file__).resolve()), *argv]
    try:
        reject_torch_training_by_default(
            "train_llama_fast.py",
            native_target="nfn train --base-model llama",
            model_family="llama",
            family_native_cli_env="NFN_NATIVE_LLAMA_CLI",
            family_native_cli_name="nfn_llama_native_train",
        )
    finally:
        if argv is not None:
            sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
