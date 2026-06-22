from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cli_utils import create_argument_parser
from native_training_guard import reject_torch_training_by_default
from train_llama_fast import LLAMA_DEFAULTS, _env_float, _env_int, _env_str
from train_llama_fast import default_output_path as _fast_default_output_path


MODE_NAME = "llama_megakernel"
GRAPH_NAME = f"{MODE_NAME}_sdk"


def mode_name(*, fast: bool = False) -> str:
    return "llama_fast_megakernel" if fast else MODE_NAME


def default_output_path(*, fast: bool = False) -> Path:
    if fast:
        return _fast_default_output_path(megakernel=True)
    return _fast_default_output_path(megakernel=False).with_name("llama_megakernel.pt")


def interrupted_output_path(*, fast: bool = False) -> Path:
    return default_output_path(fast=fast).with_name(f"{mode_name(fast=fast)}.interrupted.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = create_argument_parser(description="Train llama_megakernel through the NeuralFn native CUDA harness.")
    parser.add_argument("--fast", action="store_true", help="Select the llama_fast_megakernel template metadata.")
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
        args.output = str(default_output_path(fast=bool(getattr(args, "fast", False))))
    return args


def main(argv: list[str] | None = None) -> int:
    """Dispatch LLaMA megakernel training/preflight to the compiled native frontend."""

    original_argv = sys.argv
    if argv is not None:
        sys.argv = [str(Path(__file__).resolve()), *argv]
    try:
        reject_torch_training_by_default(
            "train_llama_megakernel.py",
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
