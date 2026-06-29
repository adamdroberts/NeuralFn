from __future__ import annotations

import argparse
from pathlib import Path
import sys
import uuid


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from native_training_guard import reject_torch_training_by_default


MODE_NAME = "nanogpt"
GRAPH_NAME = f"{MODE_NAME}_sdk"
DEFAULT_ARTIFACT = Path("artifacts") / "nanogpt.pt"
INTERRUPTED_ARTIFACT = Path("artifacts") / "nanogpt.interrupted.pt"
DEFAULT_DATASET_ALIAS = "roneneldan__TinyStories__TinyStoriesV2-GPT4"

NANOGPT_DEFAULTS = {
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
    "eval_every_steps": 1000,
    "train_log_every": 1,
    "max_wallclock_seconds": 0.0,
    "warmup_steps": 60,
    "warmdown_fraction": 0.0,
    "vocab_size": 1_024,
    "num_layers": 5,
    "model_dim": 320,
    "num_heads": 5,
    "bias": False,
    "dropout_p": 0.1,
    "logit_softcap": 0.0,
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
    "template_name": MODE_NAME,
    "model_family": "gpt",
}


def mode_name(*, megakernel: bool = False) -> str:
    return "nanogpt_megakernel" if megakernel else MODE_NAME


def default_output_path(*, megakernel: bool = False) -> Path:
    if megakernel:
        return DEFAULT_ARTIFACT.with_name("nanogpt_megakernel.pt")
    return DEFAULT_ARTIFACT


def interrupted_output_path(*, megakernel: bool = False) -> Path:
    if megakernel:
        return DEFAULT_ARTIFACT.with_name("nanogpt_megakernel.interrupted.pt")
    return INTERRUPTED_ARTIFACT


def graph_name(*, megakernel: bool = False) -> str:
    return f"{mode_name(megakernel=megakernel)}_sdk"


def build_parser() -> argparse.ArgumentParser:
    from train_gpt_native import build_parser as _native_gpt_build_parser

    parser = _native_gpt_build_parser()
    parser.description = "Train nanogpt with the NeuralFn native CUDA harness."
    parser.set_defaults(
        model_family="gpt",
        template_name=MODE_NAME,
        train_transformer_lm=True,
        output="",
    )
    return parser


def resolve_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not str(getattr(args, "output", "") or "").strip():
        args.output = str(default_output_path(megakernel=False))
    args.model_family = "gpt"
    args.template_name = MODE_NAME
    if not hasattr(args, "runtime"):
        args.runtime = "native-cuda"
    return args


def main(argv: list[str] | None = None) -> int:
    """Dispatch NanoGPT training to the compiled native CUDA/C++ frontend."""

    original_argv = sys.argv
    if argv is not None:
        sys.argv = [str(Path(__file__).resolve()), *argv]
    try:
        native_args = set(sys.argv[1:])
        token_lm_native = bool(
            native_args
            & {
                "--train-token-lm",
                "--native-cuda-smoke-token-train-step",
                "--smoke-token-train-step",
                "--native-cuda-smoke-embedding-norm-step",
                "--smoke-embedding-norm-step",
                "--native-cuda-smoke-qkv-layout-step",
                "--smoke-qkv-layout-step",
                "--native-cuda-smoke-fused-qkv-attention-step",
                "--smoke-fused-qkv-attention-step",
            }
        )
        reject_torch_training_by_default(
            "train_nanogpt.py",
            native_target=(
                "nfn train --base-model nanogpt --train-token-lm"
                if token_lm_native
                else "nfn train --base-model nanogpt --train-transformer-lm"
            ),
            model_family="nanogpt" if token_lm_native else "gpt",
            native_default_args=[] if token_lm_native else ["--template-name", MODE_NAME, "--train-transformer-lm"],
            family_native_cli_env="NFN_NATIVE_NANOGPT_CLI" if token_lm_native else "NFN_NATIVE_GPT_CLI",
            family_native_cli_name="nfn_nanogpt_native_train" if token_lm_native else "nfn_gpt_native_train",
        )
    finally:
        if argv is not None:
            sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
