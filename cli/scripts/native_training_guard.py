from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


_ALLOW_ENV = "NFN_ALLOW_TORCH_TRAINING"
_NATIVE_ACTION_FLAGS = {
    "--check-tile-ops",
    "--help",
    "--json",
    "--print-plan",
    "--sample-token-batch",
    "--smoke-attention-step",
    "--smoke-embedding-norm-step",
    "--smoke-fused-qkv-attention-step",
    "--smoke-lm-step",
    "--smoke-mlp-step",
    "--smoke-optimizer-step",
    "--smoke-qkv-layout-step",
    "--smoke-tile-ops",
    "--smoke-token-train-step",
    "--smoke-training-loop-step",
    "--smoke-transformer-block-step",
    "--train-token-lm",
}


def torch_training_allowed() -> bool:
    return str(os.environ.get(_ALLOW_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_native_train_cli() -> str:
    requested = os.environ.get("NFN_NATIVE_TRAIN_CLI", "").strip()
    if requested:
        return requested
    root = Path(__file__).resolve().parents[2]
    built = root / "build" / "nfn_native_train"
    if built.exists():
        return str(built)
    return "nfn-native-train"


def _has_native_action(args: list[str]) -> bool:
    for arg in args:
        if arg in _NATIVE_ACTION_FLAGS:
            return True
    return False


def _forwarded_args(model_family: str, native_default_args: list[str]) -> list[str]:
    args = list(sys.argv[1:])
    defaults = [] if _has_native_action(args) else native_default_args
    return [_resolve_native_train_cli(), "--base-model", model_family, *defaults, *args]


def reject_torch_training_by_default(
    script_name: str,
    *,
    native_target: str,
    model_family: str | None = None,
    native_default_args: list[str] | None = None,
) -> None:
    """Exit before legacy training scripts import Torch.

    The project default is native CUDA/C++ training. Legacy graph-backed Python
    harnesses remain importable for tests and SDK migration work, but direct
    script execution should not silently start the slow TorchTrainer path.
    """

    if torch_training_allowed():
        return
    family = (model_family or native_target.rsplit(" ", 1)[-1]).strip()
    command = _forwarded_args(family, list(native_default_args or ()))
    env = os.environ.copy()
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if any(flag in sys.argv[1:] for flag in ("--dry-run", "--native-cuda-dry-run", "--print-command", "--native-cuda-print-command")):
        raise SystemExit(subprocess.run(command, env=env, check=False).returncode)
    try:
        os.execvpe(command[0], command, env)
    except FileNotFoundError:
        message = (
            f"{script_name} is still a graph-backed TorchTrainer harness and is disabled by default.\n"
            f"Default NeuralFn training must use compiled native CUDA/C++ entrypoints, but {command[0]!r} was not found.\n"
            f"Build the native frontend with `bash tools/build_native_train_cli.sh`, install `nfn-native-train`, "
            f"or set NFN_NATIVE_TRAIN_CLI. For one-off legacy debugging only, set {_ALLOW_ENV}=1."
        )
        print(message, file=sys.stderr)
        raise SystemExit(127)
