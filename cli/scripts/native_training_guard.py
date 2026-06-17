from __future__ import annotations

import os
from pathlib import Path
import subprocess
import shutil
import sys


_ALLOW_ENV = "NFN_ALLOW_TORCH_TRAINING"
_NATIVE_ACTION_FLAGS = {
    "--check-tile-ops",
    "--help",
    "--json",
    "--native-cuda-check-tile-ops",
    "--native-cuda-print-plan",
    "--native-cuda-smoke-attention-step",
    "--native-cuda-smoke-embedding-lm-step",
    "--native-cuda-smoke-embedding-norm-step",
    "--native-cuda-smoke-evo-kernels",
    "--native-cuda-smoke-fused-qkv-attention-step",
    "--native-cuda-smoke-lm-step",
    "--native-cuda-smoke-mlp-step",
    "--native-cuda-smoke-norm-residual-step",
    "--native-cuda-smoke-optimizer-step",
    "--native-cuda-smoke-qkv-layout-step",
    "--native-cuda-smoke-tile-ops",
    "--native-cuda-smoke-token-train-step",
    "--native-cuda-smoke-training-loop-step",
    "--native-cuda-smoke-transformer-block-step",
    "--native-cuda-smoke-transformer-lm-step",
    "--print-plan",
    "--sample-token-batch",
    "--smoke-attention-step",
    "--smoke-embedding-lm-step",
    "--smoke-embedding-norm-step",
    "--smoke-fused-qkv-attention-step",
    "--smoke-lm-step",
    "--smoke-mlp-step",
    "--smoke-norm-residual-step",
    "--smoke-optimizer-step",
    "--smoke-qkv-layout-step",
    "--smoke-tile-ops",
    "--smoke-token-train-step",
    "--smoke-training-loop-step",
    "--smoke-transformer-block-step",
    "--smoke-transformer-lm-step",
    "--smoke-evo-kernels",
    "--train-token-lm",
}
_NATIVE_BOOL_ALIASES = {
    "--native-cuda-check-tile-ops": "--check-tile-ops",
    "--native-cuda-print-plan": "--print-plan",
    "--native-cuda-smoke-attention-step": "--smoke-attention-step",
    "--native-cuda-smoke-embedding-lm-step": "--smoke-embedding-lm-step",
    "--native-cuda-smoke-embedding-norm-step": "--smoke-embedding-norm-step",
    "--native-cuda-smoke-evo-kernels": "--smoke-evo-kernels",
    "--native-cuda-smoke-fused-qkv-attention-step": "--smoke-fused-qkv-attention-step",
    "--native-cuda-smoke-lm-step": "--smoke-lm-step",
    "--native-cuda-smoke-mlp-step": "--smoke-mlp-step",
    "--native-cuda-smoke-norm-residual-step": "--smoke-norm-residual-step",
    "--native-cuda-smoke-optimizer-step": "--smoke-optimizer-step",
    "--native-cuda-smoke-qkv-layout-step": "--smoke-qkv-layout-step",
    "--native-cuda-smoke-tile-ops": "--smoke-tile-ops",
    "--native-cuda-smoke-token-train-step": "--smoke-token-train-step",
    "--native-cuda-smoke-training-loop-step": "--smoke-training-loop-step",
    "--native-cuda-smoke-transformer-block-step": "--smoke-transformer-block-step",
    "--native-cuda-smoke-transformer-lm-step": "--smoke-transformer-lm-step",
}
_NATIVE_VALUE_ALIASES = {
    "--native-cuda-cuda-runtime-lib": "--cuda-runtime-lib",
    "--native-cuda-tile-ops-lib": "--tile-ops-lib",
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


def _resolve_family_native_cli(env_var: str, command_name: str) -> str | None:
    requested = os.environ.get(env_var, "").strip()
    if requested:
        return requested
    root = Path(__file__).resolve().parents[2]
    built = root / "build" / command_name
    if built.exists():
        return str(built)
    resolved = shutil.which(command_name)
    if resolved:
        return resolved
    return None


def _family_native_cli_env(model_family: str) -> str:
    suffix = "".join(ch if ch.isalnum() else "_" for ch in model_family.upper()).strip("_")
    return f"NFN_NATIVE_{suffix}_CLI"


def _family_native_cli_name(model_family: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in model_family.lower()).strip("_")
    return f"nfn_{normalized}_native_train"


def _has_native_action(args: list[str]) -> bool:
    for arg in args:
        if arg in _NATIVE_ACTION_FLAGS:
            return True
    return False


def _normalize_forwarded_args(args: list[str]) -> list[str]:
    normalized: list[str] = []
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg in _NATIVE_BOOL_ALIASES:
            normalized.append(_NATIVE_BOOL_ALIASES[arg])
            idx += 1
            continue
        value_alias = _NATIVE_VALUE_ALIASES.get(arg)
        if value_alias is not None:
            normalized.append(value_alias)
            if idx + 1 < len(args):
                normalized.append(args[idx + 1])
            idx += 2
            continue
        matched_value_alias = next((flag for flag in _NATIVE_VALUE_ALIASES if arg.startswith(flag + "=")), None)
        if matched_value_alias is not None:
            normalized.append(f"{_NATIVE_VALUE_ALIASES[matched_value_alias]}={arg.split('=', 1)[1]}")
            idx += 1
            continue
        normalized.append(arg)
        idx += 1
    return normalized


def _forwarded_args(model_family: str, native_default_args: list[str]) -> list[str]:
    args = _normalize_forwarded_args(list(sys.argv[1:]))
    defaults = [] if _has_native_action(args) else native_default_args
    return [_resolve_native_train_cli(), "--base-model", model_family, *defaults, *args]


def _family_forwarded_args(command: str, native_default_args: list[str]) -> list[str]:
    args = _normalize_forwarded_args(list(sys.argv[1:]))
    defaults = [] if _has_native_action(args) else native_default_args
    return [command, *defaults, *args]


def reject_torch_training_by_default(
    script_name: str,
    *,
    native_target: str,
    model_family: str | None = None,
    native_default_args: list[str] | None = None,
    family_native_cli_env: str | None = None,
    family_native_cli_name: str | None = None,
) -> None:
    """Exit before legacy training scripts import Torch.

    The project default is native CUDA/C++ training. Legacy graph-backed Python
    harnesses remain importable for tests and SDK migration work, but direct
    script execution should not silently start the slow TorchTrainer path.
    """

    if torch_training_allowed():
        return
    family = (model_family or native_target.rsplit(" ", 1)[-1]).strip()
    resolved_family_env = family_native_cli_env or _family_native_cli_env(family)
    resolved_family_name = family_native_cli_name or _family_native_cli_name(family)
    family_command = (
        _resolve_family_native_cli(resolved_family_env, resolved_family_name)
        if resolved_family_env and resolved_family_name
        else None
    )
    default_args = list(native_default_args or ())
    command = _family_forwarded_args(family_command, default_args) if family_command else _forwarded_args(family, default_args)
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
