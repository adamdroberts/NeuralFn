from __future__ import annotations

import os
from pathlib import Path
import subprocess
import shlex
import shutil
import sys


_NATIVE_ACTION_FLAGS = {
    "--check-tile-ops",
    "--help",
    "--json",
    "--native-cuda-check-tile-ops",
    "--native-cuda-print-plan",
    "--native-cuda-smoke-attention-step",
    "--native-cuda-smoke-diffusion-denoise-step",
    "--native-cuda-smoke-diffusion-objective-step",
    "--native-cuda-smoke-diffusion-full-loop-step",
    "--native-cuda-smoke-dense-jepa-full-loop-step",
    "--native-cuda-smoke-dense-jepa-train-step",
    "--native-cuda-smoke-embedding-lm-step",
    "--native-cuda-smoke-embedding-norm-step",
    "--native-cuda-smoke-evo-kernels",
    "--native-cuda-smoke-fused-qkv-attention-step",
    "--native-cuda-smoke-family-layout-checkpoint-step",
    "--native-cuda-smoke-hnet-byte-patch-step",
    "--native-cuda-smoke-hnet-byte-patch-backward-step",
    "--native-cuda-smoke-hnet-byte-lm-loop-step",
    "--native-cuda-smoke-jamba-chunk-state-step",
    "--native-cuda-smoke-jamba-mamba-state-step",
    "--native-cuda-smoke-jamba-layer-schedule-step",
    "--native-cuda-smoke-jepa-ar-loss-step",
    "--native-cuda-smoke-jepa-projector-step",
    "--native-cuda-smoke-jepa-target-encoder-step",
    "--native-cuda-smoke-lm-step",
    "--native-cuda-smoke-llama-lm-head-step",
    "--native-cuda-smoke-llama-loop",
    "--native-cuda-smoke-llama-token-lm-train-step",
    "--native-cuda-smoke-llama-composed-train-step",
    "--native-cuda-smoke-llama-full-loop-step",
    "--native-cuda-smoke-llama-attention-block-step",
    "--native-cuda-smoke-llama-packed-attention-step",
    "--native-cuda-smoke-llama-rope-attention-block-step",
    "--native-cuda-smoke-llama-rope-block-train-step",
    "--native-cuda-smoke-llama-train-step",
    "--native-cuda-smoke-moe-route-expert-step",
    "--native-cuda-smoke-moe-transformer-block-step",
    "--native-cuda-smoke-moe-transformer-block-train-step",
    "--native-cuda-smoke-moe-transformer-lm-train-step",
    "--native-cuda-smoke-moe-full-loop-step",
    "--native-cuda-smoke-moe-jepa-loss-composition-step",
    "--native-cuda-smoke-mlp-step",
    "--native-cuda-smoke-norm-residual-step",
    "--native-cuda-smoke-optimizer-step",
    "--native-cuda-smoke-qkv-layout-step",
    "--native-cuda-smoke-semantic-alignment-step",
    "--native-cuda-smoke-semantic-dense-jepa-train-step",
    "--native-cuda-smoke-semantic-jepa-loss-composition-step",
    "--native-cuda-smoke-semantic-router-moe-train-step",
    "--native-cuda-smoke-semantic-route-loss-step",
    "--native-cuda-smoke-seq2seq-cross-attention-step",
    "--native-cuda-smoke-seq2seq-full-encoder-decoder-loop-step",
    "--native-cuda-smoke-seq2seq-loss-composition-step",
    "--native-cuda-smoke-ttt-composite-inner-step",
    "--native-cuda-smoke-ttt-full-transformer-loop-step",
    "--native-cuda-smoke-ttt-linear-inner-step",
    "--native-cuda-smoke-universal-act-halt-step",
    "--native-cuda-smoke-universal-recurrent-step",
    "--native-cuda-smoke-universal-transformer-loop-step",
    "--native-cuda-smoke-tile-ops",
    "--native-cuda-smoke-token-train-step",
    "--native-cuda-smoke-training-loop-step",
    "--native-cuda-smoke-transformer-block-step",
    "--native-cuda-smoke-transformer-lm-step",
    "--print-plan",
    "--sample-token-batch",
    "--smoke-attention-step",
    "--smoke-diffusion-denoise-step",
    "--smoke-diffusion-objective-step",
    "--smoke-diffusion-full-loop-step",
    "--smoke-dense-jepa-full-loop-step",
    "--smoke-dense-jepa-train-step",
    "--smoke-embedding-lm-step",
    "--smoke-embedding-norm-step",
    "--smoke-fused-qkv-attention-step",
    "--smoke-family-layout-checkpoint-step",
    "--smoke-hnet-byte-patch-step",
    "--smoke-hnet-byte-patch-backward-step",
    "--smoke-hnet-byte-lm-loop-step",
    "--smoke-jamba-chunk-state-step",
    "--smoke-jamba-mamba-state-step",
    "--smoke-jamba-layer-schedule-step",
    "--smoke-jepa-ar-loss-step",
    "--smoke-jepa-target-encoder-step",
    "--smoke-llama-attention-block-step",
    "--smoke-llama-packed-attention-step",
    "--smoke-llama-rope-attention-block-step",
    "--smoke-llama-rope-block-train-step",
    "--smoke-llama-token-lm-train-step",
    "--smoke-llama-composed-train-step",
    "--smoke-llama-full-loop-step",
    "--smoke-lm-step",
    "--smoke-mlp-step",
    "--smoke-moe-transformer-block-step",
    "--smoke-moe-transformer-block-train-step",
    "--smoke-moe-transformer-lm-train-step",
    "--smoke-moe-full-loop-step",
    "--smoke-moe-jepa-loss-composition-step",
    "--smoke-norm-residual-step",
    "--smoke-optimizer-step",
    "--smoke-qkv-layout-step",
    "--smoke-semantic-dense-jepa-train-step",
    "--smoke-semantic-jepa-loss-composition-step",
    "--smoke-semantic-router-moe-train-step",
    "--smoke-semantic-route-loss-step",
    "--smoke-seq2seq-cross-attention-step",
    "--smoke-seq2seq-full-encoder-decoder-loop-step",
    "--smoke-seq2seq-loss-composition-step",
    "--smoke-ttt-composite-inner-step",
    "--smoke-ttt-full-transformer-loop-step",
    "--smoke-ttt-linear-inner-step",
    "--smoke-universal-act-halt-step",
    "--smoke-universal-recurrent-step",
    "--smoke-universal-transformer-loop-step",
    "--smoke-tile-ops",
    "--smoke-token-train-step",
    "--smoke-training-loop-step",
    "--smoke-transformer-block-step",
    "--smoke-transformer-lm-step",
    "--smoke-evo-kernels",
    "--train-transformer-lm",
    "--train-token-lm",
}
_AUTO_CUDA_VISIBLE_DEVICE_VALUES = {"auto", "dedicated", "dedicated-auto"}


def resolve_cuda_visible_devices_value(requested: str | None) -> str:
    value = str(requested or "").strip()
    normalized = value.lower()
    if normalized in {"", "none", "off"}:
        return ""
    if normalized not in _AUTO_CUDA_VISIBLE_DEVICE_VALUES:
        return value
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,display_active,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "0"
    first_index = ""
    best_index = ""
    best_util: int | None = None
    for raw_line in proc.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        index, display, util_text = parts[:3]
        if not first_index:
            first_index = index
        try:
            util = int(util_text)
        except ValueError:
            util = 0
        if display == "Disabled" and (best_util is None or util < best_util):
            best_index = index
            best_util = util
    return best_index or first_index or "0"
_NATIVE_BOOL_ALIASES = {
    "--native-cuda-check-tile-ops": "--check-tile-ops",
    "--native-cuda-dry-run": "--dry-run",
    "--native-cuda-print-command": "--print-command",
    "--native-cuda-print-plan": "--print-plan",
    "--native-cuda-startup-only": "--startup-only",
    "--native-cuda-smoke-attention-step": "--smoke-attention-step",
    "--native-cuda-smoke-diffusion-denoise-step": "--smoke-diffusion-denoise-step",
    "--native-cuda-smoke-diffusion-objective-step": "--smoke-diffusion-objective-step",
    "--native-cuda-smoke-diffusion-full-loop-step": "--smoke-diffusion-full-loop-step",
    "--native-cuda-smoke-dense-jepa-full-loop-step": "--smoke-dense-jepa-full-loop-step",
    "--native-cuda-smoke-dense-jepa-train-step": "--smoke-dense-jepa-train-step",
    "--native-cuda-smoke-embedding-lm-step": "--smoke-embedding-lm-step",
    "--native-cuda-smoke-embedding-norm-step": "--smoke-embedding-norm-step",
    "--native-cuda-smoke-evo-kernels": "--smoke-evo-kernels",
    "--native-cuda-smoke-fused-qkv-attention-step": "--smoke-fused-qkv-attention-step",
    "--native-cuda-smoke-hnet-byte-patch-step": "--smoke-hnet-byte-patch-step",
    "--native-cuda-smoke-hnet-byte-patch-backward-step": "--smoke-hnet-byte-patch-backward-step",
    "--native-cuda-smoke-hnet-byte-lm-loop-step": "--smoke-hnet-byte-lm-loop-step",
    "--native-cuda-smoke-jamba-chunk-state-step": "--smoke-jamba-chunk-state-step",
    "--native-cuda-smoke-jamba-mamba-state-step": "--smoke-jamba-mamba-state-step",
    "--native-cuda-smoke-jamba-layer-schedule-step": "--smoke-jamba-layer-schedule-step",
    "--native-cuda-smoke-jepa-ar-loss-step": "--smoke-jepa-ar-loss-step",
    "--native-cuda-smoke-jepa-projector-step": "--smoke-jepa-projector-step",
    "--native-cuda-smoke-jepa-target-encoder-step": "--smoke-jepa-target-encoder-step",
    "--native-cuda-smoke-lm-step": "--smoke-lm-step",
    "--native-cuda-smoke-llama-lm-head-step": "--smoke-llama-lm-head-step",
    "--native-cuda-smoke-llama-loop": "--smoke-llama-loop",
    "--native-cuda-smoke-llama-token-lm-train-step": "--smoke-llama-token-lm-train-step",
    "--native-cuda-smoke-llama-composed-train-step": "--smoke-llama-composed-train-step",
    "--native-cuda-smoke-llama-full-loop-step": "--smoke-llama-full-loop-step",
    "--native-cuda-smoke-llama-attention-block-step": "--smoke-llama-attention-block-step",
    "--native-cuda-smoke-llama-packed-attention-step": "--smoke-llama-packed-attention-step",
    "--native-cuda-smoke-llama-rope-attention-block-step": "--smoke-llama-rope-attention-block-step",
    "--native-cuda-smoke-llama-rope-block-train-step": "--smoke-llama-rope-block-train-step",
    "--native-cuda-smoke-llama-train-step": "--smoke-llama-train-step",
    "--native-cuda-smoke-moe-route-expert-step": "--smoke-moe-route-expert-step",
    "--native-cuda-smoke-moe-transformer-block-step": "--smoke-moe-transformer-block-step",
    "--native-cuda-smoke-moe-transformer-block-train-step": "--smoke-moe-transformer-block-train-step",
    "--native-cuda-smoke-moe-transformer-lm-train-step": "--smoke-moe-transformer-lm-train-step",
    "--native-cuda-smoke-moe-full-loop-step": "--smoke-moe-full-loop-step",
    "--native-cuda-smoke-moe-jepa-loss-composition-step": "--smoke-moe-jepa-loss-composition-step",
    "--native-cuda-smoke-mlp-step": "--smoke-mlp-step",
    "--native-cuda-smoke-norm-residual-step": "--smoke-norm-residual-step",
    "--native-cuda-smoke-optimizer-step": "--smoke-optimizer-step",
    "--native-cuda-smoke-qkv-layout-step": "--smoke-qkv-layout-step",
    "--native-cuda-smoke-semantic-alignment-step": "--smoke-semantic-alignment-step",
    "--native-cuda-smoke-semantic-dense-jepa-train-step": "--smoke-semantic-dense-jepa-train-step",
    "--native-cuda-smoke-semantic-jepa-loss-composition-step": "--smoke-semantic-jepa-loss-composition-step",
    "--native-cuda-smoke-semantic-router-moe-train-step": "--smoke-semantic-router-moe-train-step",
    "--native-cuda-smoke-semantic-route-loss-step": "--smoke-semantic-route-loss-step",
    "--native-cuda-smoke-seq2seq-cross-attention-step": "--smoke-seq2seq-cross-attention-step",
    "--native-cuda-smoke-seq2seq-full-encoder-decoder-loop-step": "--smoke-seq2seq-full-encoder-decoder-loop-step",
    "--native-cuda-smoke-seq2seq-loss-composition-step": "--smoke-seq2seq-loss-composition-step",
    "--native-cuda-smoke-ttt-composite-inner-step": "--smoke-ttt-composite-inner-step",
    "--native-cuda-smoke-ttt-full-transformer-loop-step": "--smoke-ttt-full-transformer-loop-step",
    "--native-cuda-smoke-ttt-linear-inner-step": "--smoke-ttt-linear-inner-step",
    "--native-cuda-smoke-universal-act-halt-step": "--smoke-universal-act-halt-step",
    "--native-cuda-smoke-universal-recurrent-step": "--smoke-universal-recurrent-step",
    "--native-cuda-smoke-universal-transformer-loop-step": "--smoke-universal-transformer-loop-step",
    "--native-cuda-smoke-tile-ops": "--smoke-tile-ops",
    "--native-cuda-smoke-token-train-step": "--smoke-token-train-step",
    "--native-cuda-smoke-training-loop-step": "--smoke-training-loop-step",
    "--native-cuda-smoke-transformer-block-step": "--smoke-transformer-block-step",
    "--native-cuda-smoke-transformer-lm-step": "--smoke-transformer-lm-step",
    "--native-cuda-allow-train-val-fallback": "--allow-train-val-fallback",
    "--native-cuda-no-checkpoint": "--no-checkpoint",
    "--native-cuda-write-checkpoint": "--write-checkpoint",
}
_NATIVE_VALUE_ALIASES = {
    "--native-cuda-cuda-runtime-lib": "--cuda-runtime-lib",
    "--native-cuda-tile-ops-lib": "--tile-ops-lib",
    "--native-cuda-kernel-backend": "--backend",
    "--native-cuda-executable": "--target",
    "--native-cuda-output-dir": "--output-dir",
    "--native-cuda-lm-head-row-chunk-size": "--lm-head-row-chunk-size",
    "--native-cuda-checkpoint-every": "--native-cuda-checkpoint-every",
    "--native-cuda-sample-every": "--native-cuda-sample-every",
    "--native-cuda-generate-tokens": "--native-cuda-generate-tokens",
    "--native-cuda-activation": "--native-cuda-activation",
    "--native-cuda-moa-interval": "--native-cuda-moa-interval",
}
_NATIVE_EXECUTION_FLAGS = {
    "--check-tile-ops",
    "--print-plan",
    "--sample-token-batch",
    "--smoke-attention-step",
    "--smoke-diffusion-denoise-step",
    "--smoke-diffusion-objective-step",
    "--smoke-diffusion-full-loop-step",
    "--smoke-embedding-lm-step",
    "--smoke-embedding-norm-step",
    "--smoke-evo-kernels",
    "--smoke-fused-qkv-attention-step",
    "--smoke-hnet-byte-patch-step",
    "--smoke-hnet-byte-patch-backward-step",
    "--smoke-hnet-byte-lm-loop-step",
    "--smoke-jamba-chunk-state-step",
    "--smoke-jamba-mamba-state-step",
    "--smoke-jamba-layer-schedule-step",
    "--smoke-jepa-ar-loss-step",
    "--smoke-jepa-target-encoder-step",
    "--smoke-llama-attention-block-step",
    "--smoke-llama-packed-attention-step",
    "--smoke-llama-rope-attention-block-step",
    "--smoke-llama-rope-block-train-step",
    "--smoke-llama-token-lm-train-step",
    "--smoke-llama-composed-train-step",
    "--smoke-llama-full-loop-step",
    "--smoke-lm-step",
    "--smoke-mlp-step",
    "--smoke-moe-transformer-block-step",
    "--smoke-moe-transformer-block-train-step",
    "--smoke-moe-transformer-lm-train-step",
    "--smoke-moe-full-loop-step",
    "--smoke-norm-residual-step",
    "--smoke-optimizer-step",
    "--smoke-qkv-layout-step",
    "--smoke-semantic-route-loss-step",
    "--smoke-seq2seq-cross-attention-step",
    "--smoke-seq2seq-full-encoder-decoder-loop-step",
    "--smoke-ttt-composite-inner-step",
    "--smoke-ttt-full-transformer-loop-step",
    "--smoke-ttt-linear-inner-step",
    "--smoke-universal-act-halt-step",
    "--smoke-universal-recurrent-step",
    "--smoke-universal-transformer-loop-step",
    "--smoke-tile-ops",
    "--smoke-token-train-step",
    "--smoke-training-loop-step",
    "--smoke-transformer-block-step",
    "--smoke-transformer-lm-step",
    "--startup-only",
}


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
    if command_name == "nfn_gpt_native_train":
        linked = root / "build" / "nfn_gpt_native_train_linked"
        if linked.exists():
            return str(linked)
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
    forwarded = [command, *defaults, *args]
    if Path(command).name in {"nfn_gpt_native_train_linked", "nfn-gpt-native-train-linked"}:
        has_tile_ops_lib = any(
            arg == "--tile-ops-lib"
            or arg == "--native-cuda-tile-ops-lib"
            or arg.startswith("--tile-ops-lib=")
            or arg.startswith("--native-cuda-tile-ops-lib=")
            for arg in forwarded
        )
        if not has_tile_ops_lib:
            forwarded.extend(["--tile-ops-lib", "linked"])
    return forwarded


def _set_env_default_if_empty(env: dict[str, str], key: str, value: str) -> None:
    if value and not str(env.get(key, "")).strip():
        env[key] = value


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
    _set_env_default_if_empty(env, "CUDA_VISIBLE_DEVICES", resolve_cuda_visible_devices_value("0"))
    _set_env_default_if_empty(env, "CUDA_DEVICE_MAX_CONNECTIONS", "1")
    _set_env_default_if_empty(env, "CUDA_MODULE_LOADING", "LAZY")
    if (
        "--dry-run" in command
        and "--print-command" in command
        and not any(flag in command for flag in _NATIVE_EXECUTION_FLAGS)
    ):
        print(shlex.join(command))
        raise SystemExit(0)
    if any(flag in sys.argv[1:] for flag in ("--dry-run", "--native-cuda-dry-run", "--print-command", "--native-cuda-print-command")):
        raise SystemExit(subprocess.run(command, env=env, check=False).returncode)
    try:
        os.execvpe(command[0], command, env)
    except FileNotFoundError:
        message = (
            f"{script_name} is still a graph-backed TorchTrainer harness and is disabled by default.\n"
            f"Default NeuralFn training must use compiled native CUDA/C++ entrypoints, but {command[0]!r} was not found.\n"
            f"Build the native frontend with `bash tools/build_native_train_cli.sh`, install `nfn-native-train`, "
            f"or set NFN_NATIVE_TRAIN_CLI. Legacy graph-backed experiments must call the Python SDK trainer APIs directly."
        )
        print(message, file=sys.stderr)
        raise SystemExit(127)
