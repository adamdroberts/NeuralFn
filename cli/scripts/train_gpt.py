from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for candidate in (SCRIPT_DIR, REPO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

_TINYSTORIES_ALIAS = "roneneldan__TinyStories__TinyStoriesV2-GPT4"
_DEFAULT_EVAL_BATCHES = "20"
_AUTO_CUDA_VISIBLE_DEVICE_VALUES = {"auto", "dedicated", "dedicated-auto"}
_NATIVE_METADATA_ACTION_FLAGS = {
    "--print-plan",
    "--list-templates",
    "--check-tile-ops",
    "--startup-only",
    "--smoke-tile-ops",
    "--smoke-nvfp4-pack",
    "--smoke-optimizer-step",
    "--smoke-lm-step",
    "--smoke-attention-step",
    "--smoke-mlp-step",
    "--smoke-norm-residual-step",
    "--smoke-transformer-block-step",
    "--smoke-transformer-lm-step",
    "--smoke-embedding-lm-step",
}


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


def _arg_value(argv: list[str], *flags: str) -> str | None:
    for idx, arg in enumerate(argv):
        for flag in flags:
            if arg == flag and idx + 1 < len(argv):
                return argv[idx + 1]
            if arg.startswith(flag + "="):
                return arg.split("=", 1)[1]
    return None


def _has_any(argv: list[str], *flags: str) -> bool:
    return any(arg in flags for arg in argv)


def _explicit_arg(argv: list[str], *flags: str) -> bool:
    return any(arg in flags or any(arg.startswith(flag + "=") for flag in flags) for arg in argv)


def _native_cli_path() -> str:
    requested = os.environ.get("NFN_NATIVE_GPT_CLI", "").strip()
    if requested:
        return requested
    requested = os.environ.get("NFN_NATIVE_GPT2_CLI", "").strip()
    if requested:
        return requested
    linked = REPO_ROOT / "build" / "nfn_gpt_native_train_linked"
    if linked.exists():
        return str(linked)
    return str(REPO_ROOT / "build" / "nfn_gpt_native_train")


def _native_cli_uses_linked_tile_ops(path: str) -> bool:
    return Path(path).name in {"nfn_gpt_native_train_linked", "nfn-gpt-native-train-linked"}


def _append_value(out: list[str], flag: str, value: str) -> None:
    out.extend([flag, value])


def _output_dir_from_output(value: str) -> str:
    path = Path(value).expanduser()
    if path.suffix:
        path = path.with_suffix("")
    return str(path)


def _final_lr_fraction(argv: list[str]) -> str | None:
    explicit = _arg_value(argv, "--final-lr-fraction")
    if explicit is not None:
        return explicit
    min_lr = _arg_value(argv, "--min-lr")
    if min_lr is None:
        return None
    lr = float(_arg_value(argv, "--learning-rate") or os.environ.get("LEARNING_RATE", "0.0006"))
    if lr <= 0.0:
        return "0"
    return str(max(0.0, min(float(min_lr) / lr, 1.0)))


def _native_template_name(argv: list[str]) -> str:
    return (_arg_value(argv, "--template-name", "--template", "--preset") or "gpt").strip().lower().replace("-", "_")


def _default_native_model_family() -> str:
    invoked = Path(sys.argv[0]).stem.lower().replace("_", "-")
    if invoked == "train-gpt2":
        return "gpt2"
    return "gpt"


def _native_model_family(argv: list[str]) -> str:
    return (
        _arg_value(argv, "--model-family", "--base-model", "--model")
        or _default_native_model_family()
    ).strip().lower().replace("_", "-")


def _canonical_dense_gpt_model_family(model: str) -> str:
    return "nanogpt" if model in {"nano-gpt", "nano_gpt"} else model


def _set_split_value(out: list[str], flag: str, value: str) -> None:
    for idx, arg in enumerate(out):
        if arg == flag:
            if idx + 1 < len(out):
                out[idx + 1] = value
            else:
                out.append(value)
            return
        if arg.startswith(flag + "="):
            out[idx] = flag
            out.insert(idx + 1, value)
            return
    _append_value(out, flag, value)


def _remove_split_or_bool_flags(out: list[str], *flags: str) -> None:
    remove = set(flags)
    idx = 0
    while idx < len(out):
        arg = out[idx]
        if arg in remove:
            del out[idx]
            if arg != "--tinystories" and idx < len(out):
                del out[idx]
            continue
        if any(arg.startswith(flag + "=") for flag in remove):
            del out[idx]
            continue
        idx += 1


def _native_backend_name(argv: list[str]) -> str:
    return (_arg_value(argv, "--backend") or "tile-cuda").strip().lower().replace("_", "-")


def _has_native_activation(argv: list[str]) -> bool:
    return any(
        arg in {"--activation", "--native-cuda-activation"} or
        arg.startswith("--activation=") or
        arg.startswith("--native-cuda-activation=")
        for arg in argv
    )


def _fast_compiled_cli_argv(argv: list[str]) -> list[str] | None:
    runner = (
        _arg_value(argv, "--native-cuda-runner")
        or os.environ.get("NFN_NATIVE_GPT_RUNNER")
        or os.environ.get("NFN_NATIVE_GPT2_RUNNER", "compiled-cli")
    )
    if runner.strip().lower().replace("_", "-") not in {"", "auto", "compiled-cli"}:
        return None
    if _has_any(argv, "-h", "--help", "--native-cuda-config-out") or any(
        arg.startswith("--native-cuda-config-out=") for arg in argv
    ):
        return None
    if (_arg_value(argv, "--runtime") or "native-cuda").strip().lower().replace("_", "-") != "native-cuda":
        return None

    native_cli = _native_cli_path()
    out = [native_cli]
    tile_ops_lib_explicit = _explicit_arg(argv, "--tile-ops-lib", "--native-cuda-tile-ops-lib")
    idx = 0
    drop_value_flags = {
        "--runtime",
        "--run-id",
        "--seed",
        "--device",
        "--dataset-hf-path",
        "--dataset-variant",
        "--dataset-train-shards",
        "--dataset-train-file",
        "--dataset-val-file",
        "--tokenizer",
        "--min-lr",
        "--native-cuda-runner",
    }
    drop_bool_flags = {
        "--download-if-missing",
        "--no-download-if-missing",
        "--tokgpt2",
        "--cl100k",
        "--o200k",
        "--tile-cuda-strict",
        "--no-tile-cuda-strict",
    }
    value_aliases = {
        "--model-family": "--model-family",
        "--base-model": "--model-family",
        "--model": "--model-family",
        "--kernel-backend": "--backend",
        "--native-cuda-kernel-backend": "--backend",
        "--native-cuda-executable": "--target",
        "--native-cuda-output-dir": "--output-dir",
        "--native-cuda-tile-ops-lib": "--tile-ops-lib",
        "--native-cuda-cuda-runtime-lib": "--cuda-runtime-lib",
        "--native-cuda-lm-head-row-chunk-size": "--lm-head-row-chunk-size",
        "--native-cuda-checkpoint-every": "--native-cuda-checkpoint-every",
        "--native-cuda-sample-every": "--native-cuda-sample-every",
        "--native-cuda-generate-tokens": "--native-cuda-generate-tokens",
        "--native-cuda-activation": "--native-cuda-activation",
        "--native-cuda-moa-interval": "--native-cuda-moa-interval",
        "--template": "--template-name",
        "--preset": "--template-name",
        "--graph": "--graph-file",
    }
    bool_aliases = {
        "--native-cuda-dry-run": "--dry-run",
        "--native-cuda-print-command": "--print-command",
        "--native-cuda-print-plan": "--print-plan",
        "--native-cuda-list-templates": "--list-templates",
        "--native-cuda-startup-only": "--startup-only",
        "--native-cuda-check-tile-ops": "--check-tile-ops",
        "--native-cuda-smoke-tile-ops": "--smoke-tile-ops",
        "--native-cuda-smoke-nvfp4-pack": "--smoke-nvfp4-pack",
        "--native-cuda-smoke-optimizer-step": "--smoke-optimizer-step",
        "--native-cuda-smoke-lm-step": "--smoke-lm-step",
        "--native-cuda-smoke-attention-step": "--smoke-attention-step",
        "--native-cuda-smoke-mlp-step": "--smoke-mlp-step",
        "--native-cuda-smoke-norm-residual-step": "--smoke-norm-residual-step",
        "--native-cuda-smoke-transformer-block-step": "--smoke-transformer-block-step",
        "--native-cuda-smoke-transformer-lm-step": "--smoke-transformer-lm-step",
        "--native-cuda-smoke-embedding-lm-step": "--smoke-embedding-lm-step",
        "--native-cuda-allow-train-val-fallback": "--allow-train-val-fallback",
        "--native-cuda-no-checkpoint": "--no-checkpoint",
        "--no-checkpoint": "--no-checkpoint",
        "--native-cuda-write-checkpoint": "--write-checkpoint",
        "--write-checkpoint": "--write-checkpoint",
        "--native-cuda-require-cooperative-lm-head-backward": "--require-cooperative-lm-head-backward",
        "--require-cooperative-lm-head-backward": "--require-cooperative-lm-head-backward",
        "--native-cuda-fast-startup": "--fast-startup",
        "--fast-startup": "--fast-startup",
    }
    pass_value_flags = {
        "--model-family",
        "--dataset-alias",
        "--dataset-path",
        "--target",
        "--output-dir",
        "--eval-every-steps",
        "--eval-batches",
        "--eval-batch-size",
        "--train-loss-every-steps",
        "--train-log-every",
        "--train-log-every-steps",
        "--lm-head-row-chunk-size",
        "--batch-size",
        "--train-seq-len",
        "--train-batch-tokens",
        "--learning-rate",
        "--final-lr-fraction",
        "--weight-decay",
        "--warmup-steps",
        "--max-steps",
        "--num-layers",
        "--template-name",
        "--template",
        "--preset",
        "--graph-file",
        "--graph",
        "--cuda-runtime-lib",
        "--tile-ops-lib",
        "--activation",
        "--moa-interval",
    }

    while idx < len(argv):
        arg = argv[idx]
        if arg in drop_value_flags:
            idx += 2
            continue
        if any(arg.startswith(flag + "=") for flag in drop_value_flags):
            idx += 1
            continue
        if arg in drop_bool_flags:
            idx += 1
            continue
        if arg == "--tinystories":
            out.append("--tinystories")
            idx += 1
            continue
        if arg == "--dataset":
            if idx + 1 < len(argv):
                dataset = argv[idx + 1].strip().lower()
                if dataset == "tinystories":
                    out.append("--tinystories")
                elif dataset in {"golf1", "golf10"}:
                    shard_count = "1" if dataset == "golf1" else "10"
                    _append_value(out, "--dataset-alias", f"willdepueoai__parameter-golf__sp1024__train{shard_count}")
                else:
                    _append_value(out, "--dataset-alias", argv[idx + 1])
            idx += 2
            continue
        if arg.startswith("--dataset="):
            dataset = arg.split("=", 1)[1].strip().lower()
            if dataset == "tinystories":
                out.append("--tinystories")
            elif dataset in {"golf1", "golf10"}:
                shard_count = "1" if dataset == "golf1" else "10"
                _append_value(out, "--dataset-alias", f"willdepueoai__parameter-golf__sp1024__train{shard_count}")
            else:
                _append_value(out, "--dataset-alias", arg.split("=", 1)[1])
            idx += 1
            continue
        if arg == "--output":
            if idx + 1 < len(argv):
                _append_value(out, "--output-dir", _output_dir_from_output(argv[idx + 1]))
            idx += 2
            continue
        if arg.startswith("--output="):
            _append_value(out, "--output-dir", _output_dir_from_output(arg.split("=", 1)[1]))
            idx += 1
            continue
        if arg in value_aliases:
            if idx + 1 < len(argv):
                _append_value(out, value_aliases[arg], argv[idx + 1])
            idx += 2
            continue
        matched_alias = next((flag for flag in value_aliases if arg.startswith(flag + "=")), None)
        if matched_alias is not None:
            _append_value(out, value_aliases[matched_alias], arg.split("=", 1)[1])
            idx += 1
            continue
        if arg in bool_aliases:
            out.append(bool_aliases[arg])
            idx += 1
            continue
        if arg in pass_value_flags:
            if idx + 1 < len(argv):
                _append_value(out, arg, argv[idx + 1])
            idx += 2
            continue
        matched_pass = next((flag for flag in pass_value_flags if arg.startswith(flag + "=")), None)
        if matched_pass is not None:
            _append_value(out, matched_pass, arg.split("=", 1)[1])
            idx += 1
            continue
        out.append(arg)
        idx += 1

    model_selector = _native_model_family(out)
    model_family = _canonical_dense_gpt_model_family(model_selector)
    _set_split_value(out, "--model-family", model_family)
    list_templates_only = "--list-templates" in out
    if list_templates_only:
        _remove_split_or_bool_flags(
            out,
            "--tinystories",
            "--dataset-alias",
            "--dataset-path",
            "--eval-every-steps",
            "--eval-batches",
            "--eval-batch-size",
            "--train-loss-every-steps",
            "--train-log-every",
            "--train-log-every-steps",
        )
    if (
        model_selector == "gpt3"
        and not _explicit_arg(out, "--train-seq-len")
        and not _explicit_arg(out, "--template-name", "--template", "--preset")
        and not _explicit_arg(out, "--graph-file", "--graph")
    ):
        _append_value(out, "--train-seq-len", "2048")
    if (
        model_selector == "nanogpt"
        and not _explicit_arg(out, "--template-name", "--template", "--preset")
        and not _explicit_arg(out, "--graph-file", "--graph")
    ):
        _append_value(out, "--template-name", "nanogpt")
    if (
        not list_templates_only
        and "--dataset-alias" not in out
        and "--dataset-path" not in out
        and "--tinystories" not in out
    ):
        _append_value(out, "--dataset-alias", os.environ.get("DATASET_ALIAS", _TINYSTORIES_ALIAS))
    if _native_backend_name(out) != "tile-cuda":
        raise ValueError("native GPT kernel backend must be tile-cuda")
    if "--backend" not in out:
        _append_value(out, "--backend", "tile-cuda")
    if not list_templates_only and not _explicit_arg(out, "--eval-batches"):
        _append_value(out, "--eval-batches", os.environ.get("EVAL_BATCHES", _DEFAULT_EVAL_BATCHES))
    final_lr = _final_lr_fraction(argv)
    if final_lr is not None and "--final-lr-fraction" not in out:
        _append_value(out, "--final-lr-fraction", final_lr)
    if _native_template_name(out) == "gpt2_moa" and not _has_native_activation(out):
        _append_value(out, "--native-cuda-activation", "moa")
    if _native_cli_uses_linked_tile_ops(native_cli) and not tile_ops_lib_explicit:
        _append_value(out, "--tile-ops-lib", "linked")
    if (
        "--train-transformer-lm" not in out
        and "--no-train-transformer-lm" not in out
        and not any(flag in out for flag in _NATIVE_METADATA_ACTION_FLAGS)
    ):
        out.append("--train-transformer-lm")
    return out


def _fast_compiled_cli_main(argv: list[str]) -> int | None:
    try:
        command = _fast_compiled_cli_argv(argv)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if command is None:
        return None
    env = os.environ.copy()
    _set_env_default_if_empty(env, "CUDA_VISIBLE_DEVICES", resolve_cuda_visible_devices_value("dedicated"))
    _set_env_default_if_empty(env, "CUDA_DEVICE_MAX_CONNECTIONS", "1")
    _set_env_default_if_empty(env, "CUDA_MODULE_LOADING", "LAZY")
    if "--dry-run" in command and "--print-command" in command and not any(flag in command for flag in _NATIVE_METADATA_ACTION_FLAGS):
        print(shlex.join(command))
        return 0
    if (
        "--dry-run" in command
        or "--print-command" in command
        or "--print-plan" in command
        or "--list-templates" in command
        or "--check-tile-ops" in command
    ):
        return _run_compiled_cli_capture(command, env)
    os.execvpe(command[0], command, env)
    return 127


def _run_compiled_cli_capture(command: list[str], env: dict[str, str]) -> int:
    try:
        from neuralfn.native_gpt import run_native_gpt_compiled_cli_capture

        result = run_native_gpt_compiled_cli_capture(
            command,
            cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES", ""),
            cuda_device_max_connections=env.get("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
        )
    except (ImportError, RuntimeError, ValueError):
        return int(subprocess.run(command, env=env, check=False).returncode)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return int(result.returncode)


def _set_env_default_if_empty(env: dict[str, str], key: str, value: str) -> None:
    if value and not str(env.get(key, "")).strip():
        env[key] = value


if __name__ == "__main__":
    fast_exit = _fast_compiled_cli_main(sys.argv[1:])
    if fast_exit is not None:
        raise SystemExit(fast_exit)

from train_gpt_native import NATIVE_GPT_DEFAULTS, build_parser as _native_build_parser  # noqa: E402


GPT_DEFAULTS = {
    **NATIVE_GPT_DEFAULTS,
    "eval_batches": 20,
    "eval_batch_size": 64,
    "train_log_every": 1,
    "max_wallclock_seconds": 0.0,
    "warmdown_fraction": 0.0,
    "vocab_size": 1_024,
    "model_dim": 768,
    "num_heads": 12,
    "logit_softcap": 0.0,
    "optimizer_profile": "adamw",
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
    "kernel_backend": "tile-cuda",
    "tile_cuda_strict": True,
    "tile_cuda_activation_dtype": "nvfp4",
    "amp_dtype": "bfloat16",
    "runtime": "native-cuda",
}
GPT2_DEFAULTS = GPT_DEFAULTS
EVOLUTIONARY_DEFAULTS = {
    "population_size": 8,
    "mutation_rate": 0.1,
    "mutation_scale": 0.02,
    "crossover_rate": 0.5,
    "tournament_size": 3,
    "elite_count": 1,
}
TINYSTORIES_HF_PATH = "roneneldan/TinyStories"
TINYSTORIES_ALIAS = _TINYSTORIES_ALIAS
TINYSTORIES_TRAIN_FILE = "TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_VAL_FILE = "TinyStoriesV2-GPT4-valid.txt"
PARAMETER_GOLF_HF_PATH = "willdepueoai/parameter-golf"
DEFAULT_CACHED_TOKENIZER_VARIANT = "sp1024"
DEFAULT_ARTIFACT = Path(GPT_DEFAULTS["output"])
DEFAULT_GRAPH_ARTIFACT = DEFAULT_ARTIFACT.with_suffix(".json")
RAW_TEXT_VOCAB_SIZES = {
    "gpt2": 50257,
    "cl100k_base": 100277,
    "o200k_base": 200019,
    "sp1024": 1024,
    "sp2048": 2048,
    "sp4096": 4096,
    "sp8192": 8192,
}


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float | None = None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _add_argument_if_missing(parser: argparse.ArgumentParser, *flags: str, **kwargs) -> None:
    existing = set(parser._option_string_actions)  # noqa: SLF001 - argparse has no public lookup helper.
    if any(flag in existing for flag in flags):
        return
    parser.add_argument(*flags, **kwargs)


def parameter_golf_dataset_alias(*, train_shards: int, variant: str = DEFAULT_CACHED_TOKENIZER_VARIANT) -> str:
    return f"willdepueoai__parameter-golf__{variant}__train{int(train_shards)}"


def _selected_tokenizer(args: argparse.Namespace) -> str:
    has_override_flag = any(bool(getattr(args, name, False)) for name in ("tokgpt2", "cl100k", "o200k"))
    if bool(getattr(args, "_tokenizer_explicit", False)) and str(getattr(args, "tokenizer", "") or "").strip() and has_override_flag:
        raise ValueError("--tokenizer cannot be combined with --tokgpt2, --cl100k, or --o200k")
    if getattr(args, "tokgpt2", False):
        return "gpt2"
    if getattr(args, "cl100k", False):
        return "cl100k_base"
    if getattr(args, "o200k", False):
        return "o200k_base"
    return str(getattr(args, "tokenizer", "") or "gpt2").strip()


def _apply_raw_text_tokenizer(args: argparse.Namespace, encoding_name: str) -> None:
    vocab_size = RAW_TEXT_VOCAB_SIZES.get(encoding_name)
    if vocab_size is None:
        raise ValueError(f"Unsupported raw-text tokenizer {encoding_name!r}.")
    args.raw_text_selected = True
    args.raw_text_encoding_name = encoding_name
    args.raw_text_encoding_vocab_size = vocab_size
    current_vocab = int(getattr(args, "vocab_size", 0) or 0)
    args._raw_text_vocab_conflict = current_vocab not in (0, GPT_DEFAULTS["vocab_size"], vocab_size)
    if not args._raw_text_vocab_conflict:
        args.vocab_size = vocab_size


def apply_tinystories_dataset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not getattr(args, "tinystories", False):
        return args
    conflicts: list[str] = []
    if getattr(args, "dataset", None):
        conflicts.append("--dataset")
    if str(getattr(args, "dataset_alias", "") or "") not in ("", GPT_DEFAULTS["dataset_alias"], TINYSTORIES_ALIAS):
        conflicts.append("--dataset-alias")
    for field_name, flag, allowed in (
        ("dataset_hf_path", "--dataset-hf-path", {"", TINYSTORIES_HF_PATH, str(NATIVE_GPT_DEFAULTS.get("dataset_hf_path", ""))}),
        ("dataset_variant", "--dataset-variant", {"", None}),
        ("dataset_train_file", "--dataset-train-file", {"", TINYSTORIES_TRAIN_FILE, str(NATIVE_GPT_DEFAULTS.get("dataset_train_file", ""))}),
        ("dataset_val_file", "--dataset-val-file", {"", TINYSTORIES_VAL_FILE, str(NATIVE_GPT_DEFAULTS.get("dataset_val_file", ""))}),
    ):
        if getattr(args, field_name, None) not in allowed:
            conflicts.append(flag)
    if getattr(args, "dataset_train_shards", None) is not None:
        conflicts.append("--dataset-train-shards")
    if conflicts:
        raise ValueError(f"--tinystories cannot be combined with {', '.join(conflicts)}")
    args.dataset_alias = TINYSTORIES_ALIAS
    args.dataset_hf_path = TINYSTORIES_HF_PATH
    args.dataset_train_file = TINYSTORIES_TRAIN_FILE
    args.dataset_val_file = TINYSTORIES_VAL_FILE
    return args


def resolve_dataset_selector_args(args: argparse.Namespace) -> str:
    tokenizer = _selected_tokenizer(args)
    dataset = str(getattr(args, "dataset", "") or "").strip().lower()
    if dataset == "tinystories":
        args.dataset_alias = TINYSTORIES_ALIAS
        args.dataset_hf_path = TINYSTORIES_HF_PATH
        args.dataset_train_file = TINYSTORIES_TRAIN_FILE
        args.dataset_val_file = TINYSTORIES_VAL_FILE
        has_tokenizer_override = bool(
            getattr(args, "_tokenizer_explicit", False)
            or getattr(args, "tokgpt2", False)
            or getattr(args, "cl100k", False)
            or getattr(args, "o200k", False)
        )
        _apply_raw_text_tokenizer(args, tokenizer if has_tokenizer_override else "o200k_base")
    elif getattr(args, "tinystories", False):
        apply_tinystories_dataset_defaults(args)
        has_tokenizer_override = bool(
            getattr(args, "_tokenizer_explicit", False)
            or getattr(args, "tokgpt2", False)
            or getattr(args, "cl100k", False)
            or getattr(args, "o200k", False)
        )
        _apply_raw_text_tokenizer(args, tokenizer if has_tokenizer_override else "o200k_base")
    elif dataset in {"golf1", "golf10"}:
        variant = str(getattr(args, "dataset_variant", "") or tokenizer or DEFAULT_CACHED_TOKENIZER_VARIANT)
        if variant not in {"sp1024", "sp2048", "sp4096", "sp8192"}:
            variant = DEFAULT_CACHED_TOKENIZER_VARIANT
        shards = 1 if dataset == "golf1" else 10
        args.dataset_variant = variant
        args.dataset_hf_path = PARAMETER_GOLF_HF_PATH
        args.dataset_train_shards = shards
        args.dataset_alias = parameter_golf_dataset_alias(train_shards=shards, variant=variant)
        _apply_raw_text_tokenizer(args, variant)
    else:
        _apply_raw_text_tokenizer(args, tokenizer)
    return str(args.dataset_alias)


def resolve_mode_defaults(args):
    if not str(getattr(args, "output", "") or "").strip():
        args.output = str(default_output_path(megakernel=bool(getattr(args, "megakernel", False))))
    if getattr(args, "lr_decay_iters", None) is None:
        args.lr_decay_iters = _env_int("LR_DECAY_ITERS", None)
    if getattr(args, "min_lr", None) is None:
        env_min_lr = _env_float("MIN_LR", None)
        if env_min_lr is not None:
            args.min_lr = env_min_lr
        elif getattr(args, "lr_decay_iters", None) is not None:
            args.min_lr = float(getattr(args, "learning_rate", GPT_DEFAULTS["learning_rate"])) * 0.1
    args.warmdown_fraction = float(getattr(args, "warmdown_fraction", 0.0) or 0.0)
    if bool(getattr(args, "_raw_text_vocab_conflict", False)):
        raise ValueError(
            f"Dataset/tokenizer {getattr(args, 'raw_text_encoding_name', '')!r} requires "
            f"vocab_size={getattr(args, 'raw_text_encoding_vocab_size', None)}; got {getattr(args, 'vocab_size', None)}."
        )
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = _native_build_parser()
    _add_argument_if_missing(parser, "--pretraining-file", default="")
    _add_argument_if_missing(parser, "--vocab-size", type=int, default=GPT_DEFAULTS["vocab_size"])
    _add_argument_if_missing(parser, "--lr-decay-iters", type=int, default=_env_int("LR_DECAY_ITERS", None))
    _add_argument_if_missing(parser, "--warmdown-fraction", type=float, default=_env_float("WARMDOWN_FRACTION", 0.0))
    _add_argument_if_missing(parser, "--max-wallclock-seconds", type=float, default=_env_float("MAX_WALLCLOCK_SECONDS", 0.0))
    _add_argument_if_missing(parser, "--optimizer-profile", default=GPT_DEFAULTS["optimizer_profile"])
    _add_argument_if_missing(parser, "--embed-lr", type=float, default=GPT_DEFAULTS["embed_lr"])
    _add_argument_if_missing(parser, "--head-lr", type=float, default=GPT_DEFAULTS["head_lr"])
    _add_argument_if_missing(parser, "--tied-embed-lr", type=float, default=GPT_DEFAULTS["tied_embed_lr"])
    _add_argument_if_missing(parser, "--matrix-lr", type=float, default=GPT_DEFAULTS["matrix_lr"])
    _add_argument_if_missing(parser, "--scalar-lr", type=float, default=GPT_DEFAULTS["scalar_lr"])
    _add_argument_if_missing(parser, "--grad-clip-norm", type=float, default=GPT_DEFAULTS["grad_clip_norm"])
    _add_argument_if_missing(parser, "--all-train-rows", action="store_true")
    _add_argument_if_missing(parser, "--megakernel", action="store_true")
    _add_argument_if_missing(parser, "--evolutionary", action="store_true")
    _add_argument_if_missing(parser, "--evo-population-size", type=int, default=EVOLUTIONARY_DEFAULTS["population_size"])
    _add_argument_if_missing(parser, "--evo-mutation-rate", type=float, default=EVOLUTIONARY_DEFAULTS["mutation_rate"])
    _add_argument_if_missing(parser, "--evo-mutation-scale", type=float, default=EVOLUTIONARY_DEFAULTS["mutation_scale"])
    _add_argument_if_missing(parser, "--evo-crossover-rate", type=float, default=EVOLUTIONARY_DEFAULTS["crossover_rate"])
    _add_argument_if_missing(parser, "--evo-tournament-size", type=int, default=EVOLUTIONARY_DEFAULTS["tournament_size"])
    _add_argument_if_missing(parser, "--evo-elite-count", type=int, default=EVOLUTIONARY_DEFAULTS["elite_count"])
    _add_argument_if_missing(parser, "--evo-seed", type=int, default=None)
    original_parse_args = parser.parse_args

    def parse_args_with_tokenizer_check(args=None, namespace=None):  # type: ignore[no-untyped-def]
        explicit_args = list(sys.argv[1:] if args is None else args)
        parsed = original_parse_args(args, namespace)
        parsed._tokenizer_explicit = any(  # noqa: SLF001 - private marker for wrapper compatibility only.
            token == "--tokenizer" or str(token).startswith("--tokenizer=")
            for token in explicit_args
        )
        parsed._max_steps_explicit = any(
            token == "--max-steps" or str(token).startswith("--max-steps=")
            for token in explicit_args
        ) or ("ITERATIONS" in os.environ)
        parsed._max_wallclock_seconds_explicit = any(
            token == "--max-wallclock-seconds" or str(token).startswith("--max-wallclock-seconds=")
            for token in explicit_args
        ) or ("MAX_WALLCLOCK_SECONDS" in os.environ)
        if getattr(parsed, "min_lr", None) is None:
            env_min_lr = _env_float("MIN_LR", None)
            if env_min_lr is not None:
                parsed.min_lr = env_min_lr
        try:
            _selected_tokenizer(parsed)
        except ValueError as exc:
            parser.error(str(exc))
        return parsed

    parser.parse_args = parse_args_with_tokenizer_check  # type: ignore[method-assign]
    return parser


def build_graph(args: argparse.Namespace, dataset_name: str):
    runtime = "megakernel" if bool(getattr(args, "megakernel", False)) else "eager"
    spec = SimpleNamespace(
        template_name=getattr(args, "template_name", "gpt"),
        dataset_name=dataset_name,
        vocab_size=getattr(args, "vocab_size", GPT_DEFAULTS["vocab_size"]),
        template=SimpleNamespace(runtime=runtime),
    )
    return SimpleNamespace(name=graph_name(megakernel=bool(getattr(args, "megakernel", False))), nodes=[]), spec


def build_trainer_config(args: argparse.Namespace, *, resolved_epochs: int, **overrides) -> SimpleNamespace:
    return SimpleNamespace(
        epochs=resolved_epochs,
        batch_size=getattr(args, "batch_size", GPT_DEFAULTS["batch_size"]),
        learning_rate=getattr(args, "learning_rate", GPT_DEFAULTS["learning_rate"]),
        weight_decay=getattr(args, "weight_decay", GPT_DEFAULTS["weight_decay"]),
        device=getattr(args, "device", GPT_DEFAULTS["device"]),
        max_steps=overrides.get("max_steps", getattr(args, "max_steps", GPT_DEFAULTS["max_steps"])),
        optimizer_profile=getattr(args, "optimizer_profile", GPT_DEFAULTS["optimizer_profile"]),
        train_batch_tokens=getattr(args, "train_batch_tokens", GPT_DEFAULTS["train_batch_tokens"]),
        warmup_steps=getattr(args, "warmup_steps", GPT_DEFAULTS["warmup_steps"]),
        warmdown_fraction=getattr(args, "warmdown_fraction", 0.0),
        lr_decay_iters=overrides.get("lr_decay_iters", getattr(args, "lr_decay_iters", None)),
        min_lr=getattr(args, "min_lr", None),
        grad_clip_norm=getattr(args, "grad_clip_norm", GPT_DEFAULTS["grad_clip_norm"]),
        drop_last=overrides.get("drop_last", None),
        respect_epoch_boundaries=overrides.get("respect_epoch_boundaries", bool(getattr(args, "all_train_rows", False))),
        evolutionary=bool(getattr(args, "evolutionary", False)),
        evo_population_size=int(getattr(args, "evo_population_size", EVOLUTIONARY_DEFAULTS["population_size"])),
        evo_mutation_rate=float(getattr(args, "evo_mutation_rate", EVOLUTIONARY_DEFAULTS["mutation_rate"])),
        evo_mutation_scale=float(getattr(args, "evo_mutation_scale", EVOLUTIONARY_DEFAULTS["mutation_scale"])),
        evo_crossover_rate=float(getattr(args, "evo_crossover_rate", EVOLUTIONARY_DEFAULTS["crossover_rate"])),
        evo_tournament_size=int(getattr(args, "evo_tournament_size", EVOLUTIONARY_DEFAULTS["tournament_size"])),
        evo_elite_count=int(getattr(args, "evo_elite_count", EVOLUTIONARY_DEFAULTS["elite_count"])),
        evo_seed=int(getattr(args, "seed", GPT_DEFAULTS["seed"]) if getattr(args, "evo_seed", None) is None else args.evo_seed),
    )


def _trainer_summary(trainer_cfg: SimpleNamespace) -> dict[str, object]:
    summary = dict(vars(trainer_cfg))
    summary["optimization_method"] = "evolutionary" if bool(getattr(trainer_cfg, "evolutionary", False)) else "gradient_descent"
    if bool(getattr(trainer_cfg, "evolutionary", False)):
        summary["evolutionary"] = {
            "population_size": trainer_cfg.evo_population_size,
            "mutation_rate": trainer_cfg.evo_mutation_rate,
            "mutation_scale": trainer_cfg.evo_mutation_scale,
            "crossover_rate": trainer_cfg.evo_crossover_rate,
            "tournament_size": trainer_cfg.evo_tournament_size,
            "elite_count": trainer_cfg.evo_elite_count,
            "seed": trainer_cfg.evo_seed,
        }
        summary["ignored_gradient_optimizer_fields"] = ["learning_rate", "weight_decay", "warmup_steps", "grad_clip_norm"]
    return summary


def _namespace_to_jsonable(value):
    if isinstance(value, SimpleNamespace):
        return {key: _namespace_to_jsonable(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _namespace_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_namespace_to_jsonable(item) for item in value]
    return value


def print_resolved_summary(
    args: argparse.Namespace,
    spec,
    trainer_cfg: SimpleNamespace,
    derived: dict[str, object],
) -> dict[str, object]:
    summary = {
        "run_id": getattr(args, "run_id", ""),
        "seed": getattr(args, "seed", GPT_DEFAULTS["seed"]),
        "device": getattr(args, "device", GPT_DEFAULTS["device"]),
        "dataset_alias": getattr(args, "dataset_alias", ""),
        "artifact_path": getattr(args, "output", ""),
        "model_spec": _namespace_to_jsonable(spec) if hasattr(spec, "__dict__") else {},
        "trainer": _trainer_summary(trainer_cfg),
        "derived_schedule": derived,
    }
    print("Resolved training configuration:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def configure_console_logging() -> None:
    return None


def mode_name(*, megakernel: bool) -> str:
    return "gpt2_megakernel" if megakernel else "gpt2"


def default_output_path(*, megakernel: bool) -> Path:
    return DEFAULT_ARTIFACT.with_name("gpt2_megakernel.pt") if megakernel else DEFAULT_ARTIFACT.with_name("gpt2.pt")


def graph_name(*, megakernel: bool) -> str:
    return f"{mode_name(megakernel=megakernel)}_sdk"


def resolve_pretraining_file_dataset(args: argparse.Namespace) -> str | None:
    pretraining_file = str(getattr(args, "pretraining_file", "") or "").strip()
    if not pretraining_file:
        return None
    conflicts: list[str] = []
    if getattr(args, "tinystories", False):
        conflicts.append("--tinystories")
    if getattr(args, "dataset", None):
        conflicts.append("--dataset")
    if conflicts:
        raise ValueError(f"--pretraining-file cannot be combined with {', '.join(conflicts)}")
    corpus = Path(pretraining_file).expanduser().resolve()
    if not corpus.exists():
        raise FileNotFoundError(f"Pretraining file {corpus} does not exist.")
    if corpus.suffix.lower() != ".txt":
        raise ValueError("--pretraining-file must point to a .txt file.")
    adapter_dir = Path(tempfile.mkdtemp(prefix="nfn-pretraining-file-"))
    data_link = adapter_dir / "data.txt"
    try:
        data_link.symlink_to(corpus)
    except OSError:
        data_link.write_text(corpus.read_text(encoding="utf-8"), encoding="utf-8")
    meta = {
        "source": "pretraining_file",
        "raw_text": True,
        "data_file": "data.txt",
        "source_file": str(corpus),
        "tokenizer_backend": "tiktoken",
        "tokenizer_encoding": getattr(args, "raw_text_encoding_name", "gpt2"),
        "tokenizer_vocab_size": getattr(args, "raw_text_encoding_vocab_size", RAW_TEXT_VOCAB_SIZES["gpt2"]),
    }
    (adapter_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    args.pretraining_file = str(corpus)
    args.dataset_alias = str(adapter_dir)
    args.download_if_missing = False
    args.raw_text_selected = True
    if not getattr(args, "raw_text_encoding_name", None):
        _apply_raw_text_tokenizer(args, "gpt2")
    return str(adapter_dir)


def resolve_or_download_dataset(dataset_alias: str, **_kwargs):
    return dataset_alias, Path(dataset_alias), {}


def estimate_text_schedule(*_args, **_kwargs):
    return {}


def resolve_effective_training_schedule(
    args: argparse.Namespace,
    derived: dict[str, object],
) -> tuple[dict[str, object], int, int, int | None, float]:
    steps_per_epoch = max(int(derived.get("steps_per_epoch", 1)), 1)
    requested_max_steps = int(getattr(args, "max_steps", GPT_DEFAULTS["max_steps"]))
    requested_for_rounding = max(requested_max_steps, 1)
    if bool(getattr(args, "all_train_rows", False)) and not bool(getattr(args, "_max_steps_explicit", False)):
        requested_for_rounding = steps_per_epoch * 2
    resolved_max_steps = requested_for_rounding
    if bool(getattr(args, "all_train_rows", False)):
        resolved_max_steps = max(steps_per_epoch, ((requested_for_rounding + steps_per_epoch - 1) // steps_per_epoch) * steps_per_epoch)
    resolved_epochs = max(1, (max(resolved_max_steps, 1) + steps_per_epoch - 1) // steps_per_epoch)
    requested_lr_decay_iters = None if getattr(args, "lr_decay_iters", None) is None else int(args.lr_decay_iters)
    requested_wallclock = float(getattr(args, "max_wallclock_seconds", 0.0) or 0.0)
    resolved_wallclock = requested_wallclock
    if bool(getattr(args, "all_train_rows", False)) and not bool(getattr(args, "_max_wallclock_seconds_explicit", False)):
        resolved_wallclock = 0.0
    resolved = {
        **derived,
        "all_train_rows": bool(getattr(args, "all_train_rows", False)),
        "requested_max_steps": requested_max_steps,
        "resolved_max_steps": resolved_max_steps,
        "requested_lr_decay_iters": requested_lr_decay_iters,
        "resolved_lr_decay_iters": requested_lr_decay_iters,
        "requested_max_wallclock_seconds": requested_wallclock,
        "resolved_max_wallclock_seconds": resolved_wallclock,
        "resolved_epochs": resolved_epochs,
        "default_all_train_rows_epochs": 2,
    }
    return resolved, resolved_epochs, resolved_max_steps, requested_lr_decay_iters, resolved_wallclock


def _apply_cached_vocab_from_alias(args: argparse.Namespace, dataset_name: str) -> None:
    for variant, vocab_size in RAW_TEXT_VOCAB_SIZES.items():
        if variant.startswith("sp") and f"__{variant}__" in str(dataset_name):
            args.vocab_size = vocab_size
            args.raw_text_encoding_name = variant
            args.raw_text_encoding_vocab_size = vocab_size
            args.raw_text_selected = True
            return


def main(argv: list[str] | None = None) -> int:
    explicit = list(sys.argv[1:] if argv is None else argv)
    configure_console_logging()
    parser = build_parser()
    args = parser.parse_args(explicit)
    apply_tinystories_dataset_defaults(args)
    dataset_name = resolve_dataset_selector_args(args)
    resolve_mode_defaults(args)
    dataset_name, _dataset_path, _dataset_meta = resolve_or_download_dataset(dataset_name)
    _apply_cached_vocab_from_alias(args, dataset_name)
    graph, spec = build_graph(args, dataset_name)
    trainer_cfg = build_trainer_config(args, resolved_epochs=1)
    print_resolved_summary(args, spec, trainer_cfg, {"steps_per_epoch": 1})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
