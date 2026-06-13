from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for candidate in (SCRIPT_DIR, REPO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


_TINYSTORIES_ALIAS = "roneneldan__TinyStories__TinyStoriesV2-GPT4"
_DEFAULT_NATIVE_GPT_TARGET = "/mnt/disk2/dev/open-source/llm.kittens/train_gpt2cu"


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
    return str(REPO_ROOT / "build" / "nfn_gpt_native_train")


def _native_target_path() -> str:
    requested = os.environ.get("NFN_NATIVE_GPT_TRAIN_BIN", "").strip()
    if requested:
        return requested
    requested = os.environ.get("NFN_NATIVE_GPT2_TRAIN_BIN", "").strip()
    if requested:
        return requested
    default_path = Path(_DEFAULT_NATIVE_GPT_TARGET)
    return str(default_path if default_path.exists() else "train_gpt2cu")


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
    return (_arg_value(argv, "--template-name", "--template", "--preset") or "gpt2").strip().lower().replace("-", "_")


def _native_model_family(argv: list[str]) -> str:
    return (_arg_value(argv, "--model-family", "--base-model", "--model") or "gpt").strip().lower().replace("_", "-")


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
    if runner.strip().lower().replace("_", "-") not in {"", "compiled-cli", "cli"}:
        return None
    if _has_any(argv, "-h", "--help", "--native-cuda-config-out") or any(
        arg.startswith("--native-cuda-config-out=") for arg in argv
    ):
        return None
    if (_arg_value(argv, "--runtime") or "native-cuda").strip().lower().replace("_", "-") != "native-cuda":
        return None

    out = [_native_cli_path()]
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
        "--native-cuda-check-tile-ops": "--check-tile-ops",
        "--native-cuda-smoke-tile-ops": "--smoke-tile-ops",
        "--native-cuda-smoke-optimizer-step": "--smoke-optimizer-step",
        "--native-cuda-smoke-lm-step": "--smoke-lm-step",
        "--native-cuda-smoke-attention-step": "--smoke-attention-step",
        "--native-cuda-smoke-mlp-step": "--smoke-mlp-step",
        "--native-cuda-smoke-norm-residual-step": "--smoke-norm-residual-step",
        "--native-cuda-smoke-transformer-block-step": "--smoke-transformer-block-step",
        "--native-cuda-smoke-transformer-lm-step": "--smoke-transformer-lm-step",
        "--native-cuda-smoke-embedding-lm-step": "--smoke-embedding-lm-step",
        "--native-cuda-allow-train-val-fallback": "--allow-train-val-fallback",
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

    model_family = _native_model_family(out)
    if "--model-family" not in out:
        _append_value(out, "--model-family", model_family)
    if (
        model_family == "gpt3"
        and not _explicit_arg(out, "--train-seq-len")
        and not _explicit_arg(out, "--template-name", "--template", "--preset")
        and not _explicit_arg(out, "--graph-file", "--graph")
    ):
        _append_value(out, "--train-seq-len", "2048")
    if "--dataset-alias" not in out and "--dataset-path" not in out and "--tinystories" not in out:
        _append_value(out, "--dataset-alias", os.environ.get("DATASET_ALIAS", _TINYSTORIES_ALIAS))
    if "--target" not in out and _native_backend_name(out) == "llm-kittens":
        _append_value(out, "--target", _native_target_path())
    final_lr = _final_lr_fraction(argv)
    if final_lr is not None and "--final-lr-fraction" not in out:
        _append_value(out, "--final-lr-fraction", final_lr)
    if _native_template_name(out) == "gpt2_moa" and not _has_native_activation(out):
        _append_value(out, "--native-cuda-activation", "moa")
    if "--train-transformer-lm" not in out and "--no-train-transformer-lm" not in out:
        out.append("--train-transformer-lm")
    return out


def _fast_compiled_cli_main(argv: list[str]) -> int | None:
    command = _fast_compiled_cli_argv(argv)
    if command is None:
        return None
    env = os.environ.copy()
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "--dry-run" in command or "--print-command" in command or "--print-plan" in command or "--check-tile-ops" in command:
        return int(subprocess.run(command, env=env, check=False).returncode)
    os.execvpe(command[0], command, env)
    return 127


if __name__ == "__main__":
    fast_exit = _fast_compiled_cli_main(sys.argv[1:])
    if fast_exit is not None:
        raise SystemExit(fast_exit)

from train_gpt2_native import NATIVE_GPT_DEFAULTS, build_parser, main  # noqa: E402


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


def resolve_mode_defaults(args):
    if not str(getattr(args, "output", "") or "").strip():
        args.output = GPT_DEFAULTS["output"]
    return args


if __name__ == "__main__":
    raise SystemExit(main())
