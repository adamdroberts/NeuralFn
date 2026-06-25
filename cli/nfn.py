from __future__ import annotations

import json
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
for candidate in (SCRIPTS_DIR, ROOT.parent):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def _arg_value(argv: list[str], *flags: str) -> str | None:
    for idx, arg in enumerate(argv):
        for flag in flags:
            if arg == flag and idx + 1 < len(argv):
                return argv[idx + 1]
            if arg.startswith(flag + "="):
                return arg.split("=", 1)[1]
    return None


def _has_any(argv: list[str], *flags: str) -> bool:
    return any(arg in flags or any(arg.startswith(flag + "=") for flag in flags) for arg in argv)


def _explicit_arg(argv: list[str], *flags: str) -> bool:
    return any(arg in flags or any(arg.startswith(flag + "=") for flag in flags) for arg in argv)


def _is_lightweight_root_help(argv: list[str]) -> bool:
    if not argv:
        return True
    idx = 0
    saw_help = False
    while idx < len(argv):
        arg = argv[idx]
        if arg in {"-h", "--help"}:
            saw_help = True
            idx += 1
            continue
        if arg == "--help-style":
            if idx + 1 >= len(argv) or argv[idx + 1] not in {"short", "long", "verbose"}:
                return False
            idx += 2
            continue
        if arg.startswith("--help-style="):
            if arg.split("=", 1)[1] not in {"short", "long", "verbose"}:
                return False
            idx += 1
            continue
        return False
    return saw_help or not argv


_DENSE_GPT_NATIVE_MODELS = {"gpt", "gpt2", "gpt3", "nanogpt", "nano-gpt"}
_NATIVE_TRAIN_FAMILY_TARGETS = {
    "gpt2-evo": "nfn_gpt2_evo_native_train",
    "llama": "nfn_llama_native_train",
    "mixllama": "nfn_mixllama_native_train",
    "jepa": "nfn_jepa_native_train",
    "semantic-router-moe": "nfn_semantic_router_moe_native_train",
    "deepseek-v4": "nfn_deepseek_v4_native_train",
    "nanogpt": "nfn_nanogpt_native_train",
    "nano-gpt": "nfn_nanogpt_native_train",
}


def _print_lightweight_root_help() -> None:
    print(
        """usage: nfn [-h] [--help-style {short,long,verbose}]

Master NeuralFn CLI for train, infer, and eval.

options:
  -h, --help            Show help for the master CLI. (default: False)
  --help-style {short,long,verbose}
                        Help detail level. (default: None)
"""
    )


def _lightweight_root_main(_argv: list[str] | None = None) -> int:
    _print_lightweight_root_help()
    return 0


_LIGHTWEIGHT_COMMAND_HELP: dict[str, str] = {
    "train": """\
        usage: nfn train [options]

        Train NeuralFn models.

        common options:
          -h, --help
          --help-style {short,long,verbose}
          --base-model, --model {gpt,gpt2,gpt3,nanogpt,llama}
          --topology {dense,moe,semantic_router}
          --router-mode {standard,semantic,hash}
          --dataset-alias NAME_OR_PATH
          --tinystories
          --pretraining-file PATH
          --runtime native-cuda
          --kernel-backend tile-cuda
          --template-name NAME, --template NAME, --preset NAME
          --graph-file PATH, --graph PATH
          --tile-cuda-strict, --no-tile-cuda-strict
          --eval-every-steps N
          --native-cuda-lm-head-row-chunk-size N
          --native-cuda-no-checkpoint, --no-checkpoint
          --native-cuda-runner {auto,binding,compiled-cli,launcher}
          --native-cuda-dry-run

        examples:
          nfn train --base-model gpt --tinystories
          nfn train --base-model gpt --tinystories --template-name gpt2_moa
          nfn train --base-model gpt3 --dataset-alias /data/tokens --graph-file graph.json
          nfn train --base-model gpt --tinystories --eval-every-steps 1000
          nfn train --base-model gpt --native-cuda-runner compiled-cli

        Explicit dense GPT runs dispatch before importing the graph-backed runtime.
        The compiled frontend records the selected template or custom graph and
        fails fast when a matching CUDA Tile C++ trainer is not implemented.
        """,
    "infer": """\
        usage: nfn infer [options]

        Run inference from NeuralFn artifacts.

        common options:
          -h, --help
          --help-style {short,long,verbose}
          --graph PATH
          --weights PATH
          --checkpoint PATH
          --native-checkpoint PATH
          --checkpoint-tokenizer PATH
          --native-info
          --native-sampler-script PATH (deprecated for native .bin prompts)
          --prompt TEXT
          --prompt-tokens IDS
          --max-new-tokens N
          --temperature FLOAT
          --top-k N
          --top-p FLOAT
          --kernel-backend tile-cuda
          --tile-cuda-strict, --no-tile-cuda-strict

        examples:
          nfn infer --graph ~/NeuralFn/artifacts/gpt2_evo.json --weights ~/NeuralFn/artifacts/gpt2_evo.pt --prompt "Once upon a time"
          nfn infer --checkpoint ~/NeuralFn/artifacts/gpt2 --native-info
          nfn infer --checkpoint ~/NeuralFn/artifacts/gpt2/model_00020000.bin --native-info
          nfn infer --checkpoint ~/NeuralFn/artifacts/gpt2/model_00020000.bin --prompt-tokens 50256
          nfn infer --checkpoint ~/NeuralFn/artifacts/final_model.pt --checkpoint-tokenizer tokenizer.model --prompt "Hello"
        """,
    "eval": """\
        usage: nfn eval [options]

        Evaluate NeuralFn artifacts.

        common options:
          -h, --help
          --help-style {short,long,verbose}
          --base-model, --model {gpt2,nanogpt,llama}
          --graph PATH
          --weights PATH
          --dataset-alias NAME_OR_PATH
          --eval-batches N
          --eval-batch-size N
          --prompt-suite {auto,general,shakespeare}
          --report-path PATH
          --kernel-backend tile-cuda
          --tile-cuda-strict, --no-tile-cuda-strict

        examples:
          nfn eval --graph ~/NeuralFn/artifacts/gpt2_evo.json --weights ~/NeuralFn/artifacts/gpt2_evo.pt
          nfn eval --base-model gpt2 --dataset-alias tinystories
        """,
    "kernels": """\
        usage: nfn kernels [list|doctor|bench|examples] [options]

        Inspect CUDA Tile kernel coverage and diagnostics.

        actions:
          list       Print metadata-only CUDA Tile registry coverage.
          doctor     Print CUDA Tile toolchain diagnostics plus coverage.
          bench      Compare graph-walk, compiled Torch, and Tile-requested execution.
          examples   List or regenerate CUDA Tile SDK examples.

        options:
          -h, --help
          --help-style {short,long,verbose}
          --json
          --kind {function,module,optimizer,runtime}
          --status {tile,torch_fallback,host_only,delegated,planned}
          --iterations N
          --warmup N
          --samples N
          --device auto|cpu|cuda|cuda:N
          --output-dir PATH
          --write

        examples:
          nfn kernels list --json
          nfn kernels doctor --json
          nfn kernels examples --write --output-dir examples/tile_cuda
        """,
}


def _is_lightweight_command_help(argv: list[str]) -> bool:
    if not argv or argv[0] not in _LIGHTWEIGHT_COMMAND_HELP:
        return False
    if "-h" not in argv and "--help" not in argv:
        return False
    idx = 1
    if argv[0] == "kernels" and idx < len(argv) and argv[idx] in {"list", "doctor", "bench", "examples"}:
        idx += 1
    while idx < len(argv):
        arg = argv[idx]
        if arg in {"-h", "--help"}:
            idx += 1
            continue
        if arg == "--help-style":
            if idx + 1 >= len(argv) or argv[idx + 1] not in {"short", "long", "verbose"}:
                return False
            idx += 2
            continue
        if arg.startswith("--help-style="):
            if arg.split("=", 1)[1] not in {"short", "long", "verbose"}:
                return False
            idx += 1
            continue
        return False
    return True


def _lightweight_command_help_main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    command = tokens[0] if tokens else ""
    help_text = _LIGHTWEIGHT_COMMAND_HELP.get(command)
    if help_text is None:
        return 2
    print(textwrap.dedent(help_text).strip())
    return 0


def _is_lightweight_kernels_list(argv: list[str]) -> bool:
    if not argv or argv[0] != "kernels":
        return False
    idx = 1
    action_seen = False
    while idx < len(argv):
        arg = argv[idx]
        if arg == "list" and not action_seen:
            action_seen = True
            idx += 1
            continue
        if arg in {"--json"}:
            idx += 1
            continue
        if arg in {"--kind", "--status"} and idx + 1 < len(argv):
            idx += 2
            continue
        return False
    return True


def _lightweight_kernels_list_main(argv: list[str] | None = None) -> int:
    import json

    from neuralfn.tile_cuda.registry import TRACKED_DTYPES
    from neuralfn.tile_cuda.registry import coverage_report

    tokens = list(sys.argv[1:] if argv is None else argv)
    json_output = "--json" in tokens
    allowed_kinds = {"function", "module", "optimizer", "runtime"}
    allowed_statuses = {"tile", "torch_fallback", "host_only", "delegated", "planned"}

    def option_value(flag: str) -> str | None:
        try:
            index = tokens.index(flag)
        except ValueError:
            return None
        return tokens[index + 1] if index + 1 < len(tokens) else None

    kind_filter = option_value("--kind")
    status_filter = option_value("--status")
    if kind_filter is not None and kind_filter not in allowed_kinds:
        raise SystemExit(f"invalid --kind: {kind_filter}")
    if status_filter is not None and status_filter not in allowed_statuses:
        raise SystemExit(f"invalid --status: {status_filter}")

    report = coverage_report()
    specs = [
        spec
        for spec in report.specs
        if (kind_filter is None or spec.kind == kind_filter)
        and (status_filter is None or spec.status == status_filter)
    ]
    if json_output:
        payload = report.to_dict()
        payload["filters"] = {
            "kind": kind_filter,
            "status": status_filter,
        }
        payload["tracked_dtypes"] = list(TRACKED_DTYPES)
        payload["unfiltered_spec_count"] = len(report.specs)
        payload["filtered_spec_count"] = len(specs)
        payload["specs"] = [spec.to_dict() for spec in specs]
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"NeuralFn CUDA Tile kernel coverage: {report.accounted}/{report.total_inventory} accounted")
        for status, count in sorted(report.by_status.items()):
            print(f"  {status}: {count}")
        if kind_filter is not None or status_filter is not None:
            print(
                "Filtered specs: "
                f"{len(specs)}/{len(report.specs)} "
                f"(kind={kind_filter or '*'}, status={status_filter or '*'})"
            )
            for spec in specs:
                print(f"  {spec.inventory_key}")
        if report.missing:
            print("Missing:")
            for name in report.missing:
                print(f"  {name}")
        else:
            print("Missing: none")
    return 0


def _is_explicit_native_gpt_train(argv: list[str]) -> bool:
    if not argv or argv[0] != "train":
        return False
    if _has_any(argv, "-h", "--help", "--plan", "--plan-auto", "--jepa"):
        return False
    base_model = (_arg_value(argv, "--base-model", "--model") or "gpt").strip().lower().replace("_", "-")
    if not _is_dense_gpt_native_model(base_model):
        return False
    topology = (_arg_value(argv, "--topology") or "dense").strip().lower()
    router_mode = (_arg_value(argv, "--router-mode") or "standard").strip().lower()
    return topology == "dense" and router_mode == "standard"


def _native_infer_checkpoint_arg(argv: list[str]) -> str | None:
    if not argv or argv[0] != "infer":
        return None
    if _has_any(argv, "-h", "--help", "--graph", "--plan", "--plan-auto"):
        return None
    return _arg_value(argv, "--native-checkpoint", "--checkpoint", "--weights")


def _resolve_native_infer_checkpoint(argv: list[str]) -> Path | None:
    raw_checkpoint = _native_infer_checkpoint_arg(argv)
    if not raw_checkpoint:
        return None
    checkpoint_path = Path(raw_checkpoint).expanduser()
    try:
        from neuralfn.native_gpt import is_native_gpt_checkpoint, latest_native_gpt_checkpoint

        if checkpoint_path.is_dir():
            return latest_native_gpt_checkpoint(checkpoint_path)
        if is_native_gpt_checkpoint(checkpoint_path):
            return checkpoint_path
    except Exception:
        return None
    return None


def _is_lightweight_native_gpt_infer(argv: list[str]) -> bool:
    return _resolve_native_infer_checkpoint(argv) is not None


def _lightweight_native_gpt_infer_main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    checkpoint_path = _resolve_native_infer_checkpoint(tokens)
    checkpoint = str(checkpoint_path) if checkpoint_path is not None else None
    if not checkpoint:
        return 2
    from neuralfn.native_gpt import read_native_gpt_checkpoint_info

    info = read_native_gpt_checkpoint_info(Path(checkpoint))
    print("Native GPT checkpoint detected")
    print(f"  path: {info.path}")
    print(f"  precision: {info.precision} (version {info.version})")
    print(f"  shape: layers={info.num_layers} heads={info.num_heads} channels={info.channels} seq_len={info.max_seq_len}")
    print(f"  vocab: vocab_size={info.vocab_size} padded_vocab_size={info.padded_vocab_size}")
    if info.step is not None:
        marker = "present" if info.done_marker_exists else "missing"
        print(f"  checkpoint_step: {info.step} (DONE marker {marker})")
    if _has_any(tokens, "--native-info"):
        return 0
    return _run_lightweight_native_gpt_sampler(tokens, checkpoint)


def _native_prompt_tokens(tokens: list[str]) -> str:
    from neuralfn.native_gpt import native_gpt_prompt_tokens

    return native_gpt_prompt_tokens(
        prompt=_arg_value(tokens, "--prompt") or "",
        prompt_tokens=_arg_value(tokens, "--prompt-tokens") or "",
        encoding_name=_arg_value(tokens, "--tokenizer") or ("gpt2" if _has_any(tokens, "--tokgpt2") else "gpt2"),
    )


def _run_lightweight_native_gpt_sampler(tokens: list[str], checkpoint: str) -> int:
    try:
        from neuralfn.native_gpt import run_native_gpt_checkpoint_sampler

        result = run_native_gpt_checkpoint_sampler(
            checkpoint,
            prompt_tokens=_native_prompt_tokens(tokens),
            max_new_tokens=int(_arg_value(tokens, "--max-new-tokens") or 64),
        )
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except FileNotFoundError:
        print(
            "Native GPT prompt-token inference needs the compiled nfn_gpt_native_train binary. "
            "Build it with tools/build_native_gpt_cli.sh or set NFN_NATIVE_GPT_CLI.",
            file=sys.stderr,
        )
        return 2
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    if result.returncode == 0:
        _render_lightweight_native_sampler_text(result.stdout)
    return int(result.returncode)


def _render_lightweight_native_sampler_text(stdout: str) -> None:
    from neuralfn.native_gpt import render_native_gpt_checkpoint_sampler_text

    rendered = render_native_gpt_checkpoint_sampler_text(stdout)
    if rendered:
        print(rendered)


def _native_gpt_argv(argv: list[str]) -> list[str]:
    forwarded: list[str] = []
    drop_value_flags = {
        "--base-model",
        "--model",
        "--topology",
        "--router-mode",
        "--model-preset",
        "--run-preset",
        "--optimizer-preset",
        "--tile-cuda-report",
        "--amp-dtype",
    }
    drop_bool_flags = {
        "--no-tile-cuda-strict",
        "--tile-cuda-strict",
    }
    idx = 1
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
        forwarded.append(arg)
        idx += 1
    return forwarded


def _native_gpt_requested_runner(argv: list[str]) -> str:
    return (_arg_value(argv, "--native-cuda-runner") or "compiled-cli").strip().lower().replace("_", "-")


def _native_gpt_requested_runtime(argv: list[str]) -> str:
    return (_arg_value(argv, "--runtime") or "native-cuda").strip().lower().replace("_", "-")


def _is_dense_gpt_native_model(model: str) -> bool:
    return model.strip().lower().replace("_", "-") in _DENSE_GPT_NATIVE_MODELS


def _is_direct_native_train_cli_train(argv: list[str]) -> bool:
    if not argv or argv[0] != "train":
        return False
    if _has_any(argv, "-h", "--help", "--plan", "--plan-auto", "--jepa"):
        return False
    if _native_gpt_requested_runtime(argv) != "native-cuda":
        return False
    base_model = (_arg_value(argv, "--base-model", "--model") or "gpt").strip().lower().replace("_", "-")
    if _is_dense_gpt_native_model(base_model):
        if not _is_explicit_native_gpt_train(argv):
            return False
        runner = _native_gpt_requested_runner(argv)
        return runner in {"auto", "compiled-cli"}
    return True


def _resolve_direct_native_train_cli(model: str) -> str:
    requested_train_cli = os.environ.get("NFN_NATIVE_TRAIN_CLI", "").strip()
    if requested_train_cli:
        return requested_train_cli
    if _is_dense_gpt_native_model(model):
        requested = os.environ.get("NFN_NATIVE_GPT_CLI", "").strip()
        if requested:
            return requested
        requested = os.environ.get("NFN_NATIVE_GPT2_CLI", "").strip()
        if requested:
            return requested
        linked = ROOT.parent / "build" / "nfn_gpt_native_train_linked"
        if linked.exists():
            return str(linked)
        return str(ROOT.parent / "build" / "nfn_gpt_native_train")
    family_cli = _resolve_direct_native_train_family_cli(model)
    if family_cli:
        return family_cli
    native_train = ROOT.parent / "build" / "nfn_native_train"
    if native_train.exists():
        return str(native_train)
    return str(native_train)


def _native_train_family_cli_env(model: str) -> str:
    suffix = "".join(ch if ch.isalnum() else "_" for ch in model.upper()).strip("_")
    return f"NFN_NATIVE_{suffix}_CLI"


def _resolve_direct_native_train_family_cli(model: str) -> str | None:
    if os.environ.get("NFN_NATIVE_TRAIN_CLI", "").strip():
        return None
    normalized = model.strip().lower().replace("_", "-")
    target = _NATIVE_TRAIN_FAMILY_TARGETS.get(normalized)
    if target is None:
        return None
    requested = os.environ.get(_native_train_family_cli_env(normalized), "").strip()
    if requested:
        return requested
    built = ROOT.parent / "build" / target
    if built.exists():
        return str(built)
    resolved = shutil.which(target)
    if resolved:
        return resolved
    return None


def _native_train_model(argv: list[str]) -> str:
    return (_arg_value(argv, "--base-model", "--model") or "gpt").strip().lower().replace("_", "-")


def _canonical_dense_gpt_model_family(model: str) -> str:
    return "gpt" if _is_dense_gpt_native_model(model) else model


def _native_gpt_cli_uses_linked_tile_ops(path: str) -> bool:
    return Path(path).name in {"nfn_gpt_native_train_linked", "nfn-gpt-native-train-linked"}


_NATIVE_TRAIN_ACTION_FLAGS = {
    "--check-tile-ops",
    "--json",
    "--list-templates",
    "--list-template-support",
    "--print-plan",
    "--native-cuda-check-tile-ops",
    "--native-cuda-list-templates",
    "--native-cuda-print-plan",
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
    "--native-cuda-smoke-attention-step",
    "--native-cuda-smoke-embedding-lm-step",
    "--native-cuda-smoke-lm-step",
    "--native-cuda-smoke-mlp-step",
    "--native-cuda-smoke-norm-residual-step",
    "--native-cuda-smoke-optimizer-step",
    "--native-cuda-smoke-tile-ops",
    "--native-cuda-smoke-transformer-block-step",
    "--native-cuda-smoke-transformer-lm-step",
    "--train-embedding-lm",
    "--train-transformer-lm",
    "--train-token-lm",
}


def _has_native_train_action(args: list[str]) -> bool:
    return any(arg in _NATIVE_TRAIN_ACTION_FLAGS for arg in args)


def _native_template_name(argv: list[str]) -> str:
    return (_arg_value(argv, "--template-name", "--template", "--preset") or "gpt").strip().lower().replace("-", "_")


def _has_native_activation(argv: list[str]) -> bool:
    return any(
        arg in {"--activation", "--native-cuda-activation"} or
        arg.startswith("--activation=") or
        arg.startswith("--native-cuda-activation=")
        for arg in argv
    )


def _direct_native_train_cli_argv(argv: list[str]) -> list[str]:
    model = _native_train_model(argv)
    token_lm_requested = any(arg == "--train-token-lm" for arg in argv)
    dense_gpt = _is_dense_gpt_native_model(model) and not (model in {"nanogpt", "nano-gpt"} and token_lm_requested)
    explicit_dense_model = dense_gpt and _explicit_arg(argv, "--base-model", "--model")
    family_cli = None if dense_gpt else _resolve_direct_native_train_family_cli(model)
    native_cli = family_cli or _resolve_direct_native_train_cli("gpt" if dense_gpt else model)
    out = [native_cli]
    tile_ops_lib_explicit = _explicit_arg(argv, "--tile-ops-lib", "--native-cuda-tile-ops-lib")
    include_model = not dense_gpt and family_cli is None
    if include_model:
        out.extend(["--base-model", model])
    elif dense_gpt:
        out.extend(["--model-family", _canonical_dense_gpt_model_family(model)])
        if model == "nanogpt" and not _explicit_arg(argv, "--template-name", "--template", "--preset", "--graph-file", "--graph"):
            out.extend(["--template-name", "nanogpt"])
    if dense_gpt and not _has_native_train_action(argv):
        out.append("--train-transformer-lm")
    idx = 1
    drop_value_flags = {
        "--base-model",
        "--model",
        "--topology",
        "--router-mode",
        "--model-preset",
        "--run-preset",
        "--optimizer-preset",
        "--tile-cuda-report",
        "--amp-dtype",
        "--runtime",
        "--device",
        "--dataset-hf-path",
        "--dataset-variant",
        "--dataset-train-shards",
        "--dataset-train-file",
        "--dataset-val-file",
        "--tokenizer",
        "--native-cuda-runner",
    }
    drop_bool_flags = {
        "--no-tile-cuda-strict",
        "--tile-cuda-strict",
        "--download-if-missing",
        "--no-download-if-missing",
        "--tokgpt2",
        "--cl100k",
        "--o200k",
    }
    value_aliases = {
        "--kernel-backend": "--backend",
        "--native-cuda-executable": "--target",
        "--native-cuda-output-dir": "--output-dir",
        "--native-cuda-tile-ops-lib": "--tile-ops-lib",
        "--native-cuda-cuda-runtime-lib": "--cuda-runtime-lib",
        "--native-cuda-lm-head-row-chunk-size": "--lm-head-row-chunk-size",
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
    }
    split_value_flags = {
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
        "--native-cuda-checkpoint-every",
        "--native-cuda-sample-every",
        "--native-cuda-generate-tokens",
        "--cuda-runtime-lib",
        "--activation",
        "--native-cuda-activation",
        "--moa-interval",
        "--native-cuda-moa-interval",
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
        if arg == "--dataset":
            if idx + 1 >= len(argv):
                out.append(arg)
                idx += 1
                continue
            dataset = argv[idx + 1].strip().lower()
            if dataset == "tinystories":
                out.append("--tinystories")
            elif dataset in {"golf1", "golf10"}:
                shard_count = "1" if dataset == "golf1" else "10"
                _append_value_arg(out, "--dataset-alias", f"willdepueoai__parameter-golf__sp1024__train{shard_count}")
            else:
                _append_value_arg(out, "--dataset", argv[idx + 1])
            idx += 2
            continue
        if arg.startswith("--dataset="):
            dataset = arg.split("=", 1)[1].strip().lower()
            if dataset == "tinystories":
                out.append("--tinystories")
            elif dataset in {"golf1", "golf10"}:
                shard_count = "1" if dataset == "golf1" else "10"
                _append_value_arg(out, "--dataset-alias", f"willdepueoai__parameter-golf__sp1024__train{shard_count}")
            else:
                out.append(arg)
            idx += 1
            continue
        if arg == "--output":
            if idx + 1 < len(argv):
                _append_value_arg(out, "--output-dir", _native_output_dir_from_output(argv[idx + 1]))
            else:
                out.append(arg)
            idx += 2
            continue
        if arg.startswith("--output="):
            out.extend(["--output-dir", _native_output_dir_from_output(arg.split("=", 1)[1])])
            idx += 1
            continue
        if arg in value_aliases:
            if idx + 1 < len(argv):
                _append_value_arg(out, value_aliases[arg], argv[idx + 1])
            else:
                out.append(value_aliases[arg])
            idx += 2
            continue
        matched_value_alias = next((flag for flag in value_aliases if arg.startswith(flag + "=")), None)
        if matched_value_alias is not None:
            _append_value_arg(out, value_aliases[matched_value_alias], arg.split("=", 1)[1])
            idx += 1
            continue
        if arg in bool_aliases:
            out.append(bool_aliases[arg])
            idx += 1
            continue
        matched_split_flag = next((flag for flag in split_value_flags if arg.startswith(flag + "=")), None)
        if matched_split_flag is not None:
            _append_value_arg(out, matched_split_flag, arg.split("=", 1)[1])
            idx += 1
            continue
        out.append(arg)
        idx += 1
    if dense_gpt and _native_template_name(out) == "gpt2_moa" and not _has_native_activation(out):
        _append_value_arg(out, "--native-cuda-activation", "moa")
    if (
        dense_gpt
        and model == "gpt3"
        and not _explicit_arg(out, "--train-seq-len")
        and not _explicit_arg(out, "--template-name", "--template", "--preset")
        and not _explicit_arg(out, "--graph-file", "--graph")
    ):
        _append_value_arg(out, "--train-seq-len", "2048")
    if dense_gpt and not _explicit_arg(out, "--backend"):
        _append_value_arg(out, "--backend", "tile-cuda")
    if dense_gpt and _native_gpt_cli_uses_linked_tile_ops(native_cli) and not tile_ops_lib_explicit:
        _append_value_arg(out, "--tile-ops-lib", "linked")
    return out


def _direct_native_train_cli_main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    command = _direct_native_train_cli_argv(tokens)
    model = _native_train_model(tokens)
    token_lm_requested = any(arg == "--train-token-lm" for arg in tokens)
    direct_family_cli = (
        not (_is_dense_gpt_native_model(model) and not (model in {"nanogpt", "nano-gpt"} and token_lm_requested))
        and _resolve_direct_native_train_family_cli(model) is not None
    )
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    native_execution_flags = {
        "--print-plan",
        "--list-templates",
        "--check-tile-ops",
        "--startup-only",
        "--smoke-tile-ops",
        "--smoke-optimizer-step",
        "--smoke-lm-step",
        "--smoke-attention-step",
        "--smoke-mlp-step",
        "--smoke-norm-residual-step",
        "--smoke-transformer-block-step",
        "--smoke-transformer-lm-step",
        "--smoke-embedding-lm-step",
    }
    if (
        "--dry-run" in command
        and "--print-command" in command
        and not direct_family_cli
        and not any(flag in command for flag in native_execution_flags)
    ):
        print(shlex.join(command))
        return 0
    if "--dry-run" in command or "--print-command" in command:
        proc = subprocess.run(command, env=env, check=False)
        return int(proc.returncode)
    os.execvpe(command[0], command, env)
    return 127


def _append_value_arg(out: list[str], flag: str, value: str) -> None:
    out.extend([flag, value])


def _native_output_dir_from_output(value: str) -> str:
    path = Path(value).expanduser()
    if path.suffix:
        path = path.with_suffix("")
    return str(path)


def _is_legacy_graph_train(argv: list[str]) -> bool:
    if not argv or argv[0] != "train":
        return False
    if _has_any(argv, "-h", "--help", "--plan", "--plan-auto"):
        return False
    if _is_explicit_native_gpt_train(argv):
        return False
    return True


def _legacy_graph_train_main(_argv: list[str] | None = None) -> int:
    print(
        "This training command would enter the graph-backed TorchTrainer path, which is disabled by default.\n"
        "Default NeuralFn training must use compiled native CUDA/C++ entrypoints. Today the default compiled "
        "training route is dense GPT: nfn train --base-model gpt --tinystories.\n"
        "Build a matching native trainer for this model family before running it. Legacy graph-backed "
        "experiments must call the Python SDK trainer APIs directly instead of routing through nfn train.",
        file=sys.stderr,
    )
    return 2


def _load_full_impl():
    import nfn_impl

    return nfn_impl


def __getattr__(name: str):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(f"module 'nfn' has no attribute {name!r}")
    impl = _load_full_impl()
    try:
        return getattr(impl, name)
    except AttributeError as exc:
        raise AttributeError(f"module 'nfn' has no attribute {name!r}") from exc


def main(
    argv: list[str] | None = None,
    *,
    stdin_isatty: bool | None = None,
    stdout_isatty: bool | None = None,
) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if _is_direct_native_train_cli_train(tokens):
        return _direct_native_train_cli_main(tokens)
    if stdin_isatty is None and stdout_isatty is None:
        if _is_explicit_native_gpt_train(tokens):
            from train_gpt_native import main as train_gpt_native_main

            return int(train_gpt_native_main(_native_gpt_argv(tokens)))
        if _is_lightweight_root_help(tokens):
            return _lightweight_root_main(tokens)
        if _is_lightweight_command_help(tokens):
            return _lightweight_command_help_main(tokens)
        if _is_lightweight_kernels_list(tokens):
            return _lightweight_kernels_list_main(tokens)
        if _is_lightweight_native_gpt_infer(tokens):
            return _lightweight_native_gpt_infer_main(tokens)
        if _is_legacy_graph_train(tokens):
            return _legacy_graph_train_main(tokens)
    impl = _load_full_impl()
    kwargs: dict[str, bool] = {}
    if stdin_isatty is not None:
        kwargs["stdin_isatty"] = stdin_isatty
    if stdout_isatty is not None:
        kwargs["stdout_isatty"] = stdout_isatty
    return int(impl.main(tokens, **kwargs))


if __name__ == "__main__":
    if _is_direct_native_train_cli_train(sys.argv[1:]):
        main = _direct_native_train_cli_main
    elif _is_explicit_native_gpt_train(sys.argv[1:]):
        from train_gpt_native import main as main
    elif _is_lightweight_native_gpt_infer(sys.argv[1:]):
        main = _lightweight_native_gpt_infer_main
    elif _is_lightweight_root_help(sys.argv[1:]):
        main = _lightweight_root_main
    elif _is_lightweight_command_help(sys.argv[1:]):
        main = _lightweight_command_help_main
    elif _is_lightweight_kernels_list(sys.argv[1:]):
        main = _lightweight_kernels_list_main
    elif _is_legacy_graph_train(sys.argv[1:]):
        main = _legacy_graph_train_main
    else:
        from nfn_impl import *  # noqa: F401,F403
        from nfn_impl import main


if __name__ == "__main__":
    if _is_direct_native_train_cli_train(sys.argv[1:]):
        raise SystemExit(main(sys.argv[1:]))
    if _is_explicit_native_gpt_train(sys.argv[1:]):
        raise SystemExit(main(_native_gpt_argv(sys.argv[1:])))
    if _is_lightweight_native_gpt_infer(sys.argv[1:]):
        raise SystemExit(main(sys.argv[1:]))
    if _is_legacy_graph_train(sys.argv[1:]):
        raise SystemExit(main(sys.argv[1:]))
    raise SystemExit(main())
