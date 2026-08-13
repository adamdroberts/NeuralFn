from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
import re
import selectors
import shlex
import shutil
import subprocess
import sys
import textwrap
import time


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
_AUTO_CUDA_VISIBLE_DEVICE_VALUES = {"auto", "dedicated", "dedicated-auto"}
_DEFAULT_NATIVE_GPT_EVAL_BATCHES = "20"
_NATIVE_GPT_QUALITY_DEFAULTS = {
    "--eval-every-steps": (
        "NFN_NATIVE_GPT_EVAL_EVERY_STEPS",
        "NFN_SM120_NATIVE_EVAL_EVERY_STEPS",
        "NFN_SM120_EVAL_EVERY_STEPS",
        "5000",
    ),
    "--eval-batches": (
        "NFN_NATIVE_GPT_EVAL_BATCHES",
        "NFN_SM120_NATIVE_EVAL_BATCHES",
        "NFN_SM120_EVAL_BATCHES",
        _DEFAULT_NATIVE_GPT_EVAL_BATCHES,
    ),
    "--train-loss-every-steps": (
        "NFN_NATIVE_GPT_TRAIN_LOSS_EVERY_STEPS",
        "NFN_SM120_NATIVE_TRAIN_LOSS_EVERY_STEPS",
        "NFN_SM120_TRAIN_LOSS_EVERY_STEPS",
        "250",
    ),
    "--native-cuda-sample-every": (
        "NFN_NATIVE_GPT_SAMPLE_EVERY",
        "NFN_SM120_NATIVE_SAMPLE_EVERY",
        "NFN_SM120_SAMPLE_EVERY",
        "20000",
    ),
    "--native-cuda-generate-tokens": (
        "NFN_NATIVE_GPT_GENERATE_TOKENS",
        "NFN_SM120_NATIVE_GENERATE_TOKENS",
        "NFN_SM120_GENERATE_TOKENS",
        "144",
    ),
    "--native-cuda-checkpoint-every": (
        "NFN_NATIVE_GPT_CHECKPOINT_EVERY",
        "NFN_SM120_NATIVE_CHECKPOINT_EVERY",
        "NFN_SM120_CHECKPOINT_EVERY",
        "5000",
    ),
    "--batch-size": (
        "NFN_NATIVE_GPT_BATCH_SIZE",
        "NFN_SM120_NATIVE_BATCH_SIZE",
        "NFN_SM120_BATCH_SIZE",
        "64",
    ),
    "--train-seq-len": (
        "NFN_NATIVE_GPT_TRAIN_SEQ_LEN",
        "NFN_SM120_NATIVE_TRAIN_SEQ_LEN",
        "NFN_SM120_TRAIN_SEQ_LEN",
        "1024",
    ),
    "--train-batch-tokens": (
        "NFN_NATIVE_GPT_TRAIN_BATCH_TOKENS",
        "NFN_SM120_NATIVE_TRAIN_BATCH_TOKENS",
        "NFN_SM120_TRAIN_BATCH_TOKENS",
        "524288",
    ),
    "--learning-rate": (
        "NFN_NATIVE_GPT_LEARNING_RATE",
        "NFN_SM120_NATIVE_LEARNING_RATE",
        "NFN_SM120_LEARNING_RATE",
        "0.0006",
    ),
    "--lr-schedule": (
        "NFN_NATIVE_GPT_LR_SCHEDULE",
        "NFN_SM120_NATIVE_LR_SCHEDULE",
        "NFN_SM120_LR_SCHEDULE",
        "cosine",
    ),
    "--final-lr-fraction": (
        "NFN_NATIVE_GPT_FINAL_LR_FRACTION",
        "NFN_SM120_NATIVE_FINAL_LR_FRACTION",
        "NFN_SM120_FINAL_LR_FRACTION",
        "0.0",
    ),
    "--weight-decay": (
        "NFN_NATIVE_GPT_WEIGHT_DECAY",
        "NFN_SM120_NATIVE_WEIGHT_DECAY",
        "NFN_SM120_WEIGHT_DECAY",
        "0.1",
    ),
    "--beta1": ("NFN_NATIVE_GPT_BETA1", "NFN_SM120_NATIVE_BETA1", "NFN_SM120_BETA1", "0.9"),
    "--beta2": ("NFN_NATIVE_GPT_BETA2", "NFN_SM120_NATIVE_BETA2", "NFN_SM120_BETA2", "0.95"),
    "--adam-eps": (
        "NFN_NATIVE_GPT_ADAM_EPS",
        "NFN_SM120_NATIVE_ADAM_EPS",
        "NFN_SM120_ADAM_EPS",
        "1e-8",
    ),
    "--grad-clip-norm": (
        "NFN_NATIVE_GPT_GRAD_CLIP_NORM",
        "NFN_SM120_NATIVE_GRAD_CLIP_NORM",
        "NFN_SM120_GRAD_CLIP_NORM",
        "1.0",
    ),
    "--warmup-steps": (
        "NFN_NATIVE_GPT_WARMUP_STEPS",
        "NFN_SM120_NATIVE_WARMUP_STEPS",
        "NFN_SM120_WARMUP_STEPS",
        "60",
    ),
    "--max-steps": (
        "NFN_NATIVE_GPT_MAX_STEPS",
        "NFN_SM120_NATIVE_MAX_STEPS",
        "NFN_SM120_MAX_STEPS",
        "20000",
    ),
}
_NATIVE_GPT_METADATA_ACTION_FLAGS = {
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
_NATIVE_TRAIN_FAMILY_TARGETS = {
    "embedding": "nfn_embedding_native_train",
    "gpt2-evo": "nfn_gpt2_evo_native_train",
    "muse-glimmer": "nfn_muse_glimmer_native_train",
    "llama": "nfn_llama_native_train",
    "mixllama": "nfn_mixllama_native_train",
    "jepa": "nfn_jepa_native_train",
    "semantic-dense-jepa": "nfn_semantic_dense_jepa_native_train",
    "moe-jepa-evo": "nfn_moe_jepa_evo_native_train",
    "semantic-router-moe": "nfn_semantic_router_moe_native_train",
    "semantic-moe-jepa-evo": "nfn_semantic_router_moe_native_train",
    "semantic-moe-jepa-evo-modern": "nfn_semantic_router_moe_native_train",
    "diff-semantic-moe-jepa-evo": "nfn_semantic_router_moe_native_train",
    "deepseek-v4": "nfn_deepseek_v4_native_train",
    "jamba": "nfn_jamba_native_train",
    "seq2seq": "nfn_seq2seq_native_train",
    "diffusion": "nfn_diffusion_native_train",
    "ttt-llama": "nfn_ttt_llama_native_train",
    "hnet-lm": "nfn_hnet_lm_native_train",
    "universal-llama": "nfn_universal_llama_native_train",
    "nanogpt": "nfn_nanogpt_native_train",
    "nano-gpt": "nfn_nanogpt_native_train",
}
_NATIVE_TEMPLATE_FAMILY_ALIASES = {
    "muse-glimmer": "muse-glimmer",
    "llama": "llama",
    "llama-fast": "llama",
    "llama-fast-megakernel": "llama",
    "llama-megakernel": "llama",
    "llama-modern": "llama",
    "modern-norms-llama": "llama",
    "ternary-b158": "llama",
    "ternary-b158-modern": "llama",
    "fp8-llama": "llama",
    "mxfp4-llama": "llama",
    "gemma3": "llama",
    "diff-transformer": "llama",
    "longctx-sparse-llama": "llama",
    "qwen3-longctx": "llama",
    "kv-pca-llama": "llama",
    "kv-pca-llama-modern": "llama",
    "mixllama": "mixllama",
    "mixllama-fast": "mixllama",
    "mixllama-fast-megakernel": "mixllama",
    "moe": "mixllama",
    "moe-modern": "mixllama",
    "deepseek-v3": "mixllama",
    "deepseek-v4": "deepseek-v4",
    "llm-jepa": "jepa",
    "llm-jepa-modern": "jepa",
    "dense-jepa-evo": "jepa",
    "dense-jepa-evo-modern": "jepa",
    "semantic-dense-jepa-evo": "semantic-dense-jepa",
    "semantic-dense-jepa-evo-modern": "semantic-dense-jepa",
    "dyt-geglu-semantic-dense-jepa-evo": "semantic-dense-jepa",
    "jepa-semantic-hybrid": "semantic-dense-jepa",
    "jepa-semantic-hybrid-modern": "semantic-dense-jepa",
    "jepa-semantic-hybrid-megakernel": "semantic-dense-jepa",
    "moe-jepa-evo": "moe-jepa-evo",
    "moe-jepa-evo-modern": "moe-jepa-evo",
    "auxfree-moe-jepa-evo": "moe-jepa-evo",
    "semantic-router-moe": "semantic-router-moe",
    "semantic-router-moe-modern": "semantic-router-moe",
    "semantic-router-moe-megakernel": "semantic-router-moe",
    "semantic-moe-jepa-evo": "semantic-router-moe",
    "semantic-moe-jepa-evo-modern": "semantic-router-moe",
    "diff-semantic-moe-jepa-evo": "semantic-router-moe",
    "jamba": "jamba",
    "jamba-modern": "jamba",
    "seq2seq": "seq2seq",
    "seq2seq-modern": "seq2seq",
    "diffusion": "diffusion",
    "diffusion-modern": "diffusion",
    "ttt-llama": "ttt-llama",
    "ttt-llama-modern": "ttt-llama",
    "hnet-lm": "hnet-lm",
    "hnet-lm-modern": "hnet-lm",
    "universal-llama": "universal-llama",
    "universal-llama-modern": "universal-llama",
}
_NATIVE_FAMILY_CHECKPOINT_TEMPLATE_TARGETS = {
    name: name
    for name in _NATIVE_TRAIN_FAMILY_TARGETS
    if name not in {"gpt2-evo", "nanogpt", "nano-gpt"}
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


def _print_lightweight_root_help() -> None:
    print(
        """usage: nfn [-h] [--help-style {short,long,verbose}]

Master NeuralFn CLI for train, embed, infer, and eval.

options:
  -h, --help            Show help for the master CLI. (default: False)
  --help-style {short,long,verbose}
                        Help detail level. (default: None)

commands:
  train                 Train NeuralFn models.
  embed                 Produce native text embeddings.
  infer                 Run inference from NeuralFn artifacts.
  eval                  Evaluate NeuralFn artifacts.
  kernels               Inspect CUDA Tile kernel coverage.
  migrate graph-to-native
                        Lower a graph and optional .pt weights to Native Execution IR.
  migrate muse-glimmer-to-native
                        Strictly convert a pinned Muse Glimmer BF16 checkpoint bundle.
  migrate muse-glimmer-gguf-to-native
                        Authenticate and bundle official Glimmer K-Quant variants.
  migrate muse-glimmer-lora-to-native
                        Attach a strict native LoRA/QLoRA checkpoint to a Glimmer bundle.
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
          --tui, --interactive
          --no-tui
          --help-style {short,long,verbose}
          --base-model, --model {gpt,gpt2,gpt3,nanogpt,llama,muse-glimmer,embedding}
          --topology {dense,moe,semantic_router}
          --router-mode {standard,semantic,hash}
          --dataset-alias NAME_OR_PATH
          --tinystories
          --pretraining-file PATH
          --runtime native-cuda
          --kernel-backend tile-cuda
          --template-name NAME, --template NAME, --preset NAME
          --graph-file PATH, --graph PATH
          --graph-fingerprint SHA256
          --graph-preflight-proof PATH
          --tile-cuda-strict, --no-tile-cuda-strict
          --lr-schedule {cosine,constant}
          --lr-schedule-total-steps N
          --train-seed N
          --resume-from-checkpoint PATH
          --eval-every-steps N
          --train-log-file PATH
          --eval-log-file PATH
          --native-cuda-lm-head-row-chunk-size N
          --native-cuda-no-checkpoint, --no-checkpoint
          --native-cuda-fast-startup, --fast-startup
          --native-cuda-runner {auto,binding,compiled-cli,launcher}
          --native-cuda-dry-run
          --embedding-datasets-manifest PATH
          --embedding-dataset PATH (repeatable)
          --embedding-stage {pretrain,posttrain,finetune,resume}
          --embedding-architecture {bert,gpt-derived}
          --embedding-hf-model MODEL_ID_OR_PATH
          --embedding-hf-revision REVISION

        examples:
          nfn train
          nfn train --tui
          nfn train --base-model gpt --tinystories
          nfn train --base-model gpt --tinystories --template-name gpt2_moa
          nfn train --base-model gpt3 --dataset-alias /data/tokens --graph-file graph.json
          nfn train --base-model gpt --tinystories --eval-every-steps 1000
          nfn train --base-model gpt --native-cuda-runner compiled-cli
          nfn train --base-model muse-glimmer --checkpoint MODEL --checkpoint-sha256 SHA256 --dataset DATASET --objective sft --chat-template-sha256 SHA256
          nfn train --base-model embedding --embedding-dataset corpus.txt

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
          --verify
          --verify-all
          --required-templates TEMPLATE[,TEMPLATE...]
          --require-covered-templates
          --require-architecture-forward
          --native-sampler-script PATH (deprecated for native .bin prompts)
          --runtime {auto,cpu,native-cuda,graph}
          --weight-precision {auto,bf16,k-quant-dynamic,k-quant-17gb}
          --speculative-decoding, --speculative {off,auto,required}
          --companion-checkpoint {dflash,mmproj,lora} (repeatable)
          --prompt TEXT
          --prompt-tokens IDS
          --chat-mode {transcript,stateless}
          --system-prompt TEXT
          --chat-template {auto,plain_roles,PATH}
          --serve (native artifact directories only)
          --host HOST, --port PORT
          --served-model-name NAME
          --queue-capacity N
          --session-limit N
          --max-output-tokens N
          --prefix-cache-capacity N
          --kv-cache {off,auto,full,turboquant}
          --turboquant-profile {mse-3.5,qjl-3.5}
          --turboquant-attention-backend {cpu,tile-cuda}
          --tile-ops-lib PATH (required for the explicit tile-cuda attention backend)
          --cuda-runtime-lib PATH_OR_SONAME
          --cuda-device INDEX
          --api-key-file PATH
          --state-db PATH
          --allow-unauthenticated-remote
          --max-new-tokens N
          --temperature FLOAT (finite >=0; exact zero enables strict deterministic CUDA inference)
          --top-k N
          --top-p FLOAT
          --strict-tile-ops-lib PATH
          --kernel-backend tile-cuda
          --tile-cuda-strict, --no-tile-cuda-strict

        examples:
          nfn infer --graph ~/NeuralFn/artifacts/gpt2_evo.json --weights ~/NeuralFn/artifacts/gpt2_evo.pt --prompt "Once upon a time"
          nfn infer --checkpoint ~/NeuralFn/artifacts/gpt2 --native-info
          nfn infer --runtime native-cuda --checkpoint ~/NeuralFn/artifacts/gpt2/model_00020000.bin --prompt-tokens 50256
          nfn infer --checkpoint ~/NeuralFn/artifacts/gpt2/model_00020000.bin --native-info
          nfn infer --checkpoint ~/NeuralFn/artifacts/gpt2/model_00020000.bin --prompt-tokens 50256
          nfn infer --checkpoint artifacts/glimmer-kquant --runtime native-cuda --weight-precision auto --companion-checkpoint dflash --speculative-decoding auto --prompt "Hello"
          nfn infer --checkpoint ~/NeuralFn/artifacts/final_model.pt --checkpoint-tokenizer tokenizer.model --prompt "Hello"
          nfn infer --checkpoint ~/NeuralFn/artifacts/gpt2-native --serve

        Interactive graph inference defaults to transcript mode. Use
        --chat-mode stateless (or /mode stateless) for independent turns.
        """,
    "embed": """\
        usage: nfn embed --checkpoint PATH (--text TEXT | --input PATH)

        Produce normalized text vectors with a native NeuralFn embedding checkpoint.

        options:
          -h, --help
          --checkpoint PATH
          --text TEXT
          --input PATH        One text per line; emits one JSON object per line.

        example:
          nfn embed --checkpoint artifacts/embedding/embedding_model.bin --text "hello world"
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
    "migrate": """\
        usage: nfn migrate graph-to-native --graph GRAPH [--weights WEIGHTS] --output-dir DIR [--dry-run]
               nfn migrate muse-glimmer-to-native --source DIR --output-dir DIR [--component {text,vision,full,assistant}]
               nfn migrate muse-glimmer-gguf-to-native --gguf FILE [--gguf FILE] --tokenizer-source DIR --output-dir DIR
               nfn migrate muse-glimmer-lora-to-native --artifact DIR --checkpoint DIR

        Validate and lower a graph to the versioned Native Execution IR artifact.
        Graph-only migration remains Torch-free; supplying legacy .pt weights
        invokes the isolated checkpoint conversion worker.

        options:
          -h, --help
          --graph GRAPH       Source NeuralFn graph JSON.
          --weights WEIGHTS   Optional legacy .pt checkpoint.
          --output-dir DIR    New artifact directory; existing paths are never overwritten.
          --dry-run           Validate and print the manifest/report without writing files.

        examples:
          nfn migrate graph-to-native --graph graph.json --output-dir artifacts/native-model
          nfn migrate graph-to-native --graph graph.json --weights model.pt --output-dir artifacts/native-model
          nfn migrate graph-to-native --graph graph.json --output-dir artifacts/native-model --dry-run
          nfn migrate muse-glimmer-to-native --source Muse-Glimmer-30B --output-dir artifacts/glimmer-text
          nfn migrate muse-glimmer-to-native --source Muse-Glimmer-30B-assistant --component assistant --target-checkpoint-sha256 SHA256 --output-dir artifacts/glimmer-dflash
          nfn migrate muse-glimmer-gguf-to-native --gguf Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf --gguf Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf --tokenizer-source Muse-Glimmer-30B --output-dir artifacts/glimmer-kquant
          nfn migrate muse-glimmer-lora-to-native --artifact artifacts/glimmer-bf16 --checkpoint runs/checkpoint-step-100
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
    if argv[0] == "migrate" and idx < len(argv) and argv[idx] in {
        "graph-to-native",
        "muse-glimmer-to-native",
        "muse-glimmer-gguf-to-native",
        "muse-glimmer-lora-to-native",
    }:
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


def _is_native_embedding_infer(argv: list[str]) -> bool:
    return bool(argv and argv[0] == "embed" and not _has_any(argv, "-h", "--help"))


def _native_embedding_infer_main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    checkpoint = _arg_value(tokens, "--checkpoint")
    text_value = _arg_value(tokens, "--text")
    input_path = _arg_value(tokens, "--input")
    if not checkpoint or (text_value is None and input_path is None) or (text_value is not None and input_path is not None):
        print("nfn embed requires --checkpoint and exactly one of --text or --input", file=sys.stderr)
        return 2
    from neuralfn.native_embedding import read_embedding_checkpoint_header, resolve_native_embedding_cli, tokenize_huggingface_text

    executable = resolve_native_embedding_cli(ROOT.parent)
    texts: list[str]
    if input_path is not None:
        texts = [line for line in Path(input_path).expanduser().read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        texts = [str(text_value)]
    checkpoint_path = Path(checkpoint).expanduser()
    checkpoint_file = checkpoint_path / "embedding_model.bin" if checkpoint_path.is_dir() else checkpoint_path
    tokenizer_dir = next(
        (
            candidate
            for candidate in (checkpoint_file.parent, checkpoint_file.parent / "hf_import")
            if (candidate / "tokenizer.json").is_file() or (candidate / "vocab.txt").is_file()
        ),
        None,
    )
    max_tokens = read_embedding_checkpoint_header(checkpoint_file)["max_tokens"]
    for item in texts:
        inference_args = [executable, "--checkpoint", checkpoint]
        if tokenizer_dir is not None:
            token_ids = tokenize_huggingface_text(item, tokenizer_dir, max_tokens=max_tokens)
            inference_args.extend(["--embed-token-ids", ",".join(str(token) for token in token_ids)])
        else:
            inference_args.extend(["--embed-text", item])
        try:
            proc = subprocess.run(
                inference_args,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            print(
                f"Unable to launch native embedding inference ({exc}). "
                "Build it with `bash tools/build_native_embedding_cli.sh` or set NFN_NATIVE_EMBEDDING_CLI.",
                file=sys.stderr,
            )
            return 2
        if proc.returncode != 0:
            if proc.stderr:
                sys.stderr.write(proc.stderr)
            return int(proc.returncode)
        sys.stdout.write(proc.stdout)
    return 0


def _lightweight_command_help_main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    command = tokens[0] if tokens else ""
    help_text = _LIGHTWEIGHT_COMMAND_HELP.get(command)
    if help_text is None:
        return 2
    print(textwrap.dedent(help_text).strip())
    return 0


def _is_lightweight_graph_migrate(argv: list[str]) -> bool:
    return len(argv) >= 2 and argv[:2] == ["migrate", "graph-to-native"]


def _is_lightweight_muse_glimmer_migrate(argv: list[str]) -> bool:
    return len(argv) >= 2 and argv[:2] == ["migrate", "muse-glimmer-to-native"]


def _is_lightweight_muse_glimmer_gguf_migrate(argv: list[str]) -> bool:
    return len(argv) >= 2 and argv[:2] == ["migrate", "muse-glimmer-gguf-to-native"]


def _is_lightweight_muse_glimmer_lora_migrate(argv: list[str]) -> bool:
    return len(argv) >= 2 and argv[:2] == ["migrate", "muse-glimmer-lora-to-native"]


def _lightweight_graph_migrate_main(argv: list[str] | None = None) -> int:
    import argparse

    tokens = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="nfn migrate graph-to-native",
        description="Lower a NeuralFn graph and optional legacy .pt weights to Native Execution IR.",
    )
    parser.add_argument("--graph", required=True, metavar="GRAPH")
    parser.add_argument("--weights", metavar="WEIGHTS")
    parser.add_argument("--output-dir", required=True, metavar="DIR")
    parser.add_argument("--dry-run", action="store_true")
    try:
        args = parser.parse_args(tokens[2:])
    except SystemExit as exc:
        return int(exc.code)

    from neuralfn.native_ir import migrate_graph_to_native

    try:
        result = migrate_graph_to_native(
            args.graph,
            output_dir=args.output_dir,
            weights_path=args.weights,
            dry_run=bool(args.dry_run),
        )
    except (FileExistsError, FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.report.compatible else 2


def _lightweight_muse_glimmer_migrate_main(argv: list[str] | None = None) -> int:
    import argparse
    from dataclasses import asdict

    tokens = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="nfn migrate muse-glimmer-to-native",
        description=(
            "Authenticate and stream-convert the pinned official Muse Glimmer BF16 "
            "main or DFlash assistant checkpoint."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--source", required=True, metavar="DIR")
    parser.add_argument("--output-dir", metavar="DIR")
    parser.add_argument(
        "--component",
        choices=("text", "vision", "full", "assistant"),
        default="text",
    )
    parser.add_argument("--target-checkpoint-sha256", metavar="SHA256")
    parser.add_argument("--inspect-only", action="store_true")
    try:
        args = parser.parse_args(tokens[2:])
    except SystemExit as exc:
        return int(exc.code)
    if not args.inspect_only and not args.output_dir:
        print("--output-dir is required unless --inspect-only is selected", file=sys.stderr)
        return 2
    if args.component == "assistant" and not args.inspect_only and not args.target_checkpoint_sha256:
        print("--component assistant requires --target-checkpoint-sha256", file=sys.stderr)
        return 2

    from neuralfn.native_muse_glimmer_checkpoint import (
        MuseGlimmerCheckpointError,
        convert_official_muse_glimmer_assistant_safetensors,
        convert_official_muse_glimmer_safetensors,
        inspect_official_muse_glimmer_safetensors,
    )

    try:
        if args.inspect_only:
            bundle = inspect_official_muse_glimmer_safetensors(
                args.source,
                assistant=args.component == "assistant",
            )
            payload = {
                "source": str(bundle.root),
                "component": args.component,
                "tensor_count": len(bundle.entries),
                "parameter_count": bundle.parameter_count,
                "payload_bytes": bundle.payload_bytes,
                "shards": {
                    name: {
                        "nbytes": bundle.shard_nbytes[name],
                        "sha256": bundle.shard_sha256[name],
                    }
                    for name in sorted(bundle.shard_sha256)
                },
            }
        elif args.component == "assistant":
            converted = convert_official_muse_glimmer_assistant_safetensors(
                args.source,
                args.output_dir,
                target_checkpoint_sha256=args.target_checkpoint_sha256,
            )
            payload = asdict(converted)
        else:
            converted = convert_official_muse_glimmer_safetensors(
                args.source,
                args.output_dir,
                component=args.component,
            )
            payload = asdict(converted)
    except (FileExistsError, FileNotFoundError, MuseGlimmerCheckpointError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def _lightweight_muse_glimmer_gguf_migrate_main(
    argv: list[str] | None = None,
) -> int:
    import argparse

    tokens = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="nfn migrate muse-glimmer-gguf-to-native",
        description=(
            "Authenticate one or both canonical Muse Glimmer K-Quant GGUF files "
            "and publish a self-contained resident artifact."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--gguf", action="append", required=True, metavar="FILE")
    parser.add_argument(
        "--dflash",
        metavar="FILE",
        help="Optional canonical packed DFlash companion GGUF.",
    )
    parser.add_argument("--tokenizer-source", required=True, metavar="DIR")
    parser.add_argument("--output-dir", required=True, metavar="DIR")
    parser.add_argument(
        "--primary",
        choices=("k-quant-dynamic", "k-quant-17gb"),
    )
    try:
        args = parser.parse_args(tokens[2:])
    except SystemExit as exc:
        return int(exc.code)
    from neuralfn.native_gguf import (
        GGUFError,
        publish_muse_glimmer_kquant_execution_bundle,
    )

    try:
        manifest_path = publish_muse_glimmer_kquant_execution_bundle(
            args.gguf,
            tokenizer_source=args.tokenizer_source,
            output_root=args.output_dir,
            primary_variant=args.primary,
            dflash_path=args.dflash,
        )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileExistsError, FileNotFoundError, GGUFError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "primary_checkpoint_variant": payload["primary_checkpoint_variant"],
                "checkpoint_variants": sorted(payload["checkpoint_variants"]),
                "dflash": "dflash" in payload.get("companion_checkpoints", {}),
                "resident_cpu": True,
                "whole_model_cuda": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _lightweight_muse_glimmer_lora_migrate_main(
    argv: list[str] | None = None,
) -> int:
    import argparse

    tokens = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="nfn migrate muse-glimmer-lora-to-native",
        description=(
            "Authenticate and atomically attach a native Muse Glimmer LoRA or "
            "QLoRA adapter checkpoint to a compatible resident bundle."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--artifact", required=True, metavar="DIR")
    parser.add_argument("--checkpoint", required=True, metavar="DIR")
    try:
        args = parser.parse_args(tokens[2:])
    except SystemExit as exc:
        return int(exc.code)

    from neuralfn.native_muse_glimmer_checkpoint import (
        MuseGlimmerCheckpointError,
        attach_native_muse_glimmer_lora,
        inspect_native_muse_glimmer_lora_checkpoint,
    )

    try:
        descriptor = inspect_native_muse_glimmer_lora_checkpoint(args.checkpoint)
        manifest_path = attach_native_muse_glimmer_lora(
            args.artifact, args.checkpoint
        )
    except (
        FileExistsError,
        FileNotFoundError,
        MuseGlimmerCheckpointError,
        OSError,
        ValueError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "adapter_sha256": descriptor["target_sha256"],
                "training_adapter": descriptor["source"]["training_adapter"],
                "resident_cpu": descriptor["capabilities"]["resident_cpu"],
                "resident_cuda": descriptor["capabilities"]["resident_cuda"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _is_native_serve_request(argv: list[str]) -> bool:
    return bool(argv and argv[0] == "infer" and _has_any(argv, "--serve"))


def _read_native_ir_manifest_candidate(candidate: Path) -> dict | None:
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (
        isinstance(payload, dict)
        and payload.get("schema") == "neuralfn.native_execution_manifest"
        and payload.get("version") == 1
    ):
        return payload
    return None


def _resolve_checkpoint_sibling_manifest(requested: Path) -> Path | None:
    """Return a sibling manifest only when it binds this exact contained file."""

    manifest_path = requested.parent / "native-execution-manifest.json"
    if not manifest_path.is_file():
        return None
    payload = _read_native_ir_manifest_candidate(manifest_path)
    checkpoint = payload.get("checkpoint") if payload is not None else None
    if not isinstance(checkpoint, dict):
        return None
    relative = checkpoint.get("artifact_path")
    if not isinstance(relative, str) or not relative.strip():
        return None
    declared = Path(relative)
    if declared.is_absolute():
        return None
    artifact_root = manifest_path.parent.resolve()
    bound_checkpoint = (artifact_root / declared).resolve()
    try:
        bound_checkpoint.relative_to(artifact_root)
        requested_checkpoint = requested.resolve(strict=True)
    except (OSError, ValueError):
        return None
    return manifest_path if requested_checkpoint == bound_checkpoint else None


def _resolve_native_ir_manifest(argv: list[str]) -> Path | None:
    if (
        not argv
        or argv[0] != "infer"
        or _has_any(argv, "-h", "--help", "--serve", "--graph", "--plan", "--plan-auto")
    ):
        return None
    raw = _arg_value(argv, "--checkpoint", "--native-checkpoint")
    if not raw:
        return None
    requested = Path(raw).expanduser()
    candidate = (
        requested / "native-execution-manifest.json" if requested.is_dir() else requested
    )
    if not candidate.is_file():
        return None
    if candidate.name == "native-execution-manifest.json":
        return candidate
    sibling_manifest = _resolve_checkpoint_sibling_manifest(requested)
    if sibling_manifest is not None:
        return sibling_manifest
    if candidate.suffix.lower() == ".bin":
        # Raw native checkpoints can be hundreds of MiB.  Once no sibling
        # manifest has claimed the file, leave it to the legacy detector
        # without attempting to decode the model payload as JSON.
        return None
    if _read_native_ir_manifest_candidate(candidate) is not None:
        return candidate
    return None


def _is_native_ir_infer_request(argv: list[str]) -> bool:
    return _resolve_native_ir_manifest(argv) is not None


def _legacy_infer_inputs(argv: list[str]) -> tuple[str | None, str | None] | None:
    """Return graph/weights inputs only for the retained Python inference path."""

    if (
        not argv
        or argv[0] != "infer"
        or _has_any(argv, "-h", "--help", "--plan", "--plan-auto")
        or _resolve_native_ir_manifest(argv) is not None
    ):
        return None
    graph = _arg_value(argv, "--graph")
    weights = _arg_value(argv, "--weights")
    checkpoint = _arg_value(argv, "--checkpoint")
    if graph:
        return graph, weights or checkpoint
    candidate = checkpoint or weights
    if candidate and Path(candidate).suffix.lower() in {".pt", ".pth", ".ckpt"}:
        return None, candidate
    return None


def _legacy_infer_requests_turboquant(argv: list[str]) -> bool:
    requested = (_arg_value(argv, "--kv-cache") or "").strip().lower().replace("_", "-")
    return requested == "turboquant"


def _is_blocked_legacy_infer_request(argv: list[str]) -> bool:
    return _legacy_infer_inputs(argv) is not None and (
        _has_any(argv, "--serve") or _legacy_infer_requests_turboquant(argv)
    )


def _legacy_infer_migration_guidance(argv: list[str]) -> tuple[str, str]:
    inputs = _legacy_infer_inputs(argv)
    if inputs is None:
        raise ValueError("legacy inference migration guidance requires a legacy artifact")
    graph, weights = inputs
    if graph is not None:
        graph_path = Path(graph)
        output_dir = graph_path.parent / f"{graph_path.stem}-native"
        command = [
            "nfn",
            "migrate",
            "graph-to-native",
            "--graph",
            graph,
        ]
        if weights is not None:
            command.extend(("--weights", weights))
        command.extend(("--output-dir", str(output_dir)))
        return shlex.join(command), str(output_dir)

    assert weights is not None
    weights_path = Path(weights)
    output_dir = weights_path.parent / f"{weights_path.stem}-native"
    command = [
        "nfn",
        "migrate",
        "graph-to-native",
        "--graph",
        "MATCHING_GRAPH.json",
        "--weights",
        weights,
        "--output-dir",
        str(output_dir),
    ]
    return shlex.join(command), str(output_dir)


def _blocked_legacy_infer_main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    inputs = _legacy_infer_inputs(tokens)
    if inputs is None:
        return 2
    graph, weights = inputs
    migration, _output_dir = _legacy_infer_migration_guidance(tokens)
    requested_feature = "--serve" if _has_any(tokens, "--serve") else "TurboQuant"
    if graph is None:
        assert weights is not None
        print(
            f"Graphless Parameter Golf inference does not support {requested_feature}. "
            "A matching NeuralFn graph is a prerequisite because the checkpoint has no "
            "serialized topology.",
            file=sys.stderr,
        )
    else:
        print(
            f"Legacy graph inference does not support {requested_feature}. Migrate the "
            "supplied graph before requesting native-only features.",
            file=sys.stderr,
        )
    if graph is None:
        print(
            "After replacing MATCHING_GRAPH.json with the actual graph path, use this "
            "migration template:",
            file=sys.stderr,
        )
    else:
        print("Run this exact migration command:", file=sys.stderr)
    print(f"  {migration}", file=sys.stderr)
    print(
        "Migration validates and preserves the graph/tensor bundle; it does not make "
        "legacy weights resident-loadable. Serving and TurboQuant additionally require "
        "a compatible resident native dense-v5 checkpoint.",
        file=sys.stderr,
    )
    return 2


def _legacy_infer_main(
    argv: list[str] | None = None,
    *,
    stdin_isatty: bool | None = None,
    stdout_isatty: bool | None = None,
) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    migration, _output_dir = _legacy_infer_migration_guidance(tokens)
    graph, _weights = _legacy_infer_inputs(tokens) or (None, None)
    if graph is None:
        print(
            "DEPRECATED: graphless Parameter Golf inference remains available for "
            "compatibility but is not a resident native runtime. A matching NeuralFn "
            "graph is required before migration. Migration template:",
            file=sys.stderr,
        )
    else:
        print(
            "DEPRECATED: legacy graph inference remains available for compatibility "
            "but is not a resident native runtime. Migrate with:",
            file=sys.stderr,
        )
    print(f"  {migration}", file=sys.stderr)
    impl = _load_full_impl()
    kwargs: dict[str, bool] = {}
    if stdin_isatty is not None:
        kwargs["stdin_isatty"] = stdin_isatty
    if stdout_isatty is not None:
        kwargs["stdout_isatty"] = stdout_isatty
    return int(impl.main(tokens, **kwargs))


def _native_ir_infer_main(
    argv: list[str] | None = None,
    *,
    stdin_isatty: bool | None = None,
    stdout_isatty: bool | None = None,
) -> int:
    import argparse

    tokens = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="nfn infer",
        description=(
            "Run one Native Execution artifact through the in-process resident model/session API."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--checkpoint",
        "--native-checkpoint",
        dest="checkpoint",
        required=True,
        metavar="ARTIFACT",
    )
    parser.add_argument("--runtime", choices=("auto", "cpu", "native-cuda"), default="auto")
    parser.add_argument(
        "--weight-precision",
        choices=("auto", "bf16", "k-quant-dynamic", "k-quant-17gb"),
        default="auto",
        help="Select the resident weight artifact; auto is quality-first within the memory budget.",
    )
    parser.add_argument(
        "--speculative-decoding",
        "--speculative",
        choices=("off", "auto", "required"),
        default="auto",
        help="Use a bound DFlash assistant; required fails instead of using target-only decoding.",
    )
    parser.add_argument(
        "--companion-checkpoint",
        action="append",
        choices=("dflash", "mmproj", "lora"),
        default=[],
        metavar="NAME",
        help="Load an authenticated optional DFlash, mmproj, or native LoRA companion.",
    )
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-tokens", default="", metavar="IDS")
    parser.add_argument("--chat-mode", choices=("transcript", "stateless"), default=None)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--chat-template", default="auto", metavar="auto|plain_roles|PATH")
    parser.add_argument("--kv-cache", choices=("off", "auto", "full", "turboquant"), default="auto")
    parser.add_argument(
        "--turboquant-profile",
        choices=("mse-3.5", "qjl-3.5"),
        default="mse-3.5",
    )
    parser.add_argument(
        "--turboquant-attention-backend",
        choices=("cpu", "tile-cuda"),
        default="cpu",
    )
    parser.add_argument("--tile-ops-lib", default=None, metavar="PATH")
    parser.add_argument("--cuda-runtime-lib", default=None, metavar="PATH_OR_SONAME")
    parser.add_argument("--cuda-device", type=int, default=0, metavar="INDEX")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--native-info", action="store_true")
    try:
        args = parser.parse_args(tokens[1:])
    except SystemExit as exc:
        return int(exc.code)

    input_tty = sys.stdin.isatty() if stdin_isatty is None else bool(stdin_isatty)
    output_tty = sys.stdout.isatty() if stdout_isatty is None else bool(stdout_isatty)
    interactive = input_tty and output_tty
    manifest_path = _resolve_native_ir_manifest(tokens)
    if manifest_path is None:
        return 2

    from neuralfn.native_chat import NativeChatConfigurationError
    from neuralfn.native_cli import (
        NativeArtifactCLIConfig,
        parse_native_prompt_token_ids,
        run_native_artifact_cli,
    )
    from neuralfn.native_inference import (
        KVCacheConfig,
        NativeInferenceError,
        NativeModelLoadConfig,
    )

    try:
        config = NativeArtifactCLIConfig(
            artifact=manifest_path,
            prompt=args.prompt,
            prompt_token_ids=parse_native_prompt_token_ids(args.prompt_tokens),
            chat_mode=args.chat_mode,
            system_prompt=args.system_prompt,
            chat_template=args.chat_template,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            kv_cache=KVCacheConfig(
                mode=args.kv_cache,
                turboquant_profile=args.turboquant_profile,
                turboquant_attention_backend=args.turboquant_attention_backend,
                tile_ops_lib=(
                    args.tile_ops_lib
                    if args.turboquant_attention_backend == "tile-cuda"
                    else None
                ),
                cuda_runtime_lib=(
                    args.cuda_runtime_lib
                    if args.turboquant_attention_backend == "tile-cuda"
                    else None
                ),
                cuda_device=(
                    args.cuda_device
                    if args.turboquant_attention_backend == "tile-cuda"
                    else 0
                ),
            ),
            model_load=NativeModelLoadConfig(
                weight_precision=args.weight_precision,
                runtime=args.runtime,
                tile_ops_lib=args.tile_ops_lib,
                cuda_runtime_lib=args.cuda_runtime_lib,
                cuda_device=args.cuda_device,
                speculative_decoding=args.speculative_decoding,
                companion_checkpoints=tuple(args.companion_checkpoint),
            ),
            native_info=bool(args.native_info),
        )
        return int(run_native_artifact_cli(config, interactive=interactive))
    except KeyboardInterrupt:
        return 130
    except (
        FileNotFoundError,
        ImportError,
        NativeChatConfigurationError,
        NativeInferenceError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _native_serve_main(argv: list[str] | None = None) -> int:
    import argparse

    tokens = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="nfn infer --serve",
        description=(
            "Serve one proven resident Native Execution artifact through a lean, "
            "bounded OpenAI-compatible API."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--serve", action="store_true", help="Start the resident inference server.")
    parser.add_argument(
        "--checkpoint",
        "--native-checkpoint",
        dest="checkpoint",
        required=True,
        metavar="ARTIFACT",
        help="Native artifact directory or native-execution-manifest.json path.",
    )
    parser.add_argument("--runtime", choices=("auto", "cpu", "native-cuda"), default="auto")
    parser.add_argument(
        "--weight-precision",
        choices=("auto", "bf16", "k-quant-dynamic", "k-quant-17gb"),
        default="auto",
        help="Select the server's resident weight artifact at startup.",
    )
    parser.add_argument(
        "--speculative-decoding",
        "--speculative",
        choices=("off", "auto", "required"),
        default="auto",
        help="Set startup DFlash policy; requests cannot override this server policy.",
    )
    parser.add_argument(
        "--companion-checkpoint",
        action="append",
        choices=("dflash", "mmproj", "lora"),
        default=[],
        metavar="NAME",
        help="Load an authenticated optional DFlash, mmproj, or native LoRA companion at startup.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default=None, metavar="NAME")
    parser.add_argument(
        "--queue-capacity",
        type=int,
        default=8,
        metavar="N",
        help="Maximum waiting generations in front of the single compute worker.",
    )
    parser.add_argument(
        "--session-limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum admitted request sessions, including the running request and "
            "queued reservations (default: queue capacity plus one)."
        ),
    )
    parser.add_argument("--max-output-tokens", type=int, default=256, metavar="N")
    parser.add_argument(
        "--kv-cache",
        choices=("off", "auto", "full", "turboquant"),
        default="auto",
        help=(
            "Resident cache request. Auto selects the jointly proven lossless full cache; "
            "TurboQuant selects only a jointly proven profile and backend."
        ),
    )
    parser.add_argument(
        "--turboquant-profile",
        choices=("mse-3.5", "qjl-3.5"),
        default="mse-3.5",
    )
    parser.add_argument(
        "--turboquant-attention-backend",
        choices=("cpu", "tile-cuda"),
        default="cpu",
        help="Use CPU packed attention or the explicit strict Tile-CUDA sidecar.",
    )
    parser.add_argument("--tile-ops-lib", default=None, metavar="PATH")
    parser.add_argument("--cuda-runtime-lib", default=None, metavar="PATH_OR_SONAME")
    parser.add_argument("--cuda-device", type=int, default=0, metavar="INDEX")
    parser.add_argument(
        "--chat-template",
        default="auto",
        metavar="auto|plain_roles|PATH",
        help="Use artifact metadata or an explicit lean chat renderer fallback.",
    )
    parser.add_argument("--api-key-file", default=None, metavar="PATH")
    parser.add_argument(
        "--state-db",
        default=None,
        metavar="PATH",
        help=(
            "Enable stateful Responses with local compaction, Conversations, and "
            "restart-safe background jobs using a private versioned SQLite database at PATH."
        ),
    )
    parser.add_argument(
        "--prefix-cache-capacity",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Retain at most N proven resident Responses prefixes (requires --state-db; default: 0)."
        ),
    )
    parser.add_argument("--allow-unauthenticated-remote", action="store_true")
    parser.add_argument("--log-level", default="info")
    try:
        args = parser.parse_args(tokens[1:])
    except SystemExit as exc:
        return int(exc.code)

    from neuralfn.native_inference import (
        KVCacheConfig,
        NativeInferenceError,
        NativeModelLoadConfig,
    )
    from neuralfn.native_serve import (
        NativeServeConfig,
        NativeServingConfigurationError,
        run_native_inference_server,
    )

    try:
        config = NativeServeConfig(
            artifact=Path(args.checkpoint),
            host=args.host,
            port=args.port,
            served_model_name=args.served_model_name,
            queue_capacity=args.queue_capacity,
            session_limit=args.session_limit,
            max_output_tokens=args.max_output_tokens,
            kv_cache=KVCacheConfig(
                mode=args.kv_cache,
                turboquant_profile=args.turboquant_profile,
                turboquant_attention_backend=args.turboquant_attention_backend,
                tile_ops_lib=(
                    args.tile_ops_lib
                    if args.turboquant_attention_backend == "tile-cuda"
                    else None
                ),
                cuda_runtime_lib=(
                    args.cuda_runtime_lib
                    if args.turboquant_attention_backend == "tile-cuda"
                    else None
                ),
                cuda_device=(
                    args.cuda_device
                    if args.turboquant_attention_backend == "tile-cuda"
                    else 0
                ),
            ),
            model_load=NativeModelLoadConfig(
                weight_precision=args.weight_precision,
                runtime=args.runtime,
                tile_ops_lib=args.tile_ops_lib,
                cuda_runtime_lib=args.cuda_runtime_lib,
                cuda_device=args.cuda_device,
                session_count=(
                    args.session_limit
                    if args.session_limit is not None
                    else args.queue_capacity + 1
                ),
                speculative_decoding=args.speculative_decoding,
                companion_checkpoints=tuple(args.companion_checkpoint),
            ),
            chat_template=args.chat_template,
            api_key_file=(Path(args.api_key_file) if args.api_key_file else None),
            state_db=(Path(args.state_db) if args.state_db else None),
            prefix_cache_capacity=args.prefix_cache_capacity,
            allow_unauthenticated_remote=bool(args.allow_unauthenticated_remote),
            log_level=args.log_level,
        )
        run_native_inference_server(config)
        return 0
    except KeyboardInterrupt:
        return 130
    except (
        FileNotFoundError,
        ImportError,
        NativeInferenceError,
        NativeServingConfigurationError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 1


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


def _native_infer_artifact_arg(argv: list[str]) -> str | None:
    if not argv or argv[0] != "infer":
        return None
    return _arg_value(argv, "--native-checkpoint", "--checkpoint", "--weights", "--graph")


def _native_infer_requested_runtime(argv: list[str]) -> str:
    return (_arg_value(argv, "--runtime") or "auto").strip().lower().replace("_", "-")


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


def _resolve_native_family_infer_checkpoint(argv: list[str]) -> Path | None:
    raw_checkpoint = _native_infer_checkpoint_arg(argv)
    if not raw_checkpoint:
        return None
    checkpoint_path = Path(raw_checkpoint).expanduser()
    try:
        from neuralfn.native_family import is_native_family_checkpoint, latest_native_family_checkpoint

        if checkpoint_path.is_dir():
            return latest_native_family_checkpoint(checkpoint_path)
        if is_native_family_checkpoint(checkpoint_path):
            return checkpoint_path
    except Exception:
        return None
    return None


def _is_lightweight_native_family_infer(argv: list[str]) -> bool:
    if argv and argv[0] == "infer" and _has_any(argv, "--verify-all") and _native_infer_checkpoint_arg(argv):
        return True
    return _resolve_native_family_infer_checkpoint(argv) is not None


def _is_invalid_native_gpt_infer(argv: list[str]) -> bool:
    if not argv or argv[0] != "infer" or _has_any(argv, "-h", "--help", "--plan", "--plan-auto"):
        return False
    runtime = _native_infer_requested_runtime(argv)
    if runtime not in {"native-cuda", "tile-cuda", "cuda", "native"}:
        return False
    return _resolve_native_infer_checkpoint(argv) is None


def _invalid_native_gpt_infer_main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    artifact = _native_infer_artifact_arg(tokens)
    target = f" for {artifact}" if artifact else ""
    print(
        "Native GPT inference requires a native model_*.bin checkpoint or a directory "
        f"containing native checkpoints{target}. Exported .pt/.json graph artifacts use "
        "the legacy graph-backed runtime and are not accepted with --runtime native-cuda. "
        "Use --checkpoint /path/to/model_00020000.bin, or train with native checkpoint "
        "export enabled.",
        file=sys.stderr,
    )
    return 2


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


def _lightweight_native_family_infer_main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    from neuralfn.native_family import (
        audit_native_family_checkpoint_template_coverage,
        is_native_family_checkpoint,
        list_native_family_checkpoints,
        parse_native_family_template_list,
        read_native_family_checkpoint_info,
        render_native_family_checkpoint_sampler_text,
        sample_native_family_checkpoint,
        verify_native_family_checkpoint,
    )

    prompt_tokens = _arg_value(tokens, "--prompt-tokens") or ""
    require_architecture_forward = _has_any(tokens, "--require-architecture-forward")
    if _has_any(tokens, "--verify-all"):
        raw_checkpoint = _native_infer_checkpoint_arg(tokens)
        root = Path(raw_checkpoint).expanduser() if raw_checkpoint else checkpoint_path
        if root.is_dir():
            checkpoints = list(list_native_family_checkpoints(root))
        elif is_native_family_checkpoint(root):
            checkpoints = [root]
        else:
            checkpoints = []
        import json

        results = []
        for path in checkpoints:
            verification = verify_native_family_checkpoint(
                path,
                prompt_tokens=prompt_tokens if prompt_tokens.strip() else None,
                max_new_tokens=_arg_int_value(tokens, "--max-new-tokens", 1),
                require_architecture_forward=require_architecture_forward,
            )
            results.append(
                {
                    "path": str(verification.path),
                    "passed": verification.passed,
                    "errors": list(verification.errors),
                    "model_family": verification.info.model_family,
                    "template_name": verification.info.template_name,
                    "checkpoint_kind": verification.info.checkpoint_kind,
                    "transition_count": verification.info.transition_count,
                    "done_marker_exists": verification.info.done_marker_exists,
                    "full_template_parameter_state": verification.info.full_template_parameter_state,
                    "parameter_storage": verification.info.parameter_storage,
                    "parameter_initialization": verification.info.parameter_initialization,
                    "dense_parameter_state_reconstructable": verification.info.dense_parameter_state_reconstructable,
                    "base_parameter_initialization": verification.info.base_parameter_initialization,
                    "base_parameter_seed": verification.info.base_parameter_seed,
                    "base_parameter_scale": verification.info.base_parameter_scale,
                    "parameter_data_size_matches": verification.info.parameter_data_size_matches,
                    "writer_verification_passed": verification.info.writer_verification_passed,
                    "writer_verification_update_probe_count": verification.info.writer_verification_update_probe_count,
                    "writer_dense_base_initialization_verified": (
                        verification.info.writer_dense_base_initialization_verified
                    ),
                    "writer_dense_base_probe_count": verification.info.writer_dense_base_probe_count,
                    "writer_dense_base_probe_checksum": verification.info.writer_dense_base_probe_checksum,
                    "writer_verification_error": verification.info.writer_verification_error,
                    "architecture_forward_inference_supported": verification.info.architecture_forward_inference_supported,
                    "parameter_lm_head_inference_supported": verification.info.parameter_lm_head_inference_supported,
                    "working_model_inference_path": verification.info.working_model_inference_path,
                    "architecture_forward_inference_used": bool(
                        verification.sample.get("architecture_forward_inference_used")
                    ),
                    "parameter_lm_head_inference_used": bool(
                        verification.sample.get("parameter_lm_head_inference_used")
                    ),
                }
            )
        passed_count = sum(1 for result in results if bool(result["passed"]))
        coverage_payload = None
        required_template_raw = _arg_value(tokens, "--required-templates") or ""
        required_templates = {
            template: _NATIVE_TEMPLATE_FAMILY_ALIASES.get(
                template,
                _NATIVE_FAMILY_CHECKPOINT_TEMPLATE_TARGETS.get(template, template),
            )
            for template in parse_native_family_template_list(required_template_raw)
        }
        if _has_any(tokens, "--require-covered-templates"):
            required_templates.update(_NATIVE_FAMILY_CHECKPOINT_TEMPLATE_TARGETS)
            required_templates.update(_NATIVE_TEMPLATE_FAMILY_ALIASES)
        if required_templates:
            coverage_payload = audit_native_family_checkpoint_template_coverage(
                root,
                required_templates=required_templates,
                prompt_tokens=prompt_tokens if prompt_tokens.strip() else None,
                max_new_tokens=_arg_int_value(tokens, "--max-new-tokens", 1),
                require_architecture_forward=require_architecture_forward,
            )
        payload = {
            "status": "native-family-checkpoint-verification-set",
            "path": str(root),
            "checkpoint_count": len(results),
            "passed_count": passed_count,
            "failed_count": len(results) - passed_count,
            "passed": bool(results)
            and passed_count == len(results)
            and (coverage_payload is None or bool(coverage_payload["passed"])),
            "architecture_forward_required": require_architecture_forward,
            "results": results,
        }
        if coverage_payload is not None:
            payload["covered_template_verification"] = coverage_payload
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload["passed"] else 2

    checkpoint_path = _resolve_native_family_infer_checkpoint(tokens)
    if checkpoint_path is None:
        return 2

    info = read_native_family_checkpoint_info(checkpoint_path)
    print("Native family checkpoint detected")
    print(f"  path: {info.path}")
    print(f"  model_family: {info.model_family}")
    print(f"  template_name: {info.template_name}")
    print(f"  checkpoint_kind: {info.checkpoint_kind}")
    print(f"  vocab_size: {info.vocab_size}")
    print(f"  transition_count: {info.transition_count}")
    print(f"  steps_completed: {info.steps_completed}")
    print(f"  parameter_state_type: {info.parameter_state_type}")
    print(f"  parameter_storage: {info.parameter_storage}")
    print(f"  parameter_initialization: {info.parameter_initialization}")
    print(f"  dense_parameter_state_reconstructable: {info.dense_parameter_state_reconstructable}")
    print(f"  base_parameter_initialization: {info.base_parameter_initialization}")
    print(f"  base_parameter_seed: {info.base_parameter_seed}")
    print(f"  base_parameter_scale: {info.base_parameter_scale}")
    print(f"  full_template_parameter_state: {info.full_template_parameter_state}")
    print(f"  parameter_buffer_count: {info.parameter_buffer_count}")
    print(f"  parameter_elements: {info.parameter_elements}")
    print(f"  persisted_parameter_elements: {info.persisted_parameter_elements}")
    print(f"  trained_parameter_elements: {info.trained_parameter_elements}")
    print(f"  parameter_update_checksum: {info.parameter_update_checksum}")
    print(f"  writer_verification_passed: {info.writer_verification_passed}")
    print(f"  writer_verification_update_probe_count: {info.writer_verification_update_probe_count}")
    print(f"  writer_dense_base_initialization_verified: {info.writer_dense_base_initialization_verified}")
    print(f"  writer_dense_base_probe_count: {info.writer_dense_base_probe_count}")
    print(f"  writer_dense_base_probe_checksum: {info.writer_dense_base_probe_checksum}")
    print(f"  writer_verification_error: {info.writer_verification_error}")
    print(f"  architecture_forward_inference_supported: {info.architecture_forward_inference_supported}")
    print(f"  parameter_lm_head_inference_supported: {info.parameter_lm_head_inference_supported}")
    print(f"  working_model_inference_path: {info.working_model_inference_path}")
    print(f"  transition_sampler_inference_supported: {info.transition_sampler_inference_supported}")
    print(f"  parameter_data_path: {info.parameter_data_path or ''}")
    print(f"  parameter_data_exists: {info.parameter_data_exists}")
    print(f"  parameter_data_bytes: {info.parameter_data_bytes}")
    print(f"  expected_parameter_data_bytes: {info.expected_parameter_data_bytes}")
    print(f"  parameter_data_size_matches: {info.parameter_data_size_matches}")
    print(f"  DONE marker: {'present' if info.done_marker_exists else 'missing'}")
    if _has_any(tokens, "--verify"):
        verification = verify_native_family_checkpoint(
            checkpoint_path,
            prompt_tokens=prompt_tokens if prompt_tokens.strip() else None,
            max_new_tokens=_arg_int_value(tokens, "--max-new-tokens", 1),
            require_architecture_forward=require_architecture_forward,
        )
        import json

        print(
            json.dumps(
                {
                    "status": "native-family-checkpoint-verification",
                    "path": str(verification.path),
                    "passed": verification.passed,
                    "errors": list(verification.errors),
                    "model_family": verification.info.model_family,
                    "template_name": verification.info.template_name,
                    "checkpoint_kind": verification.info.checkpoint_kind,
                    "transition_count": verification.info.transition_count,
                    "done_marker_exists": verification.info.done_marker_exists,
                    "full_template_parameter_state": verification.info.full_template_parameter_state,
                    "parameter_storage": verification.info.parameter_storage,
                    "parameter_initialization": verification.info.parameter_initialization,
                    "dense_parameter_state_reconstructable": verification.info.dense_parameter_state_reconstructable,
                    "base_parameter_initialization": verification.info.base_parameter_initialization,
                    "base_parameter_seed": verification.info.base_parameter_seed,
                    "base_parameter_scale": verification.info.base_parameter_scale,
                    "parameter_data_exists": verification.info.parameter_data_exists,
                    "parameter_data_bytes": verification.info.parameter_data_bytes,
                    "expected_parameter_data_bytes": verification.info.expected_parameter_data_bytes,
                    "parameter_data_size_matches": verification.info.parameter_data_size_matches,
                    "trained_parameter_elements": verification.info.trained_parameter_elements,
                    "parameter_update_checksum": verification.info.parameter_update_checksum,
                    "writer_verification_passed": verification.info.writer_verification_passed,
                    "writer_verification_update_probe_count": verification.info.writer_verification_update_probe_count,
                    "writer_dense_base_initialization_verified": (
                        verification.info.writer_dense_base_initialization_verified
                    ),
                    "writer_dense_base_probe_count": verification.info.writer_dense_base_probe_count,
                    "writer_dense_base_probe_checksum": verification.info.writer_dense_base_probe_checksum,
                    "writer_verification_error": verification.info.writer_verification_error,
                    "architecture_forward_inference_supported": verification.info.architecture_forward_inference_supported,
                    "architecture_forward_required": require_architecture_forward,
                    "parameter_lm_head_inference_supported": verification.info.parameter_lm_head_inference_supported,
                    "working_model_inference_path": verification.info.working_model_inference_path,
                    "transition_sampler_inference_supported": verification.info.transition_sampler_inference_supported,
                    "sample": verification.sample,
                },
                sort_keys=True,
            )
        )
        return 0 if verification.passed else 2
    if _has_any(tokens, "--native-info"):
        return 0
    if not prompt_tokens.strip():
        print("Native family checkpoint inference requires --prompt-tokens.", file=sys.stderr)
        return 2
    try:
        payload = sample_native_family_checkpoint(
            checkpoint_path,
            prompt_tokens=prompt_tokens,
            max_new_tokens=_arg_int_value(tokens, "--max-new-tokens", 64),
        )
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    import json

    print(json.dumps(payload, sort_keys=True))
    rendered = render_native_family_checkpoint_sampler_text(payload)
    if rendered:
        print(rendered)
    return 0


def _native_prompt_tokens(tokens: list[str]) -> str:
    from neuralfn.native_gpt import native_gpt_prompt_tokens

    return native_gpt_prompt_tokens(
        prompt=_arg_value(tokens, "--prompt") or "",
        prompt_tokens=_arg_value(tokens, "--prompt-tokens") or "",
        encoding_name=_arg_value(tokens, "--tokenizer") or ("gpt2" if _has_any(tokens, "--tokgpt2") else "gpt2"),
    )


def _arg_int_value(tokens: list[str], flag: str, default: int) -> int:
    value = _arg_value(tokens, flag)
    return default if value is None else int(value)


def _arg_float_value(tokens: list[str], flag: str, default: float) -> float:
    value = _arg_value(tokens, flag)
    return default if value is None else float(value)


def _run_lightweight_native_gpt_sampler(tokens: list[str], checkpoint: str) -> int:
    try:
        from neuralfn.native_gpt import run_native_gpt_checkpoint_sampler

        result = run_native_gpt_checkpoint_sampler(
            checkpoint,
            prompt_tokens=_native_prompt_tokens(tokens),
            max_new_tokens=_arg_int_value(tokens, "--max-new-tokens", 64),
            temperature=_arg_float_value(tokens, "--temperature", 0.8),
            top_k=_arg_int_value(tokens, "--top-k", 32),
            repetition_penalty=_arg_float_value(tokens, "--repetition-penalty", 1.0),
            seed=_arg_int_value(tokens, "--seed", 1337),
            strict_tile_ops_lib=_arg_value(tokens, "--strict-tile-ops-lib"),
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
    if _has_any(argv, "-h", "--help", "--plan", "--plan-auto", "--jepa", "--tui", "--interactive"):
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
    if _is_dense_gpt_native_model(model):
        from neuralfn.native_gpt2 import resolve_fresh_native_gpt2_cli

        return resolve_fresh_native_gpt2_cli(repo_root=ROOT.parent)
    if model.strip().lower().replace("_", "-") == "embedding":
        from neuralfn.native_embedding import resolve_native_embedding_cli

        return resolve_native_embedding_cli(ROOT.parent)
    requested_train_cli = os.environ.get("NFN_NATIVE_TRAIN_CLI", "").strip()
    if requested_train_cli:
        return requested_train_cli
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
    normalized = model.strip().lower().replace("_", "-")
    if normalized != "embedding" and os.environ.get("NFN_NATIVE_TRAIN_CLI", "").strip():
        return None
    normalized = _NATIVE_TEMPLATE_FAMILY_ALIASES.get(normalized, normalized)
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
    return "nanogpt" if model in {"nano-gpt", "nano_gpt"} else model


def _native_gpt_cli_uses_linked_tile_ops(path: str) -> bool:
    return Path(path).name in {
        "nfn_gpt_native_train_linked",
        "nfn-gpt-native-train-linked",
    }


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
    "--native-cuda-smoke-llama-loop",
    "--native-cuda-smoke-llama-lm-head-step",
    "--native-cuda-smoke-llama-train-step",
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

_TRAIN_TUI_MODELS = (
    ("gpt", "Dense GPT native trainer"),
    ("embedding", "Native text bi-encoder pretraining and post-training"),
    ("gpt2", "Dense GPT-2 shape on the GPT trainer"),
    ("gpt3", "Dense GPT-3-like long-context run"),
    ("nanogpt", "NanoGPT template on the dense GPT trainer"),
    ("gpt2-evo", "GPT-2 Evo preflight (whole-block training blocked)"),
    ("llama", "LLaMA family native trainer"),
    ("mixllama", "MixLLaMA/MoE family native trainer"),
    ("jepa", "JEPA native family trainer"),
    ("semantic-router-moe", "Semantic router MoE native family trainer"),
    ("semantic-moe-jepa-evo", "Semantic MoE JEPA Evo native trainer"),
)

_TRAIN_TUI_TEMPLATES = {
    "embedding": ("default",),
    "gpt": ("gpt", "gpt2", "gpt2_moa", "gpt2_modern", "gpt3", "nanogpt"),
    "gpt2": ("gpt2", "gpt", "gpt2_moa", "gpt2_modern"),
    "gpt3": ("gpt3", "gpt", "gpt2_moa"),
    "nanogpt": ("nanogpt", "nanogpt_modern", "nanogpt_megakernel"),
    "gpt2-evo": ("default",),
    "llama": ("llama", "llama-modern", "llama-fast", "kv-pca-llama"),
    "mixllama": ("mixllama", "mixllama-fast", "moe", "deepseek-v3"),
    "jepa": ("llm-jepa", "dense-jepa-evo", "dense-jepa-evo-modern"),
    "semantic-router-moe": ("semantic-router-moe", "semantic-moe-jepa-evo"),
    "semantic-moe-jepa-evo": ("semantic-moe-jepa-evo", "semantic-moe-jepa-evo-modern", "diff-semantic-moe-jepa-evo"),
}

_TRAIN_TUI_GPT_TEMPLATE_MODELS = {"gpt", "gpt2", "gpt3", "nanogpt"}

_TRAIN_TUI_HYPERPARAMS = (
    ("--max-steps", "max_steps", "Steps", "20000", "int", "Optimizer steps to run."),
    ("--train-seq-len", "train_seq_len", "Sequence length", "1024", "int", "Tokens per training sample."),
    ("--batch-size", "batch_size", "Microbatch rows", "64", "int", "Rows sampled per native microbatch."),
    ("--train-batch-tokens", "train_batch_tokens", "Batch tokens", "524288", "int", "Effective tokens per optimizer step."),
    ("--learning-rate", "learning_rate", "Learning rate", "0.0006", "float", "AdamW learning rate."),
    ("--lr-schedule", "lr_schedule", "LR schedule", "cosine", "choice", "Learning-rate schedule: cosine or constant."),
    ("--final-lr-fraction", "final_lr_fraction", "Final LR fraction", "0.0", "float", "Final LR as a fraction of the base LR for cosine decay."),
    ("--weight-decay", "weight_decay", "Weight decay", "0.02", "float", "AdamW weight decay."),
    ("--warmup-steps", "warmup_steps", "Warmup steps", "60", "int", "Linear warmup steps."),
    ("--eval-every-steps", "eval_every_steps", "Eval cadence", "5000", "int", "Validation cadence; 0 disables eval."),
    ("--eval-batches", "eval_batches", "Eval batches", "20", "int", "Validation batches per eval pass."),
    ("--train-log-every-steps", "train_log_every_steps", "Train loss cadence", "250", "int", "Sampled train-loss cadence; 0 disables."),
    ("--native-cuda-checkpoint-every", "native_cuda_checkpoint_every", "Checkpoint cadence", "5000", "int", "Native checkpoint cadence; 0 disables periodic checkpoints."),
    ("--progress-every-steps", "progress_every_steps", "Progress cadence", "1", "int", "Native progress print cadence."),
)

_TRAIN_TUI_EMBEDDING_HYPERPARAMS = (
    ("--embedding-stage", "embedding_stage", "Training stage", "pretrain", "embedding-stage", "From-scratch pretraining, supervised post-training, fine-tuning, or exact resume."),
    ("--embedding-architecture", "embedding_architecture", "Architecture", "bert", "embedding-architecture", "Bidirectional BERT-style or GPT-derived encoder profile."),
    ("--embedding-hf-model", "embedding_hf_model", "HF base model", "", "optional-text", "Optional local path or Hugging Face model ID whose transformer weights and tokenizer are imported without Torch."),
    ("--embedding-hf-revision", "embedding_hf_revision", "HF revision", "", "optional-text", "Optional Hugging Face branch, tag, or commit for the imported base."),
    ("--base-checkpoint", "base_checkpoint", "Base checkpoint", "", "text", "Native embedding checkpoint used as a weight-only warm start."),
    ("--resume-from-checkpoint", "resume_from_checkpoint", "Resume checkpoint", "", "text", "Native embedding checkpoint used to continue a prior run."),
    ("--pooling", "pooling", "Pooling", "mean", "pooling", "Sequence pooling: mean, cls, or last."),
    ("--embedding-vocab-size", "embedding_vocab_size", "Vocabulary buckets", "32768", "int", "Stable native tokenizer hash buckets; IDs are uint32."),
    ("--hidden-dim", "hidden_dim", "Encoder width", "128", "int", "Transformer hidden width; imported HF geometry overrides this value."),
    ("--num-layers", "num_layers", "Transformer layers", "2", "int", "Self-attention/MLP encoder blocks; imported HF geometry overrides this value."),
    ("--num-heads", "num_heads", "Attention heads", "4", "int", "Attention heads per transformer block."),
    ("--intermediate-dim", "intermediate_dim", "MLP width", "512", "int", "Feed-forward intermediate width."),
    ("--activation", "activation", "Activation", "gelu-tanh", "embedding-activation", "Exact GELU or GPT-2-compatible tanh GELU approximation."),
    ("--layer-norm-epsilon", "layer_norm_epsilon", "LayerNorm epsilon", "1e-5", "positive-float", "LayerNorm stability epsilon; imported HF geometry overrides this value."),
    ("--mask-token-id", "mask_token_id", "Mask token ID", "1", "int", "Token substituted by raw-text MLM; imported HF tokenizer metadata overrides this value."),
    ("--embedding-dim", "embedding_dim", "Embedding width", "128", "int", "Vector dimension returned by nfn embed."),
    ("--max-seq-len", "max_seq_len", "Text tokens", "128", "int", "Maximum tokens per text record."),
    ("--batch-size", "batch_size", "Microbatch records", "32", "int", "Records from one dataset/objective per native microbatch."),
    ("--effective-batch-size", "effective_batch_size", "Effective records", "256", "int", "Requested effective record batch size."),
    ("--learning-rate", "learning_rate", "Learning rate", "0.001", "float", "AdamW learning rate."),
    ("--weight-decay", "weight_decay", "Weight decay", "0.01", "float", "AdamW decoupled weight decay."),
    ("--warmup-steps", "warmup_steps", "Warmup steps", "50", "int", "Linear learning-rate warmup."),
    ("--max-steps", "max_steps", "Steps", "1000", "int", "Optimizer steps for this invocation."),
    ("--mlm-probability", "mlm_probability", "MLM probability", "0.15", "probability", "Masking probability used by raw-text pretraining."),
    ("--mlm-loss-weight", "mlm_loss_weight", "MLM weight", "1.0", "float", "Masked-token reconstruction loss weight."),
    ("--contrastive-loss-weight", "contrastive_loss_weight", "Contrastive weight", "1.0", "float", "Two-view/retrieval contrastive loss weight."),
    ("--temperature", "temperature", "Temperature", "0.05", "positive-float", "Contrastive softmax temperature."),
    ("--triplet-margin", "triplet_margin", "Triplet margin", "0.2", "float", "Cosine margin for retrieval and labeled batches."),
    ("--adapter-type", "adapter_type", "Fine-tuning mode", "none", "adapter", "Full-parameter, LoRA, or QLoRA training."),
    ("--lora-rank", "lora_rank", "LoRA rank", "16", "int", "Low-rank adapter rank."),
    ("--lora-alpha", "lora_alpha", "LoRA alpha", "32", "positive-float", "Low-rank adapter scale."),
    ("--lora-dropout", "lora_dropout", "LoRA dropout", "0.05", "probability", "Adapter input dropout."),
    ("--native-cuda-checkpoint-every", "native_cuda_checkpoint_every", "Checkpoint cadence", "250", "int", "Periodic native embedding checkpoints; 0 disables periodic writes."),
    ("--progress-every-steps", "progress_every_steps", "Progress cadence", "10", "int", "Native progress print cadence."),
)

_TRAIN_TUI_ARCH_HYPERPARAMS = (
    ("--num-layers", "num_layers", "Layers", "1", "int", "Transformer blocks in the semantic route stack."),
)

_TRAIN_TUI_SEMANTIC_MOE_HYPERPARAMS = (
    ("--semantic-vocab-dims", "semantic_vocab_dims", "Semantic dims", "86", "int", "Semantic vocabulary dimensions; one semantic expert is allocated per dimension."),
    ("--semantic-shared-experts", "semantic_shared_experts", "Shared experts", "2", "int", "Always-on experts prepended to every semantic-MoE route."),
    ("--semantic-free-experts", "semantic_free_experts", "Free experts", "8", "int", "Learned free experts appended after the semantic expert bank."),
    ("--layers-per-expert", "layers_per_expert", "Layers / expert", "1", "int", "Depth assigned to each routed expert domain."),
    ("--top-k", "top_k", "Route top-k", "2", "int", "Non-shared semantic/free experts selected per routed chunk."),
    ("--route-chunk-size", "route_chunk_size", "Route chunk", "32", "int", "Token interval for updating chunk-level semantic routes."),
)

_TRAIN_TUI_EVO_HYPERPARAMS = (
    ("--evo-layer-index", "evo_layer_index", "Evo layer", "6", "int", "Evo-capable template block or layer index to mutate."),
    ("--evo-layer-interval", "evo_layer_interval", "Evo cadence", "10", "int", "Optimizer-step cadence for candidate search."),
    ("--evo-layer-population", "evo_layer_population", "Evo population", "8", "int", "Candidate count for native layer-evo search."),
    ("--evo-layer-mutation-scale", "evo_layer_mutation_scale", "Evo mutation", "0.02", "float", "Gaussian mutation scale for evo candidates."),
    ("--evo-tournament-size", "evo_tournament_size", "Evo tournament", "3", "int", "Tournament pool size for evo selection metadata and compatible graph-evo paths."),
    ("--evo-elite-count", "evo_elite_count", "Evo elite", "1", "int", "Elite candidate count; native layer-evo always keeps the current weights as candidate 0."),
)
_TRAIN_TUI_GPT2_EVO_HYPERPARAMS = _TRAIN_TUI_EVO_HYPERPARAMS

_TRAIN_TUI_ALL_HYPERPARAMS = (
    *_TRAIN_TUI_HYPERPARAMS,
    *_TRAIN_TUI_EMBEDDING_HYPERPARAMS,
    *_TRAIN_TUI_ARCH_HYPERPARAMS,
    *_TRAIN_TUI_SEMANTIC_MOE_HYPERPARAMS,
    *_TRAIN_TUI_EVO_HYPERPARAMS,
)


def _train_tui_is_evo_selection(model: str, template: str = "") -> bool:
    normalized_model = str(model or "").strip().lower().replace("_", "-")
    normalized_template = str(template or "").strip().lower().replace("_", "-")
    if normalized_model == "gpt2-evo" or normalized_template == "gpt2-evo":
        return True
    return any(part for part in (normalized_model, normalized_template) if "evo" in part)


def _train_tui_is_semantic_moe_selection(model: str, template: str = "") -> bool:
    normalized_model = str(model or "").strip().lower().replace("_", "-")
    normalized_template = str(template or "").strip().lower().replace("_", "-")
    return any(
        part in {"semantic-moe-jepa-evo", "semantic-moe-jepa-evo-modern", "diff-semantic-moe-jepa-evo"}
        for part in (normalized_model, normalized_template)
    )


def _train_tui_hyperparams_for_model(model: str, template: str = ""):
    normalized_model = str(model or "").strip().lower().replace("_", "-")
    normalized_template = str(template or "").strip().lower().replace("_", "-")
    if normalized_model == "embedding":
        return _TRAIN_TUI_EMBEDDING_HYPERPARAMS
    params = list(_TRAIN_TUI_HYPERPARAMS)
    if normalized_model.startswith("semantic") or normalized_template.startswith("semantic"):
        params.extend(_TRAIN_TUI_ARCH_HYPERPARAMS)
    if _train_tui_is_semantic_moe_selection(normalized_model, normalized_template):
        params.extend(_TRAIN_TUI_SEMANTIC_MOE_HYPERPARAMS)
    if _train_tui_is_evo_selection(normalized_model, normalized_template):
        params.extend(_TRAIN_TUI_EVO_HYPERPARAMS)
    return tuple(params)


def _train_tui_hyperparams_for_state(state: dict[str, object]):
    return _train_tui_hyperparams_for_model(
        str(state.get("model") or "gpt"),
        str(state.get("template") or ""),
    )


def _ensure_train_tui_hyperparam_defaults(state: dict[str, object]) -> None:
    for _flag, key, _label, default, _kind, _description in _train_tui_hyperparams_for_state(state):
        state.setdefault(key, default)


def _has_native_train_action(args: list[str]) -> bool:
    return any(arg in _NATIVE_TRAIN_ACTION_FLAGS for arg in args)


def _is_native_train_tui_request(
    argv: list[str],
    *,
    stdin_isatty: bool | None = None,
    stdout_isatty: bool | None = None,
) -> bool:
    if not argv or argv[0] != "train":
        return False
    if _has_any(argv, "-h", "--help", "--no-tui", "--plan", "--plan-auto"):
        return False
    if _has_any(argv, "--tui", "--interactive"):
        return True
    if len(argv) == 1:
        input_tty = sys.stdin.isatty() if stdin_isatty is None else stdin_isatty
        output_tty = sys.stdout.isatty() if stdout_isatty is None else stdout_isatty
        return bool(input_tty and output_tty)
    return False


def _strip_train_tui_flags(argv: list[str]) -> list[str]:
    return [arg for arg in argv if arg not in {"--tui", "--interactive", "--no-tui"}]


def _discover_train_tui_datasets() -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = [
        ("tinystories", "TinyStoriesV2 GPT-4 alias"),
        ("roneneldan__TinyStories__TinyStoriesV2-GPT4", "TinyStories HF alias"),
        ("golf1", "Parameter Golf cached train1"),
        ("golf10", "Parameter Golf cached train10"),
    ]
    seen = {value for value, _ in choices}
    roots = [
        os.environ.get("NFN_DATASETS_DIR", ""),
        str(Path.home() / "NeuralFn" / "datasets"),
        "/mnt/disk2/dev/open-source/llm.kittens/dev/data/tinystories",
    ]
    for raw_root in roots:
        if not raw_root:
            continue
        root = Path(raw_root).expanduser()
        if not root.exists():
            continue
        candidates = [root]
        if root.is_dir():
            try:
                candidates.extend(sorted(path for path in root.iterdir() if path.is_dir() or path.suffix == ".bin")[:12])
            except OSError:
                pass
        for candidate in candidates:
            value = str(candidate)
            if value in seen:
                continue
            seen.add(value)
            choices.append((value, f"installed: {candidate}"))
    choices.append(("path", "Enter a dataset file or directory path"))
    return choices


def _train_tui_gpt_template_names() -> list[str]:
    command = _direct_native_train_cli_argv(["train", "--base-model", "gpt", "--list-templates"])
    env = os.environ.copy()
    _set_env_default_if_empty(env, "CUDA_VISIBLE_DEVICES", resolve_cuda_visible_devices_value("0"))
    _set_env_default_if_empty(env, "CUDA_DEVICE_MAX_CONNECTIONS", "1")
    _set_env_default_if_empty(env, "CUDA_MODULE_LOADING", "LAZY")
    try:
        proc = subprocess.run(
            command,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        proc = None
    if proc is not None and proc.returncode == 0:
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = {}
        templates = payload.get("templates")
        if isinstance(templates, list):
            names = [str(item.get("name", "")).strip() for item in templates if isinstance(item, dict)]
            names = [name for name in names if name]
            if names:
                return names
        shipped = payload.get("shipped_template_catalog")
        if isinstance(shipped, list):
            names = [str(name).strip() for name in shipped if str(name).strip()]
            if names:
                return names
    try:
        from neuralfn.config import SHIPPED_GPT_TEMPLATE_PRESETS

        return ["gpt", "gpt3", *list(SHIPPED_GPT_TEMPLATE_PRESETS)]
    except Exception:
        return list(_TRAIN_TUI_TEMPLATES["gpt"])


def _train_tui_template_choices(model: str) -> list[tuple[str, str]]:
    if model in _TRAIN_TUI_GPT_TEMPLATE_MODELS:
        preferred = list(_TRAIN_TUI_TEMPLATES.get(model, ()))
        all_names = _train_tui_gpt_template_names()
        ordered = [name for name in preferred if name in all_names]
        ordered.extend(name for name in all_names if name not in ordered)
        return [(value, "native GPT template catalog") for value in ordered]
    return [(value, "native template preset") for value in _TRAIN_TUI_TEMPLATES.get(model, ("default",))]


def _tui_log_default(output_dir: str, name: str) -> str:
    base = Path(output_dir).expanduser() if output_dir else Path.home() / "NeuralFn" / "artifacts"
    return str(base / f"{name}.log")


def _train_tui_option(label: str, description: str, value: object, *, recommended: bool = False):
    from nfn_impl import OptionChoice

    return OptionChoice(label, description, value, recommended=recommended)


def _train_tui_custom(label: str, description: str, prompt: str, *, parser=None):
    from nfn_impl import OptionChoice

    return OptionChoice(label, description, {}, custom_prompt=prompt, parser=parser)


def _train_tui_value_choices(default: str, label: str, description: str, prompt: str, *, parser=None):
    return [
        _train_tui_option(f"{label} ({default})", description, default, recommended=True),
        _train_tui_custom("Custom...", f"Enter a custom value for {label.lower()}.", prompt, parser=parser),
    ]


def _native_train_default_state() -> dict[str, object]:
    model = "gpt"
    output_dir = str(Path.home() / "NeuralFn" / "artifacts" / f"{model}_tui")
    state: dict[str, object] = {
        "model": model,
        "template": _train_tui_template_choices(model)[0][0],
        "dataset": "tinystories",
        "embedding_datasets": [],
        "embedding_datasets_manifest": "",
        "output_dir": output_dir,
        "train_log_file": _tui_log_default(output_dir, "train"),
        "eval_log_file": _tui_log_default(output_dir, "eval"),
        "launch_mode": "start",
    }
    for _flag, key, _label, default, _kind, _description in _TRAIN_TUI_HYPERPARAMS:
        state[key] = default
    return state


def _native_train_command_tokens_from_state(state: dict[str, object]) -> list[str]:
    model = str(state.get("model") or "gpt")
    template = str(state.get("template") or "default")
    dataset = str(state.get("dataset") or "tinystories")
    output_dir = str(state.get("output_dir") or Path.home() / "NeuralFn" / "artifacts" / f"{model}_tui")
    command_tokens = ["train", "--base-model", model]
    if model == "embedding":
        manifest = str(state.get("embedding_datasets_manifest") or "").strip()
        datasets = state.get("embedding_datasets") or []
        if manifest:
            command_tokens.extend(["--embedding-datasets-manifest", manifest])
        elif isinstance(datasets, list):
            for item in datasets:
                source = str(item).strip()
                if source:
                    command_tokens.extend(["--embedding-dataset", source])
        if not manifest and not any(token == "--embedding-dataset" for token in command_tokens):
            fallback_dataset = str(state.get("dataset") or "").strip()
            if fallback_dataset and fallback_dataset != "tinystories":
                command_tokens.extend(["--embedding-dataset", fallback_dataset])
    elif template and template != "default":
        command_tokens.extend(["--template-name", template])
    if model == "embedding":
        pass
    elif dataset == "tinystories":
        command_tokens.append("--tinystories")
    elif dataset in {"golf1", "golf10"}:
        command_tokens.extend(["--dataset", dataset])
    else:
        command_tokens.extend(["--dataset-alias", dataset])
    command_tokens.extend(["--output-dir", output_dir])
    for flag, key, _label, default, _kind, _description in _train_tui_hyperparams_for_state(state):
        value = str(state.get(key) or default or "")
        if value:
            command_tokens.extend([flag, value])
    train_log_file = str(state.get("train_log_file") or "")
    eval_log_file = str(state.get("eval_log_file") or "")
    if train_log_file:
        command_tokens.extend(["--train-log-file", train_log_file])
    if eval_log_file:
        command_tokens.extend(["--eval-log-file", eval_log_file])
    if str(state.get("launch_mode") or "") == "dry-run":
        command_tokens.extend(["--native-cuda-dry-run", "--native-cuda-print-command"])
    return command_tokens


def _native_train_tui_fields(state: dict[str, object] | None = None) -> list[tuple[str, str, str]]:
    embedding = str((state or {}).get("model") or "") == "embedding"
    fields = [("model", "Model", "run")]
    if embedding:
        fields.append(("dataset", "Datasets", "run"))
    else:
        fields.extend((("template", "Template", "run"), ("dataset", "Dataset", "run")))
    fields.extend((("output_dir", "Output", "run"), ("logs", "Logs", "run")))
    hyperparams = _train_tui_hyperparams_for_state(state or {}) if state is not None else _TRAIN_TUI_HYPERPARAMS
    fields.extend((key, label, "hyper") for _flag, key, label, _default, _kind, _description in hyperparams)
    return fields


def _native_train_validate_tui_value(key: str, value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("value cannot be empty")
    kind = next((kind for _flag, item_key, _label, _default, kind, _description in _TRAIN_TUI_ALL_HYPERPARAMS if item_key == key), "text")
    if kind == "int":
        parsed = int(value)
        if parsed < 0:
            raise ValueError("value must be non-negative")
        if key == "evo_tournament_size" and parsed <= 0:
            raise ValueError("tournament size must be at least 1")
        if key == "evo_elite_count" and parsed < 0:
            raise ValueError("elite count must be non-negative")
        return str(parsed)
    if kind == "float":
        parsed = float(value)
        if parsed < 0:
            raise ValueError("value must be non-negative")
        return f"{parsed:g}"
    if kind == "positive-float":
        parsed = float(value)
        if parsed <= 0:
            raise ValueError("value must be greater than zero")
        return f"{parsed:g}"
    if kind == "probability":
        parsed = float(value)
        if parsed < 0 or parsed > 1:
            raise ValueError("value must be between 0 and 1")
        return f"{parsed:g}"
    if kind == "embedding-stage":
        normalized = value.strip().lower().replace("_", "-")
        if normalized not in {"pretrain", "posttrain", "finetune", "resume"}:
            raise ValueError("value must be pretrain, posttrain, finetune, or resume")
        return normalized
    if kind == "embedding-architecture":
        normalized = value.strip().lower().replace("_", "-")
        if normalized in {"gpt", "gpt-derived"}:
            return "gpt-derived"
        if normalized != "bert":
            raise ValueError("value must be bert or gpt-derived")
        return normalized
    if kind == "embedding-activation":
        normalized = value.strip().lower().replace("_", "-")
        if normalized not in {"gelu", "gelu-tanh"}:
            raise ValueError("value must be gelu or gelu-tanh")
        return normalized
    if kind == "pooling":
        normalized = value.strip().lower()
        if normalized not in {"mean", "cls", "last"}:
            raise ValueError("value must be mean, cls, or last")
        return normalized
    if kind == "adapter":
        normalized = value.strip().lower()
        if normalized not in {"none", "lora", "qlora"}:
            raise ValueError("value must be none, lora, or qlora")
        return normalized
    if kind == "choice":
        normalized = value.strip().lower().replace("_", "-")
        if normalized in {"fixed", "constant"}:
            return "constant"
        if normalized in {"cosine", "cosine-decay"}:
            return "cosine"
        raise ValueError("value must be cosine or constant")
    return value


def _native_train_choice_value(
    *,
    prompt_fn,
    title: str,
    choices: list[tuple[str, str]],
    current: str,
) -> str | None:
    visible = list(choices)
    while True:
        print()
        print(f"\033[1m{title}\033[0m")
        for idx, (value, description) in enumerate(visible[:24], start=1):
            marker = "*" if value == current else " "
            print(f" {marker} {idx:>2}. {value}  \033[2m{description}\033[0m")
        if len(visible) > 24:
            print(f"    ... {len(visible) - 24} more; type /text to filter")
        raw = prompt_fn("index, exact value, /filter, or blank to keep").strip()
        if not raw:
            return None
        if raw.startswith("/"):
            needle = raw[1:].strip().lower()
            visible = [choice for choice in choices if needle in choice[0].lower() or needle in choice[1].lower()]
            if not visible:
                print("No matches.")
                visible = list(choices)
            continue
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(visible):
                return visible[idx][0]
            print("Index out of range.")
            continue
        for value, _description in choices:
            if raw == value:
                return value
        print("Unknown value.")


def _native_train_tui_read_key(fd: int) -> str:
    import select

    ch = os.read(fd, 1).decode("utf-8", errors="ignore")
    if ch != "\x1b":
        return ch
    seq = ch
    while True:
        ready, _w, _x = select.select([fd], [], [], 0.01)
        if not ready:
            break
        seq += os.read(fd, 1).decode("utf-8", errors="ignore")
        if len(seq) >= 6:
            break
    return {
        "\x1b[A": "up",
        "\x1b[B": "down",
        "\x1b[C": "right",
        "\x1b[D": "left",
    }.get(seq, "escape")


def _render_native_train_dashboard(console, state: dict[str, object], selected: int, status: str) -> None:
    from rich.box import ROUNDED
    from rich.markup import escape as rich_escape
    from rich.table import Table

    console.file.write("\x1b[H")
    _ensure_train_tui_hyperparam_defaults(state)
    fields = _native_train_tui_fields(state)
    selected_key = fields[selected][0]
    model = rich_escape(str(state.get("model") or "gpt"))
    template = rich_escape(str(state.get("template") or "default"))
    if str(state.get("model") or "") == "embedding":
        manifest = str(state.get("embedding_datasets_manifest") or "").strip()
        sources = state.get("embedding_datasets") or []
        dataset_text = manifest or f"{len(sources) if isinstance(sources, list) else 0} sources"
    else:
        dataset_text = str(state.get("dataset") or "tinystories")
    dataset = rich_escape(dataset_text)
    steps = rich_escape(str(state.get("max_steps") or "20000"))
    console.print(f"[infer.banner] NeuralFn Native Train [/] [infer.accent]{model}[/] template={template} dataset={dataset} steps={steps}")
    console.print("[infer.status]Up/Down move  Enter edit  r run  p print command  q quit[/]")

    active_hyperparams = _train_tui_hyperparams_for_state(state)
    defaults = {key: default for _flag, key, _label, default, _kind, _description in active_hyperparams}
    descriptions = {key: description for _flag, key, _label, _default, _kind, description in active_hyperparams}
    show_meaning = console.width >= 112
    setup_table = Table(box=ROUNDED, expand=True, show_lines=False)
    setup_table.add_column("Setting", style="infer.accent", no_wrap=True, width=18)
    setup_table.add_column("Value", overflow="ellipsis", no_wrap=True, width=36)
    setup_table.add_column("Default", overflow="ellipsis", no_wrap=True, width=14)
    if show_meaning:
        setup_table.add_column("Meaning", overflow="fold")
    for key, label, group in fields:
        current = str(state.get(key) or "off")
        default = defaults.get(key, "")
        if group == "run":
            if key == "model":
                default = "gpt"
            elif key == "template":
                default = _train_tui_template_choices(str(state.get("model") or "gpt"))[0][0]
            elif key == "dataset":
                if str(state.get("model") or "") == "embedding":
                    manifest = str(state.get("embedding_datasets_manifest") or "").strip()
                    sources = state.get("embedding_datasets") or []
                    current = manifest or ", ".join(str(item) for item in sources) or "not configured"
                    default = "one or more sources"
                else:
                    default = "tinystories"
            elif key == "output_dir":
                default = str(Path.home() / "NeuralFn" / "artifacts" / f"{state.get('model') or 'gpt'}_tui")
            elif key == "logs":
                train_log = str(state.get("train_log_file") or "off")
                eval_log = str(state.get("eval_log_file") or "off")
                current = f"train={train_log}  eval={eval_log}"
                default = "default train/eval logs"
        changed = bool(default) and current != default
        style = "reverse bright_yellow" if key == selected_key else ("bright_green" if changed else "")
        row = [
            label,
            rich_escape(current),
            rich_escape(default),
        ]
        if show_meaning:
            row.append(rich_escape(descriptions.get(key, "Edit the run configuration.")))
        setup_table.add_row(*row, style=style)
    console.print(setup_table)

    selected_description = descriptions.get(selected_key, "Edit the run configuration.")
    console.print(f"[infer.status]{rich_escape(selected_description)}  Press [infer.accent]p[/] for the full command.[/]")
    if status:
        console.print(f":sparkles: [infer.preview]{rich_escape(status)}[/]")
    console.file.write("\x1b[J")
    console.file.flush()


def _edit_native_train_tui_field(console, state: dict[str, object], key: str, old_term_attrs) -> str:
    import termios
    import tty as tty_module

    fd = sys.stdin.fileno()

    def prompt(label: str) -> str:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term_attrs)
        try:
            return input(f"{label}: ")
        finally:
            tty_module.setcbreak(fd)

    if key == "model":
        choices = [(value, description) for value, description in _TRAIN_TUI_MODELS]
        selected = _native_train_choice_value(
            prompt_fn=prompt,
            title="Choose model family",
            choices=choices,
            current=str(state.get("model") or "gpt"),
        )
        if selected is None:
            return "Model unchanged."
        state["model"] = selected
        state["template"] = _train_tui_template_choices(selected)[0][0]
        output_dir = str(Path.home() / "NeuralFn" / "artifacts" / f"{selected}_tui")
        state["output_dir"] = output_dir
        state["train_log_file"] = _tui_log_default(output_dir, "train")
        state["eval_log_file"] = _tui_log_default(output_dir, "eval")
        _ensure_train_tui_hyperparam_defaults(state)
        return f"Model set to {selected}; template and default paths refreshed."
    if key == "template":
        model = str(state.get("model") or "gpt")
        selected = _native_train_choice_value(
            prompt_fn=prompt,
            title=f"Choose template for {model}",
            choices=_train_tui_template_choices(model),
            current=str(state.get("template") or "default"),
        )
        if selected is None:
            return "Template unchanged."
        state["template"] = selected
        _ensure_train_tui_hyperparam_defaults(state)
        return f"Template set to {selected}."
    if key == "dataset":
        if str(state.get("model") or "") == "embedding":
            raw = prompt("Dataset manifest path, or comma-separated dataset paths (blank keeps current)").strip()
            if not raw:
                return "Embedding datasets unchanged."
            candidate = Path(raw).expanduser()
            if "," not in raw and candidate.suffix.lower() == ".json":
                state["embedding_datasets_manifest"] = raw
                state["embedding_datasets"] = []
                return f"Embedding dataset manifest set to {raw}."
            sources = [item.strip() for item in raw.split(",") if item.strip()]
            if not sources:
                return "Embedding datasets unchanged."
            state["embedding_datasets_manifest"] = ""
            state["embedding_datasets"] = sources
            return f"Embedding dataset array set to {len(sources)} source(s)."
        selected = _native_train_choice_value(
            prompt_fn=prompt,
            title="Choose dataset",
            choices=_discover_train_tui_datasets(),
            current=str(state.get("dataset") or "tinystories"),
        )
        if selected is None:
            return "Dataset unchanged."
        if selected == "path":
            selected = prompt("Dataset file or directory path").strip()
            if not selected:
                return "Dataset unchanged."
        state["dataset"] = selected
        return f"Dataset set to {selected}."
    if key == "output_dir":
        raw = prompt("Output directory (blank keeps current)").strip()
        if not raw:
            return "Output directory unchanged."
        state["output_dir"] = raw
        state["train_log_file"] = _tui_log_default(raw, "train")
        state["eval_log_file"] = _tui_log_default(raw, "eval")
        return "Output directory updated; default log paths refreshed."
    if key == "logs":
        train_raw = prompt("Train log path, 'off', 'default', or blank keeps current").strip()
        eval_raw = prompt("Eval log path, 'off', 'default', or blank keeps current").strip()
        updates = 0
        for item_key, name, raw in (
            ("train_log_file", "train", train_raw),
            ("eval_log_file", "eval", eval_raw),
        ):
            if not raw:
                continue
            if raw.lower() == "off":
                state[item_key] = ""
            elif raw.lower() == "default":
                state[item_key] = _tui_log_default(str(state.get("output_dir") or ""), name)
            else:
                state[item_key] = raw
            updates += 1
        return "Logs unchanged." if updates == 0 else "Logs updated."
    current = str(state.get(key) or "")
    label = next((label for _flag, item_key, label, _default, _kind, _description in _TRAIN_TUI_ALL_HYPERPARAMS if item_key == key), key)
    while True:
        raw = prompt(f"{label} (current {current}, blank keeps current)").strip()
        if not raw:
            return f"{label} unchanged."
        try:
            state[key] = _native_train_validate_tui_value(key, raw)
        except ValueError as exc:
            print(f"Invalid value: {exc}")
            continue
        return f"{label} set to {state[key]}."


def _native_train_dashboard_tui_main(tokens: list[str]) -> int:
    try:
        from rich.console import Console
        from rich.theme import Theme
    except ImportError:
        from nfn_impl import run_curses_questionnaire

        state = run_curses_questionnaire("nfn train", _native_train_tui_questions(), {})
        command_tokens = _native_train_command_tokens_from_state(state)
        print()
        print("\033[1mResolved nfn train command\033[0m")
        print(shlex.join(["nfn", *command_tokens]))
        return _direct_native_train_cli_main(command_tokens, progress_tui=True)

    import termios
    import tty as tty_module

    theme = Theme(
        {
            "infer.user": "bold bright_cyan",
            "infer.assistant": "bold bright_magenta",
            "infer.system": "dim italic",
            "infer.banner": "bold white on #2a004d",
            "infer.accent": "bright_yellow",
            "infer.status": "dim",
            "infer.preview": "italic bright_green",
            "infer.ghost": "italic #808080",
            "infer.error": "bold red",
        }
    )
    console = Console(theme=theme, emoji=True, highlight=False, soft_wrap=False)
    state = _native_train_default_state()
    fields = _native_train_tui_fields(state)
    selected = 0
    status = "Ready. Move to any row and press Enter to edit; defaults are already applied."
    fd = sys.stdin.fileno()
    old_term_attrs = termios.tcgetattr(fd)
    try:
        console.clear()
        tty_module.setcbreak(fd)
        while True:
            _render_native_train_dashboard(console, state, selected, status)
            key = _native_train_tui_read_key(fd)
            if key in {"up", "k"}:
                fields = _native_train_tui_fields(state)
                selected = (selected - 1) % len(fields)
                status = ""
            elif key in {"down", "j"}:
                fields = _native_train_tui_fields(state)
                selected = (selected + 1) % len(fields)
                status = ""
            elif key in {"\r", "\n", "e"}:
                fields = _native_train_tui_fields(state)
                selected = min(selected, len(fields) - 1)
                status = _edit_native_train_tui_field(console, state, fields[selected][0], old_term_attrs)
                fields = _native_train_tui_fields(state)
                selected = min(selected, len(fields) - 1)
            elif key == "m":
                selected = next((index for index, field in enumerate(fields) if field[0] == "model"), 0)
                status = _edit_native_train_tui_field(console, state, "model", old_term_attrs)
                fields = _native_train_tui_fields(state)
            elif key == "t":
                if str(state.get("model") or "") == "embedding":
                    status = "Embedding models use Architecture instead of an LM template."
                else:
                    selected = next((index for index, field in enumerate(fields) if field[0] == "template"), 0)
                    status = _edit_native_train_tui_field(console, state, "template", old_term_attrs)
                fields = _native_train_tui_fields(state)
            elif key == "d":
                selected = next((index for index, field in enumerate(fields) if field[0] == "dataset"), 0)
                status = _edit_native_train_tui_field(console, state, "dataset", old_term_attrs)
                fields = _native_train_tui_fields(state)
            elif key == "o":
                selected = next((index for index, field in enumerate(fields) if field[0] == "output_dir"), 0)
                status = _edit_native_train_tui_field(console, state, "output_dir", old_term_attrs)
                fields = _native_train_tui_fields(state)
            elif key == "r":
                state["launch_mode"] = "start"
                break
            elif key == "p":
                state["launch_mode"] = "dry-run"
                break
            elif key in {"q", "escape", "\x03"}:
                console.clear()
                print("Training setup cancelled.")
                return 0
            else:
                status = "Use Up/Down, Enter, r, p, or q. Shortcuts: m model, t template, d dataset, o output."
    except KeyboardInterrupt:
        console.clear()
        print("Training setup cancelled.", file=sys.stderr)
        return 130
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term_attrs)
    console.clear()
    command_tokens = _native_train_command_tokens_from_state(state)
    print()
    print("\033[1mResolved nfn train command\033[0m")
    print(shlex.join(["nfn", *command_tokens]))
    return _direct_native_train_cli_main(command_tokens, progress_tui=True)


def _native_train_tui_questions():
    from nfn_impl import Question

    always = lambda _state, _explicit: True
    embedding_only = lambda state, _explicit: str(state.get("model") or "") == "embedding"
    non_embedding = lambda state, _explicit: str(state.get("model") or "") != "embedding"

    def model_options(_state):
        return [
            _train_tui_option(label, description, value, recommended=value == "gpt")
            for value, description in _TRAIN_TUI_MODELS
            for label in [value.upper() if value in {"gpt", "gpt2", "gpt3"} else value]
        ]

    def template_options(state):
        model = str(state.get("model") or "gpt")
        choices = _train_tui_template_choices(model)
        return [
            _train_tui_option(value, description, value, recommended=idx == 0)
            for idx, (value, description) in enumerate(choices)
        ]

    def dataset_options(state):
        if str(state.get("model") or "") == "embedding":
            return [
                _train_tui_custom(
                    "Dataset path...",
                    "Enter a TXT, JSONL, JSON, CSV, or Parquet source; use the dashboard for an array/manifest.",
                    "Embedding dataset path",
                )
            ]
        options = []
        for idx, (value, description) in enumerate(_discover_train_tui_datasets()):
            if value == "path":
                options.append(
                    _train_tui_custom(
                        "Custom path...",
                        "Enter a dataset file or directory path.",
                        "Dataset file or directory path",
                    )
                )
            else:
                options.append(_train_tui_option(value, description, value, recommended=idx == 0))
        return options

    def output_options(state):
        model = str(state.get("model") or "gpt")
        default = str(Path.home() / "NeuralFn" / "artifacts" / f"{model}_tui")
        return [
            _train_tui_option(f"Default ({default})", "Use the standard NeuralFn artifact directory.", default, recommended=True),
            _train_tui_custom("Custom path...", "Enter an output directory.", "Output directory"),
        ]

    def train_log_options(state):
        default = _tui_log_default(str(state.get("output_dir") or ""), "train")
        return [
            _train_tui_option(f"Default ({default})", "Capture native progress stderr to this file.", default, recommended=True),
            _train_tui_option("Off", "Do not write a train progress log.", ""),
            _train_tui_custom("Custom path...", "Enter a train progress log path.", "Train log path"),
        ]

    def eval_log_options(state):
        default = _tui_log_default(str(state.get("output_dir") or ""), "eval")
        return [
            _train_tui_option(f"Default ({default})", "Capture validation lines and final JSON to this file.", default, recommended=True),
            _train_tui_option("Off", "Do not write an eval/final JSON log.", ""),
            _train_tui_custom("Custom path...", "Enter an eval/final JSON log path.", "Eval log path"),
        ]

    def launch_options(_state):
        return [
            _train_tui_option("Start training", "Launch the resolved native trainer.", "start", recommended=True),
            _train_tui_option("Print command only", "Show the native command without starting training.", "dry-run"),
            _train_tui_option("Cancel", "Exit without launching a trainer.", "cancel"),
        ]

    questions = [
        Question("model", "Choose a model family.", model_options, always),
        Question("template", "Choose a model template.", template_options, non_embedding),
        Question("dataset", "Choose a dataset alias or path.", dataset_options, always),
        Question("output_dir", "Choose an output directory.", output_options, always),
    ]
    for params, visible in (
        (_TRAIN_TUI_HYPERPARAMS, non_embedding),
        (_TRAIN_TUI_EMBEDDING_HYPERPARAMS, embedding_only),
    ):
        for _flag, key, label, default, _kind, _description in params:
            questions.append(
                Question(
                    key,
                    f"Set {label.lower()}.",
                    lambda _state, d=default, l=label: _train_tui_value_choices(d, l, "Use the recommended native-training value.", l),
                    visible,
                )
            )
    questions.extend(
        [
            Question("train_log_file", "Choose a train progress log.", train_log_options, always),
            Question("eval_log_file", "Choose an eval/final JSON log.", eval_log_options, always),
            Question("launch_mode", "Review and launch.", launch_options, always),
        ]
    )
    return questions


def _native_train_tui_main(argv: list[str] | None = None) -> int:
    tokens = _strip_train_tui_flags(list(sys.argv[1:] if argv is None else argv))
    if sys.stdin.isatty() and sys.stdout.isatty():
        return _native_train_dashboard_tui_main(tokens)
    from nfn_impl import run_curses_questionnaire

    try:
        state = run_curses_questionnaire("nfn train", _native_train_tui_questions(), {})
    except KeyboardInterrupt:
        print("Training setup cancelled.", file=sys.stderr)
        return 130
    if str(state.get("launch_mode") or "") == "cancel":
        print("Training setup cancelled.")
        return 0
    command_tokens = _native_train_command_tokens_from_state(state)

    print()
    print("\033[1mResolved nfn train command\033[0m")
    print(shlex.join(["nfn", *command_tokens]))
    return _direct_native_train_cli_main(command_tokens, progress_tui=True)


def _native_template_name(argv: list[str]) -> str:
    return (_arg_value(argv, "--template-name", "--template", "--preset") or "gpt").strip().lower().replace("-", "_")


def _native_template_family(argv: list[str]) -> str | None:
    template = _native_template_name(argv).replace("_", "-")
    return _NATIVE_TEMPLATE_FAMILY_ALIASES.get(template)


def _has_native_activation(argv: list[str]) -> bool:
    return any(
        arg in {"--activation", "--native-cuda-activation"} or
        arg.startswith("--activation=") or
        arg.startswith("--native-cuda-activation=")
        for arg in argv
    )


def _native_env_default(names: tuple[str, ...]) -> str:
    *env_names, fallback = names
    for name in env_names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value)
    return fallback


def _append_native_gpt_quality_defaults(out: list[str]) -> None:
    for flag, env_names in _NATIVE_GPT_QUALITY_DEFAULTS.items():
        explicit_flags = (flag,)
        if flag == "--lr-schedule":
            explicit_flags += ("--learning-rate-schedule",)
        elif flag == "--final-lr-fraction":
            explicit_flags += (
                "--learning-rate-decay-frac",
                "--learning-rate-decay-fraction",
            )
        elif flag == "--train-loss-every-steps":
            explicit_flags += ("--train-log-every", "--train-log-every-steps")
        if not _explicit_arg(out, *explicit_flags):
            _append_value_arg(out, flag, _native_env_default(env_names))
    if not _has_native_activation(out):
        activation_default = "moa" if _native_template_name(out) == "gpt2_moa" else "gelu"
        _append_value_arg(
            out,
            "--native-cuda-activation",
            os.environ.get("NFN_NATIVE_GPT_ACTIVATION")
            or os.environ.get("NFN_SM120_ACTIVATION")
            or activation_default,
        )


def _native_semantic_moe_selection(argv: list[str], model: str | None = None) -> bool:
    normalized_model = (model or _native_train_model(argv)).strip().lower().replace("_", "-")
    normalized_template = _native_template_name(argv).replace("_", "-")
    return any(
        part in {"semantic-moe-jepa-evo", "semantic-moe-jepa-evo-modern", "diff-semantic-moe-jepa-evo"}
        for part in (normalized_model, normalized_template)
    )


def _append_native_semantic_moe_defaults(out: list[str]) -> None:
    defaults = {
        "--semantic-vocab-dims": "86",
        "--semantic-shared-experts": "2",
        "--semantic-free-experts": "8",
        "--layers-per-expert": "1",
        "--top-k": "2",
        "--route-chunk-size": "32",
    }
    for flag, value in defaults.items():
        if not _explicit_arg(out, flag):
            _append_value_arg(out, flag, value)


def _direct_native_train_cli_argv(argv: list[str]) -> list[str]:
    model = _native_train_model(argv)
    token_lm_requested = any(arg == "--train-token-lm" for arg in argv)
    dense_gpt = _is_dense_gpt_native_model(model) and not (
        model in {"nanogpt", "nano-gpt"} and token_lm_requested
    )
    template_family = (
        None
        if _explicit_arg(argv, "--graph-file", "--graph")
        else _native_template_family(argv)
    )
    if dense_gpt and template_family is not None:
        resolved_model = template_family
        dense_gpt = False
    else:
        resolved_model = model
    family_cli = None if dense_gpt else _resolve_direct_native_train_family_cli(resolved_model)
    native_cli = family_cli or _resolve_direct_native_train_cli("gpt" if dense_gpt else resolved_model)
    out = [native_cli]
    tile_ops_lib_explicit = _explicit_arg(argv, "--tile-ops-lib", "--native-cuda-tile-ops-lib")
    include_model = not dense_gpt and family_cli is None
    if include_model:
        out.extend(["--base-model", resolved_model])
    elif dense_gpt:
        model_family = _canonical_dense_gpt_model_family(model)
        out.extend(["--model-family", model_family])
        if model_family == "nanogpt" and not _explicit_arg(argv, "--template-name", "--template", "--preset", "--graph-file", "--graph"):
            out.extend(["--template-name", "nanogpt"])
    if _native_semantic_moe_selection(argv, model) and not _explicit_arg(argv, "--template-name", "--template", "--preset"):
        out.extend(["--template-name", model])
    if (
        not dense_gpt
        and not _explicit_arg(out, "--template-name", "--template", "--preset")
        and not _explicit_arg(out, "--graph-file", "--graph")
    ):
        model_template_family = _NATIVE_TEMPLATE_FAMILY_ALIASES.get(model)
        if model_template_family is not None and model_template_family != model:
            out.extend(["--template-name", model])
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
        "--train-log-file",
        "--eval-log-file",
    }
    drop_bool_flags = {
        "--no-tile-cuda-strict",
        "--tile-cuda-strict",
        "--no-tui",
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
        "--learning-rate-schedule": "--lr-schedule",
        "--learning-rate-decay-frac": "--final-lr-fraction",
        "--learning-rate-decay-fraction": "--final-lr-fraction",
        "--train-log-every": "--train-loss-every-steps",
        "--train-log-every-steps": "--train-loss-every-steps",
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
        "--native-cuda-smoke-llama-loop": "--smoke-llama-loop",
        "--native-cuda-smoke-llama-lm-head-step": "--smoke-llama-lm-head-step",
        "--native-cuda-smoke-llama-train-step": "--smoke-llama-train-step",
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
        "--native-cuda-fast-startup": "--fast-startup",
        "--fast-startup": "--fast-startup",
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
        "--progress-every-steps",
        "--lm-head-row-chunk-size",
        "--batch-size",
        "--train-seq-len",
        "--train-batch-tokens",
        "--learning-rate",
        "--lr-schedule",
        "--learning-rate-schedule",
        "--lr-schedule-total-steps",
        "--train-seed",
        "--resume-from-checkpoint",
        "--native-cuda-resume-from-checkpoint",
        "--final-lr-fraction",
        "--learning-rate-decay-frac",
        "--learning-rate-decay-fraction",
        "--weight-decay",
        "--beta1",
        "--beta2",
        "--adam-eps",
        "--grad-clip-norm",
        "--warmup-steps",
        "--max-steps",
        "--num-layers",
        "--semantic-vocab-dims",
        "--semantic-shared-experts",
        "--semantic-free-experts",
        "--layers-per-expert",
        "--expert-layers",
        "--expert-depth",
        "--experts",
        "--top-k",
        "--route-chunk-size",
        "--template-name",
        "--template",
        "--preset",
        "--graph-file",
        "--graph",
        "--graph-fingerprint",
        "--graph-preflight-proof",
        "--native-cuda-checkpoint-every",
        "--native-cuda-sample-every",
        "--native-cuda-generate-tokens",
        "--cuda-runtime-lib",
        "--activation",
        "--native-cuda-activation",
        "--moa-interval",
        "--native-cuda-moa-interval",
        "--experts",
        "--native-cuda-experts",
        "--top-k",
        "--native-cuda-top-k",
        "--layers-per-expert",
        "--native-cuda-layers-per-expert",
        "--router-aux-loss-coef",
        "--native-cuda-router-aux-loss-coef",
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
    explicit_tile_ops_value = _arg_value(out, "--tile-ops-lib")
    if (
        explicit_tile_ops_value == "linked"
        and not (dense_gpt and _native_gpt_cli_uses_linked_tile_ops(out[0]))
    ):
        raise ValueError(
            "--tile-ops-lib linked is only valid for the linked dense-GPT trainer; "
            "omit the flag so the family trainer resolves its Tile library, or pass the real .so path"
        )
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
    if dense_gpt and model == "gpt3" and not _explicit_arg(out, "--batch-size"):
        _append_value_arg(out, "--batch-size", "32")
    if dense_gpt and not any(flag in out for flag in _NATIVE_GPT_METADATA_ACTION_FLAGS):
        _append_native_gpt_quality_defaults(out)
    if _native_semantic_moe_selection(out, model):
        _append_native_semantic_moe_defaults(out)
    if dense_gpt and not _explicit_arg(out, "--backend"):
        _append_value_arg(out, "--backend", "tile-cuda")
    if dense_gpt and _native_gpt_cli_uses_linked_tile_ops(out[0]) and not tile_ops_lib_explicit:
        _append_value_arg(out, "--tile-ops-lib", "linked")
    return out


def _native_train_log_paths(tokens: list[str]) -> tuple[str | None, str | None]:
    return (
        _arg_value(tokens, "--train-log-file"),
        _arg_value(tokens, "--eval-log-file"),
    )


def _open_optional_log(path: str | None):
    if not path:
        return None
    log_path = Path(path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path.open("a", encoding="utf-8")


def _write_log(handle, text: str) -> None:
    if handle is not None:
        handle.write(text)
        handle.flush()


def _native_train_artifact_rows(command: list[str]) -> list[tuple[str, str]]:
    def describe(label: str, raw_path: str | None) -> tuple[str, str] | None:
        if not raw_path:
            return None
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (ROOT.parent / path).resolve()
        if not path.is_file():
            return (label, f"{path} (missing)")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return (label, f"{path} sha256={digest}")

    rows: list[tuple[str, str]] = []
    executable = describe("executable", command[0] if command else None)
    if executable is not None:
        rows.append(executable)
    tile_value = _arg_value(command, "--tile-ops-lib")
    linked_dense_gpt = bool(command) and _native_gpt_cli_uses_linked_tile_ops(command[0])
    if (tile_value == "linked" and linked_dense_gpt) or (
        tile_value is None
        and command
        and Path(command[0]).name in {"nfn_gpt_native_train", "nfn_gpt_native_train_linked"}
    ):
        tile_value = str(ROOT.parent / "build" / "libnfn_native_train_tile_ops.so")
    tile_ops = describe("tile ops", tile_value)
    if tile_ops is not None:
        rows.append(tile_ops)
    return rows


_NATIVE_PROGRESS_STEP_RE = re.compile(r"\bstep\s+(\d+)(?:/(\d+))?")
_NATIVE_VALIDATION_RE = re.compile(r"\bvalidation\b|\beval\b", re.IGNORECASE)
_NATIVE_METRIC_FIELD_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)=([^\s]+)")
_NATIVE_METRIC_LABELS = {
    "tokens_per_s": "tokens/s",
    "elapsed_s": "elapsed",
    "train_loss": "train loss",
    "microbatch_tokens": "microbatch tokens",
    "tokens_per_microbatch": "tokens/microbatch",
    "grad_accum_steps": "gradient accumulation",
    "effective_train_batch_tokens": "effective batch tokens",
    "train_microbatches_completed": "microbatches completed",
    "optimizer_step": "optimizer step",
    "eval_due": "evaluation due",
    "train_loss_due": "train-loss sample due",
}


def _format_native_metric_value(key: str, value: str) -> str:
    if key in {
        "tokens",
        "rows",
        "tokens_per_microbatch",
        "microbatch_tokens",
        "effective_train_batch_tokens",
        "train_microbatches_completed",
    }:
        try:
            return f"{int(value):,}"
        except ValueError:
            return value
    if key == "tokens_per_s":
        try:
            return f"{float(value):,.0f}"
        except ValueError:
            return value
    if key == "elapsed_s":
        try:
            return f"{float(value):,.2f}s"
        except ValueError:
            return value
    return value


def _format_native_train_line(line: str) -> str:
    text = line.rstrip("\n")
    native_prefix = "[nfn-native-train] "
    if text.startswith(native_prefix):
        text = text[len(native_prefix):]
    fields = list(_NATIVE_METRIC_FIELD_RE.finditer(text))
    if not fields:
        return text
    prefix = text[:fields[0].start()].strip()
    rendered = [prefix] if prefix else []
    cursor = fields[0].start()
    for match in fields:
        interstitial = text[cursor:match.start()].strip()
        if interstitial:
            rendered.append(interstitial)
        rendered.append(
            f"{_NATIVE_METRIC_LABELS.get(match.group(1), match.group(1).replace('_', ' '))}: "
            f"{_format_native_metric_value(match.group(1), match.group(2))}"
        )
        cursor = match.end()
    tail = text[cursor:].strip()
    if tail:
        rendered.append(tail)
    return " | ".join(rendered)


def _native_train_metric_rows(
    value: object,
    *,
    prefix: str = "",
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(value, dict):
        if not value:
            rows.append((prefix or "value", "{}"))
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_native_train_metric_rows(child, prefix=child_prefix))
        return rows
    if isinstance(value, list):
        if not value:
            rows.append((prefix or "value", "[]"))
        for index, child in enumerate(value):
            rows.extend(_native_train_metric_rows(child, prefix=f"{prefix}[{index}]"))
        return rows
    if value is None:
        rendered = "null"
    elif isinstance(value, bool):
        rendered = str(value).lower()
    else:
        rendered = str(value)
    rows.append((prefix or "value", rendered))
    return rows


def _print_native_train_metrics(payload: dict[str, object], *, stream=None) -> None:
    target = stream or sys.stdout
    print("\n\033[1;36mNeuralFn Training Metrics\033[0m", file=target)
    for key, value in _native_train_metric_rows(payload):
        print(f"  \033[1m{key}\033[0m: {value}", file=target)


def _native_train_parse_stdout_json(stdout_text: str) -> dict[str, object] | None:
    text = stdout_text.strip()
    if not text or not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _native_train_last_validation_loss(payload: dict[str, object]) -> str | None:
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        return None
    losses = validation.get("losses")
    if not isinstance(losses, list) or not losses:
        eval_count = validation.get("eval_count")
        return f"eval_count={eval_count}" if eval_count is not None else None
    last = losses[-1]
    if not isinstance(last, dict):
        return None
    loss = last.get("loss_mean")
    step = last.get("step")
    if loss is None:
        return None
    if step is None:
        return str(loss)
    return f"{loss} at step {step}"


def _native_train_checkpoint_field(payload: dict[str, object], field: str) -> object | None:
    checkpoint = payload.get("checkpoint")
    if isinstance(checkpoint, dict) and field in checkpoint:
        return checkpoint.get(field)
    return payload.get(field) or payload.get(f"model_{field}")


def _format_native_train_int(value: object) -> str | None:
    if value is None:
        return None
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        text = str(value)
        return text if text else None


def _native_train_summary_rows(
    payload: dict[str, object],
    *,
    return_code: int,
    elapsed_seconds: float,
    train_log_file: str | None,
    eval_log_file: str | None,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("exit code", str(return_code)),
        ("elapsed", f"{elapsed_seconds:0.1f}s"),
    ]
    status = payload.get("status")
    if status is not None:
        rows.append(("status", str(status)))
    passed = payload.get("passed")
    if passed is not None:
        rows.append(("passed", str(bool(passed)).lower()))
    steps = payload.get("steps_completed")
    if steps is not None:
        rows.append(("steps", str(steps)))
    validation_loss = _native_train_last_validation_loss(payload)
    if validation_loss is not None:
        rows.append(("validation", validation_loss))
    timing = payload.get("timing")
    tokens_per_second = None
    if isinstance(timing, dict):
        tokens_per_second = timing.get("train_tokens_per_second") or timing.get("setup_amortized_train_tokens_per_second")
    tokens_per_second = tokens_per_second or payload.get("setup_amortized_train_tokens_per_second")
    formatted_tps = _format_native_train_int(tokens_per_second)
    if formatted_tps is not None:
        rows.append(("tokens/s", formatted_tps))
    checkpoint_path = _native_train_checkpoint_field(payload, "checkpoint_path")
    if checkpoint_path:
        rows.append(("model", str(checkpoint_path)))
    else:
        rows.append(("model", "not written"))
    done_marker = _native_train_checkpoint_field(payload, "done_marker")
    if done_marker:
        rows.append(("done marker", str(done_marker)))
    checkpoint_size = _native_train_checkpoint_field(payload, "actual_file_size")
    formatted_size = _format_native_train_int(checkpoint_size)
    if formatted_size is not None:
        rows.append(("model bytes", formatted_size))
    if train_log_file:
        rows.append(("train log", train_log_file))
    if eval_log_file:
        rows.append(("eval log", f"{eval_log_file} (full trainer JSON)"))
    return rows


def _print_train_tui_panel(title: str, rows: list[tuple[str, str]], *, stream=None) -> None:
    target = stream or sys.stderr
    width = 88
    print("\033[1;36m+" + "-" * (width - 2) + "+\033[0m", file=target)
    print(f"\033[1;36m| {title[:width - 5]:<{width - 4}}|\033[0m", file=target)
    print("\033[1;36m+" + "-" * (width - 2) + "+\033[0m", file=target)
    for key, value in rows:
        text = f"{key}: {value}"
        print(f"| {text[:width - 5]:<{width - 4}}|", file=target)
    print("\033[1;36m+" + "-" * (width - 2) + "+\033[0m", file=target)


def _run_native_train_with_progress(
    command: list[str],
    env: dict[str, str],
    *,
    train_log_file: str | None,
    eval_log_file: str | None,
    progress_tui: bool,
) -> int:
    train_log = _open_optional_log(train_log_file)
    eval_log = _open_optional_log(eval_log_file)
    started = time.monotonic()
    last_status = ""
    artifact_rows = _native_train_artifact_rows(command)
    for label, value in artifact_rows:
        _write_log(train_log, f"[nfn-train-provenance] {label}={value}\n")
    if progress_tui:
        _print_train_tui_panel(
            "NeuralFn Training Run",
            [
                ("command", shlex.join(command)),
                *artifact_rows,
                ("train log", train_log_file or "off"),
                ("eval log", eval_log_file or "off"),
            ],
        )
    try:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        selector = selectors.DefaultSelector()
        assert process.stdout is not None
        assert process.stderr is not None
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        stdout_chunks: list[str] = []
        while selector.get_map():
            for key, _events in selector.select(timeout=0.2):
                line = key.fileobj.readline()
                if line == "":
                    selector.unregister(key.fileobj)
                    continue
                if key.data == "stdout":
                    stdout_chunks.append(line)
                    _write_log(eval_log, line)
                else:
                    _write_log(train_log, line)
                    is_validation_line = bool(_NATIVE_VALIDATION_RE.search(line))
                    if is_validation_line:
                        _write_log(eval_log, line)
                    match = _NATIVE_PROGRESS_STEP_RE.search(line)
                    if match and not is_validation_line:
                        last_status = f"step {match.group(1)}/{match.group(2) or '?'}"
                    if progress_tui:
                        category = "eval" if is_validation_line else "train" if match else "native"
                        color = "1;35" if is_validation_line else "1;32" if match else "2"
                        rendered = _format_native_train_line(line)
                        print(f"\033[{color}m[{category}] {rendered}\033[0m", file=sys.stderr)
                    else:
                        sys.stderr.write(line)
                        sys.stderr.flush()
        return_code = process.wait()
        stdout_text = "".join(stdout_chunks)
        if eval_log is not None and stdout_chunks:
            _write_log(eval_log, "\n")
        payload = _native_train_parse_stdout_json(stdout_text)
        if payload is None and stdout_text:
            sys.stdout.write(stdout_text)
            sys.stdout.flush()
        elif payload is not None:
            _print_train_tui_panel(
                "NeuralFn Training Result",
                _native_train_summary_rows(
                    payload,
                    return_code=return_code,
                    elapsed_seconds=time.monotonic() - started,
                    train_log_file=train_log_file,
                    eval_log_file=eval_log_file,
                ),
                stream=sys.stdout,
            )
            _print_native_train_metrics(payload, stream=sys.stdout)
        if progress_tui:
            status = last_status or f"finished in {time.monotonic() - started:0.1f}s"
            _print_train_tui_panel(
                "NeuralFn Training Complete",
                [("exit code", str(return_code)), ("last status", status)],
            )
        return int(return_code)
    finally:
        if train_log is not None:
            train_log.close()
        if eval_log is not None:
            eval_log.close()


def _replace_value_argument(
    argv: list[str],
    aliases: tuple[str, ...],
    canonical_flag: str,
    value: str,
) -> list[str]:
    """Replace all split/equal spellings of one value option."""

    out: list[str] = []
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg in aliases:
            if idx + 1 >= len(argv) or argv[idx + 1].startswith("--"):
                raise ValueError(f"{arg} requires a value")
            idx += 2
            continue
        if any(arg.startswith(alias + "=") for alias in aliases):
            idx += 1
            continue
        out.append(arg)
        idx += 1
    out.extend([canonical_flag, value])
    return out


def _canonical_native_graph_training_tokens(tokens: list[str], plan) -> list[str]:
    """Make the validated graph, rather than conflicting CLI labels, authoritative."""

    drop_value_flags = (
        "--base-model",
        "--model",
        "--model-family",
        "--template-name",
        "--template",
        "--preset",
        "--graph-file",
        "--graph",
        "--graph-fingerprint",
        "--graph-preflight-proof",
        "--num-layers",
        "--train-seq-len",
        "--seq-len",
        "--model-dim",
        "--native-cuda-model-dim",
        "--hidden-dim",
        "--ffn-hidden-dim",
        "--native-cuda-hidden-dim",
        "--mlp-multiplier",
        "--mlp-mult",
        "--native-cuda-mlp-multiplier",
        "--multiple-of",
        "--native-cuda-multiple-of",
        "--num-heads",
        "--native-cuda-num-heads",
        "--num-kv-heads",
        "--native-cuda-num-kv-heads",
        "--vocab-size",
        "--native-cuda-vocab-size",
        "--padded-vocab-size",
        "--native-cuda-padded-vocab-size",
        "--rope-theta",
        "--rope-base",
        "--native-cuda-rope-theta",
        "--rope-factor",
        "--native-cuda-rope-factor",
        "--original-max-position",
        "--native-cuda-original-max-position",
        "--activation",
        "--native-cuda-activation",
        "--moa-interval",
        "--native-cuda-moa-interval",
    )
    out: list[str] = [tokens[0]] if tokens else ["train"]
    idx = 1
    while idx < len(tokens):
        arg = tokens[idx]
        if arg in drop_value_flags:
            if idx + 1 >= len(tokens) or tokens[idx + 1].startswith("--"):
                raise ValueError(f"{arg} requires a value")
            idx += 2
            continue
        if any(arg.startswith(flag + "=") for flag in drop_value_flags):
            idx += 1
            continue
        out.append(arg)
        idx += 1
    if plan.trainer_family in {"llama", "mixllama"}:
        out = [arg for arg in out if arg != "--train-transformer-lm"]
    out.extend(["--base-model", plan.trainer_family])
    out.extend(["--graph-file", str(plan.launch_graph)])
    out.extend(plan.trainer_arguments)
    if plan.trainer_family == "llama" and not _has_any(
        out,
        "--dry-run",
        "--native-cuda-dry-run",
        "--print-plan",
        "--native-cuda-print-plan",
        "--check-tile-ops",
        "--native-cuda-check-tile-ops",
        "--sample-token-batch",
        "--list-templates",
        "--native-cuda-list-templates",
    ):
        out.append("--train-llama-dataset-loop")
    if plan.trainer_family == "mixllama" and not _has_any(
        out,
        "--dry-run",
        "--native-cuda-dry-run",
        "--print-plan",
        "--native-cuda-print-plan",
        "--check-tile-ops",
        "--native-cuda-check-tile-ops",
        "--sample-token-batch",
        "--list-templates",
        "--native-cuda-list-templates",
    ):
        out.append("--train-moe-dataset-loop")
    return out


_NATIVE_GRAPH_TRAIN_VALUE_FLAGS = frozenset(
    {
        "--train-seq-len",
        "--train-batch-tokens",
        "--train-loss-every-steps",
        "--train-log-every",
        "--train-log-every-steps",
    }
)


def _native_graph_training_caller_actions(tokens: list[str]) -> tuple[str, ...]:
    """Reject caller-selected execution modes before graph canonicalization."""

    actions: list[str] = []
    for raw in tokens:
        flag = str(raw).split("=", 1)[0]
        is_action = (
            flag == "--no-train-transformer-lm"
            or flag.startswith("--smoke-")
            or flag.startswith("--native-cuda-smoke-")
            or flag.startswith("--native-cuda-train-")
            or (flag.startswith("--train-") and flag not in _NATIVE_GRAPH_TRAIN_VALUE_FLAGS)
        )
        if is_action and flag not in actions:
            actions.append(flag)
    return tuple(actions)


def _native_graph_training_artifact_dir(command: list[str]) -> Path:
    output_value = _arg_value(
        command, "--output-dir", "--native-cuda-output-dir"
    ) or os.environ.get(
        "NATIVE_CUDA_OUTPUT_DIR", ""
    )
    if str(output_value).strip():
        output_dir = Path(str(output_value)).expanduser().resolve()
    else:
        output_dir = (Path.home() / "NeuralFn" / "artifacts" / "gpt").resolve()
    return output_dir / "native-ir"


def _print_native_graph_training_rejection(plan) -> None:
    issues = [
        issue.to_dict()
        for issue in (*plan.compatibility_report.issues, *plan.training_issues)
        if issue.severity == "error"
    ]
    payload = {
        "status": "native-graph-training-incompatible",
        "source_graph": str(plan.source_graph),
        "graph_fingerprint": plan.compatibility_report.graph_fingerprint,
        "structurally_compatible": plan.compatibility_report.compatible,
        "trainer_family": plan.trainer_family,
        "training_selector": plan.training_selector,
        "native_target": plan.native_target,
        "execution_ready": plan.execution_ready,
        "trainer_consumes_native_ir": plan.trainer_consumes_native_ir,
        "graph_preflight_enforced": plan.graph_preflight_enforced,
        "blockers": list(plan.blockers),
        "issues": issues,
    }
    print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)


def _direct_native_train_cli_main(
    argv: list[str] | None = None,
    *,
    progress_tui: bool = False,
) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    source_graph = _arg_value(tokens, "--graph-file", "--graph")
    graph_plan = None
    graph_plan_materialized = False
    if source_graph:
        try:
            from neuralfn.native_graph_train import plan_native_graph_training

            graph_plan = plan_native_graph_training(source_graph)
        except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"Native graph training preflight failed: {exc}", file=sys.stderr)
            return 2
        if not graph_plan.execution_ready:
            _print_native_graph_training_rejection(graph_plan)
            return 2
        unsupported_actions = _native_graph_training_caller_actions(tokens)
        if unsupported_actions:
            print(
                "Native graph training selects its reviewed production action from the graph; "
                f"remove: {', '.join(unsupported_actions)}",
                file=sys.stderr,
            )
            return 2
        try:
            tokens = _canonical_native_graph_training_tokens(tokens, graph_plan)
        except ValueError as exc:
            print(f"Invalid native graph training option: {exc}", file=sys.stderr)
            return 2
        if graph_plan.training_selector == "gpt2_diff":
            planner_inspection = _has_any(
                tokens,
                "--dry-run",
                "--native-cuda-dry-run",
                "--print-plan",
                "--native-cuda-print-plan",
            )
            print_command = _has_any(
                tokens,
                "--print-command",
                "--native-cuda-print-command",
            )
            if planner_inspection:
                # Inspection is a Python planner result, not an unproved native
                # child command.  It therefore remains non-mutating while the
                # C++ boundary stays strict for every graph-bound invocation.
                print(json.dumps(graph_plan.to_dict(), indent=2, sort_keys=True))
                return 0
            if print_command:
                print(
                    json.dumps(
                        {
                            "status": "native-graph-training-materialization-required",
                            "training_selector": graph_plan.training_selector,
                            "source_graph": str(graph_plan.source_graph),
                            "source_graph_sha256": (
                                graph_plan.compatibility_report.graph_fingerprint
                            ),
                            "executable_command": None,
                            "workflow": (
                                "Run without --print-command so the trusted planner can "
                                "materialize source-graph.json and native-training-proof.json "
                                "before constructing the native child command."
                            ),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            artifact_dir = _native_graph_training_artifact_dir(tokens)
            try:
                materialized_plan = plan_native_graph_training(
                    graph_plan.source_graph,
                    artifact_dir=artifact_dir,
                    materialize=True,
                )
            except (
                FileExistsError,
                FileNotFoundError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                print(
                    f"Unable to materialize native graph training preflight: {exc}",
                    file=sys.stderr,
                )
                return 2
            if (
                not materialized_plan.execution_ready
                or materialized_plan.compatibility_report.graph_fingerprint
                != graph_plan.compatibility_report.graph_fingerprint
                or materialized_plan.training_selector != graph_plan.training_selector
                or materialized_plan.graph_preflight_proof is None
            ):
                print(
                    "Source graph changed while preparing native training; refusing to launch.",
                    file=sys.stderr,
                )
                return 2
            graph_plan = materialized_plan
            graph_plan_materialized = True
            try:
                tokens = _canonical_native_graph_training_tokens(tokens, graph_plan)
            except ValueError as exc:
                print(f"Invalid native graph training option: {exc}", file=sys.stderr)
                return 2
            print(
                "[nfn-native-graph] "
                f"selector={graph_plan.training_selector} "
                f"fingerprint={graph_plan.compatibility_report.graph_fingerprint} "
                f"artifact={graph_plan.artifact_metadata['training_plan_path']}",
                file=sys.stderr,
            )
    try:
        command = _direct_native_train_cli_argv(tokens)
    except ValueError as exc:
        print(f"Unable to resolve native training command: {exc}", file=sys.stderr)
        return 2
    model = _native_train_model(tokens)
    token_lm_requested = any(arg == "--train-token-lm" for arg in tokens)
    routed_family = (
        _native_template_family(command)
        if _is_dense_gpt_native_model(model) and not (model in {"nanogpt", "nano-gpt"} and token_lm_requested)
        else model
    )
    direct_family_cli = (
        not (_is_dense_gpt_native_model(model) and not (model in {"nanogpt", "nano-gpt"} and token_lm_requested))
        and _resolve_direct_native_train_family_cli(model) is not None
    )
    env = os.environ.copy()
    _set_env_default_if_empty(env, "CUDA_VISIBLE_DEVICES", resolve_cuda_visible_devices_value("0"))
    _set_env_default_if_empty(env, "CUDA_DEVICE_MAX_CONNECTIONS", "1")
    _set_env_default_if_empty(env, "CUDA_MODULE_LOADING", "LAZY")
    train_log_file, eval_log_file = _native_train_log_paths(tokens)
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
        "--print-command" in command
        and not any(flag in command for flag in native_execution_flags)
    ):
        print(shlex.join(command))
        return 0
    graph_training_execution = bool(
        graph_plan is not None
        and not graph_plan_materialized
        and "--dry-run" not in command
        and "--print-command" not in command
        and not any(flag in command for flag in native_execution_flags)
    )
    if graph_training_execution:
        from neuralfn.native_graph_train import plan_native_graph_training

        artifact_dir = _native_graph_training_artifact_dir(command)
        try:
            materialized_plan = plan_native_graph_training(
                graph_plan.source_graph,
                artifact_dir=artifact_dir,
                materialize=True,
            )
        except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"Unable to materialize native graph training preflight: {exc}", file=sys.stderr)
            return 2
        if (
            not materialized_plan.execution_ready
            or materialized_plan.compatibility_report.graph_fingerprint
            != graph_plan.compatibility_report.graph_fingerprint
            or materialized_plan.training_selector != graph_plan.training_selector
        ):
            print(
                "Source graph changed while preparing native training; refusing to launch.",
                file=sys.stderr,
            )
            return 2
        graph_plan = materialized_plan
        command = _replace_value_argument(
            command,
            ("--graph-file", "--graph"),
            "--graph-file",
            str(graph_plan.launch_graph),
        )
        if graph_plan.graph_preflight_proof is not None:
            command = _replace_value_argument(
                command,
                ("--graph-preflight-proof",),
                "--graph-preflight-proof",
                str(graph_plan.graph_preflight_proof),
            )
        print(
            "[nfn-native-graph] "
            f"selector={graph_plan.training_selector} "
            f"fingerprint={graph_plan.compatibility_report.graph_fingerprint} "
            f"artifact={graph_plan.artifact_metadata['training_plan_path']}",
            file=sys.stderr,
        )
    if model == "embedding":
        try:
            from neuralfn.native_embedding import prepare_embedding_training_command

            command, _embedding_data = prepare_embedding_training_command(command, repo_root=ROOT.parent)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Unable to prepare embedding datasets: {exc}", file=sys.stderr)
            return 2
    if "--dry-run" in command or "--print-command" in command:
        if _native_command_is_dense_gpt_cli(command):
            return _run_dense_gpt_compiled_cli_capture(command, env)
        return int(subprocess.run(command, env=env, check=False).returncode)
    return _run_native_train_with_progress(
        command,
        env,
        train_log_file=train_log_file,
        eval_log_file=eval_log_file,
        progress_tui=progress_tui,
    )


def _native_command_is_dense_gpt_cli(command: list[str]) -> bool:
    if not command:
        return False
    name = Path(str(command[0])).name
    return name in {
        "nfn_gpt_native_train",
        "nfn_gpt_native_train_linked",
        "nfn-gpt-native-train",
        "nfn-gpt-native-train-linked",
    }


def _run_dense_gpt_compiled_cli_capture(command: list[str], env: dict[str, str]) -> int:
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
    if _is_native_train_tui_request(tokens, stdin_isatty=stdin_isatty, stdout_isatty=stdout_isatty):
        return _native_train_tui_main(tokens)
    if _is_native_embedding_infer(tokens):
        return _native_embedding_infer_main(tokens)
    if _is_direct_native_train_cli_train(tokens):
        return _direct_native_train_cli_main(tokens)
    if _is_lightweight_graph_migrate(tokens):
        return _lightweight_graph_migrate_main(tokens)
    if _is_lightweight_muse_glimmer_migrate(tokens):
        return _lightweight_muse_glimmer_migrate_main(tokens)
    if _is_lightweight_muse_glimmer_gguf_migrate(tokens):
        return _lightweight_muse_glimmer_gguf_migrate_main(tokens)
    if _is_lightweight_muse_glimmer_lora_migrate(tokens):
        return _lightweight_muse_glimmer_lora_migrate_main(tokens)
    if _is_blocked_legacy_infer_request(tokens):
        return _blocked_legacy_infer_main(tokens)
    if _is_native_serve_request(tokens):
        return _native_serve_main(tokens)
    if _is_native_ir_infer_request(tokens):
        return _native_ir_infer_main(
            tokens,
            stdin_isatty=stdin_isatty,
            stdout_isatty=stdout_isatty,
        )
    if _legacy_infer_inputs(tokens) is not None:
        return _legacy_infer_main(
            tokens,
            stdin_isatty=stdin_isatty,
            stdout_isatty=stdout_isatty,
        )
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
        if _is_lightweight_native_family_infer(tokens):
            return _lightweight_native_family_infer_main(tokens)
        if _is_invalid_native_gpt_infer(tokens):
            return _invalid_native_gpt_infer_main(tokens)
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
    if _is_native_embedding_infer(sys.argv[1:]):
        main = _native_embedding_infer_main
    elif _is_native_train_tui_request(sys.argv[1:]):
        main = _native_train_tui_main
    elif _is_direct_native_train_cli_train(sys.argv[1:]):
        main = _direct_native_train_cli_main
    elif _is_explicit_native_gpt_train(sys.argv[1:]):
        from train_gpt_native import main as main
    elif _is_blocked_legacy_infer_request(sys.argv[1:]):
        main = _blocked_legacy_infer_main
    elif _is_native_serve_request(sys.argv[1:]):
        main = _native_serve_main
    elif _is_native_ir_infer_request(sys.argv[1:]):
        main = _native_ir_infer_main
    elif _legacy_infer_inputs(sys.argv[1:]) is not None:
        main = _legacy_infer_main
    elif _is_lightweight_native_gpt_infer(sys.argv[1:]):
        main = _lightweight_native_gpt_infer_main
    elif _is_lightweight_native_family_infer(sys.argv[1:]):
        main = _lightweight_native_family_infer_main
    elif _is_invalid_native_gpt_infer(sys.argv[1:]):
        main = _invalid_native_gpt_infer_main
    elif _is_lightweight_root_help(sys.argv[1:]):
        main = _lightweight_root_main
    elif _is_lightweight_graph_migrate(sys.argv[1:]):
        main = _lightweight_graph_migrate_main
    elif _is_lightweight_muse_glimmer_migrate(sys.argv[1:]):
        main = _lightweight_muse_glimmer_migrate_main
    elif _is_lightweight_muse_glimmer_gguf_migrate(sys.argv[1:]):
        main = _lightweight_muse_glimmer_gguf_migrate_main
    elif _is_lightweight_muse_glimmer_lora_migrate(sys.argv[1:]):
        main = _lightweight_muse_glimmer_lora_migrate_main
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
    if _is_native_embedding_infer(sys.argv[1:]):
        raise SystemExit(main(sys.argv[1:]))
    if _is_native_train_tui_request(sys.argv[1:]):
        raise SystemExit(main(sys.argv[1:]))
    if _is_direct_native_train_cli_train(sys.argv[1:]):
        raise SystemExit(main(sys.argv[1:]))
    if _is_explicit_native_gpt_train(sys.argv[1:]):
        raise SystemExit(main(_native_gpt_argv(sys.argv[1:])))
    if _is_blocked_legacy_infer_request(sys.argv[1:]):
        raise SystemExit(main(sys.argv[1:]))
    if _is_native_serve_request(sys.argv[1:]):
        raise SystemExit(main(sys.argv[1:]))
    if _is_native_ir_infer_request(sys.argv[1:]):
        raise SystemExit(main(sys.argv[1:]))
    if _legacy_infer_inputs(sys.argv[1:]) is not None:
        raise SystemExit(main(sys.argv[1:]))
    if _is_lightweight_native_gpt_infer(sys.argv[1:]):
        raise SystemExit(main(sys.argv[1:]))
    if _is_lightweight_native_family_infer(sys.argv[1:]):
        raise SystemExit(main(sys.argv[1:]))
    if _is_lightweight_graph_migrate(sys.argv[1:]):
        raise SystemExit(main(sys.argv[1:]))
    if _is_lightweight_muse_glimmer_migrate(sys.argv[1:]):
        raise SystemExit(main(sys.argv[1:]))
    if _is_lightweight_muse_glimmer_gguf_migrate(sys.argv[1:]):
        raise SystemExit(main(sys.argv[1:]))
    if _is_lightweight_muse_glimmer_lora_migrate(sys.argv[1:]):
        raise SystemExit(main(sys.argv[1:]))
    if _is_legacy_graph_train(sys.argv[1:]):
        raise SystemExit(main(sys.argv[1:]))
    raise SystemExit(main())
