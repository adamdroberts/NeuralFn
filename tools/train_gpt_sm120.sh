#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLM_KITTENS_TINYSTORIES_DIR="${NFN_LLM_KITTENS_TINYSTORIES_DIR:-/mnt/disk2/dev/open-source/llm.kittens/dev/data/tinystories}"

if [[ -z "${NFN_NATIVE_GPT_TRAIN_BIN-}" && -x "${ROOT_DIR}/build/nfn_gpt_native_train_linked" ]]; then
  NATIVE_GPT_TRAIN_BIN="${ROOT_DIR}/build/nfn_gpt_native_train_linked"
else
  NATIVE_GPT_TRAIN_BIN="${NFN_NATIVE_GPT_TRAIN_BIN:-${ROOT_DIR}/build/nfn_gpt_native_train}"
fi

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

ACTIVATION="${NFN_SM120_ACTIVATION:-gelu}"
MOA_INTERVAL="${NFN_SM120_MOA_INTERVAL:-50}"
OUTPUT_DIR="${NFN_SM120_OUTPUT_DIR:-${ROOT_DIR}/artifacts/gpt_sm120}"
DATASET_ALIAS="${NFN_SM120_DATASET_ALIAS:-}"
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: tools/train_gpt_sm120.sh [options] [-- extra native args]

Zero-Python SM120 dense GPT training helper. It calls nfn_gpt_native_train
directly with the same core defaults as llm.kittens/train-sm120.sh:
  eval=250 sample=20000 generate=144 batch=64 seq=1024 tokens/step=524288
  weight_decay=0.1 lr=0.0006 final_lr_fraction=0 warmup=60 checkpoint=200
  max_steps=20000 activation=gelu CUDA_DEVICE_MAX_CONNECTIONS=1

Options:
  --activation NAME       gelu|relu|silu|relu2|prelu|sd-prelu|swiglu|geglu|ensemble|moa
  --moa-interval N       Mixture-of-activations interval when activation=moa
  --dataset-alias PATH   Dataset alias/path for the compiled C++ resolver
  --output-dir PATH      Native output directory
  -h, --help             Show this help

Any other option is forwarded to nfn_gpt_native_train after the defaults, so it
can override defaults such as --max-steps, --eval-every-steps, or --no-checkpoint.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --activation)
      ACTIVATION="$2"
      shift 2
      ;;
    --activation=*)
      ACTIVATION="${1#*=}"
      shift
      ;;
    --moa-interval)
      MOA_INTERVAL="$2"
      shift 2
      ;;
    --moa-interval=*)
      MOA_INTERVAL="${1#*=}"
      shift
      ;;
    --dataset-alias|--dataset-path)
      DATASET_ALIAS="$2"
      shift 2
      ;;
    --dataset-alias=*|--dataset-path=*)
      DATASET_ALIAS="${1#*=}"
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

ACTIVATION="${ACTIVATION,,}"
case "${ACTIVATION}" in
  gelu|relu|silu|relu2|prelu|sd-prelu|swiglu|geglu|ensemble|moa) ;;
  *)
    echo "Invalid --activation '${ACTIVATION}'" >&2
    exit 2
    ;;
esac

if [[ ! -x "${NATIVE_GPT_TRAIN_BIN}" ]]; then
  echo "Native GPT trainer is not executable: ${NATIVE_GPT_TRAIN_BIN}" >&2
  echo "Build it with: bash tools/build_native_gpt_cli.sh" >&2
  exit 127
fi

DATASET_ARGS=()
if [[ -n "${DATASET_ALIAS}" ]]; then
  DATASET_ARGS=(--dataset-alias "${DATASET_ALIAS}")
elif [[ -f "${LLM_KITTENS_TINYSTORIES_DIR}/TinyStories_train.bin" && -f "${LLM_KITTENS_TINYSTORIES_DIR}/TinyStories_val.bin" ]]; then
  DATASET_ARGS=(--dataset-alias "${LLM_KITTENS_TINYSTORIES_DIR}")
else
  DATASET_ARGS=(--tinystories)
fi

TILE_OPS_ARGS=()
if [[ "$(basename "${NATIVE_GPT_TRAIN_BIN}")" == "nfn_gpt_native_train_linked" ]]; then
  TILE_OPS_ARGS=(--tile-ops-lib linked)
fi

MOA_ARGS=()
if [[ "${ACTIVATION}" == "moa" ]]; then
  MOA_ARGS=(--native-cuda-moa-interval "${MOA_INTERVAL}")
fi

exec "${NATIVE_GPT_TRAIN_BIN}" \
  --model-family gpt \
  --template-name gpt \
  "${DATASET_ARGS[@]}" \
  --backend tile-cuda \
  "${TILE_OPS_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --eval-every-steps 250 \
  --eval-batches 20 \
  --native-cuda-sample-every 20000 \
  --native-cuda-generate-tokens 144 \
  --batch-size 64 \
  --train-seq-len 1024 \
  --train-batch-tokens 524288 \
  --learning-rate 0.0006 \
  --final-lr-fraction 0.0 \
  --weight-decay 0.1 \
  --warmup-steps 60 \
  --native-cuda-checkpoint-every 200 \
  --max-steps 20000 \
  --native-cuda-activation "${ACTIVATION}" \
  "${MOA_ARGS[@]}" \
  --train-transformer-lm \
  "${EXTRA_ARGS[@]}"
