#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLM_KITTENS_TINYSTORIES_DIR="${NFN_LLM_KITTENS_TINYSTORIES_DIR:-/mnt/disk2/dev/open-source/llm.kittens/dev/data/tinystories}"
COMPILED_SM120_LAUNCHER="${NFN_NATIVE_SM120_CLI:-${ROOT_DIR}/build/nfn_train_gpt_sm120}"
USE_COMPILED_SM120_LAUNCHER="${NFN_SM120_USE_COMPILED_LAUNCHER:-1}"

case "${USE_COMPILED_SM120_LAUNCHER,,}" in
  ""|"1"|"true"|"yes"|"on")
    if [[ -x "${COMPILED_SM120_LAUNCHER}" ]]; then
      exec "${COMPILED_SM120_LAUNCHER}" "$@"
    fi
    ;;
  "0"|"false"|"no"|"off")
    ;;
  *)
    echo "Invalid NFN_SM120_USE_COMPILED_LAUNCHER='${USE_COMPILED_SM120_LAUNCHER}'" >&2
    exit 2
    ;;
esac

if [[ -z "${NFN_NATIVE_GPT_TRAIN_BIN-}" && -x "${ROOT_DIR}/build/nfn_gpt_native_train_linked" ]]; then
  NATIVE_GPT_TRAIN_BIN="${ROOT_DIR}/build/nfn_gpt_native_train_linked"
else
  NATIVE_GPT_TRAIN_BIN="${NFN_NATIVE_GPT_TRAIN_BIN:-${ROOT_DIR}/build/nfn_gpt_native_train}"
fi

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

select_auto_cuda_device() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' "0"
    return
  fi
  local query_output
  if ! query_output="$(nvidia-smi --query-gpu=index,display_active,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
    printf '%s\n' "0"
    return
  fi
  printf '%s\n' "${query_output}" | awk -F, '
      {
        idx=$1; display=$2; util=$3;
        gsub(/^[ \t]+|[ \t]+$/, "", idx);
        gsub(/^[ \t]+|[ \t]+$/, "", display);
        gsub(/^[ \t]+|[ \t]+$/, "", util);
        if (first == "") first = idx;
        if (display == "Disabled" && (best == "" || util + 0 < best_util + 0)) {
          best = idx;
          best_util = util;
        }
      }
      END {
        if (best != "") print best;
        else if (first != "") print first;
        else print "0";
      }
    '
}

if [[ -z "${CUDA_VISIBLE_DEVICES-}" ]]; then
  CUDA_VISIBLE_DEVICES_DEFAULT="${NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES:-${NFN_SM120_CUDA_VISIBLE_DEVICES:-dedicated}}"
  case "${CUDA_VISIBLE_DEVICES_DEFAULT,,}" in
    ""|"none"|"off")
      ;;
    "auto"|"dedicated"|"dedicated-auto")
      export CUDA_VISIBLE_DEVICES="$(select_auto_cuda_device)"
      ;;
    *)
      export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_DEFAULT}"
      ;;
  esac
fi

ACTIVATION="${NFN_SM120_ACTIVATION:-gelu}"
MOA_INTERVAL="${NFN_SM120_MOA_INTERVAL:-50}"
OUTPUT_DIR="${NFN_SM120_OUTPUT_DIR:-${ROOT_DIR}/artifacts/gpt_sm120}"
DATASET_ALIAS="${NFN_SM120_DATASET_ALIAS:-}"
MODEL_FAMILY="${NFN_SM120_MODEL_FAMILY:-gpt}"
TEMPLATE_NAME="${NFN_SM120_TEMPLATE_NAME:-gpt}"
GRAPH_FILE="${NFN_SM120_GRAPH_FILE:-}"
TRAIN_SEQ_LEN="${NFN_SM120_TRAIN_SEQ_LEN:-1024}"
BATCH_SIZE="${NFN_SM120_BATCH_SIZE:-64}"
SEQ_LEN_EXPLICIT=0
BATCH_SIZE_EXPLICIT=0
TEMPLATE_NAME_EXPLICIT=0
ACTIVATION_EXPLICIT=0
if [[ -n "${NFN_SM120_ACTIVATION-}" ]]; then
  ACTIVATION_EXPLICIT=1
fi
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: tools/train_gpt_sm120.sh [options] [-- extra native args]

Zero-Python SM120 dense GPT training helper. By default this shell shim execs
build/nfn_train_gpt_sm120 when available. Set NFN_SM120_USE_COMPILED_LAUNCHER=0
to use the legacy shell parser, which prefers build/nfn_gpt_native_train_linked,
falls back to build/nfn_gpt_native_train, and uses the same core defaults as
llm.kittens/train-sm120.sh:
  eval=250 sample=20000 generate=144 batch=64 seq=1024 tokens/step=524288
  weight_decay=0.1 lr=0.0006 final_lr_fraction=0 warmup=60 checkpoint=200
  max_steps=20000 activation=gelu CUDA_DEVICE_MAX_CONNECTIONS=1

Options:
  --activation NAME       gelu|relu|silu|relu2|prelu|sd-prelu|swiglu|geglu|ensemble|moa
  --moa-interval N       Mixture-of-activations interval when activation=moa
  --base-model NAME      Dense GPT alias: gpt|gpt2|gpt3|nanogpt
  --model-family NAME    Alias for --base-model
  --template-name NAME   Shipped dense GPT template selector
  --template NAME        Alias for --template-name
  --preset NAME          Alias for --template-name
  --graph-file PATH      Native-compatible dense GPT custom graph metadata
  --graph PATH           Alias for --graph-file
  --dataset-alias PATH   Dataset alias/path for the compiled C++ resolver
  --output-dir PATH      Native output directory
  -h, --help             Show this help

Any other option is forwarded to the selected native GPT trainer after the
defaults, so it can override defaults such as --batch-size, --train-seq-len,
--max-steps, --eval-every-steps, or --no-checkpoint.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --activation)
      ACTIVATION="$2"
      ACTIVATION_EXPLICIT=1
      shift 2
      ;;
    --activation=*)
      ACTIVATION="${1#*=}"
      ACTIVATION_EXPLICIT=1
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
    --base-model|--model-family)
      MODEL_FAMILY="$2"
      if [[ "${TEMPLATE_NAME_EXPLICIT}" == "0" ]]; then
        TEMPLATE_NAME="$2"
      fi
      shift 2
      ;;
    --base-model=*|--model-family=*)
      MODEL_FAMILY="${1#*=}"
      if [[ "${TEMPLATE_NAME_EXPLICIT}" == "0" ]]; then
        TEMPLATE_NAME="${1#*=}"
      fi
      shift
      ;;
    --template-name|--template|--preset)
      TEMPLATE_NAME="$2"
      TEMPLATE_NAME_EXPLICIT=1
      shift 2
      ;;
    --template-name=*|--template=*|--preset=*)
      TEMPLATE_NAME="${1#*=}"
      TEMPLATE_NAME_EXPLICIT=1
      shift
      ;;
    --graph-file|--graph)
      GRAPH_FILE="$2"
      shift 2
      ;;
    --graph-file=*|--graph=*)
      GRAPH_FILE="${1#*=}"
      shift
      ;;
    --train-seq-len|--seq-len)
      TRAIN_SEQ_LEN="$2"
      SEQ_LEN_EXPLICIT=1
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --train-seq-len=*|--seq-len=*)
      TRAIN_SEQ_LEN="${1#*=}"
      SEQ_LEN_EXPLICIT=1
      EXTRA_ARGS+=("$1")
      shift
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      BATCH_SIZE_EXPLICIT=1
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --batch-size=*)
      BATCH_SIZE="${1#*=}"
      BATCH_SIZE_EXPLICIT=1
      EXTRA_ARGS+=("$1")
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

MODEL_FAMILY="${MODEL_FAMILY,,}"
TEMPLATE_NAME="${TEMPLATE_NAME,,}"
ACTIVATION="${ACTIVATION,,}"
case "${TEMPLATE_NAME}" in
  *moa*)
    if [[ "${ACTIVATION_EXPLICIT}" == "0" ]]; then
      ACTIVATION="moa"
    fi
    ;;
esac
case "${ACTIVATION}" in
  gelu|relu|silu|relu2|prelu|sd-prelu|swiglu|geglu|ensemble|moa) ;;
  *)
    echo "Invalid --activation '${ACTIVATION}'" >&2
    exit 2
    ;;
esac

case "${MODEL_FAMILY}" in
  gpt|gpt2|gpt3|nanogpt) ;;
  *)
    echo "Invalid --base-model/--model-family '${MODEL_FAMILY}'" >&2
    exit 2
    ;;
esac

case "${TEMPLATE_NAME}" in
  gpt3)
    if [[ "${SEQ_LEN_EXPLICIT}" == "0" ]]; then
      TRAIN_SEQ_LEN=2048
    fi
    if [[ "${BATCH_SIZE_EXPLICIT}" == "0" ]]; then
      BATCH_SIZE=32
    fi
    ;;
esac

if [[ ! -x "${NATIVE_GPT_TRAIN_BIN}" ]]; then
  echo "Native GPT trainer is not executable: ${NATIVE_GPT_TRAIN_BIN}" >&2
  echo "Build it with: bash tools/build_native_gpt_cli_linked.sh" >&2
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

GRAPH_ARGS=()
if [[ -n "${GRAPH_FILE}" ]]; then
  GRAPH_ARGS=(--graph-file "${GRAPH_FILE}")
fi

exec "${NATIVE_GPT_TRAIN_BIN}" \
  --model-family "${MODEL_FAMILY}" \
  --template-name "${TEMPLATE_NAME}" \
  "${GRAPH_ARGS[@]}" \
  "${DATASET_ARGS[@]}" \
  --backend tile-cuda \
  "${TILE_OPS_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --eval-every-steps 250 \
  --eval-batches 20 \
  --native-cuda-sample-every 20000 \
  --native-cuda-generate-tokens 144 \
  --batch-size "${BATCH_SIZE}" \
  --train-seq-len "${TRAIN_SEQ_LEN}" \
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
