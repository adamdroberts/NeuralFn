#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLM_KITTENS_ROOT="${LLM_KITTENS_ROOT:-/mnt/disk2/dev/open-source/llm.kittens}"
LLM_KITTENS_TRAIN_BIN="${LLM_KITTENS_TRAIN_BIN:-$LLM_KITTENS_ROOT/train_gpt2cu}"
LLM_KITTENS_TINYSTORIES_DIR="${LLM_KITTENS_TINYSTORIES_DIR:-$LLM_KITTENS_ROOT/dev/data/tinystories}"
NFN_NATIVE_GPT_TRAIN_BIN="${NFN_NATIVE_GPT_TRAIN_BIN:-$ROOT_DIR/build/nfn_gpt_native_train}"
NFN_NATIVE_TILE_OPS_LIB="${NFN_NATIVE_TILE_OPS_LIB:-$ROOT_DIR/build/libnfn_native_train_tile_ops.so}"

STEPS="${NFN_SM120_PARITY_STEPS:-10}"
SAMPLES="${NFN_SM120_PARITY_SAMPLES:-1}"
WARMUP="${NFN_SM120_PARITY_WARMUP:-0}"
CUDA_VISIBLE_DEVICES_VALUE="${NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES:-0}"
CUDA_DEVICE_MAX_CONNECTIONS_VALUE="${NFN_SM120_PARITY_CUDA_DEVICE_MAX_CONNECTIONS:-1}"
MAX_GPU_UTILIZATION="${NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT:-15}"
COMMAND_TIMEOUT_SECONDS="${NFN_SM120_PARITY_COMMAND_TIMEOUT_SECONDS:-300}"
ACTIVATION="${NFN_SM120_PARITY_ACTIVATION:-gelu}"
JSON_OUT="${NFN_SM120_PARITY_JSON_OUT:-/tmp/nfn_sm120_parity_${STEPS}step.json}"
REFERENCE_OUTPUT_DIR="${NFN_SM120_PARITY_REFERENCE_OUTPUT_DIR:-/tmp/nfn_llmk_sm120_parity}"

if [[ ! -x "$LLM_KITTENS_TRAIN_BIN" ]]; then
  echo "llm.kittens train_gpt2cu is not executable: $LLM_KITTENS_TRAIN_BIN" >&2
  exit 2
fi
if [[ ! -x "$NFN_NATIVE_GPT_TRAIN_BIN" ]]; then
  echo "NeuralFn native GPT trainer is not executable: $NFN_NATIVE_GPT_TRAIN_BIN" >&2
  exit 2
fi
if [[ ! -f "$NFN_NATIVE_TILE_OPS_LIB" ]]; then
  echo "NeuralFn Tile ops library is missing: $NFN_NATIVE_TILE_OPS_LIB" >&2
  exit 2
fi

join_command() {
  local part
  printf -v part '%q' "$1"
  printf '%s' "$part"
  shift
  for part in "$@"; do
    printf ' %q' "$part"
  done
}

baseline_cmd="$(
  join_command \
    "$LLM_KITTENS_TRAIN_BIN" \
    -i "$LLM_KITTENS_TINYSTORIES_DIR/TinyStories_train.bin" \
    -j "$LLM_KITTENS_TINYSTORIES_DIR/TinyStories_val.bin" \
    -o "$REFERENCE_OUTPUT_DIR" \
    -v 250 \
    -s "$STEPS" \
    -g 144 \
    -h 0 \
    -b 64 \
    -t 1024 \
    -d 524288 \
    -r 0 \
    -z 1 \
    -c 0.1 \
    -l 0.0006 \
    -q 0.0 \
    -u 60 \
    -n 200 \
    -y 0 \
    -e d12 \
    -af "$ACTIVATION" \
    -x "$STEPS"
)"

candidate_cmd="$(
  join_command \
    "$NFN_NATIVE_GPT_TRAIN_BIN" \
    --backend tile-cuda \
    --tinystories \
    --max-steps "$STEPS" \
    --eval-every-steps 0 \
    --no-checkpoint \
    --tile-ops-lib "$NFN_NATIVE_TILE_OPS_LIB"
)"

cd "$ROOT_DIR"
python tools/paired_kernel_speed.py \
  --baseline "$baseline_cmd" \
  --candidate "$candidate_cmd" \
  --samples "$SAMPLES" \
  --warmup "$WARMUP" \
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES_VALUE" \
  --cuda-device-max-connections "$CUDA_DEVICE_MAX_CONNECTIONS_VALUE" \
  --require-idle-selected-gpu \
  --max-selected-gpu-utilization-pct "$MAX_GPU_UTILIZATION" \
  --command-timeout-seconds "$COMMAND_TIMEOUT_SECONDS" \
  --json-out "$JSON_OUT"
