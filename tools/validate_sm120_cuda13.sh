#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/adam/miniconda3/envs/NeuralFn/bin/python}"
TRAIN_BIN="${NFN_NATIVE_GPT_TRAIN_BIN:-${ROOT_DIR}/build/nfn_gpt_native_train}"
TILE_OPS_LIB="${NFN_NATIVE_TILE_OPS_LIB:-${ROOT_DIR}/build/libnfn_native_train_tile_ops.so}"
BENCH_JSON="${NFN_SM120_CUDA13_JSON_OUT:-/tmp/nfn_sm120_cuda13_baseline.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES="${NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES:-dedicated}"

run_step() {
  printf '\n==> %s\n' "$*"
  "$@"
}

if [[ ! -x "${TRAIN_BIN}" ]]; then
  echo "Missing native GPT trainer: ${TRAIN_BIN}" >&2
  echo "Build it with: bash tools/rebuild_native_sm120.sh" >&2
  exit 2
fi

if [[ ! -f "${TILE_OPS_LIB}" ]]; then
  echo "Missing Tile ops library: ${TILE_OPS_LIB}" >&2
  echo "Build it with: bash tools/rebuild_native_sm120.sh" >&2
  exit 2
fi

run_step \
  "${TRAIN_BIN}" \
  --backend tile-cuda \
  --check-tile-ops \
  --tile-ops-lib "${TILE_OPS_LIB}"

run_step \
  "${TRAIN_BIN}" \
  --backend tile-cuda \
  --smoke-tile-ops \
  --tile-ops-lib "${TILE_OPS_LIB}"

run_step \
  "${TRAIN_BIN}" \
  --backend tile-cuda \
  --tinystories \
  --smoke-transformer-lm-step \
  --tile-ops-lib "${TILE_OPS_LIB}"

case "${NFN_SM120_CUDA13_RUN_PYTEST:-1}" in
  1|true|TRUE|yes|YES|on|ON)
    run_step "${PYTHON_BIN}" -m pytest tests/test_native_gpt2.py -q
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "Unsupported NFN_SM120_CUDA13_RUN_PYTEST=${NFN_SM120_CUDA13_RUN_PYTEST}" >&2
    exit 2
    ;;
esac

case "${NFN_SM120_CUDA13_RUN_BENCH:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    run_step env \
      NFN_NATIVE_GPT_TRAIN_BIN="${TRAIN_BIN}" \
      NFN_NATIVE_TILE_OPS_LIB="${TILE_OPS_LIB}" \
      NFN_SM120_NATIVE_STEPS="${NFN_SM120_CUDA13_BENCH_STEPS:-3}" \
      NFN_SM120_NATIVE_SAMPLES="${NFN_SM120_CUDA13_BENCH_SAMPLES:-2}" \
      NFN_SM120_NATIVE_WARMUP="${NFN_SM120_CUDA13_BENCH_WARMUP:-0}" \
      NFN_SM120_NATIVE_INCLUDE_LLMK_REFERENCE="${NFN_SM120_CUDA13_INCLUDE_LLMK_REFERENCE:-0}" \
      NFN_SM120_NATIVE_PROFILE_DIR="${NFN_SM120_CUDA13_PROFILE_DIR:-none}" \
      NFN_SM120_NATIVE_DISABLE_METRIC_RATIO_GATES="${NFN_SM120_CUDA13_DISABLE_METRIC_RATIO_GATES:-1}" \
      NFN_SM120_NATIVE_JSON_OUT="${BENCH_JSON}" \
      bash "${ROOT_DIR}/tools/bench_native_gpt_sm120_candidate.sh"
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "Unsupported NFN_SM120_CUDA13_RUN_BENCH=${NFN_SM120_CUDA13_RUN_BENCH}" >&2
    exit 2
    ;;
esac

printf '\nCUDA 13.3 SM120 validation passed.\n'
