#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/adam/miniconda3/envs/NeuralFn/bin/python}"
LINKED_TRAIN_BIN="${ROOT_DIR}/build/nfn_gpt_native_train_linked"
DYNAMIC_TRAIN_BIN="${ROOT_DIR}/build/nfn_gpt_native_train"
if [[ -n "${NFN_NATIVE_GPT_TRAIN_BIN:-}" ]]; then
  TRAIN_BIN="${NFN_NATIVE_GPT_TRAIN_BIN}"
elif [[ -x "${LINKED_TRAIN_BIN}" ]]; then
  TRAIN_BIN="${LINKED_TRAIN_BIN}"
else
  TRAIN_BIN="${DYNAMIC_TRAIN_BIN}"
fi
if [[ -n "${NFN_NATIVE_TILE_OPS_LIB:-}" ]]; then
  TILE_OPS_LIB="${NFN_NATIVE_TILE_OPS_LIB}"
elif [[ "$(basename "${TRAIN_BIN}")" == "nfn_gpt_native_train_linked" ]]; then
  TILE_OPS_LIB="linked"
else
  TILE_OPS_LIB="${ROOT_DIR}/build/libnfn_native_train_tile_ops.so"
fi
BENCH_JSON="${NFN_SM120_CUDA13_JSON_OUT:-/tmp/nfn_sm120_cuda13_baseline.json}"
CHECK_BENCH_CONTRACT="${NFN_SM120_CUDA13_CHECK_BENCH_CONTRACT:-1}"

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

if [[ "${TILE_OPS_LIB}" != "linked" && ! -f "${TILE_OPS_LIB}" ]]; then
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
  --smoke-nvfp4-pack \
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
    case "${CHECK_BENCH_CONTRACT}" in
      1|true|TRUE|yes|YES|on|ON)
        run_step "${PYTHON_BIN}" - "${BENCH_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
values = payload.get("candidate_native_metric_values") or {}
metrics = payload.get("candidate_native_metrics") or {}

def value(name):
    observed = values.get(name)
    if isinstance(observed, list) and len(observed) == 1:
        return observed[0]
    return observed

def metric_mean(name):
    item = metrics.get(name) or {}
    return item.get("mean")

checks = [
    (
        value("optimizer_tile_strategy") == "tile-size-1024-sumsq-scale-adamw",
        "optimizer_tile_strategy must stay on the fused Tile AdamW path",
    ),
    (
        value("lm_head_classifier_backward_path_class")
        == "diagnostic-cuda-graph-wrapper",
        "LM-head backward must stay on the promoted CUDA Graph wrapper until a faster true-fused Tile kernel replaces it",
    ),
    (
        value("lm_head_ce_kernel_strategy")
        in {
            "no-loss-default-specialized-dlogits-vec8-loads-scalar-stores",
            "no-loss-specialized-dlogits-vec8-loads-scalar-stores",
            "no-loss-specialized-dlogits-vec8-loads-normal-vec8-stores",
            "default-specialized-loss-bins-vec8-loads-scalar-stores",
        },
        "LM-head CE must use a promoted specialized BF16/u16 Tile route",
    ),
    (
        metric_mean("lm_head_fused_graph_prewarm_success_count") is not None
        and metric_mean("lm_head_fused_graph_prewarm_success_count") >= 1.0,
        "LM-head CUDA Graph prewarm must succeed",
    ),
    (
        metric_mean("lm_head_fused_graph_prewarm_duplicate_skip_count") is not None,
        "LM-head graph prewarm must report pointer-aware dedup telemetry",
    ),
    (
        value("block_backward_input_linear_strategy") == "tk-sm120-bf16-dinput",
        "block backward dInput must stay on the SM120 TK BF16 route",
    ),
    (
        value("block_backward_weight_linear_strategy")
        == "shape-gated-bf16-cublaslt-dweight-bgrad-first-write-then-accumulate",
        "block backward dWeight+bias must stay on the promoted cuBLASLt BGRADB route",
    ),
    (
        value("token_weight_init_strategy")
        == "device-vector4-strided-power2-deterministic-fused-bf16-shadow",
        "token-weight init must stay on the default vector4-strided BF16-shadow CUDA Tile route",
    ),
]

failed = [message for ok, message in checks if not ok]
if failed:
    print(f"SM120 CUDA 13.3 benchmark contract failed for {path}:", file=sys.stderr)
    for message in failed:
        print(f"  - {message}", file=sys.stderr)
    sys.exit(2)

print(f"SM120 CUDA 13.3 benchmark contract passed for {path}")
PY
        ;;
      0|false|FALSE|no|NO|off|OFF)
        ;;
      *)
        echo "Unsupported NFN_SM120_CUDA13_CHECK_BENCH_CONTRACT=${CHECK_BENCH_CONTRACT}" >&2
        exit 2
        ;;
    esac
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "Unsupported NFN_SM120_CUDA13_RUN_BENCH=${NFN_SM120_CUDA13_RUN_BENCH}" >&2
    exit 2
    ;;
esac

printf '\nCUDA 13.3 SM120 validation passed.\n'
