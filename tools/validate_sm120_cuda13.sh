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
PARITY_JSON="${NFN_SM120_CUDA13_PARITY_JSON_OUT:-/tmp/nfn_sm120_cuda13_parity.json}"
LM_HEAD_JSON="${NFN_SM120_CUDA13_LM_HEAD_JSON_OUT:-/tmp/nfn_sm120_cuda13_lm_head_backward.json}"
CHECK_BENCH_CONTRACT="${NFN_SM120_CUDA13_CHECK_BENCH_CONTRACT:-1}"
PARITY_STEPS="${NFN_SM120_CUDA13_PARITY_STEPS:-3}"
PARITY_SAMPLES="${NFN_SM120_CUDA13_PARITY_SAMPLES:-2}"
PARITY_WARMUP="${NFN_SM120_CUDA13_PARITY_WARMUP:-0}"
REBUILD_STALE="${NFN_SM120_CUDA13_REBUILD_STALE:-1}"
LM_HEAD_TILE_OPS_LIB="${NFN_SM120_CUDA13_LM_HEAD_TILE_OPS_LIB:-${ROOT_DIR}/build/libnfn_native_train_tile_ops.so}"
if [[ -n "${NFN_SM120_CUDA13_PARITY_ENFORCE_GATE:-}" ]]; then
  PARITY_ENFORCE_GATE="${NFN_SM120_CUDA13_PARITY_ENFORCE_GATE}"
elif [[ "${PARITY_STEPS}" =~ ^[0-9]+$ && "${PARITY_STEPS}" -lt 2 ]]; then
  PARITY_ENFORCE_GATE=0
else
  PARITY_ENFORCE_GATE=1
fi

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

case "${NFN_SM120_CUDA13_RUN_LM_HEAD_BENCH:-1}" in
  1|true|TRUE|yes|YES|on|ON)
    if [[ ! -x "${ROOT_DIR}/build/lm_head_backward_bench" ]]; then
      echo "Missing LM-head backward benchmark: ${ROOT_DIR}/build/lm_head_backward_bench" >&2
      echo "Build it with: bash tools/rebuild_native_sm120.sh" >&2
      exit 2
    fi
    if [[ ! -f "${LM_HEAD_TILE_OPS_LIB}" ]]; then
      echo "Missing LM-head Tile ops library: ${LM_HEAD_TILE_OPS_LIB}" >&2
      echo "Build it with: bash tools/rebuild_native_sm120.sh" >&2
      exit 2
    fi
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "Unsupported NFN_SM120_CUDA13_RUN_LM_HEAD_BENCH=${NFN_SM120_CUDA13_RUN_LM_HEAD_BENCH}" >&2
    exit 2
    ;;
esac

case "${NFN_SM120_CUDA13_RUN_NO_TORCH:-1}" in
  1|true|TRUE|yes|YES|on|ON)
    case "${REBUILD_STALE}" in
      1|true|TRUE|yes|YES|on|ON)
        run_step "${PYTHON_BIN}" tools/check_native_no_torch_deps.py --rebuild-stale --json
        ;;
      0|false|FALSE|no|NO|off|OFF)
        run_step "${PYTHON_BIN}" tools/check_native_no_torch_deps.py --json
        ;;
      *)
        echo "Unsupported NFN_SM120_CUDA13_REBUILD_STALE=${REBUILD_STALE}" >&2
        exit 2
        ;;
    esac
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "Unsupported NFN_SM120_CUDA13_RUN_NO_TORCH=${NFN_SM120_CUDA13_RUN_NO_TORCH}" >&2
    exit 2
    ;;
esac

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

case "${NFN_SM120_CUDA13_RUN_LM_HEAD_BENCH:-1}" in
  1|true|TRUE|yes|YES|on|ON)
    run_step env \
      NFN_LM_HEAD_BACKWARD_BENCH_BIN="${ROOT_DIR}/build/lm_head_backward_bench" \
      NFN_NATIVE_TILE_OPS_LIB="${LM_HEAD_TILE_OPS_LIB}" \
      NFN_LM_HEAD_BACKWARD_PROFILE="${NFN_SM120_CUDA13_LM_HEAD_PROFILE:-trainer-chunk}" \
      NFN_LM_HEAD_BACKWARD_WARMUP="${NFN_SM120_CUDA13_LM_HEAD_WARMUP:-1}" \
      NFN_LM_HEAD_BACKWARD_JSON_OUT="${LM_HEAD_JSON}" \
      bash "${ROOT_DIR}/tools/bench_lm_head_backward_candidate.sh"
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
esac

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
        value("graph_editor_tensor_flow") == "false",
        "native benchmark JSON must report graph_editor_tensor_flow=false",
    ),
    (
        value("torch_required") == "false",
        "native benchmark JSON must report torch_required=false",
    ),
    (
        value("optimized_kernel_contract_passed") == "true",
        "native benchmark JSON must report optimized_kernel_contract_passed=true",
    ),
    (
        metric_mean("train_loss_host_d2h_count") == 0.0,
        "training loss accumulation must stay device-resident with train_loss_host_d2h_count=0",
    ),
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
        == "no-loss-llmk-style-dlogits-vec8-loads-streaming-vec8-stores",
        "LM-head CE must stay on the promoted llm.kittens-style no-loss BF16/u16 Tile route",
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
        metric_mean("block_backward_qkv_dinput_before_dweight_count") is not None
        and metric_mean("block_backward_qkv_dinput_before_dweight_count") > 0.0,
        "QKV backward must keep the promoted dInput-before-dWeight route active",
    ),
    (
        metric_mean("block_state_layout.layer_norm_backward_affine_row_chunk_size") == 128.0,
        "LayerNorm affine backward must keep the promoted 128-row reduction chunk",
    ),
    (
        value("block_backward_weight_linear_strategy")
        == "shape-gated-bf16-cublaslt-dweight-bgrad-first-write-then-accumulate",
        "block backward dWeight+bias must stay on the promoted cuBLASLt BGRADB route",
    ),
    (
        value("token_weight_init_strategy")
        == "device-vector4-strided-power2-deterministic-fused-bf16-shadow-padded-zero",
        "token-weight init must stay on the default vector4-strided padded-zero BF16-shadow CUDA Tile route",
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

case "${NFN_SM120_CUDA13_RUN_PARITY:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    run_step env \
      NFN_NATIVE_GPT_TRAIN_BIN="${TRAIN_BIN}" \
      NFN_NATIVE_TILE_OPS_LIB="${TILE_OPS_LIB}" \
      NFN_SM120_PARITY_STEPS="${PARITY_STEPS}" \
      NFN_SM120_PARITY_SAMPLES="${PARITY_SAMPLES}" \
      NFN_SM120_PARITY_WARMUP="${PARITY_WARMUP}" \
      NFN_SM120_PARITY_PROFILE_DIR="${NFN_SM120_CUDA13_PARITY_PROFILE_DIR:-none}" \
      NFN_SM120_PARITY_JSON_OUT="${PARITY_JSON}" \
      NFN_SM120_PARITY_ENFORCE_GATE="${PARITY_ENFORCE_GATE}" \
      bash "${ROOT_DIR}/tools/bench_native_gpt_sm120_parity.sh"
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "Unsupported NFN_SM120_CUDA13_RUN_PARITY=${NFN_SM120_CUDA13_RUN_PARITY}" >&2
    exit 2
    ;;
esac

printf '\nCUDA 13.3 SM120 validation passed.\n'
