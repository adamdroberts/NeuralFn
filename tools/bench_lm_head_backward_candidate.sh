#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_BIN="${NFN_LM_HEAD_BACKWARD_BENCH_BIN:-${ROOT_DIR}/build/lm_head_backward_bench}"
TILE_OPS_LIB="${NFN_NATIVE_TILE_OPS_LIB:-${ROOT_DIR}/build/libnfn_native_train_tile_ops.so}"
JSON_OUT="${NFN_LM_HEAD_BACKWARD_JSON_OUT:-/tmp/nfn_lm_head_backward_bench.json}"
HIDDEN_DIM="${NFN_LM_HEAD_BACKWARD_HIDDEN_DIM:-768}"
VOCAB="${NFN_LM_HEAD_BACKWARD_VOCAB:-50257}"
ROW_STRIDE="${NFN_LM_HEAD_BACKWARD_ROW_STRIDE:-50304}"
CUDA_VISIBLE_DEVICES_VALUE="${NFN_LM_HEAD_BACKWARD_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-dedicated}}"
CUDA_DEVICE_RAW="${NFN_LM_HEAD_BACKWARD_CUDA_DEVICE:-auto}"
BASELINE_SYMBOL="${NFN_LM_HEAD_BACKWARD_BASELINE_SYMBOL:-nfn_native_tile_lm_head_classifier_backward_cooperative_bf16_u16}"
CANDIDATE_SYMBOL="${NFN_LM_HEAD_BACKWARD_CANDIDATE_SYMBOL:-nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16}"
PROFILE="${NFN_LM_HEAD_BACKWARD_PROFILE:-smoke}"

case "${PROFILE}" in
  smoke)
    DEFAULT_ROWS=2048
    DEFAULT_ITERATIONS=5
    DEFAULT_WARMUP=1
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=0
    ;;
  trainer-chunk|trainer_chunk)
    DEFAULT_ROWS=32768
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=1
    ;;
  trainer-row-loss|trainer_row_loss)
    DEFAULT_ROWS=32768
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=0
    ;;
  trainer-loss-bins|trainer_loss_bins)
    DEFAULT_ROWS=32768
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    DEFAULT_LOSS_BINS=1024
    DEFAULT_NO_LOSS=0
    ;;
  *)
    echo "Unknown NFN_LM_HEAD_BACKWARD_PROFILE='${PROFILE}' (expected smoke, trainer-chunk, trainer-row-loss, or trainer-loss-bins)" >&2
    exit 2
    ;;
esac

ROWS="${NFN_LM_HEAD_BACKWARD_ROWS:-${DEFAULT_ROWS}}"
ITERATIONS="${NFN_LM_HEAD_BACKWARD_ITERATIONS:-${DEFAULT_ITERATIONS}}"
WARMUP="${NFN_LM_HEAD_BACKWARD_WARMUP:-${DEFAULT_WARMUP}}"
LOSS_BINS="${NFN_LM_HEAD_BACKWARD_LOSS_BINS:-${DEFAULT_LOSS_BINS}}"
NO_LOSS="${NFN_LM_HEAD_BACKWARD_NO_LOSS:-${DEFAULT_NO_LOSS}}"
MAX_RATIO="${NFN_LM_HEAD_BACKWARD_MAX_RATIO:-}"
REQUIRE_TRUE_FUSED="${NFN_LM_HEAD_BACKWARD_REQUIRE_TRUE_FUSED:-0}"
CANDIDATE_FIRST="${NFN_LM_HEAD_BACKWARD_CANDIDATE_FIRST:-0}"
DRY_RUN="${NFN_LM_HEAD_BACKWARD_DRY_RUN:-0}"

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

case "${CUDA_VISIBLE_DEVICES_VALUE,,}" in
  ""|"none"|"off")
    ;;
  "auto"|"dedicated"|"dedicated-auto")
    SELECTED_CUDA_VISIBLE_DEVICE="$(select_auto_cuda_device)"
    export CUDA_VISIBLE_DEVICES="${SELECTED_CUDA_VISIBLE_DEVICE}"
    ;;
  *)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
    ;;
esac

case "${CUDA_DEVICE_RAW,,}" in
  "auto")
    CUDA_DEVICE=0
    ;;
  *)
    CUDA_DEVICE="${CUDA_DEVICE_RAW}"
    ;;
esac

case "${NO_LOSS,,}" in
  1|true|yes|on)
    NO_LOSS_ARG=(--no-loss)
    ;;
  0|false|no|off|"")
    NO_LOSS_ARG=()
    ;;
  *)
    echo "Invalid NFN_LM_HEAD_BACKWARD_NO_LOSS='${NO_LOSS}'" >&2
    exit 2
    ;;
esac

case "${CANDIDATE_FIRST,,}" in
  1|true|yes|on)
    CANDIDATE_FIRST_ARG=(--candidate-first)
    ;;
  0|false|no|off|"")
    CANDIDATE_FIRST_ARG=()
    ;;
  *)
    echo "Invalid NFN_LM_HEAD_BACKWARD_CANDIDATE_FIRST='${CANDIDATE_FIRST}'" >&2
    exit 2
    ;;
esac

if [[ "${#NO_LOSS_ARG[@]}" -gt 0 && "${LOSS_BINS}" != "0" ]]; then
  echo "NFN_LM_HEAD_BACKWARD_NO_LOSS cannot be combined with NFN_LM_HEAD_BACKWARD_LOSS_BINS=${LOSS_BINS}" >&2
  exit 2
fi

BENCH_ARGS=(
  --tile-ops-lib "${TILE_OPS_LIB}"
  --baseline-symbol "${BASELINE_SYMBOL}"
  --candidate-symbol "${CANDIDATE_SYMBOL}"
  --rows "${ROWS}"
  --hidden-dim "${HIDDEN_DIM}"
  --vocab "${VOCAB}"
  --row-stride "${ROW_STRIDE}"
  --iterations "${ITERATIONS}"
  --warmup "${WARMUP}"
  "${NO_LOSS_ARG[@]}"
  --loss-bins "${LOSS_BINS}"
  "${CANDIDATE_FIRST_ARG[@]}"
  --cuda-device "${CUDA_DEVICE}"
  --json-out "${JSON_OUT}"
)

case "${DRY_RUN,,}" in
  1|true|yes|on)
    printf '%q' "${BENCH_BIN}"
    for ARG in "${BENCH_ARGS[@]}"; do
      printf ' %q' "${ARG}"
    done
    printf '\n'
    exit 0
    ;;
  0|false|no|off|"")
    ;;
  *)
    echo "Invalid NFN_LM_HEAD_BACKWARD_DRY_RUN='${DRY_RUN}'" >&2
    exit 2
    ;;
esac

if [[ ! -x "${BENCH_BIN}" || "${ROOT_DIR}/neuralfn/csrc/native_train/lm_head_backward_bench.cpp" -nt "${BENCH_BIN}" ]]; then
  bash "${ROOT_DIR}/tools/build_lm_head_backward_bench.sh" "${BENCH_BIN}" >&2
fi
if [[ ! -f "${TILE_OPS_LIB}" || "${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.cu" -nt "${TILE_OPS_LIB}" || "${ROOT_DIR}/neuralfn/csrc/tile_cuda/kernels.cu" -nt "${TILE_OPS_LIB}" ]]; then
  bash "${ROOT_DIR}/tools/build_native_train_tile_ops.sh" "${TILE_OPS_LIB}" >&2
fi

"${BENCH_BIN}" \
  "${BENCH_ARGS[@]}"

case "${REQUIRE_TRUE_FUSED,,}" in
  1|true|yes|on)
    python -c 'import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text())
if not data.get("candidate_true_fused_capability", False):
    if data.get("candidate_sequence_wrapper_only", False):
        raise SystemExit("candidate strict symbol is still sequencing CE/dHidden/dWeight; candidate_true_fused_capability is false")
    raise SystemExit("candidate_true_fused_capability is false")
' "${JSON_OUT}"
    ;;
esac

if [[ -n "${MAX_RATIO}" ]]; then
  python -c 'import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text())
ratio = float(data["candidate_to_baseline_ms_per_iter_ratio"])
limit = float(sys.argv[2])
if ratio > limit:
    raise SystemExit(f"candidate_to_baseline_ms_per_iter_ratio {ratio:.6f} exceeds limit {limit:.6f}")
' "${JSON_OUT}" "${MAX_RATIO}"
fi
