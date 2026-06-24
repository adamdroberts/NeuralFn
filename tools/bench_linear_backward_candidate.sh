#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_BIN="${NFN_LINEAR_BACKWARD_BENCH_BIN:-${ROOT_DIR}/build/linear_backward_bench}"
TILE_OPS_LIB="${NFN_NATIVE_TILE_OPS_LIB:-${ROOT_DIR}/build/libnfn_native_train_tile_ops.so}"
JSON_OUT="${NFN_LINEAR_BACKWARD_JSON_OUT:-/tmp/nfn_linear_backward_bench.json}"
CUDA_VISIBLE_DEVICES_VALUE="${NFN_LINEAR_BACKWARD_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-dedicated}}"
CUDA_DEVICE_RAW="${NFN_LINEAR_BACKWARD_CUDA_DEVICE:-auto}"
PROFILE="${NFN_LINEAR_BACKWARD_PROFILE:-smoke-dinput}"

case "${PROFILE}" in
  smoke-dinput|smoke_dinput)
    DEFAULT_OPERATION="dinput-strided"
    DEFAULT_ROWS=512
    DEFAULT_INPUT_DIM=128
    DEFAULT_OUTPUT_DIM=256
    DEFAULT_GRAD_OUT_ROW_STRIDE=256
    DEFAULT_ITERATIONS=5
    DEFAULT_WARMUP=1
    ;;
  smoke-dweight|smoke_dweight)
    DEFAULT_OPERATION="dweight-strided"
    DEFAULT_ROWS=512
    DEFAULT_INPUT_DIM=128
    DEFAULT_OUTPUT_DIM=256
    DEFAULT_GRAD_OUT_ROW_STRIDE=256
    DEFAULT_ITERATIONS=5
    DEFAULT_WARMUP=1
    ;;
  mlp-proj-dinput|mlp_proj_dinput)
    DEFAULT_OPERATION="dinput-strided"
    DEFAULT_ROWS=65536
    DEFAULT_INPUT_DIM=3072
    DEFAULT_OUTPUT_DIM=768
    DEFAULT_GRAD_OUT_ROW_STRIDE=768
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    ;;
  mlp-proj-dweight|mlp_proj_dweight)
    DEFAULT_OPERATION="dweight-strided"
    DEFAULT_ROWS=65536
    DEFAULT_INPUT_DIM=3072
    DEFAULT_OUTPUT_DIM=768
    DEFAULT_GRAD_OUT_ROW_STRIDE=768
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    ;;
  mlp-fc-dinput|mlp_fc_dinput)
    DEFAULT_OPERATION="dinput-strided"
    DEFAULT_ROWS=65536
    DEFAULT_INPUT_DIM=768
    DEFAULT_OUTPUT_DIM=3072
    DEFAULT_GRAD_OUT_ROW_STRIDE=3072
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    ;;
  mlp-fc-dweight|mlp_fc_dweight)
    DEFAULT_OPERATION="dweight-strided"
    DEFAULT_ROWS=65536
    DEFAULT_INPUT_DIM=768
    DEFAULT_OUTPUT_DIM=3072
    DEFAULT_GRAD_OUT_ROW_STRIDE=3072
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    ;;
  qkv-dinput|qkv_dinput)
    DEFAULT_OPERATION="dinput-strided"
    DEFAULT_ROWS=65536
    DEFAULT_INPUT_DIM=768
    DEFAULT_OUTPUT_DIM=2304
    DEFAULT_GRAD_OUT_ROW_STRIDE=2304
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    ;;
  qkv-dweight|qkv_dweight)
    DEFAULT_OPERATION="dweight-strided"
    DEFAULT_ROWS=65536
    DEFAULT_INPUT_DIM=768
    DEFAULT_OUTPUT_DIM=2304
    DEFAULT_GRAD_OUT_ROW_STRIDE=2304
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    ;;
  attn-proj-dinput|attn_proj_dinput)
    DEFAULT_OPERATION="dinput-strided"
    DEFAULT_ROWS=65536
    DEFAULT_INPUT_DIM=768
    DEFAULT_OUTPUT_DIM=768
    DEFAULT_GRAD_OUT_ROW_STRIDE=768
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    ;;
  attn-proj-dweight|attn_proj_dweight)
    DEFAULT_OPERATION="dweight-strided"
    DEFAULT_ROWS=65536
    DEFAULT_INPUT_DIM=768
    DEFAULT_OUTPUT_DIM=768
    DEFAULT_GRAD_OUT_ROW_STRIDE=768
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    ;;
  lm-head-dinput|lm_head_dinput)
    DEFAULT_OPERATION="dinput-strided"
    DEFAULT_ROWS=32768
    DEFAULT_INPUT_DIM=768
    DEFAULT_OUTPUT_DIM=50257
    DEFAULT_GRAD_OUT_ROW_STRIDE=50304
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    ;;
  lm-head-dinput-cublaslt|lm_head_dinput_cublaslt)
    DEFAULT_OPERATION="dinput-strided"
    DEFAULT_ROWS=32768
    DEFAULT_INPUT_DIM=768
    DEFAULT_OUTPUT_DIM=50257
    DEFAULT_GRAD_OUT_ROW_STRIDE=50304
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    DEFAULT_CANDIDATE_SYMBOL="nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_strided_cublaslt_float32"
    ;;
  lm-head-dweight|lm_head_dweight)
    DEFAULT_OPERATION="dweight-strided"
    DEFAULT_ROWS=32768
    DEFAULT_INPUT_DIM=768
    DEFAULT_OUTPUT_DIM=50257
    DEFAULT_GRAD_OUT_ROW_STRIDE=50304
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    ;;
  lm-head-dweight-cublaslt|lm_head_dweight_cublaslt)
    DEFAULT_OPERATION="dweight-strided"
    DEFAULT_ROWS=32768
    DEFAULT_INPUT_DIM=768
    DEFAULT_OUTPUT_DIM=50257
    DEFAULT_GRAD_OUT_ROW_STRIDE=50304
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    DEFAULT_CANDIDATE_SYMBOL="nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_cublaslt_float32_beta"
    ;;
  *)
    echo "Unknown NFN_LINEAR_BACKWARD_PROFILE='${PROFILE}'" >&2
    echo "Expected smoke-dinput, smoke-dweight, mlp-proj-dinput, mlp-proj-dweight, mlp-fc-dinput, mlp-fc-dweight, qkv-dinput, qkv-dweight, attn-proj-dinput, attn-proj-dweight, lm-head-dinput, lm-head-dinput-cublaslt, lm-head-dweight, or lm-head-dweight-cublaslt" >&2
    exit 2
    ;;
esac

OPERATION="${NFN_LINEAR_BACKWARD_OPERATION:-${DEFAULT_OPERATION}}"
ROWS="${NFN_LINEAR_BACKWARD_ROWS:-${DEFAULT_ROWS}}"
INPUT_DIM="${NFN_LINEAR_BACKWARD_INPUT_DIM:-${DEFAULT_INPUT_DIM}}"
OUTPUT_DIM="${NFN_LINEAR_BACKWARD_OUTPUT_DIM:-${DEFAULT_OUTPUT_DIM}}"
GRAD_OUT_ROW_STRIDE="${NFN_LINEAR_BACKWARD_GRAD_OUT_ROW_STRIDE:-${DEFAULT_GRAD_OUT_ROW_STRIDE}}"
ITERATIONS="${NFN_LINEAR_BACKWARD_ITERATIONS:-${DEFAULT_ITERATIONS}}"
WARMUP="${NFN_LINEAR_BACKWARD_WARMUP:-${DEFAULT_WARMUP}}"
BETA="${NFN_LINEAR_BACKWARD_BETA:-0.0}"
MAX_RATIO="${NFN_LINEAR_BACKWARD_MAX_RATIO:-}"
REQUIRE_ROUTE_CHANGE="${NFN_LINEAR_BACKWARD_REQUIRE_ROUTE_CHANGE:-0}"
CANDIDATE_FIRST="${NFN_LINEAR_BACKWARD_CANDIDATE_FIRST:-0}"
DRY_RUN="${NFN_LINEAR_BACKWARD_DRY_RUN:-0}"

case "${OPERATION}" in
  dinput-strided)
    DEFAULT_BASELINE_SYMBOL="nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_strided_float32"
    ;;
  dweight-strided)
    DEFAULT_BASELINE_SYMBOL="nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta"
    ;;
  *)
    echo "Unknown NFN_LINEAR_BACKWARD_OPERATION='${OPERATION}' (expected dinput-strided or dweight-strided)" >&2
    exit 2
    ;;
esac

BASELINE_SYMBOL="${NFN_LINEAR_BACKWARD_BASELINE_SYMBOL:-${DEFAULT_BASELINE_SYMBOL}}"
CANDIDATE_SYMBOL="${NFN_LINEAR_BACKWARD_CANDIDATE_SYMBOL:-${DEFAULT_CANDIDATE_SYMBOL:-${BASELINE_SYMBOL}}}"
CANDIDATE_FIRST_ARGS=()
case "${CANDIDATE_FIRST,,}" in
  1|true|yes|on)
    CANDIDATE_FIRST_ARGS=(--candidate-first)
    ;;
esac

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
  "auto")
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

BENCH_ARGS=(
  --tile-ops-lib "${TILE_OPS_LIB}"
  --operation "${OPERATION}"
  --baseline-symbol "${BASELINE_SYMBOL}"
  --candidate-symbol "${CANDIDATE_SYMBOL}"
  --rows "${ROWS}"
  --input-dim "${INPUT_DIM}"
  --output-dim "${OUTPUT_DIM}"
  --grad-out-row-stride "${GRAD_OUT_ROW_STRIDE}"
  --iterations "${ITERATIONS}"
  --warmup "${WARMUP}"
  --beta "${BETA}"
  --cuda-device "${CUDA_DEVICE}"
  --json-out "${JSON_OUT}"
  "${CANDIDATE_FIRST_ARGS[@]}"
)

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  printf '%q' "${BENCH_BIN}"
  for ARG in "${BENCH_ARGS[@]}"; do
    printf ' %q' "${ARG}"
  done
  printf '\n'
  exit 0
fi

if [[ ! -x "${BENCH_BIN}" || "${ROOT_DIR}/neuralfn/csrc/native_train/linear_backward_bench.cpp" -nt "${BENCH_BIN}" ]]; then
  bash "${ROOT_DIR}/tools/build_linear_backward_bench.sh" "${BENCH_BIN}" >&2
fi
if [[ ! -f "${TILE_OPS_LIB}" || "${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.cu" -nt "${TILE_OPS_LIB}" || "${ROOT_DIR}/neuralfn/csrc/tile_cuda/kernels.cu" -nt "${TILE_OPS_LIB}" ]]; then
  bash "${ROOT_DIR}/tools/build_native_train_tile_ops.sh" "${TILE_OPS_LIB}" >&2
fi

"${BENCH_BIN}" \
  "${BENCH_ARGS[@]}"

case "${REQUIRE_ROUTE_CHANGE,,}" in
  1|true|yes|on)
    python -c 'import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text())
if not data.get("candidate_symbol_changed", False):
    raise SystemExit("candidate_symbol_changed is false; candidate and baseline symbols are identical")
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
