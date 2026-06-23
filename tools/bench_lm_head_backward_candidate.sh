#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_BIN="${NFN_LM_HEAD_BACKWARD_BENCH_BIN:-${ROOT_DIR}/build/lm_head_backward_bench}"
TILE_OPS_LIB="${NFN_NATIVE_TILE_OPS_LIB:-${ROOT_DIR}/build/libnfn_native_train_tile_ops.so}"
JSON_OUT="${NFN_LM_HEAD_BACKWARD_JSON_OUT:-/tmp/nfn_lm_head_backward_bench.json}"
HIDDEN_DIM="${NFN_LM_HEAD_BACKWARD_HIDDEN_DIM:-768}"
VOCAB="${NFN_LM_HEAD_BACKWARD_VOCAB:-50257}"
ROW_STRIDE="${NFN_LM_HEAD_BACKWARD_ROW_STRIDE:-50304}"
CUDA_DEVICE="${NFN_LM_HEAD_BACKWARD_CUDA_DEVICE:-0}"
BASELINE_SYMBOL="${NFN_LM_HEAD_BACKWARD_BASELINE_SYMBOL:-nfn_native_tile_lm_head_classifier_backward_cooperative_bf16_u16}"
CANDIDATE_SYMBOL="${NFN_LM_HEAD_BACKWARD_CANDIDATE_SYMBOL:-nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16}"
PROFILE="${NFN_LM_HEAD_BACKWARD_PROFILE:-smoke}"

case "${PROFILE}" in
  smoke)
    DEFAULT_ROWS=2048
    DEFAULT_ITERATIONS=5
    DEFAULT_WARMUP=1
    DEFAULT_LOSS_BINS=0
    ;;
  trainer-chunk|trainer_chunk)
    DEFAULT_ROWS=49152
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    DEFAULT_LOSS_BINS=0
    ;;
  trainer-loss-bins|trainer_loss_bins)
    DEFAULT_ROWS=49152
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    DEFAULT_LOSS_BINS=1024
    ;;
  *)
    echo "Unknown NFN_LM_HEAD_BACKWARD_PROFILE='${PROFILE}' (expected smoke, trainer-chunk, or trainer-loss-bins)" >&2
    exit 2
    ;;
esac

ROWS="${NFN_LM_HEAD_BACKWARD_ROWS:-${DEFAULT_ROWS}}"
ITERATIONS="${NFN_LM_HEAD_BACKWARD_ITERATIONS:-${DEFAULT_ITERATIONS}}"
WARMUP="${NFN_LM_HEAD_BACKWARD_WARMUP:-${DEFAULT_WARMUP}}"
LOSS_BINS="${NFN_LM_HEAD_BACKWARD_LOSS_BINS:-${DEFAULT_LOSS_BINS}}"
MAX_RATIO="${NFN_LM_HEAD_BACKWARD_MAX_RATIO:-}"
REQUIRE_TRUE_FUSED="${NFN_LM_HEAD_BACKWARD_REQUIRE_TRUE_FUSED:-0}"

if [[ ! -x "${BENCH_BIN}" || "${ROOT_DIR}/neuralfn/csrc/native_train/lm_head_backward_bench.cpp" -nt "${BENCH_BIN}" ]]; then
  bash "${ROOT_DIR}/tools/build_lm_head_backward_bench.sh" "${BENCH_BIN}" >&2
fi
if [[ ! -f "${TILE_OPS_LIB}" || "${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.cu" -nt "${TILE_OPS_LIB}" || "${ROOT_DIR}/neuralfn/csrc/tile_cuda/kernels.cu" -nt "${TILE_OPS_LIB}" ]]; then
  bash "${ROOT_DIR}/tools/build_native_train_tile_ops.sh" "${TILE_OPS_LIB}" >&2
fi

"${BENCH_BIN}" \
  --tile-ops-lib "${TILE_OPS_LIB}" \
  --baseline-symbol "${BASELINE_SYMBOL}" \
  --candidate-symbol "${CANDIDATE_SYMBOL}" \
  --rows "${ROWS}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --vocab "${VOCAB}" \
  --row-stride "${ROW_STRIDE}" \
  --iterations "${ITERATIONS}" \
  --warmup "${WARMUP}" \
  --loss-bins "${LOSS_BINS}" \
  --cuda-device "${CUDA_DEVICE}" \
  --json-out "${JSON_OUT}"

case "${REQUIRE_TRUE_FUSED,,}" in
  1|true|yes|on)
    python -c 'import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text())
if not data.get("candidate_true_fused_capability", False):
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
