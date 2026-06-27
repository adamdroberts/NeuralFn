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
BASELINE_SYMBOL_OVERRIDE="${NFN_LM_HEAD_BACKWARD_BASELINE_SYMBOL:-}"
CANDIDATE_SYMBOL_OVERRIDE="${NFN_LM_HEAD_BACKWARD_CANDIDATE_SYMBOL:-}"
PROFILE="${NFN_LM_HEAD_BACKWARD_PROFILE:-smoke}"
DEFAULT_BASELINE_SYMBOL="nfn_native_tile_lm_head_classifier_backward_cooperative_bf16_u16"
DEFAULT_CANDIDATE_SYMBOL="nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16"
REJECTED_PROFILE=""
REJECTED_REASON=""

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
    DEFAULT_REQUIRE_TRUE_FUSED=0
    ;;
  trainer-chunk-strict|trainer_chunk_strict|strict-trainer-chunk|strict_trainer_chunk)
    DEFAULT_ROWS=32768
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=1
    DEFAULT_REQUIRE_TRUE_FUSED=1
    ;;
  trainer-chunk-true-fused|trainer_chunk_true_fused|true-fused-trainer-chunk|true_fused_trainer_chunk)
    DEFAULT_ROWS=32768
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=1
    DEFAULT_REQUIRE_TRUE_FUSED=1
    DEFAULT_MAX_RATIO=1.000
    DEFAULT_MAX_REFERENCE_RATIO=1.000
    DEFAULT_MAX_CUBLASLT_REFERENCE_RATIO=1.000
    REJECTED_PROFILE="${PROFILE}"
    REJECTED_REASON="Production-shape focused strict true-fused LM-head profile. It forces NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE=1 and NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION=1 so the focused trainer-chunk microbench measures the cooperative single-kernel CE+dHidden+dWeight body. The profile defaults NFN_LM_HEAD_BACKWARD_MAX_RATIO, NFN_LM_HEAD_BACKWARD_MAX_REFERENCE_RATIO, and NFN_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_RATIO to 1.000; keep rejected until this focused gate proves candidate/current-wrapper and candidate/reference parity."
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE:-1}"
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION:-1}"
    ;;
  true-fused-cooperative-smoke|true_fused_cooperative_smoke|strict-true-fused-smoke|strict_true_fused_smoke)
    DEFAULT_ROWS=4
    DEFAULT_ITERATIONS=1
    DEFAULT_WARMUP=0
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=0
    DEFAULT_REQUIRE_TRUE_FUSED=1
    if [[ -z "${NFN_LM_HEAD_BACKWARD_HIDDEN_DIM+x}" ]]; then
      HIDDEN_DIM=8
    fi
    if [[ -z "${NFN_LM_HEAD_BACKWARD_VOCAB+x}" ]]; then
      VOCAB=16
    fi
    if [[ -z "${NFN_LM_HEAD_BACKWARD_ROW_STRIDE+x}" ]]; then
      ROW_STRIDE=16
    fi
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE:-1}"
    ;;
  trainer-chunk-cublaslt|trainer_chunk_cublaslt|trainer-cublaslt|trainer_cublaslt)
    DEFAULT_ROWS=32768
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=1
    DEFAULT_REQUIRE_TRUE_FUSED=0
    DEFAULT_CANDIDATE_SYMBOL="nfn_native_tile_lm_head_classifier_backward_cooperative_cublaslt_bf16_u16"
    REJECTED_PROFILE="${PROFILE}"
    REJECTED_REASON="CUDA 13.3 dedicated RTX 5090 trainer-chunk evidence rejects this cuBLASLt LM-head route: 37.070129 ms/iter vs 25.271233 ms/iter baseline, candidate/baseline ratio 1.466890."
    ;;
  trainer-row-loss-cublaslt|trainer_row_loss_cublaslt)
    DEFAULT_ROWS=32768
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=0
    DEFAULT_REQUIRE_TRUE_FUSED=0
    DEFAULT_CANDIDATE_SYMBOL="nfn_native_tile_lm_head_classifier_backward_cooperative_cublaslt_bf16_u16"
    REJECTED_PROFILE="${PROFILE}"
    REJECTED_REASON="CUDA 13.3 dedicated RTX 5090 evidence rejects the cuBLASLt LM-head route; keep this profile for intentional diagnostics only."
    ;;
  trainer-row-loss|trainer_row_loss)
    DEFAULT_ROWS=32768
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=0
    DEFAULT_REQUIRE_TRUE_FUSED=0
    ;;
  trainer-loss-bins|trainer_loss_bins)
    DEFAULT_ROWS=32768
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=1
    DEFAULT_LOSS_BINS=1024
    DEFAULT_NO_LOSS=0
    DEFAULT_REQUIRE_TRUE_FUSED=0
    ;;
  *)
    echo "Unknown NFN_LM_HEAD_BACKWARD_PROFILE='${PROFILE}' (expected smoke, trainer-chunk, trainer-chunk-strict, trainer-chunk-true-fused, true-fused-cooperative-smoke, trainer-chunk-cublaslt, trainer-row-loss, trainer-row-loss-cublaslt, or trainer-loss-bins)" >&2
    exit 2
    ;;
esac

BASELINE_SYMBOL="${BASELINE_SYMBOL_OVERRIDE:-${DEFAULT_BASELINE_SYMBOL}}"
CANDIDATE_SYMBOL="${CANDIDATE_SYMBOL_OVERRIDE:-${DEFAULT_CANDIDATE_SYMBOL}}"

ROWS="${NFN_LM_HEAD_BACKWARD_ROWS:-${DEFAULT_ROWS}}"
ITERATIONS="${NFN_LM_HEAD_BACKWARD_ITERATIONS:-${DEFAULT_ITERATIONS}}"
WARMUP="${NFN_LM_HEAD_BACKWARD_WARMUP:-${DEFAULT_WARMUP}}"
LOSS_BINS="${NFN_LM_HEAD_BACKWARD_LOSS_BINS:-${DEFAULT_LOSS_BINS}}"
NO_LOSS="${NFN_LM_HEAD_BACKWARD_NO_LOSS:-${DEFAULT_NO_LOSS}}"
MAX_RATIO="${NFN_LM_HEAD_BACKWARD_MAX_RATIO:-${DEFAULT_MAX_RATIO:-}}"
MAX_REFERENCE_RATIO="${NFN_LM_HEAD_BACKWARD_MAX_REFERENCE_RATIO:-${DEFAULT_MAX_REFERENCE_RATIO:-}}"
MAX_REFERENCE_WITH_LOGITS_RATIO="${NFN_LM_HEAD_BACKWARD_MAX_REFERENCE_WITH_LOGITS_RATIO:-}"
MAX_CUBLASLT_REFERENCE_RATIO="${NFN_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_RATIO:-${DEFAULT_MAX_CUBLASLT_REFERENCE_RATIO:-}}"
MAX_CUBLASLT_REFERENCE_WITH_LOGITS_RATIO="${NFN_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_WITH_LOGITS_RATIO:-}"
REQUIRE_TRUE_FUSED="${NFN_LM_HEAD_BACKWARD_REQUIRE_TRUE_FUSED:-${DEFAULT_REQUIRE_TRUE_FUSED:-0}}"
CANDIDATE_FIRST="${NFN_LM_HEAD_BACKWARD_CANDIDATE_FIRST:-0}"
DRY_RUN="${NFN_LM_HEAD_BACKWARD_DRY_RUN:-0}"
ALLOW_REJECTED_PROFILE="${NFN_LM_HEAD_BACKWARD_ALLOW_REJECTED_PROFILE:-0}"

if [[ -n "${REJECTED_PROFILE}" ]]; then
  case "${DRY_RUN,,}:${ALLOW_REJECTED_PROFILE,,}" in
    1:*|true:*|yes:*|on:*|*:1|*:true|*:yes|*:on)
      ;;
    *)
      echo "NFN_LM_HEAD_BACKWARD_PROFILE=${REJECTED_PROFILE} is a rejected LM-head candidate profile." >&2
      echo "${REJECTED_REASON}" >&2
      echo "Set NFN_LM_HEAD_BACKWARD_ALLOW_REJECTED_PROFILE=1 to rerun it intentionally, or NFN_LM_HEAD_BACKWARD_DRY_RUN=1 to inspect the command only." >&2
      exit 2
      ;;
  esac
fi

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

snapshot_selected_gpu_load_json() {
  local phase="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    NFN_GPU_PHASE="${phase}" \
    NFN_GPU_VISIBLE="${CUDA_VISIBLE_DEVICES-}" \
    NFN_GPU_DEVICE="${CUDA_DEVICE}" \
    python -c 'import json, os
print(json.dumps({
    "phase": os.environ["NFN_GPU_PHASE"],
    "available": False,
    "reason": "nvidia-smi-not-found",
    "cuda_visible_devices": os.environ.get("NFN_GPU_VISIBLE", ""),
    "cuda_device": os.environ.get("NFN_GPU_DEVICE", ""),
}, sort_keys=True))
'
    return 0
  fi
  local selected_gpu="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"
  selected_gpu="${selected_gpu%%,*}"
  if [[ -z "${selected_gpu}" ]]; then
    selected_gpu="${CUDA_DEVICE}"
  fi
  local gpu_query=""
  local gpu_status=0
  if gpu_query="$(nvidia-smi -i "${selected_gpu}" --query-gpu=index,name,uuid,display_active,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)"; then
    :
  else
    gpu_status=$?
  fi
  local compute_query=""
  local compute_status=0
  if compute_query="$(nvidia-smi -i "${selected_gpu}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null)"; then
    :
  else
    compute_status=$?
  fi
  NFN_GPU_PHASE="${phase}" \
  NFN_GPU_VISIBLE="${CUDA_VISIBLE_DEVICES-}" \
  NFN_GPU_DEVICE="${CUDA_DEVICE}" \
  NFN_GPU_SELECTED="${selected_gpu}" \
  NFN_GPU_QUERY="${gpu_query}" \
  NFN_GPU_QUERY_STATUS="${gpu_status}" \
  NFN_GPU_COMPUTE="${compute_query}" \
  NFN_GPU_COMPUTE_STATUS="${compute_status}" \
  python -c 'import json, os

def parse_gpu(row):
    parts = [part.strip() for part in row.split(",", 6)]
    if len(parts) < 7:
        return {}
    index, name, uuid, display_active, utilization, memory_used, memory_total = parts
    def as_int(value):
        try:
            return int(value)
        except ValueError:
            return None
    return {
        "index": index,
        "name": name,
        "uuid": uuid,
        "display_active": display_active,
        "utilization_gpu_pct": as_int(utilization),
        "memory_used_mib": as_int(memory_used),
        "memory_total_mib": as_int(memory_total),
    }

compute_rows = []
for row in os.environ.get("NFN_GPU_COMPUTE", "").splitlines():
    parts = [part.strip() for part in row.split(",", 2)]
    if len(parts) == 3:
        pid, process_name, used_memory = parts
        try:
            used_memory_mib = int(used_memory)
        except ValueError:
            used_memory_mib = None
        compute_rows.append({
            "pid": pid,
            "process_name": process_name,
            "used_memory_mib": used_memory_mib,
        })

gpu_status = int(os.environ.get("NFN_GPU_QUERY_STATUS", "0") or 0)
compute_status = int(os.environ.get("NFN_GPU_COMPUTE_STATUS", "0") or 0)
print(json.dumps({
    "phase": os.environ["NFN_GPU_PHASE"],
    "available": gpu_status == 0,
    "cuda_visible_devices": os.environ.get("NFN_GPU_VISIBLE", ""),
    "cuda_device": os.environ.get("NFN_GPU_DEVICE", ""),
    "selected_gpu": os.environ.get("NFN_GPU_SELECTED", ""),
    "gpu_query_status": gpu_status,
    "compute_query_status": compute_status,
    "gpu": parse_gpu(os.environ.get("NFN_GPU_QUERY", "").splitlines()[0] if os.environ.get("NFN_GPU_QUERY", "").splitlines() else ""),
    "compute_processes": compute_rows,
    "compute_process_count": len(compute_rows),
}, sort_keys=True))
'
}

merge_gpu_load_context_json() {
  local before_json="$1"
  local after_json="$2"
  if [[ ! -f "${JSON_OUT}" ]]; then
    return 0
  fi
  python - "${JSON_OUT}" "${before_json}" "${after_json}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
data["gpu_load_context"] = {
    "before": json.loads(sys.argv[2]),
    "after": json.loads(sys.argv[3]),
}
path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
PY
}

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

case "${REQUIRE_TRUE_FUSED,,}" in
  1|true|yes|on)
    REQUIRE_TRUE_FUSED_ARG=(--require-true-fused-candidate)
    ;;
  0|false|no|off|"")
    REQUIRE_TRUE_FUSED_ARG=()
    ;;
  *)
    echo "Invalid NFN_LM_HEAD_BACKWARD_REQUIRE_TRUE_FUSED='${REQUIRE_TRUE_FUSED}'" >&2
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
  "${REQUIRE_TRUE_FUSED_ARG[@]}"
  --cuda-device "${CUDA_DEVICE}"
  --json-out "${JSON_OUT}"
)

case "${DRY_RUN,,}" in
  1|true|yes|on)
    DRY_RUN_ENV_PREFIX=()
    for ENV_NAME in \
      NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE \
      NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION; do
      if [[ -n "${!ENV_NAME+x}" ]]; then
        DRY_RUN_ENV_PREFIX+=("${ENV_NAME}=${!ENV_NAME}")
      fi
    done
    if [[ "${#DRY_RUN_ENV_PREFIX[@]}" -gt 0 ]]; then
      printf '%q' env
      for ARG in "${DRY_RUN_ENV_PREFIX[@]}"; do
        printf ' %q' "${ARG}"
      done
      printf ' '
    fi
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

BENCH_DEPS=(
  "${ROOT_DIR}/neuralfn/csrc/native_train/lm_head_backward_bench.cpp"
  "${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.h"
)
REBUILD_BENCH=0
if [[ ! -x "${BENCH_BIN}" ]]; then
  REBUILD_BENCH=1
else
  for DEP in "${BENCH_DEPS[@]}"; do
    if [[ "${DEP}" -nt "${BENCH_BIN}" ]]; then
      REBUILD_BENCH=1
      break
    fi
  done
fi
if [[ "${REBUILD_BENCH}" == "1" ]]; then
  bash "${ROOT_DIR}/tools/build_lm_head_backward_bench.sh" "${BENCH_BIN}" >&2
fi
TILE_OPS_DEPS=(
  "${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.cu"
  "${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.h"
  "${ROOT_DIR}/neuralfn/csrc/tile_cuda/kernels.cu"
  "${ROOT_DIR}/tools/build_native_train_tile_ops.sh"
)
REBUILD_TILE_OPS=0
if [[ ! -f "${TILE_OPS_LIB}" ]]; then
  REBUILD_TILE_OPS=1
else
  for DEP in "${TILE_OPS_DEPS[@]}"; do
    if [[ "${DEP}" -nt "${TILE_OPS_LIB}" ]]; then
      REBUILD_TILE_OPS=1
      break
    fi
  done
fi
if [[ "${REBUILD_TILE_OPS}" == "1" ]]; then
  bash "${ROOT_DIR}/tools/build_native_train_tile_ops.sh" "${TILE_OPS_LIB}" >&2
fi

emit_true_fused_requirement_message() {
  if [[ ! -f "${JSON_OUT}" ]]; then
    return 0
  fi
  python -c 'import json, pathlib, sys
try:
    data = json.loads(pathlib.Path(sys.argv[1]).read_text())
except Exception:
    raise SystemExit(0)
if data.get("candidate_true_fused_capability", False):
    raise SystemExit(0)
required = data.get("next_required_kernel_body")
required_symbol = data.get("next_required_symbol")
required_capability = data.get("next_required_capability_symbol")
required_path = data.get("next_required_path_class")
if not (required or required_symbol or required_capability or required_path):
    raise SystemExit(0)
print(
    "LM-head true-fused replacement required: "
    f"next_required_symbol={required_symbol or 'unknown'}, "
    f"next_required_capability_symbol={required_capability or 'unknown'}, "
    f"next_required_path_class={required_path or 'unknown'}, "
    f"next_required_kernel_body={required or 'unknown'}",
    file=sys.stderr,
)
' "${JSON_OUT}"
}

GPU_LOAD_BEFORE="$(snapshot_selected_gpu_load_json before)"
BENCH_STATUS=0
"${BENCH_BIN}" "${BENCH_ARGS[@]}" || BENCH_STATUS=$?
GPU_LOAD_AFTER="$(snapshot_selected_gpu_load_json after)"
merge_gpu_load_context_json "${GPU_LOAD_BEFORE}" "${GPU_LOAD_AFTER}"
if [[ "${BENCH_STATUS}" != "0" ]]; then
  case "${REQUIRE_TRUE_FUSED,,}" in
    1|true|yes|on)
      emit_true_fused_requirement_message
      ;;
  esac
  exit "${BENCH_STATUS}"
fi

case "${REQUIRE_TRUE_FUSED,,}" in
  1|true|yes|on)
    python -c 'import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text())
if not data.get("candidate_true_fused_capability", False):
    required = data.get("next_required_kernel_body")
    required_symbol = data.get("next_required_symbol")
    required_capability = data.get("next_required_capability_symbol")
    required_path = data.get("next_required_path_class")
    suffix = ""
    if required or required_symbol or required_capability or required_path:
        suffix = (
            f"; next_required_symbol={required_symbol or 'unknown'}, "
            f"next_required_capability_symbol={required_capability or 'unknown'}, "
            f"next_required_path_class={required_path or 'unknown'}, "
            f"next_required_kernel_body={required or 'unknown'}"
        )
    if data.get("candidate_sequence_wrapper_only", False):
        raise SystemExit("candidate strict symbol is still sequencing CE/dHidden/dWeight; candidate_true_fused_capability is false" + suffix)
    if data.get("candidate_cuda_graph_wrapper_only", False):
        raise SystemExit("candidate strict symbol is a CUDA Graph wrapper around CE/dHidden/dWeight; candidate_true_fused_capability is false" + suffix)
    raise SystemExit("candidate_true_fused_capability is false" + suffix)
candidate = data.get("candidate", {})
true_fused_launch_count = int(candidate.get("true_fused_launch_count", 0) or 0)
if true_fused_launch_count <= 0:
    raise SystemExit("candidate_true_fused_capability is true but candidate.true_fused_launch_count is zero")
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

check_json_ratio() {
  local key="$1"
  local limit="$2"
  if [[ -z "${limit}" ]]; then
    return 0
  fi
  python -c 'import json, pathlib, sys
path, key, limit_raw = sys.argv[1], sys.argv[2], sys.argv[3]
data = json.loads(pathlib.Path(path).read_text())
ratio = float(data[key])
limit = float(limit_raw)
if ratio > limit:
    raise SystemExit(f"{key} {ratio:.6f} exceeds limit {limit:.6f}")
' "${JSON_OUT}" "${key}" "${limit}"
}

check_json_ratio \
  "candidate_to_reference_summed_ms_per_iter_ratio" \
  "${MAX_REFERENCE_RATIO}"
check_json_ratio \
  "candidate_to_reference_summed_with_logits_ms_per_iter_ratio" \
  "${MAX_REFERENCE_WITH_LOGITS_RATIO}"
check_json_ratio \
  "candidate_to_reference_cublaslt_summed_ms_per_iter_ratio" \
  "${MAX_CUBLASLT_REFERENCE_RATIO}"
check_json_ratio \
  "candidate_to_reference_cublaslt_summed_with_logits_ms_per_iter_ratio" \
  "${MAX_CUBLASLT_REFERENCE_WITH_LOGITS_RATIO}"
