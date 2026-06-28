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
RUNTIME_CONTRACT_JSON="${NFN_SM120_CUDA13_RUNTIME_CONTRACT_JSON_OUT:-/tmp/nfn_sm120_cuda13_runtime_contract.json}"
SUMMARY_JSON="${NFN_SM120_CUDA13_SUMMARY_JSON_OUT:-${NFN_SM120_CUDA13_VALIDATION_JSON_OUT:-}}"
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
if [[ -z "${SUMMARY_JSON}" ]]; then
  case "${NFN_SM120_CUDA13_RUN_BENCH:-0}" in
    1|true|TRUE|yes|YES|on|ON)
      SUMMARY_JSON="/tmp/nfn_sm120_cuda13_validation.json"
      ;;
    *)
      SUMMARY_JSON="${BENCH_JSON}"
      ;;
  esac
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

case "${NFN_SM120_CUDA13_RUN_RUNTIME_CONTRACT:-1}" in
  1|true|TRUE|yes|YES|on|ON)
    run_step \
      "${TRAIN_BIN}" \
      --backend tile-cuda \
      --tile-ops-lib "${TILE_OPS_LIB}" \
      --tinystories \
      --max-steps 1 \
      --train-transformer-lm \
      --no-checkpoint \
      --eval-every-steps 0 \
      --train-loss-every-steps 0 \
      --json-out "${RUNTIME_CONTRACT_JSON}"
    run_step "${PYTHON_BIN}" - "${RUNTIME_CONTRACT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
block_state = payload.get("block_state_layout") or {}

checks = [
    (
        payload.get("graph_editor_tensor_flow") is False,
        "runtime contract must keep graph_editor_tensor_flow=false",
    ),
    (
        payload.get("torch_required") is False,
        "runtime contract must keep torch_required=false",
    ),
    (
        payload.get("optimized_kernel_contract_passed") is True,
        "runtime contract must keep optimized_kernel_contract_passed=true",
    ),
    (
        payload.get("train_loss_host_d2h_count") == 0,
        "runtime contract must keep train_loss_host_d2h_count=0",
    ),
    (
        payload.get("linear_tk_qkv_first_use_prewarm_success_count", 0) >= 1,
        "runtime contract must keep the promoted TK QKV first-use prewarm active",
    ),
    (
        payload.get("block_backward_qkv_dinput_before_dweight_count", 0) > 0,
        "runtime contract must keep the promoted QKV dInput-before-dWeight route active",
    ),
    (
        block_state.get("layer_norm_backward_affine_row_chunk_size") == 128,
        "runtime contract must keep the promoted 128-row LayerNorm affine reduction chunk",
    ),
    (
        block_state.get("linear_backward_bias_threads_per_block") == 512,
        "runtime contract must keep the promoted 512-thread linear-bias reducer",
    ),
    (
        payload.get("lm_head_classifier_backward_path_class") == "diagnostic-cuda-graph-wrapper",
        "runtime contract must keep LM-head on the promoted CUDA Graph wrapper until a faster true-fused Tile kernel replaces it",
    ),
    (
        payload.get("lm_head_classifier_true_fused_launch_count") == 0,
        "runtime contract must not report strict true-fused LM-head launches on the default path",
    ),
]

failed = [message for ok, message in checks if not ok]
if failed:
    print(f"SM120 CUDA 13.3 runtime contract failed for {path}:", file=sys.stderr)
    for message in failed:
        print(f"  - {message}", file=sys.stderr)
    sys.exit(2)

print(f"SM120 CUDA 13.3 runtime contract passed for {path}")
PY
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "Unsupported NFN_SM120_CUDA13_RUN_RUNTIME_CONTRACT=${NFN_SM120_CUDA13_RUN_RUNTIME_CONTRACT}" >&2
    exit 2
    ;;
esac

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
      NFN_SM120_NATIVE_REQUIRE_LM_HEAD_TRUE_FUSED="${NFN_SM120_CUDA13_REQUIRE_LM_HEAD_TRUE_FUSED:-0}" \
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
true_fused_target = payload.get("native_lm_head_true_fused_target") or {}
true_fused_gate = payload.get("native_lm_head_true_fused_gate") or {}

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
        true_fused_target.get("status") == "diagnostic-cuda-graph-wrapper"
        and true_fused_target.get("graph_wrapper_active") is True
        and true_fused_target.get("next_required_path_class")
        == "strict-true-fused-tile-kernel",
        "benchmark JSON must explicitly report that strict true-fused LM-head Tile work remains",
    ),
    (
        true_fused_gate.get("enabled") is False
        and true_fused_gate.get("passed") is True,
        "default CUDA 13 validation must leave the strict LM-head true-fused gate disabled unless NFN_SM120_CUDA13_REQUIRE_LM_HEAD_TRUE_FUSED=1 is set",
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

run_step "${PYTHON_BIN}" - "${SUMMARY_JSON}" "${LM_HEAD_JSON}" "${BENCH_JSON}" "${PARITY_JSON}" "${RUNTIME_CONTRACT_JSON}" "${TRAIN_BIN}" "${TILE_OPS_LIB}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path = Path(sys.argv[1])
lm_head_path = Path(sys.argv[2])
bench_path = Path(sys.argv[3])
parity_path = Path(sys.argv[4])
runtime_contract_path = Path(sys.argv[5])
train_bin = sys.argv[6]
tile_ops_lib = sys.argv[7]

def enabled(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}

def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"failed to parse {path}: {exc}"}

lm_head = load_json(lm_head_path)
bench = load_json(bench_path) if enabled("NFN_SM120_CUDA13_RUN_BENCH", "0") else None
parity = load_json(parity_path) if enabled("NFN_SM120_CUDA13_RUN_PARITY", "0") else None
runtime_contract = load_json(runtime_contract_path) if enabled("NFN_SM120_CUDA13_RUN_RUNTIME_CONTRACT", "1") else None

summary = {
    "status": "passed",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "cuda_validator": "tools/validate_sm120_cuda13.sh",
    "train_bin": train_bin,
    "tile_ops_lib": tile_ops_lib,
    "run_no_torch": enabled("NFN_SM120_CUDA13_RUN_NO_TORCH", "1"),
    "run_runtime_contract": enabled("NFN_SM120_CUDA13_RUN_RUNTIME_CONTRACT", "1"),
    "run_lm_head_bench": enabled("NFN_SM120_CUDA13_RUN_LM_HEAD_BENCH", "1"),
    "run_pytest": enabled("NFN_SM120_CUDA13_RUN_PYTEST", "1"),
    "run_bench": enabled("NFN_SM120_CUDA13_RUN_BENCH", "0"),
    "run_parity": enabled("NFN_SM120_CUDA13_RUN_PARITY", "0"),
    "artifacts": {
        "summary_json": str(summary_path),
        "lm_head_json": str(lm_head_path) if lm_head_path.exists() else "",
        "bench_json": str(bench_path) if bench_path.exists() and bench is not None else "",
        "parity_json": str(parity_path) if parity_path.exists() and parity is not None else "",
        "runtime_contract_json": str(runtime_contract_path)
        if runtime_contract_path.exists() and runtime_contract is not None
        else "",
    },
}

if isinstance(lm_head, dict):
    candidate = lm_head.get("candidate") or {}
    summary["lm_head_true_fused_status"] = {
        "candidate_path_class": lm_head.get("candidate_path_class"),
        "candidate_true_fused_capability": lm_head.get("candidate_true_fused_capability"),
        "candidate_true_fused_production_ready": lm_head.get("candidate_true_fused_production_ready"),
        "true_fused_replacement_required": lm_head.get("true_fused_replacement_required"),
        "next_required_path_class": lm_head.get("next_required_path_class"),
        "candidate_ms_per_iter": candidate.get("ms_per_iter"),
        "candidate_to_baseline_ms_per_iter_ratio": lm_head.get("candidate_to_baseline_ms_per_iter_ratio"),
        "candidate_true_fused_launch_count": candidate.get("true_fused_launch_count"),
        "candidate_graph_replay_count": candidate.get("graph_replay_count"),
        "candidate_graph_cache_hit_count": candidate.get("graph_cache_hit_count"),
    }
elif lm_head is not None:
    summary["lm_head_true_fused_status"] = lm_head

if isinstance(bench, dict):
    summary["native_bench_status"] = {
        "metric_ratio_gates": bench.get("metric_ratio_gates"),
        "native_lm_head_true_fused_target": bench.get("native_lm_head_true_fused_target"),
        "native_lm_head_true_fused_gate": bench.get("native_lm_head_true_fused_gate"),
    }

if isinstance(parity, dict):
    summary["parity_status"] = {
        "metric_ratio_gates": parity.get("metric_ratio_gates"),
        "candidate_over_reference": parity.get("candidate_over_reference"),
    }

if isinstance(runtime_contract, dict):
    block_state = runtime_contract.get("block_state_layout") or {}
    summary["runtime_contract_status"] = {
        "graph_editor_tensor_flow": runtime_contract.get("graph_editor_tensor_flow"),
        "torch_required": runtime_contract.get("torch_required"),
        "optimized_kernel_contract_passed": runtime_contract.get("optimized_kernel_contract_passed"),
        "train_loss_host_d2h_count": runtime_contract.get("train_loss_host_d2h_count"),
        "linear_tk_qkv_first_use_prewarm_success_count": runtime_contract.get(
            "linear_tk_qkv_first_use_prewarm_success_count"
        ),
        "block_backward_qkv_dinput_before_dweight_count": runtime_contract.get(
            "block_backward_qkv_dinput_before_dweight_count"
        ),
        "layer_norm_backward_affine_row_chunk_size": block_state.get(
            "layer_norm_backward_affine_row_chunk_size"
        ),
        "linear_backward_bias_threads_per_block": block_state.get(
            "linear_backward_bias_threads_per_block"
        ),
        "lm_head_classifier_backward_path_class": runtime_contract.get(
            "lm_head_classifier_backward_path_class"
        ),
        "lm_head_classifier_true_fused_launch_count": runtime_contract.get(
            "lm_head_classifier_true_fused_launch_count"
        ),
    }

summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"CUDA 13.3 SM120 validation summary written to {summary_path}")
PY

printf '\nCUDA 13.3 SM120 validation passed.\n'
