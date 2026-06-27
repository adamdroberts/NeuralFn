#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLM_KITTENS_ROOT="${LLM_KITTENS_ROOT:-/mnt/disk2/dev/open-source/llm.kittens}"
LLM_KITTENS_TRAIN_BIN="${LLM_KITTENS_TRAIN_BIN:-$LLM_KITTENS_ROOT/train_gpt2cu}"
LLM_KITTENS_TINYSTORIES_DIR="${LLM_KITTENS_TINYSTORIES_DIR:-$LLM_KITTENS_ROOT/dev/data/tinystories}"
NFN_NATIVE_GPT_TRAIN_BIN_EXPLICIT="${NFN_NATIVE_GPT_TRAIN_BIN+x}"
if [[ -z "${NFN_NATIVE_GPT_TRAIN_BIN-}" && -x "$ROOT_DIR/build/nfn_gpt_native_train_linked" ]]; then
  NFN_NATIVE_GPT_TRAIN_BIN="$ROOT_DIR/build/nfn_gpt_native_train_linked"
else
  NFN_NATIVE_GPT_TRAIN_BIN="${NFN_NATIVE_GPT_TRAIN_BIN:-$ROOT_DIR/build/nfn_gpt_native_train}"
fi
NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN_EXPLICIT="${NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN+x}${NFN_SM120_CANDIDATE_TRAIN_BIN+x}"
NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN="${NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN:-${NFN_SM120_CANDIDATE_TRAIN_BIN:-$NFN_NATIVE_GPT_TRAIN_BIN}}"
NFN_NATIVE_TILE_OPS_LIB_EXPLICIT="${NFN_NATIVE_TILE_OPS_LIB+x}"
NFN_NATIVE_TILE_OPS_LIB="${NFN_NATIVE_TILE_OPS_LIB:-$ROOT_DIR/build/libnfn_native_train_tile_ops.so}"
NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB_EXPLICIT="${NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB-}${NFN_SM120_CANDIDATE_TILE_OPS_LIB-}"
NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB="${NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB:-$NFN_NATIVE_TILE_OPS_LIB}"

env_or_alias() {
  local primary="$1"
  local alias="$2"
  local default_value="$3"
  if [[ -n "${!primary-}" ]]; then
    printf '%s' "${!primary}"
  elif [[ -n "${!alias-}" ]]; then
    printf '%s' "${!alias}"
  else
    printf '%s' "$default_value"
  fi
}

env_or_alias3() {
  local primary="$1"
  local alias="$2"
  local parity_alias="$3"
  local default_value="$4"
  if [[ -n "${!primary-}" ]]; then
    printf '%s' "${!primary}"
  elif [[ -n "${!alias-}" ]]; then
    printf '%s' "${!alias}"
  elif [[ -n "${!parity_alias-}" ]]; then
    printf '%s' "${!parity_alias}"
  else
    printf '%s' "$default_value"
  fi
}

env_or_alias4() {
  local primary="$1"
  local alias="$2"
  local parity_alias="$3"
  local generic_alias="$4"
  local default_value="$5"
  if [[ -n "${!primary-}" ]]; then
    printf '%s' "${!primary}"
  elif [[ -n "${!alias-}" ]]; then
    printf '%s' "${!alias}"
  elif [[ -n "${!parity_alias-}" ]]; then
    printf '%s' "${!parity_alias}"
  elif [[ -n "${!generic_alias-}" ]]; then
    printf '%s' "${!generic_alias}"
  else
    printf '%s' "$default_value"
  fi
}

env_or_alias5() {
  local primary="$1"
  local native_candidate_alias="$2"
  local short_candidate_alias="$3"
  local parity_alias="$4"
  local generic_alias="$5"
  local default_value="$6"
  if [[ -n "${!primary-}" ]]; then
    printf '%s' "${!primary}"
  elif [[ -n "${!native_candidate_alias-}" ]]; then
    printf '%s' "${!native_candidate_alias}"
  elif [[ -n "${!short_candidate_alias-}" ]]; then
    printf '%s' "${!short_candidate_alias}"
  elif [[ -n "${!parity_alias-}" ]]; then
    printf '%s' "${!parity_alias}"
  elif [[ -n "${!generic_alias-}" ]]; then
    printf '%s' "${!generic_alias}"
  else
    printf '%s' "$default_value"
  fi
}

env_or_alias6() {
  local primary="$1"
  local native_alias="$2"
  local native_candidate_alias="$3"
  local short_candidate_alias="$4"
  local parity_alias="$5"
  local generic_alias="$6"
  local default_value="$7"
  if [[ -n "${!primary-}" ]]; then
    printf '%s' "${!primary}"
  elif [[ -n "${!native_alias-}" ]]; then
    printf '%s' "${!native_alias}"
  elif [[ -n "${!native_candidate_alias-}" ]]; then
    printf '%s' "${!native_candidate_alias}"
  elif [[ -n "${!short_candidate_alias-}" ]]; then
    printf '%s' "${!short_candidate_alias}"
  elif [[ -n "${!parity_alias-}" ]]; then
    printf '%s' "${!parity_alias}"
  elif [[ -n "${!generic_alias-}" ]]; then
    printf '%s' "${!generic_alias}"
  else
    printf '%s' "$default_value"
  fi
}

tile_ops_arg_for() {
  local train_bin="$1"
  local tile_ops_lib="$2"
  local explicit="$3"
  if [[ -z "$explicit" && "$(basename "$train_bin")" == "nfn_gpt_native_train_linked" ]]; then
    printf '%s' "linked"
  else
    printf '%s' "$tile_ops_lib"
  fi
}

STEPS="$(env_or_alias5 NFN_SM120_NATIVE_STEPS NFN_SM120_NATIVE_CANDIDATE_STEPS NFN_SM120_CANDIDATE_STEPS NFN_SM120_PARITY_STEPS NFN_SM120_STEPS 10)"
SAMPLES="$(env_or_alias5 NFN_SM120_NATIVE_SAMPLES NFN_SM120_NATIVE_CANDIDATE_SAMPLES NFN_SM120_CANDIDATE_SAMPLES NFN_SM120_PARITY_SAMPLES NFN_SM120_SAMPLES 3)"
WARMUP="$(env_or_alias5 NFN_SM120_NATIVE_WARMUP NFN_SM120_NATIVE_CANDIDATE_WARMUP NFN_SM120_CANDIDATE_WARMUP NFN_SM120_PARITY_WARMUP NFN_SM120_WARMUP 1)"
TRAIN_BATCH_TOKENS="$(env_or_alias5 NFN_SM120_NATIVE_TRAIN_BATCH_TOKENS NFN_SM120_NATIVE_CANDIDATE_TRAIN_BATCH_TOKENS NFN_SM120_CANDIDATE_TRAIN_BATCH_TOKENS NFN_SM120_PARITY_TRAIN_BATCH_TOKENS NFN_SM120_TRAIN_BATCH_TOKENS 524288)"
CUDA_VISIBLE_DEVICES_VALUE="$(env_or_alias5 NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES NFN_SM120_NATIVE_CANDIDATE_CUDA_VISIBLE_DEVICES NFN_SM120_CANDIDATE_CUDA_VISIBLE_DEVICES NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES NFN_SM120_CUDA_VISIBLE_DEVICES dedicated)"
CUDA_DEVICE_MAX_CONNECTIONS_VALUE="$(env_or_alias5 NFN_SM120_NATIVE_CUDA_DEVICE_MAX_CONNECTIONS NFN_SM120_NATIVE_CANDIDATE_CUDA_DEVICE_MAX_CONNECTIONS NFN_SM120_CANDIDATE_CUDA_DEVICE_MAX_CONNECTIONS NFN_SM120_PARITY_CUDA_DEVICE_MAX_CONNECTIONS NFN_SM120_CUDA_DEVICE_MAX_CONNECTIONS 1)"
MAX_GPU_UTILIZATION="$(env_or_alias5 NFN_SM120_NATIVE_MAX_GPU_UTILIZATION_PCT NFN_SM120_NATIVE_CANDIDATE_MAX_GPU_UTILIZATION_PCT NFN_SM120_CANDIDATE_MAX_GPU_UTILIZATION_PCT NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT NFN_SM120_MAX_GPU_UTILIZATION_PCT 15)"
SELECTED_GPU_UTILIZATION_RETRIES="$(env_or_alias5 NFN_SM120_NATIVE_SELECTED_GPU_UTILIZATION_RETRIES NFN_SM120_NATIVE_CANDIDATE_SELECTED_GPU_UTILIZATION_RETRIES NFN_SM120_CANDIDATE_SELECTED_GPU_UTILIZATION_RETRIES NFN_SM120_PARITY_SELECTED_GPU_UTILIZATION_RETRIES NFN_SM120_SELECTED_GPU_UTILIZATION_RETRIES 3)"
SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS="$(env_or_alias5 NFN_SM120_NATIVE_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS NFN_SM120_NATIVE_CANDIDATE_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS NFN_SM120_CANDIDATE_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS NFN_SM120_PARITY_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS NFN_SM120_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS 0.25)"
ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE="$(env_or_alias5 NFN_SM120_NATIVE_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE NFN_SM120_NATIVE_CANDIDATE_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE NFN_SM120_CANDIDATE_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE NFN_SM120_PARITY_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE NFN_SM120_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE 1)"
COMMAND_TIMEOUT_SECONDS="$(env_or_alias5 NFN_SM120_NATIVE_COMMAND_TIMEOUT_SECONDS NFN_SM120_NATIVE_CANDIDATE_COMMAND_TIMEOUT_SECONDS NFN_SM120_CANDIDATE_COMMAND_TIMEOUT_SECONDS NFN_SM120_PARITY_COMMAND_TIMEOUT_SECONDS NFN_SM120_COMMAND_TIMEOUT_SECONDS 300)"
SAMPLE_EVERY="$(env_or_alias5 NFN_SM120_NATIVE_SAMPLE_EVERY NFN_SM120_NATIVE_CANDIDATE_SAMPLE_EVERY NFN_SM120_CANDIDATE_SAMPLE_EVERY NFN_SM120_PARITY_SAMPLE_EVERY NFN_SM120_SAMPLE_EVERY 0)"
CHECKPOINT_EVERY="$(env_or_alias5 NFN_SM120_NATIVE_CHECKPOINT_EVERY NFN_SM120_NATIVE_CANDIDATE_CHECKPOINT_EVERY NFN_SM120_CANDIDATE_CHECKPOINT_EVERY NFN_SM120_PARITY_CHECKPOINT_EVERY NFN_SM120_CHECKPOINT_EVERY 0)"
GENERATE_TOKENS="$(env_or_alias5 NFN_SM120_NATIVE_GENERATE_TOKENS NFN_SM120_NATIVE_CANDIDATE_GENERATE_TOKENS NFN_SM120_CANDIDATE_GENERATE_TOKENS NFN_SM120_PARITY_GENERATE_TOKENS NFN_SM120_GENERATE_TOKENS 144)"
ACTIVATION="$(env_or_alias5 NFN_SM120_NATIVE_ACTIVATION NFN_SM120_NATIVE_CANDIDATE_ACTIVATION NFN_SM120_CANDIDATE_ACTIVATION NFN_SM120_PARITY_ACTIVATION NFN_SM120_ACTIVATION gelu)"
JSON_OUT="$(env_or_alias5 NFN_SM120_NATIVE_JSON_OUT NFN_SM120_NATIVE_CANDIDATE_JSON_OUT NFN_SM120_CANDIDATE_JSON_OUT NFN_SM120_PARITY_JSON_OUT NFN_SM120_JSON_OUT "/tmp/nfn_sm120_native_candidate_${STEPS}step.json")"
PROFILE_DIR_RAW="$(env_or_alias5 NFN_SM120_NATIVE_PROFILE_DIR NFN_SM120_NATIVE_CANDIDATE_PROFILE_DIR NFN_SM120_CANDIDATE_PROFILE_DIR NFN_SM120_PARITY_PROFILE_DIR NFN_SM120_PROFILE_DIR "/tmp/nfn_sm120_native_candidate_profiles_${STEPS}step")"
INCLUDE_LLMK_REFERENCE="$(env_or_alias6 NFN_SM120_NATIVE_INCLUDE_LLMK_REFERENCE NFN_SM120_NATIVE_INCLUDE_REFERENCE NFN_SM120_NATIVE_CANDIDATE_INCLUDE_LLMK_REFERENCE NFN_SM120_CANDIDATE_INCLUDE_LLMK_REFERENCE NFN_SM120_PARITY_INCLUDE_LLMK_REFERENCE NFN_SM120_INCLUDE_LLMK_REFERENCE 1)"
REFERENCE_OUTPUT_DIR="$(env_or_alias5 NFN_SM120_NATIVE_REFERENCE_OUTPUT_DIR NFN_SM120_NATIVE_CANDIDATE_REFERENCE_OUTPUT_DIR NFN_SM120_CANDIDATE_REFERENCE_OUTPUT_DIR NFN_SM120_PARITY_REFERENCE_OUTPUT_DIR NFN_SM120_REFERENCE_OUTPUT_DIR /tmp/nfn_llmk_sm120_candidate)"
STAGE_TIMING="$(env_or_alias5 NFN_SM120_NATIVE_STAGE_TIMING NFN_SM120_NATIVE_CANDIDATE_STAGE_TIMING NFN_SM120_CANDIDATE_STAGE_TIMING NFN_SM120_PARITY_STAGE_TIMING NFN_SM120_STAGE_TIMING 0)"
SETUP_EVENT_TIMING="$(env_or_alias5 NFN_SM120_NATIVE_SETUP_EVENT_TIMING NFN_SM120_NATIVE_CANDIDATE_SETUP_EVENT_TIMING NFN_SM120_CANDIDATE_SETUP_EVENT_TIMING NFN_SM120_PARITY_SETUP_EVENT_TIMING NFN_SM120_SETUP_EVENT_TIMING 0)"
ATTENTION_SECTION_TIMING="$(env_or_alias5 NFN_SM120_NATIVE_ATTENTION_SECTION_TIMING NFN_SM120_NATIVE_CANDIDATE_ATTENTION_SECTION_TIMING NFN_SM120_CANDIDATE_ATTENTION_SECTION_TIMING NFN_SM120_PARITY_ATTENTION_SECTION_TIMING NFN_SM120_ATTENTION_SECTION_TIMING 0)"
LINEAR_SHAPE_STATS="$(env_or_alias5 NFN_SM120_NATIVE_LINEAR_SHAPE_STATS NFN_SM120_NATIVE_CANDIDATE_LINEAR_SHAPE_STATS NFN_SM120_CANDIDATE_LINEAR_SHAPE_STATS NFN_SM120_PARITY_LINEAR_SHAPE_STATS NFN_SM120_LINEAR_SHAPE_STATS 0)"
STARTUP_ONLY="$(env_or_alias5 NFN_SM120_NATIVE_STARTUP_ONLY NFN_SM120_NATIVE_CANDIDATE_STARTUP_ONLY NFN_SM120_CANDIDATE_STARTUP_ONLY NFN_SM120_PARITY_STARTUP_ONLY NFN_SM120_STARTUP_ONLY 0)"
DRY_RUN_PLAN="$(env_or_alias5 NFN_SM120_NATIVE_DRY_RUN_PLAN NFN_SM120_NATIVE_CANDIDATE_DRY_RUN_PLAN NFN_SM120_CANDIDATE_DRY_RUN_PLAN NFN_SM120_PARITY_DRY_RUN_PLAN NFN_SM120_DRY_RUN_PLAN 0)"
CUDA_VERSION_PREFLIGHT="$(env_or_alias5 NFN_SM120_NATIVE_CUDA_VERSION_PREFLIGHT NFN_SM120_NATIVE_CANDIDATE_CUDA_VERSION_PREFLIGHT NFN_SM120_CANDIDATE_CUDA_VERSION_PREFLIGHT NFN_SM120_PARITY_CUDA_VERSION_PREFLIGHT NFN_SM120_CUDA_VERSION_PREFLIGHT 0)"
TRAIN_LOOP_EVENT_TIMING="$(env_or_alias5 NFN_SM120_NATIVE_TRAIN_LOOP_EVENT_TIMING NFN_SM120_NATIVE_CANDIDATE_TRAIN_LOOP_EVENT_TIMING NFN_SM120_CANDIDATE_TRAIN_LOOP_EVENT_TIMING NFN_SM120_PARITY_TRAIN_LOOP_EVENT_TIMING NFN_SM120_TRAIN_LOOP_EVENT_TIMING 1)"
AUTO_DISABLE_METRIC_RATIO_GATES="$(env_or_alias5 NFN_SM120_NATIVE_AUTO_DISABLE_METRIC_RATIO_GATES NFN_SM120_NATIVE_CANDIDATE_AUTO_DISABLE_METRIC_RATIO_GATES NFN_SM120_CANDIDATE_AUTO_DISABLE_METRIC_RATIO_GATES NFN_SM120_PARITY_AUTO_DISABLE_METRIC_RATIO_GATES NFN_SM120_AUTO_DISABLE_METRIC_RATIO_GATES 0)"
DISABLE_METRIC_RATIO_GATES="$(env_or_alias5 NFN_SM120_NATIVE_DISABLE_METRIC_RATIO_GATES NFN_SM120_NATIVE_CANDIDATE_DISABLE_METRIC_RATIO_GATES NFN_SM120_CANDIDATE_DISABLE_METRIC_RATIO_GATES NFN_SM120_PARITY_DISABLE_METRIC_RATIO_GATES NFN_SM120_DISABLE_METRIC_RATIO_GATES 0)"
REQUIRE_NATIVE_LM_HEAD_TRUE_FUSED="$(env_or_alias5 NFN_SM120_NATIVE_REQUIRE_LM_HEAD_TRUE_FUSED NFN_SM120_NATIVE_CANDIDATE_REQUIRE_LM_HEAD_TRUE_FUSED NFN_SM120_CANDIDATE_REQUIRE_LM_HEAD_TRUE_FUSED NFN_SM120_PARITY_REQUIRE_NATIVE_LM_HEAD_TRUE_FUSED NFN_SM120_REQUIRE_NATIVE_LM_HEAD_TRUE_FUSED 0)"
LM_HEAD_BACKWARD_PREFLIGHT="$(env_or_alias5 NFN_SM120_NATIVE_LM_HEAD_BACKWARD_PREFLIGHT NFN_SM120_NATIVE_CANDIDATE_LM_HEAD_BACKWARD_PREFLIGHT NFN_SM120_CANDIDATE_LM_HEAD_BACKWARD_PREFLIGHT NFN_SM120_PARITY_LM_HEAD_BACKWARD_PREFLIGHT NFN_SM120_LM_HEAD_BACKWARD_PREFLIGHT auto)"
LM_HEAD_BACKWARD_MAX_REFERENCE_GAP_MS="$(env_or_alias5 NFN_SM120_NATIVE_LM_HEAD_BACKWARD_MAX_REFERENCE_GAP_MS NFN_SM120_NATIVE_CANDIDATE_LM_HEAD_BACKWARD_MAX_REFERENCE_GAP_MS NFN_SM120_CANDIDATE_LM_HEAD_BACKWARD_MAX_REFERENCE_GAP_MS NFN_SM120_PARITY_LM_HEAD_BACKWARD_MAX_REFERENCE_GAP_MS NFN_SM120_LM_HEAD_BACKWARD_MAX_REFERENCE_GAP_MS "")"
LM_HEAD_BACKWARD_MAX_REFERENCE_WITH_LOGITS_GAP_MS="$(env_or_alias5 NFN_SM120_NATIVE_LM_HEAD_BACKWARD_MAX_REFERENCE_WITH_LOGITS_GAP_MS NFN_SM120_NATIVE_CANDIDATE_LM_HEAD_BACKWARD_MAX_REFERENCE_WITH_LOGITS_GAP_MS NFN_SM120_CANDIDATE_LM_HEAD_BACKWARD_MAX_REFERENCE_WITH_LOGITS_GAP_MS NFN_SM120_PARITY_LM_HEAD_BACKWARD_MAX_REFERENCE_WITH_LOGITS_GAP_MS NFN_SM120_LM_HEAD_BACKWARD_MAX_REFERENCE_WITH_LOGITS_GAP_MS "")"
LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_GAP_MS="$(env_or_alias5 NFN_SM120_NATIVE_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_GAP_MS NFN_SM120_NATIVE_CANDIDATE_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_GAP_MS NFN_SM120_CANDIDATE_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_GAP_MS NFN_SM120_PARITY_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_GAP_MS NFN_SM120_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_GAP_MS "")"
LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_WITH_LOGITS_GAP_MS="$(env_or_alias5 NFN_SM120_NATIVE_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_WITH_LOGITS_GAP_MS NFN_SM120_NATIVE_CANDIDATE_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_WITH_LOGITS_GAP_MS NFN_SM120_CANDIDATE_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_WITH_LOGITS_GAP_MS NFN_SM120_PARITY_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_WITH_LOGITS_GAP_MS NFN_SM120_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_WITH_LOGITS_GAP_MS "")"
COMMON_ENV_RAW="$(env_or_alias4 NFN_SM120_NATIVE_ENV NFN_SM120_COMMON_ENV NFN_SM120_PARITY_ENV NFN_SM120_ENV "")"
BASELINE_ENV_RAW="$(env_or_alias NFN_SM120_NATIVE_BASELINE_ENV NFN_SM120_CANDIDATE_BASELINE_ENV "")"
CANDIDATE_ENV_RAW="$(env_or_alias NFN_SM120_NATIVE_CANDIDATE_ENV NFN_SM120_CANDIDATE_ENV "")"
CANDIDATE_PROFILE="$(env_or_alias NFN_SM120_NATIVE_CANDIDATE_PROFILE NFN_SM120_CANDIDATE_PROFILE "")"
CANDIDATE_TILE_OPS_BUILD_FLAGS="$(env_or_alias NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_BUILD_FLAGS NFN_SM120_CANDIDATE_TILE_OPS_BUILD_FLAGS "")"
COMMON_EXTRA_ARGS_RAW="$(env_or_alias3 NFN_SM120_NATIVE_EXTRA_ARGS NFN_SM120_COMMON_EXTRA_ARGS NFN_SM120_PARITY_EXTRA_ARGS "")"
BASELINE_EXTRA_ARGS_RAW="$(env_or_alias NFN_SM120_NATIVE_BASELINE_EXTRA_ARGS NFN_SM120_CANDIDATE_BASELINE_EXTRA_ARGS "")"
CANDIDATE_EXTRA_ARGS_RAW="$(env_or_alias NFN_SM120_NATIVE_CANDIDATE_EXTRA_ARGS NFN_SM120_CANDIDATE_EXTRA_ARGS "")"
TIMEOUT_PRONE_CANDIDATE_PROFILE=""
REJECTED_CANDIDATE_PROFILE=""
REJECTED_CANDIDATE_REASON=""
CANDIDATE_NOTE=""
PROMOTED_QKV_LN128_PROFILE=0
DEFAULT_VS_LEGACY_PROFILE=0
STRICT_PROBE_CANDIDATE_PROFILE=""
STRICT_GROUPED_CUBLASLT_PROBE=0
AUTO_ATTENTION_SECTION_TIMING=0
FORCE_DISABLE_ROUTE_CHANGE=0
SKIP_LM_HEAD_CE_STAGE_GATE=0
LM_HEAD_BACKWARD_PREFLIGHT_PROFILE=""
if [[ -z "$CANDIDATE_EXTRA_ARGS_RAW" && -n "${NFN_SM120_NATIVE_CANDIDATE_ARGS-}" ]]; then
  CANDIDATE_EXTRA_ARGS_RAW="$NFN_SM120_NATIVE_CANDIDATE_ARGS"
fi
IS_DRY_RUN_PLAN=0
case "${DRY_RUN_PLAN,,}" in
  "1"|"true"|"yes"|"on")
    IS_DRY_RUN_PLAN=1
    ;;
esac
case "${DISABLE_METRIC_RATIO_GATES,,}" in
  "1"|"true"|"yes"|"on")
    AUTO_DISABLE_METRIC_RATIO_GATES=1
    ;;
  "0"|"false"|"no"|"off"|"")
    ;;
  *)
    echo "Unsupported NFN_SM120_NATIVE_DISABLE_METRIC_RATIO_GATES value: $DISABLE_METRIC_RATIO_GATES" >&2
    exit 2
    ;;
esac
ALLOW_REJECTED_CANDIDATE_PROFILE="$(env_or_alias NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE NFN_SM120_ALLOW_REJECTED_CANDIDATE_PROFILE 0)"
ENFORCE_REJECTED_CANDIDATE_RATIO_GATES="$(env_or_alias NFN_SM120_NATIVE_ENFORCE_REJECTED_CANDIDATE_RATIO_GATES NFN_SM120_ENFORCE_REJECTED_CANDIDATE_RATIO_GATES 0)"
MAX_CANDIDATE_RATIO_RAW="$(env_or_alias NFN_SM120_NATIVE_MAX_CANDIDATE_RATIO NFN_SM120_CANDIDATE_MAX_CANDIDATE_RATIO "")"
MIN_CANDIDATE_RATIO_RAW="$(env_or_alias NFN_SM120_NATIVE_MIN_CANDIDATE_RATIO NFN_SM120_CANDIDATE_MIN_CANDIDATE_RATIO "")"
MAX_CANDIDATE_REFERENCE_RATIO_RAW="$(env_or_alias NFN_SM120_NATIVE_MAX_CANDIDATE_REFERENCE_RATIO NFN_SM120_CANDIDATE_MAX_CANDIDATE_REFERENCE_RATIO "")"
MIN_CANDIDATE_REFERENCE_RATIO_RAW="$(env_or_alias NFN_SM120_NATIVE_MIN_CANDIDATE_REFERENCE_RATIO NFN_SM120_CANDIDATE_MIN_CANDIDATE_REFERENCE_RATIO "")"
USER_MAX_CANDIDATE_RATIO_RAW="$MAX_CANDIDATE_RATIO_RAW"
USER_MIN_CANDIDATE_RATIO_RAW="$MIN_CANDIDATE_RATIO_RAW"
USER_MAX_CANDIDATE_REFERENCE_RATIO_RAW="$MAX_CANDIDATE_REFERENCE_RATIO_RAW"
USER_MIN_CANDIDATE_REFERENCE_RATIO_RAW="$MIN_CANDIDATE_REFERENCE_RATIO_RAW"
case "${CANDIDATE_PROFILE,,}" in
  ""|"none"|"off"|"0"|"false"|"no")
    ;;
  "linked_startup"|"linked-startup"|"linked_tile_ops"|"linked-tile-ops")
    NFN_NATIVE_GPT_TRAIN_BIN="${NFN_SM120_NATIVE_BASELINE_TRAIN_BIN:-$ROOT_DIR/build/nfn_gpt_native_train}"
    NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN="${NFN_SM120_NATIVE_LINKED_STARTUP_CANDIDATE_BIN:-$ROOT_DIR/build/nfn_gpt_native_train_linked}"
    STARTUP_ONLY=1
    STEPS=0
    WARMUP="${NFN_SM120_NATIVE_WARMUP:-${NFN_SM120_NATIVE_CANDIDATE_WARMUP:-0}}"
    FORCE_DISABLE_ROUTE_CHANGE=1
    ;;
  "fast_startup"|"fast-startup"|"native_fast_startup"|"native-fast-startup")
    CANDIDATE_NOTE="Compares default native training setup against NFN_NATIVE_GPT_FAST_STARTUP=1. The fast-startup mode leaves long-training defaults unchanged, but changes setup prewarm defaults so TK QKV first-use and LM-head graph prewarm are skipped unless their explicit prewarm env vars force them back on."
    STARTUP_ONLY=1
    STEPS=0
    WARMUP="${NFN_SM120_NATIVE_WARMUP:-${NFN_SM120_NATIVE_CANDIDATE_WARMUP:-0}}"
    INCLUDE_LLMK_REFERENCE=0
    FORCE_DISABLE_ROUTE_CHANGE=1
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_FAST_STARTUP=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_FAST_STARTUP=1"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-setup_wall_ms=0.850}"
    ;;
  "fast_startup_full"|"fast-startup-full"|"native_fast_startup_full"|"native-fast-startup-full")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 full 5-step, 2-sample gate improved setup_wall_ms to 0.655522x by skipping TK QKV first-use and LM-head graph setup prewarms, but rejected default promotion because train_loop_wall_ms_per_step regressed to 1.017654x, first-step CUDA-event time to 1.086326x, train_tokens_per_second to 0.982655x, and candidate-over-llm.kittens train-loop wall to 1.010462x. Keep fast_startup as startup-only/preflight policy unless long-run gates prove the first-step cost is amortized."
    CANDIDATE_NOTE="Runs real optimizer steps with NFN_NATIVE_GPT_FAST_STARTUP=1 instead of startup-only mode. This shows whether skipped setup prewarms simply move work into the first training step."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_FAST_STARTUP=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_FAST_STARTUP=1"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-train_loop_wall_ms_per_step=1.005 train_loop_cuda_event_steady_state_wall_ms_per_step=1.005 setup_wall_ms=0.800}"
    MIN_CANDIDATE_RATIO_RAW="${MIN_CANDIDATE_RATIO_RAW:-train_tokens_per_second=0.995}"
    ;;
  "lm_head_tk_dinput_32768"|"lm-head-tk-dinput-32768")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 2-sample same-script gate routed LM-head dHidden through TK dInput but regressed train_loop_wall_ms_per_step to 1.045528x and stage.lm_head_backward.dhidden.total_ms to 1.132973x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_TK_DINPUT_ENABLE_SHAPE=768,32768,50304,N,N"
    ;;
  "lm_head_cublaslt_dhidden_32768"|"lm-head-cublaslt-dhidden-32768")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 3-step, 3-sample stage-timed gate moved 48 LM-head dHidden calls from BF16 GEMMEx to cuBLASLt but regressed train_loop_wall_ms_per_step to 1.000384x, stage.lm_head_backward.dhidden.total_ms to 1.000199x, and stage.block_backward.total_ms to 1.001504x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE=768,32768,50304,N,N NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1 NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,32768,50304,N,N,0"
    ;;
  "lm_head_dhidden_fast16bf_32768"|"lm-head-dhidden-fast16bf-32768"|"lm_head_dhidden_gemmex_fast16bf_32768"|"lm-head-dhidden-gemmex-fast16bf-32768")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2-sample stage-timed rerun requested FAST_16BF for 48 LM-head dHidden calls but regressed stage.lm_head_backward.total_ms to 1.004489x while stage.lm_head_backward.dhidden.total_ms stayed flat at 1.000265x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF_SHAPE=768,32768,50304,N,N"
    ;;
  "lm_head_cooperative_cublaslt"|"lm-head-cooperative-cublaslt"|"lm_head_cublaslt_cooperative"|"lm-head-cublaslt-cooperative")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-25 3-step, 2-sample stage-timed rerun selected the LM-head cooperative cuBLASLt wrapper and moved lm_head_cooperative_backward_strategy to diagnostic-cublaslt-sequence-wrapper-ce-dhidden-dweight-not-parity, but rejected it at 1.077251x train_loop_wall_ms_per_step, 1.083727x steady-state CUDA-event step time, 1.335573x stage.lm_head_backward.total_ms, and 1.477219x stage.lm_head_backward.cooperative.total_ms."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_CUBLASLT=1"
    ;;
  "lm_head_graph_body_cublaslt"|"lm-head-graph-body-cublaslt"|"lm_head_cooperative_graph_body_cublaslt"|"lm-head-cooperative-graph-body-cublaslt")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-26 3-step, 1-sample stage-timed gate routed the cooperative CUDA graph body through the existing strided cuBLASLt LM-head dHidden/dWeight kernels, but rejected it at 1.079498x train_loop_wall_ms_per_step, 1.083841x steady-state CUDA-event step time, 1.344089x stage.lm_head_backward.total_ms, and 1.495431x stage.lm_head_backward.cooperative.total_ms."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_GRAPH_BODY_CUBLASLT=1"
    ;;
  "lm_head_tk_dweight_32768"|"lm-head-tk-dweight-32768")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2-sample stage-timed rerun moved 48 LM-head dWeight calls to TK but regressed train_loop_wall_ms_per_step to 1.052253x and stage.lm_head_backward.dweight.total_ms to 1.337552x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_TK_DWEIGHT_ENABLE_SHAPE=768,50304,32768,N,T"
    ;;
  "lm_head_tk_dweight_49152"|"lm-head-tk-dweight-49152")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 historical-shape 2-step, 2-sample stage-timed gate moved linear_tk_dweight_gemm_count from 0 to 16 for the 49152-row LM-head chunk but regressed train_loop_wall_ms_per_step to 1.303473x, stage.lm_head_backward.dweight.total_ms to 1.201790x, and stage.block_backward.total_ms to 1.557413x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_TK_DWEIGHT_ENABLE_SHAPE=768,50304,49152,N,T"
    ;;
  "lm_head_prepack_bf16_hidden_off"|"lm-head-prepack-bf16-hidden-off"|"lm_head_no_prepack_bf16_hidden"|"lm-head-no-prepack-bf16-hidden")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 3-step, 2-sample stage-timed gate changed the LM-head dWeight strategy from full-final-norm BF16 prepack to per-chunk BF16 packing, but regressed train_loop_wall_ms_per_step to 1.049342x, steady-state CUDA-event step time to 1.064113x, stage.lm_head_backward.total_ms to 1.055161x, dHidden to 1.000521x, and dWeight to 1.008148x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=1"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=0"
    ;;
  "lm_head_prepack_bf16_hidden_on"|"lm-head-prepack-bf16-hidden-on"|"lm_head_full_prepack_bf16_hidden"|"lm-head-full-prepack-bf16-hidden")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 5-step, 3-sample gate compared the older full-final-norm BF16 prepack route against the current per-chunk LM-head route and rejected it because stage.lm_head_backward.dhidden.total_ms regressed to 1.000690x despite train-loop mean 0.997953x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=1"
    ;;
  "lm_head_bf16_hidden_from_final_norm"|"lm-head-bf16-hidden-from-final-norm"|"lm_head_final_norm_bf16_hidden"|"lm-head-final-norm-bf16-hidden")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 3-step, 2-sample gate changed LM-head BF16 hidden staging from separate prepack to final LayerNorm output, but regressed train_loop_wall_ms_per_step to 1.009000x, steady CUDA-event step time to 1.000147x, and stage.lm_head_backward.dweight.total_ms to 1.000293x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=1 NFN_NATIVE_GPT_LM_HEAD_BF16_HIDDEN_FROM_FINAL_NORM=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=1 NFN_NATIVE_GPT_LM_HEAD_BF16_HIDDEN_FROM_FINAL_NORM=1"
    ;;
  "lm_head_public_vocab_strided_gemm"|"lm-head-public-vocab-strided-gemm"|"lm_head_public_vocab_strided"|"lm-head-public-vocab-strided")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 same-binary paired run routed padded LM-head dHidden and dWeight chunks through logical public-vocab strided GEMMs, but regressed train_loop_wall_ms_per_step to 1.117352x and tokens/sec to 0.895573x versus the aligned padded-vocab route."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_PUBLIC_VOCAB_STRIDED_GEMM=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_PUBLIC_VOCAB_STRIDED_GEMM=1"
    ;;
  "mlp_proj_tk_dweight_65536"|"mlp-proj-tk-dweight-65536"|"block_mlp_proj_tk_dweight_65536"|"block-mlp-proj-tk-dweight-65536")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 same-script gate moved 192 MLP projection dWeight calls to TK but regressed train_loop_wall_ms_per_step to 1.019797x and train_tokens_per_second to 0.980596x, so cuBLASLt BGRADB remains the default for this block bucket."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_TK_DWEIGHT_ENABLE_SHAPE=3072,768,65536,N,T"
    ;;
  "block_split_bgrad_65536"|"block-split-bgrad-65536"|"block_bf16_bf16_split_bgrad_65536"|"block-bf16-bf16-split-bgrad-65536"|"block_disable_bgradb_65536"|"block-disable-bgradb-65536")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 same-script gate changed cuBLASLt BGRADB route counters from 1152 to 288 but regressed train_loop_wall_ms_per_step to 1.036221x, block backward to 1.074258x, MLP FC dWeight+bias to 1.387410x, and MLP projection dWeight+bias to 1.241028x, so fused BGRADB remains the default."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_BF16_BF16_BGRAD_DISABLE_SHAPE=768,2304,65536,N,T:768,768,65536,N,T:768,3072,65536,N,T:3072,768,65536,N,T"
    ;;
  "mlp_proj_split_bgrad_65536"|"mlp-proj-split-bgrad-65536"|"block_mlp_proj_split_bgrad_65536"|"block-mlp-proj-split-bgrad-65536"|"mlp_proj_disable_bgradb_65536"|"mlp-proj-disable-bgradb-65536")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 same-script gate changed cuBLASLt bgrad route counters but regressed train_loop_wall_ms_per_step to 1.017997x and MLP projection dWeight+bias to 1.256535x, so BGRADB remains the default."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_BF16_BF16_BGRAD_DISABLE_SHAPE=3072,768,65536,N,T"
    ;;
  "layernorm_affine_row_chunk_128"|"layernorm-affine-row-chunk-128"|"ln_affine_row_chunk_128"|"ln-affine-row-chunk-128")
    ACCEPTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    ACCEPTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 current-default 5-step, 3-sample stage-timed rerun kept the 128-row LayerNorm affine reducer as the default because it improved train_loop_wall_ms_per_step to 0.998906x, steady-state CUDA-event timing to 0.999138x, train_tokens_per_second to 1.001101x, final-norm backward to 0.993427x, block backward to 0.997165x, LN2 residual backward to 0.992676x, and LN1 residual backward to 0.915224x while moving block_state_layout.layer_norm_backward_affine_row_chunk_size from 256 to 128. LM-head backward was noise-flat at 1.000764x mean / 1.000017x median, so this profile is now a default-vs-legacy proof rather than a rejected candidate."
    DEFAULT_VS_LEGACY_PROFILE=1
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=256"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=128"
    ;;
  "layernorm_affine_row_chunk_64"|"layernorm-affine-row-chunk-64"|"ln_affine_row_chunk_64"|"ln-affine-row-chunk-64")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 5-step, 3-sample stage-timed gate changed the LayerNorm affine route from 128 to 64 rows and improved train_loop_wall_ms_per_step to 0.998045x, but regressed stage.block_backward.mlp_proj.total_ms to 1.004276x and stage.lm_head_backward.total_ms to 1.000446x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=64"
    ;;
  "layernorm_affine_row_chunk_96"|"layernorm-affine-row-chunk-96"|"ln_affine_row_chunk_96"|"ln-affine-row-chunk-96")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 5-step, 3-sample stage-timed gate changed the LayerNorm affine route from 128 to 96 rows and improved train_loop_wall_ms_per_step to 0.999112x, but regressed stage.block_backward.mlp_proj.total_ms to 1.000296x and stage.lm_head_backward.total_ms to 1.000002x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=96"
    ;;
  "layernorm_affine_row_chunk_512"|"layernorm-affine-row-chunk-512"|"ln_affine_row_chunk_512"|"ln-affine-row-chunk-512")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 gate measured the 512-row LayerNorm affine route at 1.019837x train_loop_wall_ms_per_step and 1.039994x stage.block_backward.total_ms after the 128-row route became the default."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=512"
    ;;
  "linear_bias_row_chunk_256"|"linear-bias-row-chunk-256"|"bgrad_row_chunk_256"|"bgrad-row-chunk-256")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 3-step, 2-sample stage-timed gate compared the current 256-row linear-bias reduction against the legacy 512-row route. It changed block_state_layout.linear_backward_bias_row_chunk_size from 512 to 256 and improved train_loop_wall_ms_per_step to 0.993823x, but failed strict gates at 1.002081x steady-state CUDA-event step time and 1.000470x stage.block_backward.mlp_fc.dweight_bias.total_ms. The Tile-CUDA default remains 256; this profile is kept only for historical baseline reproduction."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE=512"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE=256"
    ;;
  "linear_bias_row_chunk_1024"|"linear-bias-row-chunk-1024"|"bgrad_row_chunk_1024"|"bgrad-row-chunk-1024")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 one-step gate changed the bias reducer chunk to 1024 rows but rejected it at 1.009736x train_loop_wall_ms_per_step, 1.008488x stage.block_backward.total_ms, and 1.050950x stage.block_backward.mlp_proj.dweight_bias.total_ms."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE=1024"
    ;;
  "linear_bias_threads_512"|"linear-bias-threads-512"|"bgrad_threads_512"|"bgrad-threads-512")
    CANDIDATE_NOTE="CUDA 13.3.33 dedicated RTX 5090 2026-06-25 corrected-lib 3-step, 2-sample stage-timed rerun changed block_state_layout.linear_backward_bias_threads_per_block from 256 to 512 and kept 512 as the default after measuring train_loop_wall_ms_per_step=0.992990x, train_loop_cuda_event_steady_state_wall_ms_per_step=0.998950x, train_tokens_per_second=1.007496x, stage.block_backward.total_ms=0.989262x, stage.block_backward.mlp_fc.dweight_bias.total_ms=0.972707x, and stage.block_backward.mlp_proj.dweight_bias.total_ms=0.984430x."
    DEFAULT_VS_LEGACY_PROFILE=1
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LINEAR_BACKWARD_BIAS_THREADS=256"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LINEAR_BACKWARD_BIAS_THREADS=512"
    ;;
  "lm_head_logits_bf16_fallback_32768"|"lm-head-logits-bf16-fallback-32768")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 rebuilt 3-step, 2-sample stage-timed gate disabled TK for the restored 32768-row LM-head logits shape and moved lm_head_logits_tk_gemm_count from 48 to 0, but rejected the BF16 GEMMEx fallback at 1.003097x train_loop_wall_ms_per_step, 1.000836x steady-state CUDA-event step time, 1.010331x block backward, and 1.004728x MLP projection."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=50304,32768,768,T,N"
    ;;
  "lm_head_logits_bf16_fallback_49152"|"lm-head-logits-bf16-fallback-49152")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2-step, 2-sample stage-timed gate disabled TK for the historical 49152-row LM-head logits shape and moved lm_head_logits_tk_gemm_count from 32 to 16, but rejected it at 1.005968x train_loop_wall_ms_per_step, 1.009906x stage.block_backward.total_ms, and 1.000576x stage.block_backward.mlp_proj.total_ms."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=50304,49152,768,T,N"
    ;;
  "qkv_forward_bf16_fallback_65536"|"qkv-forward-bf16-fallback-65536"|"packed_qkv_forward_bf16_fallback_65536"|"packed-qkv-forward-bf16-fallback-65536")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 2-step, 2-sample stage-timed gate reduced TK forward calls but regressed the target stage.block_forward.attention.qkv.total_ms to 1.143374x, so the TK QKV forward route remains the default."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=2304,65536,768,T,N"
    ;;
  "tk_qkv_forward_prewarm"|"tk-qkv-forward-prewarm"|"qkv_forward_tk_prewarm"|"qkv-forward-tk-prewarm")
    ACCEPTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    ACCEPTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 3-step, 3-sample no-stage-timing rerun promoted the full-shape TK QKV first-use prewarm as the default because it improved current NeuralFn train_loop_wall_ms_per_step to 0.981250x, first-step CUDA-event timing to 0.945699x, train_tokens_per_second to 1.019109x, and total_wall_ms to 0.999956x while moving linear_tk_qkv_first_use_prewarm_success_count from 0 to 1. Setup regressed to 1.252423x and strict llm.kittens reference gates from the matching reference run still narrowly failed at 1.000567x train-loop wall, 1.001159x steady-state CUDA-event timing, and 0.999507x tokens/sec, so this is an incremental default-on first-use improvement, not final train-sm120 parity."
    DEFAULT_VS_LEGACY_PROFILE=1
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_PREWARM_TK_QKV_FORWARD=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_PREWARM_TK_QKV_FORWARD=1"
    MAX_CANDIDATE_RATIO_RAW+=" train_loop_cuda_event_first_step_wall_ms_per_step=1.000"
    MAX_CANDIDATE_RATIO_RAW+=" train_loop_wall_ms_per_step=1.000"
    ;;
  "tk_qkv_forward_prewarm_1row"|"tk-qkv-forward-prewarm-1row"|"qkv_forward_tk_prewarm_1row"|"qkv-forward-tk-prewarm-1row")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 corrected current-default comparison kept the one-row TK QKV first-use prewarm rejected. Versus the promoted full-shape prewarm default, NFN_NATIVE_GPT_PREWARM_TK_QKV_FORWARD_ROWS=1 regressed train_loop_wall_ms_per_step to 1.001366x, steady-state CUDA-event timing to 1.000877x, tokens/sec to 0.998636x, and stage.block_forward.attention.qkv.first_step_avg_ms to 1.075029x; setup was noise-flat at 1.006929x mean / 0.998029x median. Strict llm.kittens reference gates also failed at 1.006153x train-loop wall, 1.006227x steady-state timing, and 0.993913x tokens/sec."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_PREWARM_TK_QKV_FORWARD=1 NFN_NATIVE_GPT_PREWARM_TK_QKV_FORWARD_ROWS=1"
    MAX_CANDIDATE_RATIO_RAW+=" train_loop_cuda_event_first_step_wall_ms_per_step=1.000"
    MAX_CANDIDATE_RATIO_RAW+=" stage.block_forward.attention.qkv.first_step_avg_ms=1.000"
    ;;
  "tk_qkv_forward_prewarm_32768"|"tk-qkv-forward-prewarm-32768"|"qkv_forward_tk_prewarm_32768"|"qkv-forward-tk-prewarm-32768")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 3-step, 1-sample stage-timed probe limited TK QKV first-use prewarm to 32768 rows. It improved setup_wall_ms to 0.961917x but regressed train_loop_wall_ms_per_step to 1.002107x, steady-state CUDA-event timing to 1.002784x, stage.block_backward.total_ms to 1.002263x, and candidate-over-llm.kittens train_loop_wall_ms_per_step to 1.001097x. Keep rejected."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_PREWARM_TK_QKV_FORWARD=1 NFN_NATIVE_GPT_PREWARM_TK_QKV_FORWARD_ROWS=32768"
    MAX_CANDIDATE_RATIO_RAW+=" train_loop_cuda_event_first_step_wall_ms_per_step=1.000"
    MAX_CANDIDATE_RATIO_RAW+=" stage.block_forward.attention.qkv.first_step_avg_ms=1.000"
    ;;
  "mlp_fc_forward_bf16_fallback_65536"|"mlp-fc-forward-bf16-fallback-65536"|"mlp_fc_gelu_forward_bf16_fallback_65536"|"mlp-fc-gelu-forward-bf16-fallback-65536")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 2-step, 2-sample stage-timed gate did not change tracked route counters and regressed train_loop_wall_ms_per_step to 1.016916x, block backward to 1.034425x, and the target stage.block_forward.mlp_fc_gelu.total_ms to 1.000722x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=3072,65536,768,N,N"
    ;;
  "fused_ln2_bf16_out_off"|"fused-ln2-bf16-out-off"|"ln2_bf16_out_off"|"ln2-bf16-out-off"|"separate_ln2_bf16_store"|"separate-ln2-bf16-store")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2-step, 2-sample stage-timed gate proved the fused LN2 BF16 output route change but rejected the separate-store rollback at 1.020138x train_loop_wall_ms_per_step, 1.013718x steady-state CUDA-event wall time, and 1.119485x stage.block_forward.mlp_fc_gelu.total_ms."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_FUSE_LN2_BF16_OUT=1"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_FUSE_LN2_BF16_OUT=0"
    ;;
  "mlp_residual_next_ln1_off"|"mlp-residual-next-ln1-off"|"fused_mlp_residual_next_ln1_off"|"fused-mlp-residual-next-ln1-off")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2-step, 2-sample stage-timed gate proved the MLP residual next-LN1 fusion route change by moving block_state_layout.mlp_residual_next_ln1_fusion_count from 176 to 0, but rejected the rollback at 1.000520x train_loop_wall_ms_per_step, 1.004202x steady-state CUDA-event wall time, and 0.999479x train_tokens_per_second."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_FUSE_MLP_RESIDUAL_NEXT_LN1=1"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_FUSE_MLP_RESIDUAL_NEXT_LN1=0"
    ;;
  "ce_bf16_threads_512"|"ce-bf16-threads-512"|"lm_head_ce_bf16_threads_512"|"lm-head-ce-bf16-threads-512")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2-sample stage-timed rerun changed lm_head_ce_bf16_threads_per_row from 1024 to 512 but regressed train_loop_wall_ms_per_step to 1.012086x, stage.lm_head_backward.total_ms to 1.051608x, and stage.lm_head_backward.ce.total_ms to 1.430612x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_CE_BF16_THREADS=512"
    ;;
  "lm_head_ce_exp2"|"lm-head-ce-exp2"|"ce_bf16_exp2"|"ce-bf16-exp2")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-26 3-step, 2-sample stage-timed rerun changed the no-loss CE strategy from no-loss-default-specialized-dlogits-vec8-loads-scalar-stores to the generic exp2 path, but rejected it at 1.019757x train_loop_wall_ms_per_step, 1.022252x steady-state CUDA-event wall time, 1.097477x LM-head backward, and 1.140828x LM-head cooperative time."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_CE_BF16_EXP2=1"
    ;;
  "lm_head_ce_vec8_io"|"lm-head-ce-vec8-io"|"ce_bf16_vec8_io"|"ce-bf16-vec8-io")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 5-step, 2-sample stage-timed rerun changed CE vector I/O strategy but failed the strict LM-head CE gate at stage.lm_head_backward.ce.total_ms=1.003780x; the apparent train-loop gain came from unrelated block-backward timing noise."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_CE_BF16_VEC_LOADS=1 NFN_NATIVE_GPT_CE_BF16_VEC_STORES=1"
    ;;
  "lm_head_ce_vec8_normal_store"|"lm-head-ce-vec8-normal-store"|"ce_bf16_vec8_normal_store"|"ce-bf16-vec8-normal-store")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 3-step, 2-sample stage-timed gate changed the CE store route to vec8-loads-normal-stores and improved CE to 0.999055x, but regressed stage.lm_head_backward.total_ms to 1.009078x, stage.lm_head_backward.logits.total_ms to 1.024165x, and stage.block_backward.mlp_proj.total_ms to 1.014568x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_CE_BF16_VEC_LOADS=1 NFN_NATIVE_GPT_CE_BF16_VEC_NORMAL_STORES=1"
    ;;
  "lm_head_ce_scalar_streaming_store"|"lm-head-ce-scalar-streaming-store"|"ce_bf16_scalar_streaming_store"|"ce-bf16-scalar-streaming-store")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 3-step, 2-sample stage-timed rerun changed the CE store route to vec8-loads-scalar-streaming-stores but regressed train_loop_wall_ms_per_step to 1.020535x, train_loop_cuda_event_steady_state_wall_ms_per_step to 1.026691x, stage.lm_head_backward.total_ms to 1.122725x, and stage.lm_head_backward.ce.total_ms to 2.054816x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_CE_BF16_VEC_LOADS=1 NFN_NATIVE_GPT_CE_BF16_SCALAR_STREAMING_STORES=1"
    ;;
  "lm_head_ce_natural_rows"|"lm-head-ce-natural-rows"|"lm_head_ce_reverse_rows_off"|"lm-head-ce-reverse-rows-off")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 10-step same-script parity sample changed LM-head CE row order from reverse to natural but regressed CUDA-event wall time to 1.019563x, steady-state CUDA-event wall time to 1.019690x, and tokens/sec to 0.978913x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_CE_REVERSE_ROWS=0"
    ;;
  "lm_head_ce_default_specialized"|"lm-head-ce-default-specialized"|"ce_bf16_default_specialized"|"ce-bf16-default-specialized")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 3-step, 3-sample gate proved the default-shape row-loss CE specialization but rejected it at 1.001545x train_loop_wall_ms_per_step, 1.000931x stage.lm_head_backward.total_ms, and 1.000331x stage.lm_head_backward.ce.total_ms."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_CE_DEFAULT_SPECIALIZED=1"
    ;;
  "lm_head_ce_no_loss_default_specialized"|"lm-head-ce-no-loss-default-specialized"|"ce_bf16_no_loss_default_specialized"|"ce-bf16-no-loss-default-specialized")
    ACCEPTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    ACCEPTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 actual-training 5-step, 2-sample rerun confirmed the default no-loss CE specialization against the older generic no-loss CE+dlogits route: train_loop_wall_ms_per_step=0.977958x, train_tokens_per_second=1.022549x, candidate-over-llm.kittens train_loop_wall_ms_per_step=0.996300x, and candidate-over-llm.kittens train_tokens_per_second=1.003608x."
    DEFAULT_VS_LEGACY_PROFILE=1
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_DEFAULT_SPECIALIZED=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_DEFAULT_SPECIALIZED=1"
    COMMON_EXTRA_ARGS_RAW="${COMMON_EXTRA_ARGS_RAW:+$COMMON_EXTRA_ARGS_RAW }--train-loss-every-steps 0"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-train_loop_wall_ms_per_step=1.000}"
    MIN_CANDIDATE_RATIO_RAW="${MIN_CANDIDATE_RATIO_RAW:-train_tokens_per_second=1.000}"
    SKIP_LM_HEAD_CE_STAGE_GATE=1
    ;;
  "lm_head_prob_only_corrections"|"lm-head-prob-only-corrections"|"lm_head_ce_prob_only_corrections"|"lm-head-ce-prob-only-corrections")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 3-step, 2-sample stage-timed gate changed the no-loss LM-head CE strategy to prob-only dlogits plus target corrections, but rejected it at 1.005050x stage.lm_head_backward.total_ms and 1.000994x steady-state CUDA-event step time."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_CORRECTIONS=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_CORRECTIONS=1"
    COMMON_EXTRA_ARGS_RAW="${COMMON_EXTRA_ARGS_RAW:+$COMMON_EXTRA_ARGS_RAW }--train-loss-every-steps 0"
    ;;
  "lm_head_prob_only_combined_corrections"|"lm-head-prob-only-combined-corrections"|"lm_head_ce_prob_only_combined_corrections"|"lm-head-ce-prob-only-combined-corrections")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 3-step, 2-sample stage-timed gate changed the no-loss LM-head CE strategy to prob-only dlogits plus one combined target-correction launch, but rejected it at 1.006574x train_loop_wall_ms_per_step and 1.003646x stage.lm_head_backward.total_ms."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_COMBINED_CORRECTIONS=0 NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_CORRECTIONS=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_COMBINED_CORRECTIONS=1"
    COMMON_EXTRA_ARGS_RAW="${COMMON_EXTRA_ARGS_RAW:+$COMMON_EXTRA_ARGS_RAW }--train-loss-every-steps 0"
    ;;
  "lm_head_prob_only_ce_target_corrections"|"lm-head-prob-only-ce-target-corrections"|"lm_head_ce_prob_only_target_corrections"|"lm-head-ce-prob-only-target-corrections")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-25 one-step stage-timed gate changed the strategy to no-loss-prob-only-dlogits-vec8-loads-normal-vec8-stores-plus-ce-target-correction and improved non-cooperative native train-loop wall to 0.968799x, but rejected promotion because stage.lm_head_backward.ce.total_ms regressed to 1.053810x and candidate-over-llm.kittens train_loop_wall_ms_per_step remained 1.101282x."
    CANDIDATE_NOTE="Runs the LM-head CE route as prob-only dlogits and folds target corrections for dHidden and dWeight into the same CE row kernel, eliminating the separate combined target-correction launch. This is a candidate step toward the row-chunked CE+dHidden+dWeight fused Tile body."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=0 NFN_NATIVE_GPT_LM_HEAD_FUSED_LOSS_BACKWARD=0 NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_CE_TARGET_CORRECTIONS=0 NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_COMBINED_CORRECTIONS=0 NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_CORRECTIONS=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=0 NFN_NATIVE_GPT_LM_HEAD_FUSED_LOSS_BACKWARD=0 NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_CE_TARGET_CORRECTIONS=1"
    COMMON_EXTRA_ARGS_RAW="${COMMON_EXTRA_ARGS_RAW:+$COMMON_EXTRA_ARGS_RAW }--train-loss-every-steps 0"
    MAX_CANDIDATE_RATIO_DEFAULTS+=("stage.lm_head_backward.total_ms=1.000" "stage.lm_head_backward.ce.total_ms=1.000" "stage.lm_head_backward.dhidden.total_ms=1.000" "stage.lm_head_backward.dweight.total_ms=1.000")
    ;;
  "lm_head_prob_only_combined_corrections_threads_512"|"lm-head-prob-only-combined-corrections-threads-512"|"lm_head_ce_prob_only_combined_corrections_threads_512"|"lm-head-ce-prob-only-combined-corrections-threads-512")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-26 3-step, 1-sample stage-timed gate changed lm_head_prob_only_target_correction_threads from 256 to 512 and improved train_loop_wall_ms_per_step to 0.993286x plus stage.lm_head_backward.total_ms to 0.986391x in the non-cooperative diagnostic schedule, but rejected it because steady-state CUDA-event step time regressed to 1.001297x and candidate-over-llm.kittens train_loop_wall_ms_per_step remained 1.039342x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=0 NFN_NATIVE_GPT_LM_HEAD_FUSED_LOSS_BACKWARD=0 NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_COMBINED_CORRECTIONS=1 NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_TARGET_CORRECTION_THREADS=256"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=0 NFN_NATIVE_GPT_LM_HEAD_FUSED_LOSS_BACKWARD=0 NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_COMBINED_CORRECTIONS=1 NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_TARGET_CORRECTION_THREADS=512"
    COMMON_EXTRA_ARGS_RAW="${COMMON_EXTRA_ARGS_RAW:+$COMMON_EXTRA_ARGS_RAW }--train-loss-every-steps 0"
    ;;
  "bf16_persistent_block_outputs_direct_ln1"|"bf16-persistent-block-outputs-direct-ln1"|"bf16_persistent_block_input_ln1_backward"|"bf16-persistent-block-input-ln1-backward")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 3-step, 2-sample stage-timed gate activated direct BF16 LN1 backward for persistent block outputs and improved stage.block_backward.ln1_residual.total_ms to 0.890572x, but rejected it at 1.029093x train_loop_wall_ms_per_step, 1.002462x steady-state CUDA-event step time, and 1.048736x stage.block_backward.total_ms."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUTS=0 NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_INPUT_LN1_BACKWARD=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUTS=1 NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_INPUT_LN1_BACKWARD=1"
    ;;
  "bf16_persistent_block_outputs6"|"bf16-persistent-block-outputs6"|"bf16_persistent_block_outputs_6"|"bf16-persistent-block-outputs-6")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-26 3-step, 2-sample stage-timed gate reduced setup_wall_ms to 0.981054x and setup.float_arena_materialize.total_ms to 0.950760x, but rejected the partial six-output BF16 persistent route at 1.006527x train_loop_wall_ms_per_step, 1.005357x steady-state CUDA-event step time, 1.001062x stage.block_backward.total_ms, and 1.007163x stage.block_backward.mlp_proj.total_ms."
    CANDIDATE_NOTE="Tests a partial BF16 persistent-output route that stores the first six inter-block outputs as BF16 while keeping the remaining persistent outputs on the default FP32 direct path. This is a startup-memory candidate intended to find a smaller setup win than the rejected all-BF16 route without forcing every backward recompute through BF16 restore."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUTS=0 NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUT_COUNT=0 NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUT_PLACEMENT=head NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_INPUT_LN1_BACKWARD=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUT_COUNT=6 NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUT_PLACEMENT=head NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_INPUT_LN1_BACKWARD=0"
    MAX_CANDIDATE_RATIO_DEFAULTS+=("train_loop_wall_ms_per_step=1.000" "train_loop_cuda_event_steady_state_wall_ms_per_step=1.002")
    ;;
  "bf16_persistent_block_outputs_last6"|"bf16-persistent-block-outputs-last6"|"bf16_persistent_block_outputs_tail6"|"bf16-persistent-block-outputs-tail6")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-26 3-step, 2-sample stage-timed gate proved the tail placement route with bf16_persistent_block_output_start=5 and store/restore counts of 144, but rejected it at 1.007922x train_loop_wall_ms_per_step, 1.007089x steady-state CUDA-event step time, 1.001975x stage.block_backward.total_ms, 1.004262x stage.block_backward.mlp_proj.total_ms, and 1.004559x setup_wall_ms."
    CANDIDATE_NOTE="Tests the same six-output BF16 persistent-output diagnostic as bf16_persistent_block_outputs6, but stores the last six inter-block outputs as BF16 and keeps earlier persistent outputs on the default FP32 direct path."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUTS=0 NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUT_COUNT=0 NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUT_PLACEMENT=head NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_INPUT_LN1_BACKWARD=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUT_COUNT=6 NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUT_PLACEMENT=tail NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_INPUT_LN1_BACKWARD=0"
    MAX_CANDIDATE_RATIO_DEFAULTS+=("train_loop_wall_ms_per_step=1.000" "train_loop_cuda_event_steady_state_wall_ms_per_step=1.002")
    ;;
  "lm_head_ce_no_loss_llmk_style_specialized"|"lm-head-ce-no-loss-llmk-style-specialized"|"ce_bf16_no_loss_llmk_style_specialized"|"ce-bf16-no-loss-llmk-style-specialized")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 5-step, 2-sample same-script rerun after rebuilding the native selector proved the no-loss llm.kittens-style CE route but failed the strict default-vs-legacy gate: train_loop_wall_ms_per_step=1.000256x and train_tokens_per_second=0.999750x. A separate llm.kittens parity rerun with the route active still failed full-trainer parity at train_loop_wall_ms_per_step=1.002592x and train_loop_cuda_event_steady_state_wall_ms_per_step=1.002692x, so keep it diagnostic-only until the true fused classifier/dHidden/dWeight Tile kernel lands."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_LLMK_STYLE_SPECIALIZED=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_LLMK_STYLE_SPECIALIZED=1"
    COMMON_EXTRA_ARGS_RAW="${COMMON_EXTRA_ARGS_RAW:+$COMMON_EXTRA_ARGS_RAW }--train-loss-every-steps 0"
    ;;
  "lm_head_ce_no_loss_vec8_normal_store_specialized"|"lm-head-ce-no-loss-vec8-normal-store-specialized"|"ce_bf16_no_loss_vec8_normal_store_specialized"|"ce-bf16-no-loss-vec8-normal-store-specialized")
    ACCEPTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    DEFAULT_VS_LEGACY_PROFILE=1
    ACCEPTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 3-step, 2-sample stage-timed rerun promoted the no-loss CE vec8 normal-store specialized kernel as the default. It changed lm_head_ce_kernel_strategy to no-loss-specialized-dlogits-vec8-loads-normal-vec8-stores and measured train_loop_wall_ms_per_step=0.999856x, train_loop_cuda_event_steady_state_wall_ms_per_step=0.999483x, stage.lm_head_backward.total_ms=0.999986x, train_tokens_per_second=1.000147x, candidate-over-llm.kittens train_loop_wall_ms_per_step=0.999042x, and candidate-over-llm.kittens train_tokens_per_second=1.001246x. Promoted-default reruns stayed inside a 0.1% same-script jitter band: one measured train_loop_wall_ms_per_step=0.999826x, steady-state CUDA-event step time=0.999545x, train_tokens_per_second=1.000173x, and LM-head total=1.000553x; another measured train_loop_wall_ms_per_step=1.000254x, steady-state CUDA-event step time=1.000304x, train_tokens_per_second=0.999745x, LM-head total=1.000629x, and candidate-over-llm.kittens train_loop_wall_ms_per_step=0.999702x. This default-vs-legacy profile therefore gates full-loop and graph-wrapper LM-head ratios at 0.1% instead of exact zero-jitter. The CUDA Graph LM-head wrapper does not emit stage.lm_head_backward.ce.total_ms."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_VEC8_NORMAL_STORE_SPECIALIZED=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_VEC8_NORMAL_STORE_SPECIALIZED=1"
    COMMON_EXTRA_ARGS_RAW="${COMMON_EXTRA_ARGS_RAW:+$COMMON_EXTRA_ARGS_RAW }--train-loss-every-steps 0"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-train_loop_wall_ms_per_step=1.001 train_loop_cuda_event_steady_state_wall_ms_per_step=1.001 stage.lm_head_backward.total_ms=1.001}"
    MIN_CANDIDATE_RATIO_RAW="${MIN_CANDIDATE_RATIO_RAW:-train_tokens_per_second=0.999}"
    ;;
  "lm_head_ce_llmk_style_specialized"|"lm-head-ce-llmk-style-specialized"|"ce_bf16_llmk_style_specialized"|"ce-bf16-llmk-style-specialized")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 same-script gate proved the llm.kittens-style row-loss CE route and improved train_loop_wall_ms_per_step to 0.997562x, but rejected it because stage.lm_head_backward.total_ms regressed to 1.000511x and stage.lm_head_backward.ce.total_ms regressed to 1.000411x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_CE_LLMK_STYLE_SPECIALIZED=1"
    ;;
  "lm_head_ce_loss_bins_llmk_style_specialized"|"lm-head-ce-loss-bins-llmk-style-specialized"|"ce_bf16_loss_bins_llmk_style_specialized"|"ce-bf16-loss-bins-llmk-style-specialized")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 same-script gate kept the llm.kittens-style loss-bin CE route diagnostic-only after short-run loss-bin launch evidence was incomplete; rerun intentionally only when checking loss-bin route counters."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=1 NFN_NATIVE_GPT_LM_HEAD_CE_LLMK_STYLE_SPECIALIZED=1"
    COMMON_EXTRA_ARGS_RAW="${COMMON_EXTRA_ARGS_RAW:+$COMMON_EXTRA_ARGS_RAW }--train-loss-every-steps 1"
    ;;
  "lm_head_ce_loss_bins_default_specialized"|"lm-head-ce-loss-bins-default-specialized"|"ce_bf16_loss_bins_default_specialized"|"ce-bf16-loss-bins-default-specialized")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 3-step, 3-sample gate proved the default-shape loss-bin CE specialization and passed train_loop_wall_ms_per_step at 0.999215x, but rejected it on stage.lm_head_backward.total_ms 1.000741x, stage.lm_head_backward.ce.total_ms 1.000339x, and stage.block_backward.mlp_proj.total_ms 1.001222x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=1 NFN_NATIVE_GPT_LM_HEAD_CE_LOSS_BINS_DEFAULT_SPECIALIZED=1"
    COMMON_EXTRA_ARGS_RAW="${COMMON_EXTRA_ARGS_RAW:+$COMMON_EXTRA_ARGS_RAW }--train-loss-every-steps 1"
    ;;
  "lm_head_loss_bins"|"lm-head-loss-bins"|"lm_head_loss_bin_reduction"|"lm-head-loss-bin-reduction")
    CANDIDATE_NOTE="CUDA 13.3.33 dedicated RTX 5090 2026-06-25 3-step, 2-sample stage-timed rerun keeps the loss-bin train-loss logging route as the default because it moved lm_head_classifier_loss_bin_launch_count from 0 to 48, improved train_loop_wall_ms_per_step to 0.981541x, steady-state CUDA-event step time to 0.982697x, train_tokens_per_second to 1.018809x, stage.lm_head_backward.total_ms to 0.927229x, stage.block_backward.total_ms to 0.999905x, and stage.block_backward.mlp_proj.total_ms to 0.995141x versus the older row-loss tail."
    DEFAULT_VS_LEGACY_PROFILE=1
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=1"
    COMMON_EXTRA_ARGS_RAW="${COMMON_EXTRA_ARGS_RAW:+$COMMON_EXTRA_ARGS_RAW }--train-loss-every-steps 1"
    ;;
  "lm_head_loss_bins_bf16_workspace_prewarm"|"lm-head-loss-bins-bf16-workspace-prewarm"|"lm_head_loss_bin_reduction_bf16_workspace_prewarm"|"lm-head-loss-bin-reduction-bf16-workspace-prewarm")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 diagnostic gate passed only for the logged train-loss route: loss-bin launches changed 0->48 and BF16 workspace prewarm succeeded while train_loop_wall_ms_per_step improved to 0.960297x and steady-state CUDA-event timing to 0.976567x. Keep it rejected for default promotion because normal no-train-loss throughput does not execute the loss-bin tail, and forcing train-loss logging adds D2H copies."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=0 NFN_NATIVE_GPT_PREWARM_BF16_WORKSPACE=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=1 NFN_NATIVE_GPT_PREWARM_BF16_WORKSPACE=1"
    COMMON_EXTRA_ARGS_RAW="${COMMON_EXTRA_ARGS_RAW:+$COMMON_EXTRA_ARGS_RAW }--train-loss-every-steps 1"
    ;;
  "lm_head_row_loss_sum_accumulate"|"lm-head-row-loss-sum-accumulate"|"lm_head_loss_sum_accumulate"|"lm-head-loss-sum-accumulate")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 3-step, 2-sample stage-timed rerun changed the row-loss tail back to sum-accumulate, but failed the strict gates at 1.000970x train_loop_cuda_event_steady_state_wall_ms_per_step and 1.000304x stage.lm_head_backward.total_ms."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE=1"
    ;;
  "lm_head_row_loss_partial_reduce"|"lm-head-row-loss-partial-reduce"|"lm_head_row_loss_sum_accumulate_off"|"lm-head-row-loss-sum-accumulate-off")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 rerun changed row-loss accumulation from sum-accumulate to partial-reduce and improved steady-state CUDA-event timing to 0.999166x, but rejected it because train_loop_wall_ms_per_step regressed to 1.002484x and train_tokens_per_second fell to 0.997528x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE=1"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE=0"
    ;;
  "cublaslt_min_waves"|"cublaslt-min-waves")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 same-script recheck changed cuBLASLt selected heuristics but failed stage gates: train_loop_wall_ms_per_step was 0.998752x, but LM-head backward regressed to 1.001151x and MLP projection total to 1.020829x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_POLICY=min_waves"
    ;;
  "cublaslt_max_waves"|"cublaslt-max-waves")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 same-script recheck changed cuBLASLt selected heuristics but regressed train_loop_wall_ms_per_step to 1.010956x, LM-head backward to 1.007568x, block backward to 1.025454x, and attention projection dWeight+bias to 1.400435x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_POLICY=max_waves"
    ;;
  "cublaslt_heavy_shape_flip"|"cublaslt-heavy-shape-flip"|"cublaslt_block_heavy_flip"|"cublaslt-block-heavy-flip")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-26 3-step, 2-sample stage-timed shape-stat gate flipped the heavy block cuBLASLt plans for 768x65536x3072:N,N, 768x65536x2304:N,N, and 768x2304x65536:N,T. The candidate proved plan-cache and linear-shape changes, but regressed train_loop_wall_ms_per_step to 1.002525x, steady-state CUDA-event timing to 1.005491x, block backward to 1.011881x, MLP FC backward to 1.031422x, and QKV backward to 1.029108x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=3072,768,65536,N,T,1:768,3072,65536,N,T,1:768,65536,3072,N,N,0:768,65536,2304,N,N,0:768,2304,65536,N,T,0"
    ;;
  "cublaslt_block_dinput"|"cublaslt-block-dinput"|"block_dinput_cublaslt"|"block-dinput-cublaslt")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 3-step, 2-sample stage-timed gate passed timing ratios but rejected promotion because the actual native trainer already used the same cuBLASLt dInput plans: route counters, strategy values, linear shape route names, and plan cache entries did not change."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE=3072,65536,768,N,N:768,65536,3072,N,N:768,65536,2304,N,N:768,65536,768,N,N"
    ;;
  "cublaslt_block_dinput_h3_65536"|"cublaslt-block-dinput-h3-65536"|"block_dinput_cublaslt_h3_65536"|"block-dinput-cublaslt-h3-65536")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 3-sample same-script confirmation changed only the intended cuBLASLt dInput plans but regressed train_loop_wall_ms_per_step to 1.005964x, LM-head backward to 1.011344x, block backward to 1.004976x, and MLP projection total to 1.010875x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,65536,3072,N,N,3:768,65536,2304,N,N,3"
    ;;
  "cublaslt_qkv_dweight_h0_65536"|"cublaslt-qkv-dweight-h0-65536"|"qkv_dweight_cublaslt_h0_65536"|"qkv-dweight-cublaslt-h0-65536")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 same-script stage-timed gate changed the QKV dWeight+bias cuBLASLt plan from heuristic 1 to 0 but regressed stage.block_backward.qkv.dweight_bias.total_ms to 1.003363x and block backward to 1.000055x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,2304,65536,N,T,0"
    ;;
  "cublaslt_attn_proj_dweight_h0_65536"|"cublaslt-attn-proj-dweight-h0-65536"|"attn_proj_dweight_cublaslt_h0_65536"|"attn-proj-dweight-cublaslt-h0-65536")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 same-script stage-timed gate for NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,768,65536,N,T,0 changed no tracked route counters, strategy values, or cuBLASLt plan-cache entries, then failed strict gates on steady-state CUDA-event timing, LM-head backward, and MLP projection. Treat the apparent attention-projection timing win as noise until a real route or plan change is visible."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,768,65536,N,T,0"
    ;;
  "cublaslt_grouped_probe"|"cublaslt-grouped-probe"|"grouped_cublaslt_probe"|"grouped-cublaslt-probe")
    CANDIDATE_NOTE="CUDA 13.3.33 dedicated RTX 5090 probe currently reports grouped layout status 0 with grouped matmul status 15; the classic cuBLAS grouped BF16 probe still fails separately with status 700. This profile is telemetry-only until grouped matmul is supported."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_PROBE_CUBLASLT_GROUPED_LAYOUT=1 NFN_NATIVE_GPT_PROBE_CUBLASLT_GROUPED_MATMUL=1"
    AUTO_DISABLE_METRIC_RATIO_GATES=1
    ;;
  "cublaslt_grouped_probe_required"|"cublaslt-grouped-probe-required"|"grouped_cublaslt_probe_required"|"grouped-cublaslt-probe-required")
    STRICT_PROBE_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    STRICT_GROUPED_CUBLASLT_PROBE=1
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_PROBE_CUBLASLT_GROUPED_LAYOUT=1 NFN_NATIVE_GPT_PROBE_CUBLASLT_GROUPED_MATMUL=1"
    AUTO_DISABLE_METRIC_RATIO_GATES=1
    ;;
  "tk_dgelu_dinput"|"tk-dgelu-dinput")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 recheck showed this compile-flag profile is no longer a valid route candidate: the linked baseline already reports linear_tk_dgelu_dinput_gemm_count=288, the generated candidate reports the same route counters, and the native route-change gate fails. Keep the default fused TK dGELU dInput route; use this profile only for intentional historical no-op/rejection reproduction."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU=1"
    CANDIDATE_TILE_OPS_BUILD_FLAGS="${CANDIDATE_TILE_OPS_BUILD_FLAGS:+$CANDIDATE_TILE_OPS_BUILD_FLAGS }-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP"
    ;;
  "tk_dgelu_approx_tanh"|"tk-dgelu-approx-tanh")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 recheck of the non-approx dGELU compile-flag profile showed the linked baseline already uses the fused TK dGELU dInput route. The tanh approximation variant is therefore historical/diagnostic-only until it proves a distinct route and passes same-script gates."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU=1"
    CANDIDATE_TILE_OPS_BUILD_FLAGS="${CANDIDATE_TILE_OPS_BUILD_FLAGS:+$CANDIDATE_TILE_OPS_BUILD_FLAGS }-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP -DLLMK_SM120_APPROX_DGELU_TANH=1"
    ;;
  "mlp_proj_dgelu_fallback"|"mlp-proj-dgelu-fallback"|"mlp_proj_dgelu_cublas_fallback"|"mlp-proj-dgelu-cublas-fallback")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-27 3-step, 2-sample stage-timed gate proved the intended route movement by dropping linear_tk_dgelu_dinput_gemm_count from 288 to 0 and block_backward_dinput_tk_gemm_count from 288 to 0, but rejected promotion because train_loop_wall_ms_per_step regressed to 1.013580x, steady-state CUDA-event timing to 1.012958x, block backward to 1.027454x, MLP projection total to 1.107897x, MLP projection dInput to 1.207964x, and candidate-over-llm.kittens train-loop wall to 1.016979x."
    CANDIDATE_NOTE="Disables only the fused TK dGELU dInput shape used by the GPT MLP projection backward path, leaving the default fused dGELU feature enabled. The route gate must show linear_tk_dgelu_dinput_gemm_count dropping and fallback GEMM counters moving before any timing result is considered."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_TK_DGELU_DINPUT_DISABLE_SHAPE=3072,65536,768,N,N"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:+$MAX_CANDIDATE_RATIO_RAW }train_loop_wall_ms_per_step=1.000 train_loop_cuda_event_steady_state_wall_ms_per_step=1.000 stage.block_backward.total_ms=1.000 stage.block_backward.mlp_proj.total_ms=1.000 stage.block_backward.mlp_proj.dinput.total_ms=1.000"
    MIN_CANDIDATE_RATIO_RAW="${MIN_CANDIDATE_RATIO_RAW:+$MIN_CANDIDATE_RATIO_RAW }train_tokens_per_second=1.000"
    ;;
  "attention_atomic_dq"|"attention-atomic-dq")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 same-script gate failed route detection and regressed attention-backward section gates."
    CANDIDATE_TILE_OPS_BUILD_FLAGS="${CANDIDATE_TILE_OPS_BUILD_FLAGS:+$CANDIDATE_TILE_OPS_BUILD_FLAGS }-DLLMK_SM120_ATOMIC_DQ"
    ;;
  "attention_bwd_block_32"|"attention-bwd-block-32"|"attention_backward_block_32"|"attention-backward-block-32")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 post-Bsymbolic 1-step, 1-sample stage-timed gate compiled the TK attention backward block size with -DLLMK_SM120_ATTN_BWD_BLOCK=32 and proved the route changed from attention_backward_tk_block_size=16 to 32, but rejected it because attention_backward_tk_timing_us regressed to 1.058092x and stage.lm_head_backward.total_ms missed the strict gate at 1.002283x."
    CANDIDATE_TILE_OPS_BUILD_FLAGS="${CANDIDATE_TILE_OPS_BUILD_FLAGS:+$CANDIDATE_TILE_OPS_BUILD_FLAGS }-DLLMK_SM120_ATTN_BWD_BLOCK=32"
    AUTO_ATTENTION_SECTION_TIMING=1
    ;;
  "attention_bwd_block_64"|"attention-bwd-block-64"|"attention_backward_block_64"|"attention-backward-block-64")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 post-Bsymbolic 1-step, 1-sample stage-timed gate compiled the TK attention backward block size with -DLLMK_SM120_ATTN_BWD_BLOCK=64 and proved the route changed from attention_backward_tk_block_size=16 to 64, but rejected it because train_loop_wall_ms_per_step regressed to 3.343485x, stage.block_backward.total_ms to 5.034009x, and attention_backward_tk_timing_us to 24.139285x."
    CANDIDATE_TILE_OPS_BUILD_FLAGS="${CANDIDATE_TILE_OPS_BUILD_FLAGS:+$CANDIDATE_TILE_OPS_BUILD_FLAGS }-DLLMK_SM120_ATTN_BWD_BLOCK=64"
    AUTO_ATTENTION_SECTION_TIMING=1
    ;;
  "bf16_attention_grad_out"|"bf16-attention-grad-out"|"attention_bf16_grad_out"|"attention-bf16-grad-out")
    DEFAULT_VS_LEGACY_PROFILE=1
    CANDIDATE_NOTE="Compares the promoted default BF16 attention grad-out handoff against the legacy direct float32 scratch route. CUDA 13.3 dedicated RTX 5090 actual-training 5-step, 2-sample promotion gate measured the promoted route at 0.999028x current NeuralFn train-loop wall, 1.000975x current NeuralFn tokens/sec, 0.998462x llm.kittens reference train-loop wall, and 1.001921x llm.kittens reference tokens/sec."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-train_loop_wall_ms_per_step=1.000}"
    MIN_CANDIDATE_RATIO_RAW="${MIN_CANDIDATE_RATIO_RAW:-train_tokens_per_second=1.000}"
    ;;
  "bf16_attention_dprep_grad_out"|"bf16-attention-dprep-grad-out"|"attention_bf16_dprep_grad_out"|"attention-bf16-dprep-grad-out")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 same-script gates rejected the BF16 dprep-only attention grad-out route: the latest gate measured 1.005344x train_loop_wall_ms_per_step, 1.010632x stage.block_backward.total_ms, and 1.044177x attention backward."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_BF16_ATTENTION_DPREP_GRAD_OUT=1"
    ;;
  "packed_attention_bwd_batch_128"|"packed-attention-bwd-batch-128"|"attention_bwd_batch_128"|"attention-bwd-batch-128"|"packed_attention_backward_batch_128"|"packed-attention-backward-batch-128")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-25 attention-section gate changed attention_backward_tk_batch_cap from 64 to 128, but rejected default promotion because train_loop_wall_ms_per_step regressed to 1.013207x, steady-state CUDA-event timing to 1.000470x, stage.block_backward.total_ms to 1.029008x, attention_backward_tk_timing_us to 1.002850x, and attention_backward_dprep_timing_us to 1.000088x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP=128"
    AUTO_ATTENTION_SECTION_TIMING=1
    ;;
  "packed_attention_bwd_batch_96"|"packed-attention-bwd-batch-96"|"attention_bwd_batch_96"|"attention-bwd-batch-96"|"packed_attention_backward_batch_96"|"packed-attention-backward-batch-96")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-25 3-step, 2-sample current/native/reference attention-section gate changed attention_backward_tk_batch_cap from 64 to 96, but rejected default promotion because stage.block_backward.attn_sdpa.total_ms regressed to 1.000597x, stage.block_backward.attn_sdpa.to_qkv.total_ms to 1.000638x, attention_backward_dprep_timing_us to 1.000149x, and candidate-over-llm.kittens train_loop_wall_ms_per_step remained 1.035674x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP=96"
    AUTO_ATTENTION_SECTION_TIMING=1
    ;;
  "packed_attention_bwd_batch_32"|"packed-attention-bwd-batch-32"|"attention_bwd_batch_32"|"attention-bwd-batch-32"|"packed_attention_backward_batch_32"|"packed-attention-backward-batch-32")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-25 attention-section gate changed attention_backward_tk_batch_cap from 64 to 32 and doubled attention_backward_tk_launch_count from 288 to 576, but rejected default promotion because train_loop_wall_ms_per_step regressed to 1.010819x, steady-state CUDA-event timing to 1.009021x, stage.block_backward.attn_sdpa.to_qkv.total_ms to 1.077143x, and attention_backward_tk_timing_us to 1.066848x despite dprep timing improving to 0.943271x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP=32"
    AUTO_ATTENTION_SECTION_TIMING=1
    ;;
  "attention_dprep_float_hd64_specialized"|"attention-dprep-float-hd64-specialized"|"float_attention_dprep_hd64"|"float-attention-dprep-hd64")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 same-script gate proved the float HD64 dprep route and improved dprep timing, but rejected default promotion because adjacent hot-stage gates regressed."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_FLOAT_HD64_SPECIALIZED=1"
    ;;
  "attention_dprep_grid3d"|"attention-dprep-grid3d"|"packed_attention_dprep_grid3d"|"packed-attention-dprep-grid3d")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 paired confirmation changed the packed-attention dprep route to the 3D batch/head/time launch, but rejected default promotion because the 5-sample confirmation regressed train_loop_wall_ms_per_step to 1.008389x and train_tokens_per_second to 0.991895x versus the row-linear dprep default."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_GRID3D=1"
    AUTO_ATTENTION_SECTION_TIMING=1
    ;;
  "attention_dprep_warps_2"|"attention-dprep-warps-2"|"packed_attention_dprep_warps_2"|"packed-attention-dprep-warps-2")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 2-step, 1-sample stage-timed gate changed the packed-attention dprep warp count from 3 to 2, but rejected it because train_loop_wall_ms_per_step regressed to 1.004595x, steady-state CUDA-event timing to 1.008058x, stage.block_backward.total_ms to 1.017877x, and attention_backward_tk_timing_us to 1.002175x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_WARPS=2"
    AUTO_ATTENTION_SECTION_TIMING=1
    ;;
  "attention_dprep_warps_4"|"attention-dprep-warps-4"|"packed_attention_dprep_warps_4"|"packed-attention-dprep-warps-4")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 2-step, 1-sample stage-timed gate changed the packed-attention dprep warp count from 3 to 4, but rejected it because train_loop_wall_ms_per_step regressed to 1.005938x, stage.block_backward.total_ms to 1.020308x, stage.block_backward.attn_sdpa.total_ms to 1.001733x, and attention_backward_tk_timing_us to 1.001117x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_WARPS=4"
    AUTO_ATTENTION_SECTION_TIMING=1
    ;;
  "packed_attention_saved_lse_off"|"packed-attention-saved-lse-off"|"attention_saved_lse_off"|"attention-saved-lse-off")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 3-step, 2-sample stage-timed gate disabled stored packed-attention LSE and rejected the older no-LSE route because steady-state CUDA-event timing regressed to 1.002521x, stage.block_backward.attn_sdpa.to_qkv.total_ms to 1.002141x, attention_backward_tk_timing_us to 1.001853x, and attention_backward_dprep_timing_us to 1.001978x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE=1"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE=0"
    AUTO_ATTENTION_SECTION_TIMING=1
    ;;
  "mlp_proj_dinput_before_dweight"|"mlp-proj-dinput-before-dweight")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-25 3-step, 2-sample stage-timed rerun proved block_backward_mlp_proj_dinput_before_dweight_count moved 0->288 and kept mean train_loop_wall_ms_per_step near-flat at 0.999180x, but rejected default promotion because the target stage.block_backward.mlp_proj.dinput.total_ms regressed to 1.101843x and stage.block_backward.mlp_proj.total_ms regressed to 1.001268x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_MLP_PROJ_DINPUT_BEFORE_DWEIGHT=1"
    ;;
  "mlp_fc_dinput_before_dweight"|"mlp-fc-dinput-before-dweight")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-26 3-step, 2-sample post-reinstall recheck proved the route by moving block_backward_mlp_fc_dinput_before_dweight_count from 0 to 288 and kept train_loop_wall_ms_per_step faster at 0.998065x, but rejected the route because steady-state CUDA-event timing regressed to 1.001167x, block backward to 1.001447x, LM-head backward to 1.000127x, MLP projection backward to 1.004199x, and MLP FC backward to 1.003817x. The native default is restored to dWeight+bias before dInput; set NFN_NATIVE_GPT_MLP_FC_DINPUT_BEFORE_DWEIGHT=1 only for intentional diagnostics."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_MLP_FC_DINPUT_BEFORE_DWEIGHT=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_MLP_FC_DINPUT_BEFORE_DWEIGHT=1"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-train_loop_wall_ms_per_step=1.000 train_loop_cuda_event_steady_state_wall_ms_per_step=1.000 stage.block_backward.total_ms=1.000 stage.lm_head_backward.total_ms=1.000}"
    ;;
  "attn_proj_dinput_before_dweight"|"attn-proj-dinput-before-dweight")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 rebuilt-binary 5-step, 3-sample stage-timed confirmation proved block_backward_attn_proj_dinput_before_dweight_count moved 0->480, but rejected default promotion because train_loop_wall_ms_per_step regressed to 1.001501x, stage.lm_head_backward.total_ms to 1.000290x, stage.block_backward.total_ms to 1.003886x, stage.block_backward.mlp_proj.total_ms to 1.002417x, and stage.block_backward.attn_proj.total_ms to 1.081569x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_ATTN_PROJ_DINPUT_BEFORE_DWEIGHT=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_ATTN_PROJ_DINPUT_BEFORE_DWEIGHT=1"
    ;;
  "qkv_dinput_before_dweight"|"qkv-dinput-before-dweight")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 standalone rerun proved the route counter moved 0->288 and improved train_loop_wall_ms_per_step to 0.994580x, but rejected standalone default promotion because train_loop_cuda_event_steady_state_wall_ms_per_step regressed to 1.000517x, stage.lm_head_backward.total_ms to 1.000122x, stage.block_backward.mlp_proj.total_ms to 1.002745x, and stage.block_backward.qkv.total_ms to 1.000860x. The route is default only as part of the later qkv_dinput_ln128 combined default."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_QKV_DINPUT_BEFORE_DWEIGHT=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_QKV_DINPUT_BEFORE_DWEIGHT=1"
    ;;
  "qkv_dinput_ln128"|"qkv-dinput-ln128"|"qkv_dinput_before_dweight_ln128"|"qkv-dinput-before-dweight-ln128"|"qkv_order_ln128"|"qkv-order-ln128")
    ACCEPTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    PROMOTED_QKV_LN128_PROFILE=1
    DEFAULT_VS_LEGACY_PROFILE=1
    ACCEPTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 current-code 5-step, 2-sample same-script rerun kept the combined QKV dInput-before-dWeight plus 128-row LayerNorm affine route as the default: train_loop_wall_ms_per_step=0.998106x, train_tokens_per_second=1.001904x, candidate-over-llm.kittens train_loop_wall_ms_per_step=0.998347x, and candidate-over-llm.kittens train_tokens_per_second=1.001415x. Route proof moved block_backward_qkv_dinput_before_dweight_count from 0 to 480 and block_state_layout.layer_norm_backward_affine_row_chunk_size from 256 to 128."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_QKV_DINPUT_BEFORE_DWEIGHT=0 NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=256"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_QKV_DINPUT_BEFORE_DWEIGHT=1 NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=128"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-train_loop_wall_ms_per_step=1.000 train_loop_cuda_event_steady_state_wall_ms_per_step=1.002 stage.block_backward.total_ms=1.000}"
    ;;
  "qkv_dinput_ln64"|"qkv-dinput-ln64"|"qkv_dinput_before_dweight_ln64"|"qkv-dinput-before-dweight-ln64"|"qkv_order_ln64"|"qkv-order-ln64")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 5-step, 3-sample stage-timed confirmation combined QKV dInput-before-dWeight with the 64-row LayerNorm affine reducer and improved steady-state CUDA-event timing to 0.998529x, but rejected default promotion because train_loop_wall_ms_per_step regressed to 1.000261x, stage.lm_head_backward.total_ms to 1.000067x, stage.block_backward.total_ms to 1.000938x, stage.block_backward.mlp_proj.total_ms to 1.004308x, and stage.block_backward.qkv.total_ms to 1.007310x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_QKV_DINPUT_BEFORE_DWEIGHT=1 NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=64"
    ;;
  "lm_head_fused_loss_backward_off"|"lm-head-fused-loss-backward-off"|"lm_head_separate_loss_backward"|"lm-head-separate-loss-backward")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 5-step, 3-sample confirmation disabled the default fused LM-head loss-accumulate+dlogits classifier path, but rejected the older separate loss partial reduction plus CE backward route at 1.001484x train-loop wall, 1.003244x block backward, and 1.000194x MLP projection backward."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_FUSED_LOSS_BACKWARD=0"
    ;;
  "lm_head_classifier_ce_no_loss"|"lm-head-classifier-ce-no-loss"|"lm_head_no_loss_classifier_ce"|"lm-head-no-loss-classifier-ce")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 3-step, 2-sample stage-timed gate changed the classifier CE route but regressed train_loop_wall_ms_per_step to 1.005933x, stage.lm_head_backward.total_ms to 1.087310x, and stage.lm_head_backward.ce.total_ms to 1.848303x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_CLASSIFIER_CE_NO_LOSS=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_CLASSIFIER_CE_NO_LOSS=1"
    ;;
  "cublas_handle_prewarm"|"cublas-handle-prewarm"|"linear_cublas_handle_prewarm"|"linear-cublas-handle-prewarm")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 5-step, 3-sample stage-timed gate initialized the non-Lt cuBLAS handle in setup but regressed train_loop_wall_ms_per_step to 1.000699x, stage.lm_head_backward.total_ms to 1.000673x, stage.block_backward.total_ms to 1.000035x, and stage.block_backward.mlp_proj.total_ms to 1.001440x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_PREWARM_CUBLAS_HANDLE=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_PREWARM_CUBLAS_HANDLE=1"
    ;;
  "cuda_module_eager"|"cuda-module-eager"|"module_loading_eager"|"module-loading-eager")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-26 5-step, 3-sample rerun after the WSL CUDA reinstall compared CUDA_MODULE_LOADING=EAGER against the default LAZY native wrapper route. EAGER did not change tracked kernel routes, regressed setup_wall_ms to 3.467504x, train_loop_wall_ms_per_step to 1.009903x, first-step CUDA-event timing to 1.048444x, and train_tokens_per_second to 0.990197x, so keep LAZY as the default."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }CUDA_MODULE_LOADING=LAZY"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }CUDA_MODULE_LOADING=EAGER"
    ;;
  "bf16_workspace_prewarm"|"bf16-workspace-prewarm"|"linear_bf16_workspace_prewarm"|"linear-bf16-workspace-prewarm")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 5-step, 3-sample rerun changed only setup/prewarm counters and rejected default promotion because train_loop_wall_ms_per_step regressed to 1.000826x, steady-state CUDA-event time to 1.000283x, LM-head backward to 1.000252x, block backward to 1.001255x, and MLP projection backward to 1.000923x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_PREWARM_BF16_WORKSPACE=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_PREWARM_BF16_WORKSPACE=1"
    ;;
  "cublaslt_plan_prewarm_block_only"|"cublaslt-plan-prewarm-block-only"|"linear_cublaslt_plan_prewarm_block_only"|"linear-cublaslt-plan-prewarm-block-only")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-25 split-stage 3-step, 2-sample rerun skipped the single LM-head cuBLASLt prewarm plan and improved setup_wall_ms to 0.989552x, but rejected default promotion because train-loop wall regressed to 1.001350x, first-step CUDA-event time to 1.002789x, forward QKV first-step avg to 1.022751x, and no route, strategy, or plan-cache change passed the native route gate."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLANS=1 NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLAN_MODE=all"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLANS=1 NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLAN_MODE=block_only"
    ;;
  "cublaslt_plan_prewarm_lm_head_only"|"cublaslt-plan-prewarm-lm-head-only"|"linear_cublaslt_plan_prewarm_lm_head_only"|"linear-cublaslt-plan-prewarm-lm-head-only")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 3-step, 2-sample gate prewarmed only the LM-head cuBLASLt plan and saved setup_wall_ms to 0.947250x, but rejected default promotion because train_loop_wall_ms_per_step regressed to 1.011688x, steady-state CUDA-event time to 1.001999x, LM-head backward to 1.000280x, block backward to 1.022887x, and MLP projection backward to 1.021800x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLANS=1 NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLAN_MODE=all"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLANS=1 NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLAN_MODE=lm_head_only"
    ;;
  "cublaslt_plan_prewarm_off"|"cublaslt-plan-prewarm-off"|"linear_cublaslt_plan_prewarm_off"|"linear-cublaslt-plan-prewarm-off")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 3-step, 2-sample gate disabled cuBLASLt plan prewarm and improved setup_wall_ms to 0.834325x, but rejected default promotion because train_loop_wall_ms_per_step regressed to 1.015300x, first-step CUDA-event time to 1.044809x, train_tokens_per_second to 0.984974x, LM-head backward to 1.031614x, and block backward to 1.023253x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLANS=1 NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLAN_MODE=all"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLANS=0"
    ;;
  "tk_forward_no_n96"|"tk-forward-no-n96"|"llmk_forward_no_n96"|"llmk-forward-no-n96")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 recheck built the -DLLMK_SM120_FORWARD_N96=0 Tile ops candidate but did not change tracked route counters, strategy strings, linear shape stats, or cuBLASLt plan cache entries; the route-change gate failed and hot-stage gates regressed at stage.lm_head_backward.total_ms=1.001484x and stage.block_backward.mlp_proj.total_ms=1.001994x."
    CANDIDATE_TILE_OPS_BUILD_FLAGS="${CANDIDATE_TILE_OPS_BUILD_FLAGS:+$CANDIDATE_TILE_OPS_BUILD_FLAGS }-DLLMK_SM120_FORWARD_N96=0"
    ;;
  "llmk_sm120_reference_flags"|"llmk-sm120-reference-flags"|"llm_kittens_sm120_reference_flags"|"llm-kittens-sm120-reference-flags")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-24 post-rebuild 5-step, 2-sample stage-timed gate rebuilt the documented llm.kittens SM120 reference macro bundle, changed no tracked route or strategy counters, improved train-loop wall to 0.995837x, train tokens/sec to 1.004176x, and block backward to 0.991665x, but rejected default promotion because steady-state CUDA-event timing missed at 1.000937x while MLP FC regressed to 1.006521x and QKV regressed to 1.008280x."
    FORCE_DISABLE_ROUTE_CHANGE=1
    CANDIDATE_TILE_OPS_BUILD_FLAGS="${CANDIDATE_TILE_OPS_BUILD_FLAGS:+$CANDIDATE_TILE_OPS_BUILD_FLAGS }-DLLMK_SM120_USE_CUBLASLT_GEMM -DLLMK_SM120_ATTN_FWD_BLOCK=32 -DLLMK_SM120_ATTN_BWD_BLOCK=16 -DLLMK_SM120_BIAS_BLOCK_SIZE=512 -DLLMK_SM120_K_TILE=32 -DLLMK_SM120_SUPER_M=8 -DLLMK_SM120_HUGE_N_K_TILE=64 -DLLMK_SM120_GRAD_K_TILE=64 -DLLMK_SM120_DINP_SUPER_M=LLMK_SM120_SUPER_M -DLLMK_SM120_DWEIGHT_SUPER_M=2 -DLLMK_SM120_INPLACE_LAYOUT_SWAP=1 -DLLMK_SM120_FAST_DGELU=1 -DLLMK_SM120_APPROX_DGELU_TANH=1 -DLLMK_SM120_DPREP_WARPS=3 -DLLMK_SM120_MEMORY_BLOCK_SIZE=1024 -DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1"
    ;;
  "tk_sm120_super_m7"|"tk-sm120-super-m7"|"super_m7"|"super-m7"|"tk_super_m7"|"tk-super-m7")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 3-step, 2-sample stage-timed gate compiled TK GEMM with LLMK_SM120_SUPER_M=7 and LLMK_SM120_DINP_SUPER_M=7, proved strategy telemetry changed super_m and dinput_super_m from 8 to 7, but rejected default promotion because steady-state CUDA-event timing regressed to 1.000992x, LM-head backward to 1.000168x, and MLP projection total to 1.001198x."
    CANDIDATE_TILE_OPS_BUILD_FLAGS="${CANDIDATE_TILE_OPS_BUILD_FLAGS:+$CANDIDATE_TILE_OPS_BUILD_FLAGS }-DLLMK_SM120_SUPER_M=7 -DLLMK_SM120_DINP_SUPER_M=7"
    ;;
  "tk_sm120_super_m13"|"tk-sm120-super-m13"|"super_m13"|"super-m13"|"tk_super_m13"|"tk-super-m13")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 3-step, 2-sample stage-timed gate compiled TK GEMM with LLMK_SM120_SUPER_M=13 and LLMK_SM120_DINP_SUPER_M=13, proved strategy telemetry changed super_m and dinput_super_m from 8 to 13, but rejected default promotion because train-loop wall regressed to 1.009116x, steady-state CUDA-event timing to 1.002623x, block backward to 1.011813x, and MLP projection total to 1.010002x."
    CANDIDATE_TILE_OPS_BUILD_FLAGS="${CANDIDATE_TILE_OPS_BUILD_FLAGS:+$CANDIDATE_TILE_OPS_BUILD_FLAGS }-DLLMK_SM120_SUPER_M=13 -DLLMK_SM120_DINP_SUPER_M=13"
    ;;
  "cuda_device_max_connections_1"|"cuda-device-max-connections-1")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="This profile is a no-op in the SM120 paired wrapper: CUDA_DEVICE_MAX_CONNECTIONS already defaults to 1 for both baseline and candidate commands, matching the llm.kittens SM120 launcher policy."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }CUDA_DEVICE_MAX_CONNECTIONS=1"
    ;;
  "combined_device_arena"|"combined-device-arena")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-25 recheck still rejects the combined float+uint16 transformer device arena. The startup-only 5-sample gate changed float_allocation_strategy/uint16_allocation_strategy to combined-transformer-device-arena and moved transformer_device_arena_cuda_malloc_count 0->1, but regressed setup_wall_ms to 1.031475x, setup.uint16_arena_materialize.total_ms to 2.339592x, and setup.token_weight_init.total_ms to 1.289567x."
    STARTUP_ONLY=1
    STEPS=0
    FORCE_DISABLE_ROUTE_CHANGE=1
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_COMBINED_DEVICE_ARENA=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_COMBINED_DEVICE_ARENA=1"
    ;;
  "cuda_malloc_async"|"cuda-malloc-async"|"async_device_allocator"|"async-device-allocator")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 3-sample startup-only rerun enabled cudaMallocAsync and improved setup.token_weight_init.total_ms to 0.803693x, but rejected it because setup_wall_ms regressed to 1.126997x, setup.float_arena_materialize.total_ms to 1.175710x, and setup.uint16_arena_materialize.total_ms to 1.709907x."
    STARTUP_ONLY=1
    STEPS=0
    FORCE_DISABLE_ROUTE_CHANGE=1
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC=1"
    ;;
  "concurrent_arena_materialize"|"concurrent-arena-materialize"|"parallel_arena_materialize"|"parallel-arena-materialize")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-25 3-sample startup-only gate overlapped the float and uint16 arena cudaMalloc calls and moved concurrent_arena_materialize_count 0->1, but rejected default promotion because the setup_wall_ms median regressed to 1.003922x and uint16_arena_cuda_malloc_wall_ms regressed to 2.664592x mean despite a noisy 0.987871x mean setup wall."
    STARTUP_ONLY=1
    STEPS=0
    FORCE_DISABLE_ROUTE_CHANGE=1
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_CONCURRENT_ARENA_MATERIALIZE=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_CONCURRENT_ARENA_MATERIALIZE=1"
    ;;
  "uint16_arena_first"|"uint16-arena-first"|"bf16_arena_first"|"bf16-arena-first")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 3-sample startup-only rerun changed arena_materialize_order from float-then-uint16 to uint16-then-float, but rejected default promotion because setup_wall_ms regressed to 1.007289x, setup.uint16_arena_materialize.total_ms to 2.191721x, and setup.token_weight_init.total_ms to 1.057017x."
    STARTUP_ONLY=1
    STEPS=0
    FORCE_DISABLE_ROUTE_CHANGE=1
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_UINT16_ARENA_FIRST=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_UINT16_ARENA_FIRST=1"
    ;;
  "host_descriptor_reserve"|"host-descriptor-reserve"|"descriptor_reserve"|"descriptor-reserve")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 startup-only reruns rejected host descriptor-vector reserve as a default. The first 3-sample run improved setup_wall_ms to 0.980086x and the AdamW/parameter-fill descriptor buckets to 0.705706x/0.671318x, but the immediate rerun regressed setup_wall_ms to 1.016598x; keep it diagnostic-only because the improvement is not robust against paired-sample noise."
    STARTUP_ONLY=1
    STEPS=0
    FORCE_DISABLE_ROUTE_CHANGE=1
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_HOST_DESCRIPTOR_RESERVE=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_HOST_DESCRIPTOR_RESERVE=1"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-setup_wall_ms=1.000 setup.build_adamw_descriptors.total_ms=1.000 setup.build_parameter_fill_descriptors.total_ms=1.000}"
    ;;
  "store_mlp_blocks6"|"store-mlp-blocks6"|"stored_mlp_blocks6"|"stored-mlp-blocks6"|"mlp_activation_blocks6"|"mlp-activation-blocks6")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 startup-only gate improved setup_wall_ms to 0.884111x by cutting stored MLP activation blocks 12->6, but rejected default promotion because the 2-step, 2-sample training gate regressed train_loop_wall_ms_per_step to 1.179180x, steady-state CUDA-event step time to 1.181891x, block backward to 1.153015x, and MLP projection to 1.546830x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_STORE_MLP_BLOCKS=12"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_STORE_MLP_BLOCKS=6"
    ;;
  "store_mlp_blocks3"|"store-mlp-blocks3"|"stored_mlp_blocks3"|"stored-mlp-blocks3"|"mlp_activation_blocks3"|"mlp-activation-blocks3")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 3-step, 2-sample stage-timed gate cut stored MLP activation blocks 12->3 and improved setup_wall_ms to 0.908060x, but rejected default promotion because train_loop_wall_ms_per_step regressed to 1.278677x, steady-state CUDA-event step time to 1.277148x, train_tokens_per_second to 0.782060x, block backward to 1.217694x, and MLP projection backward to 1.458270x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_STORE_MLP_BLOCKS=12"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_STORE_MLP_BLOCKS=3"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-train_loop_wall_ms_per_step=1.001 train_loop_cuda_event_steady_state_wall_ms_per_step=1.002 stage.block_backward.total_ms=1.001 stage.block_backward.mlp_proj.total_ms=1.001}"
    MIN_CANDIDATE_RATIO_RAW="${MIN_CANDIDATE_RATIO_RAW:-train_tokens_per_second=0.999}"
    ;;
  "store_mlp_blocks9"|"store-mlp-blocks9"|"stored_mlp_blocks9"|"stored-mlp-blocks9"|"mlp_activation_blocks9"|"mlp-activation-blocks9")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 3-step, 2-sample stage-timed gate cut stored MLP activation blocks 12->9 and improved setup_wall_ms only to 0.975999x, but rejected default promotion because train_loop_wall_ms_per_step regressed to 1.084793x, steady-state CUDA-event step time to 1.082810x, train_tokens_per_second to 0.921834x, block backward to 1.073931x, and MLP projection backward to 1.155027x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_STORE_MLP_BLOCKS=12"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_STORE_MLP_BLOCKS=9"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-train_loop_wall_ms_per_step=1.001 train_loop_cuda_event_steady_state_wall_ms_per_step=1.002 stage.block_backward.total_ms=1.001 stage.block_backward.mlp_proj.total_ms=1.001}"
    MIN_CANDIDATE_RATIO_RAW="${MIN_CANDIDATE_RATIO_RAW:-train_tokens_per_second=0.999}"
    ;;
  "store_mlp_blocks6_tail"|"store-mlp-blocks6-tail"|"stored_mlp_blocks6_tail"|"stored-mlp-blocks6-tail"|"mlp_activation_blocks6_tail"|"mlp-activation-blocks6-tail")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="Unpromoted CUDA 13.3 diagnostic: compares six stored MLP activation blocks at the default head placement against six stored MLP activation blocks at tail placement. It must beat the same-script training gate before default promotion."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_STORE_MLP_BLOCKS=6 NFN_NATIVE_GPT_STORE_MLP_BLOCK_PLACEMENT=head"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_STORE_MLP_BLOCKS=6 NFN_NATIVE_GPT_STORE_MLP_BLOCK_PLACEMENT=tail"
    ;;
  "store_packed_attention_blocks6"|"store-packed-attention-blocks6"|"stored_packed_attention_blocks6"|"stored-packed-attention-blocks6"|"packed_attention_blocks6"|"packed-attention-blocks6")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 2-step, 2-sample training gate cut stored packed-attention blocks 12->6 and setup_wall_ms to 0.958626x, but rejected default promotion because train_loop_wall_ms_per_step regressed to 1.061075x, steady-state CUDA-event step time to 1.039648x, block backward to 1.032640x, MLP projection to 1.001225x, and attention dprep timing to 1.000231x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=12"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=6"
    AUTO_ATTENTION_SECTION_TIMING=1
    ;;
  "store_packed_attention_blocks6_tail"|"store-packed-attention-blocks6-tail"|"stored_packed_attention_blocks6_tail"|"stored-packed-attention-blocks6-tail"|"packed_attention_blocks6_tail"|"packed-attention-blocks6-tail")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="Unpromoted CUDA 13.3 diagnostic: compares six stored packed-attention blocks at the default head placement against six stored packed-attention blocks at tail placement. It must beat the same-script training gate before default promotion."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=6 NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCK_PLACEMENT=head"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=6 NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCK_PLACEMENT=tail"
    AUTO_ATTENTION_SECTION_TIMING=1
    ;;
  "store_residual1_off"|"store-residual1-off"|"stored_residual1_off"|"stored-residual1-off"|"residual1_activation_store_off"|"residual1-activation-store-off")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-25 1-step paired gate disabled stored residual1 activations, verified the saved-packed-attention recompute route keeps one attention-projection scratch buffer, and rejected default promotion because train_loop_wall_ms_per_step regressed to 1.059928x and tokens/sec to 0.943455x despite setup_wall_ms improving to 0.810095x; keep NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS=1 for the default path."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS=1"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS=0"
    ;;
  "full_activation_tape"|"full-activation-tape"|"no_recompute_activation_tape"|"no-recompute-activation-tape"|"no_recompute"|"no-recompute")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 paired diagnostic enabled a full forward activation tape and proved backward_recompute_blocks 11->0, but rejected default promotion because the larger tape is slower than scratch recompute on the workstation GPT shape; keep NFN_NATIVE_GPT_FULL_ACTIVATION_TAPE=0 for normal training."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_FULL_ACTIVATION_TAPE=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_FULL_ACTIVATION_TAPE=1"
    ;;
  "bgrad_first_write_direct"|"bgrad-first-write-direct"|"linear_bgrad_first_write_direct"|"linear-bgrad-first-write-direct")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-25 5-step, 3-sample rerun changed cuBLASLt bgrad route counters by moving 240 first-write bias gradients to direct writes, but regressed train_loop_wall_ms_per_step to 1.005404x, steady-state CUDA-event timing to 1.001035x, train_tokens_per_second to 0.994864x, block backward to 1.010323x, and MLP FC dWeight+bias to 1.020533x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1"
    ;;
  "bgrad_first_write_direct_qkv_65536"|"bgrad-first-write-direct-qkv-65536"|"qkv_bgrad_first_write_direct"|"qkv-bgrad-first-write-direct")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-25 3-step, 2-sample stage-timed gate moved 36 QKV first-write BGRADB calls to direct bias writes, but regressed train_loop_wall_ms_per_step to 1.007904x, steady-state CUDA-event timing to 1.001372x, and block backward to 1.014895x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_BGRAD_FIRST_WRITE_DIRECT_ENABLE_SHAPE=768,2304,65536,N,T"
    ;;
  "bgrad_first_write_direct_attn_proj_65536"|"bgrad-first-write-direct-attn-proj-65536"|"attn_proj_bgrad_first_write_direct"|"attn-proj-bgrad-first-write-direct")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-25 3-step, 2-sample stage-timed gate moved 36 attention-projection first-write BGRADB calls to direct bias writes, but regressed train_loop_wall_ms_per_step to 1.017045x, LM-head backward to 1.000402x, and block backward to 1.033149x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_BGRAD_FIRST_WRITE_DIRECT_ENABLE_SHAPE=768,768,65536,N,T"
    ;;
  "bgrad_first_write_direct_mlp_fc_65536"|"bgrad-first-write-direct-mlp-fc-65536"|"mlp_fc_bgrad_first_write_direct"|"mlp-fc-bgrad-first-write-direct")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-25 3-step, 2-sample stage-timed gate moved 36 MLP-FC first-write BGRADB calls to direct bias writes, but regressed train_loop_wall_ms_per_step to 1.012986x, LM-head backward to 1.000314x, block backward to 1.025149x, and MLP projection total to 1.059115x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_BGRAD_FIRST_WRITE_DIRECT_ENABLE_SHAPE=768,3072,65536,N,T"
    ;;
  "bgrad_first_write_direct_mlp_proj_65536"|"bgrad-first-write-direct-mlp-proj-65536"|"mlp_proj_bgrad_first_write_direct"|"mlp-proj-bgrad-first-write-direct")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-25 3-step, 2-sample stage-timed gate moved 36 MLP projection first-write BGRADB calls to direct bias writes, but regressed train_loop_wall_ms_per_step to 1.009189x, block backward to 1.014958x, MLP projection total to 1.026308x, and MLP projection dWeight+bias to 1.054470x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_BGRAD_FIRST_WRITE_DIRECT_ENABLE_SHAPE=3072,768,65536,N,T"
    ;;
  "qkv_concurrent_dinput_dweight"|"qkv-concurrent-dinput-dweight")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 same-script gate activated this route but regressed train_loop_wall_ms_per_step to 1.005526x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_BLOCK_QKV_CONCURRENT_DINPUT_DWEIGHT=1"
    ;;
  "mlp_proj_concurrent_dinput_dweight"|"mlp-proj-concurrent-dinput-dweight")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-25 3-step, 2-sample stage-timed gate proved the route enabled and moved block_backward_mlp_proj_concurrent_dinput_dweight_count from 0 to 288, but rejected default promotion because train_loop_wall_ms_per_step regressed to 1.004101x, steady-state CUDA-event timing to 1.004144x, stage.lm_head_backward.total_ms to 1.000889x, stage.block_backward.total_ms to 1.009823x, and stage.block_backward.mlp_proj.total_ms to 1.025216x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_BLOCK_MLP_PROJ_CONCURRENT_DINPUT_DWEIGHT=1"
    ;;
  "mlp_fc_concurrent_dinput_dweight"|"mlp-fc-concurrent-dinput-dweight")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-26 3-step, 1-sample stage-timed recheck proved the route changed by enabling block_backward_mlp_fc_concurrent_dinput_dweight and moving block_backward_mlp_fc_dinput_before_dweight_count from 288 to 0. It still remains rejected because stage.block_backward.mlp_fc.total_ms regressed to 1.025442x, stage.block_backward.total_ms to 1.005941x, stage.lm_head_backward.total_ms to 1.001738x, steady-state CUDA-event step time to 1.003685x, and candidate-over-llm.kittens train_loop_wall_ms_per_step stayed at 1.028729x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_BLOCK_MLP_FC_CONCURRENT_DINPUT_DWEIGHT=1"
    ;;
  "attn_proj_concurrent_dinput_dweight"|"attn-proj-concurrent-dinput-dweight")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 same-script gate activated this route but regressed train_loop_wall_ms_per_step to 1.002312x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_BLOCK_ATTN_PROJ_CONCURRENT_DINPUT_DWEIGHT=1"
    ;;
  "attn_proj_first_step_concurrent_dinput_dweight"|"attn-proj-first-step-concurrent-dinput-dweight"|"attn_proj_first_step_concurrent"|"attn-proj-first-step-concurrent")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 5-step, 3-sample stage-timed gate proved the first-step route counter moved 0->96, but rejected default promotion because train_loop_wall_ms_per_step regressed to 1.002629x, steady-state CUDA event timing to 1.001028x, stage.lm_head_backward.total_ms to 1.000503x, stage.block_backward.total_ms to 1.006184x, and stage.block_backward.attn_proj.total_ms to 1.075065x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_BLOCK_ATTN_PROJ_FIRST_STEP_CONCURRENT_DINPUT_DWEIGHT=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_BLOCK_ATTN_PROJ_FIRST_STEP_CONCURRENT_DINPUT_DWEIGHT=1"
    ;;
  "lm_head_concurrent_dhidden_dweight"|"lm-head-concurrent-dhidden-dweight")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 3-sample same-script confirmation activated the two-stream LM-head dHidden/dWeight schedule with cooperative LM-head backward disabled on the candidate, but regressed train_loop_wall_ms_per_step to 1.002970x and train tokens/sec to 0.997039x, so the cooperative CUDA Graph LM-head route remains the default."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=0 NFN_NATIVE_GPT_LM_HEAD_CONCURRENT_DHIDDEN_DWEIGHT=1"
    ;;
  "lm_head_dweight_before_dhidden"|"lm-head-dweight-before-dhidden")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 3-sample same-script confirmation activated the LM-head serial dWeight-before-dHidden schedule with cooperative LM-head backward disabled on the candidate, but regressed train_loop_wall_ms_per_step to 1.002871x and train tokens/sec to 0.997262x, so the cooperative CUDA Graph LM-head route remains the default."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=0 NFN_NATIVE_GPT_LM_HEAD_DWEIGHT_BEFORE_DHIDDEN=1"
    ;;
  "lm_head_pipeline_chunks"|"lm-head-pipeline-chunks")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 short 3-step, 2-sample stage-timed rerun activated the double-buffered LM-head pipeline schedule with cooperative LM-head backward disabled on the candidate, but the candidate command timed out after 300 seconds, so this side-stream pipeline remains rejected until the cross-stream ownership model is redesigned or replaced by a true fused/cooperative LM-head kernel."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=0 NFN_NATIVE_GPT_LM_HEAD_PIPELINE_CHUNKS=1"
    ;;
  "lm_head_overlap_last_dweight"|"lm-head-overlap-last-dweight"|"lm_head_last_dweight_overlap"|"lm-head-last-dweight-overlap")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-25 route-enabled recheck disabled cooperative LM-head only on the candidate so the last-dWeight overlap side stream actually enabled. It proved lm_head_overlap_last_dweight_enabled=true with 24 queue/sync events, but regressed train_loop_wall_ms_per_step to 1.020764x, steady-state CUDA-event step time to 1.002042x, train tokens/sec to 0.979861x, and stage.lm_head_backward.total_ms to 1.050532x versus the default cooperative CUDA Graph wrapper."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=0 NFN_NATIVE_GPT_LM_HEAD_OVERLAP_LAST_DWEIGHT=1"
    ;;
  "lm_head_row_chunk_65536"|"lm-head-row-chunk-65536"|"lm_head_full_row_chunk"|"lm-head-full-row-chunk")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-25 one-step same-script current/native/reference gate allowed the unsafe 65536-row LM-head chunk and changed lm_head_classifier_last_rows from 32768 to 65536, but rejected it because train_loop_wall_ms_per_step regressed to 7.368793x versus current native and 8.037520x versus llm.kittens, train_tokens_per_second fell to 0.135708x versus current and 0.124417x versus reference, stage.lm_head_backward.total_ms regressed to 1.022847x, and downstream stage.block_backward.attn_sdpa.to_qkv.total_ms collapsed to 63.207371x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_ALLOW_UNSAFE_LM_HEAD_ROW_CHUNK=1"
    CANDIDATE_EXTRA_ARGS_RAW="${CANDIDATE_EXTRA_ARGS_RAW:+$CANDIDATE_EXTRA_ARGS_RAW }--lm-head-row-chunk-size 65536"
    ;;
  "lm_head_row_chunk_49152"|"lm-head-row-chunk-49152")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 confirmation compared the 49152-row LM-head route against the restored 32768-row default. It changed lm_head_classifier_last_rows from 32768 to 49152, but regressed train_loop_wall_ms_per_step to 1.012983x and stage.block_backward.total_ms to 1.025087x; use 49152 only for historical diagnostics."
    BASELINE_EXTRA_ARGS_RAW="${BASELINE_EXTRA_ARGS_RAW:+$BASELINE_EXTRA_ARGS_RAW }--lm-head-row-chunk-size 32768"
    CANDIDATE_EXTRA_ARGS_RAW="${CANDIDATE_EXTRA_ARGS_RAW:+$CANDIDATE_EXTRA_ARGS_RAW }--lm-head-row-chunk-size 49152"
    ;;
  "lm_head_row_chunk_8192"|"lm-head-row-chunk-8192"|"lm_head_low_memory_row_chunk_8192"|"lm-head-low-memory-row-chunk-8192")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-24 confirmation compared the lower-memory 8192-row LM-head route against the 32768-row default. Startup-only setup wall time improved to 0.847026x and native_chunk_bf16_logit_bytes dropped to 0.25x, but the real 3-step training gate regressed train_loop_wall_ms_per_step to 1.000927x, steady-state CUDA-event step time to 1.000640x, and stage.lm_head_backward.total_ms to 1.028710x because the logits chunk count rose from 2 to 8."
    BASELINE_EXTRA_ARGS_RAW="${BASELINE_EXTRA_ARGS_RAW:+$BASELINE_EXTRA_ARGS_RAW }--lm-head-row-chunk-size 32768"
    CANDIDATE_EXTRA_ARGS_RAW="${CANDIDATE_EXTRA_ARGS_RAW:+$CANDIDATE_EXTRA_ARGS_RAW }--lm-head-row-chunk-size 8192"
    ;;
  "lm_head_row_chunk_32768"|"lm-head-row-chunk-32768"|"lm_head_old_row_chunk_32768"|"lm-head-old-row-chunk-32768")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="The Tile-CUDA default is 32768 LM-head rows again after the CUDA 13.3 dedicated RTX 5090 confirmation rejected the 49152-row route; this profile is kept only to reproduce the historical 49152-vs-32768 pair."
    BASELINE_EXTRA_ARGS_RAW="${BASELINE_EXTRA_ARGS_RAW:+$BASELINE_EXTRA_ARGS_RAW }--lm-head-row-chunk-size 49152"
    CANDIDATE_EXTRA_ARGS_RAW="${CANDIDATE_EXTRA_ARGS_RAW:+$CANDIDATE_EXTRA_ARGS_RAW }--lm-head-row-chunk-size 32768"
    ;;
  "lm_head_full_resident_reuse"|"lm-head-full-resident-reuse"|"lm_head_full_batch_reuse"|"lm-head-full-batch-reuse")
    TIMEOUT_PRONE_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_ALLOW_UNSAFE_LM_HEAD_ROW_CHUNK=1 NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS=1 NFN_NATIVE_GPT_FULL_BATCH_LM_HEAD_REUSE=1"
    CANDIDATE_EXTRA_ARGS_RAW="${CANDIDATE_EXTRA_ARGS_RAW:+$CANDIDATE_EXTRA_ARGS_RAW }--lm-head-row-chunk-size 65536"
    ;;
  "lm_head_cooperative_backward_required"|"lm-head-cooperative-backward-required"|"require_cooperative_lm_head_backward"|"require-cooperative-lm-head-backward")
    STRICT_PROBE_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    AUTO_DISABLE_METRIC_RATIO_GATES=1
    FORCE_DISABLE_ROUTE_CHANGE=1
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1"
    CANDIDATE_EXTRA_ARGS_RAW="${CANDIDATE_EXTRA_ARGS_RAW:+$CANDIDATE_EXTRA_ARGS_RAW }--require-cooperative-lm-head-backward"
    ;;
  "lm_head_cooperative_loss_bins"|"lm-head-cooperative-loss-bins")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 3-step, 2-sample same-script gate requested cooperative LM-head loss bins but did not change any tracked route counters, strategy values, linear shape stats, or cuBLASLt plan cache entries; the measured timing delta is noise until a real fused/cooperative kernel is integrated."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1 NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=1 NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_LOSS_BINS=1"
    COMMON_EXTRA_ARGS_RAW="${COMMON_EXTRA_ARGS_RAW:+$COMMON_EXTRA_ARGS_RAW }--train-loss-every-steps 1"
    ;;
  "lm_head_cooperative_backward"|"lm-head-cooperative-backward")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-24 same-window triad gate activated the cooperative LM-head CUDA Graph path but regressed current native train_loop_wall_ms_per_step to 1.015026x, steady-state CUDA-event timing to 1.005839x, train_tokens_per_second to 0.985194x, and candidate-over-llm.kittens wall to 1.018312x; keep it rejected until a true fused/cooperative kernel body replaces the diagnostic graph replay route."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1"
    ;;
  "lm_head_graph_prewarm"|"lm-head-graph-prewarm"|"lm_head_cooperative_graph_prewarm"|"lm-head-cooperative-graph-prewarm")
    ACCEPTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    ACCEPTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-26 post-MLP-FC-rollback rerun eliminated runtime LM-head graph capture and passed same-script gates at 0.985915x train_loop_wall_ms_per_step, 0.999199x steady-state CUDA-event timing, 0.957549x stage.lm_head_backward.total_ms, 0.997858x stage.block_backward.total_ms, and 0.992403x stage.block_backward.mlp_proj.total_ms. Route proof moved graph capture attempts 3->0 and graph cache hits 45->48. The native trainer defaults graph prewarm on; keep this profile as the explicit default-on versus opt-out regression check."
    DEFAULT_VS_LEGACY_PROFILE=1
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_GRAPH_PREWARM=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_GRAPH_PREWARM=1"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-train_loop_wall_ms_per_step=1.000 train_loop_cuda_event_steady_state_wall_ms_per_step=1.002 stage.lm_head_backward.total_ms=1.000 stage.block_backward.total_ms=1.000 stage.block_backward.mlp_proj.total_ms=1.000}"
    ;;
  "lm_head_graph_prewarm_dedup"|"lm-head-graph-prewarm-dedup"|"lm_head_graph_dedup"|"lm-head-graph-dedup")
    CANDIDATE_NOTE="Compares the legacy LM-head graph-prewarm loop against the default pointer-aware dedup key path. Equal-sized row chunks with different buffers are intentionally distinct keys, so this profile checks deterministic prewarm work rather than setup timing or route-change gates."
    DEFAULT_VS_LEGACY_PROFILE=1
    FORCE_DISABLE_ROUTE_CHANGE=1
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_GRAPH_PREWARM=1 NFN_NATIVE_GPT_LM_HEAD_GRAPH_PREWARM_DEDUP=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_GRAPH_PREWARM=1 NFN_NATIVE_GPT_LM_HEAD_GRAPH_PREWARM_DEDUP=1"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-lm_head_fused_graph_prewarm_success_count=1.000}"
    ;;
  "lm_head_graph_thread_cache_prewarm"|"lm-head-graph-thread-cache-prewarm"|"lm_head_graph_prime_thread_cache"|"lm-head-graph-prime-thread-cache")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-25 3-step, 1-sample stage-timed same-script gate proved the route by moving lm_head_fused_graph_thread_cache_hit_count from 45 to 48, but rejected default promotion because train_loop_wall_ms_per_step regressed to 1.003958x, steady-state CUDA-event timing to 1.002099x, stage.lm_head_backward.total_ms to 1.000922x, and train_tokens_per_second to 0.996059x."
    CANDIDATE_NOTE="Compares the default LM-head graph prewarm global-cache-only path against opt-in thread-local replay-cache priming. The candidate keeps graph prewarm enabled and sets NFN_NATIVE_GPT_LM_HEAD_GRAPH_PREWARM_THREAD_CACHE=1; the baseline leaves only the thread-local replay-cache priming disabled with NFN_NATIVE_GPT_LM_HEAD_GRAPH_PREWARM_THREAD_CACHE=0."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_GRAPH_PREWARM=1 NFN_NATIVE_GPT_LM_HEAD_GRAPH_PREWARM_THREAD_CACHE=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_GRAPH_PREWARM=1 NFN_NATIVE_GPT_LM_HEAD_GRAPH_PREWARM_THREAD_CACHE=1"
    MIN_CANDIDATE_RATIO_RAW="${MIN_CANDIDATE_RATIO_RAW:-lm_head_fused_graph_thread_cache_hit_count=1.000 train_tokens_per_second=1.000}"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-train_loop_wall_ms_per_step=1.000 train_loop_cuda_event_steady_state_wall_ms_per_step=1.002 stage.lm_head_backward.total_ms=1.000}"
    ;;
  "lm_head_graph_upload_off"|"lm-head-graph-upload-off"|"lm_head_cooperative_graph_upload_off"|"lm-head-cooperative-graph-upload-off")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-25 3-step, 2-sample stage-timed gate proved the route by moving lm_head_fused_graph_upload_success_count from 3 to 0, but rejected the opt-out path because train_loop_wall_ms_per_step regressed to 1.001492x, steady-state CUDA-event timing to 1.000055x, stage.lm_head_backward.total_ms to 1.000583x, and stage.lm_head_backward.cooperative.total_ms to 1.000593x."
    CANDIDATE_NOTE="Compares the default LM-head CUDA Graph executable upload against the opt-out route. The default calls cudaGraphUpload after graph instantiate during prewarm; this candidate sets NFN_NATIVE_GPT_LM_HEAD_GRAPH_UPLOAD=0 and must not beat default on train-loop or hot-stage timing before the upload can remain justified."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_GRAPH_PREWARM=1 NFN_NATIVE_GPT_LM_HEAD_GRAPH_UPLOAD=1"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_GRAPH_PREWARM=1 NFN_NATIVE_GPT_LM_HEAD_GRAPH_UPLOAD=0"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-train_loop_wall_ms_per_step=1.000 train_loop_cuda_event_steady_state_wall_ms_per_step=1.002 stage.lm_head_backward.total_ms=1.000 stage.lm_head_backward.cooperative.total_ms=1.000}"
    ;;
  "lm_head_graph_serial_body"|"lm-head-graph-serial-body"|"lm_head_cooperative_graph_serial"|"lm-head-cooperative-graph-serial")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-25 3-step, 2-sample stage-timed gate changed the strict LM-head ABI path class to diagnostic-cuda-graph-wrapper-serial-body, but regressed train_loop_wall_ms_per_step to 1.005992x, steady-state CUDA-event timing to 1.004767x, stage.lm_head_backward.total_ms to 1.022264x, and stage.lm_head_backward.cooperative.total_ms to 1.031743x versus the default side-stream graph body."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1 NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_CUDA_GRAPH=1 NFN_NATIVE_GPT_LM_HEAD_GRAPH_BODY_SERIAL=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1 NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_CUDA_GRAPH=1 NFN_NATIVE_GPT_LM_HEAD_GRAPH_BODY_SERIAL=1"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-train_loop_wall_ms_per_step=1.000 train_loop_cuda_event_steady_state_wall_ms_per_step=1.002 stage.lm_head_backward.total_ms=1.000}"
    ;;
  "lm_head_true_fused_cooperative"|"lm-head-true-fused-cooperative"|"lm_head_true_fused"|"lm-head-true-fused")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 post-toolkit-reinstall 1-step stage-timed rerun proved the route by moving lm_head_classifier_true_fused_launch_count from 0 to 16 and the path class to strict-true-fused-tile-kernel, but rejected the current 32x32 tiled diagnostic body because train_loop_wall_ms_per_step regressed to 5.991992x, train_tokens_per_second dropped to 0.166890x, stage.lm_head_backward.total_ms regressed to 22.660619x, and stage.lm_head_backward.cooperative.total_ms regressed to 32.243288x versus the default CUDA Graph wrapper. Keep this profile rejected until the strict body beats full-loop/reference gates."
    CANDIDATE_NOTE="Runs the trainer-sized full GPT loop with the opt-in strict true-fused cooperative LM-head body instead of the diagnostic CUDA Graph wrapper. The current strict body is a tiled diagnostic kernel, not a production Tile classifier-backward path; this profile must remain rejected until it passes the promotion gate."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE=0 NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE=1 NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION=1 NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1"
    REQUIRE_NATIVE_LM_HEAD_TRUE_FUSED=1
    LM_HEAD_BACKWARD_PREFLIGHT_PROFILE="trainer-chunk-true-fused"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-train_loop_wall_ms_per_step=1.000 train_loop_cuda_event_steady_state_wall_ms_per_step=1.002 stage.lm_head_backward.total_ms=1.000}"
    ;;
  "lm_head_true_fused_tile16"|"lm-head-true-fused-tile16"|"lm_head_true_fused_cooperative_tile16"|"lm-head-true-fused-cooperative-tile16")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="Unpromoted strict LM-head true-fused diagnostic: builds the candidate Tile ops library with NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=16 and forces CE threads to 256 before running the full GPT loop. Keep rejected until it beats the current CUDA Graph wrapper and llm.kittens reference gates."
    CANDIDATE_NOTE="Runs the trainer-sized full GPT loop with the opt-in strict true-fused cooperative LM-head body compiled as a 16x16 tiled matmul body instead of the default 32x32 diagnostic body."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE=0 NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE=1 NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION=1 NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1 NFN_NATIVE_GPT_CE_BF16_THREADS=256"
    CANDIDATE_TILE_OPS_BUILD_FLAGS="${CANDIDATE_TILE_OPS_BUILD_FLAGS:+$CANDIDATE_TILE_OPS_BUILD_FLAGS }-DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=16"
    REQUIRE_NATIVE_LM_HEAD_TRUE_FUSED=1
    LM_HEAD_BACKWARD_PREFLIGHT_PROFILE="trainer-chunk-true-fused-tile16"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-train_loop_wall_ms_per_step=1.000 train_loop_cuda_event_steady_state_wall_ms_per_step=1.002 stage.lm_head_backward.total_ms=1.000}"
    ;;
  "lm_head_true_fused_tile8"|"lm-head-true-fused-tile8"|"lm_head_true_fused_cooperative_tile8"|"lm-head-true-fused-cooperative-tile8")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="Unpromoted strict LM-head true-fused diagnostic: builds the candidate Tile ops library with NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=8 and forces CE threads to 64 before running the full GPT loop. Keep rejected until it beats the current CUDA Graph wrapper and llm.kittens reference gates."
    CANDIDATE_NOTE="Runs the trainer-sized full GPT loop with the opt-in strict true-fused cooperative LM-head body compiled as an 8x8 tiled matmul body instead of the default 32x32 diagnostic body."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE=0 NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE=1 NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION=1 NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1 NFN_NATIVE_GPT_CE_BF16_THREADS=64"
    CANDIDATE_TILE_OPS_BUILD_FLAGS="${CANDIDATE_TILE_OPS_BUILD_FLAGS:+$CANDIDATE_TILE_OPS_BUILD_FLAGS }-DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=8"
    REQUIRE_NATIVE_LM_HEAD_TRUE_FUSED=1
    LM_HEAD_BACKWARD_PREFLIGHT_PROFILE="trainer-chunk-true-fused-tile8"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-train_loop_wall_ms_per_step=1.000 train_loop_cuda_event_steady_state_wall_ms_per_step=1.002 stage.lm_head_backward.total_ms=1.000}"
    ;;
  "lm_head_true_fused_tile4"|"lm-head-true-fused-tile4"|"lm_head_true_fused_cooperative_tile4"|"lm-head-true-fused-cooperative-tile4")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 1-step stage-timed gate built the candidate Tile ops library with NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=4, forced CE threads to 16, and proved the route by moving lm_head_classifier_true_fused_launch_count from 0 to 16 with path class strict-true-fused-tile-kernel. It remains rejected because train_loop_wall_ms_per_step regressed to 30.645660x, train_tokens_per_second dropped to 0.032631x, stage.lm_head_backward.total_ms regressed to 129.582841x, and stage.lm_head_backward.cooperative.total_ms regressed to 186.457823x versus the CUDA Graph wrapper."
    CANDIDATE_NOTE="Runs the trainer-sized full GPT loop with the opt-in strict true-fused cooperative LM-head body compiled as a 4x4 tiled matmul body instead of the default 32x32 diagnostic body."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE=0 NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE=1 NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION=1 NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1 NFN_NATIVE_GPT_CE_BF16_THREADS=16"
    CANDIDATE_TILE_OPS_BUILD_FLAGS="${CANDIDATE_TILE_OPS_BUILD_FLAGS:+$CANDIDATE_TILE_OPS_BUILD_FLAGS }-DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=4"
    REQUIRE_NATIVE_LM_HEAD_TRUE_FUSED=1
    LM_HEAD_BACKWARD_PREFLIGHT_PROFILE="trainer-chunk-true-fused-tile4"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-train_loop_wall_ms_per_step=1.000 train_loop_cuda_event_steady_state_wall_ms_per_step=1.002 stage.lm_head_backward.total_ms=1.000}"
    ;;
  "lm_head_cooperative_sequence_wrapper"|"lm-head-cooperative-sequence-wrapper"|"lm_head_cooperative_cuda_graph_off"|"lm-head-cooperative-cuda-graph-off")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-24 5-step, 3-sample confirmation forced the cooperative LM-head sequence wrapper instead of cached CUDA Graph replay and proved route changes, but rejected default promotion because train_loop_wall_ms_per_step regressed to 1.003739x, steady-state CUDA-event timing to 1.001489x, and stage.lm_head_backward.total_ms to 1.000401x. A shorter 3-step, 2-sample probe passed, so keep this profile available only for intentional graph-vs-sequence diagnostics."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1 NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_CUDA_GRAPH=1"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1 NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_CUDA_GRAPH=0 NFN_NATIVE_GPT_LM_HEAD_FORCE_SEQUENCE_WRAPPER_DIAGNOSTIC=1"
    ;;
  "lm_head_cooperative_backward_off"|"lm-head-cooperative-backward-off"|"lm_head_direct_classifier_matmul"|"lm-head-direct-classifier-matmul")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 2026-06-25 10-step llm.kittens parity gate disabled the default cooperative LM-head CUDA Graph wrapper and ran the direct CE+dHidden+dWeight schedule, but regressed NeuralFn versus llm.kittens to 1.019533x train-loop wall and 1.016084x steady-state CUDA-event timing; keep this route diagnostic-only until a real fused classifier-backward kernel replaces the wrapper."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=0"
    ;;
  "lm_head_cooperative_no_loss_backward"|"lm-head-cooperative-no-loss-backward"|"lm_head_cooperative_backward_no_loss"|"lm-head-cooperative-backward-no-loss")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 dedicated RTX 5090 1-step stage-timed gate activated the cooperative LM-head sequence wrapper with no-loss CE, but regressed train_loop_wall_ms_per_step to 1.117578x, stage.lm_head_backward.total_ms to 1.294010x, and train_tokens_per_second to 0.894788x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1 NFN_NATIVE_GPT_LM_HEAD_CLASSIFIER_CE_NO_LOSS=1 NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_DEFAULT_SPECIALIZED=1"
    COMMON_EXTRA_ARGS_RAW="${COMMON_EXTRA_ARGS_RAW:+$COMMON_EXTRA_ARGS_RAW }--train-loss-every-steps 0"
    ;;
  "token_weight_vector4_strided"|"token-weight-vector4-strided")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 3-sample startup-only rerun requested the strided vector4 token-weight initializer, but rejected it because it did not change any tracked route, strategy, or plan and regressed setup_wall_ms to 1.012563x, setup.float_arena_materialize.total_ms to 1.022720x, setup.uint16_arena_materialize.total_ms to 1.017532x, and setup.token_weight_init.total_ms to 1.006804x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT=1"
    ;;
  "token_weight_threaded"|"token-weight-threaded")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 3-sample startup-only rerun changed token_weight_init_strategy to device-threaded-power2-deterministic-fused-bf16-shadow, but rejected it because setup_wall_ms regressed to 1.006278x, setup.uint16_arena_materialize.total_ms to 1.010660x, and setup.token_weight_init.total_ms to 1.025016x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1"
    ;;
  "token_weight_bf16_pattern"|"token-weight-bf16-pattern")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 3-sample startup-only rerun changed token_weight_init_strategy to device-vector4-power2-deterministic-fused-bf16-pattern-shadow, but rejected it because setup_wall_ms regressed to 1.005840x and setup.token_weight_init.total_ms to 1.015463x; setup.uint16_arena_materialize.total_ms stayed noise-flat at 0.998215x."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_PATTERN_INIT=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_PATTERN_INIT=1"
    ;;
  "token_weight_padded_init"|"token-weight-padded-init"|"token_weight_padded_zero"|"token-weight-padded-zero")
    ACCEPTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    ACCEPTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 startup-only 3-sample rerun kept the conversion-based fused padded token-weight initializer as the default after measuring setup_wall_ms at 0.988862x and setup.token_weight_init.total_ms at 0.976989x versus the older separate padding-zero/default vector4 path. The full 10-step reference run still failed llm.kittens throughput gates, so this profile remains a startup default-vs-legacy proof rather than a throughput parity claim."
    DEFAULT_VS_LEGACY_PROFILE=1
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_PADDED_INIT=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_PADDED_INIT=1"
    MAX_CANDIDATE_RATIO_RAW="${MAX_CANDIDATE_RATIO_RAW:-setup_wall_ms=1.000 setup.token_weight_init.total_ms=1.000}"
    ;;
  "token_weight_fast_int32"|"token-weight-fast-int32"|"token_weight_no_vector4"|"token-weight-no-vector4")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 3-sample startup-only rerun disabled the vector4 token-weight initializer and improved setup.token_weight_init.total_ms to 0.960426x, but rejected it because setup_wall_ms regressed to 1.014148x, setup.float_arena_materialize.total_ms to 1.012221x, and setup.uint16_arena_materialize.total_ms to 1.022183x versus the vector4 default."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT=1"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT=0"
    ;;
  "token_weight_two_pass_bf16"|"token-weight-two-pass-bf16"|"token_weight_no_fused_bf16"|"token-weight-no-fused-bf16")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3.33 dedicated RTX 5090 2026-06-27 3-sample startup-only rerun disabled the fused BF16-shadow token-weight initializer, but rejected it because setup_wall_ms regressed to 1.005287x, setup.float_arena_materialize.total_ms to 1.021698x, setup.uint16_arena_materialize.total_ms to 1.004536x, and setup.token_weight_init.total_ms to 1.005229x versus the fused BF16-shadow vector4 default."
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_INIT=1"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_INIT=0"
    ;;
  *)
    echo "Unknown NFN_SM120_NATIVE_CANDIDATE_PROFILE: $CANDIDATE_PROFILE" >&2
    echo "Known profiles: linked_startup, fast_startup, fast_startup_full, lm_head_tk_dinput_32768, lm_head_cublaslt_dhidden_32768, lm_head_dhidden_fast16bf_32768, lm_head_cooperative_cublaslt, lm_head_tk_dweight_32768, lm_head_tk_dweight_49152, lm_head_prepack_bf16_hidden_off, lm_head_prepack_bf16_hidden_on, lm_head_bf16_hidden_from_final_norm, lm_head_public_vocab_strided_gemm, mlp_proj_tk_dweight_65536, block_split_bgrad_65536, mlp_proj_split_bgrad_65536, layernorm_affine_row_chunk_128, layernorm_affine_row_chunk_64, layernorm_affine_row_chunk_96, layernorm_affine_row_chunk_512, linear_bias_row_chunk_256, linear_bias_row_chunk_1024, linear_bias_threads_512, lm_head_logits_bf16_fallback_32768, lm_head_logits_bf16_fallback_49152, qkv_forward_bf16_fallback_65536, tk_qkv_forward_prewarm, tk_qkv_forward_prewarm_1row, tk_qkv_forward_prewarm_32768, mlp_fc_forward_bf16_fallback_65536, fused_ln2_bf16_out_off, mlp_residual_next_ln1_off, bf16_persistent_block_outputs_direct_ln1, bf16_persistent_block_outputs6, bf16_persistent_block_outputs_last6, ce_bf16_threads_512, lm_head_ce_exp2, lm_head_ce_vec8_io, lm_head_ce_vec8_normal_store, lm_head_ce_scalar_streaming_store, lm_head_ce_natural_rows, lm_head_ce_default_specialized, lm_head_ce_no_loss_default_specialized, lm_head_ce_no_loss_vec8_normal_store_specialized, lm_head_prob_only_combined_corrections, lm_head_prob_only_ce_target_corrections, lm_head_prob_only_combined_corrections_threads_512, lm_head_ce_no_loss_llmk_style_specialized, lm_head_ce_llmk_style_specialized, lm_head_ce_loss_bins_llmk_style_specialized, lm_head_ce_loss_bins_default_specialized, lm_head_loss_bins, lm_head_loss_bins_bf16_workspace_prewarm, lm_head_row_loss_sum_accumulate, lm_head_row_loss_partial_reduce, cublaslt_min_waves, cublaslt_max_waves, cublaslt_heavy_shape_flip, cublaslt_plan_prewarm_block_only, cublaslt_plan_prewarm_lm_head_only, cublaslt_plan_prewarm_off, cublaslt_block_dinput, cublaslt_block_dinput_h3_65536, cublaslt_qkv_dweight_h0_65536, cublaslt_attn_proj_dweight_h0_65536, cublaslt_grouped_probe, cublaslt_grouped_probe_required, tk_dgelu_dinput, tk_dgelu_approx_tanh, mlp_proj_dgelu_fallback, attention_atomic_dq, attention_bwd_block_32, attention_bwd_block_64, bf16_attention_grad_out, bf16_attention_dprep_grad_out, packed_attention_bwd_batch_32, packed_attention_bwd_batch_96, packed_attention_bwd_batch_128, attention_dprep_float_hd64_specialized, attention_dprep_grid3d, attention_dprep_warps_2, attention_dprep_warps_4, packed_attention_saved_lse_off, mlp_proj_dinput_before_dweight, mlp_fc_dinput_before_dweight, attn_proj_dinput_before_dweight, qkv_dinput_before_dweight, qkv_dinput_ln128, qkv_dinput_ln64, lm_head_fused_loss_backward_off, lm_head_classifier_ce_no_loss, cublas_handle_prewarm, cuda_module_eager, bf16_workspace_prewarm, tk_forward_no_n96, llmk_sm120_reference_flags, tk_sm120_super_m7, tk_sm120_super_m13, cuda_device_max_connections_1, combined_device_arena, cuda_malloc_async, concurrent_arena_materialize, uint16_arena_first, host_descriptor_reserve, store_mlp_blocks3, store_mlp_blocks6, store_mlp_blocks9, store_mlp_blocks6_tail, store_packed_attention_blocks6, store_packed_attention_blocks6_tail, store_residual1_off, full_activation_tape, bgrad_first_write_direct, bgrad_first_write_direct_qkv_65536, bgrad_first_write_direct_attn_proj_65536, bgrad_first_write_direct_mlp_fc_65536, bgrad_first_write_direct_mlp_proj_65536, qkv_concurrent_dinput_dweight, mlp_proj_concurrent_dinput_dweight, mlp_fc_concurrent_dinput_dweight, attn_proj_concurrent_dinput_dweight, attn_proj_first_step_concurrent_dinput_dweight, lm_head_concurrent_dhidden_dweight, lm_head_dweight_before_dhidden, lm_head_pipeline_chunks, lm_head_overlap_last_dweight, lm_head_row_chunk_8192, lm_head_row_chunk_32768, lm_head_row_chunk_49152, lm_head_row_chunk_65536, lm_head_full_resident_reuse, lm_head_cooperative_backward, lm_head_graph_prewarm, lm_head_graph_prewarm_dedup, lm_head_graph_thread_cache_prewarm, lm_head_graph_upload_off, lm_head_graph_serial_body, lm_head_true_fused_cooperative, lm_head_true_fused_tile16, lm_head_true_fused_tile8, lm_head_true_fused_tile4, lm_head_cooperative_sequence_wrapper, lm_head_cooperative_backward_off, lm_head_cooperative_no_loss_backward, lm_head_cooperative_backward_required, lm_head_cooperative_loss_bins, token_weight_vector4_strided, token_weight_threaded, token_weight_bf16_pattern, token_weight_padded_init, token_weight_fast_int32, token_weight_two_pass_bf16" >&2
    exit 2
    ;;
esac
if [[ -n "$REJECTED_CANDIDATE_PROFILE" ]]; then
  case "${DRY_RUN_PLAN,,}:${ALLOW_REJECTED_CANDIDATE_PROFILE,,}" in
    1:*|true:*|yes:*|on:*|*:1|*:true|*:yes|*:on)
      ;;
    *)
      echo "NFN_SM120_NATIVE_CANDIDATE_PROFILE=$REJECTED_CANDIDATE_PROFILE is a rejected SM120 candidate." >&2
      echo "$REJECTED_CANDIDATE_REASON" >&2
      echo "Set NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1 to rerun it intentionally, or use NFN_SM120_NATIVE_DRY_RUN_PLAN=1 to inspect expansion only." >&2
      exit 2
      ;;
  esac
  case "${ENFORCE_REJECTED_CANDIDATE_RATIO_GATES,,}" in
    "1"|"true"|"yes"|"on")
      ;;
    "0"|"false"|"no"|"off"|"")
      if [[ -z "${MAX_CANDIDATE_RATIO_RAW-}" && -z "${MIN_CANDIDATE_RATIO_RAW-}" ]]; then
        AUTO_DISABLE_METRIC_RATIO_GATES=1
      fi
      ;;
    *)
      echo "Unsupported NFN_SM120_NATIVE_ENFORCE_REJECTED_CANDIDATE_RATIO_GATES value: $ENFORCE_REJECTED_CANDIDATE_RATIO_GATES" >&2
      exit 2
      ;;
  esac
fi
if [[ -n "$CANDIDATE_TILE_OPS_BUILD_FLAGS" ]]; then
  if [[ -z "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB_EXPLICIT" ]]; then
    NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB="/tmp/nfn_sm120_candidate_tile_ops_${CANDIDATE_PROFILE:-custom}_$$.so"
    NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB_EXPLICIT="generated"
  fi
  if [[ "$IS_DRY_RUN_PLAN" != "1" ]]; then
    NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="$CANDIDATE_TILE_OPS_BUILD_FLAGS" \
      bash "$ROOT_DIR/tools/build_native_train_tile_ops.sh" "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB" >&2
  fi
fi
TEMPLATE_NAME="$(env_or_alias5 NFN_SM120_NATIVE_TEMPLATE_NAME NFN_SM120_NATIVE_CANDIDATE_TEMPLATE_NAME NFN_SM120_CANDIDATE_TEMPLATE_NAME NFN_SM120_PARITY_TEMPLATE_NAME NFN_SM120_TEMPLATE_NAME "")"
GRAPH_FILE="$(env_or_alias5 NFN_SM120_NATIVE_GRAPH_FILE NFN_SM120_NATIVE_CANDIDATE_GRAPH_FILE NFN_SM120_CANDIDATE_GRAPH_FILE NFN_SM120_PARITY_GRAPH_FILE NFN_SM120_GRAPH_FILE "")"
ALLOW_TIMEOUT_PRONE_LM_HEAD_PROFILE="$(env_or_alias NFN_SM120_NATIVE_ALLOW_TIMEOUT_PRONE_LM_HEAD_PROFILE NFN_SM120_ALLOW_TIMEOUT_PRONE_LM_HEAD_PROFILE 0)"
if [[ -n "$TIMEOUT_PRONE_CANDIDATE_PROFILE" ]]; then
  case "${DRY_RUN_PLAN,,}:${ALLOW_TIMEOUT_PRONE_LM_HEAD_PROFILE,,}" in
    1:*|true:*|yes:*|on:*|*:1|*:true|*:yes|*:on)
      ;;
    *)
      echo "NFN_SM120_NATIVE_CANDIDATE_PROFILE=$TIMEOUT_PRONE_CANDIDATE_PROFILE is timeout-prone on the 64x1024 SM120 GPT shape." >&2
      echo "Set NFN_SM120_NATIVE_ALLOW_TIMEOUT_PRONE_LM_HEAD_PROFILE=1 to run it intentionally, or use NFN_SM120_NATIVE_DRY_RUN_PLAN=1 to inspect expansion only." >&2
      exit 2
      ;;
  esac
fi
if [[ -n "$STRICT_PROBE_CANDIDATE_PROFILE" ]]; then
  case "${DRY_RUN_PLAN,,}" in
    "1"|"true"|"yes"|"on")
      ;;
    *)
      echo "NFN_SM120_NATIVE_CANDIDATE_PROFILE=$STRICT_PROBE_CANDIDATE_PROFILE is a strict ABI preflight probe, not a speed candidate." >&2
      if [[ "$STRICT_GROUPED_CUBLASLT_PROBE" == "1" ]]; then
        echo "It is expected to fail until cuBLASLt grouped layout and grouped matmul probe statuses are both 0." >&2
      else
        echo "It is expected to fail until the loaded Tile ops library reports a true fused cooperative LM-head backward capability." >&2
      fi
      ;;
  esac
fi
REQUIRE_NATIVE_ROUTE_CHANGE="$(env_or_alias NFN_SM120_NATIVE_REQUIRE_ROUTE_CHANGE NFN_SM120_CANDIDATE_REQUIRE_ROUTE_CHANGE auto)"
if [[ "$FORCE_DISABLE_ROUTE_CHANGE" == "1" ]]; then
  REQUIRE_NATIVE_ROUTE_CHANGE=0
fi
if [[ -z "$MAX_CANDIDATE_RATIO_RAW" ]]; then
  has_candidate_change=0
  if [[ "$NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN" != "$NFN_NATIVE_GPT_TRAIN_BIN" ||
        "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB" != "$NFN_NATIVE_TILE_OPS_LIB" ||
        -n "$CANDIDATE_TILE_OPS_BUILD_FLAGS" ||
        -n "$CANDIDATE_ENV_RAW" ||
        -n "$CANDIDATE_EXTRA_ARGS_RAW" ]]; then
    has_candidate_change=1
  fi
  if [[ "$has_candidate_change" == "1" ]]; then
    case "${DRY_RUN_PLAN,,}" in
      "1"|"true"|"yes"|"on")
        has_candidate_change=0
        ;;
    esac
  fi
  if [[ "$has_candidate_change" == "1" && "$AUTO_DISABLE_METRIC_RATIO_GATES" == "0" ]]; then
    case "${STARTUP_ONLY,,}" in
      "1"|"true"|"yes"|"on")
        MAX_CANDIDATE_RATIO_RAW="setup_wall_ms=1.000"
        startup_candidate_text="$NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN $NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB $CANDIDATE_ENV_RAW $CANDIDATE_EXTRA_ARGS_RAW"
        case "$startup_candidate_text" in
          *TOKEN_WEIGHT*|*token_weight*)
            MAX_CANDIDATE_RATIO_RAW+=" setup.token_weight_init.total_ms=1.000"
            ;;
        esac
        ;;
      *)
        MAX_CANDIDATE_RATIO_RAW="train_loop_wall_ms_per_step=1.000"
        case "${TRAIN_LOOP_EVENT_TIMING,,}" in
          "1"|"true"|"yes"|"on")
            if [[ "$STEPS" =~ ^[0-9]+$ && "$STEPS" -gt 1 ]]; then
              MAX_CANDIDATE_RATIO_RAW+=" train_loop_cuda_event_steady_state_wall_ms_per_step=1.000"
            fi
            ;;
        esac
        case "${STAGE_TIMING,,}" in
          "1"|"true"|"yes"|"on")
            candidate_gate_text="$NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN $NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB $CANDIDATE_ENV_RAW $CANDIDATE_EXTRA_ARGS_RAW"
            lm_head_only_candidate_gate=0
            case "$candidate_gate_text" in
              *LM_HEAD*|*lm_head*|*CE_BF16*|*ce_bf16*)
                lm_head_only_candidate_gate=1
                ;;
            esac
            case "$candidate_gate_text" in
              *BLOCK_*|*block_*|*MLP_*|*mlp_*|*QKV*|*qkv*|*ATTN*|*attn*|*LINEAR_BACKWARD*|*linear_backward*)
                lm_head_only_candidate_gate=0
                ;;
            esac
            case "$candidate_gate_text" in
              *LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE*|*linear_backward_bias_row_chunk_size*|*LINEAR_BACKWARD_BIAS_THREADS*|*linear_backward_bias_threads_per_block*)
                ;;
              *)
                MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.total_ms=1.000"
                ;;
            esac
            if [[ "$lm_head_only_candidate_gate" != "1" ]]; then
              MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.total_ms=1.000"
              MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.mlp_proj.total_ms=1.000"
            fi
            case "$candidate_gate_text" in
              *CE_BF16*|*ce_bf16*|*LM_HEAD_CE*|*lm_head_ce*)
                if [[ "$SKIP_LM_HEAD_CE_STAGE_GATE" != "1" ]]; then
                  MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.ce.total_ms=1.000"
                fi
                ;;
            esac
            case "$candidate_gate_text" in
              *LM_HEAD_FUSED_LOSS_BACKWARD*|*lm_head_fused_loss_backward*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.ce.total_ms=1.000"
                ;;
            esac
            case "$candidate_gate_text" in
              *LM_HEAD_PREPACK_BF16_HIDDEN*|*lm_head_prepack_bf16_hidden*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.dhidden.total_ms=1.000"
                MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.dweight.total_ms=1.000"
                MAX_CANDIDATE_RATIO_RAW+=" setup.uint16_arena_materialize.total_ms=1.000"
                ;;
            esac
            case "$candidate_gate_text" in
              *TK_DINPUT*|*tk_dinput*|*CUBLASLT_ENABLE_SHAPE*|*cublaslt_enable_shape*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.dhidden.total_ms=1.000"
                ;;
            esac
            case "$candidate_gate_text" in
              *PUBLIC_VOCAB_STRIDED_GEMM*|*public_vocab_strided_gemm*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.dhidden.total_ms=1.000"
                MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.dweight.total_ms=1.000"
                ;;
            esac
            case "$candidate_gate_text" in
              *MLP_PROJ_TK_DWEIGHT*|*mlp_proj_tk_dweight*|*MLP_PROJ_SPLIT_BGRAD*|*mlp_proj_split_bgrad*|*3072,768,65536,N,T*)
                ;;
              *TK_DWEIGHT*|*tk_dweight*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.dweight.total_ms=1.000"
                ;;
            esac
            case "$candidate_gate_text" in
              *MLP_PROJ_TK_DWEIGHT*|*mlp_proj_tk_dweight*|*MLP_PROJ_SPLIT_BGRAD*|*mlp_proj_split_bgrad*|*3072,768,65536,N,T*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.mlp_proj.dweight_bias.total_ms=1.000"
                ;;
            esac
            case "$candidate_gate_text" in
              *LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE*|*linear_backward_bias_row_chunk_size*|*LINEAR_BACKWARD_BIAS_THREADS*|*linear_backward_bias_threads_per_block*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.mlp_proj.dweight_bias.total_ms=1.000"
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.mlp_fc.dweight_bias.total_ms=1.000"
                ;;
            esac
            case "$candidate_gate_text $CANDIDATE_TILE_OPS_BUILD_FLAGS" in
              *DGELU*|*dgelu*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.mlp_proj.dinput.total_ms=1.000"
                ;;
            esac
            case "$candidate_gate_text $CANDIDATE_TILE_OPS_BUILD_FLAGS" in
              *ATOMIC_DQ*|*atomic_dq*|*attention_atomic_dq*|*attention-atomic-dq*)
                AUTO_ATTENTION_SECTION_TIMING=1
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.attn_sdpa.total_ms=1.000"
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.attn_sdpa.to_qkv.total_ms=1.000"
                MAX_CANDIDATE_RATIO_RAW+=" attention_backward_tk_timing_us=1.000"
                ;;
            esac
            case "$candidate_gate_text" in
              *PACKED_ATTENTION*|*packed_attention*|*BF16_ATTENTION*|*bf16_attention*)
                AUTO_ATTENTION_SECTION_TIMING=1
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.attn_sdpa.total_ms=1.000"
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.attn_sdpa.to_qkv.total_ms=1.000"
                MAX_CANDIDATE_RATIO_RAW+=" attention_backward_tk_timing_us=1.000"
                MAX_CANDIDATE_RATIO_RAW+=" attention_backward_dprep_timing_us=1.000"
                ;;
            esac
            case "$candidate_gate_text" in
              *MLP_PROJ_DINPUT_BEFORE_DWEIGHT*|*mlp_proj_dinput_before_dweight*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.mlp_proj.dinput.total_ms=1.000"
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.mlp_proj.dweight_bias.total_ms=1.000"
                ;;
            esac
            case "$candidate_gate_text" in
              *MLP_FC_DINPUT_BEFORE_DWEIGHT*|*mlp_fc_dinput_before_dweight*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.mlp_fc.total_ms=1.000"
                ;;
            esac
            case "$candidate_gate_text" in
              *ATTN_PROJ_DINPUT_BEFORE_DWEIGHT*|*attn_proj_dinput_before_dweight*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.attn_proj.total_ms=1.000"
                ;;
            esac
            case "$candidate_gate_text" in
              *QKV_DINPUT_BEFORE_DWEIGHT*|*qkv_dinput_before_dweight*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.qkv.total_ms=1.000"
                ;;
            esac
            case "$candidate_gate_text" in
              *BLOCK_QKV_CONCURRENT_DINPUT_DWEIGHT*|*block_qkv_concurrent_dinput_dweight*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.qkv.total_ms=1.000"
                ;;
            esac
            case "$candidate_gate_text" in
              *BLOCK_MLP_PROJ_CONCURRENT_DINPUT_DWEIGHT*|*block_mlp_proj_concurrent_dinput_dweight*|*mlp_proj_concurrent_dinput_dweight*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.mlp_proj.total_ms=1.000"
                ;;
            esac
            case "$candidate_gate_text" in
              *BLOCK_MLP_FC_CONCURRENT_DINPUT_DWEIGHT*|*block_mlp_fc_concurrent_dinput_dweight*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.mlp_fc.total_ms=1.000"
                ;;
            esac
            case "$candidate_gate_text" in
              *BLOCK_ATTN_PROJ_CONCURRENT_DINPUT_DWEIGHT*|*block_attn_proj_concurrent_dinput_dweight*|*attn_proj_concurrent_dinput_dweight*|*BLOCK_ATTN_PROJ_FIRST_STEP_CONCURRENT_DINPUT_DWEIGHT*|*block_attn_proj_first_step_concurrent_dinput_dweight*|*attn_proj_first_step_concurrent_dinput_dweight*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.attn_proj.total_ms=1.000"
                ;;
            esac
            ;;
        esac
        ;;
    esac
  fi
fi
if [[ "$PROMOTED_QKV_LN128_PROFILE" == "1" &&
      -z "$USER_MAX_CANDIDATE_RATIO_RAW" &&
      "$AUTO_DISABLE_METRIC_RATIO_GATES" == "0" &&
      "$STARTUP_ONLY" != "1" &&
      "$STARTUP_ONLY" != "true" &&
      "$STARTUP_ONLY" != "yes" &&
      "$STARTUP_ONLY" != "on" ]]; then
  MAX_CANDIDATE_RATIO_RAW="train_loop_wall_ms_per_step=1.000"
  case "${TRAIN_LOOP_EVENT_TIMING,,}" in
    "1"|"true"|"yes"|"on")
      if [[ "$STEPS" =~ ^[0-9]+$ && "$STEPS" -gt 1 ]]; then
        MAX_CANDIDATE_RATIO_RAW+=" train_loop_cuda_event_steady_state_wall_ms_per_step=1.002"
      fi
      ;;
  esac
  case "${STAGE_TIMING,,}" in
    "1"|"true"|"yes"|"on")
      MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.total_ms=1.000"
      ;;
  esac
fi
if [[ "$PROMOTED_QKV_LN128_PROFILE" == "1" &&
      -z "$USER_MIN_CANDIDATE_RATIO_RAW" &&
      "$AUTO_DISABLE_METRIC_RATIO_GATES" == "0" &&
      "$STARTUP_ONLY" != "1" &&
      "$STARTUP_ONLY" != "true" &&
      "$STARTUP_ONLY" != "yes" &&
      "$STARTUP_ONLY" != "on" ]]; then
  MIN_CANDIDATE_RATIO_RAW="train_tokens_per_second=1.000"
fi

filter_generated_candidate_ratio_gates() {
  local raw="$1"
  local item
  local metric
  local filtered=()
  local run_has_steady_state=0
  local run_has_stage_timing=0
  if [[ "$STEPS" =~ ^[0-9]+$ && "$STEPS" -gt 1 ]]; then
    run_has_steady_state=1
  fi
  case "${STAGE_TIMING,,}" in
    "1"|"true"|"yes"|"on")
      run_has_stage_timing=1
      ;;
  esac
  for item in $raw; do
    metric="${item%%=*}"
    metric="${metric#*:}"
    if [[ "$run_has_steady_state" != "1" &&
          "$metric" == "train_loop_cuda_event_steady_state_wall_ms_per_step" ]]; then
      continue
    fi
    if [[ "$run_has_stage_timing" != "1" && "$metric" == stage.* ]]; then
      continue
    fi
    filtered+=("$item")
  done
  printf '%s\n' "${filtered[*]}"
}

if [[ -z "$USER_MAX_CANDIDATE_RATIO_RAW" ]]; then
  MAX_CANDIDATE_RATIO_RAW="$(filter_generated_candidate_ratio_gates "$MAX_CANDIDATE_RATIO_RAW")"
fi
if [[ -z "$USER_MIN_CANDIDATE_RATIO_RAW" ]]; then
  MIN_CANDIDATE_RATIO_RAW="$(filter_generated_candidate_ratio_gates "$MIN_CANDIDATE_RATIO_RAW")"
fi

native_gpt_source_newer_than() {
  local target="$1"
  [[ "$ROOT_DIR/neuralfn/csrc/native_gpt2/nfn_gpt2_native_train.cpp" -nt "$target" ||
     "$ROOT_DIR/neuralfn/csrc/native_train/token_shards.cpp" -nt "$target" ]]
}

tile_ops_source_newer_than() {
  local target="$1"
  [[ "$ROOT_DIR/neuralfn/csrc/native_train/tile_ops.cu" -nt "$target" ||
     "$ROOT_DIR/neuralfn/csrc/native_train/tile_ops.h" -nt "$target" ||
     "$ROOT_DIR/neuralfn/csrc/tile_cuda/kernels.cu" -nt "$target" ||
     "$ROOT_DIR/tools/build_native_train_tile_ops.sh" -nt "$target" ]]
}

ensure_native_gpt_trainer_current() {
  local train_bin="$1"
  local explicit="$2"
  if [[ -n "$explicit" ]]; then
    return 0
  fi
  case "$(basename "$train_bin")" in
    nfn_gpt_native_train_linked)
      local rebuild_linked=0
      if [[ ! -x "$train_bin" ]]; then
        rebuild_linked=1
      elif native_gpt_source_newer_than "$train_bin"; then
        rebuild_linked=1
      elif tile_ops_source_newer_than "$train_bin"; then
        rebuild_linked=1
      fi
      if [[ "$rebuild_linked" == "1" ]]; then
        bash "$ROOT_DIR/tools/build_native_gpt_cli_linked.sh" "$train_bin" >&2
      fi
      ;;
    nfn_gpt_native_train)
      local rebuild_dynamic=0
      if [[ ! -x "$train_bin" ]]; then
        rebuild_dynamic=1
      elif native_gpt_source_newer_than "$train_bin"; then
        rebuild_dynamic=1
      fi
      if [[ "$rebuild_dynamic" == "1" ]]; then
        bash "$ROOT_DIR/tools/build_native_gpt_cli.sh" "$train_bin" >&2
      fi
      ;;
  esac
}

if [[ "$IS_DRY_RUN_PLAN" != "1" ]]; then
  ensure_native_gpt_trainer_current "$NFN_NATIVE_GPT_TRAIN_BIN" "$NFN_NATIVE_GPT_TRAIN_BIN_EXPLICIT"
  ensure_native_gpt_trainer_current "$NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN" "$NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN_EXPLICIT"
fi

if [[ ! -x "$NFN_NATIVE_GPT_TRAIN_BIN" ]]; then
  echo "NeuralFn native GPT trainer is not executable: $NFN_NATIVE_GPT_TRAIN_BIN" >&2
  exit 2
fi
if [[ ! -x "$NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN" ]]; then
  echo "Candidate NeuralFn native GPT trainer is not executable: $NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN" >&2
  exit 2
fi
case "${INCLUDE_LLMK_REFERENCE,,}" in
  "1"|"true"|"yes"|"on")
    if [[ "$IS_DRY_RUN_PLAN" != "1" && ! -x "$LLM_KITTENS_TRAIN_BIN" ]]; then
      echo "llm.kittens train_gpt2cu is not executable: $LLM_KITTENS_TRAIN_BIN" >&2
      exit 2
    fi
    ;;
  "0"|"false"|"no"|"off"|"")
    ;;
  *)
    echo "Unsupported NFN_SM120_NATIVE_INCLUDE_LLMK_REFERENCE value: $INCLUDE_LLMK_REFERENCE" >&2
    exit 2
    ;;
esac
NFN_NATIVE_TILE_OPS_ARG="$(tile_ops_arg_for "$NFN_NATIVE_GPT_TRAIN_BIN" "$NFN_NATIVE_TILE_OPS_LIB" "$NFN_NATIVE_TILE_OPS_LIB_EXPLICIT")"
NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_ARG="$(tile_ops_arg_for "$NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN" "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB" "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB_EXPLICIT")"
if [[ "$IS_DRY_RUN_PLAN" != "1" && "$NFN_NATIVE_TILE_OPS_ARG" != "linked" && ! -f "$NFN_NATIVE_TILE_OPS_ARG" ]]; then
  echo "Baseline NeuralFn Tile ops library is missing: $NFN_NATIVE_TILE_OPS_LIB" >&2
  exit 2
fi
if [[ "$IS_DRY_RUN_PLAN" != "1" && "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_ARG" != "linked" && ! -f "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_ARG" ]]; then
  echo "Candidate NeuralFn Tile ops library is missing: $NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB" >&2
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

append_split_args() {
  local -n target_ref="$1"
  local raw="$2"
  local item
  for item in $raw; do
    target_ref+=("$item")
  done
}

append_env_overrides() {
  local -n target_ref="$1"
  local raw="$2"
  local item
  local part
  local rest
  local parts
  for item in $raw; do
    rest="$item"
    parts=()
    while [[ "$rest" =~ ^(.+),([A-Za-z_][A-Za-z0-9_]*=.*)$ ]]; do
      parts=("${BASH_REMATCH[2]}" "${parts[@]}")
      rest="${BASH_REMATCH[1]}"
    done
    target_ref+=("$rest")
    for part in "${parts[@]}"; do
      target_ref+=("$part")
    done
  done
}

common_args=(
  --backend tile-cuda
  --tinystories
  --max-steps "$STEPS"
  --train-batch-tokens "$TRAIN_BATCH_TOKENS"
  --eval-every-steps 0
  --native-cuda-sample-every "$SAMPLE_EVERY"
  --native-cuda-generate-tokens "$GENERATE_TOKENS"
  --native-cuda-checkpoint-every "$CHECKPOINT_EVERY"
  --native-cuda-activation "$ACTIVATION"
  --no-checkpoint
)
case "${STARTUP_ONLY,,}" in
  "1"|"true"|"yes"|"on")
    common_args+=(--startup-only)
    ;;
esac
if [[ -n "$TEMPLATE_NAME" ]]; then
  common_args+=(--template-name "$TEMPLATE_NAME")
fi
if [[ -n "$GRAPH_FILE" ]]; then
  common_args+=(--graph-file "$GRAPH_FILE")
fi
append_split_args common_args "$COMMON_EXTRA_ARGS_RAW"

baseline_args=("$NFN_NATIVE_GPT_TRAIN_BIN" "${common_args[@]}" --tile-ops-lib "$NFN_NATIVE_TILE_OPS_ARG")
candidate_args=("$NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN" "${common_args[@]}" --tile-ops-lib "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_ARG")
append_split_args baseline_args "$BASELINE_EXTRA_ARGS_RAW"
append_split_args candidate_args "$CANDIDATE_EXTRA_ARGS_RAW"

baseline_cmd="$(join_command "${baseline_args[@]}")"
candidate_cmd="$(join_command "${candidate_args[@]}")"
reference_args=(
  "$LLM_KITTENS_TRAIN_BIN"
  -i "$LLM_KITTENS_TINYSTORIES_DIR/TinyStories_train.bin"
  -j "$LLM_KITTENS_TINYSTORIES_DIR/TinyStories_val.bin"
  -o "$REFERENCE_OUTPUT_DIR"
  -v 250
  -s "$SAMPLE_EVERY"
  -g "$GENERATE_TOKENS"
  -h 0
  -b 64
  -t 1024
  -d "$TRAIN_BATCH_TOKENS"
  -r 0
  -z 1
  -c 0.1
  -l 0.0006
  -q 0.0
  -u 60
  -n "$CHECKPOINT_EVERY"
  -y 0
  -e d12
  -af "$ACTIVATION"
  -x "$STEPS"
)
reference_cmd="$(join_command "${reference_args[@]}")"
reference_paired_args=()
case "${INCLUDE_LLMK_REFERENCE,,}" in
  "1"|"true"|"yes"|"on")
    reference_paired_args=(--reference "$reference_cmd")
    ;;
esac

if [[ -n "${reference_paired_args[*]-}" &&
      -z "$USER_MAX_CANDIDATE_REFERENCE_RATIO_RAW" &&
      -z "$USER_MIN_CANDIDATE_REFERENCE_RATIO_RAW" &&
      "$DEFAULT_VS_LEGACY_PROFILE" != "1" ]]; then
  has_reference_candidate_change=0
  if [[ "$NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN" != "$NFN_NATIVE_GPT_TRAIN_BIN" ||
        "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB" != "$NFN_NATIVE_TILE_OPS_LIB" ||
        -n "$CANDIDATE_TILE_OPS_BUILD_FLAGS" ||
        -n "$CANDIDATE_ENV_RAW" ||
        -n "$CANDIDATE_EXTRA_ARGS_RAW" ]]; then
    has_reference_candidate_change=1
  fi
  if [[ "$has_reference_candidate_change" == "1" ]]; then
    case "${STARTUP_ONLY,,}" in
      "1"|"true"|"yes"|"on")
        ;;
      *)
        MAX_CANDIDATE_REFERENCE_RATIO_RAW="train_loop_wall_ms_per_step=1.000"
        case "${TRAIN_LOOP_EVENT_TIMING,,}" in
          "1"|"true"|"yes"|"on")
            if [[ "$STEPS" =~ ^[0-9]+$ && "$STEPS" -gt 1 ]]; then
              MAX_CANDIDATE_REFERENCE_RATIO_RAW+=" train_loop_cuda_event_first_step_wall_ms_per_step=1.000"
              MAX_CANDIDATE_REFERENCE_RATIO_RAW+=" train_loop_cuda_event_steady_state_wall_ms_per_step=1.000"
            fi
            ;;
        esac
        MIN_CANDIDATE_REFERENCE_RATIO_RAW="train_tokens_per_second=1.000"
        ;;
    esac
  fi
fi

profile_args=()
case "${PROFILE_DIR_RAW,,}" in
  ""|"0"|"false"|"no"|"none"|"off")
    ;;
  *)
    profile_args=(--append-native-profile-json-dir "$PROFILE_DIR_RAW")
    ;;
esac
case "${STAGE_TIMING,,}" in
  "1"|"true"|"yes"|"on")
    profile_args+=(--native-stage-timing)
    export NFN_NATIVE_GPT_STAGE_TIMING_MAX_EVENTS="${NFN_NATIVE_GPT_STAGE_TIMING_MAX_EVENTS:-80000}"
    ;;
esac

paired_args=()
if [[ -n "$CANDIDATE_PROFILE" ]]; then
  paired_args+=(--metadata "candidate_profile=$CANDIDATE_PROFILE")
fi
if [[ -n "$CANDIDATE_TILE_OPS_BUILD_FLAGS" ]]; then
  paired_args+=(--metadata "candidate_tile_ops_build_flags=$CANDIDATE_TILE_OPS_BUILD_FLAGS")
fi
if [[ -n "$CANDIDATE_NOTE" ]]; then
  paired_args+=(--metadata "candidate_note=$CANDIDATE_NOTE")
fi
if [[ "$CANDIDATE_ENV_RAW" == *"NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE=1"* ]]; then
  paired_args+=(--metadata "candidate_true_fused_cooperative_env=NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE=1")
fi
if [[ "$CANDIDATE_ENV_RAW" == *"NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION=1"* ]]; then
  paired_args+=(--metadata "candidate_true_fused_production_env=NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION=1")
fi
case "${LM_HEAD_BACKWARD_PREFLIGHT,,}" in
  "auto")
    ;;
  "0"|"false"|"no"|"off"|"none"|"")
    LM_HEAD_BACKWARD_PREFLIGHT_PROFILE=""
    ;;
  "1"|"true"|"yes"|"on")
    if [[ -z "$LM_HEAD_BACKWARD_PREFLIGHT_PROFILE" ]]; then
      LM_HEAD_BACKWARD_PREFLIGHT_PROFILE="trainer-chunk"
    fi
    ;;
  *)
    LM_HEAD_BACKWARD_PREFLIGHT_PROFILE="$LM_HEAD_BACKWARD_PREFLIGHT"
    ;;
esac
if [[ -n "$LM_HEAD_BACKWARD_PREFLIGHT_PROFILE" ]]; then
  paired_args+=(--metadata "lm_head_backward_preflight_profile=$LM_HEAD_BACKWARD_PREFLIGHT_PROFILE")
fi
if [[ "$DEFAULT_VS_LEGACY_PROFILE" == "1" ]]; then
  paired_args+=(--metadata "candidate_gate_scope=default-vs-legacy")
fi
if [[ "$FORCE_DISABLE_ROUTE_CHANGE" == "1" ]]; then
  paired_args+=(--metadata "candidate_route_change_gate=disabled")
fi
common_env_items=()
baseline_env_items=()
candidate_env_items=()
append_env_overrides common_env_items "$COMMON_ENV_RAW"
append_env_overrides baseline_env_items "$BASELINE_ENV_RAW"
append_env_overrides candidate_env_items "$CANDIDATE_ENV_RAW"
case "${CUDA_VERSION_PREFLIGHT,,}" in
  "1"|"true"|"yes"|"on")
    paired_args+=(--baseline-env "NFN_NATIVE_GPT_CUDA_VERSION_PREFLIGHT=1")
    paired_args+=(--candidate-env "NFN_NATIVE_GPT_CUDA_VERSION_PREFLIGHT=1")
    ;;
  "0"|"false"|"no"|"off")
    ;;
  *)
    echo "Unsupported NFN_SM120_NATIVE_CUDA_VERSION_PREFLIGHT value: $CUDA_VERSION_PREFLIGHT" >&2
    exit 2
    ;;
esac
case "${TRAIN_LOOP_EVENT_TIMING,,}" in
  "1"|"true"|"yes"|"on")
    paired_args+=(--baseline-env "NFN_NATIVE_GPT_TRAIN_LOOP_EVENT_TIMING=1")
    paired_args+=(--candidate-env "NFN_NATIVE_GPT_TRAIN_LOOP_EVENT_TIMING=1")
    ;;
  "0"|"false"|"no"|"off")
    ;;
  *)
    echo "Unsupported NFN_SM120_NATIVE_TRAIN_LOOP_EVENT_TIMING value: $TRAIN_LOOP_EVENT_TIMING" >&2
    exit 2
    ;;
esac
case "${SETUP_EVENT_TIMING,,}" in
  "1"|"true"|"yes"|"on")
    paired_args+=(--baseline-env "NFN_NATIVE_GPT_SETUP_EVENT_TIMING=1")
    paired_args+=(--candidate-env "NFN_NATIVE_GPT_SETUP_EVENT_TIMING=1")
    ;;
  "0"|"false"|"no"|"off")
    ;;
  *)
    echo "Unsupported NFN_SM120_NATIVE_SETUP_EVENT_TIMING value: $SETUP_EVENT_TIMING" >&2
    exit 2
    ;;
esac
case "${ATTENTION_SECTION_TIMING,,}" in
  "1"|"true"|"yes"|"on")
    AUTO_ATTENTION_SECTION_TIMING=1
    ;;
  "0"|"false"|"no"|"off")
    ;;
  *)
    echo "Unsupported NFN_SM120_NATIVE_ATTENTION_SECTION_TIMING value: $ATTENTION_SECTION_TIMING" >&2
    exit 2
    ;;
esac
for item in "${common_env_items[@]}"; do
  paired_args+=(--baseline-env "$item")
  paired_args+=(--candidate-env "$item")
done
if [[ "$AUTO_ATTENTION_SECTION_TIMING" == "1" ]]; then
  paired_args+=(--baseline-env "NFN_NATIVE_GPT_ATTENTION_BACKWARD_SECTION_TIMING=1")
  paired_args+=(--candidate-env "NFN_NATIVE_GPT_ATTENTION_BACKWARD_SECTION_TIMING=1")
fi
case "${LINEAR_SHAPE_STATS,,}" in
  "1"|"true"|"yes"|"on")
    paired_args+=(--baseline-env "NFN_NATIVE_GPT_LINEAR_SHAPE_STATS=1")
    paired_args+=(--candidate-env "NFN_NATIVE_GPT_LINEAR_SHAPE_STATS=1")
    ;;
esac
for item in "${baseline_env_items[@]}"; do
  paired_args+=(--baseline-env "$item")
done
for item in "${candidate_env_items[@]}"; do
  paired_args+=(--candidate-env "$item")
done
for item in $MAX_CANDIDATE_RATIO_RAW; do
  paired_args+=(--max-candidate-ratio "$item")
done
for item in $MIN_CANDIDATE_RATIO_RAW; do
  paired_args+=(--min-candidate-ratio "$item")
done
for item in $MAX_CANDIDATE_REFERENCE_RATIO_RAW; do
  paired_args+=(--max-candidate-reference-ratio "$item")
done
for item in $MIN_CANDIDATE_REFERENCE_RATIO_RAW; do
  paired_args+=(--min-candidate-reference-ratio "$item")
done
case "${REQUIRE_NATIVE_ROUTE_CHANGE,,}" in
  "1"|"true"|"yes"|"on")
    paired_args+=(--require-native-route-change)
    ;;
  "0"|"false"|"no"|"off")
    ;;
  "auto"|"")
    has_candidate_change=0
    if [[ "$NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN" != "$NFN_NATIVE_GPT_TRAIN_BIN" ||
          "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB" != "$NFN_NATIVE_TILE_OPS_LIB" ||
          -n "$CANDIDATE_TILE_OPS_BUILD_FLAGS" ||
          -n "$CANDIDATE_ENV_RAW" ||
          -n "$CANDIDATE_EXTRA_ARGS_RAW" ]]; then
      has_candidate_change=1
    fi
    case "${DRY_RUN_PLAN,,}" in
      "1"|"true"|"yes"|"on")
        has_candidate_change=0
        ;;
    esac
    if [[ "$has_candidate_change" == "1" ]]; then
      paired_args+=(--require-native-route-change)
    fi
    ;;
  *)
    echo "Unsupported NFN_SM120_NATIVE_REQUIRE_ROUTE_CHANGE value: $REQUIRE_NATIVE_ROUTE_CHANGE" >&2
    exit 2
    ;;
esac
case "${REQUIRE_NATIVE_LM_HEAD_TRUE_FUSED,,}" in
  "1"|"true"|"yes"|"on")
    paired_args+=(--require-native-lm-head-true-fused)
    ;;
  "0"|"false"|"no"|"off")
    ;;
  *)
    echo "Unsupported NFN_SM120_NATIVE_REQUIRE_LM_HEAD_TRUE_FUSED value: $REQUIRE_NATIVE_LM_HEAD_TRUE_FUSED" >&2
    exit 2
    ;;
esac
case "${DRY_RUN_PLAN,,}" in
  "1"|"true"|"yes"|"on")
    paired_args+=(--dry-run-plan)
    ;;
esac
case "${ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE,,}" in
  "1"|"true"|"yes"|"on")
    paired_args+=(--allow-stale-selected-gpu-utilization-without-compute-processes)
    ;;
esac

run_lm_head_backward_preflight() {
  if [[ -z "$LM_HEAD_BACKWARD_PREFLIGHT_PROFILE" ]]; then
    return 0
  fi
  case "${DRY_RUN_PLAN,,}" in
    "1"|"true"|"yes"|"on")
      return 0
      ;;
  esac
  (
    export NFN_LM_HEAD_BACKWARD_PROFILE="$LM_HEAD_BACKWARD_PREFLIGHT_PROFILE"
    export NFN_LM_HEAD_BACKWARD_ALLOW_REJECTED_PROFILE="$ALLOW_REJECTED_CANDIDATE_PROFILE"
    export NFN_LM_HEAD_BACKWARD_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
    export NFN_LM_HEAD_BACKWARD_REQUIRE_IDLE_SELECTED_GPU=1
    export NFN_LM_HEAD_BACKWARD_MAX_SELECTED_GPU_UTILIZATION_PCT="$MAX_GPU_UTILIZATION"
    export NFN_LM_HEAD_BACKWARD_SELECTED_GPU_UTILIZATION_RETRIES="$SELECTED_GPU_UTILIZATION_RETRIES"
    export NFN_LM_HEAD_BACKWARD_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS="$SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS"
    export NFN_LM_HEAD_BACKWARD_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE="$ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE"
    if [[ -n "$LM_HEAD_BACKWARD_MAX_REFERENCE_GAP_MS" ]]; then
      export NFN_LM_HEAD_BACKWARD_MAX_REFERENCE_GAP_MS="$LM_HEAD_BACKWARD_MAX_REFERENCE_GAP_MS"
    fi
    if [[ -n "$LM_HEAD_BACKWARD_MAX_REFERENCE_WITH_LOGITS_GAP_MS" ]]; then
      export NFN_LM_HEAD_BACKWARD_MAX_REFERENCE_WITH_LOGITS_GAP_MS="$LM_HEAD_BACKWARD_MAX_REFERENCE_WITH_LOGITS_GAP_MS"
    fi
    if [[ -n "$LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_GAP_MS" ]]; then
      export NFN_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_GAP_MS="$LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_GAP_MS"
    fi
    if [[ -n "$LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_WITH_LOGITS_GAP_MS" ]]; then
      export NFN_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_WITH_LOGITS_GAP_MS="$LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_WITH_LOGITS_GAP_MS"
    fi
    bash "$ROOT_DIR/tools/bench_lm_head_backward_candidate.sh"
  )
}

run_lm_head_backward_preflight

cd "$ROOT_DIR"
python tools/paired_kernel_speed.py \
  --baseline "$baseline_cmd" \
  --candidate "$candidate_cmd" \
  "${reference_paired_args[@]}" \
  --samples "$SAMPLES" \
  --warmup "$WARMUP" \
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES_VALUE" \
  --cuda-device-max-connections "$CUDA_DEVICE_MAX_CONNECTIONS_VALUE" \
  --require-idle-selected-gpu \
  --max-selected-gpu-utilization-pct "$MAX_GPU_UTILIZATION" \
  --selected-gpu-utilization-retries "$SELECTED_GPU_UTILIZATION_RETRIES" \
  --selected-gpu-utilization-retry-interval-seconds "$SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS" \
  --command-timeout-seconds "$COMMAND_TIMEOUT_SECONDS" \
  "${profile_args[@]}" \
  "${paired_args[@]}" \
  --json-out "$JSON_OUT"

if [[ "$STRICT_GROUPED_CUBLASLT_PROBE" == "1" && "$IS_DRY_RUN_PLAN" != "1" ]]; then
  python - "$JSON_OUT" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
metrics = data.get("candidate_native_metrics", {})

def metric_mean(name):
    value = metrics.get(name)
    if isinstance(value, dict):
        value = value.get("mean")
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    return value

layout_status = metric_mean("linear_cublaslt_grouped_layout_probe_status")
matmul_status = metric_mean("linear_cublaslt_grouped_matmul_probe_status")
if layout_status != 0 or matmul_status != 0:
    raise SystemExit(
        "required cuBLASLt grouped probe failed: "
        f"layout_status={layout_status!r} matmul_status={matmul_status!r}"
    )
PY
fi
