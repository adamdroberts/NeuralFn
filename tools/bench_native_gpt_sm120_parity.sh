#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLM_KITTENS_ROOT="${LLM_KITTENS_ROOT:-/mnt/disk2/dev/open-source/llm.kittens}"
LLM_KITTENS_TRAIN_BIN="${LLM_KITTENS_TRAIN_BIN:-$LLM_KITTENS_ROOT/train_gpt2cu}"
LLM_KITTENS_TINYSTORIES_DIR="${LLM_KITTENS_TINYSTORIES_DIR:-$LLM_KITTENS_ROOT/dev/data/tinystories}"
NFN_SM120_REFERENCE_CUDA_LD_LIBRARY_PATH="${NFN_SM120_REFERENCE_CUDA_LD_LIBRARY_PATH-/usr/local/cuda/lib64:/usr/lib/wsl/lib}"
NFN_NATIVE_GPT_TRAIN_BIN_EXPLICIT="${NFN_NATIVE_GPT_TRAIN_BIN+x}"
if [[ -z "${NFN_NATIVE_GPT_TRAIN_BIN-}" && -x "$ROOT_DIR/build/nfn_gpt_native_train_linked" ]]; then
  NFN_NATIVE_GPT_TRAIN_BIN="$ROOT_DIR/build/nfn_gpt_native_train_linked"
else
  NFN_NATIVE_GPT_TRAIN_BIN="${NFN_NATIVE_GPT_TRAIN_BIN:-$ROOT_DIR/build/nfn_gpt_native_train}"
fi
NFN_NATIVE_TILE_OPS_LIB_EXPLICIT="${NFN_NATIVE_TILE_OPS_LIB+x}"
NFN_NATIVE_TILE_OPS_LIB="${NFN_NATIVE_TILE_OPS_LIB:-$ROOT_DIR/build/libnfn_native_train_tile_ops.so}"

env_or_alias3() {
  local primary="$1"
  local parity_alias="$2"
  local generic_alias="$3"
  local default_value="$4"
  if [[ -n "${!primary-}" ]]; then
    printf '%s' "${!primary}"
  elif [[ -n "${!parity_alias-}" ]]; then
    printf '%s' "${!parity_alias}"
  elif [[ -n "${!generic_alias-}" ]]; then
    printf '%s' "${!generic_alias}"
  else
    printf '%s' "$default_value"
  fi
}

env_or_alias4() {
  local primary="$1"
  local native_alias="$2"
  local parity_alias="$3"
  local generic_alias="$4"
  local default_value="$5"
  if [[ -n "${!primary-}" ]]; then
    printf '%s' "${!primary}"
  elif [[ -n "${!native_alias-}" ]]; then
    printf '%s' "${!native_alias}"
  elif [[ -n "${!parity_alias-}" ]]; then
    printf '%s' "${!parity_alias}"
  elif [[ -n "${!generic_alias-}" ]]; then
    printf '%s' "${!generic_alias}"
  else
    printf '%s' "$default_value"
  fi
}

STEPS="$(env_or_alias3 NFN_SM120_NATIVE_STEPS NFN_SM120_PARITY_STEPS NFN_SM120_STEPS 10)"
SAMPLES="$(env_or_alias3 NFN_SM120_NATIVE_SAMPLES NFN_SM120_PARITY_SAMPLES NFN_SM120_SAMPLES 3)"
# Keep the default at two warmup pairs unless a same-script gate proves a
# higher value is more stable. A 2026-06-29 dedicated RTX 5090 rerun with
# NFN_SM120_PARITY_WARMUP=3 regressed the median steady-state CUDA-event ratio
# to 1.003405x and failed the existing 1.003 parity gate.
WARMUP="$(env_or_alias3 NFN_SM120_NATIVE_WARMUP NFN_SM120_PARITY_WARMUP NFN_SM120_WARMUP 2)"
TRAIN_BATCH_TOKENS="$(env_or_alias3 NFN_SM120_NATIVE_TRAIN_BATCH_TOKENS NFN_SM120_PARITY_TRAIN_BATCH_TOKENS NFN_SM120_TRAIN_BATCH_TOKENS 524288)"
CUDA_VISIBLE_DEVICES_VALUE="$(env_or_alias3 NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES NFN_SM120_CUDA_VISIBLE_DEVICES dedicated)"
CUDA_DEVICE_MAX_CONNECTIONS_VALUE="$(env_or_alias3 NFN_SM120_NATIVE_CUDA_DEVICE_MAX_CONNECTIONS NFN_SM120_PARITY_CUDA_DEVICE_MAX_CONNECTIONS NFN_SM120_CUDA_DEVICE_MAX_CONNECTIONS 1)"
MAX_GPU_UTILIZATION="$(env_or_alias3 NFN_SM120_NATIVE_MAX_GPU_UTILIZATION_PCT NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT NFN_SM120_MAX_GPU_UTILIZATION_PCT 15)"
SELECTED_GPU_UTILIZATION_RETRIES="$(env_or_alias3 NFN_SM120_NATIVE_SELECTED_GPU_UTILIZATION_RETRIES NFN_SM120_PARITY_SELECTED_GPU_UTILIZATION_RETRIES NFN_SM120_SELECTED_GPU_UTILIZATION_RETRIES 3)"
SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS="$(env_or_alias3 NFN_SM120_NATIVE_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS NFN_SM120_PARITY_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS NFN_SM120_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS 0.25)"
ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE="$(env_or_alias3 NFN_SM120_NATIVE_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE NFN_SM120_PARITY_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE NFN_SM120_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE 1)"
COMMAND_TIMEOUT_SECONDS="$(env_or_alias3 NFN_SM120_NATIVE_COMMAND_TIMEOUT_SECONDS NFN_SM120_PARITY_COMMAND_TIMEOUT_SECONDS NFN_SM120_COMMAND_TIMEOUT_SECONDS 300)"
ACTIVATION="$(env_or_alias3 NFN_SM120_NATIVE_ACTIVATION NFN_SM120_PARITY_ACTIVATION NFN_SM120_ACTIVATION gelu)"
SAMPLE_EVERY="$(env_or_alias3 NFN_SM120_NATIVE_SAMPLE_EVERY NFN_SM120_PARITY_SAMPLE_EVERY NFN_SM120_SAMPLE_EVERY 0)"
CHECKPOINT_EVERY="$(env_or_alias3 NFN_SM120_NATIVE_CHECKPOINT_EVERY NFN_SM120_PARITY_CHECKPOINT_EVERY NFN_SM120_CHECKPOINT_EVERY 0)"
GENERATE_TOKENS="$(env_or_alias3 NFN_SM120_NATIVE_GENERATE_TOKENS NFN_SM120_PARITY_GENERATE_TOKENS NFN_SM120_GENERATE_TOKENS 144)"
TRAIN_LOSS_EVERY="$(env_or_alias4 NFN_SM120_NATIVE_TRAIN_LOSS_EVERY_STEPS NFN_SM120_NATIVE_TRAIN_LOSS_EVERY NFN_SM120_PARITY_TRAIN_LOSS_EVERY_STEPS NFN_SM120_TRAIN_LOSS_EVERY_STEPS 0)"
TRAIN_LOOP_EVENT_TIMING="$(env_or_alias3 NFN_SM120_NATIVE_TRAIN_LOOP_EVENT_TIMING NFN_SM120_PARITY_TRAIN_LOOP_EVENT_TIMING NFN_SM120_TRAIN_LOOP_EVENT_TIMING 1)"
SETUP_EVENT_TIMING="$(env_or_alias3 NFN_SM120_NATIVE_SETUP_EVENT_TIMING NFN_SM120_PARITY_SETUP_EVENT_TIMING NFN_SM120_SETUP_EVENT_TIMING 0)"
ATTENTION_SECTION_TIMING="$(env_or_alias3 NFN_SM120_NATIVE_ATTENTION_SECTION_TIMING NFN_SM120_PARITY_ATTENTION_SECTION_TIMING NFN_SM120_ATTENTION_SECTION_TIMING 0)"
CANDIDATE_ENV_RAW="$(env_or_alias3 NFN_SM120_NATIVE_CANDIDATE_ENV NFN_SM120_PARITY_CANDIDATE_ENV NFN_SM120_CANDIDATE_ENV "")"
CANDIDATE_PROFILE_RAW="$(env_or_alias4 NFN_SM120_NATIVE_CANDIDATE_PROFILE NFN_SM120_NATIVE_PARITY_PROFILE NFN_SM120_PARITY_CANDIDATE_PROFILE NFN_SM120_PARITY_PROFILE "")"
REFERENCE_ENV_RAW="$(env_or_alias3 NFN_SM120_NATIVE_REFERENCE_ENV NFN_SM120_PARITY_REFERENCE_ENV NFN_SM120_REFERENCE_ENV "")"
DEFAULT_LONG_RUN_DEFER_PREWARM="$(env_or_alias3 NFN_SM120_NATIVE_DEFAULT_LONG_RUN_DEFER_PREWARM NFN_SM120_PARITY_DEFAULT_LONG_RUN_DEFER_PREWARM NFN_SM120_DEFAULT_LONG_RUN_DEFER_PREWARM 1)"
LONG_RUN_DEFER_PREWARM_MIN_WARMUP="$(env_or_alias3 NFN_SM120_NATIVE_LONG_RUN_DEFER_PREWARM_MIN_WARMUP NFN_SM120_PARITY_LONG_RUN_DEFER_PREWARM_MIN_WARMUP NFN_SM120_LONG_RUN_DEFER_PREWARM_MIN_WARMUP 60)"
LONG_RUN_DEFER_PREWARM_MIN_STEPS="$(env_or_alias3 NFN_SM120_NATIVE_LONG_RUN_DEFER_PREWARM_MIN_STEPS NFN_SM120_PARITY_LONG_RUN_DEFER_PREWARM_MIN_STEPS NFN_SM120_LONG_RUN_DEFER_PREWARM_MIN_STEPS 10)"
ALLOW_LOW_LONG_RUN_DEFER_PREWARM_DIAGNOSTIC="$(env_or_alias3 NFN_SM120_NATIVE_ALLOW_LOW_LONG_RUN_DEFER_PREWARM_DIAGNOSTIC NFN_SM120_PARITY_ALLOW_LOW_LONG_RUN_DEFER_PREWARM_DIAGNOSTIC NFN_SM120_ALLOW_LOW_LONG_RUN_DEFER_PREWARM_DIAGNOSTIC 0)"
DEFAULT_LONG_RUN_DEFER_PREWARM_APPLIED=0
DEFAULT_LONG_RUN_DEFER_PREWARM_WARMUP_FLOOR_APPLIED=0
DEFAULT_LONG_RUN_DEFER_PREWARM_STEP_FLOOR_APPLIED=0
DEFAULT_LONG_RUN_DEFER_PREWARM_WARMUP_FLOOR_DRY_RUN_WOULD_APPLY=0
DEFAULT_LONG_RUN_DEFER_PREWARM_STEP_FLOOR_DRY_RUN_WOULD_APPLY=0
DEFAULT_LONG_RUN_DEFER_PREWARM_LOW_WARMUP_DIAGNOSTIC=0
DEFAULT_LONG_RUN_DEFER_PREWARM_LOW_STEP_DIAGNOSTIC=0
JSON_OUT="$(env_or_alias3 NFN_SM120_NATIVE_JSON_OUT NFN_SM120_PARITY_JSON_OUT NFN_SM120_JSON_OUT "/tmp/nfn_sm120_parity_${STEPS}step.json")"
PROFILE_DIR_RAW="$(env_or_alias3 NFN_SM120_NATIVE_PROFILE_DIR NFN_SM120_PARITY_PROFILE_DIR NFN_SM120_PROFILE_DIR "/tmp/nfn_sm120_parity_profiles_${STEPS}step")"
STAGE_TIMING="$(env_or_alias4 NFN_SM120_NATIVE_STAGE_TIMING NFN_SM120_NATIVE_PARITY_STAGE_TIMING NFN_SM120_PARITY_STAGE_TIMING NFN_SM120_STAGE_TIMING 0)"
REFERENCE_OUTPUT_DIR="$(env_or_alias3 NFN_SM120_NATIVE_REFERENCE_OUTPUT_DIR NFN_SM120_PARITY_REFERENCE_OUTPUT_DIR NFN_SM120_REFERENCE_OUTPUT_DIR /tmp/nfn_llmk_sm120_parity)"
DRY_RUN_PLAN="$(env_or_alias3 NFN_SM120_NATIVE_DRY_RUN_PLAN NFN_SM120_PARITY_DRY_RUN_PLAN NFN_SM120_DRY_RUN_PLAN 0)"
MAX_CANDIDATE_RATIO_RAW="$(env_or_alias3 NFN_SM120_NATIVE_MAX_CANDIDATE_RATIO NFN_SM120_PARITY_MAX_CANDIDATE_RATIO NFN_SM120_MAX_CANDIDATE_RATIO "")"
MAX_CANDIDATE_RATIO_EXPLICIT=0
if [[ -n "${NFN_SM120_NATIVE_MAX_CANDIDATE_RATIO-}" ||
      -n "${NFN_SM120_PARITY_MAX_CANDIDATE_RATIO-}" ||
      -n "${NFN_SM120_MAX_CANDIDATE_RATIO-}" ]]; then
  MAX_CANDIDATE_RATIO_EXPLICIT=1
fi
MIN_CANDIDATE_RATIO_RAW="$(env_or_alias3 NFN_SM120_NATIVE_MIN_CANDIDATE_RATIO NFN_SM120_PARITY_MIN_CANDIDATE_RATIO NFN_SM120_MIN_CANDIDATE_RATIO "")"
DEFAULT_MAX_TRAIN_LOOP_RATIO="$(env_or_alias3 NFN_SM120_NATIVE_PARITY_MAX_TRAIN_LOOP_RATIO NFN_SM120_PARITY_MAX_TRAIN_LOOP_RATIO NFN_SM120_MAX_TRAIN_LOOP_RATIO 1.003)"
DEFAULT_MAX_STEADY_STATE_RATIO="$(env_or_alias3 NFN_SM120_NATIVE_PARITY_MAX_STEADY_STATE_RATIO NFN_SM120_PARITY_MAX_STEADY_STATE_RATIO NFN_SM120_MAX_STEADY_STATE_RATIO 1.003)"
DEFAULT_MIN_STEADY_STATE_TOKENS_RATIO="$(env_or_alias3 NFN_SM120_NATIVE_PARITY_MIN_STEADY_STATE_TOKENS_RATIO NFN_SM120_PARITY_MIN_STEADY_STATE_TOKENS_RATIO NFN_SM120_MIN_STEADY_STATE_TOKENS_RATIO 1.000)"
ENFORCE_GATE="$(env_or_alias3 NFN_SM120_NATIVE_ENFORCE_PARITY_GATE NFN_SM120_PARITY_ENFORCE_GATE NFN_SM120_ENFORCE_PARITY_GATE 1)"
DEFAULT_RATIO_GATE_SKIPPED_FOR_STAGE_TIMING=0
case "${ENFORCE_GATE,,}:${STAGE_TIMING,,}:$MAX_CANDIDATE_RATIO_EXPLICIT" in
  1:1:0|1:true:0|1:yes:0|1:on:0|true:1:0|true:true:0|true:yes:0|true:on:0|yes:1:0|yes:true:0|yes:yes:0|yes:on:0|on:1:0|on:true:0|on:yes:0|on:on:0)
    DEFAULT_RATIO_GATE_SKIPPED_FOR_STAGE_TIMING=1
    ;;
esac
REQUIRE_NATIVE_LM_HEAD_TRUE_FUSED="$(env_or_alias4 NFN_SM120_NATIVE_REQUIRE_LM_HEAD_TRUE_FUSED NFN_SM120_NATIVE_REQUIRE_NATIVE_LM_HEAD_TRUE_FUSED NFN_SM120_PARITY_REQUIRE_NATIVE_LM_HEAD_TRUE_FUSED NFN_SM120_REQUIRE_NATIVE_LM_HEAD_TRUE_FUSED 0)"
if [[ -n "$CANDIDATE_PROFILE_RAW" ]]; then
  cat >&2 <<EOF
NFN_SM120_NATIVE_CANDIDATE_PROFILE/NFN_SM120_PARITY_CANDIDATE_PROFILE/NFN_SM120_PARITY_PROFILE is not supported by this llm.kittens parity wrapper.
Use tools/bench_native_gpt_sm120_candidate.sh with NFN_SM120_NATIVE_CANDIDATE_PROFILE for named native-vs-native route bisection, or set NFN_SM120_PARITY_CANDIDATE_ENV explicitly when comparing NeuralFn against llm.kittens.
Refusing to run because a parity profile would otherwise be ignored and produce no-op speed evidence.
EOF
  exit 2
fi
case "${ALLOW_LOW_LONG_RUN_DEFER_PREWARM_DIAGNOSTIC,,}" in
  "1"|"true"|"yes"|"on")
    ALLOW_LOW_LONG_RUN_DEFER_PREWARM_DIAGNOSTIC=1
    ;;
  "0"|"false"|"no"|"off"|"")
    ALLOW_LOW_LONG_RUN_DEFER_PREWARM_DIAGNOSTIC=0
    ;;
  *)
    echo "Unsupported NFN_SM120_PARITY_ALLOW_LOW_LONG_RUN_DEFER_PREWARM_DIAGNOSTIC value: $ALLOW_LOW_LONG_RUN_DEFER_PREWARM_DIAGNOSTIC" >&2
    exit 2
    ;;
esac
case "${DEFAULT_LONG_RUN_DEFER_PREWARM,,}" in
  "1"|"true"|"yes"|"on")
    prewarm_policy_text="$CANDIDATE_ENV_RAW"
    case "$prewarm_policy_text" in
      *NFN_NATIVE_GPT_DEFER_PREWARM_AFTER_STEPS=*|*NFN_NATIVE_GPT2_DEFER_PREWARM_AFTER_STEPS=*|*NFN_TILE_CUDA_DEFER_PREWARM_AFTER_STEPS=*|*NFN_NATIVE_GPT_FAST_STARTUP=*|*NFN_NATIVE_GPT2_FAST_STARTUP=*|*NFN_TILE_CUDA_FAST_STARTUP=*|*NFN_NATIVE_GPT_PREWARM_TK_QKV_FORWARD=*|*NFN_NATIVE_GPT2_PREWARM_TK_QKV_FORWARD=*|*NFN_TILE_CUDA_PREWARM_TK_QKV_FORWARD=*|*NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_GRAPH_PREWARM=*|*NFN_NATIVE_GPT2_LM_HEAD_COOPERATIVE_GRAPH_PREWARM=*)
        ;;
      *)
        CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_DEFER_PREWARM_AFTER_STEPS=1"
        DEFAULT_LONG_RUN_DEFER_PREWARM_APPLIED=1
        if [[ "$LONG_RUN_DEFER_PREWARM_MIN_WARMUP" =~ ^[0-9]+$ &&
              "$WARMUP" =~ ^[0-9]+$ &&
              "$ALLOW_LOW_LONG_RUN_DEFER_PREWARM_DIAGNOSTIC" == "0" &&
              "$LONG_RUN_DEFER_PREWARM_MIN_WARMUP" -gt 0 &&
              "$WARMUP" -lt "$LONG_RUN_DEFER_PREWARM_MIN_WARMUP" ]]; then
          case "${DRY_RUN_PLAN,,}" in
            "1"|"true"|"yes"|"on")
              DEFAULT_LONG_RUN_DEFER_PREWARM_WARMUP_FLOOR_DRY_RUN_WOULD_APPLY=1
              ;;
            *)
              WARMUP="$LONG_RUN_DEFER_PREWARM_MIN_WARMUP"
              DEFAULT_LONG_RUN_DEFER_PREWARM_WARMUP_FLOOR_APPLIED=1
              ;;
          esac
        fi
        if [[ "$LONG_RUN_DEFER_PREWARM_MIN_STEPS" =~ ^[0-9]+$ &&
              "$STEPS" =~ ^[0-9]+$ &&
              "$ALLOW_LOW_LONG_RUN_DEFER_PREWARM_DIAGNOSTIC" == "0" &&
              "$LONG_RUN_DEFER_PREWARM_MIN_STEPS" -gt 0 &&
              "$STEPS" -lt "$LONG_RUN_DEFER_PREWARM_MIN_STEPS" ]]; then
          case "${DRY_RUN_PLAN,,}" in
            "1"|"true"|"yes"|"on")
              DEFAULT_LONG_RUN_DEFER_PREWARM_STEP_FLOOR_DRY_RUN_WOULD_APPLY=1
              ;;
            *)
              STEPS="$LONG_RUN_DEFER_PREWARM_MIN_STEPS"
              DEFAULT_LONG_RUN_DEFER_PREWARM_STEP_FLOOR_APPLIED=1
              ;;
          esac
        fi
        if [[ "$ALLOW_LOW_LONG_RUN_DEFER_PREWARM_DIAGNOSTIC" == "1" &&
              "$LONG_RUN_DEFER_PREWARM_MIN_WARMUP" =~ ^[0-9]+$ &&
              "$WARMUP" =~ ^[0-9]+$ &&
              "$LONG_RUN_DEFER_PREWARM_MIN_WARMUP" -gt 0 &&
              "$WARMUP" -lt "$LONG_RUN_DEFER_PREWARM_MIN_WARMUP" ]]; then
          DEFAULT_LONG_RUN_DEFER_PREWARM_LOW_WARMUP_DIAGNOSTIC=1
        fi
        if [[ "$ALLOW_LOW_LONG_RUN_DEFER_PREWARM_DIAGNOSTIC" == "1" &&
              "$LONG_RUN_DEFER_PREWARM_MIN_STEPS" =~ ^[0-9]+$ &&
              "$STEPS" =~ ^[0-9]+$ &&
              "$LONG_RUN_DEFER_PREWARM_MIN_STEPS" -gt 0 &&
              "$STEPS" -lt "$LONG_RUN_DEFER_PREWARM_MIN_STEPS" ]]; then
          DEFAULT_LONG_RUN_DEFER_PREWARM_LOW_STEP_DIAGNOSTIC=1
        fi
        ;;
    esac
    ;;
  "0"|"false"|"no"|"off"|"")
    ;;
  *)
    echo "Unsupported NFN_SM120_PARITY_DEFAULT_LONG_RUN_DEFER_PREWARM value: $DEFAULT_LONG_RUN_DEFER_PREWARM" >&2
    exit 2
    ;;
esac
if [[ -z "$MAX_CANDIDATE_RATIO_RAW" ]]; then
  case "${DRY_RUN_PLAN,,}" in
    "1"|"true"|"yes"|"on")
      ;;
    *)
      case "${ENFORCE_GATE,,}" in
        "1"|"true"|"yes"|"on")
          case "${STAGE_TIMING,,}" in
            "1"|"true"|"yes"|"on")
              if [[ "$DEFAULT_RATIO_GATE_SKIPPED_FOR_STAGE_TIMING" == "1" ]]; then
                echo "NFN SM120 parity: native stage timing is candidate-only diagnostic instrumentation; skipping default metric-ratio gates. Set NFN_SM120_PARITY_MAX_CANDIDATE_RATIO explicitly to gate this run." >&2
              fi
              ;;
            *)
              gate_stat_prefix=""
              if [[ "$SAMPLES" =~ ^[0-9]+$ && "$SAMPLES" -gt 1 ]]; then
                gate_stat_prefix="median:"
              fi
              if [[ "$DEFAULT_LONG_RUN_DEFER_PREWARM_APPLIED" == "1" ]]; then
                MAX_CANDIDATE_RATIO_RAW="${gate_stat_prefix}train_loop_cuda_event_steady_state_wall_ms_per_step=${DEFAULT_MAX_STEADY_STATE_RATIO}"
              else
                MAX_CANDIDATE_RATIO_RAW="${gate_stat_prefix}train_loop_wall_ms_per_step=${DEFAULT_MAX_TRAIN_LOOP_RATIO}"
              fi
              case "${TRAIN_LOOP_EVENT_TIMING,,}" in
                "1"|"true"|"yes"|"on")
                  if [[ "$DEFAULT_LONG_RUN_DEFER_PREWARM_APPLIED" != "1" ]]; then
                    MAX_CANDIDATE_RATIO_RAW+=" ${gate_stat_prefix}train_loop_cuda_event_steady_state_wall_ms_per_step=${DEFAULT_MAX_STEADY_STATE_RATIO}"
                  fi
                  ;;
              esac
              ;;
          esac
          ;;
      esac
      ;;
  esac
fi
if [[ -z "$MIN_CANDIDATE_RATIO_RAW" ]]; then
  case "${DRY_RUN_PLAN,,}" in
    "1"|"true"|"yes"|"on")
      ;;
    *)
      case "${ENFORCE_GATE,,}" in
        "1"|"true"|"yes"|"on")
          case "${STAGE_TIMING,,}" in
            "1"|"true"|"yes"|"on")
              ;;
            *)
              gate_stat_prefix=""
              if [[ "$SAMPLES" =~ ^[0-9]+$ && "$SAMPLES" -gt 1 ]]; then
                gate_stat_prefix="median:"
              fi
              if [[ "$DEFAULT_LONG_RUN_DEFER_PREWARM_APPLIED" == "1" ]]; then
                MIN_CANDIDATE_RATIO_RAW="${gate_stat_prefix}train_steady_state_tokens_per_second=${DEFAULT_MIN_STEADY_STATE_TOKENS_RATIO}"
              fi
              ;;
          esac
          ;;
      esac
      ;;
  esac
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
case "${DRY_RUN_PLAN,,}" in
  "1"|"true"|"yes"|"on")
    paired_args+=(--dry-run-plan)
    ;;
esac
case "${STAGE_TIMING,,}" in
  "1"|"true"|"yes"|"on")
    if [[ "$DEFAULT_RATIO_GATE_SKIPPED_FOR_STAGE_TIMING" == "1" && -z "$MAX_CANDIDATE_RATIO_RAW" ]]; then
      paired_args+=(--metadata "default_metric_ratio_gate=disabled_for_candidate_only_stage_timing")
    fi
    ;;
esac
if [[ "$DEFAULT_LONG_RUN_DEFER_PREWARM_APPLIED" == "1" ]]; then
  paired_args+=(--metadata "default_long_run_defer_prewarm=applied")
fi
if [[ "$DEFAULT_LONG_RUN_DEFER_PREWARM_WARMUP_FLOOR_APPLIED" == "1" ]]; then
  paired_args+=(--metadata "default_long_run_defer_prewarm_min_warmup_applied=$LONG_RUN_DEFER_PREWARM_MIN_WARMUP")
fi
if [[ "$DEFAULT_LONG_RUN_DEFER_PREWARM_WARMUP_FLOOR_DRY_RUN_WOULD_APPLY" == "1" ]]; then
  paired_args+=(--metadata "default_long_run_defer_prewarm_min_warmup_dry_run_would_apply=$LONG_RUN_DEFER_PREWARM_MIN_WARMUP")
fi
if [[ "$DEFAULT_LONG_RUN_DEFER_PREWARM_STEP_FLOOR_APPLIED" == "1" ]]; then
  paired_args+=(--metadata "default_long_run_defer_prewarm_min_steps_applied=$LONG_RUN_DEFER_PREWARM_MIN_STEPS")
fi
if [[ "$DEFAULT_LONG_RUN_DEFER_PREWARM_STEP_FLOOR_DRY_RUN_WOULD_APPLY" == "1" ]]; then
  paired_args+=(--metadata "default_long_run_defer_prewarm_min_steps_dry_run_would_apply=$LONG_RUN_DEFER_PREWARM_MIN_STEPS")
fi
if [[ "$DEFAULT_LONG_RUN_DEFER_PREWARM_LOW_WARMUP_DIAGNOSTIC" == "1" ]]; then
  paired_args+=(--metadata "default_long_run_defer_prewarm_low_warmup_diagnostic=$WARMUP")
fi
if [[ "$DEFAULT_LONG_RUN_DEFER_PREWARM_LOW_STEP_DIAGNOSTIC" == "1" ]]; then
  paired_args+=(--metadata "default_long_run_defer_prewarm_low_step_diagnostic=$STEPS")
fi
for item in $MAX_CANDIDATE_RATIO_RAW; do
  paired_args+=(--max-candidate-ratio "$item")
done
for item in $MIN_CANDIDATE_RATIO_RAW; do
  paired_args+=(--min-candidate-ratio "$item")
done
case "${TRAIN_LOOP_EVENT_TIMING,,}" in
  "1"|"true"|"yes"|"on")
    paired_args+=(--candidate-env "NFN_NATIVE_GPT_TRAIN_LOOP_EVENT_TIMING=1")
    ;;
esac
case "${ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE,,}" in
  "1"|"true"|"yes"|"on")
    paired_args+=(--allow-stale-selected-gpu-utilization-without-compute-processes)
    ;;
  "0"|"false"|"no"|"off")
    ;;
  *)
    echo "Unsupported NFN_SM120_PARITY_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE value: $ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE" >&2
    exit 2
    ;;
esac
case "${SETUP_EVENT_TIMING,,}" in
  "1"|"true"|"yes"|"on")
    paired_args+=(--candidate-env "NFN_NATIVE_GPT_SETUP_EVENT_TIMING=1")
    ;;
  "0"|"false"|"no"|"off")
    ;;
  *)
    echo "Unsupported NFN_SM120_PARITY_SETUP_EVENT_TIMING value: $SETUP_EVENT_TIMING" >&2
    exit 2
    ;;
esac
case "${ATTENTION_SECTION_TIMING,,}" in
  "1"|"true"|"yes"|"on")
    paired_args+=(--candidate-env "NFN_NATIVE_GPT_ATTENTION_BACKWARD_SECTION_TIMING=1")
    ;;
  "0"|"false"|"no"|"off")
    ;;
  *)
    echo "Unsupported NFN_SM120_PARITY_ATTENTION_SECTION_TIMING value: $ATTENTION_SECTION_TIMING" >&2
    exit 2
    ;;
esac
for item in $CANDIDATE_ENV_RAW; do
  paired_args+=(--candidate-env "$item")
done
if [[ -n "$NFN_SM120_REFERENCE_CUDA_LD_LIBRARY_PATH" ]]; then
  paired_args+=(--baseline-env "LD_LIBRARY_PATH=$NFN_SM120_REFERENCE_CUDA_LD_LIBRARY_PATH")
fi
for item in $REFERENCE_ENV_RAW; do
  paired_args+=(--baseline-env "$item")
done
case "${REQUIRE_NATIVE_LM_HEAD_TRUE_FUSED,,}" in
  "1"|"true"|"yes"|"on")
    paired_args+=(--require-native-lm-head-true-fused)
    ;;
  "0"|"false"|"no"|"off")
    ;;
  *)
    echo "Unsupported NFN_SM120_PARITY_REQUIRE_NATIVE_LM_HEAD_TRUE_FUSED value: $REQUIRE_NATIVE_LM_HEAD_TRUE_FUSED" >&2
    exit 2
    ;;
esac

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

ensure_default_native_gpt_trainer_current() {
  if [[ -n "$NFN_NATIVE_GPT_TRAIN_BIN_EXPLICIT" ]]; then
    return 0
  fi
  case "$(basename "$NFN_NATIVE_GPT_TRAIN_BIN")" in
    nfn_gpt_native_train_linked)
      local rebuild_linked=0
      if [[ ! -x "$NFN_NATIVE_GPT_TRAIN_BIN" ]]; then
        rebuild_linked=1
      elif native_gpt_source_newer_than "$NFN_NATIVE_GPT_TRAIN_BIN"; then
        rebuild_linked=1
      elif tile_ops_source_newer_than "$NFN_NATIVE_GPT_TRAIN_BIN"; then
        rebuild_linked=1
      fi
      if [[ "$rebuild_linked" == "1" ]]; then
        bash "$ROOT_DIR/tools/build_native_gpt_cli_linked.sh" "$NFN_NATIVE_GPT_TRAIN_BIN" >&2
      fi
      ;;
    nfn_gpt_native_train)
      local rebuild_dynamic=0
      if [[ ! -x "$NFN_NATIVE_GPT_TRAIN_BIN" ]]; then
        rebuild_dynamic=1
      elif native_gpt_source_newer_than "$NFN_NATIVE_GPT_TRAIN_BIN"; then
        rebuild_dynamic=1
      fi
      if [[ "$rebuild_dynamic" == "1" ]]; then
        bash "$ROOT_DIR/tools/build_native_gpt_cli.sh" "$NFN_NATIVE_GPT_TRAIN_BIN" >&2
      fi
      ;;
  esac
}

case "${DRY_RUN_PLAN,,}" in
  "1"|"true"|"yes"|"on")
    ;;
  *)
    ensure_default_native_gpt_trainer_current
    ;;
esac

if [[ ! -x "$LLM_KITTENS_TRAIN_BIN" ]]; then
  echo "llm.kittens train_gpt2cu is not executable: $LLM_KITTENS_TRAIN_BIN" >&2
  exit 2
fi
if [[ ! -x "$NFN_NATIVE_GPT_TRAIN_BIN" ]]; then
  echo "NeuralFn native GPT trainer is not executable: $NFN_NATIVE_GPT_TRAIN_BIN" >&2
  exit 2
fi
NFN_NATIVE_TILE_OPS_ARG="$NFN_NATIVE_TILE_OPS_LIB"
if [[ -z "$NFN_NATIVE_TILE_OPS_LIB_EXPLICIT" && "$(basename "$NFN_NATIVE_GPT_TRAIN_BIN")" == "nfn_gpt_native_train_linked" ]]; then
  NFN_NATIVE_TILE_OPS_ARG="linked"
fi
if [[ "$NFN_NATIVE_TILE_OPS_ARG" != "linked" && ! -f "$NFN_NATIVE_TILE_OPS_ARG" ]]; then
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
    -s "$SAMPLE_EVERY" \
    -g "$GENERATE_TOKENS" \
    -h 0 \
    -b 64 \
    -t 1024 \
    -d "$TRAIN_BATCH_TOKENS" \
    -r 0 \
    -z 1 \
    -c 0.1 \
    -l 0.0006 \
    -q 0.0 \
    -u 60 \
    -n "$CHECKPOINT_EVERY" \
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
    --train-batch-tokens "$TRAIN_BATCH_TOKENS" \
    --eval-every-steps 0 \
    --train-loss-every-steps "$TRAIN_LOSS_EVERY" \
    --native-cuda-sample-every "$SAMPLE_EVERY" \
    --native-cuda-generate-tokens "$GENERATE_TOKENS" \
    --native-cuda-checkpoint-every "$CHECKPOINT_EVERY" \
    --native-cuda-activation "$ACTIVATION" \
    --no-checkpoint \
    --tile-ops-lib "$NFN_NATIVE_TILE_OPS_ARG"
)"

cd "$ROOT_DIR"
set +e
python tools/paired_kernel_speed.py \
  --baseline "$baseline_cmd" \
  --candidate "$candidate_cmd" \
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
paired_status=$?
set -e

if [[ "$paired_status" != "0" && -f "$JSON_OUT" ]]; then
  python - "$JSON_OUT" >&2 <<'PY'
import glob
import json
import os
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
except Exception as exc:  # pragma: no cover - best-effort diagnostic.
    print(f"Unable to read NeuralFn parity JSON for diagnostics: {exc}")
    raise SystemExit(0)

values = data.get("candidate_native_metric_values") or {}
metrics = data.get("candidate_native_metrics") or {}
ratios = data.get("candidate_over_baseline_native_metrics") or {}
gates = data.get("metric_ratio_gates") or {}

def scalar_value(name):
    value = values.get(name)
    if isinstance(value, list) and value:
        return value[0]
    return value

def metric_mean(name):
    value = metrics.get(name)
    if isinstance(value, dict):
        return value.get("mean")
    return None

def ratio_mean(name):
    value = ratios.get(name)
    if isinstance(value, dict):
        return value.get("mean")
    return None

def bool_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None

def format_ms(value):
    try:
        return f"{float(value):.3f} ms"
    except (TypeError, ValueError):
        return "n/a"

def format_gib(value):
    try:
        return f"{float(value) / (1024.0 ** 3):.3f} GiB"
    except (TypeError, ValueError):
        return "n/a"

def candidate_profile_path():
    profile_dir = data.get("append_native_profile_json_dir")
    if not isinstance(profile_dir, str) or not profile_dir:
        return None
    matches = glob.glob(os.path.join(profile_dir, "candidate_*.json"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)

def print_profile_summary():
    sidecar = candidate_profile_path()
    if sidecar is None:
        return
    try:
        with open(sidecar, "r", encoding="utf-8") as handle:
            profile = json.load(handle)
    except Exception as exc:  # pragma: no cover - best-effort diagnostic.
        print(f"  Candidate native profile summary unavailable: {exc}")
        return
    print("  Candidate native profile sidecar:")
    print(f"    {sidecar}")
    timing = profile.get("timing") if isinstance(profile, dict) else {}
    setup = timing.get("setup_timing") if isinstance(timing, dict) else None
    stage = timing.get("stage_timing") if isinstance(timing, dict) else None
    if isinstance(setup, list) and setup:
        top_setup = sorted(
            (item for item in setup if isinstance(item, dict)),
            key=lambda item: float(item.get("total_ms") or 0.0),
            reverse=True,
        )[:5]
        if top_setup:
            print("  Top candidate setup timings:")
            for item in top_setup:
                print(
                    "    "
                    f"{item.get('name', 'unknown')}: "
                    f"{format_ms(item.get('total_ms'))} "
                    f"(count={item.get('count', 'n/a')})"
                )
    if isinstance(stage, list) and stage:
        top_stage = sorted(
            (item for item in stage if isinstance(item, dict)),
            key=lambda item: float(item.get("total_ms") or 0.0),
            reverse=True,
        )[:8]
        if top_stage:
            print("  Top candidate stage timings:")
            for item in top_stage:
                print(
                    "    "
                    f"{item.get('name', 'unknown')}: "
                    f"{format_ms(item.get('total_ms'))} "
                    f"(count={item.get('count', 'n/a')})"
                )
    for label, key in (
        ("float arena", "float_arena_request_stats"),
        ("uint16 arena", "uint16_arena_request_stats"),
    ):
        stats = profile.get(key) if isinstance(profile, dict) else None
        if not isinstance(stats, dict):
            continue
        print(
            f"  Candidate {label}: "
            f"allocated={format_gib(stats.get('total_allocated_bytes'))}, "
            f"requested={format_gib(stats.get('total_requested_bytes'))}, "
            f"requests={stats.get('request_count', 'n/a')}"
        )
        families = stats.get("top_families")
        if isinstance(families, list):
            for family in families[:3]:
                if not isinstance(family, dict):
                    continue
                print(
                    "    "
                    f"{family.get('family', 'unknown')}: "
                    f"{format_gib(family.get('bytes'))} "
                    f"(requests={family.get('request_count', 'n/a')})"
                )

if gates.get("passed") is False:
    strategy = scalar_value("lm_head_cooperative_backward_strategy")
    fused_available = scalar_value("lm_head_cooperative_backward_fused_kernel_available")
    graph_enabled = scalar_value("lm_head_cooperative_backward_cuda_graph_enabled")
    if (
        strategy == "diagnostic-cuda-graph-ce-fork-join-dhidden-dweight-not-single-kernel"
        and bool_value(fused_available) is False
        and bool_value(graph_enabled) is True
    ):
        wall_ratio = ratio_mean("train_loop_wall_ms_per_step")
        steady_ratio = ratio_mean("train_loop_cuda_event_steady_state_wall_ms_per_step")
        replays = metric_mean("lm_head_fused_graph_replay_count")
        captures = metric_mean("lm_head_fused_graph_capture_success_count")
        print("NeuralFn SM120 parity diagnostic:")
        print("  Native Tile training is active, but LM-head backward is still the diagnostic CUDA Graph wrapper.")
        print("  The reference-aligned target is fused CE/dlogits with separate logits, dHidden, and dWeight matmul stages.")
        print("  The strict true-fused single-kernel path remains an experimental gate, not the llm.kittens parity baseline.")
        if wall_ratio is not None:
            print(f"  train_loop_wall_ms_per_step ratio: {wall_ratio:.6f}")
        if steady_ratio is not None:
            print(f"  train_loop_cuda_event_steady_state_wall_ms_per_step ratio: {steady_ratio:.6f}")
        if replays is not None or captures is not None:
            print(
                "  LM-head graph counters: "
                f"replay_mean={replays if replays is not None else 'n/a'}, "
                f"capture_success_mean={captures if captures is not None else 'n/a'}"
            )
        print("  Next implementation target: optimize the fused CE/dlogits plus separate logits, dHidden,")
        print("  and dWeight stages under same-script candidate/reference gates before promoting strict")
        print("  nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16 experiments.")
        print_profile_summary()
PY
fi

exit "$paired_status"
