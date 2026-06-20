#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NFN_NATIVE_GPT_TRAIN_BIN="${NFN_NATIVE_GPT_TRAIN_BIN:-$ROOT_DIR/build/nfn_gpt_native_train}"
NFN_NATIVE_TILE_OPS_LIB="${NFN_NATIVE_TILE_OPS_LIB:-$ROOT_DIR/build/libnfn_native_train_tile_ops.so}"
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

STEPS="$(env_or_alias3 NFN_SM120_NATIVE_STEPS NFN_SM120_CANDIDATE_STEPS NFN_SM120_PARITY_STEPS 10)"
SAMPLES="$(env_or_alias3 NFN_SM120_NATIVE_SAMPLES NFN_SM120_CANDIDATE_SAMPLES NFN_SM120_PARITY_SAMPLES 3)"
WARMUP="$(env_or_alias3 NFN_SM120_NATIVE_WARMUP NFN_SM120_CANDIDATE_WARMUP NFN_SM120_PARITY_WARMUP 1)"
TRAIN_BATCH_TOKENS="$(env_or_alias3 NFN_SM120_NATIVE_TRAIN_BATCH_TOKENS NFN_SM120_CANDIDATE_TRAIN_BATCH_TOKENS NFN_SM120_PARITY_TRAIN_BATCH_TOKENS 524288)"
CUDA_VISIBLE_DEVICES_VALUE="$(env_or_alias3 NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES NFN_SM120_CANDIDATE_CUDA_VISIBLE_DEVICES NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES auto)"
CUDA_DEVICE_MAX_CONNECTIONS_VALUE="$(env_or_alias3 NFN_SM120_NATIVE_CUDA_DEVICE_MAX_CONNECTIONS NFN_SM120_CANDIDATE_CUDA_DEVICE_MAX_CONNECTIONS NFN_SM120_PARITY_CUDA_DEVICE_MAX_CONNECTIONS 1)"
MAX_GPU_UTILIZATION="$(env_or_alias3 NFN_SM120_NATIVE_MAX_GPU_UTILIZATION_PCT NFN_SM120_CANDIDATE_MAX_GPU_UTILIZATION_PCT NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT 15)"
COMMAND_TIMEOUT_SECONDS="$(env_or_alias3 NFN_SM120_NATIVE_COMMAND_TIMEOUT_SECONDS NFN_SM120_CANDIDATE_COMMAND_TIMEOUT_SECONDS NFN_SM120_PARITY_COMMAND_TIMEOUT_SECONDS 300)"
SAMPLE_EVERY="$(env_or_alias3 NFN_SM120_NATIVE_SAMPLE_EVERY NFN_SM120_CANDIDATE_SAMPLE_EVERY NFN_SM120_PARITY_SAMPLE_EVERY 0)"
CHECKPOINT_EVERY="$(env_or_alias3 NFN_SM120_NATIVE_CHECKPOINT_EVERY NFN_SM120_CANDIDATE_CHECKPOINT_EVERY NFN_SM120_PARITY_CHECKPOINT_EVERY 0)"
GENERATE_TOKENS="$(env_or_alias3 NFN_SM120_NATIVE_GENERATE_TOKENS NFN_SM120_CANDIDATE_GENERATE_TOKENS NFN_SM120_PARITY_GENERATE_TOKENS 16)"
JSON_OUT="$(env_or_alias3 NFN_SM120_NATIVE_JSON_OUT NFN_SM120_CANDIDATE_JSON_OUT NFN_SM120_PARITY_JSON_OUT "/tmp/nfn_sm120_native_candidate_${STEPS}step.json")"
PROFILE_DIR_RAW="$(env_or_alias3 NFN_SM120_NATIVE_PROFILE_DIR NFN_SM120_CANDIDATE_PROFILE_DIR NFN_SM120_PARITY_PROFILE_DIR "/tmp/nfn_sm120_native_candidate_profiles_${STEPS}step")"
STAGE_TIMING="$(env_or_alias3 NFN_SM120_NATIVE_STAGE_TIMING NFN_SM120_CANDIDATE_STAGE_TIMING NFN_SM120_PARITY_STAGE_TIMING 0)"
LINEAR_SHAPE_STATS="$(env_or_alias3 NFN_SM120_NATIVE_LINEAR_SHAPE_STATS NFN_SM120_CANDIDATE_LINEAR_SHAPE_STATS NFN_SM120_PARITY_LINEAR_SHAPE_STATS 0)"
STARTUP_ONLY="$(env_or_alias3 NFN_SM120_NATIVE_STARTUP_ONLY NFN_SM120_CANDIDATE_STARTUP_ONLY NFN_SM120_PARITY_STARTUP_ONLY 0)"
COMMON_ENV_RAW="$(env_or_alias3 NFN_SM120_NATIVE_ENV NFN_SM120_COMMON_ENV NFN_SM120_PARITY_ENV "")"
BASELINE_ENV_RAW="$(env_or_alias NFN_SM120_NATIVE_BASELINE_ENV NFN_SM120_CANDIDATE_BASELINE_ENV "")"
CANDIDATE_ENV_RAW="$(env_or_alias NFN_SM120_NATIVE_CANDIDATE_ENV NFN_SM120_CANDIDATE_ENV "")"
COMMON_EXTRA_ARGS_RAW="$(env_or_alias3 NFN_SM120_NATIVE_EXTRA_ARGS NFN_SM120_COMMON_EXTRA_ARGS NFN_SM120_PARITY_EXTRA_ARGS "")"
BASELINE_EXTRA_ARGS_RAW="$(env_or_alias NFN_SM120_NATIVE_BASELINE_EXTRA_ARGS NFN_SM120_CANDIDATE_BASELINE_EXTRA_ARGS "")"
CANDIDATE_EXTRA_ARGS_RAW="$(env_or_alias NFN_SM120_NATIVE_CANDIDATE_EXTRA_ARGS NFN_SM120_CANDIDATE_EXTRA_ARGS "")"
if [[ -z "$CANDIDATE_EXTRA_ARGS_RAW" && -n "${NFN_SM120_NATIVE_CANDIDATE_ARGS-}" ]]; then
  CANDIDATE_EXTRA_ARGS_RAW="$NFN_SM120_NATIVE_CANDIDATE_ARGS"
fi
TEMPLATE_NAME="$(env_or_alias3 NFN_SM120_NATIVE_TEMPLATE_NAME NFN_SM120_CANDIDATE_TEMPLATE_NAME NFN_SM120_PARITY_TEMPLATE_NAME "")"
GRAPH_FILE="$(env_or_alias3 NFN_SM120_NATIVE_GRAPH_FILE NFN_SM120_CANDIDATE_GRAPH_FILE NFN_SM120_PARITY_GRAPH_FILE "")"
DRY_RUN_PLAN="$(env_or_alias3 NFN_SM120_NATIVE_DRY_RUN_PLAN NFN_SM120_CANDIDATE_DRY_RUN_PLAN NFN_SM120_PARITY_DRY_RUN_PLAN 0)"
MAX_CANDIDATE_RATIO_RAW="$(env_or_alias NFN_SM120_NATIVE_MAX_CANDIDATE_RATIO NFN_SM120_CANDIDATE_MAX_CANDIDATE_RATIO "")"
if [[ -z "$MAX_CANDIDATE_RATIO_RAW" ]]; then
  has_candidate_change=0
  if [[ "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB" != "$NFN_NATIVE_TILE_OPS_LIB" ||
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
  if [[ "$has_candidate_change" == "1" ]]; then
    case "${STARTUP_ONLY,,}" in
      "1"|"true"|"yes"|"on")
        MAX_CANDIDATE_RATIO_RAW="setup_wall_ms=1.000"
        startup_candidate_text="$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB $CANDIDATE_ENV_RAW $CANDIDATE_EXTRA_ARGS_RAW"
        case "$startup_candidate_text" in
          *TOKEN_WEIGHT*|*token_weight*)
            MAX_CANDIDATE_RATIO_RAW+=" setup.token_weight_init.total_ms=1.000"
            ;;
        esac
        ;;
      *)
        MAX_CANDIDATE_RATIO_RAW="train_loop_wall_ms_per_step=1.000"
        case "${STAGE_TIMING,,}" in
          "1"|"true"|"yes"|"on")
            MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.total_ms=1.000"
            MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.total_ms=1.000"
            MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.mlp_proj.total_ms=1.000"
            candidate_gate_text="$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB $CANDIDATE_ENV_RAW $CANDIDATE_EXTRA_ARGS_RAW"
            case "$candidate_gate_text" in
              *CE_BF16*|*ce_bf16*)
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
              *LM_HEAD_PIPELINE_CHUNKS*|*lm_head_pipeline_chunks*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.pipeline_queue.total_ms=1.000"
                MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.pipeline_final_wait.total_ms=1.000"
                ;;
            esac
            ;;
        esac
        ;;
    esac
  fi
fi

if [[ ! -x "$NFN_NATIVE_GPT_TRAIN_BIN" ]]; then
  echo "NeuralFn native GPT trainer is not executable: $NFN_NATIVE_GPT_TRAIN_BIN" >&2
  exit 2
fi
if [[ ! -f "$NFN_NATIVE_TILE_OPS_LIB" ]]; then
  echo "Baseline NeuralFn Tile ops library is missing: $NFN_NATIVE_TILE_OPS_LIB" >&2
  exit 2
fi
if [[ ! -f "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB" ]]; then
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
  "$NFN_NATIVE_GPT_TRAIN_BIN"
  --backend tile-cuda
  --tinystories
  --max-steps "$STEPS"
  --train-batch-tokens "$TRAIN_BATCH_TOKENS"
  --eval-every-steps 0
  --native-cuda-sample-every "$SAMPLE_EVERY"
  --native-cuda-generate-tokens "$GENERATE_TOKENS"
  --native-cuda-checkpoint-every "$CHECKPOINT_EVERY"
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

baseline_args=("${common_args[@]}" --tile-ops-lib "$NFN_NATIVE_TILE_OPS_LIB")
candidate_args=("${common_args[@]}" --tile-ops-lib "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB")
append_split_args baseline_args "$BASELINE_EXTRA_ARGS_RAW"
append_split_args candidate_args "$CANDIDATE_EXTRA_ARGS_RAW"

baseline_cmd="$(join_command "${baseline_args[@]}")"
candidate_cmd="$(join_command "${candidate_args[@]}")"

profile_args=()
case "${PROFILE_DIR_RAW,,}" in
  ""|"0"|"false"|"no"|"none"|"off")
    ;;
  *)
    profile_args=(--append-native-profile-json-dir "$PROFILE_DIR_RAW")
    case "${STAGE_TIMING,,}" in
      "1"|"true"|"yes"|"on")
        profile_args+=(--native-stage-timing)
        export NFN_NATIVE_GPT_STAGE_TIMING_MAX_EVENTS="${NFN_NATIVE_GPT_STAGE_TIMING_MAX_EVENTS:-80000}"
        ;;
    esac
    ;;
esac

paired_args=()
common_env_items=()
baseline_env_items=()
candidate_env_items=()
append_env_overrides common_env_items "$COMMON_ENV_RAW"
append_env_overrides baseline_env_items "$BASELINE_ENV_RAW"
append_env_overrides candidate_env_items "$CANDIDATE_ENV_RAW"
for item in "${common_env_items[@]}"; do
  paired_args+=(--baseline-env "$item")
  paired_args+=(--candidate-env "$item")
done
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
case "${DRY_RUN_PLAN,,}" in
  "1"|"true"|"yes"|"on")
    paired_args+=(--dry-run-plan)
    ;;
esac

cd "$ROOT_DIR"
python tools/paired_kernel_speed.py \
  --baseline "$baseline_cmd" \
  --candidate "$candidate_cmd" \
  --samples "$SAMPLES" \
  --warmup "$WARMUP" \
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES_VALUE" \
  --cuda-device-max-connections "$CUDA_DEVICE_MAX_CONNECTIONS_VALUE" \
  --require-idle-selected-gpu \
  --max-selected-gpu-utilization-pct "$MAX_GPU_UTILIZATION" \
  --command-timeout-seconds "$COMMAND_TIMEOUT_SECONDS" \
  "${profile_args[@]}" \
  "${paired_args[@]}" \
  --json-out "$JSON_OUT"
