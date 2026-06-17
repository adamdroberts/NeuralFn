#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NFN_NATIVE_GPT_TRAIN_BIN="${NFN_NATIVE_GPT_TRAIN_BIN:-$ROOT_DIR/build/nfn_gpt_native_train}"
NFN_NATIVE_TILE_OPS_LIB="${NFN_NATIVE_TILE_OPS_LIB:-$ROOT_DIR/build/libnfn_native_train_tile_ops.so}"
NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB="${NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB:-$NFN_NATIVE_TILE_OPS_LIB}"

STEPS="${NFN_SM120_NATIVE_STEPS:-10}"
SAMPLES="${NFN_SM120_NATIVE_SAMPLES:-3}"
WARMUP="${NFN_SM120_NATIVE_WARMUP:-1}"
TRAIN_BATCH_TOKENS="${NFN_SM120_NATIVE_TRAIN_BATCH_TOKENS:-524288}"
CUDA_VISIBLE_DEVICES_VALUE="${NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES:-auto}"
CUDA_DEVICE_MAX_CONNECTIONS_VALUE="${NFN_SM120_NATIVE_CUDA_DEVICE_MAX_CONNECTIONS:-1}"
MAX_GPU_UTILIZATION="${NFN_SM120_NATIVE_MAX_GPU_UTILIZATION_PCT:-15}"
COMMAND_TIMEOUT_SECONDS="${NFN_SM120_NATIVE_COMMAND_TIMEOUT_SECONDS:-300}"
SAMPLE_EVERY="${NFN_SM120_NATIVE_SAMPLE_EVERY:-0}"
CHECKPOINT_EVERY="${NFN_SM120_NATIVE_CHECKPOINT_EVERY:-0}"
GENERATE_TOKENS="${NFN_SM120_NATIVE_GENERATE_TOKENS:-16}"
JSON_OUT="${NFN_SM120_NATIVE_JSON_OUT:-/tmp/nfn_sm120_native_candidate_${STEPS}step.json}"
PROFILE_DIR_RAW="${NFN_SM120_NATIVE_PROFILE_DIR:-/tmp/nfn_sm120_native_candidate_profiles_${STEPS}step}"
STAGE_TIMING="${NFN_SM120_NATIVE_STAGE_TIMING:-0}"
BASELINE_ENV_RAW="${NFN_SM120_NATIVE_BASELINE_ENV:-}"
CANDIDATE_ENV_RAW="${NFN_SM120_NATIVE_CANDIDATE_ENV:-}"
COMMON_EXTRA_ARGS_RAW="${NFN_SM120_NATIVE_EXTRA_ARGS:-}"
BASELINE_EXTRA_ARGS_RAW="${NFN_SM120_NATIVE_BASELINE_EXTRA_ARGS:-}"
CANDIDATE_EXTRA_ARGS_RAW="${NFN_SM120_NATIVE_CANDIDATE_EXTRA_ARGS:-}"
TEMPLATE_NAME="${NFN_SM120_NATIVE_TEMPLATE_NAME:-}"
GRAPH_FILE="${NFN_SM120_NATIVE_GRAPH_FILE:-}"

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
for item in $BASELINE_ENV_RAW; do
  paired_args+=(--baseline-env "$item")
done
for item in $CANDIDATE_ENV_RAW; do
  paired_args+=(--candidate-env "$item")
done

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
