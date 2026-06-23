#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLM_KITTENS_ROOT="${LLM_KITTENS_ROOT:-/mnt/disk2/dev/open-source/llm.kittens}"
LLM_KITTENS_TRAIN_BIN="${LLM_KITTENS_TRAIN_BIN:-$LLM_KITTENS_ROOT/train_gpt2cu}"
LLM_KITTENS_TINYSTORIES_DIR="${LLM_KITTENS_TINYSTORIES_DIR:-$LLM_KITTENS_ROOT/dev/data/tinystories}"
if [[ -z "${NFN_NATIVE_GPT_TRAIN_BIN-}" && -x "$ROOT_DIR/build/nfn_gpt_native_train_linked" ]]; then
  NFN_NATIVE_GPT_TRAIN_BIN="$ROOT_DIR/build/nfn_gpt_native_train_linked"
else
  NFN_NATIVE_GPT_TRAIN_BIN="${NFN_NATIVE_GPT_TRAIN_BIN:-$ROOT_DIR/build/nfn_gpt_native_train}"
fi
NFN_NATIVE_TILE_OPS_LIB_EXPLICIT="${NFN_NATIVE_TILE_OPS_LIB+x}"
NFN_NATIVE_TILE_OPS_LIB="${NFN_NATIVE_TILE_OPS_LIB:-$ROOT_DIR/build/libnfn_native_train_tile_ops.so}"

STEPS="${NFN_SM120_PARITY_STEPS:-${NFN_SM120_STEPS:-10}}"
SAMPLES="${NFN_SM120_PARITY_SAMPLES:-${NFN_SM120_SAMPLES:-1}}"
WARMUP="${NFN_SM120_PARITY_WARMUP:-${NFN_SM120_WARMUP:-0}}"
CUDA_VISIBLE_DEVICES_VALUE="${NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES:-${NFN_SM120_CUDA_VISIBLE_DEVICES:-auto}}"
CUDA_DEVICE_MAX_CONNECTIONS_VALUE="${NFN_SM120_PARITY_CUDA_DEVICE_MAX_CONNECTIONS:-${NFN_SM120_CUDA_DEVICE_MAX_CONNECTIONS:-1}}"
MAX_GPU_UTILIZATION="${NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT:-${NFN_SM120_MAX_GPU_UTILIZATION_PCT:-15}}"
SELECTED_GPU_UTILIZATION_RETRIES="${NFN_SM120_PARITY_SELECTED_GPU_UTILIZATION_RETRIES:-${NFN_SM120_SELECTED_GPU_UTILIZATION_RETRIES:-3}}"
SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS="${NFN_SM120_PARITY_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS:-${NFN_SM120_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS:-0.25}}"
COMMAND_TIMEOUT_SECONDS="${NFN_SM120_PARITY_COMMAND_TIMEOUT_SECONDS:-${NFN_SM120_COMMAND_TIMEOUT_SECONDS:-300}}"
ACTIVATION="${NFN_SM120_PARITY_ACTIVATION:-${NFN_SM120_ACTIVATION:-gelu}}"
SAMPLE_EVERY="${NFN_SM120_PARITY_SAMPLE_EVERY:-${NFN_SM120_SAMPLE_EVERY:-0}}"
CHECKPOINT_EVERY="${NFN_SM120_PARITY_CHECKPOINT_EVERY:-${NFN_SM120_CHECKPOINT_EVERY:-0}}"
GENERATE_TOKENS="${NFN_SM120_PARITY_GENERATE_TOKENS:-${NFN_SM120_GENERATE_TOKENS:-144}}"
TRAIN_LOSS_EVERY="${NFN_SM120_PARITY_TRAIN_LOSS_EVERY_STEPS:-${NFN_SM120_TRAIN_LOSS_EVERY_STEPS:-0}}"
TRAIN_LOOP_EVENT_TIMING="${NFN_SM120_PARITY_TRAIN_LOOP_EVENT_TIMING:-${NFN_SM120_TRAIN_LOOP_EVENT_TIMING:-1}}"
SETUP_EVENT_TIMING="${NFN_SM120_PARITY_SETUP_EVENT_TIMING:-${NFN_SM120_SETUP_EVENT_TIMING:-0}}"
ATTENTION_SECTION_TIMING="${NFN_SM120_PARITY_ATTENTION_SECTION_TIMING:-${NFN_SM120_ATTENTION_SECTION_TIMING:-0}}"
CANDIDATE_ENV_RAW="${NFN_SM120_PARITY_CANDIDATE_ENV:-${NFN_SM120_CANDIDATE_ENV:-}}"
JSON_OUT="${NFN_SM120_PARITY_JSON_OUT:-${NFN_SM120_JSON_OUT:-/tmp/nfn_sm120_parity_${STEPS}step.json}}"
PROFILE_DIR_RAW="${NFN_SM120_PARITY_PROFILE_DIR:-${NFN_SM120_PROFILE_DIR:-/tmp/nfn_sm120_parity_profiles_${STEPS}step}}"
STAGE_TIMING="${NFN_SM120_NATIVE_STAGE_TIMING:-${NFN_SM120_NATIVE_PARITY_STAGE_TIMING:-${NFN_SM120_PARITY_STAGE_TIMING:-${NFN_SM120_STAGE_TIMING:-0}}}}"
REFERENCE_OUTPUT_DIR="${NFN_SM120_PARITY_REFERENCE_OUTPUT_DIR:-${NFN_SM120_REFERENCE_OUTPUT_DIR:-/tmp/nfn_llmk_sm120_parity}}"
DRY_RUN_PLAN="${NFN_SM120_PARITY_DRY_RUN_PLAN:-${NFN_SM120_DRY_RUN_PLAN:-0}}"
MAX_CANDIDATE_RATIO_RAW="${NFN_SM120_PARITY_MAX_CANDIDATE_RATIO:-${NFN_SM120_MAX_CANDIDATE_RATIO:-}}"
ENFORCE_GATE="${NFN_SM120_PARITY_ENFORCE_GATE:-${NFN_SM120_ENFORCE_PARITY_GATE:-0}}"
if [[ -z "$MAX_CANDIDATE_RATIO_RAW" ]]; then
  case "${DRY_RUN_PLAN,,}" in
    "1"|"true"|"yes"|"on")
      ;;
    *)
      case "${ENFORCE_GATE,,}" in
        "1"|"true"|"yes"|"on")
          MAX_CANDIDATE_RATIO_RAW="train_loop_wall_ms_per_step=1.000"
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
for item in $MAX_CANDIDATE_RATIO_RAW; do
  paired_args+=(--max-candidate-ratio "$item")
done
case "${TRAIN_LOOP_EVENT_TIMING,,}" in
  "1"|"true"|"yes"|"on")
    paired_args+=(--candidate-env "NFN_NATIVE_GPT_TRAIN_LOOP_EVENT_TIMING=1")
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
    -d 524288 \
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
    --train-batch-tokens 524288 \
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
