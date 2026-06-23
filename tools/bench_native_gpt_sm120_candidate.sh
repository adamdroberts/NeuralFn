#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${NFN_NATIVE_GPT_TRAIN_BIN-}" && -x "$ROOT_DIR/build/nfn_gpt_native_train_linked" ]]; then
  NFN_NATIVE_GPT_TRAIN_BIN="$ROOT_DIR/build/nfn_gpt_native_train_linked"
else
  NFN_NATIVE_GPT_TRAIN_BIN="${NFN_NATIVE_GPT_TRAIN_BIN:-$ROOT_DIR/build/nfn_gpt_native_train}"
fi
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
CUDA_VISIBLE_DEVICES_VALUE="$(env_or_alias5 NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES NFN_SM120_NATIVE_CANDIDATE_CUDA_VISIBLE_DEVICES NFN_SM120_CANDIDATE_CUDA_VISIBLE_DEVICES NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES NFN_SM120_CUDA_VISIBLE_DEVICES auto)"
CUDA_DEVICE_MAX_CONNECTIONS_VALUE="$(env_or_alias5 NFN_SM120_NATIVE_CUDA_DEVICE_MAX_CONNECTIONS NFN_SM120_NATIVE_CANDIDATE_CUDA_DEVICE_MAX_CONNECTIONS NFN_SM120_CANDIDATE_CUDA_DEVICE_MAX_CONNECTIONS NFN_SM120_PARITY_CUDA_DEVICE_MAX_CONNECTIONS NFN_SM120_CUDA_DEVICE_MAX_CONNECTIONS 1)"
MAX_GPU_UTILIZATION="$(env_or_alias5 NFN_SM120_NATIVE_MAX_GPU_UTILIZATION_PCT NFN_SM120_NATIVE_CANDIDATE_MAX_GPU_UTILIZATION_PCT NFN_SM120_CANDIDATE_MAX_GPU_UTILIZATION_PCT NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT NFN_SM120_MAX_GPU_UTILIZATION_PCT 15)"
SELECTED_GPU_UTILIZATION_RETRIES="$(env_or_alias5 NFN_SM120_NATIVE_SELECTED_GPU_UTILIZATION_RETRIES NFN_SM120_NATIVE_CANDIDATE_SELECTED_GPU_UTILIZATION_RETRIES NFN_SM120_CANDIDATE_SELECTED_GPU_UTILIZATION_RETRIES NFN_SM120_PARITY_SELECTED_GPU_UTILIZATION_RETRIES NFN_SM120_SELECTED_GPU_UTILIZATION_RETRIES 3)"
SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS="$(env_or_alias5 NFN_SM120_NATIVE_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS NFN_SM120_NATIVE_CANDIDATE_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS NFN_SM120_CANDIDATE_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS NFN_SM120_PARITY_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS NFN_SM120_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS 0.25)"
ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE="$(env_or_alias5 NFN_SM120_NATIVE_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE NFN_SM120_NATIVE_CANDIDATE_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE NFN_SM120_CANDIDATE_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE NFN_SM120_PARITY_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE NFN_SM120_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE 1)"
COMMAND_TIMEOUT_SECONDS="$(env_or_alias5 NFN_SM120_NATIVE_COMMAND_TIMEOUT_SECONDS NFN_SM120_NATIVE_CANDIDATE_COMMAND_TIMEOUT_SECONDS NFN_SM120_CANDIDATE_COMMAND_TIMEOUT_SECONDS NFN_SM120_PARITY_COMMAND_TIMEOUT_SECONDS NFN_SM120_COMMAND_TIMEOUT_SECONDS 300)"
SAMPLE_EVERY="$(env_or_alias5 NFN_SM120_NATIVE_SAMPLE_EVERY NFN_SM120_NATIVE_CANDIDATE_SAMPLE_EVERY NFN_SM120_CANDIDATE_SAMPLE_EVERY NFN_SM120_PARITY_SAMPLE_EVERY NFN_SM120_SAMPLE_EVERY 0)"
CHECKPOINT_EVERY="$(env_or_alias5 NFN_SM120_NATIVE_CHECKPOINT_EVERY NFN_SM120_NATIVE_CANDIDATE_CHECKPOINT_EVERY NFN_SM120_CANDIDATE_CHECKPOINT_EVERY NFN_SM120_PARITY_CHECKPOINT_EVERY NFN_SM120_CHECKPOINT_EVERY 0)"
GENERATE_TOKENS="$(env_or_alias5 NFN_SM120_NATIVE_GENERATE_TOKENS NFN_SM120_NATIVE_CANDIDATE_GENERATE_TOKENS NFN_SM120_CANDIDATE_GENERATE_TOKENS NFN_SM120_PARITY_GENERATE_TOKENS NFN_SM120_GENERATE_TOKENS 16)"
JSON_OUT="$(env_or_alias5 NFN_SM120_NATIVE_JSON_OUT NFN_SM120_NATIVE_CANDIDATE_JSON_OUT NFN_SM120_CANDIDATE_JSON_OUT NFN_SM120_PARITY_JSON_OUT NFN_SM120_JSON_OUT "/tmp/nfn_sm120_native_candidate_${STEPS}step.json")"
PROFILE_DIR_RAW="$(env_or_alias5 NFN_SM120_NATIVE_PROFILE_DIR NFN_SM120_NATIVE_CANDIDATE_PROFILE_DIR NFN_SM120_CANDIDATE_PROFILE_DIR NFN_SM120_PARITY_PROFILE_DIR NFN_SM120_PROFILE_DIR "/tmp/nfn_sm120_native_candidate_profiles_${STEPS}step")"
STAGE_TIMING="$(env_or_alias5 NFN_SM120_NATIVE_STAGE_TIMING NFN_SM120_NATIVE_CANDIDATE_STAGE_TIMING NFN_SM120_CANDIDATE_STAGE_TIMING NFN_SM120_PARITY_STAGE_TIMING NFN_SM120_STAGE_TIMING 0)"
LINEAR_SHAPE_STATS="$(env_or_alias5 NFN_SM120_NATIVE_LINEAR_SHAPE_STATS NFN_SM120_NATIVE_CANDIDATE_LINEAR_SHAPE_STATS NFN_SM120_CANDIDATE_LINEAR_SHAPE_STATS NFN_SM120_PARITY_LINEAR_SHAPE_STATS NFN_SM120_LINEAR_SHAPE_STATS 0)"
STARTUP_ONLY="$(env_or_alias5 NFN_SM120_NATIVE_STARTUP_ONLY NFN_SM120_NATIVE_CANDIDATE_STARTUP_ONLY NFN_SM120_CANDIDATE_STARTUP_ONLY NFN_SM120_PARITY_STARTUP_ONLY NFN_SM120_STARTUP_ONLY 0)"
DRY_RUN_PLAN="$(env_or_alias5 NFN_SM120_NATIVE_DRY_RUN_PLAN NFN_SM120_NATIVE_CANDIDATE_DRY_RUN_PLAN NFN_SM120_CANDIDATE_DRY_RUN_PLAN NFN_SM120_PARITY_DRY_RUN_PLAN NFN_SM120_DRY_RUN_PLAN 0)"
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
AUTO_ATTENTION_SECTION_TIMING=0
AUTO_DISABLE_METRIC_RATIO_GATES=0
if [[ -z "$CANDIDATE_EXTRA_ARGS_RAW" && -n "${NFN_SM120_NATIVE_CANDIDATE_ARGS-}" ]]; then
  CANDIDATE_EXTRA_ARGS_RAW="$NFN_SM120_NATIVE_CANDIDATE_ARGS"
fi
case "${CANDIDATE_PROFILE,,}" in
  ""|"none"|"off"|"0"|"false"|"no")
    ;;
  "lm_head_tk_dinput_32768"|"lm-head-tk-dinput-32768")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_TK_DINPUT_ENABLE_SHAPE=768,32768,50304,N,N"
    ;;
  "lm_head_cublaslt_dhidden_32768"|"lm-head-cublaslt-dhidden-32768")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE=768,32768,50304,N,N NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1 NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,32768,50304,N,N,0"
    ;;
  "lm_head_dhidden_fast16bf_32768"|"lm-head-dhidden-fast16bf-32768"|"lm_head_dhidden_gemmex_fast16bf_32768"|"lm-head-dhidden-gemmex-fast16bf-32768")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF_SHAPE=768,32768,50304,N,N"
    ;;
  "lm_head_tk_dweight_32768"|"lm-head-tk-dweight-32768")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_TK_DWEIGHT_ENABLE_SHAPE=768,50304,32768,N,T"
    ;;
  "lm_head_prepack_bf16_hidden_off"|"lm-head-prepack-bf16-hidden-off"|"lm_head_no_prepack_bf16_hidden"|"lm-head-no-prepack-bf16-hidden")
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=1"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=0"
    ;;
  "mlp_proj_tk_dweight_65536"|"mlp-proj-tk-dweight-65536"|"block_mlp_proj_tk_dweight_65536"|"block-mlp-proj-tk-dweight-65536")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_TK_DWEIGHT_ENABLE_SHAPE=3072,768,65536,N,T"
    ;;
  "layernorm_affine_row_chunk_128"|"layernorm-affine-row-chunk-128"|"ln_affine_row_chunk_128"|"ln-affine-row-chunk-128")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=128"
    ;;
  "layernorm_affine_row_chunk_512"|"layernorm-affine-row-chunk-512"|"ln_affine_row_chunk_512"|"ln-affine-row-chunk-512")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=512"
    ;;
  "linear_bias_row_chunk_256"|"linear-bias-row-chunk-256"|"bgrad_row_chunk_256"|"bgrad-row-chunk-256")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE=256"
    ;;
  "linear_bias_row_chunk_1024"|"linear-bias-row-chunk-1024"|"bgrad_row_chunk_1024"|"bgrad-row-chunk-1024")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE=1024"
    ;;
  "lm_head_logits_bf16_fallback_32768"|"lm-head-logits-bf16-fallback-32768")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=50304,32768,768,T,N"
    ;;
  "qkv_forward_bf16_fallback_65536"|"qkv-forward-bf16-fallback-65536"|"packed_qkv_forward_bf16_fallback_65536"|"packed-qkv-forward-bf16-fallback-65536")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 same-script gate regressed train_loop_wall_ms_per_step to 1.011419x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=2304,65536,768,T,N"
    ;;
  "ce_bf16_threads_512"|"ce-bf16-threads-512"|"lm_head_ce_bf16_threads_512"|"lm-head-ce-bf16-threads-512")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_CE_BF16_THREADS=512"
    ;;
  "lm_head_ce_vec8_io"|"lm-head-ce-vec8-io"|"ce_bf16_vec8_io"|"ce-bf16-vec8-io")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_CE_BF16_VEC_LOADS=1 NFN_NATIVE_GPT_CE_BF16_VEC_STORES=1"
    ;;
  "lm_head_ce_vec8_normal_store"|"lm-head-ce-vec8-normal-store"|"ce_bf16_vec8_normal_store"|"ce-bf16-vec8-normal-store")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_CE_BF16_VEC_LOADS=1 NFN_NATIVE_GPT_CE_BF16_VEC_NORMAL_STORES=1"
    ;;
  "lm_head_ce_scalar_streaming_store"|"lm-head-ce-scalar-streaming-store"|"ce_bf16_scalar_streaming_store"|"ce-bf16-scalar-streaming-store")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_CE_BF16_VEC_LOADS=1 NFN_NATIVE_GPT_CE_BF16_SCALAR_STREAMING_STORES=1"
    ;;
  "lm_head_ce_default_specialized"|"lm-head-ce-default-specialized"|"ce_bf16_default_specialized"|"ce-bf16-default-specialized")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_CE_DEFAULT_SPECIALIZED=1"
    ;;
  "lm_head_ce_no_loss_default_specialized"|"lm-head-ce-no-loss-default-specialized"|"ce_bf16_no_loss_default_specialized"|"ce-bf16-no-loss-default-specialized")
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_DEFAULT_SPECIALIZED=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_DEFAULT_SPECIALIZED=1"
    COMMON_EXTRA_ARGS_RAW="${COMMON_EXTRA_ARGS_RAW:+$COMMON_EXTRA_ARGS_RAW }--train-loss-every-steps 0"
    ;;
  "lm_head_ce_llmk_style_specialized"|"lm-head-ce-llmk-style-specialized"|"ce_bf16_llmk_style_specialized"|"ce-bf16-llmk-style-specialized")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_CE_LLMK_STYLE_SPECIALIZED=1"
    ;;
  "lm_head_ce_loss_bins_llmk_style_specialized"|"lm-head-ce-loss-bins-llmk-style-specialized"|"ce_bf16_loss_bins_llmk_style_specialized"|"ce-bf16-loss-bins-llmk-style-specialized")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=1 NFN_NATIVE_GPT_LM_HEAD_CE_LLMK_STYLE_SPECIALIZED=1"
    COMMON_EXTRA_ARGS_RAW="${COMMON_EXTRA_ARGS_RAW:+$COMMON_EXTRA_ARGS_RAW }--train-loss-every-steps 1"
    ;;
  "lm_head_ce_loss_bins_default_specialized"|"lm-head-ce-loss-bins-default-specialized"|"ce_bf16_loss_bins_default_specialized"|"ce-bf16-loss-bins-default-specialized")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=1 NFN_NATIVE_GPT_LM_HEAD_CE_LOSS_BINS_DEFAULT_SPECIALIZED=1"
    COMMON_EXTRA_ARGS_RAW="${COMMON_EXTRA_ARGS_RAW:+$COMMON_EXTRA_ARGS_RAW }--train-loss-every-steps 1"
    ;;
  "lm_head_loss_bins"|"lm-head-loss-bins"|"lm_head_loss_bin_reduction"|"lm-head-loss-bin-reduction")
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=1"
    COMMON_EXTRA_ARGS_RAW="${COMMON_EXTRA_ARGS_RAW:+$COMMON_EXTRA_ARGS_RAW }--train-loss-every-steps 1"
    ;;
  "lm_head_row_loss_sum_accumulate"|"lm-head-row-loss-sum-accumulate"|"lm_head_loss_sum_accumulate"|"lm-head-loss-sum-accumulate")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE=1"
    ;;
  "lm_head_row_loss_partial_reduce"|"lm-head-row-loss-partial-reduce"|"lm_head_row_loss_sum_accumulate_off"|"lm-head-row-loss-sum-accumulate-off")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE=0"
    ;;
  "cublaslt_min_waves"|"cublaslt-min-waves")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_POLICY=min_waves"
    ;;
  "cublaslt_max_waves"|"cublaslt-max-waves")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_POLICY=max_waves"
    ;;
  "cublaslt_grouped_probe"|"cublaslt-grouped-probe"|"grouped_cublaslt_probe"|"grouped-cublaslt-probe")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_PROBE_CUBLASLT_GROUPED_LAYOUT=1 NFN_NATIVE_GPT_PROBE_CUBLASLT_GROUPED_MATMUL=1"
    AUTO_DISABLE_METRIC_RATIO_GATES=1
    ;;
  "tk_dgelu_dinput"|"tk-dgelu-dinput")
    CANDIDATE_TILE_OPS_BUILD_FLAGS="${CANDIDATE_TILE_OPS_BUILD_FLAGS:+$CANDIDATE_TILE_OPS_BUILD_FLAGS }-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP"
    ;;
  "tk_dgelu_approx_tanh"|"tk-dgelu-approx-tanh")
    CANDIDATE_TILE_OPS_BUILD_FLAGS="${CANDIDATE_TILE_OPS_BUILD_FLAGS:+$CANDIDATE_TILE_OPS_BUILD_FLAGS }-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP -DLLMK_SM120_APPROX_DGELU_TANH=1"
    ;;
  "attention_atomic_dq"|"attention-atomic-dq")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 same-script gate failed route detection and regressed attention-backward section gates."
    CANDIDATE_TILE_OPS_BUILD_FLAGS="${CANDIDATE_TILE_OPS_BUILD_FLAGS:+$CANDIDATE_TILE_OPS_BUILD_FLAGS }-DLLMK_SM120_ATOMIC_DQ"
    ;;
  "bf16_attention_grad_out"|"bf16-attention-grad-out"|"attention_bf16_grad_out"|"attention-bf16-grad-out")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1"
    ;;
  "bf16_attention_dprep_grad_out"|"bf16-attention-dprep-grad-out"|"attention_bf16_dprep_grad_out"|"attention-bf16-dprep-grad-out")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_BF16_ATTENTION_DPREP_GRAD_OUT=1"
    ;;
  "attention_dprep_float_hd64_specialized"|"attention-dprep-float-hd64-specialized"|"float_attention_dprep_hd64"|"float-attention-dprep-hd64")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_FLOAT_HD64_SPECIALIZED=1"
    ;;
  "mlp_proj_dinput_before_dweight"|"mlp-proj-dinput-before-dweight")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 same-script gate failed route detection and rejected this scheduling order."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_MLP_PROJ_DINPUT_BEFORE_DWEIGHT=1"
    ;;
  "mlp_fc_dinput_before_dweight"|"mlp-fc-dinput-before-dweight")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 same-script gate failed route detection and rejected this scheduling order."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_MLP_FC_DINPUT_BEFORE_DWEIGHT=1"
    ;;
  "attn_proj_dinput_before_dweight"|"attn-proj-dinput-before-dweight")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 same-script gate failed route detection and rejected this scheduling order."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_ATTN_PROJ_DINPUT_BEFORE_DWEIGHT=1"
    ;;
  "lm_head_fused_loss_backward_off"|"lm-head-fused-loss-backward-off"|"lm_head_separate_loss_backward"|"lm-head-separate-loss-backward")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_FUSED_LOSS_BACKWARD=0"
    ;;
  "lm_head_classifier_ce_no_loss"|"lm-head-classifier-ce-no-loss"|"lm_head_no_loss_classifier_ce"|"lm-head-no-loss-classifier-ce")
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_CLASSIFIER_CE_NO_LOSS=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_CLASSIFIER_CE_NO_LOSS=1"
    ;;
  "tk_forward_no_n96"|"tk-forward-no-n96"|"llmk_forward_no_n96"|"llmk-forward-no-n96")
    CANDIDATE_TILE_OPS_BUILD_FLAGS="${CANDIDATE_TILE_OPS_BUILD_FLAGS:+$CANDIDATE_TILE_OPS_BUILD_FLAGS }-DLLMK_SM120_FORWARD_N96=0"
    ;;
  "cuda_device_max_connections_1"|"cuda-device-max-connections-1")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }CUDA_DEVICE_MAX_CONNECTIONS=1"
    ;;
  "combined_device_arena"|"combined-device-arena")
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_COMBINED_DEVICE_ARENA=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_COMBINED_DEVICE_ARENA=1"
    ;;
  "qkv_concurrent_dinput_dweight"|"qkv-concurrent-dinput-dweight")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 same-script gate activated this route but regressed train_loop_wall_ms_per_step to 1.005526x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_BLOCK_QKV_CONCURRENT_DINPUT_DWEIGHT=1"
    ;;
  "mlp_fc_concurrent_dinput_dweight"|"mlp-fc-concurrent-dinput-dweight")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 same-script gate activated this route but regressed train_loop_wall_ms_per_step to 1.005830x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_BLOCK_MLP_FC_CONCURRENT_DINPUT_DWEIGHT=1"
    ;;
  "attn_proj_concurrent_dinput_dweight"|"attn-proj-concurrent-dinput-dweight")
    REJECTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    REJECTED_CANDIDATE_REASON="CUDA 13.3 RTX 5090 same-script gate activated this route but regressed train_loop_wall_ms_per_step to 1.002312x."
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_BLOCK_ATTN_PROJ_CONCURRENT_DINPUT_DWEIGHT=1"
    ;;
  "lm_head_concurrent_dhidden_dweight"|"lm-head-concurrent-dhidden-dweight")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_CONCURRENT_DHIDDEN_DWEIGHT=1"
    ;;
  "lm_head_dweight_before_dhidden"|"lm-head-dweight-before-dhidden")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_DWEIGHT_BEFORE_DHIDDEN=1"
    ;;
  "lm_head_pipeline_chunks"|"lm-head-pipeline-chunks")
    TIMEOUT_PRONE_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_PIPELINE_CHUNKS=1"
    ;;
  "lm_head_overlap_last_dweight"|"lm-head-overlap-last-dweight"|"lm_head_last_dweight_overlap"|"lm-head-last-dweight-overlap")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_OVERLAP_LAST_DWEIGHT=1"
    ;;
  "lm_head_row_chunk_65536"|"lm-head-row-chunk-65536"|"lm_head_full_row_chunk"|"lm-head-full-row-chunk")
    TIMEOUT_PRONE_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_ALLOW_UNSAFE_LM_HEAD_ROW_CHUNK=1"
    CANDIDATE_EXTRA_ARGS_RAW="${CANDIDATE_EXTRA_ARGS_RAW:+$CANDIDATE_EXTRA_ARGS_RAW }--lm-head-row-chunk-size 65536"
    ;;
  "lm_head_full_resident_reuse"|"lm-head-full-resident-reuse"|"lm_head_full_batch_reuse"|"lm-head-full-batch-reuse")
    TIMEOUT_PRONE_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_ALLOW_UNSAFE_LM_HEAD_ROW_CHUNK=1 NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS=1 NFN_NATIVE_GPT_FULL_BATCH_LM_HEAD_REUSE=1"
    CANDIDATE_EXTRA_ARGS_RAW="${CANDIDATE_EXTRA_ARGS_RAW:+$CANDIDATE_EXTRA_ARGS_RAW }--lm-head-row-chunk-size 65536"
    ;;
  "lm_head_cooperative_backward_required"|"lm-head-cooperative-backward-required"|"require_cooperative_lm_head_backward"|"require-cooperative-lm-head-backward")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1"
    CANDIDATE_EXTRA_ARGS_RAW="${CANDIDATE_EXTRA_ARGS_RAW:+$CANDIDATE_EXTRA_ARGS_RAW }--require-cooperative-lm-head-backward"
    ;;
  "lm_head_cooperative_loss_bins"|"lm-head-cooperative-loss-bins")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1 NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=1 NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_LOSS_BINS=1"
    COMMON_EXTRA_ARGS_RAW="${COMMON_EXTRA_ARGS_RAW:+$COMMON_EXTRA_ARGS_RAW }--train-loss-every-steps 1"
    ;;
  "lm_head_cooperative_backward"|"lm-head-cooperative-backward")
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1"
    ;;
  "token_weight_vector4_strided"|"token-weight-vector4-strided")
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT=1"
    ;;
  "token_weight_threaded"|"token-weight-threaded")
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=0"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1"
    ;;
  "token_weight_fast_int32"|"token-weight-fast-int32"|"token_weight_no_vector4"|"token-weight-no-vector4")
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT=1"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT=0"
    ;;
  "token_weight_two_pass_bf16"|"token-weight-two-pass-bf16"|"token_weight_no_fused_bf16"|"token-weight-no-fused-bf16")
    BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_INIT=1"
    CANDIDATE_ENV_RAW="${CANDIDATE_ENV_RAW:+$CANDIDATE_ENV_RAW }NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_INIT=0"
    ;;
  *)
    echo "Unknown NFN_SM120_NATIVE_CANDIDATE_PROFILE: $CANDIDATE_PROFILE" >&2
    echo "Known profiles: lm_head_tk_dinput_32768, lm_head_cublaslt_dhidden_32768, lm_head_dhidden_fast16bf_32768, lm_head_tk_dweight_32768, lm_head_prepack_bf16_hidden_off, mlp_proj_tk_dweight_65536, layernorm_affine_row_chunk_128, layernorm_affine_row_chunk_512, linear_bias_row_chunk_256, linear_bias_row_chunk_1024, lm_head_logits_bf16_fallback_32768, qkv_forward_bf16_fallback_65536, ce_bf16_threads_512, lm_head_ce_vec8_io, lm_head_ce_vec8_normal_store, lm_head_ce_scalar_streaming_store, lm_head_ce_default_specialized, lm_head_ce_no_loss_default_specialized, lm_head_ce_llmk_style_specialized, lm_head_ce_loss_bins_llmk_style_specialized, lm_head_ce_loss_bins_default_specialized, lm_head_loss_bins, lm_head_row_loss_sum_accumulate, lm_head_row_loss_partial_reduce, cublaslt_min_waves, cublaslt_max_waves, cublaslt_grouped_probe, tk_dgelu_dinput, tk_dgelu_approx_tanh, attention_atomic_dq, bf16_attention_grad_out, bf16_attention_dprep_grad_out, attention_dprep_float_hd64_specialized, mlp_proj_dinput_before_dweight, mlp_fc_dinput_before_dweight, attn_proj_dinput_before_dweight, lm_head_fused_loss_backward_off, lm_head_classifier_ce_no_loss, tk_forward_no_n96, cuda_device_max_connections_1, combined_device_arena, qkv_concurrent_dinput_dweight, mlp_fc_concurrent_dinput_dweight, attn_proj_concurrent_dinput_dweight, lm_head_concurrent_dhidden_dweight, lm_head_dweight_before_dhidden, lm_head_pipeline_chunks, lm_head_overlap_last_dweight, lm_head_row_chunk_65536, lm_head_full_resident_reuse, lm_head_cooperative_backward, lm_head_cooperative_backward_required, lm_head_cooperative_loss_bins, token_weight_vector4_strided, token_weight_threaded, token_weight_fast_int32, token_weight_two_pass_bf16" >&2
    exit 2
    ;;
esac
ALLOW_REJECTED_CANDIDATE_PROFILE="$(env_or_alias NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE NFN_SM120_ALLOW_REJECTED_CANDIDATE_PROFILE 0)"
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
fi
if [[ -n "$CANDIDATE_TILE_OPS_BUILD_FLAGS" ]]; then
  if [[ -z "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB_EXPLICIT" ]]; then
    NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB="/tmp/nfn_sm120_candidate_tile_ops_${CANDIDATE_PROFILE:-custom}_$$.so"
    NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB_EXPLICIT="generated"
  fi
  NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="$CANDIDATE_TILE_OPS_BUILD_FLAGS" \
    bash "$ROOT_DIR/tools/build_native_train_tile_ops.sh" "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB" >&2
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
MAX_CANDIDATE_RATIO_RAW="$(env_or_alias NFN_SM120_NATIVE_MAX_CANDIDATE_RATIO NFN_SM120_CANDIDATE_MAX_CANDIDATE_RATIO "")"
MIN_CANDIDATE_RATIO_RAW="$(env_or_alias NFN_SM120_NATIVE_MIN_CANDIDATE_RATIO NFN_SM120_CANDIDATE_MIN_CANDIDATE_RATIO "")"
REQUIRE_NATIVE_ROUTE_CHANGE="$(env_or_alias NFN_SM120_NATIVE_REQUIRE_ROUTE_CHANGE NFN_SM120_CANDIDATE_REQUIRE_ROUTE_CHANGE auto)"
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
              *LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE*|*linear_backward_bias_row_chunk_size*)
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
                MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.ce.total_ms=1.000"
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
              *MLP_PROJ_TK_DWEIGHT*|*mlp_proj_tk_dweight*|*3072,768,65536,N,T*)
                ;;
              *TK_DWEIGHT*|*tk_dweight*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.dweight.total_ms=1.000"
                ;;
            esac
            case "$candidate_gate_text" in
              *MLP_PROJ_TK_DWEIGHT*|*mlp_proj_tk_dweight*|*3072,768,65536,N,T*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.mlp_proj.dweight_bias.total_ms=1.000"
                ;;
            esac
            case "$candidate_gate_text" in
              *LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE*|*linear_backward_bias_row_chunk_size*)
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
              *BLOCK_QKV_CONCURRENT_DINPUT_DWEIGHT*|*block_qkv_concurrent_dinput_dweight*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.qkv.total_ms=1.000"
                ;;
            esac
            case "$candidate_gate_text" in
              *BLOCK_MLP_FC_CONCURRENT_DINPUT_DWEIGHT*|*block_mlp_fc_concurrent_dinput_dweight*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.mlp_fc.total_ms=1.000"
                ;;
            esac
            case "$candidate_gate_text" in
              *BLOCK_ATTN_PROJ_CONCURRENT_DINPUT_DWEIGHT*|*block_attn_proj_concurrent_dinput_dweight*|*attn_proj_concurrent_dinput_dweight*)
                MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.attn_proj.total_ms=1.000"
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
if [[ ! -x "$NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN" ]]; then
  echo "Candidate NeuralFn native GPT trainer is not executable: $NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN" >&2
  exit 2
fi
NFN_NATIVE_TILE_OPS_ARG="$(tile_ops_arg_for "$NFN_NATIVE_GPT_TRAIN_BIN" "$NFN_NATIVE_TILE_OPS_LIB" "$NFN_NATIVE_TILE_OPS_LIB_EXPLICIT")"
NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_ARG="$(tile_ops_arg_for "$NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN" "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB" "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB_EXPLICIT")"
if [[ "$NFN_NATIVE_TILE_OPS_ARG" != "linked" && ! -f "$NFN_NATIVE_TILE_OPS_ARG" ]]; then
  echo "Baseline NeuralFn Tile ops library is missing: $NFN_NATIVE_TILE_OPS_LIB" >&2
  exit 2
fi
if [[ "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_ARG" != "linked" && ! -f "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_ARG" ]]; then
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
