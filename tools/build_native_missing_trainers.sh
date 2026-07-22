#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/neuralfn/csrc/native_train/missing_native_train.cpp"
GPT2_EVO_SRC="${ROOT_DIR}/neuralfn/csrc/native_train/gpt2_evo_native_train.cpp"
NANOGPT_SRC="${ROOT_DIR}/neuralfn/csrc/native_train/nanogpt_native_train.cpp"
TOKEN_SHARDS_SRC="${ROOT_DIR}/neuralfn/csrc/native_train/token_shards.cpp"
OUT_DIR="${1:-${ROOT_DIR}/build}"
CXX_BIN="${CXX:-c++}"
CXX_OPT_FLAGS="${NFN_NATIVE_MISSING_CXX_OPT_FLAGS:--O3}"
PRODUCTION_LOOP_ALL="${NFN_NATIVE_MISSING_PRODUCTION_LOOP_ALL:-1}"
PRODUCTION_LOOP_TARGETS=",${NFN_NATIVE_MISSING_PRODUCTION_LOOP_TARGETS:-},"
FULL_GEOMETRY_TARGETS=",${NFN_NATIVE_MISSING_FULL_GEOMETRY_TARGETS:-},"
FULL_GEOMETRY_ALL="${NFN_NATIVE_MISSING_FULL_GEOMETRY_ALL:-1}"
BUILD_TARGETS=",${NFN_NATIVE_MISSING_BUILD_TARGETS:-},"

mkdir -p "${OUT_DIR}"

target_selected() {
  local model="$1"
  local target="$2"
  [[ "${BUILD_TARGETS}" == ",," ||
     "${BUILD_TARGETS}" == *",${model},"* ||
     "${BUILD_TARGETS}" == *",${target},"* ]]
}

dedupe_csv() {
  local input="$1"
  local output=""
  local seen=","
  local item
  IFS=',' read -ra parts <<< "${input}"
  for item in "${parts[@]}"; do
    [[ -z "${item}" ]] && continue
    if [[ "${seen}" != *",${item},"* ]]; then
      seen="${seen}${item},"
      if [[ -n "${output}" ]]; then
        output="${output},${item}"
      else
        output="${item}"
      fi
    fi
  done
  printf '%s' "${output}"
}

production_loop_for() {
  local model="$1"
  local target="$2"
  if [[ "${model}" == "hnet-lm" || "${model}" == "hnet" ||
        "${model}" == "diffusion" || "${model}" == "diffusion-modern" ||
        "${model}" == "seq2seq" || "${model}" == "seq2seq-modern" ||
        "${model}" == "ttt-llama" || "${model}" == "ttt" ||
        "${model}" == "universal-llama" || "${model}" == "universal" ||
        "${model}" == "jamba" ]]; then
    printf '1'
    return
  fi
  if [[ "${PRODUCTION_LOOP_ALL}" == "1" ]]; then
    printf '1'
    return
  fi
  if [[ "${PRODUCTION_LOOP_TARGETS}" == *",${model},"* || "${PRODUCTION_LOOP_TARGETS}" == *",${target},"* ]]; then
    printf '1'
    return
  fi
  printf '0'
}

full_geometry_for() {
  local model="$1"
  local target="$2"
  if [[ "${model}" == "hnet-lm" || "${model}" == "hnet" ||
        "${model}" == "diffusion" || "${model}" == "diffusion-modern" ||
        "${model}" == "seq2seq" || "${model}" == "seq2seq-modern" ||
        "${model}" == "ttt-llama" || "${model}" == "ttt" ||
        "${model}" == "universal-llama" || "${model}" == "universal" ||
        "${model}" == "jamba" ]]; then
    printf '1'
    return
  fi
  if [[ "${FULL_GEOMETRY_ALL}" == "1" ]]; then
    printf '1'
    return
  fi
  if [[ "${FULL_GEOMETRY_TARGETS}" == *",${model},"* || "${FULL_GEOMETRY_TARGETS}" == *",${target},"* ]]; then
    printf '1'
    return
  fi
  printf '0'
}

build_one() {
  local model="$1"
  local target="$2"
  if ! target_selected "${model}" "${target}"; then
    return
  fi
  local required="$3"
  local symbols="$4"
  local coverage_class="${5:-family-native-loop-missing}"
  local missing_requirements="${6:-}"
  local completed_requirements="${7:-}"
  local production_loop="${8:-$(production_loop_for "${model}" "${target}")}"
  local full_geometry="${9:-$(full_geometry_for "${model}" "${target}")}"
  local out="${OUT_DIR}/${target}"
  if [[ "${production_loop}" == "1" ]]; then
    symbols="${symbols},nfn_native_tile_token_embedding_u16_float32,nfn_native_tile_token_embedding_backward_weight_u16_float32,nfn_native_tile_linear_float32,nfn_native_tile_linear_backward_input_float32,nfn_native_tile_linear_backward_weight_float32,nfn_native_tile_token_cross_entropy_partials_float32,nfn_native_tile_token_cross_entropy_backward_float32,nfn_native_tile_sum_partials_float32,nfn_native_tile_sum_accumulate_float32,nfn_native_tile_fill_float32"
  fi
  if [[ "${full_geometry}" == "1" ]]; then
    symbols="${symbols},nfn_native_tile_rms_norm_float32,nfn_native_tile_rms_norm_backward_input_float32,nfn_native_tile_layer_norm_float32,nfn_native_tile_gelu_float32,nfn_native_tile_vector_binary_float32,nfn_native_tile_rotary_embedding_float32,nfn_native_tile_rotary_embedding_backward_float32,nfn_native_tile_split_qkv_to_heads_float32,nfn_native_tile_merge_heads_to_qkv_float32,nfn_native_tile_reshape_heads_float32,nfn_native_tile_merge_heads_float32,nfn_native_tile_scaled_dot_product_attention_float32,nfn_native_tile_scaled_dot_product_attention_backward_float32,nfn_native_tile_scaled_residual_add_float32,nfn_native_tile_swiglu_float32,nfn_native_tile_swiglu_backward_float32,nfn_native_tile_add_float32,nfn_native_tile_copy_float32"
    if [[ "${model}" == "mixllama" || "${model}" == "deepseek-v4" || "${model}" == "moe-jepa-evo" || "${model}" == "semantic-router-moe" ]]; then
      symbols="${symbols},nfn_native_tile_topk_route_float32,nfn_native_tile_topk_route_backward_float32,nfn_native_tile_expert_bias_add_float32,nfn_native_tile_broadcast_expert_routes_float32,nfn_native_tile_moe_swiglu_forward_float32,nfn_native_tile_moe_swiglu_backward_float32,nfn_native_tile_moe_swiglu_backward_with_route_grad_float32"
    fi
    if [[ "${model}" == "deepseek-v4" ]]; then
      symbols="${symbols},nfn_native_tile_mhc_beta_gradient_float32,nfn_native_tile_moe_swiglu_forward_quantized_float32,nfn_native_tile_moe_swiglu_backward_quantized_float32"
    fi
    if [[ "${model}" == "semantic-router-moe" ]]; then
      symbols="${symbols},nfn_native_tile_semantic_hash_table_backward_float32,nfn_native_tile_semantic_route_policy_float32,nfn_native_tile_semantic_route_policy_packed_topic_float32,nfn_native_tile_semantic_route_policy_packed_topic_matrix_float32,nfn_native_tile_semantic_vec_from_packed_topic_float32,nfn_native_tile_semantic_packed_topic_to_padded_float32,nfn_native_tile_semantic_signature_scalar_float32,nfn_native_tile_semantic_vec_append_signature_float32,nfn_native_tile_semantic_vec_split_signature_grad_float32,nfn_native_tile_semantic_signature_scalar_backward_float32,nfn_native_tile_semantic_shared_topk_route_float32,nfn_native_tile_semantic_shared_forced_topk_route_float32,nfn_native_tile_semantic_shared_topk_route_backward_float32,nfn_native_tile_semantic_shared_expert_projection_float32,nfn_native_tile_semantic_shared_expert_projection_backward_float32,nfn_native_tile_semantic_free_expert_projection_float32,nfn_native_tile_semantic_free_expert_projection_backward_float32,nfn_native_tile_semantic_router_bias_add_float32,nfn_native_tile_semantic_router_bias_backward_float32,nfn_native_tile_semantic_targets_from_tokens_u16_int64,nfn_native_tile_semantic_target_matrix_from_tokens_u16_int64,nfn_native_tile_semantic_alignment_packed_loss_backward_float32,nfn_native_tile_semantic_target_topic_packed_distillation_backward_float32"
    fi
    if [[ "${model}" == "jepa" || "${model}" == "semantic-dense-jepa" || "${model}" == "moe-jepa-evo" || "${model}" == "semantic-router-moe" || "${model}" == "diffusion" || "${model}" == "diffusion-modern" ]]; then
      symbols="${symbols},nfn_native_tile_latent_mse_loss_float32"
    fi
    if [[ "${model}" == "jepa" || "${model}" == "moe-jepa-evo" ]]; then
      symbols="${symbols},nfn_native_tile_native_family_jepa_mask_u16_float32"
    fi
    if [[ "${model}" == "jamba" ]]; then
      symbols="${symbols},nfn_native_tile_causal_chunk_state_float32,nfn_native_tile_causal_chunk_state_backward_float32"
    fi
    if [[ "${model}" == "seq2seq" || "${model}" == "seq2seq-modern" ||
          "${model}" == "ttt-llama" || "${model}" == "ttt" ||
          "${model}" == "jamba" || "${model}" == "universal-llama" ||
          "${model}" == "universal" ]]; then
      symbols="${symbols},nfn_native_tile_uint16_to_int64"
    fi
    if [[ "${model}" == "seq2seq" || "${model}" == "seq2seq-modern" ||
          "${model}" == "hnet-lm" || "${model}" == "hnet" ]]; then
      symbols="${symbols},nfn_native_tile_extract_diagonal_float32"
    fi
    if [[ "${model}" == "diffusion" || "${model}" == "diffusion-modern" ]]; then
      symbols="${symbols},nfn_native_tile_diffusion_mask_u16_int64"
    fi
    if [[ "${model}" == "ttt-llama" || "${model}" == "ttt" || "${model}" == "universal-llama" || "${model}" == "universal" ]]; then
      symbols="${symbols},nfn_native_tile_tanh_float32,nfn_native_tile_tanh_backward_float32"
    fi
    if [[ "${model}" == "hnet-lm" || "${model}" == "hnet" ]]; then
      symbols="${symbols},nfn_native_tile_uint8_to_int64,nfn_native_tile_byte_patch_embed_float32,nfn_native_tile_byte_patch_merge_float32,nfn_native_tile_byte_patch_merge_backward_float32,nfn_native_tile_byte_patch_embed_backward_float32"
    fi
    if [[ "${model}" == "universal-llama" || "${model}" == "universal" ]]; then
      symbols="${symbols},nfn_native_tile_act_pack_step_float32,nfn_native_tile_act_prepare_weights_float32,nfn_native_tile_act_unpack_step_grad_float32"
    fi
  fi
  symbols="$(dedupe_csv "${symbols}")"
  "${CXX_BIN}" -std=c++20 ${CXX_OPT_FLAGS} -Wall -Wextra -pedantic \
    -DNFN_NATIVE_MODEL_FAMILY="\"${model}\"" \
    -DNFN_NATIVE_TARGET_NAME="\"${target}\"" \
    -DNFN_NATIVE_REQUIRED_KERNELS="\"${required}\"" \
    -DNFN_NATIVE_REQUIRED_SYMBOLS="\"${symbols}\"" \
    -DNFN_NATIVE_COVERAGE_CLASS="\"${coverage_class}\"" \
    -DNFN_NATIVE_MISSING_REQUIREMENTS="\"${missing_requirements}\"" \
    -DNFN_NATIVE_COMPLETED_REQUIREMENTS="\"${completed_requirements}\"" \
    -DNFN_NATIVE_PRODUCTION_LOOP="${production_loop}" \
    -DNFN_NATIVE_FULL_GEOMETRY_FORWARD_BACKWARD="${full_geometry}" \
    -I"${ROOT_DIR}/neuralfn/csrc/native_train" \
    "${SRC}" "${TOKEN_SHARDS_SRC}" -ldl -o "${out}"
  printf '%s\n' "${out}"
}

build_nanogpt() {
  local out="${OUT_DIR}/nfn_nanogpt_native_train"
  "${CXX_BIN}" -std=c++20 ${CXX_OPT_FLAGS} -Wall -Wextra -pedantic \
    -I"${ROOT_DIR}/neuralfn/csrc/native_train" \
    "${NANOGPT_SRC}" "${TOKEN_SHARDS_SRC}" -ldl -o "${out}"
  printf '%s\n' "${out}"
}

if target_selected "gpt2-evo" "nfn_gpt2_evo_native_train"; then
  "${CXX_BIN}" -std=c++20 ${CXX_OPT_FLAGS} -Wall -Wextra -pedantic \
    -I"${ROOT_DIR}/neuralfn/csrc/native_train" \
    "${GPT2_EVO_SRC}" -ldl -o "${OUT_DIR}/nfn_gpt2_evo_native_train"
  printf '%s\n' "${OUT_DIR}/nfn_gpt2_evo_native_train"
fi
if target_selected "nanogpt" "nfn_nanogpt_native_train"; then
  build_nanogpt
fi
build_one "llama" "nfn_llama_native_train" \
  "LLaMA RoPE/RMSNorm/SwiGLU attention and MLP CUDA Tile trainer" \
  "nfn_native_tile_token_embedding_float32,nfn_native_tile_token_embedding_backward_weight_float32,nfn_native_tile_token_embedding_u16_float32,nfn_native_tile_rms_norm_float32,nfn_native_tile_rms_norm_backward_input_float32,nfn_native_tile_rotary_embedding_float32,nfn_native_tile_rotary_embedding_backward_float32,nfn_native_tile_swiglu_float32,nfn_native_tile_swiglu_backward_float32,nfn_native_tile_linear_float32,nfn_native_tile_linear_quantized_float32,nfn_native_tile_linear_backward_input_float32,nfn_native_tile_linear_backward_input_quantized_float32,nfn_native_tile_split_last_dim_float32,nfn_native_tile_merge_last_dim_float32,nfn_native_tile_split_at_last_dim_float32,nfn_native_tile_concat_last_dim_float32,nfn_native_tile_repeat_kv_float32,nfn_native_tile_repeat_kv_backward_float32,nfn_native_tile_fused_causal_attention_forward_float32,nfn_native_tile_fused_causal_attention_backward_float32,nfn_native_tile_differential_combine_float32,nfn_native_tile_differential_backward_float32,nfn_native_tile_linear_backward_weight_float32,nfn_native_tile_split_qkv_to_heads_float32,nfn_native_tile_merge_heads_to_qkv_float32,nfn_native_tile_scaled_dot_product_attention_float32,nfn_native_tile_scaled_dot_product_attention_backward_float32,nfn_native_tile_scaled_residual_add_float32,nfn_native_tile_linear_bf16_input_weight_bf16_output_float32,nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32,nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_merged_grad_float32,nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_u16_targets_with_workspace,nfn_native_tile_adamw_step_float32,nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32" \
  "covered-llama-rope-swiglu-transformer-lm" \
  "" \
  "rmsnorm-loop-composition-smoke,rope-loop-composition-smoke,swiglu-geglu-mlp-loop-composition-smoke,lm-head-linear-ce-backward-adamw-smoke,token-lm-embedding-ce-backward-adamw-smoke,composed-token-block-lm-adamw-smoke,packed-qkv-attention-forward-backward-smoke,packed-qkv-attention-block-forward-smoke,packed-qkv-rope-attention-block-integration-smoke,rope-swiglu-block-forward-backward-adamw-smoke,llama-full-forward-backward-loop-smoke,llama-sampled-ar-plus-composed-step-dataset-loop,family-parameter-layout-checkpoint-inference-smoke,optimizer-updated-full-architecture-parameter-persistence"
build_one "mixllama" "nfn_mixllama_native_train" \
  "LLaMA MoE routing, expert dispatch/combine, and grouped expert CUDA Tile trainer" \
  "nfn_native_tile_token_embedding_u16_float32,nfn_native_tile_rms_norm_float32,nfn_native_tile_linear_bf16_input_weight_bf16_output_float32,nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32,nfn_native_tile_topk_route_float32,nfn_native_tile_topk_route_backward_float32,nfn_native_tile_broadcast_expert_routes_float32,nfn_native_tile_moe_swiglu_forward_float32,nfn_native_tile_moe_swiglu_backward_float32,nfn_native_tile_linear_float32,nfn_native_tile_linear_backward_input_float32,nfn_native_tile_linear_backward_weight_float32,nfn_native_tile_token_cross_entropy_partials_float32,nfn_native_tile_token_cross_entropy_backward_float32,nfn_native_tile_route_balance_density_float32,nfn_native_tile_route_balance_loss_float32,nfn_native_tile_adamw_step_float32,nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32" \
  "covered-standard-moe-transformer-lm" \
  "" \
  "router-topk-broadcast-smoke,routed-swiglu-expert-forward-backward-smoke,load-balance-loss-adamw-smoke,standard-moe-transformer-block-forward-smoke,standard-moe-transformer-block-forward-backward-adamw-smoke,standard-moe-transformer-lm-forward-backward-adamw-smoke,standard-moe-full-forward-backward-loop-smoke,standard-moe-sampled-family-forward-backward-optimizer-step,family-parameter-layout-checkpoint-inference-smoke,optimizer-updated-full-architecture-parameter-persistence"
build_one "moe-jepa-evo" "nfn_moe_jepa_evo_native_train" \
  "standard MoE transformer loop plus JEPA target encoder/projector/predictor and AR+JEPA+router loss CUDA Tile trainer" \
  "nfn_native_tile_token_embedding_u16_float32,nfn_native_tile_rms_norm_float32,nfn_native_tile_linear_bf16_input_weight_bf16_output_float32,nfn_native_tile_topk_route_float32,nfn_native_tile_topk_route_backward_float32,nfn_native_tile_broadcast_expert_routes_float32,nfn_native_tile_moe_swiglu_forward_float32,nfn_native_tile_moe_swiglu_backward_float32,nfn_native_tile_linear_float32,nfn_native_tile_linear_backward_input_float32,nfn_native_tile_linear_backward_weight_float32,nfn_native_tile_linear_backward_weight_accumulate_float32,nfn_native_tile_route_balance_density_float32,nfn_native_tile_route_balance_loss_float32,nfn_native_tile_native_family_jepa_mask_float32,nfn_native_tile_native_family_jepa_mask_u16_float32,nfn_native_tile_latent_pool_float32,nfn_native_tile_latent_pool_backward_float32,nfn_native_tile_token_cross_entropy_partials_float32,nfn_native_tile_token_cross_entropy_backward_float32,nfn_native_tile_latent_mse_loss_float32,nfn_native_tile_fill_float32,nfn_native_tile_adamw_step_float32,nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32" \
  "covered-moe-jepa-objective" \
  "" \
  "router-topk-broadcast-smoke,routed-swiglu-expert-forward-backward-smoke,load-balance-loss-adamw-smoke,standard-moe-transformer-block-forward-smoke,standard-moe-transformer-block-forward-backward-adamw-smoke,standard-moe-transformer-lm-forward-backward-adamw-smoke,standard-moe-full-forward-backward-loop-smoke,jepa-target-encoder-forward-smoke,jepa-projector-predictor-latent-loss-smoke,ar-plus-jepa-loss-composition-smoke,dense-jepa-ar-target-projector-forward-backward-adamw-smoke,ar-plus-jepa-plus-router-loss-composition-smoke,moe-jepa-sampled-family-forward-backward-optimizer-step,family-parameter-layout-checkpoint-inference-smoke,optimizer-updated-full-architecture-parameter-persistence"
build_one "jepa" "nfn_jepa_native_train" \
  "semantic JEPA masking, projector/predictor, latent loss, and native dataset loop kernels" \
  "nfn_native_tile_linear_float32,nfn_native_tile_linear_backward_input_float32,nfn_native_tile_linear_backward_weight_accumulate_float32,nfn_native_tile_native_family_jepa_mask_float32,nfn_native_tile_native_family_jepa_mask_u16_float32,nfn_native_tile_latent_pool_float32,nfn_native_tile_latent_pool_backward_float32,nfn_native_tile_token_cross_entropy_partials_float32,nfn_native_tile_token_cross_entropy_backward_float32,nfn_native_tile_latent_mse_loss_float32,nfn_native_tile_fill_float32,nfn_native_tile_adamw_step_float32,nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32" \
  "covered-dense-jepa-objective" \
  "" \
  "jepa-target-encoder-forward-smoke,jepa-projector-predictor-latent-loss-smoke,ar-plus-jepa-loss-composition-smoke,dense-jepa-ar-target-projector-forward-backward-adamw-smoke,dense-jepa-full-forward-backward-loop-smoke,dense-jepa-sampled-family-dataset-loop,family-parameter-layout-checkpoint-inference-smoke,optimizer-updated-full-architecture-parameter-persistence"
build_one "semantic-dense-jepa" "nfn_semantic_dense_jepa_native_train" \
  "semantic dense JEPA planner, semantic-alignment, latent loss, and native dataset loop kernels" \
  "nfn_native_tile_linear_float32,nfn_native_tile_linear_backward_input_float32,nfn_native_tile_linear_backward_weight_accumulate_float32,nfn_native_tile_semantic_hash_int64,nfn_native_tile_semantic_targets_from_tokens_u16_int64,nfn_native_tile_semantic_target_matrix_from_tokens_u16_int64,nfn_native_tile_semantic_alignment_loss_items_float32,nfn_native_tile_semantic_alignment_packed_loss_backward_float32,nfn_native_tile_semantic_target_topic_packed_distillation_backward_float32,nfn_native_tile_sum_accumulate_float32,nfn_native_tile_native_family_jepa_mask_float32,nfn_native_tile_latent_pool_float32,nfn_native_tile_latent_pool_backward_float32,nfn_native_tile_token_cross_entropy_partials_float32,nfn_native_tile_token_cross_entropy_backward_float32,nfn_native_tile_latent_mse_loss_float32,nfn_native_tile_fill_float32,nfn_native_tile_adamw_step_float32,nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32" \
  "covered-semantic-dense-jepa-objective" \
  "" \
  "jepa-target-encoder-forward-smoke,jepa-projector-predictor-latent-loss-smoke,ar-plus-jepa-loss-composition-smoke,dense-jepa-ar-target-projector-forward-backward-adamw-smoke,semantic-target-shard-resolver-smoke,semantic-hash-alignment-loss-items-smoke,semantic-dense-planner-alignment-adamw-smoke,semantic-planner-forward-backward-smoke,semantic-alignment-loss-device-reduction-smoke,ar-plus-semantic-plus-jepa-loss-composition-smoke,semantic-dense-jepa-sampled-family-dataset-loop,family-parameter-layout-checkpoint-inference-smoke,optimizer-updated-full-architecture-parameter-persistence"
build_one "semantic-router-moe" "nfn_semantic_router_moe_native_train" \
  "semantic router, hash/topic routing, MoE expert, load-balance, and route-loss CUDA Tile trainer" \
  "nfn_native_tile_linear_float32,nfn_native_tile_linear_backward_input_float32,nfn_native_tile_linear_backward_weight_accumulate_float32,nfn_native_tile_topk_route_float32,nfn_native_tile_topk_route_backward_float32,nfn_native_tile_semantic_shared_topk_route_float32,nfn_native_tile_semantic_shared_forced_topk_route_float32,nfn_native_tile_semantic_shared_topk_route_backward_float32,nfn_native_tile_broadcast_expert_routes_float32,nfn_native_tile_broadcast_chunk_routes_float32,nfn_native_tile_compact_chunk_routes_float32_int64,nfn_native_tile_aggregate_chunk_route_gradients_float32,nfn_native_tile_semantic_route_distillation_backward_float32,nfn_native_tile_semantic_target_topic_distillation_backward_float32,nfn_native_tile_semantic_target_topic_packed_distillation_backward_float32,nfn_native_tile_semantic_hash_table_backward_float32,nfn_native_tile_semantic_route_policy_float32,nfn_native_tile_semantic_route_policy_packed_topic_float32,nfn_native_tile_semantic_route_policy_packed_topic_matrix_float32,nfn_native_tile_semantic_vec_from_packed_topic_float32,nfn_native_tile_semantic_packed_topic_to_padded_float32,nfn_native_tile_semantic_signature_scalar_float32,nfn_native_tile_semantic_vec_append_signature_float32,nfn_native_tile_semantic_vec_split_signature_grad_float32,nfn_native_tile_semantic_signature_scalar_backward_float32,nfn_native_tile_semantic_shared_expert_projection_float32,nfn_native_tile_semantic_shared_expert_projection_backward_float32,nfn_native_tile_semantic_free_expert_projection_float32,nfn_native_tile_semantic_free_expert_projection_backward_float32,nfn_native_tile_semantic_router_bias_add_float32,nfn_native_tile_semantic_router_bias_backward_float32,nfn_native_tile_semantic_targets_from_tokens_u16_int64,nfn_native_tile_semantic_target_matrix_from_tokens_u16_int64,nfn_native_tile_semantic_alignment_packed_loss_backward_float32,nfn_native_tile_moe_swiglu_forward_float32,nfn_native_tile_moe_swiglu_backward_float32,nfn_native_tile_semantic_hash_int64,nfn_native_tile_semantic_alignment_loss_items_float32,nfn_native_tile_route_selection_loss_partials_float32,nfn_native_tile_softmax_distillation_partials_float32,nfn_native_tile_attentionless_decoder_float32,nfn_native_tile_expert_bias_add_float32,nfn_native_tile_route_balance_density_float32,nfn_native_tile_route_balance_loss_float32,nfn_native_tile_native_family_jepa_mask_float32,nfn_native_tile_native_family_jepa_mask_u16_float32,nfn_native_tile_causal_chunk_state_float32,nfn_native_tile_causal_chunk_state_backward_float32,nfn_native_tile_latent_pool_float32,nfn_native_tile_latent_pool_backward_float32,nfn_native_tile_latent_mse_loss_float32,nfn_native_tile_evo_mutate_candidates_float32,nfn_native_tile_evo_select_best_loss_float32,nfn_native_tile_evo_adopt_candidate_float32,nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32" \
  "covered-semantic-moe-router-jepa-objective" \
  "" \
  "router-topk-broadcast-smoke,routed-swiglu-expert-forward-backward-smoke,load-balance-loss-adamw-smoke,semantic-target-shard-resolver-smoke,semantic-hash-alignment-loss-items-smoke,route-selection-distillation-balance-losses-smoke,semantic-router-forward-backward-smoke,semantic-expert-dispatch-combine-smoke,semantic-router-moe-route-expert-adamw-smoke,ar-plus-semantic-plus-jepa-loss-composition-smoke,route-evo-device-controller-smoke,semantic-router-moe-sampled-family-dataset-loop,family-parameter-layout-checkpoint-inference-smoke,optimizer-updated-full-architecture-parameter-persistence"
build_one "deepseek-v4" "nfn_deepseek_v4_native_train" \
  "DeepSeek sparse attention, MoE routing, and native optimizer CUDA Tile trainer" \
  "nfn_native_tile_token_embedding_u16_float32,nfn_native_tile_rms_norm_float32,nfn_native_tile_linear_bf16_input_weight_bf16_output_float32,nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32,nfn_native_tile_topk_route_float32,nfn_native_tile_topk_route_backward_float32,nfn_native_tile_broadcast_expert_routes_float32,nfn_native_tile_moe_swiglu_forward_float32,nfn_native_tile_moe_swiglu_backward_float32,nfn_native_tile_moe_swiglu_forward_quantized_float32,nfn_native_tile_moe_swiglu_backward_quantized_float32,nfn_native_tile_linear_float32,nfn_native_tile_linear_backward_input_float32,nfn_native_tile_linear_backward_weight_float32,nfn_native_tile_token_cross_entropy_partials_float32,nfn_native_tile_token_cross_entropy_backward_float32,nfn_native_tile_route_balance_density_float32,nfn_native_tile_route_balance_loss_float32,nfn_native_tile_adamw_step_float32,nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32" \
  "covered-standard-moe-transformer-lm" \
  "" \
  "router-topk-broadcast-smoke,routed-swiglu-expert-forward-backward-smoke,load-balance-loss-adamw-smoke,standard-moe-transformer-block-forward-smoke,standard-moe-transformer-block-forward-backward-adamw-smoke,standard-moe-transformer-lm-forward-backward-adamw-smoke,standard-moe-full-forward-backward-loop-smoke,standard-moe-sampled-family-forward-backward-optimizer-step,family-parameter-layout-checkpoint-inference-smoke,optimizer-updated-full-architecture-parameter-persistence"
build_one "jamba" "nfn_jamba_native_train" \
  "Jamba hybrid Mamba plus transformer CUDA Tile trainer" \
  "nfn_native_tile_token_embedding_u16_float32,nfn_native_tile_token_embedding_backward_weight_u16_float32,nfn_native_tile_rms_norm_float32,nfn_native_tile_rms_norm_backward_input_float32,nfn_native_tile_linear_float32,nfn_native_tile_linear_backward_input_float32,nfn_native_tile_linear_backward_weight_accumulate_float32,nfn_native_tile_linear_backward_weight_float32,nfn_native_tile_latent_mse_loss_float32,nfn_native_tile_fill_float32,nfn_native_tile_causal_chunk_state_float32,nfn_native_tile_causal_chunk_state_backward_float32,nfn_native_tile_rotary_embedding_float32,nfn_native_tile_rotary_embedding_backward_float32,nfn_native_tile_split_qkv_to_heads_float32,nfn_native_tile_merge_heads_to_qkv_float32,nfn_native_tile_reshape_heads_float32,nfn_native_tile_merge_heads_float32,nfn_native_tile_scaled_dot_product_attention_float32,nfn_native_tile_scaled_dot_product_attention_backward_float32,nfn_native_tile_scaled_residual_add_float32,nfn_native_tile_add_float32,nfn_native_tile_copy_float32,nfn_native_tile_vector_binary_float32,nfn_native_tile_mhc_beta_gradient_float32,nfn_native_tile_topk_route_float32,nfn_native_tile_topk_route_backward_float32,nfn_native_tile_broadcast_expert_routes_float32,nfn_native_tile_moe_swiglu_forward_float32,nfn_native_tile_moe_swiglu_backward_float32,nfn_native_tile_moe_swiglu_backward_with_route_grad_float32,nfn_native_tile_adamw_step_float32,nfn_native_tile_linear_bf16_input_weight_bf16_output_float32,nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32" \
  "covered-jamba-hybrid-mamba-transformer-lm" \
  "" \
  "jamba-causal-chunk-state-head-adamw-smoke,jamba-mamba-state-forward-backward-adamw-smoke,jamba-layer-schedule-native-loop-smoke,jamba-sampled-family-dataset-loop,family-parameter-layout-checkpoint-inference-smoke,optimizer-updated-full-architecture-parameter-persistence"
build_one "seq2seq" "nfn_seq2seq_native_train" \
  "encoder-decoder cross-attention and seq2seq loss CUDA Tile trainer" \
  "nfn_native_tile_token_embedding_u16_float32,nfn_native_tile_linear_float32,nfn_native_tile_linear_backward_input_float32,nfn_native_tile_linear_backward_weight_accumulate_float32,nfn_native_tile_scaled_dot_product_attention_float32,nfn_native_tile_scaled_dot_product_attention_backward_float32,nfn_native_tile_token_cross_entropy_partials_float32,nfn_native_tile_token_cross_entropy_backward_float32,nfn_native_tile_fill_float32,nfn_native_tile_adamw_step_float32,nfn_native_tile_linear_bf16_input_weight_bf16_output_float32,nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32,nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32" \
  "covered-seq2seq-objective" \
  "" \
  "seq2seq-cross-attention-ce-adamw-smoke,seq2seq-loss-composition-adamw-smoke,seq2seq-full-encoder-decoder-loop-smoke,seq2seq-sampled-family-dataset-loop,family-parameter-layout-checkpoint-inference-smoke,optimizer-updated-full-architecture-parameter-persistence"
build_one "diffusion" "nfn_diffusion_native_train" \
  "diffusion timestep scheduler, denoise head, and loss CUDA Tile trainer" \
  "nfn_native_tile_random_timesteps_float32,nfn_native_tile_mask_scheduler_int64,nfn_native_tile_token_embedding_float32,nfn_native_tile_token_embedding_backward_weight_float32,nfn_native_tile_linear_float32,nfn_native_tile_linear_backward_input_float32,nfn_native_tile_linear_backward_weight_accumulate_float32,nfn_native_tile_token_cross_entropy_partials_float32,nfn_native_tile_token_cross_entropy_backward_float32,nfn_native_tile_latent_mse_loss_float32,nfn_native_tile_diffusion_mask_u16_int64,nfn_native_tile_fill_float32,nfn_native_tile_adamw_step_float32,nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32" \
  "covered-diffusion-objective" \
  "" \
  "diffusion-denoise-linear-mse-adamw-smoke,diffusion-timestep-mask-ce-adamw-smoke,diffusion-full-loop-smoke,diffusion-sampled-family-dataset-loop,family-parameter-layout-checkpoint-inference-smoke,optimizer-updated-full-architecture-parameter-persistence"
build_one "ttt-llama" "nfn_ttt_llama_native_train" \
  "test-time-training inner update and transformer CUDA Tile trainer" \
  "nfn_native_tile_token_embedding_u16_float32,nfn_native_tile_rms_norm_float32,nfn_native_tile_linear_float32,nfn_native_tile_tanh_float32,nfn_native_tile_add_float32,nfn_native_tile_tanh_backward_float32,nfn_native_tile_linear_backward_input_float32,nfn_native_tile_linear_backward_weight_accumulate_float32,nfn_native_tile_latent_mse_loss_float32,nfn_native_tile_fill_float32,nfn_native_tile_adamw_step_float32,nfn_native_tile_linear_bf16_input_weight_bf16_output_float32,nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32" \
  "covered-ttt-transformer-lm" \
  "" \
  "ttt-linear-mse-adamw-smoke,ttt-composite-inner-forward-backward-adamw-smoke,ttt-full-transformer-loop-smoke,ttt-sampled-family-dataset-loop,family-parameter-layout-checkpoint-inference-smoke,optimizer-updated-full-architecture-parameter-persistence"
build_one "hnet-lm" "nfn_hnet_lm_native_train" \
  "HNet byte-token patch and merge CUDA Tile trainer" \
  "nfn_native_tile_byte_patch_embed_float32,nfn_native_tile_byte_patch_merge_float32,nfn_native_tile_byte_patch_merge_backward_float32,nfn_native_tile_byte_patch_embed_backward_float32,nfn_native_tile_uint8_to_int64,nfn_native_tile_linear_float32,nfn_native_tile_linear_backward_input_float32,nfn_native_tile_linear_backward_weight_accumulate_float32,nfn_native_tile_fill_float32,nfn_native_tile_adamw_step_float32,nfn_native_tile_token_embedding_u16_float32,nfn_native_tile_linear_bf16_input_weight_bf16_output_float32,nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32" \
  "covered-hnet-byte-lm" \
  "" \
  "hnet-byte-patch-embed-merge-head-adamw-smoke,hnet-byte-patch-backward-adamw-smoke,hnet-byte-lm-loop-smoke,hnet-sampled-byte-family-dataset-loop,byte-token-shard-resolver-smoke,family-parameter-layout-checkpoint-inference-smoke,optimizer-updated-full-architecture-parameter-persistence"
build_one "universal-llama" "nfn_universal_llama_native_train" \
  "universal transformer recurrent layer and halting CUDA Tile trainer" \
  "nfn_native_tile_token_embedding_u16_float32,nfn_native_tile_rms_norm_float32,nfn_native_tile_linear_float32,nfn_native_tile_linear_backward_input_float32,nfn_native_tile_linear_backward_weight_accumulate_float32,nfn_native_tile_latent_mse_loss_float32,nfn_native_tile_act_halting_bce_grad_float32,nfn_native_tile_act_weighted_sum_float32,nfn_native_tile_act_pack_step_float32,nfn_native_tile_act_prepare_weights_float32,nfn_native_tile_act_unpack_step_grad_float32,nfn_native_tile_uint16_to_int64,nfn_native_tile_fill_float32,nfn_native_tile_adamw_step_float32,nfn_native_tile_linear_bf16_input_weight_bf16_output_float32,nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32" \
  "covered-universal-transformer-lm" \
  "" \
  "universal-recurrent-linear-mse-adamw-smoke,universal-act-halt-loss-gradient-smoke,universal-transformer-loop-smoke,universal-sampled-family-dataset-loop,family-parameter-layout-checkpoint-inference-smoke,optimizer-updated-full-architecture-parameter-persistence"
