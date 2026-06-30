#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/neuralfn/csrc/native_train/missing_native_train.cpp"
GPT2_EVO_SRC="${ROOT_DIR}/neuralfn/csrc/native_train/gpt2_evo_native_train.cpp"
NANOGPT_SRC="${ROOT_DIR}/neuralfn/csrc/native_train/nanogpt_native_train.cpp"
TOKEN_SHARDS_SRC="${ROOT_DIR}/neuralfn/csrc/native_train/token_shards.cpp"
OUT_DIR="${1:-${ROOT_DIR}/build}"
CXX_BIN="${CXX:-c++}"

mkdir -p "${OUT_DIR}"

build_one() {
  local model="$1"
  local target="$2"
  local required="$3"
  local symbols="$4"
  local out="${OUT_DIR}/${target}"
  "${CXX_BIN}" -std=c++20 -O3 -Wall -Wextra -pedantic \
    -DNFN_NATIVE_MODEL_FAMILY="\"${model}\"" \
    -DNFN_NATIVE_TARGET_NAME="\"${target}\"" \
    -DNFN_NATIVE_REQUIRED_KERNELS="\"${required}\"" \
    -DNFN_NATIVE_REQUIRED_SYMBOLS="\"${symbols}\"" \
    -I"${ROOT_DIR}/neuralfn/csrc/native_train" \
    "${SRC}" "${TOKEN_SHARDS_SRC}" -ldl -o "${out}"
  printf '%s\n' "${out}"
}

build_nanogpt() {
  local out="${OUT_DIR}/nfn_nanogpt_native_train"
  "${CXX_BIN}" -std=c++20 -O3 -Wall -Wextra -pedantic \
    -I"${ROOT_DIR}/neuralfn/csrc/native_train" \
    "${NANOGPT_SRC}" "${TOKEN_SHARDS_SRC}" -ldl -o "${out}"
  printf '%s\n' "${out}"
}

"${CXX_BIN}" -std=c++20 -O3 -Wall -Wextra -pedantic \
  -I"${ROOT_DIR}/neuralfn/csrc/native_train" \
  "${GPT2_EVO_SRC}" -ldl -o "${OUT_DIR}/nfn_gpt2_evo_native_train"
printf '%s\n' "${OUT_DIR}/nfn_gpt2_evo_native_train"
build_nanogpt
build_one "llama" "nfn_llama_native_train" \
  "LLaMA RoPE/RMSNorm/SwiGLU attention and MLP CUDA Tile trainer" \
  "nfn_native_tile_token_embedding_u16_float32,nfn_native_tile_rms_norm_float32,nfn_native_tile_rms_norm_backward_input_float32,nfn_native_tile_linear_bf16_input_weight_bf16_output_float32,nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32,nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_merged_grad_float32,nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_u16_targets_with_workspace,nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32"
build_one "mixllama" "nfn_mixllama_native_train" \
  "LLaMA MoE routing, expert dispatch/combine, and grouped expert CUDA Tile trainer" \
  "nfn_native_tile_token_embedding_u16_float32,nfn_native_tile_rms_norm_float32,nfn_native_tile_linear_bf16_input_weight_bf16_output_float32,nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32,nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32"
build_one "jepa" "nfn_jepa_native_train" \
  "semantic JEPA masking, projector/predictor, latent loss, and native dataset loop kernels" \
  "nfn_native_tile_linear_float32,nfn_native_tile_linear_backward_input_float32,nfn_native_tile_linear_backward_weight_accumulate_float32,nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32"
build_one "semantic-router-moe" "nfn_semantic_router_moe_native_train" \
  "semantic router, hash/topic routing, MoE expert, load-balance, and route-loss CUDA Tile trainer" \
  "nfn_native_tile_linear_float32,nfn_native_tile_linear_backward_input_float32,nfn_native_tile_linear_backward_weight_accumulate_float32,nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32"
build_one "deepseek-v4" "nfn_deepseek_v4_native_train" \
  "DeepSeek sparse attention, MoE routing, and native optimizer CUDA Tile trainer" \
  "nfn_native_tile_token_embedding_u16_float32,nfn_native_tile_rms_norm_float32,nfn_native_tile_linear_bf16_input_weight_bf16_output_float32,nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32,nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32"
