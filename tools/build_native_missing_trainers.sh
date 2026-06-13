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
  local out="${OUT_DIR}/${target}"
  "${CXX_BIN}" -std=c++20 -O3 -Wall -Wextra -pedantic \
    -DNFN_NATIVE_MODEL_FAMILY="\"${model}\"" \
    -DNFN_NATIVE_TARGET_NAME="\"${target}\"" \
    -DNFN_NATIVE_REQUIRED_KERNELS="\"${required}\"" \
    "${SRC}" -o "${out}"
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
  "${GPT2_EVO_SRC}" -o "${OUT_DIR}/nfn_gpt2_evo_native_train"
printf '%s\n' "${OUT_DIR}/nfn_gpt2_evo_native_train"
build_nanogpt
build_one "llama" "nfn_llama_native_train" "LLaMA RoPE/RMSNorm/SwiGLU attention and MLP CUDA Tile trainer"
build_one "mixllama" "nfn_mixllama_native_train" "LLaMA MoE routing, expert dispatch/combine, and grouped expert CUDA Tile trainer"
build_one "jepa" "nfn_jepa_native_train" "semantic JEPA masking, projector/predictor, latent loss, and native dataset loop kernels"
build_one "semantic-router-moe" "nfn_semantic_router_moe_native_train" "semantic router, hash/topic routing, MoE expert, load-balance, and route-loss CUDA Tile trainer"
build_one "deepseek-v4" "nfn_deepseek_v4_native_train" "DeepSeek sparse attention, MoE routing, and native optimizer CUDA Tile trainer"
