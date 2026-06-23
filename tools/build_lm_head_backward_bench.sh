#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT_DIR}/build/lm_head_backward_bench}"
SRC="${ROOT_DIR}/neuralfn/csrc/native_train/lm_head_backward_bench.cpp"
NVCC_BIN="${NVCC:-nvcc}"
CUDA_ARCH="${NFN_TILE_CUDA_ARCH:-sm_120a}"

mkdir -p "$(dirname "${OUT}")"
"${NVCC_BIN}" -std=c++20 -O3 \
  -arch="${CUDA_ARCH}" \
  -I"${ROOT_DIR}/neuralfn/csrc/native_train" \
  "${SRC}" -lcudart -ldl -o "${OUT}"
printf '%s\n' "${OUT}"
