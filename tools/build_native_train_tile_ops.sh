#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${NFN_NATIVE_TRAIN_TILE_OPS_OUT:-${ROOT_DIR}/build/libnfn_native_train_tile_ops.so}}"
NVCC_BIN="${NVCC:-nvcc}"
KERNELS_SRC="${ROOT_DIR}/neuralfn/csrc/tile_cuda/kernels.cu"
ABI_SRC="${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.cu"
EXTRA_NVCC_FLAGS=()
EXTRA_LDLIBS=()
USE_TK_ATTENTION="${NFN_TILE_CUDA_USE_TK_ATTENTION:-1}"
CUDA_ARCH="${NFN_TILE_CUDA_ARCH:-$([[ "${USE_TK_ATTENTION}" == "1" ]] && printf 'sm_120a' || printf 'sm_120')}"
if [[ "${USE_TK_ATTENTION}" == "1" ]]; then
  LLM_KITTENS_ROOT="${LLM_KITTENS_ROOT:-/mnt/disk2/dev/open-source/llm.kittens}"
  TK_ROOT="${TK_ROOT:-/mnt/disk2/dev/open-source/ThunderKittens}"
  EXTRA_NVCC_FLAGS+=(
    "--expt-extended-lambda"
    "--expt-relaxed-constexpr"
    "--use_fast_math"
    "-DKITTENS_SM120"
    "-DENABLE_BF16"
    "-DNFN_TILE_CUDA_USE_TK_ATTENTION=1"
    "-I${LLM_KITTENS_ROOT}"
    "-I${TK_ROOT}/include"
    "-I${TK_ROOT}/prototype"
  )
  EXTRA_LDLIBS+=("-lcuda")
fi

mkdir -p "$(dirname "${OUT}")"
"${NVCC_BIN}" -std=c++20 -O3 --shared -Xcompiler -fPIC \
  -enable-tile \
  -arch="${CUDA_ARCH}" \
  -DNFN_TILE_CUDA_USE_CUBLAS_LINEAR=1 \
  -I"${ROOT_DIR}/neuralfn/csrc/native_train" \
  "${EXTRA_NVCC_FLAGS[@]}" \
  "${KERNELS_SRC}" "${ABI_SRC}" \
  -lcublas "${EXTRA_LDLIBS[@]}" \
  -o "${OUT}"
printf '%s\n' "${OUT}"
