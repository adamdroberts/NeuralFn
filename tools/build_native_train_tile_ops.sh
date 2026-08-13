#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${NFN_NATIVE_TRAIN_TILE_OPS_OUT:-${ROOT_DIR}/build/libnfn_native_train_tile_ops.so}}"
if [[ "${OUT}" == *.so ]]; then
  DEFAULT_STRICT_OUT="${OUT%.so}_strict.so"
else
  DEFAULT_STRICT_OUT="${OUT}_strict.so"
fi
STRICT_OUT="${NFN_NATIVE_STRICT_INFERENCE_TILE_OPS_OUT:-${DEFAULT_STRICT_OUT}}"
NVCC_BIN="${NVCC:-nvcc}"
KERNELS_SRC="${ROOT_DIR}/neuralfn/csrc/tile_cuda/kernels.cu"
ABI_SRC="${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.cu"
EXTRA_NVCC_FLAGS=()
EXTRA_LDLIBS=()
USE_TK_ATTENTION="${NFN_TILE_CUDA_USE_TK_ATTENTION:-1}"
CUDA_ARCH="${NFN_TILE_CUDA_ARCH:-$([[ "${USE_TK_ATTENTION}" == "1" ]] && printf 'sm_120a' || printf 'sm_120')}"
if [[ "${USE_TK_ATTENTION}" == "1" ]]; then
  if [[ "${CUDA_ARCH}" == "sm_120" ]]; then
    CUDA_ARCH="sm_120a"
  elif [[ "${CUDA_ARCH}" == "compute_120" ]]; then
    CUDA_ARCH="compute_120a"
  fi
  LLM_KITTENS_ROOT="${LLM_KITTENS_ROOT:-/mnt/disk2/dev/open-source/llm.kittens}"
  TK_ROOT="${TK_ROOT:-/mnt/disk2/dev/open-source/ThunderKittens}"
  EXTRA_NVCC_FLAGS+=(
    "--threads=0"
    "-t=0"
    "--expt-extended-lambda"
    "--expt-relaxed-constexpr"
    "--use_fast_math"
    "-forward-unknown-to-host-compiler"
    "-Xcompiler=-Wno-psabi"
    "-Xcompiler=-fno-strict-aliasing"
    "-ftemplate-backtrace-limit=0"
    "-DKITTENS_SM120"
    "-DENABLE_BF16"
    "-DLLMK_SM120_DPREP_WARPS=3"
    "-DLLMK_SM120_MEMORY_BLOCK_SIZE=1024"
    "-DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1"
    "-DLLMK_SM120_USE_CUBLASLT_GEMM"
    "-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP"
    "-DLLMK_SM120_APPROX_DGELU_TANH=1"
    "-DNFN_TILE_CUDA_USE_TK_ATTENTION=1"
    "-I${LLM_KITTENS_ROOT}"
    "-I${TK_ROOT}/include"
    "-I${TK_ROOT}/prototype"
  )
  EXTRA_LDLIBS+=("-lcuda")
fi
if [[ -n "${NFN_TILE_CUDA_EXTRA_NVCC_FLAGS:-}" ]]; then
  # Local kernel-candidate builds use simple whitespace-separated flags such as
  # -DLLMK_SM120_DWEIGHT_SUPER_M=2 -DLLMK_SM120_FORWARD_N96=0.
  read -r -a NFN_TILE_CUDA_USER_NVCC_FLAGS <<< "${NFN_TILE_CUDA_EXTRA_NVCC_FLAGS}"
  EXTRA_NVCC_FLAGS+=("${NFN_TILE_CUDA_USER_NVCC_FLAGS[@]}")
fi
if [[ -n "${NFN_TILE_CUDA_EXTRA_LDLIBS:-}" ]]; then
  read -r -a NFN_TILE_CUDA_USER_LDLIBS <<< "${NFN_TILE_CUDA_EXTRA_LDLIBS}"
  EXTRA_LDLIBS+=("${NFN_TILE_CUDA_USER_LDLIBS[@]}")
fi

mkdir -p "$(dirname "${OUT}")"
"${NVCC_BIN}" -std=c++20 -O3 --shared -Xcompiler -fPIC \
  -enable-tile \
  -arch="${CUDA_ARCH}" \
  -DNFN_TILE_CUDA_USE_CUBLAS_LINEAR=1 \
  -I"${ROOT_DIR}/neuralfn/csrc/native_train" \
  "${EXTRA_NVCC_FLAGS[@]}" \
  "${KERNELS_SRC}" "${ABI_SRC}" \
  -Xlinker -Bsymbolic \
  -lcublas -lcublasLt "${EXTRA_LDLIBS[@]}" \
  -o "${OUT}"
printf '%s\n' "${OUT}"

# Temperature-zero checkpoint inference deliberately uses a separate binary.
# Keep all training/TK/cuBLAS tuning in the normal library while compiling the
# strict sidecar without fast math, TF32/cuBLAS, TK/BF16, or packed-kernel
# feature macros.  The inference ABI then dispatches only deterministic FP32
# kernels from this library.
case "${NFN_NATIVE_BUILD_STRICT_TILE_OPS:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    # Strict Glimmer inference uses the generic CUDA Tile kernels rather than
    # the ThunderKittens SM120 specialization, so it does not require an
    # architecture-qualified suffix such as sm_120a. Inherit the selected
    # device architecture so the same release harness works on sm_80/sm_89/
    # sm_90 as well as sm_120.
    STRICT_CUDA_ARCH="${NFN_TILE_CUDA_STRICT_ARCH:-${CUDA_ARCH%a}}"
    mkdir -p "$(dirname "${STRICT_OUT}")"
    "${NVCC_BIN}" -std=c++20 -O3 --shared -Xcompiler -fPIC \
      -enable-tile \
      -arch="${STRICT_CUDA_ARCH}" \
      --ftz=false \
      --prec-div=true \
      --prec-sqrt=true \
      -DNFN_TILE_CUDA_STRICT_MATH_BUILD=1 \
      -I"${ROOT_DIR}/neuralfn/csrc/native_train" \
      "${KERNELS_SRC}" "${ABI_SRC}" \
      -Xlinker -Bsymbolic \
      -o "${STRICT_OUT}"
    printf '%s\n' "${STRICT_OUT}"
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "Unsupported NFN_NATIVE_BUILD_STRICT_TILE_OPS value: ${NFN_NATIVE_BUILD_STRICT_TILE_OPS}" >&2
    exit 2
    ;;
esac
