#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${NFN_NATIVE_REBUILD_OUT_DIR:-${ROOT_DIR}/build}"

export NFN_TILE_CUDA_USE_TK_ATTENTION="${NFN_TILE_CUDA_USE_TK_ATTENTION:-1}"
export NFN_TILE_CUDA_ARCH="${NFN_TILE_CUDA_ARCH:-sm_120a}"
export NFN_NATIVE_FORCE_REBUILD="${NFN_NATIVE_FORCE_REBUILD:-1}"

mkdir -p "${OUT_DIR}"

printf 'Rebuilding SM120 native CUDA artifacts into %s\n' "${OUT_DIR}"
printf '  NVCC=%s\n' "${NVCC:-nvcc}"
printf '  CXX=%s\n' "${CXX:-c++}"
printf '  NFN_TILE_CUDA_ARCH=%s\n' "${NFN_TILE_CUDA_ARCH}"
printf '  NFN_TILE_CUDA_USE_TK_ATTENTION=%s\n' "${NFN_TILE_CUDA_USE_TK_ATTENTION}"
printf '  NFN_NATIVE_REBUILD_BINDINGS=%s\n' "${NFN_NATIVE_REBUILD_BINDINGS:-1}"

case "${NFN_NATIVE_REBUILD_BINDINGS:-1}" in
  1|true|TRUE|yes|YES|on|ON)
    bash "${ROOT_DIR}/tools/build_native_gpt_binding.sh"
    bash "${ROOT_DIR}/tools/build_native_gpt2_binding.sh"
    bash "${ROOT_DIR}/tools/build_native_train_binding.sh"
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "Unsupported NFN_NATIVE_REBUILD_BINDINGS value: ${NFN_NATIVE_REBUILD_BINDINGS}" >&2
    exit 2
    ;;
esac

NFN_NATIVE_TRAIN_TILE_OPS_OUT="${OUT_DIR}/libnfn_native_train_tile_ops.so" \
  bash "${ROOT_DIR}/tools/build_native_train_tile_ops.sh"
# The default SM120 library already carries the TK fused dGELU dInput flags.
# Keep this sidecar for stale-artifact checks and intentional candidate replay.
NFN_NATIVE_TRAIN_TILE_OPS_OUT="${OUT_DIR}/libnfn_native_train_tile_ops_tk.so" \
NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="${NFN_TILE_CUDA_TK_EXTRA_NVCC_FLAGS:--DLLMK_SM120_USE_TK_FUSED_DGELU_DINP -DLLMK_SM120_APPROX_DGELU_TANH=1}" \
  bash "${ROOT_DIR}/tools/build_native_train_tile_ops.sh"

bash "${ROOT_DIR}/tools/build_native_gpt_cli.sh" "${OUT_DIR}/nfn_gpt_native_train"
bash "${ROOT_DIR}/tools/build_native_gpt_cli_linked.sh" "${OUT_DIR}/nfn_gpt_native_train_linked"
bash "${ROOT_DIR}/tools/build_native_gpt2_cli.sh" "${OUT_DIR}/nfn_gpt2_native_train"
bash "${ROOT_DIR}/tools/build_native_train_cli.sh" "${OUT_DIR}/nfn_native_train"
bash "${ROOT_DIR}/tools/build_native_gpt2_launcher.sh" "${OUT_DIR}/nfn_gpt2_tile_train"
bash "${ROOT_DIR}/tools/build_linear_backward_bench.sh" "${OUT_DIR}/linear_backward_bench"
bash "${ROOT_DIR}/tools/build_lm_head_backward_bench.sh" "${OUT_DIR}/lm_head_backward_bench"
bash "${ROOT_DIR}/tools/build_native_missing_trainers.sh" "${OUT_DIR}"

printf 'SM120 native rebuild complete.\n'
