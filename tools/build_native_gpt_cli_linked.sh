#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/neuralfn/csrc/native_gpt2/nfn_gpt2_native_train.cpp"
TOKEN_SHARDS_SRC="${ROOT_DIR}/neuralfn/csrc/native_train/token_shards.cpp"
TOKEN_SHARDS_HEADER="${ROOT_DIR}/neuralfn/csrc/native_train/token_shards.h"
CATALOG_HEADER="${ROOT_DIR}/neuralfn/csrc/native_train/shipped_gpt_template_presets.h"
OUT="${1:-${ROOT_DIR}/build/nfn_gpt_native_train_linked}"
OUT="$(realpath -m "${OUT}")"
MANIFEST="${OUT}.inputs.sha256"
TILE_OPS_LIB="${NFN_NATIVE_TRAIN_TILE_OPS_LIB:-${ROOT_DIR}/build/libnfn_native_train_tile_ops.so}"
CXX_BIN="${CXX:-c++}"
CXX_OPT_FLAGS="${NFN_NATIVE_GPT_CXX_OPT_FLAGS:--O0}"
FORCE_REBUILD="${NFN_NATIVE_GPT_FORCE_REBUILD:-${NFN_NATIVE_FORCE_REBUILD:-0}}"

if [[ "${FORCE_REBUILD}" != "1" && -f "${OUT}" ]]; then
  if [[ -f "${MANIFEST}" ]] && sha256sum --status --check "${MANIFEST}"; then
    printf '%s\n' "${OUT}"
    exit 0
  fi
fi

bash "${ROOT_DIR}/tools/build_native_train_tile_ops.sh" "${TILE_OPS_LIB}"
mkdir -p "$(dirname "${OUT}")"
"${CXX_BIN}" -std=c++20 ${CXX_OPT_FLAGS} -Wall -Wextra -pedantic \
  -I"${ROOT_DIR}/neuralfn/csrc/native_train" \
  "${SRC}" "${TOKEN_SHARDS_SRC}" \
  -rdynamic -Wl,--export-dynamic \
  -Wl,--no-as-needed "${TILE_OPS_LIB}" -Wl,--as-needed \
  -Wl,-rpath,"$(dirname "${TILE_OPS_LIB}")" \
  -pthread -ldl -o "${OUT}"
sha256sum \
  "${OUT}" \
  "${SRC}" \
  "${TOKEN_SHARDS_SRC}" \
  "${TOKEN_SHARDS_HEADER}" \
  "${CATALOG_HEADER}" \
  "${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.cu" \
  "${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.h" \
  "${ROOT_DIR}/neuralfn/csrc/tile_cuda/kernels.cu" \
  "${ROOT_DIR}/tools/build_native_train_tile_ops.sh" \
  "${ROOT_DIR}/tools/build_native_gpt_cli_linked.sh" \
  "${TILE_OPS_LIB}" \
  > "${MANIFEST}.tmp"
mv "${MANIFEST}.tmp" "${MANIFEST}"
printf '%s\n' "${OUT}"
