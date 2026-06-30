#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/neuralfn/csrc/native_gpt2/nfn_gpt2_native_train.cpp"
TOKEN_SHARDS_SRC="${ROOT_DIR}/neuralfn/csrc/native_train/token_shards.cpp"
TOKEN_SHARDS_HEADER="${ROOT_DIR}/neuralfn/csrc/native_train/token_shards.h"
CATALOG_HEADER="${ROOT_DIR}/neuralfn/csrc/native_train/shipped_gpt_template_presets.h"
OUT="${1:-${ROOT_DIR}/build/nfn_gpt_native_train_linked}"
TILE_OPS_LIB="${NFN_NATIVE_TRAIN_TILE_OPS_LIB:-${ROOT_DIR}/build/libnfn_native_train_tile_ops.so}"
CXX_BIN="${CXX:-c++}"
FORCE_REBUILD="${NFN_NATIVE_GPT_FORCE_REBUILD:-${NFN_NATIVE_FORCE_REBUILD:-0}}"

if [[ ! -f "${TILE_OPS_LIB}" ||
      "${ROOT_DIR}/neuralfn/csrc/tile_cuda/kernels.cu" -nt "${TILE_OPS_LIB}" ||
      "${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.cu" -nt "${TILE_OPS_LIB}" ||
      "${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.h" -nt "${TILE_OPS_LIB}" ||
      "${ROOT_DIR}/tools/build_native_train_tile_ops.sh" -nt "${TILE_OPS_LIB}" ]]; then
  bash "${ROOT_DIR}/tools/build_native_train_tile_ops.sh" "${TILE_OPS_LIB}"
fi

source_newer_than_out() {
  local source_path="$1"
  [[ "${source_path}" -nt "${OUT}" ]]
}

if [[ "${FORCE_REBUILD}" != "1" && -f "${OUT}" ]]; then
  if ! source_newer_than_out "${SRC}" &&
     ! source_newer_than_out "${TOKEN_SHARDS_SRC}" &&
     ! source_newer_than_out "${TOKEN_SHARDS_HEADER}" &&
     ! source_newer_than_out "${CATALOG_HEADER}" &&
     ! source_newer_than_out "${TILE_OPS_LIB}" &&
     ! source_newer_than_out "${BASH_SOURCE[0]}"; then
    printf '%s\n' "${OUT}"
    exit 0
  fi
fi

mkdir -p "$(dirname "${OUT}")"
"${CXX_BIN}" -std=c++20 -O3 -Wall -Wextra -pedantic \
  -I"${ROOT_DIR}/neuralfn/csrc/native_train" \
  "${SRC}" "${TOKEN_SHARDS_SRC}" \
  -rdynamic -Wl,--export-dynamic \
  -Wl,--no-as-needed "${TILE_OPS_LIB}" -Wl,--as-needed \
  -Wl,-rpath,"$(dirname "${TILE_OPS_LIB}")" \
  -pthread -ldl -o "${OUT}"
printf '%s\n' "${OUT}"
