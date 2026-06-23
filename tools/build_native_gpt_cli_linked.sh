#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/neuralfn/csrc/native_gpt2/nfn_gpt2_native_train.cpp"
TOKEN_SHARDS_SRC="${ROOT_DIR}/neuralfn/csrc/native_train/token_shards.cpp"
OUT="${1:-${ROOT_DIR}/build/nfn_gpt_native_train_linked}"
TILE_OPS_LIB="${NFN_NATIVE_TRAIN_TILE_OPS_LIB:-${ROOT_DIR}/build/libnfn_native_train_tile_ops.so}"
CXX_BIN="${CXX:-c++}"

if [[ ! -f "${TILE_OPS_LIB}" ]]; then
  bash "${ROOT_DIR}/tools/build_native_train_tile_ops.sh" "${TILE_OPS_LIB}"
fi

mkdir -p "$(dirname "${OUT}")"
"${CXX_BIN}" -std=c++20 -O3 -Wall -Wextra -pedantic \
  -I"${ROOT_DIR}/neuralfn/csrc/native_train" \
  "${SRC}" "${TOKEN_SHARDS_SRC}" \
  -rdynamic -Wl,--export-dynamic \
  -Wl,--no-as-needed "${TILE_OPS_LIB}" -Wl,--as-needed \
  -Wl,-rpath,"$(dirname "${TILE_OPS_LIB}")" \
  -ldl -o "${OUT}"
printf '%s\n' "${OUT}"
