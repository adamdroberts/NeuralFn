#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/neuralfn/csrc/native_gpt2/nfn_gpt2_native_train.cpp"
TOKEN_SHARDS_SRC="${ROOT_DIR}/neuralfn/csrc/native_train/token_shards.cpp"
TOKEN_SHARDS_HEADER="${ROOT_DIR}/neuralfn/csrc/native_train/token_shards.h"
OUT="${1:-${ROOT_DIR}/build/nfn_gpt_native_train}"
CXX_BIN="${CXX:-c++}"
FORCE_REBUILD="${NFN_NATIVE_GPT_FORCE_REBUILD:-${NFN_NATIVE_FORCE_REBUILD:-0}}"

source_newer_than_out() {
  local source_path="$1"
  [[ "${source_path}" -nt "${OUT}" ]]
}

if [[ "${FORCE_REBUILD}" != "1" && -f "${OUT}" ]]; then
  if ! source_newer_than_out "${SRC}" &&
     ! source_newer_than_out "${TOKEN_SHARDS_SRC}" &&
     ! source_newer_than_out "${TOKEN_SHARDS_HEADER}" &&
     ! source_newer_than_out "${BASH_SOURCE[0]}"; then
    printf '%s\n' "${OUT}"
    exit 0
  fi
fi

mkdir -p "$(dirname "${OUT}")"
"${CXX_BIN}" -std=c++20 -O3 -Wall -Wextra -pedantic \
  -I"${ROOT_DIR}/neuralfn/csrc/native_train" \
  "${SRC}" "${TOKEN_SHARDS_SRC}" -pthread -ldl -o "${OUT}"
printf '%s\n' "${OUT}"
