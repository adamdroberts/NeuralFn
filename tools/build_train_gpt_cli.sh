#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/neuralfn/csrc/native_train/train_gpt_sm120.cpp"
CATALOG_HEADER="${ROOT_DIR}/neuralfn/csrc/native_train/shipped_gpt_template_presets.h"
OUT="${1:-${ROOT_DIR}/build/nfn_train_gpt}"
CXX_BIN="${CXX:-c++}"
FORCE_REBUILD="${NFN_NATIVE_GPT_FORCE_REBUILD:-${NFN_NATIVE_FORCE_REBUILD:-0}}"

if [[ "${FORCE_REBUILD}" != "1" && -f "${OUT}" && ! "${SRC}" -nt "${OUT}" && ! "${CATALOG_HEADER}" -nt "${OUT}" && ! "${BASH_SOURCE[0]}" -nt "${OUT}" ]]; then
  printf '%s\n' "${OUT}"
  exit 0
fi

mkdir -p "$(dirname "${OUT}")"
"${CXX_BIN}" -std=c++20 -O3 -Wall -Wextra -pedantic "${SRC}" -o "${OUT}"
printf '%s\n' "${OUT}"
