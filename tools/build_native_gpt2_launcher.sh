#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/neuralfn/csrc/native_gpt2/nfn_gpt2_tile_train.cpp"
OUT="${1:-${ROOT_DIR}/build/nfn_gpt2_tile_train}"
CXX_BIN="${CXX:-c++}"

mkdir -p "$(dirname "${OUT}")"
"${CXX_BIN}" -std=c++20 -O3 -Wall -Wextra -pedantic "${SRC}" -o "${OUT}"
printf '%s\n' "${OUT}"
