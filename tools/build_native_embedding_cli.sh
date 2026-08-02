#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/neuralfn/csrc/native_embedding/nfn_embedding_native_train.cpp"
HEADER="${ROOT_DIR}/neuralfn/csrc/native_embedding/embedding_transformer.h"
OUT="${1:-${ROOT_DIR}/build/nfn_embedding_native_train}"
OUT="$(realpath -m "${OUT}")"
MANIFEST="${OUT}.inputs.sha256"
CXX_BIN="${CXX:-c++}"

if [[ -f "${OUT}" && -f "${MANIFEST}" ]] && grep -Fq "${HEADER}" "${MANIFEST}" && sha256sum --status --check "${MANIFEST}"; then
  printf '%s\n' "${OUT}"
  exit 0
fi

mkdir -p "$(dirname "${OUT}")"
"${CXX_BIN}" -std=c++20 -O2 -Wall -Wextra -pedantic "${SRC}" -o "${OUT}"
sha256sum "${OUT}" "${SRC}" "${HEADER}" "${ROOT_DIR}/tools/build_native_embedding_cli.sh" > "${MANIFEST}.tmp"
mv "${MANIFEST}.tmp" "${MANIFEST}"
printf '%s\n' "${OUT}"
