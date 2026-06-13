#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/neuralfn/csrc/native_gpt2/binding.cpp"
PYTHON_BIN="${PYTHON:-python}"
CXX_BIN="${CXX:-c++}"
EXT_SUFFIX="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')"
PY_INCLUDES="$("${PYTHON_BIN}" -c 'import sysconfig; paths = sysconfig.get_paths(); include = paths["include"]; plat = paths.get("platinclude") or include; print("-I" + include + ("" if plat == include else " -I" + plat))')"
OUT="${1:-${ROOT_DIR}/neuralfn/_native_gpt2${EXT_SUFFIX}}"

mkdir -p "$(dirname "${OUT}")"
"${CXX_BIN}" -std=c++20 -O3 -Wall -Wextra -pedantic -fPIC -shared ${PY_INCLUDES} "${SRC}" -o "${OUT}"
printf '%s\n' "${OUT}"
