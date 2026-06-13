#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_NATIVE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-native)
      BUILD_NATIVE=0
      shift
      ;;
    --help|-h)
      cat <<'EOF'
Usage: cli/install.sh [--no-native]

Installs NeuralFn in editable mode without installing Torch by default, then
builds native C++ bindings, the GPT-2 launcher/CLI, the unified native frontend,
the raw CUDA Tile trainer ops shared library, and compiled per-family native trainer entrypoints.

Options:
  --no-native   Skip native artifact builds.
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

python -m pip install -e "${ROOT_DIR}"
python -m pip install -e "${SCRIPT_DIR}"

if [[ "${BUILD_NATIVE}" == "1" ]]; then
  bash "${ROOT_DIR}/tools/build_native_gpt2_all.sh"
  bash "${ROOT_DIR}/tools/install_native_gpt2_commands.sh"
fi
