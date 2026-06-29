#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILED_GPT_LAUNCHER="${NFN_NATIVE_GPT_TRAIN_CLI:-${ROOT_DIR}/build/nfn_train_gpt}"
COMPILED_GPT_LAUNCHER_EXPLICIT="${NFN_NATIVE_GPT_TRAIN_CLI:-}"
USE_COMPILED_GPT_LAUNCHER="${NFN_GPT_USE_COMPILED_LAUNCHER:-1}"
AUTO_REBUILD_NATIVE="${NFN_NATIVE_GPT_AUTO_REBUILD:-1}"

native_gpt_source_newer_than() {
  local target="$1"
  [[ "$ROOT_DIR/neuralfn/csrc/native_gpt2/nfn_gpt2_native_train.cpp" -nt "$target" ||
     "$ROOT_DIR/neuralfn/csrc/native_train/token_shards.cpp" -nt "$target" ]]
}

tile_ops_source_newer_than() {
  local target="$1"
  [[ "$ROOT_DIR/neuralfn/csrc/native_train/tile_ops.cu" -nt "$target" ||
     "$ROOT_DIR/neuralfn/csrc/native_train/tile_ops.h" -nt "$target" ||
     "$ROOT_DIR/neuralfn/csrc/tile_cuda/kernels.cu" -nt "$target" ||
     "$ROOT_DIR/tools/build_native_train_tile_ops.sh" -nt "$target" ]]
}

ensure_default_native_trainer_current() {
  case "${AUTO_REBUILD_NATIVE,,}" in
    ""|"1"|"true"|"yes"|"on")
      ;;
    "0"|"false"|"no"|"off")
      return 0
      ;;
    *)
      echo "Invalid NFN_NATIVE_GPT_AUTO_REBUILD='${AUTO_REBUILD_NATIVE}'" >&2
      exit 2
      ;;
  esac
  if [[ -z "${NFN_NATIVE_GPT_TRAIN_BIN-}" ]]; then
    local linked="${ROOT_DIR}/build/nfn_gpt_native_train_linked"
    local rebuild_linked=0
    if [[ ! -x "$linked" ]]; then
      rebuild_linked=1
    elif native_gpt_source_newer_than "$linked"; then
      rebuild_linked=1
    elif tile_ops_source_newer_than "$linked"; then
      rebuild_linked=1
    fi
    if [[ "$rebuild_linked" == "1" ]]; then
      bash "$ROOT_DIR/tools/build_native_gpt_cli_linked.sh" "$linked" >&2
    fi
  fi
}

ensure_default_compiled_launcher_current() {
  case "${AUTO_REBUILD_NATIVE,,}" in
    ""|"1"|"true"|"yes"|"on")
      ;;
    "0"|"false"|"no"|"off")
      return 0
      ;;
    *)
      echo "Invalid NFN_NATIVE_GPT_AUTO_REBUILD='${AUTO_REBUILD_NATIVE}'" >&2
      exit 2
      ;;
  esac
  if [[ -z "$COMPILED_GPT_LAUNCHER_EXPLICIT" ]]; then
    bash "$ROOT_DIR/tools/build_train_gpt_cli.sh" "$COMPILED_GPT_LAUNCHER" >/dev/null
  fi
}

case "${USE_COMPILED_GPT_LAUNCHER,,}" in
  ""|"1"|"true"|"yes"|"on")
    ensure_default_compiled_launcher_current
    ensure_default_native_trainer_current
    if [[ -x "${COMPILED_GPT_LAUNCHER}" ]]; then
      exec "${COMPILED_GPT_LAUNCHER}" "$@"
    fi
    ;;
  "0"|"false"|"no"|"off")
    ;;
  *)
    echo "Invalid NFN_GPT_USE_COMPILED_LAUNCHER='${USE_COMPILED_GPT_LAUNCHER}'" >&2
    exit 2
    ;;
esac

export NFN_SM120_USE_COMPILED_LAUNCHER=0
exec "${ROOT_DIR}/tools/train_gpt_sm120.sh" "$@"
