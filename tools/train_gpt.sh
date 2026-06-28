#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILED_GPT_LAUNCHER="${NFN_NATIVE_GPT_TRAIN_CLI:-${ROOT_DIR}/build/nfn_train_gpt}"
USE_COMPILED_GPT_LAUNCHER="${NFN_GPT_USE_COMPILED_LAUNCHER:-1}"

case "${USE_COMPILED_GPT_LAUNCHER,,}" in
  ""|"1"|"true"|"yes"|"on")
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
