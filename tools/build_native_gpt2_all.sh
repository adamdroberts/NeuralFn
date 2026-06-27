#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPT_BINDING_OUT="${NFN_NATIVE_GPT_BINDING_OUT:-}"
GPT2_BINDING_OUT="${NFN_NATIVE_GPT2_BINDING_OUT:-}"
NATIVE_TRAIN_BINDING_OUT="${NFN_NATIVE_TRAIN_BINDING_OUT:-}"
LAUNCHER_OUT="${NFN_NATIVE_GPT2_LAUNCHER_OUT:-}"
GPT_CLI_OUT="${NFN_NATIVE_GPT_CLI_OUT:-}"
GPT_LINKED_CLI_OUT="${NFN_NATIVE_GPT_LINKED_CLI_OUT:-}"
GPT_TRAIN_CLI_OUT="${NFN_NATIVE_GPT_TRAIN_CLI_OUT:-}"
SM120_CLI_OUT="${NFN_NATIVE_SM120_CLI_OUT:-}"
CLI_OUT="${NFN_NATIVE_GPT2_CLI_OUT:-}"
NATIVE_TRAIN_OUT="${NFN_NATIVE_TRAIN_CLI_OUT:-}"
MISSING_TRAINERS_OUT_DIR="${NFN_NATIVE_MISSING_TRAINERS_OUT_DIR:-}"
NATIVE_TILE_OPS_OUT="${NFN_NATIVE_TRAIN_TILE_OPS_OUT:-}"

if [[ -n "${GPT_BINDING_OUT}" ]]; then
  bash "${ROOT_DIR}/tools/build_native_gpt_binding.sh" "${GPT_BINDING_OUT}"
else
  bash "${ROOT_DIR}/tools/build_native_gpt_binding.sh"
fi

if [[ -n "${GPT2_BINDING_OUT}" ]]; then
  bash "${ROOT_DIR}/tools/build_native_gpt2_binding.sh" "${GPT2_BINDING_OUT}"
else
  bash "${ROOT_DIR}/tools/build_native_gpt2_binding.sh"
fi

if [[ -n "${NATIVE_TRAIN_BINDING_OUT}" ]]; then
  bash "${ROOT_DIR}/tools/build_native_train_binding.sh" "${NATIVE_TRAIN_BINDING_OUT}"
else
  bash "${ROOT_DIR}/tools/build_native_train_binding.sh"
fi

if [[ -n "${LAUNCHER_OUT}" ]]; then
  bash "${ROOT_DIR}/tools/build_native_gpt2_launcher.sh" "${LAUNCHER_OUT}"
else
  bash "${ROOT_DIR}/tools/build_native_gpt2_launcher.sh"
fi

if [[ -n "${NATIVE_TILE_OPS_OUT}" ]]; then
  bash "${ROOT_DIR}/tools/build_native_train_tile_ops.sh" "${NATIVE_TILE_OPS_OUT}"
else
  bash "${ROOT_DIR}/tools/build_native_train_tile_ops.sh"
fi

if [[ -n "${GPT_CLI_OUT}" ]]; then
  bash "${ROOT_DIR}/tools/build_native_gpt_cli.sh" "${GPT_CLI_OUT}"
else
  bash "${ROOT_DIR}/tools/build_native_gpt_cli.sh"
fi

if [[ -n "${GPT_LINKED_CLI_OUT}" ]]; then
  bash "${ROOT_DIR}/tools/build_native_gpt_cli_linked.sh" "${GPT_LINKED_CLI_OUT}"
else
  bash "${ROOT_DIR}/tools/build_native_gpt_cli_linked.sh"
fi

if [[ -n "${GPT_TRAIN_CLI_OUT}" ]]; then
  bash "${ROOT_DIR}/tools/build_train_gpt_cli.sh" "${GPT_TRAIN_CLI_OUT}"
else
  bash "${ROOT_DIR}/tools/build_train_gpt_cli.sh"
fi

if [[ -n "${SM120_CLI_OUT}" ]]; then
  bash "${ROOT_DIR}/tools/build_train_gpt_sm120_cli.sh" "${SM120_CLI_OUT}"
else
  bash "${ROOT_DIR}/tools/build_train_gpt_sm120_cli.sh"
fi

if [[ -n "${CLI_OUT}" ]]; then
  bash "${ROOT_DIR}/tools/build_native_gpt2_cli.sh" "${CLI_OUT}"
else
  bash "${ROOT_DIR}/tools/build_native_gpt2_cli.sh"
fi

if [[ -n "${NATIVE_TRAIN_OUT}" ]]; then
  bash "${ROOT_DIR}/tools/build_native_train_cli.sh" "${NATIVE_TRAIN_OUT}"
else
  bash "${ROOT_DIR}/tools/build_native_train_cli.sh"
fi

if [[ -n "${MISSING_TRAINERS_OUT_DIR}" ]]; then
  bash "${ROOT_DIR}/tools/build_native_missing_trainers.sh" "${MISSING_TRAINERS_OUT_DIR}"
else
  bash "${ROOT_DIR}/tools/build_native_missing_trainers.sh" "${ROOT_DIR}/build"
fi
