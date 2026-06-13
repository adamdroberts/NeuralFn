#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
BIN_DIR="${NFN_NATIVE_GPT2_BIN_DIR:-$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_path("scripts"))')}"
NATIVE_CLI="${NFN_NATIVE_GPT2_CLI:-${ROOT_DIR}/build/nfn_gpt2_native_train}"
NATIVE_TRAIN_CLI="${NFN_NATIVE_TRAIN_CLI:-${ROOT_DIR}/build/nfn_native_train}"
LAUNCHER="${NFN_NATIVE_GPT2_LAUNCHER:-${ROOT_DIR}/build/nfn_gpt2_tile_train}"
MISSING_TRAINERS_DIR="${NFN_NATIVE_MISSING_TRAINERS_DIR:-${ROOT_DIR}/build}"
MISSING_TARGETS=(
  nfn_gpt2_evo_native_train
  nfn_nanogpt_native_train
  nfn_llama_native_train
  nfn_mixllama_native_train
  nfn_jepa_native_train
  nfn_semantic_router_moe_native_train
  nfn_deepseek_v4_native_train
)

mkdir -p "${BIN_DIR}"

if [[ ! -x "${NATIVE_CLI}" ]]; then
  echo "Native GPT-2 CLI not found or not executable: ${NATIVE_CLI}" >&2
  echo "Run: bash ${ROOT_DIR}/tools/build_native_gpt2_cli.sh" >&2
  exit 2
fi

if [[ ! -x "${NATIVE_TRAIN_CLI}" ]]; then
  echo "Unified native train CLI not found or not executable: ${NATIVE_TRAIN_CLI}" >&2
  echo "Run: bash ${ROOT_DIR}/tools/build_native_train_cli.sh" >&2
  exit 2
fi

if [[ ! -x "${LAUNCHER}" ]]; then
  echo "Native GPT-2 launcher not found or not executable: ${LAUNCHER}" >&2
  echo "Run: bash ${ROOT_DIR}/tools/build_native_gpt2_launcher.sh" >&2
  exit 2
fi

ln -sfn "${NATIVE_CLI}" "${BIN_DIR}/nfn-gpt2-native"
ln -sfn "${NATIVE_CLI}" "${BIN_DIR}/nfn-gpt2-native-train"
ln -sfn "${NATIVE_TRAIN_CLI}" "${BIN_DIR}/nfn-native-train"
ln -sfn "${LAUNCHER}" "${BIN_DIR}/nfn-gpt2-tile-launcher"
for target in "${MISSING_TARGETS[@]}"; do
  if [[ -x "${MISSING_TRAINERS_DIR}/${target}" ]]; then
    ln -sfn "${MISSING_TRAINERS_DIR}/${target}" "${BIN_DIR}/${target}"
    ln -sfn "${MISSING_TRAINERS_DIR}/${target}" "${BIN_DIR}/${target//_/-}"
  fi
done

printf '%s\n' "${BIN_DIR}/nfn-gpt2-native"
printf '%s\n' "${BIN_DIR}/nfn-gpt2-native-train"
printf '%s\n' "${BIN_DIR}/nfn-native-train"
printf '%s\n' "${BIN_DIR}/nfn-gpt2-tile-launcher"
for target in "${MISSING_TARGETS[@]}"; do
  if [[ -L "${BIN_DIR}/${target}" ]]; then
    printf '%s\n' "${BIN_DIR}/${target}"
  fi
  if [[ -L "${BIN_DIR}/${target//_/-}" ]]; then
    printf '%s\n' "${BIN_DIR}/${target//_/-}"
  fi
done
