#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
BIN_DIR="${NFN_NATIVE_GPT2_BIN_DIR:-$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_path("scripts"))')}"
if [[ -n "${NFN_NATIVE_GPT_CLI:-}" ]]; then
  NATIVE_CLI="${NFN_NATIVE_GPT_CLI}"
elif [[ -n "${NFN_NATIVE_GPT_LINKED_CLI:-}" ]]; then
  NATIVE_CLI="${NFN_NATIVE_GPT_LINKED_CLI}"
elif [[ -x "${ROOT_DIR}/build/nfn_gpt_native_train_linked" ]]; then
  NATIVE_CLI="${ROOT_DIR}/build/nfn_gpt_native_train_linked"
else
  NATIVE_CLI="${ROOT_DIR}/build/nfn_gpt_native_train"
fi
COMPAT_NATIVE_CLI="${NFN_NATIVE_GPT2_CLI:-${ROOT_DIR}/build/nfn_gpt2_native_train}"
NATIVE_TRAIN_CLI="${NFN_NATIVE_TRAIN_CLI:-${ROOT_DIR}/build/nfn_native_train}"
LAUNCHER="${NFN_NATIVE_GPT2_LAUNCHER:-${ROOT_DIR}/build/nfn_gpt2_tile_train}"
GPT_TRAIN_LAUNCHER="${NFN_NATIVE_GPT_TRAIN_CLI:-${ROOT_DIR}/build/nfn_train_gpt}"
SM120_LAUNCHER="${NFN_NATIVE_SM120_CLI:-${ROOT_DIR}/build/nfn_train_gpt_sm120}"
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
  echo "Native GPT CLI not found or not executable: ${NATIVE_CLI}" >&2
  echo "Run: bash ${ROOT_DIR}/tools/build_native_gpt_cli.sh" >&2
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

if [[ ! -x "${GPT_TRAIN_LAUNCHER}" ]]; then
  echo "Native GPT compiled launcher not found or not executable: ${GPT_TRAIN_LAUNCHER}" >&2
  echo "Run: bash ${ROOT_DIR}/tools/build_train_gpt_cli.sh" >&2
  exit 2
fi

if [[ ! -x "${SM120_LAUNCHER}" ]]; then
  echo "Native SM120 GPT launcher not found or not executable: ${SM120_LAUNCHER}" >&2
  echo "Run: bash ${ROOT_DIR}/tools/build_train_gpt_sm120_cli.sh" >&2
  exit 2
fi

ln -sfn "${NATIVE_CLI}" "${BIN_DIR}/nfn-gpt2-native"
ln -sfn "${NATIVE_CLI}" "${BIN_DIR}/nfn-gpt2-native-train"
ln -sfn "${NATIVE_CLI}" "${BIN_DIR}/nfn-gpt-native"
ln -sfn "${NATIVE_CLI}" "${BIN_DIR}/nfn-gpt-native-train"
if [[ -x "${COMPAT_NATIVE_CLI}" ]]; then
  ln -sfn "${COMPAT_NATIVE_CLI}" "${BIN_DIR}/nfn-gpt2-native-compat"
fi
ln -sfn "${NATIVE_TRAIN_CLI}" "${BIN_DIR}/nfn-native-train"
ln -sfn "${LAUNCHER}" "${BIN_DIR}/nfn-gpt2-tile-launcher"
ln -sfn "${GPT_TRAIN_LAUNCHER}" "${BIN_DIR}/nfn-train-gpt"
ln -sfn "${GPT_TRAIN_LAUNCHER}" "${BIN_DIR}/nfn-gpt-train"
ln -sfn "${SM120_LAUNCHER}" "${BIN_DIR}/nfn-train-gpt-sm120"
ln -sfn "${SM120_LAUNCHER}" "${BIN_DIR}/nfn-gpt-sm120-train"
for target in "${MISSING_TARGETS[@]}"; do
  if [[ -x "${MISSING_TRAINERS_DIR}/${target}" ]]; then
    ln -sfn "${MISSING_TRAINERS_DIR}/${target}" "${BIN_DIR}/${target}"
    ln -sfn "${MISSING_TRAINERS_DIR}/${target}" "${BIN_DIR}/${target//_/-}"
  fi
done

printf '%s\n' "${BIN_DIR}/nfn-gpt2-native"
printf '%s\n' "${BIN_DIR}/nfn-gpt2-native-train"
printf '%s\n' "${BIN_DIR}/nfn-gpt-native"
printf '%s\n' "${BIN_DIR}/nfn-gpt-native-train"
if [[ -L "${BIN_DIR}/nfn-gpt2-native-compat" ]]; then
  printf '%s\n' "${BIN_DIR}/nfn-gpt2-native-compat"
fi
printf '%s\n' "${BIN_DIR}/nfn-native-train"
printf '%s\n' "${BIN_DIR}/nfn-gpt2-tile-launcher"
printf '%s\n' "${BIN_DIR}/nfn-train-gpt"
printf '%s\n' "${BIN_DIR}/nfn-gpt-train"
printf '%s\n' "${BIN_DIR}/nfn-train-gpt-sm120"
printf '%s\n' "${BIN_DIR}/nfn-gpt-sm120-train"
for target in "${MISSING_TARGETS[@]}"; do
  if [[ -L "${BIN_DIR}/${target}" ]]; then
    printf '%s\n' "${BIN_DIR}/${target}"
  fi
  if [[ -L "${BIN_DIR}/${target//_/-}" ]]; then
    printf '%s\n' "${BIN_DIR}/${target//_/-}"
  fi
done
