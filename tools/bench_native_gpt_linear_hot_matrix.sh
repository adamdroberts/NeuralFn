#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINEAR_WRAPPER="${ROOT_DIR}/tools/bench_linear_backward_candidate.sh"
JSON_OUT="${NFN_LINEAR_HOT_MATRIX_JSON_OUT:-/tmp/nfn_linear_hot_matrix.json}"
PROFILE_DIR="${NFN_LINEAR_HOT_MATRIX_PROFILE_DIR:-/tmp/nfn_linear_hot_matrix_profiles}"
MAX_RATIO="${NFN_LINEAR_HOT_MATRIX_MAX_RATIO:-${NFN_LINEAR_BACKWARD_MAX_RATIO:-}}"
DRY_RUN="${NFN_LINEAR_HOT_MATRIX_DRY_RUN:-0}"

# Candidate overrides:
#   NFN_LINEAR_HOT_DINPUT_CANDIDATE_SYMBOL=...
#   NFN_LINEAR_HOT_DWEIGHT_CANDIDATE_SYMBOL=...
#   NFN_LINEAR_HOT_MLP_PROJ_DINPUT_CANDIDATE_SYMBOL=...
# Baseline overrides follow the same pattern with BASELINE_SYMBOL.
DEFAULT_PROFILES=(
  mlp-proj-dinput
  mlp-proj-dweight
  mlp-fc-dinput
  mlp-fc-dweight
  qkv-dinput
  qkv-dweight
  attn-proj-dinput
  attn-proj-dweight
  lm-head-dinput
  lm-head-dweight
)

profile_env_name() {
  local profile="$1"
  printf '%s' "${profile^^}" | tr '-' '_'
}

profile_operation() {
  case "$1" in
    *-dinput)
      printf '%s\n' "dinput"
      ;;
    *-dweight)
      printf '%s\n' "dweight"
      ;;
    *)
      echo "Unknown matrix profile '$1'" >&2
      exit 2
      ;;
  esac
}

if [[ -n "${NFN_LINEAR_HOT_MATRIX_PROFILES:-}" ]]; then
  read -r -a PROFILES <<<"${NFN_LINEAR_HOT_MATRIX_PROFILES}"
else
  PROFILES=("${DEFAULT_PROFILES[@]}")
fi

mkdir -p "${PROFILE_DIR}"

RESULT_FILES=()
FAILED=0

for PROFILE in "${PROFILES[@]}"; do
  OP_KIND="$(profile_operation "${PROFILE}")"
  PROFILE_ENV="$(profile_env_name "${PROFILE}")"
  PROFILE_JSON="${PROFILE_DIR}/${PROFILE}.json"
  RESULT_FILES+=("${PROFILE_JSON}")

  CANDIDATE_ENV_NAME="NFN_LINEAR_HOT_${PROFILE_ENV}_CANDIDATE_SYMBOL"
  BASELINE_ENV_NAME="NFN_LINEAR_HOT_${PROFILE_ENV}_BASELINE_SYMBOL"
  OP_CANDIDATE_ENV_NAME="NFN_LINEAR_HOT_${OP_KIND^^}_CANDIDATE_SYMBOL"
  OP_BASELINE_ENV_NAME="NFN_LINEAR_HOT_${OP_KIND^^}_BASELINE_SYMBOL"

  CANDIDATE_SYMBOL="${!CANDIDATE_ENV_NAME:-${!OP_CANDIDATE_ENV_NAME:-}}"
  BASELINE_SYMBOL="${!BASELINE_ENV_NAME:-${!OP_BASELINE_ENV_NAME:-}}"

  ENV_ARGS=(
    "NFN_LINEAR_BACKWARD_PROFILE=${PROFILE}"
    "NFN_LINEAR_BACKWARD_JSON_OUT=${PROFILE_JSON}"
  )
  if [[ -n "${CANDIDATE_SYMBOL}" ]]; then
    ENV_ARGS+=("NFN_LINEAR_BACKWARD_CANDIDATE_SYMBOL=${CANDIDATE_SYMBOL}")
  fi
  if [[ -n "${BASELINE_SYMBOL}" ]]; then
    ENV_ARGS+=("NFN_LINEAR_BACKWARD_BASELINE_SYMBOL=${BASELINE_SYMBOL}")
  fi
  if [[ -n "${MAX_RATIO}" ]]; then
    ENV_ARGS+=("NFN_LINEAR_BACKWARD_MAX_RATIO=${MAX_RATIO}")
  fi

  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
    printf 'profile=%s operation=%s' "${PROFILE}" "${OP_KIND}"
    for ENV_ARG in "${ENV_ARGS[@]}"; do
      printf ' %q' "${ENV_ARG}"
    done
    printf ' bash %q\n' "${LINEAR_WRAPPER}"
    continue
  fi

  if ! env "${ENV_ARGS[@]}" bash "${LINEAR_WRAPPER}"; then
    FAILED=1
  fi
done

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  exit 0
fi

python - "${JSON_OUT}" "${RESULT_FILES[@]}" <<'PY'
import json
import pathlib
import statistics
import sys

out_path = pathlib.Path(sys.argv[1])
entries = []
for raw_path in sys.argv[2:]:
    path = pathlib.Path(raw_path)
    if path.exists():
        payload = json.loads(path.read_text())
        payload["profile"] = path.stem
        entries.append(payload)

ratios = [
    float(entry["candidate_to_baseline_ms_per_iter_ratio"])
    for entry in entries
]
summary = {
    "benchmark": "native_gpt_linear_hot_matrix",
    "profile_count": len(entries),
    "profiles": [entry["profile"] for entry in entries],
    "max_candidate_to_baseline_ms_per_iter_ratio": max(ratios) if ratios else None,
    "mean_candidate_to_baseline_ms_per_iter_ratio": statistics.fmean(ratios) if ratios else None,
    "results": entries,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

exit "${FAILED}"
