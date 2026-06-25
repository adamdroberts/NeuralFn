#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${NFN_SM120_NATIVE_SWEEP_OUT_DIR:-${NFN_SM120_CANDIDATE_SWEEP_OUT_DIR:-/tmp/nfn_sm120_native_candidate_sweep_$(date +%Y%m%d_%H%M%S)}}"
ALLOW_FAILURES="${NFN_SM120_NATIVE_SWEEP_ALLOW_FAILURES:-${NFN_SM120_CANDIDATE_SWEEP_ALLOW_FAILURES:-0}}"

profiles=()
if [[ "$#" -gt 0 ]]; then
  profiles=("$@")
elif [[ -n "${NFN_SM120_NATIVE_SWEEP_PROFILES-}" ]]; then
  # shellcheck disable=SC2206
  profiles=($NFN_SM120_NATIVE_SWEEP_PROFILES)
elif [[ -n "${NFN_SM120_CANDIDATE_SWEEP_PROFILES-}" ]]; then
  # shellcheck disable=SC2206
  profiles=($NFN_SM120_CANDIDATE_SWEEP_PROFILES)
else
  profiles=(
    qkv_dinput_ln128
    lm_head_loss_bins
    cublaslt_grouped_probe
  )
fi

if [[ "${#profiles[@]}" -eq 0 ]]; then
  echo "No candidate profiles supplied." >&2
  exit 2
fi

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/json" "$OUT_DIR/profiles"

declare -a result_args=()
fail_count=0

echo "SM120 native GPT candidate sweep"
echo "  out_dir: $OUT_DIR"
echo "  profiles: ${profiles[*]}"

for profile in "${profiles[@]}"; do
  safe_profile="${profile//[^A-Za-z0-9_.-]/_}"
  json_out="$OUT_DIR/json/${safe_profile}.json"
  log_out="$OUT_DIR/logs/${safe_profile}.log"
  profile_dir="$OUT_DIR/profiles/${safe_profile}"
  echo
  echo "=== $profile ==="
  (
    cd "$ROOT_DIR" &&
      NFN_SM120_NATIVE_CANDIDATE_PROFILE="$profile" \
      NFN_SM120_NATIVE_JSON_OUT="$json_out" \
      NFN_SM120_NATIVE_PROFILE_DIR="$profile_dir" \
      bash tools/bench_native_gpt_sm120_candidate.sh
  ) >"$log_out" 2>&1
  status=$?
  if [[ "$status" -ne 0 ]]; then
    fail_count=$((fail_count + 1))
    echo "status: failed ($status)"
  else
    echo "status: passed"
  fi
  echo "json: $json_out"
  echo "log:  $log_out"
  result_args+=("${profile}|${status}|${json_out}|${log_out}")
done

summary_tsv="$OUT_DIR/summary.tsv"
python - "$summary_tsv" "${result_args[@]}" <<'PY'
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
rows = []
header = [
    "profile",
    "exit_status",
    "route_gate",
    "metric_gate",
    "train_loop_ratio_mean",
    "setup_ratio_mean",
    "token_init_ratio_mean",
    "total_wall_ratio_mean",
    "qkv_dinput_before_dweight",
    "lm_head_loss_bin_launches",
    "cublaslt_grouped_layout_status",
    "cublaslt_grouped_matmul_status",
    "json",
    "log",
]

def metric_mean(payload, name):
    value = payload.get("candidate_over_baseline_native_metrics", {}).get(name, {})
    if isinstance(value, dict):
        mean = value.get("mean")
        return "" if mean is None else f"{mean:.6f}"
    return ""

def native_value(payload, side, name):
    values = payload.get(f"{side}_native_metric_values", {}).get(name, [])
    if isinstance(values, list) and values:
        return str(values[-1])
    metrics = payload.get(f"{side}_native_metrics", {}).get(name, {})
    if isinstance(metrics, dict):
        mean = metrics.get("mean")
        if mean is not None:
            if float(mean).is_integer():
                return str(int(mean))
            return f"{mean:.6f}"
    return ""

def route_delta(payload, name):
    baseline = native_value(payload, "baseline", name)
    candidate = native_value(payload, "candidate", name)
    if baseline == "" and candidate == "":
        return ""
    return f"{baseline}->{candidate}"

for item in sys.argv[2:]:
    profile, status, json_path, log_path = item.split("|", 3)
    payload = {}
    path = pathlib.Path(json_path)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - diagnostic script path
            payload = {"_read_error": str(exc)}
    route_gate = payload.get("native_route_change_gate", {}).get("passed", "")
    metric_gate = payload.get("metric_ratio_gates", {}).get("passed", "")
    rows.append(
        [
            profile,
            status,
            str(route_gate).lower() if route_gate != "" else "",
            str(metric_gate).lower() if metric_gate != "" else "",
            metric_mean(payload, "train_loop_wall_ms_per_step"),
            metric_mean(payload, "setup_wall_ms"),
            metric_mean(payload, "setup.token_weight_init.total_ms"),
            metric_mean(payload, "total_wall_ms"),
            route_delta(payload, "block_backward_qkv_dinput_before_dweight_count"),
            route_delta(payload, "lm_head_classifier_loss_bin_launch_count"),
            route_delta(payload, "linear_cublaslt_grouped_layout_probe_status"),
            route_delta(payload, "linear_cublaslt_grouped_matmul_probe_status"),
            json_path,
            log_path,
        ]
    )

summary_path.write_text(
    "\n".join("\t".join(row) for row in [header, *rows]) + "\n",
    encoding="utf-8",
)
print(summary_path.read_text(encoding="utf-8"), end="")
PY

case "${ALLOW_FAILURES,,}" in
  "1"|"true"|"yes"|"on")
    exit 0
    ;;
esac

if [[ "$fail_count" -ne 0 ]]; then
  exit 1
fi
