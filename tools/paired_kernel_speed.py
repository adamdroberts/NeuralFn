#!/usr/bin/env python3
"""Run paired kernel speed comparisons in one process.

This is intended for CUDA kernel experiments where external GPU load can change
over time. It alternates baseline and candidate command order across samples and
reports paired ratios instead of timing each variant in separate manual runs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Sequence


@dataclass(frozen=True)
class TimedCommand:
    name: str
    argv: list[str]
    env_overrides: dict[str, str]


NATIVE_METRIC_PATHS = (
    ("steps_completed", ("steps_completed",)),
    ("train_loop_wall_ms", ("timing", "train_loop_wall_ms")),
    ("setup_wall_ms", ("timing", "setup_wall_ms")),
    ("checkpoint_wall_ms", ("timing", "checkpoint_wall_ms")),
    ("total_wall_ms", ("timing", "total_wall_ms")),
    ("train_tokens_per_second", ("timing", "train_tokens_per_second")),
    ("linear_tk_gemm_count", ("linear_tk_gemm_count",)),
    ("linear_cublaslt_gemm_count", ("linear_cublaslt_gemm_count",)),
    ("linear_bf16_a_pack_count", ("linear_bf16_a_pack_count",)),
    ("linear_bf16_a_cache_hit_count", ("linear_bf16_a_cache_hit_count",)),
    ("attention_forward_tk_launch_count", ("attention_forward_tk_launch_count",)),
    ("attention_backward_tk_launch_count", ("attention_backward_tk_launch_count",)),
    (
        "lm_head_classifier.reference_full_bf16_logit_bytes",
        ("lm_head_classifier_strategy_contract", "reference_full_bf16_logit_bytes"),
    ),
    (
        "lm_head_classifier.native_chunk_bf16_logit_bytes",
        ("lm_head_classifier_strategy_contract", "native_chunk_bf16_logit_bytes"),
    ),
    (
        "lm_head_classifier.resident_logit_reduction_ratio",
        ("lm_head_classifier_strategy_contract", "resident_logit_reduction_ratio"),
    ),
    (
        "lm_head_classifier.native_logit_chunk_rows",
        ("lm_head_classifier_strategy_contract", "native_logit_chunk_rows"),
    ),
    (
        "lm_head_classifier.native_logit_chunk_count",
        ("lm_head_classifier_strategy_contract", "native_logit_chunk_count"),
    ),
)
NATIVE_STRATEGY_METRIC_KEYS = (
    "status",
    "selected_graph_support_status",
    "lm_head_training_logits_dtype",
    "lm_head_logits_linear_strategy",
    "lm_head_dhidden_linear_strategy",
    "lm_head_dweight_strategy",
    "block_forward_linear_strategy",
    "block_backward_input_linear_strategy",
    "block_backward_weight_linear_strategy",
    "attention_backend_strategy",
    "attention_backward_strategy",
)
NATIVE_JSON_OUT_FLAGS = ("--json-out", "--profile-json", "--stage-profile-json")

LLM_KITTENS_STEP_RE = re.compile(
    r"^step\s+\d+/\d+\s+\|.*?\|\s+"
    r"(?P<step_ms>[0-9]+(?:\.[0-9]+)?)\s+ms\s+\|\s+"
    r"(?P<mfu>[0-9]+(?:\.[0-9]+)?)%\s+bf16\s+MFU\s+\|\s+"
    r"(?P<tok_s>[0-9]+(?:\.[0-9]+)?)\s+tok/s\s*$",
    re.MULTILINE,
)
LLM_KITTENS_MEMORY_RE = re.compile(
    r"^device memory usage:\s+"
    r"(?P<used>[0-9]+)\s+MiB\s+/\s+(?P<total>[0-9]+)\s+MiB\s*$",
    re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Older/baseline command, shell-quoted as one string.")
    parser.add_argument("--candidate", required=True, help="Candidate command, shell-quoted as one string.")
    parser.add_argument(
        "--baseline-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment override applied only to the baseline command. Repeat for multiple variables.",
    )
    parser.add_argument(
        "--candidate-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment override applied only to the candidate command. Repeat for multiple variables.",
    )
    parser.add_argument("--samples", type=int, default=5, help="Paired samples to collect.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup command pairs before measurement.")
    parser.add_argument(
        "--cuda-visible-devices",
        default="auto",
        help=(
            "Set CUDA_VISIBLE_DEVICES for both commands. The default 'auto' selects an idle "
            "display-disabled NVIDIA GPU when nvidia-smi can identify one. Pass an explicit "
            "device id such as 0, or pass an empty string to leave the environment unchanged."
        ),
    )
    parser.add_argument(
        "--cuda-device-max-connections",
        default="1",
        help="Set CUDA_DEVICE_MAX_CONNECTIONS for both commands. Pass an empty string to leave it unchanged.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary.")
    parser.add_argument("--json-out", default="", help="Write the JSON payload to this file.")
    parser.add_argument(
        "--append-native-profile-json-dir",
        default="",
        help=(
            "Append a unique --profile-json PATH under this directory to native NeuralFn commands "
            "that do not already specify --json-out, --profile-json, or --stage-profile-json."
        ),
    )
    parser.add_argument(
        "--native-stage-timing",
        action="store_true",
        help=(
            "Set NFN_NATIVE_GPT_STAGE_TIMING=1 for native NeuralFn commands. "
            "Use this for attribution runs; leave it off for throughput comparisons."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed commands instead of stopping at the first nonzero exit.",
    )
    parser.add_argument(
        "--command-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Per-command timeout. The default 0 disables the timeout. With "
            "--continue-on-error, timed-out commands are recorded with timed_out=true; "
            "otherwise the run stops at the first timeout."
        ),
    )
    parser.add_argument(
        "--require-idle-selected-gpu",
        action="store_true",
        help=(
            "Abort before warmup or measured samples if nvidia-smi reports any compute "
            "process on the selected CUDA GPU."
        ),
    )
    parser.add_argument(
        "--max-selected-gpu-utilization-pct",
        type=float,
        default=-1.0,
        help=(
            "Abort before warmup or measured samples when the selected CUDA GPU's "
            "nvidia-smi utilization exceeds this percentage. Negative values disable "
            "the utilization guard."
        ),
    )
    return parser.parse_args()


def parse_env_overrides(values: Sequence[str], *, option_name: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise SystemExit(f"{option_name} expects KEY=VALUE, got {raw!r}")
        key, value = raw.split("=", 1)
        if not key:
            raise SystemExit(f"{option_name} expects a non-empty environment variable name")
        overrides[key] = value
    return overrides


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def value_at_path(payload: dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def native_metrics_from_payload(payload: dict[str, Any]) -> dict[str, float | int | str | bool]:
    metrics: dict[str, float | int | str | bool] = {}
    for name, path in NATIVE_METRIC_PATHS:
        value = value_at_path(payload, path)
        if isinstance(value, (bool, int, float, str)):
            metrics[name] = value
    for key in NATIVE_STRATEGY_METRIC_KEYS:
        value = payload.get(key)
        if isinstance(value, (bool, int, float, str)):
            metrics[key] = value
    train_loop_ms = metrics.get("train_loop_wall_ms")
    steps_completed = metrics.get("steps_completed")
    if (
        isinstance(train_loop_ms, (int, float))
        and not isinstance(train_loop_ms, bool)
        and isinstance(steps_completed, (int, float))
        and not isinstance(steps_completed, bool)
        and float(steps_completed) > 0.0
    ):
        metrics["train_loop_wall_ms_per_step"] = float(train_loop_ms) / float(steps_completed)
    timing = payload.get("timing")
    if isinstance(timing, dict):
        setup_timing = timing.get("setup_timing")
        if isinstance(setup_timing, list):
            for stage in setup_timing:
                if not isinstance(stage, dict):
                    continue
                name = stage.get("name")
                if not isinstance(name, str) or not name:
                    continue
                metric_name = name if name.startswith("setup.") else "setup." + name
                for source_key, suffix in (
                    ("total_ms", "total_ms"),
                    ("avg_ms", "avg_ms"),
                    ("count", "count"),
                ):
                    value = stage.get(source_key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metrics[f"{metric_name}.{suffix}"] = value
        stage_timing = timing.get("stage_timing")
        if isinstance(stage_timing, list):
            for stage in stage_timing:
                if not isinstance(stage, dict):
                    continue
                name = stage.get("name")
                if not isinstance(name, str) or not name:
                    continue
                metric_name = "stage." + name
                for source_key, suffix in (
                    ("total_ms", "total_ms"),
                    ("avg_ms", "avg_ms"),
                    ("count", "count"),
                ):
                    value = stage.get(source_key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metrics[f"{metric_name}.{suffix}"] = value
    return metrics


def native_json_out_path_from_argv(argv: Sequence[str]) -> Path | None:
    for index, arg in enumerate(argv):
        if arg in NATIVE_JSON_OUT_FLAGS and index + 1 < len(argv):
            return Path(argv[index + 1])
        for flag in NATIVE_JSON_OUT_FLAGS:
            prefix = flag + "="
            if arg.startswith(prefix):
                return Path(arg[len(prefix) :])
    return None


def looks_like_neuralfn_native_command(argv: Sequence[str]) -> bool:
    if not argv:
        return False
    executable = Path(argv[0]).name
    return executable in {
        "nfn_gpt_native_train",
        "nfn-native-train",
        "nfn-gpt-native",
        "nfn-gpt-native-train",
    }


def argv_with_auto_profile_json(
    argv: Sequence[str],
    *,
    command_name: str,
    profile_json_dir: Path | None,
) -> list[str]:
    next_argv = list(argv)
    if profile_json_dir is None:
        return next_argv
    if not looks_like_neuralfn_native_command(next_argv):
        return next_argv
    if native_json_out_path_from_argv(next_argv) is not None:
        return next_argv
    profile_json_dir.mkdir(parents=True, exist_ok=True)
    path = profile_json_dir / f"{command_name}_{time.time_ns()}.json"
    next_argv.extend(["--profile-json", str(path)])
    return next_argv


def command_env_with_auto_stage_timing(
    command: TimedCommand,
    *,
    env: dict[str, str] | None,
    native_stage_timing: bool,
) -> dict[str, str] | None:
    command_env = env
    if command.env_overrides:
        command_env = dict(os.environ if env is None else env)
        command_env.update(command.env_overrides)
    if native_stage_timing and looks_like_neuralfn_native_command(command.argv):
        command_env = dict(os.environ if command_env is None else command_env)
        command_env.setdefault("NFN_NATIVE_GPT_STAGE_TIMING", "1")
    return command_env


def native_metrics_from_json_out(argv: Sequence[str]) -> dict[str, float | int | str | bool]:
    path = native_json_out_path_from_argv(argv)
    if path is None or not path.exists():
        return {}
    payload = extract_json_object(path.read_text(encoding="utf-8", errors="replace"))
    return native_metrics_from_payload(payload) if payload is not None else {}


def native_metrics_from_command_output(argv: Sequence[str], stdout: str) -> dict[str, float | int | str | bool]:
    payload = extract_json_object(stdout)
    if payload is not None:
        return native_metrics_from_payload(payload)
    sidecar_metrics = native_metrics_from_json_out(argv)
    if sidecar_metrics:
        sidecar_metrics["native_metrics_source"] = "json-out"
        return sidecar_metrics
    return llm_kittens_metrics_from_stdout(stdout)


def native_metrics_from_stdout(stdout: str) -> dict[str, float | int | str | bool]:
    payload = extract_json_object(stdout)
    if payload is None:
        return llm_kittens_metrics_from_stdout(stdout)
    return native_metrics_from_payload(payload)


def llm_kittens_metrics_from_stdout(stdout: str) -> dict[str, float | int | str | bool]:
    metrics: dict[str, float | int | str | bool] = {}
    step_matches = list(LLM_KITTENS_STEP_RE.finditer(stdout))
    if step_matches:
        step_ms_values = [float(match.group("step_ms")) for match in step_matches]
        tok_s_values = [float(match.group("tok_s")) for match in step_matches]
        mfu_values = [float(match.group("mfu")) for match in step_matches]
        metrics["status"] = "llm-kittens-step-log"
        metrics["train_loop_wall_ms"] = sum(step_ms_values)
        metrics["train_loop_wall_ms_per_step"] = mean(step_ms_values)
        metrics["train_tokens_per_second"] = mean(tok_s_values)
        metrics["llm_kittens_bf16_mfu_pct"] = mean(mfu_values)
        metrics["llm_kittens_last_step_wall_ms"] = step_ms_values[-1]
        metrics["llm_kittens_last_step_tokens_per_second"] = tok_s_values[-1]
        metrics["llm_kittens_last_step_bf16_mfu_pct"] = mfu_values[-1]
        metrics["llm_kittens_step_log_count"] = len(step_matches)
    memory_match = LLM_KITTENS_MEMORY_RE.search(stdout)
    if memory_match:
        metrics["llm_kittens_device_memory_used_mib"] = int(memory_match.group("used"))
        metrics["llm_kittens_device_memory_total_mib"] = int(memory_match.group("total"))
    return metrics


def timeout_output_to_text(value: str | bytes | None) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return ""


def run_once(
    command: TimedCommand,
    *,
    continue_on_error: bool,
    env: dict[str, str] | None,
    timeout_seconds: float | None,
    profile_json_dir: Path | None,
    native_stage_timing: bool,
    gpu_before: dict[str, object] | None = None,
) -> dict[str, object]:
    start = time.perf_counter()
    run_argv = argv_with_auto_profile_json(
        command.argv,
        command_name=command.name,
        profile_json_dir=profile_json_dir,
    )
    command_env = command_env_with_auto_stage_timing(
        command,
        env=env,
        native_stage_timing=native_stage_timing,
    )
    try:
        proc = subprocess.Popen(
            run_argv,
            text=True,
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=command_env,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        seconds = time.perf_counter() - start
        stdout = timeout_output_to_text(exc.stdout)
        stderr = timeout_output_to_text(exc.stderr)
        process_returncode = -1
        if "proc" in locals():
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                proc.kill()
            try:
                killed_stdout, killed_stderr = proc.communicate(timeout=5.0)
                stdout += timeout_output_to_text(killed_stdout)
                stderr += timeout_output_to_text(killed_stderr)
            except subprocess.TimeoutExpired:
                proc.kill()
            process_returncode = proc.returncode if proc.returncode is not None else -1
        if not continue_on_error:
            raise SystemExit(
                f"{command.name} timed out after {timeout_seconds:.3f}s\n"
                f"command: {shlex.join(run_argv)}\n"
                f"stdout tail:\n{stdout[-2000:]}\n"
                f"stderr tail:\n{stderr[-2000:]}"
            ) from exc
        result = {
            "name": command.name,
            "argv": run_argv,
            "seconds": seconds,
            "returncode": -1,
            "process_returncode": process_returncode,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "native_metrics": native_metrics_from_command_output(run_argv, stdout),
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        }
        if gpu_before is not None:
            result["gpu_before"] = gpu_before
            result["gpu_after"] = gpu_snapshot()
        return result
    seconds = time.perf_counter() - start
    returncode = proc.returncode if proc.returncode is not None else -1
    if returncode != 0 and not continue_on_error:
        raise SystemExit(
            f"{command.name} failed with exit {returncode}\n"
            f"command: {shlex.join(run_argv)}\n"
            f"stderr:\n{stderr}"
        )
    result = {
        "name": command.name,
        "argv": run_argv,
        "seconds": seconds,
        "returncode": returncode,
        "timed_out": False,
        "native_metrics": native_metrics_from_command_output(run_argv, stdout),
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }
    if gpu_before is not None:
        result["gpu_before"] = gpu_before
        result["gpu_after"] = gpu_snapshot()
    return result


def ordered_pair(sample_index: int, baseline: TimedCommand, candidate: TimedCommand) -> list[TimedCommand]:
    if sample_index % 2 == 0:
        return [baseline, candidate]
    return [candidate, baseline]


def summarize(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_metric_rows(rows: Sequence[dict[str, object]], command_name: str) -> dict[str, dict[str, float]]:
    values_by_metric: dict[str, list[float]] = {}
    for row in rows:
        command = row.get(command_name)
        if not isinstance(command, dict):
            continue
        metrics = command.get("native_metrics")
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values_by_metric.setdefault(key, []).append(float(value))
    return {key: summarize(values) for key, values in values_by_metric.items() if values}


def summarize_categorical_metric_rows(rows: Sequence[dict[str, object]], command_name: str) -> dict[str, list[str]]:
    values_by_metric: dict[str, list[str]] = {}
    for row in rows:
        command = row.get(command_name)
        if not isinstance(command, dict):
            continue
        metrics = command.get("native_metrics")
        if not isinstance(metrics, dict):
            continue
        for key in NATIVE_STRATEGY_METRIC_KEYS:
            value = metrics.get(key)
            if isinstance(value, bool):
                text = "true" if value else "false"
            elif isinstance(value, (int, float)):
                text = str(value)
            elif isinstance(value, str):
                text = value
            else:
                continue
            if text not in values_by_metric.setdefault(key, []):
                values_by_metric[key].append(text)
    return {key: values for key, values in values_by_metric.items() if values}


def summarize_metric_ratios(
    rows: Sequence[dict[str, object]],
    baseline_summary: dict[str, dict[str, float]],
    candidate_summary: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    ratios_by_metric: dict[str, list[float]] = {}
    shared_metrics = set(baseline_summary).intersection(candidate_summary)
    for row in rows:
        baseline = row.get("baseline")
        candidate = row.get("candidate")
        if not isinstance(baseline, dict) or not isinstance(candidate, dict):
            continue
        baseline_metrics = baseline.get("native_metrics")
        candidate_metrics = candidate.get("native_metrics")
        if not isinstance(baseline_metrics, dict) or not isinstance(candidate_metrics, dict):
            continue
        for key in shared_metrics:
            baseline_value = baseline_metrics.get(key)
            candidate_value = candidate_metrics.get(key)
            if (
                isinstance(baseline_value, (int, float))
                and not isinstance(baseline_value, bool)
                and isinstance(candidate_value, (int, float))
                and not isinstance(candidate_value, bool)
                and float(baseline_value) != 0.0
            ):
                ratios_by_metric.setdefault(key, []).append(float(candidate_value) / float(baseline_value))
    return {key: summarize(values) for key, values in ratios_by_metric.items() if values}


def run_nvidia_smi(args: Sequence[str]) -> dict[str, object]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi not found"}
    return {
        "available": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def parse_csv_rows(output: str, columns: Sequence[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(columns):
            continue
        rows.append(dict(zip(columns, parts)))
    return rows


def gpu_snapshot() -> dict[str, object]:
    gpu_columns = [
        "index",
        "name",
        "uuid",
        "pci.bus_id",
        "display_active",
        "utilization.gpu_pct",
        "memory.used_mib",
        "memory.total_mib",
    ]
    process_columns = [
        "gpu_uuid",
        "pid",
        "process_name",
        "used_memory_mib",
    ]
    gpu_query = run_nvidia_smi(
            [
            "--query-gpu=index,name,uuid,pci.bus_id,display_active,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    process_query = run_nvidia_smi(
        [
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "gpus": parse_csv_rows(str(gpu_query.get("stdout", "")), gpu_columns)
        if gpu_query.get("returncode") == 0
        else [],
        "compute_processes": parse_csv_rows(str(process_query.get("stdout", "")), process_columns)
        if process_query.get("returncode") == 0
        else [],
        "gpu_query": gpu_query,
        "compute_process_query": process_query,
    }


def _csv_int(value: object, default: int) -> int:
    if not isinstance(value, str):
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _display_is_inactive(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return normalized in {"disabled", "no", "off", "false", "0"}


def _first_cuda_device_index(cuda_visible_devices: str) -> str:
    first = cuda_visible_devices.split(",", 1)[0].strip()
    return first


def _selected_gpu(snapshot: dict[str, object], cuda_visible_devices: str) -> dict[str, object] | None:
    selected_index = _first_cuda_device_index(cuda_visible_devices)
    if not selected_index:
        return None
    gpus = snapshot.get("gpus")
    if not isinstance(gpus, list):
        return None
    for gpu in gpus:
        if isinstance(gpu, dict) and str(gpu.get("index", "")).strip() == selected_index:
            return gpu
    return None


def _selected_gpu_uuid(snapshot: dict[str, object], cuda_visible_devices: str) -> str:
    gpu = _selected_gpu(snapshot, cuda_visible_devices)
    if not isinstance(gpu, dict):
        return ""
    return str(gpu.get("uuid", "")).strip()


def _compute_process_count(snapshot: dict[str, object], gpu_uuid: str = "") -> int:
    processes = snapshot.get("compute_processes")
    if not isinstance(processes, list):
        return 0
    if not gpu_uuid:
        return len(processes)
    count = 0
    for process in processes:
        if isinstance(process, dict) and str(process.get("gpu_uuid", "")).strip() == gpu_uuid:
            count += 1
    return count


def _compute_processes_for_gpu(snapshot: dict[str, object], gpu_uuid: str) -> list[dict[str, str]]:
    processes = snapshot.get("compute_processes")
    if not isinstance(processes, list) or not gpu_uuid:
        return []
    matched: list[dict[str, str]] = []
    for process in processes:
        if isinstance(process, dict) and str(process.get("gpu_uuid", "")).strip() == gpu_uuid:
            matched.append({str(key): str(value) for key, value in process.items()})
    return matched


def require_idle_selected_gpu(
    snapshot: dict[str, object],
    cuda_visible_devices: str,
    *,
    phase: str,
) -> None:
    if not cuda_visible_devices:
        return
    selected_gpu = _selected_gpu(snapshot, cuda_visible_devices)
    if not isinstance(selected_gpu, dict):
        raise SystemExit(
            f"--require-idle-selected-gpu could not identify CUDA device "
            f"{_first_cuda_device_index(cuda_visible_devices)!r} in nvidia-smi output during {phase}"
        )
    selected_uuid = str(selected_gpu.get("uuid", "")).strip()
    processes = _compute_processes_for_gpu(snapshot, selected_uuid)
    if not processes:
        return
    process_lines = "\n".join(
        "  "
        f"pid={process.get('pid', '')} name={process.get('process_name', '')} "
        f"used_memory_mib={process.get('used_memory_mib', '')}"
        for process in processes
    )
    raise SystemExit(
        f"--require-idle-selected-gpu found {len(processes)} compute process(es) on "
        f"CUDA device {_first_cuda_device_index(cuda_visible_devices)} "
        f"({selected_uuid}) during {phase}:\n{process_lines}"
    )


def require_selected_gpu_utilization_at_most(
    snapshot: dict[str, object],
    cuda_visible_devices: str,
    max_utilization_pct: float,
    *,
    phase: str,
) -> None:
    if max_utilization_pct < 0.0 or not cuda_visible_devices:
        return
    selected_gpu = _selected_gpu(snapshot, cuda_visible_devices)
    if not isinstance(selected_gpu, dict):
        raise SystemExit(
            f"--max-selected-gpu-utilization-pct could not identify CUDA device "
            f"{_first_cuda_device_index(cuda_visible_devices)!r} in nvidia-smi output during {phase}"
        )
    utilization_pct = float(_csv_int(selected_gpu.get("utilization.gpu_pct"), 100))
    if utilization_pct <= max_utilization_pct:
        return
    raise SystemExit(
        f"--max-selected-gpu-utilization-pct={max_utilization_pct:g} rejected CUDA device "
        f"{_first_cuda_device_index(cuda_visible_devices)} during {phase}: "
        f"nvidia-smi utilization is {utilization_pct:g}%"
    )


def enforce_selected_gpu_guards(
    snapshot: dict[str, object],
    cuda_visible_devices: str,
    *,
    require_idle: bool,
    max_utilization_pct: float,
    phase: str,
) -> None:
    if require_idle:
        require_idle_selected_gpu(snapshot, cuda_visible_devices, phase=phase)
    require_selected_gpu_utilization_at_most(
        snapshot,
        cuda_visible_devices,
        max_utilization_pct,
        phase=phase,
    )


def snapshot_and_enforce_selected_gpu_guards(
    cuda_visible_devices: str,
    *,
    require_idle: bool,
    max_utilization_pct: float,
    phase: str,
) -> dict[str, object]:
    snapshot = gpu_snapshot()
    enforce_selected_gpu_guards(
        snapshot,
        cuda_visible_devices,
        require_idle=require_idle,
        max_utilization_pct=max_utilization_pct,
        phase=phase,
    )
    return snapshot


def summarize_gpu_sample_load(
    rows: Sequence[dict[str, object]],
    cuda_visible_devices: str,
) -> dict[str, object]:
    before_util: list[float] = []
    after_util: list[float] = []
    before_mem: list[float] = []
    after_mem: list[float] = []
    total_processes_before: list[float] = []
    total_processes_after: list[float] = []
    selected_processes_before: list[float] = []
    selected_processes_after: list[float] = []
    selected_index = _first_cuda_device_index(cuda_visible_devices)
    selected_uuid = ""

    for row in rows:
        before = row.get("gpu_before")
        after = row.get("gpu_after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        before_gpu = _selected_gpu(before, cuda_visible_devices)
        after_gpu = _selected_gpu(after, cuda_visible_devices)
        if isinstance(before_gpu, dict):
            selected_uuid = selected_uuid or str(before_gpu.get("uuid", "")).strip()
            before_util.append(float(_csv_int(before_gpu.get("utilization.gpu_pct"), 0)))
            before_mem.append(float(_csv_int(before_gpu.get("memory.used_mib"), 0)))
        if isinstance(after_gpu, dict):
            selected_uuid = selected_uuid or str(after_gpu.get("uuid", "")).strip()
            after_util.append(float(_csv_int(after_gpu.get("utilization.gpu_pct"), 0)))
            after_mem.append(float(_csv_int(after_gpu.get("memory.used_mib"), 0)))
        before_uuid = selected_uuid or _selected_gpu_uuid(before, cuda_visible_devices)
        after_uuid = selected_uuid or _selected_gpu_uuid(after, cuda_visible_devices)
        total_processes_before.append(float(_compute_process_count(before)))
        total_processes_after.append(float(_compute_process_count(after)))
        if before_uuid:
            selected_processes_before.append(float(_compute_process_count(before, before_uuid)))
        if after_uuid:
            selected_processes_after.append(float(_compute_process_count(after, after_uuid)))

    summary: dict[str, object] = {
        "selected_cuda_visible_devices": cuda_visible_devices,
        "selected_gpu_index": selected_index,
        "selected_gpu_uuid": selected_uuid,
        "sample_count": len(rows),
    }
    metric_rows = (
        ("selected_gpu_utilization_before_pct", before_util),
        ("selected_gpu_utilization_after_pct", after_util),
        ("selected_gpu_memory_used_before_mib", before_mem),
        ("selected_gpu_memory_used_after_mib", after_mem),
        ("compute_process_count_before", total_processes_before),
        ("compute_process_count_after", total_processes_after),
        ("selected_gpu_compute_process_count_before", selected_processes_before),
        ("selected_gpu_compute_process_count_after", selected_processes_after),
    )
    for name, values in metric_rows:
        if values:
            summary[name] = summarize(values)
    return summary


def resolve_cuda_visible_devices(
    requested: str,
    snapshot: dict[str, object],
) -> dict[str, object]:
    requested = requested.strip()
    if requested not in {"auto", "dedicated", "dedicated-auto"}:
        return {
            "requested": requested,
            "resolved": requested,
            "mode": "explicit" if requested else "unchanged",
            "reason": "explicit CUDA_VISIBLE_DEVICES value" if requested else "explicit empty value",
        }

    gpus = snapshot.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        return {
            "requested": requested,
            "resolved": "",
            "mode": "auto-unresolved",
            "reason": "nvidia-smi did not return GPU rows",
        }

    processes = snapshot.get("compute_processes")
    busy_uuids: set[str] = set()
    if isinstance(processes, list):
        for process in processes:
            if not isinstance(process, dict):
                continue
            uuid = process.get("gpu_uuid")
            if isinstance(uuid, str) and uuid.strip():
                busy_uuids.add(uuid.strip())

    candidates: list[tuple[int, int, int, dict[str, object]]] = []
    fallback: list[tuple[int, int, int, dict[str, object]]] = []
    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        index = _csv_int(gpu.get("index"), -1)
        if index < 0:
            continue
        util = _csv_int(gpu.get("utilization.gpu_pct"), 100)
        mem_used = _csv_int(gpu.get("memory.used_mib"), 1_000_000_000)
        uuid = str(gpu.get("uuid", "")).strip()
        row = (util, mem_used, index, gpu)
        fallback.append(row)
        if _display_is_inactive(gpu.get("display_active")) and uuid not in busy_uuids:
            candidates.append(row)

    selected_pool = candidates if candidates else fallback
    if not selected_pool:
        return {
            "requested": requested,
            "resolved": "",
            "mode": "auto-unresolved",
            "reason": "no parseable nvidia-smi GPU index",
        }

    selected = sorted(selected_pool, key=lambda row: (row[0], row[1], row[2]))[0]
    mode = "auto-dedicated" if candidates else "auto-fallback"
    reason = (
        "selected lowest-utilization display-disabled GPU with no compute processes"
        if candidates
        else "no idle display-disabled GPU found; selected lowest-utilization GPU"
    )
    return {
        "requested": requested,
        "resolved": str(selected[2]),
        "mode": mode,
        "reason": reason,
        "selected_gpu": selected[3],
    }


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    baseline_env = parse_env_overrides(args.baseline_env, option_name="--baseline-env")
    candidate_env = parse_env_overrides(args.candidate_env, option_name="--candidate-env")
    baseline = TimedCommand("baseline", shlex.split(args.baseline), baseline_env)
    candidate = TimedCommand("candidate", shlex.split(args.candidate), candidate_env)
    samples = max(1, args.samples)
    warmup = max(0, args.warmup)
    timeout_seconds = float(args.command_timeout_seconds or 0.0)
    command_timeout = timeout_seconds if timeout_seconds > 0.0 else None
    profile_json_dir = (
        Path(str(args.append_native_profile_json_dir)).expanduser()
        if str(args.append_native_profile_json_dir or "").strip()
        else None
    )
    gpu_before = gpu_snapshot()
    cuda_device_selection = resolve_cuda_visible_devices(str(args.cuda_visible_devices or ""), gpu_before)
    cuda_visible_devices = str(cuda_device_selection.get("resolved", "") or "").strip()
    cuda_device_max_connections = str(args.cuda_device_max_connections or "").strip()
    max_selected_gpu_utilization_pct = float(args.max_selected_gpu_utilization_pct)
    enforce_selected_gpu_guards(
        gpu_before,
        cuda_visible_devices,
        require_idle=bool(args.require_idle_selected_gpu),
        max_utilization_pct=max_selected_gpu_utilization_pct,
        phase="initial snapshot",
    )
    run_env = None
    if cuda_visible_devices or cuda_device_max_connections:
        run_env = dict(os.environ)
        if cuda_visible_devices:
            run_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        if cuda_device_max_connections:
            run_env["CUDA_DEVICE_MAX_CONNECTIONS"] = cuda_device_max_connections

    for warmup_index in range(warmup):
        enforce_selected_gpu_guards(
            gpu_snapshot(),
            cuda_visible_devices,
            require_idle=bool(args.require_idle_selected_gpu),
            max_utilization_pct=max_selected_gpu_utilization_pct,
            phase=f"warmup pair {warmup_index + 1}",
        )
        for command in ordered_pair(warmup_index, baseline, candidate):
            command_gpu_before = snapshot_and_enforce_selected_gpu_guards(
                cuda_visible_devices,
                require_idle=bool(args.require_idle_selected_gpu),
                max_utilization_pct=max_selected_gpu_utilization_pct,
                phase=f"warmup pair {warmup_index + 1} {command.name}",
            )
            run_once(
                command,
                continue_on_error=args.continue_on_error,
                env=run_env,
                timeout_seconds=command_timeout,
                profile_json_dir=profile_json_dir,
                native_stage_timing=bool(args.native_stage_timing),
                gpu_before=command_gpu_before,
            )

    sample_rows: list[dict[str, object]] = []
    baseline_seconds: list[float] = []
    candidate_seconds: list[float] = []
    ratios: list[float] = []
    for sample_index in range(samples):
        sample_gpu_before = gpu_snapshot()
        enforce_selected_gpu_guards(
            sample_gpu_before,
            cuda_visible_devices,
            require_idle=bool(args.require_idle_selected_gpu),
            max_utilization_pct=max_selected_gpu_utilization_pct,
            phase=f"measured sample {sample_index + 1}",
        )
        order_names: list[str] = []
        row: dict[str, object] = {
            "sample": sample_index + 1,
            "order": order_names,
            "gpu_before": sample_gpu_before,
        }
        by_name: dict[str, dict[str, object]] = {}
        for command in ordered_pair(sample_index, baseline, candidate):
            order_names.append(command.name)
            command_gpu_before = snapshot_and_enforce_selected_gpu_guards(
                cuda_visible_devices,
                require_idle=bool(args.require_idle_selected_gpu),
                max_utilization_pct=max_selected_gpu_utilization_pct,
                phase=f"measured sample {sample_index + 1} {command.name}",
            )
            result = run_once(
                command,
                continue_on_error=args.continue_on_error,
                env=run_env,
                timeout_seconds=command_timeout,
                profile_json_dir=profile_json_dir,
                native_stage_timing=bool(args.native_stage_timing),
                gpu_before=command_gpu_before,
            )
            by_name[command.name] = result
        baseline_time = float(by_name["baseline"]["seconds"])
        candidate_time = float(by_name["candidate"]["seconds"])
        baseline_seconds.append(baseline_time)
        candidate_seconds.append(candidate_time)
        ratios.append(candidate_time / baseline_time if baseline_time else float("inf"))
        row["baseline"] = by_name["baseline"]
        row["candidate"] = by_name["candidate"]
        row["candidate_over_baseline"] = ratios[-1]
        row["gpu_after"] = gpu_snapshot()
        sample_rows.append(row)
    gpu_after = gpu_snapshot()
    baseline_native_metrics = summarize_metric_rows(sample_rows, "baseline")
    candidate_native_metrics = summarize_metric_rows(sample_rows, "candidate")
    baseline_native_metric_values = summarize_categorical_metric_rows(sample_rows, "baseline")
    candidate_native_metric_values = summarize_categorical_metric_rows(sample_rows, "candidate")
    gpu_sample_summary = summarize_gpu_sample_load(sample_rows, cuda_visible_devices)

    return {
        "measurement": "paired_interleaved_commands",
        "samples": samples,
        "warmup": warmup,
        "cuda_visible_devices_requested": cuda_device_selection.get("requested", ""),
        "cuda_visible_devices": cuda_visible_devices,
        "cuda_device_selection": cuda_device_selection,
        "cuda_device_max_connections": cuda_device_max_connections,
        "require_idle_selected_gpu": bool(args.require_idle_selected_gpu),
        "max_selected_gpu_utilization_pct": max_selected_gpu_utilization_pct,
        "command_timeout_seconds": timeout_seconds,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "gpu_sample_summary": gpu_sample_summary,
        "baseline_command": baseline.argv,
        "candidate_command": candidate.argv,
        "baseline_env": baseline.env_overrides,
        "candidate_env": candidate.env_overrides,
        "append_native_profile_json_dir": str(profile_json_dir) if profile_json_dir is not None else "",
        "native_stage_timing": bool(args.native_stage_timing),
        "baseline_seconds": summarize(baseline_seconds),
        "candidate_seconds": summarize(candidate_seconds),
        "candidate_over_baseline": summarize(ratios),
        "baseline_native_metrics": baseline_native_metrics,
        "candidate_native_metrics": candidate_native_metrics,
        "baseline_native_metric_values": baseline_native_metric_values,
        "candidate_native_metric_values": candidate_native_metric_values,
        "candidate_over_baseline_native_metrics": summarize_metric_ratios(
            sample_rows,
            baseline_native_metrics,
            candidate_native_metrics,
        ),
        "paired_samples": sample_rows,
    }


def print_text(payload: dict[str, object]) -> None:
    print("Paired kernel speed comparison")
    print(f"  measurement: {payload['measurement']}")
    print(f"  samples: {payload['samples']}")
    print(f"  warmup: {payload['warmup']}")
    cuda_device_selection = payload.get("cuda_device_selection")
    if isinstance(cuda_device_selection, dict):
        print(
            "  cuda_visible_devices: "
            f"requested={cuda_device_selection.get('requested', '')} "
            f"resolved={cuda_device_selection.get('resolved', '')} "
            f"mode={cuda_device_selection.get('mode', '')}"
        )
    print(f"  require_idle_selected_gpu: {payload.get('require_idle_selected_gpu', False)}")
    print(
        "  max_selected_gpu_utilization_pct: "
        f"{payload.get('max_selected_gpu_utilization_pct', -1.0)}"
    )
    baseline_env = payload.get("baseline_env")
    candidate_env = payload.get("candidate_env")
    if isinstance(baseline_env, dict) and baseline_env:
        print(f"  baseline_env: {json.dumps(baseline_env, sort_keys=True)}")
    if isinstance(candidate_env, dict) and candidate_env:
        print(f"  candidate_env: {json.dumps(candidate_env, sort_keys=True)}")
    profile_json_dir = payload.get("append_native_profile_json_dir")
    if isinstance(profile_json_dir, str) and profile_json_dir:
        print(f"  append_native_profile_json_dir: {profile_json_dir}")
    print(f"  native_stage_timing: {payload.get('native_stage_timing', False)}")
    gpu_before = payload.get("gpu_before")
    if isinstance(gpu_before, dict):
        gpus = gpu_before.get("gpus")
        if isinstance(gpus, list) and gpus:
            print("  gpu_before:")
            for gpu in gpus:
                if not isinstance(gpu, dict):
                    continue
                print(
                    "    "
                    f"index={gpu.get('index', '')} name={gpu.get('name', '')} "
                    f"display_active={gpu.get('display_active', '')} "
                    f"util={gpu.get('utilization.gpu_pct', '')}% "
                    f"mem={gpu.get('memory.used_mib', '')}/{gpu.get('memory.total_mib', '')} MiB"
                )
        processes = gpu_before.get("compute_processes")
        if isinstance(processes, list):
            print(f"  gpu_compute_processes_before: {len(processes)}")
    paired_samples = payload.get("paired_samples")
    if isinstance(paired_samples, list):
        timeout_counts = {"baseline": 0, "candidate": 0}
        sample_process_counts: list[int] = []
        for row in paired_samples:
            if not isinstance(row, dict):
                continue
            for name in timeout_counts:
                command = row.get(name)
                if isinstance(command, dict) and command.get("timed_out") is True:
                    timeout_counts[name] += 1
            gpu_before_sample = row.get("gpu_before")
            if isinstance(gpu_before_sample, dict):
                processes = gpu_before_sample.get("compute_processes")
                if isinstance(processes, list):
                    sample_process_counts.append(len(processes))
        if timeout_counts["baseline"] or timeout_counts["candidate"]:
            print(
                "  command_timeouts: "
                f"baseline={timeout_counts['baseline']} candidate={timeout_counts['candidate']}"
            )
        if sample_process_counts:
            print(
                "  gpu_compute_processes_per_sample_before: "
                f"min={min(sample_process_counts)} max={max(sample_process_counts)}"
            )
    gpu_sample_summary = payload.get("gpu_sample_summary")
    if isinstance(gpu_sample_summary, dict):
        print(
            "  gpu_sample_summary: "
            f"selected_index={gpu_sample_summary.get('selected_gpu_index', '')} "
            f"selected_uuid={gpu_sample_summary.get('selected_gpu_uuid', '')}"
        )
        for key in (
            "selected_gpu_utilization_before_pct",
            "selected_gpu_utilization_after_pct",
            "selected_gpu_memory_used_before_mib",
            "selected_gpu_memory_used_after_mib",
            "selected_gpu_compute_process_count_before",
            "selected_gpu_compute_process_count_after",
            "compute_process_count_before",
            "compute_process_count_after",
        ):
            stats = gpu_sample_summary.get(key)
            if isinstance(stats, dict):
                print(
                    f"    {key}: mean={stats['mean']:.6f} median={stats['median']:.6f} "
                    f"min={stats['min']:.6f} max={stats['max']:.6f}"
                )
    for key in ("baseline_seconds", "candidate_seconds", "candidate_over_baseline"):
        stats = payload[key]
        assert isinstance(stats, dict)
        print(
            f"  {key}: mean={stats['mean']:.6f} median={stats['median']:.6f} "
            f"min={stats['min']:.6f} max={stats['max']:.6f}"
        )
    for section in ("baseline_native_metrics", "candidate_native_metrics"):
        metrics = payload.get(section)
        if not isinstance(metrics, dict) or not metrics:
            continue
        print(f"  {section}:")
        for key in (
            "train_loop_wall_ms_per_step",
            "train_loop_wall_ms",
            "steps_completed",
            "train_tokens_per_second",
            "llm_kittens_bf16_mfu_pct",
            "llm_kittens_last_step_wall_ms",
            "llm_kittens_last_step_tokens_per_second",
            "llm_kittens_last_step_bf16_mfu_pct",
            "llm_kittens_device_memory_used_mib",
            "lm_head_classifier.reference_full_bf16_logit_bytes",
            "lm_head_classifier.native_chunk_bf16_logit_bytes",
            "lm_head_classifier.resident_logit_reduction_ratio",
            "lm_head_classifier.native_logit_chunk_rows",
            "lm_head_classifier.native_logit_chunk_count",
            "setup_wall_ms",
            "setup.float_arena_materialize.total_ms",
            "setup.uint16_arena_materialize.total_ms",
            "setup.token_weight_init.total_ms",
            "setup.zero_init.total_ms",
            "setup.block_weight_bf16_initial_refresh.total_ms",
            "checkpoint_wall_ms",
            "total_wall_ms",
            "stage.lm_head_backward.total_ms",
            "stage.block_backward.total_ms",
            "stage.block_backward.mlp_fc.total_ms",
            "stage.block_backward.attn_sdpa.total_ms",
            "stage.block_backward.qkv.total_ms",
            "stage.adamw_update.total_ms",
        ):
            stats = metrics.get(key)
            if isinstance(stats, dict):
                print(
                    f"    {key}: mean={stats['mean']:.6f} median={stats['median']:.6f} "
                    f"min={stats['min']:.6f} max={stats['max']:.6f}"
                )
    for section in ("baseline_native_metric_values", "candidate_native_metric_values"):
        values = payload.get(section)
        if not isinstance(values, dict) or not values:
            continue
        print(f"  {section}:")
        for key in NATIVE_STRATEGY_METRIC_KEYS:
            observed = values.get(key)
            if isinstance(observed, list) and observed:
                print(f"    {key}: {', '.join(str(item) for item in observed)}")
    ratios = payload.get("candidate_over_baseline_native_metrics")
    if isinstance(ratios, dict) and ratios:
        print("  candidate_over_baseline_native_metrics:")
        for key in (
            "train_loop_wall_ms_per_step",
            "train_loop_wall_ms",
            "train_tokens_per_second",
            "llm_kittens_bf16_mfu_pct",
            "lm_head_classifier.reference_full_bf16_logit_bytes",
            "lm_head_classifier.native_chunk_bf16_logit_bytes",
            "lm_head_classifier.resident_logit_reduction_ratio",
            "lm_head_classifier.native_logit_chunk_rows",
            "lm_head_classifier.native_logit_chunk_count",
            "setup_wall_ms",
            "setup.float_arena_materialize.total_ms",
            "setup.uint16_arena_materialize.total_ms",
            "setup.token_weight_init.total_ms",
            "setup.zero_init.total_ms",
            "setup.block_weight_bf16_initial_refresh.total_ms",
            "total_wall_ms",
            "stage.lm_head_backward.total_ms",
            "stage.block_backward.total_ms",
            "stage.block_backward.mlp_fc.total_ms",
            "stage.block_backward.attn_sdpa.total_ms",
            "stage.block_backward.qkv.total_ms",
            "stage.adamw_update.total_ms",
        ):
            stats = ratios.get(key)
            if isinstance(stats, dict):
                print(
                    f"    {key}: mean={stats['mean']:.6f} median={stats['median']:.6f} "
                    f"min={stats['min']:.6f} max={stats['max']:.6f}"
                )


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    if str(args.json_out or "").strip():
        output_path = Path(args.json_out).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
