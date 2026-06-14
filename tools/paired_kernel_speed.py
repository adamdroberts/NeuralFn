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
import shlex
import subprocess
import time
from dataclasses import dataclass
from statistics import mean, median
from typing import Sequence


@dataclass(frozen=True)
class TimedCommand:
    name: str
    argv: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Older/baseline command, shell-quoted as one string.")
    parser.add_argument("--candidate", required=True, help="Candidate command, shell-quoted as one string.")
    parser.add_argument("--samples", type=int, default=5, help="Paired samples to collect.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup command pairs before measurement.")
    parser.add_argument(
        "--cuda-visible-devices",
        default="",
        help="Set CUDA_VISIBLE_DEVICES for both commands, e.g. 0 for a dedicated RTX 5090.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary.")
    parser.add_argument("--json-out", default="", help="Write the JSON payload to this file.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed commands instead of stopping at the first nonzero exit.",
    )
    return parser.parse_args()


def run_once(
    command: TimedCommand,
    *,
    continue_on_error: bool,
    env: dict[str, str] | None,
) -> dict[str, object]:
    start = time.perf_counter()
    proc = subprocess.run(
        command.argv,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    seconds = time.perf_counter() - start
    if proc.returncode != 0 and not continue_on_error:
        raise SystemExit(
            f"{command.name} failed with exit {proc.returncode}\n"
            f"command: {shlex.join(command.argv)}\n"
            f"stderr:\n{proc.stderr}"
        )
    return {
        "name": command.name,
        "argv": command.argv,
        "seconds": seconds,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


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
            "--query-gpu=index,name,uuid,pci.bus_id,utilization.gpu,memory.used,memory.total",
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


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    baseline = TimedCommand("baseline", shlex.split(args.baseline))
    candidate = TimedCommand("candidate", shlex.split(args.candidate))
    samples = max(1, args.samples)
    warmup = max(0, args.warmup)
    cuda_visible_devices = str(args.cuda_visible_devices or "").strip()
    run_env = None
    if cuda_visible_devices:
        run_env = dict(os.environ)
        run_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    gpu_before = gpu_snapshot()
    for warmup_index in range(warmup):
        for command in ordered_pair(warmup_index, baseline, candidate):
            run_once(command, continue_on_error=args.continue_on_error, env=run_env)

    sample_rows: list[dict[str, object]] = []
    baseline_seconds: list[float] = []
    candidate_seconds: list[float] = []
    ratios: list[float] = []
    for sample_index in range(samples):
        order_names: list[str] = []
        row: dict[str, object] = {"sample": sample_index + 1, "order": order_names}
        by_name: dict[str, dict[str, object]] = {}
        for command in ordered_pair(sample_index, baseline, candidate):
            order_names.append(command.name)
            result = run_once(command, continue_on_error=args.continue_on_error, env=run_env)
            by_name[command.name] = result
        baseline_time = float(by_name["baseline"]["seconds"])
        candidate_time = float(by_name["candidate"]["seconds"])
        baseline_seconds.append(baseline_time)
        candidate_seconds.append(candidate_time)
        ratios.append(candidate_time / baseline_time if baseline_time else float("inf"))
        row["baseline"] = by_name["baseline"]
        row["candidate"] = by_name["candidate"]
        row["candidate_over_baseline"] = ratios[-1]
        sample_rows.append(row)
    gpu_after = gpu_snapshot()

    return {
        "measurement": "paired_interleaved_commands",
        "samples": samples,
        "warmup": warmup,
        "cuda_visible_devices": cuda_visible_devices,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "baseline_command": baseline.argv,
        "candidate_command": candidate.argv,
        "baseline_seconds": summarize(baseline_seconds),
        "candidate_seconds": summarize(candidate_seconds),
        "candidate_over_baseline": summarize(ratios),
        "paired_samples": sample_rows,
    }


def print_text(payload: dict[str, object]) -> None:
    print("Paired kernel speed comparison")
    print(f"  measurement: {payload['measurement']}")
    print(f"  samples: {payload['samples']}")
    print(f"  warmup: {payload['warmup']}")
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
                    f"util={gpu.get('utilization.gpu_pct', '')}% "
                    f"mem={gpu.get('memory.used_mib', '')}/{gpu.get('memory.total_mib', '')} MiB"
                )
        processes = gpu_before.get("compute_processes")
        if isinstance(processes, list):
            print(f"  gpu_compute_processes_before: {len(processes)}")
    for key in ("baseline_seconds", "candidate_seconds", "candidate_over_baseline"):
        stats = payload[key]
        assert isinstance(stats, dict)
        print(
            f"  {key}: mean={stats['mean']:.6f} median={stats['median']:.6f} "
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
