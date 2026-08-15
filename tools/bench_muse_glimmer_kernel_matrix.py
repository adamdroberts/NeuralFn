#!/usr/bin/env python3
"""Run and rank a fail-closed Muse-Glimmer CUDA kernel matrix.

The matrix runner launches the existing end-to-end chat benchmark once per
candidate, authenticates the exact output/speculative counters, and rewrites a
JSON plus Markdown ledger after every attempt.  It is intentionally sequential:
competing CUDA processes would make the throughput ranking meaningless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


DEFAULT_OUTPUT_SHA256 = (
    "63baebaa0742852d37abf85e81c815430267789bdbb79591eb56a1e1a50b74b1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _candidate_validations(
    payload: dict[str, Any], args: argparse.Namespace
) -> list[str]:
    failures: list[str] = []
    if payload.get("output_token_ids_sha256_le_u32") != args.expected_output_sha256:
        failures.append("output-token hash mismatch")
    final_stats = payload.get("model_stats_final") or {}
    if final_stats.get("cpu_model_compute_rows") != 0:
        failures.append("CPU model compute was nonzero")
    if final_stats.get("num_layers") != 52:
        failures.append("target is not the full 52-layer model")
    trials = payload.get("trials") or []
    if len(trials) != args.repetitions:
        failures.append(
            f"expected {args.repetitions} trials, observed {len(trials)}"
        )
    for index, trial in enumerate(trials):
        session = trial.get("session_stats") or {}
        gates = {
            "accepted": (
                trial.get("speculative_accepted_tokens"), args.expected_accepted
            ),
            "proposed": (
                trial.get("speculative_proposed_tokens"), args.expected_proposed
            ),
            "target rows": (
                trial.get("speculative_target_rows"), args.expected_target_rows
            ),
            "blocks": (session.get("speculative_blocks"), args.expected_blocks),
        }
        for label, (observed, expected) in gates.items():
            if observed != expected:
                failures.append(
                    f"trial {index + 1} {label}: expected {expected}, observed {observed}"
                )
    return failures


def _recompute_progress(
    ledger: dict[str, Any],
    *,
    minimum_improvement: float,
    minimum_confirmation_repetitions: int = 5,
) -> tuple[float | None, int]:
    """Annotate confirmed bests and the consecutive no-gain candidate streak."""
    best: float | None = None
    no_improvement_streak = 0
    for entry in sorted(ledger.get("attempts", []), key=lambda item: item["attempt"]):
        score = entry.get("median_tokens_per_second")
        samples = entry.get("tokens_per_second") or []
        valid_score = (
            score
            if entry.get("valid") and isinstance(score, (int, float))
            else None
        )
        confirmed = (
            valid_score is not None
            and len(samples) >= minimum_confirmation_repetitions
        )
        entry["best_before_tokens_per_second"] = best
        entry["confirmed_for_improvement"] = confirmed
        improved = confirmed and (
            best is None or valid_score > best + minimum_improvement
        )
        if improved:
            best = float(valid_score)
            no_improvement_streak = 0
        else:
            no_improvement_streak += 1
        entry["improved_best"] = improved
        entry["best_after_tokens_per_second"] = best
        entry["consecutive_runs_without_improvement"] = no_improvement_streak
    ledger["best_tokens_per_second"] = best
    ledger["consecutive_runs_without_improvement"] = no_improvement_streak
    ledger["minimum_improvement_tokens_per_second"] = minimum_improvement
    ledger["minimum_confirmation_repetitions"] = minimum_confirmation_repetitions
    return best, no_improvement_streak


def _write_markdown(path: Path, ledger: dict[str, Any]) -> None:
    attempts = ledger.get("attempts", [])
    valid = sorted(
        (entry for entry in attempts if entry.get("valid")),
        key=lambda entry: entry.get("median_tokens_per_second", 0.0),
        reverse=True,
    )
    rank = {entry["attempt"]: index + 1 for index, entry in enumerate(valid)}
    best = ledger.get("best_tokens_per_second")
    best_text = f"{best:.3f}" if isinstance(best, (int, float)) else "none"
    lines = [
        "# Muse-Glimmer CUDA kernel matrix",
        "",
        (
            f"Completed {len(attempts)} runs; {len(valid)} passed the exact-output "
            "and full-model gates."
        ),
        (
            f"Current best: {best_text} tok/s; "
            f"consecutive runs without improvement: "
            f"{ledger.get('consecutive_runs_without_improvement', 0)}."
        ),
        "",
        (
            "| Rank | Run | Candidate | Median tok/s | Range tok/s | Exact | "
            "Confirmed | New best | No-gain streak | Combination |"
        ),
        "| ---: | ---: | --- | ---: | --- | :---: | :---: | :---: | ---: | --- |",
    ]
    for entry in sorted(attempts, key=lambda item: item["attempt"]):
        samples = entry.get("tokens_per_second", [])
        sample_range = (
            f"{min(samples):.3f}–{max(samples):.3f}" if samples else "—"
        )
        median = entry.get("median_tokens_per_second")
        median_text = f"{median:.3f}" if isinstance(median, (int, float)) else "—"
        env = entry.get("environment") or {}
        combination = ", ".join(f"{key}={value}" for key, value in sorted(env.items()))
        if not combination:
            combination = entry.get("notes") or "default"
        lines.append(
            (
                "| {rank} | {attempt} | `{name}` | {median} | "
                "{sample_range} | {exact} | {confirmed} | {improved} | "
                "{streak} | {combo} |"
            ).format(
                rank=rank.get(entry["attempt"], "—"),
                attempt=entry["attempt"],
                name=entry["name"],
                median=median_text,
                sample_range=sample_range,
                exact="yes" if entry.get("valid") else "no",
                confirmed="yes" if entry.get("confirmed_for_improvement") else "no",
                improved="yes" if entry.get("improved_best") else "no",
                streak=entry.get("consecutive_runs_without_improvement", "—"),
                combo=combination.replace("|", "\\|"),
            )
        )
    failures = [entry for entry in attempts if not entry.get("valid")]
    if failures:
        lines.extend(["", "## Rejections", ""])
        for entry in failures:
            reasons = "; ".join(entry.get("failures") or ["benchmark failed"])
            lines.append(f"- Attempt {entry['attempt']} `{entry['name']}`: {reasons}.")
    path.write_text("\n".join(lines) + "\n")


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    candidates = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("kernel matrix manifest must contain a nonempty candidates list")
    names: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("name"), str):
            raise ValueError("each kernel candidate needs a string name")
        if candidate["name"] in names:
            raise ValueError(f"duplicate kernel candidate name: {candidate['name']}")
        names.add(candidate["name"])
        environment = candidate.get("env", {})
        if not isinstance(environment, dict) or any(
            not isinstance(key, str) or (value is not None and not isinstance(value, str))
            for key, value in environment.items()
        ):
            raise ValueError(f"candidate {candidate['name']} has an invalid env object")
    return candidates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--binding-lib", type=Path, required=True)
    parser.add_argument("--default-tile-ops-lib", type=Path, required=True)
    parser.add_argument("--cuda-runtime-lib", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--benchmark-script",
        type=Path,
        default=Path(__file__).with_name("bench_muse_glimmer_native_chat.py"),
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--ledger-json", type=Path)
    parser.add_argument("--ledger-markdown", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--no-improvement-limit",
        type=int,
        default=50,
        help=(
            "stop after this many consecutive full-model runs without a new "
            "best; zero disables the saturation stop"
        ),
    )
    parser.add_argument(
        "--minimum-improvement-tokens-per-second",
        type=float,
        default=0.0,
        help="minimum strict increase required to reset the saturation streak",
    )
    parser.add_argument(
        "--minimum-confirmation-repetitions",
        type=int,
        default=5,
        help=(
            "measured repetitions required before a faster exact candidate "
            "can reset the no-improvement streak; screening runs remain ranked"
        ),
    )
    parser.add_argument("--expected-output-sha256", default=DEFAULT_OUTPUT_SHA256)
    parser.add_argument("--expected-accepted", type=int, default=28)
    parser.add_argument("--expected-proposed", type=int, default=34)
    parser.add_argument("--expected-target-rows", type=int, default=37)
    parser.add_argument("--expected-blocks", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions <= 0 or args.warmups < 0 or args.max_new_tokens <= 0:
        parser.error("repetitions/max-new-tokens must be positive and warmups nonnegative")
    if args.no_improvement_limit < 0:
        parser.error("no-improvement-limit must be nonnegative")
    if args.minimum_improvement_tokens_per_second < 0.0:
        parser.error("minimum-improvement-tokens-per-second must be nonnegative")
    if args.minimum_confirmation_repetitions <= 0:
        parser.error("minimum-confirmation-repetitions must be positive")
    return args


def main() -> int:
    args = _parse_args()
    candidates = _load_manifest(args.manifest)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    ledger_json = args.ledger_json or args.results_dir / "kernel-matrix.json"
    ledger_markdown = args.ledger_markdown or args.results_dir / "kernel-matrix.md"
    if ledger_json.exists() and not args.rerun:
        ledger = json.loads(ledger_json.read_text())
    else:
        ledger = {
            "schema": "neuralfn.muse_glimmer_cuda_kernel_matrix",
            "version": 1,
            "created_unix_seconds": time.time(),
            "expected": {
                "output_token_ids_sha256_le_u32": args.expected_output_sha256,
                "accepted_tokens": args.expected_accepted,
                "proposed_tokens": args.expected_proposed,
                "target_rows": args.expected_target_rows,
                "speculative_blocks": args.expected_blocks,
                "target_layers": 52,
                "cpu_model_compute_rows": 0,
            },
            "attempts": [],
        }
    completed = {entry["attempt"] for entry in ledger.get("attempts", [])}
    _, no_improvement_streak = _recompute_progress(
        ledger,
        minimum_improvement=args.minimum_improvement_tokens_per_second,
        minimum_confirmation_repetitions=args.minimum_confirmation_repetitions,
    )
    stop = args.stop or len(candidates)
    selected = [
        (index, candidate)
        for index, candidate in enumerate(candidates, start=1)
        if args.start <= index <= stop
    ]
    for attempt, candidate in selected:
        if (
            args.no_improvement_limit
            and no_improvement_streak >= args.no_improvement_limit
        ):
            print(
                f"saturation stop: {no_improvement_streak} consecutive runs "
                "without improvement",
                flush=True,
            )
            break
        if attempt in completed and not args.rerun:
            print(
                f"[{attempt}/{len(candidates)}] {candidate['name']}: "
                "already recorded",
                flush=True,
            )
            continue
        tile_ops = Path(candidate.get("tile_ops_lib") or args.default_tile_ops_lib)
        binding_lib = Path(candidate.get("binding_lib") or args.binding_lib)
        output_path = args.results_dir / f"attempt-{attempt:02d}-{candidate['name']}.json"
        environment = os.environ.copy()
        declared_environment = candidate.get("env", {})
        for key, value in declared_environment.items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        command = [
            str(args.python),
            str(args.benchmark_script),
            "--artifact", str(args.artifact),
            "--binding-lib", str(binding_lib),
            "--tile-ops-lib", str(tile_ops),
            "--cuda-runtime-lib", str(args.cuda_runtime_lib),
            "--cuda-device", str(args.cuda_device),
            "--weight-precision", "k-quant-17gb",
            "--dflash",
            "--compute-mode", "strict",
            "--max-new-tokens", str(args.max_new_tokens),
            "--warmups", str(args.warmups),
            "--repetitions", str(args.repetitions),
            "--json-out", str(output_path),
        ]
        print(f"[{attempt}/{len(candidates)}] {candidate['name']}: running", flush=True)
        started = time.monotonic()
        process = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        elapsed = time.monotonic() - started
        failures: list[str] = []
        payload: dict[str, Any] = {}
        if process.returncode != 0:
            failures.append(f"benchmark exited {process.returncode}")
        elif not output_path.exists():
            failures.append("benchmark did not write JSON")
        else:
            try:
                payload = json.loads(output_path.read_text())
                failures.extend(_candidate_validations(payload, args))
            except (OSError, ValueError, TypeError) as error:
                failures.append(f"could not parse benchmark JSON: {error}")
        samples = (payload.get("decode_tokens_per_second") or {}).get("samples") or []
        numeric_samples = [float(value) for value in samples]
        entry = {
            "attempt": attempt,
            "name": candidate["name"],
            "category": candidate.get("category", "unspecified"),
            "notes": candidate.get("notes", ""),
            "environment": declared_environment,
            "tile_ops_lib": str(tile_ops),
            "tile_ops_sha256": _sha256(tile_ops) if tile_ops.is_file() else None,
            "binding_lib": str(binding_lib),
            "binding_sha256": (
                _sha256(binding_lib) if binding_lib.is_file() else None
            ),
            "result_json": str(output_path),
            "elapsed_seconds": elapsed,
            "returncode": process.returncode,
            "valid": not failures,
            "failures": failures,
            "tokens_per_second": numeric_samples,
            "median_tokens_per_second": (
                statistics.median(numeric_samples) if numeric_samples else None
            ),
            "mean_tokens_per_second": (
                statistics.mean(numeric_samples) if numeric_samples else None
            ),
            "output_token_ids_sha256_le_u32": payload.get(
                "output_token_ids_sha256_le_u32"
            ),
            "stderr_tail": process.stderr[-4000:],
        }
        ledger["attempts"] = [
            old for old in ledger.get("attempts", []) if old["attempt"] != attempt
        ] + [entry]
        ledger["attempts"].sort(key=lambda item: item["attempt"])
        _, no_improvement_streak = _recompute_progress(
            ledger,
            minimum_improvement=args.minimum_improvement_tokens_per_second,
            minimum_confirmation_repetitions=args.minimum_confirmation_repetitions,
        )
        ledger["updated_unix_seconds"] = time.time()
        _atomic_json(ledger_json, ledger)
        _write_markdown(ledger_markdown, ledger)
        score = entry["median_tokens_per_second"]
        score_text = f"{score:.3f} tok/s" if score is not None else "no score"
        status = "exact" if entry["valid"] else "rejected"
        progress = "new best" if entry["improved_best"] else (
            f"no-gain streak {entry['consecutive_runs_without_improvement']}"
        )
        print(
            f"[{attempt}/{len(candidates)}] {candidate['name']}: "
            f"{score_text}, {status}, {progress}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
