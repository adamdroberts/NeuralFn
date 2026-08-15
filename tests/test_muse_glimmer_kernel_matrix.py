from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bench_muse_glimmer_kernel_matrix.py"
MANIFEST = ROOT / "tools" / "muse_glimmer_kernel_matrix.json"
SPEC = importlib.util.spec_from_file_location(
    "bench_muse_glimmer_kernel_matrix_test", TOOL
)
assert SPEC is not None and SPEC.loader is not None
matrix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = matrix
SPEC.loader.exec_module(matrix)


def test_checked_in_matrix_is_unique_and_records_the_saturation_run() -> None:
    candidates = matrix._load_manifest(MANIFEST)
    assert len(candidates) == 272
    assert len({candidate["name"] for candidate in candidates}) == len(candidates)

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    qualification = payload["qualification"]
    assert qualification["retained_combination"] == {
        "packed_weight_global_load_policy": "cg",
        "mmvq_input_rows_per_block": 8,
        "mmvq_output_channels_per_group": 2,
        "q4_output_groups_per_block": 2,
        "q5_output_groups_per_block": 1,
        "q6_output_groups_per_block": 2,
        "dp4a_schedule": 2,
        "q8_readonly_cache": True,
        "q6_weight_predecode": True,
        "register_cap": None,
    }
    assert qualification["confirmed_performance"][
        "ten_run_median_tokens_per_second"
    ] > 300.0
    ranked = qualification["ranked_confirmed_combinations"]
    assert [row["rank"] for row in ranked] == list(range(1, len(ranked) + 1))
    assert ranked[0]["status"] == "retained"
    assert ranked[0]["best_median_tokens_per_second"] == max(
        row["best_median_tokens_per_second"] for row in ranked
    )
    assert qualification["saturation"] == {
        "last_confirmed_improvement_attempt": 221,
        "first_consecutive_no_improvement_attempt": 222,
        "last_required_no_improvement_attempt": 271,
        "consecutive_full_model_runs_without_confirmed_improvement": 50,
        "extra_confirmation_attempt": 272,
        "total_full_model_candidate_runs": 272,
    }


def test_exact_output_and_full_model_gates_fail_closed() -> None:
    args = SimpleNamespace(
        expected_output_sha256="canonical",
        expected_accepted=28,
        expected_proposed=34,
        expected_target_rows=37,
        expected_blocks=3,
        repetitions=1,
    )
    trial = {
        "speculative_accepted_tokens": 28,
        "speculative_proposed_tokens": 34,
        "speculative_target_rows": 37,
        "session_stats": {"speculative_blocks": 3},
    }
    payload = {
        "output_token_ids_sha256_le_u32": "canonical",
        "model_stats_final": {"cpu_model_compute_rows": 0, "num_layers": 52},
        "trials": [trial],
    }
    assert matrix._candidate_validations(payload, args) == []

    payload["output_token_ids_sha256_le_u32"] = "different"
    payload["model_stats_final"]["cpu_model_compute_rows"] = 1
    payload["model_stats_final"]["num_layers"] = 5
    failures = matrix._candidate_validations(payload, args)
    assert "output-token hash mismatch" in failures
    assert "CPU model compute was nonzero" in failures
    assert "target is not the full 52-layer model" in failures


def test_improvement_resets_a_consecutive_no_gain_streak() -> None:
    attempts = [
        {
            "attempt": 1,
            "valid": True,
            "median_tokens_per_second": 300.0,
            "tokens_per_second": [300.0],
        },
        *[
            {
                "attempt": attempt,
                "valid": True,
                "median_tokens_per_second": 299.0,
                "tokens_per_second": [299.0],
            }
            for attempt in range(2, 52)
        ],
        {
            "attempt": 52,
            "valid": True,
            "median_tokens_per_second": 301.0,
            "tokens_per_second": [301.0],
        },
        {
            "attempt": 53,
            "valid": True,
            "median_tokens_per_second": 300.5,
            "tokens_per_second": [300.5],
        },
    ]
    ledger = {"attempts": attempts}
    best, streak = matrix._recompute_progress(
        ledger,
        minimum_improvement=0.0,
        minimum_confirmation_repetitions=1,
    )
    assert attempts[50]["consecutive_runs_without_improvement"] == 50
    assert attempts[51]["improved_best"] is True
    assert attempts[51]["consecutive_runs_without_improvement"] == 0
    assert attempts[52]["consecutive_runs_without_improvement"] == 1
    assert best == 301.0
    assert streak == 1


def test_screening_peak_does_not_reset_until_repeated() -> None:
    attempts = [
        {
            "attempt": 1,
            "valid": True,
            "median_tokens_per_second": 300.0,
            "tokens_per_second": [300.0, 300.1, 299.9],
        },
        {
            "attempt": 2,
            "valid": True,
            "median_tokens_per_second": 321.0,
            "tokens_per_second": [321.0],
        },
        {
            "attempt": 3,
            "valid": True,
            "median_tokens_per_second": 301.0,
            "tokens_per_second": [300.9, 301.0, 301.1],
        },
    ]
    ledger = {"attempts": attempts}
    best, streak = matrix._recompute_progress(
        ledger,
        minimum_improvement=0.0,
        minimum_confirmation_repetitions=3,
    )
    assert attempts[1]["confirmed_for_improvement"] is False
    assert attempts[1]["improved_best"] is False
    assert attempts[1]["consecutive_runs_without_improvement"] == 1
    assert attempts[2]["confirmed_for_improvement"] is True
    assert attempts[2]["improved_best"] is True
    assert best == 301.0
    assert streak == 0


def test_markdown_ranks_exact_candidates_by_measured_median(tmp_path: Path) -> None:
    ledger = {
        "best_tokens_per_second": 317.0,
        "consecutive_runs_without_improvement": 2,
        "attempts": [
            {
                "attempt": 1,
                "name": "control",
                "valid": True,
                "median_tokens_per_second": 300.0,
                "tokens_per_second": [299.0, 301.0],
                "improved_best": True,
                "consecutive_runs_without_improvement": 0,
                "environment": {},
            },
            {
                "attempt": 2,
                "name": "retained",
                "valid": True,
                "median_tokens_per_second": 317.0,
                "tokens_per_second": [316.0, 318.0],
                "improved_best": True,
                "consecutive_runs_without_improvement": 0,
                "environment": {"Q8_CACHE": "1"},
            },
        ],
    }
    output = tmp_path / "matrix.md"
    matrix._write_markdown(output, ledger)
    markdown = output.read_text(encoding="utf-8")
    assert "| 1 | 2 | `retained` | 317.000" in markdown
    assert "Q8_CACHE=1" in markdown
