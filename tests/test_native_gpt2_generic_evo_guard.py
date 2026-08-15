from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
BLOCKED_STATUS = "native-preflight-blocked-graph-faithful-layer-evo-missing"
BLOCKED_ERROR = (
    "direct generic dense GPT --layer-evo execution is blocked before Tile load, "
    "CUDA setup/allocation, output creation, or training mutation because its "
    "semantics do not implement authored GPT2-Evo whole-block evolution"
)
EXPECTED_MISSING_GATES = [
    "exclude every tensor in the designated transformer block from gradient and AdamW updates",
    (
        "mutate, evaluate, select, and adopt every tensor in the designated transformer "
        "block rather than only block_N.ln1.weight"
    ),
    "checkpoint and resume the whole-block evolutionary state with graph-faithful parity",
]


@pytest.fixture(scope="module")
def generic_gpt_cli(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("c++") is None:
        pytest.skip("a C++ compiler is required for the generic layer-evo guard tests")
    output = tmp_path_factory.mktemp("native-gpt-generic-evo-cli") / "nfn_gpt_native_train"
    env = os.environ.copy()
    env["NFN_NATIVE_GPT_FORCE_REBUILD"] = "1"
    env["NFN_NATIVE_GPT_CXX_OPT_FLAGS"] = "-O0"
    completed = subprocess.run(
        ["bash", str(ROOT / "tools" / "build_native_gpt2_cli.sh"), str(output)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.exists()
    return output


@pytest.fixture(scope="module")
def constructor_marking_tile_ops(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("c++") is None:
        pytest.skip("a C++ compiler is required for the generic layer-evo guard tests")
    directory = tmp_path_factory.mktemp("native-gpt-generic-evo-marker")
    source = directory / "marker.cpp"
    output = directory / "libmarker_tile_ops.so"
    source.write_text(
        "#include <cstdio>\n"
        "#include <cstdlib>\n"
        "static void mark(const char* env_name) {\n"
        "  const char* path = std::getenv(env_name);\n"
        "  if (path != nullptr) {\n"
        "    if (FILE* file = std::fopen(path, \"ab\")) {\n"
        "      std::fputc('1', file);\n"
        "      std::fclose(file);\n"
        "    }\n"
        "  }\n"
        "}\n"
        "__attribute__((constructor)) static void loaded() {\n"
        "  mark(\"NFN_GENERIC_EVO_TILE_LOAD_MARKER\");\n"
        "}\n"
        "extern \"C\" int nfn_native_tile_evo_mutate_candidates_float32(...) {\n"
        "  mark(\"NFN_GENERIC_EVO_MUTATION_MARKER\");\n"
        "  return 0;\n"
        "}\n"
        "extern \"C\" int nfn_native_tile_evo_select_best_loss_float32(...) {\n"
        "  mark(\"NFN_GENERIC_EVO_MUTATION_MARKER\");\n"
        "  return 0;\n"
        "}\n"
        "extern \"C\" int nfn_native_tile_evo_adopt_candidate_float32(...) {\n"
        "  mark(\"NFN_GENERIC_EVO_MUTATION_MARKER\");\n"
        "  return 0;\n"
        "}\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["c++", "-std=c++20", "-shared", "-fPIC", str(source), "-o", str(output)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.exists()
    return output


def _run_with_markers(
    cli: Path,
    tile_ops: Path,
    tmp_path: Path,
    *args: str,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    load_marker = tmp_path / "tile-loaded"
    mutation_marker = tmp_path / "evo-mutated"
    effects_root = tmp_path / "must-not-exist"
    env = os.environ.copy()
    env["NFN_GENERIC_EVO_TILE_LOAD_MARKER"] = str(load_marker)
    env["NFN_GENERIC_EVO_MUTATION_MARKER"] = str(mutation_marker)
    completed = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(tmp_path / "missing-dataset"),
            "--tile-ops-lib",
            str(tile_ops),
            "--cuda-runtime-lib",
            str(tile_ops),
            "--output-dir",
            str(effects_root / "output"),
            "--json-out",
            str(effects_root / "runtime.json"),
            "--max-steps",
            "1",
            *args,
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed, load_marker, mutation_marker, effects_root


def _assert_exact_blocked_report(
    completed: subprocess.CompletedProcess[str],
    *,
    requested_action: str,
    expected_returncode: int,
) -> dict[str, object]:
    assert completed.returncode == expected_returncode
    report = json.loads(completed.stdout)
    assert report == {
        "model_family": "gpt",
        "backend": "tile-cuda",
        "status": BLOCKED_STATUS,
        "passed": False,
        "template_name": "gpt",
        "resolved_native_template_name": "gpt2",
        "requested_action": requested_action,
        "selected_graph_support_status": (
            "preflight-only-graph-faithful-layer-evo-missing"
        ),
        "selected_graph_native_runnable": False,
        "training_execution_blocked": True,
        "blocking_reason": (
            "the generic dense GPT trainer AdamW-updates every tensor in the designated "
            "block and evolves only block_N.ln1.weight, but the authored GPT2-Evo "
            "contract excludes every tensor in that block from gradients and AdamW and "
            "evolves the whole block"
        ),
        "error": BLOCKED_ERROR,
        "tile_ops_load_attempted": False,
        "cuda_setup_attempted": False,
        "output_creation_attempted": False,
        "training_mutation_attempted": False,
        "missing_graph_faithful_gates": EXPECTED_MISSING_GATES,
        "layer_evo": {
            "enabled": True,
            "layer_index": 6,
            "interval": 10,
            "population": 8,
            "mutation_scale": 0.02,
            "tournament_size": 3,
            "elite_count": 1,
            "graph_faithful_whole_block_candidate_eval_enabled": False,
            "diagnostic_ln1_candidate_eval_available": True,
            "current_gradient_optimizer_scope": (
                "all-transformer-block-tensors-including-designated-evo-block"
            ),
            "current_evolution_parameter_scope": "block_N.ln1.weight-only",
            "required_evolution_parameter_scope": (
                "every-tensor-in-designated-transformer-block"
            ),
            "graph_editor_tensor_flow": False,
        },
    }
    assert "command" not in report
    return report


def test_help_marks_generic_layer_evo_execution_fail_closed(
    generic_gpt_cli: Path,
) -> None:
    completed = subprocess.run(
        [str(generic_gpt_cli), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        "generic training/startup/print-command fail closed until whole-block evolution "
        "is implemented"
    ) in completed.stdout
    assert (
        "Keep the ordinary dense GPT workflow with layer evolution disabled"
        in completed.stdout
    )


@pytest.mark.parametrize(
    "evo_args",
    [
        ("--layer-evo",),
        ("--enable-layer-evo",),
        ("--native-cuda-layer-evo",),
        ("--evo-layer-index=6",),
        ("--evo-layer-population", "8"),
    ],
)
@pytest.mark.parametrize(
    ("action_args", "requested_action"),
    [
        ((), "transformer-lm-training"),
        (("--native-cuda-startup-only",), "startup-only"),
        (("--native-cuda-print-command",), "print-command"),
    ],
)
def test_direct_generic_layer_evo_actions_fail_before_every_side_effect(
    generic_gpt_cli: Path,
    constructor_marking_tile_ops: Path,
    tmp_path: Path,
    evo_args: tuple[str, ...],
    action_args: tuple[str, ...],
    requested_action: str,
) -> None:
    completed, load_marker, mutation_marker, effects_root = _run_with_markers(
        generic_gpt_cli,
        constructor_marking_tile_ops,
        tmp_path,
        *evo_args,
        *action_args,
    )

    _assert_exact_blocked_report(
        completed,
        requested_action=requested_action,
        expected_returncode=2,
    )
    assert completed.stderr == f"nfn_gpt_native_train: {BLOCKED_ERROR}\n"
    assert not load_marker.exists()
    assert not mutation_marker.exists()
    assert not effects_root.exists()


@pytest.mark.parametrize(
    ("inspection_args", "requested_action"),
    [
        (("--print-plan",), "print-plan"),
        (("--native-cuda-dry-run",), "dry-run"),
    ],
)
def test_layer_evo_plan_inspection_is_blocked_metadata_without_side_effects(
    generic_gpt_cli: Path,
    constructor_marking_tile_ops: Path,
    tmp_path: Path,
    inspection_args: tuple[str, ...],
    requested_action: str,
) -> None:
    completed, load_marker, mutation_marker, effects_root = _run_with_markers(
        generic_gpt_cli,
        constructor_marking_tile_ops,
        tmp_path,
        "--layer-evo",
        *inspection_args,
    )

    _assert_exact_blocked_report(
        completed,
        requested_action=requested_action,
        expected_returncode=0,
    )
    assert completed.stderr == ""
    assert not load_marker.exists()
    assert not mutation_marker.exists()
    assert not effects_root.exists()


def test_final_no_layer_evo_override_preserves_ordinary_print_command(
    generic_gpt_cli: Path,
    constructor_marking_tile_ops: Path,
    tmp_path: Path,
) -> None:
    completed, load_marker, mutation_marker, effects_root = _run_with_markers(
        generic_gpt_cli,
        constructor_marking_tile_ops,
        tmp_path,
        "--enable-layer-evo",
        "--no-layer-evo",
        "--print-command",
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith(str(generic_gpt_cli))
    assert BLOCKED_STATUS not in completed.stdout
    assert completed.stderr == ""
    assert not load_marker.exists()
    assert not mutation_marker.exists()
    assert not effects_root.exists()


def test_isolated_primitive_smoke_remains_available_with_layer_evo_metadata(
    generic_gpt_cli: Path,
    constructor_marking_tile_ops: Path,
    tmp_path: Path,
) -> None:
    load_marker = tmp_path / "tile-loaded"
    mutation_marker = tmp_path / "evo-mutated"
    env = os.environ.copy()
    env["NFN_GENERIC_EVO_TILE_LOAD_MARKER"] = str(load_marker)
    env["NFN_GENERIC_EVO_MUTATION_MARKER"] = str(mutation_marker)

    completed = subprocess.run(
        [
            str(generic_gpt_cli),
            "--layer-evo",
            "--smoke-tile-ops",
            "--tile-ops-lib",
            str(constructor_marking_tile_ops),
            "--cuda-runtime-lib",
            str(tmp_path / "missing-libcudart.so"),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 2
    smoke = json.loads(completed.stdout)
    assert smoke["smoke"] == "tile_ops_fill"
    assert smoke["loaded"] is True
    assert smoke["kernel_loaded"] is False
    assert load_marker.read_text(encoding="utf-8") == "1"
    assert not mutation_marker.exists()
