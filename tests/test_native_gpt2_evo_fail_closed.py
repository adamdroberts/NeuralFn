from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from neuralfn import native_train as native_train_module


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MISSING_GATES = [
    "exclude every tensor in the designated transformer block from gradient and AdamW updates",
    (
        "mutate, evaluate, select, and adopt every tensor in the designated transformer "
        "block rather than only block_N.ln1.weight"
    ),
    "checkpoint and resume the whole-block evolutionary state with graph-faithful parity",
]


@pytest.fixture(scope="module")
def gpt2_evo_cli(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("c++") is None:
        pytest.skip("a C++ compiler is required for the GPT2-Evo fail-closed tests")
    output = tmp_path_factory.mktemp("native-gpt2-evo-cli") / "nfn_gpt2_evo_native_train"
    completed = subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O0",
            "-Wall",
            "-Wextra",
            "-pedantic",
            "-I",
            str(ROOT / "neuralfn" / "csrc" / "native_train"),
            str(ROOT / "neuralfn" / "csrc" / "native_train" / "gpt2_evo_native_train.cpp"),
            "-ldl",
            "-o",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.exists()
    return output


@pytest.fixture(scope="module")
def unified_cli(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("c++") is None:
        pytest.skip("a C++ compiler is required for the native registry test")
    output = tmp_path_factory.mktemp("native-train-registry") / "nfn_native_train"
    completed = subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O0",
            "-Wall",
            "-Wextra",
            "-pedantic",
            str(ROOT / "neuralfn" / "csrc" / "native_train" / "nfn_native_train.cpp"),
            "-o",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.exists()
    return output


@pytest.fixture(scope="module")
def marker_tile_ops_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("c++") is None:
        pytest.skip("a C++ compiler is required for the GPT2-Evo fail-closed tests")
    directory = tmp_path_factory.mktemp("native-gpt2-evo-marker-library")
    source = directory / "marker.cpp"
    output = directory / "libmarker_tile_ops.so"
    source.write_text(
        "#include <cstdio>\n"
        "#include <cstdlib>\n"
        "__attribute__((constructor)) static void loaded() {\n"
        "  const char* path = std::getenv(\"NFN_GPT2_EVO_DLOPEN_MARKER\");\n"
        "  if (path != nullptr) { if (FILE* f = std::fopen(path, \"wb\")) { std::fclose(f); } }\n"
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
    return output


def test_print_plan_reports_exact_graph_faithful_blockers(gpt2_evo_cli: Path) -> None:
    completed = subprocess.run(
        [str(gpt2_evo_cli), "--print-plan"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["status"] == "native-preflight-blocked-graph-faithful-layer-evo-missing"
    assert plan["selected_graph_support_status"] == (
        "preflight-only-graph-faithful-layer-evo-missing"
    )
    assert plan["selected_graph_native_runnable"] is False
    assert plan["training_execution_blocked"] is True
    assert plan["missing_graph_faithful_gates"] == EXPECTED_MISSING_GATES
    assert plan["layer_evo"]["current_gradient_optimizer_scope"] == (
        "all-transformer-block-tensors-including-designated-evo-block"
    )
    assert plan["layer_evo"]["current_evolution_parameter_scope"] == (
        "block_N.ln1.weight-only"
    )
    assert plan["layer_evo"]["required_evolution_parameter_scope"] == (
        "every-tensor-in-designated-transformer-block"
    )


@pytest.mark.parametrize("extra_args", [[], ["--startup-only"]])
def test_training_fails_before_delegate_exec(
    gpt2_evo_cli: Path,
    marker_tile_ops_library: Path,
    tmp_path: Path,
    extra_args: list[str],
) -> None:
    marker = tmp_path / "delegate-ran"
    dlopen_marker = tmp_path / "tile-ops-loaded"
    requested_output = tmp_path / "must-not-exist" / "model.bin"
    delegate = tmp_path / "nfn_gpt_native_train"
    delegate.write_text(
        "#!/bin/sh\nprintf delegated > \"$NFN_GPT2_EVO_DELEGATE_MARKER\"\nexit 0\n",
        encoding="utf-8",
    )
    delegate.chmod(0o755)
    env = os.environ.copy()
    env["NFN_NATIVE_GPT_CLI"] = str(delegate)
    env["NFN_GPT2_EVO_DELEGATE_MARKER"] = str(marker)
    env["NFN_GPT2_EVO_DLOPEN_MARKER"] = str(dlopen_marker)

    completed = subprocess.run(
        [
            str(gpt2_evo_cli),
            "--tile-ops-lib",
            str(marker_tile_ops_library),
            "--output",
            str(requested_output),
            *extra_args,
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 2
    assert not marker.exists()
    assert not dlopen_marker.exists()
    assert not requested_output.exists()
    assert not requested_output.parent.exists()
    assert "blocked before delegate exec, CUDA setup, allocation, or model mutation" in completed.stderr
    assert json.loads(completed.stdout)["missing_graph_faithful_gates"] == EXPECTED_MISSING_GATES


def test_print_command_is_a_non_executing_blocked_report(
    gpt2_evo_cli: Path,
    tmp_path: Path,
) -> None:
    marker = tmp_path / "delegate-ran"
    delegate = tmp_path / "nfn_gpt_native_train"
    delegate.write_text(
        "#!/bin/sh\nprintf delegated > \"$NFN_GPT2_EVO_DELEGATE_MARKER\"\n",
        encoding="utf-8",
    )
    delegate.chmod(0o755)
    env = os.environ.copy()
    env["NFN_NATIVE_GPT_CLI"] = str(delegate)
    env["NFN_GPT2_EVO_DELEGATE_MARKER"] = str(marker)

    completed = subprocess.run(
        [str(gpt2_evo_cli), "--print-command"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 2
    assert not marker.exists()
    report = json.loads(completed.stdout)
    assert report["selected_graph_native_runnable"] is False
    assert report["missing_graph_faithful_gates"] == EXPECTED_MISSING_GATES


def test_dry_run_remains_successful_inspection(gpt2_evo_cli: Path) -> None:
    completed = subprocess.run(
        [str(gpt2_evo_cli), "--dry-run"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["training_execution_blocked"] is True


def test_evo_kernel_smoke_reports_that_primitives_do_not_make_graph_runnable(
    gpt2_evo_cli: Path,
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            str(gpt2_evo_cli),
            "--smoke-evo-kernels",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 2
    smoke = json.loads(completed.stdout)
    assert smoke["smoke"] == "evo_kernels"
    assert smoke["passed"] is False
    assert smoke["selected_graph_native_runnable"] is False
    assert smoke["training_execution_blocked"] is True
    assert smoke["missing_graph_faithful_gates"] == EXPECTED_MISSING_GATES


def test_no_layer_evo_does_not_masquerade_as_gpt2_evo_training(
    gpt2_evo_cli: Path,
) -> None:
    completed = subprocess.run(
        [str(gpt2_evo_cli), "--no-layer-evo"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 2
    layer_evo = json.loads(completed.stdout)["layer_evo"]
    assert layer_evo["enabled"] is False
    assert layer_evo["diagnostic_ln1_candidate_eval_available"] is False
    assert layer_evo["current_gradient_optimizer_scope"] == "ordinary-dense-adamw-all-tensors"
    assert layer_evo["current_evolution_parameter_scope"] == "disabled"
    assert "invoke nfn_gpt_native_train directly for that distinct workflow" in completed.stderr


def test_cpp_and_python_registries_report_preflight_only(unified_cli: Path) -> None:
    completed = subprocess.run(
        [str(unified_cli), "--list-models", "--json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    cpp_models = {item["name"]: item for item in json.loads(completed.stdout)["models"]}
    python_entry = next(
        item
        for item in native_train_module._NATIVE_TRAIN_MODEL_REGISTRY
        if item["name"] == "gpt2-evo"
    )

    for entry in (cpp_models["gpt2-evo"], python_entry):
        assert entry["status"] == "preflight-only"
        assert entry["transformer_lm_status"] == (
            "blocked-graph-faithful-whole-block-evolution-missing"
        )
        assert entry["geometry_status"] == "gpt2-evo-whole-block-contract-unimplemented"
        assert entry["kernel_status"] == "evo-primitive-kernels-present"
        assert entry["trainer_loop_status"] == "blocked-before-delegate-exec"


def test_wrapper_does_not_retain_an_exec_or_double_free_path() -> None:
    source = (
        ROOT / "neuralfn" / "csrc" / "native_train" / "gpt2_evo_native_train.cpp"
    ).read_text(encoding="utf-8")

    assert "execvp(" not in source
    assert "exec_dense_gpt_delegate" not in source
    assert source.count('free_device(device_losses, "losses")') == 1
