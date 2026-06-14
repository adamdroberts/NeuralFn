from __future__ import annotations

import json
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path


def test_tile_cuda_examples_are_present_and_compile() -> None:
    example_dir = Path("examples/tile_cuda")
    expected = {
        "scalar_add_train.py",
        "dense_llm_smoke_train.py",
        "moe_router_smoke_train.py",
        "jepa_smoke_train.py",
        "strict_mode_report.py",
        "kernel_bench.py",
    }

    for name in expected:
        path = example_dir / name
        assert path.exists(), name
        py_compile.compile(str(path), doraise=True)


def test_generated_tile_cuda_registry_examples_are_present() -> None:
    generated_dir = Path("examples/tile_cuda/generated")
    generated = sorted(generated_dir.glob("*.py"))

    assert generated_dir.exists()
    assert len(generated) >= 129
    assert (generated_dir / "function_add.py").exists()
    assert (generated_dir / "module_scaled_dot_product_attention.py").exists()
    py_compile.compile(str(generated_dir / "function_add.py"), doraise=True)


def test_paired_kernel_speed_tool_compiles_and_smokes() -> None:
    script = Path("tools/paired_kernel_speed.py")
    output_path = Path(tempfile.mkdtemp()) / "paired.json"

    py_compile.compile(str(script), doraise=True)
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline",
            (
                f"{sys.executable} -c "
                "\"import os; print(os.environ.get('CUDA_VISIBLE_DEVICES', '')); "
                "print(os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', ''))\""
            ),
            "--candidate",
            (
                f"{sys.executable} -c "
                "\"print('{\\\"timing\\\": {\\\"train_loop_wall_ms\\\": 12.5, "
                "\\\"train_tokens_per_second\\\": 42.0, \\\"setup_wall_ms\\\": 1.0, "
                "\\\"checkpoint_wall_ms\\\": 0.0, \\\"total_wall_ms\\\": 15.0}, "
                "\\\"linear_tk_gemm_count\\\": 3, \\\"status\\\": \\\"native-test\\\"}')\""
            ),
            "--samples",
            "1",
            "--warmup",
            "0",
            "--json",
            "--json-out",
            str(output_path),
            "--cuda-visible-devices",
            "test-device",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "paired_interleaved_commands" in proc.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["measurement"] == "paired_interleaved_commands"
    assert payload["cuda_visible_devices"] == "test-device"
    assert payload["cuda_device_max_connections"] == "1"
    assert "gpu_before" in payload
    assert "gpu_after" in payload
    assert "test-device\n1\n" in payload["paired_samples"][0]["baseline"]["stdout_tail"]
    assert "gpus" in payload["gpu_before"]
    assert "compute_processes" in payload["gpu_before"]
    assert "test-device" in payload["paired_samples"][0]["baseline"]["stdout_tail"]
    assert payload["paired_samples"][0]["candidate"]["native_metrics"]["status"] == "native-test"
    assert payload["candidate_native_metrics"]["train_loop_wall_ms"]["mean"] == 12.5
    assert payload["candidate_native_metrics"]["train_tokens_per_second"]["mean"] == 42.0
