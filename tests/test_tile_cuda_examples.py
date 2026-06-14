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
            f"{sys.executable} -c \"import os; print(os.environ.get('CUDA_VISIBLE_DEVICES', ''))\"",
            "--candidate",
            f"{sys.executable} -c \"import os; print(os.environ.get('CUDA_VISIBLE_DEVICES', ''))\"",
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
    assert "gpu_before" in payload
    assert "gpu_after" in payload
    assert "gpus" in payload["gpu_before"]
    assert "compute_processes" in payload["gpu_before"]
    assert "test-device" in payload["paired_samples"][0]["baseline"]["stdout_tail"]
