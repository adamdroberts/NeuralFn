from __future__ import annotations

import json
import importlib.util
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
                "\\\"checkpoint_wall_ms\\\": 0.0, \\\"total_wall_ms\\\": 15.0, "
                "\\\"stage_timing\\\": [{\\\"name\\\": \\\"lm_head_backward\\\", "
                "\\\"total_ms\\\": 7.0, \\\"avg_ms\\\": 3.5, \\\"count\\\": 2}]}, "
                "\\\"linear_tk_gemm_count\\\": 3, \\\"status\\\": \\\"native-test\\\"}')\""
            ),
            "--samples",
            "1",
            "--warmup",
            "0",
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
    assert payload["candidate_native_metrics"]["stage.lm_head_backward.total_ms"]["mean"] == 7.0
    assert payload["candidate_native_metrics"]["stage.lm_head_backward.avg_ms"]["mean"] == 3.5
    assert payload["candidate_native_metrics"]["stage.lm_head_backward.count"]["mean"] == 2.0
    assert "gpus" in payload["gpu_before"]
    assert "compute_processes" in payload["gpu_before"]
    assert "gpu_before" in payload["paired_samples"][0]
    assert "gpu_after" in payload["paired_samples"][0]
    assert "gpus" in payload["paired_samples"][0]["gpu_before"]
    assert "compute_processes" in payload["paired_samples"][0]["gpu_before"]
    assert "gpu_compute_processes_per_sample_before:" in proc.stdout
    assert "test-device" in payload["paired_samples"][0]["baseline"]["stdout_tail"]
    assert payload["paired_samples"][0]["candidate"]["native_metrics"]["status"] == "native-test"
    assert payload["candidate_native_metrics"]["train_loop_wall_ms"]["mean"] == 12.5
    assert payload["candidate_native_metrics"]["train_tokens_per_second"]["mean"] == 42.0


def test_paired_kernel_speed_tool_extracts_llm_kittens_step_metrics() -> None:
    script = Path("tools/paired_kernel_speed.py")
    spec = importlib.util.spec_from_file_location("paired_kernel_speed", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)

    stdout = """
device memory usage: 28819 MiB / 32606 MiB
step    1/1 | loss 11.032360 (+nanz)| norm 22.1408 (+nanz)| lr 1.00e-05 | 2493.74 ms | 40.3% bf16 MFU | 210242 tok/s
"""

    metrics = module.native_metrics_from_stdout(stdout)
    assert metrics["status"] == "llm-kittens-step-log"
    assert metrics["train_loop_wall_ms"] == 2493.74
    assert metrics["train_tokens_per_second"] == 210242.0
    assert metrics["llm_kittens_bf16_mfu_pct"] == 40.3
    assert metrics["llm_kittens_device_memory_used_mib"] == 28819
    assert metrics["llm_kittens_device_memory_total_mib"] == 32606


def test_paired_kernel_speed_tool_records_command_timeout() -> None:
    script = Path("tools/paired_kernel_speed.py")
    output_path = Path(tempfile.mkdtemp()) / "paired-timeout.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline",
            f"{sys.executable} -c \"print('baseline-ok')\"",
            "--candidate",
            f"{sys.executable} -c \"import time; time.sleep(5)\"",
            "--samples",
            "1",
            "--warmup",
            "0",
            "--json-out",
            str(output_path),
            "--continue-on-error",
            "--command-timeout-seconds",
            "0.1",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "command_timeouts: baseline=0 candidate=1" in proc.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    sample = payload["paired_samples"][0]
    assert sample["baseline"]["timed_out"] is False
    assert sample["candidate"]["timed_out"] is True
    assert sample["candidate"]["returncode"] == -1
    assert sample["candidate"]["timeout_seconds"] == 0.1
    assert payload["command_timeout_seconds"] == 0.1
