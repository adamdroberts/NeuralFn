from __future__ import annotations

import json
import importlib.util
import os
import py_compile
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest


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
                "\\\"setup_timing\\\": [{\\\"name\\\": \\\"setup.float_arena_materialize\\\", "
                "\\\"total_ms\\\": 0.7, \\\"avg_ms\\\": 0.7, \\\"count\\\": 1}], "
                "\\\"checkpoint_wall_ms\\\": 0.0, \\\"total_wall_ms\\\": 15.0, "
                "\\\"stage_timing\\\": [{\\\"name\\\": \\\"lm_head_backward\\\", "
                "\\\"total_ms\\\": 7.0, \\\"avg_ms\\\": 3.5, \\\"count\\\": 2}, "
                "{\\\"name\\\": \\\"lm_head_backward.logits\\\", "
                "\\\"total_ms\\\": 3.0, \\\"avg_ms\\\": 1.5, \\\"count\\\": 2}, "
                "{\\\"name\\\": \\\"lm_head_backward.ce\\\", "
                "\\\"total_ms\\\": 1.0, \\\"avg_ms\\\": 0.5, \\\"count\\\": 2}, "
                "{\\\"name\\\": \\\"lm_head_backward.dhidden\\\", "
                "\\\"total_ms\\\": 2.0, \\\"avg_ms\\\": 1.0, \\\"count\\\": 2}, "
                "{\\\"name\\\": \\\"lm_head_backward.dweight\\\", "
                "\\\"total_ms\\\": 1.5, \\\"avg_ms\\\": 0.75, \\\"count\\\": 2}, "
                "{\\\"name\\\": \\\"lm_head_backward.pipeline_queue\\\", "
                "\\\"total_ms\\\": 0.4, \\\"avg_ms\\\": 0.2, \\\"count\\\": 2}, "
                "{\\\"name\\\": \\\"lm_head_backward.pipeline_final_wait\\\", "
                "\\\"total_ms\\\": 0.6, \\\"avg_ms\\\": 0.3, \\\"count\\\": 2}, "
                "{\\\"name\\\": \\\"final_norm_backward\\\", "
                "\\\"total_ms\\\": 0.8, \\\"avg_ms\\\": 0.4, \\\"count\\\": 2}, "
                "{\\\"name\\\": \\\"block_backward.mlp_proj\\\", "
                "\\\"total_ms\\\": 5.0, \\\"avg_ms\\\": 2.5, \\\"count\\\": 2}, "
                "{\\\"name\\\": \\\"block_backward.mlp_proj.dweight_bias\\\", "
                "\\\"total_ms\\\": 4.0, \\\"avg_ms\\\": 2.0, \\\"count\\\": 2}, "
                "{\\\"name\\\": \\\"block_backward.mlp_fc.dinput\\\", "
                "\\\"total_ms\\\": 8.0, \\\"avg_ms\\\": 4.0, \\\"count\\\": 2}, "
                "{\\\"name\\\": \\\"block_backward.attn_proj.dinput\\\", "
                "\\\"total_ms\\\": 9.0, \\\"avg_ms\\\": 4.5, \\\"count\\\": 2}, "
                "{\\\"name\\\": \\\"block_backward.attn_sdpa.to_qkv\\\", "
                "\\\"total_ms\\\": 6.0, \\\"avg_ms\\\": 3.0, \\\"count\\\": 2}, "
                "{\\\"name\\\": \\\"embedding_backward\\\", "
                "\\\"total_ms\\\": 0.9, \\\"avg_ms\\\": 0.45, \\\"count\\\": 2}, "
                "{\\\"name\\\": \\\"gradient_zero\\\", "
                "\\\"total_ms\\\": 0.6, \\\"avg_ms\\\": 0.6, \\\"count\\\": 1}, "
                "{\\\"name\\\": \\\"gradient_clip\\\", "
                "\\\"total_ms\\\": 1.1, \\\"avg_ms\\\": 1.1, \\\"count\\\": 1}, "
                "{\\\"name\\\": \\\"adamw_update\\\", "
                "\\\"total_ms\\\": 2.2, \\\"avg_ms\\\": 2.2, \\\"count\\\": 1}]}, "
                "\\\"steps_completed\\\": 5, \\\"linear_tk_gemm_count\\\": 3, "
                "\\\"linear_cublaslt_gemm_count\\\": 4, \\\"linear_bf16_gemm_count\\\": 7, "
                "\\\"lm_head_logits_tk_gemm_count\\\": 2, "
                "\\\"lm_head_logits_cublaslt_gemm_count\\\": 0, "
                "\\\"lm_head_logits_bf16_gemm_count\\\": 2, "
                "\\\"lm_head_classifier_chunk_kernel_available\\\": true, "
                "\\\"lm_head_classifier_chunk_kernel_enabled\\\": true, "
                "\\\"lm_head_classifier_chunk_launch_count\\\": 64, "
                "\\\"lm_head_classifier_last_rows\\\": 8192, "
                "\\\"lm_head_classifier_last_vocab\\\": 50257, "
                "\\\"lm_head_classifier_last_row_stride\\\": 50304, "
                "\\\"status\\\": \\\"native-test\\\"}')\""
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
    assert payload["require_idle_selected_gpu"] is False
    assert payload["max_selected_gpu_utilization_pct"] == -1.0
    assert "require_idle_selected_gpu: False" in proc.stdout
    assert "max_selected_gpu_utilization_pct: -1.0" in proc.stdout
    assert payload["gpu_benchmark_lock_enabled"] is True
    assert payload["gpu_benchmark_lock_acquired"] is True
    assert payload["gpu_benchmark_lock_path"].endswith("nfn_paired_kernel_speed_gpu_test-device.lock")
    assert "gpu_benchmark_lock: enabled=True acquired=True" in proc.stdout
    assert "gpu_before" in payload
    assert "gpu_after" in payload
    assert "gpu_sample_summary" in payload
    assert payload["gpu_sample_summary"]["selected_cuda_visible_devices"] == "test-device"
    assert "test-device\n1\n" in payload["paired_samples"][0]["baseline"]["stdout_tail"]
    assert payload["candidate_native_metrics"]["setup.float_arena_materialize.total_ms"]["mean"] == 0.7
    assert payload["candidate_native_metrics"]["setup.float_arena_materialize.avg_ms"]["mean"] == 0.7
    assert payload["candidate_native_metrics"]["setup.float_arena_materialize.count"]["mean"] == 1.0
    assert payload["candidate_native_metrics"]["stage.lm_head_backward.total_ms"]["mean"] == 7.0
    assert payload["candidate_native_metrics"]["stage.lm_head_backward.avg_ms"]["mean"] == 3.5
    assert payload["candidate_native_metrics"]["stage.lm_head_backward.count"]["mean"] == 2.0
    assert payload["candidate_native_metrics"]["stage.lm_head_backward.logits.total_ms"]["mean"] == 3.0
    assert payload["candidate_native_metrics"]["stage.lm_head_backward.ce.total_ms"]["mean"] == 1.0
    assert payload["candidate_native_metrics"]["stage.lm_head_backward.dhidden.total_ms"]["mean"] == 2.0
    assert payload["candidate_native_metrics"]["stage.lm_head_backward.dweight.total_ms"]["mean"] == 1.5
    assert payload["candidate_native_metrics"]["stage.lm_head_backward.pipeline_queue.total_ms"]["mean"] == 0.4
    assert (
        payload["candidate_native_metrics"]["stage.lm_head_backward.pipeline_final_wait.total_ms"]["mean"]
        == 0.6
    )
    assert payload["candidate_native_metrics"]["stage.final_norm_backward.total_ms"]["mean"] == 0.8
    assert payload["candidate_native_metrics"]["stage.block_backward.mlp_proj.total_ms"]["mean"] == 5.0
    assert (
        payload["candidate_native_metrics"]["stage.block_backward.mlp_proj.dweight_bias.total_ms"]["mean"]
        == 4.0
    )
    assert payload["candidate_native_metrics"]["stage.block_backward.mlp_fc.dinput.total_ms"]["mean"] == 8.0
    assert payload["candidate_native_metrics"]["stage.block_backward.attn_proj.dinput.total_ms"]["mean"] == 9.0
    assert payload["candidate_native_metrics"]["stage.block_backward.attn_sdpa.to_qkv.total_ms"]["mean"] == 6.0
    assert payload["candidate_native_metrics"]["stage.embedding_backward.total_ms"]["mean"] == 0.9
    assert payload["candidate_native_metrics"]["stage.gradient_zero.total_ms"]["mean"] == 0.6
    assert payload["candidate_native_metrics"]["stage.gradient_clip.total_ms"]["mean"] == 1.1
    assert payload["candidate_native_metrics"]["stage.adamw_update.total_ms"]["mean"] == 2.2
    assert "gpus" in payload["gpu_before"]
    assert "compute_processes" in payload["gpu_before"]
    assert "gpu_before" in payload["paired_samples"][0]
    assert "gpu_after" in payload["paired_samples"][0]
    assert "gpu_before" in payload["paired_samples"][0]["baseline"]
    assert "gpu_after" in payload["paired_samples"][0]["baseline"]
    assert "gpu_before" in payload["paired_samples"][0]["candidate"]
    assert "gpu_after" in payload["paired_samples"][0]["candidate"]
    assert "gpus" in payload["paired_samples"][0]["gpu_before"]
    assert "compute_processes" in payload["paired_samples"][0]["gpu_before"]
    assert "gpu_compute_processes_per_sample_before:" in proc.stdout
    assert "gpu_sample_summary:" in proc.stdout
    assert "test-device" in payload["paired_samples"][0]["baseline"]["stdout_tail"]
    assert payload["paired_samples"][0]["candidate"]["native_metrics"]["status"] == "native-test"
    assert payload["candidate_native_metrics"]["train_loop_wall_ms"]["mean"] == 12.5
    assert payload["candidate_native_metrics"]["train_loop_wall_ms_per_step"]["mean"] == 2.5
    assert payload["candidate_native_metrics"]["steps_completed"]["mean"] == 5.0
    assert payload["candidate_native_metrics"]["train_tokens_per_second"]["mean"] == 42.0
    assert payload["candidate_native_metrics"]["linear_tk_gemm_count"]["mean"] == 3.0
    assert payload["candidate_native_metrics"]["linear_cublaslt_gemm_count"]["mean"] == 4.0
    assert payload["candidate_native_metrics"]["linear_bf16_gemm_count"]["mean"] == 7.0
    assert payload["candidate_native_metrics"]["lm_head_logits_tk_gemm_count"]["mean"] == 2.0
    assert payload["candidate_native_metrics"]["lm_head_logits_cublaslt_gemm_count"]["mean"] == 0.0
    assert payload["candidate_native_metrics"]["lm_head_logits_bf16_gemm_count"]["mean"] == 2.0
    assert payload["candidate_native_metrics"]["lm_head_classifier_chunk_launch_count"]["mean"] == 64.0
    assert payload["candidate_native_metrics"]["lm_head_classifier_last_rows"]["mean"] == 8192.0
    assert payload["candidate_native_metrics"]["lm_head_classifier_last_vocab"]["mean"] == 50257.0
    assert payload["candidate_native_metrics"]["lm_head_classifier_last_row_stride"]["mean"] == 50304.0
    assert "linear_tk_gemm_count: mean=3.000000" in proc.stdout
    assert "linear_cublaslt_gemm_count: mean=4.000000" in proc.stdout
    assert "linear_bf16_gemm_count: mean=7.000000" in proc.stdout
    assert "lm_head_logits_tk_gemm_count: mean=2.000000" in proc.stdout
    assert "lm_head_logits_cublaslt_gemm_count: mean=0.000000" in proc.stdout
    assert "lm_head_logits_bf16_gemm_count: mean=2.000000" in proc.stdout
    assert "lm_head_classifier_chunk_launch_count: mean=64.000000" in proc.stdout
    assert "lm_head_classifier_last_rows: mean=8192.000000" in proc.stdout
    assert (
        payload["paired_samples"][0]["candidate"]["native_metrics"]["lm_head_classifier_chunk_kernel_available"]
        is True
    )
    assert (
        payload["paired_samples"][0]["candidate"]["native_metrics"]["lm_head_classifier_chunk_kernel_enabled"]
        is True
    )
    assert "stage.lm_head_backward.logits.total_ms: mean=3.000000" in proc.stdout
    assert "stage.lm_head_backward.ce.total_ms: mean=1.000000" in proc.stdout
    assert "stage.lm_head_backward.dhidden.total_ms: mean=2.000000" in proc.stdout
    assert "stage.lm_head_backward.dweight.total_ms: mean=1.500000" in proc.stdout
    assert "stage.lm_head_backward.pipeline_queue.total_ms: mean=0.400000" in proc.stdout
    assert "stage.lm_head_backward.pipeline_final_wait.total_ms: mean=0.600000" in proc.stdout
    assert "stage.final_norm_backward.total_ms: mean=0.800000" in proc.stdout
    assert "stage.block_backward.mlp_proj.total_ms: mean=5.000000" in proc.stdout
    assert "stage.block_backward.mlp_proj.dweight_bias.total_ms: mean=4.000000" in proc.stdout
    assert "stage.block_backward.mlp_fc.dinput.total_ms: mean=8.000000" in proc.stdout
    assert "stage.block_backward.attn_proj.dinput.total_ms: mean=9.000000" in proc.stdout
    assert "stage.block_backward.attn_sdpa.to_qkv.total_ms: mean=6.000000" in proc.stdout
    assert "stage.embedding_backward.total_ms: mean=0.900000" in proc.stdout
    assert "stage.gradient_zero.total_ms: mean=0.600000" in proc.stdout
    assert "stage.gradient_clip.total_ms: mean=1.100000" in proc.stdout
    assert "stage.adamw_update.total_ms: mean=2.200000" in proc.stdout


def test_paired_kernel_speed_tool_dry_run_plan_does_not_launch_commands() -> None:
    script = Path("tools/paired_kernel_speed.py")
    output_path = Path(tempfile.mkdtemp()) / "paired-plan.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline",
            "definitely_missing_baseline_command --old",
            "--candidate",
            "definitely_missing_candidate_command --new",
            "--samples",
            "2",
            "--warmup",
            "1",
            "--cuda-visible-devices",
            "0",
            "--require-idle-selected-gpu",
            "--dry-run-plan",
            "--json-out",
            str(output_path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "dry_run_plan: true" in proc.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["dry_run_plan"] is True
    assert payload["baseline_command"] == ["definitely_missing_baseline_command", "--old"]
    assert payload["candidate_command"] == ["definitely_missing_candidate_command", "--new"]
    assert payload["sample_order_plan"] == [
        {"sample": 1, "order": ["baseline", "candidate"]},
        {"sample": 2, "order": ["candidate", "baseline"]},
    ]
    assert payload["run_env_overrides"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert payload["gpu_benchmark_lock_enabled"] is True
    assert payload["gpu_benchmark_lock_acquired"] is False
    assert payload["gpu_benchmark_lock_path"].endswith("nfn_paired_kernel_speed_gpu_0.lock")
    assert "paired_samples" not in payload


def test_native_gpt_sm120_parity_wrapper_uses_reference_shape() -> None:
    script = Path("tools/bench_native_gpt_sm120_parity.sh")

    proc = subprocess.run(
        ["bash", "-n", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    text = script.read_text(encoding="utf-8")
    assert "tools/paired_kernel_speed.py" in text
    assert "--require-idle-selected-gpu" in text
    assert "--max-selected-gpu-utilization-pct" in text
    assert "--selected-gpu-utilization-retries" in text
    assert "--selected-gpu-utilization-retry-interval-seconds" in text
    assert 'NFN_SM120_SELECTED_GPU_UTILIZATION_RETRIES:-3' in text
    assert 'NFN_SM120_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS:-0.25' in text
    assert 'CUDA_VISIBLE_DEVICES_VALUE="${NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES:-${NFN_SM120_CUDA_VISIBLE_DEVICES:-auto}}"' in text
    assert "TinyStories_train.bin" in text
    assert "TinyStories_val.bin" in text
    assert "-b 64" in text
    assert "-t 1024" in text
    assert "-d 524288" in text
    assert "-l 0.0006" in text
    assert "-q 0.0" in text
    assert "-u 60" in text
    assert "NFN_SM120_PARITY_SAMPLE_EVERY" in text
    assert "NFN_SM120_PARITY_CHECKPOINT_EVERY" in text
    assert "NFN_SM120_PARITY_GENERATE_TOKENS" in text
    assert "NFN_SM120_PARITY_PROFILE_DIR" in text
    assert "NFN_SM120_PARITY_DRY_RUN_PLAN" in text
    assert "NFN_SM120_STEPS" in text
    assert "NFN_SM120_JSON_OUT" in text
    assert "NFN_NATIVE_GPT_STAGE_TIMING_MAX_EVENTS" in text
    assert "paired_args=()" in text
    assert "profile_args=()" in text
    assert '\"none\"|\"off\"' in text
    assert "-s \"$SAMPLE_EVERY\"" in text
    assert "-g \"$GENERATE_TOKENS\"" in text
    assert "-n \"$CHECKPOINT_EVERY\"" in text
    assert "-af \"$ACTIVATION\"" in text
    assert "--backend tile-cuda" in text
    assert "--max-steps \"$STEPS\"" in text
    assert "--train-batch-tokens 524288" in text
    assert "--eval-every-steps 0" in text
    assert "--native-cuda-sample-every \"$SAMPLE_EVERY\"" in text
    assert "--native-cuda-generate-tokens \"$GENERATE_TOKENS\"" in text
    assert "--native-cuda-checkpoint-every \"$CHECKPOINT_EVERY\"" in text
    assert "--no-checkpoint" in text
    assert "--tile-ops-lib \"$NFN_NATIVE_TILE_OPS_LIB\"" in text
    assert '"${profile_args[@]}"' in text
    assert '"${paired_args[@]}"' in text


def test_native_gpt_sm120_candidate_wrapper_forwards_bisection_controls() -> None:
    script = Path("tools/bench_native_gpt_sm120_candidate.sh")

    proc = subprocess.run(
        ["bash", "-n", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    text = script.read_text(encoding="utf-8")
    assert "tools/paired_kernel_speed.py" in text
    assert "--require-idle-selected-gpu" in text
    assert "--max-selected-gpu-utilization-pct" in text
    assert "--selected-gpu-utilization-retries" in text
    assert "--selected-gpu-utilization-retry-interval-seconds" in text
    assert "NFN_SM120_SELECTED_GPU_UTILIZATION_RETRIES 3" in text
    assert "NFN_SM120_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS 0.25" in text
    assert "env_or_alias()" in text
    assert "env_or_alias3()" in text
    assert "env_or_alias4()" in text
    assert "append_env_overrides()" in text
    assert "[A-Za-z_][A-Za-z0-9_]*=" in text
    assert "NFN_SM120_PARITY_STEPS" in text
    assert "NFN_SM120_PARITY_PROFILE_DIR" in text
    assert "NFN_SM120_PARITY_DRY_RUN_PLAN" in text
    assert "NFN_SM120_CANDIDATE_CUDA_VISIBLE_DEVICES" in text
    assert "NFN_SM120_CANDIDATE_STEPS" in text
    assert "NFN_SM120_CANDIDATE_SAMPLES" in text
    assert "NFN_SM120_CANDIDATE_STAGE_TIMING" in text
    assert "NFN_SM120_CANDIDATE_LINEAR_SHAPE_STATS" in text
    assert "NFN_SM120_CANDIDATE_JSON_OUT" in text
    assert "NFN_SM120_STEPS" in text
    assert "NFN_SM120_SAMPLES" in text
    assert "NFN_SM120_JSON_OUT" in text
    assert "NFN_SM120_MAX_GPU_UTILIZATION_PCT" in text
    assert "NFN_SM120_NATIVE_SELECTED_GPU_UTILIZATION_RETRIES" in text
    assert "NFN_SM120_CANDIDATE_SELECTED_GPU_UTILIZATION_RETRIES" in text
    assert "NFN_SM120_PARITY_SELECTED_GPU_UTILIZATION_RETRIES" in text
    assert "NFN_SM120_SELECTED_GPU_UTILIZATION_RETRIES" in text
    assert "NFN_SM120_NATIVE_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS" in text
    assert "NFN_SM120_CANDIDATE_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS" in text
    assert "NFN_SM120_PARITY_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS" in text
    assert "NFN_SM120_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS" in text
    assert "NFN_SM120_DRY_RUN_PLAN" in text
    assert "NFN_SM120_NATIVE_CANDIDATE_ENV" in text
    assert "NFN_SM120_CANDIDATE_ENV" in text
    assert "NFN_SM120_NATIVE_ENV" in text
    assert "NFN_SM120_COMMON_ENV" in text
    assert "NFN_SM120_PARITY_ENV" in text
    assert "NFN_SM120_COMMON_EXTRA_ARGS" in text
    assert "NFN_SM120_NATIVE_CANDIDATE_ARGS" in text
    assert "NFN_SM120_CANDIDATE_EXTRA_ARGS" in text
    assert "NFN_SM120_CANDIDATE_CANDIDATE_EXTRA_ARGS" not in text
    assert "NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN" in text
    assert "NFN_SM120_CANDIDATE_TRAIN_BIN" in text
    assert "NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB" in text
    assert "NFN_SM120_NATIVE_TEMPLATE_NAME" in text
    assert "NFN_SM120_CANDIDATE_TEMPLATE_NAME" in text
    assert "NFN_SM120_NATIVE_GRAPH_FILE" in text
    assert "NFN_SM120_CANDIDATE_GRAPH_FILE" in text
    assert "NFN_SM120_NATIVE_DRY_RUN_PLAN" in text
    assert "NFN_SM120_CANDIDATE_DRY_RUN_PLAN" in text
    assert "--template-name \"$TEMPLATE_NAME\"" in text
    assert "--graph-file \"$GRAPH_FILE\"" in text
    assert "--train-batch-tokens \"$TRAIN_BATCH_TOKENS\"" in text
    assert "--eval-every-steps 0" in text
    assert "--no-checkpoint" in text
    assert "--dry-run-plan" in text
    assert '"${profile_args[@]}"' in text
    assert '"${paired_args[@]}"' in text
    assert "NFN_NATIVE_GPT_LINEAR_SHAPE_STATS=1" in text


def test_native_gpt_sm120_candidate_wrapper_accepts_short_aliases(tmp_path: Path) -> None:
    script = Path("tools/bench_native_gpt_sm120_candidate.sh")
    output_path = tmp_path / "candidate-alias.json"

    env = os.environ.copy()
    env.update(
        {
            "NFN_SM120_CANDIDATE_DRY_RUN_PLAN": "1",
            "NFN_SM120_CANDIDATE_STEPS": "2",
            "NFN_SM120_CANDIDATE_SAMPLES": "1",
            "NFN_SM120_CANDIDATE_WARMUP": "0",
            "NFN_SM120_CANDIDATE_PROFILE_DIR": "none",
            "NFN_SM120_CANDIDATE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_CANDIDATE_ENV": "NFN_ALIAS_PROBE=1",
            "NFN_SM120_CANDIDATE_EXTRA_ARGS": "--lm-head-row-chunk-size 32768",
            "NFN_SM120_CANDIDATE_JSON_OUT": str(output_path),
        }
    )

    proc = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    assert "samples: 1" in proc.stdout
    assert "warmup: 0" in proc.stdout
    assert "cuda_visible_devices: requested=7 resolved=7 mode=explicit" in proc.stdout
    assert "--max-steps 2" in proc.stdout
    assert "  baseline:" in proc.stdout
    assert "  candidate:" in proc.stdout
    baseline_command = proc.stdout.split("  baseline:", 1)[1].split("  candidate:", 1)[0]
    candidate_command = proc.stdout.split("  candidate:", 1)[1]
    assert "--lm-head-row-chunk-size 32768" not in baseline_command
    assert "--lm-head-row-chunk-size 32768" in candidate_command


def test_native_gpt_sm120_candidate_wrapper_accepts_candidate_train_bin(tmp_path: Path) -> None:
    script = Path("tools/bench_native_gpt_sm120_candidate.sh")
    output_path = tmp_path / "candidate-train-bin.json"
    baseline_train = tmp_path / "baseline_train"
    candidate_train = tmp_path / "candidate_train"
    baseline_tile_ops = tmp_path / "baseline_tile_ops.so"
    candidate_tile_ops = tmp_path / "candidate_tile_ops.so"

    for executable in (baseline_train, candidate_train):
        executable.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    baseline_tile_ops.write_text("", encoding="utf-8")
    candidate_tile_ops.write_text("", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_STEPS": "2",
            "NFN_SM120_NATIVE_SAMPLES": "1",
            "NFN_SM120_NATIVE_WARMUP": "0",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_NATIVE_GPT_TRAIN_BIN": str(baseline_train),
            "NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN": str(candidate_train),
            "NFN_NATIVE_TILE_OPS_LIB": str(baseline_tile_ops),
            "NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB": str(candidate_tile_ops),
            "NFN_SM120_NATIVE_JSON_OUT": str(output_path),
        }
    )

    proc = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    assert "  baseline:" in proc.stdout
    assert "  candidate:" in proc.stdout
    baseline_command = proc.stdout.split("  baseline:", 1)[1].split("  candidate:", 1)[0]
    candidate_command = proc.stdout.split("  candidate:", 1)[1]
    assert str(baseline_train) in baseline_command
    assert str(candidate_train) not in baseline_command
    assert str(candidate_train) in candidate_command
    assert str(baseline_train) not in candidate_command
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["baseline_command"][0] == str(baseline_train)
    assert payload["candidate_command"][0] == str(candidate_train)


def test_native_gpt_sm120_candidate_wrapper_applies_common_env_to_both_commands(tmp_path: Path) -> None:
    script = Path("tools/bench_native_gpt_sm120_candidate.sh")
    output_path = tmp_path / "candidate-common-env.json"

    env = os.environ.copy()
    env.update(
        {
            "NFN_SM120_CANDIDATE_DRY_RUN_PLAN": "1",
            "NFN_SM120_CANDIDATE_STEPS": "2",
            "NFN_SM120_CANDIDATE_SAMPLES": "1",
            "NFN_SM120_CANDIDATE_WARMUP": "0",
            "NFN_SM120_CANDIDATE_PROFILE_DIR": "none",
            "NFN_SM120_CANDIDATE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_COMMON_ENV": "NFN_SHARED_PROFILING=1",
            "NFN_SM120_CANDIDATE_ENV": "NFN_CANDIDATE_ONLY=1",
            "NFN_SM120_CANDIDATE_JSON_OUT": str(output_path),
        }
    )

    proc = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    assert "  baseline:" in proc.stdout
    assert "  candidate:" in proc.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["baseline_env"] == {"NFN_SHARED_PROFILING": "1"}
    assert payload["candidate_env"] == {
        "NFN_SHARED_PROFILING": "1",
        "NFN_CANDIDATE_ONLY": "1",
    }


def test_native_gpt_sm120_candidate_wrapper_splits_comma_separated_env_assignments(
    tmp_path: Path,
) -> None:
    script = Path("tools/bench_native_gpt_sm120_candidate.sh")
    output_path = tmp_path / "candidate-comma-env.json"

    env = os.environ.copy()
    env.update(
        {
            "NFN_SM120_CANDIDATE_DRY_RUN_PLAN": "1",
            "NFN_SM120_CANDIDATE_STEPS": "2",
            "NFN_SM120_CANDIDATE_SAMPLES": "1",
            "NFN_SM120_CANDIDATE_WARMUP": "0",
            "NFN_SM120_CANDIDATE_PROFILE_DIR": "none",
            "NFN_SM120_CANDIDATE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_COMMON_ENV": "NFN_SHARED=1,NFN_SHARED_2=2",
            "NFN_SM120_CANDIDATE_ENV": (
                "NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,3072,65536,N,T,0,"
                "NFN_SECOND_TOGGLE=1"
            ),
            "NFN_SM120_CANDIDATE_JSON_OUT": str(output_path),
        }
    )

    proc = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["baseline_env"] == {"NFN_SHARED": "1", "NFN_SHARED_2": "2"}
    assert payload["candidate_env"] == {
        "NFN_SHARED": "1",
        "NFN_SHARED_2": "2",
        "NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE": "768,3072,65536,N,T,0",
        "NFN_SECOND_TOGGLE": "1",
    }


def test_native_gpt_sm120_candidate_wrapper_accepts_legacy_candidate_args_alias(tmp_path: Path) -> None:
    script = Path("tools/bench_native_gpt_sm120_candidate.sh")
    output_path = tmp_path / "candidate-args-alias.json"

    env = os.environ.copy()
    env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_STEPS": "2",
            "NFN_SM120_NATIVE_SAMPLES": "1",
            "NFN_SM120_NATIVE_WARMUP": "0",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_ARGS": "--lm-head-row-chunk-size 4096",
            "NFN_SM120_NATIVE_JSON_OUT": str(output_path),
        }
    )

    proc = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    assert "  baseline:" in proc.stdout
    assert "  candidate:" in proc.stdout
    baseline_command = proc.stdout.split("  baseline:", 1)[1].split("  candidate:", 1)[0]
    candidate_command = proc.stdout.split("  candidate:", 1)[1]
    assert "--lm-head-row-chunk-size 4096" not in baseline_command
    assert "--lm-head-row-chunk-size 4096" in candidate_command


def test_native_gpt_sm120_candidate_wrapper_accepts_parity_aliases(tmp_path: Path) -> None:
    script = Path("tools/bench_native_gpt_sm120_candidate.sh")
    output_path = tmp_path / "candidate-parity-alias.json"

    env = os.environ.copy()
    env.update(
        {
            "NFN_SM120_PARITY_DRY_RUN_PLAN": "1",
            "NFN_SM120_PARITY_STEPS": "2",
            "NFN_SM120_PARITY_SAMPLES": "1",
            "NFN_SM120_PARITY_WARMUP": "0",
            "NFN_SM120_PARITY_PROFILE_DIR": "none",
            "NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_PARITY_JSON_OUT": str(output_path),
        }
    )

    proc = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    assert "samples: 1" in proc.stdout
    assert "warmup: 0" in proc.stdout
    assert "cuda_visible_devices: requested=7 resolved=7 mode=explicit" in proc.stdout
    assert "--max-steps 2" in proc.stdout
    assert "--append-native-profile-json-dir" not in proc.stdout


def test_native_gpt_sm120_candidate_wrapper_accepts_generic_aliases(tmp_path: Path) -> None:
    script = Path("tools/bench_native_gpt_sm120_candidate.sh")
    output_path = tmp_path / "candidate-generic-alias.json"

    env = os.environ.copy()
    env.update(
        {
            "NFN_SM120_DRY_RUN_PLAN": "1",
            "NFN_SM120_STEPS": "2",
            "NFN_SM120_SAMPLES": "1",
            "NFN_SM120_WARMUP": "0",
            "NFN_SM120_PROFILE_DIR": "none",
            "NFN_SM120_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_JSON_OUT": str(output_path),
        }
    )

    proc = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    assert "samples: 1" in proc.stdout
    assert "warmup: 0" in proc.stdout
    assert "cuda_visible_devices: requested=7 resolved=7 mode=explicit" in proc.stdout
    assert "--max-steps 2" in proc.stdout
    assert "--append-native-profile-json-dir" not in proc.stdout


def test_native_gpt_sm120_candidate_wrapper_stage_timing_without_profile_dir(tmp_path: Path) -> None:
    script = Path("tools/bench_native_gpt_sm120_candidate.sh")
    output_path = tmp_path / "candidate-stage-no-profile.json"

    env = os.environ.copy()
    env.update(
        {
            "NFN_SM120_DRY_RUN_PLAN": "1",
            "NFN_SM120_STEPS": "2",
            "NFN_SM120_SAMPLES": "1",
            "NFN_SM120_WARMUP": "0",
            "NFN_SM120_PROFILE_DIR": "none",
            "NFN_SM120_STAGE_TIMING": "1",
            "NFN_SM120_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_JSON_OUT": str(output_path),
        }
    )

    proc = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    assert "--append-native-profile-json-dir" not in proc.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["native_stage_timing"] is True
    assert payload["append_native_profile_json_dir"] == ""


def test_native_gpt_sm120_parity_wrapper_accepts_generic_aliases(tmp_path: Path) -> None:
    script = Path("tools/bench_native_gpt_sm120_parity.sh")
    output_path = tmp_path / "parity-generic-alias.json"

    env = os.environ.copy()
    env.update(
        {
            "NFN_SM120_DRY_RUN_PLAN": "1",
            "NFN_SM120_STEPS": "2",
            "NFN_SM120_SAMPLES": "1",
            "NFN_SM120_WARMUP": "0",
            "NFN_SM120_PROFILE_DIR": "none",
            "NFN_SM120_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_JSON_OUT": str(output_path),
        }
    )

    proc = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    assert "samples: 1" in proc.stdout
    assert "warmup: 0" in proc.stdout
    assert "cuda_visible_devices: requested=7 resolved=7 mode=explicit" in proc.stdout
    assert "--max-steps 2" in proc.stdout
    assert "--append-native-profile-json-dir" not in proc.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["selected_gpu_utilization_retries"] == 3
    assert payload["selected_gpu_utilization_retry_interval_seconds"] == 0.25


def test_native_gpt_sm120_parity_wrapper_stage_timing_without_profile_dir(tmp_path: Path) -> None:
    script = Path("tools/bench_native_gpt_sm120_parity.sh")
    output_path = tmp_path / "parity-stage-no-profile.json"

    env = os.environ.copy()
    env.update(
        {
            "NFN_SM120_DRY_RUN_PLAN": "1",
            "NFN_SM120_STEPS": "2",
            "NFN_SM120_SAMPLES": "1",
            "NFN_SM120_WARMUP": "0",
            "NFN_SM120_PROFILE_DIR": "none",
            "NFN_SM120_STAGE_TIMING": "1",
            "NFN_SM120_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_JSON_OUT": str(output_path),
        }
    )

    proc = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    assert "--append-native-profile-json-dir" not in proc.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["native_stage_timing"] is True
    assert payload["append_native_profile_json_dir"] == ""


def test_native_gpt_sm120_candidate_wrapper_defaults_measured_candidate_gates(tmp_path: Path) -> None:
    script = Path("tools/bench_native_gpt_sm120_candidate.sh")
    output_path = tmp_path / "candidate-default-gates.json"

    proc = subprocess.run(
        ["bash", "-n", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    text = script.read_text(encoding="utf-8")
    assert 'MAX_CANDIDATE_RATIO_RAW="setup_wall_ms=1.000"' in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" setup.token_weight_init.total_ms=1.000"' in text
    assert "*TOKEN_WEIGHT*|*token_weight*" in text
    assert 'MAX_CANDIDATE_RATIO_RAW="train_loop_wall_ms_per_step=1.000"' in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.total_ms=1.000"' in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.total_ms=1.000"' in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.mlp_proj.total_ms=1.000"' in text
    assert "*CE_BF16*|*ce_bf16*" in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.ce.total_ms=1.000"' in text
    assert "*LM_HEAD_PREPACK_BF16_HIDDEN*|*lm_head_prepack_bf16_hidden*" in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.dhidden.total_ms=1.000"' in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.dweight.total_ms=1.000"' in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" setup.uint16_arena_materialize.total_ms=1.000"' in text
    assert "NFN_SM120_NATIVE_CANDIDATE_PROFILE" in text
    assert "NFN_SM120_CANDIDATE_PROFILE" in text
    assert "lm_head_tk_dinput_32768" in text
    assert "lm_head_cublaslt_dhidden_32768" in text
    assert "cublaslt_min_waves" in text
    assert "cublaslt_max_waves" in text
    assert "tk_dgelu_dinput" in text
    assert "tk_dgelu_approx_tanh" in text
    assert "attention_atomic_dq" in text
    assert "qkv_concurrent_dinput_dweight" in text
    assert "mlp_fc_concurrent_dinput_dweight" in text
    assert "lm_head_pipeline_chunks" in text
    assert "token_weight_vector4_strided" in text
    assert "token_weight_threaded" in text
    assert "token_weight_fast_int32" in text
    assert "token_weight_two_pass_bf16" in text
    assert "NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_BUILD_FLAGS" in text
    assert "NFN_SM120_CANDIDATE_TILE_OPS_BUILD_FLAGS" in text
    assert "-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP" in text
    assert "-DLLMK_SM120_APPROX_DGELU_TANH=1" in text
    assert "-DLLMK_SM120_ATOMIC_DQ" in text
    assert "tools/build_native_train_tile_ops.sh" in text
    assert "NFN_NATIVE_LINEAR_TK_DINPUT_ENABLE_SHAPE=768,32768,50304,N,N" in text
    assert "NFN_NATIVE_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE=768,32768,50304,N,N" in text
    assert "*TK_DINPUT*|*tk_dinput*|*CUBLASLT_ENABLE_SHAPE*|*cublaslt_enable_shape*" in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.mlp_proj.dinput.total_ms=1.000"' in text
    assert "*ATOMIC_DQ*|*atomic_dq*|*attention_atomic_dq*|*attention-atomic-dq*" in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.attn_sdpa.total_ms=1.000"' in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.attn_sdpa.to_qkv.total_ms=1.000"' in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" attention_backward_tk_timing_us=1.000"' in text
    assert "*PACKED_ATTENTION*|*packed_attention*|*BF16_ATTENTION*|*bf16_attention*" in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" attention_backward_dprep_timing_us=1.000"' in text
    assert "NFN_NATIVE_GPT_LM_HEAD_PIPELINE_CHUNKS=1" in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.pipeline_queue.total_ms=1.000"' not in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.pipeline_final_wait.total_ms=1.000"' not in text
    assert "*BLOCK_QKV_CONCURRENT_DINPUT_DWEIGHT*|*block_qkv_concurrent_dinput_dweight*" in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.qkv.total_ms=1.000"' in text
    assert "*BLOCK_MLP_FC_CONCURRENT_DINPUT_DWEIGHT*|*block_mlp_fc_concurrent_dinput_dweight*" in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.mlp_fc.total_ms=1.000"' in text
    assert '"1"|"true"|"yes"|"on")' in text
    assert "has_candidate_change=0" in text

    env = os.environ.copy()
    env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_ENV": "NFN_CANDIDATE_ONLY=1",
            "NFN_SM120_NATIVE_JSON_OUT": str(output_path),
        }
    )

    dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )

    assert dry_run.returncode == 0, dry_run.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["metric_ratio_gates"]["enabled"] is False

    profile_output_path = tmp_path / "candidate-profile-dry-run.json"
    profile_env = os.environ.copy()
    profile_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_tk_dinput_32768",
            "NFN_SM120_NATIVE_JSON_OUT": str(profile_output_path),
        }
    )

    profile_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=profile_env,
    )

    assert profile_dry_run.returncode == 0, profile_dry_run.stderr
    profile_payload = json.loads(profile_output_path.read_text(encoding="utf-8"))
    assert (
        profile_payload["candidate_env"]["NFN_NATIVE_LINEAR_TK_DINPUT_ENABLE_SHAPE"]
        == "768,32768,50304,N,N"
    )
    assert profile_payload["metric_ratio_gates"]["enabled"] is False

    min_waves_output_path = tmp_path / "candidate-cublaslt-min-waves-dry-run.json"
    min_waves_env = os.environ.copy()
    min_waves_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "cublaslt_min_waves",
            "NFN_SM120_NATIVE_JSON_OUT": str(min_waves_output_path),
        }
    )

    min_waves_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=min_waves_env,
    )

    assert min_waves_dry_run.returncode == 0, min_waves_dry_run.stderr
    min_waves_payload = json.loads(min_waves_output_path.read_text(encoding="utf-8"))
    assert (
        min_waves_payload["candidate_env"][
            "NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_POLICY"
        ]
        == "min_waves"
    )
    assert min_waves_payload["metric_ratio_gates"]["enabled"] is False

    max_waves_output_path = tmp_path / "candidate-cublaslt-max-waves-dry-run.json"
    max_waves_env = os.environ.copy()
    max_waves_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "cublaslt_max_waves",
            "NFN_SM120_NATIVE_JSON_OUT": str(max_waves_output_path),
        }
    )

    max_waves_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=max_waves_env,
    )

    assert max_waves_dry_run.returncode == 0, max_waves_dry_run.stderr
    max_waves_payload = json.loads(max_waves_output_path.read_text(encoding="utf-8"))
    assert (
        max_waves_payload["candidate_env"][
            "NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_POLICY"
        ]
        == "max_waves"
    )
    assert max_waves_payload["metric_ratio_gates"]["enabled"] is False

    qkv_output_path = tmp_path / "candidate-qkv-concurrent-dry-run.json"
    qkv_env = os.environ.copy()
    qkv_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "qkv_concurrent_dinput_dweight",
            "NFN_SM120_NATIVE_JSON_OUT": str(qkv_output_path),
        }
    )

    qkv_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=qkv_env,
    )

    assert qkv_dry_run.returncode == 0, qkv_dry_run.stderr
    qkv_payload = json.loads(qkv_output_path.read_text(encoding="utf-8"))
    assert (
        qkv_payload["candidate_env"][
            "NFN_NATIVE_GPT_BLOCK_QKV_CONCURRENT_DINPUT_DWEIGHT"
        ]
        == "1"
    )
    assert qkv_payload["metric_ratio_gates"]["enabled"] is False

    token_profiles = {
        "token_weight_vector4_strided": {
            "NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT": "1"
        },
        "token_weight_threaded": {
            "NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT": "1"
        },
        "token_weight_fast_int32": {
            "NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT": "0"
        },
        "token_weight_two_pass_bf16": {
            "NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_INIT": "0"
        },
    }
    for profile_name, expected_env in token_profiles.items():
        token_output_path = tmp_path / f"candidate-{profile_name}-dry-run.json"
        token_env = os.environ.copy()
        token_env.update(
            {
                "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
                "NFN_SM120_NATIVE_STARTUP_ONLY": "1",
                "NFN_SM120_NATIVE_PROFILE_DIR": "none",
                "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
                "NFN_SM120_NATIVE_CANDIDATE_PROFILE": profile_name,
                "NFN_SM120_NATIVE_JSON_OUT": str(token_output_path),
            }
        )

        token_dry_run = subprocess.run(
            ["bash", str(script)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=token_env,
        )

        assert token_dry_run.returncode == 0, token_dry_run.stderr
        token_payload = json.loads(token_output_path.read_text(encoding="utf-8"))
        for env_name, env_value in expected_env.items():
            assert token_payload["candidate_env"][env_name] == env_value
        assert token_payload["metric_ratio_gates"]["enabled"] is False


def test_paired_kernel_speed_tool_applies_command_specific_env() -> None:
    script = Path("tools/paired_kernel_speed.py")
    output_path = Path(tempfile.mkdtemp()) / "paired-env.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline",
            f"{sys.executable} -c \"import os; print(os.environ.get('NFN_BASELINE_ONLY', 'missing'))\"",
            "--candidate",
            f"{sys.executable} -c \"import os; print(os.environ.get('NFN_CANDIDATE_ONLY', 'missing'))\"",
            "--baseline-env",
            "NFN_BASELINE_ONLY=old",
            "--candidate-env",
            "NFN_CANDIDATE_ONLY=new",
            "--samples",
            "1",
            "--warmup",
            "0",
            "--json-out",
            str(output_path),
            "--cuda-visible-devices",
            "",
            "--cuda-device-max-connections",
            "",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert 'baseline_env: {"NFN_BASELINE_ONLY": "old"}' in proc.stdout
    assert 'candidate_env: {"NFN_CANDIDATE_ONLY": "new"}' in proc.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["baseline_env"] == {"NFN_BASELINE_ONLY": "old"}
    assert payload["candidate_env"] == {"NFN_CANDIDATE_ONLY": "new"}
    assert "old\n" in payload["paired_samples"][0]["baseline"]["stdout_tail"]
    assert "new\n" in payload["paired_samples"][0]["candidate"]["stdout_tail"]


def test_paired_kernel_speed_tool_warns_when_candidate_env_does_not_change_route_counters() -> None:
    script = Path("tools/paired_kernel_speed.py")
    output_path = Path(tempfile.mkdtemp()) / "paired-route.json"
    native_json = (
        "{"
        "\\\"steps_completed\\\": 1, "
        "\\\"timing\\\": {\\\"train_loop_wall_ms\\\": 10.0}, "
        "\\\"linear_tk_gemm_count\\\": 1632, "
        "\\\"linear_cublaslt_gemm_count\\\": 2208, "
        "\\\"linear_bf16_gemm_count\\\": 1824"
        "}"
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline",
            f"{sys.executable} -c \"print('{native_json}')\"",
            "--candidate",
            f"{sys.executable} -c \"print('{native_json}')\"",
            "--candidate-env",
            "NFN_NATIVE_LINEAR_TK_DINPUT_DISABLE_SHAPE=3072,65536,768,N,N",
            "--samples",
            "1",
            "--warmup",
            "0",
            "--json-out",
            str(output_path),
            "--cuda-visible-devices",
            "",
            "--cuda-device-max-connections",
            "",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    route_changes = payload["native_route_counter_changes"]
    assert route_changes["has_route_counter_change"] is False
    assert route_changes["changed_count"] == 0
    assert route_changes["tracked_count"] == 3
    assert route_changes["unchanged"] == [
        "linear_tk_gemm_count",
        "linear_cublaslt_gemm_count",
        "linear_bf16_gemm_count",
    ]
    assert "native_route_counter_changes: has_route_counter_change=false changed_count=0" in proc.stdout
    assert "tracked route counters did not change" in proc.stdout


def test_paired_kernel_speed_tool_reports_cublaslt_plan_change_without_route_counter_warning() -> None:
    script = Path("tools/paired_kernel_speed.py")
    output_path = Path(tempfile.mkdtemp()) / "paired-plan-change.json"
    baseline_json = (
        "{"
        "\\\"steps_completed\\\": 1, "
        "\\\"timing\\\": {\\\"train_loop_wall_ms\\\": 10.0}, "
        "\\\"linear_tk_gemm_count\\\": 1632, "
        "\\\"linear_cublaslt_gemm_count\\\": 2208, "
        "\\\"linear_bf16_gemm_count\\\": 1824, "
        "\\\"linear_shape_stats\\\": [{"
        "\\\"path_name\\\": \\\"cublaslt\\\", "
        "\\\"m\\\": 768, \\\"n\\\": 50304, \\\"k\\\": 32768, "
        "\\\"op_a_name\\\": \\\"N\\\", \\\"op_b_name\\\": \\\"T\\\", "
        "\\\"calls\\\": 16, \\\"total_us\\\": 160000, \\\"avg_us\\\": 10000, "
        "\\\"cublaslt_selected_heuristic\\\": 1, "
        "\\\"cublaslt_returned_heuristics\\\": 9, "
        "\\\"cublaslt_workspace_bytes\\\": 134217728"
        "}]"
        "}"
    )
    candidate_json = baseline_json.replace(
        "\\\"cublaslt_selected_heuristic\\\": 1",
        "\\\"cublaslt_selected_heuristic\\\": 0",
    ).replace(
        "\\\"total_us\\\": 160000, \\\"avg_us\\\": 10000",
        "\\\"total_us\\\": 158400, \\\"avg_us\\\": 9900",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline",
            f"{sys.executable} -c \"print('{baseline_json}')\"",
            "--candidate",
            f"{sys.executable} -c \"print('{candidate_json}')\"",
            "--candidate-env",
            "NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,50304,32768,N,T,0",
            "--samples",
            "1",
            "--warmup",
            "0",
            "--json-out",
            str(output_path),
            "--cuda-visible-devices",
            "",
            "--cuda-device-max-connections",
            "",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    route_changes = payload["native_route_counter_changes"]
    assert route_changes["has_route_counter_change"] is False
    shape_stats = payload["native_linear_shape_stats"]
    assert shape_stats["has_cublaslt_plan_change"] is True
    assert shape_stats["cublaslt_plan_changed_count"] == 1
    plan_change = shape_stats["cublaslt_plan_changed"][0]
    assert plan_change["shape"] == "cublaslt:768x50304x32768:N,T"
    assert plan_change["baseline_selected_heuristics"] == [1]
    assert plan_change["candidate_selected_heuristics"] == [0]
    assert plan_change["candidate_avg_us_over_baseline"] == 0.99
    assert "cublaslt_plan_changes: changed_count=1" in proc.stdout
    assert "tracked route counters did not change" not in proc.stdout


def test_paired_kernel_speed_tool_reports_startup_strategy_values() -> None:
    script = Path("tools/paired_kernel_speed.py")
    output_path = Path(tempfile.mkdtemp()) / "paired-startup-strategy.json"
    baseline_json = (
        "{"
        "\\\"steps_completed\\\": 0, "
        "\\\"device_allocator_strategy\\\": \\\"cudaMalloc\\\", "
        "\\\"device_cuda_malloc_async_requested\\\": false, "
        "\\\"device_cuda_malloc_async_enabled\\\": false, "
        "\\\"device_cuda_malloc_async_symbol_loaded\\\": true, "
        "\\\"device_cuda_free_async_symbol_loaded\\\": true, "
        "\\\"skip_exit_device_free_enabled\\\": true, "
        "\\\"token_weight_init_strategy\\\": \\\"device-vector4-power2-deterministic-fused-bf16-shadow\\\", "
        "\\\"token_weight_vector4_init_enabled\\\": true, "
        "\\\"token_weight_fast_int32_init_enabled\\\": true, "
        "\\\"token_weight_bf16_initial_refresh_fusion_enabled\\\": true"
        "}"
    )
    candidate_json = baseline_json.replace(
        "\\\"device_allocator_strategy\\\": \\\"cudaMalloc\\\"",
        "\\\"device_allocator_strategy\\\": \\\"cudaMallocAsync-null-stream\\\"",
    ).replace(
        "\\\"device_cuda_malloc_async_requested\\\": false",
        "\\\"device_cuda_malloc_async_requested\\\": true",
    ).replace(
        "\\\"device_cuda_malloc_async_enabled\\\": false",
        "\\\"device_cuda_malloc_async_enabled\\\": true",
    ).replace(
        "\\\"device-vector4-power2-deterministic-fused-bf16-shadow\\\"",
        "\\\"device-vector4-strided-power2-deterministic-fused-bf16-shadow\\\"",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline",
            f"{sys.executable} -c \"print('{baseline_json}')\"",
            "--candidate",
            f"{sys.executable} -c \"print('{candidate_json}')\"",
            "--samples",
            "1",
            "--warmup",
            "0",
            "--json-out",
            str(output_path),
            "--cuda-visible-devices",
            "",
            "--cuda-device-max-connections",
            "",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["baseline_native_metric_values"]["device_allocator_strategy"] == [
        "cudaMalloc"
    ]
    assert payload["candidate_native_metric_values"]["device_allocator_strategy"] == [
        "cudaMallocAsync-null-stream"
    ]
    assert payload["candidate_native_metric_values"]["device_cuda_malloc_async_enabled"] == [
        "true"
    ]
    assert payload["candidate_native_metric_values"]["token_weight_init_strategy"] == [
        "device-vector4-strided-power2-deterministic-fused-bf16-shadow"
    ]
    assert "device_allocator_strategy: cudaMallocAsync-null-stream" in proc.stdout
    assert (
        "token_weight_init_strategy: "
        "device-vector4-strided-power2-deterministic-fused-bf16-shadow"
    ) in proc.stdout


def test_paired_kernel_speed_tool_fails_metric_ratio_gate() -> None:
    script = Path("tools/paired_kernel_speed.py")
    output_path = Path(tempfile.mkdtemp()) / "paired-ratio-gate.json"
    baseline_json = (
        "{"
        "\\\"steps_completed\\\": 1, "
        "\\\"timing\\\": {"
        "\\\"train_loop_wall_ms\\\": 10.0, "
        "\\\"stage_timing\\\": [{\\\"name\\\": \\\"lm_head_backward\\\", \\\"total_ms\\\": 5.0}]"
        "}"
        "}"
    )
    candidate_json = (
        "{"
        "\\\"steps_completed\\\": 1, "
        "\\\"timing\\\": {"
        "\\\"train_loop_wall_ms\\\": 12.0, "
        "\\\"stage_timing\\\": [{\\\"name\\\": \\\"lm_head_backward\\\", \\\"total_ms\\\": 6.5}]"
        "}"
        "}"
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline",
            f"{sys.executable} -c \"print('{baseline_json}')\"",
            "--candidate",
            f"{sys.executable} -c \"print('{candidate_json}')\"",
            "--samples",
            "1",
            "--warmup",
            "0",
            "--json-out",
            str(output_path),
            "--cuda-visible-devices",
            "",
            "--cuda-device-max-connections",
            "",
            "--max-candidate-ratio",
            "stage.lm_head_backward.total_ms=1.1",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    gates = payload["metric_ratio_gates"]
    assert gates["enabled"] is True
    assert gates["passed"] is False
    assert gates["results"] == [
        {
            "metric": "stage.lm_head_backward.total_ms",
            "stat": "mean",
            "max_ratio": 1.1,
            "actual_ratio": 1.3,
            "actual_mean_ratio": 1.3,
            "missing": False,
            "passed": False,
        }
    ]
    assert "metric_ratio_gates: passed=false" in proc.stdout
    assert "mean:stage.lm_head_backward.total_ms: actual_ratio=1.300000" in proc.stdout
    assert "metric ratio gate failed: stage.lm_head_backward.total_ms" in proc.stderr


def test_paired_kernel_speed_tool_fails_metric_ratio_gate_by_median() -> None:
    script = Path("tools/paired_kernel_speed.py")
    tmp = Path(tempfile.mkdtemp())
    output_path = tmp / "paired-ratio-gate-median.json"
    counter_path = tmp / "candidate-counter.txt"
    candidate_script = tmp / "candidate.py"
    baseline_json = (
        "{"
        "\\\"steps_completed\\\": 1, "
        "\\\"timing\\\": {\\\"train_loop_wall_ms\\\": 100.0}"
        "}"
    )
    candidate_script.write_text(
        "from pathlib import Path\n"
        f"counter = Path({str(counter_path)!r})\n"
        "idx = int(counter.read_text() if counter.exists() else '0')\n"
        "values = [90.0, 102.0, 103.0]\n"
        "counter.write_text(str(idx + 1))\n"
        "value = values[min(idx, len(values) - 1)]\n"
        "print('{\"steps_completed\": 1, \"timing\": {\"train_loop_wall_ms\": %s}}' % value)\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline",
            f"{sys.executable} -c \"print('{baseline_json}')\"",
            "--candidate",
            f"{sys.executable} {candidate_script}",
            "--samples",
            "3",
            "--warmup",
            "0",
            "--json-out",
            str(output_path),
            "--cuda-visible-devices",
            "",
            "--cuda-device-max-connections",
            "",
            "--max-candidate-ratio",
            "median:train_loop_wall_ms_per_step=1.0",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    result = payload["metric_ratio_gates"]["results"][0]
    assert result["metric"] == "train_loop_wall_ms_per_step"
    assert result["stat"] == "median"
    assert result["actual_ratio"] == 1.02
    assert result["passed"] is False
    assert "median:train_loop_wall_ms_per_step: actual_ratio=1.020000" in proc.stdout


def test_paired_kernel_speed_tool_metric_ratio_gate_fails_missing_metric() -> None:
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

    gates = module.evaluate_metric_ratio_limits(
        {"candidate_over_baseline_native_metrics": {}},
        [module.MetricRatioLimit(metric="stage.lm_head_backward.total_ms", max_ratio=1.01)],
    )

    assert gates == {
        "enabled": True,
        "passed": False,
        "results": [
            {
                "metric": "stage.lm_head_backward.total_ms",
                "stat": "mean",
                "max_ratio": 1.01,
                "actual_ratio": None,
                "actual_mean_ratio": None,
                "missing": True,
                "passed": False,
            }
        ],
    }


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
    assert metrics["train_loop_wall_ms_per_step"] == 2493.74
    assert metrics["train_tokens_per_second"] == 210242.0
    assert metrics["llm_kittens_bf16_mfu_pct"] == 40.3
    assert metrics["llm_kittens_device_memory_used_mib"] == 28819
    assert metrics["llm_kittens_device_memory_total_mib"] == 32606


def test_paired_kernel_speed_tool_sums_llm_kittens_step_time() -> None:
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
step    1/2 | loss 11.0 (+nanz)| norm 22.1 (+nanz)| lr 1.00e-05 | 2400.00 ms | 40.0% bf16 MFU | 210000 tok/s
step    2/2 | loss 10.0 (+nanz)| norm 20.0 (+nanz)| lr 2.00e-05 | 2600.00 ms | 42.0% bf16 MFU | 220000 tok/s
"""

    metrics = module.native_metrics_from_stdout(stdout)
    assert metrics["status"] == "llm-kittens-step-log"
    assert metrics["train_loop_wall_ms"] == 5000.0
    assert metrics["train_loop_wall_ms_per_step"] == 2500.0
    assert metrics["train_tokens_per_second"] == 215000.0
    assert metrics["llm_kittens_bf16_mfu_pct"] == 41.0
    assert metrics["llm_kittens_last_step_wall_ms"] == 2600.0
    assert metrics["llm_kittens_last_step_tokens_per_second"] == 220000.0
    assert metrics["llm_kittens_step_log_count"] == 2


def test_paired_kernel_speed_tool_summarizes_native_route_counter_changes() -> None:
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

    baseline = {
        "linear_tk_gemm_count": {"mean": 1632.0, "median": 1632.0, "min": 1632.0, "max": 1632.0},
        "linear_cublaslt_gemm_count": {"mean": 2208.0, "median": 2208.0, "min": 2208.0, "max": 2208.0},
    }
    candidate = {
        "linear_tk_gemm_count": {"mean": 1344.0, "median": 1344.0, "min": 1344.0, "max": 1344.0},
        "linear_cublaslt_gemm_count": {"mean": 2208.0, "median": 2208.0, "min": 2208.0, "max": 2208.0},
    }

    changes = module.summarize_native_route_counter_changes(baseline, candidate)

    assert changes["has_route_counter_change"] is True
    assert changes["changed_count"] == 1
    assert changes["tracked_count"] == 2
    assert changes["changed"]["linear_tk_gemm_count"] == {
        "baseline_mean": 1632.0,
        "candidate_mean": 1344.0,
        "delta": -288.0,
        "ratio": 1344.0 / 1632.0,
    }
    assert changes["unchanged"] == ["linear_cublaslt_gemm_count"]
    assert "linear_bf16_gemm_count" in changes["missing"]


def test_paired_kernel_speed_tool_reads_native_json_out_sidecar(tmp_path: Path) -> None:
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

    sidecar = tmp_path / "native-profile.json"
    sidecar.write_text(
        json.dumps(
            {
                "status": "native-sidecar-test",
                "steps_completed": 4,
                "timing": {
                    "train_loop_wall_ms": 20.0,
                    "train_tokens_per_second": 123.0,
                    "stage_timing": [
                        {"name": "block_backward", "total_ms": 9.0, "avg_ms": 3.0, "count": 3}
                    ],
                },
                "linear_tk_gemm_count": 8,
                "linear_cublaslt_gemm_count": 11,
                "linear_bf16_gemm_count": 13,
                "lm_head_logits_tk_gemm_count": 4,
                "lm_head_logits_cublaslt_gemm_count": 0,
                "lm_head_logits_bf16_gemm_count": 4,
                "lm_head_classifier_chunk_kernel_available": True,
                "lm_head_classifier_chunk_kernel_enabled": True,
                "lm_head_classifier_chunk_launch_count": 32,
                "lm_head_classifier_last_rows": 4096,
                "lm_head_classifier_last_vocab": 50257,
                "lm_head_classifier_last_row_stride": 50304,
                "attention_backward_dprep_timing_us": 30000,
                "attention_backward_dprep_timing_count": 12,
                "attention_backward_tk_timing_us": 240000,
                "attention_backward_tk_timing_count": 96,
                "lm_head_logits_linear_strategy": "padded-lm-head-bf16-cublaslt-fallback",
                "lm_head_dhidden_linear_strategy": "bf16-cublas-gemmex",
                "lm_head_ce_loss_backward_strategy": "separate-loss-partials-reduction-then-dlogits",
                "lm_head_classifier_strategy_contract": {
                    "reference_full_bf16_logit_bytes": 6593445888,
                    "native_chunk_bf16_logit_bytes": 825819136,
                    "resident_logit_reduction_ratio": 8.0,
                    "native_logit_chunk_rows": 8192,
                    "native_logit_chunk_count": 8,
                },
                "linear_shape_stats": [
                    {
                        "path_name": "cublaslt",
                        "m": 768,
                        "n": 50304,
                        "k": 8192,
                        "op_a_name": "N",
                        "op_b_name": "T",
                        "calls": 8,
                        "total_us": 4000,
                        "avg_us": 500,
                        "cublaslt_selected_heuristic": 1,
                        "cublaslt_returned_heuristics": 8,
                        "cublaslt_workspace_bytes": 134217728,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    metrics = module.native_metrics_from_command_output(
        ["nfn_gpt_native_train", "--profile-json", str(sidecar)],
        "",
    )

    assert metrics["native_metrics_source"] == "json-out"
    assert metrics["status"] == "native-sidecar-test"
    assert metrics["steps_completed"] == 4
    assert metrics["train_loop_wall_ms"] == 20.0
    assert metrics["train_loop_wall_ms_per_step"] == 5.0
    assert metrics["train_tokens_per_second"] == 123.0
    assert metrics["linear_tk_gemm_count"] == 8
    assert metrics["linear_cublaslt_gemm_count"] == 11
    assert metrics["linear_bf16_gemm_count"] == 13
    assert metrics["lm_head_logits_tk_gemm_count"] == 4
    assert metrics["lm_head_logits_cublaslt_gemm_count"] == 0
    assert metrics["lm_head_logits_bf16_gemm_count"] == 4
    assert metrics["lm_head_classifier_chunk_kernel_available"] is True
    assert metrics["lm_head_classifier_chunk_kernel_enabled"] is True
    assert metrics["lm_head_classifier_chunk_launch_count"] == 32
    assert metrics["lm_head_classifier_last_rows"] == 4096
    assert metrics["lm_head_classifier_last_vocab"] == 50257
    assert metrics["lm_head_classifier_last_row_stride"] == 50304
    assert metrics["stage.block_backward.total_ms"] == 9.0
    assert metrics["attention_backward_dprep_timing_us"] == 30000
    assert metrics["attention_backward_dprep_timing_count"] == 12
    assert metrics["attention_backward_tk_timing_us"] == 240000
    assert metrics["attention_backward_tk_timing_count"] == 96
    assert metrics["lm_head_logits_linear_strategy"] == "padded-lm-head-bf16-cublaslt-fallback"
    assert metrics["lm_head_dhidden_linear_strategy"] == "bf16-cublas-gemmex"
    assert metrics["lm_head_ce_loss_backward_strategy"] == "separate-loss-partials-reduction-then-dlogits"
    assert metrics["lm_head_classifier.reference_full_bf16_logit_bytes"] == 6593445888
    assert metrics["lm_head_classifier.native_chunk_bf16_logit_bytes"] == 825819136
    assert metrics["lm_head_classifier.resident_logit_reduction_ratio"] == 8.0
    assert metrics["lm_head_classifier.native_logit_chunk_rows"] == 8192
    assert metrics["lm_head_classifier.native_logit_chunk_count"] == 8
    shape_stats = module.native_linear_shape_stats_from_command_output(
        ["nfn_gpt_native_train", "--profile-json", str(sidecar)],
        "",
    )
    assert shape_stats == [
        {
            "path_name": "cublaslt",
            "m": 768,
            "n": 50304,
            "k": 8192,
            "op_a_name": "N",
            "op_b_name": "T",
            "calls": 8,
            "total_us": 4000,
            "avg_us": 500,
            "cublaslt_selected_heuristic": 1,
            "cublaslt_returned_heuristics": 8,
            "cublaslt_workspace_bytes": 134217728,
        }
    ]

    rows = [{"baseline": {"native_metrics": metrics}, "candidate": {"native_metrics": metrics}}]
    assert module.summarize_categorical_metric_rows(rows, "baseline") == {
        "status": ["native-sidecar-test"],
        "lm_head_logits_linear_strategy": ["padded-lm-head-bf16-cublaslt-fallback"],
        "lm_head_dhidden_linear_strategy": ["bf16-cublas-gemmex"],
        "lm_head_ce_loss_backward_strategy": ["separate-loss-partials-reduction-then-dlogits"],
    }


def test_paired_kernel_speed_tool_summarizes_linear_shape_stats() -> None:
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

    baseline_stat = {
        "path_name": "cublaslt",
        "m": 768,
        "n": 50304,
        "k": 8192,
        "op_a_name": "N",
        "op_b_name": "T",
        "calls": 8,
        "total_us": 4000,
        "avg_us": 500,
        "cublaslt_selected_heuristic": 1,
        "cublaslt_returned_heuristics": 8,
        "cublaslt_workspace_bytes": 134217728,
    }
    candidate_stat = dict(baseline_stat, avg_us=450, total_us=3600, cublaslt_selected_heuristic=0)
    rows = [
        {
            "baseline": {"native_linear_shape_stats": [baseline_stat]},
            "candidate": {"native_linear_shape_stats": [candidate_stat]},
        }
    ]

    summary = module.summarize_linear_shape_stats(rows)

    assert summary["enabled"] is True
    assert summary["shared_count"] == 1
    row = summary["shared"][0]
    assert row["shape"] == "cublaslt:768x50304x8192:N,T"
    assert row["candidate_avg_us_over_baseline"] == 0.9
    assert row["baseline"]["cublaslt_selected_heuristics"] == [1]
    assert row["candidate"]["cublaslt_selected_heuristics"] == [0]
    assert summary["has_cublaslt_plan_change"] is True
    assert summary["cublaslt_plan_changed_count"] == 1
    assert summary["cublaslt_plan_changed"][0]["shape"] == "cublaslt:768x50304x8192:N,T"


def test_paired_kernel_speed_failure_reports_native_json_sidecar_error(tmp_path: Path) -> None:
    script = Path("tools/paired_kernel_speed.py")
    failing = tmp_path / "native_fail.py"
    output_path = tmp_path / "paired.json"
    failing.write_text(
        "import json\n"
        "import sys\n"
        "path = sys.argv[sys.argv.index('--profile-json') + 1]\n"
        "open(path, 'w', encoding='utf-8').write(json.dumps({\n"
        "    'status': 'native-transformer-lm-failed',\n"
        "    'error': 'CUDA driver is unavailable to the native trainer',\n"
        "}))\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline",
            f"{sys.executable} -c \"print('baseline-ok')\"",
            "--candidate",
            f"{sys.executable} {failing} --profile-json {tmp_path / 'candidate.json'}",
            "--samples",
            "1",
            "--warmup",
            "0",
            "--json-out",
            str(output_path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 1
    assert "candidate failed with exit 7" in proc.stderr
    assert "native JSON summary:" in proc.stderr
    assert "status: native-transformer-lm-failed" in proc.stderr
    assert "error: CUDA driver is unavailable to the native trainer" in proc.stderr


def test_paired_kernel_speed_failure_reports_stdout_tail(tmp_path: Path) -> None:
    script = Path("tools/paired_kernel_speed.py")
    output_path = tmp_path / "paired.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline",
            (
                f"{sys.executable} -c "
                "\"print('CUDA driver version is insufficient for CUDA runtime version'); "
                "raise SystemExit(7)\""
            ),
            "--candidate",
            f"{sys.executable} -c \"print('candidate-ok')\"",
            "--samples",
            "1",
            "--warmup",
            "0",
            "--json-out",
            str(output_path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 1
    assert "baseline failed with exit 7" in proc.stderr
    assert "stdout tail:" in proc.stderr
    assert "CUDA driver version is insufficient for CUDA runtime version" in proc.stderr
    assert "stderr tail:" in proc.stderr


def test_paired_kernel_speed_tool_stage_timing_is_explicit() -> None:
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

    command = module.TimedCommand(
        name="candidate",
        argv=["nfn_gpt_native_train", "--profile-json", "/tmp/native.json"],
        env_overrides={},
    )

    assert (
        module.command_env_with_auto_stage_timing(
            command,
            env={},
            native_stage_timing=False,
        )
        == {}
    )
    assert module.command_env_with_auto_stage_timing(
        command,
        env={},
        native_stage_timing=True,
    )["NFN_NATIVE_GPT_STAGE_TIMING"] == "1"


def test_paired_kernel_speed_tool_extracts_forward_stage_timing() -> None:
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

    metrics = module.native_metrics_from_payload(
        {
            "timing": {
                "stage_timing": [
                    {"name": "train.model_forward", "total_ms": 12.5, "avg_ms": 2.5, "count": 5},
                    {"name": "block_forward.attention", "total_ms": 3.0, "avg_ms": 0.6, "count": 5},
                    {"name": "block_recompute.mlp_proj", "total_ms": 4.0, "avg_ms": 0.8, "count": 5},
                    {
                        "name": "block_backward.qkv.dinput_dweight_concurrent",
                        "total_ms": 6.0,
                        "avg_ms": 1.2,
                        "count": 5,
                    },
                ]
            }
        }
    )

    assert metrics["stage.train.model_forward.total_ms"] == 12.5
    assert metrics["stage.block_forward.attention.total_ms"] == 3.0
    assert metrics["stage.block_recompute.mlp_proj.count"] == 5
    assert metrics["stage.block_backward.qkv.dinput_dweight_concurrent.total_ms"] == 6.0


def test_paired_kernel_speed_tool_auto_selects_idle_display_disabled_gpu() -> None:
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

    snapshot = {
        "gpus": [
            {
                "index": "0",
                "name": "NVIDIA GeForce RTX 5090",
                "uuid": "GPU-compute",
                "pci.bus_id": "00000000:01:00.0",
                "display_active": "Disabled",
                "utilization.gpu_pct": "0",
                "memory.used_mib": "334",
                "memory.total_mib": "32607",
            },
            {
                "index": "1",
                "name": "NVIDIA Display Adapter",
                "uuid": "GPU-display",
                "pci.bus_id": "00000000:02:00.0",
                "display_active": "Enabled",
                "utilization.gpu_pct": "1",
                "memory.used_mib": "1024",
                "memory.total_mib": "16384",
            },
        ],
        "compute_processes": [
            {
                "gpu_uuid": "GPU-display",
                "pid": "1234",
                "process_name": "desktop",
                "used_memory_mib": "512",
            }
        ],
    }

    selection = module.resolve_cuda_visible_devices("auto", snapshot)
    assert selection["resolved"] == "0"
    assert selection["mode"] == "auto-dedicated"

    explicit = module.resolve_cuda_visible_devices("", snapshot)
    assert explicit["resolved"] == ""
    assert explicit["mode"] == "unchanged"


def test_paired_kernel_speed_tool_gpu_lock_rejects_overlap() -> None:
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

    lock_path = module.gpu_benchmark_lock_path("unit-lock-test")
    assert lock_path is not None
    with module.GpuBenchmarkLock(lock_path, enabled=True, timeout_seconds=0.0) as first_lock:
        assert first_lock.acquired is True
        with pytest.raises(SystemExit, match="GPU lock is already held"):
            with module.GpuBenchmarkLock(lock_path, enabled=True, timeout_seconds=0.0):
                pass


def test_paired_kernel_speed_tool_require_idle_selected_gpu_checks_selected_uuid() -> None:
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

    idle_snapshot = {
        "gpus": [
            {
                "index": "0",
                "uuid": "GPU-compute",
                "display_active": "Disabled",
                "utilization.gpu_pct": "0",
                "memory.used_mib": "334",
            },
            {
                "index": "1",
                "uuid": "GPU-display",
                "display_active": "Enabled",
                "utilization.gpu_pct": "20",
                "memory.used_mib": "2048",
            },
        ],
        "compute_processes": [
            {
                "gpu_uuid": "GPU-display",
                "pid": "100",
                "process_name": "desktop",
                "used_memory_mib": "512",
            }
        ],
    }
    module.require_idle_selected_gpu(idle_snapshot, "0", phase="unit test")

    busy_snapshot = {
        **idle_snapshot,
        "compute_processes": [
            *idle_snapshot["compute_processes"],
            {
                "gpu_uuid": "GPU-compute",
                "pid": "200",
                "process_name": "trainer",
                "used_memory_mib": "4096",
            },
        ],
    }
    with pytest.raises(SystemExit, match="trainer"):
        module.require_idle_selected_gpu(busy_snapshot, "0", phase="unit test")


def test_paired_kernel_speed_tool_selected_gpu_utilization_guard() -> None:
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

    snapshot = {
        "gpus": [
            {
                "index": "0",
                "uuid": "GPU-compute",
                "display_active": "Disabled",
                "utilization.gpu_pct": "3",
                "memory.used_mib": "334",
            },
            {
                "index": "1",
                "uuid": "GPU-display",
                "display_active": "Enabled",
                "utilization.gpu_pct": "95",
                "memory.used_mib": "2048",
            },
        ],
        "compute_processes": [],
    }
    module.require_selected_gpu_utilization_at_most(
        snapshot,
        "0",
        3.0,
        phase="unit test",
    )
    with pytest.raises(SystemExit, match="utilization is 3%"):
        module.require_selected_gpu_utilization_at_most(
            snapshot,
            "0",
            2.0,
            phase="unit test",
        )


def test_paired_kernel_speed_tool_selected_gpu_utilization_guard_retries_idle_gpu() -> None:
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

    high_snapshot = {
        "gpus": [
            {
                "index": "0",
                "uuid": "GPU-compute",
                "display_active": "Disabled",
                "utilization.gpu_pct": "21",
                "memory.used_mib": "640",
            },
        ],
        "compute_processes": [],
    }
    low_snapshot = {
        **high_snapshot,
        "gpus": [
            {
                **high_snapshot["gpus"][0],
                "utilization.gpu_pct": "3",
            }
        ],
    }
    snapshots = iter([low_snapshot])

    module.enforce_selected_gpu_guards(
        high_snapshot,
        "0",
        require_idle=True,
        max_utilization_pct=15.0,
        utilization_retries=2,
        utilization_retry_interval_seconds=0.0,
        snapshot_supplier=lambda: next(snapshots),
        phase="unit test",
    )


def test_paired_kernel_speed_tool_selected_gpu_utilization_guard_rejects_busy_gpu_without_retry() -> None:
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

    busy_snapshot = {
        "gpus": [
            {
                "index": "0",
                "uuid": "GPU-compute",
                "display_active": "Disabled",
                "utilization.gpu_pct": "21",
                "memory.used_mib": "640",
            },
        ],
        "compute_processes": [
            {
                "gpu_uuid": "GPU-compute",
                "pid": "200",
                "process_name": "trainer",
                "used_memory_mib": "4096",
            },
        ],
    }

    with pytest.raises(SystemExit, match="trainer"):
        module.enforce_selected_gpu_guards(
            busy_snapshot,
            "0",
            require_idle=True,
            max_utilization_pct=15.0,
            utilization_retries=2,
            utilization_retry_interval_seconds=0.0,
            snapshot_supplier=lambda: pytest.fail("busy GPUs must not be retried"),
            phase="unit test",
        )


def test_paired_kernel_speed_tool_records_command_timeout() -> None:
    script = Path("tools/paired_kernel_speed.py")
    temp_dir = Path(tempfile.mkdtemp())
    output_path = temp_dir / "paired-timeout.json"
    marker_path = temp_dir / "child-survived.txt"
    child_script = temp_dir / "timeout_child.py"
    spawner_script = temp_dir / "timeout_spawner.py"
    child_script.write_text(
        "import time\n"
        "from pathlib import Path\n"
        "time.sleep(1.0)\n"
        f"Path({str(marker_path)!r}).write_text('alive')\n",
        encoding="utf-8",
    )
    spawner_script.write_text(
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        f"subprocess.Popen([sys.executable, {str(child_script)!r}])\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline",
            f"{sys.executable} -c \"print('baseline-ok')\"",
            "--candidate",
            f"{sys.executable} {spawner_script}",
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
    time.sleep(1.5)
    assert not marker_path.exists()


def test_paired_kernel_speed_tool_terminates_process_group() -> None:
    script = Path("tools/paired_kernel_speed.py")
    spec = importlib.util.spec_from_file_location("paired_kernel_speed", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    temp_dir = Path(tempfile.mkdtemp())
    marker_path = temp_dir / "child-survived.txt"
    child_script = temp_dir / "interrupt_child.py"
    spawner_script = temp_dir / "interrupt_spawner.py"
    child_script.write_text(
        "import time\n"
        "from pathlib import Path\n"
        "time.sleep(1.0)\n"
        f"Path({str(marker_path)!r}).write_text('alive')\n",
        encoding="utf-8",
    )
    spawner_script.write_text(
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        f"subprocess.Popen([sys.executable, {str(child_script)!r}])\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )

    proc = subprocess.Popen(
        [sys.executable, str(spawner_script)],
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(0.2)
        returncode = module.terminate_process_group(
            proc,
            first_signal=signal.SIGTERM,
            wait_seconds=1.0,
        )
    finally:
        if proc.poll() is None:
            proc.kill()

    assert returncode != 0
    time.sleep(1.2)
    assert not marker_path.exists()
