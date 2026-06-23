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
                "\\\"bf16_to_f32_vec4_count\\\": 5, "
                "\\\"linear_cublaslt_bgrad_gemm_count\\\": 2, "
                "\\\"linear_cublaslt_bgrad_direct_write_count\\\": 1, "
                "\\\"linear_cublaslt_bgrad_accumulate_count\\\": 1, "
                "\\\"float_arena_cuda_malloc_wall_ms\\\": 0.61, "
                "\\\"float_arena_pointer_assign_wall_ms\\\": 0.02, "
                "\\\"uint16_arena_cuda_malloc_wall_ms\\\": 0.51, "
                "\\\"uint16_arena_pointer_assign_wall_ms\\\": 0.01, "
                "\\\"transformer_device_arena_cuda_malloc_wall_ms\\\": 0.0, "
                "\\\"transformer_device_arena_pointer_assign_wall_ms\\\": 0.0, "
                "\\\"lm_head_logits_tk_gemm_count\\\": 2, "
                "\\\"lm_head_logits_cublaslt_gemm_count\\\": 0, "
                "\\\"lm_head_logits_bf16_gemm_count\\\": 2, "
                "\\\"lm_head_dhidden_tk_gemm_count\\\": 0, "
                "\\\"lm_head_dhidden_cublaslt_gemm_count\\\": 0, "
                "\\\"lm_head_dhidden_bf16_gemm_count\\\": 2, "
                "\\\"block_backward_dinput_tk_gemm_count\\\": 1, "
                "\\\"block_backward_dinput_cublaslt_gemm_count\\\": 0, "
                "\\\"block_backward_dinput_bf16_gemm_count\\\": 5, "
                "\\\"block_backward_mlp_proj_dinput_before_dweight_count\\\": 0, "
                "\\\"block_backward_mlp_fc_dinput_before_dweight_count\\\": 0, "
                "\\\"block_backward_attn_proj_dinput_before_dweight_count\\\": 0, "
                "\\\"lm_head_classifier_chunk_kernel_available\\\": true, "
                "\\\"lm_head_classifier_chunk_kernel_enabled\\\": true, "
                "\\\"lm_head_classifier_chunk_launch_count\\\": 64, "
                "\\\"lm_head_classifier_last_rows\\\": 8192, "
                "\\\"lm_head_classifier_last_vocab\\\": 50257, "
                "\\\"lm_head_classifier_last_row_stride\\\": 50304, "
                "\\\"block_state_layout\\\": {\\\"layer_norm_backward_affine_row_chunk_size\\\": 512}, "
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
    assert payload["allow_stale_selected_gpu_utilization_without_compute_processes"] is False
    assert "require_idle_selected_gpu: False" in proc.stdout
    assert "max_selected_gpu_utilization_pct: -1.0" in proc.stdout
    assert "allow_stale_selected_gpu_utilization_without_compute_processes: False" in proc.stdout
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
    assert payload["candidate_native_metrics"]["linear_cublaslt_bgrad_gemm_count"]["mean"] == 2.0
    assert (
        payload["candidate_native_metrics"]["linear_cublaslt_bgrad_direct_write_count"]["mean"]
        == 1.0
    )
    assert payload["candidate_native_metrics"]["linear_cublaslt_bgrad_accumulate_count"]["mean"] == 1.0
    assert payload["candidate_native_metrics"]["float_arena_cuda_malloc_wall_ms"]["mean"] == 0.61
    assert payload["candidate_native_metrics"]["float_arena_pointer_assign_wall_ms"]["mean"] == 0.02
    assert payload["candidate_native_metrics"]["uint16_arena_cuda_malloc_wall_ms"]["mean"] == 0.51
    assert payload["candidate_native_metrics"]["uint16_arena_pointer_assign_wall_ms"]["mean"] == 0.01
    assert payload["candidate_native_metrics"]["transformer_device_arena_cuda_malloc_wall_ms"]["mean"] == 0.0
    assert payload["candidate_native_metrics"]["transformer_device_arena_pointer_assign_wall_ms"]["mean"] == 0.0
    assert payload["candidate_native_metrics"]["linear_bf16_gemm_count"]["mean"] == 7.0
    assert payload["candidate_native_metrics"]["bf16_to_f32_vec4_count"]["mean"] == 5.0
    assert payload["candidate_native_metrics"]["lm_head_logits_tk_gemm_count"]["mean"] == 2.0
    assert payload["candidate_native_metrics"]["lm_head_logits_cublaslt_gemm_count"]["mean"] == 0.0
    assert payload["candidate_native_metrics"]["lm_head_logits_bf16_gemm_count"]["mean"] == 2.0
    assert payload["candidate_native_metrics"]["lm_head_dhidden_tk_gemm_count"]["mean"] == 0.0
    assert payload["candidate_native_metrics"]["lm_head_dhidden_cublaslt_gemm_count"]["mean"] == 0.0
    assert payload["candidate_native_metrics"]["lm_head_dhidden_bf16_gemm_count"]["mean"] == 2.0
    assert payload["candidate_native_metrics"]["block_backward_dinput_tk_gemm_count"]["mean"] == 1.0
    assert payload["candidate_native_metrics"]["block_backward_dinput_cublaslt_gemm_count"]["mean"] == 0.0
    assert payload["candidate_native_metrics"]["block_backward_dinput_bf16_gemm_count"]["mean"] == 5.0
    assert payload["candidate_native_metrics"]["lm_head_classifier_chunk_launch_count"]["mean"] == 64.0
    assert payload["candidate_native_metrics"]["lm_head_classifier_last_rows"]["mean"] == 8192.0
    assert payload["candidate_native_metrics"]["lm_head_classifier_last_vocab"]["mean"] == 50257.0
    assert payload["candidate_native_metrics"]["lm_head_classifier_last_row_stride"]["mean"] == 50304.0
    assert (
        payload["candidate_native_metrics"][
            "block_state_layout.layer_norm_backward_affine_row_chunk_size"
        ]["mean"]
        == 512.0
    )
    assert "linear_tk_gemm_count: mean=3.000000" in proc.stdout
    assert "linear_cublaslt_gemm_count: mean=4.000000" in proc.stdout
    assert "linear_cublaslt_bgrad_gemm_count: mean=2.000000" in proc.stdout
    assert "linear_cublaslt_bgrad_direct_write_count: mean=1.000000" in proc.stdout
    assert "linear_cublaslt_bgrad_accumulate_count: mean=1.000000" in proc.stdout
    assert "linear_bf16_gemm_count: mean=7.000000" in proc.stdout
    assert "bf16_to_f32_vec4_count: mean=5.000000" in proc.stdout
    assert "lm_head_logits_tk_gemm_count: mean=2.000000" in proc.stdout
    assert "lm_head_logits_cublaslt_gemm_count: mean=0.000000" in proc.stdout
    assert "lm_head_dhidden_bf16_gemm_count: mean=2.000000" in proc.stdout
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
    assert "NFN_SM120_PARITY_MAX_CANDIDATE_RATIO" in text
    assert "NFN_SM120_MAX_CANDIDATE_RATIO" in text
    assert "NFN_SM120_PARITY_ENFORCE_GATE" in text
    assert "NFN_SM120_ENFORCE_PARITY_GATE" in text
    assert "NFN_SM120_PARITY_ATTENTION_SECTION_TIMING" in text
    assert "NFN_SM120_ATTENTION_SECTION_TIMING" in text
    assert 'paired_args+=(--candidate-env "NFN_NATIVE_GPT_ATTENTION_BACKWARD_SECTION_TIMING=1")' in text
    assert "Unsupported NFN_SM120_PARITY_ATTENTION_SECTION_TIMING value" in text
    assert 'MAX_CANDIDATE_RATIO_RAW="train_loop_wall_ms_per_step=1.000"' in text
    assert 'case "${ENFORCE_GATE,,}"' in text
    assert "--max-candidate-ratio" in text
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
    assert "--native-cuda-activation \"$ACTIVATION\"" in text
    assert "--no-checkpoint" in text
    assert "--tile-ops-lib \"$NFN_NATIVE_TILE_OPS_ARG\"" in text
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
    assert "env_or_alias5()" in text
    assert "NFN_SM120_NATIVE_CANDIDATE_STEPS" in text
    assert "NFN_SM120_NATIVE_CANDIDATE_SAMPLES" in text
    assert "NFN_SM120_NATIVE_CANDIDATE_WARMUP" in text
    assert "NFN_SM120_NATIVE_CANDIDATE_CUDA_VISIBLE_DEVICES" in text
    assert "NFN_SM120_NATIVE_CANDIDATE_MAX_GPU_UTILIZATION_PCT" in text
    assert "NFN_SM120_NATIVE_CANDIDATE_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE" in text
    assert "NFN_SM120_NATIVE_CANDIDATE_DRY_RUN_PLAN" in text
    assert "NFN_SM120_NATIVE_CANDIDATE_TEMPLATE_NAME" in text
    assert "NFN_SM120_NATIVE_CANDIDATE_GRAPH_FILE" in text
    assert "append_env_overrides()" in text
    assert "[A-Za-z_][A-Za-z0-9_]*=" in text
    assert "NFN_SM120_NATIVE_REQUIRE_ROUTE_CHANGE" in text
    assert "--require-native-route-change" in text
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


def test_native_gpt_sm120_candidate_wrapper_accepts_native_candidate_common_aliases(tmp_path: Path) -> None:
    script = Path("tools/bench_native_gpt_sm120_candidate.sh")
    output_path = tmp_path / "native-candidate-common-alias.json"

    env = os.environ.copy()
    env.update(
        {
            "NFN_SM120_NATIVE_CANDIDATE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_CANDIDATE_STEPS": "4",
            "NFN_SM120_NATIVE_CANDIDATE_SAMPLES": "2",
            "NFN_SM120_NATIVE_CANDIDATE_WARMUP": "0",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CANDIDATE_JSON_OUT": str(output_path),
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
    assert "samples: 2" in proc.stdout
    assert "warmup: 0" in proc.stdout
    assert "--max-steps 4" in proc.stdout
    assert "/tmp/nfn_sm120_native_candidate_profiles_10step" not in proc.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["samples"] == 2
    assert payload["warmup"] == 0
    assert payload["allow_stale_selected_gpu_utilization_without_compute_processes"] is True
    baseline_max_steps = payload["baseline_command"].index("--max-steps")
    candidate_max_steps = payload["candidate_command"].index("--max-steps")
    assert payload["baseline_command"][baseline_max_steps + 1] == "4"
    assert payload["candidate_command"][candidate_max_steps + 1] == "4"


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
    assert payload["baseline_env"] == {
        "NFN_NATIVE_GPT_CUDA_VERSION_PREFLIGHT": "1",
        "NFN_NATIVE_GPT_TRAIN_LOOP_EVENT_TIMING": "1",
        "NFN_SHARED_PROFILING": "1",
    }
    assert payload["candidate_env"] == {
        "NFN_NATIVE_GPT_CUDA_VERSION_PREFLIGHT": "1",
        "NFN_NATIVE_GPT_TRAIN_LOOP_EVENT_TIMING": "1",
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
    assert payload["baseline_env"] == {
        "NFN_NATIVE_GPT_CUDA_VERSION_PREFLIGHT": "1",
        "NFN_NATIVE_GPT_TRAIN_LOOP_EVENT_TIMING": "1",
        "NFN_SHARED": "1",
        "NFN_SHARED_2": "2",
    }
    assert payload["candidate_env"] == {
        "NFN_NATIVE_GPT_CUDA_VERSION_PREFLIGHT": "1",
        "NFN_NATIVE_GPT_TRAIN_LOOP_EVENT_TIMING": "1",
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
            "NFN_SM120_NATIVE_STAGE_TIMING": "1",
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
    text = script.read_text(encoding="utf-8")
    assert "NFN_SM120_NATIVE_STAGE_TIMING" in text


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
            "NFN_SM120_ACTIVATION": "sd-prelu",
            "NFN_SM120_PROFILE_DIR": "none",
            "NFN_SM120_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_JSON_OUT": str(output_path),
            "NFN_SM120_CANDIDATE_ENV": "NFN_NATIVE_GPT_LM_HEAD_CE_REVERSE_ROWS=0",
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
    assert payload["metric_ratio_gates"]["enabled"] is False
    assert payload["candidate_env"]["NFN_NATIVE_GPT_LM_HEAD_CE_REVERSE_ROWS"] == "0"
    assert payload["baseline_command"][payload["baseline_command"].index("-af") + 1] == "sd-prelu"
    assert (
        payload["candidate_command"][payload["candidate_command"].index("--native-cuda-activation") + 1]
        == "sd-prelu"
    )


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
    assert 'MAX_CANDIDATE_RATIO_RAW+=" train_loop_cuda_event_steady_state_wall_ms_per_step=1.000"' in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.total_ms=1.000"' in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.total_ms=1.000"' in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.mlp_proj.total_ms=1.000"' in text
    assert "MIN_CANDIDATE_RATIO_RAW" in text
    assert "NFN_SM120_NATIVE_MIN_CANDIDATE_RATIO" in text
    assert "NFN_SM120_CANDIDATE_MIN_CANDIDATE_RATIO" in text
    assert "NFN_SM120_NATIVE_TRAIN_LOOP_EVENT_TIMING" in text
    assert "NFN_SM120_CANDIDATE_TRAIN_LOOP_EVENT_TIMING" in text
    assert 'paired_args+=(--baseline-env "NFN_NATIVE_GPT_TRAIN_LOOP_EVENT_TIMING=1")' in text
    assert 'paired_args+=(--candidate-env "NFN_NATIVE_GPT_TRAIN_LOOP_EVENT_TIMING=1")' in text
    assert 'paired_args+=(--min-candidate-ratio "$item")' in text
    assert "*CE_BF16*|*ce_bf16*|*LM_HEAD_CE*|*lm_head_ce*" in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.ce.total_ms=1.000"' in text
    assert "*LM_HEAD_PREPACK_BF16_HIDDEN*|*lm_head_prepack_bf16_hidden*" in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.dhidden.total_ms=1.000"' in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.dweight.total_ms=1.000"' in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" setup.uint16_arena_materialize.total_ms=1.000"' in text
    assert "NFN_SM120_NATIVE_CANDIDATE_PROFILE" in text
    assert "NFN_SM120_CANDIDATE_PROFILE" in text
    assert "lm_head_tk_dinput_32768" in text
    assert "lm_head_cublaslt_dhidden_32768" in text
    assert "lm_head_dhidden_fast16bf_32768" in text
    assert "NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF_SHAPE=768,32768,50304,N,N" in text
    assert "lm_head_tk_dweight_32768" in text
    assert "NFN_NATIVE_LINEAR_TK_DWEIGHT_ENABLE_SHAPE=768,50304,32768,N,T" in text
    assert "bf16_attention_grad_out" in text
    assert "NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1" in text
    assert "0.995826x train_loop_wall_ms_per_step" in text
    assert "attention_bwd_block_32" in text
    assert "-DLLMK_SM120_ATTN_BWD_BLOCK=32" in text
    assert "attention_backward_tk_timing_us regressed to 1.000555x" in text
    assert "bf16_attention_dprep_grad_out" in text
    assert "NFN_NATIVE_GPT_BF16_ATTENTION_DPREP_GRAD_OUT=1" in text
    assert "1.005344x train_loop_wall_ms_per_step" in text
    assert "attention_dprep_float_hd64_specialized" in text
    assert "NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_FLOAT_HD64_SPECIALIZED=1" in text
    assert "lm_head_prepack_bf16_hidden_off" in text
    assert "lm_head_prepack_bf16_hidden_on" in text
    assert "NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=1" in text
    assert "NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=0" in text
    assert "cublas_handle_prewarm" in text
    assert "NFN_NATIVE_GPT_PREWARM_CUBLAS_HANDLE=1" in text
    assert "train_loop_wall_ms_per_step to 1.000699x" in text
    assert "mlp_proj_tk_dweight_65536" in text
    assert "NFN_NATIVE_LINEAR_TK_DWEIGHT_ENABLE_SHAPE=3072,768,65536,N,T" in text
    assert "mlp_proj_split_bgrad_65536" in text
    assert "NFN_NATIVE_LINEAR_BF16_BF16_BGRAD_DISABLE_SHAPE=3072,768,65536,N,T" in text
    assert "layernorm_affine_row_chunk_128" in text
    assert "NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=256" in text
    assert "NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=128" in text
    assert "layernorm_affine_row_chunk_64" in text
    assert "NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=64" in text
    assert "stage.block_backward.mlp_proj.total_ms to 1.004276x" in text
    assert "layernorm_affine_row_chunk_96" in text
    assert "NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=96" in text
    assert "stage.block_backward.mlp_proj.total_ms to 1.000296x" in text
    assert "layernorm_affine_row_chunk_512" in text
    assert "NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=512" in text
    assert "1.019837x train_loop_wall_ms_per_step" in text
    assert "linear_bias_row_chunk_1024" in text
    assert "NFN_NATIVE_GPT_LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE=1024" in text
    assert "1.009736x train_loop_wall_ms_per_step" in text
    assert "lm_head_logits_bf16_fallback_32768" in text
    assert "NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=50304,32768,768,T,N" in text
    assert "lm_head_logits_bf16_fallback_49152" in text
    assert "NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=50304,49152,768,T,N" in text
    assert "1.005968x train_loop_wall_ms_per_step" in text
    assert "lm_head_logits_tk_gemm_count from 32 to 16" in text
    assert "qkv_forward_bf16_fallback_65536" in text
    assert "NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=2304,65536,768,T,N" in text
    assert "regressed the target stage.block_forward.attention.qkv.total_ms to 1.143374x" in text
    assert "mlp_fc_forward_bf16_fallback_65536" in text
    assert "NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=3072,65536,768,N,N" in text
    assert "stage.block_forward.mlp_fc_gelu.total_ms to 1.000722x" in text
    assert "ce_bf16_threads_512" in text
    assert "NFN_NATIVE_GPT_CE_BF16_THREADS=512" in text
    assert "lm_head_ce_vec8_normal_store" in text
    assert "NFN_NATIVE_GPT_CE_BF16_VEC_NORMAL_STORES=1" in text
    assert "stage.lm_head_backward.total_ms to 1.009078x" in text
    assert "lm_head_ce_scalar_streaming_store" in text
    assert "NFN_NATIVE_GPT_CE_BF16_SCALAR_STREAMING_STORES=1" in text
    assert "stage.lm_head_backward.ce.total_ms to 2.054816x" in text
    assert "lm_head_ce_natural_rows" in text
    assert "NFN_NATIVE_GPT_LM_HEAD_CE_REVERSE_ROWS=0" in text
    assert "regressed CUDA-event wall time to 1.019563x" in text
    assert "lm_head_ce_default_specialized" in text
    assert "NFN_NATIVE_GPT_LM_HEAD_CE_DEFAULT_SPECIALIZED=1" in text
    assert "lm_head_ce_no_loss_llmk_style_specialized" in text
    assert "NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_LLMK_STYLE_SPECIALIZED=1" in text
    assert "lm_head_loss_bins" in text
    assert "NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=1" in text
    assert "lm_head_row_loss_sum_accumulate" in text
    assert "NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE=1" in text
    assert "1.000970x train_loop_cuda_event_steady_state_wall_ms_per_step" in text
    assert "1.000304x stage.lm_head_backward.total_ms" in text
    assert "cublaslt_min_waves" in text
    assert "cublaslt_max_waves" in text
    assert "cublaslt_qkv_dweight_h0_65536" in text
    assert "NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,2304,65536,N,T,0" in text
    assert "regressed stage.block_backward.qkv.dweight_bias.total_ms to 1.003363x" in text
    assert "cublaslt_grouped_probe" in text
    assert "NFN_NATIVE_GPT_PROBE_CUBLASLT_GROUPED_LAYOUT=1" in text
    assert "NFN_NATIVE_GPT_PROBE_CUBLASLT_GROUPED_MATMUL=1" in text
    assert "tk_dgelu_dinput" in text
    assert "tk_dgelu_approx_tanh" in text
    assert "NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU=0" in text
    assert "NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU=1" in text
    assert "-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP" in text
    assert "attention_atomic_dq" in text
    assert "tk_forward_no_n96" in text
    assert "-DLLMK_SM120_FORWARD_N96=0" in text
    assert "cuda_device_max_connections_1" in text
    assert "CUDA_DEVICE_MAX_CONNECTIONS=1" in text
    assert "combined_device_arena" in text
    assert "NFN_NATIVE_GPT_COMBINED_DEVICE_ARENA=1" in text
    assert "cuda_malloc_async" in text
    assert "NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC=1" in text
    assert "setup.uint16_arena_materialize.total_ms to 1.775565x" in text
    assert "bgrad_first_write_direct" in text
    assert "NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1" in text
    assert "qkv_concurrent_dinput_dweight" in text
    assert "mlp_fc_concurrent_dinput_dweight" in text
    assert "attn_proj_concurrent_dinput_dweight" in text
    assert "lm_head_concurrent_dhidden_dweight" in text
    assert "lm_head_dweight_before_dhidden" in text
    assert "NFN_NATIVE_GPT_LM_HEAD_DWEIGHT_BEFORE_DHIDDEN=1" in text
    assert "lm_head_pipeline_chunks" in text
    assert "lm_head_overlap_last_dweight" in text
    assert "regressed train_loop_wall_ms_per_step to 1.001676x" in text
    assert "NFN_NATIVE_GPT_LM_HEAD_OVERLAP_LAST_DWEIGHT=1" in text
    assert "lm_head_row_chunk_49152" in text
    assert "--lm-head-row-chunk-size 49152" in text
    assert "lm_head_row_chunk_32768" in text
    assert "--lm-head-row-chunk-size 32768" in text
    assert "1.001939x train_loop_cuda_event_steady_state_wall_ms_per_step" in text
    assert "1.000885x stage.lm_head_backward.total_ms" in text
    assert "lm_head_row_chunk_65536" in text
    assert "NFN_NATIVE_GPT_ALLOW_UNSAFE_LM_HEAD_ROW_CHUNK=1" in text
    assert "--lm-head-row-chunk-size 65536" in text
    assert "lm_head_full_resident_reuse" in text
    assert "NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS=1" in text
    assert "NFN_NATIVE_GPT_FULL_BATCH_LM_HEAD_REUSE=1" in text
    assert "lm_head_cooperative_loss_bins" in text
    assert "STRICT_PROBE_CANDIDATE_PROFILE" in text
    assert "strict ABI preflight probe, not a speed candidate" in text
    assert "AUTO_DISABLE_METRIC_RATIO_GATES=1" in text
    assert "FORCE_DISABLE_ROUTE_CHANGE=1" in text
    assert "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_LOSS_BINS=1" in text
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
    assert "NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1" in text
    assert "NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,32768,50304,N,N,0" in text
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
    assert "qkv_dinput_before_dweight" in text
    assert "NFN_NATIVE_GPT_QKV_DINPUT_BEFORE_DWEIGHT=1" in text
    assert "target stage.block_backward.qkv.total_ms regressed to 1.001003x" in text
    assert "*QKV_DINPUT_BEFORE_DWEIGHT*|*qkv_dinput_before_dweight*" in text
    assert "*BLOCK_QKV_CONCURRENT_DINPUT_DWEIGHT*|*block_qkv_concurrent_dinput_dweight*" in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.qkv.total_ms=1.000"' in text
    assert "*BLOCK_MLP_FC_CONCURRENT_DINPUT_DWEIGHT*|*block_mlp_fc_concurrent_dinput_dweight*" in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.mlp_fc.total_ms=1.000"' in text
    assert "*BLOCK_ATTN_PROJ_CONCURRENT_DINPUT_DWEIGHT*|*block_attn_proj_concurrent_dinput_dweight*|*attn_proj_concurrent_dinput_dweight*" in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.block_backward.attn_proj.total_ms=1.000"' in text
    assert 'MAX_CANDIDATE_RATIO_RAW+=" stage.lm_head_backward.dhidden_dweight_concurrent.total_ms=1.000"' not in text
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

    fast16bf_output_path = tmp_path / "candidate-fast16bf-dhidden-dry-run.json"
    fast16bf_env = os.environ.copy()
    fast16bf_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_dhidden_fast16bf_32768",
            "NFN_SM120_NATIVE_JSON_OUT": str(fast16bf_output_path),
        }
    )

    fast16bf_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=fast16bf_env,
    )

    assert fast16bf_dry_run.returncode == 0, fast16bf_dry_run.stderr
    fast16bf_payload = json.loads(fast16bf_output_path.read_text(encoding="utf-8"))
    assert (
        fast16bf_payload["candidate_env"][
            "NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF_SHAPE"
        ]
        == "768,32768,50304,N,N"
    )
    assert fast16bf_payload["metric_ratio_gates"]["enabled"] is False

    tk_dweight_output_path = tmp_path / "candidate-tk-dweight-dry-run.json"
    tk_dweight_env = os.environ.copy()
    tk_dweight_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_tk_dweight_32768",
            "NFN_SM120_NATIVE_JSON_OUT": str(tk_dweight_output_path),
        }
    )

    tk_dweight_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=tk_dweight_env,
    )

    assert tk_dweight_dry_run.returncode == 0, tk_dweight_dry_run.stderr
    tk_dweight_payload = json.loads(tk_dweight_output_path.read_text(encoding="utf-8"))
    assert (
        tk_dweight_payload["candidate_env"][
            "NFN_NATIVE_LINEAR_TK_DWEIGHT_ENABLE_SHAPE"
        ]
        == "768,50304,32768,N,T"
    )
    assert tk_dweight_payload["metric_ratio_gates"]["enabled"] is False

    tk_dgelu_output_path = tmp_path / "candidate-tk-dgelu-dinput-dry-run.json"
    tk_dgelu_env = os.environ.copy()
    tk_dgelu_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "tk_dgelu_dinput",
            "NFN_SM120_NATIVE_JSON_OUT": str(tk_dgelu_output_path),
        }
    )

    tk_dgelu_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=tk_dgelu_env,
    )

    assert tk_dgelu_dry_run.returncode == 0, tk_dgelu_dry_run.stderr
    tk_dgelu_payload = json.loads(tk_dgelu_output_path.read_text(encoding="utf-8"))
    assert tk_dgelu_payload["baseline_env"]["NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU"] == "0"
    assert tk_dgelu_payload["candidate_env"]["NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU"] == "1"
    assert any(
        str(part).startswith("/tmp/nfn_sm120_candidate_tile_ops_tk_dgelu_dinput_")
        for part in tk_dgelu_payload["candidate_command"]
    )
    assert tk_dgelu_payload["metric_ratio_gates"]["enabled"] is False

    mlp_proj_tk_dweight_output_path = tmp_path / "candidate-mlp-proj-tk-dweight-dry-run.json"
    mlp_proj_tk_dweight_env = os.environ.copy()
    mlp_proj_tk_dweight_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "mlp_proj_tk_dweight_65536",
            "NFN_SM120_NATIVE_JSON_OUT": str(mlp_proj_tk_dweight_output_path),
        }
    )

    mlp_proj_tk_dweight_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=mlp_proj_tk_dweight_env,
    )

    assert mlp_proj_tk_dweight_dry_run.returncode == 0, mlp_proj_tk_dweight_dry_run.stderr
    mlp_proj_tk_dweight_payload = json.loads(
        mlp_proj_tk_dweight_output_path.read_text(encoding="utf-8")
    )
    assert (
        mlp_proj_tk_dweight_payload["candidate_env"][
            "NFN_NATIVE_LINEAR_TK_DWEIGHT_ENABLE_SHAPE"
        ]
        == "3072,768,65536,N,T"
    )
    assert mlp_proj_tk_dweight_payload["metric_ratio_gates"]["enabled"] is False

    ln_chunk_output_path = tmp_path / "candidate-ln-row-chunk-dry-run.json"
    ln_chunk_env = os.environ.copy()
    ln_chunk_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "layernorm_affine_row_chunk_512",
            "NFN_SM120_NATIVE_JSON_OUT": str(ln_chunk_output_path),
        }
    )

    ln_chunk_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=ln_chunk_env,
    )

    assert ln_chunk_dry_run.returncode == 0, ln_chunk_dry_run.stderr
    ln_chunk_payload = json.loads(ln_chunk_output_path.read_text(encoding="utf-8"))
    assert (
        ln_chunk_payload["candidate_env"][
            "NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE"
        ]
        == "512"
    )
    assert ln_chunk_payload["metric_ratio_gates"]["enabled"] is False

    rejected_ln_chunk_512_env = os.environ.copy()
    rejected_ln_chunk_512_env.update(
        {
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "layernorm_affine_row_chunk_512",
            "NFN_SM120_NATIVE_JSON_OUT": str(tmp_path / "rejected-ln-row-chunk-512.json"),
        }
    )

    rejected_ln_chunk_512_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=rejected_ln_chunk_512_env,
    )

    assert rejected_ln_chunk_512_run.returncode == 2
    assert "layernorm_affine_row_chunk_512 is a rejected SM120 candidate" in (
        rejected_ln_chunk_512_run.stderr
    )
    assert "NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1" in (
        rejected_ln_chunk_512_run.stderr
    )

    rejected_linear_bias_1024_env = os.environ.copy()
    rejected_linear_bias_1024_env.update(
        {
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "linear_bias_row_chunk_1024",
            "NFN_SM120_NATIVE_JSON_OUT": str(
                tmp_path / "rejected-linear-bias-row-chunk-1024.json"
            ),
        }
    )

    rejected_linear_bias_1024_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=rejected_linear_bias_1024_env,
    )

    assert rejected_linear_bias_1024_run.returncode == 2
    assert "linear_bias_row_chunk_1024 is a rejected SM120 candidate" in (
        rejected_linear_bias_1024_run.stderr
    )
    assert "NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1" in (
        rejected_linear_bias_1024_run.stderr
    )

    rejected_attention_profiles = (
        "bf16_attention_grad_out",
        "bf16_attention_dprep_grad_out",
        "attention_dprep_float_hd64_specialized",
    )
    for rejected_attention_profile in rejected_attention_profiles:
        rejected_attention_env = os.environ.copy()
        rejected_attention_env.update(
            {
                "NFN_SM120_NATIVE_PROFILE_DIR": "none",
                "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
                "NFN_SM120_NATIVE_CANDIDATE_PROFILE": rejected_attention_profile,
                "NFN_SM120_NATIVE_JSON_OUT": str(
                    tmp_path / f"rejected-{rejected_attention_profile}.json"
                ),
            }
        )

        rejected_attention_run = subprocess.run(
            ["bash", str(script)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=rejected_attention_env,
        )

        assert rejected_attention_run.returncode == 2
        assert (
            f"{rejected_attention_profile} is a rejected SM120 candidate"
            in rejected_attention_run.stderr
        )
        assert "NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1" in (
            rejected_attention_run.stderr
        )

    ln_chunk_64_output_path = tmp_path / "candidate-ln-row-chunk-64-dry-run.json"
    ln_chunk_64_env = os.environ.copy()
    ln_chunk_64_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "layernorm_affine_row_chunk_64",
            "NFN_SM120_NATIVE_JSON_OUT": str(ln_chunk_64_output_path),
        }
    )

    ln_chunk_64_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=ln_chunk_64_env,
    )

    assert ln_chunk_64_dry_run.returncode == 0, ln_chunk_64_dry_run.stderr
    ln_chunk_64_payload = json.loads(
        ln_chunk_64_output_path.read_text(encoding="utf-8")
    )
    assert (
        ln_chunk_64_payload["candidate_env"][
            "NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE"
        ]
        == "64"
    )
    assert ln_chunk_64_payload["metric_ratio_gates"]["enabled"] is False

    rejected_ln_chunk_64_env = os.environ.copy()
    rejected_ln_chunk_64_env.update(
        {
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "layernorm_affine_row_chunk_64",
            "NFN_SM120_NATIVE_JSON_OUT": str(tmp_path / "rejected-ln-row-chunk-64.json"),
        }
    )

    rejected_ln_chunk_64_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=rejected_ln_chunk_64_env,
    )

    assert rejected_ln_chunk_64_run.returncode == 2
    assert "layernorm_affine_row_chunk_64 is a rejected SM120 candidate" in (
        rejected_ln_chunk_64_run.stderr
    )
    assert "NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1" in (
        rejected_ln_chunk_64_run.stderr
    )

    ln_chunk_96_output_path = tmp_path / "candidate-ln-row-chunk-96-dry-run.json"
    ln_chunk_96_env = os.environ.copy()
    ln_chunk_96_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "layernorm_affine_row_chunk_96",
            "NFN_SM120_NATIVE_JSON_OUT": str(ln_chunk_96_output_path),
        }
    )

    ln_chunk_96_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=ln_chunk_96_env,
    )

    assert ln_chunk_96_dry_run.returncode == 0, ln_chunk_96_dry_run.stderr
    ln_chunk_96_payload = json.loads(
        ln_chunk_96_output_path.read_text(encoding="utf-8")
    )
    assert (
        ln_chunk_96_payload["candidate_env"][
            "NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE"
        ]
        == "96"
    )
    assert ln_chunk_96_payload["metric_ratio_gates"]["enabled"] is False

    rejected_ln_chunk_96_env = os.environ.copy()
    rejected_ln_chunk_96_env.update(
        {
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "layernorm_affine_row_chunk_96",
            "NFN_SM120_NATIVE_JSON_OUT": str(tmp_path / "rejected-ln-row-chunk-96.json"),
        }
    )

    rejected_ln_chunk_96_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=rejected_ln_chunk_96_env,
    )

    assert rejected_ln_chunk_96_run.returncode == 2
    assert "layernorm_affine_row_chunk_96 is a rejected SM120 candidate" in (
        rejected_ln_chunk_96_run.stderr
    )
    assert "NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1" in (
        rejected_ln_chunk_96_run.stderr
    )

    logits_fallback_output_path = tmp_path / "candidate-logits-fallback-dry-run.json"
    logits_fallback_env = os.environ.copy()
    logits_fallback_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_logits_bf16_fallback_32768",
            "NFN_SM120_NATIVE_JSON_OUT": str(logits_fallback_output_path),
        }
    )

    logits_fallback_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=logits_fallback_env,
    )

    assert logits_fallback_dry_run.returncode == 0, logits_fallback_dry_run.stderr
    logits_fallback_payload = json.loads(
        logits_fallback_output_path.read_text(encoding="utf-8")
    )
    assert (
        logits_fallback_payload["candidate_env"][
            "NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE"
        ]
        == "50304,32768,768,T,N"
    )
    assert logits_fallback_payload["metric_ratio_gates"]["enabled"] is False

    logits_fallback_49152_output_path = (
        tmp_path / "candidate-logits-fallback-49152-dry-run.json"
    )
    logits_fallback_49152_env = os.environ.copy()
    logits_fallback_49152_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_logits_bf16_fallback_49152",
            "NFN_SM120_NATIVE_JSON_OUT": str(logits_fallback_49152_output_path),
        }
    )

    logits_fallback_49152_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=logits_fallback_49152_env,
    )

    assert logits_fallback_49152_dry_run.returncode == 0, (
        logits_fallback_49152_dry_run.stderr
    )
    logits_fallback_49152_payload = json.loads(
        logits_fallback_49152_output_path.read_text(encoding="utf-8")
    )
    assert (
        logits_fallback_49152_payload["candidate_env"][
            "NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE"
        ]
        == "50304,49152,768,T,N"
    )
    assert logits_fallback_49152_payload["metric_ratio_gates"]["enabled"] is False

    rejected_logits_fallback_49152_env = os.environ.copy()
    rejected_logits_fallback_49152_env.update(
        {
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_logits_bf16_fallback_49152",
            "NFN_SM120_NATIVE_JSON_OUT": str(
                tmp_path / "rejected-logits-fallback-49152.json"
            ),
        }
    )

    rejected_logits_fallback_49152_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=rejected_logits_fallback_49152_env,
    )

    assert rejected_logits_fallback_49152_run.returncode == 2
    assert "lm_head_logits_bf16_fallback_49152 is a rejected SM120 candidate" in (
        rejected_logits_fallback_49152_run.stderr
    )
    assert "NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1" in (
        rejected_logits_fallback_49152_run.stderr
    )

    qkv_forward_fallback_output_path = tmp_path / "candidate-qkv-forward-fallback-dry-run.json"
    qkv_forward_fallback_env = os.environ.copy()
    qkv_forward_fallback_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "qkv_forward_bf16_fallback_65536",
            "NFN_SM120_NATIVE_JSON_OUT": str(qkv_forward_fallback_output_path),
        }
    )

    qkv_forward_fallback_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=qkv_forward_fallback_env,
    )

    assert qkv_forward_fallback_dry_run.returncode == 0, qkv_forward_fallback_dry_run.stderr
    qkv_forward_fallback_payload = json.loads(
        qkv_forward_fallback_output_path.read_text(encoding="utf-8")
    )
    assert (
        qkv_forward_fallback_payload["candidate_env"][
            "NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE"
        ]
        == "2304,65536,768,T,N"
    )
    assert qkv_forward_fallback_payload["metric_ratio_gates"]["enabled"] is False

    rejected_qkv_env = os.environ.copy()
    rejected_qkv_env.update(
        {
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "qkv_forward_bf16_fallback_65536",
            "NFN_SM120_NATIVE_JSON_OUT": str(tmp_path / "rejected-qkv.json"),
        }
    )

    rejected_qkv_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=rejected_qkv_env,
    )

    assert rejected_qkv_run.returncode == 2
    assert "rejected SM120 candidate" in rejected_qkv_run.stderr
    assert "NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1" in rejected_qkv_run.stderr

    mlp_fc_forward_fallback_output_path = (
        tmp_path / "candidate-mlp-fc-forward-fallback-dry-run.json"
    )
    mlp_fc_forward_fallback_env = os.environ.copy()
    mlp_fc_forward_fallback_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "mlp_fc_forward_bf16_fallback_65536",
            "NFN_SM120_NATIVE_JSON_OUT": str(mlp_fc_forward_fallback_output_path),
        }
    )

    mlp_fc_forward_fallback_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=mlp_fc_forward_fallback_env,
    )

    assert mlp_fc_forward_fallback_dry_run.returncode == 0, (
        mlp_fc_forward_fallback_dry_run.stderr
    )
    mlp_fc_forward_fallback_payload = json.loads(
        mlp_fc_forward_fallback_output_path.read_text(encoding="utf-8")
    )
    assert (
        mlp_fc_forward_fallback_payload["candidate_env"][
            "NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE"
        ]
        == "3072,65536,768,N,N"
    )
    assert mlp_fc_forward_fallback_payload["metric_ratio_gates"]["enabled"] is False

    rejected_mlp_fc_env = os.environ.copy()
    rejected_mlp_fc_env.update(
        {
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "mlp_fc_forward_bf16_fallback_65536",
            "NFN_SM120_NATIVE_JSON_OUT": str(tmp_path / "rejected-mlp-fc.json"),
        }
    )

    rejected_mlp_fc_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=rejected_mlp_fc_env,
    )

    assert rejected_mlp_fc_run.returncode == 2
    assert "mlp_fc_forward_bf16_fallback_65536 is a rejected SM120 candidate" in (
        rejected_mlp_fc_run.stderr
    )
    assert "NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1" in (
        rejected_mlp_fc_run.stderr
    )

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

    grouped_probe_output_path = tmp_path / "candidate-cublaslt-grouped-probe-dry-run.json"
    grouped_probe_env = os.environ.copy()
    grouped_probe_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "cublaslt_grouped_probe",
            "NFN_SM120_NATIVE_JSON_OUT": str(grouped_probe_output_path),
        }
    )

    grouped_probe_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=grouped_probe_env,
    )

    assert grouped_probe_dry_run.returncode == 0, grouped_probe_dry_run.stderr
    grouped_probe_payload = json.loads(
        grouped_probe_output_path.read_text(encoding="utf-8")
    )
    assert (
        grouped_probe_payload["candidate_env"][
            "NFN_NATIVE_GPT_PROBE_CUBLASLT_GROUPED_LAYOUT"
        ]
        == "1"
    )
    assert (
        grouped_probe_payload["candidate_env"][
            "NFN_NATIVE_GPT_PROBE_CUBLASLT_GROUPED_MATMUL"
        ]
        == "1"
    )
    assert grouped_probe_payload["metric_ratio_gates"]["enabled"] is False
    assert "AUTO_DISABLE_METRIC_RATIO_GATES=1" in text
    assert "NFN_SM120_NATIVE_AUTO_DISABLE_METRIC_RATIO_GATES" in text
    assert "DISABLE_METRIC_RATIO_GATES" in text
    assert "NFN_SM120_NATIVE_DISABLE_METRIC_RATIO_GATES" in text

    max_connections_output_path = tmp_path / "candidate-max-connections-dry-run.json"
    max_connections_env = os.environ.copy()
    max_connections_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "cuda_device_max_connections_1",
            "NFN_SM120_NATIVE_JSON_OUT": str(max_connections_output_path),
        }
    )

    max_connections_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=max_connections_env,
    )

    assert max_connections_dry_run.returncode == 0, max_connections_dry_run.stderr
    max_connections_payload = json.loads(
        max_connections_output_path.read_text(encoding="utf-8")
    )
    assert (
        max_connections_payload["candidate_env"]["CUDA_DEVICE_MAX_CONNECTIONS"]
        == "1"
    )
    assert max_connections_payload["metric_ratio_gates"]["enabled"] is False

    ce_threads_output_path = tmp_path / "candidate-ce-threads-dry-run.json"
    ce_threads_env = os.environ.copy()
    ce_threads_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_STAGE_TIMING": "1",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "ce_bf16_threads_512",
            "NFN_SM120_NATIVE_JSON_OUT": str(ce_threads_output_path),
        }
    )

    ce_threads_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=ce_threads_env,
    )

    assert ce_threads_dry_run.returncode == 0, ce_threads_dry_run.stderr
    ce_threads_payload = json.loads(
        ce_threads_output_path.read_text(encoding="utf-8")
    )
    assert (
        ce_threads_payload["candidate_env"]["NFN_NATIVE_GPT_CE_BF16_THREADS"]
        == "512"
    )
    assert ce_threads_payload["metric_ratio_gates"]["enabled"] is False
    assert "1.430612x" in text
    assert "NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1" in text

    ce_threads_rejected_env = os.environ.copy()
    ce_threads_rejected_env.update(
        {
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "ce_bf16_threads_512",
            "NFN_SM120_NATIVE_JSON_OUT": str(tmp_path / "candidate-ce-threads-rejected.json"),
        }
    )

    ce_threads_rejected = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=ce_threads_rejected_env,
    )

    assert ce_threads_rejected.returncode == 2
    assert "ce_bf16_threads_512 is a rejected SM120 candidate" in ce_threads_rejected.stderr
    assert "stage.lm_head_backward.ce.total_ms to 1.430612x" in ce_threads_rejected.stderr

    ce_vec8_output_path = tmp_path / "candidate-ce-vec8-dry-run.json"
    ce_vec8_env = os.environ.copy()
    ce_vec8_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_STAGE_TIMING": "1",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_ce_vec8_io",
            "NFN_SM120_NATIVE_JSON_OUT": str(ce_vec8_output_path),
        }
    )

    ce_vec8_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=ce_vec8_env,
    )

    assert ce_vec8_dry_run.returncode == 0, ce_vec8_dry_run.stderr
    ce_vec8_payload = json.loads(ce_vec8_output_path.read_text(encoding="utf-8"))
    assert (
        ce_vec8_payload["candidate_env"]["NFN_NATIVE_GPT_CE_BF16_VEC_LOADS"]
        == "1"
    )
    assert (
        ce_vec8_payload["candidate_env"]["NFN_NATIVE_GPT_CE_BF16_VEC_STORES"]
        == "1"
    )
    assert ce_vec8_payload["metric_ratio_gates"]["enabled"] is False

    ce_vec8_rejected_env = os.environ.copy()
    ce_vec8_rejected_env.update(
        {
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_ce_vec8_io",
            "NFN_SM120_NATIVE_JSON_OUT": str(tmp_path / "candidate-ce-vec8-rejected.json"),
        }
    )

    ce_vec8_rejected = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=ce_vec8_rejected_env,
    )

    assert ce_vec8_rejected.returncode == 2
    assert "lm_head_ce_vec8_io is a rejected SM120 candidate" in ce_vec8_rejected.stderr
    assert "stage.lm_head_backward.ce.total_ms=1.003780x" in ce_vec8_rejected.stderr
    assert "NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1" in ce_vec8_rejected.stderr

    classifier_ce_no_loss_output_path = (
        tmp_path / "candidate-classifier-ce-no-loss-dry-run.json"
    )
    classifier_ce_no_loss_env = os.environ.copy()
    classifier_ce_no_loss_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_STAGE_TIMING": "1",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_classifier_ce_no_loss",
            "NFN_SM120_NATIVE_JSON_OUT": str(classifier_ce_no_loss_output_path),
        }
    )

    classifier_ce_no_loss_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=classifier_ce_no_loss_env,
    )

    assert classifier_ce_no_loss_dry_run.returncode == 0, (
        classifier_ce_no_loss_dry_run.stderr
    )
    classifier_ce_no_loss_payload = json.loads(
        classifier_ce_no_loss_output_path.read_text(encoding="utf-8")
    )
    assert (
        classifier_ce_no_loss_payload["baseline_env"][
            "NFN_NATIVE_GPT_LM_HEAD_CLASSIFIER_CE_NO_LOSS"
        ]
        == "0"
    )
    assert (
        classifier_ce_no_loss_payload["candidate_env"][
            "NFN_NATIVE_GPT_LM_HEAD_CLASSIFIER_CE_NO_LOSS"
        ]
        == "1"
    )
    assert classifier_ce_no_loss_payload["metric_ratio_gates"]["enabled"] is False

    classifier_ce_no_loss_rejected_env = os.environ.copy()
    classifier_ce_no_loss_rejected_env.update(
        {
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_classifier_ce_no_loss",
            "NFN_SM120_NATIVE_JSON_OUT": str(
                tmp_path / "candidate-classifier-ce-no-loss-rejected.json"
            ),
        }
    )

    classifier_ce_no_loss_rejected = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=classifier_ce_no_loss_rejected_env,
    )

    assert classifier_ce_no_loss_rejected.returncode == 2
    assert "lm_head_classifier_ce_no_loss is a rejected SM120 candidate" in (
        classifier_ce_no_loss_rejected.stderr
    )
    assert "stage.lm_head_backward.ce.total_ms to 1.848303x" in (
        classifier_ce_no_loss_rejected.stderr
    )

    prepack_on_dry_run_path = tmp_path / "candidate-prepack-on-dry-run.json"
    prepack_on_dry_run_env = os.environ.copy()
    prepack_on_dry_run_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_prepack_bf16_hidden_on",
            "NFN_SM120_NATIVE_JSON_OUT": str(prepack_on_dry_run_path),
        }
    )

    prepack_on_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=prepack_on_dry_run_env,
    )

    assert prepack_on_dry_run.returncode == 0, prepack_on_dry_run.stderr
    prepack_on_payload = json.loads(prepack_on_dry_run_path.read_text(encoding="utf-8"))
    assert prepack_on_payload["baseline_env"]["NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN"] == "0"
    assert prepack_on_payload["candidate_env"]["NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN"] == "1"

    cublas_handle_rejected_env = os.environ.copy()
    cublas_handle_rejected_env.update(
        {
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "cublas_handle_prewarm",
            "NFN_SM120_NATIVE_JSON_OUT": str(tmp_path / "candidate-cublas-handle-rejected.json"),
        }
    )

    cublas_handle_rejected = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=cublas_handle_rejected_env,
    )

    assert cublas_handle_rejected.returncode == 2
    assert (
        "cublas_handle_prewarm is a rejected SM120 candidate"
        in cublas_handle_rejected.stderr
    )
    assert "stage.lm_head_backward.total_ms to 1.000673x" in cublas_handle_rejected.stderr

    ce_specialized_output_path = tmp_path / "candidate-ce-specialized-dry-run.json"
    ce_specialized_env = os.environ.copy()
    ce_specialized_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_STAGE_TIMING": "1",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_ce_default_specialized",
            "NFN_SM120_NATIVE_JSON_OUT": str(ce_specialized_output_path),
        }
    )

    ce_specialized_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=ce_specialized_env,
    )

    assert ce_specialized_dry_run.returncode == 0, ce_specialized_dry_run.stderr
    ce_specialized_payload = json.loads(
        ce_specialized_output_path.read_text(encoding="utf-8")
    )
    assert (
        ce_specialized_payload["candidate_env"][
            "NFN_NATIVE_GPT_LM_HEAD_CE_DEFAULT_SPECIALIZED"
        ]
        == "1"
    )
    assert ce_specialized_payload["metric_ratio_gates"]["enabled"] is False
    ce_specialized_rejected_env = os.environ.copy()
    ce_specialized_rejected_env.update(
        {
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_ce_default_specialized",
            "NFN_SM120_NATIVE_JSON_OUT": str(tmp_path / "candidate-ce-specialized-rejected.json"),
        }
    )
    ce_specialized_rejected = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=ce_specialized_rejected_env,
    )
    assert ce_specialized_rejected.returncode == 2
    assert "lm_head_ce_default_specialized is a rejected SM120 candidate" in (
        ce_specialized_rejected.stderr
    )
    assert "1.001545x train_loop_wall_ms_per_step" in ce_specialized_rejected.stderr

    ce_no_loss_llmk_output_path = tmp_path / "candidate-ce-no-loss-llmk-dry-run.json"
    ce_no_loss_llmk_env = os.environ.copy()
    ce_no_loss_llmk_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_STAGE_TIMING": "1",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_ce_no_loss_llmk_style_specialized",
            "NFN_SM120_NATIVE_JSON_OUT": str(ce_no_loss_llmk_output_path),
        }
    )

    ce_no_loss_llmk_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=ce_no_loss_llmk_env,
    )

    assert ce_no_loss_llmk_dry_run.returncode == 0, ce_no_loss_llmk_dry_run.stderr
    ce_no_loss_llmk_payload = json.loads(
        ce_no_loss_llmk_output_path.read_text(encoding="utf-8")
    )
    assert (
        ce_no_loss_llmk_payload["candidate_env"][
            "NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_LLMK_STYLE_SPECIALIZED"
        ]
        == "1"
    )
    assert "--train-loss-every-steps" in ce_no_loss_llmk_payload["candidate_command"]
    assert "0" in ce_no_loss_llmk_payload["candidate_command"]
    assert ce_no_loss_llmk_payload["metric_ratio_gates"]["enabled"] is False

    loss_bins_output_path = tmp_path / "candidate-loss-bins-dry-run.json"
    loss_bins_env = os.environ.copy()
    loss_bins_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_STAGE_TIMING": "1",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_loss_bins",
            "NFN_SM120_NATIVE_JSON_OUT": str(loss_bins_output_path),
        }
    )

    loss_bins_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=loss_bins_env,
    )

    assert loss_bins_dry_run.returncode == 0, loss_bins_dry_run.stderr
    loss_bins_payload = json.loads(
        loss_bins_output_path.read_text(encoding="utf-8")
    )
    assert (
        loss_bins_payload["candidate_env"][
            "NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION"
        ]
        == "1"
    )
    assert loss_bins_payload["metric_ratio_gates"]["enabled"] is False
    loss_bins_rejected_env = os.environ.copy()
    loss_bins_rejected_env.update(
        {
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_loss_bins",
            "NFN_SM120_NATIVE_JSON_OUT": str(tmp_path / "candidate-loss-bins-rejected.json"),
        }
    )
    loss_bins_rejected = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=loss_bins_rejected_env,
    )
    assert loss_bins_rejected.returncode == 2
    assert "lm_head_loss_bins is a rejected SM120 candidate" in loss_bins_rejected.stderr
    assert "stage.block_backward.total_ms regressed to 1.019348x" in (
        loss_bins_rejected.stderr
    )

    row_loss_sum_output_path = tmp_path / "candidate-row-loss-sum-dry-run.json"
    row_loss_sum_env = os.environ.copy()
    row_loss_sum_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_STAGE_TIMING": "1",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_row_loss_sum_accumulate",
            "NFN_SM120_NATIVE_JSON_OUT": str(row_loss_sum_output_path),
        }
    )

    row_loss_sum_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=row_loss_sum_env,
    )

    assert row_loss_sum_dry_run.returncode == 0, row_loss_sum_dry_run.stderr
    row_loss_sum_payload = json.loads(
        row_loss_sum_output_path.read_text(encoding="utf-8")
    )
    assert (
        row_loss_sum_payload["baseline_env"][
            "NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE"
        ]
        == "0"
    )
    assert (
        row_loss_sum_payload["candidate_env"][
            "NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE"
        ]
        == "1"
    )
    assert row_loss_sum_payload["metric_ratio_gates"]["enabled"] is False

    row_loss_sum_rejected_env = os.environ.copy()
    row_loss_sum_rejected_env.update(
        {
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_row_loss_sum_accumulate",
            "NFN_SM120_NATIVE_JSON_OUT": str(
                tmp_path / "candidate-row-loss-sum-rejected.json"
            ),
        }
    )
    row_loss_sum_rejected = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=row_loss_sum_rejected_env,
    )
    assert row_loss_sum_rejected.returncode == 2
    assert "lm_head_row_loss_sum_accumulate is a rejected SM120 candidate" in (
        row_loss_sum_rejected.stderr
    )
    assert "1.000304x stage.lm_head_backward.total_ms" in (
        row_loss_sum_rejected.stderr
    )

    row_chunk_32768_output_path = tmp_path / "candidate-row-chunk-32768-dry-run.json"
    row_chunk_32768_env = os.environ.copy()
    row_chunk_32768_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_STAGE_TIMING": "1",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_row_chunk_32768",
            "NFN_SM120_NATIVE_JSON_OUT": str(row_chunk_32768_output_path),
        }
    )

    row_chunk_32768_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=row_chunk_32768_env,
    )

    assert row_chunk_32768_dry_run.returncode == 0, row_chunk_32768_dry_run.stderr
    row_chunk_32768_payload = json.loads(
        row_chunk_32768_output_path.read_text(encoding="utf-8")
    )
    assert "--lm-head-row-chunk-size" in row_chunk_32768_payload["baseline_command"]
    assert "49152" in row_chunk_32768_payload["baseline_command"]
    assert "--lm-head-row-chunk-size" in row_chunk_32768_payload["candidate_command"]
    assert "32768" in row_chunk_32768_payload["candidate_command"]
    assert row_chunk_32768_payload["metric_ratio_gates"]["enabled"] is False

    row_chunk_32768_rejected_env = os.environ.copy()
    row_chunk_32768_rejected_env.update(
        {
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_row_chunk_32768",
            "NFN_SM120_NATIVE_JSON_OUT": str(
                tmp_path / "candidate-row-chunk-32768-rejected.json"
            ),
        }
    )
    row_chunk_32768_rejected = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=row_chunk_32768_rejected_env,
    )
    assert row_chunk_32768_rejected.returncode == 2
    assert "lm_head_row_chunk_32768 is a rejected SM120 candidate" in (
        row_chunk_32768_rejected.stderr
    )
    assert "1.000885x stage.lm_head_backward.total_ms" in (
        row_chunk_32768_rejected.stderr
    )

    combined_arena_output_path = tmp_path / "candidate-combined-arena-dry-run.json"
    combined_arena_env = os.environ.copy()
    combined_arena_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "combined_device_arena",
            "NFN_SM120_NATIVE_JSON_OUT": str(combined_arena_output_path),
        }
    )

    combined_arena_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=combined_arena_env,
    )

    assert combined_arena_dry_run.returncode == 0, combined_arena_dry_run.stderr
    combined_arena_payload = json.loads(
        combined_arena_output_path.read_text(encoding="utf-8")
    )
    assert (
        combined_arena_payload["baseline_env"][
            "NFN_NATIVE_GPT_COMBINED_DEVICE_ARENA"
        ]
        == "0"
    )
    assert (
        combined_arena_payload["candidate_env"][
            "NFN_NATIVE_GPT_COMBINED_DEVICE_ARENA"
        ]
        == "1"
    )
    assert combined_arena_payload["metric_ratio_gates"]["enabled"] is False

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

    rejected_qkv_concurrent_env = os.environ.copy()
    rejected_qkv_concurrent_env.update(
        {
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "qkv_concurrent_dinput_dweight",
            "NFN_SM120_NATIVE_JSON_OUT": str(tmp_path / "rejected-qkv-concurrent.json"),
        }
    )

    rejected_qkv_concurrent_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=rejected_qkv_concurrent_env,
    )

    assert rejected_qkv_concurrent_run.returncode == 2
    assert "rejected SM120 candidate" in rejected_qkv_concurrent_run.stderr
    assert (
        "NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1"
        in rejected_qkv_concurrent_run.stderr
    )

    attn_proj_output_path = tmp_path / "candidate-attn-proj-concurrent-dry-run.json"
    attn_proj_env = os.environ.copy()
    attn_proj_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "attn_proj_concurrent_dinput_dweight",
            "NFN_SM120_NATIVE_JSON_OUT": str(attn_proj_output_path),
        }
    )

    attn_proj_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=attn_proj_env,
    )

    assert attn_proj_dry_run.returncode == 0, attn_proj_dry_run.stderr
    attn_proj_payload = json.loads(attn_proj_output_path.read_text(encoding="utf-8"))
    assert (
        attn_proj_payload["candidate_env"][
            "NFN_NATIVE_GPT_BLOCK_ATTN_PROJ_CONCURRENT_DINPUT_DWEIGHT"
        ]
        == "1"
    )
    assert attn_proj_payload["metric_ratio_gates"]["enabled"] is False

    lm_head_concurrent_output_path = tmp_path / "candidate-lm-head-concurrent-dry-run.json"
    lm_head_concurrent_env = os.environ.copy()
    lm_head_concurrent_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_concurrent_dhidden_dweight",
            "NFN_SM120_NATIVE_JSON_OUT": str(lm_head_concurrent_output_path),
        }
    )

    lm_head_concurrent_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=lm_head_concurrent_env,
    )

    assert lm_head_concurrent_dry_run.returncode == 0, lm_head_concurrent_dry_run.stderr
    lm_head_concurrent_payload = json.loads(
        lm_head_concurrent_output_path.read_text(encoding="utf-8")
    )
    assert (
        lm_head_concurrent_payload["candidate_env"][
            "NFN_NATIVE_GPT_LM_HEAD_CONCURRENT_DHIDDEN_DWEIGHT"
        ]
        == "1"
    )
    assert lm_head_concurrent_payload["metric_ratio_gates"]["enabled"] is False

    lm_head_dweight_first_output_path = tmp_path / "candidate-lm-head-dweight-before-dhidden-dry-run.json"
    lm_head_dweight_first_env = os.environ.copy()
    lm_head_dweight_first_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_dweight_before_dhidden",
            "NFN_SM120_NATIVE_JSON_OUT": str(lm_head_dweight_first_output_path),
        }
    )

    lm_head_dweight_first_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=lm_head_dweight_first_env,
    )

    assert lm_head_dweight_first_dry_run.returncode == 0, lm_head_dweight_first_dry_run.stderr
    lm_head_dweight_first_payload = json.loads(
        lm_head_dweight_first_output_path.read_text(encoding="utf-8")
    )
    assert (
        lm_head_dweight_first_payload["candidate_env"][
            "NFN_NATIVE_GPT_LM_HEAD_DWEIGHT_BEFORE_DHIDDEN"
        ]
        == "1"
    )
    assert lm_head_dweight_first_payload["metric_ratio_gates"]["enabled"] is False

    lm_head_overlap_output_path = tmp_path / "candidate-lm-head-overlap-last-dweight-dry-run.json"
    lm_head_overlap_env = os.environ.copy()
    lm_head_overlap_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_overlap_last_dweight",
            "NFN_SM120_NATIVE_JSON_OUT": str(lm_head_overlap_output_path),
        }
    )

    lm_head_overlap_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=lm_head_overlap_env,
    )

    assert lm_head_overlap_dry_run.returncode == 0, lm_head_overlap_dry_run.stderr
    lm_head_overlap_payload = json.loads(
        lm_head_overlap_output_path.read_text(encoding="utf-8")
    )
    assert (
        lm_head_overlap_payload["candidate_env"][
            "NFN_NATIVE_GPT_LM_HEAD_OVERLAP_LAST_DWEIGHT"
        ]
        == "1"
    )
    assert lm_head_overlap_payload["metric_ratio_gates"]["enabled"] is False

    full_row_output_path = tmp_path / "candidate-lm-head-row-chunk-65536-dry-run.json"
    full_row_env = os.environ.copy()
    full_row_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_row_chunk_65536",
            "NFN_SM120_NATIVE_JSON_OUT": str(full_row_output_path),
        }
    )

    full_row_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=full_row_env,
    )

    assert full_row_dry_run.returncode == 0, full_row_dry_run.stderr
    full_row_payload = json.loads(full_row_output_path.read_text(encoding="utf-8"))
    assert (
        full_row_payload["candidate_env"][
            "NFN_NATIVE_GPT_ALLOW_UNSAFE_LM_HEAD_ROW_CHUNK"
        ]
        == "1"
    )
    assert "--lm-head-row-chunk-size" in full_row_payload["candidate_command"]
    assert "65536" in full_row_payload["candidate_command"]
    assert full_row_payload["metric_ratio_gates"]["enabled"] is False

    split_row_output_path = tmp_path / "candidate-lm-head-row-chunk-49152-dry-run.json"
    split_row_env = os.environ.copy()
    split_row_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_row_chunk_49152",
            "NFN_SM120_NATIVE_JSON_OUT": str(split_row_output_path),
        }
    )

    split_row_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=split_row_env,
    )

    assert split_row_dry_run.returncode == 0, split_row_dry_run.stderr
    split_row_payload = json.loads(split_row_output_path.read_text(encoding="utf-8"))
    assert "NFN_NATIVE_GPT_ALLOW_UNSAFE_LM_HEAD_ROW_CHUNK" not in split_row_payload["candidate_env"]
    assert "--lm-head-row-chunk-size" in split_row_payload["baseline_command"]
    assert "32768" in split_row_payload["baseline_command"]
    assert "--lm-head-row-chunk-size" in split_row_payload["candidate_command"]
    assert "49152" in split_row_payload["candidate_command"]
    assert split_row_payload["metric_ratio_gates"]["enabled"] is False

    full_resident_output_path = tmp_path / "candidate-lm-head-full-resident-reuse-dry-run.json"
    full_resident_env = os.environ.copy()
    full_resident_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_full_resident_reuse",
            "NFN_SM120_NATIVE_JSON_OUT": str(full_resident_output_path),
        }
    )

    full_resident_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=full_resident_env,
    )

    assert full_resident_dry_run.returncode == 0, full_resident_dry_run.stderr
    full_resident_payload = json.loads(
        full_resident_output_path.read_text(encoding="utf-8")
    )
    assert (
        full_resident_payload["candidate_env"][
            "NFN_NATIVE_GPT_ALLOW_UNSAFE_LM_HEAD_ROW_CHUNK"
        ]
        == "1"
    )
    assert (
        full_resident_payload["candidate_env"][
            "NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS"
        ]
        == "1"
    )
    assert (
        full_resident_payload["candidate_env"][
            "NFN_NATIVE_GPT_FULL_BATCH_LM_HEAD_REUSE"
        ]
        == "1"
    )
    assert "--lm-head-row-chunk-size" in full_resident_payload["candidate_command"]
    assert "65536" in full_resident_payload["candidate_command"]
    assert full_resident_payload["metric_ratio_gates"]["enabled"] is False

    timeout_prone_env = os.environ.copy()
    timeout_prone_env.update(
        {
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_pipeline_chunks",
            "NFN_SM120_NATIVE_JSON_OUT": str(tmp_path / "timeout-prone.json"),
        }
    )

    timeout_prone_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=timeout_prone_env,
    )

    assert timeout_prone_run.returncode == 2
    assert "timeout-prone" in timeout_prone_run.stderr
    assert "NFN_SM120_NATIVE_ALLOW_TIMEOUT_PRONE_LM_HEAD_PROFILE=1" in timeout_prone_run.stderr

    cooperative_required_output_path = (
        tmp_path / "candidate-lm-head-cooperative-required-dry-run.json"
    )
    cooperative_required_env = os.environ.copy()
    cooperative_required_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_cooperative_backward_required",
            "NFN_SM120_NATIVE_JSON_OUT": str(cooperative_required_output_path),
        }
    )

    cooperative_required_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=cooperative_required_env,
    )

    assert cooperative_required_dry_run.returncode == 0, cooperative_required_dry_run.stderr
    cooperative_required_payload = json.loads(
        cooperative_required_output_path.read_text(encoding="utf-8")
    )
    assert (
        "--require-cooperative-lm-head-backward"
        in cooperative_required_payload["candidate_command"]
    )
    assert cooperative_required_payload["candidate_env"][
        "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD"
    ] == "1"
    assert cooperative_required_payload["metric_ratio_gates"]["enabled"] is False

    cooperative_loss_bins_output_path = (
        tmp_path / "candidate-lm-head-cooperative-loss-bins-dry-run.json"
    )
    cooperative_loss_bins_env = os.environ.copy()
    cooperative_loss_bins_env.update(
        {
            "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_cooperative_loss_bins",
            "NFN_SM120_NATIVE_JSON_OUT": str(cooperative_loss_bins_output_path),
        }
    )

    cooperative_loss_bins_dry_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=cooperative_loss_bins_env,
    )

    assert cooperative_loss_bins_dry_run.returncode == 0, cooperative_loss_bins_dry_run.stderr
    cooperative_loss_bins_payload = json.loads(
        cooperative_loss_bins_output_path.read_text(encoding="utf-8")
    )
    assert (
        cooperative_loss_bins_payload["candidate_env"][
            "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD"
        ]
        == "1"
    )
    assert (
        cooperative_loss_bins_payload["candidate_env"][
            "NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION"
        ]
        == "1"
    )
    assert (
        cooperative_loss_bins_payload["candidate_env"][
            "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_LOSS_BINS"
        ]
        == "1"
    )
    assert "--train-loss-every-steps" in cooperative_loss_bins_payload["baseline_command"]
    assert "--train-loss-every-steps" in cooperative_loss_bins_payload["candidate_command"]
    assert cooperative_loss_bins_payload["metric_ratio_gates"]["enabled"] is False
    assert "did not change any tracked route counters" in text

    rejected_cooperative_loss_bins_env = os.environ.copy()
    rejected_cooperative_loss_bins_env.update(
        {
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "lm_head_cooperative_loss_bins",
            "NFN_SM120_NATIVE_JSON_OUT": str(
                tmp_path / "rejected-lm-head-cooperative-loss-bins.json"
            ),
        }
    )

    rejected_cooperative_loss_bins = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=rejected_cooperative_loss_bins_env,
    )

    assert rejected_cooperative_loss_bins.returncode == 2
    assert "lm_head_cooperative_loss_bins is a rejected SM120 candidate" in (
        rejected_cooperative_loss_bins.stderr
    )
    assert "NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1" in (
        rejected_cooperative_loss_bins.stderr
    )

    lm_head_loss_bin_profiles = {
        "lm_head_loss_bins": {
            "NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION": "1",
        },
        "lm_head_ce_loss_bins_llmk_style_specialized": {
            "NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION": "1",
            "NFN_NATIVE_GPT_LM_HEAD_CE_LLMK_STYLE_SPECIALIZED": "1",
        },
        "lm_head_ce_loss_bins_default_specialized": {
            "NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION": "1",
            "NFN_NATIVE_GPT_LM_HEAD_CE_LOSS_BINS_DEFAULT_SPECIALIZED": "1",
        },
    }
    for profile_name, expected_env in lm_head_loss_bin_profiles.items():
        loss_bins_output_path = tmp_path / f"candidate-{profile_name}-dry-run.json"
        loss_bins_env = os.environ.copy()
        loss_bins_env.update(
            {
                "NFN_SM120_NATIVE_DRY_RUN_PLAN": "1",
                "NFN_SM120_NATIVE_PROFILE_DIR": "none",
                "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
                "NFN_SM120_NATIVE_CANDIDATE_PROFILE": profile_name,
                "NFN_SM120_NATIVE_JSON_OUT": str(loss_bins_output_path),
            }
        )

        loss_bins_dry_run = subprocess.run(
            ["bash", str(script)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=loss_bins_env,
        )

        assert loss_bins_dry_run.returncode == 0, loss_bins_dry_run.stderr
        loss_bins_payload = json.loads(loss_bins_output_path.read_text(encoding="utf-8"))
        for env_name, env_value in expected_env.items():
            assert loss_bins_payload["candidate_env"][env_name] == env_value
        assert "--train-loss-every-steps" in loss_bins_payload["baseline_command"]
        assert "--train-loss-every-steps" in loss_bins_payload["candidate_command"]
        assert loss_bins_payload["metric_ratio_gates"]["enabled"] is False

    token_profiles = {
        "token_weight_vector4_strided": {
            "baseline": {"NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT": "0"},
            "candidate": {"NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT": "1"},
        },
        "token_weight_threaded": {
            "baseline": {"NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT": "0"},
            "candidate": {"NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT": "1"},
        },
        "token_weight_fast_int32": {
            "baseline": {"NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT": "1"},
            "candidate": {"NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT": "0"},
        },
        "token_weight_two_pass_bf16": {
            "baseline": {"NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_INIT": "1"},
            "candidate": {"NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_INIT": "0"},
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
        assert token_payload["baseline_env"]["NFN_NATIVE_GPT_CUDA_VERSION_PREFLIGHT"] == "1"
        assert token_payload["candidate_env"]["NFN_NATIVE_GPT_CUDA_VERSION_PREFLIGHT"] == "1"
        for env_name, env_value in expected_env["baseline"].items():
            assert token_payload["baseline_env"][env_name] == env_value
        for env_name, env_value in expected_env["candidate"].items():
            assert token_payload["candidate_env"][env_name] == env_value
        assert token_payload["metric_ratio_gates"]["enabled"] is False

    rejected_token_env = os.environ.copy()
    rejected_token_env.update(
        {
            "NFN_SM120_NATIVE_STARTUP_ONLY": "1",
            "NFN_SM120_NATIVE_PROFILE_DIR": "none",
            "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES": "7",
            "NFN_SM120_NATIVE_CANDIDATE_PROFILE": "token_weight_vector4_strided",
            "NFN_SM120_NATIVE_JSON_OUT": str(tmp_path / "rejected-token-vector4-strided.json"),
        }
    )
    rejected_token_run = subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=rejected_token_env,
    )

    assert rejected_token_run.returncode == 2
    assert "rejected SM120 candidate" in rejected_token_run.stderr
    assert "NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1" in rejected_token_run.stderr


def test_native_gpt_sm120_candidate_sweep_keeps_same_script_gates() -> None:
    script = Path("tools/sweep_native_gpt_sm120_candidates.sh")

    proc = subprocess.run(
        ["bash", "-n", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    text = script.read_text(encoding="utf-8")
    assert "tools/bench_native_gpt_sm120_candidate.sh" in text
    assert 'NFN_SM120_NATIVE_CANDIDATE_PROFILE="$profile"' in text
    assert 'NFN_SM120_NATIVE_JSON_OUT="$json_out"' in text
    assert 'NFN_SM120_NATIVE_PROFILE_DIR="$profile_dir"' in text
    assert 'token_weight_vector4_strided' in text
    assert 'token_weight_threaded' in text
    assert 'token_weight_fast_int32' in text
    assert 'token_weight_two_pass_bf16' in text
    assert 'combined_device_arena' in text
    assert 'fail_count=$((fail_count + 1))' in text
    assert 'metric_ratio_gates' in text
    assert 'native_route_change_gate' in text
    assert 'summary.tsv' in text
    assert 'NFN_SM120_NATIVE_SWEEP_ALLOW_FAILURES' in text


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
        "\\\"linear_cublaslt_bgrad_gemm_count\\\": 1152, "
        "\\\"linear_cublaslt_bgrad_direct_write_count\\\": 0, "
        "\\\"linear_cublaslt_bgrad_accumulate_count\\\": 1152, "
        "\\\"block_backward_dinput_tk_gemm_count\\\": 96, "
        "\\\"block_backward_dinput_cublaslt_gemm_count\\\": 288, "
        "\\\"block_backward_dinput_bf16_gemm_count\\\": 96, "
        "\\\"block_backward_mlp_proj_dinput_before_dweight_count\\\": 0, "
        "\\\"block_backward_mlp_fc_dinput_before_dweight_count\\\": 0, "
        "\\\"block_backward_attn_proj_dinput_before_dweight_count\\\": 0, "
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
    assert route_changes["tracked_count"] == 12
    assert route_changes["unchanged"] == [
        "linear_tk_gemm_count",
        "linear_cublaslt_gemm_count",
        "linear_cublaslt_bgrad_gemm_count",
        "linear_cublaslt_bgrad_direct_write_count",
        "linear_cublaslt_bgrad_accumulate_count",
        "linear_bf16_gemm_count",
        "block_backward_dinput_tk_gemm_count",
        "block_backward_dinput_cublaslt_gemm_count",
        "block_backward_dinput_bf16_gemm_count",
        "block_backward_mlp_proj_dinput_before_dweight_count",
        "block_backward_mlp_fc_dinput_before_dweight_count",
        "block_backward_attn_proj_dinput_before_dweight_count",
    ]
    assert payload["native_route_change_gate"] == {
        "enabled": False,
        "passed": True,
        "has_route_counter_change": False,
        "has_strategy_value_change": False,
        "has_linear_shape_change": False,
        "has_cublaslt_plan_cache_change": False,
        "failure_reason": "",
    }
    strategy_changes = payload["native_strategy_value_changes"]
    assert strategy_changes["has_strategy_value_change"] is False
    assert strategy_changes["changed_count"] == 0
    assert strategy_changes["tracked_count"] == 0
    assert "native_route_counter_changes: has_route_counter_change=false changed_count=0" in proc.stdout
    assert "native_strategy_value_changes:" not in proc.stdout
    assert "tracked route counters did not change" in proc.stdout


def test_paired_kernel_speed_tool_fails_required_native_route_change_gate() -> None:
    script = Path("tools/paired_kernel_speed.py")
    output_path = Path(tempfile.mkdtemp()) / "paired-route-gate.json"
    native_json = (
        "{"
        "\\\"steps_completed\\\": 1, "
        "\\\"timing\\\": {\\\"train_loop_wall_ms\\\": 10.0}, "
        "\\\"linear_tk_gemm_count\\\": 1632, "
        "\\\"linear_cublaslt_gemm_count\\\": 2208, "
        "\\\"block_backward_dinput_tk_gemm_count\\\": 96, "
        "\\\"block_backward_dinput_cublaslt_gemm_count\\\": 288, "
        "\\\"block_backward_dinput_bf16_gemm_count\\\": 96, "
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
            "--require-native-route-change",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["native_route_change_gate"] == {
        "enabled": True,
        "passed": False,
        "has_route_counter_change": False,
        "has_strategy_value_change": False,
        "has_linear_shape_change": False,
        "has_cublaslt_plan_cache_change": False,
        "failure_reason": "candidate-native-metrics-did-not-change-route-strategy-or-plan",
    }
    assert "native_route_change_gate: passed=false" in proc.stdout
    assert "failure_reason: candidate-native-metrics-did-not-change-route-strategy-or-plan" in proc.stdout
    assert (
        "native route change gate failed: candidate-native-metrics-did-not-change-route-strategy-or-plan"
        in proc.stderr
    )


def test_paired_kernel_speed_tool_reports_strategy_change_without_route_counter_warning() -> None:
    script = Path("tools/paired_kernel_speed.py")
    output_path = Path(tempfile.mkdtemp()) / "paired-strategy-change.json"
    baseline_json = (
        "{"
        "\\\"steps_completed\\\": 1, "
        "\\\"timing\\\": {\\\"train_loop_wall_ms\\\": 10.0}, "
        "\\\"linear_tk_gemm_count\\\": 1632, "
        "\\\"linear_cublaslt_gemm_count\\\": 2208, "
        "\\\"linear_bf16_gemm_count\\\": 1824, "
        "\\\"lm_head_ce_loss_backward_strategy\\\": "
        "\\\"fused-row-losses-reduce-and-dlogits-public-vocab-no-pad-zero-bf16-u16-targets\\\", "
        "\\\"lm_head_ce_row_loss_sum_accumulate_enabled\\\": false"
        "}"
    )
    candidate_json = baseline_json.replace(
        "\\\"fused-row-losses-reduce-and-dlogits-public-vocab-no-pad-zero-bf16-u16-targets\\\"",
        "\\\"fused-row-losses-sum-accumulate-and-dlogits-public-vocab-no-pad-zero-bf16-u16-targets\\\"",
    ).replace(
        "\\\"lm_head_ce_row_loss_sum_accumulate_enabled\\\": false",
        "\\\"lm_head_ce_row_loss_sum_accumulate_enabled\\\": true",
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
            "NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE=1",
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
    strategy_changes = payload["native_strategy_value_changes"]
    assert strategy_changes["has_strategy_value_change"] is True
    assert strategy_changes["changed_count"] == 2
    assert strategy_changes["changed"]["lm_head_ce_loss_backward_strategy"] == {
        "baseline_values": [
            "fused-row-losses-reduce-and-dlogits-public-vocab-no-pad-zero-bf16-u16-targets"
        ],
        "candidate_values": [
            "fused-row-losses-sum-accumulate-and-dlogits-public-vocab-no-pad-zero-bf16-u16-targets"
        ],
    }
    assert strategy_changes["changed"]["lm_head_ce_row_loss_sum_accumulate_enabled"] == {
        "baseline_values": ["false"],
        "candidate_values": ["true"],
    }
    assert "native_strategy_value_changes: has_strategy_value_change=true changed_count=2" in proc.stdout
    assert "lm_head_ce_loss_backward_strategy: baseline=fused-row-losses-reduce" in proc.stdout
    assert "tracked route counters did not change" not in proc.stdout


def test_paired_kernel_speed_tool_reports_optimizer_tile_size_strategy_change() -> None:
    script = Path("tools/paired_kernel_speed.py")
    output_path = Path(tempfile.mkdtemp()) / "paired-optimizer-tile-change.json"
    baseline_json = (
        "{"
        "\\\"steps_completed\\\": 1, "
        "\\\"timing\\\": {\\\"train_loop_wall_ms\\\": 10.0}, "
        "\\\"optimizer_tile_size\\\": 1024, "
        "\\\"optimizer_tile_strategy\\\": \\\"tile-size-1024-sumsq-scale-adamw\\\""
        "}"
    )
    candidate_json = baseline_json.replace("\\\"optimizer_tile_size\\\": 1024", "\\\"optimizer_tile_size\\\": 2048").replace(
        "tile-size-1024-sumsq-scale-adamw",
        "tile-size-2048-sumsq-scale-adamw",
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
            "NFN_TILE_CUDA_OPTIMIZER_TILE_SIZE=2048",
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
    strategy_changes = payload["native_strategy_value_changes"]
    assert strategy_changes["has_strategy_value_change"] is True
    assert strategy_changes["changed"]["optimizer_tile_size"] == {
        "baseline_values": ["1024"],
        "candidate_values": ["2048"],
    }
    assert strategy_changes["changed"]["optimizer_tile_strategy"] == {
        "baseline_values": ["tile-size-1024-sumsq-scale-adamw"],
        "candidate_values": ["tile-size-2048-sumsq-scale-adamw"],
    }
    assert "native_strategy_value_changes: has_strategy_value_change=true changed_count=2" in proc.stdout
    assert "tracked route counters did not change" not in proc.stdout


def test_paired_kernel_speed_tool_reports_lm_head_ce_vector_io_strategy_change() -> None:
    script = Path("tools/paired_kernel_speed.py")
    output_path = Path(tempfile.mkdtemp()) / "paired-ce-vector-io-change.json"
    baseline_json = (
        "{"
        "\\\"steps_completed\\\": 1, "
        "\\\"timing\\\": {\\\"train_loop_wall_ms\\\": 10.0}, "
        "\\\"lm_head_ce_bf16_vector_io_strategy\\\": \\\"vec8-loads-scalar-stores\\\", "
        "\\\"lm_head_ce_bf16_vec_loads_enabled\\\": true, "
        "\\\"lm_head_ce_bf16_vec_stores_enabled\\\": false, "
        "\\\"lm_head_ce_bf16_vec_normal_stores_enabled\\\": false"
        "}"
    )
    candidate_json = (
        "{"
        "\\\"steps_completed\\\": 1, "
        "\\\"timing\\\": {\\\"train_loop_wall_ms\\\": 9.9}, "
        "\\\"lm_head_ce_bf16_vector_io_strategy\\\": \\\"vec8-loads-streaming-stores\\\", "
        "\\\"lm_head_ce_bf16_vec_loads_enabled\\\": true, "
        "\\\"lm_head_ce_bf16_vec_stores_enabled\\\": true, "
        "\\\"lm_head_ce_bf16_vec_normal_stores_enabled\\\": false"
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
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    strategy_changes = payload["native_strategy_value_changes"]
    assert strategy_changes["has_strategy_value_change"] is True
    assert strategy_changes["changed"]["lm_head_ce_bf16_vector_io_strategy"] == {
        "baseline_values": ["vec8-loads-scalar-stores"],
        "candidate_values": ["vec8-loads-streaming-stores"],
    }
    assert strategy_changes["changed"]["lm_head_ce_bf16_vec_stores_enabled"] == {
        "baseline_values": ["false"],
        "candidate_values": ["true"],
    }
    assert "lm_head_ce_bf16_vector_io_strategy: baseline=vec8-loads-scalar-stores" in proc.stdout
    assert "tracked route counters did not change" not in proc.stdout


def test_paired_kernel_speed_tool_reports_lm_head_ce_thread_strategy_change() -> None:
    script = Path("tools/paired_kernel_speed.py")
    output_path = Path(tempfile.mkdtemp()) / "paired-ce-thread-change.json"
    baseline_json = (
        "{"
        "\\\"steps_completed\\\": 1, "
        "\\\"timing\\\": {\\\"train_loop_wall_ms\\\": 10.0}, "
        "\\\"lm_head_ce_bf16_threads_per_row\\\": 1024"
        "}"
    )
    candidate_json = (
        "{"
        "\\\"steps_completed\\\": 1, "
        "\\\"timing\\\": {\\\"train_loop_wall_ms\\\": 9.9}, "
        "\\\"lm_head_ce_bf16_threads_per_row\\\": 512"
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
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    strategy_changes = payload["native_strategy_value_changes"]
    assert strategy_changes["has_strategy_value_change"] is True
    assert strategy_changes["changed"]["lm_head_ce_bf16_threads_per_row"] == {
        "baseline_values": ["1024"],
        "candidate_values": ["512"],
    }
    assert "lm_head_ce_bf16_threads_per_row: baseline=1024 candidate=512" in proc.stdout
    assert "tracked route counters did not change" not in proc.stdout


def test_paired_kernel_speed_tool_reports_train_loss_copy_scope_change() -> None:
    script = Path("tools/paired_kernel_speed.py")
    output_path = Path(tempfile.mkdtemp()) / "paired-train-loss-copy.json"
    baseline_json = (
        "{"
        "\\\"steps_completed\\\": 1, "
        "\\\"timing\\\": {\\\"train_loop_wall_ms\\\": 10.0}, "
        "\\\"train_loss_host_d2h_count\\\": 8, "
        "\\\"train_loss_host_d2h_copies_per_logged_step\\\": 8, "
        "\\\"train_loss_microbatch_host_d2h_copies_elided_per_logged_step\\\": 0, "
        "\\\"train_loss_device_accumulation_strategy\\\": \\\"microbatch-host-sum\\\", "
        "\\\"train_loss_host_copy_scope\\\": \\\"once-per-accumulation-microbatch\\\""
        "}"
    )
    candidate_json = (
        "{"
        "\\\"steps_completed\\\": 1, "
        "\\\"timing\\\": {\\\"train_loop_wall_ms\\\": 9.5}, "
        "\\\"train_loss_host_d2h_count\\\": 1, "
        "\\\"train_loss_host_d2h_copies_per_logged_step\\\": 1, "
        "\\\"train_loss_microbatch_host_d2h_copies_elided_per_logged_step\\\": 7, "
        "\\\"train_loss_device_accumulation_strategy\\\": \\\"optimizer-step-device-scalar-accumulate\\\", "
        "\\\"train_loss_host_copy_scope\\\": \\\"once-per-logged-optimizer-step\\\""
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
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    candidate_metrics = payload["candidate_native_metrics"]
    assert candidate_metrics["train_loss_host_d2h_count"]["mean"] == 1
    assert candidate_metrics["train_loss_host_d2h_copies_per_logged_step"]["mean"] == 1
    assert candidate_metrics["train_loss_microbatch_host_d2h_copies_elided_per_logged_step"]["mean"] == 7
    strategy_changes = payload["native_strategy_value_changes"]
    assert strategy_changes["changed"]["train_loss_device_accumulation_strategy"] == {
        "baseline_values": ["microbatch-host-sum"],
        "candidate_values": ["optimizer-step-device-scalar-accumulate"],
    }
    assert strategy_changes["changed"]["train_loss_host_copy_scope"] == {
        "baseline_values": ["once-per-accumulation-microbatch"],
        "candidate_values": ["once-per-logged-optimizer-step"],
    }
    assert "train_loss_host_d2h_count: mean=1.000000" in proc.stdout
    assert "train_loss_host_copy_scope: baseline=once-per-accumulation-microbatch" in proc.stdout


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


def test_paired_kernel_speed_tool_reports_plan_cache_change_without_shape_timing() -> None:
    script = Path("tools/paired_kernel_speed.py")
    output_path = Path(tempfile.mkdtemp()) / "paired-plan-cache-change.json"
    baseline_json = (
        "{"
        "\\\"steps_completed\\\": 1, "
        "\\\"timing\\\": {\\\"train_loop_wall_ms\\\": 10.0}, "
        "\\\"linear_tk_gemm_count\\\": 1632, "
        "\\\"linear_cublaslt_gemm_count\\\": 2208, "
        "\\\"linear_bf16_gemm_count\\\": 1824, "
        "\\\"linear_cublaslt_plan_cache\\\": [{"
        "\\\"m\\\": 768, \\\"n\\\": 50304, \\\"k\\\": 32768, "
        "\\\"op_a_name\\\": \\\"N\\\", \\\"op_b_name\\\": \\\"T\\\", "
        "\\\"selected_heuristic\\\": 1, "
        "\\\"returned_heuristics\\\": 9, "
        "\\\"workspace_bytes\\\": 134217728, "
        "\\\"epilogue\\\": 512"
        "}]"
        "}"
    )
    candidate_json = baseline_json.replace(
        "\\\"selected_heuristic\\\": 1",
        "\\\"selected_heuristic\\\": 0",
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
    plan_cache = payload["native_cublaslt_plan_cache"]
    assert plan_cache["has_plan_cache_change"] is True
    assert plan_cache["plan_cache_changed_count"] == 1
    plan_change = plan_cache["plan_cache_changed"][0]
    assert plan_change["shape"] == "cublaslt:768x50304x32768:N,T"
    assert plan_change["changed"]["selected_heuristics"] == {
        "baseline": [1],
        "candidate": [0],
    }
    assert payload["native_linear_shape_stats"]["enabled"] is False
    assert "native_cublaslt_plan_cache:" in proc.stdout
    assert "plan_cache_changes: changed_count=1" in proc.stdout
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


def test_paired_kernel_speed_tool_fails_min_metric_ratio_gate() -> None:
    script = Path("tools/paired_kernel_speed.py")
    output_path = Path(tempfile.mkdtemp()) / "paired-min-ratio-gate.json"
    baseline_json = (
        "{"
        "\\\"steps_completed\\\": 1, "
        "\\\"timing\\\": {\\\"train_tokens_per_second\\\": 100.0}"
        "}"
    )
    candidate_json = (
        "{"
        "\\\"steps_completed\\\": 1, "
        "\\\"timing\\\": {\\\"train_tokens_per_second\\\": 97.5}"
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
            "--min-candidate-ratio",
            "train_tokens_per_second=1.0",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    result = payload["metric_ratio_gates"]["results"][0]
    assert result == {
        "metric": "train_tokens_per_second",
        "stat": "mean",
        "min_ratio": 1.0,
        "actual_ratio": 0.975,
        "actual_mean_ratio": 0.975,
        "missing": False,
        "passed": False,
    }
    assert "mean:train_tokens_per_second: actual_ratio=0.975000" in proc.stdout
    assert "min_ratio=1.000000" in proc.stdout
    assert "metric ratio gate failed: train_tokens_per_second" in proc.stderr


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
    assert metrics["train_loop_cuda_event_wall_ms"] == 2493.74
    assert metrics["train_loop_cuda_event_wall_ms_per_step"] == 2493.74
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
    assert metrics["train_loop_cuda_event_wall_ms"] == 5000.0
    assert metrics["train_loop_cuda_event_wall_ms_per_step"] == 2500.0
    assert metrics["train_loop_cuda_event_first_step_wall_ms"] == 2400.0
    assert metrics["train_loop_cuda_event_first_step_wall_ms_per_step"] == 2400.0
    assert metrics["train_loop_cuda_event_steady_state_wall_ms"] == 2600.0
    assert metrics["train_loop_cuda_event_steady_state_wall_ms_per_step"] == 2600.0
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
        "linear_tk_dweight_gemm_count": {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
        "linear_cublaslt_gemm_count": {"mean": 2208.0, "median": 2208.0, "min": 2208.0, "max": 2208.0},
        "linear_cublaslt_bgrad_gemm_count": {"mean": 1152.0, "median": 1152.0, "min": 1152.0, "max": 1152.0},
        "linear_cublaslt_bgrad_direct_write_count": {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
        "linear_cublaslt_bgrad_accumulate_count": {"mean": 1152.0, "median": 1152.0, "min": 1152.0, "max": 1152.0},
        "block_backward_dinput_tk_gemm_count": {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
        "block_backward_dinput_cublaslt_gemm_count": {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
        "block_backward_dinput_bf16_gemm_count": {"mean": 384.0, "median": 384.0, "min": 384.0, "max": 384.0},
        "block_backward_mlp_proj_dinput_before_dweight_count": {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
        "block_backward_mlp_fc_dinput_before_dweight_count": {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
        "block_backward_attn_proj_dinput_before_dweight_count": {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
        "block_backward_qkv_dinput_before_dweight_count": {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
    }
    candidate = {
        "linear_tk_gemm_count": {"mean": 1344.0, "median": 1344.0, "min": 1344.0, "max": 1344.0},
        "linear_tk_dweight_gemm_count": {"mean": 16.0, "median": 16.0, "min": 16.0, "max": 16.0},
        "linear_cublaslt_gemm_count": {"mean": 2208.0, "median": 2208.0, "min": 2208.0, "max": 2208.0},
        "linear_cublaslt_bgrad_gemm_count": {"mean": 1152.0, "median": 1152.0, "min": 1152.0, "max": 1152.0},
        "linear_cublaslt_bgrad_direct_write_count": {"mean": 12.0, "median": 12.0, "min": 12.0, "max": 12.0},
        "linear_cublaslt_bgrad_accumulate_count": {"mean": 1140.0, "median": 1140.0, "min": 1140.0, "max": 1140.0},
        "block_backward_dinput_tk_gemm_count": {"mean": 4.0, "median": 4.0, "min": 4.0, "max": 4.0},
        "block_backward_dinput_cublaslt_gemm_count": {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
        "block_backward_dinput_bf16_gemm_count": {"mean": 380.0, "median": 380.0, "min": 380.0, "max": 380.0},
        "block_backward_mlp_proj_dinput_before_dweight_count": {"mean": 12.0, "median": 12.0, "min": 12.0, "max": 12.0},
        "block_backward_mlp_fc_dinput_before_dweight_count": {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
        "block_backward_attn_proj_dinput_before_dweight_count": {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
        "block_backward_qkv_dinput_before_dweight_count": {"mean": 12.0, "median": 12.0, "min": 12.0, "max": 12.0},
    }

    changes = module.summarize_native_route_counter_changes(baseline, candidate)

    assert changes["has_route_counter_change"] is True
    assert changes["changed_count"] == 8
    assert changes["tracked_count"] == 13
    assert changes["changed"]["linear_tk_gemm_count"] == {
        "baseline_mean": 1632.0,
        "candidate_mean": 1344.0,
        "delta": -288.0,
        "ratio": 1344.0 / 1632.0,
    }
    assert changes["changed"]["linear_tk_dweight_gemm_count"] == {
        "baseline_mean": 0.0,
        "candidate_mean": 16.0,
        "delta": 16.0,
        "ratio": None,
    }
    assert changes["changed"]["linear_cublaslt_bgrad_direct_write_count"] == {
        "baseline_mean": 0.0,
        "candidate_mean": 12.0,
        "delta": 12.0,
        "ratio": None,
    }
    assert changes["changed"]["linear_cublaslt_bgrad_accumulate_count"] == {
        "baseline_mean": 1152.0,
        "candidate_mean": 1140.0,
        "delta": -12.0,
        "ratio": 1140.0 / 1152.0,
    }
    assert changes["changed"]["block_backward_dinput_tk_gemm_count"] == {
        "baseline_mean": 0.0,
        "candidate_mean": 4.0,
        "delta": 4.0,
        "ratio": None,
    }
    assert changes["changed"]["block_backward_dinput_bf16_gemm_count"] == {
        "baseline_mean": 384.0,
        "candidate_mean": 380.0,
        "delta": -4.0,
        "ratio": 380.0 / 384.0,
    }
    assert changes["changed"]["block_backward_mlp_proj_dinput_before_dweight_count"] == {
        "baseline_mean": 0.0,
        "candidate_mean": 12.0,
        "delta": 12.0,
        "ratio": None,
    }
    assert changes["changed"]["block_backward_qkv_dinput_before_dweight_count"] == {
        "baseline_mean": 0.0,
        "candidate_mean": 12.0,
        "delta": 12.0,
        "ratio": None,
    }
    assert changes["unchanged"] == [
        "linear_cublaslt_gemm_count",
        "linear_cublaslt_bgrad_gemm_count",
        "block_backward_dinput_cublaslt_gemm_count",
        "block_backward_mlp_fc_dinput_before_dweight_count",
        "block_backward_attn_proj_dinput_before_dweight_count",
    ]
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
                    "train_loop_cuda_event_wall_ms": 18.0,
                    "train_loop_cuda_event_wall_ms_per_step": 4.5,
                    "train_loop_cuda_event_first_step_wall_ms": 6.0,
                    "train_loop_cuda_event_first_step_wall_ms_per_step": 6.0,
                    "train_loop_cuda_event_steady_state_wall_ms": 12.0,
                    "train_loop_cuda_event_steady_state_wall_ms_per_step": 4.0,
                    "train_loop_cuda_event_timing_enabled": True,
                    "train_tokens_per_second": 123.0,
                    "stage_timing": [
                        {"name": "block_backward", "total_ms": 9.0, "avg_ms": 3.0, "count": 3}
                    ],
                },
                "linear_tk_gemm_count": 8,
                "linear_cublaslt_gemm_count": 11,
                "linear_bf16_gemm_count": 13,
                "bf16_to_f32_vec4_count": 21,
                "lm_head_logits_tk_gemm_count": 4,
                "lm_head_logits_cublaslt_gemm_count": 0,
                "lm_head_logits_bf16_gemm_count": 4,
                "lm_head_dhidden_tk_gemm_count": 0,
                "lm_head_dhidden_cublaslt_gemm_count": 0,
                "lm_head_dhidden_bf16_gemm_count": 4,
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
                "lm_head_ce_bf16_threads_per_row": 1024,
                "lm_head_classifier_fusion_scope": "ce-dlogits-only-logits-dhidden-dweight-remain-separate",
                "lm_head_schedule_parity_status": "reference-parity-separate-logits-ce-dhidden-dweight",
                "lm_head_cooperative_backward_required": True,
                "lm_head_cooperative_backward_abi_wrapper_available": True,
                "lm_head_cooperative_backward_sequence_wrapper_available": True,
                "lm_head_cooperative_backward_kernel_available": False,
                "lm_head_cooperative_backward_fused_kernel_available": False,
                "lm_head_cooperative_backward_route_integrated": False,
                "lm_head_cooperative_backward_kernel_enabled": False,
                "lm_head_cooperative_backward_sequence_wrapper_enabled": False,
                "lm_head_cooperative_backward_strategy": "missing-required-sm120-parity-kernel",
                "lm_head_cooperative_sequence_launch_count": 32,
                "lm_head_cooperative_sequence_ce_launch_count": 32,
                "lm_head_cooperative_sequence_dhidden_launch_count": 32,
                "lm_head_cooperative_sequence_dweight_launch_count": 32,
                "lm_head_cooperative_sequence_concurrent_count": 32,
                "lm_head_cooperative_sequence_legacy_count": 0,
                "lm_head_cooperative_sequence_loss_bin_count": 0,
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
                "linear_cublaslt_plan_cache": [
                    {
                        "m": 768,
                        "n": 50304,
                        "k": 8192,
                        "op_a_name": "N",
                        "op_b_name": "T",
                        "selected_heuristic": 1,
                        "returned_heuristics": 8,
                        "workspace_bytes": 134217728,
                        "epilogue": 512,
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
    assert metrics["train_loop_cuda_event_wall_ms"] == 18.0
    assert metrics["train_loop_cuda_event_wall_ms_per_step"] == 4.5
    assert metrics["train_loop_cuda_event_first_step_wall_ms"] == 6.0
    assert metrics["train_loop_cuda_event_first_step_wall_ms_per_step"] == 6.0
    assert metrics["train_loop_cuda_event_steady_state_wall_ms"] == 12.0
    assert metrics["train_loop_cuda_event_steady_state_wall_ms_per_step"] == 4.0
    assert metrics["train_loop_cuda_event_timing_enabled"] is True
    assert metrics["train_tokens_per_second"] == 123.0
    assert metrics["linear_tk_gemm_count"] == 8
    assert metrics["linear_cublaslt_gemm_count"] == 11
    assert metrics["linear_bf16_gemm_count"] == 13
    assert metrics["bf16_to_f32_vec4_count"] == 21
    assert metrics["lm_head_logits_tk_gemm_count"] == 4
    assert metrics["lm_head_logits_cublaslt_gemm_count"] == 0
    assert metrics["lm_head_logits_bf16_gemm_count"] == 4
    assert metrics["lm_head_dhidden_tk_gemm_count"] == 0
    assert metrics["lm_head_dhidden_cublaslt_gemm_count"] == 0
    assert metrics["lm_head_dhidden_bf16_gemm_count"] == 4
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
    assert metrics["lm_head_ce_bf16_threads_per_row"] == 1024
    assert metrics["lm_head_classifier_fusion_scope"] == "ce-dlogits-only-logits-dhidden-dweight-remain-separate"
    assert metrics["lm_head_schedule_parity_status"] == "reference-parity-separate-logits-ce-dhidden-dweight"
    assert metrics["lm_head_cooperative_backward_required"] is True
    assert metrics["lm_head_cooperative_backward_abi_wrapper_available"] is True
    assert metrics["lm_head_cooperative_backward_sequence_wrapper_available"] is True
    assert metrics["lm_head_cooperative_backward_kernel_available"] is False
    assert metrics["lm_head_cooperative_backward_fused_kernel_available"] is False
    assert metrics["lm_head_cooperative_backward_route_integrated"] is False
    assert metrics["lm_head_cooperative_backward_kernel_enabled"] is False
    assert metrics["lm_head_cooperative_backward_sequence_wrapper_enabled"] is False
    assert metrics["lm_head_cooperative_backward_strategy"] == "missing-required-sm120-parity-kernel"
    assert metrics["lm_head_cooperative_sequence_launch_count"] == 32
    assert metrics["lm_head_cooperative_sequence_ce_launch_count"] == 32
    assert metrics["lm_head_cooperative_sequence_dhidden_launch_count"] == 32
    assert metrics["lm_head_cooperative_sequence_dweight_launch_count"] == 32
    assert metrics["lm_head_cooperative_sequence_concurrent_count"] == 32
    assert metrics["lm_head_cooperative_sequence_legacy_count"] == 0
    assert metrics["lm_head_cooperative_sequence_loss_bin_count"] == 0
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
    plan_cache = module.native_cublaslt_plan_cache_from_command_output(
        ["nfn_gpt_native_train", "--profile-json", str(sidecar)],
        "",
    )
    assert plan_cache == [
        {
            "m": 768,
            "n": 50304,
            "k": 8192,
            "op_a_name": "N",
            "op_b_name": "T",
            "selected_heuristic": 1,
            "returned_heuristics": 8,
            "workspace_bytes": 134217728,
            "epilogue": 512,
        }
    ]

    rows = [{"baseline": {"native_metrics": metrics}, "candidate": {"native_metrics": metrics}}]
    assert module.summarize_categorical_metric_rows(rows, "baseline") == {
        "status": ["native-sidecar-test"],
        "lm_head_logits_linear_strategy": ["padded-lm-head-bf16-cublaslt-fallback"],
        "lm_head_dhidden_linear_strategy": ["bf16-cublas-gemmex"],
        "lm_head_ce_loss_backward_strategy": ["separate-loss-partials-reduction-then-dlogits"],
        "lm_head_ce_bf16_threads_per_row": ["1024"],
        "lm_head_classifier_fusion_scope": ["ce-dlogits-only-logits-dhidden-dweight-remain-separate"],
        "lm_head_schedule_parity_status": ["reference-parity-separate-logits-ce-dhidden-dweight"],
        "lm_head_cooperative_backward_strategy": ["missing-required-sm120-parity-kernel"],
        "lm_head_cooperative_backward_required": ["true"],
        "lm_head_cooperative_backward_abi_wrapper_available": ["true"],
        "lm_head_cooperative_backward_sequence_wrapper_available": ["true"],
        "lm_head_cooperative_backward_kernel_available": ["false"],
        "lm_head_cooperative_backward_fused_kernel_available": ["false"],
        "lm_head_cooperative_backward_route_integrated": ["false"],
        "lm_head_cooperative_backward_kernel_enabled": ["false"],
        "lm_head_cooperative_backward_sequence_wrapper_enabled": ["false"],
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


def test_paired_kernel_speed_tool_summarizes_cublaslt_plan_cache() -> None:
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

    baseline_plan = {
        "m": 768,
        "n": 50304,
        "k": 8192,
        "op_a_name": "N",
        "op_b_name": "T",
        "selected_heuristic": 1,
        "returned_heuristics": 8,
        "workspace_bytes": 134217728,
        "epilogue": 512,
    }
    candidate_plan = dict(baseline_plan, selected_heuristic=0)
    rows = [
        {
            "baseline": {"native_cublaslt_plan_cache": [baseline_plan]},
            "candidate": {"native_cublaslt_plan_cache": [candidate_plan]},
        }
    ]

    summary = module.summarize_cublaslt_plan_cache(rows)

    assert summary["enabled"] is True
    assert summary["shared_count"] == 1
    assert summary["has_plan_cache_change"] is True
    assert summary["plan_cache_changed_count"] == 1
    assert summary["plan_cache_changed"][0]["shape"] == "cublaslt:768x50304x8192:N,T"
    changed = summary["plan_cache_changed"][0]["changed"]
    assert changed["selected_heuristics"] == {"baseline": [1], "candidate": [0]}


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


def test_paired_kernel_speed_tool_recognizes_linked_native_gpt_trainer(tmp_path: Path) -> None:
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

    argv = ["build/nfn_gpt_native_train_linked", "--backend", "tile-cuda"]
    assert module.looks_like_neuralfn_native_command(argv) is True
    assert module.command_env_with_auto_stage_timing(
        module.TimedCommand(name="candidate", argv=argv, env_overrides={}),
        env={},
        native_stage_timing=True,
    )["NFN_NATIVE_GPT_STAGE_TIMING"] == "1"

    profiled = module.argv_with_auto_profile_json(
        argv,
        command_name="candidate",
        profile_json_dir=tmp_path,
    )
    assert profiled[: len(argv)] == argv
    assert "--profile-json" in profiled
    profile_path = Path(profiled[profiled.index("--profile-json") + 1])
    assert profile_path.parent == tmp_path
    assert profile_path.name.startswith("candidate_")


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
    module.enforce_selected_gpu_guards(
        high_snapshot,
        "0",
        require_idle=True,
        max_utilization_pct=15.0,
        utilization_retries=1,
        utilization_retry_interval_seconds=0.0,
        allow_stale_utilization_without_compute_processes=True,
        snapshot_supplier=lambda: pytest.fail("stale-utilization allowance should not retry"),
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
        "time.sleep(2.0)\n"
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
            "0.5",
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
    assert sample["candidate"]["timeout_seconds"] == 0.5
    assert payload["command_timeout_seconds"] == 0.5
    time.sleep(2.5)
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
