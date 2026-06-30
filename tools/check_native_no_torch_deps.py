#!/usr/bin/env python3
"""Fail if native training paths depend on Python/Torch runtime libraries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import time
from textwrap import dedent
import tomllib


REQUIRED_DEFAULT_ARTIFACTS = (
    Path("build/nfn_gpt_native_train"),
    Path("build/libnfn_native_train_tile_ops.so"),
    Path("build/nfn_gpt2_native_train"),
    Path("build/nfn_train_gpt"),
    Path("build/nfn_train_gpt_sm120"),
    Path("build/nfn_native_train"),
    Path("build/nfn-native-train"),
    Path("build/nfn_native"),
    Path("build/nfn-native"),
)
OPTIONAL_DEFAULT_ARTIFACTS = (
    Path("build/nfn_gpt_native_train_linked"),
    Path("build/nfn_gpt2_evo_native_train"),
    Path("build/nfn_nanogpt_native_train"),
    Path("build/nfn_llama_native_train"),
    Path("build/nfn_mixllama_native_train"),
    Path("build/nfn_moe_jepa_evo_native_train"),
    Path("build/nfn_jepa_native_train"),
    Path("build/nfn_semantic_router_moe_native_train"),
    Path("build/nfn_deepseek_v4_native_train"),
    Path("build/linear_backward_bench"),
    Path("build/lm_head_backward_bench"),
    Path("build/libnfn_native_train_tile_ops_tk.so"),
)
REQUIRED_DEFAULT_ARTIFACT_GLOBS = (
    "neuralfn/_native_gpt.*.so",
    "neuralfn/_native_gpt2.*.so",
    "neuralfn/_native_train.*.so",
)
OPTIONAL_DEFAULT_ARTIFACT_GLOBS = (
)
ARTIFACT_SOURCE_DEPENDENCIES = {
    Path("build/nfn_gpt_native_train"): (
        Path("neuralfn/csrc/native_gpt2/nfn_gpt2_native_train.cpp"),
        Path("neuralfn/csrc/native_train/token_shards.cpp"),
        Path("neuralfn/csrc/native_train/token_shards.h"),
        Path("neuralfn/csrc/native_train/shipped_gpt_template_presets.h"),
        Path("neuralfn/csrc/native_train/tile_ops.cu"),
        Path("neuralfn/csrc/native_train/tile_ops.h"),
        Path("neuralfn/csrc/tile_cuda/kernels.cu"),
        Path("tools/build_native_train_tile_ops.sh"),
        Path("tools/build_native_gpt_cli.sh"),
    ),
    Path("build/nfn_gpt_native_train_linked"): (
        Path("neuralfn/csrc/native_gpt2/nfn_gpt2_native_train.cpp"),
        Path("neuralfn/csrc/native_train/token_shards.cpp"),
        Path("neuralfn/csrc/native_train/token_shards.h"),
        Path("neuralfn/csrc/native_train/shipped_gpt_template_presets.h"),
        Path("neuralfn/csrc/native_train/tile_ops.cu"),
        Path("neuralfn/csrc/native_train/tile_ops.h"),
        Path("neuralfn/csrc/tile_cuda/kernels.cu"),
        Path("tools/build_native_train_tile_ops.sh"),
        Path("tools/build_native_gpt_cli_linked.sh"),
    ),
    Path("build/nfn_train_gpt_sm120"): (
        Path("neuralfn/csrc/native_train/train_gpt_sm120.cpp"),
        Path("neuralfn/csrc/native_train/shipped_gpt_template_presets.h"),
        Path("tools/build_train_gpt_sm120_cli.sh"),
    ),
    Path("build/nfn_train_gpt"): (
        Path("neuralfn/csrc/native_train/train_gpt_sm120.cpp"),
        Path("neuralfn/csrc/native_train/shipped_gpt_template_presets.h"),
        Path("tools/build_train_gpt_cli.sh"),
    ),
    Path("build/nfn_gpt2_native_train"): (
        Path("neuralfn/csrc/native_gpt2/nfn_gpt2_native_train.cpp"),
        Path("neuralfn/csrc/native_train/token_shards.cpp"),
        Path("neuralfn/csrc/native_train/token_shards.h"),
        Path("neuralfn/csrc/native_train/shipped_gpt_template_presets.h"),
        Path("tools/build_native_gpt2_cli.sh"),
    ),
    Path("build/nfn_native_train"): (
        Path("neuralfn/csrc/native_train/nfn_native_train.cpp"),
        Path("tools/build_native_train_cli.sh"),
    ),
    Path("build/nfn-native-train"): (
        Path("neuralfn/csrc/native_train/nfn_native_train.cpp"),
        Path("tools/build_native_train_cli.sh"),
    ),
    Path("build/nfn_native"): (
        Path("neuralfn/csrc/native_train/nfn_native.cpp"),
        Path("tools/build_native_nfn_cli.sh"),
    ),
    Path("build/nfn-native"): (
        Path("neuralfn/csrc/native_train/nfn_native.cpp"),
        Path("tools/build_native_nfn_cli.sh"),
    ),
    Path("build/nfn_gpt2_evo_native_train"): (
        Path("neuralfn/csrc/native_train/gpt2_evo_native_train.cpp"),
        Path("neuralfn/csrc/native_train/shipped_gpt_template_presets.h"),
        Path("tools/build_native_missing_trainers.sh"),
    ),
    Path("build/nfn_nanogpt_native_train"): (
        Path("neuralfn/csrc/native_train/nanogpt_native_train.cpp"),
        Path("neuralfn/csrc/native_train/token_shards.cpp"),
        Path("neuralfn/csrc/native_train/token_shards.h"),
        Path("tools/build_native_missing_trainers.sh"),
    ),
    Path("build/nfn_llama_native_train"): (
        Path("neuralfn/csrc/native_train/missing_native_train.cpp"),
        Path("tools/build_native_missing_trainers.sh"),
    ),
    Path("build/nfn_mixllama_native_train"): (
        Path("neuralfn/csrc/native_train/missing_native_train.cpp"),
        Path("tools/build_native_missing_trainers.sh"),
    ),
    Path("build/nfn_moe_jepa_evo_native_train"): (
        Path("neuralfn/csrc/native_train/missing_native_train.cpp"),
        Path("tools/build_native_missing_trainers.sh"),
    ),
    Path("build/nfn_jepa_native_train"): (
        Path("neuralfn/csrc/native_train/missing_native_train.cpp"),
        Path("tools/build_native_missing_trainers.sh"),
    ),
    Path("build/nfn_semantic_router_moe_native_train"): (
        Path("neuralfn/csrc/native_train/missing_native_train.cpp"),
        Path("tools/build_native_missing_trainers.sh"),
    ),
    Path("build/nfn_deepseek_v4_native_train"): (
        Path("neuralfn/csrc/native_train/missing_native_train.cpp"),
        Path("tools/build_native_missing_trainers.sh"),
    ),
    Path("build/libnfn_native_train_tile_ops.so"): (
        Path("neuralfn/csrc/native_train/tile_ops.cu"),
        Path("neuralfn/csrc/native_train/tile_ops.h"),
        Path("neuralfn/csrc/tile_cuda/kernels.cu"),
        Path("tools/build_native_train_tile_ops.sh"),
    ),
    Path("build/libnfn_native_train_tile_ops_tk.so"): (
        Path("neuralfn/csrc/native_train/tile_ops.cu"),
        Path("neuralfn/csrc/native_train/tile_ops.h"),
        Path("neuralfn/csrc/tile_cuda/kernels.cu"),
        Path("tools/build_native_train_tile_ops.sh"),
    ),
    Path("build/linear_backward_bench"): (
        Path("neuralfn/csrc/native_train/linear_backward_bench.cpp"),
        Path("neuralfn/csrc/native_train/tile_ops.h"),
    ),
    Path("build/lm_head_backward_bench"): (
        Path("neuralfn/csrc/native_train/lm_head_backward_bench.cpp"),
        Path("neuralfn/csrc/native_train/tile_ops.h"),
    ),
}
NATIVE_BINDING_SOURCE_DEPENDENCIES = {
    "_native_gpt": (
        Path("neuralfn/csrc/native_gpt2/binding.cpp"),
        Path("tools/build_native_gpt_binding.sh"),
    ),
    "_native_gpt2": (
        Path("neuralfn/csrc/native_gpt2/binding.cpp"),
        Path("tools/build_native_gpt2_binding.sh"),
    ),
    "_native_train": (
        Path("neuralfn/csrc/native_train/binding.cpp"),
        Path("tools/build_native_train_binding.sh"),
    ),
}
ARTIFACT_REBUILD_COMMANDS = {
    Path("build/nfn_gpt_native_train"): ("bash", "tools/build_native_gpt_cli.sh"),
    Path("build/nfn_gpt_native_train_linked"): ("bash", "tools/build_native_gpt_cli_linked.sh"),
    Path("build/nfn_train_gpt"): ("bash", "tools/build_train_gpt_cli.sh"),
    Path("build/nfn_train_gpt_sm120"): ("bash", "tools/build_train_gpt_sm120_cli.sh"),
    Path("build/nfn_gpt2_native_train"): ("bash", "tools/build_native_gpt2_cli.sh"),
    Path("build/nfn_native_train"): ("bash", "tools/build_native_train_cli.sh"),
    Path("build/nfn-native-train"): ("bash", "tools/build_native_train_cli.sh"),
    Path("build/nfn_native"): ("bash", "tools/build_native_nfn_cli.sh"),
    Path("build/nfn-native"): ("bash", "tools/build_native_nfn_cli.sh"),
    Path("build/nfn_gpt2_evo_native_train"): ("bash", "tools/build_native_missing_trainers.sh"),
    Path("build/nfn_nanogpt_native_train"): ("bash", "tools/build_native_missing_trainers.sh"),
    Path("build/nfn_llama_native_train"): ("bash", "tools/build_native_missing_trainers.sh"),
    Path("build/nfn_mixllama_native_train"): ("bash", "tools/build_native_missing_trainers.sh"),
    Path("build/nfn_moe_jepa_evo_native_train"): ("bash", "tools/build_native_missing_trainers.sh"),
    Path("build/nfn_jepa_native_train"): ("bash", "tools/build_native_missing_trainers.sh"),
    Path("build/nfn_semantic_router_moe_native_train"): ("bash", "tools/build_native_missing_trainers.sh"),
    Path("build/nfn_deepseek_v4_native_train"): ("bash", "tools/build_native_missing_trainers.sh"),
    Path("build/libnfn_native_train_tile_ops.so"): ("bash", "tools/build_native_train_tile_ops.sh"),
    Path("build/libnfn_native_train_tile_ops_tk.so"): (
        "bash",
        "tools/build_native_train_tile_ops.sh",
        "build/libnfn_native_train_tile_ops_tk.so",
    ),
    Path("build/linear_backward_bench"): ("bash", "tools/build_linear_backward_bench.sh"),
    Path("build/lm_head_backward_bench"): ("bash", "tools/build_lm_head_backward_bench.sh"),
    Path("neuralfn/_native_gpt"): ("bash", "tools/build_native_gpt_binding.sh"),
    Path("neuralfn/_native_gpt2"): ("bash", "tools/build_native_gpt2_binding.sh"),
    Path("neuralfn/_native_train"): ("bash", "tools/build_native_train_binding.sh"),
}
FORBIDDEN_LIBRARY_MARKERS = (
    "libtorch",
    "libtorch_cpu",
    "libtorch_cuda",
    "libc10",
    "libc10_cuda",
    "libpython",
)
NATIVE_GPT_RUNTIME_CONTRACT_MARKERS = (
    b'"graph_editor_tensor_flow"',
    b'"torch_required"',
    b'"optimized_kernel_contract_required"',
    b'"optimized_kernel_contract_passed"',
    b'"lm_head_classifier_backward_path_class"',
    b'"lm_head_cooperative_backward_fused_kernel_abi_implementation_class"',
    b'"train_timing_contract"',
    b'"train_first_step_deferred_prewarm_diagnostic"',
    b'"train_steady_state_parity_metric_available"',
    b'"attention_backward_dprep_default_warps_per_block"',
    b'"sm120_memory_block_size"',
    b'"sm120_layernorm_bwd_blocks_per_sm"',
)
REQUIRED_ARTIFACT_STRING_MARKERS = {
    Path("build/nfn_gpt_native_train"): NATIVE_GPT_RUNTIME_CONTRACT_MARKERS,
    Path("build/nfn_gpt_native_train_linked"): NATIVE_GPT_RUNTIME_CONTRACT_MARKERS,
    Path("build/nfn_gpt2_native_train"): NATIVE_GPT_RUNTIME_CONTRACT_MARKERS,
}
FORBIDDEN_PYTHON_IMPORT_ROOTS = (
    "torch",
    "numpy",
    "tiktoken",
    "server.dataset_manager",
    "train_gpt_native",
    "infer_gpt",
    "nfn_impl",
)
FORBIDDEN_PROJECT_DEPENDENCY_PREFIXES = (
    "alembic",
    "datasets",
    "fastapi",
    "mcp",
    "networkx",
    "numpy",
    "pydantic",
    "pymysql",
    "python-multipart",
    "redis",
    "sqlalchemy",
    "tiktoken",
    "torch",
    "torchvision",
    "torchaudio",
    "uvicorn",
)
REQUIRED_OPTIONAL_DEPENDENCY_PREFIXES = (
    "datasets",
    "fastapi",
    "networkx",
    "numpy",
    "tiktoken",
)
FORBIDDEN_OPTIONAL_EXTRA_NAMES = ("torch",)
FORBIDDEN_OPTIONAL_EXTRA_DEPENDENCY_PREFIXES = {
    "all": (
        "torch",
        "torchvision",
        "torchaudio",
    ),
    "torch": (
        "torch",
        "torchvision",
        "torchaudio",
    ),
}
FORBIDDEN_REQUIREMENTS_DEPENDENCY_PREFIXES = FORBIDDEN_PROJECT_DEPENDENCY_PREFIXES
DEFAULT_PYTHON_ENTRYPOINTS = (
    (
        "train_gpt_fast_command",
        (
            sys.executable,
            "cli/scripts/train_gpt.py",
            "--tinystories",
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--native-cuda-no-checkpoint",
        ),
    ),
    (
        "train_gpt_fast_exec_handoff",
        (
            sys.executable,
            "cli/scripts/train_gpt.py",
            "--tinystories",
            "--native-cuda-no-checkpoint",
            "--max-steps",
            "1",
        ),
    ),
    (
        "train_gpt2_compat_fast_command",
        (
            sys.executable,
            "cli/scripts/train_gpt2.py",
            "--tinystories",
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--native-cuda-no-checkpoint",
        ),
    ),
    (
        "train_gpt2_compat_template_name_command",
        (
            sys.executable,
            "cli/scripts/train_gpt2.py",
            "--tinystories",
            "--template-name",
            "gpt2_moa",
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--native-cuda-no-checkpoint",
        ),
    ),
    (
        "train_gpt_native_fast_command",
        (
            sys.executable,
            "cli/scripts/train_gpt_native.py",
            "--tinystories",
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--native-cuda-no-checkpoint",
        ),
    ),
    (
        "train_gpt_native_metadata_missing_alias_command",
        (
            sys.executable,
            "cli/scripts/train_gpt_native.py",
            "--dataset-alias",
            "missing_alias_for_cpp_resolver",
            "--no-download-if-missing",
            "--native-cuda-print-plan",
        ),
    ),
    (
        "train_gpt_template_catalog_command",
        (
            sys.executable,
            "cli/scripts/train_gpt.py",
            "--native-cuda-list-templates",
        ),
    ),
    (
        "train_gpt2_compat_custom_graph_command",
        (
            sys.executable,
            "cli/scripts/train_gpt2.py",
            "--tinystories",
            "--graph-file",
            "__NFN_NATIVE_GRAPH_STUB__",
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--native-cuda-no-checkpoint",
        ),
    ),
    (
        "nfn_train_fast_command",
        (
            sys.executable,
            "cli/nfn.py",
            "train",
            "--tinystories",
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--no-checkpoint",
        ),
    ),
    (
        "nfn_train_gpt_template_catalog_command",
        (
            sys.executable,
            "cli/nfn.py",
            "train",
            "--base-model",
            "gpt",
            "--list-templates",
        ),
    ),
    (
        "nfn_train_auto_runner_fast_command",
        (
            sys.executable,
            "cli/nfn.py",
            "train",
            "--tinystories",
            "--native-cuda-runner",
            "auto",
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--no-checkpoint",
        ),
    ),
    (
        "nfn_train_default_fast_command",
        (
            sys.executable,
            "cli/nfn.py",
            "train",
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--no-checkpoint",
        ),
    ),
    (
        "nfn_console_train_fast_command",
        (
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import sys",
                    "sys.argv = [",
                    "    'nfn',",
                    "    'train',",
                    "    '--tinystories',",
                    "    '--native-cuda-dry-run',",
                    "    '--native-cuda-print-command',",
                    "    '--no-checkpoint',",
                    "]",
                    "from nfn import main",
                    "raise SystemExit(int(main() or 0))",
                ]
            ),
        ),
    ),
    (
        "nfn_programmatic_train_fast_command",
        (
            sys.executable,
            "-c",
            "\n".join(
                [
                    "from nfn import main",
                    "raise SystemExit(int(main([",
                    "    'train',",
                    "    '--tinystories',",
                    "    '--native-cuda-dry-run',",
                    "    '--native-cuda-print-command',",
                    "    '--no-checkpoint',",
                    "], stdin_isatty=False, stdout_isatty=False) or 0))",
                ]
            ),
        ),
    ),
    (
        "nfn_train_gpt2_evo_family_command",
        (
            sys.executable,
            "cli/nfn.py",
            "train",
            "--base-model",
            "gpt2-evo",
            "--tinystories",
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--no-checkpoint",
        ),
    ),
    (
        "nfn_train_llama_family_command",
        (
            sys.executable,
            "cli/nfn.py",
            "train",
            "--base-model",
            "llama",
            "--tinystories",
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--no-checkpoint",
        ),
    ),
    (
        "nfn_train_mixllama_family_command",
        (
            sys.executable,
            "cli/nfn.py",
            "train",
            "--base-model",
            "mixllama",
            "--tinystories",
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--no-checkpoint",
        ),
    ),
    (
        "nfn_train_jepa_family_command",
        (
            sys.executable,
            "cli/nfn.py",
            "train",
            "--base-model",
            "jepa",
            "--tinystories",
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--no-checkpoint",
        ),
    ),
    (
        "nfn_train_semantic_router_moe_family_command",
        (
            sys.executable,
            "cli/nfn.py",
            "train",
            "--base-model",
            "semantic-router-moe",
            "--tinystories",
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--no-checkpoint",
        ),
    ),
    (
        "nfn_train_deepseek_v4_family_command",
        (
            sys.executable,
            "cli/nfn.py",
            "train",
            "--base-model",
            "deepseek-v4",
            "--tinystories",
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--no-checkpoint",
        ),
    ),
    (
        "nfn_train_nanogpt_token_lm_family_command",
        (
            sys.executable,
            "cli/nfn.py",
            "train",
            "--base-model",
            "nanogpt",
            "--train-token-lm",
            "--tinystories",
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--no-checkpoint",
        ),
    ),
    (
        "native_sdk_imports",
        (
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import sys",
                    "import neuralfn",
                    "import neuralfn.native_gpt",
                    "import neuralfn.native_gpt2",
                    "import neuralfn.native_train",
                    "for name in ('torch', 'numpy', 'tiktoken'):",
                    "    assert name not in sys.modules, f'{name} loaded during native SDK import'",
                    "print('native-sdk-ok')",
                ]
            ),
        ),
    ),
    (
        "native_sdk_public_exports",
        (
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import neuralfn",
                    "import sys",
                    "for name in ('torch', 'numpy', 'tiktoken'):",
                    "    assert name not in sys.modules, f'{name} loaded before native SDK export access'",
                    "from neuralfn import NativeGptRunConfig, NativeGpt2RunConfig, NativeTrainRunConfig",
                    "from neuralfn import EvolutionaryTrainer, HybridConfig, HybridTrainer, SurrogateTrainer",
                    "from neuralfn import build_native_gpt_compiled_cli_run_config",
                    "from neuralfn import exec_native_gpt, exec_native_gpt2, exec_native_train",
                    "from neuralfn import build_native_sm120_gpt_run_config, build_native_train_run_config, run_native_train",
                    "from neuralfn.evolutionary import EvoConfig",
                    "from neuralfn.hybrid import GraphScope",
                    "from neuralfn.inference import InferenceCache, export_to_pt, load_pt_checkpoint",
                    "from neuralfn.trainer import TrainConfig",
                    "from neuralfn import native_gpt_checkpoint_sampler_argv, native_gpt_checkpoint_sampler_env",
                    "from neuralfn import native_gpt_kernel_backend, native_gpt_parameter_count",
                    "from neuralfn import native_gpt_prompt_tokens, render_native_gpt_checkpoint_sampler_text",
                    "from neuralfn import native_gpt_template_catalog, native_gpt2_template_catalog",
                    "from neuralfn import run_native_gpt_checkpoint_sampler, run_native_gpt_compiled_cli_capture",
                    "from neuralfn import run_native_gpt2_compiled_cli_capture",
                    "from neuralfn import native_train_model_registry, native_train_runner_status",
                    "from neuralfn import resolve_native_gpt_binding_command, resolve_native_gpt2_binding_command",
                    "from neuralfn import resolve_native_sm120_train_cli, resolve_native_train_binding_command",
                    "for name in ('torch', 'numpy', 'tiktoken'):",
                    "    assert name not in sys.modules, f'{name} loaded during native SDK export access'",
                    "assert NativeGptRunConfig.__name__ == 'NativeGptRunConfig'",
                    "assert NativeGpt2RunConfig.__name__ == 'NativeGpt2RunConfig'",
                    "assert NativeTrainRunConfig.__name__ == 'NativeTrainRunConfig'",
                    "assert SurrogateTrainer(object(), TrainConfig(epochs=1)).config.epochs == 1",
                    "assert EvolutionaryTrainer(object(), EvoConfig(generations=1)).config.generations == 1",
                    "assert HybridTrainer(object(), HybridConfig(outer_rounds=1)).config.outer_rounds == 1",
                    "assert GraphScope(path=(), graph=object()).path == ()",
                    "assert InferenceCache.__name__ == 'InferenceCache'",
                    "assert callable(export_to_pt)",
                    "assert callable(load_pt_checkpoint)",
                    "assert native_gpt_kernel_backend('tile-cuda') == 'tile-cuda'",
                    "assert native_gpt_parameter_count(max_seq_len=1024, padded_vocab_size=50304, num_layers=12, channels=768) > 0",
                    "assert native_gpt_prompt_tokens(prompt_tokens='1,2,3') == '1,2,3'",
                    "try:",
                    "    native_gpt_prompt_tokens(prompt='hello')",
                    "except RuntimeError as exc:",
                    "    assert 'token-id only by default' in str(exc)",
                    "else:",
                    "    raise AssertionError('text prompts must not import Python tokenizers on the native path by default')",
                    "assert 'Generated token ids' in render_native_gpt_checkpoint_sampler_text('{\"generated_tokens\":[1,2]}')",
                    "assert callable(build_native_gpt_compiled_cli_run_config)",
                    "assert callable(native_gpt_checkpoint_sampler_argv)",
                    "assert callable(native_gpt_checkpoint_sampler_env)",
                    "assert callable(native_gpt_template_catalog)",
                    "assert callable(native_gpt2_template_catalog)",
                    "assert callable(run_native_gpt_checkpoint_sampler)",
                    "assert callable(run_native_gpt_compiled_cli_capture)",
                    "assert callable(run_native_gpt2_compiled_cli_capture)",
                    "assert callable(exec_native_gpt)",
                    "assert callable(exec_native_gpt2)",
                    "assert callable(exec_native_train)",
                    "assert callable(build_native_sm120_gpt_run_config)",
                    "assert callable(build_native_train_run_config)",
                    "assert callable(run_native_train)",
                    "assert callable(resolve_native_gpt_binding_command)",
                    "assert callable(resolve_native_gpt2_binding_command)",
                    "assert callable(resolve_native_sm120_train_cli)",
                    "assert callable(resolve_native_train_binding_command)",
                    "sm120_cfg = build_native_sm120_gpt_run_config('gpt3', ['--dry-run'], native_sm120_cli='/tmp/nfn_train_gpt_sm120')",
                    "assert sm120_cfg.argv()[:3] == ['/tmp/nfn_train_gpt_sm120', '--base-model', 'gpt3']",
                    "strict_cfg = build_native_train_run_config('gpt', native_train_cli=sys.executable)",
                    "try:",
                    "    strict_cfg.argv()",
                    "except ValueError as exc:",
                    "    assert 'compiled C++ command' in str(exc)",
                    "else:",
                    "    raise AssertionError('native train SDK must reject Python launchers by default')",
                    "diag_cfg = build_native_train_run_config('gpt', native_train_cli=sys.executable, strict_native_command=False)",
                    "assert diag_cfg.argv()[0] == sys.executable",
                    "registry = native_train_model_registry()",
                    "assert isinstance(registry, dict) and registry",
                    "assert native_train_runner_status('compiled-cli').available in (True, False)",
                    "print('native-sdk-public-exports-ok lean-sdk-public-exports-ok')",
                ]
            ),
        ),
    ),
    (
        "native_sdk_binding_imports",
        (
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import importlib",
                    "import importlib.util",
                    "loaded = []",
                    "for name in ('neuralfn._native_gpt', 'neuralfn._native_gpt2', 'neuralfn._native_train'):",
                    "    if importlib.util.find_spec(name) is not None:",
                    "        importlib.import_module(name)",
                    "        loaded.append(name)",
                    "print('native-sdk-binding-imports-ok:' + ','.join(loaded))",
                ]
            ),
        ),
    ),
)
DEFAULT_SCRIPT_NATIVE_DISPATCH_ENTRYPOINTS = (
    (
        "train_gpt2_evo_default_native_dispatch",
        (
            sys.executable,
            "cli/scripts/train_gpt2_evo.py",
        ),
    ),
    (
        "train_nanogpt_default_native_dispatch",
        (
            sys.executable,
            "cli/scripts/train_nanogpt.py",
        ),
    ),
    (
        "train_llama_fast_default_native_dispatch",
        (
            sys.executable,
            "cli/scripts/train_llama_fast.py",
        ),
    ),
    (
        "train_llama_megakernel_default_native_dispatch",
        (
            sys.executable,
            "cli/scripts/train_llama_megakernel.py",
        ),
    ),
    (
        "train_mixllama_fast_default_native_dispatch",
        (
            sys.executable,
            "cli/scripts/train_mixllama_fast.py",
        ),
    ),
    (
        "train_jepa_semantic_default_native_dispatch",
        (
            sys.executable,
            "cli/scripts/train_jepa_semantic.py",
        ),
    ),
    (
        "train_semantic_router_moe_default_native_dispatch",
        (
            sys.executable,
            "cli/scripts/train_semantic_router_moe.py",
        ),
    ),
    (
        "train_semantic_router_moe_overnight_default_native_dispatch",
        (
            sys.executable,
            "cli/scripts/train_semantic_router_moe-overnight.py",
        ),
    ),
    (
        "train_deepseek_v4_default_native_dispatch",
        (
            sys.executable,
            "cli/scripts/train_deepseek_v4.py",
        ),
    ),
)
DEFAULT_NATIVE_SHIM_IMPORT_ENTRYPOINTS = (
    (
        "train_gpt2_evo_module_import",
        (
            sys.executable,
            "-c",
            "import train_gpt2_evo; print('train_gpt2_evo-import-ok')",
        ),
    ),
    (
        "train_nanogpt_module_import",
        (
            sys.executable,
            "-c",
            "import train_nanogpt; print('train_nanogpt-import-ok')",
        ),
    ),
    (
        "train_deepseek_v4_module_import",
        (
            sys.executable,
            "-c",
            "import train_deepseek_v4; print('train_deepseek_v4-import-ok')",
        ),
    ),
    (
        "train_llama_fast_module_import",
        (
            sys.executable,
            "-c",
            "import train_llama_fast; print('train_llama_fast-import-ok')",
        ),
    ),
    (
        "train_llama_megakernel_module_import",
        (
            sys.executable,
            "-c",
            "import train_llama_megakernel; print('train_llama_megakernel-import-ok')",
        ),
    ),
    (
        "train_mixllama_fast_module_import",
        (
            sys.executable,
            "-c",
            "import train_mixllama_fast; print('train_mixllama_fast-import-ok')",
        ),
    ),
    (
        "train_semantic_router_moe_module_import",
        (
            sys.executable,
            "-c",
            "import train_semantic_router_moe; print('train_semantic_router_moe-import-ok')",
        ),
    ),
    (
        "train_semantic_router_moe_overnight_module_import",
        (
            sys.executable,
            "-c",
            "import importlib.util; from pathlib import Path; p=Path('cli/scripts/train_semantic_router_moe-overnight.py'); s=importlib.util.spec_from_file_location('train_semantic_router_moe_overnight', p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('train_semantic_router_moe_overnight-import-ok')",
        ),
    ),
    (
        "train_jepa_semantic_module_import",
        (
            sys.executable,
            "-c",
            "import train_jepa_semantic; print('train_jepa_semantic-import-ok')",
        ),
    ),
    (
        "infer_jepa_semantic_module_import",
        (
            sys.executable,
            "-c",
            "import infer_jepa_semantic; print('infer_jepa_semantic-import-ok')",
        ),
    ),
)
DEFAULT_SHELL_ENTRYPOINTS = (
    (
        "bench_linear_backward_dry_run",
        (
            "bash",
            "tools/bench_linear_backward_candidate.sh",
        ),
        {
            "NFN_LINEAR_BACKWARD_DRY_RUN": "1",
            "NFN_LINEAR_BACKWARD_PROFILE": "smoke-dinput",
            "NFN_LINEAR_BACKWARD_BENCH_BIN": "/tmp/nfn-linear-bench-stub",
            "NFN_NATIVE_TILE_OPS_LIB": "/tmp/libnfn-native-train-tile-ops-stub.so",
            "NFN_LINEAR_BACKWARD_JSON_OUT": os.devnull,
        },
    ),
    (
        "bench_lm_head_backward_dry_run",
        (
            "bash",
            "tools/bench_lm_head_backward_candidate.sh",
        ),
        {
            "NFN_LM_HEAD_BACKWARD_DRY_RUN": "1",
            "NFN_LM_HEAD_BACKWARD_PROFILE": "smoke",
            "NFN_LM_HEAD_BACKWARD_CANDIDATE_FIRST": "1",
            "NFN_LM_HEAD_BACKWARD_BENCH_BIN": "/tmp/nfn-lm-head-bench-stub",
            "NFN_NATIVE_TILE_OPS_LIB": "/tmp/libnfn-native-train-tile-ops-stub.so",
            "NFN_LM_HEAD_BACKWARD_JSON_OUT": os.devnull,
        },
    ),
    (
        "bench_native_gpt_linear_hot_matrix_dry_run",
        (
            "bash",
            "tools/bench_native_gpt_linear_hot_matrix.sh",
        ),
        {
            "NFN_LINEAR_HOT_MATRIX_DRY_RUN": "1",
            "NFN_LINEAR_HOT_MATRIX_PROFILES": "smoke-dinput smoke-dweight",
            "NFN_LINEAR_BACKWARD_DRY_RUN": "1",
            "NFN_LINEAR_BACKWARD_BENCH_BIN": "/tmp/nfn-linear-bench-stub",
            "NFN_NATIVE_TILE_OPS_LIB": "/tmp/libnfn-native-train-tile-ops-stub.so",
        },
    ),
    (
        "train_gpt_dry_run",
        (
            "bash",
            "tools/train_gpt.sh",
            "--print-command",
            "--dry-run",
            "--no-checkpoint",
        ),
        {},
    ),
    (
        "train_gpt_gpt3_dry_run",
        (
            "bash",
            "tools/train_gpt.sh",
            "--base-model",
            "gpt3",
            "--print-command",
            "--dry-run",
            "--no-checkpoint",
        ),
        {},
    ),
    (
        "train_gpt_sm120_dry_run",
        (
            "bash",
            "tools/train_gpt_sm120.sh",
            "--print-command",
            "--dry-run",
            "--no-checkpoint",
        ),
        {
            "NFN_SM120_NATIVE_BETA1": "0.87",
            "NFN_SM120_NATIVE_BETA2": "0.98",
            "NFN_SM120_NATIVE_ADAM_EPS": "1e-8",
            "NFN_SM120_NATIVE_GRAD_CLIP_NORM": "0.75",
        },
    ),
    (
        "train_gpt_sm120_gpt3_dry_run",
        (
            "bash",
            "tools/train_gpt_sm120.sh",
            "--base-model",
            "gpt3",
            "--print-command",
            "--dry-run",
            "--no-checkpoint",
        ),
        {},
    ),
    (
        "train_gpt_sm120_custom_graph_dry_run",
        (
            "bash",
            "tools/train_gpt_sm120.sh",
            "--template-name",
            "gpt2_moa",
            "--graph-file",
            "/tmp/native-compatible-gpt-graph.json",
            "--print-command",
            "--dry-run",
            "--no-checkpoint",
        ),
        {},
    ),
    (
        "train_gpt_compiled_dry_run",
        (
            "build/nfn_train_gpt",
            "--print-command",
            "--dry-run",
            "--no-checkpoint",
        ),
        {
            "NFN_NATIVE_GPT_MODEL_FAMILY": "gpt",
            "NFN_NATIVE_GPT_TEMPLATE_NAME": "gpt",
        },
    ),
    (
        "train_gpt_compiled_generic_env_dry_run",
        (
            "build/nfn_train_gpt",
            "--print-command",
            "--dry-run",
            "--no-checkpoint",
        ),
        {
            "NFN_NATIVE_GPT_MODEL_FAMILY": "gpt3",
            "NFN_NATIVE_GPT_TEMPLATE_NAME": "gpt3",
            "NFN_NATIVE_GPT_BATCH_SIZE": "32",
            "NFN_NATIVE_GPT_TRAIN_SEQ_LEN": "2048",
        },
    ),
    (
        "train_gpt_compiled_template_selector_dry_run",
        (
            "build/nfn_train_gpt",
            "--base-model",
            "gpt2_moa",
            "--print-command",
            "--dry-run",
            "--no-checkpoint",
        ),
        {},
    ),
    (
        "train_gpt_compiled_custom_graph_dry_run",
        (
            "build/nfn_train_gpt",
            "--template-name",
            "gpt2_moa",
            "--graph-file",
            "/tmp/native-compatible-gpt-graph.json",
            "--print-command",
            "--dry-run",
            "--no-checkpoint",
        ),
        {},
    ),
    (
        "train_gpt_sm120_compiled_dry_run",
        (
            "build/nfn_train_gpt_sm120",
            "--print-command",
            "--dry-run",
            "--no-checkpoint",
        ),
        {
            "NFN_SM120_NATIVE_BETA1": "0.87",
            "NFN_SM120_NATIVE_BETA2": "0.98",
            "NFN_SM120_NATIVE_ADAM_EPS": "1e-8",
            "NFN_SM120_NATIVE_GRAD_CLIP_NORM": "0.75",
        },
    ),
    (
        "train_gpt_sm120_compiled_gpt3_dry_run",
        (
            "build/nfn_train_gpt_sm120",
            "--base-model",
            "gpt3",
            "--print-command",
            "--dry-run",
            "--no-checkpoint",
        ),
        {},
    ),
    (
        "train_gpt_sm120_compiled_template_selector_dry_run",
        (
            "build/nfn_train_gpt_sm120",
            "--base-model",
            "gpt2_moa",
            "--print-command",
            "--dry-run",
            "--no-checkpoint",
        ),
        {},
    ),
    (
        "train_gpt_sm120_compiled_custom_graph_dry_run",
        (
            "build/nfn_train_gpt_sm120",
            "--template-name",
            "gpt2_moa",
            "--graph-file",
            "/tmp/native-compatible-gpt-graph.json",
            "--print-command",
            "--dry-run",
            "--no-checkpoint",
        ),
        {},
    ),
    (
        "nfn_native_compiled_train_gpt_dry_run",
        (
            "build/nfn-native",
            "train",
            "--base-model",
            "gpt",
            "--print-command",
            "--dry-run",
            "--no-checkpoint",
        ),
        {},
    ),
    (
        "nfn_native_compiled_infer_prompt_tokens_dry_run",
        (
            "build/nfn-native",
            "infer",
            "--checkpoint",
            "__NFN_NATIVE_CHECKPOINT_DIR__",
            "--prompt-tokens",
            "1,2,3",
            "--max-new-tokens",
            "2",
            "--print-command",
        ),
        {},
    ),
    (
        "nfn_native_compiled_infer_weights_info_dry_run",
        (
            "build/nfn-native",
            "infer",
            "--weights",
            "__NFN_NATIVE_CHECKPOINT_DIR__",
            "--native-info",
            "--print-command",
        ),
        {},
    ),
    (
        "native_gpt_linked_list_templates",
        (
            "build/nfn_gpt_native_train_linked",
            "--list-templates",
        ),
        {},
    ),
    (
        "native_gpt2_compat_list_templates",
        (
            "build/nfn_gpt2_native_train",
            "--list-templates",
        ),
        {},
    ),
    (
        "native_train_registry_list_models",
        (
            "build/nfn_native_train",
            "--list-models",
            "--json",
        ),
        {},
    ),
    (
        "native_train_gpt_list_templates",
        (
            "build/nfn_native_train",
            "--base-model",
            "gpt",
            "--list-templates",
        ),
        {},
    ),
    (
        "native_train_gpt_wrapper_list_templates",
        (
            "build/nfn_native_train",
            "--base-model",
            "gpt",
            "--native-cuda-list-templates",
        ),
        {},
    ),
)
NATIVE_GPT_CHECKPOINT_MAGIC = 20240326
NATIVE_GPT_CHECKPOINT_HEADER_INTS = 256
DEFAULT_MAX_ENTRYPOINT_SECONDS = 2.0
PYTHON_ENTRYPOINT_REQUIRED_STDOUT_MARKERS = {
    "infer_gpt_native_info": (
        "Native GPT checkpoint detected",
        "precision:",
        "shape:",
    ),
    "infer_gpt_native_sample_prompt_tokens": (
        "--sample-checkpoint",
        "--prompt-tokens",
        "1,2,3",
    ),
    "nfn_infer_native_info": (
        "Native GPT checkpoint detected",
        "precision:",
        "shape:",
    ),
    "nfn_infer_native_directory_info": (
        "Native GPT checkpoint detected",
        "model_00000020.bin",
    ),
    "nfn_infer_native_sample_prompt_tokens": (
        "--sample-checkpoint",
        "--prompt-tokens",
        "1,2,3",
    ),
    "nfn_console_infer_native_info": (
        "Native GPT checkpoint detected",
        "precision:",
        "shape:",
    ),
    "nfn_console_infer_native_sample_prompt_tokens": (
        "--sample-checkpoint",
        "--prompt-tokens",
        "1,2,3",
    ),
}
REQUIRED_NATIVE_DENSE_GPT_TEMPLATE_SELECTORS = (
    "gpt",
    "gpt2",
    "gpt2_modern",
    "gpt3",
    "gpt2_megakernel",
    "gpt2_moa",
    "nanogpt",
    "nanogpt_modern",
    "nanogpt_megakernel",
)
REQUIRED_NATIVE_DENSE_GPT_TEMPLATES = {
    "gpt": {"model_dim": 768, "num_heads": 12, "num_layers": 12, "seq_len": 1024},
    "gpt2": {"model_dim": 768, "num_heads": 12, "num_layers": 12, "seq_len": 1024},
    "gpt2_modern": {"model_dim": 768, "num_heads": 12, "num_layers": 12, "seq_len": 1024},
    "gpt2_megakernel": {"model_dim": 768, "num_heads": 12, "num_layers": 12, "seq_len": 1024},
    "gpt2_moa": {"model_dim": 768, "num_heads": 12, "num_layers": 12, "seq_len": 1024},
    "gpt3": {"model_dim": 768, "num_heads": 12, "num_layers": 12, "seq_len": 2048},
    "nanogpt": {"model_dim": 320, "num_heads": 5, "num_layers": 5, "seq_len": 1024},
    "nanogpt_modern": {"model_dim": 320, "num_heads": 5, "num_layers": 5, "seq_len": 1024},
    "nanogpt_megakernel": {"model_dim": 320, "num_heads": 5, "num_layers": 5, "seq_len": 1024},
}
REQUIRED_NATIVE_MISSING_TEMPLATE_SENTINELS = (
    "llama",
    "semantic_router_moe_modern",
)
NATIVE_TEMPLATE_CATALOG_ENTRYPOINTS = (
    "native_gpt_linked_list_templates",
    "native_gpt2_compat_list_templates",
    "native_train_gpt_list_templates",
    "native_train_gpt_wrapper_list_templates",
)


def _native_gpt_parameter_count(
    *,
    max_seq_len: int,
    padded_vocab_size: int,
    num_layers: int,
    channels: int,
) -> int:
    c = int(channels)
    l = int(num_layers)
    return int(
        int(padded_vocab_size) * c
        + int(max_seq_len) * c
        + l * c
        + l * c
        + l * 3 * c * c
        + l * 3 * c
        + l * c * c
        + l * c
        + l * c
        + l * c
        + l * 4 * c * c
        + l * 4 * c
        + l * c * 4 * c
        + l * c
        + c
        + c
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifacts",
        nargs="*",
        type=Path,
        default=(),
        help=(
            "Native executable/shared-library artifacts to inspect with ldd. "
            "Explicit paths are required to exist; the default scan includes "
            "required native artifacts plus optional compiled trainer artifacts "
            "that are present in build/."
        ),
    )
    parser.add_argument(
        "--skip-artifacts",
        action="store_true",
        help="Skip ldd inspection of compiled artifacts.",
    )
    parser.add_argument(
        "--skip-stale-artifacts",
        action="store_true",
        help="Skip source-mtime freshness checks for native compiled artifacts.",
    )
    parser.add_argument(
        "--rebuild-stale",
        action="store_true",
        help=(
            "Rebuild stale known native artifacts with the matching tools/build_*.sh script "
            "before running dependency and import checks."
        ),
    )
    parser.add_argument(
        "--skip-python-entrypoints",
        action="store_true",
        help="Skip import-blocked checks for native Python entrypoints.",
    )
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report.")
    parser.add_argument(
        "--max-entrypoint-seconds",
        type=float,
        default=DEFAULT_MAX_ENTRYPOINT_SECONDS,
        help=(
            "Maximum wall-clock seconds allowed for each native Python fast-path entrypoint. "
            "Use 0 to disable this startup budget."
        ),
    )
    return parser.parse_args()


def default_artifacts() -> list[Path]:
    artifacts = list(REQUIRED_DEFAULT_ARTIFACTS)
    for pattern in REQUIRED_DEFAULT_ARTIFACT_GLOBS:
        matches = sorted(Path().glob(pattern))
        artifacts.extend(matches or [Path(pattern)])
    artifacts.extend(path for path in OPTIONAL_DEFAULT_ARTIFACTS if path.exists())
    for pattern in OPTIONAL_DEFAULT_ARTIFACT_GLOBS:
        artifacts.extend(sorted(Path().glob(pattern)))
    return artifacts


def artifact_is_required(path: Path, repo_root: Path) -> bool:
    try:
        rel_path = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        rel_path = path
    rel = Path(rel_path)
    if rel in REQUIRED_DEFAULT_ARTIFACTS:
        return True
    rel_text = rel.as_posix()
    return any(Path().glob(pattern) and rel.match(pattern) for pattern in REQUIRED_DEFAULT_ARTIFACT_GLOBS) or any(
        Path(match).as_posix() == rel_text
        for pattern in REQUIRED_DEFAULT_ARTIFACT_GLOBS
        for match in Path().glob(pattern)
    )


def artifact_source_dependencies(path: Path, repo_root: Path) -> list[Path]:
    try:
        rel_path = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        rel_path = path
    dependencies = list(ARTIFACT_SOURCE_DEPENDENCIES.get(rel_path, ()))
    name = path.name
    for marker, marker_dependencies in sorted(
        NATIVE_BINDING_SOURCE_DEPENDENCIES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if name.startswith(marker):
            dependencies.extend(marker_dependencies)
            break
    return dependencies


def artifact_rebuild_command(path: Path, repo_root: Path) -> tuple[str, ...] | None:
    try:
        rel_path = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        rel_path = path
    command = ARTIFACT_REBUILD_COMMANDS.get(rel_path)
    if command is not None:
        return command
    name = path.name
    for marker, marker_command in sorted(
        (
            ("_native_gpt2", ARTIFACT_REBUILD_COMMANDS[Path("neuralfn/_native_gpt2")]),
            ("_native_gpt", ARTIFACT_REBUILD_COMMANDS[Path("neuralfn/_native_gpt")]),
            ("_native_train", ARTIFACT_REBUILD_COMMANDS[Path("neuralfn/_native_train")]),
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if name.startswith(marker):
            return marker_command
    return None


def stale_artifact_sources(path: Path, repo_root: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    artifact_mtime = path.stat().st_mtime
    stale_sources: list[dict[str, object]] = []
    for source in artifact_source_dependencies(path, repo_root):
        source_path = repo_root / source
        if not source_path.exists():
            stale_sources.append(
                {
                    "source": str(source),
                    "exists": False,
                    "source_mtime": None,
                    "artifact_mtime": artifact_mtime,
                }
            )
            continue
        source_mtime = source_path.stat().st_mtime
        if source_mtime > artifact_mtime:
            stale_sources.append(
                {
                    "source": str(source),
                    "exists": True,
                    "source_mtime": source_mtime,
                    "artifact_mtime": artifact_mtime,
                }
            )
    return stale_sources


def artifact_should_rebuild(path: Path, stale_sources: list[dict[str, object]], rebuild_stale: bool) -> bool:
    return bool(rebuild_stale and (not path.exists() or stale_sources))


def ldd_output(path: Path) -> str:
    proc = subprocess.run(
        ["ldd", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.stdout


def required_artifact_string_markers(path: Path, repo_root: Path) -> tuple[bytes, ...]:
    try:
        rel_path = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        rel_path = path
    return REQUIRED_ARTIFACT_STRING_MARKERS.get(rel_path, ())


def missing_artifact_string_markers(path: Path, repo_root: Path) -> list[str]:
    markers = required_artifact_string_markers(path, repo_root)
    if not markers:
        return []
    try:
        data = path.read_bytes()
    except OSError as exc:
        return [f"<read error: {exc}>"]
    return [
        marker.decode("utf-8", errors="replace")
        for marker in markers
        if marker not in data
    ]


def rebuild_stale_artifact(path: Path, repo_root: Path) -> dict[str, object]:
    command = artifact_rebuild_command(path, repo_root)
    if command is None:
        return {
            "available": False,
            "command": [],
            "returncode": None,
            "stdout": "",
            "stderr": "no rebuild command is registered for this artifact",
        }
    started = time.perf_counter()
    proc = subprocess.run(
        list(command),
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "available": True,
        "command": list(command),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _write_native_cli_stub(root: Path) -> Path:
    stub = root / "nfn_gpt_native_train"
    stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


def _write_gpt2_evo_cli_stub(root: Path, dense_cli: Path) -> Path:
    stub = root / "nfn_gpt2_evo_native_train"
    stub.write_text(
        "#!/bin/sh\n"
        "for arg in \"$@\"; do\n"
        "  if [ \"$arg\" = \"--print-command\" ]; then\n"
        f"    printf '%s\\n' '{dense_cli} --backend tile-cuda --train-transformer-lm --template-name gpt2 --tile-cuda-activation-dtype nvfp4 --layer-evo'\n"
        "    exit 0\n"
        "  fi\n"
        "done\n"
        "printf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


def _write_native_checkpoint_stub(root: Path) -> Path:
    checkpoint = root / "model_00000010.bin"
    max_seq_len = 8
    vocab_size = 16
    num_layers = 1
    num_heads = 1
    channels = 4
    padded_vocab_size = 16
    header = [0] * NATIVE_GPT_CHECKPOINT_HEADER_INTS
    header[:8] = [
        NATIVE_GPT_CHECKPOINT_MAGIC,
        5,
        max_seq_len,
        vocab_size,
        num_layers,
        num_heads,
        channels,
        padded_vocab_size,
    ]
    parameter_count = _native_gpt_parameter_count(
        max_seq_len=max_seq_len,
        padded_vocab_size=padded_vocab_size,
        num_layers=num_layers,
        channels=channels,
    )
    checkpoint.write_bytes(
        struct.pack("<" + "i" * NATIVE_GPT_CHECKPOINT_HEADER_INTS, *header) +
        b"\0" * (parameter_count * 2)
    )
    (root / "DONE_00000010").write_text("", encoding="utf-8")
    return checkpoint


def _write_latest_native_checkpoint_stub(root: Path) -> Path:
    checkpoint = root / "model_00000010.bin"
    if not checkpoint.exists():
        checkpoint = _write_native_checkpoint_stub(root)
    latest = root / "model_00000020.bin"
    latest.write_bytes(checkpoint.read_bytes())
    (root / "DONE_00000020").write_text("", encoding="utf-8")
    return latest


def _write_native_graph_stub(root: Path) -> Path:
    graph = root / "native-custom-gpt-graph.json"
    graph.write_text(
        json.dumps(
            {
                "name": "native-custom-gpt-graph",
                "nodes": [],
                "edges": [],
                "input_node_ids": [],
                "output_node_ids": [],
            }
        ),
        encoding="utf-8",
    )
    return graph


def _write_import_blocker(root: Path) -> None:
    root.joinpath("sitecustomize.py").write_text(
        dedent(
            f"""
            import importlib.abc

            FORBIDDEN = {FORBIDDEN_PYTHON_IMPORT_ROOTS!r}

            class _NativeNoTorchImportBlocker(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    for root in FORBIDDEN:
                        if fullname == root or fullname.startswith(root + "."):
                            raise ImportError(
                                "native no-Torch verifier blocked import of " + fullname
                            )
                    return None

            import sys
            sys.meta_path.insert(0, _NativeNoTorchImportBlocker())
            """
        ).lstrip(),
        encoding="utf-8",
    )


def python_entrypoint_report(repo_root: Path, *, max_entrypoint_seconds: float) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    startup_budget_seconds = max(0.0, float(max_entrypoint_seconds))
    with tempfile.TemporaryDirectory(prefix="nfn-native-no-torch-") as tmp:
        temp_root = Path(tmp)
        _write_import_blocker(temp_root)
        native_cli = _write_native_cli_stub(temp_root)
        gpt2_evo_cli = _write_gpt2_evo_cli_stub(temp_root, native_cli)
        native_checkpoint = _write_native_checkpoint_stub(temp_root)
        _write_latest_native_checkpoint_stub(temp_root)
        native_graph = _write_native_graph_stub(temp_root)
        env = os.environ.copy()
        env["NFN_NATIVE_GPT_CLI"] = str(native_cli)
        env["NFN_NATIVE_GPT2_CLI"] = str(native_cli)
        env["NFN_NATIVE_GPT2_EVO_CLI"] = str(gpt2_evo_cli)
        env["NFN_NATIVE_NANOGPT_CLI"] = str(native_cli)
        env["NFN_NATIVE_LLAMA_CLI"] = str(native_cli)
        env["NFN_NATIVE_MIXLLAMA_CLI"] = str(native_cli)
        env["NFN_NATIVE_JEPA_CLI"] = str(native_cli)
        env["NFN_NATIVE_DEEPSEEK_V4_CLI"] = str(native_cli)
        env["NFN_NATIVE_SEMANTIC_ROUTER_MOE_CLI"] = str(native_cli)
        cli_root = repo_root / "cli"
        scripts_root = cli_root / "scripts"
        env["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (
                str(temp_root),
                str(scripts_root),
                str(cli_root),
                str(repo_root),
                env.get("PYTHONPATH", ""),
            )
            if part
        )
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        entrypoints = [
            *DEFAULT_PYTHON_ENTRYPOINTS,
            *DEFAULT_SCRIPT_NATIVE_DISPATCH_ENTRYPOINTS,
            *DEFAULT_NATIVE_SHIM_IMPORT_ENTRYPOINTS,
            (
                "train_gpt2_evo_fast_command",
                (
                    sys.executable,
                    "cli/scripts/train_gpt2_evo.py",
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ),
            ),
            (
                "nfn_train_gpt_template_name_command",
                (
                    sys.executable,
                    "cli/nfn.py",
                    "train",
                    "--base-model",
                    "gpt",
                    "--tinystories",
                    "--template-name",
                    "gpt2_moa",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--no-checkpoint",
                ),
            ),
            (
                "nfn_train_gpt_custom_graph_command",
                (
                    sys.executable,
                    "cli/nfn.py",
                    "train",
                    "--base-model",
                    "gpt3",
                    "--tinystories",
                    "--graph-file",
                    str(native_graph),
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--no-checkpoint",
                ),
            ),
            (
                "train_nanogpt_fast_command",
                (
                    sys.executable,
                    "cli/scripts/train_nanogpt.py",
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ),
            ),
            (
                "train_llama_fast_command",
                (
                    sys.executable,
                    "cli/scripts/train_llama_fast.py",
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ),
            ),
            (
                "train_llama_megakernel_fast_command",
                (
                    sys.executable,
                    "cli/scripts/train_llama_megakernel.py",
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ),
            ),
            (
                "train_mixllama_fast_command",
                (
                    sys.executable,
                    "cli/scripts/train_mixllama_fast.py",
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ),
            ),
            (
                "train_jepa_semantic_fast_command",
                (
                    sys.executable,
                    "cli/scripts/train_jepa_semantic.py",
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ),
            ),
            (
                "train_semantic_router_moe_fast_command",
                (
                    sys.executable,
                    "cli/scripts/train_semantic_router_moe.py",
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ),
            ),
            (
                "train_semantic_router_moe_overnight_fast_command",
                (
                    sys.executable,
                    "cli/scripts/train_semantic_router_moe-overnight.py",
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ),
            ),
            (
                "train_deepseek_v4_fast_command",
                (
                    sys.executable,
                    "cli/scripts/train_deepseek_v4.py",
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ),
            ),
            (
                "infer_gpt_native_info",
                (
                    sys.executable,
                    "cli/scripts/infer_gpt.py",
                    "--native-checkpoint",
                    str(native_checkpoint),
                    "--native-info",
                ),
            ),
            (
                "infer_gpt_checkpoint_alias_directory_info",
                (
                    sys.executable,
                    "cli/scripts/infer_gpt.py",
                    "--checkpoint",
                    str(temp_root),
                    "--native-info",
                ),
            ),
            (
                "infer_gpt_native_sample_prompt_tokens",
                (
                    sys.executable,
                    "cli/scripts/infer_gpt.py",
                    "--native-checkpoint",
                    str(native_checkpoint),
                    "--prompt-tokens",
                    "1,2,3",
                    "--max-new-tokens",
                    "2",
                ),
            ),
            (
                "infer_llama_fast_help_no_torch",
                (
                    sys.executable,
                    "cli/scripts/infer_llama_fast.py",
                    "--help",
                ),
            ),
            (
                "infer_llama_megakernel_help_no_torch",
                (
                    sys.executable,
                    "cli/scripts/infer_llama_megakernel.py",
                    "--help",
                ),
            ),
            (
                "infer_mixllama_fast_help_no_torch",
                (
                    sys.executable,
                    "cli/scripts/infer_mixllama_fast.py",
                    "--help",
                ),
            ),
            (
                "infer_nanogpt_help_no_torch",
                (
                    sys.executable,
                    "cli/scripts/infer_nanogpt.py",
                    "--help",
                ),
            ),
            (
                "infer_semantic_router_moe_help_no_torch",
                (
                    sys.executable,
                    "cli/scripts/infer_semantic_router_moe.py",
                    "--help",
                ),
            ),
            (
                "infer_jepa_semantic_help_no_torch",
                (
                    sys.executable,
                    "cli/scripts/infer_jepa_semantic.py",
                    "--help",
                ),
            ),
            (
                "eval_llama_fast_help_no_torch",
                (
                    sys.executable,
                    "cli/scripts/eval_llama_fast.py",
                    "--help",
                ),
            ),
            (
                "nfn_infer_native_info",
                (
                    sys.executable,
                    "cli/nfn.py",
                    "infer",
                    "--native-checkpoint",
                    str(native_checkpoint),
                    "--native-info",
                ),
            ),
            (
                "nfn_infer_native_directory_info",
                (
                    sys.executable,
                    "cli/nfn.py",
                    "infer",
                    "--checkpoint",
                    str(temp_root),
                    "--native-info",
                ),
            ),
            (
                "nfn_infer_native_sample_prompt_tokens",
                (
                    sys.executable,
                    "cli/nfn.py",
                    "infer",
                    "--native-checkpoint",
                    str(native_checkpoint),
                    "--prompt-tokens",
                    "1,2,3",
                    "--max-new-tokens",
                    "2",
                ),
            ),
            (
                "nfn_console_infer_native_info",
                (
                    sys.executable,
                    "-c",
                    "\n".join(
                        [
                            "import sys",
                            "sys.argv = [",
                            "    'nfn',",
                            "    'infer',",
                            "    '--native-checkpoint',",
                            f"    {str(native_checkpoint)!r},",
                            "    '--native-info',",
                            "]",
                            "from nfn import main",
                            "raise SystemExit(int(main() or 0))",
                        ]
                    ),
                ),
            ),
            (
                "nfn_console_infer_native_sample_prompt_tokens",
                (
                    sys.executable,
                    "-c",
                    "\n".join(
                        [
                            "import sys",
                            "sys.argv = [",
                            "    'nfn',",
                            "    'infer',",
                            "    '--native-checkpoint',",
                            f"    {str(native_checkpoint)!r},",
                            "    '--prompt-tokens',",
                            "    '1,2,3',",
                            "    '--max-new-tokens',",
                            "    '2',",
                            "]",
                            "from nfn import main",
                            "raise SystemExit(int(main() or 0))",
                        ]
                    ),
                ),
            ),
        ]
        for name, command in entrypoints:
            command = tuple(str(native_graph) if part == "__NFN_NATIVE_GRAPH_STUB__" else part for part in command)
            started = time.perf_counter()
            proc = subprocess.run(
                list(command),
                cwd=repo_root,
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            elapsed_seconds = time.perf_counter() - started
            startup_within_budget = (
                True if startup_budget_seconds <= 0.0 else elapsed_seconds <= startup_budget_seconds
            )
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()
            missing_stdout_markers = [
                marker
                for marker in PYTHON_ENTRYPOINT_REQUIRED_STDOUT_MARKERS.get(name, ())
                if marker not in stdout
            ]
            entries.append(
                {
                    "name": name,
                    "command": list(command),
                    "returncode": proc.returncode,
                    "passed": (
                        proc.returncode == 0
                        and startup_within_budget
                        and not missing_stdout_markers
                    ),
                    "elapsed_seconds": elapsed_seconds,
                    "startup_budget_seconds": startup_budget_seconds,
                    "startup_within_budget": startup_within_budget,
                    "stdout": stdout,
                    "stderr": stderr,
                    "missing_stdout_markers": missing_stdout_markers,
                }
            )
    return entries


def shell_entrypoint_report(repo_root: Path, *, max_entrypoint_seconds: float) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    startup_budget_seconds = max(0.0, float(max_entrypoint_seconds))
    base_env = os.environ.copy()
    base_env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    launcher_builders = {
        "build/nfn_train_gpt": "tools/build_train_gpt_cli.sh",
        "build/nfn_train_gpt_sm120": "tools/build_train_gpt_sm120_cli.sh",
        "build/nfn-native": "tools/build_native_nfn_cli.sh",
        "build/nfn_native": "tools/build_native_nfn_cli.sh",
    }
    with tempfile.TemporaryDirectory(prefix="nfn-native-shell-entrypoints-") as tmp:
        temp_root = Path(tmp)
        _write_latest_native_checkpoint_stub(temp_root)
        for name, command, extra_env in DEFAULT_SHELL_ENTRYPOINTS:
            env = base_env.copy()
            env.update(extra_env)
            run_command = [
                str(temp_root) if part == "__NFN_NATIVE_CHECKPOINT_DIR__" else part
                for part in command
            ]
            build_started: float | None = None
            build_elapsed = 0.0
            first = run_command[0] if run_command else ""
            builder = launcher_builders.get(first)
            launcher_path = repo_root / first
            stale_launcher_sources = (
                stale_artifact_sources(launcher_path, repo_root)
                if builder is not None and launcher_path.exists()
                else []
            )
            if builder is not None and (not launcher_path.exists() or stale_launcher_sources):
                temp_binary = temp_root / Path(first).name
                build_started = time.perf_counter()
                build_proc = subprocess.run(
                    ["bash", builder, str(temp_binary)],
                    cwd=repo_root,
                    env=env,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                build_elapsed = time.perf_counter() - build_started
                if build_proc.returncode != 0:
                    entries.append(
                        {
                            "name": name,
                            "command": run_command,
                            "returncode": build_proc.returncode,
                            "passed": False,
                            "elapsed_seconds": build_elapsed,
                            "startup_budget_seconds": startup_budget_seconds,
                            "startup_within_budget": False,
                            "stdout": build_proc.stdout.strip(),
                            "stderr": build_proc.stderr.strip(),
                            "preflight": "build_missing_launcher",
                            "preflight_command": ["bash", builder, str(temp_binary)],
                        }
                    )
                    continue
                run_command[0] = str(temp_binary)
            started = time.perf_counter()
            proc = subprocess.run(
                run_command,
                cwd=repo_root,
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            elapsed_seconds = time.perf_counter() - started
            startup_within_budget = (
                True if startup_budget_seconds <= 0.0 else elapsed_seconds <= startup_budget_seconds
            )
            entries.append(
                {
                    "name": name,
                    "command": run_command,
                    "returncode": proc.returncode,
                    "passed": proc.returncode == 0 and startup_within_budget,
                    "elapsed_seconds": elapsed_seconds,
                    "startup_budget_seconds": startup_budget_seconds,
                    "startup_within_budget": startup_within_budget,
                    "stdout": proc.stdout.strip(),
                    "stderr": proc.stderr.strip(),
                    "preflight": "build_missing_launcher" if build_started is not None else None,
                    "preflight_elapsed_seconds": build_elapsed,
                }
            )
    return entries


def _parse_json_stdout(stdout: object) -> tuple[dict[str, object] | None, str | None]:
    if not isinstance(stdout, str) or not stdout.strip():
        return None, "empty stdout"
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON stdout: {exc}"
    if not isinstance(parsed, dict):
        return None, "JSON stdout was not an object"
    return parsed, None


def _template_geometry_matches(
    geometry: object,
    expected_geometry: dict[str, int],
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(geometry, dict):
        return False, ["missing selected_template_geometry"]
    for field, expected_value in expected_geometry.items():
        if geometry.get(field) != expected_value:
            errors.append(f"{field}={geometry.get(field)!r}, expected {expected_value!r}")
    return not errors, errors


def _validate_native_template_catalog(
    *,
    entrypoint_name: str,
    entry: dict[str, object] | None,
) -> dict[str, object]:
    report: dict[str, object] = {
        "entrypoint": entrypoint_name,
        "passed": False,
        "required_dense_templates": {},
        "missing_native_sentinels": {},
        "errors": [],
    }
    errors = report["errors"]
    if not isinstance(errors, list):
        raise AssertionError("internal report error storage mismatch")
    if entry is None:
        errors.append("entrypoint was not run")
        return report
    if not bool(entry.get("passed")):
        errors.append(f"entrypoint did not pass: returncode={entry.get('returncode')!r}")
    catalog, parse_error = _parse_json_stdout(entry.get("stdout"))
    if parse_error is not None or catalog is None:
        errors.append(parse_error or "template catalog JSON was unavailable")
        return report
    report["action"] = catalog.get("action")
    report["shipped_template_catalog_count"] = catalog.get("shipped_template_catalog_count")
    report["selector_count"] = catalog.get("selector_count")
    report["token_shards_resolved"] = catalog.get("token_shards_resolved")
    native_dense_selectors = catalog.get("native_dense_gpt_template_selectors")
    report["native_dense_gpt_template_selectors"] = native_dense_selectors
    if catalog.get("action") != "list_templates":
        errors.append(f"action={catalog.get('action')!r}, expected 'list_templates'")
    if catalog.get("token_shards_resolved") is not False:
        errors.append("template catalog listing resolved token shards")
    if native_dense_selectors != list(REQUIRED_NATIVE_DENSE_GPT_TEMPLATE_SELECTORS):
        errors.append(
            "native_dense_gpt_template_selectors="
            f"{native_dense_selectors!r}, expected {list(REQUIRED_NATIVE_DENSE_GPT_TEMPLATE_SELECTORS)!r}"
        )
    templates = catalog.get("templates")
    if not isinstance(templates, list):
        errors.append("templates was not a list")
        return report
    by_name = {
        template.get("name"): template
        for template in templates
        if isinstance(template, dict) and isinstance(template.get("name"), str)
    }
    dense_reports: dict[str, object] = {}
    for template_name, expected_geometry in REQUIRED_NATIVE_DENSE_GPT_TEMPLATES.items():
        template = by_name.get(template_name)
        template_report: dict[str, object] = {
            "passed": False,
            "status": None,
            "native_runnable": None,
            "geometry": None,
            "errors": [],
        }
        template_errors = template_report["errors"]
        if not isinstance(template_errors, list):
            raise AssertionError("internal template error storage mismatch")
        if template is None:
            template_errors.append("template missing from catalog")
        else:
            template_report["status"] = template.get("selected_graph_support_status")
            template_report["native_runnable"] = template.get("selected_graph_native_runnable")
            geometry = template.get("selected_template_geometry")
            template_report["geometry"] = geometry
            if template.get("selected_graph_support_status") != "native-transformer-lm":
                template_errors.append(
                    "status="
                    f"{template.get('selected_graph_support_status')!r}, "
                    "expected 'native-transformer-lm'"
                )
            if template.get("selected_graph_native_runnable") is not True:
                template_errors.append("selected_graph_native_runnable was not true")
            geometry_passed, geometry_errors = _template_geometry_matches(geometry, expected_geometry)
            if not geometry_passed:
                template_errors.extend(geometry_errors)
        template_report["passed"] = not template_errors
        if template_errors:
            errors.append(f"{template_name}: {'; '.join(template_errors)}")
        dense_reports[template_name] = template_report
    sentinel_reports: dict[str, object] = {}
    for template_name in REQUIRED_NATIVE_MISSING_TEMPLATE_SENTINELS:
        template = by_name.get(template_name)
        template_report = {
            "passed": False,
            "status": None,
            "native_runnable": None,
            "errors": [],
        }
        template_errors = template_report["errors"]
        if not isinstance(template_errors, list):
            raise AssertionError("internal sentinel error storage mismatch")
        if template is None:
            template_errors.append("template missing from catalog")
        else:
            template_report["status"] = template.get("selected_graph_support_status")
            template_report["native_runnable"] = template.get("selected_graph_native_runnable")
            if template.get("selected_graph_support_status") != "template-native-trainer-missing":
                template_errors.append(
                    "status="
                    f"{template.get('selected_graph_support_status')!r}, "
                    "expected 'template-native-trainer-missing'"
                )
            if template.get("selected_graph_native_runnable") is not False:
                template_errors.append("selected_graph_native_runnable was not false")
        template_report["passed"] = not template_errors
        if template_errors:
            errors.append(f"{template_name}: {'; '.join(template_errors)}")
        sentinel_reports[template_name] = template_report
    report["required_dense_templates"] = dense_reports
    report["missing_native_sentinels"] = sentinel_reports
    report["passed"] = not errors
    return report


def native_template_catalog_report(shell_report: list[dict[str, object]]) -> dict[str, object]:
    shell_entrypoints = {
        str(entry.get("name")): entry
        for entry in shell_report
        if isinstance(entry, dict) and entry.get("name") is not None
    }
    catalog_reports = [
        _validate_native_template_catalog(
            entrypoint_name=name,
            entry=shell_entrypoints.get(name),
        )
        for name in NATIVE_TEMPLATE_CATALOG_ENTRYPOINTS
    ]
    return {
        "passed": all(bool(report.get("passed")) for report in catalog_reports),
        "entrypoints": catalog_reports,
    }


def _summary_counts(entries: list[dict[str, object]]) -> dict[str, int]:
    total = len(entries)
    passed = sum(1 for entry in entries if bool(entry.get("passed")))
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
    }


def artifact_summary_counts(entries: list[dict[str, object]]) -> dict[str, int]:
    total = len(entries)
    missing = sum(1 for entry in entries if entry.get("error") == "missing")
    stale = sum(1 for entry in entries if bool(entry.get("stale_sources")))
    forbidden = sum(1 for entry in entries if bool(entry.get("forbidden")))
    passed = total - sum(
        1
        for entry in entries
        if entry.get("error") == "missing"
        or (bool(entry.get("stale_sources")) and bool(entry.get("required", True)))
        or bool(entry.get("forbidden"))
    )
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "missing": missing,
        "stale": stale,
        "forbidden": forbidden,
    }


def native_template_catalog_summary_counts(report: dict[str, object] | None) -> dict[str, int]:
    entrypoints = report.get("entrypoints") if isinstance(report, dict) else []
    if not isinstance(entrypoints, list):
        entrypoints = []
    return _summary_counts([entry for entry in entrypoints if isinstance(entry, dict)])


def project_dependency_report(repo_root: Path) -> dict[str, object]:
    pyproject = repo_root / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    dependencies = list(data.get("project", {}).get("dependencies", []))
    optional_dependencies = dict(data.get("project", {}).get("optional-dependencies", {}))
    forbidden_extra_names = sorted(
        extra_name
        for extra_name in optional_dependencies
        if str(extra_name).strip().lower().replace("_", "-") in FORBIDDEN_OPTIONAL_EXTRA_NAMES
    )
    offenders: list[str] = []
    for dependency in dependencies:
        normalized = str(dependency).strip().lower().replace("_", "-")
        for prefix in FORBIDDEN_PROJECT_DEPENDENCY_PREFIXES:
            if (
                normalized == prefix
                or normalized.startswith(prefix + ">")
                or normalized.startswith(prefix + "=")
                or normalized.startswith(prefix + "[")
            ):
                offenders.append(str(dependency))
                break
    optional_hits: dict[str, list[str]] = {prefix: [] for prefix in REQUIRED_OPTIONAL_DEPENDENCY_PREFIXES}
    forbidden_optional_hits: dict[str, list[str]] = {
        extra_name: [] for extra_name in FORBIDDEN_OPTIONAL_EXTRA_DEPENDENCY_PREFIXES
    }
    for extra_name, extra_dependencies in optional_dependencies.items():
        for dependency in extra_dependencies:
            normalized = str(dependency).strip().lower().replace("_", "-")
            for prefix in REQUIRED_OPTIONAL_DEPENDENCY_PREFIXES:
                if (
                    normalized == prefix
                    or normalized.startswith(prefix + ">")
                    or normalized.startswith(prefix + "=")
                    or normalized.startswith(prefix + "[")
                ):
                    optional_hits[prefix].append(str(extra_name))
            for prefix in FORBIDDEN_OPTIONAL_EXTRA_DEPENDENCY_PREFIXES.get(str(extra_name), ()):
                if (
                    normalized == prefix
                    or normalized.startswith(prefix + ">")
                    or normalized.startswith(prefix + "=")
                    or normalized.startswith(prefix + "[")
                ):
                    forbidden_optional_hits.setdefault(str(extra_name), []).append(str(dependency))
    missing_optional = [
        prefix for prefix, extra_names in optional_hits.items() if not extra_names
    ]
    forbidden_optional_hits = {
        extra_name: hits for extra_name, hits in forbidden_optional_hits.items() if hits
    }
    return {
        "name": "pyproject_default_dependencies",
        "path": str(pyproject),
        "passed": not offenders and not missing_optional and not forbidden_optional_hits and not forbidden_extra_names,
        "offenders": offenders,
        "forbidden_default_dependency_prefixes": list(FORBIDDEN_PROJECT_DEPENDENCY_PREFIXES),
        "forbidden_optional_extra_names": list(FORBIDDEN_OPTIONAL_EXTRA_NAMES),
        "forbidden_optional_extra_name_hits": forbidden_extra_names,
        "forbidden_optional_extra_dependency_prefixes": {
            extra_name: list(prefixes)
            for extra_name, prefixes in FORBIDDEN_OPTIONAL_EXTRA_DEPENDENCY_PREFIXES.items()
        },
        "forbidden_optional_extra_hits": forbidden_optional_hits,
        "optional_dependency_hits": optional_hits,
        "missing_optional_dependency_prefixes": missing_optional,
    }


def requirements_dependency_report(
    repo_root: Path,
    filename: str = "requirements.txt",
    name: str = "requirements_default_dependencies",
    forbidden_prefixes: tuple[str, ...] = FORBIDDEN_REQUIREMENTS_DEPENDENCY_PREFIXES,
    require_empty: bool = False,
) -> dict[str, object]:
    requirements = repo_root / filename
    dependencies: list[str] = []
    offenders: list[str] = []
    if requirements.exists():
        for raw_line in requirements.read_text(encoding="utf-8").splitlines():
            dependency = raw_line.split("#", 1)[0].strip()
            if not dependency or dependency.startswith(("-", "--")):
                continue
            dependencies.append(dependency)
            normalized = dependency.lower().replace("_", "-")
            for prefix in forbidden_prefixes:
                if (
                    normalized == prefix
                    or normalized.startswith(prefix + ">")
                    or normalized.startswith(prefix + "=")
                    or normalized.startswith(prefix + "[")
                    or normalized.startswith(prefix + "~")
                    or normalized.startswith(prefix + "!")
                ):
                    offenders.append(raw_line.strip())
                    break
    unexpected_default_dependencies = dependencies if require_empty else []
    return {
        "name": name,
        "path": str(requirements),
        "exists": requirements.exists(),
        "passed": not offenders and not unexpected_default_dependencies,
        "dependencies": dependencies,
        "dependency_count": len(dependencies),
        "require_empty": bool(require_empty),
        "unexpected_default_dependencies": unexpected_default_dependencies,
        "offenders": offenders,
        "forbidden_dependency_prefixes": list(forbidden_prefixes),
    }


def egg_info_dependency_report(repo_root: Path) -> dict[str, object]:
    egg_info = repo_root / "neuralfn.egg-info"
    metadata_files = [egg_info / "PKG-INFO", egg_info / "requires.txt"]
    offenders: list[str] = []
    forbidden_extra_hits: list[str] = []
    for path in metadata_files:
        if not path.exists():
            continue
        current_extra = ""
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current_extra = line[1:-1].strip().lower().replace("_", "-")
                if current_extra in FORBIDDEN_OPTIONAL_EXTRA_NAMES:
                    forbidden_extra_hits.append(f"{path.name}:{line}")
                continue
            candidate = line
            dependency_extra = current_extra
            if line.startswith("Requires-Dist:"):
                candidate = line.split(":", 1)[1].strip()
                marker = "; extra =="
                if marker in candidate:
                    extra_name = (
                        candidate.split(marker, 1)[1]
                        .strip()
                        .strip("'\"")
                        .lower()
                        .replace("_", "-")
                    )
                    dependency_extra = extra_name
                    if extra_name in FORBIDDEN_OPTIONAL_EXTRA_NAMES:
                        forbidden_extra_hits.append(f"{path.name}:{line}")
            normalized = candidate.lower().replace("_", "-")
            forbidden_prefixes = (
                FORBIDDEN_OPTIONAL_EXTRA_DEPENDENCY_PREFIXES.get(dependency_extra, ())
                if dependency_extra
                else FORBIDDEN_REQUIREMENTS_DEPENDENCY_PREFIXES
            )
            for prefix in forbidden_prefixes:
                if (
                    normalized == prefix
                    or normalized.startswith(prefix + ">")
                    or normalized.startswith(prefix + "=")
                    or normalized.startswith(prefix + "[")
                    or normalized.startswith(prefix + "~")
                    or normalized.startswith(prefix + "!")
                ):
                    offenders.append(f"{path.name}:{line}")
                    break
    return {
        "name": "egg_info_dependencies",
        "path": str(egg_info),
        "exists": egg_info.exists(),
        "passed": not offenders and not forbidden_extra_hits,
        "offenders": offenders,
        "forbidden_optional_extra_hits": forbidden_extra_hits,
        "forbidden_dependency_prefixes": list(FORBIDDEN_REQUIREMENTS_DEPENDENCY_PREFIXES),
        "forbidden_optional_extra_names": list(FORBIDDEN_OPTIONAL_EXTRA_NAMES),
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    artifact_report: list[dict[str, object]] = []
    failed = False
    if not args.skip_artifacts:
        artifacts = list(args.artifacts) if args.artifacts else default_artifacts()
        for artifact in artifacts:
            path = artifact.expanduser()
            required_artifact = artifact_is_required(path, repo_root)
            stale_sources = (
                [] if args.skip_stale_artifacts else stale_artifact_sources(path, repo_root)
            )
            rebuild_report: dict[str, object] | None = None
            if artifact_should_rebuild(path, stale_sources, bool(args.rebuild_stale)):
                rebuild_report = rebuild_stale_artifact(path, repo_root)
                if (
                    bool(rebuild_report.get("available"))
                    and rebuild_report.get("returncode") == 0
                ):
                    stale_sources = stale_artifact_sources(path, repo_root)
            entry: dict[str, object] = {
                "artifact": str(path),
                "required": required_artifact,
                "exists": path.exists(),
                "forbidden": [],
                "required_string_markers": [
                    marker.decode("utf-8", errors="replace")
                    for marker in required_artifact_string_markers(path, repo_root)
                ],
                "missing_string_markers": [],
                "source_dependencies": [
                    str(source) for source in artifact_source_dependencies(path, repo_root)
                ],
                "stale_sources": stale_sources,
            }
            if rebuild_report is not None:
                entry["rebuild"] = rebuild_report
            if not path.exists():
                entry["error"] = "missing"
                failed = True
                artifact_report.append(entry)
                continue
            output = ldd_output(path)
            forbidden = [
                line.strip()
                for line in output.splitlines()
                if any(marker in line for marker in FORBIDDEN_LIBRARY_MARKERS)
            ]
            entry["forbidden"] = forbidden
            if forbidden:
                failed = True
            missing_markers = missing_artifact_string_markers(path, repo_root)
            entry["missing_string_markers"] = missing_markers
            if missing_markers:
                failed = True
            if stale_sources and required_artifact:
                failed = True
            artifact_report.append(entry)

    python_report: list[dict[str, object]] = []
    shell_report: list[dict[str, object]] = []
    catalog_report: dict[str, object] | None = None
    dependency_report: dict[str, object] | None = None
    requirements_report: dict[str, object] | None = None
    requirements_full_report: dict[str, object] | None = None
    egg_info_report: dict[str, object] | None = None
    if not args.skip_python_entrypoints:
        dependency_report = project_dependency_report(repo_root)
        failed = failed or not bool(dependency_report["passed"])
        requirements_report = requirements_dependency_report(repo_root, require_empty=True)
        failed = failed or not bool(requirements_report["passed"])
        requirements_full_report = requirements_dependency_report(
            repo_root,
            filename="requirements-full.txt",
            name="requirements_full_dependencies",
            forbidden_prefixes=("torch", "torchvision", "torchaudio"),
        )
        failed = failed or not bool(requirements_full_report["passed"])
        egg_info_report = egg_info_dependency_report(repo_root)
        failed = failed or not bool(egg_info_report["passed"])
        python_report = python_entrypoint_report(
            repo_root,
            max_entrypoint_seconds=float(args.max_entrypoint_seconds),
        )
        failed = failed or any(not bool(entry["passed"]) for entry in python_report)
        shell_report = shell_entrypoint_report(
            repo_root,
            max_entrypoint_seconds=float(args.max_entrypoint_seconds),
        )
        failed = failed or any(not bool(entry["passed"]) for entry in shell_report)
        catalog_report = native_template_catalog_report(shell_report)
        failed = failed or not bool(catalog_report["passed"])

    if args.json:
        summary = {
            "artifacts": artifact_summary_counts(artifact_report),
            "python_entrypoints": _summary_counts(python_report),
            "shell_entrypoints": _summary_counts(shell_report),
            "native_template_catalogs": native_template_catalog_summary_counts(catalog_report),
        }
        print(
            json.dumps(
                {
                    "passed": not failed,
                    "summary": summary,
                    "forbidden_python_import_roots": list(FORBIDDEN_PYTHON_IMPORT_ROOTS),
                    "artifacts": artifact_report,
                    "project_dependencies": dependency_report,
                    "requirements_default_dependencies": requirements_report,
                    "requirements_full_dependencies": requirements_full_report,
                    "egg_info_dependencies": egg_info_report,
                    "python_entrypoints": python_report,
                    "shell_entrypoints": shell_report,
                    "native_template_catalogs": catalog_report,
                },
                indent=2,
            )
        )
    else:
        for entry in artifact_report:
            if not entry["exists"]:
                print(f"{entry['artifact']}: missing", file=sys.stderr)
                rebuild = entry.get("rebuild")
                if isinstance(rebuild, dict):
                    command = " ".join(str(part) for part in rebuild.get("command", []))
                    print(f"  rebuild attempted: {command or 'unavailable'}", file=sys.stderr)
                    if rebuild.get("returncode") is not None:
                        print(f"  rebuild exit code: {rebuild['returncode']}", file=sys.stderr)
                    if rebuild.get("stderr"):
                        print(str(rebuild["stderr"]), file=sys.stderr)
                continue
            forbidden = entry["forbidden"]
            if forbidden:
                print(f"{entry['artifact']}: forbidden native dependency detected", file=sys.stderr)
                for line in forbidden:
                    print(f"  {line}", file=sys.stderr)
            elif entry.get("stale_sources"):
                print(f"{entry['artifact']}: stale native artifact", file=sys.stderr)
                rebuild = entry.get("rebuild")
                if isinstance(rebuild, dict):
                    command = " ".join(str(part) for part in rebuild.get("command", []))
                    print(f"  rebuild attempted: {command or 'unavailable'}", file=sys.stderr)
                    if rebuild.get("returncode") is not None:
                        print(f"  rebuild exit code: {rebuild['returncode']}", file=sys.stderr)
                    if rebuild.get("stderr"):
                        print(str(rebuild["stderr"]), file=sys.stderr)
                for source in entry["stale_sources"]:
                    print(f"  source newer than artifact: {source['source']}", file=sys.stderr)
            elif entry.get("missing_string_markers"):
                print(f"{entry['artifact']}: missing native runtime contract markers", file=sys.stderr)
                for marker in entry["missing_string_markers"]:
                    print(f"  missing marker: {marker}", file=sys.stderr)
            else:
                print(f"{entry['artifact']}: ok")
        if dependency_report is not None:
            if dependency_report["passed"]:
                print(f"{dependency_report['name']}: ok")
            else:
                print(f"{dependency_report['name']}: failed", file=sys.stderr)
                for dependency in dependency_report["offenders"]:
                    print(f"  hard dependency: {dependency}", file=sys.stderr)
                for extra_name in dependency_report["forbidden_optional_extra_name_hits"]:
                    print(f"  forbidden optional extra: {extra_name}", file=sys.stderr)
                for prefix in dependency_report["missing_optional_dependency_prefixes"]:
                    print(f"  missing optional dependency coverage: {prefix}", file=sys.stderr)
        if requirements_report is not None:
            if requirements_report["passed"]:
                print(f"{requirements_report['name']}: ok")
            else:
                print(f"{requirements_report['name']}: failed", file=sys.stderr)
                for dependency in requirements_report["offenders"]:
                    print(f"  requirement dependency: {dependency}", file=sys.stderr)
        if requirements_full_report is not None:
            if requirements_full_report["passed"]:
                print(f"{requirements_full_report['name']}: ok")
            else:
                print(f"{requirements_full_report['name']}: failed", file=sys.stderr)
                for dependency in requirements_full_report["offenders"]:
                    print(f"  requirement dependency: {dependency}", file=sys.stderr)
        if egg_info_report is not None:
            if egg_info_report["passed"]:
                print(f"{egg_info_report['name']}: ok")
            else:
                print(f"{egg_info_report['name']}: failed", file=sys.stderr)
                for dependency in egg_info_report["offenders"]:
                    print(f"  egg-info dependency: {dependency}", file=sys.stderr)
                for extra_name in egg_info_report["forbidden_optional_extra_hits"]:
                    print(f"  forbidden egg-info optional extra: {extra_name}", file=sys.stderr)
        for entry in python_report:
            if entry["passed"]:
                print(f"{entry['name']}: ok ({float(entry['elapsed_seconds']):.3f}s)")
            else:
                print(f"{entry['name']}: failed", file=sys.stderr)
                if not entry.get("startup_within_budget", True):
                    print(
                        "startup budget exceeded: "
                        f"{float(entry['elapsed_seconds']):.3f}s > "
                        f"{float(entry['startup_budget_seconds']):.3f}s",
                        file=sys.stderr,
                    )
                if entry["stderr"]:
                    print(str(entry["stderr"]), file=sys.stderr)
        for entry in shell_report:
            if entry["passed"]:
                print(f"{entry['name']}: ok ({float(entry['elapsed_seconds']):.3f}s)")
            else:
                print(f"{entry['name']}: failed", file=sys.stderr)
                if not entry.get("startup_within_budget", True):
                    print(
                        "startup budget exceeded: "
                        f"{float(entry['elapsed_seconds']):.3f}s > "
                        f"{float(entry['startup_budget_seconds']):.3f}s",
                        file=sys.stderr,
                    )
                if entry["stderr"]:
                    print(str(entry["stderr"]), file=sys.stderr)
        if catalog_report is not None:
            if catalog_report["passed"]:
                print("native_template_catalogs: ok")
            else:
                print("native_template_catalogs: failed", file=sys.stderr)
                for report in catalog_report.get("entrypoints", []):
                    if not isinstance(report, dict) or report.get("passed"):
                        continue
                    print(f"  {report.get('entrypoint')}: failed", file=sys.stderr)
                    for error in report.get("errors", []):
                        print(f"    {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
