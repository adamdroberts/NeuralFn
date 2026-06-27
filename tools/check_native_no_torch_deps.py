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
)
OPTIONAL_DEFAULT_ARTIFACTS = (
    Path("build/nfn_native_train"),
    Path("build/nfn_train_gpt"),
    Path("build/nfn_train_gpt_sm120"),
    Path("build/nfn_gpt_native_train_linked"),
    Path("build/nfn_gpt2_native_train"),
    Path("build/nfn_gpt2_evo_native_train"),
    Path("build/nfn_nanogpt_native_train"),
    Path("build/nfn_llama_native_train"),
    Path("build/nfn_mixllama_native_train"),
    Path("build/nfn_jepa_native_train"),
    Path("build/nfn_semantic_router_moe_native_train"),
    Path("build/nfn_deepseek_v4_native_train"),
    Path("build/linear_backward_bench"),
    Path("build/lm_head_backward_bench"),
    Path("build/libnfn_native_train_tile_ops_tk.so"),
)
OPTIONAL_DEFAULT_ARTIFACT_GLOBS = (
    "neuralfn/_native*.so",
)
ARTIFACT_SOURCE_DEPENDENCIES = {
    Path("build/nfn_gpt_native_train"): (
        Path("neuralfn/csrc/native_gpt2/nfn_gpt2_native_train.cpp"),
        Path("neuralfn/csrc/native_train/token_shards.cpp"),
        Path("neuralfn/csrc/native_train/token_shards.h"),
        Path("neuralfn/csrc/native_train/shipped_gpt_template_presets.h"),
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
    ),
    Path("build/nfn_train_gpt_sm120"): (
        Path("neuralfn/csrc/native_train/train_gpt_sm120.cpp"),
        Path("tools/build_train_gpt_sm120_cli.sh"),
    ),
    Path("build/nfn_train_gpt"): (
        Path("neuralfn/csrc/native_train/train_gpt_sm120.cpp"),
        Path("tools/build_train_gpt_cli.sh"),
    ),
    Path("build/nfn_gpt2_native_train"): (
        Path("neuralfn/csrc/native_gpt2/nfn_gpt2_native_train.cpp"),
        Path("neuralfn/csrc/native_train/token_shards.cpp"),
        Path("neuralfn/csrc/native_train/token_shards.h"),
        Path("neuralfn/csrc/native_train/shipped_gpt_template_presets.h"),
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
            "import neuralfn; import neuralfn.native_gpt; import neuralfn.native_gpt2; import neuralfn.native_train; print('native-sdk-ok')",
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
                    "from neuralfn import NativeGptRunConfig, NativeGpt2RunConfig, NativeTrainRunConfig",
                    "from neuralfn import build_native_gpt_compiled_cli_run_config",
                    "from neuralfn import exec_native_gpt, exec_native_gpt2, exec_native_train",
                    "from neuralfn import build_native_sm120_gpt_run_config, build_native_train_run_config, run_native_train",
                    "from neuralfn import native_gpt_checkpoint_sampler_argv, native_gpt_checkpoint_sampler_env",
                    "from neuralfn import native_gpt_kernel_backend, native_gpt_parameter_count",
                    "from neuralfn import native_gpt_prompt_tokens, render_native_gpt_checkpoint_sampler_text",
                    "from neuralfn import run_native_gpt_checkpoint_sampler",
                    "from neuralfn import native_train_model_registry, native_train_runner_status",
                    "from neuralfn import resolve_native_gpt_binding_command, resolve_native_gpt2_binding_command",
                    "from neuralfn import resolve_native_sm120_train_cli, resolve_native_train_binding_command",
                    "assert NativeGptRunConfig.__name__ == 'NativeGptRunConfig'",
                    "assert NativeGpt2RunConfig.__name__ == 'NativeGpt2RunConfig'",
                    "assert NativeTrainRunConfig.__name__ == 'NativeTrainRunConfig'",
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
                    "assert callable(run_native_gpt_checkpoint_sampler)",
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
                    "print('native-sdk-public-exports-ok')",
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
        "train_gpt_sm120_dry_run",
        (
            "bash",
            "tools/train_gpt_sm120.sh",
            "--print-command",
            "--dry-run",
            "--no-checkpoint",
        ),
        {},
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
        "train_gpt_sm120_compiled_dry_run",
        (
            "build/nfn_train_gpt_sm120",
            "--print-command",
            "--dry-run",
            "--no-checkpoint",
        ),
        {},
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
)
NATIVE_GPT_CHECKPOINT_MAGIC = 20240326
NATIVE_GPT_CHECKPOINT_HEADER_INTS = 256
DEFAULT_MAX_ENTRYPOINT_SECONDS = 2.0


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
    artifacts.extend(path for path in OPTIONAL_DEFAULT_ARTIFACTS if path.exists())
    for pattern in OPTIONAL_DEFAULT_ARTIFACT_GLOBS:
        artifacts.extend(sorted(Path().glob(pattern)))
    return artifacts


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


def ldd_output(path: Path) -> str:
    proc = subprocess.run(
        ["ldd", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.stdout


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
        native_checkpoint = _write_native_checkpoint_stub(temp_root)
        _write_latest_native_checkpoint_stub(temp_root)
        native_graph = _write_native_graph_stub(temp_root)
        env = os.environ.copy()
        env["NFN_NATIVE_GPT_CLI"] = str(native_cli)
        env["NFN_NATIVE_GPT2_CLI"] = str(native_cli)
        env["NFN_NATIVE_GPT2_EVO_CLI"] = str(native_cli)
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
            entries.append(
                {
                    "name": name,
                    "command": list(command),
                    "returncode": proc.returncode,
                    "passed": proc.returncode == 0 and startup_within_budget,
                    "elapsed_seconds": elapsed_seconds,
                    "startup_budget_seconds": startup_budget_seconds,
                    "startup_within_budget": startup_within_budget,
                    "stdout": proc.stdout.strip(),
                    "stderr": proc.stderr.strip(),
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
    }
    with tempfile.TemporaryDirectory(prefix="nfn-native-shell-entrypoints-") as tmp:
        temp_root = Path(tmp)
        for name, command, extra_env in DEFAULT_SHELL_ENTRYPOINTS:
            env = base_env.copy()
            env.update(extra_env)
            run_command = list(command)
            build_started: float | None = None
            build_elapsed = 0.0
            first = run_command[0] if run_command else ""
            builder = launcher_builders.get(first)
            if builder is not None and not (repo_root / first).exists():
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


def requirements_dependency_report(repo_root: Path) -> dict[str, object]:
    requirements = repo_root / "requirements.txt"
    offenders: list[str] = []
    if requirements.exists():
        for raw_line in requirements.read_text(encoding="utf-8").splitlines():
            dependency = raw_line.split("#", 1)[0].strip()
            if not dependency or dependency.startswith(("-", "--")):
                continue
            normalized = dependency.lower().replace("_", "-")
            for prefix in FORBIDDEN_REQUIREMENTS_DEPENDENCY_PREFIXES:
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
    return {
        "name": "requirements_default_dependencies",
        "path": str(requirements),
        "exists": requirements.exists(),
        "passed": not offenders,
        "offenders": offenders,
        "forbidden_dependency_prefixes": list(FORBIDDEN_REQUIREMENTS_DEPENDENCY_PREFIXES),
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
            stale_sources = (
                [] if args.skip_stale_artifacts else stale_artifact_sources(path, repo_root)
            )
            rebuild_report: dict[str, object] | None = None
            if stale_sources and args.rebuild_stale:
                rebuild_report = rebuild_stale_artifact(path, repo_root)
                if (
                    bool(rebuild_report.get("available"))
                    and rebuild_report.get("returncode") == 0
                ):
                    stale_sources = stale_artifact_sources(path, repo_root)
            entry: dict[str, object] = {
                "artifact": str(path),
                "exists": path.exists(),
                "forbidden": [],
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
            if stale_sources:
                failed = True
            artifact_report.append(entry)

    python_report: list[dict[str, object]] = []
    shell_report: list[dict[str, object]] = []
    dependency_report: dict[str, object] | None = None
    requirements_report: dict[str, object] | None = None
    egg_info_report: dict[str, object] | None = None
    if not args.skip_python_entrypoints:
        dependency_report = project_dependency_report(repo_root)
        failed = failed or not bool(dependency_report["passed"])
        requirements_report = requirements_dependency_report(repo_root)
        failed = failed or not bool(requirements_report["passed"])
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

    if args.json:
        print(
            json.dumps(
                {
                    "passed": not failed,
                    "forbidden_python_import_roots": list(FORBIDDEN_PYTHON_IMPORT_ROOTS),
                    "artifacts": artifact_report,
                    "project_dependencies": dependency_report,
                    "requirements_dependencies": requirements_report,
                    "egg_info_dependencies": egg_info_report,
                    "python_entrypoints": python_report,
                    "shell_entrypoints": shell_report,
                },
                indent=2,
            )
        )
    else:
        for entry in artifact_report:
            if not entry["exists"]:
                print(f"{entry['artifact']}: missing", file=sys.stderr)
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
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
