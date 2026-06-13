"""Generic dense GPT native training helpers.

The current implementation delegates to the GPT-2-compatible native trainer
module because dense GPT/GPT-2/GPT-3 decoder training uses the same kernel
family; the selected template or custom graph describes the architecture.
"""

from .native_gpt2 import (
    NativeGpt2CheckpointInfo as NativeGptCheckpointInfo,
    NativeGpt2RunConfig as NativeGptRunConfig,
    NativeGpt2RunnerStatus as NativeGptRunnerStatus,
    build_native_gpt2_compiled_cli_run_config,
    build_native_gpt2_run_config,
    is_native_gpt2_checkpoint as is_native_gpt_checkpoint,
    latest_native_gpt2_checkpoint as latest_native_gpt_checkpoint,
    native_gpt2_parameter_count as native_gpt_parameter_count,
    native_gpt2_runner_status as native_gpt_runner_status,
    read_native_gpt2_checkpoint_info as read_native_gpt_checkpoint_info,
    resolve_native_gpt2_cli as resolve_native_gpt_cli,
    resolve_native_gpt2_executable as resolve_native_gpt_executable,
    resolve_native_gpt2_launcher as resolve_native_gpt_launcher,
    resolve_native_gpt2_token_shards as resolve_native_gpt_token_shards,
    run_native_gpt2 as run_native_gpt,
    write_native_gpt2_run_config as write_native_gpt_run_config,
)


def build_native_gpt_run_config(**kwargs):
    kwargs.setdefault("model_family", "gpt")
    return build_native_gpt2_run_config(**kwargs)


def build_native_gpt_compiled_cli_run_config(**kwargs):
    kwargs.setdefault("model_family", "gpt")
    return build_native_gpt2_compiled_cli_run_config(**kwargs)

__all__ = [
    "NativeGptCheckpointInfo",
    "NativeGptRunConfig",
    "NativeGptRunnerStatus",
    "build_native_gpt_compiled_cli_run_config",
    "build_native_gpt_run_config",
    "is_native_gpt_checkpoint",
    "latest_native_gpt_checkpoint",
    "native_gpt_parameter_count",
    "native_gpt_runner_status",
    "read_native_gpt_checkpoint_info",
    "resolve_native_gpt_cli",
    "resolve_native_gpt_executable",
    "resolve_native_gpt_launcher",
    "resolve_native_gpt_token_shards",
    "run_native_gpt",
    "write_native_gpt_run_config",
]
