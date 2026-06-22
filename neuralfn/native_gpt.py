"""Generic dense GPT native training helpers.

Dense GPT, GPT-2, and GPT-3 decoder training use the same native CUDA Tile
kernel family; the selected template or custom graph describes the architecture.
The GPT-2 compatibility module still owns the checkpoint-layout details, while
this module exposes GPT-native class names and defaults for new SDK code.
"""

from dataclasses import asdict

from .native_gpt2 import (
    NativeGpt2CheckpointInfo as _NativeGpt2CheckpointInfo,
    NativeGpt2RunConfig as _NativeGpt2RunConfig,
    NativeGpt2RunnerStatus as _NativeGpt2RunnerStatus,
    build_native_gpt2_compiled_cli_run_config,
    build_native_gpt2_run_config,
    is_native_gpt2_checkpoint as is_native_gpt_checkpoint,
    latest_native_gpt2_checkpoint as latest_native_gpt_checkpoint,
    exec_native_gpt2,
    native_gpt2_activation as native_gpt_activation,
    native_gpt2_encoding_vocab_size as native_gpt_encoding_vocab_size,
    native_gpt2_kernel_backend as native_gpt_kernel_backend,
    native_gpt2_parameter_count as native_gpt_parameter_count,
    native_gpt2_runner_status,
    normalize_native_gpt2_encoding_name as normalize_native_gpt_encoding_name,
    read_native_gpt2_checkpoint_info,
    resolve_native_gpt2_cli as resolve_native_gpt_cli,
    resolve_native_gpt2_binding_command,
    resolve_native_gpt2_executable as resolve_native_gpt_executable,
    resolve_native_gpt2_launcher as resolve_native_gpt_launcher,
    resolve_native_gpt2_token_shards as resolve_native_gpt_token_shards,
    run_native_gpt2,
    write_native_gpt2_run_config,
)


class NativeGptRunnerStatus(_NativeGpt2RunnerStatus):
    """Resolved launch mode for the generic native GPT runner."""


class NativeGptRunConfig(_NativeGpt2RunConfig):
    """Configuration for the generic native CUDA Tile dense GPT trainer."""


class NativeGptCheckpointInfo(_NativeGpt2CheckpointInfo):
    """Metadata read from a NeuralFn native dense GPT checkpoint."""


def _generic_config(config: _NativeGpt2RunConfig) -> NativeGptRunConfig:
    return NativeGptRunConfig(**asdict(config))


def _generic_status(status: _NativeGpt2RunnerStatus) -> NativeGptRunnerStatus:
    return NativeGptRunnerStatus(**asdict(status))


def _generic_checkpoint(info: _NativeGpt2CheckpointInfo) -> NativeGptCheckpointInfo:
    return NativeGptCheckpointInfo(**asdict(info))


def build_native_gpt_run_config(**kwargs):
    kwargs.setdefault("model_family", "gpt")
    config, metadata = build_native_gpt2_run_config(**kwargs)
    return _generic_config(config), metadata


def build_native_gpt_compiled_cli_run_config(**kwargs):
    kwargs.setdefault("model_family", "gpt")
    return _generic_config(build_native_gpt2_compiled_cli_run_config(**kwargs))


def native_gpt_runner_status(requested: str = "auto") -> NativeGptRunnerStatus:
    return _generic_status(native_gpt2_runner_status(requested))


def resolve_native_gpt_binding_command(config: NativeGptRunConfig) -> list[str]:
    return resolve_native_gpt2_binding_command(config)


def read_native_gpt_checkpoint_info(*args, **kwargs) -> NativeGptCheckpointInfo:
    return _generic_checkpoint(read_native_gpt2_checkpoint_info(*args, **kwargs))


def run_native_gpt(config: NativeGptRunConfig, *, runner: str = "auto") -> int:
    return run_native_gpt2(config, runner=runner)


def exec_native_gpt(config: NativeGptRunConfig, *, runner: str = "compiled-cli") -> int:
    return exec_native_gpt2(config, runner=runner)


def write_native_gpt_run_config(config: NativeGptRunConfig, path, *, runner: str = "auto"):
    return write_native_gpt2_run_config(config, path, runner=runner)

__all__ = [
    "NativeGptCheckpointInfo",
    "NativeGptRunConfig",
    "NativeGptRunnerStatus",
    "build_native_gpt_compiled_cli_run_config",
    "build_native_gpt_run_config",
    "exec_native_gpt",
    "is_native_gpt_checkpoint",
    "latest_native_gpt_checkpoint",
    "native_gpt_activation",
    "native_gpt_encoding_vocab_size",
    "native_gpt_kernel_backend",
    "native_gpt_parameter_count",
    "native_gpt_runner_status",
    "normalize_native_gpt_encoding_name",
    "read_native_gpt_checkpoint_info",
    "resolve_native_gpt_binding_command",
    "resolve_native_gpt_cli",
    "resolve_native_gpt_executable",
    "resolve_native_gpt_launcher",
    "resolve_native_gpt_token_shards",
    "run_native_gpt",
    "write_native_gpt_run_config",
]
