"""NeuralFn — a brain-inspired function-neuron graph framework."""

from .neuron import NeuronDef, module_neuron, neuron, neuron_from_source, subgraph_neuron
from .port import Port

_LAZY_EXPORTS = {
    "BuiltinNeurons": ("builtins", "BuiltinNeurons"),
    "Edge": ("graph", "Edge"),
    "NeuronInstance": ("graph", "NeuronInstance"),
    "NeuronGraph": ("graph", "NeuronGraph"),
    "SurrogateModel": ("surrogate", "SurrogateModel"),
    "probe_neuron": ("surrogate", "probe_neuron"),
    "build_surrogates": ("surrogate", "build_surrogates"),
    "SurrogateTrainer": ("trainer", "SurrogateTrainer"),
    "EvolutionaryTrainer": ("evolutionary", "EvolutionaryTrainer"),
    "HybridConfig": ("hybrid", "HybridConfig"),
    "HybridTrainer": ("hybrid", "HybridTrainer"),
    "save_graph": ("serialization", "save_graph"),
    "load_graph": ("serialization", "load_graph"),
    "SHIPPED_GPT_TEMPLATE_BASE_PRESETS": ("config", "SHIPPED_GPT_TEMPLATE_BASE_PRESETS"),
    "SHIPPED_GPT_TEMPLATE_PRESETS": ("config", "SHIPPED_GPT_TEMPLATE_PRESETS"),
    "TorchTrainConfig": ("torch_backend", "TorchTrainConfig"),
    "TorchTrainer": ("torch_backend", "TorchTrainer"),
    "build_gpt_root_graph": ("torch_templates", "build_gpt_root_graph"),
    "build_model_stage_graph": ("torch_templates", "build_model_stage_graph"),
    "NativeGptRunConfig": ("native_gpt", "NativeGptRunConfig"),
    "NativeGptCheckpointInfo": ("native_gpt", "NativeGptCheckpointInfo"),
    "NativeGptRunnerStatus": ("native_gpt", "NativeGptRunnerStatus"),
    "NativeGpt2RunConfig": ("native_gpt2", "NativeGpt2RunConfig"),
    "NativeGpt2CheckpointInfo": ("native_gpt2", "NativeGpt2CheckpointInfo"),
    "NativeGpt2RunnerStatus": ("native_gpt2", "NativeGpt2RunnerStatus"),
    "NativeTrainRunConfig": ("native_train", "NativeTrainRunConfig"),
    "NativeTrainRunnerStatus": ("native_train", "NativeTrainRunnerStatus"),
    "build_native_train_run_config": ("native_train", "build_native_train_run_config"),
    "build_native_gpt_compiled_cli_run_config": ("native_gpt", "build_native_gpt_compiled_cli_run_config"),
    "build_native_gpt_run_config": ("native_gpt", "build_native_gpt_run_config"),
    "build_native_gpt2_compiled_cli_run_config": ("native_gpt2", "build_native_gpt2_compiled_cli_run_config"),
    "build_native_gpt2_run_config": ("native_gpt2", "build_native_gpt2_run_config"),
    "exec_native_gpt": ("native_gpt", "exec_native_gpt"),
    "exec_native_gpt2": ("native_gpt2", "exec_native_gpt2"),
    "exec_native_train": ("native_train", "exec_native_train"),
    "is_native_gpt_checkpoint": ("native_gpt", "is_native_gpt_checkpoint"),
    "is_native_gpt2_checkpoint": ("native_gpt2", "is_native_gpt2_checkpoint"),
    "latest_native_gpt_checkpoint": ("native_gpt", "latest_native_gpt_checkpoint"),
    "latest_native_gpt2_checkpoint": ("native_gpt2", "latest_native_gpt2_checkpoint"),
    "native_gpt_activation": ("native_gpt", "native_gpt_activation"),
    "native_gpt_encoding_vocab_size": ("native_gpt", "native_gpt_encoding_vocab_size"),
    "native_gpt_kernel_backend": ("native_gpt", "native_gpt_kernel_backend"),
    "native_gpt_parameter_count": ("native_gpt", "native_gpt_parameter_count"),
    "native_gpt2_parameter_count": ("native_gpt2", "native_gpt2_parameter_count"),
    "native_gpt_runner_status": ("native_gpt", "native_gpt_runner_status"),
    "native_gpt2_runner_status": ("native_gpt2", "native_gpt2_runner_status"),
    "native_train_model_registry": ("native_train", "native_train_model_registry"),
    "native_train_runner_status": ("native_train", "native_train_runner_status"),
    "read_native_gpt_checkpoint_info": ("native_gpt", "read_native_gpt_checkpoint_info"),
    "read_native_gpt2_checkpoint_info": ("native_gpt2", "read_native_gpt2_checkpoint_info"),
    "normalize_native_gpt_encoding_name": ("native_gpt", "normalize_native_gpt_encoding_name"),
    "resolve_native_gpt_cli": ("native_gpt", "resolve_native_gpt_cli"),
    "resolve_native_gpt2_cli": ("native_gpt2", "resolve_native_gpt2_cli"),
    "resolve_native_gpt_executable": ("native_gpt", "resolve_native_gpt_executable"),
    "resolve_native_gpt2_executable": ("native_gpt2", "resolve_native_gpt2_executable"),
    "resolve_native_gpt_launcher": ("native_gpt", "resolve_native_gpt_launcher"),
    "resolve_native_gpt2_launcher": ("native_gpt2", "resolve_native_gpt2_launcher"),
    "resolve_native_gpt_token_shards": ("native_gpt", "resolve_native_gpt_token_shards"),
    "resolve_native_gpt2_token_shards": ("native_gpt2", "resolve_native_gpt2_token_shards"),
    "resolve_native_gpt_binding_command": ("native_gpt", "resolve_native_gpt_binding_command"),
    "resolve_native_gpt2_binding_command": ("native_gpt2", "resolve_native_gpt2_binding_command"),
    "resolve_native_train_binding_command": ("native_train", "resolve_native_train_binding_command"),
    "resolve_native_train_cli": ("native_train", "resolve_native_train_cli"),
    "run_native_gpt": ("native_gpt", "run_native_gpt"),
    "run_native_gpt2": ("native_gpt2", "run_native_gpt2"),
    "run_native_train": ("native_train", "run_native_train"),
    "write_native_gpt_run_config": ("native_gpt", "write_native_gpt_run_config"),
    "write_native_gpt2_run_config": ("native_gpt2", "write_native_gpt2_run_config"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib

        module_name, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(f".{module_name}", __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'neuralfn' has no attribute {name!r}")

__all__ = [
    "Port",
    "NeuronDef",
    "module_neuron",
    "neuron",
    "neuron_from_source",
    "subgraph_neuron",
    *_LAZY_EXPORTS,
]
