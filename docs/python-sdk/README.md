# NeuralFn Python SDK Reference

API reference for the `neuralfn` Python package.

## Installation

```bash
pip install -e .
```

The default SDK install does not require Torch. Install graph-backed training
extras explicitly when needed:

```bash
pip install -e ".[torch]"
```

From a sibling project checked out next to the repo:

```bash
pip install -e ../NeuralFn
```

This editable install now packages the shipped semantic vocabulary files under
`neuralfn/data/semantic/`, so SDK code that imports `neuralfn.semantic` can
load the `vocab_86d_*.json` semantic vocabularies without extra setup.

## Package Exports

All public symbols are available from the top-level `neuralfn` module:

```python
from neuralfn import (
    Port,
    NeuronDef, module_neuron, neuron, neuron_from_source, subgraph_neuron,
    BuiltinNeurons,
    Edge, NeuronInstance, NeuronGraph,
    SurrogateModel, probe_neuron, build_surrogates,
    SurrogateTrainer,
    EvolutionaryTrainer,
    HybridConfig, HybridTrainer,
    save_graph, load_graph,
    TorchTrainConfig, TorchTrainer,
    build_gpt_root_graph, build_model_stage_graph,
    NativeGpt2CheckpointInfo, NativeGpt2RunConfig, NativeGpt2RunnerStatus,
    NativeTrainRunConfig, NativeTrainRunnerStatus,
    build_native_gpt2_compiled_cli_run_config, build_native_gpt2_run_config, is_native_gpt2_checkpoint,
    build_native_train_run_config,
    latest_native_gpt2_checkpoint, native_gpt2_parameter_count,
    native_gpt2_runner_status, read_native_gpt2_checkpoint_info,
    native_train_model_registry, native_train_runner_status,
    resolve_native_gpt2_cli, resolve_native_gpt2_executable,
    resolve_native_gpt2_launcher, resolve_native_gpt2_token_shards, run_native_gpt2,
    resolve_native_train_cli, run_native_train,
    write_native_gpt2_run_config,
)
```

Top-level SDK exports are loaded lazily. Importing `neuralfn` or
`neuralfn.native_gpt2` no longer imports the Torch backend; Torch is loaded only
when callers access Torch-backed exports such as `TorchTrainer`,
`TorchTrainConfig`, or template graph builders.

For the native GPT-2 path, `bash tools/build_native_gpt2_binding.sh` builds the
`neuralfn._native_gpt2` C++ extension used by `run_native_gpt2(...,
runner="auto")` before falling back to the standalone launcher or subprocess
path. `build_native_gpt2_compiled_cli_run_config()` creates a dense GPT-2
compiled-CLI handoff directly from a dataset alias/path, leaving shard metadata
inspection to the C++ resolver. When that alias-only config is passed through
the C++ binding, the binding executes `compiled_cli_argv` instead of the raw
`train_gpt2cu` argv so SDK `runner="auto"` keeps the no-Python shard resolver
path. Set `kernel_backend="tile-cuda"` plus `tile_ops_lib=...` on the config to
inspect/check or run the NeuralFn-owned raw Tile GPT-2 plan. Set
`template_name="gpt2"` or `template_name="gpt2_megakernel"` for the implemented
dense native loop, or pass another shipped GPT template name/custom `graph_file`
to select that architecture and receive explicit
`selected-graph-native-trainer-missing` JSON until its C++ Tile trainer exists.
The implemented dense loop honors `train_batch_tokens` by deriving
`grad_accum_steps`, averaging that many CUDA Tile microbatch gradients in device
accumulation buffers, and applying clip plus AdamW once per optimizer step.
Set `smoke_tile_ops=True` and optionally
`cuda_runtime_lib=...` to have the compiled GPT-2 CLI launch the
raw `nfn_native_tile_fill_float32` CUDA Tile ABI and verify device-to-host
copyback without importing Torch. Set `smoke_optimizer_step=True` to allocate
GPT-2-sized parameter/gradient/AdamW buffers and run the registered-buffer
AdamW iteration through the raw Tile ABI. Set `smoke_lm_step=True` to run a
tiny full-vocab tied embedding/LM-head forward/backward/update slice through
raw Tile kernels. Set `smoke_attention_step=True` to run a tiny model-dim
attention stage through qkv projection, QKV split, SDPA forward/backward, QKV
gradient merge, projection backward, and AdamW. Set `smoke_mlp_step=True` to
run a tiny GPT-2 MLP stage through c_fc projection, GELU forward/backward,
c_proj projection backward, and AdamW. Set `smoke_norm_residual_step=True` to
run GPT-2 LayerNorm, scaled residual add, LayerNorm affine/input backward,
gradient accumulation, and AdamW through raw Tile kernels. Set
`smoke_embedding_lm_step=True` to sample a tiny cached uint16 token batch in
C++ and run token embedding, absolute position embedding, embedding residual
add, final LayerNorm, tied LM head, CE backward, embedding/norm backward, and
AdamW without graph-editor payloads. Set `train_embedding_lm=True` to run that
same GPT-2 embedding/final-norm/LM path as a real multi-step compiled loop over
cached train shards, with validation losses from validation shards controlled
by `eval_every_steps`, `eval_batches`, and `eval_batch_size`. Set
`smoke_transformer_block_step=True` to compose GPT-2 LayerNorm, fused QKV
attention, residual adds, MLP, backward passes, gradient accumulation, and
AdamW updates through raw Tile kernels. Set `smoke_transformer_lm_step=True` to
sample cached uint16 tokens and run token/position embeddings, one tiny
transformer block, final LayerNorm, tied LM head, CE forward/backward,
transformer backward, embedding backward, and AdamW for 16 parameter buffers
through raw Tile kernels while preserving range-checked GPT-2 token IDs. Set
`train_transformer_lm=True` to run the full-vocab real-dim 12-layer
transformer-LM multi-step compiled loop over cached shards. That path emits
periodic validation records in `validation.losses` when `eval_every_steps` is
positive, clips gradients on device before AdamW, reports vocab and clip
metadata plus the row-chunked LM-head workspace size in the native JSON, stages
cached uint16 token/target batches through pinned host memory, enqueues H2D
copy with `cudaMemcpyAsync`, widens them to int64 on device through one
`nfn_native_tile_uint16_to_int64`, reports `token_id_host_staging: "pinned"` and
`token_batch_staging_strategy: "direct-sampler-to-pinned-arena"` plus
`token_id_h2d_copy: "cudaMemcpyAsync-contiguous-arena"`, initializes the tied token
embedding/LM-head weight on device through
`nfn_native_tile_init_gpt2_token_weight_float32` without host materialization,
uses one scratch activation tape with backward recomputation plus persistent
block outputs, keeps block-0 allocation/initialization/AdamW-state zeroing under
the same block-vector visitors as every other transformer block, reports the
`block0_duplicate_*_elided` startup flags, including activation allocation,
under `block_state_layout`, suballocates float buffers from a single CUDA device
arena with `float_allocation_strategy: "single-arena"`, uses combined token
arenas with `token_buffer_allocation_strategy: "combined-arenas"`, and stays
out of Python/Torch.
Backend names are strict: use
`"llm-kittens"` or `"tile-cuda"`. For the unified native training frontend, `bash
tools/build_native_train_binding.sh` builds `neuralfn._native_train`, which is
used by `run_native_train(..., runner="auto")` to hand off to `nfn_native_train`
without importing Torch. Use `native_train_model_registry()` to inspect the
compiled model coverage exposed by `nfn-native-train --list-models --json`.

## Modules

| Module | Description |
|--------|-------------|
| [port](port.md) | `Port` dataclass -- declares input/output slots on neurons |
| [neuron](neuron.md) | `NeuronDef` and factory functions for defining neurons |
| [graph](graph.md) | `NeuronInstance`, `Edge`, and `NeuronGraph` -- the core graph data model |
| [builtins](builtins.md) | Built-in neuron definitions (scalar activations, torch modules, MoE, semantic routing, etc.) |
| [config](config.md) | `TemplateSpec`, `BlockSpec`, `ModelSpec` and preset builder functions |
| [torch-backend](torch-backend.md) | `CompiledTorchGraph`, `TorchTrainer`, `TorchTrainConfig`, and all `*Stage` modules |
| [tile-cuda](tile-cuda.md) | Optional CUDA Tile backend configuration, diagnostics, kernel coverage registry, and native GPT-2 trainer handoff helpers |
| [torch-templates](torch-templates.md) | Graph builders for attention, MLP, decoder blocks, and full model architectures |
| [training/](training/README.md) | Training methods: surrogate, evolutionary, and hybrid |
| [inference](inference.md) | Weight export/import, quantization, and `InferenceCache` for autoregressive generation |
| [serialization](serialization.md) | `save_graph` / `load_graph` -- JSON persistence |

## Quick Start

```python
import neuralfn as nf

# Build the default GPT-style model graph
graph = nf.build_gpt_root_graph(name="my_model")

# Save / load
nf.save_graph(graph, "model.json")
graph = nf.load_graph("model.json")

# Define a custom neuron
@nf.neuron(
    inputs=[nf.Port("x", range=(-10, 10))],
    outputs=[nf.Port("y", range=(0, 1))],
)
def my_activation(x):
    return 1.0 / (1.0 + 2.718 ** (-x))
```
