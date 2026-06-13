# NeuralFn Python SDK -- Complete Reference

This is the detailed reference for agents building with the `neuralfn` Python package. Read this when you need exact signatures, field lists, or method behavior beyond what SKILL.md covers.

---

## Port (frozen dataclass)

```python
from neuralfn import Port

Port(name: str, range: tuple[float, float] = (-1.0, 1.0), precision: float = 0.001, dtype: str = "float")
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Unique port identifier within a neuron |
| `range` | `tuple[float, float]` | `(-1.0, 1.0)` | `(low, high)` value bounds. `low` must be `< high`. |
| `precision` | `float` | `0.001` | Quantization step. Must be `> 0`. |
| `dtype` | `str` | `"float"` | Semantic type hint: `"float"`, `"int"`, or `"bool"` |

**Methods:**
- `clamp(value: float) -> float` -- restrict to range
- `quantize(value: float) -> float` -- round to nearest precision step
- `condition(value: float) -> float` -- clamp then quantize
- `to_dict() -> dict` / `Port.from_dict(d) -> Port` -- serialization

---

## NeuronDef (dataclass)

```python
from neuralfn import NeuronDef
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Display name |
| `fn` | `Callable \| None` | required | Python callable (None for module/subgraph) |
| `input_ports` | `list[Port]` | required | Input port definitions |
| `output_ports` | `list[Port]` | required | Output port definitions |
| `source_code` | `str` | `""` | Python source for function neurons |
| `kind` | `str` | `"function"` | `"function"`, `"subgraph"`, or `"module"` |
| `subgraph` | `NeuronGraph \| None` | `None` | Nested graph (subgraph kind only) |
| `module_type` | `str` | `""` | Torch module type string (module kind only) |
| `module_config` | `dict` | `{}` | Module configuration |
| `module_state` | `str` | `""` | Base64-encoded torch state_dict |
| `input_aliases` | `list[str]` | `[]` | Renamed input port names for subgraphs |
| `output_aliases` | `list[str]` | `[]` | Renamed output port names for subgraphs |
| `variant_ref` | `dict[str,str] \| None` | `None` | `{"family": "...", "version": "..."}` for variant-linked nodes |
| `id` | `str` | auto (12-char hex) | Unique definition ID |

**Properties:** `n_inputs -> int`, `n_outputs -> int`

**Methods:**
- `__call__(*args: float) -> tuple[float, ...]` -- execute (function/subgraph only; module raises TypeError)
- `refresh_interface_ports() -> None` -- re-derive ports from nested subgraph
- `to_dict() -> dict` / `NeuronDef.from_dict(d) -> NeuronDef` / `NeuronDef.from_dict_raw(d) -> NeuronDef`

---

## Neuron factory functions

### `neuron(inputs, outputs, *, name=None)`

Decorator that turns a plain Python function into a `NeuronDef` with `kind="function"`.

```python
@neuron(
    inputs=[Port("x", range=(-5, 5))],
    outputs=[Port("y", range=(0, 1))],
)
def my_fn(x):
    return 1 / (1 + math.exp(-x))
# my_fn is now a NeuronDef, not a plain function
```

### `neuron_from_source(source_code, fn_name, input_ports, output_ports, *, neuron_id=None)`

Build a NeuronDef from a raw source code string. The function is exec'd and extracted by name.

```python
ndef = neuron_from_source(
    "def relu(x):\n    return max(0, x)\n",
    "relu",
    [Port("x", range=(-10, 10))],
    [Port("y", range=(0, 10))],
)
```

### `module_neuron(*, name, module_type, input_ports, output_ports, module_config=None, module_state="", neuron_id=None)`

Create a NeuronDef wrapping a torch nn.Module stage. Kind is `"module"`. Cannot be called through scalar execution.

```python
linear = module_neuron(
    name="linear",
    module_type="linear",
    input_ports=[Port("x", range=(-1e6, 1e6))],
    output_ports=[Port("y", range=(-1e6, 1e6))],
    module_config={"input_dim": 128, "output_dim": 128, "bias": True},
)
```

### `subgraph_neuron(graph, *, name, input_aliases=None, output_aliases=None, variant_ref=None, neuron_id=None)`

Wrap a `NeuronGraph` as a reusable neuron. The graph must have `input_node_ids` and `output_node_ids` set. Validates the graph as a subgraph. Port names are derived from the child graph's I/O nodes, optionally renamed by aliases.

```python
block = subgraph_neuron(child_graph, name="my_block", input_aliases=["x"], output_aliases=["y"])
```

---

## NeuronInstance (dataclass)

```python
NeuronInstance(neuron_def: NeuronDef, instance_id: str = auto, position: tuple[float, float] = (0, 0))
```

A placed instance of a NeuronDef in a graph. Multiple instances can share the same NeuronDef.

- `name` property delegates to `neuron_def.name`
- `to_dict()` / `NeuronInstance.from_dict(d)` for serialization

---

## Edge (dataclass)

```python
Edge(id: str = auto, src_node: str = "", src_port: int = 0, dst_node: str = "", dst_port: int = 0, weight: float = 1.0, bias: float = 0.0)
```

Weighted connection. The scalar runtime computes `output = input * weight + bias`.

- `transform(value: float) -> float` -- apply the linear transform
- `to_dict()` / `Edge.from_dict(d)` for serialization

---

## NeuronGraph

```python
NeuronGraph(
    *,
    name: str = "graph",
    training_method: str = "surrogate",   # "surrogate" | "evolutionary" | "torch" | "frozen"
    runtime: str = "scalar",              # "scalar" | "torch"
    surrogate_config: dict | None = None,
    evo_config: dict | None = None,
    torch_config: dict | None = None,
    variant_library: dict[str, dict[str, NeuronGraph]] | None = None,
)
```

### Instance attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Graph name |
| `training_method` | `str` | Training strategy |
| `runtime` | `str` | Execution backend |
| `surrogate_config` | `dict` | Surrogate config overrides |
| `evo_config` | `dict` | Evolutionary config overrides |
| `torch_config` | `dict` | Torch config (device, template_spec, etc.) |
| `variant_library` | `dict[str, dict[str, NeuronGraph]]` | `{family: {version: graph}}` |
| `nodes` | `dict[str, NeuronInstance]` | All nodes keyed by instance_id |
| `edges` | `dict[str, Edge]` | All edges keyed by edge id |
| `input_node_ids` | `list[str]` | Ordered input node IDs |
| `output_node_ids` | `list[str]` | Ordered output node IDs |

### All methods

**Node/edge mutations:**
- `add_node(instance: NeuronInstance) -> str` -- returns instance_id
- `remove_node(node_id: str) -> None` -- also removes connected edges and I/O refs
- `add_edge(edge: Edge) -> str` -- raises ValueError if src/dst nodes missing
- `remove_edge(edge_id: str) -> None`

**Topology:**
- `has_cycles() -> bool`
- `topological_order() -> list[str]` -- raises if cyclic
- `validate(*, as_subgraph=False, seen=None) -> None` -- structural validation, recursive for subgraphs

**Interface:**
- `interface_input_layout() -> list[tuple[str, int, Port]]`
- `interface_output_layout() -> list[tuple[str, int, Port]]`
- `flattened_input_ports(aliases=None) -> list[Port]`
- `flattened_output_ports(aliases=None) -> list[Port]`

**Execution (scalar runtime only):**
- `execute(inputs: dict[str, tuple[float,...]], *, max_iters=50, damping=0.5, tolerance=1e-6) -> dict[str, tuple[float,...]]`
  - `inputs`: `{"node_id": (val1, val2, ...)}` for each input node
  - Returns: `{"output_node_id": (val1, ...)}` for each output node
  - Raises TypeError if graph has module nodes
  - DAG graphs: topological order. Cyclic graphs: iterative settling.
- `execute_trace(inputs, ...) -> dict[str, tuple[float,...]]` -- returns ALL node activations
- `execute_flat(flat_inputs: tuple[float,...]) -> tuple[float,...]` -- flat I/O interface

**Inspection:**
- `has_nested_subgraphs() -> bool`
- `has_module_nodes() -> bool` -- recursive
- `has_recursive_subgraphs() -> bool`

**Edge parameters (for training):**
- `get_edge_params() -> list[float]` -- `[w0, b0, w1, b1, ...]` sorted by edge id
- `set_edge_params(params: list[float]) -> None`
- `param_count() -> int` -- `len(edges) * 2`

**Variant library:**
- `resolve_variant_library() -> None` -- resolve all variant_ref nodes against the library. Falls back to inline subgraph if ports are incompatible.

**Serialization:**
- `to_dict() -> dict`
- `NeuronGraph.from_dict(d) -> NeuronGraph` -- resolves variants, validates
- `NeuronGraph.from_dict_raw(d) -> NeuronGraph` -- raw, no resolution

---

## Serialization functions

```python
from neuralfn import save_graph, load_graph

save_graph(graph: NeuronGraph, path: str | Path) -> None
load_graph(path: str | Path) -> NeuronGraph
```

---

## BuiltinNeurons

```python
from neuralfn import BuiltinNeurons

BuiltinNeurons.sigmoid          # NeuronDef (scalar)
BuiltinNeurons.linear_module    # NeuronDef (torch module)
BuiltinNeurons.all()            # list[NeuronDef] -- all 131 builtins
BuiltinNeurons.get("relu")      # lookup by name
```

### Complete builtin catalog

**Scalar activations (kind="function"):**
sigmoid, relu, tanh_neuron, threshold, identity, negate, gaussian, log_neuron, leaky_relu, prelu, relu6, elu, selu, gelu, silu, mish, softplus, softsign, hard_sigmoid, hard_tanh, hard_swish

**Scalar binary/multi-output (kind="function"):**
add (2 in, 1 out), multiply (2 in, 1 out), softmax_2 (2 in, 2 out), logsoftmax_2 (2 in, 2 out)

**Graph terminals (kind="function"):**
input_node (0 in, 1 out), output_node (1 in, 1 out)

**Torch modules (kind="module") -- attribute name -> module_type:**
token_embedding_module, absolute_position_embedding_module, linear_module, mlp_relu2_module, gelu_module, swiglu_module, rms_norm_module, layer_norm_module, dropout_module, reshape_heads_module, merge_heads_module, repeat_kv_module, rotary_embedding_module, qk_gain_module, scaled_dot_product_attention_module, causal_self_attention_module, fused_causal_attention_module, residual_mix_module, residual_add_module, kv_cache_read_module, kv_cache_write_module, kv_pca_encode_module, kv_pca_decode_module, kv_quant_pack_module, kv_quant_unpack_module, tied_lm_head_module, lm_head_module, logit_softcap_module, token_cross_entropy_module, router_logits_module, topk_route_module, expert_dispatch_module, expert_combine_module, load_balance_loss_module, aux_loss_add_module, dataset_source_module, bitlinear_ternary_module, randmap_adapter_module, mamba_module, denoise_head_module, mask_scheduler_module, random_timesteps_module, jepa_mask_module, latent_pool_module, jepa_projector_module, jepa_predictor_module, latent_mse_loss_module, byte_patch_embed_module, byte_patch_merge_module, act_halt_gate_module, act_weighted_sum_module, universal_transformer_module, ttt_linear_module

CUDA Tile SDK helpers live in `neuralfn.tile_cuda`; see `docs/python-sdk/tile-cuda.md` for `TileCudaConfig`, diagnostics, coverage reports with `dtype_support`/`by_dtype`, optimizer helpers, fp8/NVFP4 reference quantize/dequantize helpers, fp8 direct/composite projection and attention Q/K/V contracts, packed NVFP4 projection-family and attention-family activation contracts with optional `preserve_grad` source tracking, graph-level `torch_config["tile_cuda_activation_dtype"] = "nvfp4"` activation packing, category-specific no-support reasons, native GPT-2 trainer handoff helpers, and example-generation commands. `neuralfn.tile_cuda` exports are lazy: registry/config metadata imports and `nfn kernels list [--json]` must not import Torch or graph-backed runtime modules, while tensor execution/diagnostics helpers import their backing modules on first use.

`tools/build_native_train_tile_ops.sh` builds the trainer-facing raw C ABI with
`NFN_TILE_CUDA_USE_CUBLAS_LINEAR=1`, so native linear forward, dInput, dWeight,
and accumulate-dWeight calls use GPU GEMM, while native bias and
accumulate-bias backward calls use GPU GEMV over a cached device ones vector
initialized by a Tile fill kernel. This keeps native trainers off Torch/Python
while moving the GPT-style projection path toward the SM120 `llm.kittens`
throughput target; the generic Tile extension build still uses the pure Tile
fallback unless that macro is set. In the trainer build, the large linear GEMMs
first try a cached BF16 workspace plus `cublasGemmEx` with FP32 outputs and
accumulation before falling back to TF32 `cublasSgemm`; a multi-entry packed
BF16 first-GEMM-operand cache is reused for weight-forward and weight-dInput
calls until the AdamW boundary invalidates it. `NFN_TILE_CUDA_LINEAR_BF16=0`
and `NFN_NATIVE_LINEAR_BF16=0` force the optimized TF32 cuBLAS route for
profiling without rebuilding. Native JSON reports `linear_backend_strategy`,
`linear_bf16_gemm_count`, `linear_sgemm_count`, `linear_bf16_a_pack_count`,
`linear_bf16_a_cache_hit_count`, `linear_bf16_cache_reset_count`,
`linear_bf16_cached_a_capacity`, and `linear_bf16_cache_entry_count`.
Full GPT-2 `--train-transformer-lm` can opt into CUDA-event stage timing with
`NFN_NATIVE_GPT2_STAGE_TIMING=1`; keep it disabled by default for throughput
runs because it records events and synchronizes before reporting. Native JSON
under `timing` should then include `stage_timing_enabled`,
`stage_timing_event_count`, `stage_timing_dropped_event_count`, and
`stage_timing` records for token upload, model/block forward, block
recompute/backward, LM-head backward, final-norm/embedding backward, gradient
zero/clip, and AdamW update. Keep nested diagnostic records for LM-head
logits/CE/dHidden/dWeight, block forward/recompute attention and MLP phases,
and block backward MLP projection, MLP fc, LayerNorm/residual, attention
projection, attention SDPA, and QKV phases.
The trainer-facing build also defaults to the SM120 ThunderKittens bf16
attention bridge (`NFN_TILE_CUDA_USE_TK_ATTENTION=1`,
`NFN_TILE_CUDA_ARCH=sm_120a`) for GPT-2-compatible causal SDPA. Keep
`attention_backend_strategy: "tk-sm120-bf16-bridge"`,
`attention_forward_tk_launch_count`, and `attention_backward_tk_launch_count`
in native training JSON when that path runs; the older row-vector/query-row
atomic float32 SDPA kernels are fallback or diagnostic paths only.

GPT-2 wrapper dry-runs are metadata-only on the default compiled CLI runner:
`--native-cuda-dry-run --native-cuda-print-command` builds the native C++ argv
from the dataset alias/path and must not import `server.dataset_manager`, NumPy,
tiktoken, or Torch, nor materialize raw-text token shards.

Native GPT-2 trained checkpoint export uses the raw Tile ABI
`nfn_native_tile_float32_to_bf16_bits_many` to pack all device float32 weights
into one contiguous bf16 payload before one compact uint16 D2H copy. Preserve
the JSON contract: `checkpoint.payload_pack_strategy:
"device-many-float32-to-bf16-bits-contiguous"`, `payload_pack_kernel:
"nfn_native_tile_float32_to_bf16_bits_many"`, `payload_copy_strategy:
"single-contiguous-device-payload-d2h"`, `payload_cpu_bf16_conversion: false`,
`device_pack_kernel_launches`, `d2h_copy_count`, `d2h_bytes`, and
`float32_d2h_bytes_elided`.

Native GPT-2 handoff helpers are exported from `neuralfn.native_gpt2` and top-level `neuralfn`: `NativeGpt2RunConfig`, `NativeGpt2RunnerStatus`, `NativeGpt2CheckpointInfo`, `build_native_gpt2_run_config()`, `build_native_gpt2_compiled_cli_run_config()`, `native_gpt2_runner_status()`, `resolve_native_gpt2_cli()`, `resolve_native_gpt2_launcher()`, `resolve_native_gpt2_token_shards()`, `write_native_gpt2_run_config()`, `read_native_gpt2_checkpoint_info()`, `is_native_gpt2_checkpoint()`, `latest_native_gpt2_checkpoint()`, `native_gpt2_parameter_count()`, and `run_native_gpt2()`. They materialize/resolve uint16 train and validation token shards and produce compiled native commands so the plain GPT-2 training script can bypass graph-editor node data flow. `build_native_gpt2_compiled_cli_run_config()` is the lowest-startup SDK path for cached shards: it passes a dataset alias/path to the compiled GPT-2 CLI and leaves shard metadata validation to C++. The C++ resolver accepts NeuralFn `fineweb_*` shards, llm.kittens `TinyStories_train.bin` / `TinyStories_val.bin`, `--tinystories` through `/mnt/disk2/dev/open-source/llm.kittens/dev/data/tinystories` or `NFN_LLM_KITTENS_TINYSTORIES_DIR`, and direct train-bin paths with inferred validation siblings. `NativeGpt2RunConfig.kernel_backend` now defaults to `"tile-cuda"` and `NativeGpt2RunConfig.train_transformer_lm` defaults to `True`; use `kernel_backend="llm-kittens"` only when explicitly testing the external `train_gpt2cu` bridge. `neuralfn.config.SHIPPED_GPT_TEMPLATE_PRESETS` and the lazy top-level `neuralfn.SHIPPED_GPT_TEMPLATE_PRESETS` export are the canonical template-name catalog for native selector coverage. `NativeGpt2RunConfig.template_name` and `NativeGpt2RunConfig.graph_file` forward every shipped GPT template name or custom graph selection to the compiled native CLI without importing Torch; the compiled C++ plan JSON reports the synchronized `shipped_template_catalog`, `shipped_template_catalog_count`, and `template_known` fields. Dense GPT-2-compatible presets (`gpt2`, `gpt2_megakernel`, and `gpt2_moa`) map to the implemented native loop and dry-run/plan JSON reports `native-transformer-lm-ready` with `training_step_plan.status: "ready"`, while `gpt2_moa` resolves `NativeGpt2RunConfig.activation` to `moa` automatically. Structurally different shipped templates and custom graphs report `selected-graph-native-trainer-missing` until their C++ Tile trainer plans exist; unknown template names report `unknown-template`. `NativeGpt2RunConfig.smoke_transformer_lm_step` forwards `--smoke-transformer-lm-step`, which samples cached uint16 tokens, preserves range-checked GPT-2 token IDs, and runs token/position embeddings, one tiny transformer block, final LayerNorm, tied LM head, CE forward/backward, transformer backward, embedding backward, and AdamW for 16 parameter buffers through raw Tile kernels without Torch. `NativeGpt2RunConfig.train_transformer_lm` forwards `--train-transformer-lm`, which runs that transformer-LM path as a full-vocab real-dim 12-layer multi-step compiled loop over cached shards with periodic validation records in `validation.losses`, token/position embeddings, transformer blocks, final norm, a row-chunked tied LM-head/CE workspace, transformer backward, embedding backward, device-side global norm gradient clipping, and AdamW parameter updates without Python/Torch; block allocation, initialization, gradient zeroing, gradient clipping, AdamW updates, checkpoint export, activation tape, forward block execution, and backward block execution are driven from per-block C++ state/tape vectors with `activation_tape_strategy: "scratch-recompute"`. The real training loop treats `train_batch_tokens` as the effective optimizer-step batch: it derives `grad_accum_steps` from `batch_size * seq_len`, streams that many cached-shard CUDA Tile microbatches, averages gradients in device accumulation buffers through `nfn_native_tile_gradient_accumulate_float32`, then clips and runs AdamW once. Its JSON reports `microbatch_tokens`, `requested_train_batch_tokens`, `grad_accum_steps`, `effective_train_batch_tokens`, `train_microbatches_completed`, and `gradient_accumulation_strategy`. The real training loop keeps token/target batches as uint16, stages them through pinned host memory, enqueues `cudaMemcpyAsync`, widens to int64 on device through `nfn_native_tile_uint16_to_int64`, and initializes the tied token embedding/LM-head weight directly on device through `nfn_native_tile_init_gpt2_token_weight_float32`; its JSON reports `token_id_upload_strategy: "uint16-pinned-async-h2d-device-widen"`, `token_weight_init_strategy: "device-tile-deterministic"`, and `token_weight_host_materialization: false`. `NativeGpt2RunConfig.lm_head_row_chunk_size` defaults to 1024 and forwards `--lm-head-row-chunk-size`; the C++ loop reduces CE loss partials on device with `nfn_native_tile_sum_partials_float32` before a single host loss copy, and tied LM-head dWeight chunks accumulate directly into `accum_grad_token_weight` with `nfn_native_tile_linear_backward_weight_accumulate_float32` instead of using a full-vocab scratch gradient buffer per chunk or per microbatch. Its JSON reports `trained_layers: 12`, `target_layers: 12`, `block_state_layout` with the block-vector loop flags, `vocab: 50257`, `lm_head_row_chunk_size`, `lm_head_row_chunk_count`, `loss_partial_count`, `logit_workspace_elements`, `gradient_partial_count`, `gradient_clip_norm`, `sample_gradient_clip_scale`, and final 12-layer trained checkpoint metadata when steps complete. `NativeGpt2RunConfig.checkpoint_metadata_smoke` forwards `--checkpoint-metadata-smoke`, which writes a sparse version-5 bf16 native GPT-2 checkpoint-format file plus `DONE_########` marker for the requested `--num-layers` target shape so `read_native_gpt2_checkpoint_info()` and native inference metadata can validate NeuralFn-owned artifacts without CUDA. Successful `--train-transformer-lm` runs write a final 12-layer trained-weight native checkpoint plus `DONE_########` marker. Native checkpoint helpers read `model_########.bin` headers plus `DONE_########` markers without importing Torch, so `nfn infer --checkpoint PATH --native-info` can identify native artifacts instead of sending them to the graph-backed `.pt` loader. Prompt generation from native `.bin` checkpoints still needs a dedicated native inference executable. Build the SDK binding with `bash tools/build_native_gpt2_binding.sh`, launcher with `bash tools/build_native_gpt2_launcher.sh`, no-Python cached-shard CLI with `bash tools/build_native_gpt2_cli.sh`, and unified native frontend with `bash tools/build_native_train_cli.sh`; the compiled GPT-2 CLI links `token_shards.cpp` for shared no-Torch shard validation. `tools/install_native_gpt2_commands.sh` links stable commands including `nfn-gpt2-native` and `nfn-native-train`, and `nfn-native-train --base-model gpt2 --dataset-alias PATH_OR_ALIAS` bypasses Python entirely when cached shards already exist. `nfn-native-train --list-models --json` reports native coverage; dense GPT-2 defaults to the 12-layer Tile-CUDA trainer, NanoGPT reports `partial-native-trainer` for `--train-token-lm`, and LLaMA/GPT-2 evo/JEPA/semantic-MoE/DeepSeek variants report missing or preflight-only native trainers until their C++ targets exist. The CLI wrapper defaults to `--native-cuda-runner compiled-cli`, while `run_native_gpt2(..., runner="auto")` still prefers an in-process C++ binding module (`neuralfn_native_gpt2` or `neuralfn._native_gpt2`), then the compiled no-Python CLI, then the compiled launcher, then direct subprocess execution. `runner="binding"` requires the binding, `runner="compiled-cli"` requires the no-Python CLI, and `runner="launcher"` requires the launcher. Top-level `neuralfn` exports are lazy; importing `neuralfn.native_gpt2` should not import Torch. Torch is optional packaging metadata now; install `.[tile-cuda]` for native CUDA Tile tooling and `.[torch]` separately for graph-backed Torch workflows.

GPT-2-compatible SDPA should use the TK bf16 attention bridge in the
trainer-facing default build. The value-chunked row-vector forward and
query-row atomic backward float32 kernels should remain compiled for unsupported
shapes and `NFN_TILE_CUDA_USE_TK_ATTENTION=0` diagnostics, but do not make them
the default dense GPT-2 training path again.

Full GPT-2 `--train-transformer-lm` should use
`nfn_native_tile_split_qkv_to_heads_add_bias_float32` after the no-bias QKV
projection so Q/K/V bias is applied while Q/K/V head-major buffers are written
directly in one Tile launch per block. Keep `qkv_forward_layout_strategy:
"fused-split-to-heads"`, `qkv_bias_layout_strategy:
"fused-qkv-bias-split-to-heads"`, `qkv_forward_layout_kernel_launches_per_block:
1`, and the elided legacy layout launches in native plan/training JSON.

Full GPT-2 `--train-transformer-lm` should use
the direct TK QKV backward bridge, not standalone
`nfn_native_tile_merge_heads_to_qkv_float32`, in the full trainer. Avoid
separate bf16-to-float gradient conversions, the `merge_heads_to_qkv` launch,
and full-size head-gradient scratch buffers in that path. Keep
`qkv_backward_layout_strategy: "fused-heads-to-qkv"` and the elided bridge
launches in native plan/training JSON.

Full GPT-2 `--train-transformer-lm` should use
the merged-gradient attention backward contract so SDPA backward reads
row-major attention-output gradients directly from the attention projection
backward output. The current full trainer implements this through the direct
QKV bridge below. Do not reintroduce the pre-backward `reshape_heads` launch or
`grad_attn_heads` scratch buffer in the full trainer. Keep
`attention_backward_grad_layout_strategy: "merged-grad-out-direct"` and the
elided grad-output layout launch in native plan/training JSON.

Full GPT-2 `--train-transformer-lm` should use
`nfn_native_tile_split_qkv_to_heads_add_bias_float32` after a no-bias QKV
projection so Q/K/V bias is applied while writing head-major Q/K/V buffers. Do
not reintroduce standalone QKV `linear_add_bias` launches in the full trainer.
Keep `qkv_bias_layout_strategy: "fused-qkv-bias-split-to-heads"` in native
plan/training JSON.

Full GPT-2 `--train-transformer-lm` should use
`nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32`
for the TK bf16 attention backward bridge. It should write row-major `grad_qkv`
directly from bf16 head-major `dQ`/`dK`/`dV`, without allocating full-size
`grad_q_heads`, `grad_k_heads`, or `grad_v_heads` scratch buffers and without
launching separate bf16-to-float gradient conversions plus `merge_heads_to_qkv`
in the full trainer. Keep `attention_backward_qkv_bridge_strategy:
"fused-bf16-heads-to-row-qkv"` in native plan/training JSON.

Full GPT-2 `--train-transformer-lm` should use
`nfn_native_tile_gelu_add_bias_float32` for the MLP `c_fc` bias plus GELU pass.
The `c_fc` linear call should run without bias, then the fused kernel should
write both the biased preactivation for GELU backward and the GELU activation.
Keep `mlp_fc_bias_gelu_strategy: "fused-bias-preactivation-gelu"` in native
plan/training JSON.

Full GPT-2 `--train-transformer-lm` should use
`nfn_native_tile_linear_bias_residual_add_float32` for attention-output and MLP
`c_proj` bias plus residual addition. The projection linear calls should run
without bias, then the fused kernel should apply projection bias, residual
scale, and residual add together. Keep `projection_bias_residual_strategy:
"fused-linear-bias-residual-add"` in native plan/training JSON.

Full GPT-2 `--train-transformer-lm` uploads token and target shards as one
contiguous uint16 arena by sampling directly into pinned memory with
`SequentialTokenBatchSampler::next_into()`, then widens the combined arena with
one `nfn_native_tile_uint16_to_int64` launch. Keep `token_id_h2d_copy:
"cudaMemcpyAsync-contiguous-arena"`, `token_id_h2d_copy_calls_per_microbatch:
1`, `token_id_widen_strategy: "single-contiguous-arena-kernel"`,
`token_id_widen_kernel_launches_per_microbatch: 1`,
`token_batch_staging_strategy: "direct-sampler-to-pinned-arena"`, and
`token_batch_vector_materialization: false` in trainer JSON.

Full GPT-2 `--train-transformer-lm` descriptor tables are suballocated from
one device descriptor arena for parameter fill, gradient zeroing, gradient
clipping, and AdamW. Keep `descriptor_allocation_strategy:
"single-device-arena"`, `descriptor_arena_cuda_malloc_count`,
`descriptor_arena_suballocation_count`, `descriptor_upload_strategy:
"single-host-packed-arena-copy"`, `descriptor_arena_copy_count`,
`descriptor_arena_copy_calls_elided`, and `descriptor_cuda_mallocs_elided` in
trainer JSON, and do not reintroduce one `cudaMalloc` or one H2D copy per
descriptor table.

Full GPT-2 `--train-transformer-lm` keeps train-loss sampling disabled in the hot path: normal optimizer steps skip the tied LM-head CE loss/reduction pass and the post-update device synchronize, and validation cadence computes validation loss without also measuring train loss. Its JSON reports `train_loss_sparse: false`, `train_loss_sampling: "disabled"`, `train_loss_on_validation_steps: false`, `train_loss_eval_count`, and `train_loss_last_step`.

Full GPT-2 `--train-transformer-lm` uses `nfn_native_tile_copy_float32` for persistent block-output preservation instead of zero-fill plus accumulate-by-one. Keep that copy ABI in the raw native Tile symbol set so the scratch-recompute tape does not pay two launches per saved block output. The final block output copy is elided because final LayerNorm consumes it before backward recomputation starts; the default 12-layer JSON reports `persistent_block_outputs: 11` and `final_block_output_copy_elided: true`.

Full GPT-2 `--train-transformer-lm` validation forwards should not copy intermediate block outputs into persistent training-backward buffers because validation has no backward pass. Keep `validation_persistent_block_outputs` at `0` and `validation_block_output_copies_elided` true in the default JSON.

Full GPT-2 `--train-transformer-lm` should reuse the final block activations left in the scratch tape after the initial forward pass. Keep earlier blocks on scratch recomputation from persistent block outputs, but avoid recomputing the final block before its backward pass. Earlier-block recompute should stop after the MLP GELU activation because backward does not consume the recomputed MLP projection output or final residual output. The default 12-layer JSON reports `backward_recompute_blocks: 11`, `final_block_backward_recompute_elided: true`, `backward_recompute_mlp_projection_elided: true`, and `backward_recompute_final_residual_elided: true`.

Full GPT-2 `--train-transformer-lm` also fuses backward residual-gradient pair additions through `nfn_native_tile_scaled_residual_add_float32`. Keep `block_state_layout.residual_backward_fused` true for the compiled trainer and avoid reverting to zero-fill plus two gradient-accumulate launches.

Full GPT-2 `--train-transformer-lm` fuses gradient clipping into AdamW through `nfn_native_tile_adamw_step_with_device_scale_float32`. Keep `block_state_layout.adamw_device_clip_scale_fused` true and avoid reintroducing separate `scale_inplace_by_device` launches before AdamW.

Full GPT-2 `--train-transformer-lm` must use `nfn_native_tile_sumsq_partials_many_float32` over device-resident gradient descriptors for the gradient-clipping sumsq phase. Do not reintroduce one sumsq kernel launch per gradient buffer in the real 12-layer trainer. Keep `gradient_clip_strategy: "fused-multi-buffer-sumsq-device-scale"`, `gradient_sumsq_kernel_launches_per_optimizer_step`, `gradient_sumsq_per_buffer_launches_elided`, and `block_state_layout.gradient_clip_loop: false` in trainer JSON.

Full GPT-2 `--train-transformer-lm` must use `nfn_native_tile_adamw_step_many_with_device_scale_float32` over device-resident parameter descriptors for optimizer updates. Do not reintroduce one AdamW kernel launch per parameter buffer in the real 12-layer trainer. Keep `adamw_update_strategy: "fused-multi-buffer-device-scale"`, `adamw_descriptor_count`, `adamw_step_kernel_launches_per_optimizer_step`, and `adamw_per_buffer_step_launches_elided` in trainer JSON.

Full GPT-2 `--train-transformer-lm` accumulates token, position, block Linear weight, LayerNorm affine, and Linear bias gradients directly into optimizer-step accumulation buffers. The tied LM-head CE backward scale includes the microbatch accumulation factor, LM-head dWeight chunks and token-embedding backward write into `accum_grad_token_weight`, and the old full-vocab `grad_token_weight` scratch buffer is not allocated in the real 12-layer trainer. Position embedding backward uses `nfn_native_tile_absolute_position_embedding_backward_accumulate_float32`, so do not allocate `grad_position_weight` or copy it after every microbatch. Qkv, attention-output, MLP fc, and MLP projection dWeight kernels write straight into per-block accumulation buffers through the accumulate-dWeight ABI. LayerNorm affine and Linear bias gradients use `nfn_native_tile_layer_norm_backward_affine_accumulate_float32` and `nfn_native_tile_linear_backward_bias_accumulate_float32`, so do not allocate per-block scratch gradient buffers or copy them after every microbatch. Accumulation buffers must be zeroed through `nfn_native_tile_fill_many_float32`, not one fill launch per buffer. Keep `block_state_layout.gradient_zero_strategy` set to `fused-multi-buffer-accumulation-zero`, `gradient_zeroed_buffer_count` set to `0`, `gradient_zero_kernel_launches_per_optimizer_step`, `gradient_zero_per_buffer_launches_elided`, `block_state_layout.gradient_accumulation_loop` set to `false`, `block_state_layout.gradient_accumulation_copy_loop_elided` set to `true`, `token_gradient_accumulation_strategy` set to `direct-device-accumulation-buffer`, `position_gradient_accumulation_strategy` set to `direct-device-accumulation-buffer`, `layer_norm_affine_gradient_accumulation_strategy` set to `direct-device-accumulation-buffer`, `linear_bias_gradient_accumulation_strategy` set to `direct-device-accumulation-buffer`, scratch-buffer allocated fields set to `false`, `block_linear_weight_gradient_accumulation_strategy` set to `direct-device-accumulation-buffer`, `block_linear_weight_gradient_scratch_buffers_allocated` set to `false`, `block_state_layout.per_block_gradient_buffers` set to `0`, and `block_state_layout.per_block_direct_accum_gradient_buffers` set to `12`.

Full GPT-2 `--train-transformer-lm` startup keeps per-block parameter/gradient allocation, scratch-tape activation allocation, parameter initialization, and AdamW-state zeroing under the block-vector visitors. Do not reintroduce block-0 aliases into the global startup lists. Keep `block_state_layout.block0_duplicate_allocation_elided`, `block0_duplicate_activation_allocation_elided`, `block0_duplicate_parameter_initialization_elided`, and `block0_duplicate_adamw_state_zero_elided` true.

Full GPT-2 `--train-transformer-lm` suballocates float buffers from one aligned CUDA device arena. Do not reintroduce one `cudaMalloc` per float tensor in the real trainer. Keep `float_allocation_strategy: "single-arena"` plus the arena count/element JSON fields.

Full GPT-2 `--train-transformer-lm` startup zeroes that float arena once and
leaves zero biases and AdamW state at their arena-zero values. Do not re-add
per-buffer zero-fill launches for those tensors. Keep
`float_arena_zero_init_strategy: "single-arena-fill"`,
`float_arena_zero_fill_count`, `startup_per_buffer_zero_fill_elided`, and
`startup_per_buffer_zero_fill_launches_elided` in trainer JSON; the default
12-layer shape elides 369 per-buffer zero-fill launches.

Full GPT-2 `--train-transformer-lm` startup initializes nonzero constant
parameters through `nfn_native_tile_fill_many_values_float32`. Do not re-add one
fill launch per position, final-norm, residual-scale, or block constant-weight
tensor. Keep `parameter_initialization_strategy:
"fused-multi-buffer-fill-values"`, `parameter_initialization_loop: false`, and
`parameter_initialization_per_buffer_launches_elided` in trainer JSON; the
default 12-layer shape elides 74 nonzero fill launches.

Full GPT-2 `--train-transformer-lm` uses combined token arenas: one aligned device arena for widened int64 token/target buffers plus compact uint16 H2D staging, and one pinned uint16 host arena. Do not reintroduce separate token/target `cudaMalloc` calls in the real trainer. Keep `token_buffer_allocation_strategy: "combined-arenas"`, `token_device_allocation_strategy: "single-device-arena"`, `token_device_arena_cuda_malloc_count`, `token_device_arena_suballocation_count`, and `token_device_cuda_mallocs_elided` in trainer JSON.

Full GPT-2 `--train-transformer-lm` should rely on the chunked parallel atomic implementation behind `nfn_native_tile_layer_norm_backward_affine_accumulate_float32` for large row counts. Keep the overwrite ABI available for smoke paths and keep `block_state_layout.layer_norm_backward_affine_strategy` set to `auto-chunked-atomic-accumulate`.

Full GPT-2 `--train-transformer-lm` JSON includes `cuda_runtime_preflight` before allocation. Driver version `0` or a loaded runtime newer than the driver is an early native failure so live SM120 timing attempts fail at the GPU-access/runtime gate instead of a later allocation.

`NativeGpt2RunConfig.kernel_backend` is strict: `tile-cuda` is the default
NeuralFn-owned 12-layer trainer path, and `llm-kittens` is an explicit external
bridge. The `tile-cuda` path may print a plan, verify `tile_ops_lib`, or run
`--train-transformer-lm`; live SM120 throughput validation against the
`llm.kittens` script remains the next proof step. That plan includes the
GPT-2 parameter layout and forward/backward/optimizer
stage sequence, and `block_state_layout` should expose the block-vector loop
flags for allocation, initialization, gradient zeroing, gradient clipping,
AdamW updates, checkpoint export, activation tape, forward blocks, and backward
blocks. Native registry status for GPT-2 should be
`partial-native-trainer` while that throughput validation remains open.

Dense GPT-2 Tile preflight can execute a real raw-kernel smoke without Torch:
set `NativeGpt2RunConfig(smoke_tile_ops=True, tile_ops_lib=..., cuda_runtime_lib=...)`
or pass `--smoke-tile-ops --tile-ops-lib PATH` to `nfn_gpt2_native_train`.
It loads CUDA runtime, launches `nfn_native_tile_fill_float32`, copies the
tiny device buffer back, and emits JSON. Backend names are strict:
`llm-kittens` or `tile-cuda`.
Set `smoke_optimizer_step=True` or pass `--smoke-optimizer-step --tile-ops-lib PATH`
to allocate GPT-2-sized param/grad/AdamW buffers, run one raw Tile AdamW call
per registered GPT-2 parameter buffer, and sample copyback values without Torch.
Set `smoke_lm_step=True` or pass `--smoke-lm-step --tile-ops-lib PATH` to run
a tiny GPT-2-shaped tied embedding/LM-head forward/backward/update slice through
raw Tile kernels without Torch.
`check_tile_ops`, `smoke_tile_ops`, `smoke_optimizer_step`, `smoke_lm_step`,
`smoke_attention_step`, `smoke_mlp_step`, `smoke_norm_residual_step`, and
`smoke_transformer_block_step` are no-data native preflights. They run before
the compiled GPT-2 CLI resolves token shards, so SDK callers can execute them
against a missing dataset alias and should see `token_shards_resolved: false`
instead of a `fineweb_train_*.bin` error. Dataset-backed smokes such as
`smoke_embedding_lm_step` and `smoke_transformer_lm_step`, plus real training
modes, still require cached train/validation shards.
Set `smoke_embedding_lm_step=True` or pass
`--smoke-embedding-lm-step --tile-ops-lib PATH` to sample a tiny cached uint16
token batch in C++ and run GPT-2 token embedding, absolute position embedding,
embedding residual add, final LayerNorm, tied LM head, CE backward,
embedding/norm backward, and AdamW without Torch or graph-editor payloads.
Set `train_embedding_lm=True` or pass `--train-embedding-lm --tile-ops-lib PATH`
to run that GPT-2 embedding/final-norm/LM path as a real multi-step compiled
loop over cached train shards, with validation losses from validation shards
controlled by `eval_every_steps`, `eval_batches`, and `eval_batch_size`.
Set `smoke_attention_step=True` or pass `--smoke-attention-step --tile-ops-lib PATH`
to run a tiny GPT-2 model-dim attention stage through qkv projection, QKV split,
SDPA forward/backward, QKV gradient merge, projection backward, and AdamW
without Torch.
Set `smoke_mlp_step=True` or pass `--smoke-mlp-step --tile-ops-lib PATH` to
run a tiny GPT-2 MLP stage through c_fc projection, GELU forward/backward,
c_proj projection backward, and AdamW without Torch.
Set `smoke_norm_residual_step=True` or pass `--smoke-norm-residual-step --tile-ops-lib PATH`
to run GPT-2 LayerNorm, scaled residual add, LayerNorm affine/input backward,
gradient accumulation, and AdamW without Torch.
Set `smoke_transformer_block_step=True` or pass
`--smoke-transformer-block-step --tile-ops-lib PATH` to compose GPT-2 LayerNorm,
fused QKV attention, residual adds, MLP, backward passes, gradient accumulation,
and AdamW updates without Torch.

Unified native training helpers are exported from `neuralfn.native_train` and top-level `neuralfn`: `NativeTrainRunConfig`, `NativeTrainRunnerStatus`, `build_native_train_run_config()`, `resolve_native_train_cli()`, `native_train_runner_status()`, `native_train_model_registry()`, and `run_native_train()`. Build the C++ extension with `bash tools/build_native_train_binding.sh`; `run_native_train(..., runner="auto")` prefers `neuralfn_native_train` or `neuralfn._native_train`, then the compiled `nfn_native_train` CLI. Set `NFN_NATIVE_TRAIN_BINDING=0` to force CLI fallback. Installed per-family native trainers use both underscore and hyphen command names, and `nfn_native_train` resolves explicit family overrides through `NFN_NATIVE_<MODEL>_CLI` such as `NFN_NATIVE_NANOGPT_CLI`. NanoGPT has a partial native C++ trainer: `nfn_nanogpt_native_train --print-plan` validates and prints the NanoGPT shape/schedule/AdamW/token-shard contract plus contiguous parameter/gradient/AdamW-state buffers, AdamW decay/no-decay groups, and a forward/backward/optimizer `training_step_plan` as JSON without importing Python or Torch. `--check-tile-ops --tile-ops-lib PATH` validates all NanoGPT-required raw Tile ABI symbols by loading the trainer shared library from the compiled binary. `--smoke-tile-ops --tile-ops-lib PATH` also loads CUDA runtime, allocates a tiny device buffer, launches `nfn_native_tile_fill_float32`, copies it back, and verifies the value without Python or Torch. `--smoke-optimizer-step --tile-ops-lib PATH` builds the NanoGPT parameter layout, initializes contiguous param/grad/AdamW moment buffers through raw fill kernels, executes `nfn_native_tile_adamw_step_float32` once per registered parameter buffer with that buffer's decay/no-decay setting, copies param and moment buffers back, and verifies the update without Python or Torch. `--smoke-training-loop-step --tile-ops-lib PATH` exercises native optimizer-loop mechanics over that registered layout: gradient zeroing, synthetic gradient fill, global-norm clip scale finalization, device-scalar gradient scaling, and per-buffer AdamW updates. `--smoke-lm-step --tile-ops-lib PATH` runs a tiny tied-embedding language-model step through token embedding, linear logits, token CE loss/backward, linear input/weight backward, token embedding weight backward, and AdamW update kernels, then verifies loss, gradient, and weight update values. `--smoke-token-train-step --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS` samples a real native uint16 token/target batch from cached shards, runs the tied-LM forward/backward/update kernels over those IDs, and verifies sampled-batch loss, gradient, and weight update values. `--train-token-lm --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS --max-steps N` runs that tied token-embedding LM as a real multi-step native loop over cached shards with device-side gradient zeroing, token CE backward, AdamW metrics JSON, and periodic validation losses over validation shards under JSON `validation.losses`; configure cadence with `--eval-every-steps`, `--eval-batches`, and `--eval-batch-size`. `--smoke-embedding-norm-step --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS` runs sampled tokens through token plus absolute-position embeddings, residual add, LayerNorm forward/backward, tied logits, CE backward, embedding/position/norm gradients, and AdamW updates, then verifies residual, norm, loss, gradient, and weight update values. `--smoke-fused-qkv-attention-step --tile-ops-lib PATH` runs a tiny attention stage through one fused `attn.qkv.weight`, QKV split, SDPA forward/backward, QKV gradient merge, fused qkv weight backward, output projection backward, and AdamW updates for fused qkv/output weights. `--smoke-transformer-block-step --tile-ops-lib PATH` composes LayerNorm, fused-QKV attention, residual adds, MLP, backward passes, gradient accumulation, and AdamW updates for a tiny transformer block through raw native kernels. `--smoke-mlp-step --tile-ops-lib PATH` runs a tiny MLP stage through fc projection, GELU, output projection, projection/input backward, GELU backward, and AdamW updates for both MLP weights, then verifies forward, gradient, and weight update values. `--smoke-attention-step --tile-ops-lib PATH` remains the separate-Q/K/V attention comparison smoke; use `--cuda-runtime-lib PATH` or `NFN_CUDA_RUNTIME_LIB` for explicit libcudart resolution. Tied LM head input/weight backward is represented through the raw linear backward ABI. The AdamW optimizer stage is ready at the registered-buffer level. It defaults to `dropout_p=0.0`; nonzero dropout reports the missing dropout ABI as required work. Full NanoGPT transformer training still needs model-wide trainer-loop integration over the ready transformer stages. This binding is model-family generic and should be used for SDK-level handoff to the compiled native registry; keep architecture-specific setup such as shard materialization in `neuralfn.native_gpt2` until a dedicated native trainer exists for that family.

`bash tools/build_native_train_tile_ops.sh` builds `libnfn_native_train_tile_ops.so`, a raw no-Torch C ABI over CUDA Tile kernels from `neuralfn/csrc/tile_cuda/kernels.cu`. Native C++ trainer implementations should link this shared library for single-buffer and multi-buffer fill/zeroing, single-buffer and multi-buffer sumsq partials, single-buffer and multi-buffer AdamW, gradient accumulation, device-side global-norm clip scale finalization, device-scalar gradient scaling, reduction, linear, linear input/weight/weight-accumulate/bias/bias-accumulate backward, scaled residual add, fused projection bias+residual add, fused QKV split/merge for NanoGPT `qkv.weight`, fused GPT-2 QKV split-to-heads, fused GPT-2 QKV bias+split-to-heads, fused GPT-2 heads-to-QKV gradient merge, GELU forward/backward, token embedding forward/weight backward, absolute-position embedding forward/backward/backward-accumulate, RMSNorm, RMSNorm input backward, LayerNorm, LayerNorm input/affine/affine-accumulate backward, softmax, token and masked token-cross-entropy partial, token and masked token-cross-entropy logits backward, and scaled dot-product attention forward/backward kernels instead of importing the PyTorch extension binding. Use `neuralfn/csrc/native_train/token_shards.cpp` for `NFN_DATASETS_DIR` alias resolution, sorted uint16 `fineweb_train_*.bin` / `fineweb_val_*.bin` validation, llm.kittens `TinyStories_train.bin` / `TinyStories_val.bin` compatibility, direct train-bin validation sibling inference, cached-shard header skipping, token counts, native token/target batch sampling, and microbatch/gradient-accumulation metadata without graph-node payloads.

`bash tools/build_native_missing_trainers.sh` builds compiled placeholder or partial targets for model families that are not fully implemented natively yet. GPT-2 evo is a C++ native preflight target; `nfn_gpt2_evo_native_train --print-plan --eval-every-steps 1000 --tile-cuda-activation-dtype nvfp4` reports the dense GPT-2 shape, `adamw` optimizer profile, validation cadence, NVFP4 activation intent, evo-layer index/cadence/population metadata, and missing candidate-evaluation/mutation/loss-reduction/adoption kernels without Python/Torch. NanoGPT is a partial C++ native trainer; `nfn_nanogpt_native_train --print-plan --require-token-shards --sample-token-batch` reports the resolved shard metadata, first native token/target batch, contiguous parameter/gradient/AdamW-state buffers, AdamW decay/no-decay groups, forward/backward/optimizer `training_step_plan`, tied LM head backward coverage through the linear ABI, kernels already exposed through the native ABI, and missing full-transformer trainer-loop integration without Python/Torch. `nfn_nanogpt_native_train --check-tile-ops --tile-ops-lib PATH` verifies compiled binding to the raw trainer ABI before trainer-loop execution is implemented, `--smoke-tile-ops --tile-ops-lib PATH` verifies an actual fill-kernel launch/copyback through dynamically loaded CUDA runtime, `--smoke-optimizer-step --tile-ops-lib PATH` verifies registered-buffer AdamW iteration over the NanoGPT parameter layout, `--smoke-training-loop-step --tile-ops-lib PATH` verifies native optimizer-loop zero/clip/scale/update mechanics over that same registered layout, `--smoke-lm-step --tile-ops-lib PATH` verifies a tiny tied-embedding LM forward/backward/update path over raw kernels, `--smoke-token-train-step --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS` verifies the same tied-LM kernel path over a real sampled native uint16 token/target batch, `--train-token-lm --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS --max-steps N` runs that tied-LM path as a real multi-step native training loop with periodic validation losses over validation shards, `--smoke-fused-qkv-attention-step --tile-ops-lib PATH` verifies fused `attn.qkv.weight` projection, QKV split, SDPA forward/backward, QKV gradient merge, fused qkv weight backward, output projection backward, and AdamW updates over raw kernels, `--smoke-transformer-block-step --tile-ops-lib PATH` verifies a composed LayerNorm/fused-QKV attention/residual/MLP/backward/AdamW block path over raw kernels, `--smoke-mlp-step --tile-ops-lib PATH` verifies MLP projection/GELU forward/backward/update over raw kernels, and `--smoke-attention-step --tile-ops-lib PATH` verifies the separate-Q/K/V attention comparison path. The other targets report the required CUDA Tile C++ trainer work for LLaMA, MixLLaMA, JEPA, semantic-router MoE, and DeepSeek-V4. They are intentionally not working trainers; use them to keep SDK/CLI handoff on compiled native artifacts while replacing each target with real kernels.

Native trainer CE logits backward in `libnfn_native_train_tile_ops.so` uses row-wise CUDA Tile kernels for vocabularies up to 1024 and chunked row-wise kernels with reusable row-stat workspace for full GPT-class vocabularies; do not reintroduce the elementwise large-vocab fallback.

Linear weight and bias backward in `libnfn_native_train_tile_ops.so` switch large row counts away from one serial row loop per output element. Trainer builds route dWeight through cuBLAS GEMM and bias reductions through cuBLAS GEMV when `NFN_TILE_CUDA_USE_CUBLAS_LINEAR=1`; fallback builds keep row-chunked tiled atomic accumulation. A future tensor-core/GEMM-grade fallback replacement is still useful for dWeight, but do not reintroduce the serial large-row reduction path.

`TorchTrainer.active_compiled_graph` is available only while `TorchTrainer.train()` is running. Use it inside `on_step` callbacks for live validation loss against current in-memory weights; it is cleared after final graph state sync. `TorchTrainer.last_compiled_graph` retains the most recent compiled graph after `train()` returns for final validation against those same trained weights.

`TorchTrainConfig.optimizer_profile="adamw"` is the default gradient optimizer used by the CUDA training scripts. With `kernel_backend="tile_cuda"`, it dispatches batched AdamW updates and gradient clipping through CUDA Tile optimizer kernels and defaults missing `lr_decay_iters`/`min_lr` to cosine decay over the resolved training step count with `min_lr=0.0`. Use `"parameter_golf"` or `"split_muon"` only for explicit split-optimizer/Muon experiments.

Large raw-text dataset aliases materialize a local `uint16` token cache on first
training load when the tokenizer id range fits. `TorchTrainer` and the CLI then
use memmapped `fineweb_train_*.bin` shards and metadata-based schedule estimates
instead of re-tokenizing `data.txt`; graph nodes keep dataset references only.

---

## Training classes

### TrainConfig (dataclass)

```python
from neuralfn.trainer import TrainConfig

TrainConfig(
    learning_rate: float = 1e-3,
    epochs: int = 500,
    batch_size: int = 32,
    surrogate_samples: int = 10_000,
    surrogate_hidden: tuple[int, ...] = (64, 64),
    surrogate_epochs: int = 200,
    loss_fn: str = "mse",      # "mse" or "bce"
)
```

### SurrogateTrainer

```python
from neuralfn import SurrogateTrainer

trainer = SurrogateTrainer(graph: NeuronGraph, config: TrainConfig | None = None)
trainer.train(
    train_inputs: np.ndarray,    # shape (N, n_graph_inputs)
    train_targets: np.ndarray,   # shape (N, n_graph_outputs)
    *,
    on_epoch: Callable[[int, float], None] | None = None,
) -> list[float]                 # per-epoch losses
trainer.stop()                   # signal early stop
```

Attributes: `graph`, `config`, `surrogates` (dict[str, SurrogateModel]), `loss_history`.

### EvoConfig (dataclass)

```python
from neuralfn.evolutionary import EvoConfig

EvoConfig(
    population_size: int = 50,
    generations: int = 200,
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.3,
    crossover_rate: float = 0.5,
    tournament_size: int = 3,
    elite_count: int = 2,
    topology_mutations: bool = False,
    seed: int | None = None,
)
```

### EvolutionaryTrainer

```python
from neuralfn import EvolutionaryTrainer

evo = EvolutionaryTrainer(graph, config: EvoConfig | None = None, neuron_library: list[NeuronDef] | None = None)
evo.train(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    *,
    fitness_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,  # default MSE
    on_generation: Callable[[int, float], None] | None = None,
) -> list[float]
evo.stop()
```

### HybridConfig (dataclass)

```python
from neuralfn import HybridConfig

HybridConfig(
    outer_rounds: int = 3,
    loss_fn: str = "mse",
    default_surrogate: TrainConfig = TrainConfig(),
    default_evolutionary: EvoConfig = EvoConfig(),
)
```

### HybridTrainer

```python
from neuralfn import HybridTrainer

trainer = HybridTrainer(graph: NeuronGraph, config: HybridConfig | None = None)
trainer.train(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    *,
    on_step: Callable[[dict], None] | None = None,
) -> list[float]
trainer.stop()
```

Each subgraph picks its own `training_method`: `"surrogate"`, `"evolutionary"`, or `"frozen"`. The hybrid trainer walks the graph tree in post-order and trains each scope according to its method.

The `on_step` callback receives a dict with keys: `graph_path`, `graph_name`, `method`, `round`, `local_step`, `loss`.

### GraphScope (frozen dataclass)

```python
from neuralfn.hybrid import GraphScope

GraphScope(path: tuple[str, ...], graph: NeuronGraph)
```

Internal representation of a subgraph's position in the nested hierarchy.

---

## Surrogate/probe utilities

```python
from neuralfn import probe_neuron, build_surrogates, SurrogateModel

# Probe a neuron's transfer function
xs, ys = probe_neuron(neuron_def: NeuronDef, n_samples: int = 10_000)
# xs shape: (n_samples, n_inputs), ys shape: (n_samples, n_outputs)

# Build surrogates for every neuron in a graph
surrogates = build_surrogates(graph, n_samples=10_000, hidden_sizes=(64, 64), epochs=200)
# returns dict[str, SurrogateModel] mapping instance_id -> trained model
```

### SurrogateModel (nn.Module)

```python
SurrogateModel(n_inputs: int, n_outputs: int, hidden_sizes: tuple[int, ...] = (64, 64))
```

Small MLP trained to approximate a neuron's transfer function. Has a `forward(x: Tensor) -> Tensor` method.

---

## Variant library

The variant library on `NeuronGraph` is a `dict[str, dict[str, NeuronGraph]]` mapping `family -> version -> graph`.

Nodes can reference a library entry via `variant_ref={"family": "attention", "version": "default"}` on the `NeuronDef`. When `resolve_variant_library()` is called (or `from_dict()` is used), variant-ref nodes have their subgraph replaced with the library version.

If the library entry's ports are incompatible with the node's current ports, the resolver keeps the node's inline subgraph instead of throwing.

`VARIANT_FAMILY_ALIASES` provides compatibility mapping (e.g., `"attn_block"` -> `"transformer_block"`).

---

## encode/decode module state

```python
from neuralfn.neuron import encode_module_state_dict, decode_module_state_dict

blob: str = encode_module_state_dict(state_dict)   # base64-encoded torch.save
state_dict: dict = decode_module_state_dict(blob)   # torch.load from base64
```
