# neuralfn.inference

Weight export/import, adapter checkpoint helpers, quantized export/import, and autoregressive inference with KV caching.

`neuralfn.inference` is safe to import from the lean native/core SDK. Importing
checkpoint helper names or `InferenceCache` does not import Torch. Calling the
legacy `.pt` checkpoint helpers, quantized export/import helpers, semantic
table helpers, or `InferenceCache` methods still requires PyTorch to be
installed explicitly; missing PyTorch raises an `ImportError` before the
operation starts.

## Strict inference policy

`neuralfn.inference_policy` is Torch-free at import time and centralizes the
temperature-zero compute contract:

```python
from neuralfn.inference_policy import (
    compute_policy_payload,
    inference_execution,
    prepare_inference_process_environment,
    temperature_uses_strict_inference,
    validate_inference_temperature,
    validate_strict_compiled_graph,
    validate_strict_graph_support,
)

prepare_inference_process_environment()  # before importing/initializing CUDA
import torch

temperature = validate_inference_temperature(requested_temperature)

with inference_execution(temperature, torch_module=torch) as policy:
    # Keep the complete forward and token selection inside the context.
    token = run_generation()

telemetry = compute_policy_payload(policy, backend="torch_math")
```

Temperatures must be finite and nonnegative; only exact zero selects strict
execution. `temperature_uses_strict_inference(value)` exposes that exact-zero
predicate after validation. `prepare_inference_process_environment()` establishes
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and fails if CUDA was already initialized
under another value. At zero, `inference_execution()` takes the process-wide
exclusive gate, applies and verifies deterministic/precision controls,
synchronizes outstanding CUDA work, restores the previous controls, and then
releases the gate. Positive-temperature contexts take shared access and do not
change the optimized backend policy.

When the installed PyTorch exposes its hierarchical precision API, the strict
context explicitly requires `"ieee"` for the global, CUDA matmul, cuDNN,
cuDNN convolution, and cuDNN RNN `fp32_precision` controls. On earlier releases
it uses the legacy `allow_tf32=False` plus highest matmul precision controls.
The two API generations are not mixed, and an incomplete hierarchical API
fails closed.

Call `validate_strict_graph_support()` before compiling a user-supplied graph
and `validate_strict_compiled_graph()` after loading and moving weights. The
latter rejects any remaining non-FP32 floating parameters or buffers. The
graph validator also rejects training-time stateful stochastic mask/timestep
modules whose changing counters would violate repeatable inference. The
context does not compile or replace a model and cannot disable an autocast
context opened by the caller; strict callers must use a Torch-only FP32
compiled graph and keep autocast disabled.

Native dense GPT CUDA checkpoints use the Torch-free `neuralfn.native_gpt` and
compatibility `neuralfn.native_gpt2` helpers. Use
`latest_native_gpt_checkpoint(output_dir)` to resolve the latest completed
`model_########.bin` with a matching `DONE_########` marker, or
`read_native_gpt_checkpoint_info(path)` to inspect the native header shape,
precision, expected size, and checkpoint step. The compiled
`nfn_gpt_native_train --checkpoint-metadata-smoke --template-name TEMPLATE`
writer now uses the selected dense template geometry, so NanoGPT-family metadata
checkpoints report 5 layers, 5 heads, and 320 channels, while GPT-2-family
selectors report 12 layers, 12 heads, and 768 channels. For automation, run
`tools/smoke_native_gpt_template_checkpoints.py` to produce and inspect one
checkpoint for every covered dense GPT selector without importing Torch.

Native-family C++ checkpoints use the separate Torch-free
`neuralfn.native_family` helper module. Covered family dataset loops write
`*_native_family_model_00000000.json` artifacts in format
`nfn-native-family-optimizer-checkpoint-v1` for current optimizer-updated
artifacts. The loader also accepts the older
`nfn-native-family-token-transition-v1` transition artifacts; call
`read_native_family_checkpoint_info(path)` to inspect them or
`sample_native_family_checkpoint(path, prompt_tokens="1,2", max_new_tokens=64)`
to generate token IDs from their transition-table state. Use
`list_native_family_checkpoints(output_dir)` to enumerate every native-family
model artifact produced by a sweep, or
`verify_native_family_checkpoint(path)` as the strict loadability gate; it
requires the model `DONE` marker, matching sidecar byte size, nonempty
transition state, a full-template parameter state, contiguous buffer offsets,
trained sidecar slots, `parameter_lm_head_inference_supported`, the
trainer-emitted `writer_verification` block, and a bounded sample that uses a
persisted-parameter inference path. The writer verification proves the native
checkpoint writer wrote a dense optimizer-updated `.f32` sidecar before
reporting the model artifact as written. Sparse diagnostic native-family
checkpoints advertise
`working_model_inference_path: token_embedding_lm_head_sidecar_forward`. Full
live-parameter family checkpoints advertise
`native_family_architecture_sidecar_forward_v1` when sidecar metadata records
full trained-parameter coverage (`trained_parameter_elements ==
parameter_elements`). Architecture-forward verification rejects partial
sampled-update sidecars even if they otherwise contain a full-size dense base.
Older
sampler-backed artifacts with
`working_model_inference_path: token_embedding_lm_head_sidecar_forward` still
load, but they fail the stricter architecture-forward gate. The checkpoint also
contains `native_parameter_state` and a `parameter_data` entry so SDK callers
can detect persisted tensor slots, trained tensor counts, dense-base metadata,
and whether architecture-forward inference is available. The layout records per-buffer
`offset`, `byte_offset`, and `bytes` fields for the persisted `.f32` sidecar;
the native JSON also records a `parameter_update_checksum` for the sampled
sidecar writes. This is the current loadable inference contract for covered
native-family templates.
When a compiled family binary is explicitly built with
`NFN_NATIVE_PRODUCTION_LOOP=1` and the production bootstrap is ready, checkpoint
metadata labels the sidecar as
`live_family_device_parameter_store_float32_state` and records
`optimizer_updated_full_architecture_parameter_persistence: true`, because the
writer copies the live `FamilyDeviceParameterStore` after the shared
`FamilyOptimizerState` step. This proves full sidecar persistence and enables
the architecture-sidecar bounded scorer when every architecture parameter
element is persisted as trained state.
Use `audit_native_family_checkpoint_template_coverage(output_dir,
required_templates={...})` when a sweep must prove template-level coverage, not
just family-level smoke coverage. The audit runs the same strict checkpoint
verifier over the directory and then requires a passing artifact whose
normalized `template_name` matches each required template. This is the SDK
equivalent of `nfn infer --checkpoint DIR --verify-all --required-templates ...`
or `--require-covered-templates`. Pass `require_architecture_forward=True` to
`verify_native_family_checkpoint(...)` or
`audit_native_family_checkpoint_template_coverage(...)` when a gate must prove
architecture-specific forward inference from the persisted parameter state.
The stricter check requires every architecture parameter element to be trained
and persisted, so sampled architecture-sidecar artifacts and older
sampler-backed artifacts intentionally fail it while
`architecture_forward_inference_supported` remains false.

---

## export_to_pt

```python
def export_to_pt(graph: NeuronGraph, path: str | Path) -> None
```

Export the weights of a torch-based `NeuronGraph` to a `.pt` file. Compiles the graph to `CompiledTorchGraph` and saves the full `state_dict`.
The checkpoint payload is `{"state_dict": ..., "checkpoint_metadata": ...}`.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `graph` | `NeuronGraph` | The graph whose weights to export |
| `path` | `str \| Path` | Output file path |

---

## import_from_pt

```python
def import_from_pt(graph: NeuronGraph, path: str | Path) -> None
```

Import weights from a `.pt` file into a `NeuronGraph`. Loads the state_dict, compiles the graph, loads weights into the compiled module, then syncs the state back into the graph's `module_state` fields.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `graph` | `NeuronGraph` | The graph to load weights into |
| `path` | `str \| Path` | Path to the `.pt` file |

---

## load_pt_checkpoint

```python
def load_pt_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]
```

Load either a modern NeuralFn checkpoint payload with `state_dict` and
`checkpoint_metadata`, or a legacy plain state-dict checkpoint. Returns
`(state_dict, metadata)`.

---

## Adapter checkpoint helpers

### save_adapter_checkpoint

```python
def save_adapter_checkpoint(graph: NeuronGraph, path: str | Path) -> None
```

Compile `graph`, filter the state dict down to LoRA/qLoRA adapter parameters,
RandMap adapter middle/scale parameters, and value/reward head parameters, then
write an adapter-only checkpoint. The metadata includes `adapter_only=True`.

### load_adapter_checkpoint

```python
def load_adapter_checkpoint(graph: NeuronGraph, path: str | Path) -> None
```

Load an adapter-only checkpoint into `graph` with non-strict state-dict loading,
then sync the loaded adapter/head tensors back to graph `module_state`.

### merge_adapter_into_base

```python
def merge_adapter_into_base(
    base_path: str | Path,
    adapter_path: str | Path,
    out_path: str | Path,
) -> None
```

Merge LoRA `A` / `B` tensors into matching base projection weights and write a
plain full checkpoint with `merged_from_adapter=True` metadata. This is useful
when a small adapter artifact should be baked into a standalone inference
checkpoint.

---

## export_quantized_pt

```python
def export_quantized_pt(
    graph: NeuronGraph,
    path: str | Path,
    scheme: str = "int8",
) -> None
```

Export weights with quantization applied. Saves both the quantized state_dict and quantization metadata (scales, scheme) in a single checkpoint.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `NeuronGraph` | *(required)* | The graph to export |
| `path` | `str \| Path` | *(required)* | Output file path |
| `scheme` | `str` | `"int8"` | Quantization scheme: `"int8"` or `"ternary"` |

### Schemes

- **`"int8"`**: Per-channel int8 quantization with float32 scale factors for linear/projection weight tensors. Token and position embedding tables remain full precision so round-trip loss drift stays bounded and the export matches the linear-weight storage contract.
- **`"ternary"`**: Bake ternary `{-1, 0, 1}` weights for BitLinearTernary models with per-tensor scale.

---

## import_quantized_pt

```python
def import_quantized_pt(graph: NeuronGraph, path: str | Path) -> None
```

Import quantized weights from a checkpoint, dequantizing them back to float32 for execution. Reads the quantization metadata to determine the scheme and applies the inverse transform.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `graph` | `NeuronGraph` | The graph to load weights into |
| `path` | `str \| Path` | Path to the quantized checkpoint |

---

## InferenceCache

```python
class InferenceCache:
    def __init__(
        self,
        graph: NeuronGraph,
        device: str | None = None,
        *,
        compiled: CompiledTorchGraph | None = None,
    ) -> None
```

Stateful KV cache manager for autoregressive generation. Wraps a `CompiledTorchGraph` and manages cached K/V tensors across steps.

Works with both training graphs (tokens + targets -> loss) and inference-only graphs (tokens -> logits). When a training graph is detected, dummy targets are supplied automatically.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `NeuronGraph` | *(required)* | The graph to run inference on |
| `device` | `str \| None` | `None` | Device string. Falls back to `graph.torch_config["device"]` or `"cuda"`. |
| `compiled` | `CompiledTorchGraph \| None` | `None` | Reuse an already constructed and weight-loaded graph. The cache moves it to `device`; when omitted the cache compiles `graph` as before. |

Pass `compiled=` when checkpoint or adapter weights have already been loaded,
or when a temperature-zero strict FP32 graph was selected. This prevents the
cache from silently constructing a second model with different weights or
backend policy.

`SemanticInferenceCache` accepts the same keyword-only `compiled=` argument and
retains its existing semantic-vector inspection behavior.

### Methods

#### `reset() -> None`

Clear all cached KV state. Call between independent sequences.

#### `step(token_ids: Tensor) -> Tensor`

```python
def step(self, token_ids: torch.Tensor) -> torch.Tensor
```

Run one autoregressive step.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `token_ids` | `Tensor` | Shape `(batch, seq)`. On the first call, this is the full prompt. On subsequent calls, pass `(batch, 1)` for the single new token. |

**Returns:** Logits tensor of shape `(batch, vocab_size)` for the last position.

### Usage Example

```python
from neuralfn.inference import InferenceCache

cache = InferenceCache(graph, device="cuda")

# Prompt
logits = cache.step(prompt_ids)  # (batch, seq) -> (batch, vocab)

# Generate token-by-token
for _ in range(max_new_tokens):
    next_token = logits.argmax(dim=-1, keepdim=True)
    logits = cache.step(next_token)  # (batch, 1) -> (batch, vocab)

cache.reset()
```

---

## Semantic table helpers [Experimental]

```python
def export_semantic_tables(graph: NeuronGraph, path: str | Path) -> None
def import_semantic_tables(graph: NeuronGraph, path: str | Path) -> None
```

Export or import semantic routing / legacy decoder lookup tensors whose state
keys include semantic decoder, hasher, or semantic router components. These
helpers are experimental and tied to the semantic routing research presets.
