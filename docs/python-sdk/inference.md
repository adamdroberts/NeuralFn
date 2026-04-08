# neuralfn.inference

Weight export/import, quantized export/import, and autoregressive inference with KV caching.

---

## export_to_pt

```python
def export_to_pt(graph: NeuronGraph, path: str | Path) -> None
```

Export the weights of a torch-based `NeuronGraph` to a `.pt` file. Compiles the graph to `CompiledTorchGraph` and saves the full `state_dict`.

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

- **`"int8"`**: Per-channel int8 quantization with float32 scale factors for every `nn.Linear` weight tensor.
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
    ) -> None
```

Stateful KV cache manager for autoregressive generation. Wraps a `CompiledTorchGraph` and manages cached K/V tensors across steps.

Works with both training graphs (tokens + targets -> loss) and inference-only graphs (tokens -> logits). When a training graph is detected, dummy targets are supplied automatically.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `NeuronGraph` | *(required)* | The graph to run inference on |
| `device` | `str \| None` | `None` | Device string. Falls back to `graph.torch_config["device"]` or `"cuda"`. |

### Methods

#### `reset() -> None`

Clear all cached KV state. Call between independent sequences.

#### `step(token_ids: Tensor) -> Tensor`

```python
@torch.no_grad()
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
