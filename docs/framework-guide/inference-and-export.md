# Inference and Export

After training a torch-backed graph, NeuralFn provides tools for saving weights, quantizing checkpoints, and running autoregressive generation with KV caching.

## Weight export and import

### Full-precision export

```python
from neuralfn.inference import export_to_pt, import_from_pt

export_to_pt(graph, "model.pt")
```

This compiles the graph into a `CompiledTorchGraph`, extracts its `state_dict`, and saves it to a `.pt` file.

### Full-precision import

```python
import_from_pt(graph, "model.pt")
```

Loads the state dict, rebuilds the compiled module, calls `load_state_dict()`, then syncs the weights back into each node's `module_state` field in the graph. After import, the graph's serialized form (`to_dict()`) includes the loaded weights.

---

## Adapter-only checkpoints

For LoRA/qLoRA/RandMap fine-tuning runs, NeuralFn can save and reload only the
adapter and head parameters:

```python
from neuralfn.inference import (
    save_adapter_checkpoint,
    load_adapter_checkpoint,
    merge_adapter_into_base,
)

save_adapter_checkpoint(graph, "adapter.pt")
load_adapter_checkpoint(graph, "adapter.pt")
merge_adapter_into_base("base.pt", "adapter.pt", "merged.pt")
```

`save_adapter_checkpoint()` filters the compiled state dict to LoRA/qLoRA
adapter tensors, RandMap adapter middle/scale tensors, and value/reward heads.
`merge_adapter_into_base()` bakes LoRA deltas into a full checkpoint for
ordinary inference.

---

## Quantized export and import

### Export with quantization

```python
from neuralfn.inference import export_quantized_pt

export_quantized_pt(graph, "model_q.pt", scheme="int8")
```

Two schemes are supported:

| Scheme | Description |
|--------|-------------|
| `"int8"` | Per-channel int8 quantization with float32 scale factors. Applied to linear/projection weight tensors; token and position embeddings remain full precision. |
| `"ternary"` | Bakes weights to `{-1, 0, 1}` with a single mean-absolute scale per tensor. Designed for BitLinear / ternary_b158 models. |

Non-weight tensors (biases, norms, embeddings) are stored at full precision in both schemes.

### Import with dequantization

```python
from neuralfn.inference import import_quantized_pt

import_quantized_pt(graph, "model_q.pt")
```

Reads the quantized checkpoint, dequantizes all weight tensors back to float32, loads them into a compiled graph, and syncs back to the graph's node states. The graph operates at full precision after import -- quantization is a storage optimization, not a runtime one.

---

## InferenceCache

`InferenceCache` wraps a compiled graph for stateful autoregressive generation. It manages KV cache tensors across decoding steps.

```python
from neuralfn.inference import InferenceCache
import torch

cache = InferenceCache(graph, device="cuda")

prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
logits = cache.step(prompt)
next_token = logits.argmax(dim=-1)

logits2 = cache.step(next_token.unsqueeze(1))

cache.reset()
```

### How it works

1. **First call**: pass the full prompt as `(batch, seq_len)`. The cache processes all tokens and returns logits for the last position.
2. **Subsequent calls**: pass a single new token as `(batch, 1)`. The KV cache avoids recomputing attention over the full sequence.
3. **Reset**: call `cache.reset()` to clear the KV cache and start a new sequence.

### Constructor

```python
InferenceCache(graph, device=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `NeuronGraph` | required | A torch-runtime graph, typically built from a template preset. |
| `device` | `str` | from `torch_config` or `"cuda"` | Device to run inference on. |

### KV cache nodes

`InferenceCache` works with graphs that include `kv_cache_read` and `kv_cache_write` module nodes in their attention subgraphs. These nodes are automatically inserted by the template builders for presets that set `backend_capabilities["cache"] = True`.

### Training graph compatibility

For training graphs that take two inputs (tokens + targets), `InferenceCache.step()` automatically generates dummy target tensors so the forward pass runs without modification. The loss output is returned as-is, which can be useful for perplexity evaluation.

If you export or probe a tokenizer-backed cached dataset alias, keep the tokenizer contract intact: NeuralFn now preflights tokenizer-backed aliases before generation and stops early when the cached tokenizer vocab, shard ids, and checkpoint vocab disagree. This avoids the previous failure mode where decoding crashed after generation had already emitted out-of-range token ids.

---

## Typical workflow

```python
from neuralfn import build_gpt_root_graph, TorchTrainer, TorchTrainConfig
from neuralfn.config import build_llama_spec
from neuralfn.inference import export_to_pt, InferenceCache
import torch

spec = build_llama_spec(n_layer=4, n_embd=128, vocab_size=256)
graph = build_gpt_root_graph(model_spec=spec)

tokens = torch.randint(0, 256, (16, 64))
targets = torch.randint(0, 256, (16, 64))
trainer = TorchTrainer(graph, TorchTrainConfig(epochs=5, device="cuda"))
trainer.train(tokens, targets)

export_to_pt(graph, "llama_small.pt")

cache = InferenceCache(graph, device="cuda")
prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
generated = prompt.squeeze(0).tolist()

for _ in range(50):
    logits = cache.step(prompt if len(generated) == 3 else next_tok.unsqueeze(1))
    next_tok = logits.argmax(dim=-1)
    generated.append(next_tok.item())

print(generated)
```

---

Next: [Datasets](datasets.md)
