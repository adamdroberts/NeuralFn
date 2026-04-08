# Torch Models

NeuralFn's torch runtime compiles a `NeuronGraph` into a PyTorch `nn.Module`, enabling GPU-accelerated training and inference on tensor data. Module neurons serve as the building blocks.

## Module neurons

A module neuron has `kind="module"` and wraps a PyTorch stage. Unlike function neurons, module neurons cannot be called through the scalar `execute()` path -- they require compilation via `CompiledTorchGraph`.

```python
from neuralfn.neuron import module_neuron
from neuralfn import Port

embedding = module_neuron(
    name="tok_emb",
    module_type="token_embedding",
    input_ports=[Port("token_ids")],
    output_ports=[Port("embeddings")],
    module_config={"vocab_size": 256, "embed_dim": 128},
)
```

Each `module_type` string maps to a `*Stage` class in `torch_backend.py` through the `build_module()` factory.

---

## Built-in module types

The framework ships a broad catalog of module types:

| Category | Module types |
|----------|-------------|
| Embeddings | `token_embedding`, `absolute_position_embedding` |
| Linear | `linear`, `tied_lm_head`, `lm_head` |
| Normalization | `rms_norm`, `layer_norm` |
| Attention plumbing | `reshape_heads`, `merge_heads`, `repeat_kv`, `rotary_embedding`, `qk_gain` |
| Attention kernels | `scaled_dot_product_attention`, `causal_self_attention`, `fused_causal_attention` |
| Residual | `residual_mix`, `residual_add` |
| FFN | `mlp_relu2`, `swiglu`, `gelu` (module) |
| Regularization | `dropout` |
| Output | `logit_softcap`, `token_cross_entropy` |
| KV cache | `kv_cache_read`, `kv_cache_write` |
| MoE routing | `router_logits`, `topk_route`, `expert_dispatch`, `expert_combine`, `load_balance_loss`, `aux_loss_add` |
| Data | `dataset_source` |
| Compression | `bitlinear_ternary`, `randmap_adapter` |
| SSM | `mamba` |

Each module type accepts a `module_config` dict whose keys depend on the stage. Common config keys include `input_dim`, `output_dim`, `num_heads`, `num_kv_heads`, `vocab_size`, `embed_dim`, `dropout_p`, `experts`, and `top_k`.

See [Builtins](../python-sdk/builtins.md) for the complete catalog with per-module config schemas.

---

## CompiledTorchGraph

`CompiledTorchGraph` takes a `NeuronGraph` (with `runtime="torch"`) and compiles it into a single `nn.Module`:

```python
from neuralfn.torch_backend import CompiledTorchGraph

compiled = CompiledTorchGraph(graph)
```

### Key methods

| Method | Description |
|--------|-------------|
| `forward(*tensors)` | Standard `nn.Module` forward pass. Positional tensor args map to the graph's input nodes in order. |
| `trace(*tensors)` | Like `forward()`, but returns a dict of all intermediate activations keyed by node ID. |
| `sync_state_back(graph)` | Writes the compiled module's current `state_dict` back into the graph's `module_state` fields for serialization. |

`CompiledTorchGraph` inherits from `nn.Module`, so all standard PyTorch operations work: `.to(device)`, `.eval()`, `.train()`, `.parameters()`, `state_dict()`, `load_state_dict()`, etc.

---

## Building a minimal torch graph by hand

This example constructs a tiny language model from individual module neurons: token embedding, a linear projection, and a cross-entropy loss.

```python
from neuralfn import NeuronGraph, NeuronInstance, Edge, Port
from neuralfn.neuron import module_neuron
from neuralfn.torch_backend import CompiledTorchGraph
import torch

vocab_size = 256
dim = 128

tok_emb = module_neuron(
    name="tok_emb",
    module_type="token_embedding",
    input_ports=[Port("token_ids")],
    output_ports=[Port("embeddings")],
    module_config={"vocab_size": vocab_size, "embed_dim": dim},
)

proj = module_neuron(
    name="proj",
    module_type="linear",
    input_ports=[Port("x")],
    output_ports=[Port("y")],
    module_config={"input_dim": dim, "output_dim": vocab_size, "bias": False},
)

loss_fn = module_neuron(
    name="loss",
    module_type="token_cross_entropy",
    input_ports=[Port("logits"), Port("targets")],
    output_ports=[Port("loss")],
    module_config={"vocab_size": vocab_size},
)

g = NeuronGraph(name="tiny_lm", runtime="torch", training_method="torch")

g.add_node(NeuronInstance(tok_emb, instance_id="embed"))
g.add_node(NeuronInstance(proj, instance_id="head"))
g.add_node(NeuronInstance(loss_fn, instance_id="loss"))

g.add_edge(Edge(id="e1", src_node="embed", src_port=0, dst_node="head", dst_port=0))
g.add_edge(Edge(id="e2", src_node="head", src_port=0, dst_node="loss", dst_port=0))

g.input_node_ids = ["embed", "loss"]
g.output_node_ids = ["loss"]

compiled = CompiledTorchGraph(g)

tokens = torch.randint(0, vocab_size, (2, 16))
targets = torch.randint(0, vocab_size, (2, 16))
loss = compiled(tokens, targets)
print(f"Loss: {loss[0].item():.4f}")
```

The graph has two input nodes: `embed` receives token IDs and `loss` receives target labels on its second port (port index 1, wired from `head`'s output on port 0). Setting `runtime="torch"` and `training_method="torch"` tells the framework this graph operates on tensors and should be trained with the PyTorch trainer.

---

Next: [Templates and Presets](templates-and-presets.md)
