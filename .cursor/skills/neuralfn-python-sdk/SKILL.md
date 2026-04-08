---
name: neuralfn-python-sdk
description: >-
  Build neural network graphs programmatically using the neuralfn Python
  package. Use whenever the user asks to write Python code that imports
  neuralfn, creates neurons, builds graphs, wires edges, trains models,
  serializes graphs, or works with the NeuralFn graph framework directly
  in code. Do NOT use for MCP tool calls -- see neuralfn-mcp instead.
---

# NeuralFn Python SDK

Use this skill when writing Python code that imports and uses the `neuralfn` package directly. For MCP tool operations, use the `neuralfn-mcp` skill instead.

For complete API details, see [docs/python-sdk/](docs/python-sdk/README.md). For tutorials, see [docs/framework-guide/](docs/framework-guide/README.md).

## Core imports

```python
from neuralfn import (
    Port, NeuronDef, neuron, neuron_from_source, module_neuron, subgraph_neuron,
    BuiltinNeurons, NeuronGraph, NeuronInstance, Edge,
    SurrogateTrainer, EvolutionaryTrainer, HybridConfig, HybridTrainer,
    TorchTrainConfig, TorchTrainer,
    build_gpt_root_graph, build_model_stage_graph,
    save_graph, load_graph,
    SurrogateModel, probe_neuron, build_surrogates,
)
```

Additional imports for configs:

```python
from neuralfn.trainer import TrainConfig
from neuralfn.evolutionary import EvoConfig
from neuralfn.config import ModelSpec, BlockSpec, TemplateSpec
from neuralfn.inference import export_to_pt, import_from_pt, InferenceCache
```

## Creating neurons

### Scalar function neuron (`@neuron`)

```python
@neuron(
    inputs=[Port("x", range=(-5, 5), precision=0.01)],
    outputs=[Port("y", range=(0, 1), precision=0.001)],
)
def my_sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))
```

### Dynamic source neuron

```python
ndef = neuron_from_source(
    "def relu(x):\n    return max(0, x)\n",
    "relu",
    [Port("x", range=(-10, 10))],
    [Port("y", range=(0, 10))],
)
```

### Module neuron (torch stage)

```python
linear = module_neuron(
    name="linear", module_type="linear",
    input_ports=[Port("x", range=(-1e6, 1e6))],
    output_ports=[Port("y", range=(-1e6, 1e6))],
    module_config={"input_dim": 128, "output_dim": 128, "bias": True},
)
```

### Subgraph neuron (nested graph)

```python
child = NeuronGraph(name="block")
child.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
child.add_node(NeuronInstance(BuiltinNeurons.sigmoid, instance_id="act"))
child.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
child.add_edge(Edge(id="e1", src_node="in", src_port=0, dst_node="act", dst_port=0))
child.add_edge(Edge(id="e2", src_node="act", src_port=0, dst_node="out", dst_port=0))
child.input_node_ids = ["in"]
child.output_node_ids = ["out"]

block_neuron = subgraph_neuron(child, name="sig_block", input_aliases=["x"], output_aliases=["y"])
```

## Building and executing graphs

```python
g = NeuronGraph(name="my_graph", training_method="surrogate", runtime="scalar")

g.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in1"))
g.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in2"))
g.add_node(NeuronInstance(BuiltinNeurons.sigmoid, instance_id="act"))
g.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))

g.add_edge(Edge(id="e1", src_node="in1", src_port=0, dst_node="act", dst_port=0, weight=1.0, bias=0.0))
g.add_edge(Edge(id="e2", src_node="in2", src_port=0, dst_node="act", dst_port=0, weight=1.0, bias=0.0))
g.add_edge(Edge(id="e3", src_node="act", src_port=0, dst_node="out", dst_port=0))

g.input_node_ids = ["in1", "in2"]
g.output_node_ids = ["out"]

result = g.execute({"in1": (0.5,), "in2": (0.3,)})
# result: {"out": (0.689...,)}
```

## Training

### Surrogate (scalar graphs, gradient-based)

```python
import numpy as np
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
Y = np.array([[0],[1],[1],[0]], dtype=np.float32)

trainer = SurrogateTrainer(graph, TrainConfig(epochs=300, learning_rate=0.01))
losses = trainer.train(X, Y)
```

### Evolutionary (scalar graphs, population-based)

```python
evo = EvolutionaryTrainer(graph, EvoConfig(population_size=40, generations=100))
losses = evo.train(X, Y)
```

### Hybrid (nested subgraphs, per-graph method)

```python
# Set training_method on each subgraph: "surrogate", "evolutionary", or "frozen"
trainer = HybridTrainer(root, HybridConfig(outer_rounds=3))
losses = trainer.train(X, Y)
```

### Torch (tensor graphs, PyTorch training)

```python
graph = build_gpt_root_graph(preset="nanogpt", config={"n_layer": 4, "n_embd": 128})
trainer = TorchTrainer(graph, TorchTrainConfig(epochs=10, learning_rate=5e-3, device="cuda"))
losses = trainer.train(train_inputs, train_targets)
```

## Serialization

```python
save_graph(graph, "my_model.json")
loaded = load_graph("my_model.json")

# Dict round-trip
d = graph.to_dict()
g2 = NeuronGraph.from_dict(d)
```

## Common builtin neuron IDs

| Attribute | ID | Kind | Ports |
|-----------|----|------|-------|
| `input_node` | builtin-input | function | 0 in, 1 out |
| `output_node` | builtin-output | function | 1 in, 1 out |
| `sigmoid` | builtin-sigmoid | function | 1 in, 1 out |
| `relu` | builtin-relu | function | 1 in, 1 out |
| `tanh_neuron` | builtin-tanh | function | 1 in, 1 out |
| `identity` | builtin-identity | function | 1 in, 1 out |
| `add` | builtin-add | function | 2 in, 1 out |
| `multiply` | builtin-multiply | function | 2 in, 1 out |
| `gelu` | builtin-gelu | function | 1 in, 1 out |
| `silu` | builtin-silu | function | 1 in, 1 out |

Access via `BuiltinNeurons.sigmoid`, `BuiltinNeurons.relu`, etc. Full list: `BuiltinNeurons.all()`.

## Quick reference

### Port fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | required | Unique identifier |
| `range` | tuple[float, float] | (-1.0, 1.0) | Value bounds |
| `precision` | float | 0.001 | Quantization step |
| `dtype` | str | "float" | Semantic type |

### Edge fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | str | auto | Unique edge ID |
| `src_node` | str | "" | Source node instance_id |
| `src_port` | int | 0 | Source port index |
| `dst_node` | str | "" | Destination node instance_id |
| `dst_port` | int | 0 | Destination port index |
| `weight` | float | 1.0 | Multiplicative weight |
| `bias` | float | 0.0 | Additive bias |

### NeuronGraph key methods

| Method | Description |
|--------|-------------|
| `add_node(instance)` | Add a NeuronInstance |
| `add_edge(edge)` | Add an Edge |
| `remove_node(id)` | Remove node and connected edges |
| `remove_edge(id)` | Remove edge |
| `execute(inputs)` | Run with scalar inputs |
| `execute_trace(inputs)` | Run and return all activations |
| `has_cycles()` | Check for cycles |
| `topological_order()` | Get execution order |
| `validate()` | Validate graph structure |
| `to_dict()` / `from_dict(d)` | Serialization |
| `get_edge_params()` / `set_edge_params(p)` | Edge weight vector |
| `resolve_variant_library()` | Resolve variant references |

### NeuronDef kinds

| Kind | Created by | Runtime |
|------|-----------|---------|
| `"function"` | `@neuron`, `neuron_from_source` | scalar |
| `"subgraph"` | `subgraph_neuron` | scalar (recursive) |
| `"module"` | `module_neuron` | torch only |
