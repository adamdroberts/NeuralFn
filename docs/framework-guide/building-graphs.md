# Building Graphs

A `NeuronGraph` is a directed graph of `NeuronInstance` nodes connected by weighted `Edge` objects. This page covers construction, execution, topology analysis, and serialization.

## Creating a graph

```python
from neuralfn import NeuronGraph

g = NeuronGraph(
    name="my_graph",
    training_method="surrogate",
    runtime="scalar",
)
```

### Constructor keyword arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `""` | Display name for the graph. |
| `training_method` | `str` | `"surrogate"` | Training strategy: `"surrogate"`, `"evolutionary"`, `"frozen"`, or `"torch"`. |
| `runtime` | `str` | `"scalar"` | Execution backend: `"scalar"` (Python floats) or `"torch"` (PyTorch tensors). |
| `surrogate_config` | `dict` | `{}` | Default surrogate training parameters for this graph. |
| `evo_config` | `dict` | `{}` | Default evolutionary training parameters. |
| `torch_config` | `dict` | `{}` | Torch runtime configuration (device, compile flags, template_spec, etc.). |
| `variant_library` | `dict` | `{}` | Maps `family -> version -> NeuronGraph` for variant resolution. |

---

## Adding nodes

Each node is a `NeuronInstance` -- a placement of a `NeuronDef` at a unique ID:

```python
from neuralfn import NeuronInstance, BuiltinNeurons

in_node = NeuronInstance(BuiltinNeurons.input_node, instance_id="in1")
sig = NeuronInstance(BuiltinNeurons.sigmoid, instance_id="sig")
out_node = NeuronInstance(BuiltinNeurons.output_node, instance_id="out")

g.add_node(in_node)
g.add_node(sig)
g.add_node(out_node)
```

The optional `position` argument sets the node's canvas coordinates in the visual editor:

```python
NeuronInstance(BuiltinNeurons.sigmoid, instance_id="sig", position=(100, 200))
```

---

## Adding edges

Edges carry values from one node's output port to another node's input port:

```python
from neuralfn import Edge

g.add_edge(Edge(
    id="e1",
    src_node="in1",
    src_port=0,
    dst_node="sig",
    dst_port=0,
    weight=1.0,
    bias=0.0,
))
g.add_edge(Edge(
    id="e2",
    src_node="sig",
    src_port=0,
    dst_node="out",
    dst_port=0,
))
```

### Edge fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | required | Unique edge identifier. |
| `src_node` | `str` | required | Source node instance ID. |
| `src_port` | `int` | required | Source output port index. |
| `dst_node` | `str` | required | Destination node instance ID. |
| `dst_port` | `int` | required | Destination input port index. |
| `weight` | `float` | `1.0` | Multiplicative weight applied to the value in transit. |
| `bias` | `float` | `0.0` | Additive bias applied after the weight. |

The scalar runtime computes `dst_value = src_value * weight + bias` for each edge.

---

## Setting inputs and outputs

Designate which nodes serve as the graph's external interface:

```python
g.input_node_ids = ["in1"]
g.output_node_ids = ["out"]
```

Input nodes receive data from `execute()` calls. Output nodes provide the return values.

---

## Executing a graph

### Scalar execution

```python
result = g.execute({"in1": (0.5,)})
print(result)  # {"out": (0.7310585786300049,)}
```

`execute()` takes a dict mapping input node IDs to tuples of values (one per port). It returns a dict mapping output node IDs to tuples of output values.

For multiple inputs:

```python
result = g.execute({"in1": (0.5,), "in2": (0.3,)})
```

### Execution tracing

To capture every node's activation, not just the outputs:

```python
trace = g.execute_trace({"in1": (0.5,)})
# trace["sig"] contains the sigmoid node's output
```

---

## Topology

```python
g.has_cycles()         # False for feedforward graphs
g.topological_order()  # list of node IDs in execution order
g.validate()           # raises on structural errors
```

---

## Cyclic (recurrent) graphs

Graphs with cycles are supported. The `execute()` method iteratively settles the graph:

```python
result = g.execute(
    {"in1": (1.0,)},
    max_iters=50,
    damping=0.5,
    tolerance=1e-6,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iters` | `100` | Maximum settling iterations before returning. |
| `damping` | `0.0` | Blend factor between old and new activations each iteration. `0.0` = no damping. |
| `tolerance` | `1e-6` | Convergence threshold. Iteration stops when the maximum activation change falls below this. |

---

## Edge parameters

The trainable parameters of a scalar graph are its edge weights and biases:

```python
params = g.get_edge_params()   # flat numpy array of [w1, b1, w2, b2, ...]
g.set_edge_params(params)      # write parameters back
count = g.param_count()        # total number of trainable scalars
```

Surrogate and evolutionary trainers operate on this parameter vector.

---

## Serialization

Graphs round-trip through JSON-compatible dicts:

```python
d = g.to_dict()
restored = NeuronGraph.from_dict(d)
```

`from_dict()` resolves neuron definitions by name against the built-in catalog. For graphs that embed custom neurons or module state, `from_dict_raw()` preserves the raw structure without catalog resolution:

```python
raw_restored = NeuronGraph.from_dict_raw(d)
```

---

## Full example: XOR graph

This builds the classic XOR network with two hidden neurons, trains it, and verifies the outputs. The full runnable script lives in `examples/xor_graph.py`.

```python
import numpy as np
from neuralfn import (
    BuiltinNeurons, Port, neuron, NeuronGraph, NeuronInstance, Edge,
    SurrogateTrainer,
)
from neuralfn.trainer import TrainConfig


@neuron(
    inputs=[Port("a", range=(-10, 10)), Port("b", range=(-10, 10))],
    outputs=[Port("sum", range=(-20, 20))],
)
def weighted_sum(a, b):
    return a + b


def build_xor_graph() -> NeuronGraph:
    g = NeuronGraph()

    in1 = NeuronInstance(BuiltinNeurons.input_node, instance_id="in1")
    in2 = NeuronInstance(BuiltinNeurons.input_node, instance_id="in2")
    h1 = NeuronInstance(weighted_sum, instance_id="h1")
    h2 = NeuronInstance(weighted_sum, instance_id="h2")
    a1 = NeuronInstance(BuiltinNeurons.sigmoid, instance_id="a1")
    a2 = NeuronInstance(BuiltinNeurons.sigmoid, instance_id="a2")
    h3 = NeuronInstance(weighted_sum, instance_id="h3")
    a3 = NeuronInstance(BuiltinNeurons.sigmoid, instance_id="a3")
    out = NeuronInstance(BuiltinNeurons.output_node, instance_id="out")

    for node in [in1, in2, h1, h2, a1, a2, h3, a3, out]:
        g.add_node(node)

    g.input_node_ids = ["in1", "in2"]
    g.output_node_ids = ["out"]

    edges = [
        Edge(id="e1",  src_node="in1", src_port=0, dst_node="h1", dst_port=0, weight=1.0, bias=0.0),
        Edge(id="e2",  src_node="in2", src_port=0, dst_node="h1", dst_port=1, weight=1.0, bias=0.0),
        Edge(id="e3",  src_node="in1", src_port=0, dst_node="h2", dst_port=0, weight=1.0, bias=0.0),
        Edge(id="e4",  src_node="in2", src_port=0, dst_node="h2", dst_port=1, weight=1.0, bias=0.0),
        Edge(id="e5",  src_node="h1",  src_port=0, dst_node="a1", dst_port=0, weight=1.0, bias=0.0),
        Edge(id="e6",  src_node="h2",  src_port=0, dst_node="a2", dst_port=0, weight=1.0, bias=0.0),
        Edge(id="e7",  src_node="a1",  src_port=0, dst_node="h3", dst_port=0, weight=1.0, bias=0.0),
        Edge(id="e8",  src_node="a2",  src_port=0, dst_node="h3", dst_port=1, weight=1.0, bias=0.0),
        Edge(id="e9",  src_node="h3",  src_port=0, dst_node="a3", dst_port=0, weight=1.0, bias=0.0),
        Edge(id="e10", src_node="a3",  src_port=0, dst_node="out", dst_port=0, weight=1.0, bias=0.0),
    ]
    for e in edges:
        g.add_edge(e)

    return g


g = build_xor_graph()
print(f"Nodes: {len(g.nodes)}, Edges: {len(g.edges)}")
print(f"Cycles: {g.has_cycles()}")

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

trainer = SurrogateTrainer(g, TrainConfig(epochs=300, learning_rate=0.01))
losses = trainer.train(X, Y, on_epoch=lambda ep, loss: (
    print(f"epoch {ep}: {loss:.6f}") if ep % 50 == 0 else None
))

for row in X:
    out = g.execute({"in1": (float(row[0]),), "in2": (float(row[1]),)})
    print(f"{row} -> {out['out']}")
```

---

Next: [Subgraphs and Variants](subgraphs-and-variants.md)
