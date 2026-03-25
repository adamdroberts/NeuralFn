# NeuralFn

A brain-inspired neural network framework where each neuron is either a well-established pre-defined NN function or a user-defined Python function with typed I/O ports, connected in an arbitrary graph.

## How it works

1. **Choose neurons** from built-ins or define custom Python functions with `@neuron` and typed I/O ports.
2. **Wire them** into a directed graph with weighted edges.
3. **Probe & train** — the framework samples each neuron to build differentiable surrogate models, then trains connection weights via gradient descent or evolutionary search.
4. **Visual editor** — a React-based web UI for drag-and-drop graph editing with an embedded code editor.

## Quick start

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Run the XOR example

```bash
python examples/xor_graph.py
```

### Run the nested hybrid example

```bash
python examples/nested_hybrid_graph.py
```

### Start the visual editor

Terminal 1 — backend:
```bash
uvicorn server.app:app --reload --port 8000
```

Terminal 2 — frontend:
```bash
cd editor && npm run dev
```

Open http://localhost:5173

## Using built-in neurons

```python
from neuralfn import BuiltinNeurons, NeuronGraph, NeuronInstance

g = NeuronGraph()
g.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
g.add_node(NeuronInstance(BuiltinNeurons.sigmoid, instance_id="act"))
g.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
```

## Defining a custom neuron

```python
from neuralfn import neuron, Port

@neuron(
    inputs=[Port("x", range=(-5, 5), precision=0.01)],
    outputs=[Port("y", range=(0, 1), precision=0.001)],
)
def custom_sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))
```

## Graph execution

```python
from neuralfn import BuiltinNeurons, NeuronGraph, NeuronInstance, Edge

g = NeuronGraph()
g.add_node(NeuronInstance(BuiltinNeurons.sigmoid, instance_id="s1"))
# ... add more nodes and edges
result = g.execute({"input_node_id": (0.5,)})
```

## Nested graphs and mixed trainers

```python
from neuralfn import (
    BuiltinNeurons,
    Edge,
    HybridConfig,
    HybridTrainer,
    NeuronGraph,
    NeuronInstance,
    subgraph_neuron,
)

child = NeuronGraph(name="child", training_method="surrogate")
child.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
child.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
child.add_edge(Edge(id="child-edge", src_node="in", src_port=0, dst_node="out", dst_port=0))
child.input_node_ids = ["in"]
child.output_node_ids = ["out"]

root = NeuronGraph(name="root", training_method="frozen")
root.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="root_in"))
root.add_node(
    NeuronInstance(
        subgraph_neuron(child, name="child_block", input_aliases=["x"], output_aliases=["y"]),
        instance_id="child_block",
    )
)
root.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="root_out"))

trainer = HybridTrainer(root, HybridConfig(outer_rounds=3))
```

Subgraph nodes expose their ports from the nested graph’s designated `input_node_ids` and `output_node_ids`, and each graph picks its own `training_method`: `surrogate`, `evolutionary`, or `frozen`.

## Training

**Surrogate (gradient-based):**
```python
from neuralfn import SurrogateTrainer
from neuralfn.trainer import TrainConfig

trainer = SurrogateTrainer(graph, TrainConfig(epochs=300))
losses = trainer.train(X, Y)
```

**Evolutionary:**
```python
from neuralfn import EvolutionaryTrainer
from neuralfn.evolutionary import EvoConfig

evo = EvolutionaryTrainer(graph, EvoConfig(generations=200))
losses = evo.train(X, Y)
```

## Built-in neurons

Import built-ins from `BuiltinNeurons`:

```python
from neuralfn import BuiltinNeurons

BuiltinNeurons.sigmoid
BuiltinNeurons.relu
BuiltinNeurons.tanh_neuron
BuiltinNeurons.threshold
BuiltinNeurons.identity
BuiltinNeurons.negate
BuiltinNeurons.add
BuiltinNeurons.multiply
BuiltinNeurons.gaussian
BuiltinNeurons.log_neuron
BuiltinNeurons.leaky_relu
BuiltinNeurons.prelu
BuiltinNeurons.relu6
BuiltinNeurons.elu
BuiltinNeurons.selu
BuiltinNeurons.gelu
BuiltinNeurons.silu
BuiltinNeurons.mish
BuiltinNeurons.softplus
BuiltinNeurons.softsign
BuiltinNeurons.hard_sigmoid
BuiltinNeurons.hard_tanh
BuiltinNeurons.hard_swish
BuiltinNeurons.softmax_2
BuiltinNeurons.logsoftmax_2
BuiltinNeurons.input_node
BuiltinNeurons.output_node
```
