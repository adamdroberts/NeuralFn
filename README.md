# NeuralFn

A brain-inspired neural network framework where each neuron is a either a well-established pre-defined NN function or a user-defined Python function with typed I/O ports, connected in an arbitrary graph.

## How it works

1. **Define neurons** as plain Python functions with `@neuron` decorator specifying input/output ports (name, range, precision).
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

## Defining a neuron

```python
from neuralfn import neuron, Port

@neuron(
    inputs=[Port("x", range=(-5, 5), precision=0.01)],
    outputs=[Port("y", range=(0, 1), precision=0.001)],
)
def sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))
```

## Graph execution

```python
from neuralfn import NeuronGraph, NeuronInstance, Edge

g = NeuronGraph()
g.add_node(NeuronInstance(sigmoid, instance_id="s1"))
# ... add more nodes and edges
result = g.execute({"input_node_id": (0.5,)})
```

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

sigmoid,
relu,
tanh_neuron,
threshold,
identity,
negate,
add,
multiply,
gaussian,
log_neuron,
leaky_relu,
prelu,
relu6,
elu,
selu,
gelu,
silu,
mish,
softplus,
softsign,
hard_sigmoid,
hard_tanh,
hard_swish,
softmax_2,
logsoftmax_2,
input_node,
output_node
