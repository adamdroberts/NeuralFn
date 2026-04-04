# NeuralFn

A brain-inspired neural network framework where each neuron is either a well-established pre-defined NN function or a user-defined Python function with typed I/O ports, connected in an arbitrary graph.

NeuralFn now has both a scalar graph runtime and a PyTorch-backed `torch` runtime for trainable module nodes such as GPT.

## Current state of play

NeuralFn now has full support for Torch-backed large language models. The framework currently provides templates for NanoGPT, GPT-2, Llama, and MoE (Mixture of Experts) architectures. **Note: Only NanoGPT has been fully tested so far. GPT-2, Llama, and MoE support is implemented but untested.**

Training is powered by a new `dataset_source` node that can download Hugging Face datasets, seamlessly tokenize them, and feed them directly into the graph. Trained weights are embedded directly back into the graph's serialized JSON format (`module_state`), meaning your architecture and trained parameters reside safely inside the same visual graph structure. Hit save to save the model and the configuration / design after training or before.

**TODOs:**
- Need to add an export for `.pt` file.
- Need to add inference support for the custom training / graph format. :-)

## How it works

1. **Choose neurons** from built-ins or define custom Python functions with `@neuron` and typed I/O ports.
2. **Wire them** into a directed graph with weighted edges.
3. **Probe & train** — scalar graphs sample each neuron to build differentiable surrogate models, then train connection weights via gradient descent or evolutionary search.
4. **Torch modules** — tensor-native graphs can train serialised module nodes through a PyTorch backend, including nested subgraphs built from multiple trainable stages.
5. **Visual editor** — a React-based web UI for drag-and-drop graph editing with an embedded code editor.

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

### Run the GPT example

```bash
python examples/gpt_graph.py
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

## MCP Server (AI agent integration)

NeuralFn ships with an [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) server that exposes every graph-editing operation as a tool. Any MCP-compatible AI client (Cursor, Claude Desktop, etc.) can build, modify, train, and inspect NeuralFn graphs programmatically — the same operations available in the visual editor.

### Prerequisites

The FastAPI backend must be running on port 8000 before starting the MCP server:

```bash
uvicorn server.app:app --reload --port 8000
```

### Configuration

Add the NeuralFn MCP server to your client configuration. The server uses inline script metadata (PEP 723), so `uv run` resolves the `mcp` dependency automatically.

**Cursor** (`.cursor/mcp.json` in the project root):

```json
{
  "mcpServers": {
    "neuralfn": {
      "command": "uv",
      "args": ["run", "server/mcp_server.py"]
    }
  }
}
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "neuralfn": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/NeuralFn", "server/mcp_server.py"]
    }
  }
}
```

### Available tools

**Graph**
| Tool | Description |
|------|-------------|
| `get_graph` | Get the full graph structure (nodes, edges, settings). |
| `replace_graph` | Replace the entire server-side graph. |
| `update_graph_settings` | Update name, training_method, runtime, or config dicts. |
| `set_io` | Set which nodes are graph inputs/outputs. |

**Nodes**
| Tool | Description |
|------|-------------|
| `list_builtins` | List available builtin neuron definitions. |
| `add_node` | Add a builtin neuron by id. |
| `add_custom_node` | Add a function node with custom Python source code and ports. |
| `add_subgraph_node` | Add an empty subgraph node (with internal input/output nodes). |
| `add_variant_node` | Add a node linked to a variant from the variant library. |
| `get_node` | Get the full details of a single node. |
| `update_node` | Update a node's name, source code, ports, or module config. |
| `delete_node` | Delete a node. |
| `update_node_positions` | Batch-update node canvas positions. |

**Edges**
| Tool | Description |
|------|-------------|
| `add_edge` | Connect two node ports with optional weight/bias. |
| `update_edge` | Update an edge's weight and/or bias. |
| `delete_edge` | Delete an edge. |

**Variants**
| Tool | Description |
|------|-------------|
| `list_variants` | List all variant families and versions. |
| `save_node_as_variant` | Save a subgraph node into the variant library. |
| `swap_node_variant` | Swap a node to a different variant version. |

**Execution & Training**
| Tool | Description |
|------|-------------|
| `execute_graph` | Run the graph with scalar inputs. |
| `execute_trace` | Run the graph and trace intermediate outputs. |
| `trace_torch` | Trace a torch graph for tensor statistics. |
| `probe_node` | Probe a node's response curve. |
| `train_start` | Start training (surrogate/evolutionary/hybrid/torch). |
| `get_training_status` | Read the active training snapshot, latest loss, and recent events. |
| `poll_training_status` | Wait for the next training update by `event_id`, or until the run finishes. |
| `train_stop` | Stop the current training run. |
| `load_gpt_template` | Build and load a GPT/Llama/MoE graph in one call. |

**Datasets**
| Tool | Description |
|------|-------------|
| `list_datasets` | List locally available datasets. |
| `download_dataset` | Download a HuggingFace dataset. |
| `load_dataset_source` | Download/load datasets and wire them into a `dataset_source` node for torch training. |
| `delete_dataset` | Delete a local dataset. |

### Example: building a graph via MCP

An AI agent can build a simple sigmoid pipeline like this (shown as sequential tool calls):

```
1. add_node(neuron_id="builtin-input", instance_id="in", position=[100, 200])
2. add_node(neuron_id="builtin-sigmoid", instance_id="sig", position=[350, 200])
3. add_node(neuron_id="builtin-output", instance_id="out", position=[600, 200])
4. add_edge(src_node="in", src_port=0, dst_node="sig", dst_port=0)
5. add_edge(src_node="sig", src_port=0, dst_node="out", dst_port=0)
6. set_io(input_ids=["in"], output_ids=["out"])
7. execute_graph(inputs={"in": [0.5]})
```

### Example: train an MoE on FineWeb via MCP

```
1. load_gpt_template(name="fineweb_moe", preset="moe", config={"n_layer": 4, "n_head": 4, "n_embd": 128, "num_experts": 4, "top_k": 2})
2. load_dataset_source(hf_path="HuggingFaceFW/fineweb", hf_split="train", max_rows=10000, seq_len=64)
3. train_start(method="torch", epochs=10, learning_rate=0.001)
4. get_training_status()
5. poll_training_status(since_event_id=0, timeout_seconds=30)
```

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

Subgraph nodes expose their ports from the nested graph’s designated `input_node_ids` and `output_node_ids`, and each graph picks its own `training_method`: `surrogate`, `evolutionary`, `frozen`, or `torch`.

## GPT / torch graphs

Use the GPT template generator when you want a causal language model that remains explorable in the editor. The templates expand into intricate graphs of token embedding, residual-mix, RMSNorm, attention, MLP (Dense or Mixture of Experts), skip-add, head, softcap, and token cross-entropy stages. Transformer blocks are represented as nested subgraphs via the Variant Library, allowing easy exploration of architecture choices. Torch graphs should use `training_method="torch"` and `runtime="torch"`. The torch trainer is CUDA-first.

Training data for GPT graphs is managed via a `dataset_source` node. Drop the node onto your graph, select Hugging Face or local datasets from the side panel, and connect its `tokens` and `targets` ports to the network's inputs. The trainer will automatically tokenize the text, dynamically adjust the model's `vocab_size` for compatibility, and handle the data batching.

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
