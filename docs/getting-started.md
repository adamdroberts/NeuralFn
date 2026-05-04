# Getting started

This guide walks through installing NeuralFn, running the bundled examples, starting the web platform, and building your first graphs in Python.

## Prerequisites

- **Python** 3.10 or newer, with `pip`
- **Node.js** and **pnpm** (the editor uses `pnpm`; see `editor/pnpm-lock.yaml`)

## Install Python dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

## Install the NeuralFn SDK package

If you want to build against `neuralfn` from another project, install the repo
in editable mode:

From the NeuralFn repo root:

```bash
pip install -e .
```

From a sibling project checked out next to the repo:

```bash
pip install -e ../NeuralFn
```

The editable package install includes the shipped semantic vocabulary JSON
files under `neuralfn/data/semantic/`, so SDK consumers can use the semantic
routing presets without copying those assets manually.

This pulls in the core stack used by the library and platform, including **torch**, **numpy**, **fastapi**, **uvicorn**, **networkx**, **pydantic**, **sqlalchemy**, **alembic**, **redis**, **datasets**, **tiktoken**, and **mcp** (plus helpers such as **python-multipart** and **PyMySQL**).

## Install the editor

```bash
cd editor && pnpm install
```

## Run examples

Run these from the repository root (with the virtual environment activated if you use one):

```bash
python examples/xor_graph.py
```

Scalar graph with surrogate and evolutionary training on a small XOR network.

```bash
python examples/gpt_graph.py
```

Builds a torch GPT-style graph via `build_gpt_root_graph` and trains with `TorchTrainer`.

```bash
python examples/nested_hybrid_graph.py
```

Nested subgraphs with hybrid (per-subgraph) training.

## Start the platform

**Backend** (terminal 1):

```bash
uvicorn server.app:app --reload --port 8000
```

**Frontend** (terminal 2):

```bash
cd editor && pnpm dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

## First-run workflow

1. On first launch, the client calls `/api/bootstrap`. If no users exist, the login screen offers **Create Admin Workspace**.
2. Creating the first admin also creates a **default project** and **editor session**, then signs you in with an **HTTP-only session cookie**.
3. Later visits use `/api/auth/login`; the app restores the last active **project** and **session** stored on the session.
4. Any signed-in user can create additional personal projects from the shell; new projects get a **Main session** and become the active workspace.
5. **Admin** users can manage users and project memberships from the Admin surface (`/app/admin`).

## Your first graph in Python code

Below is a minimal XOR-style pipeline: custom `@neuron` nodes, `NeuronInstance` placement, `Edge` wiring, graph I/O IDs, forward execution, and **SurrogateTrainer** training. It follows the same patterns as `examples/xor_graph.py`.

```python
import numpy as np
from neuralfn import (
    BuiltinNeurons,
    Port,
    neuron,
    NeuronGraph,
    NeuronInstance,
    Edge,
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

    for node in (in1, in2, h1, h2, a1, a2, h3, a3, out):
        g.add_node(node)

    g.input_node_ids = ["in1", "in2"]
    g.output_node_ids = ["out"]

    edges = [
        Edge(id="e1", src_node="in1", src_port=0, dst_node="h1", dst_port=0, weight=1.0, bias=0.0),
        Edge(id="e2", src_node="in2", src_port=0, dst_node="h1", dst_port=1, weight=1.0, bias=0.0),
        Edge(id="e3", src_node="in1", src_port=0, dst_node="h2", dst_port=0, weight=1.0, bias=0.0),
        Edge(id="e4", src_node="in2", src_port=0, dst_node="h2", dst_port=1, weight=1.0, bias=0.0),
        Edge(id="e5", src_node="h1", src_port=0, dst_node="a1", dst_port=0, weight=1.0, bias=0.0),
        Edge(id="e6", src_node="h2", src_port=0, dst_node="a2", dst_port=0, weight=1.0, bias=0.0),
        Edge(id="e7", src_node="a1", src_port=0, dst_node="h3", dst_port=0, weight=1.0, bias=0.0),
        Edge(id="e8", src_node="a2", src_port=0, dst_node="h3", dst_port=1, weight=1.0, bias=0.0),
        Edge(id="e9", src_node="h3", src_port=0, dst_node="a3", dst_port=0, weight=1.0, bias=0.0),
        Edge(id="e10", src_node="a3", src_port=0, dst_node="out", dst_port=0, weight=1.0, bias=0.0),
    ]
    for e in edges:
        g.add_edge(e)

    return g


if __name__ == "__main__":
    g = build_xor_graph()

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    out_before = g.execute({"in1": (0.0,), "in2": (1.0,)})
    print("sample forward:", out_before["out"])

    cfg = TrainConfig(epochs=300, learning_rate=0.01)
    trainer = SurrogateTrainer(g, cfg)
    trainer.train(X, Y)
```

- **`execute`**: pass a dict mapping **input node ids** to tuples of port values (order matches input ports). Returned dict keys are **output node ids**.

For evolutionary training on the same graph, see `EvolutionaryTrainer` and `EvoConfig` in `examples/xor_graph.py`.

## Your first torch model

A minimal GPT-style graph trains tensor-native module nodes through the PyTorch backend:

```python
from neuralfn import TorchTrainConfig, TorchTrainer, build_gpt_root_graph
from neuralfn.config import build_llama_spec

spec = build_llama_spec(
    vocab_size=16,
    num_layers=4,
    model_dim=32,
    num_heads=4,
    num_kv_heads=2,
    mlp_mult=2,
    tie_embeddings=True,
    logit_softcap=30.0,
)
graph = build_gpt_root_graph(model_spec=spec)
trainer = TorchTrainer(
    graph,
    TorchTrainConfig(epochs=10, learning_rate=5e-3, batch_size=2, device="cpu"),
)
losses = trainer.train(
    [[0, 1, 2, 3], [1, 2, 3, 4]],
    [[1, 2, 3, 4], [2, 3, 4, 5]],
)
print({"initial": losses[0], "final": losses[-1]})
```

This mirrors `examples/gpt_graph.py`. Template fields (`ModelSpec`, `BlockSpec`, `TemplateSpec`) and presets are covered in the [Framework Guide](framework-guide/templates-and-presets.md) and [Architecture](architecture.md) template section.

## Environment variables (platform)

| Variable | Purpose | Default |
|----------|---------|---------|
| `NEURALFN_DATABASE_URL` | SQLAlchemy URL for users, projects, sessions, datasets, runs | `sqlite:///…/neuralfn.db` under the repo root |
| `NEURALFN_REDIS_URL` | Redis URL for shared live state (session graph cache, run events). If Redis is unreachable, the server falls back to in-memory live state | `redis://localhost:6379/1` |
| `NEURALFN_CREATE_SCHEMA_ON_STARTUP` | When not `0`, create DB tables on startup | `1` |
| `NEURALFN_SNAPSHOTS_DIR` | Directory for persisted session snapshots | `server/session_snapshots` |
| `NEURALFN_ARTIFACTS_DIR` | Directory for saved artifacts | `~/NeuralFn/artifacts` |
| `NEURALFN_ALLOW_ORIGINS` | Comma-separated CORS origins (must include the editor origin when using cookies) | `http://127.0.0.1:5173,http://localhost:5173` |
| `NEURALFN_SESSION_COOKIE_NAME` | HTTP session cookie name | `neuralfn_session` |
| `NEURALFN_SESSION_TTL_SECONDS` | Session lifetime in seconds | `1209600` |

MCP clients also use `NEURALFN_MCP_EMAIL`, `NEURALFN_MCP_PASSWORD`, and optionally `NEURALFN_BASE_URL`; see the root [README](../README.md#mcp-server-ai-agent-integration).

## Next steps

- **[Architecture](architecture.md)** — how `neuralfn/`, `server/`, and `editor/` fit together, execution and training models, and the template pipeline.
- **[Framework Guide](framework-guide/README.md)** — tutorials for neurons, graphs, variants, torch models, templates, training, inference, and datasets.
