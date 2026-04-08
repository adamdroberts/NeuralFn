# Architecture overview

NeuralFn is split into a Python graph framework, a FastAPI platform, and a React editor. This page describes how those layers compose, the main data abstractions, execution and training paths, and how templates, the server, MCP, and the editor cooperate.

For installation and first graphs in code, see [Getting started](getting-started.md). For hands-on framework tutorials, see the [Framework Guide](framework-guide/README.md).

## System layers

### `neuralfn/` — core graph framework

The installable library defines **ports**, **neuron definitions** (`NeuronDef`: Python functions, torch modules, or subgraphs), **graphs** (`NeuronGraph`, `NeuronInstance`, `Edge`), **serialization**, **variant libraries**, and **trainers** (surrogate, evolutionary, hybrid, torch). The **torch backend** compiles eligible graphs into a `CompiledTorchGraph` / `nn.Module` for tensor execution and native PyTorch optimization.

### `server/` — FastAPI platform

The backend adds **authentication** (session cookies), **projects** and **sessions**, **datasets**, **training runs**, **REST APIs** under `/api`, optional **Redis-backed live state**, SQLAlchemy persistence, Alembic migrations, and a **PersistenceWorker** that snapshots session graphs to disk. It bridges the editor and automation to the same graph payloads the core library understands.

### `editor/` — React / Vite web app

The frontend is a **React** single-page app with **React Flow** for the canvas, **Zustand** for graph and UI state, and routed pages for Editor, Datasets, Runs, Analytics, and Admin. It talks to the platform over fetch with credentials for cookie auth.

## Core abstractions

These types stack from low-level I/O to full models.

### Port

A **Port** is a typed I/O slot on a neuron: **name**, numeric **range**, **precision** (quantization step), and semantic **dtype** (`float`, `int`, `bool`). Edges move conditioned (clamped/quantized) values between ports.

### NeuronDef

A **NeuronDef** is the static definition of a neuron: a **function** neuron (wrapped from `@neuron` or source), a **module** neuron (torch stage with a forward signature), or a **subgraph** neuron (nested `NeuronGraph`). It carries input/output port lists and metadata the runtime and editor need.

### NeuronInstance

A **NeuronInstance** places a `NeuronDef` in a graph with a unique **instance id**, optional **position**, and any instance-level overrides. The graph’s executable structure is the set of instances plus edges.

### Edge

An **Edge** connects an output port on one instance to an input port on another. Scalar flow applies **weight** and **bias** (`weight * value + bias`) along with port conditioning. Many edges together define the directed wiring.

### NeuronGraph

A **NeuronGraph** is the container: **nodes** (`NeuronInstance`), **edges**, designated **input_node_ids** / **output_node_ids**, training **config**, optional **torch_config** / template metadata, and a **variant_library** mapping family keys to reusable subgraph templates. It supports both acyclic and cyclic topologies; the scalar executor either runs in topological order or settles cycles.

### Variant library

The **variant library** is a JSON-serializable map from **family** (and version) to subgraph specs so multiple nodes can share the same structural template (e.g. attention block variants). The editor and Python resolver merge libraries when loading graphs; incompatible merges fall back to inline subgraphs rather than failing hard. See [Subgraphs and variants](framework-guide/subgraphs-and-variants.md).

## Execution models

### Scalar runtime

For non-torch or mixed scalar paths, the graph executor walks the DAG in **topological order** when the graph is acyclic. **Cyclic** graphs can be executed with iterative **settling** until values stabilize (within configured limits). Values are Python scalars (or small tuples per port) flowing through `NeuronDef` callables and builtins.

### Torch runtime

For graphs whose nodes are backed by torch modules and compatible wiring, **CompiledTorchGraph** compiles a `NeuronGraph` into an **`nn.Module`** with a defined forward. Training uses ordinary PyTorch autograd on that module. Nested module neurons compile to nested submodules. Details: [Torch models](framework-guide/torch-models.md) and [Torch backend](python-sdk/torch-backend.md).

## Training methods

| Method | Role |
|--------|------|
| **Surrogate** | Probe scalar neurons, fit small MLP **surrogates**, backprop through surrogates to update **edge** weights (and related params). |
| **Evolutionary** | Treat edge weights/biases as genes; **genetic** search optimizes a fitness signal without surrogate gradients. |
| **Hybrid** | Assign **surrogate**, **evolutionary**, or **torch** (or none) **per subgraph scope** so different regions of one graph use different strategies. |
| **Torch** | **TorchTrainer** + **TorchTrainConfig** optimize the compiled torch module directly on tensor batches. |

End-to-end narratives: [Training workflows](framework-guide/training-workflows.md).

## Template system

High-level **ModelSpec** (width, depth, vocab, block and template flags) and **BlockSpec** (attention/MLP family, norm, RoPE, MoE knobs, etc.) combine with **TemplateSpec** (objective, backbone, tokenization, sparsity, compression, adapter, runtime capabilities). Builder functions such as **`build_gpt_root_graph`** expand these into a full **NeuronGraph** plus a populated **variant library** suitable for the editor and torch compiler. Preset names align across the toolbar, REST template routes, and MCP `load_gpt_template`. See [Templates and presets](framework-guide/templates-and-presets.md) and [Config](python-sdk/config.md).

## Server architecture

The FastAPI **`server.app:app`** installs CORS for credentialed browser calls, runs **`init_db`** on lifespan startup, and starts the **`PersistenceWorker`** for background snapshotting.

- **Routers** under `server/` expose `/api/bootstrap`, `/api/auth/*`, `/api/admin/*`, `/api/projects/*`, session-scoped graph and run endpoints, and datasets. Legacy wrappers may still exist in `server/routes.py` for tests.
- **Services** implement auth, workspace lifecycle, datasets, runs, and coordination with **LiveStateStore** (Redis or in-memory fallback).
- **Persistence** uses SQLAlchemy models plus the worker writing **snapshots** under `NEURALFN_SNAPSHOTS_DIR`.

**Auth**: login and bootstrap set an **HTTP-only session cookie** (`NEURALFN_SESSION_COOKIE_NAME`, TTL from `NEURALFN_SESSION_TTL_SECONDS`). Subsequent requests send the cookie; the server resolves the user and active project/session scope.

**Training progress**: run endpoints stream updates via **SSE** (server-sent events) so the UI can follow loss and steps without polling exclusively.

More detail: [Server internals](server/README.md), [Authentication](server/authentication.md), [Services](server/services.md).

## MCP bridge

The **MCP server** (see `server/mcp_server.py` and [MCP overview](mcp/README.md)) uses **FastMCP** to expose tools that call the same **REST** APIs the editor uses. Tools are **project- and session-scoped**: agents authenticate with platform credentials (`NEURALFN_MCP_EMAIL` / `NEURALFN_MCP_PASSWORD`) and operate on an existing workspace. Optional **`NEURALFN_BASE_URL`** points at a non-default API base.

## Editor architecture

The editor is **React** with **Vite**, **React Flow** for the graph canvas, and **Zustand** stores.

- **AppState** (React context) holds **auth**, bootstrap flags, and shell navigation state.
- **graphStore** (Zustand) holds the active **graph document**: nodes, edges, variant library, selection, and editor-only view state.
- **sessionSync** (and related hooks) load and autosave the session graph through the **session API**, handle **revision conflicts**, and surface **MCP agent** control banners when an external client holds the session.

API typing and fetch wrappers: [API client](editor/api-client.md). Store shape: [Store](editor/store.md). Routing and pages: [Pages](editor/pages.md).

## Related documentation

- [Getting started](getting-started.md)
- [Framework Guide](framework-guide/README.md)
- [REST API](rest-api/README.md)
- [Python SDK](python-sdk/README.md)
- [Editor](editor/README.md)
