# NeuralFn Documentation

NeuralFn is a graph-native neural network framework where every neuron -- from a scalar sigmoid to a full causal-attention block -- is a node in a directed graph with typed I/O ports. Graphs can be nested, versioned through a variant library, trained with four different methods, and executed on both a scalar runtime and a PyTorch tensor backend.

This documentation covers every part of the platform: the core Python SDK, the web editor, the REST API, the MCP integration for AI agents, and the server internals.

## Table of contents

### Getting started

- [**Getting Started**](getting-started.md) -- Installation, quickstart, and your first graph.
- [**Architecture**](architecture.md) -- System overview, module map, and how the pieces fit together.

### Framework guide (how to build with NeuralFn)

Tutorial-oriented walkthroughs with runnable Python code. Start here if you want to build neural networks with NeuralFn programmatically.

- [**Overview**](framework-guide/README.md) -- What NeuralFn is, how to think in graphs.
- [**Defining neurons**](framework-guide/defining-neurons.md) -- `@neuron`, `neuron_from_source`, `module_neuron`, `subgraph_neuron`.
- [**Building graphs**](framework-guide/building-graphs.md) -- `NeuronGraph`, nodes, edges, wiring, execution.
- [**Subgraphs and variants**](framework-guide/subgraphs-and-variants.md) -- Nested composition, the variant library, alias resolution.
- [**Torch models**](framework-guide/torch-models.md) -- Module neurons, `CompiledTorchGraph`, tensor-native architectures.
- [**Templates and presets**](framework-guide/templates-and-presets.md) -- All 17 presets, `ModelSpec`, `BlockSpec`, `TemplateSpec`.
- [**Training workflows**](framework-guide/training-workflows.md) -- Surrogate, evolutionary, hybrid, and torch training end-to-end.
- [**Inference and export**](framework-guide/inference-and-export.md) -- Weight I/O, quantization, autoregressive generation.
- [**Datasets**](framework-guide/datasets.md) -- Download, upload, wiring to `dataset_source`, byte vs tokenized paths.

### Python SDK reference

Complete API reference for every public class, function, method, and type in the `neuralfn` package.

- [**Overview**](python-sdk/README.md)
- [**Port**](python-sdk/port.md) -- `Port` dataclass.
- [**Neuron**](python-sdk/neuron.md) -- `NeuronDef`, `@neuron`, `neuron_from_source`, `module_neuron`, `subgraph_neuron`.
- [**Graph**](python-sdk/graph.md) -- `NeuronGraph`, `NeuronInstance`, `Edge`.
- [**Builtins**](python-sdk/builtins.md) -- `BuiltinNeurons` class and the full 58-entry catalog.
- [**Config**](python-sdk/config.md) -- `TemplateSpec`, `BlockSpec`, `ModelSpec`, type aliases, all preset builders.
- [**Torch backend**](python-sdk/torch-backend.md) -- `CompiledTorchGraph`, `TorchTrainConfig`, `TorchTrainer`, all `*Stage` modules.
- [**Torch templates**](python-sdk/torch-templates.md) -- Template and graph builder functions.
- **Training:**
  - [**Overview**](python-sdk/training/README.md) -- When to use each training method.
  - [**Surrogate**](python-sdk/training/surrogate.md) -- `SurrogateModel`, `probe_neuron`, `build_surrogates`.
  - [**Trainer**](python-sdk/training/trainer.md) -- `TrainConfig`, `SurrogateTrainer`.
  - [**Evolutionary**](python-sdk/training/evolutionary.md) -- `EvoConfig`, `EvolutionaryTrainer`.
  - [**Hybrid**](python-sdk/training/hybrid.md) -- `HybridConfig`, `HybridTrainer`, `GraphScope`.
- [**Inference**](python-sdk/inference.md) -- `export_to_pt`, `import_from_pt`, quantization, `InferenceCache`.
- [**Serialization**](python-sdk/serialization.md) -- `save_graph`, `load_graph`.

### REST API reference

- [**Overview**](rest-api/README.md) -- Auth model, base URL, error shapes.
- [**Bootstrap**](rest-api/bootstrap.md) -- `GET /api/bootstrap`.
- [**Auth**](rest-api/auth.md) -- Login, logout, session management.
- [**Admin**](rest-api/admin.md) -- User and membership management.
- [**Projects**](rest-api/projects.md) -- Project CRUD and analytics.
- [**Sessions**](rest-api/sessions.md) -- Session/graph CRUD, node/edge mutations, execution, tracing, templates, agent status.
- [**Datasets**](rest-api/datasets.md) -- Dataset catalog, download, upload, access grants.
- [**Runs**](rest-api/runs.md) -- Training runs and SSE streaming.

### MCP tools reference (AI agent integration)

- [**Overview**](mcp/README.md) -- Setup, auth, scope rules.
- [**Graph tools**](mcp/graph-tools.md) -- `get_graph`, `replace_graph`, `update_graph_settings`, `set_io`.
- [**Node tools**](mcp/node-tools.md) -- `add_node`, `add_custom_node`, `add_subgraph_node`, `add_variant_node`, `get_node`, `update_node`, `delete_node`.
- [**Edge tools**](mcp/edge-tools.md) -- `add_edge`, `update_edge`, `delete_edge`.
- [**Variant tools**](mcp/variant-tools.md) -- `list_variants`, `save_node_as_variant`, `swap_node_variant`.
- [**Execution and training tools**](mcp/execution-tools.md) -- `execute_graph`, `trace_torch`, `train_start`, `load_gpt_template`, and more.
- [**Dataset tools**](mcp/dataset-tools.md) -- `list_datasets`, `download_dataset`, `load_dataset_source`, `set_dataset_access`.

### Server internals

- [**Overview**](server/README.md) -- Server architecture.
- [**Configuration**](server/configuration.md) -- Environment variables, `Settings` dataclass.
- [**Database**](server/database.md) -- ORM models, schema, Alembic migrations.
- [**Authentication**](server/authentication.md) -- `AuthService`, session cookies, security.
- [**Services**](server/services.md) -- `WorkspaceService`, `RunService`, `DatasetService`, `LiveStateStore`.
- [**Models**](server/models.md) -- All Pydantic request/response models.

### Editor (frontend)

- [**Overview**](editor/README.md) -- Editor architecture.
- [**API client**](editor/api-client.md) -- TypeScript `api` object, DTO interfaces, `ApiError`.
- [**Store**](editor/store.md) -- Zustand `useGraphStore`, state shape, actions, selectors.
- [**Graph utilities**](editor/graph-utils.md) -- `FlowNodeData`, graph conversion, variant merge, path helpers.
- [**Components**](editor/components.md) -- React components and their props.
- [**Pages**](editor/pages.md) -- Route pages, `AppState`, session sync.

### Other

- [**Testing**](testing.md) -- Test suite overview and how to run tests.
- [**Agent skills**](agent-skills.md) -- AI coding agent skills for Cursor and Codex.
