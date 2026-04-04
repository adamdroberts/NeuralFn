---
name: neuralfn-mcp
description: >-
  Build, edit, train, and inspect NeuralFn neural-network graphs using the
  neuralfn MCP server tools. Use whenever the user asks to create a neural
  network, build a model (GPT, MoE, Llama, NanoGPT, etc.), train on a dataset,
  add or wire neurons, execute a graph, manage datasets, or do anything with
  NeuralFn. The MCP server name is "neuralfn".
---

# NeuralFn MCP Server

**IMPORTANT: Do NOT read or edit the NeuralFn source code to perform graph operations.** Use the `neuralfn` MCP tools directly. Every operation the visual editor supports is available as an MCP tool. The only time you should look at source code is if the user explicitly asks about the implementation itself.

All tool responses are compact summaries safe for LLM context. No tool returns oversized graph blobs.

The backend must be running: `uvicorn server.app:app --reload --port 8000`

## Architecture quick reference

- **Nodes** have a `neuron_def` (kind: `function`, `subgraph`, or `module`) with typed I/O ports.
- **Edges** connect `src_node:src_port` to `dst_node:dst_port` with `weight` and `bias`.
- **Graph I/O**: designate nodes as graph inputs/outputs via `set_io`.
- **Variant library**: reusable subgraph templates keyed by `family/version`.
- **GPT templates**: `load_gpt_template` builds AND loads a full model graph in one call.
- Common builtin IDs: `builtin-input`, `builtin-output`, `builtin-sigmoid`, `builtin-relu`, `builtin-tanh`, `builtin-add`, `builtin-multiply`, `builtin-identity`, `builtin-gelu`, `builtin-silu`.

## End-to-end workflow: train a model on a dataset

When the user asks to "train a [model] on [dataset]", follow these steps using MCP tools only:

### Step 1 -- Build and load the model graph (one call)

```
load_gpt_template(name="my_model", preset="moe", config={
  "n_layer": 4,
  "n_head": 4,
  "n_embd": 128,
  "num_experts": 4,
  "top_k": 2
})
```

This builds the full model graph server-side and loads it. The response is a compact summary with node IDs, edge count, and variant families.

### Step 2 -- Download the dataset

```
download_dataset(hf_path="HuggingFaceFW/fineweb", hf_split="train", max_rows=10000)
```

### Step 3 -- Start training

```
train_start(
  method="torch",
  epochs=10,
  learning_rate=0.001,
  dataset_names=["HuggingFaceFW__fineweb"]
)
```

That's it -- three tool calls.

## Tool reference

### Graph

| Tool | Purpose |
|------|---------|
| `get_graph()` | Compact summary: node IDs with name/kind, edges, settings, variant families. |
| `replace_graph(graph)` | Replace the entire graph. Returns compact summary. |
| `update_graph_settings(name, training_method, runtime, ...)` | Patch settings. Returns updated settings only. |
| `set_io(input_ids, output_ids)` | Set graph input/output nodes. |

### Nodes

| Tool | Purpose |
|------|---------|
| `list_builtins()` | List builtin neuron IDs and names (compact). |
| `add_node(neuron_id, instance_id, position)` | Add a builtin node. Returns brief confirmation. |
| `add_custom_node(name, source_code, input_ports, output_ports, ...)` | Add a custom function node. Ports: `[{"name": "x", "range": [-10, 10]}]`. |
| `add_subgraph_node(name, ...)` | Add an empty subgraph with internal in/out nodes. |
| `add_variant_node(family, version, ...)` | Add a node linked to a variant. |
| `get_node(node_id)` | Inspect one node (ports, source truncated, subgraph summarized). |
| `update_node(node_id, name, source_code, input_ports, output_ports, module_config, ...)` | Patch a node's definition. |
| `delete_node(node_id)` | Remove a node. |
| `update_node_positions(positions)` | Batch-move: `{"id": [x, y], ...}`. |

### Edges

| Tool | Purpose |
|------|---------|
| `add_edge(src_node, src_port, dst_node, dst_port, weight, bias)` | Connect two ports. |
| `update_edge(edge_id, weight, bias)` | Change weight/bias. |
| `delete_edge(edge_id)` | Remove an edge. |

### Variants

| Tool | Purpose |
|------|---------|
| `list_variants()` | List families and versions. |
| `save_node_as_variant(node_id, family, version, link_node)` | Save a subgraph to the library. |
| `swap_node_variant(node_id, family, version)` | Swap to a different variant version. |

### Templates, execution, and training

| Tool | Purpose |
|------|---------|
| `load_gpt_template(name, preset, config)` | Build and load a GPT/Llama/MoE graph in one call. |
| `execute_graph(inputs)` | Run with scalar inputs: `{"node_id": [values]}`. |
| `execute_trace(inputs)` | Run and trace intermediates. |
| `trace_torch(inputs)` | Torch tensor statistics. |
| `probe_node(node_id, n_samples)` | Sample a node's response curve. |
| `train_start(method, epochs, learning_rate, train_inputs, train_targets, dataset_names)` | Start training. Methods: surrogate, evolutionary, hybrid, torch. |
| `train_stop()` | Stop training. |

### Datasets

| Tool | Purpose |
|------|---------|
| `list_datasets()` | List local datasets. |
| `download_dataset(hf_path, hf_split, max_rows)` | Download from HuggingFace. |
| `delete_dataset(ds_name)` | Delete a local dataset. |

## GPT template config keys

Common config keys for `load_gpt_template`:

| Key | Default | Description |
|-----|---------|-------------|
| `n_layer` | 4 | Number of transformer layers |
| `n_head` | 4 | Attention heads |
| `n_embd` | 128 | Embedding dimension |
| `vocab_size` | 256 | Vocabulary size (auto-adjusted by trainer) |
| `num_experts` | 4 | MoE only: number of experts |
| `top_k` | 2 | MoE only: experts per token |

Presets: `"nanogpt"`, `"gpt2"`, `"llama"`, `"moe"`.

## Other common workflows

### Build a scalar graph

```
1. add_node(neuron_id="builtin-input", instance_id="in", position=[100, 200])
2. add_node(neuron_id="builtin-sigmoid", instance_id="sig", position=[350, 200])
3. add_node(neuron_id="builtin-output", instance_id="out", position=[600, 200])
4. add_edge(src_node="in", src_port=0, dst_node="sig", dst_port=0)
5. add_edge(src_node="sig", src_port=0, dst_node="out", dst_port=0)
6. set_io(input_ids=["in"], output_ids=["out"])
7. execute_graph(inputs={"in": [0.5]})
```

### Train on inline data (XOR example)

```
train_start(
  method="surrogate",
  epochs=200,
  learning_rate=0.01,
  train_inputs=[[0,0],[0,1],[1,0],[1,1]],
  train_targets=[[0],[1],[1],[0]]
)
```
