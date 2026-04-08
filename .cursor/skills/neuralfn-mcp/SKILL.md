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

For complete MCP tool reference, see [docs/mcp/](../../../docs/mcp/README.md). For the Python SDK, see the `neuralfn-python-sdk` and `neuralfn-torch` skills.

Full API documentation lives in the repo at `docs/` ([index](../../../docs/README.md)). For a single-file LLM-ready dump of all docs, see [llms-full.txt](../../../llms-full.txt).

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
load_gpt_template(name="my_model", preset="llama", config={
  "n_layer": 4,
  "n_head": 4,
  "n_embd": 128,
  "num_kv_heads": 2
})
```

### Step 2 -- Download and wire the dataset (one call)

```
load_dataset_source(
  hf_path="HuggingFaceFW/fineweb",
  hf_split="train",
  max_rows=10000,
  seq_len=64
)
```

This downloads the dataset AND wires it into the graph's `dataset_source` node.

### Step 3 -- Start training

```
train_start(method="torch", epochs=10, learning_rate=0.001)
```

### Step 4 -- Monitor progress

```
poll_training_status(since_event_id=0, timeout_seconds=30)
```

Or check current state:

```
get_training_status()
```

That's it -- four tool calls for a complete train run.

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
| `get_training_status()` | Read the active training snapshot, latest loss, and recent events. |
| `poll_training_status(since_event_id, timeout_seconds, interval_seconds)` | Wait for the next training update or until the run finishes. |
| `train_stop()` | Stop training. |

### Datasets

| Tool | Purpose |
|------|---------|
| `list_datasets()` | List datasets visible to the project. |
| `download_dataset(hf_path, hf_split, max_rows, ...)` | Download from HuggingFace into project catalog. |
| `load_dataset_source(hf_path, dataset_names, seq_len, ...)` | Download/load datasets AND wire into graph's dataset_source node. |
| `set_dataset_access(ds_name, project_ids)` | Update which projects can use a dataset. |
| `delete_dataset(ds_name)` | Delete a local dataset. |

## All 16 GPT template presets

| Preset | Architecture | Objective |
|--------|-------------|-----------|
| `nanogpt` | GPT-2 style (LayerNorm, GELU, absolute pos) | AR |
| `gpt2` | GPT-2 (with bias) | AR |
| `llama` | LLaMA (RMSNorm, SwiGLU, RoPE, GQA) | AR |
| `moe` / `mixllama` | LLaMA + Mixture of Experts | AR |
| `llama_fast` | LLaMA with torch.compile | AR |
| `mixllama_fast` | MoE with torch.compile | AR |
| `jamba` | Attention + Mamba hybrid, MoE | AR |
| `ternary_b158` | BitNet b1.58 ternary weights | AR |
| `seq2seq` | Encoder-decoder, MoE | Seq2Seq |
| `diffusion` | Discrete diffusion with denoising head | Diffusion |
| `ttt_llama` | Test-Time Training layers | AR |
| `llm_jepa` | JEPA with EMA target encoder | JEPA |
| `hnet_lm` | Raw-byte input, byte patches | AR |
| `universal_llama` | ACT-based universal transformer | AR |
| `llama_megakernel` | Fused attention, max-autotune compile | AR |
| `kv_pca_llama` | PCA-compressed KV cache | AR |

## GPT template config keys

| Key | Default | Description |
|-----|---------|-------------|
| `n_layer` | 4 | Number of transformer layers |
| `n_head` | 4 | Attention heads |
| `n_embd` | 128 | Embedding dimension |
| `vocab_size` | 256 | Vocabulary size (auto-adjusted by trainer) |
| `num_kv_heads` | 2 | GQA key/value heads |
| `mlp_multiplier` | varies | MLP hidden dimension multiplier |
| `multiple_of` | 256 | Round MLP width to this multiple |
| `experts` | 8 | MoE: number of experts |
| `top_k` | 2 | MoE: active experts per token |
| `router_aux_loss_coef` | 0.01 | MoE load-balance loss coefficient |
| `dropout_p` | 0.0 | Dropout rate |
| `tie_embeddings` | varies | Tie embedding and LM head weights |
| `logit_softcap` | 0.0 | Tanh softcap value (>0 enables) |
| `ttt_hidden_dim` | 32 | TTT hidden dimension |
| `byte_patch_size` | 4 | H-Net byte patch size |
| `byte_patch_stride` | 4 | H-Net byte patch stride |
| `max_recurrence_steps` | 4 | Universal TX max steps |
| `halt_epsilon` | 0.01 | Universal TX halt threshold |

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

### Non-AR template workflows

**Seq2Seq:**
```
load_gpt_template(name="s2s", preset="seq2seq", config={"n_layer": 4, "n_embd": 128, "experts": 4})
```
Dataset roles: `enc_tokens`, `dec_tokens`, `targets`.

**Diffusion:**
```
load_gpt_template(name="diff", preset="diffusion", config={"n_layer": 4, "n_embd": 128})
```
Dataset role: `tokens` only.

**JEPA:**
```
load_gpt_template(name="jepa", preset="llm_jepa", config={"n_layer": 4, "n_embd": 128})
```
Dataset role: `tokens` only. Uses EMA target encoder, supports `jepa_mask_strategy` ("random" or "block").

**H-Net (byte-level):**
```
load_gpt_template(name="hnet", preset="hnet_lm", config={"n_layer": 4, "n_embd": 128})
```
Uses raw bytes (vocab_size=256), byte patch embedding.

## Experimental Tools

The following MCP tools are **[Experimental]** and target the `jepa_semantic_hybrid` semantic stack. They may change or be removed.

| Tool | Parameters [Experimental] | Purpose |
|------|---------------------------|---------|
| `reverse_engineer_to_semantic` | `project_id`, `session_id`, `text` | Encode `text` to a 15-D semantic vector using the session graph’s JEPA semantic path. |
| `semantic_search` | `project_id`, `session_id`, `vector` (list of floats), `k` (default `10`) | k-nearest-neighbour lookup for a 15-D `vector`. |
| `train_jepa_semantic` | `project_id`, `session_id`, `dataset_names` (optional list), `epochs` (default `10`), `learning_rate` (default `3e-4`) | Start torch training (same entry as `train_start`) intended for graphs using the **[Experimental]** `jepa_semantic_hybrid` template. |
| `generate_with_semantics` | `project_id`, `session_id`, `prompt`, `target_vector` (optional 15-D list), `max_tokens` (default `100`) | Generate with the attentionless semantic decoder; optional `target_vector` steers toward a semantic target. |

**Disclaimer [Experimental]:** These tools are research prototypes; prefer the stable graph/dataset/training tools for production-like workflows.
