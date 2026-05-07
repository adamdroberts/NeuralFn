# Sessions

Sessions live inside a project and hold the neural-network graph, execution context, and training runs. Most graph-editing and execution endpoints are scoped to a session.

All endpoints in this section are prefixed with:

```
/api/projects/{project_id}/sessions
```

**Authentication:** required for all endpoints.

---

## Session CRUD

### GET /

Lists all sessions in the project.

**Response:** array of session objects.

```json
[
  { "id": "s_001", "name": "default", "project_id": "p_xyz" },
  { "id": "s_002", "name": "experiment-a", "project_id": "p_xyz" }
]
```

### POST /

Creates a new session.

**Request Body:**

```json
{
  "name": "experiment-b",
  "description": "Testing attention variants"
}
```

| Field | Type | Required |
|-------|------|----------|
| `name` | string | yes |
| `description` | string | no |

**Response:** the created session object.

### POST /{session_id}/activate

Sets this session as the user's active session (also sets the parent project as active).

**Response:**

```json
{
  "project": { "id": "p_xyz", "name": "My Project" },
  "session": { "id": "s_002", "name": "experiment-a" }
}
```

---

## Experimental semantic endpoints

These endpoints are research-only surfaces for the semantic routing stack.

### POST /{session_id}/semantic/encode

Returns a placeholder semantic response keyed by the vocab-grounded dimension names.

### POST /{session_id}/semantic/search

Accepts a semantic vector and returns placeholder nearest-neighbour rows.

### GET /{session_id}/semantic/dimensions

Returns the semantic dimension metadata used by the semantic routing presets.

`num_topics` is dynamic and comes from the current semantic vocabulary reference. It
is no longer a fixed 40 for each routed dimension.

**Response element shape:**

```json
{
  "index": 0,
  "name": "entity_type",
  "meaning": "who or what (person, object, abstract)",
  "expert_id": 0,
  "num_topics": 266
}
```

`expert_id` is `null` for the derived `taxonomy_hash` slot because it does not own an expert.

### POST /{session_id}/semantic/generate

Reserved for future semantic-conditioned generation. The current response remains a placeholder.

### GET /{session_id}

Returns session details including graph metadata (node/edge counts, revision).

---

## Graph CRUD

### GET /{session_id}/graph

Returns the full graph and its current revision number.

**Response:**

```json
{
  "graph": {
    "nodes": { "...": "..." },
    "edges": { "...": "..." },
    "input_ids": ["input_0"],
    "output_ids": ["output_0"],
    "settings": {}
  },
  "revision": 12
}
```

### PUT /{session_id}/graph

Replaces the entire graph. Uses optimistic concurrency -- the client must send the revision it last read.

**Request Body:**

```json
{
  "graph": { "...full graph object..." },
  "expected_revision": 12,
  "persist_snapshot": false,
  "snapshot_reason": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `graph` | object | yes | Complete graph object. |
| `expected_revision` | int | yes | Must match the server's current revision. |
| `persist_snapshot` | bool | no | If `true`, saves a named snapshot in history. |
| `snapshot_reason` | string | no | Human-readable label for the snapshot. |

**Response:** the updated graph and new revision.

**Error (409):** revision conflict. See [README](README.md) for conflict response format.

### PUT /{session_id}/graph/io

Sets the graph's input and output node IDs without replacing the whole graph.

**Request Body:**

```json
{
  "input_ids": ["input_0"],
  "output_ids": ["output_0"]
}
```

**Response:**

```json
{
  "result": "ok",
  "revision": 13
}
```

---

## Node Mutations

### POST /{session_id}/nodes

Adds a node to the graph.

**Request Body:** a `NodeModel` object specifying the neuron definition, position, and instance ID.

**Response:**

```json
{
  "node": { "...node object..." },
  "revision": 14
}
```

### PUT /{session_id}/nodes/{node_id}

Updates an existing node's neuron definition.

**Request Body:** a `NeuronDefModel` object with the fields to update.

**Response:**

```json
{
  "node": { "...updated node..." },
  "revision": 15
}
```

### DELETE /{session_id}/nodes/{node_id}

Removes a node and all its connected edges.

**Response:**

```json
{
  "status": "deleted",
  "revision": 16
}
```

---

## Edge Mutations

### POST /{session_id}/edges

Adds an edge between two node ports.

**Request Body:** an `EdgeModel` object specifying source node/port, destination node/port, weight, and bias.

**Response:**

```json
{
  "edge": { "...edge object..." },
  "revision": 17
}
```

### PUT /{session_id}/edges/{edge_id}

Updates an edge's weight and/or bias.

**Request Body:**

```json
{
  "weight": 0.5,
  "bias": 0.1
}
```

**Response:**

```json
{
  "edge": { "...updated edge..." },
  "revision": 18
}
```

### DELETE /{session_id}/edges/{edge_id}

Removes an edge.

**Response:**

```json
{
  "status": "deleted",
  "revision": 19
}
```

---

## Execution

### POST /{session_id}/execute

Runs a forward pass through the graph.

**Request Body:**

```json
{
  "inputs": { "input_0": [1.0, 2.0, 3.0] },
  "dataset_names": ["my_dataset"],
  "seq_len": 64,
  "preview_batch_size": 4
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `inputs` | dict | yes | Map of input node ID to input values. |
| `dataset_names` | list[string] | no | Datasets to bind for data-source nodes. |
| `seq_len` | int | no | Sequence length for sequence-based inputs. |
| `preview_batch_size` | int | no | Batch size for preview execution. |

**Response:** execution result dictionary containing output values, intermediate state, and metadata.

### POST /{session_id}/execute-trace

Runs a forward pass and returns a trace of every node's input/output values.

**Request Body:** same as `ExecuteRequest`.

**Response:** trace dictionary keyed by node ID.

### POST /{session_id}/trace/torch

Runs a forward pass through the compiled Torch backend and returns a trace.

**Request Body:** same as `ExecuteRequest`.

**Response:** Torch-specific trace dictionary.

### POST /{session_id}/probe/{node_id}

Probes a single node by running multiple forward passes and collecting statistics.

**Query Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_samples` | 1000 | Number of samples to collect. |

**Response:** probe statistics (mean, std, min, max, histogram, etc.).

---

## Templates

### POST /{session_id}/templates/gpt/apply

Applies a GPT template to the session's graph, replacing its contents.
The `preset` name uses the same catalog as the Python SDK, editor toolbar, and MCP `load_gpt_template`, including experimental JEPA Evo presets such as `dense_jepa_evo`, `moe_jepa_evo`, `semantic_dense_jepa_evo`, and `semantic_moe_jepa_evo`.

**Request Body:**

```json
{
  "name": "gpt",
  "config": {
    "preset": "nanogpt",
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384
  }
}
```

**Response:**

```json
{
  "revision": 20,
  "graph": { "...generated graph..." }
}
```

---

## Datasets (Session-scoped)

### POST /{session_id}/datasets/load

Loads a dataset into the graph as a data-source node.

**Request Body:** `LoadDatasetRequest` specifying dataset name, HuggingFace path, column mapping, sequence length, and target node ID.

**Response:**

```json
{
  "result": "loaded",
  "revision": 21,
  "graph": { "...updated graph..." }
}
```

---

## Agent

### GET /{session_id}/agent/status

Returns whether the AI agent loop is active for this session.

**Response:**

```json
{
  "active": false
}
```

### POST /{session_id}/agent/status

Enables or disables the agent loop.

**Request Body:**

```json
{
  "active": true
}
```

**Response:**

```json
{
  "active": true
}
```
