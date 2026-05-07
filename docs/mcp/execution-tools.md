# Execution Tools

Tools for running forward passes, tracing, probing, loading templates, and managing training.

---

## execute_graph

Runs a single forward pass through the graph with the provided inputs.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `inputs` | dict[string, list[float]] | yes | Map of input node ID to input values. |

**Returns:** execution result with output values.

---

## execute_trace

Runs a forward pass and returns a full trace showing every node's input and output values.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `inputs` | dict[string, list[float]] | yes | Map of input node ID to input values. |

**Returns:** trace dictionary keyed by node ID, with per-node input/output snapshots.

---

## trace_torch

Runs a forward pass through the compiled Torch graph and returns a Torch-specific trace.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `inputs` | dict[string, list[float]] | yes | Map of input node ID to input values. |

**Returns:** Torch-specific trace dictionary.

---

## probe_node

Probes a single node by running multiple random forward passes and collecting activation statistics.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `project_id` | string | yes | | Project ID. |
| `session_id` | string | yes | | Session ID. |
| `node_id` | string | yes | | The node to probe. |
| `n_samples` | int | no | `1000` | Number of forward-pass samples to collect. |

**Returns:** probe statistics including mean, standard deviation, min, max, and histogram data.

---

## load_gpt_template

Loads a GPT template preset into the session, replacing the current graph.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `project_id` | string | yes | | Project ID. |
| `session_id` | string | yes | | Session ID. |
| `name` | string | no | `"gpt"` | Template name. |
| `preset` | string | no | `"nanogpt"` | Preset configuration (e.g. `"nanogpt"`, `"gpt2"`, `"llama"`, `"moe"`, `"dense_jepa_evo"`, `"moe_jepa_evo"`, `"semantic_dense_jepa_evo"`, `"semantic_moe_jepa_evo"`). |
| `config` | dict | no | | Additional configuration overrides (e.g. `n_layer`, `n_head`, `n_embd`). |

**Returns:** a summary of the generated graph.

---

## train_start

Starts a training run on the current graph.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `project_id` | string | yes | | Project ID. |
| `session_id` | string | yes | | Session ID. |
| `method` | string | no | `"surrogate"` | Training method. |
| `epochs` | int | no | `10` | Number of training epochs. |
| `learning_rate` | float | no | `0.001` | Learning rate. |
| `train_inputs` | dict | no | | Explicit training input data. |
| `train_targets` | dict | no | | Explicit training target data. |
| `dataset_names` | list[string] | no | | Names of datasets to train on (alternative to explicit inputs/targets). |

**Returns:** starts the training run and returns an initial status message. Use `poll_training_status` to monitor progress.

---

## get_training_status

Returns the latest snapshot of the current or most recent training run.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |

**Returns:** run status snapshot including current epoch, loss, and run state.

---

## poll_training_status

Blocks until a training status update is available or the timeout expires. Useful for agents that want to wait for progress without busy-polling.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `project_id` | string | yes | | Project ID. |
| `session_id` | string | yes | | Session ID. |
| `since_event_id` | string | no | | Only return events after this event ID. |
| `timeout_seconds` | int | no | `30` | Maximum time to wait before returning. |
| `interval_seconds` | int | no | `1` | Polling interval within the server. |

**Returns:** the next training status update, or the current status if the timeout expires.

---

## train_stop

Stops the currently active training run.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |

**Returns:** confirmation that the run was stopped.

---

## Experimental semantic tools

These research tools target the `jepa_semantic_hybrid` stack.

| Tool | Description |
|------|-------------|
| `reverse_engineer_to_semantic` | Encodes text to the vocab-grounded 9-D semantic space exposed by the hybrid preset. |
| `semantic_search` | Queries the experimental semantic search endpoint with a 9-D vector. |
| `train_jepa_semantic` | Starts torch training for a `jepa_semantic_hybrid` graph. |
| `generate_with_semantics` | Reserved for future semantic-conditioned generation workflows. |
