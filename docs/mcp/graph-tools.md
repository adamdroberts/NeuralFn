# Graph Tools

Tools for reading, replacing, and configuring the session graph.

---

## get_graph

Returns a compact summary of the current graph, including nodes, edges, I/O assignments, and settings.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |

**Returns:** a compact graph summary with node list, edge list, input/output IDs, and graph settings.

---

## replace_graph

Replaces the session's entire graph with the provided graph object. This is a full overwrite -- all existing nodes, edges, and settings are replaced.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `graph` | dict | yes | Complete graph object (nodes, edges, input_ids, output_ids, settings). |

**Returns:** a summary of the new graph.

---

## update_graph_settings

Updates one or more graph-level settings without touching nodes or edges.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `name` | string | no | Graph display name. |
| `training_method` | string | no | Default training method (`"surrogate"`, `"torch"`, etc.). |
| `runtime` | string | no | `scalar`, `torch`, or fail-closed `native-cuda`. |
| `surrogate_config` | dict | no | Surrogate training configuration. |
| `evo_config` | dict | no | Evolutionary training configuration. |
| `torch_config` | dict | no | Torch compilation and training configuration. |

**Returns:** the updated graph settings.

Selecting `native-cuda` records intent only. `train_start(runtime="native-cuda",
...)` performs the authoritative Native IR and reviewed-adapter preflight. Do
not infer native support from a successful settings update, and do not add a
`dataset_source` node to an exact reviewed native graph; pass one cached project
dataset alias directly to `train_start`.

---

## set_io

Designates which nodes serve as the graph's inputs and outputs.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `input_ids` | list[string] | yes | Node IDs to mark as graph inputs. |
| `output_ids` | list[string] | yes | Node IDs to mark as graph outputs. |

**Returns:** confirmation with the updated I/O assignment.
