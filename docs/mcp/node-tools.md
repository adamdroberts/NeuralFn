# Node Tools

Tools for managing nodes in the session graph. Nodes represent neurons, custom code blocks, subgraphs, or variant instances.

---

## list_builtins

Lists all built-in neuron types available in NeuralFn. This is a global tool -- it does not require project or session scope.

**Parameters:** none.

**Returns:** a list of builtin neuron descriptors.

```json
[
  { "id": "sigmoid", "name": "Sigmoid", "kind": "activation" },
  { "id": "linear", "name": "Linear", "kind": "module" },
  { "id": "add", "name": "Add", "kind": "math" }
]
```

---

## add_node

Adds a built-in neuron to the graph.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `neuron_id` | string | yes | Built-in neuron type ID (from `list_builtins`). |
| `instance_id` | string | no | Custom instance ID. Auto-generated if omitted. |
| `position` | list[float] | no | `[x, y]` canvas position. |

**Returns:** confirmation with the created node details.

---

## add_custom_node

Adds a node with user-defined Python source code.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `name` | string | yes | Display name for the node. |
| `source_code` | string | yes | Python source code for the node's forward function. |
| `input_ports` | list | no | Input port definitions. |
| `output_ports` | list | no | Output port definitions. |
| `instance_id` | string | no | Custom instance ID. |
| `position` | list[float] | no | `[x, y]` canvas position. |

**Returns:** confirmation with the created node details.

---

## add_subgraph_node

Adds an empty subgraph (compound) node that can contain child nodes and edges.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `name` | string | no | Display name. Defaults to `"subgraph"`. |
| `instance_id` | string | no | Custom instance ID. |
| `position` | list[float] | no | `[x, y]` canvas position. |

**Returns:** confirmation with the created subgraph node details.

---

## add_variant_node

Adds a node from the variant library using a family and version identifier.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `family` | string | yes | Variant family name (e.g. `"attention"`, `"mlp"`). |
| `version` | string | yes | Variant version within the family (e.g. `"default"`, `"moe"`). |
| `instance_id` | string | no | Custom instance ID. |
| `position` | list[float] | no | `[x, y]` canvas position. |

**Returns:** confirmation with the created variant node details.

---

## get_node

Returns full details for a single node, including its neuron definition, ports, position, and subgraph contents (if applicable).

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `node_id` | string | yes | The node to retrieve. |

**Returns:** complete node object.

---

## update_node

Updates properties of an existing node. Only provided fields are changed.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `node_id` | string | yes | The node to update. |
| `name` | string | no | New display name. |
| `source_code` | string | no | Updated Python source code. |
| `input_ports` | list | no | Updated input port definitions. |
| `output_ports` | list | no | Updated output port definitions. |
| `module_config` | dict | no | Module configuration (e.g. hidden size, dropout). |
| `input_aliases` | dict | no | Input port alias mapping for subgraph nodes. |
| `output_aliases` | dict | no | Output port alias mapping for subgraph nodes. |

**Returns:** confirmation with updated node details.

---

## delete_node

Removes a node and all edges connected to it.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `node_id` | string | yes | The node to delete. |

**Returns:** deletion confirmation.

---

## update_node_positions

Batch-updates the canvas positions of multiple nodes.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `positions` | dict[string, list[float]] | yes | Map of node ID to `[x, y]` position. |

**Returns:** confirmation of the position updates.
