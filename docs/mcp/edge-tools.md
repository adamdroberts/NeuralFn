# Edge Tools

Tools for managing edges (connections) between node ports in the session graph.

---

## add_edge

Creates an edge connecting an output port of one node to an input port of another.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `project_id` | string | yes | | Project ID. |
| `session_id` | string | yes | | Session ID. |
| `src_node` | string | yes | | Source node ID. |
| `src_port` | string | yes | | Source output port name. |
| `dst_node` | string | yes | | Destination node ID. |
| `dst_port` | string | yes | | Destination input port name. |
| `weight` | float | no | `1.0` | Edge weight multiplier. |
| `bias` | float | no | `0.0` | Edge bias offset. |

**Returns:** confirmation with the created edge details (including the generated edge ID).

---

## update_edge

Modifies the weight and/or bias of an existing edge.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `edge_id` | string | yes | The edge to update. |
| `weight` | float | no | New weight value. |
| `bias` | float | no | New bias value. |

At least one of `weight` or `bias` should be provided.

**Returns:** confirmation with updated edge details.

---

## delete_edge

Removes an edge from the graph.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `edge_id` | string | yes | The edge to delete. |

**Returns:** deletion confirmation.
