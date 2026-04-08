# Variant Tools

Tools for managing the variant library. Variants allow saving node configurations (including subgraph internals) as reusable templates organized by family and version.

---

## list_variants

Lists all variant families and their versions in the current session's variant library.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |

**Returns:** a dictionary of variant families, each containing a list of version names.

```json
{
  "attention": ["default", "multi_query"],
  "mlp": ["default", "gated"],
  "attn_block": ["default"]
}
```

---

## save_node_as_variant

Saves a node's current configuration as a variant in the library. If the node is a subgraph, its full internal structure (child nodes and edges) is saved.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `project_id` | string | yes | | Project ID. |
| `session_id` | string | yes | | Session ID. |
| `node_id` | string | yes | | The node to save. |
| `family` | string | yes | | Variant family name (e.g. `"attention"`). |
| `version` | string | yes | | Version label (e.g. `"multi_query"`). |
| `link_node` | bool | no | `true` | If `true`, the node is linked to this variant entry so future swaps apply to it. |

**Returns:** confirmation that the variant was saved.

---

## swap_node_variant

Replaces a node's internals with a different variant version from the same or a different family. The node keeps its ID and external edges but its neuron definition (or subgraph contents) is replaced.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `node_id` | string | yes | The node to update. |
| `family` | string | yes | Target variant family. |
| `version` | string | yes | Target variant version. |

**Returns:** confirmation with the updated node details.
