# neuralfn.graph

Core graph data model: `NeuronInstance`, `Edge`, and `NeuronGraph`.

---

## Class: NeuronInstance

```python
@dataclass
class NeuronInstance:
    neuron_def: NeuronDef
    instance_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    position: tuple[float, float] = (0.0, 0.0)
```

A placed instance of a `NeuronDef` inside a graph. Multiple instances can share the same `NeuronDef`.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `neuron_def` | `NeuronDef` | *(required)* | The neuron type definition |
| `instance_id` | `str` | *(auto-generated)* | Unique 12-char hex identifier |
| `position` | `tuple[float, float]` | `(0.0, 0.0)` | Position in the visual editor |

### Properties

#### `name -> str`

Delegates to `self.neuron_def.name`.

### Methods

#### `to_dict() -> dict`

Serialize to a JSON-compatible dictionary.

#### `NeuronInstance.from_dict(d: dict) -> NeuronInstance` *(classmethod)*

Reconstruct from a serialized dictionary.

---

## Class: Edge

```python
@dataclass
class Edge:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    src_node: str = ""
    src_port: int = 0
    dst_node: str = ""
    dst_port: int = 0
    weight: float = 1.0
    bias: float = 0.0
```

A weighted connection between two neuron-instance ports. Edge transforms are applied as `value * weight + bias`.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | *(auto-generated)* | Unique 12-char hex identifier |
| `src_node` | `str` | `""` | Source node instance_id |
| `src_port` | `int` | `0` | Source output port index |
| `dst_node` | `str` | `""` | Destination node instance_id |
| `dst_port` | `int` | `0` | Destination input port index |
| `weight` | `float` | `1.0` | Multiplicative weight |
| `bias` | `float` | `0.0` | Additive bias |

### Methods

#### `transform(value: float) -> float`

Apply the edge's linear transform: `value * weight + bias`.

#### `to_dict() -> dict`

Serialize to a JSON-compatible dictionary.

#### `Edge.from_dict(d: dict) -> Edge` *(classmethod)*

Reconstruct from a serialized dictionary.

---

## Class: NeuronGraph

```python
class NeuronGraph:
    def __init__(
        self,
        *,
        name: str = "graph",
        training_method: str = "surrogate",
        runtime: str = "scalar",
        surrogate_config: dict | None = None,
        evo_config: dict | None = None,
        torch_config: dict | None = None,
        variant_library: dict[str, dict[str, NeuronGraph]] | None = None,
    ) -> None
```

A directed graph of `NeuronInstance` nodes connected by weighted `Edge` connections. Supports both DAG (feedforward) and cyclic (recurrent) execution.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"graph"` | Human-readable graph name |
| `training_method` | `str` | `"surrogate"` | Training method: `"surrogate"`, `"evolutionary"`, `"torch"`, or `"frozen"` |
| `runtime` | `str` | `"scalar"` | Execution runtime: `"scalar"` or `"torch"` |
| `surrogate_config` | `dict \| None` | `None` | Per-graph overrides for surrogate training |
| `evo_config` | `dict \| None` | `None` | Per-graph overrides for evolutionary training |
| `torch_config` | `dict \| None` | `None` | Torch-specific config (device, amp_dtype, template_spec, etc.) |
| `variant_library` | `dict \| None` | `None` | `{family: {version: NeuronGraph}}` reusable subgraph templates |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Graph name |
| `training_method` | `str` | Training method string |
| `runtime` | `str` | Execution runtime string |
| `surrogate_config` | `dict` | Surrogate training config overrides |
| `evo_config` | `dict` | Evolutionary training config overrides |
| `torch_config` | `dict` | Torch-specific configuration |
| `variant_library` | `dict[str, dict[str, NeuronGraph]]` | Variant library |
| `nodes` | `dict[str, NeuronInstance]` | All nodes keyed by instance_id |
| `edges` | `dict[str, Edge]` | All edges keyed by edge id |
| `input_node_ids` | `list[str]` | Ordered list of input node instance_ids |
| `output_node_ids` | `list[str]` | Ordered list of output node instance_ids |

---

### Node / Edge Mutations

#### `add_node(instance: NeuronInstance) -> str`

Add a node to the graph. Returns the `instance_id`.

#### `remove_node(node_id: str) -> None`

Remove a node and all connected edges. Also removes the node from `input_node_ids` and `output_node_ids`.

#### `add_edge(edge: Edge) -> str`

Add an edge to the graph. Returns the edge `id`. Raises `ValueError` if `src_node` or `dst_node` does not exist in the graph.

#### `remove_edge(edge_id: str) -> None`

Remove an edge from the graph. No-op if the edge does not exist.

---

### Topology

#### `has_cycles() -> bool`

Returns `True` if the graph contains cycles. Uses `networkx.is_directed_acyclic_graph`.

#### `topological_order() -> list[str]`

Return node instance_ids in topological order. Raises `networkx.NetworkXUnfeasible` if the graph has cycles.

#### `validate(*, as_subgraph: bool = False, seen: set[int] | None = None) -> None`

Validate the graph structure:
- If `as_subgraph=True`, requires `input_node_ids` and `output_node_ids` to be non-empty.
- Checks that all referenced input/output node IDs exist.
- Recursively validates nested subgraphs.
- Detects infinite recursive subgraph references via the `seen` set.
- Refreshes interface ports on subgraph nodes.

---

### Interface

#### `interface_input_layout() -> list[tuple[str, int, Port]]`

Returns `[(node_id, port_index, Port), ...]` for all ports on input nodes. Raises `ValueError` if an input node is missing.

#### `interface_output_layout() -> list[tuple[str, int, Port]]`

Returns `[(node_id, port_index, Port), ...]` for all ports on output nodes. Raises `ValueError` if an output node is missing.

#### `flattened_input_ports(aliases: list[str] | None = None) -> list[Port]`

Return a flat list of `Port` objects for the graph's input interface. If `aliases` is provided, port names are replaced. Raises `ValueError` if the alias count does not match the layout.

#### `flattened_output_ports(aliases: list[str] | None = None) -> list[Port]`

Return a flat list of `Port` objects for the graph's output interface. If `aliases` is provided, port names are replaced.

---

### Execution

#### `execute(inputs, *, max_iters=50, damping=0.5, tolerance=1e-6) -> dict[str, tuple[float, ...]]`

```python
def execute(
    self,
    inputs: dict[str, tuple[float, ...]],
    *,
    max_iters: int = 50,
    damping: float = 0.5,
    tolerance: float = 1e-6,
) -> dict[str, tuple[float, ...]]
```

Run the graph with scalar values.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputs` | `dict[str, tuple[float, ...]]` | *(required)* | Input-node instance_id to tuple of values |
| `max_iters` | `int` | `50` | Iteration cap for cyclic graphs |
| `damping` | `float` | `0.5` | Blending factor for recurrent settling (0=keep old, 1=use new) |
| `tolerance` | `float` | `1e-6` | Convergence threshold for cyclic mode |

**Returns:** Mapping of output-node instance_id to tuple of output values.

**Raises:** `TypeError` if the graph uses torch module nodes.

For DAG graphs, nodes are evaluated in topological order. For cyclic graphs, iterative settling is used with damping until convergence or `max_iters`.

#### `execute_trace(inputs, *, max_iters=50, damping=0.5, tolerance=1e-6) -> dict[str, tuple[float, ...]]`

Like `execute`, but returns all internal node activations (not just output nodes).

#### `execute_flat(flat_inputs: tuple[float, ...]) -> tuple[float, ...]`

Execute the graph using a flat tuple of inputs (concatenated across all input nodes) and return a flat tuple of outputs (concatenated across all output nodes).

---

### Inspection

#### `has_nested_subgraphs() -> bool`

Returns `True` if any node has `kind="subgraph"` with a non-None subgraph.

#### `has_module_nodes() -> bool`

Returns `True` if any node (or nested subgraph node) has `kind="module"`.

#### `has_recursive_subgraphs() -> bool`

Returns `True` if the graph contains nested subgraphs at any depth.

---

### Edge Parameters

#### `get_edge_params() -> list[float]`

Return a flat vector of all edge parameters: `[weight0, bias0, weight1, bias1, ...]` in sorted edge-id order.

#### `set_edge_params(params: list[float]) -> None`

Set edge weights and biases from a flat parameter vector.

#### `param_count() -> int`

Return the total number of edge parameters (`len(edges) * 2`).

---

### Variant Library

#### `resolve_variant_library() -> None`

Recursively resolve all variant references in the graph. Each subgraph node with a `variant_ref` is replaced with the corresponding graph from the variant library. Handles alias chains via `VARIANT_FAMILY_ALIASES`. Falls back to the node's inline subgraph if the library entry has incompatible ports.

---

### Serialization

#### `to_dict() -> dict`

Serialize the entire graph (nodes, edges, variant library, config) to a JSON-compatible dictionary.

#### `NeuronGraph.from_dict(d: dict) -> NeuronGraph` *(classmethod)*

Deserialize a graph from a dictionary. Resolves the variant library and validates the graph.

#### `NeuronGraph.from_dict_raw(d: dict) -> NeuronGraph` *(classmethod)*

Raw deserialization without variant resolution or validation. Raises `ValueError` if `d` is None.
