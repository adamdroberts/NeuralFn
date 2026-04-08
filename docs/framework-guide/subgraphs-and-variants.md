# Subgraphs and Variants

NeuralFn graphs are hierarchical: any node can embed an entire sub-graph behind its ports. The **variant library** extends this by letting you define named, swappable implementations for each architectural slot.

## Building a nested graph

Use `subgraph_neuron()` to wrap a `NeuronGraph` as a single `NeuronDef`:

```python
from neuralfn import NeuronGraph, NeuronInstance, Edge, BuiltinNeurons, subgraph_neuron

inner = NeuronGraph(name="inner")
inner.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
inner.add_node(NeuronInstance(BuiltinNeurons.sigmoid, instance_id="act"))
inner.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
inner.add_edge(Edge(id="e1", src_node="in", src_port=0, dst_node="act", dst_port=0))
inner.add_edge(Edge(id="e2", src_node="act", src_port=0, dst_node="out", dst_port=0))
inner.input_node_ids = ["in"]
inner.output_node_ids = ["out"]

block = subgraph_neuron(inner, name="my_block", input_aliases=["x"], output_aliases=["y"])
```

The parent graph sees `block` as a single node with one input port `x` and one output port `y`. Internally, execution delegates to the child graph.

---

## Port aliasing

`input_aliases` and `output_aliases` rename the interface ports that the parent graph sees. Their lengths must match the total number of ports across all input/output nodes in the child graph.

```python
block = subgraph_neuron(
    inner,
    name="attn_block",
    input_aliases=["hidden", "mask"],
    output_aliases=["attended"],
)
```

### Derived port lists

For programmatic inspection, `NeuronDef` provides:

- **`flattened_input_ports()`** -- collects all input ports from the child graph's `input_node_ids`, renamed by `input_aliases`.
- **`flattened_output_ports()`** -- same for `output_node_ids`, renamed by `output_aliases`.

---

## The variant library

A graph's `variant_library` is a nested dict:

```
graph.variant_library: dict[str, dict[str, NeuronGraph]]
```

It maps **family** (e.g. `"attention"`, `"mlp"`) to **version** (e.g. `"default"`, `"flash"`) to a `NeuronGraph` implementation.

```python
g = NeuronGraph(name="model", runtime="torch", training_method="torch")

g.variant_library = {
    "attention": {
        "default": sdpa_attention_graph,
        "flash": flash_attention_graph,
    },
    "mlp": {
        "default": gelu_mlp_graph,
        "swiglu": swiglu_mlp_graph,
    },
}
```

---

## Variant references

A subgraph neuron can declare a `variant_ref` instead of embedding a fixed child graph. The reference names a family and version in the library:

```python
attn_node = subgraph_neuron(
    sdpa_attention_graph,
    name="attn",
    input_aliases=["x"],
    output_aliases=["y"],
    variant_ref={"family": "attention", "version": "default"},
)
```

The inline subgraph serves as a fallback. At resolution time the library version replaces it -- unless the ports are incompatible.

---

## Resolution

`resolve_variant_library()` walks every node in a graph. For each node whose `NeuronDef` carries a `variant_ref`, it looks up the corresponding family and version in the graph's `variant_library` and swaps in that graph as the node's subgraph.

```python
from neuralfn.graph import resolve_variant_library

resolve_variant_library(g)
```

### Fallback behavior

If the library entry exists but its port count or port names are incompatible with the node's inline subgraph, the resolver **keeps the inline subgraph** instead of raising an error. This is intentional -- it prevents cross-contamination from breaking graphs when multiple templates share the same variant family namespace.

---

## Variant family aliases

Some family names have been renamed over time. The `VARIANT_FAMILY_ALIASES` table provides a compatibility layer so older saved graphs still resolve:

```
"attn_block"         -> "transformer_block"
"transformer_block"  -> "attn_block"
```

Both the Python resolver (`neuralfn/graph.py`) and the frontend resolver (`editor/src/store/graphUtils.ts`) consult this alias table before falling back. If the primary family name is not found, the resolver tries aliases in order.

---

## Cross-contamination

All shipped presets share a flat variant-family namespace at the root graph level. When the user loads template A and then template B in the same session, `mergeVariantLibrary` overwrites any family that both templates define.

If a dense preset writes `mlp@default` with 1 output port and an MoE preset then overwrites it with 2 output ports, template A's block nodes still hold inline subgraph ports that no longer match the library entry. The resolver handles this gracefully by falling back to the inline subgraph, but it is worth understanding this behavior when working with multiple templates in a single session.

---

## Example: creating and linking a variant library

```python
from neuralfn import (
    NeuronGraph, NeuronInstance, Edge, BuiltinNeurons,
    subgraph_neuron, Port,
)
from neuralfn.neuron import module_neuron

def build_mlp_variant(activation: str = "sigmoid") -> NeuronGraph:
    """Build a simple 1-in, 1-out MLP subgraph."""
    g = NeuronGraph(name=f"mlp_{activation}")
    g.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
    act_neuron = BuiltinNeurons.sigmoid if activation == "sigmoid" else BuiltinNeurons.tanh
    g.add_node(NeuronInstance(act_neuron, instance_id="act"))
    g.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
    g.add_edge(Edge(id="e1", src_node="in", src_port=0, dst_node="act", dst_port=0))
    g.add_edge(Edge(id="e2", src_node="act", src_port=0, dst_node="out", dst_port=0))
    g.input_node_ids = ["in"]
    g.output_node_ids = ["out"]
    return g

root = NeuronGraph(name="root")
root.variant_library = {
    "mlp": {
        "sigmoid": build_mlp_variant("sigmoid"),
        "tanh": build_mlp_variant("tanh"),
    },
}

root.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
root.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))

mlp_block = subgraph_neuron(
    build_mlp_variant("sigmoid"),
    name="mlp",
    input_aliases=["x"],
    output_aliases=["y"],
    variant_ref={"family": "mlp", "version": "sigmoid"},
)
root.add_node(NeuronInstance(mlp_block, instance_id="mlp1"))

root.add_edge(Edge(id="e1", src_node="in", src_port=0, dst_node="mlp1", dst_port=0))
root.add_edge(Edge(id="e2", src_node="mlp1", src_port=0, dst_node="out", dst_port=0))
root.input_node_ids = ["in"]
root.output_node_ids = ["out"]

result = root.execute({"in": (0.5,)})
print(result)
```

---

See [Python SDK: Graph](../python-sdk/graph.md) for the full method reference.

Next: [Torch Models](torch-models.md)
