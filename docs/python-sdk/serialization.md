# neuralfn.serialization

JSON persistence for `NeuronGraph` instances.

---

## save_graph

```python
def save_graph(graph: NeuronGraph, path: str | Path) -> None
```

Serialize a `NeuronGraph` to a JSON file. Calls `graph.to_dict()` and writes the result with 2-space indentation.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `graph` | `NeuronGraph` | The graph to save |
| `path` | `str \| Path` | Output file path (`.json`) |

### Example

```python
from neuralfn import save_graph, NeuronGraph

graph = NeuronGraph(name="my_graph")
save_graph(graph, "my_graph.json")
```

---

## load_graph

```python
def load_graph(path: str | Path) -> NeuronGraph
```

Deserialize a `NeuronGraph` from a JSON file. Reads the JSON, then calls `NeuronGraph.from_dict()` which resolves the variant library and validates the graph structure.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Path to the JSON file |

### Returns

`NeuronGraph` -- the deserialized and validated graph.

### Example

```python
from neuralfn import load_graph

graph = load_graph("my_graph.json")
print(graph.name)
print(len(graph.nodes), "nodes")
print(len(graph.edges), "edges")
```
