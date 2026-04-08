# Defining Neurons

A **neuron** in NeuralFn is any callable computation with a fixed set of typed input ports and output ports. The framework provides four construction paths, each producing a `NeuronDef` that can be placed into a graph as a `NeuronInstance`.

## Ports

Every input and output is described by a `Port`:

```python
from neuralfn import Port

p = Port("x", range=(-5, 5), precision=0.01, dtype="float")
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Human-readable label for the port. |
| `range` | `tuple[float, float]` | `(-1, 1)` | Valid value interval. Used for clamping and surrogate sampling. |
| `precision` | `float` | `0.001` | Quantization step size. |
| `dtype` | `str` | `"float"` | Data type hint. |

### Port methods

- **`clamp(value)`** -- restricts a value to the port's `range`.
- **`quantize(value)`** -- rounds a value to the nearest multiple of `precision`.
- **`condition(value)`** -- applies both `clamp` and `quantize` in sequence.

---

## 1. `@neuron` decorator

The simplest path. Wrap a plain Python function and declare its ports:

```python
from neuralfn import neuron, Port

@neuron(
    inputs=[Port("x", range=(-5, 5), precision=0.01)],
    outputs=[Port("y", range=(0, 1), precision=0.001)],
)
def custom_sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))
```

The decorator produces a `NeuronDef` with `kind="function"`. The function body is captured as `source_code` for serialization. Arguments are positional and map 1:1 to input ports; the return value maps to the output ports (scalar return for single-output neurons, tuple for multi-output).

---

## 2. `neuron_from_source()`

When neuron code is generated at runtime or loaded from a file, use `neuron_from_source()` to compile it into a `NeuronDef`:

```python
from neuralfn import Port
from neuralfn.neuron import neuron_from_source

code = '''
def my_relu(x):
    return max(0, x)
'''

ndef = neuron_from_source(
    code, "my_relu",
    [Port("x", range=(-10, 10))],
    [Port("y", range=(0, 10))],
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `source_code` | `str` | Python source containing at least one function definition. |
| `fn_name` | `str` | Name of the function to extract from the compiled source. |
| `input_ports` | `list[Port]` | Input port definitions. |
| `output_ports` | `list[Port]` | Output port definitions. |
| `neuron_id` | `str` (optional) | Explicit ID. Auto-generated if omitted. |

The resulting `NeuronDef` has `kind="function"`, same as the decorator form.

---

## 3. `module_neuron()`

For PyTorch `nn.Module` stages that operate on tensors rather than scalars:

```python
from neuralfn.neuron import module_neuron
from neuralfn import Port

linear = module_neuron(
    name="linear",
    module_type="linear",
    input_ports=[Port("x", range=(-1e6, 1e6))],
    output_ports=[Port("y", range=(-1e6, 1e6))],
    module_config={"input_dim": 128, "output_dim": 128, "bias": True},
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Display name. |
| `module_type` | `str` | Key that maps to a `*Stage` class in `torch_backend.py` via `build_module()`. |
| `input_ports` | `list[Port]` | Input port definitions. |
| `output_ports` | `list[Port]` | Output port definitions. |
| `module_config` | `dict` | Configuration dict passed to the stage constructor (dimensions, head counts, etc.). |
| `module_state` | `str` (optional) | Base64-encoded serialized state dict for pretrained weights. |
| `neuron_id` | `str` (optional) | Explicit ID. |

The resulting `NeuronDef` has `kind="module"`. Module neurons cannot be executed through the scalar runtime -- they require a `CompiledTorchGraph`.

---

## 4. `subgraph_neuron()`

Wraps an entire `NeuronGraph` as a single neuron, enabling hierarchical composition:

```python
from neuralfn import NeuronGraph, NeuronInstance, Edge, BuiltinNeurons, subgraph_neuron

child = NeuronGraph(name="child")
child.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
child.add_node(NeuronInstance(BuiltinNeurons.sigmoid, instance_id="act"))
child.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
child.add_edge(Edge(id="e1", src_node="in", src_port=0, dst_node="act", dst_port=0))
child.add_edge(Edge(id="e2", src_node="act", src_port=0, dst_node="out", dst_port=0))
child.input_node_ids = ["in"]
child.output_node_ids = ["out"]

block = subgraph_neuron(
    child,
    name="sigmoid_block",
    input_aliases=["x"],
    output_aliases=["y"],
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `graph` | `NeuronGraph` | The child graph to embed. Must have `input_node_ids` and `output_node_ids` set. |
| `name` | `str` | Display name for the resulting neuron. |
| `input_aliases` | `list[str]` | Rename the interface input ports. Length must match the total number of ports on `input_node_ids`. |
| `output_aliases` | `list[str]` | Rename the interface output ports. Length must match output node ports. |
| `variant_ref` | `dict` (optional) | Links this neuron to a variant library entry, e.g. `{"family": "attention", "version": "default"}`. |
| `neuron_id` | `str` (optional) | Explicit ID. |

The resulting `NeuronDef` has `kind="subgraph"` and stores the full child graph.

---

## NeuronDef properties

Every construction path produces a `NeuronDef`. Key properties:

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Display name. |
| `fn` | `callable` | The Python function (function neurons only). |
| `input_ports` | `list[Port]` | Declared input ports. |
| `output_ports` | `list[Port]` | Declared output ports. |
| `source_code` | `str` | Source text (function neurons). |
| `kind` | `str` | `"function"`, `"subgraph"`, or `"module"`. |
| `subgraph` | `NeuronGraph` | Embedded graph (subgraph neurons). |
| `module_type` | `str` | Stage key (module neurons). |
| `module_config` | `dict` | Module parameters (module neurons). |
| `module_state` | `str` | Serialized weights (module neurons). |
| `input_aliases` | `list[str]` | Port rename for subgraph interface. |
| `output_aliases` | `list[str]` | Port rename for subgraph interface. |
| `variant_ref` | `dict` | Variant library reference (subgraph neurons). |
| `id` | `str` | Unique identifier. |
| `n_inputs` | `int` | Number of input ports. |
| `n_outputs` | `int` | Number of output ports. |

---

Next: [Building Graphs](building-graphs.md)
