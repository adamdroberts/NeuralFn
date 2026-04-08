# neuralfn.neuron

Defines `NeuronDef` and the factory functions for creating neuron definitions.

## Class: NeuronDef

```python
@dataclass
class NeuronDef:
    name: str
    fn: Callable | None
    input_ports: list[Port]
    output_ports: list[Port]
    source_code: str = ""
    kind: str = "function"
    subgraph: NeuronGraph | None = None
    module_type: str = ""
    module_config: dict[str, Any] = field(default_factory=dict)
    module_state: str = ""
    input_aliases: list[str] = field(default_factory=list)
    output_aliases: list[str] = field(default_factory=list)
    variant_ref: dict[str, str] | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
```

A `NeuronDef` represents a reusable neuron type. There are three kinds:

- **`"function"`** -- a Python callable wrapped with port declarations.
- **`"module"`** -- a PyTorch `nn.Module` created at compile time via `build_module()`.
- **`"subgraph"`** -- a nested `NeuronGraph` that acts as a single neuron.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *(required)* | Display name of the neuron |
| `fn` | `Callable \| None` | *(required)* | The Python function (None for module/subgraph kinds) |
| `input_ports` | `list[Port]` | *(required)* | Ordered input port declarations |
| `output_ports` | `list[Port]` | *(required)* | Ordered output port declarations |
| `source_code` | `str` | `""` | Python source for function neurons (used for serialization) |
| `kind` | `str` | `"function"` | One of `"function"`, `"module"`, or `"subgraph"` |
| `subgraph` | `NeuronGraph \| None` | `None` | Nested graph for subgraph neurons |
| `module_type` | `str` | `""` | Module type string dispatched by `build_module()` |
| `module_config` | `dict[str, Any]` | `{}` | Configuration dict passed to the module constructor |
| `module_state` | `str` | `""` | Base64-encoded PyTorch state_dict |
| `input_aliases` | `list[str]` | `[]` | Port name aliases for subgraph interface |
| `output_aliases` | `list[str]` | `[]` | Port name aliases for subgraph interface |
| `variant_ref` | `dict[str, str] \| None` | `None` | `{"family": ..., "version": ...}` reference into the variant library |
| `id` | `str` | *(auto-generated)* | Unique 12-char hex identifier |

### Properties

#### `n_inputs -> int`

Number of input ports.

#### `n_outputs -> int`

Number of output ports.

### Methods

#### `__call__(*args: float) -> tuple[float, ...]`

Execute the neuron with scalar arguments. Output values are conditioned (clamped and quantized) through the output ports.

- **function** neurons call `self.fn(*args)`.
- **subgraph** neurons delegate to `self.subgraph.execute_flat(args)`.
- **module** neurons raise `TypeError` (they require the torch runtime).

#### `refresh_interface_ports() -> None`

For subgraph neurons, re-derive `input_ports` and `output_ports` from the nested subgraph's interface layout and aliases. No-op for other kinds.

#### `to_dict() -> dict`

Serialize to a JSON-compatible dictionary. For subgraph neurons, calls `refresh_interface_ports()` first.

#### `NeuronDef.from_dict(d: dict) -> NeuronDef` *(classmethod)*

Reconstruct a `NeuronDef` from serialized JSON. For function neurons, the callable is rebuilt by exec-ing the stored `source_code`. For subgraph neurons, the nested graph is deserialized and validated.

#### `NeuronDef.from_dict_raw(d: dict) -> NeuronDef` *(classmethod)*

Raw deserialization without validation or variant resolution. Used internally by `NeuronGraph.from_dict_raw()`.

---

## Factory Functions

### `neuron(inputs, outputs, *, name=None)`

```python
def neuron(
    inputs: Sequence[Port],
    outputs: Sequence[Port],
    *,
    name: str | None = None,
) -> Callable[[Callable], NeuronDef]
```

Decorator that turns a plain Python function into a `NeuronDef`. The function's source code is captured automatically via `inspect.getsource`.

```python
@neuron(
    inputs=[Port("x", range=(-10, 10))],
    outputs=[Port("y", range=(0, 1))],
)
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputs` | `Sequence[Port]` | *(required)* | Input port declarations |
| `outputs` | `Sequence[Port]` | *(required)* | Output port declarations |
| `name` | `str \| None` | `None` | Override the display name (defaults to `fn.__name__`) |

**Returns:** A decorator that accepts a callable and returns a `NeuronDef`.

---

### `neuron_from_source(source_code, fn_name, input_ports, output_ports, *, neuron_id=None)`

```python
def neuron_from_source(
    source_code: str,
    fn_name: str,
    input_ports: list[Port],
    output_ports: list[Port],
    *,
    neuron_id: str | None = None,
) -> NeuronDef
```

Build a `NeuronDef` from a raw source-code string. The source is exec'd and the named function is extracted. Raises `ValueError` if no callable with `fn_name` is found.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_code` | `str` | *(required)* | Python source code containing the function |
| `fn_name` | `str` | *(required)* | Name of the function to extract |
| `input_ports` | `list[Port]` | *(required)* | Input port declarations |
| `output_ports` | `list[Port]` | *(required)* | Output port declarations |
| `neuron_id` | `str \| None` | `None` | Override the auto-generated ID |

---

### `module_neuron(*, name, module_type, input_ports, output_ports, module_config=None, module_state="", neuron_id=None)`

```python
def module_neuron(
    *,
    name: str,
    module_type: str,
    input_ports: list[Port],
    output_ports: list[Port],
    module_config: dict[str, Any] | None = None,
    module_state: str = "",
    neuron_id: str | None = None,
) -> NeuronDef
```

Create a `NeuronDef` with `kind="module"`. At compile time, `build_module(module_type, module_config)` constructs the corresponding `nn.Module`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *(required)* | Display name |
| `module_type` | `str` | *(required)* | Module type string dispatched by `build_module()` |
| `input_ports` | `list[Port]` | *(required)* | Input port declarations |
| `output_ports` | `list[Port]` | *(required)* | Output port declarations |
| `module_config` | `dict \| None` | `None` | Configuration passed to the module constructor |
| `module_state` | `str` | `""` | Base64-encoded state_dict blob |
| `neuron_id` | `str \| None` | `None` | Override the auto-generated ID |

---

### `subgraph_neuron(graph, *, name, input_aliases=None, output_aliases=None, variant_ref=None, neuron_id=None)`

```python
def subgraph_neuron(
    graph: NeuronGraph,
    *,
    name: str,
    input_aliases: list[str] | None = None,
    output_aliases: list[str] | None = None,
    variant_ref: dict[str, str] | None = None,
    neuron_id: str | None = None,
) -> NeuronDef
```

Wrap a `NeuronGraph` as a reusable neuron definition with `kind="subgraph"`. The graph is validated with `as_subgraph=True`. Interface ports are derived from the graph's input/output layout, optionally renamed via aliases.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `NeuronGraph` | *(required)* | The nested graph |
| `name` | `str` | *(required)* | Display name |
| `input_aliases` | `list[str] \| None` | `None` | Rename interface input ports |
| `output_aliases` | `list[str] \| None` | `None` | Rename interface output ports |
| `variant_ref` | `dict[str, str] \| None` | `None` | Variant library reference `{"family": ..., "version": ...}` |
| `neuron_id` | `str \| None` | `None` | Override the auto-generated ID |

---

## Utility Functions

### `encode_module_state_dict(state_dict: dict) -> str`

Serialize a PyTorch `state_dict` to a base64-encoded string using `torch.save`.

### `decode_module_state_dict(blob: str) -> dict`

Deserialize a base64-encoded state_dict string back to a dictionary. Returns `{}` if `blob` is empty.
