# NeuralFn Python SDK -- Complete Reference

This is the detailed reference for agents building with the `neuralfn` Python package. Read this when you need exact signatures, field lists, or method behavior beyond what SKILL.md covers.

---

## Port (frozen dataclass)

```python
from neuralfn import Port

Port(name: str, range: tuple[float, float] = (-1.0, 1.0), precision: float = 0.001, dtype: str = "float")
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Unique port identifier within a neuron |
| `range` | `tuple[float, float]` | `(-1.0, 1.0)` | `(low, high)` value bounds. `low` must be `< high`. |
| `precision` | `float` | `0.001` | Quantization step. Must be `> 0`. |
| `dtype` | `str` | `"float"` | Semantic type hint: `"float"`, `"int"`, or `"bool"` |

**Methods:**
- `clamp(value: float) -> float` -- restrict to range
- `quantize(value: float) -> float` -- round to nearest precision step
- `condition(value: float) -> float` -- clamp then quantize
- `to_dict() -> dict` / `Port.from_dict(d) -> Port` -- serialization

---

## NeuronDef (dataclass)

```python
from neuralfn import NeuronDef
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Display name |
| `fn` | `Callable \| None` | required | Python callable (None for module/subgraph) |
| `input_ports` | `list[Port]` | required | Input port definitions |
| `output_ports` | `list[Port]` | required | Output port definitions |
| `source_code` | `str` | `""` | Python source for function neurons |
| `kind` | `str` | `"function"` | `"function"`, `"subgraph"`, or `"module"` |
| `subgraph` | `NeuronGraph \| None` | `None` | Nested graph (subgraph kind only) |
| `module_type` | `str` | `""` | Torch module type string (module kind only) |
| `module_config` | `dict` | `{}` | Module configuration |
| `module_state` | `str` | `""` | Base64-encoded torch state_dict |
| `input_aliases` | `list[str]` | `[]` | Renamed input port names for subgraphs |
| `output_aliases` | `list[str]` | `[]` | Renamed output port names for subgraphs |
| `variant_ref` | `dict[str,str] \| None` | `None` | `{"family": "...", "version": "..."}` for variant-linked nodes |
| `id` | `str` | auto (12-char hex) | Unique definition ID |

**Properties:** `n_inputs -> int`, `n_outputs -> int`

**Methods:**
- `__call__(*args: float) -> tuple[float, ...]` -- execute (function/subgraph only; module raises TypeError)
- `refresh_interface_ports() -> None` -- re-derive ports from nested subgraph
- `to_dict() -> dict` / `NeuronDef.from_dict(d) -> NeuronDef` / `NeuronDef.from_dict_raw(d) -> NeuronDef`

---

## Neuron factory functions

### `neuron(inputs, outputs, *, name=None)`

Decorator that turns a plain Python function into a `NeuronDef` with `kind="function"`.

```python
@neuron(
    inputs=[Port("x", range=(-5, 5))],
    outputs=[Port("y", range=(0, 1))],
)
def my_fn(x):
    return 1 / (1 + math.exp(-x))
# my_fn is now a NeuronDef, not a plain function
```

### `neuron_from_source(source_code, fn_name, input_ports, output_ports, *, neuron_id=None)`

Build a NeuronDef from a raw source code string. The function is exec'd and extracted by name.

```python
ndef = neuron_from_source(
    "def relu(x):\n    return max(0, x)\n",
    "relu",
    [Port("x", range=(-10, 10))],
    [Port("y", range=(0, 10))],
)
```

### `module_neuron(*, name, module_type, input_ports, output_ports, module_config=None, module_state="", neuron_id=None)`

Create a NeuronDef wrapping a torch nn.Module stage. Kind is `"module"`. Cannot be called through scalar execution.

```python
linear = module_neuron(
    name="linear",
    module_type="linear",
    input_ports=[Port("x", range=(-1e6, 1e6))],
    output_ports=[Port("y", range=(-1e6, 1e6))],
    module_config={"input_dim": 128, "output_dim": 128, "bias": True},
)
```

### `subgraph_neuron(graph, *, name, input_aliases=None, output_aliases=None, variant_ref=None, neuron_id=None)`

Wrap a `NeuronGraph` as a reusable neuron. The graph must have `input_node_ids` and `output_node_ids` set. Validates the graph as a subgraph. Port names are derived from the child graph's I/O nodes, optionally renamed by aliases.

```python
block = subgraph_neuron(child_graph, name="my_block", input_aliases=["x"], output_aliases=["y"])
```

---

## NeuronInstance (dataclass)

```python
NeuronInstance(neuron_def: NeuronDef, instance_id: str = auto, position: tuple[float, float] = (0, 0))
```

A placed instance of a NeuronDef in a graph. Multiple instances can share the same NeuronDef.

- `name` property delegates to `neuron_def.name`
- `to_dict()` / `NeuronInstance.from_dict(d)` for serialization

---

## Edge (dataclass)

```python
Edge(id: str = auto, src_node: str = "", src_port: int = 0, dst_node: str = "", dst_port: int = 0, weight: float = 1.0, bias: float = 0.0)
```

Weighted connection. The scalar runtime computes `output = input * weight + bias`.

- `transform(value: float) -> float` -- apply the linear transform
- `to_dict()` / `Edge.from_dict(d)` for serialization

---

## NeuronGraph

```python
NeuronGraph(
    *,
    name: str = "graph",
    training_method: str = "surrogate",   # "surrogate" | "evolutionary" | "torch" | "frozen"
    runtime: str = "scalar",              # "scalar" | "torch"
    surrogate_config: dict | None = None,
    evo_config: dict | None = None,
    torch_config: dict | None = None,
    variant_library: dict[str, dict[str, NeuronGraph]] | None = None,
)
```

### Instance attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Graph name |
| `training_method` | `str` | Training strategy |
| `runtime` | `str` | Execution backend |
| `surrogate_config` | `dict` | Surrogate config overrides |
| `evo_config` | `dict` | Evolutionary config overrides |
| `torch_config` | `dict` | Torch config (device, template_spec, etc.) |
| `variant_library` | `dict[str, dict[str, NeuronGraph]]` | `{family: {version: graph}}` |
| `nodes` | `dict[str, NeuronInstance]` | All nodes keyed by instance_id |
| `edges` | `dict[str, Edge]` | All edges keyed by edge id |
| `input_node_ids` | `list[str]` | Ordered input node IDs |
| `output_node_ids` | `list[str]` | Ordered output node IDs |

### All methods

**Node/edge mutations:**
- `add_node(instance: NeuronInstance) -> str` -- returns instance_id
- `remove_node(node_id: str) -> None` -- also removes connected edges and I/O refs
- `add_edge(edge: Edge) -> str` -- raises ValueError if src/dst nodes missing
- `remove_edge(edge_id: str) -> None`

**Topology:**
- `has_cycles() -> bool`
- `topological_order() -> list[str]` -- raises if cyclic
- `validate(*, as_subgraph=False, seen=None) -> None` -- structural validation, recursive for subgraphs

**Interface:**
- `interface_input_layout() -> list[tuple[str, int, Port]]`
- `interface_output_layout() -> list[tuple[str, int, Port]]`
- `flattened_input_ports(aliases=None) -> list[Port]`
- `flattened_output_ports(aliases=None) -> list[Port]`

**Execution (scalar runtime only):**
- `execute(inputs: dict[str, tuple[float,...]], *, max_iters=50, damping=0.5, tolerance=1e-6) -> dict[str, tuple[float,...]]`
  - `inputs`: `{"node_id": (val1, val2, ...)}` for each input node
  - Returns: `{"output_node_id": (val1, ...)}` for each output node
  - Raises TypeError if graph has module nodes
  - DAG graphs: topological order. Cyclic graphs: iterative settling.
- `execute_trace(inputs, ...) -> dict[str, tuple[float,...]]` -- returns ALL node activations
- `execute_flat(flat_inputs: tuple[float,...]) -> tuple[float,...]` -- flat I/O interface

**Inspection:**
- `has_nested_subgraphs() -> bool`
- `has_module_nodes() -> bool` -- recursive
- `has_recursive_subgraphs() -> bool`

**Edge parameters (for training):**
- `get_edge_params() -> list[float]` -- `[w0, b0, w1, b1, ...]` sorted by edge id
- `set_edge_params(params: list[float]) -> None`
- `param_count() -> int` -- `len(edges) * 2`

**Variant library:**
- `resolve_variant_library() -> None` -- resolve all variant_ref nodes against the library. Falls back to inline subgraph if ports are incompatible.

**Serialization:**
- `to_dict() -> dict`
- `NeuronGraph.from_dict(d) -> NeuronGraph` -- resolves variants, validates
- `NeuronGraph.from_dict_raw(d) -> NeuronGraph` -- raw, no resolution

---

## Serialization functions

```python
from neuralfn import save_graph, load_graph

save_graph(graph: NeuronGraph, path: str | Path) -> None
load_graph(path: str | Path) -> NeuronGraph
```

---

## BuiltinNeurons

```python
from neuralfn import BuiltinNeurons

BuiltinNeurons.sigmoid          # NeuronDef (scalar)
BuiltinNeurons.linear_module    # NeuronDef (torch module)
BuiltinNeurons.all()            # list[NeuronDef] -- all 91 builtins
BuiltinNeurons.get("relu")      # lookup by name
```

### Complete builtin catalog

**Scalar activations (kind="function"):**
sigmoid, relu, tanh_neuron, threshold, identity, negate, gaussian, log_neuron, leaky_relu, prelu, relu6, elu, selu, gelu, silu, mish, softplus, softsign, hard_sigmoid, hard_tanh, hard_swish

**Scalar binary/multi-output (kind="function"):**
add (2 in, 1 out), multiply (2 in, 1 out), softmax_2 (2 in, 2 out), logsoftmax_2 (2 in, 2 out)

**Graph terminals (kind="function"):**
input_node (0 in, 1 out), output_node (1 in, 1 out)

**Torch modules (kind="module") -- attribute name -> module_type:**
token_embedding_module, absolute_position_embedding_module, linear_module, mlp_relu2_module, gelu_module, swiglu_module, rms_norm_module, layer_norm_module, dropout_module, reshape_heads_module, merge_heads_module, repeat_kv_module, rotary_embedding_module, qk_gain_module, scaled_dot_product_attention_module, causal_self_attention_module, fused_causal_attention_module, residual_mix_module, residual_add_module, kv_cache_read_module, kv_cache_write_module, kv_pca_encode_module, kv_pca_decode_module, kv_quant_pack_module, kv_quant_unpack_module, tied_lm_head_module, lm_head_module, logit_softcap_module, token_cross_entropy_module, router_logits_module, topk_route_module, expert_dispatch_module, expert_combine_module, load_balance_loss_module, aux_loss_add_module, dataset_source_module, bitlinear_ternary_module, randmap_adapter_module, mamba_module, denoise_head_module, mask_scheduler_module, random_timesteps_module, jepa_mask_module, latent_pool_module, jepa_projector_module, jepa_predictor_module, latent_mse_loss_module, byte_patch_embed_module, byte_patch_merge_module, act_halt_gate_module, act_weighted_sum_module, universal_transformer_module, ttt_linear_module

---

## Training classes

### TrainConfig (dataclass)

```python
from neuralfn.trainer import TrainConfig

TrainConfig(
    learning_rate: float = 1e-3,
    epochs: int = 500,
    batch_size: int = 32,
    surrogate_samples: int = 10_000,
    surrogate_hidden: tuple[int, ...] = (64, 64),
    surrogate_epochs: int = 200,
    loss_fn: str = "mse",      # "mse" or "bce"
)
```

### SurrogateTrainer

```python
from neuralfn import SurrogateTrainer

trainer = SurrogateTrainer(graph: NeuronGraph, config: TrainConfig | None = None)
trainer.train(
    train_inputs: np.ndarray,    # shape (N, n_graph_inputs)
    train_targets: np.ndarray,   # shape (N, n_graph_outputs)
    *,
    on_epoch: Callable[[int, float], None] | None = None,
) -> list[float]                 # per-epoch losses
trainer.stop()                   # signal early stop
```

Attributes: `graph`, `config`, `surrogates` (dict[str, SurrogateModel]), `loss_history`.

### EvoConfig (dataclass)

```python
from neuralfn.evolutionary import EvoConfig

EvoConfig(
    population_size: int = 50,
    generations: int = 200,
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.3,
    crossover_rate: float = 0.5,
    tournament_size: int = 3,
    elite_count: int = 2,
    topology_mutations: bool = False,
    seed: int | None = None,
)
```

### EvolutionaryTrainer

```python
from neuralfn import EvolutionaryTrainer

evo = EvolutionaryTrainer(graph, config: EvoConfig | None = None, neuron_library: list[NeuronDef] | None = None)
evo.train(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    *,
    fitness_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,  # default MSE
    on_generation: Callable[[int, float], None] | None = None,
) -> list[float]
evo.stop()
```

### HybridConfig (dataclass)

```python
from neuralfn import HybridConfig

HybridConfig(
    outer_rounds: int = 3,
    loss_fn: str = "mse",
    default_surrogate: TrainConfig = TrainConfig(),
    default_evolutionary: EvoConfig = EvoConfig(),
)
```

### HybridTrainer

```python
from neuralfn import HybridTrainer

trainer = HybridTrainer(graph: NeuronGraph, config: HybridConfig | None = None)
trainer.train(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    *,
    on_step: Callable[[dict], None] | None = None,
) -> list[float]
trainer.stop()
```

Each subgraph picks its own `training_method`: `"surrogate"`, `"evolutionary"`, or `"frozen"`. The hybrid trainer walks the graph tree in post-order and trains each scope according to its method.

The `on_step` callback receives a dict with keys: `graph_path`, `graph_name`, `method`, `round`, `local_step`, `loss`.

### GraphScope (frozen dataclass)

```python
from neuralfn.hybrid import GraphScope

GraphScope(path: tuple[str, ...], graph: NeuronGraph)
```

Internal representation of a subgraph's position in the nested hierarchy.

---

## Surrogate/probe utilities

```python
from neuralfn import probe_neuron, build_surrogates, SurrogateModel

# Probe a neuron's transfer function
xs, ys = probe_neuron(neuron_def: NeuronDef, n_samples: int = 10_000)
# xs shape: (n_samples, n_inputs), ys shape: (n_samples, n_outputs)

# Build surrogates for every neuron in a graph
surrogates = build_surrogates(graph, n_samples=10_000, hidden_sizes=(64, 64), epochs=200)
# returns dict[str, SurrogateModel] mapping instance_id -> trained model
```

### SurrogateModel (nn.Module)

```python
SurrogateModel(n_inputs: int, n_outputs: int, hidden_sizes: tuple[int, ...] = (64, 64))
```

Small MLP trained to approximate a neuron's transfer function. Has a `forward(x: Tensor) -> Tensor` method.

---

## Variant library

The variant library on `NeuronGraph` is a `dict[str, dict[str, NeuronGraph]]` mapping `family -> version -> graph`.

Nodes can reference a library entry via `variant_ref={"family": "attention", "version": "default"}` on the `NeuronDef`. When `resolve_variant_library()` is called (or `from_dict()` is used), variant-ref nodes have their subgraph replaced with the library version.

If the library entry's ports are incompatible with the node's current ports, the resolver keeps the node's inline subgraph instead of throwing.

`VARIANT_FAMILY_ALIASES` provides compatibility mapping (e.g., `"attn_block"` -> `"transformer_block"`).

---

## encode/decode module state

```python
from neuralfn.neuron import encode_module_state_dict, decode_module_state_dict

blob: str = encode_module_state_dict(state_dict)   # base64-encoded torch.save
state_dict: dict = decode_module_state_dict(blob)   # torch.load from base64
```
