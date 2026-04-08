# neuralfn.hybrid

Per-subgraph training orchestrator for nested graph hierarchies.

---

## HybridConfig

```python
@dataclass
class HybridConfig:
    outer_rounds: int = 3
    loss_fn: str = "mse"
    default_surrogate: TrainConfig = field(default_factory=TrainConfig)
    default_evolutionary: EvoConfig = field(default_factory=EvoConfig)
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `outer_rounds` | `int` | `3` | Number of outer training rounds across all subgraphs |
| `loss_fn` | `str` | `"mse"` | Loss function: `"mse"` or `"bce"` |
| `default_surrogate` | `TrainConfig` | defaults | Default surrogate training config (merged with per-graph `surrogate_config`) |
| `default_evolutionary` | `EvoConfig` | defaults | Default evolutionary config (merged with per-graph `evo_config`) |

---

## GraphScope

```python
@dataclass(frozen=True)
class GraphScope:
    path: tuple[str, ...]
    graph: NeuronGraph
```

Identifies a subgraph within the nested hierarchy.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `path` | `tuple[str, ...]` | Tuple of node instance_ids from root to this subgraph |
| `graph` | `NeuronGraph` | The subgraph at this scope |

---

## HybridTrainer

```python
class HybridTrainer:
    def __init__(
        self,
        graph: NeuronGraph,
        config: HybridConfig | None = None,
    ) -> None
```

Trains nested graphs with per-graph surrogate or evolutionary updates. Walks the graph tree in post-order (children first, then parents), applying each subgraph's declared `training_method`.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `NeuronGraph` | *(required)* | The root graph |
| `config` | `HybridConfig \| None` | `None` | Hybrid configuration (uses defaults if None) |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `graph` | `NeuronGraph` | The root graph |
| `config` | `HybridConfig` | Hybrid training configuration |
| `loss_history` | `list[float]` | All recorded loss values across all scopes and rounds |

### Methods

#### `stop() -> None`

Signal all training loops to stop.

#### `train(train_inputs, train_targets, *, on_step=None) -> list[float]`

```python
def train(
    self,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    *,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> list[float]
```

Run the hybrid training loop.

For each outer round, iterates over all subgraph scopes in post-order. Each scope is trained according to its `training_method`:

- `"surrogate"`: gradient-based edge optimization through surrogate approximations
- `"evolutionary"`: genetic algorithm edge optimization
- `"frozen"`: no training, just evaluate and log loss

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_inputs` | `np.ndarray` | *(required)* | Shape `(N, n_graph_inputs)` |
| `train_targets` | `np.ndarray` | *(required)* | Shape `(N, n_graph_outputs)` |
| `on_step` | `Callable \| None` | `None` | Callback receiving a step-info dict |

**Step-info dict keys:**

| Key | Type | Description |
|-----|------|-------------|
| `graph_path` | `list[str]` | Path from root to current scope |
| `graph_name` | `str` | Name of the subgraph being trained |
| `method` | `str` | Training method used |
| `round` | `int` | Outer round index |
| `local_step` | `int` | Local epoch/generation within this scope |
| `loss` | `float` | Current loss value |

**Returns:** Full loss history across all scopes and rounds.

### Training Flow

1. `graph.validate()` is called to verify structure.
2. Subgraph scopes are collected in post-order (leaves first).
3. For each outer round, each scope is trained using its declared method.
4. Surrogate caches are invalidated after each scope's training completes, so subsequent scopes see the updated weights.
5. Per-graph config is merged from `HybridConfig` defaults and the graph's own `surrogate_config` / `evo_config`.
