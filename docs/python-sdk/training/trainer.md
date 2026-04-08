# neuralfn.trainer

Gradient-based training of edge weights/biases through differentiable surrogates.

---

## TrainConfig

```python
@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    epochs: int = 500
    batch_size: int = 32
    surrogate_samples: int = 10_000
    surrogate_hidden: tuple[int, ...] = (64, 64)
    surrogate_epochs: int = 200
    loss_fn: str = "mse"
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `learning_rate` | `float` | `1e-3` | Adam learning rate for edge parameters |
| `epochs` | `int` | `500` | Number of training epochs |
| `batch_size` | `int` | `32` | Mini-batch size |
| `surrogate_samples` | `int` | `10_000` | Samples for probing each neuron |
| `surrogate_hidden` | `tuple[int, ...]` | `(64, 64)` | Hidden layer sizes for surrogate MLPs |
| `surrogate_epochs` | `int` | `200` | Epochs for training each surrogate |
| `loss_fn` | `str` | `"mse"` | Loss function: `"mse"` or `"bce"` |

---

## SurrogateTrainer

```python
class SurrogateTrainer:
    def __init__(
        self,
        graph: NeuronGraph,
        config: TrainConfig | None = None,
    ) -> None
```

Trains edge weights and biases by backpropagating through differentiable surrogate approximations of each neuron.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `NeuronGraph` | *(required)* | The graph to train |
| `config` | `TrainConfig \| None` | `None` | Training configuration (uses defaults if None) |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `graph` | `NeuronGraph` | The source graph |
| `config` | `TrainConfig` | Training configuration |
| `surrogates` | `dict[str, SurrogateModel]` | Built surrogate models (populated by `build_surrogates`) |
| `loss_history` | `list[float]` | Per-epoch loss values |

### Methods

#### `stop() -> None`

Signal the training loop to stop after the current epoch.

#### `build_surrogates() -> None`

Probe all neurons and train surrogate MLPs. Freezes surrogate parameters (they are not trained further). Called automatically by `train` if `self.surrogates` is empty.

#### `train(train_inputs, train_targets, *, on_epoch=None) -> list[float]`

```python
def train(
    self,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    *,
    on_epoch: Callable[[int, float], None] | None = None,
) -> list[float]
```

Train edge weights by backpropagating through surrogates.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_inputs` | `np.ndarray` | *(required)* | Shape `(N, n_graph_inputs)` |
| `train_targets` | `np.ndarray` | *(required)* | Shape `(N, n_graph_outputs)` |
| `on_epoch` | `Callable[[int, float], None] \| None` | `None` | Callback `(epoch, loss)` |

**Returns:** List of per-epoch average losses.

After training, the optimized edge weights/biases are written back to `self.graph`.
