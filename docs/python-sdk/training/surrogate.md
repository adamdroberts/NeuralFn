# neuralfn.surrogate

Probing and surrogate model training for scalar neuron graphs.

---

## probe_neuron

```python
def probe_neuron(
    neuron_def: NeuronDef,
    n_samples: int = 10_000,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]
```

Sample a neuron's transfer function by feeding random inputs drawn uniformly from each input port's range. Returns `(inputs, outputs)` arrays of shape `(n_samples, n_ports)`.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `neuron_def` | `NeuronDef` | *(required)* | The neuron to probe |
| `n_samples` | `int` | `10_000` | Number of random samples |
| `rng` | `np.random.Generator \| None` | `None` | Random generator (uses default if None) |

### Returns

`(xs, ys)` where:
- `xs`: `np.ndarray` of shape `(n_samples, n_inputs)`, dtype `float32`
- `ys`: `np.ndarray` of shape `(n_samples, n_outputs)`, dtype `float32`

---

## SurrogateModel

```python
class SurrogateModel(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
    ) -> None
```

A small MLP trained to approximate a neuron's transfer function. Uses SiLU activations between hidden layers.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_inputs` | `int` | *(required)* | Number of input features |
| `n_outputs` | `int` | *(required)* | Number of output features |
| `hidden_sizes` | `tuple[int, ...]` | `(64, 64)` | Hidden layer dimensions |

### Methods

#### `forward(x: Tensor) -> Tensor`

Standard PyTorch forward pass through the MLP.

---

## train_surrogate

```python
def train_surrogate(
    model: SurrogateModel,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> float
```

Train a `SurrogateModel` on probed data using Adam optimizer and MSE loss.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `SurrogateModel` | *(required)* | The model to train |
| `xs` | `np.ndarray` | *(required)* | Input data from `probe_neuron` |
| `ys` | `np.ndarray` | *(required)* | Output data from `probe_neuron` |
| `epochs` | `int` | `200` | Training epochs |
| `batch_size` | `int` | `256` | Mini-batch size |
| `lr` | `float` | `1e-3` | Learning rate |

### Returns

`float` -- the final epoch-averaged MSE loss.

---

## build_surrogates

```python
def build_surrogates(
    graph: NeuronGraph,
    n_samples: int = 10_000,
    hidden_sizes: tuple[int, ...] = (64, 64),
    epochs: int = 200,
    lr: float = 1e-3,
) -> dict[str, SurrogateModel]
```

Probe every neuron in the graph and build a trained surrogate for each. Returns a dict mapping `instance_id -> trained SurrogateModel` (in eval mode).

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `NeuronGraph` | *(required)* | The graph to build surrogates for |
| `n_samples` | `int` | `10_000` | Samples per neuron |
| `hidden_sizes` | `tuple[int, ...]` | `(64, 64)` | Surrogate MLP hidden layers |
| `epochs` | `int` | `200` | Training epochs per surrogate |
| `lr` | `float` | `1e-3` | Learning rate |

### Returns

`dict[str, SurrogateModel]` -- trained surrogates keyed by node instance_id.
