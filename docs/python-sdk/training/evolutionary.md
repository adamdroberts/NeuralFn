# neuralfn.evolutionary

Genetic-algorithm optimizer for edge weights (and optionally topology).

`neuralfn.evolutionary` is safe to import from the lean native/core SDK.
Importing `EvoConfig` or constructing `EvolutionaryTrainer` does not import
NumPy. Calling `EvolutionaryTrainer.train()` still requires NumPy for random
population generation and fitness reduction; missing NumPy raises an
`ImportError` before training starts.

---

## EvoConfig

```python
@dataclass
class EvoConfig:
    population_size: int = 50
    generations: int = 200
    mutation_rate: float = 0.1
    mutation_scale: float = 0.3
    crossover_rate: float = 0.5
    tournament_size: int = 3
    elite_count: int = 2
    topology_mutations: bool = False
    seed: int | None = None
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `population_size` | `int` | `50` | Number of individuals in the population |
| `generations` | `int` | `200` | Number of generations to evolve |
| `mutation_rate` | `float` | `0.1` | Per-gene probability of mutation |
| `mutation_scale` | `float` | `0.3` | Standard deviation of Gaussian mutation noise |
| `crossover_rate` | `float` | `0.5` | Per-gene probability of taking from parent 2 |
| `tournament_size` | `int` | `3` | Tournament selection pool size |
| `elite_count` | `int` | `2` | Number of elites carried to next generation |
| `topology_mutations` | `bool` | `False` | Enable topology mutations (reserved, not yet implemented) |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |

---

## EvolutionaryTrainer

```python
class EvolutionaryTrainer:
    def __init__(
        self,
        graph: NeuronGraph,
        config: EvoConfig | None = None,
        neuron_library: list[NeuronDef] | None = None,
    ) -> None
```

Evolves edge weight/bias parameters using tournament selection, uniform crossover, and Gaussian mutation. Evaluates fitness using the actual neuron functions (not surrogates).

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `NeuronGraph` | *(required)* | The graph to optimize |
| `config` | `EvoConfig \| None` | `None` | Evolution parameters (uses defaults if None) |
| `neuron_library` | `list[NeuronDef] \| None` | `None` | Available neurons for topology mutations (reserved) |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `graph` | `NeuronGraph` | The source graph |
| `config` | `EvoConfig` | Evolution configuration |
| `neuron_library` | `list[NeuronDef]` | Neuron catalog for topology mutations |
| `loss_history` | `list[float]` | Best loss per generation |

### Methods

#### `stop() -> None`

Signal the evolution loop to stop after the current generation.

#### `train(train_inputs, train_targets, *, fitness_fn=None, on_generation=None) -> list[float]`

```python
def train(
    self,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    *,
    fitness_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    on_generation: Callable[[int, float], None] | None = None,
) -> list[float]
```

Run the evolutionary optimization loop.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_inputs` | `np.ndarray` | *(required)* | Shape `(N, n_graph_inputs)` |
| `train_targets` | `np.ndarray` | *(required)* | Shape `(N, n_graph_outputs)` |
| `fitness_fn` | `Callable \| None` | `None` | Custom fitness function `(predictions, targets) -> loss`. Defaults to MSE. |
| `on_generation` | `Callable \| None` | `None` | Callback `(generation, best_loss)` |

**Returns:** List of best-loss-per-generation values.

After training, the best individual's parameters are written back to `self.graph`.
