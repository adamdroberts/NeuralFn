# Training Methods

NeuralFn supports three scalar-graph training methods and one torch-runtime trainer. Each method optimizes different aspects of a `NeuronGraph`.

The lean native/core SDK can import the public training configuration surface
without Torch or NumPy. Surrogate training execution still relies on the legacy
NumPy/PyTorch surrogate stack, so install those packages explicitly before
calling `SurrogateTrainer.build_surrogates()` or `SurrogateTrainer.train()`.

## Overview

| Method | Module | Optimizes | Runtime | Use Case |
|--------|--------|-----------|---------|----------|
| **Surrogate** | [surrogate](surrogate.md) / [trainer](trainer.md) | Edge weights/biases | Scalar | Small graphs, differentiable approximation |
| **Evolutionary** | [evolutionary](evolutionary.md) | Edge weights/biases (+ optional topology) | Scalar | Non-differentiable neurons, exploration |
| **Hybrid** | [hybrid](hybrid.md) | Edge weights/biases across nested subgraphs | Scalar | Nested graphs with mixed training methods |
| **Torch** | [torch-backend](../torch-backend.md) | All nn.Module parameters | Torch | GPU training of transformer/LLM graphs |

## When to Use Each

**Surrogate training** (`SurrogateTrainer`) is the default for scalar graphs. It probes each neuron to build a differentiable MLP surrogate, then backpropagates through the surrogate chain to optimize edge weights/biases. Best for small-to-medium graphs where neurons are continuous functions. Importing the trainer module and constructing the trainer are lean; the probe/training calls require the optional legacy NumPy/PyTorch stack.

**Evolutionary training** (`EvolutionaryTrainer`) uses a genetic algorithm with tournament selection, crossover, and mutation. It evaluates the actual neuron functions (not surrogates) so it works with any neuron type, including discontinuous or noisy functions. Slower but more robust.

**Hybrid training** (`HybridTrainer`) orchestrates training across nested subgraph hierarchies. Each subgraph can declare its own `training_method` (`"surrogate"`, `"evolutionary"`, or `"frozen"`), and the hybrid trainer walks the graph tree in post-order, training each scope independently while evaluating the root graph for loss.

**Torch training** (`TorchTrainer`) compiles the graph into a native PyTorch `nn.Module` pipeline and trains with standard GPU-accelerated gradient descent. Required for graphs containing module neurons (transformers, attention, MoE, etc.).

## Modules

- [surrogate.md](surrogate.md) -- `probe_neuron`, `SurrogateModel`, `train_surrogate`, `build_surrogates`
- [trainer.md](trainer.md) -- `TrainConfig`, `SurrogateTrainer`
- [evolutionary.md](evolutionary.md) -- `EvoConfig`, `EvolutionaryTrainer`
- [hybrid.md](hybrid.md) -- `HybridConfig`, `GraphScope`, `HybridTrainer`
