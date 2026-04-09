# NeuralFn Python SDK Reference

API reference for the `neuralfn` Python package.

## Installation

```bash
pip install -e .
```

From a sibling project outside the repo:

```bash
pip install -e /home/adam/dev/innovation/NeuralFn
```

This editable install now packages the shipped semantic data files under
`neuralfn/data/semantic/`, so SDK code that imports `neuralfn.semantic` can
load `vocab_8d.json` and `training_100k_8d.csv` without extra setup.

## Package Exports

All public symbols are available from the top-level `neuralfn` module:

```python
from neuralfn import (
    Port,
    NeuronDef, module_neuron, neuron, neuron_from_source, subgraph_neuron,
    BuiltinNeurons,
    Edge, NeuronInstance, NeuronGraph,
    SurrogateModel, probe_neuron, build_surrogates,
    SurrogateTrainer,
    EvolutionaryTrainer,
    HybridConfig, HybridTrainer,
    save_graph, load_graph,
    TorchTrainConfig, TorchTrainer,
    build_gpt_root_graph, build_model_stage_graph,
)
```

## Modules

| Module | Description |
|--------|-------------|
| [port](port.md) | `Port` dataclass -- declares input/output slots on neurons |
| [neuron](neuron.md) | `NeuronDef` and factory functions for defining neurons |
| [graph](graph.md) | `NeuronInstance`, `Edge`, and `NeuronGraph` -- the core graph data model |
| [builtins](builtins.md) | 58 built-in neuron definitions (scalar activations, torch modules, MoE, etc.) |
| [config](config.md) | `TemplateSpec`, `BlockSpec`, `ModelSpec` and preset builder functions |
| [torch-backend](torch-backend.md) | `CompiledTorchGraph`, `TorchTrainer`, `TorchTrainConfig`, and all `*Stage` modules |
| [torch-templates](torch-templates.md) | Graph builders for attention, MLP, decoder blocks, and full model architectures |
| [training/](training/README.md) | Training methods: surrogate, evolutionary, and hybrid |
| [inference](inference.md) | Weight export/import, quantization, and `InferenceCache` for autoregressive generation |
| [serialization](serialization.md) | `save_graph` / `load_graph` -- JSON persistence |

## Quick Start

```python
import neuralfn as nf

# Build a NanoGPT model graph
graph = nf.build_gpt_root_graph(name="my_model")

# Save / load
nf.save_graph(graph, "model.json")
graph = nf.load_graph("model.json")

# Define a custom neuron
@nf.neuron(
    inputs=[nf.Port("x", range=(-10, 10))],
    outputs=[nf.Port("y", range=(0, 1))],
)
def my_activation(x):
    return 1.0 / (1.0 + 2.718 ** (-x))
```
