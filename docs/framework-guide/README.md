# NeuralFn Framework Guide

NeuralFn is a graph-native neural network framework. Every computation -- a math function, a PyTorch module, or an entire sub-model -- is a **node** with typed input and output **ports**, wired together by weighted **edges** into a directed graph. The same graph representation scales from a three-node XOR classifier up to a multi-billion-parameter transformer with mixture-of-experts routing.

This guide walks through the core concepts with runnable Python examples. It assumes you have NeuralFn installed and can `import neuralfn`.

## Pages

| Page | What it covers |
|------|---------------|
| [Defining Neurons](defining-neurons.md) | The four ways to create neurons: `@neuron` decorator, `neuron_from_source()`, `module_neuron()`, and `subgraph_neuron()`. Port types, NeuronDef properties. |
| [Building Graphs](building-graphs.md) | `NeuronGraph` construction, adding nodes and edges, execution, tracing, topology queries, cyclic graphs, edge parameters, and serialization. Full XOR walkthrough. |
| [Subgraphs and Variants](subgraphs-and-variants.md) | Nested graphs, port aliasing, the variant library, variant references, resolution and fallback behavior, family aliases. |
| [Torch Models](torch-models.md) | Module neurons, built-in module types, `CompiledTorchGraph`, and building tensor-native graphs by hand. |
| [Templates and Presets](templates-and-presets.md) | `ModelSpec` / `BlockSpec` / `TemplateSpec`, the 16 shipped presets, config keys, and building custom model variants. |
| [Training Workflows](training-workflows.md) | Surrogate, evolutionary, hybrid, and torch training with complete examples and config references. |
| [Inference and Export](inference-and-export.md) | Weight export/import, quantized checkpoints, and autoregressive generation with `InferenceCache`. |
| [Datasets](datasets.md) | Dataset management, HuggingFace downloads, upload, the `dataset_source` node, role adaptation by template shape, and byte-level paths. |

## Related resources

- [Python SDK Reference](../python-sdk/README.md) -- complete API documentation for every class and function.
- [Agent Skills](../agent-skills.md) -- AI-ready skills for driving NeuralFn from agents and assistants.
