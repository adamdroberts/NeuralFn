# Agent Skills

NeuralFn ships with agent skills that teach AI coding assistants (Cursor, Codex, and similar tools) how to work with the platform. Skills are markdown instruction files that agents read automatically when the user's task matches the skill's trigger description.

## Available skills

| Skill | Path | Trigger |
|-------|------|---------|
| **NeuralFn Python SDK** | [`.cursor/skills/neuralfn-python-sdk/SKILL.md`](../.cursor/skills/neuralfn-python-sdk/SKILL.md) | When writing Python code that imports and uses the `neuralfn` package directly -- building graphs, defining neurons, training, serialization. |
| **NeuralFn Torch Models** | [`.cursor/skills/neuralfn-torch/SKILL.md`](../.cursor/skills/neuralfn-torch/SKILL.md) | When building, training, or exporting torch-backed neural network models (GPT, Llama, MoE, etc.) using the NeuralFn Python API. |
| **NeuralFn MCP** | [`.cursor/skills/neuralfn-mcp/SKILL.md`](../.cursor/skills/neuralfn-mcp/SKILL.md) | When using MCP tools to manipulate graphs, nodes, edges, datasets, and training runs through the NeuralFn MCP server. |

## What skills teach agents

- **neuralfn-python-sdk** covers the core graph framework: how to define neurons with `@neuron` and `Port`, build `NeuronGraph` instances, wire edges, create subgraphs, use the variant library, serialize graphs, and run all four training methods. It includes quick-reference tables for every class and common builtin IDs.

- **neuralfn-torch** covers the torch template system: `ModelSpec` / `BlockSpec` / `TemplateSpec` configuration, all 18 presets with config keys, `build_gpt_root_graph()` and `build_model_stage_graph()`, `CompiledTorchGraph` compilation, `TorchTrainer` for training, dataset loading, weight export/import, and `InferenceCache` for autoregressive generation.

- **neuralfn-mcp** covers the MCP tool interface: how to use tools like `get_graph`, `add_node`, `add_edge`, `load_gpt_template`, `train_start`, and `download_dataset` to build and train models through the MCP server. Includes end-to-end workflow examples.

## How skills work

Skills are stored in `.cursor/skills/<skill-name>/SKILL.md`. When a user's request matches the skill's `description` field, the AI agent reads the skill file and follows its instructions.

Each skill has:
- A YAML frontmatter with `name` and `description` (trigger keywords)
- Concise instructions optimized for agent context windows
- Optional supporting reference files for detailed tables

## Using skills in your project

If you're using Cursor, skills in `.cursor/skills/` are automatically available. For personal skills (available across all projects), place them in `~/.cursor/skills/`.

## Related documentation

The skills draw from and link into the full documentation:

- [Framework Guide](framework-guide/README.md) -- tutorial walkthroughs that the SDK and torch skills reference
- [Python SDK Reference](python-sdk/README.md) -- complete API details for every class and function
- [MCP Tools Reference](mcp/README.md) -- complete tool parameter reference that the MCP skill summarizes
- [Templates and Presets](framework-guide/templates-and-presets.md) -- full preset table referenced by the torch skill
