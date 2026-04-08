# Agent Instructions

## Documentation updates are required

Whenever you change meaningful product behavior, setup steps, routing, auth/session flows, persistence, API contracts, MCP behavior, or operational workflows, update the docs in the same task.

- Update `README.md` with the current high-level usage, setup, and workflow guidance a user or developer needs immediately.
- Append a more detailed entry to `CHANGELOG.md` describing what changed, important implementation or migration notes, and how the change was verified.

## When this applies

Treat the documentation update requirement as mandatory for:

- user-facing feature or workflow changes
- backend or frontend setup/run instruction changes
- authentication, project/session, routing, dataset, or training workflow changes
- REST API or MCP contract changes
- operational or environment-variable changes

## Done criteria

A meaningful feature change is not complete until the relevant `README.md` and `CHANGELOG.md` updates are included.

## SDK documentation updates

Any change to public API surfaces must also update the corresponding page in `docs/`. This includes:

- **Python SDK changes** (new/changed classes, functions, methods, types in `neuralfn/`): update the matching page in `docs/python-sdk/` and, if it affects how developers build with the framework, the relevant `docs/framework-guide/` page.
- **REST API changes** (new/changed endpoints in `server/routers/`): update the matching page in `docs/rest-api/`.
- **MCP tool changes** (new/changed tools in `server/mcp_server.py`): update the matching page in `docs/mcp/` and the `.cursor/skills/neuralfn-mcp/SKILL.md` skill.
- **New builtin neurons**: update `docs/python-sdk/builtins.md`.
- **New template presets**: update `docs/python-sdk/config.md`, `docs/framework-guide/templates-and-presets.md`, and both the `neuralfn-torch` and `neuralfn-mcp` agent skills.
- **Server/editor internals**: update the relevant `docs/server/` or `docs/editor/` page.

Agent skills in `.cursor/skills/` should also be kept in sync when the tools or APIs they reference change.

## GPT template and variant library integrity

Any change that touches template graph builders (`neuralfn/torch_templates.py`), block/attention graph construction, variant wiring, `BlockSpec` / `TemplateSpec` / `ModelSpec` fields, builtin neuron port definitions (`neuralfn/builtins.py`), or torch module stages (`neuralfn/torch_backend.py`) **must** verify that all shipped presets still work end-to-end before the task is considered done.

### Required verification steps

1. Run the preset test suite:

```bash
python -m pytest tests/test_template_presets.py -x -q
```

This covers:
- `test_build_gpt_template_payload_supports_all_presets` — every preset produces a valid JSON payload with variant library and template_spec.
- `test_reported_presets_resolve_variant_libraries` — every preset's variant library resolves without port-count or port-name mismatches.
- `test_all_presets_compile_and_forward` — every preset builds, compiles to a `CompiledTorchGraph`, and runs a forward pass without errors.
- `test_apply_gpt_template_supports_all_presets` — the server-side `apply_gpt_template` path works for all presets.

2. If a new preset is added, append it to the `PRESETS` list in `tests/test_template_presets.py` so it is covered by all of the above tests.

### Common failure modes to watch for

- **Adding input/output ports to a subgraph** (e.g. attention graph) without updating the `input_aliases` / `output_aliases` passed by parent builders (`build_decoder_block_graph`, `build_model_stage_graph`, etc.). The frontend's `resolveVariantLibrary` will reject the mismatch.
- **Changing a module's forward signature** (number of inputs/outputs) without updating the corresponding `module_neuron()` port list in `builtins.py`. `CompiledTorchGraph` will fail at execution time.
- **Adding a new builtin neuron** without updating the expected-count list in `tests/test_builtin_neurons.py`.

### Variant library cross-contamination

All shipped presets share a flat variant-family namespace (`attention`, `mlp`, `attn_block`, etc.) at the root graph level. When the user loads template A and then template B, `mergeVariantLibrary` overwrites any family that both templates define. If a dense preset writes `mlp@default` with 1 output port and then an MoE preset overwrites it with 2 output ports, template A's block nodes still hold inline subgraph ports that no longer match the library entry.

Both `neuralfn/graph.py` (`resolve_variant_library`) and `editor/src/store/graphUtils.ts` (`resolveVariantLibrary`) handle this by **falling back to the node's inline subgraph** when the variant library entry is port-incompatible, instead of throwing. **Never revert this to a hard error.** If you add a new variant family or change port counts on an existing family, verify that loading any two different presets back-to-back in the same session does not break. The `VARIANT_FAMILY_ALIASES` table in `graphUtils.ts` and the Python-side `VARIANT_FAMILY_ALIASES` in `graph.py` can also cause unexpected resolution when families share alias chains.
