# Testing

NeuralFn uses a combination of `unittest.TestCase` and plain pytest-style `test_*` functions. There is no `conftest.py` or `pytest.ini`; tests are discovered from the `tests/` directory.

## Running the full suite

```bash
python -m unittest discover -s tests
```

Or with pytest (parallel-compatible):

```bash
python -m pytest tests/ -x -q
```

## Test files and coverage

| File | What it tests |
|------|---------------|
| `test_builtin_neurons.py` | Builtin catalog size, execution, JSON round-trip, custom neuron separation |
| `test_nested_graphs.py` | Subgraph execution and aliases, serialization round-trip, validation failures, hybrid trainer smoke |
| `test_torch_gpt.py` | GPT template as nested subgraph, torch graph round-trip, torch trainer loss, CUDA config guard |
| `test_template_presets.py` | All shipped presets: payload building, variant resolution, compile+forward, seq2seq families, legacy aliases, server apply path |
| `test_backend_capabilities.py` | Backend capabilities, megakernel, KV PCA, KV cache, quantized export, runtime wiring, InferenceCache |
| `test_server_nested_graphs.py` | Server API: recursive graph round-trip/execute, mixed training, torch trace, GPT template route |
| `test_platform_api.py` | Bootstrap admin, session scope, graph revision 409, project creation, dataset access |
| `test_server_dataset_loading.py` | Dataset source node rewiring, variant download options |
| `test_dataset_manager_variants.py` | Dataset manager cached variant download and token loading |
| `test_diffusion.py` | Diffusion template smoke test |
| `test_seq2seq.py` | Seq2seq template smoke test |
| `test_advanced_templates.py` | Jamba and ternary B158 template smoke tests |
| `test_kv_quant.py` | KV quantization path |
| `test_kv_pca.py` | PCA KV cache path |
| `test_adapters.py` | RandMap adapter in LLaMA-fast attention subgraph |
| `test_finetune.py` | LoRA/qLoRA, adapter checkpoints, SFT/DPO/PPO/reward graph wiring |
| `test_composed_recipes.py` | `build_composed_lm_spec()` dense/MoE/semantic recipe construction |
| `test_routing_diagnostics.py` | MoE routing diagnostics and traced route summaries |
| `test_pt_export.py` | `export_to_pt` / `import_from_pt` round-trip |

CLI-specific tests live under `cli/tests/` and cover `nfn` help behavior,
artifact defaults, composed train/infer/eval flows, dataset shortcuts,
pretraining-file handling, tokenizer promotion, validation fallback, and
megakernel artifact selection.

## Targeted checks

### Template presets (required after template/builtin/torch changes)

```bash
python -m pytest tests/test_template_presets.py -x -q
```

This runs:

- `test_build_gpt_template_payload_supports_all_presets` -- every preset produces a valid JSON payload
- `test_reported_presets_resolve_variant_libraries` -- variant resolution without port mismatches
- `test_all_presets_compile_and_forward` -- compile to `CompiledTorchGraph` and forward pass
- `test_apply_gpt_template_supports_all_presets` -- server-side `apply_gpt_template` path

### Platform API

```bash
python -m unittest discover -s tests -p "test_platform_api.py"
```

Covers bootstrap-admin, login/session scope, refresh-safe graph restore, and revision conflict handling.

### Server dataset loading

```bash
python -m unittest discover -s tests -p "test_server_dataset_loading.py"
```

### Frontend type/build validation

```bash
cd editor
pnpm build
```

Validates TypeScript types and Vite build without runtime.

### CLI checks

```bash
conda run -n NeuralFn python cli/nfn.py --help
conda run -n NeuralFn python -m pytest cli/tests/test_nfn_cli.py -q
conda run -n NeuralFn python -m pytest cli/tests/test_train_pretraining_file_flags.py -q
```

## Adding new presets

When adding a new template preset, append it to the `PRESETS` list in `tests/test_template_presets.py` so all preset tests cover it automatically.

## Adding new builtin neurons

When adding a new builtin neuron to `neuralfn/builtins.py`, update the expected-count list in `tests/test_builtin_neurons.py`.
