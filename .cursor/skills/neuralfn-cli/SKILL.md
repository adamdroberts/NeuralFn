---
name: neuralfn-cli
description: >-
  Use or modify the NeuralFn nfn CLI for train, infer, eval, dataset shortcuts,
  tokenizer selection, artifact paths, fine-tuning flags, CUDA harness scripts,
  and CLI tests. Use when the user mentions nfn, cli/, train scripts,
  pretraining files, cached datasets, tokenizers, or graph/weights artifacts.
---

# NeuralFn CLI

Use this skill for `cli/` and `nfn` work. For Python SDK internals use
`neuralfn-python-sdk`; for torch graph/preset internals use `neuralfn-torch`;
for platform MCP tools use `neuralfn-mcp`.

Canonical docs:

- `docs/cli.md` -- concise CLI workflow reference.
- `cli/README.md` -- longer operator runbook.
- `docs/framework-guide/datasets.md` -- dataset/source contracts.
- `docs/framework-guide/templates-and-presets.md` -- `ModelSpec` and fine-tuning roots.
- `llms-full.txt` -- full documentation bundle.

## Core rules

- Treat datasets and tokenizers as separate sources. Dataset aliases live under
  `~/.cache/nfn/datasets`; shared SentencePiece tokenizer assets live under
  `~/.cache/nfn/tokenizers`.
- Keep `--pretraining-file` as a first-class direct raw-text training input.
- Keep `--dataset tinystories` and `--tinystories` aligned on the same raw-file
  contract.
- Keep `sp1024`, `sp2048`, `sp4096`, and `sp8192` visible and allow missing
  shared tokenizer assets to download before training.
- Default CLI/server artifacts to `~/NeuralFn/artifacts`, unless
  `NEURALFN_ARTIFACTS_DIR` is set.
- Saved graph JSON is graph-first. Prefer graph metadata for weights,
  tokenizer, and training manifests; treat `--weights` as an override.
- Save artifacts before validation; validation failures should not erase a
  successful training artifact.

## Entry points

| Surface | File |
|---------|------|
| Master CLI | `cli/nfn.py`, `cli/nfn_impl.py` |
| Shared helpers | `cli/cli_utils.py`, `cli/scripts/cli_utils.py` |
| Dataset/tokenizer selector | `cli/scripts/train_jepa_semantic.py` |
| Inference helpers | `cli/scripts/infer_jepa_semantic.py` |
| CUDA train scripts | `cli/scripts/train_*.py` |
| CUDA infer scripts | `cli/scripts/infer_*.py` |
| CLI tests | `cli/tests/` |

## Verification

Prefer non-training checks unless the user explicitly asks for a training run:

```bash
conda run -n NeuralFn python cli/nfn.py --help
conda run -n NeuralFn python cli/nfn.py train --help
conda run -n NeuralFn python -m pytest cli/tests/test_nfn_cli.py -q
conda run -n NeuralFn python -m pytest cli/tests/test_train_pretraining_file_flags.py -q
conda run -n NeuralFn python -m pytest cli/tests/test_train_tinystories_flags.py -q
```

If a change touches template graph builders, builtin port definitions,
`BlockSpec` / `TemplateSpec` / `ModelSpec`, or torch stages, also run:

```bash
conda run -n NeuralFn python -m pytest tests/test_template_presets.py -x -q
```

## Documentation

CLI workflow changes must update `README.md`, `CHANGELOG.md`, `docs/cli.md`,
`cli/README.md`, and LLM artifacts. Public SDK/config changes must also update
the matching `docs/python-sdk/` page and this skill when CLI behavior depends
on the changed API.
