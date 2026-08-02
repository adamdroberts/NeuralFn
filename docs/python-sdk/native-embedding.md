# Native embedding preparation helpers

`neuralfn.native_embedding` is the lightweight preparation and inspection
surface used by `nfn train --base-model embedding`. It does not import Torch or
NumPy.

```python
from neuralfn.native_embedding import compile_embedding_datasets

metadata = compile_embedding_datasets(
    "embedding-datasets.json",
    "artifacts/embed/embedding_data.tsv",
    vocab_size=32_768,
    max_tokens=128,
)
```

Public helpers:

- `stable_token_id(token, vocab_size)` computes the same stable FNV-1a bucket
  used by native inference.
- `tokenize_embedding_text(text, vocab_size=..., max_tokens=...)` produces the
  uint32-compatible token IDs used in a compiled dataset.
- `load_embedding_manifest(path)` validates the top-level dataset-array shape.
- `compile_embedding_datasets(manifest, output_path, vocab_size=...,
  max_tokens=...)` loads supported sources, validates objective schemas, and
  writes `embedding_indexed_v1` plus its JSON metadata sidecar.
- `inline_embedding_manifest(sources, objective="raw")` creates the shorthand
  used for repeated CLI `--embedding-dataset` inputs.
- `prepare_embedding_training_command(command, repo_root=...)` replaces
  manifest/source flags with the compiled native data path.
- `resolve_native_embedding_cli(repo_root=None)` resolves
  `NFN_NATIVE_EMBEDDING_CLI`, the in-tree build, or an installed executable.
- `read_embedding_checkpoint_header(path)` validates `NFNEMB1` and returns the
  version, dimensions, step, and adapter kind.

Parquet and Hugging Face Dataset sources need
`pip install -e ".[embeddings]"`; local TXT, JSONL, JSON, and CSV preparation
uses only the Python standard library.
