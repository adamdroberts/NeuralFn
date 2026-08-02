# Native embedding preparation helpers

`neuralfn.native_embedding` is the lightweight preparation and inspection
surface used by `nfn train --base-model embedding`. It does not import Torch or
NumPy.

```python
from neuralfn.native_embedding import (
    compile_embedding_datasets,
    import_huggingface_embedding_model,
)

metadata = compile_embedding_datasets(
    "embedding-datasets.json",
    "artifacts/embed/embedding_data.tsv",
    vocab_size=32_768,
    max_tokens=128,
    architecture="bert",
)

imported = import_huggingface_embedding_model(
    "sentence-transformers/all-MiniLM-L6-v2",
    "artifacts/embed/hf_import/embedding_model.bin",
)
```

Public helpers:

- `stable_token_id(token, vocab_size)` computes the same stable FNV-1a bucket
  used by native inference.
- `tokenize_embedding_text(text, vocab_size=..., max_tokens=...)` produces the
  uint32-compatible token IDs used in a compiled dataset.
- `tokenize_huggingface_text(text, tokenizer_dir, max_tokens=...)` uses copied
  `tokenizer.json` assets or the standard-library BERT WordPiece fallback.
- `load_embedding_manifest(path)` validates the top-level dataset-array shape.
- `compile_embedding_datasets(manifest, output_path, vocab_size=...,
  max_tokens=..., tokenizer_dir=None, architecture="bert")` loads supported
  sources, validates objective schemas, prepends the reserved native CLS token
  for from-scratch BERT data, and writes `embedding_indexed_v1` plus its JSON
  metadata sidecar. Use `architecture="gpt-derived"` for causal scratch data;
  imported HF tokenizers supply their own special-token processing.
- `inline_embedding_manifest(sources, objective="raw")` creates the shorthand
  used for repeated CLI `--embedding-dataset` inputs.
- `prepare_embedding_training_command(command, repo_root=...)` replaces
  manifest/source flags with the compiled native data path and, when
  `--embedding-hf-model` is present, imports the base and replaces geometry
  flags with its authoritative values.
- `import_huggingface_embedding_model(source, output_path, revision=None,
  pooling="mean", normalize=True)` downloads or opens a BERT-family or
  GPT-2-family HF checkpoint, maps wrapper-prefixed tensors into the
  full native transformer, copies tokenizer assets, and writes import
  provenance. Source GELU variants and LayerNorm epsilon are retained; integer
  position/token-type buffers are regenerated. F32, F16, BF16, and sharded
  safetensors are supported without Torch. Plain PyTorch ZIP state dicts are
  also decoded through a restricted unpickler; legacy non-ZIP pickle files are
  rejected.
- `resolve_native_embedding_cli(repo_root=None)` resolves
  `NFN_NATIVE_EMBEDDING_CLI`, the in-tree build, or an installed executable.
- `read_embedding_checkpoint_header(path)` inspects legacy `NFNEMB1` headers or
  current `NFNEMB2` headers, including transformer layers, heads, and MLP width.

`NFNEMB2` is the loadable native transformer format. The trainer does not load
legacy `NFNEMB1` compact token/position models because they contain no
self-attention or feed-forward tensors; create a new transformer base instead.

Parquet and Hugging Face Dataset sources need
`pip install -e ".[embeddings]"`; local TXT, JSONL, JSON, and CSV preparation
uses only the Python standard library.
