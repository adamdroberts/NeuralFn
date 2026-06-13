# Datasets

NeuralFn manages training datasets through a server-side catalog, with download, upload, and loading APIs. A special `dataset_source` module node feeds tokenized data directly into a graph during training.

## Dataset catalog

Each NeuralFn project has its own dataset catalog. Datasets can be:

- **Downloaded from HuggingFace** via the server API.
- **Uploaded** as local text or binary files.
- **Listed** and inspected through REST or MCP tools.

---

## Downloading from HuggingFace

Use the REST endpoint to pull a dataset into a project's catalog:

```
POST /api/projects/{project_id}/datasets/download
```

Request body:

```json
{
    "repo_id": "karpathy/tiny_shakespeare",
    "filename": "input.txt"
}
```

The server downloads the file, stores it in the project's dataset directory, and registers it in the catalog.

For tokenizer-backed cached shard aliases (`data_format="uint16_shards"`), the download step now validates the shard ids against the downloaded tokenizer artifacts before the alias is accepted. If any cached token id is outside the tokenizer vocab, NeuralFn rejects the alias and asks you to re-download or rebuild it with matching tokenizer files.

---

## Uploading local files

```
POST /api/projects/{project_id}/datasets/upload
```

Attach the file as multipart form data. The server saves it and adds a catalog entry.

---

## The dataset_source node

In a torch graph, the `dataset_source` module node acts as the data input. It has `module_type="dataset_source"` and emits tokenized tensors directly into the graph during training.

```python
from neuralfn.neuron import module_neuron
from neuralfn import Port

ds = module_neuron(
    name="data",
    module_type="dataset_source",
    input_ports=[],
    output_ports=[Port("tokens"), Port("targets")],
    module_config={"dataset_id": "tiny_shakespeare", "seq_len": 128},
)
```

Template builders insert a `dataset_source` node automatically. Its output port layout adapts to the template's objective.

---

## Dataset roles by template shape

Different template objectives require different data shapes. The `dataset_source` node's output ports adapt accordingly:

| Template objective | Output roles | Description |
|-------------------|-------------|-------------|
| AR (autoregressive) | `tokens`, `targets` | Standard next-token prediction. Targets are tokens shifted by one position. |
| H-Net | `tokens`, `targets` | Raw byte sequences (vocab_size=256). Same shift as AR. |
| Universal | `tokens`, `targets` | Same as AR but fed through ACT recurrence. |
| Seq2Seq | `enc_tokens`, `dec_tokens`, `targets` | Encoder input, decoder input, and decoder targets. |
| Diffusion | `tokens` | Single token sequence. Timestep sampling is internal. |
| JEPA | `tokens` | Single token sequence. Masking and target generation are internal. |
| Semantic routing presets (`semantic_router_moe`, `jepa_semantic_hybrid`, `semantic_dense_jepa_evo`, `semantic_moe_jepa_evo`) | `tokens`, `targets` | Text next-token inputs come from `dataset_source`; the shipped `semantic_data_source` provides vocab-derived `sem_targets` separately. |

---

## Byte vs tokenized paths

Most presets use tiktoken-based tokenization: text is split into subword tokens and the model's `vocab_size` is set accordingly (e.g. 32000 for Llama-scale models, 256 for small experiments).

The `hnet_lm` preset is the exception. It operates on raw bytes:
- `vocab_size` is locked to 256.
- The dataset pipeline skips tokenization and feeds raw byte values directly.
- The `byte_patch_size` config key controls how bytes are grouped into patches for the model.

Tokenizer-backed cached shard aliases are now treated as strict contracts:
- cached token ids must stay within the tokenizer vocab
- the graph/checkpoint vocab must match that tokenizer vocab
- bad aliases are considered invalid cache artifacts and must be deleted and rebuilt or re-downloaded

Raw-text aliases also get a local training cache when the selected tokenizer can
be stored in `uint16`. The first training load streams `data.txt` and optional
`val.txt` into `fineweb_train_000000.bin` and `fineweb_val_000000.bin`, updates
`meta.json` with `data_format="uint16_shards"` and
`token_cache_format="raw_text_uint16_shards"`, and later training runs memmap
those files directly. Schedule estimation reads token counts or shard sizes
without constructing a dataset. Tokenizers whose ids exceed `uint16` remain raw
text. The `dataset_source` node stores references and `seq_len`; real text and
token arrays stay in the dataset cache rather than graph-editor node state.

The compiled native GPT-2 resolver also accepts existing llm.kittens
TinyStories token bins without converting them into NeuralFn cache shards. When
`--tinystories` is used, `nfn_gpt_native_train` checks
`/mnt/disk2/dev/open-source/llm.kittens/dev/data/tinystories` for
`TinyStories_train.bin` and `TinyStories_val.bin`; set
`NFN_LLM_KITTENS_TINYSTORIES_DIR` to override that location. Passing a direct
`TinyStories_train.bin` path as `--dataset-alias` infers the sibling validation
bin in C++, so the startup path stays Torch-free and does not materialize
raw-text shards first.

---

## MCP tools for datasets

The NeuralFn MCP server exposes dataset management tools for AI agents:

| Tool | Description |
|------|-------------|
| `download_dataset` | Download a dataset from HuggingFace into the current project. |
| `load_dataset_source` | Attach a cataloged dataset to a graph's `dataset_source` node. |
| `list_datasets` | List all datasets in the current project's catalog. |

See [MCP: Dataset tools](../mcp/dataset-tools.md) for parameters and examples.

---

## Related pages

- [REST API: Datasets](../rest-api/datasets.md) -- full endpoint documentation.
- [Templates and Presets](templates-and-presets.md) -- how template objectives determine dataset roles.
- [Training Workflows](training-workflows.md) -- using datasets with `TorchTrainer`.
