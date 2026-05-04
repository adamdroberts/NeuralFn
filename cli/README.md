# NeuralFn CLI

The in-repo CLI package exposes the `nfn` command for CUDA-oriented training,
inference, and evaluation flows outside the web editor. It builds composed
language-model recipes from a base model, topology, router mode, optional JEPA
objective, runtime, dataset, tokenizer, and run preset. It shares the same
graph builders, dataset manager, semantic vocabulary files, and artifact format
as the Python SDK and platform server.

## Setup

Create a virtualenv if needed, then install the repo root and CLI package in
editable mode:

```bash
cd ./cli
python -m venv .venv
source .venv/bin/activate
pip install -e ..
pip install -e .
```

The editable install in `./cli` registers the `nfn` entrypoint:

```bash
nfn --help
```

## Workflow model

Training is CUDA-only in practice and driven by `max_steps`, run presets, and
token-budgeted accumulation. The CLI builds graph contracts that match the
selected recipe:

- text `dataset_source` -> `tokens`, `targets`
- shipped `semantic_data_source` -> vocab-topic `sem_targets`

The semantic data path is vocab-only. The active `vocab_86d_*.json` file is the
source of truth, `semantic_data_source` materializes categorical topic IDs on
the fly, and semantic router recipes use one expert per semantic vocabulary
dimension. The `semantic_moe_jepa_evo` template adds shared and free experts
around that semantic bank, but the master CLI currently composes the
router-only and JEPA hybrid semantic recipes.

The low-level taxonomy-hash helpers still use `n_buckets` as their canonical
parameter, but the higher-level semantic APIs keep `n_sig_buckets`. The
compatibility alias is intentional, and the harness now uses the resolved
`--top-k` value when it estimates semantic rows and derived schedule metadata.

Semantic JEPA recipes train:

- routed autoregressive next-token loss
- JEPA latent loss
- semantic-alignment loss

It also exposes the parameter-golf-inspired trainer knobs that NeuralFn now
supports through `TorchTrainConfig`, while printing which reference knobs are
adapted versus only logged.

## Run the CLI

CUDA only:

```bash
nfn train --model llama --device cuda
```

The master CLI is the preferred entrypoint. Select a base model first and
optionally open the interactive planner:

```bash
nfn train --plan
nfn train --pretraining-file ./pretraining-data.txt
nfn infer --graph ~/NeuralFn/artifacts/nanogpt.json --prompt "Once upon a time"
nfn infer --checkpoint ~/NeuralFn/artifacts/final_model.pt --checkpoint-tokenizer ~/Downloads/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
nfn eval --preset semantic_router_moe --dataset shakespeare
```

`-h` on `nfn train`, `nfn infer`, and `nfn eval` now supports `short`,
`long`, and `verbose` help views, and `--plan` opens an interactive
questionnaire that only asks for omitted core options.

The legacy script entrypoints are still available:

```bash
python scripts/train_jepa_semantic.py --device cuda --max-steps 400
```

If you launch through `conda run`, use `--no-capture-output` so the progress
logs stream while the run is active:

```bash
conda run --no-capture-output -n NeuralFn python scripts/train_jepa_semantic.py --device cuda --max-steps 400
```

The script exits with an error if CUDA is unavailable or if you pass a
non-CUDA device.

Use `--evolutionary` to switch the torch trainer from gradient descent to
population-based search:

```bash
python scripts/train_jepa_semantic.py \
  --device cuda \
  --max-steps 50 \
  --evolutionary \
  --evo-population-size 32 \
  --evo-mutation-rate 0.1 \
  --evo-mutation-scale 0.05
```

The harness also ships CUDA-only autoregressive script pairs for the other
torch presets:

- `scripts/train_llama_fast.py` / `scripts/infer_llama_fast.py`
- `scripts/train_gpt2.py` / `scripts/infer_gpt2.py`
- `scripts/train_nanogpt.py` / `scripts/infer_nanogpt.py`

The GPT-2 and NanoGPT scripts support `--megakernel` directly:

```bash
python scripts/train_gpt2.py --device cuda --megakernel --max-steps 400
python scripts/infer_nanogpt.py --device cuda --megakernel --prompt "Once upon a time"
```

The same flows are available from the master CLI:

```bash
nfn train --model gpt2 --runtime megakernel
nfn infer --model nanogpt --runtime megakernel --prompt "Once upon a time"
```

By default the harness targets the existing cached-token parameter-golf alias:

- Local dataset name: `willdepueoai__parameter-golf__sp1024__train1`

If that alias is missing under `~/.cache/nfn/datasets/`, the training and inference
harnesses now try to download it automatically by default. Existing aliases are
still strict: if the alias already exists but its tokenizer-backed cache is
internally inconsistent, the harness surfaces the real tokenizer-contract error
instead of hiding it behind a missing-alias failure.

Override the cached alias with:

```bash
python scripts/train_jepa_semantic.py --dataset-alias willdepueoai__parameter-golf__sp1024__train1
```

The shared dataset-resolution flow is:

1. resolve the local cached alias under `~/.cache/nfn/datasets/`
2. if it is missing, attempt an auto-download
3. continue on success, or surface the original download / validator error

Standard cached-variant aliases like
`owner__repo__variant__trainN` are enough for automatic downloads on their own.
For non-standard aliases, pass the download contract explicitly:

- `--download-if-missing` / `--no-download-if-missing`
- `--dataset-hf-path`
- `--dataset-variant`
- `--dataset-train-shards`
- `--dataset-repo-id`
- `--dataset-remote-root-prefix`

For the cached parameter-golf shortcuts, `--tokenizer sp1024|sp2048|sp4096|sp8192`
is the canonical way to select the sentencepiece tokenizer variant. The legacy
`--dataset-variant` flag still works for cached-token aliases.

SentencePiece tokenizer assets are resolved separately from datasets:

- if the selected tokenizer is already present under `~/.cache/nfn/tokenizers`,
  the harness reuses it
- if a cached dataset already contains matching tokenizer files under its
  `tokenizers/` directory, the harness promotes them into the shared tokenizer
  cache automatically
- otherwise the harness downloads missing sentencepiece assets from the default
  tokenizer repo `sproos/parameter-golf-tokenizers`
- override that tokenizer source with `--tokenizer-hf-path`,
  `--tokenizer-repo-id`, `--tokenizer-remote-root-prefix`, and
  `--tokenizer-repo-type`

## Run inference against an exported checkpoint

The harness also ships a small CUDA-only text-generation probe for the exported
JEPA hybrid artifacts. It loads the saved graph JSON plus `.pt` weights,
traces the internal `model/softcap` or `model/lm_head` logits node, and then
samples autoregressively from that traced tensor.

```bash
python scripts/infer_jepa_semantic.py \
  --device cuda \
  --graph ~/NeuralFn/artifacts/jepa_semantic_hybrid_10min.json \
  --weights ~/NeuralFn/artifacts/jepa_semantic_hybrid_10min.pt \
  --dataset-alias willdepueoai__parameter-golf__sp1024__train1 \
  --prompt "Once upon a time" \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --top-k 32 \
  --repetition-penalty 1.1
```

The master `nfn infer` entrypoint can also load supported graphless Parameter
Golf root-GPT `.pt` checkpoints. These are not NeuralFn graph exports, so pass
the checkpoint and the matching SentencePiece model directly:

```bash
nfn infer \
  --checkpoint ~/NeuralFn/artifacts/final_model.pt \
  --checkpoint-tokenizer ~/Downloads/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  --checkpoint-log ~/Downloads/a54a53b3-7d6e-461c-975a-590030e61bd0.txt
```

`--weights ~/NeuralFn/artifacts/final_model.pt` is treated the same way when no
`--graph` is present. The graphless loader currently targets flat Parameter
Golf root-GPT checkpoints with `tok_emb.weight`, `skip_weights`, and
`blocks.*` attention/MLP tensors. It infers the architecture from tensor
shapes, uses the optional training log for safe non-structural hints such as
context window or logit softcap, and ignores newer experimental structural
hints that are not represented in the flat checkpoint tensors. CaseOps
tokenizers are displayed through a small cleanup layer that hides private-use
case markers and suppresses lossless reconstruction-only tokens during
sampling, including byte fallback, ellipsis artifacts, and the high-id
single-character fallback band. The chat UI and sampling flags are otherwise
the same as graph-backed inference.
Graphless sampling also enables a small repeat guard by default: it blocks the
fourth repeated n-gram and a fourth consecutive copy of the same token. Tune
loops with `--repetition-penalty`, `--no-repeat-ngram-size`, and
`--repeat-run-limit`; inside chat, `/repeat 1.15` raises the repetition
penalty without restarting the session.

Interactive `nfn infer` uses Tab in two ways. If the input starts with `/`, the
status line shows matching slash commands as you type, such as `/temp`,
`/top_k`, `/repeat`, `/autocomplete`, `/settings`, and `/help`; Tab completes
the visible match or lists ambiguous options, and value commands show their
expected argument. Use `/autocomplete n` to enable inline typing predictions
for `n` words, shown as 50% gray ghost text after the cursor. The prediction
keeps the model's generated boundary: a leading space starts a new word, while
no leading space completes the current word. Tab accepts the visible prediction.
Long prompts and ghost predictions can wrap across terminal rows; the input
repaint path clears the full wrapped block before drawing the next frame.
`/autocomplete 0` disables the inline mode and restores the normal prompt
behavior where Tab previews the next token and a second Tab inserts it when the
token can be inserted safely.

If `sentencepiece` is installed and the cached tokenizer model exists under the
dataset alias, the script prints both token ids and decoded text. If not, it
still runs in token-id mode and you can pass `--prompt-tokens 1,2,3`.

Raw-text tokenizer selection is now explicit across both training and
inference:

- dataset defaults are now dataset-driven instead of model-driven:
  `golf1` / `golf10` -> `sp1024`, `shakespeare` -> `cl100k_base`,
  `tinystories` -> `o200k_base`
- pass `--tokenizer gpt2|cl100k_base|o200k_base|sp1024|sp2048|sp4096|sp8192`
  to override the default
- sentencepiece downloads use the tokenizer source flags above and do not reuse
  the dataset download contract
- the legacy `--tokgpt2`, `--cl100k`, and `--o200k` flags still parse as
  shorthand aliases

Useful inference knobs:

- `--prompt`
- `--prompt-tokens`
- `--sem-targets`
- `--semantic-topics`
- `--max-new-tokens`
- `--temperature`
- `--top-k`
- `--top-p`
- `--repetition-penalty`
- `--no-repeat-ngram-size`
- `--repeat-run-limit`
- `--stop-token`
- `--log-every`
- `--context-window`
- `--logits-node`

`--semantic-topics` accepts a comma-separated dimension/topic map such as:

```bash
--semantic-topics emotion_sentiment=love,domain=psychology
```

That override uses the same fixed dimension-to-expert routing map as training.

## Important CLI knobs

Architecture:

- `--train-seq-len`
- `--num-layers`
- `--model-dim`
- `--num-heads`
- `--num-kv-heads`
- `--mlp-mult`
- `--multiple-of`
- `--experts`
- `--top-k`
- `--rope-base`
- `--qk-gain-init`
- `--logit-softcap`

Losses:

- `--ar-loss-coef`
- `--jepa-loss-coef`
- `--semantic-align-loss-coef`
- `--ema-decay`

Trainer / optimizer:

- `--max-steps`
- `--batch-size`
- `--train-batch-tokens`
- `--all-train-rows`
- `--evolutionary`
- `--evo-population-size`
- `--evo-mutation-rate`
- `--evo-mutation-scale`
- `--evo-crossover-rate`
- `--evo-tournament-size`
- `--evo-elite-count`
- `--evo-seed`
- `--optimizer-profile`
- `--learning-rate`
- `--embed-lr`
- `--head-lr`
- `--tied-embed-lr`
- `--matrix-lr`
- `--scalar-lr`
- `--warmup-steps`
- `--warmdown-fraction`
- `--max-wallclock-seconds`
- `--muon-momentum`
- `--muon-backend-steps`
- `--muon-momentum-warmup-start`
- `--muon-momentum-warmup-steps`
- `--beta1`
- `--beta2`
- `--adam-eps`
- `--grad-clip-norm`
- `--tokenizer`

The supplied lossless-caps Parameter Golf run is available as a preset stack:

```bash
nfn train \
  --model-preset parameter_golf_caseops_8192 \
  --run-preset parameter_golf_10min \
  --optimizer-preset parameter_golf_muon \
  --tokenizer sp8192
```

Choosing `--model-preset parameter_golf_caseops_8192` automatically recommends
the matching `parameter_golf_10min` run preset, `parameter_golf_muon` optimizer
preset, and `sp8192` tokenizer unless those flags are passed explicitly.

When `--evolutionary` is enabled, `--max-steps` counts generations and the
trainer ignores the gradient-only optimizer knobs such as
`--optimizer-profile`, the learning-rate family, Muon settings, Adam betas,
and gradient clipping. `--train-batch-tokens`, `--batch-size`,
`--all-train-rows`, and `--max-wallclock-seconds` still apply because they
define the data evaluated per generation.

Evaluation / logging:

- `--eval-batches`
- `--eval-batch-size`
- `--train-log-every`
- `--val-loss-every`

Most of these also accept matching environment variables such as
`ITERATIONS`, `TRAIN_BATCH_TOKENS`, `WARMUP_STEPS`, `WARMDOWN_FRACTION`,
`ROPE_BASE`, `QK_GAIN_INIT`, `EMBED_LR`, `MATRIX_LR`, and `SCALAR_LR`.

## Example

```bash
python scripts/train_jepa_semantic.py \
  --device cuda \
  --max-steps 200 \
  --train-seq-len 128 \
  --train-batch-tokens 8192 \
  --num-layers 4 \
  --model-dim 256 \
  --experts 8 \
  --embed-lr 0.02 \
  --matrix-lr 0.008 \
  --scalar-lr 0.004
```

The script will:

- reuse the cached dataset from `~/.cache/nfn/datasets/` when it already exists
- auto-download a missing cached alias by default when its download contract can
  be derived from the alias or explicit flags
- honor `--all-train-rows` by keeping partial final batches, finishing full
  epochs, and rounding `--max-steps` up to the next epoch boundary, with a
  2-epoch floor when the script defaults are left unchanged
- surface tokenizer-backed alias mismatches directly instead of replacing them
  with a generic missing-alias error
- log explicit startup, schedule, training, validation, and export stages
- log warmup and train-step progress during the run; `--train-log-every`
  controls the train-step interval
- load the cached tokenizer model for inference when it is present and
  `sentencepiece` is installed
- log the text and semantic data sources
- log the resolved `ModelSpec`, `TorchTrainConfig`, derived schedule, and
  adapted-versus-ignored reference knobs from `parameter-golf/train_gpt.py`
- derive the required epoch count from `max_steps`, the cached dataset length,
  and `train_batch_tokens`
- export weights and graph JSON when training completes

## Artifacts

By default the CLI and helper scripts write to `~/NeuralFn/artifacts`. Set
`NEURALFN_ARTIFACTS_DIR` to use a different shared artifact directory for CLI
training, inference, and graph-run defaults.

The default JEPA hybrid outputs are:

- `~/NeuralFn/artifacts/jepa_semantic_hybrid.pt`
- `~/NeuralFn/artifacts/jepa_semantic_hybrid.json`

Interrupted runs write:

- `~/NeuralFn/artifacts/jepa_semantic_hybrid.interrupted.pt`
- `~/NeuralFn/artifacts/jepa_semantic_hybrid.interrupted.json`

Press `Ctrl+C` once to request a clean stop after the current safe boundary.
Press `Ctrl+C` again to force an immediate abort.
