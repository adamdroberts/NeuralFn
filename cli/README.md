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
./install.sh
```

The installer keeps Torch optional, registers the `nfn` entrypoint, builds the
native GPT C++ binding, launcher, no-Python cached-shard CLI, unified
native training frontend, and the compiled native `nfn-native` train/infer
shim, then links `nfn-gpt-native`,
`nfn-gpt-native-train`, compatibility `nfn-gpt2-native` names,
`nfn-native-train`, `nfn-native`, and `nfn-gpt2-tile-launcher` into
the active Python scripts directory. Use `./install.sh --no-native` to skip C++
artifact builds.

The editable install registers the `nfn` entrypoint:

```bash
nfn --help
```

The root install no longer pulls in Torch by default. Root `nfn --help` /
no-argument startup, `nfn train|infer|eval --help`, `nfn kernels ... --help`,
`nfn kernels list [--json]`, CUDA Tile registry metadata, and native GPT-2
compatibility training do not import it. Install
`pip install -e ".[tile-cuda]"` for Torch-free native CUDA Tile build tooling,
`pip install -e ".[datasets]"` for tokenization/cache preparation, or
`pip install -e ".[server]"` for the editor/backend. NeuralFn no longer exposes
a `.[torch]` extra; legacy graph-backed PyTorch training/inference requires a
separately managed PyTorch install outside NeuralFn package metadata.
Importing `nfn_impl` for parser/planner helpers also keeps Torch, NumPy,
`server.dataset_manager`, graph ops, parameter-golf Torch helpers, and
`train_jepa_semantic` lazy until a command path actually needs the graph-backed
runtime.

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
nfn kernels list --json
nfn kernels bench --device auto --iterations 200 --samples 5
nfn kernels examples
```

`-h` on `nfn train`, `nfn infer`, and `nfn eval` now supports `short`,
`long`, and `verbose` help views, and `--plan` opens an interactive
questionnaire that only asks for omitted core options.

The legacy script entrypoints are still available:

```bash
python scripts/train_jepa_semantic.py --device cuda --max-steps 400
```

CUDA Tile diagnostics and example generation live under `nfn kernels`. `nfn train --help`, `nfn infer --help`, `nfn eval --help`, and `nfn kernels ... --help` use lightweight static help that does not import `nfn_impl`, Torch, or the graph-backed runtime. `nfn kernels list [--json]` is the metadata-only coverage path; add `--status tile|torch_fallback|host_only|delegated|planned` and/or `--kind function|module|optimizer|runtime` to filter the emitted specs while preserving the global coverage totals. `nfn kernels doctor` reports toolchain and coverage status, `nfn kernels bench` compares graph-walk PyTorch, static compiled PyTorch, and Tile-requested execution on a small scalar graph with paired interleaved samples and paired ratios, and `nfn kernels examples --write --output-dir examples/tile_cuda` regenerates checked-in examples plus one generated SDK snippet per registry entry. For native candidate-vs-older command comparisons, `python tools/paired_kernel_speed.py --baseline "OLD_COMMAND" --candidate "NEW_COMMAND" --cuda-visible-devices 0` alternates command order across samples on the dedicated CUDA GPU so unrelated external GPU load is shared across both variants; keep the default warmup pair enabled to exclude first-use CUDA/kernel load. For `nfn train`, `nfn infer`, and `nfn eval`, selecting `--kernel-backend tile-cuda` now defaults to strict kernel enforcement; use `--no-tile-cuda-strict` only when intentionally comparing fallback behavior.

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

The harness also ships inference entry points for existing graph-backed
artifacts:

- `scripts/train_gpt2.py` / `scripts/infer_gpt2.py`
- `scripts/infer_gpt2.py --evo`
- `scripts/infer_llama_fast.py`
- `scripts/infer_nanogpt.py`

The plain GPT training script is native-only. It resolves the dataset cache,
writes/uses uint16 token shards, and launches the compiled Tile-CUDA C++ trainer
without importing Torch or sending training batches through graph nodes. Importing
`scripts/train_gpt.py` or the compatibility `scripts/train_gpt2.py`, building
their parser, and resolving defaults are also Torch-free. Direct
`python cli/scripts/train_gpt.py ...` native runs set up
their own repo/script import path, so they do not need `PYTHONPATH`.
The compiled trainer keeps per-block allocation, parameter initialization, and
AdamW-state zeroing in the block-vector visitors, including block 0, and reports
the `block0_duplicate_*_elided` startup flags, including activation allocation,
under `block_state_layout`. It also suballocates float buffers from one aligned
CUDA device arena and token buffers from combined int64/device-uint16/pinned-uint16
arenas, reporting `float_allocation_strategy: "single-arena"` and
`token_buffer_allocation_strategy: "combined-arenas"`.
`NFN_DATASETS_DIR` overrides the native alias cache root for Python and
compiled native CLI paths. Native dense GPT inference should point `nfn infer
--checkpoint` at the compiled trainer output directory or a native
`model_########.bin` file. The GPT-2 inference script still supports `--evo`
for legacy `gpt2_evo.pt/json` eager artifacts; its parser, `--help`, and
`--evo`/`--megakernel` artifact default resolution stay off the Torch,
dataset-manager, and NumPy import path, but legacy `.pt/.json` token generation
loads the graph-backed runtime after argument parsing. Shared legacy inference
helpers imported from `scripts/infer_jepa_semantic.py`, including dataset
selector/download helpers, are lazy wrappers so test collection and parser/help
startup can import them without immediately loading the graph-backed runtime:

```bash
python scripts/train_gpt.py --device cuda --tinystories --eval-every-steps 1000
python scripts/train_gpt.py --model-family gpt3 --device cuda --tinystories --native-cuda-print-command --native-cuda-dry-run
python scripts/infer_gpt2.py --device cuda --evo --prompt "Once upon a time"
python scripts/infer_nanogpt.py --device cuda --megakernel --prompt "Once upon a time"
```

The master CLI uses the same no-Torch native dispatcher for explicit dense
GPT pretraining. Use `nfn train --base-model gpt` as the canonical command;
`gpt2` and `gpt3` are aliases, and `gpt3` only changes the default context to
2048 when no explicit template, graph, or sequence length is supplied. Unless
you override them, dense GPT training expands the same quality defaults as
`scripts/train_gpt.py`: validation every 250 optimizer steps over 20 batches,
sample/checkpoint cadence, `64 x 1024 -> 524288` token batching, AdamW defaults,
60 warmup steps, 20,000 max steps, and GELU/MOA activation defaults. With the
default `compiled-cli` runner it goes directly to the no-Python cached-shard C++
CLI before importing `train_gpt_native`, `nfn_impl`, or Torch.
`--template-name` / `--template` / `--preset` accepts every name in
`neuralfn.config.SHIPPED_GPT_TEMPLATE_PRESETS`, and `--graph-file` / `--graph`
selects a custom graph JSON; wrappers canonicalize the aliases to
`--template-name` and `--graph-file` at handoff. Unsupported template shapes fail
with native missing-trainer JSON instead of falling back to Torch.
Use `nfn train --base-model gpt --list-templates` or wrapper alias
`--native-cuda-list-templates` to print the compiled no-data support catalog
for shipped GPT templates plus the public `gpt`/`gpt3` aliases. The action
exits before dataset or token-shard resolution and reports each selector's
native support status.
`--base-model gpt` is the canonical dense GPT surface. `gpt2` and `gpt3` route
to the same C++ trainer while preserving their selected `--model-family`
labels; `gpt3` defaults to a 2048-token context only when no template, graph,
or `--train-seq-len` is explicit. The full `nfn train` parser, planner, and compatibility graph builder
accept those same dense GPT aliases; when a graph-backed compatibility path is
used, the alias is canonicalized to the GPT-compatible template builder and the
template or graph still decides the architecture. Direct
`python cli/nfn.py ...` invocations use the same lightweight dispatcher:

```bash
nfn train --base-model gpt --dataset tinystories --eval-every-steps 1000
nfn train --base-model gpt3 --dataset tinystories --native-cuda-print-command --native-cuda-dry-run
nfn train --base-model gpt --dataset tinystories --native-cuda-runner launcher
nfn train --base-model gpt --dataset tinystories --native-cuda-runner binding
```

For an already-cached uint16 dataset, bypass Python entirely. Build the linked
trainer on workstations so the Tile ops ABI is resolved from the executable and
the trainer can pass `--tile-ops-lib linked` instead of paying the dynamic
Tile-ops loader path:

```bash
bash tools/build_native_gpt_cli_linked.sh
# fallback only: bash tools/build_native_gpt_cli.sh
nfn-gpt-native --dataset-alias roneneldan__TinyStories__TinyStoriesV2-GPT4
nfn-gpt-native --dataset-alias /path/to/cached-dataset --dry-run
nfn-gpt-native --dataset-alias /path/to/cached-dataset --backend tile-cuda --print-plan
```

The Python-facing GPT harness defaults to `--native-cuda-runner compiled-cli`,
which prefers `build/nfn_gpt_native_train_linked` when it exists and falls back
to the dynamic no-Python cached-shard CLI at `build/nfn_gpt_native_train`.
Explicit `--native-cuda-runner auto` uses that same direct compiled C++ fast path
for `python scripts/train_gpt.py` and dense `nfn train` commands; use `binding`
or `launcher` only when you intentionally want those wrapper routes.
That compiled CLI exposes `--backend tile-cuda`, the default and only NeuralFn-owned
12-layer transformer/LM loop over the raw trainer ABI. Use `--backend tile-cuda
--print-plan` or `--check-tile-ops --tile-ops-lib PATH` to inspect it, and use
`--eval-every-steps 1000` when you want validation loss every 1000 optimizer
steps. Use `--train-loss-every-steps 1000`, `--train-log-every 1000`, or
`--train-log-every-steps 1000` when you also want sampled native training loss;
the default is `0` so timing-only runs do not evaluate train loss. When enabled,
the compiled loop records train loss inside the folded LM-head backward
recompute, not through graph-editor nodes or a duplicate forward LM-head loss
pass. The compiled loop honors `train_batch_tokens` by deriving
`grad_accum_steps`, running that many cached-shard CUDA Tile microbatches,
averaging gradients on device, and applying clip plus AdamW once per optimizer
step. Wrapper-level `--native-cuda-dry-run --native-cuda-print-command` is
command inspection only on the default `compiled-cli` runner: it builds the C++
argv from the dataset alias/path without importing `server.dataset_manager`,
NumPy, tiktoken, or Torch and without writing raw-text token shards.
For startup/preflight probes, pass `--native-cuda-fast-startup` or
`--fast-startup`; the wrappers normalize either spelling to the compiled C++
`--fast-startup` flag so the trainer skips throughput-only setup prewarms
without changing the normal long-training default. The compiled SM120
workstation helper does the same normalization before it execs the native GPT
trainer, so printed argv from `nfn-train-gpt-sm120` and
`tools/train_gpt_sm120.sh` stays canonical.
Full `--train-transformer-lm` runs also emit `cuda_runtime_preflight` before
allocation and fail early when the driver is unavailable or older than the
loaded CUDA runtime.
Build the SDK binding with `bash tools/build_native_gpt2_binding.sh`, the
launcher with `bash tools/build_native_gpt2_launcher.sh`, the no-Python
cached-shard CLI with `bash tools/build_native_gpt_cli.sh`, and the linked
cached-shard CLI with `bash tools/build_native_gpt_cli_linked.sh`; those CLIs
link the shared no-Torch `token_shards.cpp` resolver. Installed GPT command
symlinks prefer the linked CLI when it exists, matching the lower-startup Python
dispatch path. Build the unified
native frontend with `bash tools/build_native_train_cli.sh`; it dispatches dense GPT aliases
to the cached-shard CLI and sends NanoGPT defaults to that same transformer-LM
trainer with `--template-name nanogpt`, while explicit `--train-token-lm`
still reaches the NanoGPT token-only native target. Use
`nfn-native-train --base-model gpt ...` when you
want the compiled top-level training command, and `nfn-native-train
--list-models --json` to inspect native coverage. Build the compiled
top-level native shim with `bash tools/build_native_nfn_cli.sh`; installed
`nfn-native train ...` execs `nfn_native_train` before Python can start, and
`nfn-native infer --checkpoint artifacts/gpt/model_00020000.bin --prompt-tokens
50256,464` or `nfn-native infer --checkpoint artifacts/gpt --native-info`
execs `nfn_gpt_native_train` against native `model_*.bin` checkpoints. Checkpoint
directories resolve to the highest-step `model_########.bin`; graph-backed
`.pt/.json` inference remains on the Python `nfn infer` path. Default `nfn train` commands
hand off to this compiled frontend before graph-backed Python can start; dense
GPT and NanoGPT are implemented through the dense GPT target, and LLaMA,
GPT-2 evo, JEPA, semantic/MoE, and DeepSeek
variants intentionally report missing or preflight-only native trainers. SDK
calls to `run_native_gpt(..., runner="auto")` still try the Python SDK binding,
compiled CLI, then launcher in order and allow Python raw-text materialization;
use `binding` to require the C++ binding,
`compiled-cli` to require the cached-shard C++ frontend, or `launcher` to
require the compiled launcher. Alias-only configs from
`build_native_gpt2_compiled_cli_run_config()` still execute the compiled CLI argv
through the SDK binding, so cached-shard resolution stays in C++ even when
`runner="auto"` selects `neuralfn._native_gpt2`.

Non-dense-GPT `nfn train` commands now fail from the compiled native registry by
default. Direct legacy training scripts hand off to the same native registry
before they import Torch, because their internal implementation still uses the
graph-backed `TorchTrainer` path. The old `NFN_ALLOW_TORCH_TRAINING` bypass is
ignored by CLI training entrypoints; call the Python SDK trainer APIs directly
for one-off graph-backed experiments while native C++ trainers are added for
those model families. NanoGPT normal training invocations now use the shared
dense GPT transformer-LM route: `nfn train --base-model nanogpt ...` and
`python cli/scripts/train_nanogpt.py ...` add `--template-name nanogpt` and
`--train-transformer-lm` automatically. Pass `--train-token-lm` when you want
the older tied-token-LM NanoGPT native loop. `--dry-run` and `--print-command`
inspect the selected native route without starting the loop; explicit native
actions such as `--print-plan`, `--check-tile-ops`, or a smoke command still run
exactly as requested.

SDK callers can build the unified native binding with
`bash tools/build_native_train_binding.sh` and use `neuralfn.native_train`
(`build_native_train_run_config`, `run_native_train`,
`native_train_model_registry`) to hand off to the compiled frontend without
importing Torch.

Native C++ trainer implementations should link the raw Tile ops shared library
built by `bash tools/build_native_train_tile_ops.sh`. It exposes a C ABI over
the CUDA Tile AdamW, gradient accumulation, global-norm clip scale finalization,
device-buffer fill/zeroing, deterministic GPT-2 token-weight initialization,
device-scalar gradient scaling, reduction, linear,
linear input/weight/bias/bias-accumulate backward, scaled residual add, fused QKV split/merge for
NanoGPT `qkv.weight`, GELU forward/backward, token embedding forward/weight backward,
absolute-position embedding forward/backward/backward-accumulate, RMSNorm, RMSNorm input backward, LayerNorm, LayerNorm input/affine/affine-accumulate backward, softmax, token and masked
token cross-entropy partial, token and masked token cross-entropy logits
backward, and scaled dot-product attention forward/backward kernels without including
`torch/extension.h`. CE logits backward uses row-wise Tile kernels for
vocabularies up to 1024 and chunked row-wise kernels with reusable row-stat
workspace for full GPT-class vocabularies, avoiding the previous elementwise
large-vocab fallback. Linear weight and bias backward switch large row counts to
row-chunked tiled atomic accumulation instead of one serial row loop per output
element.
They should also use the native token-shard resolver in
`neuralfn/csrc/native_train/token_shards.cpp` for `NFN_DATASETS_DIR`,
`fineweb_train_*.bin` / `fineweb_val_*.bin` validation, token counts, and
microbatch/gradient-accumulation metadata without routing real data through
graph nodes. The same C++ code skips cached-shard headers and provides a
sequential sampler for token plus next-token target batches.

`bash tools/build_native_missing_trainers.sh` builds compiled per-family native
entrypoints (`nfn_nanogpt_native_train`, `nfn_llama_native_train`, and related
families). NanoGPT has a model-aware C++ preflight path: run
`nfn_nanogpt_native_train --print-plan --require-token-shards --sample-token-batch`
to validate the native shape, AdamW optimizer profile, cached token shards,
effective token schedule, contiguous parameter/gradient/AdamW-state buffer
layout, AdamW decay/no-decay groups, forward/backward/optimizer
`training_step_plan`, and first native token/target batch as JSON without
importing Python or Torch. The native preflight defaults to `dropout_p=0.0`;
the tied LM head backward path is represented through the raw linear backward
ABI, and nonzero `--dropout-p` reports the missing dropout ABI as required work.
Use `--check-tile-ops --tile-ops-lib PATH` to `dlopen` the raw trainer shared
library and verify every NanoGPT-required ABI symbol from the compiled binary.
Use `--smoke-tile-ops --tile-ops-lib PATH` to load CUDA runtime, allocate a tiny
device buffer, execute `nfn_native_tile_fill_float32`, copy it back, and verify
the value without Python or Torch. Pass `--cuda-runtime-lib PATH` or set
`NFN_CUDA_RUNTIME_LIB` when libcudart is not on the default loader path.
Use `--smoke-optimizer-step --tile-ops-lib PATH` to build the NanoGPT parameter
layout, initialize contiguous param/grad/AdamW moment buffers through raw fill
kernels, execute `nfn_native_tile_adamw_step_float32` once per registered
parameter buffer with that buffer's decay/no-decay setting, copy param and
moment buffers back, and verify the update on the compiled path.
Use `--smoke-training-loop-step --tile-ops-lib PATH` to exercise the native
optimizer-loop mechanics over the registered NanoGPT parameter layout: gradient
zeroing, synthetic gradient fill, global-norm clip scale finalization,
device-scalar gradient scaling, and per-buffer AdamW updates.
Use `--smoke-lm-step --tile-ops-lib PATH` to execute a tiny tied-embedding
language-model step through token embedding, linear logits, token CE
loss/backward, linear input/weight backward, token embedding weight backward,
and AdamW update kernels, then verify loss, gradient, and weight update values
without Python or Torch.
Use `--smoke-token-train-step --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS`
to sample a real native uint16 token/target batch from cached shards, execute
the tied-LM forward/backward/update kernels over those IDs, and verify
sampled-batch loss, gradient, and weight update values without Python or Torch.
Use `--train-token-lm --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS --max-steps N`
to run that tied token-embedding LM path as a real multi-step native loop over
cached shards; it streams batches with the C++ sampler, zeros gradients on
device, runs token CE backward, applies AdamW each step, and emits JSON metrics
without Python or Torch. Normal NanoGPT training now uses the shared dense GPT
transformer-LM trainer with `--template-name nanogpt`; keep `--train-token-lm`
only when you specifically want this older tied token-embedding loop.
Set `--eval-every-steps N`, `--eval-batches N`, and `--eval-batch-size N` to
emit periodic validation losses from the resolved validation token shards inside
that compiled loop; those validation batches do not pass through graph-editor
nodes, `TorchTrainer`, or Python dataset payloads.
Use `--smoke-embedding-norm-step --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS`
to run sampled tokens through token and absolute-position embeddings, residual
add, LayerNorm forward/backward, tied logits, CE backward, embedding/position
gradient kernels, and AdamW updates, then verify residual, norm, loss, gradient,
and weight update values without Python or Torch.
Use `--smoke-qkv-layout-step --tile-ops-lib PATH` to verify the fused QKV
split/merge layout kernels used between NanoGPT `attn.qkv.weight`, SDPA, and
the fused QKV gradient buffer.
Use `--smoke-fused-qkv-attention-step --tile-ops-lib PATH` to execute a tiny
attention stage through one fused `attn.qkv.weight`, QKV split, SDPA
forward/backward, QKV gradient merge, fused qkv weight backward, output
projection backward, and AdamW updates for the fused qkv/output weights.
Use `--smoke-transformer-block-step --tile-ops-lib PATH` to compose LayerNorm,
fused-QKV attention, residual adds, MLP, backward passes, gradient
accumulation, and AdamW updates for a tiny transformer block through raw native
kernels.
Use `--smoke-mlp-step --tile-ops-lib PATH` to execute a tiny MLP stage through
fc projection, GELU, output projection, projection/input backward, GELU
backward, and AdamW updates for both MLP weights, then verify forward,
gradient, and weight update values without Python or Torch.
Use `--smoke-attention-step --tile-ops-lib PATH` to execute a tiny attention
stage through Q/K/V projections, SDPA forward/backward, output projection
forward/backward, Q/K/V projection backward, and AdamW updates for all
attention weights, then verify forward, gradient, and weight update values
without Python or Torch.
The non-dense-GPT targets still intentionally fail for real training until the
family-specific CUDA Tile work lands, and give the unified frontend a compiled
target to dispatch to while the real trainers are implemented. Dense GPT,
GPT-2, GPT-3, and NanoGPT transformer-LM training use the shared
`nfn_gpt_native_train` CUDA Tile loop.
Installed per-family targets are linked with both underscore and hyphen names,
and `NFN_NATIVE_<MODEL>_CLI` can override a single family, for example
`NFN_NATIVE_NANOGPT_CLI`.

`tools/install_native_gpt2_commands.sh` links the stable command names without
running package installation. Override the destination with
`NFN_NATIVE_GPT2_BIN_DIR=/path/to/bin`.

When the dataset already has `fineweb_train_*.bin` and `fineweb_val_*.bin`
uint16 shards, the native GPT-2 path does not import `server.dataset_manager`,
NumPy, tiktoken, or Torch and does not scan the full dataset to estimate the
training schedule before launching native code.

The CUDA training harnesses default to `--optimizer-profile adamw` for
RTX 5090/SM120 runs. That profile uses the llm.kittens SM120 AdamW schedule:
20,000 steps, sequence length 1024, microbatch 64, 524,288 tokens/step,
learning rate 0.0006, weight decay 0.1, 60 warmup steps, validation cadence
250 for the shared dense GPT native trainer, and cosine decay to zero when no
explicit LR schedule is provided. In the native GPT-2 path this schedule is passed straight to
`train_gpt2cu`; in graph-backed Tile CUDA runs, `adamw` dispatches batched AdamW
updates and gradient clipping through CUDA Tile optimizer kernels rather than
`torch.optim.AdamW`.
Use `--optimizer-profile parameter_golf` only when you explicitly
want the split/Muon experimental path. The CUDA training harnesses default
`--max-wallclock-seconds` to `0`, so they run to the resolved step/epoch schedule
unless you explicitly request an early wallclock cutoff.

The same flows are available from the master CLI:

```bash
nfn train --model gpt2 --runtime megakernel
nfn infer --checkpoint ~/NeuralFn/artifacts/gpt2_evo --prompt-tokens 50256 --max-new-tokens 64
nfn infer --model nanogpt --runtime megakernel --prompt "Once upon a time"
```

By default the native GPT-2 harness targets TinyStories with the GPT-2 tokenizer:

- Local dataset name: `roneneldan__TinyStories__TinyStoriesV2-GPT4`
- Hugging Face source: `roneneldan/TinyStories`
- Train/validation files: `TinyStoriesV2-GPT4-train.txt` and `TinyStoriesV2-GPT4-valid.txt`

The cached-token parameter-golf aliases are still available through the explicit
`golf1` and `golf10` shortcuts, but they are no longer the default training
dataset.

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

Native GPT cached-shard training is stricter: `train_gpt.py`,
`train_gpt2.py`, and `train_gpt_native.py` do not auto-download missing
datasets by default. Prepare shards ahead of time, pass a direct cache path, or
opt in explicitly with `--download-if-missing` when you want the Python dataset
manager involved before the compiled CUDA Tile trainer launches.

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

Raw-text aliases are converted to a file-backed token cache when the tokenizer
fits in `uint16`. The first training load writes `fineweb_train_000000.bin` and
an optional `fineweb_val_000000.bin` beside `data.txt` / `val.txt`; schedule
estimation then uses metadata or shard sizes, and subsequent runs memmap the
token shards instead of re-tokenizing the raw text. Tokenizers whose ids exceed
`uint16` stay on the raw-text path. Graph `dataset_source` nodes keep only alias
and sequence-length config, not real text payloads.

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

Native GPT-2 checkpoints from `train_gpt2cu` are `.bin` files named
`model_########.bin` with matching `DONE_########` markers. They are recognized
without importing Torch:

```bash
nfn infer --checkpoint ~/NeuralFn/artifacts/gpt2/model_00020000.bin --native-info
python scripts/infer_gpt2.py --native-checkpoint ~/NeuralFn/artifacts/gpt2/model_00020000.bin --native-info
```

That path reports the native header shape, precision, expected size, and marker
state. Prompt generation from native `.bin` checkpoints is token-id native by
default:

```bash
nfn infer --checkpoint ~/NeuralFn/artifacts/gpt2/model_00020000.bin --prompt-tokens 50256 --max-new-tokens 64
python scripts/infer_gpt.py --native-checkpoint ~/NeuralFn/artifacts/gpt2/model_00020000.bin --prompt-tokens 50256
```

The wrapper dispatches to `nfn_gpt_native_train --sample-checkpoint` before
graph-backed inference, Torch, NumPy, tokenizers, or dataset managers are
imported. Raw text `--prompt` is intentionally disabled on the native path
unless `NFN_NATIVE_GPT_ALLOW_PYTHON_TOKENIZER=1` is set; pass
`--prompt-tokens` to keep inference tokenizer-free. Native checkpoint sampling
uses the same dedicated display-disabled CUDA GPU selector as native training
when `CUDA_VISIBLE_DEVICES` is unset; set `CUDA_VISIBLE_DEVICES` or pass an
explicit SDK `cuda_visible_devices` value to pin another device.

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
- auto-download a missing cached alias by default in the graph-backed sibling
  harnesses when its download contract can be derived from the alias or
  explicit flags; native GPT cached-shard training requires explicit
  `--download-if-missing`
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
