# neuralfn.tile_cuda

Optional CUDA Tile backend for NeuralFn. The package provides configuration, diagnostics, coverage reporting, strict graph coverage checks, and an opt-in PyTorch extension build path for CUDA Tile scalar, module, optimizer, and runtime kernels. PyTorch remains the authoritative fallback for unsupported devices, dtypes, shapes, and tensor contracts.

The public `neuralfn.tile_cuda` package exports are lazy. Importing registry/config metadata such as `from neuralfn.tile_cuda.registry import coverage_report` or running `nfn kernels list [--json]` does not import Torch or the graph-backed runtime. Tensor execution helpers, diagnostics that inspect `torch.cuda`, and extension build/load helpers still import their backing modules when those symbols are requested.

## Configuration

```python
from neuralfn.tile_cuda import TileCudaConfig

config = TileCudaConfig(
    backend="auto",       # "auto", "torch", or "tile_cuda"
    strict=False,
    report_path=None,
    build_enabled=False,  # or set NFN_TILE_CUDA_BUILD=1
    arch=None,            # or set NFN_TILE_CUDA_ARCH=sm_120
)
```

`backend="auto"` uses CUDA Tile only when the runtime and selected graph are supported. `backend="torch"` forces the existing PyTorch path. `backend="tile_cuda"` requests CUDA Tile and will fall back to PyTorch unless strict mode is enabled.

The generic Python extension source build path is intentionally opt-in:

```bash
NFN_TILE_CUDA_BUILD=1 NFN_TILE_CUDA_ARCH=sm_120 nfn kernels doctor
```

On CUDA Toolkit 13.3+, the generic Python extension and the trainer-facing raw C
ABI share the same `neuralfn/csrc/tile_cuda/kernels.cu` source. Generic BF16
conversion, BF16 activation storage, BF16 bias-add, and row-chunked bias
reduction helpers are available even when the Python extension is compiled
without the trainer-only TK attention/cuBLAS macros. After reinstalling CUDA,
use:

```bash
NFN_TILE_CUDA_TEST=1 python -m pytest \
  tests/test_tile_cuda_gpu.py \
  tests/test_tile_cuda_ops.py \
  tests/test_tile_cuda_optimizer.py \
  -q -rs
```

Those tests should execute GPU coverage; "CUDA Tile extension could not be built
or loaded" means the extension build path still needs diagnosis.

CUDA Tile native builds require CUDA Toolkit 13.3 or newer, `cuda_tile.h`, C++20, `nvcc --enable-tile`, and `ninja`. The default SDK install is the lean native/core surface; it does not install Torch, NumPy, tokenizer, dataset, graph-analysis, or server packages. Install the native build extra with:

```bash
pip install -e ".[tile-cuda]"
```

Install `pip install -e ".[datasets]"` separately for raw-text tokenization and HF dataset cache materialization, and `pip install -e ".[server]"` for the FastAPI/editor/MCP backend. NeuralFn no longer exposes a `.[torch]` extra; graph-backed PyTorch execution and the legacy PyTorch Tile extension loader require a separately managed PyTorch install outside NeuralFn's package metadata. The aggregate `.[all]` extra intentionally remains Torch-free.

The trainer-facing raw C ABI build is separate:

```bash
bash tools/build_native_train_tile_ops.sh
```

The raw ABI includes an opt-in dense GPT token-weight startup route,
`nfn_native_tile_init_gpt2_token_weight_fast_with_bf16_shadow_padded_float32`,
which initializes the public vocabulary rows and zeroes the padded vocabulary
tail plus BF16 shadow in one launch. The native trainer leaves this route
default-off because the CUDA 13.3 dedicated RTX 5090 startup-only retest
rejected it at `1.011381x` setup wall time versus the current vector4
BF16-shadow initializer, despite a noise-equivalent `0.995702x` token-init
sub-bucket. Enable it only for paired bisection with
`NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_PADDED_INIT=1`. Runtime JSON reports
`token_weight_padded_init_fusion_requested`,
`token_weight_padded_init_fusion_available`,
`token_weight_padded_init_fusion_enabled`, and
`token_weight_padding_zero_launches_elided`.

On SM120 with TK attention enabled, that script defines
`LLMK_SM120_USE_CUBLASLT_GEMM` by default to match the supported llm.kittens
CUDA 13.3 build path. It also normalizes inherited
`NFN_TILE_CUDA_ARCH=sm_120` / `compute_120` settings to `sm_120a` /
`compute_120a` for TK builds. CUDA Toolkit 13.3 rejects several raw TK GEMM
instantiations that exceed the default static shared-memory cap under the
generic target, so local raw-TK GEMM bisections should be treated as experiments
rather than the supported trainer-facing library build.

After rebuilding the native GPT CLI or trainer-facing Tile ops library, verify
the compiled artifacts did not regain Torch, c10, or Python runtime links:

```bash
python tools/check_native_no_torch_deps.py
```

Pass explicit artifact paths to check candidate builds, or add `--json` for a
machine-readable CI report. The gate also rejects hard pyproject dependencies
on Torch, NumPy, tokenizers, datasets, graph-analysis packages, and server
runtime packages so native installs stay lean. It imports the top-level native SDK
exports such as `NativeGptRunConfig`,
`build_native_gpt_compiled_cli_run_config()`, `native_gpt_kernel_backend()`,
and `native_gpt_parameter_count()` while blocking `torch`, NumPy, tiktoken,
`server.dataset_manager`, and `nfn_impl`, so lazy public exports stay on the
no-Torch path.

Native GPT runtime JSON exposes the trainer-facing cuBLASLt grouped-layout
probe as `linear_cublaslt_grouped_layout_probe_available`,
`linear_cublaslt_grouped_layout_probe_requested`,
`linear_cublaslt_grouped_layout_probe_status`, and
`linear_cublaslt_grouped_layout_supported`. Request it with
`NFN_NATIVE_GPT_PROBE_CUBLASLT_GROUPED_LAYOUT=1`,
`NFN_NATIVE_GPT2_PROBE_CUBLASLT_GROUPED_LAYOUT=1`, or
`NFN_TILE_CUDA_LINEAR_CUBLASLT_GROUPED_LAYOUT_PROBE=1`. The probe creates and
destroys a minimal grouped matrix-layout descriptor through the raw Tile C ABI
so CUDA 13.3 grouped-GEMM experiments can be gated by actual local support
before a candidate route is benchmarked. It is diagnostic-only; a supported
probe does not change the default dense GPT training schedule.

Runtime JSON also exposes a classic cuBLAS grouped BF16 GEMM execution probe as
`linear_cublas_grouped_bf16_gemm_probe_available`,
`linear_cublas_grouped_bf16_gemm_probe_requested`,
`linear_cublas_grouped_bf16_gemm_probe_status`, and
`linear_cublas_grouped_bf16_gemm_supported`. This raw Tile ABI probe is
opt-in with `NFN_NATIVE_GPT_PROBE_CUBLAS_GROUPED_BF16_GEMM=1` because rejected
or unsupported grouped BF16 launches can poison the CUDA context before model
arena allocation. When requested, a nonzero probe status fails native preflight
immediately instead of continuing into model arena allocation. The
probe launches tiny aligned BF16 grouped GEMMs through
`cublasGemmGroupedBatchedEx`, copies back the BF16 outputs, and verifies the
expected sums. Use it to gate future grouped linear-backward candidates; it is
diagnostic-only and does not change default training
dispatch.

Set `NFN_NATIVE_GPT_STAGE_TIMING=1` only for CUDA event attribution runs; the
same-script throughput wrappers leave it off by default. The dense GPT timing
JSON includes `block_backward.mlp_proj.grad_out_bf16`, which isolates the
projection-gradient float32-to-BF16 pack from the surrounding MLP projection
dWeight and dInput kernels.

Native BF16 `cublasGemmEx` fallback paths expose default-off bisection controls
for CUDA 13.3+ performance work. Set
`NFN_NATIVE_LINEAR_BF16_GEMM_EX_ALGO=N` or
`NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_ALGO=N` to try
`CUBLAS_GEMM_ALGO<N>_TENSOR_OP` globally, or use
`NFN_NATIVE_LINEAR_BF16_GEMM_EX_ALGO_SHAPE=m,n,k,opA,opB,N` /
`NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_ALGO_SHAPE=...` to isolate a single GEMM
shape while leaving all other BF16 GEMMEx calls on their existing defaults.
The dense GPT LM-head dHidden shape is
`768,8192,50304,N,N,N`. The tokens `default` and `default_tensor_op` force the
plain and tensor-op cuBLAS defaults. Leave these unset for normal training.

`NFN_NATIVE_GPT_BF16_ATTENTION_DPREP_GRAD_OUT=1` is a default-off SM120
attention-backward diagnostic. It packs the float attention dO tensor to BF16
only for the packed-attention dprep/backward wrapper, without switching the
attention projection dInput kernel to a BF16 output. Runtime JSON reports
`attention_backward_bf16_dprep_grad_out_enabled` and
`attention_backward_grad_out_dtype: "bf16-dprep-pack"` when this path is active.
The SM120 wrapper profiles `bf16_attention_grad_out`,
`bf16_attention_dprep_grad_out`, and
`attention_dprep_float_hd64_specialized` are rejected diagnostics on the current
CUDA 13.3 RTX 5090 route, so real paired-wrapper launches require
`NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1`; dry-run expansion remains
available for route inspection.

On the SM120 workstation this defaults to `NFN_TILE_CUDA_USE_TK_ATTENTION=1`,
`NFN_TILE_CUDA_ARCH=sm_120a`, and links the local llm.kittens /
ThunderKittens headers via `LLM_KITTENS_ROOT` and `TK_ROOT`. Set
`NFN_TILE_CUDA_USE_TK_ATTENTION=0` only when you intentionally want the older
float32 row-scan attention diagnostic build.
The SM120 trainer build also pins the llm.kittens-style NVCC threading,
host-compiler, data-prep, memory, and LayerNorm tuning flags for the
ThunderKittens headers. It does not define llm.kittens' cublasLt GEMM switch
because NeuralFn owns cublasLt initialization and dispatch in the native
trainer.

## Native GPT Trainer Handoff

The dense GPT native path can bypass the graph-backed `TorchTrainer` and hand cached token shards directly to a compiled C++/CUDA trainer. Prefer the generic `neuralfn.native_gpt` API for new code; `neuralfn.native_gpt2` remains the compatibility module behind the same dense decoder kernel family.

```python
from pathlib import Path
from neuralfn.native_gpt import build_native_gpt_compiled_cli_run_config, build_native_gpt_run_config, run_native_gpt

config, updated_meta = build_native_gpt_run_config(
    dataset_name="roneneldan__TinyStories__TinyStoriesV2-GPT4",
    dataset_path=Path("~/.cache/nfn/datasets/roneneldan__TinyStories__TinyStoriesV2-GPT4").expanduser(),
    dataset_meta={},
    encoding_name="gpt2",
    executable=None,
    output_dir=Path("~/NeuralFn/artifacts/gpt").expanduser(),
    eval_every_steps=1000,
    sample_every_steps=20000,
    generate_tokens=144,
    checkpoint_every_steps=200,
    batch_size=64,
    seq_len=1024,
    train_batch_tokens=524288,
    learning_rate=0.0006,
    min_lr=None,
    warmup_steps=60,
    weight_decay=0.1,
    max_steps=20000,
    num_layers=12,
    activation="gelu",
    model_family="gpt",
)

print(config.command())
run_native_gpt(config)
```

`build_native_gpt_run_config()` resolves `fineweb_train_000000.bin` plus `fineweb_val_000000.bin` directly when the dataset is already cached as matching uint16 shards. That fast path avoids importing `server.dataset_manager`, NumPy, tiktoken, or Torch and does not scan the full dataset to estimate a schedule before launch. `build_native_gpt_compiled_cli_run_config()` is even lighter for the no-Python cached-shard CLI: it accepts a dataset alias/path and leaves shard metadata validation to the compiled C++ resolver, so Python does not need `meta.json` before handoff. The compiled resolver also accepts llm.kittens-style `TinyStories_train.bin` / `TinyStories_val.bin`; `--tinystories` resolves to `/mnt/disk2/dev/open-source/llm.kittens/dev/data/tinystories` when present, `NFN_LLM_KITTENS_TINYSTORIES_DIR` overrides that directory, and a direct `TinyStories_train.bin` path infers the sibling validation file. Raw-text datasets still lazy-load the dataset manager only when the uint16 cache must be materialized. Tokenizers whose ids do not fit in `uint16`, such as `o200k_base`, are rejected for this path. `NativeGptRunConfig.compiled_cli_argv()` wraps the Tile-CUDA training command through the no-Python cached-shard CLI. The generic module exports `NativeGptRunConfig`, `NativeGptRunnerStatus`, and `NativeGptCheckpointInfo` as GPT-native dataclass subclasses, plus `native_gpt_activation()`, `native_gpt_kernel_backend()`, `normalize_native_gpt_encoding_name()`, and `native_gpt_encoding_vocab_size()`, so new SDK code does not need GPT-2-named helper imports. Build the in-process compatibility SDK binding with `bash tools/build_native_gpt2_binding.sh`, the launcher with `bash tools/build_native_gpt2_launcher.sh`, the no-Python cached-shard CLI with `bash tools/build_native_gpt_cli.sh`, the linked no-Python cached-shard CLI with `bash tools/build_native_gpt_cli_linked.sh`, and the unified no-Python frontend with `bash tools/build_native_train_cli.sh`; the compiled GPT CLI links the shared C++ `token_shards.cpp` resolver for cached-shard validation. `bash tools/build_native_gpt2_all.sh` rebuilds `libnfn_native_train_tile_ops.so` before `nfn_gpt_native_train_linked`, so the linked binary preferred by SDK/CLI startup paths is built against the current Tile kernels. The linked GPT CLI resolves Tile ops ABI symbols from `RTLD_DEFAULT` when invoked with `--tile-ops-lib linked`, avoiding the in-trainer `dlopen` setup cost while keeping the same required-symbol scan. Direct linked no-data preflight smokes, including `--smoke-tile-ops` and `--smoke-optimizer-step`, use the same `RTLD_DEFAULT` symbol resolution instead of trying to `dlopen` a file named `linked`. SDK compiled-CLI resolution, `train_gpt.py`, `nfn train`, `nfn-native-train`, and the GPT-2-evo native delegate prefer `build/nfn_gpt_native_train_linked` when it exists and no explicit GPT CLI override is set; use the normal CLI or explicit `--tile-ops-lib PATH` when a benchmark intentionally swaps `libnfn_native_train_tile_ops.so` at runtime. `tools/install_native_gpt2_commands.sh` links `nfn-gpt-native`, `nfn-gpt-native-train`, `nfn-native-train`, `nfn-gpt2-tile-launcher`, and both underscore/hyphen names for built per-family native trainer entrypoints into the active Python scripts directory. `NFN_DATASETS_DIR` overrides the native alias cache root, and `nfn-native-train --base-model gpt --dataset-alias PATH_OR_ALIAS` or `nfn train --base-model gpt --dataset-alias PATH_OR_ALIAS` bypasses Python graph training when shards already exist. The compiled `nfn_native_train` frontend also accepts Python-wrapper aliases directly, including `--dataset tinystories`, `--output`, `--kernel-backend`, `--template` / `--preset`, `--graph`, and `--native-cuda-*`, then injects dense GPT defaults before execing `nfn_gpt_native_train`. `--base-model gpt2`, `--base-model gpt3`, and `--base-model nanogpt` are dense GPT aliases for the same native kernel family; GPT3 defaults to `--train-seq-len 2048` when selected by base model or `--template-name gpt3`, unless a custom graph or explicit sequence length is supplied. The implicit GPT3 batch size is 32 to preserve the 65,536-token microbatch unless `--batch-size` is explicit. NanoGPT defaults add `--template-name nanogpt`, and the shared dense loop uses the selected 320-wide/5-head/5-layer geometry for full-transformer training. Otherwise `--template-name` and `--graph-file` are the architecture source of truth, and compiled plan/runtime JSON canonicalizes dense GPT aliases to `model_family: "gpt"` while reporting `architecture_source`, `architecture_contract`, `model_family_context_policy`, and `resolved_native_template_name` for that decision. The default public template is `template_name="gpt"`, which currently resolves to the implemented dense GPT native topology. The compatibility `NativeGpt2RunConfig` and `build_native_gpt2_*` helpers now canonicalize dense GPT selectors to `model_family="gpt"`; pass `template_name` or `graph_file` for architecture selection instead of keying off the model-family label. The shared C++ token sampler reads contiguous shard segments for each batch instead of opening a shard per sequence chunk; token-shard JSON reports `batch_read_strategy: "contiguous_shard_segments"`. `nfn-native-train --list-models --json` reports the compiled native training registry; dense GPT defaults to the NeuralFn-owned Tile-CUDA `--train-transformer-lm` loop and its JSON `block_state_layout` exposes block-vector allocation/init/zero/clip/AdamW/checkpoint/tape/forward/backward loop flags. NanoGPT is reported as an implemented dense GPT selector; explicit `--train-token-lm` remains available on `nfn_nanogpt_native_train` for token-LM diagnostics. LLaMA, JEPA, semantic/MoE, and DeepSeek entries intentionally report missing native trainers, while GPT-2 evo reports its dense GPT layer-evo delegate. The token-only NanoGPT C++ target remains available for explicit diagnostics: `nfn_nanogpt_native_train --print-plan` emits a JSON training contract for shape, schedule, AdamW settings, token-shard constraints, contiguous parameter/gradient/AdamW-state buffers, AdamW decay/no-decay groups, forward/backward/optimizer `training_step_plan`, and required CUDA Tile kernels without importing Python or Torch. `nfn_nanogpt_native_train --train-token-lm --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS --max-steps N` runs the tied token-embedding LM as a real multi-step native loop over cached shards and records validation losses from validation shards according to `--eval-every-steps`, `--eval-batches`, and `--eval-batch-size`. `nfn_nanogpt_native_train --check-tile-ops --tile-ops-lib PATH` loads `libnfn_native_train_tile_ops.so` with `dlopen` and verifies every NanoGPT-required raw ABI symbol from the compiled binary. Tied LM head input/weight backward is represented through the raw linear backward ABI in that plan. It defaults to `dropout_p=0.0`; nonzero `--dropout-p` verifies the deterministic inverted-dropout Tile ABI through `nfn_native_tile_dropout_forward_float32` and `nfn_native_tile_dropout_backward_float32`. `NFN_NATIVE_GPT2_BIN_DIR` overrides where command symlinks are installed, `NFN_NATIVE_TRAIN_CLI` overrides the unified frontend path, `NFN_NATIVE_<MODEL>_CLI` overrides a per-family native trainer such as `NFN_NATIVE_NANOGPT_CLI`, `NFN_NATIVE_GPT_CLI` overrides the compiled GPT CLI path, and `NFN_NATIVE_GPT2_LAUNCHER` overrides the launcher path. `NativeGptRunConfig` and `NativeGpt2RunConfig` default `cuda_visible_devices="0"` and `cuda_device_max_connections="1"` before launching native code; existing `CUDA_VISIBLE_DEVICES` or `CUDA_DEVICE_MAX_CONNECTIONS` environment variables still take precedence. The top-level `neuralfn` package and native cached-shard path avoid importing Torch; `cli/scripts/train_gpt.py` is the canonical native-only GPT script, while `cli/scripts/train_gpt2.py` remains a compatibility entrypoint. Direct execution with the default `compiled-cli` runner reaches the compiled C++ CLI before importing `train_gpt_native.py`. Default GPT `nfn train` commands hand off to the compiled Tile-CUDA frontend before graph-backed Python imports. The CLI wrapper defaults to `--native-cuda-runner compiled-cli`; call `run_native_gpt(..., runner="auto")` or pass another runner explicitly when you want SDK binding or launcher behavior.

The compiled dense GPT trainer accepts native layer-evolution cadence flags:
`--layer-evo` / `--native-cuda-layer-evo`, `--evo-layer-index`,
`--evo-layer-interval`, `--evo-layer-population`, and
`--evo-layer-mutation-scale`. The current implementation allocates device-side
candidate workspace for the selected block's `block_N.ln1.weight`, launches
the raw `nfn_native_tile_evo_mutate_candidates_float32`,
`nfn_native_tile_evo_select_best_loss_float32`, and
`nfn_native_tile_evo_adopt_candidate_float32` ABI kernels during the optimizer
loop, and reports `graph_editor_tensor_flow: false` in `layer_evo` plan/runtime
JSON. Candidate losses are scored by running native CUDA forward loss over the
current batch for each mutated candidate; runtime JSON reports
`forward_candidate_evals` and
`candidate_loss_source: "native-forward-loss-device-resident-current-batch"`,
`candidate_loss_transport: "device-to-device"`, and
`candidate_loss_host_roundtrips_elided` because the scalar loss is copied
directly from the native forward-loss device buffer into the candidate-loss
device array.

Dense GPT plan/runtime JSON reports
`native_geometry_contract.selected_template_geometry` and
`geometry_matches_compiled_loop`. The compiled transformer-LM loop now uses the
selected dense GPT runtime geometry for GPT-2, GPT-3, NanoGPT, and compatible
custom graph metadata; selectors such as `--template-name nanogpt` therefore
expose and use their requested width/head/layer shape in the contract.
For SDK compiled-CLI dispatch, `NativeGptRunConfig.graph_file` can also point at
an existing graph JSON that carries native-compatible GPT `template_spec`
metadata under the graph torch config. Those files report
`selected_graph_support_status: "native-transformer-lm"`, set
`native_geometry_contract.shape_source: "custom_graph_template_spec"`, and can
provide default sequence length and layer count before the fixed dense GPT C++
Tile trainer runs. Arbitrary graph JSON, incompatible metadata, and missing
graph paths still fail explicitly with `custom-graph-native-trainer-missing` or
`custom-graph-file-missing` instead of routing batches through graph-editor
nodes or Torch.

Native compiled entrypoints and SDK bindings set `CUDA_MODULE_LOADING=LAZY`
when unset before executing native trainers or loading Tile CUDA libraries,
matching the dense GPT C++ trainer. Existing user-provided
`CUDA_MODULE_LOADING` values still take precedence.

`NativeGpt2RunConfig.lm_head_row_chunk_size` defaults to 32768 and forwards
`--lm-head-row-chunk-size` through `compiled_cli_argv()`; pass 8192, 6144, or
4096 explicitly to reproduce older smaller-workspace profiles. The C++ transformer-LM
loop uses that bounded full-vocab tied LM-head workspace and reduces CE loss
partials on device with `nfn_native_tile_sum_partials_float32`, so training and
validation loss copy one device scalar to the host instead of copying once per
row chunk. Tied LM-head dWeight chunks accumulate directly into the optimizer-step
`accum_grad_token_weight` buffer with
`nfn_native_tile_linear_backward_weight_accumulate_float32` instead of using a
full-vocab scratch gradient buffer per chunk or per microbatch. The JSON reports
`lm_head_row_chunk_count` and `loss_partial_count`.

Direct `python cli/scripts/train_gpt_native.py ...` compiled-cli executions use
the Python harness for argument resolution, dry runs, and command printing, then
replace that process with the compiled C++ trainer for non-dry-run actions. The
handoff applies the same default `CUDA_VISIBLE_DEVICES`,
`CUDA_DEVICE_MAX_CONNECTIONS`, and `CUDA_MODULE_LOADING=LAZY` policy as
`run_native_gpt()`, so legacy script launches do not keep a Python parent alive
during CUDA Tile training.

The dense GPT compiled trainer also defaults to the row-loss classifier
backward route when the trainer-facing Tile ops library exports
`nfn_native_tile_lm_head_classifier_backward_row_losses_inplace_strided_no_pad_zero_bf16_bits_u16_targets`.
That ABI writes one loss per row while converting each BF16 logit chunk in-place
to dlogits; the trainer then reduces the row losses on device with
`nfn_native_tile_sum_partials_float32` and accumulates the scalar loss with
`nfn_native_tile_gradient_accumulate_float32`. Runtime JSON exposes
`lm_head_ce_row_loss_reduction_requested`,
`lm_head_ce_row_loss_reduction_available`,
`lm_head_ce_row_loss_reduction_enabled`, and
`lm_head_ce_loss_backward_strategy`. Set
`NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_REDUCTION=0` only when comparing against the
older fused scalar-loss atomic route in a same-script paired benchmark.
`NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE=0` is the default row-loss tail;
it reduces row losses with `sum_partials` plus scalar `gradient_accumulate`.
Set it to `1` only when comparing against the older single
`nfn_native_tile_sum_accumulate_float32` launch per row chunk. Use
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_row_loss_partial_reduce` to route
the promoted path against the older sum-accumulate baseline, or
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_row_loss_sum_accumulate` to opt into
the older route.
`NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=1` is the default train-loss logging
path. It routes the BF16/u16 classifier row blocks to accumulate row losses
into a fixed bin workspace before one
`nfn_native_tile_sum_accumulate_float32` tail, and JSON reports
`lm_head_ce_loss_bin_reduction_*`, `lm_head_ce_loss_bin_count_requested`, and
`lm_head_classifier_loss_bin_launch_count`. The named wrapper profile is
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_loss_bins`; the wrapper forces its
baseline command to `NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=0` so it still
measures the new default against the older row-loss route. Set
`NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=0` manually only for regression
checks against the older row-loss tail.
`NFN_NATIVE_GPT_LM_HEAD_CLASSIFIER_CE_NO_LOSS=1` is a default-off candidate for
no-loss optimizer steps. It routes the BF16/u16 LM-head CE stage through the
classifier row-loss kernel and skips loss reduction. The paired wrapper profile
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_classifier_ce_no_loss` forces that
older generic no-loss CE path on the baseline side and the classifier route on
the candidate side. JSON reports
`lm_head_classifier_ce_no_loss_requested` and
`lm_head_classifier_ce_no_loss_enabled`. Default timing-only optimizer steps
still use the generic no-loss CE+dlogits path when train-loss logging is
disabled; runtime JSON distinguishes it with
`lm_head_classifier_no_loss_chunk_count`,
`lm_head_ce_kernel_strategy: "no-loss-dlogits-vec8-loads-scalar-stores"`, and
`lm_head_ce_loss_backward_strategy:
"no-loss-dlogits-public-vocab-no-pad-zero-bf16-u16-targets"`. The paired speed
tool reports the no-loss chunk counter as a route metric so no-loss benchmarks
are not mistaken for row-loss or loss-bin train-loss paths.

Native dense-GPT runs also expose cuBLASLt BGRADB route counters:
`linear_cublaslt_bgrad_gemm_count`,
`linear_cublaslt_bgrad_direct_write_count`, and
`linear_cublaslt_bgrad_accumulate_count`. Use these with
`tools/paired_kernel_speed.py` when testing block-backward dWeight+bias
candidates, because ordinary `linear_cublaslt_gemm_count` does not distinguish a
BGRADB epilogue from a plain cuBLASLt GEMM.
`NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_DEFAULT_SPECIALIZED=1` is the default
specialization for the same timing-only path; the paired profile
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_ce_no_loss_default_specialized`
forces the baseline to `NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_DEFAULT_SPECIALIZED=0`,
keeps train-loss logging disabled, and reports
`lm_head_ce_no_loss_default_specialized_*` plus
`lm_head_ce_kernel_strategy:
"no-loss-default-specialized-dlogits-vec8-loads-scalar-stores"` when active.
Set `NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_DEFAULT_SPECIALIZED=0` only to compare
against the older generic no-loss CE+dlogits kernel.

`NFN_NATIVE_GPT_CE_BF16_SCALAR_STREAMING_STORES=1`,
`NFN_NATIVE_GPT2_CE_BF16_SCALAR_STREAMING_STORES=1`, and
`NFN_TILE_CUDA_CE_BF16_SCALAR_STREAMING_STORES=1` expose a default-off BF16 CE
scalar dlogit store cache-hint probe. The SM120 wrapper profile
`lm_head_ce_scalar_streaming_store` keeps vec8 BF16 loads and changes runtime
JSON from `vec8-loads-scalar-stores` to
`vec8-loads-scalar-streaming-stores`, but the dedicated RTX 5090 same-script
gate rejected it at `1.005702x` train-loop wall and `1.135829x` LM-head CE
time, so normal training should leave it unset.

`NFN_NATIVE_GPT_LM_HEAD_CE_DEFAULT_SPECIALIZED=1`,
`NFN_NATIVE_GPT2_LM_HEAD_CE_DEFAULT_SPECIALIZED=1`, and
`NFN_TILE_CUDA_LM_HEAD_CE_DEFAULT_SPECIALIZED=1` expose a default-off
row-loss CE kernel specialization for the current dense GPT LM-head defaults:
1024 CE threads, vec8 BF16 loads, scalar cached stores, and `expf`.
Runtime JSON reports `lm_head_ce_default_specialized_requested`,
`lm_head_ce_default_specialized_enabled`, and `lm_head_ce_kernel_strategy`;
the SM120 wrapper profile is `lm_head_ce_default_specialized`. Keep it
diagnostic-only: the CUDA 13.3 dedicated RTX 5090 same-script gate proved the
route changed but rejected it at `1.001545x` train-loop wall, `1.000931x`
LM-head backward, and `1.000331x` LM-head CE time.

Grouped cuBLASLt execution is exposed as a diagnostic probe, not a training
route. Set `NFN_NATIVE_GPT_PROBE_CUBLASLT_GROUPED_MATMUL=1`,
`NFN_NATIVE_GPT2_PROBE_CUBLASLT_GROUPED_MATMUL=1`, or
`NFN_TILE_CUDA_LINEAR_CUBLASLT_GROUPED_MATMUL_PROBE=1` to run the tiny grouped
matmul smoke during native GPT startup. Runtime JSON reports
`linear_cublaslt_grouped_matmul_probe_available`,
`linear_cublaslt_grouped_matmul_probe_requested`,
`linear_cublaslt_grouped_matmul_probe_status`, and
`linear_cublaslt_grouped_matmul_supported`. A nonzero status is recorded without
failing startup; the current CUDA 13.3 RTX 5090 result is status `15`, so
grouped cuBLASLt matmul remains a blocked candidate for LM-head/block-backward
parity even though `linear_cublaslt_grouped_layout_supported` is true.
The paired SM120 wrapper profile
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=cublaslt_grouped_probe` enables the
cuBLASLt layout and grouped-matmul probes for repeatable CUDA-upgrade checks;
the current CUDA 13.3 WSL probe still reports grouped layout status `0` and
grouped matmul status `15`. The wrapper keeps native route-change proof enabled
but skips automatic timing-ratio gates for this capability-only profile, so the
readiness check reports probe support instead of setup timing noise. The profile
deliberately omits the classic cuBLAS
grouped BF16 probe because the CUDA 13.3 recheck showed it still poisons the
selected CUDA context before model allocation when unsupported.
CUDA 13.3.33 post-reinstall paired checks keep 32768 rows as the default.
Retesting `--lm-head-row-chunk-size 8192` against the current 32768-row route
regressed train-loop wall time to `1.001841x` despite slightly improving the
isolated LM-head stage. Use 8192 only as an explicit lower-memory reproduction
knob, not as the workstation default.
`NFN_NATIVE_GPT_LM_HEAD_REVERSE_CHUNKS` /
`NFN_NATIVE_GPT2_LM_HEAD_REVERSE_CHUNKS` controls LM-head row-chunk traversal
and reports `lm_head_reverse_chunk_order_enabled` plus
`lm_head_reverse_chunk_order_strategy`. Reverse traversal is the CUDA 13.3 RTX
5090 default after a guarded three-sample same-script gate measured `0.997183x`
train-loop wall time versus the previous forward order; set the variable to `0`
only for bisection.

The dense GPT cooperative LM-head backward ABI has two distinct callable
surfaces. `nfn_native_tile_lm_head_classifier_backward_cooperative_fused_bf16_u16`
is the existing event-ordered sequence wrapper and only satisfies
`lm_head_cooperative_backward_sequence_wrapper_available`.
`nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16` is the
future strict fused classifier/dHidden/dWeight callable and must be loaded
with a nonzero
`nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused()`
capability before `lm_head_cooperative_backward_fused_kernel_available` or
`lm_head_cooperative_backward_kernel_enabled` can become true. SDK and CLI
strict runs therefore still fail on wrapper-only or placeholder-symbol builds
instead of silently benchmarking the older CE plus dHidden plus dWeight
sequence.

`NativeGptRunConfig.train_loss_every_steps` and
`NativeGpt2RunConfig.train_loss_every_steps` default to `0` and forward
`--train-loss-every-steps` through `compiled_cli_argv()`. Set it to `1000` to
record sampled native training loss every 1000 optimizer steps while keeping
validation loss on `eval_every_steps`. The compiled transformer-LM loop collects
that train loss from the folded LM-head backward recompute path before logits
are overwritten as dLogits, so no real token batch passes through graph-editor
nodes and no duplicate forward LM-head loss pass is needed for train-loss
logging. On the default BF16 logits plus direct-u16 target path, loss sampling
uses the fused raw C ABI
`nfn_native_tile_token_cross_entropy_backward_loss_inplace_strided_bf16_bits_u16_targets`
so the CE scalar accumulation and in-place BF16 dlogit write share one Tile CUDA
kernel. The train-loss scalar is accumulated on device for the whole optimizer
step across gradient-accumulation microbatches and copied to the host once per
logged step. Runtime JSON reports `lm_head_ce_loss_backward_fused_available`,
`lm_head_ce_loss_backward_strategy`,
`train_loss_device_accumulation_strategy` with value
`"optimizer-step-device-scalar-accumulate"`, `train_loss_host_copy_scope` with
value `"once-per-logged-optimizer-step"`, `train_loss_host_d2h_count`,
`train_loss_host_d2h_copies_per_logged_step: 1`, and
`train_loss_microbatch_host_d2h_copies_elided_per_logged_step`.
For loss-bin CE bisection, set
`NFN_NATIVE_GPT_LM_HEAD_CE_LOSS_BINS_DEFAULT_SPECIALIZED=1`,
`NFN_NATIVE_GPT2_LM_HEAD_CE_LOSS_BINS_DEFAULT_SPECIALIZED=1`, or
`NFN_TILE_CUDA_LM_HEAD_CE_LOSS_BINS_DEFAULT_SPECIALIZED=1` with loss-bin
reduction enabled. The Tile CUDA launcher then uses a dedicated default-shape
loss-bin CE/dlogit kernel when the CE launch still uses 1024 row threads, vec8
BF16 loads, scalar cached stores, and `expf`. Runtime JSON reports
`lm_head_ce_loss_bins_default_specialized_requested`,
`lm_head_ce_loss_bins_default_specialized_enabled`, and
`lm_head_ce_kernel_strategy` with value
`"default-specialized-loss-bins-vec8-loads-scalar-stores"`. The SM120 paired
wrapper exposes this as
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_ce_loss_bins_default_specialized`.
Keep it diagnostic-only for now: the CUDA 13.3 dedicated RTX 5090 3-step,
3-sample gate proved the route and passed train-loop wall (`0.999215x`) but
rejected the candidate on LM-head backward (`1.000741x`), LM-head CE
(`1.000339x`), and MLP projection (`1.001222x`).

For SDK-launched native GPT profiling, include
`NFN_NATIVE_LINEAR_SHAPE_STATS=1`, `NFN_TILE_CUDA_LINEAR_SHAPE_STATS=1`,
`NFN_NATIVE_GPT_LINEAR_SHAPE_STATS=1`, or `NFN_NATIVE_GPT2_LINEAR_SHAPE_STATS=1` in the
subprocess environment. The compiled trainer then reports `linear_shape_stats`
JSON buckets that identify the successful TK BF16, cuBLASLt, cuBLAS GEMMEx BF16,
and SGEMM linear dispatch shapes, call counts, `total_us`, and `avg_us`. With
the v2 Tile stats ABI, cuBLASLt rows also include
`cublaslt_selected_heuristic`, `cublaslt_returned_heuristics`, and
`cublaslt_workspace_bytes`, so candidate runs can distinguish real heuristic
changes from overrides that request an unavailable index. The
timing path uses CUDA events and synchronizes measured GEMMs, so it is intended
for kernel candidate comparisons and should stay disabled in normal training
runs.
Runtime timing separates `setup_wall_ms`, `train_loop_wall_ms`,
`post_train_sample_wall_ms`, `cleanup_wall_ms`, and `total_wall_ms`, so SDK
and CLI benchmarks can distinguish startup from in-loop training.
`--startup-only` still synchronizes the setup path, but elides the post-train
diagnostic device-to-host sample copies and reports
`post_train_diagnostic_samples_elided: true`. For startup-specific native
candidate bisections, set
`NFN_SM120_NATIVE_STARTUP_ONLY=1` when using
`tools/bench_native_gpt_sm120_candidate.sh`; the helper appends
`--startup-only` to both baseline and candidate while preserving the same
selected-GPU idle/load checks and command shape.
The native-vs-native SM120 candidate wrapper defaults those selected-GPU idle
checks to three utilization samples spaced 0.25 seconds apart, which filters
brief WSL/NVML spikes on the dedicated display-disabled RTX 5090 before a
candidate is accepted or rejected. Override with
`NFN_SM120_NATIVE_SELECTED_GPU_UTILIZATION_RETRIES` /
`NFN_SM120_SELECTED_GPU_UTILIZATION_RETRIES` and the matching
`..._RETRY_INTERVAL_SECONDS` aliases when a different polling policy is needed.
The wrapper also defaults
`NFN_SM120_NATIVE_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE=1`, which keeps a
stuck high `nvidia-smi` utilization counter from blocking a dedicated-GPU run
when the selected GPU has no compute processes. Set it to `0` for strict
utilization gating; active compute processes still fail immediately.
For candidate-only native CLI flags, use
`NFN_SM120_NATIVE_CANDIDATE_EXTRA_ARGS`, the natural
`NFN_SM120_NATIVE_CANDIDATE_ARGS` alias, or the short
`NFN_SM120_CANDIDATE_EXTRA_ARGS`; use `NFN_SM120_NATIVE_EXTRA_ARGS` or
`NFN_SM120_COMMON_EXTRA_ARGS` only for flags that intentionally apply to both
baseline and candidate. Run with `NFN_SM120_NATIVE_DRY_RUN_PLAN=1` or
`NFN_SM120_CANDIDATE_DRY_RUN_PLAN=1` first when testing shape flags so the
printed commands prove the candidate flag did not leak into the baseline.
Both SM120 benchmark wrappers accept generic `NFN_SM120_*` fallbacks for shared
controls such as steps, samples, warmup, GPU selection, profile directory, and
JSON output. Wrapper-specific aliases still take precedence, so
`NFN_SM120_NATIVE_STEPS` or `NFN_SM120_PARITY_STEPS` override
`NFN_SM120_STEPS` when both are present. The native candidate wrapper also
accepts explicit `NFN_SM120_NATIVE_CANDIDATE_*` common-shape aliases between
the canonical native names and the short `NFN_SM120_CANDIDATE_*` names; for
example, `NFN_SM120_NATIVE_CANDIDATE_STEPS=5`,
`NFN_SM120_NATIVE_CANDIDATE_SAMPLES=1`,
`NFN_SM120_NATIVE_CANDIDATE_WARMUP=0`, and
`NFN_SM120_NATIVE_CANDIDATE_JSON_OUT=/tmp/run.json` now affect the paired
workload instead of falling back to the default 10-step/3-sample run.
For one-shape TK forward bisection from the SDK, pass
`NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=m,n,k,opA,opB` or
`NFN_TILE_CUDA_LINEAR_TK_FORWARD_DISABLE_SHAPE=m,n,k,opA,opB` in the same
environment. The tuple matches the `linear_shape_stats` convention and only
gates forward/fused-GELU TK calls with fallback paths. CUDA 13.3 builds now try
the TK BF16 forward bridge by default for no-bias BF16-input/BF16-weight/
BF16-output GEMMs, including the padded LM-head logits shape
`50304,32768,768,T,N`, before falling back to cuBLAS GEMMEx. To reproduce the
older GEMMEx route for paired comparisons, set
`NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=50304,32768,768,T,N` or
`NFN_TILE_CUDA_LINEAR_TK_FORWARD_DISABLE_SHAPE=50304,32768,768,T,N`.
Native GPT runtime JSON also reports `lm_head_logits_tk_gemm_count`,
`lm_head_logits_cublaslt_gemm_count`, and
`lm_head_logits_bf16_gemm_count`, so `lm_head_logits_linear_strategy` identifies
the active LM-head logits backend without enabling the heavier
`linear_shape_stats` timing path.
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=qkv_forward_bf16_fallback_65536` and
`mlp_fc_forward_bf16_fallback_65536` reproduce the current forward-shape
fallback probes. Both are rejected on CUDA 13.3 RTX 5090: QKV fallback reduced
TK forward calls but regressed `stage.block_forward.attention.qkv.total_ms` to
`1.143374x`, while MLP FC fallback did not change tracked route counters and
regressed train-loop wall to `1.016916x`.
The SDK default `NativeGpt2RunConfig.lm_head_row_chunk_size` is 32768 rows for
the local RTX 5090/CUDA 13.3 workstation profile. This reduces default LM-head
chunk launches at the 64x1024 shape from 8 chunks to 2 and reserves about
3.30GB of resident BF16 logit workspace. Set `lm_head_row_chunk_size=8192` or
pass `--lm-head-row-chunk-size 8192` to reproduce the older lower-memory
default. Effective LM-head chunks above 32768 rows are rejected before CUDA
launch unless `NFN_NATIVE_GPT_ALLOW_UNSAFE_LM_HEAD_ROW_CHUNK=1` is set for
explicit diagnostics; runtime JSON reports `lm_head_row_chunk_safe_cap` and
`lm_head_row_chunk_unsafe_override_enabled`.

Prefer the generic dense GPT environment names for new SDK integrations:
`NFN_NATIVE_GPT_CLI`, `NFN_NATIVE_GPT_RUNNER`, and `NFN_NATIVE_GPT_BINDING`. The `llm-kittens` GPT training backend has been removed; keep `tools/bench_native_gpt_sm120_parity.sh` for reference timing. Runtime tuning prefers
`NFN_NATIVE_GPT_STAGE_TIMING`, `NFN_NATIVE_GPT_STAGE_TIMING_MAX_EVENTS`,
`NFN_NATIVE_GPT_PACKED_QKV_ATTENTION`, `NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS`,
`NFN_NATIVE_GPT_STORE_MLP_BLOCKS`,
`NFN_NATIVE_GPT_STORE_ATTENTION_ACTIVATIONS`,
`NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_ACTIVATIONS`,
`NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS`,
`NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE`,
`NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_STATS`,
`NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_BF16`,
`NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS`,
`NFN_NATIVE_GPT_CUDA_MEMSET_ZERO`,
`NFN_NATIVE_GPT_CUDA_MEMSET_GRAD_ZERO`,
`NFN_NATIVE_GPT_FUSE_RESIDUAL1_STORE`,
`NFN_NATIVE_GPT_FUSE_ATTENTION_RESIDUAL_LN2`,
`NFN_NATIVE_GPT_FUSE_MLP_RESIDUAL_NEXT_LN1`,
`NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_WARPS`,
`NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_HD64_SPECIALIZED`,
`NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU`,
`NFN_NATIVE_GPT_LN1_BF16_QKV_FORWARD`,
`NFN_NATIVE_GPT_BF16_QKV_GRAD_HANDOFF`,
`NFN_NATIVE_GPT_BF16_PROJECTION_RESIDUAL`,
`NFN_NATIVE_GPT_LM_HEAD_BF16_LOGITS`,
`NFN_NATIVE_GPT_BF16_LM_HEAD_LOSS`,
`NFN_NATIVE_GPT_PUBLIC_VOCAB_CE`,
`NFN_NATIVE_GPT_CE_BF16_EXP2`,
`NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS`, and
`NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN`,
`NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_SHADOW`,
`NFN_NATIVE_GPT_BF16_BIAS_INPLACE_TILE`, and
`NFN_NATIVE_GPT_F32_TO_BF16_VEC4`,
`NFN_NATIVE_GPT_F32_TO_BF16_MANY_VEC4`, and
`NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS_VEC4`, and
`NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC`. The older `NFN_NATIVE_GPT2_*`
variables remain compatibility fallbacks for existing GPT-2-named wrappers.
The tokenizer-visible GPT-2 vocab remains 50,257, but native transformer-LM
parameter layout pads the tied token embedding/LM-head rows to 50,304 for the
GEMM path. Compiled plan JSON reports `shape.vocab_size: 50257` and
`shape.padded_vocab_size: 50304`; training and checkpoint JSON report
`vocab: 50257` plus `padded_vocab: 50304`, and `logit_workspace_elements` is
computed from the padded row count.

`NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS=1` allocates full BF16 LM-head
logits and reuses chunk offsets during backward instead of recomputing the
classifier logits. This is a diagnostic parity path for comparing against the
llm.kittens full-logit classifier layout. It remains off by default: on CUDA
13.3 with the dedicated RTX 5090, the mode needed a lower saved packed-attention
cap to fit and measured slower than the default chunked-logit path
(`1.099054x` train-loop wall with zero saved packed-attention blocks and
`1.061321x` with four). Runtime JSON reports
`lm_head_reuse_forward_logits_enabled`, `lm_head_full_logit_elements`, and
`lm_head_bf16_logit_bytes`.
Use `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_full_resident_reuse` to reproduce
the full-resident/full-batch submode in paired benchmarks. It remains rejected:
the CUDA 13.3 dedicated RTX 5090 one-step run improved LM-head backward to
`0.705502x`, but regressed train-loop wall time to `21.830567x` and block
backward to `44.496727x` because the resident-logit footprint is too large for
the current saved-activation layout.
Native LM-head CE now softmaxes over the
public vocab and uses 50,304 only as the logits/dlogits row stride; runtime JSON
reports `lm_head_public_vocab_ce_enabled`, `lm_head_softmax_vocab`,
`lm_head_logit_row_stride`, and `lm_head_padded_dlogits_zeroed`.
Set `NFN_NATIVE_GPT_PUBLIC_VOCAB_CE=0` only to reproduce the older padded-vocab
CE behavior for paired benchmarks.
The no-data LM smoke, dataset-backed embedding-LM smoke/training path, and
transformer-LM smoke path use the same padded tensor shape while keeping shard
token validation on the public vocab.

Dense GPT projection residuals keep projection outputs in BF16 by default.
Attention and MLP projection GEMMs write BF16 projection-output scratch and the
fused residual consumers read those BF16 bits directly through
`nfn_native_tile_linear_bias_residual_add_bf16_linear_float32` and the
BF16-linear residual+LN2 ABI variants. The raw Tile ABI also exports
`nfn_native_tile_linear_bias_residual_add_bf16_linear_bf16_residual_float32`
for the opt-in BF16 persistent block-output diagnostic; it writes the FP32 MLP
residual-add output and the BF16 persistent side buffer in the same kernel
launch. Runtime and plan JSON report
`bf16_projection_residual_enabled`, `projection_bf16_scratch_elements`,
`projection_bf16_scratch_bytes`, `attention_projection_input_strategy:
"packed-o-bf16-direct-gemm-bf16-residual-consumer"`,
`mlp_proj_forward_activation_strategy:
"fused-gelu-bf16-act-direct-bf16-output-gemm"`, and
`projection_bias_residual_strategy: "fused-bf16-linear-bias-residual-add"`.
Set `NFN_NATIVE_GPT_BF16_PROJECTION_RESIDUAL=0` only for paired benchmarks
against the previous float projection-output residual path.
For the GPT-native 768-wide residual-add shape, the BF16 projection residual
helper now uses a dim-specialized CUDA kernel by default to avoid the generic
per-element output-column modulo. Set
`NFN_TILE_CUDA_DIM768_BF16_RESIDUAL_ADD=0`,
`NFN_NATIVE_GPT_DIM768_BF16_RESIDUAL_ADD=0`, or
`NFN_NATIVE_GPT2_DIM768_BF16_RESIDUAL_ADD=0` to reproduce the generic helper for
paired bisection; the dedicated RTX 5090 same-script benchmark measured the
specialized default at `0.998835x` mean train-loop wall time and `0.997518x`
median total wall time versus the generic path.

GPT-2-compatible causal SDPA now uses the packed-QKV SM120 ThunderKittens bf16
route by default in `tools/build_native_train_tile_ops.sh`. The route accepts
NeuralFn's row-major packed QKV BF16 ABI, runs the llm.kittens SM120 attention
forward/backward tiles, keeps packed BF16 QKV gradients for the QKV backward
GEMMs, and stores packed QKV/O/LSE tensors for direct backward reuse at the
workstation shape. Training JSON reports `packed_qkv_attention_enabled: true`,
`qkv_forward_layout_strategy: "packed-qkv-bf16-no-split"`,
`attention_backward_strategy:
"tk-sm120-packed-qkv-bf16-saved-activation-backward-direct-bf16-grad-scratch-handoff"`,
`attention_forward_tk_launch_count`, and `attention_backward_tk_launch_count`.
Set `NFN_NATIVE_GPT_PACKED_QKV_ATTENTION=0` to force the split-QKV fallback for
paired bisection or lower-memory runs.
Set `NFN_NATIVE_GPT_ATTENTION_BACKWARD_SECTION_TIMING=1` only for short
diagnostic runs that need packed-backward section timing: it uses CUDA events
and synchronizes the stream to report dprep and TK backward totals/counts as
`attention_backward_dprep_timing_us`,
`attention_backward_dprep_timing_count`, `attention_backward_tk_timing_us`, and
`attention_backward_tk_timing_count`.
The trainer-facing build mirrors llm.kittens' SM120 NVCC threading,
host-compiler, data-prep, memory, and LayerNorm tuning flags for those
ThunderKittens headers while keeping GEMM on NeuralFn's initialized cublasLt
path.

Trainer-facing linear GEMMs use the same native ABI, and the full dense GPT
trainer now defaults QKV, attention projection, MLP FC, and MLP projection
weights to the BF16-primary block-weight path while retaining FP32 gradients and
AdamW state. The old FP32-master/BF16-shadow path remains available with
`NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS=0`. Transformer block
forward/recompute projections and block dInput GEMMs consume those BF16 weights
through `nfn_native_tile_linear_weight_bf16_float32`,
`nfn_native_tile_linear_weight_bf16_output_float32`,
`nfn_native_tile_linear_bf16_input_weight_bf16_float32`, and
`nfn_native_tile_linear_backward_input_weight_bf16_float32`. Transformer block
dWeight+bias accumulation calls
`nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_float32` or
`nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32`,
which request cuBLASLt `CUBLASLT_EPILOGUE_BGRADB` for supported BF16 block
shapes and fall back inside the ABI to separate dWeight plus Tile bias
reduction when unsupported. Tied LM-head
logits, dHidden, and dWeight chunks also default to the BF16 classifier path.
`nfn_native_tile_linear_bf16_output_float32` writes BF16 logits,
`nfn_native_tile_linear_bf16_input_float_weight_bf16_output_float32` reuses a
full-microbatch BF16 final-norm hidden prepack with FP32 tied token weights,
and the default tied token embedding/LM-head BF16 shadow routes logits and
dHidden through `nfn_native_tile_linear_weight_bf16_output_float32`,
`nfn_native_tile_linear_bf16_input_weight_bf16_output_float32`, and
`nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_float32` while the
FP32 master remains authoritative for token embedding, AdamW state, and
checkpoint export. Set `NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_SHADOW=0` only for
paired benchmarks against the older per-step BF16 bridge/cache route. Runtime
JSON reports `token_weight_bf16_shadow_enabled` and
`token_weight_bf16_refresh_count`.
`nfn_native_tile_token_cross_entropy_partials_strided_bf16_bits` computes
validation/test CE partials over the public vocab while walking the padded
logit row stride,
`nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_with_workspace`
overwrites public-vocab logits with BF16 dlogits and zeroes padded dlogit
columns. When sampled train-loss recording is active on the direct-u16 target
path,
`nfn_native_tile_token_cross_entropy_backward_loss_inplace_strided_bf16_bits_u16_targets`
also accumulates the CE scalar while writing those dlogits, replacing the older
separate training-loss partials pass. That fused loss+backward kernel
synchronizes the block after the target-logit loss read and before the in-place
dlogit stores, so sampled train-loss and validation/test loss reads cannot race
against the BF16 logit overwrite. The BF16 dlogits feed
`nfn_native_tile_linear_backward_input_bf16_bits_float32` plus the prepacked
BF16-hidden/BF16-dlogit
`nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32`
path when full hidden prepack is explicitly enabled. The default
`NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=0` keeps LM-head hidden packing
per chunk; set it to `1` only to benchmark the older full-microbatch
final-hidden prepack route. Set
`NFN_NATIVE_GPT_CE_BF16_EXP2=1`, `NFN_NATIVE_GPT2_CE_BF16_EXP2=1`, or
`NFN_TILE_CUDA_CE_BF16_EXP2=1` only for paired profiling of the BF16 CE+dlogits
kernel's `exp2f(x * log2(e))` path; the default remains `expf`, and runtime
JSON reports `lm_head_ce_bf16_exp2_enabled`.
For SDK launches through `NativeGpt2RunConfig` or the generic
`NativeGptRunConfig`, `batch_size_explicit`, `seq_len_explicit`, and
`num_layers_explicit` control whether the compiled CLI receives those shape
flags. Leave them at the default `True` when constructing a fully specified
run. Set them to `False` when selecting a custom `graph_file` and you want the
compiled C++ runner to read compatible dense GPT `template_spec` metadata
directly from the graph. The high-level native GPT harness sets these booleans
from the user's actual argv, so omitted shape flags no longer mask graph
metadata. Unsupported geometry still reports
`custom-graph-native-trainer-missing` or
`template-native-trainer-missing`; arbitrary non-dense graph topologies and
non-GPT vocabularies are not native until matching compiled trainers exist.
The mixed float32-hidden/BF16-grad dWeight+bias ABI now uses the cuBLASLt bgrad
epilogue route by default for supported QKV profiling shapes; set
`NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD=0` or
`NFN_TILE_CUDA_LINEAR_FLOAT32_BF16_BGRAD=0` to compare it against the previous
split-bias route.
The raw ABI also exports
`nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32`
for LM-head dWeight accumulation that packs the final LayerNorm hidden state to
BF16 before accumulating against BF16 dlogits. Dense GPT enables this by default
after paired dedicated-RTX-5090 timing measured it slightly faster than the
previous float-hidden dWeight path; set
`NFN_NATIVE_GPT_LM_HEAD_BF16_DWEIGHT=0` to reproduce the older path. Runtime
JSON reports `lm_head_bf16_dweight_enabled`, `lm_head_dweight_input_dtype`,
`lm_head_bf16_hidden_elements`, and `lm_head_dweight_strategy`. Set
`NFN_NATIVE_GPT_LM_HEAD_BF16_LOGITS=0` to return the tied LM-head chunks to the
older optimized TF32 tensor-op `cublasSgemm` path for debugging. Set
`NFN_NATIVE_GPT_BF16_LM_HEAD_LOSS=0` to keep BF16 training backward while
comparing validation/test loss against the older float logits loss workspace.
Dense GPT dWeight GEMMs use first-write-then-accumulate semantics by default:
the first gradient-accumulation microbatch launches the beta-capable raw ABI
with GEMM `beta=0`, and subsequent microbatches use `beta=1`. That path covers
the tied LM-head BF16/BF16 dWeight route plus QKV, attention projection, MLP FC,
and MLP projection block dWeight+bias calls that use the trainer cuBLASLt
routes. The tied LM-head dWeight path is split into row chunks for memory; only
the first chunk of the first gradient-accumulation microbatch uses `beta=0`, and
the remaining chunks use `beta=1` so chunked token contributions accumulate
instead of replacing each other. Set
`NFN_NATIVE_GPT_DWEIGHT_FIRST_MICROBATCH_BETA_ZERO=0` (or the compatibility
`NFN_NATIVE_GPT2_DWEIGHT_FIRST_MICROBATCH_BETA_ZERO=0`) only for paired
comparisons against the previous always-accumulate path. Runtime JSON reports
`dweight_first_microbatch_beta_zero_enabled`,
`dweight_first_microbatch_beta_strategy`, `lm_head_dweight_beta_zero_scope`, and
`first-write-then-accumulate` suffixes in `lm_head_dweight_strategy`,
`block_backward_qkv_dweight_strategy`, and `block_backward_weight_linear_strategy`.
Set `NFN_NATIVE_GPT_LM_HEAD_DWEIGHT_BEFORE_DHIDDEN=1` only for paired LM-head
row-chunk order bisection. It runs LM-head dWeight before dHidden after CE
writes dlogits. A CUDA 13.3 dedicated-RTX-5090 same-script wrapper run briefly
measured `0.996095x` train-loop wall time over two samples, but the required
3-sample confirmation regressed train-loop wall time to `1.002871x` and train
tokens/sec to `0.997262x`, so the default remains CE -> dHidden -> dWeight.
Runtime JSON reports `lm_head_dweight_before_dhidden_enabled`; the paired
wrapper profile `lm_head_dweight_before_dhidden` expands to the same env switch
for reproducible same-script gates.
Set `NFN_NATIVE_GPT_LM_HEAD_PIPELINE_CHUNKS=1` only for same-script LM-head
schedule profiling. The opt-in candidate keeps the current bounded row-chunked
classifier memory model but allocates two BF16 logit chunks instead of one,
computes logits and public-vocab CE/dlogits on the default stream, and queues
dHidden plus serialized dWeight accumulation on nonblocking side streams before
the chunk buffer is reused. The route records per-slot side-stream completion
events and waits only on the BF16 logit slot being reused instead of
synchronizing whole side streams. Runtime JSON reports
`lm_head_pipeline_chunks_requested`, `lm_head_pipeline_chunks_enabled`,
`lm_head_pipeline_logit_buffer_count`,
`lm_head_pipeline_extra_bf16_logit_bytes`,
`lm_head_pipeline_slot_event_wait_count`,
`lm_head_pipeline_done_event_record_count`, and the schedule strategy
`double-buffered-logits-ce-default-stream-side-stream-dhidden-ordered-dweight-slot-events`.
Use `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_pipeline_chunks` to run this
candidate through the native SM120 wrapper's standard idle guard and default
train-loop, total LM-head, block-backward, and MLP-projection gates. The
pipeline queue and final-wait substages are extracted for candidate-side
inspection, not ratio-gated by default, because the serial baseline does not
emit those stage names. The slot-event route is still rejected as a default:
the one-step same-script RTX 5090 retest proved the event counters moved but
measured `18.448472x` native train-loop wall time versus the serial baseline.
Use `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_concurrent_dhidden_dweight` for
the simpler side-stream schedule that launches LM-head dHidden and dWeight from
separate nonblocking streams after CE writes dlogits. Stage-timed runs report
the combined `stage.lm_head_backward.dhidden_dweight_concurrent.total_ms`
bucket for candidate-side inspection; the wrapper gates train-loop and total
LM-head timing because the serial baseline emits split dHidden and dWeight
substages. The profile is rejected by default: the CUDA 13.3 dedicated RTX 5090
3-sample same-script confirmation activated the route but regressed train-loop
wall time to `1.002970x` and train tokens/sec to `0.997039x`.
Keep it disabled for normal training until the paired RTX 5090 gate proves it
beats the default serial chunk schedule.
Use `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_overlap_last_dweight` for the
narrow last-chunk overlap schedule. The route sets
`NFN_NATIVE_GPT_LM_HEAD_OVERLAP_LAST_DWEIGHT=1`, queues only the last processed
LM-head dWeight chunk on the side stream after CE, leaves dHidden on the default
stream, and waits after embedding backward before the next microbatch or
optimizer can touch the accumulated token-weight gradient. Runtime JSON reports
`lm_head_overlap_last_dweight_requested`,
`lm_head_overlap_last_dweight_available`,
`lm_head_overlap_last_dweight_enabled`,
`lm_head_overlap_last_dweight_queue_count`,
`lm_head_overlap_last_dweight_sync_count`, `lm_head_side_stream_count`,
`lm_head_dhidden_stream_enabled`, `lm_head_dweight_stream_enabled`, and the
schedule strategy
`last-processed-row-chunk-dweight-side-stream-overlaps-final-norm-block-backward`.
The paired wrapper now rejects it unless
`NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1` is set. Keep it
default-off: the CUDA 13.3 dedicated RTX 5090 5-step, 3-sample confirmation
proved the route active but regressed train-loop wall time to `1.001676x` and
train tokens/sec to `0.998350x`; the preceding stage-timed probe also missed
the total LM-head backward gate at `1.000164x`.
For cuBLASLt BGRADB dWeight+bias routes, the default writes the epilogue bias
gradient into Tile-owned scratch
and accumulates it into `grad_bias`. Runtime JSON reports
`linear_bias_gradient_first_write_bgrad_direct_enabled` when the direct shortcut
is active. Set `NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1`,
`NFN_NATIVE_GPT2_BGRAD_FIRST_WRITE_DIRECT=1`, or
`NFN_TILE_CUDA_LINEAR_BGRAD_FIRST_WRITE_DIRECT=1` only for paired comparisons
against direct first-write bias gradients. The SM120 wrapper profile
`bgrad_first_write_direct` expands to that flag and is rejected by default:
the CUDA 13.3 dedicated RTX 5090 same-script gate proved the route counter
change (`linear_cublaslt_bgrad_direct_write_count: 96`) but regressed
train-loop wall time to `1.003634x` and tokens/sec to `0.996521x`.
Set
`NFN_TILE_CUDA_LINEAR_BF16=1` or
`NFN_NATIVE_LINEAR_BF16=1` only when profiling the normal linear ABI's BF16
bridge. Set `NFN_TILE_CUDA_LINEAR_CUBLASLT=1` or
`NFN_NATIVE_LINEAR_CUBLASLT=1` only when profiling the normal linear ABI's
cached cuBLASLt TF32 path; the current 5090 GPT-2 shape keeps SGEMM as the
faster default. Shape-supported transformer-block BF16 GEMMs use cached
cuBLASLt with `CUBLAS_COMPUTE_32F_FAST_16BF` by default and select cuBLASLt
heuristic index 1 when that candidate is available on the workstation RTX 5090
shape when cuBLASLt actually returns multiple candidates. Set `NFN_TILE_CUDA_CUBLASLT_HEURISTIC_INDEX=N` or
`NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX=N` only for paired kernel profiling.
Set `NFN_TILE_CUDA_LINEAR_BF16_OUTPUT_CUBLASLT=1` or
`NFN_NATIVE_LINEAR_BF16_OUTPUT_CUBLASLT=1` only for BF16-output LM-head logits
profiling. The opt-in route moves eligible BF16-output logits GEMMs to
cuBLASLt in shape stats, but the dedicated RTX 5090 paired check measured it
neutral/slightly slower than the GEMMEx fallback, so it is not a default.
Set `NFN_TILE_CUDA_LINEAR_BF16_BF16_BGRAD=0` or
`NFN_NATIVE_GPT_FUSE_BF16_BF16_DWEIGHT_BGRAD=0` only for BF16-input/BF16-grad
block dWeight plus bias profiling. The split route keeps dWeight on the GEMM
path and runs bias reduction separately, but the dedicated RTX 5090 paired
check measured it slower than the fused BGRADB default.
If the trainer Tile ops library is built without the cuBLAS linear fast path or
the GEMM route is otherwise unavailable, large-row float32-output dWeight
fallbacks use a shared-memory 2D tiled CUDA kernel for float32/BF16 activation
and gradient combinations instead of the older row-chunked atomic dWeight
reduction. Bias-only fallback reductions still use the shared row-chunk path.
Set `NFN_TILE_CUDA_CUBLASLT_HEURISTIC_POLICY=min_waves` or
`NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_POLICY=min_waves` to compare against the
llm.kittens-style lowest-waves selector, or `max_waves` for the highest-waves
selector; explicit index overrides still win.
Set `NFN_TILE_CUDA_CUBLASLT_HEURISTIC_SHAPE=m,n,k,opA,opB,index` or
`NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=m,n,k,opA,opB,index` for a
single-shape bisection; it only changes the matching cuBLASLt plan and leaves
the default/global heuristic route in place for every other GEMM.
Shape-stat JSON includes `cublaslt_selected_heuristic`,
`cublaslt_returned_heuristics`, and `cublaslt_workspace_bytes` for cuBLASLt
rows when the v2 native Tile stats ABI is available. The CUDA 13.3 RTX 5090
path reports only one returned heuristic for the hot dense GPT MLP projection
dWeight shape `3072,768,65536,N,T`, so the dispatcher does not hardcode a
shape-specific default pin there; pass a shape override only for explicit paired
candidate bisection. The matching QKV dWeight+bias hot shape
`768,2304,65536,N,T` currently returns multiple cuBLASLt candidates, but the
`cublaslt_qkv_dweight_h0_65536` wrapper profile is rejected: pinning it from
heuristic `1` to `0` changed the intended plan and measured
`stage.block_backward.qkv.dweight_bias.total_ms=1.003363x` plus
`stage.block_backward.total_ms=1.000055x` on the CUDA 13.3 RTX 5090 stage gate.
Set `NFN_TILE_CUDA_LINEAR_CUBLASLT_WORKSPACE_MB=N` or
`NFN_NATIVE_LINEAR_CUBLASLT_WORKSPACE_MB=N` only for paired diagnostics that
change the cuBLASLt heuristic workspace cap. The default remains 128 MiB because
the normal 5-step dedicated RTX 5090 run rejected a 256 MiB cap as train-loop
neutral/slightly slower.
Trainer-facing BF16/BF16 backward GEMMs also allow larger dense GPT LM-head
chunk shapes through cuBLASLt by default; set
`NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_LARGE_SHAPES=0` or
`NFN_NATIVE_LINEAR_BF16_CUBLASLT_LARGE_SHAPES=0` to restore the previous
small-shape-only gate and fall back to BF16 `cublasGemmEx` for those chunks.
Set `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=m,n,k,opA,opB` or
`NFN_NATIVE_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=m,n,k,opA,opB` only for paired
shape bisection; for example `768,65536,3072,N,N` routes that one BF16
cuBLASLt shape bucket back through the older BF16 `cublasGemmEx` fallback while
leaving other shapes on the default cuBLASLt path.
Set `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE=m,n,k,opA,opB` or
`NFN_NATIVE_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE=m,n,k,opA,opB` for the opposite
single-shape bisection: one otherwise-gated BF16 shape is forced through
cuBLASLt while every other gated shape keeps the default fallback. This is
diagnostic-only. On CUDA 13.3, forcing the dense GPT LM-head dHidden bucket
`768,8192,50304,N,N` through cuBLASLt moved that bucket in shape stats but
measured slower than the default BF16 `cublasGemmEx` route in the paired RTX
5090 benchmark.
Set `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1` or
`NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1` only for paired diagnostics
that try LM-head-sized BF16 shapes with `k > 32768` through cuBLASLt. The
dedicated RTX 5090 check routed LM-head dHidden to cuBLASLt but measured it
slower than the default GEMMEx fallback, so this remains default-off. The
CUDA 13.3.33 WSL recheck after the toolkit reinstall still rejected it:
the route moved LM-head dHidden calls to cuBLASLt but measured `1.034147x`
train-loop wall time and `1.502430x` targeted dHidden time versus the default.
Set
`NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT=0` or
`NFN_NATIVE_LINEAR_BF16_CUBLASLT=0` to force the older BF16 `cublasGemmEx`
bridge. Tied LM-head BF16 logits use the SM120 ThunderKittens GEMM
bridge by default when the Tile ops library was built with TK support; set
`NFN_TILE_CUDA_LINEAR_TK_GEMM=0` or `NFN_NATIVE_LINEAR_TK_GEMM=0` to force the
BF16 `cublasGemmEx` fallback for diagnostics. The raw ABI also exposes
`nfn_native_tile_layer_norm_with_stats_bf16_out_float32` and
`nfn_native_tile_linear_bf16_input_weight_bf16_output_float32` for the default
LN1-BF16 packed-QKV forward path. The BF16 bridge keeps a
128-entry cache for stable packed operands such as weights and biases, but
BF16-output GEMMs repack mutable activation inputs because native scratch
activation pointers are reused with new contents. The cache is invalidated
after AdamW updates. GPT-2 training JSON reports `linear_backend_strategy:
"block-bf16-cublaslt-shape-gated-lm-head-tk-sm120-default"` when the default
block cuBLASLt plus TK LM-head path runs,
`block_forward_linear_strategy`, `block_backward_input_linear_strategy`,
`block_weight_bf16_shadow_strategy`, `block_weight_bf16_shadow_elements`,
`block_weight_bf16_shadow_bytes`, `block_weight_bf16_shadow_descriptor_count`,
`block_weight_bf16_shadow_fused_adamw_refresh_enabled`,
`block_weight_bf16_refresh_count`,
`block_weight_bf16_fused_adamw_refresh_count`,
`adamw_bf16_shadow_refresh_strategy`,
`block_backward_mlp_proj_dgelu_strategy`,
`block_backward_mlp_proj_bf16_grad_out_reuse_enabled`,
`block_backward_mlp_proj_dinput_before_dweight_enabled`,
`block_backward_mlp_fc_dinput_before_dweight_enabled`,
`block_backward_attn_proj_dinput_before_dweight_enabled`,
`block_backward_weight_linear_strategy`,
`non_block_forward_backward_linear_strategy`, `lm_head_logits_linear_strategy`,
`lm_head_dhidden_linear_strategy`,
`linear_bf16_gemm_count`, `linear_bf16_gemm_fast16bf_request_count`,
`linear_tk_gemm_count`,
`linear_tk_float_out_gemm_count`, `linear_cublaslt_gemm_count`,
`linear_cublaslt_descriptor_cache_enabled`, `linear_sgemm_count`,
`bf16_to_f32_vec4_count`,
`linear_bf16_a_pack_count`, `linear_bf16_a_cache_hit_count`,
`linear_bf16_cache_reset_count`, `linear_bf16_cached_a_capacity`, and
`linear_bf16_cache_entry_count`.
The cuBLASLt descriptor cache is enabled by default, so cached plans retain
matmul descriptors and matrix layouts instead of recreating them for every
GEMM; set `NFN_TILE_CUDA_CUBLASLT_DESCRIPTOR_CACHE=0` or
`NFN_NATIVE_LINEAR_CUBLASLT_DESCRIPTOR_CACHE=0` only for paired profiling
against the older descriptor-recreate path.
The default `non_block_forward_backward_linear_strategy` is
`"padded-lm-head-tk-sm120-bf16-gemm-default"` when TK GEMM is available.
The default dense GPT path also exposes
`nfn_native_tile_linear_bf16_input_weight_bf16_gelu_bf16_float32` for stored-MLP
FC+bias+GELU from the already-packed BF16 LN2 output,
`nfn_native_tile_linear_weight_bf16_gelu_bf16_float32` for stored-MLP
FC+bias+GELU and
`nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_float32` for fused
MLP projection dInput plus saved-BF16 GELU backward. Both consume persistent
BF16 block-weight shadows while keeping FP32 masters and optimizer state.
The default MLP projection backward path also exposes
`NFN_NATIVE_GPT_MLP_PROJ_DINPUT_BEFORE_DWEIGHT=1` as a diagnostic ordering
switch. It runs fused MLP projection dInput+dGELU before dWeight+bias to mirror
the llm.kittens `matmul_backward` order, and runtime JSON reports
`block_backward_mlp_proj_dinput_before_dweight_enabled`. It remains disabled by
default because the dedicated RTX 5090 5-step, 3-sample paired benchmark
measured `1.000405x` train-loop wall time and `0.999602x` tokens/sec versus the
current dWeight+bias-first order.
`NFN_NATIVE_GPT_MLP_FC_DINPUT_BEFORE_DWEIGHT=1` is the matching diagnostic
ordering switch for MLP FC backward. It runs dInput before dWeight+bias and
reports `block_backward_mlp_fc_dinput_before_dweight_enabled`, but remains
disabled by default because the dedicated RTX 5090 5-step, 3-sample paired
benchmark measured `1.000858x` train-loop wall time and `0.999153x` tokens/sec
versus the current dWeight+bias-first order.
`NFN_NATIVE_GPT_ATTN_PROJ_DINPUT_BEFORE_DWEIGHT=1` is the matching diagnostic
ordering switch for attention projection backward. It runs dInput before
dWeight+bias and reports
`block_backward_attn_proj_dinput_before_dweight_enabled`, but remains disabled
by default because the dedicated RTX 5090 5-step, 3-sample paired benchmark
measured `1.001009x` train-loop wall time and `0.999002x` tokens/sec versus the
current dWeight+bias-first order.
The default path still uses
`nfn_native_tile_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_only_float32`,
packs the incoming projection gradient to BF16 once, reuses that scratch for MLP
projection dWeight+bias and fused dInput+dGELU, and reports
`block_backward_mlp_proj_bf16_grad_out_reuse_enabled: true` when active.
Set `NFN_NATIVE_GPT_REUSE_MLP_PROJ_BF16_GRAD_OUT=0` to compare against the
previous per-stage pack path.
Active runs report `stored_mlp_forward_strategy` as
`"tk-sm120-fused-fc-bias-gelu-prepacked-ln2-bf16-shadow-weight"` and
`reuse_packed_ln2_fc_gelu_enabled: true` because the native trainer always uses
the prepacked-LN2 route. The older `NFN_NATIVE_GPT_REUSE_PACKED_LN2_FC_GELU=0`
fallback has been retired. Active runs report
`block_backward_mlp_proj_dgelu_strategy` as
`"tk-sm120-fused-dinput-dgelu-reused-bf16-grad-out-bf16-store-bf16-shadow-weight"`
when the default BF16 grad-out reuse is active, or as
`"tk-sm120-fused-dinput-dgelu-bf16-store-bf16-shadow-weight-bf16-grad-handoff-no-float-grad"`
when the default BF16-only handoff elides the unused FP32 gradient conversion.
Set `NFN_NATIVE_GPT_ELIDE_MLP_DGELU_FLOAT_GRAD=0` to compare against the
previous BF16 handoff path that also writes the FP32 gradient buffer,
`NFN_NATIVE_GPT_BF16_MLP_GRAD_HANDOFF=0` to compare against the older
float-gradient handoff, or `NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU=0` to force the
older separate dInput plus GELU-backward route for paired benchmarks. Runtime
JSON reports `block_backward_mlp_dgelu_float_grad_elided`. The one-buffer
`nfn_native_tile_float32_to_bf16_bits` converter defaults to a guarded vec4
path for aligned native GPT buffers; set `NFN_NATIVE_GPT_F32_TO_BF16_VEC4=0` or
`NFN_TILE_CUDA_F32_TO_BF16_VEC4=0` to compare against the scalar converter.
The multi-buffer `nfn_native_tile_float32_to_bf16_bits_many` path and stored
MLP activation pack/restore path also expose guarded vec4 candidates, but they
are default-off diagnostics after the CUDA 13.3 dedicated RTX 5090 paired run
measured the scalar route faster (`0.994143x` candidate/default train-loop wall
time and `1.005941x` tokens/sec). Set
`NFN_NATIVE_GPT_F32_TO_BF16_MANY_VEC4=1` /
`NFN_TILE_CUDA_F32_TO_BF16_MANY_VEC4=1` or
`NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS_VEC4=1` /
`NFN_TILE_CUDA_STORE_MLP_ACTIVATIONS_VEC4=1` only for same-script bisection;
the launchers fall back to scalar kernels when alignment or shape guards fail.
`NFN_TILE_CUDA_LINEAR_TK_FLOAT_OUT=1` or
`NFN_NATIVE_LINEAR_TK_FLOAT_OUT=1` enables an opt-in diagnostic bridge that runs
eligible BF16 linear forward GEMMs through the TK BF16-output path and converts
the BF16 result back to float32. This is not the default because the measured
full-shape TinyStories probe improved QKV forward timing but regressed overall
throughput; use `linear_tk_float_out_gemm_count` only when profiling that
candidate path.
The full GPT-2 transformer-LM trainer also exposes
`nfn_native_tile_token_cross_entropy_backward_inplace_with_workspace_float32`
for tied LM-head CE backward. That ABI overwrites the logits chunk with dlogits,
so the main trainer reports `logit_workspace_elements: 0`,
`grad_logit_workspace_elements: 0`,
`lm_head_training_logits_dtype: "bf16"`,
`lm_head_training_dlogits_dtype: "bf16"`,
`lm_head_loss_logits_dtype: "bf16"`,
`lm_head_bf16_loss_enabled: true`,
`lm_head_public_vocab_ce_enabled: true`,
`lm_head_softmax_vocab: 50257`,
`lm_head_logit_row_stride: 50304`,
`lm_head_padded_dlogits_zeroed: true`,
`lm_head_ce_backward_strategy: "public-vocab-strided-fused-row-bf16-logits-dlogits"`, and
`lm_head_grad_logits_workspace_allocated: false` instead of allocating separate
float logits or full-vocab `grad_logits` chunks. It also reports
`lm_head_bf16_logits_enabled: true`, `lm_head_bf16_logit_elements`, and
`lm_head_bf16_logit_bytes`. Train-loss sampling copies the scalar loss back
with a blocking device-to-host `cudaMemcpy` by default and reports
`lm_head_loss_copy_device_synchronize_enabled: false` plus
`lm_head_loss_copy_ordering: "blocking-cudaMemcpy-d2h"`; set
`NFN_NATIVE_GPT_LM_HEAD_LOSS_COPY_SYNC=1` or
`NFN_NATIVE_GPT2_LM_HEAD_LOSS_COPY_SYNC=1` only to reproduce the previous
sync-before-copy path for paired diagnostics. With
`NFN_NATIVE_GPT_BF16_LM_HEAD_LOSS=0`, training
still uses BF16 logits/dlogits but validation/test loss allocates and reports
the older float logits workspace for paired benchmarking. With
`NFN_NATIVE_GPT_LM_HEAD_BF16_LOGITS=0`, the same fields report the older
float32 logits/dlogits strategy. Transformer validation uses the training batch
size as its effective validation batch so loss stays on the tested full-row
LM-head CE shape; JSON reports `validation.requested_eval_batch_size` and the
effective `validation.eval_batch_size`.

The older float32 row-vector forward and query-row atomic backward kernels stay
compiled as a diagnostic/fallback path for unsupported attention shapes or
`NFN_TILE_CUDA_USE_TK_ATTENTION=0` builds. In that mode native JSON reports
`attention_backend_strategy: "tile-row-float32"`, row/scalar launch counters,
row-kernel attribute fields, pre-launch error codes, launch grid/block fields,
and the score-reuse factors.

The compiled GPT-2 trainer also reports host wall-clock timing under `timing`:
`setup_wall_ms`, `train_loop_wall_ms`, `validation_wall_ms`,
`train_compute_wall_ms`, `checkpoint_wall_ms`, `total_wall_ms`,
`optimizer_steps_per_second`, and `train_tokens_per_second`. The train-loop
measurement ends after an explicit end-of-loop device synchronization and before
the diagnostic final sample copies from device to host. The same `timing`
object now includes `setup_timing`, a host-side breakdown of native startup work such as
`setup.float_arena_materialize`, `setup.stored_mlp_activation_arena`,
`setup.zero_init`, and `setup.block_weight_bf16_initial_refresh`; this makes
allocation and initialization regressions visible without enabling CUDA-event
stage profiling. Set `NFN_NATIVE_GPT_STAGE_TIMING=1` to
add a CUDA-event profiler for the native transformer-LM loop; the
`NFN_NATIVE_GPT2_STAGE_TIMING` name remains a compatibility fallback. That
diagnostic mode records `stage_timing_enabled`, `stage_timing_max_events`,
`stage_timing_event_count`, `stage_timing_dropped_event_count`, and
`stage_timing` entries with per-stage `total_ms`, `count`, and `avg_ms` values
for token upload, model forward, block forward/recompute/backward, LM-head
backward, final-norm/embedding backward, gradient zero/clip, and AdamW update.
Diagnostic runs also emit nested entries
for LM-head logits/CE/dHidden/dWeight, block forward/recompute attention
substeps such as `block_forward.attention.qkv`,
`block_forward.attention.sdpa`, and `block_forward.attention.proj`, forward MLP
substeps such as `block_forward.mlp_fc_gelu.fc_gelu` and
`block_forward.mlp_proj.proj`, and block backward MLP projection, MLP fc,
LayerNorm/residual, attention projection, attention SDPA, and QKV phases. The
block backward records include individual dWeight+bias, dInput, activation,
residual-add, and attention-to-QKV entries such as
`block_backward.mlp_proj.dweight_bias`, `block_backward.mlp_proj.dinput`,
`block_backward.mlp_proj.gelu`,
`block_backward.attn_sdpa.to_qkv`, and `block_backward.qkv.dweight_bias`. The
stage profiler synchronizes before reading event timings, so leave it disabled
for normal throughput runs. The default profiler cap is 20000 events; set
`NFN_NATIVE_GPT_STAGE_TIMING_MAX_EVENTS=N` for longer profiling runs that would
otherwise report dropped events.

The GPT-2 block backward path uses
`nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_reuse_forward_from_merged_grad_float32`
after a matching original or recomputed TK attention forward has populated the
process attention workspace. That skips repacking Q/K/V and skips the duplicate
TK forward inside attention backward. Training JSON reports
`attention_backward_strategy:
"tk-sm120-bf16-reuse-forward-workspace-bridge"`,
`attention_backward_reuses_forward_workspace: true`, and
`attention_backward_recompute_forward_elided_per_block: 1` when this path runs.
Use the older
`nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32`
ABI for generic call sites that cannot guarantee the preceding matching forward.

The dense GPT-2 trainer also has an opt-in saved-attention path behind
`NFN_NATIVE_GPT2_STORE_ATTENTION_ACTIVATIONS=1`. It stores earlier-block TK BF16
Q/K/V/O plus float LSE during forward, restores saved O for the attention
projection recompute, and calls the saved-state backward ABI. This path is
CUDA-only and avoids graph-editor tensors, but it is disabled by default because
the 64x1024 TinyStories one-step probe regressed from about 74.4k tok/s to about
12.6k tok/s from the added attention-state storage traffic.

Native dense GPT SDK config builders accept `template_name` and `graph_file`,
which map to canonical compiled CLI `--template-name` and `--graph-file`
arguments; Python CLI aliases such as `--template`, `--preset`, and `--graph`
are normalized before handoff. Existing `neuralfn.native_gpt2` names remain
available, while new code can use the generic `neuralfn.native_gpt` aliases such
as `NativeGptRunConfig`, `build_native_gpt_compiled_cli_run_config()`, and
`run_native_gpt()`. CLI `--base-model gpt`, `gpt2`, and `gpt3` all select the
same dense GPT native target; the selected template/custom graph decides the
architecture and whether a matching C++ trainer is implemented. The full CLI
parser and planner expose those same aliases, with GPT3 changing the default
context to 2048 tokens when selected through `--base-model gpt3` or
`--template-name gpt3`, unless a custom graph or explicit sequence length was
supplied. The implicit GPT3 batch size is 32. Every shipped
GPT template name can be passed through this no-Torch selection path, and the
compiled C++ plan JSON reports
`shipped_template_catalog`, `shipped_template_catalog_count`,
`template_known`, and `resolved_native_template_name` so SDK callers can audit
the no-Python selector catalog. The public `gpt` template alias plus
`gpt2`, `gpt2_modern`, `gpt2_megakernel`, and `gpt2_moa` map to the implemented native
transformer-LM loop; `gpt2_moa` resolves to
`--native-cuda-activation moa` automatically. Structurally different shipped
template names and custom graph files are selected and reported in JSON, but
return `selected-graph-native-trainer-missing` for real training until their
native C++ Tile trainer plans are implemented. Missing custom graph paths return
`custom-graph-file-missing` with `graph_file_exists: false` and
`graph_file_size_bytes: -1`; existing custom graph files report their byte size
so callers can distinguish path typos from the still-missing native graph
compiler. Unknown template names return `unknown-template`.

GPT-2 evo's family-specific C++ binary exposes the same selector fields and
catalog for SDK/subprocess callers. Dense GPT-2-compatible templates, including
`gpt2_modern`, now report
`native-preflight-dense-gpt-layer-evo-delegate`,
`selected_graph_support_status: "native-dense-gpt-layer-evo-delegate"`, and
`selected_graph_native_runnable: true` because real runs exec the dense GPT
native trainer with `--layer-evo`. Runtime JSON from that delegate still reports
`candidate_loss_source:
"native-forward-loss-device-resident-current-batch"`,
`candidate_loss_transport: "device-to-device"`, and
`forward_candidate_evals` after native CUDA forward-only candidate scoring.
Structurally
different shipped templates, custom graph files, and typoed template names still
report non-runnable support statuses without importing the graph-backed runtime.
Use `nfn_gpt2_evo_native_train --smoke-evo-kernels --tile-ops-lib PATH` to
exercise the raw evo mutate/select/adopt ABI on CUDA device buffers and verify
best-candidate copyback without SDK tensor payloads, Torch, datasets, or
graph-editor nodes. `--native-cuda-dry-run --native-cuda-print-command` is also
handled by the family binary itself: it prints the final dense GPT delegate
command with `--train-transformer-lm --layer-evo`, preserving validation
cadence flags such as `--eval-every-steps`, before token-shard resolution or
graph-backed imports. The unified frontend preserves `--print-command` when it
prints the GPT-2 evo family command, so the two-stage no-Torch command
inspection path is reproducible from SDK and CLI subprocess wrappers.

The compiled transformer-LM loop treats `train_batch_tokens` as the effective
optimizer-step token batch, not just metadata. It computes
`grad_accum_steps = ceil(train_batch_tokens / (batch_size * seq_len))`, streams
that many cached-shard microbatches through CUDA Tile forward/backward kernels,
accumulates scaled gradients in device accumulation buffers with
`nfn_native_tile_gradient_accumulate_float32`, then clips and applies AdamW once
per optimizer step. The default SM120 shape (`batch_size=64`, `seq_len=1024`,
`train_batch_tokens=524288`) therefore runs eight native microbatches per
optimizer step. Native JSON reports `microbatch_tokens`,
`requested_train_batch_tokens`, `grad_accum_steps`,
`effective_train_batch_tokens`, `train_microbatches_completed`,
`gradient_accumulation_strategy`, and `gradient_accumulation_scale`.

The transformer-LM trainer keeps cached shard batches compact during upload:
tokens and targets are sampled directly into one pinned host arena, copied to
device as one contiguous uint16 arena with `cudaMemcpyAsync`, and consumed
directly by the token embedding, BF16 public-vocab CE loss/backward, and token
embedding weight-backward kernels. The older device-widen path remains available
only for paired profiling by setting `NFN_NATIVE_GPT_DIRECT_U16_TOKENS=0` or
`NFN_NATIVE_GPT2_DIRECT_U16_TOKENS=0`.
Native JSON reports
`token_id_direct_u16_enabled: true`,
`token_id_upload_strategy: "uint16-pinned-async-h2d-direct-kernel-consumption"`,
`token_id_host_staging: "pinned"`, `token_id_h2d_copy:
"cudaMemcpyAsync-contiguous-arena"`, `token_id_h2d_copy_calls_per_microbatch:
1`, `token_id_widen_strategy: "elided-direct-u16-kernels"`,
`token_id_widen_kernel_launches_per_microbatch: 0`, and
`token_batch_staging_strategy: "direct-sampler-to-pinned-arena"`,
`token_batch_vector_materialization: false`, and `token_id_host_validation:
false`; batch validation belongs at shard creation or a future device-side
validation pass, not in the per-step CPU hot path.

The same native trainer initializes the tied token embedding/LM-head weight on
device with `nfn_native_tile_init_gpt2_token_weight_fast_float32`. Native JSON
reports `token_weight_init_strategy: "device-vector4-power2-deterministic"` or
the fused BF16-shadow variant, plus `token_weight_threaded_init_enabled`,
`token_weight_vector4_init_enabled`, `token_weight_fast_int32_init_enabled`,
`token_weight_init_legacy_mod17_enabled`, and
`token_weight_host_materialization: false`, so startup no longer constructs and
copies the full token-weight matrix through host RAM. The default initializer
uses CUDA Tile, vectorized float4 stores for GPT-sized tables, and a power-of-two
deterministic value pattern for the full padded vocabulary table; direct
low-level Tile ABI calls use that same vectorized default when no token-init
environment variable is set. The default vector4 BF16-shadow writer keeps the
conversion-based route because the precomputed-pattern variant regressed
startup timing. Set `NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_PATTERN_INIT=1`,
`NFN_NATIVE_GPT2_TOKEN_WEIGHT_BF16_PATTERN_INIT=1`, or
`NFN_TILE_CUDA_TOKEN_WEIGHT_BF16_PATTERN_INIT=1` only when comparing against
that rejected BF16-pattern writer. Set
`NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT=0`,
`NFN_NATIVE_GPT2_TOKEN_WEIGHT_VECTOR4_INIT=0`, or
`NFN_TILE_CUDA_TOKEN_WEIGHT_VECTOR4_INIT=0` only when comparing against the
previous fast int32 Tile initializer. Set
`NFN_NATIVE_GPT_TOKEN_WEIGHT_FAST_INT32_INIT=0`,
`NFN_NATIVE_GPT2_TOKEN_WEIGHT_FAST_INT32_INIT=0`, or
`NFN_TILE_CUDA_TOKEN_WEIGHT_FAST_INT32_INIT=0` only when reproducing the older
int64 Tile-index startup route in a paired benchmark after vector4 is disabled.
The CUDA 13.3 dedicated RTX 5090 startup-only gate after the toolkit reinstall
measured vector4 against fast int32 at `0.949565x` token-weight init time,
`0.970270x` setup wall time, and `0.970405x` total wall time. Set
`NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1` only when comparing against the
not-promoted threaded CUDA initializer, and set
`NFN_NATIVE_GPT_TOKEN_WEIGHT_INIT_LEGACY_MOD17=1` only when reproducing the older
modulo-17 values in a paired benchmark.

The compiled GPT-2 transformer-LM trainer leaves train loss disabled unless
`train_loss_every_steps` / `--train-loss-every-steps` is positive. Ordinary
optimizer steps run the forward activations needed for backward, CE gradient
generation, gradient clipping, and AdamW only; validation cadence computes
validation loss from validation shards without also measuring train loss. When
enabled, train-loss logging uses device-side CE scalar accumulation across the
optimizer step and one D2H scalar copy for the logged step. The output fields
`train_loss_sparse: false`, `train_loss_sampling`,
`train_loss_on_validation_steps: false`, `train_loss_eval_count`,
`train_loss_last_step`, `train_loss_device_accumulation_strategy`,
`train_loss_host_copy_scope`, `train_loss_host_d2h_count`,
`train_loss_host_d2h_copies_per_logged_step`, and
`train_loss_microbatch_host_d2h_copies_elided_per_logged_step` describe that
contract.

Persistent block-output preservation writes each non-final block's MLP
residual-add output directly into the per-layer backward-recompute buffer,
removing the previous post-block `nfn_native_tile_copy_float32` launch while
preserving the scratch-recompute activation tape layout.
The final block output copy is elided because final LayerNorm consumes it before
backward recomputation starts; the default 12-layer run reports
`persistent_block_outputs: 11`, `persistent_block_output_write_strategy: "direct-residual2-output"`,
`persistent_block_output_copy_elided_count`, and
`final_block_output_copy_elided: true`.
Validation forwards stream through the scratch tape without copying block
outputs into persistent training-backward buffers, because no backward pass
follows validation. JSON reports `validation_persistent_block_outputs: 0` and
`validation_block_output_copies_elided: true`.
The backward pass reuses the final block activations that remain in the scratch
tape after the initial forward pass, so only the earlier blocks are recomputed;
the default JSON reports `backward_recompute_blocks: 11` and
`final_block_backward_recompute_elided: true`. The default workstation path
stores all trained blocks' `ln2_out`, MLP preactivation, and GELU activation
tensors into a BF16 arena during forward,
consumes them directly for MLP dWeight and GELU backward, and writes the stored
preactivation plus GELU activation through
`nfn_native_tile_linear_bf16_gelu_bf16_float32`. Supported SM120 GPT-2 shapes use
the ThunderKittens fused FC+bias+GELU BF16 store; unsupported shapes fall back to
the generic CUDA kernel. The trainer reports `mlp_activation_storage_strategy`,
`stored_mlp_forward_strategy`, `stored_mlp_activation_blocks`,
`stored_mlp_activation_elements`, `stored_mlp_activation_bytes`,
`stored_mlp_activation_store_kernel_launches`,
`stored_mlp_activation_restore_kernel_launches`, and
`stored_mlp_activation_backward_consumer_strategy`, plus
`backward_recompute_mlp_fc_gelu_elided`. Set
`NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS=0` to disable that higher-memory path or
`NFN_NATIVE_GPT_STORE_MLP_BLOCKS=N` to tune the saved-block cap; GPT-2-prefixed
names remain fallbacks. Packed-attention recompute also stores intermediate
block `residual1` tensors as BF16 by default, restoring them during backward to
skip recomputed attention projection and projection-residual work for the 11
earlier blocks in a 12-layer run. Set
`NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS=0` to disable this cache for
lower-memory comparisons, or set `NFN_NATIVE_GPT_FUSE_RESIDUAL1_STORE=0` to
compare against the older separate `float32_to_bf16` residual store. The default
fused store uses
`nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_bf16_norm_float32`,
which writes both the residual1 BF16 cache and the prepacked LN2 BF16 activation
consumed by stored-MLP FC+GELU in the same launch. Set
`NFN_NATIVE_GPT_FUSE_LN2_BF16_OUT=0` to reproduce the previous separate
`float32_to_bf16` LN2 prepack. Dense GPT forward also fuses each stored MLP
projection bias/residual into the next block's LN1 stats and BF16 output when
packed LN1 storage or scratch tape is available; set
`NFN_NATIVE_GPT_FUSE_MLP_RESIDUAL_NEXT_LN1=0` to reproduce the previous
next-block LN1 launch. Runtime JSON reports
`mlp_residual_next_ln1_fusion_enabled`, `mlp_residual_next_ln1_fusion_count`,
and `mlp_residual_next_ln1_strategy`. The cache adds about 1.03 GiB at the default
`64 x 1024 x 768` shape and reports
`residual1_activation_storage_strategy`, `residual1_activation_store_strategy`,
`residual1_backward_consumer_strategy`,
`stored_residual1_activation_blocks`, `stored_residual1_activation_elements`,
`stored_residual1_activation_bytes`, `fused_ln2_bf16_out_enabled`,
`stored_mlp_ln2_bf16_prepack_strategy`,
`stored_mlp_ln2_bf16_fused_store_kernel_launches`, and store/restore launch
counters. The default residual1 backward consumer reads the stored BF16 bits
directly through
`nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32`
and
`nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32`,
so earlier-block recompute can skip the older BF16-to-FP32 residual1 restore.
Set `NFN_NATIVE_GPT_BF16_RESIDUAL1_LN_BACKWARD=0` to compare against the older
restore-to-FP32 LayerNorm backward path. Rebuild the trainer-facing Tile ops
library after updating, since the compiled GPT-2 trainer now requires
`nfn_native_tile_bf16_bits_to_float32`,
`nfn_native_tile_store_mlp_activations_bf16_float32`,
`nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_bf16_norm_float32`,
`nfn_native_tile_layer_norm_apply_stats_bf16_out_float32`, and the direct BF16
backward consumer symbols at startup. Earlier-block
recompute stops after the
MLP GELU activation because backward does not consume the recomputed MLP
projection output or final residual output; JSON reports
`backward_recompute_mlp_projection_elided: true` and
`backward_recompute_final_residual_elided: true`.
The MLP projection backward path writes its dInput into the MLP fc gradient
buffer and runs `nfn_native_tile_gelu_backward_inplace_float32`, so the full
trainer does not allocate a separate hidden-size `grad_act` scratch buffer.
JSON reports `mlp_proj_backward_gelu_inplace: true` and
`mlp_proj_backward_grad_act_scratch_allocated: false`.

Saved packed-attention blocks also store only LN1 forward stats by default when
the BF16 QKV dWeight path is active: mean/rstd for the 11 earlier blocks in a
12-layer run. During backward recompute,
`nfn_native_tile_layer_norm_apply_stats_bf16_out_float32` applies those saved
stats to the current block input and writes the LN1 BF16 activation needed by
QKV dWeight without re-running the full reduction or reserving a full BF16 LN1
tape. Runtime JSON reports
`stored_packed_attention_ln1_stats_enabled`,
`stored_packed_attention_ln1_stats_blocks`,
`stored_packed_attention_ln1_stats_elements`, and
`stored_packed_attention_ln1_stats_bytes`. Set
`NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_STATS=0` for paired regression
benchmarks against the previous full-recompute path; `NFN_NATIVE_GPT2_*`
fallback names remain accepted for older scripts.
The saved-attention tape stores earlier-block LN1 BF16 outputs by default on
the workstation dense GPT path, uses them directly for QKV dWeight, and skips
saved-attention LN1 apply-stats recompute. Set
`NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_BF16=0` for the previous lower-memory
recompute route. Runtime JSON reports
`stored_packed_attention_ln1_bf16_enabled`,
`stored_packed_attention_ln1_bf16_blocks`,
`stored_packed_attention_ln1_bf16_elements`,
`stored_packed_attention_ln1_bf16_bytes`, and
`stored_packed_attention_ln1_bf16_strategy`.
Backward residual-gradient pair additions use
`nfn_native_tile_scaled_residual_add_float32` instead of zero-fill plus two
gradient-accumulate launches; `block_state_layout.residual_backward_fused`
reports this path. With LayerNorm backward residual fusion enabled, the trainer
does not allocate the fallback-only `grad_residual1_from_mlp` and
`grad_x_from_attn` activation scratch buffers; JSON reports
`block_state_layout.layer_norm_backward_residual_scratch_buffers_allocated`,
`block_state_layout.layer_norm_backward_residual_scratch_buffers_elided`, and
`block_state_layout.layer_norm_backward_residual_scratch_elements_elided`.
Gradient clipping feeds the device clip scalar directly into
`nfn_native_tile_adamw_step_with_device_scale_float32`, avoiding a separate
per-gradient-buffer scale pass before AdamW;
`block_state_layout.adamw_device_clip_scale_fused` reports this path.
The sum-of-squares phase uses `nfn_native_tile_sumsq_partials_many_float32` over
the same device-resident gradient descriptor table, so the default 12-layer path
emits one sumsq kernel launch per optimizer step instead of one per gradient
buffer. JSON reports `gradient_clip_strategy:
"fused-multi-buffer-sumsq-device-scale"`,
`gradient_sumsq_kernel_launches_per_optimizer_step`,
`gradient_sumsq_per_buffer_launches_elided`, and
`block_state_layout.gradient_clip_loop: false`.
AdamW updates use `nfn_native_tile_adamw_step_many_with_device_scale_float32`
over device-resident parameter descriptors, so the default 12-layer path updates
148 parameter buffers with one optimizer kernel launch per optimizer step
instead of one launch per buffer. JSON reports
`adamw_update_strategy: "fused-multi-buffer-device-scale"`,
`adamw_descriptor_count`, `adamw_step_kernel_launches_per_optimizer_step`, and
`adamw_per_buffer_step_launches_elided`. The raw ABI also exports
`nfn_native_tile_adamw_step_many_with_device_scale_bf16_shadow_float32`, which
can write optional BF16 block-weight shadow entries in the same descriptor-driven
Tile launch. Set `NFN_NATIVE_GPT_FUSE_ADAMW_BF16_SHADOW_REFRESH=1` only after
forcing `NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS=0`; the BF16-primary default
bypasses the shadow-refresh route, and prior paired dedicated RTX 5090 timing
did not improve native train-loop throughput for the fused shadow write.
The default native GPT optimizer uses the no-master BF16 block projection update
path. Token/position/norm/bias tensors keep using the float32 multi-buffer AdamW
ABI, while QKV, attention projection, MLP FC, and MLP projection weights update
their BF16 parameter buffers directly through
`nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_float32`.
The raw ABI also exports
`nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32`
for BF16-primary parameter updates that consume BF16 gradient buffers while
keeping AdamW first and second moments in float32.
`nfn_native_tile_sumsq_partials_many_bf16_bits_float32` computes float32
global-norm partials from BF16 gradient buffers so the BF16-gradient path can
preserve clipping semantics. The dense GPT trainer binds these BF16-gradient
exports and reports `adamw_bf16_param_bf16_grad_kernel_loaded` plus
`gradient_clip_bf16_sumsq_kernel_loaded`, but still uses the float-gradient
BF16-param entrypoint until the block-gradient arena and zeroing buffers move
to BF16.
Checkpoint export syncs those BF16 block weights back into FP32 staging buffers
before the existing version-5 BF16 checkpoint packer runs. Set
`NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS=0` to reproduce the older FP32-master
plus BF16-shadow refresh path for bisection. Runtime JSON reports
`block_weight_bf16_primary_param_update_enabled`,
`block_weight_bf16_gradient_storage_strategy`,
`block_weight_bf16_primary_param_update_count`,
`block_weight_bf16_primary_param_bf16_grad_update_count`,
`adamw_bf16_param_bf16_grad_kernel_loaded`,
`gradient_clip_bf16_sumsq_kernel_loaded`,
`adamw_float_update_descriptor_count`, `adamw_bf16_param_descriptor_count`,
`adamw_bf16_param_bf16_grad_descriptor_count`,
`adamw_float_update_kernel_launches`, `adamw_bf16_param_kernel_launches`,
`adamw_bf16_param_bf16_grad_kernel_launches`, and
`checkpoint.bf16_param_sync_kernel_launches`.
Native GPT startup also fuses tied token-weight initialization with the
persistent BF16 LM-head shadow refresh through
`nfn_native_tile_init_gpt2_token_weight_fast_with_bf16_shadow_float32`. The
default SDK/native launch path uses the CUDA Tile initializer inside that fused
ABI whenever `NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_SHADOW=1`, and the low-level
Tile ABI defaults to the same non-threaded initializer when no token-init
environment variable is set; set
`NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1` or
`NFN_TILE_CUDA_TOKEN_WEIGHT_THREADED_INIT=1` only for paired comparison against
the not-promoted threaded CUDA initializer, set
`NFN_NATIVE_GPT_TOKEN_WEIGHT_INIT_LEGACY_MOD17=1` to reproduce the previous
modulo-17 values, or set `NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_INIT=0` to
reproduce the older two-pass startup path. Runtime JSON reports
`token_weight_init_strategy`, `token_weight_fast_int32_init_enabled`,
`token_weight_vector4_init_enabled`, `token_weight_init_legacy_mod17_enabled`,
`token_weight_bf16_initial_refresh_fusion_enabled`,
`token_weight_bf16_initial_refresh_elided`,
`token_weight_padded_init_fusion_requested`,
`token_weight_padded_init_fusion_available`,
`token_weight_padded_init_fusion_enabled`, and
`token_weight_padding_zero_launches_elided`, and `startup_only=True` isolates
the setup cost for SDK-side paired timing.
Token, position, and block Linear weight gradients accumulate directly into
optimizer-step accumulation buffers in the full GPT-2 trainer. The tied LM-head
CE backward scale includes the microbatch accumulation factor, LM-head dWeight
chunks and token-embedding backward write into `accum_grad_token_weight`, and the
old full-vocab token-gradient scratch buffer is not allocated. Position
embedding backward uses the accumulate-position ABI, avoiding `grad_position_weight`
allocation and its per-microbatch copy pass. Each transformer block also writes
qkv, attention-output, MLP fc, MLP projection dWeight, LayerNorm affine, and
Linear bias gradients straight into block accumulation buffers, avoiding
per-block scratch gradient buffers and their per-microbatch copy loop.
Accumulation buffers are zeroed once per optimizer step. JSON reports
`token_gradient_accumulation_strategy: "direct-device-accumulation-buffer"`,
`token_gradient_scratch_buffer_allocated: false`,
`position_gradient_accumulation_strategy:
"direct-device-accumulation-buffer"`,
`position_gradient_scratch_buffer_allocated: false`,
`block_linear_weight_gradient_accumulation_strategy:
"direct-device-accumulation-buffer"`,
`block_linear_weight_gradient_scratch_buffers_allocated: false`,
`layer_norm_affine_gradient_accumulation_strategy:
"direct-device-accumulation-buffer"`,
`linear_bias_gradient_accumulation_strategy:
"direct-device-accumulation-buffer"`,
`linear_bias_gradient_first_write_bgrad_direct_enabled: true`,
`lm_head_ce_bf16_exp2_enabled: false`,
`per_block_gradient_buffers: 0`,
`per_block_direct_accum_gradient_buffers: 12`,
`gradient_accumulation_loop: false`,
`gradient_accumulation_copy_loop_elided: true`,
`gradient_zero_strategy: "fused-multi-buffer-accumulation-zero"`, and
`gradient_zeroed_buffer_count: 0` under `block_state_layout`. Large-row Linear
bias-gradient and LayerNorm affine-gradient reductions use the 512-row Tile
chunked atomic reduction path on the fallback route instead of cuBLAS SGEMV on
the default GPT `batch=64`, `seq=1024` shape, while small reductions can still
use cuBLAS where selected. Accumulation buffers are zeroed once
per optimizer step through coalesced contiguous-range `cudaMemsetAsync` by
default, falling back to `nfn_native_tile_fill_many_float32` over the same
descriptor table used by the fused AdamW call when CUDA memset is unavailable or
`NFN_NATIVE_GPT_CUDA_MEMSET_GRAD_ZERO=0` is set. JSON reports
`gradient_cuda_memset_zero_enabled`, `gradient_cuda_memset_zero_available`,
`gradient_zero_range_count`, `gradient_zero_cuda_memset_count`,
`gradient_zero_tile_fill_count`, `gradient_zero_kernel_launches_per_optimizer_step`,
and `gradient_zero_per_buffer_launches_elided`.
Startup also leaves block 0 on the same block-vector ownership path as every
other transformer block. The global startup buffer list excludes block-0
aliases for parameter/gradient allocation, scratch-tape activation allocation,
parameter initialization, and AdamW-state zeroing; JSON reports
`block0_duplicate_allocation_elided`,
`block0_duplicate_activation_allocation_elided`,
`block0_duplicate_parameter_initialization_elided`, and
`block0_duplicate_adamw_state_zero_elided` under `block_state_layout`.
The same trainer suballocates float buffers from one aligned CUDA device arena
instead of calling `cudaMalloc` for each float tensor. JSON reports
`float_allocation_strategy: "single-arena"`,
`float_allocation_cuda_malloc_count`, `float_arena_cuda_malloc_wall_ms`,
`float_arena_pointer_assign_wall_ms`, `float_allocation_request_count`,
`float_arena_requested_elements`, and `float_arena_allocated_elements`. Stored
MLP LayerNorm stats and saved packed-attention LN1 stats sidecars are part of
that float arena by default; set `NFN_NATIVE_GPT_FLOAT_STATS_ARENA=0` or
`NFN_NATIVE_GPT2_FLOAT_STATS_ARENA=0` only for paired startup comparisons
against the older separate sidecar allocations.
BF16 activation and scratch buffers are also suballocated from one uint16 CUDA
device arena by default, covering stored MLP activations, residual1 caches,
packed attention stores, LM-head BF16 logits, MLP BF16 scratch, packed-QKV BF16
scratch, saved packed-attention LN1 BF16 tape, and block BF16 weight shadows. Set
`NFN_NATIVE_GPT_COMBINED_BF16_ARENA=0` or
`NFN_NATIVE_GPT2_COMBINED_BF16_ARENA=0` to reproduce the older per-buffer BF16
`cudaMalloc` path during paired benchmarks. JSON reports
`uint16_allocation_strategy`, `uint16_allocation_cuda_malloc_count`,
`uint16_allocation_request_count`, `uint16_arena_requested_elements`,
`uint16_arena_allocated_elements`, `uint16_arena_cuda_malloc_count`,
`uint16_arena_cuda_malloc_wall_ms`,
`uint16_arena_pointer_assign_wall_ms`, and
`uint16_arena_suballocation_count`. The arena wall-time fields split startup
arena materialization into CUDA allocation time and host pointer-assignment
time.
Dense GPT native training defaults `NFN_NATIVE_GPT_COMBINED_DEVICE_ARENA=0`.
The trainer suballocates float buffers from one float arena and BF16/uint16
buffers from one uint16 arena. JSON reports
`float_allocation_strategy: "single-arena"` and
`uint16_allocation_strategy: "single-arena"` by default. Set
`NFN_NATIVE_GPT_COMBINED_DEVICE_ARENA=1` only for explicit arena bisection; the
combined route packs the dense GPT float arena and BF16/uint16 arena into one
aligned `cudaMalloc` and reports
`float_allocation_strategy: "combined-transformer-device-arena"`,
`uint16_allocation_strategy: "combined-transformer-device-arena"`,
`transformer_device_arena_requested`, `transformer_device_arena_enabled`,
`transformer_device_arena_cuda_malloc_count`,
`transformer_device_arena_cuda_malloc_wall_ms`,
`transformer_device_arena_pointer_assign_wall_ms`,
`transformer_device_arena_requested_bytes`,
`transformer_device_arena_allocated_bytes`, and
`transformer_device_arena_uint16_byte_offset`. The CUDA 13.3 dedicated RTX 5090
3-step rerun rejected the combined arena at `1.004991x` train-loop wall time
and `0.995098x` tokens/sec, and a startup-only rerun rejected it at `1.063067x`
setup wall time.
Runtime timing JSON also reports `setup_timing_accounted_ms`,
`setup_timing_unattributed_ms`, and `setup_timing_record_count` beside
`setup_wall_ms`. These fields summarize how much of native dense-GPT startup is
covered by explicit `timing.setup_timing` records and how much remains in
loader, symbol-resolution, and other pre-loop host overhead. The explicit setup
records include `setup.load_tile_ops`, `setup.load_cuda_runtime`, and
`setup.cuda_runtime_symbols` before arena materialization.
The native dense-GPT path loads Tile ops with lazy dynamic binding while still
validating required ABI symbols explicitly; runtime JSON reports
`tile_ops_dlopen_binding_strategy: "RTLD_LAZY"`, `tile_ops_dlopen_wall_ms`,
`tile_ops_required_symbol_scan_wall_ms`, and
`tile_ops_typed_symbol_load_wall_ms`.
CUDA runtime setup also reports `cuda_runtime_symbol_load_wall_ms` and
`cuda_runtime_version_preflight_wall_ms`.
When stored BF16 MLP activations cover every transformer block, the dense GPT
trainer also defers the validation-only float MLP scratch buffers (`fc_out` and
`act`) instead of reserving them in the startup float arena. The buffers are
allocated on the first validation pass that uses the preserve=false scratch tape.
Set `NFN_NATIVE_GPT_LAZY_VALIDATION_MLP_FLOAT_SCRATCH=0` or
`NFN_NATIVE_GPT2_LAZY_VALIDATION_MLP_FLOAT_SCRATCH=0` to reproduce the older
startup arena layout during paired benchmarks. JSON reports
`lazy_validation_mlp_float_scratch_enabled`,
`lazy_validation_mlp_float_scratch_elements`,
`lazy_validation_mlp_float_scratch_bytes`, and
`lazy_validation_mlp_float_scratch_cuda_malloc_count`.
Startup zeroes only AdamW first/second moment state as coalesced contiguous
ranges with Tile fills by default, overwrites nonzero weights through device
initializers, and zeroes gradients at each optimizer step. Set
`NFN_NATIVE_GPT_ZERO_ADAMW_STATE_RANGES=0` to force the older descriptor-driven
AdamW state fills, or `NFN_NATIVE_GPT_ZERO_ADAMW_STATE_ONLY=0` to force the
older full-arena zero for bisection. Do not re-add per-buffer zero-fill launches
for those tensors. JSON reports
`float_arena_zero_init_strategy: "adamw-state-contiguous-range-fill"`,
`"adamw-state-fill-many"`, or `"single-arena-fill"`,
`float_arena_zero_fill_count`, `adamw_state_zero_fill_count`,
`adamw_state_zero_range_count`, `adamw_state_zero_range_elements`,
`startup_per_buffer_zero_fill_elided`, and
`startup_per_buffer_zero_fill_launches_elided`; the default 12-layer shape
elides 369 per-buffer zero-fill launches.
Nonzero constant parameter initialization uses
`nfn_native_tile_fill_many_values_float32` over a device descriptor table for
float32 weights and `nfn_native_tile_fill_many_values_bf16_bits_float32` for
BF16-primary transformer block weights. With
`NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS=1`, the dense GPT trainer defaults
`NFN_NATIVE_GPT_DIRECT_BF16_BLOCK_WEIGHT_INIT=1`, initializes QKV, attention
projection, MLP FC, and MLP projection weights directly in the BF16 arena, and
skips the initial `nfn_native_tile_float32_to_bf16_bits_many` block-weight
refresh. Set `NFN_NATIVE_GPT_DIRECT_BF16_BLOCK_WEIGHT_INIT=0` to reproduce the
older float32 fill plus BF16 pack startup path while keeping BF16-primary AdamW
updates enabled. JSON reports
`direct_bf16_block_weight_initialization_enabled`,
`block_weight_bf16_initialization_strategy`,
`parameter_initialization_strategy: "split-float32-and-bf16-fill-many-values"`
or `"fused-multi-buffer-fill-values"`,
`parameter_initialization_kernel_launches_per_startup`,
`bf16_parameter_initialization_descriptor_count`,
`bf16_parameter_initialization_kernel_launches`,
`parameter_initialization_per_buffer_launches_elided`, and
`block_state_layout.parameter_initialization_loop: false`. The direct default
uses one float32 fill-many launch plus one BF16 fill-many launch at the
12-layer shape.
The descriptor tables used by parameter fill, gradient zeroing, gradient
clipping, and AdamW are suballocated from one device descriptor arena and
uploaded from one host-packed descriptor arena instead of ten separate small
startup allocations and ten descriptor H2D copies. JSON reports
`descriptor_allocation_strategy: "single-device-arena"`,
`descriptor_arena_cuda_malloc_count`, `descriptor_arena_requested_bytes`,
`descriptor_arena_bytes`, `descriptor_arena_suballocation_count`,
`descriptor_upload_strategy: "single-host-packed-arena-copy"`,
`descriptor_arena_copy_count`, `descriptor_arena_copy_calls_elided`, and
`descriptor_cuda_mallocs_elided`.
Token upload/storage buffers use combined arenas as well: one aligned device
arena holds both widened int64 token/target buffers and compact uint16 H2D
staging, while one pinned uint16 host arena holds compact source staging. JSON
reports `token_buffer_allocation_strategy: "combined-arenas"`,
`token_device_allocation_strategy: "single-device-arena"`,
`token_device_arena_cuda_malloc_count`,
`token_device_arena_suballocation_count`, and
`token_device_cuda_mallocs_elided`.
LayerNorm affine-gradient backward has an accumulate raw Tile ABI and uses a
chunked parallel atomic reduction for large row counts, avoiding the previous
single-block loop over every row and the scratch-copy pass. The LayerNorm
affine row chunk now defaults to 128 rows; set
`NFN_TILE_CUDA_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=N`,
`NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=N`, or
`NFN_NATIVE_GPT2_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=N` to compare chunk sizes in
paired native benchmarks. The 64-row and 96-row profiles changed the route and
improved train-loop wall to `0.998045x` and `0.999112x`, but remain rejected
because they missed the hot MLP-projection and LM-head backward gates. JSON reports
`layer_norm_backward_affine_strategy: "auto-chunked-atomic-accumulate"` under
`block_state_layout`.

For local compile-time kernel candidates, `tools/build_native_train_tile_ops.sh`
accepts whitespace-separated `NFN_TILE_CUDA_EXTRA_NVCC_FLAGS` and
`NFN_TILE_CUDA_EXTRA_LDLIBS` and appends them after the default SM120
ThunderKittens flags. This is intended for paired benchmark builds such as
setting
`NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP -DLLMK_SM120_APPROX_DGELU_TANH=1"`
and running `bash tools/build_native_train_tile_ops.sh /tmp/libnfn_candidate.so`;
the default library build should leave both variables unset. On the dedicated
RTX 5090, this fused-dGELU candidate stayed opt-in after paired timing: the
two-sample run was narrowly faster, but the four-sample repeat was
noise-equivalent at about `0.999900x` train-loop time and `1.000109x`
tokens/sec.

The default trainer-facing token-weight initializer uses a 4096-element CUDA
Tile shape. For paired startup bisection only, build alternate libraries with
`NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="-DNFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE=1024"`
or `2048`; `8192` is also accepted for local candidate builds but remains
non-default after the dedicated RTX 5090 9-sample startup-only comparison
measured `1.005585x` token-init time versus the 4096 default. The accepted
values are `1024`, `2048`, `4096`, and `8192`.
The CUDA 13.3.33 post-reinstall startup sweep kept the same conclusion:
8192, 2048, and 1024 all regressed the direct token-init timer
(`1.013697x`, `1.010289x`, and `1.016591x` respectively), so SDK callers should
leave the default library build unchanged unless they are collecting fresh
same-script startup evidence.

The optimizer and gradient-clipping support kernels have a separate
compile-time Tile-size bisection knob. Build a temporary library with
`NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="-DNFN_TILE_CUDA_OPTIMIZER_TILE_SIZE=2048"` or
`4096` to retile the `sumsq`, device-scale, and multi-buffer AdamW kernels;
normal builds default to `1024`. Dense GPT runtime JSON reports
`optimizer_tile_size`, `optimizer_tile_size_symbol_loaded`, and
`optimizer_tile_strategy`, and the paired benchmark's native strategy-change
gate tracks those fields so candidate libraries are visible even when all
runtime environment flags are otherwise unchanged. The CUDA 13.3 dedicated RTX
5090 2048 candidate passed the optimizer smoke but was rejected for default
promotion because the same-script 3-step/3-sample run measured no material
AdamW or gradient-clip improvement and missed unrelated strict hot-stage gates.

Wrapper-level `--native-cuda-dry-run --native-cuda-print-command` is metadata-only on the default `compiled-cli` runner: Python builds the compiled C++ argv from the dataset alias/path and leaves shard validation or raw-text rejection to the compiled frontend. It must not import `server.dataset_manager`, NumPy, tiktoken, or Torch, must not write `fineweb_train_*.bin` shards, and must not add the external `--target train_gpt2cu` bridge argument for the default Tile-CUDA backend. The compiled Tile-CUDA frontend also treats `--print-command` as a no-data/no-CUDA action, printing the exact `nfn_gpt_native_train ...` invocation before token-shard resolution, CUDA runtime loading, or driver preflight.

When a native smoke or trainer run reports CUDA error 35, the dense GPT C++
frontend now annotates the error with a runtime/driver versus blocked-device
hint. For workstation checks, compare sandboxed results with unsandboxed
`nvidia-smi` and verify the libcudart chosen by `--cuda-runtime-lib` or
`NFN_CUDA_RUNTIME_LIB` before classifying the failure as a Tile-CUDA kernel
regression.

`tools/check_native_no_torch_deps.py` is the native dependency gate for this
path. In addition to checking `pyproject.toml` and `requirements.txt` so Torch
stays out of default dependencies and the aggregate `.[all]` extra, it runs `ldd` checks for
Torch/c10/Python runtime libraries on the required compiled GPT trainer and Tile
ops library plus optional compiled native frontends/per-family trainers already
present in `build/` and built SDK binding modules matching `neuralfn/_native*.so`, then runs
`cli/scripts/train_gpt.py`, `cli/nfn.py train`,
`cli/scripts/infer_gpt.py --native-info`, `cli/nfn.py infer --native-checkpoint`,
and `neuralfn.native_gpt*` imports under an import blocker for `torch`, NumPy,
tiktoken, `server.dataset_manager`, and `nfn_impl`. When built, the compiled
`neuralfn._native_gpt`, `neuralfn._native_gpt2`, and `neuralfn._native_train`
binding modules are imported under the same blocker. The check uses a stub
compiled CLI and synthetic native checkpoint so it does not need CUDA.

Dense GPT native `--dry-run` / `--print-plan` JSON reports the implemented
compiled trainer as `native-transformer-lm-ready` with
`training_step_plan.status: "ready"`. SDK callers should treat
`remaining_validation` as the current work to close the measured SM120
throughput gap; `tools/bench_native_gpt_sm120_parity.sh` is the same-script RTX
5090 comparison gate against `llm.kittens/train-sm120.sh`. The parity wrapper
passes the NeuralFn candidate `--train-batch-tokens 524288` explicitly to match
the reference `-d 524288` batch-token contract instead of relying on a default.
It also passes `--train-loss-every-steps 0` to the NeuralFn side by default so
short parity runs measure the training loop rather than the compiled trainer's
raw-C++ default periodic train-loss accumulation path; set
`NFN_SM120_PARITY_TRAIN_LOSS_EVERY_STEPS` or generic
`NFN_SM120_TRAIN_LOSS_EVERY_STEPS` to opt back into timed train-loss logging.
The wrapper leaves `NFN_NATIVE_GPT_TRAIN_LOOP_EVENT_TIMING` unset by default so
candidate-only CUDA event records do not contaminate pass/fail parity timing.
Set `NFN_SM120_PARITY_TRAIN_LOOP_EVENT_TIMING=1` only for explicit attribution
runs that need `train_loop_cuda_event_wall_ms`,
`train_loop_cuda_event_wall_ms_per_step`,
`train_loop_cuda_event_steady_state_wall_ms`, and
`train_loop_cuda_event_steady_state_wall_ms_per_step` under `timing`.
Set `NFN_SM120_PARITY_CANDIDATE_ENV` or generic `NFN_SM120_CANDIDATE_ENV` to
append candidate-only `KEY=VALUE` overrides to the NeuralFn side of the parity
comparison. This is useful for one-off route checks such as
`NFN_SM120_PARITY_CANDIDATE_ENV='NFN_NATIVE_GPT_LM_HEAD_CE_REVERSE_ROWS=0'`,
which runs the LM-head CE natural-row diagnostic without changing the
llm.kittens baseline environment. For native-vs-native checks, use
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_ce_natural_rows`; the wrapper marks
it rejected by default because the CUDA 13.3 RTX 5090 same-script parity sample
measured `1.019563x` CUDA-event wall time, `1.019690x` steady-state CUDA-event
wall time, and `0.978913x` tokens/sec versus the reverse-row default.
It also mirrors `NFN_SM120_PARITY_ACTIVATION` or the generic
`NFN_SM120_ACTIVATION` fallback into both sides of the comparison, using
llm.kittens `-af` and NeuralFn `--native-cuda-activation`.
It also defaults selected-GPU idle polling to three utilization samples spaced
0.25 seconds apart before each measured command, matching the native-vs-native
candidate wrapper and reducing noise from transient WSL/NVML utilization
spikes on the dedicated RTX 5090.
When `eval_every_steps <= 0` or `eval_batches <= 0`, the compiled
transformer-LM loop skips validation sampler construction as well as validation
execution. Runtime JSON exposes `validation.runtime_enabled` and
`validation.sampler_constructed` for timing-profile checks.
Top-level `nfn infer --native-checkpoint PATH --prompt-tokens IDS` and
`nfn infer --checkpoint PATH --prompt-tokens IDS` now dispatch directly to the
compiled `nfn_gpt_native_train --sample-checkpoint` path before importing
`infer_gpt`, graph-backed inference helpers, Torch, NumPy, tiktoken, or dataset
managers. Use prompt tokens for the no-tokenizer path; text prompts may still
use tiktoken locally before the same native sampler is launched.
Measured runs default to a strict
`train_loop_wall_ms_per_step=1.000` metric-ratio gate, so the parity wrapper
exits nonzero when NeuralFn is slower than the llm.kittens reference on the
comparable train-loop metric. Use `NFN_SM120_PARITY_MAX_CANDIDATE_RATIO` or
`NFN_SM120_MAX_CANDIDATE_RATIO` only when intentionally running a diagnostic
sweep with different pass criteria; dry-run plans remain ungated.
Set `NFN_SM120_PARITY_DRY_RUN_PLAN=1` or
`NFN_SM120_NATIVE_DRY_RUN_PLAN=1` to write the resolved paired command plan,
CUDA selection, profile settings, and alternating sample order without
launching either command.
`tools/paired_kernel_speed.py` also summarizes categorical native strategy
fields under `baseline_native_metric_values` and
`candidate_native_metric_values`, including LM-head logits/dHidden/dWeight
routes, block linear routes, and attention routes. Use those summaries with the
numeric ratios to verify that a candidate benchmark actually exercised the
intended kernel path. The same payload now includes
`native_strategy_value_changes`, which compares those categorical fields between
baseline and candidate, and
`native_route_counter_changes`, which compares tracked route counters such as
TK, cuBLASLt, BF16 GEMM, LM-head logits, BF16 packing/cache, and attention
launch counts between baseline and candidate. When candidate-only environment
knobs are present but tracked counters, strategy values, and linear-shape plan
metadata are unchanged, the text output warns that timing-only improvements
should be treated as noise until a route, strategy, or separate kernel-level
attribution confirms the candidate.
Use `--require-native-route-change` to make that condition a hard failure. The
SM120 candidate wrapper enables the gate automatically for measured candidate
changes, and `NFN_SM120_NATIVE_REQUIRE_ROUTE_CHANGE=0` disables it only for
explicit diagnostics.
When native stage timing is present, the text report also prints the high-value
LM-head backward substages (`logits`, `ce`, `dhidden`, `dweight`, and optional
`dhidden_dweight_concurrent`) and block-backward substages across MLP FC,
MLP projection, LN2 residual, attention projection, attention SDPA
grad-out/to-QKV, QKV dInput/dWeight, and LN1 residual work. Use those printed
ratios for CUDA 13.3 RTX 5090 parity work before promoting any candidate that
only changes a coarse `lm_head_backward` or `block_backward` total.
For automated candidate rejection, pass repeatable
`--max-candidate-ratio [STAT:]METRIC=RATIO` arguments to
`tools/paired_kernel_speed.py`, or set
`NFN_SM120_NATIVE_MAX_CANDIDATE_RATIO` /
`NFN_SM120_CANDIDATE_MAX_CANDIDATE_RATIO` for
`tools/bench_native_gpt_sm120_candidate.sh`. Example gates are
`stage.lm_head_backward.total_ms=1.000`,
`median:train_loop_wall_ms_per_step=1.000`, and
`max:stage.block_backward.total_ms=1.010`; the statistic defaults to `mean`.
Missing metrics fail the gate so a run cannot pass without the stage
attribution it was supposed to measure.
The SM120 candidate wrapper also installs default gates for measured runs that
change the candidate Tile ops library, candidate-only environment, or
candidate-only extra args: `train_loop_wall_ms_per_step=1.000` is always gated,
and stage-timed runs additionally gate `stage.lm_head_backward.total_ms`,
`stage.block_backward.total_ms`, and `stage.block_backward.mlp_proj.total_ms` at
`1.000`. Set an explicit `NFN_SM120_NATIVE_MAX_CANDIDATE_RATIO` /
`NFN_SM120_CANDIDATE_MAX_CANDIDATE_RATIO` list to override those defaults.
The wrapper also requires a tracked route, strategy, linear-shape, or cuBLASLt
plan change for measured candidates; timing-only changes with no implementation
attribution fail even if a coarse wall-time sample happens to improve.
After the CUDA Toolkit 13.3.33 WSL reinstall, the dedicated RTX 5090 validation
pass rebuilt every native trainer and passed the GPU-visible native/Tile pytest
gate (`247` tests), the GPT template preset suite (`26` tests), the full
CUDA-visible repository suite (`1185` tests, `4` skips, plus `468` subtests), and the
no-Torch native dependency verifier. Performance candidate gates are still
allowed to fail when they reject slower routes; rejected reruns include
extra-large-K cuBLASLt LM-head dHidden, one-shape cuBLASLt heuristic overrides
for `768,65536,3072,N,N` and `768,50304,8192,N,T`, token-weight vector4/threaded
startup initializers, cudaMallocAsync arenas, full-logit LM-head reuse, and the
65536-row full-batch LM-head chunk timeout. Use
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_row_chunk_65536` only to reproduce
that rejected full-resident LM-head diagnostic; it expands to
`NFN_NATIVE_GPT_ALLOW_UNSAFE_LM_HEAD_ROW_CHUNK=1` plus
`--lm-head-row-chunk-size 65536`, and the latest dedicated RTX 5090 rerun timed
out at 360s while the safe 32768-row baseline completed. The promoted 32768-row LM-head
chunk candidate reduced classifier chunk launches from `320` to `80` over the
5-step paired run and measured `0.998625x` train-loop wall time versus the
8192-row default; the 16384-row candidate regressed at `1.008471x`.
Unsupported template names and custom graph files still report
`selected-graph-native-trainer-missing` instead of falling back to Torch.

`NativeGpt2RunConfig` carries `kernel_backend` and `tile_ops_lib` for the
compiled CLI path. `kernel_backend` now defaults to `"tile-cuda"` and
`train_transformer_lm` defaults to `True`, so SDK-built compiled configs launch
the NeuralFn-owned 12-layer transformer/LM trainer unless the caller opts out.
That trainer drives block parameter allocation, initialization, gradient
zeroing, gradient clipping, AdamW updates, checkpoint export, activation tape,
forward block execution, and backward block execution from per-block C++
state/tape vectors; `block_state_layout` includes loop flags for those paths
plus separate global and per-block gradient partial counts. The trained block
count is now the configured GPT-2 layer count; the default dense GPT-2 path uses
12 trained layers with one scratch activation tape and 11 persistent block outputs.
`NFN_NATIVE_GPT_FULL_ACTIVATION_TAPE=1` / `NFN_NATIVE_GPT2_FULL_ACTIVATION_TAPE=1`
is a diagnostic-only benchmark switch that allocates one activation tape per
block and skips backward recompute, reporting `full_activation_tape_enabled`,
`activation_tape_count`, and a `full-forward-tape...` strategy in
`block_state_layout`. The default remains scratch-recompute because the
dedicated RTX 5090 paired check measured the full-tape candidate much slower.
The Tile plan includes a GPT-2 parameter layout plus forward, backward,
and optimizer stage sequence for the current 12-layer trainer.

`run_native_gpt2(config, runner="auto")` now has an explicit runner boundary:

- `auto` prefers an in-process C++ binding module named `neuralfn_native_gpt2` or `neuralfn._native_gpt2`, then the compiled no-Python CLI, then the compiled launcher. It no longer falls through to the external `train_gpt2cu` subprocess when NeuralFn native artifacts are missing. Alias-only configs created by `build_native_gpt2_compiled_cli_run_config()` still execute the compiled CLI argv through that binding, so SDK auto mode keeps shard resolution in C++ instead of attempting raw `train_gpt2cu` with empty `-i` / `-j` values.
- `binding` requires that C++ binding and raises if it is unavailable.
- `compiled-cli` requires the compiled `nfn_gpt_native_train` binary and raises if it is unavailable. The old `cli` spelling is no longer accepted.
- `launcher` requires the compiled `nfn_gpt2_tile_train` launcher and raises if it is unavailable.

SDK callers that want the process-replacement behavior used by the CLI wrappers
can call `exec_native_gpt(config)` or the compatibility
`exec_native_gpt2(config)`. These functions default to `runner="compiled-cli"`,
apply the same CUDA device/max-connection/lazy-module-loading environment as
`run_native_gpt()`, and then call `execvpe` for `compiled-cli` or `launcher`
runners. They intentionally reject binding mode because an in-process extension
cannot replace the Python process.

`native_gpt_runner_status()` returns the resolved mode and diagnostic reason, and `write_native_gpt_run_config()` includes that status in the JSON payload. Set `NFN_NATIVE_GPT_BINDING=0` to test launcher fallback paths even when the compatibility `neuralfn._native_gpt2` binding is built locally; the older `native_gpt2_runner_status()`, `write_native_gpt2_run_config()`, and `NFN_NATIVE_GPT2_BINDING` names remain compatibility fallbacks.

The unified native frontend has a separate SDK wrapper in `neuralfn.native_train`. Build its C++ extension with `bash tools/build_native_train_binding.sh`; `run_native_train(build_native_train_run_config("gpt", ["--tinystories"]), runner="auto")` then prefers `neuralfn._native_train` before falling back to a compiled native CLI. Dense GPT configs dispatch directly to `nfn_gpt_native_train --model-family ...`, while other known compiled families (`gpt2-evo`, `llama`, `mixllama`, `jepa`, `semantic-router-moe`, and `deepseek-v4`) dispatch directly to their family binary when `build/nfn_<family>_native_train`, an installed command, or the matching `NFN_NATIVE_<FAMILY>_CLI` override is available. The binding can execute configs with `argv`, `compiled_cli_argv`, or `launcher_argv`; GPT alias-only configs prefer `compiled_cli_argv` so dataset alias and shard resolution stay inside the compiled frontend. Dense GPT SDK helpers accept `model_family="nanogpt"` and resolve the default template to `nanogpt` unless a graph or non-default template is supplied; `nano_gpt` and `nano-gpt` canonicalize to `nanogpt` before the direct C++ handoff is built. The generic CLI subprocess fallback defaults `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and `CUDA_MODULE_LOADING=LAZY` only when unset, matching the GPT-specific native launchers; set `NativeTrainRunConfig.cuda_visible_devices` or the environment to route a run elsewhere. `native_train_model_registry()` returns the JSON coverage from `nfn-native-train --list-models --json`, including `transformer_lm_status`, `token_lm_status`, and `geometry_status`; `gpt2-evo` is `implemented` with `transformer_lm_status: "native-dense-gpt-layer-evo-delegate"` because it delegates dense GPT-2-compatible runs to the native CUDA Tile transformer-LM loop with `--layer-evo`, and `nanogpt` is `implemented` with `transformer_lm_status: "native-transformer-lm"` because the shared dense GPT loop now uses the selected NanoGPT 320-wide/5-head/5-layer geometry. Set `NFN_NATIVE_TRAIN_CLI` or pass `native_train_cli=` to force the unified frontend anyway. `NFN_NATIVE_TRAIN_BINDING=0` disables the extension for fallback tests.

Dense GPT has a compiled Tile CUDA preflight in `nfn_gpt_native_train`. Use `--backend tile-cuda` / SDK `kernel_backend="tile-cuda"` for the NeuralFn-owned raw Tile ABI path. No-data preflight actions (`--check-tile-ops`, `--smoke-tile-ops`, `--smoke-optimizer-step`, `--smoke-lm-step`, `--smoke-attention-step`, `--smoke-mlp-step`, `--smoke-norm-residual-step`, and `--smoke-transformer-block-step`) run before token-shard resolution, so SDK callers can validate symbols or synthetic CUDA slices without a cached dataset; plan/smoke JSON reports `token_shards_resolved: false` when shards were skipped. `--smoke-tile-ops --tile-ops-lib PATH` loads the trainer Tile ops shared library, loads CUDA runtime, launches `nfn_native_tile_fill_float32`, copies the device buffer back, and reports JSON without Python or Torch. `--smoke-optimizer-step --tile-ops-lib PATH` allocates GPT-2-sized contiguous parameter, gradient, and AdamW moment buffers, runs one AdamW call per registered GPT-2 parameter buffer with decay/no-decay metadata, samples copyback values, and reports JSON. `--smoke-lm-step --tile-ops-lib PATH` runs a tiny GPT-2-shaped tied embedding/LM-head slice through token embedding, linear logits, full-vocab CE partials and workspace CE backward, linear input/weight backward, token embedding weight backward, and AdamW; its loss check expects summed two-row CE loss, while post-AdamW weight checks use target rows only because near-zero non-target gradients can flip sign under CUDA 13.3 floating-point noise and are covered by the separate optimizer smoke. NanoGPT's equivalent native `--smoke-lm-step` accepts loss, gradient, and weight-update absolute error up to `1e-5`; CUDA 13.3 on RTX 5090 has shown stable tied-embedding gradient drift around `6.1e-6`, which is a passing numeric smoke rather than a CUDA availability failure. NanoGPT's fused-QKV attention smoke accepts Q/K/V, attention/output, input-gradient, and output-weight-gradient drift up to `1e-4` on CUDA 13.3 while keeping QKV weight-gradient and weight-update checks at `1e-5`. NanoGPT's MLP smoke accepts output, input-gradient, and FC-gradient drift up to `1e-4` while keeping projection-gradient and weight-update checks at `1e-5`. NanoGPT's standalone attention smoke accepts attention activation/output, V-weight-gradient, and output-weight-gradient drift up to `1e-4` while keeping zero-gradient and weight-update checks tighter. `--smoke-embedding-lm-step --tile-ops-lib PATH` samples a tiny cached uint16 token batch in C++ and runs token embedding, absolute position embedding, embedding residual add, final LayerNorm, tied LM head, CE backward, embedding/norm backward, and AdamW without graph-editor payloads. `--train-embedding-lm --tile-ops-lib PATH` runs that GPT-2 embedding/final-norm/LM path as a real multi-step compiled loop over cached train shards, with validation losses from validation shards controlled by `--eval-every-steps`, `--eval-batches`, and `--eval-batch-size`. `--smoke-attention-step --tile-ops-lib PATH` runs a tiny GPT-2 model-dim attention stage through qkv projection, QKV split, SDPA forward/backward, QKV gradient merge, projection backward, and AdamW. `--smoke-mlp-step --tile-ops-lib PATH` runs a tiny GPT-2 MLP stage through c_fc projection, GELU forward/backward, c_proj projection backward, and AdamW. `--smoke-norm-residual-step --tile-ops-lib PATH` runs GPT-2 LayerNorm, scaled residual add, LayerNorm affine/input backward, gradient accumulation, and AdamW through raw Tile kernels. `--smoke-transformer-block-step --tile-ops-lib PATH` composes GPT-2 LayerNorm, fused QKV attention, real 12-head reshape/merge layout (`12 x 64`), residual adds, MLP, backward passes, gradient accumulation, projection bias gradients, and AdamW updates for all 12 GPT-2 block parameter buffers through raw Tile kernels. Dataset-backed smokes such as `--smoke-embedding-lm-step` and `--smoke-transformer-lm-step`, plus real training modes, still resolve cached train/validation shards before running. `--smoke-transformer-lm-step --tile-ops-lib PATH` samples cached uint16 tokens, preserves range-checked GPT-2 token IDs, and runs token/position embeddings, one tiny transformer block, final LayerNorm, tied LM head, CE forward/backward, transformer backward, embedding backward, and AdamW for 16 parameter buffers through raw Tile kernels. `--train-transformer-lm --tile-ops-lib PATH` runs that transformer-LM path as a full-vocab real-dim 12-layer multi-step compiled loop over cached train shards, with periodic validation records in `validation.losses` controlled by `--eval-every-steps`, `--eval-batches`, and `--eval-batch-size`. The transformer validation pass uses its own C++ validation sampler and switches the active forward batch to `eval_batch_size`; the value must be no larger than the training `batch_size` because the activation arena is allocated for the training microbatch. It uses token/position embeddings, transformer blocks, final norm, a row-chunked tied LM-head/CE workspace, transformer backward, embedding backward, device-side global norm gradient clipping, and AdamW parameter updates without Python/Torch; its CE backward path reuses the logits chunk as dlogits and its JSON reports `trained_layers: 12`, `target_layers: 12`, `vocab: 50257`, `padded_vocab: 50304`, `lm_head_public_vocab_ce_enabled`, `lm_head_softmax_vocab`, `lm_head_logit_row_stride`, `lm_head_padded_dlogits_zeroed`, `lm_head_row_chunk_size`, `logit_workspace_elements`, `grad_logit_workspace_elements`, `lm_head_ce_backward_strategy`, `lm_head_grad_logits_workspace_allocated`, `gradient_partial_count`, `gradient_clip_norm`, `sample_gradient_clip_scale`, `validation.eval_batch_size`, `validation.losses[].tokens`, and `block_state_layout` loop flags for scratch-recompute activation tape, forward blocks, and backward blocks when steps complete. `--checkpoint-metadata-smoke --output-dir PATH` writes a sparse version-5 bf16 native GPT-2 checkpoint-format file plus `DONE_########` marker for the requested `--num-layers` target shape, so `read_native_gpt2_checkpoint_info()` and native inference metadata can validate NeuralFn-owned artifacts without CUDA. Successful `--train-transformer-lm` runs write a final 12-layer trained-weight native checkpoint plus `DONE_########` marker. SDK callers can set `NativeGpt2RunConfig(smoke_tile_ops=True, smoke_optimizer_step=True, smoke_lm_step=True, smoke_embedding_lm_step=True, train_embedding_lm=True, train_transformer_lm=True, checkpoint_metadata_smoke=True, smoke_attention_step=True, smoke_mlp_step=True, smoke_norm_residual_step=True, smoke_transformer_block_step=True, smoke_transformer_lm_step=True, tile_ops_lib=..., cuda_runtime_lib=...)` or use `--cuda-runtime-lib PATH` / `NFN_CUDA_RUNTIME_LIB` when libcudart needs an explicit path.

The native GPT-2 transformer-LM trainer pads only the tensor row count: tokenizer-visible `vocab` stays 50,257, while `padded_vocab` is 50,304 for the tied token embedding/LM-head parameter, logits workspace sizing, and native checkpoint payload accounting. CE loss/backward uses `vocab` as the softmax domain and `padded_vocab` as the row stride, then zeroes padded dlogit columns before LM-head dWeight accumulation.

`NFN_NATIVE_GPT_LM_HEAD_CONCURRENT_DHIDDEN_DWEIGHT=1` and the GPT-2-prefixed fallback env name enable a diagnostic two-stream LM-head backward schedule in the compiled dense GPT trainer. After the BF16 CE+dlogits row-chunk kernel completes, the trainer records a CUDA event, waits on it from two non-blocking streams, launches LM-head dHidden and dWeight on those streams, and synchronizes both streams before the next row chunk reuses the dlogit workspace. The default remains the serial schedule because the fresh CUDA 13.3 dedicated RTX 5090 3-sample same-script confirmation measured the two-stream candidate slower (`1.002970x` train-loop wall time, `0.997039x` tokens/sec). Runtime JSON reports `lm_head_concurrent_dhidden_dweight_requested`, `lm_head_concurrent_dhidden_dweight_available`, `lm_head_concurrent_dhidden_dweight_enabled`, and `lm_head_dhidden_dweight_schedule_strategy`; stage timing reports `lm_head_backward.dhidden_dweight_concurrent` for active candidate runs.

`NFN_NATIVE_GPT_BLOCK_MLP_FC_CONCURRENT_DINPUT_DWEIGHT=1` and the GPT-2-prefixed fallback env name enable a diagnostic two-stream block-backward schedule for the MLP FC dInput and dWeight+bias pair. The trainer records a CUDA event on the default stream, waits on it from non-blocking dInput and dWeight streams, launches the independent MLP FC kernels on those streams, and synchronizes before LayerNorm backward consumes `grad_ln2`. Runtime JSON reports `block_backward_mlp_fc_concurrent_dinput_dweight_requested`, `block_backward_pair_streams_available`, and `block_backward_mlp_fc_concurrent_dinput_dweight_enabled`; paired benchmark summaries print the same route fields. The default remains the serial schedule because CUDA 13.3 RTX 5090 gating measured `1.006693x` train-loop wall time, `1.012567x` block-backward time, and `1.028021x` MLP FC backward time versus the default.

Use `NFN_SM120_NATIVE_CANDIDATE_PROFILE=mlp_fc_concurrent_dinput_dweight` to rerun this route through the native SM120 wrapper. Stage-timed runs gate `stage.block_backward.mlp_fc.total_ms` alongside the default train-loop, total LM-head, block-backward, and MLP-projection totals.

`NFN_NATIVE_GPT_BLOCK_ATTN_PROJ_CONCURRENT_DINPUT_DWEIGHT=1` and the GPT-2-prefixed fallback env name enable the matching diagnostic two-stream block-backward schedule for the attention projection dInput and dWeight+bias pair. Runtime JSON reports `block_backward_attn_proj_concurrent_dinput_dweight_requested`, `block_backward_pair_streams_available`, and `block_backward_attn_proj_concurrent_dinput_dweight_enabled`; stage timing reports `block_backward.attn_proj.dinput_dweight_concurrent` when the route runs.

Use `NFN_SM120_NATIVE_CANDIDATE_PROFILE=attn_proj_concurrent_dinput_dweight` to rerun this route through the native SM120 wrapper. Stage-timed runs gate `stage.block_backward.attn_proj.total_ms` alongside the default train-loop, total LM-head, block-backward, and MLP-projection totals. Keep the route default-off: the same-script RTX 5090 gate proved the route enabled but rejected it at `1.000183x` train-loop wall time, `1.004192x` block-backward time, and `1.089203x` attention-projection backward time.

For trainer linear-kernel bisection, `NFN_NATIVE_LINEAR_TK_DINPUT=1` still
routes every supported BF16/BF16 dInput shape through the SM120 TK bridge.
Prefer the shape-selective
`NFN_NATIVE_LINEAR_TK_DINPUT_ENABLE_SHAPE=m,n,k,opA,opB` or
`NFN_TILE_CUDA_LINEAR_TK_DINPUT_ENABLE_SHAPE=...` form for single GEMM probes;
the matching `*_DISABLE_SHAPE` aliases can exclude one shape when the broad
switch is enabled. The LM-head dHidden shape `768,8192,50304,N,N` remains
diagnostic-only because the RTX 5090 paired benchmark measured it slower than
the GEMMEx default.
For the current 32768-row LM-head chunk route, the SM120 candidate wrapper has
named profile shortcuts so the route can be retested after CUDA or driver
changes without hand-writing env strings:
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_tk_dinput_32768` expands to
`NFN_NATIVE_LINEAR_TK_DINPUT_ENABLE_SHAPE=768,32768,50304,N,N`, but is rejected
by default after the CUDA 13.3 RTX 5090 rerun routed LM-head dHidden through TK
dInput and regressed train-loop wall time to `1.045528x` plus LM-head dHidden
time to `1.132973x`. The related profile
`lm_head_cublaslt_dhidden_32768` expands to
`NFN_NATIVE_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE=768,32768,50304,N,N`. These
profiles stay default-off and must pass the same-script candidate gates before
any route promotion.
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_dhidden_fast16bf_32768` expands to
`NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF_SHAPE=768,32768,50304,N,N`, but is
also rejected by default: the CUDA 13.3 dedicated RTX 5090 stage-timed rerun
requested FAST_16BF for 48 LM-head dHidden calls, regressed total LM-head
backward to `1.004489x`, and left the targeted dHidden stage effectively flat
at `1.000265x`.
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_tk_dweight_32768` expands to
`NFN_NATIVE_LINEAR_TK_DWEIGHT_ENABLE_SHAPE=768,50304,32768,N,T`. The Tile-CUDA
candidate computes BF16/BF16 dWeight into BF16 scratch with the SM120 TK bridge
and accumulates that scratch into the existing FP32 gradient buffer. Runtime
JSON reports `linear_tk_dweight_gemm_count`; when the route is active,
`lm_head_dweight_strategy` reports
`tk-sm120-bf16-scratch-to-float32-dweight-diagnostic`. Keep it default-off: the
dedicated RTX 5090 5-step, 3-sample paired benchmark moved 80 LM-head dWeight
GEMMs from cuBLASLt to TK but regressed train-loop wall time to `1.022262x` and
`stage.lm_head_backward.dweight.total_ms` to `1.279309x`. The current CUDA 13.3
stage-timed rerun is worse, moving 48 dWeight calls to TK while regressing
train-loop wall to `1.052253x` and LM-head dWeight time to `1.337552x`.
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=mlp_proj_tk_dweight_65536` expands to
`NFN_NATIVE_LINEAR_TK_DWEIGHT_ENABLE_SHAPE=3072,768,65536,N,T`. The route uses
the same TK dWeight bridge inside the BF16/BF16 dWeight+bias ABI for the dense
GPT MLP projection bucket, then runs the existing Tile bias reducer. It remains
default-off and gates `stage.block_backward.mlp_proj.dweight_bias.total_ms` in
the paired benchmark wrapper. The CUDA 13.3 dedicated RTX 5090 one-step probe
proved the route active but rejected it at `1.019937x` train-loop wall and
`1.229754x` MLP projection dWeight+bias. After the LM-head prepack default
change, a 2-step, 2-sample rerun still regressed train-loop wall to
`1.019797x`; the wrapper now requires
`NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1` for intentional reruns.
`ce_bf16_threads_512` expands to `NFN_NATIVE_GPT_CE_BF16_THREADS=512` for
repeatable BF16 CE row-block bisection. It stays diagnostic-only: the latest
dedicated RTX 5090 stage-timed rerun changed
`lm_head_ce_bf16_threads_per_row` from `1024` to `512`, but regressed
train-loop wall to `1.012086x`, total LM-head backward to `1.051608x`, and
LM-head CE to `1.430612x` versus the 1024-thread default. Real reruns require
`NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1`; dry-run expansion stays
available without the opt-in.
`qkv_forward_bf16_fallback_65536` expands to
`NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=2304,65536,768,T,N` for repeatable
packed-QKV forward fallback bisection. It stays diagnostic-only: the latest
CUDA 13.3 RTX 5090 two-step, two-sample stage-timed gate reduced TK forward
calls but regressed the target `stage.block_forward.attention.qkv.total_ms` to
`1.143374x` versus the TK default.
`mlp_fc_forward_bf16_fallback_65536` expands to
`NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=3072,65536,768,N,N` for repeatable
MLP FC forward fallback bisection. It also stays diagnostic-only: the CUDA
13.3 RTX 5090 two-step, two-sample gate did not change tracked route counters
and regressed train-loop wall to `1.016916x`, block backward to `1.034425x`,
and the target `stage.block_forward.mlp_fc_gelu.total_ms` to `1.000722x`.
The same wrapper exposes `qkv_concurrent_dinput_dweight`, which expands to
`NFN_NATIVE_GPT_BLOCK_QKV_CONCURRENT_DINPUT_DWEIGHT=1` for repeatable
stage-timed QKV side-stream bisections. That profile remains default-off and is
gated on total QKV block-backward time; the candidate-only combined concurrent
substage is reported for inspection while the serial baseline continues to emit
split dInput and dWeight substages. The current packed-QKV one-step gate proved
the route active but rejected it at `1.009068x` train-loop wall time,
`0.991012x` tokens/sec, and `1.040672x` QKV backward.
It also exposes `lm_head_concurrent_dhidden_dweight`, which expands to
`NFN_NATIVE_GPT_LM_HEAD_CONCURRENT_DHIDDEN_DWEIGHT=1` and reports the combined
LM-head dHidden/dWeight concurrent bucket for candidate-side inspection when
stage timing is enabled. Train-loop and total LM-head timing remain the
enforceable gates for that profile because the serial baseline emits split
dHidden and dWeight substages.
The named `cuda_device_max_connections_1` profile is intentionally rejected as
a no-op: the paired SM120 wrapper already sets `CUDA_DEVICE_MAX_CONNECTIONS=1`
for both the baseline and candidate commands, matching the llm.kittens SM120
launcher policy, so it cannot prove a candidate-only kernel or scheduling
change.
Use `NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1` or
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_cooperative_backward` to exercise
the current cooperative LM-head backward ABI wrapper. The named profile is
rejected by default because the CUDA 13.3 RTX 5090 same-script rerun activated
the sequence wrapper but regressed train-loop wall time to `1.005235x` and
LM-head backward to `1.103379x`. Use
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_cooperative_backward_required` or
pass `--require-cooperative-lm-head-backward` to the compiled dense GPT CLI
when a parity/preflight run must require the strict cooperative LM-head backward
Tile ABI. Rebuilt Tile ops libraries can export the strict
`nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16` callable,
but current builds return `0` from
`nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused()`.
That means the required guard remains a strict ABI preflight and fails until a
real single-kernel or genuinely fused cooperative classifier/dHidden/dWeight
body lands. The SM120 candidate wrapper labels
`lm_head_cooperative_backward_required` as a strict probe rather than a
metric-gated speed candidate. Wrapper-only and placeholder-symbol builds still
fail the strict guard.
The non-required candidate route now explicitly enables the event-ordered
sequence wrapper and reports
`lm_head_cooperative_backward_sequence_wrapper_enabled: true`, which lets the
paired benchmark route-change gate reject no-op candidate runs while preserving
the strict fused-kernel contract.
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_ce_no_loss_llmk_style_specialized`
is the current no-loss classifier-store diagnostic. It expands to
`NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_LLMK_STYLE_SPECIALIZED=1` and keeps
`--train-loss-every-steps 0` on both sides so the comparison stays on the
no-loss optimizer-step path. The Tile CUDA route uses a dedicated BF16/u16
CE+dlogits kernel with vec8 BF16 loads and streaming vec8 stores, and runtime
JSON reports `lm_head_ce_no_loss_llmk_style_specialized_requested`,
`lm_head_ce_no_loss_llmk_style_specialized_enabled`, and
`lm_head_ce_kernel_strategy:
no-loss-llmk-style-dlogits-vec8-loads-streaming-vec8-stores`. Keep this route
diagnostic-only: the CUDA 13.3 dedicated RTX 5090 3-step, 3-sample stage-timed
gate proved the strategy change and improved train-loop wall time to
`0.994628x`, but failed strict hot-stage gates at `1.000785x` LM-head
backward, `1.002295x` LM-head CE, and `1.000536x` MLP projection.
Runtime JSON reports
`lm_head_cooperative_backward_required`,
`lm_head_cooperative_backward_requested`,
`lm_head_cooperative_backward_abi_wrapper_available`,
`lm_head_cooperative_backward_sequence_wrapper_available`,
`lm_head_cooperative_backward_kernel_available`,
`lm_head_cooperative_backward_fused_kernel_available`,
`lm_head_cooperative_backward_route_integrated`,
`lm_head_cooperative_backward_kernel_enabled`,
`lm_head_cooperative_backward_sequence_wrapper_enabled`, and
`lm_head_cooperative_backward_strategy`. Runtime JSON also reports
`lm_head_classifier_fusion_scope` and `lm_head_schedule_parity_status`, which
the paired benchmark treats as strategy values so CE-only changes are not
mistaken for the missing fused logits/CE/dHidden/dWeight LM-head schedule.
Rebuilt Tile ops libraries export the probed symbol with a typed C ABI contract
for the future cooperative route: BF16 logit/dlogit chunk, u16 targets,
optional row losses, BF16/float hidden inputs, BF16/float token weights,
dHidden, dWeight, shape metadata, loss scale, dWeight beta, flags, and stream.
Runtime JSON reports the strict ABI as available only when the loaded library
contains the wrapper symbol; wrapper-only libraries report
`lm_head_cooperative_backward_abi_wrapper_available: true` and
`lm_head_cooperative_backward_sequence_wrapper_available: true`, but
`lm_head_cooperative_backward_kernel_available: false`.
The existing
`nfn_native_tile_lm_head_classifier_backward_cooperative_fused_bf16_u16`
symbol remains the event-ordered sequence wrapper probe. The strict
probe uses the separate
`nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16` symbol plus
a nonzero result from
`nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused()`.
Current builds return `0` from that capability probe. Runtime JSON reports the
placeholder symbol as
`lm_head_cooperative_backward_fused_kernel_symbol_available` and the semantic
capability as
`lm_head_cooperative_backward_fused_kernel_capability_available`; only the
capability path makes `lm_head_cooperative_backward_kernel_available` and
`lm_head_cooperative_backward_fused_kernel_available` true. The route is not
promoted by default. The CUDA 13.3 dedicated RTX 5090 one-step, two-sample
same-script gate proved route execution with 16 cooperative sequence launches,
but rejected promotion at `1.022567x` train-loop wall and `0.977929x`
tokens/sec versus the normal native baseline.
The sequence wrapper also reports launch counters in the native training JSON:
`lm_head_cooperative_sequence_launch_count`,
`lm_head_cooperative_sequence_ce_launch_count`,
`lm_head_cooperative_sequence_dhidden_launch_count`,
`lm_head_cooperative_sequence_dweight_launch_count`,
`lm_head_cooperative_sequence_concurrent_count`,
`lm_head_cooperative_sequence_legacy_count`, and
`lm_head_cooperative_sequence_loss_bin_count`. Use these when validating a
candidate against the older kernels in the same paired benchmark; nonzero
values mean the route is still the diagnostic sequence wrapper. The paired
kernel speed tool prints these counters and treats them as route counters.
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_cooperative_loss_bins` exercises the
same strict ABI with the existing loss-bin classifier reduction inside the
cooperative sequence. The profile sets
`NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1`,
`NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=1`, and
`NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_LOSS_BINS=1`, and adds
`--train-loss-every-steps 1` to both sides of the paired run. It remains
diagnostic-only and rejected by default: the CUDA 13.3 dedicated RTX 5090
3-step, 2-sample gate requested the cooperative loss-bin route but changed no
tracked route counter, strategy value, linear shape stat, or cuBLASLt plan
entry. Real paired-wrapper launches require
`NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1` until a true
fused/cooperative LM-head kernel is integrated.

Startup-only token-weight initializer bisections can use the same profile
mechanism. `token_weight_vector4_strided`, `token_weight_threaded`,
`token_weight_fast_int32`, and `token_weight_two_pass_bf16` expand to the
matching native GPT token-initializer env flags and are intended for paired
`NFN_SM120_NATIVE_STARTUP_ONLY=1` runs. The profiles now force explicit
baseline/candidate envs for default-on switches; for example,
`token_weight_vector4_strided` compares
`NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT=0` against candidate `=1`.
Native JSON reports `token_weight_vector4_strided_init_requested` and labels
the selected path as
`device-vector4-strided-power2-deterministic[-fused-bf16-shadow]`, so the
paired route-change gate can distinguish the hidden Tile dispatch. They remain
diagnostic-only; the default compiled trainer still uses the fused vector4
FP32/BF16-shadow initializer.
On the current dedicated RTX 5090 CUDA 13.3 stack, the wrapper rejects
`token_weight_threaded` by default because a 3-sample startup-only rerun
improved total setup only through unrelated arena timing (`0.978343x`) while
the token-weight stage regressed to `1.122857x`. It also rejects
`token_weight_fast_int32` because the same startup-only gate regressed setup
wall time to `1.009714x` and token-weight initialization to `1.035894x` versus
the vector4 default. Set
`NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1` only for deliberate
post-kernel-change revalidation.
`token_weight_two_pass_bf16` is rejected for the same reason: the current
startup-only rerun kept setup wall time flat (`0.996873x`) but regressed token
initialization to `1.017739x` versus the fused BF16-shadow vector4 default.
For a direct check of the rejected vector4 BF16 pattern writer, set
`NFN_SM120_NATIVE_STARTUP_ONLY=1` and put
`NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_PATTERN_INIT=1` in
`NFN_SM120_NATIVE_CANDIDATE_ENV`; the wrapper will gate
`setup.token_weight_init.total_ms` because the candidate env mentions
token-weight initialization.

Full GPT-2 `--train-transformer-lm` runs report a `cuda_runtime_preflight`
object. Set `NFN_NATIVE_GPT_CUDA_VERSION_PREFLIGHT=1` or
`NFN_NATIVE_GPT2_CUDA_VERSION_PREFLIGHT=1` to query
`cudaRuntimeGetVersion` / `cudaDriverGetVersion` before allocation and fail
early on driver version `0` or a loaded CUDA runtime newer than the driver.
Normal workstation training leaves this version preflight off to avoid its
startup cost; allocation and kernel failures still report through native CUDA
errors.

Set `NativeGptRunConfig.startup_only=True` or
`NativeGpt2RunConfig.startup_only=True` to forward `--startup-only` to the
compiled CLI. High-level native training commands also accept
`--native-cuda-startup-only` and normalize it to the same compiled C++ flag
before dispatch; the dense native GPT parser accepts either spelling for direct
calls. Startup-only runs still resolve cached token shards, load CUDA, allocate
the full Tile-CUDA transformer training arenas, initialize native parameters,
and emit normal setup timing, but exit before optimizer steps or checkpoint
export with `status: "native-transformer-lm-startup-ready"`. Native GPT SDK
subprocess launchers set `CUDA_MODULE_LOADING=LAZY` by default when the caller
has not already set that environment variable, and runtime JSON reports the
resolved value as `cuda_module_loading`. Startup-only suppresses final
checkpoint export even when export was requested; plan/runtime JSON reports
`checkpoint_export_enabled: false` and
`checkpoint_export_startup_only_elided: true` for that case. Startup-only also
skips validation shard discovery even when validation cadence is configured,
because no validation pass can run before the process exits; JSON reports
`validation_shards_required: false` and leaves `val_shard` empty for train-only
token caches.

Set `NativeGptRunConfig.write_checkpoint=False` or
`NativeGpt2RunConfig.write_checkpoint=False` for benchmark/preflight runs that
should exclude final trained-checkpoint export. `compiled_cli_argv()` forwards
that as `--no-checkpoint`; default configs keep checkpoint export enabled.

Successful GPT-2 `--train-transformer-lm` checkpoint export packs device
float32 weights into one contiguous bf16 payload with
`nfn_native_tile_float32_to_bf16_bits_many`, copies the compact uint16 payload
to host once, and writes the native version-5 `.bin` plus `DONE_########`
marker. Training JSON reports `checkpoint.payload_pack_strategy:
"device-many-float32-to-bf16-bits-contiguous"`, `payload_pack_kernel:
"nfn_native_tile_float32_to_bf16_bits_many"`, `payload_copy_strategy:
"single-contiguous-device-payload-d2h"`, `payload_cpu_bf16_conversion: false`,
`device_pack_kernel_launches`, `d2h_copy_count`, `d2h_bytes`, and
`float32_d2h_bytes_elided`.

`bash tools/build_native_train_tile_ops.sh` builds `libnfn_native_train_tile_ops.so`, a raw C ABI over CUDA Tile kernels from `neuralfn/csrc/tile_cuda/kernels.cu`. Native C++ trainers should link this library for single-buffer and multi-buffer fill/zeroing, single-buffer and multi-buffer sumsq partials, single-buffer and multi-buffer AdamW, gradient accumulation, deterministic GPT-2 token-weight initialization, device float32-to-bf16 checkpoint payload packing, device-side global-norm clip scale finalization, device-scalar gradient scaling, reductions, linear, forced-BF16 linear, BF16-input linear, linear input/forced-BF16 input/weight/weight-accumulate/forced-BF16 weight-accumulate/forced-BF16 weight+bias-accumulate/float-input plus BF16-bits weight+bias-accumulate/bias/bias-accumulate backward, scaled residual add, fused projection bias+residual add, BF16-linear projection bias+residual add, BF16-linear projection bias+residual add with BF16 persistent side-store, and BF16-linear residual+LayerNorm, fused QKV split/merge, fused GPT-2 QKV split-to-heads, fused GPT-2 QKV bias+split-to-heads, fused GPT-2 heads-to-QKV gradient merge, fused TK bf16 attention-gradient heads-to-QKV bridge, saved TK BF16 attention workspace copy/backward, packed GPT-2 QKV TK attention backward with BF16-gradient output, reshape-heads/merge-heads, GELU forward, fused bias+GELU forward, fused bias+GELU with BF16 activation output, GELU backward, token embedding forward/weight backward, uint16-token embedding forward/weight backward, absolute-position embedding forward/backward/backward-accumulate, RMSNorm, RMSNorm input backward, LayerNorm, LayerNorm forward-with-stats, LayerNorm input plus fused input/residual-add, affine, affine-accumulate backward, stats-consuming affine+dInput+residual-add backward for float and BF16-bit inputs, LayerNorm stats-consuming input and affine-accumulate backward, softmax, float token and masked token cross-entropy partials, BF16-bits token cross-entropy partials, strided float/BF16-bits token cross-entropy partials, strided BF16-bits token cross-entropy partials with uint16 targets, token and masked token cross-entropy logits backward, strided in-place token cross-entropy logits backward, strided BF16-bits in-place token cross-entropy logits backward with uint16 targets, and scaled dot-product attention forward/backward instead of importing the PyTorch extension binding. The trainer build defines `NFN_TILE_CUDA_USE_CUBLAS_LINEAR=1` and links `libcublas`, so the exported native linear forward, BF16-input linear forward, dInput, dWeight, and accumulate-dWeight ABI symbols use GPU GEMM; the forced-BF16 and BF16-bits weight+bias accumulate ABIs use cuBLASLt `BGRADB` when supported and fall back to separate dWeight plus Tile bias-reduction launchers. The generic Tile extension build keeps the pure Tile fallback. CE logits backward uses a row-wise Tile path for vocabularies up to 1024 and a chunked row-wise path with reusable row-stat workspace for full GPT-class vocabularies; workspace-aware entrypoints consume caller-owned row-stat buffers, while the legacy non-workspace full-vocab CE backward entrypoints reuse a process-cached row-stat workspace exposed through `nfn_native_tile_token_cross_entropy_workspace_allocation_count()` and `nfn_native_tile_token_cross_entropy_workspace_row_capacity()`. The strided variants let GPT trainers compute CE over public tokenizer vocab while treating padded LM-head rows as row stride and zeroing padded dlogit columns. Linear weight, accumulate-weight, bias, and accumulate-bias backward keep the row-chunked tiled atomic fallback for builds or shapes that do not use the trainer cuBLAS path.

Run `python tools/check_native_no_torch_deps.py --skip-artifacts --json` to
verify the Python handoff layer stays Torch-free. The gate stubs the native
CLIs and blocks imports of Torch, NumPy, `tiktoken`, dataset manager modules,
`infer_gpt`, `train_gpt_native`, and `nfn_impl` while checking GPT, direct
`train_gpt_native.py` compiled-cli dry-run handoff, GPT-2-evo, NanoGPT, explicit
`nfn train --tinystories`, default `nfn train`, programmatic
`nfn.main([...], stdin_isatty=..., stdout_isatty=...)` native training, native
inference, top-level per-family `nfn train --base-model ...` dispatch,
installed `nfn:main` console-entry native inference, `neuralfn.native_train`,
and the public SDK native training exports.
It also executes the guarded legacy training scripts with no native flags
against stubbed native CLIs, proving their default direct-script path enters
native C++ before importing Torch or the Python dataset/runtime stack.
The generic compiled binding must expose both a runner symbol and a command
resolver symbol; `resolve_native_train_binding_command(config)` returns the
argv that `neuralfn._native_train` will spawn so SDK callers can assert the
compiled handoff path before training. Dense GPT bindings expose the same
inspection path through `resolve_native_gpt_binding_command(config)` and
`resolve_native_gpt2_binding_command(config)`, backed by `resolve_command` /
`resolve_native_gpt_command` / `resolve_native_gpt2_command` in the C++
extension.
For dense GPT families (`gpt`, `gpt2`, `gpt3`, and `nanogpt`), the generic
`neuralfn.native_train` SDK resolver now chooses
`build/nfn_gpt_native_train_linked` before `build/nfn_gpt_native_train` when no
explicit `NFN_NATIVE_GPT_CLI` or `NFN_NATIVE_TRAIN_CLI` override is set. This
keeps SDK-native runs on the same linked Tile-ops startup path as `nfn train`
and avoids the avoidable dynamic `dlopen` branch on workstation builds.
`exec_native_train(config)` is the generic SDK process-replacement handoff: it
resolves the same compiled native command as `run_native_train(...,
runner="compiled-cli")`, sets the CUDA environment defaults, then calls
`execvpe` so long-running training does not retain a Python parent. The binding
route remains available through `run_native_train(..., runner="auto")` when the
caller needs a returned status code.
Dense GPT training also requires optimized attention by default. If the Tile
ABI drops into the scalar attention fallback, the native trainer marks the run
failed before final checkpoint export and reports
`optimized_attention_required: true`,
`attention_forward_scalar_launch_allowed: false`, and the scalar launch count.
Pass `--allow-scalar-attention-fallback` only for diagnostic benchmark
bisections.

Full GPT-2 `--train-transformer-lm` uses `nfn_native_tile_gelu_add_bias_bf16_act_float32` to write float preactivation, float GELU activation, and BF16 GELU activation bits from one CUDA Tile launch. The BF16 saved-activation backward route uses `nfn_native_tile_gelu_backward_inplace_bf16_bits_float32` as a CUDA Tile kernel too, so the BF16 GELU forward/backward path no longer falls back to scalar CUDA element kernels. The MLP projection then consumes those BF16 bits through `nfn_native_tile_linear_bf16_input_bits_float32`, avoiding another activation pack before GEMM. Training JSON reports `mlp_proj_forward_activation_strategy`, `mlp_forward_act_bf16_elements`, and `mlp_forward_act_bf16_bytes` for this scratch.

Full GPT-2 `--train-transformer-lm` also exposes
`nfn_native_tile_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_only_float32`
for the SM120 TK fused MLP projection dInput+dGELU path when the caller already
packed the projection grad-out to BF16. The dense GPT trainer uses this ABI by
default so dWeight+bias and dInput+dGELU share one BF16 grad-out pack.
If the SM120 TK fused route is unavailable in a non-default build or shape, this
BF16-only ABI now falls back to BF16-output GEMM plus in-place BF16-bits dGELU
instead of leaving the handoff buffer unwritten.

Dense GPT-2 `--train-transformer-lm` uses the LayerNorm stats ABI by default. Forward writes row mean/rstd for each scratch-tape LayerNorm, and earlier blocks that reuse stored BF16 MLP activations keep their LN2 stats in a small float sidecar so backward can call the stats-consuming kernels without recomputing row statistics from stale scratch-tape state. Block backward uses `nfn_native_tile_layer_norm_backward_affine_residual_add_accumulate_with_stats_float32` for float LN1/fallback LN2 and `nfn_native_tile_layer_norm_backward_affine_residual_add_accumulate_with_stats_bf16_bits_float32` for stored-BF16 residual1 LN2. These fused kernels accumulate dWeight/dBias, compute dInput, apply residual scaling, and add the upstream residual gradient in one launch for GPT-width `dim <= 1024` shapes. Set `NFN_NATIVE_GPT_FUSE_LN_BACKWARD_AFFINE_RESIDUAL=0` for the previous affine-accumulate plus dInput/residual-add pair, set `NFN_NATIVE_GPT_BF16_RESIDUAL1_LN_BACKWARD=0` for the older restore-to-FP32 residual1 path, or set `NFN_NATIVE_GPT_FUSE_LN_BACKWARD_RESIDUAL=0` for the older separate LayerNorm dInput plus residual-add route. `NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=N` tunes the row-chunk size used by the chunked affine-gradient/residual fused kernels; the default is 128 rows after the CUDA 13.3 dedicated RTX 5090 same-script gate measured `0.997750x` train-loop wall time and `1.002270x` train tokens/sec versus the older 256-row route. The paired profile `layernorm_affine_row_chunk_128` forces its baseline to 256 rows so the default can still be rechecked against the older route; `layernorm_affine_row_chunk_64`, `layernorm_affine_row_chunk_96`, and `layernorm_affine_row_chunk_512` remain rejected diagnostics, and real reruns require `NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1`. The 64-row rerun changed `block_state_layout.layer_norm_backward_affine_row_chunk_size` from 128 to 64 and improved train-loop wall to `0.998045x`, but failed hot-stage gates at `1.004276x` MLP projection and `1.000446x` LM-head backward. The 96-row rerun changed the same route and improved train-loop wall to `0.999112x`, but failed the gates at `1.000296x` and `1.000002x`; the 512-row route had already regressed train-loop wall to `1.019837x`. The separate Tile linear-bias reducer used by split dWeight+bias diagnostics now defaults to 256-row chunks and can be profiled with `NFN_NATIVE_GPT_LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE=N`, `NFN_NATIVE_GPT2_LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE=N`, or `NFN_TILE_CUDA_LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE=N`; the paired wrapper profile `linear_bias_row_chunk_256` now compares the promoted 256-row route against the older 512-row baseline, while `linear_bias_row_chunk_1024` remains a rejected wider-chunk diagnostic and also requires `NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1` for real reruns. The CUDA 13.3 dedicated RTX 5090 same-script gate measured the 256-row route at `0.995306x` train-loop wall time, `1.005073x` train tokens/sec, and `0.994292x` total wall time across three five-step samples. Training JSON reports `layer_norm_stats_strategy`, `layer_norm_backward_reuses_forward_stats`, `layer_norm_stats_disabled_by_fused_residual_ln2`, `layer_norm_backward_residual_fusion_enabled`, `layer_norm_backward_affine_residual_fusion_enabled`, `layer_norm_backward_affine_residual_fused_kernel_launches`, `block_state_layout.layer_norm_backward_affine_row_chunk_size`, `block_state_layout.linear_backward_bias_row_chunk_size`, `layer_norm_backward_residual_strategy`, `residual1_backward_consumer_strategy`, `stored_mlp_layer_norm_stats_elements`, `stored_mlp_layer_norm_stats_bytes`, and `stored_mlp_layer_norm_stats_standalone_cuda_malloc_count`; the default combined-float-arena path should report a standalone malloc count of `0`.

Trainer loops that own attention forward/backward ordering can use the raw TK
attention backward-to-QKV forward-workspace reuse ABI; generic SDK callers should
prefer the normal attention backward-to-QKV ABI.

Trainer loops that want to remove scratch recompute can use
`nfn_native_tile_attention_tk_store_forward_workspace_bf16` immediately after a
matching TK causal attention forward to copy BF16 Q/K/V/O plus float LSE into
caller-owned device buffers. Later backward can call
`nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_saved_tk_bf16_from_merged_grad_float32`
with the saved buffers and a row-major merged gradient to produce row-major
`grad_qkv` without repacking Q/K/V or rerunning the forward pass. This ABI is
shape-limited to the same TK SM120 causal attention constraints as the normal
bridge and returns the native missing-kernel status when the build or shape is not
TK-compatible. The trainer ABI also exports
`nfn_native_tile_scaled_dot_product_attention_store_tk_bf16_float32`, which
writes Q/K/V/O/LSE directly into caller-owned saved-attention buffers during
forward instead of copying the process workspace afterward. For dense GPT-2, set
`NFN_NATIVE_GPT2_STORE_ATTENTION_ACTIVATIONS=1` only when profiling that direct
saved path; the current `64 x 1024` TinyStories probe still regresses to about
6.0k tokens/s because saved-attention backward dominates the step, so the
default remains the faster recompute plus process-workspace reuse path.

Dense GPT defaults to eliding unused FP32 attention-projection and
MLP-projection scratch-tape buffers when BF16 projection-residual is active.
Runtime JSON reports `float_projection_outputs_elided: true`,
`float_projection_output_elements_elided`, and matching
`block_state_layout.float_projection_output_*` counters. Set
`NFN_NATIVE_GPT_ELIDE_FLOAT_PROJECTION_OUTPUTS=0` or
`NFN_NATIVE_GPT2_ELIDE_FLOAT_PROJECTION_OUTPUTS=0` only for paired bisection
against the older reservation.

Full GPT `--train-transformer-lm` now defaults to the packed-QKV attention
layout. It writes LN1 output to BF16 with
`nfn_native_tile_layer_norm_with_stats_bf16_out_float32`, writes the QKV
projection as packed BF16 bits with QKV bias fused into the SM120 TK BF16 GEMM,
runs the SM120 TK packed attention bridge over that row-major packed QKV tensor,
and feeds the packed BF16 attention output directly into the attention
projection forward GEMM and dWeight accumulation.
Backward keeps packed attention `dQKV` in BF16 by
default: the BF16-output packed backward ABI writes directly into a non-aliased
BF16 scratch buffer that reuses the MLP BF16 scratch after MLP backward is done,
then QKV dWeight+bias reuses the saved LN1 BF16 activation with
`nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32`
and QKV dInput consumes the same BF16 gradient bits with BF16 block weights. Native
combined uint16 arena allocation must reserve LN1 BF16 output, packed QKV BF16,
and packed attention output BF16 for this scratch layout.
plan and training JSON report `qkv_forward_layout_strategy:
"packed-qkv-bf16-no-split"`, `qkv_bias_layout_strategy:
"packed-qkv-bf16-bias-fused-tk-gemm"`, `qkv_bias_fused_tk_gemm_enabled`,
`attention_projection_input_strategy:
"packed-o-bf16-direct-gemm"`, `attention_packed_output_unpack_strategy:
"elided-direct-bf16-projection"`, `attention_backward_bf16_qkv_grad_handoff_enabled`,
`attention_backward_direct_bf16_qkv_grad_scratch_enabled`,
`attention_backward_direct_bf16_qkv_grad_scratch_elements`,
`qkv_forward_ln1_bf16_enabled`,
`qkv_backward_layout_strategy: "packed-qkv-bf16-gradient-handoff"`,
`attention_backward_qkv_bridge_strategy:
"tk-sm120-packed-qkv-direct-bf16-grad-scratch-handoff"`, and
`attention_backward_strategy:
"tk-sm120-packed-qkv-bf16-backward-direct-bf16-grad-scratch-handoff"` when the packed route is
active. Set `NFN_NATIVE_GPT_DIRECT_BF16_QKV_GRAD_SCRATCH=0` to reproduce the
older workspace-to-packed-QKV-buffer copy path, or set
`NFN_NATIVE_GPT_FUSE_QKV_BIAS_TK_GEMM=0` to reproduce the older separate
packed BF16 QKV bias-add launch, or set
`NFN_NATIVE_GPT_LN1_BF16_QKV_FORWARD=0` to reproduce the previous float32-LN1
QKV forward path. BF16/BF16 QKV dWeight+bias accumulation is default-on.
Runtime JSON reports `block_backward_bf16_qkv_dweight_enabled` and
`block_backward_qkv_dweight_strategy:
"packed-ln1-bf16-qkv-bf16-grad-dweight-bias-accumulate"`. Set
`NFN_NATIVE_GPT_BF16_QKV_DWEIGHT=0` to reproduce the previous float32-LN1
dWeight path. The trainer ABI also exports
`nfn_native_tile_linear_backward_input_weight_bf16_to_bf16_bits_float32`,
`nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_bf16_merged_grad_float32`,
and
`nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_bf16_merged_grad_float32`
for the opt-in BF16 attention grad-out handoff experiment. Set
`NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1` only for paired profiling: attention
projection dInput writes BF16 grad-out bits directly, and packed attention
backward consumes those bits before writing BF16 `dQKV`. Runtime JSON reports
`attention_backward_bf16_grad_out_handoff_enabled`,
`attention_backward_grad_out_dtype`,
`attention_backward_bf16_grad_out_scratch_elements`,
`attention_backward_bf16_grad_out_scratch_bytes`, and the updated
`attention_backward_qkv_bridge_strategy`. The path remains default-off because
paired dedicated-RTX-5090 timing measured it slower than the current float
grad-out default. A Tile ops library built with
`NFN_TILE_CUDA_EXTRA_NVCC_FLAGS=-DLLMK_SM120_ATOMIC_DQ` now compiles through a
dedicated packed-QKV candidate wrapper that uses float dQ scratch and re-packs
the Q gradient into the BF16 packed `dQKV` buffer, but it remains
default-off/rejected because the dedicated RTX 5090 same-script benchmark
measured `1.134435x` train-loop wall time and `0.881527x` tokens/sec versus the
current non-atomic packed-gradient default. The trainer ABI also exports
`nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_to_bf16_bits_float32`,
which accumulates BF16 activation and BF16 gradient dWeight into a BF16 staging
buffer while still accumulating bias in float32. Dense GPT can opt into that
QKV and MLP FC staging experiment with
`NFN_NATIVE_GPT_BF16_BLOCK_DWEIGHT_STAGING=1`. With BF16 primary block weights,
the staged gradients feed
`nfn_native_tile_sumsq_partials_many_bf16_bits_float32` and
`nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32`
directly, so the old BF16-to-FP32 staging flush is skipped. It remains
default-off because the paired dedicated-RTX-5090 benchmark measured the direct
BF16 optimizer candidate at `1.0325x` the default train-loop time, while the
older flush candidate measured about `1.0245x` slower; runtime JSON reports
`block_dweight_bf16_staging_enabled`, `block_dweight_bf16_staging_strategy`,
staging allocation sizes, zero count, BF16 clip/AdamW descriptor counts, and
BF16-gradient AdamW launch counts. Set
`NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC=1` to profile dense GPT transformer-LM startup
through CUDA runtime `cudaMallocAsync` / `cudaFreeAsync` for the large device
arenas. The async allocator path is default-off because paired dedicated-RTX-5090
timing measured it slower than the default `cudaMalloc` arena path. The latest
CUDA 13.3 explicit arena-gated retest measured `1.177290x` setup wall time,
`2.243472x` float-arena materialization, `1.716820x` uint16-arena
materialization, and `1.176781x` total startup wall time. Runtime JSON
reports `device_allocator_strategy`, `device_cuda_malloc_async_requested`,
`device_cuda_malloc_async_enabled`, async symbol availability, allocation/free
counts, and `device_cuda_malloc_async_fallback_count`. The same dense GPT
transformer-LM arenas now default to split float and BF16/uint16 device arenas
(`NFN_NATIVE_GPT_COMBINED_DEVICE_ARENA=0`); runtime JSON reports
`float_allocation_strategy: "single-arena"` and
`uint16_allocation_strategy: "single-arena"` by default. Set
`NFN_NATIVE_GPT_COMBINED_DEVICE_ARENA=1` only to compare against the combined
arena route; that opt-in path reports the combined allocation strategies plus
`transformer_device_arena_requested`, `transformer_device_arena_enabled`,
`transformer_device_arena_cuda_malloc_count`,
`transformer_device_arena_requested_bytes`,
`transformer_device_arena_allocated_bytes`, and
`transformer_device_arena_uint16_byte_offset`. The CUDA 13.3 dedicated RTX 5090
3-step rerun rejected the combined arena at `1.004991x` train-loop wall time and
`0.995098x` tokens/sec.
Set
`NFN_NATIVE_GPT_BF16_QKV_GRAD_HANDOFF=0` to compare against the
older packed path that expands `dQKV` to float32 before QKV dWeight/dInput. Set
`NFN_TILE_CUDA_BF16_BIAS_INPLACE_TILE=0`,
`NFN_NATIVE_GPT_BF16_BIAS_INPLACE_TILE=0`, or
`NFN_NATIVE_GPT2_BF16_BIAS_INPLACE_TILE=0` to compare against the older scalar
CUDA BF16 bias kernel. Set `NFN_NATIVE_GPT_PACKED_QKV_ATTENTION=0` only when explicitly benchmarking or using the split-QKV fallback. The packed backward batch cap
defaults to 64 so the workstation `64 x 1024` microbatch runs as one TK backward
chunk; set `NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP=48` to reproduce
the previous split for paired benchmarks. Packed attention dprep keeps the older row-linear launch by default; set
`NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_GRID3D=1` only for paired timing of the
diagnostic 3D batch/head/time launch, which avoids per-row division/modulo but
measured slower on the dedicated RTX 5090. Set
`NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_WARPS=N` only for row-grouping
bisection of that dprep launch; the default remains `3`.
Within the row-linear route, GPT `heads=12, head_dim=64` BF16-grad dprep now
defaults to a specialized unrolled HD64 kernel. Set
`NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_HD64_SPECIALIZED=0` to reproduce the
older generic row dprep kernel for paired bisection; the GPT-2-prefixed name
remains the fallback.
Set `NFN_NATIVE_GPT_ATTENTION_BACKWARD_SECTION_TIMING=1` only for one-off
diagnostics that need direct dprep-vs-TK timing inside packed attention
backward. The diagnostic path records CUDA events around each section and
therefore synchronizes the stream; runtime JSON reports
`attention_backward_section_timing_enabled`,
`attention_backward_dprep_timing_us`,
`attention_backward_dprep_timing_count`, `attention_backward_tk_timing_us`, and
`attention_backward_tk_timing_count`. The GPT-2-prefixed
`NFN_NATIVE_GPT2_ATTENTION_BACKWARD_SECTION_TIMING` and low-level
`NFN_TILE_CUDA_ATTENTION_BACKWARD_SECTION_TIMING` names remain fallbacks.
`attention_backward_tk_launch_count`
now counts packed backward chunks instead of only wrapper calls. When
`NFN_NATIVE_GPT_STORE_ATTENTION_ACTIVATIONS=1` is set with the split path,
runtime JSON switches `attention_activation_storage_strategy` to
`"tk-bf16-direct-forward-store-saved-backward"`, `attention_backward_strategy` to
`"tk-sm120-bf16-saved-forward-workspace-bridge"`, and reports saved-attention
arena sizes plus store/restore/backward counts. For the packed path, the trainer
stores packed BF16 QKV and packed BF16 O for all 12 trained blocks by default on
the dedicated RTX 5090 workstation shape, and now stores per-row TK `lse` beside
those tensors by default; runtime JSON reports `packed_attention_activation_storage_strategy:
"packed-qkv-o-bf16-forward-store-direct-backward"`,
`stored_packed_attention_activation_blocks: 12`,
`stored_packed_attention_lse_enabled`, `stored_packed_attention_*` counts/bytes,
`attention_backward_strategy:
"tk-sm120-packed-qkv-bf16-saved-activation-backward-direct-bf16-grad-scratch-handoff"`, and
`block_recompute_saved_packed_attention` timing. The default packed recompute
path also stores intermediate residual1 tensors in BF16 and skips the
packed-attention recompute projection/residual subpath; that store is fused into
the attention residual+LN2 Tile kernel by default. Set
`NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_ACTIVATIONS=0` or
`NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS=0` for lower-memory comparisons, set
`NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE=0` to compare against the older shared
workspace-LSE path in paired benchmarks, set
`NFN_NATIVE_GPT_FUSE_RESIDUAL1_STORE=0` to compare against the older separate
residual1 store, or set `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=N` to tune
the saved packed-attention block cap. The GPT-2-prefixed variables remain
fallbacks for existing scripts.

Full GPT-2 `--train-transformer-lm` also fuses the MLP `c_fc` bias add with
GELU. `nfn_native_tile_gelu_add_bias_float32` consumes the no-bias CUBLAS
projection output, writes the biased preactivation required by GELU backward,
and writes the GELU activation in one Tile pass. Native plan and training JSON
report `mlp_fc_bias_gelu_strategy: "fused-bias-preactivation-gelu"` and one
elided legacy launch per block.

The trainer-facing native GELU ABI uses the GPT-style tanh approximation for
forward, fused bias+forward, explicit backward, and in-place backward:
`0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`. Keep that approximation aligned
across `nfn_native_tile_gelu_float32`, `nfn_native_tile_gelu_add_bias_float32`,
`nfn_native_tile_gelu_backward_float32`, and
`nfn_native_tile_gelu_backward_inplace_float32`; graph-backed Torch execution can
keep its own PyTorch GELU semantics.

The trainer ABI also exports
`nfn_native_tile_linear_bias_residual_layer_norm_float32`, which fuses the
attention projection bias add, residual add, and following MLP LayerNorm forward.
The stats-preserving variant
`nfn_native_tile_linear_bias_residual_layer_norm_with_stats_float32` also writes
mean/rstd for backward reuse, and
`nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_float32`
additionally writes the native GPT residual1 activation cache in the same
launch. `nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_bf16_norm_float32`
also writes the prepacked LN2 BF16 activation for the stored-MLP FC+GELU path,
which is the native dense-GPT default. Runtime JSON reports
`attention_residual_ln2_strategy`, `fused_ln2_bf16_out_enabled`,
`fused_ln2_bf16_norm_float_store_elision_enabled`,
`stored_mlp_ln2_bf16_prepack_strategy`,
`stored_mlp_ln2_bf16_fused_store_kernel_launches`,
`stored_mlp_ln2_bf16_float_store_elided_count`, and
`stored_mlp_ln2_bf16_float_store_elided_elements` for this path. The default
training route skips the redundant FP32 LN2 norm-output store when BF16 LN2 is
the only consumer; set `NFN_NATIVE_GPT_ELIDE_LN2_BF16_NORM_FLOAT_STORE=0` to
restore the older write for paired kernel bisection.

Full GPT-2 `--train-transformer-lm` also fuses attention-output and MLP
projection bias with residual addition. `nfn_native_tile_linear_bias_residual_add_float32`
consumes no-bias CUBLAS projection output, applies the projection bias and
residual scale, and writes the residual output in one Tile pass. Native plan and
training JSON report `projection_bias_residual_strategy:
"fused-linear-bias-residual-add"` and two elided legacy launches per block.

`neuralfn/csrc/native_train/token_shards.cpp` is the reusable no-Torch token-shard resolver and sequential batch sampler for native trainers. It resolves dataset aliases through `NFN_DATASETS_DIR`, validates sorted `fineweb_train_*.bin` / `fineweb_val_*.bin` uint16 shards, accepts llm.kittens-style `TinyStories_train.bin` / `TinyStories_val.bin`, infers validation siblings for direct train-bin paths, skips the 1024-byte cached-shard header when present, counts train/validation tokens, computes native microbatch plus gradient-accumulation metadata, and either produces token plus next-token target vectors for smoke/debug JSON or writes directly into caller-owned token/target buffers with `SequentialTokenBatchSampler::next_into()`. The full dense GPT trainer uses `next_into()` with pinned memory, so real token payloads avoid graph-editor nodes, Python dataset objects, `TokenBatch` vector materialization, vector-to-pinned copies, and the default uint16-token path also avoids per-microbatch int64 widening before embedding and CE kernels.

Direct legacy training scripts use the same family binary preference before graph-backed imports: `NFN_NATIVE_<MODEL>_CLI`, then `build/nfn_<model>_native_train`, then an installed `nfn_<model>_native_train`, falling back to the generic `nfn-native-train` registry only when no family binary is available. The pre-import guard normalizes wrapper-level native-cuda aliases such as `--native-cuda-print-plan`, `--native-cuda-check-tile-ops`, `--native-cuda-smoke-*`, `--native-cuda-tile-ops-lib`, `--native-cuda-cuda-runtime-lib`, `--native-cuda-no-checkpoint`, `--native-cuda-kernel-backend`, `--native-cuda-output-dir`, `--native-cuda-lm-head-row-chunk-size`, cadence fields, and activation fields to canonical C++ flags before forwarding to family binaries.

`bash tools/build_native_missing_trainers.sh` builds compiled per-family native targets for missing trainers. GPT-2 evo is a model-aware C++ native binary: `nfn_gpt2_evo_native_train --print-plan --eval-every-steps 1000 --tile-cuda-activation-dtype nvfp4` emits JSON for the dense GPT-2 shape, schedule, `adamw` profile, validation cadence, NVFP4 activation intent, evo-layer index/cadence/population, the dense GPT layer-evo delegate, available device-side mutation/selection/adoption ABI, and native CUDA forward-only candidate scoring without importing Python or Torch. Normal dense GPT-2-compatible runs exec `nfn_gpt_native_train --train-transformer-lm --layer-evo`; custom graphs and non-dense templates remain non-runnable until their native trainers exist. `nfn_gpt2_evo_native_train --smoke-evo-kernels --tile-ops-lib PATH` loads the raw evo ABI plus CUDA runtime, runs mutate/select/adopt on tiny device buffers, verifies best-candidate adoption by copyback, and exits before SDK tensors, datasets, or graph-editor nodes are involved. Direct `python cli/scripts/train_gpt2_evo.py --print-plan ...`, `--native-cuda-print-plan ...`, and `--native-cuda-dry-run ...` now prefer that family-specific binary through `NFN_NATIVE_GPT2_EVO_CLI`, `build/nfn_gpt2_evo_native_train`, or an installed `nfn_gpt2_evo_native_train` before falling back to the generic native registry, with native-cuda preflight/value aliases normalized before forwarding. The compiled GPT-2 evo binary also accepts those native-cuda aliases directly for `--native-cuda-print-plan`, `--native-cuda-smoke-evo-kernels`, `--native-cuda-tile-ops-lib`, and `--native-cuda-cuda-runtime-lib`. NanoGPT is a model-aware C++ preflight that emits `--print-plan` JSON for the native shape with the real GPT-2 tokenizer vocabulary (`50257`), schedule, AdamW profile, token-shard constraints, contiguous parameter/gradient/AdamW-state buffers, AdamW decay/no-decay groups, forward/backward/optimizer `training_step_plan`, tied LM head backward coverage through the linear ABI, kernels already exposed through the native ABI, and kernels still required for real training. `--check-tile-ops --tile-ops-lib PATH` validates that the compiled NanoGPT binary can bind all required raw Tile ABI symbols from the trainer shared library, including deterministic inverted-dropout forward/backward ABI when `--dropout-p` is nonzero. `--smoke-tile-ops --tile-ops-lib PATH` goes one step further by loading CUDA runtime, allocating a tiny device buffer, launching `nfn_native_tile_fill_float32`, copying the buffer back, and verifying the value without Python or Torch. `--smoke-optimizer-step --tile-ops-lib PATH` proves the same compiled path can build the NanoGPT parameter layout, initialize contiguous param/grad/AdamW moment buffers with raw fill kernels, execute `nfn_native_tile_adamw_step_float32` once per registered parameter buffer with that buffer's decay/no-decay setting, copy param and moment buffers back, and verify the update. `--smoke-training-loop-step --tile-ops-lib PATH` exercises native optimizer-loop mechanics over that registered layout: gradient zeroing, synthetic gradient fill, global-norm clip scale finalization, device-scalar gradient scaling, and per-buffer AdamW updates. `--smoke-lm-step --tile-ops-lib PATH` runs a tiny tied-embedding language-model step through token embedding, linear logits, token CE loss/backward, linear input/weight backward, token embedding weight backward, and AdamW update kernels, then verifies loss, gradient, and weight update values. `--smoke-token-train-step --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS` samples a real native uint16 token/target batch from cached shards, runs the tied-LM forward/backward/update kernels over those IDs, and verifies sampled-batch loss, gradient, and weight update values. `--train-token-lm --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS --max-steps N` runs a real multi-step tied token-embedding LM loop over cached shards with the C++ train and validation samplers, device-side gradient zeroing, token CE backward, tied weight updates, AdamW metrics JSON, and periodic validation losses under `validation.losses` without Python or Torch. `--smoke-embedding-norm-step --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS` runs sampled tokens through token plus absolute-position embeddings, residual add, LayerNorm forward/backward, tied logits, CE backward, embedding/position/norm gradients, and AdamW updates, then verifies residual, norm, loss, gradient, and weight update values. `--smoke-qkv-layout-step --tile-ops-lib PATH` verifies fused QKV split/merge layout kernels for the NanoGPT `attn.qkv.weight` activation and gradient path. `--smoke-fused-qkv-attention-step --tile-ops-lib PATH` runs a tiny attention stage through one fused `attn.qkv.weight`, QKV split, SDPA forward/backward, QKV gradient merge, fused qkv weight backward, output projection backward, and AdamW updates for fused qkv/output weights. `--smoke-transformer-block-step --tile-ops-lib PATH` composes LayerNorm, fused-QKV attention, residual adds, MLP, backward passes, gradient accumulation, and AdamW updates for a tiny transformer block through the raw native kernels. `--smoke-mlp-step --tile-ops-lib PATH` runs a tiny MLP stage through fc projection, GELU, output projection, input backward, GELU backward, and AdamW updates for both MLP weights, then verifies forward, gradient, and weight update values. `--smoke-attention-step --tile-ops-lib PATH` remains the separate-Q/K/V attention comparison smoke; use `--cuda-runtime-lib PATH` or `NFN_CUDA_RUNTIME_LIB` if libcudart is not discoverable by the dynamic loader. Pass `--require-token-shards` to force cached-token shard validation and include the resolved shard metadata in the JSON; add `--sample-token-batch` to include the first native token/target batch. Default NanoGPT transformer training is routed through the native dense GPT selector with `--template-name nanogpt`, which now drives the shared dense loop with NanoGPT width/head/layer geometry. The other non-dense model-family targets still report the CUDA Tile C++ kernels or trainer-loop integration required before each real trainer replaces its placeholder. The unified frontend dispatches to these binaries when present, so direct CLI and SDK paths stay on compiled native artifacts until each real trainer replaces its placeholder.

Native checkpoints written by `train_gpt2cu` are llm.kittens `.bin` files, not Torch `.pt` state dicts. Use the Torch-free checkpoint helpers to inspect them:

```python
from neuralfn.native_gpt2 import (
    latest_native_gpt2_checkpoint,
    native_gpt2_checkpoint_sampler_argv,
    read_native_gpt2_checkpoint_info,
)

checkpoint = latest_native_gpt2_checkpoint("~/NeuralFn/artifacts/gpt2")
info = read_native_gpt2_checkpoint_info(checkpoint)
print(info.to_dict())

argv = native_gpt2_checkpoint_sampler_argv(
    checkpoint,
    prompt_tokens="1,2,3",
    max_new_tokens=16,
)
print(argv)
```

The parser reads the native 256-int GPT header, reports precision, sequence length, vocabulary size, padded vocabulary size, layer/head/channel shape, parameter count, expected file size, and whether the matching `DONE_########` marker exists. For NeuralFn native GPT-2 version-5 checkpoints, `vocab_size` remains the public tokenizer vocabulary and `padded_vocab_size` is the tensor row count used for parameter/file-size accounting. `nfn infer --checkpoint path/to/model_########.bin --native-info` and `python cli/scripts/infer_gpt2.py --native-checkpoint path/to/model_########.bin --native-info` use the same no-Torch metadata path. The compiled dense GPT CLI can also inspect those checkpoints directly with `nfn_gpt_native_train --native-info --native-checkpoint path/to/model_########.bin` or `nfn_gpt_native_train --inspect-checkpoint path/to/model_########.bin`; it emits JSON and exits before CUDA, token-shard resolution, Torch, Python dataset setup, or graph-node execution. Token-ID native inference requests now route through the compiled CLI as `nfn_gpt_native_train --sample-checkpoint path/to/model_########.bin --prompt-tokens 1,2,3 --max-new-tokens 16`; `nfn infer --checkpoint path/to/model_########.bin --prompt-tokens 1,2,3` and `python cli/scripts/infer_gpt2.py --native-checkpoint path/to/model_########.bin --prompt-tokens 1,2,3` use that same path. It validates the checkpoint, context window, vocab bounds, and prompt-token parsing in C++, executes autoregressive CUDA Tile checkpoint forward passes, and returns up to `--max-new-tokens` IDs in `generated_tokens`. `nfn_gpt_native_train --checkpoint-layout --native-checkpoint path/to/model_########.bin` decodes the header-derived tensor layout, payload offsets, file offsets, and bounded payload samples without CUDA or Torch. `nfn_gpt_native_train --checkpoint-load-smoke --native-checkpoint path/to/model_########.bin --checkpoint-load-tensor h.0.ln_1.weight --checkpoint-load-elements 1024` validates the next sampler prerequisite by selecting a named tensor from the decoded layout, copying a bounded bf16 checkpoint payload slice to CUDA, converting it through `nfn_native_tile_bf16_bits_to_float32`, and checking GPU copyback without Torch or graph payloads. `nfn_gpt_native_train --checkpoint-logits-smoke --native-checkpoint path/to/model_########.bin --prompt-tokens 1,2,3` loads checkpoint embeddings and final norm tensors, converts bf16 weights on device, and runs token embedding, absolute position embedding, residual add, final LayerNorm, and tied LM-head logits for the last prompt token; transformer block execution is still pending for full generation. `nfn_gpt_native_train --checkpoint-qkv-smoke --native-checkpoint path/to/model_########.bin --prompt-tokens 1,2,3 --checkpoint-block-index 0` loads checkpoint embeddings plus the selected block's `ln_1` and `attn.c_attn` tensors, then runs embedding residual, block LayerNorm, and QKV projection through CUDA Tile kernels without Torch or graph payloads. `nfn_gpt_native_train --checkpoint-attention-smoke --native-checkpoint path/to/model_########.bin --prompt-tokens 1,2,3 --checkpoint-block-index 0` continues through split-to-heads, causal scaled-dot-product attention, and merge-heads on CUDA Tile kernels. `nfn_gpt_native_train --checkpoint-attention-residual-smoke --native-checkpoint path/to/model_########.bin --prompt-tokens 1,2,3 --checkpoint-block-index 0` loads `h.N.attn.c_proj` tensors and continues through attention output projection plus residual add. `nfn_gpt_native_train --checkpoint-block-smoke --native-checkpoint path/to/model_########.bin --prompt-tokens 1,2,3 --checkpoint-block-index 0` continues through `ln_2`, MLP fc, GELU+bias, MLP projection, and final block residual add. `nfn_gpt_native_train --checkpoint-block-logits-smoke --native-checkpoint path/to/model_########.bin --prompt-tokens 1,2,3 --checkpoint-block-index 0` continues through final norm and tied LM-head logits for the last prompt token, reporting top-token metadata while still running only the selected block. `nfn_gpt_native_train --checkpoint-forward-logits-smoke --native-checkpoint path/to/model_########.bin --prompt-tokens 1,2,3` runs every checkpoint GPT block in order, then final LayerNorm and the tied LM head for the last prompt token; it reports `transformer_blocks_executed: true`, `blocks_executed`, and `graph_editor_node_flow: false`. Text prompt generation from native `.bin` checkpoints now uses GPT-2 tokenization in the lightweight wrapper: `nfn infer --checkpoint path/to/model_########.bin --prompt "Once upon a time"` and `python cli/scripts/infer_gpt2.py --native-checkpoint path/to/model_########.bin --prompt "Once upon a time"` encode the prompt to token IDs and dispatch to the same compiled `nfn_gpt_native_train --sample-checkpoint ... --prompt-tokens ...` path. Successful wrapper calls reprint the compiled JSON and then print `Generated token ids` plus GPT-2-decoded `Generated text` without importing Torch. `NFN_NATIVE_GPT_SAMPLE_SCRIPT` and `--native-sampler-script` are deprecated for native `.bin` checkpoint prompts; graph-backed `nfn infer --graph ... --weights ...` remains for NeuralFn `.pt/.json` exports.

## Diagnostics

```python
from neuralfn.tile_cuda import tile_cuda_diagnostics

print(tile_cuda_diagnostics().to_dict())
```

Diagnostics report the `nvcc` path, CUDA version, `cuda_tile.h` availability, optional `torch.cuda` availability when Torch is installed, GPU name, compute capability, whether source builds are enabled, and whether the optional extension is already loaded. Explicit `TileCudaConfig(backend="tile_cuda")` enables extension build/load; `backend="auto"` only builds on demand when `NFN_TILE_CUDA_BUILD=1` or `build_enabled=True`.

## Implemented Kernels

The current registry accounts for all 138 training-relevant NeuralFn entries: 129 Tile-covered kernels or Tile compositions, 7 host-only interface/source entries, and 2 delegated compiled-graph calls. There are no `torch_fallback` entries in the default registry.

Scalar functions support contiguous CUDA `float32` and `float16`. The module kernels `loss_scale`, `logit_softcap`, `aux_loss_add`, `kl_penalty`, `residual_add`, `residual_mix`, `manifold_hyper_connection`, `qk_gain`, `dyt`, `dropout`, `act_weighted_sum`, `latent_pool`, `rms_norm`, `layer_norm`, `group_norm`, and `qk_norm` also support `float16` activations while retaining float32 parameters, weights, masks, reductions, and scale gradients. Verified projection-family modules also accept fp16 activations with float32 weights: `linear`, `lm_head`, `tied_lm_head`, `router_logits`, `value_head`, `reward_head`, `denoise_head`, `kv_pca_encode`, `kv_pca_decode`, `jepa_projector`, `jepa_predictor`, `ttt_linear`, `lora_linear`, `bitlinear_ternary`, `fp8_linear`, `mx_linear`, `randmap_adapter`, `mlp_relu2`, `swiglu`, `geglu`, `reglu`, `solu`, and `act_halt_gate`. Verified attention-family modules also accept fp16 activations with fp32 score, softmax, and route-weight accumulation: `rotary_embedding`, `scaled_dot_product_attention`, `sliding_window_attention`, `block_sparse_attention`, `streaming_attention_sinks`, `native_sparse_attention`, `differential_attention`, `causal_self_attention`, `fused_causal_attention`, `multi_latent_attention`, and `routed_attention_experts`. Verified loss/reduction modules accept fp16 logits or values with fp32 reductions: `token_cross_entropy`, `masked_token_cross_entropy`, `sequence_logp`, `latent_mse_loss`, `semantic_alignment_loss`, `dpo_pairwise_loss`, `ppo_clipped_loss`, `gae_compute`, `preference_bce_loss`, `load_balance_loss`, `route_balance_loss`, `route_selection_loss`, `route_distillation_loss`, and `softmax_distillation_loss`. Verified optimizer/runtime helpers accept fp16 tensors through fp32 Tile compute: `ema_update`, `gradient_accumulate`, `gradient_clip_norm`, `adamw_step`, `muon_step`, and `split_optimizer_step` with fp16 parameter/gradient buffers plus float32 optimizer state. Standalone `muon_newton_schulz` remains float32-only because it is the matrix orthogonalization primitive used inside Muon. The `float16` path uses the existing Tile `float32` kernels for compute and casts activation outputs back to `float16` only for activation-like outputs, so reductions and nonlinear math follow the float32-accumulate contract rather than native half math. Training-mode `dropout` uses deterministic counter-based masks for fp32/fp16 activations instead of the PyTorch RNG fallback.

Projection-family modules also accept CUDA `float8_e4m3fn` and `float8_e5m2` activation inputs for `linear`, `lm_head`, `tied_lm_head`, `router_logits`, `value_head`, `reward_head`, `denoise_head`, `kv_pca_encode`, `kv_pca_decode`, `jepa_projector`, `jepa_predictor`, `ttt_linear`, `lora_linear`, `bitlinear_ternary`, `fp8_linear`, `mx_linear`, `randmap_adapter`, `mlp_relu2`, `swiglu`, `geglu`, `reglu`, `solu`, and `act_halt_gate`. The fp8 projection contract dequantizes activations to float32, uses the Tile float32 linear kernel for accumulation, returns float32 outputs, and keeps weights, bias, and gradients in float32. Branching composite projections dequantize the fp8 input once before fan-out so CUDA gradient accumulation remains in float32.

Projection-family modules also accept packed `NVFP4Tensor` activation inputs for `linear`, LM/router/value/reward/denoise heads, tied LM head, KV PCA encode/decode, JEPA heads, deterministic LoRA/TTT/adapter projections, `bitlinear_ternary`, `fp8_linear`, `mx_linear`, MLP projections, and ACT halt projection. The NVFP4 contract dequantizes activations through NeuralFn block/tensor scale metadata, uses the Tile float32 path for accumulation, returns float32 outputs, and keeps weights, bias, and gradients in float32. For training parity checks, `quantize_nvfp4_reference(..., preserve_grad=True)` keeps an optional source tensor so dequantization can pass straight-through gradients back to the pre-quantized activation. `nf4_linear` remains outside this contract because its base weights already use a separate packed NF4 representation.

Attention modules also accept CUDA `float8_e4m3fn` and `float8_e5m2` Q/K/V activation inputs for `scaled_dot_product_attention`, `sliding_window_attention`, `block_sparse_attention`, `streaming_attention_sinks`, `native_sparse_attention`, `differential_attention`, `causal_self_attention`, `fused_causal_attention`, `multi_latent_attention`, and `routed_attention_experts`. The fp8 attention contract dequantizes Q/K/V to float32, keeps score and softmax accumulation in float32, returns float32 outputs, and dequantizes shared composite or routed inputs before fan-out so CUDA gradient accumulation remains in float32.

The same attention modules also accept packed `NVFP4Tensor` activation inputs. Q/K/V or shared composite attention inputs dequantize through NeuralFn NVFP4 scale metadata before Tile attention, RoPE, projection, and route-weight fan-out. Score and softmax accumulation stay fp32, outputs remain float32, and source gradients can flow through the optional `preserve_grad` NVFP4 source path.

Compiled CUDA Tile graphs can request automatic NVFP4 activation packing with `graph.torch_config["tile_cuda_activation_dtype"] = "nvfp4"`. This is a Tile activation-packing mode, not a PyTorch AMP dtype: `amp_dtype` remains an independent autocast setting and can be `bfloat16` for large GPT-style training runs. The compiled execution plan packs only activation ports for modules whose registry marks `nvfp4` as supported, so tied LM weights, targets, masks, losses, optimizer moments, host/source nodes, and graph editor metadata stay outside the packed activation path.

Scalar function kernels and simple elementwise modules also accept CUDA `float8_e4m3fn` and `float8_e5m2` activation inputs when outputs can be requantized to the same fp8 format. This covers unary, binary, and binary-pair scalar functions plus `loss_scale`, `logit_softcap`, `aux_loss_add`, `kl_penalty`, `residual_add`, `residual_mix`, `manifold_hyper_connection`, `qk_gain`, and `dyt`. These paths dequantize inputs to float32, run the existing Tile float32 kernels, and requantize activation outputs to the input fp8 format.

Unsupported lower-precision entries report category-specific reasons in `dtype_support`: losses/reductions require a loss-surface and saturation contract, optimizers require parameter/state storage semantics, stochastic mask producers require RNG and mask-storage semantics, semantic projectors keep a float32-only categorical contract because argmax-derived topic/signature outputs can change under activation quantization, integer/hash/routing outputs are not meaningful fp8/NVFP4 activation tensors, host-only source nodes are control-plane interfaces, and delegated graph calls inherit support from their child graph.

- unary: `identity`, `negate`, `sigmoid`, `relu`, `tanh_neuron`, `gaussian`, `log`, `leaky_relu`, `prelu`, `relu6`, `elu`, `selu`, `silu`, `mish`, `softplus`, `softsign`, `hard_sigmoid`, `hard_tanh`, `hard_swish`, `threshold`, `gelu`
- binary: `add`, `multiply`
- binary pair: `softmax_2`, `logsoftmax_2`

Elementwise modules:

- `logit_softcap`
- `loss_scale`
- `aux_loss_add`
- `kl_penalty`
- `residual_add`
- `residual_mix`
- `manifold_hyper_connection`
- `qk_gain`
- `dyt`
- `rms_norm`
- `layer_norm`
- `qk_norm`
- `reshape_heads`
- `merge_heads`
- `repeat_kv`
- `rotary_embedding`
- `absolute_position_embedding`
- `token_embedding`
- `expert_combine`
- `kv_cache_read`
- `kv_cache_write`
- `broadcast_expert_routes`
- `broadcast_chunk_routes`
- `byte_patch_merge`
- `latent_mse_loss`
- `linear`
- `lm_head`
- `tied_lm_head`
- `router_logits`
- `value_head`
- `reward_head`
- `denoise_head`
- `act_halt_gate`
- `act_weighted_sum`
- `latent_pool`
- `gelu`
- `mlp_relu2`
- `swiglu`
- `geglu`
- `reglu`
- `solu`
- `token_cross_entropy`
- `masked_token_cross_entropy`
- `sequence_logp`
- `scaled_dot_product_attention`
- `sliding_window_attention`
- `block_sparse_attention`
- `streaming_attention_sinks`
- `native_sparse_attention`
- `differential_attention`
- `causal_self_attention`
- `fused_causal_attention`
- `multi_latent_attention`
- `routed_attention_experts`
- `dpo_pairwise_loss`
- `preference_bce_loss`
- `load_balance_loss`
- `auxfree_load_balancing`
- `topk_route`
- `expert_dispatch`
- `route_balance_loss`
- `route_distillation_loss`
- `semantic_alignment_loss`
- `semantic_projector`
- `semantic_chunk_projector`
- `semantic_hasher`
- `semantic_chunk_hasher`
- `semantic_moe_router`
- `semantic_hash_router`
- `semantic_moe_jepa_evo_router`
- `attentionless_decoder`
- `dropout`
- `softmax_distillation_loss`
- `adamw_step`
- `ema_update`
- `gradient_accumulate`
- `gradient_clip_norm`
- `mamba`
- `mask_scheduler`
- `random_timesteps`
- `jepa_mask`
- `universal_transformer`
- `muon_newton_schulz`
- `muon_step`
- `split_optimizer_step`

Binary and binary-pair Tile scalar function kernels require same-shaped contiguous inputs. Vector module kernels require a last dimension matching the stage parameter vector, except `qk_gain`, which expects `[B, H, ...]` input with a gain vector of length `H`. Norm kernels cover contiguous float32 rows with last dimension up to 1024, plus `group_norm` on `[B,S,D]` when `S * group_dim <= 1024`. Layout and indexing kernels cover contiguous float32 `reshape_heads`, `merge_heads`, `repeat_kv`, `rotary_embedding`, `absolute_position_embedding`, `token_embedding`, `byte_patch_embed`, `causal_chunk_state` prefix/mean chunk states, and KV cache copy/concat paths. Projection kernels cover contiguous float32/fp16 projection paths plus fp8 and NVFP4 projection-family activation contracts. The NVFP4 subset covers packed activations for `linear`, LM/router/value/reward/denoise heads, tied LM head, KV PCA encode/decode, JEPA heads, deterministic LoRA/TTT/adapter projections, `bitlinear_ternary`, `fp8_linear`, `mx_linear`, MLP projections, and ACT halt projection; `nf4_linear` stays excluded because it owns a separate NF4 packed-weight contract. Attention kernels cover contiguous CUDA float32, verified fp16 activation inputs, fp8 Q/K/V activation inputs, and NVFP4 packed activation inputs for `scaled_dot_product_attention`, `sliding_window_attention`, `block_sparse_attention`, `streaming_attention_sinks`, deterministic `native_sparse_attention`, `differential_attention`, `causal_self_attention`, `fused_causal_attention`, `multi_latent_attention`, and `routed_attention_experts` with key sequence length up to 1024, causal or non-causal masking, `dropout_p=0`, grouped-query attention when query heads are divisible by key/value heads, right-aligned sparse masks for cache-compatible query/key lengths, split Q/K dimensions for differential attention, Tile-composed projection/RoPE/output paths for self-contained attention stages, and fp32 route-weight accumulation for routed experts. KV quantization kernels cover same-shaped contiguous float32 K/V rows with `head_dim <= 512`, packing quantized values plus per-row scale and unpacking tensors shaped `[..., 2*head_dim+1]`. Semantic projector kernels cover flat and chunked topic-head, signature, and residual projections while preserving the per-dimension topic logits contract. Semantic hash kernels cover contiguous float32 semantic vectors and chunk vectors with up to 62 hash planes per table, returning int64 bucket IDs without gradients. Route kernels cover contiguous float32 route weights plus int64 route indices, `topk_route` covers contiguous float32 logits with `top_k <= 64`, `semantic_hash_router` covers unforced hash/topic routing through native top-k selection while preserving the PyTorch forced-target ordering path, `semantic_moe_jepa_evo_router` covers chunk-level shared/semantic/free route-logit construction with Tile free-expert projection and PyTorch-compatible candidate ordering, `auxfree_load_balancing` covers native per-expert bias addition with device-side no-grad bias updates, supervised semantic route BCE for `route_selection_loss`, route distillation KL reduction for `route_distillation_loss` with PyTorch reference preprocessing for topic dimensions wider than 1024 terms, and `attentionless_decoder` covers bucket-conditioned expert-output logits with native bucket embedding plus output projection. `dropout` uses Tile identity for inference and `p=0` plus deterministic counter-based training masks for contiguous fp32/fp16 activations; `random_timesteps`, `mask_scheduler`, and `jepa_mask` use deterministic counter-based device random generation so CPU/GPU parity does not depend on global PyTorch RNG state. `adamw_step`, `ema_update`, `gradient_accumulate`, `gradient_clip_norm`, `muon_step`, and `split_optimizer_step` cover contiguous CUDA float32 tensors or fp16 parameter/gradient tensors with float32 optimizer state through fp32 Tile-compatible compute; AdamW fp16 support requires fp16 parameter/gradient buffers with fp32 first/second moments, and Muon fp16 support requires fp16 parameter/gradient buffers with fp32 momentum. `muon_newton_schulz` remains float32-only as the standalone matrix orthogonalization primitive. `latent_pool` covers masked JEPA latent pooling with mean fallback for empty masks, and `token_cross_entropy`, `masked_token_cross_entropy`, `sequence_logp`, `latent_mse_loss`, `semantic_alignment_loss`, `dpo_pairwise_loss`, `ppo_clipped_loss`, `gae_compute`, `preference_bce_loss`, `load_balance_loss`, `route_balance_loss`, `route_distillation_loss`, and `softmax_distillation_loss` produce scalar losses or log-prob reductions through Tile reductions. Dense GPT native runtime JSON now reports the LM-head CE BF16 launch shape (`lm_head_ce_bf16_threads_per_row`), vector IO strategy (`lm_head_ce_bf16_vector_io_strategy`), and the underlying vec-load/store flags, so SDK and CLI callers can verify whether a paired classifier candidate changed row-block size, scalar stores, vec8 normal stores, or vec8 streaming stores. The raw Tile ABI exposes the resolved row-block size through `nfn_native_tile_token_cross_entropy_bf16_threads_per_row`; invalid thread-count environment values fall back to the 1024-thread default before this value is reported. DPO reward outputs remain detached to match the PyTorch stage contract. Non-CUDA tensors, unsupported dtypes, non-contiguous tensors, broadcasted inputs outside these contracts, and unsupported runtime contracts fall back to PyTorch unless strict mode is enabled.

When `NFN_NATIVE_GPT_CE_BF16_VEC_STORES=1` or
`NFN_TILE_CUDA_CE_BF16_VEC_STORES=1` is enabled for dense GPT bisection, the
BF16 CE dlogit write pass reuses packed vec8 BF16 loads before writing vec8
streaming stores. The switch is still default-off because the 2026-06-22
dedicated RTX 5090 same-script gate rejected the route on train-loop,
LM-head-backward, and CE timing.

FP8 and NVFP4 expansion is tracked in `todo-tile-cuda.md`. Existing `fp8_linear`, `mx_linear`, and `nf4_linear` modules preserve their current quantized-weight semantics. General fp8 support currently covers scalar functions, simple elementwise modules, projection-family activation contracts, and attention Q/K/V activation contracts. NVFP4 support currently covers packed projection-family and attention-family activation inputs.

## Low-Precision Reference Helpers

The SDK exports deterministic CPU/Torch reference helpers for future dtype kernels:

```python
from neuralfn.tile_cuda import (
    quantize_dequantize_fp8_reference,
    quantize_nvfp4_reference,
    dequantize_nvfp4_reference,
)

x_fp8 = quantize_dequantize_fp8_reference(x, "float8_e4m3fn")
encoded = quantize_nvfp4_reference(x, preserve_grad=True)
x_nvfp4 = dequantize_nvfp4_reference(encoded)
```

`quantize_fp8_reference()` supports `float8_e4m3fn` and `float8_e5m2` using PyTorch's fp8 dtypes, and `dequantize_fp8_reference()` returns float32 values for parity checks. `NVFP4Tensor` stores packed uint8 nibbles, FP8 E4M3 block scales, an FP32 tensor scale, original shape metadata, `block_size=16`, and an optional `source` tensor. `quantize_nvfp4_reference(..., preserve_grad=True)` records that source tensor and `dequantize_nvfp4_reference()` uses a straight-through estimator so gradients flow to the pre-quantized activation while forward values remain deterministic NVFP4 dequantized values. The NVFP4 reference uses deterministic round-to-nearest against an FP4 E2M1 codebook and is intended for kernel validation; projection-family modules listed above currently accept packed NVFP4 activations.

```python
from neuralfn.tile_cuda import TileCudaConfig, build_tile_function_module

stage = build_tile_function_module("add", TileCudaConfig(backend="tile_cuda"))
out = stage(x, y)
```

```python
from neuralfn.tile_cuda import TileCudaConfig, build_tile_module

stage = build_tile_module("residual_add", {"dim": hidden_dim}, TileCudaConfig(backend="tile_cuda"))
out = stage(residual, delta)
```

## Coverage Report

```python
from neuralfn.tile_cuda import coverage_report

report = coverage_report()
assert report.complete
print(report.to_dict())
```

The coverage report is generated from the live NeuralFn builtin and torch-backend dispatch surfaces:

- `BuiltinNeurons.all()`
- `build_module()`
- `build_function_module()`
- optimizer/runtime targets used by `TorchTrainer`

Each `TileKernelSpec` keeps the legacy `dtypes` tuple and also exposes `dtype_support`, a matrix for `float32`, `float16`, `float8_e4m3fn`, `float8_e5m2`, and `nvfp4`. Supported entries are marked `"supported"`; unsupported entries explain the missing scale, representation, accumulation, stochastic-mask, matrix-state, or parity contract. `KernelCoverageReport.by_dtype` aggregates supported and unsupported counts for the same tracked dtype set.

Every entry must be accounted for as one of:

- `tile`: implemented CUDA Tile kernel
- `torch_fallback`: not yet implemented in Tile; PyTorch remains authoritative. The default registry currently has none.
- `host_only`: source or interface node with no device compute contract
- `delegated`: covered by another compiled graph or fused implementation
- `planned`: reserved for future work with an explicit reason

## Training Hot Path

Real training tensors must not pass through graph editor nodes. `CompiledTorchGraph` compiles graph topology and edge routing once, then forwards tensors through fixed modules and precomputed routing plans. Future CUDA Tile graph execution must preserve the same invariant.

`CompiledTorchGraph(..., kernel_backend="tile_cuda", tile_cuda_strict=True)` validates coverage before batches run. Any node still marked `torch_fallback` or `planned` fails at compile time in strict mode. Scalar function and simple module dtype-contract failures include the rejected tensor dtype, supported dtype set, contiguity, device, and shape.

## Examples

Checked-in examples live under `examples/tile_cuda/`. Use the CLI to list or regenerate them:

```bash
nfn kernels examples
nfn kernels examples --write --output-dir examples/tile_cuda
nfn kernels bench --device auto --iterations 200
```
