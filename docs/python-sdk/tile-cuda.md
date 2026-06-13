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

CUDA Tile native builds require CUDA Toolkit 13.3 or newer, `cuda_tile.h`, C++20, `nvcc --enable-tile`, and `ninja`. Torch is not a default SDK dependency, and the CUDA Tile extra no longer installs it. Install the native build extra with:

```bash
pip install -e ".[tile-cuda]"
```

Install `pip install -e ".[torch]"` separately only for graph-backed PyTorch execution or the legacy PyTorch Tile extension loader.

The trainer-facing raw C ABI build is separate:

```bash
bash tools/build_native_train_tile_ops.sh
```

On the SM120 workstation this defaults to `NFN_TILE_CUDA_USE_TK_ATTENTION=1`,
`NFN_TILE_CUDA_ARCH=sm_120a`, and links the local llm.kittens /
ThunderKittens headers via `LLM_KITTENS_ROOT` and `TK_ROOT`. Set
`NFN_TILE_CUDA_USE_TK_ATTENTION=0` only when you intentionally want the older
float32 row-scan attention diagnostic build.

## Native GPT-2 Trainer Handoff

The plain GPT-2 script can bypass the graph-backed `TorchTrainer` and hand cached token shards directly to a compiled C++/CUDA trainer:

```python
from pathlib import Path
from neuralfn.native_gpt2 import build_native_gpt2_compiled_cli_run_config, build_native_gpt2_run_config, run_native_gpt2

config, updated_meta = build_native_gpt2_run_config(
    dataset_name="roneneldan__TinyStories__TinyStoriesV2-GPT4",
    dataset_path=Path("~/.cache/nfn/datasets/roneneldan__TinyStories__TinyStoriesV2-GPT4").expanduser(),
    dataset_meta={},
    encoding_name="gpt2",
    executable="/mnt/disk2/dev/open-source/llm.kittens/train_gpt2cu",
    output_dir=Path("~/NeuralFn/artifacts/gpt2").expanduser(),
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
)

print(config.command())
run_native_gpt2(config)
```

`build_native_gpt2_run_config()` resolves `fineweb_train_000000.bin` plus `fineweb_val_000000.bin` directly when the dataset is already cached as matching uint16 shards. That fast path avoids importing `server.dataset_manager`, NumPy, tiktoken, or Torch and does not scan the full dataset to estimate a schedule before launch. `build_native_gpt2_compiled_cli_run_config()` is even lighter for the no-Python cached-shard CLI: it accepts a dataset alias/path and leaves shard metadata validation to the compiled C++ resolver, so Python does not need `meta.json` before handoff. The compiled resolver also accepts llm.kittens-style `TinyStories_train.bin` / `TinyStories_val.bin`; `--tinystories` resolves to `/mnt/disk2/dev/open-source/llm.kittens/dev/data/tinystories` when present, `NFN_LLM_KITTENS_TINYSTORIES_DIR` overrides that directory, and a direct `TinyStories_train.bin` path infers the sibling validation file. Raw-text datasets still lazy-load the dataset manager only when the uint16 cache must be materialized. Tokenizers whose ids do not fit in `uint16`, such as `o200k_base`, are rejected for this path. The exported `NativeGpt2RunConfig.argv()` mirrors the SM120 `train_gpt2cu` CLI (`-i/-j/-o/-v/-b/-t/-d/-l/-q/-u/-e/-af/-x`) for explicit external bridge runs, while `NativeGpt2RunConfig.launcher_argv()` wraps that command through the workstation C++ launcher and `NativeGpt2RunConfig.compiled_cli_argv()` wraps the default Tile-CUDA training command through the no-Python cached-shard CLI. Build the in-process SDK binding with `bash tools/build_native_gpt2_binding.sh`, the launcher with `bash tools/build_native_gpt2_launcher.sh`, the no-Python cached-shard CLI with `bash tools/build_native_gpt2_cli.sh`, and the unified no-Python frontend with `bash tools/build_native_train_cli.sh`; the compiled GPT-2 CLI links the shared C++ `token_shards.cpp` resolver for cached-shard validation. `tools/install_native_gpt2_commands.sh` links `nfn-gpt2-native`, `nfn-gpt2-native-train`, `nfn-native-train`, `nfn-gpt2-tile-launcher`, and both underscore/hyphen names for built per-family native trainer entrypoints into the active Python scripts directory. `NFN_DATASETS_DIR` overrides the native alias cache root, and `nfn-native-train --base-model gpt2 --dataset-alias PATH_OR_ALIAS` or `nfn-gpt2-native --dataset-alias PATH_OR_ALIAS` bypasses Python entirely when shards already exist. The shared C++ token sampler reads contiguous shard segments for each batch instead of opening a shard per sequence chunk; token-shard JSON reports `batch_read_strategy: "contiguous_shard_segments"`. `nfn-native-train --list-models --json` reports the compiled native training registry; dense GPT-2 defaults to the NeuralFn-owned 12-layer Tile-CUDA `--train-transformer-lm` loop and its JSON `block_state_layout` exposes block-vector allocation/init/zero/clip/AdamW/checkpoint/tape/forward/backward loop flags, NanoGPT reports `partial-native-trainer` for `--train-token-lm`, and LLaMA, GPT-2 evo, JEPA, semantic/MoE, and DeepSeek entries intentionally report missing or preflight-only native trainers. NanoGPT has a partial native C++ trainer: `nfn_nanogpt_native_train --print-plan` emits a JSON training contract for shape, schedule, AdamW settings, token-shard constraints, contiguous parameter/gradient/AdamW-state buffers, AdamW decay/no-decay groups, forward/backward/optimizer `training_step_plan`, and required CUDA Tile kernels without importing Python or Torch. `nfn_nanogpt_native_train --train-token-lm --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS --max-steps N` runs the tied token-embedding LM as a real multi-step native loop over cached shards and records validation losses from validation shards according to `--eval-every-steps`, `--eval-batches`, and `--eval-batch-size`, while full NanoGPT transformer training still needs model-wide loop integration. `nfn_nanogpt_native_train --check-tile-ops --tile-ops-lib PATH` loads `libnfn_native_train_tile_ops.so` with `dlopen` and verifies every NanoGPT-required raw ABI symbol from the compiled binary. Tied LM head input/weight backward is represented through the raw linear backward ABI in that plan. It defaults to `dropout_p=0.0`; nonzero `--dropout-p` reports the missing dropout ABI as required work. `NFN_NATIVE_GPT2_BIN_DIR` overrides where command symlinks are installed, `NFN_NATIVE_TRAIN_CLI` overrides the unified frontend path, `NFN_NATIVE_<MODEL>_CLI` overrides a per-family native trainer such as `NFN_NATIVE_NANOGPT_CLI`, `NFN_NATIVE_GPT2_CLI` overrides the compiled CLI path, `NFN_NATIVE_GPT2_LAUNCHER` overrides the launcher path, and `NFN_NATIVE_GPT2_TRAIN_BIN` overrides the external trainer target used only by explicit `llm-kittens` runs. `run_native_gpt2()` sets `CUDA_DEVICE_MAX_CONNECTIONS=1` by default before launching native code. The top-level `neuralfn` package and native cached-shard path avoid importing Torch; `cli/scripts/train_gpt2.py` is now a native-only entrypoint whose import, parser construction, default resolution, and default direct execution avoid graph-backed training code; direct execution with the default `compiled-cli` runner reaches the compiled C++ CLI before importing `train_gpt2_native.py`. Default GPT-2 `nfn train` commands hand off to the compiled Tile-CUDA frontend before graph-backed Python imports. The CLI wrapper defaults to `--native-cuda-runner compiled-cli`; call `run_native_gpt2(..., runner="auto")` or pass another runner explicitly when you want SDK binding or subprocess fallback behavior.

`NativeGpt2RunConfig.lm_head_row_chunk_size` defaults to 8192 and forwards
`--lm-head-row-chunk-size` through `compiled_cli_argv()`. The C++ transformer-LM
loop uses that bounded full-vocab tied LM-head workspace and reduces CE loss
partials on device with `nfn_native_tile_sum_partials_float32`, so training and
validation loss copy one device scalar to the host instead of copying once per
row chunk. Tied LM-head dWeight chunks accumulate directly into the optimizer-step
`accum_grad_token_weight` buffer with
`nfn_native_tile_linear_backward_weight_accumulate_float32` instead of using a
full-vocab scratch gradient buffer per chunk or per microbatch. The JSON reports
`lm_head_row_chunk_count` and `loss_partial_count`.

GPT-2-compatible causal SDPA now uses the SM120 ThunderKittens bf16
FlashAttention-style bridge by default in `tools/build_native_train_tile_ops.sh`.
The bridge accepts NeuralFn's float32 Q/K/V ABI, converts supported causal
`[B,H,T,D]` heads to bf16, runs the llm.kittens SM120 attention forward/backward
tiles, and converts gradients back to the float32 optimizer buffers. It is used
only for equal Q/K heads, `D in {64, 128}`, equal query/key sequence lengths,
sequence lengths divisible by the SM120 forward tile, and dense left-aligned
causal masks. Training JSON reports
`attention_backend_strategy: "tk-sm120-bf16-bridge"`,
`attention_forward_strategy: "tk-sm120-bf16-flashattention-bridge"`,
`attention_backward_strategy: "tk-sm120-bf16-recompute-forward-bridge"`,
`attention_forward_tk_launch_count`, and `attention_backward_tk_launch_count`.

Trainer-facing linear GEMMs use the same native ABI but the full GPT-2 trainer
uses a targeted split: transformer block forward/recompute projections call
`nfn_native_tile_linear_bf16_float32`, and transformer block dInput GEMMs call
`nfn_native_tile_linear_backward_input_bf16_float32`, to force the cached BF16
workspace plus `cublasGemmEx` bridge where the stable weight operand can be
cached. Transformer block dWeight accumulation calls
`nfn_native_tile_linear_backward_weight_accumulate_bf16_float32`; tied LM-head
logits, dHidden, and dWeight chunks stay on optimized TF32 tensor-op
`cublasSgemm`. Set `NFN_TILE_CUDA_LINEAR_BF16=1` or
`NFN_NATIVE_LINEAR_BF16=1` only when profiling the normal linear ABI's BF16
bridge. The BF16 bridge keeps a multi-entry packed first-GEMM-operand cache for
weight-forward and weight-dInput calls, then invalidates that cache after AdamW
updates. GPT-2 training JSON reports `linear_backend_strategy:
"block-forward-dinput-dweight-bf16-lm-head-tf32"`,
`block_forward_linear_strategy`, `block_backward_input_linear_strategy`,
`block_backward_weight_linear_strategy`,
`non_block_forward_backward_linear_strategy`, `linear_bf16_gemm_count`,
`linear_sgemm_count`, `linear_bf16_a_pack_count`,
`linear_bf16_a_cache_hit_count`, `linear_bf16_cache_reset_count`,
`linear_bf16_cached_a_capacity`, and `linear_bf16_cache_entry_count`.
The full GPT-2 transformer-LM trainer also exposes
`nfn_native_tile_token_cross_entropy_backward_inplace_with_workspace_float32`
for tied LM-head CE backward. That ABI overwrites the logits chunk with dlogits,
so the main trainer reports `grad_logit_workspace_elements: 0`,
`lm_head_ce_backward_strategy: "inplace-logits-dlogits-workspace"`, and
`lm_head_grad_logits_workspace_allocated: false` instead of allocating a
separate full-vocab `grad_logits` chunk.

The older float32 row-vector forward and query-row atomic backward kernels stay
compiled as a diagnostic/fallback path for unsupported attention shapes or
`NFN_TILE_CUDA_USE_TK_ATTENTION=0` builds. In that mode native JSON reports
`attention_backend_strategy: "tile-row-float32"`, row/scalar launch counters,
row-kernel attribute fields, pre-launch error codes, launch grid/block fields,
and the score-reuse factors.

The compiled GPT-2 trainer also reports host wall-clock timing under `timing`:
`setup_wall_ms`, `train_loop_wall_ms`, `validation_wall_ms`,
`train_compute_wall_ms`, `checkpoint_wall_ms`, `total_wall_ms`,
`optimizer_steps_per_second`, and `train_tokens_per_second`. The timers do not
add device synchronizations; the train-loop measurement ends after the existing
final sample copy from device to host. Set `NFN_NATIVE_GPT2_STAGE_TIMING=1` to
add a CUDA-event profiler for the native transformer-LM loop. That diagnostic
mode records `stage_timing_enabled`, `stage_timing_event_count`,
`stage_timing_dropped_event_count`, and `stage_timing` entries with per-stage
`total_ms`, `count`, and `avg_ms` values for token upload, model forward, block
forward/recompute/backward, LM-head backward, final-norm/embedding backward,
gradient zero/clip, and AdamW update. Diagnostic runs also emit nested entries
for LM-head logits/CE/dHidden/dWeight, block forward/recompute attention and
MLP phases, and block backward MLP projection, MLP fc, LayerNorm/residual,
attention projection, attention SDPA, and QKV phases. The block backward
records include individual dWeight, bias, dInput, activation, residual-add, and
attention-to-QKV entries such as `block_backward.mlp_proj.dweight`,
`block_backward.mlp_proj.dinput`, `block_backward.attn_sdpa.to_qkv`, and
`block_backward.qkv.dweight`. The stage profiler
synchronizes before reading event timings, so leave it disabled for normal
throughput runs.

Native GPT-2 SDK config builders accept `template_name` and `graph_file`, which
map to canonical compiled CLI `--template-name` and `--graph-file` arguments;
Python CLI aliases such as `--template`, `--preset`, and `--graph` are
normalized before handoff. Every shipped GPT template name can be passed through
this no-Torch selection path, and the compiled C++ plan JSON reports
`shipped_template_catalog`, `shipped_template_catalog_count`, and
`template_known` so SDK callers can audit the no-Python selector catalog. Dense
GPT-2-compatible presets (`gpt2`, `gpt2_megakernel`, and `gpt2_moa`) map to the
implemented native transformer-LM loop; `gpt2_moa` resolves to
`--native-cuda-activation moa` automatically. Structurally different shipped
template names and custom graph files are selected and reported in JSON, but
return `selected-graph-native-trainer-missing` for real training until their
native C++ Tile trainer plans are implemented. Unknown template names return
`unknown-template`.

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
device as one contiguous uint16 arena with `cudaMemcpyAsync`, and widened to the
existing int64 token buffers by one `nfn_native_tile_uint16_to_int64` launch.
Native JSON reports
`token_id_upload_strategy: "uint16-pinned-async-h2d-device-widen"`,
`token_id_host_staging: "pinned"`, `token_id_h2d_copy:
"cudaMemcpyAsync-contiguous-arena"`, `token_id_h2d_copy_calls_per_microbatch:
1`, `token_id_widen_strategy: "single-contiguous-arena-kernel"`,
`token_id_widen_kernel_launches_per_microbatch: 1`, and
`token_batch_staging_strategy: "direct-sampler-to-pinned-arena"`,
`token_batch_vector_materialization: false`, and `token_id_host_validation:
false`; batch validation belongs at shard creation or a future device-side
validation pass, not in the per-step CPU hot path.

The same native trainer initializes the tied token embedding/LM-head weight on
device with `nfn_native_tile_init_gpt2_token_weight_float32`. Native JSON reports
`token_weight_init_strategy: "device-tile-deterministic"` and
`token_weight_host_materialization: false`, so startup no longer constructs and
copies the full token-weight matrix through host RAM.

The compiled GPT-2 transformer-LM trainer does not sample train loss in the hot
path. Ordinary optimizer steps run the forward activations needed for backward,
CE gradient generation, gradient clipping, and AdamW only; validation cadence
computes validation loss from validation shards without also measuring train
loss. The output fields `train_loss_sparse: false`,
`train_loss_sampling: "disabled"`, `train_loss_on_validation_steps: false`,
`train_loss_eval_count`, and `train_loss_last_step` describe that contract.

Persistent block-output preservation uses `nfn_native_tile_copy_float32` instead
of a zero-fill plus accumulate-by-one pair, removing one Tile launch per block
output copy while preserving the scratch-recompute activation tape layout.
The final block output copy is elided because final LayerNorm consumes it before
backward recomputation starts; the default 12-layer run reports
`persistent_block_outputs: 11` and `final_block_output_copy_elided: true`.
Validation forwards stream through the scratch tape without copying block
outputs into persistent training-backward buffers, because no backward pass
follows validation. JSON reports `validation_persistent_block_outputs: 0` and
`validation_block_output_copies_elided: true`.
The backward pass reuses the final block activations that remain in the scratch
tape after the initial forward pass, so only the earlier blocks are recomputed;
the default JSON reports `backward_recompute_blocks: 11` and
`final_block_backward_recompute_elided: true`. Earlier-block recompute stops
after the MLP GELU activation because backward does not consume the recomputed
MLP projection output or final residual output; JSON reports
`backward_recompute_mlp_projection_elided: true` and
`backward_recompute_final_residual_elided: true`.
The MLP projection backward path writes its dInput into the MLP fc gradient
buffer and runs `nfn_native_tile_gelu_backward_inplace_float32`, so the full
trainer does not allocate a separate hidden-size `grad_act` scratch buffer.
JSON reports `mlp_proj_backward_gelu_inplace: true` and
`mlp_proj_backward_grad_act_scratch_allocated: false`.
Backward residual-gradient pair additions use
`nfn_native_tile_scaled_residual_add_float32` instead of zero-fill plus two
gradient-accumulate launches; `block_state_layout.residual_backward_fused`
reports this path.
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
`adamw_per_buffer_step_launches_elided`.
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
`per_block_gradient_buffers: 0`,
`per_block_direct_accum_gradient_buffers: 12`,
`gradient_accumulation_loop: false`,
`gradient_accumulation_copy_loop_elided: true`,
`gradient_zero_strategy: "fused-multi-buffer-accumulation-zero"`, and
`gradient_zeroed_buffer_count: 0` under `block_state_layout`. Accumulation
buffers are zeroed once per optimizer step through
`nfn_native_tile_fill_many_float32` over the same descriptor table used by the
fused AdamW call, so the default 12-layer trainer emits one zero-fill kernel
launch instead of one launch per accumulation buffer. JSON reports
`gradient_zero_kernel_launches_per_optimizer_step` and
`gradient_zero_per_buffer_launches_elided`.
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
`float_allocation_cuda_malloc_count`, `float_allocation_request_count`,
`float_arena_requested_elements`, and `float_arena_allocated_elements`.
Startup zeroes that float arena once, leaves zero biases and AdamW state at
their arena-zero values, and overwrites nonzero weights through device
initializers. Do not re-add per-buffer zero-fill launches for those tensors.
JSON reports `float_arena_zero_init_strategy: "single-arena-fill"`,
`float_arena_zero_fill_count`, `startup_per_buffer_zero_fill_elided`, and
`startup_per_buffer_zero_fill_launches_elided`; the default 12-layer shape
elides 369 per-buffer zero-fill launches.
Nonzero constant parameter initialization uses
`nfn_native_tile_fill_many_values_float32` over a device descriptor table, so
position weights, final norm, residual scale, and all block constant weights are
filled with one Tile launch instead of one launch per tensor. JSON reports
`parameter_initialization_strategy: "fused-multi-buffer-fill-values"`,
`parameter_initialization_kernel_launches_per_startup`,
`parameter_initialization_per_buffer_launches_elided`, and
`block_state_layout.parameter_initialization_loop: false`; the default
12-layer shape elides 74 per-buffer nonzero fill launches.
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
single-block loop over every row and the scratch-copy pass. JSON reports
`layer_norm_backward_affine_strategy: "auto-chunked-atomic-accumulate"` under
`block_state_layout`.

Wrapper-level `--native-cuda-dry-run --native-cuda-print-command` is metadata-only on the default `compiled-cli` runner: Python builds the compiled C++ argv from the dataset alias/path and leaves shard validation or raw-text rejection to the compiled frontend. It must not import `server.dataset_manager`, NumPy, tiktoken, or Torch, and it must not write `fineweb_train_*.bin` shards.

Dense GPT-2 native `--dry-run` / `--print-plan` JSON reports the implemented
compiled trainer as `native-transformer-lm-ready` with
`training_step_plan.status: "ready"`. SDK callers should treat
`remaining_validation` as the live SM120 benchmark gate; unsupported template
names and custom graph files still report `selected-graph-native-trainer-missing`
instead of falling back to Torch.

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
`kernel_backend="llm-kittens"` remains available only as an explicit external
bridge. The Tile plan includes a GPT-2 parameter layout plus forward, backward,
and optimizer stage sequence for the current 12-layer trainer.

`run_native_gpt2(config, runner="auto")` now has an explicit runner boundary:

- `auto` prefers an in-process C++ binding module named `neuralfn_native_gpt2` or `neuralfn._native_gpt2`, then the compiled no-Python CLI, then the compiled launcher, then direct subprocess execution. Alias-only configs created by `build_native_gpt2_compiled_cli_run_config()` still execute the compiled CLI argv through that binding, so SDK auto mode keeps shard resolution in C++ instead of attempting raw `train_gpt2cu` with empty `-i` / `-j` values.
- `binding` requires that C++ binding and raises if it is unavailable.
- `compiled-cli` requires the compiled `nfn_gpt2_native_train` binary and raises if it is unavailable. The old `cli` spelling is no longer accepted.
- `launcher` requires the compiled `nfn_gpt2_tile_train` launcher and raises if it is unavailable.
- `subprocess` always launches the external `train_gpt2cu` executable.

`native_gpt2_runner_status()` returns the resolved mode and diagnostic reason, and `write_native_gpt2_run_config()` includes that status in the JSON payload. Set `NFN_NATIVE_GPT2_BINDING=0` to test launcher/subprocess fallback paths even when `neuralfn._native_gpt2` is built locally.

The unified native frontend has a separate SDK wrapper in `neuralfn.native_train`. Build its C++ extension with `bash tools/build_native_train_binding.sh`; `run_native_train(build_native_train_run_config("gpt2", ["--tinystories"]), runner="auto")` then prefers `neuralfn._native_train` before falling back to the compiled `nfn_native_train` CLI. `native_train_model_registry()` returns the JSON coverage from `nfn-native-train --list-models --json`, and `NFN_NATIVE_TRAIN_BINDING=0` disables the extension for fallback tests.

Dense GPT-2 has a compiled Tile CUDA preflight in `nfn_gpt2_native_train`. Use strict backend names only: `--backend llm-kittens` for the external fast trainer or `--backend tile-cuda` / SDK `kernel_backend="tile-cuda"` for the NeuralFn-owned raw Tile ABI path. No-data preflight actions (`--check-tile-ops`, `--smoke-tile-ops`, `--smoke-optimizer-step`, `--smoke-lm-step`, `--smoke-attention-step`, `--smoke-mlp-step`, `--smoke-norm-residual-step`, and `--smoke-transformer-block-step`) run before token-shard resolution, so SDK callers can validate symbols or synthetic CUDA slices without a cached dataset; plan/smoke JSON reports `token_shards_resolved: false` when shards were skipped. `--smoke-tile-ops --tile-ops-lib PATH` loads the trainer Tile ops shared library, loads CUDA runtime, launches `nfn_native_tile_fill_float32`, copies the device buffer back, and reports JSON without Python or Torch. `--smoke-optimizer-step --tile-ops-lib PATH` allocates GPT-2-sized contiguous parameter, gradient, and AdamW moment buffers, runs one AdamW call per registered GPT-2 parameter buffer with decay/no-decay metadata, samples copyback values, and reports JSON. `--smoke-lm-step --tile-ops-lib PATH` runs a tiny GPT-2-shaped tied embedding/LM-head slice through token embedding, linear logits, full-vocab CE partials and workspace CE backward, linear input/weight backward, token embedding weight backward, and AdamW. `--smoke-embedding-lm-step --tile-ops-lib PATH` samples a tiny cached uint16 token batch in C++ and runs token embedding, absolute position embedding, embedding residual add, final LayerNorm, tied LM head, CE backward, embedding/norm backward, and AdamW without graph-editor payloads. `--train-embedding-lm --tile-ops-lib PATH` runs that GPT-2 embedding/final-norm/LM path as a real multi-step compiled loop over cached train shards, with validation losses from validation shards controlled by `--eval-every-steps`, `--eval-batches`, and `--eval-batch-size`. `--smoke-attention-step --tile-ops-lib PATH` runs a tiny GPT-2 model-dim attention stage through qkv projection, QKV split, SDPA forward/backward, QKV gradient merge, projection backward, and AdamW. `--smoke-mlp-step --tile-ops-lib PATH` runs a tiny GPT-2 MLP stage through c_fc projection, GELU forward/backward, c_proj projection backward, and AdamW. `--smoke-norm-residual-step --tile-ops-lib PATH` runs GPT-2 LayerNorm, scaled residual add, LayerNorm affine/input backward, gradient accumulation, and AdamW through raw Tile kernels. `--smoke-transformer-block-step --tile-ops-lib PATH` composes GPT-2 LayerNorm, fused QKV attention, real 12-head reshape/merge layout (`12 x 64`), residual adds, MLP, backward passes, gradient accumulation, projection bias gradients, and AdamW updates for all 12 GPT-2 block parameter buffers through raw Tile kernels. Dataset-backed smokes such as `--smoke-embedding-lm-step` and `--smoke-transformer-lm-step`, plus real training modes, still resolve cached train/validation shards before running. `--smoke-transformer-lm-step --tile-ops-lib PATH` samples cached uint16 tokens, preserves range-checked GPT-2 token IDs, and runs token/position embeddings, one tiny transformer block, final LayerNorm, tied LM head, CE forward/backward, transformer backward, embedding backward, and AdamW for 16 parameter buffers through raw Tile kernels. `--train-transformer-lm --tile-ops-lib PATH` runs that transformer-LM path as a full-vocab real-dim 12-layer multi-step compiled loop over cached train shards, with periodic validation records in `validation.losses` controlled by `--eval-every-steps` and `--eval-batches`. It uses token/position embeddings, transformer blocks, final norm, a row-chunked tied LM-head/CE workspace, transformer backward, embedding backward, device-side global norm gradient clipping, and AdamW parameter updates without Python/Torch; its CE backward path reuses the logits chunk as dlogits and its JSON reports `trained_layers: 12`, `target_layers: 12`, `vocab: 50257`, `lm_head_row_chunk_size`, `logit_workspace_elements`, `grad_logit_workspace_elements`, `lm_head_ce_backward_strategy`, `lm_head_grad_logits_workspace_allocated`, `gradient_partial_count`, `gradient_clip_norm`, `sample_gradient_clip_scale`, and `block_state_layout` loop flags for scratch-recompute activation tape, forward blocks, and backward blocks when steps complete. `--checkpoint-metadata-smoke --output-dir PATH` writes a sparse version-5 bf16 native GPT-2 checkpoint-format file plus `DONE_########` marker for the requested `--num-layers` target shape, so `read_native_gpt2_checkpoint_info()` and native inference metadata can validate NeuralFn-owned artifacts without CUDA. Successful `--train-transformer-lm` runs write a final 12-layer trained-weight native checkpoint plus `DONE_########` marker. SDK callers can set `NativeGpt2RunConfig(smoke_tile_ops=True, smoke_optimizer_step=True, smoke_lm_step=True, smoke_embedding_lm_step=True, train_embedding_lm=True, train_transformer_lm=True, checkpoint_metadata_smoke=True, smoke_attention_step=True, smoke_mlp_step=True, smoke_norm_residual_step=True, smoke_transformer_block_step=True, smoke_transformer_lm_step=True, tile_ops_lib=..., cuda_runtime_lib=...)` or use `--cuda-runtime-lib PATH` / `NFN_CUDA_RUNTIME_LIB` when libcudart needs an explicit path.

Full GPT-2 `--train-transformer-lm` runs report a `cuda_runtime_preflight` object before allocation. If `cudaDriverGetVersion` returns driver version `0`, or if the loaded CUDA runtime is newer than the reported driver, the trainer exits before `cudaMalloc` so benchmark failures point at GPU access/runtime compatibility instead of kernel execution.

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

`bash tools/build_native_train_tile_ops.sh` builds `libnfn_native_train_tile_ops.so`, a raw C ABI over CUDA Tile kernels from `neuralfn/csrc/tile_cuda/kernels.cu`. Native C++ trainers should link this library for single-buffer and multi-buffer fill/zeroing, single-buffer and multi-buffer sumsq partials, single-buffer and multi-buffer AdamW, gradient accumulation, deterministic GPT-2 token-weight initialization, device float32-to-bf16 checkpoint payload packing, device-side global-norm clip scale finalization, device-scalar gradient scaling, reductions, linear, forced-BF16 linear, linear input/forced-BF16 input/weight/weight-accumulate/forced-BF16 weight-accumulate/bias/bias-accumulate backward, scaled residual add, fused projection bias+residual add, fused QKV split/merge, fused GPT-2 QKV split-to-heads, fused GPT-2 QKV bias+split-to-heads, fused GPT-2 heads-to-QKV gradient merge, fused TK bf16 attention-gradient heads-to-QKV bridge, reshape-heads/merge-heads, GELU forward, fused bias+GELU forward, GELU backward, token embedding forward/weight backward, absolute-position embedding forward/backward/backward-accumulate, RMSNorm, RMSNorm input backward, LayerNorm, LayerNorm input/affine/affine-accumulate backward, softmax, token and masked token cross-entropy partials, token and masked token cross-entropy logits backward, and scaled dot-product attention forward/backward instead of importing the PyTorch extension binding. The trainer build defines `NFN_TILE_CUDA_USE_CUBLAS_LINEAR=1` and links `libcublas`, so the exported native linear forward, dInput, dWeight, and accumulate-dWeight ABI symbols use GPU GEMM, while bias and accumulate-bias backward use GPU GEMV over a cached device ones vector initialized by a Tile fill kernel. The generic Tile extension build keeps the pure Tile fallback. CE logits backward uses a row-wise Tile path for vocabularies up to 1024 and a chunked row-wise path with reusable row-stat workspace for full GPT-class vocabularies. Linear weight, accumulate-weight, bias, and accumulate-bias backward keep the row-chunked tiled atomic fallback for builds or shapes that do not use the trainer cuBLAS path.

Full GPT-2 `--train-transformer-lm` fuses both attention QKV layout directions. It uses `nfn_native_tile_split_qkv_to_heads_add_bias_float32` after a no-bias QKV projection, applying Q/K/V bias while writing Q/K/V head-major buffers directly in one Tile launch per block instead of a separate QKV bias add, QKV split, and three reshape launches. It uses `nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32` so TK SDPA backward reads the row-major attention-output gradient directly and converts bf16 `dQ`/`dK`/`dV` head-major gradients directly into row-major `grad_qkv`, replacing three bf16-to-float gradient conversion launches plus the heads-to-QKV merge launch. The full trainer no longer allocates row-major `grad_q`/`grad_k`/`grad_v` or head-major `grad_q_heads`/`grad_k_heads`/`grad_v_heads` scratch buffers. Native plan and training JSON report `qkv_forward_layout_strategy: "fused-split-to-heads"`, `qkv_bias_layout_strategy: "fused-qkv-bias-split-to-heads"`, `attention_backward_grad_layout_strategy: "merged-grad-out-direct"`, `attention_backward_qkv_bridge_strategy: "fused-bf16-heads-to-row-qkv"`, `attention_backward_strategy: "query-row-atomic-tile-score-reuse"`, `qkv_backward_layout_strategy: "fused-heads-to-qkv"`, and the elided layout launches per block.

Full GPT-2 `--train-transformer-lm` also fuses the MLP `c_fc` bias add with
GELU. `nfn_native_tile_gelu_add_bias_float32` consumes the no-bias CUBLAS
projection output, writes the biased preactivation required by GELU backward,
and writes the GELU activation in one Tile pass. Native plan and training JSON
report `mlp_fc_bias_gelu_strategy: "fused-bias-preactivation-gelu"` and one
elided legacy launch per block.

Full GPT-2 `--train-transformer-lm` also fuses attention-output and MLP
projection bias with residual addition. `nfn_native_tile_linear_bias_residual_add_float32`
consumes no-bias CUBLAS projection output, applies the projection bias and
residual scale, and writes the residual output in one Tile pass. Native plan and
training JSON report `projection_bias_residual_strategy:
"fused-linear-bias-residual-add"` and two elided legacy launches per block.

`neuralfn/csrc/native_train/token_shards.cpp` is the reusable no-Torch token-shard resolver and sequential batch sampler for native trainers. It resolves dataset aliases through `NFN_DATASETS_DIR`, validates sorted `fineweb_train_*.bin` / `fineweb_val_*.bin` uint16 shards, accepts llm.kittens-style `TinyStories_train.bin` / `TinyStories_val.bin`, infers validation siblings for direct train-bin paths, skips the 1024-byte cached-shard header when present, counts train/validation tokens, computes native microbatch plus gradient-accumulation metadata, and either produces token plus next-token target vectors for smoke/debug JSON or writes directly into caller-owned token/target buffers with `SequentialTokenBatchSampler::next_into()`. The full GPT-2 trainer uses `next_into()` with pinned memory, so real token payloads avoid graph-editor nodes, Python dataset objects, `TokenBatch` vector materialization, and vector-to-pinned copies.

`bash tools/build_native_missing_trainers.sh` builds compiled per-family native targets for missing trainers. GPT-2 evo is a model-aware C++ preflight: `nfn_gpt2_evo_native_train --print-plan --eval-every-steps 1000 --tile-cuda-activation-dtype nvfp4` emits JSON for the dense GPT-2 shape, schedule, `adamw` profile, validation cadence, NVFP4 activation intent, evo-layer index/cadence/population, available native pieces, and remaining candidate-evaluation/mutation/loss-reduction/adoption kernels without importing Python or Torch. NanoGPT is a model-aware C++ preflight that emits `--print-plan` JSON for the native shape, schedule, AdamW profile, token-shard constraints, contiguous parameter/gradient/AdamW-state buffers, AdamW decay/no-decay groups, forward/backward/optimizer `training_step_plan`, tied LM head backward coverage through the linear ABI, kernels already exposed through the native ABI, and kernels still required for real training. `--check-tile-ops --tile-ops-lib PATH` validates that the compiled NanoGPT binary can bind all required raw Tile ABI symbols from the trainer shared library. `--smoke-tile-ops --tile-ops-lib PATH` goes one step further by loading CUDA runtime, allocating a tiny device buffer, launching `nfn_native_tile_fill_float32`, copying the buffer back, and verifying the value without Python or Torch. `--smoke-optimizer-step --tile-ops-lib PATH` proves the same compiled path can build the NanoGPT parameter layout, initialize contiguous param/grad/AdamW moment buffers with raw fill kernels, execute `nfn_native_tile_adamw_step_float32` once per registered parameter buffer with that buffer's decay/no-decay setting, copy param and moment buffers back, and verify the update. `--smoke-training-loop-step --tile-ops-lib PATH` exercises native optimizer-loop mechanics over that registered layout: gradient zeroing, synthetic gradient fill, global-norm clip scale finalization, device-scalar gradient scaling, and per-buffer AdamW updates. `--smoke-lm-step --tile-ops-lib PATH` runs a tiny tied-embedding language-model step through token embedding, linear logits, token CE loss/backward, linear input/weight backward, token embedding weight backward, and AdamW update kernels, then verifies loss, gradient, and weight update values. `--smoke-token-train-step --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS` samples a real native uint16 token/target batch from cached shards, runs the tied-LM forward/backward/update kernels over those IDs, and verifies sampled-batch loss, gradient, and weight update values. `--train-token-lm --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS --max-steps N` runs a real multi-step tied token-embedding LM loop over cached shards with the C++ train and validation samplers, device-side gradient zeroing, token CE backward, tied weight updates, AdamW metrics JSON, and periodic validation losses under `validation.losses` without Python or Torch. `--smoke-embedding-norm-step --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS` runs sampled tokens through token plus absolute-position embeddings, residual add, LayerNorm forward/backward, tied logits, CE backward, embedding/position/norm gradients, and AdamW updates, then verifies residual, norm, loss, gradient, and weight update values. `--smoke-qkv-layout-step --tile-ops-lib PATH` verifies fused QKV split/merge layout kernels for the NanoGPT `attn.qkv.weight` activation and gradient path. `--smoke-fused-qkv-attention-step --tile-ops-lib PATH` runs a tiny attention stage through one fused `attn.qkv.weight`, QKV split, SDPA forward/backward, QKV gradient merge, fused qkv weight backward, output projection backward, and AdamW updates for fused qkv/output weights. `--smoke-transformer-block-step --tile-ops-lib PATH` composes LayerNorm, fused-QKV attention, residual adds, MLP, backward passes, gradient accumulation, and AdamW updates for a tiny transformer block through the raw native kernels. `--smoke-mlp-step --tile-ops-lib PATH` runs a tiny MLP stage through fc projection, GELU, output projection, projection/input backward, GELU backward, and AdamW updates for both MLP weights, then verifies forward, gradient, and weight update values. `--smoke-attention-step --tile-ops-lib PATH` remains the separate-Q/K/V attention comparison smoke; use `--cuda-runtime-lib PATH` or `NFN_CUDA_RUNTIME_LIB` if libcudart is not discoverable by the dynamic loader. Pass `--require-token-shards` to force cached-token shard validation and include the resolved shard metadata in the JSON; add `--sample-token-batch` to include the first native token/target batch. Full NanoGPT transformer training and the other model-family targets still report the CUDA Tile C++ kernels or trainer-loop integration required before each real trainer replaces its placeholder. The unified frontend dispatches to these binaries when present, so direct CLI and SDK paths stay on compiled native artifacts until each real trainer replaces its placeholder.

Native checkpoints written by `train_gpt2cu` are llm.kittens `.bin` files, not Torch `.pt` state dicts. Use the Torch-free checkpoint helpers to inspect them:

```python
from neuralfn.native_gpt2 import (
    latest_native_gpt2_checkpoint,
    read_native_gpt2_checkpoint_info,
)

checkpoint = latest_native_gpt2_checkpoint("~/NeuralFn/artifacts/gpt2")
info = read_native_gpt2_checkpoint_info(checkpoint)
print(info.to_dict())
```

The parser reads the native 256-int GPT header, reports precision, sequence length, vocabulary size, layer/head/channel shape, parameter count, expected file size, and whether the matching `DONE_########` marker exists. `nfn infer --checkpoint path/to/model_########.bin --native-info` and `python cli/scripts/infer_gpt2.py --native-checkpoint path/to/model_########.bin --native-info` use the same no-Torch path. Prompt generation from these native checkpoints still requires a dedicated native GPT-2 inference executable; graph-backed `nfn infer --graph ... --weights ...` remains for NeuralFn `.pt/.json` exports.

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

Binary and binary-pair Tile scalar function kernels require same-shaped contiguous inputs. Vector module kernels require a last dimension matching the stage parameter vector, except `qk_gain`, which expects `[B, H, ...]` input with a gain vector of length `H`. Norm kernels cover contiguous float32 rows with last dimension up to 1024, plus `group_norm` on `[B,S,D]` when `S * group_dim <= 1024`. Layout and indexing kernels cover contiguous float32 `reshape_heads`, `merge_heads`, `repeat_kv`, `rotary_embedding`, `absolute_position_embedding`, `token_embedding`, `byte_patch_embed`, `causal_chunk_state` prefix/mean chunk states, and KV cache copy/concat paths. Projection kernels cover contiguous float32/fp16 projection paths plus fp8 and NVFP4 projection-family activation contracts. The NVFP4 subset covers packed activations for `linear`, LM/router/value/reward/denoise heads, tied LM head, KV PCA encode/decode, JEPA heads, deterministic LoRA/TTT/adapter projections, `bitlinear_ternary`, `fp8_linear`, `mx_linear`, MLP projections, and ACT halt projection; `nf4_linear` stays excluded because it owns a separate NF4 packed-weight contract. Attention kernels cover contiguous CUDA float32, verified fp16 activation inputs, fp8 Q/K/V activation inputs, and NVFP4 packed activation inputs for `scaled_dot_product_attention`, `sliding_window_attention`, `block_sparse_attention`, `streaming_attention_sinks`, deterministic `native_sparse_attention`, `differential_attention`, `causal_self_attention`, `fused_causal_attention`, `multi_latent_attention`, and `routed_attention_experts` with key sequence length up to 1024, causal or non-causal masking, `dropout_p=0`, grouped-query attention when query heads are divisible by key/value heads, right-aligned sparse masks for cache-compatible query/key lengths, split Q/K dimensions for differential attention, Tile-composed projection/RoPE/output paths for self-contained attention stages, and fp32 route-weight accumulation for routed experts. KV quantization kernels cover same-shaped contiguous float32 K/V rows with `head_dim <= 512`, packing quantized values plus per-row scale and unpacking tensors shaped `[..., 2*head_dim+1]`. Semantic projector kernels cover flat and chunked topic-head, signature, and residual projections while preserving the per-dimension topic logits contract. Semantic hash kernels cover contiguous float32 semantic vectors and chunk vectors with up to 62 hash planes per table, returning int64 bucket IDs without gradients. Route kernels cover contiguous float32 route weights plus int64 route indices, `topk_route` covers contiguous float32 logits with `top_k <= 64`, `semantic_hash_router` covers unforced hash/topic routing through native top-k selection while preserving the PyTorch forced-target ordering path, `semantic_moe_jepa_evo_router` covers chunk-level shared/semantic/free route-logit construction with Tile free-expert projection and PyTorch-compatible candidate ordering, `auxfree_load_balancing` covers native per-expert bias addition with device-side no-grad bias updates, supervised semantic route BCE for `route_selection_loss`, route distillation KL reduction for `route_distillation_loss` with PyTorch reference preprocessing for topic dimensions wider than 1024 terms, and `attentionless_decoder` covers bucket-conditioned expert-output logits with native bucket embedding plus output projection. `dropout` uses Tile identity for inference and `p=0` plus deterministic counter-based training masks for contiguous fp32/fp16 activations; `random_timesteps`, `mask_scheduler`, and `jepa_mask` use deterministic counter-based device random generation so CPU/GPU parity does not depend on global PyTorch RNG state. `adamw_step`, `ema_update`, `gradient_accumulate`, `gradient_clip_norm`, `muon_step`, and `split_optimizer_step` cover contiguous CUDA float32 tensors or fp16 parameter/gradient tensors with float32 optimizer state through fp32 Tile-compatible compute; AdamW fp16 support requires fp16 parameter/gradient buffers with fp32 first/second moments, and Muon fp16 support requires fp16 parameter/gradient buffers with fp32 momentum. `muon_newton_schulz` remains float32-only as the standalone matrix orthogonalization primitive. `latent_pool` covers masked JEPA latent pooling with mean fallback for empty masks, and `token_cross_entropy`, `masked_token_cross_entropy`, `sequence_logp`, `latent_mse_loss`, `semantic_alignment_loss`, `dpo_pairwise_loss`, `ppo_clipped_loss`, `gae_compute`, `preference_bce_loss`, `load_balance_loss`, `route_balance_loss`, `route_distillation_loss`, and `softmax_distillation_loss` produce scalar losses or log-prob reductions through Tile reductions. DPO reward outputs remain detached to match the PyTorch stage contract. Non-CUDA tensors, unsupported dtypes, non-contiguous tensors, broadcasted inputs outside these contracts, and unsupported runtime contracts fall back to PyTorch unless strict mode is enabled.

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
