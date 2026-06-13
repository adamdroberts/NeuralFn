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
- Raw-text aliases whose tokenizer ids fit in `uint16` are cached as
  `fineweb_train_000000.bin` and optional `fineweb_val_000000.bin` on first
  training load. Schedule estimation should use token metadata or shard sizes
  instead of tokenizing raw text. Graph `dataset_source` nodes store aliases and
  `seq_len`, not real text/token payloads.
- Keep `sp1024`, `sp2048`, `sp4096`, and `sp8192` visible and allow missing
  shared tokenizer assets to download before training.
- Default CLI/server artifacts to `~/NeuralFn/artifacts`, unless
  `NEURALFN_ARTIFACTS_DIR` is set.
- Saved graph JSON is graph-first. Prefer graph metadata for weights,
  tokenizer, and training manifests; treat `--weights` as an override.
- Save artifacts before validation; validation failures should not erase a
  successful training artifact.
- Keep root `nfn --help` / no-argument startup and explicit dense GPT-2 native
  training dispatch off the `nfn_impl` and Torch import path.
- Keep `cli/scripts/train_gpt2.py` native-only and lightweight: importing it,
  building its parser, resolving defaults, and running direct native dispatch
  must set up repo/script imports without requiring `PYTHONPATH` and without
  importing Torch, `server.dataset_manager`, NumPy, or tiktoken.
- Keep `cli/scripts/infer_gpt2.py` parser construction, `--help`, and
  `--evo`/`--megakernel` artifact default resolution lightweight: do not import
  Torch, `server.dataset_manager`, NumPy, or graph-backed inference helpers
  until actual generation runs.
- Keep `NFN_DATASETS_DIR` honored by native GPT-2 alias resolution so Python and
  compiled native CLI paths can share a non-home dataset cache.
- Keep `nfn train|infer|eval --help` and `nfn kernels ... --help` on lightweight
  static help in `cli/nfn.py`; do not import `nfn_impl`, Torch, or graph-backed
  runtime modules for help-only commands.
- Keep `nfn kernels list [--json]` metadata-only: it should report registry
  coverage without importing `nfn_impl`, Torch, or graph-backed runtime modules.
- Keep non-GPT-2 training commands and direct legacy training scripts on
  compiled native binaries before Torch import. Direct scripts first try the
  model-family binary via `NFN_NATIVE_<MODEL>_CLI`,
  `build/nfn_<model>_native_train`, or an installed `nfn_<model>_native_train`;
  they fall back to the generic `nfn-native-train --base-model <model>` registry
  only when no family binary is available. They may enter graph-backed code only
  with `NFN_ALLOW_TORCH_TRAINING=1` for one-off debugging until native C++
  trainers exist for those model families.

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

## CUDA Tile commands

- `nfn kernels list [--json]` reports live CUDA Tile registry coverage without importing `nfn_impl`, Torch, or graph-backed runtime modules. Coverage includes `by_dtype` aggregate counts and per-spec `dtype_support` matrices for float32/fp16/fp8/NVFP4 automation. The fp8-supported entries include scalar/simple elementwise kernels, direct/composite projections, and attention Q/K/V modules with float32 accumulation where required. The NVFP4-supported entries currently cover packed projection-family activations for `linear`, LM/router/value/reward/denoise heads, tied LM head, KV PCA encode/decode, JEPA heads, deterministic LoRA/TTT/adapter projections, `bitlinear_ternary`, `fp8_linear`, `mx_linear`, MLP projections, and ACT halt projection, plus attention Q/K/V and shared attention inputs for SDPA, sparse attention variants, differential attention, causal/fused causal attention, MLA, and routed attention experts; unsupported fp8/NVFP4 entries use category-specific no-support reasons.
- `nfn train --help`, `nfn infer --help`, `nfn eval --help`, and `nfn kernels ... --help` are lightweight static help paths in `cli/nfn.py`; keep them useful for orientation, but do not route them through `nfn_impl`.
- `nfn kernels doctor [--json]` reports CUDA Tile toolchain diagnostics plus coverage.
- `nfn kernels bench [--device auto|cpu|cuda] [--iterations N] [--json]` compares old graph-walk PyTorch, static compiled PyTorch, and Tile-requested compiled execution on a small scalar graph.
- `nfn kernels examples [--write --output-dir examples/tile_cuda] [--json]` lists or regenerates checked-in examples plus one generated SDK snippet per registry entry.
- `nfn train`, `nfn infer`, and `nfn eval` accept `--kernel-backend {auto,torch,tile-cuda}`, `--tile-cuda-strict` / `--no-tile-cuda-strict`, and `--tile-cuda-report PATH`. Explicit `tile-cuda` requests CUDA Tile build tooling and now defaults to strict kernel enforcement; use `--no-tile-cuda-strict` only when intentionally debugging fallback behavior. `auto` still needs `NFN_TILE_CUDA_BUILD=1` to build on demand. The `tile-cuda` packaging extra must stay Torch-free; install `.[torch]` separately for graph-backed PyTorch execution or the legacy PyTorch Tile extension loader.
- `cli/scripts/train_gpt2.py` is native-only: direct execution with the default
  `compiled-cli` runner translates GPT-2 flags to the compiled C++ CLI before
  importing `train_gpt2_native.py`; explicit non-compiled runners still use
  `train_gpt2_native.py`. Cached uint16 train/validation shards go straight to
  the compiled GPT-2 Tile-CUDA trainer, so training token batches do not pass
  through graph-editor nodes or `TorchTrainer`.
- Cached native GPT-2 startup should avoid `server.dataset_manager`, NumPy,
  tiktoken, Torch, and pre-launch full-schedule estimation. Existing dataset
  directories are handed to C++ by alias/path without Python reading `meta.json`
  or revalidating shard metadata. The default dataset is
  `roneneldan__TinyStories__TinyStoriesV2-GPT4`; `golf1` and `golf10` are
  explicit cached-token shortcuts only. Preserve the llm.kittens TinyStories
  compatibility path: `--tinystories` may resolve directly to
  `TinyStories_train.bin` / `TinyStories_val.bin` under
  `/mnt/disk2/dev/open-source/llm.kittens/dev/data/tinystories`, with
  `NFN_LLM_KITTENS_TINYSTORIES_DIR` as the override. Direct train-bin paths
  should infer the sibling validation bin in C++, not through Python dataset
  materialization.
- Default `nfn train` commands go directly to a compiled native frontend before
  importing `train_gpt2_native`, `nfn_impl`, or Torch. GPT-2 reports
  `partial-native-trainer` and dispatches to the no-Python cached-shard CLI;
  NanoGPT `--train-token-lm` dispatches to its partial native trainer; unsupported
  families fail from the native registry.
- GPT-2 native training uses the SM120 AdamW schedule: 20,000 steps, seq len
  1024, microbatch 64, 524,288 tokens/step, LR 0.0006, weight decay 0.1, 60
  warmup steps, validation cadence 250 by default, and cosine decay to zero.
- Build native pieces with `bash tools/build_native_gpt2_binding.sh`,
  `bash tools/build_native_train_binding.sh`,
  `bash tools/build_native_train_tile_ops.sh`,
  `bash tools/build_native_gpt2_launcher.sh`,
  `bash tools/build_native_gpt2_cli.sh`,
  `bash tools/build_native_train_cli.sh`, and
  `bash tools/build_native_missing_trainers.sh`.
  `tools/install_native_gpt2_commands.sh` links `nfn-gpt2-native`,
  `nfn-native-train`, and both underscore/hyphen names for built per-family
  native targets.
- Native C++ trainers should link `libnfn_native_train_tile_ops.so` for
  single-buffer and multi-buffer AdamW, single-buffer and multi-buffer sumsq
  partials, gradient accumulation, device-buffer fill/zeroing, device-side
  global-norm clip scale finalization, device-scalar gradient scaling, reduction, linear, linear
  input/weight/weight-accumulate/bias/bias-accumulate backward, scaled residual
  add, fused QKV split/merge for NanoGPT `qkv.weight`, fused GPT-2 QKV
  split-to-heads, GPT-2 QKV bias+split-to-heads, and heads-to-QKV gradient
  merge, GELU forward/backward, token embedding forward/weight backward, absolute-position
  embedding forward/backward/backward-accumulate, RMSNorm, RMSNorm input
  backward, LayerNorm, LayerNorm input/affine/affine-accumulate backward,
  softmax, token and masked token-cross-entropy partial, token and masked
  token-cross-entropy logits backward, and scaled dot-product attention
  forward/backward kernels instead of using the PyTorch extension binding.
  `tools/build_native_train_tile_ops.sh` defaults to the SM120 ThunderKittens
  bf16 attention bridge (`NFN_TILE_CUDA_USE_TK_ATTENTION=1`,
  `NFN_TILE_CUDA_ARCH=sm_120a`) for GPT-2-compatible causal SDPA. Keep native
  JSON reporting `attention_backend_strategy: "tk-sm120-bf16-bridge"`,
  `attention_forward_strategy: "tk-sm120-bf16-flashattention-bridge"`,
  `attention_backward_strategy: "tk-sm120-bf16-recompute-forward-bridge"`,
  `attention_forward_tk_launch_count`, and
  `attention_backward_tk_launch_count` when the optimized path runs.
- The trainer-facing linear ABI should expose and preserve the linear backend
  telemetry: `linear_backend_strategy`,
  `linear_bf16_gemm_count`, `linear_sgemm_count`,
  `linear_bf16_a_pack_count`, `linear_bf16_a_cache_hit_count`,
  `linear_bf16_cache_reset_count`, `linear_bf16_cached_a_capacity`, and
  `linear_bf16_cache_entry_count`. Dense GPT transformer block forward/recompute
  projections should use `nfn_native_tile_linear_bf16_float32`, block dInput
  GEMMs should use `nfn_native_tile_linear_backward_input_bf16_float32`, and
  block dWeight accumulation should use
  `nfn_native_tile_linear_backward_weight_accumulate_bf16_float32`. The tied
  LM-head logits, dHidden, and dWeight GEMMs should stay on optimized TF32
  tensor-op `cublasSgemm`, not scalar Tile dot products. The CE backward path
  should reuse the logits chunk as dlogits through
  `nfn_native_tile_token_cross_entropy_backward_inplace_with_workspace_float32`
  and report `grad_logit_workspace_elements: 0`,
  `lm_head_ce_backward_strategy: "inplace-logits-dlogits-workspace"`, and
  `lm_head_grad_logits_workspace_allocated: false`. The trainer should
  report `linear_backend_strategy:
  "block-forward-dinput-dweight-bf16-lm-head-tf32"`,
  `block_forward_linear_strategy`, `block_backward_input_linear_strategy`,
  `block_backward_weight_linear_strategy`,
  and `non_block_forward_backward_linear_strategy`.
- The row-vector forward and query-row atomic backward float32 SDPA kernels are
  fallback/diagnostic paths for unsupported shapes or
  `NFN_TILE_CUDA_USE_TK_ATTENTION=0` builds. Do not make them the default dense
  GPT-2 training path again; keep their row/fallback/scalar counters and launch
  diagnostics available for debugging.
- Full GPT-2 `--train-transformer-lm` JSON should include a `timing` object
  with host wall-clock phase timers for setup, train loop, validation,
  checkpoint export, total runtime, optimizer steps per second, and train
  tokens per second. These timers should not add new device synchronizations.
- Full GPT-2 `--train-transformer-lm` may opt into CUDA-event stage timing via
  `NFN_NATIVE_GPT2_STAGE_TIMING=1`. Keep this diagnostic disabled by default.
  When enabled, JSON under `timing` should include `stage_timing_enabled`,
  `stage_timing_event_count`, `stage_timing_dropped_event_count`, and
  `stage_timing` records for token upload, model/block forward, block
  recompute/backward, LM-head backward, final-norm/embedding backward,
  gradient zero/clip, and AdamW update. Keep nested diagnostic records for
  LM-head logits/CE/dHidden/dWeight, block forward/recompute attention and MLP
  phases, and block backward MLP projection, MLP fc, LayerNorm/residual,
  attention projection, attention SDPA, and QKV phases. Preserve individual
  block-backward dWeight, bias, dInput, activation, residual-add, and
  attention-to-QKV records such as `block_backward.mlp_proj.dweight`,
  `block_backward.mlp_proj.dinput`, `block_backward.attn_sdpa.to_qkv`, and
  `block_backward.qkv.dweight`.
- GPT-2 block backward should use
  `nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_reuse_forward_from_merged_grad_float32`
  only after a matching TK attention forward has populated the process
  workspace. JSON should report `attention_backward_strategy:
  "tk-sm120-bf16-reuse-forward-workspace-bridge"`,
  `attention_backward_reuses_forward_workspace: true`, and
  `attention_backward_recompute_forward_elided_per_block: 1`; generic paths
  that cannot prove the forward/backward ordering should keep using
  `nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32`.
- Full GPT-2 `--train-transformer-lm` should use
  `nfn_native_tile_split_qkv_to_heads_add_bias_float32` to apply Q/K/V bias and
  write Q/K/V head-major buffers in one launch per block. Keep
  `qkv_forward_layout_strategy: "fused-split-to-heads"`,
  `qkv_bias_layout_strategy: "fused-qkv-bias-split-to-heads"`, and the elided
  legacy layout launches in native plan/training JSON.
- Full GPT-2 `--train-transformer-lm` should not use standalone
  `nfn_native_tile_merge_heads_to_qkv_float32` after TK SDPA backward in the
  full trainer. Use the direct QKV backward bridge below instead, so the full
  trainer avoids separate bf16-to-float gradient conversions, the
  `merge_heads_to_qkv` launch, and the full-size head-gradient scratch buffers.
- Full GPT-2 `--train-transformer-lm` should use
  the merged-gradient attention backward contract so SDPA backward reads
  row-major attention-output gradients directly from the projection backward
  output. The current full trainer implements this through the direct QKV bridge
  below. Do not reintroduce the pre-backward `reshape_heads` launch or
  `grad_attn_heads` scratch buffer in the full trainer. Keep
  `attention_backward_grad_layout_strategy: "merged-grad-out-direct"` and the
  elided grad-output layout launch in native plan/training JSON.
- Full GPT-2 `--train-transformer-lm` should use
  `nfn_native_tile_split_qkv_to_heads_add_bias_float32` after a no-bias QKV
  projection so Q/K/V bias is applied while writing head-major Q/K/V buffers.
  Do not reintroduce standalone QKV `linear_add_bias` launches in the full
  trainer. Keep `qkv_bias_layout_strategy: "fused-qkv-bias-split-to-heads"` in
  native plan/training JSON.
- Full GPT-2 `--train-transformer-lm` should use
  `nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32`
  for the TK bf16 attention backward bridge. It should write row-major
  `grad_qkv` directly from bf16 head-major `dQ`/`dK`/`dV`, without allocating
  full-size `grad_q_heads`, `grad_k_heads`, or `grad_v_heads` scratch buffers
  and without launching separate bf16-to-float gradient conversions plus
  `merge_heads_to_qkv` in the full trainer. Keep
  `attention_backward_qkv_bridge_strategy: "fused-bf16-heads-to-row-qkv"` in
  native plan/training JSON.
- Full GPT-2 `--train-transformer-lm` should use
  `nfn_native_tile_gelu_add_bias_float32` for the MLP `c_fc` bias plus GELU
  pass. The `c_fc` linear call should run without bias, then the fused kernel
  should write both the biased preactivation for GELU backward and the GELU
  activation. Do not reintroduce separate MLP `linear_add_bias` and
  `gelu_float32` launches in the full trainer. Keep
  `mlp_fc_bias_gelu_strategy: "fused-bias-preactivation-gelu"` in native
  plan/training JSON.
- Native trainer-facing GELU should use the GPT-style tanh approximation in
  `nfn_native_tile_gelu_float32`, `nfn_native_tile_gelu_add_bias_float32`,
  `nfn_native_tile_gelu_backward_float32`, and
  `nfn_native_tile_gelu_backward_inplace_float32`. Keep forward, fused
  bias+forward, explicit backward, and in-place backward aligned on
  `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`.
- Full GPT-2 `--train-transformer-lm` should use
  `nfn_native_tile_linear_bias_residual_add_float32` for attention-output and
  MLP `c_proj` bias plus residual addition. The projection linear calls should
  run without bias, then the fused kernel should apply projection bias,
  residual scale, and residual add together. Do not reintroduce separate
  projection `linear_add_bias` and `scaled_residual_add_float32` launches in
  the full trainer. Keep `projection_bias_residual_strategy:
  "fused-linear-bias-residual-add"` in native plan/training JSON.
- Use `neuralfn/csrc/native_train/token_shards.cpp` for sorted uint16
  train/validation shard discovery, llm.kittens `TinyStories_train.bin` /
  `TinyStories_val.bin` compatibility, direct train-bin validation sibling
  inference, header skipping, token counts, contiguous batch-segment
  token/target sampling, and microbatch/gradient-accumulation metadata without
  graph-node payloads. Token-shard JSON should report `batch_read_strategy:
  "contiguous_shard_segments"`.
- NanoGPT's target is a partial native C++ trainer. `nfn_nanogpt_native_train`
  plan, tile-op smoke, optimizer smoke, training-loop smoke, and LM-step smoke
  paths must stay Python/Torch-free. Use `--cuda-runtime-lib PATH` or
  `NFN_CUDA_RUNTIME_LIB` for explicit libcudart resolution.
- Other missing-family entrypoints such as `nfn_llama_native_train` intentionally
  fail with the family-specific CUDA Tile trainer work still required. Use
  `nfn-native-train --list-models --json` or
  `neuralfn.native_train.native_train_model_registry()` to inspect native
  coverage; use `run_native_train(build_native_train_run_config(...),
  runner="auto")` for SDK-level C++ binding handoff to the compiled frontend.
- The default runner is `compiled-cli`. Use `--eval-every-steps 1000` for
  per-1000-step validation loss, native command/config inspection flags for
  debugging, and `NFN_NATIVE_*_CLI` environment overrides for installed native
  commands. The 5090 helper scripts should not add wallclock caps by default.
  The default root install no longer installs Torch; use `.[tile-cuda]` for
  native CUDA Tile tooling and `.[torch]` for graph-backed Torch workflows.
- For the GPT-2 `compiled-cli` runner, only skip Python shard metadata validation when cached train plus validation shard files already exist. Raw-text dataset directories must still materialize token shards before C++ training. The exceptions are wrapper-level `--native-cuda-dry-run --native-cuda-print-command`, compiled Tile-CUDA `--print-command`, and no-data compiled preflights (`--check-tile-ops`, `--smoke-tile-ops`, `--smoke-optimizer-step`, `--smoke-lm-step`, `--smoke-attention-step`, `--smoke-mlp-step`, `--smoke-norm-residual-step`, and `--smoke-transformer-block-step`). Those paths build, print, or run synthetic/native ABI checks before token-shard resolution, must not import `server.dataset_manager`, NumPy, tiktoken, or Torch, must not write `fineweb_train_*.bin` shards, must not add the external `--target train_gpt2cu` bridge argument for the default Tile-CUDA backend, and should report `token_shards_resolved: false` when no dataset was opened. The explicit `llm-kittens` backend may still receive `--target` and resolve shards when printing its delegated `train_gpt2cu -i/-j` command.
- GPT-2 native command paths must accept `--template-name` / `--template` /
  `--preset` and `--graph-file` / `--graph` without importing Torch. Cover
  every name in `neuralfn.config.SHIPPED_GPT_TEMPLATE_PRESETS`, including
  `gpt2_megakernel`, `nanogpt_megakernel`, and aliases such as `mixllama`.
  Keep the compiled C++ `shipped_template_catalog` in sync with that SDK
  catalog, and assert `template_known` plus `shipped_template_catalog_count` in
  native plan tests.
  Top-level `nfn train --base-model gpt2` direct compiled-CLI handoff should add
  `--train-transformer-lm` for normal training commands, including selector
  commands, unless a plan/check/smoke/train action was already requested.
  Dense GPT-2-compatible presets (`gpt2`, `gpt2_megakernel`, and `gpt2_moa`)
  may run the current native loop; `gpt2_moa` should resolve to
  `--native-cuda-activation moa`. `--dry-run` / `--print-plan` should report
  `status: "native-transformer-lm-ready"` plus
  `training_step_plan.status: "ready"`; structurally different template names
  and custom graph files must report `selected-graph-native-trainer-missing`
  until a matching C++ Tile trainer plan exists. Unknown/unshipped template
  names must report `unknown-template` so typos are not mixed with known
  migration backlog items.
- GPT-2 native `--train-transformer-lm` must honor `train_batch_tokens` as the
  effective optimizer-step batch. Derive `grad_accum_steps` from
  `batch_size * seq_len`, stream that many cached-shard microbatches through
  CUDA Tile forward/backward, accumulate scaled gradients on device, then run
  gradient clip and AdamW once per optimizer step. JSON should report
  `microbatch_tokens`, `requested_train_batch_tokens`, `grad_accum_steps`,
  `effective_train_batch_tokens`, `train_microbatches_completed`, and
  `gradient_accumulation_strategy`.
- GPT-2 native transformer-LM training must keep cached token/target batches as
  uint16 for H2D upload, sample them directly into pinned host memory with
  `SequentialTokenBatchSampler::next_into()`, enqueue one contiguous
  `cudaMemcpyAsync` for tokens plus targets, and widen the combined arena to
  int64 on device with one `nfn_native_tile_uint16_to_int64` launch. Do not add
  per-batch CPU token expansion, `TokenBatch` vector materialization, or range
  validation back into the hot path; JSON should report
  `token_id_upload_strategy: "uint16-pinned-async-h2d-device-widen"`,
  `token_id_host_staging: "pinned"`, `token_id_h2d_copy:
  "cudaMemcpyAsync-contiguous-arena"`,
  `token_id_h2d_copy_calls_per_microbatch: 1`,
  `token_id_widen_strategy: "single-contiguous-arena-kernel"`,
  `token_id_widen_kernel_launches_per_microbatch: 1`,
  `token_batch_staging_strategy: "direct-sampler-to-pinned-arena"`,
  `token_batch_vector_materialization: false`, and `token_id_host_validation:
  false`.
- Token upload/storage buffers in full GPT-2 `--train-transformer-lm` should use
  combined arenas: one aligned device arena for widened int64 token/target
  buffers plus compact uint16 H2D staging, and one pinned uint16 host arena.
  Do not reintroduce separate token/target `cudaMalloc` calls in the real
  trainer. JSON should report `token_buffer_allocation_strategy:
  "combined-arenas"`, `token_device_allocation_strategy:
  "single-device-arena"`, `token_device_arena_cuda_malloc_count`,
  `token_device_arena_suballocation_count`, and
  `token_device_cuda_mallocs_elided`.
- GPT-2 native transformer-LM startup should initialize the tied token
  embedding/LM-head weight on device with
  `nfn_native_tile_init_gpt2_token_weight_float32`; do not rebuild the old
  154 MB host float matrix for the real training path. JSON should report
  `token_weight_init_strategy: "device-tile-deterministic"` and
  `token_weight_host_materialization: false`.
- GPT-2 native `--backend tile-cuda` / wrapper `--kernel-backend tile-cuda`
  is a NeuralFn-owned raw Tile ABI path with smoke coverage plus a tiny
  12-layer `--train-transformer-lm` loop. Full dense GPT-2 Tile training should
  keep using `libnfn_native_train_tile_ops.so`. The plan includes the GPT-2 parameter
  layout and forward/backward/optimizer stage sequence. `nfn-native-train
  --list-models` should report GPT-2 as `partial-native-trainer` until that
  full loop exists.
- `nfn_gpt2_native_train --smoke-tile-ops --tile-ops-lib PATH` / wrapper
  `--native-cuda-smoke-tile-ops` launches `nfn_native_tile_fill_float32`
  through dynamically loaded CUDA runtime and verifies copyback without Python,
  Torch, or graph-node payloads. Use `--cuda-runtime-lib PATH` or
  `NFN_CUDA_RUNTIME_LIB` when libcudart needs an explicit path. Backend names
  are strict: use `llm-kittens` or `tile-cuda`, not compatibility aliases.
- `nfn_gpt2_native_train --smoke-optimizer-step --tile-ops-lib PATH` / wrapper
  `--native-cuda-smoke-optimizer-step` allocates GPT-2-sized param/grad/AdamW
  buffers and runs one raw Tile AdamW call per registered GPT-2 parameter buffer
  with decay/no-decay metadata, then samples copyback values without Python/Torch.
- `nfn_gpt2_native_train --smoke-lm-step --tile-ops-lib PATH` / wrapper
  `--native-cuda-smoke-lm-step` runs a tiny GPT-2-shaped tied embedding/LM-head
  forward/backward/update slice through raw Tile kernels without Python/Torch.
- `libnfn_native_train_tile_ops.so` is built with
  `NFN_TILE_CUDA_USE_CUBLAS_LINEAR=1`, so trainer-facing native linear forward,
  dInput, dWeight, accumulate-dWeight, forced-BF16 forward, forced-BF16 dInput,
  and forced-BF16 accumulate-dWeight ABI symbols use GPU GEMM, and linear bias
  plus accumulate-bias backward use GPU GEMV over a cached device ones vector
  initialized by a Tile fill kernel, while keeping Torch and the PyTorch Tile
  extension out of the training process. The pure Tile direct dot-product and
  row-chunked atomic kernels remain the fallback for non-trainer builds.
- `nfn_gpt2_native_train --smoke-embedding-lm-step --tile-ops-lib PATH` /
  wrapper `--native-cuda-smoke-embedding-lm-step` samples a tiny cached uint16
  token batch in C++ and runs token embedding, absolute position embedding,
  embedding residual add, final LayerNorm, tied LM head, CE backward,
  embedding/norm backward, and AdamW without graph-editor payloads.
- `nfn_gpt2_native_train --train-embedding-lm --tile-ops-lib PATH` runs that
  GPT-2 embedding/final-norm/LM path as a real multi-step compiled loop over cached train shards, with
  validation losses from validation shards controlled by `--eval-every-steps`,
  `--eval-batches`, and `--eval-batch-size`.
- `nfn_gpt2_native_train --smoke-attention-step --tile-ops-lib PATH` / wrapper
  `--native-cuda-smoke-attention-step` runs a tiny GPT-2 model-dim attention
  stage through qkv projection, QKV split, SDPA forward/backward, QKV gradient
  merge, projection backward, and AdamW without Python/Torch.
- `nfn_gpt2_native_train --smoke-mlp-step --tile-ops-lib PATH` / wrapper
  `--native-cuda-smoke-mlp-step` runs a tiny GPT-2 MLP stage through c_fc
  projection, GELU forward/backward, c_proj projection backward, and AdamW
  without Python/Torch.
- `nfn_gpt2_native_train --smoke-norm-residual-step --tile-ops-lib PATH` /
  wrapper `--native-cuda-smoke-norm-residual-step` runs GPT-2 LayerNorm, scaled
  residual add, LayerNorm affine/input backward, gradient accumulation, and
  AdamW without Python/Torch.
- `nfn_gpt2_native_train --smoke-transformer-block-step --tile-ops-lib PATH` /
  wrapper `--native-cuda-smoke-transformer-block-step` composes GPT-2 LayerNorm,
  fused QKV attention, real 12-head reshape/merge layout (`12 x 64`), residual
  adds, MLP, backward passes, gradient accumulation, projection bias gradients,
  and AdamW updates for all 12 GPT-2 block parameter buffers without
  Python/Torch.
- `nfn_gpt2_native_train --smoke-transformer-lm-step --tile-ops-lib PATH` /
  wrapper `--native-cuda-smoke-transformer-lm-step` samples cached uint16 tokens,
  preserves range-checked GPT-2 token IDs, and runs token/position embeddings,
  one tiny transformer block, final LayerNorm, tied LM head, CE
  forward/backward, transformer backward, embedding backward, and AdamW for 16
  parameter buffers without Python/Torch.
- `nfn_gpt2_native_train --train-transformer-lm --tile-ops-lib PATH` /
  wrapper `--train-transformer-lm` is the default dense GPT-2 Tile-CUDA training
  action and runs a full-vocab real-dim 12-layer
  transformer-LM multi-step loop over cached shards with periodic validation
  records in `validation.losses`, token/position embeddings, transformer
  blocks, final norm, a row-chunked tied LM-head/CE workspace, transformer
  backward, embedding backward, device-side global norm gradient clipping, and
  AdamW parameter updates. Block parameter allocation, initialization, gradient
  zeroing, gradient clipping, AdamW updates, checkpoint export, activation tape,
  forward block execution, and backward block execution must be driven from
  per-block C++ state/tape vectors with `activation_tape_strategy:
  "scratch-recompute"`. The JSON reports `trained_layers: 12`,
  `target_layers: 12`, `vocab: 50257`,
  `lm_head_row_chunk_size`, `lm_head_row_chunk_count`, `loss_partial_count`,
  `logit_workspace_elements`, `block_state_layout` with
  `parameter_allocation_loop`, `parameter_initialization_loop`,
  `gradient_zero_loop`, `gradient_clip_loop`, `adamw_update_loop: false`,
  `adamw_update_loop_elided`, `adamw_update_strategy`,
  `checkpoint_export_loop`, `activation_tape_loop`, `forward_block_loop`, and
  `backward_block_loop`, `gradient_partial_count`, `gradient_clip_norm`, and
  `sample_gradient_clip_scale`; successful runs write a final 12-layer
  trained-weight native checkpoint plus `DONE_########` marker. Checkpoint
  export must pack all device float32 weights into one contiguous bf16 payload
  with `nfn_native_tile_float32_to_bf16_bits_many`, copy the compact uint16
  payload to host once, and report `checkpoint.payload_pack_strategy:
  "device-many-float32-to-bf16-bits-contiguous"`, `payload_pack_kernel:
  "nfn_native_tile_float32_to_bf16_bits_many"`, `payload_copy_strategy:
  "single-contiguous-device-payload-d2h"`, `payload_cpu_bf16_conversion:
  false`, `device_pack_kernel_launches`, `d2h_copy_count`, `d2h_bytes`, and
  `float32_d2h_bytes_elided`. Do not route it through Python/Torch fallback.
  Use `--no-train-transformer-lm` only for direct
  C++ plan/check/debug commands that must not start the default trainer.
- Full GPT-2 `--train-transformer-lm` defaults the tied LM-head row chunk to
  8192 rows. Use `--lm-head-row-chunk-size` on the compiled C++ CLI or
  `--native-cuda-lm-head-row-chunk-size` from root/wrapper CLI to override it.
  Loss partials must reduce on device with `nfn_native_tile_sum_partials_float32`
  before the single host loss scalar copy. Tied LM-head dWeight chunks must
  accumulate directly into `accum_grad_token_weight` with
  `nfn_native_tile_linear_backward_weight_accumulate_float32`, not through a
  full-vocab scratch gradient buffer per chunk or per microbatch.
- Full GPT-2 `--train-transformer-lm` block Linear dWeight kernels must also
  accumulate directly into per-block optimizer-step accumulation buffers. Do not
  allocate or copy scratch dWeight buffers for qkv, attention-output, MLP fc, or
  MLP projection weights in the real 12-layer trainer. Keep
  `block_linear_weight_gradient_accumulation_strategy:
  "direct-device-accumulation-buffer"`,
  `block_linear_weight_gradient_scratch_buffers_allocated: false`.
- Full GPT-2 `--train-transformer-lm` must keep train-loss sampling disabled
  for performance. Ordinary optimizer steps should skip the tied LM-head CE
  loss/reduction pass and the extra post-update device synchronize; validation
  cadence should compute validation loss without also measuring train loss.
  JSON should include `train_loss_sparse: false`,
  `train_loss_sampling: "disabled"`,
  `train_loss_on_validation_steps: false`, `train_loss_eval_count`, and
  `train_loss_last_step`.
- Persistent block-output preservation in full GPT-2 `--train-transformer-lm`
  must use `nfn_native_tile_copy_float32`, not a zero-fill plus
  `nfn_native_tile_gradient_accumulate_float32(scale=1)` pair. The final block
  output copy should be elided because final LayerNorm consumes it before
  backward recomputation starts; the default 12-layer JSON should report
  `persistent_block_outputs: 11` and `final_block_output_copy_elided: true`.
- Validation forwards in full GPT-2 `--train-transformer-lm` should not copy
  intermediate block outputs into persistent training-backward buffers because
  no backward pass follows validation. JSON should report
  `validation_persistent_block_outputs: 0` and
  `validation_block_output_copies_elided: true`.
- Scratch-recompute backward in full GPT-2 `--train-transformer-lm` should reuse
  the final block activations left in the scratch tape by the initial forward
  pass. Only earlier blocks should be recomputed; the default 12-layer JSON
  should report `backward_recompute_blocks: 11` and
  `final_block_backward_recompute_elided: true`. Earlier-block recompute should
  stop after the MLP GELU activation; backward does not consume recomputed MLP
  projection output or final residual output. Keep
  `backward_recompute_mlp_projection_elided: true` and
  `backward_recompute_final_residual_elided: true`.
- MLP projection backward in full GPT-2 `--train-transformer-lm` should write
  projection dInput directly into the MLP fc gradient buffer and then run
  `nfn_native_tile_gelu_backward_inplace_float32`. Do not reintroduce a
  hidden-size `grad_act` scratch buffer in the full trainer; keep
  `mlp_proj_backward_gelu_inplace: true` and
  `mlp_proj_backward_grad_act_scratch_allocated: false`.
- Residual-gradient pair additions in full GPT-2 `--train-transformer-lm`
  backward must use `nfn_native_tile_scaled_residual_add_float32`, not zero-fill
  plus two gradient-accumulate launches. JSON should report
  `block_state_layout.residual_backward_fused`.
- Gradient clipping in full GPT-2 `--train-transformer-lm` must pass the device
  clip scalar into `nfn_native_tile_adamw_step_with_device_scale_float32`, not
  scale every gradient buffer with a separate launch before AdamW. JSON should
  report `block_state_layout.adamw_device_clip_scale_fused`.
- The gradient-clipping sumsq phase must use
  `nfn_native_tile_sumsq_partials_many_float32` over the device-resident
  gradient descriptor table. Do not reintroduce one sumsq kernel launch per
  gradient buffer in the real 12-layer trainer. JSON should report
  `gradient_clip_strategy: "fused-multi-buffer-sumsq-device-scale"`,
  `gradient_sumsq_kernel_launches_per_optimizer_step`,
  `gradient_sumsq_per_buffer_launches_elided`, and
  `block_state_layout.gradient_clip_loop: false`.
- AdamW updates in full GPT-2 `--train-transformer-lm` must use
  `nfn_native_tile_adamw_step_many_with_device_scale_float32` over
  device-resident parameter descriptors. Do not reintroduce one optimizer kernel
  launch per parameter buffer in the real 12-layer trainer. JSON should report
  `adamw_update_strategy: "fused-multi-buffer-device-scale"`,
  `adamw_descriptor_count`, `adamw_step_kernel_launches_per_optimizer_step`,
  and `adamw_per_buffer_step_launches_elided`.
- Token and position gradients in full GPT-2 `--train-transformer-lm` must
  accumulate directly into optimizer-step accumulation buffers. The tied LM-head
  CE backward scale should include the microbatch accumulation factor, LM-head
  dWeight chunks and token embedding backward should write into
  `accum_grad_token_weight`, and the old full-vocab `grad_token_weight` scratch
  buffer should not be allocated in the real 12-layer trainer. Position
  embedding backward should use
  `nfn_native_tile_absolute_position_embedding_backward_accumulate_float32`, not
  a `grad_position_weight` scratch buffer plus copy.
- Gradient zeroing in full GPT-2 `--train-transformer-lm` should happen only on
  optimizer-step accumulation buffers. Do not add per-microbatch scratch
  gradient zeroing back to the real 12-layer trainer. LayerNorm affine and
  Linear bias gradients should use
  `nfn_native_tile_layer_norm_backward_affine_accumulate_float32` and
  `nfn_native_tile_linear_backward_bias_accumulate_float32`, not scratch buffers
  plus `gradient_accumulate` copy launches. JSON should report
  `block_state_layout.gradient_zero_strategy` as
  `"fused-multi-buffer-accumulation-zero"`,
  `gradient_zeroed_buffer_count: 0`,
  `gradient_zero_kernel_launches_per_optimizer_step`, and
  `gradient_zero_per_buffer_launches_elided`,
  `block_state_layout.gradient_accumulation_loop: false`,
  `block_state_layout.gradient_accumulation_copy_loop_elided: true`,
  `block_state_layout.per_block_gradient_buffers: 0`,
  `block_state_layout.per_block_direct_accum_gradient_buffers: 12`,
  `token_gradient_accumulation_strategy` as
  `"direct-device-accumulation-buffer"`,
  `position_gradient_accumulation_strategy` as
  `"direct-device-accumulation-buffer"`,
  `layer_norm_affine_gradient_accumulation_strategy` as
  `"direct-device-accumulation-buffer"`,
  `linear_bias_gradient_accumulation_strategy` as
  `"direct-device-accumulation-buffer"`, and the token/position/LN/bias
  scratch-buffer allocated fields as `false`.
- Full GPT-2 `--train-transformer-lm` startup must keep per-block
  parameter/gradient allocation, scratch-tape activation allocation, parameter
  initialization, and AdamW-state zeroing under the block-vector visitors. Do not
  add block-0 aliases back to the global startup lists. Trainer JSON should
  report `block_state_layout.block0_duplicate_allocation_elided`,
  `block0_duplicate_activation_allocation_elided`,
  `block0_duplicate_parameter_initialization_elided`, and
  `block0_duplicate_adamw_state_zero_elided`.
- Full GPT-2 `--train-transformer-lm` should suballocate float buffers from one
  aligned CUDA device arena. Do not reintroduce one `cudaMalloc` per float
  tensor in the real trainer. JSON should report
  `float_allocation_strategy: "single-arena"`,
  `float_allocation_cuda_malloc_count`, `float_allocation_request_count`,
  `float_arena_requested_elements`, and `float_arena_allocated_elements`.
- Full GPT-2 `--train-transformer-lm` startup should zero the float arena once
  and leave zero biases plus AdamW state at their arena-zero values. Do not
  reintroduce per-buffer zero-fill launches for those tensors. JSON should
  report `float_arena_zero_init_strategy: "single-arena-fill"`,
  `float_arena_zero_fill_count`, `startup_per_buffer_zero_fill_elided`, and
  `startup_per_buffer_zero_fill_launches_elided`; the default 12-layer shape
  elides 369 per-buffer zero-fill launches.
- Full GPT-2 `--train-transformer-lm` startup should initialize nonzero
  constant parameters through `nfn_native_tile_fill_many_values_float32`, not
  one fill launch per tensor. Keep
  `parameter_initialization_strategy: "fused-multi-buffer-fill-values"`,
  `parameter_initialization_loop: false`, and
  `parameter_initialization_per_buffer_launches_elided`; the default 12-layer
  shape elides 74 nonzero fill launches.
- Full GPT-2 `--train-transformer-lm` descriptor tables should use one device
  descriptor arena for parameter fill, gradient zeroing, gradient clipping, and
  AdamW. Keep `descriptor_allocation_strategy: "single-device-arena"`,
  `descriptor_arena_cuda_malloc_count`, `descriptor_arena_suballocation_count`,
  `descriptor_upload_strategy: "single-host-packed-arena-copy"`,
  `descriptor_arena_copy_count`, `descriptor_arena_copy_calls_elided`, and
  `descriptor_cuda_mallocs_elided`; do not reintroduce one `cudaMalloc` or one
  H2D copy per descriptor table.
- LayerNorm affine-gradient backward must keep the overwrite raw Tile ABI for
  smoke paths and use the accumulate raw Tile ABI in the real full GPT-2
  trainer. The large-row path should use chunked parallel atomic accumulation.
  GPT-2 trainer JSON should report
  `block_state_layout.layer_norm_backward_affine_strategy` as
  `auto-chunked-atomic-accumulate`.
- Full GPT-2 `--train-transformer-lm` runs must emit `cuda_runtime_preflight`
  before allocation. Treat `cudaDriverGetVersion` returning driver version `0`
  or a loaded CUDA runtime newer than the driver as an early native failure so
  SM120 benchmarking failures point at GPU access/runtime compatibility, not a
  later `cudaMalloc` error.
- `nfn_gpt2_native_train --checkpoint-metadata-smoke --output-dir PATH` writes
  a sparse version-5 bf16 native GPT-2 checkpoint-format file plus
  `DONE_########` marker for the requested `--num-layers` target shape without Python, Torch,
  or CUDA. This is metadata/checkpoint-format coverage for native artifact
  discovery.
- `nfn_nanogpt_native_train --smoke-mlp-step --tile-ops-lib PATH` runs a tiny MLP stage through raw fc projection, GELU, output projection, projection/input backward, GELU backward, and AdamW update kernels, then verifies forward, gradient, and weight update values without Python/Torch.
- `nfn_nanogpt_native_train --smoke-attention-step --tile-ops-lib PATH` runs a tiny attention stage through raw Q/K/V projections, SDPA forward/backward, output projection forward/backward, Q/K/V projection backward, and AdamW update kernels, then verifies forward, gradient, and weight update values without Python/Torch.
- `nfn_nanogpt_native_train --smoke-token-train-step --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS` samples a real native uint16 token/target batch from cached shards, runs the raw tied-LM forward/backward/update kernels over those IDs, and verifies sampled-batch loss, gradient, and weight update values without Python/Torch.
- `nfn_nanogpt_native_train --train-token-lm --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS --max-steps N` runs the tied token-embedding LM path as a real multi-step native loop over cached shards with device-side gradient zeroing, token CE backward, AdamW updates, and periodic validation loss over validation shards. Use `--eval-every-steps`, `--eval-batches`, and `--eval-batch-size`; results are emitted under JSON `validation.losses` without graph-editor node payloads, `TorchTrainer`, or Python dataset flow. It is implemented; full NanoGPT transformer training is still the remaining loop-integration work.
- Normal NanoGPT training entrypoints default to that partial native loop: `nfn train --base-model nanogpt ...` and direct `python cli/scripts/train_nanogpt.py ...` add `--train-token-lm` before dispatching to `nfn-native-train`. `--dry-run` and `--print-command` inspect that same route without starting the loop; explicit native actions such as `--print-plan`, `--check-tile-ops`, a smoke command, or `--train-token-lm` still run exactly as requested.
- `nfn_nanogpt_native_train --smoke-embedding-norm-step --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS` samples a real native uint16 token/target batch from cached shards, runs token/position embedding, residual add, LayerNorm forward/backward, tied logits, CE backward, embedding/position/norm gradient, and AdamW update kernels, then verifies copyback values without Python/Torch.
- `nfn_nanogpt_native_train --smoke-fused-qkv-attention-step --tile-ops-lib PATH` runs a tiny attention stage through one fused `attn.qkv.weight`, QKV split, SDPA forward/backward, QKV gradient merge, fused qkv weight backward, output projection backward, and AdamW updates for fused qkv/output weights without Python/Torch.
- `nfn_nanogpt_native_train --smoke-transformer-block-step --tile-ops-lib PATH` composes LayerNorm, fused-QKV attention, residual adds, MLP, backward passes, gradient accumulation, and AdamW updates for a tiny transformer block through raw native kernels without Python/Torch.
- Native GPT-2 checkpoints from `train_gpt2cu` or NeuralFn's `nfn_gpt2_native_train --checkpoint-metadata-smoke --output-dir PATH` are `model_########.bin` files with optional matching `DONE_########` markers. Keep `nfn infer --checkpoint PATH --native-info` and `python cli/scripts/infer_gpt2.py --native-checkpoint PATH --native-info` on a Torch-free metadata path via `read_native_gpt2_checkpoint_info()`; do not route native `.bin` checkpoints into the graph-backed `.pt` loader. Prompt generation from native `.bin` checkpoints still needs a dedicated native inference executable.
- `cli/scripts/train_gpt2_evo.py` remains graph-backed because native `train_gpt2cu` does not implement NeuralFn's evo-layer loop. It is disabled by default with the other legacy TorchTrainer scripts; use `NFN_ALLOW_TORCH_TRAINING=1` only for one-off debugging while a native C++ trainer is being added. The compiled preflight `nfn_gpt2_evo_native_train --print-plan --eval-every-steps 1000 --tile-cuda-activation-dtype nvfp4` reports the AdamW/NVFP4/evo-layer schedule and remaining candidate-evaluation/mutation/loss-reduction/adoption kernels without importing Python/Torch. Run existing exported artifacts with `python cli/scripts/infer_gpt2.py --evo --prompt "..."` or `nfn infer --graph ~/NeuralFn/artifacts/gpt2_evo.json --weights ~/NeuralFn/artifacts/gpt2_evo.pt --prompt "..."`; keep `infer_gpt2.py --help` and artifact default resolution no-Torch even though actual generation is still graph-backed.
- Native trainer CE logits backward in `libnfn_native_train_tile_ops.so` uses row-wise CUDA Tile kernels for vocabularies up to 1024 and chunked row-wise kernels with reusable row-stat workspace for full GPT-class vocabularies; do not reintroduce the elementwise large-vocab fallback.
- Linear weight and bias backward in `libnfn_native_train_tile_ops.so` switch large row counts away from one serial row loop per output element. Trainer builds use cuBLAS for linear forward/dInput/dWeight and bias GEMV when `NFN_TILE_CUDA_USE_CUBLAS_LINEAR=1`; fallback builds use row-chunked tiled atomic accumulation. A future tensor-core/GEMM-grade fallback replacement is still useful for dWeight, but do not reintroduce the serial large-row reduction path.

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
