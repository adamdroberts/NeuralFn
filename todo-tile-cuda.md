# NeuralFn CUDA Tile C++ backend TODO

Goal: implement a CUDA Tile C++ kernel backend for every NeuralFn compute surface that can participate in LLM or neural-network training. This is broader than the current LLM templates: coverage is defined by the public builtin catalog in `neuralfn/builtins.py`, the module dispatch in `neuralfn/torch_backend.py::build_module`, scalar function dispatch in `build_function_module`, and optimizer/runtime math used by `TorchTrainer`.

This TODO is the authoritative checklist for the CUDA Tile backend. Keep `todo-kernels.md` as the older PyTorch-reference / kittens wishlist, but update this file for CUDA Tile implementation status.

References:

- NVIDIA blog: https://developer.nvidia.com/blog/develop-high-performance-gpu-kernels-in-cpp-with-nvidia-cuda-tile/
- CUDA Tile guide: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html

## Hard requirements

- [x] Require CUDA Toolkit 13.3 or newer for the CUDA Tile C++ build path.
- [x] Require `cuda_tile.h`, C++20, `nvcc --enable-tile`, and architecture flags for the active GPU.
- [x] Require compute capability 8.x or newer for Tile kernels; prefer SM120 when available.
- [x] Launch Tile kernels with one logical thread per tile block: `kernel<<<grid, 1>>>(...)`.
- [x] Use `ct::tensor_span`, `ct::partition_view`, `ct::shape`, and masked load/store variants for tails.
- [x] Use `__restrict__` and `ct::assume_aligned(..., 16_ic)` for pointer-heavy kernels.
- [x] Keep PyTorch as the default fallback when CUDA Tile is missing or unsupported.
- [x] Add strict mode that fails if any selected training graph node lacks CUDA Tile coverage.
- [x] Never fake coverage: every unsupported node must have an explicit host-only, delegated, or fallback reason.

## Training hot-path rule

Real training tensors must not pass through graph editor node objects.

- [x] Compile graph topology into a static execution plan before training.
- [x] Make `CompiledTorchGraph.forward()` use the precompiled plan instead of walking `NeuronGraph.nodes` and `NeuronGraph._incoming()` per batch.
- [x] Add regression coverage proving forward still works after graph edge traversal is made unavailable post-compilation.
- [x] Extend the same invariant to future CUDA Tile graph execution plans.
- [x] Add a benchmark that compares old graph-walk execution, static PyTorch execution, and CUDA Tile execution.
- [x] Add an assertion helper for tests: no training forward/backward path may read editor position, viewport, React store, or mutable graph-editor metadata.
- [x] Keep editor graph objects as control-plane data only: authoring, serialization, validation, and compile-time planning.

## Native C++ trainer ABI

This section tracks the raw no-Torch C ABI used by compiled model trainers. It is separate from the PyTorch extension bindings and autograd wrappers.

- [x] Build `libnfn_native_train_tile_ops.so` from CUDA Tile kernels without `torch/extension.h`.
- [x] Expose AdamW, gradient accumulation, reductions, in-place scaling, and linear forward through `neuralfn/csrc/native_train/tile_ops.h`.
- [x] Expose gradient/device-buffer fill through the native ABI for trainer-loop zeroing.
- [x] Expose global gradient norm clip scale finalization and device-scalar gradient scaling through the native ABI.
- [x] Expose token embedding, absolute position embedding, RMSNorm, LayerNorm, softmax, scaled dot-product attention, token CE partials, and masked token CE partials through the native ABI.
- [x] Expose token CE logits backward and masked token CE logits backward through the native ABI.
- [x] Add GPT-2 evo `--print-plan` compiled C++ preflight that reports the AdamW/NVFP4/evo-layer schedule and required candidate-evaluation kernels without Python/Torch.
- [x] Add GPT-2 compiled-CLI SDK handoff config that passes cached dataset alias/path directly to the C++ shard resolver without Python `meta.json` or token-shard validation.
- [x] Add GPT-2 native `--backend tile-cuda` / SDK `kernel_backend="tile-cuda"` preflight that reports required raw Tile ABI symbols and `--check-tile-ops` validation without Python/Torch.
- [x] Default all native Tile-CUDA entrypoints and SDK bindings to `CUDA_MODULE_LOADING=LAZY` when unset, matching the dense GPT trainer before command execution or Tile library/runtime loading.
- [x] Run GPT-2 no-data Tile-CUDA preflights before token-shard resolution so `--check-tile-ops`, synthetic smoke steps, and ABI checks work without cached datasets and report `token_shards_resolved: false`.
- [x] Add GPT-2 native Tile parameter layout and forward/backward/optimizer stage plan to the compiled `--backend tile-cuda --print-plan` JSON.
- [x] Add GPT-2 `--smoke-optimizer-step` compiled C++ path that allocates GPT-2-sized param/grad/AdamW buffers, runs one raw Tile AdamW call per registered parameter buffer, and samples copyback values without Python/Torch.
- [x] Add GPT-2 `--smoke-lm-step` compiled C++ path that runs tied token embedding, full-vocab LM logits, CE partials/backward, tied weight backward, and AdamW through raw Tile kernels without Python/Torch.
- [x] Add GPT-2 `--smoke-embedding-lm-step` compiled C++ path that samples cached uint16 tokens and runs token/position embeddings, embedding residual add, final LayerNorm, tied LM head, CE backward, embedding/norm backward, and AdamW through raw Tile kernels without Python/Torch or graph-editor payloads.
- [x] Add GPT-2 `--train-embedding-lm` compiled C++ path that runs the sampled token/position embedding, final LayerNorm, tied LM head, CE backward, embedding/norm backward, AdamW, and periodic validation loop over cached shards without Python/Torch or graph-editor payloads.
- [x] Add GPT-2 `--smoke-attention-step` compiled C++ path that runs qkv projection, QKV split, scaled dot-product attention forward/backward, QKV gradient merge, projection backward, and AdamW through raw Tile kernels without Python/Torch.
- [x] Add GPT-2 `--smoke-mlp-step` compiled C++ path that runs c_fc projection, GELU forward/backward, c_proj backward, and AdamW through raw Tile kernels without Python/Torch.
- [x] Add GPT-2 `--smoke-norm-residual-step` compiled C++ path that runs LayerNorm, scaled residual add, LayerNorm backward, gradient accumulation, and AdamW through raw Tile kernels without Python/Torch.
- [x] Add GPT-2 `--smoke-transformer-block-step` compiled C++ path that composes LayerNorm, fused QKV attention, real 12-head reshape/merge layout, residual adds, MLP, backward passes, gradient accumulation, projection bias gradients, and AdamW updates for all 12 GPT-2 block parameter buffers through raw Tile kernels without Python/Torch.
- [x] Add GPT-2 `--smoke-transformer-lm-step` compiled C++ path that samples cached uint16 tokens, preserves range-checked GPT-2 token IDs, and runs token/position embeddings, one tiny transformer block, final LayerNorm, tied LM head, CE forward/backward, transformer backward, embedding backward, and AdamW for 16 parameter buffers through raw Tile kernels without Python/Torch.
- [x] Implement GPT-2 `--train-transformer-lm` as an initial full-vocab real-dim single-layer multi-step compiled C++ loop over cached shards using token/position embeddings, one transformer block, final norm, a row-chunked tied LM-head/CE workspace, transformer backward, embedding backward, periodic validation, device-side global norm gradient clipping, and 16 AdamW parameter updates without Python/Torch.
- [x] Report GPT-2 `--train-transformer-lm` `trained_layers` / `target_layers` in JSON so trained depth is directly testable.
- [x] Report GPT-2 `--train-transformer-lm` `block_state_layout` in JSON and store trained block weights/gradients/AdamW state behind an explicit per-block C++ structure as the first step toward a 12-block array.
- [x] Drive GPT-2 `--train-transformer-lm` block parameter allocation, initialization, gradient zeroing, gradient clipping, AdamW updates, and trained-weight checkpoint export from the per-block C++ state vector instead of direct `block0` optimizer wiring/export.
- [x] Drive GPT-2 `--train-transformer-lm` block activation storage plus forward/backward execution through a per-block activation tape and block loops instead of a single inline block body.
- [x] Raise GPT-2 `--train-transformer-lm` trained block count from 1 to 12 using a scratch-recompute activation tape plus persistent block outputs instead of allocating a full tape per layer.
- [x] Add GPT-2 `--checkpoint-metadata-smoke` compiled C++ path that writes a sparse version-5 bf16 native checkpoint-format file plus `DONE_########` marker for the requested `--num-layers` target shape without Python/Torch/CUDA.
- [x] Add GPT-2 `--train-transformer-lm` final trained-weight checkpoint export in native version-5 bf16 `.bin` format with `DONE_########` marker and JSON file-size accounting.
- [x] Default dense GPT-2 Python and SDK compiled-CLI handoff to `kernel_backend="tile-cuda"` plus `--train-transformer-lm`; the explicit `llm-kittens` training backend has since been removed from CLI/SDK/C++ trainer dispatch.
- [x] Keep GPT-2 wrapper `--native-cuda-dry-run --native-cuda-print-command` metadata-only on the default compiled CLI runner, with no dataset-manager/NumPy/tiktoken/Torch imports and no raw-text shard materialization.
- [x] Remove the CLI training `NFN_ALLOW_TORCH_TRAINING=1` bypass: `nfn train` and direct `cli/scripts/train_*.py` execution now dispatch to compiled native CUDA/C++ entrypoints or fail before importing Torch, while legacy graph-backed experiments must call Python SDK trainer APIs directly.
- [x] Preserve explicit zero cadences from SDK/native GPT compiled-CLI configs (`eval_every_steps=0`, `sample_every_steps=0`, and `checkpoint_every_steps=0`) so same-script kernel benchmarks can disable validation, sampling, and checkpoint cadence without the Python handoff clamping them back on.
- [x] Pin the SM120 parity wrapper's NeuralFn candidate to `--train-batch-tokens 524288`, matching the `llm.kittens/train-sm120.sh` `-d 524288` contract in the same paired script instead of relying on native defaults.
- [x] Add `tools/bench_native_gpt_sm120_candidate.sh` as the native-vs-native SM120 bisection wrapper: it keeps the dense GPT command shape fixed on both sides, compares the current Tile ops library/default env against `NFN_SM120_NATIVE_CANDIDATE_ENV` or `NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB`, preserves the `524288` token-batch contract, and reuses the selected-GPU idle/utilization guards from `tools/paired_kernel_speed.py`.
- [x] Add `NFN_SM120_NATIVE_STARTUP_ONLY=1` to `tools/bench_native_gpt_sm120_candidate.sh` so startup-focused native candidates are compared with the same baseline-vs-candidate command shape, selected-GPU idle guard, and external-load controls as train-loop kernel bisections.
- [x] Add dense GPT native profile JSON `float_arena_request_stats` and `uint16_arena_request_stats` with ranked named suballocations so startup `cudaMalloc` candidates can be selected from measured arena contributors. The 2026-06-18 startup profile showed `setup.float_arena_materialize` at 159.510 ms for an 8.49 GB float arena and `setup.uint16_arena_materialize` at 119.426 ms for a 20.10 GB BF16/uint16 arena; the largest BF16 contributors were `stored_mlp_activation_bf16_arena` (10.87 GB), `stored_packed_attention_bf16_arena` (4.83 GB), `stored_packed_attention_ln1_bf16_arena` (1.11 GB), `stored_residual1_bf16_arena` (1.11 GB), and `lm_head_bf16_logits` (824 MB).
- [x] Add grouped arena-family profile fields (`family_count`, `top_families`, `top_family_elements`, and `top_family_bytes`) to the dense GPT native `float_arena_request_stats` and `uint16_arena_request_stats` JSON. A 2026-06-18 startup-only probe showed the float arena grouped as `transformer_lm_buffer` (3.70 GB across 42 requests) and `block.*.persistent_output` (2.21 GB across 11 requests), while the BF16 arena remained dominated by `stored_mlp_activation_bf16_arena` (10.87 GB) and `stored_packed_attention_bf16_arena` (4.83 GB). A refreshed `NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD=0` paired candidate measured `1.001432x` train-loop wall and `0.998574x` tokens/sec versus default, so no kernel default changed in this slice.
- [x] Replace the opaque dense GPT native `transformer_lm_buffer` float-arena request name with per-buffer names for the main transformer-LM globals. The 2026-06-18 startup-only profile now reports concrete float families: `block.*.persistent_output` (2.21 GB), `mlp.fc.grad_out` (805 MB), and activation-sized `attention.grad_out`, `embedding_residual`, `ln1.grad_input`, `ln2.grad_input`, and `lnf.grad_input` buffers (201 MB each), so the next startup/memory candidates no longer require source-code decoding.
- [x] Elide the dense GPT native FP32 `mlp.fc.grad_out` arena buffer when the default BF16-only MLP dGELU handoff covers every trained block. This removes `805,306,368` float-arena bytes at the default shape and reports `block_backward_mlp_fc_grad_out_float_buffer_elided: true`; the 2026-06-18 dedicated RTX 5090 same-script benchmark versus `NFN_NATIVE_GPT_ELIDE_MLP_DGELU_FLOAT_GRAD=0` measured `0.969357x` train-loop wall time, `1.031616x` tokens/sec, and `0.965683x` setup wall time.
  - Same-script llm.kittens parity after this change: the 2026-06-18 3-step dedicated RTX 5090 check measured llm.kittens at `2469.283333 ms/step` and NeuralFn at `2551.806667 ms/step`, or `1.033420x` train-loop wall time and `0.967534x` tokens/sec versus `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh`; remaining parity work is still required.
- [x] Add diagnostic opt-in `NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUTS=1` for dense GPT scratch-recompute inter-block persistent outputs. It stores earlier block outputs as BF16, restores each backward input through one FP32 scratch buffer, and reports BF16 store/restore counts plus `fp32_persistent_block_output_*_elided`; at the default shape it elides `2,214,592,512` FP32 bytes while adding `1,107,296,256` BF16 bytes plus one FP32 restore scratch. Keep it rejected as a default because the 2026-06-18 dedicated RTX 5090 same-script benchmark measured `1.021212x` train-loop wall time and `0.979238x` tokens/sec versus the current default, despite better setup (`0.974595x`) and float-arena materialization (`0.896011x`).
  - Rechecked after the latest parity baseline and keep it diagnostic-only: the 2026-06-18 startup-only 3-sample run improved setup wall to `0.965877x` and total startup to `0.966002x`, but the normal 3-step, 2-sample run regressed train-loop wall time to `1.016063x`, tokens/sec to `0.984201x`, and total wall to `1.012855x`.
- [x] Keep the next 2026-06-18 LM-head/default-route rechecks rejected after the MLP grad-out elision: `NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1` measured `1.030374x` train-loop wall time and `0.970525x` tokens/sec, `NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=0` measured `1.007590x` train-loop wall time and `0.992475x` tokens/sec, `NFN_NATIVE_GPT_STORE_MLP_BLOCKS=10` measured `1.057927x` train-loop wall time and `0.945250x` tokens/sec, and `NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_SHADOW=0` measured `1.007306x` train-loop wall time and `0.992750x` tokens/sec versus the current default. No default changed; remaining parity work should focus on new kernel work for the hot GEMM/TK buckets rather than these switches.
- [x] Keep 2026-06-18 LM-head/forward-shape retunes rejected: `--lm-head-row-chunk-size 16384` measured `1.016356x` train-loop wall time and `0.983916x` tokens/sec versus the 8192-row default over a 5-step, 3-sample same-script run, and `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=2304,65536,768,T,N` measured `1.035923x` train-loop wall time and `0.965325x` tokens/sec in a 3-step paired smoke. Keep the current LM-head chunk and TK QKV forward default.
  - Rechecked `NFN_NATIVE_LINEAR_TK_FORWARD_ENABLE_SHAPE=50304,8192,768,T,N` after the latest parity baseline. A one-step shape-stat smoke confirmed the candidate routes LM-head logits to `tk_bf16` (`220209 us` over 64 calls), but the stronger 5-step, 3-sample dedicated RTX 5090 same-script run regressed to `1.007525x` train-loop wall time and `0.992536x` tokens/sec, so it is not a promoted route.
- [x] Re-check the reference-alignment Tile ops build with `NFN_TILE_CUDA_EXTRA_NVCC_FLAGS=-DLLMK_SM120_USE_CUBLASLT_GEMM` after the MLP grad-out elision and keep the NeuralFn default Tile bridge: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.001374x` mean train-loop wall time, `0.999244x` median train-loop wall time, and `0.998638x` tokens/sec versus the default library, so the candidate remains noise-level rather than a promoted build flag.
- [x] Add diagnostic opt-in `NFN_NATIVE_GPT_ELIDE_FLOAT_PROJECTION_OUTPUTS=1` for dense GPT scratch-tape FP32 projection outputs. It skips the unused `tape.attn_proj` and `tape.mlp_out` float buffers when BF16 projection-residual is active, saving two activation-sized float allocations (`402,653,184` bytes at `64 x 1024 x 768`), but it is rejected as a default because the 2026-06-18 dedicated RTX 5090 checks measured train-loop neutral and startup-wall neutral-to-slightly slower for the elided side.
- [x] Add GPT-2 `--train-transformer-lm` CUDA runtime/driver preflight JSON and fail before allocation when the driver is unavailable or older than the loaded runtime, so SM120 benchmarking has a clear native gate.
- [x] Teach the native C++ token resolver to accept llm.kittens `TinyStories_train.bin` / `TinyStories_val.bin` directly for `--tinystories`, with `NFN_LLM_KITTENS_TINYSTORIES_DIR` override and direct train-bin sibling validation inference, so GPT-2 startup can match `train-sm120.sh` without Python dataset scanning or raw-text shard materialization.
- [x] Fuse GPT-2 `--train-transformer-lm` token/target upload into one contiguous pinned-to-device uint16 arena copy and one `nfn_native_tile_uint16_to_int64` launch per microbatch, instead of one copy and one widening launch for tokens plus another pair for targets.
- [x] Add `SequentialTokenBatchSampler::next_into()` and use it in GPT-2 `--train-transformer-lm` train/validation loops so real batches write directly into pinned uint16 arenas without `TokenBatch` vector materialization or vector-to-pinned copies.
- [x] Suballocate GPT-2 `--train-transformer-lm` widened int64 token/target buffers and compact uint16 H2D staging from one aligned device token arena, reducing two token device startup `cudaMalloc` calls to one.
- [x] Skip explicit native GPT CLI exit-time `cudaFree` calls and runtime-library `dlclose()` by default with `NFN_NATIVE_GPT_SKIP_EXIT_CUDA_FREE=1`, relying on CUDA process teardown after JSON/checkpoint output and reporting `device_exit_cuda_free_elision_enabled`, `device_exit_cuda_free_skipped_count`, `runtime_library_dlclose_skipped_count`, and `timing.cleanup_wall_ms`; keep `NFN_NATIVE_GPT_SKIP_EXIT_CUDA_FREE=0` for explicit-free diagnostics. The 2026-06-18 dedicated RTX 5090 same-script checks measured startup-only total wall at `0.695283x` and one-step training total wall at `0.946923x` versus explicit frees, with train-loop time neutral at `0.999895x`.
- [x] Replace GPT-2 `--train-transformer-lm` startup per-buffer zero fills for zero biases and AdamW state with one float-arena zero fill, eliding 369 zero-fill launches at the default 12-layer shape.
- [x] Fuse GPT-2 `--train-transformer-lm` nonzero constant parameter initialization through `nfn_native_tile_fill_many_values_float32`, reducing the default 12-layer startup path from 75 per-buffer nonzero fill launches to one descriptor-driven Tile launch.
- [x] Fuse GPT-2 `--train-transformer-lm` AdamW updates through `nfn_native_tile_adamw_step_many_with_device_scale_float32`, reducing the default 12-layer optimizer step from 148 per-buffer AdamW launches to one multi-buffer launch.
- [x] Fuse GPT-2 `--train-transformer-lm` accumulation-gradient zeroing through `nfn_native_tile_fill_many_float32`, reducing the default 12-layer optimizer-step zeroing path from 148 per-buffer fill launches to one multi-buffer launch.
- [x] Fuse GPT-2 `--train-transformer-lm` gradient-clipping sumsq partial generation through `nfn_native_tile_sumsq_partials_many_float32`, reducing the default 12-layer optimizer-step clipping path from 148 per-buffer sumsq launches to one multi-buffer launch before the device clip-scale reduction.
- [x] Wire GPT-2 `--train-transformer-lm` opt-in BF16 QKV/MLP-FC dWeight staging to direct BF16-gradient clipping and BF16-primary AdamW descriptors, eliminating the staging flush when the profiling switch is enabled; keep it default-off because the same-script dedicated-RTX-5090 benchmark measured the direct BF16 candidate slower than the optimized float-gradient path.
- [x] Default GPT `--train-transformer-lm` token embedding/LM-head startup to the fast CUDA Tile power-of-two deterministic initializer, keep `NFN_NATIVE_GPT_TOKEN_WEIGHT_INIT_LEGACY_MOD17=1` for paired modulo-17 bisection, and report the selected initializer in runtime JSON.
- [x] Match llm.kittens dense GPT dWeight accumulation semantics by adding beta-capable Tile-CUDA dWeight ABI variants and making the first gradient-accumulation microbatch write with GEMM `beta=0`, then accumulate later microbatches with `beta=1`; keep `NFN_NATIVE_GPT_DWEIGHT_FIRST_MICROBATCH_BETA_ZERO=0` for paired bisection and report the active first-write-then-accumulate strategy in runtime JSON.
- [x] Fix the chunked tied LM-head variant of that first-write path so only the first LM-head row chunk of the first gradient-accumulation microbatch uses `beta=0`; later chunks now use `beta=1` and accumulate into `accum_grad_token_weight`, with runtime JSON reporting `lm_head_dweight_beta_zero_scope`.
- [x] Keep direct first-write cuBLASLt `BGRADB` bias-gradient output as an opt-in diagnostic rather than the default: the 2026-06-17 dedicated RTX 5090 same-script 3-sample check measured `NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1` at `0.999871x` train-loop wall time and `1.000129x` tokens/sec, but the 2026-06-18 re-check measured `1.000529x` train-loop wall time and `0.999486x` tokens/sec versus the current scratch-accumulate default. Use `NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1`, `NFN_NATIVE_GPT2_BGRAD_FIRST_WRITE_DIRECT=1`, or `NFN_TILE_CUDA_LINEAR_BGRAD_FIRST_WRITE_DIRECT=1` only for paired bisection.
- [x] Reuse one BF16 pack of the dense GPT MLP projection incoming gradient for both MLP projection dWeight+bias and fused dInput+dGELU through `nfn_native_tile_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_only_float32`; keep `NFN_NATIVE_GPT_REUSE_MLP_PROJ_BF16_GRAD_OUT=0` for paired bisection and report the active reused-BF16-grad-out strategy in runtime JSON.
- [x] Allow trainer-facing BF16/BF16 cuBLASLt GEMMs for larger dense GPT LM-head backward chunk shapes by default; keep `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_LARGE_SHAPES=0` or `NFN_NATIVE_LINEAR_BF16_CUBLASLT_LARGE_SHAPES=0` for paired bisection against the previous BF16 `cublasGemmEx` fallback.
- [x] Fuse GPT-2 `--train-transformer-lm` QKV projection split plus Q/K/V head reshape through `nfn_native_tile_split_qkv_to_heads_float32`, reducing the default forward layout path from four launches per block to one.
- [x] Fuse GPT-2 `--train-transformer-lm` SDPA backward Q/K/V head-gradient merge plus QKV gradient assembly through `nfn_native_tile_merge_heads_to_qkv_float32`, reducing the default backward layout path from four launches per block to one and removing the full trainer's row-major `grad_q`, `grad_k`, and `grad_v` scratch buffers.
- [x] Add `nfn_native_tile_scaled_dot_product_attention_backward_from_merged_grad_float32` and use it in GPT-2 `--train-transformer-lm` so SDPA backward reads row-major attention-output gradients directly, removing the pre-backward `reshape_heads` launch and `grad_attn_heads` scratch buffer from the full trainer.
- [x] Expose `nfn_native_tile_attention_tk_store_forward_workspace_bf16` and `nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_saved_tk_bf16_from_merged_grad_float32` so native trainer loops can store TK BF16 Q/K/V/O/LSE attention forward state per block and later run attention backward without graph-editor tensors or an extra forward recompute.
- [x] Wire GPT-2 `--train-transformer-lm` to the saved TK attention path behind `NFN_NATIVE_GPT2_STORE_ATTENTION_ACTIVATIONS=1`, with JSON counters for saved attention arena size and store/restore/backward use; keep it off by default because the workstation 64x1024 probe regressed from about 74.4k to about 12.6k tok/s.
- [x] Suballocate GPT-2 `--train-transformer-lm` AdamW, gradient-zero, gradient-clip, and parameter-fill descriptor tables from one device descriptor arena, reducing ten small startup descriptor `cudaMalloc` calls to one.
- [x] Pack GPT-2 `--train-transformer-lm` descriptor tables into one host descriptor arena and upload it with one H2D copy, reducing ten startup descriptor `cudaMemcpy` calls to one.
- [x] Live-validate GPT-2 `--train-transformer-lm` memory/runtime behavior at the SM120 default batch shape and compare throughput against `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh`.
  - 2026-06-16 dedicated RTX 5090 check: `tools/bench_native_gpt_sm120_parity.sh` with `NFN_SM120_PARITY_STEPS=10`, `NFN_SM120_PARITY_SAMPLES=1`, `CUDA_VISIBLE_DEVICES=0`, and idle-GPU guards wrote `/tmp/nfn_sm120_parity_10step_current.json`. llm.kittens averaged `2478.839 ms/step` and `212051.9 tok/s`; NeuralFn averaged `2735.720 ms/step` and `191646 tok/s`, or `1.103630x` train-loop time and `0.903769x` tokens/sec versus the reference.
  - 2026-06-17 dedicated RTX 5090 check after large-BF16 cuBLASLt and MLP grad-out reuse: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=15 bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2475.020 ms/step` and `212201.5 tok/s`; NeuralFn measured `2606.400 ms/step` and `201154 tok/s`, or `1.053082x` train-loop time and `0.947939x` tokens/sec versus the reference. The remaining gap is concentrated in `block_backward`, `lm_head_backward`, and `train.model_forward` stage buckets.
  - 2026-06-17 workflow update: the same parity wrapper now appends NeuralFn native `--profile-json` sidecars through `NFN_SM120_PARITY_PROFILE_DIR` by default without enabling CUDA-event stage timing, so same-script parity runs keep the measured command timing-neutral. Set `NFN_SM120_PARITY_STAGE_TIMING=1` for explicit attribution sidecars that carry `timing.stage_timing` buckets.
  - 2026-06-17 unprofiled dedicated RTX 5090 check: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=15 NFN_SM120_PARITY_PROFILE_DIR=none bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2477.527 ms/step` and `211956.7 tok/s`; NeuralFn measured `2611.690 ms/step` and `200747 tok/s`, or `1.054152x` train-loop time and `0.947113x` tokens/sec versus the reference.
  - 2026-06-17 refreshed dedicated RTX 5090 check after SDK runner cleanup: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=15 NFN_SM120_PARITY_PROFILE_DIR=/tmp/nfn_sm120_parity_profiles_after_auto_guard bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2474.848 ms/step` and `212204.6 tok/s`; NeuralFn measured `2614.030 ms/step` and `200567 tok/s`, or `1.056239x` train-loop time and `0.945159x` tokens/sec versus the reference. The refreshed profile still points at `block_backward` (`13016.7 ms`), `lm_head_backward` (`6361.27 ms`), `block_backward.attn_sdpa` (`2785.26 ms`), `block_backward.mlp_fc` (`2708.38 ms`), and `block_backward.qkv` (`2053.33 ms`) as the highest-value remaining buckets.
  - 2026-06-17 follow-up: the refreshed sidecar run above used the old harness behavior that implicitly enabled `NFN_NATIVE_GPT_STAGE_TIMING=1`, while a plain 5-step NeuralFn run without stage timing measured `174287 tok/s` with `stage_timing_enabled: false`. Re-run parity after this harness correction before treating sidecar-enabled throughput as canonical.
  - 2026-06-17 dedicated RTX 5090 post-alias-guard parity check with no sidecar profiling: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=15 NFN_SM120_PARITY_PROFILE_DIR=none bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2567.572 ms/step` and `206676.9 tok/s`; NeuralFn measured `2692.620 ms/step` and `194713 tok/s`, or `1.048703x` train-loop time and `0.942113x` tokens/sec versus the reference.
  - 2026-06-17 one-step stage/shape profile after that parity run wrote `/tmp/nfn_current_stage_shape_after_alias_guard.json`; the hot buckets remain `block_backward` (`1385.720 ms`), `lm_head_backward` (`723.511 ms`), `train.model_forward` (`684.852 ms`), `block_backward.mlp_proj` (`376.063 ms`), `block_backward.attn_sdpa.to_qkv` (`285.569 ms`), and `block_backward.qkv` (`214.069 ms`). Linear-shape counters still show 96-call transformer GEMMs plus 64-call LM-head chunks: TK BF16 forward shapes `2304x65536x768`, `768x65536x768`, `3072x65536x768`, `768x65536x3072`, LM-head `50304x8192x768`, cuBLASLt dWeight/dInput shapes for the same projections, and GEMMEx LM-head dHidden.
  - 2026-06-17 explicit-batch parity confirmation after pinning the NeuralFn candidate to `--train-batch-tokens 524288`: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=3 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=15 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_explicit_batch_3sample.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2626.415667 ms/step` and `199762.7 tok/s`; NeuralFn measured `2714.493333 ms/step` and `193199.666667 tok/s`, or `1.033656x` train-loop time and `0.967400x` tokens/sec versus the reference. The selected GPU reported zero compute processes for every paired sample.
  - 2026-06-17 post-4096-token-init retile parity check with no sidecar profiling: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=15 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_after_retile.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2555.070 ms/step` and `205539 tok/s`; NeuralFn measured `2676.750 ms/step` and `195867 tok/s`, or `1.047623x` train-loop time and `0.952943x` tokens/sec versus the reference.
  - 2026-06-17 post-MLP-residual-next-LN1-fusion parity check with no sidecar profiling: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=3 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=30 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_after_mlp_next_ln1_3sample.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2616.212333 ms/step` and `200607.766667 tok/s`; NeuralFn measured `2713.243333 ms/step` and `193238.666667 tok/s`, or `1.037126x` train-loop time and `0.963323x` tokens/sec versus the reference. The selected GPU reported zero compute processes for every paired sample.
  - 2026-06-18 attention-backward section profile with `NFN_NATIVE_GPT_ATTENTION_BACKWARD_SECTION_TIMING=1` over one optimizer step reported `attention_backward_tk_timing_us: 234948`, `attention_backward_dprep_timing_us: 30604`, and `attention_backward_tk_launch_count: 96`; `block_backward.attn_sdpa.to_qkv` remained `282.329 ms`, so the current SDPA-backward bucket is dominated by the TK backward kernel rather than host graph/tensor flow.
  - 2026-06-18 post-diagnostic-default refresh with the dedicated RTX 5090 idle and no sidecar profiling: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=30 NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_after_diag_defaults_20260618.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2445.644 ms/step` and `214655.2 tok/s`; NeuralFn measured `2549.320 ms/step` and `205658 tok/s`, or `1.042392x` train-loop time and `0.958085x` tokens/sec versus the reference. A current attribution run with `NFN_NATIVE_GPT_STAGE_TIMING=1 NFN_NATIVE_GPT_LINEAR_SHAPE_STATS=1` wrote `/tmp/nfn_current_stage_shape_20260618_after_diag_defaults.json` and showed the largest buckets as `block_backward` (`1277.15 ms`), `lm_head_backward` (`629.661 ms`), `train.model_forward` (`631.122 ms`), `block_backward.mlp_proj` (`342.920 ms`), `block_backward.mlp_fc` (`265.076 ms`), and `block_backward.attn_sdpa.to_qkv` (`263.480 ms`).
  - 2026-06-18 post-native-only-CLI refresh with the dedicated RTX 5090 idle and no sidecar profiling: `NFN_SM120_PARITY_STEPS=3 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=30 NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_after_cli_native_only_20260618.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2465.030 ms/step` and `212701 tok/s`; NeuralFn measured `2548.513333 ms/step` and `205723 tok/s`, or `1.033867x` train-loop time and `0.967193x` tokens/sec versus the reference. The selected GPU reported zero compute processes and `0%` utilization before the paired sample.
  - 2026-06-18 one-step stage/shape profile after the native-only CLI change wrote `/tmp/nfn_current_stage_shape_20260618_after_cli_native_only.json`; the hot buckets remain `block_backward` (`1291.990 ms`), `train.model_forward` (`639.758 ms`), `lm_head_backward` (`634.402 ms`), `block_forward` (`632.120 ms`), `block_backward.mlp_proj` (`347.437 ms`), `block_forward.attention` (`294.776 ms`), `block_backward.mlp_fc` (`269.836 ms`), `block_backward.attn_sdpa.to_qkv` (`266.074 ms`), and `lm_head_backward.logits` (`221.139 ms`).
  - 2026-06-18 current attention-section profile after the BF16-output cuBLASLt and BF16/BF16 split-BGRADB diagnostics wrote `/tmp/nfn_attention_section_current_20260618.json`; one optimizer step reported `attention_backward_tk_timing_us: 237105`, `attention_backward_dprep_timing_us: 31238`, and `attention_backward_tk_launch_count: 96`. The largest stage buckets were `block_backward` (`1303.880 ms`), `lm_head_backward` (`737.801 ms`), `train.model_forward` (`635.762 ms`), `block_backward.mlp_proj` (`345.095 ms`), `block_backward.attn_sdpa.to_qkv` (`280.175 ms`), `block_backward.mlp_fc` (`266.960 ms`), `lm_head_backward.logits` (`259.179 ms`), and `lm_head_backward.dhidden` (`236.576 ms`).
  - 2026-06-18 MLP projection attribution now records `block_backward.mlp_proj.grad_out_bf16` inside `NFN_NATIVE_GPT_STAGE_TIMING=1` profiles. The one-step dedicated RTX 5090 profile `/tmp/nfn_mlp_grad_out_timing_20260618.json` measured the BF16 pack at `22.774 ms` total (`96` calls, `0.237 ms` average), versus `174.294 ms` for `block_backward.mlp_proj.dweight_bias` and `172.114 ms` for `block_backward.mlp_proj.dinput`, so the projection-gradient bridge is measurable but the remaining MLP gap is still dominated by GEMM/TK kernels.
  - 2026-06-18 `NFN_TILE_CUDA_EXTRA_NVCC_FLAGS=-DLLMK_SM120_ATOMIC_DQ` now builds through a dedicated packed-QKV candidate wrapper that allocates float dQ scratch, zeroes it per batch chunk, launches the internal SM120 atomic-dQ backward, and re-packs the Q slice into the BF16 packed `dQKV` scratch before QKV dWeight handoff. The one-step dedicated RTX 5090 profile regressed TK backward timing to `597872 us` versus the default roughly `237105 us`, and the same-script 5-step, 2-sample candidate benchmark measured `1.134435x` train-loop wall time and `0.881527x` tokens/sec versus default. Keep the non-atomic packed-gradient backward as the default.
- [x] Default dense GPT native forward to fuse MLP projection residual into the next block's LN1 stats/BF16 output through `nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_bf16_norm_float32`, with `NFN_NATIVE_GPT_FUSE_MLP_RESIDUAL_NEXT_LN1=0` for bisection. The 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured `0.995763x` train-loop wall time and `1.004256x` tokens/sec versus opt-out; one-step stage probe reported `mlp_residual_next_ln1_fusion_count: 88`.
- [x] Add and reject default-off MLP projection backward order bisection (`NFN_NATIVE_GPT_MLP_PROJ_DINPUT_BEFORE_DWEIGHT=1`). It mirrors the llm.kittens `matmul_backward` dInput-before-dWeight consumer order for the dense GPT MLP projection, but the 2026-06-18 dedicated RTX 5090 5-step, 3-sample same-script run measured `1.000405x` train-loop wall time and `0.999602x` tokens/sec versus the current dWeight+bias-first default, so it remains diagnostic-only.
- [x] Add and reject default-off MLP FC backward order bisection (`NFN_NATIVE_GPT_MLP_FC_DINPUT_BEFORE_DWEIGHT=1`). It mirrors the llm.kittens dInput-before-dWeight consumer order for dense GPT MLP FC, but the 2026-06-18 dedicated RTX 5090 5-step, 3-sample same-script run measured `1.000858x` train-loop wall time and `0.999153x` tokens/sec versus the current dWeight+bias-first default, so it remains diagnostic-only.
- [x] Add and reject default-off attention projection backward order bisection (`NFN_NATIVE_GPT_ATTN_PROJ_DINPUT_BEFORE_DWEIGHT=1`). It mirrors the llm.kittens dInput-before-dWeight consumer order for dense GPT attention projection, but the 2026-06-18 dedicated RTX 5090 5-step, 3-sample same-script run measured `1.001009x` train-loop wall time and `0.999002x` tokens/sec versus the current dWeight+bias-first default, so it remains diagnostic-only.
- [x] Add and reject default-off attention dprep-only BF16 grad-out bisection (`NFN_NATIVE_GPT_BF16_ATTENTION_DPREP_GRAD_OUT=1`). The candidate keeps attention projection dInput on the default float output path, packs dO to BF16 just before packed-attention dprep/backward, and reports `attention_backward_grad_out_dtype: "bf16-dprep-pack"`. A one-step attribution run reduced dprep timing from roughly `30.5 ms` to `24.8 ms` but added a `22.473 ms` pack, and the 2026-06-18 dedicated RTX 5090 same-script 10-step, 3-sample benchmark measured `1.007803x` train-loop wall time and `0.992260x` tokens/sec versus default, so the current float-dO dprep route remains the default.
- [x] Add and reject default-off BF16-output cuBLASLt LM-head logits bisection (`NFN_NATIVE_LINEAR_BF16_OUTPUT_CUBLASLT=1` / `NFN_TILE_CUDA_LINEAR_BF16_OUTPUT_CUBLASLT=1`). A one-step shape-stat smoke moved `50304,8192,768,T,N` to `cublaslt`, but the 2026-06-18 dedicated RTX 5090 10-step, 3-sample paired benchmark measured `1.000629x` train-loop wall time and `0.999382x` tokens/sec versus the current BF16 GEMMEx fallback, so it remains diagnostic-only.
- [x] Add and reject default-off BF16/BF16 split dWeight+bias bisection (`NFN_NATIVE_GPT_FUSE_BF16_BF16_DWEIGHT_BGRAD=0` / `NFN_TILE_CUDA_LINEAR_BF16_BF16_BGRAD=0`). The split path keeps block dWeight on the GEMM route and separates bias reduction instead of falling to tiled dWeight, but the 2026-06-18 dedicated RTX 5090 10-step, 3-sample paired benchmark measured `1.033067x` train-loop wall time and `0.968003x` tokens/sec versus the current fused BGRADB default.
- [x] Keep rejected same-script SM120 kernel bisections documented so future work does not retest slower paths: `NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1` (`1.016116x` train-loop time), `NFN_NATIVE_GPT_BF16_BLOCK_DWEIGHT_STAGING=1` (`1.028320x`), `NFN_NATIVE_GPT_DWEIGHT_FIRST_MICROBATCH_BETA_ZERO=0` (`1.009501x`), compile-time `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` (`0.999900x`, noise-equivalent over four samples), `NFN_NATIVE_GPT_FUSE_QKV_BIAS_TK_GEMM=0` (`1.016504x`), `NFN_NATIVE_GPT_BF16_QKV_DWEIGHT=0` (`1.009870x` train-loop time and `0.990246x` tokens/sec, slower), `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX=0` (`1.006262x`), `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX=2` (`1.007752x`), `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX=3` (`1.005321x`), `NFN_NATIVE_LINEAR_CUBLASLT_DESCRIPTOR_CACHE=0` (`1.003259x`), `NFN_NATIVE_LINEAR_TK_FLOAT_OUT=1` (`0.999929x`, noise-equivalent), `NFN_NATIVE_GPT_CE_BF16_THREADS=512` (`0.999774x`, noise-equivalent), `NFN_NATIVE_GPT_CE_BF16_THREADS=256` (`1.011218x`), `NFN_NATIVE_GPT_CE_BF16_VEC_STORES=1` (`1.000271x`, noise-equivalent/slightly slower), `NFN_NATIVE_GPT_CE_BF16_EXP2=1` (`1.000721x` train-loop time and `0.999293x` tokens/sec, noise-equivalent/slightly slower), `NFN_NATIVE_GPT_BF16_LM_HEAD_LOSS=0` (`0.992803x` mean but `1.000689x` median train-loop time over five samples, not promoted), a full LM-head BF16 logit/dlogit tape prototype (`6593445888` extra logit bytes, startup fit but the one-step candidate saturated the RTX 5090 at about `31899/32607 MiB` and exceeded the useful paired-benchmark window), `NFN_NATIVE_GPT_CUDA_MEMSET_GRAD_ZERO=0` (`1.003087x` train-loop time and `0.996930x` tokens/sec, slower), `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE=0` (`1.004427x`), `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_STATS=0` (`1.008131x`), `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=11` (`1.004875x`), `NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP=128` (`0.999506x` mean but `1.001655x` median train-loop time and `1.001502x` total wall time, not promoted), `NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP=32` (`1.003150x` train-loop time and `0.996880x` tokens/sec, slower), `NFN_NATIVE_GPT_ELIDE_QKV_FLOAT_GRAD_SCRATCH=0` (`1.000265x` train-loop time and `0.999736x` tokens/sec, noise-equivalent/slightly slower), `NFN_NATIVE_GPT_STORE_MLP_BLOCKS=11` (`1.029136x`), `NFN_NATIVE_GPT_FUSE_ATTENTION_RESIDUAL_LN2=0` (`1.021641x`), `NFN_NATIVE_GPT_FUSE_LN2_BF16_OUT=0` (`1.011993x`), `NFN_NATIVE_GPT_ELIDE_LN2_BF16_NORM_FLOAT_STORE=0` (`1.004445x` train-loop time and `0.995581x` tokens/sec, slower), `NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=128` (`1.004171x`), `NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=512` (`1.022811x`), `NFN_NATIVE_GPT_LM_HEAD_ROW_CHUNK_SIZE=4096` (`0.999703x`, noise-equivalent), `--lm-head-row-chunk-size 12288` (`1.006439x` train-loop time and `0.993616x` tokens/sec, slower), `--lm-head-row-chunk-size 16384` (`1.018129x`), `--lm-head-row-chunk-size 32768` (`0.999895x` train-loop time but `1.007049x` total wall time and `1.007987x` LM-head backward, not promoted), `NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=0` (`1.009504x` train-loop time and `1.039531x` LM-head backward, slower), `NFN_NATIVE_GPT_BF16_PROJECTION_RESIDUAL=0` (`1.015959x`), `NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF=1` (`1.000141x`, noise-equivalent), `NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC=1` (`1.005702x` train-loop time and `1.125753x` setup wall time), `NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLASLT_GEMM"` (`1.014933x`, slower), `NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1` with `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX=1` for the LM-head dHidden shape (`1.028831x` train-loop time and `0.971991x` tokens/sec, slower), a primary float+uint16 startup arena candidate (`1.075212x` setup wall time and `1.047410x` total startup wall time, slower), and a narrow LM-head dHidden cuBLASLt large-`k` probe that still fell back to `cublas_gemmex_bf16` for `m=768,n=8192,k=50304`.
- [x] Reject `NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS=0`: the candidate disabled BF16 residual1 forward storage, but the 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured `1.026566x` train-loop time and `0.974147x` tokens/sec versus the default stored-residual path.
- [x] Reject `NFN_NATIVE_GPT_BF16_RESIDUAL1_LN_BACKWARD=0`: the candidate kept residual1 storage but disabled the BF16 residual LayerNorm backward consumer, and the 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured `1.014258x` train-loop time and `0.985954x` tokens/sec versus the default BF16 residual backward path.
- [x] Reject `NFN_NATIVE_GPT_LN1_BF16_QKV_FORWARD=0`: the candidate disabled the BF16 LN1-to-QKV handoff, but the 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured `1.015832x` train-loop time and `0.984423x` tokens/sec versus the default BF16 handoff.
- [x] Reject `NFN_NATIVE_GPT_FULL_ACTIVATION_TAPE=1` as a default dense GPT training strategy: the candidate allocates one activation tape per transformer block and skips backward recompute, but the 2026-06-17 dedicated RTX 5090 one-microbatch same-script run measured `61.335739x` train-loop wall time and `0.016304x` tokens/sec versus the default scratch-recompute tape, so the switch remains diagnostic-only.
- [x] Reject `NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1` as a default LM-head dHidden route: the candidate moved `m=768,n=8192,k=50304` from BF16 `cublasGemmEx` to cuBLASLt, but the 2026-06-17 dedicated RTX 5090 one-microbatch 3-sample run measured `1.021534x` train-loop wall time and `0.978930x` tokens/sec, so the large-K cap remains.
- [x] Keep `NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1` rejected for the default packed-attention backward route: a 2026-06-17 one-step stage probe reduced `block_backward.attn_sdpa.to_qkv` only from `272.560 ms` to `269.077 ms`, but slowed `block_backward.attn_proj.dinput` from `53.980 ms` to `205.680 ms` and increased total train-loop wall time from `2658.21 ms` to `2833.25 ms`.
- [x] Keep `NFN_NATIVE_GPT_REUSE_MLP_PROJ_BF16_GRAD_OUT=0` rejected for the default MLP projection backward route: a 2026-06-17 one-step stage probe improved `block_backward.mlp_proj.dweight_bias` from `173.413 ms` to `170.561 ms`, but slowed `block_backward.mlp_proj.dinput` from `176.018 ms` to `196.379 ms` and left total train-loop wall time neutral-to-slower (`2658.21 ms` to `2658.65 ms`).
- [x] Add `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=m,n,k,opA,opB,index` / `NFN_TILE_CUDA_CUBLASLT_HEURISTIC_SHAPE=m,n,k,opA,opB,index` for one-shape cuBLASLt heuristic bisection, and reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,2304,65536,N,T,0` as a default QKV dWeight route: the 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured `0.999825x` mean train-loop wall time but `1.001401x` median train-loop wall time versus the default heuristic selection.
- [x] Add and reject the cuBLASLt waves-policy selector as a default dense GPT route: `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_POLICY=min_waves` matches the llm.kittens lowest-`wavesCount` selector but measured `1.001205x` train-loop wall time and `0.998809x` tokens/sec in the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample benchmark, while `max_waves` measured `1.001045x` train-loop wall time and `0.998964x` tokens/sec. A post-atomic-route current build check reconfirmed `min_waves` as slower at `1.009572x` train-loop wall time and `0.990522x` tokens/sec in `/tmp/nfn_cublaslt_min_waves_pair_20260618.json`. Explicit shape/global index overrides still win, and the NeuralFn default remains index 1.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,2304,65536,N,T,2` as a default QKV dWeight route: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.001121x` train-loop wall time and `0.998895x` tokens/sec versus the current cuBLASLt heuristic selection.
- [x] Keep `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=3072,768,65536,N,T,0` diagnostic-only for MLP projection dWeight: the 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured only a small `0.998065x` mean and `0.998219x` median train-loop wall-time ratio, and the 2026-06-18 follow-up 3-step, 2-sample check measured `0.998915x` train-loop wall time with `1.001091x` tokens/sec. The measured delta stays within run-to-run noise, so the default cuBLASLt heuristic route is unchanged.
- [x] Keep `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=3072,768,65536,N,T,2` diagnostic-only for MLP projection dWeight: the 2026-06-18 dedicated RTX 5090 same-script 3-step, 2-sample check measured only `0.998753x` train-loop wall time and `1.001251x` tokens/sec, again inside noise and not enough to replace the default cuBLASLt heuristic.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,3072,65536,N,T,0` as a default MLP FC dWeight route: the first 3-sample run was noisy, and the 2026-06-17 dedicated RTX 5090 5-sample confirmation measured `1.015160x` train-loop wall time and `0.985688x` tokens/sec versus the default heuristic selection.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,50304,8192,N,T,0` as a default LM-head dWeight route: the 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured `1.002294x` train-loop wall time and `0.997718x` tokens/sec versus the default heuristic selection.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,65536,3072,N,N,0` as a default MLP projection dInput route: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.000778x` train-loop wall time, `1.006005x` median train-loop wall time, and `0.999289x` tokens/sec versus the default shape heuristic.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,65536,3072,N,N,4` as a default MLP projection dInput route: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.003512x` train-loop wall time, `1.004505x` median train-loop wall time, and `0.996511x` tokens/sec versus the default shape heuristic.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,65536,3072,N,N,5` as a default MLP projection dInput route: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.002550x` train-loop wall time, `1.002451x` median train-loop wall time, and `0.997462x` tokens/sec versus the default shape heuristic.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,50304,8192,N,T,4` as a default LM-head dWeight route: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.007483x` train-loop wall time, `1.007843x` median train-loop wall time, and `0.992575x` tokens/sec versus the default shape heuristic.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,3072,65536,N,T,0` as a default MLP projection dWeight+bias route: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.012195x` train-loop wall time, `1.021403x` median train-loop wall time, and `0.988237x` tokens/sec versus the default shape heuristic.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,65536,3072,N,N,0`, `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,65536,2304,N,N,0`, and `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,65536,768,N,N,0` as default dInput routes: 2026-06-17 one-step stage probes were noise-level or slowed the targeted `mlp_proj.dinput`, `qkv.dinput`, and `attn_proj.dinput` child stages, so the global default heuristic remains selected for these `N,N` shapes.
- [x] Reject quick `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=3072,65536,768,N,N,0` and `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=3072,65536,768,N,N,4` probes as default MLP-FC dInput routes: the 2026-06-18 dedicated RTX 5090 same-script 3-step single-sample checks measured `1.001214x` and `1.004334x` train-loop wall time respectively, with `0.998787x` and `0.995686x` tokens/sec versus the current default.
- [x] Reject quick `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,65536,2304,N,N,4` and `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,65536,768,N,N,4` probes as default QKV/attention-projection dInput routes: the 2026-06-18 dedicated RTX 5090 same-script 3-step single-sample checks measured `1.001084x` and `1.006493x` train-loop wall time respectively, with `0.998917x` and `0.993546x` tokens/sec versus the current default.
- [x] Add GPT-prefixed aliases for native linear shape stats so future CUDA Tile kernel bisections can use either `NFN_NATIVE_LINEAR_SHAPE_STATS=1`, `NFN_TILE_CUDA_LINEAR_SHAPE_STATS=1`, `NFN_NATIVE_GPT_LINEAR_SHAPE_STATS=1`, or `NFN_NATIVE_GPT2_LINEAR_SHAPE_STATS=1`; a 2026-06-18 one-step profile with the existing linear alias reported the current 15 hot GEMM buckets, including TK BF16 LM-head logits, GEMMEx LM-head dHidden, cuBLASLt LM-head dWeight, and the 96-call transformer block forward/backward shapes.
- [x] Extend native linear shape stats with opt-in CUDA-event timing for kernel bisection: `nfn_native_tile_trainer_linear_shape_stats_entry` now returns `total_us`, native GPT runtime JSON reports `linear_shape_stats[].total_us` and `avg_us`, and the stats mode synchronizes measured GEMMs only when shape stats are enabled.
- [x] Promote the measured fallback for the padded LM-head logits shape by disabling TK forward for `50304,8192,768,T,N` by default; the 2026-06-18 dedicated RTX 5090 same-script 10-step, 3-sample confirmation measured `0.990336x` train-loop wall time and `1.009770x` tokens/sec versus the old TK route. Use `NFN_NATIVE_LINEAR_TK_FORWARD_ENABLE_SHAPE=50304,8192,768,T,N` to restore the old route for bisection.
  - Same-script llm.kittens parity after this promotion: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=3 NFN_SM120_PARITY_PROFILE_DIR=none ... bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2502.045 ms/step` and `209661.7 tok/s`; NeuralFn measured `2565.570 ms/step` and `204362.3 tok/s`, or `1.025419x` train-loop wall time and `0.974765x` tokens/sec versus the reference. Remaining parity work is still required.
- [x] Add and reject the default-off `NFN_NATIVE_LINEAR_TK_DINPUT=1` / `NFN_TILE_CUDA_LINEAR_TK_DINPUT=1` BF16-gradient/BF16-weight dInput diagnostic for the LM-head dHidden shape. The one-step shape-stat smoke moved `768,8192,50304,N,N` from GEMMEx (`236725 us` bucket time in the prior smoke) to TK (`215821 us` bucket time), but the dedicated RTX 5090 same-script 5-step, 3-sample wall benchmark measured `1.049216x` train-loop time and `0.953102x` tokens/sec versus the default route, so GEMMEx remains the default.
- [x] Fold dense GPT native train-loss collection into the LM-head backward recompute pass. Training microbatches now skip the separate forward LM-head loss pass when train-loss recording is enabled, accumulate CE loss from the row-chunked logits already recomputed for CE backward, and then overwrite those logits with dLogits. Validation and evo candidate scoring stay on the forward-only LM-head loss path. Verification: rebuilt `build/nfn_gpt_native_train`, passed the focused native GPT test slice, and ran the GPU-visible SM120 parity harness. The parity harness keeps train-loss recording disabled for timing-only parity and still measured NeuralFn at `1.036929x` train-loop wall time versus llm.kittens, so this is a workflow/logging improvement rather than the remaining throughput fix.
- [x] Expose native dense GPT train-loss cadence as `--train-loss-every-steps N` with `--train-log-every` / `--train-log-every-steps` aliases, defaulting to `0` for benchmark parity. SDK `NativeGptRunConfig` / `NativeGpt2RunConfig` now forward `train_loss_every_steps` to the compiled CLI so train loss can be sampled without graph-editor node flow or the old duplicate LM-head forward-loss pass.
- [x] Keep the LM-head logits cuBLASLt heuristic override rejected for `50304,8192,768,T,N`: the 2026-06-18 dedicated RTX 5090 same-script 3-step single-sample probe with `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=50304,8192,768,T,N,0` did not change the cuBLASLt/TK dispatch counters and measured `1.001453x` train-loop wall time with `0.998551x` tokens/sec versus the default BF16 GEMMEx fallback.
- [x] Reject the BF16-output cuBLASLt LM-head logits probe for `50304,8192,768,T,N`: a temporary candidate library built successfully, but the 2026-06-18 dedicated RTX 5090 one-step shape-stat smoke still reported the logits bucket as BF16 `cublasGemmEx` (`330916 us` over 64 calls), so the no-op diagnostic flag was removed instead of promoted.
- [x] Reject the older float32/TF32 LM-head route as a current default fallback: `NFN_NATIVE_GPT_LM_HEAD_BF16_LOGITS=0` measured `1.229126x` train-loop wall time and `0.813607x` tokens/sec in the 2026-06-18 dedicated RTX 5090 5-step, 3-sample same-script benchmark versus the BF16 logits/dlogits default.
- [x] Add and reject `NFN_NATIVE_GPT_LM_HEAD_DWEIGHT_BEFORE_DHIDDEN=1` as a default LM-head row-chunk order: the candidate runs dWeight before dHidden after CE writes BF16 dlogits, but the 2026-06-18 dedicated RTX 5090 5-step, 3-sample benchmark measured `1.001048x` train-loop wall time and `0.998959x` tokens/sec versus the default CE -> dHidden -> dWeight order.
- [x] Keep the one-shape TK forward gate diagnostic-only and reject disabling the `3072,65536,768,T,N` MLP FC+GELU bucket: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample benchmark with `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=3072,65536,768,T,N` measured `13.608951x` train-loop wall time and `0.073530x` tokens/sec versus the TK default, proving the scalar fallback is not a viable default route.
- [x] Reject disabling the `768,65536,3072,T,N` MLP projection forward bucket as a default route: the 2026-06-18 dedicated RTX 5090 same-script 3-step, 2-sample benchmark with `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=768,65536,3072,T,N` measured `1.010042x` train-loop wall time and `0.990063x` tokens/sec versus the current TK default.
- [x] Reject disabling the `768,65536,768,T,N` attention projection forward bucket as a default route: the 2026-06-18 dedicated RTX 5090 same-script 3-step, 2-sample benchmark with `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=768,65536,768,T,N` measured `1.009872x` train-loop wall time and `0.990230x` tokens/sec versus the current TK default.
- [x] Keep the 2026-06-18 post-LM-head-beta-fix rechecks rejected: `NFN_NATIVE_GPT_DWEIGHT_FIRST_MICROBATCH_BETA_ZERO=0` still measured slower at `1.009049x` train-loop wall time and `0.991044x` tokens/sec; `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1` was noisy and worsened token-init time to `1.044352x`; a removed `NFN_NATIVE_GPT_TOKEN_WEIGHT_PATTERN16_INIT=1` prototype also worsened token-init time to `1.033752x` mean and `1.058094x` median, so the default Tile initializer and beta-zero route remain unchanged.
- [x] Reject `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_GRID3D=1` as a default packed-attention dprep route: the diagnostic 3D batch/head/time launch removes per-row division/modulo from dprep, but the 2026-06-17 dedicated RTX 5090 5-sample confirmation measured `1.008389x` train-loop wall time and `0.991895x` tokens/sec versus the row-linear dprep default.
- [x] Promote the GPT `heads=12, head_dim=64` BF16-grad packed-attention dprep specialization by default: the 2026-06-17 dedicated RTX 5090 same-script 5-step, 3-sample candidate benchmark measured `0.997290x` mean train-loop wall time and `1.002726x` mean tokens/sec versus the older generic row dprep path; set `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_HD64_SPECIALIZED=0` to reproduce the old path.
- [x] Reject a vectorized pair-load/store variant of the HD64 dprep specialization: the 2026-06-17 dedicated RTX 5090 same-script 5-step, 3-sample candidate benchmark with `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_HD64_VEC2=1` measured `1.001076x` mean train-loop wall time and `0.998961x` mean tokens/sec versus the promoted HD64 default, so the candidate switch was removed.
- [x] Re-check `NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1` with the current HD64 dprep default and keep it as a non-promoted diagnostic for low-level Tile callers: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.000529x` train-loop wall time and `0.999486x` tokens/sec versus the current environment-default path.
- [x] Reject `NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF=1` as the current default fallback compute type: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.004222x` train-loop wall time and `0.995808x` tokens/sec versus `CUBLAS_COMPUTE_32F` for non-cuBLASLt BF16 GEMMEx paths.
- [x] Re-check and keep `NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1` rejected after the HD64 dprep default: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.011370x` train-loop wall time and `0.988829x` tokens/sec versus the default float attention-gradient handoff.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_WORKSPACE_MB=256` as a default cuBLASLt workspace cap: one-microbatch timing was slightly faster, but the 2026-06-17 dedicated RTX 5090 normal 5-step 3-sample run measured `1.000863x` train-loop wall time and `0.999150x` tokens/sec versus the 128 MiB default.
- [x] Add `NFN_NATIVE_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=m,n,k,opA,opB` / `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=m,n,k,opA,opB` for one-bucket BF16 cuBLASLt fallback bisection.
- [x] Reject `NFN_NATIVE_LINEAR_BF16_CUBLASLT=0` as a current default-route fallback: the 2026-06-17 dedicated RTX 5090 one-microbatch 3-sample run measured `6.171959x` train-loop wall time and `0.162057x` tokens/sec versus the default cuBLASLt path.
- [x] Reject `NFN_NATIVE_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=768,65536,3072,N,N` as a default fallback for MLP projection dInput: the 2026-06-17 dedicated RTX 5090 one-microbatch 3-sample run measured `1.007716x` train-loop wall time and `0.992349x` tokens/sec.
- [x] Reject `NFN_NATIVE_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=768,65536,768,N,N` as a default fallback for the smaller dInput bucket: the 2026-06-17 dedicated RTX 5090 one-microbatch 3-sample run measured `1.008810x` train-loop wall time and `0.991317x` tokens/sec.
- [x] Reject `NFN_NATIVE_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=768,3072,65536,N,T` as a default fallback for MLP projection dWeight: the 2026-06-17 dedicated RTX 5090 one-microbatch 3-sample run measured `2.857882x` train-loop wall time and `0.351320x` tokens/sec.
- [x] Do not promote `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1` as the default token-weight startup initializer: the 2026-06-17 dedicated RTX 5090 startup-only 5-sample run with the current default measured noise-equivalent `0.996666x` token init time and `0.997098x` total wall time versus the CUDA Tile initializer, so the threaded kernel remains diagnostic-only.
- [x] Align the low-level token-weight Tile ABI default with the compiled trainer/docs by making `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT` default off inside `libnfn_native_train_tile_ops.so`; the 2026-06-17 dedicated RTX 5090 startup-only 5-sample comparison against `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1` measured the corrected default at `0.940074x` token init time, `0.974488x` setup wall time, and `0.976437x` total wall time.
- [x] Keep the GPT token-weight fast int32 Tile-index initializer diagnostic-only: the candidate replaces the default power-of-two int64 Tile bucket path with an int32 Tile index path under `NFN_NATIVE_GPT_TOKEN_WEIGHT_FAST_INT32_INIT=1`, but the 2026-06-18 dedicated RTX 5090 startup-only 5-sample comparison measured the old default opt-out at `0.980751x` setup wall time, `0.984024x` token-init time, and `0.993102x` total wall time versus the int32 candidate, so the default remains the existing int64 Tile initializer.
- [x] Re-check current dense GPT parity against `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh` after the GPT evo delegation work: the 2026-06-18 dedicated RTX 5090 same-script 3-step run measured llm.kittens at `2464.486667 ms/step` and NeuralFn at `2564.563333 ms/step`, or `1.040608x` train-loop wall time and `0.960888x` tokens/sec versus the reference; startup/setup remains dominated by `setup.float_arena_materialize`, `setup.uint16_arena_materialize`, and `setup.token_weight_init`.
- [x] Keep the 2026-06-18 same-script rechecks rejected for the current default route: `NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1` measured `1.002364x` train-loop wall time, `--lm-head-row-chunk-size 16384` measured `1.017689x`, `NFN_NATIVE_GPT_CE_BF16_EXP2=1` measured `1.002993x`, `NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=128` measured `1.004624x`, and `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE=0` measured `1.003253x` versus the current default.
- [x] Reject disabling direct uint16 token IDs as a default CE path: the 2026-06-18 dedicated RTX 5090 same-script 10-step check with `NFN_NATIVE_GPT_DIRECT_U16_TOKENS=0` measured `1.007744x` train-loop wall time, `0.992317x` tokens/sec, and `1.006201x` total wall time versus the direct-u16 default.
- [x] Reject the 2026-06-18 reduced-storage startup candidate (`NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS=0 NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_ACTIVATIONS=0 NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS=0`) as a model-quality training default: it reduced setup wall time to `0.652100x` but regressed train-loop wall time to `1.457105x`, so saved activation storage remains the default while startup optimization focuses on cheaper allocation/initialization.
- [x] Retile the trainer-facing native GPT token-weight CUDA Tile initializer from the previous 2048-element default to 4096 elements, with `NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE=1024|2048|4096|8192` reserved for compile-time paired bisection. The 2026-06-17 dedicated RTX 5090 startup-only same-script run measured 4096 at `0.895736x` mean token-init time versus 2048 but noisy/slower total startup (`1.179639x`), while the one-step native training comparison measured `0.991404x` total wall time, `0.999314x` train-loop wall time, and `1.000805x` tokens/sec. The 1024 candidate was rejected after measuring `1.007101x` token-init time and `1.012266x` total wall time versus 2048. The 2026-06-18 8192 candidate compiled successfully but stayed non-default after the dedicated RTX 5090 9-sample startup-only comparison against 4096 measured `1.005585x` token-init time, with total startup `0.990436x` inside broader arena-materialization noise.
- [x] Refresh the live 10-step dedicated RTX 5090 parity baseline after the diagnostic commits: NeuralFn `build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 10 --train-batch-tokens 524288 ...` reported `202662` train tokens/sec, while `/mnt/disk2/dev/open-source/llm.kittens/train_gpt2cu ... -x 10` ended at `210704` tok/s and averaged about `210.8k` tok/s after warmup, putting NeuralFn at roughly `0.962x` of the current llm.kittens run.
- [x] Refresh the live 10-step dedicated RTX 5090 parity baseline after the 2026-06-18 allocation/reporting commits: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=30 NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_after_ln1_arena_20260618.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2590.436 ms/step` and `200148.7 tok/s`; NeuralFn measured `2639.410 ms/step` and `198638 tok/s`, or `1.018906x` train-loop time and `0.992452x` tokens/sec versus the reference. The selected RTX 5090 reported zero compute processes before and after the paired sample.
- [x] Confirm that parity baseline with a 3-sample dedicated RTX 5090 run: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=3 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=30 NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_after_ln1_arena_3sample_20260618.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2564.257 ms/step` and `204478.3 tok/s`; NeuralFn measured `2650.133 ms/step` and `197865.3 tok/s`, or `1.033524x` train-loop time and `0.967701x` tokens/sec versus the reference. The selected RTX 5090 reported zero compute processes before and after every paired sample.
- [x] Refresh the dedicated RTX 5090 parity baseline after the NanoGPT routing and tiled fallback commits: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=30 NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_after_nanogpt_route_20260618.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2471.984 ms/step` and `212391.2 tok/s`; NeuralFn measured `2555.440 ms/step` and `205165.0 tok/s`, or `1.033761x` train-loop time and `0.965977x` tokens/sec versus the reference. The selected RTX 5090 reported zero compute processes before and after the paired sample.
- [x] Re-check and reject `NFN_NATIVE_GPT_LM_HEAD_ROW_CHUNK_SIZE=16384` after the NanoGPT routing and tiled fallback commits: the dedicated RTX 5090 same-script 10-step, 3-sample candidate benchmark measured `1.000599x` train-loop wall time and `0.999416x` tokens/sec versus the current 8192-row default, so the remaining LM-head work stays focused on the GEMM route instead of row-chunk tuning.
- [x] Reject `NFN_NATIVE_GPT_PUBLIC_VOCAB_CE=0` as a default CE route: the 2026-06-17 dedicated RTX 5090 normal 5-step 3-sample run measured padded-vocab CE at `1.001998x` train-loop wall time and `0.998146x` tokens/sec versus the default public-vocab strided CE, so the current strided public-vocab path stays enabled.
- [x] Retire `NFN_NATIVE_GPT_REUSE_PACKED_LN2_FC_GELU=0` / `NFN_NATIVE_GPT2_REUSE_PACKED_LN2_FC_GELU=0`: the fallback path crashed the current native trainer with CUDA error 700, and the rebuilt trainer now ignores the override while keeping `reuse_packed_ln2_fc_gelu_enabled: true` and the prepacked-LN2 FC+GELU strategy.
- [x] Default `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_BF16=1` for the dense GPT native trainer: the 2026-06-17 dedicated RTX 5090 normal 5-step 3-sample run measured `0.991351x` train-loop wall time and `1.008731x` tokens/sec versus the previous saved-attention LN1 apply-stats recompute path, while adding about `1107296256` bytes of LN1 BF16 tape at the default shape.
- [x] Reject a narrow TK plain-dInput candidate for supported BF16 block dInput GEMMs: the temporary `NFN_NATIVE_LINEAR_TK_DINPUT=1` branch routed plain BF16 dInput through `matmul_dispatch_tk_ab` with BF16 scratch plus float conversion, but the 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured it at `1.064272x` train-loop wall time and `0.939630x` tokens/sec versus the default cuBLASLt dInput route, so no runtime switch was kept.
- [x] Reject a cuBLASLt BF16-output plain-dInput candidate: the temporary `NFN_NATIVE_LINEAR_BF16_DINPUT_OUT=1` branch routed BF16-grad/BF16-weight dInput GEMMs through BF16 output scratch plus float conversion, but the 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured it at `1.040670x` train-loop wall time and `0.960945x` tokens/sec versus the default float-output cuBLASLt dInput route, so no runtime switch was kept.
- [x] Reject `NFN_NATIVE_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=3072,768,65536,N,T` as a default fallback for the current MLP projection dWeight bucket: the 2026-06-17 dedicated RTX 5090 normal 5-step 3-sample run measured `3.199175x` train-loop wall time, `0.312590x` tokens/sec, and `5.406332x` `stage.block_backward.total_ms` versus the default cuBLASLt BGRADB route.
- [x] Reject `NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD=0` as a default-route fallback: the 2026-06-17 dedicated RTX 5090 normal 5-step 3-sample run measured `0.998810x` mean train-loop wall time but `1.000652x` median train-loop wall time, with `1.001197x` mean tokens/sec and unchanged hot buckets, so disabling the mixed float32-hidden/BF16-gradient BGRADB route remains a noise-equivalent diagnostic rather than a promoted default.
- [x] Reject `NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP=96` as a default packed-attention backward cap: the 2026-06-17 dedicated RTX 5090 same-script 3-sample one-step bisection against the default cap 64 measured `1.007835x` mean and `1.015260x` median train-loop wall time, with `0.992374x` mean tokens/sec, so the default cap stays 64.
- [x] Reject `--lm-head-row-chunk-size 6144` as a default LM-head chunk route: the 2026-06-17 dedicated RTX 5090 same-script 3-sample one-step bisection against the default 8192 measured `1.003477x` mean and `1.004724x` median train-loop wall time, with `0.996541x` mean tokens/sec, so the default chunk stays 8192.
- [x] Reject `NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1` plus `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,8192,50304,N,N,0` as a default LM-head dHidden route: the 2026-06-17 dedicated RTX 5090 same-script 3-sample one-step bisection measured `1.024628x` train-loop wall time and `0.975979x` tokens/sec versus the default BF16 `cublasGemmEx` dHidden route.
- [x] Add diagnostic `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_WARPS=N` / GPT2 fallback for packed-attention backward dprep row grouping, and keep the default at 3: the 2026-06-17 dedicated RTX 5090 same-script 3-sample one-step bisections measured `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_WARPS=2` at `1.002758x` train-loop wall time and `0.997257x` tokens/sec, while `=4` measured `1.001645x` mean train-loop wall time and `0.998425x` mean tokens/sec despite a faster median, so neither is promoted.
- [x] Reject `NFN_NATIVE_GPT_DIRECT_BF16_QKV_GRAD_SCRATCH=0` as a default-route fallback: the 2026-06-17 dedicated RTX 5090 normal 5-step 3-sample run measured `1.029204x` train-loop wall time and `0.971631x` tokens/sec, with `stage.block_backward.attn_sdpa.total_ms` at `1.129386x` and `stage.block_backward.qkv.total_ms` at `1.091903x` versus the direct BF16 QKV grad scratch default.
- [x] Reject `NFN_NATIVE_GPT_FUSE_LN_BACKWARD_AFFINE_RESIDUAL=0` as a default-route fallback: the 2026-06-17 dedicated RTX 5090 normal 5-step 3-sample run measured `1.012891x` train-loop wall time and `0.987285x` tokens/sec versus the fused LayerNorm affine+dInput+residual backward default.
- [x] Do not promote `NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_INIT=0` as a startup default: the 2026-06-17 dedicated RTX 5090 startup-only 5-sample run measured only `0.993382x` setup wall time and `0.993560x` total wall time versus the fused token-weight BF16 shadow initializer, with noisy sample spread.
- [x] Reject `NFN_NATIVE_GPT_COMBINED_BF16_ARENA=0` as a startup default: the 2026-06-17 dedicated RTX 5090 startup-only 5-sample run measured `0.986733x` setup wall time, but the normal 5-step 3-sample run regressed to `1.020957x` train-loop wall time, `0.979932x` tokens/sec, and `1.019949x` total wall time versus the combined BF16 arena default.
- [x] Suballocate the saved packed-attention LN1 BF16 tape from the default combined uint16 arena instead of issuing a separate BF16 `cudaMalloc`; runtime JSON now counts that tape in `uint16_arena_suballocation_count` while preserving the `NFN_NATIVE_GPT_COMBINED_BF16_ARENA=0` per-buffer fallback.
- [x] Suballocate stored MLP LayerNorm stats and saved packed-attention LN1 stats sidecars from the default float arena instead of issuing separate float `cudaMalloc` calls during startup; keep `NFN_NATIVE_GPT_FLOAT_STATS_ARENA=0` / `NFN_NATIVE_GPT2_FLOAT_STATS_ARENA=0` for paired bisection against the older sidecar allocation path.
- [x] Reject `--lm-head-row-chunk-size 65536` for the default `64 x 1024` dense GPT shape: the candidate drove the dedicated RTX 5090 to 100% utilization and about `31926 MiB / 32607 MiB` used memory without completing a 5-step candidate sample in the expected paired-benchmark window, so the current 8192-row default remains the practical LM-head chunk size.
- [x] Re-check and reject `--lm-head-row-chunk-size 16384` after the saved-LN1-BF16 default: the 2026-06-17 dedicated RTX 5090 normal 5-step 3-sample run measured `1.014421x` train-loop wall time and `0.985784x` tokens/sec versus the 8192-row default.
- [x] Reject the BF16 GEMMEx `FAST_16BF` compute-type candidate for remaining LM-head fallback shapes: the 2026-06-18 dedicated RTX 5090 paired run with `NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF=1` / `NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_FAST_16BF=1` measured `1.001097x` train-loop wall time and `0.998921x` tokens/sec versus the current default.
- [x] Route native GPT `.bin` checkpoint text prompts through the compiled sampler path from `nfn infer --checkpoint ... --prompt ...` and `python cli/scripts/infer_gpt2.py --native-checkpoint ... --prompt ...` by GPT-2-tokenizing in the lightweight wrapper and calling `nfn_gpt_native_train --sample-checkpoint ... --prompt-tokens ...`, while keeping `--native-info` metadata inspection Torch-free.
- [x] Decode successful native GPT checkpoint sampler output in the lightweight wrapper, preserving compiled JSON while printing generated token IDs and GPT-2 text without Torch or graph-editor tensors.
- [x] Add compiled C++ native GPT checkpoint inspection with `nfn_gpt_native_train --native-info --native-checkpoint PATH` / `--inspect-checkpoint PATH`, reporting shape, precision, DONE marker state, and file-size validation before CUDA, Torch, dataset, or graph-node setup.
- [x] Route native GPT `.bin` checkpoint `--prompt-tokens` requests through compiled C++ with `nfn_gpt_native_train --sample-checkpoint PATH --prompt-tokens IDS`, validating checkpoint shape, context, vocab bounds, token parsing, executing autoregressive CUDA Tile checkpoint forward passes, and returning generated token IDs without graph-editor tensor flow.
- [x] Add native GPT checkpoint tensor-layout decode with `--checkpoint-layout`, deriving tensor shapes, payload offsets, file offsets, and bounded payload samples from the checkpoint header without CUDA, Torch, datasets, or graph-editor tensor flow.
- [x] Add native GPT checkpoint payload load smoke with `--checkpoint-load-smoke`, moving a bounded bf16 checkpoint slice through CUDA memory and Tile bf16-to-float conversion without Torch, datasets, or graph-editor tensor flow.
- [x] Extend checkpoint payload load smoke with `--checkpoint-load-tensor NAME`, selecting named tensors from the decoded layout before CUDA copy/Tile conversion so native inference can prove per-weight checkpoint loads without graph-editor tensors.
- [x] Add checkpoint-backed logits smoke with `--checkpoint-logits-smoke`, loading native checkpoint embeddings/final norm and running token embedding, position embedding, residual add, final LayerNorm, and tied LM-head logits on CUDA Tile kernels without Torch or graph-editor tensors.
- [x] Add checkpoint-backed QKV smoke with `--checkpoint-qkv-smoke`, loading native checkpoint embeddings plus selected-block `ln_1` and `attn.c_attn` tensors and running embedding residual, block LayerNorm, and QKV projection on CUDA Tile kernels without Torch or graph-editor tensors.
- [x] Add checkpoint-backed attention smoke with `--checkpoint-attention-smoke`, extending the QKV smoke through split-to-heads, causal scaled-dot-product attention, and merge-heads on CUDA Tile kernels without Torch or graph-editor tensors.
- [x] Add checkpoint-backed attention residual smoke with `--checkpoint-attention-residual-smoke`, extending the attention smoke through `attn.c_proj` and residual add on CUDA Tile kernels without Torch or graph-editor tensors.
- [x] Add checkpoint-backed block smoke with `--checkpoint-block-smoke`, extending the attention residual smoke through `ln_2`, MLP fc, GELU+bias, MLP projection, and final block residual add on CUDA Tile kernels without Torch or graph-editor tensors.
- [x] Add checkpoint-backed block logits smoke with `--checkpoint-block-logits-smoke`, extending the block smoke through final LayerNorm and tied LM-head logits for the last prompt token on CUDA Tile kernels without Torch or graph-editor tensors.
- [x] Add checkpoint-backed full-stack forward logits smoke with `--checkpoint-forward-logits-smoke`, running every checkpoint GPT block, final LayerNorm, and tied LM-head logits on CUDA Tile kernels while reporting `transformer_blocks_executed: true` and keeping graph-editor tensors out of the path.
- [x] Remove the dense GPT-2 external `llm.kittens` training bridge from normal CLI, SDK, and C++ trainer dispatch. `tools/bench_native_gpt_sm120_parity.sh` remains as a same-script reference benchmark against llm.kittens, while normal training accepts only `tile-cuda`.
- [x] Add opt-in CUDA-event packed-attention backward section timing (`NFN_NATIVE_GPT_ATTENTION_BACKWARD_SECTION_TIMING=1`) so dprep and TK backward costs are reported separately in native GPT runtime JSON without using Torch, Python tensors, or graph-editor nodes.
- [ ] Close the remaining measured SM120 throughput gap between the NeuralFn-owned `libnfn_native_train_tile_ops.so` loop and the `llm.kittens` SM120 reference script.
- [x] Extend native GPT checkpoint text-prompt inference with GPT-2 tokenization in the lightweight wrapper so `.bin` checkpoint inference no longer needs the transitional external sampler bridge.
- [x] Add GPT-2 evo raw Tile-CUDA trainer ABI for device-side candidate mutation, best-loss selection, and best-candidate adoption without graph-editor tensor flow.
- [x] Add GPT-2 evo compiled C++ `--smoke-evo-kernels` path that loads the raw evo ABI, launches mutate/select/adopt on CUDA device buffers, and verifies best-candidate copyback without Python/Torch, datasets, or graph-editor payloads.
- [x] Wire the dense GPT native `--train-transformer-lm` loop to the raw layer-evo mutate/select/adopt ABI cadence behind `--layer-evo`, targeting the selected block's float32 `ln1.weight` on device and reporting `graph_editor_tensor_flow: false` in plan/runtime JSON.
- [x] Make `nfn_gpt2_evo_native_train` delegate dense GPT-2-compatible training runs to `nfn_gpt_native_train --train-transformer-lm --layer-evo`, preserving print-plan/smoke behavior and keeping the runtime JSON explicit about the native evo candidate-loss source.
- [x] Wire GPT-2 evo native layer-evolution forward-only candidate evaluation into the dense GPT trainer loop: after AdamW, the trainer mutates `block_N.ln1.weight`, reuses the current pinned training batch, allocates the same lazy float MLP scratch used by validation-only forwards, scores every candidate through native CUDA forward loss, writes losses to device memory, selects/adopts via the raw evo ABI, and reports `candidate_loss_source: "native-forward-loss-current-batch"` plus `forward_candidate_evals` without graph-editor tensor flow.
- [x] Expose NanoGPT preflight JSON with separate `available_native_kernels` and `required_native_kernels` lists.
- [x] Add NanoGPT `--check-tile-ops` compiled C++ path that `dlopen`s `libnfn_native_train_tile_ops.so` and verifies required raw ABI symbols without Python/Torch.
- [x] Add NanoGPT `--smoke-tile-ops` compiled C++ path that dynamically loads CUDA runtime, executes `nfn_native_tile_fill_float32`, and verifies device-to-host copyback without Python/Torch.
- [x] Add NanoGPT `--smoke-optimizer-step` compiled C++ path that builds the NanoGPT parameter layout, initializes contiguous param/grad/AdamW buffers, executes `nfn_native_tile_adamw_step_float32` once per registered parameter buffer, and verifies param/moment copyback without Python/Torch.
- [x] Add NanoGPT `--smoke-training-loop-step` compiled C++ path that executes gradient zeroing, synthetic gradient fill, global-norm clip scale finalization, device-scalar gradient scaling, and per-buffer AdamW over the registered parameter layout without Python/Torch.
- [x] Add NanoGPT `--smoke-lm-step` compiled C++ path that runs token embedding, tied LM-head linear logits, token CE loss/backward, tied weight backward, and AdamW update through raw native kernels without Python/Torch.
- [x] Add NanoGPT `--smoke-token-train-step` compiled C++ path that samples a real native uint16 token/target batch from cached shards, runs tied-LM forward/backward/update kernels over those IDs, and verifies sampled-batch loss, gradient, and weight update values without Python/Torch.
- [x] Add NanoGPT `--train-token-lm` compiled C++ path that runs a real multi-step tied token-embedding LM training loop over cached native token shards without Python/Torch.
- [x] Add periodic native validation loss to NanoGPT `--train-token-lm` over resolved validation token shards without Torch, Python dataset payloads, or graph-editor node data flow.
- [x] Route NanoGPT `--train-token-lm` through `nfn-native-train` and `neuralfn.native_train.run_native_train()` so CLI and SDK dispatch stay on compiled native artifacts.
- [x] Make normal NanoGPT training entrypoints (`nfn train --base-model nanogpt ...` and `python cli/scripts/train_nanogpt.py ...`) select the shared dense GPT native `--train-transformer-lm --template-name nanogpt` route before Torch imports, while explicit `--train-token-lm` still reaches the older tied token-embedding native loop for diagnostics.
- [x] Add NanoGPT `--smoke-embedding-norm-step` compiled C++ path that samples native tokens and runs token/position embeddings, residual add, LayerNorm forward/backward, tied logits, CE backward, embedding/position/norm gradients, and AdamW updates through raw native kernels without Python/Torch.
- [x] Add NanoGPT `--smoke-mlp-step` compiled C++ path that runs MLP fc projection, GELU, output projection, backward, and AdamW updates through raw native kernels without Python/Torch.
- [x] Add NanoGPT `--smoke-attention-step` compiled C++ path that runs Q/K/V projections, SDPA forward/backward, output projection, Q/K/V projection backward, and AdamW updates through raw native kernels without Python/Torch.
- [x] Add NanoGPT native parameter/gradient buffer registry and contiguous AdamW-state layout to the C++ preflight.
- [x] Add NanoGPT AdamW parameter-group metadata over the registered C++ buffers.
- [x] Add NanoGPT native execution-stage plan with ready/requires-wiring/missing-ABI status per forward, backward, and optimizer stage.
- [x] Expose scaled residual add through the raw no-Torch native trainer ABI.
- [x] Expose fused QKV split/merge through the raw no-Torch native trainer ABI so NanoGPT can use one `qkv.weight` projection, feed contiguous Q/K/V buffers into SDPA, and pack Q/K/V gradients back into the fused projection gradient.
- [x] Expose reshape-heads and merge-heads through the raw no-Torch native trainer ABI so GPT-style trainers can feed `[batch, heads, seq, head_dim]` attention kernels without PyTorch layout helpers.
- [x] Add NanoGPT `--smoke-qkv-layout-step` compiled C++ path that executes fused-QKV split and merge kernels on device buffers and verifies exact layout copyback without Python/Torch.
- [x] Add NanoGPT `--smoke-fused-qkv-attention-step` compiled C++ path that runs fused `attn.qkv.weight` projection, QKV split, SDPA forward/backward, QKV gradient merge, fused qkv weight backward, output projection backward, and AdamW updates through raw native kernels without Python/Torch.
- [x] Add NanoGPT `--smoke-transformer-block-step` compiled C++ path that composes LayerNorm, fused-QKV attention, residual adds, MLP, backward passes, gradient accumulation, and AdamW updates for one tiny transformer block through raw native kernels without Python/Torch.
- [x] Mark NanoGPT tied LM head input/weight backward as covered by the raw linear backward native ABI.
- [x] Replace the NanoGPT-sized CE backward path (`vocab <= 1024`) with row-wise kernels that compute softmax statistics once per row instead of once per output element.
- [x] Add chunked row-wise CE backward for larger vocabularies so full GPT-class vocabularies do not use the elementwise fallback.
- [x] Expose absolute position embedding backward through the native ABI.
- [x] Expose token embedding weight backward through the native ABI.
- [x] Expose LayerNorm input and affine parameter backward through the native ABI.
- [x] Expose RMSNorm input backward through the native ABI.
- [x] Expose linear input backward through the native ABI.
- [x] Expose linear weight and bias backward through the native ABI.
- [x] Expose LayerNorm affine and Linear bias gradient-accumulate native ABI variants for optimizer-step accumulation buffers.
- [x] Replace serial linear weight/bias backward row loops with row-chunked tiled atomic accumulation for large row counts.
- [x] Remove GPT-2 full-trainer per-microbatch LayerNorm affine / Linear bias scratch buffers and copy loops by writing directly into accumulation buffers.
- [x] Add trainer-build GPU GEMM fast path for native linear forward, dInput, and dWeight behind `NFN_TILE_CUDA_USE_CUBLAS_LINEAR=1` without importing Torch or the PyTorch Tile extension.
- [x] Add trainer-build GPU GEMV fast path for native linear bias backward and accumulate-bias backward behind `NFN_TILE_CUDA_USE_CUBLAS_LINEAR=1`, using a cached device ones vector initialized by a Tile fill kernel instead of the row-chunked atomic bias fallback.
- [x] Replace fallback linear weight backward reduction kernels with GEMM-grade tiled kernels for large row counts when the trainer cuBLAS path is unavailable. The fallback float32-output dWeight paths now use a shared-memory 2D tiled CUDA kernel for float32/BF16 activation and gradient combinations, while the normal trainer build still tries cuBLAS/cuBLASLt first and bias-only fallbacks keep the shared row-chunk reduction path.
- [x] Expose GELU activation forward/backward through the native ABI.
- [ ] Add dropout forward/backward native Tile ABI if nonzero dropout training is re-enabled.
- [x] Wire MLP-stage activation and projection backward through the native NanoGPT preflight/smoke path, including fc/proj input and weight backward, GELU backward, and AdamW updates.
- [x] Expose scaled dot-product attention backward through the native ABI.
- [x] Wire attention-stage QKV/output projection backward through the native NanoGPT preflight/smoke path, including fused QKV split, SDPA backward, QKV gradient merge, fused qkv backward, output projection backward, and AdamW updates.
- [x] Replace GPT-2-compatible SDPA forward scalar-output Tile launch with a value-chunked row-vector Tile attempt for `seq_k <= 1024`, reusing each query row's score/softmax across a 2-channel value chunk when CUDA accepts the row kernel and falling back to scalar Tile attention when CUDA rejects that launch.
- [x] Add GPT-2 `--train-transformer-lm` attention-forward launch telemetry and auto-disable repeated row-kernel attempts after the first CUDA launch rejection, so live runs report row attempts, row successes, row fallbacks, scalar launches, and avoid repeated failed-launch overhead.
- [x] Tune GPT-compatible SDPA forward resources so the live SM120 trainer no longer needs row-vector or scalar-launch safety fallbacks; the current packed-QKV TK path reports `attention_forward_row_launch_fallback_count: 0`, `attention_forward_scalar_launch_count: 0`, and `attention_forward_tk_launch_count` for the full loop.
- [x] Keep the older GPT-2-compatible row-vector SDPA kernel out of the hot trainer path when packed SM120 TK attention is active, so the scalar fallback remains unused on the live dedicated-RTX-5090 probe.
- [x] Cover every shipped GPT template name in the native GPT-2 training selector via `--template-name` / `--preset`, and cover custom graph selection via `--graph-file`, returning explicit native-trainer-missing JSON for unsupported templates instead of falling back to Torch or graph-editor tensor flow.
- [x] Update dense GPT-2 native dry-run/plan JSON to report `native-transformer-lm-ready` and `training_step_plan.status: "ready"` for the implemented compiled Tile-CUDA loop; `remaining_validation` now tracks closing the measured SM120 throughput gap via `tools/bench_native_gpt_sm120_parity.sh` instead of saying live validation has not happened.
- [ ] Generalize the shared dense GPT transformer-LM loop so NanoGPT-specific dimensions/dropout can be driven entirely from the selected template or graph instead of the current dense GPT trainer shape defaults. Until then, plan/runtime JSON exposes `native_geometry_contract.name: "gpt2-compatible-fixed-dense-transformer"`, `shape_source: "compiled_dense_gpt_defaults"`, and `template_geometry_dynamic: false` / `custom_graph_geometry_dynamic: false`.

## Backend scaffolding

- [x] Add `neuralfn/tile_cuda/__init__.py`.
- [x] Add `neuralfn/tile_cuda/config.py` with `TileCudaConfig`.
- [x] Add `neuralfn/tile_cuda/registry.py` with `TileKernelSpec`, `TileKernelRegistry`, and `KernelCoverageReport`.
- [x] Add `neuralfn/tile_cuda/runtime.py` with availability checks, extension loading, and fallback policy.
- [x] Add `neuralfn/tile_cuda/autograd.py` with PyTorch autograd wrappers for Tile kernels.
- [x] Add `neuralfn/csrc/tile_cuda/` for C++/CUDA Tile extension sources.
- [x] Add `neuralfn/csrc/tile_cuda/bindings.cpp` for PyTorch custom op bindings.
- [x] Add `neuralfn/csrc/tile_cuda/kernels.cu` or split files by family once the file becomes too large.
- [x] Add packaging support for optional CUDA extension builds without breaking CPU-only installs.
- [x] Add `NFN_TILE_CUDA_BUILD=1` opt-in source build path.
- [x] Add `NFN_TILE_CUDA_ARCH` override for explicit `-arch`.
- [x] Add runtime diagnostics for CUDA version, driver version, GPU CC, Tile header availability, and extension load status.

## SDK and CLI surface

- [x] Add `TorchTrainConfig.kernel_backend: Literal["auto", "torch", "tile_cuda"] = "auto"`.
- [x] Add `TorchTrainConfig.tile_cuda_strict: bool = False`.
- [x] Add `TorchTrainConfig.tile_cuda_report_path: str | None = None`.
- [x] Let `CompiledTorchGraph` choose Tile-backed stages when requested and available.
- [x] Preserve existing PyTorch stage behavior when `kernel_backend="torch"` or Tile is unavailable in auto mode.
- [x] Add `nfn train --kernel-backend {auto,torch,tile-cuda}`.
- [x] Add `nfn infer --kernel-backend {auto,torch,tile-cuda}` where inference kernels exist.
- [x] Add `nfn eval --kernel-backend {auto,torch,tile-cuda}`.
- [x] Add `--tile-cuda-strict`.
- [x] Add `--tile-cuda-report PATH`.
- [x] Add `nfn kernels list`.
- [x] Add `nfn kernels doctor`.
- [x] Add `nfn kernels bench`.
- [x] Add `nfn kernels examples`.

## Coverage gates

- [x] Generate the coverage inventory from `BuiltinNeurons.all()`.
- [x] Generate the module inventory from `build_module()` dispatch.
- [x] Generate the scalar function inventory from `build_function_module()`.
- [x] Fail coverage tests if a builtin/module/function is missing from the Tile registry.
- [x] Fail strict training if a selected graph has an uncovered node.
- [x] Produce a JSON report with kernel status, fallback reason, dtype support, and tested shapes.
- [x] Include host-only entries for source/reference/orchestration nodes.
- [x] Include delegated entries for fused kernels that cover multiple logical graph nodes.

Per-kernel done criteria:

- [x] Forward CUDA Tile kernel or delegated fused implementation.
- [x] Backward CUDA Tile kernel, autograd composition, or explicit no-grad reason.
- [x] PyTorch custom op binding.
- [x] Autograd wrapper.
- [x] Shape contract.
- [x] Dtype contract.
- [x] CPU/PyTorch parity test.
- [x] CUDA parity test.
- [x] Gradient parity test for trainable kernels.
- [x] SDK example.
- [x] CLI smoke coverage where relevant.
- [x] Docs entry.

## Scalar function kernels

- [x] `input` / `input_node`: host-only interface marker; no compute kernel.
- [x] `output` / `output_node`: host-only interface marker; no compute kernel.
- [x] `identity`: elementwise forward/backward.
- [x] `negate`: elementwise forward/backward.
- [x] `add`: elementwise binary forward/backward.
- [x] `multiply`: elementwise binary forward/backward.
- [x] `sigmoid`: elementwise forward/backward.
- [x] `relu`: elementwise forward/backward.
- [x] `tanh_neuron`: elementwise forward/backward.
- [x] `threshold`: no-grad bool-style output, explicit training limitation.
- [x] `gaussian`: elementwise `exp(-x*x)` forward/backward.
- [x] `log`: clamped log forward/backward matching `max(x, 1e-7)`.
- [x] `leaky_relu`: elementwise forward/backward.
- [x] `prelu`: constant-slope elementwise forward/backward.
- [x] `relu6`: clipped ReLU forward/backward.
- [x] `elu`: elementwise forward/backward.
- [x] `selu`: elementwise forward/backward.
- [x] `gelu`: scalar builtin GELU forward/backward.
- [x] `silu`: elementwise forward/backward.
- [x] `mish`: elementwise forward/backward.
- [x] `softplus`: stable forward/backward.
- [x] `softsign`: elementwise forward/backward.
- [x] `hard_sigmoid`: clipped linear forward/backward.
- [x] `hard_tanh`: clipped linear forward/backward.
- [x] `hard_swish`: elementwise forward/backward.
- [x] `softmax_2`: two-input stable softmax forward/backward.
- [x] `logsoftmax_2`: two-input stable log-softmax forward/backward.

## Core tensor and LLM kernels

- [x] `token_embedding`: gather forward, indexed gradient accumulation.
- [x] `linear`: matmul plus optional bias, backward for input/weight/bias.
- [x] `tied_lm_head`: linear against shared embedding weight.
- [x] `lm_head`: vocab projection.
- [x] `logit_softcap`: `softcap * tanh(logits / softcap)`.
- [x] `token_cross_entropy`: numerically stable CE reduction.
- [x] `masked_token_cross_entropy`: CE masked by response/loss mask.
- [x] `sequence_logp`: gather log-prob sums with mask.
- [x] `residual_add`: scaled residual add.
- [x] `residual_mix`: learned primary/skip scale mix.
- [x] `manifold_hyper_connection`: sigmoid beta, bounded residual mix.

## Norm, activation, and MLP kernels

- [x] `rms_norm`: RMS norm forward/backward over last dimension.
- [x] `layer_norm`: LayerNorm forward/backward.
- [x] `group_norm`: grouped norm for `[B, S, D]`.
- [x] `qk_norm`: fused Q/K RMSNorm.
- [x] `dyt`: dynamic tanh with learnable alpha, weight, bias.
- [x] `dropout`: deterministic `p=0` and inference passthrough; stochastic training mask remains on the PyTorch RNG path.
- [x] `gelu` module: tensor GELU forward/backward.
- [x] `mlp_relu2`: linear, ReLU, square, projection.
- [x] `swiglu`: three-matrix SiLU gate.
- [x] `geglu`: three-matrix GELU gate.
- [x] `reglu`: three-matrix ReLU gate.
- [x] `solu`: softmax-gated linear unit.

## Attention and position kernels

- [x] `reshape_heads`: view/transpose contract with contiguous fallback.
- [x] `merge_heads`: transpose/reshape merge.
- [x] `repeat_kv`: grouped-query KV repeat.
- [x] `rotary_embedding`: RoPE forward/backward for Q/K.
- [x] `qk_gain`: per-head Q scale.
- [x] `scaled_dot_product_attention`: causal/non-causal SDPA.
- [x] `sliding_window_attention`: local causal window SDPA.
- [x] `block_sparse_attention`: block-local plus sink/global pattern.
- [x] `streaming_attention_sinks`: recent window plus persistent sinks.
- [x] `native_sparse_attention`: deterministic NSA reference pattern first, learned sparse selector later.
- [x] `differential_attention`: dual SDPA branches plus lambda subtraction and norm.
- [x] `causal_self_attention`: fused QKV projection, QK norm, RoPE, QK gain, SDPA, output projection.
- [x] `fused_causal_attention`: fused QKV, RoPE, SDPA, output projection.
- [x] `multi_latent_attention`: MLA low-rank KV, decoupled RoPE, SDPA, output projection.
- [x] `absolute_position_embedding`: learned positional lookup.
- [x] `routed_attention_experts`: expert-routed attention path.

## KV cache and compression kernels

- [x] `kv_cache_read`: concat cache and current KV or passthrough.
- [x] `kv_cache_write`: cache output contract, no-grad/pass-through where applicable.
- [x] `kv_pca_encode`: K/V projection to compressed dim.
- [x] `kv_pca_decode`: K/V projection back to head dim.
- [x] `kv_quant_pack`: int8 pack with per-token scale.
- [x] `kv_quant_unpack`: dequantize and split packed K/V.

## Quantization and adapter kernels

- [x] `bitlinear_ternary`: ternary weight quantization and quantized activation STE.
- [x] `fp8_linear`: E4M3/E5M2 quantized weight path, amax history, STE.
- [x] `mx_linear`: MXFP4/MXFP8 block-scale quantized weight path.
- [x] `lora_linear`: base linear plus low-rank delta.
- [x] `nf4_linear`: NF4 unpack/dequant plus LoRA delta.
- [x] `randmap_adapter`: frozen down/up maps plus trainable middle and scale.
- [x] `ttt_linear`: test-time training linear stage.

## MoE and routing kernels

- [x] `router_logits`: router projection.
- [x] `auxfree_load_balancing`: biased router update without host sync.
- [x] `topk_route`: softmax, top-k, normalize, routing stats.
- [x] `expert_dispatch`: token-to-expert dispatch and weighted combine.
- [x] `expert_combine`: identity/combine contract.
- [x] `broadcast_expert_routes`: route expansion over sequence.
- [x] `broadcast_chunk_routes`: route expansion over chunk spans.
- [x] `load_balance_loss`: router density auxiliary loss.
- [x] `aux_loss_add`: scalar loss addition.
- [x] `loss_scale`: scalar loss scaling.
- [x] `route_balance_loss`: route entropy/balance objective.
- [x] `route_selection_loss`: supervised route objective.
- [x] `route_distillation_loss`: route distillation objective.

## Semantic kernels

- [x] `semantic_data_source`: host-only source contract.
- [x] `semantic_projector`: semantic/residual/topic projection.
- [x] `semantic_alignment_loss`: semantic topic CE.
- [x] `semantic_hasher`: LSH bucket hashing.
- [x] `semantic_moe_router`: semantic top-k router.
- [x] `semantic_hash_router`: hash/topic/target-aware router.
- [x] `causal_chunk_state`: causal chunk pooling/state.
- [x] `semantic_chunk_projector`: chunk semantic/residual/topic projection.
- [x] `semantic_chunk_hasher`: chunk LSH bucket hashing.
- [x] `semantic_moe_jepa_evo_router`: shared/semantic/free expert router.
- [x] `attentionless_decoder`: hash/expert output to logits.
- [x] `softmax_distillation_loss`: teacher/student distillation loss.

## JEPA, diffusion, byte, universal, and sequence kernels

- [x] `mamba`: projection, depthwise conv, SiLU gate, output projection.
- [x] `denoise_head`: diffusion denoise projection.
- [x] `mask_scheduler`: timestep-driven token masking.
- [x] `random_timesteps`: device-side random timestep generation.
- [x] `jepa_mask`: random and block mask generation.
- [x] `latent_pool`: masked mean with fallback mean.
- [x] `jepa_projector`: projector MLP.
- [x] `jepa_predictor`: predictor MLP.
- [x] `latent_mse_loss`: detached-target MSE.
- [x] `byte_patch_embed`: byte embedding plus Conv1d patch projection.
- [x] `byte_patch_merge`: nearest interpolation back to target length.
- [x] `act_halt_gate`: mean-pool halt probability.
- [x] `act_weighted_sum`: weighted sum across recurrent states.
- [x] `universal_transformer`: recurrent attention/MLP with ACT weights.

## Fine-tuning and RLHF kernels

- [x] `sft_dataset_source`: host-only source contract.
- [x] `dpo_dataset_source`: host-only source contract.
- [x] `ppo_rollout_source`: host-only source contract.
- [x] `reference_forward`: delegated compiled graph call, no standalone Tile kernel.
- [x] `reward_forward`: delegated compiled graph call, no standalone Tile kernel.
- [x] `dpo_pairwise_loss`: sigmoid, hinge, IPO variants.
- [x] `reward_head`: pooled scalar head.
- [x] `preference_bce_loss`: Bradley-Terry BCE.
- [x] `value_head`: per-token value projection.
- [x] `ppo_clipped_loss`: clipped policy/value loss.
- [x] `kl_penalty`: reward shaping by KL.
- [x] `gae_compute`: reverse-time GAE scan.

## Optimizer and training runtime kernels

- [x] `Muon._zeropower_via_newtonschulz5`: Tile implementation for matrix updates.
- [x] `Muon.step`: fused momentum and Newton-Schulz update where practical.
- [x] AdamW update kernels for normal optimizer profile.
- [x] Split optimizer profile kernels for embedding/head/matrix/scalar parameter groups.
- [x] Gradient accumulation add-into-buffer kernels.
- [x] Gradient clipping norm and scale kernels.
- [x] EMA target update kernels for JEPA objectives.
- [x] Route-evolution evaluation path audit for Tile compatibility or explicit fallback.
- [x] GPT-2 compiled C++ `--smoke-tile-ops` path that loads raw Tile ops, launches `nfn_native_tile_fill_float32`, copies back, and reports JSON without Python/Torch.
- [x] GPT-2 compiled C++ `--smoke-optimizer-step` path over the registered GPT-2 parameter layout.
- [x] GPT-2 compiled C++ `--smoke-lm-step` path over a tiny tied embedding/LM-head forward/backward/update slice.
- [x] GPT-2 compiled C++ `--smoke-embedding-lm-step` path over sampled cached uint16 tokens, token/position embeddings, final norm, tied LM head, CE backward, embedding/norm backward, and AdamW.
- [x] GPT-2 compiled C++ `--train-embedding-lm` partial native loop over cached shards with periodic validation losses.
- [x] GPT-2 compiled C++ `--smoke-attention-step` path over a tiny model-dim qkv/SDPA/projection backward/update slice.
- [x] GPT-2 compiled C++ `--smoke-mlp-step` path over a tiny c_fc/GELU/c_proj backward/update slice.
- [x] GPT-2 compiled C++ `--smoke-norm-residual-step` path over LayerNorm/residual/backward/gradient-accumulation/update block glue kernels.
- [x] GPT-2 compiled C++ `--smoke-transformer-block-step` path over a composed LayerNorm/attention/residual/MLP/backward/update block, including projection bias gradients and AdamW updates for all 12 GPT-2 block parameter buffers.
- [x] GPT-2 compiled C++ `--smoke-transformer-lm-step` path over sampled cached GPT-2 token IDs, embeddings, one transformer block, final norm, tied LM head, CE backward, transformer backward, embedding backward, and 16-buffer AdamW update coverage.
- [x] GPT-2 compiled C++ `--train-transformer-lm` full-vocab real-dim 12-layer multi-step loop over cached train/validation shards with token-to-loss transformer kernels, row-chunked logits workspace, scratch-recompute activation tape, 148-buffer AdamW updates, validation JSON, and no Python/Torch fallback.

## Examples to add

- [x] `examples/tile_cuda/scalar_add_train.py`
- [x] `examples/tile_cuda/dense_llm_smoke_train.py`
- [x] `examples/tile_cuda/moe_router_smoke_train.py`
- [x] `examples/tile_cuda/jepa_smoke_train.py`
- [x] `examples/tile_cuda/strict_mode_report.py`
- [x] `examples/tile_cuda/kernel_bench.py`
- [x] Generated one-file SDK example for every registry entry under `examples/tile_cuda/generated/`.

## Documentation updates

- [x] Update `README.md` with CUDA Tile setup, backend selection, and fallback behavior.
- [x] Update `CHANGELOG.md` for every meaningful backend, API, or CLI change.
- [x] Add `docs/python-sdk/tile-cuda.md`.
- [x] Update `docs/python-sdk/torch-backend.md`.
- [x] Update `docs/framework-guide/training-workflows.md`.
- [x] Update `docs/cli.md`.
- [x] Update relevant `.cursor/skills/` entries if public SDK, CLI, or MCP behavior changes.

## Tests and verification

- [x] `python -m pytest tests/test_tile_cuda_coverage.py -q`
- [x] `python -m pytest tests/test_tile_cuda_static_plan.py -q`
- [x] `python -m pytest tests/test_tile_cuda_registry.py -q`
- [x] `python -m pytest tests/test_tile_cuda_examples.py -q`
- [x] `NFN_TILE_CUDA_TEST=1 python -m pytest tests/test_tile_cuda_gpu.py -q`
- [x] `python -m pytest tests/test_template_presets.py -x -q`
- [x] `python -m pytest tests/test_builtin_neurons.py -q`
- [x] `python -m pytest tests/test_backend_capabilities.py -q`
- [x] `python -m pytest tests/test_torch_gpt.py -q`
- [x] `python -m pytest cli/tests/test_nfn_cli.py -q`
- [x] `git diff --check`

## Migration notes

- [x] Do not break existing graph JSON: Tile backend selection must live in runtime config, not graph schema, unless a later migration explicitly requires it.
- [x] Do not remove the PyTorch path.
- [x] Do not change existing template preset names for Tile support.
- [x] Do not turn variant-library port mismatch fallback back into a hard error.
- [x] If a future public config or graph serialization field changes, add a clearly labeled breaking-change note to `CHANGELOG.md` and matching docs.

## Dtype expansion: fp16, fp8, and NVFP4

Goal: add fp16, fp8, and NVFP4 CUDA Tile variants for every covered kernel where the dtype is meaningful and safe. Keep explicit exclusions for host-only nodes, integer-output/hash kernels, source/orchestration nodes, and kernels whose state contract is inherently another quantization format such as NF4.

### Dtype policy gates

- [x] Add a per-kernel dtype support matrix to the registry instead of only a flat `dtypes` tuple.
- [x] Add strict-mode errors that name the requested dtype and the supported dtype set for scalar functions, simple modules, and projection modules.
- [x] Add dtype-specific coverage reports for `float32`, `float16`, `float8_e4m3fn`, `float8_e5m2`, and NVFP4.
- [x] Keep fp8/NVFP4 accumulation in fp32 unless the kernel has a proven lower-precision accumulation contract.
- [x] Add deterministic CPU reference quantize/dequantize helpers for fp8 and NVFP4.
- [x] Add GPU parity tolerances per dtype family for verified fp16, fp8, and projection-family NVFP4 contracts.

### fp16 coverage

- [x] Scalar function kernels: unary, binary, and binary-pair function nodes use Tile float32 compute with fp16 cast-in/cast-out.
- [x] Scalar module kernels: `loss_scale`, `logit_softcap`, `aux_loss_add`, and `kl_penalty` use Tile float32 compute with fp16 activation cast-in/cast-out.
- [x] Simple vector/elementwise module kernels: `residual_add`, `residual_mix`, `manifold_hyper_connection`, `qk_gain`, and `dyt` support fp16 activations with float32 scale/parameter gradients.
- [x] Reduction-adjacent elementwise module kernels: `act_weighted_sum` and `latent_pool` support fp16 activations with float32 weights or masks.
- [x] Stateful stochastic elementwise module kernels: `dropout` fp16 training masks for `0 < p < 1`; deterministic eval and `p=0` already use the fp16 identity path.
- [x] Norm kernels: `rms_norm`, `layer_norm`, `group_norm`, and `qk_norm` with fp32 reduction.
- [x] Projection kernels: `linear`, `lm_head`, `tied_lm_head`, router/value/reward/denoise heads, KV PCA projections, JEPA heads, deterministic LoRA/TTT/adapter projections, quantized-weight `bitlinear_ternary`/`fp8_linear`/`mx_linear`, MLP projections, and ACT halt projection.
- [x] Semantic projector fp16 discrete topic output contract: keep `semantic_projector` and `semantic_chunk_projector` float32-only because their argmax-derived topic/signature semantics can change under lower-precision activation quantization.
- [x] Attention kernels: SDPA, sliding/window/block/native sparse variants, differential attention, causal/fused causal attention, MLA, and routed attention experts with fp32 score/softmax or route-weight accumulation and fp16 output.
- [x] Routed attention experts fp16 route-weight accumulation contract.
- [x] Loss/reduction kernels with fp32 accumulation and fp16-compatible logits or values: token CE, masked CE, sequence logp, latent MSE, semantic alignment, DPO, PPO, GAE, preference BCE, load/route balance, route selection/distillation, and softmax distillation.
- [x] Optimizer/runtime kernels where fp16 parameter state is meaningful: `ema_update`, `gradient_accumulate`, `gradient_clip_norm`, and `adamw_step` with fp16 parameter/gradient buffers plus fp32 Adam moments.
- [x] Muon and split-optimizer fp16 matrix-state semantics: support fp16 parameter/gradient tensors with float32 momentum/Adam state; keep standalone Newton-Schulz matrix orthogonalization float32-only.
- [x] CPU registry and GPU parity tests for the fp16-supported scalar function, elementwise module, reduction-adjacent module, norm, projection, attention, loss/reduction, and optimizer/runtime families.
- [x] CPU and GPU parity tests for each newly added fp16 family beyond scalar functions and simple modules.

### fp8 coverage

- [x] Define supported fp8 formats: `float8_e4m3fn` and `float8_e5m2`.
- [x] Direct projection fp8 activation kernels with fp32 accumulation for `linear`, LM/router/value/reward/denoise heads, tied LM head, and KV PCA encode/decode.
- [x] Composite projection-family fp8 activation kernels with fp32 accumulation and scale/amax handling for JEPA heads, LoRA/TTT/adapters, quantized-weight wrappers, MLP projections, and ACT halt projection.
- [x] Attention fp8 Q/K/V input support with fp32 score/softmax accumulation where tensor-core or Tile support allows it.
- [x] Elementwise fp8 pass-through/activation kernels where inputs can be safely dequantized to fp32 and requantized.
- [x] Explicit no-fp8 reasons for losses, optimizers, integer/hash outputs, stochastic masks, and source/delegated nodes where fp8 is not meaningful.
- [x] CPU and GPU parity tests with fp8 tolerances plus explicit boundary overflow checks for PyTorch fp8 E4M3FN/E5M2 reference behavior.

### NVFP4 coverage

- [x] Define the NeuralFn NVFP4 packed representation, scale metadata, and row/block granularity.
- [x] Add pack/unpack helpers and CPU references for NVFP4.
- [x] Projection-family NVFP4 activation kernels with fp32 accumulation for `linear`, LM/router/value/reward/denoise heads, tied LM head, KV PCA encode/decode, JEPA heads, deterministic LoRA/TTT/adapter projections, `bitlinear_ternary`, `fp8_linear`, `mx_linear`, MLP projections, and ACT halt projection; `nf4_linear` stays excluded because it owns a separate packed NF4 base-weight contract.
- [x] Attention Q/K/V NVFP4 support for SDPA, sparse/window/native/streaming-sink attention variants, differential attention, causal/fused causal attention, MLA, and routed attention experts with fp32 score/softmax and route-weight accumulation.
- [x] Explicit no-NVFP4 reasons for losses, optimizers, stochastic masks, integer/hash outputs, and source/delegated nodes where NVFP4 is not meaningful.
- [x] CPU and GPU parity tests with NVFP4 tolerances for projection-family and attention-family activations plus source-gradient preservation.
- [x] NVFP4 saturation-boundary tests for packed projection inputs.
