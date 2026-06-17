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
- [x] Default dense GPT-2 Python and SDK compiled-CLI handoff to `kernel_backend="tile-cuda"` plus `--train-transformer-lm`, leaving `llm-kittens` as an explicit external bridge instead of the default training route.
- [x] Keep GPT-2 wrapper `--native-cuda-dry-run --native-cuda-print-command` metadata-only on the default compiled CLI runner, with no dataset-manager/NumPy/tiktoken/Torch imports and no raw-text shard materialization.
- [x] Add GPT-2 `--train-transformer-lm` CUDA runtime/driver preflight JSON and fail before allocation when the driver is unavailable or older than the loaded runtime, so SM120 benchmarking has a clear native gate.
- [x] Teach the native C++ token resolver to accept llm.kittens `TinyStories_train.bin` / `TinyStories_val.bin` directly for `--tinystories`, with `NFN_LLM_KITTENS_TINYSTORIES_DIR` override and direct train-bin sibling validation inference, so GPT-2 startup can match `train-sm120.sh` without Python dataset scanning or raw-text shard materialization.
- [x] Fuse GPT-2 `--train-transformer-lm` token/target upload into one contiguous pinned-to-device uint16 arena copy and one `nfn_native_tile_uint16_to_int64` launch per microbatch, instead of one copy and one widening launch for tokens plus another pair for targets.
- [x] Add `SequentialTokenBatchSampler::next_into()` and use it in GPT-2 `--train-transformer-lm` train/validation loops so real batches write directly into pinned uint16 arenas without `TokenBatch` vector materialization or vector-to-pinned copies.
- [x] Suballocate GPT-2 `--train-transformer-lm` widened int64 token/target buffers and compact uint16 H2D staging from one aligned device token arena, reducing two token device startup `cudaMalloc` calls to one.
- [x] Replace GPT-2 `--train-transformer-lm` startup per-buffer zero fills for zero biases and AdamW state with one float-arena zero fill, eliding 369 zero-fill launches at the default 12-layer shape.
- [x] Fuse GPT-2 `--train-transformer-lm` nonzero constant parameter initialization through `nfn_native_tile_fill_many_values_float32`, reducing the default 12-layer startup path from 75 per-buffer nonzero fill launches to one descriptor-driven Tile launch.
- [x] Fuse GPT-2 `--train-transformer-lm` AdamW updates through `nfn_native_tile_adamw_step_many_with_device_scale_float32`, reducing the default 12-layer optimizer step from 148 per-buffer AdamW launches to one multi-buffer launch.
- [x] Fuse GPT-2 `--train-transformer-lm` accumulation-gradient zeroing through `nfn_native_tile_fill_many_float32`, reducing the default 12-layer optimizer-step zeroing path from 148 per-buffer fill launches to one multi-buffer launch.
- [x] Fuse GPT-2 `--train-transformer-lm` gradient-clipping sumsq partial generation through `nfn_native_tile_sumsq_partials_many_float32`, reducing the default 12-layer optimizer-step clipping path from 148 per-buffer sumsq launches to one multi-buffer launch before the device clip-scale reduction.
- [x] Wire GPT-2 `--train-transformer-lm` opt-in BF16 QKV/MLP-FC dWeight staging to direct BF16-gradient clipping and BF16-primary AdamW descriptors, eliminating the staging flush when the profiling switch is enabled; keep it default-off because the same-script dedicated-RTX-5090 benchmark measured the direct BF16 candidate slower than the optimized float-gradient path.
- [x] Default GPT `--train-transformer-lm` token embedding/LM-head startup to the fast CUDA Tile power-of-two deterministic initializer, keep `NFN_NATIVE_GPT_TOKEN_WEIGHT_INIT_LEGACY_MOD17=1` for paired modulo-17 bisection, and report the selected initializer in runtime JSON.
- [x] Match llm.kittens dense GPT dWeight accumulation semantics by adding beta-capable Tile-CUDA dWeight ABI variants and making the first gradient-accumulation microbatch write with GEMM `beta=0`, then accumulate later microbatches with `beta=1`; keep `NFN_NATIVE_GPT_DWEIGHT_FIRST_MICROBATCH_BETA_ZERO=0` for paired bisection and report the active first-write-then-accumulate strategy in runtime JSON.
- [x] Elide the first beta-zero cuBLASLt `BGRADB` bias-gradient scratch add by writing that first epilogue bias gradient directly into `grad_bias`; keep `NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=0`, `NFN_NATIVE_GPT2_BGRAD_FIRST_WRITE_DIRECT=0`, or `NFN_TILE_CUDA_LINEAR_BGRAD_FIRST_WRITE_DIRECT=0` for paired bisection. The 2026-06-17 dedicated RTX 5090 same-script 3-sample check measured the direct path at `0.999871x` train-loop wall time and `1.000129x` tokens/sec versus the old scratch-first path, so this is tracked as a launch-elision cleanup rather than a material throughput gain.
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
  - 2026-06-17 workflow update: the same parity wrapper now appends NeuralFn native `--profile-json` sidecars through `NFN_SM120_PARITY_PROFILE_DIR` by default, so future same-script parity runs carry stage buckets without manually editing candidate commands.
  - 2026-06-17 unprofiled dedicated RTX 5090 check: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=15 NFN_SM120_PARITY_PROFILE_DIR=none bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2477.527 ms/step` and `211956.7 tok/s`; NeuralFn measured `2611.690 ms/step` and `200747 tok/s`, or `1.054152x` train-loop time and `0.947113x` tokens/sec versus the reference.
  - 2026-06-17 refreshed dedicated RTX 5090 check after SDK runner cleanup: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=15 NFN_SM120_PARITY_PROFILE_DIR=/tmp/nfn_sm120_parity_profiles_after_auto_guard bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2474.848 ms/step` and `212204.6 tok/s`; NeuralFn measured `2614.030 ms/step` and `200567 tok/s`, or `1.056239x` train-loop time and `0.945159x` tokens/sec versus the reference. The refreshed profile still points at `block_backward` (`13016.7 ms`), `lm_head_backward` (`6361.27 ms`), `block_backward.attn_sdpa` (`2785.26 ms`), `block_backward.mlp_fc` (`2708.38 ms`), and `block_backward.qkv` (`2053.33 ms`) as the highest-value remaining buckets.
- [x] Keep rejected same-script SM120 kernel bisections documented so future work does not retest slower paths: `NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1` (`1.016116x` train-loop time), `NFN_NATIVE_GPT_BF16_BLOCK_DWEIGHT_STAGING=1` (`1.028320x`), `NFN_NATIVE_GPT_DWEIGHT_FIRST_MICROBATCH_BETA_ZERO=0` (`1.009501x`), compile-time `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` (`0.999900x`, noise-equivalent over four samples), `NFN_NATIVE_GPT_FUSE_QKV_BIAS_TK_GEMM=0` (`1.016504x`), `NFN_NATIVE_GPT_BF16_QKV_DWEIGHT=0` (`1.009870x` train-loop time and `0.990246x` tokens/sec, slower), `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX=0` (`1.006262x`), `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX=2` (`1.007752x`), `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX=3` (`1.005321x`), `NFN_NATIVE_LINEAR_CUBLASLT_DESCRIPTOR_CACHE=0` (`1.003259x`), `NFN_NATIVE_LINEAR_TK_FLOAT_OUT=1` (`0.999929x`, noise-equivalent), `NFN_NATIVE_GPT_CE_BF16_THREADS=512` (`0.999774x`, noise-equivalent), `NFN_NATIVE_GPT_CE_BF16_THREADS=256` (`1.011218x`), `NFN_NATIVE_GPT_CE_BF16_VEC_STORES=1` (`1.000271x`, noise-equivalent/slightly slower), `NFN_NATIVE_GPT_CE_BF16_EXP2=1` (`1.000721x` train-loop time and `0.999293x` tokens/sec, noise-equivalent/slightly slower), `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE=0` (`1.004427x`), `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_STATS=0` (`1.008131x`), `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=11` (`1.004875x`), `NFN_NATIVE_GPT_STORE_MLP_BLOCKS=11` (`1.029136x`), `NFN_NATIVE_GPT_FUSE_ATTENTION_RESIDUAL_LN2=0` (`1.021641x`), `NFN_NATIVE_GPT_FUSE_LN2_BF16_OUT=0` (`1.011993x`), `NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=128` (`1.004171x`), `NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=512` (`1.022811x`), `NFN_NATIVE_GPT_LM_HEAD_ROW_CHUNK_SIZE=4096` (`0.999703x`, noise-equivalent), `--lm-head-row-chunk-size 16384` (`1.018129x`), `--lm-head-row-chunk-size 32768` (`0.999895x` train-loop time but `1.007049x` total wall time and `1.007987x` LM-head backward, not promoted), `NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=0` (`1.009504x` train-loop time and `1.039531x` LM-head backward, slower), `NFN_NATIVE_GPT_BF16_PROJECTION_RESIDUAL=0` (`1.015959x`), `NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF=1` (`1.000141x`, noise-equivalent), `NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC=1` (`1.005702x` train-loop time and `1.125753x` setup wall time), `NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLASLT_GEMM"` (`1.014933x`, slower), a primary float+uint16 startup arena candidate (`1.075212x` setup wall time and `1.047410x` total startup wall time, slower), and a narrow LM-head dHidden cuBLASLt large-`k` probe that still fell back to `cublas_gemmex_bf16` for `m=768,n=8192,k=50304`.
- [x] Reject a narrow TK plain-dInput candidate for supported BF16 block dInput GEMMs: the temporary `NFN_NATIVE_LINEAR_TK_DINPUT=1` branch routed plain BF16 dInput through `matmul_dispatch_tk_ab` with BF16 scratch plus float conversion, but the 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured it at `1.064272x` train-loop wall time and `0.939630x` tokens/sec versus the default cuBLASLt dInput route, so no runtime switch was kept.
- [x] Reject a cuBLASLt BF16-output plain-dInput candidate: the temporary `NFN_NATIVE_LINEAR_BF16_DINPUT_OUT=1` branch routed BF16-grad/BF16-weight dInput GEMMs through BF16 output scratch plus float conversion, but the 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured it at `1.040670x` train-loop wall time and `0.960945x` tokens/sec versus the default float-output cuBLASLt dInput route, so no runtime switch was kept.
- [x] Reject `--lm-head-row-chunk-size 65536` for the default `64 x 1024` dense GPT shape: the candidate drove the dedicated RTX 5090 to 100% utilization and about `31926 MiB / 32607 MiB` used memory without completing a 5-step candidate sample in the expected paired-benchmark window, so the current 8192-row default remains the practical LM-head chunk size.
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
- [ ] Finish replacing the dense GPT-2 external `llm.kittens` bridge by live-validating the NeuralFn-owned `libnfn_native_train_tile_ops.so` loop against the `llm.kittens` SM120 script and removing any remaining external bridge dependency.
- [x] Extend native GPT checkpoint text-prompt inference with GPT-2 tokenization in the lightweight wrapper so `.bin` checkpoint inference no longer needs the transitional external sampler bridge.
- [x] Add GPT-2 evo raw Tile-CUDA trainer ABI for device-side candidate mutation, best-loss selection, and best-candidate adoption without graph-editor tensor flow.
- [x] Add GPT-2 evo compiled C++ `--smoke-evo-kernels` path that loads the raw evo ABI, launches mutate/select/adopt on CUDA device buffers, and verifies best-candidate copyback without Python/Torch, datasets, or graph-editor payloads.
- [ ] Wire GPT-2 evo native layer-evolution forward-only candidate evaluation into the dense GPT trainer loop and call the evo ABI primitives without graph-editor tensor flow.
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
- [x] Make normal NanoGPT training entrypoints (`nfn train --base-model nanogpt ...` and `python cli/scripts/train_nanogpt.py ...`) select the partial native `--train-token-lm` mode before Torch imports, with `--dry-run` / `--print-command` inspecting that same route without starting the loop.
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
- [ ] Replace fallback linear weight backward reduction kernels with GEMM-grade tiled kernels for large row counts when the trainer cuBLAS path is unavailable.
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
- [x] Update dense GPT-2 native dry-run/plan JSON to report `native-transformer-lm-ready` and `training_step_plan.status: "ready"` for the implemented compiled Tile-CUDA loop, leaving only live SM120 throughput comparison under `remaining_validation`.
- [ ] Wire full NanoGPT transformer training loop to the token-shard sampler and the ready native forward/backward/optimizer stages without importing Torch.

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
