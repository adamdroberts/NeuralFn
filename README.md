# NeuralFn

NeuralFn is a graph-native neural network framework where each neuron can be a built-in primitive or a user-defined Python function with typed I/O ports, connected in arbitrary directed graphs. This repository now combines that core library with an authenticated web platform for multi-project, multi-session editing, training, analytics, and MCP-driven automation.

NeuralFn supports both a scalar graph runtime and optional PyTorch-backed graph runtimes for trainable module nodes. The default install no longer depends on Torch; native GPT training uses cached token shards and the compiled CUDA trainer path without importing `torch`.

> **Pre-alpha notice:** NeuralFn is in active pre-alpha development. The SDK, REST API, MCP tools, and graph format are all subject to rapid, breaking changes without prior deprecation. Do not depend on API stability at this stage. See the [CHANGELOG](CHANGELOG.md) for a running list of what has changed.

## Documentation

**[Read the full documentation](docs/README.md)**

| Section | What it covers |
|---------|---------------|
| [Getting Started](docs/getting-started.md) | Installation, quickstart, your first graph |
| [CLI Workflows](docs/cli.md) | `nfn` train/infer/eval workflows, datasets, tokenizers, artifacts |
| [Framework Guide](docs/framework-guide/README.md) | How to build with NeuralFn in Python -- neurons, graphs, subgraphs, training, inference |
| [Python SDK Reference](docs/python-sdk/README.md) | Every class, function, method, and type in the `neuralfn` package |
| [REST API Reference](docs/rest-api/README.md) | All HTTP endpoints with request/response shapes |
| [MCP Tools Reference](docs/mcp/README.md) | All MCP tools for AI agent integration |
| [Server Internals](docs/server/README.md) | Services, ORM models, auth, configuration |
| [Editor Reference](docs/editor/README.md) | React frontend: store, components, API client |
| [Agent Skills](docs/agent-skills.md) | AI coding agent skills for Cursor and Codex |
| [llms.txt](llms.txt) | LLM-friendly project index |
| [llms-full.txt](llms-full.txt) | Complete docs in a single file for LLM ingestion |

Native GPT benchmark and preflight runs can pass
`--native-cuda-no-checkpoint` from the Python wrappers or `--no-checkpoint`
to the compiled C++ trainer to skip final trained-checkpoint export. Default
training still writes the final native checkpoint.

Use `python tools/paired_kernel_speed.py --baseline "OLD_COMMAND"
--candidate "NEW_COMMAND" --samples N --cuda-visible-devices 0 --json-out
/tmp/result.json` for candidate-vs-current CUDA timing pinned to the dedicated
5090 while still alternating pairs in the same sampling window. The helper runs
one warmup pair by default so first-use CUDA/kernel load does not contaminate
reported samples. It sets `CUDA_DEVICE_MAX_CONNECTIONS=1` for both commands by
default; pass `--cuda-device-max-connections ""` to leave that environment
unchanged. When `nvidia-smi` is available, the JSON also includes `gpu_before`
and `gpu_after` snapshots with GPU identity, display-active state, utilization,
memory, and active compute-process rows so kernel-speed notes show which CUDA
device was measured and whether other compute work was present. The helper decodes
native binary stdout/stderr with replacement, so external CUDA trainers that
emit non-UTF-8 bytes can still be compared in the same paired run. Pass
`--command-timeout-seconds N` to cap each child command; with
`--continue-on-error`, timed-out candidates are kept in the JSON with
`timed_out: true`, `returncode: -1`, and their output tails instead of wedging
the tuning loop. If either command emits NeuralFn native JSON, the helper also
summarizes native in-loop
metrics such as `timing.train_loop_wall_ms`, `timing.train_tokens_per_second`,
setup time, checkpoint time, total native wall time, and any emitted
`timing.stage_timing` entries separately from outer command runtime. It also
parses llm.kittens `step ... ms ... tok/s` logs, so direct
NeuralFn-vs-`train_gpt2cu` comparisons report both trainers' in-loop step time
and token throughput in the same paired JSON.

The native dense-GPT BF16 LM-head CE backward path now traverses rows in
reverse block order, matching the llm.kittens fused-classifier cache-locality
pattern for logits that were just produced by the tied LM-head GEMM.
The BF16 linear operand cache is limited to stable operands such as weights;
LM-head dWeight repacks the mutable hidden activation chunks each microbatch so
gradient accumulation does not reuse stale packed activations.

## Current state of play

NeuralFn now ships Torch-backed template presets for:
- autoregressive families: `nanogpt`, `gpt2`, `llama`, `moe` / `mixllama`, `llama_fast`, `mixllama_fast`, `jamba`, `ternary_b158`, `ttt_llama`, `universal_llama`, `llama_megakernel`, and `kv_pca_llama`
- non-AR research/overlay families: `seq2seq`, `diffusion`, `llm_jepa`, `dense_jepa_evo`, `moe_jepa_evo`, `semantic_dense_jepa_evo`, `semantic_moe_jepa_evo`, and `hnet_lm`

### Research Experiments

> **Warning:** Experimental presets are research prototypes. Their APIs, performance targets, and architectural choices are exploratory and subject to change or removal. Do not depend on stability.

- **`semantic_router_moe`** -- [Experimental] AR-only MixLLaMA/MoE control preset for testing the semantic router in isolation. It keeps the normal LLaMA attention path and MoE expert MLPs, computes a single shared vocab-grounded semantic route from the pre-block hidden state, broadcasts that route across every sequence position, and trains with next-token CE plus semantic-alignment loss. It requires one expert per active semantic vocabulary dimension and its root graph is pre-wired with a text `dataset_source` (`tokens`, `targets`) plus a `semantic_data_source` (`sem_targets`).
- **`jepa_semantic_hybrid`** -- [Experimental] Hybrid JEPA Semantic LLM that fuses a Joint Embedding Predictive Architecture with a grounded semantic vocabulary, LSH bucketing, a fixed dimension-to-expert router, and attention-capable experts that operate over the full masked hidden sequence before the LM head. Training combines three connected losses: autoregressive next-token CE, JEPA latent MSE, and masked semantic topic cross-entropy. The root graph is pre-wired with a text `dataset_source` that emits `tokens` and `targets`, plus a `semantic_data_source` that emits vocab-derived `sem_targets`. Semantic vocabulary and routing metadata live in `neuralfn/data/semantic/vocab_86d_*.json`. See `neuralfn/semantic.py` for the data layer.
- **`dense_jepa_evo` / `moe_jepa_evo`** -- [Experimental] non-semantic AR+JEPA Evo controls. Both train with next-token CE plus JEPA latent alignment and use only the text contract `(tokens, targets)`. `dense_jepa_evo` uses dense LLaMA FFNs; `moe_jepa_evo` uses standard MoE routing and load-balance loss without the semantic router, semantic data source, or route-evolution controller.
- **`gpt2_evo`** -- [Experimental] dense GPT-2 with one evolution-trained layer. `build_gpt2_evo_spec()` builds a 12-layer GPT-2 where a single designated block (`layer_evo_index`, default the middle layer) is excluded from gradient optimization; every ~10 optimizer steps an interleaved evolutionary search evaluates the current block weights plus gaussian mutants forward-only on the current batch and adopts the best candidate (the current weights are always candidate 0, so the candidate loss never regresses). `nfn_gpt2_evo_native_train --print-plan` is a compiled C++ preflight that reports the AdamW/NVFP4/evo-layer schedule and the remaining native candidate-evaluation, mutation, loss-reduction, and best-candidate adoption kernels without importing Python or Torch. Direct `python cli/scripts/train_gpt2_evo.py --print-plan ...` / `--native-cuda-dry-run ...` now prefers that family-specific preflight binary directly when `NFN_NATIVE_GPT2_EVO_CLI`, `build/nfn_gpt2_evo_native_train`, or an installed `nfn_gpt2_evo_native_train` is available. The real evo trainer still exits nonzero until those kernels land; set `NFN_ALLOW_TORCH_TRAINING=1` only for one-off legacy graph debugging. Inference can run existing exported artifacts with `nfn infer --graph ~/NeuralFn/artifacts/gpt2_evo.json --weights ~/NeuralFn/artifacts/gpt2_evo.pt --prompt "..."` or `python cli/scripts/infer_gpt2.py --evo --prompt "..."`.
- **`semantic_dense_jepa_evo`** -- [Experimental] dense control for the Semantic JEPA Evo stack. It keeps the chunk-level causal semantic planner, JEPA target encoder, AR CE, JEPA latent alignment, and semantic-alignment losses, but uses normal dense LLaMA FFNs instead of the shared/semantic/free expert bank and does not run route evolution.
- **`semantic_moe_jepa_evo`** -- [Experimental] full Semantic MoE JEPA Evo template. It keeps the autoregressive decoder dense through attention, updates a causal semantic planner at chunk boundaries, routes the next chunk through 2 always-on shared experts, 86 interpretable semantic experts, and 8 free learned experts, then applies a lightweight route-evolution controller to router bias state on a configurable fraction of batches. Its root graph uses the same flat compiled contract `(tokens, targets, sem_targets)` and adds route balance, route selection, route distillation, JEPA latent alignment, semantic alignment, and AR CE losses.

![Semantic MoE JEPA Evo architecture](docs/assets/semantic_moe_jepa_evo_architecture.png)

The full GPT architecture diagram catalog is in [Templates and Presets](docs/framework-guide/templates-and-presets.md#architecture-diagrams). It includes generated diagram sheets for the core autoregressive templates, objective/research templates, and semantic routing templates:

- [Core GPT template architectures](docs/assets/gpt_template_architectures_core.png)
- [Objective and research GPT template architectures](docs/assets/gpt_template_architectures_research.png)
- [Semantic GPT template architectures](docs/assets/gpt_template_architectures_semantic.png)

The detailed Semantic MoE JEPA Evo architecture diagram is kept as the original PNG asset so the pasted layout and formatting render as intended.

Backend capabilities (`TemplateSpec.backend_capabilities`) now drive runtime behavior:
- **cache** -- KV cache nodes (`kv_cache_read` / `kv_cache_write`) can be inserted into attention graphs for inference-time autoregressive caching. `InferenceCache` in `neuralfn/inference.py` wraps a compiled graph for stateful step-by-step generation.
- **quantized_export** -- `export_quantized_pt` / `import_quantized_pt` support int8 per-channel and ternary weight quantization for smaller checkpoint files. `KVQuantPackStage` now performs real int8 quantization instead of a plain concat.
- **megakernel** -- the `llama_megakernel` preset uses `runtime="megakernel"` which fuses the entire attention layer into a single `FusedCausalAttentionStage` module and compiles with `torch.compile(mode="max-autotune", fullgraph=True)`.
- **PCA KV cache** -- the `kv_pca_llama` preset sets `compression="kv_pca"`, inserting `kv_pca_encode` / `kv_pca_decode` nodes around the KV path in attention graphs to compress cached keys/values to a lower dimension.
- **CUDA Tile backend** -- `neuralfn.tile_cuda` exposes optional CUDA Tile runtime diagnostics, a coverage report for every current builtin/module/function/optimizer target, per-kernel dtype support matrices, and an opt-in source build path. The registry currently accounts for all 138 training-relevant entries: 129 Tile-covered kernels/compositions, 7 host-only interface/source nodes, and 2 delegated compiled-graph calls, with no `torch_fallback` entries. Implemented coverage includes 25 scalar function kernels plus elementwise/layout/routing/attention/cache/loss/embedding/position/norm/projection/reduction module and optimizer kernels such as `gelu`, `token_embedding`, `linear`, `bitlinear_ternary`, `fp8_linear`, `mx_linear`, `nf4_linear`, `ttt_linear`, `lora_linear`, `randmap_adapter`, `mlp_relu2`, `swiglu`, `geglu`, `reglu`, `solu`, `mamba`, `universal_transformer`, `jepa_projector`, `jepa_predictor`, `mask_scheduler`, `random_timesteps`, `jepa_mask`, `lm_head`, `tied_lm_head`, `router_logits`, `topk_route`, `scaled_dot_product_attention`, `sliding_window_attention`, `block_sparse_attention`, `streaming_attention_sinks`, `native_sparse_attention`, `differential_attention`, `causal_self_attention`, `fused_causal_attention`, `multi_latent_attention`, `routed_attention_experts`, `expert_dispatch`, `auxfree_load_balancing`, `semantic_projector`, `semantic_chunk_projector`, `semantic_moe_router`, `semantic_hash_router`, `semantic_moe_jepa_evo_router`, `attentionless_decoder`, `value_head`, `reward_head`, `denoise_head`, `dropout`, `adamw_step`, `muon_newton_schulz`, `muon_step`, `split_optimizer_step`, `ema_update`, `gradient_accumulate`, `gradient_clip_norm`, `kv_pca_encode`, `kv_pca_decode`, `kv_quant_pack`, `kv_quant_unpack`, `act_halt_gate`, `act_weighted_sum`, `latent_pool`, `token_cross_entropy`, `masked_token_cross_entropy`, `sequence_logp`, `dpo_pairwise_loss`, `ppo_clipped_loss`, `gae_compute`, `preference_bce_loss`, `load_balance_loss`, `route_balance_loss`, `route_selection_loss`, `route_distillation_loss`, `semantic_alignment_loss`, `semantic_hasher`, `semantic_chunk_hasher`, `softmax_distillation_loss`, `rotary_embedding`, `rms_norm`, `layer_norm`, `group_norm`, `qk_norm`, `logit_softcap`, `residual_add`, `qk_gain`, `dyt`, `reshape_heads`, `repeat_kv`, `absolute_position_embedding`, `byte_patch_embed`, `kv_cache_read`, `kv_cache_write`, `broadcast_expert_routes`, `broadcast_chunk_routes`, `byte_patch_merge`, `causal_chunk_state`, and `latent_mse_loss`. PyTorch remains the fallback for unsupported devices, dtypes, shapes, and tensor contracts.

CUDA Tile source builds require CUDA Toolkit 13.3+, `cuda_tile.h`, C++20, `nvcc --enable-tile`, and `ninja`. Install the optional native build extra with `pip install -e ".[tile-cuda]"`; it no longer installs Torch. Graph-backed Torch workflows, including the legacy PyTorch Tile extension loader, can be enabled separately with `pip install -e ".[torch]"` or with both extras when you intentionally need that path. Root `nfn --help` / no-argument startup, `nfn train|infer|eval --help`, `nfn kernels ... --help`, `nfn kernels list [--json]`, CUDA Tile registry imports, and the explicit native GPT training dispatcher avoid importing `nfn_impl` and Torch. Explicit CLI `--kernel-backend tile-cuda` runs build/load the extension automatically and now defaults to strict kernel enforcement; pass `--no-tile-cuda-strict` only for debugging fallback behavior. `nfn kernels bench` uses paired interleaved samples by default: each sample times graph-walk PyTorch, compiled PyTorch, and Tile-requested execution in alternating order, then reports mean seconds and paired ratios so unrelated external GPU load affects baseline and candidate measurements similarly. For native binary or environment-flag experiments, use `python tools/paired_kernel_speed.py --baseline "OLD_COMMAND" --candidate "NEW_COMMAND" --samples N` to run older and candidate kernels in the same alternating script. `tools/build_native_train_tile_ops.sh` defaults to the SM120 ThunderKittens-backed bf16 attention bridge when `LLM_KITTENS_ROOT` and `TK_ROOT` point at the local llm.kittens and ThunderKittens checkouts; set `NFN_TILE_CUDA_USE_TK_ATTENTION=0` for the older float32 row-scan diagnostic build, or override `NFN_TILE_CUDA_ARCH` explicitly.

Scalar CUDA Tile function kernels, simple elementwise modules, stochastic dropout, norm modules, verified projection-family modules, RoPE, verified attention modules, verified loss/reduction modules, and selected optimizer/runtime helpers support contiguous CUDA `float32` and `float16` tensors. The projection coverage includes `linear`, LM/router/value/reward/denoise heads, KV PCA projections, JEPA heads, deterministic LoRA/TTT/adapter projections, `bitlinear_ternary`, `fp8_linear`, `mx_linear`, MLP projections, and ACT halt projection. The attention coverage includes SDPA, sparse attention variants, differential attention, causal/fused causal attention, MLA, and routed attention experts with fp32 route-weight accumulation. Loss/reduction coverage includes token CE, masked CE, sequence logp, latent MSE, semantic alignment, DPO, PPO, GAE, preference BCE, load/route balance, route selection/distillation, and softmax distillation. Optimizer/runtime fp16 coverage includes `ema_update`, `gradient_accumulate`, `gradient_clip_norm`, `adamw_step`, `muon_step`, and `split_optimizer_step` with fp16 parameter/gradient buffers plus float32 optimizer state; standalone `muon_newton_schulz` remains float32-only as the matrix orthogonalization primitive. The `float16` path computes through the Tile `float32` kernels and casts activation outputs back to preserve stable math while keeping module parameters, optimizer moments, weights, masks, reductions, attention score/softmax accumulation, routing probabilities, and scale gradients in float32. Training-mode dropout uses deterministic counter-based masks for fp32/fp16 activations instead of the PyTorch RNG fallback. Broader fp16 module coverage plus fp8 and NVFP4 variants are tracked in `todo-tile-cuda.md`.

Projection-family modules now also accept CUDA `float8_e4m3fn` and `float8_e5m2` activation inputs where the output contract is safe: `linear`, LM/router/value/reward/denoise heads, tied LM head, KV PCA encode/decode, JEPA heads, deterministic LoRA/TTT/adapter projections, `bitlinear_ternary`, `fp8_linear`, `mx_linear`, MLP projections, and ACT halt projection. These fp8 activations are dequantized to float32, accumulated in float32, and return float32 outputs with float32 weight/bias gradients; branching composite projections dequantize once so internal gradient accumulation stays in float32.

Projection-family modules also accept packed `NVFP4Tensor` activation inputs for `linear`, LM/router/value/reward/denoise heads, tied LM head, KV PCA encode/decode, JEPA heads, deterministic LoRA/TTT/adapter projections, `bitlinear_ternary`, `fp8_linear`, `mx_linear`, MLP projections, and ACT halt projection. NVFP4 activations dequantize through NeuralFn block/tensor scale metadata, accumulate through the Tile float32 path, return float32 outputs, and can preserve source-activation gradients through `quantize_nvfp4_reference(..., preserve_grad=True)` for STE-style training checks. `nf4_linear` remains outside this contract because its base weights already use a separate packed NF4 representation.

Attention modules now accept CUDA `float8_e4m3fn` and `float8_e5m2` Q/K/V activation inputs where score and softmax accumulation can remain fp32: SDPA, sliding/window/block/native sparse variants, streaming sink attention, differential attention, causal/fused causal attention, MLA, and routed attention experts. These paths dequantize fp8 activations to float32, return float32 outputs, and dequantize shared composite/routed inputs before projection or expert fan-out so gradient accumulation stays in float32.

The same attention-family modules also accept packed `NVFP4Tensor` activation inputs. Q/K/V or shared attention inputs dequantize through NeuralFn NVFP4 scale metadata before Tile attention, RoPE, projection, and route-weight fan-out; score and softmax accumulation stay fp32, and outputs remain float32.

Compiled CUDA Tile graphs can opt into runtime NVFP4 activation packing with `graph.torch_config["tile_cuda_activation_dtype"] = "nvfp4"`. The packer only wraps activation input ports for modules whose registry advertises NVFP4 support, leaving tied weights, masks, targets, losses, optimizer state, and graph editor/source nodes as normal tensors. `cli/scripts/train_gpt.py` is the canonical native-only GPT entrypoint; `cli/scripts/train_gpt2.py` remains a compatibility wrapper. Direct execution with the default `compiled-cli` runner translates GPT flags to the compiled C++ CLI and runs it before importing `train_gpt_native.py`; importing the compatibility module, building its parser, or resolving native defaults still does not import Torch, `server.dataset_manager`, NumPy, or tiktoken. Dense GPT native training can be requested as `--base-model gpt`, `gpt2`, or `gpt3`; all three aliases dispatch to the same compiled dense GPT target while `--template-name` or `--graph-file` records the selected architecture. Plan and runtime JSON now canonicalize all dense GPT aliases to `model_family: "gpt"` and expose the architecture contract with `architecture_source`, `architecture_contract`, and `model_family_context_policy`; `gpt3` is only a 2048-context default when the template, graph, and sequence length are all implicit. The generic SDK module `neuralfn.native_gpt` exports generic `NativeGptRunConfig`, `NativeGptRunnerStatus`, and `NativeGptCheckpointInfo` classes plus helpers such as `build_native_gpt_compiled_cli_run_config()`, `native_gpt_activation()`, and `run_native_gpt()` over the same no-Torch native path. The native GPT default dataset is TinyStoriesV2 GPT-4 (`roneneldan__TinyStories__TinyStoriesV2-GPT4`) with the GPT-2 tokenizer; `golf1` and `golf10` remain explicit cached-token shortcuts only. The native path resolves cached uint16 train/validation shards with the shared C++ native token-shard resolver and launches the compiled GPT CUDA trainer directly instead of routing training batches through graph-editor nodes or `TorchTrainer`; raw-text materialization and HuggingFace downloads still lazy-load the dataset manager only when needed. When the default `compiled-cli` runner is selected and cached `fineweb_train_*.bin` plus validation shards already exist, Python now passes the alias/path directly to the compiled resolver without reading `meta.json` or revalidating shard metadata first. The compiled resolver also accepts llm.kittens-style `TinyStories_train.bin` / `TinyStories_val.bin` token bins, and `--tinystories` resolves to `/mnt/disk2/dev/open-source/llm.kittens/dev/data/tinystories` when those files exist; set `NFN_LLM_KITTENS_TINYSTORIES_DIR` to override that directory, or pass a direct `TinyStories_train.bin` path and the native resolver will infer the sibling validation bin. The script sets up its own repo/script import path, so direct `python cli/scripts/train_gpt.py ...`, compatibility `python cli/scripts/train_gpt2.py ...`, and `runpy`-style native invocations do not require `PYTHONPATH`. The GPT inference compatibility script follows the same startup discipline for orientation/defaults: importing `cli/scripts/infer_gpt2.py`, building its parser, resolving `--evo`/`--megakernel` artifact defaults, or running `python cli/scripts/infer_gpt2.py --help` does not import Torch, the dataset manager, or NumPy; actual graph-backed generation still imports the runtime until a native inference binary exists. Native `train_gpt2cu` checkpoints (`model_########.bin` plus `DONE_########`) are recognized by `nfn infer --checkpoint ... --native-info` and `python cli/scripts/infer_gpt2.py --native-checkpoint ... --native-info` without importing Torch, so the CLI reports their native runtime shape instead of treating them as graph-backed `.pt` files. The master CLI routes default dense GPT training commands to `nfn_gpt_native_train --backend tile-cuda --train-transformer-lm` before importing `train_gpt_native`, `nfn_impl`, or Torch; `llm-kittens` is now an explicit external-bridge backend, NanoGPT `--train-token-lm` dispatches to the partial native trainer, and unsupported families are rejected by the native model registry. The compiled GPT trainer treats `train_batch_tokens` as the real optimizer-step batch: it derives `grad_accum_steps = ceil(train_batch_tokens / (batch_size * seq_len))`, streams that many cached-shard microbatches through CUDA Tile forward/backward kernels, averages gradients in device accumulation buffers, then clips and runs AdamW once. GPT-compatible causal attention now uses the SM120 ThunderKittens bf16 FlashAttention-style bridge by default in the trainer-facing Tile ops library, and training JSON reports `attention_backend_strategy: "tk-sm120-bf16-bridge"`, TK launch counts, and zero row/scalar launches when that path is active. The SM120 default `64 x 1024 -> 524288` therefore runs eight native microbatches per optimizer step, and JSON reports the requested/effective token batch plus accumulation fields. Build the workstation C++ binding with `bash tools/build_native_gpt2_binding.sh`, standalone launcher with `bash tools/build_native_gpt2_launcher.sh`, no-Python cached-shard CLI with `bash tools/build_native_gpt_cli.sh`, and unified native frontend with `bash tools/build_native_train_cli.sh`; `cli/install.sh` also links `nfn-gpt-native`, `nfn-gpt-native-train`, `nfn-native-train`, and `nfn-gpt2-tile-launcher` into the active Python scripts directory. `nfn-native-train --list-models` or `nfn-native-train --list-models --json` reports `gpt`, `gpt2`, and `gpt3` as implemented aliases of the same dense GPT native trainer, with NanoGPT still listed as `partial-native-trainer`. `NFN_DATASETS_DIR=/path/to/datasets` overrides the native alias cache root for Python and compiled native CLI paths; `nfn-native-train --base-model gpt --dataset-alias PATH_OR_ALIAS` or `nfn-gpt-native --dataset-alias PATH_OR_ALIAS` is the fastest path when shards already exist. The CLI default is now `--native-cuda-runner compiled-cli`, so installed dense GPT training commands require the no-Python cached-shard CLI by default; pass `--native-cuda-runner auto`, `binding`, `launcher`, or `subprocess` only when you intentionally want Python materialization/orchestration or another runner. Native GPT launchers default `CUDA_VISIBLE_DEVICES=0` and `CUDA_DEVICE_MAX_CONNECTIONS=1` when those variables are unset, matching the workstation layout where the RTX 5090 is the dedicated CUDA compute GPU; set either environment variable explicitly to override that routing. Use `--eval-every-steps 1000` for per-1000-step validation loss, `--native-cuda-print-command` to inspect the resolved native invocation, and `--native-cuda-config-out PATH` to save it as JSON. The 5090 shell helpers no longer pass wallclock caps; they run to the configured step schedule unless you add one explicitly. Legacy graph-backed training scripts, including GPT-2 evo, now require `NFN_ALLOW_TORCH_TRAINING=1` for one-off debugging while native C++ trainers are added.

`nfn train --tinystories` takes the same compiled dense GPT route when `--base-model gpt` is omitted.

Prefer generic dense GPT environment names for current native runs. `NFN_NATIVE_GPT_STAGE_TIMING`, `NFN_NATIVE_GPT_PACKED_QKV_ATTENTION`, `NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS`, `NFN_NATIVE_GPT_STORE_MLP_BLOCKS`, `NFN_NATIVE_GPT_STORE_ATTENTION_ACTIVATIONS`, `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_ACTIVATIONS`, `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS`, `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE`, `NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS`, `NFN_NATIVE_GPT_FUSE_RESIDUAL1_STORE`, `NFN_NATIVE_GPT_BF16_RESIDUAL1_LN_BACKWARD`, `NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP`, `NFN_NATIVE_GPT_FUSE_ATTENTION_RESIDUAL_LN2`, `NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU`, `NFN_NATIVE_GPT_BF16_MLP_GRAD_HANDOFF`, `NFN_NATIVE_GPT_FUSE_QKV_BIAS_TK_GEMM`, `NFN_NATIVE_GPT_BF16_QKV_GRAD_HANDOFF`, `NFN_NATIVE_GPT_DIRECT_BF16_QKV_GRAD_SCRATCH`, `NFN_NATIVE_GPT_BF16_QKV_DWEIGHT`, `NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD`, `NFN_NATIVE_GPT_REUSE_PACKED_LN2_FC_GELU`, `NFN_NATIVE_GPT_FUSE_LN_BACKWARD_RESIDUAL`, `NFN_NATIVE_GPT_LM_HEAD_BF16_LOGITS`, `NFN_NATIVE_GPT_BF16_LM_HEAD_LOSS`, `NFN_NATIVE_GPT_LM_HEAD_BF16_DWEIGHT`, `NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS`, and `NFN_NATIVE_GPT_BF16_BIAS_INPLACE_TILE` configure the compiled C++ dense GPT loop regardless of whether the selected alias is `gpt`, `gpt2`, `gpt3`, a template, or a custom graph. The older `NFN_NATIVE_GPT2_*` variables remain compatibility fallbacks for existing scripts.

The trainer-facing Tile-CUDA linear ABI supports both the default BF16-primary block-weight path and the older FP32-master/BF16-shadow path for native GPT training. The trainer allocates one BF16 block-weight arena for QKV, attention projection, MLP FC, and MLP projection weights; default AdamW updates that arena directly, while the old path refreshes it with one `nfn_native_tile_float32_to_bf16_bits_many` call after parameter initialization and each AdamW update. Block forward/recompute plus block dInput GEMMs route through `nfn_native_tile_linear_weight_bf16_float32`, `nfn_native_tile_linear_weight_bf16_output_float32`, `nfn_native_tile_linear_bf16_input_weight_bf16_float32`, and `nfn_native_tile_linear_backward_input_weight_bf16_float32`. Transformer block dWeight+bias accumulation still calls `nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_float32` or `nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32`; shape-supported BF16 dWeight GEMMs request the cuBLASLt `CUBLASLT_EPILOGUE_BGRADB` epilogue, write the bias gradient into Tile-owned scratch, then accumulate it into the optimizer-step bias buffer. Shape-supported transformer-block BF16 GEMMs use cached cuBLASLt with `CUBLAS_COMPUTE_32F_FAST_16BF` by default; set `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT=0` or `NFN_NATIVE_LINEAR_BF16_CUBLASLT=0` to force the older BF16 `cublasGemmEx` bridge. `nfn_native_tile_linear_backward_weight_bias_accumulate_float32_bf16_bits` also has an opt-in mixed float32-hidden/BF16-grad cuBLASLt bgrad epilogue route for QKV profiling; set `NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD=1` or `NFN_TILE_CUDA_LINEAR_FLOAT32_BF16_BGRAD=1` to compare it against the default split-bias path. It remains default-off after paired dedicated-RTX-5090 timing showed a slight train-loop regression. `nfn_native_tile_linear_backward_input_dgelu_bf16_bits_float32` still fuses stored-BF16 MLP preactivation dGELU into the MLP projection dInput GEMM and writes the result back to the trainer's float gradient buffer; set `NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU=0` to compare against the older separate dInput plus GELU route. `nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_float32` and `nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32` expose the matching raw ABI for experiments that keep that fused MLP projection gradient as BF16 bits when feeding the following MLP FC backward GEMMs; the BF16 handoff is now the default after paired dedicated-RTX-5090 timing showed it faster; set `NFN_NATIVE_GPT_BF16_MLP_GRAD_HANDOFF=0` to compare against the older float-gradient handoff. `nfn_native_tile_linear_bf16_output_float32` writes BF16 LM-head logits, `nfn_native_tile_token_cross_entropy_partials_bf16_bits` computes validation/test CE partials directly from those BF16 logits, `nfn_native_tile_token_cross_entropy_backward_inplace_bf16_bits_with_workspace` overwrites them with BF16 dlogits, and the BF16 dlogits feed `nfn_native_tile_linear_backward_input_bf16_bits_float32` plus `nfn_native_tile_linear_backward_weight_accumulate_float32_bf16_bits`. `nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32` is also available for LM-head dWeight profiling; set `NFN_NATIVE_GPT_LM_HEAD_BF16_DWEIGHT=1` to pack each final-norm hidden chunk to BF16 and use BF16/BF16 dWeight accumulation. Set `NFN_NATIVE_GPT_LM_HEAD_BF16_LOGITS=0` to return the tied LM-head chunks to the older optimized TF32 tensor-op `cublasSgemm` path for debugging, or set `NFN_NATIVE_GPT_BF16_LM_HEAD_LOSS=0` to keep BF16 training backward while comparing validation/test loss against the older float logits loss workspace. The public GPT-2 tokenizer vocab remains 50,257, but the native tied token embedding/LM-head tensor is padded to 50,304 rows for GEMM-friendly row counts; training JSON reports both `vocab: 50257` and `padded_vocab: 50304`, and dry-run plan JSON reports `shape.padded_vocab_size: 50304`. Set `NFN_TILE_CUDA_LINEAR_BF16=1` or `NFN_NATIVE_LINEAR_BF16=1` only when you want the normal linear ABI to opt into the BF16 bridge for profiling or shape-specific tuning. Set `NFN_TILE_CUDA_LINEAR_CUBLASLT=1` or `NFN_NATIVE_LINEAR_CUBLASLT=1` only when profiling the normal linear ABI through the cached cuBLASLt TF32 path; on the current GPT-2 5090 shape it is measurable but slower than the SGEMM default. Tied LM-head BF16 logits use the SM120 ThunderKittens GEMM bridge by default when the Tile ops library was built with TK support; set `NFN_TILE_CUDA_LINEAR_TK_GEMM=0` or `NFN_NATIVE_LINEAR_TK_GEMM=0` to force the BF16 `cublasGemmEx` fallback for diagnostics. The raw optimizer ABI also exposes `nfn_native_tile_adamw_step_many_with_device_scale_bf16_shadow_float32`, which writes updated FP32 master weights and optional BF16 shadow entries in the same descriptor-driven Tile launch; set `NFN_NATIVE_GPT_FUSE_ADAMW_BF16_SHADOW_REFRESH=1` only after forcing `NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS=0`, because the shadow-refresh path is bypassed by the BF16-primary default. GPT-2 training JSON reports `linear_backend_strategy`, `block_forward_linear_strategy`, `block_backward_input_linear_strategy`, `block_backward_mlp_proj_dgelu_strategy`, `block_backward_bf16_mlp_grad_handoff_enabled`, `block_backward_weight_linear_strategy`, `non_block_forward_backward_linear_strategy`, `lm_head_logits_linear_strategy`, `lm_head_dweight_strategy`, `block_weight_bf16_shadow_strategy`, `block_weight_bf16_shadow_fused_adamw_refresh_enabled`, `block_weight_bf16_shadow_elements`, `block_weight_bf16_shadow_bytes`, `block_weight_bf16_shadow_descriptor_count`, `block_weight_bf16_refresh_count`, `block_weight_bf16_fused_adamw_refresh_count`, `adamw_bf16_shadow_refresh_strategy`, and the linear GEMM/cache counters so native runs show whether large projections used BF16-primary weights, shadow weights, cuBLASLt, the BF16 bridge, fused MLP projection dGELU, BF16 MLP gradient handoff, fused AdamW shadow writes, TK LM-head GEMM, or BF16/BF16 LM-head dWeight. The tied LM-head row chunk defaults to 4096 rows, which uses about 412MB of BF16 logit workspace at the default `64 x 1024` microbatch and measured faster than the previous 8192-row profile on the dedicated RTX 5090 in paired timing; pass `--lm-head-row-chunk-size 8192` or wrapper `--native-cuda-lm-head-row-chunk-size 8192` to reproduce the older larger-workspace profile. Full `--train-transformer-lm` uses BF16 logits for validation/test loss and in-place tied LM-head CE backward by default, so the separate float logits and full-vocab `grad_logits` chunks are not allocated; default JSON reports `logit_workspace_elements: 0`, `grad_logit_workspace_elements: 0`, `lm_head_training_logits_dtype: "bf16"`, `lm_head_loss_logits_dtype: "bf16"`, `lm_head_bf16_loss_enabled: true`, `lm_head_ce_backward_strategy: "fused-row-bf16-logits-dlogits"`, `lm_head_dweight_strategy: "float32-hidden-bf16-dlogit-dweight-accumulate"`, and `lm_head_grad_logits_workspace_allocated: false`.

Native GPT now defaults block projection weights to the BF16-primary update path. Token/position/norm/bias tensors still use the float32 multi-buffer AdamW ABI, while QKV, attention projection, MLP FC, and MLP projection weights update their BF16 parameter buffers directly through `nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_float32`; checkpoint export syncs those BF16 block weights back into FP32 staging buffers before the existing version-5 BF16 checkpoint packer runs. Set `NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS=0` to reproduce the older FP32-master plus BF16-shadow refresh path for bisection. Runtime JSON reports `block_weight_bf16_primary_param_update_enabled`, `block_weight_bf16_primary_param_update_count`, `adamw_float_update_descriptor_count`, `adamw_bf16_param_descriptor_count`, `adamw_float_update_kernel_launches`, `adamw_bf16_param_kernel_launches`, and `checkpoint.bf16_param_sync_kernel_launches`.

For the trainer-facing BF16 cuBLASLt block GEMMs, NeuralFn now selects cuBLASLt heuristic index 1 by default when that candidate is available on the workstation RTX 5090 shape. Use `NFN_TILE_CUDA_CUBLASLT_HEURISTIC_INDEX=N` or `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX=N` only for paired kernel profiling.

The current dense GPT trainer also routes the stored-MLP fused FC+bias+GELU path and fused MLP-projection dInput+dGELU path through BF16 block-weight shadows. The new raw ABI symbols are `nfn_native_tile_linear_weight_bf16_gelu_bf16_float32`, `nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_float32`, `nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_float32`, and `nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32`; the older FP32-weight fused symbols remain exported for raw kernel tests and non-shadow callers.

Native GPT attention now defaults to a packed-QKV SM120 TK bridge on the compiled Tile-CUDA path. The trainer fuses QKV bias into the SM120 TK BF16 QKV GEMM by default, runs packed TK attention directly over the row-major packed QKV tensor, and feeds the packed BF16 attention output directly into the attention projection GEMM and dWeight accumulation. Set `NFN_NATIVE_GPT_FUSE_QKV_BIAS_TK_GEMM=0` to reproduce the older separate packed BF16 QKV bias-add launch; then set `NFN_TILE_CUDA_BF16_BIAS_INPLACE_TILE=0`, `NFN_NATIVE_GPT_BF16_BIAS_INPLACE_TILE=0`, or `NFN_NATIVE_GPT2_BF16_BIAS_INPLACE_TILE=0` only when comparing Tile versus scalar CUDA for that old bias launch. Backward now keeps packed attention `dQKV` in BF16 by default: `nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_merged_grad_float32` and the saved-LSE variant write BF16 gradient bits directly into a non-aliased BF16 scratch buffer that reuses the MLP BF16 scratch after MLP backward is done, then QKV dWeight+bias uses `nfn_native_tile_linear_backward_weight_bias_accumulate_float32_bf16_bits` and QKV dInput uses the BF16-gradient/BF16-weight input-backward ABI. Set `NFN_NATIVE_GPT_DIRECT_BF16_QKV_GRAD_SCRATCH=0` to reproduce the older workspace-to-packed-QKV-buffer copy path, set `NFN_NATIVE_GPT_BF16_QKV_DWEIGHT=1` to pack LN1 output into the freed packed-QKV BF16 buffer and profile BF16/BF16 QKV dWeight+bias accumulation, or set `NFN_NATIVE_GPT_BF16_QKV_GRAD_HANDOFF=0` when profiling the older float32 `grad_qkv` expansion path. The packed backward batch cap now defaults to 64, matching the workstation `64 x 1024` microbatch in one TK backward chunk; set `NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP=48` to reproduce the previous split for paired benchmarks. Set `NFN_NATIVE_GPT_PACKED_QKV_ATTENTION=0` only when comparing against the older split-to-heads bridge. Plan and runtime JSON report `packed_qkv_attention_enabled`, packed BF16 scratch bytes, `packed_qkv_float_attention_tape_elided`, `packed_qkv_float_attention_tape_elements_elided`, `qkv_bias_fused_tk_gemm_enabled`, `qkv_bias_layout_strategy: "packed-qkv-bf16-bias-fused-tk-gemm"`, `attention_projection_input_strategy`, `attention_packed_output_unpack_strategy`, `attention_backward_bf16_qkv_grad_handoff_enabled`, `attention_backward_direct_bf16_qkv_grad_scratch_enabled`, `attention_backward_direct_bf16_qkv_grad_scratch_elements`, `qkv_backward_layout_strategy: "packed-qkv-bf16-gradient-handoff"`, `attention_backward_qkv_bridge_strategy: "tk-sm120-packed-qkv-direct-bf16-grad-scratch-handoff"`, and `attention_backward_strategy: "tk-sm120-packed-qkv-bf16-backward-direct-bf16-grad-scratch-handoff"` when this default path is active. Runtime JSON also reports `block_backward_bf16_qkv_dweight_enabled` and `block_backward_qkv_dweight_strategy` when the opt-in QKV dWeight candidate is profiled. `attention_backward_tk_launch_count` now counts packed backward chunks, so forced smaller caps visibly increase the counter. The packed route now saves packed BF16 QKV, packed O, and per-row TK `lse` for all 12 trained blocks by default on the dedicated RTX 5090 workstation shape, reuses those tensors during backward, and reports `packed_attention_activation_storage_strategy: "packed-qkv-o-bf16-forward-store-direct-backward"`, `stored_packed_attention_activation_blocks: 12`, `stored_packed_attention_lse_enabled`, `stored_packed_attention_*`, `attention_backward_strategy: "tk-sm120-packed-qkv-bf16-saved-activation-backward-direct-bf16-grad-scratch-handoff"`, and `block_recompute_saved_packed_attention` timing fields. Set `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_ACTIVATIONS=0` for the previous lower-memory recompute path, set `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE=0` to reproduce the older shared-workspace LSE behavior in paired benchmarks, or set `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=N` to tune the cap; the GPT-2-prefixed names remain fallbacks. The packed recompute path also stores intermediate block `residual1` tensors as BF16 by default and consumes them directly in LN2 backward, skipping recomputed attention projection, projection-residual work, and the older BF16-to-FP32 residual1 restore for the 11 earlier blocks in a 12-layer run. The default store is fused into `nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_float32`, and default backward consumes residual1 through `nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32` plus `nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32`. Set `NFN_NATIVE_GPT_BF16_RESIDUAL1_LN_BACKWARD=0` to compare against the older restore-to-FP32 LayerNorm backward path, set `NFN_NATIVE_GPT_FUSE_RESIDUAL1_STORE=0` to compare against the older separate `float32_to_bf16` store, or set `NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS=0` to disable this cache for lower-memory comparisons. The residual1 cache uses about 1.03 GiB at the default `64 x 1024 x 768` shape and reports `residual1_activation_storage_strategy`, `residual1_activation_store_strategy`, `residual1_backward_consumer_strategy`, `stored_residual1_activation_blocks`, `stored_residual1_activation_elements`, `stored_residual1_activation_bytes`, and store/restore launch counters.

The trainer-facing native GELU ABI follows the GPT-style tanh approximation for `nfn_native_tile_gelu_float32`, `nfn_native_tile_gelu_add_bias_float32`, `nfn_native_tile_gelu_backward_float32`, and `nfn_native_tile_gelu_backward_inplace_float32`. The graph-backed Torch path can keep PyTorch GELU semantics, but native GPT-2 forward, fused bias+GELU, and backward kernels now stay internally aligned on `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`.

The raw trainer ABI also includes `nfn_native_tile_layer_norm_with_stats_float32`, `nfn_native_tile_layer_norm_backward_input_with_stats_float32`, `nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_float32`, `nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32`, `nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_float32`, and `nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32` so native trainers can store per-row LayerNorm mean/rstd during forward and reuse those stats in backward instead of recomputing them. Dense GPT-2 uses that stats-reuse path by default, stores earlier-block LN2 stats beside the BF16 MLP activations, and fuses block LayerNorm dInput with the following residual gradient add through `nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_float32`. Set `NFN_NATIVE_GPT_FUSE_LN_BACKWARD_RESIDUAL=0` to compare against the older separate LayerNorm dInput plus residual-add route. Training JSON reports `layer_norm_stats_strategy`, `layer_norm_backward_reuses_forward_stats`, `layer_norm_backward_residual_fusion_enabled`, `layer_norm_backward_residual_strategy`, `stored_mlp_layer_norm_stats_elements`, and `stored_mlp_layer_norm_stats_bytes`. `nfn_native_tile_linear_bias_residual_layer_norm_with_stats_float32` fuses attention-projection bias, residual add, and LN2 while writing mean/rstd for the existing backward stats-reuse path, and `nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_float32` additionally writes the BF16 residual1 cache in the same launch. GPT-2 enables the fused stats path by default. Set `NFN_NATIVE_GPT2_FUSE_ATTENTION_RESIDUAL_LN2=0` only when comparing against the older separate residual add plus LN2 route. Training JSON reports `attention_residual_ln2_strategy` and launch-elision counters so profiling runs can prove whether the fused kernel was active.

Set `NFN_NATIVE_GPT2_STAGE_TIMING=1` on `--train-transformer-lm` runs when you need a CUDA-event stage breakdown for the native loop. The normal host timing block still avoids extra synchronizations; this opt-in profiler records `stage_timing_enabled`, event/drop counts, and per-stage totals/averages for token upload, model forward, block forward/recompute/backward, LM-head backward, embedding/final-norm backward, gradient zero/clip, and AdamW update under `timing.stage_timing`. Diagnostic runs also include nested LM-head, block forward/recompute, and block backward substages such as `lm_head_backward.dhidden`, `lm_head_backward.dweight`, `block_forward.attention.qkv`, `block_forward.attention.sdpa`, `block_forward.mlp_fc_gelu.fc_gelu`, `block_forward.mlp_proj.proj`, `block_backward.mlp_proj.dweight_bias`, `block_backward.mlp_proj.dinput`, `block_backward.ln1_residual.dinput_add`, `block_backward.ln2_residual.dinput_add`, `block_backward.attn_sdpa.to_qkv`, `block_backward.qkv.dweight_bias`, and `block_backward.qkv.dinput` so the next kernel bottleneck is visible without Nsight.

The default GPT transformer-LM tape uses BF16 MLP activation storage on the workstation shape. It stores all 12 trained blocks' `ln2_out`, MLP preactivation, and GELU activation tensors into a BF16 arena during forward, consumes those BF16 tensors directly for MLP dWeight and fused MLP-projection dInput+dGELU, and elides MLP fc/GELU recompute. Forward storage uses `nfn_native_tile_linear_bf16_input_weight_bf16_gelu_bf16_float32` after `ln2_out` has already been packed for backward, so the FC+bias+GELU TK path reuses the prepacked BF16 input instead of converting the same LN2 output twice. Set `NFN_NATIVE_GPT_REUSE_PACKED_LN2_FC_GELU=0` to compare against the older `nfn_native_tile_linear_weight_bf16_gelu_bf16_float32` route that repacks the float LN2 output inside the GEMM helper. Set `NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS=0` to return to the lower-memory scratch-recompute path, or set `NFN_NATIVE_GPT_STORE_MLP_BLOCKS=N` to tune the saved-block cap; the GPT-2-prefixed names remain fallbacks. Training JSON reports `mlp_activation_storage_strategy`, `reuse_packed_ln2_fc_gelu_enabled`, `stored_mlp_forward_strategy`, `stored_mlp_activation_blocks`, `stored_mlp_activation_elements`, `stored_mlp_activation_bytes`, `stored_mlp_activation_store_kernel_launches`, `stored_mlp_activation_restore_kernel_launches`, `stored_mlp_activation_backward_consumer_strategy`, `block_backward_mlp_proj_dgelu_*`, `block_state_layout.backward_recompute_mlp_fc_gelu_elided`, and the resulting `block_state_layout.activation_tape_strategy`. Rebuild `libnfn_native_train_tile_ops.so` with `bash tools/build_native_train_tile_ops.sh` after updating, because the trainer now requires the raw BF16 store/direct-backward ABI symbols.

With BF16 block shadows enabled, earlier-block forward storage reports `stored_mlp_forward_strategy: "tk-sm120-fused-fc-bias-gelu-bf16-store-bf16-shadow-weight"` and fused projection backward reports `block_backward_mlp_proj_dgelu_strategy: "tk-sm120-fused-dinput-dgelu-bf16-store-bf16-shadow-weight-bf16-grad-handoff"`. The default BF16-gradient handoff keeps the fused dInput+dGELU result as BF16 bits for the following MLP FC backward and reports `block_backward_bf16_mlp_grad_handoff_enabled: true`; set `NFN_NATIVE_GPT_BF16_MLP_GRAD_HANDOFF=0` to compare against the older float-gradient handoff.

GPT-2 block backward uses `nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_reuse_forward_from_merged_grad_float32` after each original or recomputed attention forward, reusing the TK bf16 Q/K/V/O/LSE workspace instead of repacking Q/K/V and launching a duplicate TK forward inside attention backward. JSON reports `attention_backward_strategy: "tk-sm120-bf16-reuse-forward-workspace-bridge"`, `attention_backward_reuses_forward_workspace: true`, `attention_backward_uses_saved_forward_workspace: false`, and `attention_backward_recompute_forward_elided_per_block: 1` when that optimized path runs; generic callers that cannot guarantee the matching preceding TK forward should use the older `nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32` ABI. The raw ABI also exposes `nfn_native_tile_scaled_dot_product_attention_store_tk_bf16_float32`, `nfn_native_tile_attention_tk_store_forward_workspace_bf16`, and `nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_saved_tk_bf16_from_merged_grad_float32` so native trainer loops can save TK BF16 Q/K/V/O plus float LSE into caller-owned buffers either directly during attention forward or by copying the process workspace afterward, then later run attention backward without recomputing the matching forward state. The dense GPT-2 trainer can use the direct forward-store saved-attention path with `NFN_NATIVE_GPT2_STORE_ATTENTION_ACTIVATIONS=1`; it stays disabled by default because the current 64x1024 TinyStories one-step probe reported only about 6.0k tok/s, with `block_backward.attn_sdpa.to_qkv` dominating at about 81.4s per optimizer step. Default training therefore continues to use the faster recompute plus process-workspace reuse path.

Non-dense-GPT `nfn train` commands now hand off to `nfn-native-train` first and fail in C++ with the registry status unless `NFN_ALLOW_TORCH_TRAINING=1` is set for one-off legacy debugging. Dense GPT now defaults to the NeuralFn-owned Tile CUDA `--train-transformer-lm` loop for `gpt`, `gpt2`, and `gpt3`; that path has native smoke coverage plus a full-vocab real-dim 12-layer trainer and reports `block_state_layout` with block-vector allocation/init/zero/clip/AdamW/checkpoint/tape/forward/backward loop flags. NanoGPT is a partial native trainer with `--train-token-lm` over cached token shards; `nfn train --base-model nanogpt ...` and direct `python cli/scripts/train_nanogpt.py ...` invocations add that native token-LM mode by default for normal training runs, including `--dry-run` and `--print-command` inspection. The NanoGPT token-LM loop now uses `--eval-every-steps`, `--eval-batches`, and `--eval-batch-size` to emit validation loss from resolved validation token shards inside the compiled C++ loop, without sending validation payloads through graph-editor nodes or Torch. Direct legacy training scripts prefer the model-family C++ binary when present, using `NFN_NATIVE_<MODEL>_CLI`, `build/nfn_<model>_native_train`, or an installed `nfn_<model>_native_train`, then fall back to the generic native registry; LLaMA, JEPA, semantic/MoE, and DeepSeek harnesses still need real native C++ trainers before they can be default training commands.

GPT template selection is explicit on native dense GPT training commands: pass `--template-name NAME`, `--template NAME`, or `--preset NAME` to select a shipped GPT template, or pass `--graph-file PATH` / `--graph PATH` to select a custom graph JSON. Python wrappers canonicalize those aliases to `--template-name` and `--graph-file` before the compiled frontend runs, and the compiled path records the selected template or graph in JSON. Top-level `nfn train --base-model gpt` direct compiled-CLI handoff also adds the normal `--train-transformer-lm` action when no plan/check/smoke action is requested, so selector-bearing training commands stay explicit while debug commands remain metadata-only. `--base-model gpt2` and `--base-model gpt3` are aliases for that same native trainer; GPT-3 only changes the default context to 2048 when no explicit template, graph, or `--train-seq-len` is supplied. `gpt2`, `gpt2_megakernel`, and `gpt2_moa` use the implemented CUDA Tile transformer-LM trainer today; selecting `gpt2_moa` automatically resolves the native MoA activation mode. Other template shapes and custom graphs fail fast with missing-native-trainer JSON instead of falling back to Torch. The GPT-2 evo C++ preflight accepts the same template and custom graph selector aliases and reports the shipped template catalog before any graph-backed runtime can import; its JSON marks every selected graph as not yet native-runnable until the evo-layer CUDA Tile trainer is implemented.

Native GPT-2 checkpoint export stays on the raw Tile-CUDA ABI. Successful `--train-transformer-lm` runs pack all device float32 weight tensors into one contiguous bf16 payload with `nfn_native_tile_float32_to_bf16_bits_many`, copy that compact uint16 payload back to host once, and write the version-5 `.bin` plus `DONE_########` marker. The version-5 header stores public vocab 50,257 separately from padded tensor vocab 50,304, so tokenizer validation should use the public vocab while parameter-size checks use the padded vocab. Training JSON reports `checkpoint.payload_pack_strategy: "device-many-float32-to-bf16-bits-contiguous"`, `payload_pack_kernel: "nfn_native_tile_float32_to_bf16_bits_many"`, `payload_copy_strategy: "single-contiguous-device-payload-d2h"`, `payload_cpu_bf16_conversion: false`, `device_pack_kernel_launches`, `d2h_copy_count`, `d2h_bytes`, and `float32_d2h_bytes_elided`, so checkpoint export no longer materializes full float32 parameter tensors on CPU just to pack bf16 and no longer performs one D2H copy per parameter tensor.

For dense GPT Tile CUDA preflight/training, use strict backend names only: `--backend llm-kittens` or `--backend tile-cuda` on `nfn_gpt_native_train`, or `--kernel-backend tile-cuda` from the Python wrapper. The compiled C++ default is `tile-cuda` plus `--train-transformer-lm`, `--template-name gpt`, and `model_family: "gpt"`; `gpt2` and `gpt3` are selector aliases, and `gpt3` only supplies a 2048-token default context when no template, graph, or explicit sequence length is supplied. Plan and runtime JSON report `template_name` plus `resolved_native_template_name` so the public `gpt` alias is separate from the current dense GPT implementation template. Pass `--no-train-transformer-lm` only for plan/check/debug commands that should not start that trainer. Runner names are strict too: use `compiled-cli`, not the old `cli` alias. No-data preflight actions (`--check-tile-ops`, `--smoke-tile-ops`, `--smoke-optimizer-step`, `--smoke-lm-step`, `--smoke-attention-step`, `--smoke-mlp-step`, `--smoke-norm-residual-step`, and `--smoke-transformer-block-step`) run before token-shard resolution, so they can validate the raw Tile ABI or synthetic CUDA slices without a cached dataset; their JSON reports `token_shards_resolved: false` when no shards were opened. `--smoke-tile-ops --tile-ops-lib PATH` loads the raw trainer Tile ops library, loads CUDA runtime, launches `nfn_native_tile_fill_float32` on a tiny device buffer, copies it back, and reports JSON without Python, Torch, or graph-editor payloads. The full `--train-transformer-lm --tile-ops-lib PATH` path runs a full-vocab real-dim 12-layer multi-step dense GPT loop over cached shards with periodic validation losses in `validation.losses`, using token/position embeddings, transformer blocks, final norm, a row-chunked tied LM head/CE workspace, transformer backward, embedding backward, device-side global norm gradient clipping, and AdamW parameter updates without Python/Torch. Validation uses its own C++ shard sampler and active forward batch size from `--eval-batch-size` / `eval_batch_size` instead of forcing the full training microbatch through the eval pass; the value must be between 1 and the training `--batch-size` because the fixed activation arena is still sized to the training shape. The native token embedding/LM-head tensor is padded to 50,304 rows while tokenizer-visible vocab stays 50,257. The training JSON includes `cuda_runtime_preflight`; driver version `0` or a loaded runtime newer than the driver fails before allocation, so SM120 timing failures point at GPU access/runtime compatibility instead of a later `cudaMalloc` error. `--checkpoint-metadata-smoke --output-dir PATH` writes a sparse version-5 bf16 native GPT checkpoint-format file plus `DONE_########` marker for the requested `--num-layers` target shape so the Torch-free native checkpoint/inference metadata path can validate NeuralFn-owned artifacts without running CUDA. Successful `--train-transformer-lm` runs write a final 12-layer trained-weight native checkpoint plus `DONE_########` marker. Pass `--cuda-runtime-lib PATH` or set `NFN_CUDA_RUNTIME_LIB` when libcudart is not discoverable.

For wrapper-level command inspection, `python cli/scripts/train_gpt.py --tinystories --native-cuda-dry-run --native-cuda-print-command` stays metadata-only on the default `compiled-cli` runner. It prints the compiled C++ command without importing `server.dataset_manager`, NumPy, tiktoken, or Torch and without materializing raw-text token shards, and the default Tile-CUDA command no longer includes the external `--target train_gpt2cu` bridge argument. `train_gpt.py` preserves its own command name for help/errors; `train_gpt2.py` remains a compatibility entrypoint. The compiled Tile-CUDA frontend also treats `--print-command` as a no-data/no-CUDA action: it prints the exact `nfn_gpt_native_train ...` invocation and exits before token-shard resolution, CUDA runtime loading, or driver preflight. The explicit external `llm-kittens` backend still receives `--target` and resolves train/validation shards so it can print the delegated `train_gpt2cu -i/-j` command. Dense GPT `--dry-run` / `--print-plan` JSON now reports `model_family: "gpt"` by default and the implemented trainer as `native-transformer-lm-ready` with a ready `training_step_plan`; the remaining validation item is the live SM120 throughput comparison against `llm.kittens/train-sm120.sh`.

Native dense GPT training handoff accepts `--template-name NAME` / `--preset NAME` and `--graph-file PATH` / `--graph PATH` from the Python wrapper, SDK compiled-CLI config, and compiled C++ frontend. The default `--template-name gpt` is a public alias for the implemented dense GPT native topology and reports `resolved_native_template_name: "gpt2"` until the internal template name is retired. Every name in `neuralfn.config.SHIPPED_GPT_TEMPLATE_PRESETS` can be passed through this native selection path, including GPT/NanoGPT megakernel presets and aliases such as `mixllama`; the compiled C++ plan JSON exposes the same `shipped_template_catalog`, `shipped_template_catalog_count`, and `template_known` fields so the no-Python path is visibly in sync with the SDK catalog. The implemented trainer currently runs `gpt`, `gpt2`, `gpt2_megakernel`, and `gpt2_moa`; `gpt2_moa` maps to `--native-cuda-activation moa` automatically. Structurally different shipped template names and custom graph files report `selected-graph-native-trainer-missing` rather than falling back to Torch, while typoed/unshipped template names report `unknown-template`; migration work stays explicit and no real batches pass through graph-editor nodes.

The compiled GPT-2 transformer-LM trainer keeps cached token and target batches compact as uint16 during host-to-device upload, samples them directly from the C++ shard reader into one pinned host arena, enqueues one contiguous H2D `cudaMemcpyAsync` for tokens plus targets, then widens the whole arena to the trainer's int64 token buffers with one `nfn_native_tile_uint16_to_int64` launch. Training JSON reports `token_id_upload_strategy: "uint16-pinned-async-h2d-device-widen"`, `token_id_host_staging: "pinned"`, `token_batch_staging_strategy: "direct-sampler-to-pinned-arena"`, `token_batch_vector_materialization: false`, `token_id_h2d_copy: "cudaMemcpyAsync-contiguous-arena"`, `token_id_h2d_copy_calls_per_microbatch: 1`, `token_id_widen_strategy: "single-contiguous-arena-kernel"`, `token_id_widen_kernel_launches_per_microbatch: 1`, and `token_id_host_validation: false`; cached shards are trusted on this native path instead of being re-expanded or range-validated on CPU for every batch.

GPT-2 transformer-LM startup also initializes the tied token embedding/LM-head weight on device through `nfn_native_tile_init_gpt2_token_weight_float32` instead of constructing a 154 MB host float matrix and copying it to the GPU. Training JSON reports `token_weight_init_strategy: "device-tile-deterministic"` and `token_weight_host_materialization: false`.

GPT-2 transformer-LM startup has a single owner for per-block buffers. The block-vector visitors allocate parameters, gradients, AdamW state, and scratch-tape activations for every transformer block, including block 0; the global startup list now covers only token/position/final-norm/shared workspace buffers. Training JSON reports `block0_duplicate_allocation_elided`, `block0_duplicate_activation_allocation_elided`, `block0_duplicate_parameter_initialization_elided`, and `block0_duplicate_adamw_state_zero_elided` under `block_state_layout`.

Float CUDA buffers in the compiled GPT-2 transformer-LM trainer are suballocated from one aligned device arena instead of issuing one `cudaMalloc` per parameter, gradient, AdamW moment, activation, and workspace buffer. Startup now zeroes only AdamW first/second moment state as coalesced contiguous ranges with Tile fills by default, while gradients are zeroed per optimizer step and nonzero weights are still written by device initializers. Set `NFN_NATIVE_GPT_ZERO_ADAMW_STATE_RANGES=0` to force the older descriptor-driven AdamW state fills, or `NFN_NATIVE_GPT_ZERO_ADAMW_STATE_ONLY=0` to force the older full-arena zero for bisection. Nonzero constant parameter initialization is fused through `nfn_native_tile_fill_many_values_float32`, so position weights, final norm, residual scale, and all block constant weights initialize with one descriptor-driven Tile launch instead of 75 per-buffer fills at the default 12-layer shape. The AdamW, gradient-clip, gradient-zero, and parameter-fill descriptor tables are also suballocated from one device descriptor arena and uploaded from one host-packed descriptor arena, replacing ten small startup `cudaMalloc` calls and ten descriptor H2D copies with one allocation and one copy. Training JSON reports `float_allocation_strategy: "single-arena"`, `float_arena_zero_init_strategy: "adamw-state-contiguous-range-fill"`, `"adamw-state-fill-many"`, or `"single-arena-fill"`, `float_arena_zero_fill_count`, `adamw_state_zero_fill_count`, `adamw_state_zero_range_count`, `adamw_state_zero_range_elements`, `startup_per_buffer_zero_fill_elided`, `startup_per_buffer_zero_fill_launches_elided`, `parameter_initialization_strategy: "fused-multi-buffer-fill-values"`, `parameter_initialization_kernel_launches_per_startup`, `parameter_initialization_per_buffer_launches_elided`, `descriptor_allocation_strategy: "single-device-arena"`, `descriptor_upload_strategy: "single-host-packed-arena-copy"`, `descriptor_arena_cuda_malloc_count`, `descriptor_arena_suballocation_count`, `descriptor_arena_copy_count`, `descriptor_arena_copy_calls_elided`, `descriptor_cuda_mallocs_elided`, `float_allocation_cuda_malloc_count`, `float_allocation_request_count`, `float_arena_requested_elements`, and `float_arena_allocated_elements`. The `timing.setup_timing` array also breaks startup into host-side phases such as `setup.float_arena_materialize`, `setup.stored_mlp_activation_arena`, `setup.zero_init`, and `setup.block_weight_bf16_initial_refresh`, so startup regressions can be traced without enabling CUDA event stage timing.

Token upload buffers use combined arenas too: widened int64 tokens/targets and compact uint16 H2D staging share one aligned device token arena, while compact source staging uses one pinned uint16 host arena. Training JSON reports `token_buffer_allocation_strategy: "combined-arenas"`, `token_device_allocation_strategy: "single-device-arena"`, `token_device_arena_cuda_malloc_count`, `token_device_arena_suballocation_count`, `token_device_cuda_mallocs_elided`, and the token arena element counts.

The compiled GPT-2 transformer-LM trainer does not sample training loss in the hot path. Optimizer steps run the forward activations needed for backward, CE gradient generation, gradient clipping, and AdamW only; validation cadence computes validation loss from validation shards according to `--eval-every-steps` without also measuring train loss. Set `--eval-batch-size N` to reduce validation rows per eval batch; runtime JSON reports the resolved value under `validation.eval_batch_size` and each record token count under `validation.losses[].tokens`. The output fields `train_loss_sparse: false`, `train_loss_sampling: "disabled"`, `train_loss_on_validation_steps: false`, `train_loss_eval_count`, and `train_loss_last_step` describe that contract.

Persistent transformer block outputs in the compiled GPT-2 trainer use `nfn_native_tile_copy_float32`. This replaces the earlier zero-fill plus accumulate-by-one sequence and removes one Tile kernel launch per block output preservation while keeping the scratch-recompute tape layout. The final block output copy is elided because final LayerNorm consumes that tensor before backward recomputation starts; the default 12-layer run reports `persistent_block_outputs: 11` and `final_block_output_copy_elided: true`.

Validation forwards in the compiled GPT-2 trainer do not preserve block outputs into the persistent training-backward buffers. Training forwards still save earlier block outputs for backward recomputation, while validation streams through the scratch tape because no backward pass follows; JSON reports `validation_persistent_block_outputs: 0` and `validation_block_output_copies_elided: true`.

The scratch-recompute backward pass also reuses the final block activations that remain in the scratch tape after the initial forward pass. Earlier blocks are recomputed from persistent block outputs, but the final block is not recomputed before backward; the default 12-layer run now reports `backward_recompute_blocks: 11`, `final_block_backward_recompute_elided: true`, `backward_recompute_mlp_fc_gelu_elided: true`, and `activation_tape_strategy: "scratch-recompute-bf16-stored-packed-attention-and-mlp-direct-backward"`. Recomputed earlier blocks still stop before the MLP projection output and final residual output because backward does not consume them; `block_state_layout` reports `backward_recompute_mlp_projection_elided: true` and `backward_recompute_final_residual_elided: true`. Set `NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS=0` to disable the default BF16 residual1 cache for recomputed earlier blocks; set `NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS=0` to use pure `"scratch-recompute"` with `backward_recompute_mlp_fc_gelu_elided: false`.

The MLP projection backward path writes projection dInput directly into the MLP fc gradient buffer and applies GELU backward in-place with `nfn_native_tile_gelu_backward_inplace_float32`. The full trainer therefore no longer allocates or writes a separate `grad_act` hidden-size scratch buffer; `block_state_layout` reports `mlp_proj_backward_gelu_inplace: true` and `mlp_proj_backward_grad_act_scratch_allocated: false`.

Transformer block backward residual gradients are fused through `nfn_native_tile_scaled_residual_add_float32` in the compiled GPT-2 trainer. The trainer no longer zeroes a buffer and launches two gradient accumulates for each residual-gradient pair; `block_state_layout.residual_backward_fused` reports this path.

Gradient clipping in the compiled GPT-2 trainer now feeds the device clip scalar directly into `nfn_native_tile_adamw_step_with_device_scale_float32`. That removes the separate per-gradient-buffer scale pass before AdamW; `block_state_layout.adamw_device_clip_scale_fused` reports this path.

Gradient clipping's sum-of-squares phase also reuses the device-resident optimizer descriptor table through `nfn_native_tile_sumsq_partials_many_float32`. The default 12-layer trainer computes partial sums for all 148 accumulation buffers with one multi-buffer kernel launch per optimizer step instead of one launch per buffer, then keeps the existing device clip-scale reduction and fused AdamW update. JSON reports `gradient_clip_strategy: "fused-multi-buffer-sumsq-device-scale"`, `gradient_sumsq_kernel_launches_per_optimizer_step`, `gradient_sumsq_per_buffer_launches_elided`, and `block_state_layout.gradient_clip_loop: false`.

AdamW updates in the compiled GPT-2 trainer now run through descriptor-driven multi-buffer Tile launches over device-resident parameter descriptors and the device clip scalar. The default 12-layer path splits optimizer updates into float32 descriptors for token/position/norm/bias tensors and BF16 descriptors for block projection weights through `nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_float32`; JSON reports `adamw_update_strategy`, `adamw_descriptor_count`, `adamw_step_kernel_launches_per_optimizer_step`, and `adamw_per_buffer_step_launches_elided`. The raw ABI also exports `nfn_native_tile_adamw_step_many_with_device_scale_bf16_shadow_float32` for fused AdamW plus BF16 shadow refresh profiling; set `NFN_NATIVE_GPT_FUSE_ADAMW_BF16_SHADOW_REFRESH=1` only after forcing `NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS=0`, because the shadow-refresh path is bypassed by the BF16-primary default.

Token, position, block Linear weight, LayerNorm affine, and Linear bias gradients in the compiled GPT-2 trainer accumulate directly into optimizer-step accumulation buffers. The tied LM-head CE backward scale includes the microbatch accumulation factor, LM-head dWeight chunks and token-embedding backward write into `accum_grad_token_weight`, and the old full-vocab token-gradient scratch buffer is not allocated. Position embedding backward uses the accumulate-position ABI, so `grad_position_weight` is not allocated or copied after each microbatch. Each transformer block writes qkv, attention-output, MLP fc, and MLP projection dWeight plus bias through the fused weight-bias ABI where possible, and still writes LayerNorm affine gradients straight into block accumulation buffers, so the real 12-layer loop no longer allocates per-block scratch gradient buffers or runs a per-microbatch copy loop. Accumulation buffers are zeroed once per optimizer step through `nfn_native_tile_fill_many_float32`, which reuses the optimizer descriptor table and emits one zero-fill kernel launch instead of one per accumulation buffer. JSON reports `token_gradient_accumulation_strategy: "direct-device-accumulation-buffer"`, `position_gradient_accumulation_strategy: "direct-device-accumulation-buffer"`, `block_linear_weight_gradient_accumulation_strategy: "direct-device-accumulation-buffer"`, `layer_norm_affine_gradient_accumulation_strategy: "direct-device-accumulation-buffer"`, `linear_bias_gradient_accumulation_strategy: "direct-device-accumulation-buffer"`, `gradient_zero_strategy: "fused-multi-buffer-accumulation-zero"`, `gradient_zero_kernel_launches_per_optimizer_step`, `gradient_zero_per_buffer_launches_elided`, `block_state_layout.per_block_gradient_buffers: 0`, `block_state_layout.per_block_direct_accum_gradient_buffers: 12`, `block_state_layout.gradient_accumulation_loop: false`, `block_state_layout.gradient_accumulation_copy_loop_elided: true`, `block_state_layout.gradient_zero_loop: false`, and `gradient_zeroed_buffer_count: 0`.

Large-row Linear bias-gradient and LayerNorm affine-gradient reductions in the Tile CUDA library use chunked Tile atomic reductions instead of cuBLAS SGEMV on the fallback route. Linear bias gradients use 512-row chunks, while LayerNorm affine gradients default to 256-row chunks after paired RTX 5090 timing showed better native train-loop throughput than the old 512-row setting. Set `NFN_TILE_CUDA_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=N` (or `NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE` / `NFN_NATIVE_GPT2_LAYERNORM_AFFINE_ROW_CHUNK_SIZE`) to compare other LayerNorm row chunk sizes in the paired benchmark harness. This keeps the default GPT `batch=64`, `seq=1024` backward pass on the faster local reduction route while providing enough row-chunk parallelism for the dedicated RTX 5090 compute path; small reductions can still use cuBLAS where selected.

The full GPT-2 Tile CUDA trainer also fuses MLP bias+GELU with a BF16 activation scratch for the projection input. `nfn_native_tile_gelu_add_bias_bf16_act_float32` writes float preactivation, float GELU, and BF16 GELU bits from a CUDA Tile kernel in one launch, and `nfn_native_tile_gelu_backward_inplace_bf16_bits_float32` consumes saved BF16 preactivation bits through a CUDA Tile backward kernel instead of the old scalar CUDA path. `nfn_native_tile_linear_bf16_input_bits_float32` feeds BF16 GELU bits directly to the MLP projection GEMM. Training JSON reports `mlp_proj_forward_activation_strategy`, `mlp_forward_act_bf16_elements`, and `mlp_forward_act_bf16_bytes` so probes can confirm the direct BF16 activation path is active.

LayerNorm affine-gradient backward now also has an accumulate ABI and uses a chunked parallel atomic reduction for large row counts. This avoids the previous single-block loop over every token row in GPT-sized batches and removes the scratch-to-accumulation copy; `block_state_layout.layer_norm_backward_affine_strategy` reports `auto-chunked-atomic-accumulate`. The LayerNorm affine row chunk defaults to 256 rows and can be overridden with `NFN_TILE_CUDA_LAYERNORM_AFFINE_ROW_CHUNK_SIZE`.

For SDK-level native dispatch, build the unified binding with `bash tools/build_native_train_binding.sh`. `neuralfn.native_train` exports `NativeTrainRunConfig`, `build_native_train_run_config()`, `native_train_runner_status()`, `native_train_model_registry()`, `resolve_native_train_cli()`, and `run_native_train()`, so Python callers can hand off to `nfn_native_train` through the C++ extension without importing Torch. For dense GPT direct compiled-CLI handoff, use `build_native_gpt_compiled_cli_run_config()` to create a config from a dataset alias/path without inspecting token shards in Python; the older `build_native_gpt2_compiled_cli_run_config()` compatibility helper now canonicalizes dense GPT selectors to `model_family="gpt"`.

`bash tools/build_native_train_tile_ops.sh` builds `libnfn_native_train_tile_ops.so`, a raw C ABI over the CUDA Tile kernels in `neuralfn/csrc/tile_cuda/kernels.cu`. This trainer-facing library exposes single-buffer and multi-buffer fill/zeroing, single-buffer and multi-buffer sumsq partials, single-buffer and multi-buffer AdamW, gradient accumulation, deterministic GPT-2 token-weight initialization, device float32-to-bf16 checkpoint payload packing, device-side global-norm clip scale finalization, device-scalar gradient scaling, fixed scaling, reductions, linear, forced-BF16 linear, linear input/forced-BF16 input/fused dGELU input/weight/weight-accumulate/forced-BF16 weight-accumulate/forced-BF16 weight+bias-accumulate/bias/bias-accumulate backward, scaled residual add, fused projection bias+residual add, fused QKV split/merge, fused GPT-2 QKV split-to-heads, fused GPT-2 QKV bias+split-to-heads, fused GPT-2 heads-to-QKV gradient merge, fused TK bf16 attention-gradient heads-to-QKV bridge, saved TK BF16 attention workspace copy/backward, reshape-heads/merge-heads, GELU forward, fused bias+GELU forward, GELU backward, token embedding forward/weight backward, absolute-position embedding forward/backward/backward-accumulate, RMSNorm, RMSNorm input backward, LayerNorm, LayerNorm input plus fused input/residual-add, affine, and affine-accumulate backward, softmax, float token and masked token cross-entropy partials, BF16-bits token cross-entropy partials, token and masked token cross-entropy logits backward, and scaled dot-product attention forward/backward without including `torch/extension.h` or linking the PyTorch extension binding; native model trainers should link this library instead of calling graph-backed Tile wrappers. The trainer build defines `NFN_TILE_CUDA_USE_CUBLAS_LINEAR=1` and links `libcublas` so the same native linear ABI routes linear forward, dInput, dWeight, and accumulate-dWeight through GPU GEMM; forced-BF16 block weight+bias accumulation uses cuBLASLt `BGRADB` when supported and falls back to separate dWeight plus Tile bias-reduction launchers. The generic Tile extension build keeps the pure Tile fallback path. CE logits backward uses row-wise kernels for vocabularies up to 1024 and chunked row-wise kernels with reusable row-stat workspace for full GPT-class vocabularies, avoiding the previous elementwise large-vocab fallback. Linear weight, accumulate-weight, bias, and accumulate-bias backward keep the row-chunked tiled atomic fallback for builds or shapes that do not use the trainer cuBLAS path.

The default dense GPT MLP projection backward now uses the BF16-only dGELU handoff path when BF16 MLP grad handoff is active, so the Tile kernel writes the BF16 gradient consumed by the following MLP FC backward stage without also converting it to an unused FP32 gradient buffer. Training JSON reports `block_backward_mlp_dgelu_float_grad_elided` and the `...-no-float-grad` strategy suffix. Set `NFN_NATIVE_GPT_ELIDE_MLP_DGELU_FLOAT_GRAD=0` to restore the prior conversion path for paired kernel benchmarks.

For profiling only, `NFN_TILE_CUDA_LINEAR_TK_FLOAT_OUT=1` or `NFN_NATIVE_LINEAR_TK_FLOAT_OUT=1` routes eligible BF16 linear forward GEMMs through the TK BF16-output bridge and converts the result back to float32. This path reports `linear_tk_float_out_gemm_count` in native GPT-2 JSON and remains disabled by default because the current full-shape TinyStories probe regressed overall throughput.

The same raw ABI also exposes TK attention backward-to-QKV forward-workspace reuse for trainer loops that own the matching forward/backward ordering, plus a saved-workspace variant for loops that store the TK BF16 forward state per block.

GPT-2-compatible SDPA forward now attempts a dense causal full-head row-vector Tile kernel when `seq_k <= 1024`. One Tile CTA computes the score/softmax for a full query row and reuses it across all 64 value channels, instead of launching one scalar-output CTA and recomputing attention scores per output element. If CUDA rejects that row-kernel launch, the native Tile launcher records `cudaFuncGetAttributes` diagnostics, the pre-launch CUDA error state, the requested row-kernel grid/block shape, clears the error, auto-disables further row attempts for the run, and falls back to scalar Tile attention without repeated failed-launch overhead. Native GPT-2 plan/training JSON reports `attention_forward_strategy: "row-vector-tile-score-reuse"`, `attention_forward_value_chunk_size: 64`, `attention_forward_scalar_launch_fallback_enabled: true`, `attention_forward_row_launch_auto_disable_enabled: true`, runtime row/fallback/scalar launch counts, row-kernel attribute fields, pre-launch error codes, row launch grid/block fields, the row count, the old scalar output count, and the score-reuse/elision factor.

GPT-2-compatible SDPA backward now uses the same query-row score reuse contract for `qk_dim <= 64`, `value_dim <= 64`, and sequence lengths up to 1024. The native Tile launcher zeros Q/K/V gradient buffers, then launches one query-row CTA that computes the row softmax once, writes the full 64-channel `dQ`, and atomically accumulates `dK`/`dV` for every attended key row. This replaces the old scalar-output backward CTAs and avoids the key-row implementation that repeated a full query softmax scan per key row. Native plan/training JSON reports `attention_backward_strategy: "query-row-atomic-tile-score-reuse"`, `attention_backward_row_count`, `attention_backward_scalar_output_count`, `attention_backward_score_reuse_dim: 64`, and `attention_backward_scalar_cta_elision_factor: 192`.

Compiled GPT-2 `--train-transformer-lm` results include a `timing` block with host wall-clock phase timers: `setup_wall_ms`, `train_loop_wall_ms`, `validation_wall_ms`, `train_compute_wall_ms`, `checkpoint_wall_ms`, `total_wall_ms`, `optimizer_steps_per_second`, and `train_tokens_per_second`. These timers do not insert new device synchronizations; the train-loop timer ends after the existing final device-to-host sample copy, so it is useful for comparing native run shapes without changing the hot path.

The full GPT-2 transformer-LM trainer fuses both attention QKV layout directions. `nfn_native_tile_split_qkv_to_heads_add_bias_float32` consumes the no-bias QKV CUBLAS output, applies Q/K/V bias, and writes Q/K/V head-major buffers in one Tile launch per block instead of separate QKV bias add, QKV split, and three reshape launches. `nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32` lets TK SDPA backward read the row-major attention-output gradient directly and write row-major `grad_qkv` directly from bf16 head-major gradients, replacing three bf16-to-float gradient conversion launches plus the heads-to-QKV merge launch. The full trainer no longer allocates the three row-major or three head-major Q/K/V gradient scratch buffers for that path. Native plan/training JSON reports `qkv_forward_layout_strategy: "fused-split-to-heads"`, `qkv_bias_layout_strategy: "fused-qkv-bias-split-to-heads"`, `attention_backward_grad_layout_strategy: "merged-grad-out-direct"`, `attention_backward_qkv_bridge_strategy: "fused-bf16-heads-to-row-qkv"`, `qkv_backward_layout_strategy: "fused-heads-to-qkv"`, and the elided layout launches per block.

Forward attention also skips the obsolete row-major Q/K/V scratch tensors in the compiled GPT-2 trainer. The fused QKV split writes head-major `q_heads`, `k_heads`, and `v_heads` directly for attention, so the full 12-layer `64 x 1024` shape no longer reserves three unused activation buffers per tape, reducing the float arena by 150,994,944 elements, about 576 MiB. Native plan/training JSON reports `block_state_layout.forward_row_qkv_scratch_allocated: false` and `block_state_layout.forward_row_qkv_scratch_buffers_elided: 3`.

The full GPT-2 MLP path fuses the `c_fc` bias add with GELU. `nfn_native_tile_gelu_add_bias_float32` takes the no-bias CUBLAS `c_fc` output, writes the biased preactivation needed by GELU backward, and writes the GELU activation in one Tile pass instead of separate bias-add and GELU launches. Native plan/training JSON reports `mlp_fc_bias_gelu_strategy: "fused-bias-preactivation-gelu"` and the elided legacy launch per block.

The full GPT-2 projection residual path also fuses `c_proj`/attention-output bias with the residual add. Attention and MLP projection CUBLAS calls run without bias, then `nfn_native_tile_linear_bias_residual_add_float32` applies the projection bias, residual scale, and residual add in one Tile launch for each projection site. Native plan/training JSON reports `projection_bias_residual_strategy: "fused-linear-bias-residual-add"` and two elided legacy launches per block.

`neuralfn/csrc/native_train/token_shards.cpp` provides the reusable no-Torch token-shard resolver and sequential batch sampler for native trainers. It resolves `NFN_DATASETS_DIR`, validates `fineweb_train_*.bin` / `fineweb_val_*.bin` uint16 shard alignment, accepts llm.kittens-style `TinyStories_train.bin` / `TinyStories_val.bin`, infers validation siblings for direct train-bin paths, skips the 1024-byte cached-shard header when present, sorts shards, counts train/validation tokens, computes microbatch/gradient-accumulation metadata, and can either produce `[batch, seq_len]` token plus next-token target vectors for smoke/debug output or write directly into caller-owned token/target uint16 buffers with `SequentialTokenBatchSampler::next_into()`. The full GPT-2 trainer uses `next_into()` with pinned memory, so real batches do not materialize as `TokenBatch` vectors before upload and do not flow through graph nodes. The sampler reads contiguous shard segments for each batch instead of opening the shard once per sequence chunk; native token-shard JSON reports `batch_read_strategy: "contiguous_shard_segments"`.

`bash tools/build_native_missing_trainers.sh` builds compiled per-family native entrypoints for GPT-2 evo, NanoGPT, LLaMA, MixLLaMA, JEPA, semantic-router MoE, and DeepSeek-V4. GPT-2 evo now has a C++ native preflight binary: `nfn_gpt2_evo_native_train --print-plan --eval-every-steps 1000 --tile-cuda-activation-dtype nvfp4` validates the dense GPT-2 shape, `adamw` profile, NVFP4 activation intent, validation cadence, and evo-layer index/cadence/population metadata, then prints the remaining native candidate-evaluation, device-side mutation, candidate-loss reduction, and best-candidate adoption kernels without importing Python or Torch. NanoGPT now has a partial C++ native trainer: `--print-plan` parses the native NanoGPT defaults, validates the shape/AdamW contract, and emits JSON without importing Python or Torch; the JSON includes the contiguous parameter, gradient, AdamW state, clip-scale buffer layout, AdamW decay/no-decay groups, and a forward/backward/optimizer `training_step_plan` with each stage marked `ready`, `requires_wiring`, or `missing_abi`. It also supports `--check-tile-ops --tile-ops-lib PATH` to `dlopen` `libnfn_native_train_tile_ops.so` and verify every NanoGPT-required raw ABI symbol from the compiled binary before training-loop wiring, plus `--smoke-tile-ops --tile-ops-lib PATH` to `dlopen` CUDA runtime, allocate a tiny device buffer, execute `nfn_native_tile_fill_float32`, copy the result back, and verify the value without Python or Torch. `--smoke-optimizer-step --tile-ops-lib PATH` builds the NanoGPT parameter layout, allocates contiguous param/grad/AdamW moment buffers, initializes them with raw fill kernels, executes `nfn_native_tile_adamw_step_float32` once per registered parameter buffer with that buffer's decay/no-decay setting, copies param and moment buffers back, and verifies the update. `--smoke-training-loop-step --tile-ops-lib PATH` exercises the native optimizer-loop mechanics over that same registered layout: gradient zeroing, synthetic gradient fill, global-norm clip scale finalization, device-scalar gradient scaling, and per-buffer AdamW updates. `--smoke-lm-step --tile-ops-lib PATH` runs a tiny tied-embedding language-model step through token embedding, linear logits, token CE loss/backward, linear input/weight backward, token embedding weight backward, and AdamW update kernels, then verifies loss, gradient, and weight update values without Python or Torch. `--smoke-token-train-step --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS` samples a real native uint16 token/target batch from cached shards, runs the tied-LM forward/backward/update kernels over those IDs, and verifies sampled-batch loss, gradient, and weight update values without Python or Torch. `--train-token-lm --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS` promotes that path from a smoke to a real multi-step native loop: it streams cached token batches with the C++ sampler, zeros gradients on device, runs tied token-embedding LM forward/backward, applies AdamW each step, emits training JSON metrics, and writes periodic validation losses from validation shards under the JSON `validation.losses` block when `--eval-every-steps` is positive. `--smoke-embedding-norm-step --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS` samples the same native token batch, runs token plus absolute-position embeddings, residual add, LayerNorm forward/backward, tied logits, CE backward, embedding/position/norm gradients, and AdamW updates, then verifies residual, norm, loss, gradient, and weight update values without Python or Torch. `--smoke-qkv-layout-step --tile-ops-lib PATH` verifies fused QKV split/merge layout kernels for the NanoGPT `attn.qkv.weight` activation and gradient path. `--smoke-fused-qkv-attention-step --tile-ops-lib PATH` runs a tiny attention stage through one fused `attn.qkv.weight`, QKV split, SDPA forward/backward, QKV gradient merge, fused qkv weight backward, output projection backward, and AdamW updates for fused qkv/output weights. `--smoke-transformer-block-step --tile-ops-lib PATH` composes LayerNorm, fused-QKV attention, residual adds, MLP, backward passes, gradient accumulation, and AdamW updates for a tiny transformer block through the raw native kernels. `--smoke-mlp-step --tile-ops-lib PATH` runs a tiny MLP stage through fc projection, GELU, output projection, projection/input backward, GELU backward, and AdamW updates for both MLP weights, then verifies forward, gradient, and weight update values without Python or Torch. `--smoke-attention-step --tile-ops-lib PATH` remains the separate-Q/K/V attention stage smoke for comparing projection math. Use `--cuda-runtime-lib PATH` or `NFN_CUDA_RUNTIME_LIB` when libcudart is not on the default loader path. Tied LM head input/weight backward is covered in that plan by the raw linear backward ABI, and the AdamW optimizer stage is ready at the registered-buffer level. The native NanoGPT preflight defaults to `dropout_p=0.0`; pass nonzero `--dropout-p` only to inspect the currently missing dropout ABI path. Pass `--require-token-shards` to require and report resolved cached token shards in that JSON, and `--sample-token-batch` to include the first native token/target batch. Full NanoGPT transformer training and GPT-2 evo training still exit nonzero until their CUDA Tile forward/backward trainer loops land; NanoGPT `--train-token-lm` is the implemented partial native training path. The other binaries intentionally fail with family-specific CUDA Tile kernel requirements until the real native trainers are implemented, but they keep CLI, direct-script, and SDK handoff paths on compiled C++ artifacts instead of graph-backed Torch. `tools/install_native_gpt2_commands.sh` installs both underscore and hyphen command names for these entrypoints, so installed `nfn-native-train --base-model nanogpt ...` can find `nfn_nanogpt_native_train` without falling back to Python. Override a per-family target with `NFN_NATIVE_<MODEL>_CLI`, for example `NFN_NATIVE_NANOGPT_CLI=/path/to/nfn_nanogpt_native_train`.

Large raw-text datasets whose tokenizer ids fit in `uint16` are cached as `fineweb_train_*.bin` / `fineweb_val_*.bin` token shards on first training load. Subsequent runs estimate schedule length from metadata or shard sizes and memmap the token shards instead of re-tokenizing `data.txt` / `val.txt`. Tokenizers with ids outside `uint16` remain on the raw-text path. Dataset source nodes store dataset names and sequence length only; real text/token payloads stay in the dataset cache and do not pass through graph-editor node metadata.

Scalar CUDA Tile functions and simple elementwise modules also accept CUDA `float8_e4m3fn` and `float8_e5m2` activation inputs where outputs can be safely requantized: unary/binary/binary-pair scalar functions plus `loss_scale`, `logit_softcap`, `aux_loss_add`, `kl_penalty`, `residual_add`, `residual_mix`, `manifold_hyper_connection`, `qk_gain`, and `dyt`. These paths dequantize to float32 for Tile compute and requantize activation outputs back to the input fp8 format.

Strict CUDA Tile scalar function and simple module contract errors name the rejected tensor dtype plus the supported dtype set so dtype fallback issues are visible during training setup.

The SDK also provides deterministic fp8 reference quantize/dequantize helpers and an `NVFP4Tensor` reference packer using FP4 E2M1 values in blocks of 16 with FP8 E4M3 block scales and an FP32 tensor scale. NVFP4 support currently covers projection-family and attention-family activation contracts with boundary/reference coverage; optimizer, loss, and stochastic-mask contracts remain explicitly unsupported in the dtype matrix.

Registry `dtype_support` entries now use category-specific no-support reasons for lower-precision gaps, including losses/reductions, optimizers, stochastic mask producers, integer/hash/routing outputs, NF4 packed-format wrappers, host-only source nodes, and delegated child-graph calls.

`TemplateSpec.runtime` (`eager` / `compile` / `megakernel`) now drives `torch.compile` mode in `TorchTrainer.train()`, replacing the previous `TorchTrainConfig.compile` boolean as the primary control.

Project datasets are managed from a dedicated `Datasets` surface, then attached to graphs through a `dataset_source` node that now adapts its output roles to the active template shape:
- AR / H-Net / Universal: `tokens`, `targets`
- Seq2Seq: `enc_tokens`, `dec_tokens`, `targets`
- Diffusion / JEPA: `tokens`
- Semantic routing presets: `tokens`, `targets` plus a separate `semantic_data_source` node for `sem_targets`

`hnet_lm` uses a raw-byte training path (`vocab_size == 256`) instead of the normal tokenized loader. Trained weights are serialized back into graph JSON via `module_state`, and PyTorch weight round-tripping plus inference helpers live in `neuralfn/inference.py`.

The platform foundation now adds:
- authenticated users with HTTP-only session cookies
- multi-project, multi-session workspaces
- a routed React shell with Editor, Runs, Analytics, and Admin surfaces
- SQLAlchemy/Alembic-backed persistence with default SQLite and MySQL-ready configuration
- optional Redis-backed live state for refresh-safe restore and agent/run coordination
- project/session-scoped REST APIs and MCP tools

## How it works

1. **Choose neurons** from built-ins or define custom Python functions with `@neuron` and typed I/O ports.
2. **Wire them** into a directed graph with weighted edges.
3. **Probe & train** — scalar graphs sample each neuron to build differentiable surrogate models, then train connection weights via gradient descent or evolutionary search.
4. **Torch modules** — tensor-native graphs can train serialized module nodes through a PyTorch backend, including nested subgraphs built from multiple trainable stages.
5. **Use the platform** — store graphs inside project/session workspaces, inspect runs and analytics, and drive the same scoped graph APIs from the UI or MCP tools.

## Torch template workflows

- The editor toolbar, `/api/.../templates/gpt/*` routes, and MCP `load_gpt_template` all accept the same shipped preset names.
- Template graphs persist a serialized `template_spec` in `graph.torch_config`, so training, tracing, dataset loading, and exports can recover the original objective/backbone/tokenization contract without inferring it from node names.
- Dataset-backed tracing and training now route by input role rather than assuming only `(tokens, targets)`. That is what enables single-input JEPA/diffusion graphs and three-input Seq2Seq graphs to use the normal `dataset_source` workflow.
- `llm_jepa` uses an EMA target encoder and supports two masking strategies via `jepa_mask_strategy`: `"random"` (default, i.i.d. per-token) and `"block"` (contiguous span masking). Block masking is configured with `jepa_num_blocks`, `jepa_min_block_ratio`, and `jepa_max_block_ratio`. `diffusion` samples timesteps internally, and `hnet_lm` switches the dataset pipeline to raw bytes automatically.
- `semantic_router_moe` uses the same flat compiled input contract as the hybrid preset, `(tokens, targets, sem_targets)`, but keeps the backbone purely autoregressive. It projects the embedding output into vocab-topic space, hashes that semantic vector, resolves a shared batch-level expert route, broadcasts the chosen experts across the whole sequence, and applies that route to every MoE block without any JEPA encoder/EMA path.
- `jepa_semantic_hybrid` now expects three flat training inputs in compiled form: `(tokens, targets, sem_targets)`. Its dataset-backed root graph provides `tokens` and `targets` from `dataset_source`, materializes vocab-backed `sem_targets` from the active `vocab_86d_*.json` semantic vocabulary, hashes the pooled semantic state, routes the full hidden sequence into one expert per semantic vocabulary dimension, and trains the routed branch with AR next-token loss plus JEPA and semantic-alignment auxiliaries.
- `semantic_dense_jepa_evo` and `semantic_moe_jepa_evo` also expect `(tokens, targets, sem_targets)`, but work at chunk granularity instead of sequence granularity. `semantic_dense_jepa_evo` is the dense FFN control with no expert dispatch or route evolution. `semantic_moe_jepa_evo` adds the shared/semantic/free expert bank; `route_chunk_size` defaults to `32`; `semantic_shared_experts`, `semantic_free_experts`, and `experts` must stay consistent (`shared + NUM_VOCAB_DIMS + free`); and `route_evo_fraction` controls how often the trainer runs the lightweight evolutionary route-bias search.
- Semantic-only preview/training paths now synthesize safe placeholder `tokens` / `targets` tensors when a graph has `semantic_data_source` but no attached text dataset. That keeps semantic research graphs previewable and trainable without accidentally feeding categorical `sem_targets` into the token embedding path.
- The vocab-only semantic helpers keep `n_sig_buckets` on the higher-level data APIs, while the low-level hash helpers continue to use `n_buckets` and now accept `n_sig_buckets` as a compatibility alias. The sibling JEPA harness also derives semantic-row schedule estimates from the resolved `top_k`, so non-default routing runs report the right loader/epoch math before training starts.
- The trainer's vocab-only semantic dataset path now wraps the `load_training_targets()` `int64` arrays directly, so CUDA JEPA runs reach warmup/training without requiring a module-level NumPy import inside `torch_backend.py`.
- `TorchTrainConfig` now supports a parameter-golf-inspired split-optimizer profile with token-budgeted accumulation (`train_batch_tokens`), role-specific learning rates (`embed_lr`, `head_lr`, `tied_embed_lr`, `matrix_lr`, `scalar_lr`), Muon controls, warmup/warmdown scheduling, and gradient clipping.
- `CompiledTorchGraph` now executes each node through a fixed child module and precompiled edge-routing plan instead of routing mixed `Long` / BF16 / FP32 inputs through a single generic dispatcher or walking graph-editor node metadata per batch. That keeps real training tensors on the compiled hot path, makes `runtime="compile"` BF16 CUDA runs much more stable under `torch.compile`, and leaves scalar loss stages to upcast only inside their final reduction.
- `TorchTrainer.train()` now accepts an `on_step` callback for live warmup and optimizer-step progress. The sibling JEPA harness uses that hook together with `--train-log-every` so long CUDA runs keep printing visible progress.
- The sibling JEPA harness also ships `scripts/infer_jepa_semantic.py`, a CUDA-only probe that loads the exported `.json` graph plus `.pt` weights, traces the hybrid model's internal `model/softcap` or `model/lm_head` logits node, and samples text with the cached SentencePiece tokenizer from the dataset alias when available.
- The sibling SDK harness now also ships `scripts/train_semantic_router_moe.py` and `scripts/infer_semantic_router_moe.py` so the router-only control experiment can be trained and probed without the JEPA stack.
- All shipped sibling training and inference harnesses now auto-download a missing cached dataset alias by default when they can derive the standard cached-variant download contract from `owner__repo__variant__trainN` or from explicit dataset download flags. Existing cached aliases stay strict: tokenizer-backed contract mismatches still fail fast with the original validator error instead of a misleading missing-alias message.
- Saved graphs that still reference older block-family names such as `attn_block`, `transformer_block`, or `mixllama` now resolve through a compatibility alias layer instead of failing during template normalization when the equivalent current family is present.

## Editor behavior

- Toolbar, template, custom-node, subgraph, and variant-library inserts now default to the center of the visible graph viewport with a small stagger, so newly added nodes appear on screen even after panning or zooming away from the origin.
- Direct graph edits still preserve explicit positions; the viewport anchor is only the fallback when an add action does not originate from a canvas click.

## Architecture at a glance

- `neuralfn/` contains the core graph, neuron, variant, and trainer implementations.
- `server/` provides the FastAPI platform layer: auth, bootstrap, admin, projects, sessions, datasets, runs, SQLAlchemy persistence, Alembic migrations, and optional Redis live state.
- `editor/` contains the routed React app shell and graph editor UI.

## Quick start

### Install Python dependencies

```bash
pip install -r requirements.txt
```

The default requirements intentionally do not install Torch. Native dense GPT
training and dataset-cache preparation do not require it. Install graph-backed
Torch workflows explicitly when needed:

```bash
pip install -e ".[torch]"
```

### Install the SDK as a package

From the repository root:

```bash
pip install -e .
```

From a sibling project checked out next to the repo:

```bash
pip install -e ../NeuralFn
```

This installs the `neuralfn` package in editable mode and includes the shipped
semantic vocabulary files under `neuralfn/data/semantic/` as package data.

### Install the local CLI

The repo also includes the `nfn` CLI package under `cli/`:

```bash
cd cli
./install.sh
nfn --help
```

The CLI installer keeps Torch optional, builds the native GPT C++ binding,
launcher, no-Python cached-shard CLI, and unified native training frontend, and
links `nfn-gpt-native`, `nfn-gpt-native-train`, compatibility `nfn-gpt2-native`
names, and `nfn-native-train` into the active Python scripts
directory. Pass `./install.sh --no-native` to skip native artifact builds. The
generic SDK binding can execute normal `argv` configs and GPT compiled-CLI
configs; alias-only GPT configs prefer `compiled_cli_argv` so dataset aliases
stay on the compiled C++ resolver instead of raw external trainer paths. The
GPT native CLI exposes `--backend llm-kittens|tile-cuda`; `tile-cuda` is the
default NeuralFn-owned path and runs the dense `--train-transformer-lm`
trainer unless an introspection command such as `--print-plan`, `--check-tile-ops`,
or `--no-train-transformer-lm` is used. `llm-kittens` remains an explicit
external bridge. Use `nfn train --base-model gpt ...` as the canonical native
trainer entrypoint. `--base-model gpt2` and `--base-model gpt3` are dense GPT
aliases that route to the same CUDA Tile C++ trainer; `gpt3` defaults to a
2048-token context only when no explicit template, custom graph, or
`--train-seq-len` is supplied. Otherwise the selected GPT template or
`--graph-file` is the architecture source of truth. The full `nfn train`
parser and planner accept the same aliases, and graph-backed compatibility
paths canonicalize those dense GPT aliases back to the GPT-compatible template
builder instead of creating separate GPT-2/GPT-3 trainers. The Tile plan includes the GPT parameter layout and
forward/backward/optimizer stage sequence; the training JSON reports
`block_state_layout` flags for per-block allocation, initialization, gradient
zeroing, gradient clipping, AdamW update, checkpoint export, activation tape,
forward block, and backward block loops. The 12-layer path uses one scratch tape
with recompute plus persistent block outputs to avoid per-layer full-tape
allocation.
The CLI provides composed `train`, `infer`, and `eval`
workflows with dataset shortcuts, tokenizer selection, graph-linked artifacts,
and optional interactive planning. See [CLI Workflows](docs/cli.md) and
[cli/README.md](cli/README.md).
Direct `python cli/scripts/train_gpt.py ...` is the canonical dense GPT script.
It uses the same pre-import compiled CLI fast path as the compatibility
`train_gpt2.py`, so dry-run and default native training commands do not load
Torch, NumPy, the dataset manager, or the compatibility Python harness before
handing off to C++.

### Run the library examples

```bash
python examples/xor_graph.py
python examples/nested_hybrid_graph.py
python examples/gpt_graph.py
python examples/tile_cuda/scalar_add_train.py
python examples/tile_cuda/kernel_bench.py
```

### Install editor dependencies

The repo currently tracks frontend dependencies with `pnpm` (`editor/pnpm-lock.yaml`):

```bash
cd editor
pnpm install
```

### Platform configuration

By default, the backend starts with a local SQLite database at `neuralfn.db`, stores snapshots in `server/session_snapshots`, stores artifacts in `~/NeuralFn/artifacts`, and allows the standard Vite dev origins. Configure the platform with environment variables as needed:

| Variable | Purpose | Default |
|----------|---------|---------|
| `NEURALFN_DATABASE_URL` | SQLAlchemy database URL. Use MySQL in shared environments. | `sqlite:///.../neuralfn.db` |
| `NEURALFN_REDIS_URL` | Optional Redis live state/event store. If unset, the server uses in-memory live state. | unset |
| `NEURALFN_CREATE_SCHEMA_ON_STARTUP` | Auto-create tables on app startup. Set to `0` if you want migration-only schema management. | `1` |
| `NEURALFN_SNAPSHOTS_DIR` | Filesystem location for persisted session snapshots. | `server/session_snapshots` |
| `NEURALFN_ARTIFACTS_DIR` | Filesystem location for saved artifacts. | `~/NeuralFn/artifacts` |
| `NEURALFN_ALLOW_ORIGINS` | Comma-separated CORS origins. Must include the frontend origin when using cookies. | `http://127.0.0.1:5173,http://localhost:5173` |
| `NEURALFN_SESSION_COOKIE_NAME` | Session cookie name used by the web app and API. | `neuralfn_session` |
| `NEURALFN_SESSION_TTL_SECONDS` | Session lifetime in seconds. | `1209600` |

`.gitignore` excludes the default SQLite file, downloaded datasets under `~/.cache/nfn/datasets/`, session snapshots, artifacts, local `.env` files, and common caches so they are not committed.

If you want migration-managed startup instead of auto-creating tables, run:

```bash
alembic upgrade head
```

and set:

```bash
export NEURALFN_CREATE_SCHEMA_ON_STARTUP=0
```

### Start the platform

Terminal 1 — backend:

```bash
uvicorn server.app:app --reload --port 8000
```

Terminal 2 — frontend:

```bash
cd editor
pnpm dev
```

Open <http://localhost:5173>.

### Running in Desktop Mode (Electron)

NeuralFn can also be run as a unified desktop application. In desktop mode, it runs completely offline, storing its database (`neuralfn.db`), snapshots, and training artifacts securely inside the user's OS application data directory. **No Redis or external service is required**—the app automatically boots with zero-config local Memory and SQLite storage.

#### Install Desktop dependencies:
```bash
# Install desktop packages
npm run desktop:install
```

#### Start the Desktop App locally:
To compile the React editor, copy assets, scan and bind to an open port, and launch the Electron frame:
```bash
npm run desktop:start
```

#### Package for Windows, Mac, and Linux:
To bundle everything (backend scripts, assets, electron launcher) into a standalone production installer or executable:
```bash
npm run desktop:build
```
The compiled artifacts will be generated in `desktop/dist/`.


### First-run workflow

1. On the first launch, the app checks `/api/bootstrap`. If no users exist, the login screen switches into **Create Admin Workspace** mode.
2. Creating the first admin also creates a default project and editor session, then signs the user in with an HTTP-only session cookie.
3. Subsequent logins reuse `/api/auth/login` and restore the last active project/session scope stored on the auth session.
4. Any authenticated user can create a personal project from the header controls. Each new project is seeded with a `Main session` automatically and becomes the active workspace immediately.
5. Admin users can additionally create users and manage project memberships from the Admin surface.

## Routed platform surfaces

After authentication, the app routes into the scoped shell under `/app`:

- `Editor` — `/app/projects/:projectId/sessions/:sessionId/editor`
- `Datasets` — `/app/projects/:projectId/sessions/:sessionId/datasets`
- `Runs` — `/app/projects/:projectId/sessions/:sessionId/runs`
- `Analytics` — `/app/projects/:projectId/sessions/:sessionId/analytics`
- `Admin` — `/app/admin`

The app shell keeps the active project and session in the header, lets authenticated users create a new personal project, updates the server-side active scope when you switch workspaces, and routes each surface to the matching project/session pair.

## Refresh-safe session restore

The editor is no longer an anonymous in-memory graph:

- editor routes always load a concrete `projectId` and `sessionId`
- the editor hydrates the session graph and revision from the backend on load
- autosave writes back through the session-scoped graph API
- if the backend detects a revision conflict, the client reloads the latest graph instead of silently overwriting it
- if an MCP agent is actively controlling the session, the UI shows a banner and reloads the graph after the agent releases control

## API surface

The platform API is mounted under `/api` and is split into dedicated routers:

- `/api/bootstrap` for initial app bootstrap state
- `/api/auth/*` for bootstrap-admin, login/logout, identity, and active-session selection
- `/api/admin/*` for user/project membership management
- `/api/projects/*` for project listing, creation, and analytics
- `/api/projects/{project_id}/datasets/*` for dataset catalog listing, download/upload, access grants, and deletion
- `/api/projects/{project_id}/sessions/{session_id}/*` for graph editing, dataset wiring, execution, tracing, templates, and agent status
- `/api/projects/{project_id}/sessions/{session_id}/runs/*` for training runs and run status

Legacy helper wrappers still exist in `server/routes.py` for older route-based tests, but all product-facing editor and MCP flows now use explicit project/session-scoped endpoints.

## MCP Server (AI agent integration)

NeuralFn ships with an [MCP](https://modelcontextprotocol.io/) server that exposes graph-editing and training operations as tools. The MCP server now authenticates against the platform and works against explicit project/session scopes instead of a single global graph.

### Prerequisites

- The FastAPI backend must be running on port `8000` before starting the MCP server.
- The MCP server needs valid platform credentials. Export them in the environment used to launch your MCP client:

```bash
export NEURALFN_MCP_EMAIL="admin@example.com"
export NEURALFN_MCP_PASSWORD="secret123"
# Optional if your API is not running at http://localhost:8000/api
export NEURALFN_BASE_URL="http://localhost:8000/api"
```

- MCP currently assumes the target project and session already exist. Create/select them in the web UI or via the HTTP API first.

### Configuration

Add the NeuralFn MCP server to your client configuration. The server uses inline script metadata (PEP 723), so `uv run` resolves the `mcp` dependency automatically.

**Codex project config** (`.codex/config.toml` in the project root):

```toml
[mcp_servers.neuralfn]
command = "uv"
args = ["run", "server/mcp_server.py"]
cwd = "/path/to/NeuralFn"
```

**Cursor** (`.cursor/mcp.json` in the project root):

```json
{
  "mcpServers": {
    "neuralfn": {
      "command": "uv",
      "args": ["run", "server/mcp_server.py"]
    }
  }
}
```

Codex reads MCP server definitions from `config.toml` rather than Cursor's `.cursor/mcp.json`. Project-scoped Codex MCP config only applies to trusted projects.

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "neuralfn": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/NeuralFn", "server/mcp_server.py"]
    }
  }
}
```

### MCP scope rules

- Most graph, node, edge, variant, execution, template, and training tools require both `project_id` and `session_id`.
- Dataset catalog tools (`list_datasets`, `download_dataset`, `set_dataset_access`, `delete_dataset`) are project-scoped and require `project_id`.
- `list_builtins()` is global and does not require project/session context.

### Available tools

**Graph** (`project_id`, `session_id`)

| Tool | Description |
|------|-------------|
| `get_graph` | Get the current graph summary for one project/session. |
| `replace_graph` | Replace the entire graph for one project/session. |
| `update_graph_settings` | Update graph name, training method, runtime, or config dicts. |
| `set_io` | Set which nodes are graph inputs/outputs. |

**Nodes** (`project_id`, `session_id`)

| Tool | Description |
|------|-------------|
| `list_builtins` | List available builtin neuron definitions. |
| `add_node` | Add a builtin neuron by id. |
| `add_custom_node` | Add a function node with custom Python source code and ports. |
| `add_subgraph_node` | Add an empty subgraph node (with internal input/output nodes). |
| `add_variant_node` | Add a node linked to a variant from the variant library. |
| `get_node` | Get the full details of a single node. |
| `update_node` | Update a node's name, source code, ports, or module config. |
| `delete_node` | Delete a node. |
| `update_node_positions` | Batch-update node canvas positions. |

**Edges** (`project_id`, `session_id`)

| Tool | Description |
|------|-------------|
| `add_edge` | Connect two node ports with optional weight/bias. |
| `update_edge` | Update an edge's weight and/or bias. |
| `delete_edge` | Delete an edge. |

**Variants** (`project_id`, `session_id`)

| Tool | Description |
|------|-------------|
| `list_variants` | List all variant families and versions in the current scoped graph. |
| `save_node_as_variant` | Save a subgraph node into the variant library. |
| `swap_node_variant` | Swap a node to a different variant version. |

**Execution & Training** (`project_id`, `session_id`)

| Tool | Description |
|------|-------------|
| `execute_graph` | Run the graph with scalar inputs. |
| `execute_trace` | Run the graph and trace intermediate outputs. |
| `trace_torch` | Trace a torch graph for tensor statistics. |
| `probe_node` | Probe a node's response curve. |
| `train_start` | Start training (surrogate/evolutionary/hybrid/torch). |
| `get_training_status` | Read the active training snapshot, latest loss, and recent events. |
| `poll_training_status` | Wait for the next training update by `event_id`, or until the run finishes. |
| `train_stop` | Stop the current training run. |
| `load_gpt_template` | Build and load a GPT/Llama/MoE graph in one call. |

**Datasets** (`project_id` or `project_id` + `session_id`)

| Tool | Description |
|------|-------------|
| `list_datasets` | List datasets visible to a project. |
| `download_dataset` | Download a Hugging Face dataset into the project catalog and optionally share it with other accessible projects. |
| `load_dataset_source` | Download/load datasets and wire them into a `dataset_source` node for one session graph. |
| `set_dataset_access` | Update which accessible projects can use one dataset. |
| `delete_dataset` | Delete a dataset from the project catalog. |

### Example: building a graph inside an existing session

An AI agent can build a simple sigmoid pipeline like this (shown as sequential tool calls):

```text
1. add_node(project_id="proj_123", session_id="sess_456", neuron_id="builtin-input", instance_id="in", position=[100, 200])
2. add_node(project_id="proj_123", session_id="sess_456", neuron_id="builtin-sigmoid", instance_id="sig", position=[350, 200])
3. add_node(project_id="proj_123", session_id="sess_456", neuron_id="builtin-output", instance_id="out", position=[600, 200])
4. add_edge(project_id="proj_123", session_id="sess_456", src_node="in", src_port=0, dst_node="sig", dst_port=0)
5. add_edge(project_id="proj_123", session_id="sess_456", src_node="sig", src_port=0, dst_node="out", dst_port=0)
6. set_io(project_id="proj_123", session_id="sess_456", input_ids=["in"], output_ids=["out"])
7. execute_graph(project_id="proj_123", session_id="sess_456", inputs={"in": [0.5]})
```

### Example: train an MoE on FineWeb via MCP

```text
1. load_gpt_template(project_id="proj_123", session_id="sess_456", name="fineweb_moe", preset="moe", config={"n_layer": 4, "n_head": 4, "n_embd": 128, "num_experts": 4, "top_k": 2})
2. load_dataset_source(project_id="proj_123", session_id="sess_456", hf_path="HuggingFaceFW/fineweb", hf_split="train", max_rows=10000, seq_len=64)
3. train_start(project_id="proj_123", session_id="sess_456", method="torch", epochs=10, learning_rate=0.001)
4. get_training_status(project_id="proj_123", session_id="sess_456")
5. poll_training_status(project_id="proj_123", session_id="sess_456", since_event_id=0, timeout_seconds=30)
```

## Testing

Run the Python test suite with:

```bash
python -m unittest discover -s tests
```

Useful targeted checks:

```bash
python -m unittest discover -s tests -p "test_platform_api.py"
python -m unittest discover -s tests -p "test_server_dataset_loading.py"
python -m unittest discover -s tests -p "test_server_nested_graphs.py"
```

The platform API suite covers bootstrap-admin, login/session scope behavior, refresh-safe graph restore, and revision conflict handling.

For frontend type/build validation:

```bash
cd editor
pnpm build
```

## Using built-in neurons

```python
from neuralfn import BuiltinNeurons, NeuronGraph, NeuronInstance

g = NeuronGraph()
g.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
g.add_node(NeuronInstance(BuiltinNeurons.sigmoid, instance_id="act"))
g.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
```

## Defining a custom neuron

```python
from neuralfn import neuron, Port

@neuron(
    inputs=[Port("x", range=(-5, 5), precision=0.01)],
    outputs=[Port("y", range=(0, 1), precision=0.001)],
)
def custom_sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))
```

## Graph execution

```python
from neuralfn import BuiltinNeurons, NeuronGraph, NeuronInstance, Edge

g = NeuronGraph()
g.add_node(NeuronInstance(BuiltinNeurons.sigmoid, instance_id="s1"))
# ... add more nodes and edges
result = g.execute({"input_node_id": (0.5,)})
```

## Nested graphs and mixed trainers

```python
from neuralfn import (
    BuiltinNeurons,
    Edge,
    HybridConfig,
    HybridTrainer,
    NeuronGraph,
    NeuronInstance,
    subgraph_neuron,
)

child = NeuronGraph(name="child", training_method="surrogate")
child.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
child.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
child.add_edge(Edge(id="child-edge", src_node="in", src_port=0, dst_node="out", dst_port=0))
child.input_node_ids = ["in"]
child.output_node_ids = ["out"]

root = NeuronGraph(name="root", training_method="frozen")
root.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="root_in"))
root.add_node(
    NeuronInstance(
        subgraph_neuron(child, name="child_block", input_aliases=["x"], output_aliases=["y"]),
        instance_id="child_block",
    )
)
root.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="root_out"))

trainer = HybridTrainer(root, HybridConfig(outer_rounds=3))
```

Subgraph nodes expose their ports from the nested graph’s designated `input_node_ids` and `output_node_ids`, and each graph picks its own `training_method`: `surrogate`, `evolutionary`, `frozen`, or `torch`.

## GPT / torch graphs

Use the GPT template generator when you want a causal language model that remains explorable in the editor. The templates expand into intricate graphs of token embedding, residual-mix, RMSNorm, attention, MLP (Dense or Mixture of Experts), skip-add, head, softcap, and token cross-entropy stages. Transformer blocks are represented as nested subgraphs via the Variant Library, allowing easy exploration of architecture choices. Torch graphs should use `training_method="torch"` and `runtime="torch"`. The torch trainer is CUDA-first.

Training data for GPT graphs is managed via a `dataset_source` node. Use the `Datasets` tab to download or upload datasets and choose which of your accessible projects can see them, then select those datasets from the `dataset_source` node side panel and connect its output roles to the network's inputs. For standard AR-style graphs that means `tokens` and `targets`; for semantic routing presets that means `tokens` and `targets` plus the shipped `semantic_data_source` node feeding `sem_targets`. Dataset-backed training now resolves from the saved graph node configuration, so the node is the source of truth rather than a temporary bottom-panel selector. The trainer will automatically tokenize the text, handle batching, and still auto-expand `vocab_size` for manual or tokenizer-less inputs when needed. Tokenizer-backed cached shard aliases are stricter now: NeuralFn validates shard token ids, tokenizer artifacts, and the graph/checkpoint vocab up front, and it fails fast if they disagree instead of silently resizing embeddings or crashing during decode. The recovery path for a bad cached alias is to delete and rebuild or re-download it with matching tokenizer artifacts.

## Training

**Surrogate (gradient-based):**
```python
from neuralfn import SurrogateTrainer
from neuralfn.trainer import TrainConfig

trainer = SurrogateTrainer(graph, TrainConfig(epochs=300))
losses = trainer.train(X, Y)
```

**Evolutionary:**
```python
from neuralfn import EvolutionaryTrainer
from neuralfn.evolutionary import EvoConfig

evo = EvolutionaryTrainer(graph, EvoConfig(generations=200))
losses = evo.train(X, Y)
```

## Built-in neurons

The editor palette and MCP `list_builtins` / `add_node` tools draw from a single catalog defined in [`neuralfn/builtins.py`](neuralfn/builtins.py) (`BuiltinNeurons.all()`), also served at `/builtins` for the web app. **Scalar** entries (`kind: function`) run on the scalar graph runtime and work with surrogate or evolutionary training. **Torch module** entries (`kind: module`) wrap `torch.nn` stages and expect a graph with `runtime: "torch"` and `training_method: "torch"` (see [GPT / torch graphs](#gpt--torch-graphs) and `examples/gpt_graph.py`).

Import in Python:

```python
from neuralfn import BuiltinNeurons

BuiltinNeurons.sigmoid          # scalar
BuiltinNeurons.linear_module    # torch module — pair with runtime="torch"
```

**Duplicate display name:** the catalog contains **two** definitions whose `name` field is `gelu`: the scalar `@neuron` and the tensor module (`module_type: "gelu"`). In JSON they differ by `kind` and (for the module) `module_type`; when resolving by name in code, `BUILTIN_MAP` keeps a single winner—prefer `BuiltinNeurons.gelu` vs `BuiltinNeurons.gelu_module` explicitly.

### Scalar activations and unary ops

- **sigmoid** — Logistic activation \(1 / (1 + e^{-x})\).
- **relu** — ReLU: \(\max(0, x)\).
- **tanh_neuron** — Hyperbolic tangent.
- **threshold** — Step: 1 if \(x \ge 0\), else 0.
- **identity** — Passthrough.
- **negate** — Unary negation.
- **gaussian** — \(e^{-x^2}\).
- **log_neuron** — Natural log with a small floor on \(x\) (Python: `BuiltinNeurons.log_neuron`).
- **leaky_relu** — Leaky ReLU (small slope for \(x < 0\)).
- **prelu** — Parametric ReLU–style slope for negative inputs (fixed coefficient in this scalar form).
- **relu6** — ReLU clipped at 6.
- **elu** — Exponential linear unit for negative inputs.
- **selu** — Scaled ELU constants for self-normalizing-style behavior at scalar resolution.
- **gelu** (scalar) — Gaussian error linear unit using `erf`.
- **silu** — SiLU / Swish: \(x \cdot \sigma(x)\).
- **mish** — \(x \cdot \tanh(\text{softplus}(x))\).
- **softplus** — Smooth ReLU-like \(\log(1 + e^x)\).
- **softsign** — \(x / (1 + |x|)\).
- **hard_sigmoid** — Piecewise-linear sigmoid approximation.
- **hard_tanh** — Piecewise-linear tanh approximation.
- **hard_swish** — Piecewise-linear Swish approximation.

### Scalar binary ops and two-logit heads

- **add** — Sum of two inputs.
- **multiply** — Product of two inputs.
- **softmax_2** — Softmax over two scalars; two probability outputs.
- **logsoftmax_2** — Log-softmax over two scalars; two log-probability outputs.

### Graph terminals

- **input** — Graph input terminal (Python: `BuiltinNeurons.input_node`).
- **output** — Graph output terminal (Python: `BuiltinNeurons.output_node`).

### Torch — embeddings and positions

- **token_embedding** — Embedding lookup: token IDs to hidden states; second output exposes embedding weights (for tied heads).
- **absolute_position_embedding** — Adds learned position vectors along the sequence (expects token-derived stream shape).

### Torch — linear and MLP blocks

- **linear** — Trainable dense layer \(y = xW + b\) (dimensions and bias from `module_config`).
- **mlp_relu2** — Two-layer MLP with ReLU-squared activation between projections (width from `module_config`).
- **gelu** (module) — Tensor GELU activation (`module_type: "gelu"`; Python: `BuiltinNeurons.gelu_module`).
- **swiglu** — SwiGLU-style gated MLP block (LLaMA-style; width/multiple-of from `module_config`).

### Torch — normalization and regularization

- **rms_norm** — RMS normalization over the last dimension.
- **layer_norm** — Layer normalization over the last dimension.
- **dropout** — Dropout during training (rate `p` in `module_config`).

### Torch — attention and residuals

- **reshape_heads** — Reshape projected hidden states into multi-head layout for attention.
- **merge_heads** — Merge per-head tensors back to model width.
- **repeat_kv** — Repeat grouped key/value heads to match query head count (GQA).
- **rotary_embedding** — Apply RoPE to Q and K tensors (two inputs, two outputs).
- **qk_gain** — Learned per-head scaling on the query stream before attention scores.
- **scaled_dot_product_attention** — Multi-head scaled dot-product attention (Q, K, V in; causal mask from `module_config`).
- **causal_self_attention** — Full causal self-attention block (projections, RoPE, GQA repeat, SDPA, output projection) as one stage.
- **residual_mix** — Learned per-channel blend of main path and skip (`x` vs `x0`).
- **residual_add** — Skip connection: residual plus learned per-channel scaled delta.
- **kv_cache_read** — Concatenate prior cached K/V with current K/V along the sequence dimension when caches are provided; otherwise passthrough.
- **kv_cache_write** — Identity on K/V outputs; used as a structural marker in inference-style graphs.

### Torch — language-model head and loss

- **tied_lm_head** — Projects hidden states to logits using supplied embedding weights (second input).
- **lm_head** — Standalone linear head to vocabulary logits.
- **logit_softcap** — Stabilizes logits (e.g. tanh-based cap) before softmax/loss.
- **token_cross_entropy** — Cross-entropy loss between logits and integer token targets.

### Torch — mixture-of-experts

- **router_logits** — Linear router from hidden state to per-expert scores.
- **topk_route** — Softmax over router, then top-k expert weights and indices.
- **expert_dispatch** — Sparse expert MLP (SwiGLU per expert) weighted by routing.
- **expert_combine** — Identity passthrough for graph wiring after dispatch.
- **load_balance_loss** — Auxiliary load-balancing loss from router statistics; passes router logits through.
- **aux_loss_add** — Adds scaled auxiliary tensor loss to the main scalar loss (`coef` in `module_config`).

### Torch — data source

- **dataset_source** — No inputs; emits `tokens` and `targets` from configured project datasets (`dataset_names`, `seq_len` in `module_config`). See [GPT / torch graphs](#gpt--torch-graphs) for wiring and the Datasets UI.

### Alphabetical index (selected catalog entries)

`absolute_position_embedding`, `add`, `aux_loss_add`, `causal_self_attention`, `dataset_source`, `dropout`, `elu`, `expert_combine`, `expert_dispatch`, `gaussian`, `gelu` (scalar function), `gelu` (torch module), `hard_sigmoid`, `hard_swish`, `hard_tanh`, `identity`, `input`, `kv_cache_read`, `kv_cache_write`, `layer_norm`, `leaky_relu`, `linear`, `lm_head`, `load_balance_loss`, `log_neuron`, `logit_softcap`, `logsoftmax_2`, `merge_heads`, `mish`, `mlp_relu2`, `multiply`, `negate`, `output`, `prelu`, `qk_gain`, `relu`, `relu6`, `repeat_kv`, `reshape_heads`, `residual_add`, `residual_mix`, `rms_norm`, `rotary_embedding`, `router_logits`, `scaled_dot_product_attention`, `selu`, `sigmoid`, `silu`, `softmax_2`, `softplus`, `softsign`, `swiglu`, `tanh_neuron`, `threshold`, `tied_lm_head`, `token_cross_entropy`, `token_embedding`, `topk_route`.

For the full, current builtin catalog including the experimental JEPA semantic modules, see [`docs/python-sdk/builtins.md`](docs/python-sdk/builtins.md) and [`neuralfn/builtins.py`](neuralfn/builtins.py).
