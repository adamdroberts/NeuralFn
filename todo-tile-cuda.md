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
