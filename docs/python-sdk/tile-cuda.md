# neuralfn.tile_cuda

Optional CUDA Tile backend for NeuralFn. The package provides configuration, diagnostics, coverage reporting, strict graph coverage checks, and an opt-in PyTorch extension build path for CUDA Tile scalar, module, optimizer, and runtime kernels. PyTorch remains the authoritative fallback for unsupported devices, dtypes, shapes, and tensor contracts.

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

The source build path is intentionally opt-in:

```bash
NFN_TILE_CUDA_BUILD=1 NFN_TILE_CUDA_ARCH=sm_120 nfn kernels doctor
```

CUDA Tile builds require CUDA Toolkit 13.3 or newer, `cuda_tile.h`, C++20, `nvcc --enable-tile`, `ninja`, and a CUDA-capable PyTorch process. Install the optional build extra with:

```bash
pip install -e ".[tile-cuda]"
```

## Diagnostics

```python
from neuralfn.tile_cuda import tile_cuda_diagnostics

print(tile_cuda_diagnostics().to_dict())
```

Diagnostics report the `nvcc` path, CUDA version, `cuda_tile.h` availability, `torch.cuda` availability, GPU name, compute capability, whether source builds are enabled, and whether the optional extension is already loaded.

## Implemented Kernels

The current registry accounts for all 138 training-relevant NeuralFn entries: 129 Tile-covered kernels or Tile compositions, 7 host-only interface/source entries, and 2 delegated compiled-graph calls. There are no `torch_fallback` entries in the default registry.

Scalar functions:

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

Binary and binary-pair Tile kernels require same-shaped contiguous inputs. Vector module kernels require a last dimension matching the stage parameter vector, except `qk_gain`, which expects `[B, H, ...]` input with a gain vector of length `H`. Norm kernels cover contiguous float32 rows with last dimension up to 1024, plus `group_norm` on `[B,S,D]` when `S * group_dim <= 1024`. Layout and indexing kernels cover contiguous float32 `reshape_heads`, `merge_heads`, `repeat_kv`, `rotary_embedding`, `absolute_position_embedding`, `token_embedding`, `byte_patch_embed`, `causal_chunk_state` prefix/mean chunk states, and KV cache copy/concat paths. Projection kernels cover contiguous float32 `linear`, `bitlinear_ternary` with PyTorch-reference STE quantization, `fp8_linear` with PyTorch-reference FP8 weight STE, `mx_linear` with PyTorch-reference MXFP4/MXFP8 weight STE, `nf4_linear` with `compute_dtype="fp32"` and `dropout=0`, `ttt_linear`, `lora_linear` with `dropout=0`, `randmap_adapter`, `lm_head`, and `tied_lm_head` forwards with autograd-composed input/weight/bias gradients, including `kv_pca_encode`, `kv_pca_decode`, `jepa_projector`, `jepa_predictor`, `mlp_relu2`, `swiglu`, `geglu`, `reglu`, `solu` row-softmax gating, `expert_dispatch` per-expert SiLU MLP projections, `mamba` input/output projections, and `universal_transformer` attention/MLP projections. Attention kernels cover contiguous CUDA float32 `scaled_dot_product_attention`, `sliding_window_attention`, `block_sparse_attention`, `streaming_attention_sinks`, deterministic `native_sparse_attention`, `differential_attention`, `causal_self_attention`, `fused_causal_attention`, `multi_latent_attention`, and `routed_attention_experts` with key sequence length up to 1024, causal or non-causal masking, `dropout_p=0`, grouped-query attention when query heads are divisible by key/value heads, right-aligned sparse masks for cache-compatible query/key lengths, split Q/K dimensions for differential attention, and Tile-composed projection/RoPE/output paths for self-contained attention stages. KV quantization kernels cover same-shaped contiguous float32 K/V rows with `head_dim <= 512`, packing quantized values plus per-row scale and unpacking tensors shaped `[..., 2*head_dim+1]`. Semantic projector kernels cover flat and chunked topic-head, signature, and residual projections while preserving the per-dimension topic logits contract. Semantic hash kernels cover contiguous float32 semantic vectors and chunk vectors with up to 62 hash planes per table, returning int64 bucket IDs without gradients. Route kernels cover contiguous float32 route weights plus int64 route indices, `topk_route` covers contiguous float32 logits with `top_k <= 64`, `semantic_hash_router` covers unforced hash/topic routing through native top-k selection while preserving the PyTorch forced-target ordering path, `semantic_moe_jepa_evo_router` covers chunk-level shared/semantic/free route-logit construction with Tile free-expert projection and PyTorch-compatible candidate ordering, `auxfree_load_balancing` covers native per-expert bias addition with device-side no-grad bias updates, supervised semantic route BCE for `route_selection_loss`, route distillation KL reduction for `route_distillation_loss` with PyTorch reference preprocessing for topic dimensions wider than 1024 terms, and `attentionless_decoder` covers bucket-conditioned expert-output logits with native bucket embedding plus output projection. `dropout` uses Tile identity for inference and `p=0`; `random_timesteps`, `mask_scheduler`, and `jepa_mask` use deterministic counter-based device random generation so CPU/GPU parity does not depend on global PyTorch RNG state. `adamw_step`, `muon_newton_schulz`, `muon_step`, and `split_optimizer_step` cover optimizer update math through Tile primitives or Tile-compatible tensor composition with CPU fallback. `ema_update` covers same-shaped contiguous CUDA float32 target/source tensors and updates the target in-place under `no_grad`. `gradient_accumulate` covers same-shaped contiguous CUDA float32 accumulation buffers and gradients with a scalar scale factor. `gradient_clip_norm` covers multi-tensor global L2 norm reduction plus in-place scaling for contiguous CUDA float32 gradients. `latent_pool` covers masked JEPA latent pooling with mean fallback for empty masks, and `token_cross_entropy`, `masked_token_cross_entropy`, `sequence_logp`, `latent_mse_loss`, `semantic_alignment_loss`, `dpo_pairwise_loss`, `ppo_clipped_loss`, `gae_compute`, `preference_bce_loss`, `load_balance_loss`, `route_balance_loss`, `route_distillation_loss`, and `softmax_distillation_loss` produce scalar losses or log-prob reductions through Tile reductions. DPO reward outputs remain detached to match the PyTorch stage contract. Non-CUDA tensors, unsupported dtypes, non-contiguous tensors, broadcasted inputs outside these contracts, and unsupported runtime contracts fall back to PyTorch unless strict mode is enabled.

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

Every entry must be accounted for as one of:

- `tile`: implemented CUDA Tile kernel
- `torch_fallback`: not yet implemented in Tile; PyTorch remains authoritative. The default registry currently has none.
- `host_only`: source or interface node with no device compute contract
- `delegated`: covered by another compiled graph or fused implementation
- `planned`: reserved for future work with an explicit reason

## Training Hot Path

Real training tensors must not pass through graph editor nodes. `CompiledTorchGraph` compiles graph topology and edge routing once, then forwards tensors through fixed modules and precomputed routing plans. Future CUDA Tile graph execution must preserve the same invariant.

`CompiledTorchGraph(..., kernel_backend="tile_cuda", tile_cuda_strict=True)` validates coverage before batches run. Any node still marked `torch_fallback` or `planned` fails at compile time in strict mode.

## Examples

Checked-in examples live under `examples/tile_cuda/`. Use the CLI to list or regenerate them:

```bash
nfn kernels examples
nfn kernels examples --write --output-dir examples/tile_cuda
nfn kernels bench --device auto --iterations 200
```
