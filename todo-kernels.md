# NeuralFn: modern-LLM kernel additions TODO

Tracks new builtin neurons / Stages to add into NeuralFn so it covers the
modern LLM training and inference surface. These currently have **no**
registration in `neuralfn/builtins.py` or `neuralfn/torch_backend.py`.

**Long-term plan:** every new Stage starts as a PyTorch reference
implementation here, then gets re-pointed at a corresponding `llm.kittens`
kernel once the optimised back-end lands. The kernel-side checklist lives in
`../../open-source/llm.kittens/nfn-coverage-todo.md` (§21–§33 mirror the
sections below).

Per addition, tick when each piece is in place:

- **builtin** — `NeuronDef` registered in `neuralfn/builtins.py` with input /
  output ports
- **stage** — `nn.Module` Stage class in `neuralfn/torch_backend.py` plus
  `build_module` dispatch
- **template** — wiring into `neuralfn/torch_templates.py` where the op
  belongs to a higher-level block (skip for orthogonal additions)
- **test** — unit coverage under `tests/`
- **kittens** — switched to call `llm.kittens` instead of the local PyTorch
  reference (tick when the optimised path is live)

Sections numbered to match `nfn-coverage-todo.md` §21–§33.

---

## §21. Inference / decoding

- [ ] **top_k_sampling** — keep top-k logits, renormalise, sample
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **top_p_sampling (nucleus)** — cumulative-prob threshold sampling
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **min_p_sampling** — prob ≥ min_p · max_prob threshold
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **typical_p_sampling** — locally typical sampling
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **temperature_scaling** — `logits /= T`
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **repetition_penalty** — penalise previously-emitted tokens
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **logit_bias** — additive per-token bias map
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **paged_attention** — vLLM-style block-table KV indexing
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **continuous_batching_dispatch** — multi-request gather/scatter
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **chunked_prefill** — long-prompt chunked prefill stage
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **speculative_decoding (draft + verify)** — token tree verify
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **beam_search_step** — per-step beam scoring + reorder
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **grammar_constrained_decode** — FSM/grammar accept-set logit mask
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens

## §22. Attention variants (modern)

- [ ] **sliding_window_attention** — Mistral/Gemma local window
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **multi_latent_attention (MLA)** — DeepSeek-style compressed KV
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **cross_attention** — encoder/decoder Q over external K/V
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **alibi_attention** — additive linear position bias
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **linear_attention (Based / Hedgehog / RetNet / RWKV)** — kernel-feature
      linear attention with state
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **block_sparse_attention** — Longformer local + global
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **streaming_attention_sinks** — StreamingLLM persistent sinks
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **ring_attention** — context-parallel shard-and-rotate
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **native_sparse_attention (NSA)** — DeepSeek-V3.2 native sparse pattern
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **differential_attention** — two-softmax-branch subtraction
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **varlen_attention (cu_seqlens)** — packed-sequence attention API
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens

## §23. RoPE / position-encoding variants

- [ ] **yarn_rope_scaling** — YaRN context extension
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **ntk_aware_rope_scaling** — frequency-base interpolation
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **linear_rope_scaling (PI)** — position interpolation
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **rope_2d** — 2D RoPE for vision/multimodal
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **alibi_bias** — precompute/apply ALiBi slopes
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **xpos / nope** — alternative position schemes
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens

## §24. Optimizers beyond AdamW + Muon

- [ ] **Lion** — sign-momentum optimizer
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **Sophia** — diagonal-Hessian clipped second-order
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **Shampoo / SOAP** — preconditioned optimizer
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **AdEMAMix** — Adam with mixed slow/fast EMA
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **AdamW8bit** — block-quantized optimizer states
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **Adafactor** — factored second-moment
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **ema_update** — fast EMA weight update
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **swa_average** — Stochastic Weight Averaging accumulator
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens

## §25. FP8 / low-precision (SM120 / Blackwell)

- [ ] **fp8_gemm (E4M3 / E5M2)** — Blackwell tensor-core FP8 matmul
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **fp8_quantize** — bf16/fp32 → fp8
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **fp8_dequantize** — fp8 → bf16/fp32
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **mxfp8 / mxfp4 GEMM** — microscaled FP8/FP4 matmul
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **stochastic_rounding** — bf16/fp8 weight update with SR
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **amax_history_tracking** — TE-style amax update
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens

## §26. Distributed beyond ZeRO

- [ ] **all_to_all** — MoE expert-parallel token shuffle
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **column_parallel_linear** — TP linear sharded along output dim
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **row_parallel_linear** — TP linear sharded along input dim
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **sequence_parallel_norm** — SP-sharded LN/RMSNorm + all-gather
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **pipeline_send_recv** — 1F1B / zero-bubble PP helpers
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **context_parallel_allgather** — ring-attn coordination
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens

## §27. Modern MoE features

- [ ] **auxfree_load_balancing** — DeepSeek-V3 bias-adjusted routing
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **capacity_factor_dispatch** — capacity-limited dispatch with drop
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **expert_parallel_dispatch** — top-k routing + all-to-all (depends on §26)
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **soft_moe** — continuous learned-slot MoE
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **mixture_of_depths (MoD)** — token-level early exit
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens

## §28. Norm variants

- [ ] **group_norm** — per-group LayerNorm
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **dyt (Dynamic Tanh)** — recent LN replacement
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **qk_norm (fused)** — RMSNorm on Q and K fused into attention
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens

## §29. Activation gates beyond SwiGLU

- [ ] **geglu** — gated GELU
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **reglu** — gated ReLU
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **solu** — softmax-gated linear unit
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens

## §30. Vision / multimodal

- [ ] **conv2d** — general 2D conv
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **patch_embed_2d** — Conv2d + flatten + linear projection
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **abs_pos_embed_2d** — 2D learned positional embedding
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **cross_modal_cross_attention** — vision↔text bridges
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens

## §31. Long-context KV management

- [ ] **h2o_eviction** — Heavy-Hitter Oracle KV pruning
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **snapkv** — prompt-aware KV compression
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **landmark_attention** — landmark-token summaries
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **infini_attention** — segment-level compressed memory
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **sink_token_cache** — persistent attention sinks
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens

## §32. Gradient / training infrastructure

- [ ] **gradient_checkpoint_hook** — re-materialisation entry/exit
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **selective_recompute** — Flash-Attn-style activation drop + recompute
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **loss_scale_dynamic** — dynamic loss scaling with overflow detection
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **gradient_accumulate** — fused add-into-grad-buffer
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens

## §33. Tokenizer & data path

- [ ] **gpu_bpe_tokenizer** — on-device byte-pair tokenisation
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **sequence_packing** — variable-length pack + cu_seqlens
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
- [ ] **document_causal_mask (cu_seqlens)** — per-document causal mask for
      packed batches
      - [ ] builtin  - [ ] stage  - [ ] template  - [ ] test  - [ ] kittens
