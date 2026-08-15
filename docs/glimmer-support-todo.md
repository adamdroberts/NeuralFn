# Muse Glimmer native support: implementation status and remaining work

> **Status (2026-08-15): the planned software paths are implemented.** NeuralFn now has
> an exact `muse_glimmer` GPT preset, strict BF16 and official GGUF conversion,
> resident C++ CPU and whole-model CUDA text execution, K-Quant-Dynamic and
> K-Quant-17GB, VRAM-aware precision selection, DFlash speculative decoding,
> native pretraining/SFT/LoRA/QLoRA/DPO/reward/PPO, frozen K-Quant LoRA
> SFT/DPO, target-bound DFlash distillation, multi-GPU pipeline training, and
> CPU/CUDA vision execution.
>
> Capability claims remain independent and fail closed. Real CUDA 13.3.33 NVCC
> `sm_80`, `sm_89`, `sm_90`, and `sm_120` source builds now prove the
> normal/strict libraries and all Glimmer ABI v1 entry points compile.
> A source-built RTX 5090 run now proves real-device kernel parity under all
> four compute-sanitizer tools and full-size 32-GB-class Dynamic+DFlash+mmproj
> execution through 8K with zero CPU model-compute rows. A pinned llama.cpp
> full-size target-only raw-prompt check also matches all 16 greedy token IDs.
> An explicit K-Quant-17GB run on that larger card now completes the 24-GB
> profile tier with a measured 20,359,217,152-byte peak CUDA delta. Full
> logit/chat/DFlash oracle coverage and an 80-GB-or-larger BF16 run remain
> release gates. Profile tiers are minimum capacities, so the result retains
> both its 24-billion-byte tier and the card's real total. The official GGUF `mmproj`
> remains still-image-only because its temporal patch projection is collapsed;
> full-BF16 image/video vision can execute on CPU or whole-model CUDA.
>
> The final accepted current-source strict sidecar
> (`d15e946e16b1ef5b643a4556f8f719e49efa1466ad12c4957dbcaaa73953e994`)
> also passed the full direct-device probe under memcheck, synccheck,
> initcheck, and racecheck. On the canonical K-Quant-17GB target, a bounded
> 41-token ATEM prompt used the complete 52-layer Muse-Glimmer-30B (no smaller
> surrogate) and measured 78.608 target-only and **271.244 DFlash tok/s
> median** over ten measured 32-token trials after one warmup. The DFlash mean
> was 271.792 tok/s (269.303–274.820). Both modes returned the same canonical
> token hash; each DFlash trial accepted 28/34 proposals and used 37 target
> rows. A same-final-binary A/B measured 264.724 tok/s with only the cooperative
> multirow dual-RMS megakernel disabled, so the retained kernel adds 6.520
> tok/s (+2.46%) with identical output, acceptance, memory, and zero-CPU
> counters. Earlier exact controls retain the 7.87% target megakernel gain and
> the packed-weight-hoist gain (158.779 off versus 235.657 on). The
> earlier 256.69 DFlash number used a non-canonical approximate verifier and is
> retracted. The exact numbers are 4.95% and 16.21% above Meta's published
> 74.9/233.4, although that upstream workload is undisclosed. Against a
> directly reproduced pinned llama.cpp run on the same local workload,
> NeuralFn is 1.04% faster target-only and 20.18% faster with DFlash; no
> reproducible ExecuTorch comparator was available.

This file began as the implementation plan for
`meta-models/Muse-Glimmer-30B`. It now records what landed, the exact kernel
surface, and the smaller set of work that remains. A checked item means code
and deterministic tests exist. It does not substitute for a live hardware
benchmark or an upstream full-size model parity run.

## 1. Capability matrix

| Capability | Status | Boundary |
| --- | --- | --- |
| Exact GPT template | **Implemented** | `muse_glimmer` preserves asymmetric Q/K/V widths, local/local/local/global attention, NoPE global layers, Q/K RMS + scale, gated attention, four centered norms, decomposed SwiGLU, untied head, multiplier, and softcap. |
| Torch decoder training | **Implemented** | AR, masked SFT, LoRA, NF4 QLoRA, DPO with a frozen reference, reward modeling, and PPO use the shared exact Glimmer body. |
| Torch DFlash and vision graphs | **Implemented** | Separate assistant and vision/media-fusion graph builders; neither is silently inserted into an ordinary text root. |
| Native C++/CUDA pretraining | **Implemented, single-device or pipeline-parallel** | Exact 627-tensor text layout, uint32 records, activation recomputation, AdamW, strict save/resume, source SHA binding, and contiguous NCCL stages. The production 8-stage plan fits its declared 80-GB-class budget; live hardware validation remains pending. |
| Native full SFT | **Implemented, single-device or pipeline-parallel** | Structured uint32 records carry targets, loss masks, boundaries, and exact ATEM lineage. Distributed mode is full-BF16 only. |
| Native LoRA / QLoRA SFT | **Implemented** | All eight projection roles are supported. QLoRA freezes a deterministic NF4 group-64 base and updates only LoRA matrices. |
| Native DPO / reward / PPO | **Implemented, single device** | DPO uses frozen-reference sequence log-probabilities; reward training uses a masked scalar head; PPO performs online rollouts with frozen reference/reward models, per-token log-probabilities, KL, GAE, clipped minibatch updates, and strict resume. |
| K-Quant adapter tuning | **Implemented for LoRA SFT/DPO** | Official GGUF bytes remain immutable; packed forward/`dX`, adapter-only state, exact profile/tensor-table lineage, and same-base frozen DPO reference are enforced. Packed PPO/reward and lossy merge remain gated. |
| DFlash distillation | **Implemented in the Torch trainer** | A separate assistant-only trainer freezes the target, captures five taps, samples anchors, supports D-PACE/decay and self-logit KD, resumes exactly, audits greedy acceptance, and exports a target-bound native BF16 assistant. It does not claim Meta's unpublished training provenance. |
| Distributed 30B training | **Implemented for full-BF16 AR/SFT pipeline parallelism** | One process/device per contiguous layer stage, NCCL P2P plus global reductions, per-rank admission, authenticated rank shards, atomic DONE marker, and strict distributed resume. Adapter/preference pipeline modes and tensor/data parallelism remain separate future work. |
| Resident target CPU | **Implemented** | BF16 and official mixed F32/Q4_K/Q5_K/Q6_K profiles use hybrid local-ring/global-full KV caches and transactional verification. |
| Resident target CUDA | **Implemented for text** | Target weights and model compute stay on the selected CUDA device. The current ABI uses FP32 activation buffers with BF16 or packed resident weights, grouped packed-projection megakernels, and one captured graph replay per repeated greedy token. |
| K-Quant-Dynamic / K-Quant-17GB | **Implemented** | Strict GGUF v3 parser, authenticated canonical profiles, exact packed CPU/CUDA dequant/GEMM dispatch, no whole-model dequantization. |
| Automatic weight precision | **Implemented** | `auto` is quality-first and byte-budgeted on CUDA; explicit values are strict pins. CPU `auto` chooses the authenticated primary. |
| DFlash speculative decoding | **Implemented** | BF16 and packed assistants, CPU/CUDA assistant execution, target block verification, greedy and lossless sampled acceptance, and atomic cache commit/crop. |
| Native LoRA deployment | **Implemented** | Strict adapter inspection/attachment and direct CPU/CUDA deltas on Q/K/V/O/gate and MLP projections. Adapted targets reject an unbound stock DFlash assistant. |
| Image inference | **Implemented on CPU/CUDA** | Embedded full-BF16 vision and official K-Quant `mmproj`; Chat Completions accepts bounded base64 image data URLs. CUDA companion load is atomic and requires vision ABI v1. |
| Video inference | **Implemented for full BF16 CPU/CUDA Python API** | Exact 2 FPS / 96-frame sampling, temporal-2 patching, timestamps, prompt expansion, and placeholder fusion. The collapsed GGUF `mmproj` remains `video=false`. |
| Whole-model CUDA vision | **Implemented; packed mmproj validated on 32-GB hardware** | Device-resident packed/BF16 vision weights, position preparation, wide LayerNorm, 2-D RoPE full/window attention, pixel shuffle, adapter/projection and final RMS execute through vision ABI v1 with no request-time CPU model compute. The RTX 5090 qualification covers the official packed mmproj; full-BF16 80-GB validation remains pending. |

The effective runtime capability is the intersection of the authenticated
manifest and binding-reported ABI. A manifest cannot grant a capability on its
own, and an available kernel cannot bypass an artifact hash or compatibility
failure.

## 2. Pinned upstream contract

The implementation and fixtures use immutable references:

- Main model/tokenizer/config revision:
  `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`.
- DFlash assistant revision:
  `e8192f3a8f617f74be2ce220360c89ef4789f39f`.
- Official GGUF bundle revision:
  `43c7eadd41352a299ea8e0a36b3157978dd63596`.
- Transformers semantic oracle:
  `d1123114da1ab4395198146f4f84dae7fe8b693e`.
- Secondary llama.cpp GGUF/layout oracle: commit `62bf73d`; invoke with
  `--jinja` so the canonical ATEM template is honored.

The official target contract is vocabulary 202,048; hidden size 6,656; FFN
19,968; 52 layers; 32 query heads; 2 KV heads; explicit head dimension 128;
2,048 local window; and maximum context 131,072. Local layers use RoPE theta
500,000 and global layers use NoPE. Query normalization applies factor 3.87.
The final transform is:

```text
20 * tanh((0.19611613513818404 * raw_logits) / 20)
```

The DFlash contract is five layers, 32 Q / 8 KV heads, block size 16, 15
proposal positions, mask token 201818, and target residual taps at zero-based
layers `[1, 13, 25, 37, 49]`. GGUF metadata stores their one-based equivalents
`[2, 14, 26, 38, 50]`; conversion normalizes this exactly once.

Canonical packed artifacts:

| Profile | Artifact | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `k-quant-17gb` | `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` | 16,756,683,904 | `4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e` |
| `k-quant-dynamic` | `Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf` | 19,653,960,832 | `ac7023d6a4c704eb9af54ab53e476a66b7f5b6c0ef2fc4a8dde5253c291a6c38` |
| DFlash | `dflash-Muse-Glimmer-30B-Q4_K_M.gguf` | 1,631,208,128 | `b2e808bf656086fe86bd0d0bd990f01d33e377537a07c02d45371517c8b264ef` |
| Vision | `mmproj-Muse-Glimmer-30B-Q4_K_M.gguf` | 1,400,328,928 | `f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6` |

Both main GGUFs are version 3, architecture `muse-glimmer`, quantization
version 2, file type 15, and 731 tensors. The 17GB mix is 313 F32 / 365 Q4_K /
1 Q5_K / 52 Q6_K. Dynamic is 313 F32 / 51 Q4_K / 130 Q5_K / 237 Q6_K.
`Q4_K_XL` is a profile filename, not a distinct tensor encoding; execution
dispatches each tensor by its authenticated table.

## 3. Implemented GPT template and Torch graph

- [x] Added `BackboneType="muse_glimmer"` and backward-compatible explicit
  head, attention-width, FFN-width, layer schedule, norm, gate, output, vision,
  and DFlash fields in [`config.py`](../neuralfn/config.py).
- [x] Added exact local and global attention variants, four-norm decoder blocks,
  checkpoint-addressable SwiGLU projections, post-embedding RMS, untied LM
  head, and ordered logit transform in
  [`torch_templates.py`](../neuralfn/torch_templates.py).
- [x] Added `build_muse_glimmer_assistant_graph()`,
  `build_muse_glimmer_dflash_distillation_graph()`,
  `build_muse_glimmer_vision_graph()`, and media fusion as separate graph
  contracts.
- [x] Shared one Glimmer body across AR, logits, hidden-state, SFT, DPO, reward,
  and PPO wrappers; DPO policy weights are shared and its reference is frozen.
- [x] Added exact Torch stages for asymmetric attention, centered/scaleless
  normalization, DFlash block attention, the vision tower, perception
  projection, and placeholder scatter.
- [x] Registered the preset in the Python/C++ catalogs, editor dropdown, SDK
  docs, framework docs, and agent skills.
- [x] Preserved inline-subgraph fallback for cross-preset family collisions.
- [x] Added the preset to all mandatory preset tests and ordered pair tests.

The public template entry point is:

```python
from neuralfn.torch_templates import build_gpt_template_payload

payload = build_gpt_template_payload(
    "glimmer",
    {"preset": "muse_glimmer"},
)
```

## 4. Formats, tokenizer, and migration

- [x] Added versioned little-endian uint32 token shards and structured SFT
  records, while retaining uint16 read compatibility. Tests cover IDs 65535,
  65536, 201818, and 202047 plus corrupt/endian/range failures.
- [x] Added exact `tokenizer.json` loading, byte-safe incremental decode, added
  token validation, deterministic ATEM rendering, tokenizer/template hashes,
  and EOS `[200001, 200008]`. EOM 200007 is not an EOS.
- [x] Added streaming, bounded-memory safetensors conversion for all 627 text
  tensors, 809 vision tensors, and 58 assistant tensors. Conversion validates
  index/shard hash, name, shape, dtype, duplication, overlap, and total bytes.
- [x] Added strict GGUF v3 parsing and tensor maps for both main profiles,
  DFlash, and mmproj. Unknown metadata/type/layout and noncanonical profile
  hashes fail closed.
- [x] Added additive manifest-v1 fields for checkpoint variants, companion
  checkpoints, speculative decoding, kernel profiles, memory profiles, and
  compatibility bindings while preserving the original `checkpoint` shape.
- [x] Added strict target/assistant/mmproj/adapter digest allowlists and exact
  tokenizer/config/processor/interface binding.

Migration commands:

```bash
# BF16 target or full conditional-generation bundle
nfn migrate muse-glimmer-to-native \
  --source /models/Muse-Glimmer-30B \
  --component full \
  --output-dir artifacts/glimmer-bf16

# Official packed mains plus optional companions
nfn migrate muse-glimmer-gguf-to-native \
  --gguf /models/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
  --gguf /models/Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf \
  --dflash /models/dflash-Muse-Glimmer-30B-Q4_K_M.gguf \
  --mmproj /models/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf \
  --tokenizer-source /models/Muse-Glimmer-30B \
  --output-dir artifacts/glimmer-kquant

# Authenticate and atomically attach a native LoRA/QLoRA checkpoint
nfn migrate muse-glimmer-lora-to-native \
  --artifact artifacts/glimmer-bf16 \
  --checkpoint runs/glimmer-adapter/checkpoint-step-100
```

Migration never downloads an unrequested variant during model startup and
never substitutes retained lowercase legacy GGUF files for the canonical
profiles.

## 5. Native kernel inventory

### 5.1 Implemented raw C ABI

The Glimmer resident/training code dynamically resolves these versioned
symbols from [`tile_ops.h`](../neuralfn/csrc/native_train/tile_ops.h):

| Operation | Implemented symbol | Contract |
| --- | --- | --- |
| Typed packed weights | `nfn_native_tile_packed_weight_validate_v1`, `nfn_native_tile_packed_weight_dequantize_float32_v1` | F32, BF16, Q4_K, Q5_K, Q6_K, and native NF4 group-64 descriptors; byte extent/shape/row-stride validation. |
| Packed projection | `nfn_native_tile_linear_packed_weight_float32_v1` | Arbitrary rectangular target/assistant/vision matrices without whole-weight dequantization. |
| Packed Q8 activation projection | `nfn_native_tile_quantize_q8_1_float32_v1`, `nfn_native_tile_linear_packed_weight_q8_1_float32_v1`, `nfn_native_tile_linear_packed_weight_q8_1_multi_decode_float32_v1` | Signed Q8 activation blocks plus Q4_K/Q5_K/Q6_K dot products. The one-token runtime quantizes each shared normalized input once; the multi-decode ABI launches Q/K/V/gate or MLP gate/up together. Exact-zero inference retains FP32 activations. |
| Greedy row selection | `nfn_native_tile_argmax_rows_float32_v1` | Deterministic row-wise argmax through the 202,048-way vocabulary; equal maxima choose the lowest token ID. Greedy DFlash transfers only IDs/selected values, while sampled lossless decoding retains full p/q distributions. |
| Packed backward input | `nfn_native_tile_linear_backward_input_packed_weight_float32_v1` | Frozen packed-base `dX` used by native QLoRA; no packed `dWeight`. |
| Embedding | `nfn_native_tile_glimmer_embedding_gather_float32_v1`, `nfn_native_tile_glimmer_embedding_gather_device_i64_float32_v1`, `nfn_native_tile_glimmer_embedding_batch_i32_float32_v1` | Vocabulary 202,048, device-indirect greedy graph input, and signed 32-bit training IDs. |
| Wide RMSNorm | `nfn_native_tile_glimmer_rms_norm_affine_float32_v1`, `nfn_native_tile_glimmer_rms_norm_affine_capture_residual_float32_v1`, `nfn_native_tile_glimmer_rms_norm_affine_capture_residual_q8_1_float32_v1`, `nfn_native_tile_glimmer_rms_norm_affine_add_residual_float32_v1`, `nfn_native_tile_glimmer_dual_rms_add_capture_cooperative_batch_float32_v1`, `nfn_native_tile_glimmer_rms_norm_backward_float32_v1` | Width 6,656, weightless/ordinary/centered modes, eps 1e-5/1e-8, FP32 reduction. Optional fused forms capture the residual, emit bit-identical Q8 metadata, add the saved residual with exact FP32 order, or partition each multirow verifier norm across eight cooperative blocks. The cooperative form preserves the baseline 256-lane reduction order, accepts F32 affine descriptors, and falls back for BF16 norms. |
| Positioned RoPE and Q/K prep | `nfn_native_tile_glimmer_positioned_rope_float32_v1`, `nfn_native_tile_glimmer_positioned_rope_batch_float32_v1`, `nfn_native_tile_glimmer_qk_norm_scale_rope_float32_v1` | Explicit absolute positions, Q32/KV2 or KV8, half-split/interleaved layout. One-token target decode fuses per-head Q/K RMS, query factor 3.87, and optional local RoPE; global layers select NoPE. |
| Decode GQA | `nfn_native_tile_glimmer_gqa_decode_float32_v1`, `nfn_native_tile_glimmer_fused_decode_attention_float32_v1`, `nfn_native_tile_glimmer_fused_decode_attention_device_position_float32_v1` | Local ring or global cache, Q32/KV2, head 128, no fixed 1,024-key truncation. Fused forms preserve exact Q/K normalization, optional RoPE, attention, and cache order; the device-position form is graph-replay safe. |
| Training attention | `nfn_native_tile_glimmer_attention_forward_float32_v1`, `..._backward_float32_v1` | Local causal window or global causal NoPE, online softmax and backward. |
| Cache commit | `nfn_native_tile_glimmer_cache_commit_bf16_v1`, `nfn_native_tile_glimmer_cache_commit_rows_bf16_v1`, `nfn_native_tile_glimmer_cache_commit_layers_bf16_v1` | Single-row, transactional multirow, or all-layer BF16 hybrid local/global cache append with absolute positions and ring wraparound. |
| Attention gate | `nfn_native_tile_glimmer_sigmoid_gate_float32_v1`, `..._backward_float32_v1` | `attention * sigmoid(gate)` before O projection. |
| Logit transform | `nfn_native_tile_glimmer_logit_transform_float32_v1`, `..._backward_float32_v1` | Multiplier-before-softcap and exact derivative. |
| DFlash attention | `nfn_native_tile_dflash_block_attention_float32_v1`, `nfn_native_tile_dflash_block_attention_short_split_float32_v1` | Accepted prefix plus bidirectional 16-row block, KV8, window 2,048. The optional split-score/value specialization is fail-closed to at most 128 keys and uses caller-owned scratch; the general kernel remains the fallback. |
| Vision preparation | `nfn_native_tile_glimmer_vision_prepare_float32_v1` | Learned 32×32 bilinear position interpolation plus authenticated window permutation. |
| Vision LayerNorm | `nfn_native_tile_glimmer_vision_layer_norm_float32_v1` | Affine FP32 LayerNorm through width 8,192; production vision width is 1,536. |
| Vision attention | `nfn_native_tile_glimmer_vision_attention_float32_v1` | Noncausal segmented full/window attention with half-split or interleaved 2-D RoPE. |
| Vision merge | `nfn_native_tile_glimmer_vision_pixel_shuffle_float32_v1` | Indexed inverse permutation and exact dimension-major 2×2 pixel shuffle. |
| Masked loss | `nfn_native_tile_glimmer_masked_cross_entropy_i32_float32_v1` | Signed int32 targets/mask, 202,048 vocabulary, Glimmer transformed logits. |
| Sequence log-probability | `nfn_native_tile_sequence_logp_i32_float32_{forward,backward}_v1` | Masked chosen/rejected sequence log-probabilities for a 202,048-way head. |
| DPO loss | `nfn_native_tile_dpo_pairwise_loss_float32_{forward,backward}_v1` | Sigmoid, hinge, or IPO policy gradients against an immutable reference. |
| Reward model | `nfn_native_tile_masked_reward_head_float32_{forward,backward}_v1`, `nfn_native_tile_preference_bce_loss_float32_{forward,backward}_v1` | Last-selected-token scalar head and Bradley-Terry pair loss. |
| PPO | `nfn_native_tile_token_logp_entropy_i32_float32_{forward,backward}_v1`, `nfn_native_tile_masked_ppo_loss_float32_{forward,backward}_v1` | Per-token policy log-probability/entropy and masked clipped policy/value objective. GAE and rollout state are trainer orchestration. |
| Optimizer | `nfn_native_tile_glimmer_adamw_bf16_float32_v1` | BF16 parameters, FP32 gradients/moments, clipping and exact resume state. |

Existing generic linears, add/scale, SwiGLU, gradient accumulation, and copying
are reused where their contracts are exact. The legacy generic SDPA and fused
causal-attention functions are explicitly not used: their fixed 1,024-key
geometry and equal-width QKV assumptions cannot represent Glimmer.

`NFN_NATIVE_TILE_PACKED_WEIGHT_NF4_GROUP64` is a native training-only encoding:
each group stores one little-endian FP32 absmax followed by 32 low-nibble-first
NF4 code bytes. It is not GGUF K-quant and is never assigned a GGML type ID.

### 5.2 Completed orchestration and remaining performance work

The formerly missing correctness paths now exist:

1. **Whole-model CUDA vision** uploads the packed/BF16 projection weights and
   one-time F32 norm/bias/position data, then runs patch projection, position
   preparation, LayerNorm, 50 full/window transformer layers, 2×2 merge,
   adaptor, projection, and scaleless RMS on the selected CUDA stream. Only
   final decoder-width rows cross D2H. Missing vision symbols fail during
   companion attachment before sessions are created.
2. **Native DPO/reward/PPO** use the raw ABI rows above with structured records,
   immutable authenticated reference/reward checkpoints, shared policy state,
   real rollout/GAE/minibatch state, and objective-specific strict checkpoints.
3. **K-Quant LoRA SFT/DPO** streams each authenticated GGUF tensor through its
   native type descriptor, computes packed base forward/backward-input without
   whole-model dequantization, and persists adapter-only state bound to the
   exact main profile and tensor table.
4. **DFlash distillation** is a separate Torch training workflow because it
   consumes an HF-style frozen target with hidden-state taps. Its recipe and
   lineage are serialized; production export emits only the canonical 58
   assistant tensors and never copies target embeddings or the shared LM head.
5. **Distributed 30B AR/SFT** uses contiguous pipeline stages, dynamically
   loaded NCCL send/receive plus global reductions, stage-local memory
   admission, and authenticated per-rank model/optimizer/state shards.

The current packed-inference optimization replaces the former serial output
dot product with encoding-specialized warp GEMV for Q4_K, Q5_K, and Q6_K.
One-token decode retains two reduction warps per output group and uses four
independent channel groups for ordinary Q4_K/Q6_K projections, one for the
Q5_K vocabulary head, and one for the hot paired Q4_K FFN gate/up shape after
full-model A/Bs showed four regressed that larger grid. Multirow target
verification keeps eight proposal rows in registers per block and decodes
each packed weight block once for reuse across those rows. One row group per
CTA is the accepted occupancy/register balance; 16 rows and two row groups
lost the matched DFlash test.
Target prefill now uses causal 16-token verification blocks, batched
embedding/RoPE, one batched cache commit, batched assistant-tap capture,
persistent verification scratch, and one staged D2H transfer for the five taps
instead of per-row/per-layer synchronization. DFlash anchor/mask embedding,
accepted-context injection, RoPE, cache commit, and target verification use
the same batched seams.

One-token target decode additionally groups Q/K/V/attention-gate projections
into one multi-decode launch and groups MLP gate/up into another. The packed
dot products retain their independently authenticated Q4_K/Q5_K/Q6_K tensor
descriptors; the grouping is a launch contract, not a new on-disk encoding.
Per-head Q/K RMS, query scaling, and local positioned RoPE are also one
optional launch, while the same entry point skips RoPE on each fourth/global
layer. Older ABI-v1 sidecars fall back to the exact standalone composition.

Greedy DFlash no longer copies both 15×202,048 assistant and target logit
matrices to the host merely to find their maxima. The shared CUDA target head
runs the deterministic device argmax for assistant proposals, current target
state, and all verification/bonus rows, returning at most 16 IDs and selected
values. The sampled path deliberately keeps full logits because exact p/q
acceptance and rejection-residual sampling require the complete vocabulary.

Ordinary greedy target-only decoding now uses that same device argmax for its
current vocabulary row. The one-token target path also writes each layer's K/V
row directly into the hybrid cache after attention instead of staging and then
issuing 104 device-to-device copies per token. Transactional visibility is
still controlled by the cache's logical length, which advances only after the
entire token succeeds. A local-ring overwrite is exactly one 2,048-token
window behind and therefore cannot be observed by a retry; a global layer
writes a fresh slot. Multi-token target verification deliberately retains its
staging buffers because an accepted prefix may be shorter than the tentative
16-row block.

Two optional wide-norm entry points fuse residual capture and residual add
with the existing 6,656-wide affine RMSNorm. The resident requires the pair as
a unit and otherwise uses the older exact composition. On the RTX 5090,
focused CUDA-event measurements on the post-revert final sidecar reduced
one-row norm-plus-capture from 0.0098 to 0.0057 ms (1.713×) and 16-row
norm-plus-capture from 0.0099 to 0.0077 ms (1.282×), with zero output error.
The fused residual-add form is
bit-identical to the separate FP32 add and was throughput-neutral in this
microbenchmark.

The strict one-row path also has a paired, optional handoff contract:
`nfn_native_tile_glimmer_dual_rms_add_capture_mmvq_q8_float32_v1` computes the
same two adjacent norms/residual outputs and writes exact `block_q8_1` bytes
directly from each warp; then
`nfn_native_tile_k_quant_mmvq_multi_linear_prequantized_float32_v1` consumes
that workspace without launching activation quantization. The resident loads
the symbols only as a complete pair and uses them only for rows=1. The raw
probe compares every ordinary and handoff output bit-for-bit and checks short
workspace/invalid-row rejection. A first implementation that reread the
normalized row after a grid barrier was slower and was removed.

Remaining CUDA work is performance and hardware qualification, not a silent
fallback path: BF16 activation/fused matmul variants, optimized vision
attention/LayerNorm, tensor/data parallelism, the 80-GB full-size run, and
full-size layer/logit/chat/DFlash upstream-oracle parity beyond the completed
16-token raw target check. Live NVCC, four-tool sanitizer coverage, and the
32-GB Dynamic full-size benchmark are complete. The strict path stores
activations and accumulates in FP32 while retaining BF16 or packed weights;
the separately counted Q8 seam changes only the packed input representation.

The generic optional Q8 activation seam remains restricted to the separately
gated positive-temperature projection path. The new strict dual-RMS handoff is
different: its Q8 representation is byte-identical to the existing exact MMVQ
quantizer and is accepted for exact-zero decode only because the raw ABI probe
and full-artifact token hash match the ordinary composition. Generated-token
hashes are still narrower evidence than full-logit parity.

Standalone sigmoid or logit-transform fusion is no longer a correctness gap;
the versioned functions above exist.

### 5.3 Still-missing CUDA performance kernels

No kernel below is required for correctness of the advertised resident text,
DFlash, post-training, or still-image paths; those paths fail closed and have
no CPU model-compute fallback. They are the concrete missing pieces for the
remaining throughput and scale goals:

The accepted launch-amortized target baseline is no longer a sequence of
ordinary one-projection launches. It groups one-row Q/K/V/attention-gate and
MLP gate/up packed projections so each group shares one activation
quantization, fuses the sigmoid-gate or SwiGLU handoff into the packed output
projection, and captures LM-head → device argmax → device-token embedding → 52
layers → cache/final hidden as one repeated greedy-token CUDA graph. The graph
uses versioned device-token and device-position entry points; host arguments
are not patched between replays. Exact multirow verification does not use the
one-row projection/handoff wrappers, but its MMVQ kernel now hoists each
decoded packed weight block across eight input rows. The exact runtime-off and
compile-time-hoist-off A/B evidence is in section 11.

Nsight Systems on the final exact DFlash route identifies packed verification
MMVQ, not launch overhead, as the remaining bottleneck: Q4_K 19,968-output
rows consumed about 55.9 ms, Q6_K 6,656-output rows 27.1 ms, Q4_K 6,656-output
rows 22.1 ms, and Q4_K 4,096-output rows 11.4 ms in the profiled turn. Exact
batched MMVQ accounted for roughly 134 ms, versus about 14.4 ms of norms, 9.4
ms of attention, and 8.2 ms of the Q5_K head. A new fusion is accepted only if
it preserves the canonical token hash and wins a matched full-model on/off
test; fewer launches alone is insufficient.

1. **BF16 activation packed GEMM/GEMV** — a proposed
   `nfn_native_tile_linear_packed_weight_bf16_activation_v1` family for
   Q4_K/Q5_K/Q6_K and BF16 weights, including backward-input. The current
   resident keeps FP32 activations and converts/accumulates around packed
   weights, which costs bandwidth and prevents tensor-core BF16 tiling.
2. **Arithmetic-exact persistent packed MMVQ** — the accepted target
   megakernels remove launches and share activation quantization, while the
   verifier reuses a decoded packed weight block across eight Q8_1 rows.
   Output groups still stream independent packed rows. A next-generation
   kernel must reuse larger decoded weight tiles across output work without
   changing the pinned accumulation/rounding order. It
   must cover 6,656→4,096/256/19,968, 19,968→6,656, and
   6,656→202,048 tail shapes for verifier row counts 2..16. This is the primary
   next performance margin and long-context efficiency rather than a blocker
   to the now-completed pinned llama.cpp comparison. Another grouping wrapper
   is not sufficient. Shared whole-weight staging, two-output/two-warp, and upstream
   four-warp candidates all lost the full-model A/B or changed output and were
   removed.
3. **Chunked packed vocabulary selection** — target/head kernels that stream
   the 202,048-row packed LM head directly into deterministic argmax/top-k and
   sampled p/q reductions without materializing the full FP32 logits matrix.
   Exact sampled DFlash additionally needs on-device `min(1,p/q)` acceptance
   and `max(p-q,0)` residual sampling over the complete vocabulary.
4. **Flash-style hybrid GQA** — BF16/paged online-attention prefill and decode
   specialized for Q32/KV2/H128, local ring 2,048, and global context 131,072,
   plus Q32/KV8 DFlash block attention. The current kernels are correct at
   those lengths but use FP32 scalar/cache traversal and leave substantial
   long-context throughput on the table.
5. **Fused DFlash proposal/verification** — retain the five target taps,
   assistant block state, shared-head reductions, acceptance prefix, and
   transactional cache metadata on device across a 16-slot step. Today those
   operations are CUDA-resident but remain multiple launches with host-visible
   step orchestration.
6. **Tensor-parallel inference collectives** — sharded packed Q/K/V/O, FFN,
   embedding, and vocabulary-head kernels plus NCCL all-reduce/all-gather for
   BF16 and Dynamic deployment beyond a single device. Pipeline-parallel
   training exists; tensor-parallel resident inference does not.
7. **Optimized vision kernels** — tensor-core patch/adaptor projections and
   varlen BF16 full/window attention for the 1,536-wide vision tower. The
   current vision ABI is correct and CUDA-only at request time, but is not a
   throughput-optimized multimodal stack.

An attempted fusion of target GQA, BF16 cache commit, and sigmoid gating was
bit-exact and reduced launch count, but repeated target throughput was flat or
worse; it was rejected and is not present in the accepted source. A later
two-kernel attention split passed an initial smoke but changed greedy output on
repeat and was fully reverted. A non-streaming `decode_many` binding measured
74.72 tok/s, below the accepted path, and was also reverted. Full-token CUDA
graph replay and the grouped projection megakernels were retained only after
repeated exact-output A/Bs showed a gain. Set `NFN_GLIMMER_CUDA_GRAPHS=0` or
`NFN_GLIMMER_MMQ_MEGAKERNELS=0` only for development bisection.

For DFlash, the accepted policy uses exact five-row-tail MMVQ where applicable,
reuses Q8 workspaces across K/V/gate and MLP-up projections, overlaps
independent projections on auxiliary streams, writes K/V directly into
transactional staging, and selects short-context split attention only below
its proved 128-key bound. The cooperative dual-RMS kernel partitions each row
over eight blocks while reproducing the ordinary FP32 bits. Its final
same-binary control measured 264.724 tok/s off versus 271.244 on (+2.46%), with
the same canonical hash and 28/34 acceptance. Packed-weight decode hoisting is
also retained from an earlier compile-time A/B at 158.779 tok/s off versus
235.657 on. Wider Q5/Q6 row groups, FFN-specific row tiling, all-row attention
splitting, concurrent split quantization, shared whole-weight staging, and
extra output warps either regressed end-to-end throughput or changed the exact
path and remain absent.

## 6. Resident target and DFlash execution

- [x] Added strict resident dispatch for BF16 and both official K profiles.
- [x] Added normalized/raw embeddings, exact local/global schedule, hybrid KV
  storage, target hidden taps, multi-token verification, and transactional
  cache append/commit/truncate/reset.
- [x] Added device-resident packed/BF16 target linears, GQA, norms, RoPE,
  gating, MLP, head, softcap, and sampling orchestration.
- [x] Added strict BF16 and packed DFlash loaders, raw target embedding reuse,
  five-tap projection, five assistant blocks, shared target head, proposal,
  greedy prefix acceptance, lossless sampled acceptance, and synchronized
  cache rollback.
- [x] Added `off|auto|required` speculation. `auto` may fall back only before
  session mutation; `required` fails closed. Every committed token is streamed
  individually and EOS/max-token clipping occurs inside the atomic step.
- [x] Added proposed/accepted/rejected and target/assistant step counters.
- [x] Disabled stock DFlash after a native adapter is attached unless an exact
  compatible assistant lineage is supplied.
- [x] Added 16-token causal CUDA prefill, persistent verification workspaces,
  batched raw embedding/positioned RoPE/cache commit, and batched target-tap
  capture/context injection for the paired runtime.
- [x] Added device-indirect token embedding and device-position fused attention
  so a complete repeated greedy target token can execute as one CUDA graph
  replay without host node patching. Grouped Q/K/V/gate and MLP gate/up
  megakernels share activation quantization and fuse gate/SwiGLU handoff.
- [x] Restored the installed colorful multi-turn TUI with live runtime/cache/
  DFlash statistics, per-turn prefill/decode/acceptance metrics, transcript
  controls, and ATEM channel-safe output. Private `to=self` reasoning is never
  displayed or reinserted; only `to=user` text enters history.

Example:

```bash
nfn infer \
  --checkpoint artifacts/glimmer-kquant \
  --runtime native-cuda \
  --weight-precision auto \
  --speculative-decoding auto \
  --companion-checkpoint dflash \
  --tile-ops-lib /absolute/path/libnfn_native_train_tile_ops.so \
  --prompt "Explain speculative decoding briefly." \
  --native-info
```

For the multi-turn TUI, omit `--prompt`, add `--chat-mode transcript` and an
optional `--system-prompt`, then use `/show`, `/reset`, `/clear`, `/mode`, and
`/exit` inside the resident process. An explicit
`--weight-precision k-quant-17gb` remains valid on a 32-GB device; only `auto`
prefers Dynamic when the complete measured budget fits.
`--strict-tile-ops-lib PATH` is accepted as an alias of `--tile-ops-lib PATH`
for the whole-model CUDA sidecar.

## 7. Automatic precision selection

The public selector is independent from activation dtype and KV-cache format:

```text
--weight-precision {auto,bf16,k-quant-dynamic,k-quant-17gb}
```

Python uses `NativeModelLoadConfig(weight_precision="auto", ...)`. On CUDA,
selection occurs before a resident handle is created:

1. validate every descriptor and the primary/checkpoint invariant;
2. select the CUDA device and query current free/total bytes;
3. budget resident target and enabled companion weights, load staging,
   workspace, hybrid target/assistant KV, tentative verification state,
   configured context/session count, and reserve;
4. walk `bf16`, Dynamic, then 17GB in fidelity order among available,
   kernel-supported candidates;
5. authenticate only the selected main plus every enabled companion;
6. pass an effective manifest copy to the optional load-with-options ABI;
7. report the exact decision and rejection reasons.

The default reserve is at least 2 GiB and 10% of total VRAM. Exact manifest
memory profiles, not filenames or file size, decide fit. Explicit precision is
a strict pin and never downgrades. `auto` never retries a lower profile after a
load-time OOM, never silently shrinks context, and never disables required
DFlash/vision. Optional DFlash may be omitted to retain a higher-fidelity target;
required DFlash participates in the initial budget.

CPU `auto` uses the authenticated primary profile and does not pretend that a
CUDA VRAM query is a host-RAM policy.

## 8. Post-training support

| Objective | Torch graph/trainer | Native C++/CUDA | Artifact/deployment |
| --- | --- | --- | --- |
| Full masked SFT | Yes | Yes | Full BF16 checkpoint + optimizer/RNG/data cursor |
| LoRA SFT | Yes, eight roles/layer | Yes | Strict adapter v1; CPU/CUDA direct execution and deterministic merge utility |
| NF4 QLoRA SFT | Yes | Yes | Immutable native NF4 base during training; adapter records QLoRA provenance and deploys against its exact BF16 lineage |
| K-Quant adapter SFT | N/A for generic Torch | Yes, LoRA | Immutable official GGUF base; adapter pins profile, target digest and authenticated tensor table |
| DPO | Yes | Yes | Frozen reference digest; native also supports LoRA over the identical K-Quant base |
| Reward model | Yes | Yes | Sequence-mask-aware last-token scalar head, pairwise BCE and strict reward artifact |
| PPO | Yes | Yes | Online policy rollout, frozen reference/reward, value head, per-token log-probabilities, KL, GAE, clipped minibatch epochs and resume |
| DFlash distillation | Yes, dedicated trainer | Not a C++ loop | Frozen target/taps/shared head, D-PACE or decay, self-logit KD, exact resume/acceptance audit, native assistant export |
| Multimodal tuning | Vision graph exists | No native loop | TODO |

Native SFT record batches are not flat next-token bins: they include input IDs,
targets, loss masks, example boundaries, tokenizer/chat hashes, and objective
metadata. Empty masks, mismatched boundaries, wrong source model digests, and
cross-mode resume fail before an optimizer step.

LoRA targets are exactly:

```text
q_proj,k_proj,v_proj,o_proj,attn_gate_proj,gate_proj,up_proj,down_proj
```

QLoRA is distinct from K-Quant. It streams the pinned BF16 source into bounded
row chunks, constructs immutable NF4 group-64 device buffers, computes packed
base forward and `dX`, and stores no base `dW` or AdamW state. Only A/B matrices
are saved. Resume reconstructs the exact NF4 base from the pinned BF16 source
and refuses LoRA/QLoRA cross-mode state.

K-Quant adapter tuning instead uses the official Q4_K/Q5_K/Q6_K GGUF tensor
table directly. It never relabels that base as NF4, never mutates or saves
packed base codes, and does not claim a lossless merged K artifact. Forced
profile, digest, objective, reference lineage and adapter state must match on
resume. The supported native objectives are SFT and DPO; packed-base reward or
PPO stays rejected until those separate artifact contracts are defined.

`DFlashDistillationTrainer` is deliberately target-bound and assistant-only.
It freezes the main model, samples valid random anchors, gathers zero-based
target layers `[1,13,25,37,49]`, reuses raw target embeddings and the frozen
target head, supports `dpace` and `decay` weighting, saves optimizer/RNG/data
cursor/recipe lineage, and can run a greedy online acceptance audit. Its native
export is available only for production geometry and the pinned tokenizer,
config and ATEM hashes. The metadata explicitly says that NeuralFn does not
claim reproduction of Meta's unpublished released-assistant recipe.

Full-BF16 AR and SFT can also use `--pipeline-parallel-size N`. The CLI starts
one process per distinct `--pipeline-cuda-devices` entry; rank 0 owns the token
embedding, the last rank owns final norm/head, and middle ranks own contiguous
decoder ranges. NCCL transfers activations/gradients and reduces global loss
and gradient norm. Each rank independently checks its free bytes plus reserve.
Checkpoints contain authenticated rank-local BF16 parameters, F32 moments,
binary cursor/RNG state, a distributed manifest, and a final `DONE` marker.
Changing world size or stage ownership on resume fails closed.

## 9. Vision and media boundary

Implemented:

- exact aspect-ratio LANCZOS image preprocessing, RGB 0.5/0.5 normalization,
  temporal-2/channel-major patch packing, learned position interpolation,
  window/full attention schedule, 2-D RoPE, 2×2 merge, projector, and
  decoder-width normalization;
- full-BF16 embedded vision on the portable C++ CPU runner;
- packed official mmproj CPU execution for still images;
- full-BF16 and packed-mmproj whole-model CUDA vision with device-resident
  linears, wide affine LayerNorm, 2-D RoPE segmented attention, exact merge and
  adaptor/projector execution; request-time failures never fall back to CPU;
- exact decoded-video frame sampling (default 96 frames at 2 FPS), even-frame
  padding, per-temporal-pair timestamps, 144-frame-token admission cap, ATEM
  video prompt fragments, and replacement positions;
- OpenAI-compatible Chat Completions image data URLs and direct Python
  `encode_images()` / `encode_videos()` APIs.

The official packed mmproj has a 588-wide patch projection formed by collapsing
the temporal-2 BF16 weights. It is exact for duplicated still-image frames but
cannot distinguish two video frames, so its manifest says `video=false`.
External URL fetching and video-container decoding remain outside the resident
process; Python video callers supply decoded frame sequences.

Remaining:

- [ ] a bounded, versioned server video-content extension if container input is
  desired; it must not be confused with OpenAI's image part contract;
- [ ] native multimodal training and DFlash on multimodal prefixes.

## 10. Verification completed in this implementation

- [x] Immutable main/assistant/vision fixture with revision and source hashes.
- [x] Tiny deterministic local/global decoder forward/backward and final-logit
  parity, including norm/gate/QK/RoPE/NoPE ordering.
- [x] Assistant context/noise/block/logit/acceptance and cache rollback oracles.
- [x] Strict safetensors and GGUF parser negative matrices.
- [x] Q4_K/Q5_K/Q6_K block and packed projection parity, including tail rows.
- [x] BF16 and K resident CPU target tests; fake-CUDA target and DFlash tests
  assert the device orchestration and prohibit CPU model fallback.
- [x] VRAM selector boundary, explicit pin, missing/corrupt profile,
  companion-budget, and CPU-auto tests.
- [x] uint32 shard, structured SFT, native full update, LoRA, QLoRA, strict
  save/resume, frozen-base, and adapter attach/load tests.
- [x] Native DPO/reference, reward-head/BCE, online PPO/GAE/minibatch and strict
  objective-specific checkpoint/resume tests.
- [x] K-Quant LoRA SFT/DPO frozen-byte, packed-forward/`dX`, profile lineage,
  adapter save/resume and mismatch tests.
- [x] Torch SFT/LoRA/QLoRA/DPO/reward/PPO formula and gradient tests plus
  target-frozen DFlash distillation, bit-exact resume and acceptance tests.
- [x] Two-rank fake-NCCL pipeline parity with the single-rank update, rank-shard
  reconstruction, resume and production 8-stage 80-GB-class planner tests.
- [x] Native image/video preprocessing, vision CPU oracle, media fusion, and
  fake-CUDA vision parity with nonzero device workspace/kernel counters.
- [x] Mandatory all-preset build/resolve/compile/forward/apply and ordered
  cross-preset tests.
- [x] Native inference binding and exact Glimmer trainer compile with a host C++
  toolchain.

Still required before performance or full-hardware release claims:

- [x] build the real normal and strict Tile-CUDA libraries with NVCC and
  validate the exported ABIs. CUDA 13.3.33 compiled one source proof for
  `sm_80`, `sm_89`, `sm_90`, and `sm_120`; training, strict-math,
  packed-weight, inference, and vision all
  reported ABI v1. The pass found and fixed a missing `math_constants.h`
  include. Its JSON is compiler-only (`release_qualified=false`) and records
  the GCC 16 `-allow-unsupported-compiler` caveat;
- [x] run kernel parity on a real device under compute-sanitizer. The current
  runner builds a raw CUDA probe against the same strict Tile sidecar and runs
  it under `memcheck`, `synccheck`, `initcheck`, and `racecheck` before the
  full-artifact benchmark. The RTX 5090 pass covered packed Q4_K linear/`dX`,
  gate, wide RMSNorm, positioned RoPE, 2,048-window GQA, cache commit, DFlash
  block attention, 202,048-way masked CE, DPO/reward/PPO, and the vision
  prepare/LayerNorm/attention/pixel-shuffle kernels with zero errors or race
  hazards;
- [x] rebuild and run the 2026-08-14 current-source direct-device probe after
  the packed/prefill optimization. It passed Q4_K/Q5_K/Q6_K multilinear and
  backward-input, Q8 activation/packed linear, 202,048-way row argmax, wide
  norm plus fused residual capture/add, positioned RoPE,
  windowed GQA, single and multirow cache commit, DFlash attention,
  202,048-way losses, post-training objectives, and vision on device 0;
- [x] rerun `memcheck`, `synccheck`, `initcheck`, and `racecheck` against the
  exact accepted sidecar. The first three reported zero errors and racecheck
  reported zero hazards/errors while executing the full direct-device list,
  including multi-projection Q8, fused Q/K norm-scale-RoPE, wide norm/Q8
  capture, full-vocabulary argmax, device-indirect embedding, device-position
  fused attention/cache commit, DFlash, post-training, and vision. The accepted
  strict binary has SHA-256
  `d15e946e16b1ef5b643a4556f8f719e49efa1466ad12c4957dbcaaa73953e994`;
- [x] run bounded authenticated target-only and DFlash ATEM chat benchmarks
  against that exact sidecar and binding. Target-only produced the same
  canonical 32-token hash in ten trials at 78.608 tok/s median. The final
  current-source DFlash path produced that same canonical hash in ten trials
  with 28/34 accepted proposals and 37 target rows at 271.244 tok/s median
  (271.792 mean). A same-final-binary run with only cooperative batched RMS
  disabled measured 264.724 tok/s. Earlier retained controls measured 72.871
  target tok/s with its runtime megakernels off, and 158.779 DFlash tok/s with
  packed-weight hoisting off versus 235.657 enabled. These are exact-workload
  measurements, not a general quality claim or a matched reproduction of
  Meta's undisclosed benchmark;
- [ ] finish one current-source canonical BF16, Dynamic, 17GB, and DFlash
  matrix across the representative 24/32/80-GB profile tiers. The standalone
  32→Dynamic run is complete with the default VRAM-driven `auto` selector,
  DFlash, CUDA mmproj, 128/2,048/8,192 contexts, three trials each, and zero
  CPU model-compute rows. The standalone 24→17GB profile is also complete on
  the available larger GPU with an explicit precision pin, one
  128/8,192-token capacity trial, and measured peak usage below 24 billion
  bytes. Their source proofs differ because the latter includes the corrected
  tier policy, so Dynamic must be rerun with the future 80→BF16 result before
  the matrix verifier can combine them. BF16 still needs at least 80 billion
  physical bytes; the qualifier records actual capacity and will not
  synthesize it;
- [x] compare one bounded full-size target-only greedy token path against the
  pinned llama.cpp oracle. Build 10349 at commit `62bf73d` and NeuralFn return
  the same 16 IDs for the authenticated Dynamic artifact and BOS+`Hello` raw
  prefix, with zero NeuralFn CPU model-compute rows;
- [ ] compare full-size layer outputs/logits, ATEM-rendered chat, sampled
  generation, and DFlash proposal/acceptance against the pinned Transformers
  and llama.cpp `--jinja` oracles;
- [ ] publish the complete three-class p50/p95 TTFT, tokens/s, DFlash
  acceptance, resident bytes, load peak, and context/session matrix. The
  measured packed-profile rows are published below; 80-GB remains open;
- [x] qualify the current exact CUDA paths for bounded production-performance
  claims. The exact local request, hashes, distributions, memory, CPU counters,
  and megakernel controls are published below. Software capability bits remain
  ABI/artifact gated and are not general benchmark claims.

Focused CUDA-event measurements on the RTX 5090 compare the final current
sidecar with the previously qualified v5 sidecar. They use deterministic
synthetic packed tensors at real Glimmer matrix geometry and verify output
error before timing:

| Projection geometry | Rows | Final time | Speedup vs v5 |
| --- | ---: | ---: | ---: |
| Q4_K 6,656→4,096 | 1 | 0.018 ms | 2.260× |
| Q4_K 6,656→19,968 | 1 | 0.055 ms | 3.174× |
| Q6_K 19,968→6,656 | 1 | 0.083 ms | 3.183× |
| Q5_K 6,656→202,048 | 1 | 0.626 ms | 3.215× |
| Q4_K 6,656→256 | 16 | 0.032 ms | 1.258× |
| Q6_K 6,656→256 | 16 | 0.038 ms | 1.156× |
| Q4_K 6,656→4,096 | 16 | 0.106 ms | 5.273× |
| Q4_K 6,656→19,968 | 16 | 0.376 ms | 7.961× |
| Q6_K 19,968→6,656 | 16 | 0.454 ms | 9.001× |
| Q5_K 6,656→202,048 | 16 | 4.198 ms | 8.917× |

The largest observed accumulation-order difference in these rows was
0.00415. This is focused kernel evidence only. It does not establish target
tok/s, DFlash tok/s, acceptance, ATEM quality, or a win over Meta/llama.cpp/
ExecuTorch; those require the fresh-process full-model run above.

For one-row K/V projections, a later geometry guard routes 256-output packed
matrices back to the full-block/output kernel because one warp per output
under-fills the RTX 5090. Against the immediately preceding warp-only build,
6,656→256 measured 0.008 ms for Q4_K (1.504×), 0.008 ms for Q5_K
(1.502×), and 0.010 ms for Q6_K (1.370×), with the wide and 16-row
paths unchanged. `tools/bench_muse_glimmer_residual_norm.py` separately
reproduces the fused-norm measurements above; neither microbenchmark is an
end-to-end throughput claim.

The fresh-process current-source chat gate used the canonical K-Quant-17GB
artifact on the 33,708,376,064-byte RTX 5090. Every row used strict computation
(`temperature=0`) and the same 41-token rendered ATEM prompt:

| Mode | Generated tokens / trials | Decode throughput | Accepted / proposed per trial | Peak sampled CUDA delta | Aggregate launches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target, accepted megakernels | 32 / 10 | 78.608 tok/s median (78.515–78.809) | n/a | 18,949,865,472 B | 17,963 |
| Target, same binary with runtime megakernels off | 32 / 10 | 72.871 tok/s median (72.787–73.068) | n/a | 18,954,059,776 B | 31,691 |
| DFlash, final current-source path | 32 / 10 | **271.244 tok/s median**, 271.792 mean (269.303–274.820) | 28 / 34 | 20,654,850,048 B | 38,269 target + 3,762 assistant |
| DFlash, same final binary with cooperative batched RMS off | 32 / 10 | 264.724 tok/s median (262.292–269.188) | 28 / 34 | 20,654,850,048 B | 38,269 target + 3,762 assistant |
| DFlash, earlier packed-weight hoist on | 32 / 10 | 235.657 tok/s median (235.455–239.393) | 28 / 34 | 20,652,752,896 B | 38,269 target + 3,762 assistant |
| DFlash, earlier otherwise-identical packed-weight hoist off | 32 / 10 | 158.779 tok/s median (143.194–160.181) | 28 / 34 | 20,652,752,896 B | 38,269 target + 3,762 assistant |

Every row has one warmup in addition to the ten measured trials; cumulative
launch counters include that warmup. All rows produced token hash
`63baebaa0742852d37abf85e81c815430267789bdbb79591eb56a1e1a50b74b1`,
and reported zero CPU model-compute rows. DFlash rows processed 37 target rows
in three assistant blocks. The target comparison retains grouping plus the
direct dual-RMS→Q8→MMVQ handoff by measured evidence: 7.87% more throughput,
43.32% fewer launches, and identical output. The verifier comparison retains
the cooperative dual-RMS megakernel by final same-binary evidence: 2.46% more
throughput with the same launch count, acceptance, output, and sampled peak.
The earlier compile-time comparison retains packed-weight block hoisting by
48.42% with the same acceptance/output path.

The final clean sidecar and matching binding SHA-256 digests are respectively
`d15e946e16b1ef5b643a4556f8f719e49efa1466ad12c4957dbcaaa73953e994` and
`5e3d983fbed735a4dee281ff2c07f165e8e32b47f807a0abe1d3504d58ec870d`.
The final DFlash and same-binary cooperative-RMS-off JSON digests are
`e3af55be90ab2c7271db6d83e61c2d470a6f371a82d98ec2327e1b7f0feb9ceb`
and `3b78d9f065080d4c03bcb7b31bdc768efc16215752dff05d26c77d3aedd5cd03`.
The 40-kernel probe binary digest is
`a68d34fbda89ae05e6d7d95fa2531dd1e3d494a8b8444e5cc42726934a12204b`.

The previously recorded 256.69-tok/s DFlash run returned non-canonical hash
`5868f53efb03b40e2b6c094e13efb981baacf1c23fe622479e3eefcfc3eaa20d`;
it used an approximate verifier arithmetic path and is rejected. Speculative
throughput is also acceptance-sensitive: a second prompt accepted 22/74 and
measured 73.60 tok/s despite the same implementation and 32-token length.

Meta's published RTX 5090 table reports 74.9 target-only and 233.4 DFlash
tok/s. The local exact target and DFlash results are numerically 4.95% and
16.21% higher, but Meta does not publish the exact prompt/token count/acceptance
distribution, so those ratios are not reproducible head-to-head scores. The
direct local oracle used llama.cpp build 10349 at
pinned commit `62bf73d`, the same K-Quant-17GB artifact, 41-token ATEM prompt,
32-token greedy policy, GPU, and DFlash companion. It measured 77.8 target-only
and 225.7 DFlash tok/s. NeuralFn is 1.04% and 20.18% faster in those matched
local results. No comparable pinned ExecuTorch Glimmer artifact/command was
available in the audited upstream contract, so the document records that
comparison as unavailable instead of manufacturing a result.

After the three runs, `tools/qualify_muse_glimmer_gpu.py verify --result ...`
requires all classes, canonical packed hashes, full 52-layer/6,656-wide
geometry, DFlash and vision CUDA counters, positive sampled VRAM, sanitizer
proof, an 8K-or-larger context, and one identical source-tree hash. It cannot
turn fake-CUDA coverage or a partial profile into a hardware release result.

The completed 2026-08-13 24-GB-tier result explicitly selected
K-Quant-17GB on the physical RTX 5090 instead of rejecting the larger device or
letting production `auto` upgrade it to Dynamic. The result records
`profile_tier.minimum_total_vram_bytes=24000000000`,
`profile_tier.physical_total_vram_bytes=33708376064`, and
`profile_tier.larger_device=true`. Source proof
`26d412085b7941a2a7d55c00b49b6fdc885c0ab2d5709083c101a367e0105469`
and result JSON SHA-256
`cc70412c7299a3cc84d2247a9d81466104e47b5d411c571b236ce5b4f8c8c30a`
bind the run to canonical target SHA
`4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e`.

K-Quant-17GB+DFlash+mmproj loaded in 24.52 seconds. CUDA's fresh-worker
baseline sampler measured a 20,359,217,152-byte (18.961-GiB) peak delta and
9,398,059,008-byte minimum free value; after close, only 4,259,840 bytes of the
sampled delta remained. Target-only, DFlash, and vision CUDA weights occupied
16,743,521,568, 1,618,131,968, and 1,400,328,928 bytes respectively. All four
sanitizer tools passed and `cpu_model_compute_rows=0`.

Single-trial capacity measurements from that exact build:

| Prompt tokens | Prefill | TTFT | 16-token DFlash decode | Acceptance |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 21.76 tok/s | 5.937 s | 2.303 tok/s | 3 / 127 (2.36%) |
| 8,192 | 8.96 tok/s | 914.161 s | 1.490 tok/s | 0 / 135 |

P50 and p95 are the same observation because each row has one repetition. The
packed mmproj probe encoded four 588-wide rows into one 6,656-wide decoder row
in 14.35 ms. Treat these as capacity/correctness measurements, not a stable
performance distribution. The peak is 3,640,782,848 bytes below the declared
24-GB tier.

The completed 2026-08-13 32-GB-class result used an RTX 5090 (`sm_120`), CUDA
runtime/driver 13.3, total memory 33,708,376,064 bytes, and source proof
`cbb3dade82f3939eeee0355ff44a3bd76473fdd524cbbe7295c47a7a85d0b957`.
The result JSON SHA-256 is
`7d2a1bcf22a28f0bc9408329974c5f18de4805c83f193af2c10290e0d2508174`.
`auto` selected the canonical Dynamic artifact SHA
`ac7023d6a4c704eb9af54ab53e476a66b7f5b6c0ef2fc4a8dde5253c291a6c38`;
load took 27.08 seconds, the sampled peak delta was 23,171,432,448 bytes, and
minimum sampled free memory was 6,168,707,072 bytes. Target-only, assistant,
and vision CUDA weights were 19,640,798,496, 1,618,131,968, and 1,400,328,928
bytes respectively; the target's aggregate counter includes vision and reports
21,041,127,424 bytes. `cpu_model_compute_rows=0`.

Three-trial measurements from that exact build:

| Prompt tokens | Prefill p50 / p95 | TTFT p50 / p95 | 16-token DFlash decode p50 / p95 | Acceptance |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 19.67 / 19.74 tok/s | 6.568 / 6.586 s | 2.134 / 2.158 tok/s | 9 / 375 (2.4%) |
| 2,048 | 12.77 / 12.77 tok/s | 160.466 / 160.515 s | 1.618 / 1.619 tok/s | 0 / 405 |
| 8,192 | 8.26 / 8.54 tok/s | 992.434 / 1050.315 s | 1.410 / 1.416 tok/s | 0 / 405 |

All repetitions at a given context produced identical greedy tokens. The
repeated-token prompt is a capacity/correctness fixture, so its DFlash
acceptance is not a representative quality benchmark. The packed mmproj probe
encoded four 588-wide rows into one 6,656-wide decoder row in 15.82 ms p50.
These numbers expose the remaining long-context performance work; they are not
production throughput targets.

The independent target oracle used llama.cpp build 10349 at exact commit
`62bf73d25c53b8161f8a22894d4f90c4aebbd7d0`, the canonical Dynamic artifact
SHA above, greedy target-only decoding, and raw prefix `[200000, 19873]`
(BOS + `Hello`). Both runtimes produced:

```text
[24, 372, 1045, 10016, 328, 2885, 262, 5091, 8811, 511, 917, 4921, 768, 328, 2885, 262]
```

That sequence decodes to `, I am trying to create a simple script that will
allow me to create a`. NeuralFn reported `cpu_model_compute_rows=0`. This is
one exact raw-prompt greedy-token proof. It does not claim general logit,
sampling, ATEM chat, DFlash, or quality parity; those remain separate gates.
