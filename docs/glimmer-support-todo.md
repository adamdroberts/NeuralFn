# Muse Glimmer native support: implementation status and remaining work

> **Status (2026-08-13): core text support is implemented.** NeuralFn now has
> an exact `muse_glimmer` GPT preset, strict BF16 and official GGUF conversion,
> resident C++ CPU and whole-model CUDA text execution, K-Quant-Dynamic and
> K-Quant-17GB, VRAM-aware precision selection, DFlash speculative decoding,
> native pretraining/SFT/LoRA/QLoRA, and exact Torch post-training graphs.
>
> Capability claims remain independent and fail closed. Native DPO/reward/PPO,
> K-Quant adapter tuning, DFlash distillation, distributed 30B training, and
> whole-model CUDA vision are still TODO. Full-BF16 vision runs on CPU; the
> official GGUF `mmproj` supports CPU still-image execution only because its
> temporal patch projection is collapsed.

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
| Native C++/CUDA pretraining | **Implemented, single device** | Exact 627-tensor text layout, uint32 records, activation recomputation, AdamW, strict save/resume, and source SHA binding. A practical full 30B run still needs sufficient device memory or future tensor parallelism. |
| Native full SFT | **Implemented, single device** | Structured uint32 records carry targets, loss masks, boundaries, and exact ATEM lineage. |
| Native LoRA / QLoRA SFT | **Implemented** | All eight projection roles are supported. QLoRA freezes a deterministic NF4 group-64 base and updates only LoRA matrices. |
| Native DPO / reward / PPO | **Not implemented** | The exact Torch paths are production-correct, but there is no raw native preference/rollout trainer yet. |
| Resident target CPU | **Implemented** | BF16 and official mixed F32/Q4_K/Q5_K/Q6_K profiles use hybrid local-ring/global-full KV caches and transactional verification. |
| Resident target CUDA | **Implemented for text** | Target weights and model compute stay on the selected CUDA device. The current ABI uses FP32 activation buffers with BF16 or packed resident weights. |
| K-Quant-Dynamic / K-Quant-17GB | **Implemented** | Strict GGUF v3 parser, authenticated canonical profiles, exact packed CPU/CUDA dequant/GEMM dispatch, no whole-model dequantization. |
| Automatic weight precision | **Implemented** | `auto` is quality-first and byte-budgeted on CUDA; explicit values are strict pins. CPU `auto` chooses the authenticated primary. |
| DFlash speculative decoding | **Implemented** | BF16 and packed assistants, CPU/CUDA assistant execution, target block verification, greedy and lossless sampled acceptance, and atomic cache commit/crop. |
| Native LoRA deployment | **Implemented** | Strict adapter inspection/attachment and direct CPU/CUDA deltas on Q/K/V/O/gate and MLP projections. Adapted targets reject an unbound stock DFlash assistant. |
| Image inference | **Implemented on CPU** | Embedded full-BF16 vision and official K-Quant `mmproj`; Chat Completions accepts bounded base64 image data URLs. |
| Video inference | **Implemented for full BF16 CPU/Python API** | Exact 2 FPS / 96-frame sampling, temporal-2 patching, timestamps, prompt expansion, and placeholder fusion. GGUF `mmproj` and CUDA vision remain unavailable. |
| Whole-model CUDA vision | **Not implemented** | `vision_cuda=false`; a CUDA target load with `mmproj` fails before model/session mutation. |

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
  --gguf /models/dflash-Muse-Glimmer-30B-Q4_K_M.gguf \
  --gguf /models/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf \
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
| Packed backward input | `nfn_native_tile_linear_backward_input_packed_weight_float32_v1` | Frozen packed-base `dX` used by native QLoRA; no packed `dWeight`. |
| Embedding | `nfn_native_tile_glimmer_embedding_gather_float32_v1`, `nfn_native_tile_glimmer_embedding_batch_i32_float32_v1` | Vocabulary 202,048 and signed 32-bit training IDs. |
| Wide RMSNorm | `nfn_native_tile_glimmer_rms_norm_affine_float32_v1`, `nfn_native_tile_glimmer_rms_norm_backward_float32_v1` | Width 6,656, weightless/ordinary/centered modes, eps 1e-5/1e-8, FP32 reduction. |
| Positioned RoPE | `nfn_native_tile_glimmer_positioned_rope_float32_v1`, `nfn_native_tile_glimmer_positioned_rope_batch_float32_v1` | Explicit absolute positions, Q32/KV2 or KV8, half-split/interleaved layout. |
| Decode GQA | `nfn_native_tile_glimmer_gqa_decode_float32_v1` | Local ring or global cache, Q32/KV2, head 128, no fixed 1,024-key truncation. |
| Training attention | `nfn_native_tile_glimmer_attention_forward_float32_v1`, `..._backward_float32_v1` | Local causal window or global causal NoPE, online softmax and backward. |
| Cache commit | `nfn_native_tile_glimmer_cache_commit_bf16_v1` | BF16 hybrid local/global cache append with absolute positions. |
| Attention gate | `nfn_native_tile_glimmer_sigmoid_gate_float32_v1`, `..._backward_float32_v1` | `attention * sigmoid(gate)` before O projection. |
| Logit transform | `nfn_native_tile_glimmer_logit_transform_float32_v1`, `..._backward_float32_v1` | Multiplier-before-softcap and exact derivative. |
| DFlash attention | `nfn_native_tile_dflash_block_attention_float32_v1` | Accepted prefix plus bidirectional 16-row block, KV8, window 2,048. |
| Masked loss | `nfn_native_tile_glimmer_masked_cross_entropy_i32_float32_v1` | Signed int32 targets/mask, 202,048 vocabulary, Glimmer transformed logits. |
| Optimizer | `nfn_native_tile_glimmer_adamw_bf16_float32_v1` | BF16 parameters, FP32 gradients/moments, clipping and exact resume state. |

Existing generic linears, add/scale, SwiGLU, gradient accumulation, and copying
are reused where their contracts are exact. The legacy generic SDPA and fused
causal-attention functions are explicitly not used: their fixed 1,024-key
geometry and equal-width QKV assumptions cannot represent Glimmer.

`NFN_NATIVE_TILE_PACKED_WEIGHT_NF4_GROUP64` is a native training-only encoding:
each group stores one little-endian FP32 absmax followed by 32 low-nibble-first
NF4 code bytes. It is not GGUF K-quant and is never assigned a GGML type ID.

### 5.2 Remaining CUDA kernels and orchestration

These are the concrete missing native pieces:

1. **Whole-model CUDA vision.** Add versioned kernels for 1,176→1,536 patch
   projection, bilinear 32×32 position interpolation, indexed window
   permutation/inverse, affine LayerNorm forward, biased rectangular packed
   linears, 2-D positioned RoPE, noncausal varlen full/window attention,
   exact GELU, 2×2 pixel shuffle, and scaleless RMS. The resident vision weight
   plan must remain device-resident and report zero CPU model compute before
   setting `vision_cuda=true`.
2. **Native DPO.** Export efficient
   `nfn_native_tile_sequence_logp_i32_float32_v1` forward/backward and a raw
   `nfn_native_tile_dpo_pairwise_loss_float32_v1` forward/backward. The trainer
   still needs structured chosen/rejected records, a frozen reference handle,
   shared policy weights, and strict reference/resume lineage.
3. **Native reward/PPO.** Add sequence-mask-aware reward/value heads and
   backward, then a real resident rollout engine with frozen reference and
   reward models, per-token logprobs, GAE, KL, clipped objectives, minibatches,
   and complete resume state. Placeholder rollouts are not accepted.
4. **K-Quant adapter training.** Reuse packed forward and backward-input, but
   add a trainer-side authenticated GGUF tensor store and pin the resolved
   profile/digest/tensor table in every adapter checkpoint. Do not route it
   through NF4 or mutate official packed bytes.
5. **DFlash distillation.** Add the pinned corruption/reveal objective,
   target-tap capture, shared frozen target embedding/head, assistant-only
   checkpoint lineage, and acceptance-quality evaluation.
6. **Distributed 30B training.** Add tensor/data/pipeline parallel placement,
   collective kernels, per-device byte plans, distributed optimizer/checkpoint
   state, and all-device fit checks. Free bytes from separate GPUs must never
   be summed as if one allocation could span them.

Standalone sigmoid or logit-transform fusion is no longer a correctness gap;
the versioned functions above exist. BF16 activation/fused matmul paths remain
performance work because the current exact native ABI stores activations and
accumulates in FP32 while retaining BF16 or packed weights.

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
| K-Quant adapter SFT | No | No | TODO; must pin exact GGUF profile and preserve base bytes |
| DPO | Yes | No | Torch checkpoint binds a frozen reference digest |
| Reward model | Yes | No | Torch sequence-mask-aware scalar head |
| PPO | Yes | No | Torch rollout uses real policy/reference/reward/value/logprob/GAE state |
| DFlash distillation | Graph only | No | TODO; stock assistant inference is supported |
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

## 9. Vision and media boundary

Implemented:

- exact aspect-ratio LANCZOS image preprocessing, RGB 0.5/0.5 normalization,
  temporal-2/channel-major patch packing, learned position interpolation,
  window/full attention schedule, 2-D RoPE, 2×2 merge, projector, and
  decoder-width normalization;
- full-BF16 embedded vision on the portable C++ CPU runner;
- packed official mmproj CPU execution for still images;
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

- [ ] whole-model CUDA vision kernels and runner;
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
- [x] Torch SFT/LoRA/QLoRA/DPO/reward/PPO formula and gradient tests.
- [x] Native image/video preprocessing, vision CPU oracle, and media fusion.
- [x] Mandatory all-preset build/resolve/compile/forward/apply and ordered
  cross-preset tests.
- [x] Native inference binding and exact Glimmer trainer compile with a host C++
  toolchain.

Still required before performance or full-hardware release claims:

- [ ] build the real Tile-CUDA library with NVCC and run kernel parity under
  compute-sanitizer;
- [ ] run canonical BF16, Dynamic, 17GB, and DFlash end-to-end on representative
  24/32/80-GB GPUs, including long-context and memory-counter assertions;
- [ ] compare full-size logits/tokens against the pinned Transformers and
  llama.cpp `--jinja` oracles;
- [ ] publish p50/p95 TTFT, tokens/s, DFlash acceptance, resident bytes, load
  peak, and context/session envelopes;
- [ ] complete the native DPO/reward/PPO, K-adapter, DFlash-distillation,
  distributed-training, and CUDA-vision items above before setting those
  individual capability bits.

The development environment used for the final local pass did not contain
NVCC or a usable CUDA device. It compiled the C++ binding/trainer, ran the
fake-CUDA device ABI tests, and kept live-GPU claims explicitly pending rather
than silently falling back to CPU.
