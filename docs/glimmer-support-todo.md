# Muse Glimmer native support TODO

> **Status (2026-08-12): unsupported.** NeuralFn does not currently have an
> exact Muse Glimmer GPT template, a Glimmer native trainer/checkpoint family,
> a resident Glimmer C++ forward, K-Quant model-weight loading, or DFlash
> speculative decoding. The existing Tile-CUDA inference option is a hybrid
> historical-attention path with model compute on CPU; it is not whole-model
> CUDA inference.
>
> **Audit baseline:** NeuralFn commit
> `62616baf90bcf3c837c37acb9aca017cd6d2b618`. This document is a plan, not a
> capability claim. Proposed C ABI names below do not exist yet.

This plan covers the target decoder, the multimodal vision path, the separate
DFlash assistant, the official K-Quant-Dynamic and K-Quant-17GB artifacts,
native pretraining, post-training, and native C++/CUDA serving for
`meta-models/Muse-Glimmer-30B`.

## 1. Current answer and support boundary

| Capability | Current state | Blocking evidence |
| --- | --- | --- |
| Exact `muse_glimmer` GPT template | **No** | [`BlockSpec`](../neuralfn/config.py) has one homogeneous block specification; the current attention builder derives `head_dim = model_dim // num_heads` and cannot represent hidden size 6,656 with a 4,096-wide Q projection. |
| Torch reference training | **No, not for Glimmer** | General AR/SFT/DPO/PPO/reward graphs exist, but the decoder block, normalization, per-layer local/global schedule, gated attention, output scaling, vision tower, and DFlash assistant are not represented exactly. |
| Native C++/CUDA pretraining | **No** | [`native_registry.py`](../neuralfn/native_registry.py) has no Glimmer family/target. The reviewed graph adapter also rejects non-null `template_spec.finetune`. |
| Native C++/CUDA post-training | **No** | There is no Glimmer checkpoint initializer or native fine-tuning loop. LoRA/NF4/DPO/PPO modules in the Torch/Tile extension are not a Glimmer-native raw C ABI or production trainer. |
| Resident target-model inference | **No** | [`resident_binding.cpp`](../neuralfn/csrc/native_gpt2/resident_binding.cpp) loads only dense-v5, canonical LLaMA, and exact standard-MoE formats. |
| Whole-model CUDA inference | **No** | The documented Tile path keeps weights, projections, and row encoding on CPU and offloads only packed historical attention; see [Native CLI Inference](native-cli-inference.md#cache-and-sampling-controls). |
| K-Quant-Dynamic / K-Quant-17GB | **No** | NeuralFn has no GGUF reader, GGML `Q4_K`/`Q5_K`/`Q6_K` packed-weight descriptors, or packed CPU/CUDA GEMMs. Existing “quantized” linears still accept full FP32/BF16 weights and are not compatible. |
| Automatic VRAM precision selection | **No** | Model loading has no weight-precision option or variant catalog and does not query CUDA free memory. Existing CUDA device fields configure only the optional Tile KV-cache path. |
| DFlash speculative decoding | **No** | [`native_inference.py`](../neuralfn/native_inference.py) repeatedly calls one model/session `decode_one`, which commits one token. There is no assistant handle, proposal block, target verification pass, accepted-prefix calculation, or transactional cache rollback. |
| Image/video inference or tuning | **No** | The resident text API is text-only and the Glimmer vision encoder/projector is absent. |

Do not register or advertise Glimmer in the native capability registry merely
because the template can be rendered or a checkpoint can be inspected. Keep
the existing fail-closed separation between structural lowering, trainer
registration, architecture persistence, a real native forward, resident
inference, cache support, serving, and speculative decoding.

Track independent capability bits for BF16, K-Quant-17GB, and
K-Quant-Dynamic on CPU and whole-model CUDA, plus separate bits for DFlash,
vision, and VRAM auto-selection. Passing one packed profile must not promote the
other, and an `auto` bundle may consider only profiles whose exact backend bit
is true.

## 2. Pinned reference contract

Implementation must pin immutable upstream revisions and create local golden
fixtures before changing builders or kernels. The model repositories inspected
for this audit were:

- [Muse-Glimmer-30B model card](https://huggingface.co/meta-models/Muse-Glimmer-30B) and [main config at revision `a4e59da`](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/a4e59da52a7bc87ae7251dd5545c0dd437c44b68/config.json).
- [Muse-Glimmer-30B-assistant](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant) and [assistant config at revision `e8192f3`](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant/blob/e8192f3a8f617f74be2ce220360c89ef4789f39f/config.json).
- [Official Muse-Glimmer-30B GGUF bundle](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF) at revision
  `43c7eadd41352a299ea8e0a36b3157978dd63596`. Use the canonical filenames and
  hashes in section 2.4, not the retained lowercase legacy files.
- The official Transformers implementations at commit
  [`d1123114da1ab4395198146f4f84dae7fe8b693e`](https://github.com/huggingface/transformers/commit/d1123114da1ab4395198146f4f84dae7fe8b693e):
  [Muse Glimmer](https://github.com/huggingface/transformers/blob/d1123114da1ab4395198146f4f84dae7fe8b693e/src/transformers/models/muse_glimmer/modular_muse_glimmer.py),
  [the DFlash assistant](https://github.com/huggingface/transformers/blob/d1123114da1ab4395198146f4f84dae7fe8b693e/src/transformers/models/muse_glimmer_assistant/modular_muse_glimmer_assistant.py),
  and [DFlash candidate generation](https://github.com/huggingface/transformers/blob/d1123114da1ab4395198146f4f84dae7fe8b693e/src/transformers/generation/candidate_generator.py).
  Their raw file SHA-256 values are respectively
  `adc0b4efdfb5e1e4a245ec6738ef4d3de92550c080ac623c495b0ff5e7479caf`,
  `74263100a445bd0f74fc9f7b8867bbe41a6f09d41db8bce910699e1d24738642`,
  and `cf3ba7f85407bd24f5a6016135809137d57492fca9d4373f7813242c1b7b75c2`.
  Bind the main tokenizer/config/chat files to the already pinned `a4e59da`
  model snapshot and the assistant config to `e8192f3`; golden manifests store
  every individual file hash rather than resolving a moving branch.
- Use the independent [llama.cpp Muse Glimmer support merge at `62bf73d`](https://github.com/ggml-org/llama.cpp/commit/62bf73d)
  (upstream build `b10353` or later) only as a secondary GGUF/layout/runtime
  cross-check, and record the full commit in fixtures. Run its chat oracle with
  `--jinja`; the canonical embedded template requires it. The pinned model
  config and official Transformers implementation remain the semantic oracle.

### 2.1 Target text decoder

The exact target contract is not ordinary LLaMA or the current `gemma3`
preset:

- BF16, vocabulary 202,048, hidden size 6,656, 52 decoder layers, and
  intermediate size 19,968.
- 32 query heads, 2 key/value heads, and explicit head dimension 128. The Q
  projection is 6,656 -> 4,096 and K/V are each 6,656 -> 256; therefore the
  attention width is not the model width.
- A repeating three-local/one-global layer schedule. Local layers use a causal
  2,048-token sliding window and RoPE with theta 500,000. Global layers use
  full causal attention and NoPE; a single RoPE setting for all layers is
  incorrect.
- Weightless per-head RMS normalization of Q and K, followed by a query factor
  of 3.87. Attention softmax scaling remains `1 / sqrt(128)`.
- Sigmoid attention gating: project the normalized block input from 6,656 to
  4,096, apply sigmoid, multiply the merged attention output, then apply the
  4,096 -> 6,656 output projection.
- Four centered RMSNorms per decoder block. Centered means the learned factor
  is `(1 + weight)`, not `weight`. Input/pre-FFN norms use epsilon `1e-5`;
  post-attention/post-FFN norms use epsilon `1e-8` and normalize the sublayer
  result before the residual add (Gemma-2-style sandwich normalization).
- A weightless RMS normalization after token embedding, a learned final RMS
  norm, SwiGLU, bias-free linears, and untied embeddings/LM head.
- Logits use both an output multiplier and a softcap:
  `20 * tanh((0.19611613513818404 * raw_logits) / 20)`.
- Maximum position count 131,072. A dense `[sequence, sequence]` mask is never
  acceptable at this context length.

### 2.2 DFlash assistant and speculative protocol

The assistant is a separate artifact and is not a smaller copy of the target:

- Five layers, hidden size 6,656, intermediate size 19,968, 32 Q heads, 8 KV
  heads, head dimension 128, RoPE theta 500,000, and a 2,048-token window.
- It consumes target hidden states from layers `{1, 13, 25, 37, 49}`. The five
  6,656-wide vectors are concatenated and projected 33,280 -> 6,656 before the
  assistant stack.
- It uses raw target token embeddings for the noisy block; applying the target
  decoder's post-embedding RMS norm here is incorrect.
- Assistant attention is non-causal within the diffusion block. Queries are
  the current block, while K/V combine accepted target context with the current
  noise block. This is not the target decoder's causal attention kernel.
- Configured `block_size` is 16 and `mask_token_id` is 201,818. The current HF
  candidate generator constructs a 16-position window consisting of one
  accepted anchor plus 15 masked candidate positions, then applies the shared
  target LM head to assistant outputs after the anchor. Preserve the pinned
  implementation's exact off-by-one contract in tests rather than relying on
  the model card's shorthand description.
- A target verification forward scores the proposed block in parallel. Greedy
  generation accepts the matching prefix through the first mismatch and emits
  the target correction/bonus token. Sampled generation needs the full
  speculative `p/q` acceptance and residual-distribution rule.
- Target and assistant caches must record accepted context, append temporary
  candidate state, and crop/roll back rejected state without rebuilding an
  inconsistent position history.

### 2.3 Vision tower and projector

Full `MuseGlimmerForConditionalGeneration` support also requires:

- A 50-layer, 1,536-wide vision transformer with 16 heads, 8,960-wide FFNs,
  patch size 14, temporal patch size 2, and merge size 2.
- Dynamic image/video resizing and patch packing, learned 32x32 positional
  embeddings with bilinear interpolation, 2-D RoPE, and the configured mixture
  of windowed and full vision attention.
- A 2x2 pixel shuffle/merge from 1,536 to 6,144, then bias-free
  `6,144 -> 4,096 -> 4,096 -> 6,656` projection with GELU and the reference
  normalization order.
- Scatter/replacement of image/video placeholder-token embeddings and exact
  image/video position bookkeeping.

Text-only target support is a valid intermediate milestone only when its
manifest says `modalities=["text"]`; it must not claim compatibility with the
full conditional-generation artifact.

### 2.4 Official K-Quant artifact profiles

Pin these exact files from the official GGUF revision above. Their product
names identify complete mixed-per-tensor profiles; neither name is a new scalar
dtype or an integer value that may be passed to NeuralFn's existing quantized
linear ABI.

| Profile | Canonical artifact | Exact bytes and SHA-256 | Pinned tensor-type inventory |
| --- | --- | --- | --- |
| `k-quant-17gb` | `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` | 16,756,683,904; `4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e` | 313 F32, 365 Q4_K, 1 Q5_K, 52 Q6_K |
| `k-quant-dynamic` | `Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf` | 19,653,960,832; `ac7023d6a4c704eb9af54ab53e476a66b7f5b6c0ef2fc4a8dde5253c291a6c38` | 313 F32, 51 Q4_K, 130 Q5_K, 237 Q6_K |

Both canonical files are GGUF v3, declare architecture `muse-glimmer`,
`general.quantization_version=2`, `general.file_type=15` (MOSTLY_Q4_K_M),
contain 731 tensors, and retain the same 202,048-token/52-layer decoder
geometry. `Q4_K_XL` is therefore a canonical filename/profile label, not a
distinct embedded file type. Both output heads are Q5_K: it is the 17GB
artifact's lone Q5_K tensor and one of Dynamic's 130. Dynamic clearly uses more
Q5_K/Q6_K tensors, but its tensor-selection algorithm is not documented
upstream; do not invent claims about calibration, importance matrices, or
sensitivity routing.

The canonical optional companions in the same pinned repository are:

- `dflash-Muse-Glimmer-30B-Q4_K_M.gguf`, 1,631,208,128 bytes, SHA-256
  `b2e808bf656086fe86bd0d0bd990f01d33e377537a07c02d45371517c8b264ef`;
  it declares architecture `dflash`, contains 58 tensors, and has 22 F32,
  26 Q4_K, and 10 Q6_K tensors. Its target-layer metadata
  `{2,14,26,38,50}` is the one-based GGUF equivalent of the zero-based model
  config `{1,13,25,37,49}`; normalize explicitly once and regression-test it.
- `mmproj-Muse-Glimmer-30B-Q4_K_M.gguf`, 1,400,328,928 bytes, SHA-256
  `f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6`;
  despite its filename, this is the full vision encoder/projector companion.
  It declares `clip`/`mmproj`, contains 809 tensors, and mixes 506 F32,
  200 Q4_K, 100 Q6_K, and 3 BF16 tensors.

The two main files are text-only by themselves. Meta positions 17GB as the
24-GB-VRAM starting profile and Dynamic as the higher-quality 32-GB-VRAM
profile, reporting 1.0% and 0.2% average degradation respectively across its
15-benchmark comparison. Its rough published resident totals are approximately
17/19/20 GB for 17GB text/text+vision/text+vision+DFlash, and 20/22/23 GB for
the corresponding Dynamic modes. Those are hardware guidance, not a sufficient
allocation oracle: NeuralFn must budget the parsed tensors, configured
context/cache, workspaces, companions, and safety headroom in bytes before
loading.

## 3. What each completion label means

- **Template-ready:** the graph describes the exact target decoder and matches
  the pinned Torch reference forward/backward on tiny deterministic fixtures.
  It says nothing about native execution.
- **Native-train-ready:** the compiled C++/CUDA loop consumes the graph-bound
  topology, performs a real optimizer step, persists every architecture tensor
  and state field, resumes exactly, and passes parity gates.
- **Post-train-ready:** a separately named objective (full SFT, LoRA SFT,
  NF4 QLoRA SFT, K-Quant adapter SFT, DPO, reward model, or PPO) has its own
  data, loss, frozen-parameter, checkpoint, resume, and evaluation proof. One
  objective does not imply the others.
- **Target-resident-ready:** the target model performs prefill and decode in the
  C++ resident runtime with a proved lossless KV cache. CPU logits with a CUDA
  side operation do not qualify as whole-model CUDA.
- **K-Quant-ready:** the exact pinned mixed F32/Q4_K/Q5_K/Q6_K profile stays
  packed in resident memory, every encountered tensor type has a proved
  CPU/CUDA execution path, and output parity/memory gates pass. Loading GGUF
  metadata or dequantizing the full model does not qualify.
- **VRAM-auto-ready:** the resident CUDA loader selects the highest-fidelity
  compatible artifact that fits a byte-accounted free-memory budget and reports
  its decision. A filename heuristic or use of total VRAM alone does not
  qualify.
- **DFlash-ready:** target plus assistant reproduce non-speculative target
  output, implement correct acceptance and rollback, and report acceptance,
  memory, and latency metrics. Loading an assistant checkpoint alone does not
  qualify.
- **Multimodal-ready:** image and video preprocessing, vision execution,
  placeholder fusion, native API input, and parity are all proved.

## 4. Add `muse_glimmer` as a GPT template

### 4.1 Public model-spec additions

Make additions backward-compatible and default-neutral for every existing
preset. A concrete design should add:

- `"muse_glimmer"` to `BackboneType`. Do not classify it as LLaMA merely to
  inherit a currently green native capability.
- `BlockSpec.head_dim: int | None` and
  `BlockSpec.attention_inner_dim: int | None`; when absent, preserve the current
  `model_dim // num_heads` behavior. Glimmer sets 128 and 4,096.
- An explicit FFN width (`intermediate_size`) so 19,968 is serialized rather than
  reconstructed through multiplier/rounding rules.
- A repeatable per-layer attention pattern, preferably a typed
  `LayerAttentionSpec` containing attention kind, window, position encoding,
  and RoPE theta. Glimmer's pattern is local/local/local/global with the global
  entry using NoPE.
- `qk_norm_kind="weightless_rms"`, `qk_norm_eps`, and `q_scale_factor=3.87`.
- `attention_gate="sigmoid"` with its projection width.
- `norm_layout="sandwich"`, `centered_rms_norm=True`, `norm_eps=1e-5`, and
  `post_norm_eps=1e-8`, plus an explicit weightless embedding norm.
- `ModelSpec.max_position_embeddings=131072` and
  `ModelSpec.output_multiplier=0.19611613513818404`. Keep
  `logit_softcap=20.0` as a separate operation.
- Optional typed `VisionSpec` and `DFlashSpec` metadata. Do not force the
  assistant into the normal autoregressive root's token/target ports.

Update `model_spec_to_dict`/load validation and Native IR serialization at the
same time. These are public API additions and therefore require the SDK and
framework documentation listed below.

### 4.2 Builders and graph layout

- [ ] Add `build_muse_glimmer_spec()` with the production defaults above and a
  tiny `preview_defaults=True`/test geometry that retains the non-square
  hidden-to-attention relationship.
- [ ] Add a Glimmer attention builder that accepts explicit Q/K/V widths,
  weightless QK norm, Q scaling, optional RoPE, and sigmoid attention gating.
- [ ] Add a Glimmer decoder block with the exact four-norm sandwich ordering;
  do not reuse the current two-norm pre-normalized decoder block.
- [ ] Make the model-stage loop instantiate the local/local/local/global
  pattern per layer instead of linking every layer to one homogeneous block.
- [ ] Add post-embedding weightless RMS norm, the correct learned final norm,
  untied LM head, output multiplier, and softcap in their reference order.
- [ ] Add a tensor-valued scale builtin (`tensor_scale`) for the output
  multiplier. Do not repurpose the existing loss-typed `loss_scale` node.
- [ ] Decompose SwiGLU into checkpoint-addressable `gate_proj`, `up_proj`, SiLU,
  multiply, and `down_proj` nodes; the current monolithic module cannot map or
  adapt each Glimmer projection independently.
- [ ] Refactor one Glimmer-aware body factory used by AR-loss, SFT, logits,
  hidden-state, DPO, reward, and PPO wrappers. The current wrappers duplicate a
  generic two-norm decoder and would otherwise silently lose Glimmer semantics.
- [ ] Add `build_muse_glimmer_assistant_graph()` as a companion graph with
  target-hidden-state, raw-noise-embedding, position, mask, and cache contracts.
  It is not an AR GPT root and should not silently be selected for ordinary
  training.
- [ ] Add separate vision/projector graph builders before setting multimodal
  capability true.
- [ ] Use unique variant families such as `muse_glimmer_attention`,
  `muse_glimmer_block`, and `muse_glimmer_dflash_attention`. Avoid aliasing them
  to the flat `attention`/`attn_block` families unless port compatibility is
  proved. Preserve the inline-subgraph fallback in Python and the editor.

### 4.3 Preset registration and required documentation

- [ ] Add `muse_glimmer` to `SHIPPED_GPT_TEMPLATE_BASE_PRESETS` and dispatch it
  in `build_model_spec_from_config()`. Do not add it to
  `MODERN_BASE_PRESETS`; it already has an exact architecture contract.
- [ ] Add it to
  [`shipped_gpt_template_presets.h`](../neuralfn/csrc/native_train/shipped_gpt_template_presets.h)
  for catalog parity, but keep native execution capability false until its
  independent gates pass.
- [ ] Add
  `<option value="muse_glimmer">Muse Glimmer 30B</option>` to
  [`Toolbar.tsx`](../editor/src/components/Toolbar.tsx).
- [ ] If new builtin neuron types are introduced, update
  [`tests/test_builtin_neurons.py`](../tests/test_builtin_neurons.py) and
  [builtin-neuron docs](python-sdk/builtins.md).
- [ ] Update fixed shipped-preset counts and native coverage tables, including
  assertions outside `test_template_presets.py`.
- [ ] Update [`README.md`](../README.md), [`CHANGELOG.md`](../CHANGELOG.md),
  [Python config docs](python-sdk/config.md),
  [templates and presets](framework-guide/templates-and-presets.md), and the
  [`neuralfn-torch`](../.cursor/skills/neuralfn-torch/SKILL.md) and
  [`neuralfn-mcp`](../.cursor/skills/neuralfn-mcp/SKILL.md) skills in the same
  implementation change.
- [ ] Run the mandatory preset gate:

  ```bash
  python -m pytest tests/test_template_presets.py -x -q
  ```

  The catalog-derived `PRESETS` list must include `muse_glimmer`; all preset
  payload, variant resolution, compile/forward, server apply, and ordered
  back-to-back preset-pair tests must pass.

The first shipped `muse_glimmer` preset should be explicitly text-decoder-only.
Attach the vision and DFlash graphs through versioned companion metadata after
their ports and runtimes are stable, so later additions do not silently change
the root serialization contract.

## 5. Tokenizer, dataset, and checkpoint prerequisites

### 5.1 Replace the native `uint16` token ceiling

This is a hard prerequisite, not an optimization. Glimmer's vocabulary is
202,048; BOS/EOS and multimodal/mask IDs are also above 65,535. The current
native cached-shard path requires `uint16_shards` and recommends a tokenizer
with at most 65,536 entries.

- [ ] Define a versioned `uint32_shards` (or signed `int32_shards`) format with
  endianness, header, tokenizer fingerprint, vocabulary, special-token map,
  shard content digest, document boundaries, and loss-mask/role-span sidecars.
- [ ] Keep read compatibility with existing uint16 datasets and choose the
  width from metadata; never reinterpret old shards.
- [ ] Add C++ samplers, pinned host/device arenas, range checks, resume sampler
  state, and dataset identity binding for 32-bit IDs.
- [ ] Add 32-bit target variants for embedding backward, CE, fused LM-head
  training, validation, DPO log-prob gathering, and DFlash mask construction.
  A correctness fallback may widen u32 IDs to the existing int64 embedding/CE
  APIs on device, but production must avoid per-step host conversion.
- [ ] Widen graph token-port metadata beyond `[0, 65535]` without breaking
  existing serialized graphs.

### 5.2 Exact tokenizer and conversation format

- [ ] Load and fingerprint the artifact's `tokenizer.json`, added-token table,
  normalizer/pre-tokenizer, BOS/EOS behavior, image/video/mask IDs, and chat
  template. Add golden encode/decode and rendered-conversation vectors.
- [ ] Extend [`native_chat.py`](../neuralfn/native_chat.py), which currently
  accepts artifact-declared tiktoken encodings only. Do not label an unrelated
  tiktoken vocabulary as Glimmer-compatible.
- [ ] Preserve the artifact's ATEM/Jinja and special-token behavior through a
  reviewed deterministic renderer rather than executing arbitrary template
  code in the resident process. First prove system/user/assistant text; reject
  unimplemented reasoning, tools, image, or video constructs instead of
  falling back to `plain_roles`.
- [ ] Bind the generation-stop contract exactly: `<|end_of_text|>` (200001) and
  `<|eot|>` (200008) are EOS tokens; `<|eom|>` is a message boundary and must
  not be promoted to a generation stop. Cover all three in chat/streaming
  goldens and compare the canonical GGUF template under llama.cpp `--jinja`.
- [ ] Include the tokenizer and template hashes in target, assistant, adapter,
  and dataset manifests.

### 5.3 Checkpoint import and persistence

- [ ] Implement streaming, bounded-memory safetensors index/shard import for
  the target language model, vision tower/projector, and assistant.
- [ ] Map every upstream tensor explicitly, including four norms per target
  block, attention gate projections, untied embedding/head, per-layer
  local/global metadata, assistant context projection, and vision tensors.
- [ ] Preserve centered-norm parameters exactly. If NeuralFn stores the raw
  centered parameter, the kernel must multiply by `(1 + weight)`; if an
  ordinary affine RMSNorm representation is deliberately used, add one during
  conversion for the four centered block norms only. Do not add one to the
  final ordinary norm. Prove Q/K permutation/layout against NeuralFn's RoPE
  convention rather than copying another converter's unpermute blindly.
- [ ] Define proposed versioned formats such as
  `neuralfn.native_family_muse_glimmer.bf16.v1` and
  `neuralfn.native_family_muse_glimmer_dflash.bf16.v1`. These names remain
  provisional until implemented and reviewed.
- [ ] Preserve the exact official K-Quant files as immutable source artifacts
  or convert them to a versioned NeuralFn packed container without changing any
  tensor's quantization type. Record the source GGUF SHA, complete tensor-type
  table, byte layout, and conversion-tool revision either way.
- [ ] Bind each assistant manifest to the tokenizer/config lineage, target
  hidden-layer set, block size, LM-head contract, and an explicit allowlist of
  compatible target profile/artifact digests. The official quantized companion
  may list both pinned K-Quant mains after pairwise parity tests; a newly trained
  assistant defaults to one exact target digest. Never infer compatibility from
  equal hidden size alone.
- [ ] Give the mmproj manifest an equivalent allowlist binding target
  profile/digest, processor/tokenizer/placeholder hashes, visual-token geometry,
  and the exact 6,656-wide projection interface. Encode ownership explicitly:
  the full BF16 conditional-generation artifact has embedded vision and must not
  attach or budget mmproj, while text-only K-Quant mains require the compatible
  mmproj before setting vision capability.
- [ ] Extend Native Execution Manifest v1 additively with optional
  `primary_checkpoint_variant`, `checkpoint_variants`,
  `companion_checkpoints`, and `speculative_decoding` fields. Preserve the
  existing top-level `checkpoint` object and old-manifest round trips; do not
  replace it with a nested primary shape merely to attach variants or the
  assistant.
- [ ] Make every variant reuse the current executable checkpoint fields exactly:
  `format`, relative `artifact_path`, `target_nbytes`, and `target_sha256`.
  Require the top-level `checkpoint` to equal those four fields of
  `checkpoint_variants[primary_checkpoint_variant]`; disagreement is invalid.
  Put original-GGUF path/format/SHA and converter revision in a separate
  `source_provenance` object so a converted container's executable digest is
  never confused with its source digest.
- [ ] Give each variant a `required_kernel_profile`, parsed tensor-encoding
  inventory, exact `resident_weight_bytes`, `peak_load_staging_bytes`,
  `max_workspace_bytes`, and a versioned measured-memory profile containing
  `minimum_total_vram_bytes` plus its backend/device/config fingerprint. The
  binding independently advertises supported kernel profiles; a manifest
  cannot self-attest support. Auto fidelity order is fixed by the public IDs
  in code (`bf16` > Dynamic > 17GB), not by a caller-controlled numeric rank.
- [ ] A future tensor-parallel variant needs a per-device shard/placement table
  with artifact digest, resident/staging/workspace bytes, and cache allocation
  for each device. Until that schema and all-devices-fit check exist, limit
  Glimmer `auto` to one whole-model CUDA device.
- [ ] Persist optimizer state, master weights, RNG, sampler position, gradient
  accumulation, objective, quantization/adapters, source graph SHA, and source
  model revisions. `DONE` must be published last after all bytes are durable.
- [ ] Prove import with metadata/tensor-shape validation, sampled tensor hashes,
  full payload digest, and target/assistant reference-logit parity. A successful
  metadata read is not an inference proof.

### 5.4 Strict GGUF/K-Quant loading

- [ ] Add a bounded, fail-closed GGUF v3 reader for the pinned
  `general.architecture=muse-glimmer` artifacts. Validate magic/version,
  endianness, metadata types, dimensions, tensor names, byte offsets, alignment,
  non-overlap, exact extents, file length, and SHA before exposing any pointer
  to the runtime.
- [ ] Match every tensor against an explicit Glimmer allowlist and shape table.
  Reject missing, duplicate, unexpected, or unsupported tensors and encoding
  IDs. This milestone is support for the pinned Muse Glimmer profiles, not a
  claim that arbitrary GGUF models are supported.
- [ ] Introduce an immutable typed-weight descriptor containing the data pointer
  and byte extent, rows/columns, GGML encoding (including dense F32/BF16),
  block/row stride, and scale/min metadata where applicable. Keep each main or
  companion's per-tensor types intact; especially, do not collapse Dynamic to
  one global quantization kind.
- [ ] Never map these formats onto the existing integer `quant_kind`. That ABI
  accepts `const float *` master weights and one dense path can fall through to
  raw float reads for unknown kinds, which could silently interpret packed
  bytes as FP32. Use a new versioned descriptor and reject unknown encodings.
- [ ] Preserve the canonical embedded tokenizer/chat-template material and bind
  its hash to the artifact. The repository retains older lowercase artifacts;
  selection and download resolution must use the canonical pinned names. A
  noncanonical digest needs a separately registered profile plus full parser,
  kernel, memory, and output-parity proof; a caller-provided checksum alone
  proves integrity, not compatibility.
- [ ] Make the variant bundle self-contained or explicitly configured. `auto`
  selects only among verified local/catalogued artifacts; it must not silently
  download a missing higher-precision file during model startup.

## 6. Native/CUDA kernel gap inventory

The names in the **Proposed raw ABI** column are design placeholders. They are
listed to make the missing work concrete; they are not present in
[`tile_ops.h`](../neuralfn/csrc/native_train/tile_ops.h).

### 6.1 Existing primitives that can be reused or extended

| Existing raw Tile-CUDA primitive | Usable part | Glimmer limitation |
| --- | --- | --- |
| `nfn_native_tile_linear_*` and backward variants | Bias-free asymmetric GEMMs and optimizer handoff | Must be planned for 6,656/4,096/19,968/33,280/202,048 dimensions and BF16 resident weights; grouped/fused projection work is a performance follow-up. |
| `nfn_native_tile_swiglu_float32` / backward | SwiGLU pointwise math | Needs BF16 activation/storage integration at Glimmer widths; GEMMs remain separate. |
| `nfn_native_tile_token_embedding_float32` | Int64 lookup and weight gradient | Current native dataset/fast paths use uint16. A 32-bit path and BF16 output are missing. |
| `nfn_native_tile_rotary_embedding_float32` / backward | Small reference RoPE | Needs cached-position offsets, local/global dispatch, BF16, and long-context/decode validation. |
| `nfn_native_tile_rms_norm_float32` / input backward | Weightless tiny-width reference | The implementation materializes a fixed 1,024-element tile, so it is not correct for hidden widths 1,536, 4,096, or 6,656. It has no affine weight or weight gradient. |
| `nfn_native_tile_qk_rms_norm_packed_bf16_*` | Packed Q/K RMS math | Assumes Q and K have the same head count and a `3 * model_dim` packed QKV layout. Glimmer uses Q32/KV2 (target) or Q32/KV8 (assistant), explicit head dim 128, and separate projection widths. It also lacks Q factor 3.87. |
| `nfn_native_tile_scaled_dot_product_attention_*` | Tiny/reference GQA and q-length/k-length interface | The generic implementation uses a fixed 1,024-key tile and its reviewed binding is capped accordingly; raw sparse/window execution explicitly rejects `seq_k > 1024`, while a raw dense call above 1,024 can ignore later keys rather than reject. Glimmer's local window is 2,048 and global context is 131,072, so neither may use this route. Optimized packed paths also require equal Q/KV heads and narrower head geometry. |
| `nfn_native_tile_fused_causal_attention_*` | A reference composition for canonical dense attention | Explicitly blacklist it for Glimmer: it assumes Q width and O input equal `model_dim`, applies RoPE uniformly, has no local-window/gate schedule, and its current SDPA element-count handoff is not proved for `seq_len > 1`. |
| `nfn_native_tile_add_float32`, `vector_binary`, `concat_last_dim`, `tanh` | Reference composition | No named raw sigmoid-gate kernel, no sandwich-norm fusion, and no fused output multiplier+softcap contract. |
| Token and masked CE raw APIs | Reference AR/SFT loss with int64 targets | Optimized BF16/strided classifier paths are uint16-target-centric and do not fuse Glimmer's output multiplier/softcap. Full `[rows, 202048]` logits are too costly for production post-training. |
| `nfn_native_tile_linear_quantized_float32` / NVFP4 linears | Reference fake-quant/activation-quant experiments | Not K-Quant: stored weights remain FP32 or BF16 and are quantized while executing. There is no packed-byte pointer, Q4_K/Q5_K/Q6_K block layout, scale/min metadata, or resident-memory saving. |
| AdamW, clipping, accumulation, fill/reduction APIs | Optimizer mechanics | A Glimmer parameter table, frozen/adapted groups, checkpoint state, and end-to-end loop do not exist. |

Torch/Tile extension registry entries such as `lora_linear`, `nf4_linear`,
`dpo_pairwise_loss`, `reward_head`, and `ppo_clipped_loss` are useful references,
but the raw native trainer ABI exports only masked CE from that post-training
set. Do not count extension-backed Python modules as C++-native support.

### 6.2 Required target-decoder training kernels

| Required operation | Proposed raw ABI | Exact missing contract |
| --- | --- | --- |
| Wide weightless RMSNorm | `nfn_native_tile_wide_rms_norm_bf16_{forward,backward}` | Arbitrary width through at least 6,656, FP32 reduction, BF16 I/O, saved `rstd`, deterministic tail handling, and no learned factor for embedding/QK use. |
| Wide learned/centered RMSNorm | `nfn_native_tile_centered_rms_norm_bf16_{forward,backward}` | Learned `(1 + weight)` mode plus ordinary learned-weight mode, `dX` and FP32-accumulated `dWeight`, eps `1e-5` and `1e-8`, widths 1,536/4,096/6,656. |
| Sandwich norm + residual | `nfn_native_tile_post_rms_residual_bf16_{forward,backward}` | Normalize the attention/FFN output before residual addition while preserving the pre-residual tensor and exact gradient branches. This may first be composed from correct wide norm and add kernels, then fused. |
| Asymmetric Q/K RMS + Q scaling | `nfn_native_tile_gqa_qk_rms_scale_bf16_{forward,backward}` | Separate Q32 and KV2 arrays, head dim 128, weightless norm, Q-only factor 3.87, saved stats, no materialized repeated KV. Assistant mode must also accept KV8 without a second incompatible ABI. |
| Local causal GQA forward/backward | `nfn_native_tile_gqa_local_attention_bf16_{forward,backward}` | Q32/KV2, head 128, causal window 2,048, RoPE positions, stable online softmax/LSE, arbitrary training sequence, dropout zero, and no dense attention matrix. |
| Global causal GQA forward/backward | `nfn_native_tile_gqa_global_attention_bf16_{forward,backward}` | Q32/KV2, head 128, NoPE, context to 131,072, block-streaming/FlashAttention memory behavior, saved/recomputed LSE, and deterministic parity tolerances. |
| Sigmoid attention gate | `nfn_native_tile_sigmoid_mul_bf16_{forward,backward}` | Elementwise `attention * sigmoid(gate_proj(normed_input))` over 4,096 channels, including gradients for both inputs. The gate projection itself reuses linear GEMM. |
| Glimmer logit transform | `nfn_native_tile_glimmer_logits_bf16_{forward,backward}` | Apply multiplier `0.19611613513818404` before softcap 20 and propagate both derivatives. Support a strided/chunked 202,048-row vocabulary without changing untied-head semantics. |
| Fused large-vocab CE | `nfn_native_tile_glimmer_lm_ce_bf16_i32_{forward,backward}` | Chunked untied LM head, output multiplier+softcap, signed int32 targets so `ignore_index=-100` is representable, optional mask weights, row losses, `dHidden`, `dHead`, and no full-logit allocation. A u32 variant must use a separate validity mask and forbid sentinel IDs. This is required for pretraining and SFT. |
| 32-bit embedding fast path | `nfn_native_tile_token_embedding_u32_bf16_{forward,backward_weight}` | IDs through 202,047 plus exact range failure, BF16 output, collision-safe weight-gradient accumulation, and compatibility with tied/untied choices. |

Optional performance fusions—grouped Q/K/V/gate projection, fused
SwiGLU+projection, activation recomputation, and norm/residual epilogues—must
come after the unfused exact path passes the same oracle.

The asymmetric Q/K normalization and Q factor do not require a new fused kernel
for the first correctness path: treat each 128-wide head as a row, use a
correct existing/reference weightless RMSNorm, and compose scaling. Likewise,
FP32 scale+tanh and elementary pointwise operations can establish the final
logit/gate oracle. The proposed BF16 fusions become mandatory only for the
production performance/precision path. Wide 6,656 norms, long-context GQA,
cache-aware decode, and memory-bounded 202k-vocabulary head/loss paths are hard
correctness or feasibility requirements from the start.

### 6.3 Required target resident-inference kernels and cache ABI

| Required operation | Proposed raw ABI/runtime contract | Exact missing contract |
| --- | --- | --- |
| Local prefill/decode GQA | `nfn_native_tile_gqa_local_prefill_bf16`, `..._decode_bf16` | Q32/KV2/head128, 2,048-token ring cache, absolute position tracking for RoPE, prompt chunks and `q_len` 1..16, wraparound-safe gather, no KV repetition. |
| Global prefill/decode GQA | `nfn_native_tile_gqa_global_prefill_bf16`, `..._decode_bf16` | NoPE, paged/block KV cache through 131,072, `q_len` 1..16 verification, online softmax over pages, and batch-one correctness before batching. |
| Transactional KV updates | Versioned resident cache methods `mark`, `append_tentative`, `commit(n)`, `crop(mark+n)` | A cache/session ABI rather than just arithmetic: target verification must be able to score candidates without irrevocably committing all of them. Local-ring and global-paged layers must roll back to the same logical token position. |
| Multirow target head | `nfn_native_tile_glimmer_lm_head_select_bf16` | Compute argmax/top-k/log-probs for 1..16 rows over 202,048 tokens with multiplier+softcap, avoiding a host copy of the full matrix. |
| Device sampling | `nfn_native_tile_sample_logits_bf16` | Temperature, top-k, top-p, deterministic seed/counter, selected-token log-prob, and exact-zero greedy mode. A host correctness fallback is acceptable initially, but it is not the performance endpoint. |

Whole-model CUDA requires a resident memory planner and C++ runner that keep
BF16 weights/activations and KV caches on the selected device. Adding these
symbols only to the trainer sidecar does not make the resident binding use
them.

The official BF16 main's two safetensors total exactly 59,553,435,272 bytes
(roughly 55.46 GiB) and already include the vision encoder/projector. They are
before KV cache, activations, allocator overhead, and the separately stored
assistant—not before vision. A text-only BF16 conversion may extract a smaller
language subset, but its manifest must report measured component bytes rather
than reuse the full-checkpoint number. Conversely, each K-Quant main is
text-only and must add the mmproj bytes exactly once when vision is enabled.
A 32-GiB GPU therefore cannot be the full BF16 single-device completion target.
The two reviewed K-Quant profiles below are the single-device plan for
24/32-GB hardware; tensor parallelism is still required for BF16 where its
exact budget does not fit. At 131,072 tokens, target BF16 KV is approximately
1.7 GiB for batch one when local layers use 2,048-token rings and global layers
retain full history; storing full history for every layer would waste
substantially more memory.

### 6.4 Required K-Quant CPU/CUDA kernels and packed runtime

The pinned mains contain only F32, Q4_K, Q5_K, and Q6_K tensors, but in
different mixtures. Implement against the parsed tensor table, not the
marketing suffix. The initial ABI may use one descriptor-dispatched symbol or
one symbol per encoding; in both cases an unknown type must fail before launch.
For the pinned GGML quantization-v2 layouts, the reference decoders must handle
256-value superblocks with exact serialized block widths of 144 bytes for
Q4_K, 176 bytes for Q5_K, and 210 bytes for Q6_K, including their FP16
block scales, packed scale/min fields, high-bit planes, and signed sub-block
scales. Lock these facts with header fixtures rather than relying on C++ struct
padding.

| Required operation | Proposed raw ABI/runtime contract | Exact missing contract |
| --- | --- | --- |
| Packed/dense tensor descriptor | `NfnNativeTileWeightDescriptorV1` | Byte pointer/extent, logical rows/columns, Q4_K/Q5_K/Q6_K/F32/BF16 type, block size, row stride, scale/min layout where applicable, alignment, and source-tensor digest. It must represent mixed main/companion models—including mmproj's BF16 tensors—without a full-precision shadow copy. |
| Portable dequant/dot oracle | `nfn_native_kquant_dequant_row_{q4_k,q5_k,q6_k}_f32` and packed CPU matvec/matmul | Bit-exact block decoding and FP32 accumulation for each encountered GGML type, tail validation, and independent parity with the pinned GGUF reference. This makes resident C++ CPU execution possible without whole-model dequantization. |
| CUDA packed projection | `nfn_native_tile_kquant_linear_bf16_forward` | Direct Q4_K/Q5_K/Q6_K weight consumption with BF16 activation/output and FP32 accumulation for 6,656->4,096/256, 4,096->6,656, and 6,656<->19,968. Dequantize bounded tiles in registers/shared memory; never materialize the full weight or model. |
| CUDA packed projection dX | `nfn_native_tile_kquant_linear_bf16_backward_input` | Gradient with respect to activations for an immutable packed base, required for frozen-base LoRA/DPO paths. Base `dWeight` is intentionally absent until a separately specified quant-aware training format exists. |
| Mixed projection/fusion dispatch | Resident packed linear planner | Dispatch every tensor by its descriptor. Fuse Q/K/O or gate/up only when shapes, encodings, blocks, and alignment are compatible; V may use a different precision and must remain separate when required. F32 tensors use a proved dense path. |
| Packed embedding gather | `nfn_native_tile_kquant_embedding_bf16` | Direct row lookup for the exact embedding encoding, wide i32/u32 token IDs, block-tail handling, and no dense table expansion. Only required when the parsed artifact marks that table packed. |
| Packed 202k LM head | `nfn_native_tile_kquant_lm_head_{select,ce,sequence_logp}_bf16_i32` | Chunked Q4_K/Q5_K/Q6_K classifier matmul for 1..16 rows, Glimmer multiplier/softcap, top-k/log-prob/CE outputs, and no full dequantized head. CE/log-prob backward returns `dHidden` but no packed-base `dWeight`, enabling frozen-base adapter SFT/DPO. The 17GB output head is Q5_K, so this is required for that profile. |
| Packed companion execution | The same descriptors/kernels plus shape-specific dispatch | The official DFlash GGUF mixes F32/Q4_K/Q6_K. Cover 33,280->6,656 context projection; 6,656->4,096 Q/gate; 6,656->1,024 K/V for KV8; 4,096->6,656 O; and 6,656<->19,968 MLP shapes. The full mmproj vision companion has F32/Q4_K/Q6_K/BF16 tensors, so its exact 809-tensor shape/type table and dense BF16 dispatch are required before vision capability; do not infer its layout from the filename. |

The current native family store allocates four bytes per parameter and the
resident LLaMA loader reads a complete `vector<float>`. K-Quant therefore needs
a separate immutable packed store, streaming/mapped host input, direct device
upload, and typed views. Neither NF4 QLoRA, NVFP4 activation packing,
TurboQuant KV compression, nor the existing fake-quant linear is reusable as a
format alias.

### 6.5 Required DFlash kernels and speculative runtime operations

| Required operation | Proposed raw ABI/runtime contract | Exact missing contract |
| --- | --- | --- |
| Target hidden capture | Executor taps, optionally `nfn_native_tile_gather_target_hiddens_bf16` | The hard requirement is pointer/liveness capture of only layers `{1,13,25,37,49}` for newly accepted rows. Existing concat plus GEMM can establish correctness; a fused BF16 gather/pack is an optimization. The context projection is 33,280 -> 6,656. |
| Noise-block construction | `nfn_native_tile_dflash_build_noise_i32` | Last accepted anchor plus 15 mask IDs, position IDs, valid-length/loss mask, and raw target embedding lookup without the target embedding RMS norm. Existing `nfn_native_tile_mask_scheduler_int64` is only a partial random-mask primitive and emits no loss mask/exact anchor protocol; the u16 diffusion mask hardcodes an unusable mask ID. |
| Assistant asymmetric attention | `nfn_native_tile_dflash_attention_bf16_{forward,backward}` | Q32/KV8/head128; non-causal/bidirectional 16-position query block; accepted-context plus noise K/V; 2,048 window; asymmetric RoPE query/key positions; training LSE/backward. Target causal kernels are not a substitute. |
| DFlash cache maintenance | Versioned assistant cache `record_accepted`, `append_noise`, `crop_noise`, `rebase` | Cache accepted context once, discard the previous noise window before the next proposal, and update query offsets/masks after partial acceptance. |
| Shared-head proposal scoring | Reuse `nfn_native_tile_glimmer_lm_head_select_bf16` in multirow mode | Apply the target LM head and logit transform to assistant rows after the anchor; return 15 candidate distributions/tokens with sampling processors in the correct order. |
| Target block verification | Resident `verify_candidates`/`decode_block` operation | One target forward over candidate tokens plus the bonus position, selected target hidden-state capture, no immediate full commit, and exact cache checkpoint/crop behavior. Calling `decode_one` 15 times is not speculative decoding. |
| Greedy prefix acceptance | `nfn_native_tile_speculative_prefix_match_i32` | Find the first target/draft mismatch, return the accepted-prefix count, and select the correction/bonus token without reading stale padded rows. |
| Sampled acceptance/rejection | `nfn_native_tile_speculative_accept_bf16` | Stable `min(1,p/q)` acceptance, counter-based RNG, first rejection, and residual sampling proportional to `max(p-q,0)` over the full 202,048-token p and q distributions on device. Selected-token log-probs alone are insufficient. Pin logits-processor order and semantics to the chosen upstream revision and prove deterministic seeded replay. |

The resident API must own two compatible handles (target and assistant) while
sharing the target tokenizer, raw embedding table, and LM head. The assistant
artifact must fail closed when the selected target profile/digest is absent
from its compatibility allowlist or its hidden-layer contract differs.

### 6.6 Required vision kernels

These may follow text and DFlash support, but all are required before claiming
the full conditional-generation model:

- `nfn_native_tile_glimmer_patch_pack_bf16_{forward,backward}` for temporal-2,
  RGB, 14x14 patches and the 1,176 -> 1,536 projection input layout.
- `nfn_native_tile_bilinear_pos_embed_2d_bf16_{forward,backward}` for the learned
  32x32 table at dynamic image grids.
- `nfn_native_tile_rope_2d_bf16_{forward,backward}` with the reference
  height/width interleave and position transformation.
- `nfn_native_tile_vision_varlen_attention_bf16_{forward,backward}` for packed
  images/videos, `cu_seqlens`, window reindexing, and configured full layers.
- The wide affine norm kernel above at width 1,536, plus existing GELU/GEMMs.
- `nfn_native_tile_pixel_shuffle_2x2_bf16_{forward,backward}` for
  `4 * 1536 -> 6144` and
  `nfn_native_tile_scatter_modal_embeddings_bf16_{forward,backward}` for exact
  image/video placeholder replacement.

Host preprocessing is also required: resize/token-budget policy, normalization,
temporal frame grouping, patch-grid metadata, placeholder validation, and API
multipart/content-part decoding are not CUDA kernels.

### 6.7 Required post-training kernels

| Objective | Reusable support | Missing native production ABI/work |
| --- | --- | --- |
| Full SFT | General backward, masked CE reference, AdamW | Use the fused Glimmer i32 masked LM-head/CE path; response-span masks, packed sequences, all target backward kernels, activation checkpointing, and full-state resume. |
| LoRA SFT | General GEMMs can compose `base + scale * B(A(x))` | Add `nfn_native_tile_lora_linear_bf16_{forward,backward}` or an equivalent proved composition that freezes base weights, accumulates only A/B (and opted-in bias) gradients, supports all non-square Glimmer projections, and serializes adapters by canonical tensor path. |
| QLoRA SFT | Torch `nf4_linear` is a functional reference | Add `nfn_native_tile_nf4_lora_linear_bf16_{forward,backward}` that consumes packed NF4 plus group scales directly, returns base `dX` and A/B gradients only, and has an explicit format contract. The current Python stage dequantizes the complete base to FP32 on every forward, so it is not a viable production primitive. Do not conflate NF4 QLoRA with TurboQuant/KV-cache or other inference quantization. |
| K-Quant adapter SFT | Dense LoRA A/B GEMMs and the proposed packed forward are reusable | Add the packed linear `dX` path, keep every packed base byte immutable, and optimize only dense adapters. Bind the adapter to the exact source GGUF digest, resolved profile, and complete tensor-type table. Call this K-Quant adapter tuning, not NF4 QLoRA. |
| DPO | Torch/Tile extension has forward references | Raw C++ needs `nfn_native_tile_sequence_logp_bf16_i32_{forward,backward}` (mask, ignore index, Glimmer logit transform), policy/reference dual-forward scheduling, `nfn_native_tile_dpo_pairwise_loss_float32_{forward,backward}`, and policy-only backward. The existing extension-only sequence-logp/DPO forwards and Torch backwards are not a raw native ABI or efficient 202k-vocab implementation. |
| Reward model | Linear head can reuse GEMM | Add last-valid-token gather, shared scalar head semantics, pairwise Bradley-Terry/BCE loss forward/backward, checkpoint role metadata, and evaluation metrics. |
| PPO | Torch/Tile extension has a reference clipped loss | Native rollout, policy/ref/reward/value scheduling, selected log-probs and entropy, GAE/returns, clipped policy/value losses, KL, minibatch epochs, RNG, and resumable rollout state are absent. GPU kernels may be split into `sequence_logp_entropy`, `gae_returns`, and `ppo_clipped_loss`; the orchestration is equally required. |
| DFlash assistant training | Masked CE and target GEMMs are conceptual building blocks | Add verified block corruption/noise schedule, target-hidden capture, assistant attention backward, shared-head masked loss, and a freeze/distillation contract. The exact upstream training objective must be pinned from an authoritative recipe; do not infer it only from generation code. |

## 7. Native trainer implementation plan

- [ ] Add a dedicated `muse-glimmer` family and compiled target rather than
  routing it through `llama` by shape. Keep registry fields false initially.
- [ ] Lower the exact text graph into a versioned Native IR execution plan with
  a per-layer local/global schedule, explicit Q/K/V widths, four norm tensors,
  gate projection, untied head, output transform, tokenizer width, and context
  geometry.
- [ ] Build a complete, ordered parameter table and byte-size oracle for all 52
  decoder layers. Add the assistant and vision tables separately.
- [ ] Implement a tiny deterministic target forward first, then backward,
  clipping/optimizer, validation, save, and resume. No transition sampler or
  metadata-only checkpoint counts as completion.
- [ ] Add activation checkpointing/recomputation and sharded optimizer/checkpoint
  I/O before attempting production 30B training; prove the unfused math first.
- [ ] Add tensor/pipeline/data parallelism as a separate milestone if one GPU
  cannot hold the selected training mode. Record topology and world-size rules
  in the checkpoint contract rather than silently assuming single-device 30B
  training.
- [ ] Only promote `trainer_registered`, architecture persistence, and native
  forward flags after live CUDA train/save/resume/parity tests pass for the
  exact source-bound graph.

## 8. Post-training plan

### 8.0 Existing scaffolding that must be corrected first

The general Torch graphs are useful prototypes, but this audit found that they
are not a production post-training base for Glimmer yet:

- The standard attention and MLP builders do not pass `lora_role` for their
  ordinary Q/K/V/O/FFN projections, so selecting LoRA/QLoRA can currently
  produce zero adapter projections. Existing tests tolerate an empty adapter
  state; Glimmer tests must require the expected nonzero site count.
- Monolithic SwiGLU prevents separate `gate_proj`, `up_proj`, and `down_proj`
  checkpoint mapping, adapter selection, and per-site metadata.
- SFT, logits, and hidden-body helpers duplicate the generic homogeneous
  two-norm decoder rather than sharing the architecture-specific model stage.
- DPO builds independent policy forwards instead of one weight-shared policy.
  Add pair concatenation/splitting or a real shared-module facility, and ensure
  only the policy receives gradients.
- The reference-forward stage requires both graph and weight paths, while
  `FineTuneSpec` lacks the reference graph path and the current DPO builder
  leaves it empty. Add `ref_graph_path`/`reward_graph_path` or embed exact frozen
  inference graphs in the artifacts.
- Reward-model construction duplicates bodies/heads and does not attach the
  fine-tune prehook metadata needed to load the base checkpoint. Its head also
  needs a sequence mask and the final valid response token, not unconditionally
  `hidden[:, -1]`.
- PPO currently has separate logits/body policy copies, omits the real
  reference/reward/KL flow, and its rollout implementation supplies zero
  placeholder tokens, rewards, log-probs, and values. Treat it as a graph
  skeleton until real autoregressive rollouts exist.
- `adapter_only_save` is metadata today rather than a proved trained-artifact
  export. Per-site alpha/scaling and canonical base-tensor paths must survive
  save/load.
- Current QLoRA quantizes adapter-selected sites, not necessarily the complete
  frozen base expected for full-memory-saving QLoRA. Separate the set of base
  linears to quantize from the set receiving LoRA deltas.

These gaps apply even before Glimmer's custom body and native ABI are added.

### 8.1 Initialization and data contracts

- [ ] Initialize from the pinned BF16 target safetensors without changing norm
  conventions, projection orientation, untied head, or logit transform.
- [ ] Add conversation/SFT datasets with exact rendered chat tokens,
  assistant-only loss masks, optional packing boundaries, EOS policy, u32 IDs,
  and deterministic resume. Reject examples whose image/video placeholders do
  not match media inputs.
- [ ] Define `FineTuneSpec` capability validation per objective. The current
  graph-level fields are configuration syntax, not evidence that the native
  trainer supports the objective.
- [ ] Thread fine-tune and adapter kwargs through the preset factory centrally;
  individual builders must not discard them.

### 8.2 Full SFT and adapter tuning

- [ ] **Full SFT:** train every selected text-decoder parameter with masked CE;
  keep vision frozen by default until multimodal kernels land. Support explicit
  freeze/unfreeze sets and persist them.
- [ ] **LoRA:** add a Glimmer target profile for
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, attention `gate_proj`, and MLP
  `gate_proj`/`up_proj`/`down_proj`. Resolve the duplicate `gate_proj` names by
  full tensor path. Make LM head, embeddings, vision, and projector opt-in.
  The existing default of only Q/V is not an adequate complete Glimmer profile.
- [ ] Prove adapter-only save/load, merge/unmerge parity, rank/alpha/dropout,
  base-hash rejection, and continued training with optimizer moments.
- [ ] Define a strict native adapter artifact that binds the base artifact,
  graph/topology, tokenizer, exact tensor names/shapes, rank, alpha, effective
  scaling, dtype, per-tensor hashes, and optional optimizer state. The current
  `.pt` substring filtering and permissive load are not this contract.
- [ ] **QLoRA:** start with one documented NF4 group format and BF16 compute.
  Validate quantization error and memory; never load an unrelated quantized
  inference artifact as though it were the native NF4 base format.
- [ ] **K-Quant adapter tuning:** after packed forward and backward-input kernels
  pass, allow LoRA over a frozen 17GB or Dynamic base without expanding it to a
  dense master. Save adapter-only by default and pin the resolved profile,
  source file SHA, tensor-type table, tokenizer, and graph hashes.
- [ ] Keep K-Quant inference profiles separate from full-parameter training.
  Existing AdamW/native “quantized training” updates full FP32 masters and
  cannot mutate GGML packed codes. Full pretraining/full SFT must use BF16/FP32
  trainable masters unless a distinct quant-aware optimizer and re-quantization
  contract is designed and verified.
- [ ] Do not merge adapters into an official K-Quant file in place. A merge
  requires dequantizing to a dense checkpoint and either serving that result or
  running a pinned per-type re-quantizer with quality evaluation, producing a
  new lossy artifact and digest.
- [ ] Resolve `weight_precision="auto"` once when starting adapter training and
  persist the effective profile in all checkpoints/resume metadata. A resumed
  lineage must never switch base formats because current free VRAM changed.

### 8.3 Preference and reinforcement objectives

- [ ] **DPO first:** implement paired packing/masks, exact policy and frozen
  reference sequence log-probs, sigmoid/hinge/IPO variants already represented
  by `FineTuneSpec`, adapter-only optimization, and reference/base sharing that
  cannot accidentally mutate the reference.
- [ ] Permit K-Quant DPO only after packed LM-head sequence-logp/`dHidden` and
  projection `dX` paths pass: share one immutable authenticated packed base
  between policy/reference, apply distinct policy/reference adapter lineages as
  configured, and pin both resolved profiles/digests. Do not duplicate or
  dequantize two 17GB/Dynamic bases.
- [ ] **Reward model next:** add a scalar head over the last valid target hidden
  state, pairwise loss, calibration/evaluation, and a non-generative checkpoint
  role. Decide explicitly whether the target LM head remains present/frozen.
- [ ] **PPO last:** require a proved resident rollout engine, frozen reference,
  compatible reward model, trainable value head, GAE, KL accounting, rollout
  checkpointing, and deterministic recovery. A standalone clipped-loss kernel
  is not PPO support. Add per-token log-probabilities; the current
  sequence-level sum is insufficient.

### 8.4 DFlash after target post-training

The released assistant is bound behaviorally to the released target. Full SFT,
merged LoRA, or LM-head changes may invalidate its proposal distribution even
when target outputs remain valid.

- [ ] Record the exact target revision/adapter/merged-weight digest—or the
  reviewed finite compatibility allowlist for the stock quantized companion—in
  every assistant manifest and report acceptance rate by target profile and
  workload.
- [ ] Permit the stock assistant with an adapted target only behind parity and
  acceptance-quality tests; otherwise disable speculation and use target-only
  decoding.
- [ ] Add an assistant distillation/training pipeline that freezes the final
  target, captures layers `{1,13,25,37,49}`, constructs the verified diffusion
  corruption blocks, trains the five-layer assistant, and uses the target
  embedding/head exactly as inference will.
- [ ] Expose the configured target-layer residual inputs through named runtime
  taps or one hidden-tap collector. Avoid changing every decoder block's outer
  ports solely for DFlash, which would create variant-port compatibility debt.
- [ ] For adapter-based deployments, choose and document one of: train an
  assistant adapter bound to the target adapter, train a replacement assistant,
  or merge the target adapter and distill against the merged target.
- [ ] Multimodal instruction tuning and DFlash on multimodal prefixes remain
  blocked until the vision prefix, positions, target hidden capture, and cache
  semantics all pass parity.

## 9. Resident C++ inference and serving plan

- [ ] Add target and assistant artifact inspectors with strict format, tensor,
  tokenizer, source-revision, ABI, device, and target-binding checks.
- [ ] Add an exact C++ CPU target runner as a portable resident oracle with the
  asymmetric projections, scheduled local/global attention, hybrid cache,
  attention gate, sandwich norms, and output transform. Do not stretch the
  canonical LLaMA runner, whose geometry and layer semantics are different.
- [ ] Add a whole-model GPU resident runner: stream/load the selected BF16 or
  packed K-Quant weights once, prefill prompts, maintain all 52 layer caches
  according to the pattern—39 local ring caches and 13 global paged caches—and
  decode without CPU projection or full-weight-dequantization fallbacks.
- [ ] First prove target-only greedy generation against the pinned reference at
  short and long contexts, then sampling and stop handling.
- [ ] Bind artifact generation defaults, including all declared EOS IDs, BOS,
  padding, temperature, top-p, and top-k. Do not inherit generic NeuralFn
  defaults when the artifact declares different values.
- [ ] Replace the one-token-immediate-commit session boundary with a
  transactional block API while keeping `decode_one` as a target-only wrapper.
- [ ] Keep resident ABI 1 compatible and expose speculation through an optional
  feature ABI (`load_companion` plus atomic `speculative_step`) rather than
  breaking every existing target-only binding. The step returns committed token
  IDs and proposal/acceptance counters; Python streams those committed IDs one
  at a time.
- [ ] Load the assistant, share target embedding/head safely, run proposal and
  verification, commit the accepted prefix, and update both caches. Prove the
  paired algorithm against the CPU target/assistant oracle before promoting
  the all-CUDA feature ABI.
- [ ] Expose explicit request controls such as
  `speculative=off|auto|required`,
  assistant path, maximum proposal block, and diagnostics. `auto` must require
  an exact compatible assistant and may fall back only before request/session
  mutation; `required` must fail rather than silently downgrade. Initially
  reject batch size greater than one and unproved constrained-output/prefix-COW
  combinations.
- [ ] Enforce maximum-new-token, EOS, stop, and cancellation boundaries inside
  the atomic step so a verified block cannot emit tokens after a terminal
  condition or leave half-committed caches.
- [ ] Return proposed/accepted/rejected token counts, acceptance histogram,
  target/assistant timing, cache bytes, and effective backend without exposing
  internal hidden states.
- [ ] Add image/video content only after the API, preprocessing, vision runner,
  and placeholder fusion are exact. Preserve current early rejection until
  then.

### 9.1 Weight-precision parameter and automatic VRAM policy

Use **weight precision** for the public name because this selects a resident
checkpoint/kernel profile, not activation compute dtype, AMP, or KV-cache
compression. Add the same load-time control to one-shot inference and server
startup:

```text
--weight-precision {auto,bf16,k-quant-dynamic,k-quant-17gb}
```

The default is `auto`. Add a typed Python surface rather than overloading
`KVCacheConfig`, whose CUDA fields currently belong to the optional Tile cache:

```python
NativeModelLoadConfig(
    weight_precision="auto",
    cuda_device=0,
    cuda_runtime_lib=None,
)
```

Thread it through `NativeInferenceModel.load(..., load_config=...)`,
`NativeArtifactCLIConfig.model_load`, and `NativeServeConfig.model_load`. Reuse
the existing CLI `--cuda-device`/`--cuda-runtime-lib` values as inputs to this
model-level config. Refactor argument routing so those generic values always
reach whole-model load configuration, but reach `KVCacheConfig` only when
`turboquant_attention_backend="tile-cuda"`; the current KV validator must not
reject a model-only device selection. Initially require model weights and a
Tile-CUDA cache to use the same device and runtime library; cross-device caches
need an explicit transfer/ownership contract. A server owns one resident weight
set, so weight precision is a startup parameter, not a per-generation request
field.

`auto` is a VRAM policy only for whole-model CUDA. For the portable CPU runner,
an omitted/default `auto` resolves to the manifest's authenticated
`primary_checkpoint_variant` if its CPU kernel profile is supported; it does
not inspect system RAM or walk the CUDA fidelity order. A caller that wants a
different CPU profile selects it explicitly, subject to CPU kernel capability
and host-memory checks. Add a separate RAM-aware CPU policy later if needed.
Before K-Quant CPU kernels land, only the proved BF16 CPU primary may load; no
CUDA or Tile-cache behavior is implied.

The deterministic `auto` algorithm is:

1. Structurally validate the manifest, primary/variant invariant, fixed profile
   IDs, memory-profile version, and locally declared `checkpoint_variants`; do
   not hash every 17–60 GB alternative payload yet.
2. Resolve the CUDA device, call `cudaSetDevice`, then query current free and
   total bytes with `cudaMemGetInfo` immediately before creating a resident
   handle. Every whole-model Glimmer CUDA load—including explicit and
   single-variant loads—fails closed if the probe cannot establish fit; this
   does not change legacy non-Glimmer CPU loading.
3. For one-shot inference, use the resolved context, batch/session count, cache
   format, required speculation, and modalities. For a server, use configured
   worst-case context, enabled features, and `session_limit`, or install a
   pre-mutation admission controller that reserves and enforces the same bytes
   for every accepted request. Vision-enabled plans must also cap image count,
   video count/frames, resolution, derived visual-token count, and per-session
   preprocessing/attention scratch; reject requests that exceed the planned
   envelope before mutating a session.
4. Compute, in bytes,
   `runtime = resident_weight_bytes + enabled_companion_resident_bytes +
   peak_load_staging_bytes + max_workspace_bytes + kv_cache_bytes(config) +
   graph_and_session_bytes(config)`, then `reserve = max(2 * 2^30,
   ceil(total_vram_bytes / 10))` and `required_free_vram_bytes = runtime +
   reserve`. Every component excludes the reserve, so it is added exactly once.
   Parsed descriptors and the binding's memory oracle must reproduce the
   manifest components before allocation; never derive them from a filename or
   on-disk byte count. When DFlash is enabled, the cache/session terms include
   target and assistant KV, hidden taps, tentative 16-row transactional state,
   full proposal/verification distributions or bounded equivalents, acceptance
   scratch, and rollback metadata for every reserved session—not only assistant
   weight bytes.
5. Walk the compatible, available candidates in fidelity order:
   `bf16`, `k-quant-dynamic`, `k-quant-17gb`. Select the first whose tested
   `minimum_total_vram_bytes` and exact free-byte equation both fit. Seed and
   benchmark the measured profiles on Meta's 32-GB Dynamic and 24-GB 17GB
   target classes, but store the resulting threshold as an exact byte count
   tied to the kernel/device/config fingerprint; the marketing labels are not
   themselves comparison constants. BF16 likewise uses a measured envelope
   rather than assuming a nominal 64-GB device always fits. Below the proved
   17GB envelope, fail unless a future separately proved sharding/offload
   profile is explicitly selected.
6. Authenticate the selected main payload and every enabled companion/tokenizer
   artifact: contained path, exact size, SHA, tensor table, and compatibility.
   Recompute its memory equation from authenticated descriptors. A declared but
   corrupt preferred candidate fails the load rather than causing a silent
   downgrade; only a genuinely absent or binding-declared-unsupported candidate
   may be skipped by `auto` with a recorded reason.
7. Build an authenticated effective-manifest copy whose existing `checkpoint`
   object is replaced by the selected variant and whose selection proof is
   attached, then call a new optional `load_model_with_options` feature ABI.
   The legacy ABI-1 `load_model` and on-disk manifest remain unchanged for old
   artifacts. Selection is complete before any model/session mutation.
8. If no candidate fits, fail with free/total bytes and a per-candidate
   breakdown of weights, companions, cache, workspace, reserve, missing file,
   or missing kernel. Do not silently shrink context, disable required
   vision/DFlash, offload layers to CPU, download artifacts, or retry a lower
   profile after an allocation failure.

An explicit value is a strict pin: unavailable, incompatible, or insufficient
memory is an error and never causes downgrade. Supplying the exact path of one
contained variant creates a `path-pin`; a conflicting explicit value is an
error. For a future tensor-parallel placement, apply the same equation to every
device's declared shard plan and require all devices to fit—never add free bytes
across GPUs and call the sum usable. Report per-device rather than scalar memory
stats. Until that schema exists, reject multi-device `auto`. Until a true
whole-model CUDA loader exists, do not advertise this VRAM policy on the current
CPU-resident/Tile-cache hybrid path.

Resolve optional `speculative=auto` after the quality-first weight choice: enable
the compatible assistant only if it also fits, otherwise retain the selected
target precision and report target-only fallback. It must not silently choose a
lower-quality target merely to preserve optional speculation. In contrast,
`speculative=required` participates in the initial budget, so weight `auto` may
select a lower fitting target profile to honor it; an explicit weight profile
that cannot fit the required assistant fails.

Expose the decision through `--native-info`, model stats, and server startup
metadata:

- `requested_weight_precision` and `effective_weight_precision`;
- `weight_precision_selection` (`auto-vram`, `explicit`, `path-pin`, or
  `legacy-single`);
- selected artifact SHA and kernel profile;
- `cuda_free_bytes_at_selection`, `cuda_total_bytes_at_selection`,
  `required_free_vram_bytes`, and `vram_reserve_bytes`;
- a structured selection/rejection reason for every candidate considered.

Legacy one-checkpoint artifacts remain backward-compatible under `auto` and do
not acquire a fabricated K-Quant capability. Additive manifest parsing must
round-trip the new optional fields in upgraded readers/writers without changing
the existing `checkpoint` object's meaning.

- [ ] Update [`README.md`](../README.md), [`CHANGELOG.md`](../CHANGELOG.md),
  [Native CLI Inference](native-cli-inference.md), the matching Python SDK
  native-inference/config pages, native serving/operations docs, and CLI/Python
  agent skills when this load contract is implemented. Document that weight
  precision, activation compute dtype, and KV-cache precision are independent.

## 10. Verification and release gates

### 10.1 Reference and template gates

- [ ] Store tiny deterministic main/assistant/vision configs and inputs with an
  immutable upstream source manifest.
- [ ] Compare every main sublayer and final logits, including local vs global,
  NoPE vs RoPE, QK normalization/scaling, attention gate, four norm locations,
  embedding norm, output multiplier, and softcap.
- [ ] Compare assistant context projection, raw embedding path, attention masks,
  cache positions, proposal logits, and candidate IDs.
- [ ] Add strict checkpoint-mapping tests for every tensor, centered-norm
  parameter conversion, Q/K layout/permutation, wrong shape/dtype, missing,
  unexpected, duplicate, truncated, and hash-mismatched safetensors shards.
- [ ] Test all existing presets and every ordered old/new preset pair to catch
  variant-library cross-contamination.

### 10.2 Kernel and native-training gates

- [ ] Unit-test each proposed raw ABI against FP32/Torch oracles across tail
  widths, BF16 tolerances, invalid IDs, local-window boundaries, cache wrap,
  and 131,072-position arithmetic.
- [ ] Add a direct negative test proving every legacy raw/generic SDPA entry
  rejects `seq_k > 1024`; never allow a dense call to return plausible output
  after ignoring keys beyond its fixed tile.
- [ ] Compare one full target block forward/backward and parameter gradients,
  then a multi-layer local/local/local/global slice.
- [ ] Run one real optimizer step, save, inspect, reload, and prove continued
  loss/logit/update parity. Repeat for full SFT and each adapter mode before
  advertising it.
- [ ] Prove the 202,048-vocabulary loss against an independent formula and
  verify that padded/chunked columns cannot affect probability normalization.
- [ ] Exercise token IDs `65535`, `65536`, `201818`, and `202047`, plus
  out-of-range, wrong-endian, corrupt-header, and old-uint16 compatibility cases.
- [ ] Run live GPU memory-safety/sanitizer tests and long-context tests that
  demonstrate no dense quadratic allocation.

### 10.3 K-Quant and VRAM-selection gates

- [ ] Add tiny synthetic GGUF v3 fixtures and reject unknown tensor types,
  wrong architecture/version/endian, bad dimensions, duplicate names,
  misalignment, overlap, truncation, path traversal, trailing/partial payloads,
  and size/SHA mismatch.
- [ ] Assert canonical main/companion filenames, revisions, hashes, header
  architecture/version/file-type fields, tensor counts/inventories, both Q5_K
  output heads, DFlash one-based-to-zero-based layer normalization, and mmproj
  BF16 dispatch. Ensure retained lowercase legacy files are never selected as
  canonical aliases.
- [ ] Compare every Q4_K/Q5_K/Q6_K block decoder byte-for-byte/numerically with
  a pinned independent GGUF reference, including scale/min decoding, signed
  lanes, multiple blocks, and invalid/tail rows.
- [ ] Compare packed CPU and CUDA matvec/GEMM against explicit dequantization for
  all Glimmer asymmetric projection shapes and tail tiles. Exercise each exact
  per-tensor mixture, F32 passthrough, the 202,048-column head, multirow target
  verification, and the quantized DFlash companion.
- [ ] Run target logits, greedy tokens, sampled distributions, local/global
  attention boundaries, and long-context cache checks separately for canonical
  17GB and Dynamic against the pinned reference runtime. Do not use agreement
  between two NeuralFn implementations as the only oracle.
- [ ] Prove with allocation/device counters that neither loader nor forward
  materializes a full dense weight/model, that reported resident and peak-load
  bytes match measurement, and that whole-model CUDA performs zero CPU model
  compute. Treat a bounded tile-dequant workspace separately from forbidden
  full-weight expansion.
- [ ] Exercise the pinned llama.cpp cross-check with `--jinja`, and verify EOS
  IDs 200001/200008 terminate while the EOM message boundary does not.
- [ ] Unit-test `auto` with mocked free/total VRAM across exact boundary bytes,
  reserve rounding, changed context, cache format, DFlash/vision companions,
  missing variants, missing kernels, explicit/single-variant probe failure,
  concurrent-session reservations, maximum image/video/visual-token admission,
  multi-GPU rejection and future per-device plans, and no-fit errors. Assert
  the order BF16 > Dynamic > 17GB only among candidates that satisfy both the
  hardware envelope and exact budget.
- [ ] Test the joint policy: optional speculation cannot lower target precision,
  required speculation is included before profile choice, and an explicit
  profile plus required assistant fails when their combined budget does not fit.
  Assert the exact per-session byte delta from speculation `off` to `required`,
  including assistant KV, taps, tentative block, verify/proposal, and rollback
  buffers.
- [ ] Verify explicit profile selection never downgrades; exact-file pinning and
  conflicting flags fail; `auto` makes no implicit download/feature/context
  changes; a present-but-corrupt preferred candidate fails rather than stepping
  down; and a load-time OOM is surfaced without retrying another profile.
- [ ] Verify startup hashes only the selected main among alternative mains, but
  also every enabled companion and external tokenizer/template artifact. Check
  authenticated metadata against the binding's memory oracle, surface the
  choice and byte budget in CLI/server stats, preserve old one-checkpoint
  behavior, and round-trip all new optional manifest fields.
- [ ] Add constrained-memory live allocation/load smoke tests on representative
  24-GB and 32-GB-class GPUs in addition to mocked byte boundaries; assert 17GB
  and Dynamic selection respectively with the tested context/companion envelope.
- [ ] Test the public parameter contract directly: omission resolves to `auto`;
  all four values round-trip through one-shot CLI, serve CLI,
  `NativeArtifactCLIConfig`, `NativeServeConfig`, `NativeModelLoadConfig`, and
  `NativeInferenceModel.load`; CPU `auto` uses the authenticated primary; a
  model-only CUDA device bypasses Tile-cache-only validation; and request bodies
  cannot override the server's startup precision.

### 10.4 Speculative-decoding gates

- [ ] For zero, partial, and full acceptance, compare emitted tokens, target
  logits, selected target hidden states, target cache, assistant cache, and
  logical position against target-only replay.
- [ ] Greedy DFlash must be token-for-token identical to greedy target-only
  generation over a fixed prompt corpus, including stop tokens and context
  boundaries.
- [ ] Sampled DFlash must pass seeded replay tests and statistical checks of the
  target distribution, including `q=0`, complete rejection, top-k/top-p, and
  logits processors.
- [ ] Verify assistant/target hash mismatch, tokenizer mismatch, block mismatch,
  stale adapter, cache overflow, and unsupported multimodal inputs fail closed.
- [ ] Benchmark target-only and DFlash on the same hardware and prompts. Report
  time-to-first-token, decode tokens/s, acceptance rate, memory, and p50/p95;
  do not promise the upstream speedup figure before measuring NeuralFn's path.

### 10.5 Post-training gates

- [ ] Full and adapter SFT: overfit a tiny batch, verify loss-mask isolation,
  frozen-parameter immutability, save/resume, merge parity, and eval generation.
- [ ] Assert exact nonzero LoRA/QLoRA node counts for every configured Glimmer
  target path; an empty adapter artifact must fail the test.
- [ ] QLoRA: quantize/load deterministically, verify only adapter gradients and
  optimizer state, and compare merged/dequantized outputs within declared
  tolerance.
- [ ] K-Quant adapter tuning: compare packed-base forward/`dX` and A/B gradients
  with a dense oracle, prove the base bytes remain identical, and reject resume
  or load against a different resolved profile, GGUF digest, or tensor-type
  table. Verify `auto` is recorded once and never re-resolved on resume.
- [ ] DPO/reward/PPO: compare losses and gradients to independent formulas,
  prove frozen reference/reward immutability, and resume mid-objective state.
- [ ] DFlash assistant training: overfit a tiny corruption set, reload against
  the exact target, improve proposal loss/acceptance, and preserve target-only
  output correctness.

## 11. Recommended milestone order

1. **Freeze the oracle:** pin model, assistant, tokenizer, chat template,
   Transformers revisions, both canonical K-Quant mains, and companions;
   generate tiny golden fixtures and tensor-type inventories.
2. **Ship the exact text template:** add `muse_glimmer` with no native or
   multimodal capability claim; pass all preset/variant tests.
3. **Land formats:** u32 datasets, tokenizer/chat support, safetensors import,
   the strict Glimmer GGUF reader/packed descriptors, multi-variant manifests,
   full target/assistant/mmproj tensor manifests, and checkpoint round trips.
4. **Implement target-only resident CPU:** establish the exact C++ target
   oracle for BF16 and each packed K type, prefill/decode, hybrid cache,
   sampling, CLI, and serving.
5. **Implement target-only resident CUDA:** land shared architecture kernels,
   BF16 and packed Q4_K/Q5_K/Q6_K projection/head paths, direct resident loads,
   and exact memory accounting. Prove 17GB and Dynamic independently with zero
   CPU model-compute/full-dequant fallback.
6. **Enable load-time precision policy:** add `--weight-precision`, the Python
   load config, free-VRAM probing, strict explicit selection, default `auto`,
   decision telemetry, and mocked/live boundary tests. Keep every capability
   bit false until its exact profile passes.
7. **Add stock DFlash inference:** strict BF16/packed assistant import, CPU proposal/verify
   oracle, transactional caches, greedy then sampled acceptance, and finally
   the all-CUDA feature ABI.
8. **Implement native pretraining:** add every backward kernel, the exact
   graph-bound loop, optimizer, validation, save/resume, and production
   distributed-memory work.
9. **Add SFT and adapters:** full SFT, LoRA, NF4 QLoRA, then frozen-base K-Quant
   adapter tuning, with target-bound artifacts and capability flags per mode.
10. **Add assistant post-training:** distill/retrain an assistant against each
   post-trained target lineage; gate stale assistants.
11. **Add DPO, reward modeling, then PPO:** promote each independently.
12. **Add vision and multimodal tuning/inference:** only then advertise full
    `MuseGlimmerForConditionalGeneration` support.

Completion requires all applicable documentation and changelog updates in the
same implementation changes. Until the corresponding milestone's tests and
capability proof pass, user-facing surfaces should continue to report Glimmer
as unsupported rather than route it through a superficially similar LLaMA or
Gemma preset.
