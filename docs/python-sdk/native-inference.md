# Resident Native Inference SDK

`neuralfn.native_inference` defines the additive, dependency-light contract
between Native Execution IR artifacts and an in-process resident model binding.
The Python layer does not spawn a native CLI and does not reinterpret a graph.

## Availability and proof boundary

The contract and its first compiled engine foundation are implemented. Build
the default `neuralfn._native_inference` extension with:

```bash
bash tools/build_native_inference_binding.sh
```

That extension loads native dense GPT-family bf16 v5, canonical native-family
LLaMA float32, exact graph-bound standard-MoE float32, or strict Muse Glimmer
BF16/K-Quant checkpoint weights once, keeps them immutable, and runs the
corresponding real autoregressive forward/sampling math in C++.
It provides isolated session token/RNG/cancellation state and all ABI v1
lifecycle operations without spawning a process. Legacy reviewed families
report `cpu-reference-resident`; a strict Muse Glimmer load may instead report
its whole-model native-CUDA text backend. `auto` and `full` use preallocated,
lossless per-layer K/V storage plus final-hidden history; `off` keeps the
full-prefix recomputation path as a parity oracle. The reviewed `gpt2`,
`gpt2_megakernel`, `gpt2_moa`, `gpt2_zloss`, `gpt2_qknorm`, `gpt2_stable`, and
`gpt2_softcap` dense-v5 artifacts are supported. QK-norm applies per-head RMS
normalization with `eps=1e-6` before attention and cache encoding; softcap
applies `cap*tanh(logits/cap)` after the tied output projection. Artifacts with
an even head dimension also prove cache ABI v1: explicit `turboquant` uses a
native packed K/V cache with either the `mse-3.5` or `qjl-3.5` profile and
attends over compressed historical rows directly.

Exact `gpt2_moa` uses the same resident/cache engine only after migration of
`model_XXXXXXXX.moa.json` alongside its named dense-v5 `.bin` and empty
`DONE_XXXXXXXX`. The v1 metadata must bind the byte-identical source graph,
canonical `[gelu, relu, silu, relu2]` candidates, one selected activation, and
a positive interval. Inference fixes every MLP to that recorded activation;
prefill/decode, recompute/full cache, and supported-even-head CPU TurboQuant do
not rerun the training-time candidate probe.

The canonical eager `llama` and graph-equivalent compile `llama_fast` paths
accept only a migrated inference-checkpoint v2 artifact whose `model.f32`
tensor image exactly matches the reviewed
RMSNorm/RoPE/GQA/SwiGLU/untied-head contract. It uses half-split
production-sign RoPE, maps query heads to grouped K/V heads, keeps
preallocated lossless K/V and final-hidden history per session, and supports
`off`, `auto`, and `full`. Explicit TurboQuant is unsupported and fails closed.
The raw-token CLI and SDK paths are proved; text CLI and HTTP serving remain
conditional on supported tokenizer/presentation metadata.
Both source profiles normalize to checkpoint/runtime ABI identity `llama`; the
Native IR manifest retains the source runtime and graph SHA-256.

The exact standard-MoE `moe`, `mixllama`, and compile-runtime `mixllama_fast`
profiles accept only a migrated
`neuralfn.native_family_standard_moe.inference_checkpoint` v1 bundle. The
inspector and C++ loader require the source graph SHA, DONE marker, exact
float32 file/tensor hashes, floating expert width, unaligned-width sentinel,
router coefficient, softmax top-k-renormalized routing, no shared experts, and
the ordered RMSNorm/GQA/router/expert tensor table. The resident engine executes
routed SwiGLU experts and supports `off`, `auto`, and `full`; explicit
TurboQuant fails closed. Raw-token SDK/CLI inference is proved. Text/HTTP use
still depends on artifact tokenizer and chat metadata.

Muse Glimmer uses a separate exact resident family rather than the canonical
LLaMA adapter. A runnable bundle is created by
`nfn migrate muse-glimmer-to-native` for pinned BF16 safetensors or
`nfn migrate muse-glimmer-gguf-to-native` for the canonical official
K-Quant-Dynamic/K-Quant-17GB files. The C++ CPU and CUDA text runners preserve
the 6,656 residual width, 4,096 Q/gate width, 256 K/V width, explicit head
dimension 128, three-local/one-global RoPE/NoPE schedule, four centered norms,
untied head, and multiplier-before-softcap output. Local layers use ring caches
at 2,048 rows while global layers retain the full configured context.

An authenticated `dflash` companion adds BF16 or packed five-layer speculative
decoding on CPU or CUDA. It consumes target layers `[1,13,25,37,49]`, proposes
15 tokens in a 16-position block, verifies with the target in parallel, and
atomically commits/crops both caches. Greedy and lossless sampled modes preserve
the target distribution. The load policy is `off`, `auto`, or `required`;
`required` fails before mutation when the companion, cache mode, device
backend, or digest binding is unavailable.

When the sidecar exports deterministic row argmax, ordinary greedy target
decoding selects the current 202,048-way row on device. Greedy DFlash likewise
keeps assistant and target selection on device and transfers only token
IDs/selected values. Equal maxima choose the lowest token ID. Sampled
generation and sampled speculation continue to transfer/retain full rows;
lossless speculative acceptance and positive-residual sampling require the
complete p and q distributions.

The accepted strict sidecar additionally groups one-row packed projections and
uses an optional dual-RMS→Q8→MMVQ handoff. The paired symbols
`nfn_native_tile_glimmer_dual_rms_add_capture_mmvq_q8_float32_v1` and
`nfn_native_tile_k_quant_mmvq_multi_linear_prequantized_float32_v1` must both
be present or both be absent; an incomplete ABI fails during model load. The
handoff still writes all ordinary FP32 norm/residual outputs and is used only
for one-row target decode. Multirow DFlash MMVQ instead hoists decoded packed
weights, reuses prequantized activations, and can resolve the optional
`nfn_native_tile_glimmer_dual_rms_add_capture_cooperative_batch_float32_v1`
and `nfn_native_tile_dflash_block_attention_short_split_float32_v1` symbols.
The cooperative norm is selected only for supported F32 affine descriptors;
BF16 norms retain the ordinary exact kernel. Set
`NFN_GLIMMER_COOPERATIVE_BATCH_RMS=0`,
`NFN_GLIMMER_SHORT_ATTENTION_SPLIT=0`, or
`NFN_GLIMMER_VERIFIER_PROJECTION_OVERLAP=0` only for matched development
bisections. These are execution details, not new checkpoint encodings.

Presentation code must not treat a decoded Glimmer continuation as plain
assistant text. `neuralfn.native_chat.parse_native_assistant_response()`
returns a `NativeAssistantResponse` with `visible_text`, separately held
`reasoning_text`, raw text, and channel-completion flags. For the ATEM renderer
it exposes only `to=user` text as display/transcript-safe; malformed control
fragments and private-only truncation fail closed. The process-local CLI uses
that parser before appending history.

`neuralfn.native_cli.run_native_artifact_cli()` accepts an optional
`interactive_ui` implementing `NativeArtifactCLIUI` (`handle`, `read_line`,
and `progress`). UI events carry ready/model state, completed turns with
separate prefill/decode timing, warnings, commands, and live statistics. This
keeps presentation optional: noninteractive callers and injected test UIs use
the same resident lifecycle without importing Rich, while the installed
`nfn` command supplies `RichNativeInferenceUI` for a TTY.

Full-BF16 artifacts expose embedded CPU or whole-model CUDA vision for images
and decoded videos. Official packed bundles can attach `mmproj` for CPU or CUDA
still images. The packed temporal projection cannot represent two distinct
frames, so `video=false` remains exact for mmproj. CUDA vision requires the
versioned preparation/LayerNorm/2-D-RoPE-attention/pixel-shuffle ABI; weights
are attached atomically before sessions and request-time failure never falls
back to CPU.

This remains a topology-specific foundation rather than an all-family runtime.
`nfn migrate graph-to-native` accepts either a graph-compatible native dense-v5
`.bin`, strict source-bound dense-MoA metadata JSON, canonical LLaMA
inference-checkpoint v2 metadata JSON, or exact
standard-MoE inference-checkpoint v1 metadata JSON. Dense input is copied as
owner-only `model.bin`; validated family sidecars are copied as owner-only
`model.f32`. All record exact size/SHA-256 and tensor
layout and stamp resident ABI v1 plus resident/lossless-cache/lean-serving
model capabilities. Python verifies the contained file before binding load.
MoA always requires the supplied graph bytes to match its source digest. When
LLaMA or standard-MoE metadata declares `training.source_graph`, migration requires its
verified source SHA-256 to equal the graph supplied to
`migrate_graph_to_native()` rather than accepting geometry alone;
the LLaMA C++ loader also hashes the exact float buffer it loaded and probes
EOF before accepting it. Graph-only and generic legacy `.pt` migrations remain
deliberately unbound; their `weights.bin` tensor bundle is not a resident
checkpoint.

The focused tests use both an injected stateful binding double for coordination
semantics and the compiled extension over a real tiny dense checkpoint. The
compiled proof covers one immutable load, cached/recompute whole-logit parity
before and after decode/truncate/reset, two isolated sessions, exact cache-byte
accounting, lifecycle/cancellation, exact-zero temperature, migration-to-load,
the serving lifecycle, no subprocess spawn, portable/native TurboQuant packed
bytes and numerical agreement, both compressed profiles, deterministic greedy
decode, exact compressed-byte telemetry, independent QK/softcap formula parity,
cache/recompute parity for those variants, and migrated-artifact loading. A
separate canonical LLaMA fixture uses an independent float-boundary oracle to
prove RMSNorm, RoPE, GQA attention, gate-first SwiGLU, untied-head logits,
whole-logit `full`/`off` parity, decode/truncate/refill/reset, interleaved
sessions, cancellation, exact-prefix reuse, raw-token CLI dispatch, and
fail-closed checkpoint/topology tampering. A standard-MoE fixture migrates the
exact source graph and strict checkpoint, imports every native tensor into the
compiled Torch graph, compares all prefix logits with the resident engine,
independently reconstructs cross-entropy plus the configured all-expert
router loss, verifies router gradients, and exercises SDK prefix reuse and
ordinary raw-token CLI inference. That fixture by itself is not evidence of
live CUDA/Tile performance or TurboQuant agreement, sparse-window or other non-dense-family
coverage, LLaMA TurboQuant, LLaMA text-tokenizer integration, or full OpenAI
Responses/tool compatibility. Separately, a bounded live RTX 5090 acceptance
ran one real graph-authored CUDA optimizer step for exact canonical LLaMA and
standard MoE, migrated the graph-bound production checkpoints, loaded them
through a freshly rebuilt resident extension, and generated raw tokens with the
ordinary full-cache CLI. That live result proves the train-to-resident handoff
for those two exact profile clusters; it is not a CUDA performance or all-family
result.

`NativeInferenceModel.load()` succeeds only when all of these agree:

- the artifact uses Native Execution Manifest schema/version 1;
- artifact capabilities prove native and resident inference;
- `kernel_abi.resident_inference` is version 1 with status `ready`;
- a TurboQuant-capable artifact additionally declares
  `kernel_abi.turboquant_cache` version 1 with status `ready`;
- effective session-prefix COW additionally requires artifact capability
  `session_prefix_cow=true`, `kernel_abi.session_prefix_cow` integer version 1,
  status `ready`, operation `fork_session`, the exact checkpoint-format profile
  (`dense-full-cache-kv-final-hidden-v1`,
  `llama-full-cache-gqa-kv-final-hidden-v1`, or
  `standard-moe-full-cache-gqa-kv-final-hidden-v1`), and matching binding
  boolean/profile inventory plus callable;
- effective dense CPU TurboQuant prefix COW separately requires artifact
  capability `session_prefix_cow_cpu_turboquant=true`,
  `kernel_abi.session_prefix_cow_cpu_turboquant` integer version 1, status
  `ready`, operation `fork_session`, exact profile
  `dense-cpu-turboquant-mse-qjl-packed-kv-final-hidden-v1`, and backend
  `cpu-reference-packed`, plus the matching binding boolean, exact profile in
  the binding's prefix-COW inventory, and callable `fork_session`;
- the imported or injected binding exposes every required in-process method;
- the binding reports resident ABI version 1 and matching capabilities; and
- effective structured output additionally requires artifact capability plus
  `kernel_abi.structured_output` integer version 1, status `ready`, profile
  `json-schema-ascii-byte-greedy-v1`, token selection
  `current_logits_exact_prefill`, binding boolean
  `current_logits_exact_prefill=true`, and a callable read-only
  `current_logits`; effective function tools additionally require artifact
  `capabilities.function_tools=true`, effective structured-output support, and
  integer version 1/status `ready` profile `responses-forced-function-call-v1` with
  linked `structured_output_profile=json-schema-ascii-byte-greedy-v1`; and
- either explicit dense resident fields plus canonical QK-norm/softcap edge
  chains, the exact MoA source/selection/DONE contract, the exact canonical
  LLaMA v2 geometry/semantics/tensor/topology
  contract, or the exact standard-MoE v1 source/geometry/semantics/tensor
  contract passes independent manifest and binding validation; and
- the requested cache mode is jointly supported by artifact and binding.

There is no subprocess fallback. Explicit TurboQuant never falls back to a
full cache.

## Public types

All types are lazy top-level exports:

```python
from neuralfn import (
    GenerationConfig,
    GenerationEvent,
    GenerationResult,
    KVCacheConfig,
    NativeModelLoadConfig,
    NativeInferenceCapabilities,
    NativeInferenceCancelledError,
    NativeInferenceCapabilityError,
    NativeInferenceClosedError,
    NativeInferenceError,
    NativeInferenceModel,
    NativeInferenceSession,
)
```

The configuration, capability, event, and result types are frozen dataclasses.
Model/session objects own lifecycle and synchronized mutable state.
`NativeInferenceCapabilities.session_prefix_cow_cpu_turboquant` is the
effective boolean for the exact dense CPU packed-TurboQuant fork profile; it is
independent of the full-cache `session_prefix_cow` boolean.

The exact prefix-COW profile names are public module-level constants (they are
not lazy top-level `neuralfn` exports):

```python
from neuralfn.native_inference import (
    CPU_TURBOQUANT_SESSION_PREFIX_COW_PROFILE,
    LLAMA_SESSION_PREFIX_COW_PROFILE,
    SESSION_PREFIX_COW_PROFILE,
    STANDARD_MOE_SESSION_PREFIX_COW_PROFILE,
)
```

- `SESSION_PREFIX_COW_PROFILE` is the dense full-cache profile
  `dense-full-cache-kv-final-hidden-v1`.
- `CPU_TURBOQUANT_SESSION_PREFIX_COW_PROFILE` is the dense CPU packed MSE/QJL
  profile `dense-cpu-turboquant-mse-qjl-packed-kv-final-hidden-v1`.
- `LLAMA_SESSION_PREFIX_COW_PROFILE` is the canonical LLaMA full-cache GQA
  profile `llama-full-cache-gqa-kv-final-hidden-v1`.
- `STANDARD_MOE_SESSION_PREFIX_COW_PROFILE` is the standard-MoE full-cache GQA
  profile `standard-moe-full-cache-gqa-kv-final-hidden-v1`.

These constants let callers compare capability/profile inventory values without
duplicating wire strings; importing one does not enable the corresponding
artifact or binding capability.

## Model and session lifecycle

The intended use is:

```python
with NativeInferenceModel.load(
    "artifacts/model-native",
    kv_cache=KVCacheConfig(mode="auto"),
) as model:
    session = model.create_session(seed=7)
    session.prefill([101, 102, 103])
    result = session.decode(
        GenerationConfig(max_new_tokens=32, temperature=0.8),
        on_token=lambda event: print(event.token_id),
    )
    print(result.text)
```

For a multi-variant Muse Glimmer bundle, model-weight selection is separate
from KV-cache configuration:

```python
from neuralfn import GenerationConfig, KVCacheConfig, NativeInferenceModel
from neuralfn.native_inference import NativeModelLoadConfig

load = NativeModelLoadConfig(
    runtime="native-cuda",
    weight_precision="auto",  # bf16 > Dynamic > 17GB among fitting profiles
    cuda_device=0,
    tile_ops_lib="/absolute/path/libnfn_native_train_tile_ops.so",
    context_tokens=32_768,
    session_count=1,
    companion_checkpoints=("dflash",),
    speculative_decoding="auto",
)
with NativeInferenceModel.load(
    "artifacts/glimmer-kquant",
    load_config=load,
    kv_cache=KVCacheConfig(mode="full"),
) as model:
    with model.create_session(seed=7) as session:
        session.prefill([200000, 200022, 1556, 200023])
        result = session.decode(GenerationConfig(max_new_tokens=32))
        print(
            result.speculative_proposed_tokens,
            result.speculative_accepted_tokens,
            result.speculative_rejected_tokens,
        )
```

`NativeModelLoadConfig.weight_precision` accepts `auto`, `bf16`,
`k-quant-dynamic`, or `k-quant-17gb`. On CUDA, `auto` queries current free and
total VRAM before model allocation and budgets target/companion weights,
staging, workspace, hybrid target/assistant caches, configured context and
session count, and reserve. It chooses quality-first only among authenticated,
available, binding-supported profiles that fit. An explicit value is a strict
pin and never downgrades; a 32-GB device may explicitly load
`k-quant-17gb` when the measured byte budget fits. CPU `auto` selects the
authenticated primary and does not reuse the CUDA VRAM policy. Weight
precision, activation storage, and `KVCacheConfig` are independent.

Packed Glimmer CUDA statistics additionally expose
`cuda_q8_activation_quantizations`, `cuda_q8_packed_linears`,
`cuda_device_argmax_calls`, and `cuda_device_argmax_rows`. The argmax counters
are direct proof that ordinary greedy target rows or greedy DFlash proposal/
verification rows stayed on device. Model activation state remains FP32.
Positive temperature permits the separately counted generic Q8-activation
packed path. Exact-zero MMVQ uses its pinned Q8_1 dot-product representation;
the direct RMS handoff may produce those exact bytes without incrementing the
generic Q8 counters. It was accepted only after bit-exact ABI checks and an
identical full-artifact token hash. `NFN_GLIMMER_MMQ_MEGAKERNELS=0` disables
the grouped/handoff runtime path for development A/Bs; it is not a supported
production tuning flag.

On the documented RTX 5090, the final current-source exact-zero run measured
271.244 DFlash tok/s median over ten trials. The same final binary with only
`NFN_GLIMMER_COOPERATIVE_BATCH_RMS=0` measured 264.724 tok/s (+2.46% enabled),
with identical output, 28/34 acceptance, 37 target rows, sampled memory, and
zero CPU model rows. Earlier retained controls measured 78.608 target tok/s
with the target megakernels versus 72.871 without (+7.87%), and 235.657
DFlash tok/s with packed-weight block hoisting versus 158.779 without
(+48.42%). These numbers apply only to the pinned 41-token/32-token
K-Quant-17GB workload; see `docs/native-cli-inference.md` for hashes,
distributions, memory, sanitizer proof, and comparator caveats.

Glimmer media helpers are model methods:

```python
encoded_image = model.encode_images([image_data_url])
encoded_video = model.encode_videos([decoded_pillow_frames], fps=2.0)
```

They return exact packed-media metadata and projected decoder-width rows.
`NativeInferenceSession.prefill_with_embeddings()` replaces the rendered
`<|patch|>` or `<|video|>` positions transactionally. External URL fetching
and video-container decoding are intentionally outside the resident process.

`NativeInferenceModel` exposes `load`, `create_session`, `fork_session`,
`encode_media`, `encode_images`, `encode_videos`, `prefill`, `decode`,
`current_logits`, `truncate`, `reset`, `cancel`, `stats`, and `close`. A
`NativeInferenceSession` exposes the same state operations after creation.
Closing is idempotent, and closing a model closes its remaining sessions.
Model close publishes one atomic admission boundary under the lifecycle lock.
A create/fork already inside native allocation is registered before the
snapshot and is closed by that teardown; later creates/forks fail before native
allocation. Session operations admitted before the boundary may finish while
close waits for their operation lock, while newly admitted prefill, decode,
logits, truncate, reset, stats, and context-entry calls raise
`NativeInferenceClosedError` before binding compute. The close owner closes
each native session and model handle exactly once and receives the first
teardown error. Concurrent or reentrant duplicate `model.close()` calls return
without waiting, because their caller may already hold a session operation
lock needed by the teardown owner.

After a non-empty prefill, `session.current_logits()` (or
`model.current_logits(session)`) returns an immutable tuple of finite logits
for the current prefix. It is a read-only parity/quality diagnostic: it does
not sample or append a token and it does not mutate cache or RNG state. A
binding that does not expose the diagnostic fails explicitly instead of
returning fabricated log probabilities.

Immutable model weights and kernel tables belong to the model handle. Token
history, RNG seed, cache/recurrent state, cancellation, and telemetry belong to
each session handle. The Python coordinator serializes model compute through a
single lock; it does not claim batching. TurboQuant rotation, QJL, bit-width,
and Lloyd-Max tables are built deterministically once per dense
model/profile and shared by that model's sessions. In `full` mode the reviewed
dense, canonical LLaMA, and exact standard-MoE reference adapters own fixed-capacity K/V
and final-hidden arrays per session. In `off` mode each reports
`recompute_full_prefix: true` and recomputes
the committed prefix for each decode. In `turboquant` mode each session owns
fixed-capacity packed K/V bytes plus lossless final-hidden history.

Reviewed dense-v5, canonical-LLaMA, and exact standard-MoE full-cache artifacts
additionally prove additive `session_prefix_cow` feature ABI v1 with operation
`fork_session` and their exact format-specific profile. Reviewed dense-v5
artifacts separately prove
`session_prefix_cow_cpu_turboquant` for the exact CPU packed MSE/QJL profile.
Effective support for either path requires the corresponding artifact
capability and feature record, binding boolean/profile inventory, and callable
to agree. Cache-off, nonstandard MoE, Tile-CUDA TurboQuant sessions, and any
dense model configured for Tile-CUDA TurboQuant attention remain rejected.
Prefill/decode detachment is transactional: a failed or cancelled append
restores the original shared allocation and its detach counters before the
error reaches the caller.

## Exact-prefix synchronization

`session.prefill(tokens)` computes the exact longest common prefix with the
session's committed history:

- an unchanged prefix performs no native prefill;
- a changed suffix truncates native/Python state to the common position and
  prefills only the new suffix; and
- a zero-length common prefix resets and rebuilds from position zero.

The last rule is important after front-of-context trimming: a matching suffix
cannot reuse absolute-position or RoPE state at different positions.

For a jointly proved dense full-cache or CPU TurboQuant model, fork a non-empty
committed prefix without replaying it:

```python
parent = model.create_session(seed=7)
parent.prefill([101, 102, 103])
child = model.fork_session(parent, token_count=2, seed=11)
child.prefill([101, 102, 900])
assert parent.token_ids == (101, 102, 103)
```

`token_count=None` selects the complete source history; an explicit value must
be in `1..len(source.token_ids)`. A child gets independent Python/native token
history, logical cache length, RNG and active generation seed, cancellation,
counters, reset/truncate, and lifecycle state. Full-cache sessions share native
K/V and final-hidden vectors. A CPU TurboQuant source must be a reviewed dense
session whose effective cache mode is `turboquant`, whose profile is `mse-3.5`
or `qjl-3.5`, and whose attention backend is CPU; it shares its packed K/V byte
store plus lossless final-hidden vector. Either path keeps sharing until an
owner appends, at which point that writer copies every full-capacity component
before publishing either replacement. Reset and truncate are logical and do
not copy, but a later append does. Closing any owner only releases its
reference. Cross-model, closed/poisoned, cache-off, unsupported-family,
Tile-configured-model, Tile-CUDA-session, empty, and out-of-range forks fail
before a child is exposed.

Session stats report `prefix_cow_forks_created`,
`prefix_cow_forked_from_tokens`, `prefix_cow_storage_use_count`,
`prefix_cow_shared_storage`, `prefix_cow_shared_cached_tokens`,
`prefix_cow_shared_capacity_bytes`, `prefix_cow_detach_count`, and
`prefix_cow_detached_capacity_bytes`. The scope string
`this-session-valid-rows-in-shared-allocation` means parent and shorter child
report their own valid row counts while sharing the same capacity allocation.
The same telemetry applies to full-cache and dense CPU packed-TurboQuant
sharing. The standalone HTTP app can now consume these exact primitives through
its optional process-local Responses LRU and schema-v4 conversation CAS. That
does not change the primitive itself: Chat Completions and background Responses
stay cold, and Tile device storage remains unsupported. The manager is an
internal serving component rather than a new top-level SDK type.

The binding must commit `decode_one` atomically before returning. Python then
commits its token mirror before invoking `on_token`, so callbacks observe the
same committed history on both sides. The compiled dense, canonical LLaMA, and
standard-MoE bindings check cancellation inside layers. If an in-flight
`prefill()` is interrupted, native
state rolls back to the previously committed prefix and the SDK raises
`NativeInferenceCancelledError`; the session is cancelled but not poisoned.
If an in-flight `decode()` is interrupted before token commitment, `decode()`
returns `GenerationResult(finish_reason="cancelled", cancelled=True)` without
an event for that token. Call `reset()` before reuse. Reset clears history,
cache, cancellation, and active generation seed and restores the constructor
seed's RNG state, so a stochastic retry matches a fresh same-seed session.

## Cache modes and temperature

`KVCacheConfig.mode` accepts `auto`, `off`, `full`, or `turboquant`.

- `off` needs no cache proof.
- `auto` and `full` require a jointly proven lossless resident cache; `auto`
  currently resolves to `full` rather than compression.
- `turboquant` additionally requires joint TurboQuant proof and accepts
  `mse-3.5` or `qjl-3.5` as the requested profile.

The current reviewed dense reference binding implements all four selections;
canonical LLaMA and exact standard MoE implement `off`, `auto`, and `full`.
`auto` resolves to `full`
only when both artifact and binding prove it. Explicit `turboquant` requires
the reviewed dense cache ABI v1, an even head dimension, and joint
artifact/binding proof; canonical LLaMA, standard MoE, and other unsupported geometry or
family state fail before a session is created and never substitute a full
cache. The native CPU codec uses the portable oracle's deterministic rotation,
Lloyd-Max codebooks, mixed-bit layout, norms, and QJL projection. It scores
compressed keys and accumulates compressed values one row at a time, so it
never materializes the entire dequantized cache. CPU attention remains the
default.

Reviewed dense-v5 artifacts additionally expose
`capabilities.turboquant_tile_attention` and the separate
`kernel_abi.turboquant_tile_attention` v1 symbol when the artifact context is
at most 16,384 and its head dimension is even and in 2..256. Select it
explicitly with:

```python
cache = KVCacheConfig(
    mode="turboquant",
    turboquant_profile="qjl-3.5",
    turboquant_attention_backend="tile-cuda",
    tile_ops_lib="/absolute/path/libnfn_native_train_tile_ops_strict.so",
    cuda_runtime_lib=None,  # optionally pin a path or soname
    cuda_device=0,
)
model = NativeInferenceModel.load("artifacts/model-native", kv_cache=cache)
```

The SDK resolves `tile_ops_lib` to a regular file and jointly requires the
artifact capability/feature ABI, binding capability, and binding configure
operation. C++ then verifies base Tile ABI v1, strict-math ABI v1, feature ABI
v1 and its forward symbol, CUDA runtime/device, and model geometry. Failure is
reported before a Tile session exists; an explicit request never falls back to
CPU. Omitting the new fields preserves the prior CPU session payload.

This is a hybrid path: weights, projections, deterministic row encoding, and
the host packed cache stay on CPU. Model/profile tables and a session-owned
packed cache live on the selected GPU; each committed row is uploaded once,
historical compressed attention runs only through CUDA, and the attention
output is downloaded. Session telemetry includes
`turboquant_attention_backend`, sidecar/runtime/device,
`turboquant_gpu_launches`, `turboquant_row_uploads`, H2D/D2H bytes, and
`turboquant_cpu_compressed_attention_calls`. Live MSE/QJL tests require
positive CUDA telemetry and zero CPU compressed calls. This is dispatch and
correctness proof, not a performance or quality-neutrality claim.

## Resident TurboQuant benchmark

`tools/bench_native_resident_turboquant.py` is the fail-closed comparison
harness for full cache, CPU `mse-3.5`/`qjl-3.5`, and optional Tile-CUDA
`mse-3.5`/`qjl-3.5`. Pass an exact artifact, freshly built resident extension,
strict Tile sidecar, explicit CUDA runtime, and an evaluation token-ID file:

```bash
python tools/bench_native_resident_turboquant.py \
  --artifact artifacts/model-native \
  --binding-lib /absolute/path/_native_inference.so \
  --tile-ops-lib /absolute/path/libnfn_native_train_tile_ops_strict.so \
  --cuda-runtime-lib /usr/local/cuda/lib64/libcudart.so.13 \
  --contexts 1024,4096,16384 \
  --tokens-file /absolute/path/evaluation-token-ids.json \
  --json-out /absolute/path/turboquant-benchmark.json
```

Every mode/context uses separate fresh timing and quality/VRAM workers. TTFT
includes session creation, prefill, the first decode, and synchronous Tile
transfers, but excludes model load. Decode throughput uses an independent
prefill and fixed exact-zero generation. Quality prefills the complete context,
then truncates backward across the configured tail window and calls the public
`current_logits()` diagnostic for exact teacher-forced NLL, perplexity, and
argmax agreement. Free-running greedy agreement is recorded separately. Cache
telemetry distinguishes live bytes, uncompressed-equivalent live bytes, and
fixed allocated capacity.

VRAM is a baseline-subtracted `cudaMemGetInfo` sample from the selected device,
not a per-process allocator high-water mark. Run on an otherwise idle GPU. A
Tile quality/VRAM worker must observe a positive delta; all Tile timing workers
must report positive launches, row uploads, H2D/D2H bytes, and zero CPU
compressed-attention calls. Missing dependencies, short generations, invalid
tokens, mismatched telemetry, or unavailable VRAM make the complete parent run
fail. The output always sets `speedup_claimed: false`.

The 2026-08-08 RTX 5090 calibration used a synthetic nontrivial dense-v5
fixture (one layer, dimension 2, vocabulary 4, fixed 16K capacity), repeated
token zero, no warmup, one timing sample, 16 greedy tokens, and a 128-token
quality tail. It is mechanics evidence, not a representative trained-model or
language-quality benchmark:

| Mode | TTFT 1K / 4K / 16K (s) | Decode 1K / 4K / 16K (tok/s) |
|---|---:|---:|
| full | 0.004725 / 0.070766 / 1.038670 | 77378 / 23299 / 7629 |
| MSE CPU | 0.040002 / 0.648258 / 10.612356 | 11459 / 3026 / 774 |
| QJL CPU | 0.051263 / 0.786051 / 12.698064 | 9476 / 2381 / 645 |
| MSE Tile | 0.450961 / 4.850681 / 69.612433 | 1501 / 442 / 120 |
| QJL Tile | 0.425212 / 4.943382 / 69.876879 | 1475 / 437 / 119 |

At 16K the live/capacity bytes were `393216/393216` for full,
`294912/294912` for MSE, and `376832/376832` for QJL. Both Tile profiles
measured a 4 MiB sampled device-global delta at every context because the one
16K artifact preallocates full device capacity; CPU/full measured zero after
the CUDA measurement baseline. Signed perplexity deltas versus full were about
`-3.31e-5`, `-3.60e-5`, and `-3.70e-5` at 1K/4K/16K. Every Tile result exactly
matched its same-profile CPU oracle. Free-running agreement with full was
16/16 everywhere; teacher-forced agreement was 128/128 at 1K and 16K and
127/128 at 4K. The negative synthetic perplexity delta is not a quality gain,
and the slower Tile timing is not generalized beyond this launch-dominated
tiny fixture. A trained representative artifact and matching corpus remain
required before any product performance or quality-neutrality claim.

`GenerationConfig.temperature` follows the strict inference contract: exact
`0.0` and `-0.0` request strict model computation; every positive value,
including a positive subnormal, remains on the ordinary sampling path.
Negative and non-finite values are rejected.
With exact-zero temperature, `auto`/`full` retain the lossless strict-model
path. Explicit TurboQuant keeps strict deterministic model computation and
deterministic lossy encoding: repeated runs match, but equality with full-cache
logits or tokens is not promised. Telemetry distinguishes
`strict_model_compute` from `lossy_cache`.

## Binding ABI

The default extension name is `neuralfn._native_inference`. An injected binding
is useful for integration tests. ABI version 1 requires these callables:

```text
resident_inference_abi_version
resident_inference_capabilities
load_model / close_model
create_session / close_session
prefill / decode_one
truncate_session / reset_session / cancel_session
model_stats / session_stats
```

When `turboquant_tile_attention` is advertised, the additive
`configure_model_turboquant_attention(model_handle, config)` operation must
also be callable. Its successful result confirms `configured=true` and
`backend="tile-cuda"`. Resident ABI remains v1; this operation is optional for
CPU-only bindings and sessions.

When effective `session_prefix_cow` or
`session_prefix_cow_cpu_turboquant` is advertised, the additive
`fork_session(model_handle, source_session_handle, {"token_count": N,
"seed": S})` operation must also be callable. The base resident ABI remains
version 1. Its `session_prefix_cow_abi` inventory must contain the exact
advertised profile. The compiled binding accepts this operation for a non-empty
lossless full-cache source owned by the same dense, canonical-LLaMA, or
standard-MoE model, or for the exact reviewed dense CPU packed-TurboQuant
source. It rejects cross-kind, Tile-CUDA-session, and
Tile-configured-model forks; artifact/binding feature proof is still required
at the public SDK boundary.

`current_logits` is likewise an additive diagnostic rather than a required ABI
v1 operation for ordinary text generation. A binding may advertise the
primitive `current_logits_exact_prefill=true` only when read-only logits and
the existing exact-prefix `prefill` operation jointly support token-level
selection. Python reports `NativeInferenceCapabilities.structured_output=true`
only when that primitive and the exact artifact capability/feature ABI agree;
`function_tools=true` additionally requires the exact function profile. The
compiled binding continues to report its C++-owned `structured_output` and
`function_tools` booleans as false because the bounded protocol and grammar
engine are owned by Python.

Non-cancellation binding errors poison the affected session so callers cannot
continue from possibly divergent native/Python state. Recreate the session
after such a failure. The private binding's typed `InterruptedError` is the
sole exception: the SDK converts it to the recoverable prefill exception or
cancelled decode result described above.

The compiled implementation requires
`checkpoint.artifact_path` to be a relative path contained by the artifact
root. Dense-v5 load independently checks its explicit topology and format.
Canonical LLaMA load re-runs the dependency-light registry proof, requires the
exact v2 geometry, semantics, names, roles, shapes, offsets, byte order, and
per-tensor hashes, verifies artifact containment/size/full SHA-256, and hashes
the exact loaded float buffer before accepting it. It rejects path escapes,
unsupported family classifications, malformed checkpoint declarations,
unsupported cache requests, and invalid sampling controls. `model_stats()` and
`session_stats()` report load/forward/subprocess counts, requested/effective
cache, cached tokens, actual/capacity/uncompressed bytes, compression ratio,
strict/lossy flags, prefix reuse, fallback reason, decode rows processed, and
the Tile-CUDA telemetry above when selected, plus full-cache or dense CPU
packed-TurboQuant COW ownership/detach counters when applicable.
The old `_native_gpt` / `_native_gpt2` one-shot command-capture bindings remain
available for compatibility and still launch their compiled command; they have
not yet been rewritten as wrappers over this engine.

## Standalone serving integration

`neuralfn.native_serve` builds a separate FastAPI application around this SDK;
it never imports the editor backend or falls back to one-shot subprocess
generation. Install `.[serve]`, then use `nfn infer --checkpoint ARTIFACT
--serve`. `NativeServingRuntime.load()` validates the manifest tokenizer, chat
template, context limit, resident ABI, binding, and requested cache before
Uvicorn can bind.

Resident Responses prefix reuse is opt-in and entry-count bounded:

```python
from pathlib import Path

from neuralfn.native_serve import NativeServeConfig, prepare_native_inference_server

config = NativeServeConfig(
    artifact=Path("artifacts/model-native"),
    state_db=Path("native-inference-state.sqlite3"),
    prefix_cache_capacity=64,
)
app, runtime, auth = prepare_native_inference_server(config)
```

`NativeServeConfig.prefix_cache_capacity` is a non-negative integer and
defaults to `0`, which constructs no cache and preserves cold-per-request
behavior. A positive value requires `state_db` plus an effective lossless
`full` cache with `session_prefix_cow`, or reviewed dense CPU TurboQuant with
`session_prefix_cow_cpu_turboquant`. Cache-off, unproved COW, and Tile-CUDA
TurboQuant fail before bind. The matching CLI flag is
`--prefix-cache-capacity N`. This public field is appended after `log_level` in
the frozen dataclass, so existing positional meanings are unchanged; code that
assumes the exact dataclass field count/shape must accept the added trailing
field. Keyword construction remains recommended.

For foreground HTTP Responses, a stored `previous_response_id` or exact
conversation revision selects a scope-local candidate. The app independently
verifies the complete rendered token LCP before forking; alias ancestry alone
is never enough. Only stored `completed`/`incomplete` results publish after
durable finish. Failed/cancelled/background results do not publish, while
`store=False` can hit but cannot admit. `cached_tokens` is bounded by exact LCP
and native cached rows; `cache_write_tokens` counts only newly written prompt
rows, excluding decode. Response/conversation/item deletion purges the whole
API-key scope and fences old leases. Restart is cold.

The serving surface exposes `/health`, `/v1/models`, `/v1/models/{model}`, and
bounded `/v1/chat/completions`. Chat is text by default; a jointly proven CPU
Muse Glimmer vision artifact may additionally accept base64 image data URLs. A
bounded single-worker queue creates one isolated SDK session per request and
forwards SSE deltas only after token commitment. `NativeServeConfig`, `BearerAuth`,
`NativeServingRuntime`, and `create_native_inference_app()` are available from
`neuralfn.native_serve` for embedding the isolated app.

Setting `NativeServeConfig.state_db` opts the app into the separate versioned
SQLite store and mounts text Responses create/retrieve/delete, input items and
token counting, scoped lineage and local compaction, Conversations CRUD/items,
semantic Responses SSE, and durable background/cancel processing. Embedders that construct
`NativeServingRuntime` directly must provide an open `NativeStateStore` to
enable those routes; a runtime without one retains Chat Completions only. They
must also pass the actual selector through
`chat_template_selection` (normally `"auto"`). The runtime rechecks the raw
manifest's exact `chat_template.tool_template` before retaining effective
function support, and any explicit `plain_roles` or path selection downgrades
both constrained capabilities because it is not the artifact-selected
presentation contract.

Import the store as
`from neuralfn.native_state import NativeStateStore`, pass the open instance as
`state_store=...`, and call `runtime.close()` during application teardown; that
shuts down the prefix cache before closing both the resident model and the
owned store. If direct runtime
construction fails before ownership transfers, close the store in the caller.

When enabled, `/health` adds a `prefix_cache` object whose exact manager
snapshot includes `capacity`, retained `entries`, response/conversation alias
counts, `active_leases`, `in_flight_forks`, `hits`, `misses`, `evictions`,
`scope_purges`, `commits`, `rejections`, cumulative cached/write-token
observations, shared/private/detach capacity observations, retained capacity
observations, and `byte_accounting_scope`. Capacity is an entry count. The byte
fields are not unique physical memory: native COW sharing can cause multiple
sessions to report one allocation. Capacity zero omits this health object.

The HTTP app, not the public phase methods below, owns the combined resident
lease lifecycle. Shutdown stops and awaits the background driver, then awaits
foreground SSE drivers through durable finish, drains the queue, and lets
runtime close cache, model, and store in that order.
`NativeResponsesService.execute()` plus `finish()` remains a supported
deliberately cold embedding path and reports both cache detail fields as zero.
Direct embedders that require the HTTP cache behavior should embed the app
rather than import private `_execute_resident` helpers.

### Stateful serving module APIs

The lower-level state and Responses service types are supported module APIs for
embedders that need to own the request lifecycle themselves:

```python
from neuralfn.native_state import (
    ANONYMOUS_API_KEY_FINGERPRINT,
    NATIVE_STATE_SCHEMA_VERSION,
    NativeStateConflictError,
    NativeStateError,
    NativeStateStore,
    api_key_fingerprint,
)
from neuralfn.native_responses import (
    CompletedNativeResponse,
    NativeResponsesAPIError,
    NativeResponsesService,
    PreparedNativeResponse,
)
```

`NativeStateStore(path)` owns a private schema-v4 SQLite database. It is a
context manager and `close()` is idempotent. Records are partitioned by the
non-reversible value from `api_key_fingerprint()`; the store persists JSON and
token-derived history, never resident KV buffers. `NativeStateError` covers
store failures and `NativeStateConflictError` identifies immutable-ID
conflicts plus optimistic-concurrency failures; its public `code` is
`"conversation_conflict"` for a stale conversation revision.
`ANONYMOUS_API_KEY_FINGERPRINT` is the fixed scope used when no
Bearer key is configured; authenticated embedders should derive their scope
with `api_key_fingerprint()` instead. Back up a v1/v2/v3 database before
opening it with this release when rollback matters: migration adds
`conversations.items_revision` in place, treats existing item history as
revision zero, and older binaries reject schema v4. Migration cannot recover
the item revision observed by a conversation-linked background response queued
by an older binary; the server terminalizes that legacy job as failed with
`conversation_snapshot_unavailable` instead of generating against current
items. A legacy previous-response-only queued job reconstructs only a currently
completed/incomplete lineage or fails with `response_lineage_unavailable`.

Schema-v4 conversation methods expose the revision fence directly:

```python
items, revision = store.conversation_items_snapshot(scope, conversation_id)

created, next_revision = store.append_conversation_items_with_revision(
    scope, conversation_id, new_items
)
deleted, next_revision_or_none = store.delete_conversation_item_with_revision(
    scope, conversation_id, item_id
)
```

`conversation_items_snapshot()` returns ordered items and revision from one
transaction, raising `KeyError` when the conversation is absent.
`append_conversation_items_with_revision()` returns the appended records and
new revision. `delete_conversation_item_with_revision()` returns `(deleted,
revision)`, using `None` only when the conversation itself is missing. The
legacy `append_conversation_items()` and `delete_conversation_item()` return
their prior shapes but increment the same revision.

Use `finish_foreground_response(scope, response_id, *, status,
response_patch, response_items, conversation_id=None, conversation_items=(),
expected_conversation_revision=None)` for an atomic terminal boundary. It
returns `(stored_response, committed_revision_or_none)` or `None` when the
response is absent. When conversation arguments are supplied, response state,
output rows, conversation rows, and revision CAS commit together; a stale
expected revision raises `NativeStateConflictError(code="conversation_conflict")`
and commits none of them. `finish_background_job()` now accepts the same
optional `conversation_id`, `conversation_items`, and
`expected_conversation_revision` keyword arguments and performs that CAS inside
its response/job/event terminal transaction.

`NativeResponsesService(runtime, state_store)` exposes the same bounded
Responses contract used by the standalone server. The principal foreground
phases are `prepare(scope, payload) -> PreparedNativeResponse`,
`persist(prepared)`, `execute(prepared, ...) -> CompletedNativeResponse`, then
`finish(prepared, completed)` or `fail(prepared, ...)`. `prepare()` performs
validation, lineage resolution, prompt rendering, and constrained/function
planning without opening a resident session; `execute()` owns one isolated
session; `finish()` or `fail()` terminalizes durable state. Preserve that
ordering, including validation before admission or state mutation, when
embedding the service. The service also provides the matching retrieval,
deletion, token-count, compaction, background replay/cancel, and Conversation
methods used by the HTTP routes. `NativeResponsesAPIError` carries
`status_code`, `message`, `error_type`, `param`, and `code`; `payload()` returns
the REST error envelope. The frozen prepared/completed dataclasses are phase
handoff values, not durable serialization formats.

Conversation preparation uses the schema-v4 snapshot. If another item mutation
or response completion wins before finish, the service terminalizes the stale
stored response as failed and raises `NativeResponsesAPIError` 409 with
`param="conversation"` and `code="conversation_conflict"`; it does not append
output/conversation rows or publish a prefix. A previous-response deletion race
uses `param="previous_response_id"` and
`code="response_lineage_conflict"`: finish re-reads the complete stored lineage
under the transition lock, so any deleted or changed ancestry fails before
output/cache publication. Reconstructed legacy lineage is revalidated the same
way. HTTP streaming/background flows represent these conflicts with a stored
semantic `response.failed` terminal.

The cache transition lock and purge epoch belong to one process. Do not share a
cache-enabled state database across server/service processes, and do not mutate
it out of band through a second raw `NativeStateStore`; route response,
conversation, and item mutations through the owning
`NativeResponsesService`. Cross-process cache deletion linearizability is not
claimed.

`neuralfn.native_constrained` is a serving-internal implementation module and
is not exported as a supported top-level SDK surface. The compatibility-stable
public seam is the manifest/capability contract, read-only
`NativeInferenceSession.current_logits()`, exact-prefix `prefill()`, and the
Responses HTTP API; direct imports of its compiler/grammar/inventory helpers
are not promised compatibility.

The seven reviewed dense-v5 topologies have end-to-end text serving coverage;
`gpt2_moa` reaches that gate only through its strict migrated metadata contract.
Canonical LLaMA proves the resident model/lean serving ABI, but its checkpoint
migration does not synthesize a tiktoken codec or chat template; startup still
fails before bind unless the artifact independently carries supported
presentation metadata. Compatible resident artifacts default to lossless
`auto`/`full`; graph-only artifacts, generic `.pt` bundles, bare MoA `.bin`
files, differential/modern variants, and unimplemented families remain rejected.
When the effective capability gate above succeeds, the stateful Responses
route supports one Python-owned constrained profile: strict flat root-object
schemas with required scalar string/integer/number/boolean properties (or
finite homogeneous enums), printable-ASCII byte tokens, greedy allowed-token
selection from `current_logits()`, and exact-prefix commitment through
`prefill()`. The same engine can emit one forced client-executed function call
and later consume its string `function_call_output`; it never invokes the
function. These calls require artifact-selected presentation metadata,
`store=true`, and buffered foreground execution. Schema/argument generation
additionally requires `temperature=0` and `top_p=1`; the separate function
result continuation is ordinary text generation and may use ordinary sampling
controls, but requires disabled truncation.
General/parallel/hosted tools, nested/array schemas, constrained streaming or
background work, Chat Completions tools, Responses multimedia, audio/files,
server video, and hosted resources remain explicitly unsupported. Bounded Chat
Completions base64 images may use an authenticated Glimmer CPU or whole-model
CUDA vision artifact.
See the
[REST contract](../rest-api/native-inference-serving.md) and
[server architecture](../server/native-inference-serving.md).
