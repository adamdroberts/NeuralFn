# `multi-turn-graph-migrate-turboquant-todo.md`

## Summary and completion contract

Build a native-first inference stack that supports multi-turn CLI use, real resident model serving, TurboQuant KV caches, graph-to-native migration, and OpenAI-compatible Chat Completions plus broad Responses resources.

Original constraints the implementation must fully correct (the dense
milestone has now corrected some of these, but all-family completion has not):

- Interactive graph and resident-artifact inference now default to transcript replay; the legacy raw native-checkpoint sampler remains a one-shot compatibility path.
- Seven reviewed dense-v5 GPT topologies, including source-bound `gpt2_moa`, the exact canonical native-family
  `llama`/compile-alias `llama_fast` profiles, and the exact standard-MoE
  `moe`/`mixllama`/`mixllama_fast` cluster now execute genuine resident
  autoregressive forwards; every other native-family checkpoint still uses
  diagnostic transition sampling.
- Those seven dense-v5 topologies plus the canonical LLaMA eager/compile and exact
  standard-MoE profiles now have a lossless full K/V cache with an `off`
  recomputation oracle. Differential/modern variants, unbound MoA `.bin` files,
  every other family, and the legacy
  `InferenceCache` still lack retained production state.
- Those reviewed dense-v5 artifacts with even head dimensions, including exact
  source-bound MoA artifacts, now also have native packed CPU `mse-3.5` and
  `qjl-3.5` TurboQuant caches with direct compressed-row attention. An explicit
  separately gated hybrid Tile-CUDA backend now consumes those same packed
  records for historical attention while CPU remains the default and owns
  weights/projections/encoding. Other dense variants and every other family
  remain incomplete.
- Graph models remain CLI-compatible during deprecation but are not served directly.
- Every shipped text family must ultimately receive a real native forward runner; diagnostic sampling cannot satisfy completion.

Default decisions:

- Graphs remain the authoring format and lower ahead of time into a versioned Native Execution IR. Direct C++ graph interpretation is not added.
- `nfn infer --serve` accepts native artifacts only and serves one resident model per process.
- Full and TurboQuant caches are native-runtime features. Legacy graph CLI inference may replay the full prefix until migrated.
- Native-family work is complete only when all optimizer-updated architecture tensors are persisted and exercised by a real forward—not merely checkpoint metadata or transition tables.
- Arbitrary custom Python graph nodes remain unsupported unless they register a native lowering; every shipped text preset must lower successfully.

## 1. Graph migration and native runner program

- [x] Define `NativeExecutionManifest`/Native IR v1 containing the resolved topology, family/objective, tensor table and dtypes, tokenizer and chat-template metadata, context limits, stop tokens, kernel ABI, checkpoint fingerprint, session-state kinds, and cache/tool/structured-output capabilities.
- [x] Keep graph JSON serialization unchanged. Compile resolved inline subgraphs and variant libraries into Native IR while preserving the existing port-compatible inline-subgraph fallback and back-to-back preset safety.
- [ ] Extend `nfn train --runtime native-cuda --graph-file ...` to lower every shipped text preset and invoke its native trainer.
  - [x] Add the exact canonical `llama` graph-file adapter: reuse the active
    resident topology proof, derive complete trainer geometry/provenance,
    snapshot and SHA-bind the graph, re-plan SDK/server handoffs, keep
    dry-run/plan/check operations non-training, honor family weight decay, and
    bind v2 checkpoint discovery/migration to the same source digest. Seven
    dense GPT-2 adapters are execution-ready; structurally reviewed
    `gpt2_diff` stays fail-closed until separately proved below.
  - [x] Promote only the graph-equivalent compile-runtime `llama_fast` profile
    through the same canonical LLaMA adapter. Normalize its native trainer and
    checkpoint identity to `llama`, preserve selector/preset/runtime/SHA
    provenance through CLI, SDK, MCP/server, migration, and resident loading,
    and keep `llama_fast_megakernel` plus neighboring LLaMA profiles closed.
    Independent focused verification passes `133` tests.
    - [x] Fix the separate `train_llama_megakernel.py` compatibility shim to
      preserve its named native selector for normal and preflight actions and
      consume `--fast` as `llama-fast-megakernel`. This is routing correctness,
      not adapter promotion: audit proved the current fused ABI is a host-side
      chain of generic kernels, so both megakernel profiles remain closed.
  - [x] Promote the graph-equivalent standard-MoE cluster (`moe`, `mixllama`,
    `mixllama_fast`) only after all three remaining parity gates pass:
    floating expert-width/rounding parity; the exact all-expert router
    auxiliary-loss forward and gradient with its configured coefficient; and a
    strict source-SHA/tensor-table checkpoint plus real MoE forward/parity
    loader. Keep `mixllama_fast_megakernel`, `moe_modern`, `deepseek_v3`, and
    every JEPA/MoE-JEPA neighbor closed. Full C++ parameter persistence alone
    is not graph-faithful evidence.
    - [x] Preserve floating `mlp_multiplier` plus optional `multiple_of` in the
      expert graph/runtime and prove Torch/Tile/native packed-weight parity.
      - [x] All shipped MoE builders now serialize the floating multiplier and
        optional alignment into every standard/semantic dispatch node. Torch
        and Tile use the same validated floor-then-optional-ceil geometry and
        packed `w1`/`w3`/`w2` shapes; deterministic CPU forward/backward
        parity, the 34-test mandatory preset gate, and the 111-pass CPU Tile
        sweep cover this bounded slice.
      - [x] Make the reviewed graph adapter encode graph `multiple_of=None` as
        the native trainer's explicit `--multiple-of 0` sentinel, then prove
        its packed C++ parameter table matches the Torch/Tile tensor shapes.
        The exact graph planner now passes floating multiplier plus sentinel,
        and a fresh host family plan reports hidden width `21`, 21 ordered
        buffers (`3 + 9 * 2`), and packed expert shapes matching Torch/Tile.
    - [x] Implement the graph's exact router auxiliary loss and all-expert
      softmax-Jacobian gradient in the production C++/Tile path.
      The production ABI computes
      `E * sum(mean_rows(softmax(router_logits))^2)`, accumulates its configured
      gradient before router backward, treats coefficient zero as an exact
      no-op, and rejects invalid/narrowed or incompatible profile coefficients.
    - [x] Add strict graph-bound MoE checkpoint inspection, real resident MoE
      inference, graph/native logits-loss-gradient parity, and live CUDA
      train/save/reload/generate acceptance.
      - [x] Add the DONE-gated, source-SHA-bound standard-MoE v1 inspector,
        exact tensor table and float32 sidecar migration, real resident
        top-k-routed expert forward, graph/resident all-prefix logit parity,
        independently reconstructed CE plus router-auxiliary loss, finite
        nonzero router gradients, SDK prefix reuse, raw-token CLI inference,
        strict tamper/neighbor rejection, and lossless full/off cache parity.
      - [x] Run the live CUDA train/save/reload/generate acceptance. After a
        stale Tile sidecar was correctly rejected, a fresh sidecar exported the
        router-auxiliary ABI. The exact graph then completed one RTX 5090
        optimizer step, emitted the source-bound 12-tensor standard-MoE v1
        checkpoint, migrated losslessly, loaded through the rebuilt resident
        extension, and generated `3,3` through raw-token full-cache CLI.
  - [x] Run the policy-authorized live canonical LLaMA graph-file acceptance:
    one real CUDA optimizer step, inspect the v2 checkpoint/source digest,
    migrate it, rebuild/load the resident binding, and generate raw tokens. The
    exact graph emitted an 11-tensor source-bound LLaMA v2 checkpoint, migrated
    losslessly, loaded through the rebuilt extension, and generated `3,3`
      through raw-token full-cache CLI on the RTX 5090.
  - [x] Promote exact `gpt2_moa` inference through a strict sibling metadata
    contract. A completed `model_XXXXXXXX.bin` must have
    `model_XXXXXXXX.moa.json` plus an empty `DONE_XXXXXXXX`; metadata schema
    `neuralfn.native_dense_moa.inference_checkpoint` v1 binds the dense-v5
    bytes and byte-identical source graph, the canonical
    `[gelu,relu,silu,relu2]` candidates, one selected activation, and a
    positive probe interval. Migration accepts the metadata JSON rather than a
    bare MoA `.bin`, copies the dense weights as `model.bin`, and the CPU
    resident engine uses the selected activation through prefill/decode,
    `off`/`auto`/`full`, and supported-even-head TurboQuant cache paths.
    Graph-bound resume requires the sibling metadata, restores the recorded
    selection without a fresh probe, and fails closed when it is absent or
    tampered. The direct selector-only first leg remains supported and writes
    ordinary dense-v5, but an unbound MoA resume now fails explicitly instead
    of resetting the activation to GELU. The
    combined MoA/native-graph/CLI slice passes 58 tests, including 17 focused
    MoA tests and a live one-step source/model-hash-verified export.
    - [x] Close the non-GELU graph/resident parity gap without rewriting the
      authoring graph. Migration copies the exact validated metadata as
      `model.moa.json` and binds its path/size/SHA-256. The public
      `native_moa_graph_runtime` parity/debug loader revalidates graph, dense-v5,
      metadata, and tensor-table bytes, imports every native tensor, and
      overlays only the canonical per-layer MLP activation. Tanh-GELU, ReLU,
      SiLU, and ReLU-squared all match resident logits and an independent
      formula/CE oracle; focused parity passes 11 tests and the combined gate
      passes 96. Older migrated MoA directories without the copied metadata
      must be remigrated. This does not change the low-level resident loader's
      manifest trust boundary or make the unsigned artifact authentic against
      coordinated replacement.
  - [x] Make graph-authored `gpt2_diff` fail honest while retaining structural
    Native IR lowering. Its adapter no longer proves architecture persistence,
    execution readiness, or native forward because dense-v5 omits the graph's
    learned scalar lambda for every layer and the current differential native
    forward is exact only on its packed-QKV branch. Generic persistence and
    resident readiness remain `12`/`54`; the exact proof-bound training-only
    promotion is recorded separately below.
    - [x] Remove the unsafe ordinary-attention fallback from the low-level
      differential trainer path. Reject before Tile load/allocation/mutation
      unless packed QKV is enabled, `seq_len >= 16`, head geometry is
      divisible/even, BF16 QKV-gradient handoff is enabled, and both
      differential learned-lambda forward/backward symbols plus the mandatory
      workspace-release Tile ABI symbol are present. Defensive
      runtime guards keep differential forward/backward out of ordinary SDPA;
      the focused fail-closed and combined packed/preflight suites prove zero
      steps, initialization, AdamW, checkpoint, or output-dir mutation on
      rejection. This is a safety gate, not learned-lambda persistence.
    - [x] Implement and independently prove the bounded low-level learned-lambda
      training bundle. One device FP32 lambda per layer participates in real
      pre-quantization RMSNorm-aware backward, deterministic reduction, clipping,
      and AdamW; additive
      `diff_parameters_XXXXXXXX.bin`, `diff_optimizer_XXXXXXXX.bin`, and
      source-bound `model_XXXXXXXX.diff.json` siblings preserve lambda plus
      moments while leaving dense-v5/dense optimizer bytes unchanged. Metadata
      schema `neuralfn.native_gpt2_diff.training_checkpoint` v2 retains
      `checkpoint_kind=trained_dense_v5_plus_diff_v1` for the unchanged binary
      formats. Continuation preflight completes before Tile/CUDA/H2D and binds
      graph and all five artifacts, exact headers/geometry/state,
      optimizer/microbatch and sampler counters, seed explicitness/value,
      batch/accumulation shape, optimizer/LR settings and absolute horizon,
      LM-head chunk, effective BF16 routes, a canonical profile of supported
      effective numerics, and resolver-ordered training-shard contents through
      retained stable no-follow descriptors. Validation shards are excluded.
      `--max-steps` adds work;
      omitted resume horizon inherits v2 metadata and explicit mismatch rejects.
      Final export is create-only, fsyncs DONE last, and rolls back newly created
      files on in-process failure; it is not atomic-rename publication, does not
      reject ancestor symlinks, and excludes metadata smoke. Bounded host/runtime
      proof covers math, drift/tamper/symlink rejection, and five-binary equality
      for same-build straight four-step versus split two-plus-two continuation.
      It does not prove JSON/DONE byte identity, cross-build determinism,
      validation identity, migration/resident execution, or performance.
      The learned trainer requires the learned-lambda ABI. The retained legacy
      fixed-lambda ABI remains outside it with rounded-output/non-layer-local
      backward correctness debt.
    - [x] Restore exact graph-planned training through a trusted local proof
      without promoting migration or resident inference. The planner validates
      the same bounded immutable graph bytes through a forward-closed raw JSON
      schema, exact GPT-2 configuration, and exact active topology, then emits
      canonical `native-training-proof.json` bound to the source SHA, validator
      contracts, reviewed shape hashes, and geometry. Every graph-bound
      differential plan/check/startup/train invocation requires the exact
      graph/fingerprint/proof triplet before plan output, Tile, CUDA, H2D, or
      mutation; the proof contract SHA is also bound into schema-v2 continuation
      metadata and resume. The unkeyed digest proves integrity only inside the
      trusted local planner handoff, not caller authenticity. This makes exact
      graph training `13` ready/`53` blocked while generic persistence and
      resident readiness remain `12`/`54`.
      - [x] Keep server completion discovery bounded for production-size
        bundles. It derives exact sizes from proof geometry and streams each
        no-follow artifact independently through SHA/header/BF16/FP32/moment
        validation with at most 1 MiB payload chunks. The full server module
        passes 32 tests, including a 32 MiB traced-memory regression below
        8 MiB peak allocation; the combined proof/server/registry/CLI slice
        passes 155 tests and the independent host review passes 217 with five
        live skips.
      - [ ] Re-run the final eight-test learned-lambda CUDA matrix against the
        exact proof-bound source snapshot. The approved launch was rejected
        before process creation by the temporary execution quota; the earlier
        eight-test live pass predates the final planner proof and therefore is
        retained only as diagnostic evidence, not the final acceptance run.
    - [ ] Make the learned differential workspace registry device-aware before
      advertising the raw ABI for callers that switch CUDA devices in one
      process. It is currently keyed by stream only; the shipped trainer's
      single selected device is covered, but default-stream keys can collide
      across devices and teardown runs under the caller's current device.
    - [x] Keep the downstream boundary explicit while it remains unimplemented.
      Generic registry/migration capability still reports the missing
      architecture-persistence and resident-consumer gates even though the
      exact trusted planner can issue a training-only proof. Graph-to-native
      migration rejects `.pt`, raw `.bin`, and metadata-v2 `.diff.json` weights
      before generic checkpoint dispatch or output creation, with guidance to
      retain the complete training bundle.
    - [ ] Restore graph-planned migration/resident execution only after an
      inspector and migration contract consume the differential bundle and the
      resident engine implements exact packed differential attention/cache
      semantics. The ready/blocked matrix remains `12`/`54`; dense-v5 alone is
      insufficient.
  - [x] Fail GPT2-Evo closed at both native entrypoints instead of advertising
    the semantically different dense `--layer-evo` loop. The authored graph
    excludes and evolves every tensor in one block; the retained loop AdamW-
    updates the block and evolves only `block_N.ln1.weight`. Family and generic
    normal/startup/print-command paths now reject before delegate/Tile/CUDA/
    output/mutation, while plan/dry-run expose the exact three missing gates and
    isolated primitive/metadata smokes remain diagnostic. The family slice
    passes nine tests and the constructor-marking direct guard passes twenty.
    - [ ] Re-enable only after whole-block gradient/AdamW exclusion, whole-block
      mutation/evaluation/selection/adoption, and exact evolutionary checkpoint/
      resume state match the authored graph with independent CUDA/parity proof.
  - [x] Correct NanoGPT's shared-target capability overclaim. `nanogpt`,
    `nanogpt_megakernel`, and `nanogpt_modern` remain structurally lowerable and
    route to `nfn_gpt_native_train`, but the family now reports persistence and
    resident inference false plus `diagnostic-transition-only`: its graphs
    author bias-free linears and dropout while the shared dense-v5 loop is
    biased and dropout-free. Thirteen focused registry tests and 41 combined
    registry/graph-training tests pass without weakening reviewed GPT-2 proof.
    - [ ] Restore exact NanoGPT execution only after a source-bound checkpoint
      contract represents bias-free parameters (zero-validated compatibility
      slots or a new format), all three dropout sites execute and backpropagate
      deterministically across resume, eager/megakernel/modern topology is
      independently validated, and real native/resident logits plus live CUDA
      save/reload parity pass.
- [x] Add `nfn migrate graph-to-native --graph GRAPH [--weights WEIGHTS] --output-dir DIR [--dry-run]`:

  - [x] Validate every node and variant before loading weights.
  - [x] Convert compatible legacy `.pt` weights through an isolated optional-Torch migration path.
  - [x] Emit a compatibility report with unsupported node paths, tensor mappings, checksums, and the resulting model capabilities.
  - [x] Never overwrite the source graph, weights, or an existing output directory.

- [x] Extend the graph/editor runtime union with `native-cuda`, add a native compatibility preflight to the existing run workflow, and show node-specific lowering failures before training starts.
- [x] Route editor-started native runs through the same compiler/trainer registry as the CLI; persist the resulting Native IR and checkpoint path with the run.
- [x] Update MCP training behavior to accept `runtime: native-cuda` and return the same compatibility and artifact metadata.
- [x] Preserve legacy graph/Parameter Golf CLI inference with a deprecation warning. Reject `--serve` and explicit TurboQuant on those artifacts with the exact migration command needed.
  - [x] Graph-backed warnings print the exact shell-safe command for the supplied paths. Graphless Parameter Golf checkpoints fail closed on topology-dependent features and print an explicit `MATCHING_GRAPH.json` migration template because no exact graph can be inferred; generic migrated `.pt` bundles remain nonresident until paired with a separately compatible resident checkpoint.

Implement real native training persistence and resident inference adapters for every shipped text family:

| Family class | Resident session state | TurboQuant policy |
|---|---|---|
| Dense GPT/GPT-2/GPT-3/NanoGPT/GPT2-Evo, LLaMA/GQA, MoE, semantic/JEPA, differential and sparse variants | Per-layer K/V plus routing metadata | TurboQuant all standard retained K/V |
| DeepSeek/MLA and KV-PCA variants | Native latent or PCA cache | Run natively; reject TurboQuant until a separately validated latent/PCA codec exists |
| Jamba, TTT-LLaMA, Universal-LLaMA, HNet-LM | Attention K/V plus Mamba, TTT, recurrent, or partial-patch state | TurboQuant attention K/V only; retain auxiliary state losslessly |
| Seq2seq | Encoder state, decoder self-K/V, static cross-K/V | TurboQuant compatible decoder and cross-attention K/V |
| Text diffusion | Denoising state rather than autoregressive K/V | Native inference with cache capability off; reject TurboQuant |

- [ ] Replace ordinary `nfn infer` use of native-family transition sampling with the real adapter once each family passes its gates. Retain transition inspection only under an explicitly diagnostic command.
  - [x] Route a migrated dense checkpoint-file invocation through its resident adapter when a sibling Native Execution v1 manifest binds that exact contained file; standalone raw dense checkpoints retain the one-shot compatibility sampler.
  - [x] Route canonical `llama` and its exact compile-runtime alias
    `llama_fast` through normalized `llama` inference-checkpoint v2 artifacts
    and the in-process resident adapter when invoked by artifact directory,
    manifest, or exact contained `model.f32`. Preserve source-profile
    provenance. The raw-token CLI path is tokenizer-free and uses real logits;
    text prompts and HTTP serving still require separately supported tokenizer
    metadata.
  - [x] Route exact standard-MoE `moe`, `mixllama`, and `mixllama_fast`
    migrated artifacts through the strict graph-bound checkpoint and resident
    router/expert adapter for SDK and ordinary raw-token CLI inference. Keep
    text/HTTP conditional on tokenizer/chat metadata and TurboQuant closed.
  - [ ] Replace transition sampling for each non-dense family only after its production forward, persistence, Native IR, and resident-session gates pass.
- [x] Drive coverage from the shipped preset and native-family registries so new text presets cannot ship without training persistence, Native IR lowering, and inference coverage.
  - [x] `tests/test_native_registry_coverage.py` derives all expectations from
    `SHIPPED_GPT_TEMPLATE_PRESETS`, `NATIVE_TRAIN_FAMILY_TARGETS`,
    `native_trainer_specs()`, `native_graph_training_adapters()`, graph
    preflight, and `capability_proof_for()`. It requires exact unique family
    ownership and source-target parity, lowers every real preset graph, and
    requires persistence/resident inference to be either proved or accompanied
    by an explicit adapter/plan/missing-gate blocker. The current registry
    matrix is 66 presets: 12 persistence+resident ready and 54 explicitly
    blocked; this status gate does not promote those 54. The registry-focused
    combined slice passes 80 tests.
- [x] Exclude embeddings and other non-text-generation artifacts from chat/model-serving catalogs. Require both the independently derived `capabilities.serve` proof and `model.text_generation=true`; reject a stale manifest that overclaims serving before resident load or `/v1/models` advertisement.

## 2. Resident inference, multi-turn, and TurboQuant

### Resident runtime

- [x] Introduce additive SDK types: `NativeInferenceModel`, `NativeInferenceSession`, `NativeInferenceCapabilities`, `NativeInferenceCancelledError`, `GenerationConfig`, `KVCacheConfig`, `GenerationEvent`, and `GenerationResult`.
- [ ] Give every native adapter the common operations `load`, `create_session`, `prefill`, `decode`, `truncate`, `reset`, `cancel`, `stats`, and `close`.
- [ ] Load immutable weights and kernel tables once per model. Keep token history, RNG, KV/recurrent state, and cancellation status isolated per session.
  - [x] The seven reviewed dense-v5 preset topologies load weights once and share each deterministic TurboQuant table set once per model/profile; exact MoA loads only after its selected-activation metadata/source/DONE contract validates, and full and compressed cache/RNG/history/cancellation state remain session-local.
  - [x] Canonical LLaMA loads one immutable float32 tensor image per model and keeps RoPE/GQA K/V, final-hidden history, RNG, token history, cancellation, and rollback state isolated per session.
  - [x] Exact standard MoE loads one immutable graph-bound float32 tensor image
    per model and isolates RoPE/GQA K/V, routing execution, RNG, token history,
    cancellation, rollback, and cache state per session; cross-model session use
    is rejected.
  - [ ] Prove the same ownership contract for every remaining native family adapter.
- [ ] Replace the current subprocess-style native binding with an in-process C++ engine. Preserve old one-shot commands as compatibility wrappers over that engine.
- [x] Implement exact-prefix synchronization: compute the longest common token prefix, truncate reusable state, prefill only the suffix, and rebuild from zero after front-of-context trimming.
- [ ] Add explicit cache positions for absolute embeddings and RoPE, right-aligned causal masks for one-token queries, preallocated append storage, true token callbacks, and cancellation checks between layers/tokens.
  - [x] The reviewed dense-v5 MHA/absolute-position paths use explicit append positions and preallocated caches, commit callbacks only after native/Python state, and cooperatively cancel inside layers with rollback/recoverable reset.
  - [x] Canonical LLaMA uses explicit RoPE positions, GQA head mapping, right-aligned one-token causal attention, preallocated append storage, committed callbacks, and recoverable cancellation/rollback.
  - [x] Exact standard MoE reuses the proved RoPE/GQA/right-aligned attention
    core and adds top-k-renormalized routed experts with the same explicit
    positions, preallocated append storage, committed callbacks, and recoverable
    lifecycle semantics.
  - [ ] Add and prove the RoPE/GQA/right-aligned-mask and non-dense state paths.
- [x] Use one server worker and a bounded model compute queue initially. Sessions may interleave safely between tokens; no batching claim is made until separately implemented.
- [x] Deprecate the misleading legacy `InferenceCache` documentation without changing its public signature in this task.

### Multi-turn CLI

- [x] Make interactive TTY inference default to `--chat-mode transcript`; retain `--chat-mode stateless` and `/mode stateless`.
- [x] Add `--system-prompt` and `--chat-template auto|plain_roles|PATH` to both lightweight and full CLI parsers/help.
- [x] Represent turns as role messages supporting `developer`, `system`, `user`, `assistant`, and tool items.
- [x] Prefer the artifact-provided chat template. CLI may warn and use `plain_roles`; serving must require an explicit fallback when the manifest lacks a template.
- [x] Retain an initial prompt as the first turn, stop/strip configured role or EOS delimiters, and keep CLI history process-local.
- [x] Reserve output-token capacity before rendering. Preserve leading instructions and the newest user/tool group, drop the oldest complete conversational groups, and fail if the mandatory remainder still exceeds context.
- [x] Keep non-TTY one-shot inference unchanged.

### Real KV cache and TurboQuant

- [ ] Implement lossless full caches before compression and prove cached logits/tokens match full-prefix recomputation.
  - [x] The seven reviewed dense-v5 preset topologies have a preallocated lossless K/V cache with whole-logit/token parity across decode, truncate, reset, context limits, and interleaved sessions; QK-norm, softcap, and selected MoA activation also have independent formula/topology proof.
  - [x] Require explicit dense resident fields and independently reject disconnected, relocated, duplicated, or bypassed QK-norm/softcap port chains and incomplete/tampered MoA selection contracts in both Native IR capability proof and the C++ loader; cover multi-layer/multi-head formula/cache parity and all seven named materialized presets.
  - [x] Canonical LLaMA has an independently implemented RMSNorm/RoPE/GQA/SwiGLU/untied-head oracle with whole-logit `full` versus `off` parity, plus decode, truncate/refill, reset, interleaved-session, cancellation, and exact-prefix reuse coverage.
  - [x] Exact standard MoE has an independent Torch route oracle and migrated-
    graph parity proof across RMSNorm/RoPE/GQA, softmax top-k renormalization,
    routed experts, whole logits, decode, truncate/refill, reset, ownership, and
    exact-prefix reuse; TurboQuant remains explicitly false.
  - [ ] Implement and prove the family-specific lossless state matrices listed above.
- [x] Expose:

  - `--kv-cache auto|off|full|turboquant`
  - `--turboquant-profile mse-3.5|qjl-3.5`
  - `--turboquant-attention-backend cpu|tile-cuda`, with explicit strict
    sidecar/runtime/device options for `tile-cuda`
  - `auto` selects a lossless cache; TurboQuant remains explicitly opt-in.

- [x] Implement the paper-aligned codec: deterministic random rotation, Lloyd–Max centroids, vector norms, genuinely packed mixed-bit indices, and optional one-bit QJL residual correction for key/query inner products. Values use the MSE reconstruction path; QJL is not applied to weighted value accumulation. Base the implementation on the [TurboQuant paper](https://arxiv.org/abs/2504.19874) and [Google Research overview](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/).
- [x] Add a portable reference implementation and native CUDA/Tile kernels that attend directly over compressed storage; never materialize the entire dequantized cache during decode.
  - [x] Add the dependency-free portable reference and reviewed-dense native CPU packed cache; score keys and accumulate values directly from per-row compressed storage without materializing the cache matrix.
  - [x] Add an additive Tile-sidecar CUDA feature ABI that consumes CPU-v1
    packed MSE/QJL records directly, supports MHA/GQA plus the exact current
    row, and uses deterministic chunked online softmax through 16K without a
    dequantized cache matrix. Fresh default fast-math/TK and strict sidecars
    pass all `19` live RTX 5090 tests against portable and native CPU oracles,
    including dimensions 8/64/128, 1023/1024/1025 and 4K/16K boundaries,
    repeated-launch counters, and invalid descriptors.
  - [x] Wire that feature ABI into reviewed-dense resident sessions behind
    explicit `turboquant_attention_backend=tile-cuda`, leaving absent/`cpu`
    payloads unchanged. The model shares device tables by profile; sessions own
    streams, packed device caches, and scratch. CPU encodes/uploads each newly
    committed row, CUDA exclusively scores/reconstructs historical compressed
    attention, and telemetry proves launches/uploads/transfers plus zero CPU
    compressed calls. Missing/stale/non-strict sidecars and unsupported
    geometry fail closed without fallback. Fresh RTX 5090 direct-binding and
    public-SDK lifecycle tests pass MSE and QJL (`4` live tests total). The
    synthetic transfer-inclusive 1K/4K/16K calibration below now proves the
    measurement path; representative trained-model/corpus evidence remains an
    explicit follow-up before any product performance claim.
- [x] Keep existing `kv_quant_pack/unpack` behavior unchanged because altering its dtype/shape would be breaking.
- [x] Make explicit TurboQuant fail closed for unsupported family state, head geometry, ABI, or kernels; never silently substitute a full cache.
- [x] Preserve strict FP32 behavior for temperature zero with `auto`/`full`. With explicit TurboQuant at temperature zero, use deterministic model computation plus deterministic lossy cache encoding: repeated runs must match, but FP32/TurboQuant logits and tokens are not promised to match.
- [x] Tag compute telemetry accordingly: requested/effective cache, profile, strict-model-compute flag, lossy-cache flag, cached tokens, actual and uncompressed bytes, compression ratio, prefix reuse, fallback reason, and decode rows processed.

## 3. `nfn infer --serve` and OpenAI contracts

- [x] Add a standalone inference-only FastAPI app and lean `[serve]` dependency extra. Do not initialize the editor database, cookie auth, or persistence worker.
- [x] Add `nfn infer --checkpoint MODEL --serve` with `--host`, `--port`, `--served-model-name`, `--state-db`, queue/session limits, cache flags, `--api-key-file`, and `NFN_INFER_API_KEY`. `--queue-capacity` bounds waiters; `--session-limit` independently bounds all admitted running-plus-queued request-session reservations, defaults to `queue_capacity + 1`, and returns a distinct OpenAI-shaped 429 when reached.
- [x] Default to `127.0.0.1:8000` without auth. Require a Bearer key for non-loopback binding unless `--allow-unauthenticated-remote` is explicitly supplied.
- [x] Load and validate the model, tokenizer, Native IR, chat template, resident binding, and cache ABI before opening the listening socket.
- [ ] Expose:

  - `GET /health`
  - `GET /v1/models`
  - `GET /v1/models/{model}`
  - `POST /v1/chat/completions`
  - Responses create/retrieve/delete, input-item listing, input-token counting, cancellation, and compaction
  - Conversation create/retrieve/update/delete and item create/retrieve/list/delete

  Match the current official [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create), [Models](https://developers.openai.com/api/reference/resources/models/methods/list), [Responses](https://developers.openai.com/api/reference/python/resources/responses/methods/create), and [Conversations](https://developers.openai.com/api/reference/resources/conversations/methods/create) contracts.

  - [x] Mount every route listed above when `--state-db` is configured, including scoped local lossless compaction and all listed Conversation item resources.
  - [x] Match the current response-delete resource shape and the `after`/`limit`/`order` cursor contract for response input-item and Conversation item list routes.
  - [ ] Prove full current-official shape/SDK compatibility; local compaction deliberately returns a NeuralFn-local durable reference rather than portable OpenAI ciphertext or a token-reducing summary.

- [x] Implement real Chat Completions SSE chunks ending in `data: [DONE]`.
- [x] Implement Responses semantic SSE events with stable IDs and increasing sequence numbers; terminate with `response.completed`, `response.incomplete`, or `response.failed`, not `[DONE]`. Follow the official [Responses streaming-event lifecycle](https://developers.openai.com/api/reference/resources/responses/streaming-events).
- [ ] Support text messages, instructions, `previous_response_id`, conversation state, metadata, truncation, usage, function tool definitions/calls/results, structured JSON-schema output, `store`, and `background`.
  - [x] Support the bounded text subset: messages, instructions, scoped `previous_response_id`, conversation state, metadata, truncation, usage, `store`, and durable `background`/cancel.
  - [x] Add the honest bounded constrained slice: Responses-only buffered,
    stored, foreground, greedy output for strict flat required-scalar JSON
    schemas through `json-schema-ascii-byte-greedy-v1`; plus exactly one forced
    strict client-executed function through
    `responses-forced-function-call-v1` and a separate typed string
    `function_call_output` continuation. Capability requires exact artifact
    feature metadata, binding `current_logits_exact_prefill`, artifact-selected
    presentation, and complete byte-exact printable-ASCII tokenizer preflight.
    The decoder masks before commitment, selects from `current_logits`, and
    commits only the exact allowed prefix through `prefill`; it never validates
    free-form output after ordinary decode. Typed lineage rejects corrupt,
    unknown, mismatched, duplicate, and already-resolved calls. Focused engine
    plus serving regression passes 95 tests.
  - [ ] Extend beyond the bounded slice: nested/array/general JSON Schema,
    auto/required/multiple/parallel/custom/hosted tools, tool history in
    Conversations or compaction/counting, constrained stream/background work,
    and Chat Completions tools remain fail-closed.
- [ ] Function tools are client-executed: NeuralFn produces and consumes tool-call items but does not execute the function. Require a manifest tool template and constrained decoder.
  - [x] Prove that contract for the single forced Responses function profile:
    emit stable `fc_`/`call_` items, persist typed history, accept only the
    client-owned result linked through `previous_response_id`, and perform zero
    server-side function execution. The parent remains open for the broader
    official tool-choice and streaming surface.
- [x] Persist stored responses, conversation items, and background jobs in a versioned local SQLite database using WAL and restrictive file permissions. Scope records by API-key fingerprint; keep GPU cache snapshots ephemeral and reconstruct them from stored tokens.
  - [x] Advance the semantic fence to schema version 3 for durable typed
    function-call/result history. Version-1/2 databases migrate in place
    without row rewrites; older binaries then reject v3 rather than silently
    discarding the new item types.
  - [x] Advance the state store to schema version 4 with monotonic conversation
    item revisions. Preparation snapshots ordered items/revision together;
    foreground terminal response/output/conversation publication and
    background job/event terminalization CAS that revision atomically. A stale
    branch is `conversation_conflict` (buffered 409 or semantic
    `response.failed` after stream/background start). Version-1/2/3 stores
    migrate in place with existing history at revision zero; require a backup
    for rollback because older binaries reject v4. Legacy queued conversation
    jobs cannot recover their historical snapshot and fail with
    `conversation_snapshot_unavailable`; legacy previous-response-only jobs
    reconstruct only completed/incomplete lineage or fail with
    `response_lineage_unavailable`.
- [x] On restart, preserve queued background jobs and mark interrupted in-progress jobs failed with `server_restarted` rather than risk duplicate generation.
- [ ] Use copy-on-write prefix state for branching `previous_response_id` or conversation histories.
  - [x] Audit the actual resident ownership boundary. At that baseline,
    Responses/Chat created a fresh session, prefilled the complete rendered
    prompt, and closed it; SQLite stored JSON/token-derived history only.
    Dense/LLaMA/MoE full, packed CPU TurboQuant, and Tile-CUDA buffers were flat
    per-session allocations, and the ABI had no clone/snapshot/fork operation.
    Retaining tokens or reusing one mutable LCP cursor would not preserve
    alternate branch tails and must not be called COW.
  - [x] Add the first exact dense full-cache `session_prefix_cow` v1 feature
    ABI and public `model.fork_session(source, token_count=None, seed=0)`.
    Effective proof is restricted to resident-ready dense-v5 lossless-cache
    artifacts plus the exact binding capability/callable. Native K/V and
    final-hidden allocations use whole-storage sharing and parent/child
    detach-before-prefill/decode-write; token history, logical length, RNG,
    cancellation, reset/truncate, counters, and close remain independent.
    Short-prefix tail isolation, divergent parent/children, both writer paths,
    lifecycle/rejection, and telemetry are covered by a freshly rebuilt
    binding. Model close now fences native create/fork registration and newly
    admitted session work, drains every pre-fence handle exactly once, and
    keeps duplicate close nonblocking to avoid operation-lock deadlock; the
    deterministic two-session race is covered. LLaMA/MoE/Tile regressions
    remain green. This is an SDK/native primitive only, not serving reuse.
  - [x] Extend the same exact whole-storage ownership contract to canonical
    LLaMA and standard-MoE full-cache sessions. Their distinct GQA profiles are
    jointly gated by checkpoint format, artifact ABI, binding profile
    inventory, and callable. Standard-MoE preserves its model/session wrapper
    while delegating cache ownership to its contained LLaMA implementation;
    cross-kind forks fail closed. Partial-prefix tail isolation, parent/child
    detach paths, close survival, telemetry, and independent RNG/cancel state
    pass the freshly rebuilt combined 151-test Native-IR, SDK, dense, LLaMA,
    and MoE resident matrix. Failed post-detach prefill/decode also restores
    the original shared allocation and telemetry before cancellation returns.
  - [x] Extend the native/SDK session-fork primitive to the reviewed dense CPU
    packed TurboQuant cache. The separately gated artifact capability and
    public effective capability are
    `session_prefix_cow_cpu_turboquant`; its exact ABI profile is
    `dense-cpu-turboquant-mse-qjl-packed-kv-final-hidden-v1`, operation
    `fork_session`, backend `cpu-reference-packed`. Effective support requires
    the artifact record, binding boolean, exact binding profile inventory, and
    callable to agree. A source must be dense with effective cache
    `turboquant`, profile `mse-3.5` or `qjl-3.5`, and CPU attention; a
    Tile-configured model/session rejects. Parent and child share packed K/V
    plus lossless final-hidden whole-capacity storage until the first
    prefill/decode append. Logical truncate/reset does not detach, and a
    failed/cancelled append restores the shared stores plus ownership/detach
    telemetry. The freshly rebuilt dense/Native-IR/SDK matrix passes 130
    tests, LLaMA/MoE regressions pass 27, and an independent rebuilt-binding
    audit passes 21 focused cases. The default Tile module retains 8 passing
    CPU tests and 4 opt-in skips; an attempted live run stopped before session
    creation at CUDA status 35, so it supplies no Tile-COW evidence. This
    checked sub-slice is a session-fork primitive only, not serving prefix
    reuse.
  - [ ] Extend shared immutable prefix ownership to Tile device K/V while
    keeping stream/scratch private, prove live GPU detach semantics, and admit
    Tile to serving only after the joint device-COW gate passes.
  - [x] Add the bounded serving slice for foreground HTTP Responses. The
    default-zero `--prefix-cache-capacity` entry limit requires the state store
    plus jointly proven dense/LLaMA/standard-MoE full COW or reviewed dense CPU
    TurboQuant COW; cache-off, missing proof, and Tile reject. Scope-local
    response/conversation-revision aliases select candidates, but reuse forks
    only an independently verified exact token LCP. The deterministic LRU has
    lease-safe eviction, cold restart, whole-scope deletion purge/epoch fences,
    and a single-process/in-band mutation boundary. Only stored foreground
    completed/incomplete results admit after atomic durable finish;
    `store:false` may hit but cannot admit, while Chat, background, failed,
    cancelled, and public `execute()+finish()` paths stay cold. Usage bounds
    cached tokens by native rows and counts newly written prompt rows only.
    Previous-response lineage is revalidated before finish and fails with
    `response_lineage_conflict` when ancestry changes. Shutdown awaits
    background/foreground drivers and drains the queue before cache/model/state.
  - [ ] Prove unique physical-byte accounting across COW-shared host and Tile
    device allocations. Current health byte fields are deliberately labeled
    sums of per-session capacity observations and may represent one shared
    allocation more than once; they are not a unique physical-memory claim.
- [x] Reject image/audio/file inputs, OpenAI-hosted tools, or unsupported reasoning/tool/structured-output modes with an OpenAI-shaped capability error when the loaded native text model lacks them.
- [x] Explicitly exclude legacy `/v1/completions`, Responses WebSocket, Realtime, hosted web/file/code tools, vector/file stores, and beta multi-agent Responses from this task.
- [x] Normalize validation and runtime failures into OpenAI error envelopes; cover 400 validation/context errors, 401 auth, 404 model/state IDs, 429 queue saturation, cancellation, and 500 internal failures.

## 4. Verification, rollout, and documentation

### Required verification

- [ ] For every shipped text preset/family, run a tiny real train step proving every expected architecture tensor is optimizer-updated, save/reload the production checkpoint, execute the native forward, and generate output.
  - [x] Complete the canonical LLaMA graph-authored live train/migrate/reload/
    generate slice. The exact tiny graph completed one real RTX 5090 optimizer
    step, wrote and passed inspection of its source-SHA-bound 11-tensor v2
    checkpoint, migrated losslessly, loaded through a freshly rebuilt resident
    extension, and generated `3,3` through the ordinary raw-token full-cache
    CLI. This closes the canonical LLaMA slice only; the every-preset parent
    remains open.
- [ ] Compare migrated graph and native logits/loss on deterministic tiny fixtures before deprecating the graph path.
  - [x] Prove the canonical LLaMA slice by SHA-binding the exact migrated source
    graph, importing every native ABI tensor (including packed gate/up and
    padded-vocabulary handling), comparing all prompt-prefix graph logits with
    both resident-native logits and an independent oracle, and matching graph
    cross-entropy loss. Remaining-family migration parity stays open.
  - [x] Prove the six graph-equivalent reviewed dense-v5 profiles (`gpt2`,
    `gpt2_megakernel`, `gpt2_zloss`, `gpt2_qknorm`, `gpt2_stable`, and
    `gpt2_softcap`) by SHA-binding and losslessly migrating each exact source,
    accounting for every ABI tensor in the compiled Torch graph (including
    packed-QKV splitting and padded vocabulary), and matching every prompt-
    prefix resident logit against both that graph and an independent formula.
    Match CE or the exact configured z-loss reconstructed from resident logits.
    Source-bound `gpt2_moa` remains open for non-GELU selected activations
    because its authoring graph still contains a literal GELU stage while the
    committed selection lives in sibling checkpoint metadata.
  - [x] Prove the exact standard-MoE slice by SHA-binding the migrated graph,
    importing every root/attention/router/packed-expert native tensor into that
    graph, matching every prompt-prefix resident logit, independently rebuilding
    CE plus the configured all-expert router loss, and verifying router
    gradients. Other families remain open.
- [ ] Test full-cache versus recompute parity across MHA/GQA, absolute/RoPE positions, sparse windows, multiple layers, context boundaries, reset/truncate, two interleaved sessions, and cross-turn prefix reuse.
  - [x] Cover the canonical multi-layer GQA/RoPE LLaMA slice through context boundaries, reset/truncate/refill, two interleaved sessions, and SDK exact-prefix reuse; MHA/absolute coverage is supplied by the reviewed dense slice.
  - [x] Cover the exact multi-layer standard-MoE GQA/RoPE slice through
    full/off decode parity, reset/truncate/refill, ownership isolation, and SDK
    exact-prefix reuse.
  - [ ] Add sparse-window coverage and the remaining family/state geometries before closing the matrix.
    - [x] Prove the existing recompute semantics without promoting a resident
      cache: all four sparse variants match full-prefix slices for N=19 GQA
      left/middle/right query chunks, exact window/block/sink/stride key sets,
      partial blocks, excluded-history perturbations, and long-history logical
      cardinality. The public Tile extension now builds with its explicit
      native-train include path and passes 12 real strict RTX 5090 sparse
      launches plus four 1025-key rejection tests. The raw float32 C ABI also
      rejects sparse `seq_k > 1024` before launch and preserves the inclusive
      1024 boundary (`2` compiled sidecar tests).
    - [ ] Implement a genuine resident sparse cache with physical eviction,
      bounded K/V allocation, reset/truncate/interleaving/cross-turn proof, and
      exact graph-owned sparse geometry before closing this item. NSA's strided
      visible-key set intentionally grows with history, and the current native
      family runner still hard-codes one preset geometry.
- [ ] Test hybrid state isolation for Mamba, TTT, universal recurrence, HNet patching, seq2seq encoder/cross-cache, and diffusion generation.
- [x] Add TurboQuant goldens for rotation, centroids, norms, 3-bit straddling, mixed-bit packing, deterministic seeds, QJL statistical behavior, exact byte accounting, and CPU-reference/CUDA agreement.
  - [x] Cover every listed portable golden plus byte-for-byte/numerical portable-to-native-CPU agreement, both resident profiles, deterministic decode, truncate/reset/cancel, and exact telemetry.
  - [x] Cover exact source-bound MoA with the same native packed CPU cache:
    every canonical selected activation agrees across recompute, full, and both
    TurboQuant profiles without re-running training-time candidate probes.
  - [x] Add CPU-reference/CUDA/Tile agreement. The live 19-test gate rebuilds
    both sidecar modes and proves both profiles, MHA/GQA, current-only and long
    chunked attention, deterministic replay, launch counters, and validation.
- [x] Benchmark full versus TurboQuant cache at 1K/4K/16K supported contexts on the RTX 5090, recording peak VRAM, time to first token, decode tokens/sec, cache bytes, perplexity delta, and greedy-token agreement. Make no quality-neutral or speedup claim unless measured.
  - [x] Add the fail-closed
    `tools/bench_native_resident_turboquant.py` v1 harness. Every mode/context
    uses separate fresh timing and quality/VRAM workers through the public SDK;
    Tile requires positive sampled device-global VRAM, launch, upload, H2D/D2H,
    and zero-CPU-call evidence. Six focused tests and CPU real-binding smoke
    coverage pass.
  - [x] Run full, MSE-CPU, QJL-CPU, MSE-Tile, and QJL-Tile at 1K/4K/16K on an
    idle RTX 5090. The bounded nontrivial one-layer/dimension-2/vocabulary-4
    synthetic fixture used fixed 16K capacity, no warmup, one timing sample,
    16 greedy tokens, and a 128-token quality tail. At 16K, TTFT was
    1.039/10.612/12.698/69.612/69.877 seconds and decode throughput was
    7629/774/645/120/119 tokens/sec in the order above. Tile measured 4 MiB,
    matched its CPU profile exactly, and made zero CPU compressed calls; full
    agreement was 16/16 free-running and 128/128 teacher-forced except 127/128
    at 4K. Signed synthetic perplexity deltas were approximately
    -3.31e-5/-3.60e-5/-3.70e-5 at 1K/4K/16K. The run explicitly makes no
    speedup or quality-neutrality claim.
  - [ ] Before making any product performance or language-quality claim, rerun
    the same harness with a representative trained 16K artifact, its matching
    tokenizer/corpus, warmups, and repeated timing samples. The synthetic
    acceptance proves mechanics and exposes launch/transfer cost only.
- [x] Assert temperature-zero full-cache exact repeatability and temperature-zero TurboQuant repeatability with explicit `lossy_cache` telemetry.
- [x] Test CLI default transcript mode, initial-turn retention, system/tool
  history, reset/mode changes, stop delimiters, trimming, and oversized-turn
  errors across every effective text adapter. The common driver is exercised
  through ordinary graph, semantic graph, graphless Parameter Golf, and
  resident-artifact paths; catalog guards bind it to all 66 shipped graph
  presets and all 12 execution-ready native aliases while `gpt2_diff` and the
  other unproved selectors remain explicitly excluded. Developer/tool items
  are covered at the public role-message renderer/resolver boundary because
  the terminal has no tool-execution or structured-history injection command.
  Raw checkpoints, compiled token-ID inference, `--prompt-tokens`, and direct
  `infer_*.py` compatibility helpers remain explicitly one-shot and are not
  transcript adapters. This matrix found and fixed stateless legacy turns
  dropping `--system-prompt`; 39 focused tests,
  the 48-test combined legacy guard/transcript slice, and the
  98-test/10-subtest core CLI suite
  pass.
- [ ] Run live tests with the official Python OpenAI SDK for Chat Completions, Responses, streaming, function tools, structured outputs, stored IDs, conversations, background/cancel, SQLite restart, auth, and error classes.
  - [x] Re-audit and pin a fail-honest `openai==2.44.0` typed-client slice
    covering Models, buffered/streamed Chat Completions,
    buffered/semantic-streamed Responses, stored IDs/input items/token
    counting/lineage/local compaction, Conversations/items, background cancel,
    auth, SQLite close/reopen ID persistence, and
    `400`/`401`/`404`/`409`/`429`/`500` SDK exception classes. It additionally
    proves `client.responses.parse(..., text_format=PydanticModel)` populates
    typed `output_parsed`, `openai.pydantic_function_tool(...)` produces a
    typed `ParsedResponseFunctionToolCall` with parsed arguments, the client
    executes the function, and a separate typed `function_call_output`
    continuation returns ordinary assistant text. Chat tools/response formats,
    unsupported constrained modes, wrong call IDs, and all broader selections
    remain fail-closed before generation. Exact cached-SDK run: 18 SDK tests;
    combined with 55 ASGI serving tests, 73 passed. The module skips rather
    than claiming evidence when that exact optional SDK is unavailable.
  - [x] Run the pinned SDK through a real loopback Uvicorn HTTP/1.1 socket,
    proving Models, buffered Chat, Chat SSE framing/usage, Pydantic structured
    Responses parsing, ordered semantic Responses SSE, a forced Pydantic
    function plus client-owned result continuation, Bearer-auth exception
    mapping, typed `400`/`404` errors, synchronous and `AsyncOpenAI` semantic
    streaming, stored background-stream cursor resumption, foreground stream-
    close cancellation/session disposal, background stream-close continuation,
    and graceful server/runtime/thread shutdown. An optional tenth case loads a
    fresh resident binding and strict sidecar, serves a tiny synthetic dense-v5
    artifact on the RTX 5090, and requires a positive Tile-CUDA attention launch
    count. The expanded socket module passes `12` cases with the opt-in
    resident-CUDA case skipped by default; both SDK modules pass `30` with that
    one skip, and the resident case has separately passed live. Real transport
    additions prove explicit-CA TLS verification, an observed HTTPS `CONNECT`
    tunnel, and ALPN-negotiated HTTP/2 Models plus Responses SSE. The test HTTP/2
    edge buffers its completed upstream HTTP/1.1 SSE response, so incremental
    reverse-proxy forwarding and direct Uvicorn HTTP/2 remain unproved.
  - [ ] Full-current optional fields, broader schema/tool modes,
    production direct/incremental HTTP/2 proxy behavior, representative trained and every shipped
    model/tokenizer artifact remain open; the parent is not closed by these
    bounded typed-client slices or the tiny synthetic CUDA proof.
- [x] Prove serving loads the model once and performs no subprocess spawn after startup; test disconnect cancellation, poisoned-session disposal, bounded-queue behavior, and prefix isolation.
  - [x] Prove one resident load, zero post-start subprocess spawns, bounded-queue saturation, request-session/prefix isolation, and cooperative in-flight native cancellation/reset.
  - [x] Add disconnect-driven cancellation and explicit poisoned-session disposal coverage; after a native failure, close the failed session, recover the persistent single worker, and serve the next request from a fresh session.
- [x] Run the mandatory preset gate:

  `python -m pytest tests/test_template_presets.py -x -q`

- [x] Add an all-preset-pairs variant-resolution test, native binding/ABI builds, focused native family suites, lean-import tests, editor production build, documentation link checks, and `git diff --check`.
  - [x] Current bounded evidence: mandatory preset gate `34 passed`;
    task-specific CLI/MCP/editor/native/serving/TurboQuant suite `221 passed`;
    canonical LLaMA graph-planner/CLI/SDK/server/checkpoint/migration/resident/
    dependency suite `121 passed`; standard-MoE focused planner/CLI/server/
    checkpoint/migration/resident suite `117 passed`, combined Native IR/
    inference/CLI/server regression `153 passed`, and expanded resident-MoE
    parity/lifecycle suite `13 passed`; focused serving/state suite `54 passed`;
    official-SDK slice `7 passed`; fresh LLaMA and standard-MoE host family
    builds plus direct plan/SHA/unknown-argument safety probes; fresh Tile
    library exporting the router-auxiliary ABI; resident binding build; direct
    TypeScript plus Vite production build (`998` modules); scoped local
    documentation-link and whitespace checks. A bounded live RTX 5090
    acceptance additionally completed one exact graph-authored CUDA optimizer
    step for canonical LLaMA and standard MoE, inspected 11- and 12-tensor
    graph-bound production checkpoints, migrated both losslessly, loaded both
    through the freshly rebuilt resident binding, and generated `3,3` through
    ordinary raw-token full-cache CLI. Post-fix focused regressions passed `18`,
    `58`, and `88` tests. The additive TurboQuant CUDA attention ABI passed all
    `19` live portable/native-CPU agreement tests against freshly rebuilt
    default fast-math/TK and strict sidecars through a 16K total context. Its
    explicit resident bridge additionally passed direct-binding and public-SDK
    MSE/QJL lifecycle tests on the RTX, with positive GPU launch/transfer
    telemetry and zero CPU compressed-attention calls. The
    isolated standard-MoE executable passes dependency inspection; the broad
    lean audit is currently red only because unrelated shared `build/` trainers
    are stale. The synthetic transfer-inclusive resident benchmark path is now
    exercised at 1K/4K/16K; remaining families and representative
    trained-model/real-corpus performance evidence stay open.

### Rollout and compatibility

- [x] Land behind capability metadata and advertise a model only after its real adapter passes parity and checkpoint gates. Native IR derives resident/cache/serve bits from topology-specific proofs plus a bound checkpoint, and the server independently requires `capabilities.serve=true` and `model.text_generation=true` before resident load or catalog advertisement.
- [ ] Deliver in gated milestones—core IR/resident engine, dense and standard transformer adapters, specialized/hybrid adapters, TurboQuant, editor migration, then serving—but do not mark the todo complete until every shipped text family passes.
- [x] Keep dense v5 checkpoints and existing one-shot CLI calls readable. Require a rebuilt resident binding for serving and report missing/stale resident ABI plainly; retain the appropriate strict Tile sidecar requirement on legacy one-shot native sampling rather than imposing it on the standalone CPU resident engine.
- [ ] Add explicit `CHANGELOG.md` breaking-change entries for the interactive default becoming transcript mode, diagnostic native-family sampling being replaced by real inference, graph-runtime deprecation, and any Native IR/checkpoint or native ABI migration.
  - [x] Record the interactive transcript default and its `--chat-mode
    stateless` rollback, plus the legacy graph/Parameter Golf deprecation
    warning, exact migration command, and fail-closed serve/TurboQuant change.
  - [x] Record the additive Native Execution v1 tensor-layout field, strict
    graph-bound LLaMA/MoE/MoA checkpoints and resume fences, Responses state-v3
    migration, sparse raw-ABI bound, and additive TurboQuant Tile feature ABI
    with explicit migration/rebuild guidance where required.
  - [x] Record canonical LLaMA and exact standard-MoE ordinary resident
    inference replacing diagnostic transition sampling, including their strict
    checkpoint and binding requirements.
  - [ ] Add the corresponding breaking/migration record as each remaining
    family's diagnostic transition sampler is replaced; those adapters are not
    yet promoted, so the parent remains open.
- [x] Preserve graph CLI rollback until migrated-native parity is proven; do not remove graph files, checkpoints, or editor authoring support in this task. Legacy graph and graphless Parameter Golf inference remain reachable with exact migration guidance, while native-only serve/TurboQuant requests fail before altering the source artifact; editor authoring and graph serialization remain intact.

### Documentation done criteria

- [ ] Update `README.md` and append the complete implementation, migration, verification, and breaking-change record to `CHANGELOG.md`.
  - [x] Update both files for the exact standard-MoE graph adapter, auxiliary
    loss, strict checkpoint/migration, resident/ordinary inference, Native IR
    tensor-layout addition, migration guidance, bounded verification, and open
    bounded live-CUDA acceptance and its remaining all-family boundary.
  - [x] Update both files for source-bound `gpt2_moa` checkpoint migration and
    CPU resident/cache/TurboQuant support, plus the breaking fail-closed
    `gpt2_diff` graph-execution gate and its learned-lambda/packed-path blockers.
  - [x] Update both files for learned `gpt2_diff` metadata v2, strict
    continuation and training-shard/numerics identity, create-only DONE-gated
    final export, public SDK/CLI forwarding, bounded five-binary evidence, and
    the explicit still-blocked migration/resident boundary. Record breaking
    guidance for version-1 resume rejection and overwrite refusal.
- [x] Update `docs/cli.md`, `cli/README.md`, Python SDK inference/config documentation, framework inference/training/template guides, REST API indexes and serving pages, server configuration/authentication, and editor training/runtime documentation.
- [x] Document the new OpenAI routes, SQLite retention/security, remote auth rule, Native IR schema, graph migration workflow, cache capability matrix, TurboQuant precision semantics, and temperature-zero distinction.
- [x] Update `.cursor/skills/neuralfn-cli`, `neuralfn-python-sdk`, and `neuralfn-torch`; update MCP documentation and `.cursor/skills/neuralfn-mcp` because native training becomes an accepted MCP workflow.
- [x] Update the required dedicated SDK, REST, MCP, server/editor, and skill documentation and focused tests for the public surfaces introduced so far. No public builtin or template preset was added, so the builtin catalog and hardcoded template dropdown did not change.
  - [x] Re-trigger the documentation gate for the bounded public
    tools/structured-output slice: update README/CHANGELOG, REST and server
    contracts, Python SDK/Native IR and framework pages, CLI guidance, and the
    CLI/Python agent skills. Record the schema-v3 rollback warning, exact
    artifact/binding/tokenizer gates, bounded schema, client-executed function
    continuation, explicit exclusions, and official SDK 2.44 evidence. No MCP
    surface, builtin, template preset, or editor dropdown changed in this slice.
