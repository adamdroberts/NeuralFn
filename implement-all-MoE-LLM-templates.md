# NeuralFn: implement all MoE + dense LLM template production trainers TODO

Tracks the work needed so **every** shipped GPT template preset trains for real
via `nfn train --native-cuda`. Dense-GPT families (`gpt`, `gpt2`, `gpt3`,
`nanogpt`, plus the `gpt2-evo` delegate), all native LLaMA/MoE/JEPA families,
HNet byte-LM, diffusion, seq2seq, TTT, Jamba, and universal-transformer now
run production native loops. All shipped native families are compiled from
the single parameterized source `neuralfn/csrc/native_train/missing_native_train.cpp`
and run full-template CUDA-Tile forward/backward/AdamW loops over real
token/byte shards. Family-specific quantized, sparse, and megakernel deltas
remain part of the final preset audit.

**Definition of "production"** (what flips a family from diagnostic to
covered): full-template-geometry forward + backward + AdamW on Tile ABI
kernels, parameters resident in device buffers across all steps, real-shard
batches with gradient accumulation, resumable checkpoints in the existing
`nfn-native-family-float32-parameter-state-v1` sidecar schema
(`neuralfn/native_family.py`), and the status flipped end-to-end (binary JSON,
build script, Python registry, CLI, tests, docs).

**Reference implementation:** `neuralfn/csrc/native_gpt2/nfn_gpt2_native_train.cpp`
— persistent device parameters, dlopen'd batched AdamW descriptors
(`adamw_step_many_with_device_scale_*`), grad accumulation, global-norm clip,
warmup+cosine LR, checkpoint/resume. **No new CUDA kernels are required**: all
needed Tile kernels (packed-QKV attention fwd/bwd, RMSNorm bwd, rotary,
`moe_swiglu_forward/backward`, `topk_route`, balance losses,
`causal_chunk_state[_backward]` for Jamba, fused AdamW-many, grad-accum,
norm-clip) already exist in `neuralfn/csrc/native_train/tile_ops.h`. The work
is host-side composition + persistence.

Per family, tick when each piece is in place:

- **step** — full-geometry `FamilyProductionStep` forward/backward on Tile ABI
- **persist** — `FamilyDeviceParameterStore` + `FamilyOptimizerState` carried
  across steps (no per-step alloc/init/free)
- **ckpt** — real trained f32 sidecar (`trained_parameter_elements ==
  parameter_elements`, true checksum) + resume + `--checkpoint-every-steps`
- **flip** — build-script production define + `print_json`/loop status +
  registries (`neuralfn/native_train.py`, `cli/nfn.py`)
- **test** — `tests/test_native_gpt2.py`, `cli/tests/test_native_family_infer.py`,
  `tests/test_template_presets.py` updated; persistence + resume unit tests
- **docs** — `docs/cli.md`, `README.md`, `CHANGELOG.md`

Recommended order: §1 → §2 mixllama → §3 deepseek-v4 → §4 moe-jepa-evo →
§5 semantic-router-moe → §6 jamba → §7 dense families (llama first, since
mixllama-production = llama-production + router/experts). §8 dense-GPT audit
can run at any time.

---

## §0. Status matrix — template preset → native family → status

Source of truth: `neuralfn/config.py` (`SHIPPED_GPT_TEMPLATE_PRESETS`,
`TemplateSpec.sparsity`) and `neuralfn/native_train.py`
(`native_train_family_for_template_name`, `_NATIVE_TRAIN_MODEL_REGISTRY`).
61 presets total. Presets with family `dense-gpt` dispatch to
`nfn_gpt_native_train` and are already production ("implemented").

| Preset | Sparsity | Native family | Status |
|---|---|---|---|
| gpt2 | dense | dense-gpt | implemented |
| gpt2_megakernel | dense | dense-gpt | implemented |
| gpt2_moa | moe-attention | dense-gpt | implemented |
| gpt2_modern | dense | dense-gpt | implemented |
| nanogpt | dense | dense-gpt | implemented |
| nanogpt_megakernel | dense | dense-gpt | implemented |
| nanogpt_modern | dense | dense-gpt | implemented |
| mixllama | moe | mixllama | covered |
| mixllama_fast | moe | mixllama | covered |
| mixllama_fast_megakernel | moe | mixllama | covered |
| moe | moe | mixllama | covered |
| moe_modern | moe | mixllama | covered |
| deepseek_v3 | moe | mixllama | covered |
| deepseek_v4 | moe | deepseek-v4 | covered |
| moe_jepa_evo | moe | moe-jepa-evo | covered |
| moe_jepa_evo_modern | moe | moe-jepa-evo | covered |
| auxfree_moe_jepa_evo | moe | moe-jepa-evo | covered |
| semantic_router_moe | moe | semantic-router-moe | covered |
| semantic_router_moe_megakernel | moe | semantic-router-moe | covered |
| semantic_router_moe_modern | moe | semantic-router-moe | covered |
| semantic_moe_jepa_evo | moe | semantic-router-moe | covered |
| semantic_moe_jepa_evo_modern | moe | semantic-router-moe | covered |
| diff_semantic_moe_jepa_evo | moe | semantic-router-moe | covered |
| jamba | moe | jamba | covered |
| jamba_modern | moe | jamba | covered |
| llama | dense | llama | covered |
| llama_modern | dense | llama | covered |
| llama_fast | dense | llama | covered |
| llama_fast_megakernel | dense | llama | covered |
| llama_megakernel | dense | llama | covered |
| modern_norms_llama | dense | llama | covered |
| ternary_b158 | dense | llama | covered |
| ternary_b158_modern | dense | llama | covered |
| fp8_llama | dense | llama | covered |
| mxfp4_llama | dense | llama | covered |
| gemma3 | dense | llama | covered |
| diff_transformer | dense | llama | covered |
| longctx_sparse_llama | dense | llama | covered |
| qwen3_longctx | dense | llama | covered |
| kv_pca_llama | dense | llama | covered |
| kv_pca_llama_modern | dense | llama | covered |
| llm_jepa | dense | jepa | covered |
| llm_jepa_modern | dense | jepa | covered |
| dense_jepa_evo | dense | jepa | covered |
| dense_jepa_evo_modern | dense | jepa | covered |
| semantic_dense_jepa_evo | dense | semantic-dense-jepa | covered |
| semantic_dense_jepa_evo_modern | dense | semantic-dense-jepa | covered |
| dyt_geglu_semantic_dense_jepa_evo | dense | semantic-dense-jepa | covered |
| jepa_semantic_hybrid | moe | semantic-dense-jepa | covered |
| jepa_semantic_hybrid_megakernel | moe | semantic-dense-jepa | covered |
| jepa_semantic_hybrid_modern | moe | semantic-dense-jepa | covered |
| seq2seq | dense | seq2seq | covered |
| seq2seq_modern | dense | seq2seq | covered |
| diffusion | dense | diffusion | covered |
| diffusion_modern | dense | diffusion | covered |
| ttt_llama | dense | ttt-llama | covered |
| ttt_llama_modern | dense | ttt-llama | covered |
| universal_llama | dense | universal-llama | covered |
| universal_llama_modern | dense | universal-llama | covered |
| hnet_lm | dense (bytes) | hnet-lm | covered |
| hnet_lm_modern | dense (bytes) | hnet-lm | covered |

Registry note: `moe-jepa-evo-modern` and `auxfree-moe-jepa-evo` are distinct
registry entries but share the `nfn_moe_jepa_evo_native_train` binary — a §4
flip covers all three entries.

### Current implementation milestone

The exact `llama` family now has a full-geometry Tile-ABI composition behind
the two explicit production gates. It keeps token embedding, configured dense
RMSNorm/QKV/RoPE/causal attention/SwiGLU layers, final RMSNorm, and the LM head
on device, collects their gradients into the persistent optimizer state, and
reports `persistent_tile_llama_full_geometry_forward_backward` in the normal
build. The matrix above is now
covered by default; preset-specific quantized/sparse and megakernel deltas
remain part of the final audit. HNet, diffusion, seq2seq, TTT, Jamba, and
universal are promoted through the complete native production path below.

The full graph now applies each live `attention_norm.weight`, `ffn_norm.weight`,
and `final_norm.weight` through the native vector-binary Tile primitive rather
than silently using unparameterized RMSNorm. Its backward path derives affine
norm-weight gradients from the full device batch before collecting them into
the persistent optimizer map. Full geometry also follows the template's
separate Q/K/V projection and query/KV-head contract (defaults `4`/`2`), using
the existing reshape/merge-heads Tile primitives; `--num-heads`,
`--num-kv-heads`, and `--rope-theta` are surfaced in plan/checkpoint metadata.
The native build script includes these head-layout symbols and
`nfn_native_tile_vector_binary_float32` in every full-geometry symbol contract.
The full attention call also selects the shipped preset's sparse geometry for
`gemma3`, `longctx_sparse_llama`, `qwen3_longctx`, `deepseek_v4`, and
`diff_semantic_moe_jepa_evo`, including backward arguments and plan metadata;
these presets no longer silently fall back to dense attention.

Native MLP sizing now follows the template family defaults when `--hidden-dim`
is omitted: LLaMA-style families use the `8/3` SwiGLU multiplier rounded to
`256`, while GPT-style families retain `4x`. `--mlp-multiplier` and
`--multiple-of` provide explicit geometry overrides, and the resolved values
are included in plan/layout metadata.

The same shared graph now has a standard-MoE route for `mixllama` and
`deepseek-v4`: live router top-k, per-layer expert gate/up/down SwiGLU,
residual composition, router/expert gradients, and persistent optimizer
handoff. These families now enable both production selectors by default;
`NFN_NATIVE_MISSING_FULL_GEOMETRY_TARGETS=mixllama,deepseek-v4` remains a
controlled target-selection override. JEPA and semantic-router objective branches select the same routed graph when their
template-specific buffers are present.

Dense `jepa` and `semantic-dense-jepa` now have a full objective branch for
live target-encoder/projector/predictor matrices and latent-MSE gradient
handoff. MoE-JEPA template names select the same branch and combine it with
live JEPA matrices.

The semantic-router full graph now also consumes loop-derived semantic target
rows through `semantic_vocab_projection.weight` and merges semantic CE
gradients. Its configured expert count must satisfy the shared-plus-vocab-plus-
free expert contract. The standard MoE, MoE-JEPA, and semantic-router
objective branches now share this full persistent graph. Specialty families
are isolated from the legacy sampled bridge, which assumes the diagnostic
global `qkv`/MoE layout; this prevents a full-specialized dispatch from
running an incompatible second objective. HNet,
diffusion, seq2seq, TTT, universal, and Jamba now have dedicated full native
paths and are promoted below.
The opt-in specialty paths now also compose the
shared full hidden-backbone step, so their added state/objective buffers and
backbone buffers both enter the persistent gradient map. Seq2seq remains on
its packed encoder/decoder path, while HNet has a dedicated full byte-patch
transformer path and is claimed complete by the §7.8 implementation below.

---

## §1. Shared production infrastructure (one-time)

New module `neuralfn/csrc/native_train/family_production_train.h` (header-only,
or `.cpp` added to `build_one`'s source list next to `token_shards.cpp` in
`tools/build_native_missing_trainers.sh`), reused by every family:

- [x] **FamilyDeviceParameterStore** — one device f32 buffer per
      `ParameterBufferSpec`; make `family_parameter_buffers()`
      (`missing_native_train.cpp` ~line 1193) config-driven (honor
      `cfg.experts`, `cfg.top_k`, `cfg.num_layers`, semantic dims instead of
      hard-coded 8 experts / 12 layers); deterministic init reusing
      `native_family_dense_base_value` (~line 13185) so
      `dense_parameter_state_reconstructable` stays honest;
      `load_from_sidecar(path)` for resume; `copy_to_host()` for checkpointing
      - Partial: `family_production_train.h` now provides the header-only
        `FamilyDeviceParameterStore` with one CUDA f32 allocation per buffer
        spec, deterministic host initialization callback support,
        named buffer lookup, typed device-parameter views,
        `load_from_sidecar(path)`, `copy_to_host()`, and host-sidecar writing.
        `missing_native_train.cpp` now also converts
        `family_parameter_buffers()` to `FamilyParameterBufferSpec` and exposes
        `build_native_family_production_state(...)`, which can initialize the
        store from `native_family_dense_base_value()` or a resolved sidecar.
        `family_parameter_buffers()` now resolves `model_dim`, `hidden_dim`,
        public `vocab_size`, and `padded_vocab_size` from CLI/config controls
        in addition to layers, experts, top-k, semantic dims, and
        layers-per-expert, so production-state byte counts no longer depend on
        a hardcoded 768/50304 geometry.
        Dataset loops report `production_state_runtime`; default diagnostic
        builds skip it, while `NFN_NATIVE_PRODUCTION_LOOP=1` builds must prepare
        the store before entering the loop. The production-step bridge now
        validates every expected buffer against that live store by name, offset,
        element count, trainable flag, allocation state, and total element count
        before writing optimizer gradients. The plan/runtime
        contract now also reports named production parameter-role counts for the
        base transformer, MoE, JEPA, semantic, seq2seq, diffusion, TTT,
        universal, HNet, and Jamba groups, and production bootstrap verifies
        those role bindings against the live store before training. The plan
        contract now selects a family production-step descriptor with required
        roles and planned full-geometry forward/backward stages, and the bridge
        validates that descriptor before writing gradients. Dataset loops
        instantiate descriptors through
        `make_native_family_production_step(...)` behind the shared
        `FamilyProductionStep` interface. The active descriptor implementation
        is now `persistent_sampled_lm_chained_block_routed_moe_jepa_semantic_specialized_bridge`, which reads live
        embedding/head parameters for a bounded sampled-LM objective and live
        attention/norm plus dense FFN block weights for bounded backbone and
        SwiGLU reconstruction objectives, including attention/norm target-row
        gradients plus dense-FFN input and target gradients into
        `token_embedding.weight`. It also carries sampled hidden rows
        forward across sampled layers through chained sampled RMSNorm with sampled attention/FFN/final-norm backward,
        sampled Q/K/V softmax attention with sampled RoPE plus
        sequence-local causal sampled-row attention, attention-out,
        FFN norm, and either dense SwiGLU or routed MoE expert SwiGLU before a
        live final norm, sampled LM-head CE, and target-embedding loss, so base-transformer, MoE,
        and LM-head gradients now depend on propagated per-layer hidden state
        rather than only independent local objectives. The chained dense and
        routed SwiGLU backward now distributes gate/up gradients across every
        sampled FFN input dimension used by the forward activation. Runtime JSON now reports
        `production_step_sampled_attention_row_count` and
        `production_step_sampled_causal_attention_context_count` so the sampled
        QKV/RoPE attention coverage is visible separately from the generic
        chained-block row/layer counters.
        For MoE layouts it also reads live router/top-k and expert SwiGLU
        weights for bounded score-selected, top-k-cardinality-bounded,
        top-k-scaled router classification with router-dimension-aligned
        route gradients and optional aux-free router-bias scoring,
        sampled multi-route expert reconstruction that combines all selected
        expert outputs by router probability while backpropagating
        router/expert-input and target-reconstruction gradients into
        `token_embedding.weight`, plus bounded selected-route balance
        gradients through the router/top-k path, and
        score-selected routed multi-expert combine gradients that weight
        sampled expert outputs by router probabilities while backpropagating
        combine input, route-count-independent target, and route-balance gradients into
        `token_embedding.weight`. For JEPA layouts it also reads live target-encoder,
        projector, and predictor weights for bounded latent prediction
        gradients. For semantic layouts it also reads live planner, alignment,
        semantic-vocab projection weights, and sampled shared/free semantic
        expert bias vectors for bounded semantic latent and semantic-class
        gradients, using loop-derived semantic target batches when semantic
        dataset loops provide them. For seq2seq, diffusion, TTT, universal, HNet,
        and Jamba layouts it also reads live specialty projection/state buffers
        for bounded parameter-dependent gradients with input and target
        embedding gradients; HNet additionally samples four-byte patches
        through `byte_patch_embed.weight` and `byte_patch_merge.weight`,
        backpropagates through both buffers, and reports
        `production_step_sampled_hnet_byte_patch_row_count`. It still needs to be
        replaced by real full-geometry family forward/backward steps. The bridge
        now validates that every trainable buffer has parameter-dependent sparse
        finite nonzero gradients and runtime JSON reports
        `production_step_parameter_dependent_gradient_buffer_count` plus
        `production_step_chained_block_layer_count` and
        `production_step_chained_block_row_count`; semantic production runs
        also report semantic target batch/row counters so the bridge can prove
        loop-derived semantic supervision was supplied and consumed. Sparse gradients are also
        filtered to finite nonzero entries before optimizer accumulation so
        production runs can prove both gradient-buffer coverage and chained
        block-state execution.
        Production-enabled builds now replace the sampled host LM objective
        with a live Tile-ABI token-LM path: real rows run through
        `token_embedding_u16`, LM-head linear logits, token cross-entropy
        backward, LM-head weight backward, and token-embedding weight backward,
        and report `production_step_tile_lm_row_count`. The full and specialty
        paths process LM logits in bounded 128-row chunks, accumulating hidden
        and LM-head gradients instead of allocating `rows * vocab` logits.
        Exact preset parity and fused megakernel deltas remain separate audit
        work.
- [x] **FamilyOptimizerState** — m/v buffers, batched
      `nfn_native_tile_adamw_step_many_with_device_scale_*` descriptor
      assembly (clone the dense trainer's strategy), grad-accum
      (`gradient_accumulate_float32`) and global-norm clip
      (`global_norm_clip_scale_float32`) helpers, warmup+cosine
      `scheduled_learning_rate` (already at `missing_native_train.cpp` ~247)
      - Partial: `family_production_train.h` now provides
        `FamilyOptimizerState` with per-buffer grad/m/v CUDA allocations,
        many-tensor descriptor arrays, `gradient_accumulate_float32`,
        `sumsq_partials_many_float32`, `global_norm_clip_scale_float32`,
        `adamw_step_many_with_device_scale_float32`, and shared cosine/warmup
        LR scheduling helpers. The shared production bootstrap now constructs
        this optimizer over the live parameter-store buffers when a production
        build enters a dataset loop. The checkpoint writer now routes sampled
        LM gradients through a concrete
        `FamilyProductionStep::forward_backward(...)` bridge, sets them into
        `FamilyOptimizerState`, and runs the shared many-tensor AdamW step
        before live-store sidecar writing. That bridge now computes
        parameter-dependent sampled-softmax gradients from live
        `token_embedding.weight` and `lm_head.weight`, adds bounded dense
        SwiGLU FFN gradients from live `layers.N.ffn_gate_up.weight` and
        `layers.N.ffn_down.weight`, adds a bounded chained block-state pass
        over sampled transformer layers through final norm, then strictly validates that every
        trainable parameter buffer has at least one parameter-dependent sparse
        gradient. LLaMA, dense JEPA,
        semantic dense JEPA, semantic-router MoE, shared standard-MoE/MoE-JEPA,
        Jamba, seq2seq, diffusion, TTT, universal, and HNet loops now also run
        this persistent sampled-LM plus chained block-state and dense-FFN production step during
        production-enabled train microbatches, accumulate sparse gradients across
        `batch_plan.grad_accum_steps`, run one AdamW update on the final
        accumulation microbatch, and report loop-time step, optimizer-step, and
        gradient counts. Sparse bridge outputs are finite/nonzero checked before
        optimizer accumulation, and `FamilyOptimizerState` rejects non-finite
        sparse scales, raw values, scaled values, and accumulated values before
        device-buffer writes. All emitted family binaries now enter the
        production/full-geometry path by default; the shared bridge remains
        the common implementation boundary for preset-delta work.
- [x] **FamilyProductionStep interface** — `forward_backward(batch, grads) ->
      losses`; one implementation per family (§2–§7)
      - Partial: `family_production_train.h` now defines
        `FamilyProductionBatchView`, `FamilyProductionLosses`,
        `FamilyProductionStepContext`, `FamilyProductionStepResult`, and the
        abstract `FamilyProductionStep::forward_backward(...)` contract over
        `FamilyDeviceParameterStore` + `FamilyOptimizerState`. The current
        sampled-LM/FFN bridge requires and validates the live parameter store,
        copies it to host to build parameter-dependent embedding/head and dense
        FFN gradients, and reports persistent-buffer validation counts; it also honors
        `accumulation_step` / `accumulation_steps` by delaying AdamW until the
        final accumulation microbatch. Each family now has a production-step
        descriptor with required parameter roles and planned stages, and a
        factory selects the descriptor-backed sparse implementation. Production
        macro builds now use the family-specific Tile-ABI composition inside
        that factory, with core and specialty selectors covered by default.
- [x] **Real checkpoint writer** — upgrade `write_native_family_token_model`
      (~line 13890) to write the host-copied trained parameters into the f32
      sidecar: `trained_parameter_elements == parameter_elements`, real FNV
      checksum via `mix_native_family_float_checksum`, schema byte-for-byte
      identical to `nfn-native-family-float32-parameter-state-v1` (readers:
      `neuralfn/native_family.py` `_native_family_parameter_state` ~350,
      `audit_native_family_checkpoint_template_coverage` ~242)
      - Partial: `family_production_train.h` now provides
        `write_family_full_parameter_sidecar()` and
        `FamilyFullParameterCheckpointInfo`, which write a host-copied
        contiguous f32 parameter vector, hash every float with the same FNV
        mixing pattern, and report
        `trained_parameter_elements == parameter_elements`. Production-enabled
        dataset loops now pass the ready bootstrap into checkpoint writing, so
        the writer applies persistent sampled-LM plus dense-FFN gradients through a concrete
        `FamilyProductionStep::forward_backward(...)` bridge, copies the live
        `FamilyDeviceParameterStore` to host, and emits that complete sidecar.
        Production-state checkpoint metadata labels that path as
        `live_family_device_parameter_store_float32_state`, and the default
        full-geometry family binaries report
        `optimizer_updated_full_architecture_parameter_persistence: true`.
        Preset-specific checkpoint parity remains part of the final audit.
- [x] **Resume path** — `--native-checkpoint <dir>` loads the f32 sidecar into
      the parameter store before the loop
      - Partial: `--native-checkpoint` / `--checkpoint` now resolve directory,
        model JSON, or `.f32` sidecar inputs and stream-validate the sidecar in
        `--print-plan`; `build_native_family_production_state(...)` can load
        the resolved sidecar into `FamilyDeviceParameterStore`, compute a
        checksum from the live device store after initialization/resume, and
        production builds call that bootstrap before entering the loop.
        `production_state_runtime` reports whether the checksum was computed and
        the checksum value, then reports post-step checksum count/latest/changed
        evidence after production-step calls, so resumed runs have observable
        proof that the sidecar seeded device state and that the resident store
        changed after AdamW or route-evo.
- [x] **`--checkpoint-every-steps` flag** in `Config` + parse_args
      - Partial: token and byte family dataset loops now call the native-family
        model writers after completed optimizer steps divisible by
        `checkpoint_every_steps`, using step-suffixed checkpoint prefixes while
        preserving the final checkpoint path. `--checkpoint-every-steps 0`
        disables these periodic intermediate family checkpoints.
- [x] **Status-flip plumbing** — new `-DNFN_NATIVE_PRODUCTION_LOOP=1` define
      enables persistent device state, while the separate
      `-DNFN_NATIVE_FULL_GEOMETRY_FORWARD_BACKWARD=1` define selects the full
      family step before `family_production_missing_requirements()` (~297)
      returns empty,
      `print_json` (~896–1032) then flips `status` →
      `native-trainer-covered` / `production_training_loop: true`; loop
      emitters (`print_moe_jepa_dataset_loop_json` ~15559,
      `print_semantic_router_moe_dataset_loop_json` ~16047,
      `print_single_substep_dataset_loop_json` ~18876, etc.) stop hard-coding
      `production_training_loop: false` in controlled diagnostic builds
      - Both native gates are enabled by default for every emitted family
        binary. Status, missing-requirement, and checkpoint metadata now agree
        on covered production behavior; target/all environment variables
        remain compatibility overrides for controlled builds.
- [x] **build_one production args** — `tools/build_native_missing_trainers.sh`:
      per-family, empty `missing_requirements` (6th arg), move
      `optimizer-updated-full-architecture-parameter-persistence` into
      `completed_requirements` (7th arg), add the production define
      - Partial: `build_one` accepts an 8th `production_loop` argument and a
        9th full-geometry argument, passing
        `-DNFN_NATIVE_PRODUCTION_LOOP` and
        `-DNFN_NATIVE_FULL_GEOMETRY_FORWARD_BACKWARD`, defaulting through
        `production_loop_for(...)`. Set
        `NFN_NATIVE_MISSING_PRODUCTION_LOOP_TARGETS=model_or_binary,...` or
        `NFN_NATIVE_MISSING_PRODUCTION_LOOP_ALL=1` for controlled target
        selection. Normal builds now default both gates to `1`; production
        binaries report
        `optimizer-updated-full-architecture-parameter-persistence` in
        `native_training_completed_requirements` and leave the missing list
        empty.
- [x] **bf16 shadow parameters** — wire the family trainers to the dense
      trainer's `adamw_step_many_...bf16_shadow` variants
      - `FamilyOptimizerState` now assigns bf16-shadow offsets for every
        trainable family parameter buffer, allocates a shadow arena when the
        Tile bf16-shadow AdamW symbol is present, and dispatches
        `nfn_native_tile_adamw_step_many_with_device_scale_bf16_shadow_float32`
        before falling back to the float32 many-tensor AdamW path. Family
        plan/runtime JSON reports bf16-shadow support and runtime enablement.
- [x] **CUDA-graph capture** of the production step (capture once, replay per
      optimizer step)
      - `FamilyOptimizerState` now loads optional CUDA graph runtime
        APIs and can capture/replay the device optimizer body
        (global-norm clip + many-tensor AdamW + trainable-gradient zeroing)
        when production-loop builds or
        `NFN_NATIVE_FAMILY_PRODUCTION_CUDA_GRAPH=1` enable it. Full LLaMA-derived
      device gradients, including affine RMSNorm diagonals, now accumulate
      directly into persistent optimizer buffers. Parameter-store H2D uploads
      now use ordered `cudaMemcpyAsync` when available, and default diagnostic
      reads remain wrapper-side/outside the retained graph body. The production
      step persists across batches and the retained forward/backward+optimizer
      graph replays across later matching optimizer steps.
	      Optimizer-only graph replay refreshes device-resident schedule scalars
	      when the Tile device-hyperparameter AdamW ABI is available and only
	      scalar-argument fallback graphs recapture on LR/schedule scalar changes.
	      Runtime and plan JSON
	      now report
	      `cuda_graph_capture_scope: "retained_forward_backward_optimizer_step"` and
	      `optimizer_cuda_graph_captures_zero_gradients: true`; captured optimizer
	      replay zeroes trainable gradients with `nfn_native_tile_fill_many_float32`
	      instead of host-to-device zero-copy staging. Runtime and plan JSON
	      now expose `full_step_cuda_graph_blockers` and
	      `full_step_cuda_graph_capture_ready: true` when no implementation
	      blockers remain. Temporary
	      workspace leases now replay from a warmed fixed sequence without mutating
	      the pool/active metadata vectors inside `forward_backward`; steady replay
	      steps now only arm/disarm the warmed lease cursors by default, with strict
	      begin/end validation available through
	      `NFN_NATIVE_FAMILY_TEMPORARY_REPLAY_VALIDATE=1`, so replay leases are no
		      longer reported as a full-step graph blocker. The blocker list is now
		      family-aware and empty for the shipped native-family production
		      builds: standard LLaMA/MoE/JEPA plus Jamba, seq2seq, TTT, universal,
		      HNet, diffusion, and semantic-family paths are all promoted to the
		      retained replay contract, so no family now adds the specialty branch
		      host-control blocker. A 13-family three-step runtime audit now
		      verifies `capture_count=1`, `replay_count=2`, and replay-ready
		      telemetry for LLaMA, MixLLaMA, MoE-JEPA, dense JEPA,
		      semantic-dense JEPA, semantic-router MoE, DeepSeek-V4, Jamba,
		      seq2seq, diffusion, TTT, HNet, and universal-transformer,
	      semantic targets now derive from the device uint16 token batch inside
	      `forward_backward` with
	      `nfn_native_tile_semantic_targets_from_tokens_u16_int64`; semantic
	      dataset loops now keep the production batch token-only, compute host
	      checksums without building full target arrays, and only derive a host
	      target sample for final JSON reporting,
	      the shared
	      LLaMA/full-family Tile helper now propagates the
	      production-step stream through its Tile forward/backward and direct
	      gradient-accumulation calls, and specialty Tile helpers now propagate the
	      same stream through their Tile forward/backward calls. Helper H2D staging
		      copies now pass that stream to `cudaMemcpyAsync` when available.
	      The production wrapper now creates and reuses a nonblocking production
	      stream whenever native-family CUDA graph support is enabled, stream-orders
	      first-step gradient zeroing, runs forward/backward and clip/AdamW/zero-grad
	      on that stream, and synchronizes it only before wrapper-side scalar-vector
	      reporting readback. Runtime and plan JSON
	      report `full_step_forward_backward_production_stream_plumbed: true`,
	      `full_step_loss_reporting_stream_synchronized_before_readback: true`, and
	      the runtime `production_step_cuda_stream_*` fields. They also now report
	      `full_step_forward_backward_launch_sequence_stream_ordered: true`,
	      family-specific `full_step_forward_backward_launch_sequence_capture_eligible`,
	      `full_step_forward_backward_graph_replay_ready: false`, and zero
	      `full_step_forward_backward_graph_capture_count`/
	      `full_step_forward_backward_graph_replay_count` counters on plan output
	      so replay readiness is explicit without pretending the plan itself has
	      run a graph. Eligible-family runtime steps, covering standard
	      LLaMA/MoE/JEPA plus Jamba, seq2seq, TTT, universal, HNet, diffusion,
	      and semantic-family paths, now
	      capture, instantiate, launch, and retain one warmed
	      forward/backward graph body after temporary replay leases are available;
	      later matching reporting-mode eligible-family steps replay that retained
	      graph and increment the runtime replay counter.
	      Many-tensor AdamW now has graph-friendly Tile ABI variants that read
	      LR/beta/eps/bias-correction scalars from a persistent device
	      hyperparameter buffer, so retained graph replay will not reuse stale
	      optimizer scalar launch arguments once the full forward/backward graph is
	      safe to keep across steps. Optimizer-only CUDA graphs using that ABI now
	      refresh the buffer before replay and avoid recapture for schedule-scalar
	      changes; scalar-argument fallback graphs still recapture on the full
	      hyperparameter tuple.
      Full-family token/target batches now copy from the sampler-owned uint16
      arrays directly to device and widen LM targets with
      `nfn_native_tile_uint16_to_int64`, so the previous host int64 target
      staging vector is gone from that step. Semantic-family steps now
	      derive per-row semantic target IDs and validity bytes on device from the
	      staged uint16 token batch with
	      `nfn_native_tile_semantic_targets_from_tokens_u16_int64`, and report
	      `full_family_semantic_targets_device_token_derivation: true`. The
	      semantic-router and semantic-dense-JEPA dataset loops now pass only
	      semantic dimensions/terms into that production path, avoiding per-step
	      host semantic-target vector materialization.
      The LLaMA/full-family
      path reuses a production host workspace for JEPA loss/gradient,
      semantic-target, route, chunk-route, semantic routing
      staging, and loss/gradient collection.
      Full-family LM, JEPA, and semantic losses now reduce loss partials on
      device with `nfn_native_tile_sum_partials_float32`; LM-head chunk CE loss
      now accumulates into a device scalar with
      `nfn_native_tile_sum_accumulate_float32` and performs the reporting D2H
      readback only after the LM-head backward/weight-gradient loop. Runtime and
      plan JSON report
      `full_family_lm_loss_device_accumulation_before_reporting: true` and
      `full_family_lm_loss_reporting_readback_after_lm_backward: true`. JEPA,
      compact semantic CE, and per-term semantic alignment loss/count totals now
      also accumulate into device scalars before final scalar reporting readback,
      and report
      `full_family_jepa_semantic_loss_device_accumulation_before_reporting: true`.
	      Full-family scalar reporting now packs LM, JEPA, semantic CE, semantic
		      route distillation, and per-term alignment totals into one device scalar
		      vector before a single D2H reporting copy, and reports
		      `full_family_loss_reporting_single_scalar_vector_readback: true` plus
		      `full_family_loss_reporting_post_backward_scalar_vector_readback: true`;
		      native-family dataset steps now request that host readback only on
	      reporting/progress steps, skip it on non-reporting optimizer steps, and
	      report `full_family_loss_reporting_readback_skipped_on_non_reporting_steps: true`
	      plus runtime `production_step_loss_reporting_readback_count`/
	      `production_step_loss_reporting_skipped_count`,
	      reporting-step D2H copies now run wrapper-side after `forward_backward`
	      and report `full_family_loss_reporting_readback_deferred_outside_full_step_capture: true`
	      plus `full_family_loss_reporting_readback_wrapper_side: true`.
      JEPA
      target/online pooled gradients now expand through device-side
      `latent_pool_backward` before reporting.
      Full-family JEPA MSE prediction/target latent
      gradients now form on device with `vector_binary`; JEPA mask values now
      construct on device with `nfn_native_tile_native_family_jepa_mask_float32`.
      Non-semantic dense/MoE JEPA now also materializes masked uint16 decoder
      token IDs and matching mask weights on device with
      `nfn_native_tile_native_family_jepa_mask_u16_float32`, then feeds those
      masked IDs through decoder token embedding and decoder embedding-weight
      backward. The compact target projection uses a separate original-token
      embedding buffer and accumulates target-branch embedding gradients against
      the original token IDs.
      Semantic-router MoE chunk-route forward compaction now runs on device via
      `nfn_native_tile_compact_chunk_routes_float32_int64` before chunk
      broadcast and reports
      `full_family_semantic_chunk_route_device_compaction: true`.
      Chunk route-gradient aggregation now runs on device via
      `nfn_native_tile_aggregate_chunk_route_gradients_float32` before device semantic hash/table gradient backward and reports
      `full_family_semantic_chunk_route_gradient_device_aggregation: true`.
      Standard non-semantic MoE selected-route backward now runs on device via
      `nfn_native_tile_topk_route_backward_float32` and reports
      `full_family_standard_moe_route_backward_device: true`.
      Semantic-router MoE route distillation now runs on device via
      `nfn_native_tile_semantic_route_distillation_backward_float32` and reports
      `full_family_semantic_route_distillation_device_backward: true`.
      Semantic JEPA reduced target-topic route distillation now runs on device
      via
      `nfn_native_tile_semantic_target_topic_distillation_backward_float32` and
      reports
      `full_family_semantic_target_topic_route_distillation_device_backward: true`.
      Semantic route and target-topic distillation reporting losses now
      accumulate into a device scalar with `nfn_native_tile_sum_accumulate_float32`
      before a single host reporting readback, and report
      `full_family_semantic_route_distillation_device_accumulation_before_reporting: true`.
      With semantic chunk routing enabled, target-topic teacher logits now come
      from device mean target chunks via
      `nfn_native_tile_causal_chunk_state_float32` and report
      `full_family_semantic_target_topic_chunk_state_device_forward: true`.
	      Native family JEPA plan/runtime JSON now explicitly reports the remaining
	      graph-parity deltas while exposing
	      `full_family_jepa_masked_decoder_tokens_device_materialization: true`:
	      `full_family_semantic_masked_online_tokens_device_materialization: true`
		      now reports that semantic JEPA also materializes masked online uint16
		      token IDs on device with the shared JEPA U16 mask kernel,
	      `full_family_jepa_target_original_token_embedding_device_forward: true`,
		      `full_family_jepa_masked_decoder_tokens_exact_parity: true`,
	      `full_family_jepa_target_encoder_backbone_parameter_layout: true`,
	      `full_family_jepa_target_encoder_projector_parameter_layout: true`,
	      `full_family_jepa_target_encoder_ema_frozen_parameter_layout: true`,
	      `full_family_jepa_target_backbone_device_forward: true`,
	      `full_family_jepa_target_projector_mlp_device_forward: true`,
	      `full_family_jepa_target_branch_stop_gradient: true`,
		      family-aware `full_family_jepa_target_backbone_exact_parity: true`
		      for dense JEPA, MoE-JEPA, and semantic-router JEPA target backbones,
		      and `full_family_jepa_exact_parity_delta`.
			      Semantic-router variants now report the consumed semantic
			      chunk-projector and masked-online encoder exact-parity fields as true,
			      while full-step CUDA graph capture is verified by the current
			      retained-replay audit instead of by this JEPA parity field.
	      Semantic JEPA plan/runtime JSON also reports
	      `full_family_semantic_jepa_target_backbone_device_forward: true`,
	      `full_family_semantic_masked_online_encoder_backbone_device_forward: true`,
	      `full_family_semantic_chunk_projector_per_term_topic_head_exact_parity: true`
	      and `full_family_semantic_masked_online_encoder_exact_parity: true`, so
	      the active semantic-router frozen target-backbone forward, masked-online
	      hidden-backbone forward, consumed chunk-projector surfaces, and reduced
	      target-topic distillation are reflected in plan/runtime JSON.
		      The native semantic-family parameter store includes a dedicated
		      trainable `semantic_chunk_topic_head.weight` for noncanonical fallback
		      semantic CE/topic logits and target-topic chunk logits, while canonical
		      86-d runs now report
			      `full_family_semantic_canonical_compact_topic_head_retired: true`.
		      Canonical 86-d semantic-family runs now also include the template-shaped
		      `semantic_projector.topic_heads.weight` per-term head plus
		      `semantic_projector.sig_head.weight` and persistent non-AdamW
		      `semantic_projector.residual_head.{0,2}.weight` buffers, derive the full
		      semantic target matrix on device for token-shard batches, run packed
		      per-term semantic alignment loss/backward through
		      `nfn_native_tile_semantic_alignment_packed_loss_backward_float32`, and
		      report
		      `full_family_semantic_chunk_projector_per_term_topic_head_parameter_layout: true`,
			      `full_family_semantic_chunk_projector_template_parameter_layout: true`,
			      `full_family_semantic_chunk_projector_residual_device_forward: true`,
			      `full_family_semantic_chunk_projector_residual_adamw_skip_exact: true`,
			      `full_family_semantic_chunk_projector_online_target_parameter_surface: true`,
			      `full_family_semantic_chunk_projector_modulelist_state_dict_aliases: true`,
			      `full_family_semantic_chunk_projector_per_term_alignment_device_forward_backward: true`,
			      `full_family_semantic_target_topic_per_term_route_distillation_device_backward: true`,
			      `full_family_semantic_canonical_compact_topic_head_retired: true`,
			      `full_family_semantic_canonical_compact_topic_head_parameter_retired: true`,
			      and `full_family_semantic_target_matrix_device_token_derivation: true`.
			      Semantic JEPA target-topic route distillation now consumes the packed
			      per-term teacher logits through
			      `nfn_native_tile_semantic_target_topic_packed_distillation_backward_float32`.
			      Canonical 86-d runs no longer allocate, forward, backpropagate, or collect
			      gradients through the legacy compact 86-wide topic head; it remains
			      only for noncanonical fallback compatibility. Per-term topic-head exact
			      parity now reports true for the consumed semantic MoE-JEPA production
			      graph surface.
			      The layout-smoke `parameter_layout` and checkpoint
			      `architecture_parameter_layout` metadata now export identical packed
			      native topic-head buffers as per-dimension Torch
			      `topic_heads.N.weight` aliases so native sidecars can be mapped back
			      to the template `ModuleList` shape without giving up the contiguous
			      runtime tensor.
				      Canonical semantic-router route policy now consumes packed per-term
				      topic confidence scores through
				      `nfn_native_tile_semantic_route_policy_packed_topic_float32` and reports
				      `full_family_semantic_route_policy_packed_topic_scores_device_forward: true`.
					      It also consumes the device semantic target matrix through
					      `nfn_native_tile_semantic_route_policy_packed_topic_matrix_float32`,
					      boosts every valid per-dimension target before top-k, and reports
					      `full_family_semantic_route_policy_target_matrix_device_forward: true`.
					      Canonical route hashing now also derives the native semantic vector
					      from packed per-term topic-head argmax coordinates through
					      `nfn_native_tile_semantic_vec_from_packed_topic_float32` and reports
					      `full_family_semantic_chunk_projector_topic_argmax_semantic_vec_device_forward: true`
					      plus
					      `full_family_semantic_compact_semantic_vector_projection_retired_from_canonical_route: true`.
					      Canonical semantic-router production steps now also
					      materialize padded Torch-shaped per-dimension topic logits with
					      `nfn_native_tile_semantic_packed_topic_to_padded_float32`
					      after route-time, target-topic, masked-online, and target
					      projector per-term projections. The packed logits remain the
					      native math surface, padded invalid terms are zero-filled, and
					      runtime JSON reports
					      `full_family_semantic_chunk_projector_padded_topic_logits_device_forward: true`.
					      Semantic-router JEPA now also computes the masked-online
					      semantic projector signature scalar on device from
					      `online_chunk_projector.sig_head.weight` through
					      `nfn_native_tile_semantic_signature_scalar_float32`
					      and reports
					      `full_family_semantic_masked_online_projector_signature_scalar_device_forward: true`.
						      The same semantic projector path now runs the online and
						      target residual MLPs from
						      `online_chunk_projector.residual_head.{0,2}.weight` and
						      `target_chunk_projector.residual_head.{0,2}.weight`
						      and reports
						      `full_family_semantic_masked_online_projector_residual_device_forward: true`
						      plus
						      `full_family_semantic_target_projector_residual_device_forward: true`.
					      Because the semantic encoder graph surfaces but does not consume
					      that residual tuple output, the native parameter store now keeps
					      the residual-head weights persistent but out of AdamW, reporting
					      `full_family_semantic_masked_online_projector_residual_adamw_skip_exact: true`.
					      Exact per-term chunk-projector parity now reports true for the
					      consumed production graph surface.
		      Canonical semantic MoE-JEPA now retires the legacy compact
	      `jepa_target_encoder.weight`, `jepa_online_encoder.weight`,
	      `jepa_projector.weight`, and `jepa_predictor.weight` buffers that are
	      absent from the Torch template state dict, and reports
	      `full_family_semantic_legacy_compact_jepa_buffers_retired: true`.
	      Older canonical semantic MoE-JEPA native checkpoints with those extra
	      buffers should be regenerated or migrated before resume.
	      The expanded hidden-backbone masked-online path now replays
	      `jepa_online_encoder.backbone.*` before backward, backpropagates through
	      `jepa_online_encoder.backbone.final_norm.weight`, and reports
	      `full_family_semantic_masked_online_encoder_backbone_final_norm_device_backward: true`.
	      It also backpropagates through the replayed last MoE FFN/router layer,
	      collecting `jepa_online_encoder.backbone.layers.N.ffn_norm.weight`,
	      `.router.weight`, `.experts.gate_up.weight`, and `.experts.down.weight`,
	      and reports
	      `full_family_semantic_masked_online_encoder_backbone_last_moe_layer_device_backward: true`.
	      The replayed last attention layer now backpropagates through attention
	      output, scaled-dot-product attention, RoPE, Q/K/V projections, and
	      attention norm, collecting `attention_norm.weight`, `q_proj.weight`,
	      `k_proj.weight`, `v_proj.weight`, and `attention_out.weight`, and
	      reports
	      `full_family_semantic_masked_online_encoder_backbone_last_attention_device_backward: true`.
	      The expanded online-backbone backward now traverses every layer in
	      reverse, immediately collecting each layer's online-backbone gradients
	      before reusing the scratch buffers, and reports
	      `full_family_semantic_masked_online_encoder_backbone_all_layers_device_backward: true`.
	      The semantic JEPA loss path now materializes online and target packed
	      topic vectors from `online_chunk_projector.topic_heads.weight` and
	      `target_chunk_projector.topic_heads.weight`, appends the softmax-expected
	      signature scalar from the matching chunk-projector `sig_head.weight`, runs the native
	      `jepa_semantic_predictor.net.{0,2}.weight` GELU MLP over the 87-d
	      semantic vector, collects predictor and online signature-head weight
	      gradients, and reports
	      `full_family_semantic_jepa_topic_vector_predictor_device_forward_backward: true`
	      plus `full_family_semantic_jepa_signature_coordinate_backward: true`.
	      Masked-online encoder exact parity now reports true for the consumed
	      production graph surface.
      Semantic hash/table gradients now reduce on device via
      `nfn_native_tile_semantic_hash_table_backward_float32` and report
      `full_family_semantic_hash_table_gradient_device_backward: true`; that
      backward launch now remains stream-ordered without an immediate device-wide
      synchronize and reports
      `full_family_semantic_hash_table_backward_stream_ordered_no_sync: true`.
      Semantic-router free expert logits now project from the native semantic
      vector via `nfn_native_tile_semantic_free_expert_projection_float32`, and
      reverse mode uses
      `nfn_native_tile_semantic_free_expert_projection_backward_float32` plus
      `nfn_native_tile_semantic_router_bias_backward_float32` to update the
      template-shaped `semantic_router.free_head.{weight,bias}` buffers. Shared
      route logits use `semantic_router.shared_logits` through
      `nfn_native_tile_semantic_router_bias_add_float32`; plan/runtime JSON reports
      `full_family_semantic_shared_expert_projection_device_forward_backward: true`
      and
      `full_family_semantic_free_expert_projection_device_forward_backward: true`.
      The same layout reports
      `full_family_semantic_router_free_head_template_parameter_layout: true`.
      Canonical per-term routing now also materializes the chunk projector
      signature scalar, appends it to the semantic route vector for free-expert
      scoring, splits that coordinate in reverse mode, and updates
      `semantic_projector.sig_head.weight`; plan/runtime JSON reports
      `full_family_semantic_chunk_projector_signature_coordinate_device_forward_backward: true`.
      DeepSeek mHC beta-logit gradients now reduce on device via
      `nfn_native_tile_mhc_beta_gradient_float32` and report
      `full_family_mhc_beta_gradient_device_reduction: true`.
      Direct optimizer mode also skips the fallback host collection pass for
      RMSNorm weight gradients after device accumulation and reports
      `full_family_norm_weight_host_collection_elided_when_direct: true`.
      HNet byte-LM and Universal ACT halt-loss
      specialty helpers also reduce their loss partials on device and report
      `specialty_lm_act_loss_device_reduction: true`.
      Runtime JSON now reports `temporary_pool_buffer_count`,
      `temporary_active_buffer_count`, `temporary_metadata_reserved_buffer_count`,
      and `temporary_active_buffer_high_water_count` so workspace-pool warmup,
      reserved lease metadata, and active lease high-water state are visible at
      step boundaries. `temporary_replay_lease_count` reports warmed temporary
      buffer leases served from the recorded sequence, and
      `temporary_replay_plan_buffer_count` reports the sequence length. Strict
      diagnostic begin/end validation, including per-lease size/free-order
      checks, can be restored with `NFN_NATIVE_FAMILY_TEMPORARY_REPLAY_VALIDATE=1`;
      the default warmed path skips those checks inside `forward_backward` and
      only advances the lease cursors. Strict mode records allocation and free
      replay sequences separately, so reuse-heavy helpers validate against their
      actual free order.
      Replay readiness now validates the recorded pointer/size sequence against
      the inactive pool instead of requiring plan length to equal active
      high-water, so reuse-heavy specialty steps can replay warmed temporary
      leases.
      Seq2seq, TTT, Jamba, diffusion, universal, and HNet full-graph
      token/byte-LM staging now match the full-family direct batch path:
      sampler-owned uint16 tokens or raw HNet bytes upload directly, targets
      widen on device with `nfn_native_tile_uint16_to_int64` or
      `nfn_native_tile_uint8_to_int64`, and diffusion builds deterministic
      masked tokens plus int64 denoising targets with
      `nfn_native_tile_diffusion_mask_u16_int64`, removing the previous
      per-step host token/target staging vectors for those specialty paths.
      Universal ACT now also packs recurrent states/logits, prepares halt
      weights/remainders, and unpacks ACT gradients on device with dedicated
      raw Tile ABI helpers, removing the previous ACT host control loop.
      Jamba's Mamba-state/head gradients now accumulate directly into the
      persistent `FamilyOptimizerState` device gradient buffers with
      `nfn_native_tile_gradient_accumulate_float32`, removing that specialty
      gradient path from the host sparse-gradient map in production mode and
      reporting `specialty_jamba_device_gradient_accumulation: true`.
      TTT inner-update specialty gradients now use the same direct device
      optimizer accumulation path for token embedding, inner base/down/up, and
      LM-head gradients, and report
      `specialty_ttt_device_gradient_accumulation: true`.
      Universal recurrent/halt, embedding, and LM-head specialty gradients now
      also accumulate directly into persistent optimizer buffers on device and
      report `specialty_universal_device_gradient_accumulation: true`.
      Seq2seq full-buffer gradients now use the same direct device optimizer
      accumulation path and report
      `specialty_seq2seq_device_gradient_accumulation: true`; norm-vector
      gradients extract diagonals on device and report
      `specialty_hnet_seq2seq_norm_device_gradient_accumulation: true`.
      Combined encoder/cross-QKV slices now assemble into stacked device
      gradient buffers before optimizer accumulation and report
      `specialty_seq2seq_stacked_qkv_device_gradient_accumulation: true`.
      HNet byte patch embed/merge, per-layer projection/FFN, LM-head, and
      norm-vector gradients now also accumulate directly into persistent
      optimizer buffers on device and report
      `specialty_hnet_device_gradient_accumulation: true` plus
      `specialty_hnet_seq2seq_norm_device_gradient_accumulation: true`.
      Direct specialty gradient accumulation now skips the former
      post-backward full-device synchronization before optimizer handoff and
      reports `specialty_direct_gradient_sync_elision: true`; specialty helpers
      still retain host-side launch orchestration, loss/reporting readback, and
      other control stages. Optional diagnostic null-stream runs only perform
      post-step checksum D2H readback when
      `NFN_NATIVE_FAMILY_PRODUCTION_STEP_CHECKSUM=1`; future non-null
      capture-stream runs skip it, so the diagnostic checksum is no longer
      listed as a full-step capture blocker.

## §2. mixllama — base MoE family

Presets: `mixllama`, `mixllama_fast`, `mixllama_fast_megakernel`, `moe`,
`moe_modern`, `deepseek_v3`. Binary: `build/nfn_mixllama_native_train`.
Existing slice `print_sampled_moe_jepa_family_step_json` (~14417,
`include_jepa=false`) already runs `topk_route` → `broadcast_expert_routes` →
`moe_swiglu_forward` → CE, backward via `moe_swiglu_backward` /
`linear_backward_*` / CE-backward, balance loss, and a device AdamW step —
at toy geometry. Production step per layer at full geometry: embed → RMSNorm →
packed-QKV attention → residual → RMSNorm → router top-k → moe_swiglu →
combine → LM-head CE; mirror backward incl. attention-to-QKV, RMSNorm and
embedding backward; balance loss folded into router grads; grad-accum across
`batch_plan` micro-steps; one fused AdamW-many update per step.
Full per-layer layouts now match the template's router surface: the sampled-only
`router.topk_scale` buffer is omitted, and aux-free layouts apply the live
non-optimizer `layers.N.router.auxfree_bias.weight` before Tile top-k routing.
The full route-weight backward now returns expert-output route gradients through
the selected-route softmax Jacobian and updates the per-layer router weight;
configured `layers_per_expert` is now executed as a sequential expert-depth
stack with per-depth gate/up/down gradients and accumulated route gradients;
semantic route contracts are covered by the default full-family path; exact
preset parity remains in the final delta audit.

- [x] step  - [x] persist  - [x] ckpt  - [x] flip  - [x] test  - [x] docs

The `deepseek_v3` and `deepseek_v3_modern` selectors now use native MLA rather
than ordinary K/V projections: latent KV-A/KV-B projections, shared RoPE key
replication, reconstructed per-head K/V attention, and reverse gradients for
both MLA matrices are implemented. The native trainer RoPE adapter now converts
element-count ABI calls to the shared launcher batch geometry and guards compact
RoPE paired loads/stores, fixing the prior DeepSeek-V3 MLA illegal-access
failure. A reduced DeepSeek-V3 one-step run with `model_dim=64`, `hidden_dim=128`,
`seq_len=4`, and vocab `256` passes normally and under `compute-sanitizer` with
zero reported errors. Plan JSON now reports DeepSeek-V3 as
`production_step_family: "deepseek_v3"` with `auxfree_moe_balance: true`, while
the V4-only mHC residual stage stays scoped to `deepseek_v4`.

## §3. deepseek-v4

Presets: `deepseek_v4`. Binary: `build/nfn_deepseek_v4_native_train`.
Reuses the §2 `MixLlamaProductionStep` with config deltas (expert count,
top-k, dims). Mostly a build-script + registry + tests flip once §2 lands.
The full native path now also persists the template's per-layer
`residual.beta_logit.weight`, applies the constrained mHC alpha/beta residual
mix through the Tile vector-binary op at both residual sites, and collects
beta-logit gradients during reverse mode. The native DeepSeek-V4 selector now
uses FP8 E4M3 quantized-linear dispatch for its native linear projections;
its expert SwiGLU path now uses MXFP4 E2M1 block-32 quantized expert reads with
straight-through float32 expert gradients. Its MoE router now uses the
template's sqrt-softplus scorer through raw Tile forward/backward symbols. For
the shipped `deepseek_v4` template, native-sparse CSA parity is explicit:
plan/runtime JSON reports
`full_family_deepseek_v4_native_sparse_csa_exact_parity: true`, while
`full_family_deepseek_v4_mla_template_required: false` and
`full_family_deepseek_v4_learned_csa_indexer_template_required: false` make
clear that MLA and a learned CSA indexer are not part of this preset. The
template uses the single-stream `ManifoldHyperConnectionStage`, so native
plan/runtime JSON now reports `full_family_mhc_single_stream_exact_parity: true`
and `full_family_mhc_multi_stream_template_required: false`; a future
parallel-stream mHC variant is a template extension rather than an unmet field
in the current graph.
The full step also applies per-head Q/K RMSNorm before RoPE for the configured
DeepSeek/Gemma QK-normalized presets and reverses it before Q/K projection
gradients. Plan JSON now separates active DeepSeek-V3 MLA from DeepSeek-V4
native-sparse CSA: inactive MLA/KV-PCA dimensions report `0`, with
`mla_attention_enabled`, `kv_pca_enabled`, and
`native_sparse_csa_attention_enabled` making the active attention contract
explicit.

- [x] step  - [x] persist  - [x] ckpt  - [x] flip  - [x] test  - [x] docs
- [x] sparse-attention refinement in the production step (deepseek-v4
      attention pattern, not the plain packed-QKV path). The full
      `deepseek-v4` graph uses separate Q/K/V projections, QK RMSNorm, the
      native sink/window/block/compression-stride sparse attention geometry in
      both directions, mHC residual mixing, and persistent router/expert
      gradients. Current-template native-sparse CSA parity is reported
      explicitly, and plan metadata marks MLA plus a learned CSA indexer as not
      required by this preset; current-template single-stream mHC parity is
      reported explicitly; plan metadata now distinguishes this
      CSA path from the active DeepSeek-V3 MLA path with explicit booleans and
      zeroed inactive dimensions.
- [x] sqrt-softplus router scoring in the native production step. The
      `deepseek_v4` template graph writes `topk_route.score_fn="sqrt_softplus"`,
      plan/runtime JSON reports `architecture.router_score_fn`, and the native
      loop uses `nfn_native_tile_topk_route_sqrt_softplus_float32` plus its
      backward ABI for selected-route weights and router-logit gradients.

## §4. moe-jepa-evo

Presets: `moe_jepa_evo`, `moe_jepa_evo_modern`, `auxfree_moe_jepa_evo`
(3 registry entries, one binary `build/nfn_moe_jepa_evo_native_train`).
Extends §2 with the JEPA branch: target-encoder / projector / predictor
buffers (already in the parameter layout), `latent_pool` +
`latent_mse_loss` forward and gradients, AR+JEPA+router loss composition.
The native MoE-JEPA target branch now persists `target_encoder.backbone.*`
router/expert buffers plus `target_encoder.projector.*`, runs original tokens
through the frozen target MoE stack, applies the frozen target projector MLP,
and reports `full_family_jepa_target_backbone_exact_parity: true` for the
frozen target-backbone surface while keeping the target branch stop-gradient.
The auxfree preset drops balance-loss gradients. Partial: the shared persistent
sampled bridge now marks `auxfree_moe_jepa_evo` descriptors with
`auxfree_moe_balance` and skips the chained-block, selected-route, and
routed-combine sampled route-balance gradient terms for that preset. Non-aux-free
chained-block MoE sampling now folds a bounded route-balance term into the
router gradients after sampled attention and before residual expert output
composition. Aux-free and modern MoE-family
layouts also get a non-optimizer `router.auxfree_bias.weight` buffer that the
sampled chained-block, selected-router, and routed-combine top-k scoring paths
add before route selection, and production-step runtime JSON reports
`production_step_auxfree_bias_refresh_count` when that bias is refreshed from
sampled route-load imbalance outside AdamW. Full Tile-ABI aux-free bias
application and selected-route backward are now wired before/through top-k.
The full MoE-JEPA path now loads and invokes `latent_pool` for both target and
online branches, computes `latent_mse_loss` over `[batch, model_dim]` latents,
and uses device-side `latent_pool_backward` to broadcast target/online gradients
back to sequence rows using the same mask weights. Native CLI controls
`--jepa-mask-ratio` and
`--jepa-mask-strategy random|block` are recorded in plan/layout metadata.
CUDA execution is covered for reduced loops, and non-semantic masked decoder
tokens now materialize on device before the decoder token embedding via
`nfn_native_tile_native_family_jepa_mask_u16_float32`; plan/runtime JSON now
	reports `full_family_jepa_masked_decoder_tokens_exact_parity: true` for that
	non-semantic online-decoder input path. Dense JEPA, MoE-JEPA, and
	semantic-router JEPA now also report family-aware
	`full_family_jepa_target_backbone_exact_parity: true` for the frozen
	target-backbone surface; the consumed semantic masked-online encoder and
	chunk-projector parity fields now report true, with full-step CUDA graph
	capture tracked separately.
Semantic JEPA now materializes
masked online uint16 token IDs on device, embeds them through a separate compact
native `jepa_online_encoder.weight` branch, and reports
`full_family_semantic_masked_online_tokens_device_materialization: true` plus
`full_family_semantic_masked_online_encoder_device_forward_backward: true`; the
hidden-backbone semantic encoder subgraph remains final verification/delta work.
Semantic route parity remains final verification/delta work. The default family
status gate is covered.

- [x] step  - [x] persist  - [x] ckpt  - [x] flip  - [x] test  - [x] docs

## §5. semantic-router-moe

Presets: `semantic_router_moe`, `semantic_router_moe_megakernel`,
`semantic_router_moe_modern`, `semantic_moe_jepa_evo`,
`semantic_moe_jepa_evo_modern`, `diff_semantic_moe_jepa_evo`.
Binary: `build/nfn_semantic_router_moe_native_train`. Adds on §4:
semantic targets via `derive_semantic_targets_from_tokens` (~413), semantic
planner/alignment buffers, `route_selection_loss_partials` + softmax
distillation, chunk routes (`broadcast_chunk_routes`), optional evo controller
(`evo_mutate_candidates` / `select_best_loss` / `adopt_candidate`) on router
weights at `--evo-layer-interval` cadence.
Partial: the shared persistent sampled bridge now threads loop-derived semantic
targets into the MoE top-k scorers. The chained-block MoE, selected
router/expert, and routed-combine paths map the first non-ignored semantic
target through the documented shared + semantic-vocab expert contract, add a
bounded logit bias before route selection, force that semantic expert into the
bounded sampled top-k candidate set when needed, train the selected
router/expert CE against the semantic route index, add a bounded selected-router
softmax distillation term from a smoothed semantic-route teacher distribution,
reuse the `route_chunk_size` anchor row's semantic route for sampled non-anchor
rows before top-k selection,
adopt bounded post-AdamW sampled semantic route-evo router-weight candidates on
the configured `--evo-layer-interval` cadence,
and report
`production_step_semantic_route_bias_count` plus
`production_step_semantic_route_forced_count` plus
`production_step_semantic_route_distillation_count` plus
`production_step_semantic_route_broadcast_count` plus
`production_step_semantic_route_evo_adoption_count`. Full Tile-ABI semantic-router
now biases the mapped semantic-vocabulary expert before top-k, reports row/layer
route-policy bias/distillation/broadcast counters for full-geometry runs, and
keeps the forced counter scoped to actual top-k replacement events; full standard
route-weight backward is also shared with this path. The full layout now
persists the Tile semantic-hash projection, hash-bucket embeddings, table-gate
logits, and per-dimension bias; reverse mode reduces selected route-logit
gradients into the trainable embedding, gate, and bias buffers. Full semantic
rows also add the bounded smoothed-target route-distillation objective to the
reduced router gradient. Full semantic routing uses the configured
chunk-anchor broadcast and aggregates its route gradients back to anchor rows.
Route-evo production refreshes now apply the adopted router-weight candidate
through `nfn_native_tile_evo_adopt_candidate_float32` into the live
`router.weight` device buffer, mirror the device token-derived semantic target
rule for token-shard batches that omit host semantic-target matrices, and report
`full_family_semantic_route_evo_device_adoption: true`. Semantic-router
expert combine now reports
`full_family_semantic_expert_combine_device_forward_backward: true` because
chunk-route broadcast feeds the weighted selected-expert `moe_forward` path and
reverse mode uses `moe_backward_with_route_grad` for route-weight, input, and
expert gradients. Shared semantic experts are now always present in the native
route tensor: semantic-router route forward/backward uses
`nfn_native_tile_semantic_shared_topk_route_float32` and
`nfn_native_tile_semantic_shared_topk_route_backward_float32` with
`route_width = semantic_shared_experts + top_k`, reporting
`full_family_semantic_router_shared_experts_always_on_route_width: true` and
`full_family_semantic_shared_plus_topk_route_device_forward_backward: true`.
Canonical per-term routing now also uses
`nfn_native_tile_semantic_shared_forced_topk_route_float32` so rows with valid
semantic target dimensions restrict dynamic top-k candidate selection to those
forced semantic experts while preserving always-on shared experts and unmasked
`route_logits` for distillation; it reports
`full_family_semantic_router_forced_target_candidate_mask_device_forward: true`.
This closes the route-width and forced-candidate ABI mismatches but does not
flip the broader semantic exact-parity guards. Fresh semantic-family parameter
stores now initialize
`semantic_hash.proj.weight` from the NumPy-compatible
`RandomState(42).randn(tables, planes, dims)` sequence used by the Torch
	`SemanticChunkHasherStage` and report
	`full_family_semantic_hash_projection_seeded_exact: true`; canonical per-term
		topic-head training now retires the compact 86-wide head from semantic CE and
		target-topic distillation, and canonical route policy now uses packed per-term
		topic confidence scores plus device semantic-target-matrix boosts, while
		shared/free expert projection is covered in full
	geometry. The
reduced native target-topic route-distillation path is now covered
by `nfn_native_tile_semantic_target_topic_distillation_backward_float32`, with
`full_family_semantic_chunk_projector_per_term_topic_head_exact_parity: true`
and `full_family_semantic_masked_online_encoder_exact_parity: true` covering
	the consumed semantic MoE-JEPA production graph surfaces. Native semantic-family checkpoints now
	persist the dedicated trainable `semantic_chunk_topic_head.weight` only for
	noncanonical fallback runs, while canonical runtime JSON reports
	`full_family_semantic_canonical_compact_topic_head_retired: true` and
	`full_family_semantic_canonical_compact_topic_head_parameter_retired: true`.
Semantic JEPA masked online token IDs now materialize on device with the shared
JEPA U16 mask kernel and report
`full_family_semantic_masked_online_tokens_device_materialization: true`; canonical
semantic-router JEPA now feeds those tokens through
`jepa_online_encoder.backbone.*` in forward mode and reports
`full_family_semantic_masked_online_encoder_backbone_device_forward: true`. The
separate compact native masked-online encoder remains the fallback path and
reports `full_family_semantic_masked_online_encoder_device_forward_backward: true`.
When semantic chunk routing is active, the JEPA objective now uses mean
target chunks and prefix masked-online chunks through the raw Tile
`causal_chunk_state` forward/backward ABI and reports
`full_family_semantic_jepa_chunk_state_objective_device_forward_backward: true`.
The native semantic JEPA parameter store now also carries separate
`jepa_online_encoder.backbone.*` and frozen `jepa_target_encoder.backbone.*`
surfaces plus packed `*.semantic_projector.*` weights, reporting
`full_family_semantic_jepa_encoder_backbone_parameter_layout: true`,
`full_family_semantic_jepa_encoder_projector_parameter_layout: true`,
`full_family_semantic_jepa_encoder_state_dict_aliases: true`, and
`full_family_semantic_jepa_target_encoder_ema_frozen_parameter_layout: true`.
Checkpoint/layout metadata now maps those native encoder buffers back to the
Torch `online_encoder.*` and `target_encoder.*` state-dict names, including
split MoE `dispatch.w1`, `dispatch.w2`, and `dispatch.w3` aliases over the
packed native expert tensors.
Hidden-backbone semantic encoder parity is still required before masked-online
exact parity can flip.

- [x] step  - [x] persist  - [x] ckpt  - [x] flip  - [x] test  - [x] docs

## §6. jamba — hybrid Mamba + attention MoE

Presets: `jamba`, `jamba_modern`. Binary: `build/nfn_jamba_native_train`.
Hybrid layer schedule interleaving `causal_chunk_state[_backward]` Mamba
layers (mamba.in_proj / state buffers) with attention+MoE layers; replaces the
`print_single_substep_dataset_loop_json` path with the production loop.
Partial: the shared persistent sampled bridge now adds a bounded recurrent
Mamba-state objective for Jamba layouts. It carries sampled state across real
batch rows, uses the live `mamba.in_proj.weight` input/gate rows plus
`mamba.state.weight`, backpropagates into those buffers and
`token_embedding.weight`, and reports
`production_step_sampled_jamba_mamba_state_row_count`. Full Tile-ABI
`causal_chunk_state[_backward]` state/head composition is now available behind
the full-geometry selector, using live Mamba input/state and LM-head buffers;
the full-selector now also composes the shared hidden-backbone Tile step for
generic attention/FFN/MoE buffers before the common optimizer update, and Jamba
full layouts now persist per-layer router/expert buffers. The full-selector
uploads sampler-owned uint16 token/target batches directly and widens targets on
device with `nfn_native_tile_uint16_to_int64`, avoiding the earlier per-step
host token/target staging vectors. The full-selector composes the interleaved
Mamba state/head path with the attention/MoE transformer backbone and persists
covered family gradients through shared many-tensor AdamW. Jamba is promoted in
normal builds.

- [x] step  - [x] persist  - [x] ckpt  - [x] flip  - [x] test  - [x] docs

---

## §7. Dense families (specialty coverage status varies — same 6 ticks each)

Same shared infrastructure (§1); only the per-family step composition differs.
The full-family CE, latent-MSE, and ACT loss paths now allocate and reduce
Tile block partials for real batch sizes; scalar loss buffers are no longer
used by the specialized seq2seq, Jamba, HNet, universal, diffusion, or TTT
helpers.
`llama` first: the mixllama step minus router/experts (plain SwiGLU MLP), so
§2 and §7-llama share almost everything.

### §7.1 llama (`build/nfn_llama_native_train`)
Presets: `llama`, `llama_modern`, `llama_fast`, `llama_fast_megakernel`,
`llama_megakernel`, `modern_norms_llama`, `ternary_b158`,
`ternary_b158_modern`, `fp8_llama`, `mxfp4_llama`, `gemma3`,
`diff_transformer`, `longctx_sparse_llama`, `qwen3_longctx`, `kv_pca_llama`,
`kv_pca_llama_modern`.

- [x] step  - [x] persist  - [x] ckpt  - [x] flip  - [x] test  - [x] docs
- [x] quantized preset deltas in the production step: `ternary_b158`,
      `fp8_llama`, `mxfp4_llama` (beyond the shared f32 step). The full
      LLaMA Tile graph now dispatches per-row ternary, FP8 E4M3, or MXFP4
      E2M1 block-32 linear forward/input-backward kernels selected by preset;
      weight gradients remain float32 straight-through updates and plan JSON
      reports `architecture.linear_quantization`. Live reduced one-step CUDA
      dataset loops now pass for `ternary-b158`, `fp8-llama`, and
      `mxfp4-llama` with checkpoint writing and the raw Tile ops library.
- [x] sparse/long-context attention preset deltas: `longctx_sparse_llama`,
      `qwen3_longctx`, `kv_pca_llama` variants
      - Partial: `longctx_sparse_llama` and `qwen3_longctx` now execute their
        native sparse sink/window/block geometry in both attention directions;
        `qwen3_longctx` also applies the template's YaRN RoPE interpolation
        (default factor 4, original context 2048) with CLI overrides and plan
        JSON reporting. `kv_pca_llama` now persists shared per-head K/V
        encode/decode matrices per layer, executes compression/reconstruction
        around attention, and collects all four matrix gradients; plan JSON
        reports `attention_variant: "kv_pca"` and the default compressed
        width `head_dim // 4`. Live reduced one-step CUDA dataset loops now
        pass for `longctx-sparse-llama`, `qwen3-longctx`, and `kv-pca-llama`;
        plan checks confirm native sparse attention, Qwen YaRN scaling, and
        KV-PCA respectively.
- [x] `diff_transformer` differential-attention delta. The native full
      LLaMA graph now persists the shared learnable `attention.diff_lambda`,
      splits query/key head channels for two causal attention branches,
      applies differential combine plus head-wise RMSNorm in both directions,
      and collects the scalar lambda gradient into AdamW. Plan JSON reports
      `attention_variant: "differential"`; pre-change differential sidecars
      require regeneration because the parameter layout gained one scalar. A
      live reduced one-step CUDA dataset loop now passes for `diff-transformer`
      with checkpoint writing.

### §7.2 jepa (`build/nfn_jepa_native_train`)
Presets: `llm_jepa`, `llm_jepa_modern`, `dense_jepa_evo`,
`dense_jepa_evo_modern`. Llama step + JEPA latent branch (see §4, without
router/experts). Non-semantic masked decoder tokens now materialize on device.
The native parameter store now also persists the frozen
`target_encoder.backbone.*` and `target_encoder.projector.*` surfaces for
non-semantic JEPA. Dense JEPA uses dense FFN target layers; MoE-JEPA uses
router/expert target layers. The runtime now forwards original tokens through
the frozen target backbone and frozen target projector MLP as a stop-gradient
branch and reports
`full_family_jepa_target_backbone_device_forward: true`,
`full_family_jepa_target_projector_mlp_device_forward: true`, and
`full_family_jepa_target_branch_stop_gradient: true`; dense JEPA, MoE-JEPA, and
semantic-router JEPA target-backbone exact parity is true for the frozen
hidden-backbone surface, while semantic encoder/projector parity work remains
tracked separately.
Existing slice:
`print_dense_jepa_dataset_loop_json` ~16819.

- [x] step  - [x] persist  - [x] ckpt  - [x] flip  - [x] test  - [x] docs

### §7.3 semantic-dense-jepa (`build/nfn_semantic_dense_jepa_native_train`)
Presets: `semantic_dense_jepa_evo`, `semantic_dense_jepa_evo_modern`,
`dyt_geglu_semantic_dense_jepa_evo`, `jepa_semantic_hybrid`,
`jepa_semantic_hybrid_megakernel`, `jepa_semantic_hybrid_modern`.
§7.2 + semantic targets/alignment (see §5). Existing slice:
`print_semantic_dense_jepa_dataset_loop_json` ~17279.
The production path now treats target-topic route distillation as MoE-router
only, loads the packed per-term topic padding Tile symbol for non-router
semantic runs, and sizes semantic-dense JEPA target-topic fallback buffers by
the actual row-mode path rather than the route-chunk template path. A cleaned
one-step semantic-dense JEPA dataset loop passes, and the full three-step CUDA
graph audit reports one retained forward/backward graph capture plus two
replays for this family.

- [x] step  - [x] persist  - [x] ckpt  - [x] flip  - [x] test  - [x] docs

### §7.4 seq2seq (`build/nfn_seq2seq_native_train`)
Presets: `seq2seq`, `seq2seq_modern`. Encoder–decoder composition with
cross-attention fwd/bwd.
Partial: the shared persistent sampled bridge now adds bounded
decoder-to-encoder cross-attention coverage for seq2seq layouts. It samples
real batch rows, forms decoder queries from live
`decoder.cross_attention.qkv.weight`, encoder keys/values from live
`encoder.layers.qkv.weight`, backpropagates into those QKV buffers and
`token_embedding.weight`, and reports
`production_step_sampled_seq2seq_cross_attention_row_count`. Full Tile-ABI
encoder/decoder stacked attention and cross-attention backward integration now
runs by default for the exact `seq2seq` target. The production composition runs
encoder self-attention, decoder causal self-attention, decoder-to-encoder
cross-attention, encoder and decoder SwiGLU FFNs, token CE, and reverse-mode
QKV/FFN/LM/embedding gradients through the Tile ABI. It uploads sampler-owned
uint16 token/target batches directly and widens targets on device with
`nfn_native_tile_uint16_to_int64`, avoiding the earlier per-step host
token/target staging vectors. Both stack FFNs consume the live per-layer
`ffn_gate_up.weight`/`ffn_down.weight` buffers and accumulate encoder plus
decoder gradients before optimizer handoff. The persistent production contract
now covers the encoder/decoder parameter roles, checkpoint writer, and shared
many-tensor AdamW handoff.

- [x] step  - [x] persist  - [x] ckpt  - [x] flip  - [x] test  - [x] docs

### §7.5 diffusion (`build/nfn_diffusion_native_train`)
Presets: `diffusion`, `diffusion_modern`. Masked-denoising objective on the
llama trunk.
The full Tile-ABI diffusion step now matches the template graph: deterministic
per-sequence timesteps drive the mask scheduler, masked tokens enter the live
token embedding, the complete LLaMA trunk runs on device, and the live
`denoise_head.weight` produces CE logits against the original unmasked tokens.
The reverse pass persists token-embedding, transformer, final-norm, and
denoise-head gradients before shared many-tensor AdamW. The graph has no
timestep-embedding parameter, so the native layout does not invent one. The
diffusion family is promoted to full production status in normal builds. The
full-selector uploads sampler-owned uint16 tokens directly and uses
`nfn_native_tile_diffusion_mask_u16_int64` to construct masked uint16 inputs and
int64 denoising targets on device, avoiding the previous per-step host
token/target staging vectors.

- [x] step  - [x] persist  - [x] ckpt  - [x] flip  - [x] test  - [x] docs

### §7.6 ttt-llama (`build/nfn_ttt_llama_native_train`)
Presets: `ttt_llama`, `ttt_llama_modern`. Test-time-training inner-loop state
update inside the production step.
Partial: the shared persistent sampled bridge now adds bounded TTT inner-update
coverage. It samples real batch rows, runs live `ttt.inner_base.weight`,
`ttt.inner_down.weight`, tanh, and `ttt.inner_up.weight` as a residual update,
backpropagates into those three buffers and `token_embedding.weight`, and
reports `production_step_sampled_ttt_inner_update_row_count`. The full Tile-ABI
path runs the live base/down/tanh/up residual inner update, token CE, and
reverse-mode gradients through the TTT buffers and embedding before composing
the full transformer backbone. Native layout sizing honors the template's configurable
inner width (`--ttt-hidden-dim`, default `32`) for `ttt.inner_down.weight` and
`ttt.inner_up.weight`; sampled and full paths derive the same width from the
live buffer geometry.
The full-selector composes the shared hidden-backbone Tile step for the base
transformer buffers, then persists the inner-update and backbone gradients
through shared many-tensor AdamW. TTT is promoted in normal builds.

- [x] step  - [x] persist  - [x] ckpt  - [x] flip  - [x] test  - [x] docs

### §7.7 universal-llama (`build/nfn_universal_llama_native_train`)
Presets: `universal_llama`, `universal_llama_modern`. Weight-tied recurrent
block applied `depth` times; gradients accumulate into the shared block.
Partial: the shared persistent sampled bridge now adds bounded universal
recurrent coverage. It samples real batch rows, unrolls the live
`universal.recurrent.weight` block for three tied steps, weights each step with
the live `universal.halt_gate.weight`, backpropagates into recurrent/halt
weights and `token_embedding.weight`, and reports
`production_step_sampled_universal_recurrent_step_row_count`. Full Tile-ABI
weight-tied recurrent depth with ACT halting is available behind
`NFN_NATIVE_FULL_GEOMETRY_FORWARD_BACKWARD=1` for the exact `universal-llama`
target. The opt-in path now honors configurable `--max-recurrence-steps`
(default `4`) and `--halt-epsilon` (default `0.01`), builds ACT remainder
weights, runs the native weighted-sum and halting-BCE primitives, and
backpropagates LM gradients through every recurrent state plus the shared
recurrent/halting weights and embedding. The full-selector also composes the
shared hidden-backbone Tile step so generic attention/FFN buffers enter the
persistent gradient map, then persists the tied recurrence, ACT, backbone, and
LM gradients through shared many-tensor AdamW. The full-selector uploads
sampler-owned uint16 token/target batches directly and widens targets on device
with `nfn_native_tile_uint16_to_int64`; ACT recurrent-state/logit packing,
halt-weight/remainder preparation, and ACT gradient fanout now run on device
through `nfn_native_tile_act_pack_step_float32`,
`nfn_native_tile_act_prepare_weights_float32`, and
`nfn_native_tile_act_unpack_step_grad_float32`, avoiding the earlier host ACT
control vectors. Universal transformer is promoted in normal builds.

- [x] step  - [x] persist  - [x] ckpt  - [x] flip  - [x] test  - [x] docs

### §7.8 hnet-lm (`build/nfn_hnet_lm_native_train`)
Presets: `hnet_lm`, `hnet_lm_modern`. Byte shards (`resolve_byte_shards`)
instead of token shards; hierarchical chunking stays from the existing loop
(`print_hnet_byte_dataset_loop_json` ~19326).

HNet now has a dedicated full Tile-ABI byte-patch transformer step behind
`NFN_NATIVE_FULL_GEOMETRY_FORWARD_BACKWARD=1`. It runs the configured byte-patch
embedding, every hidden-backbone RMSNorm/QKV/RoPE/causal-attention/SwiGLU
block, final RMSNorm, byte merge, byte LM head, and byte CE. The
reverse pass persists gradients for final/per-layer norms, separate GQA Q/K/V,
attention output, gate/up and down projections, byte patch embedding/merge,
and the LM head before shared many-tensor AdamW. The full graph uploads raw
byte token/target batches directly and widens them on device with
`nfn_native_tile_uint8_to_int64` before byte patch embedding and byte CE,
avoiding the earlier per-step host int64 byte staging vectors. HNet is promoted to
`native-trainer-covered`, `native-loop-covered`, and full-architecture
parameter persistence when built with the production and full-geometry gates.

The native patch geometry is now configuration-driven: `--byte-patch-size`
(default `4`) controls the live merge-buffer width and sampled patch objective,
while `--byte-patch-stride` (defaulting to the patch size) controls the full
Tile-ABI patch sequence length and merge expansion. The full and sampled paths
therefore no longer assume four patch vectors.

Verification in this pass: default and HNet full-macro translation units
compile; the expanded v38 native build matrix produces all 15 family
binaries; all 13 family plan JSON outputs resolve LLaMA-style MLP geometry to
`hidden_dim=2048`, `mlp_multiplier=8/3`, and `multiple_of=256` at
`model_dim=768`; explicit width/multiplier/rounding overrides are reflected in
metadata; the HNet plan reports full geometry with no missing requirements and
the HNet layout/checkpoint smoke verifies native model metadata; the universal
full-macro plan resolves custom ACT recurrence depth
and epsilon; and the universal layout/checkpoint smoke passes while recording
those controls. The diffusion full-macro layout/checkpoint smoke confirms
`denoise_head.weight` has `vocab_size * model_dim` elements and no native-only
timestep parameter. Full `moe_modern` and `auxfree_moe_jepa_evo` plans resolve
the per-layer router/expert layouts without `router.topk_scale`, retain the
non-optimizer aux-free bias where configured, and declare the expert-bias Tile
symbol plus selected route-weight backward. Follow-up live CUDA validation on
the current GPU passed reduced one-step dataset loops with checkpoint writes for
`moe-modern`, `auxfree-moe-jepa-evo`, `dense-jepa-evo`,
`semantic-dense-jepa-evo`, `semantic-router-moe`, `jamba`, `seq2seq`,
`diffusion`, `ttt-llama`, `universal-llama`, and `hnet-lm`. HNet used byte
shards with `token_batch_source: native_uint8_byte_shards`; the token families
used `/tmp/nfn-diffusion-token-shards` with vocab 256. Plan checks for the same
family matrix reported `native-trainer-covered`, `production_training_loop:
true`, `full_geometry: true`, and empty missing-requirements lists.

- [x] step  - [x] persist  - [x] ckpt  - [x] flip  - [x] test  - [x] docs

---

## §8. Dense-GPT audit — verify the "implemented" paths actually work

The registry marks `gpt`/`gpt2`/`gpt3`/`nanogpt`/`gpt2-evo` (and the
`gpt2_moa`/megakernel/modern presets) as production, but some are suspected
broken. Audit each end-to-end and record/fix breakage here:

- [x] `python tools/smoke_native_gpt_template_checkpoints.py` over every
      dense-GPT preset — record any preset that fails to produce a valid
      checkpoint. The no-NumPy metadata sweep passed all 9 dense selectors
      (`gpt`, `gpt3`, GPT-2/NanoGPT modern and megakernel selectors, and
      `gpt2_moa`) with native-info shape, size, and DONE-marker checks.
- [x] Short real run per family: `nfn train --native-cuda --base-model gpt2
      --max-steps 20 ...` — loss decreases, checkpoint written, resume works
      - Host-shell validation on 2026-07-10 completed 20 reduced-geometry GPT-2
        steps (`train_loss` 10.8249 at step 1 and 8.83214 at step 20) and wrote
        `model_00000020.bin` (91,459,072 bytes). Follow-up shell validation
        added and exercised dense GPT `--resume-from-checkpoint` weight
        warm-start from `/tmp/neuralfn-dense-gpt-live-seq64/model_00000020.bin`
        for two additional seq64 one-layer steps. The resumed run reported
        `resume_checkpoint_loaded=true`, `resume_checkpoint_step=20`,
        `resume_mode=bf16_weight_warm_start`, and
        `total_optimizer_steps_completed=22`, then wrote
        `/tmp/neuralfn-dense-gpt-resume-seq64/model_00000022.bin`
        (91,545,088 bytes) with a DONE marker and native-info
        `checkpoint_step=22`. Follow-up full-resume validation added exact
        fp32 parameter and AdamW optimizer sidecars. A fresh two-step seq64
        one-layer GPT-2 run wrote `model_00000002.bin`,
        `parameters_00000002.bin`, and `optimizer_00000002.bin`; resuming that
        checkpoint for two additional steps reported
        `resume_mode=float32_parameter_and_adamw_state_resume`,
        `resume_parameter_state_restored=true`,
        `resume_optimizer_state_restored=true`,
        `resume_sampler_seek_applied=true`, and
        `total_optimizer_steps_completed=4`. A straight four-step control
        produced byte-identical `model_00000004.bin`,
        `parameters_00000004.bin`, and `optimizer_00000004.bin` artifacts
        compared with the two-plus-two resumed run.
- [x] `--sample-checkpoint` inference from each produced checkpoint emits
      coherent tokens (no NaN logits)
      - The sampler-compatible 64-context 20-step checkpoint emitted eight
        finite greedy tokens, with finite sampled/top logits and
        `passed=true`. Fixed sampler padding to the linked SM120 path's 64-row
        attention granularity after the former 16-row preflight aborted.
- [x] `gpt2-evo` delegate preflight (`nfn_gpt2_evo_native_train`) dispatches
      to `nfn_gpt_native_train_linked --train-transformer-lm --layer-evo`
      with the resolved GPT-2 template and optimizer/evolution arguments.
- [x] `gpt2_moa` (mixture-of-activations) trains through the shared native
      GPT-2 MLP backbone: the rebuilt Tile ABI dispatches GELU/ReLU/SiLU/ReLU2,
      the loop probes all four losses at step 1 and every configured interval,
      selects the lowest-loss candidate before backward, disables GELU-only
      fused paths, and emits selection/probe counters in runtime JSON. A live
      CUDA fix now keeps MoA training-time split MLP `fc_out`/activation scratch
      allocated instead of using validation-only lazy scratch, and the no-bias
      BF16 GELU fallback kernels accept null bias pointers. Verification passed
      under `compute-sanitizer` with `ERROR SUMMARY: 0 errors`, then a two-step
      train/checkpoint run wrote `model_00000002.bin`, a resume run loaded
      `resume_checkpoint_step=2` and advanced to optimizer step 3, and native
      checkpoint sampling executed all transformer blocks plus final logits.
- [x] File findings as checklist rows below with preset + failure mode:
      - [x] `gpt2`, `gpt3`, `nanogpt`, modern/megakernel selectors, and
            `gpt2_moa`: metadata-only checkpoint coverage passed; native MoA
            ABI/setup passed. Live dense CUDA coverage now includes GPT-2 MoA
            train/checkpoint, resume, and sampler execution, plus strict
            128-row `gpt2_megakernel` and `nanogpt_megakernel` runs with
            `optimized_kernel_contract_passed=true`.

---

## §9. Verification playbook (per family flip)

1. [x] `bash tools/build_native_missing_trainers.sh` — all binaries compile
       (`-Wall -Wextra -pedantic`)
2. [x] Direct binary smoke, e.g.
       `build/nfn_mixllama_native_train --train-moe-dataset-loop --max-steps 2
       --batch-size 4 --train-seq-len 128 --output-dir artifacts/smoke_mixllama`
       → `production_training_loop: true`, `parameter_update_checksum` differs
       between runs of different step counts, sidecar bytes == elements × 4
       - `tools/smoke_native_family_live_training.py` passed real optimizer and
         checkpoint steps for 55/56 selectors in one sweep; the sole
         DeepSeek-v3 MLA layout failure was fixed by persisting `q_proj`, and
         its focused rerun passed. Canonical LLaMA, MoE, JEPA, semantic,
         DeepSeek-v4, Jamba, seq2seq, diffusion, TTT, universal, and HNet runs
         all changed the live parameter-store checksum and wrote full sidecars.
3. [x] Resume determinism: 2 steps + checkpoint + 2 resumed steps ==
       4 straight steps (checksum compare)
       - With constant LR and reduced LLaMA geometry, the uninterrupted
         four-step sidecar and the two-step plus two resumed-step sidecar are
         byte-identical (`cmp` exit 0) with shared SHA-256
         `fa63750965e3c3062335fab563d2c4000ad9ebfad40f8768dcf8227536da056c`.
         Resume restored Adam moments, BF16 shadow, absolute optimizer step,
         and sampler position; zero-initialized pooled workspaces removed the
         prior restart-dependent stale-buffer drift.
4. [x] `python tools/smoke_native_family_template_checkpoints.py` and
       `nfn infer ... verify --require-architecture-forward` on the produced
       checkpoints
       - `python tools/smoke_native_family_template_checkpoints.py
         --native-bin-dir build --output-dir
         /tmp/nfn-family-template-checkpoints-noarch --json` passed all 56
         layout checkpoint smokes and basic loadability checks. A separate
         diagnostic `--require-architecture-forward` run correctly rejected
         those sparse layout-smoke sidecars. A real live LLaMA checkpoint from
         `/tmp/nfn-family-live-training/llama/llama_step_1_native_family_model_00000000.json`
         passed `python -m cli.nfn infer --checkpoint ... --verify
         --require-architecture-forward` with
         `native_family_architecture_sidecar_forward_v1` and
         `architecture_forward_inference_used: true`.
       - `python tools/smoke_native_family_live_training.py --native-bin-dir
         build --tile-ops-lib build/libnfn_native_train_tile_ops.so
         --output-dir /tmp/nfn-family-live-training --keep-going --json`
         passed 54 non-HNet templates; a focused rerun with a tiny byte shard
         dataset passed both HNet templates, for 56/56 live template coverage.
5. [x] Targeted pytest: `pytest tests/test_native_gpt2.py -k <family>`,
       `pytest cli/tests/test_native_family_infer.py`,
       `pytest tests/test_template_presets.py`
       - `python -m pytest tests/test_native_gpt2.py -q -k 'native_family or
         missing_family' --maxfail=1` passed (`2 passed, 139 deselected`).
       - `python -m pytest cli/tests/test_native_family_infer.py -q` passed
         (`16 passed`).
       - `python -m pytest tests/test_template_presets.py -x -q` passed
         (`31 passed, 14 warnings`).
       - `python tools/check_native_no_torch_deps.py --json`, `python -m
         py_compile neuralfn/native_family.py`, and `git diff --check` passed.
6. [x] End-to-end CLI: `nfn train --native-cuda --base-model <preset>
       --max-steps 2 ...` dispatches to the family binary and reports the
       production status
       - A real reduced-geometry `python -m cli.nfn train --runtime
         native-cuda --base-model llama --template-name llama ...` run
         dispatched to `nfn_llama_native_train`, completed a GPU optimizer
         step, wrote the model/parameter/optimizer checkpoint set, and rendered
         `status: native-family-dataset-loop-ran`, `passed: true`, `steps: 1`.

Status-assertion flips to remember per family:
`tests/test_native_gpt2.py` dataset-loop sentinels (~688–751), statuses map
(~5346–5364), per-family payload asserts (`production_training_loop is False`
→ `True`); `_NATIVE_TRAIN_MODEL_REGISTRY` entry in `neuralfn/native_train.py`;
`cli/nfn.py` mirror tables; `docs/cli.md`, `README.md`, `CHANGELOG.md`.

The v50 native no-Torch gate now passes all 69 Python entrypoints, 24 shell
entrypoints, and 4 native template catalogs after rebuilding the canonical
dense-GPT trainer. The verifier expects promoted family binaries to report an
empty `native_training_missing_requirements` list. The v49 default matrix
compiled all 15 emitted binaries and every non-dense
family `--print-plan` reported covered full-geometry status with an empty
missing-requirements list. Follow-up live CUDA checks on the current GPU now
cover the reduced family matrix, dense GPT MoA train/resume/sample, and
megakernel rows below. The current retained-replay audit covers the 13
native-family production binaries for three steps with
`full_step_cuda_graph_capture_ready: true`,
`full_step_forward_backward_graph_capture_count: 1`,
`full_step_forward_backward_graph_replay_count: 2`, replay-ready telemetry, and
empty blockers. The dense GPT diagnostic sweep also runs all 9 dense selectors
for one optimizer step with Torch disabled, graph-editor tensor flow disabled,
and empty missing-requirements lists when `--allow-basic-kernel-fallback` is
explicitly set for the tiny smoke shape; normal dense GPT training still keeps
the strict optimized-kernel contract enabled by default.

Fresh verification after the retained replay promotions: rebuilding
`build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh` cleared
the stale dense-trainer refusal in shell wrappers. A default artifact-enabled
`python tools/check_native_no_torch_deps.py --json` run then passed after
refreshing stale native artifacts with their shell build scripts: 30/30
artifacts, 69/69 Python entrypoints, 24/24 shell entrypoints, and 4/4 native
template catalogs, with zero missing, stale, or forbidden artifacts.

The full LLaMA-derived production step now accumulates device-generated
gradients directly into persistent optimizer buffers by trainable parameter
name, scaled across accumulation steps. This removes the main device-to-host
gradient map for projection/attention/FFN/LM-head/embedding gradients. Direct
specialty gradient paths now also accumulate into persistent optimizer buffers,
and semantic route-evo adoption uses the Tile adopt ABI for the selected router
candidate. The current production-family evidence shows retained graph replay is
ready for the shipped family binaries; wrapper-side loss reporting and sampler
orchestration remain outside the retained graph body by design.

Native LLaMA/MoE/JEPA `_megakernel` selectors now dispatch the standard
attention block through fused causal-attention forward/backward ABI symbols,
including projections, RoPE, causal attention, merge, and output projection.
Dense-GPT megakernel selectors continue to use the packed-QKV Tile path in the
dense trainer; 128-row live CUDA runs now pass the strict optimized-kernel
contract for both dense megakernel selectors.

## §10. Megakernel presets

Fused megakernel execution for every `_megakernel` preset once its family's
production step lands:

- [x] `mixllama_fast_megakernel` (§2) — native fused causal-attention ABI
      dispatch is wired and both symbols are present in the plan contract; a
      reduced one-step live CUDA run passed and wrote a checkpoint.
- [x] `semantic_router_moe_megakernel` (§5) — native fused causal-attention
      ABI dispatch is wired and symbol-checked; a reduced one-step live CUDA run
      passed and wrote a checkpoint.
- [x] `llama_fast_megakernel`, `llama_megakernel` (§7.1) — native fused
      causal-attention forward/backward dispatch is wired and symbol-checked;
      reduced one-step live CUDA runs passed and wrote checkpoints.
- [x] `jepa_semantic_hybrid_megakernel` (§7.3) — native fused
      causal-attention ABI dispatch is wired and symbol-checked; a reduced
      one-step live CUDA run passed and wrote a checkpoint.
- [x] `nanogpt_megakernel`, `gpt2_megakernel` — dense native plans and runtime
      JSON confirm `megakernel_selection_engaged: true` and the
      `packed-qkv-bf16-row-tile` execution strategy. Strict 128-row live CUDA
      runs passed for both selectors with `optimized_kernel_contract_passed:
      true`, `lm_head_logits_linear_strategy: "tk-sm120-bf16-gemm-default"`,
      and one LM-head TK logits GEMM counted per run. A 64-row diagnostic is too
      small for the TK LM-head row multiple and correctly falls back to BF16
      GEMMEx, so dense megakernel smoke commands must use at least 128 active
      rows when the optimized-kernel contract is required.
