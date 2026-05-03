# Changelog

`README.md` captures the current product and setup story. This file captures the more detailed history behind meaningful changes, including migration notes and verification.

Future updates should append new entries here rather than replacing older notes.

## Unreleased

### 2026-05-03 Semantic MoE JEPA Evo architecture image correction

#### Changed

- **Architecture image reference** -- restored the Semantic MoE JEPA Evo documentation to use the original PNG asset at `docs/assets/semantic_moe_jepa_evo_architecture.png` after the SVG conversion lost formatting.

#### Verification

- `file docs/assets/semantic_moe_jepa_evo_architecture.png`
- `rg -n "semantic_moe_jepa_evo_architecture\\.svg|semantic_moe_jepa_evo_architecture\\.png" README.md CHANGELOG.md docs .cursor llms-full.txt`

### 2026-05-03 Deep documentation sync for CLI, fine-tuning, and torch templates

#### Changed

- **CLI documentation surface** -- added `docs/cli.md`, refreshed `cli/README.md`, linked CLI workflows from `README.md`, `docs/README.md`, `llms.txt`, and `docs/agent-skills.md`, and documented the current `nfn train`, `nfn infer`, and `nfn eval` workflow model.
- **Current SDK and torch-template references** -- updated Python SDK, framework-guide, and repo-local agent-skill docs for the current `ModelSpec`-first builder API, composed recipe/fine-tuning roots, qLoRA/LoRA fields, adapter checkpoint helpers, fine-tuning stages, and the 115-builtin catalog.
- **LLM and agent artifacts** -- added the `.cursor/skills/neuralfn-cli` skill, refreshed the Torch and Python SDK skills, and regenerated `llms-full.txt` from README, changelog, docs, CLI docs, and repo-local skills.
- **Example API drift cleanup** -- updated `examples/gpt_graph.py` to build a `ModelSpec` with `build_llama_spec()` and pass it to `build_gpt_root_graph(model_spec=...)`.

#### Verification

- `conda run -n NeuralFn python cli/nfn.py --help`
- `conda run -n NeuralFn python cli/nfn.py train --help`
- `conda run -n NeuralFn python cli/nfn.py infer --help`
- `conda run -n NeuralFn python cli/nfn.py eval --help`
- `conda run -n NeuralFn python examples/gpt_graph.py`
- `conda run -n NeuralFn python -m pytest cli/tests/test_nfn_cli.py -q` -> `58 passed`
- `conda run -n NeuralFn python -m py_compile examples/gpt_graph.py neuralfn/config.py neuralfn/torch_templates.py`
- Stale-reference scan over `README.md`, `docs/`, `.cursor/skills/`, `cli/README.md`, `llms.txt`, and examples found no current API-artifact hits for the old template signatures or artifact defaults.
- Known residual test failures outside this documentation pass: `conda run -n NeuralFn python -m pytest cli/tests/test_nfn_cli.py cli/tests/test_train_pretraining_file_flags.py cli/tests/test_train_tinystories_flags.py -q` currently fails in `test_pretraining_file_explicit_sentencepiece_requires_shared_model` and `test_load_val_token_dataset_falls_back_to_train_holdout_without_val_file`.

### 2026-05-03 Script path cleanup before push

#### Changed

- **Portable CLI run scripts** -- removed workstation-specific absolute `PYTHONPATH` values from the 5090 helper scripts. They now derive the project root from the script location, run from `./cli`, and keep artifact paths under `$HOME/NeuralFn/artifacts`.
- **Portable setup, cache, and history examples** -- replaced local absolute setup and verification paths in docs/changelog examples with repo-relative paths, and changed the tiktoken encoding cache default to `~/tiktoken_encodings`.

#### Verification

- Workstation-path scan over shell scripts, Python files, Markdown, and changelog examples -> no local absolute path matches.
- `bash -n 5090-mini-run.sh 5090-llama-smoke.sh 5090-llama-baseline.sh 5090-llama-overnight.sh`
- `conda run -n NeuralFn python -m py_compile ../server/dataset_manager.py`
- `conda run -n NeuralFn python -m pytest tests/test_dataset_manager_downloads.py ../tests/test_dataset_manager_variants.py -q` -> `5 passed`
- `git diff --check`

### 2026-05-03 CLI and graph-run artifact store migration

#### Breaking changes

- **Default artifact paths moved** -- implicit CLI training outputs, CLI inference graph/weights defaults, interactive inference graph picking, eval reports, and server/editor graph-run artifacts now use `~/NeuralFn/artifacts` instead of repo-local `cli/artifacts` or `server/artifacts`. Callers with hardcoded old paths should pass explicit `--output`, `--graph`, `--weights`, or `--report-path` values, or update scripts to the new shared directory.

#### Changed

- **Shared artifact root** -- CLI helpers and standalone training/inference scripts now resolve default artifacts through a shared `NEURALFN_ARTIFACTS_DIR` override, defaulting to `~/NeuralFn/artifacts`.
- **Current artifact migration** -- existing CLI graph/checkpoint/eval files were moved from `cli/artifacts` into `~/NeuralFn/artifacts`; no compatibility copy or symlink was left behind.
- **In-repo CLI import roots** -- the CLI import bootstrap now points at the enclosing NeuralFn repo and the local `cli/scripts` directory, matching the relocated in-repo CLI layout.
- **Platform artifact default** -- `server/settings.py` now defaults `NEURALFN_ARTIFACTS_DIR` to `~/NeuralFn/artifacts`, so graph-run artifacts share the same local artifact store as CLI runs.

#### Verification

- `conda run -n NeuralFn python -m pytest tests/test_nfn_cli.py tests/test_infer_megakernel_artifacts.py ../tests/test_platform_api.py::SettingsDefaultsTest -q` -> `79 passed, 68 subtests passed`
- `conda run -n NeuralFn python -m py_compile cli_utils.py scripts/cli_utils.py nfn_impl.py scripts/train_jepa_semantic.py scripts/infer_jepa_semantic.py scripts/train_llama_fast.py scripts/infer_llama_fast.py scripts/train_gpt2.py scripts/infer_gpt2.py scripts/train_nanogpt.py scripts/infer_nanogpt.py scripts/train_mixllama_fast.py scripts/infer_mixllama_fast.py scripts/train_semantic_router_moe.py scripts/train_semantic_router_moe-overnight.py scripts/infer_semantic_router_moe.py scripts/train_llama_megakernel.py scripts/infer_llama_megakernel.py ../server/settings.py ../tests/test_platform_api.py`
- `conda run -n NeuralFn python -c "..."` graph metadata check -> `checked=21 missing=0`
- `conda run -n NeuralFn python scripts/infer_jepa_semantic.py --help` and `conda run -n NeuralFn python nfn.py infer --help` both rendered help successfully.

### 2026-05-03 [Experimental] Semantic MoE JEPA Evo GPT template

#### Added

- **`semantic_moe_jepa_evo` objective and preset** -- added a full Semantic MoE JEPA Evo GPT template that combines an autoregressive decoder, chunk-level causal semantic planner, JEPA training-only target path, and a hybrid expert bank with 2 shared experts, one expert per semantic vocabulary dimension, and 8 free learned experts.
- **Chunk router module stack** -- added builtin/stage coverage for causal chunk state extraction, chunk semantic projection, chunk LSH hashing, chunk-to-token route broadcasting, semantic MoE JEPA Evo routing, route balance loss, route selection loss, and route distillation loss.
- **Route-evolution controller** -- `TorchTrainer` can now periodically run lightweight evolutionary search over the new router bias/table parameters during normal gradient training. `route_evo_fraction`, `route_evo_population`, `route_evo_mutation_scale`, and `route_evo_seed` control the cadence and search shape.
- **Architecture asset** -- added `docs/assets/semantic_moe_jepa_evo_architecture.png`, a single-image architecture infographic for the new template.

#### Changed

- **Template catalog wiring** -- `build_model_spec_from_config()`, `build_gpt_root_graph()`, `build_gpt_template_payload()`, the editor GPT template dropdown, and server-side template application now recognize `semantic_moe_jepa_evo`.
- **Semantic routing losses** -- chunked topic logits now flow through semantic alignment, route selection, route distillation, and balance losses alongside AR CE and JEPA latent alignment.
- **Preset coverage** -- the new preset is included in `tests/test_template_presets.py`, so payload generation, variant resolution, compile/forward execution, and server-side apply coverage run with the rest of the shipped catalog.

#### Verification

- `conda run -n NeuralFn python -m pytest tests/test_jepa_semantic.py -q -k "semantic_moe_jepa_evo"` -> `5 passed`
- `conda run -n NeuralFn python -m pytest tests/test_template_presets.py -x -q` -> `17 passed`
- `conda run -n NeuralFn python -m pytest tests/test_builtin_neurons.py -q` -> `7 passed`
- `conda run -n NeuralFn python -m py_compile neuralfn/config.py neuralfn/builtins.py neuralfn/torch_backend.py neuralfn/torch_templates.py tests/test_jepa_semantic.py tests/test_template_presets.py tests/test_builtin_neurons.py`
- `npm run build` (from `editor/`)
- `git diff --check`

### 2026-04-10 Harness auto-download for missing cached dataset aliases

#### Changed

- **Shared harness dataset resolver** -- the sibling training and inference harnesses under `neuralfn-sdk-harness/scripts/` now share one dataset-alias resolver rooted in `train_jepa_semantic.py`. `train_mixllama_fast.py`, `train_semantic_router_moe.py`, `infer_jepa_semantic.py`, `infer_mixllama_fast.py`, and `infer_semantic_router_moe.py` all use the same missing-alias behavior now.
- **Auto-download on cache miss by default** -- when `--dataset-alias` is missing locally, the harnesses now attempt a real `download_hf_dataset(...)` call instead of failing immediately. Standard cached-variant aliases such as `owner__repo__variant__trainN` are parsed into a download contract automatically, and all harnesses now expose explicit override flags for non-standard aliases: `--download-if-missing/--no-download-if-missing`, `--dataset-hf-path`, `--dataset-variant`, `--dataset-train-shards`, `--dataset-repo-id`, and `--dataset-remote-root-prefix`.
- **Strict validator ordering preserved** -- existing aliases are still treated as authoritative cache entries. If an alias already exists but its tokenizer-backed cached shards are inconsistent, the harness does not auto-redownload or delete it; it surfaces the original `DatasetTokenizerMismatchError` directly.

#### Verification

- `conda run -n NeuralFn python -m pytest tests/test_tokenizer_vocab_contract.py -q`
- `conda run -n NeuralFn python -m py_compile cli/scripts/train_jepa_semantic.py cli/scripts/train_mixllama_fast.py cli/scripts/train_semantic_router_moe.py cli/scripts/infer_jepa_semantic.py cli/scripts/infer_mixllama_fast.py cli/scripts/infer_semantic_router_moe.py`

### 2026-04-10 Cached tokenizer contract validation and fail-fast vocab checks

#### Changed

- **Strict cached tokenizer contract** -- tokenizer-backed `uint16_shards` aliases are now validated against their downloaded tokenizer artifacts before they are accepted, loaded for training, or used for inference. NeuralFn now scans cached shard ids, resolves the tokenizer artifact from the alias metadata, and rejects aliases whose cached ids exceed the tokenizer vocab.
- **Fail-fast torch training and trace previews** -- `TorchTrainer` and `trace_torch_graph()` now validate tokenizer-backed cached aliases against the graph vocab before they reach the old auto-resize path. Manual tensors and tokenizer-less inputs still keep the compatibility auto-expand behavior, but cached aliases with tokenizer metadata no longer silently resize embeddings or LM heads.
- **Inference preflight and decode guard** -- the sibling inference harnesses for `jepa_semantic_hybrid`, `semantic_router_moe`, and `mixllama_fast` now compare the dataset tokenizer vocab against the loaded graph/checkpoint vocab before prompt encoding or decode. When decode still sees an out-of-range token id, it now raises a controlled `ValueError` instead of surfacing the raw SentencePiece traceback.
- **Bad cache remediation path** -- cached aliases that fail the tokenizer contract are now treated as invalid cache artifacts. Variant downloads clean up the partially created alias on failure, and the recommended recovery path is to delete and rebuild or re-download the alias with matching tokenizer files.

#### Verification

- `conda run -n NeuralFn python -m pytest tests/test_dataset_manager_variants.py -q`
- `conda run -n NeuralFn python -m pytest tests/test_tokenizer_vocab_contract.py -q`
- `conda run -n NeuralFn python -m pytest tests/test_template_presets.py -x -q`
- `conda run -n NeuralFn python -m py_compile cli/scripts/infer_jepa_semantic.py cli/scripts/infer_semantic_router_moe.py cli/scripts/infer_mixllama_fast.py`

### 2026-04-10 [Experimental] semantic_router_moe preset and router-only harness

#### Added

- **`semantic_router` objective + `semantic_router_moe` preset** -- added a new experimental AR-only MixLLaMA/MoE path that isolates the semantic hash router from JEPA. The preset keeps standard causal attention and MoE expert MLPs, computes one shared semantic route from the pre-block hidden state, and trains next-token CE plus semantic-alignment loss.
- **Shared-route broadcast builtin/stage** -- added `broadcast_expert_routes_module` / `BroadcastExpertRoutesStage` so batch-level expert selections from `semantic_hash_router` can be expanded to the per-token routing tensors expected by the standard MoE dispatcher.
- **Sibling harness scripts** -- added `neuralfn-sdk-harness/scripts/train_semantic_router_moe.py` and `neuralfn-sdk-harness/scripts/infer_semantic_router_moe.py` so the router-only control experiment can be trained and sampled independently of the JEPA hybrid workflow.

#### Changed

- **Template/root graph wiring** -- `build_model_spec_from_config()`, `build_gpt_root_graph()`, and the torch template builders now recognize `semantic_router_moe` and build a root graph with `dataset_source -> (tokens, targets)` plus `semantic_data_source -> sem_targets`, matching the flat compiled contract `(tokens, targets, sem_targets)`.
- **Externally routed MoE blocks** -- the new semantic-router stage uses normal LLaMA attention blocks and standard `expert_dispatch` / `expert_combine`, but replaces the learned token gate with an externally supplied route shared across all MoE blocks in the stage.
- **Semantic-only fallback safety** -- trainer and trace-preview semantic-only paths no longer feed categorical `sem_targets` into the token embedding path. They now synthesize safe placeholder `tokens` / `targets` tensors while preserving the real `sem_targets`, which fixes preview/training failures for semantic-only graphs and control experiments.
- **Preset/skill/docs surfaces** -- the toolbar dropdown, framework guide, SDK docs, and both NeuralFn agent skills now include `semantic_router_moe` and describe it as the router-only control experiment alongside `jepa_semantic_hybrid`.

#### Verification

- `conda run -n NeuralFn python -m pytest tests/test_template_presets.py -x -q` -> `15 passed`
- `conda run -n NeuralFn python -m pytest tests/test_jepa_semantic.py -q` -> `39 passed`

### 2026-04-10 Expanded canonical semantic router vocab

#### Changed

- **Canonical vocab promotion** -- `neuralfn/data/semantic/vocab_8d.json` now contains the expanded router-oriented vocabulary, and the temporary `vocab_8d_expanded_router.json` file has been removed. All loaders, training code, inference code, and API surfaces continue to use the canonical `vocab_8d.json` path.
- **Vocabulary loader validation** -- `ConversationalVocabulary` now validates that the canonical vocab file contains exactly the expected 8 routed dimensions, list-of-string term arrays, and internally consistent optional metadata such as `term_counts` and `total_terms`.
- **Dynamic topic-count docs/tests** -- semantic docs and tests now treat `num_topics` and projector/router shapes as dynamic per-dimension values derived from the expanded vocab instead of implying a fixed 40-topic layout.

#### Breaking changes

- **Old JEPA semantic checkpoints are incompatible** -- checkpoints and interrupted artifacts trained against the previous 40-term semantic vocab are expected to fail to load correctly against the expanded canonical vocab and must be retrained.

#### Verification

- Not run in this task. User explicitly requested no local test or training execution.

### 2026-04-10 JEPA trainer semantic dataset crash fix

#### Changed

- **Trainer semantic dataset loading** -- `TorchTrainer._load_semantic_dataset()` now wraps the vocab-derived `load_training_targets()` arrays directly instead of re-casting through `np.int64`. This removes the trainer-startup `NameError` on CUDA JEPA runs after the vocab-only semantic refactor.

#### Verification

- Not run in this task. User explicitly requested no local test or training execution.

### 2026-04-10 JEPA harness startup crash fix

#### Changed

- **Semantic hash helper compatibility** -- `signature_to_bucket()` and `signature_to_float()` in `neuralfn/semantic.py` still treat `n_buckets` as the canonical low-level parameter, but they now also accept `n_sig_buckets` as a compatibility alias. Passing both with different values raises a clear `ValueError` instead of failing later in the harness startup path.
- **Normalized vocab-only target builders** -- the internal vocab-target materializers now call the low-level signature helpers with the canonical bucket argument, which removes the `TypeError` that blocked `load_training_targets()` during JEPA harness startup.
- **Harness schedule accuracy** -- `neuralfn-sdk-harness/scripts/train_jepa_semantic.py` now threads the resolved CLI `--top-k` into schedule estimation, so semantic-row counts and derived epoch/accumulation summaries no longer fall back to the preset default when you override routing width.

#### Verification

- Not run in this task. User explicitly requested no local test or training execution.

### 2026-04-09 [Experimental] Vocab-only semantic routing

#### Changed

- **Vocab-only semantic supervision** -- `vocab_8d.json` is now the authoritative semantic source for `jepa_semantic_hybrid`. `load_training_data()` remains available as a compatibility wrapper, but it now materializes deterministic vocab-derived samples instead of reading a shipped CSV. `semantic_data_source` likewise generates categorical semantic topic targets on the fly from the vocab metadata.
- **Fixed dimension-to-expert routing** -- the hybrid preset now requires exactly 8 experts, one per vocab dimension: `entity_type`, `action`, `property`, `emotion_sentiment`, `domain`, `temporal`, `causality`, and `social_register`. `top_k` is still configurable but capped to 8, training-time routing is teacher-forced from active semantic targets, and inference-time auto/manual routing uses the same map.
- **Semantic head and loss contract** -- `SemanticProjectorStage` now predicts per-dimension topic logits in addition to the internal 9-D semantic state, and `SemanticAlignmentLossStage` now applies masked categorical cross-entropy over those vocab-topic logits rather than MSE over quantized semantic vectors.
- **Inference topic overrides** -- `neuralfn-sdk-harness/scripts/infer_jepa_semantic.py` now defaults to ignore-sentinel semantic targets in auto mode and supports manual topic forcing via `--semantic-topics dimension=topic,...`. Logged routing summaries now report the resolved expert IDs and their semantic dimensions.
- **Public metadata updates** -- the semantic REST/MCP surfaces now describe the stack as 9-D instead of 15-D, and `/semantic/dimensions` now includes the fixed `expert_id` map plus `num_topics` per dimension.

#### Breaking changes

- **`sem_targets` meaning changed** -- callers must now treat `sem_targets` as categorical vocab-topic IDs with `-100` ignore sentinels in the first 8 slots plus a derived taxonomy-hash slot in position 8. The old quantized-vector interpretation is no longer valid.
- **`jepa_semantic_hybrid` expert count is fixed** -- passing any `experts` value other than `8` to `build_jepa_semantic_hybrid_spec()` now raises a `ValueError`.
- **Shipped semantic CSV removed** -- `neuralfn/data/semantic/training_100k_8d.csv` is no longer packaged or used by the semantic workflow. Consumers that depended on that file should switch to `vocab_8d.json` plus `load_training_targets()` / `load_training_data()`.

#### Verification

- Not run in this task. User explicitly requested no local test or training execution.

### 2026-04-09 JEPA harness inference probe

#### Added

- **Sibling inference script** -- `neuralfn-sdk-harness/scripts/infer_jepa_semantic.py` now provides a small CUDA-only generation probe for exported `jepa_semantic_hybrid` checkpoints. It loads the saved graph JSON and `.pt` weights, auto-detects the traced logits node (`model/softcap` or `model/lm_head`), feeds dummy `targets` and `sem_targets` into the training root graph, and autoregressively samples next tokens.
- **Cached tokenizer reuse** -- the new script reuses the cached tokenizer artifacts stored under the dataset alias and decodes prompts/output with SentencePiece when available. It also supports raw `--prompt-tokens` input so the workflow still works in token-id mode when SentencePiece is unavailable.

#### Verification

- Not run in this task. User explicitly requested no local test or training execution.

### 2026-04-09 Torch training progress logging

#### Changed

- **`TorchTrainer.train()` progress callback** -- torch training now accepts an optional `on_step` callback that receives structured warmup and optimizer-step progress dictionaries. This keeps progress reporting in caller code instead of hardwiring trainer logging policy.
- **Sibling harness console output** -- `neuralfn-sdk-harness/scripts/train_jepa_semantic.py` now configures line-buffered console logging, emits explicit startup / training / validation / export stage markers, and uses `--train-log-every` to print periodic warmup and train-step progress.

#### Verification

- Not run in this task. User explicitly requested no local test or training execution.

### 2026-04-09 Torch compile BF16 stability

#### Changed

- **Fixed compiled node dispatch** -- `CompiledTorchGraph` now instantiates fixed child modules for function nodes and executes every non-trace node directly through its child module. This removes the old `_execute_node` hot path that forced `torch.compile` to specialize one generic dispatcher across different node IDs, input arities, and mixed `Long` / BF16 / FP32 inputs.
- **Contained non-loss float promotions** -- `SemanticProjectorStage`, `SemanticMoERouterStage`, `SemanticHashRouterStage`, and `AttentionlessDecoderStage` now cast their non-loss outputs back to the incoming activation dtype before returning to the graph. Full-precision math is still used inside scalar loss reductions such as token cross-entropy and MSE losses.

#### Verification

- Not run in this task. User explicitly requested no local test or training execution.

### 2026-04-09 [Experimental] JEPA semantic routed-expert training

This is still a **research prototype**. The architecture and trainer surface remain experimental, but the routed branch is now part of the actual training objective instead of being dead wiring.

#### Changed

- **`jepa_semantic_hybrid` training objective** -- the experimental hybrid preset now trains three connected loss terms: autoregressive next-token cross-entropy on the routed expert branch, JEPA latent MSE, and semantic-alignment loss. The routed branch now hashes the pooled semantic vector, performs hash-aware expert routing, runs attention-capable experts over the full masked hidden sequence, and feeds the result into the LM head.
- **Hybrid encoder / template wiring** -- `build_jepa_semantic_encoder_graph()` now outputs `(semantic_vec, hidden)` rather than `(semantic_vec, residual)`. `build_jepa_semantic_model_stage_graph()` now consumes `tokens`, `targets`, and `sem_targets`, adds `semantic_hash_router`, `routed_attention_experts`, and per-loss `loss_scale` nodes, then combines scaled AR/JEPA/semantic losses into the final scalar loss.
- **Hybrid root graph contract** -- `build_gpt_root_graph()` now wires the `jepa_semantic_hybrid` root with a `dataset_source` emitting `tokens` and `targets`, plus the existing `semantic_data_source` emitting `sem_targets`. Dataset-backed tracing and training for the preset now populate all three roles.
- **Torch trainer profile surface** -- `TorchTrainConfig` now includes the parameter-golf-inspired split-optimizer knobs used by the hybrid harness: `optimizer_profile`, `train_batch_tokens`, `beta1`, `beta2`, `adam_eps`, `grad_clip_norm`, `warmup_steps`, `warmdown_fraction`, `max_wallclock_seconds`, `embed_lr`, `head_lr`, `tied_embed_lr`, `matrix_lr`, `scalar_lr`, `muon_momentum`, `muon_backend_steps`, `muon_momentum_warmup_start`, and `muon_momentum_warmup_steps`.
- **Torch trainer implementation** -- `TorchTrainer.train()` now supports token-budgeted gradient accumulation, optional parameter-golf split optimizers, Muon for matrix-shaped parameters, warmup priming, warmdown LR scaling, optional Muon momentum warmup, gradient clipping, and proper semantic-only fallback tensors for role layouts that include `tokens`, `targets`, and `sem_targets`.
- **Experimental builtins / stages** -- added `loss_scale_module`, `semantic_hash_router_module`, and `routed_attention_experts_module` plus the matching `LossScaleStage`, `SemanticHashRouterStage`, and `RoutedAttentionExpertsStage`.
- **Sibling SDK harness** -- `neuralfn-sdk-harness/scripts/train_jepa_semantic.py` is now a CUDA-only, step-driven entrypoint centered on `max_steps`, derived epochs, JEPA-tuned defaults, the new trainer profile surface, and explicit logging of adapted versus ignored `parameter-golf/train_gpt.py` knobs.

#### Breaking changes

- **Hybrid compiled/input contract** -- callers that previously ran `CompiledTorchGraph` for `jepa_semantic_hybrid` with two flat inputs `(tokens, sem_targets)` must now supply `(tokens, targets, sem_targets)`. The preset's `dataset_source` port layout likewise changed from `["tokens"]` to `["tokens", "targets"]`.

#### Verification

- `conda run -n NeuralFn python -m pytest tests/test_jepa_semantic.py -q` -> `29 passed`
- `conda run -n NeuralFn python -m pytest tests/test_template_presets.py -x -q` -> `15 passed`
- `conda run -n NeuralFn python -m pytest tests/test_builtin_neurons.py -q` -> `4 passed`
- `PYTHONPATH=. conda run -n NeuralFn python cli/scripts/train_jepa_semantic.py --help` -> CLI import and argument surface verified
- `conda run -n NeuralFn python -c "import torch; print(torch.cuda.is_available())"` -> `True`

### 2026-04-08 [Experimental] Hybrid JEPA Semantic LLM preset

This is a **research prototype**, not a stable feature. All APIs, data formats, and architectural decisions introduced here are experimental and may change significantly or be removed based on findings.

#### Added

- **`jepa_semantic_hybrid` GPT template preset** -- a new first-class preset loaded via `load_gpt_template(preset="jepa_semantic_hybrid")`. Combines JEPA self-supervised learning with a 9-dimensional grounded semantic space, LSH-based hashing, semantic MoE routing, and an attention-less decoder stage.
- **9D semantic space** -- 8 vocabulary-grounded dimensions (`entity_type`, `action`, `property`, `emotion_sentiment`, `domain`, `temporal`, `causality`, `social_register`) plus a 9th taxonomy hash dimension derived from the entity/action/domain signature.
- **`neuralfn/semantic.py`** -- `SemanticMatrix`, `SemanticHasher`, `ConversationalVocabulary` (loads `vocab_8d.json`, encodes/decodes 9D vectors), `signature_to_float()` (deterministic MD5-based taxonomy hash), `load_training_data()` compatibility helpers, and `generate_synthetic_semantic_data()` (uses vocabulary-backed samples when available).
- **Shipped data assets** in `neuralfn/data/semantic/`: `vocab_8d.json` (core vocabulary terms and dimension metadata).
- **7 new `nn.Module` stages** in `neuralfn/torch_backend.py`: `SemanticDataSourceStage`, `SemanticProjectorStage`, `SemanticAlignmentLossStage`, `SemanticHasherStage`, `SemanticMoERouterStage` (legacy compatibility), `AttentionlessDecoderStage` (legacy compatibility), and `SoftmaxDistillationLossStage`.
- **Dual data source wiring** -- the `jepa_semantic_hybrid` root graph has both a `tokens_in` terminal (for regular text datasets) and a `semantic_data_source` node (auto-loads 100k training rows). Both are visible in the editor. The model subgraph has two input ports: `tokens` feeds the JEPA mask/encoder pipeline, `sem_targets` feeds the `SemanticAlignmentLossStage` for supervised vocabulary alignment. Training combines JEPA self-supervised loss with semantic alignment loss via `aux_loss_add`.
- **Session-scoped template apply path** -- the editor now loads `jepa_semantic_hybrid` through the session `/templates/gpt/apply` endpoint instead of piecing together a preview payload client-side. This ensures the exact backend-built root graph is persisted and displayed, including root nodes, edges, and `input_node_ids` / `output_node_ids`.
- **Semantic trace preview fixes** -- `trace_torch_graph` and the `/trace/torch` route now treat `semantic_data_source` as a built-in data source, skip project dataset access checks for the `__semantic_builtin__` marker, and generate both `tokens` and `sem_targets` preview inputs correctly. This removes the red-button preview failure before training starts.
- **Template payload `extra_nodes`** -- `build_gpt_template_payload` now returns `extra_nodes` and `extra_edges` so the editor's `onAddGPT` handler can display root-level nodes like `semantic_data_source` alongside the model subgraph node.
- **7 new builtin neuron defs** in `neuralfn/builtins.py`: `semantic_data_source_module`, `semantic_projector_module`, `semantic_alignment_loss_module`, `semantic_hasher_module`, `semantic_moe_router_module`, `attentionless_decoder_module`, `softmax_distillation_loss_module`.
- **`build_jepa_semantic_hybrid_spec()`** in `neuralfn/config.py` with new `ModelSpec` fields: `semantic_dim` (default 9), `semantic_residual_dim`, `semantic_n_lsh_tables`, `semantic_n_lsh_planes`, `semantic_table_path`.
- **`build_jepa_semantic_encoder_graph()`** and **`build_jepa_semantic_model_stage_graph()`** in `neuralfn/torch_templates.py`.
- **`"jepa_semantic"` objective type** added to `ObjectiveType` literal and handled in `build_gpt_root_graph` and `TorchTrainer`.
- **4 new MCP tools**: `reverse_engineer_to_semantic`, `semantic_search`, `train_jepa_semantic`, `generate_with_semantics`.
- **4 new REST endpoints**: `POST .../semantic/encode`, `POST .../semantic/search`, `GET .../semantic/dimensions`, `POST .../semantic/generate`.
- **`SemanticInferenceCache`** subclass and `export_semantic_tables` / `import_semantic_tables` helpers in `neuralfn/inference.py`.
- **`tests/test_jepa_semantic.py`** with 15 tests covering the semantic data layer, all new stages, and end-to-end preset compile/forward/training.
- **Python packaging metadata** -- added `pyproject.toml` so NeuralFn can be installed via `pip install -e .` (or `pip install -e /path/to/NeuralFn` from a sibling project). The editable install includes `neuralfn/data/semantic/*.json` and `*.csv` as package data for SDK consumers.

#### Notes

- The preset uses the JEPA latent MSE loss as its sole training signal during the initial self-supervised phase. The decoder/hasher/router paths are wired in the graph for inference and later distillation phases.
- The `TorchTrainer` now treats `objective == "jepa_semantic"` the same as `"jepa"` for EMA target updates.
- Semantic data artifacts (`neuralfn/data/semantic/`) are generated, not tracked in git.

### 2026-04-08 Full SDK documentation and agent skills

#### Added

- Comprehensive `docs/` directory with 50+ markdown pages covering every part of the platform:
  - **Framework guide** (`docs/framework-guide/`): 9 tutorial-oriented pages teaching developers how to build with NeuralFn in Python -- defining neurons, building graphs, subgraphs/variants, torch models, templates/presets, training workflows, inference/export, and datasets.
  - **Python SDK reference** (`docs/python-sdk/`): 15 pages documenting every public class, function, method, property, and type in the `neuralfn` package including all 58 builtin neurons, all 16 presets, all 40+ Stage classes, and all training methods.
  - **REST API reference** (`docs/rest-api/`): 8 pages covering all 60+ HTTP endpoints with method, path, auth requirements, request/response shapes, and error codes.
  - **MCP tools reference** (`docs/mcp/`): 7 pages documenting all 35+ MCP tools with parameters, descriptions, and workflow examples.
  - **Server internals** (`docs/server/`): 6 pages covering Settings, ORM models, auth system, services, and Pydantic models.
  - **Editor reference** (`docs/editor/`): 6 pages covering the TypeScript API client, Zustand store, graph utilities, components, and pages.
  - **Testing guide** (`docs/testing.md`): test suite overview with targeted check commands.
  - **Agent skills page** (`docs/agent-skills.md`): links to all AI coding agent skills with descriptions.
- Three AI coding agent skills in `.cursor/skills/`:
  - `neuralfn-python-sdk`: teaches agents how to build graphs with the core Python SDK.
  - `neuralfn-torch`: teaches agents how to build, train, and export torch-backed models using presets and the template system.
  - `neuralfn-mcp` (updated): expanded with all 16 presets, full config key table, missing tools (`load_dataset_source`, `poll_training_status`, `get_training_status`, `set_dataset_access`), and non-AR workflow examples.
- `AGENTS.md` updated with documentation maintenance rules: any change to public APIs must update the corresponding `docs/` page and relevant agent skills.
- `README.md` updated with link to `docs/` directory.

#### Notes

- The framework guide and API reference are complementary: the guide teaches how to build, the reference has exact signatures. They cross-link to each other.
- All docs use relative markdown links for GitHub navigation.
- Agent skills are designed to stay under 500 lines for optimal context window usage, with supporting reference files where needed.

### 2026-04-08 JEPA block masking

#### Added

- `JEPAMaskStage` now supports a `mask_strategy` parameter: `"random"` (default, backward-compatible i.i.d. per-token masking) and `"block"` (contiguous span masking). Block masking samples `num_blocks` contiguous spans per sequence with lengths drawn uniformly from `[min_block_ratio * seq_len, max_block_ratio * seq_len]`, forcing the predictor to reason about larger semantic structures rather than interpolating from adjacent unmasked tokens.
- New `ModelSpec` fields: `jepa_mask_strategy` (str), `jepa_num_blocks` (int, default 4), `jepa_min_block_ratio` (float, default 0.1), `jepa_max_block_ratio` (float, default 0.25). All wired through `_base_model_spec` and the `build_jepa_model_stage_graph` template builder.
- Default `jepa_mask` builtin neuron config extended with the new keys.
- Two new tests in `tests/test_template_presets.py`:
  - `test_jepa_block_masking_produces_contiguous_spans` — verifies block masks produce contiguous spans of at least `min_block_len`, masked positions are replaced, unmasked positions are preserved, and random mode still produces scattered masks.
  - `test_jepa_block_masking_config_wires_through_template` — verifies `ModelSpec` fields propagate through the template builder into the compiled graph's module config.

#### Verification

- `python -m pytest tests/test_template_presets.py -q` — all tests pass, including the existing EMA target encoder test.

### 2026-04-08 Implement backend_capabilities: cache, quantized_export, megakernel, PCA KV cache

#### Added

- `resolve_backend_capabilities(spec)` in `neuralfn/config.py` auto-derives capability flags from `TemplateSpec` fields (`runtime`, `compression`, etc.) and is called from `_base_model_spec()` so every preset gets a correct capability map.
- `FusedCausalAttentionStage` in `neuralfn/torch_backend.py` combines QKV projection, reshape, RoPE, SDPA, merge, and output projection into a single `nn.Module` for the megakernel runtime's aggressive kernel fusion scope.  Registered as `fused_causal_attention` builtin module.
- `export_quantized_pt(graph, path, scheme)` and `import_quantized_pt(graph, path)` in `neuralfn/inference.py` supporting `int8` (per-channel) and `ternary` quantization schemes.
- `InferenceCache` class in `neuralfn/inference.py` for stateful autoregressive generation with KV cache management.  Reads device from the graph's `torch_config` and handles both training (tokens+targets) and inference-only graphs.
- New presets `llama_megakernel` (`runtime="megakernel"`) and `kv_pca_llama` (`compression="kv_pca"`) in `neuralfn/config.py`, registered in `build_model_spec_from_config`.
- `build_dense_attention_graph` now accepts `enable_cache`, `enable_pca`, `pca_compressed_dim`, and `fused_megakernel` flags to optionally insert KV cache read/write nodes, PCA encode/decode nodes, or collapse to a single fused attention node.
- 24 new tests in `tests/test_backend_capabilities.py` covering capabilities resolution, megakernel forward, PCA attention, KV cache graph structure, quantized export round-trips, runtime wiring, inference cache, and new preset registration.

#### Changed

- `TorchTrainer.train()` now reads `template.runtime` from the graph's serialized `template_spec` to select the compilation mode (`eager` → none, `compile` → `torch.compile`, `megakernel` → `torch.compile(mode="max-autotune", fullgraph=True)`).  `TorchTrainConfig.compile` still acts as an override when explicitly True.
- `KVQuantPackStage` upgraded from a plain concat stub to real int8 quantization with per-token scale factors.  `KVQuantUnpackStage` performs the inverse dequantization.
- `TemplateSpec.backend_capabilities` defaults updated: `cache` and `quantized_export` now default to `True`.
- All template graph builders (`build_model_stage_graph`, `build_hidden_backbone_graph`, `build_seq2seq_model_stage_graph`, `build_diffusion_model_stage_graph`) now forward PCA/megakernel flags to `build_dense_attention_graph` via `_attn_flags()`.
- `test_kv_quant.py` tolerance widened from exact match to `atol=0.02` to account for int8 quantization noise.
- Built-in neuron catalog test updated to include `fused_causal_attention_module` (80 entries).

#### Verification

- `python -m pytest tests/test_backend_capabilities.py tests/test_template_presets.py tests/test_kv_quant.py tests/test_kv_pca.py tests/test_builtin_neurons.py -q` → 41 passed.

### 2026-04-08 Gitignore for local data and caches

#### Changed

- Expanded `.gitignore` to exclude SQLite databases, `server/datasets/`, `server/session_snapshots/`, `server/artifacts/`, local `.env` files, Python/Node/tool caches (including `*.tsbuildinfo`), coverage artifacts, and common log/OS junk. Documented this next to the platform configuration table in `README.md`.

### 2026-04-05 Remaining roadmap templates and training wiring

#### Added

- Added the remaining shipped template specs and graph builders for `ttt_llama`, `llm_jepa`, `hnet_lm`, and `universal_llama`.
- Added new torch module stages and builtins for JEPA masking/pooling/projector/predictor/loss, raw-byte patch embedding/merge, ACT halting, universal recurrence, and internal diffusion timestep sampling.
- Added raw-byte dataset loading helpers plus new regression coverage in `tests/test_template_presets.py` for preset routing, JEPA EMA behavior, H-Net byte loading, Universal halting, and dataset-source role wiring.

#### Changed

- `build_gpt_root_graph()` now persists a serialized `template_spec` into `graph.torch_config` and routes objective-specific graphs explicitly for AR, Seq2Seq, Diffusion, JEPA, H-Net, and Universal templates.
- `build_gpt_template_payload()` and the session/template application path now resolve the full shipped preset catalog consistently, including the previously broken `seq2seq` and `diffusion` payload paths.
- Dataset-backed tracing, `dataset_source` insertion, and torch training now route by input role instead of assuming only `(tokens, targets)` or `(enc_tokens, dec_tokens, targets)`.
- H-Net training now switches to raw-byte dataset loading automatically and enforces `vocab_size == 256`.
- The editor template picker and MCP `load_gpt_template` surface now reflect the expanded preset set instead of only the older NanoGPT/GPT-2/LLaMA/MoE subset.
- The built-in neuron catalog test now tracks the current 79-entry builtin registry rather than the obsolete 37-entry snapshot.

#### Removed

- Removed the temporary root-level debug helpers `debug_pt.py`, `debug_start_run.py`, `debug_start_run_2.py`, `tmp_update.py`, `tmp_verify.py`, `tmp_templates.py`, and `tmp_update_builtins.py`.

#### Verification

- Verified the directly affected tests with `python -m pytest tests/test_template_presets.py tests/test_builtin_neurons.py tests/test_diffusion.py tests/test_seq2seq.py tests/test_server_dataset_loading.py -q` (`16 passed`).
- Verified the updated legacy nested-graph wrappers with `python -m pytest tests/test_server_nested_graphs.py -k "gpt_template_route_returns_variant_library_payload or torch_trace_can_sample_from_dataset_source" -q` (`2 passed`).

### 2026-04-04 README built-in neuron catalog

#### Changed

- Expanded `README.md` **Built-in neurons** into a full reference for all 58 definitions from `neuralfn/builtins.py`, grouped by role (scalar vs torch module), with notes on graph terminals (`input` / `output`), duplicate `gelu` names, and an alphabetical index.

#### Verification

- Cross-checked names and groupings against `neuralfn/builtins.py` (`BuiltinNeurons.all()` / `_BUILTIN_ATTR_MAP`).

### 2026-04-06 Template compatibility and viewport-aware insertion

#### Changed

- Fixed the active `seq2seq` template regression by making `enc_block` and `dec_block` link to the families the preset actually exports: `enc_attention`, `dec_attention`, `cross_attention`, `mlp_dense`, and `mlp_moe`.
- Added variant-family compatibility aliases in both the backend resolver and the editor graph normalizer so older saved graphs that still refer to `attn_block`, `transformer_block`, or `mixllama` can resolve against the equivalent current family when it exists.
- Changed editor insertion defaults so toolbar actions, GPT template inserts, and variant-library inserts use the center of the visible graph viewport plus a deterministic stagger instead of toolbar-button screen coordinates or random off-screen positions.

#### Notes

- Canonical template outputs were left unchanged for the working presets. The compatibility aliases are fallback-only and still prefer an exact family match when one exists.
- The `mixllama` compatibility path is intended for older saved block-family references. It does not rewrite stored graphs; it only broadens resolution at load time.

#### Verification

- Added regression coverage in `tests/test_template_presets.py` for the reported presets (`moe`, `mixllama_fast`, `jamba`, `ternary_b158`, `seq2seq`), the `seq2seq` internal family references, and legacy family alias resolution.
- Verified with `python -m pytest tests/test_template_presets.py tests/test_seq2seq.py -q`.
- Verified the relevant nested-graph template and dataset trace paths with `python -m pytest tests/test_server_nested_graphs.py -k "gpt_template_route_returns_variant_library_payload or torch_trace_can_sample_from_dataset_source" -q`.
- Verified editor type/build wiring with `pnpm --dir editor build`.

### 2026-04-04 Codex project MCP config

#### Added

- Added project-scoped Codex MCP configuration at `.codex/config.toml` for the local `neuralfn` server using `uv run server/mcp_server.py`.

#### Changed

- Updated the MCP setup docs in `README.md` to distinguish Codex's `.codex/config.toml` from Cursor's `.cursor/mcp.json`.

#### Verification

- Verified the config format against the OpenAI Codex MCP docs for project-scoped trusted workspaces and confirmed the repo now contains `.codex/config.toml`.

### 2026-04-04 Datasets tab and personal projects

#### Added

- A dedicated `Datasets` routed surface in the React shell for downloading Hugging Face datasets, uploading local files, inspecting the project-visible catalog, and editing which accessible projects can use each dataset.
- Persistent dataset catalog storage via `dataset_assets` and `project_dataset_grants`, plus an Alembic migration to materialize the new access-control tables.
- Self-serve project creation for authenticated users, with every new project automatically seeded with a `Main session` and activated immediately in the current auth session.
- MCP dataset access management through the new `set_dataset_access` tool and optional `project_ids` sharing on dataset downloads/loads.

#### Changed

- Dataset visibility is no longer just route scoping over a shared filesystem scan. Datasets are now registered in the database and filtered by explicit project grants.
- Existing filesystem datasets under `server/datasets/` are reconciled into the DB-backed catalog on access so they remain visible after the access-control change.
- The editor no longer manages dataset selection from the bottom training strip. Dataset-backed training now resolves from the saved `dataset_source` node configuration in the session graph.
- The training panel is simplified to manual JSON entry plus run status/trace output, while dataset download/upload flows live in the new `Datasets` tab.

#### Operational notes

- Apply the new Alembic revision after the platform foundation migration to create `dataset_assets` and `project_dataset_grants`.
- Environments that still rely on `NEURALFN_CREATE_SCHEMA_ON_STARTUP=1` will auto-create the new tables on startup because they are part of the SQLAlchemy metadata.
- The first dataset catalog request after upgrading reconciles any existing on-disk datasets into the DB catalog and grants them to the projects that already exist at that point.

#### Verification

- Verified backend imports and bytecode with `python -m compileall server tests/test_platform_api.py`.
- Added platform API coverage for non-admin project creation, dataset grant filtering, and graph-driven dataset-backed runs in `tests/test_platform_api.py`.
- Verified the frontend route and type wiring with `pnpm --dir editor build`.
- Attempted to run `uv run --with-requirements requirements.txt python -m unittest discover -s tests -p "test_platform_api.py"`, but this environment could not resolve PyPI to install missing Python dependencies (`fastapi`, `torch`, `tiktoken`, etc.).

### 2026-04-04 Platform foundation

#### Added

- SQLAlchemy-backed persistence for users, auth sessions, projects, memberships, editor sessions, session snapshots, and training runs.
- Alembic migration scaffolding for the durable platform schema, with SQLite as the default local database and MySQL-ready configuration through `NEURALFN_DATABASE_URL`.
- Built-in authentication with bootstrap-admin flow, login/logout endpoints, active-session selection, PBKDF2 password hashing, opaque session tokens, and HTTP-only session cookies.
- Project-scoped datasets plus project/session-scoped graph, session, and run APIs under `/api/projects/{project_id}/...`.
- A routed React app shell with dedicated Editor, Runs, Analytics, and Admin surfaces.
- Refresh-safe session hydration/autosave flow that loads graphs by project/session, tracks revisions, and reloads after `409` conflicts.
- Optional Redis-backed live state for session graph state, run events, and agent coordination, with in-memory fallback for local development.
- MCP authentication and tool scoping so graph/training tools now operate on explicit `project_id` and `session_id` context.

#### Changed

- The platform no longer assumes a single anonymous in-memory graph. Workspace state is now organized by authenticated user, project, and editor session.
- The frontend now boots through `/api/bootstrap`, routes through `/login` and `/app/...`, and persists the active project/session on the server-side auth session.
- Training status and session restore behavior are no longer tied to global process state; they flow through the scoped services and live-state store.
- Legacy helper wrappers remain in `server/routes.py` only to keep older route-oriented tests working against a dedicated legacy workspace.

#### Operational notes

- Local startup defaults to `sqlite:///neuralfn.db` plus filesystem snapshots/artifacts unless overridden with environment variables.
- For migration-managed environments, run `alembic upgrade head` and set `NEURALFN_CREATE_SCHEMA_ON_STARTUP=0`.
- `NEURALFN_ALLOW_ORIGINS` must include the frontend origin because the app uses cookie-authenticated cross-origin requests during local development.
- MCP clients must provide `NEURALFN_MCP_EMAIL` and `NEURALFN_MCP_PASSWORD`, and may override `NEURALFN_BASE_URL` when the API is not hosted at `http://localhost:8000/api`.

#### Verification

- Added `tests/test_platform_api.py` to cover bootstrap-admin, active-session switching, refresh-safe graph restore, idle run status, and revision-conflict handling.
- Verified backend imports/bytecode with `python -m compileall server tests/test_platform_api.py`.
- Verified the new platform API coverage with `python -m unittest discover -s tests -p "test_platform_api.py"`.
- Verified the frontend wiring with `cd editor && pnpm build`.
