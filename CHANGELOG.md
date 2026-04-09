# Changelog

`README.md` captures the current product and setup story. This file captures the more detailed history behind meaningful changes, including migration notes and verification.

Future updates should append new entries here rather than replacing older notes.

## Unreleased

### 2026-04-08 [Experimental] Hybrid JEPA Semantic LLM preset

This is a **research prototype**, not a stable feature. All APIs, data formats, and architectural decisions introduced here are experimental and may change significantly or be removed based on findings.

#### Added

- **`jepa_semantic_hybrid` GPT template preset** -- a new first-class preset loaded via `load_gpt_template(preset="jepa_semantic_hybrid")`. Combines JEPA self-supervised learning with a 9-dimensional grounded semantic space, LSH-based hashing, semantic MoE routing, and an attention-less decoder stage.
- **9D semantic space** -- 8 vocabulary-grounded dimensions (`entity_type`, `action`, `property`, `emotion_sentiment`, `domain`, `temporal`, `causality`, `social_register`) plus a 9th taxonomy hash dimension derived from the `semantic_signature` (entity+action+domain trigram) via learned softmax compression. Grounded in a 320-term vocabulary (40 terms per dimension) extracted from 100k semantic matrix analysis.
- **`neuralfn/semantic.py`** -- `SemanticMatrix`, `SemanticHasher`, `ConversationalVocabulary` (loads `vocab_8d.json`, encodes/decodes 9D vectors), `signature_to_float()` (deterministic MD5-based taxonomy hash), `load_training_data()` (reads the shipped 100k-row CSV into matrix format), and `generate_synthetic_semantic_data()` (now uses real vocabulary when available).
- **Shipped data assets** in `neuralfn/data/semantic/`: `vocab_8d.json` (320 core vocabulary terms, 8 dims x 40 terms, with coverage stats) and `training_100k_8d.csv` (100k training rows with 88,362 unique semantic signatures).
- **7 new `nn.Module` stages** in `neuralfn/torch_backend.py`: `SemanticDataSourceStage` (auto-loads the shipped 100k-row semantic CSV), `SemanticProjectorStage` (model_dim -> 9D with learned signature softmax + residual), `SemanticAlignmentLossStage`, `SemanticHasherStage` (LSH in-graph), `SemanticMoERouterStage` (cosine-to-centroid routing), `AttentionlessDecoderStage` (bucket-conditioned decode), `SoftmaxDistillationLossStage`.
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
