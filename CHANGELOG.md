# Changelog

`README.md` captures the current product and setup story. This file captures the more detailed history behind meaningful changes, including migration notes and verification.

Future updates should append new entries here rather than replacing older notes.

## Unreleased

### 2026-04-04 README built-in neuron catalog

#### Changed

- Expanded `README.md` **Built-in neurons** into a full reference for all 58 definitions from `neuralfn/builtins.py`, grouped by role (scalar vs torch module), with notes on graph terminals (`input` / `output`), duplicate `gelu` names, and an alphabetical index.

#### Verification

- Cross-checked names and groupings against `neuralfn/builtins.py` (`BuiltinNeurons.all()` / `_BUILTIN_ATTR_MAP`).

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
