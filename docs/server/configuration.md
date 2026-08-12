# Server Configuration

All server configuration is managed through the `Settings` dataclass in `server/settings.py`. Settings are loaded from environment variables with sensible defaults.

## Settings Dataclass

`Settings` is a **frozen dataclass** (immutable after construction). Each field maps to an environment variable.

| Field | Env Var | Default | Description |
|-------|---------|---------|-------------|
| `database_url` | `NEURALFN_DATABASE_URL` | `sqlite:///...neuralfn.db` | SQLAlchemy database connection URL. The default path is resolved relative to the repository root. |
| `redis_url` | `NEURALFN_REDIS_URL` | `redis://localhost:6379/1` | Optional Redis URL for shared live state and Redis-backed persistence queues. Set `NEURALFN_REDIS_URL=` to disable Redis and use in-process live state plus synchronous local persistence. |
| `session_cookie_name` | `NEURALFN_SESSION_COOKIE_NAME` | `neuralfn_session` | Name of the HTTP-only cookie used for session authentication. |
| `session_ttl_seconds` | `NEURALFN_SESSION_TTL_SECONDS` | `1209600` (14 days) | Lifetime of an authentication session in seconds. |
| `snapshots_dir` | `NEURALFN_SNAPSHOTS_DIR` | `server/session_snapshots` | Directory where session snapshot JSON files are stored. |
| `artifacts_dir` | `NEURALFN_ARTIFACTS_DIR` | `~/NeuralFn/artifacts` | Directory for training artifacts (checkpoints, logs). |
| `create_schema_on_startup` | `NEURALFN_CREATE_SCHEMA_ON_STARTUP` | `"1"` | When truthy, the database schema is created automatically on application startup. Set to `"0"` to disable (useful when relying on Alembic migrations). |
| `allow_origins` | `NEURALFN_ALLOW_ORIGINS` | Vite dev URLs | Comma-separated list of allowed CORS origins. Defaults include the standard Vite dev server addresses. |
| `mcp_email` | `NEURALFN_MCP_EMAIL` | `None` | Email address used by the MCP bridge to authenticate against the REST API. |
| `mcp_password` | `NEURALFN_MCP_PASSWORD` | `None` | Password used by the MCP bridge to authenticate against the REST API. |

## Singleton Access

```python
from server.settings import get_settings

settings = get_settings()
```

`get_settings()` returns a module-level singleton. The `Settings` instance is created once on first call and reused for the lifetime of the process.

## Root Directory

`Settings` computes `root_dir` from `__file__`, resolving to the repository root (the parent of the `server/` package directory). Relative paths such as `snapshots_dir` are resolved against `root_dir` at runtime. Artifacts default to `~/NeuralFn/artifacts` so CLI and graph-run outputs share one local store unless `NEURALFN_ARTIFACTS_DIR` overrides it.

Editor/MCP native training writes one private tree per run below
`artifacts_dir/runs/<run-id>/`: the inert editor graph, materialized Native IR
and compatibility/training-plan sidecars, then the compiled trainer's
checkpoint directory. Existing run paths are never overwritten.

## Example `.env`

```
NEURALFN_DATABASE_URL=mysql+pymysql://user:pass@localhost/neuralfn
NEURALFN_REDIS_URL=redis://localhost:6379/0
NEURALFN_SESSION_TTL_SECONDS=604800
NEURALFN_ALLOW_ORIGINS=https://app.example.com
NEURALFN_MCP_EMAIL=agent@example.com
NEURALFN_MCP_PASSWORD=secret
```

For single-process local development or tests, set `NEURALFN_REDIS_URL=` to
avoid connecting to the default local Redis URL. In that mode, live state is
held in memory and persistence updates are written synchronously in-process.

## Standalone native inference configuration

`nfn infer --serve` does not read the editor `Settings` singleton above. Its
configuration is intentionally isolated in `NativeServeConfig` and CLI flags:

| CLI/env | Default | Description |
|---|---|---|
| `--checkpoint` | required | Native artifact directory or manifest path |
| `--host` / `--port` | `127.0.0.1` / `8000` | Listening address |
| `--served-model-name` | manifest name/family | Model ID exposed under `/v1/models` |
| `--queue-capacity` | `8` | Maximum waiting requests before HTTP 429 |
| `--session-limit` | `queue_capacity + 1` | Maximum admitted running-plus-queued request-session reservations before `session_limit_exceeded` HTTP 429 |
| `--max-output-tokens` | `256` | Per-request output reservation ceiling |
| `--kv-cache` | `auto` | Lossless full cache when jointly proven; `off` selects recomputation; explicit `turboquant` selects the proved reviewed-dense packed CPU cache |
| `--turboquant-profile` | `mse-3.5` | Explicit packed-cache profile (`mse-3.5` or `qjl-3.5`), accepted only with joint artifact/binding/cache-ABI proof |
| `--chat-template` | `auto` | Artifact renderer, explicit `plain_roles`, or placeholder file |
| `--state-db` | unset | Private versioned SQLite state file enabling scoped Responses/Conversations, local compaction, durable background jobs, and resumable background-stream events |
| `--api-key-file` | unset | Private file containing accepted Bearer keys |
| `NFN_INFER_API_KEY` | unset | Single Bearer key supplied outside process arguments |
| `--allow-unauthenticated-remote` | false | Explicit override of the remote-auth safety rule |

The inference app never opens the editor database and has no Redis, CORS, MCP,
snapshot, or editor artifact-root setting. Without `--state-db` it is
stateless beyond resident request sessions. With `--state-db`, it opens only
the separate private SQLite store described above. Opening a schema-version-1
store automatically adds the scoped response-event ledger and records schema
version 2; existing response, item, conversation, compaction, and background-job
rows are retained. The file is also the source of truth for replay cursors, so
moving or deleting it invalidates resumability along with the other stateful
resources. See
[Standalone Native Inference Server](native-inference-serving.md).
