# Server Configuration

All server configuration is managed through the `Settings` dataclass in `server/settings.py`. Settings are loaded from environment variables with sensible defaults.

## Settings Dataclass

`Settings` is a **frozen dataclass** (immutable after construction). Each field maps to an environment variable.

| Field | Env Var | Default | Description |
|-------|---------|---------|-------------|
| `database_url` | `NEURALFN_DATABASE_URL` | `sqlite:///...neuralfn.db` | SQLAlchemy database connection URL. The default path is resolved relative to the repository root. |
| `redis_url` | `NEURALFN_REDIS_URL` | `None` | Optional Redis URL for the `RedisLiveStateStore`. When unset, ephemeral state is held in memory. |
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

## Example `.env`

```
NEURALFN_DATABASE_URL=mysql+pymysql://user:pass@localhost/neuralfn
NEURALFN_REDIS_URL=redis://localhost:6379/0
NEURALFN_SESSION_TTL_SECONDS=604800
NEURALFN_ALLOW_ORIGINS=https://app.example.com
NEURALFN_MCP_EMAIL=agent@example.com
NEURALFN_MCP_PASSWORD=secret
```
