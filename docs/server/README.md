# NeuralFn Server Architecture

The NeuralFn backend is a **FastAPI** application defined in `server/app.py`. It manages neural-network graph state, training runs, datasets, authentication, and exposes both a REST API and an MCP bridge for agent-driven workflows.

## Application Lifecycle

The FastAPI app uses a lifespan context manager to:

1. Initialize the database (optionally creating the schema on startup).
2. Start the `PersistenceWorker` for async DB writes.
3. Mount routers and configure CORS.
4. On shutdown, drain the persistence queue and close connections.

CORS origins are configured via the `NEURALFN_ALLOW_ORIGINS` environment variable (defaults to the Vite dev server URLs).

## Router Tree

All REST endpoints live under the `/api` prefix. Sub-routers handle distinct domains:

| Sub-router | Prefix | Responsibility |
|------------|--------|----------------|
| bootstrap | `/api/bootstrap` | First-run admin setup, app status |
| auth | `/api/auth` | Login, logout, current user |
| admin | `/api/admin` | User management, memberships |
| projects | `/api/projects` | Project CRUD, membership, analytics |
| catalog | `/api/catalog` | Builtin neuron definitions |
| datasets | `/api/datasets` | Dataset catalog, download, upload, access |
| sessions | `/api/sessions` | Session CRUD, graph read/write, execution, training |
| runs | `/api/runs` | Training run history and control |

## Service Layer

Business logic is isolated from route handlers in service classes:

- **AuthService** -- user management, login/logout, session tokens, permission checks.
- **WorkspaceService** -- projects, editor sessions, graph mutations, snapshots, analytics.
- **RunService** -- training run lifecycle, background threads, progress streaming.
- **DatasetService** -- HuggingFace dataset downloads, uploads, project-level access grants.

See [services.md](services.md) for detailed method listings.

## Persistence

- **SQLAlchemy + Alembic** for relational data (users, projects, sessions, runs, datasets). Default backend is SQLite; MySQL is also supported. See [database.md](database.md).
- **LiveStateStore** for ephemeral, high-frequency state (active graph revisions, run progress). Backed by an in-memory store or Redis when `NEURALFN_REDIS_URL` is set. See [services.md](services.md).
- **PersistenceWorker** decouples hot-path graph updates from slower DB writes by processing them through an async queue.

## MCP Bridge

`server/mcp_server.py` runs a **FastMCP** server that exposes NeuralFn operations as MCP tools. This allows AI agents to create graphs, add/wire neurons, start training, and inspect results through the standard MCP protocol. The MCP server authenticates with the same credential system as the REST API (configured via `NEURALFN_MCP_EMAIL` / `NEURALFN_MCP_PASSWORD`).

## Sub-pages

- [Configuration](configuration.md) -- environment variables and the Settings dataclass.
- [Database](database.md) -- engine setup, ORM models, migrations.
- [Authentication](authentication.md) -- password hashing, session cookies, auth middleware.
- [Services](services.md) -- WorkspaceService, RunService, DatasetService, LiveStateStore, PersistenceWorker, graph_ops.
- [Models](models.md) -- Pydantic request/response schemas.
