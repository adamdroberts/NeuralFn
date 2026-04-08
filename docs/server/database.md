# Database Layer

## Engine and Session Setup

`server/db.py` provides the core database plumbing:

- **`Base`** -- SQLAlchemy `DeclarativeBase` subclass used by all ORM models.
- **`get_engine()`** -- creates and caches the SQLAlchemy engine from `settings.database_url`.
- **`get_session_factory()`** -- returns a `sessionmaker` bound to the engine.
- **`init_db()`** -- calls `Base.metadata.create_all()` to create tables (used when `create_schema_on_startup` is enabled).
- **`get_db_session()`** -- FastAPI dependency that yields a scoped `Session` and handles commit/rollback.
- **`session_scope()`** -- context manager for use outside of request handlers (background workers, CLI scripts).

The default backend is **SQLite** (file-based, zero-config). **MySQL** is supported via the PyMySQL driver (`mysql+pymysql://` URL scheme).

## Migrations

Alembic migrations live in the `alembic/` directory at the repository root. When `create_schema_on_startup` is disabled, run migrations manually:

```bash
alembic upgrade head
```

## ORM Models

All models are defined in `server/db_models.py`.

### User

| Column | Type | Notes |
|--------|------|-------|
| `id` | String (UUID) | Primary key |
| `email` | String | Unique |
| `display_name` | String | |
| `password_hash` | String | bcrypt hash |
| `is_admin` | Boolean | |
| `created_at` | DateTime | UTC |
| `updated_at` | DateTime | UTC |

### AuthSession

| Column | Type | Notes |
|--------|------|-------|
| `id` | String (UUID) | Primary key |
| `token_hash` | String | Unique, SHA-256 of session token |
| `user_id` | String | FK to User |
| `current_project_id` | String | Nullable, FK to Project |
| `current_editor_session_id` | String | Nullable, FK to EditorSession |
| `created_at` | DateTime | UTC |
| `last_seen_at` | DateTime | UTC |
| `expires_at` | DateTime | UTC |

### Project

| Column | Type | Notes |
|--------|------|-------|
| `id` | String (UUID) | Primary key |
| `slug` | String | Unique |
| `name` | String | |
| `description` | String | Nullable |
| `created_by_user_id` | String | FK to User |
| `created_at` | DateTime | UTC |
| `updated_at` | DateTime | UTC |

### ProjectMembership

| Column | Type | Notes |
|--------|------|-------|
| `id` | String (UUID) | Primary key |
| `project_id` | String | FK to Project |
| `user_id` | String | FK to User |
| `role` | String | e.g. "owner", "editor", "viewer" |
| `created_at` | DateTime | UTC |

Unique constraint on `(project_id, user_id)`.

### EditorSession

| Column | Type | Notes |
|--------|------|-------|
| `id` | String (UUID) | Primary key |
| `project_id` | String | FK to Project |
| `name` | String | |
| `description` | String | Nullable |
| `branch_name` | String | Nullable |
| `latest_revision` | Integer | Monotonically increasing |
| `created_by_user_id` | String | FK to User |
| `updated_by_user_id` | String | Nullable, FK to User |
| `created_at` | DateTime | UTC |
| `updated_at` | DateTime | UTC |

### SessionSnapshot

| Column | Type | Notes |
|--------|------|-------|
| `id` | String (UUID) | Primary key |
| `project_id` | String | FK to Project |
| `session_id` | String | FK to EditorSession |
| `revision` | Integer | Snapshot revision number |
| `reason` | String | Nullable, human-readable reason |
| `storage_path` | String | Path to snapshot JSON file |
| `created_by_user_id` | String | FK to User |
| `created_at` | DateTime | UTC |

### TrainingRun

| Column | Type | Notes |
|--------|------|-------|
| `id` | String (UUID) | Primary key |
| `project_id` | String | FK to Project |
| `session_id` | String | FK to EditorSession |
| `started_by_user_id` | String | FK to User |
| `status` | String | e.g. "pending", "running", "completed", "failed", "stopped" |
| `requested_method` | String | Method requested by the user |
| `resolved_method` | String | Nullable, method actually used |
| `graph_name` | String | Nullable |
| `dataset_names` | JSON | List of dataset names used |
| `seq_len` | Integer | Nullable |
| `last_loss` | Float | Nullable |
| `last_step` | Integer | Nullable |
| `error` | String | Nullable, error message on failure |
| `created_at` | DateTime | UTC |
| `started_at` | DateTime | Nullable, UTC |
| `updated_at` | DateTime | UTC |
| `completed_at` | DateTime | Nullable, UTC |

### DatasetAsset

| Column | Type | Notes |
|--------|------|-------|
| `id` | String (UUID) | Primary key |
| `name` | String | Unique |
| `source` | String | e.g. "huggingface", "upload" |
| `hf_path` | String | Nullable, HuggingFace dataset path |
| `hf_split` | String | Nullable |
| `text_column` | String | Nullable |
| `num_tokens` | Integer | Nullable |
| `num_rows` | Integer | Nullable |
| `variant` | String | Nullable |
| `train_shards` | Integer | Nullable |
| `val_shards` | Integer | Nullable |
| `data_format` | String | Nullable |
| `repo_id` | String | Nullable |
| `remote_root_prefix` | String | Nullable |
| `created_by_user_id` | String | FK to User |
| `created_at` | DateTime | UTC |
| `updated_at` | DateTime | UTC |

### ProjectDatasetGrant

| Column | Type | Notes |
|--------|------|-------|
| `id` | String (UUID) | Primary key |
| `project_id` | String | FK to Project |
| `dataset_id` | String | FK to DatasetAsset |
| `created_at` | DateTime | UTC |

Unique constraint on `(project_id, dataset_id)`.

## Helper Functions

`server/db_models.py` also provides:

- **`utcnow()`** -- returns the current time as a timezone-aware UTC datetime.
- **`ensure_utc(dt)`** -- normalizes a datetime to UTC, attaching tzinfo if naive.
- **`uuid_str()`** -- generates a new UUID4 as a string, used for primary keys.
