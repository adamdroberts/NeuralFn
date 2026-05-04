from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _default_database_url(root_dir: Path) -> str:
    # SQLite keeps local development and tests self-contained while the schema
    # remains fully compatible with MySQL via SQLAlchemy + Alembic.
    return f"sqlite:///{root_dir / 'neuralfn.db'}"


def _default_artifacts_dir() -> Path:
    return Path.home() / "NeuralFn" / "artifacts"


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    database_url: str
    redis_url: str | None
    session_cookie_name: str
    session_ttl_seconds: int
    snapshots_dir: Path
    artifacts_dir: Path
    create_schema_on_startup: bool
    allow_origins: list[str]
    mcp_email: str | None
    mcp_password: str | None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root_dir = Path(__file__).resolve().parent.parent
    allow_origins_raw = os.getenv("NEURALFN_ALLOW_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173")
    redis_url_raw = os.getenv("NEURALFN_REDIS_URL", "redis://localhost:6379/1")
    if redis_url_raw and not any(redis_url_raw.endswith(f"/{i}") for i in range(16)):
        if "?" in redis_url_raw:
            parts = redis_url_raw.split("?")
            redis_url = f"{parts[0].rstrip('/')}/1?{parts[1]}"
        else:
            redis_url = f"{redis_url_raw.rstrip('/')}/1"
    else:
        redis_url = redis_url_raw

    return Settings(
        root_dir=root_dir,
        database_url=os.getenv("NEURALFN_DATABASE_URL", _default_database_url(root_dir)),
        redis_url=redis_url,
        session_cookie_name=os.getenv("NEURALFN_SESSION_COOKIE_NAME", "neuralfn_session"),
        session_ttl_seconds=int(os.getenv("NEURALFN_SESSION_TTL_SECONDS", "1209600")),
        snapshots_dir=Path(os.getenv("NEURALFN_SNAPSHOTS_DIR", root_dir / "server" / "session_snapshots")),
        artifacts_dir=Path(os.getenv("NEURALFN_ARTIFACTS_DIR", str(_default_artifacts_dir()))).expanduser(),
        create_schema_on_startup=os.getenv("NEURALFN_CREATE_SCHEMA_ON_STARTUP", "1") != "0",
        allow_origins=_split_csv(allow_origins_raw),
        mcp_email=os.getenv("NEURALFN_MCP_EMAIL"),
        mcp_password=os.getenv("NEURALFN_MCP_PASSWORD"),
    )
