from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .settings import Settings, get_settings


class Base(DeclarativeBase):
    pass


def _engine_kwargs(settings: Settings) -> dict:
    kwargs = {"future": True, "pool_pre_ping": True}
    if settings.database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return kwargs


@lru_cache(maxsize=1)
def get_engine():
    settings = get_settings()
    return create_engine(settings.database_url, **_engine_kwargs(settings))


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, expire_on_commit=False)


def init_db() -> None:
    from . import db_models  # noqa: F401

    settings = get_settings()
    settings.snapshots_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    if settings.create_schema_on_startup:
        Base.metadata.create_all(bind=get_engine())


def get_db_session() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
