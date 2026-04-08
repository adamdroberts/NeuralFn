from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from .db import get_db_session
from .settings import get_settings
from .services.auth_service import AuthContext, get_auth_service
from .services.run_service import get_run_service
from .services.session_service import get_workspace_service


def get_db() -> Session:
    yield from get_db_session()


def get_optional_auth(
    request: Request,
    db: Session = Depends(get_db),
) -> AuthContext | None:
    settings = get_settings()
    raw_token = request.cookies.get(settings.session_cookie_name)
    if not raw_token:
        return None
    return get_auth_service().authenticate_token(db, raw_token)


def require_auth(context: AuthContext | None = Depends(get_optional_auth)) -> AuthContext:
    if context is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return context


def workspace_service():
    return get_workspace_service()


def run_service():
    return get_run_service()


def session_cookie_name() -> str:
    return get_settings().session_cookie_name
