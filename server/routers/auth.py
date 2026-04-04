from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from ..dependencies import get_db, require_auth, session_cookie_name
from ..models import ActiveSessionRequest, BootstrapAdminRequest, LoginRequest
from ..services.auth_service import AuthContext, AuthError, get_auth_service
from ..services.session_service import get_workspace_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/bootstrap-admin")
def bootstrap_admin(
    body: BootstrapAdminRequest,
    response: Response,
    db: Session = Depends(get_db),
    cookie_name: str = Depends(session_cookie_name),
) -> dict:
    auth_service = get_auth_service()
    workspace = get_workspace_service()
    try:
        user = auth_service.bootstrap_admin(
            db,
            email=body.email,
            password=body.password,
            display_name=body.display_name,
        )
        project, session_row = workspace.ensure_default_workspace(db, user)
        user, raw_token, auth_session = auth_service.login(db, email=body.email, password=body.password)
        auth_service.set_active_scope(
            db,
            auth_session,
            project_id=project["id"],
            session_id=session_row["id"],
        )
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response.set_cookie(
        cookie_name,
        raw_token,
        httponly=True,
        samesite="lax",
        max_age=14 * 24 * 3600,
        path="/",
    )
    return {
        "user": workspace.serialize_user(user),
        "project": project,
        "session": session_row,
    }


@router.post("/login")
def login(
    body: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
    cookie_name: str = Depends(session_cookie_name),
) -> dict:
    auth_service = get_auth_service()
    workspace = get_workspace_service()
    try:
        user, raw_token, auth_session = auth_service.login(db, email=body.email, password=body.password)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    projects = workspace.list_projects(db, user)
    if projects and not auth_session.current_project_id:
        sessions = workspace.list_sessions(db, user, projects[0]["id"])
        auth_service.set_active_scope(
            db,
            auth_session,
            project_id=projects[0]["id"],
            session_id=sessions[0]["id"] if sessions else None,
        )

    response.set_cookie(
        cookie_name,
        raw_token,
        httponly=True,
        samesite="lax",
        max_age=14 * 24 * 3600,
        path="/",
    )
    return {"user": workspace.serialize_user(user)}


@router.post("/logout")
def logout(
    response: Response,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
    cookie_name: str = Depends(session_cookie_name),
) -> dict:
    raw_token = auth.auth_session.token_hash
    # The cookie contains the raw token and is deleted client-side. The DB row
    # is already identified by the authenticated auth session.
    db.delete(auth.auth_session)
    db.commit()
    response.delete_cookie(cookie_name, path="/")
    return {"status": "logged_out"}


@router.get("/me")
def me(auth: AuthContext = Depends(require_auth)) -> dict:
    return {
        "id": auth.user.id,
        "email": auth.user.email,
        "display_name": auth.user.display_name,
        "is_admin": auth.user.is_admin,
    }


@router.put("/active-session")
def set_active_session(
    body: ActiveSessionRequest,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    workspace = get_workspace_service()
    auth_service = get_auth_service()
    if body.project_id and body.session_id:
        workspace.get_session_bundle(db, auth.user, body.project_id, body.session_id)
    auth_service.set_active_scope(
        db,
        auth.auth_session,
        project_id=body.project_id,
        session_id=body.session_id,
    )
    return {
        "project_id": body.project_id,
        "session_id": body.session_id,
    }
