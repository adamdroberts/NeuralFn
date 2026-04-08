from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..dependencies import get_db, require_auth
from ..models import CreateUserRequest, ProjectMembershipRequest
from ..services.auth_service import AuthContext, AuthError, PermissionDenied, get_auth_service
from ..services.session_service import get_workspace_service

router = APIRouter(prefix="/admin", tags=["admin"])


def _require_admin(auth: AuthContext) -> None:
    if not auth.user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")


@router.get("/users")
def list_users(
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> list[dict]:
    _require_admin(auth)
    workspace = get_workspace_service()
    return [workspace.serialize_user(user) for user in get_auth_service().list_users(db)]


@router.post("/users")
def create_user(
    body: CreateUserRequest,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    _require_admin(auth)
    try:
        user = get_auth_service().create_user(
            db,
            email=body.email,
            password=body.password,
            display_name=body.display_name,
            is_admin=body.is_admin,
        )
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return get_workspace_service().serialize_user(user)


@router.get("/projects/{project_id}/memberships")
def list_memberships(
    project_id: str,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> list[dict]:
    _require_admin(auth)
    return get_workspace_service().list_project_memberships(db, auth.user, project_id)


@router.post("/projects/{project_id}/memberships")
def upsert_membership(
    project_id: str,
    body: ProjectMembershipRequest,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    _require_admin(auth)
    auth_service = get_auth_service()
    workspace = get_workspace_service()
    target_user = None
    if body.user_id:
        target_user = next((user for user in auth_service.list_users(db) if user.id == body.user_id), None)
    elif body.email:
        target_user = auth_service.get_user_by_email(db, body.email)
    if target_user is None:
        raise HTTPException(status_code=404, detail="Target user not found")
    try:
        return workspace.add_project_membership(
            db,
            auth.user,
            project_id=project_id,
            target_user=target_user,
            role=body.role,
        )
    except PermissionDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
