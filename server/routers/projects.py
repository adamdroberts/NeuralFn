from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..dependencies import get_db, require_auth
from ..models import ProjectCreateRequest
from ..services.auth_service import AuthContext, PermissionDenied, get_auth_service
from ..services.session_service import get_workspace_service

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("")
def list_projects(
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> list[dict]:
    return get_workspace_service().list_projects(db, auth.user)


@router.post("")
def create_project(
    body: ProjectCreateRequest,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        created = get_workspace_service().create_project(
            db,
            auth.user,
            name=body.name,
            description=body.description,
        )
        get_auth_service().set_active_scope(
            db,
            auth.auth_session,
            project_id=created["project"]["id"],
            session_id=created["default_session"]["id"],
        )
        return created
    except PermissionDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.get("/{project_id}")
def get_project(
    project_id: str,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        return get_workspace_service().get_project(db, auth.user, project_id)
    except PermissionDenied as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{project_id}/analytics")
def project_analytics(
    project_id: str,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        return get_workspace_service().analytics_summary(db, auth.user, project_id)
    except PermissionDenied as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
