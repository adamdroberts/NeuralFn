from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..dependencies import get_db, get_optional_auth
from ..services.auth_service import AuthContext, get_auth_service
from ..services.session_service import get_workspace_service

router = APIRouter()


@router.get("/bootstrap")
def bootstrap(
    auth: AuthContext | None = Depends(get_optional_auth),
    db: Session = Depends(get_db),
) -> dict:
    auth_service = get_auth_service()
    workspace = get_workspace_service()
    requires_setup = auth_service.setup_required(db)

    if auth is None:
        return {
            "requires_setup": requires_setup,
            "authenticated": False,
            "user": None,
            "projects": [],
            "active_project_id": None,
            "active_session_id": None,
            "active_session": None,
        }

    projects = workspace.list_projects(db, auth.user)
    if not projects and auth.user.is_admin:
        project, session_row = workspace.ensure_default_workspace(db, auth.user)
        projects = [project]
        auth_service.set_active_scope(
            db,
            auth.auth_session,
            project_id=project["id"],
            session_id=session_row["id"],
        )

    active_project_id = auth.auth_session.current_project_id
    active_session_id = auth.auth_session.current_editor_session_id

    if projects and (active_project_id is None or active_project_id not in {project["id"] for project in projects}):
        active_project_id = projects[0]["id"]
        sessions = workspace.list_sessions(db, auth.user, active_project_id)
        active_session_id = sessions[0]["id"] if sessions else None
        auth_service.set_active_scope(
            db,
            auth.auth_session,
            project_id=active_project_id,
            session_id=active_session_id,
        )

    active_session = None
    if active_project_id and active_session_id:
        try:
            active_session = workspace.get_session_detail(db, auth.user, active_project_id, active_session_id)
        except Exception:
            active_session = None

    return {
        "requires_setup": requires_setup,
        "authenticated": True,
        "user": workspace.serialize_user(auth.user),
        "projects": projects,
        "active_project_id": active_project_id,
        "active_session_id": active_session_id,
        "active_session": active_session,
    }
