from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable

from sqlalchemy import and_, desc, select
from sqlalchemy.orm import Session

from neuralfn.graph import NeuronGraph

from ..db_models import EditorSession, Project, ProjectMembership, SessionSnapshot, TrainingRun, User, ensure_utc, utcnow
from ..settings import get_settings
from .auth_service import PermissionDenied
from .live_state import RevisionConflict, SessionGraphState, get_live_state_store


def _slugify(value: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return base or "project"


def _sanitize_path_component(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "snapshot"


@dataclass
class ProjectAccess:
    project: Project
    role: str


@dataclass
class SessionBundle:
    project: Project
    session: EditorSession
    graph_state: SessionGraphState


class WorkspaceService:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._live_state = get_live_state_store()

    def serialize_user(self, user: User) -> dict[str, Any]:
        return {
            "id": user.id,
            "email": user.email,
            "display_name": user.display_name,
            "is_admin": user.is_admin,
        }

    def _iso(self, value) -> str | None:
        normalized = ensure_utc(value)
        return normalized.isoformat() if normalized is not None else None

    def serialize_project(self, project: Project, *, role: str) -> dict[str, Any]:
        return {
            "id": project.id,
            "slug": project.slug,
            "name": project.name,
            "description": project.description,
            "role": role,
            "created_at": self._iso(project.created_at),
            "updated_at": self._iso(project.updated_at),
        }

    def serialize_session(self, session: EditorSession) -> dict[str, Any]:
        return {
            "id": session.id,
            "project_id": session.project_id,
            "name": session.name,
            "description": session.description,
            "branch_name": session.branch_name,
            "latest_revision": session.latest_revision,
            "created_at": self._iso(session.created_at),
            "updated_at": self._iso(session.updated_at),
        }

    def serialize_membership(self, membership: ProjectMembership, user: User) -> dict[str, Any]:
        return {
            "id": membership.id,
            "project_id": membership.project_id,
            "user": self.serialize_user(user),
            "role": membership.role,
            "created_at": self._iso(membership.created_at),
        }

    def list_projects(self, db: Session, user: User) -> list[dict[str, Any]]:
        if user.is_admin:
            projects = list(db.scalars(select(Project).order_by(Project.created_at.asc())))
            return [self.serialize_project(project, role="admin") for project in projects]

        rows = db.execute(
            select(Project, ProjectMembership.role)
            .join(ProjectMembership, ProjectMembership.project_id == Project.id)
            .where(ProjectMembership.user_id == user.id)
            .order_by(Project.created_at.asc())
        )
        return [self.serialize_project(project, role=role) for project, role in rows]

    def create_project(
        self,
        db: Session,
        user: User,
        *,
        name: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        base_slug = _slugify(name)
        slug = base_slug
        counter = 1
        while db.scalar(select(Project).where(Project.slug == slug)) is not None:
            counter += 1
            slug = f"{base_slug}-{counter}"
        project = Project(
            slug=slug,
            name=name.strip() or "Project",
            description=description.strip() if description else None,
            created_by_user_id=user.id,
        )
        db.add(project)
        db.flush()
        membership = ProjectMembership(project_id=project.id, user_id=user.id, role="data_scientist")
        db.add(membership)
        db.commit()
        db.refresh(project)
        session_row = self.create_session(
            db,
            user,
            project_id=project.id,
            name="Main session",
            description="Seeded workspace session",
        )
        return {
            "project": self.serialize_project(project, role="admin" if user.is_admin else "data_scientist"),
            "default_session": session_row,
        }

    def _project_access(self, db: Session, user: User, project_id: str) -> ProjectAccess:
        project = db.get(Project, project_id)
        if project is None:
            raise PermissionDenied("Project not found")
        if user.is_admin:
            return ProjectAccess(project=project, role="admin")
        membership = db.scalar(
            select(ProjectMembership).where(
                and_(
                    ProjectMembership.project_id == project_id,
                    ProjectMembership.user_id == user.id,
                )
            )
        )
        if membership is None:
            raise PermissionDenied("You do not have access to this project")
        return ProjectAccess(project=project, role=membership.role)

    def get_project(self, db: Session, user: User, project_id: str) -> dict[str, Any]:
        access = self._project_access(db, user, project_id)
        return self.serialize_project(access.project, role=access.role)

    def list_project_memberships(self, db: Session, user: User, project_id: str) -> list[dict[str, Any]]:
        self._project_access(db, user, project_id)
        memberships = list(
            db.scalars(
                select(ProjectMembership)
                .where(ProjectMembership.project_id == project_id)
                .order_by(ProjectMembership.created_at.asc())
            )
        )
        users = {member.user_id: db.get(User, member.user_id) for member in memberships}
        return [
            self.serialize_membership(member, users[member.user_id])  # type: ignore[arg-type]
            for member in memberships
            if users.get(member.user_id) is not None
        ]

    def add_project_membership(
        self,
        db: Session,
        user: User,
        *,
        project_id: str,
        target_user: User,
        role: str = "data_scientist",
    ) -> dict[str, Any]:
        if not user.is_admin:
            raise PermissionDenied("Only admins can update project memberships")
        self._project_access(db, user, project_id)
        membership = db.scalar(
            select(ProjectMembership).where(
                and_(
                    ProjectMembership.project_id == project_id,
                    ProjectMembership.user_id == target_user.id,
                )
            )
        )
        if membership is None:
            membership = ProjectMembership(project_id=project_id, user_id=target_user.id, role=role)
            db.add(membership)
        else:
            membership.role = role
        db.commit()
        db.refresh(membership)
        return self.serialize_membership(membership, target_user)

    def create_session(
        self,
        db: Session,
        user: User,
        *,
        project_id: str,
        name: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        access = self._project_access(db, user, project_id)
        session_row = EditorSession(
            project_id=access.project.id,
            name=name.strip() or "Session",
            description=description.strip() if description else None,
            created_by_user_id=user.id,
            updated_by_user_id=user.id,
            latest_revision=0,
        )
        db.add(session_row)
        db.commit()
        db.refresh(session_row)
        initial_graph = NeuronGraph(name=session_row.name).to_dict()
        self._live_state.ensure_session_graph(session_row.id, initial_graph, revision=0)
        self.create_snapshot(
            db,
            project=access.project,
            session_row=session_row,
            graph=initial_graph,
            revision=0,
            created_by_user_id=user.id,
            reason="initial",
        )
        return self.serialize_session(session_row)

    def ensure_default_workspace(self, db: Session, user: User) -> tuple[dict[str, Any], dict[str, Any]]:
        projects = self.list_projects(db, user)
        if projects:
            project_id = projects[0]["id"]
            sessions = self.list_sessions(db, user, project_id)
            if sessions:
                return projects[0], sessions[0]
            return projects[0], self.create_session(db, user, project_id=project_id, name="Main session")
        created = self.create_project(db, user, name="Default Project", description="Seeded workspace")
        return created["project"], created["default_session"]

    def list_sessions(self, db: Session, user: User, project_id: str) -> list[dict[str, Any]]:
        self._project_access(db, user, project_id)
        sessions = list(
            db.scalars(
                select(EditorSession)
                .where(EditorSession.project_id == project_id)
                .order_by(EditorSession.updated_at.desc())
            )
        )
        return [self.serialize_session(session_row) for session_row in sessions]

    def _load_snapshot_graph(self, snapshot: SessionSnapshot | None) -> tuple[dict[str, Any], int] | None:
        if snapshot is None:
            return None
        path = Path(snapshot.storage_path)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), snapshot.revision

    def _hydrate_live_graph(self, db: Session, session_row: EditorSession) -> SessionGraphState:
        live = self._live_state.get_session_graph(session_row.id)
        if live is not None:
            return live

        snapshot = db.scalar(
            select(SessionSnapshot)
            .where(SessionSnapshot.session_id == session_row.id)
            .order_by(desc(SessionSnapshot.revision), desc(SessionSnapshot.created_at))
        )
        loaded = self._load_snapshot_graph(snapshot)
        if loaded is not None:
            graph, revision = loaded
            return self._live_state.overwrite_session_graph(session_row.id, graph, revision)

        graph = NeuronGraph(name=session_row.name).to_dict()
        return self._live_state.ensure_session_graph(session_row.id, graph, revision=session_row.latest_revision)

    def get_session_bundle(self, db: Session, user: User, project_id: str, session_id: str) -> SessionBundle:
        access = self._project_access(db, user, project_id)
        session_row = db.scalar(
            select(EditorSession).where(
                and_(EditorSession.project_id == project_id, EditorSession.id == session_id)
            )
        )
        if session_row is None:
            raise PermissionDenied("Session not found")
        graph_state = self._hydrate_live_graph(db, session_row)
        return SessionBundle(project=access.project, session=session_row, graph_state=graph_state)

    def get_session_detail(self, db: Session, user: User, project_id: str, session_id: str) -> dict[str, Any]:
        bundle = self.get_session_bundle(db, user, project_id, session_id)
        return {
            "project": self.serialize_project(bundle.project, role=self._project_access(db, user, project_id).role),
            "session": self.serialize_session(bundle.session),
            "graph": bundle.graph_state.graph,
            "revision": bundle.graph_state.revision,
        }

    def update_session_graph(
        self,
        db: Session,
        user: User,
        *,
        project_id: str,
        session_id: str,
        graph: dict[str, Any],
        expected_revision: int | None = None,
        persist_snapshot: bool = False,
        snapshot_reason: str = "autosave",
    ) -> dict[str, Any]:
        bundle = self.get_session_bundle(db, user, project_id, session_id)
        try:
            state = self._live_state.put_session_graph(
                session_id,
                graph,
                expected_revision=expected_revision,
            )
        except RevisionConflict:
            raise
        bundle.session.latest_revision = state.revision
        bundle.session.updated_at = utcnow()
        bundle.session.updated_by_user_id = user.id
        db.commit()
        db.refresh(bundle.session)
        if persist_snapshot:
            self.create_snapshot(
                db,
                project=bundle.project,
                session_row=bundle.session,
                graph=state.graph,
                revision=state.revision,
                created_by_user_id=user.id,
                reason=snapshot_reason,
            )
        return {
            "project": self.serialize_project(bundle.project, role=self._project_access(db, user, project_id).role),
            "session": self.serialize_session(bundle.session),
            "graph": state.graph,
            "revision": state.revision,
        }

    def mutate_session_graph(
        self,
        db: Session,
        user: User,
        *,
        project_id: str,
        session_id: str,
        mutator: Callable[[NeuronGraph], None],
        snapshot_reason: str | None = None,
    ) -> dict[str, Any]:
        bundle = self.get_session_bundle(db, user, project_id, session_id)
        graph = NeuronGraph.from_dict(bundle.graph_state.graph)
        mutator(graph)
        return self.update_session_graph(
            db,
            user,
            project_id=project_id,
            session_id=session_id,
            graph=graph.to_dict(),
            expected_revision=bundle.graph_state.revision,
            persist_snapshot=snapshot_reason is not None,
            snapshot_reason=snapshot_reason or "mutation",
        )

    def create_snapshot(
        self,
        db: Session,
        *,
        project: Project,
        session_row: EditorSession,
        graph: dict[str, Any],
        revision: int,
        created_by_user_id: str | None,
        reason: str,
    ) -> dict[str, Any]:
        reason_slug = _sanitize_path_component(reason)
        directory = self._settings.snapshots_dir / project.id / session_row.id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"rev-{revision:06d}-{reason_slug}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(graph, handle, indent=2, sort_keys=True)
        snapshot = SessionSnapshot(
            project_id=project.id,
            session_id=session_row.id,
            revision=revision,
            reason=reason,
            storage_path=str(path),
            created_by_user_id=created_by_user_id,
        )
        db.add(snapshot)
        db.commit()
        db.refresh(snapshot)
        return {
            "id": snapshot.id,
            "revision": snapshot.revision,
            "reason": snapshot.reason,
            "storage_path": snapshot.storage_path,
            "created_at": self._iso(snapshot.created_at),
        }

    def analytics_summary(self, db: Session, user: User, project_id: str) -> dict[str, Any]:
        access = self._project_access(db, user, project_id)
        sessions = self.list_sessions(db, user, project_id)
        runs = list(
            db.scalars(
                select(TrainingRun)
                .where(TrainingRun.project_id == project_id)
                .order_by(TrainingRun.started_at.desc())
                .limit(20)
            )
        )
        status_counts: dict[str, int] = {}
        for run in runs:
            status_counts[run.status] = status_counts.get(run.status, 0) + 1
        return {
            "project": self.serialize_project(access.project, role=access.role),
            "session_count": len(sessions),
            "recent_run_count": len(runs),
            "status_counts": status_counts,
            "latest_runs": [
                {
                    "id": run.id,
                    "session_id": run.session_id,
                    "status": run.status,
                    "resolved_method": run.resolved_method,
                    "last_loss": run.last_loss,
                    "started_at": self._iso(run.started_at),
                    "completed_at": self._iso(run.completed_at),
                }
                for run in runs
            ],
        }

    def latest_snapshot(self, db: Session, session_id: str) -> SessionSnapshot | None:
        return db.scalar(
            select(SessionSnapshot)
            .where(SessionSnapshot.session_id == session_id)
            .order_by(desc(SessionSnapshot.revision), desc(SessionSnapshot.created_at))
        )


_workspace_service: WorkspaceService | None = None


def get_workspace_service() -> WorkspaceService:
    global _workspace_service
    if _workspace_service is None:
        _workspace_service = WorkspaceService()
    return _workspace_service
