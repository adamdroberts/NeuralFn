from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter
from sqlalchemy import select
from starlette.responses import StreamingResponse

from neuralfn.graph import NeuronGraph

from .db import get_session_factory, init_db
from .models import ExecuteRequest, GPTTemplateRequest, GraphModel, LoadDatasetRequest, TrainRequest
from .routers.admin import router as admin_router
from .routers.auth import router as auth_router
from .routers.bootstrap import router as bootstrap_router
from .routers.catalog import router as catalog_router
from .routers.datasets import router as datasets_router
from .routers.projects import router as projects_router
from .routers.runs import router as runs_router
from .routers.sessions import router as sessions_router
from .services.auth_service import get_auth_service
from .services.graph_ops import build_template_payload, execute_graph, execute_trace, load_dataset_source_into_graph, trace_torch_graph
from .services.run_service import get_run_service
from .services.session_service import get_workspace_service

router = APIRouter(prefix="/api")
router.include_router(bootstrap_router)
router.include_router(auth_router)
router.include_router(admin_router)
router.include_router(projects_router)
router.include_router(catalog_router)
router.include_router(datasets_router)
router.include_router(sessions_router)
router.include_router(runs_router)


LEGACY_EMAIL = "legacy-tests@neuralfn.local"
LEGACY_PASSWORD = "legacy-tests"
LEGACY_PROJECT_NAME = "Legacy Project"
LEGACY_PROJECT_SLUG = "legacy-project"
LEGACY_SESSION_NAME = "Legacy Session"


def _legacy_scope() -> tuple[str, str, str]:
    init_db()
    auth_service = get_auth_service()
    workspace = get_workspace_service()
    session_factory = get_session_factory()
    with session_factory() as db:
        from .db_models import EditorSession, Project

        user = auth_service.get_user_by_email(db, LEGACY_EMAIL)
        if user is None:
            user = auth_service.create_user(
                db,
                email=LEGACY_EMAIL,
                password=LEGACY_PASSWORD,
                display_name="Legacy Tests",
                is_admin=True,
            )

        project_row = db.scalar(select(Project).where(Project.slug == LEGACY_PROJECT_SLUG))
        if project_row is None:
            project = workspace.create_project(
                db,
                user,
                name=LEGACY_PROJECT_NAME,
                description="Legacy test workspace",
            )["project"]
        else:
            project = workspace.serialize_project(project_row, role="admin")

        session_row = db.scalar(
            select(EditorSession).where(
                EditorSession.project_id == project["id"],
                EditorSession.name == LEGACY_SESSION_NAME,
            )
        )
        if session_row is None:
            session_row = workspace.create_session(db, user, project_id=project["id"], name=LEGACY_SESSION_NAME)
        else:
            session_row = workspace.serialize_session(session_row)
        return user.id, project["id"], session_row["id"]


def build_gpt_template(body: GPTTemplateRequest) -> dict:
    return build_template_payload(body)


def get_graph() -> dict:
    user_id, project_id, session_id = _legacy_scope()
    session_factory = get_session_factory()
    with session_factory() as db:
        from .db_models import User

        user = db.get(User, user_id)
        return get_workspace_service().get_session_detail(db, user, project_id, session_id)["graph"]  # type: ignore[arg-type]


def put_graph(body: GraphModel) -> dict:
    user_id, project_id, session_id = _legacy_scope()
    session_factory = get_session_factory()
    with session_factory() as db:
        from .db_models import User

        user = db.get(User, user_id)
        updated = get_workspace_service().update_session_graph(
            db,
            user,  # type: ignore[arg-type]
            project_id=project_id,
            session_id=session_id,
            graph=body.model_dump(),
            expected_revision=None,
            persist_snapshot=False,
        )
        return updated["graph"]


def execute(body: ExecuteRequest) -> dict:
    user_id, project_id, session_id = _legacy_scope()
    session_factory = get_session_factory()
    with session_factory() as db:
        from .db_models import User

        user = db.get(User, user_id)
        bundle = get_workspace_service().get_session_bundle(db, user, project_id, session_id)  # type: ignore[arg-type]
        return execute_graph(NeuronGraph.from_dict(bundle.graph_state.graph), body)


def execute_trace_legacy(body: ExecuteRequest) -> dict:
    user_id, project_id, session_id = _legacy_scope()
    session_factory = get_session_factory()
    with session_factory() as db:
        from .db_models import User

        user = db.get(User, user_id)
        bundle = get_workspace_service().get_session_bundle(db, user, project_id, session_id)  # type: ignore[arg-type]
        return execute_trace(NeuronGraph.from_dict(bundle.graph_state.graph), body)


def execute_trace(body: ExecuteRequest) -> dict:
    return execute_trace_legacy(body)


def torch_trace(body: ExecuteRequest) -> dict:
    user_id, project_id, session_id = _legacy_scope()
    session_factory = get_session_factory()
    with session_factory() as db:
        from .db_models import User

        user = db.get(User, user_id)
        bundle = get_workspace_service().get_session_bundle(db, user, project_id, session_id)  # type: ignore[arg-type]
        return trace_torch_graph(NeuronGraph.from_dict(bundle.graph_state.graph), body)


def load_dataset(body: LoadDatasetRequest) -> dict:
    user_id, project_id, session_id = _legacy_scope()
    session_factory = get_session_factory()
    with session_factory() as db:
        from .db_models import User

        user = db.get(User, user_id)
        bundle = get_workspace_service().get_session_bundle(db, user, project_id, session_id)  # type: ignore[arg-type]
        graph = NeuronGraph.from_dict(bundle.graph_state.graph)
        result = load_dataset_source_into_graph(graph, body)
        get_workspace_service().update_session_graph(
            db,
            user,  # type: ignore[arg-type]
            project_id=project_id,
            session_id=session_id,
            graph=graph.to_dict(),
            expected_revision=bundle.graph_state.revision,
            persist_snapshot=False,
        )
        return result


def train_start(body: TrainRequest) -> StreamingResponse:
    user_id, project_id, session_id = _legacy_scope()
    session_factory = get_session_factory()
    with session_factory() as db:
        from .db_models import User

        user = db.get(User, user_id)
        handle = get_run_service().start_run(
            db,
            user,  # type: ignore[arg-type]
            project_id=project_id,
            session_id=session_id,
            body=body,
        )

    async def event_stream():
        idx = 0
        while not handle.done_event.is_set():
            await asyncio.sleep(0.1)
            while idx < len(handle.progress_queue):
                msg = handle.progress_queue[idx]
                yield f"data: {json.dumps(msg)}\n\n"
                idx += 1
        while idx < len(handle.progress_queue):
            msg = handle.progress_queue[idx]
            yield f"data: {json.dumps(msg)}\n\n"
            idx += 1
        yield "data: {\"done\": true}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def get_training_status(
    since_event_id: int | None = None,
    history_limit: int = 25,
) -> dict:
    user_id, project_id, session_id = _legacy_scope()
    session_factory = get_session_factory()
    with session_factory() as db:
        from .db_models import User

        user = db.get(User, user_id)
        return get_run_service().get_run_snapshot(
            db,
            user,  # type: ignore[arg-type]
            project_id,
            session_id,
            since_event_id=since_event_id,
            history_limit=max(0, min(history_limit, 500)),
        )


def train_stop() -> dict:
    user_id, project_id, session_id = _legacy_scope()
    session_factory = get_session_factory()
    with session_factory() as db:
        from .db_models import User

        user = db.get(User, user_id)
        latest = get_run_service().get_latest_run(db, user, project_id, session_id)  # type: ignore[arg-type]
        if not latest or not latest.get("run_id"):
            return {"status": "not_running"}
        return get_run_service().stop_run(db, user, project_id, session_id, latest["run_id"])  # type: ignore[arg-type]
