from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from ..dependencies import get_db, require_auth
from ..models import TrainRequest
from ..services.auth_service import AuthContext, PermissionDenied
from ..services.run_service import get_run_service

router = APIRouter(prefix="/projects/{project_id}/sessions/{session_id}/runs", tags=["runs"])


def _wrap_permission(exc: PermissionDenied) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


@router.get("")
def list_runs(
    project_id: str,
    session_id: str,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> list[dict]:
    try:
        return get_run_service().list_runs(db, auth.user, project_id, session_id)
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc


@router.get("/active")
def get_active_run(
    project_id: str,
    session_id: str,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        return get_run_service().get_latest_run(db, auth.user, project_id, session_id) or {
            "status": "idle",
            "running": False,
            "done": False,
            "events": [],
        }
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc


@router.post("")
def start_run(
    project_id: str,
    session_id: str,
    body: TrainRequest,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    try:
        handle = get_run_service().start_run(
            db,
            auth.user,
            project_id=project_id,
            session_id=session_id,
            body=body,
        )
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error(f"Failed to start training run: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}") from exc

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


@router.post("/{run_id}/stop")
def stop_run(
    project_id: str,
    session_id: str,
    run_id: str,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        return get_run_service().stop_run(db, auth.user, project_id, session_id, run_id)
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
