from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import redis
from sqlalchemy import select

from ..db import session_scope
from ..db_models import EditorSession, SessionSnapshot, Project, ensure_utc, utcnow
from ..settings import get_settings

logger = logging.getLogger(__name__)

class PersistenceWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._redis = redis.Redis.from_url(self._settings.redis_url, decode_responses=True) if self._settings.redis_url else None
        if self._redis:
            logger.info(f"Persistence worker connected to Redis at {self._settings.redis_url}")
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._queue_key = "neuralfn:persistence:queue"

    def enqueue_update(
        self,
        session_id: str,
        revision: int,
        user_id: str,
        graph: dict[str, Any],
        persist_snapshot: bool = False,
        snapshot_reason: str = "autosave"
    ) -> None:
        payload = {
            "type": "update_session",
            "session_id": session_id,
            "revision": revision,
            "user_id": user_id,
            "graph": graph,
            "persist_snapshot": persist_snapshot,
            "snapshot_reason": snapshot_reason,
            "timestamp": time.time()
        }
        self._enqueue(payload)

    def enqueue_run_update(
        self,
        run_id: str,
        last_loss: float | None = None,
        last_step: int | None = None,
        status: str | None = None,
        error: str | None = None,
        completed_at: float | None = None
    ) -> None:
        payload = {
            "type": "update_run",
            "run_id": run_id,
            "last_loss": last_loss,
            "last_step": last_step,
            "status": status,
            "error": error,
            "completed_at": completed_at,
            "timestamp": time.time()
        }
        self._enqueue(payload)

    def _enqueue(self, payload: dict[str, Any]) -> None:
        if self._redis:
            try:
                self._redis.rpush(self._queue_key, json.dumps(payload))
                return
            except redis.RedisError as exc:
                logger.warning(
                    "Redis persistence enqueue failed; falling back to synchronous local persistence: %s",
                    exc,
                )
                self._redis = None
        self._process_task(payload)

    def start(self) -> None:
        if self._redis and (self._thread is None or not self._thread.is_alive()):
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            logger.info("Persistence worker started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            logger.info("Persistence worker stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                # Use blpop with timeout to allow checking stop_event
                result = self._redis.blpop(self._queue_key, timeout=1.0)
                if result:
                    _, raw_payload = result
                    payload = json.loads(raw_payload)
                    self._process_task(payload)
            except Exception as e:
                logger.error(f"Error in persistence worker: {e}")
                time.sleep(1.0)

    def _process_task(self, payload: dict[str, Any]) -> None:
        task_type = payload.get("type")
        if task_type == "update_session":
            self._handle_update_session(payload)
        elif task_type == "update_run":
            self._handle_update_run(payload)

    def _handle_update_session(self, payload: dict[str, Any]) -> None:
        session_id = payload["session_id"]
        revision = payload["revision"]
        user_id = payload["user_id"]
        graph = payload["graph"]
        persist_snapshot = payload.get("persist_snapshot", False)
        snapshot_reason = payload.get("snapshot_reason", "autosave")

        with session_scope() as db:
            session_row = db.get(EditorSession, session_id)
            if not session_row:
                logger.error(f"Session {session_id} not found for persistence")
                return

            # Only update if the revision is newer
            if session_row.latest_revision < revision:
                session_row.latest_revision = revision
                session_row.updated_at = utcnow()
                session_row.updated_by_user_id = user_id
                
            if persist_snapshot:
                project = db.get(Project, session_row.project_id)
                self._create_snapshot(db, project, session_row, graph, revision, user_id, snapshot_reason)

    def _handle_update_run(self, payload: dict[str, Any]) -> None:
        run_id = payload["run_id"]
        from ..db_models import TrainingRun
        with session_scope() as db:
            run_row = db.get(TrainingRun, run_id)
            if not run_row:
                return
            if payload.get("last_loss") is not None:
                run_row.last_loss = payload["last_loss"]
            if payload.get("last_step") is not None:
                run_row.last_step = payload["last_step"]
            if payload.get("status") is not None:
                run_row.status = payload["status"]
            if payload.get("error") is not None:
                run_row.error = payload["error"]
            if payload.get("completed_at") is not None:
                from datetime import datetime
                run_row.completed_at = datetime.fromtimestamp(payload["completed_at"])
            run_row.updated_at = utcnow()

    def _create_snapshot(
        self,
        db: Any,
        project: Project,
        session_row: EditorSession,
        graph: dict[str, Any],
        revision: int,
        user_id: str,
        reason: str
    ) -> None:
        from .session_service import _sanitize_path_component
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
            created_by_user_id=user_id,
        )
        db.add(snapshot)

_worker: PersistenceWorker | None = None

def get_persistence_worker() -> PersistenceWorker:
    global _worker
    if _worker is None:
        _worker = PersistenceWorker()
    return _worker
