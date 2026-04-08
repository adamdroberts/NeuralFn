from __future__ import annotations

from dataclasses import dataclass
import json
import threading
import time
from functools import lru_cache
from typing import Any

from ..settings import get_settings


class RevisionConflict(Exception):
    def __init__(self, current_revision: int):
        super().__init__(f"Revision conflict. Current revision is {current_revision}.")
        self.current_revision = current_revision


@dataclass
class SessionGraphState:
    graph: dict[str, Any]
    revision: int
    updated_at: float | None = None


class LiveStateStore:
    def ensure_session_graph(self, session_id: str, graph: dict[str, Any], revision: int = 0) -> SessionGraphState:
        raise NotImplementedError

    def get_session_graph(self, session_id: str) -> SessionGraphState | None:
        raise NotImplementedError

    def put_session_graph(
        self,
        session_id: str,
        graph: dict[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> SessionGraphState:
        raise NotImplementedError

    def overwrite_session_graph(self, session_id: str, graph: dict[str, Any], revision: int) -> SessionGraphState:
        raise NotImplementedError

    def touch_agent(self, session_id: str, active: bool) -> None:
        raise NotImplementedError

    def is_agent_active(self, session_id: str, ttl_seconds: float = 5.0) -> bool:
        raise NotImplementedError

    def set_active_run(self, session_id: str, run_id: str | None) -> None:
        raise NotImplementedError

    def get_active_run(self, session_id: str) -> str | None:
        raise NotImplementedError

    def initialize_run(self, run_id: str, session_id: str, status: dict[str, Any]) -> None:
        raise NotImplementedError

    def patch_run_status(self, run_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def append_run_event(self, run_id: str, event: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def get_run_snapshot(
        self,
        run_id: str,
        *,
        since_event_id: int | None = None,
        history_limit: int = 25,
    ) -> dict[str, Any] | None:
        raise NotImplementedError


class MemoryLiveStateStore(LiveStateStore):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session_graphs: dict[str, dict[str, Any]] = {}
        self._session_revisions: dict[str, int] = {}
        self._session_updated_at: dict[str, float] = {}
        self._agent_last_active: dict[str, float] = {}
        self._active_runs: dict[str, str] = {}
        self._run_status: dict[str, dict[str, Any]] = {}
        self._run_events: dict[str, list[dict[str, Any]]] = {}
        self._run_event_seq: dict[str, int] = {}

    def ensure_session_graph(self, session_id: str, graph: dict[str, Any], revision: int = 0) -> SessionGraphState:
        with self._lock:
            if session_id not in self._session_graphs:
                self._session_graphs[session_id] = json.loads(json.dumps(graph))
                self._session_revisions[session_id] = revision
                self._session_updated_at[session_id] = time.time()
            return SessionGraphState(
                graph=json.loads(json.dumps(self._session_graphs[session_id])),
                revision=self._session_revisions[session_id],
                updated_at=self._session_updated_at.get(session_id),
            )

    def get_session_graph(self, session_id: str) -> SessionGraphState | None:
        with self._lock:
            if session_id not in self._session_graphs:
                return None
            return SessionGraphState(
                graph=json.loads(json.dumps(self._session_graphs[session_id])),
                revision=self._session_revisions.get(session_id, 0),
                updated_at=self._session_updated_at.get(session_id),
            )

    def put_session_graph(
        self,
        session_id: str,
        graph: dict[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> SessionGraphState:
        with self._lock:
            current_revision = self._session_revisions.get(session_id, 0)
            if expected_revision is not None and current_revision != expected_revision:
                raise RevisionConflict(current_revision)
            next_revision = current_revision + 1
            self._session_graphs[session_id] = json.loads(json.dumps(graph))
            self._session_revisions[session_id] = next_revision
            self._session_updated_at[session_id] = time.time()
            return SessionGraphState(
                graph=json.loads(json.dumps(graph)),
                revision=next_revision,
                updated_at=self._session_updated_at[session_id],
            )

    def overwrite_session_graph(self, session_id: str, graph: dict[str, Any], revision: int) -> SessionGraphState:
        with self._lock:
            self._session_graphs[session_id] = json.loads(json.dumps(graph))
            self._session_revisions[session_id] = revision
            self._session_updated_at[session_id] = time.time()
            return SessionGraphState(
                graph=json.loads(json.dumps(graph)),
                revision=revision,
                updated_at=self._session_updated_at[session_id],
            )

    def touch_agent(self, session_id: str, active: bool) -> None:
        with self._lock:
            self._agent_last_active[session_id] = time.time() if active else 0.0

    def is_agent_active(self, session_id: str, ttl_seconds: float = 5.0) -> bool:
        with self._lock:
            return (time.time() - self._agent_last_active.get(session_id, 0.0)) < ttl_seconds

    def set_active_run(self, session_id: str, run_id: str | None) -> None:
        with self._lock:
            if run_id is None:
                self._active_runs.pop(session_id, None)
            else:
                self._active_runs[session_id] = run_id

    def get_active_run(self, session_id: str) -> str | None:
        with self._lock:
            return self._active_runs.get(session_id)

    def initialize_run(self, run_id: str, session_id: str, status: dict[str, Any]) -> None:
        with self._lock:
            self._run_status[run_id] = json.loads(json.dumps(status))
            self._run_events[run_id] = []
            self._run_event_seq[run_id] = 0
            self._active_runs[session_id] = run_id

    def patch_run_status(self, run_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            status = self._run_status.setdefault(run_id, {})
            status.update(json.loads(json.dumps(patch)))
            return json.loads(json.dumps(status))

    def append_run_event(self, run_id: str, event: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            seq = self._run_event_seq.get(run_id, 0) + 1
            self._run_event_seq[run_id] = seq
            payload = {"event_id": seq, "timestamp": time.time(), **json.loads(json.dumps(event))}
            events = self._run_events.setdefault(run_id, [])
            events.append(payload)
            status = self._run_status.setdefault(run_id, {})
            status["event_id"] = seq
            status["history_length"] = len(events)
            status["last_event"] = json.loads(json.dumps(payload))
            status["updated_at"] = payload["timestamp"]
            if payload.get("loss") is not None:
                status["last_loss"] = float(payload["loss"])
            if payload.get("local_step") is not None:
                status["last_step"] = int(payload["local_step"])
            elif payload.get("step") is not None:
                status["last_step"] = int(payload["step"])
            return json.loads(json.dumps(payload))

    def get_run_snapshot(
        self,
        run_id: str,
        *,
        since_event_id: int | None = None,
        history_limit: int = 25,
    ) -> dict[str, Any] | None:
        with self._lock:
            if run_id not in self._run_status:
                return None
            snapshot = json.loads(json.dumps(self._run_status[run_id]))
            events = list(self._run_events.get(run_id, []))
        if since_event_id is not None:
            events = [event for event in events if int(event.get("event_id", 0)) > since_event_id]
        if history_limit >= 0:
            events = events[-history_limit:] if history_limit else []
        snapshot["events"] = json.loads(json.dumps(events))
        snapshot["thread_alive"] = bool(snapshot.get("running"))
        return snapshot


class RedisLiveStateStore(LiveStateStore):
    def __init__(self, redis_url: str) -> None:
        import redis
        import logging
        logger = logging.getLogger(__name__)

        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._redis.ping()
        logger.info(f"Connected to Redis live state store at {redis_url}")

    def _session_graph_key(self, session_id: str) -> str:
        return f"neuralfn:session:{session_id}:graph"

    def _session_revision_key(self, session_id: str) -> str:
        return f"neuralfn:session:{session_id}:revision"

    def _session_updated_key(self, session_id: str) -> str:
        return f"neuralfn:session:{session_id}:updated_at"

    def _agent_key(self, session_id: str) -> str:
        return f"neuralfn:session:{session_id}:agent_last_active"

    def _active_run_key(self, session_id: str) -> str:
        return f"neuralfn:session:{session_id}:active_run"

    def _run_status_key(self, run_id: str) -> str:
        return f"neuralfn:run:{run_id}:status"

    def _run_events_key(self, run_id: str) -> str:
        return f"neuralfn:run:{run_id}:events"

    def _run_seq_key(self, run_id: str) -> str:
        return f"neuralfn:run:{run_id}:event_seq"

    def ensure_session_graph(self, session_id: str, graph: dict[str, Any], revision: int = 0) -> SessionGraphState:
        graph_key = self._session_graph_key(session_id)
        revision_key = self._session_revision_key(session_id)
        updated_key = self._session_updated_key(session_id)
        if not self._redis.exists(graph_key):
            now = time.time()
            with self._redis.pipeline() as pipe:
                pipe.set(graph_key, json.dumps(graph))
                pipe.set(revision_key, revision)
                pipe.set(updated_key, now)
                pipe.execute()
            return SessionGraphState(graph=graph, revision=revision, updated_at=now)
        return self.get_session_graph(session_id)  # type: ignore[return-value]

    def get_session_graph(self, session_id: str) -> SessionGraphState | None:
        graph_key = self._session_graph_key(session_id)
        raw_graph = self._redis.get(graph_key)
        if raw_graph is None:
            return None
        raw_revision = self._redis.get(self._session_revision_key(session_id))
        raw_updated = self._redis.get(self._session_updated_key(session_id))
        return SessionGraphState(
            graph=json.loads(raw_graph),
            revision=int(raw_revision or 0),
            updated_at=float(raw_updated) if raw_updated is not None else None,
        )

    def put_session_graph(
        self,
        session_id: str,
        graph: dict[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> SessionGraphState:
        import redis

        graph_key = self._session_graph_key(session_id)
        revision_key = self._session_revision_key(session_id)
        updated_key = self._session_updated_key(session_id)
        while True:
            try:
                with self._redis.pipeline() as pipe:
                    pipe.watch(revision_key)
                    current_revision = int(pipe.get(revision_key) or 0)
                    if expected_revision is not None and current_revision != expected_revision:
                        pipe.unwatch()
                        raise RevisionConflict(current_revision)
                    next_revision = current_revision + 1
                    now = time.time()
                    pipe.multi()
                    pipe.set(graph_key, json.dumps(graph))
                    pipe.set(revision_key, next_revision)
                    pipe.set(updated_key, now)
                    pipe.execute()
                    return SessionGraphState(graph=graph, revision=next_revision, updated_at=now)
            except redis.WatchError:
                continue

    def overwrite_session_graph(self, session_id: str, graph: dict[str, Any], revision: int) -> SessionGraphState:
        now = time.time()
        with self._redis.pipeline() as pipe:
            pipe.set(self._session_graph_key(session_id), json.dumps(graph))
            pipe.set(self._session_revision_key(session_id), revision)
            pipe.set(self._session_updated_key(session_id), now)
            pipe.execute()
        return SessionGraphState(graph=graph, revision=revision, updated_at=now)

    def touch_agent(self, session_id: str, active: bool) -> None:
        self._redis.set(self._agent_key(session_id), time.time() if active else 0.0)

    def is_agent_active(self, session_id: str, ttl_seconds: float = 5.0) -> bool:
        raw = self._redis.get(self._agent_key(session_id))
        return raw is not None and (time.time() - float(raw)) < ttl_seconds

    def set_active_run(self, session_id: str, run_id: str | None) -> None:
        key = self._active_run_key(session_id)
        if run_id is None:
            self._redis.delete(key)
        else:
            self._redis.set(key, run_id)

    def get_active_run(self, session_id: str) -> str | None:
        return self._redis.get(self._active_run_key(session_id))

    def initialize_run(self, run_id: str, session_id: str, status: dict[str, Any]) -> None:
        with self._redis.pipeline() as pipe:
            pipe.set(self._run_status_key(run_id), json.dumps(status))
            pipe.delete(self._run_events_key(run_id))
            pipe.set(self._run_seq_key(run_id), 0)
            pipe.set(self._active_run_key(session_id), run_id)
            pipe.execute()

    def patch_run_status(self, run_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        status_key = self._run_status_key(run_id)
        current = json.loads(self._redis.get(status_key) or "{}")
        current.update(json.loads(json.dumps(patch)))
        self._redis.set(status_key, json.dumps(current))
        return current

    def append_run_event(self, run_id: str, event: dict[str, Any]) -> dict[str, Any]:
        seq = int(self._redis.incr(self._run_seq_key(run_id)))
        payload = {"event_id": seq, "timestamp": time.time(), **json.loads(json.dumps(event))}
        self._redis.rpush(self._run_events_key(run_id), json.dumps(payload))
        status = json.loads(self._redis.get(self._run_status_key(run_id)) or "{}")
        status["event_id"] = seq
        events_len = self._redis.llen(self._run_events_key(run_id))
        status["history_length"] = events_len
        status["last_event"] = payload
        status["updated_at"] = payload["timestamp"]
        if payload.get("loss") is not None:
            status["last_loss"] = float(payload["loss"])
        if payload.get("local_step") is not None:
            status["last_step"] = int(payload["local_step"])
        elif payload.get("step") is not None:
            status["last_step"] = int(payload["step"])
        self._redis.set(self._run_status_key(run_id), json.dumps(status))
        return payload

    def get_run_snapshot(
        self,
        run_id: str,
        *,
        since_event_id: int | None = None,
        history_limit: int = 25,
    ) -> dict[str, Any] | None:
        raw_status = self._redis.get(self._run_status_key(run_id))
        if raw_status is None:
            return None
        snapshot = json.loads(raw_status)
        raw_events = self._redis.lrange(self._run_events_key(run_id), 0, -1)
        events = [json.loads(raw) for raw in raw_events]
        if since_event_id is not None:
            events = [event for event in events if int(event.get("event_id", 0)) > since_event_id]
        if history_limit >= 0:
            events = events[-history_limit:] if history_limit else []
        snapshot["events"] = events
        snapshot["thread_alive"] = bool(snapshot.get("running"))
        return snapshot


@lru_cache(maxsize=1)
def get_live_state_store() -> LiveStateStore:
    settings = get_settings()
    if settings.redis_url:
        try:
            return RedisLiveStateStore(settings.redis_url)
        except Exception:
            pass
    return MemoryLiveStateStore()
