from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import threading
import time
from typing import Any

import numpy as np
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from neuralfn.evolutionary import EvoConfig, EvolutionaryTrainer
from neuralfn.graph import NeuronGraph
from neuralfn.hybrid import HybridConfig, HybridTrainer
from neuralfn.trainer import SurrogateTrainer, TrainConfig
from neuralfn.torch_backend import TorchTrainConfig, TorchTrainer

from ..db import get_session_factory
from ..db_models import EditorSession, TrainingRun, User, ensure_utc, utcnow
from ..models import TrainRequest
from .dataset_service import get_dataset_service
from .graph_ops import find_attached_dataset_config
from .live_state import get_live_state_store
from .session_service import get_workspace_service


@dataclass
class ActiveRunHandle:
    run_id: str
    session_id: str
    project_id: str
    started_by_user_id: str | None
    progress_queue: list[dict[str, Any]] = field(default_factory=list)
    done_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    trainer: SurrogateTrainer | EvolutionaryTrainer | HybridTrainer | TorchTrainer | None = None


class RunService:
    def __init__(self) -> None:
        self._live_state = get_live_state_store()
        self._workspace = get_workspace_service()
        self._datasets = get_dataset_service()
        self._session_factory = get_session_factory()
        self._lock = threading.Lock()
        self._active_runs: dict[str, ActiveRunHandle] = {}

    def _new_run_status(
        self,
        *,
        run_id: str,
        resolved_method: str,
        graph: NeuronGraph,
        dataset_names: list[str],
        seq_len: int,
        requested_method: str | None,
    ) -> dict[str, Any]:
        now = time.time()
        return {
            "run_id": run_id,
            "status": "running",
            "running": True,
            "done": False,
            "method": resolved_method,
            "requested_method": requested_method,
            "graph_name": graph.name,
            "graph_training_method": graph.training_method,
            "runtime": graph.runtime,
            "dataset_names": list(dataset_names),
            "seq_len": seq_len,
            "event_id": 0,
            "history_length": 0,
            "last_event": None,
            "last_loss": None,
            "last_step": None,
            "started_at": now,
            "updated_at": now,
            "completed_at": None,
            "stop_requested": False,
            "error": None,
        }

    def _iso(self, value: datetime | None) -> str | None:
        normalized = ensure_utc(value)
        return normalized.isoformat() if normalized is not None else None

    def _timestamp(self, value: datetime | None) -> float | None:
        normalized = ensure_utc(value)
        return normalized.timestamp() if normalized is not None else None

    def _update_run_row(
        self,
        db: Session,
        run_id: str,
        *,
        status: str | None = None,
        last_loss: float | None = None,
        last_step: int | None = None,
        error: str | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        run_row = db.get(TrainingRun, run_id)
        if run_row is None:
            return
        if status is not None:
            run_row.status = status
        if last_loss is not None:
            run_row.last_loss = last_loss
        if last_step is not None:
            run_row.last_step = last_step
        if error is not None:
            run_row.error = error
        if completed_at is not None:
            run_row.completed_at = completed_at
        run_row.updated_at = utcnow()
        db.commit()

    def _resolve_dataset_inputs(
        self,
        db: Session,
        user: User,
        *,
        project_id: str,
        graph: NeuronGraph,
        body: TrainRequest,
    ) -> tuple[list[str], int]:
        dataset_names = [name for name in list(body.dataset_names or []) if name]
        if not dataset_names:
            dataset_cfg = find_attached_dataset_config(graph) or {}
            dataset_names = [name for name in list(dataset_cfg.get("dataset_names") or []) if name]
            seq_len = int(body.seq_len or dataset_cfg.get("seq_len") or 64)
        else:
            seq_len = int(body.seq_len or 64)
        if dataset_names:
            self._datasets.ensure_dataset_access(
                db,
                user,
                project_id=project_id,
                dataset_names=dataset_names,
            )
        return dataset_names, seq_len

    def _load_training_arrays(
        self,
        db: Session,
        user: User,
        *,
        project_id: str,
        graph: NeuronGraph,
        body: TrainRequest,
    ) -> tuple[np.ndarray | list, np.ndarray | list, list[str], int]:
        dataset_names, seq_len = self._resolve_dataset_inputs(
            db,
            user,
            project_id=project_id,
            graph=graph,
            body=body,
        )
        template_spec = dict(graph.torch_config.get("template_spec", {}))
        tokenization = str(template_spec.get("template", {}).get("tokenization", "sp"))
        if dataset_names:
            if tokenization == "byte_hnet":
                inputs, targets = self._datasets.load_dataset_bytes_for_project(
                    db,
                    user,
                    project_id=project_id,
                    dataset_names=dataset_names,
                    seq_len=seq_len,
                )
            else:
                inputs, targets = self._datasets.load_dataset_tokens_for_project(
                    db,
                    user,
                    project_id=project_id,
                    dataset_names=dataset_names,
                    seq_len=seq_len,
                )
            return inputs, targets, dataset_names, seq_len
        return (
            body.train_inputs,
            body.train_targets,
            [],
            int(body.seq_len or 64),
        )

    def _finish_run(
        self,
        handle: ActiveRunHandle,
        *,
        graph: NeuronGraph,
        status: str,
        error: str | None = None,
    ) -> None:
        now = utcnow()
        self._live_state.patch_run_status(
            handle.run_id,
            {
                "status": status,
                "running": False,
                "done": True,
                "completed_at": time.time(),
                "updated_at": time.time(),
                "error": error,
            },
        )
        self._live_state.set_active_run(handle.session_id, None)
        with self._lock:
            self._active_runs.pop(handle.run_id, None)

        # Background the SQLite update
        from .persistence_worker import get_persistence_worker
        snapshot = self._live_state.get_run_snapshot(handle.run_id, history_limit=0)
        get_persistence_worker().enqueue_run_update(
            run_id=handle.run_id,
            status=status,
            last_loss=snapshot.get("last_loss") if snapshot else None,
            last_step=snapshot.get("last_step") if snapshot else None,
            error=error,
            completed_at=time.time()
        )

        with self._session_factory() as db:
            user = db.get(User, handle.started_by_user_id) if handle.started_by_user_id else None
            session_row = db.get(EditorSession, handle.session_id)
            if user is not None and session_row is not None and status in {"completed", "stopped"}:
                self._workspace.update_session_graph(
                    db,
                    user,
                    project_id=handle.project_id,
                    session_id=handle.session_id,
                    graph=graph.to_dict(),
                    expected_revision=None,
                    persist_snapshot=True,
                    snapshot_reason=f"run_{status}",
                )

    def list_runs(self, db: Session, user: User, project_id: str, session_id: str) -> list[dict[str, Any]]:
        self._workspace.get_session_bundle(db, user, project_id, session_id)
        rows = list(
            db.scalars(
                select(TrainingRun)
                .where(TrainingRun.project_id == project_id, TrainingRun.session_id == session_id)
                .order_by(desc(TrainingRun.started_at))
                .limit(100)
            )
        )
        return [
            {
                "id": row.id,
                "status": row.status,
                "requested_method": row.requested_method,
                "resolved_method": row.resolved_method,
                "graph_name": row.graph_name,
                "dataset_names": row.dataset_names,
                "seq_len": row.seq_len,
                "last_loss": row.last_loss,
                "last_step": row.last_step,
                "error": row.error,
                "started_at": self._iso(row.started_at),
                "completed_at": self._iso(row.completed_at),
            }
            for row in rows
        ]

    def get_latest_run(self, db: Session, user: User, project_id: str, session_id: str) -> dict[str, Any] | None:
        self._workspace.get_session_bundle(db, user, project_id, session_id)
        active_run_id = self._live_state.get_active_run(session_id)
        if active_run_id:
            snapshot = self._live_state.get_run_snapshot(active_run_id)
            if snapshot:
                with self._lock:
                    handle = self._active_runs.get(active_run_id)
                snapshot["thread_alive"] = bool(handle and handle.thread and handle.thread.is_alive())
                return snapshot

        row = db.scalar(
            select(TrainingRun)
            .where(TrainingRun.project_id == project_id, TrainingRun.session_id == session_id)
            .order_by(desc(TrainingRun.started_at))
        )
        if row is None:
            return None
        return {
            "run_id": row.id,
            "status": row.status,
            "running": row.status == "running",
            "done": row.status in {"completed", "stopped", "error"},
            "requested_method": row.requested_method,
            "method": row.resolved_method,
            "graph_name": row.graph_name,
            "dataset_names": row.dataset_names,
            "seq_len": row.seq_len,
            "last_loss": row.last_loss,
            "last_step": row.last_step,
            "error": row.error,
            "started_at": self._timestamp(row.started_at),
            "completed_at": self._timestamp(row.completed_at),
            "updated_at": self._timestamp(row.updated_at),
            "events": [],
            "thread_alive": False,
            "history_length": 0,
            "event_id": 0,
            "stop_requested": False,
        }

    def get_run_snapshot(
        self,
        db: Session,
        user: User,
        project_id: str,
        session_id: str,
        *,
        since_event_id: int | None = None,
        history_limit: int = 25,
    ) -> dict[str, Any]:
        self._workspace.get_session_bundle(db, user, project_id, session_id)
        active_run_id = self._live_state.get_active_run(session_id)
        if active_run_id:
            snapshot = self._live_state.get_run_snapshot(
                active_run_id,
                since_event_id=since_event_id,
                history_limit=history_limit,
            )
            if snapshot is not None:
                with self._lock:
                    handle = self._active_runs.get(active_run_id)
                snapshot["thread_alive"] = bool(handle and handle.thread and handle.thread.is_alive())
                return snapshot
        return self.get_latest_run(db, user, project_id, session_id) or {
            "status": "idle",
            "running": False,
            "done": False,
            "events": [],
            "history_length": 0,
            "event_id": 0,
        }

    def stop_run(self, db: Session, user: User, project_id: str, session_id: str, run_id: str) -> dict[str, Any]:
        self._workspace.get_session_bundle(db, user, project_id, session_id)
        with self._lock:
            handle = self._active_runs.get(run_id)
        if handle is None:
            return {"status": "not_running"}
        self._live_state.patch_run_status(
            run_id,
            {"stop_requested": True, "updated_at": time.time()},
        )
        if handle.trainer is not None:
            handle.trainer.stop()
        return {"status": "stopping"}

    def start_run(
        self,
        db: Session,
        user: User,
        *,
        project_id: str,
        session_id: str,
        body: TrainRequest,
    ) -> ActiveRunHandle:
        bundle = self._workspace.get_session_bundle(db, user, project_id, session_id)
        current_active = self._live_state.get_active_run(session_id)
        if current_active:
            snapshot = self._live_state.get_run_snapshot(current_active, history_limit=0)
            if snapshot and snapshot.get("running"):
                raise ValueError("A training run is already active for this session")

        graph = NeuronGraph.from_dict(bundle.graph_state.graph)
        
        use_torch = (
            body.method == "torch"
            or graph.training_method == "torch"
            or graph.runtime == "torch"
            or graph.has_module_nodes()
        )
        
        # Optimization: Don't load full dataset into memory if we are using TorchTrainer
        # and there is an attached dataset_source node.
        attached_ds_cfg = find_attached_dataset_config(graph)
        if use_torch and attached_ds_cfg and not body.dataset_names:
            train_in, train_tgt = np.array([]), np.array([])
            dataset_names = attached_ds_cfg.get("dataset_names", [])
            seq_len = int(body.seq_len or attached_ds_cfg.get("seq_len") or 64)
            # Just ensure access
            self._datasets.ensure_dataset_access(db, user, project_id=project_id, dataset_names=dataset_names)
        else:
            train_in, train_tgt, dataset_names, seq_len = self._load_training_arrays(
                db,
                user,
                project_id=project_id,
                graph=graph,
                body=body,
            )

        use_legacy = body.method in {"surrogate", "evolutionary"} and not graph.has_nested_subgraphs() and not use_torch
        resolved_method = "torch" if use_torch else (body.method if use_legacy else "hybrid")
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Starting training run: requested_method={body.method}, resolved_method={resolved_method}, torch={use_torch}")

        run_row = TrainingRun(
            project_id=project_id,
            session_id=session_id,
            started_by_user_id=user.id,
            status="running",
            requested_method=body.method,
            resolved_method=resolved_method,
            graph_name=graph.name,
            dataset_names=dataset_names,
            seq_len=seq_len,
        )
        db.add(run_row)
        db.commit()
        db.refresh(run_row)

        handle = ActiveRunHandle(
            run_id=run_row.id,
            session_id=session_id,
            project_id=project_id,
            started_by_user_id=user.id,
        )
        status = self._new_run_status(
            run_id=run_row.id,
            resolved_method=resolved_method,
            graph=graph,
            dataset_names=dataset_names,
            seq_len=seq_len,
            requested_method=body.method,
        )
        self._live_state.initialize_run(run_row.id, session_id, status)
        with self._lock:
            self._active_runs[run_row.id] = handle

        def record_event(payload: dict[str, Any]) -> dict[str, Any]:
            event = self._live_state.append_run_event(run_row.id, payload)
            handle.progress_queue.append(event)
            # Background the SQLite update
            from .persistence_worker import get_persistence_worker
            get_persistence_worker().enqueue_run_update(
                run_id=run_row.id,
                last_loss=event.get("loss"),
                last_step=event.get("local_step", event.get("step")),
            )
            return event

        # Emit initial event
        record_event({"status": "starting", "message": f"Training session started using {resolved_method} method"})

        def on_progress(step: int, loss: float) -> None:
            record_event({"step": step, "loss": loss})

        def on_hybrid_progress(info: dict[str, Any]) -> None:
            record_event(info)

        def run_surrogate() -> None:
            cfg = TrainConfig(
                learning_rate=body.learning_rate,
                epochs=body.epochs,
                loss_fn=body.loss_fn,
            )
            trainer = SurrogateTrainer(graph, cfg)
            handle.trainer = trainer
            trainer.train(np.asarray(train_in, dtype=np.float32), np.asarray(train_tgt, dtype=np.float32), on_epoch=on_progress)

        def run_evolutionary() -> None:
            cfg = EvoConfig(
                population_size=body.population_size,
                generations=body.generations,
            )
            trainer = EvolutionaryTrainer(graph, cfg)
            handle.trainer = trainer
            trainer.train(np.asarray(train_in, dtype=np.float32), np.asarray(train_tgt, dtype=np.float32), on_generation=on_progress)

        def run_hybrid() -> None:
            cfg = HybridConfig(
                outer_rounds=body.outer_rounds,
                loss_fn=body.loss_fn,
                default_surrogate=TrainConfig(
                    learning_rate=body.learning_rate,
                    epochs=body.epochs,
                    loss_fn=body.loss_fn,
                ),
                default_evolutionary=EvoConfig(
                    population_size=body.population_size,
                    generations=body.generations,
                ),
            )
            trainer = HybridTrainer(graph, cfg)
            handle.trainer = trainer
            trainer.train(np.asarray(train_in, dtype=np.float32), np.asarray(train_tgt, dtype=np.float32), on_step=on_hybrid_progress)

        def run_torch() -> None:
            cfg = TorchTrainConfig(
                learning_rate=body.learning_rate,
                epochs=body.epochs,
                batch_size=body.batch_size,
                weight_decay=body.weight_decay,
                device=str(graph.torch_config.get("device", "cuda")),
                amp_dtype=str(graph.torch_config.get("amp_dtype", "bfloat16")),
            )
            trainer = TorchTrainer(graph, cfg)
            handle.trainer = trainer
            trainer.train(train_in, train_tgt, on_epoch=on_progress)

        target = (
            run_torch
            if use_torch
            else (run_surrogate if use_legacy and body.method == "surrogate" else run_evolutionary if use_legacy else run_hybrid)
        )

        def runner() -> None:
            try:
                target()
            except Exception as exc:
                record_event({"error": str(exc)})
                self._finish_run(handle, graph=graph, status="error", error=str(exc))
            else:
                status_value = self._live_state.get_run_snapshot(run_row.id, history_limit=0)
                final_status = "stopped" if status_value and status_value.get("stop_requested") else "completed"
                self._finish_run(handle, graph=graph, status=final_status)
            finally:
                handle.trainer = None
                handle.done_event.set()

        handle.thread = threading.Thread(target=runner, daemon=True)
        handle.thread.start()
        return handle


_run_service: RunService | None = None


def get_run_service() -> RunService:
    global _run_service
    if _run_service is None:
        _run_service = RunService()
    return _run_service
