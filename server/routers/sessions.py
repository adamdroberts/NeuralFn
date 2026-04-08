from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from neuralfn.graph import NeuronGraph

from ..dependencies import get_db, require_auth
from ..models import AgentStatusModel, EdgeModel, EdgeUpdateModel, ExecuteRequest, GPTTemplateRequest, LoadDatasetRequest, NeuronDefModel, NodeModel, SessionCreateRequest, SessionGraphUpdateRequest
from ..services.auth_service import AuthContext, PermissionDenied
from ..services.dataset_service import get_dataset_service
from ..services.graph_ops import (
    GraphOperationError,
    add_edge_to_graph,
    add_node_to_graph,
    apply_gpt_template,
    delete_edge_from_graph,
    delete_node_from_graph,
    execute_graph,
    execute_trace,
    find_attached_dataset_config,
    load_dataset_source_into_graph,
    probe_graph_node,
    set_graph_io,
    summarize_graph_for_agent,
    trace_torch_graph,
    update_edge_in_graph,
    update_node_in_graph,
)
from ..services.live_state import RevisionConflict, get_live_state_store
from ..services.session_service import get_workspace_service

router = APIRouter(prefix="/projects/{project_id}/sessions", tags=["sessions"])


def _wrap_permission(exc: PermissionDenied) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


def _graph_mutation(
    db: Session,
    auth: AuthContext,
    *,
    project_id: str,
    session_id: str,
    mutator: Callable[[NeuronGraph], dict | None],
) -> tuple[dict | None, dict]:
    workspace = get_workspace_service()
    bundle = workspace.get_session_bundle(db, auth.user, project_id, session_id)
    graph = NeuronGraph.from_dict(bundle.graph_state.graph)
    result = mutator(graph)
    try:
        updated = workspace.update_session_graph(
            db,
            auth.user,
            project_id=project_id,
            session_id=session_id,
            graph=graph.to_dict(),
            expected_revision=bundle.graph_state.revision,
            persist_snapshot=False,
        )
    except RevisionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return result, updated


@router.get("")
def list_sessions(
    project_id: str,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> list[dict]:
    try:
        return get_workspace_service().list_sessions(db, auth.user, project_id)
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc


@router.post("")
def create_session(
    project_id: str,
    body: SessionCreateRequest,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        return get_workspace_service().create_session(
            db,
            auth.user,
            project_id=project_id,
            name=body.name,
            description=body.description,
        )
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc


@router.post("/{session_id}/activate")
def activate_session(
    project_id: str,
    session_id: str,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        bundle = get_workspace_service().get_session_bundle(db, auth.user, project_id, session_id)
        return {
            "project": get_workspace_service().serialize_project(bundle.project, role="admin" if auth.user.is_admin else "data_scientist"),
            "session": get_workspace_service().serialize_session(bundle.session),
        }
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc


@router.get("/{session_id}")
def get_session(
    project_id: str,
    session_id: str,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        return get_workspace_service().get_session_detail(db, auth.user, project_id, session_id)
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc


@router.get("/{session_id}/graph")
def get_graph(
    project_id: str,
    session_id: str,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        detail = get_workspace_service().get_session_detail(db, auth.user, project_id, session_id)
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    return {
        "graph": detail["graph"],
        "revision": detail["revision"],
    }


@router.put("/{session_id}/graph")
def put_graph(
    project_id: str,
    session_id: str,
    body: SessionGraphUpdateRequest,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    workspace = get_workspace_service()
    try:
        return workspace.update_session_graph(
            db,
            auth.user,
            project_id=project_id,
            session_id=session_id,
            graph=body.graph.model_dump(),
            expected_revision=body.expected_revision,
            persist_snapshot=body.persist_snapshot,
            snapshot_reason=body.snapshot_reason,
        )
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    except RevisionConflict as exc:
        current = workspace.get_session_detail(db, auth.user, project_id, session_id)
        raise HTTPException(
            status_code=409,
            detail={
                "message": str(exc),
                "current_revision": exc.current_revision,
                "graph": current["graph"],
            },
        ) from exc


@router.put("/{session_id}/graph/io")
def update_io(
    project_id: str,
    session_id: str,
    body: dict,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        result, updated = _graph_mutation(
            db,
            auth,
            project_id=project_id,
            session_id=session_id,
            mutator=lambda graph: set_graph_io(graph, body.get("input_ids", []), body.get("output_ids", [])),
        )
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    except RevisionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"result": result, "revision": updated["revision"]}


@router.post("/{session_id}/nodes")
def add_node(
    project_id: str,
    session_id: str,
    body: NodeModel,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        result, updated = _graph_mutation(
            db,
            auth,
            project_id=project_id,
            session_id=session_id,
            mutator=lambda graph: add_node_to_graph(graph, body),
        )
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    except GraphOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"node": result, "revision": updated["revision"]}


@router.put("/{session_id}/nodes/{node_id}")
def update_node(
    project_id: str,
    session_id: str,
    node_id: str,
    body: NeuronDefModel,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        result, updated = _graph_mutation(
            db,
            auth,
            project_id=project_id,
            session_id=session_id,
            mutator=lambda graph: update_node_in_graph(graph, node_id, body),
        )
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    except GraphOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"node": result, "revision": updated["revision"]}


@router.delete("/{session_id}/nodes/{node_id}")
def delete_node(
    project_id: str,
    session_id: str,
    node_id: str,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        _result, updated = _graph_mutation(
            db,
            auth,
            project_id=project_id,
            session_id=session_id,
            mutator=lambda graph: delete_node_from_graph(graph, node_id),
        )
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    except GraphOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "deleted", "revision": updated["revision"]}


@router.post("/{session_id}/edges")
def add_edge(
    project_id: str,
    session_id: str,
    body: EdgeModel,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        result, updated = _graph_mutation(
            db,
            auth,
            project_id=project_id,
            session_id=session_id,
            mutator=lambda graph: add_edge_to_graph(graph, body),
        )
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    except GraphOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"edge": result, "revision": updated["revision"]}


@router.put("/{session_id}/edges/{edge_id}")
def update_edge(
    project_id: str,
    session_id: str,
    edge_id: str,
    body: EdgeUpdateModel,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        result, updated = _graph_mutation(
            db,
            auth,
            project_id=project_id,
            session_id=session_id,
            mutator=lambda graph: update_edge_in_graph(graph, edge_id, body),
        )
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    except GraphOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"edge": result, "revision": updated["revision"]}


@router.delete("/{session_id}/edges/{edge_id}")
def delete_edge(
    project_id: str,
    session_id: str,
    edge_id: str,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        _result, updated = _graph_mutation(
            db,
            auth,
            project_id=project_id,
            session_id=session_id,
            mutator=lambda graph: delete_edge_from_graph(graph, edge_id),
        )
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    except GraphOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "deleted", "revision": updated["revision"]}


@router.post("/{session_id}/execute")
def run_execute(
    project_id: str,
    session_id: str,
    body: ExecuteRequest,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        bundle = get_workspace_service().get_session_bundle(db, auth.user, project_id, session_id)
        return execute_graph(NeuronGraph.from_dict(bundle.graph_state.graph), body)
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    except GraphOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{session_id}/execute-trace")
def run_execute_trace(
    project_id: str,
    session_id: str,
    body: ExecuteRequest,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        bundle = get_workspace_service().get_session_bundle(db, auth.user, project_id, session_id)
        return execute_trace(NeuronGraph.from_dict(bundle.graph_state.graph), body)
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    except GraphOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{session_id}/trace/torch")
def run_trace_torch(
    project_id: str,
    session_id: str,
    body: ExecuteRequest,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        bundle = get_workspace_service().get_session_bundle(db, auth.user, project_id, session_id)
        graph = NeuronGraph.from_dict(bundle.graph_state.graph)
        dataset_names = list(body.dataset_names or (find_attached_dataset_config(graph) or {}).get("dataset_names") or [])
        if dataset_names:
            get_dataset_service().ensure_dataset_access(
                db,
                auth.user,
                project_id=project_id,
                dataset_names=dataset_names,
            )
        return trace_torch_graph(graph, body)
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    except GraphOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{session_id}/probe/{node_id}")
def probe_node(
    project_id: str,
    session_id: str,
    node_id: str,
    n_samples: int = 1000,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        bundle = get_workspace_service().get_session_bundle(db, auth.user, project_id, session_id)
        return probe_graph_node(NeuronGraph.from_dict(bundle.graph_state.graph), node_id, n_samples=n_samples)
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    except GraphOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{session_id}/templates/gpt/apply")
def apply_gpt(
    project_id: str,
    session_id: str,
    body: GPTTemplateRequest,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    workspace = get_workspace_service()
    try:
        graph = apply_gpt_template(body)
        updated = workspace.update_session_graph(
            db,
            auth.user,
            project_id=project_id,
            session_id=session_id,
            graph=graph.to_dict(),
            expected_revision=None,
            persist_snapshot=True,
            snapshot_reason="template_apply",
        )
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    return {
        "revision": updated["revision"],
        "graph": updated["graph"],
    }


@router.post("/{session_id}/datasets/load")
def load_dataset_source(
    project_id: str,
    session_id: str,
    body: LoadDatasetRequest,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    downloaded_info: dict | None = None
    effective_body = body
    try:
        if body.dataset_names:
            get_dataset_service().ensure_dataset_access(
                db,
                auth.user,
                project_id=project_id,
                dataset_names=body.dataset_names,
            )
        if body.hf_path:
            downloaded_info = get_dataset_service().download_dataset(
                db,
                auth.user,
                project_id=project_id,
                request=body,
            )
            merged_names = list(dict.fromkeys([*(body.dataset_names or []), downloaded_info["name"]]))
            effective_body = body.model_copy(
                update={
                    "dataset_names": merged_names,
                    "hf_path": None,
                }
            )
        result, updated = _graph_mutation(
            db,
            auth,
            project_id=project_id,
            session_id=session_id,
            mutator=lambda graph: load_dataset_source_into_graph(graph, effective_body),
        )
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    except GraphOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if downloaded_info is not None:
        result["downloaded"] = downloaded_info
    return {
        "result": result,
        "revision": updated["revision"],
        "graph": updated["graph"]
    }


@router.get("/{session_id}/agent/status")
def get_agent_status(
    project_id: str,
    session_id: str,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        get_workspace_service().get_session_bundle(db, auth.user, project_id, session_id)
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    return {"active": get_live_state_store().is_agent_active(session_id)}


@router.post("/{session_id}/agent/status")
def set_agent_status(
    project_id: str,
    session_id: str,
    body: AgentStatusModel,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    try:
        get_workspace_service().get_session_bundle(db, auth.user, project_id, session_id)
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    get_live_state_store().touch_agent(session_id, body.active)
    return {"active": body.active}
