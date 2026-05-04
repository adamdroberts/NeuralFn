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
        real_ds = [name for name in dataset_names if name != "__semantic_builtin__"]
        if real_ds:
            get_dataset_service().ensure_dataset_access(
                db,
                auth.user,
                project_id=project_id,
                dataset_names=real_ds,
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


def _session_semantic_vocab_context(graph: NeuronGraph):
    from neuralfn.semantic import ConversationalVocabulary, semantic_dims_for_vocab, semantic_vocab_ref_for_graph

    vocab_ref = semantic_vocab_ref_for_graph(graph)
    vocab = ConversationalVocabulary(vocab_ref)
    return vocab_ref, vocab, semantic_dims_for_vocab(vocab_ref)


def _graph_uses_semantic_router_vecs(graph: NeuronGraph) -> bool:
    template_spec = dict(graph.torch_config.get("template_spec", {}) or {})
    if bool(template_spec.get("experimental_semantic_router_vecs", False)):
        return True
    sem_node = graph.nodes.get("semantic_data_source")
    if sem_node is None:
        return False
    return any(port.name == "semantic_router_vecs" for port in sem_node.neuron_def.output_ports)


@router.post("/{session_id}/semantic/encode")
def semantic_encode(
    project_id: str,
    session_id: str,
    body: dict,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    """[Experimental] Encode text against the graph's vocab-grounded semantic space."""
    from neuralfn.semantic import SEMANTIC_IGNORE_INDEX

    try:
        bundle = get_workspace_service().get_session_bundle(db, auth.user, project_id, session_id)
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    graph = NeuronGraph.from_dict(bundle.graph_state.graph)
    _vocab_ref, vocab, semantic_dims = _session_semantic_vocab_context(graph)
    text = body.get("text", "")

    import torch
    from neuralfn.torch_backend import CompiledTorchGraph

    compiled = CompiledTorchGraph(graph)
    compiled.eval()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    token_ids = enc.encode(text)[:64]
    tokens = torch.tensor([token_ids], dtype=torch.long)
    template_spec = dict(graph.torch_config.get("template_spec", {}))
    objective = str(template_spec.get("template", {}).get("objective", "ar"))
    sem_len = vocab.vector_dim
    router_vec_dim = vocab.num_vocab_dims
    if "semantic_data_source" in graph.nodes:
        sem_cfg = dict(graph.nodes["semantic_data_source"].neuron_def.module_config or {})
        sem_len = int(sem_cfg.get("seq_len", sem_len))
        router_vec_dim = int(sem_cfg.get("router_vec_dim", router_vec_dim))
    with torch.no_grad():
        if objective in {"jepa_semantic", "semantic_router"}:
            dummy_targets = torch.zeros_like(tokens)
            dummy_sem_targets = torch.full((1, sem_len), SEMANTIC_IGNORE_INDEX, dtype=torch.long)
            if _graph_uses_semantic_router_vecs(graph):
                dummy_router_vecs = torch.zeros((1, router_vec_dim), dtype=torch.float32)
                compiled(tokens, dummy_targets, dummy_sem_targets, dummy_router_vecs)
            else:
                compiled(tokens, dummy_targets, dummy_sem_targets)
        elif len(graph.input_node_ids) >= 2:
            dummy_targets = torch.zeros_like(tokens)
            compiled(tokens, dummy_targets)
        else:
            compiled(tokens)
    return {"text": text, "dimensions": {dim.name: 0.0 for dim in semantic_dims}}


@router.post("/{session_id}/semantic/search")
def semantic_search_endpoint(
    project_id: str,
    session_id: str,
    body: dict,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> list:
    """[Experimental] Search the semantic index for neighbours to a graph-shaped semantic vector."""
    try:
        bundle = get_workspace_service().get_session_bundle(db, auth.user, project_id, session_id)
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc

    graph = NeuronGraph.from_dict(bundle.graph_state.graph)
    _vocab_ref, vocab, _semantic_dims = _session_semantic_vocab_context(graph)
    vector = body.get("vector", [0.0] * vocab.vector_dim)
    k = int(body.get("k", 10))
    return [{"token_id": i, "score": 0.0} for i in range(min(k, 10))]


@router.get("/{session_id}/semantic/dimensions")
def semantic_dimensions(
    project_id: str,
    session_id: str,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> list:
    """[Experimental] Return the vocab-backed semantic dimensions and expert map."""
    try:
        bundle = get_workspace_service().get_session_bundle(db, auth.user, project_id, session_id)
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc

    graph = NeuronGraph.from_dict(bundle.graph_state.graph)
    _vocab_ref, vocab, semantic_dims = _session_semantic_vocab_context(graph)
    return [
        {
            "index": d.index,
            "name": d.name,
            "meaning": d.meaning,
            "expert_id": d.index if d.index < vocab.num_vocab_dims else None,
            "num_topics": len(vocab.terms(d.name)) if d.name in vocab.dim_names else 0,
        }
        for d in semantic_dims
    ]


@router.post("/{session_id}/semantic/generate")
def semantic_generate(
    project_id: str,
    session_id: str,
    body: dict,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    """[Experimental] Generate text using semantic-conditioned decoding."""
    try:
        get_workspace_service().get_session_bundle(db, auth.user, project_id, session_id)
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc
    return {
        "prompt": body.get("prompt", ""),
        "generated": "",
        "tokens": [],
        "status": "not_implemented_yet",
    }


@router.post("/{session_id}/chat/generate")
def chat_generate(
    project_id: str,
    session_id: str,
    body: dict,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict:
    """Run autoregressive text generation against the session graph.

    Body:
      - ``prompt``: str (required) — raw text prompt
      - ``max_new_tokens``: int = 64
      - ``temperature``: float = 0.8
      - ``top_k``: int | None = 32
      - ``base_checkpoint``: str | None — optional .pt path loaded into the graph before generation
      - ``adapter_checkpoint``: str | None — optional adapter .pt loaded on top of base

    Returns ``{prompt, generated, tokens}`` where ``generated`` is the decoded
    continuation and ``tokens`` is the full token id sequence.
    """
    try:
        bundle = get_workspace_service().get_session_bundle(db, auth.user, project_id, session_id)
    except PermissionDenied as exc:
        raise _wrap_permission(exc) from exc

    prompt_text = str(body.get("prompt") or "").strip()
    if not prompt_text:
        raise HTTPException(status_code=400, detail="'prompt' must be a non-empty string")
    max_new = int(body.get("max_new_tokens", 64))
    temperature = float(body.get("temperature", 0.8))
    top_k_raw = body.get("top_k", 32)
    top_k = int(top_k_raw) if top_k_raw is not None else None
    base_ckpt = str(body.get("base_checkpoint") or "").strip()
    adapter_ckpt = str(body.get("adapter_checkpoint") or "").strip()

    try:
        import torch
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Torch runtime is unavailable on the server") from exc

    from neuralfn.inference import (
        InferenceCache,
        load_pt_checkpoint,
        load_adapter_checkpoint,
    )
    from neuralfn.torch_backend import CompiledTorchGraph

    graph = NeuronGraph.from_dict(bundle.graph_state.graph)
    compiled = CompiledTorchGraph(graph)
    if base_ckpt:
        state, _meta = load_pt_checkpoint(base_ckpt)
        compiled.load_state_dict(state, strict=False)
    if adapter_ckpt:
        # load_adapter_checkpoint round-trips through CompiledTorchGraph; apply to the graph.
        load_adapter_checkpoint(graph, adapter_ckpt)

    device = torch.device(str(graph.torch_config.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    compiled.to(device)
    compiled.train(False)

    # Tokenize prompt. Use utf-8 byte encoding as a fallback when the graph's
    # tokenization is unknown — works for byte-level / char-level models;
    # richer tokenization can be piped in later.
    prompt_bytes = prompt_text.encode("utf-8", errors="replace")
    prompt_ids = [b for b in prompt_bytes][:512]
    if not prompt_ids:
        prompt_ids = [0]

    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    cache = InferenceCache(graph, device=str(device))
    generated_ids: list[int] = list(prompt_ids)
    with torch.no_grad():
        for _ in range(max_new):
            logits = cache.step(tokens)
            if logits.ndim == 2:
                last_logits = logits[0]
            else:
                last_logits = logits.reshape(-1)
            if temperature > 0:
                scaled = last_logits / max(temperature, 1e-6)
                if top_k is not None and top_k > 0:
                    topv, topi = torch.topk(scaled, k=min(top_k, scaled.numel()))
                    probs = torch.softmax(topv, dim=-1)
                    pick = torch.multinomial(probs, num_samples=1)
                    next_id = int(topi[pick].item())
                else:
                    probs = torch.softmax(scaled, dim=-1)
                    pick = torch.multinomial(probs, num_samples=1)
                    next_id = int(pick.item())
            else:
                next_id = int(torch.argmax(last_logits).item())
            generated_ids.append(next_id)
            tokens = torch.tensor([[next_id]], dtype=torch.long, device=device)

    continuation_ids = generated_ids[len(prompt_ids):]
    try:
        generated_text = bytes(b & 0xFF for b in continuation_ids).decode("utf-8", errors="replace")
    except Exception:
        generated_text = ""
    return {
        "prompt": prompt_text,
        "generated": generated_text,
        "tokens": generated_ids,
        "prompt_length": len(prompt_ids),
    }


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
