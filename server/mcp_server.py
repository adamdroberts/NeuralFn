# /// script
# dependencies = ["mcp>=1.0.0"]
# ///

from __future__ import annotations

import http.cookiejar
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.server.fastmcp import FastMCP

from server.settings import get_settings

mcp = FastMCP("NeuralFn Editor")

BASE_URL = os.getenv("NEURALFN_BASE_URL", "http://localhost:8000/api")
COOKIE_JAR = http.cookiejar.CookieJar()
OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(COOKIE_JAR))


def _ensure_authenticated() -> None:
    if any(cookie.name == get_settings().session_cookie_name for cookie in COOKIE_JAR):
        return
    settings = get_settings()
    if not settings.mcp_email or not settings.mcp_password:
        raise RuntimeError(
            "Set NEURALFN_MCP_EMAIL and NEURALFN_MCP_PASSWORD so the MCP server can authenticate."
        )
    payload = {"email": settings.mcp_email, "password": settings.mcp_password}
    _request("POST", "auth/login", payload, allow_unauthenticated=True)


def _request(
    method: str,
    endpoint: str,
    data: dict | None = None,
    *,
    allow_unauthenticated: bool = False,
) -> Any:
    if not allow_unauthenticated:
        _ensure_authenticated()
    url = f"{BASE_URL}/{endpoint}"
    req = urllib.request.Request(url, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(data).encode("utf-8")
    try:
        with OPENER.open(req) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8")
        raise RuntimeError(f"API Error {exc.code}: {error_body}") from exc


def _session_prefix(project_id: str, session_id: str) -> str:
    return f"projects/{project_id}/sessions/{session_id}"


def _project_prefix(project_id: str) -> str:
    return f"projects/{project_id}"


def _summarize_graph(graph: dict) -> dict:
    nodes_summary = {}
    for nid, node in graph.get("nodes", {}).items():
        ndef = node.get("neuron_def", {})
        entry: dict[str, Any] = {"name": ndef.get("name", ""), "kind": ndef.get("kind", "function")}
        module_type = ndef.get("module_type", "")
        if module_type:
            entry["module_type"] = module_type
        nodes_summary[nid] = entry
    edges_summary = [
        {"id": eid, "src": f"{edge['src_node']}:{edge['src_port']}", "dst": f"{edge['dst_node']}:{edge['dst_port']}"}
        for eid, edge in graph.get("edges", {}).items()
    ]
    variant_families = {
        family: list(versions.keys())
        for family, versions in graph.get("variant_library", {}).items()
    }
    return {
        "name": graph.get("name", ""),
        "training_method": graph.get("training_method", ""),
        "runtime": graph.get("runtime", ""),
        "nodes": nodes_summary,
        "edges": edges_summary,
        "input_node_ids": graph.get("input_node_ids", []),
        "output_node_ids": graph.get("output_node_ids", []),
        "variant_families": variant_families,
    }


def _summarize_node(node: dict) -> dict:
    ndef = node.get("neuron_def", {})
    result: dict[str, Any] = {
        "instance_id": node.get("instance_id", ""),
        "position": node.get("position"),
        "name": ndef.get("name", ""),
        "kind": ndef.get("kind", "function"),
        "input_ports": [p.get("name", "") for p in ndef.get("input_ports", [])],
        "output_ports": [p.get("name", "") for p in ndef.get("output_ports", [])],
    }
    source = ndef.get("source_code", "")
    if source:
        result["source_code"] = source[:200] + ("..." if len(source) > 200 else "")
    if ndef.get("module_type"):
        result["module_type"] = ndef["module_type"]
    if ndef.get("module_config"):
        result["module_config"] = ndef["module_config"]
    if ndef.get("variant_ref"):
        result["variant_ref"] = ndef["variant_ref"]
    if ndef.get("input_aliases"):
        result["input_aliases"] = ndef["input_aliases"]
    if ndef.get("output_aliases"):
        result["output_aliases"] = ndef["output_aliases"]
    subgraph = ndef.get("subgraph")
    if isinstance(subgraph, dict):
        result["subgraph"] = f"(nested graph: {len(subgraph.get('nodes', {}))} nodes)"
    return result


def _brief_node(node: dict) -> dict:
    ndef = node.get("neuron_def", {})
    return {
        "instance_id": node.get("instance_id", ""),
        "name": ndef.get("name", ""),
        "kind": ndef.get("kind", "function"),
        "position": node.get("position"),
    }


class AgentSession:
    def __init__(self, project_id: str, session_id: str) -> None:
        self._project_id = project_id
        self._session_id = session_id

    def __enter__(self):
        try:
            _request(
                "POST",
                f"{_session_prefix(self._project_id, self._session_id)}/agent/status",
                {"active": True},
            )
        except Exception as exc:
            print(f"Warning: failed to set agent status: {exc}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _fetch_training_status(project_id: str, session_id: str) -> dict[str, Any]:
    return _request("GET", f"{_session_prefix(project_id, session_id)}/runs/active")


@mcp.tool()
def get_graph(project_id: str, session_id: str) -> dict:
    """Get a summary of the scoped graph for one project/session."""
    payload = _request("GET", f"{_session_prefix(project_id, session_id)}/graph")
    return _summarize_graph(payload["graph"])


@mcp.tool()
def replace_graph(project_id: str, session_id: str, graph: dict) -> dict:
    """Replace the entire graph for a specific project/session."""
    with AgentSession(project_id, session_id):
        result = _request(
            "PUT",
            f"{_session_prefix(project_id, session_id)}/graph",
            {"graph": graph, "persist_snapshot": False},
        )
        return _summarize_graph(result["graph"])


@mcp.tool()
def update_graph_settings(
    project_id: str,
    session_id: str,
    name: str | None = None,
    training_method: str | None = None,
    runtime: str | None = None,
    surrogate_config: dict | None = None,
    evo_config: dict | None = None,
    torch_config: dict | None = None,
) -> dict:
    """Update graph-level settings for one project/session."""
    with AgentSession(project_id, session_id):
        current = _request("GET", f"{_session_prefix(project_id, session_id)}/graph")
        graph = current["graph"]
        if name is not None:
            graph["name"] = name
        if training_method is not None:
            graph["training_method"] = training_method
            if training_method == "torch" and runtime is None:
                graph["runtime"] = "torch"
        if runtime is not None:
            graph["runtime"] = runtime
        if surrogate_config is not None:
            graph["surrogate_config"] = surrogate_config
        if evo_config is not None:
            graph["evo_config"] = evo_config
        if torch_config is not None:
            graph["torch_config"] = torch_config
        result = _request(
            "PUT",
            f"{_session_prefix(project_id, session_id)}/graph",
            {
                "graph": graph,
                "expected_revision": current["revision"],
                "persist_snapshot": False,
            },
        )
        return {
            "status": "updated",
            "revision": result["revision"],
            "name": graph["name"],
            "training_method": graph["training_method"],
            "runtime": graph["runtime"],
        }


@mcp.tool()
def set_io(project_id: str, session_id: str, input_ids: list[str], output_ids: list[str]) -> dict:
    """Set the input/output node ids for a specific project/session graph."""
    with AgentSession(project_id, session_id):
        return _request(
            "PUT",
            f"{_session_prefix(project_id, session_id)}/graph/io",
            {"input_ids": input_ids, "output_ids": output_ids},
        )


@mcp.tool()
def list_builtins() -> list:
    """List available builtin neuron ids and names."""
    raw = _request("GET", "builtins")
    return [{"id": item.get("id", ""), "name": item.get("name", ""), "kind": item.get("kind", "function")} for item in raw]


@mcp.tool()
def load_gpt_template(project_id: str, session_id: str, name: str = "gpt", preset: str = "nanogpt", config: dict | None = None) -> dict:
    """Build and load a shipped language-model template into one project/session."""
    with AgentSession(project_id, session_id):
        payload = {"name": name, "config": {"preset": preset, **(config or {})}}
        result = _request("POST", f"{_session_prefix(project_id, session_id)}/templates/gpt/apply", payload)
        return result


@mcp.tool()
def add_node(project_id: str, session_id: str, neuron_id: str, instance_id: str | None = None, position: list[float] | None = None) -> dict:
    """Add a builtin neuron node to a scoped graph."""
    with AgentSession(project_id, session_id):
        builtins = _request("GET", "builtins")
        neuron_def = next((item for item in builtins if item["id"] == neuron_id), None)
        if neuron_def is None:
            raise ValueError(f"Builtin neuron {neuron_id} not found.")
        payload = {
            "instance_id": instance_id or "",
            "neuron_def": neuron_def,
            "position": position or [0.0, 0.0],
        }
        result = _request("POST", f"{_session_prefix(project_id, session_id)}/nodes", payload)
        return _brief_node(result["node"])


@mcp.tool()
def add_custom_node(
    project_id: str,
    session_id: str,
    name: str,
    source_code: str,
    input_ports: list[dict] | None = None,
    output_ports: list[dict] | None = None,
    instance_id: str | None = None,
    position: list[float] | None = None,
) -> dict:
    """Add a custom function node to a scoped graph."""
    with AgentSession(project_id, session_id):
        def _port(port: dict) -> dict:
            return {
                "name": port["name"],
                "range": port.get("range", [-10.0, 10.0]),
                "precision": port.get("precision", 0.001),
                "dtype": port.get("dtype", "float"),
            }

        payload = {
            "instance_id": instance_id or "",
            "position": position or [0.0, 0.0],
            "neuron_def": {
                "id": "",
                "name": name,
                "kind": "function",
                "input_ports": [_port(port) for port in (input_ports or [{"name": "x"}])],
                "output_ports": [_port(port) for port in (output_ports or [{"name": "y"}])],
                "source_code": source_code,
            },
        }
        result = _request("POST", f"{_session_prefix(project_id, session_id)}/nodes", payload)
        return _brief_node(result["node"])


@mcp.tool()
def add_subgraph_node(project_id: str, session_id: str, name: str = "subgraph", instance_id: str | None = None, position: list[float] | None = None) -> dict:
    """Add an empty subgraph node to a scoped graph."""
    with AgentSession(project_id, session_id):
        in_id = f"n-in-{uuid.uuid4().hex[:6]}"
        out_id = f"n-out-{uuid.uuid4().hex[:6]}"
        edge_id = f"e-{uuid.uuid4().hex[:8]}"
        subgraph = {
            "name": f"{name} graph",
            "training_method": "surrogate",
            "runtime": "scalar",
            "surrogate_config": {},
            "evo_config": {},
            "torch_config": {},
            "variant_library": {},
            "nodes": {
                in_id: {
                    "instance_id": in_id,
                    "position": [50, 150],
                    "neuron_def": {
                        "id": "builtin-input",
                        "name": "input",
                        "kind": "function",
                        "input_ports": [{"name": "in", "range": [-100, 100], "precision": 0.001, "dtype": "float"}],
                        "output_ports": [{"name": "out", "range": [-100, 100], "precision": 0.001, "dtype": "float"}],
                        "source_code": "def input_node(x):\n    return x\n",
                    },
                },
                out_id: {
                    "instance_id": out_id,
                    "position": [350, 150],
                    "neuron_def": {
                        "id": "builtin-output",
                        "name": "output",
                        "kind": "function",
                        "input_ports": [{"name": "in", "range": [-100, 100], "precision": 0.001, "dtype": "float"}],
                        "output_ports": [{"name": "out", "range": [-100, 100], "precision": 0.001, "dtype": "float"}],
                        "source_code": "def output_node(x):\n    return x\n",
                    },
                },
            },
            "edges": {
                edge_id: {
                    "id": edge_id,
                    "src_node": in_id,
                    "src_port": 0,
                    "dst_node": out_id,
                    "dst_port": 0,
                    "weight": 1.0,
                    "bias": 0.0,
                }
            },
            "input_node_ids": [in_id],
            "output_node_ids": [out_id],
        }
        payload = {
            "instance_id": instance_id or "",
            "position": position or [0.0, 0.0],
            "neuron_def": {
                "id": "",
                "name": name,
                "kind": "subgraph",
                "input_ports": [],
                "output_ports": [],
                "source_code": "",
                "subgraph": subgraph,
                "input_aliases": ["x"],
                "output_aliases": ["y"],
            },
        }
        result = _request("POST", f"{_session_prefix(project_id, session_id)}/nodes", payload)
        return _brief_node(result["node"])


@mcp.tool()
def add_variant_node(project_id: str, session_id: str, family: str, version: str, instance_id: str | None = None, position: list[float] | None = None) -> dict:
    """Add a node linked to one existing variant library entry."""
    with AgentSession(project_id, session_id):
        current = _request("GET", f"{_session_prefix(project_id, session_id)}/graph")
        graph = current["graph"]
        variant_graph = graph.get("variant_library", {}).get(family, {}).get(version)
        if variant_graph is None:
            raise ValueError(f"Variant '{family}@{version}' not found.")
        payload = {
            "instance_id": instance_id or "",
            "position": position or [0.0, 0.0],
            "neuron_def": {
                "id": "",
                "name": f"{family}_{version}",
                "kind": "subgraph",
                "input_ports": [],
                "output_ports": [],
                "source_code": "",
                "subgraph": variant_graph,
                "input_aliases": [],
                "output_aliases": [],
                "variant_ref": {"family": family, "version": version},
            },
        }
        result = _request("POST", f"{_session_prefix(project_id, session_id)}/nodes", payload)
        return _brief_node(result["node"])


@mcp.tool()
def get_node(project_id: str, session_id: str, node_id: str) -> dict:
    """Get a single node from a scoped graph."""
    graph = _request("GET", f"{_session_prefix(project_id, session_id)}/graph")["graph"]
    node = graph.get("nodes", {}).get(node_id)
    if node is None:
        raise ValueError(f"Node '{node_id}' not found.")
    return _summarize_node(node)


@mcp.tool()
def update_node(
    project_id: str,
    session_id: str,
    node_id: str,
    name: str | None = None,
    source_code: str | None = None,
    input_ports: list[dict] | None = None,
    output_ports: list[dict] | None = None,
    module_config: dict | None = None,
    input_aliases: list[str] | None = None,
    output_aliases: list[str] | None = None,
) -> dict:
    """Update one node within a scoped graph."""
    with AgentSession(project_id, session_id):
        graph = _request("GET", f"{_session_prefix(project_id, session_id)}/graph")["graph"]
        node = graph.get("nodes", {}).get(node_id)
        if node is None:
            raise ValueError(f"Node '{node_id}' not found.")
        ndef = node["neuron_def"]
        if name is not None:
            ndef["name"] = name
        if source_code is not None:
            ndef["source_code"] = source_code
        if input_ports is not None:
            ndef["input_ports"] = input_ports
        if output_ports is not None:
            ndef["output_ports"] = output_ports
        if module_config is not None:
            ndef["module_config"] = module_config
        if input_aliases is not None:
            ndef["input_aliases"] = input_aliases
        if output_aliases is not None:
            ndef["output_aliases"] = output_aliases
        result = _request("PUT", f"{_session_prefix(project_id, session_id)}/nodes/{node_id}", ndef)
        return _summarize_node(result["node"])


@mcp.tool()
def delete_node(project_id: str, session_id: str, node_id: str) -> dict:
    """Delete a node from one scoped graph."""
    with AgentSession(project_id, session_id):
        return _request("DELETE", f"{_session_prefix(project_id, session_id)}/nodes/{node_id}")


@mcp.tool()
def update_node_positions(project_id: str, session_id: str, positions: dict[str, list[float]]) -> dict:
    """Batch-update canvas positions by replacing the scoped graph document."""
    with AgentSession(project_id, session_id):
        current = _request("GET", f"{_session_prefix(project_id, session_id)}/graph")
        graph = current["graph"]
        count = 0
        for node_id, pos in positions.items():
            if node_id in graph["nodes"]:
                graph["nodes"][node_id]["position"] = pos
                count += 1
        _request(
            "PUT",
            f"{_session_prefix(project_id, session_id)}/graph",
            {
                "graph": graph,
                "expected_revision": current["revision"],
                "persist_snapshot": False,
            },
        )
        return {"status": "updated", "count": count}


@mcp.tool()
def add_edge(project_id: str, session_id: str, src_node: str, src_port: int, dst_node: str, dst_port: int, weight: float = 1.0, bias: float = 0.0) -> dict:
    """Add an edge to a scoped graph."""
    with AgentSession(project_id, session_id):
        payload = {
            "src_node": src_node,
            "src_port": src_port,
            "dst_node": dst_node,
            "dst_port": dst_port,
            "weight": weight,
            "bias": bias,
        }
        return _request("POST", f"{_session_prefix(project_id, session_id)}/edges", payload)


@mcp.tool()
def update_edge(project_id: str, session_id: str, edge_id: str, weight: float | None = None, bias: float | None = None) -> dict:
    """Update one edge in a scoped graph."""
    with AgentSession(project_id, session_id):
        payload: dict[str, float] = {}
        if weight is not None:
            payload["weight"] = weight
        if bias is not None:
            payload["bias"] = bias
        return _request("PUT", f"{_session_prefix(project_id, session_id)}/edges/{edge_id}", payload)


@mcp.tool()
def delete_edge(project_id: str, session_id: str, edge_id: str) -> dict:
    """Delete one edge from a scoped graph."""
    with AgentSession(project_id, session_id):
        return _request("DELETE", f"{_session_prefix(project_id, session_id)}/edges/{edge_id}")


@mcp.tool()
def list_variants(project_id: str, session_id: str) -> dict:
    """List variant families and versions for a scoped graph."""
    graph = _request("GET", f"{_session_prefix(project_id, session_id)}/graph")["graph"]
    return {family: list(versions.keys()) for family, versions in graph.get("variant_library", {}).items()}


@mcp.tool()
def save_node_as_variant(project_id: str, session_id: str, node_id: str, family: str, version: str, link_node: bool = True) -> dict:
    """Save a subgraph node as a variant within the same scoped graph."""
    with AgentSession(project_id, session_id):
        current = _request("GET", f"{_session_prefix(project_id, session_id)}/graph")
        graph = current["graph"]
        node = graph.get("nodes", {}).get(node_id)
        if node is None:
            raise ValueError(f"Node '{node_id}' not found.")
        ndef = node.get("neuron_def", {})
        if ndef.get("kind") != "subgraph" or not ndef.get("subgraph"):
            raise ValueError("Selected node is not a subgraph.")
        graph.setdefault("variant_library", {}).setdefault(family, {})[version] = ndef["subgraph"]
        if link_node:
            ndef["variant_ref"] = {"family": family, "version": version}
        _request(
            "PUT",
            f"{_session_prefix(project_id, session_id)}/graph",
            {"graph": graph, "expected_revision": current["revision"], "persist_snapshot": False},
        )
        return {"status": "saved", "family": family, "version": version}


@mcp.tool()
def swap_node_variant(project_id: str, session_id: str, node_id: str, family: str, version: str) -> dict:
    """Swap a linked variant on one scoped graph node."""
    with AgentSession(project_id, session_id):
        current = _request("GET", f"{_session_prefix(project_id, session_id)}/graph")
        graph = current["graph"]
        node = graph.get("nodes", {}).get(node_id)
        if node is None:
            raise ValueError(f"Node '{node_id}' not found.")
        variant_graph = graph.get("variant_library", {}).get(family, {}).get(version)
        if variant_graph is None:
            raise ValueError(f"Variant '{family}@{version}' not found.")
        node["neuron_def"]["subgraph"] = variant_graph
        node["neuron_def"]["variant_ref"] = {"family": family, "version": version}
        _request(
            "PUT",
            f"{_session_prefix(project_id, session_id)}/graph",
            {"graph": graph, "expected_revision": current["revision"], "persist_snapshot": False},
        )
        return {"status": "swapped", "family": family, "version": version}


@mcp.tool()
def execute_graph(project_id: str, session_id: str, inputs: dict[str, list[float]]) -> dict:
    """Execute the scoped graph with scalar inputs."""
    with AgentSession(project_id, session_id):
        return _request("POST", f"{_session_prefix(project_id, session_id)}/execute", {"inputs": inputs})


@mcp.tool()
def execute_trace(project_id: str, session_id: str, inputs: dict[str, list[float]]) -> dict:
    """Execute and trace a scoped graph."""
    with AgentSession(project_id, session_id):
        return _request("POST", f"{_session_prefix(project_id, session_id)}/execute-trace", {"inputs": inputs})


@mcp.tool()
def trace_torch(project_id: str, session_id: str, inputs: dict[str, list[float]]) -> dict:
    """Trace tensor stats for one scoped torch graph."""
    with AgentSession(project_id, session_id):
        return _request("POST", f"{_session_prefix(project_id, session_id)}/trace/torch", {"inputs": inputs})


@mcp.tool()
def probe_node(project_id: str, session_id: str, node_id: str, n_samples: int = 1000) -> dict:
    """Probe one node inside a scoped graph."""
    return _request("POST", f"{_session_prefix(project_id, session_id)}/probe/{node_id}?n_samples={n_samples}")


@mcp.tool()
def train_start(
    project_id: str,
    session_id: str,
    method: str = "surrogate",
    epochs: int = 10,
    learning_rate: float = 0.001,
    train_inputs: list[list[float]] | None = None,
    train_targets: list[list[float]] | None = None,
    dataset_names: list[str] | None = None,
) -> dict:
    """Start training for one project/session in the background."""
    payload = {
        "method": method,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "train_inputs": train_inputs or [],
        "train_targets": train_targets or [],
        "dataset_names": dataset_names,
    }

    def fire_and_forget() -> None:
        req = urllib.request.Request(f"{BASE_URL}/{_session_prefix(project_id, session_id)}/runs", method="POST")
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(payload).encode("utf-8")
        try:
            _ensure_authenticated()
            OPENER.open(req, timeout=1)
        except Exception:
            pass

    threading.Thread(target=fire_and_forget, daemon=True).start()
    return {
        "status": "started",
        "requested_method": method,
        "message": "Training request fired in the background.",
        "poll_hint": "Use get_training_status() or poll_training_status() with the same project/session ids.",
    }


@mcp.tool()
def get_training_status(project_id: str, session_id: str) -> dict:
    """Get the latest training snapshot for one project/session."""
    with AgentSession(project_id, session_id):
        return _fetch_training_status(project_id, session_id)


@mcp.tool()
def poll_training_status(
    project_id: str,
    session_id: str,
    since_event_id: int | None = None,
    timeout_seconds: float = 30.0,
    interval_seconds: float = 1.0,
) -> dict:
    """Poll until a scoped run emits a newer event, finishes, or times out."""
    with AgentSession(project_id, session_id):
        interval = max(float(interval_seconds), 0.1)
        deadline = time.monotonic() + max(float(timeout_seconds), 0.0)
        latest = _fetch_training_status(project_id, session_id)
        while True:
            if (
                since_event_id is None
                or latest.get("event_id", 0) > since_event_id
                or not latest.get("running")
                or latest.get("error")
            ):
                return latest
            if time.monotonic() >= deadline:
                latest["timed_out"] = True
                return latest
            time.sleep(interval)
            latest = _fetch_training_status(project_id, session_id)


@mcp.tool()
def train_stop(project_id: str, session_id: str) -> dict:
    """Stop the active training run for one project/session."""
    with AgentSession(project_id, session_id):
        status = _fetch_training_status(project_id, session_id)
        run_id = status.get("run_id")
        if not run_id:
            return {"status": "not_running"}
        return _request("POST", f"{_session_prefix(project_id, session_id)}/runs/{run_id}/stop")


@mcp.tool()
def list_datasets(project_id: str) -> list:
    """List datasets accessible from one project."""
    return _request("GET", f"{_project_prefix(project_id)}/datasets")


@mcp.tool()
def download_dataset(
    project_id: str,
    hf_path: str,
    hf_split: str = "train",
    max_rows: int | None = None,
    variant: str | None = None,
    train_shards: int | None = None,
    skip_manifest: bool = False,
    with_docs: bool = False,
    repo_id: str | None = None,
    remote_root_prefix: str = "datasets",
    project_ids: list[str] | None = None,
) -> dict:
    """Download a dataset into the scoped project catalog."""
    payload = {
        "hf_path": hf_path,
        "hf_split": hf_split,
        "max_rows": max_rows,
        "variant": variant,
        "train_shards": train_shards,
        "skip_manifest": skip_manifest,
        "with_docs": with_docs,
        "repo_id": repo_id,
        "remote_root_prefix": remote_root_prefix,
        "project_ids": project_ids,
    }
    return _request("POST", f"{_project_prefix(project_id)}/datasets/download", payload)


@mcp.tool()
def load_dataset_source(
    project_id: str,
    session_id: str,
    dataset_names: list[str] | None = None,
    hf_path: str | None = None,
    hf_split: str = "train",
    text_column: str = "text",
    max_rows: int | None = None,
    alias: str | None = None,
    variant: str | None = None,
    train_shards: int | None = None,
    skip_manifest: bool = False,
    with_docs: bool = False,
    repo_id: str | None = None,
    remote_root_prefix: str = "datasets",
    seq_len: int = 64,
    node_id: str = "dataset_source",
    append: bool = False,
    project_ids: list[str] | None = None,
) -> dict:
    """Attach one or more datasets to a dataset_source node in a scoped graph."""
    payload = {
        "dataset_names": dataset_names or [],
        "hf_path": hf_path,
        "hf_split": hf_split,
        "text_column": text_column,
        "max_rows": max_rows,
        "alias": alias,
        "variant": variant,
        "train_shards": train_shards,
        "skip_manifest": skip_manifest,
        "with_docs": with_docs,
        "repo_id": repo_id,
        "remote_root_prefix": remote_root_prefix,
        "seq_len": seq_len,
        "node_id": node_id,
        "append": append,
        "project_ids": project_ids,
    }
    with AgentSession(project_id, session_id):
        return _request("POST", f"{_session_prefix(project_id, session_id)}/datasets/load", payload)


@mcp.tool()
def set_dataset_access(project_id: str, ds_name: str, project_ids: list[str]) -> dict:
    """Update which accessible projects can see one dataset."""
    return _request("PUT", f"{_project_prefix(project_id)}/datasets/{ds_name}/access", {"project_ids": project_ids})


@mcp.tool()
def delete_dataset(project_id: str, ds_name: str) -> dict:
    """Delete one dataset from the project-visible catalog."""
    return _request("DELETE", f"{_project_prefix(project_id)}/datasets/{ds_name}")


if __name__ == "__main__":
    mcp.run()
