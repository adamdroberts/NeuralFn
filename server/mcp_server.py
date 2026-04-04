# /// script
# dependencies = ["mcp>=1.0.0"]
# ///

import json
import time
import uuid
import urllib.request
import urllib.error
import urllib.parse
import threading
from typing import Any, Optional
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("NeuralFn Editor")

BASE_URL = "http://localhost:8000/api"

def _request(method: str, endpoint: str, data: dict | None = None) -> Any:
    url = f"{BASE_URL}/{endpoint}"
    req = urllib.request.Request(url, method=method)
    if data is not None:
        req.add_header('Content-Type', 'application/json')
        req.data = json.dumps(data).encode('utf-8')
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        raise RuntimeError(f"API Error {e.code}: {error_body}") from e


def _fetch_training_status(
    *,
    since_event_id: int | None = None,
    history_limit: int = 10,
) -> dict[str, Any]:
    params: dict[str, int] = {"history_limit": max(0, history_limit)}
    if since_event_id is not None:
        params["since_event_id"] = since_event_id
    endpoint = "train/status"
    if params:
        endpoint = f"{endpoint}?{urllib.parse.urlencode(params)}"
    return _request("GET", endpoint)


def _summarize_training_status(status: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "run_id": status.get("run_id"),
        "status": status.get("status"),
        "running": status.get("running"),
        "done": status.get("done"),
        "method": status.get("method"),
        "requested_method": status.get("requested_method"),
        "graph_name": status.get("graph_name"),
        "dataset_names": status.get("dataset_names", []),
        "seq_len": status.get("seq_len"),
        "event_id": status.get("event_id", 0),
        "history_length": status.get("history_length", 0),
        "last_loss": status.get("last_loss"),
        "last_step": status.get("last_step"),
        "stop_requested": status.get("stop_requested", False),
        "error": status.get("error"),
        "started_at": status.get("started_at"),
        "updated_at": status.get("updated_at"),
        "completed_at": status.get("completed_at"),
        "thread_alive": status.get("thread_alive", False),
        "events": status.get("events", []),
    }
    last_event = status.get("last_event")
    if last_event is not None and not summary["events"]:
        summary["last_event"] = last_event
    return summary


# ── Response summarization helpers ────────────────────────────────────

def _summarize_graph(graph: dict) -> dict:
    """Compact graph overview: node table, edge count, settings, variant families."""
    nodes_summary = {}
    for nid, node in graph.get("nodes", {}).items():
        ndef = node.get("neuron_def", {})
        entry: dict[str, Any] = {"name": ndef.get("name", ""), "kind": ndef.get("kind", "function")}
        mt = ndef.get("module_type", "")
        if mt:
            entry["module_type"] = mt
        nodes_summary[nid] = entry
    edges_summary = [
        {"id": eid, "src": f"{e['src_node']}:{e['src_port']}", "dst": f"{e['dst_node']}:{e['dst_port']}"}
        for eid, e in graph.get("edges", {}).items()
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
    """Node detail with ports by name, source truncated, subgraph replaced with summary."""
    ndef = node.get("neuron_def", {})
    result: dict[str, Any] = {
        "instance_id": node.get("instance_id", ""),
        "position": node.get("position"),
        "name": ndef.get("name", ""),
        "kind": ndef.get("kind", "function"),
        "input_ports": [p.get("name", "") for p in ndef.get("input_ports", [])],
        "output_ports": [p.get("name", "") for p in ndef.get("output_ports", [])],
    }
    src = ndef.get("source_code", "")
    if src:
        result["source_code"] = src[:200] + ("..." if len(src) > 200 else "")
    mt = ndef.get("module_type", "")
    if mt:
        result["module_type"] = mt
    mc = ndef.get("module_config")
    if mc:
        result["module_config"] = mc
    sg = ndef.get("subgraph")
    if sg and isinstance(sg, dict):
        n_nodes = len(sg.get("nodes", {}))
        result["subgraph"] = f"(nested graph: {n_nodes} nodes)"
    vr = ndef.get("variant_ref")
    if vr:
        result["variant_ref"] = vr
    aliases_in = ndef.get("input_aliases", [])
    aliases_out = ndef.get("output_aliases", [])
    if aliases_in:
        result["input_aliases"] = aliases_in
    if aliases_out:
        result["output_aliases"] = aliases_out
    return result


def _brief_node(node: dict) -> dict:
    """Minimal node info returned after add/create operations."""
    ndef = node.get("neuron_def", {})
    return {
        "instance_id": node.get("instance_id", ""),
        "name": ndef.get("name", ""),
        "kind": ndef.get("kind", "function"),
        "position": node.get("position"),
    }


# ── Agent session ─────────────────────────────────────────────────────

class AgentSession:
    def __enter__(self):
        try:
            _request("POST", "agent/status", {"active": True})
        except Exception as e:
            print(f"Warning: Failed to set agent status: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# ── Graph tools ───────────────────────────────────────────────────────

@mcp.tool()
def get_graph() -> dict:
    """Get a summary of the current graph: node IDs with name/kind, edges, settings, and variant families."""
    return _summarize_graph(_request("GET", "graph"))

@mcp.tool()
def replace_graph(graph: dict) -> dict:
    """Replace the entire server-side graph. Returns a compact summary of the loaded graph."""
    with AgentSession():
        result = _request("PUT", "graph", graph)
        return _summarize_graph(result)

@mcp.tool()
def update_graph_settings(
    name: str = None,
    training_method: str = None,
    runtime: str = None,
    surrogate_config: dict = None,
    evo_config: dict = None,
    torch_config: dict = None,
) -> dict:
    """Update graph-level settings without replacing nodes/edges.

    training_method: 'surrogate', 'evolutionary', 'frozen', or 'torch'.
    runtime: 'scalar' or 'torch'.
    """
    with AgentSession():
        graph = _request("GET", "graph")
        if name is not None:
            graph["name"] = name
        if training_method is not None:
            graph["training_method"] = training_method
            if training_method == "torch":
                graph["runtime"] = "torch"
        if runtime is not None:
            graph["runtime"] = runtime
        if surrogate_config is not None:
            graph["surrogate_config"] = surrogate_config
        if evo_config is not None:
            graph["evo_config"] = evo_config
        if torch_config is not None:
            graph["torch_config"] = torch_config
        _request("PUT", "graph", graph)
        return {
            "status": "updated",
            "name": graph["name"],
            "training_method": graph["training_method"],
            "runtime": graph["runtime"],
        }

@mcp.tool()
def set_io(input_ids: list[str], output_ids: list[str]) -> dict:
    """Set the graph's input and output nodes."""
    with AgentSession():
        return _request("PUT", "graph/io", {"input_ids": input_ids, "output_ids": output_ids})


# ── Builtin / template tools ─────────────────────────────────────────

@mcp.tool()
def list_builtins() -> list:
    """List available builtin neuron IDs and names that can be passed to add_node."""
    raw = _request("GET", "builtins")
    return [{"id": b.get("id", ""), "name": b.get("name", ""), "kind": b.get("kind", "function")} for b in raw]

@mcp.tool()
def load_gpt_template(
    name: str = "gpt",
    preset: str = "nanogpt",
    config: dict = None,
) -> dict:
    """Build a GPT/Llama/MoE model graph and load it into the server in one step.

    Presets: 'nanogpt', 'gpt2', 'llama', 'moe'.
    Config keys: n_layer, n_head, n_embd, vocab_size, num_experts (moe), top_k (moe).
    Returns a compact summary of the loaded graph.
    """
    with AgentSession():
        payload = {"name": name, "config": {"preset": preset, **(config or {})}}
        return _request("POST", "templates/gpt/apply", payload)


# ── Node tools ────────────────────────────────────────────────────────

@mcp.tool()
def add_node(neuron_id: str, instance_id: str = None, position: list[float] = None) -> dict:
    """Add a node to the graph from a builtin neuron id."""
    with AgentSession():
        builtins = _request("GET", "builtins")
        neuron_def = next((b for b in builtins if b["id"] == neuron_id), None)
        if not neuron_def:
            raise ValueError(f"Builtin neuron {neuron_id} not found.")
        payload = {
            "instance_id": instance_id or "",
            "neuron_def": neuron_def,
            "position": position or [0.0, 0.0]
        }
        return _brief_node(_request("POST", "nodes", payload))

@mcp.tool()
def add_custom_node(
    name: str,
    source_code: str,
    input_ports: list[dict] = None,
    output_ports: list[dict] = None,
    instance_id: str = None,
    position: list[float] = None,
) -> dict:
    """Add a custom function node with user-defined Python source code.

    Each port dict should have: name (str), range (list of 2 floats, default [-10,10]),
    precision (float, default 0.001), dtype (str, default "float").
    """
    with AgentSession():
        def _port(p: dict) -> dict:
            return {
                "name": p["name"],
                "range": p.get("range", [-10.0, 10.0]),
                "precision": p.get("precision", 0.001),
                "dtype": p.get("dtype", "float"),
            }

        in_ports = [_port(p) for p in (input_ports or [{"name": "x"}])]
        out_ports = [_port(p) for p in (output_ports or [{"name": "y"}])]

        payload = {
            "instance_id": instance_id or "",
            "neuron_def": {
                "id": "",
                "name": name,
                "kind": "function",
                "input_ports": in_ports,
                "output_ports": out_ports,
                "source_code": source_code,
            },
            "position": position or [0.0, 0.0],
        }
        return _brief_node(_request("POST", "nodes", payload))


@mcp.tool()
def add_subgraph_node(
    name: str = "subgraph",
    instance_id: str = None,
    position: list[float] = None,
) -> dict:
    """Add an empty subgraph node with default input and output nodes inside it."""
    with AgentSession():
        in_id = f"n-in-{uuid.uuid4().hex[:6]}"
        out_id = f"n-out-{uuid.uuid4().hex[:6]}"
        edge_id = f"e-{uuid.uuid4().hex[:8]}"

        input_def = {
            "id": "builtin-input", "name": "input", "kind": "function",
            "input_ports": [{"name": "in", "range": [-100, 100], "precision": 0.001, "dtype": "float"}],
            "output_ports": [{"name": "out", "range": [-100, 100], "precision": 0.001, "dtype": "float"}],
            "source_code": "def input_node(x):\n    return x\n",
        }
        output_def = {
            "id": "builtin-output", "name": "output", "kind": "function",
            "input_ports": [{"name": "in", "range": [-100, 100], "precision": 0.001, "dtype": "float"}],
            "output_ports": [{"name": "out", "range": [-100, 100], "precision": 0.001, "dtype": "float"}],
            "source_code": "def output_node(x):\n    return x\n",
        }

        subgraph = {
            "name": f"{name} graph",
            "training_method": "surrogate",
            "runtime": "scalar",
            "surrogate_config": {},
            "evo_config": {},
            "torch_config": {},
            "variant_library": {},
            "nodes": {
                in_id: {"instance_id": in_id, "position": [50, 150], "neuron_def": input_def},
                out_id: {"instance_id": out_id, "position": [350, 150], "neuron_def": output_def},
            },
            "edges": {
                edge_id: {
                    "id": edge_id, "src_node": in_id, "src_port": 0,
                    "dst_node": out_id, "dst_port": 0, "weight": 1.0, "bias": 0.0,
                }
            },
            "input_node_ids": [in_id],
            "output_node_ids": [out_id],
        }

        payload = {
            "instance_id": instance_id or "",
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
            "position": position or [0.0, 0.0],
        }
        return _brief_node(_request("POST", "nodes", payload))


@mcp.tool()
def add_variant_node(
    family: str,
    version: str,
    instance_id: str = None,
    position: list[float] = None,
) -> dict:
    """Add a node linked to an existing variant in the variant library."""
    with AgentSession():
        graph = _request("GET", "graph")
        library = graph.get("variant_library", {})
        family_versions = library.get(family)
        if not family_versions:
            raise ValueError(f"Variant family '{family}' not found in library.")
        variant_graph = family_versions.get(version)
        if not variant_graph:
            raise ValueError(f"Variant '{family}@{version}' not found.")

        payload = {
            "instance_id": instance_id or "",
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
            "position": position or [0.0, 0.0],
        }
        return _brief_node(_request("POST", "nodes", payload))


@mcp.tool()
def get_node(node_id: str) -> dict:
    """Get details of a single node: ports, source code (truncated), module config, variant ref."""
    graph = _request("GET", "graph")
    node = graph.get("nodes", {}).get(node_id)
    if not node:
        raise ValueError(f"Node '{node_id}' not found.")
    return _summarize_node(node)


@mcp.tool()
def update_node(
    node_id: str,
    name: str = None,
    source_code: str = None,
    input_ports: list[dict] = None,
    output_ports: list[dict] = None,
    module_config: dict = None,
    input_aliases: list[str] = None,
    output_aliases: list[str] = None,
) -> dict:
    """Update a node's neuron definition. Only the provided fields are changed.

    Works for function nodes (name, source_code, ports) and module nodes (module_config).
    """
    with AgentSession():
        graph = _request("GET", "graph")
        node = graph.get("nodes", {}).get(node_id)
        if not node:
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

        result = _request("PUT", f"nodes/{node_id}", ndef)
        return _summarize_node(result)


@mcp.tool()
def delete_node(node_id: str) -> dict:
    """Delete a node from the graph by id."""
    with AgentSession():
        return _request("DELETE", f"nodes/{node_id}")

@mcp.tool()
def update_node_positions(positions: dict[str, list[float]]) -> dict:
    """Update the canvas positions of multiple nodes.
    positions is a dictionary mapping node_id to [x, y] coordinates.
    """
    with AgentSession():
        graph = _request("GET", "graph")
        count = 0
        for node_id, pos in positions.items():
            if node_id in graph["nodes"]:
                graph["nodes"][node_id]["position"] = pos
                count += 1
        _request("PUT", "graph", graph)
        return {"status": "updated", "count": count}


# ── Edge tools ────────────────────────────────────────────────────────

@mcp.tool()
def add_edge(src_node: str, src_port: int, dst_node: str, dst_port: int, weight: float = 1.0, bias: float = 0.0) -> dict:
    """Add an edge between two nodes in the graph."""
    with AgentSession():
        payload = {
            "src_node": src_node,
            "src_port": src_port,
            "dst_node": dst_node,
            "dst_port": dst_port,
            "weight": weight,
            "bias": bias
        }
        return _request("POST", "edges", payload)

@mcp.tool()
def update_edge(edge_id: str, weight: float = None, bias: float = None) -> dict:
    """Update an edge's weight and/or bias."""
    with AgentSession():
        payload = {}
        if weight is not None:
            payload["weight"] = weight
        if bias is not None:
            payload["bias"] = bias
        return _request("PUT", f"edges/{edge_id}", payload)

@mcp.tool()
def delete_edge(edge_id: str) -> dict:
    """Delete an edge from the graph by id."""
    with AgentSession():
        return _request("DELETE", f"edges/{edge_id}")


# ── Variant tools ─────────────────────────────────────────────────────

@mcp.tool()
def list_variants() -> dict:
    """List all variant families and their versions in the variant library."""
    graph = _request("GET", "graph")
    library = graph.get("variant_library", {})
    return {
        family: list(versions.keys())
        for family, versions in library.items()
    }

@mcp.tool()
def save_node_as_variant(node_id: str, family: str, version: str, link_node: bool = True) -> dict:
    """Save a subgraph node's nested graph as a variant in the variant library.

    If link_node is True (default), the node is converted to a variant-linked reference.
    """
    with AgentSession():
        graph = _request("GET", "graph")
        node = graph.get("nodes", {}).get(node_id)
        if not node:
            raise ValueError(f"Node '{node_id}' not found.")
        ndef = node.get("neuron_def", {})
        if ndef.get("kind") != "subgraph" or not ndef.get("subgraph"):
            raise ValueError("Selected node is not a subgraph.")

        subgraph = ndef["subgraph"]
        library = graph.get("variant_library", {})
        if family not in library:
            library[family] = {}
        library[family][version] = subgraph
        graph["variant_library"] = library

        if link_node:
            ndef["variant_ref"] = {"family": family, "version": version}

        _request("PUT", "graph", graph)
        return {"status": "saved", "family": family, "version": version}

@mcp.tool()
def swap_node_variant(node_id: str, family: str, version: str) -> dict:
    """Swap a variant-linked subgraph node to a different version of the same (or another) family."""
    with AgentSession():
        graph = _request("GET", "graph")
        node = graph.get("nodes", {}).get(node_id)
        if not node:
            raise ValueError(f"Node '{node_id}' not found.")
        ndef = node.get("neuron_def", {})
        if ndef.get("kind") != "subgraph":
            raise ValueError("Selected node is not a subgraph.")

        library = graph.get("variant_library", {})
        variant_graph = library.get(family, {}).get(version)
        if not variant_graph:
            raise ValueError(f"Variant '{family}@{version}' not found in library.")

        ndef["subgraph"] = variant_graph
        ndef["variant_ref"] = {"family": family, "version": version}

        _request("PUT", "graph", graph)
        return {"status": "swapped", "family": family, "version": version}


# ── Execution & training tools ────────────────────────────────────────

@mcp.tool()
def execute_graph(inputs: dict[str, list[float]]) -> dict:
    """Execute the current graph with the given scalar inputs."""
    with AgentSession():
        return _request("POST", "execute", {"inputs": inputs})

@mcp.tool()
def execute_trace(inputs: dict[str, list[float]]) -> dict:
    """Execute the graph and trace all intermediate outputs."""
    with AgentSession():
        return _request("POST", "execute-trace", {"inputs": inputs})

@mcp.tool()
def trace_torch(inputs: dict[str, list[float]]) -> dict:
    """Trace the execution of a torch graph to get stats on intermediate tensors."""
    with AgentSession():
        return _request("POST", "trace/torch", {"inputs": inputs})

@mcp.tool()
def probe_node(node_id: str, n_samples: int = 1000) -> dict:
    """Probe an individual node to see its outputs across a range of inputs."""
    return _request("POST", f"probe/{node_id}?n_samples={n_samples}")

@mcp.tool()
def train_start(
    method: str = "surrogate",
    epochs: int = 10,
    learning_rate: float = 0.001,
    train_inputs: list[list[float]] = None,
    train_targets: list[list[float]] = None,
    dataset_names: list[str] = None
) -> dict:
    """Start training the active graph.
    Methods: 'surrogate', 'evolutionary', 'hybrid', 'torch'.
    """
    with AgentSession():
        payload = {
            "method": method,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "train_inputs": train_inputs or [],
            "train_targets": train_targets or [],
            "dataset_names": dataset_names
        }

        def fire_and_forget():
            req = urllib.request.Request(f"{BASE_URL}/train/start", method="POST")
            req.add_header('Content-Type', 'application/json')
            req.data = json.dumps(payload).encode('utf-8')
            try:
                urllib.request.urlopen(req, timeout=1)
            except Exception:
                pass
        t = threading.Thread(target=fire_and_forget)
        t.start()
        return {
            "status": "started",
            "requested_method": method,
            "message": "Training request fired in the background.",
            "poll_hint": "Use get_training_status() for a snapshot or poll_training_status() to wait for the next loss update.",
        }

@mcp.tool()
def get_training_status(
    since_event_id: int = None,
    history_limit: int = 10,
) -> dict:
    """Get the current training snapshot. Pass since_event_id to only return newer events."""
    with AgentSession():
        status = _fetch_training_status(since_event_id=since_event_id, history_limit=history_limit)
        return _summarize_training_status(status)

@mcp.tool()
def poll_training_status(
    since_event_id: int = None,
    timeout_seconds: float = 30.0,
    interval_seconds: float = 1.0,
    history_limit: int = 10,
) -> dict:
    """Poll until training emits a newer event, finishes, or times out.

    Pass the last seen event_id to wait for the next loss update.
    """
    with AgentSession():
        if since_event_id is None:
            return _summarize_training_status(
                _fetch_training_status(history_limit=history_limit)
            )

        interval = max(float(interval_seconds), 0.1)
        deadline = time.monotonic() + max(float(timeout_seconds), 0.0)
        latest = _fetch_training_status(since_event_id=since_event_id, history_limit=history_limit)
        while True:
            if latest.get("events") or not latest.get("running") or latest.get("error"):
                return _summarize_training_status(latest)
            if time.monotonic() >= deadline:
                result = _summarize_training_status(latest)
                result["timed_out"] = True
                return result
            time.sleep(interval)
            latest = _fetch_training_status(since_event_id=since_event_id, history_limit=history_limit)

@mcp.tool()
def train_stop() -> dict:
    """Stop the current training loop."""
    with AgentSession():
        return _request("POST", "train/stop")


# ── Dataset tools ─────────────────────────────────────────────────────

@mcp.tool()
def list_datasets() -> list:
    """List local datasets."""
    return _request("GET", "datasets")

@mcp.tool()
def download_dataset(
    hf_path: str,
    hf_split: str = "train",
    max_rows: int = None,
    variant: str = None,
    train_shards: int = None,
    skip_manifest: bool = False,
    with_docs: bool = False,
    repo_id: str = None,
    remote_root_prefix: str = "datasets",
) -> dict:
    """Download a HuggingFace dataset.

    For manifest-driven cached FineWeb variants, pass `variant` (for example
    `sp1024` or `byte260`) plus optional `train_shards`.
    """
    with AgentSession():
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
        }
        return _request("POST", "datasets/download", payload)

@mcp.tool()
def load_dataset_source(
    dataset_names: list[str] = None,
    hf_path: str = None,
    hf_split: str = "train",
    text_column: str = "text",
    max_rows: int = None,
    alias: str = None,
    variant: str = None,
    train_shards: int = None,
    skip_manifest: bool = False,
    with_docs: bool = False,
    repo_id: str = None,
    remote_root_prefix: str = "datasets",
    seq_len: int = 64,
    node_id: str = "dataset_source",
    append: bool = False,
) -> dict:
    """Download/load datasets and wire them into a dataset_source node on the active graph.

    Provide `hf_path` to download from HuggingFace, or `dataset_names` to attach
    already-downloaded local datasets. Returns the configured dataset source node
    plus a compact summary of the updated graph.
    """
    with AgentSession():
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
        }
        return _request("POST", "datasets/load", payload)

@mcp.tool()
def delete_dataset(ds_name: str) -> dict:
    """Delete a dataset from local storage."""
    with AgentSession():
        return _request("DELETE", f"datasets/{ds_name}")


if __name__ == "__main__":
    mcp.run()
