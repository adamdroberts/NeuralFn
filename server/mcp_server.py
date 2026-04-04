import json
import urllib.request
import urllib.error
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


class AgentSession:
    def __enter__(self):
        try:
            _request("POST", "agent/status", {"active": True})
        except Exception as e:
            print(f"Warning: Failed to set agent status: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@mcp.tool()
def get_graph() -> dict:
    """Get the current NeuralFn graph structure including nodes and edges."""
    return _request("GET", "graph")

@mcp.tool()
def list_builtins() -> list:
    """List available builtin neuron definitions that can be added to the graph."""
    return _request("GET", "builtins")

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
        return _request("POST", "nodes", payload)

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
def delete_node(node_id: str) -> dict:
    """Delete a node from the graph by id."""
    with AgentSession():
        return _request("DELETE", f"nodes/{node_id}")

@mcp.tool()
def delete_edge(edge_id: str) -> dict:
    """Delete an edge from the graph by id."""
    with AgentSession():
        return _request("DELETE", f"edges/{edge_id}")

@mcp.tool()
def update_node_positions(positions: dict[str, list[float]]) -> dict:
    """
    Update the canvas positions of multiple nodes.
    positions is a dictionary mapping node_id to [x, y] coordinates.
    """
    with AgentSession():
        graph = _request("GET", "graph")
        for node_id, pos in positions.items():
            if node_id in graph["nodes"]:
                graph["nodes"][node_id]["position"] = pos
        return _request("PUT", "graph", graph)

@mcp.tool()
def set_io(input_ids: list[str], output_ids: list[str]) -> dict:
    """Set the graph's input and output nodes."""
    with AgentSession():
        return _request("PUT", "graph/io", {"input_ids": input_ids, "output_ids": output_ids})

@mcp.tool()
def build_gpt_template(name: str = "gpt", config: dict = None) -> dict:
    """Build a GPT model template. Returns the generated Template payload to be incorporated into the graph."""
    with AgentSession():
        payload = {"name": name, "config": config or {}}
        return _request("POST", "templates/gpt", payload)

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
) -> str:
    """
    Start training the active graph.
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
        return f"Training started ({method}) streaming request fired in background."

@mcp.tool()
def train_stop() -> dict:
    """Stop the current training loop."""
    with AgentSession():
        return _request("POST", "train/stop")

@mcp.tool()
def list_datasets() -> list:
    """List local datasets."""
    return _request("GET", "datasets")

@mcp.tool()
def download_dataset(hf_path: str, hf_split: str = "train", max_rows: int = None) -> dict:
    """Download a HuggingFace dataset."""
    with AgentSession():
        payload = {
            "hf_path": hf_path,
            "hf_split": hf_split,
            "max_rows": max_rows
        }
        return _request("POST", "datasets/download", payload)

@mcp.tool()
def delete_dataset(ds_name: str) -> dict:
    """Delete a dataset from local storage."""
    with AgentSession():
        return _request("DELETE", f"datasets/{ds_name}")

if __name__ == "__main__":
    mcp.run()
