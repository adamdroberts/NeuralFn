from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .graph import NeuronGraph


def _clear_module_state_from_graph_payload(payload: dict[str, Any]) -> None:
    for node in payload.get("nodes", {}).values():
        neuron_def = node.get("neuron_def", {})
        if isinstance(neuron_def, dict):
            neuron_def["module_state"] = ""
            subgraph = neuron_def.get("subgraph")
            if isinstance(subgraph, dict):
                _clear_module_state_from_graph_payload(subgraph)
    for versions in payload.get("variant_library", {}).values():
        if not isinstance(versions, dict):
            continue
        for graph_payload in versions.values():
            if isinstance(graph_payload, dict):
                _clear_module_state_from_graph_payload(graph_payload)


def save_graph(
    graph: NeuronGraph,
    path: str | Path,
    *,
    include_module_state: bool = True,
) -> None:
    """Serialise a NeuronGraph to a JSON file."""
    data = graph.to_dict()
    if not include_module_state:
        _clear_module_state_from_graph_payload(data)
    Path(path).write_text(json.dumps(data, indent=2))


def load_graph(path: str | Path) -> NeuronGraph:
    """Deserialise a NeuronGraph from a JSON file."""
    data = json.loads(Path(path).read_text())
    return NeuronGraph.from_dict(data)
