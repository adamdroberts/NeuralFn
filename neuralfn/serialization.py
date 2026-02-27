from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .graph import NeuronGraph


def save_graph(graph: NeuronGraph, path: str | Path) -> None:
    """Serialise a NeuronGraph to a JSON file."""
    data = graph.to_dict()
    Path(path).write_text(json.dumps(data, indent=2))


def load_graph(path: str | Path) -> NeuronGraph:
    """Deserialise a NeuronGraph from a JSON file."""
    data = json.loads(Path(path).read_text())
    return NeuronGraph.from_dict(data)
