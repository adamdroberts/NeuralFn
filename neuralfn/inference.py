import torch
from pathlib import Path
from typing import Any

from .graph import NeuronGraph
from .torch_backend import CompiledTorchGraph

def export_to_pt(graph: NeuronGraph, path: str | Path) -> None:
    """Export the weights of a compiled or uncompiled torch-based NeuronGraph to a .pt file."""
    # First, make sure we have a compiled graph to extract state from
    compiled = CompiledTorchGraph(graph)
    state_dict = compiled.state_dict()
    torch.save(state_dict, path)

def import_from_pt(graph: NeuronGraph, path: str | Path) -> None:
    """Import weights from a .pt file into a NeuronGraph's module_state."""
    state_dict = torch.load(path, weights_only=True)
    compiled = CompiledTorchGraph(graph)
    compiled.load_state_dict(state_dict)
    compiled.sync_state_back(graph)
