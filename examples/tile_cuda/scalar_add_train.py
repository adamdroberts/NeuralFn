"""NeuralFn CUDA Tile example."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cli"))

import torch
from neuralfn import BuiltinNeurons, Edge, NeuronGraph, NeuronInstance
from neuralfn.torch_backend import CompiledTorchGraph


def build_graph() -> NeuronGraph:
    graph = NeuronGraph(name="tile_scalar_add")
    graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="x"))
    graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="y"))
    graph.add_node(NeuronInstance(BuiltinNeurons.add, instance_id="add"))
    graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
    graph.add_edge(Edge(src_node="x", src_port=0, dst_node="add", dst_port=0))
    graph.add_edge(Edge(src_node="y", src_port=0, dst_node="add", dst_port=1))
    graph.add_edge(Edge(src_node="add", src_port=0, dst_node="out", dst_port=0))
    graph.input_node_ids = ["x", "y"]
    graph.output_node_ids = ["out"]
    return graph


compiled = CompiledTorchGraph(build_graph(), kernel_backend="tile_cuda")
x = torch.tensor([1.0, 2.0])
y = torch.tensor([3.0, 4.0])
print(compiled(x, y)[0])
