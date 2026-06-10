"""NeuralFn CUDA Tile example."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cli"))

from neuralfn import BuiltinNeurons, Edge, NeuronGraph, NeuronInstance
from neuralfn.torch_backend import CompiledTorchGraph


graph = NeuronGraph(name="tile_strict_report")
graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="x"))
graph.add_node(NeuronInstance(BuiltinNeurons.relu, instance_id="relu"))
graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
graph.add_edge(Edge(src_node="x", src_port=0, dst_node="relu", dst_port=0))
graph.add_edge(Edge(src_node="relu", src_port=0, dst_node="out", dst_port=0))
graph.input_node_ids = ["x"]
graph.output_node_ids = ["out"]
CompiledTorchGraph(
    graph,
    kernel_backend="tile_cuda",
    tile_cuda_strict=False,
    tile_cuda_report_path="tile_cuda_report.json",
)
print("wrote tile_cuda_report.json with fallback-safe diagnostics")
try:
    CompiledTorchGraph(graph, kernel_backend="tile_cuda", tile_cuda_strict=True)
except RuntimeError as exc:
    print(f"strict mode rejected unavailable or uncovered Tile backend: {exc}")
