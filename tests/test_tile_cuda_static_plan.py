from __future__ import annotations

import torch

from neuralfn import BuiltinNeurons, Edge, NeuronGraph, NeuronInstance
from neuralfn.torch_backend import CompiledTorchGraph


def _assert_forward_does_not_read_editor_graph(compiled: CompiledTorchGraph, graph: NeuronGraph) -> None:
    def fail_incoming(_node_id: str):
        raise AssertionError("training forward must use the static execution plan")

    graph._incoming = fail_incoming  # type: ignore[method-assign]
    for node in graph.nodes.values():
        node.position = (999999.0, 999999.0)


def test_compiled_training_forward_ignores_editor_graph_metadata_after_compile() -> None:
    graph = NeuronGraph(name="tile_static_plan")
    graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="x", position=(0, 0)))
    graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="y", position=(10, 0)))
    graph.add_node(NeuronInstance(BuiltinNeurons.add, instance_id="add", position=(20, 0)))
    graph.add_node(NeuronInstance(BuiltinNeurons.relu, instance_id="relu", position=(30, 0)))
    graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out", position=(40, 0)))
    graph.add_edge(Edge(src_node="x", src_port=0, dst_node="add", dst_port=0))
    graph.add_edge(Edge(src_node="y", src_port=0, dst_node="add", dst_port=1))
    graph.add_edge(Edge(src_node="add", src_port=0, dst_node="relu", dst_port=0))
    graph.add_edge(Edge(src_node="relu", src_port=0, dst_node="out", dst_port=0))
    graph.input_node_ids = ["x", "y"]
    graph.output_node_ids = ["out"]

    compiled = CompiledTorchGraph(graph, kernel_backend="tile_cuda", tile_cuda_strict=False)
    _assert_forward_does_not_read_editor_graph(compiled, graph)

    output = compiled(torch.tensor([-2.0, 1.0]), torch.tensor([1.0, 3.0]))[0]

    torch.testing.assert_close(output, torch.tensor([0.0, 4.0]))
