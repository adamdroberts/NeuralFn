import unittest
import asyncio

from neuralfn import BuiltinNeurons, Edge, NeuronGraph, NeuronInstance, subgraph_neuron
from server.models import ExecuteRequest, GraphModel, TrainRequest
from server.routes import execute, get_graph, put_graph, train_start


def make_child_graph(name: str, method: str) -> NeuronGraph:
    graph = NeuronGraph(name=name, training_method=method)
    graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
    graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
    graph.add_edge(Edge(id=f"{name}-edge", src_node="in", src_port=0, dst_node="out", dst_port=0))
    graph.input_node_ids = ["in"]
    graph.output_node_ids = ["out"]
    return graph


def make_recursive_payload() -> dict:
    child = make_child_graph("child_graph", "surrogate")
    root = NeuronGraph(name="root", training_method="frozen")
    root.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="root_in"))
    root.add_node(
        NeuronInstance(
            subgraph_neuron(
                child,
                name="child_node",
                input_aliases=["x"],
                output_aliases=["y"],
            ),
            instance_id="child_node",
        )
    )
    root.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="root_out"))
    root.add_edge(Edge(id="e1", src_node="root_in", src_port=0, dst_node="child_node", dst_port=0))
    root.add_edge(Edge(id="e2", src_node="child_node", src_port=0, dst_node="root_out", dst_port=0))
    root.input_node_ids = ["root_in"]
    root.output_node_ids = ["root_out"]
    return root.to_dict()


class ServerNestedGraphsTest(unittest.TestCase):
    @staticmethod
    async def _consume_stream(response) -> str:
        chunks: list[str] = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                chunks.append(chunk.decode())
            else:
                chunks.append(chunk)
        return "".join(chunks)

    def test_recursive_graph_round_trip_and_execute(self) -> None:
        payload = make_recursive_payload()
        put_res = put_graph(GraphModel.model_validate(payload))
        self.assertEqual("root", put_res["name"])

        data = get_graph()
        self.assertEqual("root", data["name"])
        self.assertEqual("subgraph", data["nodes"]["child_node"]["neuron_def"]["kind"])
        self.assertEqual("child_graph", data["nodes"]["child_node"]["neuron_def"]["subgraph"]["name"])

        exec_res = execute(ExecuteRequest(inputs={"root_in": [0.25]}))
        self.assertIn("root_out", exec_res)

    def test_mixed_training_stream_includes_graph_identity(self) -> None:
        payload = make_recursive_payload()
        put_graph(GraphModel.model_validate(payload))

        response = train_start(
            TrainRequest(
                method=None,
                train_inputs=[[0.0], [1.0]],
                train_targets=[[0.0], [1.0]],
                outer_rounds=1,
                epochs=1,
                generations=1,
                population_size=4,
                learning_rate=0.01,
            )
        )
        body = asyncio.run(self._consume_stream(response))

        self.assertIn("graph_path", body)
        self.assertIn("method", body)

    def test_legacy_single_graph_training_still_streams_steps(self) -> None:
        legacy = NeuronGraph(name="legacy")
        legacy.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
        legacy.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
        legacy.add_edge(Edge(id="legacy-edge", src_node="in", src_port=0, dst_node="out", dst_port=0))
        legacy.input_node_ids = ["in"]
        legacy.output_node_ids = ["out"]

        put_graph(GraphModel.model_validate(legacy.to_dict()))
        response = train_start(
            TrainRequest(
                method="surrogate",
                train_inputs=[[0.0], [1.0]],
                train_targets=[[0.0], [1.0]],
                epochs=1,
                learning_rate=0.01,
            )
        )
        body = asyncio.run(self._consume_stream(response))

        self.assertIn("\"step\"", body)


if __name__ == "__main__":
    unittest.main()
