import asyncio
import threading
import unittest
from unittest.mock import patch

from neuralfn import BuiltinNeurons, Edge, NeuronGraph, NeuronInstance, build_gpt_root_graph, subgraph_neuron
from server.models import ExecuteRequest, GraphModel, TrainRequest
from server.routes import (
    build_gpt_template,
    execute,
    get_graph,
    get_training_status,
    put_graph,
    torch_trace,
    train_start,
)


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


def make_torch_payload() -> dict:
    graph = build_gpt_root_graph(name="torch_root")
    graph.torch_config = {"device": "cpu", "amp_dtype": "bfloat16"}
    return graph.to_dict()


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

    def test_training_status_exposes_live_and_incremental_loss_updates(self) -> None:
        legacy = NeuronGraph(name="legacy_status")
        legacy.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
        legacy.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
        legacy.add_edge(Edge(id="legacy-status-edge", src_node="in", src_port=0, dst_node="out", dst_port=0))
        legacy.input_node_ids = ["in"]
        legacy.output_node_ids = ["out"]
        put_graph(GraphModel.model_validate(legacy.to_dict()))

        first_event_emitted = threading.Event()
        allow_finish = threading.Event()

        def fake_train(_self, train_inputs, train_targets, *, on_epoch=None):
            self.assertEqual((2, 1), train_inputs.shape)
            self.assertEqual((2, 1), train_targets.shape)
            if on_epoch is not None:
                on_epoch(0, 1.5)
            first_event_emitted.set()
            self.assertTrue(allow_finish.wait(timeout=1.0))
            if on_epoch is not None:
                on_epoch(1, 0.5)
            return [1.5, 0.5]

        with patch("server.services.run_service.SurrogateTrainer.train", new=fake_train):
            response = train_start(
                TrainRequest(
                    method="surrogate",
                    train_inputs=[[0.0], [1.0]],
                    train_targets=[[0.0], [1.0]],
                    epochs=2,
                    learning_rate=0.01,
                )
            )
            self.assertTrue(first_event_emitted.wait(timeout=1.0))

            live_status = get_training_status(history_limit=10)
            self.assertEqual("running", live_status["status"])
            self.assertTrue(live_status["running"])
            self.assertEqual(1.5, live_status["last_loss"])
            self.assertEqual(0, live_status["last_step"])
            live_event_id = live_status["event_id"]
            self.assertGreaterEqual(live_event_id, 1)
            self.assertEqual(2, len(live_status["events"]))

            pending_status = get_training_status(since_event_id=live_event_id, history_limit=10)
            self.assertEqual([], pending_status["events"])

            allow_finish.set()
            body = asyncio.run(self._consume_stream(response))

        final_status = get_training_status(history_limit=10)
        self.assertEqual("completed", final_status["status"])
        self.assertFalse(final_status["running"])
        self.assertTrue(final_status["done"])
        self.assertEqual(0.5, final_status["last_loss"])
        self.assertEqual(1, final_status["last_step"])
        self.assertEqual(0, final_status["history_length"])
        self.assertEqual([], final_status["events"])
        self.assertIn("\"done\": true", body)

    def test_torch_training_streams_and_updates_module_state(self) -> None:
        put_graph(GraphModel.model_validate(make_torch_payload()))
        response = train_start(
            TrainRequest(
                method="torch",
                train_inputs=[[0, 1, 2, 3], [1, 2, 3, 4]],
                train_targets=[[1, 2, 3, 4], [2, 3, 4, 5]],
                epochs=2,
                batch_size=1,
                learning_rate=0.005,
                weight_decay=0.0,
            )
        )
        body = asyncio.run(self._consume_stream(response))
        self.assertIn("\"step\"", body)
        self.assertIn("\"done\": true", body)
        data = get_graph()
        child = data["nodes"]["model"]["neuron_def"]["subgraph"]
        self.assertTrue(child["nodes"]["token_embed"]["neuron_def"]["module_state"])
        self.assertIn("transformer_block", data["variant_library"])

    def test_torch_trace_exposes_nested_stage_summaries(self) -> None:
        put_graph(GraphModel.model_validate(make_torch_payload()))
        response = torch_trace(ExecuteRequest(inputs={"tokens_in": [0, 1, 2, 3], "targets_in": [1, 2, 3, 4]}))
        trace = response["trace"]
        self.assertEqual("manual", response["source"])
        self.assertIn("model/token_embed", trace)
        self.assertIn("model/final_norm", trace)
        self.assertEqual([1, 4, 128], trace["model/final_norm"][0]["shape"])
        self.assertEqual([0, 1, 2, 3], response["sample_inputs"]["tokens_in"])

    def test_torch_trace_can_sample_from_dataset_source(self) -> None:
        payload = make_torch_payload()
        payload["nodes"]["dataset_source"] = {
            "instance_id": "dataset_source",
            "position": [0, 0],
            "neuron_def": BuiltinNeurons.dataset_source_module.to_dict(),
        }
        payload["nodes"]["dataset_source"]["neuron_def"]["module_config"] = {
            "dataset_names": ["sample_ds"],
            "seq_len": 4,
        }
        payload["edges"]["e_tokens_model"]["src_node"] = "dataset_source"
        payload["edges"]["e_tokens_model"]["src_port"] = 0
        payload["edges"]["e_targets_model"]["src_node"] = "dataset_source"
        payload["edges"]["e_targets_model"]["src_port"] = 1
        payload["input_node_ids"] = ["dataset_source"]
        payload["nodes"].pop("tokens_in")
        payload["nodes"].pop("targets_in")
        put_graph(GraphModel.model_validate(payload))

        with patch("server.services.graph_ops.load_dataset_tokens", return_value=([[0, 1, 2, 3]], [[1, 2, 3, 4]])):
            response = torch_trace(ExecuteRequest())

        self.assertEqual("dataset", response["source"])
        self.assertEqual([0, 1, 2, 3], response["sample_inputs"]["tokens"])
        self.assertIn("model", response["trace"])

    def test_torch_trace_accepts_integer_json_for_scalar_activations(self) -> None:
        graph = NeuronGraph(name="float_ops", runtime="torch", training_method="torch")
        graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
        graph.add_node(NeuronInstance(BuiltinNeurons.leaky_relu, instance_id="act"))
        graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
        graph.add_edge(Edge(id="e1", src_node="in", src_port=0, dst_node="act", dst_port=0))
        graph.add_edge(Edge(id="e2", src_node="act", src_port=0, dst_node="out", dst_port=0))
        graph.input_node_ids = ["in"]
        graph.output_node_ids = ["out"]
        put_graph(GraphModel.model_validate(graph.to_dict()))

        response = torch_trace(ExecuteRequest(inputs={"in": [1, 2, 3]}))
        self.assertEqual("manual", response["source"])
        self.assertEqual([1.0, 2.0, 3.0], response["sample_inputs"]["in"])
        self.assertIn("act", response["trace"])

    def test_gpt_template_route_returns_variant_library_payload(self) -> None:
        payload = build_gpt_template(type("Body", (), {"name": "gpt", "config": {}})())
        self.assertIn("node_def", payload)
        self.assertIn("variant_library", payload)
        self.assertIn("transformer_block", payload["variant_library"])
        block_refs = [
            node["neuron_def"]["variant_ref"]
            for node in payload["node_def"]["subgraph"]["nodes"].values()
            if node["neuron_def"].get("variant_ref")
        ]
        self.assertTrue(block_refs)
        self.assertEqual({"family": "transformer_block", "version": "default"}, block_refs[0])


if __name__ == "__main__":
    unittest.main()
