import unittest
from unittest.mock import patch

from neuralfn import build_gpt_root_graph
from server.models import GraphModel, LoadDatasetRequest
from server.routes import get_graph, load_dataset, put_graph


class ServerDatasetLoadingTest(unittest.TestCase):
    def test_load_dataset_rewires_gpt_inputs_to_dataset_source(self) -> None:
        graph = build_gpt_root_graph(name="torch_root")
        graph.torch_config = {"device": "cpu", "amp_dtype": "bfloat16"}
        put_graph(GraphModel.model_validate(graph.to_dict()))

        with patch("server.services.graph_ops.download_hf_dataset", return_value={"name": "HuggingFaceFW__fineweb"}):
            result = load_dataset(
                LoadDatasetRequest(
                    hf_path="HuggingFaceFW/fineweb",
                    max_rows=128,
                    seq_len=96,
                )
            )

        self.assertEqual("dataset_source", result["dataset_source_node_id"])
        self.assertEqual(["HuggingFaceFW__fineweb"], result["dataset_names"])
        self.assertEqual(["dataset_source"], result["graph"]["input_node_ids"])

        data = get_graph()
        self.assertIn("dataset_source", data["nodes"])
        self.assertNotIn("tokens_in", data["nodes"])
        self.assertNotIn("targets_in", data["nodes"])
        self.assertEqual("dataset_source", data["edges"]["e_tokens_model"]["src_node"])
        self.assertEqual(0, data["edges"]["e_tokens_model"]["src_port"])
        self.assertEqual("dataset_source", data["edges"]["e_targets_model"]["src_node"])
        self.assertEqual(1, data["edges"]["e_targets_model"]["src_port"])
        cfg = data["nodes"]["dataset_source"]["neuron_def"]["module_config"]
        self.assertEqual(["HuggingFaceFW__fineweb"], cfg["dataset_names"])
        self.assertEqual(96, cfg["seq_len"])

    def test_load_dataset_passes_variant_download_options(self) -> None:
        graph = build_gpt_root_graph(name="torch_root")
        graph.torch_config = {"device": "cpu", "amp_dtype": "bfloat16"}
        put_graph(GraphModel.model_validate(graph.to_dict()))

        with patch(
            "server.services.graph_ops.download_hf_dataset",
            return_value={"name": "willdepueoai__parameter-golf__sp1024__train10"},
        ) as mocked:
            load_dataset(
                LoadDatasetRequest(
                    hf_path="willdepueoai/parameter-golf",
                    variant="sp1024",
                    train_shards=10,
                    repo_id="willdepueoai/parameter-golf",
                )
            )

        _, kwargs = mocked.call_args
        self.assertEqual("sp1024", kwargs["variant"])
        self.assertEqual(10, kwargs["train_shards"])
        self.assertEqual("willdepueoai/parameter-golf", kwargs["repo_id"])


if __name__ == "__main__":
    unittest.main()
