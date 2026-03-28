import unittest
from unittest.mock import patch

from neuralfn import TorchTrainConfig, TorchTrainer, build_gpt_root_graph, load_graph, save_graph


def make_gpt_graph():
    graph = build_gpt_root_graph(
        config={
            "vocab_size": 16,
            "num_layers": 4,
            "model_dim": 32,
            "num_heads": 4,
            "num_kv_heads": 2,
            "mlp_mult": 2,
            "tie_embeddings": True,
        }
    )
    graph.torch_config = {"device": "cpu", "amp_dtype": "bfloat16"}
    return graph


class TorchGPTTest(unittest.TestCase):
    def test_gpt_template_is_a_subgraph_with_internal_stages(self) -> None:
        graph = make_gpt_graph()
        self.assertIn("attention", graph.variant_library)
        self.assertIn("mlp", graph.variant_library)
        self.assertIn("transformer_block", graph.variant_library)
        gpt_node = graph.nodes["gpt"]
        self.assertEqual("subgraph", gpt_node.neuron_def.kind)
        self.assertIsNotNone(gpt_node.neuron_def.subgraph)
        child = gpt_node.neuron_def.subgraph
        assert child is not None
        self.assertIn("token_embedding", child.nodes)
        self.assertIn("final_norm", child.nodes)
        self.assertIn("token_cross_entropy", child.nodes)
        self.assertTrue(any(node.neuron_def.kind == "subgraph" for node in child.nodes.values()))
        self.assertEqual(
            {"family": "transformer_block", "version": "baseline"},
            child.nodes["encoder_block_0"].neuron_def.variant_ref,
        )

    def test_torch_graph_round_trip_preserves_nested_module_metadata(self) -> None:
        graph = make_gpt_graph()
        save_graph(graph, "/tmp/gpt_graph_test.json")
        loaded = load_graph("/tmp/gpt_graph_test.json")
        gpt_node = loaded.nodes["gpt"]
        self.assertEqual("subgraph", gpt_node.neuron_def.kind)
        child = gpt_node.neuron_def.subgraph
        assert child is not None
        self.assertEqual("module", child.nodes["token_embedding"].neuron_def.kind)
        self.assertEqual(
            {"family": "transformer_block", "version": "baseline"},
            child.nodes["encoder_block_0"].neuron_def.variant_ref,
        )
        self.assertEqual("torch", loaded.training_method)
        self.assertEqual("torch", loaded.runtime)

    def test_torch_trainer_reduces_loss_and_persists_nested_state(self) -> None:
        graph = make_gpt_graph()
        train_inputs = [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
        ]
        train_targets = [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
        ]
        trainer = TorchTrainer(
            graph,
            TorchTrainConfig(
                epochs=6,
                learning_rate=5e-3,
                batch_size=2,
                weight_decay=0.0,
                device="cpu",
            ),
        )
        losses = trainer.train(train_inputs, train_targets)
        self.assertGreater(losses[0], losses[-1])
        child = graph.nodes["gpt"].neuron_def.subgraph
        assert child is not None
        self.assertTrue(child.nodes["token_embedding"].neuron_def.module_state)

    def test_cuda_config_does_not_fall_back_to_cpu(self) -> None:
        graph = make_gpt_graph()
        graph.torch_config["device"] = "cuda"
        trainer = TorchTrainer(graph, TorchTrainConfig(epochs=1, batch_size=1))
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                trainer.train([[0, 1, 2, 3]], [[1, 2, 3, 4]])


if __name__ == "__main__":
    unittest.main()
