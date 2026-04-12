import unittest
from unittest.mock import patch

from neuralfn import TorchTrainConfig, TorchTrainer, build_gpt_root_graph, load_graph, save_graph

from neuralfn.config import build_gpt2_megakernel_spec, build_gpt2_spec, build_nanogpt_megakernel_spec
from neuralfn.torch_templates import build_model_spec_from_config


def make_gpt_graph():
    spec = build_gpt2_spec(
        vocab_size=16,
        num_layers=4,
        model_dim=32,
        num_heads=4,
    )
    graph = build_gpt_root_graph(name="model_root", model_spec=spec)
    graph.torch_config = {"device": "cpu", "amp_dtype": "bfloat16"}
    return graph


def make_megakernel_gpt_graph(*, device: str = "cpu"):
    graph = make_gpt_graph()
    graph.torch_config = {
        **graph.torch_config,
        "device": device,
        "template_spec": {"template": {"runtime": "megakernel", "objective": "ar"}},
    }
    return graph


class TorchGPTTest(unittest.TestCase):
    def test_gpt_megakernel_builders_set_backbone_and_runtime(self) -> None:
        cases = (
            (build_gpt2_megakernel_spec, "gpt2"),
            (build_nanogpt_megakernel_spec, "nanogpt"),
        )
        for builder, backbone in cases:
            with self.subTest(backbone=backbone):
                spec = builder(vocab_size=16, num_layers=2, model_dim=32, num_heads=4)
                self.assertEqual(backbone, spec.template.backbone)
                self.assertEqual("megakernel", spec.template.runtime)

    def test_build_model_spec_from_config_recognizes_gpt_megakernel_presets(self) -> None:
        cases = (
            ("gpt2_megakernel", "gpt2"),
            ("nanogpt_megakernel", "nanogpt"),
        )
        for preset, backbone in cases:
            with self.subTest(preset=preset):
                spec = build_model_spec_from_config({"preset": preset}, preview_defaults=True)
                self.assertEqual(backbone, spec.template.backbone)
                self.assertEqual("megakernel", spec.template.runtime)

    def test_gpt_template_is_a_subgraph_with_internal_stages(self) -> None:
        graph = make_gpt_graph()
        self.assertIn("attention", graph.variant_library)
        self.assertIn("mlp", graph.variant_library)
        self.assertIn("transformer_block", graph.variant_library)
        gpt_node = graph.nodes["model"]
        self.assertEqual("subgraph", gpt_node.neuron_def.kind)
        self.assertIsNotNone(gpt_node.neuron_def.subgraph)
        child = gpt_node.neuron_def.subgraph
        assert child is not None
        self.assertIn("token_embed", child.nodes)
        self.assertIn("final_norm", child.nodes)
        self.assertIn("ce", child.nodes)
        self.assertTrue(any(node.neuron_def.kind == "subgraph" for node in child.nodes.values()))
        self.assertEqual(
            {"family": "transformer_block", "version": "default"},
            child.nodes["block_0"].neuron_def.variant_ref,
        )

    def test_torch_graph_round_trip_preserves_nested_module_metadata(self) -> None:
        graph = make_gpt_graph()
        save_graph(graph, "/tmp/gpt_graph_test.json")
        loaded = load_graph("/tmp/gpt_graph_test.json")
        gpt_node = loaded.nodes["model"]
        self.assertEqual("subgraph", gpt_node.neuron_def.kind)
        child = gpt_node.neuron_def.subgraph
        assert child is not None
        self.assertEqual("module", child.nodes["token_embed"].neuron_def.kind)
        self.assertEqual(
            {"family": "transformer_block", "version": "default"},
            child.nodes["block_0"].neuron_def.variant_ref,
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
        child = graph.nodes["model"].neuron_def.subgraph
        assert child is not None
        self.assertTrue(child.nodes["token_embed"].neuron_def.module_state)

    def test_cuda_config_does_not_fall_back_to_cpu(self) -> None:
        graph = make_gpt_graph()
        graph.torch_config["device"] = "cuda"
        trainer = TorchTrainer(graph, TorchTrainConfig(epochs=1, batch_size=1))
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                trainer.train([[0, 1, 2, 3]], [[1, 2, 3, 4]])

    def test_cudagraph_marker_helper_calls_marker_for_megakernel_cuda(self) -> None:
        fake_cuda_device = type("FakeDevice", (), {"type": "cuda"})()
        with patch("neuralfn.torch_backend.torch.compiler.cudagraph_mark_step_begin") as marker:
            TorchTrainer._maybe_mark_cudagraph_step_begin("megakernel", fake_cuda_device)
        marker.assert_called_once_with()

    def test_cudagraph_marker_helper_skips_cpu_and_non_megakernel(self) -> None:
        fake_cuda_device = type("FakeDevice", (), {"type": "cuda"})()
        fake_cpu_device = type("FakeDevice", (), {"type": "cpu"})()
        with patch("neuralfn.torch_backend.torch.compiler.cudagraph_mark_step_begin") as marker:
            TorchTrainer._maybe_mark_cudagraph_step_begin("compile", fake_cuda_device)
            TorchTrainer._maybe_mark_cudagraph_step_begin("megakernel", fake_cpu_device)
        marker.assert_not_called()

    def test_compile_training_graph_uses_no_cudagraphs_for_cuda_megakernel(self) -> None:
        fake_cuda_device = type("FakeDevice", (), {"type": "cuda"})()
        module = object()
        with patch("neuralfn.torch_backend.torch.compile", return_value="compiled") as compile_mock:
            compiled = TorchTrainer._compile_training_graph(
                module,
                template_runtime="megakernel",
                device=fake_cuda_device,
                force_compile=False,
            )
        self.assertEqual("compiled", compiled)
        compile_mock.assert_called_once_with(module, mode="max-autotune-no-cudagraphs", fullgraph=True)

    def test_megakernel_train_marks_cudagraph_once_per_main_loop_microbatch(self) -> None:
        graph = make_megakernel_gpt_graph()
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
                epochs=1,
                learning_rate=5e-3,
                batch_size=1,
                weight_decay=0.0,
                device="cpu",
                train_batch_tokens=8,
                max_steps=2,
            ),
        )
        marker_calls: list[tuple[str, str]] = []
        with patch("neuralfn.torch_backend.torch.compile", side_effect=lambda module, *args, **kwargs: module):
            with patch.object(
                TorchTrainer,
                "_maybe_mark_cudagraph_step_begin",
                side_effect=lambda runtime, device: marker_calls.append((runtime, device.type)),
            ):
                losses = trainer.train(train_inputs, train_targets)
        self.assertEqual(1, len(losses))
        self.assertEqual(
            [("megakernel", "cpu")] * 4,
            marker_calls,
        )

    def test_megakernel_train_marks_cudagraph_once_per_warmup_microbatch(self) -> None:
        graph = make_megakernel_gpt_graph()
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
                epochs=1,
                learning_rate=5e-3,
                batch_size=1,
                weight_decay=0.0,
                device="cpu",
                train_batch_tokens=8,
                max_steps=1,
                optimizer_profile="parameter_golf",
                warmup_steps=1,
            ),
        )
        marker_calls: list[tuple[str, str]] = []
        with patch("neuralfn.torch_backend.torch.compile", side_effect=lambda module, *args, **kwargs: module):
            with patch.object(
                TorchTrainer,
                "_maybe_mark_cudagraph_step_begin",
                side_effect=lambda runtime, device: marker_calls.append((runtime, device.type)),
            ):
                losses = trainer.train(train_inputs, train_targets)
        self.assertEqual(1, len(losses))
        self.assertEqual(
            [("megakernel", "cpu")] * 4,
            marker_calls,
        )


if __name__ == "__main__":
    unittest.main()
