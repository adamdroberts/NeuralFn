import unittest
from unittest.mock import patch

import torch
from torch import nn

from neuralfn import (
    BuiltinNeurons,
    Edge,
    NeuronGraph,
    NeuronInstance,
    TorchTrainConfig,
    TorchTrainer,
    build_gpt_root_graph,
    load_graph,
    save_graph,
)

from neuralfn.config import build_gpt2_megakernel_spec, build_gpt2_spec, build_nanogpt_megakernel_spec
from neuralfn.tile_cuda import NVFP4Tensor, dequantize_nvfp4_reference
from neuralfn.torch_backend import CompiledTorchGraph
from neuralfn.torch_templates import build_model_spec_from_config, clone_neuron_def


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
    def test_compiled_torch_graph_forward_uses_static_execution_plan(self) -> None:
        graph = NeuronGraph(name="compiled_hot_path", training_method="torch", runtime="torch")
        graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="x"))
        graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="y"))
        graph.add_node(NeuronInstance(BuiltinNeurons.add, instance_id="add"))
        graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
        graph.add_edge(Edge(id="e1", src_node="x", src_port=0, dst_node="add", dst_port=0))
        graph.add_edge(Edge(id="e2", src_node="y", src_port=0, dst_node="add", dst_port=1))
        graph.add_edge(Edge(id="e3", src_node="add", src_port=0, dst_node="out", dst_port=0))
        graph.input_node_ids = ["x", "y"]
        graph.output_node_ids = ["out"]

        compiled = CompiledTorchGraph(graph)

        def fail_incoming(_node_id: str):
            raise AssertionError("Compiled forward must not traverse graph editor edges")

        graph._incoming = fail_incoming  # type: ignore[method-assign]
        graph.nodes = None  # type: ignore[assignment]
        result = compiled(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))

        self.assertEqual(1, len(result))
        self.assertTrue(torch.equal(result[0], torch.tensor([4.0, 6.0])))

    def test_compiled_torch_graph_packs_nvfp4_tile_activation_inputs_only(self) -> None:
        graph = NeuronGraph(name="compiled_nvfp4_pack", training_method="torch", runtime="torch")
        graph.torch_config = {"tile_cuda_activation_dtype": "nvfp4"}
        graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="hidden"))
        graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="weight"))
        graph.add_node(
            NeuronInstance(
                clone_neuron_def(BuiltinNeurons.tied_lm_head_module),
                instance_id="tied_head",
            )
        )
        graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
        graph.add_edge(Edge(id="e_hidden_head", src_node="hidden", src_port=0, dst_node="tied_head", dst_port=0))
        graph.add_edge(Edge(id="e_weight_head", src_node="weight", src_port=0, dst_node="tied_head", dst_port=1))
        graph.add_edge(Edge(id="e_head_out", src_node="tied_head", src_port=0, dst_node="out", dst_port=0))
        graph.input_node_ids = ["hidden", "weight"]
        graph.output_node_ids = ["out"]

        class CaptureTiedHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.seen_hidden = None
                self.seen_weight = None

            def forward(self, hidden, tied_weight):
                self.seen_hidden = hidden
                self.seen_weight = tied_weight
                hidden_tensor = dequantize_nvfp4_reference(hidden) if isinstance(hidden, NVFP4Tensor) else hidden
                return hidden_tensor @ tied_weight.t()

        compiled = CompiledTorchGraph(graph, kernel_backend="torch")
        compiled.resolved_kernel_backend = "tile_cuda"
        capture = CaptureTiedHead()
        compiled.node_modules["tied_head"] = capture
        hidden = torch.randn(2, 4).contiguous()
        weight = torch.randn(3, 4).contiguous()

        (logits,) = compiled(hidden, weight)

        self.assertEqual((2, 3), tuple(logits.shape))
        self.assertIsInstance(capture.seen_hidden, NVFP4Tensor)
        self.assertIs(capture.seen_weight, weight)

    def test_compiled_torch_graph_propagates_nvfp4_activation_config_to_subgraphs(self) -> None:
        graph = make_gpt_graph()
        graph.torch_config = {**graph.torch_config, "tile_cuda_activation_dtype": "nvfp4"}

        compiled = CompiledTorchGraph(graph, kernel_backend="torch")

        child = compiled.node_modules["model"]
        self.assertIsInstance(child, CompiledTorchGraph)
        self.assertEqual("nvfp4", child.tile_cuda_activation_dtype)

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

    def test_torch_trainer_exposes_active_compiled_graph_during_step_callbacks(self) -> None:
        graph = make_gpt_graph()
        trainer = TorchTrainer(
            graph,
            TorchTrainConfig(
                epochs=1,
                learning_rate=5e-3,
                batch_size=2,
                weight_decay=0.0,
                device="cpu",
                max_steps=1,
            ),
        )
        seen_active = []

        def on_step(_info):
            seen_active.append(trainer.active_compiled_graph is not None)

        trainer.train(
            [[0, 1, 2, 3], [1, 2, 3, 4]],
            [[1, 2, 3, 4], [2, 3, 4, 5]],
            on_step=on_step,
        )

        self.assertEqual([True], seen_active)
        self.assertIsNone(trainer.active_compiled_graph)
        self.assertIsNotNone(trainer.last_compiled_graph)

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
