import math
import tempfile
import unittest
from pathlib import Path

from neuralfn import (
    BuiltinNeurons,
    Edge,
    NeuronDef,
    NeuronGraph,
    NeuronInstance,
    Port,
    load_graph,
    neuron,
    save_graph,
)


EXPECTED_BUILTINS = [
    "sigmoid",
    "relu",
    "tanh_neuron",
    "threshold",
    "identity",
    "negate",
    "add",
    "multiply",
    "gaussian",
    "log_neuron",
    "leaky_relu",
    "prelu",
    "relu6",
    "elu",
    "selu",
    "gelu",
    "silu",
    "mish",
    "softplus",
    "softsign",
    "hard_sigmoid",
    "hard_tanh",
    "hard_swish",
    "softmax_2",
    "logsoftmax_2",
    "input_node",
    "output_node",
    "token_embedding_module",
    "linear_module",
    "rms_norm_module",
    "reshape_heads_module",
    "merge_heads_module",
    "repeat_kv_module",
    "rotary_embedding_module",
    "qk_gain_module",
    "scaled_dot_product_attention_module",
    "sliding_window_attention_module",
    "block_sparse_attention_module",
    "streaming_attention_sinks_module",
    "native_sparse_attention_module",
    "differential_attention_module",
    "residual_mix_module",
    "causal_self_attention_module",
    "fused_causal_attention_module",
    "multi_latent_attention_module",
    "residual_add_module",
    "manifold_hyper_connection_module",
    "mlp_relu2_module",
    "tied_lm_head_module",
    "lm_head_module",
    "logit_softcap_module",
    "layer_norm_module",
    "dropout_module",
    "gelu_module",
    "swiglu_module",
    "geglu_module",
    "reglu_module",
    "solu_module",
    "dyt_module",
    "group_norm_module",
    "qk_norm_module",
    "absolute_position_embedding_module",
    "kv_cache_read_module",
    "kv_cache_write_module",
    "kv_pca_encode_module",
    "kv_pca_decode_module",
    "kv_quant_pack_module",
    "kv_quant_unpack_module",
    "router_logits_module",
    "auxfree_load_balancing_module",
    "topk_route_module",
    "expert_dispatch_module",
    "expert_combine_module",
    "load_balance_loss_module",
    "aux_loss_add_module",
    "loss_scale_module",
    "token_cross_entropy_module",
    "dataset_source_module",
    "bitlinear_ternary_module",
    "fp8_linear_module",
    "mx_linear_module",
    "randmap_adapter_module",
    "mamba_module",
    "denoise_head_module",
    "mask_scheduler_module",
    "random_timesteps_module",
    "jepa_mask_module",
    "latent_pool_module",
    "jepa_projector_module",
    "jepa_predictor_module",
    "latent_mse_loss_module",
    "byte_patch_embed_module",
    "byte_patch_merge_module",
    "act_halt_gate_module",
    "act_weighted_sum_module",
    "universal_transformer_module",
    "ttt_linear_module",
    "semantic_data_source_module",
    "semantic_projector_module",
    "semantic_alignment_loss_module",
    "semantic_hasher_module",
    "semantic_moe_router_module",
    "semantic_hash_router_module",
    "causal_chunk_state_module",
    "semantic_chunk_projector_module",
    "semantic_chunk_hasher_module",
    "semantic_moe_jepa_evo_router_module",
    "broadcast_chunk_routes_module",
    "route_balance_loss_module",
    "route_selection_loss_module",
    "route_distillation_loss_module",
    "broadcast_expert_routes_module",
    "routed_attention_experts_module",
    "attentionless_decoder_module",
    "softmax_distillation_loss_module",
    "lora_linear_module",
    "nf4_linear_module",
    "masked_token_cross_entropy_module",
    "reference_forward_module",
    "sft_dataset_source_module",
    "sequence_logp_module",
    "dpo_pairwise_loss_module",
    "dpo_dataset_source_module",
    "reward_head_module",
    "preference_bce_loss_module",
    "value_head_module",
    "ppo_clipped_loss_module",
    "kl_penalty_module",
    "reward_forward_module",
    "ppo_rollout_source_module",
    "gae_compute_module",
]


class BuiltinNeuronsTest(unittest.TestCase):
    def test_catalog_exposes_all_expected_builtins(self) -> None:
        builtins = BuiltinNeurons.all()
        builtin_attrs = [getattr(BuiltinNeurons, name) for name in EXPECTED_BUILTINS]

        self.assertEqual(len(EXPECTED_BUILTINS), len(builtins))
        self.assertEqual(builtin_attrs, builtins)

        for name in EXPECTED_BUILTINS:
            neuron_def = getattr(BuiltinNeurons, name)
            self.assertIsInstance(neuron_def, NeuronDef)
            self.assertIn(neuron_def, builtins)

        self.assertIs(BuiltinNeurons.get("sigmoid"), BuiltinNeurons.sigmoid)
        self.assertIs(BuiltinNeurons.get("log_neuron"), BuiltinNeurons.log_neuron)
        self.assertIs(BuiltinNeurons.get("log"), BuiltinNeurons.log_neuron)
        self.assertIs(BuiltinNeurons.get("input_node"), BuiltinNeurons.input_node)
        self.assertIs(BuiltinNeurons.get("input"), BuiltinNeurons.input_node)
        self.assertIs(BuiltinNeurons.get("output_node"), BuiltinNeurons.output_node)
        self.assertIs(BuiltinNeurons.get("output"), BuiltinNeurons.output_node)
        self.assertIs(BuiltinNeurons.get("token_embedding_module"), BuiltinNeurons.token_embedding_module)
        self.assertIs(BuiltinNeurons.get("token_embedding"), BuiltinNeurons.token_embedding_module)
        self.assertIs(BuiltinNeurons.get("rms_norm"), BuiltinNeurons.rms_norm_module)
        self.assertIs(BuiltinNeurons.get("causal_self_attention"), BuiltinNeurons.causal_self_attention_module)

        serialized_names = [neuron_def.name for neuron_def in builtins]
        self.assertIn("log", serialized_names)
        self.assertIn("input", serialized_names)
        self.assertIn("output", serialized_names)
        self.assertIn("token_embedding", serialized_names)
        self.assertIn("token_cross_entropy", serialized_names)
        self.assertIn("jepa_mask", serialized_names)
        self.assertIn("universal_transformer", serialized_names)
        self.assertIn("ttt_linear", serialized_names)

    def test_graph_executes_with_builtin_helper_nodes(self) -> None:
        graph = NeuronGraph()
        graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
        graph.add_node(NeuronInstance(BuiltinNeurons.sigmoid, instance_id="act"))
        graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
        graph.add_edge(Edge(id="e1", src_node="in", src_port=0, dst_node="act", dst_port=0))
        graph.add_edge(Edge(id="e2", src_node="act", src_port=0, dst_node="out", dst_port=0))
        graph.input_node_ids = ["in"]
        graph.output_node_ids = ["out"]

        result = graph.execute({"in": (0.0,)})

        self.assertIn("out", result)
        self.assertAlmostEqual(0.5, result["out"][0], places=6)

    def test_builtin_ports_default_to_unquantized_precision_except_discrete_contracts(self) -> None:
        self.assertIsNone(BuiltinNeurons.add.input_ports[0].precision)
        self.assertIsNone(BuiltinNeurons.add.output_ports[0].precision)
        self.assertIsNone(BuiltinNeurons.residual_add_module.input_ports[0].precision)
        self.assertIsNone(BuiltinNeurons.token_cross_entropy_module.output_ports[0].precision)
        self.assertEqual(1.0, BuiltinNeurons.token_embedding_module.input_ports[0].precision)
        self.assertEqual(1.0, BuiltinNeurons.threshold.output_ports[0].precision)
        self.assertEqual("bool", BuiltinNeurons.threshold.output_ports[0].dtype)

    def test_port_round_trip_supports_unquantized_precision(self) -> None:
        port = Port("x", range=(-10, 10), precision=None)
        self.assertAlmostEqual(0.123456, port.condition(0.123456))

        payload = port.to_dict()
        self.assertIsNone(payload["precision"])
        loaded = Port.from_dict(payload)
        self.assertIsNone(loaded.precision)
        self.assertAlmostEqual(0.123456, loaded.condition(0.123456))

        legacy = Port.from_dict({"name": "x", "range": [-10, 10], "dtype": "float"})
        self.assertIsNone(legacy.precision)
        self.assertAlmostEqual(0.1234, legacy.condition(0.1234))

    def test_port_round_trip_supports_unbounded_range(self) -> None:
        port = Port("x", precision=None)
        self.assertIsNone(port.range)
        self.assertEqual(1_000_000.0, port.condition(1_000_000.0))

        payload = port.to_dict()
        self.assertIsNone(payload["range"])
        loaded = Port.from_dict(payload)
        self.assertIsNone(loaded.range)
        self.assertEqual(-1_000_000.0, loaded.condition(-1_000_000.0))

    def test_save_load_round_trip_preserves_builtin_serialized_names(self) -> None:
        graph = NeuronGraph()
        graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
        graph.add_node(NeuronInstance(BuiltinNeurons.log_neuron, instance_id="log"))
        graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
        graph.add_edge(Edge(id="e1", src_node="in", src_port=0, dst_node="log", dst_port=0))
        graph.add_edge(Edge(id="e2", src_node="log", src_port=0, dst_node="out", dst_port=0))
        graph.input_node_ids = ["in"]
        graph.output_node_ids = ["out"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"
            save_graph(graph, path)
            loaded = load_graph(path)

        self.assertEqual("input", loaded.nodes["in"].name)
        self.assertEqual("log", loaded.nodes["log"].name)
        self.assertEqual("output", loaded.nodes["out"].name)

        result = loaded.execute({"in": (1.5,)})
        self.assertEqual(BuiltinNeurons.log_neuron(1.5)[0], result["out"][0])

    def test_custom_neuron_path_remains_separate(self) -> None:
        @neuron(
            inputs=[Port("x", range=(-10, 10), precision=0.001)],
            outputs=[Port("y", range=(-20, 20), precision=0.001)],
        )
        def custom_scale(x: float) -> float:
            return x * 2.0

        self.assertIsInstance(custom_scale, NeuronDef)
        self.assertEqual((6.0,), custom_scale(3.0))
        self.assertNotIn(custom_scale, BuiltinNeurons.all())

        with self.assertRaises(KeyError):
            BuiltinNeurons.get("custom_scale")


if __name__ == "__main__":
    unittest.main()
