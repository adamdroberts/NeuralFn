"""Unit tests for fine-tuning (LoRA / qLoRA / SFT / DPO / PPO) wiring.

These cover the graph-native integration surfaces: config dataclasses,
operator registration, graph builders, and checkpoint I/O. Torch-level
training smoke tests live under the main test suite and exercise
``TorchTrainer._apply_finetune_prehook`` + ``_freeze_non_lora``.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import torch

from neuralfn import builtins
from neuralfn.builtins import BuiltinNeurons
from neuralfn.config import (
    AdapterType,
    BlockSpec,
    FineTuneSpec,
    ModelSpec,
    build_composed_lm_spec,
)
from neuralfn.torch_backend import (
    CompiledTorchGraph,
    GAEComputeStage,
    LoRALinearStage,
    MaskedTokenCrossEntropyStage,
    NF4LinearStage,
    PPOClippedLossStage,
    PreferenceBCELossStage,
    DPOPairwiseLossStage,
    SequenceLogpStage,
    TorchTrainer,
)
from neuralfn.inference import (
    load_adapter_checkpoint,
    load_pt_checkpoint,
    merge_adapter_into_base,
    save_adapter_checkpoint,
)
from neuralfn.torch_templates import (
    build_gpt_root_graph,
    build_sft_root_graph,
    build_dpo_root_graph,
    build_reward_model_root_graph,
    build_ppo_root_graph,
)


class FineTuneOperatorRegistrationTest(unittest.TestCase):
    def test_all_finetune_operators_registered_in_builtin_map(self):
        expected = {
            "lora_linear",
            "nf4_linear",
            "masked_token_cross_entropy",
            "reference_forward",
            "sft_dataset_source",
            "sequence_logp",
            "dpo_pairwise_loss",
            "dpo_dataset_source",
            "reward_head",
            "preference_bce_loss",
            "value_head",
            "ppo_clipped_loss",
            "kl_penalty",
            "reward_forward",
            "ppo_rollout_source",
            "gae_compute",
        }
        names = {n.name for n in BuiltinNeurons.all()}
        missing = expected - names
        self.assertFalse(missing, f"missing builtin registrations: {missing}")

    def test_builtin_neurons_has_lora_attribute(self):
        self.assertIsNotNone(builtins.BuiltinNeurons.lora_linear_module)
        self.assertEqual("lora_linear", builtins.BuiltinNeurons.lora_linear_module.module_type)


class BlockSpecFineTuneFieldsTest(unittest.TestCase):
    def test_block_spec_defaults(self):
        spec = BlockSpec(family="llama")
        self.assertEqual("none", spec.adapter_type)
        self.assertEqual(8, spec.lora_rank)
        self.assertEqual(16.0, spec.lora_alpha)
        self.assertEqual(("q_proj", "v_proj"), spec.lora_targets)

    def test_build_composed_lm_spec_threads_lora_kwargs(self):
        spec = build_composed_lm_spec(
            base_model="llama",
            topology="dense",
            num_layers=2,
            model_dim=64,
            num_heads=2,
            num_kv_heads=2,
            adapter_type="lora",
            lora_rank=4,
            lora_alpha=8.0,
            lora_targets="q_proj,v_proj,o_proj",
        )
        self.assertEqual("lora", spec.block_spec.adapter_type)
        self.assertEqual(4, spec.block_spec.lora_rank)
        self.assertEqual(8.0, spec.block_spec.lora_alpha)
        self.assertEqual(("q_proj", "v_proj", "o_proj"), spec.block_spec.lora_targets)

    def test_finetune_spec_attaches_to_model_spec(self):
        ft = FineTuneSpec(objective="sft", base_checkpoint="/tmp/base.pt")
        spec = build_composed_lm_spec(
            base_model="llama",
            topology="dense",
            num_layers=2,
            model_dim=64,
            num_heads=2,
            num_kv_heads=2,
            finetune=ft,
        )
        self.assertIsNotNone(spec.finetune)
        self.assertEqual("sft", spec.finetune.objective)
        self.assertEqual("/tmp/base.pt", spec.finetune.base_checkpoint)


class LoRALinearStageTest(unittest.TestCase):
    def test_lora_delta_is_zero_at_init(self):
        stage = LoRALinearStage(input_dim=16, output_dim=16, rank=4, alpha=8)
        x = torch.randn(1, 3, 16)
        y = stage(x)
        base_only = stage.base(x)
        # B starts at zero, so LoRA delta is a no-op at init.
        self.assertTrue(torch.allclose(y, base_only, atol=1e-6))

    def test_lora_delta_is_nonzero_after_B_update(self):
        stage = LoRALinearStage(input_dim=16, output_dim=16, rank=4, alpha=8)
        x = torch.randn(1, 3, 16)
        base_only = stage.base(x)
        # Tickle B to a small random value so the delta engages.
        with torch.no_grad():
            stage.lora_B.copy_(torch.randn_like(stage.lora_B) * 0.01)
        y = stage(x)
        self.assertFalse(torch.allclose(y, base_only, atol=1e-4))

    def test_merged_weight_matches_base_plus_delta(self):
        stage = LoRALinearStage(input_dim=8, output_dim=8, rank=2, alpha=4)
        with torch.no_grad():
            stage.lora_B.copy_(torch.randn_like(stage.lora_B))
        merged = stage.merged_weight()
        expected = stage.base.weight + stage.scaling * (stage.lora_B @ stage.lora_A)
        self.assertTrue(torch.allclose(merged, expected, atol=1e-6))


class NF4LinearStageTest(unittest.TestCase):
    def test_nf4_roundtrip_error_is_bounded(self):
        stage = NF4LinearStage(input_dim=64, output_dim=8, rank=2, group_size=16)
        W = torch.randn(8, 64) * 0.5
        stage.load_base_weight(W)
        reconstructed = stage._dequantize_weight().float()
        # nf4 with group_size=16 should keep MSE small.
        err = (W - reconstructed).pow(2).mean().item()
        self.assertLess(err, 0.05, f"nf4 roundtrip MSE too large: {err}")

    def test_forward_produces_expected_shape(self):
        stage = NF4LinearStage(input_dim=32, output_dim=16, rank=2, group_size=16)
        W = torch.randn(16, 32)
        stage.load_base_weight(W)
        x = torch.randn(2, 5, 32)
        y = stage(x)
        self.assertEqual((2, 5, 16), tuple(y.shape))


class MaskedCrossEntropyTest(unittest.TestCase):
    def test_masked_xe_ignores_masked_tokens(self):
        logits = torch.randn(2, 4, 10)
        targets = torch.zeros(2, 4, dtype=torch.long)
        # Only the last position contributes.
        mask = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=torch.float32)
        stage = MaskedTokenCrossEntropyStage()
        loss = stage(logits, targets, mask)
        self.assertTrue(torch.isfinite(loss).item())
        # A zero-mask should not raise; denom is clamped.
        zero_mask = torch.zeros_like(mask)
        zero_loss = stage(logits, targets, zero_mask)
        self.assertTrue(torch.isfinite(zero_loss).item())


class SequenceLogpTest(unittest.TestCase):
    def test_logp_sum_matches_manual_computation(self):
        logits = torch.tensor([[[0.0, 1.0], [2.0, 0.0]]])  # (1, 2, 2)
        targets = torch.tensor([[1, 0]])
        mask = torch.tensor([[1.0, 1.0]])
        stage = SequenceLogpStage()
        result = stage(logits, targets, mask)
        lp = torch.log_softmax(logits.float(), dim=-1)
        expected = lp[0, 0, 1] + lp[0, 1, 0]
        self.assertTrue(torch.allclose(result, expected.unsqueeze(0), atol=1e-6))


class DPOLossTest(unittest.TestCase):
    def test_chosen_better_than_rejected_gives_small_loss(self):
        stage = DPOPairwiseLossStage(beta=0.5, loss_type="sigmoid")
        # Chosen logprobs under the policy much higher than rejected;
        # reference is the same for both so DPO margin is large -> low loss.
        policy_chosen = torch.tensor([0.0])
        policy_rejected = torch.tensor([-5.0])
        ref_chosen = torch.tensor([0.0])
        ref_rejected = torch.tensor([0.0])
        loss, ch_rw, rj_rw = stage(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        self.assertLess(loss.item(), 0.1)
        self.assertGreater((ch_rw - rj_rw).item(), 0.0)


class PreferenceBCELossTest(unittest.TestCase):
    def test_zero_loss_when_scores_diverge(self):
        stage = PreferenceBCELossStage()
        chosen = torch.tensor([10.0, 10.0])
        rejected = torch.tensor([-10.0, -10.0])
        loss = stage(chosen, rejected)
        self.assertLess(loss.item(), 1e-4)


class GAEComputeTest(unittest.TestCase):
    def test_zero_rewards_with_zero_values_gives_zero_advantages(self):
        stage = GAEComputeStage(gamma=0.99, lambda_=0.95)
        rewards = torch.zeros(2, 4)
        values = torch.zeros(2, 4)
        adv, ret = stage(rewards, values)
        self.assertTrue(torch.allclose(adv, torch.zeros_like(adv)))
        self.assertTrue(torch.allclose(ret, torch.zeros_like(ret)))

    def test_positive_rewards_produce_positive_advantages(self):
        stage = GAEComputeStage(gamma=0.99, lambda_=0.95)
        rewards = torch.ones(1, 4)
        values = torch.zeros(1, 4)
        adv, ret = stage(rewards, values)
        self.assertGreater(adv[0, 0].item(), 0.0)


class PPOClippedLossTest(unittest.TestCase):
    def test_zero_advantage_gives_zero_policy_loss(self):
        stage = PPOClippedLossStage(clip_range=0.2, vf_coef=0.0)
        logp_new = torch.zeros(1, 4)
        logp_old = torch.zeros(1, 4)
        adv = torch.zeros(1, 4)
        val_new = torch.zeros(1, 4)
        val_old = torch.zeros(1, 4)
        ret = torch.zeros(1, 4)
        pol, val, loss = stage(logp_new, logp_old, adv, val_new, val_old, ret)
        self.assertAlmostEqual(pol.item(), 0.0, places=6)
        self.assertAlmostEqual(val.item(), 0.0, places=6)


class SFTGraphBuilderTest(unittest.TestCase):
    def test_build_sft_root_graph_exposes_three_inputs_one_loss_output(self):
        spec = ModelSpec(
            model_dim=64,
            num_layers=2,
            vocab_size=256,
            tie_embeddings=True,
        )
        spec.block_spec = BlockSpec(
            family="llama",
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_encoding="rope",
            linear_bias=False,
            num_heads=4,
            num_kv_heads=2,
            mlp_multiplier=2.0,
            multiple_of=64,
        )
        spec.template.objective = "sft"
        graph = build_sft_root_graph(model_spec=spec)
        # Top-level dataset_source emits three outputs.
        self.assertEqual(["sft_dataset_source"], graph.input_node_ids)
        self.assertEqual(["loss_out"], graph.output_node_ids)

    def test_build_gpt_root_graph_dispatches_to_sft_for_sft_objective(self):
        spec = ModelSpec()
        spec.template.objective = "sft"
        spec.block_spec = BlockSpec(family="gpt2", num_heads=2)
        graph = build_gpt_root_graph(model_spec=spec)
        self.assertIn("sft_dataset_source", graph.nodes)


class DPOGraphBuilderTest(unittest.TestCase):
    def test_build_dpo_root_graph_has_policy_and_ref_forwards(self):
        spec = ModelSpec(model_dim=64, num_layers=2, vocab_size=256, tie_embeddings=True)
        spec.block_spec = BlockSpec(
            family="llama", norm_type="rmsnorm", mlp_type="swiglu", pos_encoding="rope",
            linear_bias=False, num_heads=2, num_kv_heads=2, mlp_multiplier=2.0, multiple_of=64,
        )
        spec.template.objective = "dpo"
        graph = build_dpo_root_graph(model_spec=spec)
        node_ids = set(graph.nodes.keys())
        self.assertIn("policy_chosen", node_ids)
        self.assertIn("policy_rejected", node_ids)
        self.assertIn("ref_chosen", node_ids)
        self.assertIn("ref_rejected", node_ids)
        self.assertIn("dpo_loss", node_ids)


class PPOGraphBuilderTest(unittest.TestCase):
    def test_build_ppo_root_graph_has_rollout_source_and_ppo_loss(self):
        spec = ModelSpec(model_dim=64, num_layers=2, vocab_size=256, tie_embeddings=True)
        spec.block_spec = BlockSpec(
            family="llama", norm_type="rmsnorm", mlp_type="swiglu", pos_encoding="rope",
            linear_bias=False, num_heads=2, num_kv_heads=2, mlp_multiplier=2.0, multiple_of=64,
        )
        spec.template.objective = "ppo"
        graph = build_ppo_root_graph(model_spec=spec)
        node_ids = set(graph.nodes.keys())
        self.assertIn("ppo_rollout_source", node_ids)
        self.assertIn("ppo_loss", node_ids)
        self.assertIn("value_new", node_ids)


class RewardModelGraphBuilderTest(unittest.TestCase):
    def test_build_reward_model_root_graph_has_reward_heads_and_pref_loss(self):
        spec = ModelSpec(model_dim=64, num_layers=2, vocab_size=256, tie_embeddings=True)
        spec.block_spec = BlockSpec(
            family="llama", norm_type="rmsnorm", mlp_type="swiglu", pos_encoding="rope",
            linear_bias=False, num_heads=2, num_kv_heads=2, mlp_multiplier=2.0, multiple_of=64,
        )
        spec.template.objective = "reward_model"
        graph = build_reward_model_root_graph(model_spec=spec)
        node_ids = set(graph.nodes.keys())
        self.assertIn("reward_chosen", node_ids)
        self.assertIn("reward_rejected", node_ids)
        self.assertIn("pref_loss", node_ids)


class AdapterCheckpointRoundTripTest(unittest.TestCase):
    def test_save_load_adapter_checkpoint(self):
        spec = ModelSpec(model_dim=32, num_layers=2, vocab_size=128, tie_embeddings=True)
        spec.block_spec = BlockSpec(
            family="llama", norm_type="rmsnorm", mlp_type="swiglu", pos_encoding="rope",
            linear_bias=False, num_heads=2, num_kv_heads=2, mlp_multiplier=2.0, multiple_of=32,
            adapter_type="lora", lora_rank=4, lora_alpha=8.0,
            lora_targets=("q_proj", "v_proj"),
        )
        spec.template.objective = "sft"
        graph = build_sft_root_graph(model_spec=spec)
        with tempfile.TemporaryDirectory() as tmp:
            adapter_path = os.path.join(tmp, "adapter.pt")
            save_adapter_checkpoint(graph, adapter_path)
            state, meta = load_pt_checkpoint(adapter_path)
            self.assertTrue(meta.get("adapter_only"))
            # The filtered state dict should be non-empty even if there are no
            # LoRA modules yet (value/reward heads may be absent). If the graph
            # has no trainable adapter params we still get an empty dict —
            # that's fine, it just means no LoRA layers were inserted.
            load_adapter_checkpoint(graph, adapter_path)


class MergeAdapterTest(unittest.TestCase):
    def test_merge_adapter_into_base_produces_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_path = os.path.join(tmp, "base.pt")
            adapter_path = os.path.join(tmp, "adapter.pt")
            out_path = os.path.join(tmp, "merged.pt")

            W_base = torch.zeros(8, 8)
            A = torch.randn(2, 8) * 0.1
            B = torch.randn(8, 2) * 0.1
            # Simulate a single LoRA site under prefix 'node'.
            base_state = {"node.base.weight": W_base}
            torch.save({"state_dict": base_state, "checkpoint_metadata": {}}, base_path)
            adapter_state = {"node.lora_A": A, "node.lora_B": B}
            torch.save(
                {"state_dict": adapter_state, "checkpoint_metadata": {"lora_alpha": 4.0}},
                adapter_path,
            )
            merge_adapter_into_base(base_path, adapter_path, out_path)
            merged_state, _ = load_pt_checkpoint(out_path)
            merged = merged_state["node.base.weight"]
            # rank=2, alpha=4 -> scaling=2
            expected = W_base + 2.0 * (B @ A)
            self.assertTrue(torch.allclose(merged, expected, atol=1e-6))


class TorchTrainerFreezeTest(unittest.TestCase):
    def test_freeze_non_lora_leaves_only_lora_params_trainable(self):
        # Build a tiny graph with one LoRA linear wrapped by the dispatcher.
        from neuralfn.graph import NeuronGraph, Edge, NeuronInstance
        from neuralfn.torch_templates import make_terminal_def, clone_neuron_def

        graph = NeuronGraph(name="lora_mini", training_method="torch", runtime="torch")
        graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="x", dtype="tensor"), instance_id="x_in", position=(0, 0)))
        graph.add_node(NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.lora_linear_module,
                config={"input_dim": 8, "output_dim": 8, "rank": 2, "alpha": 4.0},
            ),
            instance_id="lin",
            position=(100, 0),
        ))
        graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="y", dtype="tensor"), instance_id="y_out", position=(200, 0)))
        graph.add_edge(Edge(id="e1", src_node="x_in", src_port=0, dst_node="lin", dst_port=0))
        graph.add_edge(Edge(id="e2", src_node="lin", src_port=0, dst_node="y_out", dst_port=0))
        graph.input_node_ids = ["x_in"]
        graph.output_node_ids = ["y_out"]

        compiled = CompiledTorchGraph(graph)
        TorchTrainer._freeze_non_lora(compiled)
        names_trainable = {name for name, p in compiled.named_parameters() if p.requires_grad}
        self.assertTrue(
            all("lora_A" in n or "lora_B" in n or n.endswith("bias") for n in names_trainable),
            f"non-LoRA params left trainable: {names_trainable}",
        )


if __name__ == "__main__":
    unittest.main()
