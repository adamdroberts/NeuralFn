"""Unit tests for fine-tuning (LoRA / qLoRA / SFT / DPO / PPO) wiring.

These cover the graph-native integration surfaces: config dataclasses,
operator registration, graph builders, and checkpoint I/O. Torch-level
training smoke tests live under the main test suite and exercise
``TorchTrainer._apply_finetune_prehook`` + ``_freeze_non_lora``.
"""

from __future__ import annotations

import os
import hashlib
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
    build_muse_glimmer_spec,
)
from neuralfn.torch_backend import (
    CompiledTorchGraph,
    GAEComputeStage,
    LoRALinearStage,
    MaskedTokenCrossEntropyStage,
    NF4LinearStage,
    PPOClippedLossStage,
    PreferenceBCELossStage,
    MaskedRewardHeadStage,
    MaskedPPOClippedLossStage,
    DPOPairwiseLossStage,
    SequenceLogpStage,
    TorchTrainer,
    TorchTrainConfig,
    PPOTrainer,
)
from neuralfn.inference import (
    load_adapter_checkpoint,
    load_pt_checkpoint,
    merge_adapter_into_base,
    save_adapter_checkpoint,
    export_to_pt,
)
from neuralfn.torch_templates import (
    _logits_model_stage_graph,
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
            "token_logp_entropy",
            "dpo_pairwise_loss",
            "dpo_dataset_source",
            "pair_batch_concat",
            "pair_batch_split",
            "reward_head",
            "masked_reward_head",
            "policy_logits_value",
            "preference_bce_loss",
            "value_head",
            "ppo_clipped_loss",
            "masked_ppo_clipped_loss",
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

    def test_finetune_spec_rejects_invalid_objective_and_digest(self):
        with self.assertRaisesRegex(ValueError, "Unsupported fine-tuning objective"):
            FineTuneSpec(objective="not-an-objective")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            FineTuneSpec(objective="sft", tokenizer_sha256="short")


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
        # An empty effective mask is a malformed SFT record, never a valid
        # zero-loss batch.
        zero_mask = torch.zeros_like(mask)
        with self.assertRaisesRegex(ValueError, "at least one unmasked"):
            stage(logits, targets, zero_mask)

    def test_masked_xe_excludes_ignore_index_from_denominator(self):
        logits = torch.randn(1, 3, 7)
        targets = torch.tensor([[1, -100, 2]])
        mask = torch.ones(1, 3)
        actual = MaskedTokenCrossEntropyStage()(logits, targets, mask)
        expected = torch.nn.functional.cross_entropy(
            logits[:, (0, 2), :].reshape(-1, 7),
            torch.tensor([1, 2]),
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))


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


class MaskedRewardHeadTest(unittest.TestCase):
    def test_uses_last_selected_token_not_padded_tail(self):
        stage = MaskedRewardHeadStage(model_dim=2)
        with torch.no_grad():
            stage.proj.weight.copy_(torch.tensor([[1.0, 0.0]]))
        hidden = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [99.0, 0.0]]])
        score = stage(hidden, torch.tensor([[1.0, 1.0, 0.0]]))
        self.assertEqual(3.0, score.item())
        with self.assertRaisesRegex(ValueError, "at least one selected"):
            stage(hidden, torch.zeros(1, 3))


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

    def test_masked_ppo_ignores_prompt_and_uses_entropy(self):
        stage = MaskedPPOClippedLossStage(
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.1,
            normalize_advantages=False,
        )
        shape = (1, 4)
        zeros = torch.zeros(shape)
        advantages = torch.tensor([[1000.0, 1000.0, 1.0, 1.0]])
        mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        entropy = torch.ones(shape)
        policy, value, bonus, loss = stage(
            zeros, zeros, advantages, zeros, zeros, zeros, mask, entropy
        )
        self.assertAlmostEqual(-1.0, policy.item(), places=6)
        self.assertAlmostEqual(0.0, value.item(), places=6)
        self.assertAlmostEqual(1.0, bonus.item(), places=6)
        self.assertAlmostEqual(-1.1, loss.item(), places=6)


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
    def test_build_dpo_root_graph_has_one_shared_policy_and_reference_forward(self):
        spec = ModelSpec(model_dim=64, num_layers=2, vocab_size=256, tie_embeddings=True)
        spec.block_spec = BlockSpec(
            family="llama", norm_type="rmsnorm", mlp_type="swiglu", pos_encoding="rope",
            linear_bias=False, num_heads=2, num_kv_heads=2, mlp_multiplier=2.0, multiple_of=64,
        )
        spec.template.objective = "dpo"
        graph = build_dpo_root_graph(model_spec=spec)
        node_ids = set(graph.nodes.keys())
        self.assertIn("policy", node_ids)
        self.assertIn("reference", node_ids)
        self.assertNotIn("policy_chosen", node_ids)
        self.assertNotIn("policy_rejected", node_ids)
        self.assertIn("pair_tokens", node_ids)
        self.assertIn("split_policy_logp", node_ids)
        self.assertIn("dpo_loss", node_ids)

    def test_glimmer_dpo_forward_backward_uses_frozen_strict_reference(self):
        from neuralfn.serialization import save_graph
        from neuralfn.torch_backend import ReferenceForwardStage

        with tempfile.TemporaryDirectory() as tmp:
            ref_graph_path = os.path.join(tmp, "reference.json")
            ref_weights_path = os.path.join(tmp, "reference.pt")
            ref_spec = AdapterCheckpointRoundTripTest._tiny_glimmer(
                "none",
                finetune=FineTuneSpec(objective="dpo"),
            )
            ref_spec.template.objective = "dpo"
            ref_graph = _logits_model_stage_graph("reference_logits", ref_spec)
            save_graph(ref_graph, ref_graph_path, include_module_state=False)
            export_to_pt(ref_graph, ref_weights_path)

            policy_spec = AdapterCheckpointRoundTripTest._tiny_glimmer(
                "none",
                finetune=FineTuneSpec(
                    objective="dpo",
                    ref_graph_path=ref_graph_path,
                    ref_checkpoint=ref_weights_path,
                ),
            )
            policy_spec.template.objective = "dpo"
            graph = build_dpo_root_graph(model_spec=policy_spec)
            compiled = CompiledTorchGraph(graph)
            chosen = torch.randint(0, 128, (2, 5))
            rejected = torch.randint(0, 128, (2, 5))
            chosen_targets = torch.randint(0, 128, (2, 5))
            rejected_targets = torch.randint(0, 128, (2, 5))
            chosen_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.float32)
            rejected_mask = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 1, 0, 0]], dtype=torch.float32)
            loss = compiled(
                chosen,
                rejected,
                chosen_targets,
                rejected_targets,
                chosen_mask,
                rejected_mask,
            )[0]
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            policy_params = list(compiled.node_modules["policy"].parameters())
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy_params))
            reference = compiled.node_modules["reference"]
            self.assertIsInstance(reference, ReferenceForwardStage)
            self.assertIsNotNone(reference._compiled)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in reference._compiled.parameters()))


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
        self.assertIn("policy_body", node_ids)
        self.assertIn("policy_heads", node_ids)
        self.assertNotIn("policy", node_ids)
        self.assertNotIn("value_new", node_ids)

    def test_real_ppo_rollout_uses_policy_reference_reward_and_updates(self):
        from neuralfn.graph import NeuronGraph, Edge, NeuronInstance
        from neuralfn.serialization import save_graph
        from neuralfn.torch_templates import make_terminal_def, clone_neuron_def

        with tempfile.TemporaryDirectory() as tmp:
            # Strict frozen reference logits graph.
            ref_graph_path = os.path.join(tmp, "ref.json")
            ref_weights_path = os.path.join(tmp, "ref.pt")
            ref_spec = AdapterCheckpointRoundTripTest._tiny_glimmer(
                "none", finetune=FineTuneSpec(objective="ppo")
            )
            ref_graph = _logits_model_stage_graph("ref_logits", ref_spec)
            save_graph(ref_graph, ref_graph_path, include_module_state=False)
            export_to_pt(ref_graph, ref_weights_path)

            # A strict scalar reward graph whose score depends on generated IDs.
            reward_graph_path = os.path.join(tmp, "reward.json")
            reward_weights_path = os.path.join(tmp, "reward.pt")
            reward_graph = NeuronGraph(name="reward_inference", training_method="torch", runtime="torch")
            reward_graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in"))
            reward_graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.token_embedding_module, config={"vocab_size": 128, "model_dim": 8}), instance_id="embed"))
            reward_graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.reward_head_module, config={"model_dim": 8, "pool": "mean"}), instance_id="reward"))
            reward_graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="reward", dtype="tensor"), instance_id="reward_out"))
            reward_graph.add_edge(Edge(id="r1", src_node="tokens_in", src_port=0, dst_node="embed", dst_port=0))
            reward_graph.add_edge(Edge(id="r2", src_node="embed", src_port=0, dst_node="reward", dst_port=0))
            reward_graph.add_edge(Edge(id="r3", src_node="reward", src_port=0, dst_node="reward_out", dst_port=0))
            reward_graph.input_node_ids = ["tokens_in"]
            reward_graph.output_node_ids = ["reward_out"]
            save_graph(reward_graph, reward_graph_path, include_module_state=False)
            export_to_pt(reward_graph, reward_weights_path)

            ft = FineTuneSpec(
                objective="ppo",
                base_checkpoint=ref_weights_path,
                base_checkpoint_sha256=hashlib.sha256(open(ref_weights_path, "rb").read()).hexdigest(),
                ref_graph_path=ref_graph_path,
                ref_checkpoint=ref_weights_path,
                reward_graph_path=reward_graph_path,
                reward_checkpoint=reward_weights_path,
                rollout_length=3,
                ppo_epochs_per_rollout=2,
                ppo_minibatch_size=2,
            )
            spec = AdapterCheckpointRoundTripTest._tiny_glimmer("none", finetune=ft)
            spec.template.objective = "ppo"
            graph = build_ppo_root_graph(model_spec=spec)
            graph.torch_config["device"] = "cpu"
            trainer = PPOTrainer(
                graph,
                TorchTrainConfig(device="cpu", learning_rate=1e-3, max_steps=1),
                rollout_length=3,
                ppo_epochs_per_rollout=2,
                ppo_minibatch_size=2,
                kl_coef=0.1,
                top_k=8,
                seed=123,
            )
            losses = trainer.train(torch.randint(0, 128, (4, 3)))
            self.assertEqual(2, len(losses))
            self.assertTrue(all(torch.isfinite(torch.tensor(loss)) for loss in losses))
            rollout = trainer.last_rollout
            self.assertIsNotNone(rollout)
            self.assertEqual((4, 6), tuple(rollout.tokens.shape))
            self.assertEqual(12.0, rollout.loss_mask.sum().item())
            self.assertGreater(rollout.logp_old.abs().sum().item(), 0.0)
            self.assertGreater(rollout.value_old.abs().sum().item(), 0.0)
            self.assertGreater(rollout.advantages.abs().sum().item(), 0.0)


class RewardModelGraphBuilderTest(unittest.TestCase):
    def test_build_reward_model_root_graph_shares_body_and_masked_reward_head(self):
        spec = ModelSpec(model_dim=64, num_layers=2, vocab_size=256, tie_embeddings=True)
        spec.block_spec = BlockSpec(
            family="llama", norm_type="rmsnorm", mlp_type="swiglu", pos_encoding="rope",
            linear_bias=False, num_heads=2, num_kv_heads=2, mlp_multiplier=2.0, multiple_of=64,
        )
        spec.template.objective = "reward_model"
        graph = build_reward_model_root_graph(model_spec=spec)
        node_ids = set(graph.nodes.keys())
        self.assertIn("body", node_ids)
        self.assertIn("reward_head", node_ids)
        self.assertIn("split_rewards", node_ids)
        self.assertNotIn("body_chosen", node_ids)
        self.assertNotIn("body_rejected", node_ids)
        self.assertIn("pref_loss", node_ids)

        compiled = CompiledTorchGraph(graph)
        chosen = torch.randint(0, 256, (2, 5))
        rejected = torch.randint(0, 256, (2, 5))
        targets = torch.randint(0, 256, (2, 5))
        masks = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 1, 0, 0]], dtype=torch.float32)
        loss = compiled(chosen, rejected, targets, targets, masks, masks)[0]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


class AdapterCheckpointRoundTripTest(unittest.TestCase):
    @staticmethod
    def _tiny_glimmer(adapter_type: str, *, finetune: FineTuneSpec) -> ModelSpec:
        spec = build_muse_glimmer_spec(
            model_dim=32,
            num_layers=2,
            vocab_size=128,
            num_heads=4,
            num_kv_heads=2,
            head_dim=4,
            attention_inner_dim=16,
            intermediate_size=48,
            window_size=8,
            adapter_type=adapter_type,
            lora_rank=2,
            lora_alpha=4.0,
            qlora_group_size=8,
            qlora_compute_dtype="float32",
            finetune=finetune,
        )
        spec.template.objective = "sft"
        return spec

    def test_glimmer_lora_has_all_eight_projection_sites_per_layer_and_trains(self):
        token_digest = "a" * 64
        with tempfile.TemporaryDirectory() as tmp:
            base_path = os.path.join(tmp, "base.pt")
            dense_spec = self._tiny_glimmer(
                "none",
                finetune=FineTuneSpec(objective="sft"),
            )
            export_to_pt(build_sft_root_graph(model_spec=dense_spec), base_path)
            base_digest = hashlib.sha256(open(base_path, "rb").read()).hexdigest()
            ft = FineTuneSpec(
                objective="sft",
                base_checkpoint=base_path,
                base_checkpoint_sha256=base_digest,
                tokenizer_sha256=token_digest,
                adapter_only_save=True,
            )
            graph = build_sft_root_graph(model_spec=self._tiny_glimmer("lora", finetune=ft))
            compiled = CompiledTorchGraph(graph)
            TorchTrainer._apply_finetune_prehook(compiled, graph)
            sites = [(name, module) for name, module in compiled.named_modules() if isinstance(module, LoRALinearStage)]
            self.assertEqual(16, len(sites))
            expected_roles = {
                "q_proj", "k_proj", "v_proj", "attn_gate_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            }
            self.assertEqual(expected_roles, {name.rsplit(".", 1)[-1] for name, _ in sites})
            trainable = {name for name, p in compiled.named_parameters() if p.requires_grad}
            self.assertTrue(trainable)
            self.assertTrue(all(name.endswith(("lora_A", "lora_B")) for name in trainable))
            tokens = torch.randint(0, 128, (2, 5))
            targets = torch.randint(0, 128, (2, 5))
            mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.float32)
            loss = compiled(tokens, targets, mask)[0]
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(any(module.lora_B.grad is not None and module.lora_B.grad.abs().sum() > 0 for _, module in sites))

            compiled.sync_state_back(graph)
            adapter_path = os.path.join(tmp, "adapter.pt")
            save_adapter_checkpoint(graph, adapter_path)
            state, meta = load_pt_checkpoint(adapter_path)
            self.assertTrue(meta.get("adapter_only"))
            self.assertEqual("neuralfn.adapter.v1", meta.get("format"))
            self.assertEqual(32, len(state))
            self.assertEqual(base_digest, meta["base_artifact"]["sha256"])
            self.assertEqual(token_digest, meta["tokenizer_sha256"])
            load_adapter_checkpoint(graph, adapter_path)

    def test_glimmer_qlora_quantizes_base_and_only_adapters_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_path = os.path.join(tmp, "base.pt")
            export_to_pt(
                build_sft_root_graph(
                    model_spec=self._tiny_glimmer("none", finetune=FineTuneSpec(objective="sft"))
                ),
                base_path,
            )
            digest = hashlib.sha256(open(base_path, "rb").read()).hexdigest()
            graph = build_sft_root_graph(
                model_spec=self._tiny_glimmer(
                    "qlora",
                    finetune=FineTuneSpec(
                        objective="sft",
                        base_checkpoint=base_path,
                        base_checkpoint_sha256=digest,
                        tokenizer_sha256="b" * 64,
                    ),
                )
            )
            compiled = CompiledTorchGraph(graph)
            TorchTrainer._apply_finetune_prehook(compiled, graph)
            modules = [module for module in compiled.modules() if isinstance(module, NF4LinearStage)]
            self.assertEqual(16, len(modules))
            self.assertTrue(all(module.qweight.dtype == torch.uint8 for module in modules))
            trainable = {name for name, p in compiled.named_parameters() if p.requires_grad}
            self.assertTrue(trainable)
            self.assertTrue(all(name.endswith(("lora_A", "lora_B")) for name in trainable))

    def test_glimmer_lora_sft_trainer_accepts_structured_masks_and_keeps_base_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_path = os.path.join(tmp, "base.pt")
            dense_graph = build_sft_root_graph(
                model_spec=self._tiny_glimmer("none", finetune=FineTuneSpec(objective="sft"))
            )
            export_to_pt(dense_graph, base_path)
            base_digest = hashlib.sha256(open(base_path, "rb").read()).hexdigest()
            graph = build_sft_root_graph(
                model_spec=self._tiny_glimmer(
                    "lora",
                    finetune=FineTuneSpec(
                        objective="sft",
                        base_checkpoint=base_path,
                        base_checkpoint_sha256=base_digest,
                        tokenizer_sha256="d" * 64,
                    ),
                )
            )
            graph.torch_config["device"] = "cpu"
            trainer = TorchTrainer(
                graph,
                TorchTrainConfig(
                    device="cpu",
                    batch_size=2,
                    epochs=2,
                    max_steps=2,
                    learning_rate=5e-3,
                    warmdown_fraction=0.0,
                ),
            )
            data = {
                "tokens": torch.randint(0, 128, (4, 5)),
                "targets": torch.randint(0, 128, (4, 5)),
                "loss_mask": torch.tensor(
                    [[0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [0, 0, 0, 1, 1], [0, 0, 1, 1, 0]],
                    dtype=torch.float32,
                ),
            }
            losses = trainer.train(data)
            self.assertEqual(2, len(losses))
            self.assertTrue(all(torch.isfinite(torch.tensor(loss)) for loss in losses))
            compiled = trainer.last_compiled_graph
            self.assertIsNotNone(compiled)
            self.assertTrue(any(
                isinstance(module, LoRALinearStage) and module.lora_B.detach().abs().sum() > 0
                for module in compiled.modules()
            ))
            base_state, _ = load_pt_checkpoint(base_path)
            for name, module in compiled.named_modules():
                if not isinstance(module, LoRALinearStage):
                    continue
                source_name = name + ".proj.weight"
                self.assertIn(source_name, base_state)
                self.assertTrue(torch.equal(module.base.weight.detach().cpu(), base_state[source_name]))

    def test_empty_adapter_artifact_is_rejected(self):
        spec = ModelSpec(model_dim=16, num_layers=1, vocab_size=32)
        spec.template.objective = "sft"
        graph = build_sft_root_graph(model_spec=spec)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "empty adapter"):
                save_adapter_checkpoint(graph, os.path.join(tmp, "empty.pt"), require_provenance=False)

    def test_sft_training_checkpoint_restores_model_optimizer_and_step(self):
        graph = build_sft_root_graph(
            model_spec=self._tiny_glimmer("none", finetune=FineTuneSpec(objective="sft"))
        )
        graph.torch_config["device"] = "cpu"
        data = {
            "tokens": torch.randint(0, 128, (4, 5)),
            "targets": torch.randint(0, 128, (4, 5)),
            "loss_mask": torch.ones(4, 5),
        }
        trainer = TorchTrainer(
            graph,
            TorchTrainConfig(
                device="cpu", batch_size=2, max_steps=1, epochs=1,
                learning_rate=1e-3, warmdown_fraction=0.0,
            ),
        )
        trainer.train(data)
        self.assertEqual(1, trainer.last_global_step)
        with tempfile.TemporaryDirectory() as tmp:
            resume_path = os.path.join(tmp, "resume.pt")
            trainer.save_training_checkpoint(resume_path)
            resumed_spec = self._tiny_glimmer(
                "none",
                finetune=FineTuneSpec(objective="sft", resume_checkpoint=resume_path),
            )
            resumed_graph = build_sft_root_graph(model_spec=resumed_spec)
            resumed_graph.torch_config["device"] = "cpu"
            resumed = TorchTrainer(
                resumed_graph,
                TorchTrainConfig(
                    device="cpu", batch_size=2, max_steps=2, epochs=2,
                    learning_rate=1e-3, warmdown_fraction=0.0,
                ),
            )
            resumed.train(data)
            self.assertEqual(2, resumed.last_global_step)
            self.assertTrue(resumed.last_optimizer_states)


class MergeAdapterTest(unittest.TestCase):
    def test_merge_adapter_into_base_produces_sum(self):
        from neuralfn.graph import NeuronGraph, Edge, NeuronInstance
        from neuralfn.torch_templates import make_terminal_def, clone_neuron_def

        with tempfile.TemporaryDirectory() as tmp:
            base_path = os.path.join(tmp, "base.pt")
            adapter_path = os.path.join(tmp, "adapter.pt")
            out_path = os.path.join(tmp, "merged.pt")

            graph = NeuronGraph(name="merge_adapter", training_method="torch", runtime="torch")
            graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="x", dtype="tensor"), instance_id="x_in"))
            graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.lora_linear_module, config={"input_dim": 8, "output_dim": 8, "rank": 2, "alpha": 4.0}), instance_id="node"))
            graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="y", dtype="tensor"), instance_id="y_out"))
            graph.add_edge(Edge(id="e1", src_node="x_in", src_port=0, dst_node="node", dst_port=0))
            graph.add_edge(Edge(id="e2", src_node="node", src_port=0, dst_node="y_out", dst_port=0))
            graph.input_node_ids = ["x_in"]
            graph.output_node_ids = ["y_out"]
            compiled = CompiledTorchGraph(graph)
            module = compiled.node_modules["node"]
            W_base = torch.zeros(8, 8)
            A = torch.randn(2, 8) * 0.1
            B = torch.randn(8, 2) * 0.1
            with torch.no_grad():
                module.lora_A.copy_(A)
                module.lora_B.copy_(B)
            compiled.sync_state_back(graph)
            base_state = {"node_modules.node.base.weight": W_base}
            torch.save({"state_dict": base_state, "checkpoint_metadata": {}}, base_path)
            save_adapter_checkpoint(
                graph,
                adapter_path,
                base_checkpoint=base_path,
                tokenizer_sha256="c" * 64,
            )
            merge_adapter_into_base(base_path, adapter_path, out_path)
            merged_state, _ = load_pt_checkpoint(out_path)
            merged = merged_state["node_modules.node.base.weight"]
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
