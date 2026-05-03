"""CLI tests for fine-tuning flags (Phase 1-4)."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
NEURALFN_ROOT = ROOT.parent
SCRIPTS_DIR = ROOT / "scripts"
for candidate in (ROOT, NEURALFN_ROOT, SCRIPTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import nfn_impl


class FineTuneFlagParseTest(unittest.TestCase):
    def _parse_train(self, *argv: str):
        parser = nfn_impl.build_command_parser("train", style="long")
        return parser.parse_args(list(argv))

    def test_training_mode_flag_parses_all_objectives(self):
        for mode in ("pretrain", "sft", "dpo", "ppo", "reward_model"):
            args = self._parse_train("--training-mode", mode)
            self.assertEqual(mode, args.training_mode)

    def test_adapter_type_flag_parses_all_variants(self):
        for adapter in ("none", "lora", "qlora", "randmap"):
            args = self._parse_train("--adapter-type", adapter)
            self.assertEqual(adapter, args.adapter_type)

    def test_training_mode_rejects_unknown(self):
        parser = nfn_impl.build_command_parser("train", style="long")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--training-mode", "bogus"])

    def test_lora_knobs_parse(self):
        args = self._parse_train(
            "--adapter-type", "lora",
            "--lora-rank", "16",
            "--lora-alpha", "32",
            "--lora-dropout", "0.05",
            "--lora-targets", "q_proj,v_proj,o_proj",
            "--lora-bias",
        )
        self.assertEqual("lora", args.adapter_type)
        self.assertEqual(16, args.lora_rank)
        self.assertEqual(32.0, args.lora_alpha)
        self.assertEqual(0.05, args.lora_dropout)
        self.assertEqual("q_proj,v_proj,o_proj", args.lora_targets)
        self.assertTrue(args.lora_bias)

    def test_qlora_knobs_parse(self):
        args = self._parse_train(
            "--adapter-type", "qlora",
            "--qlora-group-size", "128",
            "--qlora-compute-dtype", "fp16",
        )
        self.assertEqual(128, args.qlora_group_size)
        self.assertEqual("fp16", args.qlora_compute_dtype)

    def test_checkpoint_flags_parse(self):
        args = self._parse_train(
            "--base-checkpoint", "artifacts/base.pt",
            "--ref-checkpoint", "artifacts/ref.pt",
            "--reward-checkpoint", "artifacts/reward.pt",
            "--adapter-only-save",
        )
        self.assertEqual("artifacts/base.pt", args.base_checkpoint)
        self.assertEqual("artifacts/ref.pt", args.ref_checkpoint)
        self.assertEqual("artifacts/reward.pt", args.reward_checkpoint)
        self.assertTrue(args.adapter_only_save)

    def test_dpo_flags_parse(self):
        args = self._parse_train(
            "--training-mode", "dpo",
            "--dpo-beta", "0.3",
            "--dpo-loss-type", "hinge",
        )
        self.assertEqual("dpo", args.training_mode)
        self.assertEqual(0.3, args.dpo_beta)
        self.assertEqual("hinge", args.dpo_loss_type)

    def test_ppo_flags_parse(self):
        args = self._parse_train(
            "--training-mode", "ppo",
            "--kl-coef", "0.05",
            "--ppo-clip", "0.1",
            "--ppo-vf-coef", "0.25",
            "--ppo-ent-coef", "0.01",
            "--rollout-length", "128",
            "--ppo-epochs-per-rollout", "8",
        )
        self.assertEqual("ppo", args.training_mode)
        self.assertEqual(0.05, args.kl_coef)
        self.assertEqual(0.1, args.ppo_clip)
        self.assertEqual(0.25, args.ppo_vf_coef)
        self.assertEqual(0.01, args.ppo_ent_coef)
        self.assertEqual(128, args.rollout_length)
        self.assertEqual(8, args.ppo_epochs_per_rollout)


class RecipeFromStateFinetuneTest(unittest.TestCase):
    def test_recipe_from_state_carries_training_mode(self):
        recipe = nfn_impl.recipe_from_state({
            "base_model": "llama",
            "topology": "dense",
            "router_mode": "none",
            "training_mode": "sft",
            "adapter_type": "lora",
        })
        self.assertEqual("sft", recipe.training_mode)
        self.assertEqual("lora", recipe.adapter_type)

    def test_recipe_from_state_defaults_pretrain_and_none(self):
        recipe = nfn_impl.recipe_from_state({"base_model": "llama"})
        self.assertEqual("pretrain", recipe.training_mode)
        self.assertEqual("none", recipe.adapter_type)

    def test_recipe_from_state_falls_back_to_pretrain_for_invalid_mode(self):
        recipe = nfn_impl.recipe_from_state({"training_mode": "nonsense"})
        self.assertEqual("pretrain", recipe.training_mode)

    def test_recipe_from_state_falls_back_to_none_for_invalid_adapter(self):
        recipe = nfn_impl.recipe_from_state({"adapter_type": "frobnicate"})
        self.assertEqual("none", recipe.adapter_type)

    def test_composed_recipe_exposes_training_mode_as_default_field(self):
        recipe = nfn_impl.ComposedRecipe(
            base_model="llama",
            topology="dense",
            router_mode="none",
            use_jepa=False,
            runtime="default",
        )
        self.assertEqual("pretrain", recipe.training_mode)
        self.assertEqual("none", recipe.adapter_type)


if __name__ == "__main__":
    unittest.main()
