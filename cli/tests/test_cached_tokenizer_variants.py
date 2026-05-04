from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys
from unittest.mock import patch
import unittest


ROOT = Path(__file__).resolve().parents[1]
NEURALFN_ROOT = ROOT.parent
SCRIPTS_DIR = ROOT / "scripts"
for candidate in (ROOT, NEURALFN_ROOT, SCRIPTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import neuralfn.semantic as semantic_module
import train_jepa_semantic


TRAIN_SCRIPT_CASES = [
    ("train_jepa_semantic", 1),
    ("train_gpt2", 1),
    ("train_nanogpt", 1),
    ("train_llama_fast", 1),
    ("train_llama_megakernel", 1),
    ("train_mixllama_fast", 1),
    ("train_semantic_router_moe", 1),
]


class CachedTokenizerVariantTest(unittest.TestCase):
    def load_module(self, module_name: str):
        return importlib.import_module(module_name)

    def parse_args(self, module, cli_args: list[str]):
        with patch.dict(os.environ, {"OUTPUT": ""}, clear=False):
            parser = module.build_parser()
            args = parser.parse_args(cli_args)
        if hasattr(module, "apply_tinystories_dataset_defaults"):
            module.apply_tinystories_dataset_defaults(args)
        module.resolve_dataset_selector_args(args)
        if hasattr(module, "resolve_mode_defaults"):
            module.resolve_mode_defaults(args)
        return args

    def test_golf_dataset_variant_updates_alias_for_all_train_scripts(self) -> None:
        for module_name, train_shards in TRAIN_SCRIPT_CASES:
            module = self.load_module(module_name)
            with self.subTest(script=module_name):
                args = self.parse_args(module, ["--dataset", "golf1", "--dataset-variant", "sp4096"])
                self.assertEqual("willdepueoai/parameter-golf", args.dataset_hf_path)
                self.assertEqual("sp4096", args.dataset_variant)
                self.assertEqual(train_shards, args.dataset_train_shards)
                self.assertEqual(
                    train_jepa_semantic.parameter_golf_dataset_alias(train_shards=train_shards, variant="sp4096"),
                    args.dataset_alias,
                )

    def test_canonical_tokenizer_flag_updates_golf_alias_for_all_train_scripts(self) -> None:
        for module_name, train_shards in TRAIN_SCRIPT_CASES:
            module = self.load_module(module_name)
            with self.subTest(script=module_name):
                args = self.parse_args(module, ["--dataset", "golf1", "--tokenizer", "sp8192"])
                self.assertEqual("sp8192", args.dataset_variant)
                self.assertEqual("sp8192", args.tokenizer)
                self.assertEqual(
                    train_jepa_semantic.parameter_golf_dataset_alias(train_shards=train_shards, variant="sp8192"),
                    args.dataset_alias,
                )

    def test_semantic_vocab_ref_tracks_supported_tokenizer_variants(self) -> None:
        self.assertEqual("vocab_86d_cl100k.json", semantic_module.semantic_vocab_ref_for_tokenizer("cl100k_base"))
        self.assertEqual("vocab_86d_o200k.json", semantic_module.semantic_vocab_ref_for_tokenizer("o200k_base"))
        self.assertEqual("vocab_86d_sp1024.json", semantic_module.semantic_vocab_ref_for_tokenizer("sp1024"))
        self.assertEqual("vocab_86d_sp2048.json", semantic_module.semantic_vocab_ref_for_tokenizer("sp2048"))
        self.assertEqual("vocab_86d_sp4096.json", semantic_module.semantic_vocab_ref_for_tokenizer("sp4096"))
        self.assertEqual("vocab_86d_sp8192.json", semantic_module.semantic_vocab_ref_for_tokenizer("sp8192"))

    def test_apply_cached_tokenizer_vocab_policy_sets_default_vocab_size(self) -> None:
        args = argparse.Namespace(vocab_size=1024)
        dataset_meta = {
            "data_format": "uint16_shards",
            "tokenizer_name": "fineweb_4096_bpe",
            "tokenizer_files": ["fineweb_4096_bpe.model"],
        }

        with patch.object(
            train_jepa_semantic,
            "validate_cached_tokenizer_contract",
            return_value={"tokenizer_vocab_size": 4096},
        ):
            resolved_meta = train_jepa_semantic.apply_cached_tokenizer_vocab_policy(
                args,
                dataset_name="willdepueoai__parameter-golf__sp4096__train1",
                dataset_path=Path("/tmp/fake"),
                dataset_meta=dataset_meta,
                default_vocab_size=1024,
            )

        self.assertEqual(4096, args.vocab_size)
        self.assertEqual(4096, resolved_meta["tokenizer_vocab_size"])

    def test_apply_cached_tokenizer_vocab_policy_rejects_conflicting_vocab_size(self) -> None:
        args = argparse.Namespace(vocab_size=2048)
        dataset_meta = {
            "data_format": "uint16_shards",
            "tokenizer_name": "fineweb_4096_bpe",
            "tokenizer_files": ["fineweb_4096_bpe.model"],
        }

        with patch.object(
            train_jepa_semantic,
            "validate_cached_tokenizer_contract",
            return_value={"tokenizer_vocab_size": 4096},
        ):
            with self.assertRaisesRegex(ValueError, "requires vocab_size=4096"):
                train_jepa_semantic.apply_cached_tokenizer_vocab_policy(
                    args,
                    dataset_name="willdepueoai__parameter-golf__sp4096__train1",
                    dataset_path=Path("/tmp/fake"),
                    dataset_meta=dataset_meta,
                    default_vocab_size=1024,
                )

    def test_train_gpt2_main_applies_cached_tokenizer_vocab_before_build_graph(self) -> None:
        module = self.load_module("train_gpt2")
        dataset_alias = train_jepa_semantic.parameter_golf_dataset_alias(train_shards=1, variant="sp4096")
        sentinel = RuntimeError("stop after build_graph")

        def fake_build_graph(args, dataset_name: str):
            self.assertEqual(dataset_alias, dataset_name)
            self.assertEqual(4096, args.vocab_size)
            raise sentinel

        with patch.object(module, "configure_console_logging"), patch("torch.cuda.is_available", return_value=True):
            with patch.object(
                module,
                "resolve_or_download_dataset",
                return_value=(
                    dataset_alias,
                    Path("/tmp/fake"),
                    {
                        "data_format": "uint16_shards",
                        "tokenizer_name": "fineweb_4096_bpe",
                        "tokenizer_files": ["fineweb_4096_bpe.model"],
                    },
                ),
            ), patch.object(
                train_jepa_semantic,
                "validate_cached_tokenizer_contract",
                return_value={"tokenizer_vocab_size": 4096},
            ), patch.object(
                module,
                "estimate_text_schedule",
                return_value={"drop_last": False, "respect_epoch_boundaries": True},
            ), patch.object(
                module,
                "resolve_effective_training_schedule",
                return_value=({"drop_last": False, "respect_epoch_boundaries": True}, 1, 1, 1, 1.0),
            ), patch.object(
                module,
                "build_trainer_config",
                return_value=object(),
            ), patch.object(module, "build_graph", side_effect=fake_build_graph), patch.object(
                sys,
                "argv",
                ["train_gpt2.py", "--dataset", "golf1", "--dataset-variant", "sp4096"],
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after build_graph"):
                    module.main()


if __name__ == "__main__":
    unittest.main()
