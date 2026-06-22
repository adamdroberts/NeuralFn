from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
NEURALFN_ROOT = ROOT.parent
if str(NEURALFN_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURALFN_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from server import dataset_manager as dm

TINYSTORIES_SCRIPT_CASES = [
    "train_jepa_semantic",
    "train_gpt2",
    "train_semantic_router_moe",
]
GPT2_FAMILY_SCRIPT_CASES = [
    "train_gpt2",
]
NON_GPT2_TINYSTORIES_SCRIPT_CASES = [
    module_name for module_name in TINYSTORIES_SCRIPT_CASES if module_name not in GPT2_FAMILY_SCRIPT_CASES
]


class TrainTinyStoriesFlagTest(unittest.TestCase):
    def load_module(self, module_name: str):
        return importlib.import_module(module_name)

    def parse_args(self, module, cli_args: list[str]):
        with patch.dict(os.environ, {"OUTPUT": ""}, clear=False):
            parser = module.build_parser()
            args = parser.parse_args(cli_args)
        module.apply_tinystories_dataset_defaults(args)
        module.resolve_dataset_selector_args(args)
        if hasattr(module, "resolve_mode_defaults"):
            module.resolve_mode_defaults(args)
        return args

    def test_tinystories_flag_populates_raw_hf_contract_for_all_train_scripts(self) -> None:
        shared_module = self.load_module("train_jepa_semantic")
        for module_name in TINYSTORIES_SCRIPT_CASES:
            module = self.load_module(module_name)
            with self.subTest(script=module_name):
                args = self.parse_args(module, ["--tinystories"])
                self.assertEqual(args.dataset_alias, shared_module.TINYSTORIES_ALIAS)
                self.assertEqual(args.dataset_hf_path, shared_module.TINYSTORIES_HF_PATH)
                self.assertEqual(args.dataset_train_file, shared_module.TINYSTORIES_TRAIN_FILE)
                self.assertEqual(args.dataset_val_file, shared_module.TINYSTORIES_VAL_FILE)

    def test_dataset_tinystories_shortcut_populates_raw_hf_contract_for_all_train_scripts(self) -> None:
        shared_module = self.load_module("train_jepa_semantic")
        for module_name in TINYSTORIES_SCRIPT_CASES:
            module = self.load_module(module_name)
            with self.subTest(script=module_name):
                args = self.parse_args(module, ["--dataset", "tinystories"])
                self.assertEqual(args.dataset_alias, shared_module.TINYSTORIES_ALIAS)
                self.assertEqual(args.dataset_hf_path, shared_module.TINYSTORIES_HF_PATH)
                self.assertEqual(args.dataset_train_file, shared_module.TINYSTORIES_TRAIN_FILE)
                self.assertEqual(args.dataset_val_file, shared_module.TINYSTORIES_VAL_FILE)

    def test_tinystories_conflicts_fail_fast(self) -> None:
        for module_name in TINYSTORIES_SCRIPT_CASES:
            module = self.load_module(module_name)
            with self.subTest(script=module_name):
                parser = module.build_parser()
                args = parser.parse_args(["--tinystories", "--dataset-hf-path", "custom/repo"])
                with self.assertRaisesRegex(ValueError, "--dataset-hf-path"):
                    module.apply_tinystories_dataset_defaults(args)

    def test_load_val_token_dataset_prefers_explicit_validation_text(self) -> None:
        module = self.load_module("train_jepa_semantic")
        seq_len = 8
        val_text = "Tiny validation sample. " * 12
        expected_tokens = dm.encode_raw_text(val_text, encoding_name="o200k_base")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_dir = Path(tmpdir) / "tiny_alias"
            ds_dir.mkdir(parents=True, exist_ok=True)
            (ds_dir / "data.txt").write_text("train text " * 40, encoding="utf-8")
            (ds_dir / "val.txt").write_text(val_text, encoding="utf-8")

            dataset = module.load_val_token_dataset(ds_dir, seq_len=seq_len, encoding_name="o200k_base")
            x, y = dataset[0]

        self.assertEqual(x.tolist(), expected_tokens[:seq_len])
        self.assertEqual(y.tolist(), expected_tokens[1 : seq_len + 1])

    def test_load_val_token_dataset_falls_back_to_train_holdout_without_val_file(self) -> None:
        module = self.load_module("train_jepa_semantic")
        seq_len = 8
        train_text = "training corpus only " * 80

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ds_dir = root / "tiny_alias"
            ds_dir.mkdir(parents=True, exist_ok=True)
            (ds_dir / "data.txt").write_text(train_text, encoding="utf-8")

            with patch.object(dm, "DATASETS_DIR", root):
                dataset = module.load_val_token_dataset(ds_dir, seq_len=seq_len, encoding_name="o200k_base")
                full_tokens = module._load_tokens_for("tiny_alias", None, encoding_name="o200k_base")

        holdout_size = max(seq_len + 1, int(len(full_tokens) * 0.1))
        expected = full_tokens[-holdout_size:]
        x, y = dataset[0]

        self.assertEqual(x.tolist(), expected[:seq_len])
        self.assertEqual(y.tolist(), expected[1 : seq_len + 1])

    def test_tinystories_uses_o200k_for_all_train_scripts(self) -> None:
        expected_vocab = dm.raw_text_encoding_vocab_size("o200k_base")
        for module_name in TINYSTORIES_SCRIPT_CASES:
            module = self.load_module(module_name)
            with self.subTest(script=module_name):
                args = self.parse_args(module, ["--dataset", "tinystories"])
                self.assertTrue(args.raw_text_selected)
                self.assertEqual("o200k_base", args.raw_text_encoding_name)
                self.assertEqual(expected_vocab, args.raw_text_encoding_vocab_size)
                self.assertEqual(expected_vocab, args.vocab_size)

    def test_cl100k_switches_all_train_scripts(self) -> None:
        expected_vocab = dm.raw_text_encoding_vocab_size("cl100k_base")
        for module_name in TINYSTORIES_SCRIPT_CASES:
            module = self.load_module(module_name)
            with self.subTest(script=module_name):
                args = self.parse_args(module, ["--dataset", "tinystories", "--cl100k"])
                self.assertTrue(args.raw_text_selected)
                self.assertEqual("cl100k_base", args.raw_text_encoding_name)
                self.assertEqual(expected_vocab, args.raw_text_encoding_vocab_size)
                self.assertEqual(expected_vocab, args.vocab_size)

    def test_o200k_switches_all_train_scripts(self) -> None:
        expected_vocab = dm.raw_text_encoding_vocab_size("o200k_base")
        for module_name in TINYSTORIES_SCRIPT_CASES:
            module = self.load_module(module_name)
            with self.subTest(script=module_name):
                args = self.parse_args(module, ["--dataset", "tinystories", "--o200k"])
                self.assertTrue(args.raw_text_selected)
                self.assertEqual("o200k_base", args.raw_text_encoding_name)
                self.assertEqual(expected_vocab, args.raw_text_encoding_vocab_size)
                self.assertEqual(expected_vocab, args.vocab_size)

    def test_toggpt2_switches_all_train_scripts(self) -> None:
        expected_vocab = dm.raw_text_encoding_vocab_size("gpt2")
        for module_name in TINYSTORIES_SCRIPT_CASES:
            module = self.load_module(module_name)
            with self.subTest(script=module_name):
                args = self.parse_args(module, ["--dataset", "tinystories", "--tokgpt2"])
                self.assertTrue(args.raw_text_selected)
                self.assertEqual("gpt2", args.raw_text_encoding_name)
                self.assertEqual(expected_vocab, args.raw_text_encoding_vocab_size)
                self.assertEqual(expected_vocab, args.vocab_size)

    def test_canonical_tokenizer_flag_accepts_sentencepiece_variants(self) -> None:
        expected_vocab = dm.raw_text_encoding_vocab_size("sp1024")
        for module_name in TINYSTORIES_SCRIPT_CASES:
            module = self.load_module(module_name)
            with self.subTest(script=module_name):
                args = self.parse_args(module, ["--dataset", "tinystories", "--tokenizer", "sp1024"])
                self.assertTrue(args.raw_text_selected)
                self.assertEqual("sp1024", args.raw_text_encoding_name)
                self.assertEqual(expected_vocab, args.raw_text_encoding_vocab_size)
                self.assertEqual(expected_vocab, args.vocab_size)

    def test_conflicting_raw_text_vocab_fails_fast(self) -> None:
        for module_name in ("train_semantic_router_moe", "train_gpt2"):
            module = self.load_module(module_name)
            with self.subTest(script=module_name):
                parser = module.build_parser()
                args = parser.parse_args(["--dataset", "tinystories", "--vocab-size", "2048"])
                module.resolve_dataset_selector_args(args)
                with self.assertRaisesRegex(ValueError, "requires vocab_size"):
                    module.resolve_mode_defaults(args)

    def test_tokenizer_override_flags_are_mutually_exclusive(self) -> None:
        for module_name in TINYSTORIES_SCRIPT_CASES:
            module = self.load_module(module_name)
            with self.subTest(script=module_name):
                parser = module.build_parser()
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--tokenizer", "sp1024", "--cl100k"])

    def test_raw_text_dataset_download_validates_sentencepiece_before_download(self) -> None:
        module = self.load_module("train_jepa_semantic")
        events: list[str] = []

        def fake_validate(*_args, **_kwargs):
            events.append("validate")
            return "sp8192"

        def fake_download(*_args, **_kwargs):
            events.append("download")
            raise RuntimeError("stop after download invocation")

        with patch.object(module, "resolve_existing_dataset", side_effect=FileNotFoundError("missing")), patch.object(
            module,
            "validate_raw_text_tokenizer_availability",
            side_effect=fake_validate,
        ), patch.object(module, "download_hf_dataset", side_effect=fake_download):
            with self.assertRaisesRegex(RuntimeError, "stop after download invocation"):
                module.resolve_or_download_dataset(
                    "tiny_alias",
                    dataset_hf_path=module.TINYSTORIES_HF_PATH,
                    dataset_train_file=module.TINYSTORIES_TRAIN_FILE,
                    dataset_val_file=module.TINYSTORIES_VAL_FILE,
                    raw_text_encoding_name="sp8192",
                    tokenizer_hf_path="sproos/parameter-golf-tokenizers",
                    tokenizer_repo_type="model",
                )

        self.assertEqual(["validate", "download"], events)


if __name__ == "__main__":
    unittest.main()
