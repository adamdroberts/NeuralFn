from __future__ import annotations

import importlib
import json
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

import server.dataset_manager as dataset_manager_module


TRAIN_SCRIPT_CASES = [
    "train_jepa_semantic",
    "train_gpt2",
]


class TrainPretrainingFileFlagTest(unittest.TestCase):
    def load_module(self, module_name: str):
        return importlib.import_module(module_name)

    def parse_args(self, module, cli_args: list[str]):
        with patch.dict(os.environ, {"OUTPUT": ""}, clear=False):
            parser = module.build_parser()
            args = parser.parse_args(cli_args)
        module.apply_tinystories_dataset_defaults(args)
        module.resolve_dataset_selector_args(args)
        module.resolve_pretraining_file_dataset(args)
        if hasattr(module, "resolve_mode_defaults"):
            module.resolve_mode_defaults(args)
        return args

    def test_parser_accepts_pretraining_file_for_all_train_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = Path(tmpdir) / "corpus.txt"
            corpus.write_text("tiny corpus\n" * 64, encoding="utf-8")
            resolved_corpus = str(corpus.resolve())
            for module_name in TRAIN_SCRIPT_CASES:
                module = self.load_module(module_name)
                with self.subTest(script=module_name):
                    args = self.parse_args(module, ["--pretraining-file", str(corpus)])
                    self.assertEqual(resolved_corpus, args.pretraining_file)
                    if module_name == "train_gpt2":
                        self.assertFalse(args.download_if_missing)
                    else:
                        self.assertTrue(args.download_if_missing)
                    adapter_dir = Path(args.dataset_alias)
                    self.assertTrue(adapter_dir.is_dir())
                    self.assertTrue((adapter_dir / "meta.json").exists())
                    self.assertTrue((adapter_dir / "data.txt").is_symlink())
                    self.assertEqual(resolved_corpus, os.readlink(adapter_dir / "data.txt"))

    def test_pretraining_file_uses_raw_text_defaults_for_all_train_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = Path(tmpdir) / "corpus.txt"
            corpus.write_text("hello world\n" * 128, encoding="utf-8")
            for module_name in TRAIN_SCRIPT_CASES:
                module = self.load_module(module_name)
                with self.subTest(script=module_name):
                    args = self.parse_args(module, ["--pretraining-file", str(corpus)])
                    self.assertTrue(args.raw_text_selected)
                    self.assertEqual(args.raw_text_encoding_vocab_size, args.vocab_size)

    def test_pretraining_file_conflicts_with_dataset_shortcuts_for_all_train_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = Path(tmpdir) / "corpus.txt"
            corpus.write_text("tiny corpus\n" * 16, encoding="utf-8")
            for module_name in TRAIN_SCRIPT_CASES:
                module = self.load_module(module_name)
                with self.subTest(script=module_name):
                    with patch.dict(os.environ, {"OUTPUT": ""}, clear=False):
                        parser = module.build_parser()
                        args = parser.parse_args(["--dataset", "golf1", "--pretraining-file", str(corpus)])
                    module.apply_tinystories_dataset_defaults(args)
                    module.resolve_dataset_selector_args(args)
                    with self.assertRaisesRegex(ValueError, "--dataset"):
                        module.resolve_pretraining_file_dataset(args)

    def test_pretraining_file_requires_existing_txt_file(self) -> None:
        module = self.load_module("train_jepa_semantic")
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.txt"
            with patch.dict(os.environ, {"OUTPUT": ""}, clear=False):
                parser = module.build_parser()
                args = parser.parse_args(["--pretraining-file", str(missing)])
            module.apply_tinystories_dataset_defaults(args)
            module.resolve_dataset_selector_args(args)
            with self.assertRaises(FileNotFoundError):
                module.resolve_pretraining_file_dataset(args)

            non_txt = Path(tmpdir) / "corpus.md"
            non_txt.write_text("not plain text", encoding="utf-8")
            with patch.dict(os.environ, {"OUTPUT": ""}, clear=False):
                args = parser.parse_args(["--pretraining-file", str(non_txt)])
            module.apply_tinystories_dataset_defaults(args)
            module.resolve_dataset_selector_args(args)
            with self.assertRaisesRegex(ValueError, r"\.txt"):
                module.resolve_pretraining_file_dataset(args)

    def test_pretraining_file_adapter_path_resolves_as_local_dataset(self) -> None:
        module = self.load_module("train_jepa_semantic")
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = Path(tmpdir) / "corpus.txt"
            corpus.write_text("hello world\n" * 64, encoding="utf-8")
            args = self.parse_args(module, ["--pretraining-file", str(corpus)])

            dataset_name, dataset_path, dataset_meta = module.resolve_or_download_dataset(
                args.dataset_alias,
                download_if_missing=False,
                raw_text_encoding_name=str(getattr(args, "raw_text_encoding_name", "gpt2")),
            )

            self.assertEqual(str(Path(args.dataset_alias).resolve()), dataset_name)
            self.assertEqual(Path(args.dataset_alias).resolve(), dataset_path)
            self.assertEqual("local_pretraining_file", dataset_meta.get("source"))
            self.assertEqual(str(corpus.resolve()), dataset_meta.get("pretraining_file"))

    def test_pretraining_file_explicit_sentencepiece_requires_shared_model(self) -> None:
        module = self.load_module("train_jepa_semantic")
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = Path(tmpdir) / "corpus.txt"
            corpus.write_text("hello world\n" * 64, encoding="utf-8")
            with patch.dict(os.environ, {"OUTPUT": ""}, clear=False):
                parser = module.build_parser()
                args = parser.parse_args(
                    ["--pretraining-file", str(corpus), "--tokenizer", "sp8192", "--no-download-if-missing"]
                )
            module.apply_tinystories_dataset_defaults(args)
            module.resolve_dataset_selector_args(args)
            module.resolve_pretraining_file_dataset(args)
            with patch.object(
                module,
                "resolve_sentencepiece_model_path",
                side_effect=FileNotFoundError("Raw-text tokenizer 'sp8192' requires a shared sentencepiece model."),
            ):
                with self.assertRaisesRegex(FileNotFoundError, "sp8192"):
                    module.resolve_mode_defaults(args)

    def test_pretraining_file_sentencepiece_downloads_shared_assets_when_missing(self) -> None:
        module = self.load_module("train_jepa_semantic")
        downloaded_specs: list[tuple[str, str, str]] = []

        def fake_download(repo: str, remote_path: str, target_path: Path, *, repo_type: str = "dataset") -> None:
            downloaded_specs.append((repo, repo_type, remote_path))
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if remote_path == "tokenizers/fineweb_8192_bpe.model":
                target_path.write_text("fake sentencepiece model", encoding="utf-8")
                return
            if remote_path == "tokenizers/fineweb_8192_bpe.vocab":
                target_path.write_text("fake sentencepiece vocab", encoding="utf-8")
                return
            raise AssertionError(f"unexpected download path: {remote_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = Path(tmpdir) / "corpus.txt"
            corpus.write_text("hello world\n" * 64, encoding="utf-8")
            tokenizers_dir = Path(tmpdir) / "tokenizers"
            with patch.dict(os.environ, {"OUTPUT": ""}, clear=False):
                parser = module.build_parser()
                args = parser.parse_args(["--pretraining-file", str(corpus), "--tokenizer", "sp8192"])
            module.apply_tinystories_dataset_defaults(args)
            module.resolve_dataset_selector_args(args)
            module.resolve_pretraining_file_dataset(args)
            with patch.object(dataset_manager_module, "SENTENCEPIECE_TOKENIZERS_DIR", tokenizers_dir), patch.object(
                module,
                "_download_hf_file",
                side_effect=fake_download,
            ):
                module.resolve_mode_defaults(args)

            self.assertEqual("sp8192", args.raw_text_encoding_name)
            self.assertTrue((tokenizers_dir / "fineweb_8192_bpe.model").exists())
            self.assertTrue((tokenizers_dir / "fineweb_8192_bpe.vocab").exists())
            self.assertEqual(
                [
                    ("sproos/parameter-golf-tokenizers", "model", "tokenizers/sp8192.model"),
                    ("sproos/parameter-golf-tokenizers", "model", "tokenizers/fineweb_8192_bpe.model"),
                    ("sproos/parameter-golf-tokenizers", "model", "tokenizers/sp8192.vocab"),
                    ("sproos/parameter-golf-tokenizers", "model", "tokenizers/fineweb_8192_bpe.vocab"),
                ],
                downloaded_specs,
            )

    def test_sentencepiece_tokenizer_is_promoted_from_dataset_cache_before_download(self) -> None:
        module = self.load_module("train_jepa_semantic")

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "cached_dataset"
            tokenizer_dir = dataset_dir / "tokenizers"
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "meta.json").write_text(json.dumps({"tokenizer_files": ["fineweb_8192_bpe.model", "fineweb_8192_bpe.vocab"]}), encoding="utf-8")
            (tokenizer_dir / "fineweb_8192_bpe.model").write_text("cached model", encoding="utf-8")
            (tokenizer_dir / "fineweb_8192_bpe.vocab").write_text("cached vocab", encoding="utf-8")
            shared_tokenizers_dir = Path(tmpdir) / "shared-tokenizers"

            with patch.object(dataset_manager_module, "SENTENCEPIECE_TOKENIZERS_DIR", shared_tokenizers_dir), patch.object(
                module,
                "_download_hf_file",
                side_effect=AssertionError("tokenizer download should not run when dataset cache already has the artifacts"),
            ):
                resolved = module.validate_raw_text_tokenizer_availability(
                    "sp8192",
                    download_if_missing=True,
                    dataset_path=dataset_dir,
                    dataset_meta={"tokenizer_files": ["fineweb_8192_bpe.model", "fineweb_8192_bpe.vocab"]},
                )

            self.assertEqual("sp8192", resolved)
            self.assertEqual("cached model", (shared_tokenizers_dir / "fineweb_8192_bpe.model").read_text(encoding="utf-8"))
            self.assertEqual("cached vocab", (shared_tokenizers_dir / "fineweb_8192_bpe.vocab").read_text(encoding="utf-8"))

    def test_pretraining_file_refresh_errors_propagate_without_alias_rewrite(self) -> None:
        module = self.load_module("train_jepa_semantic")
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = Path(tmpdir) / "corpus.txt"
            corpus.write_text("hello world\n" * 64, encoding="utf-8")
            args = self.parse_args(module, ["--pretraining-file", str(corpus)])

            with patch.object(
                module,
                "validate_raw_text_tokenizer_availability",
                return_value="sp8192",
            ), patch.object(
                module,
                "refresh_raw_text_dataset_metadata",
                side_effect=FileNotFoundError("missing sentencepiece model"),
            ):
                with self.assertRaisesRegex(FileNotFoundError, "missing sentencepiece model"):
                    module.resolve_or_download_dataset(
                        args.dataset_alias,
                        download_if_missing=False,
                        raw_text_encoding_name="sp8192",
                    )


if __name__ == "__main__":
    unittest.main()
