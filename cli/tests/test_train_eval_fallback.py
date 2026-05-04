from __future__ import annotations

import importlib
import io
import math
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
NEURALFN_ROOT = ROOT.parent
for candidate in (NEURALFN_ROOT, SCRIPTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


class TrainEvalFallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = importlib.import_module("train_jepa_semantic")

    def test_pretraining_adapter_uses_training_holdout_when_no_val_file_exists(self) -> None:
        seq_len = 8
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = Path(tmpdir) / "corpus.txt"
            corpus.write_text("adapter training text\n" * 256, encoding="utf-8")
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "data.txt").symlink_to(corpus)

            dataset = self.module.load_val_token_dataset(adapter_dir, seq_len=seq_len, encoding_name="gpt2")

            tokens = self.module.encode_raw_text(corpus.read_text(encoding="utf-8"), encoding_name="gpt2")
            expected_holdout = max(seq_len + 1, math.ceil(len(tokens) * self.module.VAL_HOLDOUT_FRACTION))
            self.assertEqual((expected_holdout - 1) // seq_len, len(dataset))
            sample_x, sample_y = dataset[0]
            self.assertEqual(seq_len, int(sample_x.shape[0]))
            self.assertEqual(seq_len, int(sample_y.shape[0]))

    def test_explicit_val_file_takes_precedence_over_training_holdout(self) -> None:
        seq_len = 8
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "raw_text_dataset"
            dataset_dir.mkdir()
            train_text = "train token stream\n" * 512
            val_text = "validation token stream\n" * 48
            (dataset_dir / "data.txt").write_text(train_text, encoding="utf-8")
            (dataset_dir / "val.txt").write_text(val_text, encoding="utf-8")

            dataset = self.module.load_val_token_dataset(dataset_dir, seq_len=seq_len, encoding_name="gpt2")

            train_tokens = self.module.encode_raw_text(train_text, encoding_name="gpt2")
            val_tokens = self.module.encode_raw_text(val_text, encoding_name="gpt2")
            expected_val_chunks = (len(val_tokens) - 1) // seq_len
            expected_train_holdout = max(seq_len + 1, math.ceil(len(train_tokens) * self.module.VAL_HOLDOUT_FRACTION))
            self.assertNotEqual((expected_train_holdout - 1) // seq_len, expected_val_chunks)
            self.assertEqual(expected_val_chunks, len(dataset))

    def test_cached_token_fallback_uses_twenty_percent_holdout(self) -> None:
        seq_len = 5
        cached_tokens = list(range(100))
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "cached_alias"
            dataset_dir.mkdir()

            with patch.object(self.module, "_load_tokens_for", return_value=cached_tokens) as patched_load_tokens:
                dataset = self.module.load_val_token_dataset(dataset_dir, seq_len=seq_len, encoding_name="gpt2")

            expected_holdout = max(seq_len + 1, math.ceil(len(cached_tokens) * self.module.VAL_HOLDOUT_FRACTION))
            self.assertEqual((expected_holdout - 1) // seq_len, len(dataset))
            self.assertEqual("cached_alias", patched_load_tokens.call_args.args[0])

    def test_safe_evaluate_validation_loss_returns_nan_and_warns(self) -> None:
        stderr = io.StringIO()

        def raise_missing_val() -> float:
            raise FileNotFoundError("missing validation file")

        with redirect_stderr(stderr):
            val_loss = self.module.safe_evaluate_validation_loss(raise_missing_val)

        self.assertTrue(math.isnan(val_loss))
        self.assertIn("Validation skipped: missing validation file", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
