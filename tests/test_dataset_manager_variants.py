import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from server import dataset_manager as dm


class DatasetManagerVariantsTest(unittest.TestCase):
    def _fake_download(self, repo_id: str, relative_path: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative_path == "datasets/manifest.json":
            destination.write_text(
                """
{
  "datasets": [
    {
      "name": "fineweb10B_sp1024",
      "tokenizer_name": "fineweb_1024_bpe",
      "stats": {"files_train": 10, "files_val": 1}
    }
  ],
  "tokenizers": [
    {
      "name": "fineweb_1024_bpe",
      "model_path": "tokenizers/fineweb_1024_bpe.model"
    }
  ]
}
""".strip(),
                encoding="utf-8",
            )
            return destination

        if relative_path.endswith("fineweb_train_000000.bin"):
            np.array([1, 2, 3, 4, 5, 6], dtype=np.uint16).tofile(destination)
            return destination
        if relative_path.endswith("fineweb_train_000001.bin"):
            np.array([7, 8, 9, 10, 11, 12], dtype=np.uint16).tofile(destination)
            return destination
        if relative_path.endswith("fineweb_val_000000.bin"):
            np.array([100, 101, 102], dtype=np.uint16).tofile(destination)
            return destination

        destination.write_text("tokenizer", encoding="utf-8")
        return destination

    def test_downloads_cached_variant_and_loads_train_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(dm, "DATASETS_DIR", Path(tmpdir)):
                with patch("server.dataset_manager._download_hf_file", side_effect=self._fake_download):
                    with patch("server.dataset_manager._tokenizer_vocab_size_from_artifacts", return_value=1024):
                        result = dm.download_hf_dataset(
                            "willdepueoai/parameter-golf",
                            variant="sp1024",
                            train_shards=2,
                        )

                        self.assertEqual("willdepueoai__parameter-golf__sp1024__train2", result["name"])
                        self.assertEqual("sp1024", result["variant"])
                        self.assertEqual(2, result["train_shards"])
                        self.assertEqual(1, result["val_shards"])
                        self.assertEqual("uint16_shards", result["data_format"])

                        inputs, targets = dm.load_dataset_tokens([result["name"]], seq_len=4)
                        self.assertEqual([[1, 2, 3, 4], [5, 6, 7, 8]], inputs)
                        self.assertEqual([[2, 3, 4, 5], [6, 7, 8, 9]], targets)

    def test_rejects_cached_variant_when_shards_exceed_tokenizer_vocab(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(dm, "DATASETS_DIR", Path(tmpdir)):
                with patch("server.dataset_manager._download_hf_file", side_effect=self._fake_download):
                    with patch("server.dataset_manager._tokenizer_vocab_size_from_artifacts", return_value=8):
                        with self.assertRaises(dm.DatasetTokenizerMismatchError) as ctx:
                            dm.download_hf_dataset(
                                "willdepueoai/parameter-golf",
                                variant="sp1024",
                                train_shards=2,
                            )
                        self.assertFalse((Path(tmpdir) / "willdepueoai__parameter-golf__sp1024__train2").exists())

            message = str(ctx.exception)
            self.assertIn("willdepueoai__parameter-golf__sp1024__train2", message)
            self.assertIn("Tokenizer vocab size: 8", message)
            self.assertIn("Observed max token id in cached shards: 102", message)

    def test_downloads_explicit_raw_train_and_val_files(self) -> None:
        downloads: list[tuple[str, str]] = []

        def fake_explicit_download(hf_path: str, filename: str, destination: Path, *, max_rows: int | None) -> int:
            downloads.append((hf_path, filename))
            destination.parent.mkdir(parents=True, exist_ok=True)
            if filename.endswith("-train.txt"):
                destination.write_text("train line 1\ntrain line 2\n", encoding="utf-8")
                return 2
            if filename.endswith("-valid.txt"):
                destination.write_text("valid line 1\nvalid line 2\n", encoding="utf-8")
                return 2
            raise AssertionError(f"unexpected filename {filename}")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(dm, "DATASETS_DIR", Path(tmpdir)):
                with patch("server.dataset_manager._download_explicit_raw_hf_text_file", side_effect=fake_explicit_download):
                    with patch("server.dataset_manager._download_raw_hf_text", side_effect=AssertionError("fallback should not run")):
                        result = dm.download_hf_dataset(
                            "roneneldan/TinyStories",
                            alias="roneneldan__TinyStories__TinyStoriesV2-GPT4",
                            train_file="TinyStoriesV2-GPT4-train.txt",
                            val_file="TinyStoriesV2-GPT4-valid.txt",
                            encoding_name="o200k_base",
                        )
                        ds_dir = Path(tmpdir) / "roneneldan__TinyStories__TinyStoriesV2-GPT4"
                        meta = json.loads((ds_dir / "meta.json").read_text(encoding="utf-8"))

                        self.assertEqual("roneneldan__TinyStories__TinyStoriesV2-GPT4", result["name"])
                        self.assertEqual(
                            [
                                ("roneneldan/TinyStories", "TinyStoriesV2-GPT4-train.txt"),
                                ("roneneldan/TinyStories", "TinyStoriesV2-GPT4-valid.txt"),
                            ],
                            downloads,
                        )
                        self.assertEqual("TinyStoriesV2-GPT4-train.txt", meta["train_file"])
                        self.assertEqual("TinyStoriesV2-GPT4-valid.txt", meta["val_file"])
                        self.assertEqual(2, meta["num_rows"])
                        self.assertEqual(2, meta["val_rows"])
                        self.assertEqual("o200k_base", meta["tokenizer_encoding"])
                        self.assertEqual(dm.raw_text_encoding_vocab_size("o200k_base"), meta["tokenizer_vocab_size"])
                        self.assertTrue((ds_dir / "data.txt").exists())
                        self.assertTrue((ds_dir / "val.txt").exists())

    def test_load_dataset_tensors_preserves_large_o200k_token_ids(self) -> None:
        text = "<|endoftext|> TinyStories"
        expected_tokens = dm.encode_raw_text(text, encoding_name="o200k_base")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_dir = Path(tmpdir) / "tiny_o200k"
            ds_dir.mkdir(parents=True, exist_ok=True)
            (ds_dir / "data.txt").write_text(text, encoding="utf-8")
            (ds_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "source": "huggingface",
                        "hf_path": "roneneldan/TinyStories",
                        "tokenizer_encoding": "o200k_base",
                        "tokenizer_vocab_size": dm.raw_text_encoding_vocab_size("o200k_base"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            with patch.object(dm, "DATASETS_DIR", Path(tmpdir)):
                dataset = dm.load_dataset_tensors(["tiny_o200k"], seq_len=1, encoding_name="o200k_base")
                x, y = dataset[0]

        self.assertEqual(expected_tokens[0], 199999)
        self.assertEqual(x.tolist(), expected_tokens[:1])
        self.assertEqual(y.tolist(), expected_tokens[1:2])


if __name__ == "__main__":
    unittest.main()
