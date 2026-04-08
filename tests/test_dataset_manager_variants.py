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


if __name__ == "__main__":
    unittest.main()
