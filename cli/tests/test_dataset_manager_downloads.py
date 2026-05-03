from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
NEURALFN_ROOT = ROOT.parent
if str(NEURALFN_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURALFN_ROOT))

import server.dataset_manager as dataset_manager_module


class DatasetManagerDownloadTest(unittest.TestCase):
    def test_download_hf_file_passes_through_repo_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_source = Path(tmpdir) / "cache" / "fineweb_8192_bpe.model"
            cache_source.parent.mkdir(parents=True, exist_ok=True)
            cache_source.write_text("fake tokenizer", encoding="utf-8")
            destination = Path(tmpdir) / "target" / "fineweb_8192_bpe.model"

            with patch("huggingface_hub.hf_hub_download", return_value=str(cache_source)) as mock_download:
                resolved = dataset_manager_module._download_hf_file(
                    "sproos/parameter-golf-tokenizers",
                    "tokenizers/fineweb_8192_bpe.model",
                    destination,
                    repo_type="model",
                )

            self.assertEqual(destination, resolved)
            self.assertTrue(destination.exists())
            self.assertEqual("fake tokenizer", destination.read_text(encoding="utf-8"))
            self.assertEqual("model", mock_download.call_args.kwargs["repo_type"])


if __name__ == "__main__":
    unittest.main()
