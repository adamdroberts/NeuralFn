from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
NEURALFN_ROOT = ROOT.parent
SCRIPTS_DIR = ROOT / "scripts"
if str(NEURALFN_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURALFN_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from neuralfn.torch_backend import resolve_torch_train_drop_last


SCRIPT_CASES = [
    "train_jepa_semantic",
    "train_mixllama_fast",
    "train_llama_fast",
    "train_llama_megakernel",
    "train_gpt2",
    "train_nanogpt",
    "train_semantic_router_moe",
]
OVERNIGHT_SCRIPT_PATH = ROOT / "scripts" / "train_semantic_router_moe-overnight.py"


class _LenOnlyDataset:
    def __init__(self, rows: int) -> None:
        self.rows = int(rows)

    def __len__(self) -> int:
        return self.rows


class TrainDropLastTest(unittest.TestCase):
    def load_module(self, module_name: str):
        return importlib.import_module(module_name)

    def load_path_module(self, module_name: str, module_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def parse_args(self, module, cli_args: list[str]):
        with patch.dict(os.environ, {"OUTPUT": ""}, clear=False):
            parser = module.build_parser()
            args = parser.parse_args(cli_args)
        module.resolve_mode_defaults(args)
        return args

    def test_backend_helper_auto_enables_drop_last_only_for_cuda_megakernel(self) -> None:
        self.assertTrue(
            resolve_torch_train_drop_last(
                drop_last=None,
                template_runtime="megakernel",
                device="cuda",
                dataset_rows=3270,
                batch_size=8,
            )
        )
        self.assertFalse(
            resolve_torch_train_drop_last(
                drop_last=None,
                template_runtime="compile",
                device="cuda",
                dataset_rows=3270,
                batch_size=8,
            )
        )
        self.assertFalse(
            resolve_torch_train_drop_last(
                drop_last=None,
                template_runtime="megakernel",
                device="cuda",
                dataset_rows=6,
                batch_size=8,
            )
        )

    def test_text_schedule_drops_partial_batch_for_cuda_megakernel(self) -> None:
        module = self.load_module("train_jepa_semantic")
        with patch.object(module, "load_dataset_tensors", return_value=_LenOnlyDataset(3270)):
            derived = module.estimate_text_schedule(
                "dummy_dataset",
                seq_len=128,
                batch_size=8,
                train_batch_tokens=8192,
                template_runtime="megakernel",
                device="cuda",
            )
        self.assertTrue(derived["drop_last"])
        self.assertEqual(derived["train_rows"], 3264)
        self.assertEqual(derived["dropped_train_rows"], 6)
        self.assertEqual(derived["loader_batches"], 408)
        self.assertEqual(derived["steps_per_epoch"], 51)

    def test_text_schedule_all_train_rows_keeps_partial_batch_and_marks_tail_step(self) -> None:
        module = self.load_module("train_jepa_semantic")
        with patch.object(module, "load_dataset_tensors", return_value=_LenOnlyDataset(3270)):
            derived = module.estimate_text_schedule(
                "dummy_dataset",
                seq_len=128,
                batch_size=8,
                train_batch_tokens=8192,
                template_runtime="megakernel",
                device="cuda",
                all_train_rows=True,
            )
        self.assertFalse(derived["drop_last"])
        self.assertEqual(derived["train_rows"], 3270)
        self.assertEqual(derived["dropped_train_rows"], 0)
        self.assertEqual(derived["loader_batches"], 409)
        self.assertEqual(derived["steps_per_epoch"], 52)
        self.assertTrue(derived["respect_epoch_boundaries"])
        self.assertTrue(derived["has_short_epoch_tail_step"])
        self.assertEqual(derived["epoch_tail_grad_accum_steps"], 1)

    def test_semantic_schedule_keeps_partial_batch_out_of_megakernel_epochs(self) -> None:
        module = self.load_module("train_jepa_semantic")
        with patch.object(module, "load_dataset_tensors", return_value=_LenOnlyDataset(3270)):
            with patch.object(module, "load_semantic_tokens", return_value=torch.zeros((100000, 9), dtype=torch.long)):
                derived = module.estimate_schedule(
                    "dummy_dataset",
                    seq_len=128,
                    batch_size=8,
                    train_batch_tokens=8192,
                    top_k=2,
                    template_runtime="megakernel",
                    device="cuda",
                )
        self.assertTrue(derived["drop_last"])
        self.assertEqual(derived["text_rows"], 3270)
        self.assertEqual(derived["train_rows"], 3264)
        self.assertEqual(derived["dropped_train_rows"], 6)
        self.assertEqual(derived["loader_batches"], 408)
        self.assertEqual(derived["steps_per_epoch"], 51)

    def test_semantic_schedule_all_train_rows_keeps_partial_batch_and_marks_tail_step(self) -> None:
        module = self.load_module("train_jepa_semantic")
        with patch.object(module, "load_dataset_tensors", return_value=_LenOnlyDataset(3270)):
            with patch.object(module, "load_semantic_tokens", return_value=torch.zeros((100000, 9), dtype=torch.long)):
                derived = module.estimate_schedule(
                    "dummy_dataset",
                    seq_len=128,
                    batch_size=8,
                    train_batch_tokens=8192,
                    top_k=2,
                    template_runtime="megakernel",
                    device="cuda",
                    all_train_rows=True,
                )
        self.assertFalse(derived["drop_last"])
        self.assertEqual(derived["train_rows"], 3270)
        self.assertEqual(derived["dropped_train_rows"], 0)
        self.assertEqual(derived["loader_batches"], 409)
        self.assertEqual(derived["steps_per_epoch"], 52)
        self.assertTrue(derived["respect_epoch_boundaries"])
        self.assertTrue(derived["has_short_epoch_tail_step"])
        self.assertEqual(derived["epoch_tail_grad_accum_steps"], 1)

    def test_build_trainer_config_threads_drop_last_for_all_training_scripts(self) -> None:
        for module_name in SCRIPT_CASES:
            module = self.load_module(module_name)
            cli_args = ["--megakernel"] if module_name != "train_llama_megakernel" else []
            args = self.parse_args(module, cli_args)
            with self.subTest(script=module_name):
                trainer_cfg = module.build_trainer_config(args, resolved_epochs=3, drop_last=True)
                self.assertTrue(trainer_cfg.drop_last)

    def test_build_trainer_config_enables_epoch_boundaries_for_all_train_rows(self) -> None:
        for module_name in SCRIPT_CASES:
            module = self.load_module(module_name)
            cli_args = ["--all-train-rows"]
            if module_name != "train_llama_megakernel":
                cli_args.insert(0, "--megakernel")
            args = self.parse_args(module, cli_args)
            with self.subTest(script=module_name):
                trainer_cfg = module.build_trainer_config(args, resolved_epochs=3)
                self.assertFalse(trainer_cfg.drop_last)
                self.assertTrue(trainer_cfg.respect_epoch_boundaries)

    def test_overnight_parser_accepts_all_train_rows(self) -> None:
        module = self.load_path_module("train_semantic_router_moe_overnight_test", OVERNIGHT_SCRIPT_PATH)
        args = self.parse_args(module, ["--megakernel", "--all-train-rows"])
        trainer_cfg = module.build_trainer_config(args, resolved_epochs=3)
        self.assertTrue(args.all_train_rows)
        self.assertFalse(trainer_cfg.drop_last)
        self.assertTrue(trainer_cfg.respect_epoch_boundaries)


if __name__ == "__main__":
    unittest.main()
