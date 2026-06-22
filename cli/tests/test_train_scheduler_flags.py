from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
NEURALFN_ROOT = ROOT.parent
SCRIPTS_DIR = ROOT / "scripts"
if str(NEURALFN_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURALFN_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


SCRIPT_CASES = [
    "train_jepa_semantic",
    "train_llama_fast",
    "train_llama_megakernel",
    "train_gpt2",
    "train_semantic_router_moe",
]
OVERNIGHT_SCRIPT_PATH = ROOT / "scripts" / "train_semantic_router_moe-overnight.py"


class TrainSchedulerFlagTest(unittest.TestCase):
    def load_module(self, module_name: str):
        return importlib.import_module(module_name)

    def load_path_module(self, module_name: str, module_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def parse_args(self, module, cli_args: list[str], *, env: dict[str, str] | None = None):
        env_updates = {"OUTPUT": ""}
        if env is not None:
            env_updates.update(env)
        with patch.dict(os.environ, env_updates, clear=False):
            parser = module.build_parser()
            args = parser.parse_args(cli_args)
        module.resolve_mode_defaults(args)
        return args

    def test_default_scheduler_values_leave_cosine_decay_unset(self) -> None:
        for module_name in SCRIPT_CASES:
            module = self.load_module(module_name)
            with self.subTest(script=module_name):
                args = self.parse_args(module, ["--max-steps", "123", "--learning-rate", "0.002"])
                self.assertIsNone(args.lr_decay_iters)
                self.assertIsNone(args.min_lr)
                self.assertAlmostEqual(args.warmdown_fraction, 0.0)

    def test_scheduler_value_precedence_prefers_cli_then_env_then_computed_default(self) -> None:
        for module_name in SCRIPT_CASES:
            module = self.load_module(module_name)
            env = {"LR_DECAY_ITERS": "77", "MIN_LR": "0.00009"}

            with self.subTest(script=module_name, source="env"):
                args = self.parse_args(module, ["--max-steps", "123", "--learning-rate", "0.002"], env=env)
                self.assertEqual(args.lr_decay_iters, 77)
                self.assertAlmostEqual(args.min_lr, 0.00009)

            with self.subTest(script=module_name, source="computed_default_min_lr"):
                args = self.parse_args(
                    module,
                    ["--max-steps", "123", "--learning-rate", "0.002"],
                    env={"LR_DECAY_ITERS": "77"},
                )
                self.assertEqual(args.lr_decay_iters, 77)
                self.assertAlmostEqual(args.min_lr, 0.0002)

            with self.subTest(script=module_name, source="cli"):
                args = self.parse_args(
                    module,
                    [
                        "--max-steps",
                        "123",
                        "--learning-rate",
                        "0.002",
                        "--lr-decay-iters",
                        "55",
                        "--min-lr",
                        "0.00003",
                    ],
                    env=env,
                )
                self.assertEqual(args.lr_decay_iters, 55)
                self.assertAlmostEqual(args.min_lr, 0.00003)

    def test_build_trainer_config_threads_scheduler_values(self) -> None:
        for module_name in SCRIPT_CASES:
            module = self.load_module(module_name)
            with self.subTest(script=module_name):
                args = self.parse_args(
                    module,
                    [
                        "--max-steps",
                        "123",
                        "--learning-rate",
                        "0.002",
                        "--lr-decay-iters",
                        "55",
                        "--min-lr",
                        "0.00003",
                    ],
                )
                trainer_cfg = module.build_trainer_config(args, resolved_epochs=3)
                self.assertEqual(trainer_cfg.max_steps, 123)
                self.assertEqual(trainer_cfg.lr_decay_iters, 55)
                self.assertAlmostEqual(trainer_cfg.min_lr, 0.00003)
                self.assertAlmostEqual(trainer_cfg.warmdown_fraction, 0.0)
                self.assertIsNone(trainer_cfg.drop_last)

    def test_resolved_summary_includes_scheduler_fields(self) -> None:
        for module_name in SCRIPT_CASES:
            module = self.load_module(module_name)
            with self.subTest(script=module_name):
                args = self.parse_args(module, ["--max-steps", "12", "--learning-rate", "0.0012"])
                graph, spec = module.build_graph(args, "dummy_dataset")
                trainer_cfg = module.build_trainer_config(args, resolved_epochs=1)
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    module.print_resolved_summary(args, spec, trainer_cfg, {"steps_per_epoch": 1})
                output = stream.getvalue()
                self.assertIn('"warmdown_fraction"', output)
                self.assertIn('"lr_decay_iters"', output)
                self.assertIn('"min_lr"', output)
                self.assertIn('"drop_last"', output)

    def test_all_train_rows_rounds_effective_max_steps_without_implicit_lr_decay(self) -> None:
        for module_name in SCRIPT_CASES:
            module = self.load_module(module_name)
            cli_args = ["--max-steps", "53", "--all-train-rows"]
            if module_name != "train_llama_megakernel":
                cli_args.insert(0, "--megakernel")
            with self.subTest(script=module_name):
                args = self.parse_args(module, cli_args)
                (
                    derived,
                    resolved_epochs,
                    resolved_max_steps,
                    resolved_lr_decay_iters,
                    _resolved_max_wallclock_seconds,
                ) = (
                    module.resolve_effective_training_schedule(args, {"steps_per_epoch": 52})
                )
                self.assertEqual(resolved_epochs, 2)
                self.assertEqual(resolved_max_steps, 104)
                self.assertIsNone(resolved_lr_decay_iters)
                self.assertEqual(derived["resolved_max_steps"], 104)
                self.assertEqual(derived["requested_max_steps"], 53)
                self.assertTrue(derived["all_train_rows"])

    def test_all_train_rows_preserves_explicit_lr_decay_iters(self) -> None:
        for module_name in SCRIPT_CASES:
            module = self.load_module(module_name)
            cli_args = [
                "--max-steps",
                "53",
                "--all-train-rows",
                "--lr-decay-iters",
                "77",
            ]
            if module_name != "train_llama_megakernel":
                cli_args.insert(0, "--megakernel")
            with self.subTest(script=module_name):
                args = self.parse_args(module, cli_args)
                (
                    _derived,
                    _resolved_epochs,
                    resolved_max_steps,
                    resolved_lr_decay_iters,
                    _resolved_max_wallclock_seconds,
                ) = (
                    module.resolve_effective_training_schedule(args, {"steps_per_epoch": 52})
                )
                self.assertEqual(resolved_max_steps, 104)
                self.assertEqual(resolved_lr_decay_iters, 77)

    def test_all_train_rows_default_max_steps_uses_two_epoch_floor(self) -> None:
        for module_name in SCRIPT_CASES:
            module = self.load_module(module_name)
            cli_args = ["--all-train-rows"]
            if module_name != "train_llama_megakernel":
                cli_args.insert(0, "--megakernel")
            with self.subTest(script=module_name):
                args = self.parse_args(module, cli_args)
                derived, resolved_epochs, resolved_max_steps, resolved_lr_decay_iters = (
                    module.resolve_effective_training_schedule(args, {"steps_per_epoch": 300})[:4]
                )
                self.assertEqual(resolved_epochs, 2)
                self.assertEqual(resolved_max_steps, 600)
                self.assertIsNone(resolved_lr_decay_iters)
                self.assertEqual(derived["default_all_train_rows_epochs"], 2)
                self.assertEqual(derived["resolved_max_wallclock_seconds"], 0.0)

    def test_all_train_rows_treats_iterations_env_as_explicit(self) -> None:
        for module_name in SCRIPT_CASES:
            module = self.load_module(module_name)
            cli_args = ["--all-train-rows"]
            if module_name != "train_llama_megakernel":
                cli_args.insert(0, "--megakernel")
            with self.subTest(script=module_name):
                args = self.parse_args(module, cli_args, env={"ITERATIONS": "5"})
                (
                    derived,
                    resolved_epochs,
                    resolved_max_steps,
                    resolved_lr_decay_iters,
                    resolved_max_wallclock_seconds,
                ) = module.resolve_effective_training_schedule(args, {"steps_per_epoch": 100})
                self.assertEqual(resolved_epochs, 1)
                self.assertEqual(resolved_max_steps, 100)
                self.assertIsNone(resolved_lr_decay_iters)
                self.assertEqual(derived["requested_max_steps"], 5)
                self.assertEqual(resolved_max_wallclock_seconds, 0.0)

    def test_all_train_rows_preserves_explicit_wallclock_seconds(self) -> None:
        for module_name in SCRIPT_CASES:
            module = self.load_module(module_name)
            cli_args = [
                "--all-train-rows",
                "--max-wallclock-seconds",
                "7200",
            ]
            if module_name != "train_llama_megakernel":
                cli_args.insert(0, "--megakernel")
            with self.subTest(script=module_name):
                args = self.parse_args(module, cli_args)
                (
                    _derived,
                    _resolved_epochs,
                    _resolved_max_steps,
                    _resolved_lr_decay_iters,
                    resolved_max_wallclock_seconds,
                ) = module.resolve_effective_training_schedule(args, {"steps_per_epoch": 100})
                self.assertEqual(resolved_max_wallclock_seconds, 7200.0)

    def test_all_train_rows_treats_wallclock_env_as_explicit(self) -> None:
        for module_name in SCRIPT_CASES:
            module = self.load_module(module_name)
            cli_args = ["--all-train-rows"]
            if module_name != "train_llama_megakernel":
                cli_args.insert(0, "--megakernel")
            with self.subTest(script=module_name):
                args = self.parse_args(module, cli_args, env={"MAX_WALLCLOCK_SECONDS": "7200"})
                (
                    _derived,
                    _resolved_epochs,
                    _resolved_max_steps,
                    _resolved_lr_decay_iters,
                    resolved_max_wallclock_seconds,
                ) = module.resolve_effective_training_schedule(args, {"steps_per_epoch": 100})
                self.assertEqual(resolved_max_wallclock_seconds, 7200.0)

    def test_overnight_script_resolves_all_train_rows_schedule(self) -> None:
        module = self.load_path_module("train_semantic_router_moe_overnight_scheduler_test", OVERNIGHT_SCRIPT_PATH)
        args = self.parse_args(module, ["--megakernel", "--max-steps", "53", "--all-train-rows"])
        (
            derived,
            resolved_epochs,
            resolved_max_steps,
            resolved_lr_decay_iters,
            resolved_max_wallclock_seconds,
        ) = module.resolve_effective_training_schedule(args, {"steps_per_epoch": 52})
        self.assertEqual(resolved_epochs, 2)
        self.assertEqual(resolved_max_steps, 104)
        self.assertIsNone(resolved_lr_decay_iters)
        self.assertTrue(derived["all_train_rows"])
        self.assertEqual(resolved_max_wallclock_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
