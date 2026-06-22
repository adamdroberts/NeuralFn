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
    "train_gpt2",
]


class TrainEvolutionaryFlagTest(unittest.TestCase):
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

    def test_parser_accepts_evolutionary_flags_for_all_training_scripts(self) -> None:
        for module_name in SCRIPT_CASES:
            module = self.load_module(module_name)
            cli_args = [
                "--evolutionary",
                "--evo-population-size",
                "7",
                "--evo-mutation-rate",
                "0.2",
                "--evo-mutation-scale",
                "0.05",
                "--evo-crossover-rate",
                "0.4",
                "--evo-tournament-size",
                "2",
                "--evo-elite-count",
                "1",
                "--evo-seed",
                "123",
            ]
            with self.subTest(script=module_name):
                args = self.parse_args(module, cli_args)
                self.assertTrue(args.evolutionary)
                self.assertEqual(args.evo_population_size, 7)
                self.assertAlmostEqual(args.evo_mutation_rate, 0.2)
                self.assertAlmostEqual(args.evo_mutation_scale, 0.05)
                self.assertAlmostEqual(args.evo_crossover_rate, 0.4)
                self.assertEqual(args.evo_tournament_size, 2)
                self.assertEqual(args.evo_elite_count, 1)
                self.assertEqual(args.evo_seed, 123)

    def test_build_trainer_config_threads_evolutionary_fields_and_seed_fallback(self) -> None:
        for module_name in SCRIPT_CASES:
            module = self.load_module(module_name)
            cli_args = [
                "--seed",
                "77",
                "--evolutionary",
                "--evo-population-size",
                "7",
                "--evo-mutation-rate",
                "0.2",
                "--evo-mutation-scale",
                "0.05",
                "--evo-crossover-rate",
                "0.4",
                "--evo-tournament-size",
                "2",
                "--evo-elite-count",
                "1",
            ]
            with self.subTest(script=module_name):
                args = self.parse_args(module, cli_args)
                trainer_cfg = module.build_trainer_config(args, resolved_epochs=3)
                self.assertTrue(trainer_cfg.evolutionary)
                self.assertEqual(trainer_cfg.evo_population_size, 7)
                self.assertAlmostEqual(trainer_cfg.evo_mutation_rate, 0.2)
                self.assertAlmostEqual(trainer_cfg.evo_mutation_scale, 0.05)
                self.assertAlmostEqual(trainer_cfg.evo_crossover_rate, 0.4)
                self.assertEqual(trainer_cfg.evo_tournament_size, 2)
                self.assertEqual(trainer_cfg.evo_elite_count, 1)
                self.assertEqual(trainer_cfg.evo_seed, 77)

    def test_resolved_summary_includes_evolutionary_metadata(self) -> None:
        for module_name in SCRIPT_CASES:
            module = self.load_module(module_name)
            with self.subTest(script=module_name):
                args = self.parse_args(
                    module,
                    [
                        "--evolutionary",
                        "--evo-population-size",
                        "7",
                        "--evo-seed",
                        "123",
                    ],
                )
                graph, spec = module.build_graph(args, "dummy_dataset")
                trainer_cfg = module.build_trainer_config(args, resolved_epochs=1)
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    module.print_resolved_summary(args, spec, trainer_cfg, {"steps_per_epoch": 1})
                output = stream.getvalue()
                self.assertIn('"optimization_method": "evolutionary"', output)
                self.assertIn('"ignored_gradient_optimizer_fields"', output)
                self.assertIn('"population_size": 7', output)
                self.assertIn('"seed": 123', output)

if __name__ == "__main__":
    unittest.main()
