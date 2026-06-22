from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


SCRIPT_CASES = [
    ("train_jepa_semantic", "jepa_semantic_hybrid", "jepa_semantic_hybrid_megakernel", "compile"),
    ("train_gpt2", "gpt2", "gpt2_megakernel", "eager"),
]


class TrainMegakernelFlagTest(unittest.TestCase):
    def load_module(self, module_name: str):
        return importlib.import_module(module_name)

    def parse_args(self, module, cli_args: list[str]):
        with patch.dict(os.environ, {"OUTPUT": ""}, clear=False):
            parser = module.build_parser()
            args = parser.parse_args(cli_args)
        module.resolve_mode_defaults(args)
        return args

    def test_default_output_paths_follow_runtime_mode(self) -> None:
        for module_name, regular_mode, megakernel_mode, _regular_runtime in SCRIPT_CASES:
            module = self.load_module(module_name)
            with self.subTest(script=module_name, runtime="compile"):
                args = self.parse_args(module, [])
                self.assertFalse(args.megakernel)
                self.assertEqual(module.mode_name(megakernel=False), regular_mode)
                self.assertEqual(Path(args.output), module.default_output_path(megakernel=False))

            with self.subTest(script=module_name, runtime="megakernel"):
                args = self.parse_args(module, ["--megakernel"])
                self.assertTrue(args.megakernel)
                self.assertEqual(module.mode_name(megakernel=True), megakernel_mode)
                self.assertEqual(Path(args.output), module.default_output_path(megakernel=True))

    def test_output_precedence_prefers_cli_then_env_then_mode_default(self) -> None:
        for module_name, _, _, _ in SCRIPT_CASES:
            module = self.load_module(module_name)
            env_output = "/tmp/from-env.pt"
            cli_output = "/tmp/from-cli.pt"

            with self.subTest(script=module_name, source="env"):
                with patch.dict(os.environ, {"OUTPUT": env_output}, clear=False):
                    parser = module.build_parser()
                    args = parser.parse_args(["--megakernel"])
                module.resolve_mode_defaults(args)
                self.assertEqual(args.output, env_output)

            with self.subTest(script=module_name, source="cli"):
                with patch.dict(os.environ, {"OUTPUT": env_output}, clear=False):
                    parser = module.build_parser()
                    args = parser.parse_args(["--megakernel", "--output", cli_output])
                module.resolve_mode_defaults(args)
                self.assertEqual(args.output, cli_output)

    def test_build_graph_uses_runtime_specific_name_and_spec(self) -> None:
        for module_name, regular_mode, megakernel_mode, regular_runtime in SCRIPT_CASES:
            module = self.load_module(module_name)

            with self.subTest(script=module_name, runtime="compile"):
                args = self.parse_args(module, [])
                graph, spec = module.build_graph(args, "dummy_dataset")
                self.assertEqual(graph.name, module.graph_name(megakernel=False))
                self.assertEqual(graph.name, f"{regular_mode}_sdk")
                self.assertEqual(spec.template.runtime, regular_runtime)

            with self.subTest(script=module_name, runtime="megakernel"):
                args = self.parse_args(module, ["--megakernel"])
                graph, spec = module.build_graph(args, "dummy_dataset")
                self.assertEqual(graph.name, module.graph_name(megakernel=True))
                self.assertEqual(graph.name, f"{megakernel_mode}_sdk")
                self.assertEqual(spec.template.runtime, "megakernel")


if __name__ == "__main__":
    unittest.main()
