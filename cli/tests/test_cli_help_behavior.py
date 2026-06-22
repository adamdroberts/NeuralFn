from __future__ import annotations

import importlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from contextlib import redirect_stderr


ROOT = Path(__file__).resolve().parents[1]
NEURALFN_ROOT = ROOT.parent
SCRIPTS_DIR = ROOT / "scripts"
if str(NEURALFN_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURALFN_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


SCRIPT_CASES = [
    ("train_gpt2", "Train dense GPT templates with the NeuralFn native CUDA harness."),
    ("train_deepseek_v4", "Train deepseek_v4 through the NeuralFn native CUDA harness."),
    ("train_jepa_semantic", "Train jepa_semantic_hybrid with the NeuralFn CUDA harness."),
    ("train_llama_fast", "Train llama_fast with the NeuralFn CUDA harness."),
    ("train_llama_megakernel", "Train llama_megakernel with the NeuralFn CUDA harness."),
    ("train_mixllama_fast", "Train mixllama_fast through the NeuralFn native CUDA harness."),
    ("train_nanogpt", "Train nanogpt with the NeuralFn native CUDA harness."),
    ("train_semantic_router_moe", "Train semantic_router_moe with the NeuralFn CUDA harness."),
    ("train_semantic_router_moe-overnight", "Train semantic_router_moe overnight with the NeuralFn CUDA harness."),
    ("infer_gpt", "Run text generation with exported GPT artifacts on CUDA."),
    ("infer_gpt2", "Run text generation with exported GPT artifacts on CUDA."),
    ("infer_jepa_semantic", "Run text generation with exported jepa_semantic_hybrid artifacts on CUDA."),
    ("infer_llama_fast", "Run text generation with exported llama_fast artifacts on CUDA."),
    ("infer_llama_megakernel", "Run text generation with exported llama_megakernel artifacts on CUDA."),
    ("infer_mixllama_fast", "Run text generation with exported mixllama_fast artifacts on CUDA."),
    ("infer_nanogpt", "Run text generation with exported nanogpt artifacts on CUDA."),
    ("infer_semantic_router_moe", "Run text generation with exported semantic_router_moe artifacts on CUDA."),
]


class CliHelpBehaviorTest(unittest.TestCase):
    def load_module(self, module_name: str):
        if module_name == "train_semantic_router_moe-overnight":
            spec = importlib.util.spec_from_file_location(
                "train_semantic_router_moe_overnight",
                SCRIPTS_DIR / f"{module_name}.py",
            )
            assert spec is not None
            assert spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        return importlib.import_module(module_name)

    def test_descriptions_are_short_and_present_in_help(self) -> None:
        for module_name, description in SCRIPT_CASES:
            module = self.load_module(module_name)
            parser = module.build_parser()
            with self.subTest(script=module_name):
                self.assertEqual(description, parser.description)
                self.assertTrue(parser.description)
                self.assertLessEqual(len(parser.description), 100)
                self.assertFalse(parser.allow_abbrev)
                self.assertIn(description, parser.format_help())

    def test_invalid_flags_print_full_help_before_exiting(self) -> None:
        for module_name, description in SCRIPT_CASES:
            module = self.load_module(module_name)
            parser = module.build_parser()
            stderr = io.StringIO()
            with self.subTest(script=module_name):
                with redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as exc_info:
                        parser.parse_args(["--definitely-invalid-flag"])
                self.assertEqual(2, exc_info.exception.code)
                output = stderr.getvalue()
                self.assertIn("usage:", output)
                self.assertIn(description, output)
                self.assertIn("error: unrecognized arguments: --definitely-invalid-flag", output)

    def test_partial_long_flags_do_not_abbreviate(self) -> None:
        module = self.load_module("train_llama_fast")
        parser = module.build_parser()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as exc_info:
                parser.parse_args(["--dev", "cuda"])
        self.assertEqual(2, exc_info.exception.code)
        self.assertIn("error: unrecognized arguments: --dev cuda", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
