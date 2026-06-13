from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import argparse
import torch


ROOT = Path(__file__).resolve().parents[1]
NEURALFN_ROOT = ROOT.parent
SCRIPTS_DIR = ROOT / "scripts"
if str(NEURALFN_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURALFN_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cli_utils import artifact_path


MEGAKERNEL_SCRIPT_CASES = [
    ("infer_jepa_semantic", "jepa_semantic_hybrid", "jepa_semantic_hybrid_megakernel"),
    ("infer_gpt2", "gpt2", "gpt2_megakernel"),
    ("infer_mixllama_fast", "mixllama_fast", "mixllama_fast_megakernel"),
    ("infer_nanogpt", "nanogpt", "nanogpt_megakernel"),
    ("infer_semantic_router_moe", "semantic_router_moe", "semantic_router_moe_megakernel"),
]
RAW_TEXT_TOKENIZER_FLAG_SCRIPTS = [
    "infer_jepa_semantic",
    "infer_gpt2",
    "infer_mixllama_fast",
    "infer_nanogpt",
    "infer_semantic_router_moe",
    "infer_llama_fast",
    "infer_llama_megakernel",
]
REPETITION_PENALTY_FLAG_SCRIPTS = [
    *RAW_TEXT_TOKENIZER_FLAG_SCRIPTS,
    "eval_llama_fast",
]


class InferMegakernelArtifactTest(unittest.TestCase):
    def load_module(self, module_name: str):
        return importlib.import_module(module_name)

    def parse_args(self, module, cli_args: list[str]):
        parser = module.build_parser()
        args = parser.parse_args(cli_args)
        if hasattr(module, "resolve_mode_defaults"):
            module.resolve_mode_defaults(args)
        return args

    def test_megakernel_inference_defaults_follow_runtime_flag(self) -> None:
        for module_name, compile_mode, megakernel_mode in MEGAKERNEL_SCRIPT_CASES:
            module = self.load_module(module_name)

            with self.subTest(script=module_name, runtime="compile"):
                args = self.parse_args(module, [])
                self.assertFalse(args.megakernel)
                self.assertEqual(module.mode_name(megakernel=False), compile_mode)
                self.assertEqual(Path(args.weights), module.default_weights_artifact(megakernel=False))
                self.assertEqual(Path(args.graph), module.default_graph_artifact(megakernel=False))

            with self.subTest(script=module_name, runtime="megakernel"):
                args = self.parse_args(module, ["--megakernel"])
                self.assertTrue(args.megakernel)
                self.assertEqual(module.mode_name(megakernel=True), megakernel_mode)
                self.assertEqual(Path(args.weights), module.default_weights_artifact(megakernel=True))
                self.assertEqual(Path(args.graph), module.default_graph_artifact(megakernel=True))

    def test_gpt2_inference_evo_defaults_follow_eager_artifacts(self) -> None:
        module = self.load_module("infer_gpt2")

        args = self.parse_args(module, ["--evo"])

        self.assertTrue(args.evo)
        self.assertFalse(args.megakernel)
        self.assertEqual("gpt2_evo", module.mode_name(megakernel=False, evo=True))
        self.assertEqual(artifact_path("gpt2_evo.pt"), Path(args.weights))
        self.assertEqual(artifact_path("gpt2_evo.json"), Path(args.graph))

    def test_llama_megakernel_wrapper_defaults_stay_runtime_specific(self) -> None:
        module = self.load_module("infer_llama_megakernel")

        args = self.parse_args(module, [])
        self.assertFalse(args.fast)
        self.assertEqual(Path(args.weights), module.default_weights_artifact(fast=False))
        self.assertEqual(Path(args.graph), module.default_graph_artifact(fast=False))

        fast_args = self.parse_args(module, ["--fast"])
        self.assertTrue(fast_args.fast)
        self.assertEqual(Path(fast_args.weights), module.default_weights_artifact(fast=True))
        self.assertEqual(Path(fast_args.graph), module.default_graph_artifact(fast=True))

    def test_raw_text_inference_scripts_accept_tokenizer_override_flags(self) -> None:
        for module_name in RAW_TEXT_TOKENIZER_FLAG_SCRIPTS:
            module = self.load_module(module_name)
            with self.subTest(script=module_name, flag="default"):
                args = self.parse_args(module, [])
                self.assertIsNone(getattr(args, "raw_text_encoding_override", None))
            with self.subTest(script=module_name, flag="tokenizer"):
                args = self.parse_args(module, ["--tokenizer", "sp2048"])
                self.assertEqual("sp2048", args.raw_text_encoding_override)
            with self.subTest(script=module_name, flag="cl100k"):
                args = self.parse_args(module, ["--cl100k"])
                self.assertEqual("cl100k_base", args.raw_text_encoding_override)
            with self.subTest(script=module_name, flag="o200k"):
                args = self.parse_args(module, ["--o200k"])
                self.assertEqual("o200k_base", args.raw_text_encoding_override)
            with self.subTest(script=module_name, flag="tokgpt2"):
                args = self.parse_args(module, ["--tokgpt2"])
                self.assertEqual("gpt2", args.raw_text_encoding_override)

    def test_raw_text_inference_tokenizer_override_flags_are_mutually_exclusive(self) -> None:
        for module_name in RAW_TEXT_TOKENIZER_FLAG_SCRIPTS:
            module = self.load_module(module_name)
            with self.subTest(script=module_name):
                parser = module.build_parser()
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--tokenizer", "sp2048", "--o200k"])

    def test_resolve_semantic_targets_normalizes_explicit_phrase_topics(self) -> None:
        module = self.load_module("infer_jepa_semantic")
        vocab = module.ConversationalVocabulary("vocab_86d_gpt2.json")

        sem_targets, semantic_overrides = module.resolve_semantic_targets(
            "",
            "politeness_strategy=happy to help",
            vocab.vector_dim,
            torch.device("cpu"),
            vocab,
        )

        self.assertEqual({"politeness_strategy": "happy_to_help"}, semantic_overrides)
        politeness_idx = vocab.dimension_to_expert["politeness_strategy"]
        self.assertEqual(
            vocab.term_to_index("politeness_strategy", "happy_to_help"),
            int(sem_targets[0, politeness_idx]),
        )

    def test_resolve_semantic_targets_auto_matches_contiguous_phrase_text(self) -> None:
        module = self.load_module("infer_jepa_semantic")
        vocab = module.ConversationalVocabulary("vocab_86d_gpt2.json")

        sem_targets, semantic_overrides = module.resolve_semantic_targets(
            "",
            "",
            vocab.vector_dim,
            torch.device("cpu"),
            vocab,
            sequence_text="I am always happy to help when someone asks clearly.",
        )

        self.assertEqual({"politeness_strategy": "happy_to_help"}, semantic_overrides)
        politeness_idx = vocab.dimension_to_expert["politeness_strategy"]
        self.assertEqual(
            vocab.term_to_index("politeness_strategy", "happy_to_help"),
            int(sem_targets[0, politeness_idx]),
        )

    def test_resolve_semantic_targets_numeric_overrides_beat_auto_phrase_matching(self) -> None:
        module = self.load_module("infer_jepa_semantic")
        vocab = module.ConversationalVocabulary("vocab_86d_gpt2.json")
        semantic_values = [module.SEMANTIC_IGNORE_INDEX] * vocab.vector_dim
        politeness_idx = vocab.dimension_to_expert["politeness_strategy"]
        semantic_values[politeness_idx] = vocab.term_to_index("politeness_strategy", "formal")
        raw_targets = ",".join(str(value) for value in semantic_values)

        sem_targets, semantic_overrides = module.resolve_semantic_targets(
            raw_targets,
            "",
            vocab.vector_dim,
            torch.device("cpu"),
            vocab,
            sequence_text="I am happy to help with that request.",
        )

        self.assertEqual({}, semantic_overrides)
        self.assertEqual(semantic_values, sem_targets[0].tolist())

    def test_generation_scripts_accept_repetition_penalty_flag(self) -> None:
        for module_name in REPETITION_PENALTY_FLAG_SCRIPTS:
            module = self.load_module(module_name)
            with self.subTest(script=module_name, flag="default"):
                args = self.parse_args(module, [])
                self.assertEqual(1.0, getattr(args, "repetition_penalty"))
            with self.subTest(script=module_name, flag="enabled"):
                args = self.parse_args(module, ["--repetition-penalty", "1.25"])
                self.assertEqual(1.25, args.repetition_penalty)

    def test_repetition_penalty_parser_rejects_values_below_one(self) -> None:
        module = self.load_module("infer_jepa_semantic")
        with self.assertRaises(argparse.ArgumentTypeError):
            module.repetition_penalty_arg("0.99")

    def test_apply_repetition_penalty_adjusts_seen_token_logits(self) -> None:
        module = self.load_module("infer_jepa_semantic")
        logits = torch.tensor([[3.0, 1.5, -0.5, -1.0]], dtype=torch.float32)
        penalized = module.apply_repetition_penalty(
            logits,
            token_history=[0, 0, 2],
            repetition_penalty=1.2,
        )
        self.assertAlmostEqual(2.5, float(penalized[0, 0]), places=5)
        self.assertAlmostEqual(1.5, float(penalized[0, 1]), places=5)
        self.assertAlmostEqual(-0.6, float(penalized[0, 2]), places=5)
        self.assertAlmostEqual(-1.0, float(penalized[0, 3]), places=5)

    def test_sample_next_token_greedy_respects_repetition_penalty(self) -> None:
        module = self.load_module("infer_jepa_semantic")
        logits = torch.tensor([[3.0, 2.8, 0.1]], dtype=torch.float32)
        generator = torch.Generator()
        generator.manual_seed(1337)

        without_penalty = module.sample_next_token(
            logits,
            temperature=0.0,
            top_k=0,
            token_history=[0],
            repetition_penalty=1.0,
            generator=generator,
        )
        with_penalty = module.sample_next_token(
            logits,
            temperature=0.0,
            top_k=0,
            token_history=[0],
            repetition_penalty=1.2,
            generator=generator,
        )
        self.assertEqual(0, without_penalty)
        self.assertEqual(1, with_penalty)

    def test_top_p_filter_drops_tokens_outside_nucleus(self) -> None:
        module = self.load_module("infer_jepa_semantic")
        logits = torch.tensor([[10.0, 9.0, 1.0]], dtype=torch.float32)

        filtered = module.top_p_filter(logits, 0.7)

        self.assertTrue(torch.isfinite(filtered[0, 0]))
        self.assertFalse(torch.isfinite(filtered[0, 1]))
        self.assertFalse(torch.isfinite(filtered[0, 2]))

    def test_runtime_mismatch_reports_targeted_graph_hint(self) -> None:
        module = self.load_module("infer_jepa_semantic")
        graph = SimpleNamespace(
            name="semantic_router_moe_sdk",
            torch_config={"template_spec": {"template": {"runtime": "compile"}}},
        )
        state_dict = {
            "node_modules.model.node_modules.block_0.node_modules.attention.node_modules.fused_attn.q_proj.weight":
                torch.zeros(1, 1)
        }
        graph_path = artifact_path("semantic_router_moe.json")
        weights_path = artifact_path("semantic_router_moe_megakernel.pt")

        with self.assertRaises(RuntimeError) as exc_info:
            module.validate_inference_artifact_runtime_compatibility(
                graph=graph,
                state_dict=state_dict,
                graph_path=graph_path,
                weights_path=weights_path,
            )

        message = str(exc_info.exception)
        self.assertIn("Graph runtime: compile", message)
        self.assertIn("Checkpoint runtime: megakernel", message)
        self.assertIn(str(weights_path.with_suffix(".json")), message)

    def test_runtime_mismatch_reports_targeted_weights_hint(self) -> None:
        module = self.load_module("infer_jepa_semantic")
        graph = SimpleNamespace(
            name="semantic_router_moe_megakernel_sdk",
            torch_config={"template_spec": {"template": {"runtime": "megakernel"}}},
        )
        state_dict = {
            "node_modules.model.node_modules.block_0.node_modules.attention.node_modules.q_proj.proj.weight":
                torch.zeros(1, 1)
        }
        graph_path = artifact_path("semantic_router_moe_megakernel.json")
        weights_path = artifact_path("semantic_router_moe.pt")

        with self.assertRaises(RuntimeError) as exc_info:
            module.validate_inference_artifact_runtime_compatibility(
                graph=graph,
                state_dict=state_dict,
                graph_path=graph_path,
                weights_path=weights_path,
            )

        message = str(exc_info.exception)
        self.assertIn("Graph runtime: megakernel", message)
        self.assertIn("Checkpoint runtime: compile", message)
        self.assertIn(str(graph_path.with_suffix(".pt")), message)

    def test_unknown_checkpoint_runtime_does_not_raise(self) -> None:
        module = self.load_module("infer_jepa_semantic")
        graph = SimpleNamespace(
            name="semantic_router_moe_sdk",
            torch_config={"template_spec": {"template": {"runtime": "compile"}}},
        )
        state_dict = {"node_modules.model.node_modules.block_0.some_other.weight": torch.zeros(1, 1)}
        module.validate_inference_artifact_runtime_compatibility(
            graph=graph,
            state_dict=state_dict,
            graph_path=artifact_path("semantic_router_moe.json"),
            weights_path=artifact_path("semantic_router_moe.pt"),
        )

    def test_eager_runtime_metadata_is_recognized(self) -> None:
        module = self.load_module("infer_jepa_semantic")
        graph = SimpleNamespace(
            name="gpt2_evo_sdk",
            torch_config={"template_spec": {"template": {"runtime": "eager"}}},
        )
        state_dict = {"node_modules.model.node_modules.token_embed.weight": torch.zeros(1, 1)}

        self.assertEqual("eager", module.infer_graph_template_runtime(graph))
        self.assertEqual("eager", module.infer_checkpoint_runtime(state_dict, {"template_runtime": "eager"}))
        module.validate_inference_artifact_runtime_compatibility(
            graph=graph,
            state_dict=state_dict,
            checkpoint_metadata={"template_runtime": "eager"},
            graph_path=artifact_path("gpt2_evo.json"),
            weights_path=artifact_path("gpt2_evo.pt"),
        )

    def test_load_compiled_graph_uses_weights_referenced_by_graph_when_not_explicit(self) -> None:
        module = self.load_module("infer_jepa_semantic")
        graph_path = artifact_path("semantic_router_moe.json")
        fake_graph = SimpleNamespace(
            name="semantic_router_moe_sdk",
            torch_config={"artifact_metadata": {"weights_file": "semantic_router_moe.pt"}},
            nodes={},
        )
        loaded_paths: list[Path] = []
        state_dict = {"node_modules.model.node_modules.token_embed.embedding.weight": torch.zeros(1, 1)}

        class FakeCompiled:
            def __init__(self, _graph):
                self.loaded_state_dict = None

            def load_state_dict(self, payload):
                self.loaded_state_dict = payload
                return None

            def to(self, _device):
                return self

            def eval(self):
                return self

        def fake_load_checkpoint(path, *, map_location="cpu"):
            loaded_paths.append(Path(path))
            return state_dict, {}

        with (
            patch.object(module, "load_graph", return_value=fake_graph),
            patch.object(module, "load_pt_checkpoint", side_effect=fake_load_checkpoint),
            patch.object(module, "CompiledTorchGraph", FakeCompiled),
        ):
            graph, compiled, loaded_state_dict, resolved_weights_path = module.load_compiled_inference_graph(
                graph_path=graph_path,
                weights_path=None,
                device=torch.device("cpu"),
            )

        self.assertIs(graph, fake_graph)
        self.assertEqual(state_dict, loaded_state_dict)
        self.assertEqual(graph_path.with_suffix(".pt"), resolved_weights_path)
        self.assertEqual([graph_path.with_suffix(".pt")], loaded_paths)
        self.assertEqual(state_dict, compiled.loaded_state_dict)

    def test_load_tokenizer_from_graph_manifest_supports_tiktoken(self) -> None:
        module = self.load_module("infer_jepa_semantic")
        graph = SimpleNamespace(
            torch_config={
                "tokenizer_manifest": {
                    "backend": "tiktoken",
                    "encoding_name": "o200k_base",
                    "tokenizer_vocab_size": 199998,
                }
            }
        )

        tokenizer, tokenizer_path, tokenizer_name = module.load_tokenizer_from_graph_manifest(graph)

        self.assertIsNotNone(tokenizer)
        self.assertIsNotNone(tokenizer_path)
        self.assertEqual("o200k_base", tokenizer_name)

    def test_resolve_raw_text_encoding_name_prefers_saved_template_spec_tokenizer(self) -> None:
        module = self.load_module("infer_jepa_semantic")
        graph = SimpleNamespace(
            torch_config={
                "template_spec": {
                    "template": {"backbone": "llama"},
                    "raw_text_encoding_name": "sp8192",
                },
                "tokenizer_manifest": {
                    "backend": "tiktoken",
                    "encoding_name": "o200k_base",
                    "tokenizer_vocab_size": 199998,
                },
            }
        )

        resolved = module.resolve_raw_text_encoding_name(graph)

        self.assertEqual("sp8192", resolved)

    def test_resolve_raw_text_encoding_name_falls_back_to_tokenizer_manifest_for_legacy_graphs(self) -> None:
        module = self.load_module("infer_jepa_semantic")
        graph = SimpleNamespace(
            torch_config={
                "template_spec": {
                    "template": {"backbone": "llama"},
                },
                "tokenizer_manifest": {
                    "backend": "tiktoken",
                    "encoding_name": "o200k_base",
                    "tokenizer_vocab_size": 199998,
                },
            }
        )

        resolved = module.resolve_raw_text_encoding_name(graph)

        self.assertEqual("o200k_base", resolved)

    def test_resolve_raw_text_encoding_name_still_allows_explicit_override(self) -> None:
        module = self.load_module("infer_jepa_semantic")
        graph = SimpleNamespace(
            torch_config={
                "template_spec": {
                    "template": {"backbone": "llama"},
                    "raw_text_encoding_name": "sp8192",
                },
                "tokenizer_manifest": {
                    "backend": "tiktoken",
                    "encoding_name": "o200k_base",
                    "tokenizer_vocab_size": 199998,
                },
            }
        )

        resolved = module.resolve_raw_text_encoding_name(graph, encoding_override="gpt2")

        self.assertEqual("gpt2", resolved)


if __name__ == "__main__":
    unittest.main()
