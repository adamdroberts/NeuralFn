from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
NEURALFN_ROOT = ROOT.parent
SCRIPTS_DIR = ROOT / "scripts"
for candidate in (ROOT, NEURALFN_ROOT, SCRIPTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import nfn
import nfn_impl
from cli_utils import artifact_path, artifact_root
from infer_jepa_semantic import resolve_autocast_settings
from parameter_golf_runtime import (
    ParameterGolfConfig,
    ParameterGolfGPT,
    ParameterGolfSentencePieceTokenizer,
    sanitize_caseops_decoded_text,
)
from train_jepa_semantic import dataset_download_kwargs_from_args, parameter_golf_dataset_alias


class NfnCliTest(unittest.TestCase):
    def test_artifact_root_defaults_to_home_neuralfn_artifacts(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEURALFN_ARTIFACTS_DIR", None)
            self.assertEqual(Path.home() / "NeuralFn" / "artifacts", artifact_root())
            self.assertEqual(Path.home() / "NeuralFn" / "artifacts" / "model.pt", artifact_path("model.pt"))

    def test_artifact_root_honors_env_override_with_user_expansion(self) -> None:
        with patch.dict(os.environ, {"NEURALFN_ARTIFACTS_DIR": "~/custom-nfn-artifacts"}, clear=False):
            self.assertEqual(Path.home() / "custom-nfn-artifacts", artifact_root())
            self.assertEqual(Path.home() / "custom-nfn-artifacts" / "model.pt", artifact_path("model.pt"))

    def _question_choices(self, key: str, state: dict[str, object]) -> list[nfn_impl.OptionChoice]:
        question = next(question for question in nfn_impl.training_questionnaire(set()) if question.key == key)
        return question.options_factory(state)

    def test_recipe_from_state_supports_full_composed_combo(self) -> None:
        recipe = nfn.recipe_from_state(
            {
                "base_model": "nanogpt",
                "topology": "moe",
                "router_mode": "semantic",
                "use_jepa": True,
                "megakernel": True,
            }
        )
        self.assertEqual("nanogpt", recipe.base_model)
        self.assertEqual("moe", recipe.topology)
        self.assertEqual("semantic", recipe.router_mode)
        self.assertTrue(recipe.use_jepa)
        self.assertEqual("megakernel", recipe.runtime)
        self.assertEqual("nanogpt_semantic_moe_jepa_megakernel", recipe.mode_name())

    def test_dense_gpt_aliases_are_universal_native_family(self) -> None:
        parser = nfn_impl.build_command_parser("train", style="long")
        for base_model in ("gpt", "gpt2", "gpt3"):
            with self.subTest(base_model=base_model):
                args = parser.parse_args(["--base-model", base_model, "--tinystories"])
                self.assertEqual(base_model, args.base_model)

                recipe = nfn.recipe_from_state({"base_model": base_model, "topology": "dense"})
                self.assertEqual("gpt2", recipe.mode_name())
                self.assertEqual("gpt2", nfn_impl.spec_base_model(recipe))

                defaults = nfn_impl.base_model_defaults(base_model)
                self.assertEqual(base_model, defaults["model_family"])

        self.assertEqual(2048, nfn_impl.base_model_defaults("gpt3")["train_seq_len"])

    def test_render_help_mentions_base_model_router_and_presets(self) -> None:
        help_text = nfn.render_help("train", style="verbose")
        self.assertIn("--base-model", help_text)
        self.assertIn("--router-mode", help_text)
        self.assertIn("--model-preset", help_text)
        self.assertIn("--optimizer-preset", help_text)
        self.assertIn("--pretraining-file", help_text)

    def test_kernels_list_json_reports_complete_registry(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            rc = nfn_impl.main(["kernels", "list", "--json"], stdin_isatty=False, stdout_isatty=False)

        self.assertEqual(0, rc)
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["complete"])
        self.assertEqual([], payload["missing"])
        self.assertEqual(payload["total_inventory"], payload["accounted"])
        keys = {spec["inventory_key"] for spec in payload["specs"]}
        self.assertIn("function:gelu", keys)
        self.assertIn("module:gelu", keys)

    def test_kernels_doctor_json_reports_diagnostics_and_coverage_summary(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            rc = nfn_impl.main(["kernels", "doctor", "--json"], stdin_isatty=False, stdout_isatty=False)

        self.assertEqual(0, rc)
        payload = json.loads(out.getvalue())
        self.assertIn("diagnostics", payload)
        self.assertIn("coverage", payload)
        self.assertTrue(payload["coverage"]["complete"])
        self.assertIn("cuda_tile_header", payload["diagnostics"])

    def test_kernels_bench_json_reports_execution_modes(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            rc = nfn_impl.main(
                ["kernels", "bench", "--json", "--device", "cpu", "--iterations", "1", "--warmup", "0"],
                stdin_isatty=False,
                stdout_isatty=False,
            )

        self.assertEqual(0, rc)
        payload = json.loads(out.getvalue())
        self.assertEqual("cpu", payload["device"])
        self.assertEqual(1, payload["iterations"])
        self.assertEqual(0, payload["warmup"])
        self.assertIn("graph_walk_pytorch", payload["seconds"])
        self.assertIn("compiled_pytorch", payload["seconds"])
        self.assertIn("compiled_tile_cuda_requested", payload["seconds"])

    def test_kernels_examples_json_lists_and_writes_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                rc = nfn_impl.main(
                    ["kernels", "examples", "--json", "--write", "--output-dir", tmp],
                    stdin_isatty=False,
                    stdout_isatty=False,
                )

            self.assertEqual(0, rc)
            payload = json.loads(out.getvalue())
            self.assertTrue(payload["written"])
            self.assertGreater(payload["generated_count"], 0)
            self.assertTrue((Path(tmp) / "scalar_add_train.py").exists())
            generated_dir = Path(payload["generated_dir"])
            self.assertTrue(generated_dir.exists())
            self.assertTrue(any(generated_dir.glob("function_*.py")))

    def test_raw_text_tokenizer_choices_show_sentencepiece_variants_even_when_missing(self) -> None:
        recipe = nfn.recipe_from_state({"base_model": "gpt2", "topology": "dense"})
        availability = {"sp1024": True, "sp2048": False, "sp4096": True, "sp8192": False}
        with patch.object(
            nfn_impl,
            "raw_text_tokenizer_is_available",
            side_effect=lambda name: not str(name).startswith("sp") or availability.get(str(name), False),
        ):
            choices = nfn.tokenizer_choices(recipe, {"dataset": "shakespeare"})
        labels = [choice.label for choice in choices]
        self.assertEqual(
            [
                "GPT-2 tokenizer",
                "cl100k tokenizer",
                "o200k tokenizer",
                "sp1024 tokenizer",
                "sp2048 tokenizer",
                "sp4096 tokenizer",
                "sp8192 tokenizer",
            ],
            labels,
        )
        descriptions = {choice.label: choice.description for choice in choices}
        self.assertIn("Downloads shared tokenizer assets before training if missing.", descriptions["sp2048 tokenizer"])
        self.assertIn("Downloads shared tokenizer assets before training if missing.", descriptions["sp8192 tokenizer"])

    def test_golf_tokenizer_choices_offer_supported_sentencepiece_variants(self) -> None:
        choices = self._question_choices("tokenizer", {"dataset": "golf1"})
        variants = [choice.value["tokenizer"] for choice in choices]
        self.assertEqual(["sp1024", "sp2048", "sp4096", "sp8192"], variants)
        recommended = next(choice for choice in choices if choice.recommended)
        self.assertEqual({"tokenizer": "sp1024"}, recommended.value)

    def test_dataset_defaults_drive_raw_text_tokenizer_recommendations(self) -> None:
        recipe = nfn.recipe_from_state({"base_model": "nanogpt", "topology": "dense"})
        choices = nfn.tokenizer_choices(recipe, {"dataset": "tinystories"})
        recommended = next(choice for choice in choices if choice.recommended)
        self.assertEqual({"tokenizer": "o200k_base"}, recommended.value)

        gpt2_recipe = nfn.recipe_from_state({"base_model": "gpt2", "topology": "dense"})
        shakespeare_choices = nfn.tokenizer_choices(gpt2_recipe, {"dataset": "shakespeare"})
        shakespeare_recommended = next(choice for choice in shakespeare_choices if choice.recommended)
        self.assertEqual({"tokenizer": "cl100k_base"}, shakespeare_recommended.value)

    def test_pretraining_file_uses_raw_text_tokenizer_choices(self) -> None:
        recipe = nfn.recipe_from_state({"base_model": "gpt2", "topology": "dense"})
        with patch.object(nfn_impl, "raw_text_tokenizer_is_available", side_effect=lambda name: not str(name).startswith("sp")):
            choices = nfn.tokenizer_choices(recipe, {"pretraining_file": "/tmp/corpus.txt"})
        labels = [choice.label for choice in choices]
        self.assertEqual(
            [
                "GPT-2 tokenizer",
                "cl100k tokenizer",
                "o200k tokenizer",
                "sp1024 tokenizer",
                "sp2048 tokenizer",
                "sp4096 tokenizer",
                "sp8192 tokenizer",
            ],
            labels,
        )

    def test_advanced_hyperparameter_choices_include_resolved_values(self) -> None:
        state: dict[str, object] = {
            "base_model": "llama",
            "topology": "dense",
            "router_mode": "standard",
            "use_jepa": False,
            "megakernel": False,
            "model_preset": "harness_default",
            "run_preset": "default",
            "_action": "advanced",
            "num_heads": 6,
        }
        recipe = nfn.recipe_from_state(state)
        preset = nfn_impl.model_preset_values(recipe, "harness_default")

        layer_labels = [choice.label for choice in self._question_choices("num_layers", state)]
        self.assertIn(f"Recommended ({preset['num_layers']} layers)", layer_labels)
        self.assertIn(f"Smaller ({max(2, int(preset['num_layers']) - 1)} layers)", layer_labels)
        self.assertIn(f"Larger ({int(preset['num_layers']) + 1} layers)", layer_labels)

        model_dim_labels = [choice.label for choice in self._question_choices("model_dim", state)]
        self.assertIn(f"Recommended (d_model {int(preset['model_dim'])})", model_dim_labels)

        kv_head_labels = [choice.label for choice in self._question_choices("num_kv_heads", state)]
        self.assertIn("Match heads (6 KV heads)", kv_head_labels)
        self.assertIn("Half heads (3 KV heads)", kv_head_labels)

        max_steps_labels = [choice.label for choice in self._question_choices("max_steps", state)]
        default_steps = int(nfn_impl.RUN_PRESET_VALUES["default"]["max_steps"])
        self.assertIn(f"Recommended ({default_steps} steps)", max_steps_labels)
        self.assertIn(f"Shorter ({max(20, default_steps // 2)} steps)", max_steps_labels)

        token_budget_labels = [choice.label for choice in self._question_choices("train_batch_tokens", state)]
        default_tokens = int(nfn_impl.RUN_PRESET_VALUES["default"]["train_batch_tokens"])
        self.assertIn(f"Recommended ({default_tokens:,} tokens/step)", token_budget_labels)
        self.assertIn(f"Smaller ({max(1024, default_tokens // 2):,} tokens/step)", token_budget_labels)

    def test_preset_descriptions_include_concrete_values(self) -> None:
        dense_state: dict[str, object] = {
            "base_model": "llama",
            "topology": "dense",
            "router_mode": "standard",
            "use_jepa": False,
            "megakernel": False,
        }
        dense_recipe = nfn.recipe_from_state(dense_state)
        model_choices = {choice.value: choice for choice in self._question_choices("model_preset", dense_state)}
        harness_defaults = nfn_impl.model_preset_values(dense_recipe, "harness_default")
        self.assertIn(f"{int(harness_defaults['num_layers'])} layers", model_choices["harness_default"].description)
        self.assertIn(f"d_model {int(harness_defaults['model_dim'])}", model_choices["harness_default"].description)
        self.assertIn(f"{int(harness_defaults['num_heads'])} heads", model_choices["harness_default"].description)

        moe_state: dict[str, object] = {
            "base_model": "llama",
            "topology": "moe",
            "router_mode": "semantic",
            "use_jepa": False,
            "megakernel": False,
        }
        moe_recipe = nfn.recipe_from_state(moe_state)
        moe_choices = {choice.value: choice for choice in self._question_choices("model_preset", moe_state)}
        moe_defaults = nfn_impl.model_preset_values(moe_recipe, "harness_default")
        self.assertIn(f"{int(moe_defaults['experts'])} experts", moe_choices["harness_default"].description)
        self.assertIn(f"top-k {int(moe_defaults['top_k'])}", moe_choices["harness_default"].description)

        run_choices = {choice.value: choice for choice in self._question_choices("run_preset", dense_state)}
        self.assertIn("400 steps", run_choices["default"].description)
        self.assertIn("24,576 tokens/step", run_choices["default"].description)
        self.assertIn("batch 8", run_choices["default"].description)

        optimizer_choices = {choice.value: choice for choice in self._question_choices("optimizer_preset", dense_state)}
        self.assertIn("profile adamw", optimizer_choices["gradient_default"].description)
        self.assertIn("lr 6e-4", optimizer_choices["gradient_default"].description)
        self.assertIn("pop 50", optimizer_choices["evolutionary_balanced"].description)
        self.assertIn("mut 0.1", optimizer_choices["evolutionary_balanced"].description)
        self.assertIn("elite 2", optimizer_choices["evolutionary_balanced"].description)
        self.assertIn(nfn_impl.PARAMETER_GOLF_CASEOPS_MODEL_PRESET, model_choices)
        self.assertIn("11 layers", model_choices[nfn_impl.PARAMETER_GOLF_CASEOPS_MODEL_PRESET].description)
        self.assertIn("Parameter Golf 10min", run_choices["parameter_golf_10min"].label)
        self.assertIn("Parameter Golf Muon", optimizer_choices["parameter_golf_muon"].label)

    def test_parameter_golf_model_preset_fills_training_stack(self) -> None:
        state: dict[str, object] = {
            "base_model": "llama",
            "topology": "dense",
            "router_mode": "standard",
            "use_jepa": False,
            "megakernel": False,
            "model_preset": nfn_impl.PARAMETER_GOLF_CASEOPS_MODEL_PRESET,
        }
        resolved = nfn_impl.apply_recommended_presets("train", state, {"model_preset"})

        self.assertEqual("parameter_golf_10min", resolved["run_preset"])
        self.assertEqual("parameter_golf_muon", resolved["optimizer_preset"])
        self.assertEqual("sp8192", resolved["tokenizer"])
        self.assertEqual(8192, resolved["vocab_size"])
        self.assertEqual(11, resolved["num_layers"])
        self.assertEqual(512, resolved["model_dim"])
        self.assertEqual(8, resolved["num_heads"])
        self.assertEqual(4, resolved["num_kv_heads"])

    def test_next_visible_question_key_reveals_raw_text_tokenizer_after_dataset_choice(self) -> None:
        questions = nfn_impl.training_questionnaire(set())
        state: dict[str, object] = {
            "base_model": "llama",
            "topology": "dense",
            "router_mode": "standard",
            "use_jepa": False,
            "megakernel": False,
            "dataset": "shakespeare",
        }
        nfn_impl.normalize_dataset_selector_state(state)
        next_key = nfn_impl.next_visible_question_key(questions, state, set(), "dataset")
        self.assertEqual("tokenizer", next_key)

    def test_next_visible_question_key_reveals_cached_tokenizer_after_golf_dataset_choice(self) -> None:
        questions = nfn_impl.training_questionnaire(set())
        state: dict[str, object] = {
            "base_model": "llama",
            "topology": "dense",
            "router_mode": "standard",
            "use_jepa": False,
            "megakernel": False,
            "dataset": "golf1",
        }
        nfn_impl.normalize_dataset_selector_state(state)
        next_key = nfn_impl.next_visible_question_key(questions, state, set(), "dataset")
        self.assertEqual("tokenizer", next_key)

    def test_next_visible_question_key_returns_none_for_last_visible_question(self) -> None:
        questions = nfn_impl.training_questionnaire(set())
        state: dict[str, object] = {
            "base_model": "llama",
            "topology": "dense",
            "router_mode": "standard",
            "use_jepa": False,
            "megakernel": False,
            "dataset": "golf1",
            "model_preset": "harness_default",
            "run_preset": "default",
            "_action": "advanced",
        }
        nfn_impl.normalize_dataset_selector_state(state)
        next_key = nfn_impl.next_visible_question_key(questions, state, set(), "train_batch_tokens")
        self.assertIsNone(next_key)

    def test_maybe_plan_normalizes_tinystories_dataset_alias(self) -> None:
        resolved = nfn.maybe_plan(
            "train",
            {
                "dataset": "tinystories",
                "dataset_alias": nfn_impl.DEFAULT_DATASET_ALIAS,
            },
            {"dataset"},
            interactive=False,
        )
        self.assertEqual("tinystories", resolved["dataset"])
        self.assertEqual(
            nfn_impl.TINYSTORIES_DATASET_CONTRACT["dataset_alias"],
            resolved["dataset_alias"],
        )
        self.assertEqual(
            nfn_impl.TINYSTORIES_DATASET_CONTRACT["dataset_hf_path"],
            resolved["dataset_hf_path"],
        )
        self.assertEqual(
            nfn_impl.TINYSTORIES_DATASET_CONTRACT["dataset_train_file"],
            resolved["dataset_train_file"],
        )

    def test_maybe_plan_normalizes_parameter_golf_dataset_variant_alias(self) -> None:
        resolved = nfn.maybe_plan(
            "train",
            {
                "dataset": "golf10",
                "dataset_variant": "sp4096",
            },
            {"dataset", "dataset_variant"},
            interactive=False,
        )
        self.assertEqual("golf10", resolved["dataset"])
        self.assertEqual("sp4096", resolved["dataset_variant"])
        self.assertEqual(
            parameter_golf_dataset_alias(train_shards=10, variant="sp4096"),
            resolved["dataset_alias"],
        )
        self.assertEqual("willdepueoai/parameter-golf", resolved["dataset_hf_path"])
        self.assertEqual(10, resolved["dataset_train_shards"])

    def test_maybe_plan_with_pretraining_file_does_not_force_dataset(self) -> None:
        resolved = nfn.maybe_plan(
            "train",
            {
                "pretraining_file": "/tmp/corpus.txt",
            },
            {"pretraining_file"},
            interactive=False,
        )
        self.assertEqual("/tmp/corpus.txt", resolved["pretraining_file"])
        self.assertNotIn("dataset", resolved)
        self.assertIn("tokenizer", resolved)

    def test_state_to_cli_args_includes_canonical_tokenizer_flag(self) -> None:
        state = {
            "dataset": "golf1",
            "tokenizer": "sp8192",
            "dataset_alias": parameter_golf_dataset_alias(train_shards=1, variant="sp8192"),
        }
        argv = nfn_impl.state_to_cli_args("train", state)
        idx = argv.index("--tokenizer")
        self.assertEqual("sp8192", argv[idx + 1])

    def test_state_to_cli_args_includes_tokenizer_source_flags(self) -> None:
        argv = nfn_impl.state_to_cli_args(
            "train",
            {
                "pretraining_file": "/tmp/corpus.txt",
                "tokenizer": "sp8192",
                "tokenizer_hf_path": "sproos/parameter-golf-tokenizers",
                "tokenizer_repo_type": "model",
                "tokenizer_remote_root_prefix": "tokenizers",
            },
        )
        self.assertIn("--tokenizer-hf-path", argv)
        self.assertIn("sproos/parameter-golf-tokenizers", argv)
        self.assertIn("--tokenizer-repo-type", argv)
        self.assertIn("model", argv)
        self.assertIn("--tokenizer-remote-root-prefix", argv)
        self.assertIn("tokenizers", argv)

    def test_state_to_cli_args_includes_amp_dtype_flag(self) -> None:
        argv = nfn_impl.state_to_cli_args(
            "train",
            {
                "base_model": "llama",
                "topology": "dense",
                "amp_dtype": "float16",
            },
        )
        self.assertIn("--amp-dtype", argv)
        idx = argv.index("--amp-dtype")
        self.assertEqual("float16", argv[idx + 1])

    def test_state_to_cli_args_prefers_pretraining_file_over_dataset_alias(self) -> None:
        argv = nfn_impl.state_to_cli_args(
            "train",
            {
                "base_model": "gpt2",
                "pretraining_file": "/tmp/corpus.txt",
                "dataset_alias": "derived-adapter-path",
            },
        )
        self.assertIn("--pretraining-file", argv)
        self.assertNotIn("--dataset-alias", argv)

    def test_pretraining_file_is_train_only(self) -> None:
        train_args = nfn_impl.build_command_parser("train", style="long").parse_args(["--pretraining-file", "/tmp/corpus.txt"])
        self.assertEqual("/tmp/corpus.txt", train_args.pretraining_file)
        with self.assertRaises(SystemExit):
            nfn_impl.build_command_parser("infer", style="long").parse_args(["--pretraining-file", "/tmp/corpus.txt"])
        with self.assertRaises(SystemExit):
            nfn_impl.build_command_parser("eval", style="long").parse_args(["--pretraining-file", "/tmp/corpus.txt"])

    def test_amp_dtype_parser_accepts_supported_values(self) -> None:
        args = nfn_impl.build_command_parser("train", style="long").parse_args(["--amp-dtype", "float32"])
        self.assertEqual("float32", args.amp_dtype)
        with self.assertRaises(SystemExit):
            nfn_impl.build_command_parser("train", style="long").parse_args(["--amp-dtype", "fp32"])

    def test_infer_parser_accepts_top_p(self) -> None:
        args = nfn_impl.build_command_parser("infer", style="long").parse_args(["--top-p", "0.95"])
        self.assertAlmostEqual(0.95, args.top_p)

    def test_build_graph_for_training_defaults_to_float32_amp(self) -> None:
        resolved = nfn.maybe_plan(
            "train",
            {
                "base_model": "llama",
                "topology": "dense",
                "dataset": "golf1",
            },
            {"base_model", "topology", "dataset"},
            interactive=False,
        )
        recipe = nfn_impl.recipe_from_state(resolved)
        args = nfn_impl.namespace_from_state("train", resolved)
        nfn_impl.ensure_train_defaults(args, recipe)
        graph, _spec = nfn_impl.build_graph_for_training(args, recipe, "dummy_dataset")
        self.assertEqual("float32", graph.torch_config["amp_dtype"])

    def test_resolve_autocast_settings_disables_float32_autocast(self) -> None:
        class Graph:
            torch_config = {}

        amp_dtype, amp_name, use_amp = resolve_autocast_settings(Graph())
        self.assertEqual(torch.float32, amp_dtype)
        self.assertEqual("float32", amp_name)
        self.assertFalse(use_amp)

    def test_run_train_respects_no_download_for_missing_pretraining_sentencepiece(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = Path(tmpdir) / "corpus.txt"
            corpus.write_text("hello world\n" * 64, encoding="utf-8")
            state = nfn.maybe_plan(
                "train",
                {
                    "pretraining_file": str(corpus),
                    "tokenizer": "sp8192",
                    "raw_text_encoding_override": "sp8192",
                    "download_if_missing": False,
                    "eval_batches": 0,
                },
                {"pretraining_file", "tokenizer", "raw_text_encoding_override", "download_if_missing", "eval_batches"},
                interactive=False,
            )
            with patch.object(nfn_impl, "configure_console_logging"), patch("torch.cuda.is_available", return_value=True):
                with patch.object(
                    nfn_impl,
                    "validate_raw_text_tokenizer_availability",
                    side_effect=FileNotFoundError("Raw-text tokenizer 'sp8192' requires a shared sentencepiece model."),
                ), patch.object(
                    nfn_impl,
                    "resolve_or_download_dataset",
                    side_effect=AssertionError("dataset resolution should not run"),
                ):
                    with self.assertRaisesRegex(FileNotFoundError, "sp8192"):
                        nfn_impl.run_train(state)

    def test_run_train_saves_artifacts_before_validation_and_skips_eval_errors(self) -> None:
        state = nfn.maybe_plan(
            "train",
            {
                "base_model": "llama",
                "topology": "dense",
                "dataset": "golf1",
            },
            {"base_model", "topology", "dataset"},
            interactive=False,
        )

        class FakeGraph:
            input_node_ids = ["tokens", "targets"]
            torch_config = {}

        class FakeTrainer:
            def __init__(self, graph, trainer_cfg):
                self.graph = graph
                self.trainer_cfg = trainer_cfg

            def train(self, *_args, **_kwargs):
                return [1.0]

            def stop(self) -> None:
                return None

        fake_graph = FakeGraph()
        events: list[str] = []
        stdout = io.StringIO()
        stderr = io.StringIO()

        def fake_save_artifacts(*_args, **_kwargs) -> None:
            events.append("save")

        def fake_evaluate(*_args, **_kwargs) -> float:
            events.append("eval")
            raise FileNotFoundError("missing validation dataset")

        with patch.object(nfn_impl, "configure_console_logging"), patch("torch.cuda.is_available", return_value=True), patch.object(
            nfn_impl,
            "resolve_or_download_dataset",
            return_value=("cached_dataset", Path("/tmp/cached_dataset"), {}),
        ), patch.object(
            nfn_impl,
            "apply_cached_tokenizer_vocab_policy",
            side_effect=lambda args, **kwargs: dict(kwargs["dataset_meta"]),
        ), patch.object(
            nfn_impl,
            "estimate_text_schedule",
            return_value={"drop_last": True, "respect_epoch_boundaries": False},
        ), patch.object(
            nfn_impl,
            "resolve_effective_training_schedule",
            return_value=({"drop_last": True, "respect_epoch_boundaries": False}, 1, 1, 1, 60.0),
        ), patch.object(
            nfn_impl,
            "build_trainer_config",
            return_value=object(),
        ), patch.object(
            nfn_impl,
            "build_trainer_summary",
            return_value={},
        ), patch.object(
            nfn_impl,
            "build_graph_for_training",
            return_value=(fake_graph, {}),
        ), patch.object(
            nfn_impl,
            "print_graph_summary",
        ), patch.object(
            nfn_impl,
            "sanitized_model_spec_dict",
            return_value={},
        ), patch.object(
            nfn_impl,
            "build_progress_logger",
            return_value=(lambda *_args, **_kwargs: None, lambda *_args, **_kwargs: None),
        ), patch.object(
            nfn_impl,
            "TorchTrainer",
            FakeTrainer,
        ), patch.object(
            nfn_impl,
            "save_artifacts",
            side_effect=fake_save_artifacts,
        ), patch.object(
            nfn_impl,
            "evaluate_text_model",
            side_effect=fake_evaluate,
        ):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = nfn_impl.run_train(state)

        self.assertEqual(0, rc)
        self.assertEqual(["save", "eval"], events)
        self.assertIn("Validation loss: skipped", stdout.getvalue())
        self.assertIn("Validation skipped: missing validation dataset", stderr.getvalue())

    def test_build_trainer_config_uses_safe_evolutionary_defaults_when_disabled(self) -> None:
        planned = nfn.maybe_plan(
            "train",
            {"base_model": "llama"},
            {"base_model"},
            interactive=False,
        )
        args = nfn_impl.namespace_from_state("train", planned)
        trainer_cfg = nfn_impl.build_trainer_config(
            args,
            resolved_epochs=1,
            derived={"drop_last": True, "respect_epoch_boundaries": False},
        )
        self.assertFalse(trainer_cfg.evolutionary)
        self.assertEqual(50, trainer_cfg.evo_population_size)
        self.assertAlmostEqual(0.1, trainer_cfg.evo_mutation_rate)
        self.assertAlmostEqual(0.3, trainer_cfg.evo_mutation_scale)
        self.assertAlmostEqual(0.5, trainer_cfg.evo_crossover_rate)
        self.assertEqual(3, trainer_cfg.evo_tournament_size)
        self.assertEqual(2, trainer_cfg.evo_elite_count)

    def test_build_trainer_config_hydrates_tile_cuda_flags(self) -> None:
        planned = nfn.maybe_plan(
            "train",
            {
                "base_model": "llama",
                "seed": 1337,
                "kernel_backend": "tile-cuda",
                "tile_cuda_strict": True,
                "tile_cuda_report": "/tmp/nfn-tile-report.json",
            },
            {"base_model", "kernel_backend", "tile_cuda_strict", "tile_cuda_report"},
            interactive=False,
        )
        args = nfn_impl.namespace_from_state("train", planned)
        trainer_cfg = nfn_impl.build_trainer_config(
            args,
            resolved_epochs=1,
            derived={"drop_last": True, "respect_epoch_boundaries": False},
        )
        self.assertEqual("tile-cuda", trainer_cfg.kernel_backend)
        self.assertTrue(trainer_cfg.tile_cuda_strict)
        self.assertEqual("/tmp/nfn-tile-report.json", trainer_cfg.tile_cuda_report_path)

    def test_tile_cuda_backend_defaults_to_strict_kernels(self) -> None:
        args = nfn_impl.namespace_from_state(
            "train",
            {
                "base_model": "llama",
                "seed": 1337,
                "kernel_backend": "tile-cuda",
                "batch_size": 1,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "max_steps": 1,
                "train_batch_tokens": 1024,
                "warmup_steps": 0,
                "warmdown_fraction": 0.0,
                "max_wallclock_seconds": 0.0,
                "optimizer_profile": "adamw",
                "embed_lr": 1e-3,
                "head_lr": 1e-3,
                "tied_embed_lr": 1e-3,
                "matrix_lr": 1e-3,
                "scalar_lr": 1e-3,
                "muon_momentum": 0.95,
                "muon_backend_steps": 5,
                "muon_momentum_warmup_start": 0.85,
                "muon_momentum_warmup_steps": 64,
                "beta1": 0.9,
                "beta2": 0.95,
                "adam_eps": 1e-8,
                "grad_clip_norm": 1.0,
            },
        )
        trainer_cfg = nfn_impl.build_trainer_config(
            args,
            resolved_epochs=1,
            derived={"drop_last": True, "respect_epoch_boundaries": False},
        )
        self.assertTrue(trainer_cfg.tile_cuda_strict)

        args.tile_cuda_strict = False
        trainer_cfg = nfn_impl.build_trainer_config(
            args,
            resolved_epochs=1,
            derived={"drop_last": True, "respect_epoch_boundaries": False},
        )
        self.assertFalse(trainer_cfg.tile_cuda_strict)

    def test_namespace_from_state_hydrates_train_parser_defaults(self) -> None:
        args = nfn_impl.namespace_from_state(
            "train",
            {
                "base_model": "nanogpt",
                "topology": "dense",
                "dataset": "golf10",
            },
        )
        self.assertTrue(hasattr(args, "dataset_hf_path"))
        self.assertTrue(hasattr(args, "dataset_variant"))
        self.assertTrue(hasattr(args, "dataset_train_file"))
        self.assertTrue(hasattr(args, "dataset_val_file"))
        self.assertEqual("cuda", args.device)
        self.assertIsNone(args.dataset_hf_path)
        self.assertEqual("golf10", args.dataset)

    def test_dataset_download_kwargs_from_args_tolerates_sparse_namespace(self) -> None:
        kwargs = dataset_download_kwargs_from_args(argparse.Namespace(dataset_alias="custom_alias"))
        self.assertFalse(kwargs["download_if_missing"])
        self.assertIsNone(kwargs["dataset_hf_path"])
        self.assertIsNone(kwargs["dataset_variant"])
        self.assertIsNone(kwargs["dataset_train_file"])
        self.assertIsNone(kwargs["dataset_val_file"])
        self.assertIsNone(kwargs["tokenizer_hf_path"])
        self.assertIsNone(kwargs["tokenizer_repo_id"])
        self.assertIsNone(kwargs["tokenizer_remote_root_prefix"])
        self.assertIsNone(kwargs["tokenizer_repo_type"])

    def test_plan_auto_recommends_base_model_then_presets(self) -> None:
        resolved = nfn.maybe_plan("train", {"plan_auto": True}, set(), interactive=False)
        self.assertEqual("llama", resolved["base_model"])
        self.assertEqual("dense", resolved["topology"])
        self.assertFalse(resolved["use_jepa"])
        self.assertEqual("harness_default", resolved["model_preset"])
        self.assertEqual("default", resolved["run_preset"])
        self.assertEqual("gradient_default", resolved["optimizer_preset"])
        self.assertEqual("golf1", resolved["dataset"])

    def test_plan_auto_preserves_full_combo_and_fills_presets(self) -> None:
        resolved = nfn.maybe_plan(
            "train",
            {
                "base_model": "gpt2",
                "topology": "moe",
                "router_mode": "semantic",
                "use_jepa": True,
                "megakernel": True,
            },
            {"base_model", "topology", "router_mode", "use_jepa", "megakernel"},
            interactive=False,
        )
        self.assertEqual("gpt2", resolved["base_model"])
        self.assertEqual("semantic", resolved["router_mode"])
        self.assertTrue(resolved["use_jepa"])
        self.assertTrue(resolved["megakernel"])
        self.assertIn("model_preset", resolved)
        self.assertIn("run_preset", resolved)
        self.assertIn("optimizer_preset", resolved)

    def test_main_plan_auto_prints_resolved_command_and_dispatches(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        def fake_execute(command: str, state: dict[str, object]) -> int:
            calls.append((command, dict(state)))
            return 0

        stdout = io.StringIO()
        with patch.object(nfn_impl, "execute", side_effect=fake_execute):
            with redirect_stdout(stdout):
                rc = nfn.main(["train", "--plan-auto"], stdin_isatty=False, stdout_isatty=False)
        self.assertEqual(0, rc)
        self.assertEqual("train", calls[0][0])
        self.assertIn("Resolved command:", stdout.getvalue())
        self.assertIn("--base-model", stdout.getvalue())
        self.assertEqual("llama", calls[0][1]["base_model"])

    def test_main_plan_auto_dispatches_tinystories_with_normalized_alias(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        def fake_execute(command: str, state: dict[str, object]) -> int:
            calls.append((command, dict(state)))
            return 0

        with patch.object(nfn_impl, "execute", side_effect=fake_execute):
            rc = nfn.main(
                ["train", "--plan-auto", "--dataset", "tinystories"],
                stdin_isatty=False,
                stdout_isatty=False,
            )
        self.assertEqual(0, rc)
        self.assertEqual("train", calls[0][0])
        self.assertEqual("tinystories", calls[0][1]["dataset"])
        self.assertEqual(
            nfn_impl.TINYSTORIES_DATASET_CONTRACT["dataset_alias"],
            calls[0][1]["dataset_alias"],
        )

    def test_main_infer_graph_dispatches_without_planner(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        def fake_execute(command: str, state: dict[str, object]) -> int:
            calls.append((command, dict(state)))
            return 0

        with patch.object(nfn_impl, "execute", side_effect=fake_execute), patch.object(
            nfn_impl,
            "maybe_plan",
            side_effect=AssertionError("infer should not route through maybe_plan"),
        ):
            rc = nfn.main(["infer", "--graph", "/tmp/model.json", "--base-model", "nanogpt"], stdin_isatty=False, stdout_isatty=False)
        self.assertEqual(0, rc)
        self.assertEqual("infer", calls[0][0])
        self.assertEqual("/tmp/model.json", calls[0][1]["graph"])
        self.assertEqual("nanogpt", calls[0][1]["base_model"])
        self.assertFalse(calls[0][1]["_tty"])

    def test_main_infer_checkpoint_dispatches_without_planner(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        def fake_execute(command: str, state: dict[str, object]) -> int:
            calls.append((command, dict(state)))
            return 0

        with patch.object(nfn_impl, "execute", side_effect=fake_execute), patch.object(
            nfn_impl,
            "maybe_plan",
            side_effect=AssertionError("infer should not route through maybe_plan"),
        ):
            rc = nfn.main(
                [
                    "infer",
                    "--checkpoint",
                    "/tmp/final_model.pt",
                    "--checkpoint-tokenizer",
                    "/tmp/tokenizer.model",
                    "--checkpoint-log",
                    "/tmp/train.log",
                    "--prompt",
                    "hi",
                ],
                stdin_isatty=False,
                stdout_isatty=False,
            )
        self.assertEqual(0, rc)
        self.assertEqual("infer", calls[0][0])
        self.assertEqual("/tmp/final_model.pt", calls[0][1]["checkpoint"])
        self.assertEqual("/tmp/tokenizer.model", calls[0][1]["checkpoint_tokenizer"])
        self.assertEqual("/tmp/train.log", calls[0][1]["checkpoint_log"])
        self.assertEqual("hi", calls[0][1]["prompt"])
        self.assertFalse(calls[0][1]["_tty"])

    def test_main_infer_plan_flags_are_ignored(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        def fake_execute(command: str, state: dict[str, object]) -> int:
            calls.append((command, dict(state)))
            return 0

        stderr = io.StringIO()
        with patch.object(nfn_impl, "execute", side_effect=fake_execute), redirect_stderr(stderr):
            rc = nfn.main(["infer", "--graph", "/tmp/model.json", "--plan-auto"], stdin_isatty=False, stdout_isatty=False)
        self.assertEqual(0, rc)
        self.assertEqual("infer", calls[0][0])
        self.assertIn("artifact-first; ignoring --plan/--plan-auto", stderr.getvalue())

    def test_main_infer_without_graph_fails_cleanly_when_noninteractive(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            rc = nfn.main(["infer", "--prompt", "hello"], stdin_isatty=False, stdout_isatty=False)
        self.assertEqual(1, rc)
        self.assertIn("Non-interactive infer requires --graph or --checkpoint/--weights.", stderr.getvalue())

    def test_resolve_graphless_checkpoint_path_accepts_weights_without_graph(self) -> None:
        args = argparse.Namespace(graph=None, checkpoint=None, weights="~/NeuralFn/artifacts/final_model.pt")
        resolved = nfn_impl.resolve_graphless_checkpoint_path(args)
        self.assertEqual(Path("~/NeuralFn/artifacts/final_model.pt").expanduser().resolve(), resolved)

    def test_parameter_golf_generation_suppresses_artifact_tokens(self) -> None:
        logits = torch.tensor([[1.0, 9.0, 2.0, 8.0]])
        masked = nfn_impl.suppress_parameter_golf_token_ids(logits, token_ids=(1, 3), stop_token=None)
        self.assertTrue(torch.isneginf(masked[0, 1]))
        self.assertTrue(torch.isneginf(masked[0, 3]))
        self.assertEqual(2, int(torch.argmax(masked, dim=-1).item()))

        with_stop = nfn_impl.suppress_parameter_golf_token_ids(logits, token_ids=(1, 3), stop_token=3)
        self.assertTrue(torch.isneginf(with_stop[0, 1]))
        self.assertFalse(torch.isneginf(with_stop[0, 3]))

    def test_parameter_golf_repeat_bans_catches_runs_and_ngrams(self) -> None:
        self.assertEqual(
            [7],
            nfn_impl.parameter_golf_repeat_bans(
                [1, 2, 7, 7, 7],
                repeat_run_limit=3,
                no_repeat_ngram_size=0,
            ),
        )
        self.assertEqual(
            [3],
            nfn_impl.parameter_golf_repeat_bans(
                [1, 2, 3, 1, 2],
                repeat_run_limit=0,
                no_repeat_ngram_size=3,
            ),
        )

    def test_caseops_decode_sanitizer_hides_private_use_markers(self) -> None:
        self.assertEqual(
            "hello world",
            sanitize_caseops_decoded_text("\ue001hello   \ufffdworld\ue004"),
        )

    def test_caseops_tokenizer_suppresses_lossless_artifact_pieces(self) -> None:
        class FakeProcessor:
            pieces = ["<pad>", "<s>", "<0xE2>", "...", "▁...", "word", "", "a", "b"]

            def vocab_size(self):
                return len(self.pieces)

            def is_unknown(self, token_id):
                return False

            def is_unused(self, token_id):
                return False

            def is_byte(self, token_id):
                return token_id == 2

            def id_to_piece(self, token_id):
                return self.pieces[token_id]

            def pad_id(self):
                return 0

            def bos_id(self):
                return 1

        tokenizer = ParameterGolfSentencePieceTokenizer(
            FakeProcessor(),
            tokenizer_path=Path("caseops.model"),
            caseops=True,
        )
        self.assertIn(2, tokenizer.suppressed_token_ids)
        self.assertIn(3, tokenizer.suppressed_token_ids)
        self.assertIn(4, tokenizer.suppressed_token_ids)
        self.assertIn(6, tokenizer.suppressed_token_ids)
        self.assertNotIn(5, tokenizer.suppressed_token_ids)
        self.assertIn(7, tokenizer.lossless_fallback_token_ids)
        self.assertIn(8, tokenizer.lossless_fallback_token_ids)

    def test_run_infer_noninteractive_graph_prompt_uses_single_turn_flow(self) -> None:
        class FakeTokenizer:
            def encode(self, text, out_type=int):
                return [ord(ch) for ch in text]

            def decode(self, token_ids):
                return "".join(chr(int(token_id)) for token_id in token_ids)

        fake_context = nfn_impl.InferRuntimeContext(
            args=argparse.Namespace(
                repetition_penalty=1.0,
                log_every=0,
                stop_token=None,
                logits_node="auto",
                seed=1337,
            ),
            graph_path=Path("/tmp/model.json"),
            resolved_weights_path=Path("/tmp/model.pt"),
            graph=SimpleNamespace(nodes={}),
            compiled=object(),
            state_dict={},
            tokenizer=FakeTokenizer(),
            tokenizer_path=None,
            tokenizer_name="fake",
            raw_text_encoding_name="fake",
            dataset_alias="fake",
            device=torch.device("cpu"),
            generator=torch.Generator(),
            amp_dtype=torch.float32,
            amp_name="float32",
            context_window=32,
        )
        stdout = io.StringIO()

        with patch.object(nfn_impl, "configure_console_logging"), patch.object(
            nfn_impl,
            "build_infer_runtime_context",
            return_value=fake_context,
        ), patch.object(
            nfn_impl,
            "build_infer_generation",
            return_value=(
                {
                    "generated_token_ids": [33],
                    "all_token_ids": [104, 105, 33],
                    "generated_text": "!",
                    "full_text": "hi!",
                    "resolved_logits_key": "model/lm_head",
                },
                {},
            ),
        ):
            with redirect_stdout(stdout):
                rc = nfn_impl.run_infer({"graph": "/tmp/model.json", "prompt": "hi", "_tty": False})

        self.assertEqual(0, rc)
        rendered = stdout.getvalue()
        self.assertIn("Prompt token ids: [104, 105]", rendered)
        self.assertIn("Generated text:", rendered)
        self.assertIn("hi!", rendered)

    def test_run_infer_noninteractive_parameter_golf_checkpoint_generates(self) -> None:
        class FakeTokenizer:
            def encode(self, text, out_type=int):
                return [int(part) for part in text.split(",") if part]

            def decode(self, token_ids):
                return "<" + ",".join(str(int(token_id)) for token_id in token_ids) + ">"

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_path = root / "tiny_parameter_golf.pt"
            tokenizer_path = root / "tiny.model"
            config = ParameterGolfConfig(
                vocab_size=16,
                num_layers=2,
                model_dim=8,
                num_heads=2,
                num_kv_heads=1,
                mlp_mult=2,
                tie_embeddings=True,
                context_window=8,
            )
            torch.save(ParameterGolfGPT(config).state_dict(), checkpoint_path)
            stdout = io.StringIO()

            with patch.object(nfn_impl, "configure_console_logging"), patch.object(
                nfn_impl,
                "resolve_parameter_golf_tokenizer_path",
                return_value=tokenizer_path,
            ), patch.object(
                nfn_impl,
                "load_sentencepiece_tokenizer",
                return_value=FakeTokenizer(),
            ):
                with redirect_stdout(stdout):
                    rc = nfn_impl.run_infer(
                        {
                            "checkpoint": str(checkpoint_path),
                            "prompt_tokens": "1,2",
                            "device": "cpu",
                            "max_new_tokens": 1,
                            "temperature": 0.0,
                            "top_k": 0,
                            "top_p": 1.0,
                            "_tty": False,
                        }
                    )

        self.assertEqual(0, rc)
        rendered = stdout.getvalue()
        self.assertIn("Prompt token ids: [1, 2]", rendered)
        self.assertIn("Generated token ids:", rendered)
        self.assertIn("Full text:", rendered)

    def test_infer_graph_picker_filters_eval_reports_and_missing_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            good_graph = artifacts / "good.json"
            good_weights = artifacts / "good.pt"
            missing_graph = artifacts / "missing.json"
            eval_report = artifacts / "stale.eval.json"
            good_graph.write_text("{}", encoding="utf-8")
            good_weights.write_bytes(b"pt")
            missing_graph.write_text("{}", encoding="utf-8")
            eval_report.write_text("{}", encoding="utf-8")

            def fake_load_graph(path: Path):
                stem = Path(path).stem
                return SimpleNamespace(
                    name=stem,
                    torch_config={
                        "artifact_metadata": {"weights_file": f"{stem}.pt"},
                        "tokenizer_manifest": {"encoding_name": "sp8192"},
                        "template_spec": {"template": {"runtime": "compile"}},
                    },
                    nodes={
                        "dataset_source": SimpleNamespace(
                            neuron_def=SimpleNamespace(module_config={"seq_len": 128})
                        )
                    },
                )

            with patch.object(nfn_impl, "artifact_root", return_value=artifacts), patch.object(nfn_impl, "load_graph", side_effect=fake_load_graph):
                options = nfn_impl.infer_graph_picker_options()

        labels = [option.label for option in options]
        self.assertIn("good.json", labels)
        self.assertNotIn("missing.json", labels)
        self.assertNotIn("stale.eval.json", labels)
        self.assertEqual("Custom path...", labels[-1])

    def test_warn_ignored_infer_recipe_overrides_lists_recipe_flags(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            nfn_impl.warn_ignored_infer_recipe_overrides(
                {
                    "base_model": "nanogpt",
                    "topology": "moe",
                    "router_mode": "semantic",
                    "use_jepa": True,
                },
                Path("/tmp/model.json"),
            )
        warning = stderr.getvalue()
        self.assertIn("--base-model", warning)
        self.assertIn("--topology", warning)
        self.assertIn("--router-mode", warning)
        self.assertIn("--jepa", warning)

    def test_resolve_infer_chat_prompt_drops_oldest_turns_first(self) -> None:
        class FakeTokenizer:
            def encode(self, text, out_type=int):
                return [ord(ch) for ch in text]

            def decode(self, token_ids):
                return "".join(chr(int(token_id)) for token_id in token_ids)

        context = nfn_impl.InferRuntimeContext(
            args=argparse.Namespace(repetition_penalty=1.0, log_every=0, stop_token=None, logits_node="auto", seed=1337),
            graph_path=Path("/tmp/model.json"),
            resolved_weights_path=Path("/tmp/model.pt"),
            graph=SimpleNamespace(nodes={}),
            compiled=object(),
            state_dict={},
            tokenizer=FakeTokenizer(),
            tokenizer_path=None,
            tokenizer_name="fake",
            raw_text_encoding_name="fake",
            dataset_alias="fake",
            device=torch.device("cpu"),
            generator=torch.Generator(),
            amp_dtype=torch.float32,
            amp_name="float32",
            context_window=90,
        )

        prompt_text, _prompt_ids, dropped_turns = nfn_impl.resolve_infer_chat_prompt(
            context,
            mode="transcript",
            history=[
                ("aaaaaaaaaaaa", "bbbbbbbbbbbb"),
                ("cccccccccccc", "dddddddddddd"),
            ],
            draft="eeeeeeeeeeee",
            include_assistant_prompt=True,
        )

        self.assertEqual(1, dropped_turns)
        self.assertNotIn("aaaaaaaaaaaa", prompt_text)
        self.assertIn("cccccccccccc", prompt_text)

    def _fake_os_read_feeder(self, byte_sequence: bytes):
        chunks = [byte_sequence[i:i + 1] for i in range(len(byte_sequence))]
        iterator = iter(chunks)

        def fake_read(_fd: int, _n: int) -> bytes:
            return next(iterator)

        return fake_read

    def test_read_infer_tty_key_decodes_4_byte_emoji(self) -> None:
        fake = self._fake_os_read_feeder("🎉".encode("utf-8"))
        with patch.object(nfn_impl.os, "read", side_effect=fake):
            key = nfn_impl._read_infer_tty_key(0)
        self.assertEqual("🎉", key)

    def test_read_infer_tty_key_decodes_2_byte_accent(self) -> None:
        fake = self._fake_os_read_feeder("é".encode("utf-8"))
        with patch.object(nfn_impl.os, "read", side_effect=fake):
            key = nfn_impl._read_infer_tty_key(0)
        self.assertEqual("é", key)

    def test_read_infer_tty_key_decodes_3_byte_cjk(self) -> None:
        fake = self._fake_os_read_feeder("あ".encode("utf-8"))
        with patch.object(nfn_impl.os, "read", side_effect=fake):
            key = nfn_impl._read_infer_tty_key(0)
        self.assertEqual("あ", key)

    def test_read_infer_tty_key_returns_empty_on_invalid_utf8(self) -> None:
        fake = self._fake_os_read_feeder(b"\xff\x00")
        with patch.object(nfn_impl.os, "read", side_effect=fake):
            key = nfn_impl._read_infer_tty_key(0)
        self.assertEqual("", key)

    def test_complete_infer_slash_command_unique_match_adds_trailing_space(self) -> None:
        new_buffer, _status = nfn_impl.complete_infer_slash_command("/te")
        self.assertEqual("/temp ", new_buffer)

    def test_complete_infer_slash_command_settings_prefix(self) -> None:
        new_buffer, _status = nfn_impl.complete_infer_slash_command("/sett")
        self.assertEqual("/settings", new_buffer)

    def test_complete_infer_slash_command_no_value_no_trailing_space(self) -> None:
        new_buffer, _status = nfn_impl.complete_infer_slash_command("/he")
        self.assertEqual("/help", new_buffer)

    def test_infer_slash_status_for_buffer_suggests_settings_prefix(self) -> None:
        status = nfn_impl.infer_slash_status_for_buffer("/sett")
        self.assertIsNotNone(status)
        self.assertIn("/settings", status)
        self.assertIn("Open interactive settings menu", status)

    def test_infer_slash_status_for_buffer_lists_multiple_matches(self) -> None:
        status = nfn_impl.infer_slash_status_for_buffer("/t")
        self.assertIsNotNone(status)
        self.assertIn("/temp", status)
        self.assertIn("/top_k", status)
        self.assertIn("/top_p", status)

    def test_infer_slash_status_for_buffer_unknown_prefix(self) -> None:
        status = nfn_impl.infer_slash_status_for_buffer("/xyzzy")
        self.assertIsNotNone(status)
        self.assertIn("No command matches", status)

    def test_infer_slash_status_for_buffer_value_hint(self) -> None:
        status = nfn_impl.infer_slash_status_for_buffer("/temp")
        self.assertIsNotNone(status)
        self.assertIn("<float>", status)

    def test_complete_infer_slash_command_lists_multiple_matches(self) -> None:
        new_buffer, status = nfn_impl.complete_infer_slash_command("/t")
        self.assertIsNone(new_buffer)
        self.assertIn("/temp", status)
        self.assertIn("/top_k", status)
        self.assertIn("/top_p", status)

    def test_complete_infer_slash_command_unknown_prefix(self) -> None:
        new_buffer, status = nfn_impl.complete_infer_slash_command("/xyzzy")
        self.assertIsNone(new_buffer)
        self.assertIn("No command matches", status)

    def test_complete_infer_slash_command_already_has_value(self) -> None:
        new_buffer, status = nfn_impl.complete_infer_slash_command("/temp 0.")
        self.assertIsNone(new_buffer)
        self.assertIn("<float>", status)

    def test_complete_infer_slash_command_alias(self) -> None:
        new_buffer, _status = nfn_impl.complete_infer_slash_command("/temperatu")
        self.assertEqual("/temperature ", new_buffer)

    def test_complete_infer_slash_command_autocomplete_prefix(self) -> None:
        new_buffer, _status = nfn_impl.complete_infer_slash_command("/auto")
        self.assertEqual("/autocomplete ", new_buffer)

    def test_apply_infer_setting_command_updates_temperature(self) -> None:
        settings = nfn_impl.InferChatSettings(
            top_k=32, top_p=0.95, temperature=0.8, max_new_tokens=64
        )
        result = nfn_impl.apply_infer_setting_command("/temp 0.5", settings)
        self.assertIsNotNone(result)
        updated, _status = result
        self.assertAlmostEqual(0.5, updated.temperature)
        self.assertEqual(32, updated.top_k)

    def test_apply_infer_setting_command_parses_int_top_k(self) -> None:
        settings = nfn_impl.InferChatSettings(
            top_k=32, top_p=0.95, temperature=0.8, max_new_tokens=64
        )
        result = nfn_impl.apply_infer_setting_command("/top_k 12", settings)
        self.assertIsNotNone(result)
        updated, _status = result
        self.assertEqual(12, updated.top_k)
        self.assertIsInstance(updated.top_k, int)

    def test_apply_infer_setting_command_alias_max_new(self) -> None:
        settings = nfn_impl.InferChatSettings(
            top_k=32, top_p=0.95, temperature=0.8, max_new_tokens=64
        )
        result = nfn_impl.apply_infer_setting_command("/max_new_tokens 128", settings)
        self.assertIsNotNone(result)
        updated, _status = result
        self.assertEqual(128, updated.max_new_tokens)

    def test_apply_infer_setting_command_updates_repetition_penalty(self) -> None:
        settings = nfn_impl.InferChatSettings(
            top_k=32, top_p=0.95, temperature=0.8, max_new_tokens=64
        )
        result = nfn_impl.apply_infer_setting_command("/repeat 1.15", settings)
        self.assertIsNotNone(result)
        updated, _status = result
        self.assertAlmostEqual(1.15, updated.repetition_penalty)

    def test_apply_infer_setting_command_updates_autocomplete_words(self) -> None:
        settings = nfn_impl.InferChatSettings(
            top_k=32, top_p=0.95, temperature=0.8, max_new_tokens=64
        )
        result = nfn_impl.apply_infer_setting_command("/autocomplete 3", settings)
        self.assertIsNotNone(result)
        updated, _status = result
        self.assertEqual(3, updated.autocomplete_words)

    def test_apply_infer_setting_command_rejects_negative_autocomplete_words(self) -> None:
        settings = nfn_impl.InferChatSettings(
            top_k=32, top_p=0.95, temperature=0.8, max_new_tokens=64
        )
        result = nfn_impl.apply_infer_setting_command("/autocomplete -1", settings)
        self.assertIsNotNone(result)
        updated, status = result
        self.assertEqual(settings, updated)
        self.assertIn("Invalid", status)

    def test_apply_infer_setting_command_invalid_value_preserves_settings(self) -> None:
        settings = nfn_impl.InferChatSettings(
            top_k=32, top_p=0.95, temperature=0.8, max_new_tokens=64
        )
        result = nfn_impl.apply_infer_setting_command("/temp banana", settings)
        self.assertIsNotNone(result)
        updated, status = result
        self.assertEqual(settings, updated)
        self.assertIn("Invalid", status)

    def test_apply_infer_setting_command_query_form_without_value(self) -> None:
        settings = nfn_impl.InferChatSettings(
            top_k=32, top_p=0.95, temperature=0.8, max_new_tokens=64
        )
        result = nfn_impl.apply_infer_setting_command("/temp", settings)
        self.assertIsNotNone(result)
        updated, status = result
        self.assertEqual(settings, updated)
        self.assertIn("0.8", status)

    def test_apply_infer_setting_command_returns_none_for_non_setter(self) -> None:
        settings = nfn_impl.InferChatSettings(
            top_k=32, top_p=0.95, temperature=0.8, max_new_tokens=64
        )
        self.assertIsNone(nfn_impl.apply_infer_setting_command("/help", settings))
        self.assertIsNone(nfn_impl.apply_infer_setting_command("hello world", settings))

    def test_trim_infer_autocomplete_words_preserves_leading_space(self) -> None:
        self.assertEqual(
            " quick brown",
            nfn_impl.trim_infer_autocomplete_words(" quick brown fox jumps", 2),
        )

    def test_build_infer_inline_autocomplete_trims_to_word_count(self) -> None:
        class FakeTokenizer:
            def encode(self, text, out_type=int):
                return [ord(ch) for ch in text]

            def decode(self, token_ids):
                return "".join(chr(int(token_id)) for token_id in token_ids)

        context = nfn_impl.InferRuntimeContext(
            args=argparse.Namespace(repetition_penalty=1.0, log_every=0, stop_token=None, logits_node="auto", seed=1337),
            graph_path=Path("/tmp/model.json"),
            resolved_weights_path=Path("/tmp/model.pt"),
            graph=SimpleNamespace(nodes={}),
            compiled=object(),
            state_dict={},
            tokenizer=FakeTokenizer(),
            tokenizer_path=None,
            tokenizer_name="fake",
            raw_text_encoding_name="fake",
            dataset_alias="fake",
            device=torch.device("cpu"),
            generator=torch.Generator(),
            amp_dtype=torch.float32,
            amp_name="float32",
            context_window=90,
        )
        settings = nfn_impl.InferChatSettings(
            top_k=32,
            top_p=0.95,
            temperature=0.8,
            max_new_tokens=64,
            autocomplete_words=2,
        )
        with patch.object(
            nfn_impl,
            "build_infer_generation",
            return_value=({"generated_text": " quick brown fox jumps"}, {}),
        ):
            suggestion, dropped_turns = nfn_impl.build_infer_inline_autocomplete(
                context,
                settings=settings,
                mode="stateless",
                history=[],
                buffer_text="the",
            )

        self.assertEqual(0, dropped_turns)
        self.assertIsNotNone(suggestion)
        self.assertEqual(" quick brown", suggestion.text)
        self.assertTrue(suggestion.insertable)

    def test_build_infer_inline_autocomplete_preserves_unspaced_continuation(self) -> None:
        class FakeTokenizer:
            def encode(self, text, out_type=int):
                return [ord(ch) for ch in text]

            def decode(self, token_ids):
                return "".join(chr(int(token_id)) for token_id in token_ids)

        context = nfn_impl.InferRuntimeContext(
            args=argparse.Namespace(repetition_penalty=1.0, log_every=0, stop_token=None, logits_node="auto", seed=1337),
            graph_path=Path("/tmp/model.json"),
            resolved_weights_path=Path("/tmp/model.pt"),
            graph=SimpleNamespace(nodes={}),
            compiled=object(),
            state_dict={},
            tokenizer=FakeTokenizer(),
            tokenizer_path=None,
            tokenizer_name="fake",
            raw_text_encoding_name="fake",
            dataset_alias="fake",
            device=torch.device("cpu"),
            generator=torch.Generator(),
            amp_dtype=torch.float32,
            amp_name="float32",
            context_window=90,
        )
        settings = nfn_impl.InferChatSettings(
            top_k=32,
            top_p=0.95,
            temperature=0.8,
            max_new_tokens=64,
            autocomplete_words=2,
        )
        with patch.object(
            nfn_impl,
            "build_infer_generation",
            return_value=({"generated_text": "quick brown fox jumps"}, {}),
        ):
            suggestion, _dropped_turns = nfn_impl.build_infer_inline_autocomplete(
                context,
                settings=settings,
                mode="stateless",
                history=[],
                buffer_text="the",
            )

        self.assertIsNotNone(suggestion)
        self.assertEqual("quick brown", suggestion.text)

    def test_render_infer_input_line_appends_inline_ghost(self) -> None:
        output = io.StringIO()
        console = nfn_impl.Console(
            file=io.StringIO(),
            width=80,
            color_system=None,
            theme=nfn_impl.INFER_THEME,
        )
        with patch.object(nfn_impl.sys, "stdout", output):
            nfn_impl._render_infer_input_line(
                console,
                "You ",
                "the",
                3,
                "status",
                ghost_text=" cat",
            )
        self.assertIn("the cat", output.getvalue())
        self.assertIn("\r\n", output.getvalue())

    def test_render_infer_input_line_tracks_wrapped_rows(self) -> None:
        output = io.StringIO()
        console = nfn_impl.Console(
            file=io.StringIO(),
            width=16,
            color_system=None,
            theme=nfn_impl.INFER_THEME,
            soft_wrap=False,
        )
        with patch.object(nfn_impl.sys, "stdout", output):
            rows, cursor_row = nfn_impl._render_infer_input_line(
                console,
                "You ",
                "this lead me to a place called the house",
                len("this lead me to a place called the house"),
                "status",
                ghost_text=" of the sinner",
            )

        self.assertGreater(rows, 2)
        self.assertGreater(cursor_row, 0)
        self.assertNotIn(nfn_impl.INFER_INPUT_CURSOR_MARK, output.getvalue())

    def test_render_infer_input_line_clears_previous_wrapped_rows(self) -> None:
        output = io.StringIO()
        console = nfn_impl.Console(
            file=io.StringIO(),
            width=16,
            color_system=None,
            theme=nfn_impl.INFER_THEME,
            soft_wrap=False,
        )
        with patch.object(nfn_impl.sys, "stdout", output):
            rows, cursor_row = nfn_impl._render_infer_input_line(
                console,
                "You ",
                "this lead me to a place called the house",
                len("this lead me to a place called the house"),
                "status",
                ghost_text=" of the sinner",
            )
            output.seek(0)
            output.truncate(0)
            nfn_impl._render_infer_input_line(
                console,
                "You ",
                "short",
                len("short"),
                "status",
                previous_rows=rows,
                previous_cursor_row=cursor_row,
            )

        self.assertGreater(rows, 2)
        self.assertGreaterEqual(output.getvalue().count("\033[2K"), rows)
        self.assertIn("\r\n", output.getvalue())

    def test_read_infer_chat_line_tab_accepts_inline_autocomplete(self) -> None:
        settings = nfn_impl.InferChatSettings(
            top_k=32,
            top_p=0.95,
            temperature=0.8,
            max_new_tokens=64,
            autocomplete_words=1,
        )
        suggestion = nfn_impl.InferInlineAutocomplete(
            text="ello",
            display_text="ello",
            insertable=True,
            buffer_snapshot="h",
            mode="stateless",
            settings_signature=nfn_impl.infer_settings_signature(settings),
        )
        fake_stdin = SimpleNamespace(fileno=lambda: 0)
        console = nfn_impl.Console(
            file=io.StringIO(),
            width=80,
            color_system=None,
            theme=nfn_impl.INFER_THEME,
        )
        with (
            patch.object(nfn_impl.sys, "stdin", fake_stdin),
            patch.object(nfn_impl.sys, "stdout", io.StringIO()),
            patch.object(nfn_impl.termios, "tcgetattr", return_value=[]),
            patch.object(nfn_impl.termios, "tcsetattr"),
            patch.object(nfn_impl.tty_module, "setraw"),
            patch.object(nfn_impl, "_read_infer_tty_key", side_effect=["h", "\t", "\n"]),
            patch.object(
                nfn_impl,
                "build_infer_inline_autocomplete",
                return_value=(suggestion, 0),
            ),
        ):
            line = nfn_impl.read_infer_chat_line(
                SimpleNamespace(),
                settings=settings,
                mode="stateless",
                history=[],
                console=console,
            )

        self.assertEqual("hello", line)

    def test_render_infer_help_table_contains_emoji_title(self) -> None:
        table = nfn_impl.render_infer_help_table()
        console = nfn_impl.Console(
            record=True,
            width=80,
            color_system=None,
            theme=nfn_impl.INFER_THEME,
        )
        console.print(table)
        rendered = console.export_text()
        self.assertIn("/help", rendered)
        self.assertIn("/clear", rendered)
        self.assertIn("Tab", rendered)


if __name__ == "__main__":
    unittest.main()
