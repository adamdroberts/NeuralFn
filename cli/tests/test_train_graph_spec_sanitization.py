from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
NEURALFN_ROOT = ROOT.parent
SCRIPTS_DIR = ROOT / "scripts"
for candidate in (ROOT, NEURALFN_ROOT, SCRIPTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import nfn
import nfn_impl
import train_jepa_semantic
from neuralfn.config import build_composed_lm_spec, model_spec_to_dict
from neuralfn.semantic import NUM_VOCAB_DIMS
from neuralfn.torch_templates import build_gpt_root_graph
from server.dataset_manager import raw_text_encoding_name_for_template_spec


LLAMA_COMMON_KW = {
    "vocab_size": 128,
    "num_layers": 2,
    "model_dim": 64,
    "num_heads": 4,
    "num_kv_heads": 4,
    "mlp_mult": 2.0,
    "multiple_of": 64,
    "rope_base": 10_000.0,
    "qk_gain_init": 1.0,
    "logit_softcap": 0.0,
}
JEPA_ONLY_FIELDS = (
    "jepa_latent_dim",
    "jepa_mask_ratio",
    "jepa_mask_strategy",
    "jepa_num_blocks",
    "jepa_min_block_ratio",
    "jepa_max_block_ratio",
    "ema_decay",
    "jepa_loss_coef",
)
SEMANTIC_ONLY_FIELDS = (
    "semantic_dim",
    "semantic_residual_dim",
    "semantic_n_lsh_tables",
    "semantic_n_lsh_planes",
    "semantic_table_path",
    "semantic_vocab_ref",
    "experimental_semantic_router_vecs",
    "semantic_align_loss_coef",
)


def build_dense_ar_spec():
    return build_composed_lm_spec(
        base_model="llama",
        topology="dense",
        router_mode="none",
        use_jepa=False,
        runtime="compile",
        **LLAMA_COMMON_KW,
    )


def build_ar_jepa_spec():
    return build_composed_lm_spec(
        base_model="llama",
        topology="dense",
        router_mode="none",
        use_jepa=True,
        runtime="compile",
        **LLAMA_COMMON_KW,
    )


def build_semantic_router_spec(*, use_jepa: bool):
    return build_composed_lm_spec(
        base_model="llama",
        topology="moe",
        router_mode="semantic",
        use_jepa=use_jepa,
        runtime="compile",
        experts=NUM_VOCAB_DIMS,
        top_k=2,
        **LLAMA_COMMON_KW,
    )


class TrainGraphSpecSanitizationTest(unittest.TestCase):
    def assert_absent(self, payload: dict[str, object], *keys: str) -> None:
        for key in keys:
            self.assertNotIn(key, payload)

    def capture_saved_torch_config(
        self,
        graph,
        *,
        training_manifest: dict[str, object],
        raw_text_encoding_name: str = "gpt2",
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            weights_path = tmp_path / "weights.pt"
            graph_path = tmp_path / "graph.json"
            captured: dict[str, object] = {}

            def fake_export(_graph, path: Path) -> None:
                Path(path).write_bytes(b"pt")

            def fake_save(_graph, path: Path, include_module_state: bool = False) -> None:
                del include_module_state
                captured["torch_config"] = json.loads(json.dumps(_graph.torch_config))
                Path(path).write_text("{}", encoding="utf-8")

            with patch.object(train_jepa_semantic, "export_to_pt", side_effect=fake_export):
                with patch.object(train_jepa_semantic, "save_graph", side_effect=fake_save):
                    train_jepa_semantic.save_artifacts(
                        graph,
                        weights_path,
                        graph_path,
                        training_manifest=training_manifest,
                        raw_text_encoding_name=raw_text_encoding_name,
                    )
            return dict(captured["torch_config"])

    def test_sanitize_plain_ar_spec_prunes_objective_specific_fields(self) -> None:
        sanitized = train_jepa_semantic.sanitized_model_spec_dict(build_dense_ar_spec())
        self.assertEqual("ar", sanitized["template"]["objective"])
        self.assert_absent(sanitized, *JEPA_ONLY_FIELDS)
        self.assert_absent(sanitized, *SEMANTIC_ONLY_FIELDS)
        self.assert_absent(sanitized, "ar_loss_coef", "max_recurrence_steps", "halt_epsilon")
        self.assertIn("block_spec", sanitized)

    def test_sanitize_semantic_router_spec_keeps_semantic_fields(self) -> None:
        sanitized = train_jepa_semantic.sanitized_model_spec_dict(build_semantic_router_spec(use_jepa=False))
        self.assertEqual("semantic_router", sanitized["template"]["objective"])
        self.assertIn("ar_loss_coef", sanitized)
        self.assertIn("semantic_vocab_ref", sanitized)
        self.assertIn("semantic_align_loss_coef", sanitized)
        self.assert_absent(sanitized, *JEPA_ONLY_FIELDS)

    def test_sanitize_ar_jepa_spec_keeps_jepa_fields(self) -> None:
        sanitized = train_jepa_semantic.sanitized_model_spec_dict(build_ar_jepa_spec())
        self.assertEqual("ar_jepa", sanitized["template"]["objective"])
        self.assertIn("ar_loss_coef", sanitized)
        self.assertIn("ema_decay", sanitized)
        self.assertIn("jepa_loss_coef", sanitized)
        self.assertIn("jepa_latent_dim", sanitized)
        self.assert_absent(sanitized, *SEMANTIC_ONLY_FIELDS)

    def test_sanitize_semantic_router_jepa_spec_keeps_both_buckets(self) -> None:
        sanitized = train_jepa_semantic.sanitized_model_spec_dict(build_semantic_router_spec(use_jepa=True))
        self.assertEqual("semantic_router_jepa", sanitized["template"]["objective"])
        self.assertIn("ar_loss_coef", sanitized)
        self.assertIn("ema_decay", sanitized)
        self.assertIn("jepa_loss_coef", sanitized)
        self.assertIn("semantic_vocab_ref", sanitized)
        self.assertIn("semantic_align_loss_coef", sanitized)

    def test_sanitized_model_spec_keeps_exact_raw_text_tokenizer_name(self) -> None:
        sanitized = train_jepa_semantic.sanitized_model_spec_dict(
            build_dense_ar_spec(),
            raw_text_encoding_name="sp8192",
        )
        self.assertEqual("sp8192", sanitized["raw_text_encoding_name"])

    def test_template_spec_raw_text_encoding_helper_prefers_saved_exact_tokenizer(self) -> None:
        resolved = raw_text_encoding_name_for_template_spec(
            {
                "template": {"backbone": "llama"},
                "raw_text_encoding_name": "sp8192",
            }
        )
        self.assertEqual("sp8192", resolved)

    def test_build_graph_for_training_sanitizes_composed_cli_template_spec(self) -> None:
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

        template_spec = dict(graph.torch_config.get("template_spec", {}) or {})
        self.assertEqual("ar", template_spec["template"]["objective"])
        self.assert_absent(template_spec, *JEPA_ONLY_FIELDS)
        self.assert_absent(template_spec, *SEMANTIC_ONLY_FIELDS)
        self.assert_absent(template_spec, "ar_loss_coef", "max_recurrence_steps", "halt_epsilon")

    def test_save_artifacts_strips_semantic_metadata_from_plain_ar_artifacts(self) -> None:
        spec = build_dense_ar_spec()
        graph = build_gpt_root_graph(name="plain_ar", model_spec=spec)
        torch_config = self.capture_saved_torch_config(
            graph,
            training_manifest={"model_spec": model_spec_to_dict(spec)},
            raw_text_encoding_name="o200k_base",
        )

        template_spec = dict(torch_config.get("template_spec", {}) or {})
        artifact_metadata = dict(torch_config.get("artifact_metadata", {}) or {})
        manifest_spec = dict(torch_config.get("training_manifest", {}).get("model_spec", {}) or {})

        self.assertEqual("o200k_base", template_spec["raw_text_encoding_name"])
        self.assert_absent(template_spec, *JEPA_ONLY_FIELDS)
        self.assert_absent(template_spec, *SEMANTIC_ONLY_FIELDS)
        self.assert_absent(template_spec, "ar_loss_coef", "max_recurrence_steps", "halt_epsilon")
        self.assert_absent(artifact_metadata, "semantic_vocab_ref", "experimental_semantic_router_vecs")
        self.assertEqual("o200k_base", manifest_spec["raw_text_encoding_name"])
        self.assert_absent(manifest_spec, *JEPA_ONLY_FIELDS)
        self.assert_absent(manifest_spec, *SEMANTIC_ONLY_FIELDS)
        self.assert_absent(manifest_spec, "ar_loss_coef", "max_recurrence_steps", "halt_epsilon")

    def test_save_artifacts_keeps_semantic_metadata_for_semantic_router_artifacts(self) -> None:
        spec = build_semantic_router_spec(use_jepa=False)
        graph = build_gpt_root_graph(name="semantic_router", model_spec=spec)
        torch_config = self.capture_saved_torch_config(
            graph,
            training_manifest={"model_spec": model_spec_to_dict(spec)},
        )

        template_spec = dict(torch_config.get("template_spec", {}) or {})
        artifact_metadata = dict(torch_config.get("artifact_metadata", {}) or {})
        manifest_spec = dict(torch_config.get("training_manifest", {}).get("model_spec", {}) or {})

        self.assertIn("semantic_vocab_ref", template_spec)
        self.assertIn("semantic_align_loss_coef", template_spec)
        self.assertIn("ar_loss_coef", template_spec)
        self.assert_absent(template_spec, *JEPA_ONLY_FIELDS)
        self.assertIn("semantic_vocab_ref", artifact_metadata)
        self.assertIn("experimental_semantic_router_vecs", artifact_metadata)
        self.assertIn("semantic_vocab_ref", manifest_spec)
        self.assertIn("semantic_align_loss_coef", manifest_spec)
        self.assertIn("ar_loss_coef", manifest_spec)
        self.assert_absent(manifest_spec, *JEPA_ONLY_FIELDS)


if __name__ == "__main__":
    unittest.main()
