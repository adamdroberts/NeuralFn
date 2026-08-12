from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from neuralfn.config import SHIPPED_GPT_TEMPLATE_PRESETS
from neuralfn.native_ir import compile_native_graph_payload
from neuralfn.native_graph_train import plan_native_graph_training
from neuralfn.native_registry import (
    capability_proof_for,
    native_graph_training_adapters,
    native_trainer_specs,
)
from neuralfn.native_train import NATIVE_TRAIN_FAMILY_TARGETS
from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config


_NEGATIVE_EVIDENCE_MARKERS = (
    "blocked",
    "diagnostic",
    "missing",
    "unimplemented",
    "unproven",
    "unsupported",
)


def _normalized(value: object) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _preset_graph_payload(preset: str) -> dict[str, object]:
    spec = build_model_spec_from_config(
        {
            "preset": preset,
            "num_layers": 1,
            "model_dim": 32,
            "num_heads": 4,
            "num_kv_heads": 4,
            "multiple_of": 16,
            "vocab_size": 50_257,
        },
        preview_defaults=True,
    )
    return build_gpt_root_graph(name=preset, model_spec=spec).to_dict()


def test_shipped_text_presets_are_partitioned_by_the_native_family_registry() -> None:
    """A new shipped preset must receive an explicit native-family owner."""

    trainers = native_trainer_specs()
    trainer_targets = {spec.family: spec.native_target for spec in trainers}
    source_targets = {
        _normalized(family): target
        for family, target in NATIVE_TRAIN_FAMILY_TARGETS.items()
    }
    assert trainer_targets == source_targets

    registered_presets = [
        preset
        for trainer in trainers
        for preset in trainer.shipped_presets
    ]
    counts = Counter(registered_presets)
    assert len(set(SHIPPED_GPT_TEMPLATE_PRESETS)) == len(SHIPPED_GPT_TEMPLATE_PRESETS)
    assert set(counts) == set(SHIPPED_GPT_TEMPLATE_PRESETS)
    assert all(count == 1 for count in counts.values())


def test_nanogpt_family_registry_does_not_inherit_dense_v5_capabilities() -> None:
    trainers = {spec.family: spec for spec in native_trainer_specs()}

    nanogpt = trainers["nanogpt"]
    assert nanogpt.native_target == "nfn_gpt_native_train"
    assert nanogpt.trainer_registered is True
    assert nanogpt.architecture_persistence_proven is False
    assert nanogpt.native_forward == "diagnostic-transition-only"
    assert nanogpt.resident_inference is False
    assert "dense-native-checkpoint-and-forward-v5" not in nanogpt.evidence
    assert {
        "nanogpt-bias-free-linear-parameter-persistence-unproven",
        "nanogpt-dropout-native-architecture-forward-unproven",
        "nanogpt-bias-dropout-resident-inference-unproven",
    }.issubset(nanogpt.evidence)

    # NanoGPT's fail-closed family correction must not weaken the reviewed GPT-2
    # family that happens to share the executable target.
    gpt2 = trainers["gpt2"]
    assert gpt2.native_target == "nfn_gpt_native_train"
    assert gpt2.architecture_persistence_proven is True
    assert gpt2.native_forward == "one-shot-architecture-forward"
    assert gpt2.resident_inference is True
    assert "dense-native-checkpoint-and-forward-v5" in gpt2.evidence


@pytest.mark.parametrize(
    ("preset", "selector_evidence", "selector_gate"),
    (
        (
            "nanogpt",
            "nanogpt-eager-bias-dropout-contract-unproven",
            "nanogpt_eager_bias_dropout_graph_contract",
        ),
        (
            "nanogpt_megakernel",
            "nanogpt-megakernel-bias-dropout-contract-unproven",
            "nanogpt_megakernel_bias_dropout_graph_contract",
        ),
        (
            "nanogpt_modern",
            "nanogpt-modern-rmsnorm-rope-geglu-bias-dropout-contract-unproven",
            "nanogpt_modern_rmsnorm_rope_geglu_bias_dropout_graph_contract",
        ),
    ),
)
def test_nanogpt_plan_reports_selector_semantic_gates_without_losing_routing(
    tmp_path: Path,
    preset: str,
    selector_evidence: str,
    selector_gate: str,
) -> None:
    graph_path = tmp_path / f"{preset}.json"
    graph_path.write_text(json.dumps(_preset_graph_payload(preset)), encoding="utf-8")

    plan = plan_native_graph_training(graph_path)
    payload = plan.to_dict()
    proof = payload["compatibility_report"]["capability_proof"]

    assert plan.trainer_family == "nanogpt"
    assert plan.training_selector == preset
    assert plan.native_target == "nfn_gpt_native_train"
    assert plan.trainer_registered is True
    assert plan.compatibility_report.compatible is True
    assert plan.manifest is not None
    assert plan.manifest.capabilities["native_ir"] is True
    assert plan.manifest.capabilities["architecture_persistence"] is False
    assert plan.manifest.capabilities["native_inference"] is False
    assert plan.manifest.capabilities["resident_inference"] is False
    assert plan.architecture_persistence_proven is False
    assert plan.execution_ready is False
    assert proof["architecture_persistence_proven"] is False
    assert proof["native_forward_proven"] is False
    assert proof["resident_inference_proven"] is False
    assert "dense-native-checkpoint-and-forward-v5" not in proof["evidence"]
    assert selector_evidence in proof["evidence"]
    assert {
        "architecture_parameter_persistence_proof",
        "real_native_architecture_forward",
        "nanogpt_bias_free_linear_parameter_persistence",
        "nanogpt_dropout_native_architecture_forward",
        "nanogpt_bias_dropout_resident_inference_adapter",
        selector_gate,
    }.issubset(proof["missing_gates"])


@pytest.mark.parametrize(
    "preset",
    (
        "gpt2",
        "gpt2_megakernel",
        "gpt2_moa",
        "gpt2_qknorm",
        "gpt2_softcap",
        "gpt2_stable",
        "gpt2_zloss",
    ),
)
def test_reviewed_gpt2_profiles_retain_dense_v5_capability_proof(preset: str) -> None:
    manifest = compile_native_graph_payload(_preset_graph_payload(preset))
    proof = capability_proof_for(manifest.model, manifest.topology)

    assert proof.architecture_persistence_proven is True
    assert proof.native_forward_proven is True
    assert proof.resident_inference_proven is True
    assert proof.lossless_cache_proven is True
    assert "dense-native-checkpoint-and-forward-v5" in proof.evidence
    assert "resident-dense-in-process-abi-v1" in proof.evidence


def test_shipped_text_presets_have_explicit_native_coverage_status(
    tmp_path: Path,
) -> None:
    """Gate lowering, persistence, and resident-inference status from registries.

    This does not promote a registry declaration into runtime proof.  Each
    shipped graph is lowered and planned.  A gate may be either proved or
    explicitly blocked, but it may never be absent or silently inferred from
    family membership.
    """

    trainers = {spec.family: spec for spec in native_trainer_specs()}
    preset_owners = {
        preset: trainer
        for trainer in trainers.values()
        for preset in trainer.shipped_presets
    }
    adapters = {
        adapter.selector: adapter
        for adapter in native_graph_training_adapters()
    }

    for preset in SHIPPED_GPT_TEMPLATE_PRESETS:
        graph_path = tmp_path / f"{preset}.json"
        graph_path.write_text(
            json.dumps(_preset_graph_payload(preset)),
            encoding="utf-8",
        )

        plan = plan_native_graph_training(graph_path)
        assert plan.manifest is not None, preset
        assert plan.compatibility_report.graph_valid, (
            preset,
            plan.compatibility_report.to_dict(),
        )
        assert plan.compatibility_report.compatible, (
            preset,
            plan.compatibility_report.to_dict(),
        )

        owner = preset_owners[preset]
        assert plan.trainer_family == owner.family, preset
        assert plan.native_target == owner.native_target, preset
        assert plan.trainer_registered is True, preset

        proof = capability_proof_for(plan.manifest.model, plan.manifest.topology)
        assert proof.text_generation is True, preset
        assert proof.native_ir_lowering is True, preset
        assert proof.unsupported_modules == (), preset
        assert proof.trainer_registered is True, preset

        adapter = adapters.get(plan.training_selector)
        if plan.architecture_persistence_proven:
            assert adapter is not None, preset
            if preset == "gpt2_diff":
                # Generic classification stays blocked; only the exact
                # graph-training validator may promote this plan and issue its
                # source-bound materialized proof.
                assert adapter.architecture_persistence_proven is False
                assert proof.architecture_persistence_proven is False
                assert proof.resident_inference_proven is False
                assert "gpt2-diff-materialized-graph-training-proof-v1" in (
                    adapter.evidence
                )
            else:
                assert adapter.architecture_persistence_proven is True, preset
            assert plan.execution_ready is True, preset
        else:
            assert plan.execution_ready is False, preset
            explicit_training_blockers = [
                f"{issue.code}:{issue.message}"
                for issue in plan.training_issues
                if issue.severity == "error"
            ]
            if adapter is not None and not adapter.architecture_persistence_proven:
                explicit_training_blockers.extend(
                    evidence
                    for evidence in adapter.evidence
                    if any(marker in evidence for marker in _NEGATIVE_EVIDENCE_MARKERS)
                )
            assert explicit_training_blockers, (
                preset,
                "missing explicit training-persistence blocked reason",
                plan.to_dict(),
            )

        if proof.resident_inference_proven:
            assert proof.native_forward_proven is True, preset
            assert proof.lossless_cache_proven is True, preset
            assert proof.serving_proven is True, preset
        else:
            assert proof.missing_gates, (
                preset,
                "missing explicit resident-inference blocked reason",
                proof.to_dict(),
            )
            assert (
                "real_native_architecture_forward" in proof.missing_gates
                or "resident_inference_adapter" in proof.missing_gates
            ), (preset, proof.missing_gates)
