"""Deterministic capability registries for Native Execution IR.

This module intentionally separates three different claims:

* a compiled native trainer target is registered;
* a graph module has an explicit Native IR structural lowerer; and
* a production inference capability has passed its runtime gates.

In particular, the native-family transition sampler is diagnostic evidence, not
an architecture forward and not resident inference.  Keeping those facts in
separate frozen records prevents a trainer registry entry from silently turning
into a serving capability.

The registry is dependency-light by design.  It does not import PyTorch and it
does not probe binaries, environment variables, or the filesystem, so callers
receive the same snapshot for the same installed NeuralFn source tree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping, Sequence

_NATIVE_IR_VERSION = 1

# Keep this registry dependency-light.  In particular, importing config pulls
# in the optional semantic NumPy stack, while importing native_train makes the
# capability boundary depend on launcher implementation details.  These are
# explicit ABI registries and intentionally require a reviewed edit when the
# shipped preset/trainer catalogs change.
_NATIVE_TRAIN_FAMILY_TARGETS: dict[str, str] = {
    "gpt": "nfn_gpt_native_train",
    "gpt2": "nfn_gpt_native_train",
    "gpt3": "nfn_gpt_native_train",
    "nanogpt": "nfn_gpt_native_train",
    "gpt2-evo": "nfn_gpt2_evo_native_train",
    "llama": "nfn_llama_native_train",
    "muse-glimmer": "nfn_muse_glimmer_native_train",
    "mixllama": "nfn_mixllama_native_train",
    "jepa": "nfn_jepa_native_train",
    "semantic-dense-jepa": "nfn_semantic_dense_jepa_native_train",
    "moe-jepa-evo": "nfn_moe_jepa_evo_native_train",
    "auxfree-moe-jepa-evo": "nfn_moe_jepa_evo_native_train",
    "moe-jepa-evo-modern": "nfn_moe_jepa_evo_native_train",
    "semantic-router-moe": "nfn_semantic_router_moe_native_train",
    "deepseek-v4": "nfn_deepseek_v4_native_train",
    "jamba": "nfn_jamba_native_train",
    "seq2seq": "nfn_seq2seq_native_train",
    "diffusion": "nfn_diffusion_native_train",
    "ttt-llama": "nfn_ttt_llama_native_train",
    "hnet-lm": "nfn_hnet_lm_native_train",
    "universal-llama": "nfn_universal_llama_native_train",
}

_SHIPPED_PRESETS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "deepseek-v4": ("deepseek_v4",),
    "diffusion": ("diffusion", "diffusion_modern"),
    "gpt2": (
        "gpt2", "gpt2_diff", "gpt2_megakernel", "gpt2_moa", "gpt2_modern",
        "gpt2_qknorm", "gpt2_softcap", "gpt2_stable", "gpt2_zloss",
    ),
    "hnet-lm": ("hnet_lm", "hnet_lm_modern"),
    "jamba": ("jamba", "jamba_modern"),
    "jepa": ("dense_jepa_evo", "dense_jepa_evo_modern", "llm_jepa", "llm_jepa_modern"),
    "llama": (
        "diff_transformer", "fp8_llama", "gemma3", "kv_pca_llama",
        "kv_pca_llama_modern", "llama", "llama_fast", "llama_fast_megakernel",
        "llama_megakernel", "llama_modern", "longctx_sparse_llama",
        "modern_norms_llama", "mxfp4_llama", "qwen3_longctx", "ternary_b158",
        "ternary_b158_modern",
    ),
    "muse-glimmer": ("muse_glimmer",),
    "mixllama": (
        "deepseek_v3", "mixllama", "mixllama_fast", "mixllama_fast_megakernel",
        "moe", "moe_modern",
    ),
    "moe-jepa-evo": ("auxfree_moe_jepa_evo", "moe_jepa_evo", "moe_jepa_evo_modern"),
    "nanogpt": ("nanogpt", "nanogpt_megakernel", "nanogpt_modern"),
    "semantic-dense-jepa": (
        "dyt_geglu_semantic_dense_jepa_evo", "jepa_semantic_hybrid",
        "jepa_semantic_hybrid_megakernel", "jepa_semantic_hybrid_modern",
        "semantic_dense_jepa_evo", "semantic_dense_jepa_evo_modern",
    ),
    "semantic-router-moe": (
        "diff_semantic_moe_jepa_evo", "semantic_moe_jepa_evo",
        "semantic_moe_jepa_evo_modern", "semantic_router_moe",
        "semantic_router_moe_megakernel", "semantic_router_moe_modern",
    ),
    "seq2seq": ("seq2seq", "seq2seq_modern"),
    "ttt-llama": ("ttt_llama", "ttt_llama_modern"),
    "universal-llama": ("universal_llama", "universal_llama_modern"),
}


@dataclass(frozen=True, slots=True)
class NativeTrainerSpec:
    """One deterministic entry in the compiled native trainer registry.

    ``native_forward`` describes the strongest currently proved inference path.
    ``diagnostic-transition-only`` must never be interpreted as model inference.
    ``resident_inference`` is an independent gate and is false until an
    in-process session adapter is implemented and verified.
    """

    family: str
    native_target: str
    shipped_presets: tuple[str, ...]
    text_generation: bool
    trainer_registered: bool
    architecture_persistence_proven: bool
    native_forward: str
    resident_inference: bool
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shipped_presets"] = list(self.shipped_presets)
        payload["evidence"] = list(self.evidence)
        return payload


@dataclass(frozen=True, slots=True)
class NativeGraphTrainingAdapter:
    """One reviewed graph-file adapter for a production native trainer.

    A graph adapter is deliberately narrower than :class:`NativeTrainerSpec`.
    Trainer registration proves that an executable exists for a family; this
    record proves that a particular serialized graph profile can be validated
    ahead of time and then represented faithfully by that executable's
    graph-file/selector contract.  The current adapters validate Native IR in
    Python and launch the existing selector-driven dense trainer, so
    ``trainer_consumes_native_ir`` remains false.
    """

    selector: str
    family: str
    native_target: str
    adapter_mode: str
    trainer_consumes_native_ir: bool
    architecture_persistence_proven: bool
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = list(self.evidence)
        return payload


@dataclass(frozen=True, slots=True)
class NativeLoweringSpec:
    """An explicit structural Native IR lowerer for a builtin module type.

    Registration means that the module's type, ports, configuration, and state
    can be represented in Native IR v1.  It deliberately does not claim that a
    resident kernel exists for every model topology containing the module.
    """

    module_type: str
    opcode: str
    ir_versions: tuple[int, ...] = (_NATIVE_IR_VERSION,)
    preserves_module_config: bool = True
    preserves_module_state: bool = True
    source: str = "explicit-shipped-text-lowering-v1"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ir_versions"] = list(self.ir_versions)
        return payload


@dataclass(frozen=True, slots=True)
class NativeCapabilityProof:
    """Fail-closed proof snapshot for one classified model/topology pair."""

    model_family: str
    family_class: str
    objective: str
    text_generation: bool
    session_state_kinds: tuple[str, ...]
    module_types: tuple[str, ...]
    unsupported_modules: tuple[str, ...]
    native_ir_lowering: bool
    trainer_registered: bool
    architecture_persistence_proven: bool
    native_forward_proven: bool
    resident_inference_proven: bool
    lossless_cache_proven: bool
    turboquant_cache_proven: bool
    serving_proven: bool
    evidence: tuple[str, ...]
    missing_gates: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "session_state_kinds",
            "module_types",
            "unsupported_modules",
            "evidence",
            "missing_gates",
        ):
            payload[key] = list(payload[key])
        return payload


# This is deliberately an explicit allow-list.  The entries are the builtin
# module operations exercised by the shipped text preset set at Native IR v1's
# introduction.  Do not derive this tuple from BUILTIN_NEURONS: doing so would
# make a newly added Python/Torch module native-compatible without a lowering
# review.  A new shipped module therefore fails preflight until it is added here.
_EXPLICIT_NATIVE_MODULE_TYPES: tuple[str, ...] = (
    "absolute_position_embedding",
    "aux_loss_add",
    "auxfree_load_balancing",
    "bitlinear_ternary",
    "broadcast_chunk_routes",
    "broadcast_expert_routes",
    "byte_patch_embed",
    "byte_patch_merge",
    "causal_chunk_state",
    "dataset_source",
    "denoise_head",
    "differential_attention",
    "dpo_dataset_source",
    "dpo_pairwise_loss",
    "dropout",
    "dyt",
    "expert_combine",
    "expert_dispatch",
    "fp8_linear",
    "fused_causal_attention",
    "geglu",
    "gelu",
    "jepa_mask",
    "jepa_predictor",
    "jepa_projector",
    "kv_pca_decode",
    "kv_pca_encode",
    "latent_mse_loss",
    "latent_pool",
    "layer_norm",
    "linear",
    "lora_linear",
    "nf4_linear",
    "lm_head",
    "load_balance_loss",
    "logit_softcap",
    "loss_scale",
    "tensor_scale",
    "mamba",
    "manifold_hyper_connection",
    "mask_scheduler",
    "masked_reward_head",
    "masked_ppo_clipped_loss",
    "masked_token_cross_entropy",
    "merge_heads",
    "multiply",
    "pair_batch_concat",
    "pair_batch_split",
    "policy_logits_value",
    "preference_bce_loss",
    "ppo_rollout_source",
    "multi_latent_attention",
    "mx_linear",
    "native_sparse_attention",
    "qk_norm",
    "random_timesteps",
    "reference_forward",
    "repeat_kv",
    "reshape_heads",
    "residual_add",
    "rms_norm",
    "rotary_embedding",
    "route_balance_loss",
    "route_distillation_loss",
    "route_selection_loss",
    "routed_attention_experts",
    "router_logits",
    "scaled_dot_product_attention",
    "semantic_alignment_loss",
    "semantic_chunk_hasher",
    "semantic_chunk_projector",
    "semantic_data_source",
    "semantic_hash_router",
    "semantic_hasher",
    "semantic_moe_jepa_evo_router",
    "semantic_projector",
    "sequence_logp",
    "sigmoid",
    "silu",
    "sliding_window_attention",
    "sft_dataset_source",
    "swiglu",
    "tied_lm_head",
    "token_cross_entropy",
    "token_embedding",
    "token_logp_entropy",
    "topk_route",
    "ttt_linear",
    "universal_transformer",
)


_DENSE_FORWARD_FAMILIES = frozenset({"gpt", "gpt2", "gpt3", "gpt2-evo"})

# NanoGPT currently routes to the shared dense executable, but that executable's
# persisted parameter layout and forward are the biased, dropout-free GPT-2
# contract.  The shipped NanoGPT graphs instead author bias-free linear layers
# and dropout, so family routing must not inherit the dense-v5 capability proof.
# Keep the gaps explicit in both the family registry and selector-level proofs.
_NANOGPT_UNPROVEN_EVIDENCE: tuple[str, ...] = (
    "nanogpt-bias-free-linear-parameter-persistence-unproven",
    "nanogpt-dropout-native-architecture-forward-unproven",
    "nanogpt-bias-dropout-resident-inference-unproven",
)
_NANOGPT_MISSING_GATES: tuple[str, ...] = (
    "nanogpt_bias_free_linear_parameter_persistence",
    "nanogpt_dropout_native_architecture_forward",
    "nanogpt_bias_dropout_resident_inference_adapter",
)
_NANOGPT_SELECTOR_GAPS: dict[str, tuple[str, str]] = {
    "nanogpt": (
        "nanogpt-eager-bias-dropout-contract-unproven",
        "nanogpt_eager_bias_dropout_graph_contract",
    ),
    "nanogpt_megakernel": (
        "nanogpt-megakernel-bias-dropout-contract-unproven",
        "nanogpt_megakernel_bias_dropout_graph_contract",
    ),
    "nanogpt_modern": (
        "nanogpt-modern-rmsnorm-rope-geglu-bias-dropout-contract-unproven",
        "nanogpt_modern_rmsnorm_rope_geglu_bias_dropout_graph_contract",
    ),
}

# These are the reviewed dense graph-file selectors whose authored topology
# can be validated explicitly.  Membership alone does not grant execution:
# selector-specific state/forward gaps remain fail-closed below.  The compiled
# catalog also accepts modern and NanoGPT selector labels, but its training loop
# still uses GPT-2 LayerNorm/GELU/absolute-position blocks with learned linear
# biases and no dropout.  Those labels therefore must not be promoted to
# graph-authored adapters.  Canonical LLaMA is added separately below because
# its exact topology gate must not promote unrelated LLaMA-family presets.
# ``llama_fast`` is the one reviewed alias: it has the same active graph and
# semantics as canonical LLaMA, with the compile runtime profile recorded
# separately from the eager profile.  The standard-MoE selectors are registered
# separately as well: they share one exact RMSNorm/RoPE/GQA/softmax-top-k graph
# contract and must not promote modern, aux-free, fused, semantic, DeepSeek, or
# JEPA neighbors.
_REVIEWED_GRAPH_TRAINING_SELECTORS: tuple[str, ...] = (
    "gpt2",
    "gpt2_diff",
    "gpt2_megakernel",
    "gpt2_moa",
    "gpt2_qknorm",
    "gpt2_softcap",
    "gpt2_stable",
    "gpt2_zloss",
)

# These selectors remain structurally lowerable and retain an explicit graph
# adapter so preflight can validate their exact authored topology.  They must
# not inherit the dense-family persistence/forward claims merely from
# classification: the graph-training planner may promote an exact validated,
# source-bound profile after issuing its materialized handoff proof, while the
# generic Native IR capability and resident-inference gates stay fail-closed.
_UNPROVEN_GRAPH_TRAINING_SELECTORS: frozenset[str] = frozenset({"gpt2_diff"})
_TEXT_OBJECTIVES = frozenset(
    {
        "ar",
        "diffusion",
        "jepa",
        "ar_jepa",
        "jepa_semantic",
        "semantic_router",
        "semantic_router_jepa",
        "semantic_dense_jepa_evo",
        "semantic_moe_jepa_evo",
        "seq2seq",
        "sft",
        "dpo",
        "ppo",
    }
)


def _normalize_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _preset_family(preset: Any) -> str:
    normalized = _normalize_name(preset)
    if not normalized:
        return ""
    if normalized == "nano-gpt" or normalized.startswith("nanogpt-") or normalized == "nanogpt":
        return "nanogpt"
    if normalized == "gpt2" or normalized.startswith("gpt2-"):
        return "gpt2"
    if normalized in {"gpt", "gpt3"}:
        return normalized
    for family, presets in _SHIPPED_PRESETS_BY_FAMILY.items():
        if any(_normalize_name(candidate) == normalized for candidate in presets):
            return family
    return ""


def _presets_by_family() -> dict[str, tuple[str, ...]]:
    return {
        family: tuple(sorted(presets))
        for family, presets in sorted(_SHIPPED_PRESETS_BY_FAMILY.items())
    }


def native_trainer_specs() -> tuple[NativeTrainerSpec, ...]:
    """Return the canonical trainer snapshot in stable family-name order."""

    presets_by_family = _presets_by_family()
    specs: list[NativeTrainerSpec] = []
    for family, native_target in sorted(_NATIVE_TRAIN_FAMILY_TARGETS.items()):
        normalized_family = _normalize_name(family)
        dense_forward = normalized_family in _DENSE_FORWARD_FAMILIES
        if dense_forward:
            capability_evidence = (
                "dense-native-checkpoint-and-forward-v5",
                "resident-dense-in-process-abi-v1",
                "resident-dense-lossless-kv-cache-v1",
            )
        elif normalized_family == "muse-glimmer":
            capability_evidence = (
                "muse-glimmer-exact-native-training-abi-v1",
                "muse-glimmer-bf16-parameter-checkpoint-resume-v1",
                "muse-glimmer-frozen-base-native-lora-v1",
                "muse-glimmer-frozen-nf4-base-native-qlora-v1",
                "resident-muse-glimmer-cpu-cuda-abi-v1",
                "resident-muse-glimmer-hybrid-lossless-kv-cache-v1",
            )
        elif normalized_family == "nanogpt":
            capability_evidence = _NANOGPT_UNPROVEN_EVIDENCE
        else:
            capability_evidence = ("native-family-transition-sampler-is-diagnostic",)
        specs.append(
            NativeTrainerSpec(
                family=normalized_family,
                native_target=str(native_target),
                shipped_presets=presets_by_family.get(normalized_family, ()),
                text_generation=True,
                trainer_registered=True,
                architecture_persistence_proven=(
                    dense_forward or normalized_family == "muse-glimmer"
                ),
                native_forward=(
                    "one-shot-architecture-forward"
                    if dense_forward or normalized_family == "muse-glimmer"
                    else "diagnostic-transition-only"
                ),
                resident_inference=(
                    dense_forward or normalized_family == "muse-glimmer"
                ),
                evidence=(
                    "native_train.NATIVE_TRAIN_FAMILY_TARGETS",
                    *capability_evidence,
                ),
            )
        )
    return tuple(specs)


def native_graph_training_adapters() -> tuple[NativeGraphTrainingAdapter, ...]:
    """Return reviewed production graph-file adapters in selector order."""

    dense_adapters = tuple(
        NativeGraphTrainingAdapter(
            selector=selector,
            family="gpt2",
            native_target=_NATIVE_TRAIN_FAMILY_TARGETS["gpt2"],
            adapter_mode="validated-dense-graph-file-v1",
            trainer_consumes_native_ir=False,
            architecture_persistence_proven=(
                selector not in _UNPROVEN_GRAPH_TRAINING_SELECTORS
            ),
            evidence=(
                "native-ir-safe-payload-compiler-v1",
                "dense-gpt2-graph-shape-contract-v1",
                *(
                    (
                        "gpt2-diff-low-level-learned-lambda-training-bundle-v2-proven",
                        "gpt2-diff-materialized-graph-training-proof-v1",
                        "gpt2-diff-native-ir-migration-resident-bundle-consumer-unimplemented",
                        "gpt2-diff-resident-differential-forward-unimplemented",
                    )
                    if selector == "gpt2_diff"
                    else ("dense-native-checkpoint-and-forward-v5",)
                ),
            ),
        )
        for selector in _REVIEWED_GRAPH_TRAINING_SELECTORS
    )
    llama_adapters = tuple(
        NativeGraphTrainingAdapter(
            selector=selector,
            family="llama",
            native_target=_NATIVE_TRAIN_FAMILY_TARGETS["llama"],
            adapter_mode="validated-canonical-llama-graph-file-v1",
            trainer_consumes_native_ir=False,
            architecture_persistence_proven=True,
            evidence=(
                "native-ir-safe-payload-compiler-v1",
                "canonical-llama-active-topology-contract-v1",
                "canonical-llama-graph-resident-logits-loss-parity-v1",
                "native-family-llama-float32-checkpoint-v1",
                *(
                    ("llama-fast-compile-runtime-alias-v1",)
                    if selector == "llama_fast"
                    else ()
                ),
            ),
        )
        for selector in ("llama", "llama_fast")
    )
    standard_moe_adapters = tuple(
        NativeGraphTrainingAdapter(
            selector=selector,
            family="mixllama",
            native_target=_NATIVE_TRAIN_FAMILY_TARGETS["mixllama"],
            adapter_mode="validated-standard-moe-graph-file-v1",
            trainer_consumes_native_ir=False,
            architecture_persistence_proven=True,
            evidence=(
                "native-ir-safe-payload-compiler-v1",
                "canonical-standard-moe-active-topology-contract-v1",
                "standard-moe-floating-expert-width-v1",
                "standard-moe-exact-router-aux-loss-gradient-v1",
                "native-family-standard-moe-float32-checkpoint-v1",
                "resident-standard-moe-cpu-reference-abi-v1",
                *(
                    ("mixllama-fast-compile-runtime-alias-v1",)
                    if selector == "mixllama_fast"
                    else ()
                ),
            ),
        )
        for selector in ("moe", "mixllama", "mixllama_fast")
    )
    glimmer_adapter = NativeGraphTrainingAdapter(
        selector="muse_glimmer",
        family="muse-glimmer",
        native_target=_NATIVE_TRAIN_FAMILY_TARGETS["muse-glimmer"],
        adapter_mode="validated-muse-glimmer-graph-file-v1",
        trainer_consumes_native_ir=False,
        architecture_persistence_proven=True,
        evidence=(
            "native-ir-safe-payload-compiler-v1",
            "muse-glimmer-exact-active-topology-contract-v1",
            "muse-glimmer-exact-native-training-abi-v1",
            "muse-glimmer-bf16-parameter-checkpoint-resume-v1",
            "muse-glimmer-frozen-base-native-lora-v1",
            "muse-glimmer-frozen-nf4-base-native-qlora-v1",
            "muse-glimmer-native-dpo-v1",
            "muse-glimmer-native-reward-model-v1",
            "muse-glimmer-native-online-ppo-v1",
        ),
    )
    return (*dense_adapters, *llama_adapters, *standard_moe_adapters, glimmer_adapter)


def native_lowering_specs() -> tuple[NativeLoweringSpec, ...]:
    """Return explicit Native IR module lowerers in stable operation order."""

    return tuple(
        NativeLoweringSpec(module_type=module_type, opcode=module_type)
        for module_type in _EXPLICIT_NATIVE_MODULE_TYPES
    )


def registered_native_module_types() -> tuple[str, ...]:
    """Return the deterministic, fail-closed module-type allow-list."""

    return tuple(spec.module_type for spec in native_lowering_specs())


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _template_parts(model: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    spec = _mapping(model.get("template_spec"))
    template = _mapping(spec.get("template"))
    block = _mapping(spec.get("block_spec"))
    return spec, template, block


def _resident_muse_glimmer_v1_graph_compatible(
    model_mapping: Mapping[str, Any],
    topology_mapping: Mapping[str, Any],
    classification: Mapping[str, Any],
    *,
    allow_training_objectives: bool = False,
) -> bool:
    """Require the immutable production Muse Glimmer decoder contract.

    The compiled trainer consumes the authenticated BF16 tensor table rather
    than arbitrary Native IR.  This gate therefore admits only the exact
    shipped 52-layer graph; preview/tiny graphs remain test oracles and cannot
    inherit production capability.
    """

    del topology_mapping  # module lowering is checked independently
    if classification.get("model_family") != "muse-glimmer":
        return False
    model = _mapping(model_mapping)
    spec, template, block = _template_parts(model)
    exact_spec = {
        "model_dim": 6656,
        "num_layers": 52,
        "vocab_size": 202048,
        "tie_embeddings": False,
        "max_position_embeddings": 131072,
        "output_multiplier": 0.19611613513818404,
        "logit_softcap": 20.0,
    }
    if any(spec.get(key) != value for key, value in exact_spec.items()):
        return False
    objective = _normalize_name(template.get("objective"))
    if objective not in (
        {"ar", "sft", "dpo", "reward-model", "ppo"}
        if allow_training_objectives
        else {"ar"}
    ):
        return False
    exact_template = {
        "backbone": "muse_glimmer",
        "runtime": "eager",
        "sparsity": "dense",
    }
    if any(_normalize_name(template.get(key)) != _normalize_name(value)
           for key, value in exact_template.items()):
        return False
    exact_block = {
        "family": "muse_glimmer",
        "norm_type": "rmsnorm",
        "mlp_type": "swiglu",
        "num_heads": 32,
        "num_kv_heads": 2,
        "head_dim": 128,
        "attention_inner_dim": 4096,
        "intermediate_size": 19968,
        "is_causal": True,
        "linear_bias": False,
        "use_qk_norm": True,
        "qk_norm_kind": "weightless_rms",
        "qk_norm_eps": 1.0e-5,
        "q_scale_factor": 3.87,
        "attention_gate": "sigmoid",
        "attention_gate_dim": 4096,
        "norm_layout": "sandwich",
        "centered_rms_norm": True,
        "norm_eps": 1.0e-5,
        "post_norm_eps": 1.0e-8,
        "embedding_norm_kind": "weightless_rms",
        "embedding_norm_eps": 1.0e-5,
    }
    if any(block.get(key) != value for key, value in exact_block.items()):
        return False
    pattern = block.get("layer_attention_pattern")
    if not isinstance(pattern, (tuple, list)) or len(pattern) != 4:
        return False
    expected = (
        ("local", 2048, "rope", 500000.0),
        ("local", 2048, "rope", 500000.0),
        ("local", 2048, "rope", 500000.0),
        ("full", None, "none", 500000.0),
    )
    for item, wanted in zip(pattern, expected, strict=True):
        entry = _mapping(item)
        actual = (
            _normalize_name(entry.get("kind")),
            entry.get("window_size"),
            _normalize_name(entry.get("pos_encoding")),
            entry.get("rope_theta"),
        )
        if actual != wanted:
            return False
    if objective == "ar":
        return spec.get("finetune") in (None, {})
    finetune = _mapping(spec.get("finetune"))
    adapter_type = _normalize_name(block.get("adapter_type"))
    if not (
        _normalize_name(finetune.get("objective")) == objective
        and _normalize_name(template.get("adapter")) == "none"
        and adapter_type in {"none", "lora", "qlora"}
        and all(
            re.fullmatch(r"[0-9a-f]{64}", str(finetune.get(key) or "")) is not None
            for key in ("tokenizer_sha256", "chat_template_sha256")
        )
    ):
        return False
    base_weight_precision = _normalize_name(
        finetune.get("base_weight_precision") or "bf16"
    )
    if base_weight_precision not in {
        "bf16",
        "k-quant-17gb",
        "k-quant-dynamic",
    }:
        return False
    if base_weight_precision != "bf16":
        canonical_digests = {
            "k-quant-17gb": "4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e",
            "k-quant-dynamic": "ac7023d6a4c704eb9af54ab53e476a66b7f5b6c0ef2fc4a8dde5253c291a6c38",
        }
        if (
            adapter_type != "lora"
            or objective not in {"sft", "dpo"}
            or not str(finetune.get("base_checkpoint") or "")
            or str(finetune.get("base_checkpoint_sha256") or "")
            != canonical_digests[base_weight_precision]
            or finetune.get("adapter_only_save") is not True
        ):
            return False
        if objective in {"dpo", "ppo"} and (
            str(finetune.get("ref_checkpoint") or "")
            != str(finetune.get("base_checkpoint") or "")
            or str(finetune.get("ref_checkpoint_sha256") or "")
            != str(finetune.get("base_checkpoint_sha256") or "")
        ):
            return False
    if objective == "dpo":
        try:
            beta = float(finetune.get("beta"))
            smoothing = float(finetune.get("dpo_label_smoothing"))
        except (TypeError, ValueError):
            return False
        if (
            not str(finetune.get("ref_checkpoint") or "")
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(finetune.get("ref_checkpoint_sha256") or ""),
            )
            is None
            or not math.isfinite(beta)
            or beta <= 0.0
            or not math.isfinite(smoothing)
            or not 0.0 <= smoothing < 0.5
            or _normalize_name(finetune.get("dpo_loss_type"))
            not in {"sigmoid", "hinge", "ipo"}
        ):
            return False
    elif objective == "reward-model" and adapter_type != "none":
        return False
    elif objective == "ppo":
        try:
            kl_coef = float(finetune.get("kl_coef"))
            clip = float(finetune.get("ppo_clip"))
            value_coefficient = float(finetune.get("ppo_vf_coef"))
            entropy_coefficient = float(finetune.get("ppo_ent_coef"))
            rollout_length = int(finetune.get("rollout_length"))
            epochs = int(finetune.get("ppo_epochs_per_rollout"))
            minibatch = int(finetune.get("ppo_minibatch_size"))
            gamma = float(finetune.get("gae_gamma"))
            gae_lambda = float(finetune.get("gae_lambda"))
        except (TypeError, ValueError):
            return False
        if (
            not str(finetune.get("ref_checkpoint") or "")
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(finetune.get("ref_checkpoint_sha256") or ""),
            )
            is None
            or not str(finetune.get("reward_checkpoint") or "")
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(finetune.get("reward_checkpoint_sha256") or ""),
            )
            is None
            or not math.isfinite(kl_coef)
            or kl_coef < 0.0
            or not math.isfinite(clip)
            or not 0.0 < clip < 1.0
            or not math.isfinite(value_coefficient)
            or value_coefficient < 0.0
            or not math.isfinite(entropy_coefficient)
            or entropy_coefficient < 0.0
            or rollout_length <= 0
            or epochs <= 0
            or minibatch <= 0
            or not math.isfinite(gamma)
            or not 0.0 <= gamma <= 1.0
            or not math.isfinite(gae_lambda)
            or not 0.0 <= gae_lambda <= 1.0
        ):
            return False
    if adapter_type == "none":
        return True
    allowed_targets = {
        "q_proj", "k_proj", "v_proj", "o_proj", "attn_gate_proj",
        "gate_proj", "up_proj", "down_proj",
    }
    raw_targets = block.get("lora_targets")
    targets = tuple(str(value) for value in raw_targets) if isinstance(raw_targets, (list, tuple)) else ()
    try:
        rank = int(block.get("lora_rank"))
        alpha = float(block.get("lora_alpha"))
        dropout = float(block.get("lora_dropout"))
    except (TypeError, ValueError):
        return False
    return (
        bool(targets)
        and len(targets) == len(set(targets))
        and set(targets) <= allowed_targets
        and rank > 0
        and alpha > 0.0
        and 0.0 <= dropout < 1.0
        and block.get("lora_bias") is False
        and finetune.get("adapter_only_save") is True
        and (
            adapter_type != "qlora"
            or (
                block.get("qlora_group_size") == 64
                and _normalize_name(block.get("qlora_compute_dtype"))
                in {"bf16", "bfloat16"}
            )
        )
    )


def classify_native_graph_training_selector(
    model_mapping: Mapping[str, Any],
    topology_mapping: Mapping[str, Any],
) -> str:
    """Infer a reviewed production selector from immutable IR metadata.

    This classifier does not itself grant execution permission.  Callers must
    still require an entry from :func:`native_graph_training_adapters` and
    validate the selector's exact topology/configuration contract.  Returning
    unsupported selectors such as ``gpt2_modern`` or ``nanogpt`` is useful for
    precise fail-closed diagnostics.  LLaMA selectors are returned only for
    the exact canonical RMSNorm/RoPE/GQA/dense/SwiGLU topology: ``llama`` for
    the eager runtime profile and ``llama_fast`` for the compile runtime
    profile.  The exact standard-MoE graph similarly maps eager ``mixllama``
    (and the historical ``moe`` alias when the serialized graph retains that
    name) plus compile ``mixllama_fast`` to the shared native MixLLaMA ABI.
    Related presets remain unclassified and therefore cannot inherit either
    adapter.
    """

    model = _mapping(model_mapping)
    topology = _mapping(topology_mapping)
    spec, template, block = _template_parts(model)
    classification = classify_native_model(model, topology)
    family = str(classification["model_family"])
    if family == "muse-glimmer":
        return (
            "muse_glimmer"
            if _resident_muse_glimmer_v1_graph_compatible(
                model, topology, classification, allow_training_objectives=True
            )
            else ""
        )
    if family == "llama":
        if not _resident_llama_v1_graph_compatible(model, topology, classification):
            return ""
        runtime = _normalize_name(template.get("runtime"))
        return {"eager": "llama", "compile": "llama_fast"}.get(runtime, "")
    if family == "mixllama":
        if not _resident_standard_moe_v1_graph_compatible(
            model, topology, classification
        ):
            return ""
        runtime = _normalize_name(template.get("runtime"))
        if runtime == "compile":
            return "mixllama_fast"
        if runtime != "eager":
            return ""
        # ``moe`` and ``mixllama`` serialize the same template specification.
        # Preserve the old alias when it remains visible in the graph name;
        # arbitrary names normalize to the canonical MixLLaMA selector.
        visible_name = _normalize_name(model.get("name"))
        return "moe" if visible_name == "moe" or visible_name.startswith("moe-") else "mixllama"
    if family not in {"gpt2", "nanogpt"}:
        return ""

    runtime = _normalize_name(template.get("runtime") or "eager")
    norm_type = _normalize_name(block.get("norm_type") or "")
    mlp_type = _normalize_name(block.get("mlp_type") or "")
    pos_encoding = _normalize_name(block.get("pos_encoding") or "")
    modern = (
        norm_type == "rmsnorm"
        or mlp_type == "geglu"
        or pos_encoding in {"rope", "rotary", "rotary-embedding"}
    )
    if modern:
        return f"{family}_modern"
    if runtime == "megakernel":
        return f"{family}_megakernel"
    if family == "nanogpt":
        return "nanogpt"

    activation_mode = _normalize_name(block.get("activation_mode") or "single")
    attention_variant = _normalize_name(block.get("attention_variant") or "dense")
    use_qk_norm = bool(block.get("use_qk_norm", False))
    try:
        z_loss_coef = float(spec.get("z_loss_coef") or 0.0)
    except (TypeError, ValueError):
        z_loss_coef = 0.0
    try:
        logit_softcap = float(spec.get("logit_softcap") or 0.0)
    except (TypeError, ValueError):
        logit_softcap = 0.0

    if activation_mode == "moa":
        return "gpt2_moa"
    if attention_variant == "differential":
        return "gpt2_diff"
    if use_qk_norm and z_loss_coef > 0.0:
        return "gpt2_stable"
    if use_qk_norm:
        return "gpt2_qknorm"
    if logit_softcap > 0.0:
        return "gpt2_softcap"
    if z_loss_coef > 0.0:
        return "gpt2_zloss"
    return "gpt2"


def _topology_module_types(
    topology: Mapping[str, Any],
    *,
    include_variants: bool = True,
) -> tuple[str, ...]:
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            kind = _normalize_name(value.get("kind"))
            module_type = value.get("module_type")
            if kind == "module":
                operation = value.get("operation") or module_type
                if operation:
                    found.add(str(operation).strip())
            elif module_type:
                # Raw graph-like mappings may carry module_type without a
                # lowered ``kind`` field.  The explicit registry still gates it.
                found.add(str(module_type).strip())
            for key, child in value.items():
                if not include_variants and str(key) in {"variant_library", "variant_graphs"}:
                    continue
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                visit(child)

    visit(topology)
    return tuple(sorted(item for item in found if item))


def _active_topology_module_types(topology: Mapping[str, Any]) -> tuple[str, ...]:
    """Return entry-graph operations without inactive variant-library bodies."""

    graphs = topology.get("graphs")
    if isinstance(graphs, Sequence) and not isinstance(graphs, (str, bytes, bytearray)):
        path_annotated = [
            graph
            for graph in graphs
            if isinstance(graph, Mapping) and str(graph.get("path") or "").strip()
        ]
        if path_annotated:
            active = [
                graph
                for graph in path_annotated
                if str(graph.get("path")) == "root"
                or str(graph.get("path")).startswith("root/")
            ]
            return _topology_module_types({"graphs": active}, include_variants=False)
    return _topology_module_types(topology, include_variants=False)


def _active_topology_graphs(
    topology: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...] | None:
    """Return structurally valid entry-graph bodies, excluding variant copies."""

    graphs = topology.get("graphs")
    if not isinstance(graphs, Sequence) or isinstance(graphs, (str, bytes, bytearray)):
        return None
    active: list[Mapping[str, Any]] = []
    for graph in graphs:
        if not isinstance(graph, Mapping):
            return None
        path = graph.get("path")
        if not isinstance(path, str) or not path:
            return None
        if path == "root" or path.startswith("root/"):
            nodes = graph.get("nodes")
            edges = graph.get("edges")
            if (
                not isinstance(nodes, Sequence)
                or isinstance(nodes, (str, bytes, bytearray))
                or not isinstance(edges, Sequence)
                or isinstance(edges, (str, bytes, bytearray))
            ):
                return None
            active.append(graph)
    return tuple(active)


def _operation_nodes(
    graph: Mapping[str, Any],
    operation: str,
) -> tuple[Mapping[str, Any], ...] | None:
    nodes = graph.get("nodes")
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes, bytearray)):
        return None
    result: list[Mapping[str, Any]] = []
    for node in nodes:
        if not isinstance(node, Mapping):
            return None
        path = node.get("path")
        if not isinstance(path, str) or not path:
            return None
        if str(node.get("operation") or node.get("module_type") or "") == operation:
            result.append(node)
    return tuple(result)


def _edge_signatures(
    graph: Mapping[str, Any],
) -> tuple[tuple[str, int, str, int], ...] | None:
    edges = graph.get("edges")
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes, bytearray)):
        return None
    result: list[tuple[str, int, str, int]] = []
    for edge in edges:
        if not isinstance(edge, Mapping):
            return None
        src_node = edge.get("src_node")
        dst_node = edge.get("dst_node")
        src_port = edge.get("src_port")
        dst_port = edge.get("dst_port")
        if (
            not isinstance(src_node, str)
            or not src_node
            or not isinstance(dst_node, str)
            or not dst_node
            or isinstance(src_port, bool)
            or not isinstance(src_port, int)
            or src_port < 0
            or isinstance(dst_port, bool)
            or not isinstance(dst_port, int)
            or dst_port < 0
        ):
            return None
        result.append((src_node, src_port, dst_node, dst_port))
    return tuple(result)


def _unique_operation_path(
    graph: Mapping[str, Any],
    operation: str,
) -> str | None:
    nodes = _operation_nodes(graph, operation)
    if nodes is None or len(nodes) != 1:
        return None
    path = nodes[0].get("path")
    return path if isinstance(path, str) and path else None


def _instance_operation_path(
    graph: Mapping[str, Any],
    *,
    instance_id: str,
    operation: str,
) -> str | None:
    nodes = graph.get("nodes")
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes, bytearray)):
        return None
    matches = [
        node
        for node in nodes
        if isinstance(node, Mapping)
        and node.get("instance_id") == instance_id
        and node.get("operation") == operation
    ]
    if len(matches) != 1:
        return None
    path = matches[0].get("path")
    return path if isinstance(path, str) and path else None


def _has_exact_transform_chain(
    graph: Mapping[str, Any],
    required_edges: Sequence[tuple[str, int, str, int]],
) -> bool:
    """Require each transform edge and forbid competing port connections."""

    edges = _edge_signatures(graph)
    if edges is None:
        return False
    for signature in required_edges:
        if edges.count(signature) != 1:
            return False
        src_node, src_port, dst_node, dst_port = signature
        if sum(edge[:2] == (src_node, src_port) for edge in edges) != 1:
            return False
        if sum(edge[2:] == (dst_node, dst_port) for edge in edges) != 1:
            return False
    return True


def _exact_instance_nodes(
    graph: Mapping[str, Any],
    expected_operations: Mapping[str, str],
) -> dict[str, Mapping[str, Any]] | None:
    """Return nodes when a graph has exactly the reviewed instance/operation set."""

    nodes = graph.get("nodes")
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes, bytearray)):
        return None
    by_instance: dict[str, Mapping[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, Mapping):
            return None
        instance_id = node.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id or instance_id in by_instance:
            return None
        by_instance[instance_id] = node
    if set(by_instance) != set(expected_operations):
        return None
    if any(
        str(by_instance[instance_id].get("operation") or "") != operation
        for instance_id, operation in expected_operations.items()
    ):
        return None
    return by_instance


def _has_exact_instance_edges(
    graph: Mapping[str, Any],
    expected_edges: Sequence[tuple[str, int, str, int]],
) -> bool:
    """Require the complete reviewed edge set, expressed with instance ids."""

    graph_path = graph.get("path")
    if not isinstance(graph_path, str) or not graph_path:
        return False
    prefix = f"{graph_path}/nodes/"
    expected = tuple(
        sorted(
            (f"{prefix}{src}", src_port, f"{prefix}{dst}", dst_port)
            for src, src_port, dst, dst_port in expected_edges
        )
    )
    actual = _edge_signatures(graph)
    return actual is not None and tuple(sorted(actual)) == expected


def _node_config_matches(
    node: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    """Match explicit contract fields without treating bools as numeric values."""

    config = _mapping(node.get("module_config"))
    for field, required in expected.items():
        if field not in config:
            return False
        actual = config[field]
        if isinstance(required, bool):
            if actual is not required:
                return False
        elif isinstance(required, (int, float)) and not isinstance(required, bool):
            if isinstance(actual, bool) or actual != required:
                return False
        elif actual != required:
            return False
    return True


def _dense_parameter_free_topology_compatible(
    topology: Mapping[str, Any],
    *,
    num_layers: int,
    use_qk_norm: bool,
    logit_softcap: float,
) -> bool:
    """Prove the exact active dataflow for resident-only dense transforms."""

    active = _active_topology_graphs(topology)
    if active is None:
        return False
    graphs_by_path: dict[str, list[Mapping[str, Any]]] = {}
    qk_nodes: list[Mapping[str, Any]] = []
    softcap_nodes: list[Mapping[str, Any]] = []
    for graph in active:
        path = str(graph["path"])
        graphs_by_path.setdefault(path, []).append(graph)
        graph_qk_nodes = _operation_nodes(graph, "qk_norm")
        graph_softcap_nodes = _operation_nodes(graph, "logit_softcap")
        if graph_qk_nodes is None or graph_softcap_nodes is None:
            return False
        qk_nodes.extend(graph_qk_nodes)
        softcap_nodes.extend(graph_softcap_nodes)

    if len(qk_nodes) != (num_layers if use_qk_norm else 0):
        return False
    for node in qk_nodes:
        config = _mapping(node.get("module_config"))
        epsilon = config.get("eps")
        if isinstance(epsilon, bool) or epsilon != 1.0e-6:
            return False

    if use_qk_norm:
        for layer in range(num_layers):
            path = (
                "root/nodes/model/subgraph/nodes/"
                f"block_{layer}/subgraph/nodes/attention/subgraph"
            )
            matching = graphs_by_path.get(path, [])
            if len(matching) != 1:
                return False
            graph = matching[0]
            # There are three reshape nodes, so resolve Q/K by their canonical
            # instance identities while keeping the edge proof port-based.
            q_heads = _instance_operation_path(
                graph, instance_id="q_heads", operation="reshape_heads"
            )
            k_heads = _instance_operation_path(
                graph, instance_id="k_heads", operation="reshape_heads"
            )
            qk_norm = _unique_operation_path(graph, "qk_norm")
            sdpa = _unique_operation_path(graph, "scaled_dot_product_attention")
            if not all(isinstance(value, str) and value for value in (q_heads, k_heads, qk_norm, sdpa)):
                return False
            if not _has_exact_transform_chain(
                graph,
                (
                    (q_heads, 0, qk_norm, 0),
                    (k_heads, 0, qk_norm, 1),
                    (qk_norm, 0, sdpa, 0),
                    (qk_norm, 1, sdpa, 1),
                ),
            ):
                return False

    if len(softcap_nodes) != (1 if logit_softcap > 0.0 else 0):
        return False
    for node in softcap_nodes:
        config = _mapping(node.get("module_config"))
        softcap = config.get("softcap")
        if isinstance(softcap, bool) or softcap != logit_softcap:
            return False

    if logit_softcap > 0.0:
        matching = graphs_by_path.get("root/nodes/model/subgraph", [])
        if len(matching) != 1:
            return False
        graph = matching[0]
        tied_head = _unique_operation_path(graph, "tied_lm_head")
        softcap = _unique_operation_path(graph, "logit_softcap")
        consumer = _unique_operation_path(graph, "token_cross_entropy")
        if not all(isinstance(value, str) and value for value in (tied_head, softcap, consumer)):
            return False
        if not _has_exact_transform_chain(
            graph,
            (
                (tied_head, 0, softcap, 0),
                (softcap, 0, consumer, 0),
            ),
        ):
            return False
    return True


def _trainer_family(
    model: Mapping[str, Any],
    *,
    objective: str,
    backbone: str,
    module_types: tuple[str, ...],
) -> str:
    explicit = _normalize_name(model.get("trainer_family"))
    if explicit in _NATIVE_TRAIN_FAMILY_TARGETS:
        return explicit

    preset = model.get("preset") or model.get("template_name")
    preset_family = _preset_family(preset)
    if preset_family:
        return preset_family

    operations = frozenset(module_types)
    if objective == "diffusion" or {"mask_scheduler", "denoise_head"} <= operations:
        return "diffusion"
    if objective == "seq2seq":
        return "seq2seq"
    if backbone == "jamba" or (backbone == "unknown" and "mamba" in operations):
        return "jamba"
    if backbone == "ttt" or (backbone == "unknown" and "ttt_linear" in operations):
        return "ttt-llama"
    if backbone == "universal" or (backbone == "unknown" and "universal_transformer" in operations):
        return "universal-llama"
    if backbone == "hnet" or (
        backbone == "unknown" and {"byte_patch_embed", "byte_patch_merge"} & operations
    ):
        return "hnet-lm"
    if {"native_sparse_attention", "manifold_hyper_connection"} <= operations:
        return "deepseek-v4"
    if objective in {"semantic_moe_jepa_evo", "semantic_router", "semantic_router_jepa"}:
        return "semantic-router-moe"
    has_jepa = bool(
        {
            "jepa_mask",
            "jepa_projector",
            "jepa_predictor",
            "latent_mse_loss",
        }
        & operations
    )
    has_semantic = bool(
        {
            "semantic_projector",
            "semantic_alignment_loss",
            "semantic_chunk_projector",
            "semantic_chunk_hasher",
        }
        & operations
    )
    has_moe = bool({"expert_dispatch", "expert_combine", "topk_route"} & operations)
    if objective in {"semantic_dense_jepa_evo", "jepa_semantic"}:
        return "semantic-dense-jepa"
    if has_jepa and has_semantic and backbone == "llama":
        return "semantic-dense-jepa"
    if "semantic_moe_jepa_evo_router" in operations:
        return "semantic-router-moe"
    if has_jepa and has_moe:
        return "moe-jepa-evo"
    if has_jepa:
        return "jepa"
    if {"semantic_moe_router", "semantic_hash_router"} & operations:
        return "semantic-router-moe"

    model_family = _normalize_name(model.get("family"))
    if model_family in _NATIVE_TRAIN_FAMILY_TARGETS:
        return model_family
    if backbone in _NATIVE_TRAIN_FAMILY_TARGETS:
        return backbone
    if backbone in {"gpt2", "nanogpt", "llama", "mixllama"}:
        return backbone
    return ""


def classify_native_model(
    model_mapping: Mapping[str, Any],
    topology_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify a manifest model/topology without runtime or filesystem probes.

    The returned mapping is intentionally composed only of scalar values and
    tuples, with every tuple sorted or explicitly ordered.
    """

    model = _mapping(model_mapping)
    topology = _mapping(topology_mapping)
    _spec, template, block = _template_parts(model)
    objective = str(model.get("objective") or template.get("objective") or "unknown").strip().lower()
    backbone = _normalize_name(
        model.get("backbone")
        or model.get("family")
        or template.get("backbone")
        or block.get("family")
    ) or "unknown"
    compression = _normalize_name(
        model.get("compression")
        or template.get("compression")
        or block.get("compression")
    ) or "none"
    module_types = _topology_module_types(topology)
    active_module_types = _active_topology_module_types(topology)
    operations = frozenset(active_module_types)
    trainer_family = _trainer_family(
        model,
        objective=objective,
        backbone=backbone,
        module_types=active_module_types,
    )

    if objective == "diffusion":
        family_class = "text_diffusion"
    elif objective == "seq2seq":
        family_class = "seq2seq"
    elif compression == "kv-pca" or {"kv_pca_encode", "kv_pca_decode"} & operations:
        family_class = "pca_kv"
    elif "multi_latent_attention" in operations:
        family_class = "latent_kv"
    elif trainer_family in {"jamba", "ttt-llama", "universal-llama", "hnet-lm"}:
        family_class = "hybrid_state"
    elif trainer_family or backbone in {"gpt2", "nanogpt", "llama", "mixllama"}:
        family_class = "autoregressive_transformer"
    else:
        family_class = "unknown"

    if family_class == "text_diffusion":
        session_state_kinds = ("denoising",)
        turboquant_policy = "unsupported-no-kv-cache"
    elif family_class == "seq2seq":
        session_state_kinds = ("encoder", "decoder_kv", "cross_kv")
        turboquant_policy = "decoder-and-cross-attention-kv"
    elif family_class == "pca_kv":
        session_state_kinds = ("pca_kv",)
        turboquant_policy = "unsupported-native-pca-codec-required"
    elif family_class == "latent_kv":
        session_state_kinds = ("latent_kv",)
        turboquant_policy = "unsupported-native-latent-codec-required"
    elif trainer_family == "jamba":
        session_state_kinds = ("attention_kv", "mamba")
        turboquant_policy = "attention-kv-only"
    elif trainer_family == "ttt-llama":
        session_state_kinds = ("attention_kv", "ttt")
        turboquant_policy = "attention-kv-only"
    elif trainer_family == "universal-llama":
        session_state_kinds = ("attention_kv", "recurrent", "act")
        turboquant_policy = "attention-kv-only"
    elif trainer_family == "hnet-lm":
        session_state_kinds = ("attention_kv", "byte_patch")
        turboquant_policy = "attention-kv-only"
    elif family_class == "autoregressive_transformer":
        session_state_kinds = ("kv",)
        turboquant_policy = "standard-retained-kv"
    else:
        session_state_kinds = ()
        turboquant_policy = "unsupported"

    text_generation = objective in _TEXT_OBJECTIVES and trainer_family != "embedding"
    return {
        "model_family": trainer_family or "unknown",
        "family_class": family_class,
        "objective": objective,
        "backbone": backbone,
        "compression": compression,
        "text_generation": text_generation,
        "session_state_kinds": session_state_kinds,
        "turboquant_policy": turboquant_policy,
        "module_types": module_types,
        "active_module_types": active_module_types,
    }


def _resident_dense_v5_graph_compatible(
    model_mapping: Mapping[str, Any],
    topology_mapping: Mapping[str, Any],
    classification: Mapping[str, Any],
) -> bool:
    """Return whether the graph matches the reviewed dense-v5 resident ABI.

    Family-level trainer registration is intentionally insufficient here:
    Differential attention, RoPE/GQA, adapters, and compressed linear variants
    have different forward semantics even when they share a dense trainer
    selector. MoA is accepted only for its exact shared-backbone graph contract;
    a materialized artifact still requires the separately validated,
    source-bound selected-activation checkpoint sidecar. Parameter-free QK RMS
    normalization and output-logit soft-capping are accepted only with exact
    active topology/config proof.
    """

    if str(classification.get("model_family")) not in _DENSE_FORWARD_FAMILIES:
        return False
    if str(classification.get("family_class")) != "autoregressive_transformer":
        return False
    if str(classification.get("compression")) != "none":
        return False
    active_operations = frozenset(str(value) for value in classification.get("active_module_types", ()))
    required_operations = frozenset(
        {
            "absolute_position_embedding",
            "gelu",
            "layer_norm",
            "linear",
            "scaled_dot_product_attention",
            "tied_lm_head",
            "token_embedding",
        }
    )
    allowed_operations = required_operations | {
        "dropout",
        "logit_softcap",
        "merge_heads",
        "qk_norm",
        "reshape_heads",
        "token_cross_entropy",
    }
    if not required_operations.issubset(active_operations):
        return False
    if not active_operations.issubset(allowed_operations):
        return False
    model = _mapping(model_mapping)
    spec, template, block = _template_parts(model)
    expected = {
        "norm_type": "layernorm",
        "mlp_type": "gelu",
        "pos_encoding": "absolute",
        "attention_variant": "dense",
        "residual_type": "add",
        "compression": "none",
    }
    for field, required_value in expected.items():
        if field not in block or _normalize_name(block[field]) != _normalize_name(required_value):
            return False
    activation_mode = _normalize_name(block.get("activation_mode"))
    if activation_mode not in {"single", "moa"}:
        return False
    if activation_mode == "moa":
        candidates = block.get("moa_activations")
        if (
            not isinstance(candidates, (list, tuple))
            or tuple(_normalize_name(value) for value in candidates)
            != ("gelu", "relu", "silu", "relu2")
        ):
            return False
        interval = block.get("moa_interval")
        if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
            return False
    if block.get("linear_bias") is not True:
        return False
    if "use_qk_norm" not in block:
        return False
    use_qk_norm = block["use_qk_norm"]
    if not isinstance(use_qk_norm, bool):
        return False
    try:
        if "dropout_p" not in block or isinstance(block["dropout_p"], bool):
            return False
        if float(block["dropout_p"]) != 0.0:
            return False
        if "logit_softcap" not in spec or isinstance(spec["logit_softcap"], bool):
            return False
        logit_softcap = float(spec["logit_softcap"])
        if any(
            isinstance(value, bool)
            for value in (
                spec.get("num_layers"),
                spec.get("model_dim"),
                spec.get("vocab_size"),
                block.get("num_heads"),
            )
        ):
            return False
        num_layers = int(spec.get("num_layers"))
        model_dim = int(spec.get("model_dim"))
        vocab_size = int(spec.get("vocab_size"))
        num_heads = int(block.get("num_heads"))
    except (TypeError, ValueError):
        return False
    if (
        not math.isfinite(logit_softcap)
        or logit_softcap < 0.0
        or num_layers <= 0
        or model_dim <= 0
        or vocab_size <= 0
        or num_heads <= 0
        or model_dim % num_heads
        or (activation_mode == "moa" and (use_qk_norm or logit_softcap != 0.0))
    ):
        return False
    topology = _mapping(topology_mapping)
    if not _dense_parameter_free_topology_compatible(
        topology,
        num_layers=num_layers,
        use_qk_norm=use_qk_norm,
        logit_softcap=logit_softcap,
    ):
        return False
    if spec.get("tie_embeddings") is not True:
        return False
    if "adapter" not in template or _normalize_name(template["adapter"]) != "none":
        return False
    num_kv_heads = block.get("num_kv_heads")
    try:
        if (
            num_kv_heads not in (None, "")
            and (isinstance(num_kv_heads, bool) or int(num_kv_heads) != num_heads)
        ):
            return False
    except (TypeError, ValueError):
        return False
    return True


def _resident_llama_v1_graph_compatible(
    model_mapping: Mapping[str, Any],
    topology_mapping: Mapping[str, Any],
    classification: Mapping[str, Any],
) -> bool:
    """Recognize only reviewed canonical lossless LLaMA runtime profiles.

    This is deliberately an exact active-topology proof rather than a family
    allow-list.  Neighboring LLaMA presets reuse the same trainer-family label,
    but introduce distinct normalization, attention, compression, or position
    semantics that the first CPU reference adapter does not implement.  The
    eager ``llama`` and compile ``llama_fast`` profiles are accepted separately
    because their active graph and model semantics are identical.
    """

    if str(classification.get("model_family")) != "llama":
        return False
    if str(classification.get("family_class")) != "autoregressive_transformer":
        return False
    if str(classification.get("objective")) != "ar":
        return False
    if str(classification.get("backbone")) != "llama":
        return False
    if str(classification.get("compression")) != "none":
        return False

    expected_active_operations = frozenset(
        {
            "linear",
            "lm_head",
            "merge_heads",
            "repeat_kv",
            "reshape_heads",
            "rms_norm",
            "rotary_embedding",
            "scaled_dot_product_attention",
            "swiglu",
            "token_cross_entropy",
            "token_embedding",
        }
    )
    active_operations = frozenset(
        str(value) for value in classification.get("active_module_types", ())
    )

    model = _mapping(model_mapping)
    if _normalize_name(model.get("family")) != "llama":
        return False
    if _normalize_name(model.get("backbone")) != "llama":
        return False
    spec, template, block = _template_parts(model)
    expected_template = {
        "objective": "ar",
        "backbone": "llama",
        "tokenization": "sp",
        "sparsity": "dense",
        "router_mode": "none",
        "compression": "none",
        "adapter": "none",
    }
    if any(
        field not in template
        or _normalize_name(template[field]) != _normalize_name(required)
        for field, required in expected_template.items()
    ):
        return False
    runtime = _normalize_name(template.get("runtime"))
    if runtime not in {"eager", "compile"}:
        return False
    backend_capabilities = template.get("backend_capabilities")
    expected_backend_capabilities = {
        "compile": runtime == "compile",
        "sdpa": True,
        "cache": True,
        "quantized_export": True,
        "megakernel": False,
    }
    if (
        not isinstance(backend_capabilities, Mapping)
        or dict(backend_capabilities) != expected_backend_capabilities
    ):
        return False
    expected_block = {
        "family": "llama",
        "norm_type": "rmsnorm",
        "mlp_type": "swiglu",
        "pos_encoding": "rope",
        "attention_backend": "sdpa",
        "attention_variant": "dense",
        "residual_type": "add",
        "compression": "none",
        "adapter_type": "none",
        "activation_mode": "single",
    }
    if any(
        field not in block
        or _normalize_name(block[field]) != _normalize_name(required)
        for field, required in expected_block.items()
    ):
        return False
    for field, required in {
        "is_causal": True,
        "linear_bias": False,
        "use_qk_norm": False,
    }.items():
        if field not in block or block[field] is not required:
            return False
    if "rope_scaling" not in block or block["rope_scaling"] is not None:
        return False
    try:
        integer_values = (
            spec.get("num_layers"),
            spec.get("model_dim"),
            spec.get("vocab_size"),
            block.get("num_heads"),
            block.get("num_kv_heads"),
            block.get("multiple_of"),
            block.get("adapter_dim"),
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in integer_values
        ):
            return False
        float_values = (
            block.get("dropout_p"),
            block.get("rope_theta"),
            block.get("mlp_multiplier"),
            spec.get("logit_softcap"),
            spec.get("z_loss_coef"),
        )
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in float_values
        ):
            return False
        num_layers = integer_values[0]
        model_dim = integer_values[1]
        vocab_size = integer_values[2]
        num_heads = integer_values[3]
        num_kv_heads = integer_values[4]
        multiple_of = integer_values[5]
        adapter_dim = integer_values[6]
        dropout_p = float(float_values[0])
        rope_theta = float(float_values[1])
        mlp_multiplier = float(float_values[2])
        logit_softcap = float(float_values[3])
        z_loss_coef = float(float_values[4])
    except (TypeError, ValueError):
        return False
    if (
        num_layers <= 0
        or model_dim <= 0
        or vocab_size <= 0
        or num_heads <= 0
        or num_kv_heads <= 0
        or num_kv_heads > num_heads
        or model_dim % num_heads
        or num_heads % num_kv_heads
        or (model_dim // num_heads) % 2
        or multiple_of <= 0
        or dropout_p != 0.0
        or adapter_dim != 0
        or rope_theta != 10_000.0
        or not math.isfinite(mlp_multiplier)
        or mlp_multiplier <= 0.0
        or logit_softcap != 0.0
        or z_loss_coef != 0.0
    ):
        return False
    if spec.get("tie_embeddings") is not False:
        return False

    uses_gqa_repeat = num_kv_heads != num_heads
    if not uses_gqa_repeat:
        expected_active_operations = expected_active_operations - {"repeat_kv"}
    if active_operations != expected_active_operations:
        return False

    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    active_graphs = _active_topology_graphs(_mapping(topology_mapping))
    if active_graphs is None:
        return False
    graphs_by_path: dict[str, Mapping[str, Any]] = {}
    for graph in active_graphs:
        path = str(graph.get("path") or "")
        if not path or path in graphs_by_path:
            return False
        graphs_by_path[path] = graph
    expected_paths = {"root", "root/nodes/model/subgraph"}
    for layer in range(num_layers):
        block_path = f"root/nodes/model/subgraph/nodes/block_{layer}/subgraph"
        expected_paths.update(
            {
                block_path,
                f"{block_path}/nodes/attention/subgraph",
                f"{block_path}/nodes/mlp/subgraph",
            }
        )
    if set(graphs_by_path) != expected_paths:
        return False

    root = graphs_by_path["root"]
    if _exact_instance_nodes(
        root,
        {
            "tokens_in": "graph.input",
            "targets_in": "graph.input",
            "model": "subgraph.call",
            "loss_out": "graph.output",
        },
    ) is None or not _has_exact_instance_edges(
        root,
        (
            ("tokens_in", 0, "model", 0),
            ("targets_in", 0, "model", 1),
            ("model", 0, "loss_out", 0),
        ),
    ):
        return False

    model_graph = graphs_by_path["root/nodes/model/subgraph"]
    model_operations = {
        "tokens_in": "graph.input",
        "targets_in": "graph.input",
        "token_embed": "token_embedding",
        **{f"block_{layer}": "subgraph.call" for layer in range(num_layers)},
        "final_norm": "rms_norm",
        "lm_head": "lm_head",
        "ce": "token_cross_entropy",
        "loss_out": "graph.output",
    }
    model_nodes = _exact_instance_nodes(model_graph, model_operations)
    model_edges: list[tuple[str, int, str, int]] = [
        ("tokens_in", 0, "token_embed", 0),
        ("token_embed", 0, "block_0", 0),
        (f"block_{num_layers - 1}", 0, "final_norm", 0),
        ("final_norm", 0, "lm_head", 0),
        ("lm_head", 0, "ce", 0),
        ("targets_in", 0, "ce", 1),
        ("ce", 0, "loss_out", 0),
    ]
    model_edges.extend(
        (f"block_{layer}", 0, f"block_{layer + 1}", 0)
        for layer in range(num_layers - 1)
    )
    if model_nodes is None or not _has_exact_instance_edges(model_graph, model_edges):
        return False
    if not _node_config_matches(
        model_nodes["token_embed"],
        {"vocab_size": vocab_size, "model_dim": model_dim},
    ):
        return False
    if not _node_config_matches(
        model_nodes["final_norm"], {"eps": 1.0e-6, "model_dim": model_dim}
    ):
        return False
    if not _node_config_matches(
        model_nodes["lm_head"],
        {"vocab_size": vocab_size, "model_dim": model_dim},
    ):
        return False
    if not _node_config_matches(model_nodes["ce"], {"z_loss_coef": 0.0}):
        return False

    for layer in range(num_layers):
        block_path = f"root/nodes/model/subgraph/nodes/block_{layer}/subgraph"
        block_graph = graphs_by_path[block_path]
        block_nodes = _exact_instance_nodes(
            block_graph,
            {
                "x_in": "graph.input",
                "attn_norm": "rms_norm",
                "attention": "subgraph.call",
                "attn_add": "builtin.add",
                "mlp_norm": "rms_norm",
                "mlp": "subgraph.call",
                "mlp_add": "builtin.add",
                "x_out": "graph.output",
            },
        )
        if block_nodes is None or not _has_exact_instance_edges(
            block_graph,
            (
                ("x_in", 0, "attn_norm", 0),
                ("attn_norm", 0, "attention", 0),
                ("x_in", 0, "attn_add", 0),
                ("attention", 0, "attn_add", 1),
                ("attn_add", 0, "mlp_norm", 0),
                ("mlp_norm", 0, "mlp", 0),
                ("attn_add", 0, "mlp_add", 0),
                ("mlp", 0, "mlp_add", 1),
                ("mlp_add", 0, "x_out", 0),
            ),
        ):
            return False
        if not _node_config_matches(
            block_nodes["attn_norm"], {"eps": 1.0e-6, "model_dim": model_dim}
        ) or not _node_config_matches(
            block_nodes["mlp_norm"], {"eps": 1.0e-6, "model_dim": model_dim}
        ):
            return False

        attention_graph = graphs_by_path[f"{block_path}/nodes/attention/subgraph"]
        attention_operations = {
            "x_in": "graph.input",
            "q_proj": "linear",
            "k_proj": "linear",
            "v_proj": "linear",
            "q_heads": "reshape_heads",
            "k_heads": "reshape_heads",
            "v_heads": "reshape_heads",
            "rope": "rotary_embedding",
            "sdpa": "scaled_dot_product_attention",
            "merge": "merge_heads",
            "out_proj": "linear",
            "attn_out": "graph.output",
        }
        if uses_gqa_repeat:
            attention_operations.update(
                {"k_repeat": "repeat_kv", "v_repeat": "repeat_kv"}
            )
        attention_nodes = _exact_instance_nodes(attention_graph, attention_operations)
        attention_edges: list[tuple[str, int, str, int]] = [
            ("x_in", 0, "q_proj", 0),
            ("x_in", 0, "k_proj", 0),
            ("x_in", 0, "v_proj", 0),
            ("q_proj", 0, "q_heads", 0),
            ("k_proj", 0, "k_heads", 0),
            ("v_proj", 0, "v_heads", 0),
            ("q_heads", 0, "rope", 0),
            ("k_heads", 0, "rope", 1),
            ("rope", 0, "sdpa", 0),
            ("sdpa", 0, "merge", 0),
            ("merge", 0, "out_proj", 0),
            ("out_proj", 0, "attn_out", 0),
        ]
        if uses_gqa_repeat:
            attention_edges.extend(
                (
                    ("rope", 1, "k_repeat", 0),
                    ("k_repeat", 0, "sdpa", 1),
                    ("v_heads", 0, "v_repeat", 0),
                    ("v_repeat", 0, "sdpa", 2),
                )
            )
        else:
            attention_edges.extend(
                (("rope", 1, "sdpa", 1), ("v_heads", 0, "sdpa", 2))
            )
        if attention_nodes is None or not _has_exact_instance_edges(
            attention_graph, attention_edges
        ):
            return False
        for instance_id, input_dim, output_dim in (
            ("q_proj", model_dim, model_dim),
            ("k_proj", model_dim, kv_dim),
            ("v_proj", model_dim, kv_dim),
            ("out_proj", model_dim, model_dim),
        ):
            if not _node_config_matches(
                attention_nodes[instance_id],
                {"input_dim": input_dim, "output_dim": output_dim, "bias": False},
            ):
                return False
        for instance_id, heads in (
            ("q_heads", num_heads),
            ("k_heads", num_kv_heads),
            ("v_heads", num_kv_heads),
        ):
            if not _node_config_matches(attention_nodes[instance_id], {"num_heads": heads}):
                return False
        if not _node_config_matches(
            attention_nodes["rope"],
            {"head_dim": head_dim, "rope_base": 10_000.0, "rope_scaling": None},
        ):
            return False
        if not _node_config_matches(
            attention_nodes["sdpa"],
            {"is_causal": True, "backend": "sdpa", "dropout_p": 0.0},
        ):
            return False
        if uses_gqa_repeat:
            for instance_id in ("k_repeat", "v_repeat"):
                if not _node_config_matches(
                    attention_nodes[instance_id],
                    {"num_heads": num_heads, "num_kv_heads": num_kv_heads},
                ):
                    return False

        mlp_graph = graphs_by_path[f"{block_path}/nodes/mlp/subgraph"]
        mlp_nodes = _exact_instance_nodes(
            mlp_graph,
            {
                "x_in": "graph.input",
                "swiglu": "swiglu",
                "y_out": "graph.output",
            },
        )
        if mlp_nodes is None or not _has_exact_instance_edges(
            mlp_graph,
            (("x_in", 0, "swiglu", 0), ("swiglu", 0, "y_out", 0)),
        ):
            return False
        if not _node_config_matches(
            mlp_nodes["swiglu"],
            {
                "model_dim": model_dim,
                "mlp_mult": mlp_multiplier,
                "multiple_of": multiple_of,
            },
        ):
            return False
    return True


def _canonical_rope_gqa_attention_graph_compatible(
    graph: Mapping[str, Any],
    *,
    model_dim: int,
    num_heads: int,
    num_kv_heads: int,
    rope_theta: float,
) -> bool:
    """Validate the shared biasless RoPE/GQA attention subgraph exactly."""

    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    uses_gqa_repeat = num_kv_heads != num_heads
    operations = {
        "x_in": "graph.input",
        "q_proj": "linear",
        "k_proj": "linear",
        "v_proj": "linear",
        "q_heads": "reshape_heads",
        "k_heads": "reshape_heads",
        "v_heads": "reshape_heads",
        "rope": "rotary_embedding",
        "sdpa": "scaled_dot_product_attention",
        "merge": "merge_heads",
        "out_proj": "linear",
        "attn_out": "graph.output",
    }
    if uses_gqa_repeat:
        operations.update({"k_repeat": "repeat_kv", "v_repeat": "repeat_kv"})
    nodes = _exact_instance_nodes(graph, operations)
    edges: list[tuple[str, int, str, int]] = [
        ("x_in", 0, "q_proj", 0),
        ("x_in", 0, "k_proj", 0),
        ("x_in", 0, "v_proj", 0),
        ("q_proj", 0, "q_heads", 0),
        ("k_proj", 0, "k_heads", 0),
        ("v_proj", 0, "v_heads", 0),
        ("q_heads", 0, "rope", 0),
        ("k_heads", 0, "rope", 1),
        ("rope", 0, "sdpa", 0),
        ("sdpa", 0, "merge", 0),
        ("merge", 0, "out_proj", 0),
        ("out_proj", 0, "attn_out", 0),
    ]
    if uses_gqa_repeat:
        edges.extend(
            (
                ("rope", 1, "k_repeat", 0),
                ("k_repeat", 0, "sdpa", 1),
                ("v_heads", 0, "v_repeat", 0),
                ("v_repeat", 0, "sdpa", 2),
            )
        )
    else:
        edges.extend((("rope", 1, "sdpa", 1), ("v_heads", 0, "sdpa", 2)))
    if nodes is None or not _has_exact_instance_edges(graph, edges):
        return False
    for instance_id, input_dim, output_dim in (
        ("q_proj", model_dim, model_dim),
        ("k_proj", model_dim, kv_dim),
        ("v_proj", model_dim, kv_dim),
        ("out_proj", model_dim, model_dim),
    ):
        if not _node_config_matches(
            nodes[instance_id],
            {"input_dim": input_dim, "output_dim": output_dim, "bias": False},
        ):
            return False
    for instance_id, heads in (
        ("q_heads", num_heads),
        ("k_heads", num_kv_heads),
        ("v_heads", num_kv_heads),
    ):
        if not _node_config_matches(nodes[instance_id], {"num_heads": heads}):
            return False
    if not _node_config_matches(
        nodes["rope"],
        {"head_dim": head_dim, "rope_base": rope_theta, "rope_scaling": None},
    ) or not _node_config_matches(
        nodes["sdpa"],
        {"is_causal": True, "backend": "sdpa", "dropout_p": 0.0},
    ):
        return False
    if uses_gqa_repeat and any(
        not _node_config_matches(
            nodes[instance_id],
            {"num_heads": num_heads, "num_kv_heads": num_kv_heads},
        )
        for instance_id in ("k_repeat", "v_repeat")
    ):
        return False
    return True


def _resident_standard_moe_v1_graph_compatible(
    model_mapping: Mapping[str, Any],
    topology_mapping: Mapping[str, Any],
    classification: Mapping[str, Any],
) -> bool:
    """Recognize only the reviewed softmax-top-k standard-MoE graph cluster."""

    if (
        str(classification.get("model_family")) != "mixllama"
        or str(classification.get("family_class")) != "autoregressive_transformer"
        or str(classification.get("objective")) != "ar"
        or str(classification.get("backbone")) != "mixllama"
        or str(classification.get("compression")) != "none"
    ):
        return False
    expected_active_operations = frozenset(
        {
            "aux_loss_add",
            "expert_combine",
            "expert_dispatch",
            "linear",
            "lm_head",
            "load_balance_loss",
            "merge_heads",
            "repeat_kv",
            "reshape_heads",
            "rms_norm",
            "rotary_embedding",
            "router_logits",
            "scaled_dot_product_attention",
            "token_cross_entropy",
            "token_embedding",
            "topk_route",
        }
    )
    active_operations = frozenset(
        str(value) for value in classification.get("active_module_types", ())
    )

    model = _mapping(model_mapping)
    if _normalize_name(model.get("family")) != "mixllama" or _normalize_name(
        model.get("backbone")
    ) != "mixllama":
        return False
    spec, template, block = _template_parts(model)
    expected_template = {
        "objective": "ar",
        "backbone": "mixllama",
        "tokenization": "sp",
        "sparsity": "moe",
        "router_mode": "none",
        "compression": "none",
        "adapter": "none",
    }
    if any(
        field not in template
        or _normalize_name(template[field]) != _normalize_name(required)
        for field, required in expected_template.items()
    ):
        return False
    runtime = _normalize_name(template.get("runtime"))
    if runtime not in {"eager", "compile"}:
        return False
    if dict(_mapping(template.get("backend_capabilities"))) != {
        "compile": runtime == "compile",
        "sdpa": True,
        "cache": True,
        "quantized_export": True,
        "megakernel": False,
    }:
        return False
    expected_block = {
        "family": "mixllama",
        "norm_type": "rmsnorm",
        "mlp_type": "moe",
        "pos_encoding": "rope",
        "attention_backend": "sdpa",
        "attention_variant": "dense",
        "residual_type": "add",
        "compression": "none",
        "adapter_type": "none",
        "activation_mode": "single",
        "moe_balance_mode": "aux_loss",
        "router_score_fn": "softmax",
    }
    if any(
        field not in block
        or _normalize_name(block[field]) != _normalize_name(required)
        for field, required in expected_block.items()
    ):
        return False
    if any(
        field not in block or block[field] is not required
        for field, required in {
            "is_causal": True,
            "linear_bias": False,
            "use_qk_norm": False,
        }.items()
    ) or block.get("rope_scaling", object()) is not None:
        return False
    try:
        integer_values = (
            spec.get("num_layers"),
            spec.get("model_dim"),
            spec.get("vocab_size"),
            block.get("num_heads"),
            block.get("num_kv_heads"),
            block.get("adapter_dim"),
            block.get("experts"),
            block.get("top_k"),
            block.get("shared_experts"),
        )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_values):
            return False
        multiple_value = block.get("multiple_of")
        if multiple_value is None:
            multiple_of = 0
        elif isinstance(multiple_value, int) and not isinstance(multiple_value, bool):
            multiple_of = multiple_value
        else:
            return False
        float_values = (
            block.get("dropout_p"),
            block.get("rope_theta"),
            block.get("mlp_multiplier"),
            block.get("router_aux_loss_coef"),
            spec.get("logit_softcap"),
            spec.get("z_loss_coef"),
        )
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in float_values
        ):
            return False
        (
            num_layers,
            model_dim,
            vocab_size,
            num_heads,
            num_kv_heads,
            adapter_dim,
            experts,
            top_k,
            shared_experts,
        ) = integer_values
        dropout_p, rope_theta, mlp_multiplier, router_aux_loss_coef, logit_softcap, z_loss_coef = (
            float(value) for value in float_values
        )
    except (TypeError, ValueError):
        return False
    positive_float32 = router_aux_loss_coef == 0.0 or (
        1.401298464324817e-45 <= router_aux_loss_coef <= 3.4028234663852886e38
    )
    if (
        num_layers <= 0
        or model_dim <= 0
        or vocab_size <= 0
        or num_heads <= 0
        or num_kv_heads <= 0
        or num_kv_heads > num_heads
        or model_dim % num_heads
        or num_heads % num_kv_heads
        or (model_dim // num_heads) % 2
        or adapter_dim != 0
        or experts <= 0
        or top_k <= 0
        or top_k > experts
        or shared_experts != 0
        or multiple_of < 0
        or dropout_p != 0.0
        or rope_theta != 10_000.0
        or not math.isfinite(mlp_multiplier)
        or mlp_multiplier <= 0.0
        or not math.isfinite(router_aux_loss_coef)
        or router_aux_loss_coef < 0.0
        or not positive_float32
        or logit_softcap != 0.0
        or z_loss_coef != 0.0
        or spec.get("tie_embeddings") is not False
    ):
        return False
    if num_kv_heads == num_heads:
        expected_active_operations = expected_active_operations - {"repeat_kv"}
    if active_operations != expected_active_operations:
        return False

    active_graphs = _active_topology_graphs(_mapping(topology_mapping))
    if active_graphs is None:
        return False
    graphs_by_path: dict[str, Mapping[str, Any]] = {}
    for graph in active_graphs:
        path = str(graph.get("path") or "")
        if not path or path in graphs_by_path:
            return False
        graphs_by_path[path] = graph
    expected_paths = {"root", "root/nodes/model/subgraph"}
    for layer in range(num_layers):
        block_path = f"root/nodes/model/subgraph/nodes/block_{layer}/subgraph"
        expected_paths.update(
            {
                block_path,
                f"{block_path}/nodes/attention/subgraph",
                f"{block_path}/nodes/mlp/subgraph",
            }
        )
    if set(graphs_by_path) != expected_paths:
        return False

    root = graphs_by_path["root"]
    if _exact_instance_nodes(
        root,
        {
            "tokens_in": "graph.input",
            "targets_in": "graph.input",
            "model": "subgraph.call",
            "loss_out": "graph.output",
        },
    ) is None or not _has_exact_instance_edges(
        root,
        (
            ("tokens_in", 0, "model", 0),
            ("targets_in", 0, "model", 1),
            ("model", 0, "loss_out", 0),
        ),
    ):
        return False

    model_graph = graphs_by_path["root/nodes/model/subgraph"]
    model_operations = {
        "tokens_in": "graph.input",
        "targets_in": "graph.input",
        "token_embed": "token_embedding",
        **{f"block_{layer}": "subgraph.call" for layer in range(num_layers)},
        **{f"add_aux_{layer}": "builtin.add" for layer in range(1, num_layers)},
        "final_norm": "rms_norm",
        "lm_head": "lm_head",
        "ce": "token_cross_entropy",
        "total_loss": "aux_loss_add",
        "loss_out": "graph.output",
    }
    model_nodes = _exact_instance_nodes(model_graph, model_operations)
    model_edges: list[tuple[str, int, str, int]] = [
        ("tokens_in", 0, "token_embed", 0),
        ("token_embed", 0, "block_0", 0),
        (f"block_{num_layers - 1}", 0, "final_norm", 0),
        ("final_norm", 0, "lm_head", 0),
        ("lm_head", 0, "ce", 0),
        ("targets_in", 0, "ce", 1),
        ("ce", 0, "total_loss", 0),
        ("total_loss", 0, "loss_out", 0),
    ]
    model_edges.extend(
        (f"block_{layer}", 0, f"block_{layer + 1}", 0)
        for layer in range(num_layers - 1)
    )
    if num_layers == 1:
        model_edges.append(("block_0", 1, "total_loss", 1))
    else:
        model_edges.extend(
            (
                ("block_1", 1, "add_aux_1", 0),
                ("block_0", 1, "add_aux_1", 1),
                (f"add_aux_{num_layers - 1}", 0, "total_loss", 1),
            )
        )
        for layer in range(2, num_layers):
            model_edges.extend(
                (
                    (f"block_{layer}", 1, f"add_aux_{layer}", 0),
                    (f"add_aux_{layer - 1}", 0, f"add_aux_{layer}", 1),
                )
            )
    if model_nodes is None or not _has_exact_instance_edges(model_graph, model_edges):
        return False
    if not _node_config_matches(
        model_nodes["token_embed"], {"vocab_size": vocab_size, "model_dim": model_dim}
    ) or not _node_config_matches(
        model_nodes["final_norm"], {"eps": 1.0e-6, "model_dim": model_dim}
    ) or not _node_config_matches(
        model_nodes["lm_head"], {"vocab_size": vocab_size, "model_dim": model_dim}
    ) or not _node_config_matches(model_nodes["ce"], {"z_loss_coef": 0.0}) or not _node_config_matches(
        model_nodes["total_loss"], {"coef": router_aux_loss_coef}
    ):
        return False

    for layer in range(num_layers):
        block_path = f"root/nodes/model/subgraph/nodes/block_{layer}/subgraph"
        block_graph = graphs_by_path[block_path]
        block_nodes = _exact_instance_nodes(
            block_graph,
            {
                "x_in": "graph.input",
                "attn_norm": "rms_norm",
                "attention": "subgraph.call",
                "attn_add": "builtin.add",
                "mlp_norm": "rms_norm",
                "mlp": "subgraph.call",
                "mlp_add": "builtin.add",
                "x_out": "graph.output",
                "aux_loss_out": "graph.output",
            },
        )
        if block_nodes is None or not _has_exact_instance_edges(
            block_graph,
            (
                ("x_in", 0, "attn_norm", 0),
                ("attn_norm", 0, "attention", 0),
                ("x_in", 0, "attn_add", 0),
                ("attention", 0, "attn_add", 1),
                ("attn_add", 0, "mlp_norm", 0),
                ("mlp_norm", 0, "mlp", 0),
                ("attn_add", 0, "mlp_add", 0),
                ("mlp", 0, "mlp_add", 1),
                ("mlp_add", 0, "x_out", 0),
                ("mlp", 1, "aux_loss_out", 0),
            ),
        ):
            return False
        if not _node_config_matches(
            block_nodes["attn_norm"], {"eps": 1.0e-6, "model_dim": model_dim}
        ) or not _node_config_matches(
            block_nodes["mlp_norm"], {"eps": 1.0e-6, "model_dim": model_dim}
        ):
            return False
        if not _canonical_rope_gqa_attention_graph_compatible(
            graphs_by_path[f"{block_path}/nodes/attention/subgraph"],
            model_dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_theta=rope_theta,
        ):
            return False
        mlp_graph = graphs_by_path[f"{block_path}/nodes/mlp/subgraph"]
        mlp_nodes = _exact_instance_nodes(
            mlp_graph,
            {
                "x_in": "graph.input",
                "router": "router_logits",
                "topk": "topk_route",
                "dispatch": "expert_dispatch",
                "combine": "expert_combine",
                "lb_loss": "load_balance_loss",
                "y_out": "graph.output",
                "aux_loss_out": "graph.output",
            },
        )
        if mlp_nodes is None or not _has_exact_instance_edges(
            mlp_graph,
            (
                ("x_in", 0, "router", 0),
                ("router", 0, "topk", 0),
                ("x_in", 0, "dispatch", 0),
                ("topk", 0, "dispatch", 1),
                ("topk", 1, "dispatch", 2),
                ("dispatch", 0, "combine", 0),
                ("combine", 0, "y_out", 0),
                ("router", 0, "lb_loss", 0),
                ("topk", 0, "lb_loss", 1),
                ("topk", 1, "lb_loss", 2),
                ("lb_loss", 0, "aux_loss_out", 0),
            ),
        ):
            return False
        if not _node_config_matches(
            mlp_nodes["router"], {"model_dim": model_dim, "experts": experts}
        ) or not _node_config_matches(
            mlp_nodes["topk"], {"top_k": top_k, "experts": experts, "score_fn": "softmax"}
        ) or not _node_config_matches(
            mlp_nodes["dispatch"],
            {
                "model_dim": model_dim,
                "experts": experts,
                "mlp_mult": mlp_multiplier,
                "multiple_of": None if multiple_value is None else multiple_of,
            },
        ) or not _node_config_matches(mlp_nodes["lb_loss"], {"experts": experts}):
            return False
    return True


def _resident_dense_turboquant_compatible(model_mapping: Mapping[str, Any]) -> bool:
    """Require geometry supported by the native packed 3.5-bit codec."""

    model = _mapping(model_mapping)
    spec, _template, block = _template_parts(model)
    try:
        model_dim = int(spec.get("model_dim"))
        num_heads = int(block.get("num_heads"))
    except (TypeError, ValueError):
        return False
    if model_dim <= 0 or num_heads <= 0 or model_dim % num_heads:
        return False
    head_dim = model_dim // num_heads
    return head_dim >= 2 and head_dim % 2 == 0


def capability_proof_for(
    model_mapping: Mapping[str, Any],
    topology_mapping: Mapping[str, Any],
) -> NativeCapabilityProof:
    """Return conservative capability evidence for a Native IR candidate.

    Unsupported module operations make structural lowering fail closed.  A
    registered trainer does not imply resident inference, cache support, or
    serving; each remains false unless its topology-specific independent proof
    is present.  Reviewed dense-v5 and canonical LLaMA-v1 topologies use
    separate exact dataflow gates for the resident/lossless/serving slice.  The
    reviewed standard-MoE eager/compile cluster has its own equally strict gate.
    """

    classification = classify_native_model(model_mapping, topology_mapping)
    module_types = tuple(classification["module_types"])
    registered = frozenset(registered_native_module_types())
    unsupported = tuple(sorted(set(module_types) - registered))
    trainer_by_family = {spec.family: spec for spec in native_trainer_specs()}
    trainer = trainer_by_family.get(str(classification["model_family"]))
    lowering_supported = not unsupported

    training_selector = classify_native_graph_training_selector(
        model_mapping, topology_mapping
    )
    nanogpt_selector_gap = _NANOGPT_SELECTOR_GAPS.get(training_selector)
    selector_adapter = next(
        (
            adapter
            for adapter in native_graph_training_adapters()
            if adapter.selector == training_selector
        ),
        None,
    )
    selector_proof_blocked = bool(
        selector_adapter is not None
        and not selector_adapter.architecture_persistence_proven
    )

    evidence = ["native_registry.explicit_module-lowering-v1"]
    if trainer is not None and not selector_proof_blocked:
        evidence.extend(trainer.evidence)
    elif trainer is not None:
        # Family-level dense evidence covers proved selectors such as canonical
        # GPT-2, not every graph-authored selector routed to the same binary.
        evidence.append("native_train.NATIVE_TRAIN_FAMILY_TARGETS")
    if selector_proof_blocked and selector_adapter is not None:
        evidence.extend(selector_adapter.evidence)
    if nanogpt_selector_gap is not None:
        evidence.append(nanogpt_selector_gap[0])
    if unsupported:
        evidence.append("unsupported-module-types-rejected")

    dense_profile = _resident_dense_v5_graph_compatible(
        model_mapping, topology_mapping, classification
    )
    llama_profile = _resident_llama_v1_graph_compatible(
        model_mapping, topology_mapping, classification
    )
    standard_moe_profile = _resident_standard_moe_v1_graph_compatible(
        model_mapping, topology_mapping, classification
    )
    muse_glimmer_profile = _resident_muse_glimmer_v1_graph_compatible(
        model_mapping, topology_mapping, classification
    )
    architecture_persistence = bool(
        trainer is not None
        and not selector_proof_blocked
        and (
            trainer.architecture_persistence_proven
            or llama_profile
            or standard_moe_profile
            or muse_glimmer_profile
        )
    )

    missing: list[str] = []
    if nanogpt_selector_gap is not None:
        missing.extend((*_NANOGPT_MISSING_GATES, nanogpt_selector_gap[1]))
    if unsupported:
        missing.append("native_ir_lowerers:" + ",".join(unsupported))
    if trainer is None:
        missing.append("native_trainer_registration")
    elif not architecture_persistence:
        missing.append("architecture_parameter_persistence_proof")
    native_forward = bool(
        trainer is not None
        and not selector_proof_blocked
        and (
            trainer.native_forward == "one-shot-architecture-forward"
            or llama_profile
            or standard_moe_profile
            or muse_glimmer_profile
        )
    )
    if not native_forward:
        missing.append("real_native_architecture_forward")
    resident_dense = bool(
        lowering_supported
        and native_forward
        and trainer is not None
        and trainer.resident_inference
        and dense_profile
    )
    resident_llama = bool(lowering_supported and native_forward and llama_profile)
    resident_standard_moe = bool(
        lowering_supported and native_forward and standard_moe_profile
    )
    resident_muse_glimmer = bool(
        lowering_supported and native_forward and muse_glimmer_profile
    )
    resident_inference = (
        resident_dense or resident_llama or resident_standard_moe
        or resident_muse_glimmer
    )
    lossless_cache = resident_inference
    turboquant_cache = bool(
        resident_dense and _resident_dense_turboquant_compatible(model_mapping)
    )
    # The lean serving runtime is an in-process consumer of the same resident
    # ABI.  It still validates tokenizer/chat presentation metadata before
    # opening a socket and advertises unsupported higher-level OpenAI features
    # independently, so this proof is intentionally narrower than Responses,
    # tools, structured output, or TurboQuant support.
    # A resident forward is not by itself a chat/model-catalog capability.
    # Embedding, encoder-only, and other non-generative artifacts must remain
    # absent even if a future adapter supplies an in-process resident engine.
    serving = bool(resident_inference and classification["text_generation"])
    if resident_inference and not classification["text_generation"]:
        missing.append("text_generation_model_catalog_eligibility")
    if resident_dense:
        evidence.extend(
            (
                "resident-dense-in-process-abi-v1",
                "resident-dense-lossless-kv-cache-v1",
                "resident-dense-recompute-parity-fixture-v1",
                "resident-dense-chat-completions-serving-v1",
            )
        )
        resident_spec, _resident_template, resident_block = _template_parts(
            _mapping(model_mapping)
        )
        if resident_block.get("use_qk_norm") is True:
            evidence.append("resident-dense-qk-rmsnorm-v1")
        if float(resident_spec.get("logit_softcap", 0.0) or 0.0) > 0.0:
            evidence.append("resident-dense-logit-softcap-v1")
        if _normalize_name(resident_block.get("activation_mode")) == "moa":
            evidence.extend(
                (
                    "native-dense-moa-source-bound-checkpoint-v1",
                    "resident-dense-moa-selected-activation-v1",
                )
            )
        if turboquant_cache:
            evidence.extend(
                (
                    "resident-dense-native-turboquant-cache-v1",
                    "resident-dense-direct-packed-attention-v1",
                    "turboquant-portable-native-codec-agreement-v1",
                )
            )
    elif resident_llama:
        evidence.extend(
            (
                "native-family-llama-float32-checkpoint-v1",
                "native-family-llama-checkpoint-v2-inspector",
                "resident-llama-cpu-reference-abi-v1",
                "resident-llama-gqa-lossless-kv-cache-v1",
                "resident-llama-rope-gqa-right-aligned-v1",
                "resident-llama-lean-serving-abi-v1",
            )
        )
    elif resident_standard_moe:
        evidence.extend(
            (
                "standard-moe-floating-expert-width-v1",
                "standard-moe-exact-router-aux-loss-gradient-v1",
                "native-family-standard-moe-float32-checkpoint-v1",
                "native-family-standard-moe-checkpoint-v1-inspector",
                "resident-standard-moe-cpu-reference-abi-v1",
                "resident-standard-moe-gqa-lossless-kv-cache-v1",
                "resident-standard-moe-softmax-topk-renormalized-v1",
                "resident-standard-moe-lean-serving-abi-v1",
            )
        )
    elif resident_muse_glimmer:
        evidence.extend(
            (
                "muse-glimmer-bf16-and-kquant-strict-checkpoint-v1",
                "resident-muse-glimmer-cpu-cuda-abi-v1",
                "resident-muse-glimmer-hybrid-lossless-kv-cache-v1",
                "resident-muse-glimmer-dflash-feature-abi-v1",
                "muse-glimmer-vram-auto-weight-profile-v1",
            )
        )
    else:
        missing.extend(
            ("resident_inference_adapter", "lossless_native_cache", "native_serving_gate")
        )
    if not turboquant_cache:
        missing.append("turboquant_native_cache")

    session_state_kinds = tuple(classification["session_state_kinds"])
    if resident_inference and "final_hidden_history" not in session_state_kinds:
        session_state_kinds = (*session_state_kinds, "final_hidden_history")

    return NativeCapabilityProof(
        model_family=str(classification["model_family"]),
        family_class=str(classification["family_class"]),
        objective=str(classification["objective"]),
        text_generation=bool(classification["text_generation"]),
        session_state_kinds=session_state_kinds,
        module_types=module_types,
        unsupported_modules=unsupported,
        native_ir_lowering=lowering_supported,
        trainer_registered=trainer is not None and trainer.trainer_registered,
        architecture_persistence_proven=architecture_persistence,
        native_forward_proven=native_forward,
        resident_inference_proven=resident_inference,
        lossless_cache_proven=lossless_cache,
        turboquant_cache_proven=turboquant_cache,
        serving_proven=serving,
        evidence=tuple(dict.fromkeys(evidence)),
        missing_gates=tuple(dict.fromkeys(missing)),
    )


__all__ = [
    "NativeCapabilityProof",
    "NativeGraphTrainingAdapter",
    "NativeLoweringSpec",
    "NativeTrainerSpec",
    "capability_proof_for",
    "classify_native_graph_training_selector",
    "classify_native_model",
    "native_graph_training_adapters",
    "native_lowering_specs",
    "native_trainer_specs",
    "registered_native_module_types",
]
