from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .semantic import DEFAULT_SEMANTIC_VOCAB_REF, NUM_SEMANTIC_DIMS, NUM_VOCAB_DIMS

ObjectiveType = Literal["ar", "diffusion", "jepa", "jepa_semantic", "semantic_router", "seq2seq"]
BackboneType = Literal["gpt2", "nanogpt", "llama", "mixllama", "jamba", "universal", "ttt", "hnet"]
TokenizationType = Literal["sp", "byte_hnet"]
SparsityType = Literal["dense", "moe"]
CompressionType = Literal["none", "ternary_b158", "binary_1bit", "kv_pca"]
AdapterType = Literal["none", "lora", "randmap"]
RuntimeType = Literal["eager", "compile", "sdpa", "megakernel"]


@dataclass
class TemplateSpec:
    objective: ObjectiveType = "ar"
    backbone: BackboneType = "gpt2"
    tokenization: TokenizationType = "sp"
    sparsity: SparsityType = "dense"
    compression: CompressionType = "none"
    adapter: AdapterType = "none"
    runtime: RuntimeType = "eager"
    backend_capabilities: dict[str, bool] = field(default_factory=lambda: {
        "compile": True,
        "sdpa": True,
        "cache": True,
        "quantized_export": True,
        "megakernel": False,
    })


def resolve_backend_capabilities(spec: "TemplateSpec") -> dict[str, bool]:
    """Derive the correct ``backend_capabilities`` from the other fields on *spec*.

    Called automatically by ``_base_model_spec`` so every preset gets a
    consistent capability map without manual per-preset overrides.
    """
    return {
        "compile": spec.runtime in ("compile", "megakernel"),
        "sdpa": True,
        "cache": True,
        "quantized_export": True,
        "megakernel": spec.runtime == "megakernel",
    }


@dataclass
class BlockSpec:
    family: str  # "nanogpt" | "gpt2" | "llama" | "mixllama"
    norm_type: str = "layernorm"  # "layernorm" | "rmsnorm"
    mlp_type: str = "gelu"  # "gelu" | "swiglu" | "moe"
    pos_encoding: str = "absolute"  # "absolute" | "rope"
    attention_backend: str = "sdpa"  # "sdpa" | "flex" | "math"
    num_heads: int = 4
    num_kv_heads: int | None = None  # None => MHA, smaller => GQA/MQA
    is_causal: bool = True
    linear_bias: bool = True
    dropout_p: float = 0.0
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    mlp_multiplier: float = 4.0
    multiple_of: int | None = None
    experts: int | None = None
    top_k: int | None = None
    shared_experts: int = 0
    router_aux_loss_coef: float = 0.0
    compression: str = "none"
    adapter_dim: int = 0
    ttt_hidden_dim: int = 16
    byte_patch_size: int = 4
    byte_patch_stride: int = 4
    qk_gain_init: float = 1.0


@dataclass
class ModelSpec:
    model_dim: int = 128
    num_layers: int = 4
    vocab_size: int = 256
    tie_embeddings: bool = True
    logit_softcap: float = 0.0  # 0.0 = disabled; >0.0 = tanh softcap (Gemma, PaLM)
    block_spec: BlockSpec = field(default_factory=lambda: BlockSpec(family="gpt2"))
    template: TemplateSpec = field(default_factory=TemplateSpec)
    jepa_latent_dim: int = 128
    jepa_mask_ratio: float = 0.5
    jepa_mask_strategy: str = "random"
    jepa_num_blocks: int = 4
    jepa_min_block_ratio: float = 0.1
    jepa_max_block_ratio: float = 0.25
    ema_decay: float = 0.99
    max_recurrence_steps: int = 4
    halt_epsilon: float = 0.01
    semantic_dim: int = NUM_SEMANTIC_DIMS
    semantic_residual_dim: int = 64
    semantic_n_lsh_tables: int = 8
    semantic_n_lsh_planes: int = 12
    semantic_table_path: str = ""
    semantic_vocab_ref: str = DEFAULT_SEMANTIC_VOCAB_REF
    experimental_semantic_router_vecs: bool = False
    ar_loss_coef: float = 1.0
    jepa_loss_coef: float = 0.25
    semantic_align_loss_coef: float = 0.5


def model_spec_to_dict(spec: ModelSpec) -> dict[str, Any]:
    return asdict(spec)


def _base_model_spec(
    *,
    kwargs: dict[str, Any],
    template: TemplateSpec,
    block_spec: BlockSpec,
    default_tie_embeddings: bool,
) -> ModelSpec:
    template.backend_capabilities = resolve_backend_capabilities(template)
    model_dim = int(kwargs.get("model_dim", kwargs.get("n_embd", 128)))
    return ModelSpec(
        model_dim=model_dim,
        num_layers=int(kwargs.get("num_layers", kwargs.get("n_layer", 4))),
        vocab_size=int(kwargs.get("vocab_size", 256)),
        tie_embeddings=bool(kwargs.get("tie_embeddings", default_tie_embeddings)),
        logit_softcap=float(kwargs.get("logit_softcap", 0.0)),
        block_spec=block_spec,
        template=template,
        jepa_latent_dim=int(kwargs.get("jepa_latent_dim", model_dim)),
        jepa_mask_ratio=float(kwargs.get("jepa_mask_ratio", 0.5)),
        jepa_mask_strategy=str(kwargs.get("jepa_mask_strategy", "random")),
        jepa_num_blocks=int(kwargs.get("jepa_num_blocks", 4)),
        jepa_min_block_ratio=float(kwargs.get("jepa_min_block_ratio", 0.1)),
        jepa_max_block_ratio=float(kwargs.get("jepa_max_block_ratio", 0.25)),
        ema_decay=float(kwargs.get("ema_decay", 0.99)),
        max_recurrence_steps=int(kwargs.get("max_recurrence_steps", 4)),
        halt_epsilon=float(kwargs.get("halt_epsilon", 0.01)),
        semantic_dim=int(kwargs.get("semantic_dim", NUM_SEMANTIC_DIMS)),
        semantic_residual_dim=int(kwargs.get("semantic_residual_dim", 64)),
        semantic_n_lsh_tables=int(kwargs.get("semantic_n_lsh_tables", 8)),
        semantic_n_lsh_planes=int(kwargs.get("semantic_n_lsh_planes", 12)),
        semantic_table_path=str(kwargs.get("semantic_table_path", "")),
        semantic_vocab_ref=str(kwargs.get("semantic_vocab_ref", DEFAULT_SEMANTIC_VOCAB_REF)),
        experimental_semantic_router_vecs=bool(kwargs.get("experimental_semantic_router_vecs", False)),
        ar_loss_coef=float(kwargs.get("ar_loss_coef", 1.0)),
        jepa_loss_coef=float(kwargs.get("jepa_loss_coef", 0.25)),
        semantic_align_loss_coef=float(kwargs.get("semantic_align_loss_coef", 0.5)),
    )


def _build_nanogpt_runtime_spec(*, runtime: str, **kwargs: Any) -> ModelSpec:
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="nanogpt", runtime=runtime),
        block_spec=BlockSpec(
            family="nanogpt",
            norm_type="layernorm",
            mlp_type="gelu",
            pos_encoding="absolute",
            linear_bias=kwargs.get("bias", False),
            dropout_p=kwargs.get("dropout_p", 0.1),
            num_heads=kwargs.get("num_heads", 4),
        ),
        default_tie_embeddings=True,
    )


def build_nanogpt_spec(**kwargs: Any) -> ModelSpec:
    return _build_nanogpt_runtime_spec(runtime="eager", **kwargs)


def build_nanogpt_megakernel_spec(**kwargs: Any) -> ModelSpec:
    return _build_nanogpt_runtime_spec(runtime="megakernel", **kwargs)


def _build_gpt2_runtime_spec(*, runtime: str, **kwargs: Any) -> ModelSpec:
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="gpt2", runtime=runtime),
        block_spec=BlockSpec(
            family="gpt2",
            norm_type="layernorm",
            mlp_type="gelu",
            pos_encoding="absolute",
            linear_bias=True,
            num_heads=kwargs.get("num_heads", 4),
        ),
        default_tie_embeddings=True,
    )


def build_gpt2_spec(**kwargs: Any) -> ModelSpec:
    return _build_gpt2_runtime_spec(runtime="eager", **kwargs)


def build_gpt2_megakernel_spec(**kwargs: Any) -> ModelSpec:
    return _build_gpt2_runtime_spec(runtime="megakernel", **kwargs)


def build_llama_spec(**kwargs: Any) -> ModelSpec:
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="llama", runtime="eager"),
        block_spec=BlockSpec(
            family="llama",
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_encoding="rope",
            linear_bias=False,
            dropout_p=0.0,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            multiple_of=kwargs.get("multiple_of", 256),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
        ),
        default_tie_embeddings=False,
    )


def build_mixllama_spec(**kwargs: Any) -> ModelSpec:
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="mixllama", sparsity="moe", runtime="eager"),
        block_spec=BlockSpec(
            family="mixllama",
            norm_type="rmsnorm",
            mlp_type="moe",
            pos_encoding="rope",
            linear_bias=False,
            dropout_p=0.0,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            experts=kwargs.get("experts", 8),
            top_k=kwargs.get("top_k", 2),
            router_aux_loss_coef=kwargs.get("router_aux_loss_coef", 0.01),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
        ),
        default_tie_embeddings=False,
    )


def _build_llama_fast_runtime_spec(*, runtime: RuntimeType, **kwargs: Any) -> ModelSpec:
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(
            objective="ar",
            backbone="llama",
            tokenization="sp",
            sparsity="dense",
            compression="none",
            adapter="none",
            runtime=runtime,
        ),
        block_spec=BlockSpec(
            family="llama",
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_encoding="rope",
            linear_bias=False,
            dropout_p=0.0,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            multiple_of=kwargs.get("multiple_of", 256),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
            rope_theta=kwargs.get("rope_base", kwargs.get("rope_theta", 10_000.0)),
            qk_gain_init=kwargs.get("qk_gain_init", 1.0),
            attention_backend="sdpa",
        ),
        default_tie_embeddings=False,
    )


def build_llama_fast_spec(**kwargs: Any) -> ModelSpec:
    return _build_llama_fast_runtime_spec(runtime="compile", **kwargs)


def build_llama_fast_megakernel_spec(**kwargs: Any) -> ModelSpec:
    return _build_llama_fast_runtime_spec(runtime="megakernel", **kwargs)


def _build_mixllama_fast_runtime_spec(*, runtime: RuntimeType, **kwargs: Any) -> ModelSpec:
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(
            objective="ar",
            backbone="mixllama",
            tokenization="sp",
            sparsity="moe",
            compression="none",
            adapter="none",
            runtime=runtime,
        ),
        block_spec=BlockSpec(
            family="mixllama",
            norm_type="rmsnorm",
            mlp_type="moe",
            pos_encoding="rope",
            linear_bias=False,
            dropout_p=0.0,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            experts=kwargs.get("experts", 8),
            top_k=kwargs.get("top_k", 2),
            router_aux_loss_coef=kwargs.get("router_aux_loss_coef", 0.01),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
            rope_theta=kwargs.get("rope_base", kwargs.get("rope_theta", 10_000.0)),
            qk_gain_init=kwargs.get("qk_gain_init", 1.0),
            attention_backend="sdpa",
        ),
        default_tie_embeddings=False,
    )


def build_mixllama_fast_spec(**kwargs: Any) -> ModelSpec:
    return _build_mixllama_fast_runtime_spec(runtime="compile", **kwargs)


def build_mixllama_fast_megakernel_spec(**kwargs: Any) -> ModelSpec:
    return _build_mixllama_fast_runtime_spec(runtime="megakernel", **kwargs)


def build_jamba_hybrid_spec(**kwargs: Any) -> ModelSpec:
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="jamba", sparsity="moe", runtime="compile"),
        block_spec=BlockSpec(
            family="jamba",
            norm_type="rmsnorm",
            mlp_type="moe",
            pos_encoding="rope",
            linear_bias=False,
            experts=kwargs.get("experts", 8),
            top_k=kwargs.get("top_k", 2),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
        ),
        default_tie_embeddings=False,
    )


def build_ternary_b158_spec(**kwargs: Any) -> ModelSpec:
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="llama", compression="ternary_b158", runtime="compile"),
        block_spec=BlockSpec(
            family="llama",
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_encoding="rope",
            linear_bias=False,
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
            compression="ternary_b158",
        ),
        default_tie_embeddings=False,
    )


def build_decoder2encoder_moe_spec(**kwargs: Any) -> ModelSpec:
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(objective="seq2seq", backbone="llama", sparsity="moe", runtime="compile"),
        block_spec=BlockSpec(
            family="llama",
            norm_type="rmsnorm",
            mlp_type="moe",
            pos_encoding="rope",
            linear_bias=False,
            experts=kwargs.get("experts", 8),
            top_k=kwargs.get("top_k", 2),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
        ),
        default_tie_embeddings=False,
    )


def build_diffllama_spec(**kwargs: Any) -> ModelSpec:
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(objective="diffusion", backbone="llama", runtime="compile"),
        block_spec=BlockSpec(
            family="llama",
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_encoding="rope",
            linear_bias=False,
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
        ),
        default_tie_embeddings=False,
    )


def build_ttt_llama_spec(**kwargs: Any) -> ModelSpec:
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="ttt", runtime="compile"),
        block_spec=BlockSpec(
            family="ttt",
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_encoding="rope",
            linear_bias=False,
            dropout_p=0.0,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            multiple_of=kwargs.get("multiple_of", 256),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
            attention_backend="sdpa",
            ttt_hidden_dim=int(kwargs.get("ttt_hidden_dim", 32)),
        ),
        default_tie_embeddings=False,
    )


def build_llm_jepa_spec(**kwargs: Any) -> ModelSpec:
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(objective="jepa", backbone="llama", runtime="compile"),
        block_spec=BlockSpec(
            family="llama",
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_encoding="rope",
            linear_bias=False,
            dropout_p=0.0,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            multiple_of=kwargs.get("multiple_of", 256),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
            attention_backend="sdpa",
        ),
        default_tie_embeddings=False,
    )


def build_hnet_lm_spec(**kwargs: Any) -> ModelSpec:
    effective_kwargs = dict(kwargs)
    effective_kwargs["vocab_size"] = 256
    return _base_model_spec(
        kwargs=effective_kwargs,
        template=TemplateSpec(backbone="hnet", tokenization="byte_hnet", runtime="compile"),
        block_spec=BlockSpec(
            family="hnet",
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_encoding="rope",
            linear_bias=False,
            dropout_p=0.0,
            mlp_multiplier=effective_kwargs.get("mlp_multiplier", 8.0 / 3.0),
            multiple_of=effective_kwargs.get("multiple_of", 64),
            num_heads=effective_kwargs.get("num_heads", 4),
            num_kv_heads=effective_kwargs.get("num_kv_heads", 2),
            attention_backend="sdpa",
            byte_patch_size=int(effective_kwargs.get("byte_patch_size", 4)),
            byte_patch_stride=int(effective_kwargs.get("byte_patch_stride", effective_kwargs.get("byte_patch_size", 4))),
        ),
        default_tie_embeddings=False,
    )


def build_universal_llama_spec(**kwargs: Any) -> ModelSpec:
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="universal", runtime="compile"),
        block_spec=BlockSpec(
            family="universal",
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_encoding="rope",
            linear_bias=False,
            dropout_p=0.0,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            multiple_of=kwargs.get("multiple_of", 256),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
            attention_backend="sdpa",
        ),
        default_tie_embeddings=False,
    )


def build_llama_megakernel_spec(**kwargs: Any) -> ModelSpec:
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="llama", runtime="megakernel"),
        block_spec=BlockSpec(
            family="llama",
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_encoding="rope",
            linear_bias=False,
            dropout_p=0.0,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            multiple_of=kwargs.get("multiple_of", 256),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
            attention_backend="sdpa",
        ),
        default_tie_embeddings=False,
    )


def build_kv_pca_llama_spec(**kwargs: Any) -> ModelSpec:
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="llama", compression="kv_pca", runtime="compile"),
        block_spec=BlockSpec(
            family="llama",
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_encoding="rope",
            linear_bias=False,
            dropout_p=0.0,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            multiple_of=kwargs.get("multiple_of", 256),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
            attention_backend="sdpa",
            compression="kv_pca",
        ),
        default_tie_embeddings=False,
    )


def _build_jepa_semantic_hybrid_runtime_spec(*, runtime: RuntimeType, **kwargs: Any) -> ModelSpec:
    """Experimental: Hybrid JEPA Semantic LLM preset."""
    experts = int(kwargs.get("experts", NUM_VOCAB_DIMS))
    if experts != NUM_VOCAB_DIMS:
        raise ValueError(
            f"jepa_semantic_hybrid requires exactly {NUM_VOCAB_DIMS} experts (one per vocab dimension)"
        )
    top_k = min(int(kwargs.get("top_k", 2)), NUM_VOCAB_DIMS)
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(
            objective="jepa_semantic",
            backbone="llama",
            sparsity="moe",
            runtime=runtime,
        ),
        block_spec=BlockSpec(
            family="llama",
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_encoding="rope",
            linear_bias=False,
            dropout_p=0.0,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            multiple_of=kwargs.get("multiple_of", 256),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
            attention_backend="sdpa",
            experts=experts,
            top_k=top_k,
            router_aux_loss_coef=kwargs.get("router_aux_loss_coef", 0.01),
            rope_theta=kwargs.get("rope_base", kwargs.get("rope_theta", 10000.0)),
            qk_gain_init=kwargs.get("qk_gain_init", 1.0),
        ),
        default_tie_embeddings=False,
    )


def build_jepa_semantic_hybrid_spec(**kwargs: Any) -> ModelSpec:
    return _build_jepa_semantic_hybrid_runtime_spec(runtime="compile", **kwargs)


def build_jepa_semantic_hybrid_megakernel_spec(**kwargs: Any) -> ModelSpec:
    return _build_jepa_semantic_hybrid_runtime_spec(runtime="megakernel", **kwargs)


def _build_semantic_router_moe_runtime_spec(*, runtime: RuntimeType, **kwargs: Any) -> ModelSpec:
    """Experimental: AR MixLLaMA with shared semantic hash routing into MoE experts."""
    experts = int(kwargs.get("experts", NUM_VOCAB_DIMS))
    if experts != NUM_VOCAB_DIMS:
        raise ValueError(
            f"semantic_router_moe requires exactly {NUM_VOCAB_DIMS} experts (one per vocab dimension)"
        )
    top_k = min(int(kwargs.get("top_k", 2)), NUM_VOCAB_DIMS)
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(
            objective="semantic_router",
            backbone="mixllama",
            tokenization="sp",
            sparsity="moe",
            compression="none",
            adapter="none",
            runtime=runtime,
        ),
        block_spec=BlockSpec(
            family="mixllama",
            norm_type="rmsnorm",
            mlp_type="moe",
            pos_encoding="rope",
            linear_bias=False,
            dropout_p=0.0,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            experts=experts,
            top_k=top_k,
            router_aux_loss_coef=0.0,
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
            attention_backend="sdpa",
            rope_theta=kwargs.get("rope_base", kwargs.get("rope_theta", 10000.0)),
            qk_gain_init=kwargs.get("qk_gain_init", 1.0),
        ),
        default_tie_embeddings=False,
    )


def build_semantic_router_moe_spec(**kwargs: Any) -> ModelSpec:
    return _build_semantic_router_moe_runtime_spec(runtime="compile", **kwargs)


def build_semantic_router_moe_megakernel_spec(**kwargs: Any) -> ModelSpec:
    return _build_semantic_router_moe_runtime_spec(runtime="megakernel", **kwargs)
