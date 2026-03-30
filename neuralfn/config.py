from dataclasses import dataclass, field
from typing import Any


@dataclass
class BlockSpec:
    family: str  # "nanogpt" | "gpt2" | "llama" | "moe"
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


@dataclass
class ModelSpec:
    model_dim: int = 128
    num_layers: int = 4
    vocab_size: int = 256
    tie_embeddings: bool = True
    logit_softcap: float = 0.0  # 0.0 = disabled; >0.0 = tanh softcap (Gemma, PaLM)
    block_spec: BlockSpec = field(default_factory=lambda: BlockSpec(family="gpt2"))


def build_nanogpt_spec(**kwargs: Any) -> ModelSpec:
    return ModelSpec(
        tie_embeddings=kwargs.get("tie_embeddings", True),
        block_spec=BlockSpec(
            family="nanogpt",
            norm_type="layernorm",
            mlp_type="gelu",
            pos_encoding="absolute",
            linear_bias=kwargs.get("bias", False),
            dropout_p=kwargs.get("dropout_p", 0.1),
            num_heads=kwargs.get("num_heads", 4),
        )
    )


def build_gpt2_spec(**kwargs: Any) -> ModelSpec:
    return ModelSpec(
        tie_embeddings=kwargs.get("tie_embeddings", True), 
        block_spec=BlockSpec(
            family="gpt2",
            norm_type="layernorm",
            mlp_type="gelu",
            pos_encoding="absolute",
            linear_bias=True,
            num_heads=kwargs.get("num_heads", 4),
        )
    )


def build_llama_spec(**kwargs: Any) -> ModelSpec:
    return ModelSpec(
        tie_embeddings=kwargs.get("tie_embeddings", False),
        block_spec=BlockSpec(
            family="llama",
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_encoding="rope",
            linear_bias=False,
            dropout_p=0.0,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0/3.0),
            multiple_of=kwargs.get("multiple_of", 256),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
        )
    )


def build_moe_spec(**kwargs: Any) -> ModelSpec:
    return ModelSpec(
        tie_embeddings=kwargs.get("tie_embeddings", False),
        block_spec=BlockSpec(
            family="moe",
            norm_type="rmsnorm",
            mlp_type="moe",
            pos_encoding="rope",
            linear_bias=False,
            dropout_p=0.0,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0/3.0),
            experts=kwargs.get("experts", 8),
            top_k=kwargs.get("top_k", 2),
            router_aux_loss_coef=kwargs.get("router_aux_loss_coef", 0.01),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
        )
    )
