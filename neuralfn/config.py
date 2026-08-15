from dataclasses import asdict, dataclass, field
import math
from typing import Any, Literal

from .semantic import DEFAULT_SEMANTIC_VOCAB_REF, NUM_SEMANTIC_DIMS, NUM_VOCAB_DIMS

ObjectiveType = Literal[
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
    "reward_model",
]
BackboneType = Literal[
    "gpt2",
    "nanogpt",
    "llama",
    "mixllama",
    "jamba",
    "universal",
    "ttt",
    "hnet",
    "muse_glimmer",
]
TokenizationType = Literal["sp", "byte_hnet"]
SparsityType = Literal["dense", "moe"]
CompressionType = Literal[
    "none",
    "ternary_b158",
    "binary_1bit",
    "kv_pca",
    "fp8_e4m3",
    "fp8_e5m2",
    "mxfp4",
    "mxfp8",
]
AdapterType = Literal["none", "lora", "qlora", "randmap"]
RuntimeType = Literal["eager", "compile", "sdpa", "megakernel"]
RouterModeType = Literal["none", "standard", "semantic"]
FineTuneObjective = Literal["pretrain", "sft", "dpo", "ppo", "reward_model"]
DPOLossType = Literal["sigmoid", "hinge", "ipo"]


@dataclass
class TemplateSpec:
    objective: ObjectiveType = "ar"
    backbone: BackboneType = "gpt2"
    tokenization: TokenizationType = "sp"
    sparsity: SparsityType = "dense"
    router_mode: RouterModeType = "none"
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
class LayerAttentionSpec:
    """One entry in a decoder's repeating per-layer attention schedule.

    Defaults describe ordinary full causal attention with no positional
    encoding. Architectures that do not set ``BlockSpec.layer_attention_pattern``
    retain their existing homogeneous block behavior.
    """

    kind: str = "full"  # "full" | "local"
    window_size: int | None = None
    pos_encoding: str = "none"  # "none" | "rope"
    rope_theta: float = 10_000.0


@dataclass
class BlockSpec:
    family: str  # "nanogpt" | "gpt2" | "llama" | "mixllama"
    norm_type: str = "layernorm"  # "layernorm" | "rmsnorm" | "dyt" | "group_norm"
    mlp_type: str = "gelu"  # "gelu" | "swiglu" | "moe" | "geglu" | "reglu" | "solu"
    pos_encoding: str = "absolute"  # "absolute" | "rope"
    attention_backend: str = "sdpa"  # "sdpa" | "flex" | "math"
    num_heads: int = 4
    num_kv_heads: int | None = None  # None => MHA, smaller => GQA/MQA
    # Explicit attention geometry. ``None`` preserves the historical
    # ``head_dim = model_dim // num_heads`` and Q-width == model-width rules.
    head_dim: int | None = None
    attention_inner_dim: int | None = None
    intermediate_size: int | None = None
    layer_attention_pattern: tuple[LayerAttentionSpec, ...] = ()
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
    adapter_type: str = "none"  # "none" | "lora" | "qlora" | "randmap"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj")
    lora_bias: bool = False
    qlora_group_size: int = 64
    qlora_compute_dtype: str = "bf16"
    ttt_hidden_dim: int = 16
    byte_patch_size: int = 4
    byte_patch_stride: int = 4
    qk_gain_init: float = 1.0
    # --- Frontier-template knobs (additive; defaults preserve every existing preset) ---
    # Attention core variant swapped in at the SDPA node in build_dense_attention_graph.
    attention_variant: str = "dense"  # "dense" | "differential" | "sliding_window" | "block_sparse" | "nsa" | "streaming"
    use_qk_norm: bool = False  # fused RMSNorm on Q and K before SDPA (DeepSeek-V3, Gemma-3)
    qk_norm_kind: str = "none"  # "none" | "weightless_rms"
    qk_norm_eps: float = 1e-6
    q_scale_factor: float = 1.0
    attention_gate: str = "none"  # "none" | "sigmoid"
    attention_gate_dim: int | None = None
    norm_layout: str = "pre"  # "pre" | "sandwich"
    centered_rms_norm: bool = False
    norm_eps: float = 1e-6
    post_norm_eps: float = 1e-6
    embedding_norm_kind: str = "none"  # "none" | "weightless_rms"
    embedding_norm_eps: float = 1e-6
    dyt_alpha_init: float = 1.0  # Dynamic Tanh initial alpha (norm_type == "dyt")
    group_norm_groups: int = 1  # groups for norm_type == "group_norm"
    diff_lambda_init: float = 0.8  # Differential Transformer initial lambda
    # MoE load balancing: "aux_loss" (classic) | "auxfree" (DeepSeek-V3 bias-adjusted, no aux loss)
    moe_balance_mode: str = "aux_loss"
    auxfree_bias_lr: float = 0.001  # update rate for the auxfree expert-load bias
    router_score_fn: str = "softmax"  # "softmax" | "sigmoid" | "sqrt_softplus" (DeepSeek-V4)
    # Residual mixing: "add" (classic) | "mhc" (Manifold-Constrained Hyper-Connections, DeepSeek-V4)
    residual_type: str = "add"
    # Sparse / long-context attention geometry (used by the windowed/sparse/streaming variants).
    window_size: int | None = None
    sparse_block_size: int = 64
    num_sinks: int = 0
    nsa_compress_stride: int = 16  # CSA "compress every m KV into one" stride
    # Low-precision (FP8 / MX) linear knobs (compression in {fp8_*, mxfp4, mxfp8}).
    mx_block_size: int = 32
    fp8_amax_history_len: int = 16
    fp8_use_stochastic_rounding: bool = True
    # --- Mixture of Activations (MoA) knobs (additive; defaults preserve single-activation MLPs) ---
    # When activation_mode == "moa", training probes each candidate activation's loss every
    # moa_interval steps over a SHARED MLP backbone and trains with the lowest-loss one.
    # Candidates are restricted to the weight-preserving pointwise activations
    # (gelu/relu/silu/relu2) — they share the same fc/proj weights, so MoA needs no extra
    # parameters and keeps pointwise training speed. Gated swiglu/geglu are intentionally
    # excluded (they'd need a separate gate projection). Mirrors the llm.kittens
    # train_gpt2cu `-af moa -ak <interval>` mode. See build_gpt2_moa_spec.
    activation_mode: str = "single"  # "single" | "moa"
    moa_activations: tuple[str, ...] = ("gelu", "relu", "silu", "relu2")
    moa_interval: int = 50


@dataclass
class FineTuneSpec:
    """Configuration for a fine-tuning run.

    ``objective`` selects the training objective. ``base_checkpoint`` points
    at the pretrained weights to initialize from (frozen for LoRA/qLoRA).
    ``ref_checkpoint`` is used by DPO/PPO for the frozen reference model.
    ``reward_checkpoint`` is used by PPO for the frozen reward model.
    """
    objective: str = "pretrain"  # "pretrain" | "sft" | "dpo" | "ppo" | "reward_model"
    base_checkpoint: str = ""
    base_checkpoint_sha256: str = ""
    # Checkpoint storage profile for native post-training. K-Quant profiles
    # keep the official GGUF bytes immutable and train LoRA parameters only;
    # they are not aliases for NF4 QLoRA.
    base_weight_precision: str = "bf16"  # "bf16" | "k-quant-17gb" | "k-quant-dynamic"
    tokenizer_sha256: str = ""
    chat_template_sha256: str = ""
    ref_graph_path: str = ""
    ref_checkpoint: str = ""
    ref_checkpoint_sha256: str = ""
    reward_graph_path: str = ""
    reward_checkpoint: str = ""
    reward_checkpoint_sha256: str = ""
    resume_checkpoint: str = ""
    adapter_only_save: bool = False
    # DPO knobs
    beta: float = 0.1
    dpo_loss_type: str = "sigmoid"
    dpo_label_smoothing: float = 0.0
    # PPO knobs
    kl_coef: float = 0.1
    ppo_clip: float = 0.2
    ppo_vf_coef: float = 0.5
    ppo_ent_coef: float = 0.0
    rollout_length: int = 64
    ppo_epochs_per_rollout: int = 4
    ppo_minibatch_size: int = 4
    gae_gamma: float = 1.0
    gae_lambda: float = 0.95

    def __post_init__(self) -> None:
        allowed_objectives = {"pretrain", "sft", "dpo", "ppo", "reward_model"}
        self.objective = str(self.objective).strip().lower()
        if self.objective not in allowed_objectives:
            raise ValueError(
                f"Unsupported fine-tuning objective {self.objective!r}; "
                f"expected one of {sorted(allowed_objectives)}"
            )
        self.base_weight_precision = str(self.base_weight_precision).strip().lower()
        if self.base_weight_precision not in {
            "bf16",
            "k-quant-17gb",
            "k-quant-dynamic",
        }:
            raise ValueError(
                "FineTuneSpec.base_weight_precision must be 'bf16', "
                "'k-quant-17gb', or 'k-quant-dynamic'"
            )
        for field_name in (
            "base_checkpoint_sha256",
            "ref_checkpoint_sha256",
            "reward_checkpoint_sha256",
            "tokenizer_sha256",
            "chat_template_sha256",
        ):
            digest = str(getattr(self, field_name)).strip().lower()
            if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)):
                raise ValueError(f"{field_name} must be an empty string or a 64-character SHA-256 digest")
            setattr(self, field_name, digest)
        if self.beta <= 0.0:
            raise ValueError("FineTuneSpec.beta must be positive")
        if self.dpo_loss_type not in {"sigmoid", "hinge", "ipo"}:
            raise ValueError("FineTuneSpec.dpo_loss_type must be 'sigmoid', 'hinge', or 'ipo'")
        if not 0.0 <= self.dpo_label_smoothing < 0.5:
            raise ValueError("FineTuneSpec.dpo_label_smoothing must be in [0, 0.5)")
        if self.kl_coef < 0.0:
            raise ValueError("FineTuneSpec.kl_coef must be non-negative")
        if not 0.0 < self.ppo_clip < 1.0:
            raise ValueError("FineTuneSpec.ppo_clip must be in (0, 1)")
        if self.ppo_vf_coef < 0.0 or self.ppo_ent_coef < 0.0:
            raise ValueError("PPO value and entropy coefficients must be non-negative")
        if self.rollout_length <= 0 or self.ppo_epochs_per_rollout <= 0 or self.ppo_minibatch_size <= 0:
            raise ValueError("PPO rollout length, epochs, and minibatch size must be positive")
        if not 0.0 <= self.gae_gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("GAE gamma and lambda must be in [0, 1]")


@dataclass
class MuseGlimmerVisionSpec:
    """Exact vision-tower/projector contract for Muse Glimmer.

    The fields are intentionally independent from :class:`BlockSpec`: vision
    attention is non-causal, biased, variable-length, and uses a different
    hidden geometry from the text decoder.
    """

    num_hidden_layers: int = 50
    hidden_size: int = 1_536
    intermediate_size: int = 8_960
    num_attention_heads: int = 16
    patch_size: int = 14
    patch_temporal: int = 2
    merge_size: int = 2
    pos_emb_height: int = 32
    pos_emb_width: int = 32
    rope_theta: float = 10_000.0
    layer_norm_eps: float = 1.0e-5
    projector_hidden_size: int = 4_096
    image_token_id: int = 200_092
    video_token_id: int = 200_091

    def __post_init__(self) -> None:
        positive = (
            self.num_hidden_layers,
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.patch_size,
            self.patch_temporal,
            self.merge_size,
            self.pos_emb_height,
            self.pos_emb_width,
            self.projector_hidden_size,
        )
        if any(int(value) <= 0 for value in positive):
            raise ValueError("Muse Glimmer vision dimensions must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("Muse Glimmer vision heads must divide hidden_size")
        if self.pos_emb_height != self.pos_emb_width:
            raise ValueError("Muse Glimmer v1 requires a square learned position table")
        if self.rope_theta <= 0.0 or self.layer_norm_eps <= 0.0:
            raise ValueError("Muse Glimmer vision RoPE/norm values must be positive")
        if self.image_token_id < 0 or self.video_token_id < 0:
            raise ValueError("Muse Glimmer media token IDs must be non-negative")

    def layer_type(self, index: int) -> str:
        if index < 0 or index >= self.num_hidden_layers:
            raise IndexError(index)
        return (
            "full_attention"
            if (index + 1) % 4 == 0 or index == self.num_hidden_layers - 1
            else "window_attention"
        )


@dataclass
class MuseGlimmerDFlashSpec:
    """Target-bound shape contract for the stock Muse Glimmer assistant."""

    num_hidden_layers: int = 5
    hidden_size: int = 6_656
    intermediate_size: int = 19_968
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    block_size: int = 16
    mask_token_id: int = 201_818
    sliding_window: int = 2_048
    rope_theta: float = 500_000.0
    target_layer_ids: tuple[int, ...] = (1, 13, 25, 37, 49)

    def __post_init__(self) -> None:
        if (
            self.num_hidden_layers <= 0
            or self.hidden_size <= 0
            or self.intermediate_size <= 0
            or self.num_attention_heads <= 0
            or self.num_key_value_heads <= 0
            or self.num_attention_heads % self.num_key_value_heads != 0
            or self.head_dim <= 0
            or self.head_dim % 2 != 0
            or self.block_size < 2
            or self.mask_token_id < 0
            or self.sliding_window <= 0
            or self.rope_theta <= 0.0
        ):
            raise ValueError("invalid Muse Glimmer DFlash contract")
        normalized = tuple(int(value) for value in self.target_layer_ids)
        if normalized != tuple(sorted(set(normalized))) or not normalized:
            raise ValueError("DFlash target_layer_ids must be sorted and unique")
        self.target_layer_ids = normalized


@dataclass
class MuseGlimmerDFlashDistillationSpec:
    """Versioned training contract for a target-bound DFlash assistant.

    The stock Meta assistant publishes its architecture and inference contract,
    but not the recipe used to train the released weights.  NeuralFn therefore
    records an explicit, reproducible recipe instead of implying bit-for-bit
    provenance.  Version 1 follows the random-anchor, self-logit-distillation,
    and D-PACE formulation pinned in the implementation documentation.
    """

    recipe: str = "neuralfn.muse_glimmer_dflash_distill.v1"
    recipe_revision: str = "686da8d893536688e86adae41aa628aa740a1a35"
    target_checkpoint_sha256: str = ""
    target_config_sha256: str = ""
    tokenizer_sha256: str = ""
    chat_template_sha256: str = ""
    num_anchors: int = 512
    loss_objective: str = "dpace"  # "dpace" | "decay"
    dpace_alpha: float = 0.5
    loss_decay_factor: float = 7.0
    self_logit_distillation: bool = True
    seed: int = 20_260_813

    def __post_init__(self) -> None:
        if self.recipe != "neuralfn.muse_glimmer_dflash_distill.v1":
            raise ValueError("Unsupported Muse Glimmer DFlash distillation recipe")
        if (
            len(self.recipe_revision) != 40
            or any(character not in "0123456789abcdef" for character in self.recipe_revision)
        ):
            raise ValueError("DFlash recipe_revision must be a lowercase Git SHA")
        for field_name in (
            "target_checkpoint_sha256",
            "target_config_sha256",
            "tokenizer_sha256",
            "chat_template_sha256",
        ):
            digest = str(getattr(self, field_name)).strip().lower()
            if digest and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    f"{field_name} must be empty or a lowercase 64-character SHA-256"
                )
            setattr(self, field_name, digest)
        self.loss_objective = str(self.loss_objective).strip().lower()
        if self.loss_objective not in {"dpace", "decay"}:
            raise ValueError("DFlash loss_objective must be 'dpace' or 'decay'")
        if self.num_anchors <= 0:
            raise ValueError("DFlash num_anchors must be positive")
        if not 0.0 < self.dpace_alpha <= 1.0:
            raise ValueError("DFlash dpace_alpha must be in (0, 1]")
        if not math.isfinite(self.loss_decay_factor) or self.loss_decay_factor < 0.0:
            raise ValueError("DFlash loss_decay_factor must be finite and non-negative")
        if self.seed < 0:
            raise ValueError("DFlash seed must be non-negative")


@dataclass
class ModelSpec:
    model_dim: int = 128
    num_layers: int = 4
    vocab_size: int = 256
    tie_embeddings: bool = True
    max_position_embeddings: int = 1024
    output_multiplier: float = 1.0
    logit_softcap: float = 0.0  # 0.0 = disabled; >0.0 = tanh softcap (Gemma, PaLM)
    # H0302: coefficient on mean(logsumexp(logits)^2). 0.0 = plain cross-entropy;
    # 1e-4 is the value used by the gpt2_zloss / gpt2_stable presets.
    z_loss_coef: float = 0.0
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
    route_chunk_size: int = 32
    semantic_shared_experts: int = 2
    semantic_free_experts: int = 8
    route_evo_enabled: bool = True
    route_evo_fraction: float = 0.10
    route_evo_population: int = 8
    route_evo_mutation_scale: float = 0.05
    route_evo_seed: int | None = None
    layer_evo_enabled: bool = False
    layer_evo_index: int | None = None
    layer_evo_fraction: float = 0.10
    layer_evo_population: int = 8
    layer_evo_mutation_scale: float = 0.02
    layer_evo_seed: int | None = None
    ar_loss_coef: float = 1.0
    jepa_loss_coef: float = 0.25
    semantic_align_loss_coef: float = 0.5
    finetune: FineTuneSpec | None = None
    vision: MuseGlimmerVisionSpec | None = None
    dflash: MuseGlimmerDFlashSpec | None = None


def model_spec_to_dict(spec: ModelSpec) -> dict[str, Any]:
    payload = asdict(spec)
    # Keep old preset payloads byte/field compatible. These optional contracts
    # are serialized only by architectures that actually declare them.
    if spec.vision is None:
        payload.pop("vision", None)
    if spec.dflash is None:
        payload.pop("dflash", None)
    return payload


def _base_model_spec(
    *,
    kwargs: dict[str, Any],
    template: TemplateSpec,
    block_spec: BlockSpec,
    default_tie_embeddings: bool,
) -> ModelSpec:
    template.backend_capabilities = resolve_backend_capabilities(template)
    model_dim = int(kwargs.get("model_dim", kwargs.get("n_embd", 128)))
    finetune_raw = kwargs.get("finetune")
    if finetune_raw is None:
        finetune = None
    elif isinstance(finetune_raw, FineTuneSpec):
        finetune = finetune_raw
    elif isinstance(finetune_raw, dict):
        finetune = FineTuneSpec(**finetune_raw)
    else:
        raise TypeError("finetune must be FineTuneSpec, a mapping, or None")
    return ModelSpec(
        model_dim=model_dim,
        num_layers=int(kwargs.get("num_layers", kwargs.get("n_layer", 4))),
        vocab_size=int(kwargs.get("vocab_size", 256)),
        tie_embeddings=bool(kwargs.get("tie_embeddings", default_tie_embeddings)),
        max_position_embeddings=int(kwargs.get("max_position_embeddings", 1024)),
        output_multiplier=float(kwargs.get("output_multiplier", 1.0)),
        logit_softcap=float(kwargs.get("logit_softcap", 0.0)),
        z_loss_coef=float(kwargs.get("z_loss_coef", 0.0)),
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
        route_chunk_size=int(kwargs.get("route_chunk_size", 32)),
        semantic_shared_experts=int(kwargs.get("semantic_shared_experts", 2)),
        semantic_free_experts=int(kwargs.get("semantic_free_experts", 8)),
        route_evo_enabled=bool(kwargs.get("route_evo_enabled", True)),
        route_evo_fraction=float(kwargs.get("route_evo_fraction", 0.10)),
        route_evo_population=int(kwargs.get("route_evo_population", 8)),
        route_evo_mutation_scale=float(kwargs.get("route_evo_mutation_scale", 0.05)),
        route_evo_seed=kwargs.get("route_evo_seed"),
        layer_evo_enabled=bool(kwargs.get("layer_evo_enabled", False)),
        layer_evo_index=(
            None if kwargs.get("layer_evo_index") is None else int(kwargs["layer_evo_index"])
        ),
        layer_evo_fraction=float(kwargs.get("layer_evo_fraction", 0.10)),
        layer_evo_population=int(kwargs.get("layer_evo_population", 8)),
        layer_evo_mutation_scale=float(kwargs.get("layer_evo_mutation_scale", 0.02)),
        layer_evo_seed=kwargs.get("layer_evo_seed"),
        ar_loss_coef=float(kwargs.get("ar_loss_coef", 1.0)),
        jepa_loss_coef=float(kwargs.get("jepa_loss_coef", 0.25)),
        semantic_align_loss_coef=float(kwargs.get("semantic_align_loss_coef", 0.5)),
        finetune=finetune,
    )


MODERN_BASE_PRESETS: tuple[str, ...] = (
    "nanogpt",
    "gpt2",
    "llama",
    "moe",
    "jamba",
    "ternary_b158",
    "seq2seq",
    "diffusion",
    "ttt_llama",
    "llm_jepa",
    "dense_jepa_evo",
    "moe_jepa_evo",
    "hnet_lm",
    "universal_llama",
    "kv_pca_llama",
    "jepa_semantic_hybrid",
    "semantic_router_moe",
    "semantic_dense_jepa_evo",
    "semantic_moe_jepa_evo",
)

SHIPPED_GPT_TEMPLATE_BASE_PRESETS: tuple[str, ...] = (
    "nanogpt",
    "nanogpt_megakernel",
    "gpt2",
    "gpt2_megakernel",
    "gpt2_moa",
    "gpt2_zloss",
    "gpt2_softcap",
    "gpt2_qknorm",
    "gpt2_diff",
    "gpt2_stable",
    "llama",
    "muse_glimmer",
    "modern_norms_llama",
    "mixllama",
    "moe",
    "llama_fast",
    "llama_fast_megakernel",
    "mixllama_fast",
    "mixllama_fast_megakernel",
    "jamba",
    "ternary_b158",
    "fp8_llama",
    "mxfp4_llama",
    "deepseek_v3",
    "deepseek_v4",
    "gemma3",
    "diff_transformer",
    "longctx_sparse_llama",
    "qwen3_longctx",
    "auxfree_moe_jepa_evo",
    "diff_semantic_moe_jepa_evo",
    "dyt_geglu_semantic_dense_jepa_evo",
    "llama_megakernel",
    "kv_pca_llama",
    "seq2seq",
    "diffusion",
    "ttt_llama",
    "llm_jepa",
    "dense_jepa_evo",
    "moe_jepa_evo",
    "jepa_semantic_hybrid",
    "jepa_semantic_hybrid_megakernel",
    "semantic_router_moe",
    "semantic_router_moe_megakernel",
    "semantic_moe_jepa_evo",
    "semantic_dense_jepa_evo",
    "hnet_lm",
    "universal_llama",
)

SHIPPED_GPT_TEMPLATE_PRESETS: tuple[str, ...] = (
    *SHIPPED_GPT_TEMPLATE_BASE_PRESETS,
    *(f"{preset}_modern" for preset in MODERN_BASE_PRESETS),
)


def _apply_modern_profile(spec: ModelSpec) -> ModelSpec:
    """Overlay a uniform 2026 "modern" recipe onto any base preset's ModelSpec.

    Additive, in-place tweaks only -- preserves objective/backbone/expert topology:
      - LayerNorm -> RMSNorm (keeps dyt/group_norm/rmsnorm)
      - GELU MLP -> GeGLU (keeps swiglu/moe/other gates)
      - absolute positions -> RoPE + YaRN long-context scaling
      - fused QK-norm on
      - MoE blocks -> DeepSeek-V3 auxiliary-loss-free load balancing
    mHC residuals and FP8 are intentionally NOT enabled here (opt-in via the
    deepseek_v4 preset / explicit kwargs). Used by the generated ``<preset>_modern``
    presets (see MODERN_BASE_PRESETS)."""
    bs = spec.block_spec
    if bs.norm_type == "layernorm":
        bs.norm_type = "rmsnorm"
    if bs.mlp_type == "gelu":
        bs.mlp_type = "geglu"
    if bs.pos_encoding == "absolute":
        bs.pos_encoding = "rope"
    bs.use_qk_norm = True
    if bs.pos_encoding == "rope" and not bs.rope_scaling:
        bs.rope_scaling = {"type": "yarn", "factor": 2.0, "original_max_position": 2048}
    if bs.mlp_type in ("moe", "mixllama"):
        bs.moe_balance_mode = "auxfree"
        bs.router_aux_loss_coef = 0.0
    return spec


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


def _resolve_composed_runtime(base_model: str, runtime: str | None) -> RuntimeType:
    normalized = str(runtime or "default").strip().lower()
    if normalized == "default":
        return "compile" if base_model == "llama" else "eager"
    if normalized not in {"eager", "compile", "megakernel"}:
        raise ValueError(f"Unsupported composed runtime {runtime!r}")
    return normalized  # type: ignore[return-value]


def _resolve_composed_objective(*, router_mode: RouterModeType, use_jepa: bool) -> ObjectiveType:
    if router_mode == "semantic":
        return "semantic_router_jepa" if use_jepa else "semantic_router"
    return "ar_jepa" if use_jepa else "ar"


def build_composed_lm_spec(
    *,
    base_model: str = "llama",
    topology: str = "dense",
    router_mode: str = "none",
    use_jepa: bool = False,
    runtime: str | None = None,
    **kwargs: Any,
) -> ModelSpec:
    normalized_model = str(base_model or "llama").strip().lower()
    if normalized_model == "mixllama":
        normalized_model = "llama"
    if normalized_model in {"gpt", "gpt3"}:
        normalized_model = "gpt2"
    if normalized_model not in {"llama", "gpt2", "nanogpt"}:
        raise ValueError(f"Unsupported composed base model {base_model!r}")

    normalized_topology = str(topology or "dense").strip().lower()
    if normalized_topology not in {"dense", "moe"}:
        raise ValueError(f"Unsupported composed topology {topology!r}")

    normalized_router = str(router_mode or "none").strip().lower()
    if normalized_topology == "dense":
        normalized_router = "none"
    elif normalized_router == "none":
        normalized_router = "standard"
    if normalized_router not in {"none", "standard", "semantic"}:
        raise ValueError(f"Unsupported composed router mode {router_mode!r}")

    resolved_runtime = _resolve_composed_runtime(normalized_model, runtime)
    resolved_objective = _resolve_composed_objective(
        router_mode=normalized_router,  # type: ignore[arg-type]
        use_jepa=bool(use_jepa),
    )

    family = normalized_model
    norm_type = "layernorm"
    dense_mlp_type = "gelu"
    pos_encoding = "absolute"
    linear_bias = True
    dropout_p = float(kwargs.get("dropout_p", 0.0))
    mlp_multiplier = float(kwargs.get("mlp_multiplier", kwargs.get("mlp_mult", 4.0)))
    multiple_of = kwargs.get("multiple_of")
    num_kv_heads = None
    tie_embeddings = True
    tokenization: TokenizationType = "sp"

    if normalized_model == "llama":
        norm_type = "rmsnorm"
        dense_mlp_type = "swiglu"
        pos_encoding = "rope"
        linear_bias = False
        dropout_p = float(kwargs.get("dropout_p", 0.0))
        mlp_multiplier = float(kwargs.get("mlp_multiplier", kwargs.get("mlp_mult", 8.0 / 3.0)))
        multiple_of = int(kwargs.get("multiple_of", 256))
        requested_kv_heads = kwargs.get("num_kv_heads")
        if requested_kv_heads is not None:
            num_kv_heads = int(requested_kv_heads)
        tie_embeddings = bool(kwargs.get("tie_embeddings", False))
    elif normalized_model == "nanogpt":
        linear_bias = bool(kwargs.get("bias", False))
        dropout_p = float(kwargs.get("dropout_p", 0.1))
        tie_embeddings = bool(kwargs.get("tie_embeddings", True))
    else:
        tie_embeddings = bool(kwargs.get("tie_embeddings", True))

    experts: int | None = None
    top_k: int | None = None
    router_aux_loss_coef = float(kwargs.get("router_aux_loss_coef", 0.0))
    if normalized_topology == "moe":
        default_experts = NUM_VOCAB_DIMS if normalized_router == "semantic" else 8
        experts = int(kwargs.get("experts", default_experts))
        if normalized_router == "semantic" and experts != NUM_VOCAB_DIMS:
            raise ValueError(
                f"Semantic MoE router requires exactly {NUM_VOCAB_DIMS} experts (one per vocab dimension)"
            )
        top_k = int(kwargs.get("top_k", 2))
        if normalized_router == "semantic":
            top_k = min(top_k, NUM_VOCAB_DIMS)
            router_aux_loss_coef = float(kwargs.get("router_aux_loss_coef", 0.01 if use_jepa else 0.0))
        else:
            router_aux_loss_coef = float(kwargs.get("router_aux_loss_coef", 0.01))

    adapter_type_kw = str(kwargs.get("adapter_type", "none")).strip().lower()
    if adapter_type_kw not in {"none", "lora", "qlora", "randmap"}:
        raise ValueError(f"Unsupported adapter_type {adapter_type_kw!r}")
    template_adapter: AdapterType = adapter_type_kw  # type: ignore[assignment]

    lora_targets_kw = kwargs.get("lora_targets")
    if lora_targets_kw is None:
        lora_targets_tuple: tuple[str, ...] = ("q_proj", "v_proj")
    elif isinstance(lora_targets_kw, str):
        lora_targets_tuple = tuple(t.strip() for t in lora_targets_kw.split(",") if t.strip())
    else:
        lora_targets_tuple = tuple(str(t).strip() for t in lora_targets_kw if str(t).strip())

    template = TemplateSpec(
        objective=resolved_objective,
        backbone=normalized_model,  # type: ignore[arg-type]
        tokenization=tokenization,
        sparsity=normalized_topology,  # type: ignore[arg-type]
        router_mode=normalized_router,  # type: ignore[arg-type]
        compression="none",
        adapter=template_adapter,
        runtime=resolved_runtime,
    )
    block_spec = BlockSpec(
        family=family,
        norm_type=norm_type,
        mlp_type="moe" if normalized_topology == "moe" else dense_mlp_type,
        pos_encoding=pos_encoding,
        attention_backend="sdpa",
        num_heads=int(kwargs.get("num_heads", 4)),
        num_kv_heads=num_kv_heads,
        is_causal=True,
        linear_bias=linear_bias,
        dropout_p=dropout_p,
        rope_theta=float(kwargs.get("rope_base", kwargs.get("rope_theta", 10_000.0))),
        mlp_multiplier=mlp_multiplier,
        multiple_of=multiple_of,
        experts=experts,
        top_k=top_k,
        shared_experts=int(kwargs.get("shared_experts", 0)),
        router_aux_loss_coef=router_aux_loss_coef,
        compression="none",
        adapter_dim=int(kwargs.get("adapter_dim", 0)),
        adapter_type=adapter_type_kw,
        lora_rank=int(kwargs.get("lora_rank", 8)),
        lora_alpha=float(kwargs.get("lora_alpha", 16.0)),
        lora_dropout=float(kwargs.get("lora_dropout", 0.0)),
        lora_targets=lora_targets_tuple,
        lora_bias=bool(kwargs.get("lora_bias", False)),
        qlora_group_size=int(kwargs.get("qlora_group_size", 64)),
        qlora_compute_dtype=str(kwargs.get("qlora_compute_dtype", "bf16")),
        qk_gain_init=float(kwargs.get("qk_gain_init", 1.0)),
    )
    spec = _base_model_spec(
        kwargs=kwargs,
        template=template,
        block_spec=block_spec,
        default_tie_embeddings=tie_embeddings,
    )
    finetune_kw = kwargs.get("finetune")
    if isinstance(finetune_kw, FineTuneSpec):
        spec.finetune = finetune_kw
    elif isinstance(finetune_kw, dict):
        spec.finetune = FineTuneSpec(**finetune_kw)
    return spec


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
            use_qk_norm=bool(kwargs.get("use_qk_norm", False)),
            attention_variant=str(kwargs.get("attention_variant", "dense")),
            diff_lambda_init=float(kwargs.get("diff_lambda_init", 0.8)),
        ),
        default_tie_embeddings=True,
    )


def build_gpt2_spec(**kwargs: Any) -> ModelSpec:
    return _build_gpt2_runtime_spec(runtime="eager", **kwargs)


# --- Dense GPT-2 pretraining-stability variants (ai-autoresearch phase) --------
# All five stay dense (no router, no experts), keep GPT-2's LayerNorm / GELU /
# learned-absolute-position backbone, and change only what the corresponding
# hypothesis targets. See gpt2-ai-research-implementation-phase-todo.md.


def build_gpt2_zloss_spec(**kwargs: Any) -> ModelSpec:
    """Dense GPT-2 with a z-loss anchored log-partition (H0302).

    Adds ``z_loss_coef * mean(logsumexp(logits, -1) ** 2)`` to the token
    cross-entropy. The term pins the softmax normalizer near zero so the logit
    scale cannot drift upward over a long pretraining run; H0302 predicts a
    50-80% reduction in the fraction of training rows whose top-1 logit gap
    exceeds 16, at validation NLL within 0.01 nats of plain cross-entropy.
    ``logsumexp`` is already the dominant term of the cross-entropy, so the
    forward cost is a squared scalar per row.
    """
    kwargs.setdefault("z_loss_coef", 1e-4)
    return _build_gpt2_runtime_spec(runtime=str(kwargs.pop("runtime", "eager")), **kwargs)


def build_gpt2_softcap_spec(**kwargs: Any) -> ModelSpec:
    """Dense GPT-2 with a tanh logit softcap (H0058, transferred from Gemma-3).

    Bounds the logits with ``softcap * tanh(logits / softcap)`` instead of
    penalizing the log-partition. This is the hard-bound counterpart to
    ``gpt2_zloss``: it caps the tail outright rather than regularizing toward
    zero, at the cost of a saturating gradient on the capped entries.
    """
    kwargs.setdefault("logit_softcap", 30.0)
    return _build_gpt2_runtime_spec(runtime=str(kwargs.pop("runtime", "eager")), **kwargs)


def build_gpt2_qknorm_spec(**kwargs: Any) -> ModelSpec:
    """Dense GPT-2 with QK-norm on the attention query/key projections.

    A fused RMSNorm on Q and K before SDPA (DeepSeek-V3 / Gemma-3 practice).
    ``use_qk_norm`` already existed on ``BlockSpec`` but no dense GPT-2 preset
    exercised it; it bounds the pre-softmax logit scale, which is the other
    place GPT-2 pretraining goes unstable at higher learning rates.
    """
    kwargs.setdefault("use_qk_norm", True)
    return _build_gpt2_runtime_spec(runtime=str(kwargs.pop("runtime", "eager")), **kwargs)


def build_gpt2_diff_spec(**kwargs: Any) -> ModelSpec:
    """Dense GPT-2 with Differential Transformer attention (H0067).

    Replaces the single softmax attention map with the difference of two maps
    scaled by a learnable per-head lambda, which cancels common-mode attention
    noise. Dense throughout; only the SDPA node changes.
    """
    kwargs.setdefault("attention_variant", "differential")
    kwargs.setdefault("diff_lambda_init", 0.8)
    return _build_gpt2_runtime_spec(runtime=str(kwargs.pop("runtime", "eager")), **kwargs)


def build_gpt2_stable_spec(**kwargs: Any) -> ModelSpec:
    """Dense GPT-2 stacking every stability change that costs ~no throughput.

    z-loss (H0302) + QK-norm bound both ends of the attention/softmax logit
    scale. Paired on the trainer side with tanh-approximate GELU (H0534, so the
    Torch and native backends agree) and no weight decay on 1-D parameters
    (H0532). This is the preset carried into the 20-step native A/B.
    """
    kwargs.setdefault("z_loss_coef", 1e-4)
    kwargs.setdefault("use_qk_norm", True)
    return _build_gpt2_runtime_spec(runtime=str(kwargs.pop("runtime", "eager")), **kwargs)


def build_gpt2_megakernel_spec(**kwargs: Any) -> ModelSpec:
    return _build_gpt2_runtime_spec(runtime="megakernel", **kwargs)


def build_gpt2_evo_spec(**kwargs: Any) -> ModelSpec:
    """Dense GPT-2 where one transformer layer is trained by evolution.

    The designated layer (``layer_evo_index``, default ``num_layers // 2``) is
    excluded from gradient optimization; every ``round(1/layer_evo_fraction)``
    steps an interleaved evolutionary search perturbs it with gaussian mutants
    and adopts the best candidate (the current weights are always candidate 0,
    so the candidate loss never regresses). All other layers, the embeddings,
    and the head train normally with gradients.
    """
    kwargs.setdefault("num_layers", 10)
    kwargs.setdefault("layer_evo_enabled", True)
    kwargs.setdefault("layer_evo_fraction", 0.10)
    kwargs.setdefault("layer_evo_population", 8)
    kwargs.setdefault("layer_evo_mutation_scale", 0.02)
    return _build_gpt2_runtime_spec(runtime="eager", **kwargs)


def build_gpt2_moa_spec(**kwargs: Any) -> ModelSpec:
    """GPT-2 with a Mixture of Activations (MoA) MLP.

    Every ``moa_interval`` steps the trainer probes each candidate activation's
    loss on the current batch and trains with the lowest-loss one. Candidates are
    the weight-preserving pointwise activations (gelu/relu/silu/relu2): they all
    share one MLP backbone (up-proj C->4C, down-proj 4C->C), so MoA adds no extra
    parameters and runs at full pointwise training speed. Gated swiglu/geglu are
    excluded (they'd need a separate gate projection / different weights). The
    backbone graph is a standard GPT-2 MLP (``mlp_type`` below); the per-window
    activation selection is a training-loop behaviour carried by the MoA block-spec
    fields. Mirrors the llm.kittens train_gpt2cu ``-af moa -ak <interval>`` mode.
    """
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="gpt2", runtime=kwargs.get("runtime", "eager")),
        block_spec=BlockSpec(
            family="gpt2",
            norm_type="layernorm",
            mlp_type=str(kwargs.get("mlp_type", "gelu")),  # shared backbone graph
            pos_encoding="absolute",
            linear_bias=True,
            num_heads=kwargs.get("num_heads", 4),
            activation_mode="moa",
            moa_activations=tuple(kwargs.get(
                "moa_activations", ("gelu", "relu", "silu", "relu2"))),
            moa_interval=int(kwargs.get("moa_interval", 50)),
        ),
        default_tie_embeddings=True,
    )


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


def build_muse_glimmer_spec(**kwargs: Any) -> ModelSpec:
    """Build the exact Muse Glimmer text-decoder architecture.

    The defaults match ``meta-models/Muse-Glimmer-30B``. Small caller-supplied
    dimensions remain supported for graph previews and parity fixtures while
    preserving the defining non-square residual/attention geometry.
    """

    model_dim = int(kwargs.get("model_dim", kwargs.get("n_embd", 6_656)))
    adapter_type = str(kwargs.get("adapter_type", "none")).strip().lower()
    if adapter_type not in {"none", "lora", "qlora"}:
        raise ValueError("Muse Glimmer adapter_type must be 'none', 'lora', or 'qlora'")
    default_lora_targets = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "attn_gate_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    lora_targets_raw = kwargs.get("lora_targets", default_lora_targets)
    if isinstance(lora_targets_raw, str):
        lora_targets = tuple(part.strip() for part in lora_targets_raw.split(",") if part.strip())
    else:
        lora_targets = tuple(str(part).strip() for part in lora_targets_raw if str(part).strip())
    unknown_lora_targets = sorted(set(lora_targets) - set(default_lora_targets))
    if unknown_lora_targets:
        raise ValueError(f"Unsupported Muse Glimmer LoRA targets: {unknown_lora_targets}")
    if adapter_type in {"lora", "qlora"} and not lora_targets:
        raise ValueError("Muse Glimmer LoRA/QLoRA requires at least one projection target")
    if int(kwargs.get("lora_rank", 8)) <= 0:
        raise ValueError("Muse Glimmer lora_rank must be positive")
    if float(kwargs.get("lora_alpha", 16.0)) <= 0.0:
        raise ValueError("Muse Glimmer lora_alpha must be positive")
    if not 0.0 <= float(kwargs.get("lora_dropout", 0.0)) < 1.0:
        raise ValueError("Muse Glimmer lora_dropout must be in [0, 1)")
    if int(kwargs.get("qlora_group_size", 64)) <= 0:
        raise ValueError("Muse Glimmer qlora_group_size must be positive")
    num_heads = int(kwargs.get("num_heads", kwargs.get("n_head", 32)))
    if num_heads <= 0:
        raise ValueError("Muse Glimmer num_heads must be positive")

    production_geometry = model_dim == 6_656 and num_heads == 32
    head_dim = int(
        kwargs.get(
            "head_dim",
            128 if production_geometry else max(2, model_dim // (2 * num_heads)),
        )
    )
    attention_inner_dim = int(kwargs.get("attention_inner_dim", num_heads * head_dim))
    if attention_inner_dim != num_heads * head_dim:
        raise ValueError(
            "Muse Glimmer attention_inner_dim must equal num_heads * head_dim "
            f"({num_heads * head_dim}), got {attention_inner_dim}"
        )
    if head_dim % 2 != 0:
        raise ValueError("Muse Glimmer head_dim must be even for RoPE")

    num_kv_heads = int(kwargs.get("num_kv_heads", 2 if production_geometry else max(1, num_heads // 2)))
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise ValueError("Muse Glimmer num_kv_heads must be positive and divide num_heads")

    local_window = int(kwargs.get("window_size", 2_048))
    rope_theta = float(kwargs.get("rope_theta", kwargs.get("rope_base", 500_000.0)))
    intermediate_size = int(
        kwargs.get("intermediate_size", 19_968 if model_dim == 6_656 else 3 * model_dim)
    )
    pattern = (
        LayerAttentionSpec("local", local_window, "rope", rope_theta),
        LayerAttentionSpec("local", local_window, "rope", rope_theta),
        LayerAttentionSpec("local", local_window, "rope", rope_theta),
        LayerAttentionSpec("full", None, "none", rope_theta),
    )

    normalized = dict(kwargs)
    normalized.setdefault("model_dim", model_dim)
    normalized.setdefault("num_layers", 52)
    normalized.setdefault("vocab_size", 202_048)
    normalized.setdefault("tie_embeddings", False)
    normalized.setdefault("max_position_embeddings", 131_072)
    normalized.setdefault("output_multiplier", 0.19611613513818404)
    normalized.setdefault("logit_softcap", 20.0)

    spec = _base_model_spec(
        kwargs=normalized,
        template=TemplateSpec(backbone="muse_glimmer", runtime="eager"),
        block_spec=BlockSpec(
            family="muse_glimmer",
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_encoding="none",
            linear_bias=False,
            dropout_p=0.0,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            attention_inner_dim=attention_inner_dim,
            intermediate_size=intermediate_size,
            layer_attention_pattern=pattern,
            rope_theta=rope_theta,
            window_size=local_window,
            use_qk_norm=True,
            qk_norm_kind="weightless_rms",
            qk_norm_eps=float(kwargs.get("qk_norm_eps", kwargs.get("norm_eps", 1e-5))),
            q_scale_factor=float(kwargs.get("q_scale_factor", 3.87)),
            attention_gate="sigmoid",
            attention_gate_dim=attention_inner_dim,
            norm_layout="sandwich",
            centered_rms_norm=True,
            norm_eps=float(kwargs.get("norm_eps", 1e-5)),
            post_norm_eps=float(kwargs.get("post_norm_eps", 1e-8)),
            embedding_norm_kind="weightless_rms",
            embedding_norm_eps=float(kwargs.get("embedding_norm_eps", kwargs.get("norm_eps", 1e-5))),
            adapter_type=adapter_type,
            adapter_dim=int(kwargs.get("adapter_dim", 0)),
            lora_rank=int(kwargs.get("lora_rank", 8)),
            lora_alpha=float(kwargs.get("lora_alpha", 16.0)),
            lora_dropout=float(kwargs.get("lora_dropout", 0.0)),
            lora_targets=lora_targets,
            lora_bias=bool(kwargs.get("lora_bias", False)),
            qlora_group_size=int(kwargs.get("qlora_group_size", 64)),
            qlora_compute_dtype=str(kwargs.get("qlora_compute_dtype", "bf16")),
        ),
        default_tie_embeddings=False,
    )
    production_contract = (
        model_dim == 6_656
        and int(spec.num_layers) == 52
        and int(spec.vocab_size) == 202_048
        and num_heads == 32
        and num_kv_heads == 2
        and head_dim == 128
    )
    enable_vision = bool(kwargs.get("enable_vision", production_contract))
    enable_dflash = bool(kwargs.get("enable_dflash", production_contract))
    if enable_vision:
        spec.vision = MuseGlimmerVisionSpec(
            num_hidden_layers=int(kwargs.get("vision_num_hidden_layers", 50)),
            hidden_size=int(kwargs.get("vision_hidden_size", 1_536)),
            intermediate_size=int(kwargs.get("vision_intermediate_size", 8_960)),
            num_attention_heads=int(kwargs.get("vision_num_attention_heads", 16)),
            patch_size=int(kwargs.get("vision_patch_size", 14)),
            patch_temporal=int(kwargs.get("vision_patch_temporal", 2)),
            merge_size=int(kwargs.get("vision_merge_size", 2)),
            pos_emb_height=int(kwargs.get("vision_pos_emb_height", 32)),
            pos_emb_width=int(kwargs.get("vision_pos_emb_width", 32)),
            rope_theta=float(kwargs.get("vision_rope_theta", 10_000.0)),
            layer_norm_eps=float(kwargs.get("vision_layer_norm_eps", 1.0e-5)),
            projector_hidden_size=int(kwargs.get("projector_hidden_size", 4_096)),
            image_token_id=int(kwargs.get("image_token_id", 200_092)),
            video_token_id=int(kwargs.get("video_token_id", 200_091)),
        )
    if enable_dflash:
        spec.dflash = MuseGlimmerDFlashSpec(
            hidden_size=model_dim,
            intermediate_size=intermediate_size,
            mask_token_id=int(kwargs.get("dflash_mask_token_id", 201_818)),
            target_layer_ids=tuple(
                int(value)
                for value in kwargs.get("dflash_target_layer_ids", (1, 13, 25, 37, 49))
            ),
        )
        if spec.dflash.target_layer_ids[-1] >= spec.num_layers:
            raise ValueError("DFlash target_layer_ids exceed the target decoder layer count")

    # Backend capability bits describe the independently implemented default
    # production contract. Small preview geometries stay graph-only unless the
    # caller explicitly enables their companion specs.
    spec.template.backend_capabilities.update(
        {
            "cache": True,
            "quantized_export": False,
            "native_train": True,
            "native_inference": True,
            "whole_model_cuda": True,
            "k_quant": True,
            "speculative_decoding": True,
            "multimodal": spec.vision is not None,
        }
    )
    return spec


def build_modern_norms_llama_spec(**kwargs: Any) -> ModelSpec:
    """Llama backbone modernised with Dynamic Tanh (DyT) norm, fused QK-norm,
    and a GeGLU MLP gate. Proves the norm/gate/qk-norm seams; all kernels have a
    complete CUDA reference in llm.kittens (no stubs)."""
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="llama", runtime="eager"),
        block_spec=BlockSpec(
            family="llama",
            norm_type=str(kwargs.get("norm_type", "dyt")),
            mlp_type=str(kwargs.get("mlp_type", "geglu")),
            pos_encoding="rope",
            use_qk_norm=bool(kwargs.get("use_qk_norm", True)),
            dyt_alpha_init=float(kwargs.get("dyt_alpha_init", 1.0)),
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
            multiple_of=kwargs.get("multiple_of"),
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
            multiple_of=kwargs.get("multiple_of"),
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
            mlp_multiplier=kwargs.get("mlp_multiplier", 4.0),
            multiple_of=kwargs.get("multiple_of"),
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


def build_fp8_llama_spec(**kwargs: Any) -> ModelSpec:
    """Llama backbone with FP8 (E4M3) weight-quantized linears (Blackwell format
    demonstrator). Uses the same compression seam as ternary_b158. PyTorch
    reference does dequant-then-bf16 matmul -- no SM120 speedup until the
    llm.kittens FP8 GEMM is wired in."""
    fp8_format = str(kwargs.get("fp8_format", "e4m3")).lower()
    compression = "fp8_e5m2" if fp8_format == "e5m2" else "fp8_e4m3"
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="llama", compression=compression, runtime="eager"),
        block_spec=BlockSpec(
            family="llama",
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_encoding="rope",
            use_qk_norm=bool(kwargs.get("use_qk_norm", True)),
            linear_bias=False,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            multiple_of=kwargs.get("multiple_of", 256),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
            compression=compression,
            fp8_amax_history_len=int(kwargs.get("fp8_amax_history_len", 16)),
            fp8_use_stochastic_rounding=bool(kwargs.get("fp8_use_stochastic_rounding", True)),
        ),
        default_tie_embeddings=False,
    )


def build_mxfp4_llama_spec(**kwargs: Any) -> ModelSpec:
    """Llama backbone with OCP MXFP4 microscaled weight linears (per-32-block
    E8M0 scale + FP4 E2M1 mantissa). DeepSeek-V4 uses FP4 for MoE experts; this
    is the dense-llama precision demonstrator."""
    mx_format = str(kwargs.get("mx_format", "mxfp4")).lower()
    compression = "mxfp8" if mx_format == "mxfp8" else "mxfp4"
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="llama", compression=compression, runtime="eager"),
        block_spec=BlockSpec(
            family="llama",
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_encoding="rope",
            use_qk_norm=bool(kwargs.get("use_qk_norm", True)),
            linear_bias=False,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            multiple_of=kwargs.get("multiple_of", 256),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
            compression=compression,
            mx_block_size=int(kwargs.get("mx_block_size", 32)),
        ),
        default_tie_embeddings=False,
    )


def build_gemma3_spec(**kwargs: Any) -> ModelSpec:
    """Gemma-2/3 style: sliding-window local attention + GeGLU + QK-norm +
    logit softcap (already supported) on an RMSNorm/RoPE/GQA Llama backbone.
    Ships windowed-every-layer; the full local/global i%n interleave is a
    backbone-loop follow-up."""
    kwargs = {**kwargs}
    kwargs.setdefault("logit_softcap", 30.0)
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="llama", runtime="eager"),
        block_spec=BlockSpec(
            family="llama",
            norm_type="rmsnorm",
            mlp_type="geglu",
            pos_encoding="rope",
            attention_variant="sliding_window",
            window_size=int(kwargs.get("window_size", 256)),
            use_qk_norm=bool(kwargs.get("use_qk_norm", True)),
            linear_bias=False,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            multiple_of=kwargs.get("multiple_of", 256),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
        ),
        default_tie_embeddings=False,
    )


def build_diff_transformer_spec(**kwargs: Any) -> ModelSpec:
    """Differential Transformer: two-softmax-branch differential attention (noise
    cancellation) + head-wise norm on an RMSNorm/SwiGLU/RoPE/GQA backbone.
    Requires an even head_dim (already enforced by RoPE)."""
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="llama", runtime="eager"),
        block_spec=BlockSpec(
            family="llama",
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_encoding="rope",
            attention_variant="differential",
            diff_lambda_init=float(kwargs.get("diff_lambda_init", 0.8)),
            linear_bias=False,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            multiple_of=kwargs.get("multiple_of", 256),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
        ),
        default_tie_embeddings=False,
    )


def build_longctx_sparse_llama_spec(**kwargs: Any) -> ModelSpec:
    """Long-context efficiency backbone: native-sparse attention (local window +
    sink tokens + strided compressed history, the DeepSeek NSA / V4-CSA spirit)
    on an RMSNorm/SwiGLU/RoPE/GQA Llama. ``attention_variant`` may be set to
    ``block_sparse``/``sliding_window``/``streaming`` for the simpler patterns."""
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="llama", runtime="eager"),
        block_spec=BlockSpec(
            family="llama",
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_encoding="rope",
            attention_variant=str(kwargs.get("attention_variant", "nsa")),
            window_size=int(kwargs.get("window_size", 128)),
            sparse_block_size=int(kwargs.get("sparse_block_size", 64)),
            num_sinks=int(kwargs.get("num_sinks", 4)),
            nsa_compress_stride=int(kwargs.get("nsa_compress_stride", 16)),
            use_qk_norm=bool(kwargs.get("use_qk_norm", True)),
            linear_bias=False,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            multiple_of=kwargs.get("multiple_of", 256),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
        ),
        default_tie_embeddings=False,
    )


def build_qwen3_longctx_spec(**kwargs: Any) -> ModelSpec:
    """Qwen/Llama long-context: GQA + YaRN RoPE scaling (now honored) + QK-norm."""
    scaling = kwargs.get("rope_scaling") or {
        "type": "yarn",
        "factor": float(kwargs.get("rope_factor", 4.0)),
        "original_max_position": int(kwargs.get("original_max_position", 2048)),
    }
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="llama", runtime="eager"),
        block_spec=BlockSpec(
            family="llama",
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_encoding="rope",
            use_qk_norm=bool(kwargs.get("use_qk_norm", True)),
            rope_scaling=scaling,
            linear_bias=False,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            multiple_of=kwargs.get("multiple_of", 256),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
        ),
        default_tie_embeddings=False,
    )


def build_auxfree_moe_jepa_evo_spec(**kwargs: Any) -> ModelSpec:
    """MoE JEPA Evo with DeepSeek-V3 auxiliary-loss-free load balancing crossed
    with route evolution -- "domain experts, evolved, balanced without aux loss"."""
    spec = build_moe_jepa_evo_spec(**kwargs)
    spec.block_spec.moe_balance_mode = "auxfree"
    spec.block_spec.auxfree_bias_lr = float(kwargs.get("auxfree_bias_lr", 0.001))
    spec.block_spec.router_aux_loss_coef = 0.0
    return spec


def build_diff_semantic_moe_jepa_evo_spec(**kwargs: Any) -> ModelSpec:
    """Differential attention crossed with NeuralFn's semantic chunk-routed MoE +
    JEPA + route evolution -- noise-cancelled attention feeding semantic routing."""
    spec = build_semantic_moe_jepa_evo_spec(**kwargs)
    spec.block_spec.attention_variant = "differential"
    spec.block_spec.use_qk_norm = bool(kwargs.get("use_qk_norm", True))
    spec.block_spec.diff_lambda_init = float(kwargs.get("diff_lambda_init", 0.8))
    return spec


def build_dyt_geglu_semantic_dense_jepa_evo_spec(**kwargs: Any) -> ModelSpec:
    """Norm-free Dynamic Tanh + GeGLU crossed with the semantic dense JEPA Evo
    research stack."""
    spec = build_semantic_dense_jepa_evo_spec(**kwargs)
    spec.block_spec.norm_type = str(kwargs.get("norm_type", "dyt"))
    spec.block_spec.mlp_type = str(kwargs.get("mlp_type", "geglu"))
    spec.block_spec.use_qk_norm = bool(kwargs.get("use_qk_norm", True))
    return spec


def build_deepseek_v3_spec(**kwargs: Any) -> ModelSpec:
    """DeepSeek-V3 style: Multi-head Latent Attention (compressed-KV + decoupled
    RoPE) + auxiliary-loss-free balanced fine-grained MoE with shared experts, on
    an RMSNorm/SwiGLU/RoPE backbone. MLA owns its own attention path."""
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="mixllama", sparsity="moe", router_mode="standard", runtime="eager"),
        block_spec=BlockSpec(
            family="mixllama",
            norm_type="rmsnorm",
            mlp_type="moe",
            pos_encoding="rope",
            attention_variant="mla",
            moe_balance_mode="auxfree",
            auxfree_bias_lr=float(kwargs.get("auxfree_bias_lr", 0.001)),
            router_aux_loss_coef=0.0,
            experts=int(kwargs.get("experts", 8)),
            top_k=int(kwargs.get("top_k", 2)),
            shared_experts=int(kwargs.get("shared_experts", 1)),
            linear_bias=False,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            multiple_of=kwargs.get("multiple_of"),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
        ),
        default_tie_embeddings=False,
    )


def build_deepseek_v4_spec(**kwargs: Any) -> ModelSpec:
    """DeepSeek-V4-Pro style capstone: native-sparse (CSA-spirit) attention +
    auxiliary-loss-free balanced MoE + Manifold-Constrained Hyper-Connection
    residuals + QK-norm + FP8 (E4M3) dense/attention linears. (Per-tensor mixed
    FP4-expert / FP8-rest and the CSA/HCA layer interleave are documented
    follow-ups; experts remain bf16 in this reference.) Muon is the recommended
    optimizer (training-time)."""
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(backbone="mixllama", sparsity="moe", router_mode="standard", compression="fp8_e4m3", runtime="eager"),
        block_spec=BlockSpec(
            family="mixllama",
            norm_type="rmsnorm",
            mlp_type="moe",
            pos_encoding="rope",
            attention_variant="nsa",
            window_size=int(kwargs.get("window_size", 128)),
            sparse_block_size=int(kwargs.get("sparse_block_size", 64)),
            num_sinks=int(kwargs.get("num_sinks", 4)),
            nsa_compress_stride=int(kwargs.get("nsa_compress_stride", 16)),
            use_qk_norm=bool(kwargs.get("use_qk_norm", True)),
            residual_type="mhc",
            moe_balance_mode="auxfree",
            auxfree_bias_lr=float(kwargs.get("auxfree_bias_lr", 0.001)),
            router_aux_loss_coef=0.0,
            router_score_fn=str(kwargs.get("router_score_fn", "sqrt_softplus")),
            experts=int(kwargs.get("experts", 8)),
            top_k=int(kwargs.get("top_k", 2)),
            shared_experts=int(kwargs.get("shared_experts", 1)),
            compression="fp8_e4m3",
            linear_bias=False,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            multiple_of=kwargs.get("multiple_of"),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
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
            mlp_multiplier=kwargs.get("mlp_multiplier", 4.0),
            multiple_of=kwargs.get("multiple_of"),
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


def build_dense_jepa_evo_spec(**kwargs: Any) -> ModelSpec:
    """Experimental: dense autoregressive decoder with JEPA target supervision."""
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(
            objective="ar_jepa",
            backbone="llama",
            tokenization="sp",
            sparsity="dense",
            router_mode="none",
            compression="none",
            adapter="none",
            runtime="compile",
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
            rope_theta=kwargs.get("rope_base", kwargs.get("rope_theta", 10000.0)),
            qk_gain_init=kwargs.get("qk_gain_init", 1.0),
        ),
        default_tie_embeddings=False,
    )


def build_moe_jepa_evo_spec(**kwargs: Any) -> ModelSpec:
    """Experimental: standard MoE autoregressive decoder with JEPA target supervision."""
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(
            objective="ar_jepa",
            backbone="mixllama",
            tokenization="sp",
            sparsity="moe",
            router_mode="standard",
            compression="none",
            adapter="none",
            runtime="compile",
        ),
        block_spec=BlockSpec(
            family="mixllama",
            norm_type="rmsnorm",
            mlp_type="moe",
            pos_encoding="rope",
            linear_bias=False,
            dropout_p=0.0,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            multiple_of=kwargs.get("multiple_of"),
            experts=int(kwargs.get("experts", 8)),
            top_k=int(kwargs.get("top_k", 2)),
            router_aux_loss_coef=float(kwargs.get("router_aux_loss_coef", 0.01)),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
            attention_backend="sdpa",
            rope_theta=kwargs.get("rope_base", kwargs.get("rope_theta", 10000.0)),
            qk_gain_init=kwargs.get("qk_gain_init", 1.0),
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
            multiple_of=kwargs.get("multiple_of"),
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


def build_semantic_dense_jepa_evo_spec(**kwargs: Any) -> ModelSpec:
    """Experimental: dense decoder with chunk semantic JEPA planning."""
    dense_kwargs = dict(kwargs)
    dense_kwargs.setdefault("route_evo_enabled", False)
    return _base_model_spec(
        kwargs=dense_kwargs,
        template=TemplateSpec(
            objective="semantic_dense_jepa_evo",
            backbone="llama",
            tokenization="sp",
            sparsity="dense",
            router_mode="semantic",
            compression="none",
            adapter="none",
            runtime="compile",
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
            rope_theta=kwargs.get("rope_base", kwargs.get("rope_theta", 10000.0)),
            qk_gain_init=kwargs.get("qk_gain_init", 1.0),
        ),
        default_tie_embeddings=False,
    )


def build_semantic_moe_jepa_evo_spec(**kwargs: Any) -> ModelSpec:
    """Experimental: chunk-routed semantic MoE with JEPA supervision and route evolution."""
    shared_experts = int(kwargs.get("semantic_shared_experts", 2))
    free_experts = int(kwargs.get("semantic_free_experts", 8))
    if shared_experts < 0:
        raise ValueError("semantic_moe_jepa_evo requires semantic_shared_experts >= 0")
    if free_experts < 0:
        raise ValueError("semantic_moe_jepa_evo requires semantic_free_experts >= 0")
    total_experts = shared_experts + NUM_VOCAB_DIMS + free_experts
    experts = int(kwargs.get("experts", total_experts))
    if experts != total_experts:
        raise ValueError(
            "semantic_moe_jepa_evo requires experts to equal "
            f"semantic_shared_experts + {NUM_VOCAB_DIMS} vocab experts + semantic_free_experts "
            f"({total_experts}), got {experts}"
        )
    top_k = min(int(kwargs.get("top_k", 2)), NUM_VOCAB_DIMS + free_experts)
    return _base_model_spec(
        kwargs=kwargs,
        template=TemplateSpec(
            objective="semantic_moe_jepa_evo",
            backbone="mixllama",
            tokenization="sp",
            sparsity="moe",
            router_mode="semantic",
            compression="none",
            adapter="none",
            runtime="compile",
        ),
        block_spec=BlockSpec(
            family="mixllama",
            norm_type="rmsnorm",
            mlp_type="moe",
            pos_encoding="rope",
            linear_bias=False,
            dropout_p=0.0,
            mlp_multiplier=kwargs.get("mlp_multiplier", 8.0 / 3.0),
            multiple_of=kwargs.get("multiple_of"),
            experts=experts,
            top_k=top_k,
            router_aux_loss_coef=float(kwargs.get("router_aux_loss_coef", 0.01)),
            num_heads=kwargs.get("num_heads", 4),
            num_kv_heads=kwargs.get("num_kv_heads", 2),
            attention_backend="sdpa",
            rope_theta=kwargs.get("rope_base", kwargs.get("rope_theta", 10000.0)),
            qk_gain_init=kwargs.get("qk_gain_init", 1.0),
        ),
        default_tie_embeddings=False,
    )
