from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from neuralfn.builtins import BuiltinNeurons
from neuralfn.neuron import NeuronDef

KernelKind = Literal["function", "module", "optimizer", "runtime"]
KernelStatus = Literal["tile", "torch_fallback", "host_only", "delegated", "planned"]
TRACKED_DTYPES = ("float32", "float16", "float8_e4m3fn", "float8_e5m2", "nvfp4")

_TILE_FUNCTION_NAMES = {
    "identity",
    "negate",
    "add",
    "multiply",
    "relu",
    "sigmoid",
    "tanh_neuron",
    "gaussian",
    "log",
    "leaky_relu",
    "prelu",
    "relu6",
    "elu",
    "selu",
    "silu",
    "mish",
    "softplus",
    "softsign",
    "hard_sigmoid",
    "hard_tanh",
    "hard_swish",
    "threshold",
    "gelu",
    "softmax_2",
    "logsoftmax_2",
}


@dataclass(frozen=True)
class TileKernelSpec:
    name: str
    kind: KernelKind
    status: KernelStatus
    fallback_reason: str = ""
    aliases: tuple[str, ...] = ()
    delegated_to: str = ""
    dtypes: tuple[str, ...] = ()
    shape_contract: str = ""
    has_forward: bool = False
    has_backward: bool = False
    no_grad_reason: str = ""
    dtype_support: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dtype_support:
            reason = "No dtype compute contract for this registry entry." if not self.dtypes else "Not supported by this registry entry."
            object.__setattr__(self, "dtype_support", _dtype_support_matrix(self.dtypes, reason))

    @property
    def inventory_key(self) -> str:
        return tile_kernel_key(self.kind, self.name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inventory_key": self.inventory_key,
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "fallback_reason": self.fallback_reason,
            "aliases": list(self.aliases),
            "delegated_to": self.delegated_to,
            "dtypes": list(self.dtypes),
            "dtype_support": dict(self.dtype_support),
            "shape_contract": self.shape_contract,
            "has_forward": self.has_forward,
            "has_backward": self.has_backward,
            "no_grad_reason": self.no_grad_reason,
        }


@dataclass(frozen=True)
class KernelCoverageReport:
    total_inventory: int
    accounted: int
    missing: tuple[str, ...]
    by_status: dict[str, int]
    by_dtype: dict[str, dict[str, int]]
    specs: tuple[TileKernelSpec, ...]

    @property
    def complete(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_inventory": self.total_inventory,
            "accounted": self.accounted,
            "missing": list(self.missing),
            "by_status": dict(self.by_status),
            "by_dtype": {dtype: dict(counts) for dtype, counts in self.by_dtype.items()},
            "complete": self.complete,
            "specs": [spec.to_dict() for spec in self.specs],
        }


@dataclass
class TileKernelRegistry:
    specs: dict[str, TileKernelSpec] = field(default_factory=dict)

    def register(self, spec: TileKernelSpec) -> None:
        self.specs[spec.inventory_key] = spec
        for alias in spec.aliases:
            self.specs[tile_kernel_key(spec.kind, alias)] = spec

    def get(self, name: str, *, kind: KernelKind | None = None) -> TileKernelSpec | None:
        if kind is not None:
            return self.specs.get(tile_kernel_key(kind, name))
        matches = [
            spec
            for spec in self.canonical_specs()
            if spec.name == name or name in spec.aliases
        ]
        return matches[0] if len(matches) == 1 else None

    def canonical_specs(self) -> tuple[TileKernelSpec, ...]:
        seen: set[str] = set()
        specs: list[TileKernelSpec] = []
        for spec in self.specs.values():
            if spec.inventory_key in seen:
                continue
            seen.add(spec.inventory_key)
            specs.append(spec)
        return tuple(sorted(specs, key=lambda item: (item.kind, item.name)))

    def coverage_report(self, inventory: Iterable[str] | None = None) -> KernelCoverageReport:
        inventory_names = tuple(sorted(set(inventory or tile_kernel_inventory())))
        missing = tuple(name for name in inventory_names if name not in self.specs)
        canonical = self.canonical_specs()
        by_status: dict[str, int] = {}
        for spec in canonical:
            by_status[spec.status] = by_status.get(spec.status, 0) + 1
        by_dtype: dict[str, dict[str, int]] = {}
        for dtype in TRACKED_DTYPES:
            supported = sum(1 for spec in canonical if spec.dtype_support.get(dtype) == "supported")
            by_dtype[dtype] = {
                "supported": supported,
                "unsupported": len(canonical) - supported,
            }
        return KernelCoverageReport(
            total_inventory=len(inventory_names),
            accounted=len(inventory_names) - len(missing),
            missing=missing,
            by_status=by_status,
            by_dtype=by_dtype,
            specs=canonical,
        )


def tile_kernel_key(kind: KernelKind, name: str) -> str:
    return f"{kind}:{name}"


def _dtype_support_matrix(
    supported: Iterable[str],
    unsupported_reason: str,
    *,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    supported_set = set(supported)
    matrix = {
        dtype: "supported" if dtype in supported_set else unsupported_reason
        for dtype in TRACKED_DTYPES
    }
    if overrides:
        for dtype, reason in overrides.items():
            if dtype not in supported_set:
                matrix[dtype] = reason
    return matrix


def builtin_function_inventory() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                tile_kernel_key("function", neuron_def.name)
                for neuron_def in BuiltinNeurons.all()
                if neuron_def.kind == "function"
            }
        )
    )


def builtin_module_inventory() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                tile_kernel_key("module", neuron_def.module_type)
                for neuron_def in BuiltinNeurons.all()
                if neuron_def.kind == "module"
            }
        )
    )


def build_module_dispatch_inventory() -> tuple[str, ...]:
    return builtin_module_inventory()


def build_function_dispatch_inventory() -> tuple[str, ...]:
    return builtin_function_inventory()


def optimizer_runtime_inventory() -> tuple[str, ...]:
    return tuple(tile_kernel_key("optimizer", name) for name in (
        "adamw_step",
        "ema_update",
        "gradient_accumulate",
        "gradient_clip_norm",
        "muon_newton_schulz",
        "muon_step",
        "split_optimizer_step",
    ))


def tile_kernel_inventory() -> tuple[str, ...]:
    names = (
        set(builtin_function_inventory())
        | set(builtin_module_inventory())
        | set(build_module_dispatch_inventory())
        | set(build_function_dispatch_inventory())
        | set(optimizer_runtime_inventory())
    )
    return tuple(sorted(names))


def _spec_for_function(name: str) -> TileKernelSpec:
    if name in {"input", "output"}:
        return TileKernelSpec(
            name=name,
            kind="function",
            status="host_only",
            fallback_reason="Graph interface marker; real tensors enter or leave the compiled plan without compute.",
            aliases=(f"{name}_node",),
            has_forward=False,
            has_backward=False,
            no_grad_reason="Interface marker.",
        )
    if name in _TILE_FUNCTION_NAMES:
        is_binary = name in {"add", "multiply"}
        is_binary_pair = name in {"softmax_2", "logsoftmax_2"}
        return TileKernelSpec(
            name=name,
            kind="function",
            status="tile",
            fallback_reason="Falls back to PyTorch for non-CUDA, unsupported dtype, non-contiguous, or broadcasted inputs.",
            dtypes=("float32", "float16", "float8_e4m3fn", "float8_e5m2"),
            dtype_support=_dtype_support_matrix(
                ("float32", "float16", "float8_e4m3fn", "float8_e5m2"),
                "Scalar Tile functions require an explicit quantize/dequantize, saturation, and gradient contract before lower-precision support is advertised.",
                overrides={
                    "nvfp4": "Requires the NeuralFn NVFP4 packed representation, scale metadata, and gradient parity coverage.",
                },
            ),
            shape_contract=(
                "binary contiguous CUDA float32, float16, float8_e4m3fn, or float8_e5m2 tensors with identical shapes"
                if is_binary
                else "binary contiguous CUDA float32, float16, float8_e4m3fn, or float8_e5m2 tensors with identical shapes and two outputs"
                if is_binary_pair
                else "unary contiguous CUDA float32, float16, float8_e4m3fn, or float8_e5m2 tensor"
            ),
            has_forward=True,
            has_backward=True,
        )
    return TileKernelSpec(
        name=name,
        kind="function",
        status="torch_fallback",
        fallback_reason="CUDA Tile implementation pending; PyTorch function stage remains authoritative.",
        dtypes=("float32", "float16", "bfloat16"),
        shape_contract="broadcast/elementwise unless documented otherwise",
        has_forward=False,
        has_backward=False,
    )


_HOST_ONLY_MODULES = {
    "dataset_source",
    "semantic_data_source",
    "sft_dataset_source",
    "dpo_dataset_source",
    "ppo_rollout_source",
}

_DELEGATED_MODULES = {
    "reference_forward": "compiled reference graph",
    "reward_forward": "compiled reward graph",
}

_TILE_MODULE_TYPES = {
    "logit_softcap",
    "loss_scale",
    "aux_loss_add",
    "act_halt_gate",
    "kl_penalty",
    "expert_combine",
    "gae_compute",
    "gelu",
    "jepa_projector",
    "jepa_predictor",
    "kv_pca_encode",
    "kv_pca_decode",
    "kv_quant_pack",
    "kv_quant_unpack",
    "kv_cache_write",
    "kv_cache_read",
    "latent_pool",
    "masked_token_cross_entropy",
    "ppo_clipped_loss",
    "absolute_position_embedding",
    "token_embedding",
    "residual_add",
    "residual_mix",
    "manifold_hyper_connection",
    "qk_gain",
    "dyt",
    "dropout",
    "reshape_heads",
    "merge_heads",
    "repeat_kv",
    "rotary_embedding",
    "rms_norm",
    "layer_norm",
    "group_norm",
    "qk_norm",
    "linear",
    "bitlinear_ternary",
    "fp8_linear",
    "mx_linear",
    "nf4_linear",
    "randmap_adapter",
    "mlp_relu2",
    "swiglu",
    "geglu",
    "reglu",
    "solu",
    "load_balance_loss",
    "lm_head",
    "tied_lm_head",
    "ttt_linear",
    "lora_linear",
    "router_logits",
    "value_head",
    "reward_head",
    "denoise_head",
    "attentionless_decoder",
    "auxfree_load_balancing",
    "expert_dispatch",
    "act_weighted_sum",
    "dpo_pairwise_loss",
    "preference_bce_loss",
    "route_balance_loss",
    "route_distillation_loss",
    "route_selection_loss",
    "semantic_alignment_loss",
    "semantic_chunk_hasher",
    "semantic_chunk_projector",
    "semantic_hash_router",
    "semantic_hasher",
    "semantic_moe_jepa_evo_router",
    "semantic_moe_router",
    "semantic_projector",
    "sequence_logp",
    "scaled_dot_product_attention",
    "sliding_window_attention",
    "block_sparse_attention",
    "streaming_attention_sinks",
    "native_sparse_attention",
    "differential_attention",
    "causal_self_attention",
    "fused_causal_attention",
    "multi_latent_attention",
    "routed_attention_experts",
    "mamba",
    "universal_transformer",
    "random_timesteps",
    "mask_scheduler",
    "jepa_mask",
    "softmax_distillation_loss",
    "token_cross_entropy",
    "topk_route",
    "broadcast_expert_routes",
    "broadcast_chunk_routes",
    "byte_patch_embed",
    "byte_patch_merge",
    "causal_chunk_state",
    "latent_mse_loss",
}

_FP16_TILE_MODULE_TYPES = {
    "logit_softcap",
    "loss_scale",
    "aux_loss_add",
    "kl_penalty",
    "residual_add",
    "residual_mix",
    "manifold_hyper_connection",
    "qk_gain",
    "dyt",
    "dropout",
    "act_weighted_sum",
    "latent_pool",
    "rms_norm",
    "layer_norm",
    "group_norm",
    "qk_norm",
    "linear",
    "lm_head",
    "tied_lm_head",
    "router_logits",
    "value_head",
    "reward_head",
    "denoise_head",
    "kv_pca_encode",
    "kv_pca_decode",
    "jepa_projector",
    "jepa_predictor",
    "ttt_linear",
    "lora_linear",
    "bitlinear_ternary",
    "fp8_linear",
    "mx_linear",
    "randmap_adapter",
    "mlp_relu2",
    "swiglu",
    "geglu",
    "reglu",
    "solu",
    "act_halt_gate",
    "rotary_embedding",
    "scaled_dot_product_attention",
    "sliding_window_attention",
    "block_sparse_attention",
    "streaming_attention_sinks",
    "native_sparse_attention",
    "differential_attention",
    "causal_self_attention",
    "fused_causal_attention",
    "multi_latent_attention",
    "routed_attention_experts",
    "latent_mse_loss",
    "token_cross_entropy",
    "masked_token_cross_entropy",
    "sequence_logp",
    "preference_bce_loss",
    "ppo_clipped_loss",
    "gae_compute",
    "dpo_pairwise_loss",
    "route_selection_loss",
    "route_balance_loss",
    "load_balance_loss",
    "semantic_alignment_loss",
    "softmax_distillation_loss",
    "route_distillation_loss",
}

_FP8_TILE_MODULE_TYPES = {
    "logit_softcap",
    "loss_scale",
    "aux_loss_add",
    "kl_penalty",
    "residual_add",
    "residual_mix",
    "manifold_hyper_connection",
    "qk_gain",
    "dyt",
    "linear",
    "lm_head",
    "tied_lm_head",
    "router_logits",
    "value_head",
    "reward_head",
    "denoise_head",
    "kv_pca_encode",
    "kv_pca_decode",
    "jepa_projector",
    "jepa_predictor",
    "ttt_linear",
    "lora_linear",
    "bitlinear_ternary",
    "fp8_linear",
    "mx_linear",
    "randmap_adapter",
    "mlp_relu2",
    "swiglu",
    "geglu",
    "reglu",
    "solu",
    "act_halt_gate",
    "scaled_dot_product_attention",
    "sliding_window_attention",
    "block_sparse_attention",
    "streaming_attention_sinks",
    "native_sparse_attention",
    "differential_attention",
    "causal_self_attention",
    "fused_causal_attention",
    "multi_latent_attention",
    "routed_attention_experts",
}

_FP8_FLOAT32_OUTPUT_MODULE_TYPES = {
    "linear",
    "lm_head",
    "tied_lm_head",
    "router_logits",
    "value_head",
    "reward_head",
    "denoise_head",
    "kv_pca_encode",
    "kv_pca_decode",
    "jepa_projector",
    "jepa_predictor",
    "ttt_linear",
    "lora_linear",
    "bitlinear_ternary",
    "fp8_linear",
    "mx_linear",
    "randmap_adapter",
    "mlp_relu2",
    "swiglu",
    "geglu",
    "reglu",
    "solu",
    "act_halt_gate",
    "scaled_dot_product_attention",
    "sliding_window_attention",
    "block_sparse_attention",
    "streaming_attention_sinks",
    "native_sparse_attention",
    "differential_attention",
    "causal_self_attention",
    "fused_causal_attention",
    "multi_latent_attention",
    "routed_attention_experts",
}

_NVFP4_TILE_MODULE_TYPES = {
    "linear",
    "lm_head",
    "tied_lm_head",
    "router_logits",
    "value_head",
    "reward_head",
    "denoise_head",
    "kv_pca_encode",
    "kv_pca_decode",
    "jepa_projector",
    "jepa_predictor",
    "ttt_linear",
    "lora_linear",
    "bitlinear_ternary",
    "fp8_linear",
    "mx_linear",
    "randmap_adapter",
    "mlp_relu2",
    "swiglu",
    "geglu",
    "reglu",
    "solu",
    "act_halt_gate",
    "scaled_dot_product_attention",
    "sliding_window_attention",
    "block_sparse_attention",
    "streaming_attention_sinks",
    "native_sparse_attention",
    "differential_attention",
    "causal_self_attention",
    "fused_causal_attention",
    "multi_latent_attention",
    "routed_attention_experts",
}

_LOSS_REDUCTION_MODULE_TYPES = {
    "token_cross_entropy",
    "masked_token_cross_entropy",
    "sequence_logp",
    "latent_mse_loss",
    "semantic_alignment_loss",
    "dpo_pairwise_loss",
    "ppo_clipped_loss",
    "gae_compute",
    "preference_bce_loss",
    "load_balance_loss",
    "route_balance_loss",
    "route_selection_loss",
    "route_distillation_loss",
    "softmax_distillation_loss",
}

_STOCHASTIC_OR_MASK_MODULE_TYPES = {
    "dropout",
    "random_timesteps",
    "mask_scheduler",
    "jepa_mask",
}

_INTEGER_HASH_OR_ROUTING_OUTPUT_MODULE_TYPES = {
    "topk_route",
    "expert_dispatch",
    "expert_combine",
    "auxfree_load_balancing",
    "semantic_hasher",
    "semantic_chunk_hasher",
    "semantic_moe_router",
    "semantic_hash_router",
    "semantic_moe_jepa_evo_router",
    "attentionless_decoder",
    "kv_quant_pack",
    "kv_quant_unpack",
    "kv_cache_write",
    "kv_cache_read",
    "broadcast_expert_routes",
    "broadcast_chunk_routes",
    "byte_patch_embed",
    "byte_patch_merge",
    "causal_chunk_state",
}

_SEMANTIC_PROJECTOR_MODULE_TYPES = {
    "semantic_projector",
    "semantic_chunk_projector",
}


def _module_dtype_support_overrides(module_type: str) -> dict[str, str]:
    if module_type in _SEMANTIC_PROJECTOR_MODULE_TYPES:
        reason = (
            "Semantic projectors emit argmax-derived topic/signature semantics; "
            "lower-precision activation quantization can change categorical decisions, "
            "so only the float32 categorical contract is advertised."
        )
        return {
            "float16": reason,
            "float8_e4m3fn": reason,
            "float8_e5m2": reason,
            "nvfp4": reason,
        }
    if module_type in _LOSS_REDUCTION_MODULE_TYPES:
        return {
            "float8_e4m3fn": "Loss/reduction kernels require fp32 accumulation and have no advertised fp8 loss-surface, saturation, or gradient contract.",
            "float8_e5m2": "Loss/reduction kernels require fp32 accumulation and have no advertised fp8 loss-surface, saturation, or gradient contract.",
            "nvfp4": "Loss/reduction kernels require dense floating tensors; NVFP4 packed activations need explicit dequantization, accumulation, and gradient semantics.",
        }
    if module_type in _STOCHASTIC_OR_MASK_MODULE_TYPES:
        return {
            "float8_e4m3fn": "Stochastic or mask-producing modules are not meaningful fp8 compute kernels until RNG/mask storage and gradient semantics are defined.",
            "float8_e5m2": "Stochastic or mask-producing modules are not meaningful fp8 compute kernels until RNG/mask storage and gradient semantics are defined.",
            "nvfp4": "Stochastic or mask-producing modules are not meaningful NVFP4 compute kernels until RNG/mask storage and packed-gradient semantics are defined.",
        }
    if module_type in _INTEGER_HASH_OR_ROUTING_OUTPUT_MODULE_TYPES:
        return {
            "float8_e4m3fn": "This module produces integer, hash, cache, routing, or dispatch outputs where fp8 activation storage is not the meaningful contract.",
            "float8_e5m2": "This module produces integer, hash, cache, routing, or dispatch outputs where fp8 activation storage is not the meaningful contract.",
            "nvfp4": "This module produces integer, hash, cache, routing, or dispatch outputs where NVFP4 packed activation storage is not the meaningful contract.",
        }
    if module_type == "nf4_linear":
        return {
            "float8_e4m3fn": "NF4 linear uses an NF4 packed base-weight contract; fp8 activation support for this wrapper needs a separate mixed NF4/fp8 scale contract.",
            "float8_e5m2": "NF4 linear uses an NF4 packed base-weight contract; fp8 activation support for this wrapper needs a separate mixed NF4/fp8 scale contract.",
            "nvfp4": "NF4 linear uses an NF4 packed base-weight contract; NVFP4 support would be a separate packed-format conversion contract.",
        }
    return {
        "float8_e4m3fn": "Requires module-specific fp8 scale/amax, saturation, fp32 accumulation, and parity coverage.",
        "float8_e5m2": "Requires module-specific fp8 scale/amax, saturation, fp32 accumulation, and parity coverage.",
        "nvfp4": "Requires the NeuralFn NVFP4 packed representation, scale metadata, fp32 accumulation, and parity coverage.",
    }


def _spec_for_module(module_type: str) -> TileKernelSpec:
    if module_type in _HOST_ONLY_MODULES:
        dtype_reason = "Source/orchestration nodes are control-plane interfaces; real training tensors are materialized by the trainer and do not pass through editor nodes."
        return TileKernelSpec(
            name=module_type,
            kind="module",
            status="host_only",
            fallback_reason="Source/orchestration node; batches are materialized by the trainer, not by a compute kernel.",
            dtype_support=_dtype_support_matrix((), dtype_reason),
            has_forward=False,
            has_backward=False,
            no_grad_reason="No device compute contract.",
        )
    if module_type in _DELEGATED_MODULES:
        dtype_reason = "Delegated graph calls inherit dtype support from the compiled child graph selected at runtime."
        return TileKernelSpec(
            name=module_type,
            kind="module",
            status="delegated",
            delegated_to=_DELEGATED_MODULES[module_type],
            fallback_reason="Delegates to another compiled graph; coverage is inherited from that graph's selected backend.",
            dtype_support=_dtype_support_matrix((), dtype_reason),
            has_forward=False,
            has_backward=False,
        )
    if module_type in _TILE_MODULE_TYPES:
        dtype_list = ["float32"]
        if module_type in _FP16_TILE_MODULE_TYPES:
            dtype_list.append("float16")
        if module_type in _FP8_TILE_MODULE_TYPES:
            dtype_list.extend(["float8_e4m3fn", "float8_e5m2"])
        if module_type in _NVFP4_TILE_MODULE_TYPES:
            dtype_list.append("nvfp4")
        dtypes = tuple(dtype_list)
        dtype_support = _dtype_support_matrix(
            dtypes,
            "This module has not completed dtype-specific CUDA Tile parity coverage for this dtype.",
            overrides=_module_dtype_support_overrides(module_type),
        )
        if module_type in _FP8_TILE_MODULE_TYPES or module_type in _NVFP4_TILE_MODULE_TYPES:
            fp8_output = (
                "return float32 outputs"
                if module_type in _FP8_FLOAT32_OUTPUT_MODULE_TYPES
                else "requantize activation outputs back to the input fp8 format"
            )
            activation_dtypes = "float32 or float16"
            low_precision_clauses: list[str] = []
            if module_type in _FP8_TILE_MODULE_TYPES:
                activation_dtypes += ", float8_e4m3fn, or float8_e5m2"
                low_precision_clauses.append(
                    f"fp8 activations dequantize to float32, compute through Tile float32 kernels, and {fp8_output}"
                )
            if module_type in _NVFP4_TILE_MODULE_TYPES:
                activation_dtypes += ", or packed NVFP4Tensor"
                low_precision_clauses.append(
                    "NVFP4Tensor activations dequantize through NeuralFn scale metadata, compute through Tile float32 kernels, and return float32 outputs"
                )
            shape_contract = (
                f"contiguous CUDA {activation_dtypes} activations matching the existing PyTorch stage contract; "
                + "; ".join(low_precision_clauses)
                + "; "
                "parameters, weights, and masks remain float32"
            )
            fallback_reason = (
                "Falls back to PyTorch for non-CUDA, unsupported activation dtype, non-contiguous tensors, or unsupported broadcast contracts."
            )
        elif module_type in _FP16_TILE_MODULE_TYPES:
            shape_contract = "contiguous CUDA float32 or float16 activations matching the existing PyTorch stage contract; parameters, weights, and masks remain float32"
            fallback_reason = "Falls back to PyTorch for non-CUDA, non-float32/non-float16 activations, non-contiguous tensors, or unsupported broadcast contracts."
        else:
            shape_contract = "contiguous CUDA float32 tensors matching the existing PyTorch stage contract"
            fallback_reason = "Falls back to PyTorch for non-CUDA, non-float32, non-contiguous, or unsupported broadcast contracts."
        return TileKernelSpec(
            name=module_type,
            kind="module",
            status="tile",
            fallback_reason=fallback_reason,
            dtypes=dtypes,
            dtype_support=dtype_support,
            shape_contract=shape_contract,
            has_forward=True,
            has_backward=True,
        )
    return TileKernelSpec(
        name=module_type,
        kind="module",
        status="torch_fallback",
        fallback_reason="CUDA Tile implementation pending; PyTorch module stage remains authoritative.",
        dtypes=("float32", "float16", "bfloat16"),
        shape_contract="matches existing PyTorch stage",
        has_forward=False,
        has_backward=False,
    )


def _spec_for_optimizer(name: str) -> TileKernelSpec:
    fp16_optimizer_dtype_support = _dtype_support_matrix(
        ("float32", "float16"),
        "Optimizer/runtime helpers require a parameter-state contract for this dtype before support is advertised.",
        overrides={
            "float8_e4m3fn": "Optimizer state updates do not yet define fp8 parameter/state storage, scale metadata, or accumulation semantics.",
            "float8_e5m2": "Optimizer state updates do not yet define fp8 parameter/state storage, scale metadata, or accumulation semantics.",
            "nvfp4": "Optimizer state updates do not yet define NVFP4 packed parameter/state storage, scale metadata, or accumulation semantics.",
        },
    )
    newton_schulz_dtype_support = _dtype_support_matrix(
        ("float32",),
        "Muon Newton-Schulz orthogonalization accepts float32 matrix updates; fp16 callers should upcast before this helper.",
        overrides={
            "float8_e4m3fn": "Muon and split optimizer updates need a separate fp8 matrix-state and scale contract.",
            "float8_e5m2": "Muon and split optimizer updates need a separate fp8 matrix-state and scale contract.",
            "nvfp4": "Muon and split optimizer updates need a separate NVFP4 packed matrix-state and scale contract.",
        },
    )
    muon_optimizer_dtype_support = _dtype_support_matrix(
        ("float32", "float16"),
        "Muon and split optimizer helpers require a parameter/state contract for this dtype before support is advertised.",
        overrides={
            "float8_e4m3fn": "Muon and split optimizer updates need a separate fp8 matrix-state and scale contract.",
            "float8_e5m2": "Muon and split optimizer updates need a separate fp8 matrix-state and scale contract.",
            "nvfp4": "Muon and split optimizer updates need a separate NVFP4 packed matrix-state and scale contract.",
        },
    )
    if name == "adamw_step":
        return TileKernelSpec(
            name=name,
            kind="optimizer",
            status="tile",
            fallback_reason="Falls back to PyTorch for non-CUDA, unsupported dtype, non-contiguous, empty, shape-mismatched, or invalid-step tensors.",
            dtypes=("float32", "float16"),
            dtype_support=fp16_optimizer_dtype_support,
            shape_contract="same-shaped contiguous float32 parameter/gradient/moments, or fp16 parameter/gradient with float32 first/second moments; updates parameter and moments in-place",
            has_forward=True,
            has_backward=False,
            no_grad_reason="Optimizer/runtime update runs under no_grad.",
        )
    if name == "ema_update":
        return TileKernelSpec(
            name=name,
            kind="optimizer",
            status="tile",
            fallback_reason="Falls back to PyTorch for non-CUDA, non-float32/non-float16, non-contiguous, empty, or shape-mismatched tensors.",
            dtypes=("float32", "float16"),
            dtype_support=fp16_optimizer_dtype_support,
            shape_contract="same-shaped contiguous float32 or fp16 target/source parameter or buffer tensors; updates target in-place through fp32 Tile compute",
            has_forward=True,
            has_backward=False,
            no_grad_reason="Optimizer/runtime update runs under no_grad.",
        )
    if name == "gradient_accumulate":
        return TileKernelSpec(
            name=name,
            kind="optimizer",
            status="tile",
            fallback_reason="Falls back to PyTorch for non-CUDA, non-float32/non-float16, non-contiguous, empty, or shape-mismatched tensors.",
            dtypes=("float32", "float16"),
            dtype_support=fp16_optimizer_dtype_support,
            shape_contract="same-shaped contiguous float32 or fp16 accumulation-buffer and gradient tensors; updates buffer in-place through fp32 Tile compute",
            has_forward=True,
            has_backward=False,
            no_grad_reason="Optimizer/runtime accumulation runs under no_grad.",
        )
    if name == "gradient_clip_norm":
        return TileKernelSpec(
            name=name,
            kind="optimizer",
            status="tile",
            fallback_reason="Falls back to PyTorch for non-CUDA, non-float32/non-float16, non-contiguous, or empty gradient tensors.",
            dtypes=("float32", "float16"),
            dtype_support=fp16_optimizer_dtype_support,
            shape_contract="one or more contiguous float32 or fp16 gradient tensors; computes global L2 norm in fp32 and scales tensors in-place",
            has_forward=True,
            has_backward=False,
            no_grad_reason="Optimizer/runtime clipping runs under no_grad.",
        )
    if name == "muon_newton_schulz":
        return TileKernelSpec(
            name=name,
            kind="optimizer",
            status="tile",
            fallback_reason="Falls back to the PyTorch tensor composition for unsupported dtype, rank, empty, or non-contiguous inputs.",
            dtypes=("float32",),
            dtype_support=newton_schulz_dtype_support,
            shape_contract="non-empty matrix update tensor; returns the Newton-Schulz orthogonalized update",
            has_forward=True,
            has_backward=False,
            no_grad_reason="Optimizer/runtime matrix update runs under no_grad.",
        )
    if name == "muon_step":
        return TileKernelSpec(
            name=name,
            kind="optimizer",
            status="tile",
            fallback_reason="Falls back to the PyTorch tensor composition for unsupported dtype, shape, empty, or non-contiguous inputs.",
            dtypes=("float32", "float16"),
            dtype_support=muon_optimizer_dtype_support,
            shape_contract="same-shaped contiguous float32 parameter/gradient/momentum tensors, or fp16 parameter/gradient with float32 Muon momentum; updates parameter and momentum in-place through fp32 matrix compute",
            has_forward=True,
            has_backward=False,
            no_grad_reason="Optimizer/runtime update runs under no_grad.",
        )
    if name == "split_optimizer_step":
        return TileKernelSpec(
            name=name,
            kind="optimizer",
            status="tile",
            fallback_reason="Falls back to the PyTorch tensor composition for unsupported dtype, shape, empty, or non-contiguous inputs.",
            dtypes=("float32", "float16"),
            dtype_support=muon_optimizer_dtype_support,
            shape_contract="same-shaped contiguous float32 parameter/gradient plus optimizer state tensors, or fp16 parameter/gradient with float32 AdamW moments and Muon momentum; dispatches matrix parameters to Muon and others to AdamW",
            has_forward=True,
            has_backward=False,
            no_grad_reason="Optimizer/runtime update runs under no_grad.",
        )
    return TileKernelSpec(
        name=name,
        kind="optimizer",
        status="torch_fallback",
        fallback_reason="CUDA Tile optimizer kernel pending; PyTorch optimizer/runtime path remains authoritative.",
        dtypes=("float32", "float16", "bfloat16"),
        shape_contract="parameter tensor or optimizer-state tensor",
        has_forward=False,
        has_backward=False,
    )


def build_default_registry() -> TileKernelRegistry:
    registry = TileKernelRegistry()
    for key in builtin_function_inventory() + build_function_dispatch_inventory():
        _kind, name = key.split(":", 1)
        registry.register(_spec_for_function(name))
    for key in builtin_module_inventory() + build_module_dispatch_inventory():
        _kind, module_type = key.split(":", 1)
        registry.register(_spec_for_module(module_type))
    for key in optimizer_runtime_inventory():
        _kind, name = key.split(":", 1)
        registry.register(_spec_for_optimizer(name))
    return registry


DEFAULT_TILE_KERNEL_REGISTRY = build_default_registry()


def coverage_report() -> KernelCoverageReport:
    return DEFAULT_TILE_KERNEL_REGISTRY.coverage_report()


def tile_kernel_spec_for(neuron_def: NeuronDef) -> TileKernelSpec | None:
    if neuron_def.kind == "module":
        return DEFAULT_TILE_KERNEL_REGISTRY.get(neuron_def.module_type, kind="module")
    return DEFAULT_TILE_KERNEL_REGISTRY.get(neuron_def.name, kind="function")
