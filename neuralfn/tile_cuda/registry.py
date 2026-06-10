from __future__ import annotations

import ast
from dataclasses import dataclass, field
import inspect
from typing import Any, Iterable, Literal

from neuralfn.builtins import BuiltinNeurons
from neuralfn.neuron import NeuronDef

KernelKind = Literal["function", "module", "optimizer", "runtime"]
KernelStatus = Literal["tile", "torch_fallback", "host_only", "delegated", "planned"]

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
        return KernelCoverageReport(
            total_inventory=len(inventory_names),
            accounted=len(inventory_names) - len(missing),
            missing=missing,
            by_status=by_status,
            specs=canonical,
        )


def _string_values_from_build_function(fn: Any, parameter_name: str) -> tuple[str, ...]:
    tree = ast.parse(inspect.getsource(fn))
    values: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_Compare(self, node: ast.Compare) -> None:
            left = node.left
            if isinstance(left, ast.Name) and left.id == parameter_name:
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                        values.append(comparator.value)
                    elif isinstance(comparator, (ast.Set, ast.Tuple, ast.List)):
                        for elt in comparator.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                values.append(elt.value)
            self.generic_visit(node)

    Visitor().visit(tree)
    return tuple(dict.fromkeys(values))


def tile_kernel_key(kind: KernelKind, name: str) -> str:
    return f"{kind}:{name}"


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
    from neuralfn.torch_backend import build_module

    return tuple(
        sorted(
            tile_kernel_key("module", value)
            for value in _string_values_from_build_function(build_module, "module_type")
        )
    )


def build_function_dispatch_inventory() -> tuple[str, ...]:
    from neuralfn.torch_backend import build_function_module

    return tuple(
        sorted(
            tile_kernel_key("function", value)
            for value in _string_values_from_build_function(build_function_module, "name")
        )
    )


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
            fallback_reason="Falls back to PyTorch for non-CUDA, non-float32, non-contiguous, or broadcasted inputs.",
            dtypes=("float32",),
            shape_contract=(
                "binary contiguous CUDA float32 tensors with identical shapes"
                if is_binary
                else "binary contiguous CUDA float32 tensors with identical shapes and two outputs"
                if is_binary_pair
                else "unary contiguous CUDA float32 tensor"
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


def _spec_for_module(module_type: str) -> TileKernelSpec:
    if module_type in _HOST_ONLY_MODULES:
        return TileKernelSpec(
            name=module_type,
            kind="module",
            status="host_only",
            fallback_reason="Source/orchestration node; batches are materialized by the trainer, not by a compute kernel.",
            has_forward=False,
            has_backward=False,
            no_grad_reason="No device compute contract.",
        )
    if module_type in _DELEGATED_MODULES:
        return TileKernelSpec(
            name=module_type,
            kind="module",
            status="delegated",
            delegated_to=_DELEGATED_MODULES[module_type],
            fallback_reason="Delegates to another compiled graph; coverage is inherited from that graph's selected backend.",
            has_forward=False,
            has_backward=False,
        )
    if module_type in _TILE_MODULE_TYPES:
        return TileKernelSpec(
            name=module_type,
            kind="module",
            status="tile",
            fallback_reason="Falls back to PyTorch for non-CUDA, non-float32, non-contiguous, or unsupported broadcast contracts.",
            dtypes=("float32",),
            shape_contract="contiguous CUDA float32 tensors matching the existing PyTorch stage contract",
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
    if name == "adamw_step":
        return TileKernelSpec(
            name=name,
            kind="optimizer",
            status="tile",
            fallback_reason="Falls back to PyTorch for non-CUDA, non-float32, non-contiguous, empty, shape-mismatched, or invalid-step tensors.",
            dtypes=("float32",),
            shape_contract="same-shaped contiguous parameter, gradient, first-moment, and second-moment tensors; updates parameter and moments in-place",
            has_forward=True,
            has_backward=False,
            no_grad_reason="Optimizer/runtime update runs under no_grad.",
        )
    if name == "ema_update":
        return TileKernelSpec(
            name=name,
            kind="optimizer",
            status="tile",
            fallback_reason="Falls back to PyTorch for non-CUDA, non-float32, non-contiguous, empty, or shape-mismatched tensors.",
            dtypes=("float32",),
            shape_contract="same-shaped contiguous target/source parameter or buffer tensors; updates target in-place",
            has_forward=True,
            has_backward=False,
            no_grad_reason="Optimizer/runtime update runs under no_grad.",
        )
    if name == "gradient_accumulate":
        return TileKernelSpec(
            name=name,
            kind="optimizer",
            status="tile",
            fallback_reason="Falls back to PyTorch for non-CUDA, non-float32, non-contiguous, empty, or shape-mismatched tensors.",
            dtypes=("float32",),
            shape_contract="same-shaped contiguous accumulation-buffer and gradient tensors; updates buffer in-place",
            has_forward=True,
            has_backward=False,
            no_grad_reason="Optimizer/runtime accumulation runs under no_grad.",
        )
    if name == "gradient_clip_norm":
        return TileKernelSpec(
            name=name,
            kind="optimizer",
            status="tile",
            fallback_reason="Falls back to PyTorch for non-CUDA, non-float32, non-contiguous, or empty gradient tensors.",
            dtypes=("float32",),
            shape_contract="one or more contiguous gradient tensors; computes global L2 norm and scales tensors in-place",
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
            dtypes=("float32",),
            shape_contract="same-shaped parameter, gradient, and Muon momentum tensors; updates parameter and momentum in-place",
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
            dtypes=("float32",),
            shape_contract="same-shaped parameter/gradient plus AdamW and Muon state tensors; dispatches matrix parameters to Muon and others to AdamW",
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
