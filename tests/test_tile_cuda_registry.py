from __future__ import annotations

from neuralfn.builtins import BuiltinNeurons
from neuralfn.tile_cuda import (
    TileCudaConfig,
    build_default_registry,
    coverage_report,
    resolve_backend,
    tile_cuda_diagnostics,
)
from neuralfn.tile_cuda.registry import (
    build_function_dispatch_inventory,
    build_module_dispatch_inventory,
    builtin_function_inventory,
    builtin_module_inventory,
    optimizer_runtime_inventory,
    tile_kernel_key,
    tile_kernel_inventory,
)
from neuralfn.torch_backend import TorchTrainConfig


def test_tile_cuda_inventory_tracks_builtin_and_dispatch_surfaces() -> None:
    builtin_functions = {
        tile_kernel_key("function", neuron_def.name)
        for neuron_def in BuiltinNeurons.all()
        if neuron_def.kind == "function"
    }
    builtin_modules = {
        tile_kernel_key("module", neuron_def.module_type)
        for neuron_def in BuiltinNeurons.all()
        if neuron_def.kind == "module"
    }

    assert set(builtin_function_inventory()) == builtin_functions
    assert set(builtin_module_inventory()) == builtin_modules
    assert set(build_module_dispatch_inventory()) == builtin_modules
    assert set(build_function_dispatch_inventory()) <= set(tile_kernel_inventory())
    assert set(optimizer_runtime_inventory()) <= set(tile_kernel_inventory())
    assert tile_kernel_key("function", "gelu") in tile_kernel_inventory()
    assert tile_kernel_key("module", "gelu") in tile_kernel_inventory()


def test_default_tile_cuda_registry_accounts_for_every_inventory_entry() -> None:
    registry = build_default_registry()
    report = registry.coverage_report()

    assert report.complete
    assert report.missing == ()
    assert report.accounted == report.total_inventory
    assert report.by_status["tile"] > 0
    assert report.by_status.get("torch_fallback", 0) == 0
    assert report.by_status["host_only"] > 0
    assert report.by_status["delegated"] > 0
    assert set(report.by_dtype) == {"float32", "float16", "float8_e4m3fn", "float8_e5m2", "nvfp4"}
    assert report.by_dtype["float32"]["supported"] > report.by_dtype["float16"]["supported"]
    assert report.by_dtype["float16"]["supported"] > report.by_dtype["float8_e4m3fn"]["supported"]
    assert report.by_dtype["float8_e4m3fn"]["supported"] > report.by_dtype["nvfp4"]["supported"]
    assert report.to_dict()["by_dtype"]["nvfp4"]["supported"] > 0


def test_default_tile_cuda_registry_entries_do_not_claim_fake_tile_kernels() -> None:
    report = coverage_report()
    specs = report.specs

    assert specs
    assert coverage_report().by_status["tile"] >= 129
    for spec in specs:
        if spec.status == "tile":
            assert spec.has_forward
        elif spec.status in {"torch_fallback", "host_only", "delegated", "planned"}:
            assert spec.fallback_reason or spec.delegated_to or spec.no_grad_reason


def test_tile_cuda_registry_advertises_fp16_only_for_verified_simple_modules() -> None:
    registry = build_default_registry()
    fp16_modules = {
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
    fp8_modules = {
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
    nvfp4_modules = {
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

    for module_type in fp16_modules:
        spec = registry.get(module_type, kind="module")
        assert spec is not None
        if module_type in fp8_modules:
            assert set(spec.dtypes) >= {"float32", "float16", "float8_e4m3fn", "float8_e5m2"}
            assert spec.dtype_support["float8_e4m3fn"] == "supported"
            assert spec.dtype_support["float8_e5m2"] == "supported"
        else:
            assert spec.dtypes == ("float32", "float16")
        if module_type in nvfp4_modules:
            assert spec.dtype_support["nvfp4"] == "supported"
            assert "nvfp4" in spec.dtypes
            assert "NVFP4Tensor activations" in spec.shape_contract
        assert "remain float32" in spec.shape_contract


def test_tile_cuda_registry_exposes_dtype_support_matrix_for_tracked_dtypes() -> None:
    registry = build_default_registry()
    tracked = {"float32", "float16", "float8_e4m3fn", "float8_e5m2", "nvfp4"}

    for spec in registry.canonical_specs():
        assert set(spec.dtype_support) == tracked
        assert set(spec.to_dict()["dtype_support"]) == tracked
        for dtype in spec.dtypes:
            if dtype in tracked:
                assert spec.dtype_support[dtype] == "supported"

    routed = registry.get("routed_attention_experts", kind="module")
    assert routed is not None
    assert routed.dtype_support["float16"] == "supported"
    assert routed.dtype_support["float8_e4m3fn"] == "supported"
    assert routed.dtype_support["float8_e5m2"] == "supported"
    assert routed.dtype_support["nvfp4"] == "supported"
    assert "NVFP4Tensor activations" in routed.shape_contract

    direct_linear = registry.get("linear", kind="module")
    assert direct_linear is not None
    assert direct_linear.dtype_support["nvfp4"] == "supported"

    semantic_projector = registry.get("semantic_projector", kind="module")
    assert semantic_projector is not None
    assert semantic_projector.dtypes == ("float32",)
    assert "argmax-derived" in semantic_projector.dtype_support["float16"]
    assert "categorical contract" in semantic_projector.dtype_support["nvfp4"]
    assert "NVFP4Tensor activations" in direct_linear.shape_contract

    dropout = registry.get("dropout", kind="module")
    assert dropout is not None
    assert dropout.dtype_support["float16"] == "supported"
    assert "Stochastic or mask-producing" in dropout.dtype_support["float8_e4m3fn"]

    token_ce = registry.get("token_cross_entropy", kind="module")
    assert token_ce is not None
    assert "Loss/reduction" in token_ce.dtype_support["float8_e4m3fn"]
    assert "NVFP4 packed" in token_ce.dtype_support["nvfp4"]

    semantic_hasher = registry.get("semantic_hasher", kind="module")
    assert semantic_hasher is not None
    assert "integer, hash" in semantic_hasher.dtype_support["float8_e5m2"]

    random_timesteps = registry.get("random_timesteps", kind="module")
    assert random_timesteps is not None
    assert "Stochastic or mask-producing" in random_timesteps.dtype_support["float8_e4m3fn"]

    dataset_source = registry.get("dataset_source", kind="module")
    assert dataset_source is not None
    assert "do not pass through editor nodes" in dataset_source.dtype_support["float8_e4m3fn"]

    reference_forward = registry.get("reference_forward", kind="module")
    assert reference_forward is not None
    assert "inherit dtype support" in reference_forward.dtype_support["nvfp4"]


def test_tile_cuda_registry_advertises_fp16_for_verified_optimizer_runtime_helpers() -> None:
    registry = build_default_registry()
    fp16_optimizers = {"adamw_step", "ema_update", "gradient_accumulate", "gradient_clip_norm", "muon_step", "split_optimizer_step"}

    for name in fp16_optimizers:
        spec = registry.get(name, kind="optimizer")
        assert spec is not None
        assert spec.dtypes == ("float32", "float16")

    muon = registry.get("muon_step", kind="optimizer")
    assert muon is not None
    assert muon.dtypes == ("float32", "float16")

    newton = registry.get("muon_newton_schulz", kind="optimizer")
    assert newton is not None
    assert newton.dtypes == ("float32",)


def test_tile_cuda_diagnostics_and_backend_resolution_are_cpu_safe() -> None:
    diagnostics = tile_cuda_diagnostics(TileCudaConfig(backend="auto"))
    payload = diagnostics.to_dict()

    assert "nvcc_path" in payload
    assert "cuda_tile_header" in payload
    assert resolve_backend(TileCudaConfig(backend="torch", strict=True)) == "torch"


def test_torch_train_config_exposes_tile_cuda_selection_fields() -> None:
    config = TorchTrainConfig()

    assert config.kernel_backend == "auto"
    assert config.tile_cuda_strict is False
    assert config.tile_cuda_report_path is None
