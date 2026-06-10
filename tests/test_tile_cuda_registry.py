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
