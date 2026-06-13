"""Optional CUDA Tile backend scaffolding for NeuralFn.

The package intentionally keeps exports lazy. Registry/config metadata is used
by CLI startup paths that must not import Torch; tensor kernels still import
their Torch-backed modules when the corresponding symbol is requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES: dict[str, str] = {
    "BINARY_OPS": ".autograd",
    "BINARY_PAIR_OPS": ".autograd",
    "TILE_FUNCTION_NAMES": ".autograd",
    "TILE_MODULE_NAMES": ".autograd",
    "UNARY_OPS": ".autograd",
    "TileCudaAutogradNotImplemented": ".autograd",
    "require_tile_cuda_kernel": ".autograd",
    "tile_binary": ".autograd",
    "tile_binary_pair": ".autograd",
    "tile_scalar_binary_module": ".autograd",
    "tile_scalar_ternary_module": ".autograd",
    "tile_scalar_unary_module": ".autograd",
    "tile_dyt_module": ".autograd",
    "tile_qk_gain_module": ".autograd",
    "tile_merge_heads_module": ".autograd",
    "tile_repeat_kv_module": ".autograd",
    "tile_reshape_heads_module": ".autograd",
    "tile_unary": ".autograd",
    "tile_vector_binary_module": ".autograd",
    "KernelBackend": ".config",
    "TileCudaConfig": ".config",
    "FP8_FORMATS": ".dtypes",
    "NVFP4_BLOCK_SIZE": ".dtypes",
    "NVFP4Tensor": ".dtypes",
    "dequantize_fp8_reference": ".dtypes",
    "dequantize_nvfp4_reference": ".dtypes",
    "quantize_dequantize_fp8_reference": ".dtypes",
    "quantize_dequantize_nvfp4_reference": ".dtypes",
    "quantize_fp8_reference": ".dtypes",
    "quantize_nvfp4_reference": ".dtypes",
    "TileCudaAuxLossAddStage": ".modules",
    "TileCudaBinaryFunctionStage": ".modules",
    "TileCudaBinaryPairFunctionStage": ".modules",
    "TileCudaDyTStage": ".modules",
    "TileCudaKLPenaltyStage": ".modules",
    "TileCudaLogitSoftcapStage": ".modules",
    "TileCudaLossScaleStage": ".modules",
    "TileCudaManifoldHyperConnectionStage": ".modules",
    "TileCudaMergeHeadsStage": ".modules",
    "TileCudaQKGainStage": ".modules",
    "TileCudaRepeatKVStage": ".modules",
    "TileCudaReshapeHeadsStage": ".modules",
    "TileCudaResidualAddStage": ".modules",
    "TileCudaResidualMixStage": ".modules",
    "TileCudaUnaryFunctionStage": ".modules",
    "build_tile_function_module": ".modules",
    "build_tile_module": ".modules",
    "tile_adamw_step": ".optimizer",
    "tile_adamw_step_batch": ".optimizer",
    "tile_adamw_step_reference": ".optimizer",
    "tile_ema_update": ".optimizer",
    "tile_ema_update_reference": ".optimizer",
    "tile_gradient_accumulate": ".optimizer",
    "tile_gradient_accumulate_reference": ".optimizer",
    "tile_gradient_clip_norm": ".optimizer",
    "tile_gradient_clip_norm_reference": ".optimizer",
    "tile_muon_newton_schulz": ".optimizer",
    "tile_muon_newton_schulz_reference": ".optimizer",
    "tile_muon_step": ".optimizer",
    "tile_muon_step_reference": ".optimizer",
    "tile_split_optimizer_step": ".optimizer",
    "tile_split_optimizer_step_reference": ".optimizer",
    "DEFAULT_TILE_KERNEL_REGISTRY": ".registry",
    "KernelCoverageReport": ".registry",
    "TileKernelRegistry": ".registry",
    "TileKernelSpec": ".registry",
    "build_default_registry": ".registry",
    "coverage_report": ".registry",
    "tile_kernel_inventory": ".registry",
    "tile_kernel_spec_for": ".registry",
    "TileCudaDiagnostics": ".runtime",
    "current_coverage_report": ".runtime",
    "is_tile_cuda_available": ".runtime",
    "load_tile_cuda_extension": ".runtime",
    "resolve_backend": ".runtime",
    "tile_cuda_diagnostics": ".runtime",
    "write_tile_cuda_report": ".runtime",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
