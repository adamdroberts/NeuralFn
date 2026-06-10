from __future__ import annotations

import os

import pytest
import torch

from neuralfn.tile_cuda import TileCudaConfig, build_tile_function_module, load_tile_cuda_extension


def test_tile_cuda_gpu_smoke_add_matches_torch() -> None:
    if os.environ.get("NFN_TILE_CUDA_TEST") != "1":
        pytest.skip("set NFN_TILE_CUDA_TEST=1 to run GPU smoke coverage")
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda is not available")
    config = TileCudaConfig(backend="tile_cuda", strict=True, build_enabled=True)
    if load_tile_cuda_extension(config) is None:
        pytest.skip("CUDA Tile extension could not be built or loaded in this environment")

    stage = build_tile_function_module("add", config)
    lhs = torch.linspace(-1.0, 1.0, 257, device="cuda", dtype=torch.float32)
    rhs = torch.linspace(1.0, -1.0, 257, device="cuda", dtype=torch.float32)

    actual = stage(lhs, rhs)

    torch.testing.assert_close(actual, lhs + rhs)
