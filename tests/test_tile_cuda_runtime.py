from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from torch.utils import cpp_extension

from neuralfn.tile_cuda import runtime
from neuralfn.tile_cuda.config import TileCudaConfig


def test_runtime_loader_supplies_native_train_include_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    extension = object()

    def fake_load(**kwargs: Any) -> object:
        captured.update(kwargs)
        return extension

    diagnostics = runtime.TileCudaDiagnostics(
        nvcc_path="/fixture/cuda/bin/nvcc",
        cuda_version="13.3",
        cuda_tile_header="/fixture/cuda/include/cuda_tile.h",
        torch_cuda_available=True,
        device_name="fixture GPU",
        compute_capability="12.0",
        build_enabled=True,
    )
    monkeypatch.setattr(runtime, "_EXTENSION_MODULE", None)
    monkeypatch.setattr(runtime, "_EXTENSION_LOAD_ERROR", "")
    monkeypatch.setattr(runtime, "_EXTENSION_LOAD_ATTEMPTED", False)
    monkeypatch.setattr(runtime, "tile_cuda_diagnostics", lambda _config: diagnostics)
    monkeypatch.setattr(cpp_extension, "load", fake_load)
    monkeypatch.setenv("NFN_TILE_CUDA_BUILD_DIR", str(tmp_path / "build"))

    loaded = runtime.load_tile_cuda_extension(
        TileCudaConfig(build_enabled=True, arch="sm_120")
    )

    assert loaded is extension
    include_paths = [Path(value).resolve() for value in captured["extra_include_paths"]]
    expected_include_path = (
        Path(runtime.__file__).resolve().parents[1] / "csrc" / "native_train"
    )
    assert include_paths == [expected_include_path]
    assert (include_paths[0] / "tile_ops.h").is_file()

    kernel_source = next(
        Path(value) for value in captured["sources"] if Path(value).name == "kernels.cu"
    )
    assert kernel_source.read_text().splitlines()[0] == '#include "tile_ops.h"'
