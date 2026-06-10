from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import torch

from .config import TileCudaConfig
from .registry import KernelCoverageReport, coverage_report

_EXTENSION_MODULE: Any | None = None
_EXTENSION_LOAD_ERROR = ""
_EXTENSION_LOAD_ATTEMPTED = False


@dataclass(frozen=True)
class TileCudaDiagnostics:
    nvcc_path: str
    cuda_version: str
    cuda_tile_header: str
    torch_cuda_available: bool
    device_name: str
    compute_capability: str
    build_enabled: bool
    extension_loaded: bool = False
    extension_error: str = ""

    @property
    def toolchain_available(self) -> bool:
        return bool(
            self.nvcc_path
            and self.cuda_tile_header
            and _version_at_least(self.cuda_version, (13, 3))
        )

    @property
    def runtime_available(self) -> bool:
        return (
            self.toolchain_available
            and self.torch_cuda_available
            and bool(self.compute_capability)
            and (self.extension_loaded or self.build_enabled)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nvcc_path": self.nvcc_path,
            "cuda_version": self.cuda_version,
            "cuda_tile_header": self.cuda_tile_header,
            "torch_cuda_available": self.torch_cuda_available,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "build_enabled": self.build_enabled,
            "extension_loaded": self.extension_loaded,
            "extension_error": self.extension_error,
            "toolchain_available": self.toolchain_available,
            "runtime_available": self.runtime_available,
        }


def _nvcc_version(nvcc_path: str) -> str:
    try:
        proc = subprocess.run(
            [nvcc_path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    match = re.search(r"release\s+(\d+\.\d+)", proc.stdout + proc.stderr)
    return match.group(1) if match else ""


def _version_at_least(version: str, minimum: tuple[int, int]) -> bool:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if not match:
        return False
    return (int(match.group(1)), int(match.group(2))) >= minimum


def _find_cuda_tile_header() -> str:
    candidates: list[Path] = []
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidates.append(Path(cuda_home) / "include" / "cuda_tile.h")
        candidates.extend(Path(cuda_home).glob("targets/*/include/cuda_tile.h"))
    candidates.extend(Path("/usr/local").glob("cuda*/targets/*/include/cuda_tile.h"))
    candidates.extend(Path("/usr/local").glob("cuda*/include/cuda_tile.h"))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def tile_cuda_diagnostics(config: TileCudaConfig | None = None) -> TileCudaDiagnostics:
    cfg = config or TileCudaConfig()
    nvcc_path = shutil.which("nvcc") or ""
    cuda_available = bool(torch.cuda.is_available())
    device_name = ""
    compute_capability = ""
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            compute_capability = f"{major}.{minor}"
        except Exception:
            cuda_available = False
    return TileCudaDiagnostics(
        nvcc_path=nvcc_path,
        cuda_version=_nvcc_version(nvcc_path) if nvcc_path else "",
        cuda_tile_header=_find_cuda_tile_header(),
        torch_cuda_available=cuda_available,
        device_name=device_name,
        compute_capability=compute_capability,
        build_enabled=bool(cfg.build_is_enabled),
        extension_loaded=_EXTENSION_MODULE is not None,
        extension_error=_EXTENSION_LOAD_ERROR,
    )


def is_tile_cuda_available(config: TileCudaConfig | None = None) -> bool:
    return tile_cuda_diagnostics(config).runtime_available


def resolve_backend(config: TileCudaConfig | None = None) -> str:
    cfg = config or TileCudaConfig()
    backend = cfg.normalized_backend()
    if backend == "torch":
        return "torch"
    if backend == "tile_cuda" and not is_tile_cuda_available(cfg):
        if cfg.strict:
            raise RuntimeError("CUDA Tile backend was requested in strict mode, but it is not available")
        return "torch"
    if backend == "auto":
        return "tile_cuda" if is_tile_cuda_available(cfg) else "torch"
    return backend


def current_coverage_report() -> KernelCoverageReport:
    return coverage_report()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _extension_sources() -> list[str]:
    source_dir = _repo_root() / "neuralfn" / "csrc" / "tile_cuda"
    return [str(source_dir / "bindings.cpp"), str(source_dir / "kernels.cu")]


def _resolved_arch_flag(config: TileCudaConfig, diagnostics: TileCudaDiagnostics) -> str:
    arch = config.resolved_arch
    if arch:
        return arch if arch.startswith("sm_") else f"sm_{arch.replace('.', '')}"
    if diagnostics.compute_capability:
        return f"sm_{diagnostics.compute_capability.replace('.', '')}"
    return "sm_80"


def load_tile_cuda_extension(config: TileCudaConfig | None = None) -> Any | None:
    """Load or build the optional CUDA Tile PyTorch extension.

    The build path is intentionally opt-in with ``NFN_TILE_CUDA_BUILD=1`` or
    ``TileCudaConfig(build_enabled=True)`` so importing NeuralFn remains safe on
    CPU-only hosts.
    """

    global _EXTENSION_LOAD_ATTEMPTED, _EXTENSION_LOAD_ERROR, _EXTENSION_MODULE
    if _EXTENSION_MODULE is not None:
        return _EXTENSION_MODULE
    cfg = config or TileCudaConfig()
    diagnostics = tile_cuda_diagnostics(cfg)
    if not cfg.build_is_enabled:
        _EXTENSION_LOAD_ERROR = "Set NFN_TILE_CUDA_BUILD=1 or TileCudaConfig(build_enabled=True) to build the optional extension."
        return None
    if not diagnostics.toolchain_available:
        _EXTENSION_LOAD_ERROR = "CUDA Tile toolchain is unavailable; CUDA Toolkit 13.3+, nvcc, and cuda_tile.h are required."
        return None
    if not diagnostics.torch_cuda_available:
        _EXTENSION_LOAD_ERROR = "torch.cuda is not available in this process."
        return None
    if _EXTENSION_LOAD_ATTEMPTED:
        return None
    _EXTENSION_LOAD_ATTEMPTED = True
    try:
        from torch.utils.cpp_extension import load

        build_dir = Path(os.environ.get("NFN_TILE_CUDA_BUILD_DIR", "/tmp/neuralfn_tile_cuda_extension"))
        build_dir.mkdir(parents=True, exist_ok=True)
        arch_flag = _resolved_arch_flag(cfg, diagnostics)
        _EXTENSION_MODULE = load(
            name="neuralfn_tile_cuda_ext",
            sources=_extension_sources(),
            build_directory=str(build_dir),
            extra_cflags=["-std=c++20"],
            extra_cuda_cflags=["-std=c++20", "--enable-tile", f"-arch={arch_flag}"],
            verbose=bool(os.environ.get("NFN_TILE_CUDA_VERBOSE")),
        )
        _EXTENSION_LOAD_ERROR = ""
    except Exception as exc:  # pragma: no cover - depends on local CUDA compiler state.
        _EXTENSION_LOAD_ERROR = str(exc)
        _EXTENSION_MODULE = None
    return _EXTENSION_MODULE


def write_tile_cuda_report(path: str | Path, config: TileCudaConfig | None = None) -> None:
    payload = {
        "diagnostics": tile_cuda_diagnostics(config).to_dict(),
        "coverage": coverage_report().to_dict(),
    }
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
