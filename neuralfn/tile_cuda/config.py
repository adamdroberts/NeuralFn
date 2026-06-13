from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal


KernelBackend = Literal["auto", "torch", "tile_cuda"]


@dataclass(frozen=True)
class TileCudaConfig:
    """Runtime selection for the optional CUDA Tile backend."""

    backend: KernelBackend = "auto"
    strict: bool = False
    report_path: str | None = None
    build_enabled: bool = False
    arch: str | None = None

    def normalized_backend(self) -> KernelBackend:
        value = str(self.backend).strip().lower().replace("-", "_")
        if value not in {"auto", "torch", "tile_cuda"}:
            raise ValueError(f"Unsupported kernel backend: {self.backend!r}")
        return value  # type: ignore[return-value]

    @property
    def build_is_enabled(self) -> bool:
        value = os.environ.get("NFN_TILE_CUDA_BUILD", "")
        return bool(
            self.build_enabled
            or self.normalized_backend() == "tile_cuda"
            or value.strip().lower() in {"1", "true", "yes", "on"}
        )

    @property
    def resolved_arch(self) -> str | None:
        return self.arch or os.environ.get("NFN_TILE_CUDA_ARCH") or None

    @property
    def report_file(self) -> Path | None:
        return Path(self.report_path) if self.report_path else None
