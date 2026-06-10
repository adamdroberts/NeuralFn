"""Generated CUDA Tile SDK example for module:randmap_adapter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:randmap_adapter")
if spec is None:
    raise SystemExit("missing registry spec: module:randmap_adapter")

print("module_randmap_adapter", spec.status, spec.shape_contract)
