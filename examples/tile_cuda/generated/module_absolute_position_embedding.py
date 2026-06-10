"""Generated CUDA Tile SDK example for module:absolute_position_embedding."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:absolute_position_embedding")
if spec is None:
    raise SystemExit("missing registry spec: module:absolute_position_embedding")

print("module_absolute_position_embedding", spec.status, spec.shape_contract)
