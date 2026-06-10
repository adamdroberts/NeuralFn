"""Generated CUDA Tile SDK example for module:mask_scheduler."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:mask_scheduler")
if spec is None:
    raise SystemExit("missing registry spec: module:mask_scheduler")

print("module_mask_scheduler", spec.status, spec.shape_contract)
