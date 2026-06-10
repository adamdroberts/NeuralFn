"""Generated CUDA Tile SDK example for module:loss_scale."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:loss_scale")
if spec is None:
    raise SystemExit("missing registry spec: module:loss_scale")

print("module_loss_scale", spec.status, spec.shape_contract)
