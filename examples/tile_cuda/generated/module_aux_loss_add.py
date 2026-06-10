"""Generated CUDA Tile SDK example for module:aux_loss_add."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:aux_loss_add")
if spec is None:
    raise SystemExit("missing registry spec: module:aux_loss_add")

print("module_aux_loss_add", spec.status, spec.shape_contract)
