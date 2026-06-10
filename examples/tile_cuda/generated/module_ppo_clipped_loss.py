"""Generated CUDA Tile SDK example for module:ppo_clipped_loss."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:ppo_clipped_loss")
if spec is None:
    raise SystemExit("missing registry spec: module:ppo_clipped_loss")

print("module_ppo_clipped_loss", spec.status, spec.shape_contract)
