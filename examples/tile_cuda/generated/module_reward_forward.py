"""Generated CUDA Tile SDK example for module:reward_forward."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:reward_forward")
if spec is None:
    raise SystemExit("missing registry spec: module:reward_forward")

print("module_reward_forward", spec.status, spec.shape_contract)
