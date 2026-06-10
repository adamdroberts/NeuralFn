"""Generated CUDA Tile SDK example for module:broadcast_expert_routes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:broadcast_expert_routes")
if spec is None:
    raise SystemExit("missing registry spec: module:broadcast_expert_routes")

print("module_broadcast_expert_routes", spec.status, spec.shape_contract)
