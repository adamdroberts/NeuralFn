"""Generated CUDA Tile SDK example for module:semantic_moe_router."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:semantic_moe_router")
if spec is None:
    raise SystemExit("missing registry spec: module:semantic_moe_router")

print("module_semantic_moe_router", spec.status, spec.shape_contract)
