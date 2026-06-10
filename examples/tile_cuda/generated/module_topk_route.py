"""Generated CUDA Tile SDK example for module:topk_route."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:topk_route")
if spec is None:
    raise SystemExit("missing registry spec: module:topk_route")

print("module_topk_route", spec.status, spec.shape_contract)
