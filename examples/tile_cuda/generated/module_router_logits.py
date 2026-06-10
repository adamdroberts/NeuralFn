"""Generated CUDA Tile SDK example for module:router_logits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:router_logits")
if spec is None:
    raise SystemExit("missing registry spec: module:router_logits")

print("module_router_logits", spec.status, spec.shape_contract)
