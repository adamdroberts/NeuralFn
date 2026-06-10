"""Generated CUDA Tile SDK example for module:scaled_dot_product_attention."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:scaled_dot_product_attention")
if spec is None:
    raise SystemExit("missing registry spec: module:scaled_dot_product_attention")

print("module_scaled_dot_product_attention", spec.status, spec.shape_contract)
