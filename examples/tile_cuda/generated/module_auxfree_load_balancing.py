"""Generated CUDA Tile SDK example for module:auxfree_load_balancing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:auxfree_load_balancing")
if spec is None:
    raise SystemExit("missing registry spec: module:auxfree_load_balancing")

print("module_auxfree_load_balancing", spec.status, spec.shape_contract)
