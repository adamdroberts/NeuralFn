"""Generated CUDA Tile SDK example for module:merge_heads."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:merge_heads")
if spec is None:
    raise SystemExit("missing registry spec: module:merge_heads")

print("module_merge_heads", spec.status, spec.shape_contract)
