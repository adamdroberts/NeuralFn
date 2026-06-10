"""Generated CUDA Tile SDK example for module:rotary_embedding."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:rotary_embedding")
if spec is None:
    raise SystemExit("missing registry spec: module:rotary_embedding")

print("module_rotary_embedding", spec.status, spec.shape_contract)
