"""Generated CUDA Tile SDK example for module:value_head."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:value_head")
if spec is None:
    raise SystemExit("missing registry spec: module:value_head")

print("module_value_head", spec.status, spec.shape_contract)
