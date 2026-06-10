"""Generated CUDA Tile SDK example for module:mlp_relu2."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:mlp_relu2")
if spec is None:
    raise SystemExit("missing registry spec: module:mlp_relu2")

print("module_mlp_relu2", spec.status, spec.shape_contract)
