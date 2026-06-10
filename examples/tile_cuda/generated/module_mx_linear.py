"""Generated CUDA Tile SDK example for module:mx_linear."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:mx_linear")
if spec is None:
    raise SystemExit("missing registry spec: module:mx_linear")

print("module_mx_linear", spec.status, spec.shape_contract)
