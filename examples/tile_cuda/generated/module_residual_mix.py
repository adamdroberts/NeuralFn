"""Generated CUDA Tile SDK example for module:residual_mix."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:residual_mix")
if spec is None:
    raise SystemExit("missing registry spec: module:residual_mix")

print("module_residual_mix", spec.status, spec.shape_contract)
