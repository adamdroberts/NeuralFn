"""Generated CUDA Tile SDK example for function:leaky_relu."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("function:leaky_relu")
if spec is None:
    raise SystemExit("missing registry spec: function:leaky_relu")

print("function_leaky_relu", spec.status, spec.shape_contract)
