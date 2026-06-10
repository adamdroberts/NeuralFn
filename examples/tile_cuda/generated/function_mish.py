"""Generated CUDA Tile SDK example for function:mish."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("function:mish")
if spec is None:
    raise SystemExit("missing registry spec: function:mish")

print("function_mish", spec.status, spec.shape_contract)
