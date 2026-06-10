"""Generated CUDA Tile SDK example for function:hard_swish."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("function:hard_swish")
if spec is None:
    raise SystemExit("missing registry spec: function:hard_swish")

print("function_hard_swish", spec.status, spec.shape_contract)
