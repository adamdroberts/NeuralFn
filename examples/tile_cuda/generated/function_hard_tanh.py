"""Generated CUDA Tile SDK example for function:hard_tanh."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("function:hard_tanh")
if spec is None:
    raise SystemExit("missing registry spec: function:hard_tanh")

print("function_hard_tanh", spec.status, spec.shape_contract)
