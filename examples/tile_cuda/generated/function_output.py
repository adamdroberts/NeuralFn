"""Generated CUDA Tile SDK example for function:output."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("function:output")
if spec is None:
    raise SystemExit("missing registry spec: function:output")

print("function_output", spec.status, spec.shape_contract)
