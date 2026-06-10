"""Generated CUDA Tile SDK example for function:input."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("function:input")
if spec is None:
    raise SystemExit("missing registry spec: function:input")

print("function_input", spec.status, spec.shape_contract)
