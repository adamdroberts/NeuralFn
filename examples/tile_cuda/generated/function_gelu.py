"""Generated CUDA Tile SDK example for function:gelu."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("function:gelu")
if spec is None:
    raise SystemExit("missing registry spec: function:gelu")

print("function_gelu", spec.status, spec.shape_contract)
