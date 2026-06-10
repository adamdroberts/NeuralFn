"""Generated CUDA Tile SDK example for function:prelu."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("function:prelu")
if spec is None:
    raise SystemExit("missing registry spec: function:prelu")

print("function_prelu", spec.status, spec.shape_contract)
