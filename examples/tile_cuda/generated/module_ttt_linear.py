"""Generated CUDA Tile SDK example for module:ttt_linear."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:ttt_linear")
if spec is None:
    raise SystemExit("missing registry spec: module:ttt_linear")

print("module_ttt_linear", spec.status, spec.shape_contract)
