"""Generated CUDA Tile SDK example for module:differential_attention."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:differential_attention")
if spec is None:
    raise SystemExit("missing registry spec: module:differential_attention")

print("module_differential_attention", spec.status, spec.shape_contract)
