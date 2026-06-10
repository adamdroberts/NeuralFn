"""Generated CUDA Tile SDK example for module:qk_gain."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:qk_gain")
if spec is None:
    raise SystemExit("missing registry spec: module:qk_gain")

print("module_qk_gain", spec.status, spec.shape_contract)
