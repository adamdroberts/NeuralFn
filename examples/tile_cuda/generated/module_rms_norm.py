"""Generated CUDA Tile SDK example for module:rms_norm."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:rms_norm")
if spec is None:
    raise SystemExit("missing registry spec: module:rms_norm")

print("module_rms_norm", spec.status, spec.shape_contract)
