"""Generated CUDA Tile SDK example for module:gae_compute."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:gae_compute")
if spec is None:
    raise SystemExit("missing registry spec: module:gae_compute")

print("module_gae_compute", spec.status, spec.shape_contract)
