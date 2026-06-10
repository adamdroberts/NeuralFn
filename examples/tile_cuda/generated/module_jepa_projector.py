"""Generated CUDA Tile SDK example for module:jepa_projector."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:jepa_projector")
if spec is None:
    raise SystemExit("missing registry spec: module:jepa_projector")

print("module_jepa_projector", spec.status, spec.shape_contract)
