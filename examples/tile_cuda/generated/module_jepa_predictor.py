"""Generated CUDA Tile SDK example for module:jepa_predictor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:jepa_predictor")
if spec is None:
    raise SystemExit("missing registry spec: module:jepa_predictor")

print("module_jepa_predictor", spec.status, spec.shape_contract)
