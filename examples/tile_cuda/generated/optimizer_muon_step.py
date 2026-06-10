"""Generated CUDA Tile SDK example for optimizer:muon_step."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("optimizer:muon_step")
if spec is None:
    raise SystemExit("missing registry spec: optimizer:muon_step")

print("optimizer_muon_step", spec.status, spec.shape_contract)
