"""Generated CUDA Tile SDK example for optimizer:muon_newton_schulz."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("optimizer:muon_newton_schulz")
if spec is None:
    raise SystemExit("missing registry spec: optimizer:muon_newton_schulz")

print("optimizer_muon_newton_schulz", spec.status, spec.shape_contract)
