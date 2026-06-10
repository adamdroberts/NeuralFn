"""Generated CUDA Tile SDK example for optimizer:adamw_step."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("optimizer:adamw_step")
if spec is None:
    raise SystemExit("missing registry spec: optimizer:adamw_step")

print("optimizer_adamw_step", spec.status, spec.shape_contract)
