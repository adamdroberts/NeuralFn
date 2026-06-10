"""Generated CUDA Tile SDK example for module:kl_penalty."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:kl_penalty")
if spec is None:
    raise SystemExit("missing registry spec: module:kl_penalty")

print("module_kl_penalty", spec.status, spec.shape_contract)
