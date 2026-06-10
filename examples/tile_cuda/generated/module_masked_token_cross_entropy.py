"""Generated CUDA Tile SDK example for module:masked_token_cross_entropy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:masked_token_cross_entropy")
if spec is None:
    raise SystemExit("missing registry spec: module:masked_token_cross_entropy")

print("module_masked_token_cross_entropy", spec.status, spec.shape_contract)
