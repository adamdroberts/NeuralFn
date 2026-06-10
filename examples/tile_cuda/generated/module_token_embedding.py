"""Generated CUDA Tile SDK example for module:token_embedding."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:token_embedding")
if spec is None:
    raise SystemExit("missing registry spec: module:token_embedding")

print("module_token_embedding", spec.status, spec.shape_contract)
