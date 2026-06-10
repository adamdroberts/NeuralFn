"""Generated CUDA Tile SDK example for module:kv_pca_encode."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:kv_pca_encode")
if spec is None:
    raise SystemExit("missing registry spec: module:kv_pca_encode")

print("module_kv_pca_encode", spec.status, spec.shape_contract)
