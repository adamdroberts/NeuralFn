"""Generated CUDA Tile SDK example for module:native_sparse_attention."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:native_sparse_attention")
if spec is None:
    raise SystemExit("missing registry spec: module:native_sparse_attention")

print("module_native_sparse_attention", spec.status, spec.shape_contract)
