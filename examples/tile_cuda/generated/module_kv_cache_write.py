"""Generated CUDA Tile SDK example for module:kv_cache_write."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:kv_cache_write")
if spec is None:
    raise SystemExit("missing registry spec: module:kv_cache_write")

print("module_kv_cache_write", spec.status, spec.shape_contract)
