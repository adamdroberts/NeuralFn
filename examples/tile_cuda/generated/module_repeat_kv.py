"""Generated CUDA Tile SDK example for module:repeat_kv."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:repeat_kv")
if spec is None:
    raise SystemExit("missing registry spec: module:repeat_kv")

print("module_repeat_kv", spec.status, spec.shape_contract)
