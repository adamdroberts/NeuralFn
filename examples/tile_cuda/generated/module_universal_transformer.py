"""Generated CUDA Tile SDK example for module:universal_transformer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:universal_transformer")
if spec is None:
    raise SystemExit("missing registry spec: module:universal_transformer")

print("module_universal_transformer", spec.status, spec.shape_contract)
