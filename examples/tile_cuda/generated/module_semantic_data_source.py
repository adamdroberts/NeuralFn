"""Generated CUDA Tile SDK example for module:semantic_data_source."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:semantic_data_source")
if spec is None:
    raise SystemExit("missing registry spec: module:semantic_data_source")

print("module_semantic_data_source", spec.status, spec.shape_contract)
