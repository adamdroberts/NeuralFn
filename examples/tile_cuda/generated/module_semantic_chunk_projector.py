"""Generated CUDA Tile SDK example for module:semantic_chunk_projector."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:semantic_chunk_projector")
if spec is None:
    raise SystemExit("missing registry spec: module:semantic_chunk_projector")

print("module_semantic_chunk_projector", spec.status, spec.shape_contract)
