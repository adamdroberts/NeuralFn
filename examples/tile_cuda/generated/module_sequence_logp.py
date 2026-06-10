"""Generated CUDA Tile SDK example for module:sequence_logp."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:sequence_logp")
if spec is None:
    raise SystemExit("missing registry spec: module:sequence_logp")

print("module_sequence_logp", spec.status, spec.shape_contract)
