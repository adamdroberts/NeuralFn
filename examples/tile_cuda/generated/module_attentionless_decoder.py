"""Generated CUDA Tile SDK example for module:attentionless_decoder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:attentionless_decoder")
if spec is None:
    raise SystemExit("missing registry spec: module:attentionless_decoder")

print("module_attentionless_decoder", spec.status, spec.shape_contract)
