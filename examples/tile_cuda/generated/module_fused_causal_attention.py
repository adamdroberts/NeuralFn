"""Generated CUDA Tile SDK example for module:fused_causal_attention."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:fused_causal_attention")
if spec is None:
    raise SystemExit("missing registry spec: module:fused_causal_attention")

print("module_fused_causal_attention", spec.status, spec.shape_contract)
