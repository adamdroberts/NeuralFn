"""Generated CUDA Tile SDK example for module:routed_attention_experts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:routed_attention_experts")
if spec is None:
    raise SystemExit("missing registry spec: module:routed_attention_experts")

print("module_routed_attention_experts", spec.status, spec.shape_contract)
