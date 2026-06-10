"""Generated CUDA Tile SDK example for module:kv_quant_pack."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:kv_quant_pack")
if spec is None:
    raise SystemExit("missing registry spec: module:kv_quant_pack")

print("module_kv_quant_pack", spec.status, spec.shape_contract)
