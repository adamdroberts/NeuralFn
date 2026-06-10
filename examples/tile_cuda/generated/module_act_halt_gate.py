"""Generated CUDA Tile SDK example for module:act_halt_gate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:act_halt_gate")
if spec is None:
    raise SystemExit("missing registry spec: module:act_halt_gate")

print("module_act_halt_gate", spec.status, spec.shape_contract)
