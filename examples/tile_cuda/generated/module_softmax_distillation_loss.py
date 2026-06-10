"""Generated CUDA Tile SDK example for module:softmax_distillation_loss."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("module:softmax_distillation_loss")
if spec is None:
    raise SystemExit("missing registry spec: module:softmax_distillation_loss")

print("module_softmax_distillation_loss", spec.status, spec.shape_contract)
