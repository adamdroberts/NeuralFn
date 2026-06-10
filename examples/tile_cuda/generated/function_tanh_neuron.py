"""Generated CUDA Tile SDK example for function:tanh_neuron."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neuralfn.tile_cuda import DEFAULT_TILE_KERNEL_REGISTRY


spec = DEFAULT_TILE_KERNEL_REGISTRY.specs.get("function:tanh_neuron")
if spec is None:
    raise SystemExit("missing registry spec: function:tanh_neuron")

print("function_tanh_neuron", spec.status, spec.shape_contract)
