"""NeuralFn CUDA Tile example."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cli"))

from neuralfn.torch_backend import CompiledTorchGraph
from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config


spec = build_model_spec_from_config({"preset": "llm_jepa", "vocab_size": 128, "num_layers": 1, "model_dim": 32, "num_heads": 4}, preview_defaults=True)
graph = build_gpt_root_graph(name="llm_jepa_tile_cuda_smoke", model_spec=spec)
compiled = CompiledTorchGraph(graph, kernel_backend="tile_cuda")
print(f"prepared {graph.name} with {len(graph.nodes)} nodes via {compiled.resolved_kernel_backend}")
