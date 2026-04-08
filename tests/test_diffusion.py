import torch
from neuralfn.config import build_diffllama_spec
from neuralfn.torch_templates import build_gpt_root_graph
from neuralfn.torch_backend import CompiledTorchGraph

def test_diffusion_template():
    spec = build_diffllama_spec(num_layers=2)
    graph = build_gpt_root_graph(name="diffusion_test", model_spec=spec)
    
    compiled = CompiledTorchGraph(graph)
    tokens = torch.randint(0, 256, (1, 16))

    loss = compiled(tokens)[0]
    assert loss.ndim == 0
    print("Diffusion template verified.")

if __name__ == "__main__":
    test_diffusion_template()
