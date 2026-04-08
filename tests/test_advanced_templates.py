import torch
from neuralfn.config import build_jamba_hybrid_spec, build_ternary_b158_spec
from neuralfn.torch_templates import build_gpt_root_graph
from neuralfn.torch_backend import CompiledTorchGraph

def test_jamba_template():
    spec = build_jamba_hybrid_spec(num_layers=4, experts=4, top_k=2)
    graph = build_gpt_root_graph(name="jamba_test", model_spec=spec)
    
    compiled = CompiledTorchGraph(graph)
    tokens = torch.randint(0, 256, (1, 16))
    targets = torch.randint(0, 256, (1, 16))
    
    loss = compiled(tokens, targets)[0]
    assert loss.ndim == 0
    print("Jamba template verified.")

def test_ternary_b158_template():
    spec = build_ternary_b158_spec(num_layers=1)
    graph = build_gpt_root_graph(name="ternary_test", model_spec=spec)
    
    compiled = CompiledTorchGraph(graph)
    tokens = torch.randint(0, 256, (1, 16))
    targets = torch.randint(0, 256, (1, 16))
    
    loss = compiled(tokens, targets)[0]
    assert loss.ndim == 0
    print("Ternary B1.58 template verified.")

if __name__ == "__main__":
    test_jamba_template()
    test_ternary_b158_template()
