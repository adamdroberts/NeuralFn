import os
import tempfile
import torch

from neuralfn.config import build_llama_fast_spec
from neuralfn.torch_templates import build_gpt_root_graph
from neuralfn.inference import export_to_pt, import_from_pt
from neuralfn.torch_backend import CompiledTorchGraph

spec = build_llama_fast_spec(num_layers=1, model_dim=64, num_heads=2, num_kv_heads=1, vocab_size=100)
graph = build_gpt_root_graph(name="test_llama", model_spec=spec)

compiled_1 = CompiledTorchGraph(graph)
compiled_1.eval()

tokens = torch.randint(0, 100, (1, 16))
targets = torch.randint(0, 100, (1, 16))

with torch.no_grad():
    out_1 = compiled_1(tokens, targets)
    
compiled_1.sync_state_back(graph)

with tempfile.TemporaryDirectory() as tmpdir:
    pt_path = os.path.join(tmpdir, "model.pt")
    export_to_pt(graph, pt_path)
    
    graph_2 = build_gpt_root_graph(name="test_llama", model_spec=spec)
    import_from_pt(graph_2, pt_path)
    
    compiled_2 = CompiledTorchGraph(graph_2)
    compiled_2.eval()
    
    with torch.no_grad():
        out_2 = compiled_2(tokens, targets)

print("out_1 sum:", out_1[0].sum().item())
print("out_2 sum:", out_2[0].sum().item())

for name, param in compiled_1.named_parameters():
    param2 = dict(compiled_2.named_parameters())[name]
    diff = (param - param2).abs().max().item()
    if diff > 0:
        print(f"Diff in {name}: {diff}")
