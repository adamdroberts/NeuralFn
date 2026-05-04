import os
import tempfile
import torch
import pytest

from neuralfn.config import build_llama_fast_spec
from neuralfn.torch_templates import build_gpt_root_graph
from neuralfn.inference import export_to_pt, import_from_pt, load_pt_checkpoint
from neuralfn.torch_backend import CompiledTorchGraph

def test_pt_export_import_roundtrip():
    # Build a small fast llama graph
    spec = build_llama_fast_spec(num_layers=1, model_dim=64, num_heads=2, num_kv_heads=1, vocab_size=100)
    raw_graph = build_gpt_root_graph(name="test_llama", model_spec=spec)
    # Simulate a real loaded graph where variants are cloned per instance
    from neuralfn.graph import NeuronGraph
    graph = NeuronGraph.from_dict(raw_graph.to_dict())

    # Compile and get initial output
    compiled_1 = CompiledTorchGraph(graph)
    compiled_1.eval()
    
    tokens = torch.randint(0, 100, (1, 16))
    targets = torch.randint(0, 100, (1, 16))
    
    with torch.no_grad():
        out_1 = compiled_1(tokens, targets)
        
    # Sync weights back to the graph before export
    compiled_1.sync_state_back(graph)

    # Export to .pt
    with tempfile.TemporaryDirectory() as tmpdir:
        pt_path = os.path.join(tmpdir, "model.pt")
        export_to_pt(graph, pt_path)
        
        assert os.path.exists(pt_path)
        state_dict, checkpoint_metadata = load_pt_checkpoint(pt_path)
        assert state_dict
        assert checkpoint_metadata.get("template_runtime") == "compile"
        
        # Load into a fresh graph (which should have random weights initially)
        raw_graph_2 = build_gpt_root_graph(name="test_llama", model_spec=spec)
        graph_2 = NeuronGraph.from_dict(raw_graph_2.to_dict())
        import_from_pt(graph_2, pt_path)
        
        compiled_2 = CompiledTorchGraph(graph_2)
        compiled_2.eval()
        
        with torch.no_grad():
            out_2 = compiled_2(tokens, targets)
            
    # Check outputs are exactly identical
    assert torch.allclose(out_1[0], out_2[0], atol=1e-6)
