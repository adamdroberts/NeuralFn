import torch
from neuralfn.config import build_llama_fast_spec
from neuralfn.torch_templates import build_gpt_root_graph
from neuralfn.torch_backend import CompiledTorchGraph

def test_randmap_adapter():
    # Enable adapter with dim 32
    spec = build_llama_fast_spec(num_layers=1, model_dim=128)
    spec.block_spec.adapter_dim = 32
    
    graph = build_gpt_root_graph(name="adapter_test", model_spec=spec)
    graph.resolve_variant_library()
    
    model_node = graph.nodes['model']
    model_stage = model_node.neuron_def.subgraph
    assert model_stage is not None
    
    block_0 = model_stage.nodes['block_0']
    block_graph = block_0.neuron_def.subgraph
    assert block_graph is not None, "block_0.neuron_def.subgraph should not be None after resolve_variant_library"
    
    # Check if adapter nodes exist in the attention component of block_0
    # The attention component is also a variant in model_stage.variant_library (moved to root)
    # Actually, in build_decoder_block_graph, attention is a linked variant.
    
    attn_node = block_graph.nodes['attention']
    attn_graph = attn_node.neuron_def.subgraph
    assert attn_graph is not None
    
    adapter_nodes = [nid for nid in attn_graph.nodes.keys() if 'adapter' in nid]
    print(f"Found adapter nodes in attention: {adapter_nodes}")
    assert len(adapter_nodes) >= 4 # q, k, v, out
    
    compiled = CompiledTorchGraph(graph)
    tokens = torch.randint(0, 256, (1, 16))
    targets = torch.randint(0, 256, (1, 16))
    
    loss = compiled(tokens, targets)[0]
    assert loss.ndim == 0
    print("Random-map adapter verified.")

if __name__ == "__main__":
    test_randmap_adapter()
