import torch
from neuralfn.graph import NeuronGraph, NeuronInstance, Edge
from neuralfn.builtins import BuiltinNeurons
from neuralfn.torch_backend import CompiledTorchGraph

def test_kv_quant():
    graph = NeuronGraph(name="kv_quant_test", runtime="torch")
    
    # Input K, V
    graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="k_in"))
    graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="v_in"))
    
    # Quant Pack
    graph.add_node(NeuronInstance(BuiltinNeurons.kv_quant_pack_module, instance_id="quant_pack"))
    graph.add_edge(Edge(src_node="k_in", src_port=0, dst_node="quant_pack", dst_port=0))
    graph.add_edge(Edge(src_node="v_in", src_port=0, dst_node="quant_pack", dst_port=1))
    
    # Quant Unpack
    unpack = BuiltinNeurons.kv_quant_unpack_module
    unpack.module_config = {"head_dim": 64}
    graph.add_node(NeuronInstance(unpack, instance_id="quant_unpack"))
    graph.add_edge(Edge(src_node="quant_pack", src_port=0, dst_node="quant_unpack", dst_port=0))
    
    # Outputs
    graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="k_out"))
    graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="v_out"))
    
    graph.add_edge(Edge(src_node="quant_unpack", src_port=0, dst_node="k_out", dst_port=0))
    graph.add_edge(Edge(src_node="quant_unpack", src_port=1, dst_node="v_out", dst_port=0))
    
    graph.input_node_ids = ["k_in", "v_in"]
    graph.output_node_ids = ["k_out", "v_out"]
    
    compiled = CompiledTorchGraph(graph)
    
    k = torch.randn(1, 4, 64)
    v = torch.randn(1, 4, 64)
    
    out_k, out_v = compiled(k, v)
    
    assert out_k.shape == (1, 4, 64)
    assert out_v.shape == (1, 4, 64)
    assert torch.allclose(out_k, k)
    assert torch.allclose(out_v, v)
    print("KV Quantization nodes verified successfully.")

if __name__ == "__main__":
    test_kv_quant()
