import torch
from neuralfn.graph import NeuronGraph, NeuronInstance, Edge
from neuralfn.builtins import BuiltinNeurons
from neuralfn.torch_backend import CompiledTorchGraph

def test_kv_pca():
    graph = NeuronGraph(name="kv_pca_test", runtime="torch")
    
    # Input K, V
    graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="k_in"))
    graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="v_in"))
    
    # PCA Encode
    encode = BuiltinNeurons.kv_pca_encode_module
    encode.module_config = {"head_dim": 64, "compressed_dim": 16}
    graph.add_node(NeuronInstance(encode, instance_id="pca_encode"))
    
    graph.add_edge(Edge(src_node="k_in", src_port=0, dst_node="pca_encode", dst_port=0))
    graph.add_edge(Edge(src_node="v_in", src_port=0, dst_node="pca_encode", dst_port=1))
    
    # PCA Decode
    decode = BuiltinNeurons.kv_pca_decode_module
    decode.module_config = {"head_dim": 64, "compressed_dim": 16}
    graph.add_node(NeuronInstance(decode, instance_id="pca_decode"))
    
    graph.add_edge(Edge(src_node="pca_encode", src_port=0, dst_node="pca_decode", dst_port=0))
    graph.add_edge(Edge(src_node="pca_encode", src_port=1, dst_node="pca_decode", dst_port=1))
    
    # Outputs
    graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="k_out"))
    graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="v_out"))
    
    graph.add_edge(Edge(src_node="pca_decode", src_port=0, dst_node="k_out", dst_port=0))
    graph.add_edge(Edge(src_node="pca_decode", src_port=1, dst_node="v_out", dst_port=0))
    
    graph.input_node_ids = ["k_in", "v_in"]
    graph.output_node_ids = ["k_out", "v_out"]
    
    compiled = CompiledTorchGraph(graph)
    
    k = torch.randn(1, 4, 64)
    v = torch.randn(1, 4, 64)
    
    out_k, out_v = compiled(k, v)
    
    assert out_k.shape == (1, 4, 64)
    assert out_v.shape == (1, 4, 64)
    print("KV PCA nodes verified successfully.")

if __name__ == "__main__":
    test_kv_pca()
