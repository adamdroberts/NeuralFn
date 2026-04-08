import torch
from neuralfn.config import build_decoder2encoder_moe_spec
from neuralfn.torch_templates import build_gpt_root_graph
from neuralfn.torch_backend import CompiledTorchGraph

def test_seq2seq_template():
    spec = build_decoder2encoder_moe_spec(num_layers=2, experts=4, top_k=2)
    graph = build_gpt_root_graph(name="seq2seq_test", model_spec=spec)
    
    compiled = CompiledTorchGraph(graph)
    enc_tokens = torch.randint(0, 256, (1, 16))
    dec_tokens = torch.randint(0, 256, (1, 16))
    targets = torch.randint(0, 256, (1, 16))
    
    loss = compiled(enc_tokens, dec_tokens, targets)[0]
    assert loss.ndim == 0
    print("Seq2Seq (Encoder-Decoder) template verified.")

if __name__ == "__main__":
    test_seq2seq_template()
