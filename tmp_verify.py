import torch
from neuralfn.config import build_nanogpt_spec, build_llama_spec, build_moe_spec
from neuralfn.torch_templates import build_gpt_root_graph
from neuralfn.torch_backend import CompiledTorchGraph

def main():
    specs = [
        ("nanogpt", build_nanogpt_spec(num_heads=2)),
        ("llama", build_llama_spec(num_heads=2, num_kv_heads=1)),
        ("moe", build_moe_spec(num_heads=2, experts=2, top_k=1))
    ]

    for name, spec in specs:
        print(f"Building {name}...")
        try:
            graph = build_gpt_root_graph(name=f"{name}_root", model_spec=spec)
            compiled = CompiledTorchGraph(graph)
            # Dummy forward pass
            tokens = torch.randint(0, spec.vocab_size, (2, 8))
            targets = torch.randint(0, spec.vocab_size, (2, 8))
            outputs = compiled(tokens, targets)
            print(f"  [OK] Output shape: {outputs[0].shape}")
            
            # Trace pass
            if name == "moe":
               trace_outs, traces = compiled.trace(tokens, targets)
               print(f"  [OK] MoE trace keys: {len(traces)}")

        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
