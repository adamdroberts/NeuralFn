import time
import torch
from neuralfn.config import build_llama_fast_spec
from neuralfn.torch_templates import build_gpt_root_graph
from neuralfn.graph import NeuronInstance, NeuronGraph
from neuralfn.builtins import BuiltinNeurons
from neuralfn.torch_backend import TorchTrainer, TorchTrainConfig

def benchmark():
    print("Building llama_fast spec...")
    spec = build_llama_fast_spec(num_layers=2, model_dim=256, num_heads=4, num_kv_heads=2, vocab_size=1024)
    graph = build_gpt_root_graph(name="llama_benchmark", model_spec=spec)
    
    print("Adding dataset_source node...")
    # Add a dataset_source node to the graph
    import copy
    from neuralfn.torch_templates import clone_neuron_def
    
    ds_def = clone_neuron_def(BuiltinNeurons.dataset_source_module, config={
        "dataset_names": ["willdepueoai__parameter-golf__sp1024__train10"], # full 10 shards
        "seq_len": 256
    })
    
    # We replace tokens_in and targets_in with ds_node
    # Actually, we can just inject dataset_source and make it the only input_node_id
    graph.add_node(NeuronInstance(ds_def, instance_id="ds_source"))
    graph.input_node_ids = ["ds_source"]
    
    # Wire ds_source to model
    # The existing edges are e_tokens_model and e_targets_model
    # Let's just remove them and re-add from ds_source
    graph.remove_edge("e_tokens_model")
    graph.remove_edge("e_targets_model")
    
    from neuralfn.graph import Edge
    graph.add_edge(Edge(id="e_ds_tokens_model", src_node="ds_source", src_port=0, dst_node="model", dst_port=0))
    graph.add_edge(Edge(id="e_ds_targets_model", src_node="ds_source", src_port=1, dst_node="model", dst_port=1))
    
    # Remove old input nodes
    graph.remove_node("tokens_in")
    graph.remove_node("targets_in")
    
    # Also fix output detection. If we just leave it alone TorchTrainer._auto_detect_outputs will find loss_out.
    
    # TorchTrainer will automatically load the dataset
    print("Initializing TorchTrainer...")
    config = TorchTrainConfig(
        learning_rate=3e-4,
        epochs=1,
        batch_size=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        amp_dtype="bfloat16",
        compile=True,
        max_steps=50
    )
    
    trainer = TorchTrainer(graph, config)
    
    print("Starting training step...")
    steps = 0
    t0 = time.time()
    
    def on_epoch(epoch, avg_loss):
        nonlocal steps
        print(f"Finished {config.max_steps} steps: loss = {avg_loss:.4f}")
        
    trainer.train([], [], on_epoch=on_epoch)
    
    t1 = time.time()
    elapsed = t1 - t0
    tokens_processed = config.max_steps * config.batch_size * 256
    throughput = tokens_processed / elapsed
    print(f"Training finished {config.max_steps} steps in {elapsed:.2f} seconds.")
    print(f"Throughput: {throughput:.2f} tokens/second")

if __name__ == "__main__":
    benchmark()
