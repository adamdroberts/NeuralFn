from __future__ import annotations

from neuralfn import TorchTrainConfig, TorchTrainer, build_gpt_root_graph


if __name__ == "__main__":
    graph = build_gpt_root_graph(
        config={
            "vocab_size": 16,
            "num_layers": 4,
            "model_dim": 32,
            "num_heads": 4,
            "num_kv_heads": 2,
            "mlp_mult": 2,
            "tie_embeddings": True,
            "logit_softcap": 30.0,
        }
    )
    train_inputs = [
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
    ]
    train_targets = [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
    ]
    trainer = TorchTrainer(graph, TorchTrainConfig(epochs=10, learning_rate=5e-3, batch_size=2))
    losses = trainer.train(train_inputs, train_targets)
    print({"initial_loss": losses[0], "final_loss": losses[-1], "epochs": len(losses)})
