"""XOR demo — builds a small graph of custom neurons and trains it."""

import sys
sys.path.insert(0, ".")

import numpy as np
from neuralfn import (
    Port, NeuronDef, neuron, NeuronGraph, NeuronInstance, Edge,
    SurrogateTrainer, EvolutionaryTrainer, save_graph,
)
from neuralfn.builtins import input_node, output_node, sigmoid
from neuralfn.trainer import TrainConfig
from neuralfn.evolutionary import EvoConfig


@neuron(
    inputs=[Port("a", range=(-10, 10)), Port("b", range=(-10, 10))],
    outputs=[Port("sum", range=(-20, 20))],
)
def weighted_sum(a, b):
    return a + b


def build_xor_graph() -> NeuronGraph:
    g = NeuronGraph()

    in1 = NeuronInstance(input_node, instance_id="in1")
    in2 = NeuronInstance(input_node, instance_id="in2")
    h1 = NeuronInstance(weighted_sum, instance_id="h1")
    h2 = NeuronInstance(weighted_sum, instance_id="h2")
    a1 = NeuronInstance(sigmoid, instance_id="a1")
    a2 = NeuronInstance(sigmoid, instance_id="a2")
    h3 = NeuronInstance(weighted_sum, instance_id="h3")
    a3 = NeuronInstance(sigmoid, instance_id="a3")
    out = NeuronInstance(output_node, instance_id="out")

    for node in [in1, in2, h1, h2, a1, a2, h3, a3, out]:
        g.add_node(node)

    g.input_node_ids = ["in1", "in2"]
    g.output_node_ids = ["out"]

    edges = [
        Edge(id="e1", src_node="in1", src_port=0, dst_node="h1", dst_port=0, weight=1.0, bias=0.0),
        Edge(id="e2", src_node="in2", src_port=0, dst_node="h1", dst_port=1, weight=1.0, bias=0.0),
        Edge(id="e3", src_node="in1", src_port=0, dst_node="h2", dst_port=0, weight=1.0, bias=0.0),
        Edge(id="e4", src_node="in2", src_port=0, dst_node="h2", dst_port=1, weight=1.0, bias=0.0),
        Edge(id="e5", src_node="h1", src_port=0, dst_node="a1", dst_port=0, weight=1.0, bias=0.0),
        Edge(id="e6", src_node="h2", src_port=0, dst_node="a2", dst_port=0, weight=1.0, bias=0.0),
        Edge(id="e7", src_node="a1", src_port=0, dst_node="h3", dst_port=0, weight=1.0, bias=0.0),
        Edge(id="e8", src_node="a2", src_port=0, dst_node="h3", dst_port=1, weight=1.0, bias=0.0),
        Edge(id="e9", src_node="h3", src_port=0, dst_node="a3", dst_port=0, weight=1.0, bias=0.0),
        Edge(id="e10", src_node="a3", src_port=0, dst_node="out", dst_port=0, weight=1.0, bias=0.0),
    ]
    for e in edges:
        g.add_edge(e)

    return g


def main():
    g = build_xor_graph()
    print(f"Graph: {len(g.nodes)} nodes, {len(g.edges)} edges")
    print(f"Has cycles: {g.has_cycles()}")

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    print("\n--- Before training ---")
    for row in X:
        out = g.execute({"in1": (float(row[0]),), "in2": (float(row[1]),)})
        print(f"  {row} -> {out['out']}")

    print("\n--- Surrogate training ---")
    cfg = TrainConfig(epochs=300, learning_rate=0.01)
    trainer = SurrogateTrainer(g, cfg)
    losses = trainer.train(X, Y, on_epoch=lambda ep, l: (
        print(f"  epoch {ep:3d}  loss={l:.6f}") if ep % 50 == 0 else None
    ))
    print(f"  final loss: {losses[-1]:.6f}")

    print("\n--- After surrogate training ---")
    for row in X:
        out = g.execute({"in1": (float(row[0]),), "in2": (float(row[1]),)})
        print(f"  {row} -> {out['out']}")

    save_graph(g, "examples/xor_trained.json")
    print("\nSaved to examples/xor_trained.json")

    print("\n--- Evolutionary training (fresh graph) ---")
    g2 = build_xor_graph()
    evo_cfg = EvoConfig(population_size=40, generations=100)
    evo = EvolutionaryTrainer(g2, evo_cfg)
    evo_losses = evo.train(X, Y, on_generation=lambda gen, l: (
        print(f"  gen {gen:3d}  loss={l:.6f}") if gen % 20 == 0 else None
    ))
    print(f"  final loss: {evo_losses[-1]:.6f}")

    print("\n--- After evolutionary training ---")
    for row in X:
        out = g2.execute({"in1": (float(row[0]),), "in2": (float(row[1]),)})
        print(f"  {row} -> {out['out']}")


if __name__ == "__main__":
    main()
