"""Nested graph demo with one surrogate-trained block and one evolutionary block."""

import sys

sys.path.insert(0, ".")

import numpy as np

from neuralfn import (
    BuiltinNeurons,
    Edge,
    HybridConfig,
    HybridTrainer,
    NeuronGraph,
    NeuronInstance,
    subgraph_neuron,
)
from neuralfn.evolutionary import EvoConfig
from neuralfn.trainer import TrainConfig


def make_affine_block(name: str, method: str) -> NeuronGraph:
    graph = NeuronGraph(
        name=name,
        training_method=method,
        surrogate_config={
            "epochs": 30,
            "learning_rate": 0.05,
            "batch_size": 8,
            "surrogate_samples": 512,
            "surrogate_epochs": 40,
        },
        evo_config={
            "population_size": 20,
            "generations": 20,
            "mutation_rate": 0.3,
            "mutation_scale": 0.4,
            "elite_count": 2,
            "seed": 11,
        },
    )
    graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
    graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
    graph.add_edge(
        Edge(
            id=f"{name}-edge",
            src_node="in",
            src_port=0,
            dst_node="out",
            dst_port=0,
            weight=0.1,
            bias=0.0,
        )
    )
    graph.input_node_ids = ["in"]
    graph.output_node_ids = ["out"]
    return graph


def build_root_network() -> NeuronGraph:
    surrogate_block = make_affine_block("surrogate_block", "surrogate")
    evolutionary_block = make_affine_block("evolutionary_block", "evolutionary")

    root = NeuronGraph(name="root_network", training_method="frozen")
    root.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="root_in"))
    root.add_node(
        NeuronInstance(
            subgraph_neuron(
                surrogate_block,
                name="surrogate_stage",
                input_aliases=["signal_in"],
                output_aliases=["signal_mid"],
            ),
            instance_id="surrogate_stage",
        )
    )
    root.add_node(
        NeuronInstance(
            subgraph_neuron(
                evolutionary_block,
                name="evolutionary_stage",
                input_aliases=["signal_mid"],
                output_aliases=["signal_out"],
            ),
            instance_id="evolutionary_stage",
        )
    )
    root.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="root_out"))

    root.add_edge(Edge(id="e1", src_node="root_in", src_port=0, dst_node="surrogate_stage", dst_port=0))
    root.add_edge(Edge(id="e2", src_node="surrogate_stage", src_port=0, dst_node="evolutionary_stage", dst_port=0))
    root.add_edge(Edge(id="e3", src_node="evolutionary_stage", src_port=0, dst_node="root_out", dst_port=0))
    root.input_node_ids = ["root_in"]
    root.output_node_ids = ["root_out"]
    return root


def main() -> None:
    graph = build_root_network()
    xs = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(-1, 1)
    ys = (2.0 * xs + 1.0).astype(np.float32)

    print("Nested graph before training:")
    for row in xs[:4]:
        out = graph.execute({"root_in": (float(row[0]),)})
        print(f"  x={row[0]: .3f} -> {out['root_out']}")

    trainer = HybridTrainer(
        graph,
        HybridConfig(
            outer_rounds=3,
            default_surrogate=TrainConfig(
                learning_rate=0.05,
                epochs=25,
                batch_size=8,
                surrogate_samples=512,
                surrogate_epochs=40,
            ),
            default_evolutionary=EvoConfig(
                population_size=20,
                generations=20,
                mutation_rate=0.3,
                mutation_scale=0.4,
                elite_count=2,
                seed=11,
            ),
        ),
    )
    trainer.train(
        xs,
        ys,
        on_step=lambda info: (
            print(
                f"  round={info['round']} method={info['method']:<12} "
                f"graph={info['graph_name']:<18} step={info['local_step']:<3} "
                f"loss={info['loss']:.6f}"
            )
            if info["local_step"] in {0, 9, 19, 24}
            else None
        ),
    )

    print("\nNested graph after training:")
    for row in xs[:4]:
        out = graph.execute({"root_in": (float(row[0]),)})
        print(f"  x={row[0]: .3f} -> {out['root_out']}")


if __name__ == "__main__":
    main()
