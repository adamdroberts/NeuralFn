import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuralfn import (
    BuiltinNeurons,
    Edge,
    HybridConfig,
    HybridTrainer,
    NeuronGraph,
    NeuronInstance,
    load_graph,
    save_graph,
    subgraph_neuron,
)
from neuralfn.evolutionary import EvoConfig
from neuralfn.trainer import TrainConfig


def make_affine_graph(name: str, method: str) -> NeuronGraph:
    graph = NeuronGraph(
        name=name,
        training_method=method,
        surrogate_config={
            "epochs": 20,
            "learning_rate": 0.05,
            "batch_size": 8,
            "surrogate_samples": 256,
            "surrogate_epochs": 25,
        },
        evo_config={
            "population_size": 12,
            "generations": 12,
            "mutation_rate": 0.3,
            "mutation_scale": 0.4,
            "elite_count": 2,
            "seed": 7,
        },
    )
    graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
    graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
    graph.add_edge(Edge(id=f"{name}-e", src_node="in", src_port=0, dst_node="out", dst_port=0, weight=0.1, bias=0.0))
    graph.input_node_ids = ["in"]
    graph.output_node_ids = ["out"]
    return graph


def make_nested_root() -> NeuronGraph:
    surrogate_stage = make_affine_graph("surrogate_stage", "surrogate")
    evolutionary_stage = make_affine_graph("evolutionary_stage", "evolutionary")

    root = NeuronGraph(name="root_network", training_method="frozen")
    root.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="root_in"))
    root.add_node(
        NeuronInstance(
            subgraph_neuron(
                surrogate_stage,
                name="surrogate_block",
                input_aliases=["signal_in"],
                output_aliases=["signal_mid"],
            ),
            instance_id="surrogate_block",
        )
    )
    root.add_node(
        NeuronInstance(
            subgraph_neuron(
                evolutionary_stage,
                name="evolutionary_block",
                input_aliases=["signal_mid"],
                output_aliases=["signal_out"],
            ),
            instance_id="evolutionary_block",
        )
    )
    root.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="root_out"))

    root.add_edge(Edge(id="e1", src_node="root_in", src_port=0, dst_node="surrogate_block", dst_port=0))
    root.add_edge(Edge(id="e2", src_node="surrogate_block", src_port=0, dst_node="evolutionary_block", dst_port=0))
    root.add_edge(Edge(id="e3", src_node="evolutionary_block", src_port=0, dst_node="root_out", dst_port=0))
    root.input_node_ids = ["root_in"]
    root.output_node_ids = ["root_out"]
    return root


def mse_on_dataset(graph: NeuronGraph, xs: np.ndarray, ys: np.ndarray) -> float:
    preds = []
    for row in xs:
        out = graph.execute({"root_in": (float(row[0]),)})
        preds.append(list(out["root_out"]))
    preds_np = np.asarray(preds, dtype=np.float32)
    return float(np.mean((preds_np - ys) ** 2))


class NestedGraphsTest(unittest.TestCase):
    def test_subgraph_node_executes_and_preserves_aliases(self) -> None:
        root = make_nested_root()
        subgraph_def = root.nodes["surrogate_block"].neuron_def

        self.assertEqual("subgraph", subgraph_def.kind)
        self.assertEqual(["signal_in"], subgraph_def.input_aliases)
        self.assertEqual(["signal_mid"], subgraph_def.output_aliases)

        result = root.execute({"root_in": (0.5,)})
        self.assertIn("root_out", result)
        self.assertEqual(1, len(result["root_out"]))

    def test_nested_graph_round_trip_preserves_training_metadata(self) -> None:
        root = make_nested_root()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested.json"
            save_graph(root, path)
            loaded = load_graph(path)

        surrogate_block = loaded.nodes["surrogate_block"].neuron_def
        self.assertEqual("subgraph", surrogate_block.kind)
        self.assertIsNotNone(surrogate_block.subgraph)
        self.assertEqual("surrogate", surrogate_block.subgraph.training_method)
        self.assertEqual("evolutionary", loaded.nodes["evolutionary_block"].neuron_def.subgraph.training_method)

    def test_subgraph_validation_failures(self) -> None:
        missing_io = NeuronGraph(name="missing_io")
        missing_io.add_node(NeuronInstance(BuiltinNeurons.identity, instance_id="n1"))
        with self.assertRaises(ValueError):
            subgraph_neuron(missing_io, name="bad_subgraph")

        child = make_affine_graph("child", "surrogate")
        with self.assertRaises(ValueError):
            subgraph_neuron(child, name="bad_aliases", input_aliases=["a", "b"])

        recursive = make_affine_graph("recursive", "surrogate")
        recursive.add_node(
            NeuronInstance(
                subgraph_neuron(recursive, name="self_ref"),
                instance_id="self_ref",
            )
        )
        with self.assertRaises(ValueError):
            recursive.validate()

    def test_hybrid_training_smoke_and_param_isolation(self) -> None:
        root = make_nested_root()
        xs = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(-1, 1)
        ys = (2.0 * xs + 1.0).astype(np.float32)

        trainer = HybridTrainer(
            root,
            HybridConfig(
                outer_rounds=2,
                loss_fn="mse",
                default_surrogate=TrainConfig(
                    learning_rate=0.05,
                    epochs=15,
                    batch_size=8,
                    surrogate_samples=256,
                    surrogate_epochs=25,
                ),
                default_evolutionary=EvoConfig(
                    population_size=10,
                    generations=10,
                    mutation_rate=0.3,
                    mutation_scale=0.4,
                    elite_count=2,
                    seed=3,
                ),
            ),
        )

        scopes = {scope.graph.name: scope for scope in trainer._graph_scopes_post_order(root)}
        surrogate_scope = scopes["surrogate_stage"]
        evolutionary_scope = scopes["evolutionary_stage"]

        baseline = mse_on_dataset(root, xs, ys)
        evo_before = list(evolutionary_scope.graph.get_edge_params())
        trainer._shadow_surrogates((), root)
        self.assertTrue(trainer._shadow_cache)

        trainer._train_surrogate_scope(surrogate_scope, xs, ys, round_idx=0, on_step=None)
        self.assertEqual(evo_before, evolutionary_scope.graph.get_edge_params())
        self.assertEqual({}, trainer._shadow_cache)

        surrogate_before = list(surrogate_scope.graph.get_edge_params())
        trainer._train_evolutionary_scope(evolutionary_scope, xs, ys, round_idx=0, on_step=None)
        self.assertEqual(surrogate_before, surrogate_scope.graph.get_edge_params())

        trained = trainer.train(xs, ys)
        self.assertTrue(trained)
        self.assertLess(mse_on_dataset(root, xs, ys), baseline)


if __name__ == "__main__":
    unittest.main()
