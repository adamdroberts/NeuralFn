from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .graph import Edge, NeuronGraph, NeuronInstance
from .neuron import NeuronDef


@dataclass
class EvoConfig:
    population_size: int = 50
    generations: int = 200
    mutation_rate: float = 0.1
    mutation_scale: float = 0.3
    crossover_rate: float = 0.5
    tournament_size: int = 3
    elite_count: int = 2
    topology_mutations: bool = False
    seed: int | None = None


class EvolutionaryTrainer:
    """Genetic-algorithm optimizer for edge weights (and optionally topology)."""

    def __init__(
        self,
        graph: NeuronGraph,
        config: EvoConfig | None = None,
        neuron_library: list[NeuronDef] | None = None,
    ) -> None:
        self.graph = graph
        self.config = config or EvoConfig()
        self.neuron_library = neuron_library or []
        self._stop = False
        self.loss_history: list[float] = []
        self._rng = np.random.default_rng(self.config.seed)

    def stop(self) -> None:
        self._stop = True

    def train(
        self,
        train_inputs: np.ndarray,
        train_targets: np.ndarray,
        *,
        fitness_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
        on_generation: Callable[[int, float], None] | None = None,
    ) -> list[float]:
        """Run the evolutionary optimisation loop.

        Uses actual neuron functions (not surrogates) for evaluation.
        """
        self._stop = False
        self.loss_history = []
        cfg = self.config

        if fitness_fn is None:
            fitness_fn = _mse_fitness

        base_params = self.graph.get_edge_params()
        pop = self._init_population(base_params)

        for gen in range(cfg.generations):
            if self._stop:
                break

            scores = [
                self._evaluate(ind, train_inputs, train_targets, fitness_fn)
                for ind in pop
            ]

            ranked = sorted(zip(scores, pop), key=lambda x: x[0])
            best_loss = ranked[0][0]
            self.loss_history.append(best_loss)

            if on_generation:
                on_generation(gen, best_loss)

            next_pop: list[list[float]] = []
            for i in range(cfg.elite_count):
                next_pop.append(list(ranked[i][1]))

            while len(next_pop) < cfg.population_size:
                p1 = self._tournament(scores, pop)
                p2 = self._tournament(scores, pop)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                next_pop.append(child)

            pop = next_pop

        best = min(zip(
            [self._evaluate(ind, train_inputs, train_targets, fitness_fn) for ind in pop],
            pop,
        ), key=lambda x: x[0])
        self.graph.set_edge_params(best[1])
        return self.loss_history

    # ── internals ─────────────────────────────────────────────────────

    def _init_population(self, base: list[float]) -> list[list[float]]:
        pop: list[list[float]] = [list(base)]
        for _ in range(self.config.population_size - 1):
            ind = [
                v + self._rng.normal(0, self.config.mutation_scale)
                for v in base
            ]
            pop.append(ind)
        return pop

    def _evaluate(
        self,
        params: list[float],
        inputs: np.ndarray,
        targets: np.ndarray,
        fitness_fn: Callable[[np.ndarray, np.ndarray], float],
    ) -> float:
        self.graph.set_edge_params(params)
        preds = []
        for i in range(len(inputs)):
            inp_map: dict[str, tuple[float, ...]] = {}
            idx = 0
            for nid in self.graph.input_node_ids:
                node = self.graph.nodes[nid]
                n_out = node.neuron_def.n_outputs
                inp_map[nid] = tuple(float(inputs[i, j]) for j in range(idx, idx + n_out))
                idx += n_out
            out = self.graph.execute(inp_map)
            row: list[float] = []
            for nid in self.graph.output_node_ids:
                row.extend(out.get(nid, ()))
            preds.append(row)
        return fitness_fn(np.array(preds, dtype=np.float32), targets)

    def _tournament(self, scores: list[float], pop: list[list[float]]) -> list[float]:
        idxs = self._rng.choice(len(pop), size=self.config.tournament_size, replace=False)
        best_idx = min(idxs, key=lambda i: scores[i])
        return list(pop[best_idx])

    def _crossover(self, p1: list[float], p2: list[float]) -> list[float]:
        child: list[float] = []
        for a, b in zip(p1, p2):
            if self._rng.random() < self.config.crossover_rate:
                child.append(b)
            else:
                child.append(a)
        return child

    def _mutate(self, ind: list[float]) -> list[float]:
        for i in range(len(ind)):
            if self._rng.random() < self.config.mutation_rate:
                ind[i] += self._rng.normal(0, self.config.mutation_scale)
        return ind


def _mse_fitness(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((preds - targets) ** 2))
