from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from .evolutionary import EvoConfig
from .graph import NeuronGraph
from .surrogate import SurrogateModel, build_surrogates
from .trainer import TrainConfig


@dataclass
class HybridConfig:
    outer_rounds: int = 3
    loss_fn: str = "mse"
    default_surrogate: TrainConfig = field(default_factory=TrainConfig)
    default_evolutionary: EvoConfig = field(default_factory=EvoConfig)


@dataclass(frozen=True)
class GraphScope:
    path: tuple[str, ...]
    graph: NeuronGraph


class HybridTrainer:
    """Train nested graphs with per-graph surrogate/evolutionary updates."""

    def __init__(self, graph: NeuronGraph, config: HybridConfig | None = None) -> None:
        self.graph = graph
        self.config = config or HybridConfig()
        self._stop = False
        self.loss_history: list[float] = []
        self._shadow_cache: dict[tuple[str, ...], dict[str, SurrogateModel]] = {}
        self._topology_cache: dict[tuple[str, ...], list[str]] = {}

    def stop(self) -> None:
        self._stop = True

    def train(
        self,
        train_inputs: np.ndarray,
        train_targets: np.ndarray,
        *,
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[float]:
        self.graph.validate()
        self._stop = False
        self.loss_history = []

        scopes = self._graph_scopes_post_order(self.graph)
        if not scopes:
            return self.loss_history

        for round_idx in range(self.config.outer_rounds):
            if self._stop:
                break

            for scope in scopes:
                if self._stop:
                    break
                method = scope.graph.training_method
                if method == "frozen":
                    loss = self._evaluate_root_loss(train_inputs, train_targets)
                    self.loss_history.append(loss)
                    if on_step:
                        on_step(
                            {
                                "graph_path": list(scope.path),
                                "graph_name": scope.graph.name,
                                "method": method,
                                "round": round_idx,
                                "local_step": 0,
                                "loss": loss,
                            }
                        )
                    continue

                if method == "surrogate":
                    self._train_surrogate_scope(
                        scope,
                        train_inputs,
                        train_targets,
                        round_idx=round_idx,
                        on_step=on_step,
                    )
                elif method == "evolutionary":
                    self._train_evolutionary_scope(
                        scope,
                        train_inputs,
                        train_targets,
                        round_idx=round_idx,
                        on_step=on_step,
                    )
                else:
                    raise ValueError(
                        f"Unsupported training method '{method}' on graph '{scope.graph.name}'"
                    )

        return self.loss_history

    def _graph_scopes_post_order(
        self,
        graph: NeuronGraph,
        path: tuple[str, ...] = (),
    ) -> list[GraphScope]:
        scopes: list[GraphScope] = []
        for nid, node in graph.nodes.items():
            child = node.neuron_def.subgraph if node.neuron_def.kind == "subgraph" else None
            if child is not None:
                scopes.extend(self._graph_scopes_post_order(child, path + (nid,)))
        scopes.append(GraphScope(path=path, graph=graph))
        return scopes

    def _scope_graph(self, path: tuple[str, ...]) -> NeuronGraph:
        graph = self.graph
        for nid in path:
            node = graph.nodes[nid]
            child = node.neuron_def.subgraph
            if child is None:
                raise ValueError(f"Graph path {'/'.join(path)} is invalid at node '{nid}'")
            graph = child
        return graph

    def _merged_surrogate_config(self, graph: NeuronGraph) -> TrainConfig:
        merged = asdict(self.config.default_surrogate)
        merged.update(graph.surrogate_config)
        hidden = merged.get("surrogate_hidden", (64, 64))
        if isinstance(hidden, list):
            merged["surrogate_hidden"] = tuple(hidden)
        return TrainConfig(**merged)

    def _merged_evo_config(self, graph: NeuronGraph) -> EvoConfig:
        merged = asdict(self.config.default_evolutionary)
        merged.update(graph.evo_config)
        return EvoConfig(**merged)

    def _topo_order(self, path: tuple[str, ...], graph: NeuronGraph) -> list[str]:
        cached = self._topology_cache.get(path)
        if cached is not None:
            return cached
        try:
            topo = graph.topological_order()
        except Exception:
            topo = list(graph.nodes.keys())
        self._topology_cache[path] = topo
        return topo

    def _shadow_surrogates(
        self,
        path: tuple[str, ...],
        graph: NeuronGraph,
    ) -> dict[str, SurrogateModel]:
        cached = self._shadow_cache.get(path)
        if cached is not None:
            return cached
        cfg = self._merged_surrogate_config(graph)
        surrogates = build_surrogates(
            graph,
            n_samples=cfg.surrogate_samples,
            hidden_sizes=cfg.surrogate_hidden,
            epochs=cfg.surrogate_epochs,
        )
        for model in surrogates.values():
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
        self._shadow_cache[path] = surrogates
        return surrogates

    def _invalidate_shadow_cache(self) -> None:
        self._shadow_cache.clear()
        self._topology_cache.clear()

    def _train_surrogate_scope(
        self,
        scope: GraphScope,
        train_inputs: np.ndarray,
        train_targets: np.ndarray,
        *,
        round_idx: int,
        on_step: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        graph = scope.graph
        if graph.param_count() == 0:
            loss = self._evaluate_root_loss(train_inputs, train_targets)
            self.loss_history.append(loss)
            return

        cfg = self._merged_surrogate_config(graph)
        edge_ids = sorted(graph.edges.keys())
        raw = graph.get_edge_params()
        params = torch.tensor(raw, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([params], lr=cfg.learning_rate)
        loss_fn = nn.MSELoss() if self.config.loss_fn == "mse" else nn.BCEWithLogitsLoss()

        x_t = torch.tensor(train_inputs, dtype=torch.float32)
        y_t = torch.tensor(train_targets, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(x_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
        )

        target_path = scope.path
        for epoch in range(cfg.epochs):
            if self._stop:
                break
            epoch_loss = 0.0
            seen = 0
            for xb, yb in loader:
                pred = self._forward_graph(
                    self.graph,
                    (),
                    xb,
                    target_path=target_path,
                    target_params=params,
                    target_edge_ids=edge_ids,
                )
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
                seen += xb.size(0)
            avg = epoch_loss / max(seen, 1)
            self.loss_history.append(avg)
            if on_step:
                on_step(
                    {
                        "graph_path": list(scope.path),
                        "graph_name": graph.name,
                        "method": "surrogate",
                        "round": round_idx,
                        "local_step": epoch,
                        "loss": avg,
                    }
                )

        with torch.no_grad():
            graph.set_edge_params([float(val.item()) for val in params])
        self._invalidate_shadow_cache()

    def _forward_graph(
        self,
        graph: NeuronGraph,
        path: tuple[str, ...],
        flat_inputs: torch.Tensor,
        *,
        target_path: tuple[str, ...],
        target_params: torch.Tensor,
        target_edge_ids: list[str],
    ) -> torch.Tensor:
        batch = flat_inputs.size(0)
        node_vals: dict[str, torch.Tensor] = {}
        device = flat_inputs.device

        input_idx = 0
        for nid in graph.input_node_ids:
            node = graph.nodes[nid]
            n_out = node.neuron_def.n_outputs
            node_vals[nid] = flat_inputs[:, input_idx : input_idx + n_out]
            input_idx += n_out

        surrogates = self._shadow_surrogates(path, graph)
        topo = self._topo_order(path, graph)

        if path == target_path:
            param_map = {
                eid: (target_params[i * 2], target_params[i * 2 + 1])
                for i, eid in enumerate(target_edge_ids)
            }
        else:
            param_map = {
                eid: (
                    torch.tensor(edge.weight, dtype=torch.float32, device=device),
                    torch.tensor(edge.bias, dtype=torch.float32, device=device),
                )
                for eid, edge in graph.edges.items()
            }

        for nid in topo:
            if nid in node_vals:
                continue
            node = graph.nodes[nid]
            n_in = node.neuron_def.n_inputs
            accum = torch.zeros(batch, n_in, dtype=torch.float32, device=device)
            for eid, edge in graph.edges.items():
                if edge.dst_node != nid:
                    continue
                src_vals = node_vals.get(edge.src_node)
                if src_vals is None:
                    continue
                w, b_ = param_map[eid]
                src_col = src_vals[:, edge.src_port : edge.src_port + 1]
                accum[:, edge.dst_port : edge.dst_port + 1] += src_col * w + b_

            child_path = path + (nid,)
            child = node.neuron_def.subgraph if node.neuron_def.kind == "subgraph" else None
            if child is not None and target_path[: len(child_path)] == child_path:
                node_vals[nid] = self._forward_graph(
                    child,
                    child_path,
                    accum,
                    target_path=target_path,
                    target_params=target_params,
                    target_edge_ids=target_edge_ids,
                )
            else:
                surrogate = surrogates.get(nid)
                node_vals[nid] = surrogate(accum) if surrogate is not None else accum

        parts = [node_vals[nid] for nid in graph.output_node_ids if nid in node_vals]
        if not parts:
            return torch.zeros(batch, 1, dtype=torch.float32, device=device)
        return torch.cat(parts, dim=1)

    def _train_evolutionary_scope(
        self,
        scope: GraphScope,
        train_inputs: np.ndarray,
        train_targets: np.ndarray,
        *,
        round_idx: int,
        on_step: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        graph = scope.graph
        if graph.param_count() == 0:
            loss = self._evaluate_root_loss(train_inputs, train_targets)
            self.loss_history.append(loss)
            return

        cfg = self._merged_evo_config(graph)
        rng = np.random.default_rng(cfg.seed)
        base = graph.get_edge_params()

        population: list[list[float]] = [list(base)]
        for _ in range(cfg.population_size - 1):
            population.append(
                [val + rng.normal(0, cfg.mutation_scale) for val in base]
            )

        for gen in range(cfg.generations):
            if self._stop:
                break
            scores = [
                self._evaluate_scope_params(scope.path, individual, train_inputs, train_targets)
                for individual in population
            ]
            ranked = sorted(zip(scores, population), key=lambda item: item[0])
            best_loss = ranked[0][0]
            self.loss_history.append(best_loss)
            if on_step:
                on_step(
                    {
                        "graph_path": list(scope.path),
                        "graph_name": graph.name,
                        "method": "evolutionary",
                        "round": round_idx,
                        "local_step": gen,
                        "loss": best_loss,
                    }
                )

            next_population: list[list[float]] = [list(ind) for _score, ind in ranked[: cfg.elite_count]]
            while len(next_population) < cfg.population_size:
                p1 = self._tournament(ranked, cfg.tournament_size, rng)
                p2 = self._tournament(ranked, cfg.tournament_size, rng)
                child = []
                for a, b in zip(p1, p2):
                    child.append(b if rng.random() < cfg.crossover_rate else a)
                for idx in range(len(child)):
                    if rng.random() < cfg.mutation_rate:
                        child[idx] += rng.normal(0, cfg.mutation_scale)
                next_population.append(child)
            population = next_population

        best_params = min(
            population,
            key=lambda params: self._evaluate_scope_params(scope.path, params, train_inputs, train_targets),
        )
        graph.set_edge_params(best_params)
        self._invalidate_shadow_cache()

    def _tournament(
        self,
        ranked: list[tuple[float, list[float]]],
        tournament_size: int,
        rng: np.random.Generator,
    ) -> list[float]:
        idxs = rng.choice(len(ranked), size=min(tournament_size, len(ranked)), replace=False)
        best_idx = min(idxs, key=lambda idx: ranked[idx][0])
        return list(ranked[best_idx][1])

    def _evaluate_scope_params(
        self,
        path: tuple[str, ...],
        params: list[float],
        train_inputs: np.ndarray,
        train_targets: np.ndarray,
    ) -> float:
        graph = self._scope_graph(path)
        original = graph.get_edge_params()
        graph.set_edge_params(params)
        try:
            return self._evaluate_root_loss(train_inputs, train_targets)
        finally:
            graph.set_edge_params(original)

    def _evaluate_root_loss(
        self,
        train_inputs: np.ndarray,
        train_targets: np.ndarray,
    ) -> float:
        preds = self._predict_root(train_inputs)
        if self.config.loss_fn == "mse":
            return float(np.mean((preds - train_targets) ** 2))
        eps = 1e-7
        preds = np.clip(preds, eps, 1 - eps)
        targets = np.clip(train_targets, 0.0, 1.0)
        return float(
            -np.mean(targets * np.log(preds) + (1.0 - targets) * np.log(1.0 - preds))
        )

    def _predict_root(self, train_inputs: np.ndarray) -> np.ndarray:
        preds: list[list[float]] = []
        for row in train_inputs:
            inputs: dict[str, tuple[float, ...]] = {}
            idx = 0
            for nid in self.graph.input_node_ids:
                node = self.graph.nodes[nid]
                n_out = node.neuron_def.n_outputs
                inputs[nid] = tuple(float(row[j]) for j in range(idx, idx + n_out))
                idx += n_out
            outputs = self.graph.execute(inputs)
            flat: list[float] = []
            for nid in self.graph.output_node_ids:
                flat.extend(outputs.get(nid, ()))
            preds.append(flat)
        if not preds:
            return np.zeros((len(train_inputs), 1), dtype=np.float32)
        return np.asarray(preds, dtype=np.float32)
