from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from .graph import NeuronGraph
from .surrogate import SurrogateModel, build_surrogates


@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    epochs: int = 500
    batch_size: int = 32
    surrogate_samples: int = 10_000
    surrogate_hidden: tuple[int, ...] = (64, 64)
    surrogate_epochs: int = 200
    loss_fn: str = "mse"


class SurrogateTrainer:
    """Gradient-based training of edge weights/biases through differentiable surrogates."""

    def __init__(self, graph: NeuronGraph, config: TrainConfig | None = None) -> None:
        self.graph = graph
        self.config = config or TrainConfig()
        self.surrogates: dict[str, SurrogateModel] = {}
        self._stop = False
        self.loss_history: list[float] = []

    def stop(self) -> None:
        self._stop = True

    def build_surrogates(self) -> None:
        self.surrogates = build_surrogates(
            self.graph,
            n_samples=self.config.surrogate_samples,
            hidden_sizes=self.config.surrogate_hidden,
            epochs=self.config.surrogate_epochs,
        )
        for m in self.surrogates.values():
            for p in m.parameters():
                p.requires_grad_(False)

    def train(
        self,
        train_inputs: np.ndarray,
        train_targets: np.ndarray,
        *,
        on_epoch: Callable[[int, float], None] | None = None,
    ) -> list[float]:
        """Train edge weights by backpropagating through surrogates.

        Args:
            train_inputs:  (N, n_graph_inputs) array
            train_targets: (N, n_graph_outputs) array
            on_epoch: optional callback(epoch, loss)

        Returns:
            List of per-epoch losses.
        """
        if not self.surrogates:
            self.build_surrogates()

        self._stop = False
        self.loss_history = []

        edge_ids = sorted(self.graph.edges.keys())
        n_params = len(edge_ids) * 2
        raw = [0.0] * n_params
        for i, eid in enumerate(edge_ids):
            e = self.graph.edges[eid]
            raw[i * 2] = e.weight
            raw[i * 2 + 1] = e.bias
        params = torch.tensor(raw, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([params], lr=self.config.learning_rate)

        loss_fn = nn.MSELoss() if self.config.loss_fn == "mse" else nn.BCEWithLogitsLoss()

        x_t = torch.tensor(train_inputs, dtype=torch.float32)
        y_t = torch.tensor(train_targets, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(x_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True,
        )

        topo = self._topo_order()

        for epoch in range(self.config.epochs):
            if self._stop:
                break
            epoch_loss = 0.0
            count = 0
            for xb, yb in loader:
                pred = self._forward(xb, params, edge_ids, topo)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
                count += xb.size(0)

            avg = epoch_loss / max(count, 1)
            self.loss_history.append(avg)

            if on_epoch:
                on_epoch(epoch, avg)

        self._write_params_back(params, edge_ids)
        return self.loss_history

    def _topo_order(self) -> list[str]:
        try:
            return self.graph.topological_order()
        except Exception:
            return list(self.graph.nodes.keys())

    def _forward(
        self,
        inputs: torch.Tensor,
        params: torch.Tensor,
        edge_ids: list[str],
        topo: list[str],
    ) -> torch.Tensor:
        """Differentiable forward pass through the surrogate graph."""
        batch = inputs.size(0)
        node_vals: dict[str, torch.Tensor] = {}

        input_idx = 0
        for nid in self.graph.input_node_ids:
            node = self.graph.nodes[nid]
            n_out = node.neuron_def.n_outputs
            node_vals[nid] = inputs[:, input_idx : input_idx + n_out]
            input_idx += n_out

        param_map: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for i, eid in enumerate(edge_ids):
            w = params[i * 2]
            b = params[i * 2 + 1]
            param_map[eid] = (w, b)

        for nid in topo:
            if nid in node_vals:
                continue
            node = self.graph.nodes[nid]
            n_in = node.neuron_def.n_inputs

            accum = torch.zeros(batch, n_in)
            for eid, edge in self.graph.edges.items():
                if edge.dst_node != nid:
                    continue
                src_vals = node_vals.get(edge.src_node)
                if src_vals is None:
                    continue
                w, b_ = param_map[eid]
                src_col = src_vals[:, edge.src_port : edge.src_port + 1]
                accum[:, edge.dst_port : edge.dst_port + 1] += src_col * w + b_

            surrogate = self.surrogates.get(nid)
            if surrogate is not None:
                node_vals[nid] = surrogate(accum)
            else:
                node_vals[nid] = accum

        parts = []
        for nid in self.graph.output_node_ids:
            if nid in node_vals:
                parts.append(node_vals[nid])
        if not parts:
            return torch.zeros(batch, 1)
        return torch.cat(parts, dim=1)

    def _write_params_back(self, params: torch.Tensor, edge_ids: list[str]) -> None:
        with torch.no_grad():
            for i, eid in enumerate(edge_ids):
                self.graph.edges[eid].weight = params[i * 2].item()
                self.graph.edges[eid].bias = params[i * 2 + 1].item()
