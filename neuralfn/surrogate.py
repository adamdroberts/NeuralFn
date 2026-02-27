from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .neuron import NeuronDef
from .graph import NeuronGraph


def probe_neuron(
    neuron_def: NeuronDef,
    n_samples: int = 10_000,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a neuron's transfer function by feeding random inputs.

    Returns (inputs, outputs) arrays of shape (n_samples, n_ports).
    """
    rng = rng or np.random.default_rng()
    n_in = neuron_def.n_inputs
    n_out = neuron_def.n_outputs

    xs = np.empty((n_samples, n_in), dtype=np.float32)
    for i, port in enumerate(neuron_def.input_ports):
        lo, hi = port.range
        xs[:, i] = rng.uniform(lo, hi, size=n_samples).astype(np.float32)

    ys = np.empty((n_samples, n_out), dtype=np.float32)
    for row in range(n_samples):
        args = tuple(float(xs[row, j]) for j in range(n_in))
        result = neuron_def(*args)
        for j in range(n_out):
            ys[row, j] = result[j] if j < len(result) else 0.0

    return xs, ys


class SurrogateModel(nn.Module):
    """Small MLP trained to approximate a neuron's transfer function."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_inputs
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            prev = h
        layers.append(nn.Linear(prev, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_surrogate(
    model: SurrogateModel,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> float:
    """Train a SurrogateModel on probed data.  Returns final loss."""
    device = next(model.parameters()).device
    x_t = torch.tensor(xs, dtype=torch.float32, device=device)
    y_t = torch.tensor(ys, dtype=torch.float32, device=device)

    dataset = torch.utils.data.TensorDataset(x_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    last_loss = float("inf")
    for _ in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        last_loss = epoch_loss / len(dataset)

    return last_loss


def build_surrogates(
    graph: NeuronGraph,
    n_samples: int = 10_000,
    hidden_sizes: tuple[int, ...] = (64, 64),
    epochs: int = 200,
    lr: float = 1e-3,
) -> dict[str, SurrogateModel]:
    """Probe every neuron in the graph and build a surrogate for each.

    Returns a dict mapping instance_id -> trained SurrogateModel.
    """
    surrogates: dict[str, SurrogateModel] = {}
    for nid, node in graph.nodes.items():
        ndef = node.neuron_def
        xs, ys = probe_neuron(ndef, n_samples)
        model = SurrogateModel(ndef.n_inputs, ndef.n_outputs, hidden_sizes)
        train_surrogate(model, xs, ys, epochs=epochs, lr=lr)
        model.eval()
        surrogates[nid] = model
    return surrogates
