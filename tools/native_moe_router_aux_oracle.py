"""Dependency-light CPU oracle for the shipped standard-MoE graph aux loss."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


Matrix = tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class RouterAuxResult:
    probabilities: Matrix
    density: tuple[float, ...]
    raw_loss: float
    weighted_loss: float
    gradient: Matrix


def _finite_matrix(values: Sequence[Sequence[float]], name: str) -> Matrix:
    if not values:
        raise ValueError(f"{name} must contain at least one row")
    width = len(values[0])
    if width == 0:
        raise ValueError(f"{name} must contain at least one expert")
    matrix: list[tuple[float, ...]] = []
    for row in values:
        if len(row) != width:
            raise ValueError(f"{name} must be rectangular")
        converted = tuple(float(value) for value in row)
        if not all(math.isfinite(value) for value in converted):
            raise ValueError(f"{name} must contain only finite values")
        matrix.append(converted)
    return tuple(matrix)


def standard_moe_router_aux(
    router_logits: Sequence[Sequence[float]],
    coefficient: float,
) -> RouterAuxResult:
    """Compute exact loss and all-expert softmax-Jacobian gradient.

    The raw graph loss is ``experts * sum(mean_tokens(softmax(logits)) ** 2)``.
    ``coefficient`` is applied once to that raw loss and its gradient.
    """

    logits = _finite_matrix(router_logits, "router_logits")
    coefficient = float(coefficient)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("coefficient must be finite and non-negative")

    probabilities: list[tuple[float, ...]] = []
    for row in logits:
        row_max = max(row)
        exponentials = tuple(math.exp(value - row_max) for value in row)
        denominator = sum(exponentials)
        probabilities.append(tuple(value / denominator for value in exponentials))

    rows = len(probabilities)
    experts = len(probabilities[0])
    density = tuple(
        sum(probabilities[row][expert] for row in range(rows)) / rows
        for expert in range(experts)
    )
    raw_loss = experts * sum(value * value for value in density)
    weighted_loss = coefficient * raw_loss

    if coefficient == 0.0:
        gradient: Matrix = tuple(tuple(0.0 for _ in range(experts)) for _ in range(rows))
    else:
        gradient_scale = 2.0 * coefficient * experts / rows
        gradient = tuple(
            tuple(
                gradient_scale
                * probability
                * (density[expert] - density_probability_dot)
                for expert, probability in enumerate(row)
            )
            for row in probabilities
            for density_probability_dot in [
                sum(density[expert] * probability for expert, probability in enumerate(row))
            ]
        )

    return RouterAuxResult(
        probabilities=tuple(probabilities),
        density=density,
        raw_loss=raw_loss,
        weighted_loss=weighted_loss,
        gradient=gradient,
    )


def accumulate_router_aux_gradient(
    expert_route_gradient: Sequence[Sequence[float]],
    router_aux_gradient: Sequence[Sequence[float]],
) -> Matrix:
    """Add the aux gradient without discarding the expert-route contribution."""

    base = _finite_matrix(expert_route_gradient, "expert_route_gradient")
    auxiliary = _finite_matrix(router_aux_gradient, "router_aux_gradient")
    if len(base) != len(auxiliary) or len(base[0]) != len(auxiliary[0]):
        raise ValueError("gradient shapes must match")
    return tuple(
        tuple(base_value + aux_value for base_value, aux_value in zip(base_row, aux_row))
        for base_row, aux_row in zip(base, auxiliary)
    )


def weighted_multi_layer_router_aux_loss(
    layer_logits: Iterable[Sequence[Sequence[float]]],
    coefficient: float,
) -> float:
    """Apply the coefficient once to the sum of raw per-layer graph losses."""

    coefficient = float(coefficient)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("coefficient must be finite and non-negative")
    raw_total = sum(standard_moe_router_aux(logits, 1.0).raw_loss for logits in layer_logits)
    return coefficient * raw_total
