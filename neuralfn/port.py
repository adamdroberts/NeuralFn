from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Port:
    """Declares a single input or output slot on a neuron.

    Attributes:
        name:      Human-readable identifier (unique within a neuron).
        range:     (low, high) bounds for values on this port.
        precision: Smallest meaningful step.  Values are quantized to this
                   resolution when flowing through an edge.
        dtype:     Semantic type hint — "float", "int", or "bool".
    """

    name: str
    range: tuple[float, float] = (-1.0, 1.0)
    precision: float = 0.001
    dtype: str = "float"

    def __post_init__(self) -> None:
        lo, hi = self.range
        if lo >= hi:
            raise ValueError(f"Port '{self.name}': range low ({lo}) must be < high ({hi})")
        if self.precision <= 0:
            raise ValueError(f"Port '{self.name}': precision must be > 0")

    def clamp(self, value: float) -> float:
        lo, hi = self.range
        return max(lo, min(hi, value))

    def quantize(self, value: float) -> float:
        return round(value / self.precision) * self.precision

    def condition(self, value: float) -> float:
        """Clamp then quantize a value to this port's spec."""
        return self.quantize(self.clamp(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "range": list(self.range),
            "precision": self.precision,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Port:
        return cls(
            name=d["name"],
            range=tuple(d["range"]),
            precision=d.get("precision", 0.001),
            dtype=d.get("dtype", "float"),
        )
