from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Port:
    """Declares a single input or output slot on a neuron.

    Attributes:
        name:      Human-readable identifier (unique within a neuron).
        range:     Optional (low, high) bounds for values on this port. When
                   ``None``, the port does not clamp values — useful when the
                   network handles capping/normalization elsewhere.
        precision: Smallest meaningful step.  Values are quantized to this
                   resolution when flowing through an edge. ``None`` disables
                   quantization.
        dtype:     Semantic type hint — "float", "int", or "bool".
    """

    name: str
    range: tuple[float, float] | None = None
    precision: float | None = None
    dtype: str = "float"

    def __post_init__(self) -> None:
        if self.range is not None:
            lo, hi = self.range
            if lo >= hi:
                raise ValueError(f"Port '{self.name}': range low ({lo}) must be < high ({hi})")
        if self.precision is not None and self.precision <= 0:
            raise ValueError(f"Port '{self.name}': precision must be > 0")

    def clamp(self, value: float) -> float:
        if self.range is None:
            return value
        lo, hi = self.range
        return max(lo, min(hi, value))

    def quantize(self, value: float) -> float:
        if self.precision is None:
            return value
        return round(value / self.precision) * self.precision

    def condition(self, value: float) -> float:
        """Clamp then quantize a value to this port's spec."""
        return self.quantize(self.clamp(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "range": list(self.range) if self.range is not None else None,
            "precision": self.precision,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Port:
        raw_range = d.get("range")
        return cls(
            name=d["name"],
            range=tuple(raw_range) if raw_range is not None else None,
            precision=d.get("precision"),
            dtype=d.get("dtype", "float"),
        )
