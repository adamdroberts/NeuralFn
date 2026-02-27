from __future__ import annotations

import inspect
import math
import textwrap
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .port import Port


@dataclass
class NeuronDef:
    """Immutable definition of a neuron type.

    Created by the ``@neuron`` decorator or from serialised JSON.
    """

    name: str
    fn: Callable[..., Any]
    input_ports: list[Port]
    output_ports: list[Port]
    source_code: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __call__(self, *args: float) -> tuple[float, ...]:
        result = self.fn(*args)
        if not isinstance(result, (tuple, list)):
            result = (result,)
        conditioned: list[float] = []
        for val, port in zip(result, self.output_ports):
            conditioned.append(port.condition(float(val)))
        return tuple(conditioned)

    @property
    def n_inputs(self) -> int:
        return len(self.input_ports)

    @property
    def n_outputs(self) -> int:
        return len(self.output_ports)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "input_ports": [p.to_dict() for p in self.input_ports],
            "output_ports": [p.to_dict() for p in self.output_ports],
            "source_code": self.source_code,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NeuronDef:
        """Reconstruct a NeuronDef from serialised JSON.

        The callable is rebuilt by exec-ing the stored source code.
        """
        import neuralfn.port as _port_mod
        ns: dict[str, Any] = {"math": math, "Port": _port_mod.Port}
        ns["neuron"] = lambda **_kw: (lambda fn: fn)
        pre_keys = set(ns.keys())
        exec(d["source_code"], ns)  # noqa: S102
        fn_name = d["name"]
        fn = ns.get(fn_name)
        if fn is None:
            new_keys = set(ns.keys()) - pre_keys - {"__builtins__"}
            fns = [ns[k] for k in new_keys if callable(ns[k]) and not isinstance(ns[k], type)]
            fn = fns[0] if fns else (lambda *a: 0.0)
        return cls(
            id=d["id"],
            name=fn_name,
            fn=fn,
            input_ports=[Port.from_dict(p) for p in d["input_ports"]],
            output_ports=[Port.from_dict(p) for p in d["output_ports"]],
            source_code=d["source_code"],
        )


def neuron(
    inputs: Sequence[Port],
    outputs: Sequence[Port],
    *,
    name: str | None = None,
) -> Callable[[Callable[..., Any]], NeuronDef]:
    """Decorator that turns a plain function into a ``NeuronDef``."""

    def wrapper(fn: Callable[..., Any]) -> NeuronDef:
        try:
            src = textwrap.dedent(inspect.getsource(fn))
        except OSError:
            src = ""
        return NeuronDef(
            name=name or fn.__name__,
            fn=fn,
            input_ports=list(inputs),
            output_ports=list(outputs),
            source_code=src,
        )

    return wrapper


def neuron_from_source(
    source_code: str,
    fn_name: str,
    input_ports: list[Port],
    output_ports: list[Port],
    *,
    neuron_id: str | None = None,
) -> NeuronDef:
    """Build a NeuronDef from a raw source-code string (used by the editor)."""
    ns: dict[str, Any] = {"math": math}
    ns["neuron"] = lambda **_kw: (lambda fn: fn)
    exec(source_code, ns)  # noqa: S102
    fn = ns.get(fn_name)
    if fn is None:
        callables = [v for v in ns.values() if callable(v) and not isinstance(v, type)]
        if not callables:
            raise ValueError(f"No callable named '{fn_name}' found in source")
        fn = callables[0]
    return NeuronDef(
        id=neuron_id or uuid.uuid4().hex[:12],
        name=fn_name,
        fn=fn,
        input_ports=input_ports,
        output_ports=output_ports,
        source_code=source_code,
    )
