from __future__ import annotations

import base64
import inspect
import math
import io
import textwrap
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch

from .port import Port

if TYPE_CHECKING:
    from .graph import NeuronGraph


@dataclass
class NeuronDef:
    """Immutable definition of a neuron type.

    Created by the ``@neuron`` decorator or from serialised JSON.
    """

    name: str
    fn: Callable[..., Any] | None
    input_ports: list[Port]
    output_ports: list[Port]
    source_code: str = ""
    kind: str = "function"
    subgraph: NeuronGraph | None = None
    module_type: str = ""
    module_config: dict[str, Any] = field(default_factory=dict)
    module_state: str = ""
    input_aliases: list[str] = field(default_factory=list)
    output_aliases: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __call__(self, *args: float) -> tuple[float, ...]:
        if self.kind == "subgraph":
            if self.subgraph is None:
                raise ValueError(f"Subgraph neuron '{self.name}' is missing a nested graph")
            result = self.subgraph.execute_flat(tuple(float(arg) for arg in args))
        elif self.kind == "module":
            raise TypeError(
                f"Module neuron '{self.name}' requires the torch runtime and cannot be called "
                "through scalar graph execution"
            )
        else:
            if self.fn is None:
                raise ValueError(f"Function neuron '{self.name}' is missing a callable")
            result = self.fn(*args)
        if not isinstance(result, (tuple, list)):
            result = (result,)
        conditioned: list[float] = []
        for val, port in zip(result, self.output_ports):
            conditioned.append(port.condition(float(val)))
        return tuple(conditioned)

    def refresh_interface_ports(self) -> None:
        """Refresh aliased interface ports from the nested subgraph definition."""
        if self.kind != "subgraph" or self.subgraph is None:
            return
        self.input_ports = self.subgraph.flattened_input_ports(self.input_aliases or None)
        self.output_ports = self.subgraph.flattened_output_ports(self.output_aliases or None)
        self.input_aliases = [port.name for port in self.input_ports]
        self.output_aliases = [port.name for port in self.output_ports]

    @property
    def n_inputs(self) -> int:
        return len(self.input_ports)

    @property
    def n_outputs(self) -> int:
        return len(self.output_ports)

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "subgraph":
            self.refresh_interface_ports()
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "input_ports": [p.to_dict() for p in self.input_ports],
            "output_ports": [p.to_dict() for p in self.output_ports],
            "source_code": self.source_code,
            "subgraph": self.subgraph.to_dict() if self.subgraph is not None else None,
            "module_type": self.module_type,
            "module_config": self.module_config,
            "module_state": self.module_state,
            "input_aliases": self.input_aliases,
            "output_aliases": self.output_aliases,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NeuronDef:
        """Reconstruct a NeuronDef from serialised JSON.

        The callable is rebuilt by exec-ing the stored source code.
        """
        kind = d.get("kind", "function")
        if kind == "subgraph":
            from .graph import NeuronGraph

            subgraph_data = d.get("subgraph")
            if subgraph_data is None:
                raise ValueError(f"Subgraph neuron '{d['name']}' is missing nested graph data")
            subgraph = NeuronGraph.from_dict(subgraph_data)
            return subgraph_neuron(
                subgraph,
                name=d["name"],
                input_aliases=d.get("input_aliases") or None,
                output_aliases=d.get("output_aliases") or None,
                neuron_id=d.get("id"),
            )

        if kind == "module":
            return module_neuron(
                name=d["name"],
                module_type=d.get("module_type", d["name"]),
                input_ports=[Port.from_dict(p) for p in d["input_ports"]],
                output_ports=[Port.from_dict(p) for p in d["output_ports"]],
                module_config=d.get("module_config") or {},
                module_state=d.get("module_state", ""),
                neuron_id=d.get("id"),
            )

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
            kind=kind,
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
            kind="function",
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
        kind="function",
    )


def module_neuron(
    *,
    name: str,
    module_type: str,
    input_ports: list[Port],
    output_ports: list[Port],
    module_config: dict[str, Any] | None = None,
    module_state: str = "",
    neuron_id: str | None = None,
) -> NeuronDef:
    return NeuronDef(
        id=neuron_id or uuid.uuid4().hex[:12],
        name=name,
        fn=None,
        input_ports=input_ports,
        output_ports=output_ports,
        source_code="",
        kind="module",
        module_type=module_type,
        module_config=dict(module_config or {}),
        module_state=module_state,
    )


def subgraph_neuron(
    graph: NeuronGraph,
    *,
    name: str,
    input_aliases: list[str] | None = None,
    output_aliases: list[str] | None = None,
    neuron_id: str | None = None,
) -> NeuronDef:
    """Wrap a ``NeuronGraph`` as a reusable neuron definition."""
    graph.validate(as_subgraph=True)
    input_ports = graph.flattened_input_ports(input_aliases)
    output_ports = graph.flattened_output_ports(output_aliases)
    return NeuronDef(
        id=neuron_id or uuid.uuid4().hex[:12],
        name=name,
        fn=None,
        input_ports=input_ports,
        output_ports=output_ports,
        source_code="",
        kind="subgraph",
        subgraph=graph,
        input_aliases=[port.name for port in input_ports],
        output_aliases=[port.name for port in output_ports],
    )


def encode_module_state_dict(state_dict: dict[str, Any]) -> str:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_module_state_dict(blob: str) -> dict[str, Any]:
    if not blob:
        return {}
    data = base64.b64decode(blob.encode("ascii"))
    return torch.load(io.BytesIO(data), map_location="cpu")
