from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from .neuron import NeuronDef
from .port import Port


@dataclass
class NeuronInstance:
    """A placed instance of a NeuronDef inside a graph."""

    neuron_def: NeuronDef
    instance_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    position: tuple[float, float] = (0.0, 0.0)

    @property
    def name(self) -> str:
        return self.neuron_def.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "neuron_def": self.neuron_def.to_dict(),
            "position": list(self.position),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NeuronInstance:
        return cls(
            neuron_def=NeuronDef.from_dict(d["neuron_def"]),
            instance_id=d["instance_id"],
            position=tuple(d.get("position", [0, 0])),
        )


@dataclass
class Edge:
    """Weighted connection between two neuron-instance ports."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    src_node: str = ""
    src_port: int = 0
    dst_node: str = ""
    dst_port: int = 0
    weight: float = 1.0
    bias: float = 0.0

    def transform(self, value: float) -> float:
        return value * self.weight + self.bias

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "src_node": self.src_node,
            "src_port": self.src_port,
            "dst_node": self.dst_node,
            "dst_port": self.dst_port,
            "weight": self.weight,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Edge:
        return cls(
            id=d["id"],
            src_node=d["src_node"],
            src_port=d["src_port"],
            dst_node=d["dst_node"],
            dst_port=d["dst_port"],
            weight=d.get("weight", 1.0),
            bias=d.get("bias", 0.0),
        )


class NeuronGraph:
    """A directed graph of NeuronInstances connected by weighted Edges.

    Supports both DAG (feedforward) and cyclic (recurrent) execution.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, NeuronInstance] = {}
        self.edges: dict[str, Edge] = {}
        self.input_node_ids: list[str] = []
        self.output_node_ids: list[str] = []

    # ── node / edge mutations ─────────────────────────────────────────

    def add_node(self, instance: NeuronInstance) -> str:
        self.nodes[instance.instance_id] = instance
        return instance.instance_id

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        to_remove = [
            eid for eid, e in self.edges.items()
            if e.src_node == node_id or e.dst_node == node_id
        ]
        for eid in to_remove:
            self.edges.pop(eid, None)
        self.input_node_ids = [n for n in self.input_node_ids if n != node_id]
        self.output_node_ids = [n for n in self.output_node_ids if n != node_id]

    def add_edge(self, edge: Edge) -> str:
        if edge.src_node not in self.nodes or edge.dst_node not in self.nodes:
            raise ValueError("Both src and dst nodes must exist in the graph")
        self.edges[edge.id] = edge
        return edge.id

    def remove_edge(self, edge_id: str) -> None:
        self.edges.pop(edge_id, None)

    # ── topology helpers ──────────────────────────────────────────────

    def _build_nx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for nid in self.nodes:
            g.add_node(nid)
        for e in self.edges.values():
            g.add_edge(e.src_node, e.dst_node)
        return g

    def has_cycles(self) -> bool:
        return not nx.is_directed_acyclic_graph(self._build_nx())

    def topological_order(self) -> list[str]:
        return list(nx.topological_sort(self._build_nx()))

    # ── incoming edges lookup ─────────────────────────────────────────

    def _incoming(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges.values() if e.dst_node == node_id]

    # ── execution ─────────────────────────────────────────────────────

    def execute(
        self,
        inputs: dict[str, tuple[float, ...]],
        *,
        max_iters: int = 50,
        damping: float = 0.5,
        tolerance: float = 1e-6,
    ) -> dict[str, tuple[float, ...]]:
        """Run the graph.

        Args:
            inputs: mapping of input-node instance_id -> tuple of values.
            max_iters: iteration cap for cyclic graphs.
            damping: blending factor for recurrent settling (0=keep old, 1=use new).
            tolerance: convergence threshold for cyclic mode.

        Returns:
            mapping of output-node instance_id -> tuple of output values.
        """
        if self.has_cycles():
            return self._execute_cyclic(inputs, max_iters, damping, tolerance)
        return self._execute_dag(inputs)

    def _execute_dag(
        self, inputs: dict[str, tuple[float, ...]],
    ) -> dict[str, tuple[float, ...]]:
        values: dict[str, tuple[float, ...]] = {}

        for nid in inputs:
            values[nid] = inputs[nid]

        for nid in self.topological_order():
            if nid in values and nid in inputs:
                continue
            node = self.nodes[nid]
            port_accum = self._gather_inputs(nid, values, node)
            result = node.neuron_def(*port_accum)
            values[nid] = result

        return {nid: values.get(nid, ()) for nid in self.output_node_ids}

    def _execute_cyclic(
        self,
        inputs: dict[str, tuple[float, ...]],
        max_iters: int,
        damping: float,
        tolerance: float,
    ) -> dict[str, tuple[float, ...]]:
        values: dict[str, tuple[float, ...]] = {}

        for nid, node in self.nodes.items():
            if nid in inputs:
                values[nid] = inputs[nid]
            else:
                n_out = node.neuron_def.n_outputs
                values[nid] = tuple(0.0 for _ in range(n_out))

        for _ in range(max_iters):
            new_values: dict[str, tuple[float, ...]] = {}
            max_delta = 0.0

            for nid, node in self.nodes.items():
                if nid in inputs:
                    new_values[nid] = inputs[nid]
                    continue

                port_accum = self._gather_inputs(nid, values, node)
                raw = node.neuron_def(*port_accum)
                old = values[nid]
                blended = tuple(
                    damping * r + (1 - damping) * o for r, o in zip(raw, old)
                )
                new_values[nid] = blended
                max_delta = max(max_delta, max(abs(r - o) for r, o in zip(raw, old)))

            values = new_values
            if max_delta < tolerance:
                break

        return {nid: values.get(nid, ()) for nid in self.output_node_ids}

    def _gather_inputs(
        self,
        nid: str,
        values: dict[str, tuple[float, ...]],
        node: NeuronInstance,
    ) -> tuple[float, ...]:
        """Collect and condition incoming edge values for a node."""
        n_in = node.neuron_def.n_inputs
        port_accum = [0.0] * n_in
        for e in self._incoming(nid):
            src_vals = values.get(e.src_node, ())
            if e.src_port < len(src_vals):
                raw = e.transform(src_vals[e.src_port])
                dst_port_spec = node.neuron_def.input_ports[e.dst_port]
                port_accum[e.dst_port] += dst_port_spec.condition(raw)
        return tuple(port_accum)

    # ── edge parameter vector (for training) ──────────────────────────

    def get_edge_params(self) -> list[float]:
        params: list[float] = []
        for eid in sorted(self.edges):
            e = self.edges[eid]
            params.extend([e.weight, e.bias])
        return params

    def set_edge_params(self, params: list[float]) -> None:
        idx = 0
        for eid in sorted(self.edges):
            self.edges[eid].weight = params[idx]
            self.edges[eid].bias = params[idx + 1]
            idx += 2

    def param_count(self) -> int:
        return len(self.edges) * 2

    # ── serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": {eid: e.to_dict() for eid, e in self.edges.items()},
            "input_node_ids": self.input_node_ids,
            "output_node_ids": self.output_node_ids,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NeuronGraph:
        g = cls()
        for nid, nd in d["nodes"].items():
            g.nodes[nid] = NeuronInstance.from_dict(nd)
        for eid, ed in d["edges"].items():
            g.edges[eid] = Edge.from_dict(ed)
        g.input_node_ids = d.get("input_node_ids", [])
        g.output_node_ids = d.get("output_node_ids", [])
        return g
