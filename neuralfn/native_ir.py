"""Versioned graph-to-native lowering and migration artifacts.

Native Execution IR is an ahead-of-time contract. It deliberately does not
interpret graph JSON in C++: graphs are validated and resolved in Python, then
lowered to a deterministic manifest that a resident native runtime can load.
The module keeps PyTorch optional; it is imported only when a legacy .pt
checkpoint is explicitly migrated.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    from .graph import NeuronGraph


NATIVE_EXECUTION_MANIFEST_SCHEMA = "neuralfn.native_execution_manifest"
NATIVE_EXECUTION_MANIFEST_VERSION = 1
NATIVE_COMPATIBILITY_REPORT_SCHEMA = "neuralfn.native_compatibility_report"
NATIVE_COMPATIBILITY_REPORT_VERSION = 1
NATIVE_TENSOR_BUNDLE_FORMAT = "neuralfn.raw_tensor_bundle.v1"
STRUCTURED_OUTPUT_PROFILE = "json-schema-ascii-byte-greedy-v1"
STRUCTURED_OUTPUT_TOKEN_SELECTION = "current_logits_exact_prefill"
FUNCTION_TOOL_TEMPLATE_PROFILE = "responses-forced-function-call-v1"
SESSION_PREFIX_COW_PROFILE = "dense-full-cache-kv-final-hidden-v1"
CPU_TURBOQUANT_SESSION_PREFIX_COW_PROFILE = (
    "dense-cpu-turboquant-mse-qjl-packed-kv-final-hidden-v1"
)
LLAMA_SESSION_PREFIX_COW_PROFILE = "llama-full-cache-gqa-kv-final-hidden-v1"
STANDARD_MOE_SESSION_PREFIX_COW_PROFILE = (
    "standard-moe-full-cache-gqa-kv-final-hidden-v1"
)


@dataclass(frozen=True)
class NativeTensorSpec:
    """One tensor in the portable, contiguous native weight bundle."""

    name: str
    source_name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int
    sha256: str
    role: str = "parameter"
    byte_order: str = "little"
    layout: str = "row_major"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NativeTensorSpec":
        return cls(
            name=str(payload["name"]),
            source_name=str(payload["source_name"]),
            dtype=str(payload["dtype"]),
            shape=tuple(int(dim) for dim in payload.get("shape", ())),
            offset=int(payload["offset"]),
            nbytes=int(payload["nbytes"]),
            sha256=str(payload["sha256"]),
            role=str(payload.get("role", "parameter")),
            byte_order=str(payload.get("byte_order", "little")),
            layout=str(payload.get("layout", "row_major")),
        )


@dataclass(frozen=True)
class NativeExecutionManifest:
    """Serializable Native Execution IR v1 artifact contract."""

    source_graph: dict[str, Any]
    model: dict[str, Any]
    topology: dict[str, Any]
    tensors: tuple[NativeTensorSpec, ...]
    tokenizer: dict[str, Any]
    chat_template: dict[str, Any]
    context_limits: dict[str, Any]
    stop_tokens: tuple[int, ...]
    kernel_abi: dict[str, Any]
    checkpoint: dict[str, Any] | None
    session_state_kinds: tuple[str, ...]
    capabilities: dict[str, bool]
    schema: str = NATIVE_EXECUTION_MANIFEST_SCHEMA
    version: int = NATIVE_EXECUTION_MANIFEST_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "source_graph": _json_safe(self.source_graph),
            "model": _json_safe(self.model),
            "topology": _json_safe(self.topology),
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "tokenizer": _json_safe(self.tokenizer),
            "chat_template": _json_safe(self.chat_template),
            "context_limits": _json_safe(self.context_limits),
            "stop_tokens": list(self.stop_tokens),
            "kernel_abi": _json_safe(self.kernel_abi),
            "checkpoint": _json_safe(self.checkpoint),
            "session_state_kinds": list(self.session_state_kinds),
            "capabilities": dict(self.capabilities),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NativeExecutionManifest":
        schema = str(payload.get("schema", ""))
        version = int(payload.get("version", 0))
        if schema != NATIVE_EXECUTION_MANIFEST_SCHEMA:
            raise ValueError(f"Unsupported Native Execution manifest schema {schema!r}")
        if version != NATIVE_EXECUTION_MANIFEST_VERSION:
            raise ValueError(f"Unsupported Native Execution manifest version {version}")
        required = {
            "source_graph",
            "model",
            "topology",
            "tensors",
            "tokenizer",
            "chat_template",
            "context_limits",
            "stop_tokens",
            "kernel_abi",
            "session_state_kinds",
            "capabilities",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"Native Execution manifest is missing: {', '.join(missing)}")
        return cls(
            source_graph=dict(payload["source_graph"]),
            model=dict(payload["model"]),
            topology=dict(payload["topology"]),
            tensors=tuple(NativeTensorSpec.from_dict(item) for item in payload["tensors"]),
            tokenizer=dict(payload["tokenizer"]),
            chat_template=dict(payload["chat_template"]),
            context_limits=dict(payload["context_limits"]),
            stop_tokens=tuple(int(token) for token in payload["stop_tokens"]),
            kernel_abi=dict(payload["kernel_abi"]),
            checkpoint=(dict(payload["checkpoint"]) if payload.get("checkpoint") is not None else None),
            session_state_kinds=tuple(str(kind) for kind in payload["session_state_kinds"]),
            capabilities={str(key): bool(value) for key, value in dict(payload["capabilities"]).items()},
            schema=schema,
            version=version,
        )

    @classmethod
    def load(cls, path: str | Path) -> "NativeExecutionManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class NativeLoweringIssue:
    path: str
    code: str
    message: str
    severity: str = "error"
    node_kind: str = ""
    operation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NativeCompatibilityReport:
    graph_valid: bool
    issues: tuple[NativeLoweringIssue, ...]
    graph_fingerprint: str
    capabilities: dict[str, bool]
    capability_proof: dict[str, Any] = field(default_factory=dict)
    tensor_mappings: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    schema: str = NATIVE_COMPATIBILITY_REPORT_SCHEMA
    version: int = NATIVE_COMPATIBILITY_REPORT_VERSION

    @property
    def compatible(self) -> bool:
        return self.graph_valid and not any(issue.severity == "error" for issue in self.issues)

    @property
    def unsupported_node_paths(self) -> tuple[str, ...]:
        return tuple(
            sorted({issue.path for issue in self.issues if issue.code.startswith("unsupported_")})
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "compatible": self.compatible,
            "graph_valid": self.graph_valid,
            "graph_fingerprint": self.graph_fingerprint,
            "unsupported_node_paths": list(self.unsupported_node_paths),
            "issues": [issue.to_dict() for issue in self.issues],
            "warnings": list(self.warnings),
            "tensor_mappings": [_json_safe(mapping) for mapping in self.tensor_mappings],
            "capabilities": dict(self.capabilities),
            "capability_proof": _json_safe(self.capability_proof),
        }


@dataclass(frozen=True)
class NativeMigrationResult:
    """Result of a dry-run or materialized graph-to-native migration."""

    manifest: NativeExecutionManifest | None
    report: NativeCompatibilityReport
    output_dir: Path | None = None
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
            "manifest": self.manifest.to_dict() if self.manifest is not None else None,
            "compatibility_report": self.report.to_dict(),
        }


@dataclass
class _PreparedGraph:
    resolved_payload: dict[str, Any] | None
    raw_payload: dict[str, Any]
    raw_bytes: bytes
    issues: list[NativeLoweringIssue] = field(default_factory=list)
    graph_valid: bool = False


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_source(source: str) -> str:
    return "\n".join(line.rstrip() for line in str(source).strip().splitlines())


def _port_signature(payload: Mapping[str, Any]) -> str:
    return _canonical_json_bytes(
        {
            "name": payload.get("name"),
            "input_ports": payload.get("input_ports", []),
            "output_ports": payload.get("output_ports", []),
            "source_code": _normalize_source(str(payload.get("source_code", ""))),
        }
    ).decode("utf-8")


def _registered_lowerers() -> tuple[set[str], set[str]]:
    from .native_registry import registered_native_module_types

    module_types = set(registered_native_module_types())
    # Native IR v1 deliberately has no general Python-function lowerer.  The
    # sole non-interface function in the shipped text graphs is this exact
    # builtin add definition; source and ports are part of its registration.
    add_signature = _port_signature(
        {
            "name": "add",
            "input_ports": [
                {"name": "a", "range": None, "precision": None, "dtype": "float"},
                {"name": "b", "range": None, "precision": None, "dtype": "float"},
            ],
            "output_ports": [
                {"name": "sum", "range": None, "precision": None, "dtype": "float"},
            ],
            "source_code": (
                '@neuron(\n'
                '    inputs=[Port("a"), Port("b")],\n'
                '    outputs=[Port("sum")],\n'
                '    name="add",\n'
                ')\n'
                'def add(a, b):\n'
                '    return a + b\n'
            ),
        }
    )
    return module_types, {add_signature}


def _placeholder_source(output_count: int) -> str:
    if output_count <= 1:
        return "def _native_placeholder(*args):\n    return args[0] if args else 0.0\n"
    zeros = ", ".join("0.0" for _ in range(output_count))
    return f"def _native_placeholder(*args):\n    return ({zeros})\n"


def _json_pointer_token(value: str) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _is_finite_json_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _validate_graph_structure(
    graph_payload: Mapping[str, Any],
    *,
    graph_path: str,
    issues: list[NativeLoweringIssue],
) -> None:
    """Validate fields that ``NeuronGraph.validate`` currently leaves unchecked.

    This pass works exclusively on JSON values and runs before any graph object
    is constructed.  Besides producing stable node/edge paths, it prevents a
    malformed payload from reaching ``NeuronDef.from_dict``.
    """

    nodes = graph_payload.get("nodes", {})
    edges = graph_payload.get("edges", {})
    if not isinstance(nodes, dict):
        issues.append(NativeLoweringIssue(f"{graph_path}/nodes", "invalid_nodes", "Graph nodes must be an object."))
        return
    if not isinstance(edges, dict):
        issues.append(NativeLoweringIssue(f"{graph_path}/edges", "invalid_edges", "Graph edges must be an object."))
        return

    port_counts: dict[str, tuple[int, int]] = {}
    for raw_node_id, node_payload in nodes.items():
        node_id = str(raw_node_id)
        node_path = f"{graph_path}/nodes/{_json_pointer_token(node_id)}"
        if not isinstance(node_payload, dict):
            issues.append(NativeLoweringIssue(node_path, "invalid_node", "Node must be an object."))
            continue
        instance_id = node_payload.get("instance_id")
        if not isinstance(instance_id, str) or instance_id != node_id:
            issues.append(
                NativeLoweringIssue(
                    node_path,
                    "invalid_instance_id",
                    f"Node key {node_id!r} must match its string instance_id.",
                )
            )
        definition = node_payload.get("neuron_def")
        if not isinstance(definition, dict):
            issues.append(NativeLoweringIssue(node_path, "invalid_node", "Node is missing a neuron_def object."))
            continue
        input_ports = definition.get("input_ports", [])
        output_ports = definition.get("output_ports", [])
        if not isinstance(input_ports, list) or not isinstance(output_ports, list):
            issues.append(
                NativeLoweringIssue(
                    node_path,
                    "invalid_ports",
                    "input_ports and output_ports must be arrays.",
                )
            )
            continue
        for direction, ports in (("input", input_ports), ("output", output_ports)):
            names: set[str] = set()
            for index, port in enumerate(ports):
                port_path = f"{node_path}/{direction}_ports/{index}"
                if not isinstance(port, dict) or not isinstance(port.get("name"), str):
                    issues.append(
                        NativeLoweringIssue(port_path, "invalid_port", "Port must be an object with a string name.")
                    )
                    continue
                name = str(port["name"])
                if name in names:
                    issues.append(
                        NativeLoweringIssue(port_path, "duplicate_port_name", f"Duplicate {direction} port name {name!r}.")
                    )
                names.add(name)
        port_counts[node_id] = (len(input_ports), len(output_ports))
        position = node_payload.get("position", [0, 0])
        if (
            not isinstance(position, list)
            or len(position) != 2
            or any(not _is_finite_json_number(value) for value in position)
        ):
            issues.append(
                NativeLoweringIssue(node_path, "invalid_position", "Node position must contain two finite numbers.")
            )
        kind = str(definition.get("kind", "function"))
        if kind == "subgraph":
            child = definition.get("subgraph")
            if isinstance(child, dict):
                _validate_graph_structure(child, graph_path=f"{node_path}/subgraph", issues=issues)
            variant_ref = definition.get("variant_ref")
            if variant_ref is not None:
                if not isinstance(variant_ref, dict):
                    issues.append(
                        NativeLoweringIssue(node_path, "invalid_variant_ref", "variant_ref must be an object.")
                    )
                elif not str(variant_ref.get("family", "")).strip() or not str(variant_ref.get("version", "")).strip():
                    issues.append(
                        NativeLoweringIssue(
                            node_path,
                            "incomplete_variant_ref",
                            "variant_ref requires non-empty family and version fields.",
                        )
                    )

    for boundary_name in ("input_node_ids", "output_node_ids"):
        boundary = graph_payload.get(boundary_name, [])
        boundary_path = f"{graph_path}/{boundary_name}"
        if not isinstance(boundary, list) or any(not isinstance(node_id, str) for node_id in boundary):
            issues.append(
                NativeLoweringIssue(boundary_path, "invalid_graph_interface", f"{boundary_name} must be an array of node IDs.")
            )
            continue
        if len(set(boundary)) != len(boundary):
            issues.append(
                NativeLoweringIssue(boundary_path, "duplicate_graph_interface", f"{boundary_name} contains duplicate node IDs.")
            )
        for node_id in boundary:
            if node_id not in nodes:
                issues.append(
                    NativeLoweringIssue(boundary_path, "missing_interface_node", f"Graph interface references missing node {node_id!r}.")
                )

    for raw_edge_id, edge_payload in edges.items():
        edge_id = str(raw_edge_id)
        edge_path = f"{graph_path}/edges/{_json_pointer_token(edge_id)}"
        if not isinstance(edge_payload, dict):
            issues.append(NativeLoweringIssue(edge_path, "invalid_edge", "Edge must be an object."))
            continue
        if edge_payload.get("id") != edge_id:
            issues.append(
                NativeLoweringIssue(edge_path, "invalid_edge_id", f"Edge key {edge_id!r} must match its string id.")
            )
        src_node = edge_payload.get("src_node")
        dst_node = edge_payload.get("dst_node")
        if not isinstance(src_node, str) or src_node not in nodes:
            issues.append(
                NativeLoweringIssue(edge_path, "missing_edge_source", f"Edge source node {src_node!r} does not exist.")
            )
        if not isinstance(dst_node, str) or dst_node not in nodes:
            issues.append(
                NativeLoweringIssue(edge_path, "missing_edge_destination", f"Edge destination node {dst_node!r} does not exist.")
            )
        for field_name, node_id, count_index in (
            ("src_port", src_node, 1),
            ("dst_port", dst_node, 0),
        ):
            value = edge_payload.get(field_name)
            count = port_counts.get(node_id, (0, 0))[count_index] if isinstance(node_id, str) else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= count:
                issues.append(
                    NativeLoweringIssue(
                        edge_path,
                        "invalid_edge_port",
                        f"{field_name}={value!r} is outside the available port range 0..{max(count - 1, -1)}.",
                    )
                )
        for field_name, default in (("weight", 1.0), ("bias", 0.0)):
            value = edge_payload.get(field_name, default)
            if not _is_finite_json_number(value):
                issues.append(
                    NativeLoweringIssue(edge_path, "invalid_edge_parameter", f"{field_name} must be a finite number.")
                )

    variants = graph_payload.get("variant_library", {})
    if not isinstance(variants, dict):
        issues.append(
            NativeLoweringIssue(f"{graph_path}/variant_library", "invalid_variant_library", "variant_library must be an object.")
        )
        return
    for family, versions in variants.items():
        family_path = f"{graph_path}/variant_library/{_json_pointer_token(str(family))}"
        if not isinstance(family, str) or not family.strip() or not isinstance(versions, dict):
            issues.append(
                NativeLoweringIssue(family_path, "invalid_variant_family", "Variant family must have a non-empty name and map versions to graphs.")
            )
            continue
        for version, variant in versions.items():
            variant_path = f"{family_path}/{_json_pointer_token(str(version))}"
            if not isinstance(version, str) or not version.strip() or not isinstance(variant, dict):
                issues.append(
                    NativeLoweringIssue(variant_path, "invalid_variant", "Variant must have a non-empty version and contain a graph object.")
                )
                continue
            _validate_graph_structure(variant, graph_path=variant_path, issues=issues)


def _scan_and_sanitize_graph_payload(
    graph_payload: dict[str, Any],
    *,
    graph_path: str,
    module_types: set[str],
    function_signatures: set[str],
    issues: list[NativeLoweringIssue],
) -> None:
    nodes = graph_payload.get("nodes", {})
    if not isinstance(nodes, dict):
        issues.append(NativeLoweringIssue(graph_path, "invalid_nodes", "Graph nodes must be an object."))
        return
    input_ids = {str(item) for item in graph_payload.get("input_node_ids", [])}
    output_ids = {str(item) for item in graph_payload.get("output_node_ids", [])}
    for node_id, node_payload in nodes.items():
        node_path = f"{graph_path}/nodes/{_json_pointer_token(str(node_id))}"
        if not isinstance(node_payload, dict) or not isinstance(node_payload.get("neuron_def"), dict):
            issues.append(NativeLoweringIssue(node_path, "invalid_node", "Node is missing a neuron_def object."))
            continue
        definition = node_payload["neuron_def"]
        kind = str(definition.get("kind", "function"))
        if kind == "module":
            operation = str(definition.get("module_type") or definition.get("name") or "")
            if operation not in module_types:
                issues.append(
                    NativeLoweringIssue(
                        node_path,
                        "unsupported_module",
                        f"No Native IR lowerer is registered for module type {operation!r}.",
                        node_kind=kind,
                        operation=operation,
                    )
                )
        elif kind == "function":
            boundary = str(node_id) in input_ids or str(node_id) in output_ids
            signature = _port_signature(definition)
            if not boundary and signature not in function_signatures:
                issues.append(
                    NativeLoweringIssue(
                        node_path,
                        "unsupported_function",
                        "Custom Python functions require an explicit Native IR lowerer.",
                        node_kind=kind,
                        operation=str(definition.get("name", "")),
                    )
                )
            # Never execute graph-supplied Python during native preflight.
            definition["source_code"] = _placeholder_source(
                len(definition.get("output_ports", []) or [])
            )
        elif kind == "subgraph":
            child = definition.get("subgraph")
            if not isinstance(child, dict):
                issues.append(
                    NativeLoweringIssue(
                        node_path,
                        "invalid_subgraph",
                        "Subgraph node has no inline graph after variant resolution.",
                        node_kind=kind,
                    )
                )
            else:
                _scan_and_sanitize_graph_payload(
                    child,
                    graph_path=f"{node_path}/subgraph",
                    module_types=module_types,
                    function_signatures=function_signatures,
                    issues=issues,
                )
        else:
            issues.append(
                NativeLoweringIssue(
                    node_path,
                    "unsupported_node_kind",
                    f"Unsupported neuron kind {kind!r}.",
                    node_kind=kind,
                )
            )
    variants = graph_payload.get("variant_library", {})
    if not isinstance(variants, dict):
        issues.append(
            NativeLoweringIssue(
                f"{graph_path}/variant_library",
                "invalid_variant_library",
                "variant_library must be an object.",
            )
        )
        return
    for family, versions in variants.items():
        if not isinstance(versions, dict):
            issues.append(
                NativeLoweringIssue(
                    f"{graph_path}/variant_library/{_json_pointer_token(str(family))}",
                    "invalid_variant_family",
                    "Variant family must map versions to graphs.",
                )
            )
            continue
        for version, variant in versions.items():
            variant_path = (
                f"{graph_path}/variant_library/{_json_pointer_token(str(family))}/"
                f"{_json_pointer_token(str(version))}"
            )
            if not isinstance(variant, dict):
                issues.append(
                    NativeLoweringIssue(
                        variant_path,
                        "invalid_variant",
                        "Variant entry must be a graph object.",
                    )
                )
                continue
            _scan_and_sanitize_graph_payload(
                variant,
                graph_path=variant_path,
                module_types=module_types,
                function_signatures=function_signatures,
                issues=issues,
            )


_VARIANT_FAMILY_ALIASES: dict[str, tuple[str, ...]] = {
    "gpt2": ("transformer_block",),
    "nanogpt": ("transformer_block",),
    "llama": ("transformer_block",),
    "attn_block": ("transformer_block",),
    "transformer_block": ("attn_block",),
    "mixllama": ("attn_block",),
}


class _RawGraphResolutionError(ValueError):
    def __init__(self, path: str, code: str, message: str) -> None:
        super().__init__(message)
        self.path = path
        self.code = code


def _raw_interface_layout(
    graph_payload: Mapping[str, Any],
    boundary_name: str,
) -> list[tuple[str, dict[str, Any]]]:
    nodes = graph_payload.get("nodes", {})
    layout: list[tuple[str, dict[str, Any]]] = []
    for node_id in graph_payload.get(boundary_name, []):
        node = nodes[node_id]
        definition = node["neuron_def"]
        for port in definition.get("output_ports", []):
            layout.append((str(node_id), deepcopy(port)))
    return layout


def _raw_aliased_ports(
    layout: Sequence[tuple[str, dict[str, Any]]],
    aliases: Sequence[str] | None,
    *,
    path: str,
    direction: str,
) -> list[dict[str, Any]]:
    if aliases is not None and len(aliases) != len(layout):
        raise _RawGraphResolutionError(
            path,
            "variant_alias_mismatch",
            f"Variant has {len(layout)} {direction} ports but received {len(aliases)} aliases.",
        )
    names = list(aliases) if aliases is not None else [
        f"{node_id}.{port.get('name', '')}" for node_id, port in layout
    ]
    ports: list[dict[str, Any]] = []
    for name, (_node_id, port) in zip(names, layout):
        ports.append(
            {
                "name": str(name),
                "range": deepcopy(port.get("range")),
                "precision": port.get("precision"),
                "dtype": port.get("dtype", "float"),
            }
        )
    return ports


def _raw_ports_compatible(
    current: Sequence[Mapping[str, Any]],
    target: Sequence[Mapping[str, Any]],
) -> bool:
    if len(current) != len(target):
        return False
    return all(
        str(left.get("name", "")) == str(right.get("name", ""))
        and left.get("range") == right.get("range")
        and left.get("precision") == right.get("precision")
        and str(left.get("dtype", "float")) == str(right.get("dtype", "float"))
        for left, right in zip(current, target)
    )


def _refresh_raw_subgraph_interface(
    definition: dict[str, Any],
    child: Mapping[str, Any],
    *,
    path: str,
) -> None:
    input_aliases = list(definition.get("input_aliases", []) or []) or None
    output_aliases = list(definition.get("output_aliases", []) or []) or None
    inputs = _raw_aliased_ports(
        _raw_interface_layout(child, "input_node_ids"),
        input_aliases,
        path=path,
        direction="input",
    )
    outputs = _raw_aliased_ports(
        _raw_interface_layout(child, "output_node_ids"),
        output_aliases,
        path=path,
        direction="output",
    )
    definition["input_ports"] = inputs
    definition["output_ports"] = outputs
    definition["input_aliases"] = [str(port["name"]) for port in inputs]
    definition["output_aliases"] = [str(port["name"]) for port in outputs]


def _resolve_raw_variant_library(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve variants with the same port-compatible inline fallback as graph.py."""

    root = deepcopy(dict(payload))
    source_library = deepcopy(dict(root.get("variant_library", {}) or {}))
    resolved_variants: dict[tuple[str, str], dict[str, Any]] = {}
    resolving: list[tuple[str, str]] = []

    def resolve_variant(
        family: str,
        version: str,
        *,
        request_path: str,
    ) -> dict[str, Any]:
        key = (family, version)
        if key in resolved_variants:
            return deepcopy(resolved_variants[key])
        if key in resolving:
            cycle = " -> ".join(f"{fam}@{ver}" for fam, ver in [*resolving, key])
            raise _RawGraphResolutionError(
                request_path,
                "recursive_variant_ref",
                f"Recursive variant reference detected: {cycle}",
            )
        selected: Mapping[str, Any] | None = None
        for candidate_family in (family, *_VARIANT_FAMILY_ALIASES.get(family, ())):
            versions = source_library.get(candidate_family)
            if isinstance(versions, Mapping) and version in versions:
                selected = versions[version]
                break
        if selected is None:
            raise _RawGraphResolutionError(
                request_path,
                "missing_variant",
                f"Missing variant {family!r}@{version!r}.",
            )
        resolving.append(key)
        variant = deepcopy(dict(selected))
        resolve_graph(variant, f"variants/{_json_pointer_token(family)}@{_json_pointer_token(version)}")
        resolving.pop()
        resolved_variants[key] = variant
        return deepcopy(variant)

    def resolve_graph(graph_payload: dict[str, Any], graph_path: str) -> None:
        for node_id, node in graph_payload.get("nodes", {}).items():
            definition = node.get("neuron_def", {})
            if definition.get("kind", "function") != "subgraph":
                continue
            node_path = f"{graph_path}/nodes/{_json_pointer_token(str(node_id))}"
            variant_ref = definition.get("variant_ref") or {}
            if variant_ref:
                family = str(variant_ref.get("family", "")).strip()
                version = str(variant_ref.get("version", "")).strip()
                if not family or not version:
                    raise _RawGraphResolutionError(
                        node_path,
                        "incomplete_variant_ref",
                        "variant_ref requires non-empty family and version fields.",
                    )
                resolved = resolve_variant(family, version, request_path=node_path)
                try:
                    expected_inputs = _raw_aliased_ports(
                        _raw_interface_layout(resolved, "input_node_ids"),
                        list(definition.get("input_aliases", []) or []) or None,
                        path=node_path,
                        direction="input",
                    )
                    expected_outputs = _raw_aliased_ports(
                        _raw_interface_layout(resolved, "output_node_ids"),
                        list(definition.get("output_aliases", []) or []) or None,
                        path=node_path,
                        direction="output",
                    )
                except _RawGraphResolutionError:
                    child = definition.get("subgraph")
                    if isinstance(child, dict):
                        resolve_graph(child, f"{node_path}/subgraph")
                        _refresh_raw_subgraph_interface(definition, child, path=node_path)
                    continue
                current_inputs = definition.get("input_ports", []) or []
                current_outputs = definition.get("output_ports", []) or []
                inputs_ok = not current_inputs or _raw_ports_compatible(current_inputs, expected_inputs)
                outputs_ok = not current_outputs or _raw_ports_compatible(current_outputs, expected_outputs)
                if inputs_ok and outputs_ok:
                    definition["subgraph"] = resolved
                    _refresh_raw_subgraph_interface(definition, resolved, path=node_path)
                else:
                    child = definition.get("subgraph")
                    if isinstance(child, dict):
                        resolve_graph(child, f"{node_path}/subgraph")
                        _refresh_raw_subgraph_interface(definition, child, path=node_path)
                continue
            child = definition.get("subgraph")
            if isinstance(child, dict):
                resolve_graph(child, f"{node_path}/subgraph")
                _refresh_raw_subgraph_interface(definition, child, path=node_path)

    resolved_library: dict[str, dict[str, dict[str, Any]]] = {}
    for family, versions in sorted(source_library.items()):
        resolved_library[str(family)] = {}
        for version in sorted(versions):
            resolved_library[str(family)][str(version)] = resolve_variant(
                str(family),
                str(version),
                request_path=(
                    f"root/variant_library/{_json_pointer_token(str(family))}/"
                    f"{_json_pointer_token(str(version))}"
                ),
            )
    root["variant_library"] = resolved_library
    resolve_graph(root, "root")
    return root


def _prepare_graph_file(path: Path, *, raw_bytes: bytes | None = None) -> _PreparedGraph:
    if raw_bytes is None:
        raw_bytes = path.read_bytes()
    try:
        raw_payload = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _PreparedGraph(
            resolved_payload=None,
            raw_payload={},
            raw_bytes=raw_bytes,
            issues=[NativeLoweringIssue("root", "invalid_graph_json", str(exc))],
            graph_valid=False,
        )
    if not isinstance(raw_payload, dict):
        return _PreparedGraph(
            resolved_payload=None,
            raw_payload={},
            raw_bytes=raw_bytes,
            issues=[NativeLoweringIssue("root", "invalid_graph_json", "Graph JSON must be an object.")],
            graph_valid=False,
        )
    sanitized = deepcopy(raw_payload)
    issues: list[NativeLoweringIssue] = []
    _validate_graph_structure(sanitized, graph_path="root", issues=issues)
    if issues:
        return _PreparedGraph(
            resolved_payload=None,
            raw_payload=raw_payload,
            raw_bytes=raw_bytes,
            issues=issues,
            graph_valid=False,
        )
    module_types, function_signatures = _registered_lowerers()
    _scan_and_sanitize_graph_payload(
        sanitized,
        graph_path="root",
        module_types=module_types,
        function_signatures=function_signatures,
        issues=issues,
    )
    try:
        resolved = _resolve_raw_variant_library(sanitized)
        resolved_issues: list[NativeLoweringIssue] = []
        _validate_graph_structure(resolved, graph_path="root", issues=resolved_issues)
        if resolved_issues:
            issues.extend(resolved_issues)
            raise _RawGraphResolutionError(
                "root",
                "resolved_graph_invalid",
                "Variant resolution produced an invalid graph interface.",
            )
    except _RawGraphResolutionError as exc:
        if not any(issue.path == exc.path and issue.code == exc.code for issue in issues):
            issues.append(NativeLoweringIssue(exc.path, exc.code, str(exc)))
        return _PreparedGraph(
            resolved_payload=None,
            raw_payload=raw_payload,
            raw_bytes=raw_bytes,
            issues=issues,
            graph_valid=False,
        )
    return _PreparedGraph(
        resolved_payload=resolved,
        raw_payload=raw_payload,
        raw_bytes=raw_bytes,
        issues=issues,
        graph_valid=True,
    )


def _port_to_dict(port: Any) -> dict[str, Any]:
    return _json_safe(port.to_dict())


def _node_operation(graph: NeuronGraph, node_id: str) -> str:
    definition = graph.nodes[node_id].neuron_def
    if definition.kind == "module":
        return str(definition.module_type)
    if definition.kind == "subgraph":
        return "subgraph.call"
    if node_id in graph.input_node_ids:
        return "graph.input"
    if node_id in graph.output_node_ids:
        return "graph.output"
    return f"builtin.{definition.name}"


def _lower_graph_topology(graph: NeuronGraph) -> dict[str, Any]:
    graphs: list[dict[str, Any]] = []
    variant_graphs: dict[str, str] = {}
    seen: dict[int, str] = {}

    def lower(current: NeuronGraph, graph_path: str) -> str:
        existing = seen.get(id(current))
        if existing is not None:
            return existing
        seen[id(current)] = graph_path
        lowered_nodes: list[dict[str, Any]] = []
        for node_id in sorted(current.nodes):
            node = current.nodes[node_id]
            definition = node.neuron_def
            node_path = f"{graph_path}/nodes/{_json_pointer_token(node_id)}"
            child_path: str | None = None
            if definition.kind == "subgraph" and definition.subgraph is not None:
                child_path = lower(definition.subgraph, f"{node_path}/subgraph")
            state_checksum = (
                _sha256_bytes(definition.module_state.encode("utf-8"))
                if definition.module_state
                else None
            )
            lowered_nodes.append(
                {
                    "path": node_path,
                    "instance_id": node_id,
                    "name": definition.name,
                    "kind": definition.kind,
                    "operation": _node_operation(current, node_id),
                    "module_config": _json_safe(definition.module_config),
                    "module_state_sha256": state_checksum,
                    "input_ports": [_port_to_dict(port) for port in definition.input_ports],
                    "output_ports": [_port_to_dict(port) for port in definition.output_ports],
                    "position": [float(node.position[0]), float(node.position[1])],
                    "variant_ref": _json_safe(definition.variant_ref),
                    "subgraph": child_path,
                }
            )
        lowered_edges = [
            {
                "path": f"{graph_path}/edges/{_json_pointer_token(edge_id)}",
                "id": edge_id,
                "src_node": f"{graph_path}/nodes/{_json_pointer_token(edge.src_node)}",
                "src_port": int(edge.src_port),
                "dst_node": f"{graph_path}/nodes/{_json_pointer_token(edge.dst_node)}",
                "dst_port": int(edge.dst_port),
                "weight": float(edge.weight),
                "bias": float(edge.bias),
            }
            for edge_id, edge in sorted(current.edges.items())
        ]
        graphs.append(
            {
                "path": graph_path,
                "name": current.name,
                "training_method": current.training_method,
                "runtime": current.runtime,
                "nodes": lowered_nodes,
                "edges": lowered_edges,
                "input_nodes": [
                    f"{graph_path}/nodes/{_json_pointer_token(node_id)}"
                    for node_id in current.input_node_ids
                ],
                "output_nodes": [
                    f"{graph_path}/nodes/{_json_pointer_token(node_id)}"
                    for node_id in current.output_node_ids
                ],
            }
        )
        return graph_path

    lower(graph, "root")
    for family, versions in sorted(graph.variant_library.items()):
        for version, variant in sorted(versions.items()):
            reference = f"{family}@{version}"
            variant_graphs[reference] = lower(
                variant,
                f"variants/{_json_pointer_token(family)}@{_json_pointer_token(version)}",
            )
    graphs.sort(key=lambda item: item["path"])
    return {
        "entry_graph": "root",
        "graphs": graphs,
        "variant_graphs": dict(sorted(variant_graphs.items())),
        "resolved": True,
    }


def _lower_payload_topology(payload: Mapping[str, Any]) -> dict[str, Any]:
    graphs: list[dict[str, Any]] = []
    variant_graphs: dict[str, str] = {}

    def lower(current: Mapping[str, Any], graph_path: str) -> str:
        nodes = current.get("nodes", {})
        input_ids = {str(value) for value in current.get("input_node_ids", [])}
        output_ids = {str(value) for value in current.get("output_node_ids", [])}
        lowered_nodes: list[dict[str, Any]] = []
        for node_id in sorted(nodes):
            node = nodes[node_id]
            definition = node["neuron_def"]
            kind = str(definition.get("kind", "function"))
            node_path = f"{graph_path}/nodes/{_json_pointer_token(str(node_id))}"
            child_path: str | None = None
            child = definition.get("subgraph")
            if kind == "subgraph" and isinstance(child, Mapping):
                child_path = lower(child, f"{node_path}/subgraph")
            if kind == "module":
                operation = str(definition.get("module_type") or definition.get("name") or "")
            elif kind == "subgraph":
                operation = "subgraph.call"
            elif str(node_id) in input_ids:
                operation = "graph.input"
            elif str(node_id) in output_ids:
                operation = "graph.output"
            else:
                operation = f"builtin.{definition.get('name', '')}"
            module_state = str(definition.get("module_state", "") or "")
            lowered_nodes.append(
                {
                    "path": node_path,
                    "instance_id": str(node_id),
                    "name": str(definition.get("name", "")),
                    "kind": kind,
                    "operation": operation,
                    "module_config": _json_safe(definition.get("module_config", {}) or {}),
                    "module_state_sha256": (
                        _sha256_bytes(module_state.encode("utf-8")) if module_state else None
                    ),
                    "input_ports": _json_safe(definition.get("input_ports", []) or []),
                    "output_ports": _json_safe(definition.get("output_ports", []) or []),
                    "position": [float(value) for value in node.get("position", [0, 0])],
                    "variant_ref": _json_safe(definition.get("variant_ref")),
                    "subgraph": child_path,
                }
            )
        lowered_edges: list[dict[str, Any]] = []
        for edge_id, edge in sorted(current.get("edges", {}).items()):
            lowered_edges.append(
                {
                    "path": f"{graph_path}/edges/{_json_pointer_token(str(edge_id))}",
                    "id": str(edge_id),
                    "src_node": (
                        f"{graph_path}/nodes/{_json_pointer_token(str(edge['src_node']))}"
                    ),
                    "src_port": int(edge["src_port"]),
                    "dst_node": (
                        f"{graph_path}/nodes/{_json_pointer_token(str(edge['dst_node']))}"
                    ),
                    "dst_port": int(edge["dst_port"]),
                    "weight": float(edge.get("weight", 1.0)),
                    "bias": float(edge.get("bias", 0.0)),
                }
            )
        graphs.append(
            {
                "path": graph_path,
                "name": str(current.get("name", "graph")),
                "training_method": str(current.get("training_method", "surrogate")),
                "runtime": str(current.get("runtime", "scalar")),
                "nodes": lowered_nodes,
                "edges": lowered_edges,
                "input_nodes": [
                    f"{graph_path}/nodes/{_json_pointer_token(str(node_id))}"
                    for node_id in current.get("input_node_ids", [])
                ],
                "output_nodes": [
                    f"{graph_path}/nodes/{_json_pointer_token(str(node_id))}"
                    for node_id in current.get("output_node_ids", [])
                ],
            }
        )
        return graph_path

    lower(payload, "root")
    for family, versions in sorted((payload.get("variant_library", {}) or {}).items()):
        for version, variant in sorted(versions.items()):
            reference = f"{family}@{version}"
            variant_graphs[reference] = lower(
                variant,
                f"variants/{_json_pointer_token(str(family))}@{_json_pointer_token(str(version))}",
            )
    graphs.sort(key=lambda item: item["path"])
    return {
        "entry_graph": "root",
        "graphs": graphs,
        "variant_graphs": dict(sorted(variant_graphs.items())),
        "resolved": True,
    }


def _template_spec(graph: NeuronGraph) -> dict[str, Any]:
    return dict((graph.torch_config or {}).get("template_spec", {}) or {})


def _model_identity(graph: NeuronGraph) -> dict[str, Any]:
    spec = _template_spec(graph)
    template = dict(spec.get("template", {}) or {})
    block = dict(spec.get("block_spec", {}) or {})
    objective = str(template.get("objective") or "unknown")
    backbone = str(template.get("backbone") or block.get("family") or "unknown")
    attention_variant = str(block.get("attention_variant") or "dense")
    compression = str(template.get("compression") or block.get("compression") or "none")
    if objective == "diffusion":
        family_class = "text_diffusion"
    elif objective == "seq2seq":
        family_class = "seq2seq"
    elif compression == "kv_pca":
        family_class = "pca_kv"
    elif attention_variant == "mla":
        family_class = "latent_kv"
    elif backbone in {"jamba", "ttt", "universal", "hnet"}:
        family_class = "hybrid_state"
    elif backbone in {"gpt2", "nanogpt", "llama", "mixllama"}:
        family_class = "autoregressive_transformer"
    else:
        family_class = "unknown"
    return {
        "name": graph.name,
        "family": backbone,
        "backbone": backbone,
        "family_class": family_class,
        "objective": objective,
        "template_spec": _json_safe(spec),
    }


def _model_identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    torch_config = dict(payload.get("torch_config", {}) or {})
    spec = dict(torch_config.get("template_spec", {}) or {})
    template = dict(spec.get("template", {}) or {})
    block = dict(spec.get("block_spec", {}) or {})
    objective = str(template.get("objective") or "unknown")
    backbone = str(template.get("backbone") or block.get("family") or "unknown")
    attention_variant = str(block.get("attention_variant") or "dense")
    compression = str(template.get("compression") or block.get("compression") or "none")
    if objective == "diffusion":
        family_class = "text_diffusion"
    elif objective == "seq2seq":
        family_class = "seq2seq"
    elif compression == "kv_pca":
        family_class = "pca_kv"
    elif attention_variant == "mla":
        family_class = "latent_kv"
    elif backbone in {"jamba", "ttt", "universal", "hnet"}:
        family_class = "hybrid_state"
    elif backbone in {"gpt2", "nanogpt", "llama", "mixllama"}:
        family_class = "autoregressive_transformer"
    else:
        family_class = "unknown"
    return {
        "name": str(payload.get("name", "graph")),
        "family": backbone,
        "backbone": backbone,
        "family_class": family_class,
        "objective": objective,
        "template_spec": _json_safe(spec),
    }


def _session_state_kinds(model: Mapping[str, Any]) -> tuple[str, ...]:
    family = str(model.get("family", ""))
    family_class = str(model.get("family_class", ""))
    if family_class == "text_diffusion":
        return ("denoising",)
    if family_class == "seq2seq":
        return ("encoder", "decoder_kv", "cross_kv")
    if family_class == "pca_kv":
        return ("pca_kv",)
    if family_class == "latent_kv":
        return ("latent_kv",)
    if family == "jamba":
        return ("attention_kv", "mamba")
    if family == "ttt":
        return ("attention_kv", "ttt")
    if family == "universal":
        return ("attention_kv", "recurrent", "act")
    if family == "hnet":
        return ("attention_kv", "byte_patch")
    if family_class == "autoregressive_transformer":
        return ("kv",)
    return ()


def _is_dense_moa_model(model: Mapping[str, Any]) -> bool:
    if str(model.get("family") or "").strip().lower().replace("-", "_") != "gpt2":
        return False
    template_spec = model.get("template_spec")
    if not isinstance(template_spec, Mapping):
        return False
    block_spec = template_spec.get("block_spec")
    if not isinstance(block_spec, Mapping):
        return False
    return (
        str(block_spec.get("activation_mode") or "")
        .strip()
        .lower()
        .replace("-", "_")
        == "moa"
    )


def _is_dense_differential_model(model: Mapping[str, Any]) -> bool:
    family = str(model.get("family") or "").strip().lower().replace("-", "_")
    if family != "gpt2":
        return False
    template_spec = model.get("template_spec")
    if not isinstance(template_spec, Mapping):
        return False
    block_spec = template_spec.get("block_spec")
    if not isinstance(block_spec, Mapping):
        return False
    return (
        str(block_spec.get("attention_variant") or "")
        .strip()
        .lower()
        .replace("-", "_")
        == "differential"
    )


def _moa_checkpoint_contract_ready(
    proof: Any,
    checkpoint: Mapping[str, Any],
) -> bool:
    evidence = tuple(str(value) for value in getattr(proof, "evidence", ()))
    if "native-dense-moa-source-bound-checkpoint-v1" not in evidence:
        return True
    source_graph = checkpoint.get("source_graph")
    moa = checkpoint.get("moa")
    if not isinstance(source_graph, Mapping) or not isinstance(moa, Mapping):
        return False
    source_sha256 = source_graph.get("sha256")
    metadata_sha256 = moa.get("metadata_sha256")
    metadata_artifact_path = moa.get("metadata_artifact_path")
    metadata_nbytes = moa.get("metadata_nbytes")
    def valid_sha256(value: Any) -> bool:
        return bool(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )
    return bool(
        moa.get("schema") == "neuralfn.native_dense_moa.inference_checkpoint"
        and moa.get("version") == 1
        and moa.get("preset") == "gpt2_moa"
        and moa.get("selected_activation") in {"gelu", "relu", "silu", "relu2"}
        and moa.get("candidate_activations")
        == ["gelu", "relu", "silu", "relu2"]
        and isinstance(moa.get("interval"), int)
        and not isinstance(moa.get("interval"), bool)
        and moa.get("interval") > 0
        and valid_sha256(source_sha256)
        and moa.get("source_graph_sha256") == source_sha256
        and valid_sha256(metadata_sha256)
        and isinstance(metadata_artifact_path, str)
        and bool(metadata_artifact_path)
        and Path(metadata_artifact_path).name == metadata_artifact_path
        and metadata_artifact_path not in {".", ".."}
        and "\\" not in metadata_artifact_path
        and metadata_artifact_path != checkpoint.get("artifact_path")
        and isinstance(metadata_nbytes, int)
        and not isinstance(metadata_nbytes, bool)
        and metadata_nbytes > 0
        and source_graph.get("byte_identity_verified") is True
    )


def _resident_artifact_ready(
    proof: Any,
    checkpoint: Mapping[str, Any] | None,
) -> bool:
    if not proof.resident_inference_proven or not isinstance(checkpoint, Mapping):
        return False
    checkpoint_format = checkpoint.get("format")
    model_family = str(getattr(proof, "model_family", "") or "")
    format_matches_adapter = (
        checkpoint_format == "neuralfn.native_dense_gpt.v5"
        and model_family in {"gpt", "gpt2", "gpt3", "nanogpt", "gpt2-evo"}
    ) or (
        checkpoint_format == "neuralfn.native_family_llama.f32.v1"
        and model_family == "llama"
    ) or (
        checkpoint_format == "neuralfn.native_family_standard_moe.f32.v1"
        and model_family == "mixllama"
    )
    return (
        format_matches_adapter
        and _moa_checkpoint_contract_ready(proof, checkpoint)
        and isinstance(checkpoint.get("artifact_path"), str)
        and bool(str(checkpoint.get("artifact_path") or "").strip())
    )


def _turboquant_tile_attention_artifact_ready(
    proof: Any,
    checkpoint: Mapping[str, Any] | None,
) -> bool:
    if (
        not _resident_artifact_ready(proof, checkpoint)
        or not proof.turboquant_cache_proven
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("format") != "neuralfn.native_dense_gpt.v5"
    ):
        return False
    geometry = checkpoint.get("geometry")
    if not isinstance(geometry, Mapping):
        return False
    values = (
        geometry.get("max_seq_len"),
        geometry.get("num_heads"),
        geometry.get("channels"),
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        return False
    max_seq_len, num_heads, channels = values
    if (
        max_seq_len <= 0
        or max_seq_len > 16_384
        or num_heads <= 0
        or channels <= 0
        or channels % num_heads
    ):
        return False
    head_dim = channels // num_heads
    return 2 <= head_dim <= 256 and head_dim % 2 == 0


def _session_prefix_cow_artifact_profile(
    proof: Any,
    checkpoint: Mapping[str, Any] | None,
) -> str | None:
    """Return the exact reviewed full-cache COW profile for this adapter."""

    if (
        not _resident_artifact_ready(proof, checkpoint)
        or not isinstance(checkpoint, Mapping)
        or not proof.lossless_cache_proven
    ):
        return None
    checkpoint_format = checkpoint.get("format")
    model_family = str(getattr(proof, "model_family", "") or "")
    if (
        checkpoint_format == "neuralfn.native_dense_gpt.v5"
        and model_family in {"gpt", "gpt2", "gpt3", "nanogpt", "gpt2-evo"}
    ):
        return SESSION_PREFIX_COW_PROFILE
    if (
        checkpoint_format == "neuralfn.native_family_llama.f32.v1"
        and model_family == "llama"
    ):
        return LLAMA_SESSION_PREFIX_COW_PROFILE
    if (
        checkpoint_format == "neuralfn.native_family_standard_moe.f32.v1"
        and model_family == "mixllama"
    ):
        return STANDARD_MOE_SESSION_PREFIX_COW_PROFILE
    return None


def _session_prefix_cow_artifact_ready(
    proof: Any,
    checkpoint: Mapping[str, Any] | None,
) -> bool:
    return _session_prefix_cow_artifact_profile(proof, checkpoint) is not None


def _turboquant_session_prefix_cow_artifact_profile(
    proof: Any,
    checkpoint: Mapping[str, Any] | None,
) -> str | None:
    """Return the reviewed dense CPU packed-TurboQuant COW profile."""

    if (
        not _resident_artifact_ready(proof, checkpoint)
        or not isinstance(checkpoint, Mapping)
        or not proof.lossless_cache_proven
        or not proof.turboquant_cache_proven
    ):
        return None
    if (
        checkpoint.get("format") == "neuralfn.native_dense_gpt.v5"
        and str(getattr(proof, "model_family", "") or "")
        in {"gpt", "gpt2", "gpt3", "nanogpt", "gpt2-evo"}
    ):
        return CPU_TURBOQUANT_SESSION_PREFIX_COW_PROFILE
    return None


def _structured_output_metadata_ready(
    tokenizer: Mapping[str, Any] | None,
    *,
    resident_ready: bool,
) -> bool:
    constrained = (
        tokenizer.get("constrained_decoding")
        if isinstance(tokenizer, Mapping)
        else None
    )
    return bool(
        resident_ready
        and isinstance(constrained, Mapping)
        and set(constrained) == {"version", "profile", "token_selection"}
        and type(constrained.get("version")) is int
        and constrained.get("version") == 1
        and constrained.get("profile") == STRUCTURED_OUTPUT_PROFILE
        and constrained.get("token_selection") == STRUCTURED_OUTPUT_TOKEN_SELECTION
    )


def _function_tool_metadata_ready(
    chat_template: Mapping[str, Any] | None,
    *,
    structured_output_ready: bool,
) -> bool:
    tool_template = (
        chat_template.get("tool_template")
        if isinstance(chat_template, Mapping)
        else None
    )
    return bool(
        structured_output_ready
        and isinstance(tool_template, Mapping)
        and set(tool_template) == {"version", "profile"}
        and type(tool_template.get("version")) is int
        and tool_template.get("version") == 1
        and tool_template.get("profile") == FUNCTION_TOOL_TEMPLATE_PROFILE
    )


def _capabilities(
    proof: Any,
    *,
    compatible: bool,
    checkpoint: Mapping[str, Any] | None = None,
    tokenizer: Mapping[str, Any] | None = None,
    chat_template: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    resident_ready = bool(compatible and _resident_artifact_ready(proof, checkpoint))
    tile_turboquant_ready = bool(
        compatible and _turboquant_tile_attention_artifact_ready(proof, checkpoint)
    )
    session_prefix_cow_ready = bool(
        compatible and _session_prefix_cow_artifact_ready(proof, checkpoint)
    )
    turboquant_session_prefix_cow_ready = bool(
        compatible
        and _turboquant_session_prefix_cow_artifact_profile(proof, checkpoint)
        is not None
    )
    structured_output_ready = _structured_output_metadata_ready(
        tokenizer,
        resident_ready=resident_ready,
    )
    function_tools_ready = _function_tool_metadata_ready(
        chat_template,
        structured_output_ready=structured_output_ready,
    )
    return {
        "native_ir": bool(compatible and proof.native_ir_lowering),
        "native_training": bool(
            compatible
            and proof.trainer_registered
            and proof.architecture_persistence_proven
        ),
        "architecture_persistence": bool(
            compatible and proof.architecture_persistence_proven
        ),
        "native_inference": bool(compatible and proof.native_forward_proven),
        "resident_inference": resident_ready,
        "lossless_kv_cache": bool(resident_ready and proof.lossless_cache_proven),
        "turboquant_kv_cache": bool(resident_ready and proof.turboquant_cache_proven),
        "turboquant_tile_attention": tile_turboquant_ready,
        "session_prefix_cow": session_prefix_cow_ready,
        "session_prefix_cow_cpu_turboquant": turboquant_session_prefix_cow_ready,
        "serve": bool(resident_ready and proof.serving_proven),
        "function_tools": function_tools_ready,
        "structured_output": structured_output_ready,
    }


def _kernel_abi(
    proof: Any,
    checkpoint: Mapping[str, Any] | None,
    *,
    tokenizer: Mapping[str, Any] | None = None,
    chat_template: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resident_ready = _resident_artifact_ready(proof, checkpoint)
    tile_turboquant_ready = _turboquant_tile_attention_artifact_ready(
        proof,
        checkpoint,
    )
    session_prefix_cow_profile = _session_prefix_cow_artifact_profile(proof, checkpoint)
    session_prefix_cow_ready = session_prefix_cow_profile is not None
    turboquant_session_prefix_cow_profile = (
        _turboquant_session_prefix_cow_artifact_profile(proof, checkpoint)
    )
    turboquant_session_prefix_cow_ready = (
        turboquant_session_prefix_cow_profile is not None
    )
    structured_output_ready = _structured_output_metadata_ready(
        tokenizer,
        resident_ready=resident_ready,
    )
    function_tools_ready = _function_tool_metadata_ready(
        chat_template,
        structured_output_ready=structured_output_ready,
    )
    return {
        "tile_ops": {"symbol": "nfn_native_tile_ops_abi_version", "version": 1},
        "strict_math": {
            "symbol": "nfn_native_tile_strict_math_abi_version",
            "version": 1,
        },
        "resident_inference": {
            "version": 1 if resident_ready else None,
            "status": "ready" if resident_ready else "not_implemented",
        },
        "turboquant_cache": {
            "version": 1 if resident_ready and proof.turboquant_cache_proven else None,
            "status": (
                "ready"
                if resident_ready and proof.turboquant_cache_proven
                else "not_implemented"
            ),
            "backend": (
                "cpu-reference-packed"
                if resident_ready and proof.turboquant_cache_proven
                else None
            ),
        },
        "turboquant_tile_attention": {
            "symbol": "nfn_native_tile_turboquant_attention_forward_v1",
            "version": 1 if tile_turboquant_ready else None,
            "status": "ready" if tile_turboquant_ready else "not_implemented",
            "backend": "tile-cuda-hybrid" if tile_turboquant_ready else None,
        },
        "session_prefix_cow": {
            "version": 1 if session_prefix_cow_ready else None,
            "status": "ready" if session_prefix_cow_ready else "not_implemented",
            "profile": session_prefix_cow_profile,
            "operation": "fork_session" if session_prefix_cow_ready else None,
        },
        "session_prefix_cow_cpu_turboquant": {
            "version": 1 if turboquant_session_prefix_cow_ready else None,
            "status": (
                "ready"
                if turboquant_session_prefix_cow_ready
                else "not_implemented"
            ),
            "profile": turboquant_session_prefix_cow_profile,
            "operation": (
                "fork_session" if turboquant_session_prefix_cow_ready else None
            ),
            "backend": (
                "cpu-reference-packed"
                if turboquant_session_prefix_cow_ready
                else None
            ),
        },
        "structured_output": {
            "version": 1 if structured_output_ready else None,
            "status": "ready" if structured_output_ready else "not_implemented",
            "profile": STRUCTURED_OUTPUT_PROFILE if structured_output_ready else None,
            "token_selection": (
                STRUCTURED_OUTPUT_TOKEN_SELECTION if structured_output_ready else None
            ),
        },
        "function_tools": {
            "version": 1 if function_tools_ready else None,
            "status": "ready" if function_tools_ready else "not_implemented",
            "profile": FUNCTION_TOOL_TEMPLATE_PROFILE if function_tools_ready else None,
            "structured_output_profile": (
                STRUCTURED_OUTPUT_PROFILE if function_tools_ready else None
            ),
        },
    }


def _walk_graphs(graph: NeuronGraph) -> Iterable[NeuronGraph]:
    seen: set[int] = set()

    def visit(current: NeuronGraph) -> Iterable[NeuronGraph]:
        if id(current) in seen:
            return
        seen.add(id(current))
        yield current
        for node in current.nodes.values():
            child = node.neuron_def.subgraph
            if child is not None:
                yield from visit(child)
        for versions in current.variant_library.values():
            for variant in versions.values():
                yield from visit(variant)

    yield from visit(graph)


def _context_limits(graph: NeuronGraph, model: Mapping[str, Any]) -> dict[str, Any]:
    spec = dict(model.get("template_spec", {}) or {})
    candidates: list[int] = []
    for key in ("context_window", "max_seq_len"):
        value = spec.get(key)
        if value not in (None, ""):
            try:
                candidates.append(int(value))
            except (TypeError, ValueError):
                pass
    return {
        "max_context_tokens": max(candidates) if candidates else None,
        "max_output_tokens": None,
    }


def _context_limits_payload(model: Mapping[str, Any]) -> dict[str, Any]:
    spec = dict(model.get("template_spec", {}) or {})
    candidates: list[int] = []
    for key in ("context_window", "max_seq_len"):
        value = spec.get(key)
        if value not in (None, ""):
            try:
                candidates.append(int(value))
            except (TypeError, ValueError):
                pass
    return {
        "max_context_tokens": max(candidates) if candidates else None,
        "max_output_tokens": None,
    }


def _tokenizer_metadata(graph: NeuronGraph, model: Mapping[str, Any]) -> dict[str, Any]:
    torch_config = dict(graph.torch_config or {})
    manifest = dict(torch_config.get("tokenizer_manifest", {}) or {})
    template = dict((model.get("template_spec", {}) or {}).get("template", {}) or {})
    if "tokenization" not in manifest and template.get("tokenization"):
        manifest["tokenization"] = template["tokenization"]
    return _json_safe(manifest)


def _tokenizer_metadata_payload(
    payload: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    torch_config = dict(payload.get("torch_config", {}) or {})
    manifest = dict(torch_config.get("tokenizer_manifest", {}) or {})
    template = dict((model.get("template_spec", {}) or {}).get("template", {}) or {})
    if "tokenization" not in manifest and template.get("tokenization"):
        manifest["tokenization"] = template["tokenization"]
    return _json_safe(manifest)


def _chat_template_metadata(graph: NeuronGraph, tokenizer: Mapping[str, Any]) -> dict[str, Any]:
    torch_config = dict(graph.torch_config or {})
    artifact = dict(torch_config.get("artifact_metadata", {}) or {})
    value = (
        torch_config.get("chat_template")
        or artifact.get("chat_template")
        or tokenizer.get("chat_template")
    )
    tool_template = artifact.get("tool_template") or tokenizer.get("tool_template")
    return {
        "source": "artifact" if value else "missing",
        "template": _json_safe(value),
        "tool_template": _json_safe(tool_template),
    }


def _chat_template_metadata_payload(
    payload: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
) -> dict[str, Any]:
    torch_config = dict(payload.get("torch_config", {}) or {})
    artifact = dict(torch_config.get("artifact_metadata", {}) or {})
    value = (
        torch_config.get("chat_template")
        or artifact.get("chat_template")
        or tokenizer.get("chat_template")
    )
    tool_template = artifact.get("tool_template") or tokenizer.get("tool_template")
    return {
        "source": "artifact" if value else "missing",
        "template": _json_safe(value),
        "tool_template": _json_safe(tool_template),
    }


def _stop_tokens(tokenizer: Mapping[str, Any]) -> tuple[int, ...]:
    tokens: set[int] = set()
    # BOS and padding IDs are tokenizer controls, not generation stop tokens.
    for key in ("eos_token_id",):
        value = tokenizer.get(key)
        if value is not None:
            try:
                tokens.add(int(value))
            except (TypeError, ValueError):
                pass
    for key in ("stop_token_ids", "eos_token_ids"):
        values = tokenizer.get(key, ())
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for value in values:
                try:
                    tokens.add(int(value))
                except (TypeError, ValueError):
                    pass
    return tuple(sorted(tokens))


def compile_graph_to_native_manifest(
    graph: NeuronGraph,
    *,
    source_path: str | Path | None = None,
    graph_fingerprint: str | None = None,
    tensors: Sequence[NativeTensorSpec] = (),
    checkpoint: Mapping[str, Any] | None = None,
) -> NativeExecutionManifest:
    """Lower a validated, resolved graph into Native Execution IR v1."""

    # Variant resolution refreshes subgraph interfaces in place.  Native IR
    # compilation is additive, so never mutate the caller's authoring graph.
    source_graph = deepcopy(graph)
    working_graph = deepcopy(graph)
    working_graph.resolve_variant_library()
    working_graph.validate()
    topology = _lower_graph_topology(working_graph)
    from .native_registry import capability_proof_for, classify_native_model

    model = _model_identity(working_graph)
    classification = classify_native_model(model, topology)
    model = {
        **model,
        "family": classification["model_family"],
        "family_class": classification["family_class"],
        "objective": classification["objective"],
        "backbone": classification["backbone"],
        "compression": classification["compression"],
        "text_generation": classification["text_generation"],
        "turboquant_policy": classification["turboquant_policy"],
    }
    proof = capability_proof_for(model, topology)
    tokenizer = _tokenizer_metadata(working_graph, model)
    chat_template = _chat_template_metadata(working_graph, tokenizer)
    source = {
        "path": str(Path(source_path).expanduser().resolve()) if source_path is not None else None,
        "sha256": graph_fingerprint or _sha256_bytes(_canonical_json_bytes(source_graph.to_dict())),
        "serialization_changed": False,
    }
    return NativeExecutionManifest(
        source_graph=source,
        model=model,
        topology=topology,
        tensors=tuple(tensors),
        tokenizer=tokenizer,
        chat_template=chat_template,
        context_limits=_context_limits(working_graph, model),
        stop_tokens=_stop_tokens(tokenizer),
        kernel_abi=_kernel_abi(
            proof,
            checkpoint,
            tokenizer=tokenizer,
            chat_template=chat_template,
        ),
        checkpoint=dict(checkpoint) if checkpoint is not None else None,
        session_state_kinds=proof.session_state_kinds,
        capabilities=_capabilities(
            proof,
            compatible=True,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            chat_template=chat_template,
        ),
    )


def _manifest_from_resolved_payload(
    resolved_payload: Mapping[str, Any],
    *,
    source_path: str | Path | None,
    graph_fingerprint: str,
    tensors: Sequence[NativeTensorSpec] = (),
    checkpoint: Mapping[str, Any] | None = None,
) -> NativeExecutionManifest:
    from .native_registry import capability_proof_for, classify_native_model

    topology = _lower_payload_topology(resolved_payload)
    model = _model_identity_payload(resolved_payload)
    classification = classify_native_model(model, topology)
    model = {
        **model,
        "family": classification["model_family"],
        "family_class": classification["family_class"],
        "objective": classification["objective"],
        "backbone": classification["backbone"],
        "compression": classification["compression"],
        "text_generation": classification["text_generation"],
        "turboquant_policy": classification["turboquant_policy"],
    }
    proof = capability_proof_for(model, topology)
    tokenizer = _tokenizer_metadata_payload(resolved_payload, model)
    chat_template = _chat_template_metadata_payload(resolved_payload, tokenizer)
    source = {
        "path": str(Path(source_path).expanduser().resolve()) if source_path is not None else None,
        "sha256": graph_fingerprint,
        "serialization_changed": False,
    }
    return NativeExecutionManifest(
        source_graph=source,
        model=model,
        topology=topology,
        tensors=tuple(tensors),
        tokenizer=tokenizer,
        chat_template=chat_template,
        context_limits=_context_limits_payload(model),
        stop_tokens=_stop_tokens(tokenizer),
        kernel_abi=_kernel_abi(
            proof,
            checkpoint,
            tokenizer=tokenizer,
            chat_template=chat_template,
        ),
        checkpoint=dict(checkpoint) if checkpoint is not None else None,
        session_state_kinds=proof.session_state_kinds,
        capabilities=_capabilities(
            proof,
            compatible=True,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            chat_template=chat_template,
        ),
    )


def compile_native_graph_payload(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> NativeExecutionManifest:
    """Safely compile a serialized graph mapping without executing its source.

    Unlike :func:`compile_graph_to_native_manifest`, this entrypoint never
    constructs ``NeuronGraph``/``NeuronDef`` objects and therefore does not
    require graph-analysis, NumPy, or Torch packages.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("Graph payload must be a mapping.")
    original = deepcopy(dict(payload))
    candidate = deepcopy(original)
    issues: list[NativeLoweringIssue] = []
    _validate_graph_structure(candidate, graph_path="root", issues=issues)
    if not issues:
        module_types, function_signatures = _registered_lowerers()
        _scan_and_sanitize_graph_payload(
            candidate,
            graph_path="root",
            module_types=module_types,
            function_signatures=function_signatures,
            issues=issues,
        )
    if issues:
        first = issues[0]
        raise ValueError(f"{first.code} at {first.path}: {first.message}")
    resolved = _resolve_raw_variant_library(candidate)
    resolved_issues: list[NativeLoweringIssue] = []
    _validate_graph_structure(resolved, graph_path="root", issues=resolved_issues)
    if resolved_issues:
        first = resolved_issues[0]
        raise ValueError(f"{first.code} at {first.path}: {first.message}")
    return _manifest_from_resolved_payload(
        resolved,
        source_path=source_path,
        graph_fingerprint=_sha256_bytes(_canonical_json_bytes(original)),
    )


def _native_tensor_name(source_name: str) -> str:
    components = [component for component in re.split(r"[./]+", source_name) if component]
    safe = [re.sub(r"[^A-Za-z0-9_-]+", "_", component) for component in components]
    return "parameters/" + "/".join(safe)


def _load_pt_tensor_bundle(
    path: Path,
) -> tuple[tuple[NativeTensorSpec, ...], bytes, dict[str, Any]]:
    worker = Path(__file__).with_name("_pt_migrate_worker.py")
    with tempfile.TemporaryDirectory(prefix="neuralfn-pt-migrate-") as temp_root:
        worker_output = Path(temp_root) / "converted"
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                str(worker),
                "--input",
                str(path),
                "--output-dir",
                str(worker_output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            if "No module named 'torch'" in detail or "requires PyTorch" in detail:
                raise ImportError(
                    "Migrating legacy .pt weights requires a separately installed PyTorch. "
                    "The default NeuralFn package remains Torch-free."
                )
            raise RuntimeError(
                "The isolated .pt migration worker rejected the checkpoint"
                + (f": {detail}" if detail else ".")
            )
        descriptor_path = worker_output / "result.json"
        bundle_path = worker_output / "weights.bin"
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        if descriptor.get("schema") != "neuralfn.pt_migration_worker_result":
            raise ValueError("The isolated .pt migration worker returned an unknown schema.")
        if int(descriptor.get("version", 0)) != 1:
            raise ValueError("The isolated .pt migration worker returned an unsupported version.")
        bundle_bytes = bundle_path.read_bytes()
        checkpoint = dict(descriptor.get("checkpoint", {}) or {})
        if checkpoint.get("target_format") != NATIVE_TENSOR_BUNDLE_FORMAT:
            raise ValueError("The isolated .pt migration worker returned an unknown bundle format.")
        if int(checkpoint.get("target_nbytes", -1)) != len(bundle_bytes):
            raise ValueError("The isolated .pt migration worker returned an invalid bundle length.")
        if checkpoint.get("target_sha256") != _sha256_bytes(bundle_bytes):
            raise ValueError("The isolated .pt migration worker returned an invalid bundle checksum.")
        if checkpoint.get("source_sha256") != _sha256_file(path):
            raise ValueError("The source checkpoint changed during isolated migration.")
        specs = tuple(
            NativeTensorSpec.from_dict(item)
            for item in descriptor.get("tensors", [])
        )
        occupied_end = 0
        for tensor in specs:
            if tensor.offset < 0 or tensor.nbytes < 0:
                raise ValueError(f"Invalid tensor bounds for {tensor.name!r} in migrated bundle.")
            if any(dim < 0 for dim in tensor.shape):
                raise ValueError(f"Invalid tensor shape for {tensor.name!r} in migrated bundle.")
            if tensor.offset < occupied_end or tensor.offset % 64:
                raise ValueError(f"Invalid tensor offset for {tensor.name!r} in migrated bundle.")
            end = tensor.offset + tensor.nbytes
            if end > len(bundle_bytes):
                raise ValueError(f"Tensor {tensor.name!r} exceeds the migrated bundle.")
            if _sha256_bytes(bundle_bytes[tensor.offset:end]) != tensor.sha256:
                raise ValueError(f"Tensor {tensor.name!r} failed migrated checksum validation.")
            occupied_end = end
        return specs, bundle_bytes, checkpoint


def _load_native_dense_checkpoint(
    path: Path,
    manifest: NativeExecutionManifest,
) -> tuple[tuple[NativeTensorSpec, ...], dict[str, Any], dict[str, Any]]:
    from .native_dense_checkpoint import inspect_native_dense_checkpoint

    info = inspect_native_dense_checkpoint(path)
    info.validate_model(manifest.model)
    declared_context = manifest.context_limits.get("max_context_tokens")
    if declared_context not in (None, "") and int(declared_context) != info.max_seq_len:
        raise ValueError(
            f"Graph max_context_tokens={int(declared_context)} does not match native checkpoint "
            f"max_seq_len={info.max_seq_len}."
        )
    tensors = tuple(NativeTensorSpec.from_dict(tensor.to_dict()) for tensor in info.tensors)
    checkpoint = info.checkpoint_descriptor(artifact_path="model.bin")
    context_limits = {
        **manifest.context_limits,
        "max_context_tokens": info.max_seq_len,
    }
    return tensors, checkpoint, context_limits


def _load_native_moa_checkpoint(
    path: Path,
    manifest: NativeExecutionManifest,
    *,
    graph_fingerprint: str,
    source_graph_path: Path,
) -> tuple[tuple[NativeTensorSpec, ...], dict[str, Any], dict[str, Any], Path]:
    from .native_moa_checkpoint import inspect_native_moa_checkpoint

    info = inspect_native_moa_checkpoint(
        path,
        source_graph_path=source_graph_path,
        model=manifest.model,
    )
    if info.source_graph_sha256 != graph_fingerprint:
        raise ValueError(
            "Native dense MoA checkpoint source graph SHA-256 does not match "
            "the graph supplied for migration."
        )
    declared_context = manifest.context_limits.get("max_context_tokens")
    if declared_context not in (None, "") and int(declared_context) != info.max_seq_len:
        raise ValueError(
            f"Graph max_context_tokens={int(declared_context)} does not match native dense "
            f"MoA checkpoint max_seq_len={info.max_seq_len}."
        )
    tensors = tuple(
        NativeTensorSpec.from_dict(tensor.to_dict()) for tensor in info.tensors
    )
    checkpoint = info.checkpoint_descriptor(artifact_path="model.bin")
    context_limits = {
        **manifest.context_limits,
        "max_context_tokens": info.max_seq_len,
    }
    return tensors, checkpoint, context_limits, info.model_path


def _load_native_family_llama_checkpoint(
    path: Path,
    manifest: NativeExecutionManifest,
    *,
    graph_fingerprint: str,
) -> tuple[tuple[NativeTensorSpec, ...], dict[str, Any], dict[str, Any], Path]:
    from .native_family_checkpoint import inspect_native_family_llama_checkpoint

    info = inspect_native_family_llama_checkpoint(path)
    info.validate_model(manifest.model)
    source_graph = info.training.get("source_graph")
    if isinstance(source_graph, Mapping) and source_graph.get("sha256") != graph_fingerprint:
        raise ValueError(
            "Native family LLaMA checkpoint source graph SHA-256 does not match "
            "the graph supplied for migration."
        )
    declared_context = manifest.context_limits.get("max_context_tokens")
    if declared_context not in (None, "") and int(declared_context) != info.max_seq_len:
        raise ValueError(
            f"Graph max_context_tokens={int(declared_context)} does not match native family "
            f"LLaMA checkpoint max_seq_len={info.max_seq_len}."
        )
    tensors = tuple(NativeTensorSpec.from_dict(tensor.to_dict()) for tensor in info.tensors)
    checkpoint = info.checkpoint_descriptor(artifact_path="model.f32")
    context_limits = {
        **manifest.context_limits,
        "max_context_tokens": info.max_seq_len,
    }
    return tensors, checkpoint, context_limits, info.artifact_path


def _load_native_family_standard_moe_checkpoint(
    path: Path,
    manifest: NativeExecutionManifest,
    *,
    graph_fingerprint: str,
) -> tuple[tuple[NativeTensorSpec, ...], dict[str, Any], dict[str, Any], Path]:
    from .native_moe_checkpoint import inspect_native_family_standard_moe_checkpoint

    info = inspect_native_family_standard_moe_checkpoint(path)
    info.validate_model(manifest.model)
    source_graph = info.training.get("source_graph")
    if not isinstance(source_graph, Mapping) or source_graph.get("sha256") != graph_fingerprint:
        raise ValueError(
            "Native family standard-MoE checkpoint source graph SHA-256 does not match "
            "the graph supplied for migration."
        )
    declared_context = manifest.context_limits.get("max_context_tokens")
    if declared_context not in (None, "") and int(declared_context) != info.max_seq_len:
        raise ValueError(
            f"Graph max_context_tokens={int(declared_context)} does not match native family "
            f"standard-MoE checkpoint max_seq_len={info.max_seq_len}."
        )
    tensors = tuple(NativeTensorSpec.from_dict(tensor.to_dict()) for tensor in info.tensors)
    checkpoint = info.checkpoint_descriptor(artifact_path="model.f32")
    context_limits = {
        **manifest.context_limits,
        "max_context_tokens": info.max_seq_len,
    }
    return tensors, checkpoint, context_limits, info.artifact_path


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _materialize_migration(
    output_dir: Path,
    *,
    manifest: NativeExecutionManifest,
    report: NativeCompatibilityReport,
    weight_bundle: bytes | None,
    native_checkpoint_path: Path | None = None,
) -> None:
    output_dir.mkdir(mode=stat.S_IRWXU, parents=True, exist_ok=False)
    _write_json_exclusive(output_dir / "native-execution-manifest.json", manifest.to_dict())
    _write_json_exclusive(output_dir / "compatibility-report.json", report.to_dict())
    if weight_bundle is not None:
        weights_path = output_dir / "weights.bin"
        with weights_path.open("xb") as handle:
            handle.write(weight_bundle)
        weights_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    if native_checkpoint_path is not None:
        checkpoint = manifest.checkpoint or {}
        artifact_name = str(checkpoint.get("artifact_path") or "model.bin")
        if (
            not artifact_name
            or Path(artifact_name).name != artifact_name
            or artifact_name in {".", ".."}
            or "\\" in artifact_name
        ):
            raise ValueError("Native checkpoint artifact_path must be a safe filename.")
        checkpoint_path = output_dir / artifact_name
        with native_checkpoint_path.open("rb") as source, checkpoint_path.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        checkpoint_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        if checkpoint_path.stat().st_size != int(checkpoint.get("target_nbytes", -1)):
            raise ValueError("Native checkpoint changed while the migration artifact was copied.")
        if _sha256_file(checkpoint_path) != checkpoint.get("target_sha256"):
            raise ValueError("Native checkpoint changed while the migration artifact was copied.")
        moa = checkpoint.get("moa")
        if isinstance(moa, Mapping):
            source_metadata_value = checkpoint.get("source_metadata_path")
            if not isinstance(source_metadata_value, str) or not source_metadata_value:
                raise ValueError("Native MoA checkpoint is missing its source metadata path.")
            try:
                source_metadata_path = Path(source_metadata_value).expanduser().resolve(
                    strict=True
                )
            except FileNotFoundError as exc:
                raise ValueError(
                    "Native MoA source metadata changed while the migration artifact was copied."
                ) from exc
            if not source_metadata_path.is_file():
                raise ValueError("Native MoA source metadata is not a regular file.")
            metadata_name = moa.get("metadata_artifact_path")
            if (
                not isinstance(metadata_name, str)
                or not metadata_name
                or Path(metadata_name).name != metadata_name
                or metadata_name in {".", ".."}
                or "\\" in metadata_name
                or metadata_name == artifact_name
            ):
                raise ValueError(
                    "Native MoA metadata_artifact_path must be a distinct safe filename."
                )
            metadata_nbytes = moa.get("metadata_nbytes")
            if (
                isinstance(metadata_nbytes, bool)
                or not isinstance(metadata_nbytes, int)
                or metadata_nbytes <= 0
            ):
                raise ValueError("Native MoA metadata_nbytes must be a positive integer.")
            metadata_sha256 = moa.get("metadata_sha256")
            if (
                not isinstance(metadata_sha256, str)
                or len(metadata_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in metadata_sha256
                )
            ):
                raise ValueError("Native MoA metadata_sha256 is not canonical.")
            metadata_path = output_dir / metadata_name
            with source_metadata_path.open("rb") as source, metadata_path.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            metadata_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            if metadata_path.stat().st_size != metadata_nbytes:
                raise ValueError(
                    "Native MoA metadata changed while the migration artifact was copied."
                )
            if _sha256_file(metadata_path) != metadata_sha256:
                raise ValueError(
                    "Native MoA metadata changed while the migration artifact was copied."
                )


def migrate_graph_to_native(
    graph_path: str | Path,
    *,
    output_dir: str | Path,
    weights_path: str | Path | None = None,
    dry_run: bool = False,
    _source_bytes: bytes | None = None,
) -> NativeMigrationResult:
    """Validate, lower, and optionally materialize a graph-native artifact.

    Graph validation and lowerer checks always complete before a supplied
    checkpoint is opened. Existing output directories are rejected even when
    empty; migration never modifies the source graph or source checkpoint.
    """

    source_path = Path(graph_path).expanduser().resolve()
    destination_input = Path(output_dir).expanduser()
    if destination_input.exists() or destination_input.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {destination_input.resolve()}"
        )
    destination = destination_input.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {destination}")
    if _source_bytes is None and not source_path.is_file():
        raise FileNotFoundError(f"Graph file does not exist: {source_path}")
    if _source_bytes is not None and type(_source_bytes) is not bytes:
        raise TypeError("_source_bytes must be immutable bytes when supplied.")
    prepared = _prepare_graph_file(source_path, raw_bytes=_source_bytes)
    graph_fingerprint = _sha256_bytes(prepared.raw_bytes)
    fallback_capabilities = {
        "native_ir": False,
        "native_training": False,
        "architecture_persistence": False,
        "native_inference": False,
        "resident_inference": False,
        "lossless_kv_cache": False,
        "turboquant_kv_cache": False,
        "turboquant_tile_attention": False,
        "session_prefix_cow": False,
        "serve": False,
        "function_tools": False,
        "structured_output": False,
    }
    if prepared.resolved_payload is None:
        report = NativeCompatibilityReport(
            graph_valid=prepared.graph_valid,
            issues=tuple(prepared.issues),
            graph_fingerprint=graph_fingerprint,
            capabilities=fallback_capabilities,
        )
        return NativeMigrationResult(None, report, None, dry_run=dry_run)

    base_manifest = _manifest_from_resolved_payload(
        prepared.resolved_payload,
        source_path=source_path,
        graph_fingerprint=graph_fingerprint,
    )
    from .native_registry import capability_proof_for

    proof = capability_proof_for(base_manifest.model, base_manifest.topology)
    capabilities = _capabilities(
        proof,
        compatible=not prepared.issues,
        tokenizer=base_manifest.tokenizer,
        chat_template=base_manifest.chat_template,
    )
    warnings: list[str] = []
    if weights_path is None:
        warnings.append(
            "No checkpoint was supplied; the manifest contains topology and current capabilities only."
        )
    if base_manifest.model.get("family") == "unknown":
        warnings.append("The graph does not declare a recognized model family.")
    if not base_manifest.tokenizer:
        warnings.append("The graph does not declare tokenizer metadata.")
    if base_manifest.chat_template.get("source") == "missing":
        warnings.append("The graph does not declare a chat template.")
    if base_manifest.context_limits.get("max_context_tokens") is None:
        warnings.append("The graph does not declare an authoritative context limit.")
    report = NativeCompatibilityReport(
        graph_valid=prepared.graph_valid,
        issues=tuple(prepared.issues),
        graph_fingerprint=graph_fingerprint,
        capabilities=capabilities,
        capability_proof=proof.to_dict(),
        warnings=tuple(warnings),
    )
    manifest = replace(base_manifest, capabilities=capabilities)
    if not report.compatible:
        return NativeMigrationResult(manifest, report, None, dry_run=dry_run)

    weight_bundle: bytes | None = None
    native_checkpoint_source: Path | None = None
    if weights_path is not None:
        checkpoint_path = Path(weights_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
        if _is_dense_differential_model(manifest.model):
            raise ValueError(
                "Native gpt2_diff migration does not yet consume "
                "neuralfn.native_gpt2_diff.training_checkpoint version 2 or its "
                "learned-lambda sidecars; raw dense-v5 and legacy .pt checkpoints "
                "are insufficient, and resident differential inference is not "
                "implemented. Keep the complete training bundle for a future "
                "migration/resident consumer."
            )
        suffix = checkpoint_path.suffix.lower()
        if suffix == ".pt":
            tensor_specs, weight_bundle, checkpoint = _load_pt_tensor_bundle(checkpoint_path)
            context_limits = manifest.context_limits
        elif suffix == ".bin":
            if _is_dense_moa_model(manifest.model):
                raise ValueError(
                    "Native dense MoA migration requires the completed, source-bound "
                    "model_XXXXXXXX.moa.json metadata checkpoint; a raw dense-v5 .bin "
                    "does not preserve the selected activation."
                )
            tensor_specs, checkpoint, context_limits = _load_native_dense_checkpoint(
                checkpoint_path,
                manifest,
            )
            native_checkpoint_source = checkpoint_path
        elif suffix == ".json":
            if manifest.model.get("family") == "mixllama":
                (
                    tensor_specs,
                    checkpoint,
                    context_limits,
                    native_checkpoint_source,
                ) = _load_native_family_standard_moe_checkpoint(
                    checkpoint_path,
                    manifest,
                    graph_fingerprint=graph_fingerprint,
                )
            elif _is_dense_moa_model(manifest.model):
                (
                    tensor_specs,
                    checkpoint,
                    context_limits,
                    native_checkpoint_source,
                ) = _load_native_moa_checkpoint(
                    checkpoint_path,
                    manifest,
                    graph_fingerprint=graph_fingerprint,
                    source_graph_path=source_path,
                )
            else:
                (
                    tensor_specs,
                    checkpoint,
                    context_limits,
                    native_checkpoint_source,
                ) = _load_native_family_llama_checkpoint(
                    checkpoint_path,
                    manifest,
                    graph_fingerprint=graph_fingerprint,
                )
        else:
            raise ValueError(
                "Graph migration accepts legacy .pt tensor checkpoints, native dense v5 .bin "
                "checkpoints, or validated dense-MoA/native-family LLaMA/standard-MoE metadata "
                ".json checkpoints; "
                "raw .f32 sidecars are rejected."
            )
        capabilities = _capabilities(
            proof,
            compatible=not prepared.issues,
            checkpoint=checkpoint,
            tokenizer=manifest.tokenizer,
            chat_template=manifest.chat_template,
        )
        manifest = replace(
            manifest,
            tensors=tensor_specs,
            checkpoint=checkpoint,
            context_limits=context_limits,
            kernel_abi=_kernel_abi(
                proof,
                checkpoint,
                tokenizer=manifest.tokenizer,
                chat_template=manifest.chat_template,
            ),
            capabilities=capabilities,
        )
        report = replace(
            report,
            capabilities=capabilities,
            tensor_mappings=tuple(
                {
                    "source_name": tensor.source_name,
                    "native_name": tensor.name,
                    "dtype": tensor.dtype,
                    "shape": list(tensor.shape),
                    "sha256": tensor.sha256,
                }
                for tensor in tensor_specs
            ),
        )

    if not dry_run:
        _materialize_migration(
            destination,
            manifest=manifest,
            report=report,
            weight_bundle=weight_bundle,
            native_checkpoint_path=native_checkpoint_source,
        )
        return NativeMigrationResult(manifest, report, destination, dry_run=False)
    return NativeMigrationResult(manifest, report, None, dry_run=True)


__all__ = [
    "NATIVE_COMPATIBILITY_REPORT_SCHEMA",
    "NATIVE_COMPATIBILITY_REPORT_VERSION",
    "NATIVE_EXECUTION_MANIFEST_SCHEMA",
    "NATIVE_EXECUTION_MANIFEST_VERSION",
    "NATIVE_TENSOR_BUNDLE_FORMAT",
    "NativeCompatibilityReport",
    "NativeExecutionManifest",
    "NativeLoweringIssue",
    "NativeMigrationResult",
    "NativeTensorSpec",
    "compile_graph_to_native_manifest",
    "compile_native_graph_payload",
    "migrate_graph_to_native",
]
