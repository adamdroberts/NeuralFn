"""Fail-closed Native IR preflight for graph-authored native training.

Serialized graph JSON is compiled through the source-inert Native Execution IR
path before any trainer is selected.  A structurally compatible graph is not
automatically executable: the exact graph profile must also match a reviewed
production adapter in :mod:`neuralfn.native_registry`.

The production adapters target the existing selector-driven dense GPT trainer
and the canonical LLaMA family trainer.  Those executables do not parse Native
IR, so each adapter validates an exact graph/configuration contract, snapshots
the validated source bytes, and passes that snapshot together with immutable
selector, geometry, and source-fingerprint provenance.  This is recorded
honestly as ``trainer_consumes_native_ir=False`` and
``graph_preflight_enforced=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping, Sequence
import uuid

from .native_ir import (
    NativeCompatibilityReport,
    NativeExecutionManifest,
    NativeLoweringIssue,
    migrate_graph_to_native,
)
from .native_registry import (
    NativeGraphTrainingAdapter,
    NativeTrainerSpec,
    capability_proof_for,
    classify_native_graph_training_selector,
    classify_native_model,
    native_graph_training_adapters,
    native_trainer_specs,
)


_TRAINING_PLAN_SCHEMA = "neuralfn.native_graph_training_plan"
_TRAINING_PLAN_VERSION = 1
_TRAINING_PROOF_SCHEMA = "neuralfn.native_graph_training_proof"
_TRAINING_PROOF_VERSION = 1
_GPT2_DIFF_CONFIGURATION_CONTRACT = "dense-gpt2-graph-configuration-v1"
_GPT2_DIFF_TOPOLOGY_CONTRACT = "dense-gpt2-active-topology-v1"
_GPT2_DIFF_VALIDATOR_CONTRACT = "dense-gpt2-exact-graph-validator-v1"
_GPT2_DIFF_MAX_GRAPH_BYTES = 16 * 1024 * 1024
_NATIVE_SIGNED_INT_MAX = (1 << 31) - 1
_NATIVE_SIGNED_INT64_MAX = (1 << 63) - 1
_ADAPTER_NODE_PATH = "root/nodes/model"
_LLAMA_DEFAULT_CONTEXT_TOKENS = 1024
_LLAMA_RMS_NORM_EPS = 1.0e-6
_LLAMA_TRAINING_SELECTORS = frozenset({"llama", "llama_fast"})
_STANDARD_MOE_TRAINING_SELECTORS = frozenset(
    {"moe", "mixllama", "mixllama_fast"}
)

# Shape hashes are over operation/port/edge structure only.  Geometry and
# behavior switches are checked independently below.  Block indices are
# normalized so the same reviewed block contract applies to every layer.
_ROOT_SHAPE_SHA256 = "228b19b94790989400bc770185370fd42761a1c1634451b3bcc3b770788b2b93"
_GPT2_BLOCK_SHAPE_SHA256 = "e722edf71b3a6c37f85a1c63d68df586d7c79f9c9730d1f948ee673102951bad"
_GPT2_MLP_SHAPE_SHA256 = "58a0daeb34c0b19ced46d6bc27f18935201077638c17d42b4f3ac290d3c4a71f"
_GPT2_ATTENTION_SHAPE_BY_SELECTOR = {
    "gpt2": "3833301aa9577752b762fbd902a244e11e2812854c346898593acec80fb652b5",
    "gpt2_diff": "13d8cd97dd07cbc6808839acab609ead157402fe10da5d9de053622d69958937",
    "gpt2_megakernel": "3833301aa9577752b762fbd902a244e11e2812854c346898593acec80fb652b5",
    "gpt2_moa": "3833301aa9577752b762fbd902a244e11e2812854c346898593acec80fb652b5",
    "gpt2_qknorm": "5c5011c703b0aacec56acab0ef355bc57f1f985fd33daf7008f977c3430f42f7",
    "gpt2_softcap": "3833301aa9577752b762fbd902a244e11e2812854c346898593acec80fb652b5",
    "gpt2_stable": "5c5011c703b0aacec56acab0ef355bc57f1f985fd33daf7008f977c3430f42f7",
    "gpt2_zloss": "3833301aa9577752b762fbd902a244e11e2812854c346898593acec80fb652b5",
}

_GPT2_DIFF_TEMPLATE_CONTRACT: dict[str, Any] = {
    "adapter": "none",
    "backbone": "gpt2",
    "backend_capabilities": {
        "cache": True,
        "compile": False,
        "megakernel": False,
        "quantized_export": True,
        "sdpa": True,
    },
    "compression": "none",
    "objective": "ar",
    "router_mode": "none",
    "runtime": "eager",
    "sparsity": "dense",
    "tokenization": "sp",
}
_GPT2_DIFF_INACTIVE_SPEC_CONTRACT: dict[str, Any] = {
    "ema_decay": 0.99,
    "experimental_semantic_router_vecs": False,
    "halt_epsilon": 0.01,
    "jepa_loss_coef": 0.25,
    "jepa_mask_ratio": 0.5,
    "jepa_mask_strategy": "random",
    "jepa_max_block_ratio": 0.25,
    "jepa_min_block_ratio": 0.1,
    "jepa_num_blocks": 4,
    "layer_evo_fraction": 0.1,
    "layer_evo_index": None,
    "layer_evo_mutation_scale": 0.02,
    "layer_evo_population": 8,
    "layer_evo_seed": None,
    "max_recurrence_steps": 4,
    "route_chunk_size": 32,
    "route_evo_enabled": True,
    "route_evo_fraction": 0.1,
    "route_evo_mutation_scale": 0.05,
    "route_evo_population": 8,
    "route_evo_seed": None,
    "semantic_align_loss_coef": 0.5,
    "semantic_dim": 87,
    "semantic_free_experts": 8,
    "semantic_n_lsh_planes": 12,
    "semantic_n_lsh_tables": 8,
    "semantic_residual_dim": 64,
    "semantic_shared_experts": 2,
    "semantic_table_path": "",
    "semantic_vocab_ref": "vocab_86d_o200k.json",
}
_GPT2_DIFF_INACTIVE_BLOCK_CONTRACT: dict[str, Any] = {
    "adapter_dim": 0,
    "auxfree_bias_lr": 0.001,
    "byte_patch_size": 4,
    "byte_patch_stride": 4,
    "dyt_alpha_init": 1.0,
    "experts": None,
    "fp8_amax_history_len": 16,
    "fp8_use_stochastic_rounding": True,
    "group_norm_groups": 1,
    "lora_alpha": 16.0,
    "lora_bias": False,
    "lora_dropout": 0.0,
    "lora_rank": 8,
    "lora_targets": ["q_proj", "v_proj"],
    "moa_activations": ["gelu", "relu", "silu", "relu2"],
    "moa_interval": 50,
    "moe_balance_mode": "aux_loss",
    "multiple_of": None,
    "mx_block_size": 32,
    "nsa_compress_stride": 16,
    "num_sinks": 0,
    "qk_gain_init": 1.0,
    "qlora_compute_dtype": "bf16",
    "qlora_group_size": 64,
    "rope_scaling": None,
    "rope_theta": 10000.0,
    "router_aux_loss_coef": 0.0,
    "router_score_fn": "softmax",
    "shared_experts": 0,
    "sparse_block_size": 64,
    "top_k": None,
    "ttt_hidden_dim": 16,
    "window_size": None,
}
_GPT2_DIFF_REQUIRED_SPEC_KEYS = frozenset(
    {
        "template",
        "block_spec",
        "ar_loss_coef",
        "z_loss_coef",
        "logit_softcap",
        "finetune",
        "layer_evo_enabled",
        "tie_embeddings",
        "model_dim",
        "num_layers",
        "vocab_size",
        "jepa_latent_dim",
        *_GPT2_DIFF_INACTIVE_SPEC_CONTRACT,
    }
)
_GPT2_DIFF_OPTIONAL_GEOMETRY_KEYS = frozenset(
    {"padded_vocab_size", "seq_len", "max_seq_len", "context_window"}
)
_GPT2_DIFF_ACTIVE_BLOCK_KEYS = frozenset(
    {
        "family",
        "norm_type",
        "mlp_type",
        "pos_encoding",
        "attention_backend",
        "attention_variant",
        "compression",
        "adapter_type",
        "residual_type",
        "activation_mode",
        "is_causal",
        "linear_bias",
        "use_qk_norm",
        "diff_lambda_init",
        "dropout_p",
        "mlp_multiplier",
        "num_heads",
        "num_kv_heads",
    }
)


@dataclass(frozen=True, slots=True)
class NativeGraphTrainPlan:
    """Immutable graph-training compatibility and routing result."""

    source_graph: Path
    launch_graph: Path
    graph_preflight_proof: Path | None
    trainer_family: str
    training_selector: str
    native_target: str | None
    adapter_mode: str | None
    trainer_arguments: tuple[str, ...]
    trainer_registered: bool
    architecture_persistence_proven: bool
    execution_ready: bool
    trainer_consumes_native_ir: bool
    graph_preflight_enforced: bool
    manifest: NativeExecutionManifest | None
    compatibility_report: NativeCompatibilityReport
    training_issues: tuple[NativeLoweringIssue, ...]
    artifact_metadata: dict[str, Any]
    blockers: tuple[str, ...]

    @property
    def training_compatible(self) -> bool:
        return self.compatibility_report.compatible and not any(
            issue.severity == "error" for issue in self.training_issues
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _TRAINING_PLAN_SCHEMA,
            "version": _TRAINING_PLAN_VERSION,
            "source_graph": str(self.source_graph),
            "launch_graph": str(self.launch_graph),
            "graph_preflight_proof": (
                str(self.graph_preflight_proof)
                if self.graph_preflight_proof is not None
                else None
            ),
            "trainer_family": self.trainer_family,
            "training_selector": self.training_selector,
            "native_target": self.native_target,
            "adapter_mode": self.adapter_mode,
            "trainer_arguments": list(self.trainer_arguments),
            "trainer_registered": self.trainer_registered,
            "architecture_persistence_proven": self.architecture_persistence_proven,
            "execution_ready": self.execution_ready,
            "trainer_consumes_native_ir": self.trainer_consumes_native_ir,
            "graph_preflight_enforced": self.graph_preflight_enforced,
            "manifest": self.manifest.to_dict() if self.manifest is not None else None,
            "compatibility_report": self.compatibility_report.to_dict(),
            "training_compatibility": {
                "compatible": self.training_compatible,
                "issues": [issue.to_dict() for issue in self.training_issues],
            },
            "artifact_metadata": dict(self.artifact_metadata),
            "blockers": list(self.blockers),
        }


def _unused_dry_run_destination() -> Path:
    """Return a collision-resistant path without creating it."""

    root = Path(tempfile.gettempdir()).resolve()
    for _ in range(8):
        candidate = root / f"neuralfn-native-graph-train-{uuid.uuid4().hex}"
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise RuntimeError("Could not reserve a non-existent Native IR dry-run path.")


def _trainer_for_family(family: str) -> NativeTrainerSpec | None:
    return next((spec for spec in native_trainer_specs() if spec.family == family), None)


def _adapter_for_selector(selector: str) -> NativeGraphTrainingAdapter | None:
    return next(
        (adapter for adapter in native_graph_training_adapters() if adapter.selector == selector),
        None,
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _exact_json_value_matches(actual: Any, expected: Any) -> bool:
    """Compare JSON values without Python's bool/int/float equality aliases."""

    if isinstance(expected, Mapping):
        return bool(
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(
                _exact_json_value_matches(actual[key], expected[key])
                for key in expected
            )
        )
    if isinstance(expected, (list, tuple)):
        return bool(
            isinstance(actual, type(expected))
            and len(actual) == len(expected)
            and all(
                _exact_json_value_matches(left, right)
                for left, right in zip(actual, expected)
            )
        )
    if type(expected) is bool:
        return type(actual) is bool and actual is expected
    if type(expected) is int:
        return type(actual) is int and actual == expected
    if type(expected) is float:
        return bool(
            type(actual) is float
            and math.isfinite(actual)
            and actual == expected
        )
    return type(actual) is type(expected) and actual == expected


def _is_finite_json_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _strict_raw_gpt2_graph_schema_error(payload: Mapping[str, Any]) -> str | None:
    """Validate the forward-closed serialized schema behind the exact proof."""

    graph_keys = {
        "name",
        "training_method",
        "runtime",
        "surrogate_config",
        "evo_config",
        "torch_config",
        "variant_library",
        "nodes",
        "edges",
        "input_node_ids",
        "output_node_ids",
    }
    node_keys = {"instance_id", "neuron_def", "position"}
    neuron_keys = {
        "id",
        "name",
        "kind",
        "input_ports",
        "output_ports",
        "source_code",
        "subgraph",
        "module_type",
        "module_config",
        "module_state",
        "variant_ref",
        "input_aliases",
        "output_aliases",
    }
    edge_keys = {"id", "src_node", "src_port", "dst_node", "dst_port", "weight", "bias"}
    port_keys = {"name", "range", "precision", "dtype"}
    canonical_sources = {
        "",
        "def input(x):\n    return x\n",
        "def output(x):\n    return x\n",
        '@neuron(\n    inputs=[Port("a"), Port("b")],\n'
        '    outputs=[Port("sum")],\n    name="add",\n)\n'
        "def add(a, b):\n    return a + b\n",
    }
    function_sources = {
        "input": "def input(x):\n    return x\n",
        "output": "def output(x):\n    return x\n",
        "add": '@neuron(\n    inputs=[Port("a"), Port("b")],\n'
        '    outputs=[Port("sum")],\n    name="add",\n)\n'
        "def add(a, b):\n    return a + b\n",
    }
    root_variant_families = {
        "attention",
        "mamba",
        "mlp",
        "attn_block",
        "mamba_block",
        "transformer_block",
    }

    def string_list(value: Any) -> bool:
        return isinstance(value, list) and all(type(item) is str for item in value)

    def validate_port(
        port: Any,
        *,
        neuron_kind: str,
        module_type: str,
        port_collection: str,
        port_index: int,
        path: str,
    ) -> str | None:
        if not isinstance(port, Mapping) or set(port) != port_keys:
            return f"{path} port fields are not the canonical serialized set"
        if type(port.get("name")) is not str or type(port.get("dtype")) is not str:
            return f"{path} port name/dtype must be exact JSON strings"
        token_index_port = bool(
            neuron_kind == "module"
            and port_collection == "input_ports"
            and (
                (module_type == "token_embedding" and port_index == 0)
                or (module_type == "token_cross_entropy" and port_index == 1)
            )
        )
        boundary_token_port = bool(
            port.get("dtype") == "tokens"
            and neuron_kind in {"function", "subgraph"}
        )
        expected_range: Any = [0, 65535] if token_index_port else None
        expected_precision: Any = 1.0 if token_index_port or boundary_token_port else None
        if not _exact_json_value_matches(port.get("range"), expected_range):
            return f"{path} port range is not canonical for the native graph contract"
        if not _exact_json_value_matches(port.get("precision"), expected_precision):
            return f"{path} port precision is not canonical for the native graph contract"
        return None

    def validate_graph(graph: Any, *, path: str, is_root: bool) -> str | None:
        if not isinstance(graph, Mapping) or set(graph) != graph_keys:
            return f"{path} graph fields are not the canonical serialized set"
        if type(graph.get("name")) is not str or not graph.get("name"):
            return f"{path}.name must be a non-empty JSON string"
        if graph.get("training_method") != "torch":
            return f"{path}.training_method must be exactly 'torch'"
        # The root value is orchestration/authoring metadata: SDK-authored
        # graphs retain ``torch`` while editor native runs declare
        # ``native-cuda``.  Nested executable stages are Torch-authored module
        # graphs in both workflows and must remain exactly ``torch``.
        allowed_runtime = {"torch", "native-cuda"} if is_root else {"torch"}
        if graph.get("runtime") not in allowed_runtime:
            return f"{path}.runtime is not a reviewed graph-training runtime"
        if graph.get("surrogate_config") != {} or graph.get("evo_config") != {}:
            return f"{path} surrogate/evolutionary configuration must be empty"
        torch_config = graph.get("torch_config")
        if is_root:
            if (
                not isinstance(torch_config, Mapping)
                or set(torch_config) != {"device", "amp_dtype", "template_spec"}
                or torch_config.get("device") != "cuda"
                or torch_config.get("amp_dtype") != "float32"
                or not isinstance(torch_config.get("template_spec"), Mapping)
            ):
                return f"{path}.torch_config is not the canonical root training configuration"
        elif torch_config != {}:
            return f"{path}.torch_config must be empty on nested graph stages"
        nodes = graph.get("nodes")
        edges = graph.get("edges")
        if not isinstance(nodes, Mapping) or not isinstance(edges, Mapping):
            return f"{path} nodes/edges must be JSON objects"
        if not string_list(graph.get("input_node_ids")) or not string_list(
            graph.get("output_node_ids")
        ):
            return f"{path} graph boundary IDs must be JSON string arrays"

        for node_id, node in nodes.items():
            node_path = f"{path}/nodes/{node_id}"
            if type(node_id) is not str or not isinstance(node, Mapping) or set(node) != node_keys:
                return f"{node_path} node fields are not the canonical serialized set"
            if node.get("instance_id") != node_id:
                return f"{node_path}.instance_id does not match its node key"
            position = node.get("position")
            if (
                not isinstance(position, list)
                or len(position) != 2
                or any(not _is_finite_json_number(value) for value in position)
            ):
                return f"{node_path}.position must be a finite two-number display coordinate"
            neuron = node.get("neuron_def")
            if not isinstance(neuron, Mapping) or set(neuron) != neuron_keys:
                return f"{node_path}.neuron_def fields are not the canonical serialized set"
            neuron_kind = neuron.get("kind")
            module_type = neuron.get("module_type")
            if neuron_kind not in {"function", "module", "subgraph"}:
                return f"{node_path}.neuron_def.kind is not canonical"
            if (
                type(neuron.get("id")) is not str
                or re.fullmatch(r"[0-9a-f]{12}", neuron["id"]) is None
                or type(neuron.get("name")) is not str
                or type(module_type) is not str
                or not isinstance(neuron.get("module_config"), Mapping)
                or neuron.get("module_state") != ""
                or neuron.get("source_code") not in canonical_sources
                or not string_list(neuron.get("input_aliases"))
                or not string_list(neuron.get("output_aliases"))
            ):
                return f"{node_path}.neuron_def contains a noncanonical serialized value"
            variant_ref = neuron.get("variant_ref")
            if variant_ref is not None and (
                not isinstance(variant_ref, Mapping)
                or set(variant_ref) != {"family", "version"}
                or type(variant_ref.get("family")) is not str
                or type(variant_ref.get("version")) is not str
            ):
                return f"{node_path}.neuron_def.variant_ref is not canonical"
            subgraph = neuron.get("subgraph")
            input_ports = neuron.get("input_ports")
            output_ports = neuron.get("output_ports")
            if not isinstance(input_ports, list) or not isinstance(output_ports, list):
                return f"{node_path}.neuron_def ports must be JSON arrays"
            if neuron_kind == "function":
                if (
                    neuron.get("module_type") != ""
                    or neuron.get("module_config") != {}
                    or neuron.get("input_aliases") != []
                    or neuron.get("output_aliases") != []
                    or variant_ref is not None
                    or subgraph is not None
                    or neuron.get("source_code")
                    != function_sources.get(str(neuron.get("name") or ""))
                ):
                    return f"{node_path} function definition is not the exact canonical boundary/add contract"
            elif neuron_kind == "module":
                if (
                    not neuron.get("module_type")
                    or neuron.get("name") != neuron.get("module_type")
                    or neuron.get("source_code") != ""
                    or neuron.get("input_aliases") != []
                    or neuron.get("output_aliases") != []
                    or variant_ref is not None
                    or subgraph is not None
                ):
                    return f"{node_path} module definition is not the exact canonical module contract"
            elif neuron_kind == "subgraph":
                expected_input_aliases = [
                    port.get("name") if isinstance(port, Mapping) else None
                    for port in input_ports
                ]
                expected_output_aliases = [
                    port.get("name") if isinstance(port, Mapping) else None
                    for port in output_ports
                ]
                if (
                    neuron.get("module_type") != ""
                    or neuron.get("module_config") != {}
                    or neuron.get("source_code") != ""
                    or neuron.get("input_aliases") != expected_input_aliases
                    or neuron.get("output_aliases") != expected_output_aliases
                ):
                    return f"{node_path} subgraph aliases/configuration do not match its exact boundary ports"
                expected_variant_ref = (
                    None
                    if node_id == "model"
                    else {"family": "transformer_block", "version": "default"}
                    if re.fullmatch(r"block_[0-9]+", node_id)
                    else {"family": "attention", "version": "default"}
                    if node_id == "attention"
                    else {"family": "mlp", "version": "default"}
                    if node_id == "mlp"
                    else object()
                )
                if not _exact_json_value_matches(variant_ref, expected_variant_ref):
                    return f"{node_path}.neuron_def.variant_ref is not the canonical active variant"
                if not isinstance(subgraph, Mapping):
                    return f"{node_path}.neuron_def.subgraph is missing"
                nested_error = validate_graph(
                    subgraph,
                    path=f"{node_path}/subgraph",
                    is_root=False,
                )
                if nested_error is not None:
                    return nested_error
            for collection in ("input_ports", "output_ports"):
                ports = neuron.get(collection)
                for index, port in enumerate(ports):
                    port_error = validate_port(
                        port,
                        neuron_kind=neuron_kind,
                        module_type=module_type,
                        port_collection=collection,
                        port_index=index,
                        path=f"{node_path}.neuron_def.{collection}[{index}]",
                    )
                    if port_error is not None:
                        return port_error

        for edge_id, edge in edges.items():
            edge_path = f"{path}/edges/{edge_id}"
            if type(edge_id) is not str or not isinstance(edge, Mapping) or set(edge) != edge_keys:
                return f"{edge_path} edge fields are not the canonical serialized set"
            if (
                edge.get("id") != edge_id
                or type(edge.get("src_node")) is not str
                or type(edge.get("dst_node")) is not str
                or type(edge.get("src_port")) is not int
                or type(edge.get("dst_port")) is not int
                or type(edge.get("weight")) is not float
                or type(edge.get("bias")) is not float
            ):
                return f"{edge_path} edge values are not canonical"

        variant_library = graph.get("variant_library")
        if not isinstance(variant_library, Mapping):
            return f"{path}.variant_library must be a JSON object"
        if is_root:
            if set(variant_library) != root_variant_families:
                return f"{path}.variant_library families are not the canonical reviewed set"
            for family, versions in variant_library.items():
                if not isinstance(versions, Mapping) or set(versions) != {"default"}:
                    return f"{path}.variant_library.{family} versions are not canonical"
                variant_error = validate_graph(
                    versions["default"],
                    path=f"{path}/variant_library/{family}@default",
                    is_root=False,
                )
                if variant_error is not None:
                    return variant_error
        elif variant_library != {}:
            return f"{path}.variant_library must be empty below the root graph"
        return None

    return validate_graph(payload, path="root", is_root=True)


def _read_source_graph_snapshot(source_graph: Path) -> bytes:
    """Read one regular-file snapshot without permitting an unbounded race.

    The descriptor is opened with ``O_NOFOLLOW`` where the host provides it,
    then both the advertised size and the actual read are bounded.  A writer
    that grows the already-open file after ``fstat`` can therefore cause at
    most ``limit + 1`` bytes to be retained before the planner rejects it.
    """

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source_graph, flags)
    try:
        source_stat = os.fstat(descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError(
                f"Native graph training source is not a regular file: {source_graph}."
            )
        if source_stat.st_size > _GPT2_DIFF_MAX_GRAPH_BYTES:
            raise ValueError(
                "Native graph training source exceeds the native verified-graph bound "
                f"of {_GPT2_DIFF_MAX_GRAPH_BYTES} bytes: {source_stat.st_size}."
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            source_bytes = handle.read(_GPT2_DIFF_MAX_GRAPH_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(source_bytes) > _GPT2_DIFF_MAX_GRAPH_BYTES:
        raise ValueError(
            "Native graph training source exceeded the native verified-graph bound "
            f"of {_GPT2_DIFF_MAX_GRAPH_BYTES} bytes while it was read."
        )
    if not source_bytes:
        raise ValueError("Native graph training source must be non-empty.")
    return source_bytes


def _strict_graph_json_issue(raw_bytes: bytes) -> NativeLoweringIssue | None:
    """Reject JSON spellings that cannot be attested unambiguously.

    The Native IR loader deliberately reports ordinary JSON failures through
    its compatibility report.  The materialized training proof has a stronger
    trust boundary: duplicate object keys and non-finite numeric spellings must
    not be normalized differently by the Python planner and native consumer.
    """

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate object key {key!r}")
            result[key] = value
        return result

    def parse_finite_float(raw: str) -> float:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number {raw!r}")
        return value

    def reject_nonstandard_constant(raw: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {raw!r}")

    if not raw_bytes:
        return _issue(
            "ambiguous_training_graph_json",
            "Exact graph-training proof requires a non-empty source graph.",
            path="root",
            operation="graph",
        )
    if len(raw_bytes) > _GPT2_DIFF_MAX_GRAPH_BYTES:
        return _issue(
            "ambiguous_training_graph_json",
            "Exact graph-training proof requires source graph bytes no larger "
            f"than {_GPT2_DIFF_MAX_GRAPH_BYTES} bytes; got {len(raw_bytes)}.",
            path="root",
            operation="graph",
        )
    try:
        payload = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_float=parse_finite_float,
            parse_constant=reject_nonstandard_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return _issue(
            "ambiguous_training_graph_json",
            "Exact graph-training proof requires unique object keys and finite "
            f"standard JSON numbers: {exc}",
            path="root",
            operation="graph",
        )
    if not isinstance(payload, Mapping):
        return _issue(
            "ambiguous_training_graph_json",
            "Exact graph-training proof requires a JSON object.",
            path="root",
            operation="graph",
        )
    schema_error = _strict_raw_gpt2_graph_schema_error(payload)
    if schema_error is not None:
        return _issue(
            "ambiguous_training_graph_json",
            "Exact graph-training proof requires the forward-closed serialized "
            f"graph schema: {schema_error}.",
            path="root",
            operation="graph",
        )
    stack: list[Any] = [payload]
    exact_boolean_module_fields = frozenset({"bias", "is_causal"})
    exact_integer_module_fields = frozenset(
        {
            "input_dim",
            "max_seq_len",
            "model_dim",
            "num_heads",
            "output_dim",
            "vocab_size",
        }
    )
    finite_numeric_module_fields = frozenset({"dropout_p", "eps", "lambda_init"})
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            if {"src_node", "dst_node", "weight", "bias"}.issubset(value):
                for field in ("weight", "bias"):
                    raw = value[field]
                    if type(raw) is not float or not math.isfinite(raw):
                        return _issue(
                            "ambiguous_training_graph_json",
                            "Exact graph-training proof requires every raw edge "
                            f"{field} to be a finite JSON float, not {type(raw).__name__}.",
                            path="root",
                            operation="graph",
                        )
            module_type = value.get("module_type")
            module_config = value.get("module_config")
            if isinstance(module_type, str) and isinstance(module_config, Mapping):
                for field in exact_boolean_module_fields & set(module_config):
                    if type(module_config[field]) is not bool:
                        return _issue(
                            "ambiguous_training_graph_json",
                            "Exact graph-training proof requires raw module field "
                            f"{module_type}.{field} to be a JSON boolean.",
                            path="root",
                            operation=module_type or "graph",
                        )
                for field in exact_integer_module_fields & set(module_config):
                    if type(module_config[field]) is not int:
                        return _issue(
                            "ambiguous_training_graph_json",
                            "Exact graph-training proof requires raw module field "
                            f"{module_type}.{field} to be a JSON integer.",
                            path="root",
                            operation=module_type or "graph",
                        )
                for field in finite_numeric_module_fields & set(module_config):
                    raw = module_config[field]
                    if type(raw) is not float or not math.isfinite(raw):
                        return _issue(
                            "ambiguous_training_graph_json",
                            "Exact graph-training proof requires raw module field "
                            f"{module_type}.{field} to be a finite JSON float.",
                            path="root",
                            operation=module_type or "graph",
                        )
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return None


def _issue(
    code: str,
    message: str,
    *,
    path: str = _ADAPTER_NODE_PATH,
    operation: str = "subgraph.call",
) -> NativeLoweringIssue:
    return NativeLoweringIssue(
        path=path,
        code=code,
        message=message,
        severity="error",
        node_kind="subgraph",
        operation=operation,
    )


def _active_graphs(topology: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    graphs = topology.get("graphs", ())
    if not isinstance(graphs, Sequence) or isinstance(graphs, (str, bytes, bytearray)):
        return ()
    return tuple(
        graph
        for graph in graphs
        if isinstance(graph, Mapping)
        and (
            str(graph.get("path") or "") == "root"
            or str(graph.get("path") or "").startswith("root/")
        )
    )


def _first_operation_path(
    topology: Mapping[str, Any],
    operations: set[str] | frozenset[str],
) -> tuple[str, str] | None:
    for graph in _active_graphs(topology):
        nodes = graph.get("nodes", ())
        if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes, bytearray)):
            continue
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            operation = str(node.get("operation") or "")
            if operation in operations:
                return str(node.get("path") or _ADAPTER_NODE_PATH), operation
    return None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (OverflowError, TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (OverflowError, TypeError, ValueError):
        return default


def _normalized(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _expect_equal(
    issues: list[NativeLoweringIssue],
    *,
    actual: Any,
    expected: Any,
    field: str,
) -> None:
    if actual != expected:
        issues.append(
            _issue(
                "unsupported_training_configuration",
                f"{field} must be {expected!r} for the production graph adapter; got {actual!r}.",
            )
        )


def _validate_exact_gpt2_diff_serialized_configuration(
    model: Mapping[str, Any],
    spec: Mapping[str, Any],
    template: Mapping[str, Any],
    block: Mapping[str, Any],
) -> tuple[NativeLoweringIssue, ...]:
    issues: list[NativeLoweringIssue] = []
    expected_model_keys = {
        "name",
        "family",
        "backbone",
        "family_class",
        "objective",
        "compression",
        "text_generation",
        "turboquant_policy",
        "template_spec",
    }
    expected_model_values = {
        "family": "gpt2",
        "backbone": "gpt2",
        "family_class": "autoregressive_transformer",
        "objective": "ar",
        "compression": "none",
        "text_generation": True,
        "turboquant_policy": "standard-retained-kv",
    }
    if (
        set(model) != expected_model_keys
        or type(model.get("name")) is not str
        or not str(model.get("name") or "")
        or any(
            not _exact_json_value_matches(model.get(key), expected)
            for key, expected in expected_model_values.items()
        )
    ):
        issues.append(
            _issue(
                "unsupported_training_configuration",
                "gpt2_diff model metadata must use the exact reviewed key/value contract.",
            )
        )

    allowed_spec_keys = (
        _GPT2_DIFF_REQUIRED_SPEC_KEYS | _GPT2_DIFF_OPTIONAL_GEOMETRY_KEYS
    )
    if (
        not _GPT2_DIFF_REQUIRED_SPEC_KEYS.issubset(spec)
        or not set(spec).issubset(allowed_spec_keys)
    ):
        issues.append(
            _issue(
                "unsupported_training_configuration",
                "gpt2_diff template_spec fields are not the exact reviewed set; "
                "unknown or missing serialized semantics require a new validator contract.",
            )
        )
    for key, expected in _GPT2_DIFF_INACTIVE_SPEC_CONTRACT.items():
        if not _exact_json_value_matches(spec.get(key), expected):
            issues.append(
                _issue(
                    "unsupported_training_configuration",
                    f"template_spec.{key} must retain its exact canonical inactive value "
                    f"{expected!r}; the native trainer does not consume this field.",
                )
            )
    if not _exact_json_value_matches(
        spec.get("jepa_latent_dim"), spec.get("model_dim")
    ):
        issues.append(
            _issue(
                "unsupported_training_configuration",
                "template_spec.jepa_latent_dim must retain its canonical model_dim-derived value; "
                "the native trainer does not consume this field.",
            )
        )
    for key, expected in {
        "ar_loss_coef": 1.0,
        "z_loss_coef": 0.0,
        "logit_softcap": 0.0,
        "finetune": None,
        "layer_evo_enabled": False,
        "tie_embeddings": True,
        "vocab_size": 50257,
    }.items():
        if not _exact_json_value_matches(spec.get(key), expected):
            issues.append(
                _issue(
                    "unsupported_training_configuration",
                    f"template_spec.{key} must be the exact canonical value {expected!r}.",
                )
            )

    if not _exact_json_value_matches(template, _GPT2_DIFF_TEMPLATE_CONTRACT):
        issues.append(
            _issue(
                "unsupported_training_configuration",
                "gpt2_diff template fields must use the exact reviewed key/value contract.",
            )
        )

    expected_block_keys = (
        _GPT2_DIFF_ACTIVE_BLOCK_KEYS | set(_GPT2_DIFF_INACTIVE_BLOCK_CONTRACT)
    )
    if set(block) != expected_block_keys:
        issues.append(
            _issue(
                "unsupported_training_configuration",
                "gpt2_diff block_spec fields are not the exact reviewed set; unknown "
                "or missing serialized semantics require a new validator contract.",
            )
        )
    for key, expected in _GPT2_DIFF_INACTIVE_BLOCK_CONTRACT.items():
        if not _exact_json_value_matches(block.get(key), expected):
            issues.append(
                _issue(
                    "unsupported_training_configuration",
                    f"block_spec.{key} must retain its exact canonical inactive value "
                    f"{expected!r}; the native trainer does not consume this field.",
                )
            )
    for key, expected in {
        "family": "gpt2",
        "norm_type": "layernorm",
        "mlp_type": "gelu",
        "pos_encoding": "absolute",
        "attention_backend": "sdpa",
        "attention_variant": "differential",
        "compression": "none",
        "adapter_type": "none",
        "residual_type": "add",
        "activation_mode": "single",
        "is_causal": True,
        "linear_bias": True,
        "use_qk_norm": False,
        "diff_lambda_init": 0.8,
        "dropout_p": 0.0,
        "mlp_multiplier": 4.0,
    }.items():
        if not _exact_json_value_matches(block.get(key), expected):
            issues.append(
                _issue(
                    "unsupported_training_configuration",
                    f"block_spec.{key} must be the exact canonical value {expected!r}.",
                )
            )
    return tuple(issues)


def _validate_gpt2_configuration(
    model: Mapping[str, Any],
    selector: str,
) -> tuple[NativeLoweringIssue, ...]:
    issues: list[NativeLoweringIssue] = []
    spec = _mapping(model.get("template_spec"))
    template = _mapping(spec.get("template"))
    block = _mapping(spec.get("block_spec"))
    if selector == "gpt2_diff":
        issues.extend(
            _validate_exact_gpt2_diff_serialized_configuration(
                model,
                spec,
                template,
                block,
            )
        )

    for raw, field in (
        (template.get("objective"), "template.objective"),
        (template.get("backbone"), "template.backbone"),
        (template.get("tokenization"), "template.tokenization"),
        (template.get("sparsity"), "template.sparsity"),
        (template.get("router_mode"), "template.router_mode"),
        (template.get("compression"), "template.compression"),
        (template.get("adapter"), "template.adapter"),
        (template.get("runtime"), "template.runtime"),
        (block.get("family"), "block_spec.family"),
        (block.get("norm_type"), "block_spec.norm_type"),
        (block.get("mlp_type"), "block_spec.mlp_type"),
        (block.get("pos_encoding"), "block_spec.pos_encoding"),
        (block.get("attention_backend"), "block_spec.attention_backend"),
        (block.get("attention_variant"), "block_spec.attention_variant"),
        (block.get("compression"), "block_spec.compression"),
        (block.get("adapter_type"), "block_spec.adapter_type"),
        (block.get("residual_type"), "block_spec.residual_type"),
        (block.get("activation_mode"), "block_spec.activation_mode"),
    ):
        if type(raw) is not str:
            issues.append(
                _issue(
                    "unsupported_training_configuration",
                    f"{field} must be an exact JSON string.",
                )
            )
    for raw, field in (
        (spec.get("tie_embeddings"), "template_spec.tie_embeddings"),
        (spec.get("layer_evo_enabled"), "template_spec.layer_evo_enabled"),
        (block.get("is_causal"), "block_spec.is_causal"),
        (block.get("linear_bias"), "block_spec.linear_bias"),
        (block.get("use_qk_norm"), "block_spec.use_qk_norm"),
    ):
        if type(raw) is not bool:
            issues.append(
                _issue(
                    "unsupported_training_configuration",
                    f"{field} must be an exact JSON boolean.",
                )
            )
    numeric_fields = [
        (spec.get("ar_loss_coef"), "template_spec.ar_loss_coef"),
        (spec.get("z_loss_coef"), "template_spec.z_loss_coef"),
        (spec.get("logit_softcap"), "template_spec.logit_softcap"),
        (block.get("dropout_p"), "block_spec.dropout_p"),
        (block.get("mlp_multiplier"), "block_spec.mlp_multiplier"),
    ]
    if selector == "gpt2_diff":
        numeric_fields.append(
            (block.get("diff_lambda_init"), "block_spec.diff_lambda_init")
        )
    for raw, field in numeric_fields:
        if type(raw) is not float or not math.isfinite(raw):
            issues.append(
                _issue(
                    "unsupported_training_configuration",
                    f"{field} must be a finite JSON float.",
                )
            )

    exact_fields = (
        (_normalized(template.get("objective")), "ar", "template.objective"),
        (_normalized(template.get("backbone")), "gpt2", "template.backbone"),
        (_normalized(template.get("tokenization")), "sp", "template.tokenization"),
        (_normalized(template.get("sparsity")), "dense", "template.sparsity"),
        (_normalized(template.get("router_mode")), "none", "template.router_mode"),
        (_normalized(template.get("compression")), "none", "template.compression"),
        (_normalized(template.get("adapter")), "none", "template.adapter"),
        (_normalized(block.get("family")), "gpt2", "block_spec.family"),
        (_normalized(block.get("norm_type")), "layernorm", "block_spec.norm_type"),
        (_normalized(block.get("mlp_type")), "gelu", "block_spec.mlp_type"),
        (_normalized(block.get("pos_encoding")), "absolute", "block_spec.pos_encoding"),
        (_normalized(block.get("attention_backend")), "sdpa", "block_spec.attention_backend"),
        (_normalized(block.get("compression")), "none", "block_spec.compression"),
        (_normalized(block.get("adapter_type")), "none", "block_spec.adapter_type"),
        (_normalized(block.get("residual_type")), "add", "block_spec.residual_type"),
    )
    for actual, expected, field in exact_fields:
        _expect_equal(issues, actual=actual, expected=expected, field=field)

    backend_capabilities = template.get("backend_capabilities")
    expected_backend_capabilities = {
            "cache": True,
            "compile": selector == "gpt2_megakernel",
            "megakernel": selector == "gpt2_megakernel",
            "quantized_export": True,
            "sdpa": True,
        }
    if (
        not isinstance(backend_capabilities, Mapping)
        or set(backend_capabilities) != set(expected_backend_capabilities)
        or any(type(value) is not bool for value in backend_capabilities.values())
    ):
        issues.append(
            _issue(
                "unsupported_training_configuration",
                "template.backend_capabilities must use the exact reviewed boolean map.",
            )
        )
    _expect_equal(
        issues,
        actual=dict(_mapping(backend_capabilities)),
        expected=expected_backend_capabilities,
        field="template.backend_capabilities",
    )
    _expect_equal(
        issues,
        actual=_as_float(spec.get("ar_loss_coef")),
        expected=1.0,
        field="template_spec.ar_loss_coef",
    )
    if "finetune" not in spec or spec.get("finetune") is not None:
        issues.append(
            _issue(
                "unsupported_training_configuration",
                "template_spec.finetune must be explicitly null for the production graph adapter.",
            )
        )
    _expect_equal(
        issues,
        actual=spec.get("layer_evo_enabled"),
        expected=False,
        field="template_spec.layer_evo_enabled",
    )

    _expect_equal(
        issues,
        actual=bool(spec.get("tie_embeddings", False)),
        expected=True,
        field="template_spec.tie_embeddings",
    )
    _expect_equal(
        issues,
        actual=bool(block.get("is_causal", False)),
        expected=True,
        field="block_spec.is_causal",
    )
    _expect_equal(
        issues,
        actual=bool(block.get("linear_bias", False)),
        expected=True,
        field="block_spec.linear_bias",
    )
    _expect_equal(
        issues,
        actual=_as_float(block.get("dropout_p")),
        expected=0.0,
        field="block_spec.dropout_p",
    )
    _expect_equal(
        issues,
        actual=_as_float(block.get("mlp_multiplier")),
        expected=4.0,
        field="block_spec.mlp_multiplier",
    )

    runtime = _normalized(template.get("runtime") or "eager")
    _expect_equal(
        issues,
        actual=runtime,
        expected="megakernel" if selector == "gpt2_megakernel" else "eager",
        field="template.runtime",
    )
    _expect_equal(
        issues,
        actual=_normalized(block.get("activation_mode") or "single"),
        expected="moa" if selector == "gpt2_moa" else "single",
        field="block_spec.activation_mode",
    )
    _expect_equal(
        issues,
        actual=_normalized(block.get("attention_variant") or "dense"),
        expected="differential" if selector == "gpt2_diff" else "dense",
        field="block_spec.attention_variant",
    )
    _expect_equal(
        issues,
        actual=bool(block.get("use_qk_norm", False)),
        expected=selector in {"gpt2_qknorm", "gpt2_stable"},
        field="block_spec.use_qk_norm",
    )
    _expect_equal(
        issues,
        actual=_as_float(spec.get("z_loss_coef")),
        expected=1.0e-4 if selector in {"gpt2_zloss", "gpt2_stable"} else 0.0,
        field="template_spec.z_loss_coef",
    )
    _expect_equal(
        issues,
        actual=_as_float(spec.get("logit_softcap")),
        expected=30.0 if selector == "gpt2_softcap" else 0.0,
        field="template_spec.logit_softcap",
    )
    if selector == "gpt2_diff":
        _expect_equal(
            issues,
            actual=_as_float(block.get("diff_lambda_init")),
            expected=0.8,
            field="block_spec.diff_lambda_init",
        )
    if selector == "gpt2_moa":
        raw_candidates = block.get("moa_activations")
        declared_candidates = (
            tuple(raw_candidates)
            if isinstance(raw_candidates, (list, tuple))
            and all(isinstance(candidate, str) for candidate in raw_candidates)
            else ()
        )
        _expect_equal(
            issues,
            actual=declared_candidates,
            expected=("gelu", "relu", "silu", "relu2"),
            field="block_spec.moa_activations",
        )
        moa_interval = block.get("moa_interval")
        if (
            isinstance(moa_interval, bool)
            or not isinstance(moa_interval, int)
            or moa_interval <= 0
        ):
            issues.append(
                _issue(
                    "unsupported_training_configuration",
                    "block_spec.moa_interval must be a positive integer for the "
                    "production graph adapter.",
                )
            )

    for raw, field in (
        (spec.get("model_dim"), "template_spec.model_dim"),
        (spec.get("num_layers"), "template_spec.num_layers"),
        (spec.get("vocab_size"), "template_spec.vocab_size"),
        (block.get("num_heads"), "block_spec.num_heads"),
    ):
        if type(raw) is not int:
            issues.append(
                _issue(
                    "unsupported_training_geometry",
                    f"{field} must be an exact JSON integer.",
                )
            )
    model_dim = _as_int(spec.get("model_dim"))
    num_layers = _as_int(spec.get("num_layers"))
    vocab_size = _as_int(spec.get("vocab_size"))
    padded_vocab_raw = spec.get("padded_vocab_size")
    num_heads = _as_int(block.get("num_heads"))
    num_kv_raw = block.get("num_kv_heads")
    if num_kv_raw not in (None, "") and type(num_kv_raw) is not int:
        issues.append(
            _issue(
                "unsupported_training_geometry",
                "block_spec.num_kv_heads must be an exact JSON integer when declared.",
            )
        )
    num_kv_heads = num_heads if num_kv_raw in (None, "") else _as_int(num_kv_raw)
    for value, field in (
        (model_dim, "template_spec.model_dim"),
        (num_layers, "template_spec.num_layers"),
        (vocab_size, "template_spec.vocab_size"),
        (num_heads, "block_spec.num_heads"),
        (num_kv_heads, "block_spec.num_kv_heads"),
    ):
        if value > _NATIVE_SIGNED_INT_MAX:
            issues.append(
                _issue(
                    "unsupported_training_geometry",
                    f"{field} exceeds the native signed-int ABI limit.",
                )
            )
    if model_dim <= 0 or num_heads <= 0 or model_dim % num_heads:
        issues.append(
            _issue(
                "unsupported_training_geometry",
                "template_spec.model_dim must be positive and divisible by block_spec.num_heads.",
            )
        )
    if num_layers <= 0:
        issues.append(
            _issue(
                "unsupported_training_geometry",
                "template_spec.num_layers must be a positive integer.",
            )
        )
    if num_kv_heads != num_heads:
        issues.append(
            _issue(
                "unsupported_training_geometry",
                "The dense production adapter requires num_kv_heads == num_heads.",
            )
        )
    if vocab_size != 50257:
        issues.append(
            _issue(
                "unsupported_training_geometry",
                "The current dense graph-file adapter requires vocab_size=50257; "
                f"got {vocab_size}.",
            )
        )
    if padded_vocab_raw not in (None, "") and type(padded_vocab_raw) is not int:
        issues.append(
            _issue(
                "unsupported_training_geometry",
                "template_spec.padded_vocab_size must be an exact JSON integer when declared.",
            )
        )
    if padded_vocab_raw not in (None, "") and _as_int(padded_vocab_raw) != 50304:
        issues.append(
            _issue(
                "unsupported_training_geometry",
                "The current dense graph-file adapter requires padded_vocab_size=50304 when declared.",
            )
        )
    if model_dim > _NATIVE_SIGNED_INT_MAX // 4:
        issues.append(
            _issue(
                "unsupported_training_geometry",
                "template_spec.model_dim makes the exact 4x MLP width overflow "
                "the native signed-int ABI.",
            )
        )
    multiple_of = block.get("multiple_of")
    if multiple_of not in (None, "", 0):
        if type(multiple_of) is not int:
            issues.append(
                _issue(
                    "unsupported_training_geometry",
                    "block_spec.multiple_of must be an exact JSON integer when declared.",
                )
            )
        multiple = _as_int(multiple_of)
        if multiple <= 0 or (model_dim * 4) % multiple:
            issues.append(
                _issue(
                    "unsupported_training_geometry",
                    "block_spec.multiple_of must leave the exact 4x dense MLP width unchanged.",
                )
            )
    declared_contexts: list[tuple[str, int]] = []
    for field in ("seq_len", "max_seq_len", "context_window"):
        value = spec.get(field)
        if value in (None, ""):
            continue
        if type(value) is not int:
            issues.append(
                _issue(
                    "unsupported_training_geometry",
                    f"template_spec.{field} must be an exact JSON integer when declared.",
                )
            )
        parsed = _as_int(value)
        if parsed > _NATIVE_SIGNED_INT_MAX:
            issues.append(
                _issue(
                    "unsupported_training_geometry",
                    f"template_spec.{field} exceeds the native signed-int ABI limit.",
                )
            )
        if parsed <= 0:
            issues.append(
                _issue(
                    "unsupported_training_geometry",
                    f"template_spec.{field} must be positive when declared.",
                )
            )
        else:
            declared_contexts.append((field, parsed))
    if len({value for _field, value in declared_contexts}) > 1:
        rendered = ", ".join(
            f"template_spec.{field}={value}" for field, value in declared_contexts
        )
        issues.append(
            _issue(
                "unsupported_training_geometry",
                "All declared GPT-2 context aliases must resolve to one exact "
                f"training length; got {rendered}.",
            )
        )
    return tuple(issues)


def _normalize_block_path(value: Any) -> str:
    return re.sub(r"block_\d+", "block_*", str(value))


def _shape_hash(graph: Mapping[str, Any]) -> str:
    nodes_payload: list[dict[str, Any]] = []
    nodes = graph.get("nodes", ())
    if isinstance(nodes, Sequence) and not isinstance(nodes, (str, bytes, bytearray)):
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            input_ports = node.get("input_ports", ())
            output_ports = node.get("output_ports", ())
            nodes_payload.append(
                {
                    "instance_id": _normalize_block_path(node.get("instance_id", "")),
                    "kind": str(node.get("kind") or ""),
                    "operation": str(node.get("operation") or ""),
                    "subgraph": (
                        _normalize_block_path(node.get("subgraph"))
                        if node.get("subgraph")
                        else None
                    ),
                    "inputs": [
                        (str(port.get("name") or ""), str(port.get("dtype") or ""))
                        for port in input_ports
                        if isinstance(port, Mapping)
                    ],
                    "outputs": [
                        (str(port.get("name") or ""), str(port.get("dtype") or ""))
                        for port in output_ports
                        if isinstance(port, Mapping)
                    ],
                }
            )
    edges_payload: list[tuple[str, int, str, int]] = []
    edges = graph.get("edges", ())
    if isinstance(edges, Sequence) and not isinstance(edges, (str, bytes, bytearray)):
        for edge in edges:
            if not isinstance(edge, Mapping):
                continue
            edges_payload.append(
                (
                    _normalize_block_path(edge.get("src_node", "")),
                    _as_int(edge.get("src_port"), -1),
                    _normalize_block_path(edge.get("dst_node", "")),
                    _as_int(edge.get("dst_port"), -1),
                )
            )
    payload = {
        "path": _normalize_block_path(graph.get("path", "")),
        "nodes": nodes_payload,
        "edges": edges_payload,
        "inputs": [_normalize_block_path(item) for item in graph.get("input_nodes", ())],
        "outputs": [_normalize_block_path(item) for item in graph.get("output_nodes", ())],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _local_node_id(value: Any) -> str:
    return str(value or "").rsplit("/", 1)[-1]


def _validate_model_stage(
    graph: Mapping[str, Any],
    *,
    num_layers: int,
    softcap: bool,
) -> bool:
    expected_nodes = {
        "ce": "token_cross_entropy",
        "embed_add": "builtin.add",
        "final_norm": "layer_norm",
        "loss_out": "graph.output",
        "pos_embed": "absolute_position_embedding",
        "targets_in": "graph.input",
        "tied_lm_head": "tied_lm_head",
        "token_embed": "token_embedding",
        "tokens_in": "graph.input",
        **({"softcap": "logit_softcap"} if softcap else {}),
        **{f"block_{index}": "subgraph.call" for index in range(num_layers)},
    }
    actual_nodes = {
        str(node.get("instance_id") or ""): str(node.get("operation") or "")
        for node in graph.get("nodes", ())
        if isinstance(node, Mapping)
    }
    if actual_nodes != expected_nodes:
        return False

    expected_edges = {
        ("token_embed", 0, "embed_add", 0),
        ("pos_embed", 0, "embed_add", 1),
        ("embed_add", 0, "block_0", 0),
        (f"block_{num_layers - 1}", 0, "final_norm", 0),
        ("final_norm", 0, "tied_lm_head", 0),
        ("targets_in", 0, "ce", 1),
        ("token_embed", 1, "tied_lm_head", 1),
        ("ce", 0, "loss_out", 0),
        ("tokens_in", 0, "token_embed", 0),
        ("tokens_in", 0, "pos_embed", 0),
        *{
            (f"block_{index}", 0, f"block_{index + 1}", 0)
            for index in range(num_layers - 1)
        },
    }
    if softcap:
        expected_edges.update(
            {
                ("tied_lm_head", 0, "softcap", 0),
                ("softcap", 0, "ce", 0),
            }
        )
    else:
        expected_edges.add(("tied_lm_head", 0, "ce", 0))
    actual_edges = {
        (
            _local_node_id(edge.get("src_node")),
            _as_int(edge.get("src_port"), -1),
            _local_node_id(edge.get("dst_node")),
            _as_int(edge.get("dst_port"), -1),
        )
        for edge in graph.get("edges", ())
        if isinstance(edge, Mapping)
    }
    if actual_edges != expected_edges:
        return False
    return (
        tuple(_local_node_id(item) for item in graph.get("input_nodes", ()))
        == ("tokens_in", "targets_in")
        and tuple(_local_node_id(item) for item in graph.get("output_nodes", ()))
        == ("loss_out",)
    )


def _validate_gpt2_topology(
    topology: Mapping[str, Any],
    selector: str,
    *,
    num_layers: int,
    model_dim: int,
    num_heads: int,
    vocab_size: int,
    sequence_length: int,
) -> tuple[NativeLoweringIssue, ...]:
    if selector == "gpt2_diff":
        head_dim = model_dim // num_heads if num_heads > 0 else 0
        if sequence_length < 16 or sequence_length > _NATIVE_SIGNED_INT_MAX:
            return (
                _issue(
                    "unsupported_training_geometry",
                    "The learned gpt2_diff packed ABI requires max_seq_len in "
                    f"[16, {_NATIVE_SIGNED_INT_MAX}].",
                ),
            )
        if head_dim < 2 or head_dim % 2:
            return (
                _issue(
                    "unsupported_training_geometry",
                    "The learned gpt2_diff packed ABI requires an even head_dim >= 2.",
                ),
            )
        graph_products = (
            sequence_length * model_dim,
            vocab_size * model_dim,
            num_layers * model_dim * model_dim * 16,
        )
        if any(value > _NATIVE_SIGNED_INT64_MAX for value in graph_products):
            return (
                _issue(
                    "unsupported_training_geometry",
                    "The learned gpt2_diff graph geometry exceeds the native "
                    "signed-64-bit tensor/addressing contract.",
                ),
            )
    graphs = {str(graph.get("path") or ""): graph for graph in _active_graphs(topology)}
    model_stage_path = "root/nodes/model/subgraph"
    model_stage = graphs.get(model_stage_path)
    if not isinstance(model_stage, Mapping):
        return (
            _issue(
                "unsupported_training_topology",
                "The active graph is missing the reviewed dense GPT-2 model stage.",
            ),
        )
    # Bound every later range(num_layers) by the already-lowered, source-size-
    # bounded topology.  A tiny graph must not make the planner allocate or
    # iterate billions of expected block paths merely by declaring a huge
    # template_spec.num_layers value.
    actual_block_ids = {
        _local_node_id(node.get("instance_id"))
        for node in model_stage.get("nodes", ())
        if isinstance(node, Mapping)
        and re.fullmatch(r"block_[0-9]+", _local_node_id(node.get("instance_id")))
    }
    if num_layers != len(actual_block_ids) or any(
        f"block_{index}" not in actual_block_ids
        for index in range(len(actual_block_ids))
    ):
        return (
            _issue(
                "unsupported_training_topology",
                "template_spec.num_layers must exactly match the contiguous active block topology.",
            ),
        )
    expected_paths = {"root", "root/nodes/model/subgraph"}
    for index in range(num_layers):
        block_path = f"root/nodes/model/subgraph/nodes/block_{index}/subgraph"
        expected_paths.update(
            {
                block_path,
                f"{block_path}/nodes/attention/subgraph",
                f"{block_path}/nodes/mlp/subgraph",
            }
        )
    if set(graphs) != expected_paths:
        return (
            _issue(
                "unsupported_training_topology",
                "The active graph hierarchy does not match the reviewed dense GPT-2 adapter.",
            ),
        )
    expected_module_configs: dict[str, tuple[dict[str, Any], ...]] = {
        "absolute_position_embedding": (
            {"max_seq_len": sequence_length, "model_dim": model_dim},
        ),
        "differential_attention": (
            {"lambda_init": 0.8, "is_causal": True, "dropout_p": 0.0},
        ),
        "gelu": ({},),
        "layer_norm": ({"model_dim": model_dim, "eps": 1.0e-5},),
        "linear": (
            {"input_dim": model_dim, "output_dim": model_dim, "bias": True},
            {"input_dim": model_dim, "output_dim": model_dim * 4, "bias": True},
            {"input_dim": model_dim * 4, "output_dim": model_dim, "bias": True},
        ),
        "logit_softcap": ({"softcap": 30.0},),
        "merge_heads": ({},),
        "qk_norm": ({"eps": 1.0e-6},),
        "reshape_heads": ({"num_heads": num_heads},),
        "scaled_dot_product_attention": (
            {"is_causal": True, "backend": "sdpa", "dropout_p": 0.0},
        ),
        "tied_lm_head": ({},),
        "token_cross_entropy": (
            {
                "z_loss_coef": (
                    1.0e-4 if selector in {"gpt2_zloss", "gpt2_stable"} else 0.0
                )
            },
        ),
        "token_embedding": ({"vocab_size": vocab_size, "model_dim": model_dim},),
    }
    for graph in graphs.values():
        for edge in graph.get("edges", ()):
            if not isinstance(edge, Mapping):
                continue
            weight = edge.get("weight")
            bias = edge.get("bias")
            if (
                type(weight) is not float
                or type(bias) is not float
                or not math.isfinite(weight)
                or not math.isfinite(bias)
                or weight != 1.0
                or bias != 0.0
            ):
                return (
                    _issue(
                        "unsupported_training_topology",
                        "The production adapter requires unit-weight, zero-bias graph edges; "
                        "the resident trainer does not interpret edge transforms.",
                        path=str(edge.get("dst_node") or _ADAPTER_NODE_PATH),
                    ),
                )
        for node in graph.get("nodes", ()):
            if not isinstance(node, Mapping):
                continue
            operation = str(node.get("operation") or "")
            if operation.startswith("graph.") or operation in {"subgraph.call", "builtin.add"}:
                continue
            if node.get("module_state_sha256") not in (None, ""):
                return (
                    _issue(
                        "unsupported_training_state",
                        "Embedded module state is not consumed by the graph-file trainer adapter.",
                        path=str(node.get("path") or _ADAPTER_NODE_PATH),
                        operation=operation,
                    ),
                )
            allowed_configs = expected_module_configs.get(operation)
            actual_config = dict(_mapping(node.get("module_config")))
            if allowed_configs is None or not any(
                _exact_json_value_matches(actual_config, allowed)
                for allowed in allowed_configs
            ):
                return (
                    _issue(
                        "unsupported_training_configuration",
                        f"Node configuration for {operation!r} is not represented by the {selector} adapter.",
                        path=str(node.get("path") or _ADAPTER_NODE_PATH),
                        operation=operation,
                    ),
                )
    if _shape_hash(graphs["root"]) != _ROOT_SHAPE_SHA256:
        return (
            _issue(
                "unsupported_training_topology",
                "The root training graph interface does not match the reviewed token/target/loss contract.",
                path="root",
                operation="graph",
            ),
        )
    if not _validate_model_stage(
        graphs["root/nodes/model/subgraph"],
        num_layers=num_layers,
        softcap=selector == "gpt2_softcap",
    ):
        return (
            _issue(
                "unsupported_training_topology",
                "The model-stage node/edge contract does not match the reviewed dense GPT-2 adapter.",
            ),
        )
    expected_attention = _GPT2_ATTENTION_SHAPE_BY_SELECTOR[selector]
    for index in range(num_layers):
        block_path = f"root/nodes/model/subgraph/nodes/block_{index}/subgraph"
        if _shape_hash(graphs[block_path]) != _GPT2_BLOCK_SHAPE_SHA256:
            return (
                _issue(
                    "unsupported_training_topology",
                    "The decoder-block node/edge contract is not represented by the production adapter.",
                    path=f"root/nodes/model/subgraph/nodes/block_{index}",
                ),
            )
        attention_path = f"{block_path}/nodes/attention/subgraph"
        if _shape_hash(graphs[attention_path]) != expected_attention:
            return (
                _issue(
                    "unsupported_training_topology",
                    f"The {selector} attention graph does not match its reviewed native adapter.",
                    path=f"{block_path}/nodes/attention",
                ),
            )
        mlp_path = f"{block_path}/nodes/mlp/subgraph"
        if _shape_hash(graphs[mlp_path]) != _GPT2_MLP_SHAPE_SHA256:
            return (
                _issue(
                    "unsupported_training_topology",
                    "The dense GELU MLP graph does not match the reviewed native adapter.",
                    path=f"{block_path}/nodes/mlp",
                ),
            )
    return ()


def _llama_context_geometry(
    manifest: NativeExecutionManifest,
) -> tuple[int, str]:
    """Resolve context from graph metadata, otherwise the family-loop default."""

    spec = _mapping(manifest.model.get("template_spec"))
    declared: list[tuple[str, int]] = []
    for field in ("seq_len", "max_seq_len", "context_window"):
        if field not in spec or spec[field] in (None, ""):
            continue
        value = spec[field]
        if isinstance(value, bool) or not isinstance(value, int):
            return 0, f"template_spec.{field}"
        declared.append((f"template_spec.{field}", value))
    if declared:
        values = {value for _source, value in declared}
        if len(values) != 1:
            return 0, "conflicting-template-context-fields"
        return declared[0][1], declared[0][0]

    manifest_context = manifest.context_limits.get("max_context_tokens")
    if manifest_context not in (None, ""):
        if isinstance(manifest_context, bool) or not isinstance(manifest_context, int):
            return 0, "manifest.context_limits.max_context_tokens"
        return manifest_context, "manifest.context_limits.max_context_tokens"
    return _LLAMA_DEFAULT_CONTEXT_TOKENS, "native-family-trainer-default"


def _llama_padded_vocab_geometry(
    manifest: NativeExecutionManifest,
) -> tuple[int, str]:
    spec = _mapping(manifest.model.get("template_spec"))
    vocab_size = _as_int(spec.get("vocab_size"))
    if "padded_vocab_size" in spec and spec["padded_vocab_size"] not in (None, ""):
        value = spec["padded_vocab_size"]
        if isinstance(value, bool) or not isinstance(value, int):
            return 0, "template_spec.padded_vocab_size"
        return value, "template_spec.padded_vocab_size"
    # The graph describes public embedding/head rows, while the family trainer
    # otherwise defaults to GPT's 50304 physical rows.  Passing the graph vocab
    # explicitly is the only source-faithful default for arbitrary graph sizes.
    return vocab_size, "template_spec.vocab_size"


def _llama_hidden_dim(model: Mapping[str, Any]) -> int:
    spec = _mapping(model.get("template_spec"))
    block = _mapping(spec.get("block_spec"))
    model_dim = _as_int(spec.get("model_dim"))
    multiplier = _as_float(block.get("mlp_multiplier"))
    multiple_of = _as_int(block.get("multiple_of"))
    hidden_dim = max(1, int(model_dim * multiplier))
    if multiple_of > 0:
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def _llama_architecture_provenance(
    manifest: NativeExecutionManifest,
    graph_fingerprint: str,
    selector: str,
) -> dict[str, Any]:
    """Serialize canonical ABI geometry without hiding source-profile identity."""

    spec = _mapping(manifest.model.get("template_spec"))
    template = _mapping(spec.get("template"))
    block = _mapping(spec.get("block_spec"))
    model_dim = _as_int(spec.get("model_dim"))
    num_heads = _as_int(block.get("num_heads"))
    num_kv_heads = _as_int(block.get("num_kv_heads"))
    context, context_source = _llama_context_geometry(manifest)
    padded_vocab_size, padded_vocab_source = _llama_padded_vocab_geometry(manifest)
    geometry = {
        "max_seq_len": context,
        "vocab_size": _as_int(spec.get("vocab_size")),
        "padded_vocab_size": padded_vocab_size,
        "num_layers": _as_int(spec.get("num_layers")),
        "model_dim": model_dim,
        "hidden_dim": _llama_hidden_dim(manifest.model),
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": model_dim // num_heads if num_heads > 0 else 0,
        "rope_theta": _as_float(block.get("rope_theta")),
        "rope_scaling_factor": 1.0,
        "rms_norm_eps": _LLAMA_RMS_NORM_EPS,
        "mlp_multiplier": _as_float(block.get("mlp_multiplier")),
        "multiple_of": _as_int(block.get("multiple_of")),
    }
    return {
        "schema": "neuralfn.native_graph_training.llama_architecture",
        "version": 1,
        "selector": selector,
        "source_selector": selector,
        "source_preset": selector,
        "source_runtime": str(template.get("runtime") or ""),
        "native_template_name": "llama",
        "checkpoint_identity": "llama",
        "graph_fingerprint": graph_fingerprint,
        "geometry": geometry,
        "semantics": {
            "normalization": "rmsnorm",
            "position_encoding": "rope",
            "attention": (
                "dense-mha"
                if num_kv_heads == num_heads
                else ("dense-mqa" if num_kv_heads == 1 else "dense-gqa")
            ),
            "mlp": "gate-first-swiglu",
            "linear_bias": False,
            "dropout_p": 0.0,
            "tie_embeddings": False,
            "rope_scaling_type": "none",
        },
        "sources": {
            "source_selector": "classified-exact-active-topology-and-runtime-profile",
            "source_preset": "classified-exact-active-topology-and-runtime-profile",
            "source_runtime": "template_spec.template.runtime",
            "native_template_name": "canonical-llama-cpp-abi",
            "checkpoint_identity": "canonical-llama-cpp-abi",
            "max_seq_len": context_source,
            "vocab_size": "template_spec.vocab_size",
            "padded_vocab_size": padded_vocab_source,
            "num_layers": "template_spec.num_layers",
            "model_dim": "template_spec.model_dim",
            "hidden_dim": "derived-floor-multiply-then-ceil-to-multiple_of",
            "num_heads": "template_spec.block_spec.num_heads",
            "num_kv_heads": "template_spec.block_spec.num_kv_heads",
            "head_dim": "derived-model_dim-div-num_heads",
            "rope_theta": "template_spec.block_spec.rope_theta",
            "rope_scaling_factor": "template_spec.block_spec.rope_scaling=null",
            "rms_norm_eps": "active-topology-rms_norm.module_config.eps",
            "mlp_multiplier": "template_spec.block_spec.mlp_multiplier",
            "multiple_of": "template_spec.block_spec.multiple_of",
        },
    }


def _standard_moe_architecture_provenance(
    manifest: NativeExecutionManifest,
    graph_fingerprint: str,
    selector: str,
) -> dict[str, Any]:
    """Serialize the exact standard-MoE ABI and source-profile identity."""

    spec = _mapping(manifest.model.get("template_spec"))
    template = _mapping(spec.get("template"))
    block = _mapping(spec.get("block_spec"))
    model_dim = _as_int(spec.get("model_dim"))
    num_heads = _as_int(block.get("num_heads"))
    num_kv_heads = _as_int(block.get("num_kv_heads"))
    context, context_source = _llama_context_geometry(manifest)
    padded_vocab_size, padded_vocab_source = _llama_padded_vocab_geometry(manifest)
    graph_multiple = block.get("multiple_of")
    native_multiple = 0 if graph_multiple is None else _as_int(graph_multiple)
    native_template_name = "mixllama-fast" if selector == "mixllama_fast" else "mixllama"
    geometry = {
        "max_seq_len": context,
        "vocab_size": _as_int(spec.get("vocab_size")),
        "padded_vocab_size": padded_vocab_size,
        "num_layers": _as_int(spec.get("num_layers")),
        "model_dim": model_dim,
        "hidden_dim": _llama_hidden_dim(manifest.model),
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": model_dim // num_heads if num_heads > 0 else 0,
        "experts": _as_int(block.get("experts")),
        "top_k": _as_int(block.get("top_k")),
        "rope_theta": _as_float(block.get("rope_theta")),
        "rope_scaling_factor": 1.0,
        "rms_norm_eps": _LLAMA_RMS_NORM_EPS,
        "mlp_multiplier": _as_float(block.get("mlp_multiplier")),
        "multiple_of": native_multiple,
        "router_aux_loss_coef": _as_float(block.get("router_aux_loss_coef")),
    }
    return {
        "schema": "neuralfn.native_graph_training.standard_moe_architecture",
        "version": 1,
        "selector": selector,
        "source_selector": selector,
        "source_preset": selector,
        "source_runtime": str(template.get("runtime") or ""),
        "native_template_name": native_template_name,
        "checkpoint_identity": "mixllama",
        "graph_fingerprint": graph_fingerprint,
        "geometry": geometry,
        "semantics": {
            "norm_type": "rmsnorm",
            "mlp_type": "moe",
            "pos_encoding": "rope",
            "attention_variant": "dense",
            "residual_type": "add",
            "router_score_fn": "softmax",
            "router_selection": "topk_renormalized",
            "moe_balance_mode": "aux_loss",
            "linear_bias": False,
            "dropout_p": 0.0,
            "tie_embeddings": False,
            "use_qk_norm": False,
            "shared_experts": 0,
        },
        "sources": {
            "source_selector": "classified-exact-active-topology-runtime-and-visible-alias",
            "source_preset": "classified-exact-active-topology-runtime-and-visible-alias",
            "source_runtime": "template_spec.template.runtime",
            "native_template_name": "canonical-standard-moe-cpp-abi",
            "checkpoint_identity": "canonical-standard-moe-cpp-abi",
            "max_seq_len": context_source,
            "vocab_size": "template_spec.vocab_size",
            "padded_vocab_size": padded_vocab_source,
            "num_layers": "template_spec.num_layers",
            "model_dim": "template_spec.model_dim",
            "hidden_dim": "derived-floor-multiply-then-optional-ceil-to-multiple_of",
            "num_heads": "template_spec.block_spec.num_heads",
            "num_kv_heads": "template_spec.block_spec.num_kv_heads",
            "head_dim": "derived-model_dim-div-num_heads",
            "experts": "template_spec.block_spec.experts",
            "top_k": "template_spec.block_spec.top_k",
            "rope_theta": "template_spec.block_spec.rope_theta",
            "rope_scaling_factor": "template_spec.block_spec.rope_scaling=null",
            "rms_norm_eps": "active-topology-rms_norm.module_config.eps",
            "mlp_multiplier": "template_spec.block_spec.mlp_multiplier",
            "multiple_of": (
                "native-zero-sentinel-for-template_spec.block_spec.multiple_of=null"
                if graph_multiple is None
                else "template_spec.block_spec.multiple_of"
            ),
            "router_aux_loss_coef": "template_spec.block_spec.router_aux_loss_coef",
        },
    }


def _validate_llama_training_contract(
    manifest: NativeExecutionManifest,
) -> tuple[NativeLoweringIssue, ...]:
    """Require the exact canonical resident profile plus trainer-safe payloads."""

    proof = capability_proof_for(manifest.model, manifest.topology)
    if not (
        proof.model_family == "llama"
        and proof.architecture_persistence_proven
        and proof.native_forward_proven
        and "resident-llama-cpu-reference-abi-v1" in proof.evidence
    ):
        return (
            _issue(
                "unsupported_training_adapter",
                "The graph does not match the canonical LLaMA RMSNorm/RoPE/GQA/"
                "dense-attention/gate-first-SwiGLU/biasless/dropout0/untied contract.",
            ),
        )

    context, context_source = _llama_context_geometry(manifest)
    if context <= 0:
        return (
            _issue(
                "unsupported_training_geometry",
                f"{context_source} must resolve to one positive context length.",
            ),
        )
    spec = _mapping(manifest.model.get("template_spec"))
    vocab_size = _as_int(spec.get("vocab_size"))
    padded_vocab_size, padded_source = _llama_padded_vocab_geometry(manifest)
    if padded_vocab_size < vocab_size:
        return (
            _issue(
                "unsupported_training_geometry",
                f"{padded_source} must be at least template_spec.vocab_size.",
            ),
        )

    for graph in _active_graphs(manifest.topology):
        for edge in graph.get("edges", ()):
            if not isinstance(edge, Mapping):
                continue
            if edge.get("weight") != 1.0 or edge.get("bias") != 0.0:
                return (
                    _issue(
                        "unsupported_training_topology",
                        "The canonical LLaMA trainer requires unit-weight, zero-bias graph edges; "
                        "it does not interpret edge transforms.",
                        path=str(edge.get("dst_node") or _ADAPTER_NODE_PATH),
                    ),
                )
        for node in graph.get("nodes", ()):
            if not isinstance(node, Mapping):
                continue
            if node.get("module_state_sha256") not in (None, ""):
                return (
                    _issue(
                        "unsupported_training_state",
                        "Embedded module state is not consumed by the canonical LLaMA graph-file adapter.",
                        path=str(node.get("path") or _ADAPTER_NODE_PATH),
                        operation=str(node.get("operation") or "subgraph.call"),
                    ),
                )
    return ()


def _validate_standard_moe_training_contract(
    manifest: NativeExecutionManifest,
) -> tuple[NativeLoweringIssue, ...]:
    """Require the exact graph-faithful standard-MoE resident/trainer profile."""

    proof = capability_proof_for(manifest.model, manifest.topology)
    if not (
        proof.model_family == "mixllama"
        and proof.architecture_persistence_proven
        and proof.native_forward_proven
        and "resident-standard-moe-cpu-reference-abi-v1" in proof.evidence
        and "standard-moe-exact-router-aux-loss-gradient-v1" in proof.evidence
    ):
        return (
            _issue(
                "unsupported_training_adapter",
                "The graph does not match the canonical standard-MoE RMSNorm/RoPE/GQA/"
                "softmax-top-k-renormalized/SwiGLU-expert/aux-loss/biasless/dropout0/"
                "untied contract.",
            ),
        )
    context, context_source = _llama_context_geometry(manifest)
    if context <= 0:
        return (
            _issue(
                "unsupported_training_geometry",
                f"{context_source} must resolve to one positive context length.",
            ),
        )
    spec = _mapping(manifest.model.get("template_spec"))
    vocab_size = _as_int(spec.get("vocab_size"))
    padded_vocab_size, padded_source = _llama_padded_vocab_geometry(manifest)
    if padded_vocab_size < vocab_size:
        return (
            _issue(
                "unsupported_training_geometry",
                f"{padded_source} must be at least template_spec.vocab_size.",
            ),
        )
    for graph in _active_graphs(manifest.topology):
        for edge in graph.get("edges", ()):
            if not isinstance(edge, Mapping):
                continue
            if edge.get("weight") != 1.0 or edge.get("bias") != 0.0:
                return (
                    _issue(
                        "unsupported_training_topology",
                        "The standard-MoE trainer requires unit-weight, zero-bias graph edges; "
                        "it does not interpret edge transforms.",
                        path=str(edge.get("dst_node") or _ADAPTER_NODE_PATH),
                    ),
                )
        for node in graph.get("nodes", ()):
            if not isinstance(node, Mapping):
                continue
            if node.get("module_state_sha256") not in (None, ""):
                return (
                    _issue(
                        "unsupported_training_state",
                        "Embedded module state is not consumed by the standard-MoE graph-file adapter.",
                        path=str(node.get("path") or _ADAPTER_NODE_PATH),
                        operation=str(node.get("operation") or "subgraph.call"),
                    ),
                )
    return ()


def _training_issues(
    manifest: NativeExecutionManifest,
    *,
    family: str,
    selector: str,
    trainer: NativeTrainerSpec | None,
    adapter: NativeGraphTrainingAdapter | None,
) -> tuple[NativeLoweringIssue, ...]:
    topology = manifest.topology
    if trainer is None or not trainer.trainer_registered:
        return (
            _issue(
                "native_trainer_unregistered",
                f"No canonical native trainer is registered for family {family!r}.",
            ),
        )
    if adapter is None:
        if selector in {"gpt2_modern", "nanogpt_modern"}:
            located = _first_operation_path(topology, {"rms_norm", "rotary_embedding", "geglu"})
            path, operation = located or (_ADAPTER_NODE_PATH, "subgraph.call")
            return (
                _issue(
                    "unsupported_training_adapter",
                    f"{selector} requires RMSNorm/RoPE/GeGLU semantics that the current dense trainer does not implement.",
                    path=path,
                    operation=operation,
                ),
            )
        if selector in {"nanogpt", "nanogpt_megakernel"}:
            located = _first_operation_path(topology, {"dropout"})
            path, operation = located or (_ADAPTER_NODE_PATH, "subgraph.call")
            return (
                _issue(
                    "unsupported_training_adapter",
                    f"{selector} requires NanoGPT's bias-free linear/dropout contract; the current dense trainer uses biased GPT-2 layers and no dropout.",
                    path=path,
                    operation=operation,
                ),
            )
        if not trainer.architecture_persistence_proven:
            located = _first_operation_path(
                topology,
                {
                    "rms_norm",
                    "rotary_embedding",
                    "swiglu",
                    "expert_dispatch",
                    "mamba",
                    "ttt_linear",
                    "universal_transformer",
                    "mask_scheduler",
                },
            )
            path, operation = located or (_ADAPTER_NODE_PATH, "subgraph.call")
            return (
                _issue(
                    "architecture_persistence_unproven",
                    f"The {family} executable has no proved graph-faithful architecture persistence adapter; its transition sampler is diagnostic only.",
                    path=path,
                    operation=operation,
                ),
            )
        return (
            _issue(
                "unsupported_training_adapter",
                f"No reviewed production graph adapter matches selector {selector or family!r}.",
            ),
        )

    if not adapter.architecture_persistence_proven and selector != "gpt2_diff":
        return (
            _issue(
                "architecture_persistence_unproven",
                f"The reviewed {selector or family!r} adapter has no proved "
                "architecture persistence contract.",
            ),
        )

    if selector in _LLAMA_TRAINING_SELECTORS:
        return _validate_llama_training_contract(manifest)
    if selector in _STANDARD_MOE_TRAINING_SELECTORS:
        return _validate_standard_moe_training_contract(manifest)

    # gpt2_diff deliberately remains unproved in the generic capability
    # registry because classification happens before this exact validator.  Its
    # learned-v2 graph-training path is promoted only here, after both the
    # configuration and active-topology contracts pass and a materialized proof
    # can bind the launch snapshot.  Migration and resident inference stay
    # independently fail-closed.
    issues = list(_validate_gpt2_configuration(manifest.model, selector))
    if not issues:
        spec = _mapping(manifest.model.get("template_spec"))
        issues.extend(
            _validate_gpt2_topology(
                topology,
                selector,
                num_layers=_as_int(spec.get("num_layers")),
                model_dim=_as_int(spec.get("model_dim")),
                num_heads=_as_int(_mapping(spec.get("block_spec")).get("num_heads")),
                vocab_size=_as_int(spec.get("vocab_size")),
                sequence_length=_gpt2_sequence_length(manifest),
            )
        )
    return tuple(issues)


def _gpt2_sequence_length(manifest: NativeExecutionManifest) -> int:
    spec = _mapping(manifest.model.get("template_spec"))
    for field in ("seq_len", "max_seq_len", "context_window"):
        value = spec.get(field)
        if value not in (None, ""):
            return _as_int(value)
    located: list[int] = []
    for graph in _active_graphs(manifest.topology):
        for node in graph.get("nodes", ()):
            if not isinstance(node, Mapping):
                continue
            if str(node.get("operation") or "") == "absolute_position_embedding":
                located.append(_as_int(_mapping(node.get("module_config")).get("max_seq_len")))
    unique = sorted({value for value in located if value > 0})
    return unique[0] if len(unique) == 1 else 0


def _gpt2_diff_training_proof_contract(
    manifest: NativeExecutionManifest,
    source_graph_sha256: str,
) -> dict[str, Any]:
    """Build the exact source-bound contract after rerunning both validators."""

    selector = classify_native_graph_training_selector(manifest.model, manifest.topology)
    if selector != "gpt2_diff":
        raise RuntimeError(
            "A gpt2_diff native-training proof cannot attest a different selector."
        )
    if re.fullmatch(r"[0-9a-f]{64}", source_graph_sha256) is None:
        raise RuntimeError("A gpt2_diff native-training proof requires a canonical source SHA-256.")
    spec = _mapping(manifest.model.get("template_spec"))
    block = _mapping(spec.get("block_spec"))
    model_dim = _as_int(spec.get("model_dim"))
    num_layers = _as_int(spec.get("num_layers"))
    num_heads = _as_int(block.get("num_heads"))
    num_kv_raw = block.get("num_kv_heads")
    num_kv_heads = num_heads if num_kv_raw in (None, "") else _as_int(num_kv_raw)
    vocab_size = _as_int(spec.get("vocab_size"))
    padded_vocab_raw = spec.get("padded_vocab_size")
    padded_vocab_size = (
        50304
        if padded_vocab_raw in (None, "")
        else _as_int(padded_vocab_raw)
    )
    max_seq_len = _gpt2_sequence_length(manifest)
    issues = list(_validate_gpt2_configuration(manifest.model, selector))
    if not issues:
        issues.extend(
            _validate_gpt2_topology(
                manifest.topology,
                selector,
                num_layers=num_layers,
                model_dim=model_dim,
                num_heads=num_heads,
                vocab_size=vocab_size,
                sequence_length=max_seq_len,
            )
        )
    if issues:
        first = issues[0]
        raise RuntimeError(
            "Refusing to issue a gpt2_diff native-training proof after exact "
            f"validation failed at {first.path}: {first.message}"
        )
    return {
        "adapter_mode": "validated-dense-graph-file-v1",
        "attention_shape_sha256": _GPT2_ATTENTION_SHAPE_BY_SELECTOR[selector],
        "block_shape_sha256": _GPT2_BLOCK_SHAPE_SHA256,
        "configuration_contract": _GPT2_DIFF_CONFIGURATION_CONTRACT,
        "geometry": {
            "head_dim": model_dim // num_heads,
            "max_seq_len": max_seq_len,
            "mlp_hidden_dim": model_dim * 4,
            "model_dim": model_dim,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "num_layers": num_layers,
            "padded_vocab_size": padded_vocab_size,
            "vocab_size": vocab_size,
        },
        "mlp_shape_sha256": _GPT2_MLP_SHAPE_SHA256,
        "passed": True,
        "root_shape_sha256": _ROOT_SHAPE_SHA256,
        "schema": _TRAINING_PROOF_SCHEMA,
        "source_graph_sha256": source_graph_sha256,
        "topology_contract": _GPT2_DIFF_TOPOLOGY_CONTRACT,
        "training_selector": selector,
        "validator_contract": _GPT2_DIFF_VALIDATOR_CONTRACT,
        "version": _TRAINING_PROOF_VERSION,
    }


def _gpt2_diff_training_proof_bytes(
    manifest: NativeExecutionManifest,
    source_graph_sha256: str,
) -> tuple[bytes, str]:
    """Serialize the strict raw-contract-hash envelope consumed by C++."""

    contract = _gpt2_diff_training_proof_contract(manifest, source_graph_sha256)
    contract_bytes = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    contract_sha256 = hashlib.sha256(contract_bytes).hexdigest()
    envelope = (
        b'{"contract":'
        + contract_bytes
        + b',"contract_sha256":"'
        + contract_sha256.encode("ascii")
        + b'"}\n'
    )
    return envelope, contract_sha256


def _adapter_trainer_arguments(
    manifest: NativeExecutionManifest | None,
    selector: str,
    adapter: NativeGraphTrainingAdapter | None,
    graph_fingerprint: str,
    graph_preflight_proof: Path | None = None,
) -> tuple[str, ...]:
    if manifest is None or adapter is None:
        return ()
    spec = _mapping(manifest.model.get("template_spec"))
    block = _mapping(spec.get("block_spec"))
    if selector in _LLAMA_TRAINING_SELECTORS:
        if re.fullmatch(r"[0-9a-f]{64}", graph_fingerprint) is None:
            return ()
        provenance = _llama_architecture_provenance(
            manifest,
            graph_fingerprint,
            selector,
        )
        geometry = _mapping(provenance.get("geometry"))
        return (
            "--template-name",
            "llama",
            "--num-layers",
            str(_as_int(geometry.get("num_layers"))),
            "--model-dim",
            str(_as_int(geometry.get("model_dim"))),
            "--hidden-dim",
            str(_as_int(geometry.get("hidden_dim"))),
            "--mlp-multiplier",
            format(_as_float(geometry.get("mlp_multiplier")), ".17g"),
            "--multiple-of",
            str(_as_int(geometry.get("multiple_of"))),
            "--num-heads",
            str(_as_int(geometry.get("num_heads"))),
            "--num-kv-heads",
            str(_as_int(geometry.get("num_kv_heads"))),
            "--vocab-size",
            str(_as_int(geometry.get("vocab_size"))),
            "--padded-vocab-size",
            str(_as_int(geometry.get("padded_vocab_size"))),
            "--train-seq-len",
            str(_as_int(geometry.get("max_seq_len"))),
            "--rope-theta",
            format(_as_float(geometry.get("rope_theta")), ".17g"),
            "--rope-factor",
            format(_as_float(geometry.get("rope_scaling_factor")), ".17g"),
            "--graph-fingerprint",
            graph_fingerprint,
        )
    if selector in _STANDARD_MOE_TRAINING_SELECTORS:
        if re.fullmatch(r"[0-9a-f]{64}", graph_fingerprint) is None:
            return ()
        provenance = _standard_moe_architecture_provenance(
            manifest,
            graph_fingerprint,
            selector,
        )
        geometry = _mapping(provenance.get("geometry"))
        return (
            "--template-name",
            str(provenance["native_template_name"]),
            "--num-layers",
            str(_as_int(geometry.get("num_layers"))),
            "--model-dim",
            str(_as_int(geometry.get("model_dim"))),
            "--hidden-dim",
            str(_as_int(geometry.get("hidden_dim"))),
            "--mlp-multiplier",
            format(_as_float(geometry.get("mlp_multiplier")), ".17g"),
            "--multiple-of",
            str(_as_int(geometry.get("multiple_of"))),
            "--num-heads",
            str(_as_int(geometry.get("num_heads"))),
            "--num-kv-heads",
            str(_as_int(geometry.get("num_kv_heads"))),
            "--vocab-size",
            str(_as_int(geometry.get("vocab_size"))),
            "--padded-vocab-size",
            str(_as_int(geometry.get("padded_vocab_size"))),
            "--train-seq-len",
            str(_as_int(geometry.get("max_seq_len"))),
            "--rope-theta",
            format(_as_float(geometry.get("rope_theta")), ".17g"),
            "--rope-factor",
            format(_as_float(geometry.get("rope_scaling_factor")), ".17g"),
            "--experts",
            str(_as_int(geometry.get("experts"))),
            "--top-k",
            str(_as_int(geometry.get("top_k"))),
            "--layers-per-expert",
            "1",
            "--router-aux-loss-coef",
            format(_as_float(geometry.get("router_aux_loss_coef")), ".17g"),
            "--graph-fingerprint",
            graph_fingerprint,
        )
    arguments = [
        "--template-name",
        selector,
        "--num-layers",
        str(_as_int(spec.get("num_layers"))),
        "--train-seq-len",
        str(_gpt2_sequence_length(manifest)),
        "--native-cuda-activation",
        "moa" if selector == "gpt2_moa" else "gelu",
    ]
    if selector in {"gpt2_diff", "gpt2_moa"}:
        if re.fullmatch(r"[0-9a-f]{64}", graph_fingerprint) is None:
            return ()
        if selector == "gpt2_moa":
            arguments.extend(
                ["--moa-interval", str(_as_int(block.get("moa_interval")))]
            )
        arguments.extend(["--graph-fingerprint", graph_fingerprint])
    if selector == "gpt2_diff" and graph_preflight_proof is not None:
        arguments.extend(
            ["--graph-preflight-proof", str(graph_preflight_proof)]
        )
    return tuple(arguments)


def _blocking_reasons(
    report: NativeCompatibilityReport,
    training_issues: Sequence[NativeLoweringIssue],
) -> tuple[str, ...]:
    reasons = [
        f"{issue.code}:{issue.path}"
        for issue in (*report.issues, *training_issues)
        if issue.severity == "error"
    ]
    return tuple(dict.fromkeys(reasons))


def _write_exclusive(path: Path, payload: bytes, *, text: bool = False) -> None:
    mode = "x" if text else "xb"
    kwargs: dict[str, Any] = {"encoding": "utf-8"} if text else {}
    with path.open(mode, **kwargs) as handle:
        handle.write(payload.decode("utf-8") if text else payload)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _materialize_training_sidecars(
    plan: NativeGraphTrainPlan,
    *,
    artifact_dir: Path,
    source_bytes: bytes,
) -> None:
    retained_fingerprint = hashlib.sha256(source_bytes).hexdigest()
    if retained_fingerprint != plan.compatibility_report.graph_fingerprint:
        raise RuntimeError(
            "Retained native graph source bytes do not match the validated training plan."
        )
    current_source_bytes = _read_source_graph_snapshot(plan.source_graph)
    fingerprint = hashlib.sha256(current_source_bytes).hexdigest()
    if fingerprint != plan.compatibility_report.graph_fingerprint:
        raise RuntimeError(
            "Source graph changed during native training preflight; refusing to launch a stale validation result."
        )
    proof_bytes: bytes | None = None
    proof_contract_sha256: str | None = None
    if plan.graph_preflight_proof is not None:
        expected_proof_path = artifact_dir / "native-training-proof.json"
        if plan.graph_preflight_proof.resolve() != expected_proof_path.resolve():
            raise RuntimeError("Native graph proof path does not match the training plan.")
        if (
            plan.training_selector != "gpt2_diff"
            or not plan.execution_ready
            or plan.manifest is None
        ):
            raise RuntimeError(
                "Only an execution-ready exact gpt2_diff plan may issue a native-training proof."
            )
        strict_issue = _strict_graph_json_issue(source_bytes)
        if strict_issue is not None:
            raise RuntimeError(strict_issue.message)
        proof_bytes, proof_contract_sha256 = _gpt2_diff_training_proof_bytes(
            plan.manifest,
            fingerprint,
        )
        if proof_contract_sha256 != plan.artifact_metadata.get(
            "graph_preflight_proof_contract_sha256"
        ):
            raise RuntimeError(
                "Native graph proof contract changed during training materialization."
            )
    snapshot_path = artifact_dir / "source-graph.json"
    if snapshot_path.resolve() != plan.launch_graph.resolve():
        raise RuntimeError("Native graph snapshot path does not match the training plan.")
    _write_exclusive(snapshot_path, source_bytes)
    if proof_bytes is not None and plan.graph_preflight_proof is not None:
        _write_exclusive(plan.graph_preflight_proof, proof_bytes)
    plan_bytes = (
        json.dumps(plan.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _write_exclusive(artifact_dir / "native-training-plan.json", plan_bytes)


def plan_native_graph_training(
    graph_path: str | Path,
    *,
    artifact_dir: str | Path | None = None,
    materialize: bool = False,
) -> NativeGraphTrainPlan:
    """Safely compile, classify, and optionally snapshot a training graph.

    Materialization requires a new explicit directory.  It writes the Native
    IR manifest/report, an exact validated source-byte snapshot, and a training
    plan sidecar.  Existing paths are never overwritten.  The source graph's
    serialization is neither rewritten nor used to construct source-executing
    ``NeuronDef`` objects.
    """

    if materialize and artifact_dir is None:
        raise ValueError("materialize=True requires an explicit artifact_dir.")

    source_graph = Path(graph_path).expanduser().resolve()
    source_bytes = _read_source_graph_snapshot(source_graph)
    requested_artifact_dir = (
        Path(artifact_dir).expanduser().resolve() if artifact_dir is not None else None
    )
    migration_destination = requested_artifact_dir or _unused_dry_run_destination()
    migration = migrate_graph_to_native(
        source_graph,
        output_dir=migration_destination,
        dry_run=not materialize,
        _source_bytes=source_bytes,
    )

    manifest = migration.manifest
    if manifest is None:
        family = "unknown"
        selector = ""
        trainer = None
        adapter = None
        training_issues: tuple[NativeLoweringIssue, ...] = ()
    else:
        classification = classify_native_model(manifest.model, manifest.topology)
        family = str(classification["model_family"])
        selector = classify_native_graph_training_selector(manifest.model, manifest.topology)
        trainer = _trainer_for_family(family)
        adapter = _adapter_for_selector(selector)
        training_issues = (
            _training_issues(
                manifest,
                family=family,
                selector=selector,
                trainer=trainer,
                adapter=adapter,
            )
            if migration.report.compatible
            else ()
        )
        if migration.report.compatible and selector == "gpt2_diff":
            if hashlib.sha256(source_bytes).hexdigest() != migration.report.graph_fingerprint:
                strict_issue = _issue(
                    "source_graph_changed_during_preflight",
                    "Retained source bytes changed between Native IR validation and "
                    "the exact graph-training proof gate.",
                    path="root",
                    operation="graph",
                )
            else:
                strict_issue = _strict_graph_json_issue(source_bytes)
            if strict_issue is not None:
                training_issues = (strict_issue, *training_issues)

    trainer_registered = bool(trainer is not None and trainer.trainer_registered)
    exact_gpt2_diff_training_proven = bool(
        selector == "gpt2_diff"
        and adapter is not None
        and migration.report.compatible
        and not any(issue.severity == "error" for issue in training_issues)
    )
    persistence_proven = bool(
        adapter is not None
        and (
            adapter.architecture_persistence_proven
            or exact_gpt2_diff_training_proven
        )
    )
    trainer_consumes_native_ir = bool(
        adapter is not None and adapter.trainer_consumes_native_ir
    )
    graph_fingerprint = migration.report.graph_fingerprint
    execution_ready = bool(
        migration.report.compatible
        and not any(issue.severity == "error" for issue in training_issues)
        and trainer_registered
        and persistence_proven
        and adapter is not None
    )
    materialized_dir = migration.output_dir
    graph_preflight_proof = (
        materialized_dir / "native-training-proof.json"
        if (
            materialized_dir is not None
            and execution_ready
            and exact_gpt2_diff_training_proven
        )
        else None
    )
    proof_contract_sha256: str | None = None
    if graph_preflight_proof is not None and manifest is not None:
        _proof_bytes, proof_contract_sha256 = _gpt2_diff_training_proof_bytes(
            manifest,
            graph_fingerprint,
        )
    trainer_arguments = _adapter_trainer_arguments(
        manifest,
        selector,
        adapter,
        graph_fingerprint,
        graph_preflight_proof,
    )
    snapshot_path = (
        materialized_dir / "source-graph.json"
        if materialized_dir is not None
        else source_graph
    )
    artifact_metadata: dict[str, Any] = {
        "kind": "native-graph-training-preflight",
        "native_ir_schema": manifest.schema if manifest is not None else None,
        "native_ir_version": manifest.version if manifest is not None else None,
        "graph_fingerprint": migration.report.graph_fingerprint,
        "source_graph": str(source_graph),
        "source_graph_serialization_changed": False,
        "requested_artifact_dir": (
            str(requested_artifact_dir) if requested_artifact_dir is not None else None
        ),
        "materialized": materialized_dir is not None,
        "artifact_dir": str(materialized_dir) if materialized_dir is not None else None,
        "manifest_path": (
            str(materialized_dir / "native-execution-manifest.json")
            if materialized_dir is not None
            else None
        ),
        "compatibility_report_path": (
            str(materialized_dir / "compatibility-report.json")
            if materialized_dir is not None
            else None
        ),
        "training_plan_path": (
            str(materialized_dir / "native-training-plan.json")
            if materialized_dir is not None
            else None
        ),
        "graph_preflight_proof_path": (
            str(graph_preflight_proof)
            if graph_preflight_proof is not None
            else None
        ),
        "graph_preflight_proof_schema": (
            _TRAINING_PROOF_SCHEMA if graph_preflight_proof is not None else None
        ),
        "graph_preflight_proof_version": (
            _TRAINING_PROOF_VERSION if graph_preflight_proof is not None else None
        ),
        "graph_preflight_proof_validator_contract": (
            _GPT2_DIFF_VALIDATOR_CONTRACT
            if graph_preflight_proof is not None
            else None
        ),
        "graph_preflight_proof_contract_sha256": proof_contract_sha256,
        "source_graph_snapshot_path": (
            str(snapshot_path) if materialized_dir is not None else None
        ),
        "weights_path": None,
        "trainer_family": family,
        "training_selector": selector,
        "native_target": trainer.native_target if trainer is not None else None,
        "adapter_mode": adapter.adapter_mode if adapter is not None else None,
        "adapter_evidence": list(adapter.evidence) if adapter is not None else [],
        "architecture_provenance": (
            _llama_architecture_provenance(manifest, graph_fingerprint, selector)
            if (
                manifest is not None
                and selector in _LLAMA_TRAINING_SELECTORS
                and adapter is not None
            )
            else _standard_moe_architecture_provenance(
                manifest, graph_fingerprint, selector
            )
            if (
                manifest is not None
                and selector in _STANDARD_MOE_TRAINING_SELECTORS
                and adapter is not None
            )
            else None
        ),
        "trainer_arguments": list(trainer_arguments),
        "graph_preflight_enforced": True,
        "trainer_consumes_native_ir": trainer_consumes_native_ir,
        "execution_ready": execution_ready,
        "training_issue_count": len(training_issues),
    }

    plan = NativeGraphTrainPlan(
        source_graph=source_graph,
        launch_graph=snapshot_path,
        graph_preflight_proof=graph_preflight_proof,
        trainer_family=family,
        training_selector=selector,
        native_target=trainer.native_target if trainer is not None else None,
        adapter_mode=adapter.adapter_mode if adapter is not None else None,
        trainer_arguments=trainer_arguments,
        trainer_registered=trainer_registered,
        architecture_persistence_proven=persistence_proven,
        execution_ready=execution_ready,
        trainer_consumes_native_ir=trainer_consumes_native_ir,
        graph_preflight_enforced=True,
        manifest=manifest,
        compatibility_report=migration.report,
        training_issues=training_issues,
        artifact_metadata=artifact_metadata,
        blockers=_blocking_reasons(migration.report, training_issues),
    )
    if materialized_dir is not None:
        _materialize_training_sidecars(
            plan,
            artifact_dir=materialized_dir,
            source_bytes=source_bytes,
        )
    return plan


def preflight_native_graph_training(
    graph_path: str | Path,
    *,
    artifact_dir: str | Path | None = None,
    materialize: bool = False,
) -> NativeGraphTrainPlan:
    """Alias spelling for callers that present this planner as preflight."""

    return plan_native_graph_training(
        graph_path,
        artifact_dir=artifact_dir,
        materialize=materialize,
    )


__all__ = [
    "NativeGraphTrainPlan",
    "plan_native_graph_training",
    "preflight_native_graph_training",
]
