from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .graph import NeuronGraph

if TYPE_CHECKING:
    import torch
    from .torch_backend import CompiledTorchGraph


@lru_cache(maxsize=1)
def _load_torch_inference_stack():
    try:
        import torch
        from .torch_backend import CompiledTorchGraph
    except ImportError as exc:
        raise ImportError(
            "neuralfn.inference is importable in the lean native SDK, but "
            "legacy .pt checkpoint export/import and InferenceCache require "
            "PyTorch to be installed explicitly."
        ) from exc
    return torch, CompiledTorchGraph


def _checkpoint_metadata_for_graph(graph: NeuronGraph) -> dict[str, Any]:
    torch_config = dict(getattr(graph, "torch_config", {}) or {})
    template_spec = dict(torch_config.get("template_spec", {}) or {})
    template = dict(template_spec.get("template", {}) or {})
    runtime = str(template.get("runtime", "")).strip().lower()
    metadata: dict[str, Any] = {}
    if runtime:
        metadata["template_runtime"] = runtime
    metadata["graph_topology_sha256"] = _graph_topology_sha256(graph)
    return metadata


def load_pt_checkpoint(
    path: str | Path,
    *,
    map_location: str | "torch.device" | None = "cpu",
) -> tuple[dict[str, "torch.Tensor"], dict[str, Any]]:
    torch, _CompiledTorchGraph = _load_torch_inference_stack()
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        return dict(checkpoint["state_dict"]), dict(checkpoint.get("checkpoint_metadata", {}) or {})
    if isinstance(checkpoint, dict):
        return dict(checkpoint), {}
    raise TypeError(f"Unsupported checkpoint payload type {type(checkpoint)!r} in {path!r}.")


def export_to_pt(graph: NeuronGraph, path: str | Path) -> None:
    """Export the weights of a compiled or uncompiled torch-based NeuronGraph to a .pt file."""
    torch, CompiledTorchGraph = _load_torch_inference_stack()
    compiled = CompiledTorchGraph(graph)
    state_dict = compiled.state_dict()
    checkpoint = {
        "state_dict": state_dict,
        "checkpoint_metadata": _checkpoint_metadata_for_graph(graph),
    }
    torch.save(checkpoint, path)


def import_from_pt(graph: NeuronGraph, path: str | Path) -> None:
    """Import weights from a .pt file into a NeuronGraph's module_state."""
    _torch, CompiledTorchGraph = _load_torch_inference_stack()
    state_dict, _checkpoint_metadata = load_pt_checkpoint(path)
    compiled = CompiledTorchGraph(graph)
    compiled.load_state_dict(state_dict)
    compiled.sync_state_back(graph)


# ---------------------------------------------------------------------------
# Adapter-only checkpointing (LoRA / qLoRA)
# ---------------------------------------------------------------------------

ADAPTER_CHECKPOINT_FORMAT = "neuralfn.adapter.v1"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(tensor: "torch.Tensor") -> str:
    torch, _CompiledTorchGraph = _load_torch_inference_stack()
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _scrub_graph_payload_for_topology(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove weights and run-local paths while preserving executable topology."""
    result = copy.deepcopy(payload)

    def visit(graph_payload: dict[str, Any]) -> None:
        torch_config = graph_payload.get("torch_config")
        if isinstance(torch_config, dict):
            torch_config.pop("finetune_spec", None)
            template_spec = torch_config.get("template_spec")
            if isinstance(template_spec, dict):
                template_spec.pop("finetune", None)
            for key in (
                "device",
                "tile_cuda_report_path",
                "dataset_names",
                "checkpoint_path",
                "resume_checkpoint",
                "amp_dtype",
                "drop_last",
                "respect_epoch_boundaries",
                "optimization_method",
                "resolved_lr_decay_iters",
                "resolved_min_lr",
            ):
                torch_config.pop(key, None)
        for node in graph_payload.get("nodes", {}).values():
            neuron_def = node.get("neuron_def", {}) if isinstance(node, dict) else {}
            if not isinstance(neuron_def, dict):
                continue
            neuron_def.pop("id", None)
            neuron_def["module_state"] = ""
            subgraph = neuron_def.get("subgraph")
            if isinstance(subgraph, dict):
                visit(subgraph)
        for versions in graph_payload.get("variant_library", {}).values():
            if not isinstance(versions, dict):
                continue
            for subgraph in versions.values():
                if isinstance(subgraph, dict):
                    visit(subgraph)

    visit(result)
    return result


def _graph_topology_sha256(graph: NeuronGraph) -> str:
    canonical = json.dumps(
        _scrub_graph_payload_for_topology(graph.to_dict()),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _finetune_metadata(graph: NeuronGraph) -> dict[str, Any]:
    raw = (getattr(graph, "torch_config", {}) or {}).get("finetune_spec", {})
    return dict(raw) if isinstance(raw, dict) else {}


def _adapter_parameter_layout(compiled: "CompiledTorchGraph") -> tuple[set[str], dict[str, dict[str, Any]]]:
    """Return exact adapter/head parameter names plus canonical LoRA site metadata."""
    names: set[str] = set()
    sites: dict[str, dict[str, Any]] = {}
    for module_name, module in compiled.named_modules():
        class_name = type(module).__name__
        if class_name in {"LoRALinearStage", "NF4LinearStage"}:
            prefix = f"{module_name}." if module_name else ""
            names.add(prefix + "lora_A")
            names.add(prefix + "lora_B")
            if getattr(module, "bias", None) is not None:
                names.add(prefix + "bias")
            sites[module_name] = {
                "kind": "nf4_lora" if class_name == "NF4LinearStage" else "lora",
                "input_dim": int(module.input_dim),
                "output_dim": int(module.output_dim),
                "rank": int(module.rank),
                "alpha": float(module.alpha),
                "scaling": float(module.scaling),
                "dropout": float(getattr(getattr(module, "lora_dropout", None), "p", 0.0)),
                "base_dtype": str(
                    getattr(module, "compute_dtype", "dense")
                    if class_name == "NF4LinearStage"
                    else module.base.weight.dtype
                ),
                "group_size": int(getattr(module, "group_size", 0)),
            }
        elif class_name in {
            "RandMapAdapterStage",
            "RewardHeadStage",
            "MaskedRewardHeadStage",
            "ValueHeadStage",
            "PolicyLogitsValueStage",
        }:
            prefix = f"{module_name}." if module_name else ""
            for local_name, _parameter in module.named_parameters(recurse=True):
                if class_name == "PolicyLogitsValueStage" and not local_name.startswith("value_head."):
                    continue
                names.add(prefix + local_name)
    return names, sites


def _normalized_sha256(value: str, *, label: str) -> str:
    digest = str(value).strip().lower()
    if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)):
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    return digest


def _resolve_adapter_provenance(
    graph: NeuronGraph,
    *,
    base_checkpoint: str | Path | None,
    base_checkpoint_sha256: str | None,
    tokenizer_sha256: str | None,
) -> tuple[dict[str, Any], str]:
    finetune = _finetune_metadata(graph)
    base_path_raw = base_checkpoint if base_checkpoint is not None else finetune.get("base_checkpoint", "")
    base_path = Path(base_path_raw) if str(base_path_raw) else None
    declared_base_sha = _normalized_sha256(
        str(base_checkpoint_sha256 or finetune.get("base_checkpoint_sha256", "")),
        label="base checkpoint SHA-256",
    )
    if base_path is not None:
        if not base_path.is_file():
            raise FileNotFoundError(f"Adapter base checkpoint does not exist: {base_path}")
        actual_base_sha = _sha256_file(base_path)
        if declared_base_sha and actual_base_sha != declared_base_sha:
            raise ValueError("Adapter base checkpoint SHA-256 does not match FineTuneSpec")
        base_descriptor = {
            "sha256": actual_base_sha,
            "size_bytes": base_path.stat().st_size,
        }
    elif declared_base_sha:
        base_descriptor = {"sha256": declared_base_sha, "size_bytes": None}
    else:
        base_descriptor = {"sha256": "", "size_bytes": None}
    tokenizer_digest = _normalized_sha256(
        str(tokenizer_sha256 or finetune.get("tokenizer_sha256", "")),
        label="tokenizer SHA-256",
    )
    return base_descriptor, tokenizer_digest


def _validate_adapter_tensor_manifest(
    state_dict: dict[str, "torch.Tensor"],
    metadata: dict[str, Any],
) -> None:
    manifest = metadata.get("tensor_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Adapter checkpoint is missing tensor_manifest")
    if set(manifest) != set(state_dict):
        missing = sorted(set(manifest) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(manifest))
        raise ValueError(f"Adapter tensor manifest mismatch: missing={missing}, unexpected={unexpected}")
    for name in sorted(state_dict):
        tensor = state_dict[name]
        entry = manifest.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"Adapter tensor manifest entry {name!r} is invalid")
        if list(tensor.shape) != list(entry.get("shape", [])):
            raise ValueError(f"Adapter tensor {name!r} has the wrong shape")
        if str(tensor.dtype) != str(entry.get("dtype", "")):
            raise ValueError(f"Adapter tensor {name!r} has the wrong dtype")
        if _tensor_sha256(tensor) != str(entry.get("sha256", "")):
            raise ValueError(f"Adapter tensor {name!r} failed SHA-256 validation")


def save_adapter_checkpoint(
    graph: NeuronGraph,
    path: str | Path,
    *,
    base_checkpoint: str | Path | None = None,
    base_checkpoint_sha256: str | None = None,
    tokenizer_sha256: str | None = None,
    optimizer_state: dict[str, Any] | None = None,
    require_provenance: bool = True,
) -> None:
    """Write a strict ``neuralfn.adapter.v1`` LoRA/QLoRA/head artifact.

    The artifact binds the exact graph topology, base artifact, tokenizer,
    projection sites, tensor shapes/dtypes, and a SHA-256 for every saved
    tensor. Empty adapter artifacts are rejected.
    """
    torch, CompiledTorchGraph = _load_torch_inference_stack()
    compiled = CompiledTorchGraph(graph)
    adapter_names, sites = _adapter_parameter_layout(compiled)
    full_state = compiled.state_dict()
    missing_layout = sorted(adapter_names - set(full_state))
    if missing_layout:
        raise RuntimeError(f"Compiled adapter layout is missing parameters: {missing_layout}")
    adapter_state = {name: full_state[name].detach().cpu() for name in sorted(adapter_names)}
    if not adapter_state:
        raise ValueError("Refusing to save an empty adapter checkpoint")
    base_descriptor, tokenizer_digest = _resolve_adapter_provenance(
        graph,
        base_checkpoint=base_checkpoint,
        base_checkpoint_sha256=base_checkpoint_sha256,
        tokenizer_sha256=tokenizer_sha256,
    )
    if require_provenance and (not base_descriptor["sha256"] or not tokenizer_digest):
        raise ValueError(
            "Strict adapter export requires a base checkpoint SHA-256 and tokenizer SHA-256; "
            "provide a base_checkpoint/FineTuneSpec provenance binding"
        )
    tensor_manifest = {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": _tensor_sha256(tensor),
        }
        for name, tensor in adapter_state.items()
    }
    metadata = {
        **_checkpoint_metadata_for_graph(graph),
        "format": ADAPTER_CHECKPOINT_FORMAT,
        "adapter_only": True,
        "provenance_complete": bool(base_descriptor["sha256"] and tokenizer_digest),
        "graph_topology_sha256": _graph_topology_sha256(graph),
        "base_artifact": base_descriptor,
        "tokenizer_sha256": tokenizer_digest,
        "sites": sites,
        "tensor_manifest": tensor_manifest,
    }
    torch.save(
        {
            "state_dict": adapter_state,
            "checkpoint_metadata": metadata,
            "optimizer_state": optimizer_state,
        },
        path,
    )


def load_adapter_checkpoint(
    graph: NeuronGraph,
    path: str | Path,
    *,
    base_checkpoint: str | Path | None = None,
    tokenizer_sha256: str | None = None,
    optimizer: Any | None = None,
    strict: bool = True,
) -> None:
    """Strictly validate and load an adapter-only artifact into ``graph``."""
    torch, CompiledTorchGraph = _load_torch_inference_stack()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise TypeError("Adapter checkpoint must contain a state_dict")
    state_dict = dict(payload["state_dict"])
    metadata = dict(payload.get("checkpoint_metadata", {}) or {})
    if metadata.get("format") != ADAPTER_CHECKPOINT_FORMAT:
        if strict:
            raise ValueError(f"Unsupported adapter checkpoint format {metadata.get('format')!r}")
        compiled = CompiledTorchGraph(graph)
        if not state_dict:
            raise ValueError("Refusing to load an empty legacy adapter checkpoint")
        compiled.load_state_dict(state_dict, strict=False)
        compiled.sync_state_back(graph)
        return
    if strict and not bool(metadata.get("provenance_complete", False)):
        raise ValueError("Adapter checkpoint does not contain complete base/tokenizer provenance")
    _validate_adapter_tensor_manifest(state_dict, metadata)
    compiled = CompiledTorchGraph(graph)
    expected_names, _sites = _adapter_parameter_layout(compiled)
    if set(state_dict) != expected_names:
        missing = sorted(expected_names - set(state_dict))
        unexpected = sorted(set(state_dict) - expected_names)
        raise ValueError(f"Adapter layout does not match graph: missing={missing}, unexpected={unexpected}")
    expected_state = compiled.state_dict()
    for name, tensor in state_dict.items():
        expected = expected_state[name]
        if tuple(tensor.shape) != tuple(expected.shape) or tensor.dtype != expected.dtype:
            raise ValueError(f"Adapter tensor {name!r} is incompatible with the destination graph")
    if metadata.get("graph_topology_sha256") != _graph_topology_sha256(graph):
        raise ValueError("Adapter graph topology SHA-256 does not match the destination graph")
    base_descriptor, tokenizer_digest = _resolve_adapter_provenance(
        graph,
        base_checkpoint=base_checkpoint,
        base_checkpoint_sha256=None,
        tokenizer_sha256=tokenizer_sha256,
    )
    expected_base = dict(metadata.get("base_artifact", {}) or {}).get("sha256", "")
    if expected_base and base_descriptor.get("sha256") != expected_base:
        raise ValueError("Adapter base artifact SHA-256 does not match the destination base")
    expected_tokenizer = str(metadata.get("tokenizer_sha256", ""))
    if expected_tokenizer and tokenizer_digest != expected_tokenizer:
        raise ValueError("Adapter tokenizer SHA-256 does not match the destination tokenizer")
    incompatible = compiled.load_state_dict(state_dict, strict=False)
    unexpected_nonadapter = sorted(set(incompatible.unexpected_keys))
    if unexpected_nonadapter:
        raise ValueError(f"Unexpected adapter tensors: {unexpected_nonadapter}")
    compiled.sync_state_back(graph)
    if optimizer is not None:
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state is None:
            raise ValueError("Adapter checkpoint does not contain optimizer state")
        optimizer.load_state_dict(optimizer_state)


def merge_adapter_into_base(
    base_path: str | Path,
    adapter_path: str | Path,
    out_path: str | Path,
) -> None:
    """Bake LoRA ``A`` / ``B`` into the frozen base weight and write a plain ``.pt``.

    Computes ``W ← W_base + (alpha/rank) * B @ A`` per LoRA site. The result is
    a standard checkpoint with no adapter state, suitable for ordinary
    inference with no runtime LoRA dispatch.
    """
    torch, _CompiledTorchGraph = _load_torch_inference_stack()
    base_state, base_meta = load_pt_checkpoint(base_path)
    adapter_state, adapter_meta = load_pt_checkpoint(adapter_path)
    if adapter_meta.get("format") != ADAPTER_CHECKPOINT_FORMAT:
        raise ValueError("merge_adapter_into_base requires a neuralfn.adapter.v1 artifact")
    _validate_adapter_tensor_manifest(adapter_state, adapter_meta)
    expected_base_sha = str(dict(adapter_meta.get("base_artifact", {}) or {}).get("sha256", ""))
    if expected_base_sha and _sha256_file(base_path) != expected_base_sha:
        raise ValueError("Adapter was created for a different base checkpoint")

    # Group adapter keys by their module prefix so we can find pairs of A/B
    # plus optionally alpha/rank metadata.
    merged = dict(base_state)
    prefixes: set[str] = set()
    for key in adapter_state:
        if ".lora_A" in key:
            prefixes.add(key.rsplit(".lora_A", 1)[0])
        elif ".lora_B" in key:
            prefixes.add(key.rsplit(".lora_B", 1)[0])

    sites = dict(adapter_meta.get("sites", {}) or {})
    consumed: set[str] = set()
    if not prefixes:
        raise ValueError("Adapter artifact has no LoRA projection pairs to merge")
    for prefix in sorted(prefixes):
        a_key = f"{prefix}.lora_A"
        b_key = f"{prefix}.lora_B"
        base_key = f"{prefix}.base.weight"
        if a_key not in adapter_state or b_key not in adapter_state:
            raise ValueError(f"Adapter site {prefix!r} is missing its A/B tensor pair")
        if base_key not in base_state:
            # The pretrained checkpoint may encode the base as ``.proj.weight``
            # (plain LinearStage). Try that remap.
            alt = f"{prefix}.proj.weight"
            if alt in base_state:
                base_weight = base_state[alt]
                out_key = alt
            else:
                raise ValueError(f"Base checkpoint has no weight for adapter site {prefix!r}")
        else:
            base_weight = base_state[base_key]
            out_key = base_key
        A = adapter_state[a_key].float()
        B = adapter_state[b_key].float()
        rank = A.shape[0]
        site = dict(sites.get(prefix, {}) or {})
        if int(site.get("rank", -1)) != rank:
            raise ValueError(f"Adapter site {prefix!r} rank metadata is invalid")
        scaling = float(site.get("scaling", float("nan")))
        if not (scaling == scaling):
            raise ValueError(f"Adapter site {prefix!r} is missing scaling metadata")
        if tuple(B.shape) != (base_weight.shape[0], rank) or A.shape[1] != base_weight.shape[1]:
            raise ValueError(f"Adapter site {prefix!r} shapes do not match the base weight")
        merged[out_key] = (base_weight.float() + scaling * (B @ A)).to(base_weight.dtype)
        consumed.update({a_key, b_key})

    # Preserve separately trained scalar/value/reward heads, but never leak
    # LoRA bookkeeping tensors into the merged dense checkpoint.
    for name in sorted(set(adapter_state) - consumed):
        if ".lora_" in name:
            raise ValueError(f"Unpaired LoRA tensor {name!r} remains after merge")
        merged[name] = adapter_state[name]

    torch.save(
        {
            "state_dict": merged,
            "checkpoint_metadata": {
                **base_meta,
                "merged_from_adapter": True,
                "adapter_format": ADAPTER_CHECKPOINT_FORMAT,
                "adapter_graph_topology_sha256": adapter_meta.get("graph_topology_sha256", ""),
                "adapter_tokenizer_sha256": adapter_meta.get("tokenizer_sha256", ""),
            },
        },
        out_path,
    )


# ---------------------------------------------------------------------------
# Quantized export / import
# ---------------------------------------------------------------------------

def _quantize_int8(tensor: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    """Per-channel int8 quantization: returns (quantized_int8, scale_fp32)."""
    torch, _CompiledTorchGraph = _load_torch_inference_stack()
    amax = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-7)
    scale = amax / 127.0
    quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
    return quantized, scale.squeeze(-1)


def _dequantize_int8(quantized: "torch.Tensor", scale: "torch.Tensor") -> "torch.Tensor":
    return quantized.float() * scale.unsqueeze(-1)


def export_quantized_pt(
    graph: NeuronGraph,
    path: str | Path,
    scheme: str = "int8",
) -> None:
    """Export weights with quantization applied.

    Schemes:
      - ``"int8"``: per-channel int8 with scale factors for every ``nn.Linear`` weight.
      - ``"ternary"``: bake ternary {-1, 0, 1} weights for BitLinearTernary models.
    """
    torch, CompiledTorchGraph = _load_torch_inference_stack()
    compiled = CompiledTorchGraph(graph)
    state_dict = compiled.state_dict()
    quant_sd: dict[str, "torch.Tensor"] = {}
    scales: dict[str, "torch.Tensor"] = {}

    for key, param in state_dict.items():
        is_embedding_weight = any(part in key for part in ("token_embedding", "position_embedding", "pos_embedding"))
        if (
            param.ndim < 2
            or is_embedding_weight
            or not any(
                key.endswith(s)
                for s in (".weight", ".proj.weight", ".k_proj.weight", ".v_proj.weight", ".q_proj.weight", ".out_proj.weight")
            )
        ):
            quant_sd[key] = param
            continue
        if scheme == "ternary":
            scale = param.abs().mean()
            w_quant = torch.round(param / (scale + 1e-7)).clamp(-1, 1).to(torch.int8)
            quant_sd[key] = w_quant
            scales[key] = scale
        else:
            q, s = _quantize_int8(param)
            quant_sd[key] = q
            scales[key] = s

    torch.save({"state_dict": quant_sd, "quant_metadata": {"scheme": scheme, "scales": scales}}, path)


def import_quantized_pt(graph: NeuronGraph, path: str | Path) -> None:
    """Import quantized weights, dequantizing them back to float for execution."""
    torch, CompiledTorchGraph = _load_torch_inference_stack()
    checkpoint = torch.load(path, weights_only=False)
    quant_sd = checkpoint["state_dict"]
    meta = checkpoint["quant_metadata"]
    scheme = meta["scheme"]
    scales = meta["scales"]

    restored: dict[str, "torch.Tensor"] = {}
    for key, param in quant_sd.items():
        if key in scales:
            if scheme == "ternary":
                restored[key] = param.float() * scales[key]
            else:
                restored[key] = _dequantize_int8(param, scales[key])
        else:
            restored[key] = param

    compiled = CompiledTorchGraph(graph)
    compiled.load_state_dict(restored)
    compiled.sync_state_back(graph)


# ---------------------------------------------------------------------------
# Inference cache for autoregressive generation with KV cache
# ---------------------------------------------------------------------------

class InferenceCache:
    """Stateful KV cache manager for autoregressive generation.

    Wraps a ``CompiledTorchGraph`` whose attention subgraphs may include
    ``kv_cache_read`` / ``kv_cache_write`` nodes, feeding cached K/V tensors
    back across steps.

    Works with both training graphs (tokens + targets -> loss) and
    inference-only graphs (tokens -> logits).  When a training graph is
    detected the cache automatically supplies dummy targets so the forward
    pass runs, and the loss output is returned as-is (useful for
    perplexity evaluation).
    """

    def __init__(
        self,
        graph: NeuronGraph,
        device: str | None = None,
        *,
        compiled: "CompiledTorchGraph | None" = None,
    ) -> None:
        torch, CompiledTorchGraph = _load_torch_inference_stack()
        self.compiled = compiled if compiled is not None else CompiledTorchGraph(graph)
        self.compiled.eval()
        resolved = device or str(graph.torch_config.get("device", "cuda"))
        self.device = torch.device(resolved)
        self.compiled.to(self.device)
        self._cache: dict[str, "torch.Tensor"] = {}

        self._n_inputs = len(graph.interface_input_layout())
        self._vocab_size: int = 0
        ts = dict(graph.torch_config.get("template_spec", {}))
        self._vocab_size = int(ts.get("vocab_size", 256))

    def reset(self) -> None:
        self._cache.clear()

    def step(self, token_ids: "torch.Tensor") -> "torch.Tensor":
        """Run one autoregressive step, returning the first output tensor.

        ``token_ids`` shape ``(batch, seq)`` -- on the first call this is the
        full prompt; on subsequent calls it should be a single token
        ``(batch, 1)``.

        For training graphs (tokens + targets -> loss) dummy targets are
        generated automatically.
        """
        torch, _CompiledTorchGraph = _load_torch_inference_stack()
        with torch.no_grad():
            token_ids = token_ids.to(self.device)
            if self._n_inputs >= 2:
                dummy_targets = torch.zeros_like(token_ids)
                outputs = self.compiled(token_ids, dummy_targets)
            else:
                outputs = self.compiled(token_ids)
        logits = outputs[0]
        return logits[:, -1, :] if logits.ndim == 3 else logits


class SemanticInferenceCache(InferenceCache):
    """Experimental: inference cache for the JEPA semantic hybrid preset.

    Extends ``InferenceCache`` to also expose the 9-D semantic vector
    produced by the encoder for inspection / conditioned generation.
    """

    def __init__(
        self,
        graph: NeuronGraph,
        device: str | None = None,
        *,
        compiled: "CompiledTorchGraph | None" = None,
    ) -> None:
        super().__init__(graph, device, compiled=compiled)
        self._last_semantic_vec: "torch.Tensor" | None = None

    @property
    def last_semantic_vec(self) -> "torch.Tensor" | None:
        return self._last_semantic_vec

    def step(self, token_ids: "torch.Tensor") -> "torch.Tensor":
        torch, _CompiledTorchGraph = _load_torch_inference_stack()
        with torch.no_grad():
            logits = super().step(token_ids)
            return logits


def export_semantic_tables(graph: NeuronGraph, path: str | Path) -> None:
    """Experimental: export semantic routing and legacy decoder lookup tables."""
    torch, CompiledTorchGraph = _load_torch_inference_stack()
    compiled = CompiledTorchGraph(graph)
    state = compiled.state_dict()
    semantic_keys = {k: v for k, v in state.items() if "decoder" in k or "hasher" in k or "sem_router" in k}
    torch.save({"semantic_tables": semantic_keys}, path)


def import_semantic_tables(graph: NeuronGraph, path: str | Path) -> None:
    """Experimental: import semantic routing and legacy decoder lookup tables."""
    torch, CompiledTorchGraph = _load_torch_inference_stack()
    checkpoint = torch.load(path, weights_only=True)
    compiled = CompiledTorchGraph(graph)
    tables = checkpoint.get("semantic_tables", {})
    current = compiled.state_dict()
    current.update(tables)
    compiled.load_state_dict(current)
    compiled.sync_state_back(graph)
