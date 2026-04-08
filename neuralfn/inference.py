from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Any

from .graph import NeuronGraph
from .torch_backend import CompiledTorchGraph


def export_to_pt(graph: NeuronGraph, path: str | Path) -> None:
    """Export the weights of a compiled or uncompiled torch-based NeuronGraph to a .pt file."""
    compiled = CompiledTorchGraph(graph)
    state_dict = compiled.state_dict()
    torch.save(state_dict, path)


def import_from_pt(graph: NeuronGraph, path: str | Path) -> None:
    """Import weights from a .pt file into a NeuronGraph's module_state."""
    state_dict = torch.load(path, weights_only=True)
    compiled = CompiledTorchGraph(graph)
    compiled.load_state_dict(state_dict)
    compiled.sync_state_back(graph)


# ---------------------------------------------------------------------------
# Quantized export / import
# ---------------------------------------------------------------------------

def _quantize_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel int8 quantization: returns (quantized_int8, scale_fp32)."""
    amax = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-7)
    scale = amax / 127.0
    quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
    return quantized, scale.squeeze(-1)


def _dequantize_int8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
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
    from .torch_backend import BitLinearTernaryStage

    compiled = CompiledTorchGraph(graph)
    state_dict = compiled.state_dict()
    quant_sd: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}

    for key, param in state_dict.items():
        if param.ndim < 2 or not any(key.endswith(s) for s in (".weight", ".proj.weight", ".k_proj.weight", ".v_proj.weight", ".q_proj.weight", ".out_proj.weight")):
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
    checkpoint = torch.load(path, weights_only=False)
    quant_sd = checkpoint["state_dict"]
    meta = checkpoint["quant_metadata"]
    scheme = meta["scheme"]
    scales = meta["scales"]

    restored: dict[str, torch.Tensor] = {}
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

    def __init__(self, graph: NeuronGraph, device: str | None = None) -> None:
        self.compiled = CompiledTorchGraph(graph)
        self.compiled.eval()
        resolved = device or str(graph.torch_config.get("device", "cuda"))
        self.device = torch.device(resolved)
        self.compiled.to(self.device)
        self._cache: dict[str, torch.Tensor] = {}

        self._n_inputs = len(graph.interface_input_layout())
        self._vocab_size: int = 0
        ts = dict(graph.torch_config.get("template_spec", {}))
        self._vocab_size = int(ts.get("vocab_size", 256))

    def reset(self) -> None:
        self._cache.clear()

    @torch.no_grad()
    def step(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Run one autoregressive step, returning the first output tensor.

        ``token_ids`` shape ``(batch, seq)`` -- on the first call this is the
        full prompt; on subsequent calls it should be a single token
        ``(batch, 1)``.

        For training graphs (tokens + targets -> loss) dummy targets are
        generated automatically.
        """
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

    Extends ``InferenceCache`` to also expose the 15-D semantic vector
    produced by the encoder for inspection / conditioned generation.
    """

    def __init__(self, graph: NeuronGraph, device: str | None = None) -> None:
        super().__init__(graph, device)
        self._last_semantic_vec: torch.Tensor | None = None

    @property
    def last_semantic_vec(self) -> torch.Tensor | None:
        return self._last_semantic_vec

    @torch.no_grad()
    def step(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits = super().step(token_ids)
        return logits


def export_semantic_tables(graph: NeuronGraph, path: str | Path) -> None:
    """Experimental: export lookup tables for the attentionless decoder."""
    compiled = CompiledTorchGraph(graph)
    state = compiled.state_dict()
    semantic_keys = {k: v for k, v in state.items() if "decoder" in k or "hasher" in k or "sem_router" in k}
    torch.save({"semantic_tables": semantic_keys}, path)


def import_semantic_tables(graph: NeuronGraph, path: str | Path) -> None:
    """Experimental: import lookup tables for the attentionless decoder."""
    checkpoint = torch.load(path, weights_only=True)
    compiled = CompiledTorchGraph(graph)
    tables = checkpoint.get("semantic_tables", {})
    current = compiled.state_dict()
    current.update(tables)
    compiled.load_state_dict(current)
    compiled.sync_state_back(graph)
