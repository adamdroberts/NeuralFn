from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .graph import NeuronGraph
from .neuron import decode_module_state_dict, encode_module_state_dict


def default_gpt_config() -> dict[str, Any]:
    return {
        "vocab_size": 256,
        "num_layers": 4,
        "model_dim": 128,
        "num_heads": 4,
        "num_kv_heads": 2,
        "mlp_mult": 2,
        "tie_embeddings": True,
        "logit_softcap": 30.0,
        "rope_base": 10000.0,
        "qk_gain_init": 1.0,
    }


def default_token_embedding_config() -> dict[str, Any]:
    cfg = default_gpt_config()
    return {
        "vocab_size": cfg["vocab_size"],
        "model_dim": cfg["model_dim"],
    }


def default_rms_norm_config() -> dict[str, Any]:
    return {"eps": 1e-6}


def default_attention_config() -> dict[str, Any]:
    cfg = default_gpt_config()
    return {
        "model_dim": cfg["model_dim"],
        "num_heads": cfg["num_heads"],
        "num_kv_heads": cfg["num_kv_heads"],
        "rope_base": cfg["rope_base"],
        "qk_gain_init": cfg["qk_gain_init"],
    }


def default_residual_mix_config() -> dict[str, Any]:
    return {"dim": default_gpt_config()["model_dim"], "primary_init": 1.0, "skip_init": 0.0}


def default_residual_add_config() -> dict[str, Any]:
    return {"dim": default_gpt_config()["model_dim"], "init_scale": 1.0}


def default_mlp_config() -> dict[str, Any]:
    cfg = default_gpt_config()
    return {"model_dim": cfg["model_dim"], "mlp_mult": cfg["mlp_mult"]}


def default_lm_head_config() -> dict[str, Any]:
    cfg = default_gpt_config()
    return {"model_dim": cfg["model_dim"], "vocab_size": cfg["vocab_size"]}


def default_logit_softcap_config() -> dict[str, Any]:
    return {"softcap": default_gpt_config()["logit_softcap"]}


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        cos = freqs.cos()[None, None, :, :].to(dtype=dtype)
        sin = freqs.sin()[None, None, :, :].to(dtype=dtype)
        return cos, sin


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class TokenEmbeddingStage(nn.Module):
    def __init__(self, vocab_size: int, model_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)

    def forward(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        return self.embedding(token_ids), self.embedding.weight


class RMSNormStage(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class ResidualMixStage(nn.Module):
    def __init__(self, dim: int, primary_init: float = 1.0, skip_init: float = 0.0) -> None:
        super().__init__()
        self.primary_scale = nn.Parameter(torch.full((dim,), primary_init, dtype=torch.float32))
        self.skip_scale = nn.Parameter(torch.full((dim,), skip_init, dtype=torch.float32))

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        return (
            self.primary_scale[None, None, :].to(dtype=x.dtype) * x
            + self.skip_scale[None, None, :].to(dtype=x.dtype) * x0
        )


class ResidualAddStage(nn.Module):
    def __init__(self, dim: int, init_scale: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), init_scale, dtype=torch.float32))

    def forward(self, residual: Tensor, delta: Tensor) -> Tensor:
        return residual + self.scale[None, None, :].to(dtype=residual.dtype) * delta


class CausalSelfAttentionStage(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        head_dim = model_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        kv_dim = num_kv_heads * head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(head_dim, rope_base)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, model_dim = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seq_len, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        return self.out_proj(y.transpose(1, 2).contiguous().reshape(batch, seq_len, model_dim))


class MLPReluSquaredStage(nn.Module):
    def __init__(self, model_dim: int, mlp_mult: int) -> None:
        super().__init__()
        hidden = model_dim * mlp_mult
        self.fc = nn.Linear(model_dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, model_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class TiedLMHeadStage(nn.Module):
    def forward(self, hidden: Tensor, tied_weight: Tensor) -> Tensor:
        return F.linear(hidden, tied_weight)


class LMHeadStage(nn.Module):
    def __init__(self, model_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(model_dim, vocab_size, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.proj(hidden)


class LogitSoftcapStage(nn.Module):
    def __init__(self, softcap: float) -> None:
        super().__init__()
        if softcap <= 0.0:
            raise ValueError("softcap must be positive")
        self.softcap = float(softcap)

    def forward(self, logits: Tensor) -> Tensor:
        return self.softcap * torch.tanh(logits / self.softcap)


class TokenCrossEntropyStage(nn.Module):
    def forward(self, logits: Tensor, target_ids: Tensor) -> Tensor:
        flat_logits = logits.reshape(-1, logits.size(-1))
        return F.cross_entropy(flat_logits.float(), target_ids.reshape(-1), reduction="mean")


def build_module(module_type: str, module_config: dict[str, Any]) -> nn.Module:
    cfg = dict(module_config or {})
    if module_type == "token_embedding":
        return TokenEmbeddingStage(
            vocab_size=int(cfg["vocab_size"]),
            model_dim=int(cfg["model_dim"]),
        )
    if module_type == "rms_norm":
        return RMSNormStage(eps=float(cfg.get("eps", 1e-6)))
    if module_type == "residual_mix":
        return ResidualMixStage(
            dim=int(cfg["dim"]),
            primary_init=float(cfg.get("primary_init", 1.0)),
            skip_init=float(cfg.get("skip_init", 0.0)),
        )
    if module_type == "residual_add":
        return ResidualAddStage(
            dim=int(cfg["dim"]),
            init_scale=float(cfg.get("init_scale", 1.0)),
        )
    if module_type == "causal_self_attention":
        return CausalSelfAttentionStage(
            model_dim=int(cfg["model_dim"]),
            num_heads=int(cfg["num_heads"]),
            num_kv_heads=int(cfg["num_kv_heads"]),
            rope_base=float(cfg["rope_base"]),
            qk_gain_init=float(cfg["qk_gain_init"]),
        )
    if module_type == "mlp_relu2":
        return MLPReluSquaredStage(
            model_dim=int(cfg["model_dim"]),
            mlp_mult=int(cfg["mlp_mult"]),
        )
    if module_type == "tied_lm_head":
        return TiedLMHeadStage()
    if module_type == "lm_head":
        return LMHeadStage(
            model_dim=int(cfg["model_dim"]),
            vocab_size=int(cfg["vocab_size"]),
        )
    if module_type == "logit_softcap":
        return LogitSoftcapStage(softcap=float(cfg["softcap"]))
    if module_type == "token_cross_entropy":
        return TokenCrossEntropyStage()
    raise KeyError(f"Unsupported module type: {module_type}")


def _wrap_output(value: Any) -> tuple[Tensor, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _apply_tensor_function(name: str, args: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    if name in {"input", "output", "identity"}:
        return (args[0],)
    if name == "add":
        return (args[0] + args[1],)
    if name == "multiply":
        return (args[0] * args[1],)
    if name == "negate":
        return (-args[0],)
    if name == "relu":
        return (torch.relu(args[0]),)
    if name == "sigmoid":
        return (torch.sigmoid(args[0]),)
    if name == "tanh_neuron":
        return (torch.tanh(args[0]),)
    if name == "leaky_relu":
        return (F.leaky_relu(args[0], negative_slope=0.01),)
    if name == "gelu":
        return (F.gelu(args[0]),)
    if name == "silu":
        return (F.silu(args[0]),)
    if name == "softplus":
        return (F.softplus(args[0]),)
    if name == "hard_tanh":
        return (F.hardtanh(args[0]),)
    raise TypeError(f"Function node '{name}' is not supported by the torch runtime")


class CompiledTorchGraph(nn.Module):
    def __init__(self, graph: NeuronGraph) -> None:
        super().__init__()
        self.graph = graph
        if graph.has_cycles():
            raise ValueError(f"Torch runtime does not support cyclic graphs: '{graph.name}'")
        self.order = graph.topological_order() if graph.nodes else []
        self.node_modules = nn.ModuleDict()
        for nid, node in graph.nodes.items():
            ndef = node.neuron_def
            if ndef.kind == "module":
                module = build_module(ndef.module_type, ndef.module_config)
                if ndef.module_state:
                    module.load_state_dict(decode_module_state_dict(ndef.module_state))
                self.node_modules[nid] = module
            elif ndef.kind == "subgraph" and ndef.subgraph is not None:
                self.node_modules[nid] = CompiledTorchGraph(ndef.subgraph)

    def forward(self, *flat_inputs: Tensor) -> tuple[Tensor, ...]:
        outputs, _trace = self._run(flat_inputs, want_trace=False)
        return outputs

    def trace(self, *flat_inputs: Tensor) -> tuple[tuple[Tensor, ...], dict[str, tuple[Tensor, ...]]]:
        return self._run(flat_inputs, want_trace=True)

    def _run(
        self,
        flat_inputs: tuple[Tensor, ...],
        *,
        want_trace: bool,
    ) -> tuple[tuple[Tensor, ...], dict[str, tuple[Tensor, ...]]]:
        provided = self._expand_flat_inputs(flat_inputs)
        values: dict[str, tuple[Tensor, ...]] = {}
        traces: dict[str, tuple[Tensor, ...]] = {}
        for nid, tensor_tuple in provided.items():
            values[nid] = tensor_tuple
            traces[nid] = tensor_tuple

        for nid in self.order:
            if nid in values and nid in provided:
                continue
            node = self.graph.nodes[nid]
            args = self._gather_inputs(nid, values)
            if node.neuron_def.kind == "subgraph" and want_trace:
                child_outputs, child_trace = self.node_modules[nid].trace(*args)
                values[nid] = _wrap_output(child_outputs)
                traces[nid] = values[nid]
                for child_key, child_value in child_trace.items():
                    traces[f"{nid}/{child_key}"] = child_value
            else:
                values[nid] = self._execute_node(nid, node.neuron_def, args)
                traces[nid] = values[nid]

        outputs = self._flatten_outputs(values)
        return outputs, traces if want_trace else {}

    def _expand_flat_inputs(self, flat_inputs: tuple[Tensor, ...]) -> dict[str, tuple[Tensor, ...]]:
        expanded: dict[str, tuple[Tensor, ...]] = {}
        idx = 0
        for nid in self.graph.input_node_ids:
            node = self.graph.nodes[nid]
            n_out = node.neuron_def.n_outputs
            chunk = flat_inputs[idx : idx + n_out]
            if len(chunk) != n_out:
                raise ValueError(
                    f"Graph '{self.graph.name}' expected {len(self.graph.interface_input_layout())} "
                    f"flattened inputs but received {len(flat_inputs)}"
                )
            expanded[nid] = tuple(chunk)
            idx += n_out
        if idx != len(flat_inputs):
            raise ValueError(
                f"Graph '{self.graph.name}' expected {idx} flattened inputs but received {len(flat_inputs)}"
            )
        return expanded

    def _flatten_outputs(self, values: dict[str, tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
        flattened: list[Tensor] = []
        for nid in self.graph.output_node_ids:
            flattened.extend(values.get(nid, ()))
        return tuple(flattened)

    def _gather_inputs(self, node_id: str, values: dict[str, tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
        node = self.graph.nodes[node_id]
        gathered: list[Tensor | None] = [None] * node.neuron_def.n_inputs
        for edge in self.graph._incoming(node_id):
            src_vals = values.get(edge.src_node)
            if src_vals is None:
                raise ValueError(f"Node '{node_id}' is missing source values from '{edge.src_node}'")
            if edge.src_port >= len(src_vals):
                raise ValueError(f"Edge '{edge.id}' src_port={edge.src_port} exceeds source outputs")
            if gathered[edge.dst_port] is not None:
                raise ValueError(
                    f"Torch graph '{self.graph.name}' has multiple incoming edges into "
                    f"node '{node_id}' port {edge.dst_port}; use an explicit combine node instead"
                )
            gathered[edge.dst_port] = src_vals[edge.src_port]

        missing = [idx for idx, value in enumerate(gathered) if value is None]
        if missing:
            raise ValueError(f"Node '{node_id}' is missing required torch inputs at ports {missing}")
        return tuple(value for value in gathered if value is not None)

    def _execute_node(
        self,
        node_id: str,
        ndef: Any,
        args: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        if ndef.kind == "module":
            return _wrap_output(self.node_modules[node_id](*args))
        if ndef.kind == "subgraph":
            return _wrap_output(self.node_modules[node_id](*args))
        return _apply_tensor_function(ndef.name, args)

    def sync_state_back(self, graph: NeuronGraph | None = None) -> None:
        target_graph = graph or self.graph
        for nid, node in target_graph.nodes.items():
            if node.neuron_def.kind == "module":
                node.neuron_def.module_state = encode_module_state_dict(
                    self.node_modules[nid].to("cpu").state_dict()
                )
            elif node.neuron_def.kind == "subgraph" and node.neuron_def.subgraph is not None:
                compiled_child = self.node_modules[nid]
                if isinstance(compiled_child, CompiledTorchGraph):
                    compiled_child.sync_state_back(node.neuron_def.subgraph)


@dataclass
class TorchTrainConfig:
    learning_rate: float = 3e-4
    epochs: int = 50
    batch_size: int = 8
    weight_decay: float = 0.01
    device: str = "cuda"
    amp_dtype: str = "bfloat16"


class TorchTrainer:
    def __init__(self, graph: NeuronGraph, config: TorchTrainConfig | None = None) -> None:
        self.graph = graph
        self.config = config or TorchTrainConfig()
        self._stop = False
        self.loss_history: list[float] = []

    def stop(self) -> None:
        self._stop = True

    def train(
        self,
        train_inputs: list[list[int]] | Tensor,
        train_targets: list[list[int]] | Tensor,
        *,
        on_epoch: Callable[[int, float], None] | None = None,
    ) -> list[float]:
        compiled = CompiledTorchGraph(self.graph)

        device_name = str(self.graph.torch_config.get("device", self.config.device or "cuda")).lower()
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Torch training is configured for CUDA, but no CUDA device is available")
        device = torch.device(device_name)
        amp_name = str(self.graph.torch_config.get("amp_dtype", self.config.amp_dtype)).lower()
        amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
        use_amp = device.type == "cuda"

        compiled.to(device)
        optimizer = torch.optim.AdamW(
            compiled.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        x = torch.as_tensor(train_inputs, dtype=torch.long)
        y = torch.as_tensor(train_targets, dtype=torch.long)
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("Torch GPT training expects integer token arrays of shape [batch, seq_len]")
        if x.shape != y.shape:
            raise ValueError("train_inputs and train_targets must have the same [batch, seq_len] shape")

        input_ports = self.graph.flattened_input_ports()
        if len(input_ports) == 1:
            dataset = torch.utils.data.TensorDataset(x)
        elif len(input_ports) == 2:
            dataset = torch.utils.data.TensorDataset(x, y)
        else:
            raise ValueError(
                f"Torch trainer currently supports 1 or 2 external graph inputs, got {len(input_ports)}"
            )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        self._stop = False
        self.loss_history = []
        compiled.train()

        for epoch in range(self.config.epochs):
            if self._stop:
                break
            total_loss = 0.0
            total_rows = 0
            for batch in loader:
                if isinstance(batch, Tensor):
                    batch = (batch,)
                if len(batch) == 1:
                    flat_inputs = (batch[0].to(device),)
                else:
                    flat_inputs = tuple(item.to(device) for item in batch)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = compiled(*flat_inputs)
                    if len(outputs) != 1:
                        raise ValueError(
                            f"Torch training graph '{self.graph.name}' must expose exactly one scalar loss output"
                        )
                    loss = outputs[0]
                if loss.ndim != 0:
                    raise ValueError("Torch training output must be a scalar loss tensor")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_rows = int(flat_inputs[0].size(0))
                total_loss += float(loss.item()) * batch_rows
                total_rows += batch_rows

            avg_loss = total_loss / max(total_rows, 1)
            self.loss_history.append(avg_loss)
            if on_epoch is not None:
                on_epoch(epoch, avg_loss)

        compiled.sync_state_back(self.graph)
        self.graph.training_method = "torch"
        self.graph.runtime = "torch"
        self.graph.torch_config = {
            **self.graph.torch_config,
            "device": device.type,
            "amp_dtype": amp_name,
        }
        return self.loss_history
