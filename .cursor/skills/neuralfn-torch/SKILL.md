---
name: neuralfn-torch
description: >-
  Build, train, and export torch-backed neural network models (GPT, Llama,
  MoE, Jamba, JEPA, diffusion, etc.) using the NeuralFn Python API. Use
  whenever the user asks to build a language model, train a transformer,
  use template presets, configure ModelSpec/BlockSpec, compile a torch graph,
  export weights, or do autoregressive inference with NeuralFn in Python code.
  For MCP tool operations, use neuralfn-mcp instead.
---

# NeuralFn Torch Models

Use this skill when building, training, or exporting torch-backed models with the NeuralFn Python API. For core graph operations, see `neuralfn-python-sdk`. For MCP tools, see `neuralfn-mcp`.

For detailed torch backend and preset reference, see [presets-reference.md](presets-reference.md).

Full API documentation lives in the repo at `docs/` ([index](../../../docs/README.md)). For a single-file LLM-ready dump of all docs, see [llms-full.txt](../../../llms-full.txt).

## End-to-end example: build, train, export

```python
from neuralfn import build_gpt_root_graph, TorchTrainer, TorchTrainConfig
from neuralfn.inference import export_to_pt, InferenceCache
import torch

# 1. Build model graph
graph = build_gpt_root_graph(
    name="my_llama",
    preset="llama",
    config={"n_layer": 4, "n_head": 4, "n_embd": 128, "num_kv_heads": 2}
)

# 2. Train
trainer = TorchTrainer(graph, TorchTrainConfig(
    epochs=10, learning_rate=5e-3, batch_size=4, device="cuda"
))
losses = trainer.train(
    train_inputs=[[1,2,3,4],[2,3,4,5],[3,4,5,6]],
    train_targets=[[2,3,4,5],[3,4,5,6],[4,5,6,7]],
)

# 3. Export
export_to_pt(graph, "my_llama.pt")

# 4. Inference
cache = InferenceCache(graph, device="cuda")
prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
logits = cache.step(prompt)
next_token = logits.argmax(dim=-1)
```

## Building graphs with presets

```python
from neuralfn import build_gpt_root_graph

graph = build_gpt_root_graph(name="model", preset="nanogpt", config={
    "n_layer": 4,      # transformer layers
    "n_head": 4,       # attention heads
    "n_embd": 128,     # model dimension
    "vocab_size": 256,  # auto-adjusted by trainer
})
```

The graph has `runtime="torch"`, `training_method="torch"`, and a populated `variant_library` with attention and MLP subgraph variants.

## All 16 presets

| Preset | Architecture | Key features |
|--------|-------------|--------------|
| `nanogpt` | GPT-2 style | LayerNorm, GELU MLP, absolute position embeddings |
| `gpt2` | GPT-2 | LayerNorm, GELU MLP, absolute pos, linear bias |
| `llama` | LLaMA | RMSNorm, SwiGLU, RoPE, GQA |
| `moe` / `mixllama` | LLaMA + MoE | RMSNorm, MoE MLP, RoPE, GQA |
| `llama_fast` | LLaMA + compile | Like llama with `torch.compile` |
| `mixllama_fast` | MoE + compile | Like moe with `torch.compile` |
| `jamba` | Jamba hybrid | Attention + Mamba interleaved, MoE |
| `ternary_b158` | BitNet b1.58 | Ternary {-1, 0, 1} weights |
| `seq2seq` | Encoder-decoder | Seq2Seq objective, MoE MLP |
| `diffusion` | Discrete diffusion | Diffusion objective with denoising head |
| `ttt_llama` | TTT-Linear | Test-time training attention replacement |
| `llm_jepa` | LLM-JEPA | JEPA with EMA target encoder |
| `hnet_lm` | H-Net | Raw byte input, byte patch embedding |
| `universal_llama` | Universal TX | ACT-based adaptive recurrence |
| `llama_megakernel` | Fused LLaMA | FusedCausalAttention, max-autotune compile |
| `kv_pca_llama` | PCA KV cache | PCA-compressed keys/values |

## Common config keys

| Key | Default | Description |
|-----|---------|-------------|
| `n_layer` / `num_layers` | 4 | Transformer layers |
| `n_head` / `num_heads` | 4 | Attention heads |
| `n_embd` / `model_dim` | 128 | Model dimension |
| `vocab_size` | 256 | Vocabulary (auto-adjusted by trainer) |
| `num_kv_heads` | 2 | GQA key/value heads |
| `mlp_multiplier` | 8/3 (llama) or 4 (gpt2) | MLP hidden multiplier |
| `multiple_of` | 256 | Round MLP width to multiple |
| `experts` | 8 | MoE: number of experts |
| `top_k` | 2 | MoE: experts per token |
| `dropout_p` | 0.0 or 0.1 | Dropout rate |
| `tie_embeddings` | varies | Tie embedding/LM head weights |
| `logit_softcap` | 0.0 | Tanh softcap (>0 enables) |
| `ttt_hidden_dim` | 32 | TTT hidden dimension |
| `byte_patch_size` | 4 | H-Net byte patch size |
| `max_recurrence_steps` | 4 | Universal TX max steps |

## Programmatic spec building

```python
from neuralfn.config import build_llama_spec, ModelSpec
from neuralfn.torch_templates import build_model_stage_graph, build_gpt_template_payload

# Build a ModelSpec directly
spec = build_llama_spec(n_layer=6, n_embd=256, num_heads=8, num_kv_heads=4)

# Build just the model stage subgraph
stage_graph = build_model_stage_graph(spec)

# Build a full payload (graph + variant library + template_spec)
payload = build_gpt_template_payload("my_model", "llama", {"n_layer": 6, "n_embd": 256})
```

Spec builders: `build_nanogpt_spec`, `build_gpt2_spec`, `build_llama_spec`, `build_mixllama_spec`, `build_llama_fast_spec`, `build_mixllama_fast_spec`, `build_jamba_hybrid_spec`, `build_ternary_b158_spec`, `build_decoder2encoder_moe_spec`, `build_diffllama_spec`, `build_ttt_llama_spec`, `build_llm_jepa_spec`, `build_hnet_lm_spec`, `build_universal_llama_spec`, `build_llama_megakernel_spec`, `build_kv_pca_llama_spec`.

## TorchTrainConfig

| Field | Default | Description |
|-------|---------|-------------|
| `learning_rate` | 3e-4 | Adam learning rate |
| `epochs` | 10 | Training epochs |
| `batch_size` | 32 | Batch size |
| `weight_decay` | 0.1 | AdamW weight decay |
| `device` | "cuda" | Device ("cuda", "cpu") |
| `amp_dtype` | None | AMP dtype (e.g. torch.float16) |
| `compile` | False | Use torch.compile |
| `activation_checkpointing` | False | Gradient checkpointing |
| `fsdp2_enabled` | False | FSDP2 sharding |
| `max_steps` | None | Step cap (None = epoch-based) |

## Training with datasets

```python
# With inline data
losses = trainer.train(
    train_inputs=[[1,2,3,4],[2,3,4,5]],
    train_targets=[[2,3,4,5],[3,4,5,6]],
)

# With HuggingFace dataset name (must be downloaded first via server API)
losses = trainer.train(dataset_names=["HuggingFaceFW__fineweb"], seq_len=64)
```

Dataset roles by objective:
- AR / H-Net / Universal: `tokens`, `targets`
- Seq2Seq: `enc_tokens`, `dec_tokens`, `targets`
- Diffusion / JEPA: `tokens`

## CompiledTorchGraph

```python
from neuralfn.torch_backend import CompiledTorchGraph

compiled = CompiledTorchGraph(graph)  # compiles NeuronGraph to nn.Module
compiled.to("cuda")

# Forward pass
outputs = compiled(token_ids, targets)

# Trace (returns dict of node_id -> tensor stats)
trace = compiled.trace(token_ids, targets)

# Sync weights back to graph JSON
compiled.sync_state_back(graph)
```

## Weight export/import

```python
from neuralfn.inference import export_to_pt, import_from_pt, export_quantized_pt, import_quantized_pt

export_to_pt(graph, "model.pt")
import_from_pt(graph, "model.pt")

export_quantized_pt(graph, "model_q.pt", scheme="int8")   # or "ternary"
import_quantized_pt(graph, "model_q.pt")
```

## InferenceCache (autoregressive generation)

```python
from neuralfn.inference import InferenceCache
import torch

cache = InferenceCache(graph, device="cuda")
prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
logits = cache.step(prompt)         # full prompt
next_tok = logits.argmax(dim=-1)
logits2 = cache.step(next_tok.unsqueeze(1))  # single token step
cache.reset()                       # clear for new sequence
```

Works with graphs that have `kv_cache_read` / `kv_cache_write` nodes. For training graphs (2 inputs), dummy targets are supplied automatically.
