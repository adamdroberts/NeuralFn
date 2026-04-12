# Templates and Presets

NeuralFn ships a template system that generates complete, trainable transformer graphs from a preset name and a config dict. This is the fastest way to get a working model.

## Spec hierarchy

Three dataclasses define a model:

- **`ModelSpec`** -- top-level model dimensions: `model_dim`, `num_layers`, `vocab_size`, `tie_embeddings`, `logit_softcap`, plus the block and template specs.
- **`BlockSpec`** -- per-block architecture: norm type, MLP type, positional encoding, attention backend, head counts, MoE settings, compression, adapter.
- **`TemplateSpec`** -- high-level switches: `objective` (ar, diffusion, jepa, seq2seq), `backbone`, `tokenization`, `sparsity` (dense, moe), `compression`, `adapter`, `runtime` (eager, compile, megakernel), and `backend_capabilities`.

---

## Building a model from a preset

The simplest entry point:

```python
from neuralfn import build_gpt_root_graph

graph = build_gpt_root_graph(name="my_llama", preset="llama", config={
    "n_layer": 6,
    "n_embd": 256,
    "num_heads": 8,
    "num_kv_heads": 4,
    "vocab_size": 32000,
})
```

This returns a fully wired `NeuronGraph` with `runtime="torch"` and `training_method="torch"`, ready for `TorchTrainer`.

### Other builder functions

| Function | Returns | Use case |
|----------|---------|----------|
| `build_gpt_root_graph(name, preset, config)` | `NeuronGraph` | Complete graph with I/O nodes. |
| `build_model_stage_graph(spec)` | `NeuronGraph` | Just the model-stage subgraph (no dataset_source or loss). |
| `build_gpt_template_payload(name, preset, config)` | `dict` | JSON payload with `graph`, `variant_library`, and `template_spec` for the server/editor API. |
| `build_model_spec_from_config(preset, config)` | `ModelSpec` | Dispatches to the right `build_*_spec()` function. |

---

## Preset table

| Preset | Builder | Backbone | Objective | Sparsity | Key features |
|--------|---------|----------|-----------|----------|--------------|
| `nanogpt` | `build_nanogpt_spec` | nanogpt | ar | dense | LayerNorm, GELU MLP, absolute position embeddings |
| `gpt2` | `build_gpt2_spec` | gpt2 | ar | dense | LayerNorm, GELU MLP, absolute position embeddings, bias |
| `llama` | `build_llama_spec` | llama | ar | dense | RMSNorm, SwiGLU, RoPE, GQA |
| `moe` / `mixllama` | `build_mixllama_spec` | mixllama | ar | moe | RMSNorm, MoE MLP, RoPE, GQA |
| `llama_fast` | `build_llama_fast_spec` | llama | ar | dense | Like llama + `torch.compile` |
| `mixllama_fast` | `build_mixllama_fast_spec` | mixllama | ar | moe | Like moe + `torch.compile` |
| `jamba` | `build_jamba_hybrid_spec` | jamba | ar | moe | Hybrid attention + Mamba SSM, MoE, compile |
| `ternary_b158` | `build_ternary_b158_spec` | llama | ar | dense | BitLinear ternary weights |
| `seq2seq` | `build_decoder2encoder_moe_spec` | llama | seq2seq | moe | Encoder-decoder, MoE |
| `diffusion` | `build_diffllama_spec` | llama | diffusion | dense | Discrete diffusion objective |
| `ttt_llama` | `build_ttt_llama_spec` | ttt | ar | dense | Test-Time Training layers |
| `llm_jepa` | `build_llm_jepa_spec` | llama | jepa | dense | JEPA with EMA target encoder |
| `hnet_lm` | `build_hnet_lm_spec` | hnet | ar | dense | Raw-byte vocab (256), byte patches |
| `universal_llama` | `build_universal_llama_spec` | universal | ar | dense | ACT-based universal transformer |
| `llama_megakernel` | `build_llama_megakernel_spec` | llama | ar | dense | Fused attention, max-autotune compile |
| `kv_pca_llama` | `build_kv_pca_llama_spec` | llama | ar | dense | PCA-compressed KV cache |

### [Experimental] Presets

| Preset [Experimental] | Builder [Experimental] | Backbone | Objective [Experimental] | Sparsity | Key features [Experimental] |
|-----------------------|------------------------|----------|----------------------------|----------|----------------------------|
| `semantic_router_moe` | `build_semantic_router_moe_spec` | mixllama | `semantic_router` | moe | AR-only control experiment: vocab-grounded semantic projection + LSH + fixed 8-expert topic routing shared across all MoE blocks. |
| `jepa_semantic_hybrid` | `build_jepa_semantic_hybrid_spec` | llama | `jepa_semantic` | moe | JEPA + 9D vocab-grounded semantic state + LSH + fixed 8-expert topic routing + full-sequence attention experts (research prototype). |

**Disclaimer [Experimental]:** The semantic routing presets are experimental; graph layout, config keys, and training APIs may change. Both `semantic_router_moe` and `jepa_semantic_hybrid` use the root/data contract text `tokens` + text `targets` plus a separate `semantic_data_source` that provides vocab-topic `sem_targets`. Both presets require exactly 8 experts, one for each vocabulary dimension. `semantic_router_moe` is the router-only control; `jepa_semantic_hybrid` adds the JEPA path on top.

---

## Common config keys

These keys can be passed in the `config` dict to any preset builder. Aliases are shown where applicable.

| Key | Alias | Default | Description |
|-----|-------|---------|-------------|
| `n_layer` | `num_layers` | `4` | Number of transformer blocks. |
| `n_head` | `num_heads` | `4` | Number of attention heads. |
| `n_embd` | `model_dim` | `128` | Hidden dimension. |
| `vocab_size` | -- | `256` | Vocabulary size. |
| `num_kv_heads` | -- | `2` | Number of KV heads for GQA. `None` for full MHA. |
| `mlp_multiplier` | -- | `4.0` (GPT), `8/3` (Llama) | FFN hidden-dim multiplier. |
| `multiple_of` | -- | `256` | Round FFN hidden dim to this multiple (Llama-family). |
| `experts` | -- | `8` | Number of MoE experts (MoE presets). |
| `top_k` | -- | `2` | Top-K expert routing (MoE presets). |
| `rope_base` / `rope_theta` | -- | `10000.0` | RoPE base for attention-enabled presets, including hybrid routed experts. |
| `qk_gain_init` | -- | `1.0` | Initial query scaling for attention-enabled presets. |
| `dropout_p` | -- | `0.0`-`0.1` | Dropout probability. |
| `tie_embeddings` | -- | varies | Tie input embedding and output projection weights. |
| `logit_softcap` | -- | `0.0` | Tanh softcap on logits (0 = disabled). |
| `ar_loss_coef` | -- | `1.0` | Scalar for routed AR loss on semantic routing presets. |
| `jepa_loss_coef` | -- | `0.25` | Scalar for the hybrid preset's JEPA latent loss. |
| `semantic_align_loss_coef` | -- | `0.5` | Scalar for semantic-alignment loss on semantic routing presets. |
| `ttt_hidden_dim` | -- | `32` | Hidden dim for TTT layers. |
| `byte_patch_size` | -- | `4` | Byte patch window for H-Net. |
| `max_recurrence_steps` | -- | `4` | Max ACT recurrence steps (universal transformer). |

---

## Dispatching to the right builder

`build_model_spec_from_config()` maps a preset name to its builder function:

```python
from neuralfn.config import build_model_spec_from_config

spec = build_model_spec_from_config("llama", {
    "n_layer": 8,
    "n_embd": 512,
    "num_heads": 8,
    "num_kv_heads": 4,
})
```

This returns a `ModelSpec` that can be passed to `build_model_stage_graph(spec)` for lower-level graph construction.

---

## Example: building a custom Llama variant

```python
from neuralfn import build_gpt_root_graph
from neuralfn.torch_backend import CompiledTorchGraph
import torch

graph = build_gpt_root_graph(
    name="llama_small",
    preset="llama",
    config={
        "n_layer": 4,
        "n_embd": 128,
        "num_heads": 4,
        "num_kv_heads": 2,
        "vocab_size": 256,
        "mlp_multiplier": 8.0 / 3.0,
        "multiple_of": 64,
        "tie_embeddings": False,
    },
)

compiled = CompiledTorchGraph(graph)
n_params = sum(p.numel() for p in compiled.parameters())
print(f"Parameters: {n_params:,}")

tokens = torch.randint(0, 256, (1, 32))
targets = torch.randint(0, 256, (1, 32))
loss = compiled(tokens, targets)
print(f"Forward pass loss: {loss[0].item():.4f}")
```

---

Next: [Training Workflows](training-workflows.md)
