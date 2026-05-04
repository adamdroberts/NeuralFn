# Templates and Presets

NeuralFn ships a template system that generates complete, trainable transformer graphs from a preset name and a config dict. This is the fastest way to get a working model.

## Spec hierarchy

Three dataclasses define a model:

- **`ModelSpec`** -- top-level model dimensions: `model_dim`, `num_layers`, `vocab_size`, `tie_embeddings`, `logit_softcap`, plus the block and template specs.
- **`BlockSpec`** -- per-block architecture: norm type, MLP type, positional encoding, attention backend, head counts, MoE settings, compression, adapter.
- **`TemplateSpec`** -- high-level switches: `objective` (ar, diffusion, jepa, seq2seq, sft, dpo, ppo, reward_model), `backbone`, `tokenization`, `sparsity` (dense, moe), `router_mode`, `compression`, `adapter`, `runtime` (eager, compile, megakernel), and `backend_capabilities`.

---

## Building a model from a preset

The simplest entry point:

```python
from neuralfn import build_gpt_root_graph
from neuralfn.config import build_llama_spec

spec = build_llama_spec(
    n_layer=6,
    n_embd=256,
    num_heads=8,
    num_kv_heads=4,
    vocab_size=32000,
)
graph = build_gpt_root_graph(name="my_llama", model_spec=spec)
```

This returns a fully wired `NeuronGraph` with `runtime="torch"` and `training_method="torch"`, ready for `TorchTrainer`.

### Other builder functions

| Function | Returns | Use case |
|----------|---------|----------|
| `build_gpt_root_graph(name, model_spec)` | `NeuronGraph` | Complete graph with I/O nodes from an existing `ModelSpec`. |
| `build_model_stage_graph(name, model_spec)` | `NeuronGraph` | Just the named model-stage subgraph (no dataset_source or loss). |
| `build_gpt_template_payload(name, config)` | `dict` | JSON payload with `variant_library`, graph settings, model node, extra nodes, and extra edges for the server/editor API. |
| `build_model_spec_from_config(config)` | `ModelSpec` | Dispatches to the right `build_*_spec()` function based on `config["preset"]`. |

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
| `semantic_router_moe` | `build_semantic_router_moe_spec` | mixllama | `semantic_router` | moe | AR-only control experiment: vocab-grounded semantic projection + LSH + one expert per semantic vocabulary dimension shared across all MoE blocks. |
| `jepa_semantic_hybrid` | `build_jepa_semantic_hybrid_spec` | llama | `jepa_semantic` | moe | JEPA + vocab-grounded semantic state + LSH + fixed dimension-to-expert topic routing + full-sequence attention experts (research prototype). |
| `semantic_moe_jepa_evo` | `build_semantic_moe_jepa_evo_spec` | mixllama | `semantic_moe_jepa_evo` | moe | Full chunk-routed Semantic MoE JEPA Evo: 2 shared experts, semantic-vocab experts, 8 free experts, JEPA target supervision, and periodic route evolution. |

**Disclaimer [Experimental]:** The semantic routing presets are experimental; graph layout, config keys, and training APIs may change. `semantic_router_moe`, `jepa_semantic_hybrid`, and `semantic_moe_jepa_evo` use the root/data contract text `tokens` + text `targets` plus a separate `semantic_data_source` that provides vocab-topic `sem_targets`. The router-only and hybrid presets require one expert per semantic vocabulary dimension. `semantic_moe_jepa_evo` adds shared and free experts around the semantic expert bank, so `experts` must equal `semantic_shared_experts + NUM_VOCAB_DIMS + semantic_free_experts`.

---

## Common config keys

These keys can be passed in config-dict flows such as `build_model_spec_from_config()` and the server/editor template APIs. Direct `build_*_spec()` calls use the canonical Python keyword names; aliases are shown where applicable.

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
| `jepa_loss_coef` | -- | `0.25` | Scalar for JEPA latent loss on JEPA semantic presets. |
| `semantic_align_loss_coef` | -- | `0.5` | Scalar for semantic-alignment loss on semantic routing presets. |
| `semantic_vocab_ref` | -- | default vocab | Semantic vocabulary file used by semantic projector/router stages. |
| `route_chunk_size` | -- | `32` | Chunk size for `semantic_moe_jepa_evo` route updates. |
| `semantic_shared_experts` | -- | `2` | Always-on shared experts for `semantic_moe_jepa_evo`. |
| `semantic_free_experts` | -- | `8` | Free learned experts for `semantic_moe_jepa_evo`. |
| `route_evo_enabled` | -- | `true` | Enable periodic route-evolution search for `semantic_moe_jepa_evo`. |
| `route_evo_fraction` | -- | `0.10` | Approximate fraction of optimizer steps that run route evolution. |
| `route_evo_population` | -- | `8` | Candidate count for route evolution. |
| `route_evo_mutation_scale` | -- | `0.05` | Mutation scale for route-evolution candidates. |
| `ttt_hidden_dim` | -- | `32` | Hidden dim for TTT layers. |
| `byte_patch_size` | -- | `4` | Byte patch window for H-Net. |
| `max_recurrence_steps` | -- | `4` | Max ACT recurrence steps (universal transformer). |
| `adapter_type` | -- | `"none"` | Adapter implementation: `"none"`, `"lora"`, `"qlora"`, or `"randmap"`. |
| `lora_rank` / `lora_alpha` | -- | `8` / `16.0` | LoRA/qLoRA rank and scaling. |
| `lora_targets` | -- | `("q_proj", "v_proj")` | Projection roles wrapped by LoRA/qLoRA. |
| `qlora_group_size` | -- | `64` | NF4 group size for qLoRA base projections. |

---

## Composed recipes and fine-tuning roots

The `nfn` CLI and lower-level Python callers can use `build_composed_lm_spec()`
to build a `ModelSpec` from base-model/topology/router choices instead of a
single named preset:

```python
from neuralfn.config import FineTuneSpec, build_composed_lm_spec
from neuralfn.torch_templates import build_gpt_root_graph

spec = build_composed_lm_spec(
    base_model="llama",
    topology="moe",
    router_mode="semantic",
    use_jepa=True,
    adapter_type="lora",
    lora_rank=8,
    finetune=FineTuneSpec(objective="sft", base_checkpoint="base.pt"),
)
spec.template.objective = "sft"
graph = build_gpt_root_graph(name="sft_model", model_spec=spec)
```

Fine-tuning objectives dispatch to dedicated roots:

| Objective | Root graph | Dataset source | Loss path |
|-----------|------------|----------------|-----------|
| `sft` | `build_sft_root_graph` | `sft_dataset_source` | masked token CE |
| `dpo` | `build_dpo_root_graph` | `dpo_dataset_source` | policy/reference logp -> DPO loss |
| `ppo` | `build_ppo_root_graph` | `ppo_rollout_source` | clipped policy/value loss plus KL/reward shaping |
| `reward_model` | `build_reward_model_root_graph` | `dpo_dataset_source` | reward heads -> preference BCE |

---

## Dispatching to the right builder

`build_model_spec_from_config()` maps `config["preset"]` to its builder function:

```python
from neuralfn.torch_templates import build_model_spec_from_config

spec = build_model_spec_from_config({
    "preset": "llama",
    "n_layer": 8,
    "n_embd": 512,
    "num_heads": 8,
    "num_kv_heads": 4,
})
```

This returns a `ModelSpec` that can be passed to `build_model_stage_graph("model_stage", spec)` for lower-level graph construction.

---

## Example: building a custom Llama variant

```python
from neuralfn import build_gpt_root_graph
from neuralfn.torch_backend import CompiledTorchGraph
from neuralfn.config import build_llama_spec
import torch

spec = build_llama_spec(
    n_layer=4,
    n_embd=128,
    num_heads=4,
    num_kv_heads=2,
    vocab_size=256,
    mlp_multiplier=8.0 / 3.0,
    multiple_of=64,
    tie_embeddings=False,
)
graph = build_gpt_root_graph(name="llama_small", model_spec=spec)

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
