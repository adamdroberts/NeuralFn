# neuralfn.builtins

Library of 58 built-in neuron definitions ready to drop into a graph.

## Class: BuiltinNeurons

```python
class BuiltinNeurons:
    sigmoid = ...       # NeuronDef
    relu = ...          # NeuronDef
    # ... all 58 builtins as class attributes
```

Each built-in neuron is exposed as a class attribute of type `NeuronDef`.

### Class Methods

#### `BuiltinNeurons.all() -> list[NeuronDef]`

Return the full built-in neuron catalog (all 58 entries).

#### `BuiltinNeurons.get(name: str) -> NeuronDef`

Lookup a built-in by attribute name or serialized display name. Raises `KeyError` if not found.

```python
relu = BuiltinNeurons.get("relu")
attn = BuiltinNeurons.get("causal_self_attention_module")
```

---

## Module Constants

### `BUILTIN_NEURONS: list[NeuronDef]`

Flat list of all built-in `NeuronDef` instances.

### `BUILTIN_MAP: dict[str, NeuronDef]`

Maps display name (`ndef.name`) to `NeuronDef`.

### `BUILTIN_ATTR_MAP: dict[str, NeuronDef]`

Maps Python attribute name to `NeuronDef`. Use this when the attribute name differs from the display name (e.g. `"tanh_neuron"` attribute vs. `"tanh_neuron"` display name).

---

## Full Catalog

### Scalar Activations

| Attribute | Display Name | Kind | Inputs | Outputs |
|-----------|-------------|------|--------|---------|
| `sigmoid` | sigmoid | function | 1 | 1 |
| `relu` | relu | function | 1 | 1 |
| `tanh_neuron` | tanh_neuron | function | 1 | 1 |
| `threshold` | threshold | function | 1 | 1 |
| `identity` | identity | function | 1 | 1 |
| `negate` | negate | function | 1 | 1 |
| `gaussian` | gaussian | function | 1 | 1 |
| `log_neuron` | log | function | 1 | 1 |
| `leaky_relu` | leaky_relu | function | 1 | 1 |
| `prelu` | prelu | function | 1 | 1 |
| `relu6` | relu6 | function | 1 | 1 |
| `elu` | elu | function | 1 | 1 |
| `selu` | selu | function | 1 | 1 |
| `gelu` | gelu | function | 1 | 1 |
| `silu` | silu | function | 1 | 1 |
| `mish` | mish | function | 1 | 1 |
| `softplus` | softplus | function | 1 | 1 |
| `softsign` | softsign | function | 1 | 1 |
| `hard_sigmoid` | hard_sigmoid | function | 1 | 1 |
| `hard_tanh` | hard_tanh | function | 1 | 1 |
| `hard_swish` | hard_swish | function | 1 | 1 |

### Scalar Binary Ops

| Attribute | Display Name | Kind | Inputs | Outputs |
|-----------|-------------|------|--------|---------|
| `add` | add | function | 2 | 1 |
| `multiply` | multiply | function | 2 | 1 |
| `softmax_2` | softmax_2 | function | 2 | 2 |
| `logsoftmax_2` | logsoftmax_2 | function | 2 | 2 |

### Graph Terminals

| Attribute | Display Name | Kind | Inputs | Outputs |
|-----------|-------------|------|--------|---------|
| `input_node` | input | function | 1 | 1 |
| `output_node` | output | function | 1 | 1 |

### Torch -- Embeddings

| Attribute | Display Name | Kind | Inputs | Outputs |
|-----------|-------------|------|--------|---------|
| `token_embedding_module` | token_embedding | module | 1 | 2 |
| `absolute_position_embedding_module` | absolute_position_embedding | module | 1 | 1 |

### Torch -- Linear / MLP

| Attribute | Display Name | Kind | Inputs | Outputs |
|-----------|-------------|------|--------|---------|
| `linear_module` | linear | module | 1 | 1 |
| `mlp_relu2_module` | mlp_relu2 | module | 1 | 1 |
| `gelu_module` | gelu | module | 1 | 1 |
| `swiglu_module` | swiglu | module | 1 | 1 |

### Torch -- Normalization

| Attribute | Display Name | Kind | Inputs | Outputs |
|-----------|-------------|------|--------|---------|
| `rms_norm_module` | rms_norm | module | 1 | 1 |
| `layer_norm_module` | layer_norm | module | 1 | 1 |
| `dropout_module` | dropout | module | 1 | 1 |

### Torch -- Attention

| Attribute | Display Name | Kind | Inputs | Outputs |
|-----------|-------------|------|--------|---------|
| `reshape_heads_module` | reshape_heads | module | 1 | 1 |
| `merge_heads_module` | merge_heads | module | 1 | 1 |
| `repeat_kv_module` | repeat_kv | module | 1 | 1 |
| `rotary_embedding_module` | rotary_embedding | module | 2 | 2 |
| `qk_gain_module` | qk_gain | module | 1 | 1 |
| `scaled_dot_product_attention_module` | scaled_dot_product_attention | module | 3 | 1 |
| `causal_self_attention_module` | causal_self_attention | module | 1 | 1 |
| `fused_causal_attention_module` | fused_causal_attention | module | 1 | 1 |
| `residual_mix_module` | residual_mix | module | 2 | 1 |
| `residual_add_module` | residual_add | module | 2 | 1 |
| `kv_cache_read_module` | kv_cache_read | module | 4 | 2 |
| `kv_cache_write_module` | kv_cache_write | module | 2 | 2 |
| `kv_pca_encode_module` | kv_pca_encode | module | 2 | 2 |
| `kv_pca_decode_module` | kv_pca_decode | module | 2 | 2 |
| `kv_quant_pack_module` | kv_quant_pack | module | 2 | 1 |
| `kv_quant_unpack_module` | kv_quant_unpack | module | 1 | 2 |

### Torch -- LM Head / Loss

| Attribute | Display Name | Kind | Inputs | Outputs |
|-----------|-------------|------|--------|---------|
| `tied_lm_head_module` | tied_lm_head | module | 2 | 1 |
| `lm_head_module` | lm_head | module | 1 | 1 |
| `logit_softcap_module` | logit_softcap | module | 1 | 1 |
| `token_cross_entropy_module` | token_cross_entropy | module | 2 | 1 |

### Torch -- MoE (Mixture of Experts)

| Attribute | Display Name | Kind | Inputs | Outputs |
|-----------|-------------|------|--------|---------|
| `router_logits_module` | router_logits | module | 1 | 1 |
| `topk_route_module` | topk_route | module | 1 | 2 |
| `expert_dispatch_module` | expert_dispatch | module | 3 | 1 |
| `expert_combine_module` | expert_combine | module | 1 | 1 |
| `load_balance_loss_module` | load_balance_loss | module | 3 | 2 |
| `aux_loss_add_module` | aux_loss_add | module | 2 | 1 |

### Torch -- Data / Special

| Attribute | Display Name | Kind | Inputs | Outputs |
|-----------|-------------|------|--------|---------|
| `dataset_source_module` | dataset_source | module | 0 | 2 |
| `bitlinear_ternary_module` | bitlinear_ternary | module | 1 | 1 |
| `randmap_adapter_module` | randmap_adapter | module | 1 | 1 |
| `mamba_module` | mamba | module | 1 | 1 |
| `denoise_head_module` | denoise_head | module | 1 | 1 |
| `mask_scheduler_module` | mask_scheduler | module | 2 | 1 |
| `random_timesteps_module` | random_timesteps | module | 1 | 1 |
| `jepa_mask_module` | jepa_mask | module | 1 | 2 |
| `latent_pool_module` | latent_pool | module | 2 | 1 |
| `jepa_projector_module` | jepa_projector | module | 1 | 1 |
| `jepa_predictor_module` | jepa_predictor | module | 1 | 1 |
| `latent_mse_loss_module` | latent_mse_loss | module | 2 | 1 |
| `byte_patch_embed_module` | byte_patch_embed | module | 1 | 1 |
| `byte_patch_merge_module` | byte_patch_merge | module | 2 | 1 |
| `act_halt_gate_module` | act_halt_gate | module | 1 | 1 |
| `act_weighted_sum_module` | act_weighted_sum | module | 2 | 1 |
| `universal_transformer_module` | universal_transformer | module | 1 | 2 |
| `ttt_linear_module` | ttt_linear | module | 1 | 1 |

---

## Experimental Builtins [Experimental]

The module neurons below are **[Experimental]** (JEPA semantic hybrid stack). Port names match `NeuronDef` I/O; shapes follow the active `module_config` on the node.

| Attribute [Experimental] | `module_type` [Experimental] | Inputs [Experimental] | Outputs [Experimental] |
|--------------------------|------------------------------|------------------------|-------------------------|
| `semantic_projector_module` | `semantic_projector` | `hidden` | `semantic_vec`, `residual` |
| `semantic_alignment_loss_module` | `semantic_alignment_loss` | `pred`, `target` | `loss` |
| `semantic_hasher_module` | `semantic_hasher` | `semantic_vec` | `bucket_indices` |
| `semantic_moe_router_module` | `semantic_moe_router` | `semantic_vec` | `expert_weights`, `expert_indices` |
| `attentionless_decoder_module` | `attentionless_decoder` | `bucket_indices`, `expert_output` | `logits` |
| `softmax_distillation_loss_module` | `softmax_distillation_loss` | `teacher_logits`, `student_logits` | `loss` |

**Disclaimer [Experimental]:** These builtins are tied to a research prototype; port semantics and `module_config` keys may change.
