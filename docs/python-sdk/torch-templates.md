# neuralfn.torch_templates

Graph builders for attention subgraphs, MLP blocks, decoder blocks, backbone graphs, and full model architectures. These functions construct `NeuronGraph` instances wired with the appropriate builtin neurons and variant libraries.

---

## Utility Functions

### `make_terminal_def(*, role, port_name, dtype, neuron_id=None) -> NeuronDef`

Create a terminal (input/output) neuron definition with a passthrough function. Used to define graph interface nodes.

### `clone_neuron_def(ndef, *, config=None) -> NeuronDef`

Deep-clone a `NeuronDef` via round-trip serialization. Optionally merge extra keys into `module_config`.

### `get_linear_module_def(input_dim, output_dim, spec) -> NeuronDef`

Return the appropriate linear module def based on the `BlockSpec` compression,
adapter, and family settings. Dispatches to `bitlinear_ternary`, `ttt_linear`,
`lora_linear`, `nf4_linear`, or standard `linear`.

### `maybe_wrap_with_adapter(graph, node_id, model_dim, spec, *, position) -> str`

If `spec.adapter_type == "randmap"` and `spec.adapter_dim > 0`, insert a
`randmap_adapter` node after `node_id` and rewire downstream edges. LoRA and
qLoRA are handled by `get_linear_module_def()` at projection creation time.
Returns the new output node ID (or the original if no adapter was added).

### `link_variant_neuron(graph, *, family, version, name, input_aliases=None, output_aliases=None) -> NeuronDef`

Create a `subgraph_neuron` with a `variant_ref` pointing into the variant library.

---

## Attention Graphs

### `build_dense_attention_graph(name, model_dim, spec, *, is_cross=False, enable_cache=False, enable_pca=False, pca_compressed_dim=None, fused_megakernel=False) -> NeuronGraph`

Build a complete attention subgraph. The graph includes Q/K/V projections, optional RoPE, optional GQA repeat-KV, SDPA, merge-heads, and output projection. Supports:

- Cross-attention (`is_cross=True`)
- KV cache nodes (`enable_cache=True`)
- PCA compression/decompression (`enable_pca=True`)
- Fused megakernel mode (`fused_megakernel=True`): single `fused_causal_attention` node

---

## MLP Graphs

### `build_dense_mlp_graph(name, model_dim, spec) -> NeuronGraph`

Build a dense MLP subgraph. Dispatches to GELU, SwiGLU, or ReLU-squared based on `spec.mlp_type`.

### `build_mixllama_mlp_graph(name, model_dim, spec) -> NeuronGraph`

Build a Mixture-of-Experts MLP subgraph with router, top-K routing, expert dispatch, combine, and load-balance loss.

### `build_mamba_graph(name, model_dim, spec) -> NeuronGraph`

Build a Mamba SSM subgraph (single Mamba module node).

---

## Block Graph

### `build_decoder_block_graph(name, model_dim, spec, attn_graph, mlp_graph) -> NeuronGraph`

Build a single decoder block: pre-norm, attention (or Mamba) subgraph, residual add, pre-norm, MLP subgraph, residual add. Uses the provided `attn_graph` and `mlp_graph` as nested subgraph neurons.

---

## Backbone Graphs

### `build_hidden_backbone_graph(name, model_spec) -> NeuronGraph`

Build the hidden-state backbone: stacked decoder blocks with a final normalization layer. Used as a building block for multi-component architectures (seq2seq, JEPA, etc.).

### `build_jepa_encoder_graph(name, model_spec) -> NeuronGraph`

Build a JEPA encoder: token embedding, position embedding, stacked blocks, final norm.

### `build_jepa_semantic_encoder_graph(name, model_spec) -> NeuronGraph` [Experimental]

Build the experimental hybrid JEPA encoder: hidden backbone + masked latent pool + semantic projector. Outputs `(semantic_vec, hidden, topic_logits)` so the routed expert branch can attend over the full hidden sequence while JEPA losses operate on the pooled semantic state and vocab-topic head.

---

## Model Stage Graphs

### `build_seq2seq_model_stage_graph(name, model_spec) -> NeuronGraph`

Build a full seq2seq model: encoder backbone + decoder backbone with cross-attention, LM head, and cross-entropy loss.

### `build_diffusion_model_stage_graph(name, model_spec) -> NeuronGraph`

Build a discrete diffusion model: random timesteps, mask scheduler, backbone, denoise head, and cross-entropy loss.

### `build_jepa_model_stage_graph(name, model_spec) -> NeuronGraph`

Build a JEPA model: online encoder, target encoder (EMA), JEPA mask, projector, predictor, latent pool, and latent MSE loss.

### `build_jepa_semantic_model_stage_graph(name, model_spec) -> NeuronGraph` [Experimental]

Build the experimental JEPA semantic hybrid stage. Inputs are `tokens`, `targets`, and `sem_targets`. The stage performs JEPA masking and EMA target encoding, hashes the pooled semantic vector, routes the full masked hidden sequence through `semantic_hash_router` + `routed_attention_experts`, and uses the semantic vocabulary dimension map for both training-time teacher forcing and inference-time auto routing. `sem_targets` carry categorical topic IDs with `-100` ignore sentinels.

### `build_semantic_moe_jepa_evo_model_stage_graph(name, model_spec) -> NeuronGraph` [Experimental]

Build the full Semantic MoE JEPA Evo stage. Inputs are `tokens`, `targets`, and `sem_targets`. The stage keeps dense causal attention on the AR path, builds prefix-safe chunk semantic states, selects shared/semantic/free experts for each chunk, broadcasts chunk routes to token routes for the MoE FFN, and trains AR CE plus JEPA latent, semantic alignment, route balance, route selection, and route distillation losses.

### `build_sft_model_stage_graph(name, model_spec) -> NeuronGraph`

Build a supervised fine-tuning stage with `(tokens, targets, loss_mask) -> loss`.
It mirrors the autoregressive decoder body but uses
`masked_token_cross_entropy` so prompt tokens can be excluded from loss.

### `build_sft_root_graph(*, name="model_root", model_spec=None) -> NeuronGraph`

Build the root graph for SFT: `sft_dataset_source -> model -> loss`.

### `build_dpo_root_graph(*, name="model_root", model_spec=None) -> NeuronGraph`

Build the Direct Preference Optimization root graph. It consumes chosen/rejected
token pairs from `dpo_dataset_source`, runs policy and frozen-reference forwards,
reduces sequence log probabilities, and feeds `dpo_pairwise_loss`.

### `build_reward_model_root_graph(*, name="model_root", model_spec=None) -> NeuronGraph`

Build the reward-model root graph. It consumes chosen/rejected preference pairs,
runs a body subgraph, applies reward heads, and optimizes
`preference_bce_loss`.

### `build_ppo_root_graph(*, name="model_root", model_spec=None) -> NeuronGraph`

Build the PPO inner-loop root graph. It consumes rollout-buffer tensors,
combines policy, reference, reward/value heads, and emits a clipped PPO loss.
`PPOTrainer` in `neuralfn.torch_backend` orchestrates rollout collection and
inner optimization around this graph.

### `build_hnet_model_stage_graph(name, model_spec) -> NeuronGraph`

Build a hierarchical byte-level model: byte patch embed, backbone, byte patch merge, and cross-entropy loss.

### `build_universal_model_stage_graph(name, model_spec) -> NeuronGraph`

Build a universal transformer model: embedding, universal transformer with ACT, LM head, and cross-entropy loss.

---

## High-Level Builders

### `build_model_spec_from_config(config, *, preview_defaults=False) -> ModelSpec`

```python
def build_model_spec_from_config(
    config: dict[str, Any],
    *,
    preview_defaults: bool = False,
) -> ModelSpec
```

Dispatch to the appropriate `build_*_spec()` function based on `config["preset"]`. Normalizes legacy key names (`n_layer` -> `num_layers`, `n_embd` -> `model_dim`, `n_head` -> `num_heads`).

Recognized presets: `"nanogpt"`, `"nanogpt_megakernel"`, `"gpt2"`,
`"gpt2_megakernel"`, `"llama"`, `"mixllama"` / `"moe"`, `"llama_fast"`,
`"llama_fast_megakernel"`, `"mixllama_fast"`,
`"mixllama_fast_megakernel"`, `"jamba"`, `"ternary_b158"`,
`"llama_megakernel"`, `"kv_pca_llama"`, `"seq2seq"`, `"diffusion"`,
`"ttt_llama"`, `"llm_jepa"`, `"semantic_router_moe"`,
`"semantic_router_moe_megakernel"`, `"jepa_semantic_hybrid"`,
`"jepa_semantic_hybrid_megakernel"`, `"semantic_moe_jepa_evo"`,
`"hnet_lm"`, and `"universal_llama"`.

### `build_model_stage_graph(name, model_spec) -> NeuronGraph`

```python
def build_model_stage_graph(name: str, model_spec: ModelSpec) -> NeuronGraph
```

Build the standard autoregressive model stage graph: token embedding, optional position embedding, N decoder blocks (via variant library), final norm, LM head (tied or standalone), optional logit softcap, and cross-entropy loss. Inputs: `tokens`, `targets`. Output: `loss`.

### `build_gpt_root_graph(*, name="model_root", model_spec=None) -> NeuronGraph`

```python
def build_gpt_root_graph(
    *,
    name: str = "model_root",
    model_spec: ModelSpec | None = None,
) -> NeuronGraph
```

Build the top-level root graph that wraps a model stage subgraph. Dispatches to
the appropriate stage builder based on `model_spec.template.objective` and
`model_spec.template.backbone`. If `model_spec` is None, uses default
`ModelSpec()`. Objectives `sft`, `dpo`, `ppo`, and `reward_model` dispatch to
their dedicated fine-tuning root graph builders.

The root graph's `torch_config` is populated with device, AMP dtype, and the full `template_spec`.

For `semantic_router_moe`, `jepa_semantic_hybrid`, and `semantic_moe_jepa_evo`, the root graph exposes:

- `dataset_source` with output roles `tokens`, `targets`
- `semantic_data_source` with output role `sem_targets` generated from the active semantic vocabulary reference
- a compiled flat input contract of `(tokens, targets, sem_targets)`

`semantic_router_moe` builds an AR-only stage that reuses normal LLaMA attention, computes one shared semantic route from the pre-block hidden state, broadcasts that route across the full sequence, and feeds the same routed experts into every MoE block. `jepa_semantic_hybrid` keeps its separate JEPA path on top of that semantic routing stack. `semantic_moe_jepa_evo` is the full chunk-routed architecture: it updates semantic routes at chunk boundaries, prepends always-on shared experts, selects semantic/free experts for the next chunk, and wires route balance/selection/distillation losses into the total loss.

### `build_gpt_template_payload(name, config) -> dict`

```python
def build_gpt_template_payload(name: str, config: dict[str, Any]) -> dict[str, Any]
```

Build a complete template payload for the editor frontend. Returns a dict with:
- `variant_library`: serialized variant library graphs
- `graph_settings`: training_method, runtime, torch_config
- `node_def`: the serialized model subgraph neuron definition
