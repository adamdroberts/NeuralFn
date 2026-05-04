# NeuralFn Torch Models -- Complete Reference

This is the detailed reference for agents building torch-backed models with NeuralFn. Read this when you need exact signatures, the full Stage class inventory, preset-specific config details, or builder function parameters.

---

## Template system overview

Three dataclasses configure a model:

1. **`TemplateSpec`** -- high-level architecture switches (objective, backbone, sparsity, router, compression, adapter, runtime)
2. **`BlockSpec`** -- per-transformer-block settings (norm, MLP type, position encoding, attention, heads, MoE params)
3. **`ModelSpec`** -- top-level dimensions (model_dim, num_layers, vocab_size) plus block_spec and template

The chain: preset config -> `build_model_spec_from_config(config)` or `build_*_spec(**kwargs)` -> `ModelSpec` -> `build_model_stage_graph(name, spec)` or `build_gpt_root_graph(name=name, model_spec=spec)`.

---

## TemplateSpec (dataclass)

```python
from neuralfn.config import TemplateSpec

TemplateSpec(
    objective: ObjectiveType = "ar",              # "ar" | "diffusion" | "jepa" | "seq2seq" | "sft" | "dpo" | "ppo" | "reward_model" | semantic variants
    backbone: BackboneType = "gpt2",              # "gpt2" | "nanogpt" | "llama" | "mixllama" | "jamba" | "universal" | "ttt" | "hnet"
    tokenization: TokenizationType = "sp",        # "sp" | "byte_hnet"
    sparsity: SparsityType = "dense",             # "dense" | "moe"
    compression: CompressionType = "none",        # "none" | "ternary_b158" | "binary_1bit" | "kv_pca"
    adapter: AdapterType = "none",                # "none" | "lora" | "qlora" | "randmap"
    router_mode: RouterModeType = "none",         # "none" | "standard" | "semantic"
    runtime: RuntimeType = "eager",               # "eager" | "compile" | "sdpa" | "megakernel"
    backend_capabilities: dict[str, bool] = {...}, # auto-resolved by resolve_backend_capabilities()
)
```

## BlockSpec (dataclass)

```python
from neuralfn.config import BlockSpec

BlockSpec(
    family: str,                          # "nanogpt" | "gpt2" | "llama" | "mixllama" | "jamba" | "ttt" | "hnet" | "universal"
    norm_type: str = "layernorm",         # "layernorm" | "rmsnorm"
    mlp_type: str = "gelu",              # "gelu" | "swiglu" | "moe"
    pos_encoding: str = "absolute",      # "absolute" | "rope"
    attention_backend: str = "sdpa",     # "sdpa" | "flex" | "math"
    num_heads: int = 4,
    num_kv_heads: int | None = None,     # None = MHA, smaller = GQA/MQA
    is_causal: bool = True,
    linear_bias: bool = True,
    dropout_p: float = 0.0,
    rope_theta: float = 10000.0,
    rope_scaling: dict | None = None,
    mlp_multiplier: float = 4.0,
    multiple_of: int | None = None,
    experts: int | None = None,
    top_k: int | None = None,
    shared_experts: int = 0,
    router_aux_loss_coef: float = 0.0,
    compression: str = "none",
    adapter_dim: int = 0,
    adapter_type: str = "none",
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj"),
    lora_bias: bool = False,
    qlora_group_size: int = 64,
    qlora_compute_dtype: str = "bf16",
    qk_gain_init: float = 1.0,
    ttt_hidden_dim: int = 16,
    byte_patch_size: int = 4,
    byte_patch_stride: int = 4,
)
```

## ModelSpec (dataclass)

```python
from neuralfn.config import ModelSpec

ModelSpec(
    model_dim: int = 128,
    num_layers: int = 4,
    vocab_size: int = 256,
    tie_embeddings: bool = True,
    logit_softcap: float = 0.0,          # 0.0 = disabled, >0 = tanh cap
    block_spec: BlockSpec = BlockSpec(family="gpt2"),
    template: TemplateSpec = TemplateSpec(),
    jepa_latent_dim: int = 128,
    jepa_mask_ratio: float = 0.5,
    jepa_mask_strategy: str = "random",  # "random" | "block"
    jepa_num_blocks: int = 4,
    jepa_min_block_ratio: float = 0.1,
    jepa_max_block_ratio: float = 0.25,
    ema_decay: float = 0.99,
    max_recurrence_steps: int = 4,
    halt_epsilon: float = 0.01,
    semantic_dim: int = NUM_SEMANTIC_DIMS,
    semantic_vocab_ref: str = DEFAULT_SEMANTIC_VOCAB_REF,
    route_chunk_size: int = 32,
    semantic_shared_experts: int = 2,
    semantic_free_experts: int = 8,
    route_evo_enabled: bool = True,
    route_evo_fraction: float = 0.10,
    route_evo_population: int = 8,
    route_evo_mutation_scale: float = 0.05,
    route_evo_seed: int | None = None,
    ar_loss_coef: float = 1.0,
    jepa_loss_coef: float = 0.25,
    semantic_align_loss_coef: float = 0.5,
    finetune: FineTuneSpec | None = None,
)
```

---

## Shipped presets -- detailed

### `nanogpt` -- `build_nanogpt_spec(**kwargs)`
- Backbone: nanogpt, Objective: ar, Runtime: eager
- LayerNorm, GELU MLP, absolute position embeddings
- `linear_bias` from `kwargs.get("bias", False)`, `dropout_p` from `kwargs.get("dropout_p", 0.1)`
- `tie_embeddings=True`

### `gpt2` -- `build_gpt2_spec(**kwargs)`
- Backbone: gpt2, Objective: ar, Runtime: eager
- LayerNorm, GELU MLP, absolute position embeddings
- `linear_bias=True`, `tie_embeddings=True`

### `llama` -- `build_llama_spec(**kwargs)`
- Backbone: llama, Objective: ar, Runtime: eager
- RMSNorm, SwiGLU MLP, RoPE, GQA
- `linear_bias=False`, `dropout_p=0.0`
- `mlp_multiplier` default `8/3`, `multiple_of` default `256`
- `num_kv_heads` default `2`, `tie_embeddings=False`

### `moe` / `mixllama` -- `build_mixllama_spec(**kwargs)`
- Backbone: mixllama, Objective: ar, Sparsity: moe, Runtime: eager
- RMSNorm, MoE MLP (SwiGLU per expert), RoPE, GQA
- `experts` default `8`, `top_k` default `2`, `router_aux_loss_coef` default `0.01`
- `tie_embeddings=False`

### `llama_fast` -- `build_llama_fast_spec(**kwargs)`
- Same as llama but `runtime="compile"` (uses `torch.compile`)

### `mixllama_fast` -- `build_mixllama_fast_spec(**kwargs)`
- Same as moe/mixllama but `runtime="compile"`

### `jamba` -- `build_jamba_hybrid_spec(**kwargs)`
- Backbone: jamba, Sparsity: moe, Runtime: compile
- Hybrid attention + Mamba SSM interleaved, MoE MLP
- `experts` default `8`, `top_k` default `2`

### `ternary_b158` -- `build_ternary_b158_spec(**kwargs)`
- Backbone: llama, Compression: ternary_b158, Runtime: compile
- Uses `BitLinearTernaryStage` for ternary {-1, 0, 1} weights

### `seq2seq` -- `build_decoder2encoder_moe_spec(**kwargs)`
- Objective: seq2seq, Backbone: llama, Sparsity: moe, Runtime: compile
- Encoder-decoder architecture, dataset roles: `enc_tokens`, `dec_tokens`, `targets`

### `diffusion` -- `build_diffllama_spec(**kwargs)`
- Objective: diffusion, Backbone: llama, Runtime: compile
- Discrete diffusion with denoising head and mask scheduler
- Dataset role: `tokens` only

### `ttt_llama` -- `build_ttt_llama_spec(**kwargs)`
- Backbone: ttt, Runtime: compile
- TTT-Linear layers replace standard attention
- `ttt_hidden_dim` from `kwargs.get("ttt_hidden_dim", 32)`

### `llm_jepa` -- `build_llm_jepa_spec(**kwargs)`
- Objective: jepa, Backbone: llama, Runtime: compile
- JEPA with EMA target encoder, latent prediction
- `jepa_mask_strategy`: `"random"` (default) or `"block"`
- Dataset role: `tokens` only

### `semantic_router_moe` -- `build_semantic_router_moe_spec(**kwargs)`
- Objective: semantic_router, Backbone: mixllama, Runtime: compile
- AR-only semantic router control: shared vocab-grounded route broadcast across every MoE block
- Dataset roles: `tokens`, `targets`, plus `semantic_data_source -> sem_targets`
- Requires one expert per semantic vocabulary dimension; trains next-token CE + semantic alignment

### `jepa_semantic_hybrid` -- `build_jepa_semantic_hybrid_spec(**kwargs)`
- Objective: jepa_semantic, Backbone: llama, Runtime: compile
- Experimental JEPA + vocab-grounded semantic router hybrid with one expert per semantic vocabulary dimension
- Dataset roles: `tokens`, `targets`, plus `semantic_data_source -> sem_targets`
- Trains AR next-token CE, JEPA latent alignment, and masked semantic topic loss

### `semantic_moe_jepa_evo` -- `build_semantic_moe_jepa_evo_spec(**kwargs)`
- Objective: semantic_moe_jepa_evo, Backbone: mixllama, Runtime: compile
- Full Semantic MoE JEPA Evo prototype: dense AR attention, chunk-level causal semantic planner, JEPA target supervision, route balance/selection/distillation losses, and lightweight route evolution.
- Dataset roles: `tokens`, `targets`, plus `semantic_data_source -> sem_targets`
- Expert bank defaults: `semantic_shared_experts=2`, `NUM_VOCAB_DIMS` semantic experts, `semantic_free_experts=8`; `experts` must equal their sum.
- Route evolution defaults: `route_chunk_size=32`, `route_evo_fraction=0.10`, `route_evo_population=8`, `route_evo_mutation_scale=0.05`

### `hnet_lm` -- `build_hnet_lm_spec(**kwargs)`
- Backbone: hnet, Tokenization: byte_hnet, Runtime: compile
- Raw byte vocab (`vocab_size=256` forced), byte patch embedding/merge
- `byte_patch_size` default `4`, `byte_patch_stride` default matches patch_size
- `multiple_of` default `64`

### `universal_llama` -- `build_universal_llama_spec(**kwargs)`
- Backbone: universal, Runtime: compile
- ACT-based adaptive computation time, shared transformer weights
- `max_recurrence_steps` default `4`, `halt_epsilon` default `0.01`

### `llama_megakernel` -- `build_llama_megakernel_spec(**kwargs)`
- Backbone: llama, Runtime: megakernel
- Uses `FusedCausalAttentionStage`, compiles with `torch.compile(mode="max-autotune", fullgraph=True)`

### `kv_pca_llama` -- `build_kv_pca_llama_spec(**kwargs)`
- Backbone: llama, Compression: kv_pca, Runtime: compile
- Inserts `kv_pca_encode`/`kv_pca_decode` around KV path in attention
- BlockSpec has `compression="kv_pca"`

### Megakernel/config-dispatch variants
- `nanogpt_megakernel` -- `build_nanogpt_megakernel_spec(**kwargs)`
- `gpt2_megakernel` -- `build_gpt2_megakernel_spec(**kwargs)`
- `llama_fast_megakernel` -- `build_llama_fast_megakernel_spec(**kwargs)`
- `mixllama_fast_megakernel` -- `build_mixllama_fast_megakernel_spec(**kwargs)`
- `semantic_router_moe_megakernel` -- `build_semantic_router_moe_megakernel_spec(**kwargs)` [Experimental]
- `jepa_semantic_hybrid_megakernel` -- `build_jepa_semantic_hybrid_megakernel_spec(**kwargs)` [Experimental]

---

## Config key reference

Config-dict keys can be passed to `build_model_spec_from_config(config)` and server/editor template APIs. Direct `build_*_spec()` calls accept the canonical keyword names shown below; for example, use `num_heads` for direct calls and `n_head` only in config-dict dispatch.

| Key | Aliases | Default | Applies to | Description |
|-----|---------|---------|------------|-------------|
| `n_layer` | `num_layers` | `4` | all | Number of transformer blocks |
| `n_head` | `num_heads` | `4` | all | Attention heads |
| `n_embd` | `model_dim` | `128` | all | Hidden dimension |
| `vocab_size` | -- | `256` | all (forced 256 for hnet) | Vocabulary size (auto-adjusted by TorchTrainer) |
| `num_kv_heads` | -- | `2` | llama-family | GQA key/value heads. None = full MHA. |
| `tie_embeddings` | -- | varies | all | Tie input embedding and output head weights |
| `logit_softcap` | -- | `0.0` | all | Tanh softcap on logits (0 = disabled) |
| `mlp_multiplier` | -- | `4.0` (gpt), `8/3` (llama) | dense | FFN hidden dim multiplier |
| `multiple_of` | -- | `256` (llama), `64` (hnet) | llama-family | Round FFN hidden to this multiple |
| `dropout_p` | -- | `0.0`-`0.1` | all | Dropout probability |
| `bias` | -- | `False` | nanogpt | Linear bias |
| `experts` | -- | `8` | moe presets | Number of MoE experts |
| `top_k` | -- | `2` | moe presets | Active experts per token |
| `router_aux_loss_coef` | -- | `0.01` | moe presets | Load-balance loss coefficient |
| `router_mode` | -- | `"none"` | composed recipes | Disabled, standard learned, or semantic router composition |
| `adapter_type` | -- | `"none"` | composed recipes/fine-tuning | `"none"`, `"lora"`, `"qlora"`, or `"randmap"` |
| `lora_rank` / `lora_alpha` | -- | `8` / `16.0` | lora/qlora | Adapter rank and scaling |
| `lora_targets` | -- | `("q_proj", "v_proj")` | lora/qlora | Projection roles wrapped by adapters |
| `qlora_group_size` | -- | `64` | qlora | NF4 quantization group size |
| `ttt_hidden_dim` | -- | `32` | ttt_llama | TTT hidden dimension |
| `byte_patch_size` | -- | `4` | hnet_lm | Byte patch window size |
| `byte_patch_stride` | -- | `byte_patch_size` | hnet_lm | Byte patch stride |
| `max_recurrence_steps` | -- | `4` | universal_llama | Max ACT recurrence steps |
| `halt_epsilon` | -- | `0.01` | universal_llama | ACT halt threshold |
| `jepa_latent_dim` | -- | `model_dim` | llm_jepa | JEPA latent dimension |
| `jepa_mask_ratio` | -- | `0.5` | llm_jepa | Fraction of tokens masked |
| `jepa_mask_strategy` | -- | `"random"` | llm_jepa | `"random"` or `"block"` |
| `jepa_num_blocks` | -- | `4` | llm_jepa | Number of contiguous mask spans (block mode) |
| `jepa_min_block_ratio` | -- | `0.1` | llm_jepa | Min block length as fraction of seq |
| `jepa_max_block_ratio` | -- | `0.25` | llm_jepa | Max block length as fraction of seq |
| `ema_decay` | -- | `0.99` | llm_jepa | EMA target encoder decay |
| `semantic_vocab_ref` | -- | default vocab | semantic routing | Semantic vocabulary file for topic targets and routing |
| `route_chunk_size` | -- | `32` | semantic_moe_jepa_evo | Tokens per chunk route update |
| `semantic_shared_experts` | -- | `2` | semantic_moe_jepa_evo | Always-on shared experts |
| `semantic_free_experts` | -- | `8` | semantic_moe_jepa_evo | Free learned experts after semantic experts |
| `route_evo_enabled` | -- | `True` | semantic_moe_jepa_evo | Enable periodic route-evolution search |
| `route_evo_fraction` | -- | `0.10` | semantic_moe_jepa_evo | Fraction of steps that run route evolution |
| `route_evo_population` | -- | `8` | semantic_moe_jepa_evo | Route-evolution candidate count |
| `route_evo_mutation_scale` | -- | `0.05` | semantic_moe_jepa_evo | Route-evolution mutation scale |

---

## Builder functions

### Top-level

```python
from neuralfn import build_gpt_root_graph, build_model_stage_graph
from neuralfn.config import build_llama_spec

# Complete graph with dataset_source, model stage, loss -- ready for TorchTrainer
spec = build_llama_spec(n_layer=4, n_embd=128, num_heads=4, num_kv_heads=2)
graph = build_gpt_root_graph(name="model", model_spec=spec)

# Just the model stage subgraph (no I/O wiring)
stage = build_model_stage_graph("model_stage", spec)
```

### build_gpt_template_payload(name, config) -> dict

Returns a JSON dict with `variant_library`, `graph_settings`, `node_def`, `extra_nodes`, and `extra_edges`. Used by the server/editor template API.

### build_model_spec_from_config(config, *, preview_defaults=False) -> ModelSpec

Dispatches `config["preset"]` to the correct `build_*_spec()` function. Handles preset aliases such as `"moe"` -> `build_mixllama_spec`, and routes composed recipes through `build_composed_lm_spec()`.

### Subgraph builders

These are called internally by the template system but can be used directly:

| Function | Builds |
|----------|--------|
| `build_dense_attention_graph(spec, layer)` | Dense attention subgraph (Q/K/V proj, RoPE, SDPA, out proj) |
| `build_dense_mlp_graph(spec, layer)` | Dense MLP subgraph (GELU or SwiGLU) |
| `build_mixllama_mlp_graph(spec, layer)` | MoE MLP subgraph (router, dispatch, experts, combine) |
| `build_mamba_graph(spec, layer)` | Mamba SSM subgraph |
| `build_decoder_block_graph(spec, layer)` | Full decoder block (norm, attention, norm, MLP, residuals) |
| `build_hidden_backbone_graph(spec)` | Stack of decoder blocks |
| `build_jepa_encoder_graph(spec)` | JEPA encoder with masking and EMA |
| `build_seq2seq_model_stage_graph(spec)` | Seq2seq encoder-decoder model stage |
| `build_diffusion_model_stage_graph(spec)` | Diffusion model stage with denoising |
| `build_jepa_model_stage_graph(spec)` | JEPA model stage |
| `build_hnet_model_stage_graph(spec)` | H-Net byte-level model stage |
| `build_universal_model_stage_graph(spec)` | Universal transformer with ACT |

---

## CompiledTorchGraph (nn.Module)

```python
from neuralfn.torch_backend import CompiledTorchGraph

compiled = CompiledTorchGraph(graph: NeuronGraph)
```

Compiles a `NeuronGraph` into an `nn.Module` by walking nodes in topological order and instantiating each module neuron's Stage class via `build_module()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `graph` | `NeuronGraph` | Source graph |
| `order` | `list[str]` | Topological execution order |
| `node_modules` | `nn.ModuleDict` | Compiled modules by node ID |

**Methods:**
- `forward(*inputs: Tensor) -> tuple[Tensor, ...]` -- run forward pass
- `trace(*inputs: Tensor) -> tuple[tuple[Tensor,...], dict[str, tuple[Tensor,...]]]` -- forward + per-node trace
- `sync_state_back(graph: NeuronGraph | None = None) -> None` -- write weights back to module_state

---

## TorchTrainConfig (dataclass)

```python
from neuralfn import TorchTrainConfig

TorchTrainConfig(
    learning_rate: float = 3e-4,
    epochs: int = 10,
    batch_size: int = 32,
    weight_decay: float = 0.1,
    device: str = "cuda",
    amp_dtype: str | None = None,            # "bfloat16", "float16", or None
    compile: bool = False,
    activation_checkpointing: bool = False,
    fsdp2_enabled: bool = False,
    max_steps: int | None = None,            # None = full epochs
)
```

---

## TorchTrainer

```python
from neuralfn import TorchTrainer

trainer = TorchTrainer(graph: NeuronGraph, config: TorchTrainConfig | None = None)
```

**Methods:**
- `train(train_inputs=None, train_targets=None, *, on_step=None, dataset_names=None, text_column=None, seq_len=64) -> list[float]`
  - With inline data: `trainer.train([[1,2,3],[2,3,4]], [[2,3,4],[3,4,5]])`
  - With datasets: `trainer.train(dataset_names=["HuggingFaceFW__fineweb"], seq_len=64)`
  - Auto-adjusts vocab_size if the dataset tokenizer has a different vocabulary
- `stop() -> None` -- signal early stop

**Attributes:** `graph`, `config`, `loss_history`

---

## All Stage classes

Each maps to a `module_type` string and is instantiated by `build_module()`:

| Stage | module_type | I/O | Description |
|-------|-------------|-----|-------------|
| `TokenEmbeddingStage` | `token_embedding` | 1 in, 2 out | Token lookup; outputs (hidden, weight) |
| `AbsolutePositionEmbeddingStage` | `absolute_position_embedding` | 1 in, 1 out | Learned position vectors |
| `LinearStage` | `linear` | 1 in, 1 out | Dense projection |
| `BitLinearTernaryStage` | `bitlinear_ternary` | 1 in, 1 out | Ternary weight linear |
| `RandMapAdapterStage` | `randmap_adapter` | 1 in, 1 out | Random-map adapter |
| `RMSNormStage` | `rms_norm` | 1 in, 1 out | RMS normalization |
| `LayerNormStage` | `layer_norm` | 1 in, 1 out | Layer normalization |
| `DropoutStage` | `dropout` | 1 in, 1 out | Dropout |
| `GeluStage` | `gelu` | 1 in, 1 out | GELU activation |
| `SwiGLUStage` | `swiglu` | 1 in, 1 out | SwiGLU gated MLP |
| `MLPReluSquaredStage` | `mlp_relu2` | 1 in, 1 out | MLP with ReLU-squared |
| `ReshapeHeadsStage` | `reshape_heads` | 1 in, 1 out | Reshape to multi-head |
| `MergeHeadsStage` | `merge_heads` | 1 in, 1 out | Merge multi-head back |
| `RepeatKVStage` | `repeat_kv` | 1 in, 1 out | Repeat KV heads for GQA |
| `RotaryEmbeddingStage` | `rotary_embedding` | 2 in, 2 out | RoPE on Q and K |
| `QKGainStage` | `qk_gain` | 1 in, 1 out | Per-head Q scaling |
| `ScaledDotProductAttentionStage` | `scaled_dot_product_attention` | 3 in, 1 out | Q,K,V -> attended output |
| `CausalSelfAttentionStage` | `causal_self_attention` | 1 in, 1 out | Full attention block |
| `FusedCausalAttentionStage` | `fused_causal_attention` | 1 in, 1 out | Fused attention (megakernel) |
| `ResidualMixStage` | `residual_mix` | 2 in, 1 out | Learned residual blend |
| `ResidualAddStage` | `residual_add` | 2 in, 1 out | Scaled residual add |
| `KVCacheReadStage` | `kv_cache_read` | 2 in, 2 out | Read from KV cache |
| `KVCacheWriteStage` | `kv_cache_write` | 2 in, 2 out | Write to KV cache |
| `KVPCAEncodeStage` | `kv_pca_encode` | 2 in, 2 out | PCA-compress KV |
| `KVPCADecodeStage` | `kv_pca_decode` | 2 in, 2 out | PCA-decompress KV |
| `KVQuantPackStage` | `kv_quant_pack` | 2 in, 2 out | Quantize+pack KV |
| `KVQuantUnpackStage` | `kv_quant_unpack` | 2 in, 2 out | Unpack+dequantize KV |
| `TiedLMHeadStage` | `tied_lm_head` | 2 in, 1 out | LM head with tied weights |
| `LMHeadStage` | `lm_head` | 1 in, 1 out | Standalone LM head |
| `LogitSoftcapStage` | `logit_softcap` | 1 in, 1 out | Tanh logit cap |
| `TokenCrossEntropyStage` | `token_cross_entropy` | 2 in, 1 out | CE loss |
| `RouterLogitsStage` | `router_logits` | 1 in, 1 out | MoE router projection |
| `TopKRouteStage` | `topk_route` | 1 in, 2 out | Top-K expert routing |
| `ExpertDispatchStage` | `expert_dispatch` | 3 in, 1 out | Dispatch to experts |
| `ExpertCombineStage` | `expert_combine` | 1 in, 1 out | Combine expert outputs |
| `LoadBalanceLossStage` | `load_balance_loss` | 1 in, 2 out | Aux load-balance loss |
| `AuxLossAddStage` | `aux_loss_add` | 2 in, 1 out | Add weighted aux loss |
| `DatasetSourceStage` | `dataset_source` | 0 in, 2+ out | Emits tokens/targets |
| `MambaStage` | `mamba` | 1 in, 1 out | Mamba SSM block |
| `TTTLinearStage` | `ttt_linear` | 1 in, 1 out | TTT-Linear layer |
| `DenoiseHeadStage` | `denoise_head` | 1 in, 1 out | Denoising prediction |
| `MaskSchedulerStage` | `mask_scheduler` | 1 in, 1 out | Mask noise scheduler |
| `RandomTimestepsStage` | `random_timesteps` | 1 in, 2 out | Random diffusion timesteps |
| `JEPAMaskStage` | `jepa_mask` | 1 in, 2 out | JEPA input masking |
| `LatentPoolStage` | `latent_pool` | 2 in, 1 out | Masked latent pooling |
| `JEPAProjectorStage` | `jepa_projector` | 1 in, 1 out | JEPA projection head |
| `JEPAPredictorStage` | `jepa_predictor` | 1 in, 1 out | JEPA predictor head |
| `LatentMSELossStage` | `latent_mse_loss` | 2 in, 1 out | Latent MSE loss |
| `SemanticProjectorStage` | `semantic_projector` | 1 in, 3 out | Semantic vector, residual, topic logits |
| `SemanticAlignmentLossStage` | `semantic_alignment_loss` | 2 in, 1 out | Masked semantic topic CE |
| `SemanticHasherStage` | `semantic_hasher` | 1 in, 1 out | Semantic LSH buckets |
| `SemanticHashRouterStage` | `semantic_hash_router` | 4 in, 2 out | Semantic-vocab expert routing |
| `CausalChunkStateStage` | `causal_chunk_state` | 1 in, 1 out | Prefix-safe chunk states |
| `SemanticChunkProjectorStage` | `semantic_chunk_projector` | 1 in, 3 out | Chunk semantic vector, residual, topic logits |
| `SemanticChunkHasherStage` | `semantic_chunk_hasher` | 1 in, 1 out | Chunk semantic LSH buckets |
| `SemanticMoeJepaEvoRouterStage` | `semantic_moe_jepa_evo_router` | 4 in, 3 out | Chunk shared/semantic/free expert routing |
| `BroadcastChunkRoutesStage` | `broadcast_chunk_routes` | 3 in, 2 out | Chunk routes expanded to token routes |
| `RouteBalanceLossStage` | `route_balance_loss` | 1 in, 1 out | Route-density balance loss |
| `RouteSelectionLossStage` | `route_selection_loss` | 2 in, 1 out | Semantic target route supervision |
| `RouteDistillationLossStage` | `route_distillation_loss` | 2 in, 1 out | Target-topic route distillation |
| `BytePatchEmbedStage` | `byte_patch_embed` | 1 in, 1 out | Byte patch embedding |
| `BytePatchMergeStage` | `byte_patch_merge` | 1 in, 1 out | Byte patch merge |
| `ACTHaltGateStage` | `act_halt_gate` | 1 in, 2 out | ACT halt gate |
| `ACTWeightedSumStage` | `act_weighted_sum` | 3 in, 1 out | ACT weighted accumulation |
| `UniversalTransformerStage` | `universal_transformer` | 1 in, 2 out | Universal TX with ACT |

---

## Weight export/import

```python
from neuralfn.inference import export_to_pt, import_from_pt, export_quantized_pt, import_quantized_pt

export_to_pt(graph: NeuronGraph, path: str | Path) -> None
import_from_pt(graph: NeuronGraph, path: str | Path) -> None

export_quantized_pt(graph: NeuronGraph, path: str | Path, scheme: str = "int8") -> None
#   scheme: "int8" (per-channel int8) or "ternary" ({-1,0,1})
import_quantized_pt(graph: NeuronGraph, path: str | Path) -> None
```

---

## InferenceCache

```python
from neuralfn.inference import InferenceCache

cache = InferenceCache(graph: NeuronGraph, device: str | None = None)
```

Stateful KV cache manager for autoregressive generation. Wraps a `CompiledTorchGraph`.

**Methods:**
- `reset() -> None` -- clear KV cache for new sequence
- `step(token_ids: Tensor) -> Tensor` -- run one step, returns logits for last position
  - First call: full prompt `(batch, seq)`. Subsequent calls: single token `(batch, 1)`.
  - For training graphs (2 inputs), dummy targets are auto-generated.

**Attributes:** `compiled`, `device`, `_cache`
