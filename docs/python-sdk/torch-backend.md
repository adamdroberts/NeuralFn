# neuralfn.torch_backend

PyTorch compilation and training infrastructure. Compiles `NeuronGraph` instances into `nn.Module` pipelines for GPU training and inference.

---

## CompiledTorchGraph

```python
class CompiledTorchGraph(nn.Module):
    def __init__(self, graph: NeuronGraph) -> None
```

Compiles a `NeuronGraph` into an `nn.Module` by walking the graph in topological order and instantiating a PyTorch module for each node. Raises `ValueError` if the graph has cycles.

Function nodes are also materialized as fixed child `nn.Module` instances, so `torch.compile` sees per-node execution rather than a single generic dispatcher. That keeps compiled CUDA BF16 graphs from repeatedly recompiling when the graph mixes `Long`, BF16, and FP32 tensors. Loss stages still promote to FP32 only inside their scalar reduction.

### Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `graph` | `NeuronGraph` | The graph to compile (must be acyclic) |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `graph` | `NeuronGraph` | The source graph |
| `order` | `list[str]` | Topological node order |
| `node_modules` | `nn.ModuleDict` | Compiled modules keyed by node instance_id |

### Methods

#### `forward(*inputs: Tensor) -> tuple[Tensor, ...]`

Run a forward pass. Positional tensor arguments map to the graph's input nodes in order. Returns a tuple of output tensors.

#### `trace(*inputs: Tensor) -> tuple[tuple[Tensor, ...], dict[str, tuple[Tensor, ...]]]`

Run a forward pass and return both the output tensors and a trace dict mapping every node instance_id to its output tensors. For subgraph nodes, child traces are prefixed with `"{parent_id}/{child_id}"`.

#### `sync_state_back(graph: NeuronGraph | None = None) -> None`

Write the current PyTorch state_dict back into each node's `module_state` field (base64-encoded). Recursively syncs nested subgraph nodes. If `graph` is None, syncs to `self.graph`.

---

## TorchTrainConfig

```python
@dataclass
class TorchTrainConfig:
    learning_rate: float = 3e-4
    epochs: int = 50
    batch_size: int = 8
    weight_decay: float = 0.01
    device: str = "cuda"
    amp_dtype: str = "bfloat16"
    compile: bool = False
    activation_checkpointing: bool = False
    fsdp2_enabled: bool = False
    max_steps: int | None = None
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip_norm: float = 0.0
    optimizer_profile: str = "adamw"
    train_batch_tokens: int | None = None
    warmup_steps: int = 0
    warmdown_fraction: float = 0.75
    lr_decay_iters: int | None = None
    min_lr: float | None = None
    max_wallclock_seconds: float = 0.0
    embed_lr: float | None = None
    head_lr: float | None = None
    tied_embed_lr: float | None = None
    matrix_lr: float | None = None
    scalar_lr: float | None = None
    muon_momentum: float = 0.95
    muon_backend_steps: int = 5
    muon_momentum_warmup_start: float = 0.85
    muon_momentum_warmup_steps: int = 500
    drop_last: bool | None = None
    respect_epoch_boundaries: bool = False
    evolutionary: bool = False
    evo_population_size: int = 50
    evo_mutation_rate: float = 0.1
    evo_mutation_scale: float = 0.3
    evo_crossover_rate: float = 0.5
    evo_tournament_size: int = 3
    evo_elite_count: int = 2
    evo_seed: int | None = None
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `learning_rate` | `float` | `3e-4` | AdamW learning rate |
| `epochs` | `int` | `50` | Number of training epochs |
| `batch_size` | `int` | `8` | Batch size |
| `weight_decay` | `float` | `0.01` | AdamW weight decay |
| `device` | `str` | `"cuda"` | Device string |
| `amp_dtype` | `str` | `"bfloat16"` | AMP dtype (`"bfloat16"`, `"float16"`, or `"float32"`) |
| `compile` | `bool` | `False` | Enable `torch.compile` |
| `activation_checkpointing` | `bool` | `False` | Enable activation checkpointing |
| `fsdp2_enabled` | `bool` | `False` | Enable FSDP2 sharding |
| `max_steps` | `int \| None` | `None` | Stop after this many steps (None=no limit) |
| `beta1` | `float` | `0.9` | Adam-family beta1 for the split optimizer profile |
| `beta2` | `float` | `0.95` | Adam-family beta2 for the split optimizer profile |
| `adam_eps` | `float` | `1e-8` | Adam epsilon |
| `grad_clip_norm` | `float` | `0.0` | Global grad clipping threshold (`0.0` disables clipping) |
| `optimizer_profile` | `str` | `"adamw"` | `"adamw"` for the legacy single-optimizer path or `"parameter_golf"` for split optimizers + Muon |
| `train_batch_tokens` | `int \| None` | `None` | Target token budget per optimization step; drives gradient accumulation |
| `warmup_steps` | `int` | `0` | Optional warmup-priming steps for the parameter-golf profile |
| `warmdown_fraction` | `float` | `0.75` | Fraction of total optimizer steps used for linear LR warmdown at the tail of training |
| `lr_decay_iters` | `int \| None` | `None` | When set, enables cosine LR decay across this many optimizer steps and overrides `warmdown_fraction` |
| `min_lr` | `float \| None` | `None` | LR floor used by cosine decay; when omitted and `lr_decay_iters` is set, defaults to `learning_rate / 10` |
| `max_wallclock_seconds` | `float` | `0.0` | Optional wallclock cap for early stopping only; does not change LR schedule semantics |
| `embed_lr` | `float \| None` | `None` | Learning rate for embedding parameters |
| `head_lr` | `float \| None` | `None` | Learning rate for untied LM head parameters |
| `tied_embed_lr` | `float \| None` | `None` | Learning rate for tied embedding/head weights |
| `matrix_lr` | `float \| None` | `None` | Learning rate for matrix-shaped parameters optimized with Muon |
| `scalar_lr` | `float \| None` | `None` | Learning rate for scalar/vector/control parameters |
| `muon_momentum` | `float` | `0.95` | Muon momentum |
| `muon_backend_steps` | `int` | `5` | Newton-Schulz iterations used by Muon |
| `muon_momentum_warmup_start` | `float` | `0.85` | Starting Muon momentum during warmup |
| `muon_momentum_warmup_steps` | `int` | `500` | Number of steps used to ramp Muon momentum to its final value |
| `drop_last` | `bool \| None` | `None` | Override the runtime-specific drop-last policy |
| `respect_epoch_boundaries` | `bool` | `False` | Keep each epoch to one loader pass and allow a short final accumulation step instead of cycling batches |
| `evolutionary` | `bool` | `False` | Use population-based search over trainable torch parameters instead of gradient descent |
| `evo_population_size` | `int` | `50` | Number of candidates scored each generation |
| `evo_mutation_rate` | `float` | `0.1` | Probability of mutating each parameter during offspring generation |
| `evo_mutation_scale` | `float` | `0.3` | Gaussian noise scale used for evolutionary mutations |
| `evo_crossover_rate` | `float` | `0.5` | Probability of copying each parameter from the second parent |
| `evo_tournament_size` | `int` | `3` | Number of candidates compared during tournament selection |
| `evo_elite_count` | `int` | `2` | Number of top candidates copied unchanged into the next generation |
| `evo_seed` | `int \| None` | `None` | Optional RNG seed for the evolutionary population |

---

## TorchTrainer

```python
class TorchTrainer:
    def __init__(self, graph: NeuronGraph, config: TorchTrainConfig | None = None) -> None
```

End-to-end trainer for torch-runtime graphs. Handles dataset loading, compilation, AMP, and training loop.

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `NeuronGraph` | *(required)* | The graph to train |
| `config` | `TorchTrainConfig \| None` | `None` | Training configuration (uses defaults if None) |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `graph` | `NeuronGraph` | The source graph |
| `config` | `TorchTrainConfig` | Training configuration |
| `loss_history` | `list[float]` | Per-epoch loss values |

### Methods

#### `stop() -> None`

Signal the training loop to stop after the current epoch.

#### `train(train_inputs, train_targets, *, on_epoch=None, on_step=None) -> list[float]`

```python
def train(
    self,
    train_inputs: list[list[int]] | Tensor,
    train_targets: list[list[int]] | Tensor,
    *,
    on_epoch: Callable[[int, float], None] | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> list[float]
```

Run the training loop. Expects integer token arrays of shape `[batch, seq_len]`.

If the graph contains a `dataset_source` node with configured `dataset_names`, the dataset is loaded automatically and `train_inputs`/`train_targets` are ignored. Role-aware dataset loading now supports semantic routing layouts as well, including `semantic_router_moe`, `jepa_semantic_hybrid`, and `semantic_moe_jepa_evo` graphs whose flat compiled input contract is `(tokens, targets, sem_targets)`. For these presets, `sem_targets` are categorical vocab-topic IDs rather than quantized semantic vectors. When a graph only has `semantic_data_source`, the trainer now synthesizes safe placeholder `tokens` / `targets` tensors instead of feeding categorical semantic IDs into the token embedding path.

For `semantic_moe_jepa_evo`, normal gradient training can run periodic route evolution after optimizer steps. The trainer evaluates a small population of router-bias/table candidates on recent macro-batches and writes the best candidate back to the route-only parameters; the main model weights still train through the configured optimizer.

When `config.evolutionary` is `True`, the trainer keeps the same dataset loading, epoch accounting, `max_steps`, `train_batch_tokens`, wallclock cap, callbacks, and export flow, but replaces optimizer steps with generation-based selection, crossover, and mutation over the flattened trainable parameter vector. In this mode, `max_steps` counts generations and the gradient-only optimizer knobs (`optimizer_profile`, learning-rate settings, Muon, Adam betas/epsilon, warmup/warmdown, and grad clipping) are ignored.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_inputs` | `list[list[int]] \| Tensor` | *(required)* | Input token sequences |
| `train_targets` | `list[list[int]] \| Tensor` | *(required)* | Target token sequences |
| `on_epoch` | `Callable \| None` | `None` | Callback `(epoch_idx, loss)` called after each epoch |
| `on_step` | `Callable \| None` | `None` | Callback receiving a warmup/train progress dict after each optimizer step |

**Returns:** List of per-epoch loss values.

`on_step` payloads always include `phase`, `loss`, `elapsed_seconds`, `optimization_method`, and `learning_rates` (empty for evolutionary mode). Train-phase payloads also include `step`, `max_steps`, `epoch`, `max_epochs`, `epoch_step`, `steps_per_epoch`, and `grad_accum_steps`. Warmup payloads include `step` and `warmup_steps`.

---

## build_module

```python
def build_module(module_type: str, module_config: dict[str, Any]) -> nn.Module
```

Factory function that maps a `module_type` string and config dict to a concrete `nn.Module` instance. Dispatches to the appropriate `*Stage` class.

---

## RoleMappedDataset

```python
class RoleMappedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset: torch.utils.data.Dataset, roles: list[str]) -> None
```

Wraps a base dataset and maps graph input port roles (`"tokens"`, `"targets"`, `"enc_tokens"`, `"dec_tokens"`) to dataset columns.

### Methods

#### `__len__() -> int`

Returns the length of the base dataset.

#### `__getitem__(idx: int) -> tuple[Tensor, ...]`

Returns a tuple of tensors mapped according to the role list.

---

## Stage Classes

Each `*Stage` class is an `nn.Module` implementing a single computational step. These are instantiated by `build_module()` and composed by `CompiledTorchGraph`.

| Stage Class | Module Type | Description |
|-------------|-------------|-------------|
| `TokenEmbeddingStage` | `token_embedding` | Token lookup embedding, outputs (hidden, weight) |
| `RMSNormStage` | `rms_norm` | Root-mean-square normalization |
| `LayerNormStage` | `layer_norm` | Standard layer normalization |
| `ResidualMixStage` | `residual_mix` | Learnable residual mixing (primary + skip scales) |
| `ResidualAddStage` | `residual_add` | Scaled residual addition |
| `CausalSelfAttentionStage` | `causal_self_attention` | Full self-attention with Q/K/V proj, RoPE, GQA, QK gain |
| `FusedCausalAttentionStage` | `fused_causal_attention` | Fused attention for megakernel runtime |
| `MLPReluSquaredStage` | `mlp_relu2` | MLP with ReLU-squared activation |
| `TiedLMHeadStage` | `tied_lm_head` | LM head using tied embedding weight |
| `LMHeadStage` | `lm_head` | Standalone LM head projection |
| `LogitSoftcapStage` | `logit_softcap` | Tanh-based logit soft-capping |
| `TokenCrossEntropyStage` | `token_cross_entropy` | Cross-entropy loss for token prediction |
| `LinearStage` | `linear` | General linear projection |
| `BitLinearTernaryStage` | `bitlinear_ternary` | 1.58-bit ternary quantized linear |
| `RandMapAdapterStage` | `randmap_adapter` | Random-map adapter (fixed random proj + learnable) |
| `MambaStage` | `mamba` | Mamba SSM block |
| `ReshapeHeadsStage` | `reshape_heads` | Reshape tensor to multi-head format |
| `MergeHeadsStage` | `merge_heads` | Merge multi-head format back |
| `RepeatKVStage` | `repeat_kv` | Repeat KV heads for GQA |
| `RotaryEmbeddingStage` | `rotary_embedding` | Apply rotary position embeddings to Q and K |
| `QKGainStage` | `qk_gain` | Per-head learnable Q scaling |
| `ScaledDotProductAttentionStage` | `scaled_dot_product_attention` | SDPA core (supports multiple backends) |
| `DropoutStage` | `dropout` | Standard dropout |
| `GeluStage` | `gelu` | GELU activation (tensor) |
| `SwiGLUStage` | `swiglu` | SwiGLU feed-forward block |
| `AbsolutePositionEmbeddingStage` | `absolute_position_embedding` | Learnable absolute position embeddings |
| `KVCacheReadStage` | `kv_cache_read` | Read from KV cache |
| `KVCacheWriteStage` | `kv_cache_write` | Write to KV cache |
| `KVPCAEncodeStage` | `kv_pca_encode` | PCA-compress KV tensors |
| `KVPCADecodeStage` | `kv_pca_decode` | PCA-decompress KV tensors |
| `KVQuantPackStage` | `kv_quant_pack` | Quantize and pack KV tensors |
| `KVQuantUnpackStage` | `kv_quant_unpack` | Unpack and dequantize KV tensors |
| `RouterLogitsStage` | `router_logits` | MoE router logit projection |
| `TopKRouteStage` | `topk_route` | Top-K expert routing |
| `ExpertDispatchStage` | `expert_dispatch` | Dispatch tokens to selected experts |
| `BroadcastExpertRoutesStage` | `broadcast_expert_routes` | Expand a shared batch-level expert route across the sequence axis |
| `ExpertCombineStage` | `expert_combine` | Combine expert outputs |
| `LoadBalanceLossStage` | `load_balance_loss` | MoE load-balancing auxiliary loss |
| `AuxLossAddStage` | `aux_loss_add` | Weighted addition of auxiliary loss to main loss |
| `LossScaleStage` | `loss_scale` | Scalar multiplier for a single loss term |
| `DatasetSourceStage` | `dataset_source` | Inline dataset source node |
| `SemanticDataSourceStage` | `semantic_data_source` | Inline vocab-topic target source used by the hybrid preset |
| `RandomTimestepsStage` | `random_timesteps` | Generate random diffusion timesteps |
| `JEPAMaskStage` | `jepa_mask` | JEPA input masking (random or block) |
| `LatentPoolStage` | `latent_pool` | Masked latent pooling |
| `JEPAProjectorStage` | `jepa_projector` | JEPA projection head |
| `JEPAPredictorStage` | `jepa_predictor` | JEPA predictor head |
| `LatentMSELossStage` | `latent_mse_loss` | MSE loss in latent space |
| `BytePatchEmbedStage` | `byte_patch_embed` | Byte-level patch embedding (HNet) |
| `BytePatchMergeStage` | `byte_patch_merge` | Byte-level patch merge (HNet) |
| `ACTHaltGateStage` | `act_halt_gate` | Adaptive Computation Time halt gate |
| `ACTWeightedSumStage` | `act_weighted_sum` | ACT weighted state accumulation |
| `UniversalTransformerStage` | `universal_transformer` | Universal transformer with ACT |
| `DenoiseHeadStage` | `denoise_head` | Denoising prediction head (diffusion) |
| `MaskSchedulerStage` | `mask_scheduler` | Mask-based noise scheduler (diffusion) |
| `TTTLinearStage` | `ttt_linear` | Test-time training linear layer |
| `SemanticProjectorStage` | `semantic_projector` | Semantic projection head that emits both the internal 9-D state and per-dimension topic logits |
| `SemanticAlignmentLossStage` | `semantic_alignment_loss` | Masked categorical loss over vocab-topic logits |
| `SemanticHasherStage` | `semantic_hasher` | In-graph LSH bucketing |
| `SemanticMoERouterStage` | `semantic_moe_router` | Legacy cosine-router stage kept for compatibility |
| `SemanticHashRouterStage` | `semantic_hash_router` | Hash-aware router that maps semantic vocabulary dimensions onto fixed experts |
| `CausalChunkStateStage` | `causal_chunk_state` | Prefix-safe chunk state extraction for chunk-level semantic routing |
| `SemanticChunkProjectorStage` | `semantic_chunk_projector` | Chunk semantic projection with topic logits and residual state |
| `SemanticChunkHasherStage` | `semantic_chunk_hasher` | LSH bucketing for chunk semantic vectors |
| `SemanticMoeJepaEvoRouterStage` | `semantic_moe_jepa_evo_router` | Chunk-level semantic/free expert router with shared experts and route-evolution parameters |
| `BroadcastChunkRoutesStage` | `broadcast_chunk_routes` | Expand chunk-level expert routes to per-token MoE routes |
| `RouteBalanceLossStage` | `route_balance_loss` | Balance loss over route logits |
| `RouteSelectionLossStage` | `route_selection_loss` | Supervised route loss against semantic targets |
| `RouteDistillationLossStage` | `route_distillation_loss` | Distill target semantic topic distributions into route logits |
| `RoutedAttentionExpertsStage` | `routed_attention_experts` | Attention-capable experts applied to the full hidden sequence |
| `AttentionlessDecoderStage` | `attentionless_decoder` | Legacy decoder stage retained for compatibility |
| `SoftmaxDistillationLossStage` | `softmax_distillation_loss` | Distillation loss for experimental semantic workflows |

---

## Config Factory Functions

Default configuration constructors used by builtin neuron definitions:

| Function | Returns |
|----------|---------|
| `default_gpt_config()` | Base GPT config dict |
| `default_token_embedding_config()` | Token embedding config |
| `default_rms_norm_config()` | RMS norm config |
| `default_attention_config()` | Causal self-attention config |
| `default_fused_attention_config()` | Fused attention config |
| `default_linear_config()` | Linear projection config |
| `default_reshape_heads_config(num_heads=None)` | Reshape heads config |
| `default_merge_heads_config()` | Merge heads config |
| `default_repeat_kv_config()` | Repeat KV config |
| `default_rotary_embedding_config()` | Rotary embedding config |
| `default_qk_gain_config()` | QK gain config |
| `default_scaled_dot_product_attention_config()` | SDPA config |
| `default_residual_mix_config()` | Residual mix config |
| `default_residual_add_config()` | Residual add config |
| `default_mlp_config()` | MLP config |
| `default_lm_head_config()` | LM head config |
| `default_logit_softcap_config()` | Logit softcap config |
| `default_dataset_source_config()` | Dataset source config |
| `default_loss_scale_config()` | Loss-scale config |
| `default_kv_pca_config()` | KV PCA config |
| `default_kv_quant_unpack_config()` | KV quant unpack config |
