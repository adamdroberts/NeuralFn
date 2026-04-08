# neuralfn.torch_backend

PyTorch compilation and training infrastructure. Compiles `NeuronGraph` instances into `nn.Module` pipelines for GPU training and inference.

---

## CompiledTorchGraph

```python
class CompiledTorchGraph(nn.Module):
    def __init__(self, graph: NeuronGraph) -> None
```

Compiles a `NeuronGraph` into an `nn.Module` by walking the graph in topological order and instantiating a PyTorch module for each node. Raises `ValueError` if the graph has cycles.

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

#### `train(train_inputs, train_targets, *, on_epoch=None) -> list[float]`

```python
def train(
    self,
    train_inputs: list[list[int]] | Tensor,
    train_targets: list[list[int]] | Tensor,
    *,
    on_epoch: Callable[[int, float], None] | None = None,
) -> list[float]
```

Run the training loop. Expects integer token arrays of shape `[batch, seq_len]`.

If the graph contains a `dataset_source` node with configured `dataset_names`, the dataset is loaded automatically and `train_inputs`/`train_targets` are ignored.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_inputs` | `list[list[int]] \| Tensor` | *(required)* | Input token sequences |
| `train_targets` | `list[list[int]] \| Tensor` | *(required)* | Target token sequences |
| `on_epoch` | `Callable \| None` | `None` | Callback `(epoch_idx, loss)` called after each epoch |

**Returns:** List of per-epoch loss values.

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
| `ExpertCombineStage` | `expert_combine` | Combine expert outputs |
| `LoadBalanceLossStage` | `load_balance_loss` | MoE load-balancing auxiliary loss |
| `AuxLossAddStage` | `aux_loss_add` | Weighted addition of auxiliary loss to main loss |
| `DatasetSourceStage` | `dataset_source` | Inline dataset source node |
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
| `default_kv_pca_config()` | KV PCA config |
| `default_kv_quant_unpack_config()` | KV quant unpack config |
