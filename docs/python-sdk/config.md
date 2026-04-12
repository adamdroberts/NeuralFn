# neuralfn.config

Type aliases, configuration dataclasses, and preset builder functions for model architecture specification.

---

## Type Aliases

```python
ObjectiveType    = Literal["ar", "diffusion", "jepa", "jepa_semantic", "seq2seq"]
BackboneType     = Literal["gpt2", "nanogpt", "llama", "mixllama", "jamba", "universal", "ttt", "hnet"]
TokenizationType = Literal["sp", "byte_hnet"]
SparsityType     = Literal["dense", "moe"]
CompressionType  = Literal["none", "ternary_b158", "binary_1bit", "kv_pca"]
AdapterType      = Literal["none", "lora", "randmap"]
RuntimeType      = Literal["eager", "compile", "sdpa", "megakernel"]
```

---

## TemplateSpec

```python
@dataclass
class TemplateSpec:
    objective: ObjectiveType = "ar"
    backbone: BackboneType = "gpt2"
    tokenization: TokenizationType = "sp"
    sparsity: SparsityType = "dense"
    compression: CompressionType = "none"
    adapter: AdapterType = "none"
    runtime: RuntimeType = "eager"
    backend_capabilities: dict[str, bool] = field(default_factory=lambda: {
        "compile": True,
        "sdpa": True,
        "cache": True,
        "quantized_export": True,
        "megakernel": False,
    })
```

High-level description of a model architecture template. Controls which graph builders and runtime features are used.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `objective` | `ObjectiveType` | `"ar"` | Training objective |
| `backbone` | `BackboneType` | `"gpt2"` | Transformer backbone variant |
| `tokenization` | `TokenizationType` | `"sp"` | Tokenization mode |
| `sparsity` | `SparsityType` | `"dense"` | Dense or MoE |
| `compression` | `CompressionType` | `"none"` | Weight compression scheme |
| `adapter` | `AdapterType` | `"none"` | Adapter type |
| `runtime` | `RuntimeType` | `"eager"` | Execution runtime |
| `backend_capabilities` | `dict[str, bool]` | *(see above)* | Auto-derived capability flags |

---

## resolve_backend_capabilities

```python
def resolve_backend_capabilities(spec: TemplateSpec) -> dict[str, bool]
```

Derive the correct `backend_capabilities` from the other fields on `spec`. Called automatically by the preset builders.

Returns a dict with keys: `"compile"`, `"sdpa"`, `"cache"`, `"quantized_export"`, `"megakernel"`.

---

## BlockSpec

```python
@dataclass
class BlockSpec:
    family: str
    norm_type: str = "layernorm"
    mlp_type: str = "gelu"
    pos_encoding: str = "absolute"
    attention_backend: str = "sdpa"
    num_heads: int = 4
    num_kv_heads: int | None = None
    is_causal: bool = True
    linear_bias: bool = True
    dropout_p: float = 0.0
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    mlp_multiplier: float = 4.0
    multiple_of: int | None = None
    experts: int | None = None
    top_k: int | None = None
    shared_experts: int = 0
    router_aux_loss_coef: float = 0.0
    compression: str = "none"
    adapter_dim: int = 0
    ttt_hidden_dim: int = 16
    byte_patch_size: int = 4
    byte_patch_stride: int = 4
```

Specifies the architecture of a single transformer block.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `family` | `str` | *(required)* | Block family: `"nanogpt"`, `"gpt2"`, `"llama"`, `"mixllama"`, `"jamba"`, `"ttt"`, `"hnet"`, `"universal"` |
| `norm_type` | `str` | `"layernorm"` | `"layernorm"` or `"rmsnorm"` |
| `mlp_type` | `str` | `"gelu"` | `"gelu"`, `"swiglu"`, or `"moe"` |
| `pos_encoding` | `str` | `"absolute"` | `"absolute"` or `"rope"` |
| `attention_backend` | `str` | `"sdpa"` | `"sdpa"`, `"flex"`, or `"math"` |
| `num_heads` | `int` | `4` | Number of attention heads |
| `num_kv_heads` | `int \| None` | `None` | Number of KV heads (None=MHA, smaller=GQA/MQA) |
| `is_causal` | `bool` | `True` | Whether attention is causal |
| `linear_bias` | `bool` | `True` | Whether linear layers have bias |
| `dropout_p` | `float` | `0.0` | Dropout probability |
| `rope_theta` | `float` | `10000.0` | RoPE frequency base |
| `rope_scaling` | `dict \| None` | `None` | Optional RoPE scaling config |
| `mlp_multiplier` | `float` | `4.0` | MLP hidden dimension multiplier |
| `multiple_of` | `int \| None` | `None` | Round MLP hidden dim to this multiple |
| `experts` | `int \| None` | `None` | Number of MoE experts |
| `top_k` | `int \| None` | `None` | Top-K expert routing |
| `shared_experts` | `int` | `0` | Number of shared experts |
| `router_aux_loss_coef` | `float` | `0.0` | Router auxiliary loss coefficient |
| `compression` | `str` | `"none"` | Per-block compression: `"none"`, `"ternary_b158"`, `"kv_pca"` |
| `adapter_dim` | `int` | `0` | Adapter bottleneck dimension (0=disabled) |
| `ttt_hidden_dim` | `int` | `16` | TTT layer hidden dimension |
| `byte_patch_size` | `int` | `4` | Byte-level patch size (HNet) |
| `byte_patch_stride` | `int` | `4` | Byte-level patch stride (HNet) |

---

## ModelSpec

```python
@dataclass
class ModelSpec:
    model_dim: int = 128
    num_layers: int = 4
    vocab_size: int = 256
    tie_embeddings: bool = True
    logit_softcap: float = 0.0
    block_spec: BlockSpec = field(default_factory=lambda: BlockSpec(family="gpt2"))
    template: TemplateSpec = field(default_factory=TemplateSpec)
    jepa_latent_dim: int = 128
    jepa_mask_ratio: float = 0.5
    jepa_mask_strategy: str = "random"
    jepa_num_blocks: int = 4
    jepa_min_block_ratio: float = 0.1
    jepa_max_block_ratio: float = 0.25
    ema_decay: float = 0.99
    max_recurrence_steps: int = 4
    halt_epsilon: float = 0.01
```

Complete model architecture specification.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_dim` | `int` | `128` | Hidden dimension |
| `num_layers` | `int` | `4` | Number of transformer blocks |
| `vocab_size` | `int` | `256` | Vocabulary size |
| `tie_embeddings` | `bool` | `True` | Tie input embedding and LM head weights |
| `logit_softcap` | `float` | `0.0` | Logit soft-capping (0=disabled, >0=tanh softcap) |
| `block_spec` | `BlockSpec` | GPT-2 defaults | Per-block architecture spec |
| `template` | `TemplateSpec` | defaults | High-level template spec |
| `jepa_latent_dim` | `int` | `128` | JEPA latent dimension |
| `jepa_mask_ratio` | `float` | `0.5` | JEPA mask ratio |
| `jepa_mask_strategy` | `str` | `"random"` | JEPA masking strategy |
| `jepa_num_blocks` | `int` | `4` | JEPA block count |
| `jepa_min_block_ratio` | `float` | `0.1` | Minimum JEPA block ratio |
| `jepa_max_block_ratio` | `float` | `0.25` | Maximum JEPA block ratio |
| `ema_decay` | `float` | `0.99` | EMA decay for JEPA target encoder |
| `max_recurrence_steps` | `int` | `4` | Universal transformer max recurrence |
| `halt_epsilon` | `float` | `0.01` | ACT halting threshold |

---

## model_spec_to_dict

```python
def model_spec_to_dict(spec: ModelSpec) -> dict[str, Any]
```

Convert a `ModelSpec` to a plain dictionary via `dataclasses.asdict`.

---

## Preset Builder Functions

All preset builders accept `**kwargs` to override default values. Common kwargs include `model_dim` (or `n_embd`), `num_layers` (or `n_layer`), `vocab_size`, `num_heads`, `num_kv_heads`, `tie_embeddings`, `logit_softcap`, and block-specific parameters.

| Function | Backbone | Sparsity | Runtime | Notes |
|----------|----------|----------|---------|-------|
| `build_nanogpt_spec(**kwargs)` | nanogpt | dense | eager | LayerNorm, GELU MLP, absolute pos |
| `build_gpt2_spec(**kwargs)` | gpt2 | dense | eager | LayerNorm, GELU MLP, absolute pos, linear bias |
| `build_llama_spec(**kwargs)` | llama | dense | eager | RMSNorm, SwiGLU, RoPE, GQA |
| `build_mixllama_spec(**kwargs)` | mixllama | moe | eager | RMSNorm, MoE MLP, RoPE, GQA |
| `build_llama_fast_spec(**kwargs)` | llama | dense | compile | Llama with torch.compile |
| `build_mixllama_fast_spec(**kwargs)` | mixllama | moe | compile | MoE Llama with torch.compile |
| `build_jamba_hybrid_spec(**kwargs)` | jamba | moe | compile | Jamba hybrid (attention + Mamba interleaved) |
| `build_ternary_b158_spec(**kwargs)` | llama | dense | compile | BitLinear ternary compression |
| `build_decoder2encoder_moe_spec(**kwargs)` | llama | moe | compile | Seq2seq with MoE decoder |
| `build_diffllama_spec(**kwargs)` | llama | dense | compile | Discrete diffusion objective |
| `build_ttt_llama_spec(**kwargs)` | ttt | dense | compile | Test-time training linear layers |
| `build_llm_jepa_spec(**kwargs)` | llama | dense | compile | JEPA self-supervised objective |
| `build_hnet_lm_spec(**kwargs)` | hnet | dense | compile | Byte-level hierarchical tokenization |
| `build_universal_llama_spec(**kwargs)` | universal | dense | compile | Universal transformer with ACT |
| `build_llama_megakernel_spec(**kwargs)` | llama | dense | megakernel | Fused attention megakernel |
| `build_kv_pca_llama_spec(**kwargs)` | llama | dense | compile | KV cache PCA compression |
| `build_semantic_router_moe_spec(**kwargs)` | mixllama | moe | compile | **[Experimental]** AR-only semantic-router control with shared routed MoE blocks |

---

## [Experimental] Semantic routing fields and builders

### Additional `ModelSpec` fields [Experimental]

These fields exist on `ModelSpec` for the **[Experimental]** semantic routing presets (`semantic_router_moe` and `jepa_semantic_hybrid`) and related graphs:

| Field [Experimental] | Type | Default | Description |
|----------------------|------|---------|-------------|
| `semantic_dim` | `int` | `9` | Dimensionality of the grounded semantic vector (8 vocabulary dimensions + 1 taxonomy hash). |
| `semantic_residual_dim` | `int` | `64` | Residual projector width used by the semantic projector module. |
| `semantic_n_lsh_tables` | `int` | `8` | Number of LSH tables for bucketing semantic vectors. |
| `semantic_n_lsh_planes` | `int` | `12` | Number of hyperplanes per table (hash width). |
| `semantic_table_path` | `str` | `""` | Optional filesystem path for persistent LSH / semantic table data (empty = in-memory default). |
| `ar_loss_coef` | `float` | `1.0` | Scalar applied to the routed autoregressive loss term. |
| `jepa_loss_coef` | `float` | `0.25` | Scalar applied to the JEPA latent MSE term (`jepa_semantic_hybrid` only). |
| `semantic_align_loss_coef` | `float` | `0.5` | Scalar applied to the masked semantic topic cross-entropy term. |

### `build_semantic_router_moe_spec` [Experimental]

```python
def build_semantic_router_moe_spec(**kwargs: Any) -> ModelSpec
```

**[Experimental]** Factory for the `semantic_router_moe` template: AR-only MixLLaMA/MoE with vocab-topic semantic projection, LSH hashing, a fixed 8-expert dimension map, and a shared externally supplied MoE route reused across every decoder block. The generated stage trains two connected losses: next-token CE and masked semantic topic cross-entropy.

**Breaking-change note [Experimental]:**
- `semantic_router_moe` requires exactly `8` experts so the routed semantic dimensions line up one-to-one with the expert map.
- The compiled/root input contract is `(tokens, targets, sem_targets)`.

In addition to the usual LLaMA/MoE overrides (`n_layer` / `num_layers`, `n_embd` / `model_dim`, `experts`, `top_k`, etc.), this builder also recognizes:

- `rope_base` / `rope_theta` for the normal attention path
- `qk_gain_init` for attention query scaling
- `ar_loss_coef` and `semantic_align_loss_coef` for the two loss terms

**Disclaimer [Experimental]:** This builder is a research-control preset intended to isolate the semantic router hypothesis without JEPA. The graph shape and tuning knobs may change.

### `build_jepa_semantic_hybrid_spec` [Experimental]

```python
def build_jepa_semantic_hybrid_spec(**kwargs: Any) -> ModelSpec
```

**[Experimental]** Factory for the `jepa_semantic_hybrid` template: JEPA objective with LLaMA-style blocks, MoE sparsity, `torch.compile` runtime, plus vocab-topic semantic projection, LSH hashing, a fixed dimension-to-expert router, and routed attention experts over the full hidden sequence. The generated stage trains three connected losses: AR next-token CE, JEPA latent MSE, and masked semantic topic cross-entropy.

**Breaking-change note [Experimental]:**
- `jepa_semantic_hybrid` now requires exactly `8` experts.
- `sem_targets` are categorical topic IDs with `-100` ignore sentinels in the first 8 positions plus a derived taxonomy-hash slot, not quantized semantic vectors.

In addition to the usual LLaMA/MoE overrides (`n_layer` / `num_layers`, `n_embd` / `model_dim`, `experts`, `top_k`, etc.), this builder also recognizes:

- `rope_base` / `rope_theta` for routed expert attention
- `qk_gain_init` for expert-attention query scaling
- `ar_loss_coef`, `jepa_loss_coef`, and `semantic_align_loss_coef` for the three loss terms

**Breaking-change note [Experimental]:** The hybrid preset's compiled/root input contract is now `(tokens, targets, sem_targets)` instead of `(tokens, sem_targets)`.

**Disclaimer [Experimental]:** This builder and its `ModelSpec` fields are research prototypes and may change.
