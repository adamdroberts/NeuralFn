# neuralfn.config

Type aliases, configuration dataclasses, and preset builder functions for model architecture specification.

---

## Type Aliases

```python
ObjectiveType    = Literal[
    "ar", "diffusion", "jepa", "ar_jepa", "jepa_semantic",
    "semantic_router", "semantic_router_jepa", "semantic_dense_jepa_evo",
    "semantic_moe_jepa_evo",
    "seq2seq", "sft", "dpo", "ppo", "reward_model",
]
BackboneType     = Literal["gpt2", "nanogpt", "llama", "mixllama", "jamba", "universal", "ttt", "hnet"]
TokenizationType = Literal["sp", "byte_hnet"]
SparsityType     = Literal["dense", "moe"]
CompressionType  = Literal["none", "ternary_b158", "binary_1bit", "kv_pca"]
AdapterType      = Literal["none", "lora", "qlora", "randmap"]
RuntimeType      = Literal["eager", "compile", "sdpa", "megakernel"]
RouterModeType   = Literal["none", "standard", "semantic"]
FineTuneObjective = Literal["pretrain", "sft", "dpo", "ppo", "reward_model"]
DPOLossType       = Literal["sigmoid", "hinge", "ipo"]
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
    router_mode: RouterModeType = "none"
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
| `router_mode` | `RouterModeType` | `"none"` | MoE router family: none, learned standard routing, or semantic routing |
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

## GPT Template Catalog

```python
SHIPPED_GPT_TEMPLATE_BASE_PRESETS: tuple[str, ...]
SHIPPED_GPT_TEMPLATE_PRESETS: tuple[str, ...]
```

`SHIPPED_GPT_TEMPLATE_BASE_PRESETS` is the canonical SDK catalog for exact names accepted by `build_model_spec_from_config(config={"preset": ...})`, including aliases and megakernel variants such as `mixllama`, `nanogpt_megakernel`, and `gpt2_megakernel`.

`SHIPPED_GPT_TEMPLATE_PRESETS` extends the base catalog with every generated `<preset>_modern` overlay from `MODERN_BASE_PRESETS`. Native GPT training selectors (`--template-name`, `--template`, `--preset`) and SDK compiled-CLI configs accept every name in this tuple. The compiled GPT launchers also accept any shipped preset through `--base-model <preset>`; they normalize the runtime family to `gpt` and forward the selected preset as `--template-name`, while `gpt`, `gpt2`, `gpt3`, and `nanogpt` remain direct dense GPT family aliases. The compiled dense GPT loop currently runs `gpt`, `gpt2`, `gpt2_modern`, `gpt2_megakernel`, `gpt2_moa`, `gpt3`, `nanogpt`, `nanogpt_modern`, and `nanogpt_megakernel`; the selected template geometry controls context length, width, heads, layers, and dropout metadata. Non-dense and custom-graph selections return explicit native-trainer-missing JSON instead of falling back to Torch or graph-editor tensor flow.

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
    adapter_type: str = "none"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj")
    lora_bias: bool = False
    qlora_group_size: int = 64
    qlora_compute_dtype: str = "bf16"
    ttt_hidden_dim: int = 16
    byte_patch_size: int = 4
    byte_patch_stride: int = 4
    qk_gain_init: float = 1.0
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
| `adapter_type` | `str` | `"none"` | Adapter implementation: `"none"`, `"lora"`, `"qlora"`, or `"randmap"` |
| `lora_rank` | `int` | `8` | LoRA rank for `lora` / `qlora` projections |
| `lora_alpha` | `float` | `16.0` | LoRA scaling alpha |
| `lora_dropout` | `float` | `0.0` | Dropout applied to LoRA inputs |
| `lora_targets` | `tuple[str, ...]` | `("q_proj", "v_proj")` | Projection node roles to wrap with LoRA/qLoRA |
| `lora_bias` | `bool` | `False` | Whether LoRA-wrapped projections include bias |
| `qlora_group_size` | `int` | `64` | qLoRA NF4 quantization group size along input dim |
| `qlora_compute_dtype` | `str` | `"bf16"` | qLoRA dequantization compute dtype |
| `ttt_hidden_dim` | `int` | `16` | TTT layer hidden dimension |
| `byte_patch_size` | `int` | `4` | Byte-level patch size (HNet) |
| `byte_patch_stride` | `int` | `4` | Byte-level patch stride (HNet) |
| `qk_gain_init` | `float` | `1.0` | Initial query/key gain used by routed semantic attention variants |
| `attention_variant` | `str` | `"dense"` | Attention core: `"dense"`, `"differential"`, `"sliding_window"`, `"block_sparse"`, `"nsa"`, `"streaming"`, `"mla"` |
| `use_qk_norm` | `bool` | `False` | Fused RMSNorm on Q and K before SDPA (DeepSeek-V3, Gemma-3) |
| `dyt_alpha_init` | `float` | `1.0` | Initial alpha for `norm_type="dyt"` (Dynamic Tanh) |
| `group_norm_groups` | `int` | `1` | Groups for `norm_type="group_norm"` |
| `diff_lambda_init` | `float` | `0.8` | Initial lambda for differential attention |
| `moe_balance_mode` | `str` | `"aux_loss"` | MoE load balancing: `"aux_loss"` or `"auxfree"` (DeepSeek-V3 bias-adjusted) |
| `auxfree_bias_lr` | `float` | `0.001` | Update rate for the auxfree expert-load bias |
| `router_score_fn` | `str` | `"softmax"` | Router affinity scoring (`softmax`/`sigmoid`/`sqrt_softplus`) |
| `residual_type` | `str` | `"add"` | Residual mix: `"add"` or `"mhc"` (Manifold-Constrained Hyper-Connection) |
| `window_size` | `int \| None` | `None` | Local window for sliding-window / streaming / NSA attention |
| `sparse_block_size` | `int` | `64` | Block size for block-sparse attention |
| `num_sinks` | `int` | `0` | Persistent attention-sink tokens (StreamingLLM / NSA) |
| `nsa_compress_stride` | `int` | `16` | CSA "compress every m KV" stride for native-sparse attention |
| `mx_block_size` | `int` | `32` | OCP microscaling block size for `compression` in `{mxfp4, mxfp8}` |
| `fp8_amax_history_len` | `int` | `16` | Amax-history length for FP8 delayed scaling |
| `fp8_use_stochastic_rounding` | `bool` | `True` | Stochastic-rounding intent flag for FP8 weight updates |

`norm_type` also accepts `"dyt"` and `"group_norm"`; `mlp_type` also accepts `"geglu"`, `"reglu"`, `"solu"`; `compression` also accepts `"fp8_e4m3"`, `"fp8_e5m2"`, `"mxfp4"`, `"mxfp8"`. See the frontier presets in `docs/framework-guide/templates-and-presets.md`.

---

## FineTuneSpec

```python
@dataclass
class FineTuneSpec:
    objective: str = "pretrain"
    base_checkpoint: str = ""
    ref_checkpoint: str = ""
    reward_checkpoint: str = ""
    adapter_only_save: bool = False
    beta: float = 0.1
    dpo_loss_type: str = "sigmoid"
    dpo_label_smoothing: float = 0.0
    kl_coef: float = 0.1
    ppo_clip: float = 0.2
    ppo_vf_coef: float = 0.5
    ppo_ent_coef: float = 0.0
    rollout_length: int = 64
    ppo_epochs_per_rollout: int = 4
    ppo_minibatch_size: int = 4
    gae_gamma: float = 1.0
    gae_lambda: float = 0.95
```

Fine-tuning metadata attached to `ModelSpec.finetune`. `build_gpt_root_graph()`
uses `model_spec.template.objective` to dispatch to the SFT, DPO, PPO, or
reward-model root graph builders. `base_checkpoint` initializes the policy/base
weights, `ref_checkpoint` is used by DPO/PPO reference forwards, and
`reward_checkpoint` is used by PPO reward scoring.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `objective` | `str` | `"pretrain"` | `"sft"`, `"dpo"`, `"ppo"`, or `"reward_model"` for fine-tuning graphs |
| `base_checkpoint` | `str` | `""` | Pretrained base weights path |
| `ref_checkpoint` | `str` | `""` | Frozen reference weights for DPO/PPO |
| `reward_checkpoint` | `str` | `""` | Frozen reward model weights for PPO |
| `adapter_only_save` | `bool` | `False` | Save only adapter/head state when requested by caller |
| `beta` | `float` | `0.1` | DPO reward-temperature parameter |
| `dpo_loss_type` | `str` | `"sigmoid"` | DPO loss variant: `"sigmoid"`, `"hinge"`, or `"ipo"` |
| `dpo_label_smoothing` | `float` | `0.0` | Label smoothing for DPO preference labels |
| `kl_coef` | `float` | `0.1` | PPO KL-to-reference coefficient |
| `ppo_clip` | `float` | `0.2` | PPO clipping range |
| `ppo_vf_coef` | `float` | `0.5` | PPO value-loss coefficient |
| `ppo_ent_coef` | `float` | `0.0` | PPO entropy bonus coefficient |
| `rollout_length` | `int` | `64` | PPO rollout length in tokens |
| `ppo_epochs_per_rollout` | `int` | `4` | PPO optimization epochs per collected rollout |
| `ppo_minibatch_size` | `int` | `4` | PPO minibatch size for rollout optimization |
| `gae_gamma` | `float` | `1.0` | Discount factor for generalized advantage estimation |
| `gae_lambda` | `float` | `0.95` | Lambda parameter for generalized advantage estimation |

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
    semantic_dim: int = NUM_SEMANTIC_DIMS
    semantic_residual_dim: int = 64
    semantic_n_lsh_tables: int = 8
    semantic_n_lsh_planes: int = 12
    semantic_table_path: str = ""
    semantic_vocab_ref: str = DEFAULT_SEMANTIC_VOCAB_REF
    experimental_semantic_router_vecs: bool = False
    route_chunk_size: int = 32
    semantic_shared_experts: int = 2
    semantic_free_experts: int = 8
    route_evo_enabled: bool = True
    route_evo_fraction: float = 0.10
    route_evo_population: int = 8
    route_evo_mutation_scale: float = 0.05
    route_evo_seed: int | None = None
    layer_evo_enabled: bool = False
    layer_evo_index: int | None = None
    layer_evo_fraction: float = 0.10
    layer_evo_population: int = 8
    layer_evo_mutation_scale: float = 0.02
    layer_evo_seed: int | None = None
    ar_loss_coef: float = 1.0
    jepa_loss_coef: float = 0.25
    semantic_align_loss_coef: float = 0.5
    finetune: FineTuneSpec | None = None
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
| `semantic_dim` | `int` | `NUM_SEMANTIC_DIMS` | Grounded semantic vector width for semantic routing presets |
| `semantic_vocab_ref` | `str` | default semantic vocab | Semantic vocabulary reference used by projector/router stages |
| `route_chunk_size` | `int` | `32` | Chunk size for `semantic_dense_jepa_evo` planner updates and `semantic_moe_jepa_evo` route updates |
| `semantic_shared_experts` | `int` | `2` | Always-on shared experts for `semantic_moe_jepa_evo` |
| `semantic_free_experts` | `int` | `8` | Free learned experts for `semantic_moe_jepa_evo` |
| `route_evo_enabled` | `bool` | `True` | Enable periodic route-evolution search for `semantic_moe_jepa_evo` |
| `route_evo_fraction` | `float` | `0.10` | Approximate fraction of training steps that run route evolution |
| `route_evo_population` | `int` | `8` | Candidate count for route evolution |
| `route_evo_mutation_scale` | `float` | `0.05` | Gaussian mutation scale for route-evolution candidates |
| `route_evo_seed` | `int \| None` | `None` | Optional deterministic route-evolution seed |
| `layer_evo_enabled` | `bool` | `False` | Train one designated transformer block by interleaved evolutionary search instead of gradients |
| `layer_evo_index` | `int \| None` | `None` | Block index trained by evolution (`None` resolves to `num_layers // 2`) |
| `layer_evo_fraction` | `float` | `0.10` | Fraction of optimizer steps that run the layer-evo search (`0.10` ≈ every 10th step) |
| `layer_evo_population` | `int` | `8` | Layer-evo candidate count; the current weights are always candidate 0 (elite) |
| `layer_evo_mutation_scale` | `float` | `0.02` | Gaussian mutation scale for layer-evo candidates |
| `layer_evo_seed` | `int \| None` | `None` | Optional deterministic layer-evo seed |
| `ar_loss_coef` | `float` | `1.0` | Autoregressive loss scale on composed/semantic objectives |
| `jepa_loss_coef` | `float` | `0.25` | JEPA latent loss scale |
| `semantic_align_loss_coef` | `float` | `0.5` | Semantic topic-alignment loss scale |
| `finetune` | `FineTuneSpec \| None` | `None` | Optional fine-tuning objective/checkpoint metadata |

---

## model_spec_to_dict

```python
def model_spec_to_dict(spec: ModelSpec) -> dict[str, Any]
```

Convert a `ModelSpec` to a plain dictionary via `dataclasses.asdict`.

---

## Preset Builder Functions

All preset builders accept `**kwargs` to override default values. Common kwargs include `model_dim` (or `n_embd`), `num_layers` (or `n_layer`), `vocab_size`, `num_heads`, `num_kv_heads`, `tie_embeddings`, `logit_softcap`, and block-specific parameters.

### `build_composed_lm_spec`

```python
def build_composed_lm_spec(
    *,
    base_model: str = "llama",
    topology: str = "dense",
    router_mode: str = "none",
    use_jepa: bool = False,
    runtime: str | None = None,
    **kwargs: Any,
) -> ModelSpec
```

Build a `ModelSpec` from the same recipe vocabulary used by the `nfn` CLI:
`base_model` is `llama`, `gpt`, `gpt2`, `gpt3`, or `nanogpt`; `gpt`, `gpt2`,
and `gpt3` all use the GPT-compatible template builder and leave architecture
selection to the chosen template or graph; `topology` is `dense` or `moe`;
`router_mode` is `standard` or `semantic` for MoE graphs; `use_jepa` overlays
the additive JEPA objective; and `runtime` can be default, `eager`, `compile`,
or `megakernel`. Pass adapter options such as `adapter_type="lora"` or
`adapter_type="qlora"` through `**kwargs`; attach `finetune=FineTuneSpec(...)`
when constructing fine-tuning graphs.

| Function | Backbone | Sparsity | Runtime | Notes |
|----------|----------|----------|---------|-------|
| `build_nanogpt_spec(**kwargs)` | nanogpt | dense | eager | LayerNorm, GELU MLP, absolute pos |
| `build_nanogpt_megakernel_spec(**kwargs)` | nanogpt | dense | megakernel | NanoGPT shape with megakernel runtime metadata |
| `build_gpt2_spec(**kwargs)` | gpt2 | dense | eager | LayerNorm, GELU MLP, absolute pos, linear bias |
| `build_gpt2_megakernel_spec(**kwargs)` | gpt2 | dense | megakernel | GPT-2 shape with megakernel runtime metadata |
| `build_gpt2_evo_spec(**kwargs)` | gpt2 | dense | eager | **[Experimental]** GPT-2 where one block (`layer_evo_index`, default middle) is excluded from the optimizer and trained by an interleaved evolutionary search (`layer_evo_*` knobs); all other parameters train by gradient. The 5090 harness `cli/scripts/train_gpt2_evo.py` is a Torch-free native shim that delegates to the compiled C++ CUDA Tile GPT-2-evo/dense-GPT path with a 12-layer SM120 AdamW run, requested NVFP4 activation intent, and live validation loss every 1000 steps. Native plan/runtime JSON also reports the effective dense-trainer activation storage; until native FP4 packing is wired into projection/attention inputs, NVFP4 requests are effective `bf16-float32-mixed` with packing inactive. Native inference uses `nfn infer --checkpoint .../gpt2_evo --prompt-tokens IDS`, while legacy graph-backed `.pt/.json` artifacts can still use `python cli/scripts/infer_gpt2.py --evo` or explicit `nfn infer --graph ... --weights ...` |
| `build_llama_spec(**kwargs)` | llama | dense | eager | RMSNorm, SwiGLU, RoPE, GQA |
| `build_mixllama_spec(**kwargs)` | mixllama | moe | eager | RMSNorm, MoE MLP, RoPE, GQA |
| `build_llama_fast_spec(**kwargs)` | llama | dense | compile | Llama with torch.compile |
| `build_llama_fast_megakernel_spec(**kwargs)` | llama | dense | megakernel | LLaMA-fast shape with fused runtime metadata |
| `build_mixllama_fast_spec(**kwargs)` | mixllama | moe | compile | MoE Llama with torch.compile |
| `build_mixllama_fast_megakernel_spec(**kwargs)` | mixllama | moe | megakernel | MoE LLaMA-fast shape with fused runtime metadata |
| `build_jamba_hybrid_spec(**kwargs)` | jamba | moe | compile | Jamba hybrid (attention + Mamba interleaved) |
| `build_ternary_b158_spec(**kwargs)` | llama | dense | compile | BitLinear ternary compression |
| `build_decoder2encoder_moe_spec(**kwargs)` | llama | moe | compile | Seq2seq with MoE decoder |
| `build_diffllama_spec(**kwargs)` | llama | dense | compile | Discrete diffusion objective |
| `build_ttt_llama_spec(**kwargs)` | ttt | dense | compile | Test-time training linear layers |
| `build_llm_jepa_spec(**kwargs)` | llama | dense | compile | JEPA self-supervised objective |
| `build_dense_jepa_evo_spec(**kwargs)` | llama | dense | compile | **[Experimental]** non-semantic AR+JEPA Evo control with dense FFNs |
| `build_moe_jepa_evo_spec(**kwargs)` | mixllama | moe | compile | **[Experimental]** non-semantic AR+JEPA Evo control with standard MoE routing |
| `build_hnet_lm_spec(**kwargs)` | hnet | dense | compile | Byte-level hierarchical tokenization |
| `build_universal_llama_spec(**kwargs)` | universal | dense | compile | Universal transformer with ACT |
| `build_llama_megakernel_spec(**kwargs)` | llama | dense | megakernel | Fused attention megakernel |
| `build_kv_pca_llama_spec(**kwargs)` | llama | dense | compile | KV cache PCA compression |
| `build_semantic_router_moe_spec(**kwargs)` | mixllama | moe | compile | **[Experimental]** AR-only semantic-router control with shared routed MoE blocks |
| `build_semantic_router_moe_megakernel_spec(**kwargs)` | mixllama | moe | megakernel | **[Experimental]** semantic-router control with megakernel runtime metadata |
| `build_jepa_semantic_hybrid_spec(**kwargs)` | llama | moe | compile | **[Experimental]** JEPA semantic hybrid with fixed dimension-to-expert routing |
| `build_jepa_semantic_hybrid_megakernel_spec(**kwargs)` | llama | moe | megakernel | **[Experimental]** JEPA semantic hybrid with megakernel runtime metadata |
| `build_semantic_dense_jepa_evo_spec(**kwargs)` | llama | dense | compile | **[Experimental]** dense Semantic JEPA Evo control with chunk planner and no expert routing |
| `build_semantic_moe_jepa_evo_spec(**kwargs)` | mixllama | moe | compile | **[Experimental]** chunk-routed Semantic MoE JEPA Evo with route evolution |

---

### `build_dense_jepa_evo_spec` [Experimental]

```python
def build_dense_jepa_evo_spec(**kwargs: Any) -> ModelSpec
```

Build a non-semantic AR+JEPA Evo control using dense LLaMA decoder blocks. The root graph uses `(tokens, targets)`, trains next-token CE plus JEPA latent alignment, and does not attach `semantic_data_source` or semantic router modules.

### `build_moe_jepa_evo_spec` [Experimental]

```python
def build_moe_jepa_evo_spec(**kwargs: Any) -> ModelSpec
```

Build the non-semantic MoE counterpart. It uses the same `(tokens, targets)` AR+JEPA contract as `dense_jepa_evo`, but swaps dense FFNs for standard MoE blocks with `experts`, `top_k`, and `router_aux_loss_coef`.

---

## [Experimental] Semantic routing fields and builders

### Additional `ModelSpec` fields [Experimental]

These fields exist on `ModelSpec` for the **[Experimental]** semantic routing presets (`semantic_router_moe`, `jepa_semantic_hybrid`, `semantic_dense_jepa_evo`, and `semantic_moe_jepa_evo`) and related graphs:

| Field [Experimental] | Type | Default | Description |
|----------------------|------|---------|-------------|
| `semantic_dim` | `int` | `NUM_SEMANTIC_DIMS` | Dimensionality of the grounded semantic vector (`NUM_VOCAB_DIMS` vocabulary dimensions + taxonomy hash slots). |
| `semantic_residual_dim` | `int` | `64` | Residual projector width used by the semantic projector module. |
| `semantic_n_lsh_tables` | `int` | `8` | Number of LSH tables for bucketing semantic vectors. |
| `semantic_n_lsh_planes` | `int` | `12` | Number of hyperplanes per table (hash width). |
| `semantic_table_path` | `str` | `""` | Optional filesystem path for persistent LSH / semantic table data (empty = in-memory default). |
| `semantic_vocab_ref` | `str` | default vocab ref | Semantic vocabulary file such as `vocab_86d_o200k.json`. |
| `route_chunk_size` | `int` | `32` | Chunk boundary interval used by `semantic_dense_jepa_evo` and `semantic_moe_jepa_evo`. |
| `semantic_shared_experts` | `int` | `2` | Always-on shared experts prepended to each selected route. |
| `semantic_free_experts` | `int` | `8` | Learned free experts appended after semantic experts. |
| `route_evo_enabled` | `bool` | `True` | Enables periodic evolutionary search over route bias/table parameters. |
| `route_evo_fraction` | `float` | `0.10` | Fraction of optimizer steps that run route evolution (`0.10` means roughly every 10th step). |
| `route_evo_population` | `int` | `8` | Number of route-evolution candidates to evaluate. |
| `route_evo_mutation_scale` | `float` | `0.05` | Noise scale used to mutate route parameters. |
| `route_evo_seed` | `int \| None` | `None` | Optional seed for reproducible route-evolution candidates. |
| `ar_loss_coef` | `float` | `1.0` | Scalar applied to the routed autoregressive loss term. |
| `jepa_loss_coef` | `float` | `0.25` | Scalar applied to the JEPA latent MSE term on JEPA semantic presets. |
| `semantic_align_loss_coef` | `float` | `0.5` | Scalar applied to the masked semantic topic cross-entropy term. |

### `build_semantic_router_moe_spec` [Experimental]

```python
def build_semantic_router_moe_spec(**kwargs: Any) -> ModelSpec
```

**[Experimental]** Factory for the `semantic_router_moe` template: AR-only MixLLaMA/MoE with vocab-topic semantic projection, LSH hashing, a fixed semantic dimension-to-expert map, and a shared externally supplied MoE route reused across every decoder block. The generated stage trains two connected losses: next-token CE and masked semantic topic cross-entropy.

**Breaking-change note [Experimental]:**
- `semantic_router_moe` requires exactly `NUM_VOCAB_DIMS` experts so the routed semantic dimensions line up one-to-one with the expert map.
- The compiled/root input contract is `(tokens, targets, sem_targets)`.

In addition to the usual LLaMA/MoE overrides (`n_layer` / `num_layers`, `n_embd` / `model_dim`, `experts`, `top_k`, etc.), this builder also recognizes:

- `rope_base` / `rope_theta` for the normal attention path
- `qk_gain_init` for attention query scaling
- `ar_loss_coef` and `semantic_align_loss_coef` for the two loss terms

**Disclaimer [Experimental]:** This builder is a research-control preset intended to isolate the semantic router hypothesis without JEPA. The graph shape and tuning knobs may change.

### `build_semantic_moe_jepa_evo_spec` [Experimental]

```python
def build_semantic_moe_jepa_evo_spec(**kwargs: Any) -> ModelSpec
```

**[Experimental]** Factory for the `semantic_moe_jepa_evo` template: MixLLaMA/MoE with dense causal attention, a chunk-level causal semantic planner, JEPA target supervision, semantic/free expert routing, and periodic route-evolution search. The generated stage trains AR CE, JEPA latent alignment, semantic topic alignment, route balance, route selection, and route distillation losses.

Builder-specific rules:

- `experts` must equal `semantic_shared_experts + NUM_VOCAB_DIMS + semantic_free_experts`.
- `top_k` selects non-shared experts; shared experts are prepended to every route.
- `route_chunk_size` controls how often the causal planner updates routes.
- `route_evo_enabled` and the `route_evo_*` fields control lightweight evolutionary search over router bias/table parameters during `TorchTrainer.train()`.

**Disclaimer [Experimental]:** This is the full architecture prototype for the semantic MoE router, not a stable production preset.

### `build_semantic_dense_jepa_evo_spec` [Experimental]

```python
def build_semantic_dense_jepa_evo_spec(**kwargs: Any) -> ModelSpec
```

**[Experimental]** Factory for the `semantic_dense_jepa_evo` template: dense LLaMA-style decoder blocks with the same chunk-level causal semantic planner and JEPA target path used by `semantic_moe_jepa_evo`. The generated stage trains AR CE, JEPA latent alignment, and semantic topic alignment losses, but does not build expert routes or run route evolution.

Builder-specific notes:

- `route_chunk_size` controls how often the causal planner summarizes prefix state.
- `experts`, `top_k`, `semantic_shared_experts`, `semantic_free_experts`, and route-evolution fields are not used by the dense decoder path.
- The compiled/root input contract is `(tokens, targets, sem_targets)`.

**Disclaimer [Experimental]:** This is a dense comparison/control preset for the Semantic JEPA Evo architecture, not a stable production preset.

### `build_jepa_semantic_hybrid_spec` [Experimental]

```python
def build_jepa_semantic_hybrid_spec(**kwargs: Any) -> ModelSpec
```

**[Experimental]** Factory for the `jepa_semantic_hybrid` template: JEPA objective with LLaMA-style blocks, MoE sparsity, `torch.compile` runtime, plus vocab-topic semantic projection, LSH hashing, a fixed dimension-to-expert router, and routed attention experts over the full hidden sequence. The generated stage trains three connected losses: AR next-token CE, JEPA latent MSE, and masked semantic topic cross-entropy.

**Breaking-change note [Experimental]:**
- `jepa_semantic_hybrid` now requires exactly `NUM_VOCAB_DIMS` experts.
- `sem_targets` are categorical topic IDs with `-100` ignore sentinels in the semantic vocabulary positions plus derived semantic hash slots, not quantized semantic vectors.

In addition to the usual LLaMA/MoE overrides (`n_layer` / `num_layers`, `n_embd` / `model_dim`, `experts`, `top_k`, etc.), this builder also recognizes:

- `rope_base` / `rope_theta` for routed expert attention
- `qk_gain_init` for expert-attention query scaling
- `ar_loss_coef`, `jepa_loss_coef`, and `semantic_align_loss_coef` for the three loss terms

**Breaking-change note [Experimental]:** The hybrid preset's compiled/root input contract is now `(tokens, targets, sem_targets)` instead of `(tokens, sem_targets)`.

**Disclaimer [Experimental]:** This builder and its `ModelSpec` fields are research prototypes and may change.
