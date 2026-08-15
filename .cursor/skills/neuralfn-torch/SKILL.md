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
from neuralfn.config import build_llama_spec
from neuralfn.inference import export_to_pt, InferenceCache
import torch

# 1. Build model graph
spec = build_llama_spec(n_layer=4, num_heads=4, n_embd=128, num_kv_heads=2)
graph = build_gpt_root_graph(name="my_llama", model_spec=spec)

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

For exact-zero strict inference, construct the FP32
`CompiledTorchGraph(..., kernel_backend="torch")` under the shared inference
policy and pass it as `InferenceCache(graph, device="cuda", compiled=compiled)`.
The strict policy covers the forward as well as argmax, disables autocast,
TF32, reduced-precision reductions, and non-math SDPA, and enables fail-closed
deterministic algorithms. Do not claim positive-temperature `top_k=1` is strict
compute; it only makes token selection greedy.
Call `prepare_inference_process_environment()` before CUDA, validate the graph
and loaded FP32 state with `validate_strict_graph_support()` and
`validate_strict_compiled_graph()`, then hold
`inference_execution(0, torch_module=torch)` around the complete generation.
The context selects either the complete hierarchical
`fp32_precision="ieee"` controls or the legacy TF32-disable controls according
to the installed PyTorch version; do not mix those API generations.

## Building graphs with presets

```python
from neuralfn import build_gpt_root_graph
from neuralfn.config import build_nanogpt_spec

spec = build_nanogpt_spec(
    n_layer=4,      # transformer layers
    num_heads=4,    # attention heads
    n_embd=128,     # model dimension
    vocab_size=256, # auto-adjusted by trainer
)
graph = build_gpt_root_graph(name="model", model_spec=spec)
```

The graph has `runtime="torch"`, `training_method="torch"`, and a populated `variant_library` with attention and MLP subgraph variants.

## Shipped preset catalog

| Preset | Architecture | Key features |
|--------|-------------|--------------|
| `nanogpt` | GPT-2 style | LayerNorm, GELU MLP, absolute position embeddings |
| `gpt2` | GPT-2 | LayerNorm, GELU MLP, absolute pos, linear bias |
| `llama` | LLaMA | RMSNorm, SwiGLU, RoPE, GQA |
| `muse_glimmer` | Muse Glimmer 30B text decoder | Asymmetric GQA, 3-local/1-global RoPE/NoPE, gated attention, centered sandwich RMSNorms, exact untied logit transform |
| `moe` / `mixllama` | LLaMA + MoE | RMSNorm, MoE MLP, RoPE, GQA |
| `llama_fast` | LLaMA + compile | Like llama with `torch.compile` |
| `mixllama_fast` | MoE + compile | Like moe with `torch.compile` |
| `jamba` | Jamba hybrid | Attention + Mamba interleaved, MoE |
| `ternary_b158` | BitNet b1.58 | Ternary {-1, 0, 1} weights |
| `seq2seq` | Encoder-decoder | Seq2Seq objective, MoE MLP |
| `diffusion` | Discrete diffusion | Diffusion objective with denoising head |
| `ttt_llama` | TTT-Linear | Test-time training attention replacement |
| `llm_jepa` | LLM-JEPA | JEPA with EMA target encoder |
| `dense_jepa_evo` | Dense JEPA Evo | Non-semantic AR+JEPA control with dense FFNs |
| `moe_jepa_evo` | MoE JEPA Evo | Non-semantic AR+JEPA control with standard MoE routing |
| `semantic_router_moe` | Semantic Router MoE | AR-only semantic router control with shared routed MoE blocks |
| `semantic_dense_jepa_evo` | Semantic Dense JEPA Evo | Chunk-level semantic planner, dense FFNs, JEPA supervision, no route evolution |
| `semantic_moe_jepa_evo` | Semantic MoE JEPA Evo | Chunk-level semantic router, shared/semantic/free experts, JEPA supervision, route evolution |
| `hnet_lm` | H-Net | Raw byte input, byte patch embedding |
| `universal_llama` | Universal TX | ACT-based adaptive recurrence |
| `llama_megakernel` | Fused LLaMA | FusedCausalAttention, max-autotune compile |
| `kv_pca_llama` | PCA KV cache | PCA-compressed keys/values |
| `deepseek_v3` | DeepSeek-V3 | MLA + auxfree-balanced MoE + shared experts |
| `deepseek_v4` | DeepSeek-V4-Pro | NSA attention + auxfree MoE + mHC residuals + QK-norm + FP8 |

Muse Glimmer additionally exposes separate
`build_muse_glimmer_assistant_graph()`,
`build_muse_glimmer_dflash_distillation_graph()`,
`build_muse_glimmer_vision_graph()`, and media-fusion builders. Keep them
separate from the ordinary autoregressive root. DFlash distillation must use
`MuseGlimmerDFlashDistillationSpec` plus `DFlashDistillationTrainer`; the target
is frozen and complete target/config/tokenizer/ATEM lineage is mandatory. Its
SFT, LoRA/NF4-QLoRA, DPO, reward, and PPO wrappers must all use the shared exact
Glimmer body; never rebuild a generic two-norm LLaMA body inside a fine-tuning
root. Native K-Quant, DFlash and CUDA vision execution are independently
artifact/binding-gated, not capabilities the Torch preset may infer.
| `gemma3` | Gemma-2/3 | Sliding-window attention + GeGLU + QK-norm + softcap |

For MoE presets, `mlp_multiplier` remains floating-point through graph
serialization and both Torch/Tile module builders. Expert width is
`max(1, int(model_dim * mlp_multiplier))`, rounded upward only when
`multiple_of > 0`; `None` and `0` mean unaligned. The shipped standard-MoE
presets default to unaligned `8/3`. Do not load a checkpoint created under the
former integer-truncated expert-width behavior without rebuilding or remapping
its `w1`, `w2`, and `w3` tensors.

The exact native graph adapter currently covers only `moe`, `mixllama`, and
`mixllama_fast`. It maps unaligned graph width to `--multiple-of 0`, carries the
configured all-expert softmax router auxiliary loss, and requires a strict
source-SHA/tensor-table checkpoint before resident inference. Do not infer the
same contract for modern, megakernel, DeepSeek, aux-free, shared-expert, or JEPA
variants.
| `diff_transformer` | Differential TX | Two-softmax differential attention + head-wise norm |
| `qwen3_longctx` | Qwen long-ctx | GQA + YaRN RoPE scaling + QK-norm |
| `longctx_sparse_llama` | Long-ctx sparse | NSA / block-sparse / sliding-window / streaming |
| `modern_norms_llama` | Modern norms | DyT + QK-norm + GeGLU |
| `fp8_llama` / `mxfp4_llama` | Blackwell precision | FP8 E4M3 / MXFP4 microscaled weight linears |
| `auxfree_moe_jepa_evo`, `diff_semantic_moe_jepa_evo`, `dyt_geglu_semantic_dense_jepa_evo` | NeuralFn crosses | Modern kernels × JEPA/semantic/route-evo stacks |
| `<preset>_modern` | Modernized | Any base preset + RMSNorm/QK-norm/RoPE-YaRN/GeGLU/auxfree (see `MODERN_BASE_PRESETS`) |

Treat `longctx_sparse_llama` as recompute-only sparse attention until a
resident adapter with physical eviction is proved. Tile sparse attention has a
1024-key launch ceiling: strict Python rejects longer prefixes and the raw
float32 sparse forward/backward C ABIs return `cudaErrorInvalidValue` before
launch. Do not describe right-aligned mask/recompute parity as a resident cache.

## Common config keys

| Key | Default | Description |
|-----|---------|-------------|
| `n_layer` / `num_layers` | 4 | Transformer layers |
| `n_head` / `num_heads` | 4 | Attention heads |
| `n_embd` / `model_dim` | 128 | Model dimension |
| `vocab_size` | 256 | Vocabulary (auto-adjusted by trainer) |
| `num_kv_heads` | 2 | GQA key/value heads |
| `mlp_multiplier` | 8/3 (llama) or 4 (gpt2) | MLP hidden multiplier; fractional SwiGLU values are preserved |
| `multiple_of` | 256 | Round the computed MLP width up to this multiple |
| `experts` | 8 | MoE: number of experts |
| `top_k` | 2 | MoE: experts per token |
| `dropout_p` | 0.0 or 0.1 | Dropout rate |
| `tie_embeddings` | varies | Tie embedding/LM head weights |
| `logit_softcap` | 0.0 | Tanh softcap (>0 enables) |
| `ttt_hidden_dim` | 32 | TTT hidden dimension |
| `byte_patch_size` | 4 | H-Net byte patch size |
| `max_recurrence_steps` | 4 | Universal TX max steps |

For LLaMA-style template graphs, Torch and CUDA Tile instantiate affine
RMSNorm whenever the node config contains `model_dim`; the learnable float32
scale starts at ones. A legacy node without `model_dim` remains parameter-free.
SwiGLU computes `int(model_dim * mlp_mult)` and then rounds up to `multiple_of`,
so do not coerce fractional multipliers to integers. Older template checkpoints
need one-valued norm weights added, and custom non-`8/3` SwiGLU checkpoints may
need their three projection tensors remapped before strict loading.

## Programmatic spec building

```python
from neuralfn.config import build_llama_spec, ModelSpec
from neuralfn.torch_templates import build_model_stage_graph, build_gpt_template_payload

# Build a ModelSpec directly
spec = build_llama_spec(n_layer=6, n_embd=256, num_heads=8, num_kv_heads=4)

# Build just the model stage subgraph
stage_graph = build_model_stage_graph("model_stage", spec)

# Build a full payload for server/editor template APIs
payload = build_gpt_template_payload("my_model", {"preset": "llama", "n_layer": 6, "n_embd": 256})
```

Spec builders: `build_nanogpt_spec`, `build_nanogpt_megakernel_spec`, `build_gpt2_spec`, `build_gpt2_megakernel_spec`, `build_llama_spec`, `build_mixllama_spec`, `build_llama_fast_spec`, `build_llama_fast_megakernel_spec`, `build_mixllama_fast_spec`, `build_mixllama_fast_megakernel_spec`, `build_jamba_hybrid_spec`, `build_ternary_b158_spec`, `build_decoder2encoder_moe_spec`, `build_diffllama_spec`, `build_ttt_llama_spec`, `build_llm_jepa_spec`, `build_dense_jepa_evo_spec`, `build_moe_jepa_evo_spec`, `build_semantic_router_moe_spec`, `build_semantic_router_moe_megakernel_spec`, `build_jepa_semantic_hybrid_spec`, `build_jepa_semantic_hybrid_megakernel_spec`, `build_semantic_dense_jepa_evo_spec`, `build_semantic_moe_jepa_evo_spec`, `build_hnet_lm_spec`, `build_universal_llama_spec`, `build_llama_megakernel_spec`, `build_kv_pca_llama_spec`, and `build_composed_lm_spec`. Frontier builders: `build_deepseek_v3_spec`, `build_deepseek_v4_spec`, `build_gemma3_spec`, `build_diff_transformer_spec`, `build_qwen3_longctx_spec`, `build_longctx_sparse_llama_spec`, `build_modern_norms_llama_spec`, `build_fp8_llama_spec`, `build_mxfp4_llama_spec`, `build_auxfree_moe_jepa_evo_spec`, `build_diff_semantic_moe_jepa_evo_spec`, `build_dyt_geglu_semantic_dense_jepa_evo_spec`. Modernized variants are generated as `<preset>_modern` (dispatch strips the suffix and applies `_apply_modern_profile`).

## TorchTrainConfig

| Field | Default | Description |
|-------|---------|-------------|
| `learning_rate` | 3e-4 | Adam learning rate |
| `epochs` | 50 | Training epochs |
| `batch_size` | 8 | Batch size |
| `weight_decay` | 0.01 | AdamW weight decay |
| `device` | "cuda" | Device ("cuda", "cpu") |
| `amp_dtype` | "float32" | AMP dtype; float32 disables autocast |
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
- Semantic routing presets: `tokens`, `targets`, plus `semantic_data_source -> sem_targets`

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

## Migrate graph and `.pt` weights to Native Execution IR

Use the versioned Native IR migration boundary instead of treating `.pt` plus
graph JSON as a native runtime artifact:

```python
from neuralfn import migrate_graph_to_native

result = migrate_graph_to_native(
    "model.json",
    weights_path="model.pt",
    output_dir="model-native",
    dry_run=False,
)
assert result.report.compatible
```

The graph is validated and all variants are resolved before the checkpoint is
opened. `.pt` loading happens only in an isolated `weights_only` worker, and
the destination must not exist. The source graph/checkpoint are unchanged.
Structural lowering covers all 67 shipped text presets. Trainer registration
does not prove execution: NanoGPT's bias-free/dropout graph is not represented
by the biased, dropout-free dense-v5 loop, so all NanoGPT selectors retain
false persistence/forward/resident gates. Canonical LLaMA is separately
promoted by its exact topology/checkpoint proof. A
successful migration proves resident sessions, retained lossless
K/V, and lean serving only when one of the seven reviewed dense preset topologies
(`gpt2`, megakernel, MoA, z-loss, QK-norm, stable, or softcap) is paired with a
compatible native dense-v5 `.bin`; exact QK/softcap nodes/configs are required,
and supported even head geometry also proves native packed CPU TurboQuant cache
ABI v1. MoA specifically requires migration of source-bound
`model_XXXXXXXX.moa.json` beside its named dense-v5 model and empty DONE marker;
the metadata fixes canonical candidates, selected activation, and positive
interval for CPU resident inference. Resume validates that metadata and
restores its activation without probing; missing or tampered metadata fails
closed. This is graph-bound: direct selector-only MoA first-leg training still
emits ordinary dense-v5, but an unbound resume is not exact and must fail.
Canonical `llama` is a separate proved path when the graph is paired
with native-family inference-checkpoint v2 metadata: migration validates and
copies `model.f32`, and its resident adapter implements lossless
RMSNorm/RoPE/GQA/SwiGLU inference with TurboQuant rejected. Generic `.pt`,
graph-only, bare MoA `.bin` files, differential/modern variants, and other
non-dense families remain closed. Compatible reviewed-dense artifacts now
advertise the separately versioned Tile-sidecar attention feature; explicit
strict-sidecar configuration moves only packed historical attention to CUDA,
with CPU still owning model compute and encoding. See
`docs/python-sdk/native-ir.md`.

For training orchestration, use `plan_native_graph_training()` to obtain the
same compatibility report and canonical target. Launch only when
`execution_ready` and `graph_preflight_enforced` are true. The seven reviewed
GPT-2 graph profiles plus exact canonical `llama`, its graph-equivalent
compile-runtime alias `llama_fast`, and exact standard-MoE
`moe`/`mixllama`/`mixllama_fast`, plus proof-bound `gpt2_diff` training, make
13 ready profiles and use an immutable validated
graph snapshot plus canonical selector/geometry. LLaMA requires the proved
RMSNorm/RoPE/MHA-or-GQA/dense-attention/gate-first-SwiGLU contract and binds
the source SHA through training, checkpoint discovery, and migration. Both
LLaMA source profiles use native identity `llama` and retain their source
selector/runtime in provenance. Because
the current executables do not parse Native IR, these ready plans honestly
report `trainer_consumes_native_ir` false. Unsupported
profiles must fail with their node-specific issue and must not route to a
diagnostic transition sampler.
`gpt2_diff` graph training is allowed only through the trusted planner's
canonical source/configuration/topology/shape/geometry proof. The native child
requires graph, fingerprint, and proof before plan/Tile/CUDA work; its unkeyed
digest is local-handoff integrity, not authenticity. The low-level packed trainer writes learned per-layer
lambda/moments in an additive graph-bound bundle with version-2 strict
continuation metadata over the source, five binaries, ordered training shards,
counters/sampler, seed, accumulation, optimizer/LR horizon, BF16 routes, and
a canonical profile of supported effective numerics. Validation shards are excluded, and resume
preflight finishes before Tile/CUDA/H2D. This does not provide a Torch fallback:
the learned path requires the learned-lambda ABI, while the legacy fixed-lambda
ABI retains rounded-output/non-layer-local backward debt. Migration and resident
inference still do not validate or execute the bundle.

## InferenceCache (autoregressive generation)

```python
from neuralfn.inference import InferenceCache
import torch

cache = InferenceCache(graph, device="cuda")
prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
logits = cache.step(prompt)         # full prompt
next_tok = logits.argmax(dim=-1)
prefix = torch.cat((prompt, next_tok.unsqueeze(1)), dim=1)
logits2 = cache.step(prefix)         # current wrapper recomputes the full prefix
cache.reset()                        # clears the reserved compatibility state
```

Despite its historical name, the current `InferenceCache` does not populate or
read retained K/V state across calls. Graphs may contain `kv_cache_read` /
`kv_cache_write` declarations, but the wrapper does not bind them to a
lossless cache. Pass the full prefix for context-correct generation. For
training graphs (2 inputs), dummy targets are supplied automatically. This API
is separate from the additive `neuralfn.native_inference` model/session
contract. That Python contract is implemented, but current artifacts do not
all prove a production resident binding or lossless cache. The seven reviewed
dense-v5 preset topologies (`gpt2`, `gpt2_megakernel`, `gpt2_moa`, `gpt2_zloss`,
`gpt2_qknorm`, `gpt2_stable`, and `gpt2_softcap`) do, with compiled
cache/recompute parity tests. Canonical LLaMA does as well when backed by the
exact v2 float32 checkpoint contract; its raw-token CLI/SDK route is proved,
but its text-serving presentation metadata and TurboQuant are not. Generic
`.pt`, graph-only, bare MoA `.bin` files, differential/modern variants, and other non-dense
artifacts do not. See
`docs/python-sdk/native-inference.md`.

The portable `neuralfn.turboquant` module is the correctness oracle and shared
table source for the exact-dense native packed CPU resident cache. It does not
change the Torch/Tile graph `kv_quant_pack` or `kv_quant_unpack` dtype/shape
contract. Its records now have an independently proved additive CUDA attention
feature ABI in the trainer Tile sidecar. Explicit reviewed-dense resident
dispatch is live-proved for MSE/QJL; non-dense cache support and
transfer-inclusive performance remain unproved.

## Experimental Presets

### `semantic_router_moe` [Experimental]

- **Preset:** `semantic_router_moe` [Experimental]
- **Load in Python:** `from neuralfn.config import build_semantic_router_moe_spec`; then `spec = build_semantic_router_moe_spec(**kwargs)` and `build_gpt_root_graph(name=..., model_spec=spec)`.
- **Load via MCP / server:** `load_gpt_template(name=..., preset="semantic_router_moe", config={...})` [Experimental].
- **What it does [Experimental]:** AR-only MixLLaMA/MoE control preset that computes a vocab-grounded semantic route once from the pre-block hidden state, hashes it, teacher-forces/auto-selects one expert per semantic vocabulary dimension, broadcasts that route across the whole sequence, and applies it to every MoE block. Trains next-token CE plus semantic-alignment loss, with no JEPA encoder/EMA path.
- **Disclaimer [Experimental]:** Research-control preset only. It exists to isolate the router hypothesis before adding JEPA complexity.

### `semantic_moe_jepa_evo` [Experimental]

- **Preset:** `semantic_moe_jepa_evo` [Experimental]
- **Load in Python:** `from neuralfn.config import build_semantic_moe_jepa_evo_spec`; then `spec = build_semantic_moe_jepa_evo_spec(**kwargs)` and `build_gpt_root_graph(name=..., model_spec=spec)`.
- **Load via MCP / server:** `load_gpt_template(name=..., preset="semantic_moe_jepa_evo", config={...})` [Experimental].
- **What it does [Experimental]:** Full Semantic MoE JEPA Evo architecture. Dense causal attention stays on the AR path; a prefix-safe chunk planner predicts semantic latents and route distributions; routes combine always-on shared experts, semantic-vocabulary experts, and free learned experts; and the trainer can periodically evolve route bias/table state.
- **Config rules [Experimental]:** `experts` must equal `semantic_shared_experts + NUM_VOCAB_DIMS + semantic_free_experts`. Defaults are `route_chunk_size=32`, `semantic_shared_experts=2`, `semantic_free_experts=8`, `route_evo_fraction=0.10`, and `route_evo_population=8`.
- **Disclaimer [Experimental]:** Research prototype only. Graph shape, loss mix, and route-evolution behavior may change.

### `semantic_dense_jepa_evo` [Experimental]

- **Preset:** `semantic_dense_jepa_evo` [Experimental]
- **Load in Python:** `from neuralfn.config import build_semantic_dense_jepa_evo_spec`; then `spec = build_semantic_dense_jepa_evo_spec(**kwargs)` and `build_gpt_root_graph(name=..., model_spec=spec)`.
- **Load via MCP / server:** `load_gpt_template(name=..., preset="semantic_dense_jepa_evo", config={...})` [Experimental].
- **What it does [Experimental]:** Dense control for the Semantic JEPA Evo architecture. It keeps the prefix-safe chunk planner, JEPA target supervision, AR CE, JEPA latent alignment, and semantic-alignment losses, but uses dense LLaMA FFNs with no expert dispatch, route losses, or route-evolution loop.
- **Config rules [Experimental]:** `route_chunk_size` controls planner chunk boundaries. Expert-count and route-evolution fields are ignored by the dense decoder path.
- **Disclaimer [Experimental]:** Dense comparison/control preset only. Graph layout and tuning knobs may change.

### `jepa_semantic_hybrid` [Experimental]

- **Preset:** `jepa_semantic_hybrid` [Experimental]
- **Load in Python:** `from neuralfn.config import build_jepa_semantic_hybrid_spec`; then `spec = build_jepa_semantic_hybrid_spec(**kwargs)` and `build_gpt_root_graph(name=..., model_spec=spec)`.
- **Load via MCP / server:** `load_gpt_template(name=..., preset="jepa_semantic_hybrid", config={...})` [Experimental].
- **What it does [Experimental]:** Joint Embedding Predictive Architecture (JEPA) combined with a vocab-grounded semantic state, **LSH** bucketing, a fixed dimension-to-expert semantic router, and routed full-sequence attention experts. `sem_targets` are categorical topic IDs with ignore sentinels, not quantized semantic vectors.
- **Disclaimer [Experimental]:** Research prototype only—APIs, graph shape, and training behavior may change without notice.
