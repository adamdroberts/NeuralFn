# Training Workflows

NeuralFn supports four training methods, each suited to different graph types. The `training_method` field on a `NeuronGraph` determines which trainer applies.

| `training_method` | Runtime | Trainer class | Graph type |
|-------------------|---------|--------------|------------|
| `"surrogate"` | `"scalar"` | `SurrogateTrainer` | Function neurons with scalar I/O |
| `"evolutionary"` | `"scalar"` | `EvolutionaryTrainer` | Function neurons with scalar I/O |
| `"frozen"` | either | (none) | Subgraphs that should not be trained |
| `"torch"` | `"torch"` | `TorchTrainer` | Module neurons with tensor I/O |

For hierarchical graphs with mixed training methods, `HybridTrainer` orchestrates training across subgraph boundaries.

Default CLI training now requires a compiled native CUDA/C++ entrypoint. Dense
GPT has that path through `nfn train --base-model gpt`; `gpt2` and `gpt3` are
aliases for the same native trainer, and `gpt3` only changes the default
context window to 2048 when no explicit template, graph, or `--train-seq-len`
is supplied. Other graph-backed `TorchTrainer` harnesses are disabled before
Torch import, and the old `NFN_ALLOW_TORCH_TRAINING` CLI bypass is ignored.
Legacy graph-backed experiments should call the Python SDK trainer APIs
directly while native trainers are being added.

GPT template selection is explicit on the native path. The default public
template alias is `gpt`, which currently resolves to the implemented dense GPT
native topology and is reported separately as `resolved_native_template_name` in
compiled JSON. Pass `--template-name NAME` / `--preset NAME` to select any name
in `neuralfn.config.SHIPPED_GPT_TEMPLATE_PRESETS`, or `--graph-file PATH` for a
custom graph JSON. `gpt`, `gpt2`, and `gpt2_megakernel`, plus `gpt2_moa` with
native MoA activation, use the implemented compiled CUDA Tile trainer; unsupported
templates and graph files fail fast with native missing-trainer JSON instead of
falling back to Torch.

Dense GPT native transformer training now fuses token embedding, absolute
position embedding, and the scaled embedding residual add in the raw Tile-CUDA
ABI. The default direct-u16 token path uses
`nfn_native_tile_token_position_embedding_residual_u16_float32`, reports
`embedding_residual_fusion_enabled: true`, and elides the separate `token_out`
and `position_out` FP32 activation buffers from the startup arena. Set
`NFN_NATIVE_GPT_FUSE_EMBEDDING_RESIDUAL=0` only for paired diagnostics against
the older three-launch embedding path.

---

## 1. Surrogate training

Surrogate training builds a differentiable neural-network approximation (the "surrogate") of each neuron's behavior, then uses gradient descent on the surrogate to optimize the graph's edge weights and biases.

```python
from neuralfn import NeuronGraph, SurrogateTrainer
from neuralfn.trainer import TrainConfig
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

graph = build_xor_graph()  # see building-graphs.md

trainer = SurrogateTrainer(graph, TrainConfig(epochs=300, learning_rate=0.01))
losses = trainer.train(
    X, Y,
    on_epoch=lambda ep, loss: print(f"epoch {ep}: {loss:.6f}"),
)
print(f"Final loss: {losses[-1]:.6f}")
```

### TrainConfig fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `learning_rate` | `float` | `0.001` | Gradient descent step size. |
| `epochs` | `int` | `500` | Number of training epochs. |
| `batch_size` | `int` | `32` | Mini-batch size. |
| `surrogate_samples` | `int` | `10000` | Number of random samples used to fit each neuron's surrogate. |
| `surrogate_hidden` | `tuple` | `(64, 64)` | Hidden layer sizes of the surrogate network. |
| `surrogate_epochs` | `int` | `200` | Epochs to train each surrogate. |
| `loss_fn` | `str` | `"mse"` | Loss function: `"mse"` or `"bce"`. |

---

## 2. Evolutionary training

A population-based optimizer that mutates edge parameters and selects for lower loss. No gradients or surrogates required.

```python
from neuralfn import EvolutionaryTrainer
from neuralfn.evolutionary import EvoConfig

evo = EvolutionaryTrainer(graph, EvoConfig(population_size=40, generations=100))
losses = evo.train(
    X, Y,
    on_generation=lambda gen, loss: print(f"gen {gen}: {loss:.6f}"),
)
print(f"Final loss: {losses[-1]:.6f}")
```

### EvoConfig fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `population_size` | `int` | `50` | Number of individuals per generation. |
| `generations` | `int` | `200` | Number of generations to evolve. |
| `mutation_rate` | `float` | `0.1` | Probability of mutating each parameter. |
| `mutation_scale` | `float` | `0.3` | Standard deviation of Gaussian mutation noise. |
| `crossover_rate` | `float` | `0.5` | Probability of crossover between parents. |
| `tournament_size` | `int` | `3` | Number of candidates in tournament selection. |
| `elite_count` | `int` | `2` | Number of top individuals copied unchanged to the next generation. |
| `topology_mutations` | `bool` | `False` | Whether to allow structural mutations (add/remove edges). |
| `seed` | `int` | `None` | Random seed for reproducibility. |

---

## 3. Hybrid training

For graphs that contain subgraphs with different training methods, `HybridTrainer` runs the appropriate trainer on each sub-graph in rounds:

```python
from neuralfn import HybridTrainer, HybridConfig

trainer = HybridTrainer(root_graph, HybridConfig(outer_rounds=3))
losses = trainer.train(
    X, Y,
    on_step=lambda info: print(info),
)
```

Each child graph's `training_method` determines how it is trained:
- `"surrogate"` subgraphs use `SurrogateTrainer` internally.
- `"evolutionary"` subgraphs use `EvolutionaryTrainer`.
- `"frozen"` subgraphs are skipped entirely.

### HybridConfig fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `outer_rounds` | `int` | `3` | Number of full passes over all subgraphs. |
| `loss_fn` | `str` | `"mse"` | Loss function applied at the root level. |
| `default_surrogate` | `TrainConfig` | default `TrainConfig()` | Fallback surrogate config for subgraphs that do not specify their own. |
| `default_evolutionary` | `EvoConfig` | default `EvoConfig()` | Fallback evolutionary config. |

---

## 4. Torch training

For tensor-native graphs built from module neurons (or generated by the template system):

```python
from neuralfn import TorchTrainer, TorchTrainConfig, build_gpt_root_graph
from neuralfn.config import build_nanogpt_spec
import torch

spec = build_nanogpt_spec(n_layer=4, n_embd=128, vocab_size=256)
graph = build_gpt_root_graph(model_spec=spec)

tokens = torch.randint(0, 256, (8, 64))
targets = torch.randint(0, 256, (8, 64))

trainer = TorchTrainer(graph, TorchTrainConfig(
    epochs=10,
    learning_rate=5e-3,
    batch_size=2,
    device="cuda",
))

def log_progress(info: dict[str, object]) -> None:
    if info["phase"] == "warmup":
        print(f"warmup {info['step']}/{info['warmup_steps']} loss={info['loss']:.4f}")
        return
    if info["step"] % 50 == 0:
        print(
            f"step {info['step']}/{info['max_steps']} "
            f"epoch {info['epoch']}/{info['max_epochs']} "
            f"loss={info['loss']:.4f}"
        )

losses = trainer.train(tokens, targets, on_step=log_progress)
print(f"Final loss: {losses[-1]:.4f}")
```

### TorchTrainConfig fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `learning_rate` | `float` | `3e-4` | AdamW learning rate. |
| `epochs` | `int` | `50` | Training epochs. |
| `batch_size` | `int` | `8` | Batch size. |
| `weight_decay` | `float` | `0.01` | AdamW weight decay. |
| `device` | `str` | `"cuda"` | Device to train on. |
| `amp_dtype` | `str` | `"float32"` | Automatic mixed precision dtype. `"float32"` disables AMP; `"bfloat16"` and `"float16"` enable autocast. |
| `compile` | `bool` | `False` | Whether to `torch.compile` the model (overridden by `TemplateSpec.runtime`). |
| `activation_checkpointing` | `bool` | `False` | Enable gradient checkpointing to save memory. |
| `fsdp2_enabled` | `bool` | `False` | Enable FSDP2 for multi-GPU data parallelism. |
| `max_steps` | `int` | `None` | Stop after this many steps regardless of epochs. `None` = unlimited. |
| `respect_epoch_boundaries` | `bool` | `False` | Keeps epochs aligned to one loader pass and allows a short final accumulation step instead of cycling into the next epoch. |
| `kernel_backend`, `tile_cuda_strict`, `tile_cuda_report_path` | str/bool/str | `"auto"`, `False`, `None` | Optional CUDA Tile backend selection and reporting fields. The current Tile registry accounts for all 138 training-relevant entries with 129 Tile-covered kernels/compositions, 7 host-only entries, and 2 delegated graph calls; PyTorch remains the fallback for unsupported tensor contracts unless strict mode is enabled. |
| `optimizer_profile` | `str` | `"adamw"` | `"adamw"` for the single-optimizer path; when `kernel_backend="tile_cuda"` it uses batched CUDA Tile AdamW steps, Tile gradient clipping, and default cosine decay to zero when no explicit LR schedule is supplied. Use `"parameter_golf"` only for split optimizers + Muon. |
| `train_batch_tokens` | `int \| None` | `None` | Token budget per optimizer step. Enables gradient accumulation by token count instead of raw batch count. |
| `beta1`, `beta2`, `adam_eps` | floats | `0.9`, `0.95`, `1e-8` | Adam-family optimizer hyperparameters. |
| `embed_lr`, `head_lr`, `tied_embed_lr`, `matrix_lr`, `scalar_lr` | floats / `None` | `None` | Optional split learning rates for the parameter-golf profile. |
| `muon_momentum`, `muon_backend_steps`, `muon_momentum_warmup_start`, `muon_momentum_warmup_steps` | float/int | `0.95`, `5`, `0.85`, `500` | Muon optimizer controls for matrix-shaped parameters. |
| `warmup_steps`, `warmdown_fraction`, `lr_decay_iters`, `min_lr`, `max_wallclock_seconds` | int/float | `0`, `0.75`, `None`, `None`, `0.0` | Schedule controls for warmup priming, fractional tail warmdown, explicit cosine LR decay, LR floor, and wallclock cutoffs. `warmdown_fraction` controls the final share of optimizer steps used for linear warmdown. When `lr_decay_iters` is set, cosine decay overrides `warmdown_fraction`; omitting `min_lr` while cosine decay is enabled uses `learning_rate / 10`. On Tile CUDA `adamw` runs, missing `lr_decay_iters` defaults to the resolved training step count and missing `min_lr` defaults to `0.0`. Native GPT wrapper scripts default to 600 warmup steps for quality runs and still honor explicit `--warmup-steps` or `WARMUP_STEPS`. `max_wallclock_seconds` only stops training early; it does not change the LR schedule. |
| `grad_clip_norm` | `float` | `0.0` | Global grad clipping threshold. |

`TorchTrainer` automatically adjusts `vocab_size` when the training data's token range exceeds the graph's configured vocabulary, ensuring the embedding and output layers are compatible.

Torch-runtime training compiles graph topology, input/output layout, and edge routing into a static execution plan before batches run. Real training tensors flow through fixed child modules and the precomputed plan; they do not pass through graph editor node objects, canvas positions, viewport state, or mutable editor metadata. CUDA Tile execution plans must preserve the same control-plane/data-plane split.

For Torch-free native GPT launchers, `neuralfn.native_train.build_native_train_run_config()` can enforce the dense-GPT strict LM-head parity guard with `require_cooperative_lm_head_backward=True`. The SDK appends `--require-cooperative-lm-head-backward` once, rejects non-dense family targets, and keeps the handoff in the compiled native frontend without importing Torch. Current CUDA Tile builds still fail that guard because the LM-head backward route is a diagnostic CUDA Graph wrapper rather than the future fused classifier/dHidden/dWeight kernel.

For dense GPT startup/preflight probes, pass `fast_startup=True` to the same native SDK helper to append `--fast-startup` once. This skips throughput-only setup prewarms through the native prewarm policy without requiring environment variables; normal training defaults remain unchanged.

For long CUDA runs, `on_step` is usually the right hook for live CLI progress because it fires once per warmup step and once per optimizer step instead of waiting for epoch boundaries.

For the experimental semantic routing presets, dataset-backed training resolves a three-role flat input contract: `(tokens, targets, sem_targets)`. `semantic_router_moe` uses that contract for an AR-only router-control experiment, `jepa_semantic_hybrid` adds JEPA loss on top of the same routed branch, `semantic_dense_jepa_evo` keeps the chunk-level semantic planner with dense FFNs, and `semantic_moe_jepa_evo` routes at chunk granularity with a shared + semantic + free expert bank. `semantic_data_source` generates categorical vocab-topic targets from the active semantic vocabulary reference; inactive dimensions use `-100` ignore sentinels, and the first `NUM_VOCAB_DIMS` positions line up with the semantic expert map. When only semantic data is available, the trainer synthesizes safe placeholder `tokens` / `targets` tensors instead of feeding `sem_targets` into the embedding path.

`semantic_moe_jepa_evo` also enables a lightweight route-evolution controller during normal torch training. After optimizer steps selected by `route_evo_fraction`, `TorchTrainer` evaluates a small candidate population over recent macro-batches and writes the best candidate back to the router's route-only parameters. This does not replace gradient training for the backbone; it only tunes route bias/table state where the semantic router benefits from search.

### Fine-tuning objectives

The torch template builders also include fine-tuning root graphs:

| Objective | Graph contract | Use case |
|-----------|----------------|----------|
| `sft` | `sft_dataset_source -> tokens, targets, loss_mask` | Supervised fine-tuning with prompt masking. |
| `dpo` | `dpo_dataset_source -> chosen/rejected tokens, targets, masks` | Direct Preference Optimization with policy/reference log-probabilities. |
| `ppo` | `ppo_rollout_source -> rollout tensors` | PPO inner-loop updates orchestrated by `PPOTrainer`. |
| `reward_model` | `dpo_dataset_source -> chosen/rejected pairs` | Preference reward head training. |

Use `FineTuneSpec` on `ModelSpec.finetune` and set
`model_spec.template.objective` to the chosen objective before calling
`build_gpt_root_graph()`. The CLI exposes this path through
`nfn train --training-mode sft|dpo|ppo|reward_model` and adapter flags such as
`--adapter-type lora`, `--adapter-type qlora`, and `--adapter-only-save`.

---

## How training_method, runtime, and trainers relate

The relationship is straightforward:

- A graph with `runtime="scalar"` holds function neurons and uses scalar `execute()`. Set `training_method` to `"surrogate"` or `"evolutionary"` and use the corresponding trainer.
- A graph with `runtime="torch"` holds module neurons and compiles to a PyTorch module. Set `training_method="torch"` and use `TorchTrainer`.
- A graph with `training_method="frozen"` is never trained, regardless of runtime. It acts as a fixed-function block.
- `HybridTrainer` handles the case where a root graph contains subgraphs with different `training_method` values.

For torch graphs that read from a tokenizer-backed cached `dataset_source`, `TorchTrainer` now validates the cached shard ids and tokenizer artifacts before training starts. Manual tensors and tokenizer-less datasets can still trigger the old vocab auto-expand path when needed, but cached aliases that advertise tokenizer artifacts now fail fast if their shard ids or graph vocab disagree with that tokenizer contract.

---

Next: [Inference and Export](inference-and-export.md)
