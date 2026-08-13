# Training Workflows

NeuralFn supports four training methods, each suited to different graph types. The `training_method` field on a `NeuronGraph` determines which trainer applies.

| `training_method` | Runtime | Trainer class | Graph type |
|-------------------|---------|--------------|------------|
| `"surrogate"` | `"scalar"` | `SurrogateTrainer` | Function neurons with scalar I/O |
| `"evolutionary"` | `"scalar"` | `EvolutionaryTrainer` | Function neurons with scalar I/O |
| `"frozen"` | either | (none) | Subgraphs that should not be trained |
| `"torch"` | `"torch"` | `TorchTrainer` | Module neurons with tensor I/O |

For hierarchical graphs with mixed training methods, `HybridTrainer` orchestrates training across subgraph boundaries.

The scalar trainer modules are import-safe on the lean native/core SDK:
importing `TrainConfig`, `EvoConfig`, `SurrogateTrainer`, or
`EvolutionaryTrainer` and constructing scalar or hybrid trainer instances does
not load NumPy or Torch. Actual scalar training still needs the legacy
numerical stack: surrogate training requires NumPy plus PyTorch, evolutionary
training requires NumPy, and hybrid training inherits those requirements from
the scopes it executes.

Default CLI training now requires a compiled native CUDA/C++ entrypoint. Dense
GPT has that path through `nfn train --base-model gpt`; `gpt2` and `gpt3` are
aliases for the same native trainer, and `gpt3` only changes the default
context window to 2048 when no explicit template, graph, or `--train-seq-len`
is supplied. Other graph-backed `TorchTrainer` harnesses are disabled before
Torch import, and the old `NFN_ALLOW_TORCH_TRAINING` CLI bypass is ignored.
Legacy graph-backed experiments should call the Python SDK trainer APIs
directly while native trainers are being added.

The `nfn train` dense-GPT workflow checks build freshness before launching. It
uses only `build/nfn_gpt_native_train_linked` and never automatically falls
back to the generic or compatibility frontends. Its build writes a SHA-256
manifest for the executable, trainer, preset and Tile inputs, Tile library, and
build scripts. Missing or changed hashed inputs mark it stale without relying
on mtimes. An interactive run then warns and offers a forced linked rebuild.
Non-interactive runs stop with the rebuild command. Set
`NFN_NATIVE_GPT_AUTO_REBUILD=1` for opt-in automatic repair. Dense-GPT
`nfn train` ignores native CLI environment overrides; use a lower-level SDK or
compatibility entrypoint when intentionally testing a separate artifact.

The `nfn train` TUI does not collapse native metric lines. It preserves every
live key/value field and formats it into labeled segments. Once the native
trainer returns, the TUI keeps its concise outcome card but follows it with a
recursive dotted-path view of every result field, including loss histories,
timing, geometry, kernel/fallback counters, memory diagnostics, and checkpoint
metadata. The output is human-readable rather than a raw JSON dump.

GPT template selection is explicit on the native path. The default public
template alias is `gpt`, which currently resolves to the implemented dense GPT
native topology and is reported separately as `resolved_native_template_name` in
compiled JSON. Pass `--template-name NAME` / `--preset NAME` to select any name
in `neuralfn.config.SHIPPED_GPT_TEMPLATE_PRESETS`. A `--graph-file PATH`
request is stricter: the CLI first performs source-inert Native IR lowering,
then validates the exact trainer-facing graph contract. Twelve reviewed graphs
are execution-ready: `gpt2`, `gpt2_megakernel`, `gpt2_moa`, `gpt2_qknorm`,
`gpt2_softcap`, `gpt2_stable`, `gpt2_zloss`, canonical `llama`, and its exact
compile-runtime alias `llama_fast`, plus exact standard-MoE `moe`, `mixllama`,
and `mixllama_fast`, route to a
compiled CUDA Tile trainer. LLaMA requires the exact proved RMSNorm/RoPE/MHA-
or-GQA/dense-attention/gate-first-SwiGLU, biasless, dropout-zero, untied graph.
The graph controls the selector and geometry even when conflicting
CLI flags are supplied. A real run snapshots the validated graph and plan in
`OUTPUT_DIR/native-ir`; dry-run and command-printing modes do not create it.
The LLaMA family binary rehashes that snapshot before dataset/CUDA setup and at
checkpoint write, applies caller `--weight-decay` to non-norm/non-bias
parameters, and emits graph provenance in its inference-checkpoint v2 metadata.
Both source profiles use native template/checkpoint identity `llama`; the plan
retains the original selector, preset, runtime, and source SHA-256.
Standard-MoE planning additionally proves softmax top-k-renormalized routing,
auxiliary-loss balance, no shared experts, floating expert width, and exact
root/block/attention/MLP edge wiring. It maps graph `multiple_of=None` to
`--multiple-of 0`, passes the configured router coefficient, and selects
`--train-moe-dataset-loop`; its strict metadata binds the ordered tensor table
and float32 sidecar to the same graph digest. CPU/build/migration/resident parity
is verified.

Exact `gpt2_moa` plans also pass the graph fingerprint, canonical
GELU/ReLU/SiLU/ReLU2 candidates, and positive probe interval to the dense
trainer. A completed run writes `model_XXXXXXXX.moa.json` beside the dense-v5
model and empty DONE marker. Downstream migration must use that metadata JSON,
which records the final selected activation and binds it to the exact source
graph/model; inference does not rerun candidate probes. A resumed run must find
and validate the sibling metadata, restores that activation without a fresh
probe, and rejects missing or tampered metadata. This strict workflow is
graph-bound: direct selector-only `gpt2_moa` first-leg training remains
supported and emits ordinary dense-v5, but an unbound resume fails explicitly
instead of resetting the activation to GELU.

`gpt2_diff` is the thirteenth execution-ready **graph-training** profile, but
only through `plan_native_graph_training(..., materialize=True)`. The trusted
planner validates the exact serialized configuration and active differential
topology from one source-byte snapshot, then writes a canonical
`native-training-proof.json` that binds the source SHA-256, validator and shape
contracts, and native geometry. The trainer requires the materialized graph,
fingerprint, and proof paths together before plan/Tile/CUDA work. The proof
digest is an unkeyed local-handoff integrity check, not an authenticity or
signature mechanism; do not trust arbitrary caller-created graph/proof pairs.

Its low-level compiled path learns one FP32 lambda
per layer and writes graph-bound differential parameter, optimizer, and strict
metadata sidecars while keeping dense-v5 byte-compatible. The metadata is
`neuralfn.native_gpt2_diff.training_checkpoint` version 2, with the unchanged
binary-artifact kind `trained_dense_v5_plus_diff_v1`. Resume is
continuation-only and verifies, before Tile/CUDA/H2D, the source graph and all
five binaries, optimizer/microbatch and sampler counters, seed contract,
batch/accumulation shape, ordered training-shard identity, optimizer/LR and
absolute schedule horizon, LM-head chunk, effective BF16 routes, and a canonical
numerics profile of supported effective routes. Stable no-follow descriptors are retained
for the training shards; validation shards are not identity-bound. `--max-steps` is
additional work. Omit `--lr-schedule-total-steps` on resume to inherit the
version-2 horizon, and repeat an explicit first-leg `--train-seed` value.
Version-1 metadata is rejected.

The public dense-GPT Python builders accept an explicit
`final_lr_fraction`. It takes precedence over a fraction derived from
`min_lr`; schedule, final-LR, and train-loss aliases are normalized before
quality defaults so caller intent is not overwritten on configured or direct
launch paths.

Final differential export is create-only and DONE-gated: exclusive no-follow
regular files are fsynced, the directory is fsynced, DONE is created/fsynced
last, and the directory is fsynced again. In-process failure attempts to clean
up only newly created targets. This is not an atomic-rename protocol, ancestor
symlinks are outside its guarantee, and metadata-smoke output is not covered.
The trainer also fails before Tile load or state mutation unless packed QKV,
sequence length at least 16, divisible/even head geometry, BF16 QKV-gradient
handoff, and the learned-lambda differential forward, backward, and
workspace-release Tile ABI symbols are present.
The retained legacy fixed-lambda ABI is outside this learned path and still has
rounded-output/non-layer-local backward correctness debt.
Exact graph-training plans report local persistence/execution readiness true
only after issuing the proof. Generic Native IR capability, migration, and
resident inference remain false because those consumers do not yet consume the
additive state or implement differential attention. The graph-training split is
13 ready and 54 blocked; persistence plus resident inference remain 12 ready
and 55 blocked.

NanoGPT command routing is likewise not graph-faithful execution proof. The
shared dense target can select NanoGPT geometry, but its persisted and executed
contract still has GPT-2 linear biases and no authored dropout. Because the
shipped NanoGPT graphs require bias-free linears and dropout, all three NanoGPT
selectors retain explicit persistence, native-forward, and resident blockers.
Do not migrate or serve those outputs as exact NanoGPT artifacts until the
bias/dropout and selector-specific contracts are independently proved.

A bounded RTX 5090 acceptance ran one real optimizer step for an
exact tiny canonical-LLaMA graph and an exact tiny standard-MoE graph, inspected
their graph-bound production checkpoints, migrated them losslessly, rebuilt the
resident extension, and generated raw tokens with the ordinary full-cache CLI.
Build the family trainer and Tile sidecar from the same sources before running
this workflow; a stale sidecar without the standard-MoE router-auxiliary symbol
is rejected before the optimizer step. This acceptance covers those exact
reviewed profiles only, not the remaining families, resident TurboQuant GPU
integration, or performance.
Dry-run/plan/check operations dominate training actions, and caller-selected
smoke/train modes are rejected by the graph-authoritative CLI.
Other graph profiles fail before trainer resolution with node-specific
compatibility JSON instead of falling back to Torch or a diagnostic transition
sampler. The adapter records that graph preflight is enforced while the current
trainer remains selector-driven and does not parse Native IR directly.

### Muse Glimmer native training and post-training

The production `muse_glimmer` graph has a dedicated source-bound native target,
`nfn_muse_glimmer_native_train`. Unlike generic preview-preset planning, it
requires the exact 52-layer/627-tensor topology and pinned BF16 source digest.
It consumes versioned uint32 shards for AR pretraining or structured SFT
records carrying `input_ids`, targets, loss masks, boundaries, tokenizer/chat
hashes, and split/objective metadata.

The C++/CUDA loop implements full parameter updates, all-eight-projection LoRA,
frozen NF4 group-64 QLoRA, DPO, sequence-masked reward modeling, and online
PPO. It persists optimizer moments, RNG/sampler/rollout cursor, data identity,
graph/source/reference/reward hashes, objective/adapter mode, and
activation-recomputation settings. Resume validates those fields before Tile or
CUDA state is created. QLoRA reconstructs immutable packed base rows from the
same BF16 source, computes packed base forward and backward-input only, and
saves adapter matrices rather than a second base model.

```bash
nfn train --base-model muse-glimmer \
  --checkpoint artifacts/glimmer-bf16/muse-glimmer-full.bf16 \
  --checkpoint-sha256 SHA256 \
  --dataset datasets/glimmer-sft \
  --objective sft \
  --chat-template-sha256 cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678 \
  --adapter qlora \
  --lora-targets q_proj,k_proj,v_proj,o_proj,attn_gate_proj,gate_proj,up_proj,down_proj \
  --qlora-group-size 64 \
  --tile-ops-lib /absolute/path/libnfn_native_train_tile_ops.so \
  --output-dir runs/glimmer-qlora
```

Native DPO consumes structured chosen/rejected masks, performs policy and
frozen-reference forwards, reduces sequence log-probabilities, and applies
sigmoid/hinge/IPO loss. Reward training updates a last-selected-token scalar
head with Bradley-Terry loss while the LM head stays frozen. PPO collects real
online rollouts, evaluates frozen reference and reward checkpoints, subtracts
per-token KL, computes GAE/returns, and performs clipped value/policy minibatch
epochs; it never synthesizes zero placeholder rollouts.

Official `k-quant-17gb` and `k-quant-dynamic` bases support native LoRA SFT and
DPO. Their GGUF codes are immutable and distinct from NF4 QLoRA. Every adapter
checkpoint pins the profile, base digest and tensor type table; packed DPO uses
the identical base as its frozen reference. Packed reward/PPO and lossy adapter
merge remain rejected.

Full-BF16 AR/SFT also supports contiguous pipeline parallelism:

```bash
nfn train --base-model muse-glimmer \
  --checkpoint artifacts/glimmer-bf16/muse-glimmer-text.bf16 \
  --checkpoint-sha256 SHA256 \
  --dataset datasets/glimmer-pretrain \
  --objective ar \
  --pipeline-parallel-size 8 \
  --pipeline-cuda-devices 0,1,2,3,4,5,6,7 \
  --nccl-lib /absolute/path/libnccl.so \
  --tile-ops-lib /absolute/path/libnfn_native_train_tile_ops.so \
  --output-dir runs/glimmer-8stage
```

The launcher creates one process per stage. Rank 0 owns embeddings, the final
rank owns norm/head, NCCL carries activations/gradients and global reductions,
and each rank performs an independent free-memory admission check. Rank-local
BF16 model shards, F32 moment shards, cursor/RNG state, manifest and final
`DONE` marker resume only under the identical world/stage layout. The current
distributed contract is full-BF16 AR/SFT; adapters, preference objectives,
tensor parallelism and data parallelism remain separately gated.

`build_muse_glimmer_dflash_distillation_graph()` and
`DFlashDistillationTrainer` provide separate target-frozen DFlash training with
random anchors, five taps, shared embedding/head, D-PACE or decay weighting,
self-logit distillation, exact resume, acceptance audit and native assistant
export. This is NeuralFn's recorded recipe, not a claim about Meta's unpublished
training provenance. Native multimodal tuning remains fail-closed.

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
| `warmup_steps`, `warmdown_fraction`, `lr_decay_iters`, `min_lr`, `max_wallclock_seconds` | int/float | `0`, `0.75`, `None`, `None`, `0.0` | Schedule controls for warmup priming, fractional tail warmdown, explicit cosine LR decay, LR floor, and wallclock cutoffs. `warmdown_fraction` controls the final share of optimizer steps used for linear warmdown. When `lr_decay_iters` is set, cosine decay overrides `warmdown_fraction`; omitting `min_lr` while cosine decay is enabled uses `learning_rate / 10`. On Tile CUDA `adamw` runs, missing `lr_decay_iters` defaults to the resolved training step count and missing `min_lr` defaults to `0.0`. Native GPT wrappers and compiled no-Python GPT entrypoints default to the 60-step SM120 reference warmup and still honor explicit `--warmup-steps` or `WARMUP_STEPS`/native warmup env aliases. `max_wallclock_seconds` only stops training early; it does not change the LR schedule. |
| `grad_clip_norm` | `float` | `0.0` | Global grad clipping threshold. |

`TorchTrainer` automatically adjusts `vocab_size` when the training data's token range exceeds the graph's configured vocabulary, ensuring the embedding and output layers are compatible.

Torch-runtime training compiles graph topology, input/output layout, and edge routing into a static execution plan before batches run. Real training tensors flow through fixed child modules and the precomputed plan; they do not pass through graph editor node objects, canvas positions, viewport state, or mutable editor metadata. CUDA Tile execution plans must preserve the same control-plane/data-plane split.

For Torch-free native GPT launchers, `neuralfn.native_train.build_native_train_run_config()` can enforce the dense-GPT strict LM-head parity guard with `require_cooperative_lm_head_backward=True`. The SDK appends `--require-cooperative-lm-head-backward` once, rejects non-dense family targets, and keeps the handoff in the compiled native frontend without importing Torch. Current CUDA Tile builds still fail that guard because the LM-head backward route is a diagnostic CUDA Graph wrapper rather than the future fused classifier/dHidden/dWeight kernel.

The same SDK helper expands dense GPT quality defaults before the direct native
C++ handoff. `gpt`, `gpt2`, `gpt3`, and `nanogpt` SDK configs inherit the CLI's
validation cadence, AdamW settings, token-batch shape, warmup, max-step, and
activation defaults unless a flag or `NFN_NATIVE_GPT_*` / `NFN_SM120_*` override
is explicit. GPT3 gets the 2048-context/batch-32 default and NanoGPT gets the
`nanogpt` template default; metadata-only actions stay schedule-free.

For dense GPT startup/preflight probes, pass `fast_startup=True` to the same native SDK helper to append `--fast-startup` once. This skips throughput-only setup prewarms through the native prewarm policy without requiring environment variables; normal training defaults remain unchanged.

For startup bisection, `NFN_NATIVE_GPT_CONCURRENT_PARAMETER_INIT=1` enables the
diagnostic path that launches token-weight initialization and independent
non-token parameter fill on separate nonblocking CUDA streams, then synchronizes
before BF16 block-weight refresh and training. Runtime JSON reports
`concurrent_parameter_init_requested`, `concurrent_parameter_init_enabled`, and
`concurrent_parameter_init_count`; the paired SM120 wrapper exposes the same
route as `NFN_SM120_NATIVE_CANDIDATE_PROFILE=concurrent_parameter_init`. It is
off by default because the latest 7-warmup paired gate rejected startup
promotion: steady-state timing stayed effectively flat, but setup wall and
startup-plus-first-step both regressed.

For embedding-route bisection,
`NFN_NATIVE_GPT_EMBEDDING_BF16_SHADOW=1` switches the native fused direct-u16
token/position/residual embedding kernel to read the maintained BF16
token-weight shadow instead of the FP32 token table. Runtime JSON reports
`embedding_bf16_shadow_requested`, `embedding_bf16_shadow_enabled`,
`embedding_bf16_shadow_kernel_loaded`, and `embedding_residual_strategy`; the
paired SM120 wrapper exposes the same route as
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=embedding_bf16_shadow`. It is off by
default because the latest paired gate rejected default promotion: setup wall
time improved, but train-loop wall time, steady-state CUDA-event step time, and
tokens/sec regressed.

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
