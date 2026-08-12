---
name: neuralfn-python-sdk
description: >-
  Build neural network graphs programmatically using the neuralfn Python
  package. Use whenever the user asks to write Python code that imports
  neuralfn, creates neurons, builds graphs, wires edges, trains models,
  serializes graphs, or works with the NeuralFn graph framework directly
  in code. Do NOT use for MCP tool calls -- see neuralfn-mcp instead.
---

# NeuralFn Python SDK

Use this skill when writing Python code that imports and uses the `neuralfn` package directly. For MCP tool operations, use the `neuralfn-mcp` skill instead.

For the complete class/method reference and graph-building walkthrough, see [reference.md](reference.md).

Full API documentation lives in the repo at `docs/` ([index](../../../docs/README.md)). For a single-file LLM-ready dump of all docs, see [llms-full.txt](../../../llms-full.txt).

## Core imports

```python
from neuralfn import (
    Port, NeuronDef, neuron, neuron_from_source, module_neuron, subgraph_neuron,
    BuiltinNeurons, NeuronGraph, NeuronInstance, Edge,
    SurrogateTrainer, EvolutionaryTrainer, HybridConfig, HybridTrainer,
    TorchTrainConfig, TorchTrainer,
    build_gpt_root_graph, build_model_stage_graph,
    save_graph, load_graph,
    SurrogateModel, probe_neuron, build_surrogates,
    NativeCompatibilityReport, NativeExecutionManifest, NativeLoweringIssue,
    NativeMigrationResult, NativeTensorSpec,
    NativeCapabilityProof, NativeLoweringSpec, NativeTrainerSpec,
    GenerationConfig, GenerationEvent, GenerationResult, KVCacheConfig,
    NativeInferenceCapabilities, NativeInferenceCancelledError,
    NativeInferenceModel, NativeInferenceSession,
    TurboQuantEncodedVector, TurboQuantReferenceCodec,
    deterministic_random_rotation, lloyd_max_centroids,
    pack_mixed_bit_indices, unpack_mixed_bit_indices,
    compile_graph_to_native_manifest, migrate_graph_to_native,
    capability_proof_for, classify_native_model,
    native_lowering_specs, native_trainer_specs, registered_native_module_types,
)
```

Additional imports for configs:

```python
from neuralfn.trainer import TrainConfig
from neuralfn.evolutionary import EvoConfig
from neuralfn.config import ModelSpec, BlockSpec, TemplateSpec
from neuralfn.inference import export_to_pt, import_from_pt, InferenceCache
```

When authoring MoE `BlockSpec` or `expert_dispatch` module config, preserve
`mlp_multiplier` as a finite positive float. Torch and Tile compute expert
width as `max(1, int(model_dim * mlp_multiplier))` and round upward only for a
positive `multiple_of`; `None` and `0` are unaligned. Legacy dispatch nodes
that omit the multiplier retain the `4.0` compatibility default. Checkpoints
from the former integer-truncated width require expert-tensor migration.

`InferenceCache(graph, device=..., compiled=...)` may reuse an already
weight-loaded `CompiledTorchGraph`; use this for temperature-zero strict FP32
inference so cache construction cannot replace the selected backend. Native GPT
sampler helpers accept `strict_tile_ops_lib=` for the required exact-zero
`libnfn_native_train_tile_ops_strict.so` sidecar. Temperatures must be finite
and nonnegative; only exact zero activates strict deterministic CUDA compute.
Application-owned Torch loops must call the Torch-free
`prepare_inference_process_environment()` before CUDA, validate with
`validate_inference_temperature()`, use a Torch-only FP32 model verified by
`validate_strict_graph_support()` / `validate_strict_compiled_graph()`, and keep
the complete forward and argmax inside
`inference_execution(temperature, torch_module=torch)`.
The context uses the complete hierarchical PyTorch `fp32_precision="ieee"`
API when available and otherwise the legacy TF32-disable API; never mix them.

Resident `NativeInferenceSession.current_logits()` is the public read-only
diagnostic for parity, NLL, and perplexity evaluation after a non-empty
prefill. It returns finite logits without sampling, appending a token, or
mutating cache/RNG state; do not reach into private model/session handles.

## Native Execution IR migration

Use Native Execution IR v1 as the additive, ahead-of-time boundary from graph
authoring to native artifacts:

```python
from neuralfn import migrate_graph_to_native

result = migrate_graph_to_native(
    "model.json",
    weights_path="model.pt",  # optional; omit for Torch-free graph migration
    output_dir="model-native",
    dry_run=True,
)
if not result.report.compatible:
    for issue in result.report.issues:
        print(issue.path, issue.code, issue.message)
```

`compile_graph_to_native_manifest(graph)` deep-copies, validates, resolves
variants, and lowers an in-memory graph without mutating it.
`compile_native_graph_payload(payload)` performs source-safe raw JSON lowering
without importing Torch, NumPy, or NetworkX.
`migrate_graph_to_native()` additionally preserves and fingerprints exact file
bytes, safely preflights serialized source, optionally converts `.pt` tensors,
and exclusively materializes a new directory. Existing destinations/symlinks
are always rejected. Graph preflight must finish before checkpoint access;
arbitrary graph-supplied Python and unregistered module types fail closed. The
optional checkpoint worker runs isolated with `weights_only=True`.

Use `plan_native_graph_training()` / `preflight_native_graph_training()` for a
canonical target and optional exclusive IR/plan materialization. Require
`execution_ready` and `graph_preflight_enforced` before launching. The reviewed
`gpt2`, `gpt2_megakernel`, `gpt2_moa`, `gpt2_qknorm`,
`gpt2_softcap`, `gpt2_stable`, `gpt2_zloss`, canonical `llama`, and its exact
compile-runtime alias `llama_fast`, plus exact standard-MoE `moe`, `mixllama`,
and `mixllama_fast`, plus trusted-planner proof-bound `gpt2_diff` training, are
ready after exact validation (13 ready; 53 blocked). `gpt2_diff` migration and
resident inference remain blocked. Materialized graph training requires the
exact `graph_file` / `graph_fingerprint` / `graph_preflight_proof` triplet; the
unkeyed proof digest is local-handoff integrity, not caller authenticity. Its
low-level trainer learns and persists per-layer lambda/moments in a
source-bound additive bundle. Its version-2 strict continuation metadata binds
all five binaries, source graph, resolver-ordered training shards,
counters/sampler, seed, accumulation shape, optimizer/LR horizon, BF16 routes,
and a canonical profile of supported effective numerics before Tile/CUDA/H2D; validation
shards are excluded. Native IR migration and resident inference do not consume it. The
trainer must reject before Tile load/mutation unless
packed QKV, `seq_len >= 16`,
divisible/even head geometry, BF16 QKV-gradient handoff, and the differential
learned-lambda forward, backward, and workspace-release Tile ABI symbols are
present.
Do not substitute the retained fixed-lambda ABI; it remains outside the learned
path with rounded-output/non-layer-local backward correctness debt.
For public dense-GPT SDK handoff, carry `lr_schedule`,
`lr_schedule_total_steps`, `train_seed`, `resume_from_checkpoint`, and
`graph_fingerprint`, and `graph_preflight_proof` through `NativeGptRunConfig` / `NativeGpt2RunConfig` and
their builders. The builders also accept optional `final_lr_fraction`; an
explicit fraction overrides one derived from `min_lr`, and schedule, final-LR,
and train-loss aliases are normalized before quality defaults. Use
`compiled_cli_argv()` for the native-only continuation fields; the
legacy llm.kittens short argv must reject rather than omit them. Dataset-alias
and strict Tile binding configs prefer the compiled argv. A direct config with
strict fields, no dataset alias, and a legacy short executable is invalid; use
the compiled builder or name both the dataset alias and compiled trainer.
Non-Tile configs with strict continuation fields are also invalid.
LLaMA SDK command construction must
re-run the planner, require plan-owned arguments exactly once, and bind the
source SHA; a caller-provided path/template/digest is insufficient. The current
executables still receive an immutable source snapshot plus selector/geometry,
so `trainer_consumes_native_ir` remains false.
Both LLaMA profiles normalize native template/checkpoint identity to `llama`
while retaining source selector/runtime/SHA provenance.
Standard-MoE plans must preserve floating width, map `multiple_of=None` to
native `0`, and bind experts, top-k, router auxiliary coefficient, runtime, and
source SHA. Their normal action is `--train-moe-dataset-loop`, and migration
accepts only the strict graph-bound v1 metadata/sidecar bundle.
For graph-bound MoA resume, require the source-bound `.moa.json`, restore its
selected activation without a new probe, and fail closed if the sidecar is absent or
changed. Direct selector-only first-leg training remains supported and emits
ordinary dense-v5, but reject an unbound resume rather than resetting to GELU.
Treat all other profiles as node-specific failures and never substitute a
diagnostic transition sampler.

Use `native_lowering_specs()`, `native_trainer_specs()`,
`registered_native_module_types()`, `classify_native_model()`, and
`capability_proof_for()` for deterministic capability inspection. Do not infer
runtime support from structural lowering or trainer registration. Although all
66 shipped text presets structurally lower, trainer registration and executable
routing are not forward proof. NanoGPT specifically remains blocked because its
bias-free/dropout graph differs from the biased, dropout-free shared dense-v5
contract; require false persistence/forward/resident gates for all three
NanoGPT selectors. Canonical LLaMA and exact standard MoE are
separately promoted by topology/checkpoint proof. Seven reviewed
dense preset topologies (`gpt2`, megakernel, MoA, z-loss, QK-norm, stable, and
softcap) paired with compatible dense-v5 `.bin` checkpoints additionally prove
the in-process resident ABI, preallocated lossless K/V cache, and lean serving
path, and migration stamps that ready artifact. QK-norm/softcap explicit fields,
active configs, canonical layer placement, and port-level edge chains must match
exactly; disconnected or bypassed transform nodes fail closed in both Python
proof and the C++ loader. Supported even head geometry additionally
proves native packed CPU TurboQuant cache ABI v1. Generic `.pt`, graph-only,
bare MoA `.bin` files, differential/modern variants, and other non-dense
adapters remain closed. Compatible reviewed-dense artifacts separately prove
`turboquant_tile_attention` for contexts through 16K and even head dimensions
in 2..256. Explicit configuration consumes the strict Tile-sidecar ABI for GPU
historical compressed attention while CPU remains the default. See
`docs/python-sdk/native-ir.md` for the schema and all public dataclasses.

Exact `gpt2_moa` must be migrated through source-bound
`model_XXXXXXXX.moa.json`, never its bare `.bin`. Require the named dense-v5
model, empty DONE marker, byte-identical graph, canonical
GELU/ReLU/SiLU/ReLU2 candidates, selected activation, and positive interval.
Migration must copy the exact metadata as `model.moa.json` and bind its
path/size/SHA-256; remigrate older artifact directories that lack that copy.
The resident CPU path then uses that activation through prefill/decode,
`off`/`auto`/`full`, and supported-even-head TurboQuant without reprobes.
For graph parity/debugging, import `load_native_moa_graph_runtime` from
`neuralfn.native_moa_graph_runtime`; it requires the original byte-identical
graph, revalidates model/metadata/tensor hashes, imports every dense-v5 tensor,
and overlays only canonical MLP activation stages. Do not present that
Torch-only diagnostic loader as the resident serving path.

Canonical `llama` is the first proved non-dense exception. Migrate only its
native-family inference-checkpoint v2 metadata JSON, never the raw `.f32` or
diagnostic v1 metadata. The inspector validates the complete
RMSNorm/RoPE/GQA/SwiGLU/untied-head float32 tensor contract and copies the
sidecar as `model.f32`; the resident adapter supports `off`/`auto`/`full` and
rejects TurboQuant. Its raw-token SDK/CLI route is proved, but HTTP/text use
still needs supported tokenizer/chat metadata.

Exact standard MoE is the second proved non-dense exception. Migrate only
`neuralfn.native_family_standard_moe.inference_checkpoint` v1 metadata whose
DONE marker, float32 sidecar, whole-file/tensor hashes, ordered router/expert
layout, geometry/semantics, and source graph SHA all validate. Its resident
adapter supports `off`/`auto`/`full`, ordinary raw-token SDK/CLI inference, and
rejects TurboQuant. Modern, megakernel, DeepSeek, aux-free, shared-expert, and
JEPA neighbors remain closed.

Tile sparse-attention SDK calls support key prefixes through 1024 positions.
Strict Python must reject larger CUDA prefixes, and the raw float32 sparse
forward/backward ABIs must return `cudaErrorInvalidValue` before launch for
`seq_k > 1024`. Current right-aligned GQA coverage proves recompute-slice mask
semantics only; never claim a resident sparse cache or physical eviction.

## Resident native inference contract

Use `NativeInferenceModel.load()` and `NativeInferenceSession` only with a
Native IR artifact and in-process binding that jointly prove resident ABI v1.
The Python contract exposes `create_session`, `prefill`, `decode`, `truncate`,
`reset`, `cancel`, `stats`, and `close`, synchronizes exact prefixes, and calls
token callbacks only after state commitment. It never falls back to a
subprocess. `KVCacheConfig(mode="auto"|"full")` requires proven lossless cache
support; explicit `turboquant` additionally requires joint TurboQuant proof and
never silently substitutes full cache. `bash tools/build_native_inference_binding.sh`
builds the reviewed dense-v5, canonical LLaMA, and exact standard-MoE CPU
reference engines. For a
compatible bound artifact, `auto`/`full` use the retained lossless cache, `off` uses the
full-prefix-recompute oracle, and explicit `turboquant` uses the jointly proved
native packed CPU cache only for supported dense topologies. To select the
hybrid backend, require `turboquant_attention_backend="tile-cuda"`, a strict
absolute `tile_ops_lib`, and optional CUDA runtime/device; never fall back when
that explicit request fails. Model compute/encoding stay on CPU while packed
historical attention uses CUDA. Require a contained fingerprinted
`checkpoint.artifact_path`. Do not present this as whole-model CUDA,
differential/modern, unbound MoA, or other non-dense adapter coverage. See
`docs/python-sdk/native-inference.md`.

Reviewed resident-ready dense-v5, canonical-LLaMA, and exact standard-MoE
artifacts in lossless `full` mode may jointly prove additive
`session_prefix_cow` v1 with their exact dense, LLaMA-GQA, or standard-MoE-GQA
profile. Only then call
`model.fork_session(source, token_count=None, seed=0)`. The source prefix must
be non-empty; cross-model, closed/poisoned, off, and Tile-configured sources
fail closed. Native K/V and final-hidden allocations are shared until the first
parent/child append detaches all components; RNG, tokens, cancellation,
reset/truncate, counters, and close stay private.

For dense CPU TurboQuant, require the separate effective capability
`session_prefix_cow_cpu_turboquant`, exact artifact/binding profile
`dense-cpu-turboquant-mse-qjl-packed-kv-final-hidden-v1`, backend
`cpu-reference-packed`, binding profile inventory, and callable
`fork_session`. The source must have effective cache `turboquant`, profile
`mse-3.5` or `qjl-3.5`, and CPU attention. Its packed K/V and lossless
final-hidden stores share whole capacity until a prefill/decode append; logical
truncate/reset does not detach. Reject Tile-CUDA-configured models and sessions.
When code needs the exact profile string, import the public module constant
`CPU_TURBOQUANT_SESSION_PREFIX_COW_PROFILE` from
`neuralfn.native_inference` instead of duplicating it; importing the constant
does not prove or enable the capability.
For either COW path, detachment is transactional: failed/cancelled appends
restore the original shared allocation and ownership/detach telemetry before
surfacing the error. These remain the native session-fork primitives and do not
cover nonstandard MoE or Tile device storage. The standalone HTTP app may layer
its separate default-off, process-local Responses LRU and schema-v4
conversation CAS over the proved full/CPU-TurboQuant forks; never treat token
replay alone as COW.

Treat model close as an atomic admission fence. A create/fork already holding
the model lifecycle lock is registered and included in teardown; later
create/fork and newly admitted session operations fail before binding work.
The teardown owner closes each handle once and receives the first error.
Concurrent/reentrant duplicate `model.close()` calls intentionally return
without waiting, because waiting while holding a session operation lock could
deadlock the owner.

Benchmark resident full/CPU/Tile cache modes with
`tools/bench_native_resident_turboquant.py`, an exact artifact, fresh binding,
strict sidecar, explicit CUDA runtime, and matching token corpus. Treat its
VRAM value as a sampled device-global baseline delta, not a per-process high
water. Omitted tokens select a synthetic mechanics corpus. Never infer a
speedup or quality-neutrality claim from the shipped tiny-fixture calibration;
the JSON intentionally sets `speedup_claimed=false`.

Treat cooperative cancellation as recoverable, not as a generic binding
failure. In-flight `prefill()` raises `NativeInferenceCancelledError` after
native rollback; in-flight `decode()` returns a cancelled result without the
uncommitted token. Call `reset()` before reuse; it restores initial RNG state.
All other binding exceptions poison the session.

`neuralfn.native_serve` is a separate inference-only FastAPI application for
artifacts that already pass those resident gates. It loads one model before
bind, creates an isolated session per request, bounds admission around one
worker, and implements Models plus buffered/streamed text-only Chat
Completions. Supplying `--state-db` additionally mounts the bounded text
Responses/Conversations subset, API-key-scoped lineage and local compaction,
semantic Responses SSE, and durable background/cancel processing. It does not
provide a subprocess fallback. `NativeServeConfig.prefix_cache_capacity`
(CLI `--prefix-cache-capacity`) is an entry-count limit that defaults to zero.
A trailing public dataclass field was added after `log_level`; existing
positional meanings remain stable, but exact-shape serializers/matches must
accept it and new callers should prefer keywords.
A positive value requires the state store plus effective lossless-full COW or
reviewed dense CPU-TurboQuant COW; reject Tile-CUDA. Foreground stored
previous-response/conversation aliases select candidates, but reuse requires an
exact rendered-token LCP and native cached rows. Admit only stored completed/
incomplete outcomes after durable finish. `store:false` may hit but cannot
admit; failed/cancelled/background/Chat stay cold, and restart is cold. Count
cache writes from prepared-prompt rows only, never decode. `/health.prefix_cache`
reports entry/LRU/token and per-session capacity observations; do not label its
bytes physical. Whole-scope purge and epoch-fence response/conversation/item
deletion. Shutdown awaits background and tracked foreground SSE drivers,
drains the queue, then closes cache/model/state in order. An exact additive profile permits buffered,
stored, foreground strict flat JSON-schema Responses when artifact ABI,
binding `current_logits_exact_prefill`, artifact-selected presentation, and
byte-exact tokenizer preflight agree. Exact tool-template metadata additionally
permits one forced client-executed function call and a later string
`function_call_output`; NeuralFn never executes it. Schema/argument generation
is greedy (`temperature=0`, `top_p=1`); the result continuation is ordinary
text sampling but stays stored/buffered/foreground with truncation disabled.
The state store is schema v4: v1/v2/v3 migrate in place with existing
conversation history at revision zero, so require a backup before upgrade when
rollback matters because older binaries reject v4. A migrated legacy queued
conversation job lacks a historical revision snapshot and must fail with
`conversation_snapshot_unavailable`; legacy previous-response-only work may
reconstruct only completed/incomplete lineage or fail with
`response_lineage_unavailable`. Preparation snapshots
conversation items and revision together; terminal response/output/
conversation persistence CASes it atomically before cache admission. A stale
branch becomes `conversation_conflict` (buffered 409 or semantic
`response.failed` after stream/background start). Finish must re-read the full
previous-response lineage and raise `response_lineage_conflict` on deleted or
changed ancestry before output/cache publication. Keep general/parallel/
hosted tools, nested or array schemas, constrained streams/background work,
Chat Completions tools, multimodal content, and batching fail-closed. Require
Bearer auth for remote binds unless the operator explicitly acknowledges the
unauthenticated override.
See `docs/rest-api/native-inference-serving.md`.

For direct stateful embedding, use the explicitly exported module APIs
`NativeStateStore`/`NativeStateError`/`NativeStateConflictError` and
`api_key_fingerprint` from `neuralfn.native_state`, plus
`NativeResponsesService`/`NativeResponsesAPIError` and the frozen
prepared/completed phase values from `neuralfn.native_responses`. Keep the
service lifecycle ordered as prepare, persist, execute, then finish or fail;
validation and lineage resolution must happen before admission, mutation, or
resident-session creation. The runtime owns and closes a successfully attached
store; callers close it themselves if runtime construction fails. Pass the
actual `chat_template_selection` to `NativeServingRuntime` (normally `"auto"`):
the runtime rechecks the raw manifest tool template, and explicit
`plain_roles` or path selection disables both structured-output and function-
tool capabilities. Treat
`neuralfn.native_constrained` as serving-internal, not a supported direct SDK
surface. `conversation_items_snapshot`,
`append_conversation_items_with_revision`,
`delete_conversation_item_with_revision`, and
`finish_foreground_response` are the public schema-v4 primitives;
`finish_background_job` accepts the same optional conversation ID/items/
expected-revision CAS inputs. Stale CAS raises
`NativeStateConflictError.code == "conversation_conflict"`. Public
`NativeResponsesService.execute()+finish()` deliberately stays cold; the HTTP
app owns its private combined resident lifecycle. One cache-enabled state DB
has one owning service process, and all semantic mutations must flow through
that service because its lock/epoch is process-local. See
`docs/python-sdk/native-inference.md` for imports and ownership.

`TurboQuantReferenceCodec` is the dependency-free correctness oracle for
deterministic rotation, Lloyd-Max scalar quantization, mixed-bit packing, MSE
value reconstruction, and optional QJL key/query residual correction. Exact
dense-v5 artifacts with supported head geometry use the agreeing native CPU
packed implementation. A separate versioned Tile-sidecar CUDA attention ABI
agrees for MSE/QJL, MHA/GQA, and chunked contexts through 16K; explicit
reviewed-dense resident dispatch is live-proved for both profiles with positive
GPU transfer/launch telemetry and zero CPU compressed calls. Other families
and transfer-inclusive performance remain fail-closed. Keep the old
graph `kv_quant_pack/unpack` format unchanged. See
`docs/python-sdk/turboquant-reference.md`.

## Creating neurons

### Scalar function neuron (`@neuron`)

```python
@neuron(
    inputs=[Port("x", range=(-5, 5), precision=0.01)],
    outputs=[Port("y", range=(0, 1), precision=0.001)],
)
def my_sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))
```

### Dynamic source neuron

```python
ndef = neuron_from_source(
    "def relu(x):\n    return max(0, x)\n",
    "relu",
    [Port("x", range=(-10, 10))],
    [Port("y", range=(0, 10))],
)
```

### Module neuron (torch stage)

```python
linear = module_neuron(
    name="linear", module_type="linear",
    input_ports=[Port("x", range=(-1e6, 1e6))],
    output_ports=[Port("y", range=(-1e6, 1e6))],
    module_config={"input_dim": 128, "output_dim": 128, "bias": True},
)
```

### Subgraph neuron (nested graph)

```python
child = NeuronGraph(name="block")
child.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
child.add_node(NeuronInstance(BuiltinNeurons.sigmoid, instance_id="act"))
child.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
child.add_edge(Edge(id="e1", src_node="in", src_port=0, dst_node="act", dst_port=0))
child.add_edge(Edge(id="e2", src_node="act", src_port=0, dst_node="out", dst_port=0))
child.input_node_ids = ["in"]
child.output_node_ids = ["out"]

block_neuron = subgraph_neuron(child, name="sig_block", input_aliases=["x"], output_aliases=["y"])
```

## Building and executing graphs

```python
g = NeuronGraph(name="my_graph", training_method="surrogate", runtime="scalar")

g.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in1"))
g.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in2"))
g.add_node(NeuronInstance(BuiltinNeurons.sigmoid, instance_id="act"))
g.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))

g.add_edge(Edge(id="e1", src_node="in1", src_port=0, dst_node="act", dst_port=0, weight=1.0, bias=0.0))
g.add_edge(Edge(id="e2", src_node="in2", src_port=0, dst_node="act", dst_port=0, weight=1.0, bias=0.0))
g.add_edge(Edge(id="e3", src_node="act", src_port=0, dst_node="out", dst_port=0))

g.input_node_ids = ["in1", "in2"]
g.output_node_ids = ["out"]

result = g.execute({"in1": (0.5,), "in2": (0.3,)})
# result: {"out": (0.689...,)}
```

## Training

### Surrogate (scalar graphs, gradient-based)

```python
import numpy as np
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
Y = np.array([[0],[1],[1],[0]], dtype=np.float32)

trainer = SurrogateTrainer(graph, TrainConfig(epochs=300, learning_rate=0.01))
losses = trainer.train(X, Y)
```

### Evolutionary (scalar graphs, population-based)

```python
evo = EvolutionaryTrainer(graph, EvoConfig(population_size=40, generations=100))
losses = evo.train(X, Y)
```

### Hybrid (nested subgraphs, per-graph method)

```python
# Set training_method on each subgraph: "surrogate", "evolutionary", or "frozen"
trainer = HybridTrainer(root, HybridConfig(outer_rounds=3))
losses = trainer.train(X, Y)
```

### Torch (tensor graphs, PyTorch training)

```python
from neuralfn.config import build_nanogpt_spec

spec = build_nanogpt_spec(n_layer=4, n_embd=128, num_heads=4)
graph = build_gpt_root_graph(name="nanogpt", model_spec=spec)
trainer = TorchTrainer(graph, TorchTrainConfig(epochs=10, learning_rate=5e-3, device="cuda"))
losses = trainer.train(train_inputs, train_targets)
```

## Serialization

```python
save_graph(graph, "my_model.json")
loaded = load_graph("my_model.json")

# Dict round-trip
d = graph.to_dict()
g2 = NeuronGraph.from_dict(d)
```

## Common builtin neuron IDs

| Attribute | ID | Kind | Ports |
|-----------|----|------|-------|
| `input_node` | builtin-input | function | 0 in, 1 out |
| `output_node` | builtin-output | function | 1 in, 1 out |
| `sigmoid` | builtin-sigmoid | function | 1 in, 1 out |
| `relu` | builtin-relu | function | 1 in, 1 out |
| `tanh_neuron` | builtin-tanh | function | 1 in, 1 out |
| `identity` | builtin-identity | function | 1 in, 1 out |
| `add` | builtin-add | function | 2 in, 1 out |
| `multiply` | builtin-multiply | function | 2 in, 1 out |
| `gelu` | builtin-gelu | function | 1 in, 1 out |
| `silu` | builtin-silu | function | 1 in, 1 out |

Access via `BuiltinNeurons.sigmoid`, `BuiltinNeurons.relu`, etc. Full list: `BuiltinNeurons.all()`.

## Quick reference

### Port fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | required | Unique identifier |
| `range` | tuple[float, float] | (-1.0, 1.0) | Value bounds |
| `precision` | float | 0.001 | Quantization step |
| `dtype` | str | "float" | Semantic type |

### Edge fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | str | auto | Unique edge ID |
| `src_node` | str | "" | Source node instance_id |
| `src_port` | int | 0 | Source port index |
| `dst_node` | str | "" | Destination node instance_id |
| `dst_port` | int | 0 | Destination port index |
| `weight` | float | 1.0 | Multiplicative weight |
| `bias` | float | 0.0 | Additive bias |

### NeuronGraph key methods

| Method | Description |
|--------|-------------|
| `add_node(instance)` | Add a NeuronInstance |
| `add_edge(edge)` | Add an Edge |
| `remove_node(id)` | Remove node and connected edges |
| `remove_edge(id)` | Remove edge |
| `execute(inputs)` | Run with scalar inputs |
| `execute_trace(inputs)` | Run and return all activations |
| `has_cycles()` | Check for cycles |
| `topological_order()` | Get execution order |
| `validate()` | Validate graph structure |
| `to_dict()` / `from_dict(d)` | Serialization |
| `get_edge_params()` / `set_edge_params(p)` | Edge weight vector |
| `resolve_variant_library()` | Resolve variant references |

### NeuronDef kinds

| Kind | Created by | Runtime |
|------|-----------|---------|
| `"function"` | `@neuron`, `neuron_from_source` | scalar |
| `"subgraph"` | `subgraph_neuron` | scalar (recursive) |
| `"module"` | `module_neuron` | torch only |
