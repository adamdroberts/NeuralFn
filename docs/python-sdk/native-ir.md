# Native Execution IR

Native Execution IR v1 is NeuralFn's versioned, ahead-of-time artifact contract
for native execution. Python resolves and validates the authoring graph, then
lowers its topology and metadata into a deterministic manifest. A native
runtime consumes that artifact contract; it does not interpret graph JSON or
execute graph-supplied Python.

## Current status

| Capability | Status in v1 |
|---|---|
| Structural lowering | All 67 shipped text presets and all root variant-library entries lower through an explicit module allow-list. |
| Graph preservation | File migration leaves the source bytes unchanged; programmatic compilation deep-copies the graph before variant resolution. |
| Checkpoint conversion | Legacy `.pt` conversion is isolated; compatible native dense-v5 `.bin`, source-bound dense-MoA metadata `.json`, canonical native-family LLaMA metadata `.json`, and graph-bound standard-MoE metadata `.json` inputs are inspected without Torch and bound directly. All run only after graph preflight. |
| Native architecture forward | One-shot family proof exists for ordinary dense `gpt`, `gpt2`, and `gpt3`; exact `gpt2_moa`, canonical native-family `llama`, and standard-MoE `moe`/`mixllama`/`mixllama_fast` additionally prove their persisted architecture forwards. The strict pinned Muse Glimmer converter/trainer/resident path proves its production topology separately from generic graph migration. GPT2-Evo is preflight-only because its native delegate does not match the authored whole-block evolution contract. Graph-authored `gpt2_diff` is excluded until migration/resident consumers validate its additive learned-lambda bundle, and NanoGPT is excluded until its bias/dropout semantics are persisted. |
| Resident model/session runtime | Seven reviewed dense-v5 preset topologies (`gpt2`, megakernel, MoA, z-loss, QK-norm, stable, and softcap), canonical `llama`, and exact standard-MoE `moe`/`mixllama`/`mixllama_fast` load through an in-process CPU ABI v1 engine. Strict pinned Muse Glimmer BF16/K-Quant bundles separately support resident CPU and whole-model CUDA text. Neighboring variants and unproved family adapters fail closed. MoA additionally requires its strict source-bound sibling metadata. |
| Retained lossless cache | The reviewed dense-v5, canonical LLaMA, and exact standard-MoE artifacts support preallocated `auto`/`full` K/V state with whole-logit parity against `off` recomputation. Muse Glimmer uses its exact local-ring/global-full hybrid cache, including paired transactional target/assistant state when DFlash is enabled. |
| TurboQuant cache | Bound reviewed dense-v5 artifacts with even head dimensions prove native packed CPU `mse-3.5`/`qjl-3.5` cache ABI v1. Contexts at most 16K with even head dimensions in 2..256 separately prove `turboquant_tile_attention` feature ABI v1; an explicit strict-sidecar request can run packed historical attention on CUDA while CPU remains the default. Other families remain unavailable. |
| Session prefix COW | Bound resident-ready dense-v5, canonical-LLaMA, and exact standard-MoE artifacts with lossless-cache proof emit `capabilities.session_prefix_cow=true` plus `kernel_abi.session_prefix_cow={version:1,status:"ready",profile:<format-specific-profile>,operation:"fork_session"}`. The full-cache profiles are `dense-full-cache-kv-final-hidden-v1`, `llama-full-cache-gqa-kv-final-hidden-v1`, and `standard-moe-full-cache-gqa-kv-final-hidden-v1`. Reviewed dense-v5 TurboQuant artifacts separately emit `capabilities.session_prefix_cow_cpu_turboquant=true` and `kernel_abi.session_prefix_cow_cpu_turboquant={version:1,status:"ready",profile:"dense-cpu-turboquant-mse-qjl-packed-kv-final-hidden-v1",operation:"fork_session",backend:"cpu-reference-packed"}`. Both are whole-storage SDK/native session-fork primitives; Tile device state and serving lineage remain unavailable. |
| Native serving | Compatible dense-v5, canonical LLaMA, exact standard-MoE, and strict Muse Glimmer bundles prove their respective resident/lean-serving ABIs; LLaMA and MoE still need separately supported tokenizer/chat metadata, while Glimmer binds authenticated `tokenizer.json` plus ATEM. `--state-db` adds bounded text Responses/Conversations, local compaction, and durable background work. Exact metadata can enable buffered strict flat JSON schema and one forced client-executed function call/result. Chat media is limited to independently proven CPU Glimmer base64 images; broader tools, schemas, Responses multimedia, and CUDA vision remain unavailable. |

Structural compatibility is not proof of a native forward, resident adapter,
cache, or serving path. Read `manifest.capabilities` and the compatibility report's
`capability_proof` instead of inferring support from a preset or registered
trainer name.

The shipped-catalog guard derives its matrix from the preset and native-family
registries rather than a duplicate allow-list. Every preset must lower and have
one native-family owner; persistence and resident inference must be either
proved or accompanied by explicit machine-readable blocked evidence. The
current persistence-plus-resident status is 12 ready and 55 explicitly blocked
out of 67. Exact trusted-planner graph training separately has 13 ready and 54
blocked because proof-bound `gpt2_diff` is training-only. These guards prevent a
new preset from shipping with an unclassified native status, but they do not
turn a blocked declaration into runtime proof.

Muse Glimmer is deliberately outside those generic graph-migration counts. Its
production artifact is built by a pinned family converter, not by inferring a
30B checkpoint layout from the preview-sized preset graph. The dedicated path
now provides:

- strict streaming BF16 target/full/assistant conversion;
- strict canonical GGUF K-Quant-Dynamic, K-Quant-17GB, DFlash, and mmproj
  inspection/bundling;
- additive `primary_checkpoint_variant`, `checkpoint_variants`,
  `companion_checkpoints`, `speculative_decoding`, memory-profile, and
  compatibility fields while preserving the v1 top-level `checkpoint` object;
- resident CPU and whole-model CUDA text execution, hybrid lossless cache,
  DFlash, and load-time weight-precision selection; and
- exact graph preflight into the dedicated `nfn_muse_glimmer_native_train`
  target for production AR/SFT and native LoRA/QLoRA configurations.

The generic catalog's preview `muse_glimmer` graph remains blocked from its
ordinary one-checkpoint migration proof because preview geometry is not the
pinned 52-layer production checkpoint. This is intentional: use the strict
family commands below rather than relabeling a preview graph as resident-ready.

NanoGPT remains in the blocked set even though its family is registered and its
commands resolve to `nfn_gpt_native_train`. The authored NanoGPT graphs require
bias-free linear layers and dropout; the shared dense-v5 trainer/checkpoint is
the biased, dropout-free GPT-2 contract. `native_trainer_specs()` therefore
reports NanoGPT persistence false, `diagnostic-transition-only`, and resident
inference false, and `capability_proof_for()` adds selector-specific blockers
for `nanogpt`, `nanogpt_megakernel`, and `nanogpt_modern`. Do not infer graph
execution from target resolution or geometry selection.

## CLI migration

Convert the pinned Muse Glimmer BF16 family without Torch:

```bash
nfn migrate muse-glimmer-to-native \
  --source /models/Muse-Glimmer-30B \
  --component full \
  --output-dir artifacts/glimmer-bf16
```

Bundle one or both canonical official packed profiles and optional companions:

```bash
nfn migrate muse-glimmer-gguf-to-native \
  --gguf /models/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
  --gguf /models/Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf \
  --gguf /models/dflash-Muse-Glimmer-30B-Q4_K_M.gguf \
  --tokenizer-source /models/Muse-Glimmer-30B \
  --output-dir artifacts/glimmer-kquant
```

Attach a strict native LoRA/QLoRA checkpoint atomically:

```bash
nfn migrate muse-glimmer-lora-to-native \
  --artifact artifacts/glimmer-bf16 \
  --checkpoint runs/glimmer-adapter/checkpoint-step-100
```

The converters pin exact repository revisions, source file hashes, tensor
names/shapes/dtypes/counts, tokenizer/chat hashes, and official GGUF profile
digests. Unknown encodings, retained lowercase legacy files, partial shards,
wrong target lineage, and source/companion mismatches fail before publication.
See [Resident Native Inference](native-inference.md) for model-load and VRAM
selection APIs.

Validate without writing an artifact:

```bash
nfn migrate graph-to-native \
  --graph artifacts/model.json \
  --output-dir artifacts/model-native \
  --dry-run
```

Materialize a graph-only artifact:

```bash
nfn migrate graph-to-native \
  --graph artifacts/model.json \
  --output-dir artifacts/model-native
```

Convert an optional legacy checkpoint at the same boundary:

```bash
nfn migrate graph-to-native \
  --graph artifacts/model.json \
  --weights artifacts/model.pt \
  --output-dir artifacts/model-native
```

Bind a compatible native dense-v5 checkpoint to the resident ABI:

```bash
nfn migrate graph-to-native \
  --graph artifacts/model.json \
  --weights artifacts/model_00020000.bin \
  --output-dir artifacts/model-resident
```

Bind an exact trained `gpt2_moa` checkpoint through its metadata JSON, never
through the bare dense-v5 sidecar:

```bash
nfn migrate graph-to-native \
  --graph artifacts/gpt2-moa.json \
  --weights artifacts/model_00020000.moa.json \
  --output-dir artifacts/gpt2-moa-native
```

`model_00020000.moa.json` must use
`neuralfn.native_dense_moa.inference_checkpoint` v1 and sit beside the named
`model_00020000.bin` and empty `DONE_00020000`. It binds the model size/hash and
byte-identical source graph, declares `preset=gpt2_moa` and
`checkpoint_kind=trained_dense_v5`, and records one activation selected from
the canonical `[gelu, relu, silu, relu2]` candidates plus a positive interval.
Migration rehashes both graph and model and retains the selection contract in
the Native Execution manifest. It also copies the exact validated metadata as
owner-only `model.moa.json` and records that copy's path, size, and SHA-256.

Migrate a self-describing canonical LLaMA family checkpoint through its
metadata JSON (never through the raw sidecar directly):

```bash
nfn migrate graph-to-native \
  --graph artifacts/llama-graph.json \
  --weights artifacts/llama_native_family_model_00000000.json \
  --output-dir artifacts/llama-native
```

The metadata must carry the additive
`neuralfn.native_family_llama.inference_checkpoint` v2 contract emitted by a
live, full-architecture canonical `llama` training run. Diagnostic family v1
metadata and direct `.f32` paths fail closed.

Migrate an exact graph-authored standard-MoE checkpoint through its metadata
JSON:

```bash
nfn migrate graph-to-native \
  --graph artifacts/mixllama.json \
  --weights artifacts/mixllama_native_family_model_00000000.json \
  --output-dir artifacts/mixllama-native
```

The metadata must carry
`neuralfn.native_family_standard_moe.inference_checkpoint` version 1, identify
`moe`, `mixllama`, or `mixllama-fast`, contain a safe float32 sidecar and DONE
marker, and bind its exact source-graph SHA-256. The inspector re-derives all 17
geometry fields, routing semantics, and the canonical ordered tensor table.

`--output-dir` is required for both normal and dry-run calls and must not
already exist or be a symlink. A dry run emits the same manifest/report payload
as JSON on stdout but creates no directory. Compatible migrations return exit
code 0. Incompatible graphs return exit code 2 with node-specific issues and no
output directory.

Graph-only, native `.bin`, dense-MoA `.json`, native-family LLaMA `.json`, and standard-MoE `.json`
migration stay on the lean SDK path and do not import Torch, NumPy, or
graph-analysis packages.
Legacy `.pt` input requires a separately installed PyTorch only inside the
isolated migration worker.

## Artifact layout

A successful materialized migration creates a new owner-only directory:

```text
model-native/
├── native-execution-manifest.json
├── compatibility-report.json
├── weights.bin                       # only for legacy .pt input
├── model.bin                         # compatible dense-v5 or validated MoA metadata input
├── model.moa.json                    # exact copied MoA metadata, MoA input only
└── model.f32                         # canonical LLaMA or standard-MoE JSON input
```

The migration never overwrites an existing destination and never modifies the
source graph or checkpoint.

`inspect_native_moa_checkpoint()` in `neuralfn.native_moa_checkpoint` validates
the exact `model_XXXXXXXX.moa.json` filename, matching dense-v5 sibling and
empty DONE marker, model size/hash and geometry, source graph filename/hash/
byte identity, canonical candidate order, selected activation, and positive
interval. A bare MoA `.bin`, stale graph, changed selection, missing DONE
marker, or mismatched sibling fails before an artifact is created.

For graph/runtime differential work, the public module
`neuralfn.native_moa_graph_runtime` can load that exact artifact into the
authored Torch graph without rewriting the graph:

```python
import torch

from neuralfn.native_moa_graph_runtime import load_native_moa_graph_runtime

runtime = load_native_moa_graph_runtime(
    "gpt2-moa.json",
    "gpt2-moa-native",
)
token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
logits, = runtime.forward(token_ids)
print(runtime.binding.selected_activation)
```

The module exports `NATIVE_MOA_GRAPH_RUNTIME_PROFILE`,
`NativeMoaGraphRuntime`, `NativeMoaGraphRuntimeBinding`,
`NativeMoaGraphRuntimeError`, and `load_native_moa_graph_runtime()`. The loader
requires the original byte-identical graph, rehashes the dense checkpoint and
copied metadata, accounts for every tensor-table entry, imports packed QKV and
padded-vocabulary weights, and replaces only canonical
`model/block_N/mlp/gelu` stages with the committed tanh-GELU, ReLU, SiLU, or
ReLU-squared selection. It is a Torch parity/debug runtime, not the resident
serving path. It proves internal artifact consistency, not authenticity against
a coordinated rewrite of every artifact file/hash; the low-level C++ resident
loader also continues to trust the manifest selection rather than rehashing
`model.moa.json`.

**Breaking change:** MoA artifacts migrated before the copied-metadata
descriptor (`metadata_artifact_path`, `metadata_nbytes`, and
`metadata_sha256`) no longer satisfy the effective MoA capability gate.
Re-run `nfn migrate graph-to-native` from the original graph-bound
`model_XXXXXXXX.moa.json` bundle to produce `model.moa.json` and the current
manifest before loading or serving it.

`inspect_native_family_llama_checkpoint()` in
`neuralfn.native_family_checkpoint` is the dependency-light source inspector.
It requires a safe relative sidecar path and the matching DONE marker, checks
the exact sidecar size and SHA-256, re-derives canonical RMSNorm/RoPE/GQA/
SwiGLU geometry, and proves that tensor records form one exact contiguous
little-endian float32 layout. Its checkpoint descriptor uses format
`neuralfn.native_family_llama.f32.v1`, artifact path `model.f32`, and explicit
geometry/semantics maps. The source `.f32` sidecar is copied only after these
checks and the copy is rehashed before migration returns.

`inspect_native_family_standard_moe_checkpoint()` in
`neuralfn.native_moe_checkpoint` applies the same contained-path, DONE-marker,
size, whole-file SHA, and copy-before-load rules to standard MoE. It additionally
requires root embedding/final-norm/head tensors followed by each layer's two
norms, Q/K/V/out projections, router, packed gate/up experts, and expert down
projection. Its descriptor format is
`neuralfn.native_family_standard_moe.f32.v1`. The checkpoint and graph must
agree on floating `mlp_multiplier`, `multiple_of` (`None` maps to native `0`),
expert count, top-k, and router auxiliary-loss coefficient.

`native-execution-manifest.json` uses schema
`neuralfn.native_execution_manifest`, version 1. Its main fields are:

- `source_graph`: resolved source path, SHA-256 fingerprint, and an explicit
  `serialization_changed: false` marker.
- `model`: model family/class/objective, source template spec, text-generation
  classification, and cache-policy classification.
- `topology`: deterministic graph/node/edge paths, resolved inline subgraphs,
  and all root variant-library entries.
- `tensors`: stable native names, source names, dtype, shape, byte offset,
  length, byte order, explicit memory `layout`, role, and per-tensor SHA-256.
- `tokenizer`, `chat_template`, `context_limits`, and `stop_tokens`: serving
  metadata carried from the graph when authoritative values exist. Missing
  values stay explicit and produce warnings; they are not invented.
- `kernel_abi`: required Tile and strict-math ABI versions. A compatible bound
  dense-v5 (including validated MoA), canonical LLaMA, or exact standard-MoE artifact records resident ABI version 1 with
  status `ready`; only supported dense even-head geometry also records
  `turboquant_cache` ABI version 1 and backend `cpu-reference-packed`.
  Resident-ready dense-v5, canonical-LLaMA, and exact standard-MoE
  lossless-cache artifacts separately record additive `session_prefix_cow`
  version 1/status `ready`, operation `fork_session`, and the corresponding
  dense, LLaMA-GQA, or standard-MoE-GQA full-cache profile.
  Supported reviewed dense-v5 TurboQuant artifacts separately record
  `session_prefix_cow_cpu_turboquant` with version 1/status `ready`, operation
  `fork_session`, profile
  `dense-cpu-turboquant-mse-qjl-packed-kv-final-hidden-v1`, and backend
  `cpu-reference-packed`. This record is deliberately distinct from both the
  full-cache COW ABI and the Tile-CUDA attention ABI.
  Graph-only and generic `.pt` artifacts explicitly retain `not_implemented`.
- `checkpoint`: source and target formats, source/target hashes, output length,
  safe-loader flags, and preserved JSON-safe checkpoint metadata when weights
  were supplied.
- `primary_checkpoint_variant` and `checkpoint_variants`: optional additive
  Muse Glimmer weight-profile catalog. Each selected entry repeats the existing
  executable checkpoint fields exactly and adds an authenticated kernel and
  memory contract; the top-level `checkpoint` must equal the primary entry.
- `companion_checkpoints` and `speculative_decoding`: optional exact
  assistant/mmproj/adapter descriptors with target/tokenizer/processor
  allowlists and the transactional DFlash profile. These fields round-trip in
  manifest v1 and do not reinterpret legacy one-checkpoint artifacts.
- `session_state_kinds`: a classification of the state a future adapter would
  need, such as `kv`, `latent_kv`, or hybrid attention/recurrent state. This is
  not evidence that the state is implemented.
- `capabilities`: conservative booleans for structural Native IR, training and
  architecture persistence, native forward, resident inference, cache modes,
  serving, tools, and structured output.

The CPU TurboQuant COW record is artifact evidence, not sufficient by itself.
At load time the SDK also requires the binding boolean
`session_prefix_cow_cpu_turboquant=true`, the exact profile in the binding's
`session_prefix_cow_abi.profiles` inventory, and a callable `fork_session`.
Only a dense source session with effective `turboquant`, an `mse-3.5` or
`qjl-3.5` profile, and CPU attention may consume it. Tile-configured models and
Tile-CUDA sessions fail closed. The packed K/V and lossless final-hidden stores
share whole capacity until the first append; truncate/reset stays logical, and
a failed or cancelled append restores sharing and ownership/detach telemetry.
This does not advertise a serving prefix cache, LRU, revision/CAS protocol, or
Tile device-state COW.

Structured/tool capability metadata is additive and exact. A resident-ready
artifact sets `structured_output=true` and emits
`kernel_abi.structured_output` version 1 only when its carried tokenizer
metadata contains exactly:

```json
{
  "constrained_decoding": {
    "version": 1,
    "profile": "json-schema-ascii-byte-greedy-v1",
    "token_selection": "current_logits_exact_prefill"
  }
}
```

Source graphs carry that object under
`torch_config.tokenizer_manifest`. Function support additionally requires
`torch_config.artifact_metadata.tool_template` (or the equivalent tokenizer
metadata) to contain exactly:

```json
{"version":1,"profile":"responses-forced-function-call-v1"}
```

The compiler then emits `capabilities.function_tools=true` and the matching
`kernel_abi.function_tools` version-1 feature record. The ready records are
normalized exactly as follows:

```json
{
  "structured_output": {
    "version": 1,
    "status": "ready",
    "profile": "json-schema-ascii-byte-greedy-v1",
    "token_selection": "current_logits_exact_prefill"
  },
  "function_tools": {
    "version": 1,
    "status": "ready",
    "profile": "responses-forced-function-call-v1",
    "structured_output_profile": "json-schema-ascii-byte-greedy-v1"
  }
}
```

Boolean/floating
versions, unknown contract fields, mismatched profiles, graph-only artifacts,
or missing resident proof stay `not_implemented`. These manifest claims are
still only one side of a joint gate: serving also requires the binding's
read-only-logits/exact-prefill primitive and a byte-exact tokenizer startup
preflight.

`compatibility-report.json` uses schema
`neuralfn.native_compatibility_report`, version 1. It contains graph validity,
compatibility, the source fingerprint, stable issue paths/codes, unsupported
node paths, warnings, optional tensor mappings, the capability booleans, and
the complete capability proof with evidence and missing gates.

## Preflight and checkpoint isolation

Migration deliberately orders work as follows:

1. Parse raw graph JSON and validate nodes, ports, edges, graph interfaces, and
   variant-library structure.
2. Compare every module against the explicit Native IR lowerer registry.
3. Reject arbitrary non-boundary Python function neurons. The serialized
   payload compiler does not construct `NeuronDef` objects, so preflight cannot
   execute graph-supplied source.
4. Resolve the variant library and lower a deep copy to deterministic topology.
5. Only after the report is compatible, optionally launch the `.pt` worker.
6. Materialize the artifact only after every manifest, tensor, and checksum
   validation succeeds.

The `.pt` worker runs Python isolated mode and calls
`torch.load(..., map_location="cpu", weights_only=True)`. It emits only JSON
plus a raw contiguous tensor bundle. The parent revalidates the worker schema,
64-byte tensor alignment, non-overlapping offsets, bounds, bundle length, and
SHA-256 hashes before writing the final artifact.

Unknown module types and custom Python functions fail closed. Registering a new
Python/Torch builtin elsewhere does not automatically make it Native IR
compatible; the explicit native lowerer registry must be reviewed and updated.

## Python migration API

All migration types and helpers are lazy top-level exports:

```python
from neuralfn import (
    NativeCompatibilityReport,
    NativeExecutionManifest,
    NativeLoweringIssue,
    NativeMigrationResult,
    NativeTensorSpec,
    compile_graph_to_native_manifest,
    compile_native_graph_payload,
    migrate_graph_to_native,
)
```

File migration mirrors the CLI:

```python
result = migrate_graph_to_native(
    "artifacts/model.json",
    weights_path="artifacts/model.pt",  # omit for graph-only migration
    output_dir="artifacts/model-native",
    dry_run=True,
)

if not result.report.compatible:
    for issue in result.report.issues:
        print(issue.path, issue.code, issue.message)
```

`NativeMigrationResult` contains the optional manifest, compatibility report,
materialized output path (or `None`), and dry-run flag. `to_dict()` returns the
same JSON-safe shape emitted by the CLI.

Compile an in-memory graph without mutating it:

```python
manifest = compile_graph_to_native_manifest(graph)
payload = manifest.to_dict()
round_tripped = NativeExecutionManifest.from_dict(payload)
```

`compile_graph_to_native_manifest()` validates and resolves a deep copy. It is
appropriate when the caller already owns a validated `NeuronGraph`; file
migration adds raw-byte fingerprinting, safe source preflight, checkpoint
conversion, reporting, and exclusive materialization.

For an already parsed serialized payload, use the source-safe,
dependency-light compiler directly:

```python
manifest = compile_native_graph_payload(graph_json_mapping)
```

This validates raw nodes, ports, edges, and variants without constructing
`NeuronGraph` or `NeuronDef`, so it cannot execute serialized source and does
not import Torch, NumPy, or NetworkX.

The public data types are frozen dataclasses:

- `NativeTensorSpec` describes one tensor in a native bundle, including its
  explicit `layout` (`row_major` by default), and supports `to_dict()` /
  `from_dict()`. Older mappings without `layout` read as `row_major`; new
  manifests always emit the field.
- `NativeExecutionManifest` supports `to_dict()`, `from_dict()`, and
  `load(path)` with strict schema/version checks.
- `NativeLoweringIssue` carries a path, stable code, message, severity, node
  kind, and operation.
- `NativeCompatibilityReport` exposes `compatible`,
  `unsupported_node_paths`, and `to_dict()`.
- `NativeMigrationResult` exposes `to_dict()` for CLI/API handoff.

Schema constants such as `NATIVE_EXECUTION_MANIFEST_VERSION` and
`NATIVE_TENSOR_BUNDLE_FORMAT` are available from `neuralfn.native_ir`.

## Capability registry API

The registry is deterministic, dependency-light, and deliberately separates
structural lowering, trainer registration, architecture persistence, native
forward proof, resident inference, caching, and serving:

```python
from neuralfn import (
    NativeCapabilityProof,
    NativeLoweringSpec,
    NativeTrainerSpec,
    capability_proof_for,
    classify_native_model,
    native_lowering_specs,
    native_trainer_specs,
    registered_native_module_types,
)

lowerers = native_lowering_specs()
trainers = native_trainer_specs()
module_types = registered_native_module_types()

classification = classify_native_model(manifest.model, manifest.topology)
proof = capability_proof_for(manifest.model, manifest.topology)
checkpoint = manifest.checkpoint or {}
resident_formats = {
    "neuralfn.native_dense_gpt.v5",
    "neuralfn.native_family_llama.f32.v1",
    "neuralfn.native_family_standard_moe.f32.v1",
}
if proof.resident_inference_proven and checkpoint.get("format") not in resident_formats:
    # Topology proof alone does not bind a checkpoint. Graph-only manifests,
    # generic .pt bundles, and diagnostic family metadata keep the effective
    # artifact capability false.
    assert manifest.capabilities["resident_inference"] is False
```

- `NativeLoweringSpec` records the explicit module opcode, supported IR
  versions, and state/config preservation contract.
- `NativeTrainerSpec` records the family target and strongest proved forward
  status. `diagnostic-transition-only` is not model inference.
- `NativeCapabilityProof` reports classified state kinds, encountered and
  unsupported module types, evidence, and every missing gate.
- `classify_native_model()` returns deterministic model, objective, family,
  state, module, and cache-policy classification without probing binaries or
  the filesystem.
- `capability_proof_for()` combines classification with the explicit lowering
  and trainer registries and leaves every unproved runtime gate false.

## Graph-authored native training preflight

The shared planner exposes the source-safe migration result, exact adapter
compatibility, and canonical family target without pretending the current
selector-driven dense trainer consumes Native IR:

```python
from neuralfn import plan_native_graph_training

plan = plan_native_graph_training("model.json")
print(plan.training_selector, plan.native_target)
assert plan.trainer_consumes_native_ir is False
if plan.execution_ready:
    assert plan.graph_preflight_enforced is True
    assert plan.training_compatible is True
```

Pass `artifact_dir=...` and `materialize=True` to exclusively create the Native
IR manifest/report directory during preflight. A compatible materialization
adds an exact-byte `source-graph.json` snapshot and
`native-training-plan.json`; an exact `gpt2_diff` materialization also adds
`native-training-proof.json`. Existing paths are never overwritten.
`NativeGraphTrainPlan` also contains `launch_graph`, `trainer_family`,
`graph_preflight_proof`, `training_selector`, `native_target`, `adapter_mode`, canonical
`trainer_arguments`, the complete compatibility report and manifest,
node-specific `training_issues`, artifact metadata, proof flags, and stable
blockers.

All 67 shipped text presets still lower structurally and route to a registered
family for diagnostic reporting. Thirteen exact profiles currently have reviewed
execution-ready production adapters: `gpt2`, `gpt2_megakernel`, `gpt2_moa`,
`gpt2_qknorm`, `gpt2_softcap`, `gpt2_stable`,
`gpt2_zloss`, canonical `llama`, and its exact compile-runtime alias
`llama_fast`, plus standard-MoE `moe`, `mixllama`, and `mixllama_fast`, and
proof-bound `gpt2_diff` training. Those
plans set `execution_ready=True` only after exact
topology, configuration, geometry, and edge-transform validation. LLaMA also
requires the proved RMSNorm/RoPE/MHA-or-GQA/dense-attention/gate-first-SwiGLU,
biasless, dropout-zero, untied contract and rejects embedded state that its
trainer would ignore. The compile alias normalizes `native_template_name` and
checkpoint identity to `llama` while preserving its source selector/runtime in
provenance. Standard-MoE plans additionally require standard softmax routing,
top-k renormalization, auxiliary-loss balance, no shared experts, exact graph
edges, and finite float32-representable coefficient/width geometry. Their
compile alias retains `source_runtime=compile` while using native template
`mixllama-fast`. The other 53
profiles report precise incompatibilities; a registered diagnostic transition
sampler is never treated as architecture-persistent training.

`gpt2_diff` becomes locally training-ready only after the trusted planner
validates its exact serialized configuration and active topology from the
source-byte identity used by Native IR. Materialization emits a canonical
`neuralfn.native_graph_training_proof` v1 record containing source, validator,
shape, and geometry identities. The native boundary requires graph,
fingerprint, and proof paths together before plan/Tile/CUDA work and carries the
proof contract SHA into strict resume. This unkeyed SHA is local-handoff
integrity, not authenticity; callers that can forge both graph and proof remain
inside the trust boundary.

The low-level dense trainer
now owns one learned FP32 lambda per layer, includes its real gradient in
clipping and AdamW, and writes source-bound differential parameter/optimizer
sidecars plus strict metadata without changing dense-v5. The metadata schema is
`neuralfn.native_gpt2_diff.training_checkpoint` version 2, while the unchanged
five binary formats retain `checkpoint_kind=trained_dense_v5_plus_diff_v1`.
Continuation-only resume validates the source graph, DONE-gated binary bundle,
training-shard identity, counters/sampler position, seed, accumulation shape,
optimizer/LR horizon, LM-head chunk, effective BF16 routes, and a canonical
numerics profile of supported effective routes before Tile/CUDA/H2D. Stable no-follow
descriptors bind training reads; validation shards are excluded. Export is exclusive and
DONE-last/fsynced, but not atomic-rename publication; ancestor symlinks and
metadata-smoke output are outside that guarantee. It rejects before Tile load
or mutation unless packed QKV is enabled, sequence length is at least 16, head
geometry is divisible and even, BF16 QKV-gradient handoff is enabled, and the
learned-lambda forward, backward, and workspace-release Tile ABI symbols exist. The selector
uses only the learned-lambda ABI; the retained legacy fixed-lambda ABI remains
outside it with rounded-output/non-layer-local backward correctness debt. The
generic registry adapter nevertheless reports downstream persistence/resident
proof false. Exact graph-training plans locally report
`architecture_persistence_proven=True` and `execution_ready=True` only after
proof issuance; Native IR manifests, migration, and resident inference remain
false because those consumers do not yet inspect, copy, or execute the additive
differential state.
Supplying `.pt`, raw `.bin`, or metadata-v2 `.diff.json` weights with this graph
raises the explicit unimplemented-bundle error before generic checkpoint
inspection or output creation. Retain the complete training bundle for a
future migration/resident consumer.

The thirteen ready graph-training adapters validate Native IR and then launch a selector-driven
trainer with the immutable source snapshot plus canonical selector and
geometry. They therefore intentionally keep `trainer_consumes_native_ir=False` while setting
`graph_preflight_enforced=True`. This distinguishes a safe graph-file adapter
from a future trainer that parses and executes Native IR itself. For canonical
LLaMA, `trainer_arguments` contains the complete graph-derived geometry and
lowercase source SHA-256 but not an inspection/training action. CLI dry-run,
plan, symbol-check, and sample operations remain non-training; a real launch
selects `--train-llama-dataset-loop`. `NativeTrainRunConfig` re-runs the planner
and requires each plan-owned argument exactly once, so a caller-provided path,
template, or digest cannot grant the adapter.

For standard MoE, `trainer_arguments` includes the exact floating
`--mlp-multiplier`, explicit `--multiple-of 0` for an unaligned graph, expert
count, top-k, one layer per expert stage, router auxiliary-loss coefficient, and
source digest. Real CLI launches select `--train-moe-dataset-loop`. The
production Tile path implements the graph formula
`E * sum(mean_rows(softmax(router_logits))^2)` and accumulates its all-expert
Jacobian gradient before the router weight backward. CPU/build/migration/
resident parity is complete for this reviewed adapter. A bounded RTX 5090 run
also completed one exact graph-authored optimizer step, emitted and inspected
the strict 12-tensor checkpoint, migrated it losslessly, loaded the rebuilt
resident extension, and generated raw tokens. That result covers the reviewed
standard-MoE cluster only; other families and performance remain separate open
gates. The reviewed-dense resident Tile-CUDA TurboQuant slice has its own
independent proof and does not extend to standard MoE.

Canonical LLaMA v2 checkpoints may include
`training.source_graph = {filename, sha256, byte_identity_verified}`. The C++
trainer checks that byte identity before dataset/CUDA setup and again during
checkpoint writing. Server checkpoint discovery compares the digest with the
materialized plan, and `migrate_graph_to_native()` compares it with the graph
being migrated. `byte_identity_verified` is deliberately narrower than the
Python plan's `graph_preflight_enforced`: the family binary does not parse or
independently validate graph topology.

## What this does not provide yet

Native IR v1 remains an artifact and compatibility boundary. A compatible
reviewed dense graph plus dense-v5 `.bin` now becomes a directly loadable resident
artifact with a lossless K/V cache, native packed CPU TurboQuant cache for
supported head geometry, and lean serving capability. The supported preset set
is `gpt2`, `gpt2_megakernel`, `gpt2_moa`, `gpt2_zloss`, `gpt2_qknorm`,
`gpt2_stable`, and `gpt2_softcap`, with explicit resident fields and exact active port-level
dataflow proof for QK RMS normalization and logit softcap. Every Q/K transform
must sit between its layer's `q_heads`/`k_heads` nodes and SDPA; positive
softcap must sit between the tied head and token-loss consumer. Python proof and
the C++ loader both reject disconnected or bypassed transform nodes. MoA also
requires the source-bound sibling metadata/DONE contract and fixes inference to
the recorded activation; a bare dense-v5 MoA `.bin` is not supported. This does
not cover differential/modern dense variants,
GQA/RoPE/sparse-window cache shapes, non-dense or hybrid state adapters,
whole-model GPU execution, or the
broader tool/structured-output or multimedia Responses surface beyond the
bounded flat-schema/single-forced-function profile. Generic `.pt` tensor bundles are structural
migrations only. The legacy Python `InferenceCache` compatibility wrapper is
separate from Native IR and is not proof of retained native cache state.

These limitations are encoded in the manifest and report rather than hidden
behind fallback behavior. Consumers should refuse a requested runtime feature
when its corresponding capability is false.
