# Inference and Export

After training a torch-backed graph, NeuralFn provides tools for saving weights,
quantizing checkpoints, and running autoregressive generation through the
legacy Torch compatibility wrapper.

The inference module itself is import-safe on the lean native/core SDK: imports
such as `from neuralfn.inference import export_to_pt, InferenceCache` do not
load Torch. The actual `.pt` checkpoint operations, quantized export/import,
semantic table helpers, and `InferenceCache` execution are legacy
Torch-backed workflows and require PyTorch to be installed explicitly.

For an application-owned generation loop, import the Torch-free controls from
`neuralfn.inference_policy`, call
`prepare_inference_process_environment()` before Torch/CUDA initialization,
validate the requested temperature, and keep the complete forward plus token
selection inside `inference_execution(temperature, torch_module=torch)`. Exact
zero requires a Torch-only FP32 compiled graph with autocast disabled; validate
the graph before compilation and the compiled floating state after loading via
`validate_strict_graph_support()` and `validate_strict_compiled_graph()`.
Positive temperatures retain the normal optimized backend.
Strict policy enforcement uses PyTorch's hierarchical `fp32_precision="ieee"`
controls when the complete API is available and otherwise uses the legacy
TF32-disable controls; it does not mix the two API generations.

## Native Execution IR migration

Native Execution IR v1 is the ahead-of-time boundary for converting an
authoring graph into a deterministic native artifact contract. It resolves
nested subgraphs and variant-library entries on a deep copy, so the original
graph object and serialized graph JSON remain unchanged.

```bash
nfn migrate graph-to-native \
  --graph model.json \
  --output-dir model-native \
  --dry-run

nfn migrate graph-to-native \
  --graph model.json \
  --weights model.pt \
  --output-dir model-native
```

Graph preflight always completes before optional checkpoint access. Graph-only
migration is Torch-free; `.pt` conversion runs in an isolated worker with
`weights_only=True`. The destination must be new. A materialized artifact
contains a versioned manifest and compatibility report plus either an aligned
generic `weights.bin` for legacy `.pt` input or a fingerprinted `model.bin` for
a compatible native dense-v5 checkpoint. Exact `gpt2_moa` instead requires the
source-bound `model_XXXXXXXX.moa.json` beside its named dense-v5 `.bin` and
empty DONE marker; migration retains the selected activation/canonical
candidate set/positive interval contract, copies the model as `model.bin`, and
copies the exact metadata as owner-only `model.moa.json` with path/size/hash
fields in the manifest.
Canonical native-family `llama`
inference-checkpoint v2 metadata is also Torch-free: migration validates its
full float32 tensor contract and copies the sidecar as `model.f32`; direct
`.f32` input and diagnostic family metadata fail closed.

The current IR structurally lowers all 67 shipped text presets, but this does
not mean that every family can execute or serve natively. Seven reviewed dense
GPT preset topologies (`gpt2`, megakernel, MoA, z-loss, QK-norm, stable, and
softcap) paired with a compatible dense-v5 `.bin` have the current
resident/lossless-cache/lean-serving proof. MoA must be migrated through its
metadata JSON, which binds the byte-identical source graph and freezes one of
`gelu`, `relu`, `silu`, or `relu2`; a bare MoA `.bin` is insufficient.
Canonical `llama` and its exact
compile-runtime alias `llama_fast`, paired with the normalized `llama` v2
float32 checkpoint, have a separate resident RMSNorm/RoPE/GQA/SwiGLU/untied-
head forward and lossless-cache proof; its model ABI is serving-capable, but
text serving still requires supported tokenizer/chat metadata. The exact
standard-MoE `moe`, `mixllama`, and `mixllama_fast` cluster similarly
requires graph-bound v1 metadata and proves real top-k-routed expert forward,
lossless cache, raw-token SDK/CLI inference, and lean-serving ABI; it rejects
TurboQuant and still needs tokenizer/chat metadata for text serving. Graph-only
artifacts, generic `.pt` bundles, neighboring differential/modern/JEPA
variants, and every
other family remain closed. Supported even-head dense artifacts also prove the native packed CPU
TurboQuant cache. Compatible contexts/head dimensions separately prove the
additive Tile-sidecar attention feature; an explicit SDK/CLI request can keep
model compute and encoding on CPU while running packed historical attention on
CUDA. CPU is the default, missing/stale CUDA dependencies fail closed, and
other families remain unavailable. See
[Native Execution IR](../python-sdk/native-ir.md) for the schema, Python API,
capability registry, and exact safety contract.

Muse Glimmer uses a pinned family converter rather than the generic graph plus
`.pt` route. Convert BF16 with `nfn migrate muse-glimmer-to-native`, or bundle
the canonical official Dynamic/17GB GGUF mains and optional DFlash/mmproj
companions with `nfn migrate muse-glimmer-gguf-to-native`. These commands
authenticate every source/config/tokenizer/tensor hash and emit additive
checkpoint variants without changing the v1 `checkpoint` object. The exact
Glimmer resident path supports C++ CPU and whole-model CUDA **text** execution,
packed weights without whole-model dequantization, hybrid local/global KV,
DFlash, and strict load-time weight selection:

```bash
nfn infer \
  --checkpoint artifacts/glimmer-kquant \
  --runtime native-cuda \
  --weight-precision auto \
  --speculative-decoding auto \
  --companion-checkpoint dflash \
  --tile-ops-lib /absolute/path/libnfn_native_train_tile_ops.so \
  --prompt "Hello"
```

CUDA `auto` budgets current free VRAM, configured context/session cache,
enabled companions, staging, workspace, and reserve before choosing BF16,
Dynamic, or 17GB in quality order. Explicit precision never downgrades. CPU
`auto` selects the authenticated primary. Full-BF16 and packed-mmproj vision
are CPU-only; a CUDA request with mmproj fails before load because
`vision_cuda=false`.

When comparing the trained MoA choice with the authored graph, use
`load_native_moa_graph_runtime()` from
`neuralfn.native_moa_graph_runtime`. It requires the original byte-identical
graph and current migrated artifact, verifies the graph/model/metadata/tensor
hash relationships, imports every dense-v5 tensor, and overlays only the
canonical per-layer activation stage with the committed selection. This is a
Torch parity/debug path; serving still uses the resident artifact contract.
Older migrated MoA directories without `model.moa.json` must be remigrated from
their original graph-bound metadata bundle.

## Resident inference contract

`neuralfn.native_inference` provides additive model/session coordination for an
in-process binding. It loads immutable model state once, isolates mutable
token/RNG/cache state per session, synchronizes the exact longest common
prefix, exposes token callbacks and cancellation, and rejects unproved cache
modes. It never falls back to a subprocess. The first binding contains real
dense GPT-family bf16-v5, canonical LLaMA float32, exact standard-MoE float32
CPU resident engines, and exact Muse Glimmer BF16/K-Quant CPU plus CUDA text
engines.
`auto`/`full` use preallocated lossless K/V state with exact recomputation
parity, while `off` remains the
full-prefix-recompute oracle. Reviewed dense artifacts with even head dimensions
also support the native packed CPU `mse-3.5`/`qjl-3.5` cache behind joint cache
ABI proof. Compatible `.bin` migration stamps the ready ABI and checkpoint
fingerprint. QK-norm is per-head RMS normalization at `eps=1e-6`; positive
softcap is applied to tied-head logits. Migration and load independently require
the explicit dense ABI fields and canonical `q_heads`/`k_heads -> qk_norm ->
SDPA` and `tied head -> softcap -> logits consumer` port chains; a present but
bypassed transform does not receive resident capability. MoA independently
requires its source/model hashes, empty DONE marker, canonical candidates,
selected activation, and positive interval; the resident prefill/decode/cache
paths do not reprobe candidates. Canonical LLaMA load
instead requires the exact v2 tensor layout and reviewed topology,
independently rehashes the loaded float image, and rejects TurboQuant. The
standard-MoE loader independently requires graph SHA, preset/runtime,
floating width, router semantics/coefficient, tensor layout, and whole-file
hash before using its top-k-routed expert engine. Do not describe the legacy
dense, LLaMA, standard-MoE, or Tile TurboQuant paths as whole-model CUDA; only
the separately gated Glimmer text runner has that contract. None is coverage
for differential/modern and other family adapters. The reviewed-dense CPU path is the default; its explicitly
configured hybrid Tile backend moves only compressed historical attention to
CUDA. See
[Resident Native Inference SDK](../python-sdk/native-inference.md) for the
public types and binding contract.

For a resident-ready dense-v5, canonical-LLaMA, or exact standard-MoE artifact
in `full` mode, check
`model.capabilities.session_prefix_cow` before calling
`model.fork_session(source, token_count=..., seed=...)`. The additive v1
feature uses an exact format-specific profile, shares native K/V plus
final-hidden storage, and detaches the first writer, while histories, RNG,
cancellation, and lifecycle remain independent.
For a reviewed dense-v5 source whose effective cache is `turboquant`, whose
profile is `mse-3.5` or `qjl-3.5`, and whose attention backend is CPU, instead
require `model.capabilities.session_prefix_cow_cpu_turboquant`. Its exact ABI
profile is `dense-cpu-turboquant-mse-qjl-packed-kv-final-hidden-v1` with backend
`cpu-reference-packed`; artifact capability/ABI, binding boolean/profile
inventory, and `fork_session` must all agree. The child shares the packed K/V
bytes and lossless final-hidden allocation. The first parent/child
prefill/decode append copies whole capacity before publishing the new stores;
logical truncate/reset does not detach. Tile-CUDA-configured models and
sessions reject the fork.
Failed or cancelled appends restore the pre-call shared storage and detach
telemetry, so a retry starts from the same ownership state.
Model shutdown is a lifecycle fence: registered pre-fence children are drained,
while later forks and newly admitted session work fail before binding mutation.
Do not use a duplicate `close()` call as a completion wait; duplicate/reentrant
closes return immediately to avoid lock-order deadlocks.
Do not use the name COW for token replay or exact-prefix prefill. These are
session-fork primitives and remain unimplemented for nonstandard MoE or Tile
device state. The standalone server now layers a separate optional
scope-isolated LRU and schema-v4 conversation revision/CAS over the proven
lossless-full and dense CPU-TurboQuant forks. That serving layer still verifies
the exact rendered token LCP; it does not reinterpret replay as native COW.

For process-local use, `nfn infer --checkpoint ARTIFACT` detects the Native
Execution manifest before legacy checkpoint dispatch. TTY sessions default to
role-message transcripts and reuse resident prefixes; non-TTY use remains a
single request. Canonical LLaMA supports tokenizer-free non-interactive
`--prompt-tokens`, rendering generated token IDs when no text codec is present;
text/interactive use still requires a supported artifact tokenizer. See
[Resident Native Artifact CLI Inference](../native-cli-inference.md).

## Native inference serving

Install `.[serve]` and route an independently resident-ready artifact through
the isolated server:

```bash
nfn infer --checkpoint artifacts/model-native --serve \
  --chat-template plain_roles \
  --state-db ./native-inference-state.sqlite3 \
  --prefix-cache-capacity 64
```

That explicit `plain_roles` fallback is suitable for the ordinary text routes
but deliberately disables structured/function capability. For the constrained
profile, the artifact itself must carry the supported renderer and serving
must use the default `--chat-template auto` (or state it explicitly).

Startup validates presentation metadata, context limits, authentication,
artifact proof, binding ABI, and cache mode before the socket can bind. The
server keeps one model resident, creates an isolated session per request, and
bounds admission around one generation worker. It implements Models and
bounded Chat Completions, including genuine token SSE ending in `[DONE]`.
Chat is text by default; a jointly proven CPU Muse Glimmer vision artifact can
also accept bounded base64 image data URLs.
With `--state-db`, the same isolated server mounts the bounded text Responses
and Conversations subset, scope-bound local compaction, durable background
jobs, and semantic Responses SSE.
`--prefix-cache-capacity N` is a separate, default-off foreground Responses
optimization. A positive entry count requires that state store and exact
effective full-cache COW or reviewed dense CPU-TurboQuant COW. It rejects
Tile-CUDA. Stored `previous_response_id` and `(conversation, revision)` aliases
identify candidates, after which the server still computes and verifies the
exact token LCP. Only stored completed/incomplete outcomes publish after their
durable terminal transaction. Failed, cancelled, non-stored, and background
responses do not enter the LRU; `store: false` may hit an existing parent but
cannot admit its result. Background and Chat stay cold. Reported
`cached_tokens` is bounded by the exact LCP and native cached rows, while
`cache_write_tokens` counts only newly written prompt rows, never decode.
Restart is cold because native sessions remain process-local.

Schema v4 adds a monotonic conversation-item revision. Preparation snapshots
items plus revision transactionally; completion atomically commits response
terminal/output state, conversation rows, and the revision CAS before cache
publication. A concurrent item mutation yields `conversation_conflict` (HTTP
409 for buffered work and semantic `response.failed` for an open
stream/background response) with no stale admission. Terminalization also
revalidates the complete previous-response lineage; deleted or changed ancestry
fails with `response_lineage_conflict` before output or cache publication.
Version-1/2/3 stores migrate in place with existing conversation history at
revision zero. Back up before the first v4 open when rollback matters, because
older binaries reject v4. An already queued legacy conversation job has no
historical revision snapshot and fails with
`conversation_snapshot_unavailable`; legacy previous-response-only work must
reconstruct a completed/incomplete lineage or fail with
`response_lineage_unavailable`, and remains subject to finish-time
revalidation. Deleting a response, conversation, or item conservatively purges
the whole API-key cache scope and fences old leases. Cache byte metrics are
per-session capacity observations, not unique physical bytes. The
cache-enabled state database must have one owning service process; out-of-band
raw-store or second-process mutation cannot participate in its process-local
transition lock/epoch. The HTTP app alone owns the combined cache lease and
durable-finish lifecycle; direct public `execute()` plus `finish()` stays cold.
Shutdown awaits the background and tracked foreground drivers, drains the
queue, then closes cache, model, and state in that order.
A separately proven Responses profile adds buffered strict flat JSON-schema
output and one forced client-executed function call/result continuation. It is
available only when Native Execution metadata, the resident binding, the exact
artifact-selected chat template, and a byte-exact tokenizer preflight all
agree. It never executes client functions. Constrained requests are greedy,
stored, foreground, and non-streaming. The later client-result continuation is
ordinary text generation but stays stored, buffered, foreground, and uses
disabled truncation. General/parallel/hosted tools, nested or
array schemas, Chat Completions tools, multimodal input, and batching remain
deliberately unsupported. Compatible dense-v5
migrations default to the lossless full cache; generic `.pt`, graph-only, and
unimplemented artifacts fail the resident gate. Canonical LLaMA has a resident
model ABI but still fails server startup without supported presentation
metadata; explicit `turboquant` is available only for reviewed-dense artifacts.
It defaults to the packed CPU cache and accepts the separately gated
`tile-cuda` attention backend when a strict sidecar is supplied. See
[Native Inference Serving API](../rest-api/native-inference-serving.md).

The separate dependency-free `TurboQuantReferenceCodec` implements the
paper-aligned MSE/QJL oracle, including deterministic rotation, Lloyd-Max
codebooks, norms, and packed mixed-bit indices. It leaves the legacy graph
`kv_quant_pack/unpack` contract unchanged. The reviewed-dense CPU binding now
attends directly over packed historical rows and matches the oracle's packed
bytes and numerical operations for both profiles. The explicit hybrid CUDA
backend consumes the same bytes and has live SDK dispatch proof; this does not
prove non-dense support or a performance improvement. See
[TurboQuant Portable Reference](../python-sdk/turboquant-reference.md).

## Interactive transcript prompts

TTY graph inference defaults to `--chat-mode transcript`; choose
`--chat-mode stateless` or enter `/mode stateless` for independent prompts.
`--system-prompt TEXT` is retained as a leading instruction. The
`--chat-template auto|plain_roles|PATH` selector prefers a tokenizer/artifact
template, accepts a data-only file with `{{messages}}` and optional
`{{assistant_prompt}}` markers, and otherwise warns before using the CLI-only
plain-role renderer.

Transcript prompts reserve `--max-new-tokens` before rendering. When necessary,
the oldest complete conversation groups are removed while leading
developer/system instructions and the newest user/tool group stay mandatory.
An oversized mandatory remainder is an error. Configured role/EOS text
delimiters are stripped from displayed/stored assistant output. CLI history is
process-local, `/reset` retains the configured system prompt, and non-TTY
one-shot behavior is unchanged.

## Weight export and import

### Full-precision export

```python
from neuralfn.inference import export_to_pt, import_from_pt

export_to_pt(graph, "model.pt")
```

This compiles the graph into a `CompiledTorchGraph`, extracts its `state_dict`, and saves it to a `.pt` file.

### Full-precision import

```python
import_from_pt(graph, "model.pt")
```

Loads the state dict, rebuilds the compiled module, calls `load_state_dict()`, then syncs the weights back into each node's `module_state` field in the graph. After import, the graph's serialized form (`to_dict()`) includes the loaded weights.

---

## Adapter-only checkpoints

For LoRA/qLoRA/RandMap fine-tuning runs, NeuralFn can save and reload only the
adapter and head parameters:

```python
from neuralfn.inference import (
    save_adapter_checkpoint,
    load_adapter_checkpoint,
    merge_adapter_into_base,
)

save_adapter_checkpoint(graph, "adapter.pt")
load_adapter_checkpoint(graph, "adapter.pt")
merge_adapter_into_base("base.pt", "adapter.pt", "merged.pt")
```

`save_adapter_checkpoint()` filters the compiled state dict to LoRA/qLoRA
adapter tensors, RandMap adapter middle/scale tensors, and value/reward heads.
`merge_adapter_into_base()` bakes LoRA deltas into a full checkpoint for
ordinary inference.

---

## Quantized export and import

### Export with quantization

```python
from neuralfn.inference import export_quantized_pt

export_quantized_pt(graph, "model_q.pt", scheme="int8")
```

Two schemes are supported:

| Scheme | Description |
|--------|-------------|
| `"int8"` | Per-channel int8 quantization with float32 scale factors. Applied to linear/projection weight tensors; token and position embeddings remain full precision. |
| `"ternary"` | Bakes weights to `{-1, 0, 1}` with a single mean-absolute scale per tensor. Designed for BitLinear / ternary_b158 models. |

Non-weight tensors (biases, norms, embeddings) are stored at full precision in both schemes.

### Import with dequantization

```python
from neuralfn.inference import import_quantized_pt

import_quantized_pt(graph, "model_q.pt")
```

Reads the quantized checkpoint, dequantizes all weight tensors back to float32, loads them into a compiled graph, and syncs back to the graph's node states. The graph operates at full precision after import -- quantization is a storage optimization, not a runtime one.

---

## InferenceCache

`InferenceCache` is a legacy compatibility wrapper around a compiled Torch
graph. Despite its historical name, the current implementation does not
populate or read retained K/V tensors across calls. For context-correct
generation, pass the full prefix on each call. It is not the resident native
session/cache API and does not establish lossless-cache or TurboQuant support.

```python
from neuralfn.inference import InferenceCache
import torch

cache = InferenceCache(graph, device="cuda")

prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
logits = cache.step(prompt)
next_token = logits.argmax(dim=-1)

prefix = torch.cat((prompt, next_token.unsqueeze(1)), dim=1)
logits2 = cache.step(prefix)

cache.reset()
```

### How it works

1. **Each call**: pass the full prefix as `(batch, seq_len)`. The wrapper runs
   the compiled graph and returns logits for the last position.
2. **Subsequent tokens**: append the selected token to the prefix and submit the
   expanded prefix again. The current wrapper recomputes the graph.
3. **Reset**: `reset()` clears the reserved compatibility dictionary. There is
   currently no populated retained K/V state to release.

### Constructor

```python
InferenceCache(graph, device=None, *, compiled=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `NeuronGraph` | required | A torch-runtime graph, typically built from a template preset. |
| `device` | `str` | from `torch_config` or `"cuda"` | Device to run inference on. |
| `compiled` | `CompiledTorchGraph` | `None` | Optional precompiled, weight-loaded model to reuse instead of constructing another graph. |

Temperature-zero generation should pass the strict FP32 compiled graph through
`compiled=` so the cache cannot bypass the selected compute policy. Existing
callers that omit the keyword retain the original construction behavior.
`SemanticInferenceCache` exposes the same keyword for semantic graph callers.

### KV cache nodes

Some graphs declare `kv_cache_read` / `kv_cache_write` module nodes or template
cache capability metadata. The current compatibility wrapper does not bind
those declarations to retained tensors across calls. Treat them as graph
metadata until an execution adapter proves and exposes the corresponding state.

### Training graph compatibility

For training graphs that take two inputs (tokens + targets), `InferenceCache.step()` automatically generates dummy target tensors so the forward pass runs without modification. The loss output is returned as-is, which can be useful for perplexity evaluation.

If you export or probe a tokenizer-backed cached dataset alias, keep the tokenizer contract intact: NeuralFn now preflights tokenizer-backed aliases before generation and stops early when the cached tokenizer vocab, shard ids, and checkpoint vocab disagree. This avoids the previous failure mode where decoding crashed after generation had already emitted out-of-range token ids.

---

## Typical workflow

```python
from neuralfn import build_gpt_root_graph, TorchTrainer, TorchTrainConfig
from neuralfn.config import build_llama_spec
from neuralfn.inference import export_to_pt, InferenceCache
import torch

spec = build_llama_spec(n_layer=4, n_embd=128, vocab_size=256)
graph = build_gpt_root_graph(model_spec=spec)

tokens = torch.randint(0, 256, (16, 64))
targets = torch.randint(0, 256, (16, 64))
trainer = TorchTrainer(graph, TorchTrainConfig(epochs=5, device="cuda"))
trainer.train(tokens, targets)

export_to_pt(graph, "llama_small.pt")

cache = InferenceCache(graph, device="cuda")
prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
generated = prompt.squeeze(0).tolist()

for _ in range(50):
    prefix = torch.tensor([generated], dtype=torch.long)
    logits = cache.step(prefix)
    next_tok = logits.argmax(dim=-1)
    generated.append(next_tok.item())

print(generated)
```

---

Next: [Datasets](datasets.md)
