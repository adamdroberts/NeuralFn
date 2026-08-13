# Resident Native Artifact CLI Inference

`nfn infer --checkpoint ARTIFACT` recognizes a Native Execution Manifest v1
directory or manifest file before the legacy native `model_*.bin` sampler and
before the graph-backed CLI. Passing the checkpoint file inside a migrated
artifact, such as `--checkpoint model-native/model.bin` or
`--checkpoint llama-native/model.f32`, selects the same
resident path only when the sibling manifest names that exact contained file
in `checkpoint.artifact_path`. A mismatched, absolute, or escaping declaration
does not claim the raw checkpoint. This path imports neither `nfn_impl` nor
Torch. It loads one `NativeInferenceModel`, creates one process-local
`NativeInferenceSession`, and never falls back to a subprocess.

An exact `gpt2_moa` artifact is created by migrating its source-bound sibling
`model_XXXXXXXX.moa.json`, not by binding the dense-v5 `.bin` alone. The
metadata must name that `.bin` and its empty `DONE_XXXXXXXX`, match the supplied
graph hash, preserve candidates `[gelu,relu,silu,relu2]`, and record one
selected activation plus a positive interval. The migrated artifact still
contains `model.bin`; its manifest carries the validated MoA contract.

The artifact must pass the resident ABI and cache capability gates accepted by
`NativeInferenceModel`. Text prompts additionally require a positive manifest
context limit and an artifact-declared tiktoken encoding. A reviewed dense-v5
artifact normally contains `model.bin`; a canonical LLaMA family artifact
contains `model.f32`:

```text
model-native/
  native-execution-manifest.json
  model.bin                         # reviewed dense-v5
  # or model.f32                    # canonical LLaMA
```

## Legacy artifact boundary

The manifest-first route does not remove existing graph-backed or graphless
Parameter Golf inference. Calls with `--graph GRAPH [--weights WEIGHTS]`,
`--checkpoint MODEL.pt`, or `--weights MODEL.pt` without a graph still forward
to the compatibility runtime after printing one deprecation warning.

For a graph-backed artifact, the warning prints a `shlex`-safe, exact migration
command using the received graph/weights paths and an adjacent
`<graph-stem>-native` destination. A graphless Parameter Golf checkpoint does
not serialize its topology, so the CLI does not guess a sibling graph. It says
that a matching NeuralFn graph is required and prints a command template with
the explicit placeholder `MATCHING_GRAPH.json`:

```bash
nfn migrate graph-to-native \
  --graph MATCHING_GRAPH.json \
  --weights artifacts/final_model.pt \
  --output-dir artifacts/final_model-native
```

Adding `--serve` or explicitly selecting TurboQuant with
`--kv-cache turboquant` / `--kv-cache=turboquant` changes this from a warning
to an exit-2 rejection before the native server or full compatibility runtime
starts. Migration of a legacy `.pt` validates and preserves a generic
graph/tensor bundle; it does not create a resident-loadable checkpoint.
Serving and TurboQuant require a separately compatible resident native
checkpoint and the normal manifest/binding capability proof below.

## One-shot use

When stdin or stdout is not a TTY, inference performs exactly one generation
and exits:

```bash
nfn infer \
  --checkpoint model-native \
  --prompt "Explain prefix caching briefly." \
  --max-new-tokens 64 \
  --temperature 0 \
  --kv-cache auto
```

`--checkpoint model-native/native-execution-manifest.json` and
`--checkpoint model-native/model.bin` are equivalent for that proven artifact.
For a canonical LLaMA migration, the equivalent exact contained path is
`--checkpoint llama-native/model.f32`.
Standalone dense-v5 `.bin` files without an exact sibling manifest remain on
the legacy token-only one-shot sampler; migrate them with their authoring graph
before requesting resident sessions or caches.

`--prompt-tokens 12,34,56` bypasses text rendering for a raw token-prefix
request. Specify either `--prompt` or `--prompt-tokens`, not both. If the
artifact has no supported text tokenizer, this non-interactive raw-token path
uses a token-ID codec, prints one warning, and renders generated output as
comma-separated token IDs. This is the proved tokenizer-free CLI route for
canonical LLaMA. Text prompts and interactive transcript rendering still need
an artifact-declared supported tokenizer; the fallback does not fabricate text
or make the artifact HTTP-serving-ready.

### Muse Glimmer weights and speculative decoding

Strict Muse Glimmer bundles additionally accept:

- `--runtime auto|cpu|native-cuda`;
- `--weight-precision auto|bf16|k-quant-dynamic|k-quant-17gb`;
- `--speculative-decoding off|auto|required`; and
- repeatable `--companion-checkpoint dflash|mmproj|lora`.

```bash
nfn infer \
  --checkpoint artifacts/glimmer-kquant \
  --runtime native-cuda \
  --weight-precision auto \
  --speculative-decoding auto \
  --companion-checkpoint dflash \
  --tile-ops-lib /absolute/path/libnfn_native_train_tile_ops.so \
  --prompt "Explain the local/global attention schedule." \
  --native-info
```

CUDA `auto` queries free/total memory on the selected device before model
allocation and budgets the authenticated target, enabled companions, load
staging, workspace, hybrid target/assistant KV, configured context/session
state, verification scratch, and reserve. It chooses BF16, Dynamic, then 17GB
in fidelity order only among candidates that fit and whose kernel profile the
binding proves. Explicit precision is a strict pin. CPU `auto` uses the
authenticated primary and does not query VRAM.

DFlash requires the lossless full/hybrid target cache. `auto` may omit an
unavailable assistant only before model/session mutation; `required` fails.
Every accepted target/assistant block is committed transactionally and emitted
one token at a time. `--native-info` reports requested/effective precision,
selection proof and VRAM bytes, effective speculation, and acceptance counters.
A loaded native LoRA disables the stock assistant unless an exact adapted
assistant lineage is supplied.

## Interactive transcript

When both stdin and stdout are TTYs, the default mode is `transcript`:

```bash
nfn infer \
  --checkpoint model-native \
  --system-prompt "Be concise." \
  --prompt "Start with a cache definition."
```

The optional initial `--prompt` is generated and retained as the first user /
assistant turn. Later prompts are re-rendered from process-local role messages.
The same resident session receives every rendered prefix through
`session.prefill(...)`; exact longest-prefix synchronization reuses matching
lossless-cache state and truncates/rebuilds when the rendered prefix diverges.

Interactive commands are:

| Command | Effect |
|---|---|
| `/mode stateless` | Keep leading system/developer instructions but make each subsequent prompt independent |
| `/mode transcript` | Resume process-local transcript rendering |
| `/reset` | Clear transcript turns, retain `--system-prompt`, and reset resident token/cache state |
| `/help` | Show the compact command list |
| `/exit` or `/quit` | Close the session/model and exit |

`--chat-mode stateless` selects stateless mode at startup. Non-TTY execution
also defaults to stateless, but it remains a single request regardless of the
mode flag.

## Chat rendering and context trimming

`--chat-template auto` prefers a supported template from the Native Execution
manifest. The lean renderer accepts `plain_roles` or a literal `{messages}` /
`{{messages}}` placeholder, with an optional `{assistant_prompt}` /
`{{assistant_prompt}}` marker. Arbitrary Jinja is not evaluated. Muse Glimmer
uses its dedicated deterministic ATEM renderer and exact tokenizer hashes; it
never falls back to `plain_roles`. For other families, if auto cannot use the
artifact template, the CLI emits one warning and uses `plain_roles` for that
process. Select `--chat-template plain_roles` to make the fallback
explicit, or pass a data-only template file:

```text
BEGIN
{{messages}}
NEXT={{assistant_prompt}}
```

Before each generation, the CLI reserves `--max-new-tokens` from the manifest
context limit. If needed, it removes the oldest complete user-led
conversation groups while preserving leading developer/system instructions and
the newest request. It fails rather than dropping that mandatory remainder.
Front trimming intentionally causes exact-prefix synchronization to rebuild
state from position zero.

Manifest stop-token IDs are passed to `GenerationConfig`. Declared textual
stop strings, EOS text, and role delimiters are stripped before assistant text
is displayed or retained in transcript history.

## Cache and sampling controls

The resident CLI accepts:

- `--kv-cache auto|full|off|turboquant`
- `--turboquant-profile mse-3.5|qjl-3.5`
- `--turboquant-attention-backend cpu|tile-cuda`
- `--tile-ops-lib PATH`, optional `--cuda-runtime-lib PATH_OR_SONAME`, and
  `--cuda-device INDEX`
- `--max-new-tokens`, `--temperature`, `--top-k`, `--top-p`, and `--seed`

`auto` is the default and resolves only through joint artifact/binding proof.
For a bound dense-v5 or canonical LLaMA artifact with the current resident ABI,
it selects the lossless full KV cache. `off` requests full-prefix
recomputation. Explicit
TurboQuant selects the proved packed native CPU cache only when a reviewed
dense artifact and binding jointly prove its codec/cache ABI. The supported
preset topologies are `gpt2`, megakernel, MoA, z-loss, QK-norm, stable, and
softcap. MoA uses the checkpoint-selected activation for prefill and decode and
does not rerun its candidate probes.
Canonical LLaMA supports only `auto`, `full`, and `off`; its explicit
TurboQuant request fails closed. Differential/modern variants, bare MoA `.bin`
files, and every non-dense TurboQuant path remain fail-closed.

For a separately feature-gated reviewed-dense artifact, explicitly select the
hybrid CUDA attention path with a freshly built strict sidecar:

```bash
nfn infer --checkpoint model-native \
  --kv-cache turboquant \
  --turboquant-profile qjl-3.5 \
  --turboquant-attention-backend tile-cuda \
  --tile-ops-lib /absolute/path/libnfn_native_train_tile_ops_strict.so \
  --cuda-device 0
```

CPU remains the default. An explicit Tile request checks the separate feature
ABI, strict-math ABI, CUDA runtime/device, and geometry before creating a
session; it never silently uses CPU or full cache. The path keeps model compute
and row encoding on CPU and moves packed historical attention to CUDA. It is
not a whole-model GPU or performance claim.

Exact temperature zero (including negative zero) retains the resident SDK's
strict deterministic model-compute contract. Positive values, including
positive subnormals, remain ordinary sampling requests; negative and non-finite
values are rejected.

This CLI path proves real resident inference for the seven reviewed dense-v5
topologies, canonical LLaMA, exact standard-MoE profiles, and strict Muse
Glimmer BF16/K-Quant text profiles. Glimmer's CUDA text runner is separate from
the legacy hybrid Tile attention path above; full-BF16/mmproj vision remains
CPU-only and a CUDA mmproj request fails before load. It does not make other native families
resident-ready and does not add graph interpretation, tools, persistence,
batching, or server state. The standalone HTTP surface is documented separately in
[Native Inference Serving](rest-api/native-inference-serving.md).
