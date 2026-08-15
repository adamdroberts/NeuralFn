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
conda activate NeuralFn
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
binding proves. The official 24 GB and 32 GB hardware tiers are decimal
vendor-memory classes; actual admission still uses CUDA's exact free/total byte
counts and the full byte budget. Explicit precision is a strict pin. It is not
an upper bound on physical VRAM: explicitly selecting `k-quant-17gb` on a
32-GB or larger device is valid when the authenticated target, requested
companions, caches, workspace, and reserve fit. CPU `auto` uses the
authenticated primary and does not query VRAM.

DFlash requires the lossless full/hybrid target cache. `auto` may omit an
unavailable assistant only before model/session mutation; `required` fails.
Every accepted target/assistant block is committed transactionally and emitted
one token at a time. `--native-info` reports requested/effective precision,
selection proof and VRAM bytes, effective speculation, and acceptance counters.
A loaded native LoRA disables the stock assistant unless an exact adapted
assistant lineage is supplied.

### Full-size CUDA qualification

`tools/qualify_muse_glimmer_gpu.py` is the release-evidence harness for this
path. Unlike `--native-info`, it starts from a fresh worker, builds the current
sources with real NVCC for the selected compute capability, loads the strict
math sidecar, runs a source- and binary-bound raw-kernel probe under
compute-sanitizer `memcheck`, `synccheck`, `initcheck`, and `racecheck`, and
then benchmarks the same build's authenticated target, DFlash, and vision path
without sanitizer instrumentation. It only accepts the canonical mapping
24 GB→`k-quant-17gb`, 32 GB→`k-quant-dynamic`, and 80 GB→`bf16`. These are
minimum decimal-byte profile tiers, not exact physical-device bands. A larger
GPU may qualify a lower tier; the worker explicitly pins that lower profile
because production `auto` must continue choosing the highest-fidelity profile
that fits. The highest eligible tier still exercises `auto`. Every result
records the actual CUDA total, requested/effective selection, sampled peak
delta, and minimum free memory. Devices below the tier minimum still fail.

Each result contains the source, binary, manifest and selected-checkpoint
hashes; CUDA runtime/driver/compute capability; model and companion resident
bytes; sampled VRAM; p50/p95 TTFT and decode throughput; speculative
acceptance; vision timing; and CUDA/CPU compute counters. `verify` accepts a
three-result matrix only when every run has full production geometry, a
passing zero-error sanitizer summary, an 8K-or-larger context, positive VRAM,
DFlash and vision CUDA execution, zero CPU model-compute rows, and the same
source proof. Source builds require CUDA Toolkit 13.3+ with `cuda_tile.h` and
NVCC Tile C++ support. A compiler-only proof is available when the build host
has a toolkit but no usable GPU:

```bash
python tools/qualify_muse_glimmer_gpu.py build \
  --cuda-arch sm_120 \
  --nvcc /usr/local/cuda/bin/nvcc \
  --build-dir build/glimmer-compile-sm120 \
  --json-out build/glimmer-compile-sm120.json
```

That result deliberately reports `status: "source-built"` and
`release_qualified: false`. It verifies compilation and ABI versions only; it
does not substitute for device execution, compute-sanitizer, or the three
full-size `run` results. See the complete commands in the root README.

The 2026-08-13 packed-profile runs are complete on an RTX 5090 with CUDA 13.3.
For the 32-GB tier, the default policy selected K-Quant-Dynamic; target,
DFlash, and mmproj were CUDA-resident; all four sanitizer tools reported zero
errors/hazards; and the 128/2,048/8,192-token full-size trials reported zero
CPU model-compute rows. At 8,192 tokens, three-trial p50 prefill was 8.26
tok/s, p50 TTFT was 992.434 s, and p50 16-token speculative decode was 1.410
tok/s.

For the 24-GB tier, an explicit K-Quant-17GB selection on the same larger card
also passed all four sanitizer tools, CUDA target+DFlash+mmproj, zero-CPU
telemetry, and 128/8,192-token trials. It measured a 20,359,217,152-byte
(18.961-GiB) peak CUDA delta and retained at least 9,398,059,008 bytes free.
The 8K single trial measured 8.96 prefill tok/s, 914.161-second TTFT, and 1.490
tok/s for 16-token DFlash decode. Its result records both the 24-billion-byte
minimum tier and the actual 33,708,376,064-byte device total. Only the
80-GB-or-larger BF16 result is still unmeasured; the three-tier verifier also
requires all input results to share one current source proof. Because the
standalone Dynamic result predates the qualifier's tier-policy edit, it must be
rerun with the future BF16 result before the final matrix can verify.

A separate full-size target-only oracle check used llama.cpp build 10349 at
pinned commit `62bf73d25c53b8161f8a22894d4f90c4aebbd7d0` and the same canonical
Dynamic artifact. For raw prefix `[200000, 19873]` (BOS + `Hello`), both
runtimes returned the same 16 greedy token IDs:
`[24, 372, 1045, 10016, 328, 2885, 262, 5091, 8811, 511, 917, 4921, 768, 328, 2885, 262]`.
NeuralFn reported zero CPU model-compute rows. Treat this as one target
raw-prompt proof, not as full logit, ATEM chat, sampled, DFlash, or quality
parity.

### Current-source chat and kernel benchmarks

`tools/bench_muse_glimmer_native_chat.py` benchmarks an actual rendered ATEM
chat turn rather than a repeated-token capacity fixture. It records the exact
rendered prompt token IDs and digest, model-load time, new/reused prefill
tokens, separate prefill and decode times, target/DFlash steps and acceptance,
generated IDs/text, resident counters, and sampled free/total VRAM. Use
`--compute-mode strict` for exact-zero strict model computation or
`--compute-mode throughput` for deterministic top-k-one positive-temperature
execution. Always report the selected mode; the two are not interchangeable.

```bash
python tools/bench_muse_glimmer_native_chat.py \
  --artifact artifacts/glimmer-kquant \
  --binding-lib /absolute/path/_native_inference.so \
  --tile-ops-lib /absolute/path/libnfn_native_train_tile_ops_strict.so \
  --cuda-runtime-lib /usr/local/cuda/lib64/libcudart.so.13 \
  --weight-precision k-quant-17gb \
  --compute-mode strict \
  --max-new-tokens 64 \
  --json-out /var/mnt/disk2/tmp/glimmer-native-chat.json
```

Add `--dflash` for the paired run. Run each runtime from a fresh process on an
otherwise idle GPU and use the same prompt, stop policy, generated-token
count, and artifact when comparing NeuralFn with pinned llama.cpp. Meta's
published 74.9/233.4 tok/s RTX 5090 table does not disclose the exact prompt
set, so it is an external target, not an exactly reproducible local oracle.

The accepted 2026-08-15 current-source K-Quant-17GB run used one 41-token
rendered ATEM prompt on the local 33,708,376,064-byte RTX 5090. The target was
the complete 52-layer Muse-Glimmer-30B, not a tiny stand-in. Target-only and
DFlash each generated 32 tokens at exact zero temperature after one warmup and
ten measured trials. Target-only measured **78.608 tok/s** median
(78.515–78.809), with token hash
`63baebaa0742852d37abf85e81c815430267789bdbb79591eb56a1e1a50b74b1` and an
18,949,865,472-byte sampled peak CUDA delta. DFlash measured **271.244 tok/s**
median and 271.792 tok/s mean (269.303–274.820); every trial accepted 28/34
proposals, processed 37 target rows in three assistant blocks, returned the
same canonical hash, and sampled a 20,654,850,048-byte peak. Both modes
reported zero CPU model-compute rows. DFlash is a 3.451x speedup for this exact
request.

The target path captures the complete repeated greedy token body—LM head,
device argmax, device-indirect embedding, all 52 decoder layers, cache writes,
and final hidden state—as one CUDA graph. Its grouped packed-projection
megakernels share activation quantization across Q/K/V/gate and MLP gate/up,
then fuse sigmoid-gate or SwiGLU handoff into the packed output projection.
The direct dual-RMS handoff also writes an exact llama `block_q8_1` activation
while producing the normal hidden, normalized, and residual-capture outputs;
the following prequantized MMVQ skips a reread and standalone quantization.
A same-binary A/B retained these megakernels: enabled measured 78.608 tok/s
and 17,963 launches over the complete ten-trial run; disabled measured 72.871
tok/s and 31,691 launches. Output hashes were identical, so the accepted
default is a 7.87% throughput gain and 43.32% launch-count reduction.
`NFN_GLIMMER_MMQ_MEGAKERNELS=0` and
`NFN_GLIMMER_CUDA_GRAPHS=0` are development-only bisection gates; production
should leave both defaults enabled.

The speculative verifier uses row-exact MMVQ, reuses its exact Q8 activation
for independent projections, overlaps K/V work with gate projection, writes
K/V directly to transactional staging, and uses a short-context split
attention kernel. Its accepted cooperative dual-RMS megakernel partitions
each 6,656-wide row across eight cooperative blocks without changing the
baseline 256-lane FP32 reduction order. A same-final-binary A/B with only
`NFN_GLIMMER_COOPERATIVE_BATCH_RMS=0` measured 264.724 tok/s median versus
271.244 enabled: +6.520 tok/s (+2.46%), with identical 28/34 acceptance,
output hash, 37 target rows, sampled peak, and zero CPU rows. Packed-weight
block hoisting remains enabled after an earlier exact compile-time A/B
measured 158.779 tok/s off and 235.657 on. Rejected candidates—including
all-row attention splitting, concurrent split quantization, wider Q5/Q6 row
groups, FFN-specific row tiling, and the approximate verifier—remain absent.

The accepted strict sidecar SHA-256 is
`d15e946e16b1ef5b643a4556f8f719e49efa1466ad12c4957dbcaaa73953e994` and the
matching native binding SHA-256 is
`5e3d983fbed735a4dee281ff2c07f165e8e32b47f807a0abe1d3504d58ec870d`.
Its 40-kernel direct probe, including the bit-exact cooperative dual RMS,
short-context DFlash attention, dual-RMS→Q8→MMVQ handoff, device-indirect
embedding, and device-position fused attention, passed
memcheck, synccheck, initcheck, and racecheck with zero errors/hazards.

The earlier 256.69-tok/s DFlash number used a non-canonical approximate
verifier and returned a different token hash; it is rejected evidence, not a
production benchmark. Exact speculative speed also follows acceptance: on a
second prompt, 22/74 accepted proposals produced 73.60 tok/s. The exact target
and DFlash results are numerically 4.95% and 16.21% above Meta's published
74.9/233.4 numbers. Meta's exact workload is undisclosed, so neither ratio is a
matched reproduction. The pinned llama.cpp
build 10349 at commit `62bf73d` was run directly on the same prompt, artifact,
GPU, 32-token greedy policy and companions: it reached 77.8 target-only and
225.7 DFlash tok/s. NeuralFn is 1.04% and 20.18% faster in those matched local
comparisons. A pinned, reproducible ExecuTorch command/artifact was not
available, so no ExecuTorch comparison is claimed. Remaining long-context and
multi-session performance work is
tracked in
[`glimmer-support-todo.md`](glimmer-support-todo.md#53-still-missing-cuda-performance-kernels).

`tools/bench_muse_glimmer_packed_linear.py` is the smaller development gate
for one real Glimmer projection geometry. It compares two sidecars, checks
numeric error before timing with CUDA events, and supports one-token, 16-row,
and prequantized-Q8 measurements. Its output must stay labeled as focused
kernel evidence rather than end-to-end tok/s.

`tools/bench_muse_glimmer_residual_norm.py` is the corresponding focused gate
for 6,656-wide RMSNorm/residual composition. It compares the separate
RMSNorm+copy/add baseline with
`nfn_native_tile_glimmer_rms_norm_affine_capture_residual_float32_v1` and
`nfn_native_tile_glimmer_rms_norm_affine_add_residual_float32_v1`, checks both
outputs before timing, and reports one-row or multirow CUDA-event latency. The
resident treats those two optional symbols as one feature and fails an
incomplete pair; verification can therefore distinguish the exact fused path
from the backward-compatible composed path.

## Interactive transcript

When both stdin and stdout are TTYs, the installed CLI opens the colorful Rich
native-chat TUI and defaults to `transcript`:

```bash
conda activate NeuralFn
nfn infer \
  --checkpoint artifacts/glimmer-kquant \
  --runtime native-cuda \
  --weight-precision k-quant-17gb \
  --companion-checkpoint dflash \
  --speculative-decoding auto \
  --chat-mode transcript \
  --system-prompt "Be concise."
```

Shell activation supplies the installed `nfn` entry point through `PATH`.
Automation or sandbox shells that do not retain activation can use
`conda run -n NeuralFn nfn infer ...` (or the environment's absolute `nfn`
path); a `command not found` from such a shell does not imply the package is
missing from the named environment.

When the strict sidecar is not installed in a default search location, add
either `--tile-ops-lib /absolute/path/libnfn_native_train_tile_ops_strict.so`
or its exact-artifact alias `--strict-tile-ops-lib` with the same path.

The model banner reports the effective runtime, weight precision, cache,
speculation mode, tokenizer/template, context limit, and sampling settings.
Each assistant panel includes new/reused prefill tokens, prefill time, decode
tok/s, and DFlash accepted/proposed counts. The optional initial `--prompt` is
generated and retained as the first user/assistant turn. Later prompts are
re-rendered from process-local role messages. The same resident session
receives every rendered prefix through `session.prefill(...)`; exact
longest-prefix synchronization reuses matching lossless-cache state and
truncates/rebuilds when the rendered prefix diverges.

The TUI is presentation around the same lean process-local native driver; it
does not launch a server or a subprocess. The `native_infer_tui` module is
included in the installed `nfn` package. Non-TTY output stays plain and
machine-friendly. The CLI default is 512 completion tokens so an ATEM model
has room to finish private reasoning and enter its user-directed channel;
`--max-new-tokens` remains an explicit override.

Interactive commands are:

| Command | Effect |
|---|---|
| `/mode stateless` | Keep leading system/developer instructions but make each subsequent prompt independent |
| `/mode transcript` | Resume process-local transcript rendering |
| `/show` or `/stats` | Show live runtime, precision, cache, DFlash, CUDA-memory and kernel counters |
| `/reset` | Clear transcript turns, retain `--system-prompt`, and reset resident token/cache state |
| `/clear` | Clear the terminal and redraw the current model banner |
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
stop strings, EOS text, and role delimiters are stripped before ordinary
assistant text is displayed or retained in transcript history. Muse Glimmer
uses a stricter ATEM protocol parser: it separates `to=self` and `to=user`
messages, never displays or stores the private channel, and adds only
user-directed text to transcript history. A malformed control envelope or a
completion that ends inside private reasoning fails closed instead of turning
that reasoning into an assistant answer. A length-limited `to=user` answer is
displayed with a warning but remains safe to retain.

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

For packed Glimmer targets, the positive-temperature CUDA path may quantize a
normalized one-token activation to signed Q8 blocks once and reuse it across
the Q/K/V/attention-gate projections, then do the same for the shared MLP
input. The versioned symbols are
`nfn_native_tile_quantize_q8_1_float32_v1` and
`nfn_native_tile_linear_packed_weight_q8_1_float32_v1`; model statistics expose
their quantization and packed-linear call counts. Exact-zero inference keeps
the strict FP32-activation packed projection path. The focused benchmark can
measure either path, but a Q8 kernel speedup is not by itself a language-quality
or whole-model throughput result.

Ordinary greedy target decoding and greedy DFlash use
`nfn_native_tile_argmax_rows_float32_v1` when the sidecar provides it. The
target-only path selects its current vocabulary row on device. Assistant
proposals and target verification/bonus rows likewise choose their
deterministic lowest-ID argmax on device and return only IDs plus selected
values instead of copying two 15×202,048 logit matrices to the host.
`/show` reports `cuda_device_argmax_calls` and `cuda_device_argmax_rows`, so a
real transcript or benchmark can prove that this route executed.
Sampled generation deliberately does not use this shortcut. Lossless sampled
speculation retains full p and q rows for acceptance ratios and rejection-
residual sampling.

This CLI path proves real resident inference for the seven reviewed dense-v5
topologies, canonical LLaMA, exact standard-MoE profiles, and strict Muse
Glimmer BF16/K-Quant text profiles. Glimmer's CUDA text runner is separate from
the legacy hybrid Tile attention path above. Full-BF16 vision and packed
still-image mmproj can use CPU or whole-model CUDA; CUDA requires the separate
vision ABI and never falls back to CPU after load/session mutation. It does not make other native families
resident-ready and does not add graph interpretation, tools, persistence,
batching, or server state. The standalone HTTP surface is documented separately in
[Native Inference Serving](rest-api/native-inference-serving.md).
