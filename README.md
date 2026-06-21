# NeuralFn

NeuralFn is a graph-native neural network framework where each neuron can be a built-in primitive or a user-defined Python function with typed I/O ports, connected in arbitrary directed graphs. This repository now combines that core library with an authenticated web platform for multi-project, multi-session editing, training, analytics, and MCP-driven automation.

NeuralFn supports a scalar graph runtime plus native CUDA trainers. Legacy graph-backed Torch modules remain in the source tree for old experiments, but Torch is no longer an installable NeuralFn dependency extra. The default install is now the lean native/core SDK surface: it does not install Torch, NumPy, tokenizer, dataset, graph-analysis, or server packages. Native GPT training uses cached token shards and the compiled CUDA trainer path without importing `torch`.

The top-level `nfn` module keeps that startup contract when imported as a CLI
shim: `from nfn import main; main()` can serve root help and native fast-path
inference without importing `nfn_impl` or `torch`. Planner helpers such as
`maybe_plan`, `recipe_from_state`, and `render_help` are still available from
`nfn`, but they load the graph-backed planner lazily when first accessed.

Install extras only for the workflows you need:

```bash
pip install -e .                 # native/core SDK and CLI metadata
pip install -e ".[tile-cuda]"     # local CUDA Tile builds
pip install -e ".[datasets]"      # raw-text tokenization and HF dataset cache setup
pip install -e ".[graph]"         # Python graph analysis/runtime helpers
pip install -e ".[server]"        # FastAPI editor/backend and MCP server
pip install -e ".[all]"           # full native/server/dataset workstation, without Torch
```

`.[all]` intentionally excludes Torch, and there is no `.[torch]` extra. If you
need to run legacy graph-backed Torch code while it still exists in the tree,
install PyTorch explicitly in that environment outside NeuralFn's package
metadata.

CUDA Tile development targets CUDA Toolkit 13.3+ on the SM120 workstation. The
generic Python Tile extension and the trainer-facing raw C ABI both build from
`neuralfn/csrc/tile_cuda/kernels.cu`; after toolkit changes, run
`bash tools/rebuild_native_sm120.sh` to rebuild
`build/libnfn_native_train_tile_ops.so`, the compiled GPT/native training
frontends, the GPT-2 compatibility frontend, and the missing-template native
stubs against the current CUDA toolkit. The script defaults to
`NFN_TILE_CUDA_ARCH=sm_120a` and `NFN_TILE_CUDA_USE_TK_ATTENTION=1`; set
`NFN_NATIVE_REBUILD_OUT_DIR=/path/to/build` to write the refreshed artifacts
somewhere other than `build/`. After rebuilding, run
`NFN_TILE_CUDA_TEST=1 python -m pytest tests/test_tile_cuda_gpu.py tests/test_tile_cuda_ops.py tests/test_tile_cuda_optimizer.py -q -rs`
to confirm the extension executes GPU tests instead of skipping. After the
CUDA Toolkit 13.3.33 WSL reinstall, the dedicated RTX 5090 path is correctness
green for the revisited native and Tile CUDA gates. After reinstalling the WSL
CUDA toolkit (`cuda-toolkit-13-3`), the GPU-visible full suite passed with
`1185 passed, 4 skipped, 20 warnings, 468 subtests passed`; the focused native/Tile CUDA
gates, GPT template preset suite, and native no-Torch guard all pass. Dense GPT
native training now routes the no-bias BF16 LM-head logits GEMM through the TK
BF16 forward bridge by default for the
default `50304,32768,768,T,N` row-chunk shape. The default tied LM-head row
chunk is now 32768 rows for the workstation 5090 profile, reducing the default
64x1024 LM-head chunk count from 8 to 2 while reserving about 3.30GB of resident
BF16 logit workspace. Pass `--lm-head-row-chunk-size 8192`,
`--native-cuda-lm-head-row-chunk-size 8192`, or
`NativeGpt2RunConfig(lm_head_row_chunk_size=8192, ...)` to reproduce the older
lower-memory default. A full-batch 65536-row chunk is not a default candidate:
the CUDA 13.3 RTX 5090 paired run timed out at the 300s sample limit after
leaving the GPU utilization counter stuck until the WSL driver recovered. The
native runner now rejects LM-head chunks above 32768 rows before launching CUDA
unless `NFN_NATIVE_GPT_ALLOW_UNSAFE_LM_HEAD_ROW_CHUNK=1` is set for explicit
paired diagnostics. The
current staged CUDA 13.3.33
10-step parity sample on the dedicated RTX 5090 measured NeuralFn at
`2564.590 ms/step` versus llm.kittens at `2447.451 ms/step`
(`1.047862x` train-loop wall time, `0.952442x` tokens/sec); the latest
instrumentation-free sample measured a smaller but still-open gap at
`1.010870x` with NeuralFn at `2522.440 ms/step` versus llm.kittens at
`2495.348667 ms/step`. The remaining gap is still native GPU kernel work rather than
Torch, Python, or graph-editor execution. Because parity samples can move with
reference-run noise, keep using
`tools/bench_native_gpt_sm120_parity.sh` before declaring final parity on a new
build. A post-reinstall wrapper timing check also confirms that
`python cli/scripts/train_gpt.py --tinystories --native-cuda-dry-run --native-cuda-print-command`
returns the compiled C++ delegate in about `0.03s`, so the old Python schedule
estimation path is not on the current native GPT startup route. A native
startup-only paired run with and without `--tinystories` measured `1.000758x`
setup wall time, which assigns startup cost to CUDA arena materialization and
weight initialization rather than dataset alias resolution. LM-head cuBLASLt
expansion, BF16 arena removal, token-shadow removal, threaded token
initialization, LM-head 4096-row chunks, disabling cuBLASLt plan prewarm, and
direct bias-gradient first-write probes remain rejected diagnostic switches
rather than workflow guidance. The QKV dInput/dWeight side-stream diagnostic is
also rejected as a default: set
`NFN_NATIVE_GPT_BLOCK_QKV_CONCURRENT_DINPUT_DWEIGHT=1` only for paired
bisection, since the CUDA 13.3 RTX 5090 same-script run enabled the route but
slowed train-loop wall time to `1.008821x`. Dense GPT startup defaults the tied token-weight
initializer to the vector4 CUDA Tile route; set
`NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT=0` or
`NFN_TILE_CUDA_TOKEN_WEIGHT_VECTOR4_INIT=0` only for paired bisection against
the previous fast int32 path.
The opt-in vector4-strided token-weight initializer
(`NFN_TILE_CUDA_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT=1` /
`NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT=1`) is also diagnostic-only:
it improved the token-init substage in a startup-only RTX 5090 run but did not
beat the strict setup-wall gate.
Dense GPT native BF16 classifier/CE now uses vectorized BF16 row loads by
default to match the llm.kittens fused-classifier memory access pattern,
including the final dlogit write pass when scalar stores are selected; set
`NFN_NATIVE_GPT_CE_BF16_VEC_LOADS=0` or
`NFN_TILE_CUDA_CE_BF16_VEC_LOADS=0` only for paired scalar-load bisection.

For local server/editor development, leaving `NEURALFN_REDIS_URL` empty keeps
live state and persistence in-process. Redis-backed deployments still enqueue
through Redis, but the no-Redis fallback now persists synchronously so tests and
single-workstation runs do not race teardown or lose the latest session/run
state. Raw-text dataset metadata can also resolve the known `gpt2`,
`cl100k_base`, and `o200k_base` vocabulary sizes offline; actual tokenization
still requires the tokenizer assets or the `tiktoken` cache.

> **Pre-alpha notice:** NeuralFn is in active pre-alpha development. The SDK, REST API, MCP tools, and graph format are all subject to rapid, breaking changes without prior deprecation. Do not depend on API stability at this stage. See the [CHANGELOG](CHANGELOG.md) for a running list of what has changed.

## Documentation

**[Read the full documentation](docs/README.md)**

| Section | What it covers |
|---------|---------------|
| [Getting Started](docs/getting-started.md) | Installation, quickstart, your first graph |
| [CLI Workflows](docs/cli.md) | `nfn` train/infer/eval workflows, datasets, tokenizers, artifacts |
| [Framework Guide](docs/framework-guide/README.md) | How to build with NeuralFn in Python -- neurons, graphs, subgraphs, training, inference |
| [Python SDK Reference](docs/python-sdk/README.md) | Every class, function, method, and type in the `neuralfn` package |
| [REST API Reference](docs/rest-api/README.md) | All HTTP endpoints with request/response shapes |
| [MCP Tools Reference](docs/mcp/README.md) | All MCP tools for AI agent integration |
| [Server Internals](docs/server/README.md) | Services, ORM models, auth, configuration |
| [Editor Reference](docs/editor/README.md) | React frontend: store, components, API client |
| [Agent Skills](docs/agent-skills.md) | AI coding agent skills for Cursor and Codex |
| [llms.txt](llms.txt) | LLM-friendly project index |
| [llms-full.txt](llms-full.txt) | Complete docs in a single file for LLM ingestion |

Native GPT benchmark and preflight runs can pass
`--native-cuda-no-checkpoint` from the Python wrappers or `--no-checkpoint`
to the compiled C++ trainer to skip final trained-checkpoint export. Default
training still writes the final native checkpoint.
When sampled training loss is enabled with `--train-loss-every-steps`, the
default BF16/u16-token native GPT path now fuses public-vocab CE loss
accumulation with the in-place BF16 dlogits kernel, avoiding the older separate
loss-partials pass. The fused kernel synchronizes after the target-logit loss
read and before overwriting logits with dlogits, so sampled train loss does not
race the in-place classifier backward write. Validation loss still uses
`--eval-every-steps` and remains separate from sampled train loss.
Dense GPT native training also now defaults to eliding the unused FP32
attention-projection and MLP-projection scratch-tape buffers when BF16
projection-residual is active. Set
`NFN_NATIVE_GPT_ELIDE_FLOAT_PROJECTION_OUTPUTS=0` only for paired bisection
against the older reservation.

Use `python tools/paired_kernel_speed.py --baseline "OLD_COMMAND"
--candidate "NEW_COMMAND" --samples N --json-out /tmp/result.json` for
candidate-vs-current CUDA timing. The helper defaults
`--cuda-visible-devices` to `auto`, selecting an idle display-disabled NVIDIA
GPU from `nvidia-smi` when one is available; pass an explicit device id such as
`--cuda-visible-devices 0` to pin manually, or `--cuda-visible-devices ""` to
leave the environment unchanged. It still alternates pairs in the same sampling
window and runs one warmup pair by default so first-use CUDA/kernel load does
not contaminate reported samples. It sets `CUDA_DEVICE_MAX_CONNECTIONS=1` for
both commands by default; pass `--cuda-device-max-connections ""` to leave that
environment unchanged. Pass repeatable `--baseline-env KEY=VALUE` or
`--candidate-env KEY=VALUE` flags for environment-gated kernel candidates; these
overrides apply only to that side of the pair and are recorded in the JSON/text
output. Pass repeatable `--max-candidate-ratio [STAT:]METRIC=RATIO` gates when a
candidate must not regress a hot native metric such as
`stage.lm_head_backward.total_ms`, `stage.block_backward.total_ms`, or
`train_loop_wall_ms_per_step`; `STAT` defaults to `mean` and can be `median`,
`min`, or `max`, so use gates such as
`median:train_loop_wall_ms_per_step=1.000` for noisy GPU timing. Missing metrics
fail the gate so a stage-timing or JSON-output mistake cannot be accepted as a
passing kernel result.
`--command-timeout-seconds N` terminates the timed-out command's process
group so a slow native candidate does not leave child GPU work running after the
sample is recorded. Interrupting the helper also terminates the active command
process group before re-raising, so aborted memory-heavy candidates do not keep
running on the selected GPU. Pass `--require-idle-selected-gpu` for dedicated
benchmark runs that must abort if `nvidia-smi` reports any compute process on
the selected CUDA GPU before each warmup or measured command. The idle check is scoped
to the selected GPU UUID, so a separate display GPU can still be active. Pass
`--max-selected-gpu-utilization-pct N` to also reject samples when the selected
CUDA GPU's `nvidia-smi` utilization is already above `N` before each warmup or
measured command. When `nvidia-smi` is available, the JSON also includes
the resolved `cuda_device_selection`, run-level
`gpu_before` / `gpu_after` snapshots and per-sample `paired_samples[].gpu_before`
/ `paired_samples[].gpu_after` snapshots, plus command-level
`paired_samples[].baseline.gpu_before` / `gpu_after` and
`paired_samples[].candidate.gpu_before` / `gpu_after` snapshots with GPU
identity, display-active state, utilization, memory, and active compute-process
rows so kernel-speed notes show which CUDA device was measured and whether
other compute work was present for a specific command. Text and JSON output also include
`gpu_sample_summary`, which summarizes selected-GPU utilization, memory, and
compute-process counts before and after measured samples; use this summary when
checking that candidate-vs-baseline timing was not skewed by other GPU load.
Native stage-timed JSON already records every CUDA bucket, and the text summary
prints the LM-head backward substages (`logits`, `ce`, `dhidden`, `dweight`,
optional `dhidden_dweight_concurrent`) plus block-backward substages such as
`mlp_proj.*`, `attn_sdpa.to_qkv`, and `qkv.dweight_bias` so parity runs can
attribute the current RTX 5090 gap without opening sidecar JSON by hand. The
same text summary also surfaces optimizer-step support stages:
`stage.final_norm_backward.total_ms`, `stage.embedding_backward.total_ms`,
`stage.gradient_zero.total_ms`, `stage.gradient_clip.total_ms`, and
`stage.adamw_update.total_ms`.
When a command exits nonzero and `--continue-on-error` is not set, the helper now
prints both stdout and stderr tails so CUDA driver/runtime messages from
external baselines are not hidden behind an empty stderr block.
For native-vs-native dense GPT kernel bisections, `tools/bench_native_gpt_sm120_candidate.sh`
accepts the canonical `NFN_SM120_NATIVE_*` environment variables and the shorter
`NFN_SM120_CANDIDATE_*` aliases for steps, samples, warmup, profile directory,
CUDA device selection, candidate env, template/graph selection, and JSON output.
The SM120 wrappers also accept generic `NFN_SM120_*` names such as
`NFN_SM120_STEPS`, `NFN_SM120_SAMPLES`, `NFN_SM120_WARMUP`,
`NFN_SM120_CUDA_VISIBLE_DEVICES`, `NFN_SM120_PROFILE_DIR`, and
`NFN_SM120_JSON_OUT` as the lowest-priority fallback, so a copied parity or
candidate command does not silently return to default step/sample counts.
Set `NFN_SM120_STAGE_TIMING=1` or the wrapper-specific stage-timing aliases to
collect native CUDA-event stage buckets even when `NFN_SM120_PROFILE_DIR=none`;
profile sidecars and stage attribution are independent controls.
Use `NFN_SM120_NATIVE_ENV`, `NFN_SM120_COMMON_ENV`, or `NFN_SM120_PARITY_ENV`
for shared environment variables that must be applied to both commands, such as
`NFN_NATIVE_GPT_LINEAR_SHAPE_STATS=1` during route attribution. Use
`NFN_SM120_NATIVE_CANDIDATE_ENV` or `NFN_SM120_CANDIDATE_ENV` only for
candidate-only switches. Env lists can be separated with spaces or with commas
between assignments, so both `KEY=1 OTHER=2` and `KEY=1,OTHER=2` are accepted;
commas inside a value, such as
`NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,3072,65536,N,T,0`, are kept as
part of that value.
Use `NFN_SM120_COMMON_EXTRA_ARGS` only for args that must be appended to both
commands, and use `NFN_SM120_NATIVE_CANDIDATE_EXTRA_ARGS`,
`NFN_SM120_NATIVE_CANDIDATE_ARGS`, or `NFN_SM120_CANDIDATE_EXTRA_ARGS` for
candidate-only CLI flags such as `--lm-head-row-chunk-size 32768`; the dry-run plan prints both resolved
commands so command separation can be audited before launching a long GPU job.
When comparing a saved or freshly compiled native trainer executable, set
`NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN=/tmp/nfn_gpt_native_train_candidate`
or the shorter `NFN_SM120_CANDIDATE_TRAIN_BIN=...`; the baseline still comes
from `NFN_NATIVE_GPT_TRAIN_BIN` and the candidate command no longer silently
reuses the baseline trainer path.
This keeps quick candidate runs from silently falling back to the wrapper's
default 10-step, 3-sample profile when using the shorter alias names.
Set `NFN_SM120_NATIVE_MAX_CANDIDATE_RATIO` or
`NFN_SM120_CANDIDATE_MAX_CANDIDATE_RATIO` to a whitespace-separated list of
`METRIC=RATIO` gates, for example
`stage.lm_head_backward.total_ms=1.000 train_loop_wall_ms_per_step=1.005`, so
native candidate runs fail nonzero when they worsen the LM-head/block hot
buckets even if total command timing looks flat.
When no explicit ratio list is supplied, measured candidate runs that actually
change the candidate Tile ops library, candidate-only env, or candidate-only
extra args now default to `train_loop_wall_ms_per_step=1.000`; if native stage
timing is enabled they also gate `stage.lm_head_backward.total_ms`,
`stage.block_backward.total_ms`, and `stage.block_backward.mlp_proj.total_ms` at
`1.000`. Stage-timed CE-kernel candidates whose candidate text mentions
`CE_BF16` also gate `stage.lm_head_backward.ce.total_ms=1.000`; LM-head hidden
prepack candidates mentioning `LM_HEAD_PREPACK_BF16_HIDDEN` additionally gate
`stage.lm_head_backward.dhidden.total_ms`,
`stage.lm_head_backward.dweight.total_ms`, and
`setup.uint16_arena_materialize.total_ms`. LM-head pipeline overlap candidates
still use the comparable train-loop and total LM-head gates; their
candidate-only `stage.lm_head_backward.pipeline_queue.total_ms` and
`stage.lm_head_backward.pipeline_final_wait.total_ms` fields are reported for
inspection instead of ratio-gated against a serial baseline that never emits
them. Attention candidates mentioning `PACKED_ATTENTION` or `BF16_ATTENTION`
gate
`stage.block_backward.attn_sdpa.total_ms`,
`stage.block_backward.attn_sdpa.to_qkv.total_ms`,
`attention_backward_tk_timing_us`, and
`attention_backward_dprep_timing_us` so packed-attention bisections fail on the
hot substage even when total command timing is noisy. Dry-run planning and
no-op baseline-vs-baseline checks stay ungated.
QKV side-stream candidates mentioning `BLOCK_QKV_CONCURRENT_DINPUT_DWEIGHT`
also gate `stage.block_backward.qkv.total_ms`; the candidate-only combined
`stage.block_backward.qkv.dinput_dweight_concurrent.total_ms` substage is still
reported for inspection, but it cannot be ratio-gated directly because the
serial baseline emits split dInput and dWeight substages instead.
For repeatable CUDA/driver bisection of known LM-head dHidden routes, set
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_tk_dinput_32768` (or the shorter
`NFN_SM120_CANDIDATE_PROFILE`) to expand the candidate env to
`NFN_NATIVE_LINEAR_TK_DINPUT_ENABLE_SHAPE=768,32768,50304,N,N`, or use
`lm_head_cublaslt_dhidden_32768` to expand to
`NFN_NATIVE_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE=768,32768,50304,N,N`. These
profiles are measurement shortcuts only; both routes remain default-off until
the same-script candidate gate beats the current route.
For compile-time Tile ops candidates, set
`NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_BUILD_FLAGS` or
`NFN_SM120_CANDIDATE_TILE_OPS_BUILD_FLAGS`; the wrapper builds a temporary
candidate `libnfn_native_train_tile_ops.so` with those extra nvcc flags and
compares it against the baseline library without overwriting `build/`. The
named profiles `tk_dgelu_dinput` and `tk_dgelu_approx_tanh` build the
llm.kittens SM120 fused dGELU dInput candidates with
`-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP` and, for the latter,
`-DLLMK_SM120_APPROX_DGELU_TANH=1`. Stage-timed dGELU candidates also gate
`stage.block_backward.mlp_proj.dinput.total_ms=1.000`. These are still
candidate-only routes: NeuralFn's CUDA 13.3 same-script `tk_dgelu_dinput`
profile passed the 3-step, 3-sample gate, but the longer 10-step, 3-sample gate
rejected promotion because `stage.block_backward.total_ms` measured `1.000689x`
even while `stage.block_backward.mlp_proj.dinput.total_ms` stayed faster at
`0.983891x`. The llm.kittens same-session notes show the same pattern of
short-run wins but rejected stable defaults after longer gates.
The named `qkv_concurrent_dinput_dweight` profile expands to
`NFN_NATIVE_GPT_BLOCK_QKV_CONCURRENT_DINPUT_DWEIGHT=1` for repeatable
stage-timed reruns of the default-off QKV side-stream diagnostic.
The helper decodes
native binary stdout/stderr with replacement, so external CUDA trainers that
emit non-UTF-8 bytes can still be compared in the same paired run. For
llm.kittens output, parsed `step ... ms ... tok/s` rows now report
`train_loop_wall_ms` as the sum of parsed step times,
`train_loop_wall_ms_per_step` as their mean, plus last-step fields under the
`llm_kittens_last_step_*` keys. Pass
`--command-timeout-seconds N` to cap each child command; with
`--continue-on-error`, timed-out candidates are kept in the JSON with
`timed_out: true`, `returncode: -1`, and their output tails instead of wedging
the tuning loop. If either command emits NeuralFn native JSON, the helper also
summarizes native in-loop
metrics such as `timing.train_loop_wall_ms`, `timing.train_tokens_per_second`,
setup time, checkpoint time, total native wall time, and any emitted
`timing.setup_timing` and `timing.stage_timing` entries separately from outer
command runtime. Compiled native GPT commands can write that JSON directly with
`--json-out PATH`; `--profile-json PATH` and `--stage-profile-json PATH` are
aliases for profiling runs such as
`NFN_NATIVE_GPT_STAGE_TIMING=1 build/nfn_gpt_native_train ... --profile-json /tmp/nfn_profile.json`.
Those profile files also include `float_arena_request_stats` and
`uint16_arena_request_stats`, ranked by suballocation name, elements, bytes,
and arena offset, so startup optimization can target the buffers that dominate
the large native `cudaMalloc` arenas.
The paired helper also detects those native JSON-output flags in child commands
and reads the sidecar file when stdout is empty, so stage-timed native runs can
keep stdout small without losing metric summaries or paired ratios.
When
native JSON includes `steps_completed`, the helper also reports
`train_loop_wall_ms_per_step` so total-loop NeuralFn runs can be compared fairly
with trainers that only log per-step timing. It also parses llm.kittens
`step ... ms ... tok/s` logs, so direct NeuralFn-vs-`train_gpt2cu` comparisons
report both trainers' normalized in-loop step time and token throughput in the
same paired JSON. Paired native JSON also summarizes route/strategy strings such
as `lm_head_logits_linear_strategy`, `lm_head_dhidden_linear_strategy`,
`lm_head_dweight_strategy`, block linear strategies, and attention strategies
under `baseline_native_metric_values` and `candidate_native_metric_values`, so
kernel-candidate results show whether a route actually changed. It also emits
`native_route_counter_changes`, comparing tracked route counters such as TK,
cuBLASLt, BF16 GEMM, LM-head logits, BF16 packing/cache, and attention launch
counts between baseline and candidate. If candidate-specific environment knobs
are set but those counters do not change, the text report warns that timing-only
improvements should be treated as noise until a route change or separate
kernel-level attribution confirms the candidate.
CUDA 13.3 grouped cuBLASLt layout readiness is also reported as
`linear_cublaslt_grouped_layout_probe_available`,
`linear_cublaslt_grouped_layout_probe_status`, and
`linear_cublaslt_grouped_layout_supported`; those fields are diagnostics for
future grouped-GEMM candidates and do not mean the current default route uses
grouped matmul.
The native trainer now prewarms CUDA 13.3 BF16 cuBLASLt plans by default for
real training runs, but leaves prewarm off for `--startup-only` probes so
startup diagnostics do not pay the extra setup cost. Set
`NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLANS=0`,
`NFN_NATIVE_GPT2_PREWARM_CUBLASLT_PLANS=0`, or
`NFN_TILE_CUDA_LINEAR_CUBLASLT_PREWARM=0` to disable it for paired bisection, or
set the same variables to `1` to force it during startup-only diagnostics.
Native JSON reports
`linear_cublaslt_plan_prewarm_available`,
`linear_cublaslt_plan_prewarm_enabled`,
`linear_cublaslt_plan_prewarm_attempted_count`,
`linear_cublaslt_plan_prewarm_success_count`, and
`linear_cublaslt_plan_prewarm_failure_count`, and setup timing includes
`setup.cublaslt_plan_prewarm`. The dedicated RTX 5090 CUDA 13.3 retest measured
the prewarmed route at `0.989405x` train-loop wall time and `0.964634x`
LM-head backward time, while startup-only remains unprewarmed by default.

The compiled `nfn_native_train` frontend can now be used directly as the
startup-fast dense GPT training command instead of relying on the Python
argument shim. It accepts the same common wrapper flags as `nfn train` and
`cli/scripts/train_gpt.py`, including `--dataset tinystories`, `--output`,
`--kernel-backend`, `--template` / `--preset`, `--graph`, and
`--native-cuda-*` aliases, then injects the dense GPT defaults
`--train-transformer-lm`, `--backend tile-cuda`, the TinyStories alias fallback,
and GPT-3's implicit `--train-seq-len 2048` / `--batch-size 32` when selected
through `--base-model gpt3` or `--template-name gpt3` before execing
`nfn_gpt_native_train`. High-level aliases are normalized before the exec; for
example `--native-cuda-no-checkpoint` is passed to the compiled trainer as the
single native `--no-checkpoint` flag. This removes Python startup from the
direct compiled frontend path; it does not by itself close the remaining SM120
in-loop throughput gap tracked by the parity benchmark.
The native dense GPT selector recognizes the shipped GPT-2 geometry aliases
`gpt`, `gpt2`, `gpt2_modern`, `gpt2_megakernel`, `gpt2_moa`, and `gpt3` as
compiled-loop runnable. NanoGPT aliases are recognized as dense GPT templates
but still report a geometry-mismatch native work item until the shared loop is
made shape-dynamic; custom graph files report explicit native planner work
instead of routing tensors through Torch or editor graph nodes.

Native dense-GPT plan and runtime JSON now include
`lm_head_classifier_strategy_contract`, which makes the remaining SM120 parity
tradeoff explicit. The llm.kittens reference keeps full resident BF16 logits for
all microbatch rows before `fused_classifier`, while NeuralFn keeps only the
row-chunked BF16 logits/dlogits buffer and overwrites logits in-place during CE
backward. At the default `64 x 1024` shape the contract reports 65,536 full
logit rows versus the 8,192-row NeuralFn chunk, 6.59GB of reference-style BF16
logits versus 825.8MB resident NeuralFn BF16 logits, and an 8x resident-logit
reduction. Use this object with `tools/paired_kernel_speed.py` stage metrics
when evaluating a fused classifier/LM-head-backward kernel or a memory-gated
full-logit candidate. The paired benchmark helper extracts and prints the
contract's full/chunk BF16 byte counts, chunk rows/count, and resident-logit
reduction ratio alongside LM-head stage timing.
The raw Tile ABI also exposes the LM-head classifier row-chunk route as a
distinct symbol,
`nfn_native_tile_lm_head_classifier_backward_loss_inplace_strided_no_pad_zero_bf16_bits_u16_targets`,
plus the no-loss
`nfn_native_tile_lm_head_classifier_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace`
variant and `nfn_native_tile_lm_head_classifier_*` counters. Native dense-GPT
runs now report `lm_head_classifier_chunk_kernel_available`,
`lm_head_classifier_chunk_kernel_enabled`,
`lm_head_classifier_chunk_launch_count`, and the last rows/vocab/stride handled
by that route. `tools/paired_kernel_speed.py` includes the chunk launch count
and last rows/vocab/stride in native metric summaries and
`native_route_counter_changes`, so same-script candidate runs can prove when an
LM-head classifier route changes instead of relying only on coarse stage timing.
This keeps the LM-head classifier path auditable while the next kernel step
replaces the ABI internals with a cooperative classifier plus dHidden/dWeight
kernel.
`NFN_NATIVE_GPT_LM_HEAD_FUSED_LOSS_BACKWARD=0` (or the GPT-2 alias
`NFN_NATIVE_GPT2_LM_HEAD_FUSED_LOSS_BACKWARD=0`) disables the default fused
loss-accumulate+dlogits classifier path for same-script bisection, making
training loss use the separate loss-partials reduction before CE backward.
Runtime JSON reports `lm_head_fused_loss_backward_enabled`,
`lm_head_ce_loss_backward_fused_enabled`, and
`lm_head_ce_loss_backward_strategy`. Keep the default enabled: the CUDA 13.3
RTX 5090 confirmation run for the disabled route regressed train-loop wall time
to `1.001484x` and failed the block-backward gates.

The native dense-GPT BF16 LM-head CE backward path now uses reverse
row-chunk traversal by default because paired dedicated-RTX-5090 CUDA 13.3
timing measured it slightly faster for the current tied LM-head workspace.
Set `NFN_NATIVE_GPT_LM_HEAD_REVERSE_CHUNKS=0` or
`NFN_NATIVE_GPT2_LM_HEAD_REVERSE_CHUNKS=0` to reproduce the previous forward
chunk order during bisection.
`NFN_NATIVE_GPT_LM_HEAD_DWEIGHT_BEFORE_DHIDDEN=1` is a diagnostic-only
row-chunk order probe that runs LM-head dWeight before dHidden after CE writes
dlogits; the dedicated RTX 5090 5-step, 3-sample check measured `1.001048x`
train-loop wall time and `0.998959x` tokens/sec, so the default remains
CE -> dHidden -> dWeight.
`NFN_NATIVE_GPT_LM_HEAD_PIPELINE_CHUNKS=1` is a new opt-in LM-head schedule
candidate for same-script benchmarking. It doubles the bounded BF16 logit
scratch from one 8,192-row chunk to two chunks, computes logits/CE on the
default stream, and queues dHidden plus ordered dWeight on side streams while
the next chunk starts. Runtime JSON reports
`lm_head_pipeline_chunks_requested`, `lm_head_pipeline_chunks_enabled`,
`lm_head_pipeline_logit_buffer_count`, `lm_head_pipeline_extra_bf16_logit_bytes`,
and the active `lm_head_dhidden_dweight_schedule_strategy`. Use
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_pipeline_chunks` to route the
candidate through the standard same-script SM120 wrapper and its default stage
gates. Leave it unset for normal training until paired RTX 5090 gates prove a
durable win.
The BF16 linear operand cache is limited to stable operands such as weights;
LM-head dWeight repacks the mutable hidden activation chunks each microbatch so
gradient accumulation does not reuse stale packed activations.
The native dense-GPT tied token embedding/LM-head weight now keeps a persistent
BF16 shadow by default for LM-head logits and dHidden GEMMs while preserving the
FP32 master for token embedding, AdamW state, and checkpoint export. Set
`NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_SHADOW=0` only for paired benchmarks against
the older per-step BF16 bridge/cache route; runtime JSON reports
`token_weight_bf16_shadow_enabled` and `token_weight_bf16_refresh_count`.
The native float32-input/BF16-gradient dWeight+bias path now uses the optimized
cuBLASLt bias-gradient epilogue by default for supported shapes; set
`NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD=0` to reproduce the previous
split dWeight plus Tile bias-reduction path in paired benchmarks.
The cuBLASLt planner reports selected heuristic metadata in shape stats so
candidate bisections can prove which cuBLASLt plan actually ran. On the CUDA
13.3 RTX 5090 path the dense GPT MLP projection dWeight shape
`3072,768,65536,N,T` reports one returned heuristic, so there is no hardcoded
default pin for that shape. Set
`NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=3072,768,65536,N,T,INDEX` only for
explicit paired candidate bisection. The SM120 native candidate wrapper also
provides default-off `cublaslt_min_waves` and `cublaslt_max_waves` profiles for
global cuBLASLt policy bisection. Leave both unset for normal training: the
CUDA 13.3 RTX 5090 3-step/3-sample gates rejected `min_waves` at
`1.002017x` train-loop wall time and rejected `max_waves` at `1.008622x`, with
block-backward regressions in both cases. Shape-specific cuBLASLt overrides
also remain diagnostic-only: overriding LM-head dWeight
`768,50304,32768,N,T` from heuristic `1` to `0` was nearly neutral but failed
strict LM-head gates at `stage.lm_head_backward.total_ms=1.000373x` and
`stage.lm_head_backward.dweight.total_ms=1.000445x`; heuristic `2` regressed
that shape to `1.078836x`. Overriding QKV dInput
`768,65536,2304,N,N` from heuristic `1` to `0` regressed the shape to
`1.082474x` and `stage.block_backward.qkv.dinput.total_ms` to `1.077788x`.
When `NFN_SM120_NATIVE_LINEAR_SHAPE_STATS=1` is enabled, the paired benchmark
JSON now reports `native_linear_shape_stats.has_cublaslt_plan_change` and
`cublaslt_plan_changed` entries, so a shape-specific plan override is treated
as kernel-level attribution even when aggregate GEMM route counters do not
change.
Dense GPT dWeight GEMMs now also match the llm.kittens accumulation contract:
the first gradient-accumulation microbatch writes dWeight with GEMM `beta=0`,
and later microbatches accumulate with `beta=1`. This is enabled by default for
LM-head, QKV, attention projection, and MLP block dWeights through the raw
Tile-CUDA beta ABI. The tied LM-head is additionally row-chunked for memory, so
only the first LM-head chunk of the first gradient-accumulation microbatch uses
`beta=0`; later LM-head chunks in that same microbatch use `beta=1` so every
token chunk contributes to `accum_grad_token_weight`. Set
`NFN_NATIVE_GPT_DWEIGHT_FIRST_MICROBATCH_BETA_ZERO=0` only for paired
benchmarks against the older always-accumulate path. Runtime JSON reports
`dweight_first_microbatch_beta_zero_enabled`,
`dweight_first_microbatch_beta_strategy`, `lm_head_dweight_beta_zero_scope`, and
`first-write-then-accumulate` strategy suffixes for the active dWeight routes.
Dense GPT native training also folds train-loss collection into the LM-head
backward recompute pass. Validation, evo candidate scoring, and other
forward-only checks still use the normal forward LM-head loss path, but training
microbatches no longer run a separate LM-head logits pass when train-loss
recording is enabled with `--train-loss-every-steps N` or
`--train-log-every N`. The backward path accumulates the same CE partials
before mutating the row-chunked logits into dLogits, keeping real token data
inside the compiled CUDA Tile/C++ loop instead of routing it through
graph-editor nodes. The default cadence is `0` for timing-only training and
benchmarks; use `--eval-every-steps N` separately for validation loss.
Dense GPT training now treats scalar attention fallback as invalid by default:
runtime JSON reports `optimized_attention_required: true` and fails the run if
`attention_forward_scalar_launch_count` is nonzero, before final checkpoint
export. Use `--allow-scalar-attention-fallback` only for diagnostic bisections
where a slow fallback is intentionally being measured.
`python tools/check_native_no_torch_deps.py --skip-artifacts --json` verifies
that GPT, GPT-2-evo, NanoGPT, LLaMA fast/megakernel, MixLLaMA, JEPA semantic,
semantic-router MoE, DeepSeek-V4, explicit `nfn train --tinystories`, default
`nfn train`, native inference, and SDK native training handoff surfaces still
run under an import blocker for Torch, NumPy, `tiktoken`, dataset manager
imports, and `nfn_impl`. The verifier also records `elapsed_seconds` for each
native Python fast-path entrypoint and fails by default if any wrapper takes
more than 2 seconds before handing off to the compiled path; pass
`--max-entrypoint-seconds 0` only when collecting no-budget diagnostics.
For cuBLASLt BGRADB dWeight plus bias routes, the default writes the epilogue
bias gradient into Tile-owned
scratch and accumulates it into `grad_bias`. Set
`NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1`,
`NFN_NATIVE_GPT2_BGRAD_FIRST_WRITE_DIRECT=1`, or
`NFN_TILE_CUDA_LINEAR_BGRAD_FIRST_WRITE_DIRECT=1` only for paired benchmarks
against the direct first-write path.

## Current state of play

NeuralFn now ships Torch-backed template presets for:
- autoregressive families: `nanogpt`, `gpt2`, `llama`, `moe` / `mixllama`, `llama_fast`, `mixllama_fast`, `jamba`, `ternary_b158`, `ttt_llama`, `universal_llama`, `llama_megakernel`, and `kv_pca_llama`
- non-AR research/overlay families: `seq2seq`, `diffusion`, `llm_jepa`, `dense_jepa_evo`, `moe_jepa_evo`, `semantic_dense_jepa_evo`, `semantic_moe_jepa_evo`, and `hnet_lm`

### Research Experiments

> **Warning:** Experimental presets are research prototypes. Their APIs, performance targets, and architectural choices are exploratory and subject to change or removal. Do not depend on stability.

- **`semantic_router_moe`** -- [Experimental] AR-only MixLLaMA/MoE control preset for testing the semantic router in isolation. It keeps the normal LLaMA attention path and MoE expert MLPs, computes a single shared vocab-grounded semantic route from the pre-block hidden state, broadcasts that route across every sequence position, and trains with next-token CE plus semantic-alignment loss. It requires one expert per active semantic vocabulary dimension and its root graph is pre-wired with a text `dataset_source` (`tokens`, `targets`) plus a `semantic_data_source` (`sem_targets`).
- **`jepa_semantic_hybrid`** -- [Experimental] Hybrid JEPA Semantic LLM that fuses a Joint Embedding Predictive Architecture with a grounded semantic vocabulary, LSH bucketing, a fixed dimension-to-expert router, and attention-capable experts that operate over the full masked hidden sequence before the LM head. Training combines three connected losses: autoregressive next-token CE, JEPA latent MSE, and masked semantic topic cross-entropy. The root graph is pre-wired with a text `dataset_source` that emits `tokens` and `targets`, plus a `semantic_data_source` that emits vocab-derived `sem_targets`. Semantic vocabulary and routing metadata live in `neuralfn/data/semantic/vocab_86d_*.json`. See `neuralfn/semantic.py` for the data layer.
- **`dense_jepa_evo` / `moe_jepa_evo`** -- [Experimental] non-semantic AR+JEPA Evo controls. Both train with next-token CE plus JEPA latent alignment and use only the text contract `(tokens, targets)`. `dense_jepa_evo` uses dense LLaMA FFNs; `moe_jepa_evo` uses standard MoE routing and load-balance loss without the semantic router, semantic data source, or route-evolution controller.
- **`gpt2_evo`** -- [Experimental] dense GPT-2 with one evolution-trained layer. `build_gpt2_evo_spec()` builds a 12-layer GPT-2 where a single designated block (`layer_evo_index`, default the middle layer) is excluded from gradient optimization; every ~10 optimizer steps an interleaved evolutionary search evaluates the current block weights plus gaussian mutants forward-only on the current batch and adopts the best candidate (the current weights are always candidate 0, so the candidate loss never regresses). `nfn_gpt2_evo_native_train --print-plan` is a compiled C++ preflight that reports the AdamW/NVFP4/evo-layer schedule against the real GPT-2 tokenizer vocabulary (`50257`) by default, not a reduced parameter-golf vocabulary. Dense GPT-2-compatible evo training now delegates to `nfn_gpt_native_train --train-transformer-lm --layer-evo`, so real training runs stay on the no-Torch CUDA Tile path and runtime JSON reports `graph_editor_tensor_flow: false`, `forward_candidate_evals`, `candidate_loss_source: "native-forward-loss-device-resident-current-batch"`, and `candidate_loss_transport: "device-to-device"`. The raw Tile-CUDA trainer ABI includes device-side evo candidate mutation, best-loss selection, and best-candidate adoption, and `nfn_gpt2_evo_native_train --smoke-evo-kernels --tile-ops-lib PATH` loads the raw Tile ops library plus CUDA runtime, launches those three kernels on synthetic device buffers, verifies best-loss selection/adoption by device-to-host copyback, and exits without Python, Torch, datasets, or graph-editor tensor flow. Direct `python cli/scripts/train_gpt2_evo.py --print-plan ...`, `--native-cuda-print-plan ...`, or `--native-cuda-dry-run ...` now prefers that family-specific binary directly when `NFN_NATIVE_GPT2_EVO_CLI`, `build/nfn_gpt2_evo_native_train`, or an installed `nfn_gpt2_evo_native_train` is available; wrapper-level `--native-cuda-print-plan`, `--native-cuda-check-tile-ops`, `--native-cuda-smoke-*`, `--native-cuda-tile-ops-lib`, and `--native-cuda-cuda-runtime-lib` aliases are normalized before forwarding. The compiled GPT-2 evo binary also accepts the same native-cuda aliases directly for print-plan, smoke, and library-path preflight commands. Training entrypoints no longer expose a TorchTrainer escape hatch; legacy graph-backed experiments must call Python SDK trainer APIs directly instead of routing through `nfn train` or direct `cli/scripts/train_*.py` execution. Inference can run existing exported artifacts with `nfn infer --graph ~/NeuralFn/artifacts/gpt2_evo.json --weights ~/NeuralFn/artifacts/gpt2_evo.pt --prompt "..."` or `python cli/scripts/infer_gpt2.py --evo --prompt "..."`.

  GPT-2 evo native command inspection is two-stage and stays compiled: `nfn-native-train --base-model gpt2-evo --dry-run --print-command ...` prints the family preflight command with `--print-command` preserved, and `nfn_gpt2_evo_native_train --native-cuda-dry-run --native-cuda-print-command ...` prints the final dense GPT CUDA Tile delegate with `--train-transformer-lm --layer-evo` before token-shard resolution, dataset scanning, Torch imports, or graph-editor tensor flow.
- **`semantic_dense_jepa_evo`** -- [Experimental] dense control for the Semantic JEPA Evo stack. It keeps the chunk-level causal semantic planner, JEPA target encoder, AR CE, JEPA latent alignment, and semantic-alignment losses, but uses normal dense LLaMA FFNs instead of the shared/semantic/free expert bank and does not run route evolution.
- **`semantic_moe_jepa_evo`** -- [Experimental] full Semantic MoE JEPA Evo template. It keeps the autoregressive decoder dense through attention, updates a causal semantic planner at chunk boundaries, routes the next chunk through 2 always-on shared experts, 86 interpretable semantic experts, and 8 free learned experts, then applies a lightweight route-evolution controller to router bias state on a configurable fraction of batches. Its root graph uses the same flat compiled contract `(tokens, targets, sem_targets)` and adds route balance, route selection, route distillation, JEPA latent alignment, semantic alignment, and AR CE losses.

![Semantic MoE JEPA Evo architecture](docs/assets/semantic_moe_jepa_evo_architecture.png)

The full GPT architecture diagram catalog is in [Templates and Presets](docs/framework-guide/templates-and-presets.md#architecture-diagrams). It includes generated diagram sheets for the core autoregressive templates, objective/research templates, and semantic routing templates:

- [Core GPT template architectures](docs/assets/gpt_template_architectures_core.png)
- [Objective and research GPT template architectures](docs/assets/gpt_template_architectures_research.png)
- [Semantic GPT template architectures](docs/assets/gpt_template_architectures_semantic.png)

The detailed Semantic MoE JEPA Evo architecture diagram is kept as the original PNG asset so the pasted layout and formatting render as intended.

Backend capabilities (`TemplateSpec.backend_capabilities`) now drive runtime behavior:
- **cache** -- KV cache nodes (`kv_cache_read` / `kv_cache_write`) can be inserted into attention graphs for inference-time autoregressive caching. `InferenceCache` in `neuralfn/inference.py` wraps a compiled graph for stateful step-by-step generation.
- **quantized_export** -- `export_quantized_pt` / `import_quantized_pt` support int8 per-channel and ternary weight quantization for smaller checkpoint files. Int8 export now leaves token/position embedding tables at full precision and quantizes only linear/projection weight tensors, matching the documented storage-optimization contract. `KVQuantPackStage` now performs real int8 quantization instead of a plain concat.
- **megakernel** -- the `llama_megakernel` preset uses `runtime="megakernel"` which fuses the entire attention layer into a single `FusedCausalAttentionStage` module and compiles with `torch.compile(mode="max-autotune", fullgraph=True)`.
- **PCA KV cache** -- the `kv_pca_llama` preset sets `compression="kv_pca"`, inserting `kv_pca_encode` / `kv_pca_decode` nodes around the KV path in attention graphs to compress cached keys/values to a lower dimension.
- **CUDA Tile backend** -- `neuralfn.tile_cuda` exposes optional CUDA Tile runtime diagnostics, a coverage report for every current builtin/module/function/optimizer target, per-kernel dtype support matrices, and an opt-in source build path. The registry currently accounts for all 138 training-relevant entries: 129 Tile-covered kernels/compositions, 7 host-only interface/source nodes, and 2 delegated compiled-graph calls, with no `torch_fallback` entries. Implemented coverage includes 25 scalar function kernels plus elementwise/layout/routing/attention/cache/loss/embedding/position/norm/projection/reduction module and optimizer kernels such as `gelu`, `token_embedding`, `linear`, `bitlinear_ternary`, `fp8_linear`, `mx_linear`, `nf4_linear`, `ttt_linear`, `lora_linear`, `randmap_adapter`, `mlp_relu2`, `swiglu`, `geglu`, `reglu`, `solu`, `mamba`, `universal_transformer`, `jepa_projector`, `jepa_predictor`, `mask_scheduler`, `random_timesteps`, `jepa_mask`, `lm_head`, `tied_lm_head`, `router_logits`, `topk_route`, `scaled_dot_product_attention`, `sliding_window_attention`, `block_sparse_attention`, `streaming_attention_sinks`, `native_sparse_attention`, `differential_attention`, `causal_self_attention`, `fused_causal_attention`, `multi_latent_attention`, `routed_attention_experts`, `expert_dispatch`, `auxfree_load_balancing`, `semantic_projector`, `semantic_chunk_projector`, `semantic_moe_router`, `semantic_hash_router`, `semantic_moe_jepa_evo_router`, `attentionless_decoder`, `value_head`, `reward_head`, `denoise_head`, `dropout`, `adamw_step`, `muon_newton_schulz`, `muon_step`, `split_optimizer_step`, `ema_update`, `gradient_accumulate`, `gradient_clip_norm`, `kv_pca_encode`, `kv_pca_decode`, `kv_quant_pack`, `kv_quant_unpack`, `act_halt_gate`, `act_weighted_sum`, `latent_pool`, `token_cross_entropy`, `masked_token_cross_entropy`, `sequence_logp`, `dpo_pairwise_loss`, `ppo_clipped_loss`, `gae_compute`, `preference_bce_loss`, `load_balance_loss`, `route_balance_loss`, `route_selection_loss`, `route_distillation_loss`, `semantic_alignment_loss`, `semantic_hasher`, `semantic_chunk_hasher`, `softmax_distillation_loss`, `rotary_embedding`, `rms_norm`, `layer_norm`, `group_norm`, `qk_norm`, `logit_softcap`, `residual_add`, `qk_gain`, `dyt`, `reshape_heads`, `repeat_kv`, `absolute_position_embedding`, `byte_patch_embed`, `kv_cache_read`, `kv_cache_write`, `broadcast_expert_routes`, `broadcast_chunk_routes`, `byte_patch_merge`, `causal_chunk_state`, and `latent_mse_loss`. PyTorch remains the fallback for unsupported devices, dtypes, shapes, and tensor contracts.

CUDA Tile source builds require CUDA Toolkit 13.3+, `cuda_tile.h`, C++20,
`nvcc --enable-tile`, and `ninja`. Install the optional native build extra with
`pip install -e ".[tile-cuda]"`; it does not install Torch. Legacy graph-backed
Torch workflows, including the PyTorch Tile extension loader, require a
separately managed PyTorch install outside NeuralFn's package extras. Root
`nfn --help` / no-argument startup,
`nfn train|infer|eval --help`, `nfn kernels ... --help`,
`nfn kernels list [--json]`, CUDA Tile registry imports, and the explicit native
GPT training dispatcher avoid importing `nfn_impl` and Torch. Explicit CLI
`--kernel-backend tile-cuda` runs build/load the extension automatically and now
defaults to strict kernel enforcement; pass `--no-tile-cuda-strict` only for
debugging fallback behavior. `nfn kernels bench` uses paired interleaved samples
by default: each sample times graph-walk PyTorch, compiled PyTorch, and
Tile-requested execution in alternating order, then reports mean seconds and
paired ratios so unrelated external GPU load affects baseline and candidate
measurements similarly. For native binary or environment-flag experiments, run
`tools/paired_kernel_speed.py` with `--baseline "OLD_COMMAND"`,
`--candidate "NEW_COMMAND"`, and `--samples N` to compare older and candidate
kernels in the same alternating script. For dense GPT SM120 kernel bisection,
`tools/bench_native_gpt_sm120_candidate.sh` wraps that helper with the same
native command on both sides, selected-GPU idle guards, and
`--train-batch-tokens 524288` by default. Set
`NFN_SM120_NATIVE_CANDIDATE_ENV="KEY=VALUE OTHER=1"` or
`NFN_SM120_NATIVE_CANDIDATE_ENV="KEY=VALUE,OTHER=1"` to test an env-gated
candidate against the current default, or set
`NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN=/tmp/nfn_gpt_native_train_candidate` to
compare a candidate compiled trainer executable against
`build/nfn_gpt_native_train`. Set
`NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB=/tmp/libnfn_candidate.so` to compare a
candidate Tile ops build against `build/libnfn_native_train_tile_ops.so`. Common
controls include `NFN_SM120_NATIVE_STEPS`, `NFN_SM120_NATIVE_SAMPLES`,
`NFN_SM120_NATIVE_WARMUP`, `NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES`,
`NFN_SM120_NATIVE_TEMPLATE_NAME`, `NFN_SM120_NATIVE_GRAPH_FILE`, and
`NFN_SM120_NATIVE_STAGE_TIMING=1` for attribution sidecars. Generic
`NFN_SM120_*` aliases are accepted as lowest-priority fallbacks for shared
shape/output controls, while native-specific names still win over candidate and
parity aliases. Set candidate-only
CLI flags with `NFN_SM120_NATIVE_CANDIDATE_EXTRA_ARGS`, the natural
`NFN_SM120_NATIVE_CANDIDATE_ARGS` alias, or the short
`NFN_SM120_CANDIDATE_EXTRA_ARGS`; use `NFN_SM120_NATIVE_EXTRA_ARGS` or
`NFN_SM120_COMMON_EXTRA_ARGS` only when both commands must receive the same
argument. Set
`NFN_SM120_NATIVE_DRY_RUN_PLAN=1` to emit the resolved paired commands, selected
CUDA device policy, and alternating sample order without launching the GPU jobs.
LM-head row-order bisections can use
`NFN_SM120_CANDIDATE_ENV=NFN_NATIVE_GPT_LM_HEAD_REVERSE_CHUNKS=0` to compare
the previous forward chunk traversal against the current reverse-row-chunk
default. After the CUDA 13.3 reinstall on the dedicated RTX 5090, a guarded
three-sample same-script gate measured reverse traversal faster on average
(`0.997183x` train-loop wall time), so it is now the workstation default.
Set
`NFN_SM120_NATIVE_STARTUP_ONLY=1` for startup bisections; the wrapper appends
`--startup-only` to both baseline and candidate while preserving the same
dataset, batch, checkpoint-disabled, selected-GPU, and idle-load controls. When
a measured startup-only run has candidate-only changes and no explicit
`NFN_SM120_NATIVE_MAX_CANDIDATE_RATIO` override, the wrapper gates
`setup_wall_ms=1.000` instead of the absent `train_loop_wall_ms_per_step`
metric, so startup candidates are judged on setup time rather than rejected for
missing train-loop metrics. Startup-only candidates whose candidate library
path, candidate env, or candidate-only CLI args mention token-weight
initialization also get an automatic
`setup.token_weight_init.total_ms=1.000` gate, so token initializer changes
must improve their own measured setup stage and cannot pass only because other
startup work moved in the opposite direction. Common token-weight startup
bisections can use
`NFN_SM120_NATIVE_CANDIDATE_PROFILE=token_weight_vector4_strided`,
`token_weight_threaded`, `token_weight_fast_int32`, or
`token_weight_two_pass_bf16` instead of spelling out the individual env flags;
these profiles are diagnostic-only and do not change the default vector4 fused
BF16-shadow initializer. After the CUDA 13.3 reinstall on the dedicated RTX
5090, a 9-sample startup-only gate for `token_weight_vector4_strided` passed
mean setup/token-init gates (`0.997653x` setup wall, `0.996379x` token init)
but stayed too close to noise, with median setup slightly slower, so it remains
default-off. Add
`--append-native-profile-json-dir /tmp/nfn-profiles`
when comparing native NeuralFn commands that do not already write JSON; the
harness appends unique `--profile-json` files without changing the timed native
command. Add `--native-stage-timing` only for attribution runs that should set
`NFN_NATIVE_GPT_STAGE_TIMING=1` and report paired `stage.*` metrics such as
`stage.train.model_forward.total_ms`, `stage.block_forward.total_ms`,
`stage.block_recompute.total_ms`, and `stage.block_backward.total_ms` beside
total step time. The text report also includes block-backward child metrics for
MLP projection/FC, attention projection, attention SDPA, QKV, and LayerNorm
residual substages, plus final-norm, embedding-backward, gradient-zero,
gradient-clip, and AdamW update support stages, so candidate gates can target
metrics such as
`stage.block_backward.qkv.dinput.total_ms` or
`stage.block_backward.attn_proj.dinput.total_ms` directly. When native profile
JSON includes packed-attention backward section counters, the same summary also
prints dprep/TK timing fields such as `attention_backward_tk_timing_us` and
their candidate-over-baseline ratios. Set
`NFN_SM120_NATIVE_LINEAR_SHAPE_STATS=1`,
`NFN_SM120_CANDIDATE_LINEAR_SHAPE_STATS=1`, or
`NFN_SM120_PARITY_LINEAR_SHAPE_STATS=1` for route-attribution runs that should
enable `NFN_NATIVE_GPT_LINEAR_SHAPE_STATS=1` on both native commands. The paired
JSON and text output then include `native_linear_shape_stats`, sorted by
baseline shape time, with per-shape average time ratios and cuBLASLt selected
heuristic lists when the v2 stats ABI is available. Leave this disabled for
normal throughput comparisons because shape stats synchronize measured GEMMs.
Paired native summaries also carry categorical startup route values such as
`device_allocator_strategy`, `device_cuda_malloc_async_enabled`, and
`token_weight_init_strategy`, so allocator and token-initializer candidates show
that the intended native path was selected.
If a
native command exits nonzero after
writing its profile JSON, the harness failure message includes the native
`status` and `error` fields so CUDA-driver, symbol, or dataset failures are not
hidden behind an empty stderr. Use
`tools/bench_native_gpt_sm120_parity.sh` for the canonical RTX 5090
SM120 parity check against `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh`;
it runs the llm.kittens `train_gpt2cu` reference and
`build/nfn_gpt_native_train --backend tile-cuda` through the same paired harness
with selected-GPU idle guards. The NeuralFn candidate side passes
`--train-batch-tokens 524288` explicitly so it stays locked to the reference
`train-sm120.sh` `-d 524288` contract even if native trainer defaults change.
The parity wrapper uses the same default selected-GPU idle polling policy as
the native candidate wrapper: three utilization samples spaced 0.25 seconds
apart before each measured command, configurable through
`NFN_SM120_PARITY_SELECTED_GPU_UTILIZATION_RETRIES` /
`NFN_SM120_SELECTED_GPU_UTILIZATION_RETRIES` and the matching
`..._RETRY_INTERVAL_SECONDS` aliases.
Measured parity runs also default to
`train_loop_wall_ms_per_step=1.000` as a metric-ratio gate, so the wrapper exits
nonzero when NeuralFn is slower than llm.kittens on the comparable in-loop
training metric. Override with `NFN_SM120_PARITY_MAX_CANDIDATE_RATIO` or
`NFN_SM120_MAX_CANDIDATE_RATIO` only for diagnostic sweeps; dry-run plans stay
ungated.
The parity wrapper defaults short runs to
timing-only sample/checkpoint cadence (`NFN_SM120_PARITY_SAMPLE_EVERY=0`,
`NFN_SM120_PARITY_CHECKPOINT_EVERY=0`); compare
`train_loop_wall_ms_per_step` and `train_tokens_per_second` in the native
metrics block rather than child-process `seconds`, because the llm.kittens
reference still performs its built-in validation passes around short runs. It
also writes NeuralFn native sidecars through `--append-native-profile-json-dir`,
defaulting to
`/tmp/nfn_sm120_parity_profiles_${NFN_SM120_PARITY_STEPS:-10}step`; set
`NFN_SM120_PARITY_PROFILE_DIR` to keep those profiles somewhere else, or set it
to `none`, `off`, or `0` when you need an actual throughput comparison without
JSON sidecars. Candidate timeouts kill the command process group before the
result is recorded, so oversized CUDA experiments do not leave a native trainer
running on the selected GPU after the paired harness moves on. Set
generic `NFN_SM120_*` aliases for common parity controls only when you want the
same shell environment to be reusable across parity and native-candidate
wrappers; explicit `NFN_SM120_PARITY_*` values override them. Set
`NFN_SM120_PARITY_STAGE_TIMING=1` when you need CUDA-event
stage attribution; stage-timed parity runs default
`NFN_NATIVE_GPT_STAGE_TIMING_MAX_EVENTS=80000` unless already set, so 10-step
SM120 sidecars should not silently truncate stage totals. Set the
cadence variables to `20000` and `200` when deliberately reproducing the full
script sample/checkpoint cadence. `tools/build_native_train_tile_ops.sh`
defaults to the SM120 ThunderKittens-backed bf16 attention bridge when
`LLM_KITTENS_ROOT` and `TK_ROOT` point at the local llm.kittens and
ThunderKittens checkouts; that build now mirrors the llm.kittens SM120 NVCC
threading, host-compiler, data-prep, memory, LayerNorm, and cuBLASLt GEMM
tuning flags while leaving runtime dispatch on NeuralFn's initialized cuBLASLt
path. CUDA Toolkit 13.3 rejects several raw TK GEMM instantiations that exceed
the default static shared-memory cap, so the trainer-facing shared-library build
defines `LLMK_SM120_USE_CUBLASLT_GEMM` by default and normalizes inherited
`NFN_TILE_CUDA_ARCH=sm_120` / `compute_120` settings to `sm_120a` /
`compute_120a` when TK attention is enabled. Set `NFN_TILE_CUDA_USE_TK_ATTENTION=0`
for the older float32 row-scan diagnostic build, or override
`NFN_TILE_CUDA_ARCH` explicitly.

For same-script kernel candidate builds, `tools/build_native_train_tile_ops.sh` accepts whitespace-separated `NFN_TILE_CUDA_EXTRA_NVCC_FLAGS` and `NFN_TILE_CUDA_EXTRA_LDLIBS`, for example `NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP -DLLMK_SM120_APPROX_DGELU_TANH=1" bash tools/build_native_train_tile_ops.sh /tmp/libnfn_candidate.so`; leave those variables unset for the default supported library.

The SM120 bridge initializes the llm.kittens cuBLASLt handles before dispatching
through the default compile-mode path. Raw TK GEMM bisections that intentionally
avoid cuBLASLt should be built as local experiments only, because CUDA 13.3 may
reject over-shared kernel instantiations at `ptxas` time.

Native GPT BF16 cross-entropy kernels default to 1024 threads per row. For paired launch-configuration bisection, set `NFN_NATIVE_GPT_CE_BF16_THREADS`, `NFN_NATIVE_GPT2_CE_BF16_THREADS`, or `NFN_TILE_CUDA_CE_BF16_THREADS` to one of `128`, `256`, `512`, or `1024`; unsupported values fall back to 1024.

For BF16 classifier dlogit store bisection, set `NFN_NATIVE_GPT_CE_BF16_VEC_STORES=1`, `NFN_NATIVE_GPT2_CE_BF16_VEC_STORES=1`, or `NFN_TILE_CUDA_CE_BF16_VEC_STORES=1` to test the opt-in 128-bit streaming-store path. It remains disabled by default because the CUDA 13.3.33 RTX 5090 paired benchmark after the BF16 vector-load default measured scalar stores as the steadier route.

The default scalar-store BF16 classifier path still uses vectorized BF16 reads
for the final dlogit pass when `NFN_NATIVE_GPT_CE_BF16_VEC_LOADS` is enabled.
This keeps the llm.kittens-style 128-bit read pattern without promoting the
previously rejected streaming-store path. The CUDA 13.3.33 dedicated RTX 5090
same-script gate measured the final-pass vector-load candidate at `0.995665x`
mean train-loop wall time, `0.997949x` LM-head backward time, and `0.994846x`
CE time versus the prior scalar final-pass load path.
`NFN_NATIVE_GPT_CE_BF16_VEC_NORMAL_STORES=1`,
`NFN_NATIVE_GPT2_CE_BF16_VEC_NORMAL_STORES=1`, or
`NFN_TILE_CUDA_CE_BF16_VEC_NORMAL_STORES=1` is a separate default-off
diagnostic for normal cached `int4` packed dlogit stores. It is not promoted:
the CUDA 13.3.33 RTX 5090 same-script stage gate measured CE at `1.002720x`
against the scalar-store default, so scalar stores remain the normal route.

Native BF16 `cublasGemmEx` fallback paths default to `CUBLAS_COMPUTE_32F` for the non-cuBLASLt cases. For same-script fallback bisection, set `NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF=1` or `NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_FAST_16BF=1` to test `CUBLAS_COMPUTE_32F_FAST_16BF` without changing cuBLASLt dispatch. The fallback algorithm can also be bisected without rebuilding: set `NFN_NATIVE_LINEAR_BF16_GEMM_EX_ALGO=N` / `NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_ALGO=N` for a global `CUBLAS_GEMM_ALGO<N>_TENSOR_OP` probe, or set `NFN_NATIVE_LINEAR_BF16_GEMM_EX_ALGO_SHAPE=m,n,k,opA,opB,N` / `NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_ALGO_SHAPE=...` for one hot shape such as LM-head dHidden `768,8192,50304,N,N,0`. Use `default` or `default_tensor_op` to force the cuBLAS defaults. These switches are diagnostic-only; unset them for normal training so the existing per-call default remains unchanged.

When `libnfn_native_train_tile_ops.so` is built without the trainer cuBLAS linear fast path, large-row linear dWeight fallbacks now use a shared-memory 2D tiled CUDA kernel for float32-output dWeight accumulation across float32/BF16 activation and gradient combinations. The normal workstation build still tries cuBLAS/cuBLASLt first; the tiled fallback only replaces the older row-chunked atomic dWeight reduction after those GEMM routes are unavailable. Bias-only fallback reductions keep the shared row-chunk path.

`tools/bench_native_gpt_sm120_parity.sh` now defaults `NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=auto`, so the parity run selects an idle display-disabled NVIDIA GPU on mixed display/compute workstations. It also defaults selected-GPU utilization polling to three samples with a 0.25 second interval before each measured command, matching the native-vs-native candidate wrapper. Set `NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0` or another explicit value to pin the benchmark manually, and tune `NFN_SM120_PARITY_SELECTED_GPU_UTILIZATION_RETRIES` / `NFN_SM120_PARITY_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS` only when the local NVML idle signal needs a different policy.
Set `NFN_SM120_PARITY_DRY_RUN_PLAN=1` to inspect the llm.kittens baseline, NeuralFn candidate, CUDA selection, and sample order before starting a long parity run.

Scalar CUDA Tile function kernels, simple elementwise modules, stochastic dropout, norm modules, verified projection-family modules, RoPE, verified attention modules, verified loss/reduction modules, and selected optimizer/runtime helpers support contiguous CUDA `float32` and `float16` tensors. The projection coverage includes `linear`, LM/router/value/reward/denoise heads, KV PCA projections, JEPA heads, deterministic LoRA/TTT/adapter projections, `bitlinear_ternary`, `fp8_linear`, `mx_linear`, MLP projections, and ACT halt projection. The attention coverage includes SDPA, sparse attention variants, differential attention, causal/fused causal attention, MLA, and routed attention experts with fp32 route-weight accumulation. Loss/reduction coverage includes token CE, masked CE, sequence logp, latent MSE, semantic alignment, DPO, PPO, GAE, preference BCE, load/route balance, route selection/distillation, and softmax distillation. Optimizer/runtime fp16 coverage includes `ema_update`, `gradient_accumulate`, `gradient_clip_norm`, `adamw_step`, `muon_step`, and `split_optimizer_step` with fp16 parameter/gradient buffers plus float32 optimizer state; standalone `muon_newton_schulz` remains float32-only as the matrix orthogonalization primitive. The `float16` path computes through the Tile `float32` kernels and casts activation outputs back to preserve stable math while keeping module parameters, optimizer moments, weights, masks, reductions, attention score/softmax accumulation, routing probabilities, and scale gradients in float32. Training-mode dropout uses deterministic counter-based masks for fp32/fp16 activations instead of the PyTorch RNG fallback. Broader fp16 module coverage plus fp8 and NVFP4 variants are tracked in `todo-tile-cuda.md`.

Projection-family modules now also accept CUDA `float8_e4m3fn` and `float8_e5m2` activation inputs where the output contract is safe: `linear`, LM/router/value/reward/denoise heads, tied LM head, KV PCA encode/decode, JEPA heads, deterministic LoRA/TTT/adapter projections, `bitlinear_ternary`, `fp8_linear`, `mx_linear`, MLP projections, and ACT halt projection. These fp8 activations are dequantized to float32, accumulated in float32, and return float32 outputs with float32 weight/bias gradients; branching composite projections dequantize once so internal gradient accumulation stays in float32.

Projection-family modules also accept packed `NVFP4Tensor` activation inputs for `linear`, LM/router/value/reward/denoise heads, tied LM head, KV PCA encode/decode, JEPA heads, deterministic LoRA/TTT/adapter projections, `bitlinear_ternary`, `fp8_linear`, `mx_linear`, MLP projections, and ACT halt projection. NVFP4 activations dequantize through NeuralFn block/tensor scale metadata, accumulate through the Tile float32 path, return float32 outputs, and can preserve source-activation gradients through `quantize_nvfp4_reference(..., preserve_grad=True)` for STE-style training checks. `nf4_linear` remains outside this contract because its base weights already use a separate packed NF4 representation.

Attention modules now accept CUDA `float8_e4m3fn` and `float8_e5m2` Q/K/V activation inputs where score and softmax accumulation can remain fp32: SDPA, sliding/window/block/native sparse variants, streaming sink attention, differential attention, causal/fused causal attention, MLA, and routed attention experts. These paths dequantize fp8 activations to float32, return float32 outputs, and dequantize shared composite/routed inputs before projection or expert fan-out so gradient accumulation stays in float32.

The same attention-family modules also accept packed `NVFP4Tensor` activation inputs. Q/K/V or shared attention inputs dequantize through NeuralFn NVFP4 scale metadata before Tile attention, RoPE, projection, and route-weight fan-out; score and softmax accumulation stay fp32, and outputs remain float32.

Compiled CUDA Tile graphs can opt into runtime NVFP4 activation packing with `graph.torch_config["tile_cuda_activation_dtype"] = "nvfp4"`. The packer only wraps activation input ports for modules whose registry advertises NVFP4 support, leaving tied weights, masks, targets, losses, optimizer state, and graph editor/source nodes as normal tensors. `cli/scripts/train_gpt.py` is the canonical native-only GPT entrypoint; `cli/scripts/train_gpt2.py` remains a compatibility wrapper. Direct execution with the default `compiled-cli` runner translates GPT flags to the compiled C++ CLI and runs it before importing `train_gpt_native.py`; importing the compatibility module, building its parser, or resolving native defaults still does not import Torch, `server.dataset_manager`, NumPy, or tiktoken. Dense GPT native training can be requested as `--base-model gpt`, `gpt2`, `gpt3`, or `nanogpt`; all four aliases dispatch to the same compiled dense GPT target while `--template-name` or `--graph-file` records the selected architecture. NanoGPT defaults add `--template-name nanogpt`, which currently reports `template-geometry-native-trainer-missing` for full-transformer training until the dense loop can use the NanoGPT width/head/dropout geometry. When no template is passed, the script and master CLI both use the generic `gpt` template selector, not the compatibility `gpt2` label. Plan and runtime JSON now canonicalize all dense GPT aliases to `model_family: "gpt"` and expose the architecture contract with `architecture_source`, `architecture_contract`, `model_family_context_policy`, and `native_geometry_contract`. That object is the machine-readable truth for the current compiled loop: `gpt2-compatible-fixed-dense-transformer`, `shape_source: "compiled_dense_gpt_defaults"` for preset selection or `"custom_graph_template_spec"` for compatible custom graph metadata, model dim 768, 12 heads, GELU 4x MLP, GPT-2 tokenizer vocab 50,257, padded LM-head rows 50,304, absolute positions, LayerNorm, dropout 0, dynamic template geometry disabled, and custom graph geometry enabled only when an existing graph file carries native-compatible GPT `template_spec` metadata. It also reports `selected_template_geometry` plus `geometry_matches_compiled_loop`, so selecting `nanogpt` visibly records the requested 320-wide/5-layer/dropout-0.1 preset shape even though the current transformer-LM loop is still fixed. `gpt3` supplies the 2048-context default when selected through `--base-model gpt3` or `--template-name gpt3`; unless `--batch-size` is explicit, the native CLI also uses batch size 32 so the default microbatch remains 65,536 tokens. Compatible custom graphs can provide sequence length and layer-count defaults from their `template_spec`; explicit `--train-seq-len`, `--batch-size`, and `--num-layers` remain authoritative. The generic SDK module `neuralfn.native_gpt` exports generic `NativeGptRunConfig`, `NativeGptRunnerStatus`, and `NativeGptCheckpointInfo` classes plus helpers such as `build_native_gpt_compiled_cli_run_config()`, `native_gpt_activation()`, and `run_native_gpt()` over the same no-Torch native path. The native GPT default dataset is TinyStoriesV2 GPT-4 (`roneneldan__TinyStories__TinyStoriesV2-GPT4`) with the GPT-2 tokenizer; `golf1` and `golf10` remain explicit cached-token shortcuts only. The native path resolves cached uint16 train/validation shards with the shared C++ native token-shard resolver and launches the compiled GPT CUDA trainer directly instead of routing training batches through graph-editor nodes or `TorchTrainer`; raw-text materialization and HuggingFace downloads still lazy-load the dataset manager only when needed. When the default `compiled-cli` runner is selected and cached `fineweb_train_*.bin` plus validation shards already exist, Python now passes the alias/path directly to the compiled resolver without reading `meta.json` or revalidating shard metadata first. The compiled resolver also accepts llm.kittens-style `TinyStories_train.bin` / `TinyStories_val.bin` token bins, and `--tinystories` resolves to `/mnt/disk2/dev/open-source/llm.kittens/dev/data/tinystories` when those files exist; set `NFN_LLM_KITTENS_TINYSTORIES_DIR` to override that directory, or pass a direct `TinyStories_train.bin` path and the native resolver will infer the sibling validation bin. The script sets up its own repo/script import path, so direct `python cli/scripts/train_gpt.py ...`, compatibility `python cli/scripts/train_gpt2.py ...`, and `runpy`-style native invocations do not require `PYTHONPATH`. `cli/scripts/infer_gpt.py` is the matching canonical GPT inference entrypoint; `cli/scripts/infer_gpt2.py` remains a compatibility wrapper with the same startup discipline for orientation/defaults. Importing either inference module, building its parser, resolving `--evo`/`--megakernel` artifact defaults, or running `python cli/scripts/infer_gpt.py --help` does not import Torch, the dataset manager, or NumPy; graph-backed `.pt/.json` generation imports that runtime only after argument parsing. Native `train_gpt2cu` and NeuralFn GPT checkpoints (`model_########.bin` plus optional `DONE_########`) are recognized by `nfn infer --native-checkpoint ... --native-info`, `nfn infer --checkpoint ... --native-info`, and `python cli/scripts/infer_gpt.py --native-checkpoint ... --native-info` without importing Torch, so the CLI reports their native runtime shape instead of treating them as graph-backed `.pt` files. For native checkpoint prompt generation, `nfn infer --native-checkpoint model_########.bin --prompt "..."`, `nfn infer --checkpoint model_########.bin --prompt "..."`, and `python cli/scripts/infer_gpt.py --native-checkpoint model_########.bin --prompt "..."` tokenize the text with the GPT-2 tokenizer and dispatch to `nfn_gpt_native_train --sample-checkpoint ... --prompt-tokens ...`; explicit `--prompt-tokens` uses the same compiled CUDA Tile sampler directly. `--native-sampler-script` and `NFN_NATIVE_GPT_SAMPLE_SCRIPT` are deprecated and no longer needed for native GPT `.bin` checkpoint prompts. The master CLI routes default dense GPT training commands to `nfn_gpt_native_train --backend tile-cuda --train-transformer-lm` before importing `train_gpt_native`, `nfn_impl`, or Torch; `tile-cuda` is the only native GPT training backend, NanoGPT defaults dispatch to the native dense GPT selector with `--template-name nanogpt` and fail fast on the current geometry mismatch, compatible dense GPT custom graph files dispatch to the same native trainer, arbitrary or incompatible custom graphs report `custom-graph-native-trainer-missing`, and unsupported families are rejected by the native model registry. The old `NFN_ALLOW_TORCH_TRAINING` training bypass is intentionally ignored by CLI entrypoints. The compiled GPT trainer treats `train_batch_tokens` as the real optimizer-step batch: it derives `grad_accum_steps = ceil(train_batch_tokens / (batch_size * seq_len))`, streams that many cached-shard microbatches through CUDA Tile forward/backward kernels, averages gradients in device accumulation buffers, then clips and runs AdamW once. GPT-compatible causal attention now uses the SM120 ThunderKittens bf16 FlashAttention-style bridge by default in the trainer-facing Tile ops library, and training JSON reports `attention_backend_strategy: "tk-sm120-bf16-bridge"`, TK launch counts, and zero row/scalar launches when that path is active. The SM120 default `64 x 1024 -> 524288` therefore runs eight native microbatches per optimizer step, while the GPT3 default `32 x 2048 -> 524288` preserves the same token microbatch; JSON reports the requested/effective token batch plus accumulation fields. Build the workstation C++ binding with `bash tools/build_native_gpt2_binding.sh`, standalone launcher with `bash tools/build_native_gpt2_launcher.sh`, no-Python cached-shard CLI with `bash tools/build_native_gpt_cli.sh`, and unified native frontend with `bash tools/build_native_train_cli.sh`; `cli/install.sh` also links `nfn-gpt-native`, `nfn-gpt-native-train`, `nfn-native-train`, and `nfn-gpt2-tile-launcher` into the active Python scripts directory. `nfn-native-train --list-models` or `nfn-native-train --list-models --json` reports `gpt`, `gpt2`, and `gpt3` as implemented aliases of the same dense GPT native trainer; `nanogpt` is known but reports a full-transformer geometry mismatch until that native loop is generalized. `NFN_DATASETS_DIR=/path/to/datasets` overrides the native alias cache root for Python and compiled native CLI paths; `nfn-native-train --base-model gpt --dataset-alias PATH_OR_ALIAS` or `nfn-gpt-native --dataset-alias PATH_OR_ALIAS` is the fastest path when shards already exist. The CLI default is now `--native-cuda-runner compiled-cli`, so installed dense GPT training commands require the no-Python cached-shard CLI by default; pass `--native-cuda-runner auto`, `binding`, or `launcher` only when you intentionally want SDK binding or launcher selection. Native GPT launchers and the master `nfn train` native dispatcher default `CUDA_VISIBLE_DEVICES=0` and `CUDA_DEVICE_MAX_CONNECTIONS=1` when those variables are unset, matching the workstation layout where the RTX 5090 is the dedicated CUDA compute GPU; set either environment variable explicitly to override that routing. Use `--eval-every-steps 1000` for per-1000-step validation loss, or pass `--eval-every-steps 0`, `--native-cuda-sample-every 0`, and `--native-cuda-checkpoint-every 0` to disable those cadences for same-script kernel benchmarks; the SDK preserves zero instead of clamping it back on. Use `--native-cuda-print-command` to inspect the resolved native invocation, and `--native-cuda-config-out PATH` to save it as JSON. The 5090 shell helpers no longer pass wallclock caps; they run to the configured step schedule unless you add one explicitly. Legacy graph-backed training scripts, including GPT-2 evo, now dispatch to compiled native entrypoints or fail before importing Torch; use Python SDK trainer APIs directly for one-off graph-backed debugging while native C++ trainers are added.

Native compiled entrypoints, SDK bindings, the master `nfn train` native dispatcher, the direct `cli/scripts/train_gpt.py` / `train_gpt2.py` compiled-CLI fast path, and legacy training-script native guards set `CUDA_MODULE_LOADING=LAZY` when unset before executing native trainers or loading Tile CUDA libraries, matching the dense GPT C++ trainer. Existing user-provided `CUDA_MODULE_LOADING` values still take precedence.

The unified native model registry exposes capability-specific status fields in addition to the legacy top-level status. `nfn-native-train --list-models --json` now includes `transformer_lm_status`, `token_lm_status`, and `geometry_status` for each model. `nanogpt` is reported as `partial-native-trainer`: its full transformer path still reports `template-geometry-native-trainer-missing` until the dense native loop supports NanoGPT geometry, while the explicit `--train-token-lm` path remains implemented.

Unsupported GPT templates and custom graph selections are rejected by the compiled native GPT CLI before token-shard resolution when they are not runnable by the current native loop. The error JSON reports `token_shards_resolved: false`, so missing datasets do not hide native graph/template coverage gaps or add avoidable startup work.

For same-script native GPT benchmarks through `nfn train`, pass `--native-cuda-no-checkpoint` or `--no-checkpoint` to skip final checkpoint export, in addition to setting validation, sample, and checkpoint cadences to `0` when you want timing-only runs.

The GPT-2 compatibility wrapper remains import-compatible with the legacy
training scripts for parser and test helpers: it accepts `--pretraining-file`,
Tinystories and parameter-golf shortcuts, tokenizer override flags,
evolutionary flags, scheduler flags, and `--megakernel` metadata helpers
without importing Torch. Direct execution still takes the compiled native GPT
path before loading the compatibility helpers. `--all-train-rows` now uses the
documented two-epoch floor unless `--max-steps` or `ITERATIONS` is explicit,
and raw-text validation fallback uses a deterministic 10% tail holdout.
NanoGPT transformer-LM direct execution defaults to the dense GPT native
trainer with `--template-name nanogpt`; setting `NFN_NATIVE_NANOGPT_CLI`
explicitly makes the direct wrapper use the NanoGPT family command instead.

Dense GPT native training now accepts `--layer-evo` /
`--native-cuda-layer-evo` plus `--evo-layer-index`,
`--evo-layer-interval`, `--evo-layer-population`, and
`--evo-layer-mutation-scale` on the compiled C++ trainer. The current native
cadence allocates device candidate workspace for the selected block's
`block_N.ln1.weight`, runs raw Tile-CUDA mutate/select/adopt ABI kernels during
the optimizer loop, scores each candidate with the native CUDA forward loss on
the current batch, and reports `layer_evo.graph_editor_tensor_flow: false`,
`layer_evo.forward_candidate_evals`,
`layer_evo.candidate_loss_source:
"native-forward-loss-device-resident-current-batch"`,
`layer_evo.candidate_loss_transport: "device-to-device"`, and
`layer_evo.candidate_loss_host_roundtrips_elided` in plan/runtime JSON.

`nfn train --tinystories` takes the same compiled dense GPT route when `--base-model gpt` is omitted.

Native checkpoint prompt-token requests now take a compiled C++ path instead of
the transitional Python sampler bridge: `nfn infer --native-checkpoint
model_########.bin --prompt-tokens 1,2,3` and `python cli/scripts/infer_gpt.py
--native-checkpoint model_########.bin --prompt-tokens 1,2,3` dispatch to
`nfn_gpt_native_train --sample-checkpoint ... --prompt-tokens ...`. That path
validates the checkpoint, context window, vocab bounds, and token list before
CUDA, dataset setup, Torch, or graph-editor node flow, then executes
autoregressive CUDA Tile checkpoint forward passes and returns up to
`--max-new-tokens` IDs in `generated_tokens`. Text prompts are tokenized with
the GPT-2 tokenizer in the lightweight wrapper and then use the same compiled
sampler path, so native `.bin` checkpoint prompts no longer need the external
sampler bridge. After a successful compiled sampler run, the wrapper also prints
the generated token IDs and GPT-2-decoded generated text without importing
Torch.

`nfn_gpt_native_train --checkpoint-load-smoke --native-checkpoint
model_########.bin --checkpoint-load-elements 1024` is the next compiled
inference prerequisite: it reads a bounded bf16 checkpoint payload slice, copies
it to CUDA memory, converts it through `nfn_native_tile_bf16_bits_to_float32`,
and verifies GPU copyback without Torch, datasets, or graph-editor tensors.
Add `--checkpoint-load-tensor h.0.ln_1.weight` to load a named tensor slice by
the decoded checkpoint layout offset instead of the payload start.
Use `nfn_gpt_native_train --checkpoint-layout --native-checkpoint
model_########.bin` to decode the native tensor layout, payload offsets, file
offsets, and bounded payload samples from the checkpoint header as compiled C++
JSON without CUDA, datasets, Torch, or graph-editor tensors.
`nfn_gpt_native_train --checkpoint-logits-smoke --native-checkpoint
model_########.bin --prompt-tokens 1,2,3` now loads checkpoint embeddings and
final norm tensors, converts them on device, and runs token embedding, position
embedding, residual add, final LayerNorm, and tied LM-head logits through CUDA
Tile kernels for the last prompt token. Transformer blocks are still pending for
complete prompt generation.
`nfn_gpt_native_train --checkpoint-qkv-smoke --native-checkpoint
model_########.bin --prompt-tokens 1,2,3 --checkpoint-block-index 0` loads the
same checkpoint embeddings plus `h.N.ln_1` and `h.N.attn.c_attn` tensors, then
runs embedding residual, the selected block's first LayerNorm, and QKV
projection through CUDA Tile kernels. This is a checkpoint-backed transformer
block-stage smoke only.
`nfn_gpt_native_train --checkpoint-attention-smoke --native-checkpoint
model_########.bin --prompt-tokens 1,2,3 --checkpoint-block-index 0` continues
through split-to-heads, causal scaled-dot-product attention, and merge-heads on
CUDA Tile kernels.
`nfn_gpt_native_train --checkpoint-attention-residual-smoke --native-checkpoint
model_########.bin --prompt-tokens 1,2,3 --checkpoint-block-index 0` then loads
`h.N.attn.c_proj` and runs attention output projection plus residual add on
CUDA Tile kernels.
`nfn_gpt_native_train --checkpoint-block-smoke --native-checkpoint
model_########.bin --prompt-tokens 1,2,3 --checkpoint-block-index 0` continues
through `ln_2`, MLP fc, GELU+bias, MLP projection, and the final block residual
add on CUDA Tile kernels.
`nfn_gpt_native_train --checkpoint-block-logits-smoke --native-checkpoint
model_########.bin --prompt-tokens 1,2,3 --checkpoint-block-index 0` continues
through final LayerNorm and tied LM-head logits for the last prompt token.
`nfn_gpt_native_train --checkpoint-forward-logits-smoke --native-checkpoint
model_########.bin --prompt-tokens 1,2,3` runs every checkpoint GPT block in
order, then final LayerNorm and the tied LM head for the last prompt token. It
reports `transformer_blocks_executed: true`, `blocks_executed`, and
`graph_editor_node_flow: false`; generation-loop sampling remains the next
native inference step.

The compiled dense GPT trainer can inspect native `model_########.bin`
checkpoints without CUDA, Torch, Python dataset setup, or graph nodes using
`nfn_gpt_native_train --native-info --native-checkpoint model_########.bin` or
`nfn_gpt_native_train --inspect-checkpoint model_########.bin`. The JSON reports
shape, precision, file-size validation, DONE marker state, and
`prompt_generation_status: "native-token-sampler-available"`. Text prompts are
tokenized by the lightweight wrapper before it calls that native token sampler.

Native GPT launchers also default `CUDA_MODULE_LOADING=LAZY` when the variable is unset, alongside the existing dedicated-GPU defaults, so direct C++ and SDK launches avoid eager CUDA module-load startup cost unless the caller overrides the environment. Use `--startup-only` or SDK `startup_only=True` to run full Tile-CUDA transformer setup, emit `status: "native-transformer-lm-startup-ready"`, and exit before optimizer steps or checkpoint export when measuring startup. Startup-only accepts `--max-steps 0` for setup-only probes; normal training still requires a positive step count. Plan/runtime JSON reports the effective checkpoint state with `checkpoint_export_enabled: false` plus `checkpoint_export_startup_only_elided: true` when startup-only suppresses a requested final export.

Benchmark-mode native GPT runs with `--native-cuda-sample-every 0` also skip the two post-train diagnostic device-to-host samples for token weight and clip scale. Runtime JSON reports `post_train_diagnostic_samples_elided: true`, `post_train_diagnostic_sample_d2h_count: 0`, and `post_train_diagnostic_sample_d2h_count_elided: 2`; pass a positive sample cadence when you intentionally want those end-of-run diagnostics.

For the generic SDK path, build `neuralfn._native_gpt` with `bash tools/build_native_gpt_binding.sh`; build `neuralfn._native_gpt2` with `bash tools/build_native_gpt2_binding.sh` only for GPT-2 compatibility imports. `tools/build_native_gpt2_all.sh` and `cli/install.sh` build both modules, and `run_native_gpt(..., runner="auto")` prefers the generic `_native_gpt` binding before falling back to the compatibility binding, compiled CLI, or launcher. The auto route no longer falls through to an external `train_gpt2cu` subprocess when NeuralFn native artifacts are missing, and `runner="subprocess"` is no longer a GPT training runner. The native GPT bindings and generic `neuralfn._native_train` binding launch compiled native commands with `posix_spawnp()` instead of `fork()` plus `execvp()`, and set `CUDA_MODULE_LOADING=LAZY` when the caller has not supplied a module-loading policy, so SDK runs avoid avoidable Python-process fork overhead and eager CUDA module loading. The generic binding also exposes `resolve_native_train_binding_command(config)`, which returns the exact compiled argv that the binding will spawn without importing Torch or touching dataset/graph payloads.

When the Python native GPT harness or SDK compiled-CLI handoff selects a custom
graph with `--graph-file`, it now preserves graph-driven native geometry by
omitting `--batch-size`, `--train-seq-len`, and `--num-layers` unless the caller
set those fields explicitly. This lets the compiled C++ runner read compatible
`template_spec` metadata such as `seq_len` and `num_layers` directly from the
graph and adjust the microbatch shape without going through Torch or graph
editor tensors. `NativeGpt2RunConfig` and the generic `NativeGptRunConfig`
expose `batch_size_explicit`, `seq_len_explicit`, and `num_layers_explicit` for
SDK callers that need the same behavior. The compiled dense GPT loop still
requires the fixed GPT-2-compatible width/head/vocab/dropout contract until the
native loop is generalized for arbitrary dense GPT template geometry.

The explicit `llm-kittens` GPT training backend has been removed. Keep `tools/bench_native_gpt_sm120_parity.sh` for NeuralFn-vs-llm.kittens reference timing; normal CLI and SDK training stay on the NeuralFn-owned `tile-cuda` backend.

Prefer generic dense GPT environment names for current native runs. `NFN_NATIVE_GPT_STAGE_TIMING`, `NFN_NATIVE_GPT_STAGE_TIMING_MAX_EVENTS`, `NFN_NATIVE_GPT_PACKED_QKV_ATTENTION`, `NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS`, `NFN_NATIVE_GPT_STORE_MLP_BLOCKS`, `NFN_NATIVE_GPT_STORE_ATTENTION_ACTIVATIONS`, `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_ACTIVATIONS`, `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS`, `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE`, `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_BF16`, `NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS`, `NFN_NATIVE_GPT_FUSE_RESIDUAL1_STORE`, `NFN_NATIVE_GPT_BF16_RESIDUAL1_LN_BACKWARD`, `NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP`, `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_GRID3D`, `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_WARPS`, `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_HD64_SPECIALIZED`, `NFN_NATIVE_GPT_FUSE_ATTENTION_RESIDUAL_LN2`, `NFN_NATIVE_GPT_FUSE_LN2_BF16_OUT`, `NFN_NATIVE_GPT_FUSE_MLP_RESIDUAL_NEXT_LN1`, `NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU`, `NFN_NATIVE_GPT_MLP_PROJ_DINPUT_BEFORE_DWEIGHT`, `NFN_NATIVE_GPT_MLP_FC_DINPUT_BEFORE_DWEIGHT`, `NFN_NATIVE_GPT_ATTN_PROJ_DINPUT_BEFORE_DWEIGHT`, `NFN_NATIVE_GPT_BF16_MLP_GRAD_HANDOFF`, `NFN_NATIVE_GPT_BF16_PROJECTION_RESIDUAL`, `NFN_NATIVE_GPT_DIM768_BF16_RESIDUAL_ADD`, `NFN_NATIVE_GPT_ELIDE_FLOAT_PROJECTION_OUTPUTS`, `NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUTS`, `NFN_NATIVE_GPT_FUSE_QKV_BIAS_TK_GEMM`, `NFN_NATIVE_GPT_LN1_BF16_QKV_FORWARD`, `NFN_NATIVE_GPT_BF16_QKV_GRAD_HANDOFF`, `NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT`, `NFN_NATIVE_GPT_DIRECT_BF16_QKV_GRAD_SCRATCH`, `NFN_NATIVE_GPT_BF16_QKV_DWEIGHT`, `NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD`, `NFN_NATIVE_GPT_FUSE_LN_BACKWARD_RESIDUAL`, `NFN_NATIVE_GPT_LM_HEAD_BF16_LOGITS`, `NFN_NATIVE_GPT_BF16_LM_HEAD_LOSS`, `NFN_NATIVE_GPT_PUBLIC_VOCAB_CE`, `NFN_NATIVE_GPT_DIRECT_U16_TOKENS`, `NFN_NATIVE_GPT_LM_HEAD_BF16_DWEIGHT`, `NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN`, `NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS`, `NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_SHADOW`, `NFN_NATIVE_GPT_TOKEN_WEIGHT_FAST_INT32_INIT`, `NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT`, `NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS`, `NFN_NATIVE_GPT_BF16_BIAS_INPLACE_TILE`, `NFN_NATIVE_GPT_F32_TO_BF16_VEC4`, `NFN_NATIVE_GPT_F32_TO_BF16_MANY_VEC4`, `NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS_VEC4`, `NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC`, and `NFN_NATIVE_GPT_SKIP_EXIT_CUDA_FREE` configure the compiled C++ dense GPT loop regardless of whether the selected alias is `gpt`, `gpt2`, `gpt3`, a template, or a custom graph. The older `NFN_NATIVE_GPT2_*` variables remain compatibility fallbacks for existing scripts.

`NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS=1` is an opt-in dense GPT classifier diagnostic. It allocates a full BF16 LM-head logits buffer, fills it once after the transformer forward, and makes LM-head backward consume chunk offsets from that buffer instead of recomputing logits. This mirrors the llm.kittens classifier-reuse shape, but it needs about 6.14 GiB for full logits at the default `64 x 1024 x 50304` padded-vocab shape. On the 32 GiB RTX 5090, use it only with a lower saved packed-attention cap such as `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=0` or `4`. Leave it disabled for normal training: same-script CUDA 13.3 paired checks measured the zero-block reuse candidate at `1.099054x` train-loop wall time and the four-block candidate at `1.061321x`, while an eight-block smoke fit but degraded to an unusable `83.5s` one-step train loop near the memory cliff. Runtime JSON reports `lm_head_reuse_forward_logits_enabled`, `lm_head_full_logit_elements`, and `lm_head_bf16_logit_bytes`; the paired benchmark helper ratios those fields with the saved packed-attention block count.

`NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUTS=1` is a startup/memory diagnostic for the scratch-recompute transformer-LM path. It stores the inter-block persistent outputs as BF16, restores each prior block input into one FP32 scratch buffer during backward, and reports `bf16_persistent_block_outputs_enabled`, store/restore counts, and `fp32_persistent_block_output_*_elided` in runtime JSON. At the default `64 x 1024 x 768` 12-layer shape it elides `2,214,592,512` FP32 persistent-output bytes while adding `1,107,296,256` BF16 bytes plus one FP32 restore scratch. Leave it disabled for normal training: the 2026-06-18 dedicated RTX 5090 paired benchmark measured `1.021212x` train-loop wall time and `0.979238x` tokens/sec versus default, despite improving setup wall time to `0.974595x` and float-arena materialization to `0.896011x`.

`NFN_NATIVE_GPT_BF16_BLOCK_DWEIGHT_STAGING=1` is an opt-in profiling switch for the `nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_to_bf16_bits_float32` ABI. It accumulates QKV and MLP FC BF16/BF16 dWeights into BF16 staging buffers. When BF16 primary block weights are enabled, those staged gradients now feed `nfn_native_tile_sumsq_partials_many_bf16_bits_float32` and `nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32` directly, avoiding the older BF16-to-FP32 staging flush; otherwise the compatibility flush is still used. Leave it disabled for normal training: the previous flush candidate measured about `1.0245x` slower than the current cuBLASLt bgrad default, and the direct BF16 optimizer candidate measured `1.0325x` slower in train-loop time over the paired 2-step dedicated-RTX-5090 benchmark. Runtime JSON reports `block_dweight_bf16_staging_enabled`, staging allocation sizes, zero count, direct/flush strategy, BF16 clip/AdamW descriptor counts, and BF16-gradient AdamW launch counts.

`NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC=1` is an opt-in allocator profiling switch for dense GPT transformer-LM startup. It routes the trainer's large device arenas through `cudaMallocAsync` / `cudaFreeAsync` when those CUDA runtime symbols are available, falls back to `cudaMalloc` on allocation failure, and reports `device_allocator_strategy`, async symbol availability, allocation/free counts, and fallback count in runtime JSON. Leave it disabled for normal training; the latest CUDA 13.3 paired dedicated-RTX-5090 startup check with explicit arena gates measured the async allocator candidate at `1.177290x` setup wall time, `2.243472x` float-arena materialization, `1.716820x` uint16-arena materialization, and `1.176781x` total startup wall time versus the default `cudaMalloc` arena path.

Dense GPT native CLI runs default `NFN_NATIVE_GPT_SKIP_EXIT_CUDA_FREE=1`, so process-exit cleanup skips synchronous `cudaFree` calls for the large device arenas and leaves the CUDA/Tile runtime libraries loaded for process termination to reclaim. This does not change model math, optimizer state, checkpoints, or training-loop timing, but it removes the repeated multi-GB free/teardown loop from short benchmark and startup commands. Set `NFN_NATIVE_GPT_SKIP_EXIT_CUDA_FREE=0` to restore explicit device frees and runtime-library `dlclose()` for diagnostics. Runtime JSON reports `device_exit_cuda_free_elision_enabled`, `device_exit_cuda_free_skipped_count`, `runtime_library_dlclose_skipped_count`, and `timing.cleanup_wall_ms`.

Dense GPT native transformer training fuses token embedding, absolute position embedding, and the scaled embedding residual add into one Tile-CUDA launch by default through `nfn_native_tile_token_position_embedding_residual_u16_float32` for the direct-u16 token path. This removes the separate `token_out` and `position_out` FP32 activation buffers from the startup arena at the default `64 x 1024 x 768` shape. Set `NFN_NATIVE_GPT_FUSE_EMBEDDING_RESIDUAL=0`, `NFN_NATIVE_GPT2_FUSE_EMBEDDING_RESIDUAL=0`, or `NFN_TILE_CUDA_FUSE_EMBEDDING_RESIDUAL=0` to restore the older three-launch path for paired diagnostics. Runtime JSON reports `embedding_residual_fusion_enabled`, `embedding_residual_strategy`, `embedding_residual_intermediate_float_buffers_elided`, and `embedding_residual_intermediate_float_bytes_elided`.

Dense GPT native train-loss sampling copies the scalar LM-head loss back with a blocking device-to-host `cudaMemcpy` and does not issue an extra full-device `cudaDeviceSynchronize` first. Set `NFN_NATIVE_GPT_LM_HEAD_LOSS_COPY_SYNC=1` or `NFN_NATIVE_GPT2_LM_HEAD_LOSS_COPY_SYNC=1` only to reproduce the previous sync-before-copy path for paired diagnostics. Runtime JSON reports `lm_head_loss_copy_device_synchronize_enabled` and `lm_head_loss_copy_ordering`.

`NFN_NATIVE_GPT_LM_HEAD_CONCURRENT_DHIDDEN_DWEIGHT=1` is an opt-in dense GPT LM-head backward scheduling candidate. After the BF16 CE kernel overwrites a row chunk with dlogits, the trainer records a CUDA event, launches LM-head dHidden and dWeight consumers on two non-blocking CUDA streams, and synchronizes both streams before the next row chunk reuses the workspace. The default remains the serial dHidden-then-dWeight schedule because the latest CUDA 13.3 dedicated RTX 5090 same-script benchmark measured the two-stream route slower (`1.002028x` train-loop wall time, `0.997989x` tokens/sec, `1.008876x` LM-head backward stage time). Runtime JSON reports `lm_head_concurrent_dhidden_dweight_requested`, `lm_head_concurrent_dhidden_dweight_available`, `lm_head_concurrent_dhidden_dweight_enabled`, and `lm_head_dhidden_dweight_schedule_strategy`; stage timing reports `lm_head_backward.dhidden_dweight_concurrent` when the candidate path runs.

`NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1` is an opt-in packed-attention profiling switch. It makes the attention projection dInput GEMM write BF16 grad-out bits directly and feeds those bits into the packed attention backward bridge. Runtime JSON reports `attention_backward_bf16_grad_out_handoff_enabled`, `attention_backward_grad_out_dtype`, BF16 scratch sizes, and the updated QKV bridge strategy. Leave it disabled for normal training; paired dedicated-RTX-5090 timing measured the candidate `1.0155x` slower in train-loop time than the current float grad-out default.

`NFN_NATIVE_GPT_REUSE_MLP_PROJ_BF16_GRAD_OUT=0` disables the default dense GPT MLP projection backward reuse path. The default packs the MLP projection incoming gradient to BF16 once, reuses that scratch for MLP projection dWeight+bias, and feeds it into the fused dInput+dGELU raw ABI. Runtime JSON reports `block_backward_mlp_proj_bf16_grad_out_reuse_enabled` and the `tk-sm120-fused-dinput-dgelu-reused-bf16-grad-out-bf16-store-bf16-shadow-weight` strategy when the path is active.

`NFN_NATIVE_GPT_MLP_PROJ_DINPUT_BEFORE_DWEIGHT=1` is a diagnostic ordering switch for the dense GPT MLP projection backward path. It runs the fused MLP projection dInput+dGELU stage before the dWeight+bias stage to mirror the reference `matmul_backward` consumer order, while leaving the default dWeight+bias-first order in place. Runtime JSON reports `block_backward_mlp_proj_dinput_before_dweight_enabled`; leave it disabled for normal training because the dedicated RTX 5090 5-step, 3-sample paired benchmark measured it at `1.000405x` train-loop wall time and `0.999602x` tokens/sec versus the default.

`NFN_NATIVE_GPT_MLP_FC_DINPUT_BEFORE_DWEIGHT=1` is the matching diagnostic ordering switch for the dense GPT MLP FC backward path. It runs MLP FC dInput before dWeight+bias to compare against the reference consumer order, while leaving the default dWeight+bias-first order active. Runtime JSON reports `block_backward_mlp_fc_dinput_before_dweight_enabled`; leave it disabled for normal training because the dedicated RTX 5090 5-step, 3-sample paired benchmark measured it at `1.000858x` train-loop wall time and `0.999153x` tokens/sec versus the default.

`NFN_NATIVE_GPT_BLOCK_MLP_FC_CONCURRENT_DINPUT_DWEIGHT=1` and the GPT-2-prefixed alias enable a diagnostic two-stream schedule for the MLP FC backward dInput and dWeight+bias pair. The trainer records a CUDA event on the default stream, waits from non-blocking dInput and dWeight streams, launches the independent kernels on those streams, and synchronizes before LayerNorm backward consumes `grad_ln2`. Runtime JSON reports `block_backward_mlp_fc_concurrent_dinput_dweight_requested`, `block_backward_pair_streams_available`, and `block_backward_mlp_fc_concurrent_dinput_dweight_enabled`; paired benchmark summaries print those fields. Keep this default-off: the CUDA 13.3 RTX 5090 same-script gate proved the route enabled but rejected it at `1.006693x` train-loop wall time and `1.012567x` block-backward time versus the serial default.

Use `NFN_SM120_NATIVE_CANDIDATE_PROFILE=mlp_fc_concurrent_dinput_dweight` to rerun that route through the standard native-vs-native wrapper; stage-timed runs gate `stage.block_backward.mlp_fc.total_ms` in addition to the default train-loop, LM-head, block-backward, and MLP-projection totals.

`NFN_NATIVE_GPT_ATTN_PROJ_DINPUT_BEFORE_DWEIGHT=1` is the matching diagnostic ordering switch for the dense GPT attention projection backward path. It runs attention projection dInput before dWeight+bias to compare against the reference consumer order, while leaving the default dWeight+bias-first order active. Runtime JSON reports `block_backward_attn_proj_dinput_before_dweight_enabled`; leave it disabled for normal training because the dedicated RTX 5090 5-step, 3-sample paired benchmark measured it at `1.001009x` train-loop wall time and `0.999002x` tokens/sec versus the default.

`NFN_NATIVE_GPT_DIRECT_BF16_BLOCK_WEIGHT_INIT` controls startup-only initialization of the BF16-primary transformer block weights. It defaults on when `NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS` is on, initializes QKV/projection/MLP block weights directly with `nfn_native_tile_fill_many_values_bf16_bits_float32`, and skips the initial float32-to-BF16 pack. Set `NFN_NATIVE_GPT_DIRECT_BF16_BLOCK_WEIGHT_INIT=0` to reproduce the older float32 fill plus `nfn_native_tile_float32_to_bf16_bits_many` startup path while leaving BF16-primary AdamW updates enabled. Runtime JSON reports `direct_bf16_block_weight_initialization_enabled`, `block_weight_bf16_initialization_strategy`, and the split float/BF16 parameter initialization descriptor and launch counts.

When `NFN_NATIVE_GPT_FUSE_LN_BACKWARD_RESIDUAL` is enabled, the native dense GPT trainer also skips the fallback-only `grad_residual1_from_mlp` and `grad_x_from_attn` activation scratch buffers instead of reserving them in the startup float arena. Runtime JSON reports `block_state_layout.layer_norm_backward_residual_scratch_buffers_allocated`, `block_state_layout.layer_norm_backward_residual_scratch_buffers_elided`, and `block_state_layout.layer_norm_backward_residual_scratch_elements_elided` so runs show whether that fused LayerNorm residual-backward path actually reduced the allocation footprint.

By default, native dense GPT token initialization now matches the llm.kittens
contract: only the real 50,257 tokenizer rows are initialized, and the 47 padded
rows remain zero in both FP32 master and BF16 shadow storage. This lets the
public-vocab BF16 CE backward use no-pad-zero Tile entry points instead of
scrubbing padded dlogit columns every row. Set
`NFN_NATIVE_GPT_ZERO_TOKEN_PADDING=0` or `NFN_NATIVE_GPT_SKIP_CE_PAD_ZERO=0`
only for paired diagnostics against the previous full-padded-token-init and
CE-padding-scrub route. Runtime JSON reports `lm_head_ce_pad_zero_skipped`,
`token_weight_padding_zero_enabled`, `token_weight_init_elements`, and
`token_weight_padding_elements`.
`NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT` defaults on for dense GPT startup,
using the vectorized token-weight initializer while preserving the deterministic
power-of-two initialization contract and fused BF16 shadow write. Set
`NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT=0` or
`NFN_TILE_CUDA_TOKEN_WEIGHT_VECTOR4_INIT=0` to reproduce the previous fast int32
Tile initializer in paired startup benchmarks. On the CUDA 13.3 dedicated RTX
5090 startup-only check after the toolkit reinstall, the 5-sample paired gate
measured explicit vector4 against fast int32 at `0.949565x` mean token-weight
initialization time, `0.970270x` setup wall time, and `0.970405x` total wall
time, so vector4 remains the default.
`NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT=1` and
`NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1` remain startup-only profiling
switches for token-weight initialization experiments; vector4-strided and
threaded are not the default.

The trainer-facing Tile-CUDA linear ABI supports both the default BF16-primary block-weight path and the older FP32-master/BF16-shadow path for native GPT training. The trainer allocates one BF16 block-weight arena for QKV, attention projection, MLP FC, and MLP projection weights; default AdamW updates that arena directly, while the old path refreshes it with one `nfn_native_tile_float32_to_bf16_bits_many` call after parameter initialization and each AdamW update. Block forward/recompute plus block dInput GEMMs route through `nfn_native_tile_linear_weight_bf16_float32`, `nfn_native_tile_linear_weight_bf16_output_float32`, `nfn_native_tile_linear_bf16_input_weight_bf16_float32`, and `nfn_native_tile_linear_backward_input_weight_bf16_float32`. Transformer block dWeight+bias accumulation still calls `nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_float32` or `nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32`; shape-supported BF16 dWeight GEMMs request the cuBLASLt `CUBLASLT_EPILOGUE_BGRADB` epilogue. By default, BGRADB writes bias gradients into Tile-owned scratch and accumulates into the optimizer-step bias buffer; set `NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1`, `NFN_NATIVE_GPT2_BGRAD_FIRST_WRITE_DIRECT=1`, or `NFN_TILE_CUDA_LINEAR_BGRAD_FIRST_WRITE_DIRECT=1` only for paired diagnostics against direct beta-zero bias writes. Shape-supported transformer-block BF16 GEMMs use cached cuBLASLt with `CUBLAS_COMPUTE_32F_FAST_16BF` by default; set `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT=0` or `NFN_NATIVE_LINEAR_BF16_CUBLASLT=0` to force the older BF16 `cublasGemmEx` bridge. `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1` / `NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1` is diagnostic-only for LM-head-sized BF16 shapes with `k > 32768`; the 5090 paired check moved the LM-head dHidden shape to cuBLASLt but measured it slower than GEMMEx, so the default cap remains. `NFN_TILE_CUDA_LINEAR_BF16_OUTPUT_CUBLASLT=1` / `NFN_NATIVE_LINEAR_BF16_OUTPUT_CUBLASLT=1` is a default-off BF16-output cuBLASLt probe for all-BF16 and BF16-input/float-weight LM-head logits; the dedicated RTX 5090 paired check moved the logits bucket to cuBLASLt in shape stats but measured neutral/slightly slower, so GEMMEx remains the default fallback. cuBLASLt plans now retain their matmul descriptors and matrix layouts so cache hits skip per-GEMM descriptor construction; set `NFN_TILE_CUDA_CUBLASLT_DESCRIPTOR_CACHE=0` or `NFN_NATIVE_LINEAR_CUBLASLT_DESCRIPTOR_CACHE=0` only for paired profiling against the older descriptor-recreate path. `NFN_TILE_CUDA_LINEAR_CUBLASLT_WORKSPACE_MB=N` / `NFN_NATIVE_LINEAR_CUBLASLT_WORKSPACE_MB=N` can raise or lower the cuBLASLt heuristic workspace cap for paired diagnostics; the default remains 128 MiB because the normal 5-step RTX 5090 run rejected a 256 MiB cap as train-loop neutral/slightly slower. `nfn_native_tile_linear_backward_weight_bias_accumulate_float32_bf16_bits` now defaults to the mixed float32-hidden/BF16-grad cuBLASLt bgrad epilogue route for supported QKV profiling shapes; set `NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD=0` or `NFN_TILE_CUDA_LINEAR_FLOAT32_BF16_BGRAD=0` to compare against the previous split dWeight plus Tile bias-reduction path. `NFN_NATIVE_GPT_FUSE_BF16_BF16_DWEIGHT_BGRAD=0` / `NFN_TILE_CUDA_LINEAR_BF16_BF16_BGRAD=0` is the matching BF16-input/BF16-gradient split diagnostic: it keeps block dWeight on cuBLASLt/GEMMEx and runs a separate BF16 bias reduction, but the dedicated RTX 5090 paired check measured it slower than BGRADB, so the fused route remains default. `nfn_native_tile_linear_backward_input_dgelu_bf16_bits_float32` still fuses stored-BF16 MLP preactivation dGELU into the MLP projection dInput GEMM and writes the result back to the trainer's float gradient buffer; set `NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU=0` to compare against the older separate dInput plus GELU route. `nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_float32`, `nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32`, and `nfn_native_tile_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_only_float32` expose the matching raw ABI for experiments that keep that fused MLP projection gradient as BF16 bits when feeding the following MLP FC backward GEMMs; the BF16 handoff is now the default after paired dedicated-RTX-5090 timing showed it faster. `NFN_NATIVE_LINEAR_TK_DINPUT=1` remains a broad diagnostic switch for routing supported BF16/BF16 dInput shapes through the SM120 TK GEMM bridge. Prefer `NFN_NATIVE_LINEAR_TK_DINPUT_ENABLE_SHAPE=m,n,k,opA,opB` / `NFN_TILE_CUDA_LINEAR_TK_DINPUT_ENABLE_SHAPE=...` for single-shape probes, and use the matching `*_DISABLE_SHAPE` aliases for bisection; the LM-head dHidden probe `768,8192,50304,N,N` measured slower than the GEMMEx default, so no TK dInput shape is promoted. The default MLP projection backward path packs `incoming_grad` once into BF16 scratch and reuses it for dWeight+bias and fused dInput+dGELU; set `NFN_NATIVE_GPT_REUSE_MLP_PROJ_BF16_GRAD_OUT=0` to compare against the previous per-stage pack path, or `NFN_NATIVE_GPT_BF16_MLP_GRAD_HANDOFF=0` to compare against the older float-gradient handoff. `nfn_native_tile_linear_bf16_output_float32` writes BF16 LM-head logits, and `nfn_native_tile_linear_bf16_input_float_weight_bf16_output_float32` lets the compiled trainer reuse a full-microbatch BF16 final-norm hidden prepack with FP32 tied token weights for LM-head logits. `nfn_native_tile_token_cross_entropy_partials_strided_bf16_bits` computes validation/test CE partials over the public tokenizer vocab while stepping through the padded logit row stride, `nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_with_workspace` overwrites public-vocab logits with BF16 dlogits and zeroes padded dlogit columns, and the BF16 dlogits feed `nfn_native_tile_linear_backward_input_bf16_bits_float32` plus `nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32` by default through the prepacked final-norm hidden buffer. Set `NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=0` to reproduce the older per-chunk LM-head hidden packing path, or set `NFN_NATIVE_GPT_LM_HEAD_BF16_DWEIGHT=0` to reproduce the previous float-hidden LM-head dWeight path. BF16/BF16 LM-head dWeight is now the default after paired dedicated-RTX-5090 timing measured it slightly faster. Set `NFN_NATIVE_GPT_LM_HEAD_BF16_LOGITS=0` to return the tied LM-head chunks to the older optimized TF32 tensor-op `cublasSgemm` path for debugging, set `NFN_NATIVE_GPT_BF16_LM_HEAD_LOSS=0` to keep BF16 training backward while comparing validation/test loss against the older float logits loss workspace, or set `NFN_NATIVE_GPT_PUBLIC_VOCAB_CE=0` only for paired benchmarks against the previous padded-vocab CE behavior. The public GPT tokenizer vocab remains 50,257, but the native tied token embedding/LM-head tensor is padded to 50,304 rows for GEMM-friendly row counts; training JSON reports both `vocab: 50257` and `padded_vocab: 50304`, plus `lm_head_public_vocab_ce_enabled`, `lm_head_softmax_vocab`, `lm_head_logit_row_stride`, and `lm_head_padded_dlogits_zeroed`. Dry-run plan JSON reports `shape.padded_vocab_size: 50304`. Set `NFN_TILE_CUDA_LINEAR_BF16=1` or `NFN_NATIVE_LINEAR_BF16=1` only when you want the normal linear ABI to opt into the BF16 bridge for profiling or shape-specific tuning. Set `NFN_TILE_CUDA_LINEAR_CUBLASLT=1` or `NFN_NATIVE_LINEAR_CUBLASLT=1` only when profiling the normal linear ABI through the cached cuBLASLt TF32 path; on the current GPT-2 5090 shape it is measurable but slower than the SGEMM default. Tied LM-head BF16 logits use the SM120 ThunderKittens GEMM bridge by default when the Tile ops library was built with TK support; set `NFN_TILE_CUDA_LINEAR_TK_GEMM=0` or `NFN_NATIVE_LINEAR_TK_GEMM=0` to force the BF16 `cublasGemmEx` fallback for diagnostics. The raw optimizer ABI also exposes `nfn_native_tile_adamw_step_many_with_device_scale_bf16_shadow_float32`, which writes updated FP32 master weights and optional BF16 shadow entries in the same descriptor-driven Tile launch; set `NFN_NATIVE_GPT_FUSE_ADAMW_BF16_SHADOW_REFRESH=1` only after forcing `NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS=0`, because the shadow-refresh path is bypassed by the BF16-primary default. GPT training JSON reports `linear_backend_strategy`, `block_forward_linear_strategy`, `block_backward_input_linear_strategy`, `block_backward_mlp_proj_dgelu_strategy`, `block_backward_mlp_proj_bf16_grad_out_reuse_enabled`, `block_backward_bf16_mlp_grad_handoff_enabled`, `block_backward_weight_linear_strategy`, `non_block_forward_backward_linear_strategy`, `lm_head_logits_linear_strategy`, `lm_head_prepack_bf16_hidden_enabled`, `lm_head_dweight_strategy`, `linear_bias_gradient_first_write_bgrad_direct_enabled`, `linear_cublaslt_descriptor_cache_enabled`, `block_weight_bf16_shadow_strategy`, `block_weight_bf16_shadow_fused_adamw_refresh_enabled`, `block_weight_bf16_shadow_elements`, `block_weight_bf16_shadow_bytes`, `block_weight_bf16_shadow_descriptor_count`, `block_weight_bf16_refresh_count`, `block_weight_bf16_fused_adamw_refresh_count`, `adamw_bf16_shadow_refresh_strategy`, and the linear GEMM/cache counters so native runs show whether large projections used BF16-primary weights, shadow weights, cuBLASLt, the descriptor cache, the BF16 bridge, fused MLP projection dGELU, MLP projection BF16 grad-out reuse, BF16 MLP gradient handoff, fused AdamW shadow writes, TK LM-head GEMM, public-vocab CE, prepacked LM-head hidden, or BF16/BF16 LM-head dWeight. The tied LM-head row chunk defaults to 32768 rows on the dedicated RTX 5090/CUDA 13.3 workstation route, which uses about 3.30GiB of BF16 logit workspace at the default `64 x 1024` microbatch and reduces the chunk count from 8 to 2 versus the older 8192-row profile; pass `--lm-head-row-chunk-size 8192` or wrapper `--native-cuda-lm-head-row-chunk-size 8192` to reproduce that lower-memory route. Full `--train-transformer-lm` uses BF16 logits for validation/test loss and in-place tied LM-head CE backward by default, so the separate float logits and full-vocab `grad_logits` chunks are not allocated; default JSON reports `logit_workspace_elements: 0`, `grad_logit_workspace_elements: 0`, `lm_head_training_logits_dtype: "bf16"`, `lm_head_loss_logits_dtype: "bf16"`, `lm_head_bf16_loss_enabled: true`, `lm_head_prepack_bf16_hidden_enabled: true`, `lm_head_dweight_input_dtype: "bf16"`, `lm_head_ce_backward_strategy: "public-vocab-strided-fused-row-bf16-logits-dlogits"`, `lm_head_dweight_strategy: "full-final-norm-bf16-prepack-bf16-dlogit-dweight-accumulate"`, and `lm_head_grad_logits_workspace_allocated: false`.

Use `NFN_NATIVE_LINEAR_BF16_BF16_BGRAD_DISABLE_SHAPE=m,n,k,opA,opB` or
`NFN_TILE_CUDA_LINEAR_BF16_BF16_BGRAD_DISABLE_SHAPE=...` for narrower paired
bisection when a single BF16/BF16 dWeight+bias shape should use the split
dWeight plus separate bias reducer route without disabling BGRADB globally.

The current default `lm_head_ce_backward_strategy` value supersedes older notes
in the long Tile ABI paragraph: expect
`public-vocab-strided-no-pad-zero-fused-row-bf16-logits-dlogits` when zero token
padding and public-vocab BF16 CE are enabled.

`lm_head_dhidden_linear_strategy` identifies the tied LM-head dHidden route
from shape stats when profiling is enabled, distinguishing the default
BF16 GEMMEx path from TK or cuBLASLt dInput candidates in paired benchmark
JSON.

For paired BF16 cuBLASLt shape bisection, set
`NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=m,n,k,opA,opB` or
`NFN_NATIVE_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=m,n,k,opA,opB` to route one
shape bucket, such as `768,65536,3072,N,N`, back through BF16 `cublasGemmEx`
while every other supported BF16 shape stays on the default cuBLASLt path. Set
`NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE=m,n,k,opA,opB` or
`NFN_NATIVE_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE=m,n,k,opA,opB` to force one
otherwise-gated BF16 shape through cuBLASLt for paired diagnostics. The
dedicated RTX 5090 check for LM-head dHidden shape `768,8192,50304,N,N`
measured this allow-list route slower than GEMMEx, so it remains diagnostic-only.

The compiled trainer, SDK, Python wrappers, and root CLI all share that 8192-row LM-head chunk default. Use the explicit LM-head row-chunk flags only when reproducing an older smaller-workspace profile or profiling a new candidate.

For BF16 LM-head CE profiling, `NFN_NATIVE_GPT_CE_BF16_EXP2=1`, `NFN_NATIVE_GPT2_CE_BF16_EXP2=1`, or `NFN_TILE_CUDA_CE_BF16_EXP2=1` switches the in-place BF16 CE+dlogits kernel from `expf` to `exp2f(x * log2(e))`. The default remains `expf`; runtime JSON reports `lm_head_ce_bf16_exp2_enabled`, and the dedicated RTX 5090 paired check measured the exp2 candidate noise-equivalent/slightly slower.

`NFN_NATIVE_GPT_FULL_BATCH_LM_HEAD_REUSE=1` is a diagnostic-only submode for the existing `NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS=1` resident-logit probe. It computes the LM-head logits, CE/dlogits, dHidden, and dWeight over the full `batch x sequence` row range instead of keeping the old 8192-row chunk schedule while the full BF16 logit arena is resident. The switch remains off by default: the CUDA 13.3.33 RTX 5090 one-step smoke reduced `stage.lm_head_backward.total_ms` to `0.652349x`, but regressed total train-loop wall time to `32.086572x` because the resident-logit memory footprint made downstream `block_backward.attn_sdpa` collapse. Runtime JSON reports `lm_head_full_batch_reuse_schedule_enabled` and `lm_head_dhidden_dweight_schedule_strategy: "resident-full-logit-single-row-batch-gemms"` when this diagnostic path is active. Keep the default row-chunked recompute route until a fused/cooperative LM-head backward kernel avoids the full resident-logit memory cliff.

Native GPT now defaults block projection weights to the BF16-primary update path. Token/position/norm/bias tensors still use the float32 multi-buffer AdamW ABI, while QKV, attention projection, MLP FC, and MLP projection weights update their BF16 parameter buffers directly through `nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_float32`; checkpoint export syncs those BF16 block weights back into FP32 staging buffers before the existing version-5 BF16 checkpoint packer runs. The raw Tile ABI also exports `nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32` for BF16-primary parameter updates that consume BF16 gradient buffers with FP32 AdamW moments, which is the bridge for the next block-gradient-buffer migration. `nfn_native_tile_sumsq_partials_many_bf16_bits_float32` computes float32 global-norm partials from BF16 gradient buffers so that migration can preserve clipping semantics. The dense GPT trainer now binds and reports those BF16-gradient primitives, but current block gradients still accumulate in float32 buffers until the BF16 gradient arena and zeroing path lands. Set `NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS=0` to reproduce the older FP32-master plus BF16-shadow refresh path for bisection. Runtime JSON reports `block_weight_bf16_primary_param_update_enabled`, `block_weight_bf16_gradient_storage_strategy`, `block_weight_bf16_primary_param_update_count`, `block_weight_bf16_primary_param_bf16_grad_update_count`, `adamw_bf16_param_bf16_grad_kernel_loaded`, `gradient_clip_bf16_sumsq_kernel_loaded`, `adamw_float_update_descriptor_count`, `adamw_bf16_param_descriptor_count`, `adamw_bf16_param_bf16_grad_descriptor_count`, `adamw_float_update_kernel_launches`, `adamw_bf16_param_kernel_launches`, `adamw_bf16_param_bf16_grad_kernel_launches`, and `checkpoint.bf16_param_sync_kernel_launches`.

Native GPT startup initializes the tied token FP32 master weight with the vector4
CUDA Tile deterministic initializer and writes its persistent BF16 LM-head shadow
in the same ABI call,
`nfn_native_tile_init_gpt2_token_weight_fast_with_bf16_shadow_float32`, when
`NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_SHADOW=1` (the default). The low-level Tile ABI
defaults to the vectorized float4/BF16-shadow variant for GPT-sized tables when
no token-init environment variable is set, so direct ABI calls, the compiled
trainer, and runtime JSON agree. The trainer-facing Tile build defaults
token-weight initialization to a 4096-element CUDA Tile shape for the non-vector
fallback; compile candidate libraries with
`NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="-DNFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE=1024"`,
`2048`, or `8192` only for paired bisection. The 8192-tile candidate is accepted
for local benchmark builds but is not the default: the dedicated RTX 5090
9-sample startup-only comparison measured `1.005585x` token-init time versus
4096, with total startup `0.990436x` inside broader arena-materialization noise.
`NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT=0` or
`NFN_TILE_CUDA_TOKEN_WEIGHT_VECTOR4_INIT=0` reproduces the previous fast int32
CUDA Tile initializer, `NFN_NATIVE_GPT_TOKEN_WEIGHT_FAST_INT32_INIT=0` restores
the older int64 Tile-index initializer for paired startup bisection after
vector4 is disabled, `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1` or
`NFN_TILE_CUDA_TOKEN_WEIGHT_THREADED_INIT=1` is diagnostic-only for comparing the
not-promoted threaded CUDA initializer, set
`NFN_NATIVE_GPT_TOKEN_WEIGHT_INIT_LEGACY_MOD17=1` to reproduce the older
modulo-17 deterministic values, or set
`NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_INIT=0` to reproduce the older two-pass
startup path for paired benchmarks. Runtime JSON reports
`token_weight_init_strategy`, `token_weight_threaded_init_enabled`,
`token_weight_vector4_init_enabled`, `token_weight_fast_int32_init_enabled`,
`token_weight_init_legacy_mod17_enabled`,
`token_weight_bf16_initial_refresh_fusion_enabled`, and
`token_weight_bf16_initial_refresh_elided`; `--startup-only` is the cleanest way
to measure this setup-only change.
The CUDA 13.3.33 post-reinstall retile sweep also rejected the other supported
compile-time token-init tile sizes against the 4096 default: 8192 measured
`1.013697x` token-init time, 2048 measured `1.010289x`, and 1024 measured
`1.016591x` in startup-only paired runs, so token-init retile work is not the
next useful startup path.

The 8192-row tied LM-head chunk default remains the current local optimum on
the dedicated RTX 5090. A CUDA 13.3.33 post-reinstall stage-timed sweep rejected
`--lm-head-row-chunk-size 16384` (`1.016019x` train-loop wall,
`1.062838x` LM-head backward, with dHidden at `1.244198x`) and
`--lm-head-row-chunk-size 4096` (`1.004875x` train-loop wall; CE improved to
`0.961277x` but dWeight regressed to `1.039507x`). Keep row-chunk flags for
reproducing older profiles; the remaining LM-head gap needs a fused/cooperative
classifier-backward kernel or a materially different GEMM route.

For the trainer-facing BF16 cuBLASLt block GEMMs, NeuralFn now selects cuBLASLt heuristic index 1 by default when that candidate is available on the workstation RTX 5090 shape. Use `NFN_TILE_CUDA_CUBLASLT_HEURISTIC_INDEX=N` or `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX=N` only for paired kernel profiling. Set `NFN_TILE_CUDA_CUBLASLT_HEURISTIC_POLICY=min_waves` / `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_POLICY=min_waves` to compare against the llm.kittens-style lowest-waves selector, or `max_waves` to compare against the highest-waves selector; explicit index overrides still win. For a narrow hot-shape bisection, set `NFN_TILE_CUDA_CUBLASLT_HEURISTIC_SHAPE=m,n,k,opA,opB,index` or `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=m,n,k,opA,opB,index`; the shape-specific override applies only to the matching cuBLASLt plan and leaves the default/global heuristic selection in place for every other GEMM.

Trainer-facing BF16/BF16 linear backward GEMMs also allow larger cuBLASLt shapes by default, covering the dense GPT LM-head dHidden/dWeight chunk shapes that previously fell back to BF16 `cublasGemmEx`. Set `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_LARGE_SHAPES=0` or `NFN_NATIVE_LINEAR_BF16_CUBLASLT_LARGE_SHAPES=0` to reproduce the older small-shape-only cuBLASLt gate for paired bisection.

The current dense GPT trainer also routes the stored-MLP fused FC+bias+GELU path and fused MLP-projection dInput+dGELU path through BF16 block-weight shadows. The new raw ABI symbols are `nfn_native_tile_linear_weight_bf16_gelu_bf16_float32`, `nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_float32`, `nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_float32`, and `nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32`; the older FP32-weight fused symbols remain exported for raw kernel tests and non-shadow callers.

Native GPT attention now defaults to a packed-QKV SM120 TK bridge on the compiled Tile-CUDA path. The trainer fuses QKV bias into the SM120 TK BF16 QKV GEMM by default, runs LN1 through `nfn_native_tile_layer_norm_with_stats_bf16_out_float32`, feeds the resulting BF16 activation to `nfn_native_tile_linear_bf16_input_weight_bf16_output_float32`, runs packed TK attention directly over the row-major packed QKV tensor, and feeds the packed BF16 attention output directly into the attention projection GEMM and dWeight accumulation. Set `NFN_NATIVE_GPT_LN1_BF16_QKV_FORWARD=0` to reproduce the previous float32-LN1 QKV forward path. Set `NFN_NATIVE_GPT_FUSE_QKV_BIAS_TK_GEMM=0` to reproduce the older separate packed BF16 QKV bias-add launch; then set `NFN_TILE_CUDA_BF16_BIAS_INPLACE_TILE=0`, `NFN_NATIVE_GPT_BF16_BIAS_INPLACE_TILE=0`, or `NFN_NATIVE_GPT2_BF16_BIAS_INPLACE_TILE=0` only when comparing Tile versus scalar CUDA for that old bias launch. Backward now keeps packed attention `dQKV` in BF16 by default: `nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_merged_grad_float32` and the saved-LSE variant write BF16 gradient bits directly into a non-aliased BF16 scratch buffer that reuses the MLP BF16 scratch after MLP backward is done, then QKV dWeight+bias uses the saved LN1 BF16 activation through `nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32` and QKV dInput uses the BF16-gradient/BF16-weight input-backward ABI. The packed attention dprep step keeps the older row-linear launch by default; set `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_GRID3D=1` only for paired timing of the diagnostic 3D batch/head/time launch, which avoids per-row division/modulo but measured slower on the dedicated RTX 5090. `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_WARPS=N` is a diagnostic row-grouping bisection knob for that dprep launch; the default remains `3`. Set `NFN_NATIVE_GPT_ATTENTION_BACKWARD_SECTION_TIMING=1` only for diagnostic one-off runs; it uses CUDA events to synchronize and report packed-attention backward dprep and TK section totals as `attention_backward_dprep_timing_us`, `attention_backward_dprep_timing_count`, `attention_backward_tk_timing_us`, and `attention_backward_tk_timing_count`, with `NFN_NATIVE_GPT2_ATTENTION_BACKWARD_SECTION_TIMING` and `NFN_TILE_CUDA_ATTENTION_BACKWARD_SECTION_TIMING` as fallbacks. The combined uint16 arena reserves all three scratch regions used by packed attention, in order: LN1 BF16 output, packed QKV BF16, and packed attention output BF16; the older float32 `grad_qkv` scratch allocation is elided when BF16 QKV gradient handoff is active. Set `NFN_NATIVE_GPT_ELIDE_QKV_FLOAT_GRAD_SCRATCH=0` to reproduce the previous float scratch reservation for paired memory/timing checks. Set `NFN_NATIVE_GPT_DIRECT_BF16_QKV_GRAD_SCRATCH=0` to reproduce the older workspace-to-packed-QKV-buffer copy path, set `NFN_NATIVE_GPT_BF16_QKV_DWEIGHT=0` to reproduce the previous float32-LN1/BF16-grad QKV dWeight path, or set `NFN_NATIVE_GPT_BF16_QKV_GRAD_HANDOFF=0` when profiling the older float32 `grad_qkv` expansion path. The packed backward batch cap now defaults to 64, matching the workstation `64 x 1024` microbatch in one TK backward chunk; set `NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP=48` to reproduce the previous split for paired benchmarks. Set `NFN_NATIVE_GPT_PACKED_QKV_ATTENTION=0` only when comparing against the older split-to-heads bridge. Plan and runtime JSON report `packed_qkv_attention_enabled`, packed BF16 scratch bytes, `packed_qkv_float_attention_tape_elided`, `packed_qkv_float_attention_tape_elements_elided`, `qkv_forward_ln1_bf16_enabled`, `qkv_bias_fused_tk_gemm_enabled`, `qkv_bias_layout_strategy: "packed-qkv-bf16-bias-fused-tk-gemm"`, `attention_projection_input_strategy`, `attention_packed_output_unpack_strategy`, `attention_backward_bf16_qkv_grad_handoff_enabled`, `attention_backward_qkv_float_grad_scratch_elided`, `attention_backward_qkv_float_grad_scratch_bytes_elided`, `attention_backward_direct_bf16_qkv_grad_scratch_enabled`, `attention_backward_direct_bf16_qkv_grad_scratch_elements`, `qkv_backward_layout_strategy: "packed-qkv-bf16-gradient-handoff"`, `attention_backward_qkv_bridge_strategy: "tk-sm120-packed-qkv-direct-bf16-grad-scratch-handoff"`, `block_backward_bf16_qkv_dweight_enabled`, `block_backward_qkv_dweight_strategy: "packed-ln1-bf16-qkv-bf16-grad-dweight-bias-accumulate"`, and `attention_backward_strategy: "tk-sm120-packed-qkv-bf16-backward-direct-bf16-grad-scratch-handoff"` when this default path is active. `attention_backward_tk_launch_count` now counts packed backward chunks, so forced smaller caps visibly increase the counter. The packed route now saves packed BF16 QKV, packed O, per-row TK `lse`, and earlier-block LN1 BF16 outputs by default on the dedicated RTX 5090 workstation shape, reuses those tensors during backward, and reports `packed_attention_activation_storage_strategy: "packed-qkv-o-bf16-forward-store-direct-backward"`, `stored_packed_attention_activation_blocks: 12`, `stored_packed_attention_lse_enabled: true`, `stored_packed_attention_lse_bytes`, `stored_packed_attention_ln1_bf16_*`, `stored_packed_attention_*`, `stored_packed_attention_backward_consumer_strategy: "saved-packed-qkv-o-lse-bf16-backward-to-qkv"`, `attention_backward_strategy: "tk-sm120-packed-qkv-bf16-saved-activation-backward-direct-bf16-grad-scratch-handoff"`, and `block_recompute_saved_packed_attention` timing fields. Set `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_BF16=0` for the previous lower-memory saved-attention LN1 apply-stats recompute route; the default saved LN1 BF16 tape costs about 1.03 GiB at the default `64 x 1024 x 768` shape. Set `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_ACTIVATIONS=0` for the previous lower-memory recompute path, set `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE=0` to reproduce the older shared-workspace-LSE path for paired benchmarks, or set `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=N` to tune the cap; the GPT-2-prefixed names remain fallbacks. The packed recompute path also stores intermediate block `residual1` tensors as BF16 by default and consumes them directly in LN2 backward, skipping recomputed attention projection, projection-residual work, and the older BF16-to-FP32 residual1 restore for the 11 earlier blocks in a 12-layer run. The default store is fused into `nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_bf16_norm_float32`, which writes both the residual1 BF16 cache and the prepacked LN2 BF16 activation consumed by the stored-MLP FC+GELU path in one launch. Dense GPT now defaults to skipping the unused FP32 attention-projection and MLP-projection scratch-tape buffers while BF16 projection-residual is active; set `NFN_NATIVE_GPT_ELIDE_FLOAT_PROJECTION_OUTPUTS=0` only for paired bisection against the older reservation. When the BF16 LN2 output is the only training consumer, the fused residual+LN2 Tile kernels now skip the redundant FP32 `ln2_out` store by default; set `NFN_NATIVE_GPT_ELIDE_LN2_BF16_NORM_FLOAT_STORE=0` to restore the older FP32 norm-output write for paired bisection. The dense GPT forward loop also fuses each stored MLP projection bias/residual into the next block's LN1 stats and BF16 output when packed LN1 storage or the scratch tape is available; set `NFN_NATIVE_GPT_FUSE_MLP_RESIDUAL_NEXT_LN1=0` to reproduce the previous next-block LN1 launch. Runtime JSON reports `mlp_residual_next_ln1_fusion_enabled`, `mlp_residual_next_ln1_fusion_count`, and `mlp_residual_next_ln1_strategy`. Set `NFN_NATIVE_GPT_FUSE_LN2_BF16_OUT=0` to reproduce the previous separate `float32_to_bf16` LN2 prepack before MLP FC+GELU. Default backward consumes residual1 and float LN1 inputs through the fused affine+dInput+residual-add LayerNorm backward ABI (`nfn_native_tile_layer_norm_backward_affine_residual_add_accumulate_with_stats_float32` and the BF16-bits variant), replacing the older affine-accumulate plus dInput/residual pair. Set `NFN_NATIVE_GPT_FUSE_LN_BACKWARD_AFFINE_RESIDUAL=0` to reproduce the previous split LayerNorm backward pair, set `NFN_NATIVE_GPT_BF16_RESIDUAL1_LN_BACKWARD=0` to compare against the older restore-to-FP32 LayerNorm backward path, set `NFN_NATIVE_GPT_FUSE_RESIDUAL1_STORE=0` to compare against the older separate `float32_to_bf16` residual store, or set `NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS=0` to disable this cache for lower-memory comparisons. The residual1 cache uses about 1.03 GiB at the default `64 x 1024 x 768` shape and reports `residual1_activation_storage_strategy`, `residual1_activation_store_strategy`, `residual1_backward_consumer_strategy`, `stored_residual1_activation_blocks`, `stored_residual1_activation_elements`, `stored_residual1_activation_bytes`, `fused_ln2_bf16_out_enabled`, `fused_ln2_bf16_norm_float_store_elision_enabled`, `stored_mlp_ln2_bf16_prepack_strategy`, `stored_mlp_ln2_bf16_fused_store_kernel_launches`, `stored_mlp_ln2_bf16_float_store_elided_count`, `stored_mlp_ln2_bf16_float_store_elided_elements`, `float_projection_outputs_elided`, `float_projection_output_elements_elided`, `layer_norm_backward_affine_residual_fusion_enabled`, `layer_norm_backward_affine_residual_fused_kernel_launches`, and store/restore launch counters.

The trainer-facing native GELU ABI follows the GPT-style tanh approximation for `nfn_native_tile_gelu_float32`, `nfn_native_tile_gelu_add_bias_float32`, `nfn_native_tile_gelu_backward_float32`, and `nfn_native_tile_gelu_backward_inplace_float32`. The graph-backed Torch path can keep PyTorch GELU semantics, but native GPT-2 forward, fused bias+GELU, and backward kernels now stay internally aligned on `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`.

Within the packed-attention row-linear dprep route, the GPT `heads=12, head_dim=64` BF16-grad subcase defaults to a specialized unrolled HD64 kernel when `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_HD64_SPECIALIZED` is unset. Set `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_HD64_SPECIALIZED=0` to reproduce the older generic row dprep kernel for paired bisection; `NFN_NATIVE_GPT2_PACKED_ATTENTION_DPREP_HD64_SPECIALIZED` remains the fallback name.
Keep the HD64 specialized path enabled for normal training: the CUDA 13.3 RTX
5090 3-step/3-sample generic dprep probe passed total train-loop and
block-backward gates but failed the strict attention substage gates at
`stage.block_backward.attn_sdpa.total_ms=1.000768x` and
`stage.block_backward.attn_sdpa.to_qkv.total_ms=1.000766x`.
Keep the packed-attention backward batch cap at the default 64 on the
workstation shape. A CUDA 13.3 RTX 5090 3-step/3-sample
`NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP=32` probe doubled
`attention_backward_tk_launch_count` from `288` to `576` and regressed
train-loop wall time to `1.006515x`, block backward to `1.014077x`,
attention backward to `1.072186x`, attention `to_qkv` to `1.072565x`, and
`attention_backward_tk_timing_us` to `1.063855x`.
Keep `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_WARPS` at the default `3`.
Although `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_WARPS=4` passed a short
3-step/3-sample gate, the longer CUDA 13.3 RTX 5090 10-step/3-sample gate
rejected it at train-loop wall time `1.004580x`, block backward `1.009456x`,
attention backward `1.000428x`, attention `to_qkv` `1.000395x`, and
`attention_backward_tk_timing_us` `1.000828x`.
The SM120 candidate wrapper also exposes `attention_atomic_dq` as a compile-time
profile for the alternate `LLMK_SM120_ATOMIC_DQ` attention-backward route. It
is for regression bisection only on the current dense GPT shape: the CUDA 13.3
RTX 5090 3-step/3-sample gate rejected it at train-loop wall time `1.137273x`,
block backward `1.273200x`, attention backward `2.250757x`, attention `to_qkv`
`2.253944x`, and `attention_backward_tk_timing_us` `2.464325x`.

The raw trainer ABI also includes `nfn_native_tile_layer_norm_with_stats_float32`, `nfn_native_tile_layer_norm_backward_input_with_stats_float32`, `nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_float32`, `nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32`, `nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_float32`, and `nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32` so native trainers can store per-row LayerNorm mean/rstd during forward and reuse those stats in backward instead of recomputing them. Dense GPT-2 uses that stats-reuse path by default, stores earlier-block LN2 stats beside the BF16 MLP activations, and fuses block LayerNorm dInput with the following residual gradient add through `nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_float32`. Set `NFN_NATIVE_GPT_FUSE_LN_BACKWARD_RESIDUAL=0` to compare against the older separate LayerNorm dInput plus residual-add route. Training JSON reports `layer_norm_stats_strategy`, `layer_norm_backward_reuses_forward_stats`, `layer_norm_backward_residual_fusion_enabled`, `layer_norm_backward_residual_strategy`, `stored_mlp_layer_norm_stats_elements`, and `stored_mlp_layer_norm_stats_bytes`. `nfn_native_tile_linear_bias_residual_layer_norm_with_stats_float32` fuses attention-projection bias, residual add, and LN2 while writing mean/rstd for the existing backward stats-reuse path, and `nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_float32` additionally writes the BF16 residual1 cache in the same launch. GPT-2 enables the fused stats path by default. Set `NFN_NATIVE_GPT2_FUSE_ATTENTION_RESIDUAL_LN2=0` only when comparing against the older separate residual add plus LN2 route. Training JSON reports `attention_residual_ln2_strategy` and launch-elision counters so profiling runs can prove whether the fused kernel was active.

Set `NFN_NATIVE_GPT_STAGE_TIMING=1` on `--train-transformer-lm` runs when you need a CUDA-event stage breakdown for the native loop. The normal host timing block still avoids extra synchronizations; this opt-in profiler records `stage_timing_enabled`, `stage_timing_max_events`, event/drop counts, and per-stage totals/averages for token upload, model forward, block forward/recompute/backward, LM-head backward, embedding/final-norm backward, gradient zero/clip, and AdamW update under `timing.stage_timing`. Diagnostic runs also include nested LM-head, block forward/recompute, and block backward substages such as `lm_head_backward.dhidden`, `lm_head_backward.dweight`, `block_forward.attention.qkv`, `block_forward.attention.sdpa`, `block_forward.mlp_fc_gelu.fc_gelu`, `block_forward.mlp_proj.proj`, `block_backward.mlp_proj.dweight_bias`, `block_backward.mlp_proj.dinput`, `block_backward.ln1_residual.dinput_add`, `block_backward.ln2_residual.dinput_add`, `block_backward.attn_sdpa.to_qkv`, `block_backward.qkv.dweight_bias`, and `block_backward.qkv.dinput` so the next kernel bottleneck is visible without Nsight. The default cap is 20000 events; set `NFN_NATIVE_GPT_STAGE_TIMING_MAX_EVENTS=N` for longer profiling runs that would otherwise report dropped events.

The default GPT transformer-LM tape uses BF16 MLP activation storage on the workstation shape. It stores all 12 trained blocks' `ln2_out`, MLP preactivation, and GELU activation tensors into a BF16 arena during forward, consumes those BF16 tensors directly for MLP dWeight and fused MLP-projection dInput+dGELU, and elides MLP fc/GELU recompute. Forward storage always uses `nfn_native_tile_linear_bf16_input_weight_bf16_gelu_bf16_float32` after `ln2_out` has already been packed for backward, so the FC+bias+GELU TK path reuses the prepacked BF16 input instead of converting the same LN2 output twice. The stored-MLP LayerNorm stats sidecar uses the combined float arena on the default startup path instead of a second standalone allocation; training JSON reports `stored_mlp_layer_norm_stats_standalone_cuda_malloc_count`, which should be `0` when `NFN_NATIVE_GPT_FLOAT_STATS_ARENA` is left enabled. The older `NFN_NATIVE_GPT_REUSE_PACKED_LN2_FC_GELU=0` fallback has been retired. Set `NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS=0` to return to the lower-memory scratch-recompute path, or set `NFN_NATIVE_GPT_STORE_MLP_BLOCKS=N` to tune the saved-block cap; the GPT-2-prefixed names remain fallbacks. Training JSON reports `mlp_activation_storage_strategy`, `reuse_packed_ln2_fc_gelu_enabled`, `stored_mlp_forward_strategy`, `stored_mlp_activation_blocks`, `stored_mlp_activation_elements`, `stored_mlp_activation_bytes`, `stored_mlp_layer_norm_stats_standalone_cuda_malloc_count`, `stored_mlp_activation_store_kernel_launches`, `stored_mlp_activation_restore_kernel_launches`, `stored_mlp_activation_backward_consumer_strategy`, `block_backward_mlp_proj_dgelu_*`, `block_state_layout.backward_recompute_mlp_fc_gelu_elided`, and the resulting `block_state_layout.activation_tape_strategy`. Rebuild `libnfn_native_train_tile_ops.so` with `bash tools/build_native_train_tile_ops.sh` after updating, because the trainer now requires the raw BF16 store/direct-backward ABI symbols.

With BF16 block shadows enabled, earlier-block forward storage reports `stored_mlp_forward_strategy: "tk-sm120-fused-fc-bias-gelu-bf16-store-bf16-shadow-weight"` and fused projection backward reports `block_backward_mlp_proj_dgelu_strategy: "tk-sm120-fused-dinput-dgelu-bf16-store-bf16-shadow-weight-bf16-grad-handoff"`. The default BF16-gradient handoff keeps the fused dInput+dGELU result as BF16 bits for the following MLP FC backward and reports `block_backward_bf16_mlp_grad_handoff_enabled: true`; set `NFN_NATIVE_GPT_BF16_MLP_GRAD_HANDOFF=0` to compare against the older float-gradient handoff. `NFN_NATIVE_GPT_BF16_ATTENTION_DPREP_GRAD_OUT=1` is a default-off attention-backward bisection that packs float attention dO to BF16 only for the packed-attention dprep/backward wrapper while leaving attention projection dInput on the default float output path; it reports `attention_backward_bf16_dprep_grad_out_enabled` and `attention_backward_grad_out_dtype: "bf16-dprep-pack"` in runtime JSON.
Do not enable `NFN_NATIVE_GPT_BF16_ATTENTION_DPREP_GRAD_OUT` for normal
training. On the CUDA 13.3 RTX 5090 3-step/3-sample gate it reduced
`attention_backward_dprep_timing_us` to `0.843274x`, but the inserted
`block_backward.attn_sdpa.grad_out_bf16` pack made the end-to-end candidate
slower: train-loop wall time `1.005344x`, block backward `1.010632x`,
attention backward `1.044177x`, attention `to_qkv` `1.044330x`, and
`attention_backward_tk_timing_us` `1.000394x`.

GPT-2 block backward uses `nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_reuse_forward_from_merged_grad_float32` after each original or recomputed attention forward, reusing the TK bf16 Q/K/V/O/LSE workspace instead of repacking Q/K/V and launching a duplicate TK forward inside attention backward. JSON reports `attention_backward_strategy: "tk-sm120-bf16-reuse-forward-workspace-bridge"`, `attention_backward_reuses_forward_workspace: true`, `attention_backward_uses_saved_forward_workspace: false`, and `attention_backward_recompute_forward_elided_per_block: 1` when that optimized path runs; generic callers that cannot guarantee the matching preceding TK forward should use the older `nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32` ABI. The raw ABI also exposes `nfn_native_tile_scaled_dot_product_attention_store_tk_bf16_float32`, `nfn_native_tile_attention_tk_store_forward_workspace_bf16`, and `nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_saved_tk_bf16_from_merged_grad_float32` so native trainer loops can save TK BF16 Q/K/V/O plus float LSE into caller-owned buffers either directly during attention forward or by copying the process workspace afterward, then later run attention backward without recomputing the matching forward state. The dense GPT-2 trainer can use the direct forward-store saved-attention path with `NFN_NATIVE_GPT2_STORE_ATTENTION_ACTIVATIONS=1`; it stays disabled by default because the current 64x1024 TinyStories one-step probe reported only about 6.0k tok/s, with `block_backward.attn_sdpa.to_qkv` dominating at about 81.4s per optimizer step. Default training therefore continues to use the faster recompute plus process-workspace reuse path.

Non-dense-GPT `nfn train` commands now hand off to `nfn-native-train` first and fail in C++ with the registry status when no native trainer is implemented; they no longer fall through to TorchTrainer via an environment override. Dense GPT now defaults to the NeuralFn-owned Tile CUDA `--train-transformer-lm` loop for `gpt`, `gpt2`, and `gpt3`; NanoGPT dispatch adds `--template-name nanogpt` while keeping `model_family: "gpt"`, then exits with `template-geometry-native-trainer-missing` until the shared dense loop can use NanoGPT dimensions and dropout. Pass `--train-token-lm` explicitly to run the older tied token-embedding NanoGPT native loop over cached shards. The NanoGPT token-LM loop still uses `--eval-every-steps`, `--eval-batches`, and `--eval-batch-size` to emit validation loss from resolved validation token shards inside the compiled C++ loop, without sending validation payloads through graph-editor nodes or Torch. Direct legacy training scripts prefer the model-family C++ binary when present, using `NFN_NATIVE_<MODEL>_CLI`, `build/nfn_<model>_native_train`, or an installed `nfn_<model>_native_train`, then fall back to the generic native registry. Their pre-import guard normalizes wrapper-level `--native-cuda-*` aliases for dry-run/plan/check/smoke actions, library paths, backend/output/row-chunk selectors, checkpoint toggles, and native cadence/activation fields before execing C++, so direct script execution stays off the graph-backed Torch path even when users pass Python-wrapper flag names. LLaMA, JEPA, semantic/MoE, and DeepSeek harnesses still need real native C++ trainers before they can be default training commands.

GPT template selection is explicit on native dense GPT training commands: pass `--template-name NAME`, `--template NAME`, or `--preset NAME` to select a shipped GPT template, or pass `--graph-file PATH` / `--graph PATH` to select a custom graph JSON. Python wrappers canonicalize those aliases to `--template-name` and `--graph-file` before the compiled frontend runs, and the compiled path records the selected template or graph in JSON. Top-level `nfn train --base-model gpt` direct compiled-CLI handoff also adds the normal `--train-transformer-lm` action when no plan/check/smoke action is requested, so selector-bearing training commands stay explicit while debug commands remain metadata-only. `--base-model gpt2` and `--base-model gpt3` are aliases for that same native trainer; GPT3 changes the default context to 2048 through either `--base-model gpt3` or `--template-name gpt3` when no custom graph or explicit `--train-seq-len` is supplied, and defaults implicit batch size to 32 unless `--batch-size` is supplied. `gpt2`, `gpt2_megakernel`, and `gpt2_moa` use the implemented CUDA Tile transformer-LM trainer today; selecting `gpt2_moa` automatically resolves the native MoA activation mode. Other template shapes and custom graphs fail fast with missing-native-trainer JSON instead of falling back to Torch. The GPT-2 evo C++ preflight accepts the same template and custom graph selector aliases and reports the shipped template catalog before any graph-backed runtime can import; its JSON marks every selected graph as not yet native-runnable until the evo-layer CUDA Tile trainer is implemented.

Native GPT-2 checkpoint export stays on the raw Tile-CUDA ABI. Successful `--train-transformer-lm` runs pack all device float32 weight tensors into one contiguous bf16 payload with `nfn_native_tile_float32_to_bf16_bits_many`, copy that compact uint16 payload back to host once, and write the version-5 `.bin` plus `DONE_########` marker. The version-5 header stores public vocab 50,257 separately from padded tensor vocab 50,304, so tokenizer validation should use the public vocab while parameter-size checks use the padded vocab. Training JSON reports `checkpoint.payload_pack_strategy: "device-many-float32-to-bf16-bits-contiguous"`, `payload_pack_kernel: "nfn_native_tile_float32_to_bf16_bits_many"`, `payload_copy_strategy: "single-contiguous-device-payload-d2h"`, `payload_cpu_bf16_conversion: false`, `device_pack_kernel_launches`, `d2h_copy_count`, `d2h_bytes`, and `float32_d2h_bytes_elided`, so checkpoint export no longer materializes full float32 parameter tensors on CPU just to pack bf16 and no longer performs one D2H copy per parameter tensor.

The raw trainer Tile ABI avoids per-call full-vocab CE workspace allocation. Workspace-aware CE backward entrypoints still use caller-provided row-stat buffers, and the legacy non-workspace `nfn_native_tile_token_cross_entropy_backward_float32` / `nfn_native_tile_masked_token_cross_entropy_backward_float32` entrypoints now reuse a process-cached row-stat workspace for vocabularies wider than 1024. `nfn_native_tile_token_cross_entropy_workspace_allocation_count()` and `nfn_native_tile_token_cross_entropy_workspace_row_capacity()` expose that cache for native smoke tests and profiling.

For dense GPT Tile CUDA preflight/training, use `--backend tile-cuda` on `nfn_gpt_native_train`, or `--kernel-backend tile-cuda` from the Python wrapper. The compiled C++ default is `tile-cuda` plus `--train-transformer-lm`, `--template-name gpt`, and `model_family: "gpt"`; `gpt2` and `gpt3` are selector aliases, and `gpt3` supplies a 2048-token default context when selected by base model or template name, as long as no custom graph or explicit sequence length is supplied; implicit GPT3 batch size defaults to 32 unless `--batch-size` is explicit. Plan and runtime JSON report `template_name` plus `resolved_native_template_name` so the public `gpt` alias is separate from the current dense GPT implementation template. Pass `--no-train-transformer-lm` only for plan/check/debug commands that should not start that trainer. Runner names are strict too: use `compiled-cli`, not the old `cli` alias. No-data preflight actions (`--check-tile-ops`, `--smoke-tile-ops`, `--smoke-optimizer-step`, `--smoke-lm-step`, `--smoke-attention-step`, `--smoke-mlp-step`, `--smoke-norm-residual-step`, and `--smoke-transformer-block-step`) run before token-shard resolution, so they can validate the raw Tile ABI or synthetic CUDA slices without a cached dataset; their JSON reports `token_shards_resolved: false` when no shards were opened. `--smoke-tile-ops --tile-ops-lib PATH` loads the raw trainer Tile ops library, loads CUDA runtime, launches `nfn_native_tile_fill_float32` on a tiny device buffer, copies it back, and reports JSON without Python, Torch, or graph-editor payloads. `--smoke-lm-step` reports summed two-row CE loss for the synthetic GPT slice and verifies post-AdamW weight updates on target rows only; non-target tied-embedding gradients remain part of the gradient check, but their near-zero first-step AdamW sign can amplify CUDA 13.3 floating-point noise and is covered by the standalone optimizer smoke. The full `--train-transformer-lm --tile-ops-lib PATH` path runs a full-vocab real-dim 12-layer multi-step dense GPT loop over cached shards with periodic validation losses in `validation.losses`, using token/position embeddings, transformer blocks, final norm, a row-chunked tied LM head/CE workspace, transformer backward, embedding backward, device-side global norm gradient clipping, and AdamW parameter updates without Python/Torch. Validation uses its own C++ shard sampler and active forward batch size from `--eval-batch-size` / `eval_batch_size` instead of forcing the full training microbatch through the eval pass; the value must be between 1 and the training `--batch-size` because the fixed activation arena is still sized to the training shape. The native token embedding/LM-head tensor is padded to 50,304 rows while tokenizer-visible vocab stays 50,257. Plan JSON now reports the accepted sample/generate/checkpoint cadence in `schedule.sample_every_steps`, `schedule.generate_tokens`, and `schedule.checkpoint_every_steps`; runtime JSON mirrors those fields and reports `train_time_sampling_enabled: false`, `periodic_checkpoint_enabled: false`, and `final_checkpoint_export_enabled` so timing-only benchmark runs can prove `--native-cuda-sample-every 0`, `--native-cuda-checkpoint-every 0`, and `--no-checkpoint` disabled non-training work. The training JSON includes `cuda_runtime_preflight`; driver version `0` or a loaded runtime newer than the driver fails before allocation, so SM120 timing failures point at GPU access/runtime compatibility instead of a later `cudaMalloc` error. `--checkpoint-metadata-smoke --output-dir PATH` writes a sparse version-5 bf16 native GPT checkpoint-format file plus `DONE_########` marker for the requested `--num-layers` target shape so the Torch-free native checkpoint/inference metadata path can validate NeuralFn-owned artifacts without running CUDA. Successful `--train-transformer-lm` runs write a final 12-layer trained-weight native checkpoint plus `DONE_########` marker. Pass `--cuda-runtime-lib PATH` or set `NFN_CUDA_RUNTIME_LIB` when libcudart is not discoverable.

NanoGPT's native `--smoke-lm-step` accepts loss, gradient, and weight-update
absolute error up to `1e-5`. CUDA 13.3 on the RTX 5090 has shown stable tied
embedding gradient drift around `6.1e-6`; this is treated as a passing numeric
smoke, while CUDA load/runtime failures still return nonzero JSON errors.
NanoGPT's fused-QKV attention smoke accepts Q/K/V, attention/output, and
input-gradient drift up to `1e-4` on the same CUDA 13.3 path. Its
output-weight-gradient check uses the same `1e-4` envelope; QKV
weight-gradient and weight-update checks remain at `1e-5`.
NanoGPT's MLP smoke also uses `1e-4` for output, input-gradient, and
FC-gradient drift on CUDA 13.3, with projection-gradient and weight-update
checks still at `1e-5`.
NanoGPT's standalone attention smoke uses `1e-4` for attention activation,
output, V-weight-gradient, and output-weight-gradient drift; zero-gradient and
weight-update checks remain tighter.

When `--json-out`, `--profile-json`, or `--stage-profile-json` redirects native C++ output to a file, failed `--check-tile-ops` and `--train-transformer-lm` runs also mirror a one-line failure summary to stderr. The JSON sidecar remains the machine-readable source of truth, while direct shell runs and benchmark wrappers show missing Tile symbols, CUDA-driver access errors, and runtime preflight failures without requiring a manual sidecar open.

For wrapper-level command inspection, `python cli/scripts/train_gpt.py --tinystories --native-cuda-dry-run --native-cuda-print-command` stays metadata-only on the default `compiled-cli` runner. It prints the compiled C++ command without importing `server.dataset_manager`, NumPy, tiktoken, or Torch and without materializing raw-text token shards, and the default Tile-CUDA command no longer includes the external `--target train_gpt2cu` bridge argument. `train_gpt.py` preserves its own command name for help/errors; `train_gpt2.py` remains a compatibility entrypoint. The compiled Tile-CUDA frontend also treats `--print-command` as a no-data/no-CUDA action: it prints the exact `nfn_gpt_native_train ...` invocation and exits before token-shard resolution, CUDA runtime loading, or driver preflight. Dense GPT `--dry-run` / `--print-plan` JSON now reports `model_family: "gpt"` by default and the implemented trainer as `native-transformer-lm-ready` with a ready `training_step_plan`; `remaining_validation` now points at closing the measured SM120 throughput gap and names `tools/bench_native_gpt_sm120_parity.sh` as the same-script comparison gate.

Native dense GPT training handoff accepts `--template-name NAME` / `--preset NAME` and `--graph-file PATH` / `--graph PATH` from the Python wrapper, SDK compiled-CLI config, and compiled C++ frontend. The default `--template-name gpt` is a public alias for the implemented dense GPT native topology and reports `resolved_native_template_name: "gpt2"` until the internal template name is retired. Every name in `neuralfn.config.SHIPPED_GPT_TEMPLATE_PRESETS` can be passed through this native selection path, including GPT/NanoGPT megakernel presets and aliases such as `mixllama`; the compiled C++ plan JSON exposes the same `shipped_template_catalog`, `shipped_template_catalog_count`, and `template_known` fields so the no-Python path is visibly in sync with the SDK catalog. The implemented trainer currently runs `gpt`, `gpt2`, `gpt2_megakernel`, and `gpt2_moa`; `gpt2_moa` maps to `--native-cuda-activation moa` automatically. Structurally different shipped template names report `template-geometry-native-trainer-missing` or `template-native-trainer-missing`, arbitrary custom graph files report `custom-graph-native-trainer-missing`, typoed/unshipped template names report `unknown-template`, and missing custom graph paths report `custom-graph-file-missing` with `graph_file_exists: false` plus `graph_file_size_bytes: -1`. Existing custom graph files that carry native-compatible GPT `template_spec` metadata report `native-transformer-lm` and use that metadata for default sequence length and layer count while still running the fixed dense GPT C++ Tile loop; migration work stays explicit and no real batches pass through graph-editor nodes.

The compiled dense GPT transformer-LM trainer keeps cached token and target batches compact as uint16 during host-to-device upload, samples them directly from the C++ shard reader into one pinned host arena, enqueues one contiguous H2D `cudaMemcpyAsync` for tokens plus targets, then consumes the uint16 token ids directly in token embedding, BF16 public-vocab CE loss, CE backward, and token-embedding weight backward kernels. The default direct-u16 path no longer reserves the unused int64 token/target device subarena, cutting the token device arena to only the uint16 copy buffer; JSON reports `token_i64_device_arena_elided` and `token_i64_device_arena_bytes_elided`. Training JSON reports `token_id_direct_u16_enabled: true`, `token_id_upload_strategy: "uint16-pinned-async-h2d-direct-kernel-consumption"`, `token_id_host_staging: "pinned"`, `token_batch_staging_strategy: "direct-sampler-to-pinned-arena"`, `token_batch_vector_materialization: false`, `token_id_h2d_copy: "cudaMemcpyAsync-contiguous-arena"`, `token_id_h2d_copy_calls_per_microbatch: 1`, `token_id_widen_strategy: "elided-direct-u16-kernels"`, `token_id_widen_kernel_launches_per_microbatch: 0`, and `token_id_host_validation: false`; cached shards are trusted on this native path instead of being re-expanded or range-validated on CPU for every batch. Set `NFN_NATIVE_GPT_DIRECT_U16_TOKENS=0` or `NFN_NATIVE_GPT2_DIRECT_U16_TOKENS=0` only for paired benchmarks against the older single-kernel device-widening path.

GPT transformer-LM startup also initializes the tied token embedding/LM-head
weight on device through `nfn_native_tile_init_gpt2_token_weight_fast_float32`
instead of constructing a 154 MB host float matrix and copying it to the GPU.
The default vector4 Tile initializer writes the deterministic power-of-two value
pattern over the full padded vocabulary table and the BF16 shadow in one startup
route; set `NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT=0` only for paired
comparison against the previous fast int32-index Tile path, set
`NFN_NATIVE_GPT_TOKEN_WEIGHT_FAST_INT32_INIT=0` after disabling vector4 only for
paired comparison against the older int64-index Tile path, and set
`NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1` only for paired comparison against
the not-promoted threaded CUDA initializer. Training JSON reports
`token_weight_init_strategy: "device-vector4-power2-deterministic"` or the fused
BF16-shadow variant, `token_weight_threaded_init_enabled`,
`token_weight_vector4_init_enabled`, `token_weight_fast_int32_init_enabled`, and
`token_weight_host_materialization: false`.

GPT-2 transformer-LM startup has a single owner for per-block buffers. The block-vector visitors allocate parameters, gradients, AdamW state, and scratch-tape activations for every transformer block, including block 0; the global startup list now covers only token/position/final-norm/shared workspace buffers. Training JSON reports `block0_duplicate_allocation_elided`, `block0_duplicate_activation_allocation_elided`, `block0_duplicate_parameter_initialization_elided`, and `block0_duplicate_adamw_state_zero_elided` under `block_state_layout`.

Float CUDA buffers in the compiled GPT-2 transformer-LM trainer are suballocated from one aligned device arena instead of issuing one `cudaMalloc` per parameter, gradient, AdamW moment, activation, and workspace buffer. BF16 activation/scratch buffers are also suballocated from one uint16 device arena by default, covering stored MLP activations, residual1 caches, packed attention stores, LM-head BF16 logits, MLP BF16 scratch, projection-output BF16 scratch, packed-QKV BF16 scratch, and block BF16 weight shadows. Set `NFN_NATIVE_GPT_COMBINED_BF16_ARENA=0` or `NFN_NATIVE_GPT2_COMBINED_BF16_ARENA=0` to reproduce the older per-buffer BF16 `cudaMalloc` path in paired benchmarks. Startup now zeroes only AdamW first/second moment state as coalesced contiguous ranges with `cudaMemsetAsync` by default, while nonzero weights are still written by device initializers. Per-step accumulation gradients also default to coalesced contiguous-range `cudaMemsetAsync` zeroing; set `NFN_NATIVE_GPT_CUDA_MEMSET_GRAD_ZERO=0` to compare against the older descriptor-driven Tile fill-many path. Set `NFN_NATIVE_GPT_CUDA_MEMSET_ZERO=0` to compare against the older Tile fill startup-zero path, `NFN_NATIVE_GPT_ZERO_ADAMW_STATE_RANGES=0` to force the older descriptor-driven AdamW state fills, or `NFN_NATIVE_GPT_ZERO_ADAMW_STATE_ONLY=0` to force the older full-arena zero for bisection. Nonzero constant parameter initialization is fused through `nfn_native_tile_fill_many_values_float32`, so position weights, final norm, residual scale, and all block constant weights initialize with one descriptor-driven Tile launch instead of 75 per-buffer fills at the default 12-layer shape. The AdamW, gradient-clip, gradient-zero, and parameter-fill descriptor tables are also suballocated from one device descriptor arena and uploaded from one host-packed descriptor arena, replacing ten small startup `cudaMalloc` calls and ten descriptor H2D copies with one allocation and one copy. Training JSON reports `float_allocation_strategy: "single-arena"`, `uint16_allocation_strategy: "single-arena"` or `"per-buffer-cudaMalloc"`, `uint16_allocation_cuda_malloc_count`, `uint16_allocation_request_count`, `uint16_arena_requested_elements`, `uint16_arena_allocated_elements`, `uint16_arena_cuda_malloc_count`, `uint16_arena_suballocation_count`, `projection_bf16_scratch_elements`, `projection_bf16_scratch_bytes`, `float_arena_zero_init_strategy: "adamw-state-contiguous-range-cuda-memset"`, `"adamw-state-contiguous-range-fill"`, `"adamw-state-fill-many"`, `"single-arena-cuda-memset"`, or `"single-arena-fill"`, `startup_cuda_memset_zero_enabled`, `startup_cuda_memset_zero_available`, `float_arena_zero_fill_count`, `adamw_state_zero_fill_count`, `startup_cuda_memset_zero_fill_count`, `startup_tile_zero_fill_count`, `adamw_state_zero_range_count`, `adamw_state_zero_range_elements`, `gradient_cuda_memset_zero_enabled`, `gradient_cuda_memset_zero_available`, `gradient_zero_range_count`, `gradient_zero_range_elements`, `gradient_zero_cuda_memset_count`, `gradient_zero_tile_fill_count`, `startup_per_buffer_zero_fill_elided`, `startup_per_buffer_zero_fill_launches_elided`, `parameter_initialization_strategy: "fused-multi-buffer-fill-values"`, `parameter_initialization_kernel_launches_per_startup`, `parameter_initialization_per_buffer_launches_elided`, `descriptor_allocation_strategy: "single-device-arena"`, `descriptor_upload_strategy: "single-host-packed-arena-copy"`, `descriptor_arena_cuda_malloc_count`, `descriptor_arena_suballocation_count`, `descriptor_arena_copy_count`, `descriptor_arena_copy_calls_elided`, `descriptor_cuda_mallocs_elided`, `float_allocation_cuda_malloc_count`, `float_allocation_request_count`, `float_arena_requested_elements`, and `float_arena_allocated_elements`. Arena diagnostics also include `float_arena_request_stats` and `uint16_arena_request_stats`, with `top_requests` for individual suballocations plus `top_families` / `family_count` / `top_family_bytes` to aggregate repeated per-layer names such as `block.*.persistent_output`; main transformer-LM global float buffers are named individually, for example `mlp.fc.grad_out`, `attention.grad_out`, and `lm_head.float_logits`. The `timing.setup_timing` array also breaks startup into host-side phases such as `setup.float_arena_materialize`, `setup.uint16_arena_materialize`, `setup.stored_mlp_activation_arena`, `setup.projection_bf16_scratch`, `setup.zero_init`, and `setup.block_weight_bf16_initial_refresh`, so startup regressions can be traced without enabling CUDA event stage timing. When `NFN_NATIVE_GPT_STAGE_TIMING=1` is enabled, the dense GPT stage profiler also reports `block_backward.mlp_proj.grad_out_bf16`, separating the one-time per-block projection-gradient BF16 pack from the surrounding MLP projection dWeight and dInput kernels.

The saved packed-attention LN1 BF16 tape is part of that default uint16 arena too; default JSON therefore counts it as a `uint16_arena_suballocation_count` entry rather than a separate BF16 `cudaMalloc`. The stored MLP LayerNorm stats and saved packed-attention LN1 stats sidecars are likewise suballocated from the single float arena instead of issuing separate float `cudaMalloc` calls; set `NFN_NATIVE_GPT_FLOAT_STATS_ARENA=0` or `NFN_NATIVE_GPT2_FLOAT_STATS_ARENA=0` only for paired startup comparisons against the older sidecar allocation path.

When stored BF16 MLP activations cover every transformer block, the compiled GPT trainer no longer reserves validation-only float MLP scratch (`fc_out` and `act`) in the startup float arena. Those two hidden-size buffers are allocated lazily only if validation actually runs, reducing the default 12-layer `64 x 1024` float arena by `402,653,184` elements. Set `NFN_NATIVE_GPT_LAZY_VALIDATION_MLP_FLOAT_SCRATCH=0` or `NFN_NATIVE_GPT2_LAZY_VALIDATION_MLP_FLOAT_SCRATCH=0` only for paired startup comparisons. Runtime JSON reports `lazy_validation_mlp_float_scratch_enabled`, `lazy_validation_mlp_float_scratch_elements`, `lazy_validation_mlp_float_scratch_bytes`, and `lazy_validation_mlp_float_scratch_cuda_malloc_count`.

With BF16-primary block weights enabled, startup now splits nonzero constant initialization into one float32 fill-many launch for position/final-norm/scalar values and one BF16 fill-many launch for transformer block QKV/projection/MLP weights. This avoids the old initial block-weight `float32_to_bf16_bits_many` refresh while preserving the `NFN_NATIVE_GPT_DIRECT_BF16_BLOCK_WEIGHT_INIT=0` bisection path. On the dedicated RTX 5090, `tools/paired_kernel_speed.py` measured setup wall time at 512.9 ms for direct BF16 init versus 534.8 ms for the forced old path over 3 measured 5-step samples; total 5-step runtime was noise-equivalent because the train loop dominates.

Token upload buffers use combined arenas too: widened int64 tokens/targets and compact uint16 H2D staging share one aligned device token arena, while compact source staging uses one pinned uint16 host arena. Training JSON reports `token_buffer_allocation_strategy: "combined-arenas"`, `token_device_allocation_strategy: "single-device-arena"`, `token_device_arena_cuda_malloc_count`, `token_device_arena_suballocation_count`, `token_device_cuda_mallocs_elided`, and the token arena element counts.

The compiled GPT-2 transformer-LM trainer does not sample training loss in the hot path. Optimizer steps run the forward activations needed for backward, CE gradient generation, gradient clipping, and AdamW only; validation cadence computes validation loss from validation shards according to `--eval-every-steps` without also measuring train loss. Set `--eval-batch-size N` to reduce validation rows per eval batch; runtime JSON reports the resolved value under `validation.eval_batch_size` and each record token count under `validation.losses[].tokens`. The output fields `train_loss_sparse: false`, `train_loss_sampling: "disabled"`, `train_loss_on_validation_steps: false`, `train_loss_eval_count`, and `train_loss_last_step` describe that contract.

Persistent transformer block outputs in the compiled GPT trainer are written directly from each non-final block's MLP residual-add stage into the per-layer backward-recompute buffer. That removes the previous post-block `nfn_native_tile_copy_float32` preservation launch while keeping the scratch-recompute tape layout. The final block output copy is still elided because final LayerNorm consumes that tensor before backward recomputation starts; the default 12-layer run reports `persistent_block_outputs: 11`, `persistent_block_output_write_strategy: "direct-residual2-output"`, `persistent_block_output_copy_elided_count`, and `final_block_output_copy_elided: true`.

Validation forwards in the compiled GPT-2 trainer do not preserve block outputs into the persistent training-backward buffers. Training forwards still save earlier block outputs for backward recomputation, while validation streams through the scratch tape because no backward pass follows; JSON reports `validation_persistent_block_outputs: 0` and `validation_block_output_copies_elided: true`.

The scratch-recompute backward pass also reuses the final block activations that remain in the scratch tape after the initial forward pass. Earlier blocks are recomputed from persistent block outputs, but the final block is not recomputed before backward; the default 12-layer run now reports `backward_recompute_blocks: 11`, `final_block_backward_recompute_elided: true`, `backward_recompute_mlp_fc_gelu_elided: true`, and `activation_tape_strategy: "scratch-recompute-bf16-stored-packed-attention-and-mlp-direct-backward"`. Recomputed earlier blocks still stop before the MLP projection output and final residual output because backward does not consume them; `block_state_layout` reports `backward_recompute_mlp_projection_elided: true` and `backward_recompute_final_residual_elided: true`. `NFN_NATIVE_GPT_FULL_ACTIVATION_TAPE=1` / `NFN_NATIVE_GPT2_FULL_ACTIVATION_TAPE=1` is available only as a paired-benchmark diagnostic: it allocates one forward activation tape per block and reports `full_activation_tape_enabled: true`, `activation_tape_count: num_layers`, `backward_recompute_blocks: 0`, and a `full-forward-tape...` strategy, but the RTX 5090 check measured it far slower than the default scratch-recompute path, so do not enable it for normal training. Set `NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS=0` to disable the default BF16 residual1 cache for recomputed earlier blocks; set `NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS=0` to use pure `"scratch-recompute"` with `backward_recompute_mlp_fc_gelu_elided: false`.

The MLP projection backward path writes projection dInput directly into the MLP fc gradient buffer and applies GELU backward in-place with `nfn_native_tile_gelu_backward_inplace_float32`. The full trainer therefore no longer allocates or writes a separate `grad_act` hidden-size scratch buffer; `block_state_layout` reports `mlp_proj_backward_gelu_inplace: true` and `mlp_proj_backward_grad_act_scratch_allocated: false`.

Transformer block backward residual gradients are fused through `nfn_native_tile_scaled_residual_add_float32` in the compiled GPT-2 trainer. The trainer no longer zeroes a buffer and launches two gradient accumulates for each residual-gradient pair; `block_state_layout.residual_backward_fused` reports this path.

Gradient clipping in the compiled GPT-2 trainer now feeds the device clip scalar directly into `nfn_native_tile_adamw_step_with_device_scale_float32`. That removes the separate per-gradient-buffer scale pass before AdamW; `block_state_layout.adamw_device_clip_scale_fused` reports this path.

Gradient clipping's sum-of-squares phase also reuses the device-resident optimizer descriptor table through `nfn_native_tile_sumsq_partials_many_float32`. The default 12-layer trainer computes partial sums for all 148 accumulation buffers with one multi-buffer kernel launch per optimizer step instead of one launch per buffer, then keeps the existing device clip-scale reduction and fused AdamW update. JSON reports `gradient_clip_strategy: "fused-multi-buffer-sumsq-device-scale"`, `gradient_sumsq_kernel_launches_per_optimizer_step`, `gradient_sumsq_per_buffer_launches_elided`, and `block_state_layout.gradient_clip_loop: false`.

AdamW updates in the compiled GPT-2 trainer now run through descriptor-driven multi-buffer Tile launches over device-resident parameter descriptors and the device clip scalar. The default 12-layer path splits optimizer updates into float32 descriptors for token/position/norm/bias tensors and BF16 descriptors for block projection weights through `nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_float32`; JSON reports `adamw_update_strategy`, `adamw_descriptor_count`, `adamw_step_kernel_launches_per_optimizer_step`, and `adamw_per_buffer_step_launches_elided`. The float32 descriptor update also refreshes the tied token-weight BF16 LM-head shadow in the same AdamW launch by default, so the BF16-primary block-weight path no longer needs a separate post-AdamW token-shadow pack kernel. Set `NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_ADAMW_REFRESH=0` only for paired bisection against the older two-launch token-shadow path. Runtime JSON reports `token_weight_bf16_adamw_refresh_fusion_enabled`, `token_weight_bf16_fused_adamw_refresh_count`, and the fused strategy names `split-float32-token-shadow-and-bf16-param-multi-buffer-device-scale` / `elided-block-bf16-primary-token-shadow-fused-adamw` when the default path executes. After rebuilding with CUDA 13.3.33 on WSL, the dedicated RTX 5090 one-step same-script benchmark measured candidate/default at `0.998488x` train-loop wall time and `1.001519x` tokens/sec versus `NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_ADAMW_REFRESH=0`. The raw ABI also exports `nfn_native_tile_adamw_step_many_with_device_scale_bf16_shadow_float32` for fused AdamW plus BF16 shadow refresh profiling; set `NFN_NATIVE_GPT_FUSE_ADAMW_BF16_SHADOW_REFRESH=1` only after forcing `NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS=0`, because the block-shadow refresh path is bypassed by the BF16-primary default.

Token, position, block Linear weight, LayerNorm affine, and Linear bias gradients in the compiled GPT-2 trainer accumulate directly into optimizer-step accumulation buffers. The tied LM-head CE backward scale includes the microbatch accumulation factor, LM-head dWeight chunks and token-embedding backward write into `accum_grad_token_weight`, and the old full-vocab token-gradient scratch buffer is not allocated. Position embedding backward uses the accumulate-position ABI, so `grad_position_weight` is not allocated or copied after each microbatch. Each transformer block writes qkv, attention-output, MLP fc, and MLP projection dWeight plus bias through the fused weight-bias ABI where possible, and still writes LayerNorm affine gradients straight into block accumulation buffers, so the real 12-layer loop no longer allocates per-block scratch gradient buffers or runs a per-microbatch copy loop. BGRADB bias gradients use the scratch accumulation path by default; set `NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1` only for direct first-write diagnostics. Accumulation buffers are zeroed once per optimizer step through coalesced contiguous-range `cudaMemsetAsync` by default, falling back to `nfn_native_tile_fill_many_float32` over the optimizer descriptor table when CUDA memset is unavailable or `NFN_NATIVE_GPT_CUDA_MEMSET_GRAD_ZERO=0` is set. JSON reports `token_gradient_accumulation_strategy: "direct-device-accumulation-buffer"`, `position_gradient_accumulation_strategy: "direct-device-accumulation-buffer"`, `block_linear_weight_gradient_accumulation_strategy: "direct-device-accumulation-buffer"`, `layer_norm_affine_gradient_accumulation_strategy: "direct-device-accumulation-buffer"`, `linear_bias_gradient_accumulation_strategy: "direct-device-accumulation-buffer"`, `linear_bias_gradient_first_write_bgrad_direct_enabled`, `gradient_zero_strategy: "fused-multi-buffer-accumulation-zero"`, `gradient_cuda_memset_zero_enabled`, `gradient_zero_range_count`, `gradient_zero_cuda_memset_count`, `gradient_zero_tile_fill_count`, `gradient_zero_kernel_launches_per_optimizer_step`, `gradient_zero_per_buffer_launches_elided`, `block_state_layout.per_block_gradient_buffers: 0`, `block_state_layout.per_block_direct_accum_gradient_buffers: 12`, `block_state_layout.gradient_accumulation_loop: false`, `block_state_layout.gradient_accumulation_copy_loop_elided: true`, `block_state_layout.gradient_zero_loop: false`, and `gradient_zeroed_buffer_count: 0`.

Large-row Linear bias-gradient and LayerNorm affine-gradient reductions in the Tile CUDA library use chunked Tile atomic reductions instead of cuBLAS SGEMV on the fallback route. Linear bias gradients use 512-row chunks, while LayerNorm affine gradients default to 256-row chunks after paired RTX 5090 timing showed better native train-loop throughput than the old 512-row setting. Set `NFN_TILE_CUDA_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=N` (or `NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE` / `NFN_NATIVE_GPT2_LAYERNORM_AFFINE_ROW_CHUNK_SIZE`) to compare other LayerNorm row chunk sizes in the paired benchmark harness. This keeps the default GPT `batch=64`, `seq=1024` backward pass on the faster local reduction route while providing enough row-chunk parallelism for the dedicated RTX 5090 compute path; small reductions can still use cuBLAS where selected.

The full GPT-2 Tile CUDA trainer also fuses MLP bias+GELU with a BF16 activation scratch for the projection input. `nfn_native_tile_gelu_add_bias_bf16_act_float32` writes float preactivation, float GELU, and BF16 GELU bits from a CUDA Tile kernel in one launch, and `nfn_native_tile_gelu_backward_inplace_bf16_bits_float32` consumes saved BF16 preactivation bits through a CUDA Tile backward kernel instead of the old scalar CUDA path. `nfn_native_tile_linear_bf16_input_bits_float32` feeds BF16 GELU bits directly to the MLP projection GEMM. Training JSON reports `mlp_proj_forward_activation_strategy`, `mlp_forward_act_bf16_elements`, and `mlp_forward_act_bf16_bytes` so probes can confirm the direct BF16 activation path is active.

LayerNorm affine-gradient backward now also has an accumulate ABI and uses a chunked parallel atomic reduction for large row counts. This avoids the previous single-block loop over every token row in GPT-sized batches and removes the scratch-to-accumulation copy; `block_state_layout.layer_norm_backward_affine_strategy` reports `auto-chunked-atomic-accumulate`. The LayerNorm affine row chunk defaults to 256 rows and can be overridden with `NFN_TILE_CUDA_LAYERNORM_AFFINE_ROW_CHUNK_SIZE`.

For SDK-level native dispatch, build the unified binding with `bash tools/build_native_train_binding.sh`. `neuralfn.native_train` exports `NativeTrainRunConfig`, `build_native_train_run_config()`, `native_train_runner_status()`, `native_train_model_registry()`, `resolve_native_train_cli()`, and `run_native_train()`, so Python callers can hand off through the C++ extension without importing Torch. Dense GPT-family SDK configs (`gpt`, `gpt2`, `gpt3`, `nanogpt`) resolve directly to `nfn_gpt_native_train --model-family ...` when `NFN_NATIVE_GPT_CLI` is set or `build/nfn_gpt_native_train` exists, avoiding the extra generic `nfn_native_train` dispatcher process on the normal workstation path; set `NFN_NATIVE_TRAIN_CLI` or pass `native_train_cli=` to force the unified frontend. Binding discovery invalidates Python import caches and rejects stale `neuralfn._native_train` extensions that expose `run_train` without the required command resolver, then probes the package search path for a complete rebuilt extension before falling back to the compiled CLI. The generic native-train subprocess fallback now matches GPT-specific launchers by setting `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and `CUDA_MODULE_LOADING=LAZY` only when those variables are unset; override `NativeTrainRunConfig.cuda_visible_devices` or the environment when targeting a different CUDA device. For dense GPT direct compiled-CLI handoff, build `neuralfn._native_gpt` with `bash tools/build_native_gpt_binding.sh` and use `build_native_gpt_compiled_cli_run_config()` to create a config from a dataset alias/path without inspecting token shards in Python; the older `build_native_gpt2_compiled_cli_run_config()` compatibility helper now canonicalizes dense GPT selectors to `model_family="gpt"` and can still use the `_native_gpt2` compatibility binding.

`bash tools/build_native_train_tile_ops.sh` builds `libnfn_native_train_tile_ops.so`, a raw C ABI over the CUDA Tile kernels in `neuralfn/csrc/tile_cuda/kernels.cu`. This trainer-facing library exposes single-buffer and multi-buffer fill/zeroing, single-buffer and multi-buffer sumsq partials, single-buffer and multi-buffer AdamW, gradient accumulation, deterministic GPT-2 token-weight initialization, device float32-to-bf16 checkpoint payload packing, device-side global-norm clip scale finalization, device-scalar gradient scaling, fixed scaling, reductions, linear, forced-BF16 linear, linear input/forced-BF16 input/fused dGELU input/weight/weight-accumulate/forced-BF16 weight-accumulate/forced-BF16 weight+bias-accumulate/bias/bias-accumulate backward, scaled residual add, fused projection bias+residual add, BF16-linear projection bias+residual add, fused QKV split/merge, fused GPT-2 QKV split-to-heads, fused GPT-2 QKV bias+split-to-heads, fused GPT-2 heads-to-QKV gradient merge, fused TK bf16 attention-gradient heads-to-QKV bridge, saved TK BF16 attention workspace copy/backward, reshape-heads/merge-heads, GELU forward, fused bias+GELU forward, GELU backward, token embedding forward/weight backward, absolute-position embedding forward/backward/backward-accumulate, RMSNorm, RMSNorm input backward, LayerNorm, LayerNorm input plus fused input/residual-add, affine, and affine-accumulate backward, softmax, float token and masked token cross-entropy partials, BF16-bits token cross-entropy partials, strided float/BF16-bits token cross-entropy partials for public-vocab softmax over padded LM-head rows, token and masked token cross-entropy logits backward, strided in-place token cross-entropy logits backward that zeroes padded dlogit columns, and scaled dot-product attention forward/backward without including `torch/extension.h` or linking the PyTorch extension binding; native model trainers should link this library instead of calling graph-backed Tile wrappers. The trainer build defines `NFN_TILE_CUDA_USE_CUBLAS_LINEAR=1` and links `libcublas` so the same native linear ABI routes linear forward, dInput, dWeight, and accumulate-dWeight through GPU GEMM; forced-BF16 block weight+bias accumulation uses cuBLASLt `BGRADB` when supported and falls back to separate dWeight plus Tile bias-reduction launchers. The generic Tile extension build keeps the pure Tile fallback path. CE logits backward uses row-wise kernels for vocabularies up to 1024 and chunked row-wise kernels with reusable row-stat workspace for full GPT-class vocabularies, avoiding the previous elementwise large-vocab fallback. Linear weight, accumulate-weight, bias, and accumulate-bias backward keep the row-chunked tiled atomic fallback for builds or shapes that do not use the trainer cuBLAS path.

Dense GPT training now uses the LM-head classifier row-loss reduction ABI by
default when the rebuilt Tile ops library exports it. The classifier kernel
writes one CE loss per row while converting the same BF16 logit chunk in-place
to dlogits; the trainer then reduces those row losses on device with
`nfn_native_tile_sum_partials_float32` before accumulating the scalar loss. This
avoids doing a scalar `atomicAdd` from every classifier row block. Runtime JSON
reports `lm_head_ce_row_loss_reduction_requested`,
`lm_head_ce_row_loss_reduction_available`,
`lm_head_ce_row_loss_reduction_enabled`, and
`lm_head_ce_loss_backward_strategy`. Set
`NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_REDUCTION=0` or
`NFN_NATIVE_GPT2_LM_HEAD_ROW_LOSS_REDUCTION=0` only for paired bisection against
the older fused-loss atomic route. The BF16/u16 row-loss kernel reads the target
logit before it overwrites the row with dlogits, so the in-place loss path avoids
an extra post-loss CTA barrier while preserving the same row-loss and dlogit
outputs.
There is also a default-off diagnostic tail,
`NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE=1`, that replaces the generic
row-loss `sum_partials` plus scalar `gradient_accumulate` launches with one
`nfn_native_tile_sum_accumulate_float32` launch per row chunk. Keep it disabled
for normal training: the CUDA 13.3.33 RTX 5090 paired benchmark regressed to
`1.008595x` train-loop wall time and `1.015517x` LM-head CE time versus the
current row-loss reduction default.

For native GEMM profiling, set `NFN_NATIVE_LINEAR_SHAPE_STATS=1`, `NFN_TILE_CUDA_LINEAR_SHAPE_STATS=1`, `NFN_NATIVE_GPT_LINEAR_SHAPE_STATS=1`, or `NFN_NATIVE_GPT2_LINEAR_SHAPE_STATS=1` before running `nfn_gpt_native_train`. The Tile ops ABI records successful linear dispatch buckets and the GPT runtime JSON reports `linear_shape_stats` entries with `path_name`, `m`, `n`, `k`, transpose flags, call counts, `total_us`, and `avg_us` for TK BF16, TK float-output conversion, fused TK GELU/dGELU, cuBLASLt, cuBLAS GEMMEx BF16, and SGEMM paths. When the rebuilt v2 stats ABI is available, cuBLASLt rows also include `cublaslt_selected_heuristic`, `cublaslt_returned_heuristics`, and `cublaslt_workspace_bytes`, which prevents no-op heuristic overrides from being mistaken for real route changes. Timing uses CUDA events and synchronizes measured GEMMs, with a host-synchronized fallback for fused TK GELU rows whose helper dispatch is not captured by the stream event path, so leave this disabled for normal training; it is intended for paired kernel-candidate profiling on the dedicated compute GPU. `tools/paired_kernel_speed.py` also prints and ratios native backend counters such as `linear_tk_gemm_count`, `linear_cublaslt_gemm_count`, `linear_bf16_gemm_count`, and attention TK launch counts, so an active backend-route candidate is visible even when the higher-level strategy label is unchanged.

Run SM120 parity and candidate benchmarks only from a shell with real WSL GPU driver access. If `nvidia-smi` reports "GPU access blocked by the operating system" or the native JSON reports `cudaDriverGetVersion` as `0`, the result is an execution-environment failure rather than a kernel result; rerun the same command with GPU-visible execution before accepting or rejecting a candidate. `tools/paired_kernel_speed.py` now also takes a per-selected-GPU lock at `/tmp/nfn_paired_kernel_speed_gpu_<device>.lock` for measured runs, so two same-GPU paired benchmarks cannot overlap and contaminate each other before `nvidia-smi` observes a compute process. The default lock fails fast; pass `--gpu-benchmark-lock-timeout-seconds N` to wait, or `--no-gpu-benchmark-lock` only for intentionally unmanaged measurements. The utilization guard still rejects active compute load immediately, but it now retries transient high `nvidia-smi` utilization samples on an otherwise idle selected GPU; tune this with `--selected-gpu-utilization-retries` and `--selected-gpu-utilization-retry-interval-seconds` when a WSL/NVML idle poll spikes without compute processes.

`tools/bench_native_gpt_sm120_candidate.sh` accepts the native-specific `NFN_SM120_NATIVE_*` controls, the shorter `NFN_SM120_CANDIDATE_*` controls, and the shared parity-wrapper `NFN_SM120_PARITY_*` controls for common benchmark shape fields such as steps, samples, warmup, profile directory, stage timing, GPU selection, JSON output, and dry-run plan. Native-specific names win over candidate names, which win over parity names. Candidate-only env and candidate-only extra args stay separate, so `NFN_SM120_NATIVE_CANDIDATE_ENV` / `NFN_SM120_CANDIDATE_ENV` and `NFN_SM120_NATIVE_CANDIDATE_EXTRA_ARGS` / `NFN_SM120_CANDIDATE_EXTRA_ARGS` still affect only the candidate command. This keeps quick parity-to-native bisections from silently falling back to the candidate wrapper defaults of 10 steps, 3 samples, and 1 warmup.
The native-vs-native wrapper also forwards the selected-GPU utilization retry
aliases to the paired benchmark tool: use
`NFN_SM120_NATIVE_SELECTED_GPU_UTILIZATION_RETRIES`,
`NFN_SM120_CANDIDATE_SELECTED_GPU_UTILIZATION_RETRIES`,
`NFN_SM120_PARITY_SELECTED_GPU_UTILIZATION_RETRIES`, or
`NFN_SM120_SELECTED_GPU_UTILIZATION_RETRIES` to retry idle-utilization samples,
and the matching `..._RETRY_INTERVAL_SECONDS` aliases to control the pause
between samples. By default the wrapper now requires three idle samples spaced
0.25 seconds apart before each measured run. This is useful on WSL systems where
`nvidia-smi` can report a brief nonzero utilization spike on a display-disabled
GPU with no compute processes.
Stage timing is independent from profile sidecar generation: use
`NFN_SM120_NATIVE_STAGE_TIMING=1`, `NFN_SM120_CANDIDATE_STAGE_TIMING=1`, or
`NFN_SM120_STAGE_TIMING=1` with `NFN_SM120_PROFILE_DIR=none` when you need the
paired text/JSON stage buckets without writing per-run native profile files.
Candidate trainer executable selection is also separated: use
`NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN` or `NFN_SM120_CANDIDATE_TRAIN_BIN` for a
candidate compiled trainer while `NFN_NATIVE_GPT_TRAIN_BIN` remains the
baseline. This is required when measuring C++ trainer changes rather than only
Tile ops library or environment-switch changes.
For shared profiling env, prefer `NFN_SM120_COMMON_ENV` or its native/parity
aliases; the wrapper expands each key-value pair into both `--baseline-env` and
`--candidate-env` before invoking `tools/paired_kernel_speed.py`. Common,
baseline-only, and candidate-only env lists accept either whitespace-separated
assignments or comma-separated assignments; shape override values that contain
commas remain intact because splitting only occurs before a following `KEY=`
assignment.

The default dense GPT MLP projection backward now uses the BF16-only dGELU handoff path when BF16 MLP grad handoff is active, so the Tile kernel writes the BF16 gradient consumed by the following MLP FC backward stage without also converting it to an unused FP32 gradient buffer. When every trained block has stored MLP activations, the compiled trainer also skips the old FP32 `mlp.fc.grad_out` arena reservation, saving 805 MB at the default `64 x 1024 x 3072` hidden-gradient shape. The remaining one-buffer FP32-to-BF16 conversion path defaults to a guarded vec4 kernel for aligned buffers; set `NFN_NATIVE_GPT_F32_TO_BF16_VEC4=0` or `NFN_TILE_CUDA_F32_TO_BF16_VEC4=0` to compare against the scalar converter. Training JSON reports `block_backward_mlp_dgelu_float_grad_elided`, `block_backward_mlp_fc_grad_out_float_buffer_elided`, `block_backward_mlp_fc_grad_out_float_bytes_elided`, and the `...-no-float-grad` strategy suffix. Set `NFN_NATIVE_GPT_ELIDE_MLP_DGELU_FLOAT_GRAD=0` to restore the prior conversion/allocation path for paired kernel benchmarks.

The multi-buffer BF16 packer and stored-MLP activation pack/restore path also include vec4 CUDA candidates, but they remain default-off diagnostics. Set `NFN_NATIVE_GPT_F32_TO_BF16_MANY_VEC4=1` / `NFN_TILE_CUDA_F32_TO_BF16_MANY_VEC4=1` or `NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS_VEC4=1` / `NFN_TILE_CUDA_STORE_MLP_ACTIVATIONS_VEC4=1` only for paired bisection. The launchers require aligned buffers and fall back to the scalar kernels for ineligible shapes. The CUDA 13.3 dedicated RTX 5090 two-sample paired run measured the vec4-default baseline slower than the scalar candidate (`0.994143x` candidate/default train-loop wall time, `1.005941x` tokens/sec), so the scalar route stays the normal training path.

Keep compile-time kernel candidates isolated until a same-script paired run
holds up. For example, a Tile ops library built with
`NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP -DLLMK_SM120_APPROX_DGELU_TANH=1"`
remains an opt-in benchmark candidate because the dedicated RTX 5090 four-sample
repeat was noise-equivalent to the default library. The atomic-dQ packed-QKV
attention candidate,
`NFN_TILE_CUDA_EXTRA_NVCC_FLAGS=-DLLMK_SM120_ATOMIC_DQ`, now compiles through a
dedicated wrapper, but it remains rejected/default-off because the paired RTX
5090 benchmark measured `1.134435x` train-loop wall time and `0.881527x`
tokens/sec versus the current non-atomic packed-gradient default.

For profiling only, `NFN_TILE_CUDA_LINEAR_TK_FLOAT_OUT=1` or `NFN_NATIVE_LINEAR_TK_FLOAT_OUT=1` routes eligible BF16 linear forward GEMMs through the TK BF16-output bridge and converts the result back to float32. This path reports `linear_tk_float_out_gemm_count` in native GPT-2 JSON and remains disabled by default because the current full-shape TinyStories probe regressed overall throughput. The dense GPT trainer now tries the TK BF16 forward bridge before cuBLAS GEMMEx for no-bias BF16-input/BF16-weight/BF16-output GEMMs, including the default padded LM-head logits shape `50304,8192,768,T,N`. Normal native GPT JSON reports `lm_head_logits_tk_gemm_count`, `lm_head_logits_cublaslt_gemm_count`, and `lm_head_logits_bf16_gemm_count`, so `lm_head_logits_linear_strategy` stays accurate without enabling expensive `linear_shape_stats` timing. Set `NFN_TILE_CUDA_LINEAR_TK_FORWARD_DISABLE_SHAPE=50304,8192,768,T,N` or `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=50304,8192,768,T,N` to reproduce the older LM-head GEMMEx fallback for paired bisection. To bisection-test one additional forward TK shape against its fallback, set `NFN_TILE_CUDA_LINEAR_TK_FORWARD_DISABLE_SHAPE=m,n,k,opA,opB` or `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=m,n,k,opA,opB`, using the same `linear_shape_stats` tuple. The disable gate only applies to TK forward/fused-GELU launches with fallback paths; it deliberately does not affect bits-only backward dGELU kernels.

`NFN_NATIVE_LINEAR_TK_DINPUT=1` or `NFN_TILE_CUDA_LINEAR_TK_DINPUT=1` is a diagnostic-only route for BF16-gradient/BF16-weight Linear dInput shapes such as the dense GPT LM-head dHidden bucket `768,8192,50304,N,N`. It runs the existing TK BF16 matmul bridge and converts the BF16 result back to FP32 for downstream LayerNorm backward. Keep it disabled for normal training: the dedicated RTX 5090 paired benchmark measured the opt-in route at `1.049216x` train-loop wall time and `0.953102x` tokens/sec versus the default GEMMEx route.

The same raw ABI also exposes TK attention backward-to-QKV forward-workspace reuse for trainer loops that own the matching forward/backward ordering, plus a saved-workspace variant for loops that store the TK BF16 forward state per block.

GPT-2-compatible SDPA forward now attempts a dense causal full-head row-vector Tile kernel when `seq_k <= 1024`. One Tile CTA computes the score/softmax for a full query row and reuses it across all 64 value channels, instead of launching one scalar-output CTA and recomputing attention scores per output element. If CUDA rejects that row-kernel launch, the native Tile launcher records `cudaFuncGetAttributes` diagnostics, the pre-launch CUDA error state, the requested row-kernel grid/block shape, clears the error, auto-disables further row attempts for the run, and falls back to scalar Tile attention without repeated failed-launch overhead. Native GPT-2 plan/training JSON reports `attention_forward_strategy: "row-vector-tile-score-reuse"`, `attention_forward_value_chunk_size: 64`, `attention_forward_scalar_launch_fallback_enabled: true`, `attention_forward_row_launch_auto_disable_enabled: true`, runtime row/fallback/scalar launch counts, row-kernel attribute fields, pre-launch error codes, row launch grid/block fields, the row count, the old scalar output count, and the score-reuse/elision factor.

GPT-2-compatible SDPA backward now uses the same query-row score reuse contract for `qk_dim <= 64`, `value_dim <= 64`, and sequence lengths up to 1024. The native Tile launcher zeros Q/K/V gradient buffers, then launches one query-row CTA that computes the row softmax once, writes the full 64-channel `dQ`, and atomically accumulates `dK`/`dV` for every attended key row. This replaces the old scalar-output backward CTAs and avoids the key-row implementation that repeated a full query softmax scan per key row. Native plan/training JSON reports `attention_backward_strategy: "query-row-atomic-tile-score-reuse"`, `attention_backward_row_count`, `attention_backward_scalar_output_count`, `attention_backward_score_reuse_dim: 64`, and `attention_backward_scalar_cta_elision_factor: 192`.

Compiled GPT-2 `--train-transformer-lm` results include a `timing` block with host wall-clock phase timers: `setup_wall_ms`, `train_loop_wall_ms`, `validation_wall_ms`, `train_compute_wall_ms`, `post_train_sample_wall_ms`, `cleanup_wall_ms`, `checkpoint_wall_ms`, `total_wall_ms`, `optimizer_steps_per_second`, and `train_tokens_per_second`. The train-loop timer ends after an explicit end-of-loop device synchronization and before the diagnostic device-to-host sample copies used for status JSON; `--startup-only` elides those diagnostic sample copies and reports `post_train_diagnostic_samples_elided: true`. Cleanup is reported separately so startup-only probes distinguish readiness to train from teardown of large CUDA arenas.

The full GPT-2 transformer-LM trainer fuses both attention QKV layout directions. `nfn_native_tile_split_qkv_to_heads_add_bias_float32` consumes the no-bias QKV CUBLAS output, applies Q/K/V bias, and writes Q/K/V head-major buffers in one Tile launch per block instead of separate QKV bias add, QKV split, and three reshape launches. `nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32` lets TK SDPA backward read the row-major attention-output gradient directly and write row-major `grad_qkv` directly from bf16 head-major gradients, replacing three bf16-to-float gradient conversion launches plus the heads-to-QKV merge launch. The full trainer no longer allocates the three row-major or three head-major Q/K/V gradient scratch buffers for that path. Native plan/training JSON reports `qkv_forward_layout_strategy: "fused-split-to-heads"`, `qkv_bias_layout_strategy: "fused-qkv-bias-split-to-heads"`, `attention_backward_grad_layout_strategy: "merged-grad-out-direct"`, `attention_backward_qkv_bridge_strategy: "fused-bf16-heads-to-row-qkv"`, `qkv_backward_layout_strategy: "fused-heads-to-qkv"`, and the elided layout launches per block.

Forward attention also skips the obsolete row-major Q/K/V scratch tensors in the compiled GPT-2 trainer. The fused QKV split writes head-major `q_heads`, `k_heads`, and `v_heads` directly for attention, so the full 12-layer `64 x 1024` shape no longer reserves three unused activation buffers per tape, reducing the float arena by 150,994,944 elements, about 576 MiB. Native plan/training JSON reports `block_state_layout.forward_row_qkv_scratch_allocated: false` and `block_state_layout.forward_row_qkv_scratch_buffers_elided: 3`.

The full GPT-2 MLP path fuses the `c_fc` bias add with GELU. `nfn_native_tile_gelu_add_bias_float32` takes the no-bias CUBLAS `c_fc` output, writes the biased preactivation needed by GELU backward, and writes the GELU activation in one Tile pass instead of separate bias-add and GELU launches. Native plan/training JSON reports `mlp_fc_bias_gelu_strategy: "fused-bias-preactivation-gelu"` and the elided legacy launch per block.

The full GPT-2 projection residual path also fuses `c_proj`/attention-output bias with the residual add. By default, attention and MLP projection GEMMs write BF16 projection-output scratch and the residual consumers read those BF16 bits directly through `nfn_native_tile_linear_bias_residual_add_bf16_linear_float32` and the BF16-linear fused residual+LN2 variants, avoiding the rejected TK float-output conversion path. Set `NFN_NATIVE_GPT_BF16_PROJECTION_RESIDUAL=0` or `NFN_NATIVE_GPT2_BF16_PROJECTION_RESIDUAL=0` only for paired benchmarks against the older float projection-output residual path. Native plan/training JSON reports `bf16_projection_residual_enabled: true`, `attention_projection_input_strategy: "packed-o-bf16-direct-gemm-bf16-residual-consumer"`, `mlp_proj_forward_activation_strategy: "fused-gelu-bf16-act-direct-bf16-output-gemm"`, `projection_bias_residual_strategy: "fused-bf16-linear-bias-residual-add"`, and two elided legacy launches per block.
For the GPT-native 768-wide residual-add shape, the BF16 projection residual helper uses a dim-specialized CUDA kernel by default to avoid the generic per-element output-column modulo. Set `NFN_TILE_CUDA_DIM768_BF16_RESIDUAL_ADD=0`, `NFN_NATIVE_GPT_DIM768_BF16_RESIDUAL_ADD=0`, or `NFN_NATIVE_GPT2_DIM768_BF16_RESIDUAL_ADD=0` to reproduce the generic helper for paired bisection; the dedicated RTX 5090 same-script benchmark measured the specialized default at `0.998835x` mean train-loop wall time and `0.997518x` median total wall time versus the generic path.

`neuralfn/csrc/native_train/token_shards.cpp` provides the reusable no-Torch token-shard resolver and sequential batch sampler for native trainers. It resolves `NFN_DATASETS_DIR`, validates `fineweb_train_*.bin` / `fineweb_val_*.bin` uint16 shard alignment, accepts llm.kittens-style `TinyStories_train.bin` / `TinyStories_val.bin`, infers validation siblings for direct train-bin paths, skips the 1024-byte cached-shard header when present, sorts shards, counts train/validation tokens, computes microbatch/gradient-accumulation metadata, and can either produce `[batch, seq_len]` token plus next-token target vectors for smoke/debug output or write directly into caller-owned token/target uint16 buffers with `SequentialTokenBatchSampler::next_into()`. The compiled GPT trainer requests validation shard resolution only when `--eval-every-steps` and `--eval-batches` are both positive; benchmark or train-only-cache runs with `--eval-every-steps 0` skip validation shard discovery and report `validation_shards_required: false` plus `validation_shards_resolved: false` in plan/runtime JSON. The full GPT-2 trainer uses `next_into()` with pinned memory, so real batches do not materialize as `TokenBatch` vectors before upload and do not flow through graph nodes. The sampler reads contiguous shard segments for each batch instead of opening the shard once per sequence chunk; native token-shard JSON reports `batch_read_strategy: "contiguous_shard_segments"`.

Dense GPT Tile-CUDA training also stores LN1 mean/rstd stats for the earlier
saved packed-attention blocks by default when the BF16 QKV dWeight path is
active, then regenerates the needed LN1 BF16 activation during backward
recompute with
`nfn_native_tile_layer_norm_apply_stats_bf16_out_float32`. This avoids adding a
full BF16 LN1 activation tape while removing the reduction-heavy LN1 recompute
path. Runtime JSON reports `stored_packed_attention_ln1_stats_enabled`,
`stored_packed_attention_ln1_stats_blocks`,
`stored_packed_attention_ln1_stats_elements`, and
`stored_packed_attention_ln1_stats_bytes`. Set
`NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_STATS=0` for old-vs-new paired
benchmarking; GPT-2-prefixed names remain compatibility fallbacks.

`bash tools/build_native_missing_trainers.sh` builds compiled per-family native entrypoints for GPT-2 evo, NanoGPT token-only diagnostics, LLaMA, MixLLaMA, JEPA, semantic-router MoE, and DeepSeek-V4. GPT-2 evo now has a C++ native binary: `nfn_gpt2_evo_native_train --print-plan --eval-every-steps 1000 --tile-cuda-activation-dtype nvfp4` validates the dense GPT-2 shape, `adamw` profile, NVFP4 activation intent, validation cadence, and evo-layer index/cadence/population metadata, then reports that dense GPT-2-compatible templates delegate to `nfn_gpt_native_train --train-transformer-lm --layer-evo`. The delegated run stays Torch-free and reports `candidate_loss_source: "native-forward-loss-device-resident-current-batch"` plus `candidate_loss_transport: "device-to-device"` with `forward_candidate_evals` after scoring candidates through native CUDA forward loss and copying the resulting scalar loss directly between device buffers. NanoGPT normal full-transformer training now stays in the native dense GPT selector with `--template-name nanogpt` and reports `template-geometry-native-trainer-missing` until that loop supports NanoGPT dimensions and dropout; `nfn_nanogpt_native_train` remains available for explicit token-LM diagnostics and `--train-token-lm` runs. Its `--print-plan` parses the native NanoGPT defaults with the real GPT-2 tokenizer vocabulary (`50257`), validates the shape/AdamW contract, and emits JSON without importing Python or Torch; the JSON includes the contiguous parameter, gradient, AdamW state, clip-scale buffer layout, AdamW decay/no-decay groups, and a forward/backward/optimizer `training_step_plan` with each stage marked `ready`, `requires_wiring`, or `missing_abi`. It also supports `--check-tile-ops --tile-ops-lib PATH`, smoke commands for optimizer, LM, embedding/norm, QKV layout, fused attention, transformer block, MLP, and attention stages, plus `--train-token-lm --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS` for the older tied token-embedding LM loop with periodic validation loss. The unified frontend preserves `--print-command` in the printed NanoGPT token-LM delegate command, so installed `nfn-native-train --base-model nanogpt --train-token-lm --print-command ...` shows the exact second-stage command without running CUDA. The native NanoGPT preflight defaults to `dropout_p=0.0`; nonzero `--dropout-p` now reports ready deterministic inverted-dropout forward/backward Tile ABI stages through `nfn_native_tile_dropout_forward_float32` and `nfn_native_tile_dropout_backward_float32`. Pass `--require-token-shards` to require and report resolved cached token shards in that JSON, and `--sample-token-batch` to include the first native token/target batch. Non-dense-GPT family training still exits nonzero until their CUDA Tile forward/backward trainer loops land; dense GPT-2 evo uses the delegated dense trainer path above. The other binaries intentionally fail with family-specific CUDA Tile kernel requirements until the real native trainers are implemented, but they keep CLI, direct-script, and SDK handoff paths on compiled C++ artifacts instead of graph-backed Torch. `tools/install_native_gpt2_commands.sh` installs both underscore and hyphen command names for these entrypoints, so installed `nfn-native-train --base-model nanogpt ...` reports the geometry mismatch by default while `--train-token-lm` can still find `nfn_nanogpt_native_train`. Override a per-family target with `NFN_NATIVE_<MODEL>_CLI`, for example `NFN_NATIVE_NANOGPT_CLI=/path/to/nfn_nanogpt_native_train`.

Large raw-text datasets whose tokenizer ids fit in `uint16` are cached as `fineweb_train_*.bin` / `fineweb_val_*.bin` token shards on first training load. Subsequent runs estimate schedule length from metadata or shard sizes and memmap the token shards instead of re-tokenizing `data.txt` / `val.txt`. Tokenizers with ids outside `uint16` remain on the raw-text path. Dataset source nodes store dataset names and sequence length only; real text/token payloads stay in the dataset cache and do not pass through graph-editor node metadata.

Scalar CUDA Tile functions and simple elementwise modules also accept CUDA `float8_e4m3fn` and `float8_e5m2` activation inputs where outputs can be safely requantized: unary/binary/binary-pair scalar functions plus `loss_scale`, `logit_softcap`, `aux_loss_add`, `kl_penalty`, `residual_add`, `residual_mix`, `manifold_hyper_connection`, `qk_gain`, and `dyt`. These paths dequantize to float32 for Tile compute and requantize activation outputs back to the input fp8 format.

Strict CUDA Tile scalar function and simple module contract errors name the rejected tensor dtype plus the supported dtype set so dtype fallback issues are visible during training setup.

The SDK also provides deterministic fp8 reference quantize/dequantize helpers and an `NVFP4Tensor` reference packer using FP4 E2M1 values in blocks of 16 with FP8 E4M3 block scales and an FP32 tensor scale. NVFP4 support currently covers projection-family and attention-family activation contracts with boundary/reference coverage; optimizer, loss, and stochastic-mask contracts remain explicitly unsupported in the dtype matrix.

Registry `dtype_support` entries now use category-specific no-support reasons for lower-precision gaps, including losses/reductions, optimizers, stochastic mask producers, integer/hash/routing outputs, NF4 packed-format wrappers, host-only source nodes, and delegated child-graph calls.

`TemplateSpec.runtime` (`eager` / `compile` / `megakernel`) now drives `torch.compile` mode in `TorchTrainer.train()`, replacing the previous `TorchTrainConfig.compile` boolean as the primary control.

Project datasets are managed from a dedicated `Datasets` surface, then attached to graphs through a `dataset_source` node that now adapts its output roles to the active template shape:
- AR / H-Net / Universal: `tokens`, `targets`
- Seq2Seq: `enc_tokens`, `dec_tokens`, `targets`
- Diffusion / JEPA: `tokens`
- Semantic routing presets: `tokens`, `targets` plus a separate `semantic_data_source` node for `sem_targets`

`hnet_lm` uses a raw-byte training path (`vocab_size == 256`) instead of the normal tokenized loader. Trained weights are serialized back into graph JSON via `module_state`, and PyTorch weight round-tripping plus inference helpers live in `neuralfn/inference.py`.

The platform foundation now adds:
- authenticated users with HTTP-only session cookies
- multi-project, multi-session workspaces
- a routed React shell with Editor, Runs, Analytics, and Admin surfaces
- SQLAlchemy/Alembic-backed persistence with default SQLite and MySQL-ready configuration
- optional Redis-backed live state for refresh-safe restore and agent/run coordination
- project/session-scoped REST APIs and MCP tools

## How it works

1. **Choose neurons** from built-ins or define custom Python functions with `@neuron` and typed I/O ports.
2. **Wire them** into a directed graph with weighted edges.
3. **Probe & train** — scalar graphs sample each neuron to build differentiable surrogate models, then train connection weights via gradient descent or evolutionary search.
4. **Torch modules** — tensor-native graphs can train serialized module nodes through a PyTorch backend, including nested subgraphs built from multiple trainable stages.
5. **Use the platform** — store graphs inside project/session workspaces, inspect runs and analytics, and drive the same scoped graph APIs from the UI or MCP tools.

## Torch template workflows

- The editor toolbar, `/api/.../templates/gpt/*` routes, and MCP `load_gpt_template` all accept the same shipped preset names.
- Template graphs persist a serialized `template_spec` in `graph.torch_config`, so training, tracing, dataset loading, and exports can recover the original objective/backbone/tokenization contract without inferring it from node names.
- Dataset-backed tracing and training now route by input role rather than assuming only `(tokens, targets)`. That is what enables single-input JEPA/diffusion graphs and three-input Seq2Seq graphs to use the normal `dataset_source` workflow.
- `llm_jepa` uses an EMA target encoder and supports two masking strategies via `jepa_mask_strategy`: `"random"` (default, i.i.d. per-token) and `"block"` (contiguous span masking). Block masking is configured with `jepa_num_blocks`, `jepa_min_block_ratio`, and `jepa_max_block_ratio`. `diffusion` samples timesteps internally, and `hnet_lm` switches the dataset pipeline to raw bytes automatically.
- `semantic_router_moe` uses the same flat compiled input contract as the hybrid preset, `(tokens, targets, sem_targets)`, but keeps the backbone purely autoregressive. It projects the embedding output into vocab-topic space, hashes that semantic vector, resolves a shared batch-level expert route, broadcasts the chosen experts across the whole sequence, and applies that route to every MoE block without any JEPA encoder/EMA path.
- `jepa_semantic_hybrid` now expects three flat training inputs in compiled form: `(tokens, targets, sem_targets)`. Its dataset-backed root graph provides `tokens` and `targets` from `dataset_source`, materializes vocab-backed `sem_targets` from the active `vocab_86d_*.json` semantic vocabulary, hashes the pooled semantic state, routes the full hidden sequence into one expert per semantic vocabulary dimension, and trains the routed branch with AR next-token loss plus JEPA and semantic-alignment auxiliaries.
- `semantic_dense_jepa_evo` and `semantic_moe_jepa_evo` also expect `(tokens, targets, sem_targets)`, but work at chunk granularity instead of sequence granularity. `semantic_dense_jepa_evo` is the dense FFN control with no expert dispatch or route evolution. `semantic_moe_jepa_evo` adds the shared/semantic/free expert bank; `route_chunk_size` defaults to `32`; `semantic_shared_experts`, `semantic_free_experts`, and `experts` must stay consistent (`shared + NUM_VOCAB_DIMS + free`); and `route_evo_fraction` controls how often the trainer runs the lightweight evolutionary route-bias search.
- Semantic-only preview/training paths now synthesize safe placeholder `tokens` / `targets` tensors when a graph has `semantic_data_source` but no attached text dataset. That keeps semantic research graphs previewable and trainable without accidentally feeding categorical `sem_targets` into the token embedding path.
- The vocab-only semantic helpers keep `n_sig_buckets` on the higher-level data APIs, while the low-level hash helpers continue to use `n_buckets` and now accept `n_sig_buckets` as a compatibility alias. The sibling JEPA harness also derives semantic-row schedule estimates from the resolved `top_k`, so non-default routing runs report the right loader/epoch math before training starts.
- The trainer's vocab-only semantic dataset path now wraps the `load_training_targets()` `int64` arrays directly, so CUDA JEPA runs reach warmup/training without requiring a module-level NumPy import inside `torch_backend.py`.
- `TorchTrainConfig` now supports a parameter-golf-inspired split-optimizer profile with token-budgeted accumulation (`train_batch_tokens`), role-specific learning rates (`embed_lr`, `head_lr`, `tied_embed_lr`, `matrix_lr`, `scalar_lr`), Muon controls, warmup/warmdown scheduling, and gradient clipping.
- `CompiledTorchGraph` now executes each node through a fixed child module and precompiled edge-routing plan instead of routing mixed `Long` / BF16 / FP32 inputs through a single generic dispatcher or walking graph-editor node metadata per batch. That keeps real training tensors on the compiled hot path, makes `runtime="compile"` BF16 CUDA runs much more stable under `torch.compile`, and leaves scalar loss stages to upcast only inside their final reduction.
- `TorchTrainer.train()` now accepts an `on_step` callback for live warmup and optimizer-step progress. The sibling JEPA harness uses that hook together with `--train-log-every` so long CUDA runs keep printing visible progress.
- The sibling JEPA harness also ships `scripts/infer_jepa_semantic.py`, a CUDA-only probe that loads the exported `.json` graph plus `.pt` weights, traces the hybrid model's internal `model/softcap` or `model/lm_head` logits node, and samples text with the cached SentencePiece tokenizer from the dataset alias when available.
- The sibling SDK harness now also ships `scripts/train_semantic_router_moe.py` and `scripts/infer_semantic_router_moe.py` so the router-only control experiment can be trained and probed without the JEPA stack.
- All shipped sibling training and inference harnesses now auto-download a missing cached dataset alias by default when they can derive the standard cached-variant download contract from `owner__repo__variant__trainN` or from explicit dataset download flags. Existing cached aliases stay strict: tokenizer-backed contract mismatches still fail fast with the original validator error instead of a misleading missing-alias message.
- Saved graphs that still reference older block-family names such as `attn_block`, `transformer_block`, or `mixllama` now resolve through a compatibility alias layer instead of failing during template normalization when the equivalent current family is present.

## Editor behavior

- Toolbar, template, custom-node, subgraph, and variant-library inserts now default to the center of the visible graph viewport with a small stagger, so newly added nodes appear on screen even after panning or zooming away from the origin.
- Direct graph edits still preserve explicit positions; the viewport anchor is only the fallback when an add action does not originate from a canvas click.

## Architecture at a glance

- `neuralfn/` contains the core graph, neuron, variant, and trainer implementations.
- `server/` provides the FastAPI platform layer: auth, bootstrap, admin, projects, sessions, datasets, runs, SQLAlchemy persistence, Alembic migrations, and optional Redis live state.
- `editor/` contains the routed React app shell and graph editor UI.

## Quick start

### Install Python dependencies

```bash
pip install -r requirements.txt
```

The default requirements intentionally do not install Torch. Native dense GPT
training and dataset-cache preparation do not require it. The aggregate `.[all]`
extra is also Torch-free, and the package no longer exposes a `.[torch]` extra.
Legacy graph-backed Torch workflows require a separately managed PyTorch install
outside NeuralFn's package metadata.

After rebuilding native training artifacts, run the dependency gate to verify
the default package metadata still keeps Torch out of hard dependencies, the
aggregate `.[all]` extra remains Torch-free, no `torch` extra is advertised, the
compiled artifacts still avoid Torch, c10, and Python runtime libraries, and
default native GPT Python training and inference entrypoints can construct their compiled-C++ commands or
inspect native checkpoints while imports of `torch`, NumPy, tiktoken,
`server.dataset_manager`, and `nfn_impl` are blocked.
The gate also imports the top-level native SDK exports such as
`NativeGptRunConfig`, `build_native_gpt_compiled_cli_run_config()`,
`native_gpt_kernel_backend()`, and `native_gpt_parameter_count()` under the same
blocked-import guard:

```bash
python tools/check_native_no_torch_deps.py
```

### Install the SDK as a package

From the repository root:

```bash
pip install -e .
```

From a sibling project checked out next to the repo:

```bash
pip install -e ../NeuralFn
```

This installs the `neuralfn` package in editable mode and includes the shipped
semantic vocabulary files under `neuralfn/data/semantic/` as package data.

### Install the local CLI

The repo also includes the `nfn` CLI package under `cli/`:

```bash
cd cli
./install.sh
nfn --help
```

The CLI installer keeps Torch optional, builds the native GPT C++ binding,
launcher, no-Python cached-shard CLI, and unified native training frontend, and
links `nfn-gpt-native`, `nfn-gpt-native-train`, compatibility `nfn-gpt2-native`
names, and `nfn-native-train` into the active Python scripts
directory. Pass `./install.sh --no-native` to skip native artifact builds. The
generic SDK binding can execute normal `argv` configs and GPT compiled-CLI
configs; alias-only GPT configs prefer `compiled_cli_argv` so dataset aliases
stay on the compiled C++ resolver instead of raw external trainer paths. The
GPT native CLI exposes `--backend tile-cuda`; this is the
default NeuralFn-owned path and runs the dense `--train-transformer-lm`
trainer unless an introspection command such as `--print-plan`, `--check-tile-ops`,
or `--no-train-transformer-lm` is used. Use `nfn train --base-model gpt ...` as the canonical native
trainer entrypoint. `--base-model gpt2` and `--base-model gpt3` are dense GPT
aliases that route to the same CUDA Tile C++ trainer; GPT3 defaults to a
2048-token context when selected by `--base-model gpt3` or
`--template-name gpt3`, unless a custom graph or explicit `--train-seq-len` is
supplied. The implicit GPT3 batch size is 32 to preserve the 65,536-token
microbatch; explicit `--batch-size` wins. Otherwise the selected GPT template
or `--graph-file` is the architecture source of truth. The full `nfn train`
parser and planner accept the same aliases, and graph-backed compatibility
paths canonicalize those dense GPT aliases back to the GPT-compatible template
builder instead of creating separate GPT-2/GPT-3 trainers. The Tile plan includes the GPT parameter layout and
forward/backward/optimizer stage sequence; the training JSON reports
`block_state_layout` flags for per-block allocation, initialization, gradient
zeroing, gradient clipping, AdamW update, checkpoint export, activation tape,
forward block, and backward block loops. The 12-layer path uses one scratch tape
with recompute plus persistent block outputs to avoid per-layer full-tape
allocation.
The CLI provides composed `train`, `infer`, and `eval`
workflows with dataset shortcuts, tokenizer selection, graph-linked artifacts,
and optional interactive planning. See [CLI Workflows](docs/cli.md) and
[cli/README.md](cli/README.md).
Direct `python cli/scripts/train_gpt.py ...` is the canonical dense GPT script.
It uses the same pre-import compiled CLI fast path as the compatibility
`train_gpt2.py`, so dry-run and default native training commands do not load
Torch, NumPy, the dataset manager, or the compatibility Python harness before
handing off to C++.

### Run the library examples

```bash
python examples/xor_graph.py
python examples/nested_hybrid_graph.py
python examples/gpt_graph.py
python examples/tile_cuda/scalar_add_train.py
python examples/tile_cuda/kernel_bench.py
```

### Install editor dependencies

The repo currently tracks frontend dependencies with `pnpm` (`editor/pnpm-lock.yaml`):

```bash
cd editor
pnpm install
```

### Platform configuration

By default, the backend starts with a local SQLite database at `neuralfn.db`, stores snapshots in `server/session_snapshots`, stores artifacts in `~/NeuralFn/artifacts`, and allows the standard Vite dev origins. Configure the platform with environment variables as needed:

| Variable | Purpose | Default |
|----------|---------|---------|
| `NEURALFN_DATABASE_URL` | SQLAlchemy database URL. Use MySQL in shared environments. | `sqlite:///.../neuralfn.db` |
| `NEURALFN_REDIS_URL` | Optional Redis live state/event store. If unset, the server uses in-memory live state. | unset |
| `NEURALFN_CREATE_SCHEMA_ON_STARTUP` | Auto-create tables on app startup. Set to `0` if you want migration-only schema management. | `1` |
| `NEURALFN_SNAPSHOTS_DIR` | Filesystem location for persisted session snapshots. | `server/session_snapshots` |
| `NEURALFN_ARTIFACTS_DIR` | Filesystem location for saved artifacts. | `~/NeuralFn/artifacts` |
| `NEURALFN_ALLOW_ORIGINS` | Comma-separated CORS origins. Must include the frontend origin when using cookies. | `http://127.0.0.1:5173,http://localhost:5173` |
| `NEURALFN_SESSION_COOKIE_NAME` | Session cookie name used by the web app and API. | `neuralfn_session` |
| `NEURALFN_SESSION_TTL_SECONDS` | Session lifetime in seconds. | `1209600` |

`.gitignore` excludes the default SQLite file, downloaded datasets under `~/.cache/nfn/datasets/`, session snapshots, artifacts, local `.env` files, and common caches so they are not committed.

If you want migration-managed startup instead of auto-creating tables, run:

```bash
alembic upgrade head
```

and set:

```bash
export NEURALFN_CREATE_SCHEMA_ON_STARTUP=0
```

### Start the platform

Terminal 1 — backend:

```bash
uvicorn server.app:app --reload --port 8000
```

Terminal 2 — frontend:

```bash
cd editor
pnpm dev
```

Open <http://localhost:5173>.

### Running in Desktop Mode (Electron)

NeuralFn can also be run as a unified desktop application. In desktop mode, it runs completely offline, storing its database (`neuralfn.db`), snapshots, and training artifacts securely inside the user's OS application data directory. **No Redis or external service is required**—the app automatically boots with zero-config local Memory and SQLite storage.

#### Install Desktop dependencies:
```bash
# Install desktop packages
npm run desktop:install
```

#### Start the Desktop App locally:
To compile the React editor, copy assets, scan and bind to an open port, and launch the Electron frame:
```bash
npm run desktop:start
```

#### Package for Windows, Mac, and Linux:
To bundle everything (backend scripts, assets, electron launcher) into a standalone production installer or executable:
```bash
npm run desktop:build
```
The compiled artifacts will be generated in `desktop/dist/`.


### First-run workflow

1. On the first launch, the app checks `/api/bootstrap`. If no users exist, the login screen switches into **Create Admin Workspace** mode.
2. Creating the first admin also creates a default project and editor session, then signs the user in with an HTTP-only session cookie.
3. Subsequent logins reuse `/api/auth/login` and restore the last active project/session scope stored on the auth session.
4. Any authenticated user can create a personal project from the header controls. Each new project is seeded with a `Main session` automatically and becomes the active workspace immediately.
5. Admin users can additionally create users and manage project memberships from the Admin surface.

## Routed platform surfaces

After authentication, the app routes into the scoped shell under `/app`:

- `Editor` — `/app/projects/:projectId/sessions/:sessionId/editor`
- `Datasets` — `/app/projects/:projectId/sessions/:sessionId/datasets`
- `Runs` — `/app/projects/:projectId/sessions/:sessionId/runs`
- `Analytics` — `/app/projects/:projectId/sessions/:sessionId/analytics`
- `Admin` — `/app/admin`

The app shell keeps the active project and session in the header, lets authenticated users create a new personal project, updates the server-side active scope when you switch workspaces, and routes each surface to the matching project/session pair.

## Refresh-safe session restore

The editor is no longer an anonymous in-memory graph:

- editor routes always load a concrete `projectId` and `sessionId`
- the editor hydrates the session graph and revision from the backend on load
- autosave writes back through the session-scoped graph API
- if the backend detects a revision conflict, the client reloads the latest graph instead of silently overwriting it
- if an MCP agent is actively controlling the session, the UI shows a banner and reloads the graph after the agent releases control

## API surface

The platform API is mounted under `/api` and is split into dedicated routers:

- `/api/bootstrap` for initial app bootstrap state
- `/api/auth/*` for bootstrap-admin, login/logout, identity, and active-session selection
- `/api/admin/*` for user/project membership management
- `/api/projects/*` for project listing, creation, and analytics
- `/api/projects/{project_id}/datasets/*` for dataset catalog listing, download/upload, access grants, and deletion
- `/api/projects/{project_id}/sessions/{session_id}/*` for graph editing, dataset wiring, execution, tracing, templates, and agent status
- `/api/projects/{project_id}/sessions/{session_id}/runs/*` for training runs and run status

Legacy helper wrappers still exist in `server/routes.py` for older route-based tests, but all product-facing editor and MCP flows now use explicit project/session-scoped endpoints.

## MCP Server (AI agent integration)

NeuralFn ships with an [MCP](https://modelcontextprotocol.io/) server that exposes graph-editing and training operations as tools. The MCP server now authenticates against the platform and works against explicit project/session scopes instead of a single global graph.

### Prerequisites

- The FastAPI backend must be running on port `8000` before starting the MCP server.
- The MCP server needs valid platform credentials. Export them in the environment used to launch your MCP client:

```bash
export NEURALFN_MCP_EMAIL="admin@example.com"
export NEURALFN_MCP_PASSWORD="secret123"
# Optional if your API is not running at http://localhost:8000/api
export NEURALFN_BASE_URL="http://localhost:8000/api"
```

- MCP currently assumes the target project and session already exist. Create/select them in the web UI or via the HTTP API first.

### Configuration

Add the NeuralFn MCP server to your client configuration. The server uses inline script metadata (PEP 723), so `uv run` resolves the `mcp` dependency automatically.

**Codex project config** (`.codex/config.toml` in the project root):

```toml
[mcp_servers.neuralfn]
command = "uv"
args = ["run", "server/mcp_server.py"]
cwd = "/path/to/NeuralFn"
```

**Cursor** (`.cursor/mcp.json` in the project root):

```json
{
  "mcpServers": {
    "neuralfn": {
      "command": "uv",
      "args": ["run", "server/mcp_server.py"]
    }
  }
}
```

Codex reads MCP server definitions from `config.toml` rather than Cursor's `.cursor/mcp.json`. Project-scoped Codex MCP config only applies to trusted projects.

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "neuralfn": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/NeuralFn", "server/mcp_server.py"]
    }
  }
}
```

### MCP scope rules

- Most graph, node, edge, variant, execution, template, and training tools require both `project_id` and `session_id`.
- Dataset catalog tools (`list_datasets`, `download_dataset`, `set_dataset_access`, `delete_dataset`) are project-scoped and require `project_id`.
- `list_builtins()` is global and does not require project/session context.

### Available tools

**Graph** (`project_id`, `session_id`)

| Tool | Description |
|------|-------------|
| `get_graph` | Get the current graph summary for one project/session. |
| `replace_graph` | Replace the entire graph for one project/session. |
| `update_graph_settings` | Update graph name, training method, runtime, or config dicts. |
| `set_io` | Set which nodes are graph inputs/outputs. |

**Nodes** (`project_id`, `session_id`)

| Tool | Description |
|------|-------------|
| `list_builtins` | List available builtin neuron definitions. |
| `add_node` | Add a builtin neuron by id. |
| `add_custom_node` | Add a function node with custom Python source code and ports. |
| `add_subgraph_node` | Add an empty subgraph node (with internal input/output nodes). |
| `add_variant_node` | Add a node linked to a variant from the variant library. |
| `get_node` | Get the full details of a single node. |
| `update_node` | Update a node's name, source code, ports, or module config. |
| `delete_node` | Delete a node. |
| `update_node_positions` | Batch-update node canvas positions. |

**Edges** (`project_id`, `session_id`)

| Tool | Description |
|------|-------------|
| `add_edge` | Connect two node ports with optional weight/bias. |
| `update_edge` | Update an edge's weight and/or bias. |
| `delete_edge` | Delete an edge. |

**Variants** (`project_id`, `session_id`)

| Tool | Description |
|------|-------------|
| `list_variants` | List all variant families and versions in the current scoped graph. |
| `save_node_as_variant` | Save a subgraph node into the variant library. |
| `swap_node_variant` | Swap a node to a different variant version. |

**Execution & Training** (`project_id`, `session_id`)

| Tool | Description |
|------|-------------|
| `execute_graph` | Run the graph with scalar inputs. |
| `execute_trace` | Run the graph and trace intermediate outputs. |
| `trace_torch` | Trace a torch graph for tensor statistics. |
| `probe_node` | Probe a node's response curve. |
| `train_start` | Start training (surrogate/evolutionary/hybrid/torch). |
| `get_training_status` | Read the active training snapshot, latest loss, and recent events. |
| `poll_training_status` | Wait for the next training update by `event_id`, or until the run finishes. |
| `train_stop` | Stop the current training run. |
| `load_gpt_template` | Build and load a GPT/Llama/MoE graph in one call. |

**Datasets** (`project_id` or `project_id` + `session_id`)

| Tool | Description |
|------|-------------|
| `list_datasets` | List datasets visible to a project. |
| `download_dataset` | Download a Hugging Face dataset into the project catalog and optionally share it with other accessible projects. |
| `load_dataset_source` | Download/load datasets and wire them into a `dataset_source` node for one session graph. |
| `set_dataset_access` | Update which accessible projects can use one dataset. |
| `delete_dataset` | Delete a dataset from the project catalog. |

### Example: building a graph inside an existing session

An AI agent can build a simple sigmoid pipeline like this (shown as sequential tool calls):

```text
1. add_node(project_id="proj_123", session_id="sess_456", neuron_id="builtin-input", instance_id="in", position=[100, 200])
2. add_node(project_id="proj_123", session_id="sess_456", neuron_id="builtin-sigmoid", instance_id="sig", position=[350, 200])
3. add_node(project_id="proj_123", session_id="sess_456", neuron_id="builtin-output", instance_id="out", position=[600, 200])
4. add_edge(project_id="proj_123", session_id="sess_456", src_node="in", src_port=0, dst_node="sig", dst_port=0)
5. add_edge(project_id="proj_123", session_id="sess_456", src_node="sig", src_port=0, dst_node="out", dst_port=0)
6. set_io(project_id="proj_123", session_id="sess_456", input_ids=["in"], output_ids=["out"])
7. execute_graph(project_id="proj_123", session_id="sess_456", inputs={"in": [0.5]})
```

### Example: train an MoE on FineWeb via MCP

```text
1. load_gpt_template(project_id="proj_123", session_id="sess_456", name="fineweb_moe", preset="moe", config={"n_layer": 4, "n_head": 4, "n_embd": 128, "num_experts": 4, "top_k": 2})
2. load_dataset_source(project_id="proj_123", session_id="sess_456", hf_path="HuggingFaceFW/fineweb", hf_split="train", max_rows=10000, seq_len=64)
3. train_start(project_id="proj_123", session_id="sess_456", method="torch", epochs=10, learning_rate=0.001)
4. get_training_status(project_id="proj_123", session_id="sess_456")
5. poll_training_status(project_id="proj_123", session_id="sess_456", since_event_id=0, timeout_seconds=30)
```

## Testing

Run the Python test suite with:

```bash
python -m unittest discover -s tests
```

Useful targeted checks:

```bash
python -m unittest discover -s tests -p "test_platform_api.py"
python -m unittest discover -s tests -p "test_server_dataset_loading.py"
python -m unittest discover -s tests -p "test_server_nested_graphs.py"
```

The platform API suite covers bootstrap-admin, login/session scope behavior, refresh-safe graph restore, and revision conflict handling.

For frontend type/build validation:

```bash
cd editor
pnpm build
```

## Using built-in neurons

```python
from neuralfn import BuiltinNeurons, NeuronGraph, NeuronInstance

g = NeuronGraph()
g.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
g.add_node(NeuronInstance(BuiltinNeurons.sigmoid, instance_id="act"))
g.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
```

## Defining a custom neuron

```python
from neuralfn import neuron, Port

@neuron(
    inputs=[Port("x", range=(-5, 5), precision=0.01)],
    outputs=[Port("y", range=(0, 1), precision=0.001)],
)
def custom_sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))
```

## Graph execution

```python
from neuralfn import BuiltinNeurons, NeuronGraph, NeuronInstance, Edge

g = NeuronGraph()
g.add_node(NeuronInstance(BuiltinNeurons.sigmoid, instance_id="s1"))
# ... add more nodes and edges
result = g.execute({"input_node_id": (0.5,)})
```

## Nested graphs and mixed trainers

```python
from neuralfn import (
    BuiltinNeurons,
    Edge,
    HybridConfig,
    HybridTrainer,
    NeuronGraph,
    NeuronInstance,
    subgraph_neuron,
)

child = NeuronGraph(name="child", training_method="surrogate")
child.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
child.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
child.add_edge(Edge(id="child-edge", src_node="in", src_port=0, dst_node="out", dst_port=0))
child.input_node_ids = ["in"]
child.output_node_ids = ["out"]

root = NeuronGraph(name="root", training_method="frozen")
root.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="root_in"))
root.add_node(
    NeuronInstance(
        subgraph_neuron(child, name="child_block", input_aliases=["x"], output_aliases=["y"]),
        instance_id="child_block",
    )
)
root.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="root_out"))

trainer = HybridTrainer(root, HybridConfig(outer_rounds=3))
```

Subgraph nodes expose their ports from the nested graph’s designated `input_node_ids` and `output_node_ids`, and each graph picks its own `training_method`: `surrogate`, `evolutionary`, `frozen`, or `torch`.

## GPT / torch graphs

Use the GPT template generator when you want a causal language model that remains explorable in the editor. The templates expand into intricate graphs of token embedding, residual-mix, RMSNorm, attention, MLP (Dense or Mixture of Experts), skip-add, head, softcap, and token cross-entropy stages. Transformer blocks are represented as nested subgraphs via the Variant Library, allowing easy exploration of architecture choices. Torch graphs should use `training_method="torch"` and `runtime="torch"`. The torch trainer is CUDA-first.

Training data for GPT graphs is managed via a `dataset_source` node. Use the `Datasets` tab to download or upload datasets and choose which of your accessible projects can see them, then select those datasets from the `dataset_source` node side panel and connect its output roles to the network's inputs. For standard AR-style graphs that means `tokens` and `targets`; for semantic routing presets that means `tokens` and `targets` plus the shipped `semantic_data_source` node feeding `sem_targets`. Dataset-backed training now resolves from the saved graph node configuration, so the node is the source of truth rather than a temporary bottom-panel selector. The trainer will automatically tokenize the text, handle batching, and still auto-expand `vocab_size` for manual or tokenizer-less inputs when needed. Tokenizer-backed cached shard aliases are stricter now: NeuralFn validates shard token ids, tokenizer artifacts, and the graph/checkpoint vocab up front, and it fails fast if they disagree instead of silently resizing embeddings or crashing during decode. The recovery path for a bad cached alias is to delete and rebuild or re-download it with matching tokenizer artifacts.

## Training

**Surrogate (gradient-based):**
```python
from neuralfn import SurrogateTrainer
from neuralfn.trainer import TrainConfig

trainer = SurrogateTrainer(graph, TrainConfig(epochs=300))
losses = trainer.train(X, Y)
```

**Evolutionary:**
```python
from neuralfn import EvolutionaryTrainer
from neuralfn.evolutionary import EvoConfig

evo = EvolutionaryTrainer(graph, EvoConfig(generations=200))
losses = evo.train(X, Y)
```

## Built-in neurons

The editor palette and MCP `list_builtins` / `add_node` tools draw from a single catalog defined in [`neuralfn/builtins.py`](neuralfn/builtins.py) (`BuiltinNeurons.all()`), also served at `/builtins` for the web app. **Scalar** entries (`kind: function`) run on the scalar graph runtime and work with surrogate or evolutionary training. **Torch module** entries (`kind: module`) wrap `torch.nn` stages and expect a graph with `runtime: "torch"` and `training_method: "torch"` (see [GPT / torch graphs](#gpt--torch-graphs) and `examples/gpt_graph.py`).

Import in Python:

```python
from neuralfn import BuiltinNeurons

BuiltinNeurons.sigmoid          # scalar
BuiltinNeurons.linear_module    # torch module — pair with runtime="torch"
```

**Duplicate display name:** the catalog contains **two** definitions whose `name` field is `gelu`: the scalar `@neuron` and the tensor module (`module_type: "gelu"`). In JSON they differ by `kind` and (for the module) `module_type`; when resolving by name in code, `BUILTIN_MAP` keeps a single winner—prefer `BuiltinNeurons.gelu` vs `BuiltinNeurons.gelu_module` explicitly.

### Scalar activations and unary ops

- **sigmoid** — Logistic activation \(1 / (1 + e^{-x})\).
- **relu** — ReLU: \(\max(0, x)\).
- **tanh_neuron** — Hyperbolic tangent.
- **threshold** — Step: 1 if \(x \ge 0\), else 0.
- **identity** — Passthrough.
- **negate** — Unary negation.
- **gaussian** — \(e^{-x^2}\).
- **log_neuron** — Natural log with a small floor on \(x\) (Python: `BuiltinNeurons.log_neuron`).
- **leaky_relu** — Leaky ReLU (small slope for \(x < 0\)).
- **prelu** — Parametric ReLU–style slope for negative inputs (fixed coefficient in this scalar form).
- **relu6** — ReLU clipped at 6.
- **elu** — Exponential linear unit for negative inputs.
- **selu** — Scaled ELU constants for self-normalizing-style behavior at scalar resolution.
- **gelu** (scalar) — Gaussian error linear unit using `erf`.
- **silu** — SiLU / Swish: \(x \cdot \sigma(x)\).
- **mish** — \(x \cdot \tanh(\text{softplus}(x))\).
- **softplus** — Smooth ReLU-like \(\log(1 + e^x)\).
- **softsign** — \(x / (1 + |x|)\).
- **hard_sigmoid** — Piecewise-linear sigmoid approximation.
- **hard_tanh** — Piecewise-linear tanh approximation.
- **hard_swish** — Piecewise-linear Swish approximation.

### Scalar binary ops and two-logit heads

- **add** — Sum of two inputs.
- **multiply** — Product of two inputs.
- **softmax_2** — Softmax over two scalars; two probability outputs.
- **logsoftmax_2** — Log-softmax over two scalars; two log-probability outputs.

### Graph terminals

- **input** — Graph input terminal (Python: `BuiltinNeurons.input_node`).
- **output** — Graph output terminal (Python: `BuiltinNeurons.output_node`).

### Torch — embeddings and positions

- **token_embedding** — Embedding lookup: token IDs to hidden states; second output exposes embedding weights (for tied heads).
- **absolute_position_embedding** — Adds learned position vectors along the sequence (expects token-derived stream shape).

### Torch — linear and MLP blocks

- **linear** — Trainable dense layer \(y = xW + b\) (dimensions and bias from `module_config`).
- **mlp_relu2** — Two-layer MLP with ReLU-squared activation between projections (width from `module_config`).
- **gelu** (module) — Tensor GELU activation (`module_type: "gelu"`; Python: `BuiltinNeurons.gelu_module`).
- **swiglu** — SwiGLU-style gated MLP block (LLaMA-style; width/multiple-of from `module_config`).

### Torch — normalization and regularization

- **rms_norm** — RMS normalization over the last dimension.
- **layer_norm** — Layer normalization over the last dimension.
- **dropout** — Dropout during training (rate `p` in `module_config`).

### Torch — attention and residuals

- **reshape_heads** — Reshape projected hidden states into multi-head layout for attention.
- **merge_heads** — Merge per-head tensors back to model width.
- **repeat_kv** — Repeat grouped key/value heads to match query head count (GQA).
- **rotary_embedding** — Apply RoPE to Q and K tensors (two inputs, two outputs).
- **qk_gain** — Learned per-head scaling on the query stream before attention scores.
- **scaled_dot_product_attention** — Multi-head scaled dot-product attention (Q, K, V in; causal mask from `module_config`).
- **causal_self_attention** — Full causal self-attention block (projections, RoPE, GQA repeat, SDPA, output projection) as one stage.
- **residual_mix** — Learned per-channel blend of main path and skip (`x` vs `x0`).
- **residual_add** — Skip connection: residual plus learned per-channel scaled delta.
- **kv_cache_read** — Concatenate prior cached K/V with current K/V along the sequence dimension when caches are provided; otherwise passthrough.
- **kv_cache_write** — Identity on K/V outputs; used as a structural marker in inference-style graphs.

### Torch — language-model head and loss

- **tied_lm_head** — Projects hidden states to logits using supplied embedding weights (second input).
- **lm_head** — Standalone linear head to vocabulary logits.
- **logit_softcap** — Stabilizes logits (e.g. tanh-based cap) before softmax/loss.
- **token_cross_entropy** — Cross-entropy loss between logits and integer token targets.

### Torch — mixture-of-experts

- **router_logits** — Linear router from hidden state to per-expert scores.
- **topk_route** — Softmax over router, then top-k expert weights and indices.
- **expert_dispatch** — Sparse expert MLP (SwiGLU per expert) weighted by routing.
- **expert_combine** — Identity passthrough for graph wiring after dispatch.
- **load_balance_loss** — Auxiliary load-balancing loss from router statistics; passes router logits through.
- **aux_loss_add** — Adds scaled auxiliary tensor loss to the main scalar loss (`coef` in `module_config`).

### Torch — data source

- **dataset_source** — No inputs; emits `tokens` and `targets` from configured project datasets (`dataset_names`, `seq_len` in `module_config`). See [GPT / torch graphs](#gpt--torch-graphs) for wiring and the Datasets UI.

### Alphabetical index (selected catalog entries)

`absolute_position_embedding`, `add`, `aux_loss_add`, `causal_self_attention`, `dataset_source`, `dropout`, `elu`, `expert_combine`, `expert_dispatch`, `gaussian`, `gelu` (scalar function), `gelu` (torch module), `hard_sigmoid`, `hard_swish`, `hard_tanh`, `identity`, `input`, `kv_cache_read`, `kv_cache_write`, `layer_norm`, `leaky_relu`, `linear`, `lm_head`, `load_balance_loss`, `log_neuron`, `logit_softcap`, `logsoftmax_2`, `merge_heads`, `mish`, `mlp_relu2`, `multiply`, `negate`, `output`, `prelu`, `qk_gain`, `relu`, `relu6`, `repeat_kv`, `reshape_heads`, `residual_add`, `residual_mix`, `rms_norm`, `rotary_embedding`, `router_logits`, `scaled_dot_product_attention`, `selu`, `sigmoid`, `silu`, `softmax_2`, `softplus`, `softsign`, `swiglu`, `tanh_neuron`, `threshold`, `tied_lm_head`, `token_cross_entropy`, `token_embedding`, `topk_route`.

For the full, current builtin catalog including the experimental JEPA semantic modules, see [`docs/python-sdk/builtins.md`](docs/python-sdk/builtins.md) and [`neuralfn/builtins.py`](neuralfn/builtins.py).
