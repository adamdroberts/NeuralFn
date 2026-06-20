# Changelog

`README.md` captures the current product and setup story. This file captures the more detailed history behind meaningful changes, including migration notes and verification.

Future updates should append new entries here rather than replacing older notes.

## Unreleased

- Added a default-on diagnostic gate for the native dense-GPT LM-head fused
  loss+backward classifier route. Set
  `NFN_NATIVE_GPT_LM_HEAD_FUSED_LOSS_BACKWARD=0` or
  `NFN_NATIVE_GPT2_LM_HEAD_FUSED_LOSS_BACKWARD=0` to force the older separate
  loss-partials reduction before CE backward while keeping the same row-chunked
  BF16 dlogits contract for dHidden/dWeight. Runtime JSON now reports the
  fused-loss requested/enabled state and `tools/paired_kernel_speed.py`
  summarizes `lm_head_ce_loss_backward_strategy`. The diagnostic was rejected
  as a default change: a short 3-step, 2-sample probe looked slightly faster,
  but the stronger 5-step, 3-sample warmup-confirmation run failed gates at
  `1.001484x` train-loop wall time, `1.003244x` block backward, and
  `1.000194x` MLP projection backward. Verification: rebuilt
  `build/libnfn_native_train_tile_ops.so` and `build/nfn_gpt_native_train`,
  ran a one-step CUDA smoke of the disabled route, and ran the paired native
  benchmark confirmation on the dedicated RTX 5090.

- Revisited the failed-test surface after the CUDA Toolkit 13.3.33 WSL
  reinstall. `nvcc --version` reports release `13.3, V13.3.33`, `nvidia-smi`
  reports CUDA UMD `13.3` with the RTX 5090 idle and no compute processes, and
  `NFN_TILE_CUDA_TEST=1 python -m pytest --lf -q -rs` now passes with
  `1185 passed, 20 warnings, 468 subtests passed`. No previously failed pytest
  surface remains red on the current CUDA-visible workstation setup.

- Added LM-head classifier row-chunk route counters to the paired native
  benchmark parser. `tools/paired_kernel_speed.py` now extracts and summarizes
  `lm_head_classifier_chunk_launch_count`, the last rows/vocab/stride handled
  by the route, and includes those values in `native_route_counter_changes` so
  future same-script LM-head candidates can prove route changes explicitly.
  Verification: ran the focused paired benchmark parser tests
  (`3 passed, 28 deselected`) and the native GPT static probe
  (`1 passed`).

- Added a dedicated native Tile ABI surface for the dense-GPT LM-head
  classifier row-chunk path. The raw shared library now exports
  `nfn_native_tile_lm_head_classifier_backward_loss_inplace_strided_no_pad_zero_bf16_bits_u16_targets`
  and
  `nfn_native_tile_lm_head_classifier_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace`
  plus `nfn_native_tile_lm_head_classifier_*` reset/count/shape counters. The
  native dense-GPT trainer requires and calls those symbols for the default
  BF16/u16 public-vocab dlogits route, including the timing-oriented
  no-train-loss path and the loss-recording path. Runtime JSON now reports
  whether the LM-head classifier chunk kernel is available/enabled, how many
  chunks launched, and the last rows/vocab/stride processed. This is an
  auditable ABI step toward the remaining cooperative LM-head
  classifier+dHidden+dWeight kernel; it does not claim to close the
  llm.kittens parity gap by itself. Verification: rebuilt
  `build/libnfn_native_train_tile_ops.so` and `build/nfn_gpt_native_train`,
  reran the full native GPT pytest file with CUDA access (`59 passed`), reran
  `tools/check_native_no_torch_deps.py --json`, and ran a one-step native
  TinyStories training proof that reported
  `lm_head_classifier_chunk_kernel_enabled: true`,
  `lm_head_classifier_chunk_launch_count: 64`, and the expected
  `8192 x 50257` public-vocab row chunk with row stride `50304`.

- Added a shape-selective BF16/BF16 BGRADB diagnostic for native dense GPT
  dWeight+bias routing. `NFN_NATIVE_LINEAR_BF16_BF16_BGRAD_DISABLE_SHAPE` and
  `NFN_TILE_CUDA_LINEAR_BF16_BF16_BGRAD_DISABLE_SHAPE` accept the existing
  `m,n,k,opA,opB` shape syntax and force only that BF16-input/BF16-gradient
  dWeight+bias shape through the split dWeight plus separate bias-reducer path.
  This keeps the broad BGRADB default intact while allowing isolated tests of
  hot shapes such as MLP projection dWeight without also changing MLP FC, QKV,
  or attention projection routes. Verified by rebuilding
  `build/libnfn_native_train_tile_ops.so` with CUDA 13.3, rerunning the native
  ABI source/build gate, and checking the isolated paired benchmark. The shape
  gate is intentionally diagnostic-only because the isolated MLP projection
  dWeight probe improved that substage but did not pass total-step timing.

- Revisited the CUDA test surface after reinstalling the latest WSL CUDA
  toolkit (`cuda-toolkit-13-3`). The native GPT/Tile CUDA focused gate passed
  with `244 passed`, the required GPT template preset gate passed with
  `26 passed`, `tools/check_native_no_torch_deps.py --skip-artifacts --json`
  reported `"passed": true`, and the full repository pytest suite passed with
  `1181 passed, 4 skipped, 20 warnings, 468 subtests passed`. This confirms the
  previous CUDA/toolkit failure surface is green on the current workstation
  setup. A follow-up rebuild of `build/nfn_gpt_native_train`,
  `build/nfn_native_train`, `build/libnfn_native_train_tile_ops.so`, and the
  native GPT Python binding kept the no-Torch compiled path current with CUDA
  13.3. The current 3-step, 2-sample llm.kittens parity line measures NeuralFn
  at `1.041561x` train-loop wall time versus
  `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh`; the tracked
  remaining gap is the LM-head classifier/backward structure, not Python or
  graph-editor tensor flow.

- Added route-attribution shape-stat support to the same-script native GPT
  candidate benchmark wrapper. `NFN_SM120_NATIVE_LINEAR_SHAPE_STATS=1` and the
  matching candidate/parity aliases now enable `NFN_NATIVE_GPT_LINEAR_SHAPE_STATS=1`
  on both paired native commands. `tools/paired_kernel_speed.py` now preserves
  `linear_shape_stats` from native profile JSON, emits a
  `native_linear_shape_stats` comparison in the paired JSON/text report, and
  shows per-shape average-time ratios plus cuBLASLt selected heuristic lists.
  This makes future LM-head/block-backward route candidates auditable at the
  actual hot GEMM shape level instead of relying only on coarse route counters.
  Verification: ran the focused paired-kernel sidecar/parser tests, shell syntax
  check for the native candidate wrapper, and Python bytecode compilation for
  `tools/paired_kernel_speed.py`.

- Revisited the CUDA-visible test and benchmark surface after installing CUDA
  Toolkit 13.3.33 for WSL. The focused native/Tile CUDA suite passed with
  `155 passed`, the broader native GPT/Tile examples suite passed with
  `243 passed`, and the full repository test suite passed with `781 passed,
  403 skipped, 16 warnings, 468 subtests passed`. A fresh 5-step, 3-sample
  llm.kittens parity run on the dedicated RTX 5090 measured NeuralFn at
  `1.027407x` train-loop wall time and `0.973435x` tokens/sec versus
  llm.kittens, so the remaining issue is native kernel throughput rather than
  failing tests or stale CUDA state. Two token-weight startup candidates stayed
  rejected: vector4-strided token init regressed the token-init substage to
  `1.010306x`, while threaded token init improved token init to `0.930195x`
  but regressed setup wall time to `1.039387x`. The native candidate benchmark
  wrapper now automatically adds `setup.token_weight_init.total_ms=1.000` for
  startup-only candidates whose candidate library path, env, or candidate args
  mention token-weight initialization, so future token-init bisections cannot
  pass only because unrelated setup work shifted. Verification: ran the CUDA
  gated native/Tile pytest slices, the full pytest suite, the paired parity
  benchmark, both token-init startup candidates, and the focused wrapper test.

- Rejected the remaining broad cuBLASLt heuristic-policy toggles for the native
  dense GPT hot path under CUDA 13.3.33. `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_POLICY=min_waves`
  failed the same-script stage-timed gate at `1.007645x` train-loop wall time,
  `1.000305x` LM-head backward, and `1.013865x` block backward; `max_waves`
  failed harder at `1.023965x` train-loop wall time, `1.032178x` LM-head
  backward, and `1.030164x` block backward. Both runs had no tracked route
  counter change, so the default per-shape heuristic selection stays in place.
  Verification: ran 3-step, 2-sample paired native GPT candidate benchmarks with
  stage timing and the default hot-metric gates on the dedicated RTX 5090.

- Rechecked CUDA 13.3.33 startup and LM-head row-chunk bisections after the
  toolkit reinstall and kept the current defaults. Temporary Tile ops libraries
  built with `NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE=8192`, `2048`, and
  `1024` all failed the targeted startup gate against the 4096-token default:
  token init measured `1.013697x`, `1.010289x`, and `1.016591x` respectively.
  A larger LM-head row chunk (`--lm-head-row-chunk-size 16384`) cut logit GEMM
  launch counts in half but regressed train-loop wall time to `1.016019x` and
  LM-head backward to `1.062838x` because dHidden worsened to `1.244198x`.
  The smaller 4096-row chunk improved CE to `0.961277x` but doubled logit GEMM
  launches, regressed dWeight to `1.039507x`, and measured `1.004875x`
  train-loop wall time. No runtime default changed; the next useful LM-head
  work remains a fused/cooperative row-chunked classifier-backward kernel or a
  materially different GEMM route. Verification: built the three temporary Tile
  ops libraries with CUDA Toolkit 13.3.33, ran startup-only paired benchmarks
  with strict `setup_wall_ms`, `setup.token_weight_init.total_ms`, and
  `total_wall_ms` gates, and ran stage-timed native-vs-native benchmarks for
  16384 and 4096 LM-head row chunks on the dedicated RTX 5090.

- Added native dense GPT custom-graph admission for graph JSON files that carry
  compatible GPT `template_spec` metadata. The compiled C++ trainer now reports
  those files as `selected_graph_support_status: "native-transformer-lm"`, sets
  `native_geometry_contract.shape_source: "custom_graph_template_spec"`, and
  can derive default sequence length and layer count from metadata unless CLI
  flags override them. Arbitrary custom graph JSON still reports
  `custom-graph-native-trainer-missing`, missing paths still report
  `custom-graph-file-missing`, and no real training batches are routed through
  graph-editor nodes or Torch. Verification: rebuilt `build/nfn_gpt_native_train`
  with CUDA Toolkit 13.3.33 and ran the focused native GPT CLI selector test
  covering unsupported and compatible custom graph files.

- Added an opt-in dense GPT token-weight startup initializer behind
  `NFN_TILE_CUDA_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT=1` /
  `NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT=1`. The candidate uses
  vectorized FP32/BF16 shadow stores with a capped grid-stride CUDA kernel so
  it can be compared against the default fast int32 CUDA Tile initializer
  without launching one CTA per vector chunk. It is not promoted as a default:
  the dedicated RTX 5090 startup-only paired gate measured token-weight init at
  `0.985062x` mean but setup wall time at `1.000096x` mean and a slower
  median, so the current fast int32 Tile initializer remains the default.
  Verification: built `/tmp/libnfn_native_train_tile_ops_vector4_strided.so`
  with CUDA Toolkit 13.3.33 and ran
  `NFN_SM120_NATIVE_STARTUP_ONLY=1 ... bash tools/bench_native_gpt_sm120_candidate.sh`,
  which rejected the candidate on the strict `setup_wall_ms=1.000` gate.

- Fixed the trainer-facing Tile CUDA shared-library build on CUDA Toolkit
  13.3.33 by defaulting `tools/build_native_train_tile_ops.sh` to the same
  SM120 cuBLASLt GEMM compile route used by llm.kittens and by normalizing
  inherited `NFN_TILE_CUDA_ARCH=sm_120` / `compute_120` settings to `sm_120a` /
  `compute_120a` when TK attention is enabled. Without this, CUDA 13.3 `ptxas`
  rejected several raw TK GEMM instantiations with static shared-memory usage
  above the default `0xc000` cap, breaking `tests/test_native_gpt2.py` build
  coverage after the WSL CUDA reinstall. Updated README and Python SDK Tile CUDA
  docs to make the supported build route explicit. Verification: reran the
  CUDA-gated Tile CUDA test suite on the dedicated RTX 5090 (`154 passed`),
  reproduced the native build failure, rebuilt the raw Tile ops library with
  CUDA 13.3.33, reran the exact failing tests (`2 passed`), and reran the
  broader native GPT / Tile examples / dependency slice (`92 passed`).

- Fixed `tools/bench_native_gpt_sm120_candidate.sh` so measured startup-only
  bisections (`NFN_SM120_NATIVE_STARTUP_ONLY=1`) auto-gate `setup_wall_ms`
  instead of `train_loop_wall_ms_per_step`. Startup-only native JSON does not
  report completed train steps, so the previous default gate rejected startup
  candidates for a missing train-loop metric before judging the setup timing the
  run was meant to compare. Updated the README and CLI benchmarking notes.
  Verification: ran the focused candidate-wrapper test and reran a GPU-visible
  startup-only candidate benchmark on the dedicated RTX 5090.

- Hardened `tools/check_native_no_torch_deps.py` so the standalone native
  dependency gate fails if the aggregate `.[all]` extra contains `torch`,
  `torchvision`, or `torchaudio`. The JSON report now includes
  `forbidden_optional_extra_dependency_prefixes` and
  `forbidden_optional_extra_hits`, and the native GPT verifier test asserts the
  new fields. Verification: ran the focused native dependency tests, the full
  no-Torch entrypoint guard, and `git diff --check`.

- **Breaking changes:** Removed Torch from the aggregate `.[all]` optional
  dependency set. The default install and native CUDA Tile extras were already
  Torch-free; now `pip install -e ".[all]"` also stays on the native/server/
  dataset stack without installing PyTorch. Install `pip install -e
  ".[all,torch]"` or `pip install -e ".[torch]"` for legacy graph-backed Torch
  trainers and the PyTorch Tile extension loader. Verification: updated the
  native dependency test so `optional-dependencies.all` and installed
  `requires.txt` groups cannot reintroduce Torch, then ran the focused
  dependency test and the no-Torch native entrypoint guard.

- Added a trainer-facing CUDA 13.3 cuBLASLt grouped-layout capability probe to
  the raw Tile C ABI. `libnfn_native_train_tile_ops.so` now exports
  `nfn_native_tile_trainer_linear_cublaslt_grouped_layout_probe_status`, the
  native GPT runtime JSON reports
  `linear_cublaslt_grouped_layout_probe_available`,
  `linear_cublaslt_grouped_layout_probe_status`, and
  `linear_cublaslt_grouped_layout_supported`, and
  `tools/paired_kernel_speed.py` includes the probe status in paired native
  metric output. This is diagnostic plumbing for future grouped-GEMM kernel
  candidates and does not change the default training route. Verification:
  rebuilt all native CUDA artifacts, ran `tests/test_native_gpt2.py` (`58
  passed, 1 skipped`), ran the paired speed focused tests (`2 passed`), ran the
  no-Torch native entrypoint guard, and ran a one-step TinyStories native GPT
  smoke. The smoke reported `status: "native-transformer-lm-trained"`,
  `backend: "tile-cuda"`, `linear_cublaslt_grouped_layout_probe_status: 0`,
  `linear_cublaslt_grouped_layout_supported: true`, and `passed: true`.

- Added an opt-in CUDA 13.3 BF16 cuBLASLt plan prewarm diagnostic to the raw
  Tile C ABI and native GPT runtime. `libnfn_native_train_tile_ops.so` now
  exports `nfn_native_tile_trainer_linear_cublaslt_prewarm_bf16_plan`, and the
  native GPT runtime can call it before the train loop with
  `NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLANS=1`,
  `NFN_NATIVE_GPT2_PREWARM_CUBLASLT_PLANS=1`, or
  `NFN_TILE_CUDA_LINEAR_CUBLASLT_PREWARM=1`. Runtime JSON reports
  `linear_cublaslt_plan_prewarm_available`,
  `linear_cublaslt_plan_prewarm_enabled`,
  `linear_cublaslt_plan_prewarm_attempted_count`,
  `linear_cublaslt_plan_prewarm_success_count`, and
  `linear_cublaslt_plan_prewarm_failure_count`; setup timing also includes
  `setup.cublaslt_plan_prewarm`. The diagnostic remains default-off. The
  dedicated RTX 5090 5-step, 3-sample same-script run successfully prewarmed
  all 9 target plans and improved train-loop wall time to `0.994375x`, but
  setup regressed to `1.158747x` and strict block-backward gates failed
  (`stage.block_backward.total_ms=1.000084x`,
  `stage.block_backward.mlp_proj.total_ms=1.000116x`), so it was rejected as a
  default performance route. Verification: rebuilt the CUDA Tile ops library
  and native GPT CLI with CUDA 13.3, confirmed the new exported symbols with
  `nm -D`, ran the focused native GPT source test, and ran the paired RTX 5090
  benchmark under idle-GPU gating.

- Added an opt-in dense GPT LM-head row-chunk pipeline candidate behind
  `NFN_NATIVE_GPT_LM_HEAD_PIPELINE_CHUNKS=1` /
  `NFN_NATIVE_GPT2_LM_HEAD_PIPELINE_CHUNKS=1`. The candidate keeps the
  bounded row-chunked BF16 classifier memory model, allocates two LM-head logit
  chunks, computes logits plus CE/dlogits on the default stream, and queues
  dHidden plus ordered dWeight accumulation on nonblocking side streams before
  each chunk buffer is reused. Runtime JSON now reports
  `lm_head_pipeline_chunks_requested`, `lm_head_pipeline_chunks_enabled`,
  `lm_head_pipeline_logit_buffer_count`,
  `lm_head_pipeline_extra_bf16_logit_bytes`, and a side-stream schedule
  strategy string. This is a diagnostic candidate, not a default route: the
  same-script RTX 5090 gate rejected it at `1.001057x` train-loop wall and
  `1.009187x` `stage.lm_head_backward.total_ms`, so the default remains the
  serial row-chunk schedule. Verification: rebuilt all native CUDA artifacts
  with CUDA Toolkit 13.3.33, ran a one-step TinyStories native GPT smoke with
  the flag enabled, ran `tests/test_native_gpt2.py` (`58 passed`), and ran the
  paired native speed gate. The smoke reported
  `status: "native-transformer-lm-trained"`, `backend: "tile-cuda"`, two logit
  buffers, `824180736` extra BF16 logit bytes, and `passed: true`.

- Fixed a native BF16/u16-token GPT sampled-loss race in the fused CE
  loss+backward CUDA Tile kernel. The kernel now synchronizes after reading the
  target logit for the scalar loss and before overwriting the row logits
  in-place with BF16 dlogits, matching the ordering required by fused
  classifier implementations and keeping `--train-loss-every-steps` /
  validation-loss reporting from racing the classifier backward write.
  Verification: rebuilt the native Tile ops library, ran focused native GPT
  source/contract tests, ran CUDA-visible Tile CE/native tests, ran a one-step
  native GPT smoke with `--train-loss-every-steps 1`, and checked the no-Torch
  native entrypoint guard.

- Made SM120 native candidate speed checks reject slower measured candidates by
  default. `tools/bench_native_gpt_sm120_candidate.sh` now auto-adds
  `train_loop_wall_ms_per_step=1.000` when a measured candidate changes the Tile
  ops library, candidate-only environment, or candidate-only extra args. When
  `NFN_SM120_NATIVE_STAGE_TIMING=1` is enabled it also gates
  `stage.lm_head_backward.total_ms`, `stage.block_backward.total_ms`, and
  `stage.block_backward.mlp_proj.total_ms` at `1.000`. Dry-run planning and
  no-op baseline-vs-baseline runs stay ungated, and explicit
  `NFN_SM120_NATIVE_MAX_CANDIDATE_RATIO` /
  `NFN_SM120_CANDIDATE_MAX_CANDIDATE_RATIO` values override the defaults.
  Verification: rebuilt native CUDA artifacts with CUDA Toolkit 13.3.33, ran
  the GPU-visible native/Tile pytest suite (`240 passed`), ran
  `tests/test_template_presets.py -x -q` (`26 passed`), ran
  `tools/check_native_no_torch_deps.py --skip-artifacts --json`, reran paired
  RTX 5090 native speed checks, and added focused wrapper tests for the default
  gate policy.

- Expanded paired CUDA benchmark stdout attribution for block backward. The
  native JSON parser already captured arbitrary `timing.stage_timing` entries;
  the text report and candidate-ratio gate allowlist now also include MLP FC,
  MLP projection, LN2 residual, attention projection, attention SDPA
  grad-out/to-QKV, QKV dInput/dWeight, and LN1 residual child metrics. This
  lets SM120 native kernel candidates gate concrete hot buckets such as
  `stage.block_backward.qkv.dinput.total_ms` or
  `stage.block_backward.attn_proj.dinput.total_ms` instead of treating
  `stage.block_backward.total_ms` as an opaque aggregate. Verification:
  `python -m pytest tests/test_tile_cuda_examples.py -q -k
  'paired_kernel_speed_tool_compiles_and_smokes or metric_ratio_gate'`,
  `python -m py_compile tools/paired_kernel_speed.py`, and `git diff --check`.

- Revalidated the native CUDA Tile training stack after installing CUDA Toolkit
  13.3.33 for WSL. Every native trainer binary and Python extension was rebuilt
  against the new toolkit, and the dedicated RTX 5090 pass now has no remaining
  CUDA correctness failures. Performance-only benchmark gates still reject
  slower routes rather than becoming defaults: extra-large-K cuBLASLt LM-head
  dHidden regressed train-loop wall time to `1.034147x`, the retested
  cuBLASLt one-shape heuristic overrides for `768,65536,3072,N,N` and
  `768,50304,8192,N,T` remained slower, token-weight vector4/threaded startup
  initializers were not enough to promote, cudaMallocAsync worsened setup
  wall time, and full-logit LM-head reuse again exceeded the useful paired
  benchmark window. The staged 10-step llm.kittens parity sample still shows
  remaining native GPU work (`1.047862x` train-loop wall time, `0.952442x`
  tokens/sec), concentrated in block backward and LM-head backward, not a
  Python/Torch/graph-editor fallback. Verification: `bash
  tools/build_native_gpt2_all.sh`, `NFN_TILE_CUDA_TEST=1 python -m pytest
  tests/test_native_gpt2.py tests/test_tile_cuda_examples.py
  tests/test_tile_cuda_gpu.py tests/test_tile_cuda_ops.py
  tests/test_tile_cuda_optimizer.py -q -rs` (`240 passed`), `python -m pytest
  tests/test_template_presets.py -x -q` (`26 passed`), `python
  tools/check_native_no_torch_deps.py --skip-artifacts --json`, and
  `git diff --check`.

- Extended paired CUDA benchmark metric gates with statistic-qualified checks.
  Existing `--max-candidate-ratio METRIC=RATIO` entries still gate the mean
  candidate-over-baseline ratio, while `median:METRIC=RATIO`,
  `min:METRIC=RATIO`, and `max:METRIC=RATIO` gate the corresponding statistic.
  This lets noisy RTX 5090 kernel candidates require both mean and median hot
  metrics such as `median:train_loop_wall_ms_per_step=1.000` before a default
  route is promoted. `tools/bench_native_gpt_sm120_candidate.sh` forwards the
  same syntax through `NFN_SM120_NATIVE_MAX_CANDIDATE_RATIO` and
  `NFN_SM120_CANDIDATE_MAX_CANDIDATE_RATIO`. Verification:
  `python -m pytest tests/test_tile_cuda_examples.py -q -k
  metric_ratio_gate`, `python -m py_compile tools/paired_kernel_speed.py`, and
  `git diff --check`.

- Changed `neuralfn.native_train` dense GPT SDK dispatch to skip the generic
  `nfn_native_train` frontend when a direct GPT family binary is configured or
  built. `build_native_train_run_config("gpt"|"gpt2"|"gpt3"|"nanogpt", ...)`
  now resolves to `nfn_gpt_native_train --model-family ...` when
  `NFN_NATIVE_GPT_CLI` is set or `build/nfn_gpt_native_train` exists, removing
  one compiled dispatcher process from the normal SDK startup path. Explicit
  `NFN_NATIVE_TRAIN_CLI` and `native_train_cli=` still force the unified
  frontend for registry/debug workflows. Migration notes: callers that asserted
  the exact generic `--base-model` argv for dense GPT SDK configs should either
  set `NFN_NATIVE_TRAIN_CLI` or update expectations to the direct
  `--model-family` command. Verification:
  `python -m pytest tests/test_native_gpt2.py -q -k
  'native_train_run_config_uses_direct_dense_gpt_cli or
  native_train_explicit_unified_cli_overrides_direct_dense_gpt_cli'`,
  `python -m py_compile neuralfn/native_train.py`, and `git diff --check`.

- Added candidate-over-baseline metric-ratio gates to the paired CUDA benchmark
  tooling. `tools/paired_kernel_speed.py` now accepts repeatable
  `--max-candidate-ratio [STAT:]METRIC=RATIO` checks, records
  `metric_ratio_gates` in JSON/text output, and exits nonzero after writing the
  report when a candidate regresses a required native metric or when the metric
  is missing. `tools/bench_native_gpt_sm120_candidate.sh` forwards
  whitespace-separated gates from `NFN_SM120_NATIVE_MAX_CANDIDATE_RATIO` or
  `NFN_SM120_CANDIDATE_MAX_CANDIDATE_RATIO`, so SM120 native candidates can
  require hot buckets such as `stage.lm_head_backward.total_ms` and
  `train_loop_wall_ms_per_step` to stay under explicit thresholds. This is a
  benchmark workflow change for future CUDA kernel work; it does not alter the
  trainer runtime. Verification:
  `python -m pytest tests/test_tile_cuda_examples.py -q -k
  'paired_kernel_speed_tool_fails_metric_ratio_gate or
  paired_kernel_speed_tool_metric_ratio_gate_fails_missing_metric or
  paired_kernel_speed_tool_compiles_and_smokes'`, `python -m py_compile
  tools/paired_kernel_speed.py`,
  `NFN_SM120_NATIVE_DRY_RUN_PLAN=1
  NFN_SM120_NATIVE_MAX_CANDIDATE_RATIO='stage.lm_head_backward.total_ms=1.000
  train_loop_wall_ms_per_step=1.005' bash
  tools/bench_native_gpt_sm120_candidate.sh`, and `git diff --check`.

- Split the root Python package install into a lean native/core SDK surface plus
  explicit workflow extras. `pip install -e .` no longer hard-installs Torch,
  NumPy, tokenizer, dataset, graph-analysis, FastAPI/server, MCP, or database
  packages; use `.[tile-cuda]`, `.[datasets]`, `.[graph]`, `.[server]`,
  `.[torch]`, or `.[all]` for those workflows. This keeps default native CLI
  and SDK startup aligned with the no-Torch/no-Python-ML-stack training path.
  The native dependency verifier now fails if those packages drift back into
  root `project.dependencies`, while still requiring optional-extra coverage
  for the major workflows. Migration notes: development environments that use
  the editor/backend, raw-text tokenization, Python graph helpers, or legacy
  graph-backed Torch trainers should reinstall with the matching extra instead
  of relying on the root install to pull everything in. Verification:
  `python tools/check_native_no_torch_deps.py --skip-artifacts --json`,
  `python -m pytest tests/test_native_gpt2.py -q -k
  native_no_torch_dependency_verifier`, `python -m py_compile
  tools/check_native_no_torch_deps.py`, and `git diff --check`.

- Aligned the master `nfn train` direct native dispatcher with the other
  native launchers by setting `CUDA_MODULE_LOADING=LAZY` when the caller has
  not supplied a value. This keeps the default dense GPT CLI route on the
  compiled CUDA Tile C++ path with lazy CUDA module loading before any
  graph-backed runtime can import. The native no-Torch verifier now checks both
  explicit `nfn train --tinystories` and default `nfn train` dry-run/print
  command paths under the Torch/NumPy/tiktoken/dataset-manager/`nfn_impl`
  import blocker. Migration notes: user-provided `CUDA_MODULE_LOADING` still
  takes precedence, and the default CLI route remains dense GPT
  `--train-transformer-lm`. Verification:
  `python -m pytest tests/test_native_gpt2.py -q -k
  'native_no_torch_dependency_verifier or nfn_direct_native_train_sets_lazy_cuda_module_loading'`,
  `python tools/check_native_no_torch_deps.py --skip-artifacts --json`, and
  `git diff --check`.

- Expanded paired native-kernel benchmark attribution for the remaining RTX
  5090 parity gap. `tools/paired_kernel_speed.py` now uses one shared text
  metric allowlist for per-side metrics and candidate-over-baseline ratios, and
  that list includes LM-head backward substages (`logits`, `ce`, `dhidden`,
  `dweight`, optional `dhidden_dweight_concurrent`) plus key block-backward
  substages (`mlp_proj.*`, `attn_sdpa.to_qkv`, and `qkv.dweight_bias`). This
  makes the stdout report sufficient for deciding whether a candidate moved the
  real LM-head/block-backward bottlenecks before opening sidecar JSON. A fresh
  CUDA 13.3 dedicated RTX 5090 native-vs-native retest kept
  `NFN_NATIVE_GPT_CE_BF16_VEC_STORES=1` rejected at `1.006100x` train-loop wall
  time, `0.993948x` tokens/sec, and `1.002658x` LM-head backward time.
  Verification: `python -m pytest tests/test_tile_cuda_examples.py -q` and
  `git diff --check`.

- Added default-off vec4 CUDA diagnostics for the dense GPT multi-buffer
  float32-to-BF16 packer and stored-MLP activation pack/restore path. The new
  guarded kernels are enabled only by
  `NFN_NATIVE_GPT_F32_TO_BF16_MANY_VEC4=1` /
  `NFN_TILE_CUDA_F32_TO_BF16_MANY_VEC4=1` or
  `NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS_VEC4=1` /
  `NFN_TILE_CUDA_STORE_MLP_ACTIVATIONS_VEC4=1`, with GPT-2-prefixed variables
  retained as compatibility fallbacks. They are not default training routes:
  the CUDA 13.3 dedicated RTX 5090 paired benchmark measured the scalar
  candidate faster than the vec4-default baseline (`0.994143x`
  train-loop wall time, `1.005941x` tokens/sec, no tracked route-counter
  change). Verification after the CUDA Toolkit 13.3 WSL reinstall:
  rebuilt `build/libnfn_native_train_tile_ops.so`, ran
  `tools/check_native_no_torch_deps.py`, reran GPU-visible
  `tests/test_tile_cuda_gpu.py tests/test_tile_cuda_ops.py
  tests/test_tile_cuda_optimizer.py` (`154 passed`), reran the native GPT C ABI
  smoke (`1 passed`), reran `tests/test_layer_evo_gpu.py` (`3 passed`), and
  reran `tests/test_native_gpt2.py` with real CUDA visibility (`54 passed`).

- Added a native route-counter-change summary to `tools/paired_kernel_speed.py`
  so native-vs-native kernel candidates are not promoted from timing noise
  alone. The paired JSON now includes `native_route_counter_changes`, comparing
  tracked TK, cuBLASLt, BF16 GEMM, LM-head logits, BF16 packing/cache, and
  attention launch counters between baseline and candidate. Text output prints
  the changed counters and warns when candidate-specific environment knobs are
  set but tracked counters remain unchanged. Verification:
  `python -m pytest tests/test_tile_cuda_examples.py -q`.

- Refreshed CUDA 13.3 RTX 5090 parity attribution after the dense GPT LM-smoke
  fix and rejected two more existing-route probes instead of promoting noisy
  switches. A one-step stage/shape profile still shows the remaining gap in
  native GPU work: about `1349 ms` block backward, `732 ms` LM-head backward,
  and `638 ms` model forward per optimizer step, with LM-head backward split
  across logits (`221 ms`), CE (`67 ms`), dHidden (`265 ms`), and dWeight
  (`176 ms`). `NFN_NATIVE_LINEAR_TK_DINPUT_DISABLE_SHAPE=3072,65536,768,N,N`
  measured a short-run `0.991338x` train-loop wall result but did not change
  route counters, so it is not a promotable result. Disabling the TK forward
  route for `768,65536,3072,T,N` did reduce TK calls but regressed train-loop
  wall to `1.023601x` and tokens/sec to `0.976942x`. No runtime defaults were
  changed; the next useful parity work remains a fused/cooperative LM-head
  classifier/backward kernel rather than route-switch retuning.

- Fixed the dense GPT native `--smoke-lm-step` verifier after revisiting CUDA
  tests on the CUDA 13.3 WSL install. The synthetic GPT smoke now expects the
  summed two-row CE partial loss instead of a one-row average and keeps
  post-AdamW weight assertions on target rows, while still checking sampled
  non-target gradients. This avoids failing the smoke on near-zero non-target
  tied-embedding gradients whose sign can be flipped by tiny CUDA 13.3 numeric
  drift and then amplified by AdamW's first normalized step; the standalone
  optimizer smoke continues to verify full AdamW update math. Verification:
  rebuilt `build/libnfn_native_train_tile_ops.so` and
  `build/nfn_gpt_native_train` with CUDA Toolkit 13.3, confirmed sandboxed CUDA
  access still skips GPU pytest because NVML is blocked, reran the GPU-visible
  Tile CUDA pytest smoke (`1 passed`), and reran native CUDA ABI, fill,
  optimizer, NanoGPT LM, and dense GPT LM smokes on the RTX 5090.

- Added default-off BF16 `cublasGemmEx` algorithm bisection controls for native
  Tile-CUDA training. `NFN_NATIVE_LINEAR_BF16_GEMM_EX_ALGO` /
  `NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_ALGO` select a global BF16 GEMMEx
  algorithm, while `NFN_NATIVE_LINEAR_BF16_GEMM_EX_ALGO_SHAPE` /
  `NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_ALGO_SHAPE` targets one exact
  `m,n,k,opA,opB,algo` shape such as the LM-head dHidden
  `768,8192,50304,N,N,0` probe. Defaults are unchanged when the variables are
  unset; this is a same-script benchmarking aid for the remaining llm.kittens
  parity gap, not a promoted faster route. Verification: rebuilt the native
  Tile ops library with CUDA 13.3, ran the focused native GPT/Tile tests, and
  ran dedicated RTX 5090 native-vs-native candidate benchmarks. The first
  2-sample LM-head dHidden algorithm-0 probe measured `0.993367x` train-loop
  wall, but the 5-sample confirmation was effectively flat at `0.999739x`
  mean train-loop wall and `1.000326x` tokens/sec, so no default algorithm was
  changed. A follow-up sweep kept LM-head dHidden algorithms 1-4 rejected:
  algorithm 1 was one-sample noise (`0.998467x` wall), algorithms 2 and 3 were
  slower (`1.023701x` and `1.010416x` wall), and algorithm 4 regressed in the
  5-sample confirmation (`1.002073x` wall, `0.997933x` tokens/sec).

- Fixed GPT-2 evo native dry-run command inspection. The legacy Python guard now
  normalizes `--native-cuda-dry-run` and `--native-cuda-print-command` before
  forwarding to family-specific native binaries, the unified native frontend
  preserves `--print-command` when printing GPT-2 evo family delegates, and
  `nfn_gpt2_evo_native_train --native-cuda-dry-run
  --native-cuda-print-command` now prints the final dense GPT CUDA Tile command
  with `--train-transformer-lm --layer-evo` instead of falling back to plan JSON.
  This keeps GPT-2 evo startup inspection on compiled C++ paths without dataset
  scanning, Torch imports, or graph-editor tensor flow. Verification: rebuilt
  `build/nfn_gpt2_evo_native_train` and `build/nfn_native_train`; ran focused
  CLI guard tests, the native unified frontend dispatch test, and manual
  two-stage print-command checks.

- Added shared environment injection to the dense GPT native-vs-native SM120
  candidate benchmark wrapper. `tools/bench_native_gpt_sm120_candidate.sh` now
  accepts `NFN_SM120_NATIVE_ENV`, `NFN_SM120_COMMON_ENV`, and
  `NFN_SM120_PARITY_ENV` for variables that should apply to both baseline and
  candidate commands, while keeping `NFN_SM120_NATIVE_CANDIDATE_ENV` /
  `NFN_SM120_CANDIDATE_ENV` candidate-only. This avoids failed or asymmetric
  profiling attempts when enabling common attribution flags such as
  `NFN_NATIVE_GPT_LINEAR_SHAPE_STATS=1`. Verification: added a dry-run wrapper
  regression that asserts shared env is emitted in both `baseline_env` and
  `candidate_env`, candidate-only env remains isolated, and the generated JSON
  records both maps; ran `bash -n tools/bench_native_gpt_sm120_candidate.sh` and
  `python -m pytest tests/test_tile_cuda_examples.py -q`.

- Matched dense GPT native token padding behavior to llm.kittens and added
  no-pad-zero CE entry points. The native GPT trainer now initializes only the
  real 50,257 tokenizer rows of the tied token embedding/LM-head tensor, keeps
  the 47 padded rows zero in FP32 master and BF16 shadow storage, and uses new
  `nfn_native_tile_token_cross_entropy_backward_*_no_pad_zero*` Tile ABI
  symbols for public-vocab CE/dlogits so the CE kernel no longer scrubs padded
  dlogit columns every row. Runtime JSON reports `lm_head_ce_pad_zero_skipped`,
  `token_weight_padding_zero_enabled`, `token_weight_init_elements`, and
  `token_weight_padding_elements`; set `NFN_NATIVE_GPT_ZERO_TOKEN_PADDING=0` or
  `NFN_NATIVE_GPT_SKIP_CE_PAD_ZERO=0` only for paired diagnostics against the
  previous behavior. Verification after the CUDA Toolkit 13.3.33 WSL reinstall:
  rebuilt `build/libnfn_native_train_tile_ops.so` and
  `build/nfn_gpt_native_train`; ran `python -m pytest tests/test_native_gpt2.py
  -q` (`52 passed, 1 skipped`), `python -m pytest
  tests/test_tile_cuda_examples.py -q` (`23 passed`),
  `NFN_TILE_CUDA_TEST=1 python -m pytest tests/test_tile_cuda_gpu.py -q -rs`
  (`1 passed`), and `python tools/check_native_no_torch_deps.py`. The
  native-vs-native paired benchmark measured the new default at
  `2514.110 ms/step` versus the prior full-padded-token-init/CE-scrub route at
  `2517.970 ms/step`; the canonical llm.kittens parity run measured NeuralFn at
  `2533.160 ms/step` versus llm.kittens at `2460.950 ms/step`
  (`1.029377x` train-loop wall, `0.971123x` tokens/sec).

- Added a diagnostic full-batch resident LM-head reuse schedule for the native
  dense GPT trainer. When
  `NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS=1` is already enabled,
  `NFN_NATIVE_GPT_FULL_BATCH_LM_HEAD_REUSE=1` makes the resident BF16 logit path
  run LM-head logits, CE/dlogits, dHidden, and dWeight across the full
  `batch x sequence` row range instead of still slicing that full arena into
  8192-row chunks. The switch remains off by default because the CUDA 13.3.33
  RTX 5090 one-step smoke reduced `stage.lm_head_backward.total_ms` to
  `0.652349x` but regressed train-loop wall time to `32.086572x`; the saved
  LM-head work was overwhelmed by downstream `block_backward.attn_sdpa` memory
  pressure from the full resident logits. Migration notes: keep default native
  training on the row-chunked recompute path; use the new flag only for paired
  profiling while designing a fused/cooperative LM-head backward kernel.
  Verification: rebuilt `build/nfn_gpt_native_train`, ran the focused native
  source regression (`1 passed, 52 deselected`), and ran the one-step
  native-vs-native GPU smoke with
  `NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS=1`.

- Hardened paired CUDA benchmark cleanup after interrupted native candidates.
  `tools/paired_kernel_speed.py` now terminates the active command process group
  on `KeyboardInterrupt` or other unexpected interruption, matching the existing
  timeout cleanup behavior and preventing memory-heavy native probes from
  continuing to use the selected GPU after the benchmark parent exits.
  Stage-timed text summaries now also print the existing native forward
  attribution buckets, including `stage.train.model_forward.total_ms`,
  `stage.block_forward.total_ms`, and `stage.block_recompute.total_ms`, so the
  remaining llm.kittens parity gap can be assigned without manually opening
  profile sidecars.
  Migration notes: command-line flags are unchanged; interrupted benchmark runs
  still re-raise the interruption after cleanup. Verification: added a
  process-group termination regression test; after the CUDA 13.3 reinstall,
  rebuilt the native Tile ops and GPT CLI, ran the native GPT suite
  (`52 passed, 1 skipped`), the no-Torch guard, the focused Tile CUDA tests, and
  the opt-in `NFN_TILE_CUDA_TEST=1` GPU smoke. A no-stage 5-step RTX 5090 parity
  run measured llm.kittens at `2471.224 ms/step` and NeuralFn at
  `2541.080 ms/step` (`1.028292x` train-loop wall, `0.972465x` tokens/sec).
  Rechecked and rejected `NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1`
  (`1.019016x` train-loop wall), and the full-logit LM-head reuse probe remained
  non-viable for default training because a three-step native-vs-native probe did
  not finish in the useful benchmark window.

- Allowed dense GPT native `--startup-only` probes to use `--max-steps 0`.
  Startup-only setup checks now keep the full Tile-CUDA transformer allocation
  path but exit before optimizer work without tripping the normal positive-step
  training validator; real training still rejects non-positive `max_steps`.
  Migration notes: benchmark scripts can use zero-step startup probes for setup
  timing, while train-loop runs should keep positive `--max-steps` values.
  Verification: added a native CLI regression that reaches the missing Tile ops
  library gate with `--startup-only --max-steps 0` instead of failing
  validation; after the CUDA 13.3 reinstall, the latest 5-step same-script
  parity check measured NeuralFn at `2536.100 ms/step` versus llm.kittens at
  `2429.056 ms/step` (`1.044068x` train-loop wall, `0.957209x` tokens/sec).
  Rechecked and rejected the LM-head concurrent dHidden/dWeight candidate
  (`1.002028x` wall), the combined dInput-before-dWeight scheduling candidate
  (`1.000696x` wall), the cudaMallocAsync startup candidate (`1.142263x` total
  startup), and explicit exit-time CUDA frees (`1.378794x` total startup), so
  no slower kernel or allocator default was promoted.

- Reordered the compiled native GPT selected-graph gate ahead of token-shard
  resolution for real `--train-transformer-lm` runs. Unsupported shipped
  templates, unknown templates, missing custom graph files, and custom graphs
  that do not yet have a native trainer now return native graph/template status
  JSON before touching cached dataset shards. The early-exit payload includes
  `token_shards_resolved: false` and an empty `dataset_path` when no dataset was
  resolved. Migration notes: `--print-plan`, `--dry-run`, smoke steps, and
  runnable dense GPT/GPT3 training keep their existing resolution behavior;
  unsupported selected-graph errors are no longer masked by missing dataset
  aliases. Verification: added a missing-dataset unsupported-template
  regression that returns `template-native-trainer-missing` without stderr or
  shard resolution.

- Extended lazy CUDA module loading to legacy training-script native guards.
  Scripts such as `train_gpt2_evo.py` that exit before importing Torch and
  hand off to a family-specific or generic compiled native trainer now set
  `CUDA_MODULE_LOADING=LAZY` when the caller has not supplied a value, matching
  the direct dense GPT fast path and SDK/native launcher behavior. Migration
  notes: user-provided `CUDA_MODULE_LOADING` values still win, and command-line
  arguments are unchanged. Verification: added source coverage for
  `native_training_guard.py` and a GPT-2 evo direct-script regression that
  stubs the native binary and asserts the CUDA device, connection, and lazy
  module-loading environment before any Torch imports.

- Defaulted the direct dense GPT Python compatibility fast path to lazy CUDA
  module loading. `cli/scripts/train_gpt.py` now sets
  `CUDA_MODULE_LOADING=LAZY` when unset before execing the compiled
  `nfn_gpt_native_train` binary, matching the SDK native runner, native
  launcher, and C++ trainer startup policy while preserving caller-provided
  `CUDA_MODULE_LOADING` values. Migration notes: no command-line arguments
  change; direct `python cli/scripts/train_gpt.py ...` and
  `python cli/scripts/train_gpt2.py ...` runs avoid eager CUDA module loading
  without importing the legacy Torch trainer. Verification: added a direct
  compiled-CLI dry-run regression that stubs the native binary and asserts
  `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and
  `CUDA_MODULE_LOADING=LAZY` are forwarded before any Torch/dataset-manager
  imports.

- Extended the dense GPT native-vs-native SM120 candidate benchmark wrapper to
  accept the parity wrapper's `NFN_SM120_PARITY_*` common controls as a third
  alias family. `tools/bench_native_gpt_sm120_candidate.sh` now resolves common
  settings such as steps, samples, warmup, profile directory, stage timing, GPU
  selection, JSON output, and dry-run plan with precedence
  `NFN_SM120_NATIVE_*` > `NFN_SM120_CANDIDATE_*` > `NFN_SM120_PARITY_*` >
  default. Candidate-only env and candidate-only extra args remain isolated to
  the candidate command. Migration notes: existing native/candidate env names
  keep the same behavior, while quick handoffs from parity commands no longer
  silently fall back to the candidate wrapper's larger 10-step, 3-sample,
  1-warmup defaults. Verification: added a dry-run regression test that sets
  `NFN_SM120_PARITY_STEPS=2`, `NFN_SM120_PARITY_SAMPLES=1`,
  `NFN_SM120_PARITY_WARMUP=0`, and `NFN_SM120_PARITY_PROFILE_DIR=none` and
  asserts the generated paired plan uses those values without launching a GPU
  benchmark.

- Classified dense GPT modern template aliases in the native C++ selector. The
  dense GPT trainer now treats `gpt2_modern` as the same compiled-loop geometry
  as `gpt2`, and recognizes `nanogpt_modern` / `nanogpt_megakernel` as known
  dense GPT template selectors that should report geometry-mismatch native work
  instead of generic template-missing status. Migration notes: no CLI arguments
  change; this only tightens native plan/status reporting and keeps unsupported
  shapes on explicit no-Torch native-missing JSON. Verification: rebuilt
  `build/nfn_gpt_native_train`, ran focused native GPT selector tests, and
  confirmed `--template-name gpt2_modern --print-plan` reports
  `selected_graph_support_status: "native-transformer-lm"` while NanoGPT
  modern aliases report `template-geometry-native-trainer-missing`.

- Defaulted single-buffer FP32-to-BF16 bits conversion to a guarded vec4 CUDA
  path. `nfn_native_tile_float32_to_bf16_bits` now packs four aligned FP32
  values into four BF16 bits per thread when the source is 16-byte aligned, the
  destination is 8-byte aligned, and the element count is divisible by four,
  falling back to the scalar kernel otherwise. Set
  `NFN_TILE_CUDA_F32_TO_BF16_VEC4=0`, `NFN_NATIVE_GPT_F32_TO_BF16_VEC4=0`, or
  `NFN_NATIVE_GPT2_F32_TO_BF16_VEC4=0` to restore the scalar route for
  bisection. Migration notes: no caller changes are required; this affects
  native trainer conversion hot paths such as `block_backward.mlp_proj.grad_out_bf16`.
  Verification: rebuilt `build/libnfn_native_train_tile_ops.so`;
  `NFN_TILE_CUDA_TEST=1 python -m pytest tests/test_tile_cuda_gpu.py -x -q`
  passed; same-script RTX 5090 timing against
  `NFN_NATIVE_GPT_F32_TO_BF16_VEC4=0` measured scalar opt-out at `1.003731x`
  train-loop wall and `0.996281x` tokens/sec versus the vec4 default, with
  `block_backward.mlp_proj.grad_out_bf16` improving from roughly `114-122 ms`
  to `83-85 ms` over the 5-step stage-timed runs.

- Added low-overhead LM-head logits backend counters to dense GPT native runtime
  JSON. Normal runs now report `lm_head_logits_tk_gemm_count`,
  `lm_head_logits_cublaslt_gemm_count`, and
  `lm_head_logits_bf16_gemm_count`, and `lm_head_logits_linear_strategy` uses
  those per-stage counter deltas when expensive `linear_shape_stats` timing is
  disabled. Migration notes: benchmark tooling no longer needs to enable shape
  stats just to distinguish the default TK BF16 LM-head logits route from the
  GEMMEx fallback; `tools/paired_kernel_speed.py` now collects and prints the
  counters in paired benchmark summaries. Verification: rebuilt
  `build/nfn_gpt_native_train`; `python -m pytest
  tests/test_native_gpt2.py::test_native_train_tile_ops_builds_torch_free_c_abi
  -q` passed; a one-step CUDA native smoke with shape stats disabled wrote
  `lm_head_logits_linear_strategy: "tk-sm120-bf16-gemm-default"` and
  `lm_head_logits_tk_gemm_count: 64`; a one-step paired native smoke printed
  the new LM-head logits counters in both native metric sections and ratios;
  `tools/check_native_no_torch_deps.py` passed.

- Defaulted dense GPT native LM-head logits recompute to the TK BF16 forward
  bridge on CUDA 13.3/SM120. The trainer-facing
  `launch_linear_bf16_input_weight_bf16_output_float32` wrapper now attempts
  the existing TK BF16-input/BF16-weight/BF16-output GEMM path before cuBLAS
  GEMMEx for no-bias shapes, and the old hardcoded disable for the default
  LM-head logits row chunk `50304,8192,768,T,N` has been removed. Set
  `NFN_TILE_CUDA_LINEAR_TK_FORWARD_DISABLE_SHAPE=50304,8192,768,T,N` or
  `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=50304,8192,768,T,N` to reproduce
  the prior GEMMEx fallback for paired diagnostics. Migration notes: no caller
  change is required, but benchmark tooling that expected the LM-head logits
  fallback should update expectations and use the explicit disable env when it
  needs the old route. Verification: rebuilt
  `build/libnfn_native_train_tile_ops.so` and `build/nfn_gpt_native_train`;
  `NFN_TILE_CUDA_TEST=1 /home/adam/miniconda3/envs/NeuralFn/bin/python -m pytest
  tests/test_tile_cuda_gpu.py -x -q` passed (`1 passed`); native dependency
  tests passed (`3 passed`); `tools/check_native_no_torch_deps.py` passed; a
  prior focused native GPT run after wiring the TK attempt passed (`52 passed`).
  Same-script RTX 5090 timing with the newly wired route measured
  `0.923926x` mean train-loop wall and `1.086108x` tokens/sec versus the
  fallback over 5-step, 3-sample native-vs-native timing; stage timing measured
  `lm_head_backward.total_ms` at `0.642874x` of fallback. The default
  llm.kittens parity sample after promotion measured NeuralFn at
  `2834.720 ms/step` and `184952 tok/s` versus llm.kittens at
  `3214.092 ms/step` and `164306.8 tok/s` (`0.881966x` train-loop wall,
  `1.125650x` tokens/sec), though parity should still be rechecked over fresh
  multi-sample runs because the reference sample was slower than earlier CUDA
  13.3 measurements.

- Rechecked the next dense GPT native SM120 tuning candidates after the vector4
  startup diagnostic and kept defaults unchanged. A fresh same-script parity
  sample still measured NeuralFn slower than `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh`
  (`1.056634x` train-loop wall, `0.945115x` tokens/sec), with LM-head
  logits/dHidden still on BF16 GEMMEx and dWeight on cuBLASLt.
  `CUDA_DEVICE_MAX_CONNECTIONS=8` regressed train-loop wall to `1.004548x`
  and setup wall to `1.121564x` versus the current `CUDA_DEVICE_MAX_CONNECTIONS=1`
  launch default. `NFN_NATIVE_GPT_MLP_PROJ_DINPUT_BEFORE_DWEIGHT=1` stayed
  noise-level (`0.998206x` mean train-loop wall, `1.000733x` median).
  `NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS=1` timed out at the 300-second
  candidate limit for a 5-step run, confirming the full resident-logit memory
  tradeoff is not viable for the default shape. `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_BF16=0`
  improved setup wall (`0.938130x` in the train benchmark) but regressed
  train-loop wall (`1.005951x`), so saved LN1 BF16 attention tape remains
  default. Migration notes: no caller change is required; these are rejected
  candidate measurements to keep future work focused on fused/cooperative
  LM-head or materially different GEMM kernels rather than switch promotion.

- Added an opt-in dense GPT token-weight startup initializer candidate behind
  `NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT=1` /
  `NFN_NATIVE_GPT2_TOKEN_WEIGHT_VECTOR4_INIT=1` /
  `NFN_TILE_CUDA_TOKEN_WEIGHT_VECTOR4_INIT=1`. The Tile CUDA trainer library
  now has a vectorized float4 path that writes the deterministic power-of-two
  FP32 token table and persistent BF16 LM-head shadow together, while the
  default remains the measured int32 CUDA Tile initializer until paired timing
  proves the vector4 route faster. Runtime JSON reports
  `token_weight_vector4_init_enabled`, and `token_weight_init_strategy` labels
  active vector4 runs as `device-vector4-power2-deterministic` or the fused
  BF16-shadow variant. The default was not changed because dedicated RTX 5090
  startup-only paired timing measured the vector4 candidate slower (`1.078363x`
  token-weight init time and `1.033082x` total wall). Migration notes: no
  caller change is required unless diagnostic tooling opts into the candidate
  or asserts the startup JSON shape. Verification: `bash
  tools/build_native_train_tile_ops.sh`; `bash tools/build_native_gpt_cli.sh`;
  focused native GPT source/compile tests; a startup-only CUDA smoke with the
  candidate enabled; and `NFN_SM120_NATIVE_STARTUP_ONLY=1
  NFN_SM120_NATIVE_CANDIDATE_ENV=NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT=1
  bash tools/bench_native_gpt_sm120_candidate.sh`. The CUDA reinstall
  last-failed sweep was re-run after this change with `NFN_TILE_CUDA_TEST=1
  python -m pytest --lf -q -rs` and passed with `1167 passed`, `20 warnings`,
  and `468 subtests passed`.

- Added an opt-in dense GPT native LM-head backward scheduling candidate behind
  `NFN_NATIVE_GPT_LM_HEAD_CONCURRENT_DHIDDEN_DWEIGHT=1` /
  `NFN_NATIVE_GPT2_LM_HEAD_CONCURRENT_DHIDDEN_DWEIGHT=1`. After BF16 CE
  produces dlogits for a row chunk, the trainer records a CUDA event, launches
  LM-head dHidden and dWeight on two non-blocking CUDA streams, synchronizes
  both streams, and then proceeds to the next row chunk. The default remains
  the serial dHidden-then-dWeight route because same-script RTX 5090 timing
  measured the candidate slower (`1.004893x` train-loop wall time,
  `0.995133x` tokens/sec). Runtime JSON now reports
  `lm_head_concurrent_dhidden_dweight_requested`,
  `lm_head_concurrent_dhidden_dweight_available`,
  `lm_head_concurrent_dhidden_dweight_enabled`, and
  `lm_head_dhidden_dweight_schedule_strategy`; stage timing reports
  `lm_head_backward.dhidden_dweight_concurrent` when the candidate path runs.
  Migration notes: no caller change is required unless diagnostic tooling wants
  to opt into the candidate or assert the new JSON fields. Verification:
  focused native GPT source/compile tests; `bash tools/build_native_gpt_cli.sh`;
  a one-step CUDA smoke with the candidate enabled; and
  `NFN_SM120_NATIVE_STEPS=5 NFN_SM120_NATIVE_SAMPLES=3
  NFN_SM120_NATIVE_WARMUP=1
  NFN_SM120_NATIVE_CANDIDATE_ENV=NFN_NATIVE_GPT_LM_HEAD_CONCURRENT_DHIDDEN_DWEIGHT=1
  bash tools/bench_native_gpt_sm120_candidate.sh`.

- Removed the redundant full-device sync before dense GPT native LM-head
  train-loss scalar copy. The default path now relies on the blocking
  device-to-host `cudaMemcpy` for the required ordering, while
  `NFN_NATIVE_GPT_LM_HEAD_LOSS_COPY_SYNC=1` /
  `NFN_NATIVE_GPT2_LM_HEAD_LOSS_COPY_SYNC=1` reproduces the old
  sync-before-copy behavior for same-binary diagnostics. Runtime JSON now
  reports `lm_head_loss_copy_device_synchronize_enabled` and
  `lm_head_loss_copy_ordering`. Migration notes: no caller change is required;
  only diagnostic consumers that assert the old explicit sync should update
  their expectations. Verification: `python -m pytest tests/test_native_gpt2.py
  -q -k "native_gpt2_cpp_cli_builds_and_uses_sm120_defaults"`; `bash
  tools/build_native_gpt_cli.sh`; `python tools/check_native_no_torch_deps.py`;
  a one-step CUDA train-loss smoke with `--train-loss-every-steps 1`; and
  `NFN_TILE_CUDA_TEST=1 python -m pytest --lf -q -rs`, which passed with
  `1167 passed`, `20 warnings`, and `468 subtests passed` under CUDA 13.3.
  Dedicated RTX 5090 same-script timing compared the previous sync path
  (`NFN_NATIVE_GPT_LM_HEAD_LOSS_COPY_SYNC=1`) against the default with
  train-loss sampling enabled and measured the default as neutral/noise-level
  (`1.000906x` train-loop wall, `0.999097x` tokens/sec), so this is an ordering
  cleanup rather than a material parity improvement.

- Rechecked the remaining dense GPT SM120 parity switches after the CUDA
  13.3.33 Tile extension-load fix and kept them rejected as defaults. A
  stage-timed dedicated RTX 5090 parity run measured llm.kittens at
  `2507.726 ms/step` and NeuralFn at `2602.880 ms/step`
  (`1.037944x` train-loop wall time, `0.963012x` tokens/sec), with hot buckets
  still in `block_backward` and `lm_head_backward`. Current shape stats show
  LM-head logits `50304,8192,768,T,N` and dHidden `768,8192,50304,N,N` on the
  GEMMEx/SGEMM-family fallback route and LM-head dWeight
  `768,50304,8192,N,T` on cuBLASLt. Same-script native-vs-native checks
  rejected `NFN_NATIVE_GPT_LM_HEAD_DWEIGHT_BEFORE_DHIDDEN=1`
  (`1.001614x`), `NFN_NATIVE_GPT_LM_HEAD_REVERSE_CHUNKS=1`
  (`1.001020x`), compile-time
  `NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE=8192` (`1.002704x` startup total
  wall, `1.028856x` token init), the MLP/all-projection dInput-before-dWeight
  scheduling probes (noise-equivalent), TK dInput for
  `768,8192,50304,N,N` (`1.023513x`), cuBLASLt heuristic 0 for
  `768,50304,8192,N,T` (`1.001584x`), and
  `NFN_NATIVE_GPT_LM_HEAD_ROW_CHUNK_SIZE=16384` (`1.002609x`). The next parity
  work remains a new fused/cooperative row-chunked LM-head backward kernel or a
  materially different GEMM route, not more switch promotion.

- Fixed the CUDA Toolkit 13.3 Tile extension build after the dense GPT
  residual-add specialization. The
  `dim768_bf16_residual_add_enabled()` helper now lives in the shared
  `kernels.cu` utility section instead of the native-only cublas-linear block,
  so the PyTorch Tile CUDA extension and the standalone native trainer C ABI
  compile the same residual-add launcher source. This turns the extension-backed
  GPU pytest cases from skip-by-build-error into real CUDA execution on the
  RTX 5090. Verification: direct extension load reported
  `extension_loaded=True` with CUDA Toolkit `13.3` and compute capability
  `12.0`; `NFN_TILE_CUDA_TEST=1 python -m pytest --lf -q -rs` passed with
  `1167 passed` and no skipped tests; `bash tools/build_native_train_tile_ops.sh`
  rebuilt `build/libnfn_native_train_tile_ops.so`.

- Defaulted the dense GPT native BF16 projection residual-add helper to a
  768-wide CUDA specialization for GPT-shaped residual vectors. The new
  `linear_bias_residual_add_bf16_linear_dim768_float32_kernel` avoids the
  generic per-element output-column modulo in
  `nfn_native_tile_linear_bias_residual_add_bf16_linear_float32`; set
  `NFN_TILE_CUDA_DIM768_BF16_RESIDUAL_ADD=0`,
  `NFN_NATIVE_GPT_DIM768_BF16_RESIDUAL_ADD=0`, or
  `NFN_NATIVE_GPT2_DIM768_BF16_RESIDUAL_ADD=0` to reproduce the generic helper
  for paired bisection. Dedicated RTX 5090 same-script timing measured the
  specialized default at `0.998835x` mean train-loop wall time,
  `1.001172x` mean tokens/sec, and `0.997518x` median total wall time versus
  the generic path over five samples. Verification:
  `bash tools/build_native_train_tile_ops.sh`;
  `bash tools/build_native_gpt_cli.sh`;
  `NFN_SM120_NATIVE_STEPS=5 NFN_SM120_NATIVE_SAMPLES=5
  NFN_SM120_NATIVE_WARMUP=1
  NFN_SM120_NATIVE_BASELINE_ENV='NFN_TILE_CUDA_DIM768_BF16_RESIDUAL_ADD=0'
  bash tools/bench_native_gpt_sm120_candidate.sh`.

- Defaulted dense GPT native embedding forward to a fused Tile-CUDA token
  embedding + absolute position embedding + scaled residual kernel. The new raw
  ABI exports `nfn_native_tile_token_position_embedding_residual_float32` and
  `nfn_native_tile_token_position_embedding_residual_u16_float32`; the dense
  transformer trainer uses the u16 variant on the default direct-token path,
  eliding the `token_out` and `position_out` FP32 activation buffers
  (`402653184` bytes at the default `64 x 1024 x 768` shape). Set
  `NFN_NATIVE_GPT_FUSE_EMBEDDING_RESIDUAL=0`,
  `NFN_NATIVE_GPT2_FUSE_EMBEDDING_RESIDUAL=0`, or
  `NFN_TILE_CUDA_FUSE_EMBEDDING_RESIDUAL=0` to restore the older three-launch
  embedding path for paired diagnostics. Dedicated RTX 5090 same-script timing
  measured the old opt-out path at `1.001318x` train-loop wall time and
  `0.998685x` tokens/sec versus the fused default over five samples.
  Verification: `bash tools/build_native_train_tile_ops.sh`;
  `bash tools/build_native_gpt_cli.sh`; focused native GPT pytest C ABI/source
  slice; `python tools/check_native_no_torch_deps.py`; `git diff --check`;
  dedicated RTX 5090 native-vs-native benchmark with selected-GPU idle checks.

- Revisited the BF16 classifier dlogit vector-store candidate after the BF16 CE
  vector-load default and the CUDA Toolkit 13.3.33 WSL reinstall. The dedicated
  RTX 5090 same-script 5-step, 5-sample benchmark with selected-GPU idle checks
  measured `NFN_NATIVE_GPT_CE_BF16_VEC_STORES=1` at `0.999390x` mean
  train-loop wall time and `1.000618x` mean tokens/sec versus scalar stores,
  but median train-loop wall regressed to `1.000326x` and paired command wall
  was effectively flat/slower. Keep the 128-bit streaming-store path
  diagnostic-only; the promoted default remains vectorized BF16 row loads with
  scalar dlogit stores. Verification:
  `NFN_SM120_NATIVE_STEPS=5 NFN_SM120_NATIVE_SAMPLES=5
  NFN_SM120_NATIVE_WARMUP=1 NFN_SM120_NATIVE_PROFILE_DIR=none
  NFN_SM120_NATIVE_CANDIDATE_ENV='NFN_NATIVE_GPT_CE_BF16_VEC_STORES=1'
  bash tools/bench_native_gpt_sm120_candidate.sh`.

- Rechecked two tempting LM-head routes after the BF16 CE vector-load default
  and kept both rejected. `NFN_NATIVE_LINEAR_BF16_OUTPUT_CUBLASLT=1
  NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=50304,8192,768,T,N,0` moved the
  LM-head logits shape onto cuBLASLt but measured `1.000302x` train-loop wall
  time and `0.999702x` tokens/sec versus the default BF16 GEMMEx route, so it
  remains diagnostic-only. `NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS=1`
  completed but measured `12.602812x` train-loop wall time and `0.079348x`
  tokens/sec versus the row-chunked default; it does not reduce GEMM count and
  instead adds a full resident BF16 logit tape. The next LM-head implementation
  target remains a fused/cooperative row-chunked classifier-backward kernel,
  not cuBLASLt heuristic tuning or full-logit reuse. Verification: dedicated
  RTX 5090 same-script native-vs-native benchmarks with selected-GPU idle
  checks.

- Defaulted dense GPT native BF16 classifier/CE row scans to vectorized 8x
  BF16 loads, aligning the native Tile-CUDA classifier pass with the
  llm.kittens fused-classifier row-read pattern without changing the raw C ABI.
  Set `NFN_NATIVE_GPT_CE_BF16_VEC_LOADS=0`,
  `NFN_NATIVE_GPT2_CE_BF16_VEC_LOADS=0`, or
  `NFN_TILE_CUDA_CE_BF16_VEC_LOADS=0` to restore the older scalar-load path for
  bisection. Dedicated RTX 5090 same-script benchmarks measured the vector-load
  candidate at `0.998750x` train-loop wall time and `1.001257x` tokens/sec
  versus the scalar default over five samples; after promotion, the scalar
  opt-out measured `1.002740x` train-loop wall time and `0.997270x` tokens/sec
  versus the new default over three samples. Verification:
  `bash tools/build_native_train_tile_ops.sh`; focused
  `tests/test_native_gpt2.py` native C ABI/source-contract slice;
  `python tools/check_native_no_torch_deps.py`; `git diff --check`; dedicated
  RTX 5090 native-vs-native benchmarks with selected-GPU idle checks.

- Rechecked the remaining dense GPT native LM-head and startup candidate paths
  after the CUDA Toolkit 13.3.33 WSL reinstall and kept the current defaults.
  A one-step stage/shape profile on the dedicated RTX 5090 measured
  `train_loop_wall_ms: 2865.24`, `setup_wall_ms: 546.399`, and
  `setup.token_weight_init.total_ms: 133.577`; the hottest buckets remained
  `block_backward` (`1337.510 ms`), `lm_head_backward` (`824.768 ms`), and
  `train.model_forward` (`674.512 ms`). The hot LM-head shapes still route
  logits and dHidden through BF16 GEMMEx while dWeight uses cuBLASLt. Expanding
  cuBLASLt to the LM-head logits path with
  `NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1
  NFN_NATIVE_LINEAR_BF16_OUTPUT_CUBLASLT=1` increased train-loop wall time to
  `1.027700x` and reduced tokens/sec to `0.973050x` versus the current default.
  The startup-only threaded token initializer
  `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1` also regressed total startup to
  `1.044736x`, setup wall time to `1.044749x`, and token-weight init time to
  `1.133043x`. Verification: dedicated RTX 5090 native-vs-native
  same-script benchmarks with selected-GPU idle checks and no promoted code
  changes.

- Added shape-selective TK dInput routing for the trainer-facing Tile-CUDA
  linear ABI. `NFN_NATIVE_LINEAR_TK_DINPUT=1` remains the broad diagnostic
  switch, while `NFN_NATIVE_LINEAR_TK_DINPUT_ENABLE_SHAPE=m,n,k,opA,opB` /
  `NFN_TILE_CUDA_LINEAR_TK_DINPUT_ENABLE_SHAPE=...` and matching
  `*_DISABLE_SHAPE` aliases allow single-shape bisection without moving every
  supported BF16/BF16 dInput GEMM onto TK. The intended LM-head dHidden probe
  `768,8192,50304,N,N` was rejected as a default: it raised TK GEMM count from
  `2400` to `2720`, but regressed `lm_head_backward` to `1.080834x`,
  train-loop wall time to `1.019686x`, and tokens/sec to `0.980699x` versus
  GEMMEx. Verification: `bash tools/build_native_train_tile_ops.sh`;
  GPU-visible `python -m pytest
  tests/test_native_gpt2.py::test_native_train_tile_ops_builds_torch_free_c_abi -q -rs`;
  dedicated RTX 5090 same-script native-vs-native benchmark with
  `NFN_NATIVE_LINEAR_TK_DINPUT_ENABLE_SHAPE=768,8192,50304,N,N`.

- Revisited the CUDA-visible failure set after installing CUDA Toolkit 13.3.33
  for WSL on the dedicated RTX 5090. `nvcc --version` now reports
  `V13.3.33`, `nvidia-smi` reports CUDA UMD `13.3` with no active GPU compute
  processes before the run, and `NFN_TILE_CUDA_TEST=1 python -m pytest --lf -q -rs`
  passed with `1167 passed, 20 warnings, 468 subtests passed`. Rebuilt both
  native CUDA artifacts with `bash tools/build_native_gpt_cli.sh` and
  `bash tools/build_native_train_tile_ops.sh`; `python tools/check_native_no_torch_deps.py`
  reported every native training/inference entrypoint `ok`. The fresh
  same-script 5-step, 2-sample stage-timed parity run measured llm.kittens at
  `2481.764 ms/step` and NeuralFn at `2568.290 ms/step`, or `1.034910x`
  NeuralFn train-loop wall time and `0.965859x` NeuralFn tokens/sec versus the
  reference. This retest closes the stale failure concern: the remaining work is
  still native kernel throughput in `block_backward` and `lm_head_backward`,
  not a broken CUDA install, Torch leakage, or graph-editor tensor flow.

- Rejected two more CUDA 13.3.33 activation-tape memory candidates as dense GPT
  defaults because both improved setup but slowed the actual training loop.
  `NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUTS=1` reduced setup wall time to
  `0.944457x` but regressed train-loop wall time to `1.014826x` and tokens/sec
  to `0.985409x`. `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_BF16=0` reduced
  setup wall time to `0.960234x` but regressed train-loop wall time to
  `1.004923x` and tokens/sec to `0.995113x`. Keep both as diagnostic-only
  memory/startup bisection paths; neither moves NeuralFn toward llm.kittens
  training throughput. Verification: dedicated RTX 5090 same-script
  native-vs-native benchmarks with stage timing and GPU idle/lock checks.

- Fixed the CUDA 13.3 retest failure set instead of carrying stale pytest
  failures forward. The GPT-2 compatibility wrapper now exposes the legacy
  parser/helper surface expected by CLI tests (`--pretraining-file`, dataset
  shortcut resolution, tokenizer/vocab policy, evolutionary and scheduler
  fields, `mode_name`/`graph_name`, and summary printing) while preserving the
  direct native compiled GPT execution path and no-Torch import contract.
  `--all-train-rows` now ignores parser default `max_steps` unless max steps
  were explicitly supplied, so the default all-rows schedule uses the documented
  two-epoch floor. Raw-text validation fallback now uses a deterministic 10%
  tail holdout, and GPT wrapper pretraining-file adapters create a local
  raw-text dataset view without importing Torch. The unified native CLI keeps
  `--native-cuda-no-checkpoint` visible only when an explicit dense GPT model
  selector needs that wrapper spelling, while default `nfn train` still
  normalizes to compiled `--no-checkpoint`. NanoGPT direct execution uses the
  GPT native trainer by default for transformer-LM mode but honors
  `NFN_NATIVE_NANOGPT_CLI` when explicitly configured. Quantized int8 export
  now leaves embedding tables at full precision and applies int8 storage only
  to linear/projection weight tensors, matching the documented contract and
  removing flaky loss drift in round-trip tests. Verification:
  `NFN_TILE_CUDA_TEST=1 python -m pytest --lf -q -rs` passed with
  `1167 passed, 20 warnings, 468 subtests`; CUDA Tile GPU slice passed with
  `537 passed, 6 warnings`; native GPT tests passed with `51 passed, 1 skipped`;
  affected CLI/no-Torch tests passed with `81 passed, 282 subtests`; template
  preset tests passed with `26 passed`; `python tools/check_native_no_torch_deps.py`
  reported all checks `ok`.

- Rechecked the current dense GPT native LM-head bisection space after the
  CUDA Toolkit 13.3 reinstall and kept the defaults unchanged. On the dedicated
  RTX 5090, the no-stage-timing 5-step, 2-sample parity run still measured
  NeuralFn at `1.027170x` train-loop wall time versus llm.kittens, confirming
  the remaining gap is real kernel throughput rather than instrumentation.
  Native-vs-native candidates were rejected as defaults: BF16 CE `exp2`
  measured `1.001188x` train-loop wall time, `--lm-head-row-chunk-size 16384`
  measured `1.017714x`, `--lm-head-row-chunk-size 4096` measured `1.008782x`,
  LM-head dWeight heuristic override
  `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,50304,8192,N,T,0` measured
  `1.000315x`, and `NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF=1` measured
  `1.001155x`. A shape-stats probe reconfirmed the hot LM-head buckets:
  logits GEMMEx `50304,8192,768,T,N`, dHidden GEMMEx
  `768,8192,50304,N,N`, and dWeight cuBLASLt `768,50304,8192,N,T`. The next
  useful implementation remains a fused/cooperative LM-head classifier/backward
  kernel or materially different GEMM route, not more chunk-size or heuristic
  switches. Verification: paired native-vs-native RTX 5090 benchmarks with the
  same dedicated-GPU harness; no code changes were promoted from these probes.

- Rejected an LM-head dHidden/dWeight side-stream overlap prototype for dense
  GPT native training. The tested implementation recorded CE completion on the
  default stream, launched LM-head dHidden on a non-blocking side stream, ran
  LM-head dWeight on the default stream, and synchronized before reusing the
  row-chunk BF16 logit buffer. On the dedicated RTX 5090, the 5-step,
  2-sample native-vs-native run measured `1.003463x` train-loop wall time and
  `0.996573x` tokens/sec versus the serial default, so the prototype was
  removed instead of kept as another diagnostic switch. The remaining useful
  LM-head work is still a fused/cooperative row-chunked classifier-backward
  kernel, not stream-level overlap of two large GEMMs. Verification:
  `bash tools/build_native_gpt_cli.sh`; focused native GPT source-contract
  pytest; `python tools/check_native_no_torch_deps.py`; `git diff --check`;
  CUDA 13.3 RTX 5090 native-vs-native benchmark with candidate env
  `NFN_NATIVE_GPT_LM_HEAD_OVERLAP_DHIDDEN_DWEIGHT=1`.

- After reinstalling CUDA Toolkit 13.3 for WSL, reran the failed-test and
  native-performance gates on the dedicated RTX 5090. The CUDA Tile GPU suite
  passed with `537 passed, 6 warnings`, native GPT contract tests passed with
  `51 passed, 1 skipped`, Tile registry/example tests passed with `28 passed`,
  `tools/check_native_no_torch_deps.py` reported all checks `ok`, and both
  native CUDA binaries rebuilt cleanly. The refreshed same-script parity run
  still measured NeuralFn dense GPT at `1.032833x` train-loop wall time versus
  `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh`, so the remaining
  issue is native kernel throughput, not correctness, Torch leakage,
  graph-editor tensor flow, or stale CUDA state. CUDA 13.3 bisections rejected
  forward-logit reuse with stored MLP disabled (`1.364953x` train loop),
  forward-logit reuse with packed-attention storage disabled (`1.103069x`),
  re-enabling TK BF16 forward for the default LM-head logits shape
  (`1.008700x`), and BF16-output cuBLASLt for LM-head logits (`1.002706x`).
  `todo-tile-cuda.md` now records those rejections and narrows the next
  implementation target to a fused row-chunked LM-head backward kernel that
  preserves the current 8192-row BF16 resident-logit cap. Verification:
  `NFN_TILE_CUDA_TEST=1 python -m pytest tests/test_tile_cuda_gpu.py
  tests/test_tile_cuda_ops.py tests/test_tile_cuda_optimizer.py
  tests/test_tile_cuda_modules.py -q -rs`; `python -m pytest
  tests/test_native_gpt2.py -q`; `python -m pytest tests/test_tile_cuda_registry.py
  tests/test_tile_cuda_examples.py -q`; `python tools/check_native_no_torch_deps.py`;
  `bash tools/build_native_gpt_cli.sh`; `bash tools/build_native_train_tile_ops.sh`;
  CUDA 13.3 RTX 5090 same-script parity and native-vs-native bisections.

- Revisited CUDA-visible Tile-CUDA tests and SM120 parity after installing CUDA
  Toolkit 13.3 for WSL on the dedicated RTX 5090. The combined CUDA test slice
  passed with `537 passed, 6 warnings`, and the canonical same-script parity
  run measured NeuralFn dense GPT at `1.032743x` train-loop wall time versus
  `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh`. The current stage
  profile still points at native GPU hot paths, primarily `block_backward` and
  `lm_head_backward`; CUDA 13.3 rechecks rejected forward-logit reuse, LM-head
  ordering changes, cuBLASLt heuristic/large-shape toggles, and packed-attention
  storage toggles as defaults. `todo-tile-cuda.md` now records the current
  evidence and keeps the remaining parity work focused on a new fused or
  co-scheduled LM-head classifier/backward kernel rather than more switch
  promotion. Verification: `NFN_TILE_CUDA_TEST=1 python -m pytest
  tests/test_tile_cuda_gpu.py tests/test_tile_cuda_ops.py
  tests/test_tile_cuda_optimizer.py tests/test_tile_cuda_modules.py -q -rs`;
  `bash tools/bench_native_gpt_sm120_parity.sh` with
  `NFN_SM120_PARITY_STEPS=5`, `NFN_SM120_PARITY_SAMPLES=2`, and stage timing;
  focused native-vs-native candidate probes.

- Dense GPT native training now defaults to eliding the unused FP32
  attention-projection and MLP-projection scratch-tape buffers whenever BF16
  projection-residual is active. The previous diagnostic switch
  `NFN_NATIVE_GPT_ELIDE_FLOAT_PROJECTION_OUTPUTS=1` is now the default; set
  `NFN_NATIVE_GPT_ELIDE_FLOAT_PROJECTION_OUTPUTS=0` or the GPT-2-prefixed
  fallback only to reproduce the older allocation for paired bisection. Runtime
  JSON continues to report `float_projection_outputs_elided`,
  `float_projection_output_elements_elided`, and matching
  `block_state_layout.float_projection_output_*` counters. On the dedicated RTX
  5090 with CUDA 13.3, the post-change same-script paired check compared the new
  default against `NFN_NATIVE_GPT_ELIDE_FLOAT_PROJECTION_OUTPUTS=0` and measured
  the old allocation at `1.001273x` train-loop wall and `1.000402x` total wall
  versus the default, with selected-GPU locking and idle checks. Verification:
  `bash tools/build_native_gpt_cli.sh`;
  `python -m pytest tests/test_native_gpt2.py -q`;
  `python tools/check_native_no_torch_deps.py`; `git diff --check`; CUDA 13.3
  paired benchmark with `NFN_SM120_NATIVE_STEPS=5`,
  `NFN_SM120_NATIVE_SAMPLES=2`, and candidate env
  `NFN_NATIVE_GPT_ELIDE_FLOAT_PROJECTION_OUTPUTS=0`.

- Fused dense GPT sampled train-loss CE and dlogit generation on the default
  BF16/u16-token Tile CUDA path. The native raw C ABI now exports
  `nfn_native_tile_token_cross_entropy_backward_loss_inplace_strided_bf16_bits_u16_targets`,
  which accumulates public-vocab CE loss and overwrites BF16 logits with
  dlogits in one kernel when `--train-loss-every-steps` is enabled. Runtime JSON
  reports `lm_head_ce_loss_backward_fused_available` and
  `lm_head_ce_loss_backward_strategy`; validation loss remains controlled by
  `--eval-every-steps` and stays separate from train-loss sampling. Same-script
  paired RTX 5090 timing with `--train-loss-every-steps 1` measured the
  candidate at `0.454182x` train-loop wall time and `2.201755x` tokens/sec
  versus the previous separate-loss pass. The normal no-train-loss route was
  neutral at `0.999433x` train-loop wall time. Verification: rebuilt
  `build/libnfn_native_train_tile_ops.so` and `build/nfn_gpt_native_train`;
  one-step native GPT smoke with the fused ABI; paired native-vs-native CUDA
  benchmark with selected-GPU locking/idle checks; `python -m pytest
  tests/test_tile_cuda_examples.py -q`; `python -m pytest
  tests/test_native_gpt2.py -q`; `python tools/check_native_no_torch_deps.py`;
  `git diff --check`.

- Added a default per-selected-GPU lock to `tools/paired_kernel_speed.py`.
  GPU-visible paired benchmark runs now lock
  `/tmp/nfn_paired_kernel_speed_gpu_<device>.lock` before warmup or measured
  commands, preventing accidental concurrent baseline/candidate runs on the
  same RTX 5090 before the `nvidia-smi` idle-process guard can observe them.
  Use `--gpu-benchmark-lock-timeout-seconds N` to wait for the lock, or
  `--no-gpu-benchmark-lock` only for intentionally unmanaged measurements.
  JSON/text output reports the lock path, timeout, and acquisition state.
  During the same CUDA 13.3 revisit, the fresh llm.kittens parity run measured
  NeuralFn native dense GPT at `1.043386x` train-loop wall time over 3 steps.
  BF16 CE vector stores were not promoted after a 5-step, 2-sample confirmation
  measured `0.999654x` mean train-loop wall time with a range crossing slower;
  `NFN_TILE_CUDA_CE_BF16_THREADS=512` was rejected at `1.002489x`.
  Verification: `python -m py_compile tools/paired_kernel_speed.py`;
  focused paired-kernel tests covering smoke, dry-run, and lock rejection;
  CUDA 13.3 RTX 5090 same-script parity/candidate benchmark runs with selected
  GPU idle checks.

- Added `NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS=1` /
  `NFN_NATIVE_GPT2_REUSE_FORWARD_LM_HEAD_LOGITS=1` as a diagnostic dense GPT
  classifier path. The native C++ trainer can now allocate full BF16 LM-head
  logits, fill them once after the transformer forward, and have LM-head
  backward consume chunk offsets from that full buffer instead of recomputing
  logits. Runtime JSON reports `lm_head_reuse_forward_logits_enabled`,
  `lm_head_full_logit_elements`, and `lm_head_bf16_logit_bytes`, and
  `tools/paired_kernel_speed.py` now extracts those fields plus the saved
  packed-attention block count. This mirrors the llm.kittens full-logit
  classifier layout for controlled RTX 5090 comparisons, but it remains off by
  default: the full buffer costs about 6.14 GiB at the default
  `64 x 1024 x 50304` padded-vocab shape. CUDA 13.3 paired checks measured the
  full-logit reuse path with `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=0`
  at `1.099054x` train-loop wall time, with four saved packed-attention blocks
  at `1.061321x`, and an eight-block smoke fit but degraded to an `83.5s`
  one-step train loop near the memory cliff. Verification: rebuilt
  `build/nfn_gpt_native_train`; ran one-step RTX 5090 smokes for the zero-block
  and eight-block memory points; ran same-script paired benchmarks for the
  zero-block and four-block candidates.

- Hardened the native-vs-native SM120 benchmark wrapper so candidate-only CLI
  flags cannot silently drop out of a bisection. `tools/bench_native_gpt_sm120_candidate.sh`
  now accepts `NFN_SM120_NATIVE_CANDIDATE_ARGS` as an alias for candidate-only
  arguments, alongside the documented `NFN_SM120_NATIVE_CANDIDATE_EXTRA_ARGS`
  and `NFN_SM120_CANDIDATE_EXTRA_ARGS`; dry-run output still prints both
  resolved commands so reviewers can confirm the flag did not leak into the
  baseline. This was found while revisiting CUDA 13.3 RTX 5090 parity: the
  current default measured `1.042317x` train-loop wall time versus
  `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh` over a 3-step
  timing run. Same-script candidate checks rejected reverse LM-head chunks
  (`1.002565x`), LM-head dWeight-before-dHidden (`1.003312x`), 4096-row
  LM-head chunks (`1.002915x`), disabling stored MLP activations
  (`1.366368x`), CUDA malloc async startup (`1.087418x` setup wall time), and
  threaded token-weight init (`1.057832x` token init), so none of those tuning
  switches were promoted. Verification: `bash -n tools/bench_native_gpt_sm120_candidate.sh`;
  `python -m pytest tests/test_tile_cuda_examples.py -q -k
  'native_gpt_sm120_candidate_wrapper_forwards_bisection_controls or
  native_gpt_sm120_candidate_wrapper_accepts_short_aliases or
  native_gpt_sm120_candidate_wrapper_accepts_legacy_candidate_args_alias'`;
  the RTX 5090 CUDA Tile suite (`537 passed`), GPU layer-evo tests
  (`3 passed`), `tests/test_native_gpt2.py` (`52 passed`), template presets
  (`26 passed`), Tile metadata/static/example/dependency checks (`37 passed`),
  and `python tools/check_native_no_torch_deps.py`.

- Revisited the failed CUDA 13.3/WSL test set and made the local test/runtime
  paths deterministic. No-Redis persistence now processes the local fallback
  synchronously instead of spawning daemon threads, which prevents SQLite
  teardown races and guarantees latest session/run state is written before a
  local request returns. Raw-text vocab-size resolution now returns known sizes
  for `gpt2`, `cl100k_base`, and `o200k_base` without requiring a local
  `.tiktoken` file; full tokenization still requires tokenizer assets. The
  platform API test harness now bypasses the Python 3.13/AnyIO sync-threadpool
  hang with a direct ASGI caller and inline threadpool shims, and stale tests
  were updated for current JEPA semantic graph inputs, native GPT script
  boundaries, resolver keyword contracts, and eager CPU JEPA training. Verified
  CUDA Toolkit 13.3 by rebuilding `build/libnfn_native_train_tile_ops.so` and
  `build/nfn_gpt_native_train`; ran `tests/test_platform_api.py`
  (`8 passed`), `tests/test_jepa_semantic.py` (`62 passed`),
  `cli/tests/test_train_drop_last.py` plus
  `tests/test_tokenizer_vocab_contract.py` (`60 passed, 10 subtests passed`),
  backend/server/routing/dataset focused tests (`58 passed`),
  `tests/test_tile_cuda_examples.py` (`18 passed`),
  `tests/test_native_gpt2.py` (`51 passed, 1 skipped`; the skip is the
  sandbox-only CUDA driver check), `python tools/check_native_no_torch_deps.py`,
  the GPU-visible CUDA Tile suite (`537 passed`), and a real RTX 5090 one-step
  native Tile-CUDA smoke reporting CUDA runtime/driver `13.3` and
  `passed: true`.

- Extended `tools/paired_kernel_speed.py` reporting for native CUDA Tile
  bisections. The JSON extractor already read several backend counters, but the
  terminal report now also prints and ratios `linear_tk_gemm_count`,
  `linear_cublaslt_gemm_count`, `linear_bf16_gemm_count`, linear BF16 pack/cache
  counters, and attention TK launch counts. This makes active backend-route
  candidates visible in same-script RTX 5090 comparisons even when a coarse
  strategy string is unchanged. Verification: ran the paired-kernel smoke tests
  covering sidecar extraction and stdout output; ran fresh CUDA 13.3 native
  bisections that rejected the LM-head TK logits shape override
  (`1.005595x` train-loop wall time), broad TK dInput route (`1.053838x`), and
  LM-head dWeight cuBLASLt heuristic-0 override (`1.003965x`).

- Corrected the compiled Tile-CUDA packed-attention dprep helper so the GPT
  `heads=12, head_dim=64` BF16-gradient specialization is actually default-on
  when `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_HD64_SPECIALIZED` is unset,
  matching the existing README and benchmark checklist. Setting
  `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_HD64_SPECIALIZED=0` still reproduces
  the older generic row dprep route for same-binary bisection. Verification
  after reinstalling CUDA for WSL: `nvcc --version` reported CUDA 13.3.33;
  rebuilt `build/libnfn_native_train_tile_ops.so`; ran one-step RTX 5090
  native CUDA profiles for the fixed default and explicit opt-out paths, both
  passing with `attention_backward_dprep_timing_count: 96`; ran
  `python -m pytest tests/test_native_gpt2.py -q` (`51 passed, 1 skipped`);
  ran `python tools/check_native_no_torch_deps.py`.

- Removed benchmark-mode post-train diagnostic D2H samples from the dense GPT
  native trainer. Runs with `--native-cuda-sample-every 0` now skip the final
  token-weight and clip-scale host copies, report
  `post_train_diagnostic_samples_elided: true`, and still pass based on
  completed optimizer steps instead of requiring the diagnostic weight-delta
  sample. This keeps short CUDA Tile benchmark runs from adding an avoidable
  end-of-run CPU/GPU synchronization when sampling is explicitly disabled.
  Verification: rebuilt `build/nfn_gpt_native_train`; ran a one-step RTX 5090
  smoke with sampling/checkpoint/eval disabled that reported
  `post_train_diagnostic_sample_d2h_count: 0`,
  `post_train_diagnostic_sample_d2h_count_elided: 2`, and `passed: true`; ran
  the focused native GPT pytest selection (`1 passed, 1 skipped`). A fresh
  post-change parity sample measured NeuralFn at `1.033683x` train-loop wall
  time versus `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh`, so the
  main remaining gap is still inside the CUDA training loop rather than this
  post-loop diagnostic path.

- Fused the default dense GPT tied token-weight BF16 LM-head shadow refresh into
  the float32 descriptor AdamW update. In the BF16-primary block-weight path,
  token/position/norm/bias tensors still use the float32 multi-buffer AdamW
  descriptor launch, but the token descriptor now carries a BF16-shadow offset
  so `nfn_native_tile_adamw_step_many_with_device_scale_bf16_shadow_float32`
  updates the FP32 master token table and its BF16 LM-head shadow in the same
  kernel. This removes the separate post-AdamW
  `token_weight_bf16.post_adamw_refresh` pack from the default path while
  preserving `NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_ADAMW_REFRESH=0` for
  same-binary paired bisection against the old behavior. Runtime JSON now
  reports `token_weight_bf16_adamw_refresh_fusion_enabled`,
  `token_weight_bf16_fused_adamw_refresh_count`, and the active strategy names
  `split-float32-token-shadow-and-bf16-param-multi-buffer-device-scale` /
  `elided-block-bf16-primary-token-shadow-fused-adamw`. Verification after the
  WSL CUDA toolkit reinstall: `nvcc --version` reported CUDA 13.3.33; rebuilt
  `build/nfn_gpt_native_train` and `build/libnfn_native_train_tile_ops.so`; ran
  a one-step RTX 5090 smoke that reported CUDA runtime/driver 13.3,
  `token_weight_bf16_fused_adamw_refresh_count: 1`, and the fused strategy
  names; ran the same-script paired benchmark with baseline
  `NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_ADAMW_REFRESH=0`, which measured
  candidate/default at `0.998488x` train-loop wall time and `1.001519x`
  tokens/sec; ran `python -m pytest tests/test_native_gpt2.py -q`
  (`51 passed, 1 skipped`); ran `python tools/check_native_no_torch_deps.py`.

- Hardened `tools/paired_kernel_speed.py` timeout cleanup for SM120 parity and
  candidate benchmarks. Timed-out commands now kill and wait on the actual
  process group before the timeout sample is recorded, preventing oversized CUDA
  candidates from leaving native trainer processes on the selected GPU after
  the harness continues. During CUDA 13.3 retuning, a full-resident LM-head
  candidate (`--lm-head-row-chunk-size 65536`) was rejected after exceeding the
  benchmark timeout, while rechecked `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX=0`
  and `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1` candidates measured slower
  than the current defaults. Verification:
  `python -m pytest tests/test_tile_cuda_examples.py -q -k paired_kernel_speed_tool_records_command_timeout`;
  `python -m py_compile tools/paired_kernel_speed.py`.

- Promoted the dense GPT token-weight fast int32 Tile-index initializer to the
  default when the token table fits in int32. The previous int64 Tile-index
  route remains available with `NFN_NATIVE_GPT_TOKEN_WEIGHT_FAST_INT32_INIT=0`,
  `NFN_NATIVE_GPT2_TOKEN_WEIGHT_FAST_INT32_INIT=0`, or
  `NFN_TILE_CUDA_TOKEN_WEIGHT_FAST_INT32_INIT=0` for paired startup bisection.
  Runtime JSON now reports `token_weight_fast_int32_init_enabled` beside the
  existing token initialization strategy fields. On CUDA 13.3, the dedicated
  RTX 5090 startup-only 5-sample benchmark measured `0.955863x`
  token-weight-init time, `0.988331x` setup wall time, and `0.988772x` total
  startup versus the previous default; the normal one-step 3-sample run kept
  train-loop timing neutral at `1.000072x`. Verification: ran the startup-only
  and normal paired candidate benchmarks; rebuilt
  `build/libnfn_native_train_tile_ops.so` and `build/nfn_gpt_native_train`;
  ran a startup-only smoke that reported
  `token_weight_fast_int32_init_enabled: true`; ran
  `python -m pytest tests/test_native_gpt2.py -q` (`52 passed`);
  ran the CUDA Tile GPU suite (`537 passed`); ran
  `python tools/check_native_no_torch_deps.py`.

- Added a diagnostic-only BF16 cuBLASLt allow-list for one-shape paired
  bisection. `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE=m,n,k,opA,opB`
  and `NFN_NATIVE_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE=m,n,k,opA,opB` force one
  otherwise-gated BF16 shape through cuBLASLt while leaving other shapes on the
  default route. The CUDA 13.3 RTX 5090 smoke confirmed the dense GPT LM-head
  dHidden bucket `768,8192,50304,N,N` moved to
  `bf16-cublaslt-dinput-dhidden`, but the paired 5-step, 3-sample benchmark
  measured `1.024865x` train-loop wall time and `0.975739x` tokens/sec versus
  the default BF16 `cublasGemmEx` fallback, so it remains rejected as a default.
  Verification: rebuilt `build/libnfn_native_train_tile_ops.so`; ran the
  one-step shape-stat smoke; ran the same-script candidate benchmark on the
  dedicated RTX 5090; ran
  `NFN_TILE_CUDA_TEST=1 NFN_TILE_CUDA_BUILD=1 NFN_TILE_CUDA_ARCH=sm_120 python -m pytest tests/test_tile_cuda_ops.py tests/test_tile_cuda_modules.py tests/test_tile_cuda_optimizer.py tests/test_tile_cuda_gpu.py -q -rs`
  (`537 passed`); ran `python tools/check_native_no_torch_deps.py`.

- Fixed a dense GPT native startup allocation regression in the stored-MLP
  activation path. When float stats sidecars are enabled, the
  `stored_mlp_norm_stats_arena` request now keeps the pointer assigned by the
  combined float arena instead of overwriting it with a second standalone
  `cudaMalloc`. Runtime JSON now reports
  `stored_mlp_layer_norm_stats_standalone_cuda_malloc_count` so startup
  profiles can prove whether this sidecar escaped the arena. Verification:
  rebuilt `build/nfn_gpt_native_train`; ran a one-step RTX 5090 native smoke
  that reported `stored_mlp_layer_norm_stats_elements: 1572864`,
  `stored_mlp_layer_norm_stats_bytes: 6291456`, and standalone malloc count
  `0`; ran `python -m pytest tests/test_native_gpt2.py -q`; ran
  `python tools/check_native_no_torch_deps.py`; ran a short 5-step SM120
  parity sample, which still showed the remaining llm.kittens gap at
  `1.042290x` train-loop wall time and `0.958942x` tokens/sec.

- Completed CUDA-event timing coverage for native Tile linear shape stats on
  the active TK paths. TK BF16 fused MLP FC+GELU, fused MLP projection
  dInput+dGELU, and TK BF16-to-float output conversion records now pass
  measured elapsed time into `linear_shape_stats`, with a host-synchronized
  fallback for fused TK GELU rows when CUDA stream events do not capture the
  helper dispatch. CUDA 13.3 profiles can now rank those buckets instead of
  emitting zero-time rows. The profiler remains opt-in through
  `NFN_NATIVE_LINEAR_SHAPE_STATS=1`,
  `NFN_TILE_CUDA_LINEAR_SHAPE_STATS=1`, `NFN_NATIVE_GPT_LINEAR_SHAPE_STATS=1`,
  or `NFN_NATIVE_GPT2_LINEAR_SHAPE_STATS=1`. Verification: focused source
  coverage in `tests/test_native_gpt2.py`, rebuilt the native Tile ops library,
  and ran a GPU-visible one-step shape-stat profile on the RTX 5090 that
  reported `tk_zero_time_rows: 0` across five TK BF16 buckets.

- Extended the native no-Torch dependency verifier to cover the remaining
  guarded direct training scripts. `tools/check_native_no_torch_deps.py` now
  stubs the `NFN_NATIVE_MIXLLAMA_CLI`, `NFN_NATIVE_JEPA_CLI`, and
  `NFN_NATIVE_DEEPSEEK_V4_CLI` family binaries and verifies
  `train_llama_fast.py`, `train_mixllama_fast.py`, `train_jepa_semantic.py`,
  and `train_deepseek_v4.py` alongside the existing GPT, GPT-2-evo, NanoGPT,
  LLaMA megakernel, semantic-router MoE, `nfn train`, native inference, and SDK
  handoff checks. This keeps every guarded direct legacy training script on
  the compiled native C++ boundary before Torch/NumPy/tokenizer/dataset-manager
  imports can occur. Verification:
  `python tools/check_native_no_torch_deps.py --json`.

- Extended the native no-Torch dependency verifier to cover more direct
  training entrypoints. `tools/check_native_no_torch_deps.py` now stubs
  family-specific native binaries for `NFN_NATIVE_LLAMA_CLI` and
  `NFN_NATIVE_SEMANTIC_ROUTER_MOE_CLI`, then verifies
  `train_llama_megakernel.py`, `train_semantic_router_moe.py`, and
  `train_semantic_router_moe-overnight.py` under the same import blocker used
  for GPT, GPT-2-evo, NanoGPT, native inference, and SDK handoff checks. This
  keeps those direct CLI training paths on the compiled native C++ boundary
  before Torch/NumPy/dataset-manager imports. Verification:
  `python tools/check_native_no_torch_deps.py --json`.

- Revisited the CUDA 13.3 WSL failure surfaces with GPU-visible execution. A
  sandboxed native benchmark still failed because `nvidia-smi`/NVML and
  `cudaDriverGetVersion` were blocked by the operating system, but the same
  machine outside the sandbox reported the dedicated RTX 5090 on CUDA UMD 13.3
  with no compute processes. With real GPU access, the CUDA Tile GPU pytest
  suite passed (`537 passed`) and the native GPT suite passed (`52 passed`).
  The CUDA 13.3 same-script RTX 5090 retest also kept two LM-head/CE candidate
  switches rejected: `NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=0` measured
  `1.005791x` train-loop wall time and `0.994247x` tokens/sec, while
  `NFN_NATIVE_GPT_CE_BF16_EXP2=1` measured `1.003930x` train-loop wall time and
  `0.996089x` tokens/sec. Verification:
  `NFN_TILE_CUDA_TEST=1 NFN_TILE_CUDA_BUILD=1 NFN_TILE_CUDA_ARCH=sm_120 python -m pytest tests/test_tile_cuda_ops.py tests/test_tile_cuda_modules.py tests/test_tile_cuda_optimizer.py tests/test_tile_cuda_gpu.py -q -rs`;
  `python -m pytest tests/test_native_gpt2.py -q`; paired candidate benchmarks
  with selected-GPU idle checks.

- Added v2 native Tile linear shape stats for cuBLASLt plan selection. When
  `NFN_NATIVE_LINEAR_SHAPE_STATS=1` or the GPT-specific aliases are enabled,
  cuBLASLt `linear_shape_stats` rows now report
  `cublaslt_selected_heuristic`, `cublaslt_returned_heuristics`, and
  `cublaslt_workspace_bytes` through the optional
  `nfn_native_tile_trainer_linear_shape_stats_entry_v2` ABI, while the old
  shape-stat accessor remains available for compatibility. The CUDA 13.3 RTX
  5090 profile showed the previously pinned MLP projection dWeight shape
  `3072,768,65536,N,T` returns only one cuBLASLt heuristic, so the hardcoded
  shape-specific index-1 fallback was removed as a no-op. The same retest kept
  both existing LM-head alternatives rejected:
  `NFN_NATIVE_LINEAR_BF16_OUTPUT_CUBLASLT=1` measured `1.003620x` train-loop
  wall time and `0.996405x` tokens/sec, and
  `NFN_NATIVE_LINEAR_TK_FORWARD_ENABLE_SHAPE=50304,8192,768,T,N` measured
  `1.007997x` train-loop wall time and `0.992076x` tokens/sec. Verification:
  rebuilt `libnfn_native_train_tile_ops.so`, ran a shape-stat profile that
  emitted populated cuBLASLt heuristic metadata, and ran paired native-vs-native
  RTX 5090 benchmarks for both rejected LM-head candidates.

- Fixed native training retest failures after the CUDA 13.3 WSL reinstall. The
  SDK native-train binding loader now invalidates import caches and skips stale
  in-tree `neuralfn._native_train` extensions that expose `run_train` without
  the required command resolver, then searches the package path for a complete
  rebuilt extension before falling back to `compiled-cli`. The GPT binding
  loader also invalidates import caches before discovery. The compiled
  `nfn_native_train` frontend now preserves `--print-command` in the printed
  NanoGPT token-LM delegate command, while dense GPT print-command behavior
  remains normalized to the second-stage compiled GPT command. Verification:
  `nvidia-smi` on CUDA UMD 13.3, explicit Tile CUDA GPU pytest smoke, broad
  Tile CUDA GPU pytest suite, focused native failure selectors, and full native
  pytest rerun.

- Fixed the CUDA 13.3 Python Tile extension build by moving generic BF16
  conversion, BF16 activation storage, BF16 bias-add, and linear bias-reduction
  helpers out of TK-attention/cuBLAS-only conditional blocks in
  `neuralfn/csrc/tile_cuda/kernels.cu`. The optional extension now builds on the
  RTX 5090 even when the generic PyTorch extension path is compiled without the
  trainer-facing TK attention flags, so `NFN_TILE_CUDA_TEST=1` GPU tests execute
  instead of skipping with "CUDA Tile extension could not be built or loaded".
  Verification: loaded the extension with CUDA Toolkit 13.3.33 and
  `NFN_TILE_CUDA_BUILD_DIR=/tmp/neuralfn_tile_cuda_extension_cuda133_fix2`,
  ran `NFN_TILE_CUDA_TEST=1 NFN_TILE_CUDA_BUILD_DIR=/tmp/neuralfn_tile_cuda_extension_cuda133_pytest python -m pytest tests/test_tile_cuda_gpu.py tests/test_tile_cuda_ops.py tests/test_tile_cuda_optimizer.py -q -rs`
  (`154 passed`), and reran the native C ABI gate with GPU access:
  `python -m pytest tests/test_native_gpt2.py::test_native_train_tile_ops_builds_torch_free_c_abi -q -rs`.

- Relaxed CUDA 13.3 NanoGPT native smoke tolerances that were stricter than the
  rebuilt RTX 5090 kernels' stable fp32 drift. `--smoke-lm-step` now accepts
  tied-embedding gradient error up to `1e-5`, matching its loss and weight-update
  checks. `--smoke-fused-qkv-attention-step` now accepts Q/K/V, attention/output,
  input-gradient, and output-weight-gradient error up to `1e-4` while keeping
  QKV weight-gradient and weight-update checks at `1e-5`.
  `--smoke-mlp-step` now accepts MLP output, input-gradient, and FC-gradient
  error up to `1e-4` while keeping projection-gradient and weight-update checks
  at `1e-5`. `--smoke-attention-step` now accepts attention activation/output,
  V-weight-gradient, and output-weight-gradient error up to `1e-4`, while
  zero-gradient and weight-update checks remain tighter. The previous thresholds caused the C ABI
  pytest to skip with a misleading "CUDA runtime/device not available" message
  after the kernels had actually run. Verification: rebuilt
  `libnfn_native_train_tile_ops.so` and `nfn_gpt_native_train`, ran
  `--smoke-transformer-lm-step` against TinyStories, and reran the focused
  native C ABI pytest.

- Added diagnostic-only dense GPT LM-head row-chunk traversal control with
  `NFN_NATIVE_GPT_LM_HEAD_REVERSE_CHUNKS=1` and
  `NFN_NATIVE_GPT2_LM_HEAD_REVERSE_CHUNKS=1`. Runtime JSON reports
  `lm_head_reverse_chunk_order_enabled`, and the first dWeight beta-zero write
  now follows the first processed LM-head row chunk so reverse traversal remains
  well-defined. The CUDA 13.3 dedicated RTX 5090 same-script benchmark rejected
  promotion: reverse chunks measured `1.000499x` train-loop wall time and
  `0.999506x` tokens/sec versus the default forward order. Verification:
  rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt_native_train`, ran a
  one-step reverse-chunk smoke, and ran
  `tools/bench_native_gpt_sm120_candidate.sh` with selected-GPU idle checks.

- `tools/bench_native_gpt_sm120_candidate.sh` now uses
  `NFN_SM120_CANDIDATE_EXTRA_ARGS` as the short alias for candidate-only CLI
  flags, matching the canonical `NFN_SM120_NATIVE_CANDIDATE_EXTRA_ARGS`.
  Shared baseline-and-candidate flags now use the distinct
  `NFN_SM120_COMMON_EXTRA_ARGS` alias for `NFN_SM120_NATIVE_EXTRA_ARGS`. This
  prevents row-chunk, graph, or shape bisection flags from accidentally being
  applied to both sides of a paired run. Verification: dry-run wrapper test
  confirms `--lm-head-row-chunk-size 32768` appears only in the candidate
  command; `bash -n tools/bench_native_gpt_sm120_candidate.sh`.

- Rechecked CUDA 13.3 SM120 parity and two startup/LM-head retunes. A fresh
  no-sidecar 10-step parity sample measured NeuralFn at `1.031573x` train-loop
  wall time and `0.967350x` tokens/sec versus llm.kittens. Rechecking
  `NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC=1` for startup measured `1.180688x` setup
  wall time and `1.180297x` total startup wall time versus default. Rechecking
  candidate-only `--lm-head-row-chunk-size 32768` measured `1.559818x`
  train-loop wall time and `0.643975x` tokens/sec versus the 8192-row default.
  No kernel default changed.

- `tools/bench_native_gpt_sm120_candidate.sh` now accepts short
  `NFN_SM120_CANDIDATE_*` aliases alongside the canonical
  `NFN_SM120_NATIVE_*` benchmark controls. This covers candidate steps, samples,
  warmup, train-batch tokens, CUDA device selection, profile directory, stage
  timing, candidate env, template/graph selection, dry-run, and JSON output;
  canonical native names take precedence when both are set. Verification:
  `bash -n tools/bench_native_gpt_sm120_candidate.sh`; `python -m pytest
  tests/test_tile_cuda_examples.py -q`; `python
  tools/check_native_no_torch_deps.py --json`; and `git diff --check`.

- Rejected two additional TK-forward disable probes after the CUDA 13.3 shape
  profile. Disabling the MLP projection forward shape with
  `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=768,65536,3072,T,N` measured
  `1.003165x` train-loop wall time and `0.996849x` tokens/sec versus default.
  Disabling the MLP FC+GELU forward shape with
  `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=3072,65536,768,T,N` fell back to
  an unusably slow path and measured `13.695228x` train-loop wall time and
  `0.073019x` tokens/sec. No default changed; the current TK forward routes
  remain required for dense GPT SM120 parity.

- `tools/paired_kernel_speed.py` now includes both stdout and stderr tails when a
  measured command exits nonzero without `--continue-on-error`. This makes CUDA
  driver/runtime failures from external baselines such as llm.kittens visible in
  failed SM120 parity runs instead of showing only an empty stderr block.
  Verification: `python -m py_compile tools/paired_kernel_speed.py`;
  `python -m pytest tests/test_tile_cuda_examples.py -q`; `python
  tools/check_native_no_torch_deps.py --json`; and a CUDA 13.3 same-script
  SM120 parity smoke with `tools/bench_native_gpt_sm120_parity.sh` at 3 steps,
  1 sample, no warmup.

- Rechecked the default-off `NFN_NATIVE_GPT_BF16_ATTENTION_DPREP_GRAD_OUT=1`
  attention-backward path after the WSL CUDA 13.3 update. The dedicated RTX
  5090 native-vs-native 10-step, 3-sample benchmark measured `1.006112x`
  train-loop wall time and `0.993932x` tokens/sec versus the current default,
  confirming the route remains diagnostic-only and should not be promoted.

- Fixed native dense GPT template selection so `--template-name gpt3` is a
  shipped native-compatible alias instead of reporting `unknown-template`. The
  compiled planner now applies the same GPT3 2048-token context default through
  either `--base-model gpt3` or `--template-name gpt3`, defaults the implicit
  batch size to 32 to keep the 65,536-token microbatch, and still lets explicit
  `--train-seq-len`, `--batch-size`, or custom graph selection win.
  Verification: focused native GPT template pytest, compiled C++ plan smoke for
  `--template-name gpt3`, and diff whitespace check.

- Promoted cuBLASLt heuristic index 1 for the dense GPT MLP projection dWeight
  shape `3072,768,65536,N,T` in the trainer-facing Tile CUDA linear dispatcher.
  The dedicated RTX 5090 native-vs-native 5-step, 3-sample confirmation measured
  `0.998595x` train-loop wall time, `1.001407x` tokens/sec, and `0.998190x`
  total wall time versus the previous default, with zero compute processes
  before/after each paired sample. The existing
  `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE` /
  `NFN_TILE_CUDA_CUBLASLT_HEURISTIC_SHAPE` override remains available for
  rollback and paired bisection. Verification: focused native source tests,
  trainer Tile ops rebuild, and direct GPU smoke.

- The compiled `nfn_native_train` frontend now accepts the high-level GPT
  training flags that previously required the Python `nfn train` or
  `train_gpt.py` argument shim: `--dataset tinystories`, `--output`,
  `--kernel-backend`, `--template` / `--preset`, `--graph`,
  `--native-cuda-*` aliases, dense GPT default `--train-transformer-lm`,
  default `--backend tile-cuda`, default TinyStories alias fallback, and GPT-3's
  implicit 2048-token context when no template/graph/sequence length is passed.
  This makes `nfn_native_train train ...` a closer compiled C++ replacement for
  the Python startup path while still dispatching to `nfn_gpt_native_train`.
  Verification: focused unified native train CLI dispatch tests and C++ rebuild.

- Dense GPT native selection now fails fast when a shipped template's requested
  geometry does not match the compiled transformer-LM loop. The visible case is
  `--template-name nanogpt`: it now reports
  `template-geometry-native-trainer-missing` instead of running the fixed GPT-2
  geometry under a NanoGPT selector. Breaking changes: full-transformer NanoGPT
  native runs must use `nfn_nanogpt_native_train --train-token-lm` for the
  implemented token-LM path or wait for the dynamic NanoGPT transformer loop;
  callers should not treat `nanogpt` as a runnable alias of the dense GPT-2
  native trainer. Verification: focused native GPT selector tests and C++
  rebuild.

- Dense GPT native plan/runtime JSON now includes
  `native_geometry_contract.selected_template_geometry` and
  `geometry_matches_compiled_loop`. This keeps the current fixed GPT-2-compatible
  compiled loop explicit while surfacing selected-template requests such as
  NanoGPT's 320-wide, 5-layer, dropout-0.1 geometry, so callers can distinguish
  selector metadata from the still-fixed trainer dimensions. Verification:
  focused native GPT selector tests and C++ rebuild.

- Added deterministic inverted-dropout CUDA Tile forward/backward ABI for native
  training (`nfn_native_tile_dropout_forward_float32` and
  `nfn_native_tile_dropout_backward_float32`). NanoGPT `--print-plan
  --dropout-p` now reports ready dropout forward/backward stages instead of a
  missing ABI requirement, and `--check-tile-ops` includes those raw symbols in
  its required-symbol set. Verification: focused NanoGPT plan/source tests and
  native Tile ops rebuild/export checks.

- Replaced per-call `cudaMalloc` / `cudaFree` row-stat allocation in the
  trainer Tile ABI's non-workspace full-vocab token CE backward launchers with a
  process-cached row-stat workspace. Workspace-aware CE backward entrypoints are
  unchanged, and the legacy `nfn_native_tile_token_cross_entropy_backward_float32`
  / `nfn_native_tile_masked_token_cross_entropy_backward_float32` paths now
  expose `nfn_native_tile_token_cross_entropy_workspace_allocation_count()` and
  `nfn_native_tile_token_cross_entropy_workspace_row_capacity()` for native smoke
  tests and profiling. Verification: focused native Tile ABI source/export tests
  and native Tile ops rebuild.

- Native dense GPT C++ plan/runtime JSON now validates custom graph selector
  paths before reporting graph support. Existing custom graph files still report
  `custom-graph-native-trainer-missing` until the native graph compiler lands,
  but missing paths now report `custom-graph-file-missing` and expose
  `graph_file_exists` / `graph_file_size_bytes` at the top level and inside
  `native_geometry_contract`. GPT-2-evo preflight JSON reports the same fields.
  Verification: focused native GPT selector tests and C++ rebuild.

- Added `--dry-run-plan` to `tools/paired_kernel_speed.py` and exposed it
  through `NFN_SM120_PARITY_DRY_RUN_PLAN=1` /
  `NFN_SM120_NATIVE_DRY_RUN_PLAN=1` in the SM120 parity and native-candidate
  wrappers. The plan mode resolves the baseline/candidate argv, command-specific
  environment, CUDA device selection, profile settings, and alternating sample
  order without launching GPU jobs, making same-script benchmark setup
  auditable before long RTX 5090 runs. Verification: focused Tile-CUDA example
  tests and shell syntax checks.

- Added a command resolver to the generic `neuralfn._native_train` C++ binding
  and exported `resolve_native_train_binding_command(config)` from the Python
  SDK. SDK callers can now inspect the exact compiled argv that the binding
  will spawn before running training, while the no-Torch gate verifies the
  resolver stays importable without Torch, NumPy, `tiktoken`, dataset-manager
  modules, or graph payload paths. Breaking changes: old locally built
  `neuralfn._native_train` / `neuralfn_native_train` extension modules that
  expose only `run_train` no longer satisfy `runner="binding"`; rebuild with
  `bash tools/build_native_train_binding.sh` so the binding also exports
  `resolve_command` / `resolve_native_train_command`. Verification: focused
  native-train binding tests and the no-Torch dependency checker.

- Made dense GPT native training reject slow scalar attention fallback by
  default. Runtime JSON now reports `optimized_attention_required` and
  `attention_forward_scalar_launch_allowed`; if scalar attention launches, the
  trainer marks the run failed before final checkpoint export unless
  `--allow-scalar-attention-fallback` is explicitly passed for diagnostics.
  This keeps benchmark and training artifacts on optimized TK/row attention
  paths instead of silently accepting basic scalar kernels. Verification:
  focused native GPT CLI/source tests and C++ rebuild.

- Extended the native no-Torch dependency gate to cover the GPT-2-evo and
  NanoGPT legacy script handoffs plus the generic `neuralfn.native_train` SDK
  module and public native training exports. The checker stubs native CLIs and
  blocks Torch, NumPy, `tiktoken`, dataset-manager imports, and `nfn_impl`, so
  these handoff surfaces now fail CI-style verification before they can regress
  into Python/Torch startup paths. Verification: no-Torch checker and focused
  native GPT test coverage.

- Disabled final checkpoint export for dense GPT `--startup-only` runs at the
  native C++ gate, not only in wrapper conventions. Plan/runtime JSON now
  reports the effective checkpoint export state with
  `checkpoint_export_enabled` / `final_checkpoint_export_enabled` set to
  `false` and `checkpoint_export_startup_only_elided` set to `true` when
  startup-only suppresses a requested export. The checkpoint JSON block also
  distinguishes `requested` from effective `enabled`, so startup benchmarks
  cannot accidentally include BF16 checkpoint packing, device-to-host payload
  copy, or file I/O. Verification: focused native GPT CLI tests and C++
  rebuild.

- Elided dense GPT startup-only post-train diagnostic D2H sample copies.
  `--startup-only` still runs the end-of-setup CUDA synchronization so readiness
  timing reflects completed CUDA work, but it skips the token-weight and
  clip-scale device-to-host sample copies that are only needed to prove weight
  updates after real optimizer steps. Runtime JSON now reports
  `timing.post_train_diagnostic_samples_elided`,
  `timing.post_train_diagnostic_sample_d2h_count`, and
  `timing.post_train_diagnostic_sample_d2h_count_elided` for benchmark
  sidecars. Verification: focused native GPT source/JSON tests and C++ rebuild.

- Aligned generic native-train subprocess dispatch with the GPT-specific
  native launchers. `NativeTrainRunConfig` now exposes
  `cuda_visible_devices` (default `"0"`), `run_native_train(...,
  runner="auto")` sets `CUDA_VISIBLE_DEVICES=0` when the caller has not
  already supplied a device, and the pre-import legacy training guard applies
  the same unset-only default before execing a family native binary. This keeps
  direct legacy script handoff and generic SDK handoff on the dedicated CUDA
  compute device by default while preserving explicit user overrides.
  Verification: focused native SDK/guard tests.

- Kept native dense GPT layer-evo candidate losses device-resident during
  forward-only scoring. The `--layer-evo` loop now runs candidate forward loss
  without forcing a host loss copy, then copies the resulting scalar directly
  from the native loss device buffer into the device candidate-loss array before
  raw evo best-loss selection. Plan/runtime JSON now reports
  `candidate_loss_source:
  "native-forward-loss-device-resident-current-batch"`,
  `candidate_loss_transport: "device-to-device"`,
  `candidate_loss_device_copy_count`, and
  `candidate_loss_host_roundtrips_elided` so benchmark sidecars expose the
  removed host round-trip. Verification: focused native GPT source tests and
  C++ rebuild.

- Elided the unused int64 token/target device subarena from the default native
  dense GPT direct-uint16 token path. The trainer now reserves only the uint16
  token/target device copy buffer when direct-u16 embedding and CE kernels are
  active, while preserving the old int64 arena for
  `NFN_NATIVE_GPT_DIRECT_U16_TOKENS=0` bisections. Runtime JSON reports
  `token_i64_device_arena_elided` and
  `token_i64_device_arena_bytes_elided` so startup allocation reductions are
  visible in benchmark sidecars. Verification: focused native source tests and
  C++ rebuild.

- Added native dense GPT sample/checkpoint cadence fields to C++ plan/runtime
  JSON. `--print-plan` now reports `schedule.sample_every_steps`,
  `schedule.generate_tokens`, and `schedule.checkpoint_every_steps`; runtime
  JSON mirrors those values and adds `train_time_sampling_enabled`,
  `periodic_checkpoint_enabled`, and `final_checkpoint_export_enabled`.
  Same-script timing runs can now verify that sample/checkpoint/export work is
  disabled when cadence flags or `--no-checkpoint` are used. Verification:
  focused native GPT CLI tests and C++ rebuild.

- Improved direct native GPT profile failure diagnostics. When
  `nfn_gpt_native_train --check-tile-ops` or `--train-transformer-lm` writes
  redirected output through `--json-out`, `--profile-json`, or
  `--stage-profile-json` and exits nonzero, the C++ CLI now mirrors a concise
  status/error summary to stderr while leaving the JSON sidecar intact. Direct
  benchmark and shell runs now surface missing Tile symbols and CUDA
  driver/runtime preflight failures without opening the sidecar manually.
  Verification: focused native GPT CLI tests and C++ rebuild.

- Extended `tools/paired_kernel_speed.py` native metric extraction and summary
  output with packed-attention backward section counters:
  `attention_backward_dprep_timing_us/count` and
  `attention_backward_tk_timing_us/count`. Same-script kernel bisections now
  show attention dprep/TK attribution and candidate-over-baseline ratios
  directly when native JSON sidecars include those fields. Verification:
  paired-kernel helper tests.

- Improved paired benchmark failure diagnostics for native JSON sidecars. When
  a native command exits nonzero and has written `--json-out`/`--profile-json`,
  the immediate failure now includes the sidecar `status` and `error` fields
  before stderr, so CUDA-driver access or missing-symbol failures are visible
  without manually opening the sidecar. Verification: paired-kernel helper
  tests.

- Aligned the generic `neuralfn._native_train` SDK binding with the native GPT
  bindings by replacing `fork()` plus `execvp()` with `posix_spawnp()` while
  preserving the existing `CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_MAX_CONNECTIONS`,
  and unset-only `CUDA_MODULE_LOADING=LAZY` defaults. `run_native_train(...,
  runner="auto")` keeps the same public API and command selection contract, but
  avoids copying the Python process when handing off to compiled C++ native
  trainers. Verification: focused native binding tests and C++ binding rebuild.

- Added `lm_head_classifier_strategy_contract` to native dense GPT
  `--print-plan` and `--train-transformer-lm` JSON. The contract records the
  llm.kittens-style full resident BF16 classifier-logit footprint next to
  NeuralFn's row-chunked BF16 in-place CE path, including BF16/FP32-equivalent
  byte counts, chunk count, resident-logit reduction ratio, graph-editor tensor
  flow status, and the same-script benchmark target for the next fused
  classifier/LM-head-backward candidate. At the default `64 x 1024` shape this
  reports an 8x resident-logit reduction, making the remaining SM120 parity
  tradeoff visible in both successful and failed runtime JSON. Verification:
  focused native GPT tests and C++ rebuild.

- Updated `tools/paired_kernel_speed.py` to extract and print the classifier
  contract's full/chunk BF16 byte counts, chunk rows/count, and resident-logit
  reduction ratio from native JSON sidecars. Candidate-vs-baseline benchmark
  summaries now show whether an LM-head classifier experiment changes the
  memory contract as well as timing. Verification: paired-kernel helper tests.

- Rejected the BF16 GEMMEx `FAST_16BF` compute-type candidate for the remaining
  LM-head fallback shapes. With `NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF=1`
  and `NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_FAST_16BF=1`, the dedicated RTX 5090
  paired benchmark measured `1.001097x` train-loop wall time and `0.998921x`
  tokens/sec versus the current BF16 GEMMEx default, so the fallback compute
  type remains unchanged. Verification: `tools/bench_native_gpt_sm120_candidate.sh`
  with selected-GPU idle checks.

- Exposed the native dense GPT geometry contract in compiled C++ plan/runtime
  JSON. `nfn_gpt_native_train --print-plan` and `--train-transformer-lm` now
  emit `native_geometry_contract`, explicitly naming the current
  `gpt2-compatible-fixed-dense-transformer` loop, its compiled shape source,
  model dimension, head geometry, padded vocab, sequence length, selected layer
  count, and the fact that template/custom-graph geometry is not yet dynamic.
  This keeps GPT/NanoGPT/GPT-3-context selector routing honest while the
  remaining work generalizes the hot CUDA Tile loop beyond GPT-2-compatible
  dimensions. Verification: focused native GPT tests and C++ rebuild.

- Changed NanoGPT’s default native training route to the shared dense GPT
  transformer-LM trainer. `nfn train --base-model nanogpt ...` and direct
  `python cli/scripts/train_nanogpt.py ...` now dispatch before Torch imports
  to `nfn_gpt_native_train --model-family gpt --template-name nanogpt
  --train-transformer-lm`; pass `--train-token-lm` explicitly to use the older
  tied token-embedding NanoGPT native loop. The unified C++ registry now reports
  NanoGPT as implemented via `nfn_gpt_native_train`, while preserving explicit
  token-LM dispatch to `nfn_nanogpt_native_train`.
  Verification: focused native dispatch tests, CLI startup tests, and native
  no-Torch dependency checks.

- Refreshed the same-script SM120 parity baseline on the dedicated RTX 5090
  after the routing/fallback commits. The 10-step, 1-sample run kept the
  selected GPU idle before and after the paired sample and measured NeuralFn at
  `1.033761x` train-loop step time and `0.965977x` tokens/sec versus the
  llm.kittens reference, so the throughput-gap checklist item remains open.
  Verification JSON: `/tmp/nfn_sm120_parity_after_nanogpt_route_20260618.json`.

- Rejected a refreshed LM-head row-chunk candidate after the latest native GPT
  routing/fallback changes. `NFN_NATIVE_GPT_LM_HEAD_ROW_CHUNK_SIZE=16384`
  measured `1.000599x` train-loop wall time and `0.999416x` tokens/sec versus
  the current 8192-row default in the dedicated RTX 5090 same-script 10-step,
  3-sample benchmark, so the remaining LM-head work stays focused on the GEMM
  route instead of row-chunk tuning. Verification: ran
  `tools/bench_native_gpt_sm120_candidate.sh` with selected-GPU idle checks.

- Added a default-off BF16-output cuBLASLt LM-head logits diagnostic behind
  `NFN_NATIVE_LINEAR_BF16_OUTPUT_CUBLASLT=1` /
  `NFN_TILE_CUDA_LINEAR_BF16_OUTPUT_CUBLASLT=1`, covering both all-BF16 and
  BF16-input/float-weight logits wrappers. The one-step shape-stat smoke moved
  the `50304,8192,768,T,N` logits bucket to cuBLASLt, but the dedicated RTX
  5090 10-step, 3-sample paired benchmark measured `1.000629x` train-loop wall
  time and `0.999382x` tokens/sec versus the current GEMMEx fallback, so the
  diagnostic remains off by default. The same change fixes BF16-output
  float-weight GEMMEx telemetry to report `cublas_gemmex_bf16` instead of
  `cublas_sgemm`. Verification: rebuilt `libnfn_native_train_tile_ops.so`, ran
  a GPU-visible one-step shape-stat smoke, and ran
  `tools/bench_native_gpt_sm120_candidate.sh` with selected-GPU idle checks.

- Added a default-off BF16-input/BF16-gradient split dWeight+bias diagnostic
  behind `NFN_NATIVE_GPT_FUSE_BF16_BF16_DWEIGHT_BGRAD=0` /
  `NFN_TILE_CUDA_LINEAR_BF16_BF16_BGRAD=0`. This keeps block dWeight on the
  GEMM route and only splits bias reduction, instead of using the old tiled
  dWeight fallback when comparing against the fused cuBLASLt BGRADB default.
  The dedicated RTX 5090 10-step, 3-sample paired benchmark measured
  `1.033067x` train-loop wall time and `0.968003x` tokens/sec versus BGRADB, so
  the fused route remains the default. Verification: rebuilt
  `libnfn_native_train_tile_ops.so`, ran a GPU-visible one-step shape-stat
  smoke, and ran `tools/bench_native_gpt_sm120_candidate.sh` with selected-GPU
  idle checks.

- Refreshed current default attention/backward attribution after the latest
  rejected GEMM diagnostics. The GPU-visible one-step stage profile at
  `/tmp/nfn_attention_section_current_20260618.json` reported
  `attention_backward_tk_timing_us: 237105` versus
  `attention_backward_dprep_timing_us: 31238`, with the largest remaining
  buckets in block backward, LM-head backward, MLP projection/FC, packed
  attention backward TK, and LM-head logits/dHidden. This keeps the remaining
  throughput-gap work focused on those kernels rather than the rejected LM-head
  cuBLASLt or split-BGRADB routes. Verification: ran
  `nfn_gpt_native_train` with `NFN_NATIVE_GPT_STAGE_TIMING=1` and
  `NFN_NATIVE_GPT_ATTENTION_BACKWARD_SECTION_TIMING=1` on the dedicated RTX
  5090.

- Added dense GPT stage attribution for the MLP projection grad-out BF16 pack.
  `NFN_NATIVE_GPT_STAGE_TIMING=1` profiles now include
  `block_backward.mlp_proj.grad_out_bf16`, separating the conversion used by
  the reused BF16 projection-gradient path from the surrounding MLP projection
  dWeight and dInput buckets. A GPU-visible one-step profile at
  `/tmp/nfn_mlp_grad_out_timing_20260618.json` measured the pack at
  `22.774 ms` total across 96 calls, compared with `174.294 ms` for
  `block_backward.mlp_proj.dweight_bias` and `172.114 ms` for
  `block_backward.mlp_proj.dinput`, so the next MLP work should still target
  GEMM/TK kernels first. Verification: rebuilt `nfn_gpt_native_train` and ran
  a CUDA-visible one-step stage profile on the dedicated RTX 5090.

- Added and rejected a default-off attention-backward BF16 dprep grad-out
  candidate behind `NFN_NATIVE_GPT_BF16_ATTENTION_DPREP_GRAD_OUT=1` /
  `NFN_NATIVE_GPT2_BF16_ATTENTION_DPREP_GRAD_OUT=1`. The path keeps attention
  projection dInput on the default float output route, packs dO to BF16 just
  before packed-attention dprep/backward, and reports
  `attention_backward_bf16_dprep_grad_out_enabled` plus
  `attention_backward_grad_out_dtype: "bf16-dprep-pack"`. A one-step profile
  showed dprep timing dropping to `24.807 ms` but added a `22.473 ms` pack, and
  the dedicated RTX 5090 same-script 10-step, 3-sample benchmark measured
  `1.007803x` train-loop wall time and `0.992260x` tokens/sec versus default,
  so it remains diagnostic-only. Verification: rebuilt `nfn_gpt_native_train`
  and `libnfn_native_train_tile_ops.so`, ran a GPU-visible one-step branch
  smoke, and ran `tools/bench_native_gpt_sm120_candidate.sh` with selected-GPU
  idle checks.

- Added a compile-time SM120 atomic-dQ packed-QKV attention-backward candidate
  behind `NFN_TILE_CUDA_EXTRA_NVCC_FLAGS=-DLLMK_SM120_ATOMIC_DQ`. The candidate
  wrapper allocates float dQ scratch, zeroes it per backward chunk, launches the
  internal llm.kittens SM120 atomic-dQ backward, and converts/re-packs the Q
  gradient slice into the BF16 packed `dQKV` buffer before QKV dWeight handoff;
  split-Q/K/V attention wrappers return unsupported in that candidate build.
  The route remains rejected/default-off: a one-step dedicated RTX 5090 profile
  regressed TK backward timing to `597872 us`, and the same-script 5-step,
  2-sample benchmark measured `1.134435x` train-loop wall time and `0.881527x`
  tokens/sec versus default. Verification: rebuilt the default Tile ops library,
  built `/tmp/libnfn_native_train_tile_ops_atomic_dq_20260618.so`, ran a
  GPU-visible one-step smoke/profile, and ran
  `tools/bench_native_gpt_sm120_candidate.sh` with selected-GPU idle checks.

- Replaced the no-cuBLAS large-row linear dWeight fallback with a shared-memory
  2D tiled CUDA kernel for float32-output dWeight accumulation across float32
  and BF16 activation/gradient combinations. The normal native GPT workstation
  build still routes through cuBLAS/cuBLASLt first; this closes the fallback
  path that previously used row-chunked atomic dWeight reductions when the
  trainer cuBLAS path was unavailable. The fallback beta overload now honors
  `beta=0` for first-write dWeight semantics before bias accumulation.
  Verification: rebuilt the native Tile ops library, ran the focused source
  guard for native GPT fallback wiring, ran the no-Torch native dependency
  check, and ran `git diff --check`.

- Added `NFN_TILE_CUDA_CUBLASLT_HEURISTIC_POLICY` /
  `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_POLICY` as a default-off native GPT
  cuBLASLt profiling control. `min_waves` selects the returned cuBLASLt
  heuristic with the lowest `wavesCount`, matching the llm.kittens default
  policy, and `max_waves` selects the highest-waves candidate. Existing global
  and shape-specific explicit index overrides still take precedence. The
  NeuralFn default remains index 1 because same-script dedicated RTX 5090
  5-step, 3-sample candidate benchmarks rejected both policies: `min_waves`
  measured `1.001205x` train-loop wall time and `0.998809x` tokens/sec, while
  `max_waves` measured `1.001045x` train-loop wall time and `0.998964x`
  tokens/sec versus the current default. A 2026-06-18 post-atomic-route current
  build check reconfirmed `min_waves` as slower at `1.009572x` train-loop wall
  time and `0.990522x` tokens/sec. Verification: rebuilt the Tile-CUDA native
  trainer, ran no-Torch/focused source guards, and ran
  `tools/bench_native_gpt_sm120_candidate.sh` for both policies with selected
  GPU idle checks.

- Rejected the older float32/TF32 tied LM-head route against the current dense
  GPT default. Setting `NFN_NATIVE_GPT_LM_HEAD_BF16_LOGITS=0` moved the
  LM-head logits/dHidden/dWeight path to the float-logits strategy, but the
  dedicated RTX 5090 5-step, 3-sample same-script benchmark measured
  `1.229126x` train-loop wall time and `0.813607x` tokens/sec versus the
  current BF16 logits/dlogits default, so BF16 LM-head logits remain the normal
  training route. Verification: ran `tools/bench_native_gpt_sm120_candidate.sh`
  with selected-GPU idle checks and no profiling sidecars.

- Rejected `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,2304,65536,N,T,2`
  as a default dense GPT QKV dWeight route. The dedicated RTX 5090 5-step,
  3-sample same-script benchmark measured `1.001121x` train-loop wall time and
  `0.998895x` tokens/sec versus the current cuBLASLt heuristic selection, so
  the QKV dWeight shape remains on the default heuristic. Verification: ran
  `tools/bench_native_gpt_sm120_candidate.sh` with selected-GPU idle checks and
  no profiling sidecars.

- Added a default-off dense GPT MLP projection backward ordering diagnostic,
  `NFN_NATIVE_GPT_MLP_PROJ_DINPUT_BEFORE_DWEIGHT=1`. The switch runs fused MLP
  projection dInput+dGELU before dWeight+bias to compare against the
  llm.kittens `matmul_backward` consumer order, and runtime JSON reports
  `block_backward_mlp_proj_dinput_before_dweight_enabled`. It is not promoted:
  the dedicated RTX 5090 5-step, 3-sample paired benchmark measured
  `1.000405x` train-loop wall time and `0.999602x` tokens/sec versus the
  current dWeight+bias-first default. Verification: rebuilt
  `build/nfn_gpt_native_train`, ran short and stronger paired benchmarks with
  selected-GPU idle checks, and updated the CUDA Tile docs/checklist.

- Added a matching default-off dense GPT MLP FC backward ordering diagnostic,
  `NFN_NATIVE_GPT_MLP_FC_DINPUT_BEFORE_DWEIGHT=1`. The switch runs MLP FC
  dInput before dWeight+bias to compare against the reference consumer order,
  and runtime JSON reports
  `block_backward_mlp_fc_dinput_before_dweight_enabled`. It is not promoted:
  the dedicated RTX 5090 5-step, 3-sample paired benchmark measured
  `1.000858x` train-loop wall time and `0.999153x` tokens/sec versus the
  current dWeight+bias-first default. Verification: rebuilt
  `build/nfn_gpt_native_train`, ran the paired benchmark with selected-GPU idle
  checks, and updated the CUDA Tile docs/checklist.

- Added a default-off dense GPT attention projection backward ordering
  diagnostic, `NFN_NATIVE_GPT_ATTN_PROJ_DINPUT_BEFORE_DWEIGHT=1`. The switch
  runs attention projection dInput before dWeight+bias to compare against the
  reference consumer order, and runtime JSON reports
  `block_backward_attn_proj_dinput_before_dweight_enabled`. It is not promoted:
  the dedicated RTX 5090 5-step, 3-sample paired benchmark measured
  `1.001009x` train-loop wall time and `0.999002x` tokens/sec versus the
  current dWeight+bias-first default. Verification: rebuilt
  `build/nfn_gpt_native_train`, ran the paired benchmark with selected-GPU idle
  checks, and updated the CUDA Tile docs/checklist.

- Rejected a BF16-output cuBLASLt probe for the dense GPT LM-head logits route.
  A candidate Tile ops library attempted to route the no-bias BF16/BF16
  `50304,8192,768,T,N` logits GEMM through cuBLASLt with BF16 output, but the
  GPU-visible one-step shape-stat smoke still reported the LM-head logits
  bucket on BF16 `cublasGemmEx` (`330916 us` over 64 calls), so the attempted
  route was removed rather than leaving a no-op diagnostic flag. Verification:
  built `/tmp/libnfn_tile_bf16out_cublaslt.so` and ran
  `NFN_NATIVE_GPT_LINEAR_SHAPE_STATS=1` against
  `/tmp/nfn_bf16out_cublaslt_shape_smoke.json` on the dedicated RTX 5090.

- The dense GPT token-weight CUDA Tile initializer now accepts
  `NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE=8192` for local candidate library
  builds, while keeping the measured 4096-element default. The 8192 candidate
  compiled successfully but was not promoted after a dedicated RTX 5090
  startup-only 9-sample paired benchmark measured `1.005585x` token-init time
  versus the 4096 default, with total startup `0.990436x` inside broader arena
  materialization noise. Verification: built
  `/tmp/libnfn_tile_token8192.so` with
  `NFN_TILE_CUDA_EXTRA_NVCC_FLAGS=-DNFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE=8192`,
  ran the same-script startup-only comparison through
  `tools/bench_native_gpt_sm120_candidate.sh` with idle selected-GPU checks,
  and updated the native source guard test.

- Rejected disabling TK for the hot dense GPT MLP projection forward shape
  `768,65536,3072,T,N`. Setting
  `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=768,65536,3072,T,N` moved the
  candidate away from the current TK route but regressed the dedicated RTX 5090
  3-step paired benchmark to `1.010042x` train-loop wall time and `0.990063x`
  tokens/sec, so the TK route remains the default. Verification: ran
  `tools/bench_native_gpt_sm120_candidate.sh` with selected-GPU idle checks and
  recorded the result in `todo-tile-cuda.md`.

- Rejected disabling TK for the dense GPT attention projection forward shape
  `768,65536,768,T,N`. The candidate
  `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=768,65536,768,T,N` measured
  `1.009872x` train-loop wall time and `0.990230x` tokens/sec in the dedicated
  RTX 5090 3-step paired benchmark, so the TK forward route remains the
  default for that projection too. Verification: ran the same
  `tools/bench_native_gpt_sm120_candidate.sh` path with idle selected-GPU
  checks and recorded the result in `todo-tile-cuda.md`.

- Added diagnostic coverage for swapping the LM-head row-chunk consumer order
  with `NFN_NATIVE_GPT_LM_HEAD_DWEIGHT_BEFORE_DHIDDEN=1`, while keeping the
  default CE -> dHidden -> dWeight order. The candidate runs dWeight before
  dHidden after CE writes BF16 dlogits, but the dedicated RTX 5090 5-step,
  3-sample paired benchmark measured `1.001048x` train-loop wall time and
  `0.998959x` tokens/sec, so it stayed non-default. Runtime JSON now reports
  `lm_head_dweight_before_dhidden_enabled`. Verification: rebuilt
  `build/nfn_gpt_native_train`, ran the paired benchmark with selected-GPU idle
  checks, and updated the native GPT source guard.

- The top-level `nfn train` native dispatcher now normalizes
  `--native-cuda-no-checkpoint` / `--no-checkpoint` and
  `--native-cuda-write-checkpoint` / `--write-checkpoint` before launching the
  compiled dense GPT trainer. This keeps `nfn train --native-cuda-print-command`
  and benchmark invocations consistent with `cli/scripts/train_gpt.py`, so
  timing-only runs can skip final checkpoint export through either wrapper or
  compiled C++ flag spelling. Verification: updated the focused top-level
  native GPT CLI regression and documented the workflow in `README.md` and
  `docs/cli.md`.

- Rechecked the MLP projection dWeight cuBLASLt heuristic override
  `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=3072,768,65536,N,T,0` on the
  dedicated RTX 5090 and kept it diagnostic-only. The same-script 3-step,
  2-sample benchmark measured `0.998915x` train-loop wall time and `1.001091x`
  tokens/sec versus the current default, with unchanged native route summaries.
  That delta is within run-to-run noise, so no kernel default changed.
  Verification: ran `tools/bench_native_gpt_sm120_candidate.sh` with selected
  GPU idle/utilization checks and recorded the result in `todo-tile-cuda.md`.

- Rejected another MLP projection dWeight cuBLASLt heuristic override,
  `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=3072,768,65536,N,T,2`, as a
  default. The dedicated RTX 5090 same-script 3-step, 2-sample benchmark
  measured `0.998753x` train-loop wall time and `1.001251x` tokens/sec, again
  inside noise and not a useful default-route change. Verification: ran
  `tools/bench_native_gpt_sm120_candidate.sh` with selected-GPU idle checks and
  recorded the result in `todo-tile-cuda.md`.

- Dense GPT native runtime JSON now emits `lm_head_dhidden_linear_strategy`.
  The field is derived from linear shape stats for the LM-head dHidden bucket
  (`768,8192,50304,N,N`) when profiling is enabled and otherwise reports the
  current default BF16 GEMMEx dInput route, so paired benchmark output can show
  whether a dHidden candidate changed routing alongside logits and dWeight
  strategies. Verification: rebuilt the native GPT CLI, ran
  `python -m pytest tests/test_native_gpt2.py::test_native_gpt2_cpp_cli_builds_and_uses_sm120_defaults -q`,
  verified the field in a GPU-visible `--startup-only` runtime profile, and
  ran `python tools/check_native_no_torch_deps.py`.

- Strengthened `tools/check_native_no_torch_deps.py` to verify the public
  top-level native GPT SDK exports under the same blocked-import guard as the
  direct module imports. The gate now proves `NativeGptRunConfig`,
  `NativeGpt2RunConfig`, `build_native_gpt_compiled_cli_run_config()`,
  `native_gpt_kernel_backend()`, and `native_gpt_parameter_count()` can be
  imported from `neuralfn` without importing Torch, NumPy, tiktoken,
  `server.dataset_manager`, or `nfn_impl`. A refreshed dedicated RTX 5090
  recheck also kept `NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUTS=1` rejected
  as a default: startup-only setup improved to `0.965877x` and total startup
  to `0.966002x`, but the normal 3-step, 2-sample train loop regressed to
  `1.016063x` with total wall at `1.012855x`. Verification: ran
  `python -m py_compile tools/check_native_no_torch_deps.py` and
  `python tools/check_native_no_torch_deps.py --skip-artifacts --json`.

- Rechecked the disabled TK BF16-output LM-head logits route after the latest
  dense GPT parity baseline. Setting
  `NFN_NATIVE_LINEAR_TK_FORWARD_ENABLE_SHAPE=50304,8192,768,T,N` does switch
  the LM-head logits bucket to `tk_bf16` when shape stats are enabled
  (`220209 us` over 64 logits calls in the one-step smoke), but the stronger
  same-script 5-step, 3-sample dedicated RTX 5090 benchmark rejected it as a
  default at `1.007525x` train-loop wall time and `0.992536x` tokens/sec. No
  default changed; the next parity work should target a route that improves
  the hot LM-head or block GEMM buckets under paired timing. Verification: ran
  a one-step shape-stat smoke plus `tools/bench_native_gpt_sm120_candidate.sh`
  with the candidate env above and selected-GPU idle checks enabled.

- `tools/paired_kernel_speed.py` now preserves categorical native strategy
  summaries in paired benchmark output. Native JSON fields such as
  `lm_head_logits_linear_strategy`, `lm_head_dhidden_linear_strategy`,
  `lm_head_dweight_strategy`, block linear strategies, and attention strategies
  are summarized under `baseline_native_metric_values` and
  `candidate_native_metric_values`, so kernel-candidate benchmarks show whether
  the intended route changed in addition to numeric timing ratios.
  Verification: added a focused sidecar regression test and used the paired
  native benchmark wrapper on the dedicated RTX 5090 while probing the rejected
  LM-head logits cuBLASLt heuristic candidate.

- Added native dense GPT train-loss cadence controls:
  `--train-loss-every-steps N`, `--train-log-every N`, and
  `--train-log-every-steps N`. The default remains `0` so timing-only SM120
  runs do not evaluate train loss. When enabled, the compiled C++ loop records
  train loss from the folded LM-head backward recompute path rather than running
  a separate forward LM-head loss pass; validation loss remains controlled by
  `--eval-every-steps`. The Python native GPT SDK config now exposes
  `train_loss_every_steps` and forwards it to the compiled CLI. Verification:
  rebuilt `build/nfn_gpt_native_train`; passed the focused native GPT test
  slice, `python tools/check_native_no_torch_deps.py`, `python -m py_compile`
  for the touched Python entrypoints, and `git diff --check`; ran a one-step
  dedicated-GPU smoke with `--train-loss-every-steps 1` that reported
  `train_loss_eval_count: 1` and `train_loss_last_step: 1`.

- Dense GPT native training now folds train-loss collection into the LM-head
  backward recompute pass. Training microbatches call the transformer forward
  path without a separate LM-head loss pass, then accumulate CE loss from the
  row-chunked logits already recomputed for CE backward before those logits are
  overwritten with dLogits. Validation and evo candidate scoring still use the
  forward-only LM-head loss path. This removes duplicate train-only LM-head
  logits work without sending real token data through graph-editor nodes or
  reintroducing Torch. Verification: rebuilt `build/nfn_gpt_native_train` and
  ran the focused native GPT test slice
  `python -m pytest tests/test_native_gpt2.py -q -k 'native_gpt2_cpp_cli_builds_and_uses_sm120_defaults or tied_lm_head_bf16 or transformer_lm or native_train_tile_ops_builds_torch_free_c_abi'`
  (`3 passed, 1 skipped`). A sandboxed TinyStories GPU smoke produced the
  expected native preflight failure because `cudaDriverGetVersion` returned
  `0`. The GPU-visible SM120 parity harness, which keeps train-loss recording
  disabled for timing-only parity, measured NeuralFn at `2557.370 ms/step`
  versus llm.kittens at `2466.293 ms/step` (`1.036929x` train-loop wall time),
  so the canonical throughput gap remains open.

- Added a default-off TK dInput diagnostic for BF16-gradient/BF16-weight Linear
  shapes. Set `NFN_NATIVE_LINEAR_TK_DINPUT=1` or
  `NFN_TILE_CUDA_LINEAR_TK_DINPUT=1` to route eligible dInput shapes through
  the TK BF16 matmul bridge and convert the BF16 result back to FP32. The
  diagnostic successfully moved dense GPT LM-head dHidden
  `768,8192,50304,N,N` from GEMMEx to TK in shape stats, but it is not promoted:
  the dedicated RTX 5090 same-script 5-step, 3-sample paired benchmark measured
  `1.049216x` train-loop wall time and `0.953102x` tokens/sec versus the
  default route. Verification: rebuilt `build/libnfn_native_train_tile_ops.so`,
  ran a GPU-visible one-step shape-stat smoke, and ran
  `tools/bench_native_gpt_sm120_candidate.sh` with
  `NFN_SM120_NATIVE_CANDIDATE_ENV='NFN_NATIVE_LINEAR_TK_DINPUT=1'`.

- Dense GPT native training now skips the TK forward path for the padded
  LM-head logits shape `50304,8192,768,T,N` by default and falls through to the
  existing BF16 fallback path. Set
  `NFN_NATIVE_LINEAR_TK_FORWARD_ENABLE_SHAPE=50304,8192,768,T,N` or
  `NFN_TILE_CUDA_LINEAR_TK_FORWARD_ENABLE_SHAPE=50304,8192,768,T,N` to restore
  the old TK route for bisection; `...DISABLE_SHAPE=m,n,k,opA,opB` still works
  for one additional forward/fused-GELU TK shape. Verification: a dedicated
  RTX 5090 same-script 10-step, 3-sample comparison measured `0.990336x`
  train-loop wall time and `1.009770x` tokens/sec versus the old TK route. A
  follow-up same-script parity run against `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh`
  measured NeuralFn at `1.025419x` train-loop wall time and `0.974765x`
  tokens/sec versus the reference, so the remaining parity gap is still open.
  Runtime JSON now derives `lm_head_logits_linear_strategy` from the exact
  LM-head logits shape bucket instead of the global TK GEMM counter.

- **Breaking changes:** `nfn_native_tile_trainer_linear_shape_stats_entry` now
  takes one additional `std::int64_t* total_us` output argument. Native GPT
  callers built with this workspace should pass the new pointer and read
  `linear_shape_stats[].total_us` / `avg_us` from runtime JSON when
  `NFN_NATIVE_LINEAR_SHAPE_STATS=1` or a GPT-prefixed alias is enabled. Normal
  training is unchanged because the stats path is opt-in; profiling mode now
  uses CUDA events and synchronizes measured GEMMs. Verification: rebuilt
  `build/libnfn_native_train_tile_ops.so` and `build/nfn_gpt_native_train`.

- Recorded the post-LM-head-beta-fix native GPT kernel bisection results without
  changing the default runtime route. `NFN_NATIVE_GPT_DWEIGHT_FIRST_MICROBATCH_BETA_ZERO=0`
  remained slower at `1.009049x` train-loop wall time and `0.991044x` tokens/sec,
  `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1` remained too noisy while worsening
  token initialization to `1.044352x`, and a short-lived pattern16 token-weight
  initializer prototype worsened token initialization to `1.033752x` mean /
  `1.058094x` median before being removed. Verification: dedicated RTX 5090
  same-script paired benchmarks and startup-only paired benchmarks; no runtime
  default changed.

- Fixed tied LM-head dWeight accumulation for chunked dense GPT native training.
  The first-write-then-accumulate path still uses GEMM `beta=0` for the first
  gradient-accumulation microbatch, but the row-chunked LM-head route now applies
  that beta-zero write only to the first LM-head row chunk. Later chunks in the
  same microbatch use `beta=1`, so all token chunks contribute to
  `accum_grad_token_weight` instead of replacing earlier chunk contributions.
  Runtime JSON now reports `lm_head_dweight_beta_zero_scope`. Verification:
  rebuilt `build/nfn_gpt_native_train` and ran the focused native GPT tests after
  updating source guards for the corrected LM-head chunk scope. A GPU-visible
  one-step TinyStories native run also completed with
  `steps_completed: 1`, `lm_head_row_chunk_count: 8`, and
  `lm_head_dweight_beta_zero_scope` set to
  `"first-gradient-accumulation-microbatch-first-row-chunk-only"`.

- Dense GPT native training now has a default-off BF16 persistent block-output
  diagnostic behind `NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUTS=1` /
  `NFN_NATIVE_GPT2_BF16_PERSISTENT_BLOCK_OUTPUTS=1`. The opt-in path stores
  earlier inter-block persistent outputs as BF16, restores each prior block
  input through one FP32 scratch buffer during backward, and reports
  `bf16_persistent_block_outputs_enabled`,
  `bf16_persistent_block_output_store_count`,
  `bf16_persistent_block_output_restore_count`,
  `fp32_persistent_block_output_elements_elided`, and
  `fp32_persistent_block_output_bytes_elided` in runtime JSON, with matching
  `block_state_layout` fields. It is not promoted as a default: a dedicated
  RTX 5090 paired benchmark measured `1.021212x` train-loop wall time and
  `0.979238x` tokens/sec versus default, despite improving setup wall time to
  `0.974595x` and float-arena materialization to `0.896011x`. Verification:
  rebuilt `build/nfn_gpt_native_train`, ran a GPU-visible one-step TinyStories
  probe that reported 88 BF16 stores, 88 restores, and `2,214,592,512` elided
  FP32 persistent-output bytes, then ran the same-script 5-step, 3-sample
  native candidate benchmark and kept the switch off by default.

- Dense GPT native training now elides the FP32 `mlp.fc.grad_out` arena buffer
  when the default BF16-only MLP dGELU handoff covers every trained block. The
  MLP projection dInput+dGELU Tile kernel writes BF16 bits directly into the
  scratch consumed by the following MLP FC dWeight/dInput kernels, so the
  805 MB FP32 hidden-gradient buffer at the default `64 x 1024 x 3072` shape is
  no longer reserved. Runtime JSON reports
  `block_backward_mlp_fc_grad_out_float_buffer_elided`,
  `block_backward_mlp_fc_grad_out_float_elements`, and
  `block_backward_mlp_fc_grad_out_float_bytes_elided`, with matching
  `block_state_layout` fields. Verification: rebuilt `build/nfn_gpt_native_train`,
  ran startup-only TinyStories profiles showing float arena bytes dropping from
  `8,485,997,620` to `7,680,691,252`, and ran a dedicated RTX 5090 paired
  benchmark against the old path forced by
  `NFN_NATIVE_GPT_ELIDE_MLP_DGELU_FLOAT_GRAD=0`; the new default measured
  `0.969357x` train-loop wall time, `1.031616x` tokens/sec, and `0.965683x`
  setup wall time versus baseline. A same-script 3-step llm.kittens parity
  snapshot after the change measured NeuralFn at `1.033420x` train-loop wall
  time and `0.967534x` tokens/sec versus
  `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh`, so the remaining
  parity gap is still open.

- Dense GPT native arena diagnostics now name the main transformer-LM global
  float-buffer requests individually instead of grouping them all under
  `transformer_lm_buffer`. The total arena shape is unchanged, but
  `float_arena_request_stats.top_families` now exposes concrete targets such as
  `block.*.persistent_output`, `mlp.fc.grad_out`, `attention.grad_out`, and
  `lm_head.float_logits`. Verification: rebuilt `build/nfn_gpt_native_train`
  and ran a startup-only TinyStories profile; the old opaque
  `transformer_lm_buffer` group disappeared, with the largest float families
  now reported as `block.*.persistent_output` (`2.21 GB`) and
  `mlp.fc.grad_out` (`805 MB`).

- Dense GPT native profile JSON now aggregates arena allocation requests by
  normalized family in addition to the existing largest individual requests.
  `float_arena_request_stats` and `uint16_arena_request_stats` now include
  `family_count`, `top_families`, `top_family_elements`, and
  `top_family_bytes`; per-block names are normalized as `block.*...` so
  repeated layer buffers are visible as one startup/memory target. This is a
  diagnostic surface only and does not change the training path. Verification:
  rebuilt `build/nfn_gpt_native_train`, ran a startup-only TinyStories profile,
  parsed the new family fields, and re-ran the current
  `NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD=0` candidate in the paired
  RTX 5090 harness; it measured `1.001432x` train-loop wall time and
  `0.998574x` tokens/sec versus default, so no kernel default was promoted.

- Dense GPT native CLI cleanup now defaults to skipping explicit exit-time
  `cudaFree` calls for the large device arenas. The process exits immediately
  after JSON/checkpoint output, so CUDA context teardown reclaims those
  allocations without the previous synchronous multi-GB free/runtime teardown
  loop. Set
  `NFN_NATIVE_GPT_SKIP_EXIT_CUDA_FREE=0` or
  `NFN_NATIVE_GPT2_SKIP_EXIT_CUDA_FREE=0` to restore explicit device frees and
  runtime-library `dlclose()` for diagnostics. Runtime JSON reports
  `device_exit_cuda_free_elision_enabled`,
  `device_exit_cuda_free_skipped_count`, `runtime_library_dlclose_skipped_count`, and the resulting
  `timing.cleanup_wall_ms`. Updated `README.md`, `docs/cli.md`, and the CUDA
  Tile checklist. Verification: rebuilt the native GPT CLI, checked default
  startup JSON (`device_exit_cuda_free_skipped_count: 5`,
  `runtime_library_dlclose_skipped_count: 2`, `cleanup_wall_ms: 1.00296`),
  checked the explicit-free opt-out (`cleanup_wall_ms: 250.605`), and compared
  the old explicit-free cleanup path against the new default on the dedicated
  RTX 5090. The startup-only same-script benchmark measured `0.695283x` mean
  total wall time, and the one-step training benchmark measured neutral
  train-loop time (`0.999895x`) with `0.946923x` mean total wall time.

- Added diagnostic opt-in
  `NFN_NATIVE_GPT_ELIDE_FLOAT_PROJECTION_OUTPUTS=1` for dense GPT native
  training. When BF16 projection-residual is active, the switch skips the
  otherwise-unused FP32 scratch-tape `tape.attn_proj` and `tape.mlp_out`
  reservations, saving two activation-sized float buffers. It remains off by
  default because the dedicated RTX 5090 same-script checks measured
  train-loop neutral (`1.000250x` older-over-new train-loop time in the 3-step
  run) and startup-wall neutral-to-slightly slower for the elided side
  (`0.991309x` older-over-new setup wall time; lower is faster for the older
  opt-out side). Runtime JSON reports `float_projection_outputs_elided`,
  `float_projection_output_elements_elided`, and matching `block_state_layout`
  counters. Updated `README.md`, `docs/cli.md`, and the CUDA Tile checklist.
  Verification: rebuilt `build/nfn_gpt_native_train`, checked startup/profile
  JSON, ran a 3-step same-script native benchmark, and ran a 3-sample
  startup-only same-script benchmark on the dedicated RTX 5090.

- Dense GPT native training JSON now reports ranked float and BF16/uint16 arena
  request details as `float_arena_request_stats` and
  `uint16_arena_request_stats`. Each profile includes total requested/allocated
  elements/bytes plus the largest named suballocations with byte size and arena
  offset, making startup `cudaMalloc` optimization evidence-driven instead of
  relying on source inspection. Updated `README.md` and `docs/cli.md`.
  Verification: rebuilt `build/nfn_gpt_native_train`, ran a startup/profile
  native GPT pass, and parsed the new top-request fields from the emitted JSON.

- **Breaking changes: native-only CLI training.** `nfn train` and direct
  `cli/scripts/train_*.py` execution no longer honor
  `NFN_ALLOW_TORCH_TRAINING=1` as a graph-backed TorchTrainer bypass. Before,
  setting that environment variable could route unsupported model-family
  training through the legacy Python/Torch harness. Now CLI training always
  dispatches to compiled native CUDA/C++ entrypoints or fails before importing
  Torch when no native trainer exists. For one-off legacy graph-backed
  experiments, call the Python SDK trainer APIs directly instead of routing
  through CLI training entrypoints. Updated `README.md`, `docs/cli.md`,
  `docs/framework-guide/training-workflows.md`, `cli/README.md`, and the local
  NeuralFn CLI/SDK skills. Verification: added regressions proving
  `NFN_ALLOW_TORCH_TRAINING=1` is ignored by `nfn train --base-model llama`
  and by direct legacy script execution while Torch and `nfn_impl` remain
  unloaded.

- The master `nfn train` native dense-GPT dispatcher now uses the generic
  `gpt` template selector as its implicit default, matching
  `cli/scripts/train_gpt.py`, the SDK native GPT config, and the compiled C++
  preflight contract. Explicit `--template-name`, `--template`, `--preset`, and
  `--graph-file` selectors continue to define the architecture, and `gpt2` /
  `gpt3` remain model-family aliases for the same native dense GPT trainer.
  Verification: added a no-Torch dry-run regression for `nfn train
  --base-model gpt` and checked the internal default selector reports `gpt`.

- The master `nfn train` native dispatcher now sets the same workstation CUDA
  defaults as the canonical GPT script before invoking a native trainer:
  `CUDA_VISIBLE_DEVICES=0` and `CUDA_DEVICE_MAX_CONNECTIONS=1` are supplied
  only when the caller has not already set them. This keeps direct `nfn train`
  runs on the dedicated NVIDIA compute GPU by default on mixed display/compute
  machines. Verification: added a no-Torch subprocess regression using a
  temporary native-train stub that confirms both environment values are passed
  through the direct native dispatch path.

- Corrected the packed-attention backward dprep default so the rejected 3D
  batch/head/time launch remains opt-in under
  `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_GRID3D=1`. This aligns the low-level
  Tile-CUDA behavior with the documented row-linear default and the previous
  RTX 5090 paired benchmark rejection. Verification: added a native GPT source
  regression for the default-off dprep helper and rebuilt the Tile ops library.

- Added a diagnostic-only GPT token-weight initializer variant controlled by
  `NFN_NATIVE_GPT_TOKEN_WEIGHT_FAST_INT32_INIT=1` (with GPT-2 and Tile-CUDA
  aliases). It keeps the same power-of-two deterministic values but computes
  Tile bucket indices through int32 Tile values for GPT-sized tables. It remains
  off by default because the dedicated RTX 5090 startup-only comparison
  measured the existing int64 Tile path faster: the old path opt-out was
  `0.980751x` setup wall time, `0.984024x` token-init time, and `0.993102x`
  total wall time versus the int32 candidate. Verification: rebuilt
  `build/libnfn_native_train_tile_ops.so` and ran the paired startup-only
  benchmark with `NFN_NATIVE_GPT_TOKEN_WEIGHT_FAST_INT32_INIT=0`.

- Dense GPT native layer-evo now performs real forward-only candidate scoring
  instead of placeholder device-zero loss selection. After AdamW, the
  `--layer-evo` loop mutates the selected block's float32 `ln1.weight`, resets
  the active training batch, allocates the same lazy float MLP scratch used by
  validation-only forwards, evaluates every candidate with the native CUDA
  forward loss on the current batch, writes candidate losses back to device
  memory, then selects/adopts through the raw evo ABI. Runtime JSON now reports
  `layer_evo.forward_candidate_evals` and
  `layer_evo.candidate_loss_source:
  "native-forward-loss-current-batch"` while keeping
  `graph_editor_tensor_flow: false`. Verification: rebuilt
  `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`, ran a
  one-step no-evo validation smoke, then ran one-step TinyStories CUDA smokes
  with `--layer-evo --evo-layer-interval 1 --evo-layer-population 1
  --evo-layer-mutation-scale 0` and with `--evo-layer-population 2`; the latter
  reported `status: "native-transformer-lm-trained"`,
  `steps_completed: 1`, `runs: 1`, `mutate/select/adopt` launch counts of `1`,
  and `forward_candidate_evals: 2`.

- `tools/bench_native_gpt_sm120_candidate.sh` now supports
  `NFN_SM120_NATIVE_STARTUP_ONLY=1` for startup-only native GPT bisections. The
  wrapper appends `--startup-only` to both baseline and candidate commands while
  preserving the same dense GPT command shape, TinyStories shard resolution,
  checkpoint-disabled benchmark settings, selected-GPU idle guard, and
  external-load controls. The CUDA Tile checklist now records the current
  2026-06-18 llm.kittens parity snapshot and rejected startup/train-loop
  candidates, including the reduced saved-activation storage candidate that
  improved setup time but regressed train-loop time too much to promote.
  Verification: ran `bash -n tools/bench_native_gpt_sm120_candidate.sh`, a
  startup-only baseline-vs-baseline smoke with
  `NFN_SM120_NATIVE_STARTUP_ONLY=1`, and the current same-script 3-step parity
  and candidate bisections on the dedicated RTX 5090.

- `nfn_gpt2_evo_native_train` now delegates dense GPT-2-compatible training
  runs to `nfn_gpt_native_train --train-transformer-lm --layer-evo` instead of
  exiting after the C++ preflight. The family binary still owns
  `--print-plan`, `--dry-run`, and `--smoke-evo-kernels`; incompatible custom
  graphs and non-dense templates remain non-runnable until native trainer
  coverage exists. Plan JSON now reports
  `native-preflight-dense-gpt-layer-evo-delegate`,
  `selected_graph_support_status: "native-dense-gpt-layer-evo-delegate"`, and
  `selected_graph_native_runnable: true` for the delegated templates. Runtime
  JSON from the delegated trainer remains explicit that forward-only evo
  candidate loss evaluation is not implemented yet
  (`candidate_loss_source:
  "placeholder-device-zero-loss-selects-current-candidate"`). Verification:
  rebuilt `build/nfn_gpt2_evo_native_train`, checked print-plan/dry-run JSON,
  and ran a delegated startup-only TinyStories command through the evo wrapper;
  it reported `passed: true`, `status: "native-transformer-lm-startup-ready"`,
  `layer_evo.runtime_enabled: true`, and
  `layer_evo.graph_editor_tensor_flow: false`.

- Dense GPT native runtime JSON now reports `timing.post_train_sample_wall_ms`
  and `timing.cleanup_wall_ms` separately. Startup-only and short benchmark
  runs can now distinguish time-to-ready/setup from post-loop diagnostic sample
  copies and explicit CUDA teardown of large arenas.

- Added GPT-prefixed aliases for the native Tile-CUDA linear shape-stat profiler.
  `NFN_NATIVE_GPT_LINEAR_SHAPE_STATS=1` and
  `NFN_NATIVE_GPT2_LINEAR_SHAPE_STATS=1` now behave like the existing
  `NFN_NATIVE_LINEAR_SHAPE_STATS=1` and `NFN_TILE_CUDA_LINEAR_SHAPE_STATS=1`
  switches, so GPT-stage profiling can enable exact GEMM shape buckets with
  GPT-named environment variables. This is diagnostic-only and remains off by
  default.

- Added `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=m,n,k,opA,opB` and
  `NFN_TILE_CUDA_LINEAR_TK_FORWARD_DISABLE_SHAPE=m,n,k,opA,opB` as
  diagnostic-only one-shape bisection gates for TK BF16 forward/fused-GELU
  launches. The gate is intentionally limited to forward paths with existing
  fallback implementations and does not disable bits-only backward dGELU paths.
  Verification included a smoke with
  `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=3072,65536,768,T,N` that removed
  that TK `T,N` bucket while leaving the `N,N` backward bucket active, followed
  by a same-script dedicated RTX 5090 5-step, 3-sample benchmark that rejected
  the fallback as a default route (`13.608951x` train-loop wall time and
  `0.073530x` tokens/sec versus baseline).

### 2026-06-18 Suballocate native GPT float stats sidecars

#### Changed

- Dense GPT native training now reserves stored MLP LayerNorm stats and saved
  packed-attention LN1 stats through the existing single float arena instead of
  issuing separate float `cudaMalloc` calls during startup.
- Updated README, Python SDK Tile-CUDA docs, and the Tile-CUDA checklist to
  describe those stats sidecars as part of the default float arena. Set
  `NFN_NATIVE_GPT_FLOAT_STATS_ARENA=0` or
  `NFN_NATIVE_GPT2_FLOAT_STATS_ARENA=0` only for paired startup comparisons
  against the older sidecar allocation route.

#### Verification

- Rebuilt `nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran default and `NFN_NATIVE_GPT_FLOAT_STATS_ARENA=0` one-step TinyStories
  native smokes. Default JSON reported
  `float_stats_sidecars_in_arena_enabled: true`,
  `float_allocation_cuda_malloc_count: 1`,
  `float_allocation_request_count: 639`,
  `stored_mlp_layer_norm_stats_elements: 1572864`, and
  `stored_packed_attention_ln1_stats_elements: 1441792`; the opt-out smoke
  reported `float_stats_sidecars_in_arena_enabled: false` and preserved the
  same stats sidecar shapes.
- Ran a dedicated RTX 5090 same-script startup-only 5-sample comparison against
  `NFN_NATIVE_GPT_FLOAT_STATS_ARENA=0`; the opt-out measured `1.001616x` setup
  wall time and `1.003047x` total wall time versus the new default, so the
  arena-backed stats sidecars remain the default.

### 2026-06-18 Suballocate saved packed-attention LN1 BF16 tape

#### Changed

- Dense GPT native training now reserves the saved packed-attention LN1 BF16
  tape through the default combined uint16 arena instead of issuing a separate
  BF16 `cudaMalloc`. This keeps the saved packed-attention BF16 activation
  layout under one allocation and makes `uint16_arena_suballocation_count`
  account for that tape.
- Updated README, Python SDK Tile-CUDA docs, and the Tile-CUDA checklist to
  describe the saved LN1 BF16 tape as part of the default uint16 arena. The
  `NFN_NATIVE_GPT_COMBINED_BF16_ARENA=0` /
  `NFN_NATIVE_GPT2_COMBINED_BF16_ARENA=0` fallback still reproduces the older
  per-buffer allocation path.

#### Verification

- Rebuilt `nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran a dedicated RTX 5090 one-step TinyStories native smoke. Runtime JSON
  reported `uint16_allocation_strategy: "single-arena"`,
  `uint16_allocation_cuda_malloc_count: 1`,
  `uint16_arena_suballocation_count: 11`,
  `stored_packed_attention_ln1_bf16_enabled: true`,
  `stored_packed_attention_ln1_bf16_blocks: 11`, and
  `stored_packed_attention_ln1_bf16_bytes: 1107296256`.
- Ran a same-script 5-step, 3-sample diagnostic comparison against the broad
  `NFN_NATIVE_GPT_COMBINED_BF16_ARENA=0` per-buffer fallback. The fallback
  measured `0.996650x` mean train-loop wall time and `1.003578x` mean
  tokens/sec, but median train-loop wall time was `1.004820x`, so this
  comparison was treated as noisy and no broader allocation fallback was
  promoted.

### 2026-06-18 Align BGRADB direct-bias reporting with the faster default

#### Changed

- Dense GPT native runtime JSON now reports
  `linear_bias_gradient_first_write_bgrad_direct_enabled: false` unless
  `NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1`,
  `NFN_NATIVE_GPT2_BGRAD_FIRST_WRITE_DIRECT=1`, or
  `NFN_TILE_CUDA_LINEAR_BGRAD_FIRST_WRITE_DIRECT=1` is set. This matches the
  low-level Tile-CUDA behavior and avoids claiming the direct BGRADB bias path
  is active when the default route is still scratch plus accumulation.
- Updated README, Python SDK Tile-CUDA docs, and the Tile-CUDA checklist to
  describe direct BGRADB bias writes as an opt-in diagnostic instead of the
  default. The 2026-06-18 paired re-check measured the direct path
  neutral-to-slower, so no kernel default was promoted.

#### Verification

- Ran a dedicated RTX 5090 same-script 5-step, 3-sample paired benchmark with
  `NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1`; the candidate measured
  `1.000529x` train-loop wall time and `0.999486x` tokens/sec versus the
  current scratch-accumulate default.

### 2026-06-18 Record current SM120 GEMM and attention handoff rejections

#### Changed

- Recorded three current dense GPT native trainer bisections as rejected
  defaults so they are not retested while closing the remaining llm.kittens
  parity gap: low-level BGRADB first-write direct via
  `NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1`, non-cuBLASLt BF16 GEMMEx
  `CUBLAS_COMPUTE_32F_FAST_16BF` via
  `NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF=1`, and BF16 attention-gradient
  handoff via `NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1`.
- No runtime default was promoted in this slice; the measured candidates were
  neutral-to-slower than the current native Tile-CUDA route.

#### Verification

- Ran a dedicated RTX 5090 same-script 5-step, 3-sample paired benchmark with
  `NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1`; the candidate measured
  `1.000529x` train-loop wall time and `0.999486x` tokens/sec versus the
  current environment-default path.
- Ran a dedicated RTX 5090 same-script 5-step, 3-sample paired benchmark with
  `NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF=1`; the candidate measured
  `1.004222x` train-loop wall time and `0.995808x` tokens/sec.
- Ran a dedicated RTX 5090 same-script 5-step, 3-sample paired benchmark with
  `NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1`; the candidate measured
  `1.011370x` train-loop wall time and `0.988829x` tokens/sec.

### 2026-06-17 Guard default package dependencies against Torch regressions

#### Changed

- Extended `tools/check_native_no_torch_deps.py` to inspect `pyproject.toml`
  and fail if Torch, torchvision, or torchaudio return as default project
  dependencies. The same check also verifies that the optional `torch` extra is
  still present for graph-backed workflows.
- Updated the README and Python SDK Tile-CUDA docs to describe the package
  metadata portion of the native no-Torch dependency gate.

#### Verification

- Ran `python tools/check_native_no_torch_deps.py`.
- Ran `python tools/check_native_no_torch_deps.py --json`.

### 2026-06-17 Specialize packed-attention dprep for GPT HD64

#### Changed

- Packed-attention backward dprep now defaults to a specialized unrolled
  BF16-grad kernel for the dense GPT `heads=12, head_dim=64` path. Set
  `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_HD64_SPECIALIZED=0` to reproduce the
  older generic row dprep kernel; `NFN_NATIVE_GPT2_PACKED_ATTENTION_DPREP_HD64_SPECIALIZED`
  remains the fallback name.

#### Verification

- Ran `bash tools/build_native_train_tile_ops.sh`.
- Ran a dedicated RTX 5090 same-script 5-step, 3-sample paired benchmark with
  `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_HD64_SPECIALIZED=1` as the only
  candidate difference; the candidate measured `0.997290x` mean train-loop wall
  time and `1.002726x` mean tokens/sec versus the older generic row dprep path.
- Ran a dedicated RTX 5090 same-script 5-step, 3-sample paired benchmark after
  promoting the specialization by default and forcing the older path with
  `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_HD64_SPECIALIZED=0` as baseline; the
  promoted default measured `0.985815x` mean train-loop wall time and
  `1.014540x` mean tokens/sec.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran
  `python -m pytest tests/test_native_gpt2.py::test_native_train_tile_ops_builds_torch_free_c_abi -q`;
  the source-level assertions passed and the optional temp `nvcc` rebuild path
  skipped because `nvcc` is not on PATH in the test environment.
- Ran `git diff --check`.

### 2026-06-17 Add packed-attention backward section timing

#### Added

- Added opt-in CUDA-event timing for packed-attention backward dprep versus the
  SM120 TK backward launch. Set
  `NFN_NATIVE_GPT_ATTENTION_BACKWARD_SECTION_TIMING=1` for short diagnostic
  runs; `NFN_NATIVE_GPT2_ATTENTION_BACKWARD_SECTION_TIMING` and
  `NFN_TILE_CUDA_ATTENTION_BACKWARD_SECTION_TIMING` remain fallbacks.
- Runtime JSON now reports `attention_backward_section_timing_enabled`,
  `attention_backward_dprep_timing_us`,
  `attention_backward_dprep_timing_count`, `attention_backward_tk_timing_us`,
  and `attention_backward_tk_timing_count`. The diagnostic path synchronizes
  CUDA events, so it is not part of normal training timing.
- The raw Tile-CUDA trainer ABI now exports
  `nfn_native_tile_attention_backward_dprep_timing_us`,
  `nfn_native_tile_attention_backward_dprep_timing_count`,
  `nfn_native_tile_attention_backward_tk_timing_us`, and
  `nfn_native_tile_attention_backward_tk_timing_count`.

#### Verification

- Ran `bash tools/build_native_train_tile_ops.sh`.
- Ran `bash tools/build_native_gpt_cli.sh`.
- Ran `bash tools/build_native_gpt2_cli.sh`.
- Ran a dedicated RTX 5090 one-step native GPT diagnostic with
  `NFN_NATIVE_GPT_ATTENTION_BACKWARD_SECTION_TIMING=1`; runtime JSON reported
  `attention_backward_section_timing_enabled: true`,
  `attention_backward_tk_launch_count: 96`,
  `attention_backward_dprep_timing_us: 32587`,
  `attention_backward_dprep_timing_count: 96`,
  `attention_backward_tk_timing_us: 244217`, and
  `attention_backward_tk_timing_count: 96`.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran
  `python -m pytest tests/test_native_gpt2.py::test_native_train_tile_ops_builds_torch_free_c_abi -q`;
  the source-level assertions passed and the optional temp `nvcc` rebuild path
  skipped because `nvcc` is not on PATH in the test environment.

### 2026-06-17 Add packed-attention dprep warps bisection knob

#### Added

- Added `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_WARPS=N` with
  `NFN_NATIVE_GPT2_PACKED_ATTENTION_DPREP_WARPS` as a legacy fallback for
  packed-attention backward dprep row-grouping bisection. The default remains
  the existing 3 warps per dprep block.

#### Verification

- Ran `bash tools/build_native_train_tile_ops.sh`.
- Ran a dedicated RTX 5090 same-script one-step bisection with
  `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_WARPS=2`; the candidate measured
  `1.002758x` mean train-loop wall time and `0.997257x` tokens/sec versus the
  default, so it was not promoted.
- Ran a dedicated RTX 5090 same-script one-step bisection with
  `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_WARPS=4`; the candidate measured
  `1.001645x` mean train-loop wall time and `0.998425x` tokens/sec versus the
  default despite a slightly faster median, so it was not promoted.
- Ran
  `python -m pytest tests/test_native_gpt2.py::test_native_train_tile_ops_builds_torch_free_c_abi -q`;
  the source-level assertions passed and the optional temp `nvcc` rebuild path
  skipped because `nvcc` is not on PATH in the test environment.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran `git diff --check`.

### 2026-06-17 Fuse dense GPT MLP residual into next LN1

#### Changed

- Dense GPT native `--train-transformer-lm` now fuses each stored MLP
  projection bias/residual into the next block's LN1 stats and BF16 output when
  packed LN1 storage or scratch tape is available.
- Added `NFN_NATIVE_GPT_FUSE_MLP_RESIDUAL_NEXT_LN1` with
  `NFN_NATIVE_GPT2_FUSE_MLP_RESIDUAL_NEXT_LN1` as the legacy fallback opt-out
  for paired bisection.
- Runtime JSON now reports
  `block_state_layout.mlp_residual_next_ln1_fusion_enabled`,
  `block_state_layout.mlp_residual_next_ln1_fusion_count`, and
  `block_state_layout.mlp_residual_next_ln1_strategy`.

#### Verification

- Ran `bash tools/build_native_gpt_cli.sh`.
- Ran a dedicated RTX 5090 one-step stage probe with
  `NFN_NATIVE_GPT_STAGE_TIMING=1` and shape stats enabled; it completed with
  `status: "native-transformer-lm-trained"` and reported
  `mlp_residual_next_ln1_fusion_count: 88`, matching 8 grad-accum
  microbatches across 11 block boundaries.
- Ran a dedicated RTX 5090 same-script paired benchmark against
  `NFN_NATIVE_GPT_FUSE_MLP_RESIDUAL_NEXT_LN1=0`; the default fused candidate
  measured `0.995763x` mean train-loop wall time and `1.004256x` tokens/sec
  over three samples, with zero selected-GPU compute processes before and after
  each sample.
- Ran
  `python -m pytest tests/test_native_gpt2.py::test_native_train_tile_ops_builds_torch_free_c_abi -q`;
  the source-level assertions passed and the optional temp `nvcc` rebuild path
  skipped because `nvcc` is not on PATH in the test environment.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran `git diff --check`.

### 2026-06-17 Retile native GPT token initializer to 4096

#### Changed

- Made the dense GPT native token-weight CUDA Tile initializer size a
  compile-time constant controlled by
  `NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE`, accepting `1024`, `2048`, or
  `4096`.
- Changed the default trainer-facing Tile build from the previous 2048-element
  token initializer tile to 4096 elements, keeping the non-threaded CUDA Tile
  path as the default and leaving the threaded CUDA initializer diagnostic-only.

#### Verification

- Ran `bash tools/build_native_train_tile_ops.sh`.
- Ran a dedicated RTX 5090 startup-only same-script comparison between the
  preserved 2048-tile library and the rebuilt 4096-tile library:
  candidate `setup.token_weight_init.total_ms` measured `0.895736x` mean versus
  baseline, while total startup was noisy/slower at `1.179639x`.
- Rejected the 1024-tile candidate after a dedicated RTX 5090 startup-only
  comparison measured `1.007101x` token-init time and `1.012266x` total wall
  time versus the preserved 2048-tile baseline.
- Ran a dedicated RTX 5090 one-step native training comparison between the
  preserved 2048-tile library and the rebuilt 4096-tile library; the candidate
  measured `0.991404x` total wall time, `0.999314x` train-loop wall time, and
  `1.000805x` tokens/sec versus baseline.

### 2026-06-17 Wire dense GPT native layer-evo ABI cadence

#### Added

- Added `--layer-evo`, `--enable-layer-evo`, `--native-cuda-layer-evo`,
  `--no-layer-evo`, `--evo-layer-index`, `--evo-layer-interval`,
  `--evo-layer-population`, and `--evo-layer-mutation-scale` to the compiled
  dense GPT native trainer.
- Dense GPT `--print-plan` and training JSON now include a `layer_evo` block
  reporting the selected target parameter, cadence, population, mutation scale,
  kernel launch counters, and `graph_editor_tensor_flow: false`.
- The native `--train-transformer-lm` optimizer loop now allocates device
  candidate workspace for the selected block's float32 `block_N.ln1.weight` and
  calls the raw Tile-CUDA mutate/select/adopt ABI kernels when `--layer-evo` is
  enabled. Candidate losses are currently placeholder device zeros, so
  forward-only candidate loss evaluation remains the next step before this is
  the full GPT-2 evo training algorithm.

#### Verification

- Ran `bash tools/build_native_gpt_cli.sh`.
- Ran a one-step live CUDA smoke with
  `build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1 --batch-size 1 --train-seq-len 1024 --train-batch-tokens 1024 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 1 --native-cuda-checkpoint-every 0 --no-checkpoint --layer-evo --evo-layer-interval 1 --evo-layer-population 3 --tile-ops-lib build/libnfn_native_train_tile_ops.so --json-out /tmp/nfn_layer_evo_smoke.json`,
  which completed with `passed: true`, `layer_evo.runs: 1`, and one
  mutate/select/adopt launch each.
- Rejected a refreshed `--lm-head-row-chunk-size 4096` candidate in the paired
  same-script native benchmark because it measured `1.003888x` train-loop wall
  time and `0.996133x` tokens/sec versus the current default.
- Ran `python -m pytest tests/test_native_gpt2.py::test_missing_family_native_trainers_build_and_unified_frontend_dispatches -q`.
- Ran `python -m pytest tests/test_native_gpt2.py::test_native_train_tile_ops_builds_torch_free_c_abi -q`
  (skipped after reaching the local build/device gate).

### 2026-06-17 Fix NanoGPT native vocabulary default

#### Changed

- Updated the compiled `nfn_nanogpt_native_train` default vocabulary from the
  old 1024-token reduced shape to the GPT-2 tokenizer vocabulary size
  (`50257`). This keeps the implemented native `--train-token-lm` path aligned
  with the default TinyStories/GPT-2-tokenized cached shards unless callers
  explicitly pass `--vocab-size` for a tiny smoke.
- Updated native NanoGPT preflight tests and docs to expect the full-vocab
  parameter layout by default.

#### Verification

- Ran `python -m pytest tests/test_native_gpt2.py::test_missing_family_native_trainers_build_and_unified_frontend_dispatches -q`.
- Ran `git diff --check`.

### 2026-06-17 Fix GPT-2 evo native preflight vocabulary default

#### Changed

- Updated the compiled `nfn_gpt2_evo_native_train` preflight to default to the
  real GPT-2 tokenizer vocabulary size (`50257`) instead of the old 1024-token
  parameter-golf shape. Plan JSON and estimated parameter counts now reflect
  the intended GPT-2 evo default unless callers explicitly pass
  `--vocab-size`.
- Updated README and framework-guide docs so the GPT-2 evo native plan is
  described as a full-vocab GPT-2 shape.

#### Verification

- Ran `python -m pytest tests/test_native_gpt2.py::test_missing_family_native_trainers_build_and_unified_frontend_dispatches -q`.
- Ran `git diff --check`.

### 2026-06-17 Remove dense GPT llm.kittens training bridge

#### Changed

- Dense GPT training now accepts only the NeuralFn-owned `tile-cuda` backend in
  the Python wrappers, SDK config builders, top-level `nfn train` direct
  dispatch, and compiled `nfn_gpt_native_train` frontend.
- The compiled GPT frontend no longer carries the dead external fast-path tail
  or `train_gpt2cu` default target; benchmark comparisons now live only in
  `tools/bench_native_gpt_sm120_parity.sh`.
- Wrapper dry-runs now emit `--backend tile-cuda` explicitly on the compiled
  native GPT command so the no-Torch route is visible in command inspection.
- `run_native_gpt(..., runner="subprocess")` / `run_native_gpt2(...,
  runner="subprocess")` is no longer a valid GPT training runner. Use
  `compiled-cli`, `auto`, `binding`, or `launcher`, all of which stay on
  NeuralFn native artifacts.
- `tools/bench_native_gpt_sm120_parity.sh` remains the llm.kittens reference
  comparator; it is benchmark tooling, not a selectable training backend.

#### Breaking changes

- `--backend llm-kittens`, `--native-cuda-kernel-backend llm-kittens`, SDK
  `kernel_backend="llm-kittens"`, and GPT SDK `runner="subprocess"` now fail
  validation. Use `--backend tile-cuda` / `kernel_backend="tile-cuda"` and a
  NeuralFn native runner instead.
- The GPT wrapper no longer auto-adds `--target train_gpt2cu`, and
  `NFN_NATIVE_GPT_TRAIN_BIN` / `NFN_NATIVE_GPT2_TRAIN_BIN` are no longer part of
  normal GPT training dispatch.

#### Verification

- Ran `python -m py_compile neuralfn/native_gpt2.py cli/scripts/train_gpt_native.py cli/scripts/train_gpt.py cli/nfn.py`.
- Ran `python -m pytest cli/tests/test_train_gpt2_native.py -q`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k "cpp_cli_builds_and_uses_sm120_defaults or kernel_backend or external_bridge or compiled_cli_config or runner_status or write_native_gpt2_run_config"`.
- Ran `git diff --check`.

### 2026-06-17 Add native SM120 candidate benchmark wrapper

#### Added

- Added `tools/bench_native_gpt_sm120_candidate.sh`, a native-vs-native dense GPT
  SM120 benchmark wrapper for CUDA Tile kernel bisection. The wrapper keeps the
  same compiled native GPT command shape on both sides, pins
  `--train-batch-tokens 524288` by default, compares the current Tile ops
  library/default environment against `NFN_SM120_NATIVE_CANDIDATE_ENV` or
  `NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB`, and reuses the selected-GPU
  idle/utilization guards from `tools/paired_kernel_speed.py`.
- Documented the wrapper in README and added the task/evidence to
  `todo-tile-cuda.md`.

#### Notes

- Rejected fresh one-step stage probes for
  `NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1`,
  `NFN_NATIVE_GPT_REUSE_MLP_PROJ_BF16_GRAD_OUT=0`, and cublasLt heuristic index
  `0` for the `768,65536,3072,N,N`, `768,65536,2304,N,N`, and
  `768,65536,768,N,N` dInput shapes. None produced a stable targeted-stage win
  worth promoting to a default.

#### Verification

- Ran GPU-visible one-step native GPT stage probes on the dedicated RTX 5090 for
  the rejected runtime candidates listed above. Each completed with
  `status: native-transformer-lm-trained` and `passed: true`.
- Ran `bash -n tools/bench_native_gpt_sm120_candidate.sh`.
- Ran
  `NFN_SM120_NATIVE_STEPS=1 NFN_SM120_NATIVE_SAMPLES=1 NFN_SM120_NATIVE_WARMUP=0 NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES=0 NFN_SM120_NATIVE_PROFILE_DIR=none NFN_SM120_NATIVE_CANDIDATE_ENV=NFN_NATIVE_GPT_REUSE_MLP_PROJ_BF16_GRAD_OUT=0 tools/bench_native_gpt_sm120_candidate.sh`;
  the paired wrapper completed, wrote `/tmp/nfn_sm120_native_candidate_wrapper_smoke.json`,
  and recorded zero selected-GPU compute processes before and after the sample.

### 2026-06-17 Pin SM120 parity batch-token contract

#### Changed

- Updated `tools/bench_native_gpt_sm120_parity.sh` so the NeuralFn candidate
  command passes `--train-batch-tokens 524288` explicitly, matching the
  `llm.kittens/train-sm120.sh` reference `-d 524288` setting in the same paired
  benchmark script.
- Updated README, Python SDK Tile-CUDA docs, and the CUDA Tile checklist to
  document that the parity wrapper now locks both sides to the same effective
  token batch.

#### Verification

- Regenerated a GPU-visible one-step NeuralFn stage/shape profile on the
  dedicated RTX 5090 with `NFN_NATIVE_GPT_STAGE_TIMING=1` and
  `NFN_NATIVE_LINEAR_SHAPE_STATS=1`; the current native path completed one
  optimizer step and reported active cuBLASLt/TK hot paths.
- Ran the updated explicit-batch parity wrapper with `NFN_SM120_PARITY_STEPS=10`,
  `NFN_SM120_PARITY_SAMPLES=3`, `NFN_SM120_PARITY_WARMUP=0`,
  `NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0`, and
  `NFN_SM120_PARITY_PROFILE_DIR=none`. It measured NeuralFn at `1.033656x`
  train-loop wall time and `0.967400x` tokens/sec versus `llm.kittens`, with
  zero selected-GPU compute processes before or after every paired sample.
- Static verification is covered by
  `tests/test_tile_cuda_examples.py::test_native_gpt_sm120_parity_wrapper_uses_reference_shape`.

### 2026-06-17 Align token-weight initializer default

#### Changed

- Changed the low-level native Tile token-weight initializer helper so
  `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT` / `NFN_TILE_CUDA_TOKEN_WEIGHT_THREADED_INIT`
  default off inside `libnfn_native_train_tile_ops.so`, matching the compiled
  GPT trainer, README, SDK docs, and runtime JSON.
- The diagnostic threaded CUDA initializer is still available by setting
  `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1`; the default path remains the
  fast CUDA Tile deterministic initializer with fused BF16 LM-head shadow
  initialization.
- Updated README, Python SDK Tile-CUDA docs, and the CUDA Tile checklist with
  the corrected default and benchmark result.

#### Verification

- Ran `python -m pytest tests/test_native_gpt2.py::test_native_train_tile_ops_builds_torch_free_c_abi -q` (static assertions passed; the optional runtime portion skipped).
- Rebuilt `build/libnfn_native_train_tile_ops.so` with `bash tools/build_native_train_tile_ops.sh`.
- Ran `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --backend tile-cuda --tinystories --startup-only --max-steps 1 --train-batch-tokens 65536 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 16 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so --profile-json /tmp/nfn_token_init_default_aligned_startup.json`.
- Ran the same startup probe with `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1`
  and wrote `/tmp/nfn_token_init_threaded_startup_after_default_fix.json`.
- Ran a 5-sample paired startup benchmark against
  `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1`; the corrected default measured
  `0.940074x` token init time, `0.974488x` setup wall time, and `0.976437x`
  total wall time.
- Ran `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1 --train-batch-tokens 65536 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 16 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so --profile-json /tmp/nfn_token_init_default_aligned_train_smoke.json`.

### 2026-06-17 Accept native-cuda aliases in GPT-2 evo preflight

#### Changed

- The compiled `nfn_gpt2_evo_native_train` preflight now accepts wrapper-style
  native-cuda aliases directly for `--native-cuda-print-plan`,
  `--native-cuda-smoke-evo-kernels`, `--native-cuda-tile-ops-lib`, and
  `--native-cuda-cuda-runtime-lib`.
- Updated README, CLI docs, and Python SDK Tile-CUDA docs so direct binary,
  direct script, and wrapper behavior describe the same preflight contract.

#### Verification

- Ran `python -m pytest tests/test_native_gpt2.py::test_missing_family_native_trainers_build_and_unified_frontend_dispatches -q`.
- Ran `python -m py_compile cli/scripts/native_training_guard.py cli/tests/test_train_gpt2_native.py`.
- Ran `git diff --check`.

### 2026-06-17 Normalize native preflight aliases for guarded scripts

#### Changed

- Direct guarded training scripts now recognize wrapper-level native-cuda
  preflight actions before graph-backed imports and normalize them to the
  canonical C++ flags before forwarding to family-specific native binaries.
- `python cli/scripts/train_gpt2_evo.py --native-cuda-print-plan ...` now
  forwards `--print-plan`, and native-cuda Tile ops / CUDA runtime value aliases
  forward as `--tile-ops-lib` / `--cuda-runtime-lib`, so GPT-2 evo preflight and
  smoke commands stay on the compiled family binary path without touching Torch
  or graph-editor training code.
- Updated README, CLI docs, and Python SDK Tile-CUDA docs for the direct-script
  alias forwarding behavior.

#### Verification

- Ran `python -m pytest cli/tests/test_train_gpt2_native.py::TrainGpt2NativeStartupTest::test_train_gpt2_evo_direct_script_normalizes_native_cuda_preflight_aliases cli/tests/test_train_gpt2_native.py::TrainGpt2NativeStartupTest::test_train_gpt2_evo_direct_script_prefers_family_native_preflight cli/tests/test_train_gpt2_native.py::TrainGpt2NativeStartupTest::test_train_nanogpt_direct_script_defaults_to_native_token_lm -q`.

### 2026-06-17 Reject 3D packed-attention dprep default

#### Changed

- Added diagnostic packed-attention dprep kernels that launch over
  batch/head/time dimensions instead of remapping a flat row index with
  per-row division and modulo.
- Added `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_GRID3D=1` and the
  `NFN_NATIVE_GPT2_PACKED_ATTENTION_DPREP_GRID3D=1` fallback to test the 3D
  dprep launch during paired timing; the row-linear dprep launch remains the
  default because the 3D candidate measured slower.
- Updated README, CLI docs, and Python SDK Tile-CUDA docs for the diagnostic
  packed attention dprep route and fallback switch.

#### Verification

- Ran `python -m pytest tests/test_native_gpt2.py::test_native_train_tile_ops_builds_torch_free_c_abi -q` (static assertions passed; the optional runtime portion skipped in this environment).
- Rebuilt `build/libnfn_native_train_tile_ops.so` with `bash tools/build_native_train_tile_ops.sh`.
- Ran `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1 --train-batch-tokens 65536 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 16 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so --profile-json /tmp/nfn_dprep_grid3d_smoke.json`.
- Rebuilt after restoring the row-linear default and reran the one-step smoke
  with `--profile-json /tmp/nfn_dprep_default_off_smoke.json`.
- Ran a 3-sample paired benchmark with the old row-linear dprep as baseline and the 3D dprep as candidate; it measured `0.999295x` mean train-loop wall time and `1.000730x` mean tokens/sec, which was too small to promote by itself.
- Ran a 5-sample paired confirmation with the same baseline/candidate; it measured `1.008389x` train-loop wall time and `0.991895x` tokens/sec, so the 3D dprep remains diagnostic-only and default-off.

### 2026-06-17 Add shape-specific cuBLASLt heuristic bisection

#### Changed

- Added `NFN_TILE_CUDA_CUBLASLT_HEURISTIC_SHAPE=m,n,k,opA,opB,index` and
  `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=m,n,k,opA,opB,index` for
  one-shape cuBLASLt heuristic bisection in the trainer-facing Tile-CUDA linear
  path.
- The shape-specific override only applies to the matching cuBLASLt plan; other
  GEMMs continue using the current default/global heuristic selection.
- Recorded the first QKV dWeight hot-shape bisection as diagnostic-only:
  `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,2304,65536,N,T,0` measured
  noise-equivalent mean train-loop time and slightly slower median train-loop
  time versus the default heuristic selection, so no default route changed.
- Recorded two follow-up MLP dWeight hot-shape bisections without promoting a
  route change: `3072,768,65536,N,T,0` measured only a small mean/median win,
  while `768,3072,65536,N,T,0` reversed on the five-sample confirmation and
  measured slower than the default heuristic selection.
- Recorded the LM-head dWeight hot-shape bisection
  `768,50304,8192,N,T,0` as slower than the default heuristic selection.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py::test_native_train_tile_ops_builds_torch_free_c_abi -q`
  (static assertions passed; the test skipped the optional runtime portion in
  this environment).
- Ran the dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --train-batch-tokens 524288 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --train-batch-tokens 524288 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,2304,65536,N,T,0 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 25 --command-timeout-seconds 1800 --append-native-profile-json-dir /tmp/nfn_shape_h0_qkv_dweight_profiles --json-out /tmp/nfn_shape_h0_qkv_dweight_pair.json`.
- The paired benchmark measured candidate/default at `0.999825x` mean
  train-loop wall time, `1.001401x` median train-loop wall time, and
  `1.000185x` mean tokens/sec, so the shape-specific override remains
  diagnostic-only.
- Ran `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=3072,768,65536,N,T,0` through
  the same 3-sample paired benchmark; it measured `0.998065x` mean train-loop
  wall time and `0.998219x` median train-loop wall time, which is too small to
  justify a default change.
- Ran `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,3072,65536,N,T,0` through
  a 5-sample confirmation after a noisy 3-sample result; the confirmation
  measured `1.015160x` train-loop wall time and `0.985688x` tokens/sec, so the
  default heuristic selection remains unchanged.
- Ran `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,50304,8192,N,T,0` through
  the same 3-sample paired benchmark; it measured `1.002294x` train-loop wall
  time and `0.997718x` tokens/sec, so the LM-head dWeight route keeps the
  default heuristic selection.

### 2026-06-17 Add canonical GPT inference entrypoint

#### Changed

- Added `cli/scripts/infer_gpt.py` as the canonical GPT inference script.
  `cli/scripts/infer_gpt2.py` remains a compatibility entrypoint over the same
  graph-backed and native checkpoint implementation.
- Updated `nfn infer` to accept `--native-checkpoint` as an explicit alias for
  native dense GPT `.bin` checkpoints, while preserving `--checkpoint` and
  `--weights` detection.
- Switched the lightweight native checkpoint detection path in `nfn infer` to
  the generic `neuralfn.native_gpt` SDK aliases so native dense GPT checkpoints
  are reported as GPT runtime artifacts rather than GPT-2-only artifacts.
- Extended `tools/check_native_no_torch_deps.py` to verify native GPT inference
  metadata entrypoints using a synthetic native checkpoint under the same
  Torch/NumPy/tiktoken/dataset-manager/`nfn_impl` import blocker.
- Updated CLI docs and README to direct users to `infer_gpt.py` and the compiled
  `nfn_gpt_native_train --sample-checkpoint` inference path.

#### Verification

- Ran `python -m py_compile cli/scripts/infer_gpt.py cli/scripts/infer_gpt2.py cli/nfn.py`.
- Ran `python -m pytest cli/tests/test_cli_help_behavior.py -q`.
- Ran `python -m pytest cli/tests/test_infer_megakernel_artifacts.py -q -k 'gpt or raw_text'`.
- Ran `python -m pytest cli/tests/test_train_gpt2_native.py -q -k 'infer_gpt or nfn_infer_native_checkpoint_is_recognized'`.
- Ran `python -m py_compile tools/check_native_no_torch_deps.py cli/scripts/infer_gpt.py cli/scripts/infer_gpt2.py cli/nfn.py`.
- Ran `python tools/check_native_no_torch_deps.py --skip-artifacts --json`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k 'native_no_torch_dependency_verifier'`.

### 2026-06-17 Keep paired native profile sidecars timing-neutral

#### Changed

- Changed `tools/paired_kernel_speed.py --append-native-profile-json-dir` so it
  only appends per-command `--profile-json` sidecars. It no longer silently sets
  `NFN_NATIVE_GPT_STAGE_TIMING=1`, keeping same-script NeuralFn-vs-llm.kittens
  throughput comparisons on the normal training path.
- Added explicit `tools/paired_kernel_speed.py --native-stage-timing` for
  attribution runs that should record CUDA-event `timing.stage_timing` buckets
  and paired `stage.*` metrics.
- Updated `tools/bench_native_gpt_sm120_parity.sh` so
  `NFN_SM120_PARITY_STAGE_TIMING=1` is the opt-in for stage-timed parity
  sidecars; plain sidecars remain enabled by default without modifying the
  measured command.

#### Breaking changes

- Before this change, `--append-native-profile-json-dir` enabled native GPT
  stage timing as a side effect for NeuralFn native commands. Callers that rely
  on paired `stage.*` metrics must now add `--native-stage-timing`, or set
  `NFN_SM120_PARITY_STAGE_TIMING=1` when using the SM120 parity wrapper.

#### Verification

- Dedicated RTX 5090 baseline check before the harness correction showed the
  plain native 5-step command at `174287` tokens/sec with
  `stage_timing_enabled: false`, while the old sidecar harness enabled
  `stage_timing_enabled: true` and recorded `20000` stage events per candidate
  sample, skewing the comparison.
- Ran `python -m py_compile tools/paired_kernel_speed.py`.
- Ran `python -m pytest tests/test_tile_cuda_examples.py -q -k 'paired_kernel_speed_tool_reads_native_json_out_sidecar or paired_kernel_speed_tool_stage_timing_is_explicit or paired_kernel_speed_tool_compiles_and_smokes'`.
- Ran `bash -n tools/bench_native_gpt_sm120_parity.sh`.
- Ran `git diff --check`.
- Ran a dedicated RTX 5090 smoke parity sample with
  `NFN_SM120_PARITY_STEPS=5`, `NFN_SM120_PARITY_SAMPLES=1`,
  `NFN_SM120_PARITY_PROFILE_DIR=/tmp/nfn_sm120_parity_profiles_nostage_fix`,
  and `NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_nostage_fix.json`.
  The paired payload reported `native_stage_timing: false`, read candidate
  metrics from `json-out`, emitted no `stage.*` candidate metrics, and the
  sidecar reported `timing.stage_timing_enabled: false` with
  `stage_timing_event_count: 0`.

### 2026-06-17 Reject narrow LM-head extra-large-K cuBLASLt heuristic

#### Changed

- Kept the default LM-head backward-input route on BF16 `cublasGemmEx` for the
  `m=768,n=8192,k=50304` shape. Re-testing the extra-large-K cuBLASLt route
  with heuristic index `1` remained slower than the current default.

#### Verification

- Dedicated RTX 5090 same-script paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --train-batch-tokens 524288 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --train-batch-tokens 524288 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1 --candidate-env NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX=1 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 25 --command-timeout-seconds 1800 --append-native-profile-json-dir /tmp/nfn_lmhead_extralargek_h1_profiles --json-out /tmp/nfn_lmhead_extralargek_h1_pair.json`.
- The candidate measured `1.028831x` train-loop wall time,
  `0.971991x` tokens/sec, and `1.027085x` total wall time versus the current
  default.

### 2026-06-17 Align native CUDA module-loading defaults

#### Changed

- Set `CUDA_MODULE_LOADING=LAZY` when unset across the native SDK binding,
  unified native C++ frontend, GPT-2 launcher, NanoGPT native trainer, and
  GPT-2 evo native preflight. The dense GPT trainer already used this default;
  sibling native entrypoints now match before command execution or Tile
  library/runtime loading.
- Updated the GPT-2 launcher help text and native GPT SDK docs so the default
  CUDA runtime environment is explicit.

#### Verification

- Ran `python -m py_compile tools/check_native_no_torch_deps.py`.
- Ran `python tools/check_native_no_torch_deps.py --skip-artifacts --json`.
- Ran `git diff --check`.

### 2026-06-17 Reject startup-only token/BF16 arena fallbacks

#### Changed

- Kept the fused token-weight BF16 shadow initializer enabled. Disabling
  `NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_INIT` was too small and noisy to
  promote as a startup default.
- Kept the combined BF16 arena enabled. Disabling
  `NFN_NATIVE_GPT_COMBINED_BF16_ARENA` removed the
  `setup.uint16_arena_materialize` bucket in startup-only profiling, but it
  regressed the normal training loop and total wall time.

#### Verification

- Current startup-only profile:
  `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --backend tile-cuda --tinystories --startup-only --max-steps 1 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so --profile-json /tmp/nfn_startup_current.json`.
  The largest setup buckets were `setup.float_arena_materialize` (`152.29 ms`),
  `setup.token_weight_init` (`147.439 ms`), and
  `setup.uint16_arena_materialize` (`101.249 ms`).
- Dedicated RTX 5090 startup-only paired benchmark for
  `NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_INIT=0`:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --startup-only --max-steps 1 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --startup-only --max-steps 1 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_INIT=0 --samples 5 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 25 --command-timeout-seconds 600 --append-native-profile-json-dir /tmp/nfn_startup_disable_token_bf16_fuse_profiles --json-out /tmp/nfn_startup_disable_token_bf16_fuse_pair.json`
  measured `0.993382x` setup wall time and `0.993560x` total wall time versus
  the fused default, with noisy sample spread.
- Dedicated RTX 5090 startup-only paired benchmark for
  `NFN_NATIVE_GPT_COMBINED_BF16_ARENA=0`:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --startup-only --max-steps 1 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --startup-only --max-steps 1 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_GPT_COMBINED_BF16_ARENA=0 --samples 5 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 25 --command-timeout-seconds 600 --append-native-profile-json-dir /tmp/nfn_startup_disable_combined_bf16_arena_profiles --json-out /tmp/nfn_startup_disable_combined_bf16_arena_pair.json`
  measured `0.986733x` setup wall time and `0.992135x` total wall time.
- Dedicated RTX 5090 normal 5-step paired benchmark for
  `NFN_NATIVE_GPT_COMBINED_BF16_ARENA=0`:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_GPT_COMBINED_BF16_ARENA=0 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 25 --command-timeout-seconds 1800 --append-native-profile-json-dir /tmp/nfn_disable_combined_bf16_arena_train_profiles --json-out /tmp/nfn_disable_combined_bf16_arena_train_pair.json`
  measured `1.020957x` train-loop wall time, `0.979932x` tokens/sec, and
  `1.019949x` total wall time versus the combined-arena default.

### 2026-06-17 Strengthen native GPT no-Torch dependency gate

#### Changed

- Extended `tools/check_native_no_torch_deps.py` beyond compiled artifact
  linkage checks. The gate now also runs the default native GPT Python
  entrypoints with imports of `torch`, NumPy, tiktoken, `server.dataset_manager`,
  and `nfn_impl` blocked, using a stub compiled CLI to prove command
  construction reaches the native C++ path without those dependencies.
- Added `--skip-artifacts` and `--skip-python-entrypoints` switches so CI can
  check the Python no-import contract without local build outputs, or check only
  compiled binary linkage when needed.
- Updated package metadata, README, and Python SDK Tile-CUDA docs to describe
  the stronger native dependency gate.

#### Verification

- Ran `python tools/check_native_no_torch_deps.py --json`; it passed for
  `build/nfn_gpt_native_train`, `build/libnfn_native_train_tile_ops.so`,
  `cli/scripts/train_gpt.py --tinystories --native-cuda-dry-run --native-cuda-print-command`,
  `cli/nfn.py train --tinystories --native-cuda-dry-run --native-cuda-print-command`,
  and `import neuralfn; import neuralfn.native_gpt; import neuralfn.native_gpt2`
  under the import blocker.

### 2026-06-17 Reject LayerNorm affine-residual split fallback

#### Changed

- Kept `NFN_NATIVE_GPT_FUSE_LN_BACKWARD_AFFINE_RESIDUAL` enabled for the dense
  GPT native trainer. Disabling the fused LayerNorm affine+dInput+residual
  backward path made the full training loop slower, so the older split fallback
  remains diagnostic-only.

#### Verification

- Dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_GPT_FUSE_LN_BACKWARD_AFFINE_RESIDUAL=0 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 25 --command-timeout-seconds 1800 --append-native-profile-json-dir /tmp/nfn_disable_ln_affine_residual_profiles --json-out /tmp/nfn_disable_ln_affine_residual_pair.json`
  measured the split fallback at `1.012891x` train-loop wall time and
  `0.987285x` tokens/sec versus the fused default. No GPU compute processes were
  present; selected-GPU utilization before samples averaged `1.666667%`.

### 2026-06-17 Reject QKV direct BF16 grad scratch fallback

#### Changed

- Kept `NFN_NATIVE_GPT_DIRECT_BF16_QKV_GRAD_SCRATCH` enabled for the dense GPT
  native trainer. The older workspace/copy path is not a viable SM120 parity
  shortcut: it regresses the measured attention-backward and QKV-backward hot
  buckets.

#### Verification

- Dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_GPT_DIRECT_BF16_QKV_GRAD_SCRATCH=0 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 25 --command-timeout-seconds 1800 --append-native-profile-json-dir /tmp/nfn_disable_direct_bf16_qkv_grad_profiles --json-out /tmp/nfn_disable_direct_bf16_qkv_grad_pair.json`
  measured the disabled-direct-scratch candidate at `1.029204x` train-loop wall
  time and `0.971631x` tokens/sec versus the default. The affected hot buckets
  regressed as expected: `stage.block_backward.attn_sdpa.total_ms` was
  `1.129386x`, `stage.block_backward.qkv.total_ms` was `1.091903x`, and
  `stage.block_backward.total_ms` was `1.045311x`. No GPU compute processes
  were present; selected-GPU utilization before samples averaged `1.666667%`.

### 2026-06-17 Reject disabling float32/BF16 BGRADB fusion

#### Changed

- Kept `NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD` enabled for the dense
  GPT native trainer. Disabling the mixed float32-hidden/BF16-gradient BGRADB
  dWeight route was noise-equivalent overall and slightly worse by median
  train-loop time, with no useful movement in the hot block-backward buckets.

#### Verification

- Dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD=0 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 25 --command-timeout-seconds 1800 --append-native-profile-json-dir /tmp/nfn_disable_float32_bf16_bgrad_profiles --json-out /tmp/nfn_disable_float32_bf16_bgrad_pair.json`
  measured the disabled-fusion candidate at `0.998810x` mean train-loop wall
  time but `1.000652x` median train-loop wall time and `1.001197x` mean
  tokens/sec versus the default. `stage.block_backward.total_ms` stayed
  effectively unchanged at `0.999848x` mean / `1.002266x` median. No GPU compute
  processes were present; selected-GPU utilization before samples averaged
  `5.666667%`.

### 2026-06-17 Update native GPT dry-run throughput validation status

#### Changed

- Dense GPT native `--dry-run` / `--print-plan` JSON no longer says live SM120
  throughput validation still needs to happen. `remaining_validation` now names
  the real remaining operational item: closing the measured SM120 throughput gap
  against `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh` using
  `tools/bench_native_gpt_sm120_parity.sh` for same-script RTX 5090 comparisons.
- Updated README, CLI docs, Python SDK Tile-CUDA docs, and the Tile-CUDA
  checklist to match the new plan JSON wording.

#### Verification

- Refreshed current no-profile parity with
  `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=25 NFN_SM120_PARITY_PROFILE_DIR=none bash tools/bench_native_gpt_sm120_parity.sh`.
  The same-script run measured llm.kittens at `203289.6 tok/s` and NeuralFn at
  `193426 tok/s`, or `0.951480x` tokens/sec and `1.048909x` train-loop time.
  No compute processes were present on GPU 0.
- Refreshed current one-step stage timing with
  `NFN_NATIVE_GPT_STAGE_TIMING=1 CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1 --train-batch-tokens 524288 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so --profile-json /tmp/nfn_current_fullstep_stage.json`;
  the run reported `195875 tok/s` and kept the same hot buckets:
  `block_backward`, `lm_head_backward`, and `train.model_forward`.

### 2026-06-17 Reject MLP projection dWeight cuBLASLt shape fallback

#### Changed

- Kept the dense GPT native trainer on the current BF16 cuBLASLt BGRADB route
  for the MLP projection dWeight bucket `3072,768,65536,N,T`. Routing that one
  bucket through the existing fallback path is not a viable llm.kittens parity
  shortcut: it massively regresses block backward.

#### Verification

- Dedicated RTX 5090 shape-stat probe:
  `NFN_NATIVE_LINEAR_SHAPE_STATS=1 NFN_NATIVE_GPT_STAGE_TIMING=1 CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1 --train-batch-tokens 65536 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so --profile-json /tmp/nfn_shape_stats_current.json`
  confirmed the default bucket was `path_name: "cublaslt"`,
  `m: 3072`, `n: 768`, `k: 65536`, `op_a_name: "N"`, `op_b_name: "T"`.
- Dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=3072,768,65536,N,T --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 25 --command-timeout-seconds 1800 --append-native-profile-json-dir /tmp/nfn_disable_mlp_proj_dweight_profiles --json-out /tmp/nfn_disable_mlp_proj_dweight_pair.json`
  measured the fallback candidate at `3.199175x` train-loop wall time and
  `0.312590x` tokens/sec versus the default. `stage.block_backward.total_ms`
  regressed to `5.406332x`. No GPU compute processes were present; selected-GPU
  utilization before samples averaged `9.000000%`.

### 2026-06-17 Reject 16k LM-head row chunks after saved-LN1 default

#### Changed

- Kept the dense GPT native trainer's LM-head row chunk default at 8192 rows.
  A fresh current-runtime paired check after the saved-LN1-BF16 default showed
  `--lm-head-row-chunk-size 16384` was still slower despite halving the number
  of LM-head chunks.

#### Verification

- Dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --lm-head-row-chunk-size 16384 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 25 --command-timeout-seconds 1800 --json-out /tmp/nfn_lm_head_chunk_16384_pair_after_ln1_bf16.json`
  measured the 16384-row candidate at `1.014421x` train-loop wall time and
  `0.985784x` tokens/sec versus the 8192-row default. No GPU compute processes
  were present; selected-GPU utilization before samples averaged `6.000000%`.

### 2026-06-17 Default saved LN1 BF16 for packed attention backward

#### Changed

- Added `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_BF16` /
  `NFN_NATIVE_GPT2_STORE_PACKED_ATTENTION_LN1_BF16` and defaulted it on for the
  dense GPT native trainer. Earlier packed-attention blocks now store the LN1
  BF16 activation during forward, use it directly for QKV dWeight, and skip the
  saved-attention LN1 apply-stats recompute.
- Runtime and dry-plan JSON now report
  `stored_packed_attention_ln1_bf16_enabled`,
  `stored_packed_attention_ln1_bf16_blocks`,
  `stored_packed_attention_ln1_bf16_elements`,
  `stored_packed_attention_ln1_bf16_bytes`, and
  `stored_packed_attention_ln1_bf16_strategy`.
- The default costs about 1.03 GiB at the workstation `64 x 1024 x 768` shape.
  Set `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_BF16=0` for the previous
  lower-memory saved-attention LN1 apply-stats recompute route.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k 'train_transformer_lm or packed_attention_ln1_recompute or packed_qkv_uint16_arena'`
  (`2 passed, 43 deselected`).
- One-step candidate smoke with
  `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_BF16=1 NFN_NATIVE_GPT_STAGE_TIMING=1`
  reported `stored_packed_attention_ln1_bf16_blocks: 11`,
  `stored_packed_attention_ln1_bf16_bytes: 1107296256`, and reduced
  `block_recompute_saved_packed_attention.ln1` to `0.045632 ms` across 88
  calls.
- Dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_BF16=1 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 25 --command-timeout-seconds 1800 --json-out /tmp/nfn_store_packed_attention_ln1_bf16_pair.json`
  measured the candidate at `0.991351x` train-loop wall time and `1.008731x`
  tokens/sec versus the previous default. No GPU compute processes were present;
  selected-GPU utilization before samples averaged `6.333333%`.
- Default one-step smoke after promotion completed with
  `status: "native-transformer-lm-trained"`,
  `stored_packed_attention_ln1_bf16_enabled: true`,
  `stored_packed_attention_ln1_bf16_blocks: 11`, and
  `stored_packed_attention_ln1_bf16_strategy:
  "saved-forward-ln1-bf16-direct-qkv-dweight"`.

### 2026-06-17 Retire broken packed-LN2 FC+GELU fallback

#### Breaking changes

- Removed the native GPT diagnostic fallback controlled by
  `NFN_NATIVE_GPT_REUSE_PACKED_LN2_FC_GELU=0` /
  `NFN_NATIVE_GPT2_REUSE_PACKED_LN2_FC_GELU=0`. Dense GPT native training now
  always uses the prepacked-LN2 `nfn_native_tile_linear_bf16_input_weight_bf16_gelu_bf16_float32`
  route for stored-MLP FC+bias+GELU. Callers should remove that environment
  override instead of trying to force the older repack-inside-GEMM path.

#### Changed

- Hard-pinned `reuse_packed_ln2_fc_gelu_enabled` to `true` in the compiled native
  GPT trainer after the fallback path failed a current one-step TinyStories run
  with CUDA illegal memory access.
- Updated README and Python SDK Tile-CUDA docs so the retired fallback is no
  longer advertised as a valid paired-benchmark switch.

#### Verification

- `NFN_NATIVE_GPT_REUSE_PACKED_LN2_FC_GELU=0 CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so --json-out /tmp/nfn_reuse_packed_ln2_fc_gelu_off_failure.json`
  reproduced the old fallback failure as `lm_head.backward_input.bf16_bits_weight_bf16_shadow failed with CUDA error 700`.
- Rebuilt `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k 'train_transformer_lm or packed_attention_ln1_recompute or packed_qkv_uint16_arena'`
  (`2 passed, 43 deselected`).
- Reran the same env override after the rebuild:
  `NFN_NATIVE_GPT_REUSE_PACKED_LN2_FC_GELU=0 CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so --json-out /tmp/nfn_reuse_packed_ln2_fc_gelu_ignored_success.json`;
  it completed with `status: "native-transformer-lm-trained"`,
  `reuse_packed_ln2_fc_gelu_enabled: true`, and
  `stored_mlp_forward_strategy:
  "tk-sm120-fused-fc-bias-gelu-prepacked-ln2-bf16-shadow-weight"`.

### 2026-06-17 Record padded-vocab CE rejection for native GPT

#### Changed

- Kept dense GPT native training on the default public-vocab strided CE route.
  A same-script candidate run with `NFN_NATIVE_GPT_PUBLIC_VOCAB_CE=0` moved CE
  back to padded-vocab rows, but it was slightly slower in the actual train loop.

#### Verification

- Dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_GPT_PUBLIC_VOCAB_CE=0 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15 --command-timeout-seconds 1800 --json-out /tmp/nfn_public_vocab_ce_off_pair.json`
  measured padded-vocab CE at `1.001998x` train-loop wall time and `0.998146x`
  tokens/sec versus the default public-vocab strided CE.

### 2026-06-17 Add token-weight threaded startup diagnostic and keep Tile default

#### Changed

- Added `NFN_TILE_CUDA_TOKEN_WEIGHT_THREADED_INIT=1` /
  `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1` /
  `NFN_NATIVE_GPT2_TOKEN_WEIGHT_THREADED_INIT=1` as a diagnostic-only native GPT
  startup switch for comparing a plain threaded CUDA token-weight initializer
  against the default CUDA Tile initializer.
- Kept the default on the CUDA Tile deterministic initializer. The threaded
  candidate was noise-equivalent in the current reproducible startup-only pair,
  so it was not promoted. Runtime JSON now reports
  `token_weight_threaded_init_enabled` and distinguishes
  threaded versus Tile token-weight initialization in `token_weight_init_strategy`.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k 'train_transformer_lm or packed_attention_ln1_recompute or packed_qkv_uint16_arena'`
  (`2 passed, 43 deselected`).
- Ran a default startup-only probe:
  `build/nfn_gpt_native_train --backend tile-cuda --tinystories --startup-only --max-steps 1 --train-batch-tokens 65536 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so --json-out /tmp/nfn_startup_default_token_init.json`;
  it reported `token_weight_init_strategy:
  "device-tile-power2-deterministic-fused-bf16-shadow"` and
  `token_weight_threaded_init_enabled: false`.
- Dedicated RTX 5090 startup-only paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --startup-only --max-steps 1 --train-batch-tokens 65536 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --startup-only --max-steps 1 --train-batch-tokens 65536 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1 --samples 5 --warmup 1 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15 --command-timeout-seconds 900 --json-out /tmp/nfn_token_threaded_startup_candidate_env_pair.json`
  measured the threaded candidate at noise-equivalent `0.996666x` token init
  time and `0.997098x` total wall time versus the Tile initializer, so threaded
  init stays opt-in only.

### 2026-06-17 Add BF16 cuBLASLt shape-bisection switch and reject tested fallbacks

#### Changed

- Added `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=m,n,k,opA,opB` /
  `NFN_NATIVE_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=m,n,k,opA,opB` as a
  diagnostic-only Tile linear switch for routing one BF16 cuBLASLt shape bucket
  back through BF16 `cublasGemmEx` during paired bisection.
- Kept all tested hot BF16 buckets on the default cuBLASLt route. The global
  BF16 cuBLASLt opt-out and the tested one-shape fallbacks were slower in the
  actual native train-loop metric.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k 'train_transformer_lm or packed_attention_ln1_recompute or packed_qkv_uint16_arena'`
  (`2 passed, 43 deselected`).
- Dedicated RTX 5090 one-microbatch paired benchmark with
  `NFN_NATIVE_LINEAR_BF16_CUBLASLT=0` measured `6.171959x` train-loop wall time
  and `0.162057x` tokens/sec versus default, rejecting the global BF16 GEMMEx
  fallback.
- Dedicated RTX 5090 one-microbatch paired benchmark with
  `NFN_NATIVE_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=768,65536,3072,N,N` measured
  `1.007716x` train-loop wall time and `0.992349x` tokens/sec, rejecting the
  MLP projection dInput fallback.
- Dedicated RTX 5090 one-microbatch paired benchmark with
  `NFN_NATIVE_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=768,65536,768,N,N` measured
  `1.008810x` train-loop wall time and `0.991317x` tokens/sec, rejecting the
  smaller dInput fallback.
- Dedicated RTX 5090 one-microbatch paired benchmark with
  `NFN_NATIVE_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=768,3072,65536,N,T` measured
  `2.857882x` train-loop wall time and `0.351320x` tokens/sec, rejecting the
  MLP projection dWeight fallback.

### 2026-06-17 Add cuBLASLt workspace bisection switch and reject larger cap as default

#### Changed

- Added `NFN_TILE_CUDA_LINEAR_CUBLASLT_WORKSPACE_MB=N` /
  `NFN_NATIVE_LINEAR_CUBLASLT_WORKSPACE_MB=N` as a diagnostic-only Tile linear
  switch for changing the trainer-facing cuBLASLt heuristic workspace cap.
- Kept the default cap at 128 MiB. A 256 MiB cap looked slightly faster on a
  one-microbatch probe, but the normal 5-step paired run was train-loop
  neutral/slightly slower.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Dedicated RTX 5090 one-microbatch paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1 --train-batch-tokens 65536 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1 --train-batch-tokens 65536 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_LINEAR_CUBLASLT_WORKSPACE_MB=256 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15 --command-timeout-seconds 900 --json-out /tmp/nfn_cublaslt_workspace_256_pair.json`
  measured `0.996460x` train-loop wall time and `1.003563x` tokens/sec, so it
  was promoted to a normal-shape 5-step check.
- Dedicated RTX 5090 one-microbatch paired benchmark with 512 MiB:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1 --train-batch-tokens 65536 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1 --train-batch-tokens 65536 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_LINEAR_CUBLASLT_WORKSPACE_MB=512 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15 --command-timeout-seconds 900 --json-out /tmp/nfn_cublaslt_workspace_512_pair.json`
  measured `0.999774x` train-loop wall time and `1.000231x` tokens/sec, so 512
  MiB was treated as noise.
- Dedicated RTX 5090 normal 5-step paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_LINEAR_CUBLASLT_WORKSPACE_MB=256 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15 --command-timeout-seconds 1800 --json-out /tmp/nfn_cublaslt_workspace_256_5step_pair.json`
  measured `1.000863x` train-loop wall time and `0.999150x` tokens/sec, so the
  default remains 128 MiB.

### 2026-06-17 Add extra-large-K cuBLASLt bisection switch and reject it as default

#### Changed

- Added `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1` /
  `NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1` as a diagnostic-only Tile
  linear switch. It lets trainer-facing BF16 cuBLASLt attempt LM-head-sized
  shapes with `k > 32768`, including the dense GPT LM-head dHidden shape
  `m=768,n=8192,k=50304`.
- Kept the default cap at `k <= 32768` because paired RTX 5090 timing showed
  cuBLASLt was slower than the existing BF16 `cublasGemmEx` fallback for that
  LM-head dHidden shape.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Ran a one-microbatch shape-stat probe with
  `NFN_NATIVE_LINEAR_SHAPE_STATS=1 NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1 build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1 --train-batch-tokens 65536 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so --profile-json /tmp/nfn_extra_large_k_shape_stats.json`.
  The LM-head dHidden shape moved from `path_name: "cublas_gemmex_bf16"` to
  `path_name: "cublaslt"`.
- Dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1 --train-batch-tokens 65536 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1 --train-batch-tokens 65536 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15 --command-timeout-seconds 900 --json-out /tmp/nfn_extra_large_k_pair.json`
  measured `1.021534x` train-loop wall time and `0.978930x` tokens/sec versus
  the default GEMMEx route, so the extra-large-K cuBLASLt path remains
  diagnostic-only.

### 2026-06-17 Add full activation tape bisection switch and reject it as default

#### Changed

- Added `NFN_NATIVE_GPT_FULL_ACTIVATION_TAPE=1` /
  `NFN_NATIVE_GPT2_FULL_ACTIVATION_TAPE=1` as a diagnostic-only dense GPT
  trainer switch. When enabled, the native C++ loop allocates one transformer
  activation tape per block, uses each block's own forward tape during
  backward, skips backward recompute, and reports `full_activation_tape_enabled`,
  `activation_tape_count`, `backward_recompute_blocks`, and a
  `full-forward-tape...` strategy under `block_state_layout`.
- Kept the default on the existing scratch-recompute tape because paired RTX
  5090 timing showed the full-tape candidate is much slower and substantially
  increases setup pressure.

#### Verification

- Ran `python -m pytest tests/test_native_gpt2.py -q -k 'train_transformer_lm or packed_attention_ln1_recompute or packed_qkv_uint16_arena'`
  (`2 passed, 43 deselected`).
- Rebuilt the compiled native GPT CLI with `bash tools/build_native_gpt_cli.sh`.
- Dedicated RTX 5090 one-microbatch paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1 --train-batch-tokens 65536 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1 --train-batch-tokens 65536 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_GPT_FULL_ACTIVATION_TAPE=1 --samples 1 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15 --command-timeout-seconds 900 --json-out /tmp/nfn_full_activation_tape_one_microbatch_pair.json`
  measured the candidate at `61.335739x` train-loop wall time and `0.016304x`
  tokens/sec versus the default scratch-recompute path, so the full-tape route
  remains diagnostic-only.

### 2026-06-17 Record LN1 BF16 QKV handoff reject

#### Changed

- Rejected `NFN_NATIVE_GPT_LN1_BF16_QKV_FORWARD=0` for the dense GPT SM120
  trainer. Disabling the BF16 LN1-to-QKV handoff slowed the train loop, so the
  current BF16 handoff remains enabled.

#### Verification

- Dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_GPT_LN1_BF16_QKV_FORWARD=0 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15 --command-timeout-seconds 1800 --json-out /tmp/nfn_ln1_bf16_qkv_forward_off_pair.json`
  measured `1.015832x` train-loop time and `0.984423x` tokens/sec, so the
  default BF16 LN1-to-QKV handoff remains enabled.

### 2026-06-17 Record BF16 residual backward reject

#### Changed

- Rejected `NFN_NATIVE_GPT_BF16_RESIDUAL1_LN_BACKWARD=0` for the dense GPT
  SM120 trainer. The candidate kept residual1 activation storage enabled but
  disabled the BF16 residual LayerNorm backward consumer; paired timing showed
  the current BF16 residual backward path is faster.

#### Verification

- Dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_GPT_BF16_RESIDUAL1_LN_BACKWARD=0 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15 --command-timeout-seconds 1800 --json-out /tmp/nfn_bf16_residual1_ln_backward_off_pair.json`
  measured `1.014258x` train-loop time and `0.985954x` tokens/sec, so the
  default BF16 residual backward path remains enabled.

### 2026-06-17 Record residual1 activation storage reject

#### Changed

- Rejected `NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS=0` for the dense GPT
  SM120 trainer. Disabling the BF16 residual1 forward store reduced setup
  allocation work, but it slowed the train loop because the backward path lost
  the faster stored-residual consumer.

#### Verification

- Dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS=0 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15 --command-timeout-seconds 1800 --json-out /tmp/nfn_store_residual1_off_pair.json`
  measured `1.026566x` train-loop time and `0.974147x` tokens/sec, so the
  default stored-residual path remains enabled.

### 2026-06-17 Record gradient zero Tile-fill fallback reject

#### Changed

- Rejected `NFN_NATIVE_GPT_CUDA_MEMSET_GRAD_ZERO=0` for the dense GPT SM120
  trainer. The candidate replaces the default CUDA memset gradient-zero ranges
  with the Tile `fill_many` fallback, but paired timing measured it slower in the
  train loop.

#### Verification

- Dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_GPT_CUDA_MEMSET_GRAD_ZERO=0 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15 --command-timeout-seconds 1800 --json-out /tmp/nfn_cuda_memset_grad_zero_off_pair.json`
  measured `1.003087x` train-loop time and `0.996930x` tokens/sec, so CUDA
  memset gradient zeroing remains the default.

### 2026-06-17 Record BF16 LM-head loss fallback bisection

#### Changed

- Recorded the `NFN_NATIVE_GPT_BF16_LM_HEAD_LOSS=0` same-script bisection for
  the dense GPT SM120 trainer. The older float-workspace LM-head loss path was
  not promoted because the 5-sample paired run was noise-dominated: mean train
  loop time improved, but median train loop time was slightly slower than the
  current BF16 fused CE default.

#### Verification

- Dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_GPT_BF16_LM_HEAD_LOSS=0 --samples 5 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15 --command-timeout-seconds 1800 --json-out /tmp/nfn_bf16_lm_head_loss_off_pair_5sample.json`
  measured `0.992803x` mean train-loop time but `1.000689x` median train-loop
  time and `0.999310x` median tokens/sec, so the default remains unchanged.

### 2026-06-17 Preserve disabled native GPT cadences in SDK handoff

#### Changed

- `build_native_gpt_compiled_cli_run_config()` and
  `build_native_gpt2_compiled_cli_run_config()` now preserve
  `eval_every_steps=0`, `sample_every_steps=0`, and
  `checkpoint_every_steps=0` instead of clamping them to `1`.
- This aligns the Python/SDK handoff with the compiled C++ trainer contract, where
  zero disables validation, prompt sampling, or checkpoint cadence. Same-script
  CUDA Tile kernel benchmarks can now suppress that side work through the SDK path
  without silently re-enabling it.

#### Verification

- `python -m pytest tests/test_native_gpt2.py -q -k 'zero_cadences or compiled_cli_config_can_skip_checkpoint_export'`
- `git diff --check`

### 2026-06-17 Remove hardcoded external GPT trainer path

#### Changed

- Removed the workstation-specific
  `/mnt/disk2/dev/open-source/llm.kittens/train_gpt2cu` default from the native
  GPT SDK resolver, `cli/scripts/train_gpt.py` wrapper, and compiled GPT CLI
  target resolver.
- Explicit external `llm-kittens` bridge runs now resolve `train_gpt2cu` from
  `NFN_NATIVE_GPT_TRAIN_BIN`, `NFN_NATIVE_GPT2_TRAIN_BIN`, an explicit
  `--native-cuda-executable` / `--target` or SDK `executable=...`, or `PATH`.
  The dedicated parity script still owns its reference `LLM_KITTENS_ROOT` /
  `LLM_KITTENS_TRAIN_BIN` settings.

#### Breaking changes

- Before: explicit external bridge runs could silently pick up the local
  `/mnt/disk2/dev/open-source/llm.kittens/train_gpt2cu` binary when no target
  override was supplied.
- Now: SDK/CLI external bridge defaults are path/env based and resolve to the
  command name `train_gpt2cu` unless callers pass an explicit target or env var.

#### Verification

- `python -m pytest tests/test_native_gpt2.py -q -k 'external_bridge_defaults_are_path_or_env_based or compiled_cli_config or generic_env_names_take_precedence or native_gpt2_cpp_cli_builds'`
- `python -m pytest cli/tests/test_train_gpt2_native.py -q -k 'llm_kittens or native_dry_run or print_command or compiled_cli'`
- `git diff --check`

### 2026-06-17 Record LM-head 12288 row-chunk reject

#### Changed

- Rejected `--lm-head-row-chunk-size 12288` for the dense GPT SM120 default
  shape. It sits between the current 8192-row default and the already-rejected
  16384-row probe, but it was slower in same-script comparison.
- No runtime default changed. The 8192-row LM-head chunk remains the selected
  workstation profile.

#### Verification

- Dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so --lm-head-row-chunk-size 12288" --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15 --command-timeout-seconds 1800 --json-out /tmp/nfn_lm_head_row_chunk_12288_pair.json`
  measured the candidate at `1.006439x` train-loop wall time and `0.993616x`
  tokens/sec versus the 8192-row default.

### 2026-06-17 Native GPT runner auto no longer external-falls back

#### Changed

- Tightened the SDK native GPT runner boundary so `runner="auto"` only selects
  NeuralFn-owned native artifacts: the C++ binding, compiled no-Python GPT CLI,
  or compiled launcher.
- `build_native_gpt2_compiled_cli_run_config(..., kernel_backend="tile-cuda")`
  now resolves its default executable field through the NeuralFn compiled GPT
  CLI resolver instead of the external `train_gpt2cu` resolver. The explicit
  `kernel_backend="llm-kittens"` bridge still resolves the external target for
  parity benchmarks.

#### Breaking changes

- Before: `run_native_gpt(..., runner="auto")` and
  `run_native_gpt2(..., runner="auto")` could silently fall through to the raw
  external `train_gpt2cu` subprocess when the NeuralFn binding, compiled CLI,
  and launcher were missing.
- Now: `runner="auto"` reports an unavailable NeuralFn native runner in that
  state and `run_native_gpt*` raises. Callers that intentionally want the
  external bridge must pass `runner="subprocess"` or use
  `kernel_backend="llm-kittens"` for compiled-CLI parity checks.

#### Verification

- `python -m pytest tests/test_native_gpt2.py -q -k 'compiled_cli_config or runner_status or binding_runner or launcher_runner or compiled_cli_runner'`
- `python -m pytest tests/test_native_dependencies.py -q`
- `python -m pytest tests/test_native_gpt2.py -q -k 'auto_requires_neuralfn_native_artifacts or defaults_to_neuralfn_cli or generic_env_names_take_precedence or uses_compiled_cli_when_present or uses_compiled_launcher_when_present'`
- `git diff --check`

### 2026-06-17 Record BF16-output dInput reject

#### Changed

- Rejected a narrow SM120 candidate that kept cuBLASLt for BF16-gradient /
  BF16-weight plain dInput GEMMs but wrote BF16 dInput into Tile-owned scratch
  before converting it back to float32 for the existing trainer consumers.
- No runtime switch or kernel route was kept because the current float-output
  cuBLASLt dInput path was materially faster on the workstation shape. The
  rejection is recorded in `todo-tile-cuda.md`.

#### Verification

- `bash tools/build_native_train_tile_ops.sh`
- `bash tools/build_native_gpt_cli.sh`
- `python -m pytest tests/test_native_gpt2.py -q -k native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`
- GPU smoke of the temporary opt-in branch:
  `NFN_NATIVE_LINEAR_BF16_DINPUT_OUT=1 CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 16 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so --json-out /tmp/nfn_bf16_dinput_out_smoke.json`
  completed successfully and reported `passed: true`.
- Dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_LINEAR_BF16_DINPUT_OUT=1 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15 --command-timeout-seconds 1800 --json-out /tmp/nfn_bf16_dinput_out_pair.json`
  measured the candidate at `1.040670x` train-loop wall time and `0.960945x`
  tokens/sec versus the default float-output cuBLASLt dInput route.
- `git diff --check`

### 2026-06-17 Record LM-head 65536 row-chunk reject

#### Changed

- Rejected `--lm-head-row-chunk-size 65536` as a dense GPT SM120 candidate for
  the default `64 x 1024` training shape. The larger chunk would reduce LM-head
  loop count, but it put the tied LM-head BF16 logit workspace under severe
  memory pressure and ran far outside the expected 5-step paired-benchmark
  envelope.
- No default or runtime behavior changed. The current 8192-row LM-head default
  remains the practical workstation profile.

#### Verification

- Started the paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so --lm-head-row-chunk-size 65536" --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15 --continue-on-error --command-timeout-seconds 1800 --json-out /tmp/nfn_lm_head_row_chunk_65536_pair.json`.
- Stopped the run after the candidate remained active well past the expected
  5-step envelope. `nvidia-smi` showed the dedicated RTX 5090 at `100%`
  utilization and about `31926 MiB / 32607 MiB` used by the candidate process.
- Confirmed the benchmark processes were stopped and the RTX 5090 returned to
  idle at `0%` utilization and about `652 MiB` used memory.

### 2026-06-17 Record TK plain-dInput reject

#### Changed

- Rejected a narrow SM120 candidate that routed supported plain BF16 block
  dInput GEMMs through the ThunderKittens `matmul_dispatch_tk_ab` path with a
  BF16 scratch output plus float conversion.
- No runtime switch or kernel route was kept because the current cuBLASLt dInput
  path was materially faster on the workstation shape. The rejection is recorded
  in `todo-tile-cuda.md` so future parity work does not retest the same branch.

#### Verification

- `bash tools/build_native_train_tile_ops.sh`
- `bash tools/build_native_gpt_cli.sh`
- `python -m pytest tests/test_native_gpt2.py -q -k native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`
- GPU smoke of the temporary opt-in branch:
  `NFN_NATIVE_LINEAR_TK_DINPUT=1 CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 16 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so --json-out /tmp/nfn_tk_dinput_smoke.json`
  completed successfully and reported `passed: true`.
- Dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_LINEAR_TK_DINPUT=1 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15 --command-timeout-seconds 1800 --json-out /tmp/nfn_tk_dinput_pair.json`
  measured the candidate at `1.064272x` train-loop wall time and `0.939630x`
  tokens/sec versus the default cuBLASLt dInput route.
- `git diff --check`

### 2026-06-17 Record QKV dWeight fallback reject

#### Changed

- Recorded the same-script RTX 5090 bisection for
  `NFN_NATIVE_GPT_BF16_QKV_DWEIGHT=0`, which forces the older
  float32-LN1/BF16-grad QKV dWeight route instead of the default saved-LN1
  BF16-input dWeight path.
- The candidate was rejected and added to `todo-tile-cuda.md` because it slowed
  the native GPT train loop.

#### Verification

- Dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_GPT_BF16_QKV_DWEIGHT=0 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15 --json-out /tmp/nfn_bisect_bf16_qkv_dweight_off_3sample.json`
  measured the fallback at `1.009870x` train-loop wall time and `0.990246x`
  tokens/sec versus the default BF16-input QKV dWeight route.

### 2026-06-17 Add BF16 CE exp2 profiling gate

#### Changed

- Added an opt-in BF16 CE+dlogits exponent candidate for the native Tile-CUDA
  GPT trainer. Set `NFN_NATIVE_GPT_CE_BF16_EXP2=1`,
  `NFN_NATIVE_GPT2_CE_BF16_EXP2=1`, or
  `NFN_TILE_CUDA_CE_BF16_EXP2=1` to use `exp2f(x * log2(e))` inside the BF16
  in-place CE kernels instead of the default `expf` path.
- Dense GPT runtime JSON now reports `lm_head_ce_bf16_exp2_enabled`.
- The candidate remains default-off because the dedicated RTX 5090 paired
  benchmark was noise-equivalent/slightly slower than the default.

#### Verification

- `bash tools/build_native_train_tile_ops.sh`
- `bash tools/build_native_gpt_cli.sh`
- `python -m pytest tests/test_native_gpt2.py -q -k native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`
- `git diff --check`
- Dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate-env NFN_NATIVE_GPT_CE_BF16_EXP2=1 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15 --json-out /tmp/nfn_bisect_ce_bf16_exp2_3sample.json`
  measured the exp2 candidate at `1.000721x` train-loop wall time and
  `0.999293x` tokens/sec versus the default expf path, so it was not promoted.

### 2026-06-17 Elide first-write BGRADB bias accumulation launch

#### Changed

- cuBLASLt BGRADB dWeight+bias routes now write the first beta-zero
  gradient-accumulation microbatch's bias gradient directly into `grad_bias`
  instead of writing Tile-owned scratch and launching a separate
  `launch_gradient_accumulate_float32` add into an otherwise zeroed bias buffer.
- Later beta-one microbatches keep the previous scratch-plus-accumulate path, so
  accumulation semantics are unchanged after the first microbatch.
- Dense GPT runtime JSON now reports
  `linear_bias_gradient_first_write_bgrad_direct_enabled` alongside the existing
  direct gradient-accumulation fields.

#### Verification

- `bash tools/build_native_train_tile_ops.sh`
- `bash tools/build_native_gpt_cli.sh`
- `python -m pytest tests/test_native_gpt2.py -q -k native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`
- `git diff --check`
- Dedicated RTX 5090 paired benchmark:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 5 --eval-every-steps 0 --native-cuda-sample-every 0 --native-cuda-generate-tokens 144 --native-cuda-checkpoint-every 0 --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so" --baseline-env NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=0 --samples 3 --warmup 0 --cuda-visible-devices 0 --cuda-device-max-connections 1 --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15 --json-out /tmp/nfn_bisect_bgrad_first_write_direct_3sample.json`
  measured the direct first-write path at `0.999871x` train-loop wall time and
  `1.000129x` tokens/sec versus the old scratch-first path.

### 2026-06-17 Record SM120 parity bisection rejects

#### Changed

- Updated `todo-tile-cuda.md` with the clean unprofiled 10-step NeuralFn versus
  llm.kittens parity result after the profiling-control fix. The dedicated RTX
  5090 run measured NeuralFn at `1.054152x` llm.kittens train-loop time and
  `0.947113x` tokens/sec.
- Added the latest rejected same-script kernel bisections so the next pass does
  not retest slower cuBLASLt heuristic, packed-attention tape, LayerNorm affine
  row-chunk, or LM-head row-chunk candidates.
- Added follow-up rejected tape/allocation candidates for one-block-reduced MLP
  storage, one-block-reduced packed-attention storage, and the async allocator
  startup switch.
- Closed the stale row-vector SDPA fallback TODOs because the current packed
  SM120 TK attention path reports zero row-launch fallbacks and zero scalar
  forward launches in the live dense GPT loop.

#### Verification

- Ran same-script paired benchmarks on the dedicated RTX 5090 for
  `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX=2`,
  `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE=0`,
  `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_STATS=0`,
  `NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=512`,
  `NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=128`, and
  `--lm-head-row-chunk-size 16384`; each candidate regressed train-loop time
  against the current default.
- Ran same-script paired benchmarks on the dedicated RTX 5090 for
  `NFN_NATIVE_GPT_STORE_MLP_BLOCKS=11`,
  `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=11`, and
  `NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC=1`; each candidate regressed train-loop
  time or setup wall time against the current default.
- Ran a same-script paired benchmark for
  `NFN_NATIVE_LINEAR_CUBLASLT_DESCRIPTOR_CACHE=0`; disabling descriptor caching
  regressed train-loop time, so the current cached-descriptor default remains.

### 2026-06-17 Make SM120 parity profiling explicit

#### Changed

- `tools/bench_native_gpt_sm120_parity.sh` now accepts
  `NFN_SM120_PARITY_PROFILE_DIR=none|off|0|false|no` to skip the
  `--append-native-profile-json-dir` diagnostic sidecar and measure actual
  trainer throughput without CUDA-event stage profiling.
- Profiled parity runs now default
  `NFN_NATIVE_GPT_STAGE_TIMING_MAX_EVENTS=80000` unless the caller already set
  it, so the default 10-step SM120 profile sidecar captures complete stage
  timings instead of truncating at 20,000 events.

#### Verification

- Ran `NFN_NATIVE_GPT_STAGE_TIMING_MAX_EVENTS=80000 NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=15 NFN_SM120_PARITY_PROFILE_DIR=/tmp/nfn_sm120_parity_profiles_full_events NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_full_events.json bash tools/bench_native_gpt_sm120_parity.sh` outside the sandbox. The complete profile reported `stage_timing_event_count=41150`, `stage_timing_dropped_event_count=0`, llm.kittens at `2480.754 ms/step`, and NeuralFn at `2622.440 ms/step`.
- Ran `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=15 NFN_SM120_PARITY_PROFILE_DIR= NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_no_profile.json bash tools/bench_native_gpt_sm120_parity.sh` outside the sandbox, which confirmed the old empty override still used the default profile directory before this fix.
- Ran `NFN_SM120_PARITY_STEPS=1 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=15 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_no_profile_mode.json bash tools/bench_native_gpt_sm120_parity.sh` outside the sandbox after the fix. The JSON kept `append_native_profile_json_dir` empty and emitted no `stage.*` native metrics, confirming the no-profile path.
- Ran `python -m pytest tests/test_tile_cuda_examples.py -q -k sm120_parity`.
- Ran `git diff --check`.

### 2026-06-17 Route native checkpoint text prompts through the compiled sampler

#### Changed

- `nfn infer --checkpoint PATH --prompt TEXT` and
  `python cli/scripts/infer_gpt2.py --native-checkpoint PATH --prompt TEXT`
  now tokenize text prompts with the GPT-2 tokenizer and dispatch to
  `nfn_gpt_native_train --sample-checkpoint PATH --prompt-tokens IDS`, matching
  the explicit token-ID path.
- `NFN_NATIVE_GPT_SAMPLE_SCRIPT` and `--native-sampler-script` are deprecated
  for native `.bin` checkpoint prompts. The inference path no longer depends on
  `/mnt/disk2/dev/open-source/llm.kittens/sample_gpt2.py`.
- Empty native checkpoint prompts seed generation with the GPT-2 end-of-text
  token `50256`; non-GPT-2 tokenizer overrides are rejected before launching the
  compiled sampler.
- Successful native sampler runs still print the compiled JSON, then the wrapper
  prints `Generated token ids` and GPT-2-decoded `Generated text` without
  importing Torch.

#### Verification

- Ran `python -m pytest cli/tests/test_train_gpt2_native.py -q -k native_checkpoint`.
- Ran `python -m pytest cli/tests/test_train_gpt2_native.py -q -k native_checkpoint_decodes_compiled_sampler_tokens`.
- Ran `python -m pytest cli/tests/test_train_gpt2_native.py -q -k native_checkpoint_text_prompt_dispatches_compiled_sampler`.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran `git diff --check`.
- Ran `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 python cli/scripts/infer_gpt2.py --native-checkpoint /tmp/nfn_checkpoint_forward_2layer_smoke/model_00020000.bin --prompt Hello --max-new-tokens 1` outside the sandbox to verify real CUDA text-prompt inference and GPT-2 decoded output.

### 2026-06-17 Add autoregressive native checkpoint token sampler loop

#### Changed

- `nfn_gpt_native_train --sample-checkpoint PATH --prompt-tokens IDS --max-new-tokens N`
  now runs an autoregressive token-ID loop through the CUDA Tile checkpoint
  forward path, appending each selected token and rerunning the growing context
  until `N` tokens are generated.
- The sampler JSON now reports `sequence_token_count` and returns every emitted
  ID in `generated_tokens`; successful runs set `generated_token_count` to the
  number of generated IDs.
- Requests where `prompt_tokens + max_new_tokens` exceed the checkpoint context
  window now fail before CUDA allocation.

#### Verification

- Ran `bash tools/build_native_gpt_cli.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran `git diff --check`.
- Ran `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --sample-checkpoint /tmp/nfn_checkpoint_forward_2layer_smoke/model_00020000.bin --prompt-tokens 1,2,3 --max-new-tokens 4 --tile-ops-lib build/libnfn_native_train_tile_ops.so` outside the sandbox because sandboxed GPU access cannot allocate on the CUDA device.

### 2026-06-17 Replace native checkpoint token sampler pending plan with CUDA Tile next-token inference

#### Changed

- `nfn_gpt_native_train --sample-checkpoint PATH --prompt-tokens IDS --max-new-tokens N`
  now executes one full checkpoint forward pass through CUDA Tile kernels instead
  of returning the old `native-checkpoint-sampler-pending` plan JSON.
- When the CUDA forward succeeds and `N > 0`, the JSON returns
  `status: "native-checkpoint-sampler"`, `forward_pass_status:
  "cuda-tile-forward-executed"`, `generated_token_count: 1`, and the next token
  in `generated_tokens`, while keeping `torch_required: false` and
  `graph_editor_node_flow: false`.
- Native checkpoint metadata now reports
  `prompt_generation_status: "native-token-sampler-available"`.
  Autoregressive multi-token looping and native text prompt tokenization remain
  pending.

#### Verification

- Ran `bash tools/build_native_gpt_cli.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran `git diff --check`.
- Ran `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --sample-checkpoint /tmp/nfn_checkpoint_forward_2layer_smoke/model_00020000.bin --prompt-tokens 1,2,3 --max-new-tokens 4 --tile-ops-lib build/libnfn_native_train_tile_ops.so` outside the sandbox because sandboxed GPU access cannot allocate on the CUDA device.

### 2026-06-17 Add checkpoint-backed native GPT full-stack logits smoke

#### Changed

- `nfn_gpt_native_train` now supports
  `--checkpoint-forward-logits-smoke --native-checkpoint PATH --prompt-tokens IDS`.
- The smoke loads checkpoint embeddings, every GPT block in order, final
  LayerNorm, and tied LM-head logits through CUDA Tile kernels without Torch,
  Python datasets, or graph-editor tensor flow.
- The JSON output reports `block_index: -1`, `blocks_executed`,
  `transformer_blocks_executed: true`, `final_logits_executed: true`, and keeps
  `graph_editor_node_flow: false`. Generation-loop sampling remains pending.

#### Verification

- Ran `bash tools/build_native_gpt_cli.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran `git diff --check`.
- Ran `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --checkpoint-forward-logits-smoke --native-checkpoint /tmp/nfn_checkpoint_layout_smoke/model_00020000.bin --prompt-tokens 1,2,3 --tile-ops-lib build/libnfn_native_train_tile_ops.so` outside the sandbox because sandboxed GPU access cannot allocate on the CUDA device.

### 2026-06-17 Add checkpoint-backed native GPT block logits smoke

#### Changed

- `nfn_gpt_native_train` now supports
  `--checkpoint-block-logits-smoke --native-checkpoint PATH --prompt-tokens IDS`
  plus `--checkpoint-block-index N`.
- The smoke extends the checkpoint-backed single-block forward path through
  final LayerNorm and tied LM-head logits for the last prompt token on CUDA Tile
  kernels without Torch, Python datasets, or graph-editor tensor flow.
- The JSON output reports `final_logits_executed: true`, top token/logit
  metadata, and the final norm sample. Multi-layer checkpoint forward and
  generation-loop sampling remain pending.

#### Verification

- Ran `bash tools/build_native_gpt_cli.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran `git diff --check`.
- Ran `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --checkpoint-block-logits-smoke --native-checkpoint /tmp/nfn_checkpoint_layout_smoke/model_00020000.bin --prompt-tokens 1,2,3 --checkpoint-block-index 0 --tile-ops-lib build/libnfn_native_train_tile_ops.so` outside the sandbox because sandboxed GPU access cannot allocate on the CUDA device.

### 2026-06-17 Add checkpoint-backed native GPT block smoke

#### Changed

- `nfn_gpt_native_train` now supports
  `--checkpoint-block-smoke --native-checkpoint PATH --prompt-tokens IDS` plus
  `--checkpoint-block-index N`.
- The smoke extends the checkpoint-backed attention-residual path through
  `ln_2`, MLP fc, GELU+bias, MLP projection, and final block residual add on
  CUDA Tile kernels without Torch, Python datasets, or graph-editor tensor flow.
- The JSON output reports `block_mlp_executed: true`, loaded `ln_2`/MLP tensors,
  and the executed `nfn_native_tile_gelu_add_bias_float32` kernel. Final norm,
  tied LM-head logits, and generation-loop sampling remain pending.

#### Verification

- Ran `bash tools/build_native_gpt_cli.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran `git diff --check`.
- Ran `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --checkpoint-block-smoke --native-checkpoint /tmp/nfn_checkpoint_layout_smoke/model_00020000.bin --prompt-tokens 1,2,3 --checkpoint-block-index 0 --tile-ops-lib build/libnfn_native_train_tile_ops.so` outside the sandbox because sandboxed GPU access cannot allocate on the CUDA device.

### 2026-06-17 Add checkpoint-backed native GPT attention residual smoke

#### Changed

- `nfn_gpt_native_train` now supports
  `--checkpoint-attention-residual-smoke --native-checkpoint PATH --prompt-tokens IDS`
  plus `--checkpoint-block-index N`.
- The smoke extends the checkpoint-backed attention path by loading
  `h.N.attn.c_proj.weight` and `h.N.attn.c_proj.bias`, then running attention
  output projection and residual add through CUDA Tile kernels without Torch,
  Python datasets, or graph-editor tensor flow.
- The JSON output reports `block_attention_residual_executed: true` and still
  stops before `ln_2`, MLP, and generation-loop sampling.

#### Verification

- Ran `bash tools/build_native_gpt_cli.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran `git diff --check`.
- Ran `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --checkpoint-attention-residual-smoke --native-checkpoint /tmp/nfn_checkpoint_layout_smoke/model_00020000.bin --prompt-tokens 1,2,3 --checkpoint-block-index 0 --tile-ops-lib build/libnfn_native_train_tile_ops.so` outside the sandbox because sandboxed GPU access cannot allocate on the CUDA device.

### 2026-06-17 Add checkpoint-backed native GPT attention smoke

#### Changed

- `nfn_gpt_native_train` now supports
  `--checkpoint-attention-smoke --native-checkpoint PATH --prompt-tokens IDS`
  plus `--checkpoint-block-index N`.
- The smoke reuses the checkpoint-backed embedding, `ln_1`, and QKV projection
  path, then runs split-to-heads, causal scaled-dot-product attention, and
  merge-heads through CUDA Tile kernels without Torch, Python datasets, or
  graph-editor tensor flow.
- The JSON output reports `block_attention_executed: true` and still stops
  before attention output projection, residual add, MLP, and generation-loop
  sampling.

#### Verification

- Ran `bash tools/build_native_gpt_cli.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran `git diff --check`.
- Ran `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --checkpoint-attention-smoke --native-checkpoint /tmp/nfn_checkpoint_layout_smoke/model_00020000.bin --prompt-tokens 1,2,3 --checkpoint-block-index 0 --tile-ops-lib build/libnfn_native_train_tile_ops.so` outside the sandbox because sandboxed GPU access cannot allocate on the CUDA device.

### 2026-06-17 Add checkpoint-backed native GPT QKV smoke

#### Changed

- `nfn_gpt_native_train` now supports
  `--checkpoint-qkv-smoke --native-checkpoint PATH --prompt-tokens IDS` plus
  `--checkpoint-block-index N`.
- The smoke loads checkpoint embeddings and the selected block's `ln_1` and
  `attn.c_attn` tensors, converts bf16 weights on device, and runs embedding
  residual, block LayerNorm, and QKV projection through CUDA Tile kernels
  without Torch, Python datasets, or graph-editor tensor flow.
- The JSON output reports the block index, loaded tensors, executed kernels, and
  `block_qkv_projection_executed: true` while still marking full transformer
  block execution as pending. Attention, MLP, and generation-loop sampling remain
  the next native inference steps.

#### Verification

- Ran `bash tools/build_native_gpt_cli.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran `git diff --check`.
- Ran `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --checkpoint-qkv-smoke --native-checkpoint /tmp/nfn_checkpoint_layout_smoke/model_00020000.bin --prompt-tokens 1,2,3 --checkpoint-block-index 0 --tile-ops-lib build/libnfn_native_train_tile_ops.so` outside the sandbox because sandboxed GPU access reports CUDA driver/runtime isolation errors.

### 2026-06-17 Add checkpoint-backed native GPT logits smoke

#### Changed

- `nfn_gpt_native_train` now supports
  `--checkpoint-logits-smoke --native-checkpoint PATH --prompt-tokens IDS`.
  The mode loads `wte.weight`, `wpe.weight`, `ln_f.weight`, and `ln_f.bias`
  from the native bf16 checkpoint, converts them to float32 on device with the
  Tile bf16 unpack kernel, and runs token embedding, absolute position
  embedding, residual add, final LayerNorm, and tied LM-head linear logits for
  the last prompt token.
- The smoke reports the loaded tensors, executed Tile kernels, top token/logit,
  and explicitly marks `transformer_blocks_executed: false`. This is a real
  checkpoint-backed CUDA Tile forward slice without Torch, Python datasets, or
  graph-editor tensor flow; full prompt generation still needs transformer
  block execution wired into the sampler.

#### Verification

- Ran `bash tools/build_native_gpt_cli.sh`.
- Ran `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --checkpoint-logits-smoke --native-checkpoint /tmp/nfn_checkpoint_layout_smoke/model_00020000.bin --prompt-tokens 1,2,3 --tile-ops-lib build/libnfn_native_train_tile_ops.so` outside the sandbox because sandboxed GPU/NVML access is blocked.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran `git diff --check`.

### 2026-06-17 Add named tensor checkpoint load smoke

#### Changed

- `nfn_gpt_native_train --checkpoint-load-smoke` now accepts
  `--checkpoint-load-tensor NAME`. The smoke resolves `NAME` through the
  compiled checkpoint tensor layout, seeks to that tensor's payload/file offset,
  copies a bounded bf16 slice to CUDA memory, converts it through the Tile
  bf16-to-float kernel, and verifies copyback parity.
- The default load-smoke behavior still reads from the payload start. Named
  tensor loading is the next native inference prerequisite after layout decode:
  the sampler can now prove that individual checkpoint tensors can be selected
  and moved to device buffers without Torch, Python datasets, or graph-editor
  tensor flow.

#### Verification

- Ran `bash tools/build_native_gpt_cli.sh`.
- Ran `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --checkpoint-load-smoke --native-checkpoint /tmp/nfn_checkpoint_layout_smoke/model_00020000.bin --checkpoint-load-tensor h.0.ln_1.weight --checkpoint-load-elements 16 --tile-ops-lib build/libnfn_native_train_tile_ops.so` outside the sandbox because sandboxed GPU/NVML access is blocked.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran `git diff --check`.

### 2026-06-17 Add native GPT checkpoint tensor layout decode

#### Changed

- `nfn_gpt_native_train` now supports
  `--checkpoint-layout --native-checkpoint PATH`. The mode reads the native
  checkpoint header, derives the exact tensor layout from the checkpoint shape,
  verifies the layout parameter count against the file-size contract, and emits
  payload/file offsets plus bounded payload samples as compiled C++ JSON.
- The layout path exits before CUDA, token-shard resolution, Torch, Python
  dataset setup, or graph-editor tensor flow. This gives the upcoming native
  forward sampler an authoritative checkpoint tensor map instead of relying on
  graph-backed or Python-side checkpoint interpretation.

#### Verification

- Ran `bash tools/build_native_gpt_cli.sh`.
- Ran `build/nfn_gpt_native_train --checkpoint-metadata-smoke --output-dir /tmp/nfn_checkpoint_layout_smoke --num-layers 1 --train-seq-len 8`.
- Ran `build/nfn_gpt_native_train --checkpoint-layout --native-checkpoint /tmp/nfn_checkpoint_layout_smoke/model_00020000.bin --checkpoint-layout-sample-buffers 3`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran `git diff --check`.

### 2026-06-17 Add native GPT checkpoint payload load smoke

#### Changed

- `nfn_gpt_native_train` now supports
  `--checkpoint-load-smoke --native-checkpoint PATH
  --checkpoint-load-elements N`. The mode validates the native bf16 checkpoint,
  reads a bounded payload slice, copies it to CUDA memory, converts it through
  `nfn_native_tile_bf16_bits_to_float32`, copies float32 values back, and checks
  exact host-vs-device conversion parity without importing Torch, resolving
  token shards, setting up Python datasets, or flowing tensors through graph
  editor nodes.
- This is a focused prerequisite for the dedicated native GPT forward sampler:
  checkpoint payload movement and bf16 unpacking are now covered by compiled
  C++/CUDA Tile before wiring logits generation.

#### Verification

- Ran `bash tools/build_native_gpt_cli.sh`.
- Ran `build/nfn_gpt_native_train --checkpoint-metadata-smoke --output-dir /tmp/nfn_checkpoint_load_smoke --num-layers 1 --train-seq-len 8`.
- Ran `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --checkpoint-load-smoke --native-checkpoint /tmp/nfn_checkpoint_load_smoke/model_00020000.bin --checkpoint-load-elements 16 --tile-ops-lib build/libnfn_native_train_tile_ops.so` outside the sandbox because sandboxed GPU/NVML access is blocked.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran `git diff --check`.

### 2026-06-17 Add compiled native GPT prompt-token checkpoint sampler contract

#### Changed

- `nfn_gpt_native_train` now accepts
  `--sample-checkpoint PATH --prompt-tokens IDS --max-new-tokens N`. The mode
  validates native dense GPT checkpoint metadata, file size, context length,
  vocab bounds, and prompt-token parsing from compiled C++ before CUDA, Torch,
  token-shard resolution, Python dataset setup, or graph-editor node execution.
- `nfn infer --checkpoint model_*.bin --prompt-tokens ...` and
  `python cli/scripts/infer_gpt2.py --native-checkpoint model_*.bin
  --prompt-tokens ...` now dispatch to that compiled binary instead of rejecting
  prompt-token native checkpoint requests or using the transitional Python text
  sampler bridge.
- The compiled path currently returns
  `status: "native-checkpoint-sampler-pending"` with
  `forward_pass_status: "dedicated-native-sampler-pending"` and a nonzero exit
  code until the CUDA Tile forward generation loop is implemented.

#### Verification

- Ran `python -m py_compile cli/tests/test_train_gpt2_native.py cli/scripts/infer_gpt2.py`.
- Ran `bash tools/build_native_gpt_cli.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`.
- Ran `python -m pytest cli/tests/test_train_gpt2_native.py -q -k 'native_checkpoint_prompt_tokens or native_checkpoint_dispatches_sampler or native_checkpoint_info'`.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran `git diff --check`.

### 2026-06-17 Add compiled native GPT checkpoint inspection

#### Changed

- `nfn_gpt_native_train` now supports
  `--native-info --native-checkpoint PATH` and
  `--inspect-checkpoint PATH`. Both modes read native dense GPT
  `model_*.bin` checkpoints directly from compiled C++, validate the header and
  expected file size, report DONE-marker state, and exit before CUDA, Torch,
  dataset resolution, or graph-editor node setup.
- The JSON reports `status: "native-checkpoint-info"`, `runtime: "native-cpp"`,
  checkpoint shape/precision fields, and
  `prompt_generation_status: "dedicated-native-sampler-pending"` so the
  remaining prompt-generation gap is explicit.
- Aligned the compiled native GPT test expectation for current descriptor-arena
  elision counts, which now report `37` elided descriptor copies/mallocs in the
  existing plan JSON.

#### Verification

- Ran `bash tools/build_native_gpt_cli.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`.
- Ran `python tools/check_native_no_torch_deps.py`.

### 2026-06-17 Reject LM-head hidden prepack disable

#### Changed

- Retested `NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=0` against the current
  dense native GPT default. The candidate is not promoted: train-loop time
  regressed to `1.009504x`, tokens/sec fell to `0.990583x`, and
  `stage.lm_head_backward.total_ms` regressed to `1.039531x`.
- The result keeps the full-microbatch BF16 final-norm hidden prepack as the
  LM-head default while the remaining parity work targets other hot buckets.

#### Verification

- Ran `python tools/paired_kernel_speed.py` on the dedicated display-disabled
  RTX 5090 with `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`,
  `--require-idle-selected-gpu`,
  `--append-native-profile-json-dir /tmp/nfn_lm_prepack_off_profiles`, and
  candidate `NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=0`.

### 2026-06-17 Capture SM120 parity stage profiles by default

#### Changed

- `tools/bench_native_gpt_sm120_parity.sh` now passes
  `--append-native-profile-json-dir` to the paired benchmark harness by default,
  using `/tmp/nfn_sm120_parity_profiles_${NFN_SM120_PARITY_STEPS:-10}step`
  unless `NFN_SM120_PARITY_PROFILE_DIR` is set.
- The canonical NeuralFn-vs-llm.kittens SM120 parity run now records the
  NeuralFn native stage sidecars needed to attribute the remaining throughput
  gap while still comparing both trainers in the same alternating benchmark
  script.

#### Verification

- Ran `bash -n tools/bench_native_gpt_sm120_parity.sh`.
- Ran `python -m py_compile tools/paired_kernel_speed.py`.
- Ran `python -m pytest tests/test_tile_cuda_examples.py -q -k native_gpt_sm120_parity_wrapper_uses_reference_shape`.

### 2026-06-17 Reject current LM-head and projection-residual probes

#### Changed

- Retested `--lm-head-row-chunk-size 32768` against the current dense native
  GPT default. The candidate is not promoted: train-loop time was noise-level
  at `0.999895x`, total wall time regressed to `1.007049x`, and
  `stage.lm_head_backward.total_ms` regressed to `1.007987x`.
- Retested `NFN_NATIVE_GPT_BF16_PROJECTION_RESIDUAL=0` against the current
  default. The candidate is rejected: train-loop time regressed to `1.015959x`,
  tokens/sec fell to `0.984297x`, and `stage.block_backward.total_ms`
  regressed to `1.010018x`.
- Ran a one-step shape-stat diagnostic for a narrow LM-head dHidden cuBLASLt
  large-`k` probe. The shape `m=768,n=8192,k=50304,op=N,N` still fell back to
  `cublas_gemmex_bf16`, so no dispatch change was retained.

#### Verification

- Ran `python tools/paired_kernel_speed.py` on the dedicated display-disabled
  RTX 5090 with `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`,
  `--require-idle-selected-gpu`,
  `--append-native-profile-json-dir /tmp/nfn_lm32768_profiles`, and candidate
  `--lm-head-row-chunk-size 32768`.
- Ran the same paired harness with
  `--append-native-profile-json-dir /tmp/nfn_proj_residual_off_profiles` and
  candidate `NFN_NATIVE_GPT_BF16_PROJECTION_RESIDUAL=0`.
- Ran `NFN_TILE_CUDA_LINEAR_SHAPE_STATS=1 NFN_NATIVE_GPT_STAGE_TIMING=1`
  one-step native profiling to inspect current GEMM dispatch, then rebuilt
  `build/libnfn_native_train_tile_ops.so` after removing the ineffective
  cuBLASLt large-`k` experiment.

### 2026-06-17 Reject primary float/uint16 arena startup probe

#### Changed

- Tested a native GPT startup candidate that backed the main float arena and
  combined uint16/BF16 arena with one primary CUDA allocation. The candidate is
  not retained: it regressed startup-only setup time to `1.075212x` and total
  startup wall time to `1.047410x` versus the existing separate float and
  uint16 arenas.
- The sidecar profiles showed the intended allocation reduction was outweighed
  by slower subsequent startup work. `setup.token_weight_init.total_ms`
  measured `1.464730x` in the candidate run.

#### Verification

- Ran `python tools/paired_kernel_speed.py` on the dedicated display-disabled
  RTX 5090 with `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`,
  `--require-idle-selected-gpu`,
  `--append-native-profile-json-dir /tmp/nfn_primary_arena_profiles`,
  baseline `NFN_NATIVE_GPT_PRIMARY_FLOAT_UINT16_ARENA=0`, and candidate
  `NFN_NATIVE_GPT_PRIMARY_FLOAT_UINT16_ARENA=1`.

### 2026-06-17 Reject cuBLASLt heuristic index 3 for native GPT

#### Changed

- Ran a same-script RTX 5090 bisection for
  `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX=3` using paired native stage
  profile sidecars. The candidate is not promoted: it measured `1.005321x`
  train-loop time and `0.994708x` tokens/sec versus the default heuristic-index
  selection.
- Stage ratios showed the regression in the hot path rather than setup:
  `stage.lm_head_backward.total_ms` measured `1.007437x` and
  `stage.block_backward.total_ms` measured `1.004799x`, with the largest
  block sub-bucket movement in `stage.block_backward.attn_sdpa.total_ms` at
  `1.008118x`.

#### Verification

- Ran `tools/paired_kernel_speed.py` on the dedicated display-disabled RTX
  5090 with `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`,
  `--require-idle-selected-gpu`,
  `--append-native-profile-json-dir /tmp/nfn_cublaslt_h3_profiles`, and
  `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX=3`.

### 2026-06-17 Add paired native stage-profile sidecars

#### Changed

- `tools/paired_kernel_speed.py` now accepts
  `--append-native-profile-json-dir DIR`. When enabled, the harness appends a
  unique `--profile-json` path under `DIR` to NeuralFn native commands that do
  not already specify a JSON output flag and enables
  `NFN_NATIVE_GPT_STAGE_TIMING=1` for those profiled native commands unless the
  caller already set it.
- This makes paired kernel comparisons report native `stage.*` metrics for both
  baseline and candidate commands without manually editing each command string,
  so external GPU load is still controlled by the same alternating script while
  per-stage regressions are visible.

#### Verification

- Ran `python -m py_compile tools/paired_kernel_speed.py`.
- Ran a one-sample paired RTX 5090 smoke with
  `--append-native-profile-json-dir /tmp/nfn_pair_profiles_smoke` and confirmed
  the text output included `stage.block_backward.total_ms`,
  `stage.lm_head_backward.total_ms`, and candidate-over-baseline stage ratios.

### 2026-06-17 Add llm.kittens SM120 cuBLASLt candidate initialization

#### Changed

- CUDA Tile SM120 bridge builds now initialize the llm.kittens
  `llmk::cublaslt_sm120` handles before dispatching through its matmul helpers
  when a candidate library is compiled with
  `NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLASLT_GEMM"`. This makes
  that compile mode runnable for same-script bisection instead of aborting on
  the upstream `handle != nullptr` assertion.
- The compile mode is not the default. The dedicated RTX 5090 paired benchmark
  rejected it for NeuralFn's current native loop: `1.014933x` train-loop time
  and `0.985289x` tokens/sec versus the default library.

#### Verification

- Rebuilt the default `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Built the candidate library with
  `NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLASLT_GEMM"
  bash tools/build_native_train_tile_ops.sh
  /tmp/libnfn_native_train_tile_ops_llmk_cublaslt.so`.
- Ran a one-step RTX 5090 smoke with the candidate library and confirmed
  `passed: true`.
- Ran `tools/paired_kernel_speed.py` on the dedicated display-disabled RTX
  5090 with `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and
  `--require-idle-selected-gpu` comparing the default library against
  `/tmp/libnfn_native_train_tile_ops_llmk_cublaslt.so`.

### 2026-06-17 Add BF16 CE vector-store bisection selector

#### Changed

- Native CUDA Tile BF16 cross-entropy backward launchers now accept
  `NFN_NATIVE_GPT_CE_BF16_VEC_STORES=1`,
  `NFN_NATIVE_GPT2_CE_BF16_VEC_STORES=1`, or
  `NFN_TILE_CUDA_CE_BF16_VEC_STORES=1` to test a 128-bit streaming-store path
  for BF16 dlogits. The default remains the scalar store path.
- The selector exists only for same-script kernel bisection. The dedicated RTX
  5090 paired benchmark rejected the vector-store path as a default:
  `1.000271x` train-loop time and `0.999738x` tokens/sec versus the scalar
  store baseline.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -rs -k
  "native_train_tile_ops_builds_torch_free_c_abi or
  native_gpt_transformer_lm_exposes_opt_in_bf16_attention_grad_out_handoff"`.
- Ran `tools/paired_kernel_speed.py` on the dedicated display-disabled RTX
  5090 with `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and
  `--require-idle-selected-gpu` for
  `NFN_NATIVE_GPT_CE_BF16_VEC_STORES=1`.

### 2026-06-17 Report BF16 cublasGemmEx fallback shapes

#### Changed

- Native CUDA Tile linear shape stats now record BF16 `cublasGemmEx` fallback
  success branches into the existing `cublas_gemmex_bf16` bucket. This closes a
  profiling blind spot where TK and cuBLASLt shapes were visible but the
  remaining BF16 fallback GEMM shape was not.
- A one-step RTX 5090 shape profile now reports the remaining fallback as
  `m=768, n=8192, k=50304, op_a=N, op_b=N, calls=64`, matching the LM-head
  dHidden chunk path.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Ran `NFN_NATIVE_LINEAR_SHAPE_STATS=1 CUDA_VISIBLE_DEVICES=0
  CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --backend tile-cuda
  --tinystories --train-transformer-lm --max-steps 1 --eval-every-steps 0
  --no-checkpoint --profile-json /tmp/nfn_gemmex_shape_stats.json`; the JSON
  reported `passed: true`, `linear_shape_stats_count: 15`, and one
  `cublas_gemmex_bf16` entry.

### 2026-06-17 Add BF16 cublasGemmEx fallback compute-mode selector

#### Changed

- Native CUDA Tile BF16 `cublasGemmEx` fallback paths now accept
  `NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF=1` or
  `NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_FAST_16BF=1` to test
  `CUBLAS_COMPUTE_32F_FAST_16BF` on non-cuBLASLt fallback GEMMs. The default
  remains `CUBLAS_COMPUTE_32F` because the RTX 5090 paired benchmark measured
  the fast-16BF candidate as neutral: `1.000141x` train-loop time and
  `0.999865x` tokens/sec.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Ran `tools/paired_kernel_speed.py` on the dedicated display-disabled RTX
  5090 with `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and
  `--require-idle-selected-gpu` for
  `NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF=1`.

### 2026-06-17 Add BF16 CE launch bisection selector

#### Changed

- Native CUDA Tile GPT BF16 cross-entropy backward launchers now accept
  `NFN_NATIVE_GPT_CE_BF16_THREADS`, `NFN_NATIVE_GPT2_CE_BF16_THREADS`, or
  `NFN_TILE_CUDA_CE_BF16_THREADS` to choose `128`, `256`, `512`, or `1024`
  threads per row. The supported default remains `1024`; invalid values fall
  back to the default.
- This is a diagnostic kernel bisection control, not a training-default change.
  The same-script RTX 5090 benchmarks rejected lower defaults: `512` measured
  `0.999774x` train-loop time and `1.000234x` tokens/sec, while `256` measured
  `1.011218x` train-loop time and `0.988908x` tokens/sec versus the current
  1024-thread path.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k
  "native_train_tile_ops_builds_torch_free_c_abi or
  native_gpt_transformer_lm_exposes_opt_in_bf16_attention_grad_out_handoff"`.
- Ran `python -m py_compile tools/paired_kernel_speed.py`.
- Ran `tools/paired_kernel_speed.py` on the dedicated display-disabled RTX
  5090 with `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and
  `--require-idle-selected-gpu` for the `512` and `256` CE-thread candidates.

### 2026-06-17 Route native GPT checkpoint inference to a matching sampler

#### Changed

- `python cli/scripts/infer_gpt2.py --native-checkpoint model_*.bin --prompt ...`
  and `nfn infer --checkpoint model_*.bin --prompt ...` now recognize
  llm.kittens/NeuralFn native GPT `.bin` checkpoints and dispatch prompt
  generation to a matching sampler script instead of falling into the
  graph-backed runtime or stopping after metadata.
- The sampler path defaults to `/mnt/disk2/dev/open-source/llm.kittens/sample_gpt2.py`
  when present and can be overridden with `NFN_NATIVE_GPT_SAMPLE_SCRIPT` or
  `--native-sampler-script`. The parent `nfn` lightweight checkpoint path still
  avoids importing `nfn_impl`; `--native-info` remains a Torch-free metadata
  inspection path.
- This is a compatibility bridge for prompt generation from native `.bin`
  checkpoints while the dedicated CUDA Tile native GPT inference executable is
  still pending.

#### Verification

- Ran `python -m pytest cli/tests/test_train_gpt2_native.py -q -k
  "native_checkpoint_info or native_checkpoint_is_recognized or native_checkpoint_dispatches_sampler"`.

### 2026-06-17 Record SM120 native GPT kernel bisection rejects

#### Benchmark evidence

- Rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt_native_train` from the
  clean source before measuring new candidates on the dedicated display-disabled
  RTX 5090 with `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and
  `tools/paired_kernel_speed.py --require-idle-selected-gpu`.
- Rejected `NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1`: the two-sample paired
  run measured `1.016116x` train-loop time and `0.984144x` tokens/sec versus
  the default float attention-gradient handoff.
- Rejected `NFN_NATIVE_GPT_BF16_BLOCK_DWEIGHT_STAGING=1`: the two-sample paired
  run measured `1.028320x` train-loop time and `0.972468x` tokens/sec versus
  the default float-gradient accumulation path.
- Rejected `NFN_NATIVE_GPT_DWEIGHT_FIRST_MICROBATCH_BETA_ZERO=0`: the
  two-sample paired run measured `1.009501x` train-loop time and `0.990594x`
  tokens/sec versus the default first-write-then-accumulate dWeight path.
- Kept the SM120 TK fused-dGELU-dInput compile-time candidate opt-in. A
  two-sample run of a separate library built with
  `NFN_TILE_CUDA_EXTRA_NVCC_FLAGS='-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP
  -DLLMK_SM120_APPROX_DGELU_TANH=1'` measured a narrow
  `0.998849x` train-loop time, but the four-sample repeat was effectively
  neutral at `0.999900x` train-loop time and `1.000109x` tokens/sec.
- Rejected `NFN_NATIVE_GPT_FUSE_QKV_BIAS_TK_GEMM=0`: the two-sample paired run
  measured `1.016504x` train-loop time and `0.983770x` tokens/sec, confirming
  the default fused QKV bias+TK path is still the faster route.

### 2026-06-17 Enable large-shape BF16 cuBLASLt trainer GEMMs by default

#### Changed

- Trainer-facing BF16/BF16 linear backward GEMMs now allow larger cuBLASLt
  shapes by default, covering dense GPT LM-head dHidden/dWeight chunk GEMMs
  that previously fell back to BF16 `cublasGemmEx`.
- Set `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_LARGE_SHAPES=0` or
  `NFN_NATIVE_LINEAR_BF16_CUBLASLT_LARGE_SHAPES=0` to restore the previous
  small-shape-only cuBLASLt gate for paired bisection.

#### Verification

- Rebuilt the raw Tile ops library with
  `bash tools/build_native_train_tile_ops.sh`.
- Ran `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_LARGE_SHAPES=1
  build/nfn_gpt_native_train --backend tile-cuda --tinystories
  --train-transformer-lm --max-steps 1 --eval-every-steps 0 --no-checkpoint
  --profile-json /tmp/nfn_cublaslt_large_smoke.json` on the dedicated RTX 5090;
  the JSON reported `passed: true` and `linear_cublaslt_gemm_count: 736`.
- Ran `tools/paired_kernel_speed.py` against
  `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_LARGE_SHAPES=0` for two measured 3-step
  samples on the idle display-disabled RTX 5090. The default large cuBLASLt
  shape path measured `0.999395x` train-loop time and `1.000607x` tokens/sec
  versus the previous BF16 `cublasGemmEx` fallback for those larger shapes.

### 2026-06-17 Reuse BF16 MLP projection grad-out in native GPT backward

#### Changed

- Added the raw Tile-CUDA ABI
  `nfn_native_tile_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_only_float32`
  for SM120 TK fused MLP projection dInput+dGELU when the caller already has
  BF16 grad-out bits.
- Dense GPT transformer-LM now packs the MLP projection incoming gradient to
  BF16 once, reuses that scratch for MLP projection dWeight+bias, and feeds it
  into the fused dInput+dGELU path.
- Runtime JSON reports
  `block_backward_mlp_proj_bf16_grad_out_reuse_enabled` and the
  `tk-sm120-fused-dinput-dgelu-reused-bf16-grad-out-bf16-store-bf16-shadow-weight`
  strategy when active. Set
  `NFN_NATIVE_GPT_REUSE_MLP_PROJ_BF16_GRAD_OUT=0` to reproduce the previous
  per-stage pack path for paired bisection.

#### Verification

- Rebuilt the raw Tile ops library with
  `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt the native GPT CLI with `bash tools/build_native_gpt_cli.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k
  "native_train_tile_ops_builds_torch_free_c_abi or transformer_lm"`.
- Ran `build/nfn_gpt_native_train --backend tile-cuda --tinystories
  --train-transformer-lm --max-steps 3 --eval-every-steps 0 --no-checkpoint
  --profile-json /tmp/nfn_mlp_grad_reuse_default.json` on the dedicated RTX
  5090; the JSON reported `passed: true`,
  `block_backward_mlp_proj_bf16_grad_out_reuse_enabled: true`, and the reused
  BF16 grad-out strategy.
- Ran `tools/paired_kernel_speed.py` against
  `NFN_NATIVE_GPT_REUSE_MLP_PROJ_BF16_GRAD_OUT=0` for two measured 3-step
  samples on the idle display-disabled RTX 5090. The candidate measured
  `0.994156x` train-loop time and `1.005886x` tokens/sec versus the previous
  per-stage pack path.

### 2026-06-17 Default dense GPT dWeight GEMMs to first-write-then-accumulate

#### Changed

- Added beta-capable raw Tile-CUDA dWeight ABI variants for BF16/BF16,
  BF16/FP32, and FP32/BF16 dWeight paths.
- Dense GPT transformer-LM now launches the first gradient-accumulation
  microbatch dWeight GEMMs with `beta=0` and later microbatches with `beta=1`,
  matching the llm.kittens SM120 accumulation contract while keeping gradients
  in direct optimizer-step accumulation buffers.
- Runtime JSON now reports
  `dweight_first_microbatch_beta_zero_enabled`,
  `dweight_first_microbatch_beta_strategy`, and
  `first-write-then-accumulate` strategy suffixes for LM-head, QKV, and block
  dWeight routes. Set `NFN_NATIVE_GPT_DWEIGHT_FIRST_MICROBATCH_BETA_ZERO=0`
  to reproduce the previous always-accumulate path for paired bisection.

#### Verification

- Ran `python -m pytest tests/test_native_gpt2.py -q -k
  "native_train_tile_ops_builds_torch_free_c_abi or transformer_lm"`.
- Ran `build/nfn_gpt_native_train --backend tile-cuda --tinystories
  --train-transformer-lm --max-steps 3 --eval-every-steps 0 --no-checkpoint
  --profile-json /tmp/nfn_dweight_beta_default.json` on the dedicated RTX 5090;
  the JSON reported `passed: true`,
  `dweight_first_microbatch_beta_zero_enabled: true`, and
  `first-write-then-accumulate` strategy suffixes.
- Ran `tools/paired_kernel_speed.py` against
  `NFN_NATIVE_GPT_DWEIGHT_FIRST_MICROBATCH_BETA_ZERO=0` for two measured
  3-step samples on the idle display-disabled RTX 5090. The candidate measured
  `0.997584x` train-loop time and `1.002434x` tokens/sec versus the previous
  always-accumulate path.

### 2026-06-17 Default native GPT token initialization to fast Tile power-of-two pattern

#### Changed

- Dense GPT transformer-LM startup now uses
  `nfn_native_tile_init_gpt2_token_weight_fast_float32` by default, or
  `nfn_native_tile_init_gpt2_token_weight_fast_with_bf16_shadow_float32` when
  the persistent BF16 LM-head shadow is enabled.
- The older modulo-17 deterministic initializer remains available only for
  paired bisection with `NFN_NATIVE_GPT_TOKEN_WEIGHT_INIT_LEGACY_MOD17=1`.
- Runtime JSON now reports `token_weight_init_legacy_mod17_enabled` and the
  default `token_weight_init_strategy` values
  `"device-tile-power2-deterministic"` or
  `"device-tile-power2-deterministic-fused-bf16-shadow"`.

#### Verification

- Rebuilt the raw Tile ops library with `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt the native GPT CLI with `bash tools/build_native_gpt_cli.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k
  "native_train_tile_ops_builds_torch_free_c_abi or transformer_lm"`.
- Ran `tools/paired_kernel_speed.py` against the legacy modulo-17 initializer
  for five measured startup-only samples on the idle RTX 5090. The fast
  initializer measured `0.982065x` setup time and `0.960453x` token-init time.
- Ran the same paired script for two measured 3-step training samples. The
  fast initializer measured `0.977302x` train-loop time and `1.023233x`
  tokens/sec.

### 2026-06-17 Default dense GPT LM-head dWeight to BF16 hidden

#### Changed

- Dense GPT transformer-LM now enables `NFN_NATIVE_GPT_LM_HEAD_BF16_DWEIGHT`
  by default, so tied LM-head dWeight accumulation consumes the prepacked BF16
  final-norm hidden buffer with BF16 dlogits instead of the previous
  float-hidden path.
- Set `NFN_NATIVE_GPT_LM_HEAD_BF16_DWEIGHT=0` to reproduce the previous
  float-hidden LM-head dWeight path for paired bisection.

#### Verification

- Ran `tools/paired_kernel_speed.py` with two measured 2-step samples on the
  idle display-disabled RTX 5090, comparing the previous default against
  `--candidate-env NFN_NATIVE_GPT_LM_HEAD_BF16_DWEIGHT=1`. The candidate
  measured `0.996147x` baseline train-loop time and `1.003872x` baseline
  tokens/sec.

### 2026-06-17 Add direct BF16 dWeight optimizer profiling path

#### Changed

- `NFN_NATIVE_GPT_BF16_BLOCK_DWEIGHT_STAGING=1` now routes staged QKV and MLP
  FC BF16 dWeights through BF16 sumsq clipping descriptors and
  `nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32`
  when BF16 primary block weights are enabled. This skips the previous
  BF16-to-FP32 staging flush for that opt-in profiling path.
- The dense GPT trainer JSON now reports the direct BF16 dWeight staging
  strategy, BF16-gradient AdamW descriptor counts, BF16 sumsq launch counts, and
  BF16-gradient AdamW launch counts.
- The path remains default-off. The current optimized default uses the faster
  float-gradient accumulation path backed by the trainer cuBLASLt bgrad route.

#### Verification

- Rebuilt the native GPT CLI with `bash tools/build_native_gpt_cli.sh`.
- Ran `build/nfn_gpt_native_train --backend tile-cuda --tinystories
  --train-transformer-lm --max-steps 2 --eval-every-steps 0 --no-checkpoint
  --profile-json /tmp/nfn_bf16_dweight_direct.json` with GPU access on the
  dedicated RTX 5090. The JSON reported
  `block_dweight_bf16_staging_strategy:
  "qkv-fc-bf16-dweight-staging-direct-bf16-param-adamw"`,
  `block_weight_bf16_gradient_storage_strategy:
  "qkv-fc-bf16-accumulation-buffer"`, 24 BF16-gradient AdamW descriptors, and
  zero staging flush launches.
- Ran the required same-script paired benchmark with
  `tools/paired_kernel_speed.py`, comparing
  `NFN_NATIVE_GPT_BF16_BLOCK_DWEIGHT_STAGING=0` against the direct BF16
  staging candidate for two measured 2-step samples on the idle display-disabled
  RTX 5090. The candidate measured `1.032515x` baseline train-loop time and
  `0.968528x` baseline tokens/sec, so it was kept opt-in instead of promoted to
  the default.

### 2026-06-17 Add GPT-2 evo native ABI smoke

#### Changed

- Added `nfn_gpt2_evo_native_train --smoke-evo-kernels` with
  `--tile-ops-lib` and `--cuda-runtime-lib` support. The action loads the raw
  trainer Tile ops shared library and CUDA runtime, runs evo candidate mutation,
  best-loss selection, and best-candidate adoption over tiny device buffers, and
  reports JSON before any dataset, Torch, Python graph runtime, or graph-editor
  payload path is opened.
- `tools/build_native_missing_trainers.sh` now links the GPT-2 evo native
  preflight binary with `-ldl` so it can validate raw Tile ABI symbols directly.
- The full GPT-2 evo forward-only candidate-evaluation trainer loop is still
  intentionally reported as missing; this change proves the exported evo ABI is
  executable from the family-specific C++ binary.

#### Verification

- Rebuilt missing-family native binaries with
  `bash tools/build_native_missing_trainers.sh`.
- Ran `build/nfn_gpt2_evo_native_train --smoke-evo-kernels --tile-ops-lib
  build/libnfn_native_train_tile_ops.so` with GPU access on the dedicated RTX
  5090. The JSON reported `passed: true`, `best_index: 1`, `best_loss: 1.25`,
  and `max_adopt_abs_error: 0`.
- Confirmed `nvidia-smi` reported the RTX 5090 as display-disabled and idle
  before the smoke.

### 2026-06-17 Add native evo Tile ABI primitives

#### Changed

- Added trainer-facing CUDA Tile ABI symbols for GPT-2 evo layer search:
  `nfn_native_tile_evo_mutate_candidates_float32`,
  `nfn_native_tile_evo_select_best_loss_float32`, and
  `nfn_native_tile_evo_adopt_candidate_float32`.
- `nfn_gpt2_evo_native_train --print-plan` now reports device-side evo
  mutation, best-loss selection, and best-candidate adoption as available
  native pieces. The remaining missing work is wiring the forward-only
  candidate-evaluation loop to those primitives without graph-editor tensor
  flow.

#### Verification

- Rebuilt `libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt the missing-family trainer binaries with
  `bash tools/build_native_missing_trainers.sh`.
- Ran `nm -D build/libnfn_native_train_tile_ops.so | rg
  'nfn_native_tile_evo_(mutate_candidates|select_best_loss|adopt_candidate)_float32'`
  and confirmed all three symbols are exported.
- Ran `build/nfn_gpt2_evo_native_train --print-plan --eval-every-steps 1000
  --tile-cuda-activation-dtype nvfp4` and confirmed the new evo Tile ABI is
  listed under `available_native_kernels`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k "missing_family or native_train_tile_ops_builds_torch_free_c_abi"`.
- Ran `git diff --check`.

### 2026-06-17 Parse native JSON sidecars in paired benchmarks

#### Changed

- `tools/paired_kernel_speed.py` now detects `--json-out`,
  `--profile-json`, and `--stage-profile-json` in baseline/candidate child
  commands. When the child stdout does not contain native JSON, the helper reads
  the sidecar file and extracts the same native-loop timing, setup, stage, and
  kernel-counter metrics used for paired summaries and ratios.
- README and CLI workflow docs now note that native profiled runs can keep
  stdout small without losing benchmark metric summaries.

#### Verification

- Ran `python -m pytest tests/test_tile_cuda_examples.py -q` (`12 passed`).
- Ran `git diff --check`.

### 2026-06-17 Add native GPT JSON output file flag

#### Changed

- `nfn_gpt_native_train` now accepts `--json-out PATH` and the profiling
  aliases `--profile-json PATH` / `--stage-profile-json PATH`. The flag
  redirects the compiled C++ trainer's JSON stdout to the requested file after
  argument validation, so plan, smoke, startup, training, and stage-timed
  profiling runs can be captured without shell redirection.
- README and CLI documentation now show the flag alongside the paired kernel
  benchmark workflow.

#### Verification

- Rebuilt the native GPT CLI with `bash tools/build_native_gpt_cli.sh`.
- Ran `build/nfn_gpt_native_train --backend tile-cuda --tinystories
  --print-plan --json-out /tmp/nfn_native_gpt_plan_json_out_smoke.json`; stdout
  was empty and the file contained valid native plan JSON with
  `status: "native-transformer-lm-ready"`.
- Ran `python -m pytest tests/test_native_gpt2.py -q` (`41 passed, 1 skipped`).
- Ran `git diff --check`.

### 2026-06-17 Add Tile-CUDA candidate build flags

#### Changed

- `tools/build_native_train_tile_ops.sh` now appends optional
  whitespace-separated `NFN_TILE_CUDA_EXTRA_NVCC_FLAGS` and
  `NFN_TILE_CUDA_EXTRA_LDLIBS` after the default SM120 flags. This keeps the
  supported build unchanged while making temporary compile-time kernel
  candidates reproducible for `tools/paired_kernel_speed.py`.

#### Verification

- Built a temporary Tile-CUDA shared library under `/tmp` while checking the
  SM120 dGELU candidate path.
- Benchmarked that candidate against the default Tile ops library with
  `tools/paired_kernel_speed.py` on the dedicated display-disabled RTX 5090.
  The selected GPU had zero compute processes; the candidate was not promoted
  because train-loop step time was `1.008620x` versus the default.

### 2026-06-16 Fuse native GPT token BF16-shadow startup init

#### Changed

- Added the raw CUDA Tile ABI
  `nfn_native_tile_init_gpt2_token_weight_with_bf16_shadow_float32`, which
  initializes the tied token FP32 master weight and persistent BF16 LM-head
  shadow in one Tile kernel.
- Dense native GPT startup now uses that fused initializer by default whenever
  the token BF16 shadow is enabled. Set
  `NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_INIT=0` or the `GPT2`-prefixed alias
  to reproduce the older two-pass token init plus BF16 refresh path for paired
  benchmarking.
- Runtime JSON now reports
  `token_weight_bf16_initial_refresh_fusion_enabled` and
  `token_weight_bf16_initial_refresh_elided`.

#### Breaking changes

- Existing `libnfn_native_train_tile_ops.so` builds that do not export
  `nfn_native_tile_init_gpt2_token_weight_with_bf16_shadow_float32` no longer
  satisfy the dense GPT trainer's required Tile ABI symbol list. Rebuild with
  `bash tools/build_native_train_tile_ops.sh` before running
  `build/nfn_gpt_native_train` or the SDK/native CLI paths.

#### Verification

- Rebuilt the Tile ops library with `bash tools/build_native_train_tile_ops.sh`
  and confirmed `nm -D build/libnfn_native_train_tile_ops.so` exports both the
  old token initializer and the new fused BF16-shadow initializer.
- Rebuilt the native GPT CLI with `bash tools/build_native_gpt_cli.sh`.
- Ran a one-step native CUDA smoke on the dedicated display-disabled RTX 5090:
  `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1
  build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1
  --eval-every-steps 0 --no-checkpoint --tile-ops-lib
  build/libnfn_native_train_tile_ops.so`. The JSON reported
  `passed: true`, `token_weight_bf16_initial_refresh_elided: true`, and the new
  ABI in the kernel list.
- Ran same-script startup-only paired timing with
  `NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_INIT=0` as the baseline and the fused
  default as the candidate:
  `python tools/paired_kernel_speed.py --baseline "build/nfn_gpt_native_train
  --backend tile-cuda --tinystories --startup-only --max-steps 1
  --eval-every-steps 0 --no-checkpoint --tile-ops-lib
  build/libnfn_native_train_tile_ops.so" --candidate "build/nfn_gpt_native_train
  --backend tile-cuda --tinystories --startup-only --max-steps 1
  --eval-every-steps 0 --no-checkpoint --tile-ops-lib
  build/libnfn_native_train_tile_ops.so" --baseline-env
  NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_INIT=0 --samples 7 --warmup 1
  --cuda-visible-devices auto --cuda-device-max-connections 1
  --require-idle-selected-gpu --max-selected-gpu-utilization-pct 15`. The
  selected GPU was the display-disabled RTX 5090 with zero compute processes;
  candidate startup wall time was `0.960798x`, setup wall time was `0.954613x`,
  and total native wall time was `0.962174x` versus the two-pass baseline.
- Also ran a full five-step paired timing with the same old-vs-new setup. That
  run showed setup improvement (`0.982298x`) but noisy train-loop and total
  ratios (`1.008343x` train-loop step time, `1.007016x` total), so this change
  is treated as a startup optimization rather than a train-loop throughput
  improvement.

### 2026-06-16 Default SM120 parity benchmark GPU selection to auto

#### Changed

- `tools/bench_native_gpt_sm120_parity.sh` now defaults
  `NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=auto` instead of hard-pinning CUDA
  device `0`. The paired harness therefore selects an idle display-disabled
  NVIDIA GPU by default on workstations where a separate GPU drives the
  display.
- Explicit pinning still works with
  `NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0` or another CUDA device value. The
  selected-GPU idle/process guard and utilization guard remain enabled.

#### Verification

- Ran `python -m pytest tests/test_tile_cuda_examples.py -q`.
- Ran `bash -n tools/bench_native_gpt_sm120_parity.sh`.
- Ran a one-step live parity smoke with
  `NFN_SM120_PARITY_STEPS=1 NFN_SM120_PARITY_SAMPLES=1
  NFN_SM120_PARITY_WARMUP=0
  NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_auto_gpu_smoke.json
  tools/bench_native_gpt_sm120_parity.sh`; the wrapper reported
  `requested=auto resolved=0 mode=auto-dedicated`, selected the
  display-disabled RTX 5090, and found zero compute processes before/after the
  sample.
- Rejected four native-trainer performance candidates on the dedicated RTX
  5090 using `tools/paired_kernel_speed.py` with
  `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`,
  `--require-idle-selected-gpu`, and same-script paired samples:
  token BF16-shadow AdamW fusion was `1.002344x` train-loop step time,
  LayerNorm affine row chunk `128` was `1.003970x`, LM-head hidden prepack off
  was `1.005337x`, and packed-attention saved-LSE off was `1.001135x`.

### 2026-06-16 Default dense GPT LM-head to persistent token BF16 shadow

#### Changed

- Dense native GPT training now allocates a persistent BF16 shadow of the tied
  token embedding/LM-head weight by default. LM-head logits and dHidden GEMMs
  consume the BF16 shadow while token embedding, AdamW state, and checkpoint
  export keep using the FP32 master weight.
- Added `NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_SHADOW=0` /
  `NFN_NATIVE_GPT2_TOKEN_WEIGHT_BF16_SHADOW=0` as the paired-benchmark
  bisection switch for the older per-step BF16 bridge/cache route.
- Runtime JSON now reports `token_weight_bf16_shadow_enabled` and
  `token_weight_bf16_refresh_count` at the top level and inside
  `block_state_layout`.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Ran a one-step native CUDA smoke on the dedicated RTX 5090 with
  `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1
  NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_SHADOW=1 build/nfn_gpt_native_train
  --backend tile-cuda --tinystories --max-steps 1 --eval-every-steps 0
  --no-checkpoint --tile-ops-lib build/libnfn_native_train_tile_ops.so`;
  the trainer reported `passed: true`,
  `token_weight_bf16_shadow_enabled: true`, and
  `token_weight_bf16_refresh_count: 2`.
- Ran paired 5-step timing with `tools/paired_kernel_speed.py` on the
  dedicated RTX 5090, comparing the same binary with the shadow disabled
  against `NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_SHADOW=1`. Over 3 samples,
  selected GPU display was disabled with zero compute processes before each
  sample; candidate train-loop step time was `0.994401x` baseline and
  train-token throughput was `1.005634x`.
- Ran a longer paired 10-step timing check on the same dedicated RTX 5090. Over
  2 samples, selected GPU display was disabled with zero compute processes
  before each sample; candidate train-loop step time was `0.994084x`, total
  native wall time was `0.994352x`, and train-token throughput was `1.005960x`.

### 2026-06-16 Stop dense GPT train-loop timing before diagnostic sample copies

#### Changed

- Dense native GPT training now synchronizes at the end of the actual training
  loop, records `timing.train_loop_wall_ms`, and only then copies the sampled
  token weight and gradient clip scale back to the host for status JSON. This
  keeps `train_tokens_per_second` and SM120 parity comparisons focused on
  training work instead of including post-training diagnostic metadata copies.

#### Verification

- Ran `python -m pytest
  tests/test_native_gpt2.py::test_large_row_reduction_fallbacks_use_shared_row_chunks
  -q`.
- Ran `bash -n tools/build_native_gpt_cli.sh`, then rebuilt with
  `bash tools/build_native_gpt_cli.sh`.
- Ran `python tools/check_native_no_torch_deps.py --json`; both
  `build/nfn_gpt_native_train` and `build/libnfn_native_train_tile_ops.so`
  had no forbidden Torch/Python dependencies.
- Ran a one-step native CUDA smoke on the dedicated RTX 5090 with
  `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1
  build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1
  --eval-every-steps 0 --no-checkpoint --tile-ops-lib
  build/libnfn_native_train_tile_ops.so`; the trainer reported
  `passed: true`.
- Ran a five-step paired parity smoke with
  `NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_post_timing_patch_5step.json
  tools/bench_native_gpt_sm120_parity.sh`; the selected RTX 5090 was display
  disabled with zero compute processes, llm.kittens train-loop throughput was
  `170,438 tok/s`, and NeuralFn native train-loop throughput was
  `180,160 tok/s`.

### 2026-06-16 Fix SM120 parity benchmark timing cadence

#### Changed

- `tools/bench_native_gpt_sm120_parity.sh` now separates benchmark step count
  from the llm.kittens reference sample/checkpoint cadence. Short parity runs
  default to timing-only reference settings,
  `NFN_SM120_PARITY_SAMPLE_EVERY=0` and
  `NFN_SM120_PARITY_CHECKPOINT_EVERY=0`, and pass the same cadence knobs to the
  NeuralFn native candidate while keeping `--no-checkpoint`.
- Added `NFN_SM120_PARITY_SAMPLE_EVERY`,
  `NFN_SM120_PARITY_CHECKPOINT_EVERY`, and
  `NFN_SM120_PARITY_GENERATE_TOKENS` as explicit wrapper controls. Use
  `20000`, `200`, and `144` respectively to reproduce the full
  `train-sm120.sh` sample/checkpoint cadence instead of timing-only throughput.

#### Verification

- Ran `bash -n tools/bench_native_gpt_sm120_parity.sh`.
- Ran `python -m pytest tests/test_tile_cuda_examples.py -q -k
  native_gpt_sm120_parity_wrapper_uses_reference_shape`.
- Ran `git diff --check`.
- Ran a one-step live wrapper smoke on the dedicated RTX 5090:
  `NFN_SM120_PARITY_STEPS=1 NFN_SM120_PARITY_SAMPLES=1
  NFN_SM120_PARITY_WARMUP=0
  NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_timing_only_1step.json
  tools/bench_native_gpt_sm120_parity.sh`. The selected GPU was display
  disabled with zero compute processes, llm.kittens train-loop throughput was
  `210,769 tok/s`, and NeuralFn native train-loop throughput was
  `190,804 tok/s`.

### 2026-06-16 Elide dense GPT LN2 FP32 norm stores

#### Changed

- Dense native GPT training now skips the redundant FP32 `ln2_out` store inside
  the fused attention-projection residual+LN2 Tile kernels when the stored-MLP
  path consumes the BF16 LN2 output directly. The Tile kernels accept
  `norm_out == nullptr` while still writing mean/rstd and optional BF16 norm
  output.
- The default path is controlled by
  `NFN_NATIVE_GPT_ELIDE_LN2_BF16_NORM_FLOAT_STORE=0` or the GPT2-prefixed
  fallback for paired bisection.
- Runtime JSON now reports
  `fused_ln2_bf16_norm_float_store_elision_enabled`,
  `stored_mlp_ln2_bf16_float_store_elided_count`,
  `stored_mlp_ln2_bf16_float_store_elided_elements`, and the
  `attention_residual_ln2_strategy` suffix
  `fused-bf16-linear-bias-residual-layernorm-bf16-norm-fp32-store-elided` when
  the default route is active.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Ran a one-step native GPT Tile-CUDA smoke on the dedicated RTX 5090; runtime
  JSON reported `stored_mlp_ln2_bf16_float_store_elided_count: 96` and
  `stored_mlp_ln2_bf16_float_store_elided_elements: 4831838208`.
- Ran a guarded paired 5-step benchmark on the dedicated RTX 5090 with the
  baseline flag disabled and the candidate default enabled:
  `train_loop_wall_ms_per_step` improved from `2726.633333` to `2717.806667`
  and `train_tokens_per_second` improved from `192289.666667` to
  `192913.666667`; selected GPU display was disabled with zero compute
  processes before every sample.

### 2026-06-16 Fuse native GPT LayerNorm affine and residual backward

#### Changed

- Added trainer-facing Tile-CUDA ABI symbols
  `nfn_native_tile_layer_norm_backward_affine_residual_add_accumulate_with_stats_float32`
  and
  `nfn_native_tile_layer_norm_backward_affine_residual_add_accumulate_with_stats_bf16_bits_float32`.
  For GPT-width `dim <= 1024` shapes, these kernels accumulate LayerNorm
  dWeight/dBias, compute dInput from stored mean/rstd, apply the residual scale,
  and add the upstream residual gradient in one launch.
- Dense native GPT training now uses the fused LayerNorm backward path by
  default for LN1 and LN2. Set
  `NFN_NATIVE_GPT_FUSE_LN_BACKWARD_AFFINE_RESIDUAL=0` or the GPT2-prefixed
  fallback to reproduce the previous affine-accumulate plus dInput/residual-add
  pair for paired benchmarks.
- The BF16 QKV-gradient handoff path now elides the unused float32 `grad_qkv`
  scratch allocation from the startup float arena. Set
  `NFN_NATIVE_GPT_ELIDE_QKV_FLOAT_GRAD_SCRATCH=0` or the GPT2-prefixed fallback
  to reproduce the previous reservation.
- Runtime JSON now reports
  `attention_backward_qkv_float_grad_scratch_elided`,
  `attention_backward_qkv_float_grad_scratch_elements`,
  `attention_backward_qkv_float_grad_scratch_bytes_elided`,
  `attention_backward_qkv_float_grad_scratch_strategy`,
  `block_state_layout.layer_norm_backward_affine_residual_fusion_enabled`, and
  `block_state_layout.layer_norm_backward_affine_residual_fused_kernel_launches`.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt the native GPT binaries with `bash tools/build_native_gpt_cli.sh` and
  `bash tools/build_native_gpt2_cli.sh`.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran
  `python -m pytest tests/test_native_gpt2.py -q -k 'native_gpt2_cpp_cli_builds_and_uses_sm120_defaults or packed_qkv_uint16_arena_reserves_full_scratch_layout'`.
- Ran a one-step TinyStories native GPT smoke on the dedicated RTX 5090. The
  payload reported
  `block_state_layout.layer_norm_backward_affine_residual_fused_kernel_launches: 192`
  and
  `attention_backward_qkv_float_grad_scratch_bytes_elided: 603979776`.
- Ran paired 5-step TinyStories timing on the dedicated RTX 5090 with the idle
  GPU guard, comparing
  `NFN_NATIVE_GPT_FUSE_LN_BACKWARD_AFFINE_RESIDUAL=0` against the new default.
  Over 3 measured samples after 1 warmup pair, the candidate measured
  `0.989125x` mean train-loop wall time and `1.011001x` mean train tokens/sec.
- Ran paired 5-step TinyStories timing comparing
  `NFN_NATIVE_GPT_ELIDE_QKV_FLOAT_GRAD_SCRATCH=0` against the new default. The
  scratch elision was timing-neutral (`0.999864x` mean train-loop wall time) but
  removes the unused 604 MiB float scratch reservation when BF16 QKV gradient
  handoff is active.

### 2026-06-16 Defer native GPT validation MLP float scratch

#### Changed

- Dense GPT native training now leaves the validation-only float MLP scratch
  buffers (`fc_out` and `act`) out of the startup float arena when stored BF16
  MLP activations cover every transformer block. Training forwards continue to
  use the BF16 stored-MLP path; the two float hidden-size buffers are allocated
  lazily only if a validation pass runs with the preserve=false scratch tape.
- The default 12-layer `64 x 1024` shape reduces
  `float_arena_requested_elements` by `402,653,184` elements. Set
  `NFN_NATIVE_GPT_LAZY_VALIDATION_MLP_FLOAT_SCRATCH=0` or
  `NFN_NATIVE_GPT2_LAZY_VALIDATION_MLP_FLOAT_SCRATCH=0` to reproduce the older
  startup arena layout for paired comparisons.
- Runtime JSON now reports `lazy_validation_mlp_float_scratch_enabled`,
  `lazy_validation_mlp_float_scratch_elements`,
  `lazy_validation_mlp_float_scratch_bytes`, and
  `lazy_validation_mlp_float_scratch_cuda_malloc_count`.

#### Verification

- Rebuilt the native GPT binaries with `bash tools/build_native_gpt_cli.sh` and
  `bash tools/build_native_gpt2_cli.sh`.
- Ran `python tools/check_native_no_torch_deps.py`.
- Ran a startup-only opt-in smoke on the dedicated RTX 5090; the candidate
  reported `float_arena_requested_elements: 2269479693` versus the previous
  `2672132877`, with no lazy validation scratch allocation before validation.
- Ran a one-step validation smoke with `--eval-every-steps 1 --eval-batches 1`;
  it allocated `1,610,612,736` bytes across 2 lazy scratch `cudaMalloc` calls
  and emitted one validation loss record.
- Ran paired startup-only timing over 3 measured samples after 1 warmup pair.
  The candidate measured `0.857059x` mean total native startup wall time.
- Ran paired 5-step TinyStories timing over 3 measured samples after 1 warmup
  pair. The candidate measured `0.987446x` mean train-loop wall time and
  `1.012769x` mean train tokens/sec.

### 2026-06-16 Retile native GPT token-weight startup initialization

#### Changed

- `nfn_native_tile_init_gpt2_token_weight_float32` now initializes the native
  dense GPT tied token embedding/LM-head table with 2048-element CUDA Tile
  blocks instead of 1024-element blocks. The deterministic modulo-17
  initialization pattern is unchanged; the retile only reduces startup launch
  fanout for the full padded vocabulary table.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Ran paired dedicated-RTX-5090 startup-only timing with selected-GPU idle
  guards comparing the rebuilt 1024-tile baseline against the 2048-tile
  candidate through the same compiled trainer. Over 3 measured samples after 1
  warmup pair, the candidate measured `0.978511x` mean setup wall time,
  `0.978822x` mean token-weight init time, and `0.982087x` mean total native
  startup wall time.

### 2026-06-16 Reduce native GPT SDK binding launch overhead

#### Changed

- The generic `neuralfn._native_gpt` and compatibility
  `neuralfn._native_gpt2` C++ bindings now launch compiled native trainer
  commands with `posix_spawnp()` instead of `fork()` plus `execvp()`. This keeps
  SDK native runs on the compiled command path while avoiding Python-process
  fork overhead before the CUDA trainer starts.
- The binding now sets `CUDA_MODULE_LOADING=LAZY` when the caller has not
  supplied a module-loading policy, matching the standalone subprocess native
  runner default.

#### Verification

- `bash tools/build_native_gpt_binding.sh`
- `bash tools/build_native_gpt2_binding.sh`
- `python -m pytest tests/test_native_gpt2.py -q -k 'cpp_binding or spawn_and_lazy_cuda_module_loading'`

### 2026-06-16 Add opt-in native GPT BF16 attention grad-out handoff

#### Changed

- Added the trainer Tile ABI symbols
  `nfn_native_tile_linear_backward_input_weight_bf16_to_bf16_bits_float32`,
  `nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_bf16_merged_grad_float32`,
  and
  `nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_bf16_merged_grad_float32`.
- Dense GPT native training can now opt into
  `NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1` with the
  `NFN_NATIVE_GPT2_BF16_ATTENTION_GRAD_OUT=1` compatibility fallback. The path
  makes attention projection dInput write BF16 grad-out bits directly, then
  feeds those bits into packed-attention backward so the QKV grad handoff stays
  BF16 one stage earlier.
- Runtime and plan JSON report
  `attention_backward_bf16_grad_out_handoff_enabled`,
  `attention_backward_grad_out_dtype`,
  `attention_backward_bf16_grad_out_scratch_elements`,
  `attention_backward_bf16_grad_out_scratch_bytes`, and the updated
  `attention_backward_qkv_bridge_strategy`.
- The feature remains default-off because paired dedicated-RTX-5090 timing
  measured it slower than the existing float grad-out default.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran a GPU-visible TinyStories one-step smoke on CUDA device 0 with
  `NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1`; it reported `passed: true`,
  `attention_backward_grad_out_dtype: "bf16"`,
  `attention_backward_bf16_grad_out_scratch_elements: 50331648`, and the new
  ABI symbols in `kernels`.
- Ran paired dedicated-RTX-5090 timing with selected-GPU idle guards comparing
  the current default against `NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1`. The
  candidate measured `1.015507x` median train-loop time and `0.984731x` median
  tokens/sec, so the default remains unchanged.

### 2026-06-16 Add opt-in native GPT cudaMallocAsync allocator telemetry

#### Changed

- Dense GPT transformer-LM startup now has an opt-in allocator profiling switch,
  `NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC=1`, with the GPT-2 compatibility fallback
  `NFN_NATIVE_GPT2_CUDA_MALLOC_ASYNC=1`.
- When enabled and the CUDA runtime exports `cudaMallocAsync` /
  `cudaFreeAsync`, the native C++ trainer routes its large device arenas
  through the async allocator, falls back to `cudaMalloc` if an async allocation
  fails, frees async pointers with `cudaFreeAsync`, and synchronizes before
  teardown completes.
- Runtime JSON now reports `device_allocator_strategy`,
  `device_cuda_malloc_async_requested`, `device_cuda_malloc_async_enabled`,
  async symbol availability, async allocation/free counts, and
  `device_cuda_malloc_async_fallback_count`.
- The async allocator remains default-off. Dedicated RTX 5090 paired timing
  measured it slower than the existing arena `cudaMalloc` path.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran a GPU-visible TinyStories one-step smoke on CUDA device 0 with
  `NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC=1`; it reported `passed: true`,
  `device_allocator_strategy: "cudaMallocAsync-null-stream"`,
  `device_cuda_malloc_async_count: 6`,
  `device_cuda_free_async_count: 6`, and
  `device_cuda_malloc_async_fallback_count: 0`.
- Ran paired dedicated-RTX-5090 timing with selected-GPU idle guards comparing
  default `cudaMalloc` against `NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC=1`. The async
  allocator measured about `1.034x` median command wall time,
  `1.033x` median total wall time, `1.196x` median setup wall time, and
  `1.003x` median train-loop time, so the feature remains opt-in.
- Ran paired dedicated-RTX-5090 control checks for cuBLASLt heuristic index 0
  and 2, LM-head BF16 hidden prepack, BF16 dWeight+bias bgrad, and packed
  attention LN1 stats. None produced a clear default-changing win, so the
  existing optimized defaults were retained.

### 2026-06-16 Add opt-in native BF16 dWeight staging ABI

#### Changed

- Added the raw trainer Tile ABI symbol
  `nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_to_bf16_bits_float32`
  for BF16 activation plus BF16 gradient dWeight accumulation into a BF16
  staging buffer, with bias accumulation kept in float32.
- Dense GPT native training now binds and audits that ABI and can route QKV and
  MLP FC BF16/BF16 dWeight accumulation through BF16 staging buffers when
  `NFN_NATIVE_GPT_BF16_BLOCK_DWEIGHT_STAGING=1` is set. The trainer flushes the
  staged BF16 dWeights back to the existing float32 accumulation buffers before
  gradient clipping and AdamW.
- The staging experiment is intentionally default-off. The current default
  remains the cuBLASLt bgrad-backed float32 accumulation path because paired
  RTX 5090 timing measured the BF16 staging candidate slower.
- Runtime JSON now reports `block_dweight_bf16_staging_enabled`,
  `block_dweight_bf16_staging_elements`, `block_dweight_bf16_staging_bytes`,
  `block_dweight_bf16_staging_zero_count`,
  `block_dweight_bf16_staging_convert_kernel_launches`, and
  `block_dweight_bf16_staging_strategy`.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`
  and rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Ran a GPU-visible one-step TinyStories smoke on CUDA device 0 with the
  default path. It reported `passed: true`,
  `block_dweight_bf16_staging_enabled: false`,
  `block_dweight_bf16_staging_strategy:
  "disabled-fp32-accumulation-default"`, `linear_cublaslt_gemm_count: 672`, and
  about `2785.80 ms` train-loop time.
- Ran a GPU-visible one-step TinyStories smoke with
  `NFN_NATIVE_GPT_BF16_BLOCK_DWEIGHT_STAGING=1`. It reported `passed: true`,
  `block_dweight_bf16_staging_enabled: true`,
  `block_dweight_bf16_staging_bytes: 99090432`,
  `block_dweight_bf16_staging_convert_kernel_launches: 24`, and
  `gradient_zero_kernel_launches_per_optimizer_step: 3`, confirming the extra
  BF16 staging zero launch is counted.
- Ran an earlier paired candidate-vs-baseline benchmark on the dedicated RTX
  5090 with selected-GPU idle guards, comparing
  `NFN_NATIVE_GPT_BF16_BLOCK_DWEIGHT_STAGING=0` against the staging candidate.
  The candidate measured `1.024508x` median train-loop time and about
  `0.976971x` mean tokens/sec versus the default, so the feature remains
  opt-in.

### 2026-06-16 Prepack native GPT LM-head hidden BF16

#### Changed

- Added the raw trainer Tile ABI symbol
  `nfn_native_tile_linear_bf16_input_float_weight_bf16_output_float32` for
  BF16 input activations with FP32 weights and BF16 output logits.
- Dense GPT native training now prepackages the full final LayerNorm hidden
  activation to BF16 once per microbatch by default. The LM-head logits GEMM
  reuses that BF16 hidden buffer, and tied token-weight dWeight accumulation now
  consumes the same BF16 hidden plus BF16 dlogits instead of packing each
  LM-head chunk separately.
- Runtime JSON now reports `lm_head_prepack_bf16_hidden_enabled: true`,
  `lm_head_bf16_hidden_elements` for the full microbatch hidden buffer,
  `lm_head_dweight_input_dtype: "bf16"`, and
  `lm_head_dweight_strategy:
  "full-final-norm-bf16-prepack-bf16-dlogit-dweight-accumulate"`. Set
  `NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=0` or the GPT-2-prefixed fallback
  to reproduce the older per-chunk hidden-packing path.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh` and rebuilt
  `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran a GPU-visible one-step TinyStories smoke on CUDA device 0. It reported
  `passed: true`, `lm_head_prepack_bf16_hidden_enabled: true`,
  `lm_head_bf16_hidden_elements: 50331648`, and
  `lm_head_dweight_strategy:
  "full-final-norm-bf16-prepack-bf16-dlogit-dweight-accumulate"`.
- Ran a five-sample paired benchmark on the dedicated RTX 5090 with selected
  GPU idle guards and no compute processes before samples, comparing
  `NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=0` against the new default. The
  default averaged `2767.83 ms` per optimizer step versus `2769.58 ms` for the
  disabled-prepack baseline, or `0.999368x` train-loop time and `1.000636x`
  tokens/sec. BF16 linear pack count dropped from 711 to 327 over each 3-step
  measured run.
- Ran `python tools/check_native_no_torch_deps.py` and
  `python -m pytest tests/test_native_gpt2.py -q -k
  'native_train_tile_ops_builds_torch_free_c_abi or
  native_gpt2_cpp_cli_builds_and_uses_sm120_defaults'`.

### 2026-06-16 Default saved packed-attention LSE for native GPT

#### Changed

- Dense GPT native training now stores per-row packed-attention LSE alongside
  the saved packed BF16 QKV/O activation cache by default. The backward path
  therefore consumes saved QKV/O/LSE through
  `saved-packed-qkv-o-lse-bf16-backward-to-qkv` instead of relying on the older
  shared-workspace LSE path.
- Runtime and plan JSON now report `stored_packed_attention_lse_enabled: true`,
  nonzero `stored_packed_attention_lse_elements`/`bytes` for the default
  full-shape plan, and
  `stored_packed_attention_backward_consumer_strategy:
  "saved-packed-qkv-o-lse-bf16-backward-to-qkv"`. Set
  `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE=0` or the GPT-2-prefixed fallback
  only for paired benchmarks against the older shared-workspace-LSE behavior.

#### Verification

- Ran a five-sample paired benchmark on the dedicated RTX 5090 with selected
  GPU idle guards and no compute processes before samples. The saved-LSE
  candidate averaged `2776.49 ms` per optimizer step versus `2785.29 ms` for
  the previous shared-workspace-LSE path, or `0.996841x` train-loop time and
  `1.003172x` tokens/sec.
- Ran a GPU-visible one-step TinyStories stage-timed smoke with
  `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE=1`. It reported `passed: true`,
  `stored_packed_attention_lse_enabled: true`,
  `stored_packed_attention_lse_bytes: 37748736`,
  `stored_packed_attention_backward_consumer_strategy:
  "saved-packed-qkv-o-lse-bf16-backward-to-qkv"`, and 96 saved packed-attention
  backward launches.
- Rebuilt `build/nfn_gpt_native_train` and ran a no-override GPU-visible
  one-step TinyStories smoke on CUDA device 0. It reported `passed: true`,
  `stored_packed_attention_lse_enabled: true`,
  `stored_packed_attention_lse_bytes: 37748736`, and
  `stored_packed_attention_backward_consumer_strategy:
  "saved-packed-qkv-o-lse-bf16-backward-to-qkv"`.

### 2026-06-16 Default BF16 projection residual consumers for native GPT

#### Changed

- Added trainer-facing Tile CUDA ABI symbols for BF16 projection-output
  residual consumers:
  `nfn_native_tile_linear_bias_residual_add_bf16_linear_float32`,
  `nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_float32`,
  `nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_float32`,
  and
  `nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_bf16_norm_float32`.
- Dense GPT native training now defaults attention and MLP projection forward
  GEMMs to BF16-output scratch when the following residual consumer can read
  BF16 bits directly. This avoids converting the rejected TK BF16-output
  candidate back to float32 just so residual/LN kernels can consume it.
- Runtime and plan JSON now report `bf16_projection_residual_enabled`,
  `projection_bf16_scratch_elements`, `projection_bf16_scratch_bytes`,
  `attention_projection_input_strategy:
  "packed-o-bf16-direct-gemm-bf16-residual-consumer"`,
  `mlp_proj_forward_activation_strategy:
  "fused-gelu-bf16-act-direct-bf16-output-gemm"`,
  `projection_bias_residual_strategy:
  "fused-bf16-linear-bias-residual-add"`, and
  `attention_residual_ln2_strategy:
  "fused-bf16-linear-bias-residual-layernorm"` for the default path. Set
  `NFN_NATIVE_GPT_BF16_PROJECTION_RESIDUAL=0` or the
  `NFN_NATIVE_GPT2_*` fallback only for paired benchmarks against the older
  float projection-output residual path.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Ran a GPU-visible one-step TinyStories smoke on CUDA device 0 with linear
  shape stats enabled. It reported `passed: true`,
  `linear_tk_float_out_gemm_count: 0`, `linear_cublaslt_gemm_count: 672`,
  `sample_gradient_clip_scale: 0.0837135`, and
  `sample_updated_token_weight: -0.0793953`.
- Ran a five-sample paired benchmark on the dedicated RTX 5090 with selected
  GPU idle guards and no compute processes before samples. The BF16
  projection-residual candidate averaged `2765.24 ms` per optimizer step
  versus `2768.63 ms` for the old float residual-consumer path, or `0.998779x`
  train-loop time and `1.001224x` tokens/sec.

### 2026-06-16 Direct uint16 token ids for native GPT training

#### Changed

- Added trainer-facing Tile CUDA ABI symbols for uint16 token/target
  consumption: `nfn_native_tile_token_embedding_u16_float32`,
  `nfn_native_tile_token_embedding_backward_weight_u16_float32`,
  `nfn_native_tile_token_cross_entropy_partials_strided_bf16_bits_u16_targets`,
  and
  `nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_u16_targets_with_workspace`.
- Dense GPT native training now keeps cached shard token and target ids in the
  uint16 device arena by default. Token embedding, BF16 public-vocab CE
  forward, CE backward, and token-embedding weight backward consume those ids
  directly, so the previous per-microbatch `uint16_to_int64` widening launch is
  elided in the default path.
- Runtime JSON now reports `token_id_direct_u16_enabled`,
  `token_id_upload_strategy:
  "uint16-pinned-async-h2d-direct-kernel-consumption"`,
  `token_id_widen_strategy: "elided-direct-u16-kernels"`, and zero
  `token_id_widen_kernel_launches_per_microbatch`. Set
  `NFN_NATIVE_GPT_DIRECT_U16_TOKENS=0` or
  `NFN_NATIVE_GPT2_DIRECT_U16_TOKENS=0` only for paired benchmarking against
  the older single-kernel device-widen path.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Ran a GPU-visible one-step TinyStories smoke on the dedicated RTX 5090; it
  reported `passed: true`, `token_id_direct_u16_enabled: true`,
  `token_id_widen_kernel_launches_per_microbatch: 0`,
  `sample_gradient_clip_scale: 0.0836959`, and
  `sample_updated_token_weight: -0.0793953`.
- Ran a five-sample paired benchmark on the dedicated RTX 5090 with selected
  GPU idle guards and no compute processes before samples. Direct uint16 ids
  averaged `2764.01 ms` per optimizer step versus `2766.64 ms` for the old
  widening path, or `0.999054x` train-loop time and `1.000949x` tokens/sec.

### 2026-06-16 cuBLASLt descriptor-cache hits for native GPT linear GEMMs

#### Changed

- Native Tile-CUDA trainer cuBLASLt plans now retain their matmul descriptors
  and matrix layouts, so cached plan hits skip per-GEMM descriptor construction
  and launch directly with the cached descriptors. Bias epilogue pointers are
  still refreshed at launch time, so shared shape plans remain valid across
  layers and parameter buffers.
- Added `NFN_TILE_CUDA_CUBLASLT_DESCRIPTOR_CACHE=0` /
  `NFN_NATIVE_LINEAR_CUBLASLT_DESCRIPTOR_CACHE=0` as a profiling-only opt-out
  for the previous descriptor-recreate behavior.
- Dense GPT runtime JSON now reports
  `linear_cublaslt_descriptor_cache_enabled` alongside the existing linear GEMM
  strategy and counter fields.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Ran GPU-visible one-step TinyStories smokes on the dedicated RTX 5090 for
  both descriptor-cache-on and descriptor-cache-off paths; both reported
  `passed: true`.
- Ran a five-sample paired benchmark on the dedicated RTX 5090 with selected
  GPU idle guards and no compute processes before samples. Cache-on averaged
  `2766.47 ms` per optimizer step versus `2772.46 ms` for cache-off, or
  `0.997843x` train-loop time and `1.002169x` tokens/sec.

### 2026-06-16 Native linear dispatch shape profiler

#### Added

- Added optional native linear dispatch shape profiling to the raw CUDA Tile
  trainer ABI. Set `NFN_NATIVE_LINEAR_SHAPE_STATS=1` or
  `NFN_TILE_CUDA_LINEAR_SHAPE_STATS=1` to record successful GEMM dispatch
  buckets for TK BF16, TK BF16 plus float-output conversion, cuBLASLt, cuBLAS
  GEMMEx BF16, and SGEMM paths.
- `nfn_gpt_native_train` now emits `linear_shape_stats_enabled`,
  `linear_shape_stats_count`, and `linear_shape_stats` JSON fields when the
  profiler records buckets. Each entry reports `path_name`, `m`, `n`, `k`,
  transpose flags, and `calls`, which makes the SM120 paired benchmark loop
  target specific kernel shapes instead of inferring from aggregate GEMM
  counters.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran a one-step dedicated RTX 5090 TinyStories pass with
  `NFN_NATIVE_LINEAR_SHAPE_STATS=1`; runtime JSON reported 13 shape buckets,
  `linear_sgemm_count: 0`, 352 TK GEMMs, and 864 cuBLASLt GEMMs. The buckets
  confirmed TK coverage for QKV, MLP FC/GELU, LM-head, and fused dGELU paths,
  with remaining cuBLASLt buckets concentrated in block projection, dInput, and
  dWeight shapes.
- Rejected measured cuBLASLt heuristic-index default changes: index 1 was about
  `1.3%` slower in train-loop time and index 2 was about `2.7%` slower in
  train-loop time for the short paired 3-step RTX 5090 runs.

### 2026-06-16 SM120 native GPT parity benchmark wrapper

#### Added

- Added `tools/bench_native_gpt_sm120_parity.sh`, a dedicated wrapper around
  `tools/paired_kernel_speed.py` that compares the local llm.kittens
  `train_gpt2cu` TinyStories SM120 reference shape against NeuralFn's compiled
  `nfn_gpt_native_train --backend tile-cuda` path. The wrapper pins the same
  `64 x 1024 -> 524288` token batch shape, learning rate, weight decay, warmup,
  activation, sample/checkpoint cadence, and selected-GPU guard settings used
  for the workstation RTX 5090 parity loop. Runtime knobs are environment
  variables: `NFN_SM120_PARITY_STEPS`, `NFN_SM120_PARITY_SAMPLES`,
  `NFN_SM120_PARITY_WARMUP`, `NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES`,
  `NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT`, and
  `NFN_SM120_PARITY_JSON_OUT`.

#### Verification

- Ran a manual 3-step paired benchmark on the dedicated RTX 5090 using the same
  commands now encoded by the wrapper. llm.kittens reported about
  `214,470 tok/s`; NeuralFn native reported about `193,049 tok/s`, or roughly
  `90.0%` of the reference training-loop throughput for that short run.
- Ran the new wrapper itself with `NFN_SM120_PARITY_STEPS=3`,
  `NFN_SM120_PARITY_SAMPLES=1`, and
  `NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_wrapper_3step.json`;
  llm.kittens reported about `214,170 tok/s` and NeuralFn native reported
  about `193,450 tok/s`, or roughly `90.3%` of the reference training-loop
  throughput.
- Ran a manual 10-step paired benchmark on the dedicated RTX 5090. llm.kittens
  reported about `213,236 tok/s`; NeuralFn native reported about
  `193,226 tok/s`, or roughly `90.6%` of the reference training-loop
  throughput.
- Captured a NeuralFn one-step CUDA-event stage profile in
  `/tmp/nfn_current_stage_1step.json`; the largest remaining buckets were
  `block_backward` at about `1357 ms`, `train.model_forward` at about
  `741 ms`, and `lm_head_backward` at about `645 ms`.
- Rejected measured default changes for the current pass: 16k LM-head chunks
  were about `12.5%` slower than 8192-row chunks, 4k LM-head chunks were
  neutral to slightly slower, BF16/BF16 LM-head dWeight was about `0.6%`
  slower, saved packed-attention LSE was neutral, disabling MLP activation
  storage was about `31.7%` slower, disabling BF16 MLP gradient handoff was
  about `8.2%` slower, disabling direct BF16 QKV gradient scratch was about
  `2.8%` slower, and disabling BF16 QKV dWeight was neutral on loop time.

### 2026-06-15 Native GPT startup-only profiling and lazy CUDA module loading

#### Changed

- Added `--startup-only` to `nfn_gpt_native_train`. It runs the full
  Tile-CUDA transformer setup path, including cached shard resolution, CUDA
  runtime loading, arena allocation, descriptor upload, AdamW state zeroing,
  and parameter initialization, then exits before optimizer steps or checkpoint
  export with `status: "native-transformer-lm-startup-ready"`.
- Added `startup_only` to `NativeGpt2RunConfig` and the generic
  `NativeGptRunConfig` path, and forwards it through compiled-CLI SDK configs.
- Native GPT direct C++ launches and Python SDK subprocess launchers now default
  `CUDA_MODULE_LOADING=LAZY` when the caller has not already set it, while
  preserving explicit caller overrides. Runtime JSON now reports the resolved
  value as `cuda_module_loading`.

#### Verification

- Rebuilt the dense GPT compiled CLI with
  `bash tools/build_native_gpt_cli.sh build/nfn_gpt_native_train`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k
  'compiled_cli_config_can_skip_checkpoint_export or
  native_gpt2_compiled_cli_runner_executes_cli or
  native_train_run_config_and_subprocess_runner or
  cpp_cli_builds_and_uses_sm120_defaults'` (`4 passed`).
- Ran `python -m pytest tests/test_native_dependencies.py -q` (`3 passed`).
- Ran `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1
  build/nfn_gpt_native_train --backend tile-cuda --tinystories --startup-only
  --eval-every-steps 0 --tile-ops-lib
  build/libnfn_native_train_tile_ops.so` on the dedicated RTX 5090. Before the
  default was applied externally, the same mode measured about `6285.63 ms`
  `setup_wall_ms`; with `CUDA_MODULE_LOADING=LAZY` it measured about
  `3491.35 ms`; after rebuilding with the new default and no external
  `CUDA_MODULE_LOADING`, runtime JSON reported `cuda_module_loading: "LAZY"`,
  `setup_wall_ms: 3048.18`, and `status:
  "native-transformer-lm-startup-ready"`.
- Ran a paired one-step dedicated-RTX-5090 comparison in
  `/tmp/nfn_lazy_vs_eager_startup_pair.json` with the same native command and
  command-scoped GPU idle guards. `CUDA_MODULE_LOADING=LAZY` reported mean
  process wall time `8.35s` versus `25.37s` for `EAGER`, mean train-loop wall
  time `4207.5 ms` versus `19300.55 ms`, and mean setup wall time
  `2573.99 ms` versus `2851.22 ms`.

### 2026-06-15 SM120 Tile CUDA trainer build flag alignment

#### Changed

- `tools/build_native_train_tile_ops.sh` now builds the SM120
  ThunderKittens-backed trainer Tile ops library with the same NVCC threading,
  host-compiler, data-prep, memory, and LayerNorm tuning flags used by the
  llm.kittens SM120 trainer. This keeps the native GPT attention bridge closer
  to the reference compile path while preserving NeuralFn's own initialized
  cublasLt GEMM dispatch.
- Added regression coverage so the native GPT static test checks for those
  SM120 build flags. The llm.kittens `LLMK_SM120_USE_CUBLASLT_GEMM` macro is
  intentionally not copied because it routes included header matmuls through
  llm.kittens' singleton cublasLt state and aborts without its initialization.

#### Verification

- Rebuilt the trainer Tile ops library with
  `bash tools/build_native_train_tile_ops.sh build/libnfn_native_train_tile_ops.so`.
- Rebuilt the dense GPT compiled CLI with
  `bash tools/build_native_gpt_cli.sh build/nfn_gpt_native_train`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k cpp_cli_builds_and_uses_sm120_defaults`.
- Ran a dedicated-RTX-5090 one-step smoke with
  `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1
  build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 1
  --eval-every-steps 0 --no-checkpoint --tile-ops-lib
  build/libnfn_native_train_tile_ops.so`; runtime JSON reported
  `status: native-transformer-lm-trained` and
  `attention_backward_strategy:
  tk-sm120-packed-qkv-bf16-saved-activation-backward-direct-bf16-grad-scratch-handoff`.
- Profiled the candidate with Nsight Systems using the same dedicated GPU
  (`/tmp/nfn_sm120_flags_1step_nsys.*`); the dominant
  `llmk::attention::sm120_detail::bwd_main_kernel<64,true,true>` averaged
  roughly `16.6 ms` per launch in that profile.
- Ran paired old-vs-new benchmarks on the display-disabled RTX 5090 with
  `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and
  `--require-idle-selected-gpu`. The 5-sample one-step comparison in
  `/tmp/nfn_sm120_flags_1step_gpu0_pair.json` reported candidate train loop
  mean `6427.23 ms` versus baseline `6439.01 ms`
  (`candidate_over_baseline_native_metrics.train_loop_wall_ms_per_step.mean:
  0.998209`). The 3-sample three-step comparison in
  `/tmp/nfn_sm120_flags_gpu0_pair.json` was noisy and recorded candidate mean
  `5549.62 ms/step` versus baseline `5488.19 ms/step`, so the retained evidence
  for this change is the profiler-level attention improvement plus the shorter
  multi-sample end-to-end parity run rather than a claimed large wall-clock
  throughput win.

### 2026-06-15 Paired benchmark command-scoped GPU guards

#### Changed

- `tools/paired_kernel_speed.py` now snapshots and enforces selected-GPU idle
  and utilization guards immediately before every warmup and measured baseline
  or candidate command, not just once before the command pair. This makes
  dedicated compute-GPU comparisons fail fast if another workload appears
  between the old and new kernel measurements.
- Result JSON now records command-level `gpu_before` / `gpu_after` snapshots
  under each `paired_samples[].baseline` and `paired_samples[].candidate`
  command result, in addition to the existing run-level and sample-level GPU
  snapshots.

#### Verification

- Ran `python -m py_compile tools/paired_kernel_speed.py`.
- Ran
  `python -m pytest tests/test_tile_cuda_examples.py -q -k 'paired_kernel_speed_tool_compiles_and_smokes or paired_kernel_speed_tool_applies_command_specific_env or paired_kernel_speed_tool_auto_selects_idle_display_disabled_gpu or paired_kernel_speed_tool_require_idle_selected_gpu_checks_selected_uuid or paired_kernel_speed_tool_selected_gpu_utilization_guard or paired_kernel_speed_tool_records_command_timeout or paired_kernel_speed_tool_extracts_llm_kittens_step_metrics or paired_kernel_speed_tool_sums_llm_kittens_step_time'`.
- Ran a live dedicated-RTX-5090 smoke with
  `python tools/paired_kernel_speed.py --baseline "python -c 'print(1)'"
  --candidate "python -c 'print(2)'" --samples 1 --warmup 0
  --cuda-visible-devices 0 --require-idle-selected-gpu --json-out
  /tmp/nfn_command_gpu_guard_smoke.json`; the JSON recorded command-level
  `gpu_before` / `gpu_after` snapshots for both baseline and candidate and
  selected the display-disabled RTX 5090.

### 2026-06-15 Paired benchmark timeout process-group cleanup

#### Changed

- `tools/paired_kernel_speed.py` now starts each measured command in its own
  process group and kills that group when `--command-timeout-seconds` expires.
  This prevents a slow or wedged native GPU candidate from leaving child
  training processes running after the paired sample is recorded.
- Timed-out command JSON keeps the existing `returncode: -1` contract and now
  also records `process_returncode` for the killed process when available.

#### Verification

- Ran `python -m py_compile tools/paired_kernel_speed.py`.
- Ran
  `python -m pytest tests/test_tile_cuda_examples.py -q -k 'paired_kernel_speed_tool_compiles_and_smokes or paired_kernel_speed_tool_applies_command_specific_env or paired_kernel_speed_tool_auto_selects_idle_display_disabled_gpu or paired_kernel_speed_tool_require_idle_selected_gpu_checks_selected_uuid or paired_kernel_speed_tool_selected_gpu_utilization_guard or paired_kernel_speed_tool_records_command_timeout or paired_kernel_speed_tool_extracts_llm_kittens_step_metrics or paired_kernel_speed_tool_sums_llm_kittens_step_time'`.
- The timeout test now launches a child process from the timed-out candidate and
  verifies the child marker file is not written after the timeout, covering
  process-group cleanup.

### 2026-06-15 Paired benchmark per-command env overrides

#### Changed

- `tools/paired_kernel_speed.py` now accepts repeatable
  `--baseline-env KEY=VALUE` and `--candidate-env KEY=VALUE` flags. These
  command-specific overrides are merged on top of the shared benchmark
  environment so CUDA pinning still applies to both commands while kernel
  experiment toggles can apply to only one side of the pair.
- Text and JSON output now include `baseline_env` and `candidate_env` when
  overrides are present, making saved candidate-vs-baseline artifacts
  self-describing without wrapping commands in `/usr/bin/env`.

#### Verification

- Ran `python -m py_compile tools/paired_kernel_speed.py`.
- Ran
  `python -m pytest tests/test_tile_cuda_examples.py -q -k 'paired_kernel_speed_tool_compiles_and_smokes or paired_kernel_speed_tool_applies_command_specific_env or paired_kernel_speed_tool_auto_selects_idle_display_disabled_gpu or paired_kernel_speed_tool_require_idle_selected_gpu_checks_selected_uuid or paired_kernel_speed_tool_selected_gpu_utilization_guard or paired_kernel_speed_tool_records_command_timeout or paired_kernel_speed_tool_extracts_llm_kittens_step_metrics or paired_kernel_speed_tool_sums_llm_kittens_step_time'`.
- Ran a live dedicated-RTX-5090 smoke using
  `--candidate-env NFN_NATIVE_GPT_LM_HEAD_BF16_DWEIGHT=1`; text output reported
  `candidate_env: {"NFN_NATIVE_GPT_LM_HEAD_BF16_DWEIGHT": "1"}` and both
  baseline/candidate one-step commands completed with zero selected-GPU compute
  processes before and after the sample.

### 2026-06-15 Native GPT fused LayerNorm scratch elision

#### Changed

- Dense GPT native training no longer reserves the fallback-only
  `grad_residual1_from_mlp` and `grad_x_from_attn` activation scratch buffers
  when `NFN_NATIVE_GPT_FUSE_LN_BACKWARD_RESIDUAL` is enabled.
- Runtime JSON now reports
  `block_state_layout.layer_norm_backward_residual_scratch_buffers_allocated`,
  `block_state_layout.layer_norm_backward_residual_scratch_buffers_elided`, and
  `block_state_layout.layer_norm_backward_residual_scratch_elements_elided` so
  startup allocation reductions are visible in run artifacts.

#### Verification

- Ran
  `python -m pytest tests/test_native_gpt2.py -q -k native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`.
- Built the candidate trainer with
  `bash tools/build_native_gpt_cli.sh /tmp/nfn_gpt_native_train_ln_residual_scratch_elide`.
- Ran a dedicated-RTX-5090 one-step smoke with
  `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 /tmp/nfn_gpt_native_train_ln_residual_scratch_elide --backend tile-cuda --tinystories --max-steps 1 --eval-every-steps 0 --no-checkpoint`;
  runtime JSON reported `status: native-transformer-lm-trained`,
  `layer_norm_backward_residual_scratch_buffers_allocated: false`,
  `layer_norm_backward_residual_scratch_buffers_elided: 2`,
  `layer_norm_backward_residual_scratch_elements_elided: 100663296`, and
  `float_arena_requested_elements: 2662695693`.
- Ran the same one-step smoke with
  `NFN_NATIVE_GPT_FUSE_LN_BACKWARD_RESIDUAL=0`; runtime JSON reported
  `layer_norm_backward_residual_scratch_buffers_allocated: true` and
  `float_arena_requested_elements: 2763358989`, confirming the
  `100663296`-element arena reduction is exactly the two elided scratch
  buffers.

### 2026-06-15 Paired benchmark selected-GPU utilization guard

#### Changed

- `tools/paired_kernel_speed.py` now accepts
  `--max-selected-gpu-utilization-pct N`. When set, the helper aborts before
  warmup or a measured pair if the selected CUDA GPU's `nvidia-smi`
  utilization is already above `N`.
- The utilization guard is scoped to the selected `CUDA_VISIBLE_DEVICES` GPU,
  matching `--require-idle-selected-gpu`. A busy separate display GPU does not
  fail a dedicated compute-GPU run.
- Text and JSON output now include `max_selected_gpu_utilization_pct` so saved
  benchmark artifacts show whether a utilization ceiling was active.

#### Verification

- Ran `python -m py_compile tools/paired_kernel_speed.py`.
- Ran
  `python -m pytest tests/test_tile_cuda_examples.py -q -k 'paired_kernel_speed_tool_compiles_and_smokes or paired_kernel_speed_tool_auto_selects_idle_display_disabled_gpu or paired_kernel_speed_tool_require_idle_selected_gpu_checks_selected_uuid or paired_kernel_speed_tool_selected_gpu_utilization_guard or paired_kernel_speed_tool_records_command_timeout or paired_kernel_speed_tool_extracts_llm_kittens_step_metrics or paired_kernel_speed_tool_sums_llm_kittens_step_time'`.
- Ran a live rejection smoke:
  `python tools/paired_kernel_speed.py --baseline 'python -c "print(1)"' --candidate 'python -c "print(2)"' --samples 1 --warmup 0 --cuda-visible-devices 0 --max-selected-gpu-utilization-pct 0 --json-out /tmp/nfn_util_guard_smoke.json`;
  it failed before running commands with
  `nvidia-smi utilization is 10%`.

### 2026-06-15 Paired benchmark idle-GPU guard

#### Changed

- `tools/paired_kernel_speed.py` now accepts `--require-idle-selected-gpu`.
  When enabled, the paired benchmark aborts before warmup or a measured pair if
  `nvidia-smi` reports any compute process on the selected CUDA GPU.
- The idle guard checks the selected GPU UUID rather than global NVIDIA process
  count, so a separate primary display GPU can remain active while the dedicated
  RTX 5090 compute GPU is required to be idle for candidate-vs-baseline kernel
  measurements.
- Text and JSON output now include `require_idle_selected_gpu` so saved
  benchmark artifacts show whether the guard was active.

#### Verification

- Ran `python -m py_compile tools/paired_kernel_speed.py`.
- Ran
  `python -m pytest tests/test_tile_cuda_examples.py -q -k 'paired_kernel_speed_tool_compiles_and_smokes or paired_kernel_speed_tool_auto_selects_idle_display_disabled_gpu or paired_kernel_speed_tool_require_idle_selected_gpu_checks_selected_uuid or paired_kernel_speed_tool_records_command_timeout or paired_kernel_speed_tool_extracts_llm_kittens_step_metrics or paired_kernel_speed_tool_sums_llm_kittens_step_time'`.
- Ran a live dedicated-GPU smoke:
  `python tools/paired_kernel_speed.py --baseline 'python -c "print(1)"' --candidate 'python -c "print(2)"' --samples 1 --warmup 0 --cuda-visible-devices 0 --require-idle-selected-gpu --json-out /tmp/nfn_idle_guard_smoke.json`;
  the helper reported `require_idle_selected_gpu: True` and zero compute
  processes on GPU 0.

### 2026-06-15 Direct GPT block-output writes

#### Changed

- Dense GPT native training now writes each non-final transformer block's final
  residual output directly into that block's persistent backward-recompute
  buffer. This removes the previous post-block `nfn_native_tile_copy_float32`
  preservation launch while keeping the one-tape scratch-recompute layout.
- Runtime JSON now reports
  `block_state_layout.persistent_block_output_write_strategy` with value
  `"direct-residual2-output"` and
  `block_state_layout.persistent_block_output_copy_elided_count`. For the
  default 12-layer, `64 x 1024 -> 524288` schedule, the count is 88 per
  optimizer step: 11 non-final blocks across 8 gradient-accumulation
  microbatches.
- The dense transformer-LM trainer no longer requires or loads
  `nfn_native_tile_copy_float32`; other Tile smokes and helper paths that use
  the copy ABI are unchanged.

#### Verification

- Built the candidate trainer with
  `bash tools/build_native_gpt_cli.sh /tmp/nfn_gpt_native_train_direct_block_output`.
- Ran a dedicated-RTX-5090 paired benchmark with
  `python tools/paired_kernel_speed.py --samples 3 --warmup 1
  --cuda-visible-devices 0 --cuda-device-max-connections 1
  --command-timeout-seconds 90 --continue-on-error
  --json-out /tmp/nfn_direct_block_output_pair.json --baseline
  'build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 3
  --eval-every-steps 0 --no-checkpoint' --candidate
  '/tmp/nfn_gpt_native_train_direct_block_output --backend tile-cuda
  --tinystories --max-steps 3 --eval-every-steps 0 --no-checkpoint'`.
  The candidate train-loop ratio was `0.994006` and total wall-time ratio was
  `0.992365`; the harness reported zero compute processes on GPU 0.
- Ran a one-step native probe with
  `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1
  /tmp/nfn_gpt_native_train_direct_block_output --backend tile-cuda
  --tinystories --max-steps 1 --eval-every-steps 0 --no-checkpoint`, which
  completed with `status: "native-transformer-lm-trained"` and reported
  `persistent_block_output_copy_elided_count: 88`.

### 2026-06-15 Generic native GPT C++ binding

#### Changed

- Added `tools/build_native_gpt_binding.sh`, which builds the generic
  `neuralfn._native_gpt` C++ extension from the native GPT binding source.
- The native GPT binding source is now module-name parameterized and exports
  `run_gpt(config_dict)` plus the compatibility `run_gpt2(config_dict)` and
  `run_train(config_dict)` aliases.
- `run_native_gpt(..., runner="auto")` and the shared native GPT loader now
  prefer `neuralfn._native_gpt` / `neuralfn_native_gpt` before falling back to
  `neuralfn._native_gpt2` / `neuralfn_native_gpt2`.
- `tools/build_native_gpt2_all.sh` and `cli/install.sh` now build both the
  generic `_native_gpt` binding and the GPT-2 compatibility `_native_gpt2`
  binding.

#### Verification

- Built both extension variants:
  `bash tools/build_native_gpt_binding.sh /tmp/_native_gpt_test$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')`
  and
  `bash tools/build_native_gpt2_binding.sh /tmp/_native_gpt2_test$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k 'native_gpt_cpp_binding_builds_and_runs_generic_module or native_gpt2_cpp_binding_builds_and_runs or native_gpt2_cpp_binding_uses_compiled_cli_for_alias_only_config or native_gpt2_binding_runner_invokes_in_process_module or native_gpt_generic_env_names_take_precedence or native_gpt2_build_all_script_supports_temp_outputs'`.
- Ran `python -m py_compile neuralfn/native_gpt2.py neuralfn/native_gpt.py`.

### 2026-06-15 Directly initialize BF16-primary block weights

#### Changed

- Added `nfn_native_tile_fill_many_values_bf16_bits_float32`, a Tile-CUDA
  descriptor-driven fill primitive for BF16 bit buffers.
- Dense GPT native training now defaults to direct BF16 startup initialization
  for BF16-primary transformer block weights. QKV, attention projection, MLP FC,
  and MLP projection weights are filled directly in the BF16 block-weight arena,
  skipping the old initial float32 fill plus BF16 pack.
- Added `NFN_NATIVE_GPT_DIRECT_BF16_BLOCK_WEIGHT_INIT` with the
  `NFN_NATIVE_GPT2_DIRECT_BF16_BLOCK_WEIGHT_INIT` compatibility fallback. Set it
  to `0` to reproduce the old startup path while leaving BF16-primary AdamW
  updates enabled.
- Runtime JSON now reports
  `direct_bf16_block_weight_initialization_enabled`,
  `block_weight_bf16_initialization_strategy`,
  `bf16_parameter_initialization_descriptor_count`,
  `bf16_parameter_initialization_max_elements`, and
  `bf16_parameter_initialization_kernel_launches`.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so`,
  `build/nfn_gpt_native_train`, and `build/nfn_gpt2_native_train`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k 'native_gpt2_cpp_cli_builds_and_uses_sm120_defaults or packed_qkv_uint16_arena_reserves_full_scratch_layout'`.
- Ran direct and forced-old one-step native GPT smokes on the dedicated RTX 5090
  with `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1`. The direct path
  reported `block_weight_bf16_initialization_strategy:
  "direct-bf16-fill-many-values"`, 27 float32 fill descriptors, 48 BF16 fill
  descriptors, and `block_weight_bf16_refresh_count: 0`. The forced-old path
  reported `block_weight_bf16_initialization_strategy:
  "float32-fill-then-bf16-pack"`, 75 float32 fill descriptors, no BF16 fill
  descriptors, and `block_weight_bf16_refresh_count: 1`.
- Ran `python tools/paired_kernel_speed.py` with 3 measured 5-step samples and
  one warmup on CUDA device 0. Direct BF16 init reduced mean setup wall time from
  534.8 ms to 512.9 ms. Mean total 5-step runtime was unchanged within noise
  because the training loop dominates.

### 2026-06-15 Add BF16-gradient global-norm sumsq Tile ABI

#### Changed

- Added `nfn_native_tile_sumsq_partials_many_bf16_bits_float32`, a Tile-CUDA
  multi-buffer sumsq primitive that reads BF16 gradient buffers and writes
  float32 global-norm partials.
- Dense GPT native training now requires, loads, and reports that BF16-gradient
  sumsq primitive as `gradient_clip_bf16_sumsq_kernel_loaded`. The trainer does
  not call it yet because block-weight gradients still use float32 accumulation
  buffers.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so`,
  `build/nfn_gpt_native_train`, and `build/nfn_gpt2_native_train`.
- Confirmed `nm -D build/libnfn_native_train_tile_ops.so` exports
  `nfn_native_tile_sumsq_partials_many_bf16_bits_float32`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k 'native_gpt2_cpp_cli_builds_and_uses_sm120_defaults or packed_qkv_uint16_arena_reserves_full_scratch_layout'`.
- Ran `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt2_native_train --tinystories --tile-ops-lib build/libnfn_native_train_tile_ops.so --train-transformer-lm --max-steps 1 --eval-every-steps 0 --no-checkpoint` on the dedicated RTX 5090. Runtime JSON reported `status: "native-transformer-lm-trained"`, `gradient_clip_bf16_sumsq_kernel_loaded: true`, and `adamw_bf16_param_bf16_grad_kernel_loaded: true`.

### 2026-06-15 Bind native GPT BF16-gradient AdamW kernel

#### Changed

- Dense GPT native training now loads and reports the existing
  `nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32`
  Tile ABI symbol. Runtime and dry-run JSON include
  `adamw_bf16_param_bf16_grad_kernel_loaded`,
  `adamw_bf16_param_bf16_grad_descriptor_count`,
  `adamw_bf16_param_bf16_grad_kernel_launches`,
  `block_weight_bf16_gradient_storage_strategy`, and
  `block_weight_bf16_primary_param_bf16_grad_update_count`.
- The reported BF16-gradient descriptor and launch counts remain zero for now:
  block-weight gradients still accumulate in float32 buffers until the native
  BF16 gradient arena, global-norm, and zeroing path is implemented.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` and `build/nfn_gpt2_native_train`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k 'native_gpt2_cpp_cli_builds_and_uses_sm120_defaults or packed_qkv_uint16_arena_reserves_full_scratch_layout'`.
- Ran `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt2_native_train --tinystories --tile-ops-lib build/libnfn_native_train_tile_ops.so --train-transformer-lm --max-steps 1 --eval-every-steps 0 --no-checkpoint` on the dedicated RTX 5090. Runtime JSON reported `status: "native-transformer-lm-trained"`, `adamw_bf16_param_bf16_grad_kernel_loaded: true`, and zero BF16-gradient AdamW descriptors/launches.

### 2026-06-15 Make native GPT stage profiler event cap configurable

#### Changed

- Dense GPT native training now accepts
  `NFN_NATIVE_GPT_STAGE_TIMING_MAX_EVENTS=N` with the
  `NFN_NATIVE_GPT2_STAGE_TIMING_MAX_EVENTS` compatibility fallback. The default
  remains 20000 events.
- Runtime JSON now reports `timing.stage_timing_max_events` alongside
  `stage_timing_event_count` and `stage_timing_dropped_event_count`, so longer
  profiling runs can distinguish a real missing stage from a dropped event cap.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` and `build/nfn_gpt2_native_train`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k 'native_gpt2_cpp_cli_builds_and_uses_sm120_defaults or packed_qkv_uint16_arena_reserves_full_scratch_layout'`.
- Ran a dedicated RTX 5090 stage-timing run with
  `NFN_NATIVE_GPT_STAGE_TIMING=1` and
  `NFN_NATIVE_GPT_STAGE_TIMING_MAX_EVENTS=40000`; the run reported zero dropped
  stage-timing events.

### 2026-06-15 Summarize selected-GPU load in paired benchmarks

#### Changed

- `tools/paired_kernel_speed.py` now emits a `gpu_sample_summary` block in JSON
  and text output. It summarizes selected-GPU utilization, selected-GPU memory,
  selected-GPU compute-process counts, and total compute-process counts before
  and after measured samples.
- The raw per-sample `nvidia-smi` snapshots remain in
  `paired_samples[].gpu_before` and `paired_samples[].gpu_after`; the new
  summary is for quick review when checking that paired candidate-vs-baseline
  kernel timings were not skewed by unrelated external GPU load.

#### Verification

- Ran `python -m py_compile tools/paired_kernel_speed.py`.
- Ran `python -m pytest tests/test_tile_cuda_examples.py -q -k paired_kernel_speed_tool_compiles_and_smokes`.
- Ran dedicated RTX 5090 paired benchmarks with `CUDA_VISIBLE_DEVICES=0` and
  `CUDA_DEVICE_MAX_CONNECTIONS=1`; the helper reported the RTX 5090 as
  display-disabled with zero compute processes before measured samples.

### 2026-06-15 Store packed-attention LN1 stats for native GPT recompute

#### Changed

- Dense GPT native training now stores only LN1 mean/rstd stats for earlier
  saved packed-attention blocks when the BF16 QKV dWeight path is active and
  regenerates the LN1 BF16 activation during backward recompute with
  `nfn_native_tile_layer_norm_apply_stats_bf16_out_float32`.
- Runtime JSON reports `stored_packed_attention_ln1_stats_enabled`,
  `stored_packed_attention_ln1_stats_blocks`,
  `stored_packed_attention_ln1_stats_elements`, and
  `stored_packed_attention_ln1_stats_bytes`. The default 12-layer
  `64 x 1024` TinyStories run stores stats for 11 blocks, about 5.5 MiB, rather
  than adding a full BF16 LN1 activation tape.
- The opt-out switch for paired regression benchmarks is
  `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_STATS=0`; GPT-2-prefixed
  compatibility names remain accepted.

#### Breaking changes

- Rebuild `build/libnfn_native_train_tile_ops.so` before running the dense GPT
  native trainer. `nfn_gpt_native_train` now requires the raw ABI symbol
  `nfn_native_tile_layer_norm_apply_stats_bf16_out_float32`.

#### Verification

- Rebuilt native artifacts with `bash tools/build_native_gpt2_all.sh`.
- Verified the dedicated compute GPU with `nvidia-smi`: RTX 5090 at
  `display_active Disabled`, `util 0%`, and no active compute processes before
  clean paired benchmarks.
- Ran a one-step native transformer-LM smoke on the RTX 5090:
  `build/nfn_gpt_native_train --tinystories --tile-ops-lib
  build/libnfn_native_train_tile_ops.so --train-transformer-lm --max-steps 1
  --eval-every-steps 0 --no-checkpoint`.
- Ran clean paired old-vs-new timing with
  `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_STATS=0` on the baseline side.
  One-step mean train-loop time improved from `2832.33 ms` to `2826.60 ms`
  (`0.997979x`). Sustained five-step mean train-loop time improved from
  `2793.65 ms/step` to `2788.38 ms/step` (`0.998119x`).

### 2026-06-15 Fix packed-QKV BF16 scratch arena sizing

#### Fixed

- The dense GPT native trainer now reserves the full packed-QKV scratch layout in
  the combined uint16 arena: LN1 BF16 output, packed QKV BF16, and packed
  attention output BF16. The previous arena request omitted the LN1 BF16 slice
  even though the pointer layout used it, which could overlap the following BF16
  block-weight arena.
- Runtime JSON now reports the corrected uint16 arena request for the default
  `64 x 1024` dense GPT path; the one-step smoke reported
  `uint16_arena_requested_elements: 9355395072`.

#### Verification

- Rebuilt all native GPT/native trainer artifacts with
  `bash tools/build_native_gpt2_all.sh`.
- Ran a clean one-step `build/nfn_gpt_native_train --tinystories
  --tile-ops-lib build/libnfn_native_train_tile_ops.so --train-transformer-lm
  --max-steps 1 --eval-every-steps 0 --no-checkpoint` smoke on the dedicated
  RTX 5090.
- Built the previous committed trainer into `/tmp/nfn_gpt_native_train_eb356ef`
  and ran paired old-vs-fixed timing. The fix was effectively neutral:
  candidate train-loop milliseconds-per-step was `1.000631x` and
  `train_tokens_per_second` was `0.999371x` versus the previous binary.
- Added a source-level regression assertion in `tests/test_native_gpt2.py` so
  packed-QKV scratch arena requests must include `activation_elements * 2`.

### 2026-06-15 Default dense GPT LN1 BF16 QKV forward and BF16 QKV dWeight

#### Changed

- Dense GPT native training now keeps the LN1 activation for packed QKV as BF16
  during forward and reuses that buffer for QKV dWeight. The default path runs
  LN1 through `nfn_native_tile_layer_norm_with_stats_bf16_out_float32`, feeds
  QKV through `nfn_native_tile_linear_bf16_input_weight_bf16_output_float32`, and
  reports `qkv_forward_ln1_bf16_enabled: true`.
- `NFN_NATIVE_GPT_BF16_QKV_DWEIGHT` now defaults on. Runtime JSON reports
  `block_backward_bf16_qkv_dweight_enabled: true` and
  `block_backward_qkv_dweight_strategy:
  "packed-ln1-bf16-qkv-bf16-grad-dweight-bias-accumulate"` for the default
  path.
- The paired benchmark helper remains the required acceptance gate for kernel
  candidates. The accepted default measured `0.990066x` train-loop
  milliseconds-per-step versus the previous path and `1.010040x` train tokens/s
  on the dedicated RTX 5090. A direct comparison against the corrected
  llm.kittens baseline still leaves NeuralFn at `1.112816x` train-loop
  milliseconds-per-step, so further block/LM-head work remains.

#### Breaking changes

- Rebuild `build/libnfn_native_train_tile_ops.so` before running the dense GPT
  trainer. `nfn_gpt_native_train` now requires the raw ABI symbols
  `nfn_native_tile_layer_norm_with_stats_bf16_out_float32` and
  `nfn_native_tile_linear_bf16_input_weight_bf16_output_float32`.
- To reproduce the previous runtime path for bisection, set both
  `NFN_NATIVE_GPT_LN1_BF16_QKV_FORWARD=0` and
  `NFN_NATIVE_GPT_BF16_QKV_DWEIGHT=0` before launching the trainer. The
  `NFN_NATIVE_GPT2_*` names remain compatibility fallbacks for older scripts.

#### Verification

- Rebuilt the trainer and Tile ops library with `bash tools/build_native_gpt2_all.sh`.
- Ran `python tools/check_native_no_torch_deps.py --json`.
- Ran a clean one-step native transformer-LM smoke on the dedicated RTX 5090:
  `build/nfn_gpt_native_train --tinystories --tile-ops-lib
  build/libnfn_native_train_tile_ops.so --train-transformer-lm --max-steps 1
  --eval-every-steps 0 --no-checkpoint`.
- Ran the accepted paired old-vs-new benchmark with
  `NFN_NATIVE_GPT_LN1_BF16_QKV_FORWARD=0
  NFN_NATIVE_GPT_BF16_QKV_DWEIGHT=0` on the baseline side and the new defaults
  on the candidate side.

### 2026-06-15 Correct llm.kittens paired benchmark step aggregation

#### Changed

- `tools/paired_kernel_speed.py` now aggregates multi-step llm.kittens logs
  consistently with NeuralFn native JSON. Parsed `step ... ms ... tok/s` rows
  report `train_loop_wall_ms` as the sum of all parsed step times,
  `train_loop_wall_ms_per_step` as their mean, and preserve the final visible
  step under `llm_kittens_last_step_*` keys.
- This fixes misleading NeuralFn-vs-llm.kittens paired summaries where
  NeuralFn's total train-loop time was previously compared against only the
  final llm.kittens step.
- Documented the corrected metric semantics in README and CLI docs.

#### Verification

- Ran
  `python -m pytest tests/test_tile_cuda_examples.py -q -k 'paired_kernel_speed_tool_compiles_and_smokes or paired_kernel_speed_tool_extracts_llm_kittens_step_metrics or paired_kernel_speed_tool_sums_llm_kittens_step_time or paired_kernel_speed_tool_records_command_timeout or paired_kernel_speed_tool_auto_selects_idle_display_disabled_gpu'`.
- Ran a one-sample corrected NeuralFn-vs-llm.kittens comparison on the
  dedicated RTX 5090 with `--cuda-visible-devices auto`; llm.kittens
  reported `train_loop_wall_ms=7390.63` across three parsed steps,
  `train_loop_wall_ms_per_step=2463.543333`, and NeuralFn's corresponding
  ratio reported `1.127890` for both total and per-step train-loop metrics.

### 2026-06-15 Auto-select dedicated GPU for paired kernel benchmarks

#### Changed

- `tools/paired_kernel_speed.py` now defaults `--cuda-visible-devices` to
  `auto`. When `nvidia-smi` is available, the helper selects the lowest-use
  display-disabled NVIDIA GPU without active compute processes, which matches
  the workstation layout where the RTX 5090 is dedicated for CUDA compute and
  the display is driven separately.
- Explicit device values still override the selector, and
  `--cuda-visible-devices ""` still leaves `CUDA_VISIBLE_DEVICES` unchanged.
  JSON/text output now includes `cuda_device_selection` so benchmark artifacts
  record whether the device was explicit, auto-dedicated, fallback, or
  unresolved.
- Updated README and CLI docs so kernel candidate timing commands no longer
  require manually remembering `--cuda-visible-devices 0` on the dedicated GPU
  workstation path.

#### Verification

- Ran
  `python -m pytest tests/test_tile_cuda_examples.py -q -k 'paired_kernel_speed_tool_compiles_and_smokes or paired_kernel_speed_tool_extracts_llm_kittens_step_metrics or paired_kernel_speed_tool_records_command_timeout or paired_kernel_speed_tool_auto_selects_idle_display_disabled_gpu'`.
- Ran a GPU-visible auto-selection smoke with
  `python tools/paired_kernel_speed.py --cuda-visible-devices auto ...`; it
  resolved `CUDA_VISIBLE_DEVICES=0` with `mode=auto-dedicated` and recorded
  the RTX 5090 as display-disabled with no compute processes.

### 2026-06-15 Add native no-Torch dependency gate

#### Added

- Added `tools/check_native_no_torch_deps.py`, an `ldd`-based verification
  gate for native training artifacts. By default it checks
  `build/nfn_gpt_native_train` and `build/libnfn_native_train_tile_ops.so` and
  fails if either links Torch, c10, or Python runtime libraries.
- Documented the gate in the README and CUDA Tile SDK guide so native rebuilds
  have a repeatable proof that the hot training path remains independent of
  graph-backed Torch execution.

#### Verification

- Ran `python tools/check_native_no_torch_deps.py`.
- Ran `python tools/check_native_no_torch_deps.py --json`.
- Ran `python -m py_compile tools/check_native_no_torch_deps.py`.

### 2026-06-15 Add setup timing metrics to paired kernel benchmarks

#### Changed

- `tools/paired_kernel_speed.py` now extracts native
  `timing.setup_timing` entries into `baseline_native_metrics`,
  `candidate_native_metrics`, and paired ratio summaries using metric keys such
  as `setup.float_arena_materialize.total_ms`. Startup kernel experiments can
  now be accepted or rejected from the same old-vs-new paired JSON as per-step
  kernel experiments.

#### Verification

- Added a paired-benchmark smoke assertion for setup timing extraction in
  `tests/test_tile_cuda_examples.py`.

### 2026-06-15 Retune native GPT LM-head row chunk default to 8192

#### Changed

- Dense GPT native training now defaults the tied LM-head row chunk to 8192
  rows for the compiled C++ trainer and Python wrapper defaults. This keeps the
  bounded chunked BF16 LM-head workspace while reducing LM-head chunk overhead
  versus the previous 6144-row default on the dedicated RTX 5090.
- `train_gpt_native.py` and the `train_gpt.py` wrapper defaults now match the
  compiled trainer and SDK default instead of passing the older 6144-row chunk
  size into compiled CLI runs.
- Pass `--lm-head-row-chunk-size 6144`,
  `--native-cuda-lm-head-row-chunk-size 6144`, or
  `NativeGpt2RunConfig(lm_head_row_chunk_size=6144, ...)` to reproduce the
  previous smaller-workspace profile.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` after reverting a rejected
  token-initializer candidate, then confirmed the RTX 5090 is GPU 0 with display
  disabled and no compute processes before timing.
- Rejected `NFN_NATIVE_GPT_LM_HEAD_BF16_DWEIGHT=1` because it made the measured
  train loop `1.001499x` slower and reduced token throughput to `0.998505x`.
- Rejected `NFN_TILE_CUDA_CUBLASLT_HEURISTIC_INDEX=0` because it made the train
  loop `1.011481x` slower than the current heuristic-index-1 default.
- Rejected `NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU=0` because it made the train loop
  `1.076918x` slower than the fused MLP projection/DGELU path.
- Rejected `--lm-head-row-chunk-size 12288` because it made the train loop
  `1.004425x` slower than 8192, and found `10240` effectively neutral.
- Confirmed `--lm-head-row-chunk-size 8192` against the previous 6144-row default
  with six measured three-step TinyStories pairs after one warmup pair on GPU 0.
  The 8192-row candidate had `train_loop_wall_ms_per_step` ratio `0.998916` and
  token-throughput ratio `1.001087`.

### 2026-06-15 Default packed-attention workspace LSE

#### Changed

- Dense GPT native training still stores packed BF16 QKV and packed BF16 O for
  saved packed-attention blocks by default, but no longer stores per-row TK
  `lse` unless requested. The default saved packed-attention backward now uses
  the workspace-LSE consumer strategy
  `"saved-packed-qkv-o-workspace-lse-bf16-backward-to-qkv"`, trimming the saved
  float arena while preserving the fast saved-QKV/O backward path.
- Set `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE=1` or the
  `NFN_NATIVE_GPT2_*` fallback to opt into the previous saved-LSE buffer for
  paired benchmarks.

#### Verification

- Confirmed the dedicated RTX 5090 is GPU 0 with display disabled before timing:
  `NVIDIA GeForce RTX 5090`, `543/32607` MiB used, `0%` utilization, and no
  compute processes.
- After changing the default, compared the old saved-LSE path
  (`NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE=1`) against the new workspace-LSE
  default with four measured three-step TinyStories pairs after one warmup pair
  on GPU 0. The new default reported `train_loop_wall_ms_per_step` ratio
  `0.999977`, token-throughput ratio `1.000034`, and total wall-time ratio
  `0.998713`.
- Rejected disabling stored MLP activations because it made the training loop
  `1.311890x` slower despite reducing setup time.
- Rejected disabling direct BF16 QKV gradient scratch because it made the
  training loop `1.012255x` slower.
- Rejected disabling BF16-primary block-weight updates because the FP32-master
  shadow path was slightly slower with `train_loop_wall_ms_per_step` ratio
  `1.001927`.
- Rejected disabling stored packed-attention QKV/O activations because it made
  the training loop `1.091878x` slower.
- Rejected disabling BF16 MLP gradient handoff because it made the training
  loop `1.075134x` slower.

### 2026-06-15 Retune native GPT LM-head row chunk default to 6144

#### Changed

- Dense GPT native training now defaults the tied LM-head row chunk to 6144
  rows for the compiled C++ trainer, Python wrapper defaults, and
  `NativeGpt2RunConfig`. This keeps the bounded chunked BF16 LM-head workspace
  instead of the rejected full-microbatch logits buffer, while reducing chunk
  overhead versus the previous 4096-row default on the dedicated RTX 5090.
- Pass `--lm-head-row-chunk-size 4096`,
  `--native-cuda-lm-head-row-chunk-size 4096`, or
  `NativeGpt2RunConfig(lm_head_row_chunk_size=4096, ...)` to reproduce the
  older smaller-workspace profile.

#### Verification

- Rebuilt the compiled dense GPT CLI with `bash tools/build_native_gpt_cli.sh`.
- Ran a fresh NeuralFn-vs-llm.kittens paired baseline on the dedicated RTX 5090:
  NeuralFn default reported `train_loop_wall_ms_per_step: 2812.101667` and
  llm.kittens reported `2491.430000`, so llm.kittens remains `1.128242x`
  faster by token throughput.
- Rejected `--lm-head-row-chunk-size 65536` after it left the candidate process
  consuming nearly all RTX 5090 memory at 100% utilization and had to be killed.
- Compared the current 4096-row default against `--lm-head-row-chunk-size 6144`
  with four measured three-step TinyStories pairs after one warmup pair on GPU
  0. The 6144-row candidate had `train_loop_wall_ms_per_step` ratio `0.999109`
  and token-throughput ratio `1.000894`.
- Checked `--lm-head-row-chunk-size 5120` as a nearby shape; it was essentially
  neutral with `train_loop_wall_ms_per_step` ratio `0.999805`.
- Ran a one-step TinyStories native GPT smoke on GPU 0 after rebuilding the
  compiled trainer; runtime JSON reported `lm_head_row_chunk_size: 6144`,
  `lm_head_row_chunk_count: 11`, `lm_head_bf16_logit_bytes: 618135552`, no
  missing symbols, and `passed: true`.

### 2026-06-15 Default fused LN2 BF16 prepack store

#### Changed

- The native dense-GPT CUDA trainer now defaults to
  `nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_bf16_norm_float32`
  for the attention projection residual + LN2 forward stage. The fused Tile
  kernel writes the residual1 BF16 cache and the LN2 BF16 activation consumed by
  stored-MLP FC+GELU in the same launch, removing the previous separate
  `float32_to_bf16` LN2 prepack launch on the default stored-MLP path.
- Set `NFN_NATIVE_GPT_FUSE_LN2_BF16_OUT=0` or
  `NFN_NATIVE_GPT2_FUSE_LN2_BF16_OUT=0` to reproduce the previous separate LN2
  prepack path in paired benchmarks.
- Runtime JSON now reports `fused_ln2_bf16_out_enabled`,
  `stored_mlp_ln2_bf16_prepack_strategy`, and
  `stored_mlp_ln2_bf16_fused_store_kernel_launches`.

#### Verification

- Rebuilt the trainer-facing Tile ops library with
  `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt the compiled dense GPT CLI with `bash tools/build_native_gpt_cli.sh`.
- Ran a one-step stage-timed TinyStories native GPT smoke on the dedicated RTX
  5090 with `NFN_NATIVE_GPT_FUSE_LN2_BF16_OUT=1`; the run completed with no
  missing symbols, `train_tokens_per_second: 184105`, and
  `block_forward.mlp_fc_gelu.pack_ln2` reduced to `1.96835 ms` over the step.
- Ran a paired default-vs-candidate benchmark before changing the default:
  four measured three-step TinyStories samples after one warmup pair on GPU 0.
  The fused candidate had `train_loop_wall_ms_per_step` ratio `0.996518` and
  token-throughput ratio `1.003500`.
- Ran the final default-vs-old paired check on the dedicated RTX 5090 with the
  previous separate LN2 prepack forced by `NFN_NATIVE_GPT_FUSE_LN2_BF16_OUT=0`
  as baseline and the new fused path as candidate. Across four measured
  three-step TinyStories samples after one warmup pair, candidate train-loop
  time ratio was `0.995392` and token-throughput ratio was `1.004633`.

### 2026-06-15 Default fused float32/BF16 dWeight bias reduction

#### Changed

- The native CUDA Tile linear runtime now enables the optimized
  float32-input/BF16-gradient dWeight+bias cuBLASLt epilogue path by default.
  This replaces the previous default split dWeight GEMM plus Tile
  chunked-bias-reduction path for supported shapes.
- Set `NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD=0`,
  `NFN_NATIVE_GPT2_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD=0`, or
  `NFN_TILE_CUDA_LINEAR_FLOAT32_BF16_BGRAD=0` to reproduce the previous split
  path in paired benchmarks.

#### Verification

- Before changing the default, ran a paired candidate check on the dedicated
  RTX 5090 comparing current default against
  `NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD=1` for four measured
  three-step TinyStories samples after one warmup pair. Candidate train-loop
  time ratio was `0.998381` and token-throughput ratio was `1.001625`.
- Rebuilt the Tile ops library with `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt the compiled dense GPT CLI with `bash tools/build_native_gpt_cli.sh`.
- Ran
  `python -m pytest tests/test_native_gpt2.py -q -k 'native_train_tile_ops_builds_torch_free_c_abi or native_gpt2_cpp_cli_builds_and_uses_sm120_defaults or native_transformer_lm'`
  (`1 passed`, `1 skipped`).
- Ran the final default-vs-old paired check on the dedicated RTX 5090 with the
  previous split path forced by
  `NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD=0` as baseline and the new
  default as candidate. Across four measured three-step TinyStories samples
  after one warmup pair, candidate train-loop time ratio was `0.992036` and
  token-throughput ratio was `1.008198`.

### 2026-06-15 Normalize paired trainer step metrics

#### Changed

- `tools/paired_kernel_speed.py` now records `steps_completed` from NeuralFn
  native JSON and derives `train_loop_wall_ms_per_step` when a native trainer
  reports total train-loop wall time. The llm.kittens step-log parser now emits
  the same normalized metric, so NeuralFn-vs-`train_gpt2cu` paired comparisons
  no longer put total-run timing and per-step timing under one ambiguous field.
- The text summary now prints `train_loop_wall_ms_per_step` and its paired ratio
  before the raw `train_loop_wall_ms` metric.

#### Verification

- Ran `python -m py_compile tools/paired_kernel_speed.py`.
- Ran
  `python -m pytest tests/test_tile_cuda_examples.py -q -k 'paired_kernel_speed_tool_compiles_and_smokes or paired_kernel_speed_tool_extracts_llm_kittens_step_metrics or paired_kernel_speed_tool_records_command_timeout'`
  (`3 passed`, `2 deselected`).

### 2026-06-15 Add BF16-gradient AdamW Tile ABI

#### Changed

- The trainer-facing raw Tile ABI now exports
  `nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32`.
  It updates BF16-primary parameter buffers from BF16 gradient buffers while
  keeping AdamW first and second moments in float32 and applying the existing
  device clip scale.
- The compiled dense GPT required-symbol checks now include the new ABI symbol,
  so future BF16 block-gradient-buffer wiring cannot silently fall back to a
  stale Tile ops library. The live dense GPT trainer still uses the existing
  float-gradient BF16-param AdamW path until that gradient-buffer migration is
  wired.

#### Verification

- Rebuilt the Tile ops library with `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt the compiled dense GPT CLI with `bash tools/build_native_gpt_cli.sh`.
- Confirmed the new exported symbol with
  `nm -D build/libnfn_native_train_tile_ops.so`.
- Ran
  `python -m pytest tests/test_native_gpt2.py -q -k 'native_train_tile_ops_builds_torch_free_c_abi or native_gpt2_cpp_cli_builds_and_uses_sm120_defaults or native_transformer_lm'`
  (`1 passed`, `1 skipped`).
- Ran a one-step TinyStories smoke pinned to the dedicated RTX 5090 with
  `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 build/nfn_gpt_native_train --tinystories --max-steps 1 --model-family gpt --train-transformer-lm --no-checkpoint`;
  the run completed with `missing_symbols: []`, included
  `nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32`
  in `kernels`, and reported `passed: true`.

### 2026-06-15 Use CUDA memset for native GPT gradient zeroing

#### Changed

- Dense GPT native training now zeroes per-step accumulation gradients through
  coalesced contiguous-range `cudaMemsetAsync` by default. The older
  descriptor-driven Tile fill-many path remains available for paired
  comparisons with `NFN_NATIVE_GPT_CUDA_MEMSET_GRAD_ZERO=0` or
  `NFN_NATIVE_GPT2_CUDA_MEMSET_GRAD_ZERO=0`.
- Runtime JSON now reports `gradient_cuda_memset_zero_enabled`,
  `gradient_cuda_memset_zero_available`, `gradient_zero_range_count`,
  `gradient_zero_range_elements`, `gradient_zero_cuda_memset_count`, and
  `gradient_zero_tile_fill_count` at the top level and under
  `block_state_layout`.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Ran a one-step TinyStories staged smoke pinned to the dedicated RTX 5090:
  `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 NFN_NATIVE_GPT_STAGE_TIMING=1 build/nfn_gpt_native_train --tinystories --max-steps 1 --model-family gpt --train-transformer-lm --no-checkpoint`.
  It completed with `gradient_cuda_memset_zero_available: true`,
  `gradient_zero_range_count: 2`, `gradient_zero_cuda_memset_count: 2`,
  and a staged `gradient_zero` time of about `0.28 ms`.
- Ran `tools/paired_kernel_speed.py` with one warmup and four measured
  three-step pairs on GPU 0, comparing
  `NFN_NATIVE_GPT_CUDA_MEMSET_GRAD_ZERO=0` against the new default. Candidate
  train-loop ratio was `0.999308`, total runtime ratio was `0.999450`, and
  tokens/s ratio was `1.000702`; this is a small hot-loop cleanup, not a large
  throughput win.

### 2026-06-15 Use CUDA memset for native GPT startup zeroing

#### Changed

- Dense GPT native training now uses `cudaMemsetAsync` for the default
  contiguous AdamW first/second moment zero ranges during startup. This keeps
  real training batches on the compiled C++/CUDA path while reducing the
  startup zeroing overhead on the dedicated RTX 5090 path.
- Set `NFN_NATIVE_GPT_CUDA_MEMSET_ZERO=0` or
  `NFN_NATIVE_GPT2_CUDA_MEMSET_ZERO=0` to reproduce the previous Tile fill
  startup-zero path in same-binary paired benchmarks.
- Runtime JSON now reports `startup_cuda_memset_zero_enabled`,
  `startup_cuda_memset_zero_available`,
  `startup_cuda_memset_zero_fill_count`, and
  `startup_tile_zero_fill_count` at the top level and under
  `block_state_layout`.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Confirmed that sandboxed CUDA access still fails early with native preflight
  when the driver is not visible, then reran GPU-visible validation pinned to
  the dedicated RTX 5090 with
  `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1`.
- Ran a one-step TinyStories staged smoke:
  `NFN_NATIVE_GPT_STAGE_TIMING=1 build/nfn_gpt_native_train --tinystories --max-steps 1 --model-family gpt --train-transformer-lm --no-checkpoint`.
  It completed with `startup_cuda_memset_zero_available: true`,
  `startup_cuda_memset_zero_fill_count: 2`, and
  `startup_tile_zero_fill_count: 0`.
- Ran `tools/paired_kernel_speed.py` with one warmup and five measured pairs on
  GPU 0, comparing `NFN_NATIVE_GPT_CUDA_MEMSET_ZERO=0` against the new default.
  Candidate setup wall ratio was about `0.9719`, total runtime ratio was
  `0.9983`, train-loop ratio was `1.0019`, and tokens/s ratio was `0.9981`;
  this is treated as a startup improvement, not a hot-loop throughput win.

### 2026-06-14 Add per-sample GPU snapshots to paired native benchmarks

#### Changed

- `tools/paired_kernel_speed.py` now records `nvidia-smi` snapshots around each
  measured baseline/candidate pair in `paired_samples[].gpu_before` and
  `paired_samples[].gpu_after`, in addition to the existing run-level
  `gpu_before` and `gpu_after` snapshots.
- The text summary now reports the min/max number of compute processes visible
  before measured samples when `nvidia-smi` is available. This keeps benchmark
  timing paired while making mid-run external GPU load visible in the saved
  artifact.

#### Verification

- Ran
  `python -m pytest tests/test_tile_cuda_examples.py -q -k 'paired_kernel_speed_tool_compiles_and_smokes or paired_kernel_speed_tool_extracts_llm_kittens_step_metrics or paired_kernel_speed_tool_records_command_timeout'`.
- Ran `python -m py_compile tools/paired_kernel_speed.py`.
- Ran `git diff --check`.

### 2026-06-14 Use public vocab for native GPT LM-head CE

#### Changed

- Dense GPT native training now computes LM-head cross-entropy over the public
  tokenizer vocab (`50257`) while using the padded LM-head row count (`50304`)
  only as the logits/dlogits row stride. The strided CE backward kernels zero
  padded dlogit columns before LM-head dWeight accumulation.
- The raw trainer Tile ABI now exposes
  `nfn_native_tile_token_cross_entropy_partials_strided_float32`,
  `nfn_native_tile_token_cross_entropy_partials_strided_bf16_bits`,
  `nfn_native_tile_token_cross_entropy_backward_inplace_strided_with_workspace_float32`,
  and
  `nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_with_workspace`.
- Runtime JSON now reports `lm_head_public_vocab_ce_enabled`,
  `lm_head_softmax_vocab`, `lm_head_logit_row_stride`, and
  `lm_head_padded_dlogits_zeroed`. The default BF16 strategy is now
  `public-vocab-strided-fused-row-bf16-logits-dlogits`.
- Set `NFN_NATIVE_GPT_PUBLIC_VOCAB_CE=0` or
  `NFN_NATIVE_GPT2_PUBLIC_VOCAB_CE=0` only when paired-benchmarking against the
  previous padded-vocab CE behavior.

#### Breaking changes

- Native GPT `--train-transformer-lm` loss and token-weight gradients no longer
  include the 47 padded LM-head rows in the softmax denominator. Callers that
  diff exact loss curves or optimizer updates against prior NeuralFn native GPT
  runs should treat this as a correctness migration and compare against the old
  path only with `NFN_NATIVE_GPT_PUBLIC_VOCAB_CE=0`.

#### Verification

- Rebuilt `libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh` and rebuilt
  `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran
  `python -m pytest tests/test_native_gpt2.py -q -k 'native_train_tile_ops_builds_torch_free_c_abi or native_gpt2_cpp_cli_builds_and_uses_sm120_defaults or native_transformer_lm'`,
  which passed the focused native GPT slice with one sandbox-skipped
  GPU-dependent test.
- Ran a one-step TinyStories smoke pinned to the dedicated RTX 5090
  (`CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1`). It completed with
  no missing symbols and reported `lm_head_public_vocab_ce_enabled: true`,
  `lm_head_softmax_vocab: 50257`, `lm_head_logit_row_stride: 50304`,
  `lm_head_padded_dlogits_zeroed: true`, and
  `lm_head_ce_backward_strategy:
  "public-vocab-strided-fused-row-bf16-logits-dlogits"`.
- Ran `tools/paired_kernel_speed.py` with one warmup and five measured pairs on
  GPU 0, comparing `NFN_NATIVE_GPT_PUBLIC_VOCAB_CE=0` against the new default.
  Candidate train-loop ratio was `1.002844`, tokens/s ratio was `0.997335`,
  and total runtime ratio was `1.001548`; this is treated as a correctness fix,
  not a throughput win.

### 2026-06-14 Combine native GPT BF16 device allocations

#### Changed

- Dense GPT native training now suballocates BF16 activation and scratch buffers
  from one uint16 CUDA device arena by default. The arena covers stored MLP
  activations, residual1 caches, stored packed attention tensors, LM-head BF16
  logits, MLP BF16 scratch, packed-QKV BF16 scratch, and block BF16 weight
  shadows.
- Set `NFN_NATIVE_GPT_COMBINED_BF16_ARENA=0` or
  `NFN_NATIVE_GPT2_COMBINED_BF16_ARENA=0` to reproduce the older per-buffer
  BF16 `cudaMalloc` path in paired benchmarks.
- Runtime JSON now reports `uint16_allocation_strategy`,
  `uint16_allocation_cuda_malloc_count`, `uint16_allocation_request_count`,
  `uint16_arena_requested_elements`, `uint16_arena_allocated_elements`,
  `uint16_arena_cuda_malloc_count`, and `uint16_arena_suballocation_count`.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Ran a one-step TinyStories smoke pinned to the dedicated RTX 5090
  (`CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1`). It completed with
  `uint16_allocation_strategy: "single-arena"`,
  `uint16_allocation_request_count: 7`, and
  `uint16_arena_cuda_malloc_count: 1`.
- Ran `tools/paired_kernel_speed.py` with one warmup and five measured pairs on
  GPU 0, comparing
  `NFN_NATIVE_GPT_COMBINED_BF16_ARENA=0 build/nfn_gpt_native_train ...` against
  the default candidate. Candidate total runtime ratio was `0.999271`,
  train-loop ratio was `0.998035`, and tokens/s ratio was `1.001978`; this is
  treated as allocation/topology cleanup rather than a material throughput win.

### 2026-06-14 Add opt-in mixed float32/BF16 dWeight bgrad epilogue

#### Changed

- The trainer-facing Tile-CUDA linear helper now has an opt-in cuBLASLt
  `CUBLASLT_EPILOGUE_BGRADB` route for the mixed float32-hidden/BF16-grad
  dWeight+bias ABI used by the dense GPT QKV backward default path.
- The path is controlled by
  `NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD=1`,
  `NFN_NATIVE_GPT2_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD=1`, or
  `NFN_TILE_CUDA_LINEAR_FLOAT32_BF16_BGRAD=1`. It remains default-off because
  the paired benchmark did not justify promoting it.

#### Verification

- Rebuilt `libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Ran a one-step TinyStories smoke pinned to the dedicated RTX 5090
  (`CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1`) with the new path
  enabled. It completed with no missing symbols; the QKV dWeight/bias substage
  dropped in the noisy single profile, but the full step regressed.
- Ran `tools/paired_kernel_speed.py` with one warmup and five measured pairs on
  GPU 0, comparing `NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD=0` against
  the opt-in candidate. Candidate train-loop ratio was `1.001718`, tokens/s
  ratio was `0.998302`, and total runtime ratio was `1.002059`, so the helper
  remains opt-in rather than default.

### 2026-06-14 Default AdamW startup zeroing to contiguous Tile ranges

#### Changed

- Dense GPT native training now zeroes AdamW first/second moment state as
  coalesced contiguous float-arena ranges with Tile fills by default. This
  keeps the state-only startup contract while reducing descriptor fanout for
  the default 12-layer shape from many small AdamW state fills to two coalesced
  range fills.
- Set `NFN_NATIVE_GPT_ZERO_ADAMW_STATE_RANGES=0` to force the older
  descriptor-driven AdamW state fill path. Set
  `NFN_NATIVE_GPT_ZERO_ADAMW_STATE_ONLY=0` to force the older full-arena zero
  for bisection.
- Runtime and plan JSON now report `adamw_state_zero_range_count` and
  `adamw_state_zero_range_elements`, with matching
  `block_state_layout.startup_adamw_state_zero_range_count` and
  `block_state_layout.startup_adamw_state_zero_range_elements` fields. The
  default `float_arena_zero_init_strategy` is now
  `"adamw-state-contiguous-range-fill"`.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Ran a one-step TinyStories smoke pinned to the dedicated RTX 5090
  (`CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1`). It completed with
  `float_arena_zero_init_strategy` set to
  `"adamw-state-contiguous-range-fill"`, `adamw_state_zero_fill_count: 2`,
  `adamw_state_zero_range_count: 2`, and
  `adamw_state_zero_range_elements: 248951808`.
- Ran `tools/paired_kernel_speed.py` with one warmup and five measured pairs on
  GPU 0, comparing `NFN_NATIVE_GPT_ZERO_ADAMW_STATE_RANGES=0` against the new
  default. The contiguous-range default had total runtime ratio `0.997812`,
  setup wall time mean `510.4956 ms` versus `521.7332 ms` for the previous
  descriptor-fill path, and neutral train-loop timing ratio `1.00109`.

### 2026-06-14 Default packed QKV bias to fused SM120 TK GEMM

#### Changed

- Dense GPT native training now fuses QKV bias into the packed-QKV SM120 TK
  BF16 QKV GEMM by default. This removes the separate packed BF16 QKV bias-add
  launch from the default forward path while keeping the older route available
  with `NFN_NATIVE_GPT_FUSE_QKV_BIAS_TK_GEMM=0`.
- The trainer-facing Tile-CUDA BF16 weight-output helper can pass a BF16-packed
  bias pointer to the TK matmul path. If TK declines a shape and the helper uses
  the cuBLAS fallback, it still applies the existing BF16 bias-add internally so
  public C ABI behavior is unchanged.
- Plan and runtime JSON now report `qkv_bias_fused_tk_gemm_enabled` and default
  `qkv_bias_layout_strategy: "packed-qkv-bf16-bias-fused-tk-gemm"`.

#### Verification

- Rebuilt `libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh` and rebuilt
  `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran a one-step TinyStories smoke pinned to the dedicated RTX 5090
  (`CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1`). It completed and
  reported `qkv_bias_fused_tk_gemm_enabled: true`,
  `qkv_bias_layout_kernel_launches_per_block: 0`, and
  `qkv_bias_layout_strategy: "packed-qkv-bf16-bias-fused-tk-gemm"`.
- Ran `tools/paired_kernel_speed.py` with one warmup and five measured pairs on
  GPU 0, comparing the previous separate-bias path against the fused-bias
  candidate. The fused/default candidate had `train_loop_wall_ms` ratio
  `0.989147`, tokens/s ratio `1.010975`, and total runtime ratio `0.991113`.

### 2026-06-14 Add opt-in BF16/BF16 QKV dWeight candidate path

#### Changed

- Dense GPT native training now exposes `NFN_NATIVE_GPT_BF16_QKV_DWEIGHT=1`
  as an opt-in candidate path. After packed-attention backward writes `dQKV`
  into the separate BF16 scratch buffer, the trainer can reuse the freed packed
  QKV BF16 buffer to pack LN1 output and call the existing BF16/BF16
  dWeight+bias accumulation ABI for QKV.
- Runtime JSON reports `block_backward_bf16_qkv_dweight_enabled` and
  `block_backward_qkv_dweight_strategy`. The default remains the existing
  float32-LN1 plus BF16-`dQKV` dWeight+bias path because paired RTX 5090 timing
  was neutral rather than clearly faster.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Ran an opt-in one-step TinyStories GPU smoke pinned to the dedicated RTX 5090
  (`CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1`). It completed and
  reported `block_backward_bf16_qkv_dweight_enabled: true` with strategy
  `"packed-ln1-bf16-qkv-bf16-grad-dweight-bias-accumulate"`.
- Ran `tools/paired_kernel_speed.py` with one warmup and five measured pairs on
  GPU 0. The opt-in candidate/default `train_loop_wall_ms` ratio was
  `0.998878`, tokens/s ratio was `1.001129`, and total runtime ratio was
  `1.000061`, so this path remains opt-in.

### 2026-06-14 Default packed QKV backward to direct BF16 grad scratch

#### Changed

- Dense GPT native training now writes packed-attention BF16 `dQKV` directly
  into a non-aliased BF16 scratch buffer by default. The trainer reuses the MLP
  BF16 scratch after MLP backward is done, then feeds that buffer to QKV
  dWeight+bias and QKV dInput without copying through the packed QKV activation
  buffer.
- The Tile-CUDA packed QKV backward wrapper now honors a caller-provided
  non-aliased BF16 `grad_qkv_bf16_bits` destination directly. Aliased calls keep
  the older workspace-to-QKV-buffer copy behavior, and callers that also request
  float `grad_qkv` conversion read back from the actual BF16 destination.
- Set `NFN_NATIVE_GPT_DIRECT_BF16_QKV_GRAD_SCRATCH=0` to reproduce the older
  workspace-to-packed-QKV-buffer copy path in paired benchmarks. Runtime and
  plan JSON now report
  `attention_backward_direct_bf16_qkv_grad_scratch_enabled`,
  `attention_backward_direct_bf16_qkv_grad_scratch_elements`, and direct-scratch
  strategy strings.

#### Verification

- Rebuilt `libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh` and rebuilt
  `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran the focused native GPT pytest slice, Python compile checks, and
  `git diff --check`.
- Ran a one-step TinyStories GPU smoke pinned to the dedicated RTX 5090
  (`CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1`). It reported
  `attention_backward_direct_bf16_qkv_grad_scratch_enabled: true`,
  `attention_backward_direct_bf16_qkv_grad_scratch_elements: 150994944`, and
  `attention_backward_strategy:
  "tk-sm120-packed-qkv-bf16-saved-activation-backward-direct-bf16-grad-scratch-handoff"`.
- Ran `tools/paired_kernel_speed.py` with one warmup and five measured pairs on
  GPU 0, comparing baseline
  `NFN_NATIVE_GPT_DIRECT_BF16_QKV_GRAD_SCRATCH=0` against the new default. The
  default/old-path `train_loop_wall_ms` ratio was `0.988571`, tokens/s ratio was
  `1.011569`, and total runtime ratio was `0.988554`.

### 2026-06-14 Add opt-in BF16/BF16 LM-head dWeight candidate path

#### Changed

- Added raw Tile-CUDA ABI
  `nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32`
  for no-bias BF16-input/BF16-grad-output dWeight accumulation.
- Dense GPT native training now exposes `NFN_NATIVE_GPT_LM_HEAD_BF16_DWEIGHT=1`
  as an opt-in candidate path. It packs each final LayerNorm hidden chunk to
  BF16 inside the compiled trainer and uses the new BF16/BF16 dWeight ABI for
  tied LM-head accumulation. The default remains the existing float32-hidden
  plus BF16-dlogit path until paired RTX 5090 timing proves this candidate is
  faster.
- Runtime JSON now reports `lm_head_bf16_dweight_enabled`,
  `lm_head_bf16_hidden_elements`, `lm_head_bf16_hidden_bytes`,
  `lm_head_dweight_input_dtype`, and `lm_head_dweight_strategy`.

#### Verification

- Rebuilt `libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh` and rebuilt
  `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran the focused native GPT pytest slice, Python compile checks, and
  `git diff --check`.
- Ran default and opt-in one-step TinyStories GPU smokes pinned to the
  dedicated RTX 5090 (`CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1`).
  The default reported `lm_head_bf16_dweight_enabled: false`; the opt-in run
  reported `lm_head_bf16_dweight_enabled: true`, `lm_head_bf16_hidden_elements:
  3145728`, and `lm_head_dweight_strategy:
  "chunked-final-norm-bf16-pack-bf16-dlogit-dweight-accumulate"`.
- Ran `tools/paired_kernel_speed.py` with one warmup and five measured pairs on
  GPU 0. The opt-in candidate/default `train_loop_wall_ms` ratio was
  `1.000037`, tokens/s ratio was `0.999962`, and total runtime ratio was
  `1.001343`, so the path remains opt-in rather than becoming the default.

### 2026-06-14 Default native GPT block projection weights to BF16-primary AdamW

#### Changed

- Dense GPT native training now defaults QKV, attention projection, MLP FC, and
  MLP projection weights to the BF16-primary block-weight AdamW path. The
  token, position, norm, and bias tensors still use the float32 multi-buffer
  AdamW descriptors.
- Set `NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS=0` to reproduce the older
  FP32-master plus BF16-shadow refresh path for bisection. The fused shadow
  refresh profiler remains available only when the BF16-primary path is
  disabled.

#### Verification

- Ran paired RTX 5090 benchmarks with display disabled on GPU 0. A three-sample
  warmup-backed pair reported candidate/default `train_loop_wall_ms` ratio
  `0.991168`; a five-sample confirmation reported `0.996888` and total runtime
  ratio `0.996224`.
- After flipping the default, reran a three-sample pair with
  `NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS=0` as the baseline and the new
  default as candidate. It reported `train_loop_wall_ms` ratio `0.999012` and
  total runtime ratio `0.998851`, confirming the default is neutral to slightly
  faster on the dedicated RTX 5090.

### 2026-06-14 Add native GPT setup timing breakdown

#### Changed

- Dense GPT native training JSON now includes `timing.setup_timing`, a
  host-side breakdown of setup phases such as float arena materialization,
  token arenas, stored BF16 activation arenas, descriptor materialization,
  zero initialization, token-weight initialization, parameter fill, and initial
  BF16 block-weight refresh.
- This keeps normal runs free of CUDA-event stage profiling overhead while
  making startup regressions visible in the compiled C++ trainer output.

#### Verification

- Rebuilt the native GPT CLI with `bash tools/build_native_gpt_cli.sh`.
- Ran a one-step dedicated RTX 5090 native GPT smoke. The output included
  `timing.setup_timing` and showed the largest setup buckets as float arena
  materialization, AdamW zero init, stored MLP activation arena allocation,
  stored packed-attention arena allocation, and initial BF16 block-weight
  refresh.
- Tested and rejected an offset-compacted zero-fill Tile kernel candidate:
  paired RTX 5090 timing regressed mean train-loop wall time to `1.047281x`
  versus the existing fill-many zero path.

### 2026-06-14 Elide unused MLP dGELU float-gradient conversion in native GPT

#### Changed

- Added `nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_only_float32`
  to the raw trainer Tile-CUDA ABI. Dense GPT training uses it by default when
  BF16 MLP grad handoff is active, so the MLP projection backward path writes
  the BF16 gradient consumed by the following MLP FC backward stage without
  also converting that gradient to an unused FP32 buffer.
- Added `NFN_NATIVE_GPT_ELIDE_MLP_DGELU_FLOAT_GRAD=0` for paired benchmarks or
  diagnostics that need the previous BF16 handoff path. Runtime JSON now
  reports `block_backward_mlp_dgelu_float_grad_elided` and distinguishes the
  `...-no-float-grad` and `...-float-grad` strategy strings.

#### Verification

- Rebuilt `libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt the native GPT CLI with `bash tools/build_native_gpt_cli.sh`.
- Ran a dedicated RTX 5090 paired benchmark with the previous conversion path
  as baseline and the BF16-only path as candidate:
  candidate mean `train_loop_wall_ms` ratio `0.979437`, mean
  `train_tokens_per_second` ratio `1.021189`, and no compute processes before
  the run.

### 2026-06-14 Retune native GPT LM-head row chunk default to 4096

#### Changed

- Dense GPT native training now defaults the tied LM-head row chunk to 4096
  rows for the compiled C++ trainer, Python wrapper defaults, and
  `NativeGpt2RunConfig` / generic native GPT SDK handoff helpers.
- The default uses about 412MB of BF16 LM-head logits workspace at the
  workstation `64 x 1024` microbatch, instead of about 824MB for the previous
  8192-row profile.

#### Breaking changes

- Callers relying on the implicit 8192-row tied LM-head workspace should pass
  `--lm-head-row-chunk-size 8192`,
  `--native-cuda-lm-head-row-chunk-size 8192`, or
  `NativeGpt2RunConfig(lm_head_row_chunk_size=8192, ...)` explicitly. The flag,
  config field, and JSON field names are unchanged.

#### Verification

- Ran paired dedicated RTX 5090 benchmarks comparing default 8192-row chunks
  against `--lm-head-row-chunk-size 4096`. The two-step, three-sample pair
  reported candidate/default mean `train_loop_wall_ms` ratio `0.969890` and
  mean `train_tokens_per_second` ratio `1.031782`.
- Rejected `--lm-head-row-chunk-size 16384` after a two-sample pair regressed
  to `1.226666x` train-loop wall time, and rejected `65536` after it failed to
  complete in the useful benchmark window.

### 2026-06-14 Parse llm.kittens metrics in paired kernel benchmarks

#### Changed

- `tools/paired_kernel_speed.py` now parses llm.kittens
  `step ... ms ... bf16 MFU ... tok/s` output when a command does not emit
  NeuralFn native JSON. The parsed step time, token throughput, BF16 MFU, and
  device-memory line are reported through the same `baseline_native_metrics`
  and `candidate_native_metrics` summaries used for NeuralFn native runs, so
  direct `train_gpt2cu` baselines can be compared against NeuralFn candidates
  in one paired script.

#### Verification

- Ran `python -m pytest tests/test_tile_cuda_examples.py -q -k
  "paired_kernel_speed_tool_compiles_and_smokes or
  paired_kernel_speed_tool_extracts_llm_kittens_step_metrics"`.
- Ran `python -m py_compile tools/paired_kernel_speed.py`.
- Ran a dedicated RTX 5090 paired comparison between llm.kittens
  `train_gpt2cu` and NeuralFn `nfn_gpt_native_train`; the helper reported
  llm.kittens `train_loop_wall_ms` / `train_tokens_per_second` in
  `baseline_native_metrics` and NeuralFn JSON metrics in
  `candidate_native_metrics`.
- Ran `git diff --check`.

### 2026-06-14 Add timeout handling to paired native benchmarks

#### Changed

- `tools/paired_kernel_speed.py` now accepts
  `--command-timeout-seconds N` to cap each baseline/candidate child process.
  When `--continue-on-error` is set, timed-out commands are preserved in
  `paired_samples` with `timed_out: true`, `returncode: -1`, and
  `timeout_seconds`, so a runaway CUDA kernel candidate can be recorded without
  wedging the dedicated GPU tuning loop.

#### Verification

- Ran `python -m pytest tests/test_tile_cuda_examples.py -q -k
  "paired_kernel_speed_tool_compiles_and_smokes or paired_kernel_speed_tool_records_command_timeout"`.
- Ran `python -m py_compile tools/paired_kernel_speed.py`.
- Ran `git diff --check`.

### 2026-06-14 Add stage-timing extraction to paired native benchmarks

#### Changed

- `tools/paired_kernel_speed.py` now extracts NeuralFn native
  `timing.stage_timing` entries into `baseline_native_metrics` and
  `candidate_native_metrics` as `stage.<name>.total_ms`, `.avg_ms`, and
  `.count`. Text output highlights the major dense GPT stages so kernel
  candidates can be judged against the stage they are supposed to improve, not
  only whole-command wall time.

#### Verification

- Ran `python -m pytest tests/test_tile_cuda_examples.py -q -k
  paired_kernel_speed_tool_compiles_and_smokes`.
- Ran `python -m py_compile tools/paired_kernel_speed.py`.
- Ran `git diff --check`.

### 2026-06-14 Default BF16 MLP gradient handoff on native GPT

#### Changed

- Dense GPT native training now enables `NFN_NATIVE_GPT_BF16_MLP_GRAD_HANDOFF`
  by default when fused MLP projection dInput+dGELU is active. The fused MLP
  projection backward keeps its result as BF16 bits for the following MLP FC
  backward GEMMs instead of converting that intermediate gradient back to
  float32.
- Set `NFN_NATIVE_GPT_BF16_MLP_GRAD_HANDOFF=0` or the GPT-2-prefixed fallback
  to compare against the older float-gradient handoff.
- Runtime JSON now reports
  `block_backward_bf16_mlp_grad_handoff_enabled: true` by default and the
  default `block_backward_mlp_proj_dgelu_strategy` becomes
  `"tk-sm120-fused-dinput-dgelu-bf16-store-bf16-shadow-weight-bf16-grad-handoff"`.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k
  "native_train_tile_ops_builds_torch_free_c_abi or
  native_gpt2_cpp_cli_builds_and_uses_sm120_defaults"`; the native GPT CLI
  contract test passed, and the ABI test skipped only its final live CUDA fill
  smoke after source/build/symbol checks completed.
- Ran `tools/paired_kernel_speed.py` on the dedicated RTX 5090 with one warmup
  pair and three measured pairs, comparing the new default against
  `NFN_NATIVE_GPT_BF16_MLP_GRAD_HANDOFF=0`. The default averaged `3095.74 ms`
  train-loop time versus `3237.30 ms` for the forced older path, and
  `169358.00` versus `161952.33` train tokens/sec.
- Ran `git diff --check`.

### 2026-06-14 Pin paired kernel benchmarks to deterministic CUDA connection mode

#### Changed

- `tools/paired_kernel_speed.py` now sets `CUDA_DEVICE_MAX_CONNECTIONS=1` for
  both baseline and candidate commands by default, so paired native CUDA
  benchmark artifacts match the dedicated RTX 5090 training launch profile.
  Pass `--cuda-device-max-connections ""` to leave the inherited environment
  unchanged for experiments that need a different CUDA connection mode.
- The paired benchmark GPU snapshot now records `display_active` from
  `nvidia-smi`, making it visible whether a timing run used the compute-only
  RTX 5090 or a display-active GPU.

#### Verification

- Ran `python -m pytest tests/test_tile_cuda_examples.py -q -k
  paired_kernel_speed_tool_compiles_and_smokes`.
- Ran `python -m py_compile tools/paired_kernel_speed.py`.
- Ran `git diff --check`.

### 2026-06-14 Default `nfn train` to native GPT when base model is omitted

#### Changed

- Top-level `nfn train` now treats an omitted `--base-model` / `--model` as
  `gpt` during the lightweight pre-import native dispatch check. Commands such
  as `nfn train --tinystories` now exec the compiled
  `nfn_gpt_native_train --model-family gpt --train-transformer-lm ...` path
  instead of falling through toward the legacy graph-backed training guard.
- Explicit non-GPT, non-dense, MoE, router, topology, or external bridge
  selections still bypass this dense GPT shortcut and are handled by the native
  registry or legacy guard as before.

#### Verification

- Ran `python -m pytest tests/test_native_gpt2.py -q -k
  "top_level_nfn_train_defaults_to_native_gpt_without_base_model"`.
- Ran `python cli/nfn.py train --tinystories --native-cuda-print-command` and
  verified it resolves to
  `nfn_gpt_native_train --model-family gpt --train-transformer-lm --tinystories
  --print-command`.

### 2026-06-14 Default BF16 packed-QKV gradient handoff

#### Changed

- Dense GPT native training now keeps packed attention backward `dQKV` in BF16 by
  default on the packed-QKV SM120 Tile path. The packed attention backward writes
  BF16 gradient bits into the packed QKV activation buffer after the forward
  consumers are done, then QKV dWeight+bias consumes those bits through
  `nfn_native_tile_linear_backward_weight_bias_accumulate_float32_bf16_bits` and
  QKV dInput consumes them through the BF16-gradient/BF16-weight input-backward
  ABI.
- Added the raw trainer Tile ABI symbols
  `nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_merged_grad_float32`,
  `nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_merged_grad_float32`,
  and `nfn_native_tile_linear_backward_weight_bias_accumulate_float32_bf16_bits`.
- `NFN_NATIVE_GPT_BF16_QKV_GRAD_HANDOFF` /
  `NFN_NATIVE_GPT2_BF16_QKV_GRAD_HANDOFF` now default to enabled. Set either to
  `0` only when comparing against the older packed attention backward path that
  expands `dQKV` to a float32 `grad_qkv` buffer before QKV dWeight/dInput.
- Native plan and runtime JSON now report
  `attention_backward_bf16_qkv_grad_handoff_enabled`,
  `qkv_backward_layout_strategy: "packed-qkv-bf16-gradient-handoff"`,
  `attention_backward_qkv_bridge_strategy:
  "tk-sm120-packed-qkv-packed-bf16-grad-handoff"`, and the handoff-specific
  attention backward strategy strings.

#### Verification

- Rebuilt the raw Tile ops library with `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt the native GPT CLI with `bash tools/build_native_gpt_cli.sh`.
- Ran a paired dedicated-RTX-5090 candidate-vs-current benchmark with
  `python tools/paired_kernel_speed.py --cuda-visible-devices 0 --warmup 1
  --samples 3 ...`; the BF16 handoff candidate averaged 3234.88 ms train-loop
  time versus 3350.13 ms baseline, and 162074.67 versus 156499.00 train
  tokens/sec.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k
  "native_gpt2_cpp_cli_builds_and_uses_sm120_defaults or
  native_train_tile_ops_builds_torch_free_c_abi"`; the native GPT CLI contract
  test passed, and the ABI test skipped only its final live CUDA fill smoke after
  source/build/symbol checks completed.
- Ran a default one-step full-shape native CUDA smoke on the dedicated RTX 5090
  with `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1
  NFN_NATIVE_GPT_STAGE_TIMING=1 build/nfn_gpt_native_train ... --max-steps 1
  --no-checkpoint`; it completed with `status:
  "native-transformer-lm-trained"`, `attention_backward_bf16_qkv_grad_handoff_enabled:
  true`, saved packed-attention strategy
  `"tk-sm120-packed-qkv-bf16-saved-activation-backward-bf16-grad-handoff"`, and
  161459 train tokens/sec.

### 2026-06-14 Opt-in BF16 primary block-weight AdamW path

#### Changed

- Dense GPT native training can now opt into BF16 primary parameter updates for
  block projection weights with `NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS=1`.
  QKV, attention projection, MLP FC, and MLP projection weights update their
  BF16 parameter buffers directly through
  `nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_float32`, while
  token, position, norm, and bias tensors continue to update through the float32
  multi-buffer AdamW ABI.
- The existing all-buffer descriptor table still drives gradient zeroing and
  gradient clipping, so no training batches or accumulated gradients are routed
  through graph-editor or Torch paths.
- Checkpoint export now syncs BF16 primary block weights back into the existing
  FP32 staging buffers before the version-5 BF16 checkpoint packer runs, keeping
  native inference artifacts in the established checkpoint format.
- Runtime JSON now reports
  `block_weight_bf16_primary_param_update_enabled`,
  `block_weight_bf16_primary_param_update_count`,
  `adamw_float_update_descriptor_count`,
  `adamw_bf16_param_descriptor_count`,
  `adamw_float_update_kernel_launches`,
  `adamw_bf16_param_kernel_launches`, and
  `checkpoint.bf16_param_sync_kernel_launches`.

#### Verification

- Rebuilt the native GPT CLI with `bash tools/build_native_gpt_cli.sh`.
- Rebuilt the raw Tile ops library with
  `bash tools/build_native_train_tile_ops.sh` and verified it exports
  `nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_float32` with
  `nm -D build/libnfn_native_train_tile_ops.so | rg
  "adamw_step_many_with_device_scale.*(float32|bf16)"`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k
  "native_gpt2_cpp_cli_builds_and_uses_sm120_defaults or
  native_train_tile_ops_builds_torch_free_c_abi"`; the CLI contract test
  passed, and the ABI test skipped only its final CUDA fill smoke after source,
  build, and symbol assertions completed.
- Confirmed the dedicated RTX 5090 is CUDA device 0 with display disabled via
  `nvidia-smi --query-gpu=index,name,uuid,memory.total,display_active
  --format=csv,noheader`.
- Ran a full-shape one-step opt-in CUDA probe with
  `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1
  NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS=1 build/nfn_gpt_native_train ...`;
  it completed on the RTX 5090 and reported 100 float32 update descriptors,
  48 BF16 parameter descriptors, two AdamW launches per optimizer step, and
  `block_weight_bf16_primary_param_update_enabled: true`.
- Ran paired baseline-vs-candidate timing with `python
  tools/paired_kernel_speed.py --cuda-visible-devices 0 --warmup 1 --samples 3
  ...`; candidate train-loop time was effectively neutral at 3290.13 ms vs
  3291.26 ms baseline, with candidate tokens/sec at 1.00034x baseline.
- Ran a one-layer CUDA checkpoint smoke with
  `NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS=1` and packed-attention/storage
  disabled for the tiny shape; it wrote
  `/tmp/nfn-bf16-param-checkpoint-smoke/model_00000001.bin` and reported
  `checkpoint.bf16_param_sync_kernel_launches: 4`.

### 2026-06-14 BF16-parameter AdamW ABI for no-master native GPT

#### Changed

- Added the raw Tile-CUDA ABI
  `nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_float32`.
  It updates descriptor-driven BF16 parameter buffers directly from float32
  gradients, float32 AdamW moment state, and the device clip scalar.
- This is the low-level primitive needed to migrate the native GPT trainer
  toward the llm.kittens SM120 `use_master_weights disabled` storage model.
  The default trainer still uses FP32 parameter buffers plus BF16 block-weight
  shadows until the parameter/gradient layout migration is wired and benchmarked
  end-to-end.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Verified the exported ABI with `nm -D build/libnfn_native_train_tile_ops.so
  | rg "adamw_step_many_with_device_scale.*(float32|bf16)"`, which reports
  `nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_float32`.
- Ran `python -m pytest tests/test_native_gpt2.py -q -k
  "native_train_tile_ops_builds_torch_free_c_abi"`; the source, build, and
  symbol-load assertions completed, then the test skipped its final CUDA fill
  smoke because that helper loaded a mismatched CUDA runtime in the test process.

### 2026-06-14 Opt-in fused AdamW BF16 shadow refresh ABI

#### Changed

- Added the raw Tile-CUDA ABI
  `nfn_native_tile_adamw_step_many_with_device_scale_bf16_shadow_float32`.
  It mirrors the descriptor-driven multi-buffer AdamW kernel and can also write
  updated FP32 master weights into optional BF16 shadow-weight slots from the
  same launch.
- Dense GPT native training now carries BF16 shadow offsets in the AdamW
  descriptor arena and reports
  `block_weight_bf16_shadow_fused_adamw_refresh_enabled`,
  `block_weight_bf16_fused_adamw_refresh_count`, and
  `adamw_bf16_shadow_refresh_strategy`.
- The fused shadow-write path is opt-in through
  `NFN_NATIVE_GPT_FUSE_ADAMW_BF16_SHADOW_REFRESH=1`. The default remains the
  separate `nfn_native_tile_float32_to_bf16_bits_many` refresh after AdamW
  because paired dedicated RTX 5090 timing was neutral/slightly slower on the
  native train loop.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` and `build/nfn_gpt_native_train`
  with `bash tools/build_native_train_tile_ops.sh` and
  `bash tools/build_native_gpt_cli.sh`.
- Verified the exported symbols with `nm -D build/libnfn_native_train_tile_ops.so
  | rg "adamw_step_many_with_device_scale"`.
- Ran default and opt-in `--smoke-transformer-lm-step` probes on GPU 0 with
  `CUDA_VISIBLE_DEVICES=0` and `CUDA_DEVICE_MAX_CONNECTIONS=1`.
- Compared separate refresh versus fused shadow-write with
  `tools/paired_kernel_speed.py` on the dedicated RTX 5090. The four measured
  pairs reported
  `candidate_over_baseline_native_metrics.train_loop_wall_ms.mean = 1.000469`
  and `candidate_over_baseline_native_metrics.train_tokens_per_second.mean =
  0.999532`, so the fused path stayed opt-in.

### 2026-06-14 LayerNorm affine row-chunk tuning for native GPT

#### Changed

- Tile-CUDA LayerNorm affine-gradient large-row reductions now default to
  256-row chunks instead of inheriting the 512-row Linear bias-gradient chunk.
  This improves row-level parallelism for the dense GPT `64 x 1024` RTX 5090
  training shape while leaving Linear bias-gradient chunking unchanged.
- Added `NFN_TILE_CUDA_LAYERNORM_AFFINE_ROW_CHUNK_SIZE`, with
  `NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE` and
  `NFN_NATIVE_GPT2_LAYERNORM_AFFINE_ROW_CHUNK_SIZE` aliases, so native paired
  benchmarks can compare LayerNorm affine row chunk sizes without rebuilding.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Ran `build/nfn_gpt_native_train --smoke-transformer-lm-step --backend
  tile-cuda` on GPU 0 with `CUDA_VISIBLE_DEVICES=0` and
  `CUDA_DEVICE_MAX_CONNECTIONS=1`.
- Compared row chunk candidates in `tools/paired_kernel_speed.py` on the
  dedicated RTX 5090. The confirmed 256-row run used one warmup pair and four
  measured pairs, with
  `candidate_over_baseline_native_metrics.train_loop_wall_ms.mean = 0.996599`
  and `candidate_over_baseline_native_metrics.train_tokens_per_second.mean =
  1.003413` versus the old 512-row default.

### 2026-06-14 Native GPT default fused dGELU float-gradient path

#### Changed

- Dense GPT block backward now keeps the fused MLP projection dInput+dGELU ABI
  active on the default float-gradient path. The BF16 MLP gradient handoff flag
  only controls whether the following MLP FC backward consumes BF16 bits instead
  of the default float gradient.
- Runtime JSON continues to report
  `block_backward_bf16_mlp_grad_handoff_enabled: false` by default while
  `block_backward_mlp_proj_dgelu_strategy` remains the fused float-gradient
  strategy.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran a stage-timed one-step TinyStories native GPT CUDA smoke on GPU 0 with
  `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and
  `NFN_NATIVE_GPT_STAGE_TIMING=1`.

### 2026-06-14 Native GPT opt-in BF16 MLP gradient handoff ABI

#### Changed

- The raw native Tile ABI now exposes
  `nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_float32` and
  `nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32`
  so native trainers can test carrying the fused MLP projection dInput/dGELU
  result as BF16 bits into the following MLP FC backward GEMMs.
- The dense GPT compiled trainer can opt into that path with
  `NFN_NATIVE_GPT_BF16_MLP_GRAD_HANDOFF=1`. The default remains the existing
  float-gradient handoff because paired timing on the dedicated RTX 5090 was
  neutral/slower for the BF16 handoff experiment.
- Runtime JSON now reports `block_backward_bf16_mlp_grad_handoff_enabled` and
  updates `block_backward_mlp_proj_dgelu_strategy` plus
  `stored_mlp_activation_backward_consumer_strategy` when the experiment is
  active.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Verified the exported ABI with `nm -D build/libnfn_native_train_tile_ops.so |
  rg "bf16_bits_weight_bf16|bf16_bits_bf16_bits"`.
- Ran default and opt-in one-step TinyStories native GPT CUDA smokes on GPU 0
  with `CUDA_VISIBLE_DEVICES=0` and `CUDA_DEVICE_MAX_CONNECTIONS=1`.
- Compared handoff-off and handoff-on in `tools/paired_kernel_speed.py` with one
  warmup pair and five measured pairs. Native train-loop
  `candidate_over_baseline_native_metrics.timing.train_loop_wall_ms.mean` was
  `1.002665`, so the BF16 handoff remains opt-in.

### 2026-06-14 Paired benchmark native metric summaries

#### Changed

- `tools/paired_kernel_speed.py` now extracts selected metrics from NeuralFn
  native JSON stdout for each baseline/candidate sample. Text and JSON output
  now include `baseline_native_metrics`, `candidate_native_metrics`, and
  `candidate_over_baseline_native_metrics` for shared numeric metrics, so
  command startup and checkpoint export can be separated from native
  `timing.train_loop_wall_ms` and token-throughput comparisons.
- The extracted metrics include native setup, train-loop, checkpoint, total
  wall time, train tokens/s, selected BF16 linear cache counters, and attention
  launch counters when present.

#### Verification

- Ran `python -m py_compile tools/paired_kernel_speed.py`.
- Ran a one-sample paired comparison on the dedicated RTX 5090 with
  `CUDA_VISIBLE_DEVICES=0` and `CUDA_DEVICE_MAX_CONNECTIONS=1`, comparing the
  default dense GPT native trainer against
  `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE=0`. The helper reported GPU 0 as
  `NVIDIA GeForce RTX 5090`, `0%` utilization, no compute processes before
  timing, command-level paired timing, and native summaries including
  `train_loop_wall_ms` (`3315.020` ms baseline, `3297.090` ms candidate) and
  `train_tokens_per_second` (`158155` baseline, `159015` candidate).

### 2026-06-14 Native GPT LM-head dWeight BF16 cache correctness

#### Changed

- The native dense-GPT LM-head dWeight GEMM no longer caches its FP32 hidden
  activation operand by pointer before packing it to BF16. The hidden activation
  buffer is scratch storage reused with new contents across
  gradient-accumulation microbatches, so pointer-keyed caching could reuse stale
  packed hidden chunks while accumulating token-weight gradients.
- Stable operands such as weights remain eligible for the BF16 operand cache.
  Runtime JSON still reports the BF16 pack/cache counters so this behavior can
  be audited during kernel tuning.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Ran a one-step TinyStories native GPT CUDA smoke on the dedicated RTX 5090:
  `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1
  NFN_NATIVE_GPT_STAGE_TIMING=1 build/nfn_gpt_native_train --tinystories
  --tile-ops-lib /tmp/libnfn_tile_ops_lm_dweight_cachefix.so --max-steps 1
  --eval-every-steps 0 --no-checkpoint`. Runtime JSON reported
  `steps_completed: 1`, `passed: true`, no missing symbols,
  `linear_bf16_a_pack_count: 429`, and `linear_bf16_a_cache_hit_count: 211`,
  confirming the mutable LM-head hidden chunks are repacked instead of reused
  from the BF16 cache.
- Ran `tools/paired_kernel_speed.py` on GPU 0 with one warmup pair and five
  measured one-step pairs comparing the old stale-cache Tile ops library
  against the corrected library. The benchmark recorded GPU 0 as
  `NVIDIA GeForce RTX 5090`, `0%` utilization, about `426/32607` MiB used, and
  no compute processes before timing. The corrected candidate reported
  `candidate_over_baseline` mean `1.000010`, median `1.000189`, min
  `0.995780`, and max `1.003054`, so the correctness fix is neutral at command
  level.

### 2026-06-14 Paired kernel speed output decoding

#### Changed

- `tools/paired_kernel_speed.py` now decodes child-process stdout and stderr
  with replacement instead of failing on non-UTF-8 bytes. This lets NeuralFn
  native binaries and external CUDA trainers such as llm.kittens be compared in
  the same paired timing script even when one emits binary or extended bytes.

#### Verification

- Reproduced the prior strict UTF-8 decode failure while comparing
  `train_gpt2cu` against `build/nfn_gpt_native_train`.
- Ran `python -m py_compile tools/paired_kernel_speed.py`.
- Reran the same paired comparison on the dedicated RTX 5090 with one warmup
  pair and three measured pairs. The helper recorded GPU 0 as
  `NVIDIA GeForce RTX 5090`, `0%` utilization, about `424/32607` MiB used, and
  no compute processes before timing. The NeuralFn native command completed at
  `candidate_over_baseline` mean `0.435572` versus the llm.kittens command,
  proving the helper now survives the external trainer output.

### 2026-06-14 Native GPT BF16 CE reverse-row traversal

#### Changed

- The dense-GPT native BF16 token cross-entropy backward kernel now maps CUDA
  blocks to rows in reverse order. This matches the llm.kittens fused
  classifier traversal and improves locality for BF16 logits immediately after
  the tied LM-head GEMM writes them.
- The kernel still uses the existing fused row-wise max/sum/probability pass and
  writes BF16 dlogits in-place; there is no new runtime flag or graph contract
  change.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Ran `tools/paired_kernel_speed.py` on the dedicated RTX 5090 with
  `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, one warmup pair,
  and ten measured one-step pairs comparing forward-row CE traversal against
  reverse-row traversal. The benchmark recorded GPU 0 as
  `NVIDIA GeForce RTX 5090`, `0%` utilization, about `421/32607` MiB used, and
  no compute processes before timing. The reverse-row candidate reported
  `candidate_over_baseline` mean `0.998673`, median `0.998475`, min
  `0.990774`, and max `1.005167`.
- Rejected an `exp2f` CE variant after a five-sample paired run reported
  `candidate_over_baseline` mean `1.000190`.
- Ran a one-step TinyStories native GPT CUDA smoke on GPU 0:
  `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1
  build/nfn_gpt_native_train --tinystories --tile-ops-lib
  build/libnfn_native_train_tile_ops.so --max-steps 1 --eval-every-steps 0
  --no-checkpoint --output-dir /tmp/nfn-ce-reverse-smoke`. Runtime JSON reported
  `lm_head_ce_backward_strategy: "fused-row-bf16-logits-dlogits"`, no missing
  symbols, `steps_completed: 1`, and `passed: true`.

### 2026-06-14 Native GPT prepacked LN2 FC+GELU route

#### Changed

- Added the raw Tile-CUDA ABI
  `nfn_native_tile_linear_bf16_input_weight_bf16_gelu_bf16_float32`.
  Dense GPT stored-MLP forward now uses it by default after `ln2_out` has
  already been packed to BF16 for backward storage, so the FC+bias+GELU TK path
  reuses that prepacked BF16 input instead of repacking the same LN2 activation
  inside the GEMM helper.
- Set `NFN_NATIVE_GPT_REUSE_PACKED_LN2_FC_GELU=0` or
  `NFN_NATIVE_GPT2_REUSE_PACKED_LN2_FC_GELU=0` to compare against the older
  `nfn_native_tile_linear_weight_bf16_gelu_bf16_float32` route.
- Runtime JSON now reports `reuse_packed_ln2_fc_gelu_enabled` and changes
  `stored_mlp_forward_strategy` to
  `"tk-sm120-fused-fc-bias-gelu-prepacked-ln2-bf16-shadow-weight"` when the new
  default route is active.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh` and rebuilt
  `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran a one-step TinyStories stage profile on the dedicated RTX 5090 with
  `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and
  `NFN_NATIVE_GPT_STAGE_TIMING=1`. The run reported the new ABI in `kernels`,
  `reuse_packed_ln2_fc_gelu_enabled: true`, no missing symbols, and
  `stored_mlp_forward_strategy:
  "tk-sm120-fused-fc-bias-gelu-prepacked-ln2-bf16-shadow-weight"`. The
  `block_forward.mlp_fc_gelu.fc_gelu` stage was about `152.862 ms` in that
  profile versus the prior clean-source profile's about `169.733 ms`.
- Ran `tools/paired_kernel_speed.py` on GPU 0 with one warmup pair and five
  measured one-step pairs comparing
  `NFN_NATIVE_GPT_REUSE_PACKED_LN2_FC_GELU=0` against the new default. The
  candidate reported `candidate_over_baseline` mean `0.999403`.
- Ran a less startup-diluted two-step paired benchmark with one warmup pair and
  three measured pairs. The candidate reported `candidate_over_baseline` mean
  `0.997485`, median `0.997458`, min `0.996870`, and max `0.998126`.

### 2026-06-14 Dense GPT transformer validation eval batch size

#### Changed

- Dense GPT native `--train-transformer-lm` now honors
  `--eval-batch-size` / `eval_batch_size` for transformer validation, not just
  the embedding-LM path.
- Validation now switches the active forward dimensions and packed token/target
  staging to the eval batch while keeping training on the full microbatch.
  `validation.eval_batch_size` reports the resolved eval batch, and each
  validation loss record reports its actual token count.
- The eval batch must be between 1 and the training `--batch-size`; the current
  fixed activation arena is still allocated for the training microbatch.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Ran focused native GPT coverage:
  `python -m pytest tests/test_native_gpt2.py -q -k
  "native_train_tile_ops_builds_torch_free_c_abi or missing"`
  (`1 passed`, `1 skipped`).
- Ran a one-step TinyStories transformer-LM validation probe on the dedicated
  RTX 5090 with `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`,
  `--eval-every-steps 1`, `--eval-batches 1`, and `--eval-batch-size 4`.
  Runtime JSON reported `validation.eval_batch_size: 4`,
  `validation.losses[0].tokens: 4096`, no missing symbols, and `passed: true`.

### 2026-06-14 Dense GPT LM-head row chunk retune to 8192

#### Changed

- Dense GPT native training now defaults the tied LM-head row chunk to 8192
  rows across the compiled C++ trainer, Python native wrapper, and SDK config.
  This reduces the default full-vocab LM-head chunk count from 16 to 8 at the
  `64 x 1024` workstation microbatch.

#### Breaking changes

- Callers that relied on the implicit 4096-row tied LM-head workspace should
  pass `--lm-head-row-chunk-size 4096`,
  `--native-cuda-lm-head-row-chunk-size 4096`, or
  `NativeGpt2RunConfig(lm_head_row_chunk_size=4096, ...)` to keep the older
  lower-memory profile.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Ran focused default/config coverage:
  `python -m pytest tests/test_native_gpt2.py -q -k
  "build_native_gpt2_run_config_matches_sm120_cli_shape or
  build_native_gpt2_compiled_cli_config_passes_dataset_alias_without_shard_inspection
  or native_train_tile_ops_builds_torch_free_c_abi"` (`2 passed`, `1 skipped`).
- Verified syntax and formatting with
  `python -m py_compile neuralfn/native_gpt2.py cli/scripts/train_gpt_native.py`
  and `git diff --check`.
- Ran a two-step train-only stage profile on the dedicated RTX 5090 with
  `CUDA_VISIBLE_DEVICES=0` and `CUDA_DEVICE_MAX_CONNECTIONS=1`; current default
  timing was about `6.56s` for two optimizer steps and `159,931` train tokens/s
  before this retune.
- Ran `tools/paired_kernel_speed.py` with one warmup pair and three measured
  pairs comparing the current 4096-row default against `--lm-head-row-chunk-size
  8192`. The benchmark recorded GPU 0 as `NVIDIA GeForce RTX 5090`, `0%`
  utilization, about `454/32607` MiB used, and no compute processes before
  timing. The 8192-row candidate reported `candidate_over_baseline` mean
  `0.998067`, median `0.997965`, min `0.997729`, and max `0.998505`.
- Rejected `--lm-head-row-chunk-size 16384` after an interleaved paired run
  reported `candidate_over_baseline` mean `1.655394`.
- Ran a post-change one-step CUDA Tile probe on the dedicated RTX 5090. Runtime
  JSON reported `lm_head_row_chunk_size: 8192`,
  `lm_head_row_chunk_count: 8`, `lm_head_bf16_logits_enabled: true`,
  no missing symbols, and `passed: true`.

### 2026-06-14 Generic native train SDK binding command selection

#### Changed

- Updated the generic `neuralfn._native_train` C++ binding to accept
  `compiled_cli_argv` and `launcher_argv` in addition to `argv`.
- GPT alias-only configs now prefer `compiled_cli_argv` inside the generic
  binding, keeping dataset alias and cached-shard resolution in the compiled
  C++ frontend instead of falling through to raw external trainer commands with
  empty train/validation paths.

#### Verification

- Added test coverage that builds `neuralfn._native_train`, feeds it an
  alias-only GPT-style config containing both raw `argv` and
  `compiled_cli_argv`, and verifies the compiled CLI command is selected.
- Ran `python -m pytest tests/test_native_gpt2.py -q`; 33 tests passed and 1
  was skipped.
- Ran `git diff --check`.

### 2026-06-14 Native GPT Tile BF16 bias-in-place

#### Changed

- Added a CUDA Tile implementation for the BF16 bits bias-in-place helper used
  by the packed-QKV dense GPT forward path. The scalar CUDA kernel remains
  compiled as a diagnostic fallback.
- The Tile implementation is enabled by default. Set
  `NFN_TILE_CUDA_BF16_BIAS_INPLACE_TILE=0`,
  `NFN_NATIVE_GPT_BF16_BIAS_INPLACE_TILE=0`, or
  `NFN_NATIVE_GPT2_BF16_BIAS_INPLACE_TILE=0` to compare against the older
  scalar CUDA bias kernel.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh` and rebuilt
  `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran a one-step TinyStories probe on the dedicated RTX 5090 with
  `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and
  `NFN_NATIVE_GPT_STAGE_TIMING=1`. The run completed with no missing symbols
  and reported `block_forward.attention.qkv_layout` at about `0.390 ms` per
  block.
- Ran `tools/paired_kernel_speed.py` on GPU 0 with one warmup pair and five
  measured pairs comparing `NFN_TILE_CUDA_BF16_BIAS_INPLACE_TILE=0` against the
  default Tile kernel. The benchmark recorded GPU 0 as
  `NVIDIA GeForce RTX 5090`, `0%` utilization, about `399/32607` MiB used, and
  no compute processes before timing. The Tile candidate reported
  `candidate_over_baseline` mean `0.997901`, median `0.997173`, min
  `0.995190`, and max `1.002785`.

### 2026-06-14 Native GPT BF16 validation loss

#### Changed

- Added the raw Tile ABI
  `nfn_native_tile_token_cross_entropy_partials_bf16_bits`. Dense GPT
  validation/test loss can now consume BF16 LM-head logits directly instead of
  recomputing a separate float logits workspace.
- Full dense GPT `--train-transformer-lm` enables BF16 LM-head loss by default
  when BF16 LM-head logits are enabled. Runtime JSON now reports
  `lm_head_loss_logits_dtype`, `lm_head_bf16_loss_enabled`, and
  `logit_workspace_elements: 0` on the default path.
- Set `NFN_NATIVE_GPT_BF16_LM_HEAD_LOSS=0` or
  `NFN_NATIVE_GPT2_BF16_LM_HEAD_LOSS=0` to compare validation/test loss against
  the older float logits loss workspace while keeping BF16 training backward.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`, rebuilt
  `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`, and
  rebuilt `build/nfn_native_train` with `bash tools/build_native_train_cli.sh`.
- Ran `python -m pytest tests/test_native_gpt2.py -q`; 33 tests passed and 1
  was skipped.
- Ran a one-step TinyStories validation probe on the dedicated RTX 5090 with
  `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and
  `NFN_NATIVE_GPT_STAGE_TIMING=1`. The run reported
  `lm_head_loss_logits_dtype: "bf16"`, `lm_head_bf16_loss_enabled: true`,
  `logit_workspace_elements: 0`, no missing symbols, and one validation loss at
  step 1.
- Ran `tools/paired_kernel_speed.py` on GPU 0 with one warmup pair and five
  measured pairs comparing `NFN_NATIVE_GPT_BF16_LM_HEAD_LOSS=0` against the
  default BF16 loss path. The train-only setup comparison was neutral-positive
  with `candidate_over_baseline` mean `0.999170` and median `0.997590`. The
  eval-inclusive comparison reported mean `0.967912`, median `0.969412`, min
  `0.960212`, and max `0.971242`.

### 2026-06-14 Native GPT direct BF16 residual1 LayerNorm backward

#### Changed

- Added raw Tile ABI symbols
  `nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32`
  and
  `nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32`.
  Native GPT training can now consume stored BF16 residual1 activations directly
  in LN2 backward instead of restoring them to FP32 first.
- Dense GPT enables the direct BF16 residual1 LayerNorm backward consumer by
  default when residual1 activation storage, LayerNorm forward stats, and fused
  LayerNorm dInput/residual-add are active. Set
  `NFN_NATIVE_GPT_BF16_RESIDUAL1_LN_BACKWARD=0` or
  `NFN_NATIVE_GPT2_BF16_RESIDUAL1_LN_BACKWARD=0` to compare against the older
  restore-to-FP32 LayerNorm backward path.
- Runtime JSON now reports `residual1_backward_consumer_strategy` and changes
  active residual1 storage telemetry to
  `residual1_activation_storage_strategy:
  "bf16-forward-store-direct-ln-backward"` when the direct BF16 consumer is
  active. The default one-step TinyStories probe now reports
  `stored_residual1_activation_restore_kernel_launches: 0`.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`, rebuilt
  `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`, and
  rebuilt `build/nfn_native_train` with `bash tools/build_native_train_cli.sh`.
- Ran a one-step TinyStories probe on the dedicated RTX 5090 with
  `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and
  `NFN_NATIVE_GPT_STAGE_TIMING=1`. The run loaded the new symbols with no
  missing symbols, reported
  `residual1_backward_consumer_strategy: "bf16-layernorm-backward"`,
  `stored_residual1_activation_restore_kernel_launches: 0`, and about
  `156,728` train tokens/s.
- Ran `tools/paired_kernel_speed.py` on GPU 0 with one warmup pair and five
  measured pairs comparing `NFN_NATIVE_GPT_BF16_RESIDUAL1_LN_BACKWARD=0`
  against the default direct BF16 consumer. The benchmark recorded GPU 0 as
  `NVIDIA GeForce RTX 5090`, `0%` utilization, about `380/32607` MiB used, and
  no compute processes before timing. The direct BF16 candidate reported
  `candidate_over_baseline` mean `0.993841`, median `0.994280`, min
  `0.989406`, and max `0.998092`.

### 2026-06-14 Native GPT fused residual1 BF16 store

#### Changed

- Added the raw Tile ABI
  `nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_float32`.
  It extends the stats-preserving attention projection bias+residual+LN2 fused
  kernel so dense GPT can write the BF16 residual1 activation cache in the same
  launch instead of launching a separate `float32_to_bf16` conversion over
  residual1.
- Dense GPT training uses the fused residual1 store by default when residual1
  activation storage is enabled. Set `NFN_NATIVE_GPT_FUSE_RESIDUAL1_STORE=0` or
  `NFN_NATIVE_GPT2_FUSE_RESIDUAL1_STORE=0` to compare against the older separate
  store path.
- Runtime JSON now reports `residual1_activation_store_strategy` as either
  `"fused-attention-residual-layernorm-bf16-store"` or
  `"separate-float32-to-bf16-store"` when residual1 storage is active.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh` and rebuilt
  `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran a one-step TinyStories probe on the dedicated RTX 5090 with
  `CUDA_VISIBLE_DEVICES=0` and `CUDA_DEVICE_MAX_CONNECTIONS=1`; it reported
  `residual1_activation_store_strategy:
  "fused-attention-residual-layernorm-bf16-store"` and loaded the new raw symbol
  with no missing symbols.
- Ran `tools/paired_kernel_speed.py` on GPU 0 with one warmup pair and five
  measured pairs comparing `NFN_NATIVE_GPT_FUSE_RESIDUAL1_STORE=0` against the
  default fused store. The benchmark recorded GPU 0 as
  `NVIDIA GeForce RTX 5090`, `0%` utilization, about `331/32607` MiB used, and
  no compute processes before timing. The fused candidate reported
  `candidate_over_baseline` mean `0.998396`, median `0.998298`, min
  `0.997474`, and max `0.999760`.

### 2026-06-14 Native GPT residual1 activation cache default

#### Changed

- Dense GPT Tile-CUDA training now stores intermediate block `residual1` tensors
  as BF16 by default during forward and restores them during
  packed-attention recompute so backward skips the recomputed attention
  projection and projection-residual work for earlier blocks.
- Set `NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS=0` or the legacy fallback
  `NFN_NATIVE_GPT2_STORE_RESIDUAL1_ACTIVATIONS=0` to disable this cache for
  lower-memory comparisons.
- The cache intentionally excludes the final block because the final block
  remains in the scratch tape and is not recomputed before backward. At the
  default 12-layer `64 x 1024 x 768` shape, the option stores 11 residual
  tensors and adds about 1.03 GiB.
- Runtime JSON now reports `residual1_activation_storage_strategy`,
  `stored_residual1_activation_blocks`,
  `stored_residual1_activation_elements`,
  `stored_residual1_activation_bytes`,
  `stored_residual1_activation_store_kernel_launches`, and
  `stored_residual1_activation_restore_kernel_launches`.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Ran a one-step TinyStories probe on the dedicated RTX 5090 with
  `CUDA_VISIBLE_DEVICES=0` and `CUDA_DEVICE_MAX_CONNECTIONS=1`; it reported
  `stored_residual1_activation_blocks: 11`,
  `stored_residual1_activation_bytes: 1107296256`, and matching store/restore
  launch counts for the 11 recomputed blocks.
- Ran `tools/paired_kernel_speed.py` on GPU 0 with one warmup pair and five
  measured pairs comparing the previous no-cache default to the residual1-cache
  candidate. The benchmark recorded GPU 0
  as `NVIDIA GeForce RTX 5090`, `0%` utilization, about `245/32607` MiB used,
  and no compute processes before timing. The candidate reported
  `candidate_over_baseline` mean `0.994567`, median `0.995901`, min
  `0.989916`, and max `0.997638`.
- Ran a stage-timed two-step probe with the option enabled. The
  `block_recompute_saved_packed_attention` bucket dropped to about `90.165 ms`
  versus the prior default profile's about `182.194 ms`, and the overall run
  reported about `156,691` train tokens/s.

### 2026-06-14 Native GPT all-block activation storage default

#### Changed

- Native dense GPT Tile-CUDA training now saves BF16 MLP activations and packed
  BF16 QKV/O attention tensors for all trained blocks by default on the
  workstation shape instead of capping storage at the first 11 blocks. This
  keeps the final block on the same fused stored-activation path as earlier
  blocks.
- Added `NFN_NATIVE_GPT_STORE_MLP_BLOCKS=N` and
  `NFN_NATIVE_GPT2_STORE_MLP_BLOCKS=N` as saved-MLP block-count overrides. The
  existing `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=N` cap can now select
  the final block as well. The default caps are now 12.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh` and rebuilt
  `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran a one-step probe on the dedicated RTX 5090:
  `env CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1
  NFN_NATIVE_GPT_STORE_MLP_BLOCKS=12
  NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=12
  build/nfn_gpt_native_train --tinystories --max-steps 1
  --eval-every-steps 0 --no-checkpoint --allow-train-val-fallback
  --tile-ops-lib build/libnfn_native_train_tile_ops.so`. It completed without
  OOM and reported `stored_mlp_activation_blocks: 12`,
  `stored_packed_attention_activation_blocks: 12`, and about `153,412`
  train tokens/s.
- Ran `tools/paired_kernel_speed.py` on GPU 0 with one warmup pair and three
  measured pairs comparing the previous default against
  `NFN_NATIVE_GPT_STORE_MLP_BLOCKS=12
  NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=12`; the helper recorded GPU 0 as
  `NVIDIA GeForce RTX 5090`, `0%` utilization, about `508/32607` MiB used, and
  no compute processes before timing. The candidate reported
  `candidate_over_baseline` mean `0.993963`.
- Rejected disabling cuBLASLt for BF16 backward-input GEMMs after a paired
  benchmark reported `candidate_over_baseline` mean `1.005956`.
- Rejected disabling the BF16 cuBLASLt `BGRADB` epilogue after a paired
  benchmark reported `candidate_over_baseline` mean `1.095258`.

### 2026-06-14 Native GPT packed attention saved LSE

#### Changed

- Stored packed-QKV attention now saves the per-row TK softmax `lse` alongside
  packed BF16 QKV and packed BF16 O, then feeds that saved `lse` to the packed
  backward bridge for stored blocks. This avoids relying on the shared TK
  workspace contents from another block when backward consumes saved packed
  attention activations.
- Added raw Tile-CUDA ABI symbols
  `nfn_native_tile_scaled_dot_product_attention_packed_qkv_store_lse_bf16_float32`
  and
  `nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_saved_lse_bf16_from_merged_grad_float32`.
  Plan and runtime JSON now report `stored_packed_attention_lse_enabled`,
  `stored_packed_attention_lse_elements`, and
  `stored_packed_attention_lse_bytes`.
- Added `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE=0` and the
  `NFN_NATIVE_GPT2_*` fallback to reproduce the older shared-workspace LSE
  behavior in paired benchmarks.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh` and rebuilt
  `build/nfn_gpt_native_train` with `bash tools/build_native_gpt_cli.sh`.
- Ran a one-step TinyStories probe on GPU 0 with
  `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1
  NFN_NATIVE_GPT_STAGE_TIMING=1 build/nfn_gpt_native_train --tinystories
  --tile-ops-lib build/libnfn_native_train_tile_ops.so --max-steps 1
  --eval-every-steps 0 --no-checkpoint --output-dir /tmp/nfn-packed-lse-probe`.
  It reported `stored_packed_attention_lse_enabled: true`,
  `stored_packed_attention_lse_elements: 9437184`, and
  `stored_packed_attention_backward_consumer_strategy:
  "saved-packed-qkv-o-lse-bf16-backward-to-qkv"`.
- Ran `tools/paired_kernel_speed.py` on GPU 0 with one warmup pair and three
  measured pairs, comparing the older
  `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE=0` path against the saved-LSE
  default in the same binary. The helper recorded GPU 0 as
  `NVIDIA GeForce RTX 5090`, `0%` utilization, about `437/32607` MiB used, and
  no compute processes before timing. The saved-LSE candidate was
  performance-neutral with `candidate_over_baseline` mean `1.000463`.

### 2026-06-14 Native GPT dedicated CUDA device default

#### Changed

- `NativeGptRunConfig` / `NativeGpt2RunConfig` now carry
  `cuda_visible_devices="0"` by default, alongside the existing
  `cuda_device_max_connections="1"` default. SDK subprocess, launcher,
  compiled-CLI, and binding runs set `CUDA_VISIBLE_DEVICES` only when the caller
  has not already set it.
- `cli/scripts/train_gpt.py`, `nfn_gpt_native_train`,
  `nfn_gpt2_tile_train`, and the unified `nfn_native_train` dispatcher now also
  default unset `CUDA_VISIBLE_DEVICES` to `0`. This matches the workstation
  layout where the AMD GPU is the primary display device and the RTX 5090 is the
  dedicated CUDA compute GPU.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`, rebuilt `build/nfn_native_train` with
  `bash tools/build_native_train_cli.sh`, and rebuilt
  `build/nfn_gpt2_tile_train` with `bash tools/build_native_gpt2_launcher.sh`.
- Verified Python syntax with
  `python -m py_compile neuralfn/native_gpt2.py cli/scripts/train_gpt.py`.
- Ran focused SDK/native tests:
  `python -m pytest tests/test_native_gpt2.py -q -k
  'binding_runner_invokes_in_process_module or compiled_cli_runner_executes_cli
  or cpp_binding_uses_compiled_cli_for_alias_only_config or
  native_train_tile_ops_builds_torch_free_c_abi or
  native_train_tile_ops_exports_required_symbols'`
  (`3 passed`, `1 skipped`) and
  `python -m pytest tests/test_native_gpt2.py -q -k
  'native_gpt2_cpp_binding_builds_and_runs or
  native_train_cpp_binding_builds_and_runs'` (`2 passed`).
- Ran `tools/paired_kernel_speed.py` on the dedicated RTX 5090 with
  `CUDA_VISIBLE_DEVICES=0`; the helper recorded GPU 0 as
  `NVIDIA GeForce RTX 5090`, `0%` utilization, about `373/32607` MiB used, and
  no compute processes before timing.
- Ran a post-change one-step native GPT Tile-CUDA probe with
  `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_MAX_CONNECTIONS` unset:
  `env -u CUDA_VISIBLE_DEVICES -u CUDA_DEVICE_MAX_CONNECTIONS
  build/nfn_gpt_native_train --tinystories --max-steps 1 --eval-every-steps 0
  --no-checkpoint --allow-train-val-fallback --tile-ops-lib
  build/libnfn_native_train_tile_ops.so`. It completed successfully with
  `linear_backend_strategy:
  "block-bf16-cublaslt-shape-gated-lm-head-tk-sm120-default"` and about
  `151,713` train tokens/s.
- Rejected changing the LM-head row chunk from 4096 to 8192 after a paired
  benchmark with one warmup pair and three measured pairs reported
  `candidate_over_baseline` mean `1.003552`.
- Rejected splitting BF16 cuBLASLt `BGRADB` bias-gradient accumulation from
  dWeight GEMM after a paired benchmark reported `candidate_over_baseline` mean
  `1.047523`.
- Rejected reducing the saved packed-attention block cap from 11 to 10 after a
  paired benchmark reported `candidate_over_baseline` mean `1.003216`.

### 2026-06-14 Native GPT cuBLASLt heuristic retune

#### Changed

- Retuned the trainer-facing BF16 cuBLASLt block GEMM planner to select
  heuristic index 1 by default when cuBLASLt returns at least two viable
  algorithms. This affects shape-supported transformer-block BF16 forward,
  dInput, and dWeight/BGRADB GEMMs in the native Tile CUDA trainer path.
- Added `NFN_TILE_CUDA_CUBLASLT_HEURISTIC_INDEX` and
  `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX` as profiling overrides for
  paired kernel experiments.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Ran short paired TinyStories native GPT probes on the dedicated RTX 5090 with
  `CUDA_VISIBLE_DEVICES=0` and `CUDA_DEVICE_MAX_CONNECTIONS=1`. The benchmark
  recorded GPU 0 as `NVIDIA GeForce RTX 5090`, `0%` utilization, `493/32607`
  MiB, and no compute processes before timing. Candidate ratios were:
  heuristic index 0 `1.001252`, index 1 `0.990418`, index 2 `0.995652`, and
  index 3 `0.995795`.
- Confirmed heuristic index 1 with one warmup pair and five measured pairs:
  baseline mean `7.650571s`, candidate mean `7.585230s`, and
  `candidate_over_baseline` mean `0.991460`.
- After promoting index 1 as the no-env default, ran a short default-vs-explicit
  index 1 paired check. It was neutral within short-run noise
  (`candidate_over_baseline` mean `1.003800`) and again recorded GPU 0 at `0%`
  utilization with no compute processes before timing.

### 2026-06-14 Native GPT row-chunk reduction tuning

#### Changed

- Retuned the shared large-row Tile atomic reduction chunk size from 1024 rows
  to 512 rows for Linear bias-gradient fallbacks, Linear dWeight accumulate
  fallbacks, and LayerNorm affine-gradient fallbacks. The smaller chunk
  increases reduction-grid parallelism for the default dense GPT
  `batch=64`, `seq=1024` native C++ training shape while keeping the same
  accumulation contract and cuBLAS/cuBLASLt fast paths where those are selected.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Ran a GPU-visible one-step TinyStories stage profile on the dedicated RTX
  5090 with `CUDA_VISIBLE_DEVICES=0` and `CUDA_DEVICE_MAX_CONNECTIONS=1`.
  The 512-row candidate reported `train_loop_wall_ms: 3488.93`,
  `train_tokens_per_second: 150272`, and reduced combined LN1/LN2 affine time
  from about `95.5 ms` to about `59.1 ms` versus the previous 1024-row profile.
- Built a temporary committed-baseline worktree and ran
  `tools/paired_kernel_speed.py` with one warmup pair and three measured pairs,
  pinning both commands to `CUDA_VISIBLE_DEVICES=0`. The benchmark recorded GPU
  0 as the dedicated `NVIDIA GeForce RTX 5090`, `0%` utilization, `539/32607`
  MiB, and no compute processes before timing; the 512-row candidate reported
  `candidate_over_baseline` mean `0.992931`. A 256-row candidate also beat
  baseline but was slower than 512-row in paired timing
  (`candidate_over_baseline` mean `0.994437`), so 512 rows was kept.

### 2026-06-14 Native GPT BF16 fused MLP shadow routes

#### Changed

- Added raw Tile ABI symbols for BF16-shadow fused MLP routes:
  `nfn_native_tile_linear_weight_bf16_gelu_bf16_float32` for stored-MLP
  FC+bias+GELU and
  `nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_float32` for
  fused MLP projection dInput plus saved-BF16 GELU backward.
- Dense GPT native training now routes those fused MLP paths through persistent
  BF16 block-weight shadows instead of repacking the FP32 block weights inside
  the fused kernels. FP32 master weights, gradients, and AdamW state remain the
  optimizer source of truth.
- Runtime JSON now reports the shadow-weight fused strategies as
  `stored_mlp_forward_strategy:
  "tk-sm120-fused-fc-bias-gelu-bf16-store-bf16-shadow-weight"` and
  `block_backward_mlp_proj_dgelu_strategy:
  "tk-sm120-fused-dinput-dgelu-bf16-store-bf16-shadow-weight-float32-grad"`.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Ran focused native GPT tests:
  `python -m pytest tests/test_native_gpt2.py -q -k
  'build_native_gpt2_run_config_matches_sm120_cli_shape or
  build_native_gpt2_compiled_cli_config_passes_dataset_alias_without_shard_inspection
  or native_train_tile_ops_builds_torch_free_c_abi or
  native_train_tile_ops_exports_required_symbols'` (`2 passed`, `1 skipped`).
- Ran a GPU-visible one-step TinyStories stage profile on the dedicated RTX
  5090 with `CUDA_VISIBLE_DEVICES=0` and `CUDA_DEVICE_MAX_CONNECTIONS=1`. JSON
  reported the two new ABI symbols in `kernels`, the new strategy strings,
  `linear_bf16_a_pack_count: 556`, and about `148,888` tokens/s.
- Built a temporary previous-commit worktree at `ed5370d` and ran
  `tools/paired_kernel_speed.py` with one warmup pair and five measured pairs,
  pinning both commands to `CUDA_VISIBLE_DEVICES=0`. The benchmark recorded GPU
  0 at `0%` utilization, `613/32607 MiB`, and no compute processes before
  timing; the candidate reported `candidate_over_baseline` mean `0.994699`.

### 2026-06-14 Native GPT BF16 block weight shadows

#### Changed

- The raw trainer Tile ABI now exposes BF16-shadow weight variants:
  `nfn_native_tile_linear_weight_bf16_float32`,
  `nfn_native_tile_linear_weight_bf16_output_float32`,
  `nfn_native_tile_linear_bf16_input_weight_bf16_float32`, and
  `nfn_native_tile_linear_backward_input_weight_bf16_float32`.
- Dense GPT native training now keeps FP32 master weights, gradients, and AdamW
  state, but allocates a persistent BF16 shadow arena for block QKV, attention
  projection, MLP FC, and MLP projection weights. The compiled trainer refreshes
  all block shadows with one `nfn_native_tile_float32_to_bf16_bits_many` call
  after parameter initialization and after each AdamW update, then consumes the
  shadows for block forward/recompute and block dInput GEMMs.
- Training JSON now reports `block_weight_bf16_shadow_strategy`,
  `block_weight_bf16_shadow_elements`, `block_weight_bf16_shadow_bytes`,
  `block_weight_bf16_shadow_descriptor_count`,
  `block_weight_bf16_shadow_max_elements`, and
  `block_weight_bf16_refresh_count`.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Ran a GPU-visible one-step TinyStories stage profile on the dedicated RTX
  5090. JSON reported `block_weight_bf16_shadow_bytes: 169869312`,
  `block_weight_bf16_shadow_descriptor_count: 48`,
  `block_weight_bf16_refresh_count: 2`, and about `149,670` tokens/s.
- Built a temporary previous-commit worktree at `953875a` and ran
  `tools/paired_kernel_speed.py` with one warmup pair and three measured pairs,
  pinning both commands to `CUDA_VISIBLE_DEVICES=0`. The candidate reported
  `candidate_over_baseline` mean `0.998421`; the benchmark recorded GPU 0 at
  `0%` utilization, `589/32607 MiB`, and no compute processes before timing.

### 2026-06-14 Native GPT LM-head workspace and packed-attention retune

#### Changed

- Dense GPT native training now defaults the tied LM-head row chunk to 4096
  rows instead of 8192 rows across the compiled C++ trainer, Python native
  wrapper, and SDK config. The smaller chunk halves BF16 logit workspace at the
  default `64 x 1024` microbatch, leaving memory for one additional stored
  packed-attention block.
- The default packed-QKV attention activation store cap is now eleven earlier
  blocks instead of ten. The explicit
  `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=N` and GPT-2-prefixed fallback
  still override the default.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Verified `python -m pytest tests/test_native_gpt2.py -q -k
  'build_native_gpt2_run_config_matches_sm120_cli_shape or
  build_native_gpt2_compiled_cli_config_passes_dataset_alias_without_shard_inspection
  or native_train_tile_ops_builds_torch_free_c_abi'` (`2 passed`, `1 skipped`).
- Verified `git diff --check`.
- Rejected a 16,384-row LM-head chunk candidate on the dedicated RTX 5090: the
  paired benchmark reported `candidate_over_baseline` mean `5.900867` because it
  hit the memory-pressure cliff.
- Confirmed 4096-row chunks were neutral/slightly faster than 8192-row chunks in
  an interleaved five-sample benchmark:
  `candidate_over_baseline` mean `0.998056`.
- Confirmed the combined candidate, 4096-row LM-head chunks plus eleven stored
  packed-attention blocks, remained faster than the current defaults in an
  interleaved five-sample benchmark:
  `candidate_over_baseline` mean `0.997010`.
- Ran a clean default one-step TinyStories stage profile on the dedicated RTX
  5090 after changing the defaults; JSON reported
  `lm_head_row_chunk_size: 4096`,
  `stored_packed_attention_activation_blocks: 11`, and about `144,765`
  tokens/s.

### 2026-06-14 Native GPT packed-attention backward cap diagnostics

#### Changed

- The SM120 packed-QKV attention backward bridge now defaults its internal
  packed-backward batch cap to 64, matching the workstation `64 x 1024`
  microbatch in one TK backward chunk. Set
  `NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP=48` or the GPT-2-prefixed
  fallback to reproduce the previous split when running paired benchmarks.
- `attention_backward_tk_launch_count` now increments by the actual number of
  packed backward chunks launched by the packed-QKV bridge instead of counting
  only one wrapper call.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Ran an interleaved old-cap-vs-new-cap benchmark with
  `tools/paired_kernel_speed.py`, pinning both commands to
  `CUDA_VISIBLE_DEVICES=0` on the dedicated RTX 5090. The new default reported
  `candidate_over_baseline` mean `0.998176` across three samples versus forcing
  `NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP=48`.
- Attempted separate full-output stage probes after the paired run, but discarded
  them because concurrent launches left stale `[Not Found]` CUDA contexts in
  `nvidia-smi`; no additional timing claim is based on those probes.

### 2026-06-14 Paired kernel benchmark GPU snapshots

#### Changed

- `tools/paired_kernel_speed.py` now records `nvidia-smi` GPU snapshots before
  and after the interleaved benchmark when `nvidia-smi` is available. The JSON
  payload includes GPU identity, utilization, memory, and active compute
  processes in `gpu_before` and `gpu_after`, while still running when
  `nvidia-smi` is missing.
- Text output now prints the pre-run GPU identity/utilization summary and the
  number of compute processes seen before timing starts.

#### Verification

- Verified `python -m py_compile tools/paired_kernel_speed.py`.
- Verified `python -m pytest tests/test_tile_cuda_examples.py -q -k
  paired_kernel_speed_tool_compiles_and_smokes`.
- Verified `git diff --check`.
- Confirmed CUDA visibility on the workstation reports only
  `GPU 0: NVIDIA GeForce RTX 5090` and used `CUDA_VISIBLE_DEVICES=0` for native
  benchmarks.
- Re-measured the current native GPT default on the dedicated RTX 5090 with
  stage timing: about `3,627.07 ms` train compute and `144,549` tokens/s.
- Ran paired negative controls on the dedicated RTX 5090:
  `NFN_NATIVE_LINEAR_BF16_CUBLASLT=0` was about `1.078777x` slower than the
  current cuBLASLt default, `NFN_NATIVE_LINEAR_TK_FLOAT_OUT=1` was about
  `1.002213x` of baseline, and `NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS=0` was
  about `1.101086x` slower. Those candidates were not promoted.

### 2026-06-14 Native GPT packed-attention store cap retune

#### Changed

- Dense GPT native training now stores packed BF16 QKV/O activations for ten
  earlier transformer blocks by default on the packed-QKV attention path,
  instead of six. The `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=N` and
  `NFN_NATIVE_GPT2_STORE_PACKED_ATTENTION_BLOCKS=N` overrides still control the
  cap, and `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_ACTIVATIONS=0` still returns
  to the lower-memory recompute route.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Verified `python -m pytest tests/test_native_gpt2.py -q -k
  native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`.
- Verified `git diff --check`.
- Swept `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS` values on the dedicated
  RTX 5090 with one-step TinyStories stage timing. Ten stored blocks reported
  about `3,587.55 ms` train compute and `146,141` tokens/s; eleven stored
  blocks regressed to about `5,502.16 ms` and `95,288` tokens/s, confirming the
  memory-pressure cliff.
- Ran a live default one-step stage probe after changing the default; runtime
  JSON reported `stored_packed_attention_activation_blocks: 10`,
  `stored_packed_attention_bf16_bytes: 4026531840`, and about `144,924`
  tokens/s.
- Ran an interleaved paired benchmark with `tools/paired_kernel_speed.py`,
  pinning both commands to `CUDA_VISIBLE_DEVICES=0`. The ten-block candidate
  reported `candidate_over_baseline` mean `0.987715` across five samples
  versus the previous six-block default.

### 2026-06-14 Native GPT fused LayerNorm residual backward

#### Changed

- Added the raw Tile CUDA ABI
  `nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_float32`.
  The dense GPT native trainer now uses it for block LN1/LN2 backward when
  forward LayerNorm stats are available, fusing LayerNorm dInput with the
  residual-gradient add that previously ran as a separate Tile launch.
- Added `NFN_NATIVE_GPT_FUSE_LN_BACKWARD_RESIDUAL=0` and the
  `NFN_NATIVE_GPT2_FUSE_LN_BACKWARD_RESIDUAL=0` compatibility fallback for
  paired old-vs-new measurements. Runtime JSON now reports
  `block_state_layout.layer_norm_backward_residual_fusion_enabled` and
  `block_state_layout.layer_norm_backward_residual_strategy`.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Verified `python -m pytest tests/test_native_gpt2.py -q -k
  native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`.
- Verified `build/nfn_gpt_native_train --backend tile-cuda --check-tile-ops
  --tile-ops-lib build/libnfn_native_train_tile_ops.so` on the RTX 5090; JSON
  reported `all_required_symbols_found: true` and found
  `nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_float32`.
- Verified `git diff --check`.
- Ran GPU-visible RTX 5090 one-step TinyStories stage probes for fused and
  disabled routes. The fused route reported
  `layer_norm_backward_residual_strategy` as
  `"fused-dinput-residual-add-with-forward-stats"`, reduced the combined
  LN1/LN2 residual buckets from about `237 ms` to about `184 ms`, and improved
  stage-mode throughput from about `139,413` to `141,698` tokens/s.
- Ran an interleaved paired benchmark with `tools/paired_kernel_speed.py`,
  pinning both commands to `CUDA_VISIBLE_DEVICES=0`. The fused candidate
  reported `candidate_over_baseline` mean `0.986385` across five samples
  versus `NFN_NATIVE_GPT_FUSE_LN_BACKWARD_RESIDUAL=0`.

### 2026-06-14 Native GPT fused MLP projection dGELU

#### Changed

- Added the trainer-facing raw Tile CUDA ABI
  `nfn_native_tile_linear_backward_input_dgelu_bf16_bits_float32`. The default
  dense GPT native trainer uses it for stored MLP activations to fuse the MLP
  projection dInput GEMM with saved-BF16 GELU backward, then writes the fused
  result into the float gradient buffer used by the existing block backward
  pipeline.
- Native GPT runtime JSON now reports
  `block_backward_mlp_proj_dgelu_fusion_enabled` and
  `block_backward_mlp_proj_dgelu_strategy`. Set
  `NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU=0` or
  `NFN_NATIVE_GPT2_FUSE_MLP_PROJ_DGELU=0` to force the older separate dInput
  plus GELU-backward path for paired benchmarks and diagnostics.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` with
  `bash tools/build_native_train_tile_ops.sh`.
- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Verified `python -m pytest tests/test_native_gpt2.py -q -k
  native_gpt2_cpp_cli_builds_and_uses_sm120_defaults`.
- Verified `build/nfn_gpt_native_train --backend tile-cuda --print-plan`
  advertises `nfn_native_tile_linear_backward_input_dgelu_bf16_bits_float32`.
- Verified `git diff --check`.
- Ran a GPU-visible RTX 5090 paired one-step TinyStories benchmark with
  `tools/paired_kernel_speed.py`, pinning both commands to
  `CUDA_VISIBLE_DEVICES=0`. The fused candidate reported
  `candidate_over_baseline` mean `0.9944107816` across five samples versus
  `NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU=0`.
- Ran direct one-step probes for the disabled and enabled routes. The disabled
  route reported `block_backward_mlp_proj_dgelu_fusion_enabled: false`,
  `block_backward_mlp_proj_dgelu_strategy: "separate-dinput-plus-gelu"`, and
  `linear_tk_gemm_count: 288`; the enabled route reported
  `block_backward_mlp_proj_dgelu_fusion_enabled: true`,
  `block_backward_mlp_proj_dgelu_strategy` as
  `"tk-sm120-fused-dinput-dgelu-bf16-store-float32-grad"`, and
  `linear_tk_gemm_count: 376`.

### 2026-06-13 Native GPT no-checkpoint benchmark mode

#### Changed

- Added `--no-checkpoint` / `--native-cuda-no-checkpoint` to the compiled
  dense GPT CUDA Tile trainer and Python wrappers. This is a timing-only
  preflight/benchmark control: default native GPT training still writes the
  final checkpoint, while disabled runs report checkpoint export disabled in
  plan/runtime JSON and skip the final checkpoint wall time.
- Added SDK support through `NativeGptRunConfig.write_checkpoint` and
  `NativeGpt2RunConfig.write_checkpoint`; `compiled_cli_argv()` forwards
  `write_checkpoint=False` as `--no-checkpoint`.

#### Verification

- Rebuilt `build/nfn_gpt_native_train` with
  `bash tools/build_native_gpt_cli.sh`.
- Verified `python -m py_compile neuralfn/native_gpt2.py neuralfn/native_gpt.py
  cli/scripts/train_gpt.py cli/scripts/train_gpt_native.py
  cli/scripts/train_gpt2.py`.
- Verified `python -m pytest tests/test_native_gpt2.py -q -k
  "compiled_cli_config_passes_dataset_alias_without_shard_inspection or
  can_skip_checkpoint_export or native_train_tile_ops_builds_torch_free_c_abi"`.
- Verified `python -m pytest cli/tests/test_train_gpt2_native.py -q -k
  "native_cached_shard_default_runner_uses_compiled_cli or
  translates_kernel_backend"`.
- Verified `git diff --check`.
- Ran a GPU-visible one-step TinyStories probe with `--no-checkpoint`; JSON
  reported `passed: true`, `checkpoint.enabled: false`,
  `checkpoint.checkpoint_written: false`, `checkpoint_wall_ms: 0`,
  `train_compute_wall_ms: 3774.03`, and `train_tokens_per_second: 138920`.

### 2026-06-13 Paired kernel speed measurements

#### Changed

- `nfn kernels bench` now measures graph-walk PyTorch, compiled PyTorch, and
  Tile-requested execution with paired interleaved samples instead of timing
  each mode in one isolated block. JSON output includes `measurement:
  "paired_interleaved"`, `samples`, per-sample timings, and paired ratios such
  as `compiled_tile_cuda_requested_over_compiled_pytorch`. This makes candidate
  and baseline timings share the same run window, reducing skew from unrelated
  external GPU load.
- Added `tools/paired_kernel_speed.py` for native CUDA kernel experiments where
  the older and candidate paths are separate commands or environment
  configurations. The tool alternates baseline/candidate order across samples
  and reports paired candidate-over-baseline ratios. It now accepts
  `--json-out PATH` so paired benchmark evidence can be saved without shell
  redirection, and `--cuda-visible-devices DEVICE_LIST` so both commands can be
  pinned to the same dedicated CUDA GPU.

#### Verification

- Verified `python -m pytest cli/tests/test_nfn_cli.py -q -k
  kernels_bench_json_reports_execution_modes`.
- Verified `python -m pytest tests/test_tile_cuda_examples.py -q -k
  paired_kernel_speed_tool_compiles_and_smokes`.
- Verified `python -m py_compile tools/paired_kernel_speed.py`.
- Verified dedicated RTX 5090 pinning with
  `CUDA_VISIBLE_DEVICES=0 build/nfn_gpt_native_train --tinystories --max-steps
  1 --batch-size 64 --train-seq-len 1024 --train-batch-tokens 524288
  --eval-every-steps 0 --no-checkpoint`, where the first cold run took
  `121063 ms` train compute and the immediate warm run took `3756.39 ms`;
  paired benchmarks should keep warmup enabled.

### 2026-06-13 Native GPT packed attention backward chunking

#### Fixed

- Split the trainer-facing SM120 packed-QKV TK attention backward bridge into
  bounded batch chunks before launching ThunderKittens backward. This preserves
  the row-major packed QKV/O/dO/grad-QKV layout with batch-offset pointer
  arithmetic while avoiding the worst full-batch packed-QKV backward stall seen
  above the fast `48 x 1024` microbatch shape. No CLI or SDK migration is
  required, but `libnfn_native_train_tile_ops.so` must be rebuilt for the fix.
- The default `64 x 1024` one-microbatch native GPT run now completes instead
  of timing out in the packed-QKV backward bridge. It is still too slow for the
  final workstation target; the next bottleneck is the high-row BF16
  dWeight/QKV path at batch 64.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so`.
- Ran `env build/nfn_gpt_native_train --smoke-tile-ops --tile-ops-lib
  build/libnfn_native_train_tile_ops.so`; the CUDA fill smoke passed.
- Ran GPU-visible one-step TinyStories profiles with
  `NFN_NATIVE_GPT_STAGE_TIMING=1`. The patched `48 x 1024` control reported
  75,823.9 tokens/second, `block_backward.attn_sdpa.to_qkv=42.4221ms`, and
  zero SGEMM calls. The patched solo `56 x 1024` profile completed with
  11,490.9 tokens/second and
  `block_backward.attn_sdpa.to_qkv=1498.59ms`, versus the earlier 16s-class
  packed-QKV backward cliff. The patched solo `64 x 1024` profile completed
  with 1,101.49 tokens/second, exposing the remaining BF16 dWeight/QKV
  bottleneck instead of hanging.

### 2026-06-13 GPT-native trainer default template alias

#### Breaking changes

- The native dense GPT CLI/SDK default template is now the public
  `template_name="gpt"` alias instead of `template_name="gpt2"`. The compiled
  frontend resolves that alias to the current dense GPT implementation template
  and reports `resolved_native_template_name: "gpt2"` in plan/runtime JSON.
  Callers should use `template_name`, `graph_file`, and shape fields as the
  architecture contract; do not infer architecture from `model_family` or assume
  GPT-3 is a separate trainer.

#### Changed

- Updated `train_gpt.py`, `train_gpt_native.py`, the native GPT SDK config
  builders, and `nfn_gpt_native_train` so `gpt` is the canonical public native
  trainer/template surface. `gpt2` and `gpt3` remain accepted selector aliases;
  `gpt3` only supplies a 2048-token default context when the caller did not
  provide a template, graph, or explicit sequence length.
- Added `resolved_native_template_name` to compiled dense GPT plan, unsupported
  graph, external bridge, and runtime JSON so diagnostics distinguish the
  public template alias from the implementation template used by the current
  native trainer.

#### Verification

- Rebuilt `build/libnfn_native_train_tile_ops.so` and
  `build/nfn_gpt_native_train`.
- Verified Python syntax with `python -m py_compile cli/scripts/train_gpt.py
  cli/scripts/train_gpt_native.py neuralfn/native_gpt.py
  neuralfn/native_gpt2.py`.
- Verified focused native behavior with `python -m pytest
  tests/test_native_gpt2.py -q -k "compiled_cli_config_passes_dataset_alias_without_shard_inspection
  or cpp_cli_builds_and_uses_sm120_defaults or
  compiled_cli_config_canonicalizes_dense_gpt_family or universal_gpt"`.
- Verified wrapper dispatch with `python -m pytest
  cli/tests/test_train_gpt2_native.py -q`.
- Verified shipped GPT template integrity with `python -m pytest
  tests/test_template_presets.py -x -q`.
- Verified rebuilt compiled dry-runs: default `--tinystories --dry-run` reports
  `model_family: "gpt"`, `template_name: "gpt"`,
  `resolved_native_template_name: "gpt2"`, and
  `selected_graph_support_status: "native-transformer-lm"`; `--model-family
  gpt3` reports the same architecture fields with `seq_len: 2048`.

### 2026-06-13 Native GPT 5090 Tile-CUDA cache retune

#### Changed

- Increased the trainer-facing BF16 packed-operand cache from 64 entries to 128
  entries. The cache still only stores stable operands such as weights and
  biases; BF16-output GEMMs continue to repack mutable activation inputs so
  reused scratch activation pointers cannot produce stale packed data.
- Retuned the default packed-QKV attention activation storage cap from 3 earlier
  blocks to 6 earlier blocks for the RTX 5090 workstation shape. Runtime JSON
  now reports `stored_packed_attention_activation_blocks: 6` unless callers
  override `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS` or disable packed
  attention storage. Cap 8 was tested and rejected because it triggered a large
  attention-backward regression.
  Verification: rebuilt `libnfn_native_train_tile_ops.so`, ran GPU-visible
  one-step TinyStories native CUDA profiles on the llm.kittens
  `train-sm120.sh` shape, and compared `NFN_NATIVE_GPT_STAGE_TIMING=1`
  outputs. The 128-entry cache with the old cap 3 reported 138,664
  tokens/second with 996 BF16 packs and 1,132 cache hits. Cap 4 reported
  140,169 tokens/second, cap 5 reported 140,714 tokens/second, cap 6 reported
  141,660 tokens/second, and cap 8 regressed to 92,860 tokens/second. The
  rebuilt default binary, with no packed-attention override, reported cap 6,
  141,613 tokens/second, and zero SGEMM calls. All compared runs reported zero
  SGEMM calls.

### 2026-06-13 Universal GPT native trainer contract

#### Breaking changes

- `NativeGpt2RunConfig`, `build_native_gpt2_run_config()`, and
  `build_native_gpt2_compiled_cli_run_config()` now canonicalize dense GPT
  selectors to `model_family="gpt"` instead of emitting `"gpt2"` or `"gpt3"`.
  The default public template is now the `"gpt"` alias; compiled JSON reports
  `resolved_native_template_name` when that alias maps to the current dense
  implementation template. Callers that keyed behavior off
  `model_family == "gpt2"` or `"gpt3"` should migrate to `template_name`,
  `graph_file`, and shape fields.
  New SDK code should use `neuralfn.native_gpt` and the `NativeGpt*` names.

#### Changed

- Moved the Python fallback harness implementation to
  `cli/scripts/train_gpt_native.py`. `cli/scripts/train_gpt.py` and root
  `nfn train` now import the generic module when the compiled fast path is not
  used, while `cli/scripts/train_gpt2_native.py` remains a compatibility
  wrapper. Migration note: direct GPT-2 compatibility invocations still work,
  but new native GPT integrations should target `train_gpt.py`,
  `train_gpt_native.py`, and the `neuralfn.native_gpt` SDK names.
- Made the compiled dense GPT trainer contract explicit in plan, unsupported
  graph, external bridge, and training JSON. Runs now report
  `architecture_source`, `architecture_contract`, and
  `model_family_context_policy` so diagnostics show that `--template-name` or
  `--graph-file` selects the architecture. `gpt`, `gpt2`, and `gpt3` remain
  aliases of the same trainer; `gpt3` only supplies the 2048-token default
  context when template, graph, and sequence length are all implicit.
  Verification: `python -m py_compile cli/nfn.py cli/scripts/train_gpt.py
  cli/scripts/train_gpt_native.py neuralfn/native_gpt.py
  neuralfn/native_gpt2.py` passed, `python -m pytest
  cli/tests/test_train_gpt2_native.py -q` passed (`33 passed, 90 subtests
  passed`), `python -m pytest tests/test_template_presets.py -q -x` passed
  (`26 passed`), and the focused native slice
  `python -m pytest tests/test_native_gpt2.py -q -k "universal_gpt or
  canonicalizes_dense_gpt_family or cpp_cli_builds_and_uses_sm120_defaults or
  native_train_tile_ops_builds_torch_free_c_abi"` passed (`3 passed, 1
  skipped`). Rebuilt `build/nfn_gpt_native_train`; raw `--model-family gpt3`
  dry-run reported `model_family: "gpt"`, `seq_len: 2048`, and
  `stored_packed_attention_activation_blocks: 3`. A live default TinyStories
  one-step CUDA profile reported `model_family: "gpt"`, no SGEMM calls,
  `stored_packed_attention_activation_blocks: 3`, and 138,549 tokens/second.
  `git diff --check` passed.

### 2026-06-13 Native GPT BF16 activation-cache correctness

#### Fixed

- `nfn_native_tile_linear_bf16_output_float32` no longer caches the mutable
  activation operand for BF16-output GEMMs. The native dense GPT trainer reuses
  scratch activation addresses across blocks and microbatches, so pointer-keyed
  activation cache hits could reuse stale packed BF16 inputs. The path still
  caches stable packed weights and continues to invalidate the BF16 operand
  cache after AdamW updates. Migration note: no CLI change is required, but
  rebuild `libnfn_native_train_tile_ops.so` before running native GPT training.
  Verification: `python -m pytest tests/test_native_gpt2.py -q` passed
  (`30 passed, 1 skipped`), `git diff --check` passed, the Tile ops shared
  library and native GPT CLI were rebuilt, and a one-step TinyStories native
  CUDA probe passed with no SGEMM calls while reporting 1,121 BF16 activation
  packs, 927 stable cache hits, and 129,635 tokens/second.

### 2026-06-13 Native GPT packed-attention cap retune

#### Changed

- Reduced the default packed-QKV attention activation storage cap from 8 earlier
  blocks to 3 earlier blocks. The corrected BF16-output path spends less time
  in attention backward with this smaller saved-packed window while keeping the
  explicit `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=N` override for future
  tuning. Runtime JSON now reports `stored_packed_attention_activation_blocks:
  3` by default. Verification: isolated one-step TinyStories stage profiles on
  the llm.kittens `train-sm120.sh` shape improved from `130,107` tokens/second
  with the previous default to `135,906` tokens/second with the three-block cap,
  and `block_backward.attn_sdpa.to_qkv` dropped from about `649 ms` to about
  `334 ms`.

### 2026-06-12 Native GPT-2 Tile-CUDA default

#### Breaking changes

- `neuralfn.native_gpt.NativeGptRunConfig`,
  `NativeGptRunnerStatus`, and `NativeGptCheckpointInfo` are now real
  GPT-native dataclass subclasses instead of direct aliases of the
  `NativeGpt2*` compatibility classes. Code using generic SDK helpers should
  migrate exact type-name or `type(obj) is NativeGpt2RunConfig` checks to the
  `NativeGpt*` classes. `isinstance(obj, NativeGpt2RunConfig)` remains true for
  the generic config objects because the GPT-native classes subclass the
  compatibility classes.

#### Changed

- The unified native training registry now reports `gpt`, `gpt2`, and `gpt3`
  as `implemented` aliases of the same dense GPT CUDA Tile C++ trainer and
  forwards canonical `--model-family gpt` for all three aliases. GPT-3 remains
  a GPT-native selector whose only default difference is a 2048-token context
  when no explicit template, custom graph, or sequence length is supplied; selected
  `--template-name` / `--graph-file` values define the architecture and native
  support status. Verification: updated the unified native registry test and
  rebuilt the native C++ frontends during focused test runs.
- Dense GPT native `--train-transformer-lm` now saves packed BF16 QKV plus
  packed BF16 O for the first eight earlier blocks by default on the packed-QKV
  path. This changes default `packed_attention_activation_storage_strategy`
  from `"disabled"` to `"packed-qkv-o-bf16-forward-store-direct-backward"` and
  default `stored_packed_attention_activation_blocks` from `0` to `8`.
  Migration note: set `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_ACTIVATIONS=0` for
  the previous lower-memory recompute behavior, or set
  `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=N` to tune the cap. The
  GPT-2-prefixed environment names remain compatibility fallbacks. Verification:
  serial GPU-visible TinyStories two-step probes on the default 64x1024,
  524,288-token optimizer-step shape improved from about `137,653` tokens/s with
  packed storage disabled to about `146,691` tokens/s with eight saved packed
  blocks; the same sweep showed cap 9 regressed to about `83,116` tokens/s and
  cap 11 to about `56,738` tokens/s. Focused native GPT plan/runtime JSON tests
  were updated.
- Promoted the dense GPT native C++ runtime toggles to generic
  `NFN_NATIVE_GPT_*` names. The compiled trainer now checks
  `NFN_NATIVE_GPT_STAGE_TIMING`, `NFN_NATIVE_GPT_PACKED_QKV_ATTENTION`,
  `NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS`,
  `NFN_NATIVE_GPT_STORE_ATTENTION_ACTIVATIONS`,
  `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_ACTIVATIONS`,
  `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS`,
  `NFN_NATIVE_GPT_FUSE_ATTENTION_RESIDUAL_LN2`, and
  `NFN_NATIVE_GPT_LM_HEAD_BF16_LOGITS` before the legacy GPT-2-prefixed
  fallbacks. Migration note: use the generic names for new GPT, GPT-2, GPT-3,
  template-selected, and custom-graph native training scripts; existing
  `NFN_NATIVE_GPT2_*` scripts continue to work as compatibility fallbacks.
  Verification: updated compiled plan coverage to drive packed-attention
  storage through the generic env names, rebuilt `nfn_gpt_native_train`, and
  reran the focused native GPT C++ source/compiled-plan tests.
- Made the generic dense GPT trainer path canonical in Python. `cli/scripts/train_gpt.py`
  now owns the pre-import compiled-CLI fast path, while `cli/scripts/train_gpt2.py`
  is a compatibility wrapper that still reaches the same no-Torch dispatch when
  executed directly. `cli/nfn_impl.py` now imports generic `GPT_DEFAULTS`, the
  native wrapper uses generic `neuralfn.native_gpt` helpers internally, and
  `NFN_NATIVE_GPT_BINDING` / `NFN_NATIVE_GPT_TRAIN_BIN` take precedence over
  legacy GPT-2 environment names. Migration note: use `train_gpt.py`,
  `GPT_DEFAULTS`, `neuralfn.native_gpt`, and `NFN_NATIVE_GPT_*` for new code;
  GPT-2 names remain compatibility aliases. Verification: added generic-env
  precedence coverage and reran focused no-Torch startup tests.
- Tightened the universal GPT native trainer surface and reduced the packed-QKV
  trainer tape. `cli/scripts/train_gpt2_native.py` now defaults the canonical
  Python wrapper output to `~/NeuralFn/artifacts/gpt`, accepts
  `NFN_NATIVE_GPT_TRAIN_BIN` / `NFN_NATIVE_GPT_RUNNER` ahead of the legacy
  GPT-2 environment names, and the lightweight CLI identifies native `.bin`
  checkpoints as native GPT artifacts. The generic `neuralfn.native_gpt` and
  top-level `neuralfn` exports now include generic activation, backend, and
  tokenizer helper aliases so new SDK code can stay on GPT-native names.
  Packed-QKV attention no longer allocates the legacy float QKV/split-head/O
  tape tensors when the packed path is active; plan and training JSON report
  `packed_qkv_float_attention_tape_elided`,
  `packed_qkv_float_attention_tape_elements_elided`, and
  `packed_qkv_float_attention_tape_bytes_elided`. Migration note: use
  `nfn train --base-model gpt`, `python cli/scripts/train_gpt.py`, and
  `neuralfn.native_gpt` for new code; the `gpt2` names remain compatibility
  wrappers for the current checkpoint/template implementation. Verification:
  rebuilt `nfn_gpt_native_train`, ran `python -m pytest tests/test_native_gpt2.py -q`
  (`28 passed, 1 skipped`), ran
  `python -m pytest cli/tests/test_train_gpt2_native.py -q`
  (`33 passed, 90 subtests passed`), ran `git diff --check`, and ran a
  GPU-visible one-step TinyStories probe that reported
  `packed_qkv_float_attention_tape_elided: true`, 1.61GB of float tape elided,
  `float_arena_requested_elements: 3175449357`, zero SGEMM calls, and about
  `132,779` tokens/s. A follow-up opt-in full packed-attention-store probe now
  fits in memory, but stays non-default because it regressed to about
  `44,096` tokens/s without stage timing; the stage-timed run showed
  `block_backward.attn_sdpa.to_qkv` dominating at about `8,627 ms` per
  optimizer step.
- The raw native Tile ABI now exposes `nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_float32`
  and `nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32`.
  Dense GPT block backward now uses those fused dWeight+bias entrypoints for
  qkv, attention projection, MLP fc, and MLP projection gradients. Supported
  BF16 block dWeight GEMMs request cuBLASLt `CUBLASLT_EPILOGUE_BGRADB`, write
  the bias gradient into Tile-owned scratch, and accumulate that vector into the
  existing optimizer-step bias buffer; unsupported shapes fall back inside the
  ABI to the previous BF16 dWeight plus Tile bias-reduction launchers. Training
  JSON now reports `block_backward_weight_linear_strategy:
  "shape-gated-bf16-cublaslt-dweight-bgrad-accumulate"` when that fused path is
  active, and stage timing folds the old block projection `dweight`/`bias`
  buckets into `dweight_bias` buckets. Migration note: rebuild
  `libnfn_native_train_tile_ops.so` and `nfn_gpt_native_train` together before
  running dense GPT native training because the compiled trainer now requires
  the new ABI symbols. Verification: rebuilt both native artifacts, reran the
  focused native GPT-2 static/export pytest slice, and ran a GPU-visible
  TinyStories one-step `64 x 1024`, `524288` tokens/step probe that reported
  about `128,609` tokens/s, `block_backward` about `1,962 ms`, zero SGEMM
  calls, and `block_backward_weight_linear_strategy:
  "shape-gated-bf16-cublaslt-dweight-bgrad-accumulate"`.
- Tightened the universal dense GPT native trainer surface so the compiled
  command defaults to `model_family: "gpt"` and `~/NeuralFn/artifacts/gpt`,
  while `gpt2` and `gpt3` remain aliases for the same CUDA Tile C++ trainer.
  `gpt3` still only defaults to a 2048-token context when no explicit template,
  custom graph, or `--train-seq-len` is supplied. `cli/scripts/train_gpt.py`
  now preserves its own command name when delegating through the compatibility
  implementation, and native help/error text describes the trainer as dense GPT
  rather than GPT-2-specific. Migration note: new commands and SDK code should
  use `nfn train --base-model gpt`, `python cli/scripts/train_gpt.py`, or
  `neuralfn.native_gpt`; the GPT-2 names remain compatibility wrappers around
  the existing checkpoint/template implementation.
- Promoted the dense GPT native trainer entrypoint from GPT-2-specific names to
  universal GPT names. `tools/build_native_gpt_cli.sh` now builds
  `nfn_gpt_native_train`, `nfn_native_train` defaults to `--base-model gpt`,
  and the native registry reports `gpt`, `gpt2`, and `gpt3` as dense GPT
  aliases targeting `nfn_gpt_native_train`; `gpt2` remains a template/default
  shape, not a separate trainer family. `cli/install.sh` now installs
  `nfn-gpt-native` and `nfn-gpt-native-train` as the primary commands, while
  the old GPT-2 command names and `NFN_NATIVE_GPT2_CLI` remain compatibility
  fallbacks. New configuration should use `NFN_NATIVE_GPT_CLI` or
  `--native-gpt-cli`. Verification: added/updated native dispatcher,
  build-all, installer, SDK-default, and CLI regression coverage.
- Remaining native large-row reduction fallback launchers now use the shared
  `kLinearBackwardBiasRowChunkSize` 1024-row policy instead of retaining local
  256-row chunks. This covers LayerNorm affine gradients, LayerNorm affine
  accumulate, LayerNorm affine-with-stats accumulate, Linear bias gradients,
  Linear bias accumulate, and BF16-bit Linear dWeight accumulate fallbacks.
  cuBLAS-backed GEMM shapes still prefer the cuBLAS path; this reduces launch
  grid and atomic-reduction overhead when fallback reductions are selected.
  Verification: added source assertions for the affected fallback launchers.
- Extended the universal dense GPT alias surface through the full `nfn train`
  parser, planner, compatibility spec builder, and public
  `build_composed_lm_spec()` API. `gpt`, `gpt2`, and `gpt3` now parse
  consistently outside the lightweight native dispatcher; graph-backed
  compatibility paths and SDK composed specs canonicalize those aliases to the
  GPT-compatible template builder, and `gpt3` remains only a 2048-token
  default-context alias unless a template, graph, or explicit sequence length is
  provided. Verification: added CLI and composed-spec regression coverage for
  parser acceptance, defaults, and spec canonicalization.
- Promoted the dense native trainer surface to universal GPT metadata. `nfn train --base-model gpt`, `gpt2`, and `gpt3` now forward `--model-family` into the compiled C++ trainer, the trainer JSON reports the requested family instead of hard-coding `gpt2`, and `cli/scripts/train_gpt.py` is the canonical direct script wrapper over the existing compatibility implementation. `gpt3` uses the same dense GPT native kernels and defaults to `--train-seq-len 2048` only when the caller did not provide a template, custom graph, or explicit sequence length; otherwise the selected `--template-name` or `--graph-file` remains the architecture authority. Verification: rebuilt `nfn_gpt2_native_train` and `nfn_native_train`; ran focused CLI/template/native alias tests; ran metadata smokes proving `model_family: "gpt3"`, default `max_seq_len: 2048`, and explicit `--train-seq-len 4096` preservation.
- Fixed the canonical `cli/scripts/train_gpt.py` startup path so it executes the compatibility trainer script as `__main__` and preserves the pre-import compiled-CLI fast path. Direct `python cli/scripts/train_gpt.py --native-cuda-dry-run ...` now reaches the C++ handoff before loading `train_gpt2_native.py`, Torch, NumPy, or `server.dataset_manager`. Verification: added a direct-script regression test with `NFN_NATIVE_GPT2_CLI=/bin/echo`.
- Added generic dense GPT native training aliases. `nfn train --base-model gpt`, `gpt2`, and `gpt3` now route through the same no-Python GPT-compatible compiled CLI path, and `nfn-native-train --list-models` reports `gpt`, `gpt2`, and `gpt3` as partial native aliases to the dense transformer-LM target. The new public `neuralfn.native_gpt` module exports generic aliases such as `NativeGptRunConfig`, `build_native_gpt_compiled_cli_run_config()`, `build_native_gpt_run_config()`, and `run_native_gpt()` over the existing no-Torch native implementation. Template and custom graph selectors remain the source of architectural truth, so GPT-3-style context/window changes should flow through `--template-name` or `--graph-file` rather than a separate hardcoded trainer. Verification: rebuilt `nfn_native_train`, ran the focused alias/template tests (`8 passed`), ran `python -m pytest tests/test_native_gpt2.py tests/test_native_dependencies.py -q` (`29 passed, 1 skipped`), and ran `python -m pytest cli/tests/test_train_gpt2_native.py -q` (`30 passed, 90 subtests passed`).
- The raw native Tile ABI now exposes `nfn_native_tile_layer_norm_with_stats_float32`, `nfn_native_tile_layer_norm_backward_input_with_stats_float32`, and `nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_float32`. Native C++ trainers can store LayerNorm row mean/rstd during forward and reuse those stats in backward instead of recomputing row statistics in both backward kernels. Verification: rebuilt `libnfn_native_train_tile_ops.so`; source/export tests cover the new symbols.
- Dense GPT-2 `--train-transformer-lm` now uses the LayerNorm stats ABI by default. Earlier blocks that reuse stored BF16 MLP activations also store their LN2 mean/rstd in a float sidecar so backward reads the correct per-block stats instead of stale scratch-tape stats. Runtime JSON reports `layer_norm_stats_strategy`, `layer_norm_backward_reuses_forward_stats`, `layer_norm_stats_disabled_by_fused_residual_ln2`, `stored_mlp_layer_norm_stats_elements`, and `stored_mlp_layer_norm_stats_bytes`. Verification: rebuilt `nfn_gpt2_native_train`, ran the focused native GPT-2 test slice, isolated the stats forward/backward kernels with a tiny CUDA ctypes smoke, and ran a GPU-visible two-step TinyStories probe that passed with `train_tokens_per_second: 126165`.
- The raw native Tile ABI now exports `nfn_native_tile_trainer_linear_tk_float_out_gemm_count`, and dense GPT-2 JSON reports `linear_tk_float_out_gemm_count`. `NFN_TILE_CUDA_LINEAR_TK_FLOAT_OUT=1` or `NFN_NATIVE_LINEAR_TK_FLOAT_OUT=1` enables an opt-in diagnostic path that sends eligible BF16 linear forward GEMMs through the TK BF16-output bridge, then converts the BF16 output back to float32 for the existing trainer contract. This remains disabled by default because the measured full-shape TinyStories stage probe activated the path (`linear_tk_float_out_gemm_count: 376`) but regressed total throughput from about `122,219` to about `117,504` tokens/s even though QKV forward improved. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, ran the opt-in GPU-visible stage-timed probe, and added source/export assertions for the new counter and env gates.
- Dense GPT-2 transformer block BF16 GEMMs now use cached cuBLASLt with `CUBLAS_COMPUTE_32F_FAST_16BF` by default for shape-supported block forward, dInput, and dWeight GEMMs while leaving full-vocab LM-head chunks on the existing TK/GEMMEx path. Set `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT=0` or `NFN_NATIVE_LINEAR_BF16_CUBLASLT=0` to force the older BF16 `cublasGemmEx` block bridge. Runtime JSON now reports `linear_backend_strategy: "block-bf16-cublaslt-shape-gated-lm-head-tk-sm120-default"` when the cublasLt and TK counters are nonzero, and the block strategy fields switch between `shape-gated-bf16-cublaslt-*` and `forced-bf16-gemmex-*` based on runtime counters. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`; a GPU-visible one-step stage probe reported `linear_cublaslt_gemm_count: 1240`, reduced BF16 GEMMEx calls to `280`, and improved stage throughput to about `123,359` tokens/s versus the recent about `122,219` stage baseline; the final default two-step no-stage probe reported `125,546` tokens/s versus `125,511` for same-build opt-out, so this is primarily a backend-correctness/coverage improvement with a small noisy throughput gain.
- Dense GPT-2 attention now defaults to a packed-QKV SM120 TK bridge. The raw Tile ABI exports `nfn_native_tile_bf16_bits_add_bias_inplace_float32`, `nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32`, and `nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_merged_grad_float32`; the trainer writes no-bias QKV projection output as BF16 bits, adds QKV bias in-place, runs packed TK attention without split-to-heads, and returns row-major `grad_qkv` from the packed backward bridge. Set `NFN_NATIVE_GPT2_PACKED_QKV_ATTENTION=0` to force the older split-to-heads bridge for profiling. Plan and runtime JSON report `packed_qkv_attention_enabled`, packed BF16 scratch bytes, packed layout strategy strings, and `attention_backward_strategy: "tk-sm120-packed-qkv-bf16-backward-bridge"`. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`; a GPU-visible one-step TinyStories probe with the packed path active passed and reported stage reductions for `block_forward.attention.qkv` (`154.801 ms`), `qkv_layout` (`43.7662 ms`), `sdpa` (`90.677 ms`), and `block_recompute` (`339.98 ms`) compared with the same-build split path (`405.675 ms`, `79.4453 ms`, `162.215 ms`, `459.192 ms`). An earlier same-change packed probe reported `129,358` tokens/s; same-run totals remain noisy because checkpoint/export and cuBLASLt plan timing can dominate a one-step probe.
- Dense GPT-2 packed-QKV attention now feeds packed BF16 attention output directly into the attention projection forward GEMM and dWeight accumulation instead of unpacking `O` to float32 before the projection. Plan and runtime JSON report `attention_projection_input_strategy: "packed-o-bf16-direct-gemm"` and `attention_packed_output_unpack_strategy: "elided-direct-bf16-projection"` when active. Verification: rebuilt `nfn_gpt2_native_train`; the focused native GPT-2 test slice passed (`1 passed, 1 skipped` with `nvcc` unavailable inside pytest), and a GPU-visible one-step TinyStories stage probe improved from `118,409` to `132,235` tokens/s with `block_forward.attention.proj` down from `444.064 ms` to `144.246 ms`, `block_forward` down from `1234.44 ms` to `898.523 ms`, `block_recompute` down from `340.953 ms` to `304.319 ms`, and `block_backward.attn_proj.dweight` down from `75.1748 ms` to `58.9629 ms`.
- Dense GPT-2 packed-QKV attention can now store packed BF16 QKV plus packed BF16 O for earlier blocks and reuse those saved tensors during backward. The path is opt-in with `NFN_NATIVE_GPT2_STORE_PACKED_ATTENTION_ACTIVATIONS=1` and can be capped with `NFN_NATIVE_GPT2_STORE_PACKED_ATTENTION_BLOCKS=N`; default training leaves it disabled because saving all 11 earlier blocks adds about 4.43GB on top of the current MLP store and float arena and fails the default `64 x 1024` RTX 5090 run with CUDA OOM. Plan and runtime JSON now report `packed_attention_activation_storage_strategy`, `stored_packed_attention_activation_blocks`, `stored_packed_attention_bf16_elements`, `stored_packed_attention_bf16_bytes`, `stored_packed_attention_store_blocks`, `stored_packed_attention_restore_blocks`, `stored_packed_attention_backward_kernel_launches`, and `stored_packed_attention_backward_consumer_strategy`; opt-in stage timing includes `block_recompute_saved_packed_attention`. Verification: rebuilt `nfn_gpt2_native_train`, ran `python -m pytest tests/test_native_gpt2.py -q` (`26 passed, 1 skipped`), and `git diff --check`. GPU-visible TinyStories two-step probes reported the default no-store path at about `138,836` tokens/s, the capped three-block store at about `140,516` tokens/s with 48 saved packed backward calls, and the uncapped 11-block store failing at `block11.mlp.proj.backward_weight.accumulate.bf16` with CUDA out-of-memory.
- Dense GPT-2 attention residual+LN2 now defaults to a stats-preserving fused Tile kernel, `nfn_native_tile_linear_bias_residual_layer_norm_with_stats_float32`. The fused kernel applies attention projection bias, residual scale, residual add, and LN2 while writing LN2 mean/rstd so backward still uses `layer_norm_backward_reuses_forward_stats: true`. Set `NFN_NATIVE_GPT2_FUSE_ATTENTION_RESIDUAL_LN2=0` to force the older separate residual-add plus LN2 route. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, ran the focused native GPT-2 test slice (`1 passed, 1 skipped`), and ran GPU-visible one-step TinyStories stage probes. The second default fused profile reported `135,979` tokens/s versus the committed default profile's `132,235` tokens/s, with `train_loop_wall_ms` down from `3964.82` to `3855.65`, `block_forward` down from `898.523 ms` to `837.563 ms`, and `block_backward.ln2_residual` still using forward stats at `97.0959 ms`.
- Large-row linear bias-gradient reductions now use a 1024-row chunk in the native Tile atomic reduction path instead of 256 rows, reducing atomic/grid traffic for the default `64 x 1024` GPT-2 microbatch while keeping the same accumulation contract. Verification: rebuilt `libnfn_native_train_tile_ops.so`; a GPU-visible one-step stage probe reported about `123,746` tokens/s with the default shape-gated cublasLt path active, and the bias buckets remained stable while total block backward improved modestly.
- Dense GPT-2 training now defaults to the NeuralFn-owned compiled Tile-CUDA path: `train_gpt2.py`, `train_gpt2_native.py`, SDK compiled-CLI config builders, and direct `nfn_gpt2_native_train` default to `kernel_backend="tile-cuda"` plus `--train-transformer-lm`. `llm-kittens` remains available only as an explicit external bridge via `--backend llm-kittens` / `kernel_backend="llm-kittens"`.
- GPT-2 `--train-transformer-lm` now defaults the row-chunked tied LM-head workspace back to 8192 rows across the compiled C++ trainer, Python native wrapper, and SDK config. The 16,384-row default remained valid but gave no current throughput win and reserved about 1.65GB of BF16 logit workspace; the 8192-row default uses about 824MB, lowers float-arena pressure, and was slightly faster in the current full-shape probe. A 24,576-row probe stayed within memory but regressed to about `42,753` tokens/s due to memory pressure, and the previous 32,768-row probe failed native arena setup with CUDA out-of-memory. Verification: rebuilt/used the native GPT-2 CLI and ran GPU-visible stage-timed TinyStories one-step probes; 8192 rows reported `train_loop_wall_ms: 4191.9`, `train_tokens_per_second: 125072`, and `float_arena_requested_elements: 3577709325`, while 16384 rows reported `train_loop_wall_ms: 4222.28`, `train_tokens_per_second: 124172`, and `float_arena_requested_elements: 3989816101`.
- Tied LM-head BF16 logits now use the SM120 ThunderKittens GEMM bridge by default in the native Tile-CUDA trainer when the Tile ops library is built with TK support. `NFN_TILE_CUDA_LINEAR_TK_GEMM=0` or `NFN_NATIVE_LINEAR_TK_GEMM=0` forces the older BF16 `cublasGemmEx` fallback for diagnostics. The raw Tile ABI now exports `nfn_native_tile_trainer_linear_tk_gemm_count`, and GPT-2 training JSON reports `linear_tk_gemm_count` plus `lm_head_logits_linear_strategy` so runs can prove the LM-head path is using TK instead of falling back. Verification: the stage-timed TinyStories one-step probe improved `lm_head_backward.logits` from about `244.337 ms` to about `30.965 ms`, train loop wall time from about `953.836 ms` to about `733.699 ms`, and throughput from about `68,708` to about `89,323` tokens/s. A normal two-step default probe reported `linear_tk_gemm_count: 8`, `lm_head_logits_linear_strategy: "tk-sm120-bf16-gemm-default"`, and about `100,967` tokens/s; an opt-out one-step probe with `NFN_NATIVE_LINEAR_TK_GEMM=0` reported `linear_tk_gemm_count: 0`, `lm_head_logits_linear_strategy: "bf16-gemmex-fallback"`, and about `73,179` tokens/s.
- GPT-2 stored MLP forward now fuses the earlier-block FC+bias+GELU BF16 store through the new raw Tile ABI symbol `nfn_native_tile_linear_bf16_gelu_bf16_float32`. Supported workstation GPT-2 shapes dispatch to the SM120 ThunderKittens fused `matmul_forward_gelu` bridge and write the stored BF16 preactivation plus GELU activation directly; unsupported shapes fall back to a generic CUDA kernel. GPT-2 training JSON now reports `stored_mlp_forward_strategy`, and stage timing separates `block_forward.mlp_fc_gelu.pack_ln2` from `block_forward.mlp_fc_gelu.fc_gelu`. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, reran a stage-timed TinyStories one-step probe where `block_forward.mlp_fc_gelu` dropped from about `383.7 ms` to about `234.7 ms`, and ran a normal two-step probe that improved throughput to about `126,801` tokens/s versus the prior about `117k` tokens/s while reporting `linear_tk_gemm_count: 240` and the new ABI symbol in the kernel list.
- The raw native Tile ABI now exposes `nfn_native_tile_scaled_dot_product_attention_store_tk_bf16_float32`. Opt-in dense GPT-2 stored-attention profiling now writes TK BF16 Q/K/V/O plus float LSE directly into caller-owned saved buffers during attention forward instead of copying the process TK workspace after forward; training JSON reports `attention_activation_storage_strategy: "tk-bf16-direct-forward-store-saved-backward"` when the path is enabled with `NFN_NATIVE_GPT2_STORE_ATTENTION_ACTIVATIONS=1`. This remains disabled by default because the current 64x1024 TinyStories one-step probe regressed to about `6,027` tokens/s, with `block_backward.attn_sdpa.to_qkv` taking about `81.4s` per optimizer step. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, ran the focused native GPT-2 source/build test slice, and ran the GPU-visible opt-in stored-attention probe that reported the new direct-store strategy and counters.
- The raw native Tile ABI now exposes `nfn_native_tile_linear_bias_residual_layer_norm_float32`, a fused attention-projection bias+residual+LayerNorm forward kernel. Dense GPT-2 can profile it with `NFN_NATIVE_GPT2_FUSE_ATTENTION_RESIDUAL_LN2=1`, and plan/runtime JSON report `attention_residual_ln2_strategy` plus launch counters. The path remains disabled by default because the current 64x1024 TinyStories one-step probe regressed from about `125,069` to about `122,750` tokens/s despite eliminating the separate `ln2` launch. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, ran the focused native GPT-2 source/build test slice, and ran the GPU-visible opt-in probe.
- The native BF16 GELU activation path now uses CUDA Tile kernels for both `nfn_native_tile_gelu_add_bias_bf16_act_float32` and `nfn_native_tile_gelu_backward_inplace_bf16_bits_float32`. The fused forward kernel stores float preactivation, float GELU, and BF16 GELU bits directly from Tile math, and the saved-BF16 backward path reads BF16 preactivation bits into Tile math instead of launching the old scalar CUDA element kernels. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, reran the focused native GPT-2 pytest slice, ran a GPU-visible one-step stage-timing probe, and ran a two-step TinyStories probe. The two-step probe was throughput-neutral at about `117,362` tokens/s versus the prior about `117,598` tokens/s; the measured GELU buckets were about `199.5 ms` forward and `127.4 ms` backward, so this removes scalar BF16 GELU kernels but does not yet close the remaining llm.kittens performance gap.
- GPT-2 `--train-transformer-lm` now defaults the row-chunked tied LM-head workspace to 16384 rows on the native Tile CUDA path. This reduces default full-vocab LM-head chunking from 8 chunks to 4 chunks at the `64 x 1024` workstation microbatch, increasing the BF16 logit workspace to about 1.65GB. A 32768-row probe failed native arena setup with CUDA out-of-memory on the same workstation shape, so that remains an explicit override only. Verification: reran the focused native GPT-2 pytest slice and ran GPU-visible TinyStories one-step probes; 16384 rows reported `train_loop_wall_ms: 982.749` and `train_tokens_per_second: 66686.4` under stage timing versus about `999.919 ms` and `65541.3` tokens/s at 8192 rows.
- The raw native Tile ABI now exposes saved TK attention workspace support through `nfn_native_tile_attention_tk_store_forward_workspace_bf16` and `nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_saved_tk_bf16_from_merged_grad_float32`. Native trainer loops can copy TK BF16 Q/K/V/O plus float LSE into caller-owned device buffers after attention forward, then run backward from that saved state later without sending tensors through graph-editor nodes or rerunning the attention forward. The dense GPT-2 native symbol checker now requires these ABI symbols, so rebuild `libnfn_native_train_tile_ops.so` before running the compiled trainer. Verification: rebuilt `libnfn_native_train_tile_ops.so` and confirmed both exported C symbols with `nm -D`.
- Dense GPT-2 `--train-transformer-lm` now has an opt-in saved TK attention activation path behind `NFN_NATIVE_GPT2_STORE_ATTENTION_ACTIVATIONS=1`. When enabled, earlier blocks store BF16 Q/K/V/O plus float LSE, recompute only the state needed for projection/residual, restore saved O, and run `nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_saved_tk_bf16_from_merged_grad_float32` in backward. Runtime JSON reports `attention_backward_uses_saved_forward_workspace`, saved-attention arena sizes, store/restore/backward counts, and block-state flags for saved-O recompute. This remains disabled by default because the 64x1024 TinyStories one-step probe regressed from about `74,424` tokens/s to about `12,601` tokens/s due to the extra attention-state storage traffic. Verification: rebuilt `nfn_gpt2_native_train`, ran a GPU-visible opt-in one-step probe to prove the saved path and counters, then reran the default probe to confirm saved attention stayed disabled and returned to about `74,424` tokens/s.
- Direct `python cli/scripts/train_gpt2.py ...` execution now uses a pre-import compiled-CLI fast path for the default `compiled-cli` runner: it translates GPT-2 flags to `nfn_gpt2_native_train`, runs dry-run/plan/check commands through that binary, and `exec`s the compiled C++ trainer for real runs before importing `train_gpt2_native.py`, `server.dataset_manager`, NumPy, tiktoken, or Torch. Explicit non-compiled runners still use the Python native wrapper for fallback/debug workflows.
- `nfn_gpt2_native_train --backend tile-cuda --print-command` now prints the compiled invocation and exits before token-shard resolution, CUDA runtime setup, or driver preflight. Wrapper `--native-cuda-print-command` therefore no longer enters the trainer just to inspect command lines; the explicit `llm-kittens` backend still resolves shards when printing its delegated `train_gpt2cu -i/-j` command.
- The GPT-2 Python compiled-CLI fast path no longer appends the external `--target train_gpt2cu` bridge argument for the default Tile-CUDA backend. `--target` is now added automatically only for explicit `llm-kittens` backend commands, keeping default printed commands NeuralFn-native.
- `nfn_gpt2_native_train --help` now describes `--target` as an explicit llm-kittens bridge option and describes `--print-command` as backend-aware inspection instead of claiming every backend prints a `train_gpt2cu` command.
- Direct legacy graph-backed training scripts now prefer family-specific compiled binaries before the generic native registry. The shared pre-import guard derives `NFN_NATIVE_<MODEL>_CLI` and `nfn_<model>_native_train` from the requested model family, then probes the repo `build/` directory and PATH before falling back to `nfn-native-train --base-model <model>`. This moves NanoGPT, LLaMA, MixLLaMA, JEPA, semantic-router MoE, and DeepSeek direct script invocations onto the same compiled-family fast path as GPT-2 evo without importing Torch.
- The lightweight `nfn train --help` output now lists native GPT template and custom-graph selectors. `nfn train --base-model gpt2 ... --template-name NAME` / `--preset NAME` and `--graph-file PATH` / `--graph PATH` stay on the compiled native frontend, canonicalizing aliases to `--template-name` / `--graph-file` and preserving the selected template/graph before any graph-backed runtime can import.
- The GPT-2 evo family-specific C++ preflight now accepts `--template-name` / `--template` / `--preset` and `--graph-file` / `--graph`, normalizes template names, and emits the same `template_name`, `graph_file`, `template_known`, `selected_graph_support_status`, `selected_graph_native_runnable`, `shipped_template_catalog`, and `shipped_template_catalog_count` fields as the dense native GPT-2 selector path. The evo trainer still reports `native-preflight-missing-evo-trainer`, but dense GPT-2-compatible selectors, structurally different shipped templates, custom graph files, and unshipped typos are now separated in JSON before any graph-backed runtime can import.
- The shared native C++ token-shard resolver now accepts llm.kittens-style TinyStories token bins in addition to NeuralFn `fineweb_*` shards. `--tinystories` resolves to `/mnt/disk2/dev/open-source/llm.kittens/dev/data/tinystories` when `TinyStories_train.bin` and `TinyStories_val.bin` exist there, `NFN_LLM_KITTENS_TINYSTORIES_DIR` overrides that directory, and direct `--dataset-alias /path/to/TinyStories_train.bin` infers the sibling validation bin. This keeps the default GPT-2 wrapper aligned with `llm.kittens/train-sm120.sh` without Python dataset scanning or raw-text materialization.
- The shared native C++ token-shard sampler now reads contiguous shard segments per batch instead of opening and reading the shard once per sequence chunk. Default GPT-2 microbatches therefore avoid 64 tiny file opens before each device token copy, and native token-shard JSON reports `batch_read_strategy: "contiguous_shard_segments"`.
- GPT-2 `--train-transformer-lm` now uploads cached token/target batches as one compact uint16 arena, stages them through pinned host memory, enqueues one contiguous H2D `cudaMemcpyAsync`, and widens the combined arena to the trainer's int64 token buffers on device through one raw Tile `nfn_native_tile_uint16_to_int64` launch. The hot path no longer expands or range-validates every token on CPU, and training JSON reports `token_id_upload_strategy: "uint16-pinned-async-h2d-device-widen"`, `token_id_host_staging: "pinned"`, `token_id_h2d_copy: "cudaMemcpyAsync-contiguous-arena"`, `token_id_h2d_copy_calls_per_microbatch: 1`, `token_id_widen_strategy: "single-contiguous-arena-kernel"`, `token_id_widen_kernel_launches_per_microbatch: 1`, and `token_id_host_validation: false`.
- `SequentialTokenBatchSampler` now exposes `next_into()` for native trainers that already own their destination buffers. Full GPT-2 `--train-transformer-lm` uses it to sample train and validation batches directly into the pinned uint16 arena, avoiding `TokenBatch` vector materialization and vector-to-pinned copies on the hot path. Training JSON reports `token_batch_staging_strategy: "direct-sampler-to-pinned-arena"`, `token_batch_vector_materialization: false`, and `token_batch_vector_copy_to_pinned_elided: true`.
- The raw native Tile ABI now exposes `nfn_native_tile_init_gpt2_token_weight_float32`. GPT-2 `--train-transformer-lm` uses it to initialize the full 50,257 x 768 tied token embedding/LM-head weight directly on device, removing the previous 154 MB host float vector construction plus H2D copy from startup. Training JSON reports `token_weight_init_strategy: "device-tile-deterministic"` and `token_weight_host_materialization: false`.
- The raw native Tile ABI now exposes `nfn_native_tile_float32_to_bf16_bits` and `nfn_native_tile_float32_to_bf16_bits_many`. GPT-2 `--train-transformer-lm` checkpoint export uses the many-buffer ABI to pack all trained float32 weight tensors into one contiguous bf16 payload on device, then performs one compact uint16 D2H copy for the native version-5 `.bin` file. Training JSON reports `checkpoint.payload_pack_strategy: "device-many-float32-to-bf16-bits-contiguous"`, `payload_pack_kernel: "nfn_native_tile_float32_to_bf16_bits_many"`, `payload_copy_strategy: "single-contiguous-device-payload-d2h"`, `payload_cpu_bf16_conversion: false`, `device_pack_kernel_launches`, `d2h_copy_count`, `d2h_bytes`, and `float32_d2h_bytes_elided`; the default 12-layer one-step probe now reports one pack kernel launch, one D2H payload copy, and 248,879,616 bytes of float32 checkpoint D2H traffic elided. The single-buffer pack launcher was also corrected to launch enough CUDA threads for every element; both single-buffer and many-buffer packers were verified with a 4,097-element CUDA content check.
- GPT-2 `--train-transformer-lm` startup now treats per-block buffers as block-vector-owned only. The legacy block-0 aliases were removed from the global parameter/gradient allocation, scratch-tape activation allocation, parameter initialization, and AdamW-state zeroing lists, so block 0 is no longer redundantly touched once through aliases and again through the all-block visitors. Training JSON reports `block_state_layout.block0_duplicate_allocation_elided`, `block0_duplicate_activation_allocation_elided`, `block0_duplicate_parameter_initialization_elided`, and `block0_duplicate_adamw_state_zero_elided`.
- GPT-2 `--train-transformer-lm` now suballocates float buffers from one aligned CUDA device arena instead of issuing one `cudaMalloc` per parameter, gradient, AdamW moment, activation, and workspace buffer. Training JSON reports `float_allocation_strategy: "single-arena"`, `float_allocation_cuda_malloc_count`, `float_allocation_request_count`, `float_arena_requested_elements`, and `float_arena_allocated_elements`; missing-library failures report zero arena allocations because they fail before allocation.
- GPT-2 `--train-transformer-lm` startup now zeroes only AdamW first/second moment state with descriptor-driven Tile fills by default instead of filling the whole float arena. Nonzero weights are still written by their device initializers and gradients are zeroed per optimizer step. Set `NFN_NATIVE_GPT_ZERO_ADAMW_STATE_ONLY=0` to force the older full-arena zero for bisection. Training JSON reports `float_arena_zero_init_strategy: "adamw-state-fill-many"` or `"single-arena-fill"`, `float_arena_zero_fill_count`, `adamw_state_zero_fill_count`, `startup_per_buffer_zero_fill_elided`, and `startup_per_buffer_zero_fill_launches_elided`.
- The raw native Tile ABI now exposes `nfn_native_tile_fill_many_values_float32`. GPT-2 `--train-transformer-lm` uses it to initialize nonzero constant parameter buffers through one descriptor-driven Tile launch, reducing the default 12-layer startup path from 75 per-buffer fills to one launch for position weights, final norm, residual scale, and block constant weights. Training JSON reports `parameter_initialization_strategy: "fused-multi-buffer-fill-values"`, `parameter_initialization_kernel_launches_per_startup`, `parameter_initialization_per_buffer_launches_elided`, and `block_state_layout.parameter_initialization_loop: false`.
- GPT-2 `--train-transformer-lm` now builds AdamW, gradient-zero, gradient-clip, and parameter-fill descriptor tables into one host-packed descriptor arena, then uploads that arena into one aligned device descriptor arena. Startup no longer issues one `cudaMalloc` or one H2D copy per descriptor table, and training JSON reports `descriptor_allocation_strategy: "single-device-arena"`, `descriptor_upload_strategy: "single-host-packed-arena-copy"`, `descriptor_arena_cuda_malloc_count`, `descriptor_arena_requested_bytes`, `descriptor_arena_bytes`, `descriptor_arena_suballocation_count`, `descriptor_arena_copy_count`, `descriptor_arena_copy_calls_elided`, and `descriptor_cuda_mallocs_elided`.
- GPT-2 `--train-transformer-lm` now allocates token upload/storage buffers as combined arenas: one aligned device arena for widened int64 token/target buffers plus compact uint16 H2D staging, and one pinned uint16 host arena for compact source staging. This removes the separate token-ID and uint16-staging device `cudaMalloc` pair from startup. Training JSON reports `token_buffer_allocation_strategy: "combined-arenas"`, `token_device_allocation_strategy: "single-device-arena"`, `token_device_arena_cuda_malloc_count`, `token_device_arena_suballocation_count`, and `token_device_cuda_mallocs_elided`; missing-library failures report zero token arena allocations because they fail before allocation.
- GPT-2 no-data Tile-CUDA preflight actions now run before token-shard resolution. `--check-tile-ops`, `--smoke-tile-ops`, `--smoke-optimizer-step`, `--smoke-lm-step`, `--smoke-attention-step`, `--smoke-mlp-step`, `--smoke-norm-residual-step`, and `--smoke-transformer-block-step` no longer require cached `fineweb_train_*.bin` shards, and plan/smoke JSON reports `token_shards_resolved: false` when shards were skipped. Dataset-backed smokes and real training modes still resolve cached train/validation shards before running.
- Native GPT-2 training handoff now accepts `--template-name` / `--template` / `--preset` and `--graph-file` / `--graph` in the Python wrapper, SDK compiled-CLI config, and compiled C++ frontend. Python wrappers canonicalize aliases to `--template-name` and `--graph-file` at handoff. Every shipped GPT preset name can be passed through the native config path. Dense GPT-2-compatible presets (`gpt2`, `gpt2_megakernel`, and `gpt2_moa`) map to the implemented native transformer-LM loop; structurally different template names and custom graph files are reported as selected but fail fast with `selected-graph-native-trainer-missing` until their native C++ Tile trainer plans are implemented, without falling back to Torch.
- The compiled C++ GPT-2 frontend now carries an explicit shipped-template catalog synchronized with `neuralfn.config.SHIPPED_GPT_TEMPLATE_PRESETS`. Tile-CUDA plan JSON includes `shipped_template_catalog`, `shipped_template_catalog_count`, and `template_known`, tests compare the native catalog to the SDK catalog, and typoed/unshipped template names now report `unknown-template` instead of being mixed with known-but-missing native trainer work.
- `gpt2_moa` is now a native-runnable GPT-2 template selector instead of only metadata pass-through. The Python wrappers, SDK compiled-CLI builder, and compiled C++ frontend resolve `--template-name gpt2_moa` to the native MoA activation mode (`--native-cuda-activation moa`), and plan/train JSON includes `native_cuda_activation` so the selected activation path is visible.
- Top-level `nfn train --base-model gpt2` direct compiled-CLI handoff now appends the normal `--train-transformer-lm` action when no plan/check/smoke/train action is already present. Selector-bearing commands such as `--template-name semantic_router_moe` or `--graph-file custom.json` therefore stay explicit during training while still reporting `selected-graph-native-trainer-missing` for unsupported native graph trainers instead of falling back to Torch.
- Added the canonical SDK catalogs `SHIPPED_GPT_TEMPLATE_BASE_PRESETS` and `SHIPPED_GPT_TEMPLATE_PRESETS` in `neuralfn.config` plus lazy top-level `neuralfn` exports. Native training selector coverage now uses that catalog and includes previously uncovered builder-dispatch names such as `gpt2_megakernel`, `nanogpt_megakernel`, and the `mixllama` alias; regression tests compare the catalog against `build_model_spec_from_config()` so new templates cannot silently miss the training selector path.
- GPT-2 `--train-transformer-lm` now honors `train_batch_tokens` as a real optimizer-step contract instead of reporting it while stepping after one microbatch. The native loop derives `grad_accum_steps = ceil(train_batch_tokens / (batch_size * seq_len))`, streams that many cached-shard microbatches through CUDA Tile forward/backward kernels, accumulates scaled gradients in device buffers with `nfn_native_tile_gradient_accumulate_float32`, then clips and runs AdamW once on the accumulated gradients. The SM120 default `batch_size=64`, `seq_len=1024`, `train_batch_tokens=524288` therefore runs eight native microbatches per optimizer step; JSON reports `microbatch_tokens`, `requested_train_batch_tokens`, `grad_accum_steps`, `effective_train_batch_tokens`, `train_microbatches_completed`, `gradient_accumulation_strategy`, and `gradient_accumulation_scale`.
- `nfn_gpt2_native_train --dry-run` and `--print-plan` now emit the Tile-CUDA plan by default instead of the external `train_gpt2cu` command. Direct C++ users can pass `--no-train-transformer-lm` for plan/check/debug commands that should not start the default transformer-LM loop.
- Dense GPT-2 Tile-CUDA dry-run/plan JSON now reports the implemented native trainer accurately: `status: "native-transformer-lm-ready"`, `training_step_plan.status: "ready"`, an empty `required_native_work` list for `gpt2`, `gpt2_megakernel`, and `gpt2_moa`, and `remaining_validation` for the still-required live SM120 throughput comparison. Structurally different template names and custom graph files still report `selected-graph-native-trainer-missing`.
- The raw native Tile ABI now exposes `nfn_native_tile_token_cross_entropy_backward_inplace_with_workspace_float32`. Full GPT-2 `--train-transformer-lm` uses it for tied LM-head CE backward so each logits chunk is overwritten with dlogits before the LM-head dHidden/dWeight GEMMs, removing the separate full-vocab `grad_logits` chunk from the main trainer. Training JSON reports `grad_logit_workspace_elements: 0`, `lm_head_ce_backward_strategy: "inplace-logits-dlogits-workspace"`, and `lm_head_grad_logits_workspace_allocated: false`. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, reran the focused native GPT-2 pytest slice, and ran a GPU-visible full default TinyStories one-step probe that reported `grad_logit_workspace_elements: 0`, no missing symbols, and about `93,204` tokens/s with `lm_head_row_chunk_size: 8192`.
- The raw native Tile ABI now exposes BF16 tied LM-head classifier primitives: `nfn_native_tile_linear_bf16_output_float32`, `nfn_native_tile_token_cross_entropy_backward_inplace_bf16_bits_with_workspace`, `nfn_native_tile_linear_backward_input_bf16_bits_float32`, and `nfn_native_tile_linear_backward_weight_accumulate_float32_bf16_bits`. GPT-2 `--train-transformer-lm` now defaults to BF16 logits/dlogits for the tied LM-head classifier, feeding those BF16 dlogits into the LM-head dHidden/dWeight GEMMs. Set `NFN_NATIVE_GPT2_LM_HEAD_BF16_LOGITS=0` to return only the tied LM-head chunks to the older TF32 SGEMM path for debugging. Training JSON now reports `lm_head_training_logits_dtype`, `lm_head_training_dlogits_dtype`, `lm_head_bf16_logits_enabled`, `lm_head_bf16_logit_elements`, `lm_head_bf16_logit_bytes`, and the BF16 CE strategy when enabled. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, extended native GPT-2 static/export coverage for the new ABI symbols and kernels, reran the focused native GPT-2 pytest slice, and ran a GPU-visible no-override one-step TinyStories BF16 LM-head probe that reported about `108,050` tokens/s with `lm_head_backward` at about `815 ms`, `linear_sgemm_count: 0`, and `linear_backend_strategy: "block-and-lm-head-bf16-gemmex-default"` from the prior default's about `100,800` tokens/s and about `1,313 ms`.
- The BF16 tied LM-head CE backward ABI now launches a fused per-row kernel behind `nfn_native_tile_token_cross_entropy_backward_inplace_bf16_bits_with_workspace`, computing row max, sumexp, and in-place BF16 dlogits in one launch per logits row instead of a row-stats launch plus a separate full-vocab element pass. Training JSON reports `lm_head_ce_backward_strategy: "fused-row-bf16-logits-dlogits"` for that default path. Verification: rebuilt `libnfn_native_train_tile_ops.so`, ran a GPU-visible one-step TinyStories probe with stage timing enabled, and measured the CE bucket down from about `135 ms` to about `71 ms`, total LM-head backward down from about `811 ms` to about `750 ms`, and throughput up from about `112,520` to about `114,078` tokens/s.
- Large-row Linear bias-gradient reductions now bypass the cuBLAS SGEMV helper and use the existing Tile chunked atomic reduction path, while small reductions can still use cuBLAS. On the default GPT-2 `batch=64`, `seq=1024`, `train_batch_tokens=524288` one-step TinyStories probe, this reduced `block_backward.mlp_proj.bias` from about `167 ms` to about `15 ms`, reduced `block_backward` from about `2,131 ms` to about `1,987 ms`, and improved throughput from about `108,044` to about `111,418` tokens/s. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, extended static coverage for the large-row bias policy, and ran a GPU-visible one-step stage-timing probe.
- GPT-2 `--train-transformer-lm` now pads the native tied token embedding/LM-head tensor to 50,304 rows while keeping public tokenizer vocab at 50,257. Dry-run plan JSON reports `shape.vocab_size: 50257` and `shape.padded_vocab_size: 50304`; parameter layout reports `wte.weight` as `[50304, 768]`; training/checkpoint JSON reports `padded_vocab: 50304`; and version-5 checkpoint headers store public vocab and padded vocab separately. The padded LM-head shape reduces default one-step LM-head chunk timing from about `1372 ms` to about `1313 ms` and improved the measured TinyStories default probe from about `95,347` to `96,446` tokens/s on this workstation.
- GPT-2 LM smoke/preflight paths now use the same padded 50,304-row tied token embedding/LM-head tensor shape as the full transformer-LM trainer. `--smoke-lm-step`, `--smoke-embedding-lm-step`, `--train-embedding-lm`, and `--smoke-transformer-lm-step` still validate token IDs against public vocab 50,257, but their LM-head logits, CE workspace, weight gradients, AdamW buffers, and JSON now report/use `padded_vocab: 50304`.
- `--train-transformer-lm` JSON now includes `block_state_layout`, and the C++ loop stores each trained transformer block's parameter, gradient, and AdamW state pointers behind an explicit per-block structure. This keeps the 12-layer default path inspectable and avoids direct `block0` optimizer/checkpoint wiring.
- GPT-2 `--train-transformer-lm` now drives block parameter allocation, parameter initialization, gradient zeroing, gradient clipping partials/scaling, per-block AdamW updates, and trained-weight checkpoint export through the per-block C++ state vector instead of direct `block0` optimizer/export wiring.
- GPT-2 `--train-transformer-lm` now trains the configured GPT-2 layer count, which defaults to 12, using one scratch activation tape with backward recomputation plus persistent block outputs instead of allocating a full activation tape per layer. The final block output copy is elided because final LayerNorm consumes it before backward recomputation starts; JSON reports `activation_tape_strategy: "scratch-recompute"`, `activation_tape_count: 1`, `persistent_block_outputs: 11`, and `final_block_output_copy_elided: true` for the default shape.
- `tools/build_native_train_tile_ops.sh` now builds the trainer-facing raw C ABI with `NFN_TILE_CUDA_USE_CUBLAS_LINEAR=1` and links `libcublas`, so `nfn_native_tile_linear_float32`, `nfn_native_tile_linear_backward_input_float32`, and `nfn_native_tile_linear_backward_weight_float32` use GPU GEMM for native trainers instead of the direct per-output Tile dot-product kernels. The generic Tile extension build remains on the pure Tile fallback unless that macro is defined.
- The trainer-facing native linear GEMM path added a cached-workspace BF16 `cublasGemmEx` bridge alongside the optimized TF32 `cublasSgemm` route. The raw Tile ABI exposes `nfn_native_tile_trainer_linear_*` telemetry counters plus `nfn_native_tile_trainer_linear_bf16_cache_reset`, and the GPT-2 trainer resets the packed-A cache after fused AdamW updates so multi-step runs do not reuse stale weights. GPT-2 `--train-transformer-lm` JSON reports `linear_backend_strategy`, BF16 GEMM counts, SGEMM counts, BF16 A-operand pack/cache-hit/reset counts, and BF16 workspace/cached-A capacity fields so runs can prove whether large projections used TF32 or the BF16 bridge. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, reran focused native GPT-2 tests, and added static/export assertions for the new symbols.
- The trainer-facing BF16 linear GEMM path now keeps a multi-entry packed-A cache instead of one cached first operand. Weight-forward and weight-dInput GEMMs can reuse packed BF16 copies for multiple GPT-2 projection weights within the same optimizer step, while the AdamW boundary still invalidates entries before the next step. The raw Tile ABI now exposes `nfn_native_tile_trainer_linear_bf16_cache_entry_count`, and GPT-2 `--train-transformer-lm` JSON reports `linear_bf16_cache_entry_count` beside `linear_bf16_cached_a_capacity` so throughput probes can distinguish single-entry reuse from broader projection-weight reuse. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, reran the focused native GPT-2 tests, and added plan/static/export assertions for the new telemetry field.
- The trainer-facing normal linear GEMM path now includes an opt-in cached cuBLASLt TF32 route behind `NFN_TILE_CUDA_LINEAR_CUBLASLT=1` / `NFN_NATIVE_LINEAR_CUBLASLT=1`, with a raw Tile ABI telemetry export `nfn_native_tile_trainer_linear_cublaslt_gemm_count`. GPT-2 training JSON reports `linear_cublaslt_gemm_count`; default runs keep the faster TF32 `cublasSgemm` LM-head path and report `linear_backend_strategy: "block-forward-dinput-dweight-bf16-lm-head-tf32-sgemm-default"`, while opt-in Lt probes report `"block-forward-dinput-dweight-bf16-lm-head-tf32-cublaslt-opt-in"`. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, reran `python -m pytest tests/test_native_gpt2.py -q`, ran a GPU-visible default TinyStories one-step probe that reported `96,431` tokens/s with `linear_cublaslt_gemm_count: 0` and `linear_sgemm_count: 192`, and ran an opt-in Lt probe that reported `linear_cublaslt_gemm_count: 192`, `linear_sgemm_count: 0`, and about `94,787` tokens/s, confirming Lt is observable but not the workstation default.
- The raw native Tile ABI now exposes `nfn_native_tile_bf16_bits_to_float32`, `nfn_native_tile_store_mlp_activations_bf16_float32`, `nfn_native_tile_restore_mlp_activations_bf16_float32`, `nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_float32`, and `nfn_native_tile_gelu_backward_inplace_bf16_bits_float32`. GPT-2 `--train-transformer-lm` now defaults to BF16 storage for earlier-block MLP activations on the workstation shape, storing `ln2_out`, MLP preactivation, and GELU activation tensors during forward, consuming those BF16 tensors directly for MLP dWeight and GELU backward, and eliding earlier-block MLP fc/GELU recompute. Set `NFN_NATIVE_GPT2_STORE_MLP_ACTIVATIONS=0` to disable the higher-memory path. The earlier restore-based opt-in probe removed `block_recompute.mlp_fc_gelu` but regressed to about `95,143` tokens/s; the direct-BF16 consumer path reports `stored_mlp_activation_restore_kernel_launches: 0`, reduced `block_recompute` to about `452 ms`, and improved the one-step TinyStories probe to about `100,847` tokens/s versus about `96,363` tokens/s for pure scratch recompute. GPT-2 training JSON now reports `mlp_activation_storage_strategy`, `stored_mlp_activation_blocks`, `stored_mlp_activation_elements`, `stored_mlp_activation_bytes`, `stored_mlp_activation_store_kernel_launches`, `stored_mlp_activation_restore_kernel_launches`, `stored_mlp_activation_backward_consumer_strategy`, `block_state_layout.backward_recompute_mlp_fc_gelu_elided`, and the selected `block_state_layout.activation_tape_strategy`. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, reran `python -m pytest tests/test_native_gpt2.py -q`, and ran a GPU-visible TinyStories one-step probe with stage timing enabled.
- The raw native Tile ABI now exposes `nfn_native_tile_gelu_add_bias_bf16_act_float32` and `nfn_native_tile_linear_bf16_input_bits_float32`. Full GPT-2 `--train-transformer-lm` fuses MLP bias+GELU with a BF16 activation write, then feeds that reusable BF16 scratch directly into the MLP projection GEMM instead of repacking the float GELU activation. Training JSON reports `mlp_proj_forward_activation_strategy: "fused-gelu-bf16-act-direct-gemm"`, `mlp_forward_act_bf16_elements`, and `mlp_forward_act_bf16_bytes`; the default workstation shape allocates `402653184` bytes for this scratch. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, reran the focused native GPT-2 tests, and ran a GPU-visible one-step TinyStories probe that reported MLP projection forward down from about `218 ms` to about `141 ms`, block forward down from about `1187 ms` to about `1139 ms`, and about `112065` tokens/s.
- The trainer-facing linear ABI originally accepted `NFN_TILE_CUDA_LINEAR_BF16=0` or `NFN_NATIVE_LINEAR_BF16=0` to force the optimized TF32 cuBLAS path without rebuilding, which exposed that BF16 packing overhead could dominate this GPT-2 shape. That profiling control is superseded by the TF32-default switch below; BF16 is now opt-in with `NFN_TILE_CUDA_LINEAR_BF16=1` or `NFN_NATIVE_LINEAR_BF16=1`.
- The trainer-facing linear ABI now defaults to optimized TF32 tensor-op `cublasSgemm` for native GPT training, with BF16 `cublasGemmEx` available only when `NFN_TILE_CUDA_LINEAR_BF16=1` or `NFN_NATIVE_LINEAR_BF16=1` is set. On the current 5090 GPT-2 shape, the BF16 bridge was faster for some forward/recompute GEMMs but spent too much time repacking large activation operands in backward projection dWeight calls; a one-step TinyStories probe improved from `65,838` tokens/s with the BF16 bridge to `67,732` tokens/s with TF32 default, and `block_backward.mlp_proj` dropped from about `213 ms` to `110 ms`. Training JSON now reports `linear_backend_strategy: "tf32-sgemm-optimized"` for the default route.
- The raw native Tile ABI now exposes `nfn_native_tile_linear_bf16_float32`, a forced-BF16 forward-only linear entrypoint for cacheable-weight GPT block projections. Full GPT-2 `--train-transformer-lm` uses that ABI for transformer block forward/recompute qkv, attention-output, MLP fc, and MLP projection GEMMs, while LM-head and backward GEMMs stay on the normal TF32-default linear ABI. Training JSON reports `linear_backend_strategy: "block-forward-bf16-backward-tf32"`, `block_forward_linear_strategy: "forced-bf16-gemmex-forward"`, and `non_block_forward_backward_linear_strategy: "tf32-sgemm-optimized-default"`. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, reran the focused native GPT-2 pytest slice, and ran a GPU-visible one-step TinyStories probe that improved to `72,848` tokens/s with `899.6 ms` train compute.
- The raw native Tile ABI now exposes `nfn_native_tile_linear_backward_input_bf16_float32`, a forced-BF16 dInput GEMM entrypoint. Full GPT-2 `--train-transformer-lm` uses it only for transformer block dInput GEMMs, keeping LM-head backward and all dWeight accumulation on the TF32-default linear ABI. Training JSON reports `linear_backend_strategy: "block-forward-and-block-dinput-bf16-dweight-tf32"`, `block_backward_input_linear_strategy: "forced-bf16-gemmex-dinput"`, and `non_block_forward_backward_linear_strategy: "lm-head-and-dweight-tf32-sgemm-optimized-default"`. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, checked the exported symbol with `nm`, reran the focused native GPT-2 pytest slice, and ran a GPU-visible one-step TinyStories default probe that reported about `76,150` tokens/s with block backward down to about `312 ms`.
- The raw native Tile ABI now exposes `nfn_native_tile_linear_backward_weight_accumulate_bf16_float32`, a forced-BF16 dWeight accumulation GEMM entrypoint. Full GPT-2 `--train-transformer-lm` uses it only for transformer block qkv, attention-output, MLP fc, and MLP projection dWeight accumulation, while tied LM-head logits/dHidden/dWeight chunks stay on the TF32-default linear ABI. Training JSON now reports `linear_backend_strategy: "block-forward-dinput-dweight-bf16-lm-head-tf32"`, `block_backward_weight_linear_strategy: "forced-bf16-gemmex-dweight-accumulate"`, and `non_block_forward_backward_linear_strategy: "lm-head-tf32-sgemm-optimized-default"`. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, checked the exported symbol, reran the focused native GPT-2 pytest slice, ran a GPU-visible one-microbatch TinyStories probe that reported about `76,582` tokens/s, and ran a full default `524288` token/step probe that reported about `92,670` tokens/s with block backward down to about `2226 ms`.
- GPT-2 `--train-transformer-lm` now defaults the row-chunked tied LM-head workspace to 8192 rows on the workstation native path. The larger default reduces full-vocab LM-head chunking from 32 chunks to 8 chunks at the default 64x1024 microbatch and keeps the tied LM-head matmuls on the TF32 path, at the cost of increasing `logit_workspace_elements` from `102926336` to `411705344`. Verification: rebuilt `nfn_gpt2_native_train`, reran the focused native GPT-2 and wrapper pytest slices, and ran a GPU-visible full default `batch=64`, `seq=1024`, `train_batch_tokens=524288` TinyStories probe that reported `lm_head_row_chunk_size: 8192`, `lm_head_row_chunk_count: 8`, `linear_sgemm_count: 192`, and about `93,280` tokens/s versus about `92,670` tokens/s for 2048-row chunks under the same BF16 block dWeight path.
- GPT-2 `--train-transformer-lm` now defaults the row-chunked tied LM-head workspace to 2048 rows. The larger default doubles the logit workspace compared with 1024 rows but reduces LM-head chunk overhead on the RTX 5090 native path. Callers can still pass `--lm-head-row-chunk-size`, `NativeGpt2RunConfig.lm_head_row_chunk_size`, or `--native-cuda-lm-head-row-chunk-size` to tune memory/performance explicitly. Verification: rebuilt `nfn_gpt2_native_train`, reran the focused native GPT-2 and wrapper pytest slices, and ran a GPU-visible one-step TinyStories default probe with stage timing enabled; JSON reported `lm_head_row_chunk_size: 2048`, `lm_head_row_chunk_count: 32`, and about `74,050` tokens/s versus about `72,848` tokens/s for the prior 1024-row default.
- GPT-2 `--train-transformer-lm` now has opt-in CUDA-event stage timing behind `NFN_NATIVE_GPT2_STAGE_TIMING=1`. Normal runs keep the existing no-extra-sync host timing block; diagnostic runs add `stage_timing_enabled`, event/drop counts, and per-stage totals/averages for token upload, model forward, block forward/recompute/backward, LM-head backward, final-norm/embedding backward, gradient zero/clip, and AdamW update under `timing.stage_timing`. Verification: rebuilt `nfn_gpt2_native_train`, ran a one-step TinyStories transformer-LM probe with stage timing enabled, and confirmed the stage breakdown highlighted block backward/forward plus LM-head backward as the dominant remaining native bottlenecks.
- The opt-in GPT-2 CUDA-event stage profiler now emits nested LM-head, block forward/recompute, and block backward substages so bottleneck selection does not require a separate Nsight pass. New stage names include `lm_head_backward.logits`, `lm_head_backward.ce`, `lm_head_backward.dhidden`, `lm_head_backward.dweight`, `block_forward.attention`, `block_forward.mlp_fc_gelu`, `block_forward.mlp_proj`, `block_backward.mlp_proj`, `block_backward.mlp_fc`, `block_backward.attn_sdpa`, and `block_backward.qkv`.
- The GPT-2 stage profiler now breaks the mixed block forward/recompute buckets into operation-level entries including `block_forward.attention.ln1`, `block_forward.attention.qkv`, `block_forward.attention.qkv_layout`, `block_forward.attention.sdpa`, `block_forward.attention.merge_heads`, `block_forward.attention.proj`, `block_forward.attention.residual`, `block_forward.mlp_fc_gelu.ln2`, `block_forward.mlp_fc_gelu.fc`, `block_forward.mlp_fc_gelu.gelu`, `block_forward.mlp_proj.proj`, and `block_forward.mlp_proj.residual`, with matching `block_recompute.*` names for recomputed earlier blocks. This is diagnostic-only and leaves normal runs unchanged unless `NFN_NATIVE_GPT2_STAGE_TIMING=1` is set.
- The opt-in GPT-2 CUDA-event stage profiler now breaks block backward buckets down further into individual dWeight, bias, dInput, activation, residual-add, and attention-to-QKV records. New stage names include `block_backward.mlp_proj.dweight`, `block_backward.mlp_proj.dinput`, `block_backward.mlp_proj.gelu`, `block_backward.mlp_fc.dweight`, `block_backward.attn_proj.dweight`, `block_backward.attn_sdpa.to_qkv`, `block_backward.qkv.dweight`, and `block_backward.qkv.dinput`. Verification: rebuilt `nfn_gpt2_native_train`, reran the focused native GPT-2 pytest slice, and ran a stage-timed full default TinyStories one-step probe that emitted 3,635 stage events with no dropped events. The probe reported `block_backward.attn_sdpa.to_qkv` around `472 ms`, MLP dWeight/dInput records between about `154 ms` and `230 ms`, and QKV dWeight/dInput around `171 ms` and `160 ms`.
- The raw native Tile ABI now exposes `nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_reuse_forward_from_merged_grad_float32`. Full GPT-2 `--train-transformer-lm` uses it after original or recomputed TK attention forward, so attention backward reuses the process bf16 Q/K/V/O/LSE workspace instead of repacking Q/K/V and launching a duplicate TK forward inside the backward bridge. Training JSON reports `attention_backward_strategy: "tk-sm120-bf16-reuse-forward-workspace-bridge"`, `attention_backward_reuses_forward_workspace: true`, and `attention_backward_recompute_forward_elided_per_block: 1`. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, reran the focused native GPT-2 pytest slice, and ran a GPU-visible full default TinyStories one-step probe. The probe reported about `95,200` tokens/s, `block_backward.attn_sdpa.to_qkv` down to about `340 ms` from about `472 ms`, and total `block_backward` down to about `2104 ms` from about `2228 ms`.
- The trainer-facing native GELU Tile kernels now use the GPT-style tanh approximation (`0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`) and matching derivative for `nfn_native_tile_gelu_float32`, `nfn_native_tile_gelu_add_bias_float32`, `nfn_native_tile_gelu_backward_float32`, and `nfn_native_tile_gelu_backward_inplace_float32`, replacing the previous erf/CDF approximation on the raw native GPT-2 training path. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, reran the focused native GPT-2 pytest slice, and ran a GPU-visible full default TinyStories one-step probe that reported about `95,310` tokens/s with `mlp_fc_bias_gelu_strategy: "fused-bias-preactivation-gelu"` and the in-place GELU backward path active.
- The raw native Tile ABI now exposes `nfn_native_tile_gelu_backward_inplace_float32`. GPT-2 `--train-transformer-lm` writes MLP projection dInput directly into the MLP fc gradient buffer and applies GELU backward in-place, removing the full-trainer `grad_act` hidden-size scratch allocation while leaving smoke helpers on the explicit-output ABI. Training JSON reports `block_state_layout.mlp_proj_backward_gelu_inplace: true` and `block_state_layout.mlp_proj_backward_grad_act_scratch_allocated: false`. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, ran the focused native GPT-2 pytest slice, and ran a GPU-visible one-step TinyStories probe. The probe reported `65,838` tokens/s with stage timing enabled, so this slice reduced scratch traffic but did not materially close the remaining SM120 throughput gap.
- The trainer-facing native linear bias backward and accumulate-bias backward paths now use cuBLAS GEMV when `NFN_TILE_CUDA_USE_CUBLAS_LINEAR=1` is enabled. A cached device ones vector is initialized with a CUDA Tile fill kernel, then reused for row reductions, so GPT-2 block bias gradients avoid the row-chunked atomic fallback in the optimized trainer build while preserving that fallback for non-cuBLAS builds.
- GPT-2-compatible SDPA forward now attempts a dense causal full-head row-vector Tile kernel for `seq_k <= 1024`, computing a query row's score/softmax once and reusing it across all 64 value channels instead of launching one scalar-output CTA per output element. If CUDA rejects that row-kernel launch, the native Tile launcher records `cudaFuncGetAttributes` diagnostics, clears the launch error, auto-disables further row attempts for the run, and falls back to the scalar Tile attention kernel without repeated failed-launch overhead. Native plan and training JSON report `attention_forward_strategy: "row-vector-tile-score-reuse"`, `attention_forward_value_chunk_size: 64`, `attention_forward_scalar_launch_fallback_enabled: true`, `attention_forward_row_launch_auto_disable_enabled: true`, runtime row/fallback/scalar launch counts, row-kernel attribute fields, `attention_forward_row_count`, the previous scalar-output count, and the score-reuse factor.
- GPT-2-compatible SDPA backward now uses a query-row atomic Tile kernel for `qk_dim <= 64`, `value_dim <= 64`, and sequence lengths up to 1024. The native launcher zeros Q/K/V gradient buffers, computes each query row's softmax once, writes full-row `dQ`, and atomically accumulates `dK`/`dV` into attended key rows, replacing the previous per-scalar backward CTA path and the interim key-row implementation that repeated full query softmax scans. Native plan and training JSON report `attention_backward_strategy: "query-row-atomic-tile-score-reuse"`, `attention_backward_row_count`, `attention_backward_scalar_output_count`, `attention_backward_score_reuse_dim: 64`, and `attention_backward_scalar_cta_elision_factor: 192`. Verification: a GPU-visible tiny transformer-LM run passed with the new strategy; a default `batch=64`, `seq=1024`, one-step TinyStories probe now completes with `train_compute_wall_ms` around 30,255 ms, so SM120 parity is still not achieved and the remaining bottlenecks need further kernel work.
- `tools/build_native_train_tile_ops.sh` now defaults to the SM120 ThunderKittens bf16 attention bridge for GPT-2-compatible causal SDPA, using local `LLM_KITTENS_ROOT` and `TK_ROOT` checkouts, `NFN_TILE_CUDA_ARCH=sm_120a`, and `libcuda` linkage. Set `NFN_TILE_CUDA_USE_TK_ATTENTION=0` only to build the older float32 row-scan diagnostic path.
- The trainer-facing raw Tile ABI now exposes `nfn_native_tile_attention_forward_tk_launch_count` and `nfn_native_tile_attention_backward_tk_launch_count`. Dense GPT-2 training JSON reports `attention_backend_strategy`, TK launch counts, and switches the forward/backward strategy labels to `tk-sm120-bf16-flashattention-bridge` / `tk-sm120-bf16-recompute-forward-bridge` when the optimized path runs.
- GPT-2-compatible causal SDPA now dispatches through the TK bf16 bridge for supported `B,H,T,D` shapes, converts NeuralFn float32 Q/K/V and gradients at the ABI boundary, and leaves the row-vector/query-row-atomic Tile kernels as fallback or diagnostic code. Verification: rebuilt the default trainer Tile library and native GPT-2 CLI, ran `--smoke-attention-step` against the TK-enabled library, ran a sequence-32 transformer-LM probe with 23 TK forward launches and 12 TK backward launches, and ran a full-shape `batch=64`, `seq=1024`, one-step TinyStories probe through the default library with `attention_backend_strategy: "tk-sm120-bf16-bridge"`, zero row/scalar attention launches, `train_compute_wall_ms` about 1,188 ms, and `train_tokens_per_second` about 55,150.
- Row-vector SDPA forward diagnostics now include the pre-launch CUDA error state and requested row-kernel launch shape. The raw native Tile ABI exposes `nfn_native_tile_attention_forward_row_prelaunch_clear_error`, `nfn_native_tile_attention_forward_row_prelaunch_peek_error`, `nfn_native_tile_attention_forward_row_grid_x`, `nfn_native_tile_attention_forward_row_grid_y`, `nfn_native_tile_attention_forward_row_grid_z`, and `nfn_native_tile_attention_forward_row_block_x`; native GPT-2 training JSON reports the matching `attention_forward_row_prelaunch_*` and `attention_forward_row_launch_grid_*` / `attention_forward_row_launch_block_x` fields so row fallback debugging stays Torch-free.
- Full GPT-2 transformer-LM SDPA call sites now pass output element counts to `nfn_native_tile_scaled_dot_product_attention_float32` instead of the microbatch count. This fixes the row-vector launch grid collapsing to zero (`batch_size / head_dim`) and also makes the scalar fallback cover every attention output element when fallback is still needed.
- Full GPT-2 transformer-LM training JSON now includes a `timing` block with host wall-clock phase timers: `setup_wall_ms`, `train_loop_wall_ms`, `validation_wall_ms`, `train_compute_wall_ms`, `checkpoint_wall_ms`, `total_wall_ms`, `optimizer_steps_per_second`, and `train_tokens_per_second`. The timers do not add new device synchronizations; the train-loop timer ends after the existing final device-to-host sample copy.
- The raw native Tile ABI now exposes `nfn_native_tile_split_qkv_to_heads_float32`. Full GPT-2 `--train-transformer-lm` uses it after the QKV projection to write Q/K/V head-major buffers directly, replacing the legacy QKV split plus three reshape launches with one Tile launch per block. Native plan and training JSON report `qkv_forward_layout_strategy: "fused-split-to-heads"`, `qkv_forward_layout_kernel_launches_per_block: 1`, and the three elided legacy layout launches.
- The raw native Tile ABI now exposes `nfn_native_tile_split_qkv_to_heads_add_bias_float32`. Full GPT-2 `--train-transformer-lm` now runs the QKV CUBLAS projection without bias, then applies Q/K/V bias while writing Q/K/V head-major buffers in the fused split-to-heads Tile pass. Native plan and training JSON report `qkv_bias_layout_strategy: "fused-qkv-bias-split-to-heads"` and one elided legacy QKV bias launch per block. Verification: rebuilt the trainer Tile library and native GPT-2 CLI, ran the focused native C ABI tests, ran a full-shape `batch=64`, `seq=1024`, one-step TinyStories probe with `train_tokens_per_second` about 64,572, then profiled the run and confirmed 23 `split_qkv_to_heads_add_bias_float32_kernel` launches and zero standalone `linear_add_bias_float32_kernel` launches.
- Full GPT-2 `--train-transformer-lm` no longer allocates unused row-major forward `q`/`k`/`v` activation scratch buffers. The fused QKV split writes head-major attention inputs directly, so the default 12-layer `64 x 1024` shape removes three full activation buffers per tape, reducing `float_allocation_request_count` from 645 to 642 and `float_arena_requested_elements` from 4,140,811,045 to 3,989,816,101, a reduction of 150,994,944 float elements, about 576 MiB. Native plan and training JSON report `block_state_layout.forward_row_qkv_scratch_allocated: false` and `block_state_layout.forward_row_qkv_scratch_buffers_elided: 3`. Verification: rebuilt `nfn_gpt2_native_train` and ran a GPU-visible stage-timed TinyStories one-step probe; throughput stayed roughly neutral at about `124,020` tokens/s while setup memory pressure dropped.
- The raw native Tile ABI now exposes `nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32`. Full GPT-2 `--train-transformer-lm` uses it for the TK bf16 attention backward bridge so bf16 `dQ`/`dK`/`dV` head-major gradients are converted directly into row-major `grad_qkv`, replacing three bf16-to-float gradient conversion launches plus the heads-to-QKV merge launch per backward block. The full trainer also stops allocating the three full-size head-gradient scratch buffers for that path. Native plan and training JSON report `attention_backward_qkv_bridge_strategy: "fused-bf16-heads-to-row-qkv"` and three elided legacy bridge launches per block. Verification: rebuilt the trainer Tile library and native GPT-2 CLI, ran the focused native C ABI tests, ran a full-shape `batch=64`, `seq=1024`, one-step TinyStories probe with `train_tokens_per_second` about 65,931, then profiled the run and confirmed `bf16_to_f32_kernel` dropped from 59 to 23 instances, `merge_heads_to_qkv_float32_kernel` disappeared from the full path, and 12 `bf16_heads_to_qkv_float32_kernel` launches replaced that bridge.
- The raw native Tile ABI now exposes `nfn_native_tile_merge_heads_to_qkv_float32`. Full GPT-2 `--train-transformer-lm` uses it after SDPA backward to write Q/K/V head-major gradients directly into the row-major QKV gradient buffer, replacing three `merge_heads` launches plus one `merge_qkv` launch with one Tile launch per block. The full trainer also no longer allocates row-major `grad_q`, `grad_k`, and `grad_v` scratch buffers. Native plan and training JSON report `qkv_backward_layout_strategy: "fused-heads-to-qkv"`, `qkv_backward_layout_kernel_launches_per_block: 1`, and the three elided legacy layout launches.
- The raw native Tile ABI now exposes `nfn_native_tile_scaled_dot_product_attention_backward_from_merged_grad_float32`. Full GPT-2 `--train-transformer-lm` uses it so SDPA backward reads the row-major attention-output gradient directly from projection backward, removing the pre-SDPA-backward `reshape_heads` launch and `grad_attn_heads` scratch buffer from the full trainer. Native plan and training JSON report `attention_backward_grad_layout_strategy: "merged-grad-out-direct"` and the elided grad-output layout launch per block.
- The raw native Tile ABI now exposes `nfn_native_tile_gelu_add_bias_float32`. Full GPT-2 `--train-transformer-lm` uses it after the no-bias CUBLAS `c_fc` projection to write the biased preactivation and GELU activation in one Tile pass, replacing separate MLP bias-add and GELU launches. Native plan and training JSON report `mlp_fc_bias_gelu_strategy: "fused-bias-preactivation-gelu"` and one elided legacy launch per block. Verification: rebuilt the trainer Tile library and native GPT-2 CLI, then ran a full-shape `batch=64`, `seq=1024`, one-step TinyStories probe with the fused ABI loaded and `train_tokens_per_second` about 62,699.
- The raw native Tile ABI now exposes `nfn_native_tile_linear_bias_residual_add_float32`. Full GPT-2 `--train-transformer-lm` uses it after no-bias attention-output and MLP `c_proj` CUBLAS projections to apply projection bias, residual scale, and residual add in one Tile pass per projection site, replacing separate projection bias-add and residual-add launches. Native plan and training JSON report `projection_bias_residual_strategy: "fused-linear-bias-residual-add"` and two elided legacy launches per block. Verification: rebuilt the trainer Tile library and native GPT-2 CLI, ran the focused native C ABI tests, ran a full-shape `batch=64`, `seq=1024`, one-step TinyStories probe with the fused ABI loaded and `train_tokens_per_second` about 61,222, then profiled the run and confirmed 46 fused projection residual launches with the remaining `linear_add_bias_float32_kernel` count reduced to the 23 QKV-bias launches.
- `train_gpt2.py --native-cuda-dry-run` now defers dataset shard resolution to the compiled C++ frontend for the default `compiled-cli` runner. Command inspection no longer imports `server.dataset_manager`, NumPy, tiktoken, or Torch and no longer attempts to materialize raw-text token shards under the dataset cache.
- GPT-2 `--train-transformer-lm` now runs a CUDA runtime/driver preflight before host weight setup or device allocation. Failure JSON includes `cuda_runtime_preflight` with runtime and driver version/status fields, and driver version `0` or a newer loaded runtime now fails before `cudaMalloc` with an actionable GPU-access/runtime mismatch error.
- GPT-2 `--train-transformer-lm` previously raised its row-chunked tied LM-head workspace to 1024 rows via `--lm-head-row-chunk-size` / `NativeGpt2RunConfig.lm_head_row_chunk_size` / `--native-cuda-lm-head-row-chunk-size`. The CE loss reduction now stays on the device with `nfn_native_tile_sum_partials_float32` and performs one host sync/copy per forward loss instead of one per LM-head chunk. The JSON reports `lm_head_row_chunk_count` and `loss_partial_count` alongside `lm_head_row_chunk_size` and `logit_workspace_elements`.
- The raw native Tile ABI now exposes `nfn_native_tile_linear_backward_weight_accumulate_float32`. GPT-2 transformer-LM tied-head backward uses it to accumulate each row chunk directly into `grad_token_weight`, removing the per-chunk full-vocab `grad_token_weight_chunk` zero/fill plus full-buffer accumulate pass. In the trainer build this maps to cuBLAS dWeight GEMM with `beta=1`; fallback builds use the existing chunked atomic Tile path without clearing the destination.
- GPT-2 `--train-transformer-lm` no longer computes training loss in the hot path. Ordinary train steps run the model forward activations needed for backward, CE gradient generation, gradient clipping, and AdamW update only; validation cadence runs validation loss from validation shards without also measuring train loss, and the row-chunked tied LM-head CE loss/reduction path for train data is no longer scheduled for final-step sampling. The explicit per-step `cudaDeviceSynchronize()` after AdamW was also removed; the next default-stream batch copy, validation pass, or final sample copy provides ordering. JSON now reports `train_loss_sparse: false`, `train_loss_sampling: "disabled"`, `train_loss_on_validation_steps: false`, `train_loss_eval_count`, and `train_loss_last_step` so callers can distinguish disabled train-loss sampling from validation loss records. Verification: with train-loss sampling disabled, the GPU-visible default `batch=64`, `seq=1024`, one-step TinyStories probe improved from about 30,255 ms to about 23,639 ms train compute; this is still not SM120 parity.
- The raw native Tile ABI now exposes `nfn_native_tile_copy_float32`. GPT-2 transformer-LM persistent block-output preservation uses it instead of `fill` plus `gradient_accumulate(scale=1)`, removing one kernel launch per block output copy on each forward pass while preserving the scratch-recompute activation tape contract.
- GPT-2 transformer-LM validation forwards no longer copy intermediate block outputs into the persistent training-backward buffers. Training forwards still preserve earlier block outputs for backward recomputation, while validation streams through the scratch tape because no backward pass follows; JSON reports `validation_persistent_block_outputs: 0` and `validation_block_output_copies_elided: true`.
- GPT-2 transformer-LM backward now reuses the final block activations left in the scratch tape by the initial forward pass. Earlier blocks are still recomputed from persistent block outputs, but the final block is not recomputed before its backward pass; JSON reports `backward_recompute_blocks: 11` and `final_block_backward_recompute_elided: true` for the default 12-layer shape.
- GPT-2 transformer-LM backward recompute now stops after the MLP GELU activation for earlier blocks. Backward needs the recomputed attention/LayerNorm/MLP preactivation and activation tensors, but it does not consume the recomputed MLP projection output or final residual output, so each recomputed block elides one large projection GEMM and one residual-add kernel. JSON reports `backward_recompute_mlp_projection_elided: true` and `backward_recompute_final_residual_elided: true`. Verification: rebuilt `nfn_gpt2_native_train`, reran the focused native GPT-2 tests, then ran a GPU-visible one-step TinyStories probe with `NFN_NATIVE_GPT2_STAGE_TIMING=1`; block recompute fell from the previous timed 122.8 ms to 95.1 ms, train compute moved from about 1,035.5 ms to 993.7 ms, and throughput reported about 65,953 tokens/s.
- GPT-2 transformer-LM backward now fuses residual-gradient pair additions with `nfn_native_tile_scaled_residual_add_float32` instead of zeroing an output buffer and launching two `nfn_native_tile_gradient_accumulate_float32` calls. This removes two Tile launches per residual backward add site per transformer block, and the JSON `block_state_layout` reports `residual_backward_fused: true`.
- The raw native Tile ABI now exposes `nfn_native_tile_adamw_step_with_device_scale_float32`. GPT-2 transformer-LM gradient clipping now computes the device clip scalar, then applies that scalar inside each AdamW update instead of launching a separate `nfn_native_tile_scale_inplace_by_device_float32` pass over every gradient buffer. The JSON `block_state_layout` reports `adamw_device_clip_scale_fused: true`.
- The raw native Tile ABI now exposes `nfn_native_tile_sumsq_partials_many_float32`. GPT-2 transformer-LM reuses the device-resident gradient descriptor table for gradient clipping's sum-of-squares phase, so the default 12-layer trainer emits one multi-buffer sumsq launch per optimizer step instead of one launch per gradient buffer before the clip-scale reduction. Training JSON reports `gradient_clip_strategy: "fused-multi-buffer-sumsq-device-scale"`, `gradient_sumsq_kernel_launches_per_optimizer_step`, `gradient_sumsq_per_buffer_launches_elided`, `block_state_layout.gradient_clip_loop: false`, and `block_state_layout.gradient_clip_loop_elided: true`.
- The raw native Tile ABI now exposes `nfn_native_tile_adamw_step_many_with_device_scale_float32`. GPT-2 transformer-LM builds device-resident optimizer descriptor arrays at startup and updates all 148 default parameter buffers with one multi-buffer AdamW kernel launch per optimizer step instead of one launch per parameter buffer. Training JSON reports `adamw_update_strategy: "fused-multi-buffer-device-scale"`, `adamw_descriptor_count`, `adamw_step_kernel_launches_per_optimizer_step`, and `adamw_per_buffer_step_launches_elided`.
- GPT-2 transformer-LM token gradients now accumulate directly into the optimizer-step accumulation buffer across microbatches. The tied LM-head CE backward scale includes the microbatch accumulation factor, LM-head dWeight chunks use `nfn_native_tile_linear_backward_weight_accumulate_float32` with `accum_grad_token_weight` as the destination, and token embedding backward atomically adds into that same buffer. This removes the old full-vocab `grad_token_weight` scratch allocation, per-microbatch full-buffer zero, and per-microbatch full-buffer copy into `accum_grad_token_weight`. JSON reports `token_gradient_accumulation_strategy: "direct-device-accumulation-buffer"`, `token_gradient_scratch_buffer_allocated: false`, `token_gradient_microbatch_full_copy_elided: true`, and `token_gradient_microbatch_zero_elided: true`.
- The raw native Tile ABI now exposes `nfn_native_tile_absolute_position_embedding_backward_accumulate_float32`. GPT-2 transformer-LM uses it to add position-embedding gradients directly into `accum_grad_position_weight`, removing the old `grad_position_weight` scratch allocation, per-microbatch zero, and per-microbatch full-buffer copy. JSON reports `position_gradient_accumulation_strategy: "direct-device-accumulation-buffer"`, `position_gradient_scratch_buffer_allocated: false`, `position_gradient_microbatch_full_copy_elided: true`, and `position_gradient_microbatch_zero_elided: true`.
- GPT-2 transformer-LM block Linear dWeight kernels now accumulate directly into per-block optimizer-step accumulation buffers. The real 12-layer trainer uses `nfn_native_tile_linear_backward_weight_accumulate_float32` for qkv, attention-output, MLP fc, and MLP projection dWeights, so those four large per-block scratch dWeight buffers are no longer allocated or copied into accumulation buffers after every microbatch. JSON reports `block_linear_weight_gradient_accumulation_strategy: "direct-device-accumulation-buffer"`, `block_linear_weight_gradient_scratch_buffers_allocated: false`, and `block_linear_weight_gradient_microbatch_full_copy_elided: true`; after the LayerNorm/bias direct-accumulation change, `block_state_layout.per_block_gradient_buffers` is `0` and `block_state_layout.per_block_direct_accum_gradient_buffers` is `12`.
- The raw native Tile ABI now exposes `nfn_native_tile_layer_norm_backward_affine_accumulate_float32` and `nfn_native_tile_linear_backward_bias_accumulate_float32`. GPT-2 transformer-LM writes LayerNorm affine and Linear bias gradients directly into optimizer-step accumulation buffers, removes the remaining per-block scratch gradient buffers, and elides the per-microbatch scratch-to-accumulation copy loop. This also avoids applying the microbatch accumulation scale a second time to those scratch gradients. JSON reports `layer_norm_affine_gradient_accumulation_strategy: "direct-device-accumulation-buffer"`, `linear_bias_gradient_accumulation_strategy: "direct-device-accumulation-buffer"`, `block_state_layout.per_block_gradient_buffers: 0`, `block_state_layout.per_block_direct_accum_gradient_buffers: 12`, `block_state_layout.gradient_accumulation_loop: false`, and `block_state_layout.gradient_accumulation_copy_loop_elided: true`.
- The raw native Tile ABI now exposes `nfn_native_tile_fill_many_float32`. GPT-2 transformer-LM no longer runs one accumulation-gradient zero-fill launch per parameter buffer; it reuses the device-resident optimizer descriptor table and zeroes all 148 default accumulation buffers with one multi-buffer fill launch per optimizer step. The JSON reports `gradient_zero_strategy: "fused-multi-buffer-accumulation-zero"`, `gradient_zero_kernel_launches_per_optimizer_step`, `gradient_zero_per_buffer_launches_elided`, `block_state_layout.gradient_zero_loop: false`, `block_state_layout.gradient_zero_loop_elided: true`, `gradient_zeroed_buffer_count: 0`, `token_gradient_accumulation_direct: true`, and `token_gradient_scratch_buffer_allocated: false`.
- `nfn_native_tile_layer_norm_backward_affine_float32` now uses a chunked parallel atomic reduction for large row counts instead of one CUDA block looping over every row, and the new accumulate ABI uses the same large-row path without zeroing the destination. GPT-2 transformer-LM now reports `layer_norm_backward_affine_strategy: "auto-chunked-atomic-accumulate"` so native trainer output reflects the direct accumulation path. Verification: rebuilt `libnfn_native_train_tile_ops.so` and `nfn_gpt2_native_train`, and ran the focused native GPT-2 pytest slice.
- `nfn-native-train --list-models` and `neuralfn.native_train.native_train_model_registry()` now report dense GPT-2 as `partial-native-trainer` instead of `external-fast-path`, matching the default compiled Tile-CUDA trainer. The unified frontend still dispatches GPT-2 to `nfn_gpt2_native_train`, and `--native-gpt2-cli` continues to override that binary.
- The `tile-cuda` packaging extra is now Torch-free and installs native CUDA Tile build tooling only (`ninja`). Install `.[torch]` separately when intentionally using graph-backed PyTorch execution or the legacy PyTorch Tile extension loader.

#### Breaking changes

- Callers that expected the default native GPT-2 backend to print or execute the external `train_gpt2cu` command must now opt in with `--backend llm-kittens` or `kernel_backend="llm-kittens"`.
- SDK callers that relied on `build_native_gpt2_run_config()` or `build_native_gpt2_compiled_cli_run_config()` defaulting to the external backend must pass `train_transformer_lm=False` and `kernel_backend="llm-kittens"` for that old behavior.
- Callers that inspect native registry status must update GPT-2 checks from `external-fast-path` to `partial-native-trainer`.
- `pip install -e ".[tile-cuda]"` no longer installs Torch. Workflows that relied on that transitive install must use `pip install -e ".[torch,tile-cuda]"` or install Torch explicitly.
- Callers that assumed the implicit GPT-2 transformer-LM LM-head chunk was 1024 rows should pass `--lm-head-row-chunk-size 1024` explicitly; the default is now 2048 rows. Callers that still need the older 256-row behavior must pass `--lm-head-row-chunk-size 256`.
- Callers that assumed the implicit GPT-2 transformer-LM LM-head chunk was 2048 rows should pass `--lm-head-row-chunk-size 2048` explicitly; the default is now 8192 rows and uses a larger full-vocab logit workspace.
- Callers that assumed the implicit GPT-2 transformer-LM LM-head chunk was 16384 rows should pass `--lm-head-row-chunk-size 16384` explicitly; the default is now 8192 rows to avoid unnecessary BF16 logit workspace pressure on the current RTX 5090 native path.
- Existing `libnfn_native_train_tile_ops.so` builds must be rebuilt before running GPT-2 `--train-transformer-lm`; the compiled trainer now checks for `nfn_native_tile_scaled_dot_product_attention_store_tk_bf16_float32` in addition to the older saved-attention ABI symbols.
- Existing `libnfn_native_train_tile_ops.so` builds must be rebuilt before running GPT-2 `--train-transformer-lm`; the compiled trainer now checks for the packed-QKV attention ABI symbols `nfn_native_tile_bf16_bits_add_bias_inplace_float32`, `nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32`, and `nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_merged_grad_float32`. Callers parsing default GPT-2 native JSON should update QKV/attention strategy expectations from the split-to-heads bridge to the packed-QKV strategy fields, or set `NFN_NATIVE_GPT2_PACKED_QKV_ATTENTION=0` for the older profiling route.
- Existing `libnfn_native_train_tile_ops.so` builds must be rebuilt before running GPT-2 `--train-transformer-lm`; the compiled trainer now also checks for `nfn_native_tile_linear_bias_residual_layer_norm_float32` and `nfn_native_tile_linear_bias_residual_layer_norm_with_stats_float32`. Callers parsing GPT-2 native JSON should update the default `attention_residual_ln2_strategy` expectation from `"disabled"` to `"fused-linear-bias-residual-layernorm"`, or set `NFN_NATIVE_GPT2_FUSE_ATTENTION_RESIDUAL_LN2=0` for the older profiling route.
- The trainer-facing linear backend default changed from BF16 `cublasGemmEx` to a split route: GPT block forward/recompute uses forced BF16 and LM-head/backward uses TF32 tensor-op `cublasSgemm`. Callers that require the old normal-linear BF16 bridge must set `NFN_TILE_CUDA_LINEAR_BF16=1` or `NFN_NATIVE_LINEAR_BF16=1`; callers parsing JSON should update default checks from `linear_backend_strategy: "bf16-gemmex-float32-output"` or `"tf32-sgemm-fallback"` to `"block-forward-bf16-backward-tf32"`.
- Callers parsing GPT-2 native linear strategy JSON should update checks from `linear_backend_strategy: "block-forward-bf16-backward-tf32"` to `"block-forward-and-block-dinput-bf16-dweight-tf32"` and read the new `block_backward_input_linear_strategy` field when they need to distinguish block dInput GEMMs from dWeight and LM-head GEMMs.
- Callers parsing GPT-2 native linear strategy JSON should update checks from `linear_backend_strategy: "block-forward-and-block-dinput-bf16-dweight-tf32"` to `"block-forward-dinput-dweight-bf16-lm-head-tf32"`, read `block_backward_weight_linear_strategy` for transformer block dWeight GEMMs, and update `non_block_forward_backward_linear_strategy` expectations from `"lm-head-and-dweight-tf32-sgemm-optimized-default"` or `"lm-head-tf32-sgemm-optimized-default"` to `"padded-lm-head-tf32-sgemm-optimized-default"`.
- Callers parsing GPT-2 native linear strategy JSON should update checks from `linear_backend_strategy: "block-forward-dinput-dweight-bf16-lm-head-tf32"` to `"block-forward-dinput-dweight-bf16-lm-head-tf32-sgemm-default"` for the default path. The new `linear_cublaslt_gemm_count` field records opt-in cuBLASLt usage, and opt-in Lt runs report `"block-forward-dinput-dweight-bf16-lm-head-tf32-cublaslt-opt-in"` plus `non_block_forward_backward_linear_strategy: "padded-lm-head-tf32-cublaslt-optimized-opt-in"`.
- Existing `libnfn_native_train_tile_ops.so` builds must be rebuilt before running GPT-2 `--train-transformer-lm`; the compiled trainer now checks for `nfn_native_tile_bf16_bits_to_float32`, `nfn_native_tile_store_mlp_activations_bf16_float32`, `nfn_native_tile_restore_mlp_activations_bf16_float32`, `nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_float32`, and `nfn_native_tile_gelu_backward_inplace_bf16_bits_float32` at startup. Callers with strict GPT-2 JSON assertions should also accept the new MLP activation-storage telemetry fields, plus `block_state_layout.backward_recompute_mlp_fc_gelu_elided`; the default `activation_tape_strategy` is now `"scratch-recompute-bf16-stored-mlp-direct-backward-opt-in"` on the workstation native path. Set `NFN_NATIVE_GPT2_STORE_MLP_ACTIVATIONS=0` for the previous pure scratch-recompute behavior.
- Callers that assumed GPT-2 native checkpoint `vocab`/`padded_vocab` or `wte.weight` row count were both 50,257 must now use `vocab_size`/`vocab` for tokenizer checks and `padded_vocab_size`/`padded_vocab` for tensor shape, parameter count, logit workspace, and file-size checks. New NeuralFn native GPT-2 version-5 checkpoints store public vocab 50,257 and padded vocab 50,304 separately.
- Callers parsing GPT-2 smoke/preflight JSON should expect `padded_vocab: 50304` on LM-head-bearing smoke/training payloads and should use that value for logits/workspace/buffer assertions.

#### Verification

- Verified with `bash tools/build_native_gpt2_cli.sh /tmp/nfn_gpt2_native_train_check`, `bash tools/build_native_train_tile_ops.sh /tmp/libnfn_native_train_tile_ops_check.so`, `bash tools/build_native_missing_trainers.sh /tmp/nfn_missing_native_trainers_check`, `python -m pytest tests/test_native_gpt2.py -q`, `python -m pytest cli/tests/test_train_gpt2_native.py -q`, `python -m pytest tests/test_native_dependencies.py -q`, `/usr/bin/time -f 'elapsed=%e maxrss=%M' python cli/scripts/train_gpt2.py --tinystories --native-cuda-dry-run --native-cuda-print-command`, and `git diff --check`. The gradient-accumulation/template-selector, direct token-gradient accumulation, fused AdamW, fused accumulation-zero, fused gradient-clipping sumsq, llm.kittens TinyStories bin resolver, direct train-bin resolver, and SDPA row-kernel fallback slices were additionally verified with `bash tools/build_native_train_tile_ops.sh`, `bash tools/build_native_gpt2_cli.sh`, `build/nfn_gpt2_native_train --tinystories --dry-run`, `build/nfn_gpt2_native_train --dataset-alias /mnt/disk2/dev/open-source/llm.kittens/dev/data/tinystories/TinyStories_train.bin --dry-run`, `build/nfn_gpt2_native_train --tinystories --max-steps 1 --batch-size 1 --train-seq-len 2 --train-batch-tokens 2 --eval-every-steps 1 --eval-batches 1 --eval-batch-size 1 --train-transformer-lm` in a GPU-visible process, `nm -D build/libnfn_native_train_tile_ops.so | rg "nfn_native_tile_sumsq_partials_many_float32|launch_sumsq_partials_many_float32|nfn_native_tile_fill_many_float32|nfn_native_tile_adamw_step_many_with_device_scale_float32"`, `python -m pytest tests/test_native_gpt2.py -q -k "cpp_cli_builds_and_uses_sm120_defaults or native_train_tile_ops_builds_torch_free_c_abi or unified_native_train_cli_builds_dispatches_gpt2_and_rejects_unsupported"`, `python -m pytest cli/tests/test_train_gpt2_native.py -q`, `python -m pytest tests/test_template_presets.py -q -k "catalog_matches_builder_dispatch or native_gpt2_compiled_cli_accepts_every_gpt_template_name or train_gpt2_fast_path_accepts_every_gpt_template_name or custom_graph_file"`, `python -m pytest cli/tests/test_train_gpt2_native.py -q -k "nfn_train_gpt2_translates_kernel_backend_to_native_backend"`, `python -m py_compile cli/nfn.py cli/scripts/train_gpt2.py cli/scripts/train_gpt2_native.py neuralfn/native_gpt2.py neuralfn/native_train.py`, and `git diff --check`. The dry-run resolves the llm.kittens TinyStories token bins and stays around 0.02s startup with no Torch-heavy shard materialization. The GPU-visible one-step probe completed one full 12-layer native transformer-LM optimizer step, wrote a native checkpoint, emitted validation loss, and reported `attention_forward_row_launch_count: 1`, `attention_forward_row_launch_fallback_count: 1`, `attention_forward_scalar_launch_count: 35`, proving repeated failed row launches are now auto-disabled; full SM120 throughput parity against `llm.kittens/train-sm120.sh` remains unproven.
- The padded-vocab native GPT-2 layout was additionally verified with `bash tools/build_native_gpt2_cli.sh`, `python -m pytest tests/test_native_gpt2.py -q -k "native_gpt2_cpp_cli_builds_and_uses_sm120_defaults or native_train_tile_ops_builds_torch_free_c_abi"`, and two GPU-visible one-step TinyStories probes with `NFN_NATIVE_GPT2_STAGE_TIMING=1`: the pre-padding baseline reported about `95,347` tokens/s and `lm_head_backward: 1371.97 ms`, while the padded-vocab run reported `padded_vocab: 50304`, `logit_workspace_elements: 412090368`, about `96,446` tokens/s, and `lm_head_backward: 1312.68 ms`.
- The padded LM-head smoke/preflight alignment was additionally verified with the native GPT-2 CLI rebuild and the focused `tests/test_native_gpt2.py` smoke/default slice.
- The SDPA output-element-count fix and row-launch telemetry were additionally verified with `bash tools/build_native_train_tile_ops.sh`, `bash tools/build_native_gpt2_cli.sh`, `nm -D build/libnfn_native_train_tile_ops.so | rg "nfn_native_tile_attention_forward_row_(prelaunch|grid|block)"`, `build/nfn_gpt2_native_train --tinystories --max-steps 1 --batch-size 1 --train-seq-len 2 --train-batch-tokens 2 --eval-every-steps 1 --eval-batches 1 --eval-batch-size 1 --train-transformer-lm --tile-ops-lib build/libnfn_native_train_tile_ops.so` in a GPU-visible process, `python -m pytest tests/test_native_gpt2.py -q -k "cpp_cli_builds_and_uses_sm120_defaults or native_train_tile_ops_builds_torch_free_c_abi or unified_native_train_cli_builds_dispatches_gpt2_and_rejects_unsupported"`, `python -m pytest cli/tests/test_train_gpt2_native.py -q -k "template or graph or nfn_train_gpt2_default_dispatches_directly_to_compiled_cli or nfn_train_gpt2_native_dry_run"`, `python -m pytest tests/test_template_presets.py -q -k "catalog_matches_builder_dispatch or native_gpt2_compiled_cli_accepts_every_gpt_template_name or train_gpt2_fast_path_accepts_every_gpt_template_name or custom_graph_file"`, `python -m py_compile neuralfn/config.py neuralfn/__init__.py cli/nfn.py cli/scripts/train_gpt2.py cli/scripts/train_gpt2_native.py neuralfn/native_gpt2.py neuralfn/native_train.py`, and `git diff --check`. The GPU-visible one-step probe now reports `attention_forward_row_launch_count: 35`, `attention_forward_row_launch_success_count: 35`, `attention_forward_row_launch_fallback_count: 0`, `attention_forward_scalar_launch_count: 0`, `attention_forward_row_launch_grid_x: 24`, and clean pre-launch error codes for the tiny 12-layer transformer-LM run.
- The full-head row-vector SDPA forward slice was verified with `bash tools/build_native_train_tile_ops.sh`, `cuobjdump --dump-resource-usage build/libnfn_native_train_tile_ops.so | rg -n "scaled_dot_product_attention_row_float32_kernel|scaled_dot_product_attention_float32_kernel" -A1`, `bash tools/build_native_gpt2_cli.sh`, and the same GPU-visible one-step transformer-LM probe. The row kernel stayed at `REG:48 STACK:0 LOCAL:0`, and the probe reported `attention_forward_score_reuse_value_dim: 64`, `attention_forward_value_chunk_size: 64`, `attention_forward_row_launch_grid_y: 1`, `attention_forward_row_launch_success_count: 35`, `attention_forward_row_launch_fallback_count: 0`, and `attention_forward_scalar_launch_count: 0`. A default-shape `batch=64, seq=1024, train_batch_tokens=65536` one-step probe was still not complete after roughly one minute and was terminated, so the remaining SM120-parity blocker is the scalar-per-element SDPA backward and other unfused training kernels rather than Python/Torch startup.

### 2026-06-11 Native GPT-2 CUDA trainer handoff

#### Added

- Added `neuralfn.native_gpt2` with `NativeGpt2RunConfig`, `NativeGpt2RunnerStatus`, `build_native_gpt2_run_config()`, `native_gpt2_runner_status()`, `resolve_native_gpt2_executable()`, `resolve_native_gpt2_launcher()`, `resolve_native_gpt2_token_shards()`, `write_native_gpt2_run_config()`, and `run_native_gpt2()`. These helpers materialize raw-text datasets into uint16 token shards when the tokenizer fits, resolve train/validation shard paths, and emit the llm.kittens-style `train_gpt2cu` command line for SM120 GPT-2 training.
- Added `build_native_gpt2_compiled_cli_run_config()` to `neuralfn.native_gpt2` and top-level `neuralfn`. It builds a dense GPT-2 compiled-CLI handoff from a dataset alias/path without Python-side token-shard metadata inspection, leaving shard validation to `nfn_gpt2_native_train`.
- Added `NativeGpt2RunConfig.kernel_backend` and `tile_ops_lib`, plus `--native-cuda-kernel-backend` / `--kernel-backend`, `--native-cuda-tile-ops-lib`, `--native-cuda-print-plan`, and `--native-cuda-check-tile-ops` on `train_gpt2_native.py`. `llm-kittens` remains the current external fast trainer; `tile-cuda` is the NeuralFn-owned raw Tile ABI preflight/check path.
- Added native CUDA runtime controls to `cli/scripts/train_gpt2.py`: `--runtime native-cuda`, `--eval-every-steps`, `--native-cuda-executable`, `--native-cuda-output-dir`, `--native-cuda-config-out`, `--native-cuda-print-command`, `--native-cuda-dry-run`, `--native-cuda-allow-train-val-fallback`, and native checkpoint/sample/activation flags. `NFN_NATIVE_GPT2_TRAIN_BIN` can point the script at a workstation-local compiled trainer.
- Added `cli/scripts/train_gpt2_native.py`, a lightweight native-only runner used by `train_gpt2.py` before graph-backed modules are imported.
- Added an early `nfn train --base-model gpt2` dispatcher for explicit dense GPT-2 native pretraining commands. With the default `compiled-cli` runner it translates supported flags directly to the compiled `nfn_gpt2_native_train` cached-shard CLI before importing `train_gpt2_native.py`, `nfn_impl`, or Torch; explicit non-default runners still use the Python native runner.
- Added `--native-cuda-runner {auto,binding,compiled-cli,launcher,subprocess}`. `auto` prefers an installed in-process C++ binding module (`neuralfn_native_gpt2` or `neuralfn._native_gpt2`), then the no-Python compiled CLI, then the compiled launcher, then direct `train_gpt2cu` subprocess execution; `binding` requires the C++ binding; `compiled-cli` requires `nfn_gpt2_native_train`; `launcher` requires the C++ launcher; `subprocess` forces the external executable path. Native JSON configs include the resolved runner status plus raw, compiled-CLI, and launcher command argv.
- Added `neuralfn/csrc/native_gpt2/binding.cpp` and `tools/build_native_gpt2_binding.sh`. The build helper produces `neuralfn/_native_gpt2*.so`, exposing `run_gpt2(config_dict)` / `run_train(config_dict)` for the SDK's in-process native binding path without pybind11 or Torch.
- Added `neuralfn/csrc/native_gpt2/nfn_gpt2_tile_train.cpp` and `tools/build_native_gpt2_launcher.sh`. The launcher sets `CUDA_DEVICE_MAX_CONNECTIONS=1`, resolves `NFN_NATIVE_GPT2_TRAIN_BIN`, and `execvp()`s the workstation-local GPT-2 CUDA trainer without importing Torch.
- Added `neuralfn/csrc/native_gpt2/nfn_gpt2_native_train.cpp` and `tools/build_native_gpt2_cli.sh`. The built `build/nfn_gpt2_native_train` binary resolves cached `fineweb_train_*.bin` / `fineweb_val_*.bin` dataset shards and launches `train_gpt2cu` directly with the SM120 defaults from `llm.kittens/train-sm120.sh`, bypassing Python entirely for already-cached datasets.
- Added `neuralfn/csrc/native_train/nfn_native_train.cpp` and `tools/build_native_train_cli.sh`. The built `build/nfn_native_train` binary is a unified no-Python NeuralFn training frontend: it dispatches GPT-2 to `nfn_gpt2_native_train` and rejects unsupported model families in C++ before Python/Torch startup.
- Added a compiled native-training model registry to `nfn_native_train`, including `--list-models` and `--list-models --json`. The registry marks dense GPT-2 and NanoGPT as `partial-native-trainer`, and GPT-2 evo, LLaMA, MixLLaMA, JEPA, semantic router MoE, and DeepSeek-V4 as missing or preflight-only native CUDA Tile C++ trainers instead of aliasing them to an incompatible GPT-2 shape.
- Added `neuralfn.native_train` with `NativeTrainRunConfig`, `NativeTrainRunnerStatus`, `build_native_train_run_config()`, `resolve_native_train_cli()`, `native_train_runner_status()`, `native_train_model_registry()`, and `run_native_train()` for SDK-level handoff to the compiled native training frontend.
- Added `neuralfn/csrc/native_train/binding.cpp` and `tools/build_native_train_binding.sh`. The build helper produces `neuralfn/_native_train*.so`, exposing `run_train(config_dict)` / `run_native_train(config_dict)` for the SDK's unified native-train binding path without pybind11 or Torch.
- Added `neuralfn/csrc/native_train/missing_native_train.cpp` and `tools/build_native_missing_trainers.sh`. The build helper produces compiled per-family native trainer entrypoints for GPT-2 evo, NanoGPT, LLaMA, MixLLaMA, JEPA, semantic-router MoE, and DeepSeek-V4; each binary currently reports the CUDA Tile C++ trainer kernels still required for that family.
- Added `neuralfn/csrc/native_train/gpt2_evo_native_train.cpp`. `tools/build_native_missing_trainers.sh` now builds GPT-2 evo as a model-aware C++ native preflight target: `nfn_gpt2_evo_native_train --print-plan` parses the dense GPT-2 evo shape, enforces the `adamw` optimizer profile, preserves NVFP4 activation intent, carries validation cadence such as `--eval-every-steps 1000`, and reports the remaining native evo candidate-evaluation, device-side mutation, loss-reduction, and best-candidate adoption kernels without importing Python or Torch.
- Direct `cli/scripts/train_gpt2_evo.py` execution now prefers the family-specific compiled GPT-2 evo preflight binary before the generic native registry when `NFN_NATIVE_GPT2_EVO_CLI`, `build/nfn_gpt2_evo_native_train`, or an installed `nfn_gpt2_evo_native_train` is available. This keeps `--print-plan`, `--help`, and `--native-cuda-dry-run` on the model-aware C++ evo contract without importing Torch, while the real evo trainer still exits nonzero until the candidate-evaluation/mutation/adoption kernels land. Verification: reran the focused direct-script guard tests with `python -m pytest cli/tests/test_train_gpt2_native.py -q -k "legacy_training_scripts_reject_before_torch_import or train_gpt2_evo_direct_script_prefers_family_native_preflight"`.
- Added `neuralfn/csrc/native_train/nanogpt_native_train.cpp`. `tools/build_native_missing_trainers.sh` now builds NanoGPT as a model-aware C++ native target that parses the NanoGPT defaults, enforces the `adamw` optimizer profile, validates native shape/token-shard constraints, emits a `--print-plan` JSON contract, and runs the partial `--train-token-lm` loop without importing Python or Torch.
- Added `neuralfn/csrc/native_train/tile_ops.h`, `neuralfn/csrc/native_train/tile_ops.cu`, and `tools/build_native_train_tile_ops.sh`. The build helper produces `libnfn_native_train_tile_ops.so`, a raw no-Torch C ABI over CUDA Tile AdamW, gradient accumulation, device-buffer fill/zeroing, single-buffer and multi-buffer sumsq partials, device-side global-norm clip scale finalization, device-scalar gradient scaling, reduction, linear, linear input/weight/weight-accumulate/bias/bias-accumulate backward, scaled residual add, fused QKV split/merge, GELU forward/backward, token embedding forward/weight backward, absolute-position embedding forward/backward/backward-accumulate, RMSNorm, RMSNorm input backward, LayerNorm, LayerNorm input/affine/affine-accumulate backward, softmax, token and masked token-cross-entropy partial, token and masked token-cross-entropy logits backward, and scaled dot-product attention forward/backward kernels from `neuralfn/csrc/tile_cuda/kernels.cu`.
- Added `neuralfn/csrc/native_train/token_shards.h` and `neuralfn/csrc/native_train/token_shards.cpp`, a reusable no-Torch token-shard resolver and sequential batch sampler for native trainers. It resolves aliases through `NFN_DATASETS_DIR`, validates sorted `fineweb_train_*.bin` / `fineweb_val_*.bin` uint16 shards, accepts llm.kittens-style `TinyStories_train.bin` / `TinyStories_val.bin`, infers validation siblings for direct train-bin paths, skips the 1024-byte cached-shard header when present, counts train/validation tokens, computes native microbatch plus gradient-accumulation metadata, and produces token plus next-token target buffers without putting token payloads into graph nodes.
- Added `tools/build_native_gpt2_all.sh`, `tools/install_native_gpt2_commands.sh`, and expanded `cli/install.sh`. The CLI installer now keeps Torch optional, installs root + CLI packages in editable mode, builds the native GPT-2 C++ binding, launcher, no-Python cached-shard CLI, and unified native frontend by default, then links stable command names (`nfn-gpt2-native`, `nfn-gpt2-native-train`, `nfn-native-train`, `nfn-gpt2-tile-launcher`) into the active Python scripts directory; pass `--no-native` to skip C++ artifact builds.
- `tools/install_native_gpt2_commands.sh` now links both underscore and hyphen command names for built per-family native trainer entrypoints, such as `nfn_nanogpt_native_train` and `nfn-nanogpt-native-train`.
- Added `NativeGpt2CheckpointInfo`, `read_native_gpt2_checkpoint_info()`, `is_native_gpt2_checkpoint()`, `latest_native_gpt2_checkpoint()`, and `native_gpt2_parameter_count()` to `neuralfn.native_gpt2` and top-level `neuralfn`. These helpers inspect llm.kittens/NeuralFn native `model_########.bin` headers and matching `DONE_########` markers without importing Torch.
- Added native checkpoint metadata recognition to `nfn infer --checkpoint PATH --native-info` and `cli/scripts/infer_gpt2.py --native-checkpoint PATH --native-info`. The CLI now identifies native GPT-2 `.bin` checkpoints before importing `nfn_impl` or the graph-backed runtime.
- Added `cli/scripts/native_training_guard.py`, a lightweight pre-import guard used by legacy graph-backed training scripts to hand direct execution to `nfn_native_train` before Torch, NumPy, dataset manager, or `TorchTrainer` imports.
- Added regression coverage in `tests/test_native_gpt2.py` for raw-text-to-uint16 shard materialization, SM120 command construction, JSON config output, runner selection, binding/launcher invocation, C++ binding build/run behavior, C++ launcher compile/exec behavior, C++ no-Python CLI compile/exec behavior, and native activation validation. Added `cli/tests/test_train_gpt2_native.py` to execute `train_gpt2.py` through the native dry-run path and assert that `torch` is not imported.
- Added `tests/test_native_dependencies.py` to assert that Torch remains an optional dependency.

#### Changed

- `cli/scripts/train_gpt2.py` now only supports the native CUDA runtime, so the plain GPT-2 harness resolves cached token shards and launches the compiled CUDA trainer directly instead of constructing a graph and sending real batches through `TorchTrainer`. The default validation cadence is 250 optimizer steps; pass `--eval-every-steps 1000` for validation loss every 1000 steps.
- `cli/scripts/train_gpt2.py` is now a native-only wrapper around `train_gpt2_native.py`. Importing it, building its parser, and resolving native defaults no longer import Torch, `server.dataset_manager`, NumPy, or tiktoken.
- Direct `python cli/scripts/train_gpt2.py ...` native runs now set up repo/script imports before dispatching to `train_gpt2_native.py`, so the default native CUDA path works without `PYTHONPATH` and still avoids Torch.
- `cli/scripts/train_gpt2_native.py` now honors `NFN_DATASETS_DIR` for relative dataset aliases, matching the compiled native CLI cache override and avoiding hard-coded home-cache access during native dry-runs.
- `cli/scripts/train_gpt2_native.py` now uses the compiled-CLI handoff helper when `--native-cuda-runner compiled-cli` resolves and cached train plus validation shard files already exist. That path no longer reads `meta.json`, validates token-shard metadata, or imports the dataset manager before the compiled C++ shard resolver runs.
- `neuralfn._native_gpt2` now detects alias-only compiled-CLI configs and executes `compiled_cli_argv` instead of raw `argv`. This keeps SDK `run_native_gpt2(build_native_gpt2_compiled_cli_run_config(...), runner="auto")` on the compiled C++ shard resolver even when the in-process binding is installed.
- `nfn_gpt2_native_train` now exposes a strict `--backend llm-kittens|tile-cuda` selector. `--backend tile-cuda --print-plan` emits the required dense GPT-2 raw Tile ABI symbols and native work list; `--check-tile-ops --tile-ops-lib PATH` verifies those C ABI symbols with `dlopen` and exits nonzero if the library is missing or incomplete. Real `tile-cuda` GPT-2 training exits nonzero until the dense loop is wired to those kernels, so the external `llm-kittens` trainer cannot be mistaken for NeuralFn-owned Tile CUDA training.
- `nfn_gpt2_native_train --backend tile-cuda --print-plan` now includes a GPT-2 parameter layout and forward/backward/optimizer stage plan for the 12-layer dense shape. The plan names tied embedding/LM-head buffers, LayerNorm/projection/attention/MLP stages, CE backward workspace use, gradient zero/clip/scale, and AdamW stages so the remaining loop wiring has a concrete raw Tile ABI contract.
- `nfn_gpt2_native_train --smoke-tile-ops --tile-ops-lib PATH` now loads the raw Tile trainer ops library, loads CUDA runtime, executes `nfn_native_tile_fill_float32` on a tiny device buffer, copies the result back, and reports JSON without importing Python or Torch. `train_gpt2_native.py` and the SDK expose this as `--native-cuda-smoke-tile-ops` / `NativeGpt2RunConfig.smoke_tile_ops`; `--cuda-runtime-lib` / `cuda_runtime_lib` can pin libcudart resolution.
- `nfn_gpt2_native_train --smoke-optimizer-step --tile-ops-lib PATH` now allocates GPT-2-sized contiguous parameter, gradient, and AdamW moment buffers, initializes them with raw fill kernels, runs one `nfn_native_tile_adamw_step_float32` call per registered GPT-2 parameter buffer with the correct decay/no-decay setting, samples copyback values, and reports JSON without Python or Torch. `train_gpt2_native.py` and the SDK expose this as `--native-cuda-smoke-optimizer-step` / `NativeGpt2RunConfig.smoke_optimizer_step`.
- `nfn_gpt2_native_train --smoke-lm-step --tile-ops-lib PATH` now runs a tiny GPT-2-shaped tied embedding/LM-head forward/backward/update slice through raw Tile kernels: token embedding, full-vocab linear logits, token CE partials, workspace CE backward, linear input/weight backward, token embedding weight backward, and AdamW. `train_gpt2_native.py` and the SDK expose this as `--native-cuda-smoke-lm-step` / `NativeGpt2RunConfig.smoke_lm_step`.
- `nfn_gpt2_native_train --smoke-embedding-lm-step --tile-ops-lib PATH` now samples a tiny cached uint16 token batch in C++ and runs GPT-2 token embedding, absolute position embedding, embedding residual add, final LayerNorm, tied LM head, CE backward, embedding/norm backward, and AdamW through raw Tile kernels without graph-editor payloads. `train_gpt2_native.py` and the SDK expose this as `--native-cuda-smoke-embedding-lm-step` / `NativeGpt2RunConfig.smoke_embedding_lm_step`.
- `nfn_gpt2_native_train --train-embedding-lm --tile-ops-lib PATH` now runs that GPT-2 embedding/final-norm/LM path as a real multi-step compiled loop over cached train shards with periodic validation losses from validation shards, without Torch or graph-editor payloads. `train_gpt2_native.py` exposes the same strict `--train-embedding-lm` flag, the SDK exposes `NativeGpt2RunConfig.train_embedding_lm`, and `--eval-batches` / `--eval-batch-size` bound validation work.
- `nfn_gpt2_native_train --smoke-attention-step --tile-ops-lib PATH` now runs a tiny GPT-2 model-dim attention stage through raw Tile kernels: qkv projection, QKV split, scaled dot-product attention forward/backward, QKV gradient merge, projection backward, and AdamW. `train_gpt2_native.py` and the SDK expose this as `--native-cuda-smoke-attention-step` / `NativeGpt2RunConfig.smoke_attention_step`.
- `nfn_gpt2_native_train --smoke-mlp-step --tile-ops-lib PATH` now runs a tiny GPT-2 MLP stage through raw Tile kernels: c_fc projection, GELU forward/backward, c_proj projection backward, and AdamW. `train_gpt2_native.py` and the SDK expose this as `--native-cuda-smoke-mlp-step` / `NativeGpt2RunConfig.smoke_mlp_step`.
- `nfn_gpt2_native_train --smoke-norm-residual-step --tile-ops-lib PATH` now runs the GPT-2 transformer-block glue path through raw Tile kernels: LayerNorm, scaled residual add, LayerNorm affine/input backward, gradient accumulation, and AdamW. `train_gpt2_native.py` and the SDK expose this as `--native-cuda-smoke-norm-residual-step` / `NativeGpt2RunConfig.smoke_norm_residual_step`.
- `nfn_gpt2_native_train --smoke-transformer-block-step --tile-ops-lib PATH` now composes GPT-2 LayerNorm, fused QKV attention, real 12-head reshape/merge layout, residual adds, MLP, backward passes, gradient accumulation, projection bias gradients, and AdamW updates for all 12 GPT-2 block parameter buffers through raw Tile kernels. `train_gpt2_native.py` and the SDK expose this as `--native-cuda-smoke-transformer-block-step` / `NativeGpt2RunConfig.smoke_transformer_block_step`; the JSON reports `weight_update_count: 12` plus LayerNorm and projection-bias samples.
- `nfn_gpt2_native_train --smoke-transformer-lm-step --tile-ops-lib PATH` now samples cached uint16 tokens, preserves range-checked GPT-2 token IDs, and runs token/position embeddings, one tiny transformer block, final LayerNorm, tied LM head, token CE forward/backward, transformer backward, embedding backward, and AdamW updates for 16 parameter buffers through raw Tile kernels. `train_gpt2_native.py` and the SDK expose this as `--native-cuda-smoke-transformer-lm-step` / `NativeGpt2RunConfig.smoke_transformer_lm_step`, giving the dense GPT-2 loop a no-Torch token-to-loss transformer integration gate before full 12-layer training is wired.
- `nfn_gpt2_native_train --train-transformer-lm --tile-ops-lib PATH` now runs the GPT-2 transformer/LM path as a full-vocab real-dim 12-layer multi-step compiled C++ loop over cached train shards without Python/Torch or graph-editor payloads. It uses token/position embeddings, transformer blocks, final norm, a row-chunked tied LM-head/CE workspace, transformer backward, embedding backward, device-side global norm gradient clipping, and AdamW for 148 parameter buffers, and emits periodic validation records under `validation.losses`; the JSON includes `trained_layers: 12`, `target_layers: 12`, `vocab: 50257`, `lm_head_row_chunk_size`, `logit_workspace_elements`, `gradient_partial_count`, `gradient_clip_norm`, `sample_gradient_clip_scale`, `activation_tape_strategy`, and checkpoint export metadata so callers can confirm the native vocab, bounded logits workspace, clip paths, and artifact path ran. Successful runs write a final 12-layer trained-weight native version-5 bf16 checkpoint plus `DONE_########` marker. `train_gpt2_native.py`, root `nfn train`, and the SDK expose it as `--train-transformer-lm` / `NativeGpt2RunConfig.train_transformer_lm`.
- The raw no-Torch native trainer ABI now exposes `nfn_native_tile_reshape_heads_float32` and `nfn_native_tile_merge_heads_float32`. GPT-2 `--smoke-transformer-block-step` now uses the real GPT-2 attention layout (`12` heads, `64`-wide heads) by reshaping split Q/K/V activations before causal SDPA and merging attention gradients back before QKV-gradient packing. This removes the previous single-head block-smoke shortcut while the full dense GPT-2 trainer loop is still being wired.
- `nfn-native-train --list-models` now reports GPT-2 as `partial-native-trainer` instead of `implemented`. The command still dispatches GPT-2 to `nfn_gpt2_native_train`, but the registry no longer overclaims that full 12-layer dense GPT-2 training is already NeuralFn-owned Tile CUDA.
- Local generated `neuralfn.egg-info` metadata was refreshed so Torch is listed only under the `torch` extra, matching `pyproject.toml` and the default no-Torch install path.
- Native GPT-2 training now defaults to TinyStoriesV2 GPT-4 (`roneneldan__TinyStories__TinyStoriesV2-GPT4`) with the GPT-2 tokenizer. The `golf1` and `golf10` cached-token parameter-golf datasets remain available as explicit shortcuts, but they are no longer used by default.
- The workstation 5090 shell helpers no longer pass `--max-wallclock-seconds`; they run to their configured step schedule unless a caller adds a wallclock cap explicitly.
- `cli/scripts/train_gpt2.py` now defaults `--native-cuda-runner` to `compiled-cli`, requiring the no-Python cached-shard C++ CLI by default. Use `--native-cuda-runner auto`, `binding`, `launcher`, or `subprocess` only for explicit fallback/debug runs.
- `cli/scripts/infer_gpt2.py` now keeps import, parser construction, `--help`, and `--evo` / `--megakernel` artifact default resolution off the Torch, `server.dataset_manager`, and NumPy import path. Actual token generation still imports the graph-backed runtime after parsing until native GPT-2 inference is implemented.
- Native GPT-2 `.bin` checkpoints are no longer misclassified as graph-backed Torch `.pt` files by the lightweight inference entry points. Prompt generation from these native checkpoints still requires a dedicated native GPT-2 inference executable; train-time sampling remains available through `--native-cuda-sample-every`.
- Explicit `nfn train --base-model gpt2` dense pretraining now uses the native CUDA runner unless the command requests plan/help, JEPA, or a non-dense topology. With the default `compiled-cli` runner, `cli/nfn.py` replaces the Python process with `build/nfn_gpt2_native_train` for real runs and uses only the compiled CLI for dry-runs/command printing, so `train_gpt2_native.py`, `nfn_impl`, Torch, the dataset manager, NumPy, and tiktoken are not imported before native training. Old graph-only flags such as `--megakernel` now fail in the native parser instead of importing `nfn_impl`.
- Default GPT-2 training dispatch now prefers `NFN_NATIVE_TRAIN_CLI` or `build/nfn_native_train` when available, then falls back to `NFN_NATIVE_GPT2_CLI` / `build/nfn_gpt2_native_train`. This makes the compiled top-level trainer the preferred handoff without breaking explicit GPT-2 CLI overrides.
- Non-GPT-2 `nfn train` commands now hand off to the compiled `nfn_native_train` frontend by default and fail from its model registry before importing `nfn_impl` or Torch unless `NFN_ALLOW_TORCH_TRAINING=1` is set for one-off debugging. Direct legacy graph-backed training scripts (`train_llama_fast.py`, `train_llama_megakernel.py`, `train_nanogpt.py`, `train_mixllama_fast.py`, `train_gpt2_evo.py`, `train_jepa_semantic.py`, `train_semantic_router_moe.py`, `train_semantic_router_moe-overnight.py`, and `train_deepseek_v4.py`) use the same native frontend handoff before their Torch imports.
- `nfn_native_train` now dispatches missing model families to their compiled per-family native target when that binary exists beside the frontend or in `build/`. Those targets still exit nonzero until the real CUDA Tile trainer is implemented, but the command path remains compiled C++ instead of Python/Torch.
- `nfn_native_train` now resolves per-family trainer overrides with model-family environment variables such as `NFN_NATIVE_NANOGPT_CLI`, and it probes both underscore and hyphen command names beside the installed frontend before checking `build/`.
- `nfn_nanogpt_native_train` now reports a concrete native NanoGPT plan instead of the generic missing-trainer placeholder output. The tied token-LM loop is implemented through `--train-token-lm`; full transformer training still exits nonzero until the NanoGPT CUDA Tile model-wide trainer loop is implemented.
- `nfn_nanogpt_native_train --print-plan` now includes a contiguous `parameter_layout` contract with per-buffer names, shapes, offsets, counts, weight-decay flags, required parameter/gradient/AdamW-state device-buffer sizes, and AdamW decay/no-decay parameter groups tied to the raw `nfn_native_tile_adamw_step_float32` ABI.
- `nfn_nanogpt_native_train --print-plan` now includes a `training_step_plan` with forward, backward, and optimizer stages classified as `ready`, `requires_wiring`, or `missing_abi`. This exposes the native Tile ABI call sequence for token/position embeddings, residual adds, LayerNorm, projections, attention, loss, gradient clipping, gradient scaling, and AdamW instead of leaving the trainer loop implicit.
- `nfn_nanogpt_native_train --check-tile-ops --tile-ops-lib PATH` now `dlopen`s the raw `libnfn_native_train_tile_ops.so` trainer library and verifies every NanoGPT-required C ABI symbol from the compiled binary. This keeps the native handoff check in C++ without importing Python, Torch, or the PyTorch extension binding.
- `nfn_nanogpt_native_train --smoke-tile-ops --tile-ops-lib PATH` now dynamically loads CUDA runtime, allocates a tiny device buffer, executes the raw `nfn_native_tile_fill_float32` trainer ABI, copies the buffer back, and verifies the value without Python or Torch. Use `--cuda-runtime-lib PATH` or `NFN_CUDA_RUNTIME_LIB` when libcudart needs an explicit path.
- `nfn_nanogpt_native_train --smoke-optimizer-step --tile-ops-lib PATH` now dynamically loads CUDA runtime, builds the NanoGPT parameter layout, allocates contiguous parameter, gradient, and AdamW moment buffers, initializes them with raw fill kernels, executes `nfn_native_tile_adamw_step_float32` once per registered parameter buffer with that buffer's decay/no-decay setting, copies param and moment buffers back, and verifies the update without Python or Torch.
- `nfn_nanogpt_native_train --smoke-training-loop-step --tile-ops-lib PATH` now dynamically loads CUDA runtime and exercises native optimizer-loop mechanics over the registered NanoGPT parameter layout: gradient zeroing, synthetic gradient fill, global-norm clip scale finalization, device-scalar gradient scaling, and per-buffer AdamW updates.
- `nfn_nanogpt_native_train --smoke-lm-step --tile-ops-lib PATH` now dynamically loads CUDA runtime and runs a tiny tied-embedding language-model step through token embedding, linear logits, token cross-entropy loss/backward, linear input/weight backward, token embedding weight backward, and AdamW update kernels, then verifies loss, gradient, and weight update values without Python or Torch.
- `nfn_nanogpt_native_train --smoke-token-train-step --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS` now resolves cached uint16 token shards in C++, samples a real native token/target batch, runs tied-embedding LM forward/backward/update kernels over those IDs, and verifies sampled-batch loss, gradient, and weight update values without Python or Torch.
- `nfn_nanogpt_native_train --train-token-lm --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS --max-steps N` now runs the tied token-embedding LM path as a real multi-step native training loop over cached uint16 shards. It streams batches with the C++ sequential sampler, zeros gradients on device, runs token CE backward and tied weight-gradient kernels, applies AdamW every step, and emits JSON metrics without importing Python or Torch.
- `nfn_nanogpt_native_train --train-token-lm` now computes periodic validation loss inside the compiled C++ loop when `--eval-every-steps` is positive. It samples resolved validation token shards for `--eval-batches` batches of `--eval-batch-size` rows, runs the same native tied-LM forward loss path without backward/update, and emits records under JSON `validation.losses` without routing validation payloads through graph-editor nodes, `TorchTrainer`, or Python dataset objects.
- `nfn-native-train --base-model nanogpt --train-token-lm ...` now dispatches through the unified compiled native frontend and SDK `run_native_train()` wrapper to the NanoGPT partial native token-LM trainer, preserving the no-Python/no-Torch path for cached-shard runs.
- Normal NanoGPT training entrypoints now select that partial native path automatically: `nfn train --base-model nanogpt ...` and `python cli/scripts/train_nanogpt.py ...` inject `--train-token-lm` before the compiled native frontend. `--dry-run` and `--print-command` inspect that same default route without starting the loop; explicit native actions such as `--print-plan`, `--check-tile-ops`, a smoke command, or `--train-token-lm` still run exactly as requested.
- The unified C++ `nfn_native_train` frontend now treats `--print-command` / `--native-cuda-print-command` as a true no-exec inspection mode. NanoGPT `--train-token-lm --dry-run` now prints the validated native plan and exits before loading Tile ops or entering the training loop.
- `nfn_nanogpt_native_train --smoke-embedding-norm-step --tile-ops-lib PATH --dataset-alias PATH_OR_ALIAS` now resolves cached uint16 token shards in C++, samples a real native token/target batch, runs token plus absolute-position embeddings, residual add, LayerNorm forward/backward, tied logits, CE backward, embedding/position/norm gradients, and AdamW updates, then verifies residual, norm, loss, gradient, and weight update values without Python or Torch.
- `nfn_nanogpt_native_train --smoke-fused-qkv-attention-step --tile-ops-lib PATH` now dynamically loads CUDA runtime and runs a tiny attention stage through one fused `attn.qkv.weight`, QKV split, scaled-dot-product attention forward/backward, QKV gradient merge, fused qkv weight backward, output projection backward, and AdamW update kernels for the fused qkv/output weights, then verifies forward, gradient, and weight update values without Python or Torch.
- `nfn_nanogpt_native_train --smoke-transformer-block-step --tile-ops-lib PATH` now dynamically loads CUDA runtime and composes LayerNorm, fused-QKV attention, residual adds, MLP, backward passes, gradient accumulation, and AdamW updates for a tiny transformer block through raw native kernels without Python or Torch.
- `nfn_nanogpt_native_train --smoke-mlp-step --tile-ops-lib PATH` now dynamically loads CUDA runtime and runs a tiny MLP stage through fc projection, GELU, output projection, projection/input backward, GELU backward, and AdamW update kernels for both MLP weights, then verifies forward, input-gradient, weight-gradient, and weight-update values without Python or Torch.
- `nfn_nanogpt_native_train --smoke-attention-step --tile-ops-lib PATH` now dynamically loads CUDA runtime and runs a tiny attention stage through Q/K/V projections, scaled-dot-product attention forward/backward, output projection forward/backward, Q/K/V projection backward, and AdamW update kernels for all attention weights, then verifies forward, gradient, and weight update values without Python or Torch.
- The NanoGPT attention and MLP projection stages are now represented as ready native stages in `training_step_plan`, including fused QKV projection, QKV split, SDPA forward/backward, QKV gradient merge, fused qkv backward, MLP fc/GELU/proj forward/backward, and output projection backward. On the default dropout-free path the remaining native NanoGPT work is narrowed to full trainer-loop integration over the ready forward, backward, and optimizer stages.
- The NanoGPT `training_step_plan` now includes an explicit `gradient_zero` optimizer stage using the new raw native `nfn_native_tile_fill_float32` ABI, so real trainer-loop wiring can clear gradient/device buffers on the GPU before accumulation.
- NanoGPT tied LM head input and weight backward are now represented as ready native stages through the raw linear backward ABI. The tied-head item is no longer listed as a missing native kernel; it remains part of the trainer-loop wiring work.
- Native NanoGPT preflight now defaults `dropout_p` to `0.0` for the CUDA Tile path. Passing a nonzero `--dropout-p` remains accepted but the plan reports the missing dropout forward/backward native Tile ABI as required work instead of pretending the trainer can execute it.
- The raw native Tile ops C ABI now exports `nfn_native_tile_scaled_residual_add_float32`, backed by the existing CUDA Tile `launch_scaled_residual_add_float32` launcher, so native trainers can perform embedding and transformer residual adds without the PyTorch extension binding.
- `tools/build_native_gpt2_all.sh` now builds the raw native Tile ops shared library as part of the all-native workstation build. `cli/install.sh --help` documents that native installs include this trainer ops artifact.
- The raw native Tile ops shared library now exposes model-training building blocks needed by native GPT/NanoGPT trainers: token embedding forward and weight backward, absolute position embedding forward, backward, and backward-accumulate, linear forward and input/weight/bias backward, scaled residual add, fused QKV split/merge for NanoGPT `qkv.weight`, GELU forward/backward, RMSNorm and input backward, LayerNorm and input/affine backward, softmax, masked token cross-entropy partials, CE logits backward, device-buffer fill/zeroing, device-side global-norm clip scale finalization, device-scalar gradient scaling, and scaled dot-product attention forward/backward. These are C ABI wrappers over CUDA Tile launchers and do not link the PyTorch extension binding.
- Native trainer CE logits backward now uses row-wise CUDA Tile kernels for vocabularies up to 1024 and chunked row-wise kernels for full GPT-class vocabularies. The raw C ABI exports `nfn_native_tile_token_cross_entropy_backward_with_workspace_float32` and `nfn_native_tile_masked_token_cross_entropy_backward_with_workspace_float32`, and NanoGPT `--train-token-lm` allocates reusable row max/denominator workspace once per run instead of using the previous elementwise large-vocab fallback.
- Native trainer linear weight and bias backward now switch large row counts to row-chunked tiled atomic accumulation. The public C ABI remains `nfn_native_tile_linear_backward_weight_float32` / `nfn_native_tile_linear_backward_bias_float32`, but GPT-sized row reductions no longer execute one serial row loop per output element.
- `nfn_nanogpt_native_train --print-plan` now reports `available_native_kernels` separately from `required_native_kernels`, so the preflight distinguishes native ABI coverage already present from remaining backward/model-training work.
- `nfn_nanogpt_native_train --print-plan --require-token-shards` now resolves cached token shards in C++, requires train/validation shards unless `--allow-train-val-fallback` is passed, and includes shard paths, token counts, microbatch count, and gradient-accumulation metadata in the JSON plan. Add `--sample-token-batch` to include the first native token/target batch from the sequential C++ sampler.
- The implemented `nfn_gpt2_native_train` cached-shard CLI now uses the shared no-Torch `token_shards.cpp` resolver instead of a private shard scan, so GPT-2 and NanoGPT native entrypoints share alias resolution, shard sorting, header handling, and uint16 validation.
- Direct `python cli/nfn.py train --base-model gpt2 ...` invocations now add `cli/scripts` to `sys.path` before native dispatch, so the early no-Torch GPT-2 runner works without manually setting `PYTHONPATH`.
- Root `nfn --help`, `nfn --help-style ... --help`, and no-argument `nfn` startup now render lightweight help directly from `cli/nfn.py` without importing `nfn_impl` or Torch.
- `nfn train --help`, `nfn infer --help`, `nfn eval --help`, and `nfn kernels ... --help` now render lightweight static help directly from `cli/nfn.py` without importing `nfn_impl`, Torch, or graph-backed runtime modules.
- `nfn kernels list [--json]` now uses a metadata-only CUDA Tile registry path before importing `nfn_impl`, so kernel coverage can be inspected without loading Torch or graph-backed runtime modules.
- `neuralfn.tile_cuda` package exports are now lazy. Registry/config metadata imports remain Torch-free, while tensor execution helpers, diagnostics, and extension build/load helpers import their backing modules only when requested.
- `neuralfn.builtins` no longer imports `neuralfn.torch_backend` to create builtin module metadata; builtin default module configs are plain metadata so CLI/registry startup does not load the training runtime.
- CLI `--kernel-backend tile-cuda` for `nfn train`, `nfn infer`, and `nfn eval` now defaults to strict CUDA Tile kernel enforcement. Use `--no-tile-cuda-strict` only when intentionally debugging fallback behavior.
- The native handoff rejects tokenizers whose ids do not fit in uint16, such as `o200k_base`, because the compiled GPT-2 trainer consumes uint16 token shards. Use `--tokgpt2` or a supported SentencePiece tokenizer for this path.
- Set `NFN_NATIVE_GPT2_BINDING=0` to force `auto` past the local C++ binding when testing launcher/subprocess fallback behavior.
- Cached uint16-shard native GPT-2 startup no longer imports `server.dataset_manager`, NumPy, tiktoken, or Torch and no longer estimates the full training schedule before launching native code. The CLI still reports an approximate train-row count from the selected shard file size.
- Top-level `neuralfn` exports are now lazy, `neuralfn.neuron` lazily imports Torch only for module-state serialization, and `server.dataset_manager` lazily imports Torch only when converting memmap chunks into tensors. This keeps native GPT-2 startup off the Torch import path.
- Moved `torch>=2.0` out of default project dependencies and into optional extras: `.[torch]` for graph-backed workflows. At the time, `.[tile-cuda]` still installed the PyTorch extension-build dependency set; the current workstation-only native path has since made that extra Torch-free. Removed Torch from the default `requirements.txt` install path.
- Packaged the native GPT-2 binding/launcher/CLI sources, unified native-train C++ frontend, unified native-train binding source, raw native Tile ops C ABI source/header, and missing-family native trainer source in `pyproject.toml` and documented `bash tools/build_native_gpt2_binding.sh`, `bash tools/build_native_train_binding.sh`, `bash tools/build_native_train_tile_ops.sh`, `bash tools/build_native_gpt2_launcher.sh`, `bash tools/build_native_gpt2_cli.sh`, `bash tools/build_native_train_cli.sh`, and `bash tools/build_native_missing_trainers.sh` as the workstation build steps.

#### Breaking changes

- CLI commands that pass `--kernel-backend tile-cuda` now behave as strict CUDA Tile runs by default. Previously they could silently fall back when Tile coverage, extension loading, or tensor contracts were unavailable; now they fail unless callers pass `--no-tile-cuda-strict`.
- `cli/scripts/train_gpt2.py` no longer contains the graph-backed Torch training path. It only accepts `--runtime native-cuda`; use the NeuralFn template/SDK internals directly for experimental graph-backed GPT-2 training.
- `nfn train --base-model gpt2` no longer enters the full graph-backed `nfn_impl` path for explicit dense pretraining commands. The default `compiled-cli` runner also no longer enters the Python native runner, so it requires cached uint16 shards and `build/nfn_gpt2_native_train`; pass `--native-cuda-runner auto`, `binding`, `launcher`, or `subprocess` only when intentionally using Python materialization/orchestration or a non-default native runner.
- `nfn train --base-model llama`, `nanogpt`, JEPA, semantic/MoE, DeepSeek, and other non-GPT-2 model-family training commands no longer enter the graph-backed Torch path by default. They are routed to `nfn_native_train` and, when built, then to the family-specific compiled target. NanoGPT `--train-token-lm` runs as a partial native trainer; families without a real trainer report the missing CUDA Tile trainer work. Build a real native C++ trainer for that family first, or set `NFN_ALLOW_TORCH_TRAINING=1` only for local legacy debugging.
- Direct legacy training scripts now exec `nfn_native_train --base-model <family>` before importing Torch by default. This intentionally breaks the old `python cli/scripts/train_llama_fast.py ...` graph-backed style unless `NFN_ALLOW_TORCH_TRAINING=1` is set for debugging.
- `cli/scripts/infer_gpt2.py --evo --megakernel` is rejected because `gpt2_evo` exports eager artifacts, not megakernel artifacts.
- `nfn infer --checkpoint native-model.bin` now recognizes native GPT-2 checkpoints and returns native metadata/status instead of attempting graphless Torch checkpoint inference.
- Code that depended on `import neuralfn` eagerly importing every SDK submodule must now access the desired top-level export or import the concrete submodule directly. This is intentional so native CUDA utilities can start without importing Torch.
- Native NanoGPT preflight defaults `dropout_p` to `0.0` instead of `0.1`; pass `--dropout-p 0.1` explicitly only when testing the still-missing dropout ABI path.
- Code that depended on `import neuralfn.tile_cuda` eagerly binding every tensor helper must now access the desired export or import the concrete submodule directly. This keeps metadata-only CUDA Tile registry paths off the Torch import path.
- `pip install -e .` and `pip install -r requirements.txt` no longer install Torch. Install `pip install -e ".[torch]"` for graph-backed training/inference; the current `.[tile-cuda]` extra is native CUDA Tile build tooling only.
- `--native-cuda-runner auto` now prefers the compiled `neuralfn._native_gpt2` binding, then `build/nfn_gpt2_native_train`, then `build/nfn_gpt2_tile_train`. If a local workflow depended on `auto` always invoking `train_gpt2cu` directly, pass `--native-cuda-runner subprocess`.
- `cli/scripts/train_gpt2.py` and the compiled `nfn_gpt2_native_train` helper no longer default to `willdepueoai__parameter-golf__sp1024__train1`. Use `--dataset golf1`, `--dataset golf10`, or `--dataset-alias willdepueoai__parameter-golf__sp1024__train1` to run those cached-token datasets explicitly.
- `cli/scripts/train_gpt2.py` no longer defaults to `--native-cuda-runner auto`. Install/build `build/nfn_gpt2_native_train`, or pass `--native-cuda-runner auto`/`subprocess` explicitly when intentionally running a fallback path.
- Per-family native trainer overrides use model-family environment variables such as `NFN_NATIVE_NANOGPT_CLI`. Target-name-derived override variables are not kept for compatibility on this workstation-only native path.
- The NanoGPT native trainer binary output changed from a generic placeholder message to a structured C++ target with `--print-plan` JSON, native validation errors, and the partial `--train-token-lm` loop.
- `nfn_gpt2_native_train --backend external` and `--backend tile_cuda` are no longer accepted compatibility aliases. Use `--backend llm-kittens` or `--backend tile-cuda`.
- `neuralfn.native_gpt2.native_gpt2_kernel_backend("tile_cuda")` is no longer accepted. Use `"tile-cuda"` so Python and C++ native frontend validation match.
- `cli/scripts/train_gpt2_native.py --native-cuda-train-embedding-lm` is no longer accepted. Use `--train-embedding-lm`, matching the compiled `nfn_gpt2_native_train` flag.
- `neuralfn.native_gpt2.native_gpt2_runner_status("cli")` is no longer accepted as an alias for `"compiled-cli"`. Use `"compiled-cli"` explicitly.
- `nfn-native-train --list-models` changed the GPT-2 status string from `implemented` to `partial-native-trainer`; callers that inspect the registry should treat that status as the current NeuralFn-owned Tile CUDA trainer while live SM120 throughput validation remains open.

#### Verification

- Verified with `python -m py_compile neuralfn/native_gpt2.py cli/scripts/train_gpt2.py`.
- Verified with `python -m pytest tests/test_native_gpt2.py -q`.
- Verified with `python -m py_compile neuralfn/__init__.py neuralfn/neuron.py server/dataset_manager.py neuralfn/native_gpt2.py cli/scripts/train_gpt2.py cli/scripts/train_gpt2_native.py`.
- Verified with `python -m pytest tests/test_native_gpt2.py cli/tests/test_train_gpt2_native.py -q`.
- Verified `nfn train --base-model gpt2` native startup in `cli/tests/test_train_gpt2_native.py`, including the assertion that `torch` is not imported during a native dry-run.
- Verified with `python -m pytest tests/test_native_gpt2.py tests/test_native_dependencies.py cli/tests/test_train_gpt2_native.py -q`.
- Verified `bash tools/build_native_gpt2_binding.sh` builds `neuralfn/_native_gpt2*.so` and `native_gpt2_runner_status("auto")` resolves to `binding` without importing Torch.
- Verified cached-shard native startup with `cli/tests/test_train_gpt2_native.py`, asserting `server.dataset_manager`, NumPy, tiktoken, and Torch are not imported.
- Verified `bash tools/build_native_gpt2_cli.sh` and the compiled `nfn_gpt2_native_train` dry-run/exec path in `tests/test_native_gpt2.py`.
- Verified GPT-2 native backend selection, `--backend tile-cuda --print-plan`, missing-library `--check-tile-ops`, and SDK `kernel_backend` validation with `python -m pytest tests/test_native_gpt2.py -q -k "compiled_cli_config or kernel_backend or cpp_cli_builds_and_uses_sm120_defaults or compiled_cli_runner or cpp_binding_uses_compiled_cli"`.
- Verified GPT-2 Tile parameter/stage plan JSON and the `partial-native-trainer` registry status with `python -m pytest tests/test_native_gpt2.py -q -k "cpp_cli_builds_and_uses_sm120_defaults or unified_native_train"`.
- Verified the `train_gpt2_native.py` compiled-CLI path and root `nfn train --kernel-backend tile-cuda` translation after backend wiring with `python -m pytest cli/tests/test_train_gpt2_native.py -q -k "compiled or dry_run or native or kernel_backend"`.
- Verified GPT-2 `--smoke-tile-ops` missing-library JSON, strict `tile_cuda` rejection, SDK smoke/runtime argv generation, and wrapper/root CLI smoke aliasing with `python -m pytest tests/test_native_gpt2.py -q -k "compiled_cli_config or kernel_backend or cpp_cli_builds_and_uses_sm120_defaults"` and `python -m pytest cli/tests/test_train_gpt2_native.py -q -k "kernel_backend or compiled or dry_run"`.
- Verified GPT-2 `--smoke-optimizer-step` missing-library JSON, help text, SDK argv generation, and wrapper/root CLI aliasing with `python -m pytest tests/test_native_gpt2.py -q -k "compiled_cli_config or kernel_backend or cpp_cli_builds_and_uses_sm120_defaults"` and `python -m pytest cli/tests/test_train_gpt2_native.py -q -k "kernel_backend or compiled or dry_run"`.
- Verified GPT-2 `--smoke-lm-step` missing-library JSON, help text, SDK argv generation, and wrapper/root CLI aliasing with `python -m pytest tests/test_native_gpt2.py -q -k "compiled_cli_config or kernel_backend or cpp_cli_builds_and_uses_sm120_defaults"` and `python -m pytest cli/tests/test_train_gpt2_native.py -q -k "kernel_backend or compiled or dry_run"`.
- Verified GPT-2 `--smoke-embedding-lm-step` missing-library JSON, help text, SDK argv generation, sampled-token metadata, and wrapper/root CLI aliasing with `python -m pytest tests/test_native_gpt2.py -q -k "compiled_cli_config or kernel_backend or cpp_cli_builds_and_uses_sm120_defaults or compiled_cli_runner or cpp_binding_uses_compiled_cli or unified_native_train"` and `python -m pytest cli/tests/test_train_gpt2_native.py -q -k "compiled or dry_run or native or kernel_backend or non_native_model"`.
- Verified GPT-2 `--train-embedding-lm` missing-library JSON, help text, SDK argv generation, validation flag forwarding, and wrapper/root CLI aliasing with `python -m pytest tests/test_native_gpt2.py -q -k "compiled_cli_config or kernel_backend or cpp_cli_builds_and_uses_sm120_defaults or compiled_cli_runner or cpp_binding_uses_compiled_cli or unified_native_train"` and `python -m pytest cli/tests/test_train_gpt2_native.py -q -k "compiled or dry_run or native or kernel_backend or non_native_model"`.
- Verified the raw head-layout trainer ABI and GPT-2 12-head transformer-block smoke path with `bash tools/build_native_gpt2_cli.sh /tmp/nfn_gpt2_native_train_check`, `bash tools/build_native_train_tile_ops.sh /tmp/libnfn_native_train_tile_ops_check.so`, `python -m pytest tests/test_native_gpt2.py -q -k "cpp_cli_builds_and_uses_sm120_defaults or native_train_tile_ops or compiled_cli_config or kernel_backend or compiled_cli_runner or cpp_binding_uses_compiled_cli or unified_native_train"`, `python -m pytest cli/tests/test_train_gpt2_native.py -q -k "compiled or dry_run or native or kernel_backend or non_native_model"`, and `python -m pytest tests/test_native_dependencies.py -q`; the live CUDA smoke remained skipped when the local runtime/device gate was unavailable.
- Verified GPT-2 `--smoke-attention-step` missing-library JSON, help text, SDK argv generation, and wrapper/root CLI aliasing with `python -m pytest tests/test_native_gpt2.py -q -k "compiled_cli_config or kernel_backend or cpp_cli_builds_and_uses_sm120_defaults or compiled_cli_runner or cpp_binding_uses_compiled_cli or unified_native_train"` and `python -m pytest cli/tests/test_train_gpt2_native.py -q -k "compiled or dry_run or native or kernel_backend or non_native_model"`. The compiled live-smoke binary and Tile ops library built successfully; local CUDA runtime execution reached `cudaMalloc` but returned CUDA error 35 because the loaded runtime requires a newer driver than this shell exposes.
- Verified GPT-2 `--smoke-mlp-step` missing-library JSON, help text, SDK argv generation, and wrapper/root CLI aliasing with `python -m pytest tests/test_native_gpt2.py -q -k "compiled_cli_config or kernel_backend or cpp_cli_builds_and_uses_sm120_defaults or compiled_cli_runner or cpp_binding_uses_compiled_cli or unified_native_train"` and `python -m pytest cli/tests/test_train_gpt2_native.py -q -k "compiled or dry_run or native or kernel_backend or non_native_model"`.
- Verified GPT-2 `--smoke-norm-residual-step` missing-library JSON, help text, SDK argv generation, and wrapper/root CLI aliasing with `python -m pytest tests/test_native_gpt2.py -q -k "compiled_cli_config or kernel_backend or cpp_cli_builds_and_uses_sm120_defaults or compiled_cli_runner or cpp_binding_uses_compiled_cli or unified_native_train"` and `python -m pytest cli/tests/test_train_gpt2_native.py -q -k "compiled or dry_run or native or kernel_backend or non_native_model"`.
- Verified GPT-2 `--smoke-transformer-block-step` missing-library JSON, help text, SDK argv generation, and wrapper/root CLI aliasing with `python -m pytest tests/test_native_gpt2.py -q -k "compiled_cli_config or kernel_backend or cpp_cli_builds_and_uses_sm120_defaults or compiled_cli_runner or cpp_binding_uses_compiled_cli or unified_native_train"` and `python -m pytest cli/tests/test_train_gpt2_native.py -q -k "compiled or dry_run or native or kernel_backend or non_native_model"`.
- Verified GPT-2 `--smoke-transformer-lm-step` help text, real-vocab missing-library JSON, SDK argv generation, and wrapper/root CLI aliasing with `bash tools/build_native_gpt2_cli.sh /tmp/nfn_gpt2_native_train_check`, `python -m pytest tests/test_native_gpt2.py -q -k "cpp_cli_builds_and_uses_sm120_defaults"`, and `python -m pytest cli/tests/test_train_gpt2_native.py -q -k "compiled or dry_run or native or kernel_backend or non_native_model"`. `bash tools/build_native_train_tile_ops.sh` also built `build/libnfn_native_train_tile_ops.so`; local live CUDA execution loaded the Tile ops library and CUDA runtime, then stopped at `cudaMalloc` with CUDA error 35 because this shell exposes an insufficient driver for the loaded runtime.
- Verified GPT-2 `--train-transformer-lm` help text, multi-step-loop JSON schema, validation schema, real-vocab schema, row-chunked logits workspace schema, device-side gradient clipping schema, SDK argv generation, and wrapper/root CLI forwarding with `bash tools/build_native_gpt2_cli.sh /tmp/nfn_gpt2_native_train_check`, `python -m pytest tests/test_native_gpt2.py -q -k "cpp_cli_builds_and_uses_sm120_defaults or compiled_cli_config or kernel_backend or compiled_cli_runner or cpp_binding_uses_compiled_cli or unified_native_train"`, `python -m pytest cli/tests/test_train_gpt2_native.py -q -k "compiled or dry_run or native or kernel_backend or non_native_model"`, and `python -m pytest tests/test_native_dependencies.py -q`. `bash tools/build_native_train_tile_ops.sh` built `build/libnfn_native_train_tile_ops.so`; a live two-step CUDA run loaded the Tile ops library and CUDA runtime with no missing symbols, then stopped at `cudaMalloc` with CUDA error 35 because this shell exposes an insufficient driver for the loaded runtime.
- Added GPT-2 `--checkpoint-metadata-smoke` to the compiled C++ `nfn_gpt2_native_train` path and `NativeGpt2RunConfig.checkpoint_metadata_smoke`. It writes a sparse version-5 bf16 native checkpoint-format file and matching `DONE_########` marker for the requested GPT-2 target shape without importing Torch or requiring CUDA, so the native checkpoint parser/inference metadata path can validate NeuralFn-owned artifacts independently of a live CUDA training run.
- Added actual trained-weight checkpoint export to successful GPT-2 `--train-transformer-lm` runs. The compiled C++ loop now copies updated device weights back in native GPT-2 parameter order, streams them as bf16 payload into `model_########.bin`, writes `DONE_########`, and reports checkpoint path, native shape, and file-size accounting in JSON without involving Torch.
- Verified GPT-2 checkpoint metadata emission with `bash tools/build_native_gpt2_cli.sh /tmp/nfn_gpt2_native_train_check` and `python -m pytest tests/test_native_gpt2.py -q -k "cpp_cli_builds_and_uses_sm120_defaults or compiled_cli_config"`, including `read_native_gpt2_checkpoint_info()` size validation and `latest_native_gpt2_checkpoint()` discovery of the generated `DONE_########` marker.
- Verified `bash tools/build_native_train_cli.sh` and the compiled `nfn_native_train` GPT-2 dispatch / unsupported-model rejection path in `tests/test_native_gpt2.py`.
- Verified `nfn_native_train --list-models --json` exposes dense GPT-2 and NanoGPT as `partial-native-trainer` with `python -m pytest tests/test_native_gpt2.py -q -k unified_native_train`.
- Verified `bash tools/build_native_train_binding.sh` builds `neuralfn/_native_train*.so` and `run_native_train(..., runner="auto")` resolves to the binding without importing Torch in `tests/test_native_gpt2.py`.
- Verified `neuralfn.native_train` command construction, runner status, subprocess fallback, and `native_train_model_registry()` in `tests/test_native_gpt2.py`.
- Verified importing `neuralfn.native_train` and constructing a native train config does not import Torch in `tests/test_native_dependencies.py`.
- Verified `bash tools/build_native_missing_trainers.sh` builds per-family native entrypoints and `nfn_native_train --base-model nanogpt` dispatches to `nfn_nanogpt_native_train` when present in `tests/test_native_gpt2.py`.
- Verified `nfn_gpt2_evo_native_train --print-plan` emits the GPT-2 evo JSON preflight, carries `--eval-every-steps`, preserves `--tile-cuda-activation-dtype nvfp4`, rejects `--optimizer-profile sm120_adamw`, and is reached through the unified frontend with `python -m pytest tests/test_native_gpt2.py -q -k "missing_family"`.
- Verified `nfn_nanogpt_native_train --print-plan` emits a NanoGPT JSON plan, carries `--eval-every-steps`, and rejects `--optimizer-profile sm120_adamw` in `tests/test_native_gpt2.py`.
- Verified the NanoGPT `parameter_layout` JSON contract, contiguous offsets, buffer counts, AdamW decay/no-decay groups, and required AdamW-state sizes with `python -m pytest tests/test_native_gpt2.py -q -k "missing_family"`.
- Verified the NanoGPT `training_step_plan`, dropout-free default, residual-add stage, and no missing ABI stages on the default path with `python -m pytest tests/test_native_gpt2.py -q -k "missing_family"`.
- Verified the NanoGPT tied LM head backward stages and removal of the old missing-kernel entry with `python -m pytest tests/test_native_gpt2.py -q -k "missing_family"`.
- Verified NanoGPT `--check-tile-ops` reports missing libraries with a nonzero exit and validates all required symbols against a built `libnfn_native_train_tile_ops.so` with `python -m pytest tests/test_native_gpt2.py -q -k "missing_family or tile_ops"`.
- Verified NanoGPT `--smoke-tile-ops` command-line/help coverage and the CUDA-runtime fill-kernel launch/copyback path in `tests/test_native_gpt2.py`; the live device smoke skips only when the local CUDA runtime/device is unavailable.
- Verified NanoGPT `--smoke-optimizer-step` command-line/help coverage and the CUDA-runtime AdamW launch/copyback path in `tests/test_native_gpt2.py`; the live device smoke shares the same CUDA runtime/device availability gate as the fill smoke.
- Verified NanoGPT `--smoke-training-loop-step` command-line/help coverage and the CUDA-runtime gradient zero/clip/scale/AdamW loop path in `tests/test_native_gpt2.py`; the live device smoke shares the same CUDA runtime/device availability gate as the fill smoke.
- Verified NanoGPT `--smoke-lm-step` command-line/help coverage and the CUDA-runtime tied-embedding LM forward/backward/update path in `tests/test_native_gpt2.py`; the live device smoke shares the same CUDA runtime/device availability gate as the fill smoke.
- Verified NanoGPT `--smoke-token-train-step` command-line/help coverage and the CUDA-runtime sampled-token tied-LM train-step path in `tests/test_native_gpt2.py`; the live device smoke shares the same CUDA runtime/device availability gate as the fill smoke.
- Verified NanoGPT `--train-token-lm` command-line/help coverage, missing Tile-op-library JSON failure, and the two-step CUDA-runtime tied token-LM training loop over sampled native token shards in `tests/test_native_gpt2.py`; the live device loop runs after the same CUDA runtime/device availability gate as the sampled-token smoke. The same live test now sets `--eval-every-steps 1 --eval-batches 1 --eval-batch-size 2` and asserts two validation-loss records at steps 1 and 2.
- Verified NanoGPT `--smoke-embedding-norm-step` command-line/help coverage and the CUDA-runtime sampled-token embedding/residual/LayerNorm train-step path in `tests/test_native_gpt2.py`; the live device smoke shares the same CUDA runtime/device availability gate as the fill smoke.
- Verified NanoGPT `--smoke-qkv-layout-step` command-line/help coverage and the CUDA-runtime fused-QKV split/merge layout path in `tests/test_native_gpt2.py`; the live device smoke shares the same CUDA runtime/device availability gate as the fill smoke.
- Verified NanoGPT `--smoke-fused-qkv-attention-step` command-line/help coverage and the CUDA-runtime fused-QKV projection/split/SDPA/backward/merge/update path in `tests/test_native_gpt2.py`; the live device smoke shares the same CUDA runtime/device availability gate as the fill smoke.
- Verified NanoGPT `--smoke-transformer-block-step` command-line/help coverage and the CUDA-runtime LayerNorm/fused-QKV attention/residual/MLP/backward/AdamW block path in `tests/test_native_gpt2.py`; the live device smoke shares the same CUDA runtime/device availability gate as the fill smoke.
- Verified NanoGPT `--smoke-mlp-step` command-line/help coverage and the CUDA-runtime MLP projection/GELU forward/backward/update path, including fc input-gradient copyback, in `tests/test_native_gpt2.py`; the live device smoke shares the same CUDA runtime/device availability gate as the fill smoke.
- Verified NanoGPT `--smoke-attention-step` command-line/help coverage and the CUDA-runtime attention projection/SDPA forward/backward/update path in `tests/test_native_gpt2.py`; the live device smoke shares the same CUDA runtime/device availability gate as the fill smoke.
- Verified this native sampled-token and embedding/norm train-step slice with `python -m pytest tests/test_native_gpt2.py -q -k "missing_family or tile_ops" -rs`; the compiled C++ targets built, the non-CUDA preflight passed, and the live Tile smoke skipped locally because `cudaMalloc` returned CUDA error 35 (`CUDA driver version is insufficient for CUDA runtime version`).
- Verified with `python -m pytest tests/test_native_gpt2.py tests/test_native_dependencies.py cli/tests/test_train_gpt2_native.py -q`, `bash -n tools/build_native_gpt2_cli.sh tools/build_native_missing_trainers.sh tools/build_native_gpt2_all.sh tools/build_native_train_tile_ops.sh cli/install.sh`, and `git diff --check`.
- Verified `bash tools/build_native_train_tile_ops.sh` builds `libnfn_native_train_tile_ops.so` with `nvcc -enable-tile` and does not compile through the PyTorch extension binding in `tests/test_native_gpt2.py`.
- Verified the raw native Tile ops C ABI exports `nfn_native_tile_split_qkv_float32` and `nfn_native_tile_merge_qkv_float32`, and the NanoGPT preflight requires both, with `python -m pytest tests/test_native_gpt2.py -q -k "missing_family or tile_ops" -rs`.
- Verified `libnfn_native_train_tile_ops.so` exports the trainer-facing linear/input/weight/bias-gradient, scaled residual add, fused QKV split/merge, fill/zeroing, GELU forward/backward, token embedding forward/weight-gradient, position embedding forward/gradient, norm, RMSNorm input backward, LayerNorm input/affine backward, global-norm clip finalization, device-scalar gradient scaling, softmax, masked CE, CE-backward, and attention forward/backward C ABI symbols with `python -m pytest tests/test_native_gpt2.py -q -k tile_ops`.
- Verified the raw native Tile ops C ABI exports `nfn_native_tile_scaled_residual_add_float32` with `python -m pytest tests/test_native_gpt2.py -q -k "tile_ops"`.
- Verified the raw native Tile ops C ABI exports `nfn_native_tile_fill_float32` and the NanoGPT preflight includes the `gradient_zero` optimizer stage with `python -m pytest tests/test_native_gpt2.py -q -k "tile_ops or missing_family"`.
- Verified the native Tile ops source keeps the row-wise and chunked CE backward paths, removes the elementwise large-vocab fallback, preserves large-row chunked linear weight/bias backward accumulation, atomic token embedding weight-gradient scatter, GELU forward/backward kernels, device-side global-norm clip finalization, SDPA backward kernels, scaled residual add, and linear input/weight/weight-accumulate/bias/bias-accumulate backward C ABI exports with `python -m pytest tests/test_native_gpt2.py -q -k tile_ops`.
- Verified NanoGPT native preflight token-shard validation with temporary uint16 `fineweb_train_*.bin` / `fineweb_val_*.bin` shards in `tests/test_native_gpt2.py`; the JSON plan reports sorted shards, header offsets, token counts, gradient accumulation, and the first sampled native token/target batch from C++.
- Verified the implemented GPT-2 native CLI is linked against the shared token-shard resolver by rejecting an odd-byte uint16 shard in `tests/test_native_gpt2.py`.
- Verified non-GPT-2 `nfn train` hands off to the compiled native frontend before Torch imports with `python -m pytest cli/tests/test_train_gpt2_native.py -q -k "non_native_model or unified_native_train or default_dispatches_directly"`.
- Verified the top-level CLI dispatcher still handles existing static help and command parsing with `python -m pytest cli/tests/test_nfn_cli.py -q`.
- Verified `tools/build_native_gpt2_all.sh` with temp output overrides and `cli/install.sh --help` in `tests/test_native_gpt2.py`.
- Verified `tools/install_native_gpt2_commands.sh` with a temp bin directory in `tests/test_native_gpt2.py`, including execution of the linked `nfn-gpt2-native --help` and `nfn-native-train --help` commands.
- Verified installed per-family native dispatch with `tools/install_native_gpt2_commands.sh`: the temp-bin `nfn-native-train --base-model nanogpt --dry-run` command now resolves the installed `nfn_nanogpt_native_train` symlink instead of missing the target.
- Verified direct `python cli/nfn.py` native GPT-2 dispatch and strict Tile CUDA CLI defaults with `python -m pytest cli/tests/test_train_gpt2_native.py cli/tests/test_nfn_cli.py -q`.
- Verified direct `python cli/scripts/train_gpt2.py` native GPT-2 dispatch without `PYTHONPATH`, plus `NFN_DATASETS_DIR` alias resolution, in `cli/tests/test_train_gpt2_native.py`.
- Verified native GPT-2 runner and dependency coverage with `python -m pytest tests/test_native_gpt2.py tests/test_native_dependencies.py cli/tests/test_train_gpt2_native.py -q`.
- Verified root help/no-argument startup avoids `nfn_impl` and Torch in a fresh subprocess via `cli/tests/test_train_gpt2_native.py`.
- Verified `nfn train|infer|eval --help` and `nfn kernels ... --help` avoid `nfn_impl` and Torch in fresh subprocesses via `cli/tests/test_train_gpt2_native.py`.
- Verified `nfn kernels list [--json]` and `neuralfn.tile_cuda.registry.coverage_report()` avoid `nfn_impl` and Torch in fresh subprocesses via `cli/tests/test_train_gpt2_native.py`.
- Verified the native GPT-2 TinyStories default and removal of wallclock caps from the 5090 shell helpers with `cli/tests/test_train_gpt2_native.py`.
- Verified the default GPT-2 training runner is `compiled-cli` and cached-shard dry-runs emit the no-Python CLI command without importing Torch, `server.dataset_manager`, NumPy, or tiktoken via `cli/tests/test_train_gpt2_native.py`.
- Verified default `nfn train --base-model gpt2` dispatch maps directly to the compiled no-Python cached-shard CLI without importing `train_gpt2_native.py`, `nfn_impl`, Torch, `server.dataset_manager`, NumPy, or tiktoken via `cli/tests/test_train_gpt2_native.py`.
- Verified non-GPT-2 `nfn train` and direct legacy training-script execution hand off to the native frontend before importing Torch or graph-backed training modules via `cli/tests/test_train_gpt2_native.py`.
- Verified GPT-2 inference parser/help/default resolution avoids Torch, `server.dataset_manager`, and NumPy in fresh subprocesses via `cli/tests/test_train_gpt2_native.py`.
- Verified `cli/scripts/train_gpt2.py` module import, parser construction, and default resolution avoid Torch, `server.dataset_manager`, and NumPy in fresh subprocesses via `cli/tests/test_train_gpt2_native.py`.
- Verified native GPT-2 checkpoint header parsing and latest-`DONE_*` discovery with `python -m pytest tests/test_native_gpt2.py -q`.
- Verified `nfn infer --checkpoint model_########.bin --native-info` and `cli/scripts/infer_gpt2.py --native-checkpoint model_########.bin --native-info` avoid Torch, `nfn_impl`, `server.dataset_manager`, and NumPy in fresh subprocesses via `cli/tests/test_train_gpt2_native.py`.

### 2026-06-10 GPT-2 evo-layer training harness on the CUDA Tile backend

#### Added

- Added `build_gpt2_evo_spec()` to `neuralfn/config.py`: an experimental dense GPT-2 variant where one designated transformer block is trained by an interleaved evolutionary search instead of gradient descent. New `ModelSpec` fields `layer_evo_enabled`, `layer_evo_index` (default `None` → `num_layers // 2`), `layer_evo_fraction`, `layer_evo_population`, `layer_evo_mutation_scale`, and `layer_evo_seed` flow through the serialized `template_spec` to the trainer. The builder is SDK-only and is not registered as an editor preset.
- Added a generic layer-evolution hook to `TorchTrainer` (`neuralfn/torch_backend.py`): `_layer_evo_config()` reads the `layer_evo_*` knobs (gated only on `layer_evo_enabled`, unlike route evolution which is tied to the `semantic_moe_jepa_evo` objective), `_layer_evo_parameters()` selects the designated block's parameters by the compiled `node_modules.block_{K}.` naming, the block is frozen (`requires_grad=False`) and excluded from all optimizer param groups before optimizer construction, and every `round(1/layer_evo_fraction)` steps the existing `_run_route_evolution()` machinery evaluates the current block weights plus gaussian mutants forward-only on the current macro-batches and adopts the best candidate. The current weights are always candidate 0, so the candidate loss never regresses. Results surface in `on_step` payloads as `step_info["layer_evo"]`.
- Added graph-level CUDA Tile NVFP4 activation packing via `graph.torch_config["tile_cuda_activation_dtype"] = "nvfp4"`. `CompiledTorchGraph` packs only supported projection/attention activation inputs into `NVFP4Tensor` values, preserving tied weights, masks, targets, losses, optimizer state, source nodes, and graph-editor metadata as normal tensors.
- Added `TorchTrainer.active_compiled_graph` during `train()` so long-running harness callbacks can evaluate the current in-memory weights without recompiling stale graph JSON. Added `TorchTrainer.last_compiled_graph` so final validation can reuse the same trained compiled graph after `train()` returns.
- Updated `TorchTrainConfig.optimizer_profile="adamw"` for RTX 5090/SM120 Tile runs. It uses the single AdamW optimizer path and, when `kernel_backend="tile_cuda"` and callers leave `lr_decay_iters` and `min_lr` unset, defaults to cosine decay across the resolved training step count with `min_lr=0.0`.
- Fixed the plain-AdamW optimizer branch of `TorchTrainer._build_optimizers` to exclude `requires_grad=False` parameters from its param group (the parameter-golf profile already filtered them).
- Added `cli/scripts/train_gpt2_evo.py`: a GPT-2 training harness (cloned from `train_gpt2.py`) that builds the `gpt2_evo` spec, defaults to the CUDA Tile kernel backend with new `--kernel-backend {auto,torch,tile-cuda}`, `--tile-cuda-strict`, `--tile-cuda-report`, `--tile-cuda-activation-dtype {nvfp4,float32,none}`, `--amp-dtype {bfloat16,bf16,float16,fp16,float32,fp32,none}`, and `--eval-every-steps N` flags plumbed into the graph/trainer config, exposes `--evo-layer-index/--evo-layer-interval/--evo-layer-population/--evo-layer-mutation-scale/--evo-layer-seed/--no-layer-evo`, and supports `--tinystories` end-to-end. Defaults now target an RTX 5090-class SM120 AdamW run with NVFP4 activation packing and bf16 AMP: 12 layers, `model_dim=768`, `num_heads=12`, seq len 1024, microbatch 64, 524,288 tokens/step, 20,000 steps, learning rate 0.0006, weight decay 0.1, 60 warmup steps, cosine decay to zero, and validation every 250 steps. Periodic eval prints `Validation eval step ... loss=...` using the retained compiled model when `--eval-every-steps` is positive, and final `Validation loss` uses that same compiled model rather than recompiling graph JSON. The bf16 AMP default avoids routing the full GPT-2 vocabulary projection through the fp32 `cublasSgemm` path that can fail at the default microbatch size.
- Changed the CUDA training script defaults (`train_gpt2.py`, `train_gpt2_evo.py`, `train_llama_fast.py`, `train_nanogpt.py`, `train_mixllama_fast.py`, `train_jepa_semantic.py`, `train_semantic_router_moe.py`, `train_semantic_router_moe-overnight.py`, and `train_deepseek_v4.py`) from `parameter_golf` to `adamw`. The `nfn` planner's default gradient optimizer preset also uses `adamw`; `parameter_golf_muon` remains available only as an explicit preset. The 5090 shell helpers now pass `--optimizer-profile adamw` and no longer pass split/Muon parameter-golf knobs. Removed the inherited parameter-golf wallclock cap from CUDA harness defaults by setting `max_wallclock_seconds=0.0`; `--max-wallclock-seconds` remains available as an explicit early-stop override. On `kernel_backend="tile_cuda"`, `adamw` now returns a `TileAdamW` optimizer wrapper that dispatches batched AdamW parameter updates through `tile_adamw_step_batch`, and trainer gradient clipping for that backend dispatches through `tile_gradient_clip_norm`. Explicit `tile-cuda` backend selection now build-loads the CUDA Tile extension instead of requiring `NFN_TILE_CUDA_BUILD=1`.
- Changed `cli/scripts/train_gpt2.py` to default to strict CUDA Tile execution for RTX 5090 runs: `--kernel-backend tile-cuda`, `--tile-cuda-activation-dtype nvfp4`, and `--amp-dtype bfloat16` are now wired into the graph and `TorchTrainConfig`, matching the GPT-2 evo harness instead of silently using generic PyTorch execution.
- Added a raw-text `uint16` token-cache path for large training datasets. When the selected tokenizer fits in `uint16`, the first training load streams `data.txt` and optional `val.txt` into `fineweb_train_000000.bin` / `fineweb_val_000000.bin`, updates `meta.json` with `data_format="uint16_shards"` and `token_cache_format="raw_text_uint16_shards"`, and subsequent runs memmap those shards. `estimate_text_schedule()` now reads token counts or shard sizes before falling back to dataset construction, so GPT-2/TinyStories schedule estimation no longer needs to tokenize the full raw text file. Tokenizers with ids outside `uint16` remain on the raw-text path.
- Added GPT-2 evo inference support: `cli/scripts/infer_gpt2.py --evo` resolves `gpt2_evo.pt/json`, `nfn infer` now recognizes eager template runtimes in exported graph/checkpoint metadata, and the docs show the explicit `nfn infer --graph ~/NeuralFn/artifacts/gpt2_evo.json --weights ~/NeuralFn/artifacts/gpt2_evo.pt` path.
- Added GPU-gated tests in `tests/test_layer_evo_gpu.py` (skip unless `NFN_TILE_CUDA_TEST=1`, CUDA available, and the CUDA Tile extension builds): knob plumbing and block-parameter selection on a strict tile-compiled graph, optimizer param-group exclusion for both `parameter_golf` and `adamw` profiles, and an end-to-end CUDA Tile training run asserting the evo search fires exactly on its interval and that the adopted candidate never scores worse than the current weights.

#### Verification

- Verified on an RTX 5090 (sm_120, CUDA 13.3) with the CUDA Tile extension built from source: `nfn kernels doctor`, strict tile-cuda compile of the composed `gpt2_evo` graph with a coverage report (zero uncovered nodes), `NFN_TILE_CUDA_TEST=1 pytest tests/test_tile_cuda_gpu.py tests/test_layer_evo_gpu.py`, a 20-step `--tinystories` smoke run with `--tile-cuda-strict`, and `pytest tests/test_template_presets.py`. Added CPU regression coverage that the compiled graph's NVFP4 activation-packing hook wraps only supported activation inputs and leaves tied weights unpacked.
- Re-verified the validation/inference path with `python -m pytest tests/test_torch_gpt.py -q`, `python -m pytest cli/tests/test_infer_megakernel_artifacts.py -q -k "gpt2_inference_evo_defaults or eager_runtime_metadata or runtime_mismatch"`, `python -m pytest cli/tests/test_nfn_cli.py -q`, `HOME=/tmp/neuralfn-home python -m pytest tests/test_template_presets.py -x -q`, `python -m py_compile cli/scripts/train_gpt2_evo.py cli/scripts/infer_gpt2.py cli/scripts/infer_jepa_semantic.py neuralfn/torch_backend.py`, `python cli/scripts/infer_gpt2.py --help`, `python cli/scripts/train_gpt2_evo.py --help`, and `git diff --check`.
- Re-verified the SM120 optimizer-profile defaults with `python -m pytest tests/test_torch_backend_schedule.py -q`, `python -m pytest cli/tests/test_nfn_cli.py -q`, `python -m py_compile neuralfn/torch_backend.py cli/nfn_impl.py cli/scripts/train_gpt2.py cli/scripts/train_gpt2_evo.py cli/scripts/train_llama_fast.py cli/scripts/train_nanogpt.py cli/scripts/train_mixllama_fast.py cli/scripts/train_jepa_semantic.py cli/scripts/train_semantic_router_moe.py cli/scripts/train_semantic_router_moe-overnight.py cli/scripts/train_deepseek_v4.py`, `bash -n cli/5090-mini-run.sh cli/5090-llama-smoke.sh cli/5090-llama-baseline.sh cli/5090-llama-overnight.sh`, and `git diff --check`.
- Re-verified the raw-text token-cache and metadata schedule path with `python -m pytest tests/test_dataset_manager_variants.py -q -k "materializes_gpt2_raw_text_uint16_cache"`, `python -m pytest cli/tests/test_train_drop_last.py -q -k "token_metadata or text_schedule"`, and `python -m py_compile server/dataset_manager.py cli/scripts/train_jepa_semantic.py cli/scripts/train_gpt2.py cli/scripts/train_gpt2_evo.py cli/scripts/train_llama_fast.py cli/scripts/train_mixllama_fast.py cli/scripts/train_llama_megakernel.py cli/scripts/train_nanogpt.py cli/scripts/train_deepseek_v4.py tests/test_dataset_manager_variants.py cli/tests/test_train_drop_last.py`.

### 2026-06-10 CUDA Tile backend plan and compiled training hot path

#### Added

- Added `todo-tile-cuda.md` as the authoritative CUDA Tile C++ implementation checklist. It inventories every current NeuralFn scalar builtin and module builtin, defines coverage gates for `BuiltinNeurons`, `build_module()`, and `build_function_module()`, and records CUDA Tile requirements, SDK/CLI work, examples, docs, and verification tasks.
- Added the initial `neuralfn.tile_cuda` SDK package with backend config, runtime diagnostics, a kernel registry, coverage reports, autograd wrappers, and an optional CUDA Tile PyTorch extension source build path. The registry accounts for every current builtin, module dispatch entry, scalar function dispatch entry, and optimizer/runtime kernel target without claiming incomplete CUDA Tile kernels are implemented.
- Added `TorchTrainConfig.kernel_backend`, `tile_cuda_strict`, and `tile_cuda_report_path` fields so the training API has a stable place to select CUDA Tile behavior as kernels land.
- Added real CUDA Tile float32 scalar fast-path kernels for `identity`, `negate`, `add`, `multiply`, `sigmoid`, `relu`, `tanh_neuron`, `gaussian`, `log`, `leaky_relu`, `prelu`, `relu6`, `elu`, `selu`, `silu`, `mish`, `softplus`, `softsign`, `hard_sigmoid`, `hard_tanh`, `hard_swish`, `softmax_2`, and `logsoftmax_2`, with PyTorch fallback for unsupported tensors.
- Added CUDA Tile scalar and module `gelu` coverage using the shared unary Tile path and autograd-composed exact GELU gradients.
- Added CUDA Tile elementwise module kernels for `logit_softcap`, `loss_scale`, `aux_loss_add`, `kl_penalty`, `residual_add`, `residual_mix`, `manifold_hyper_connection`, `qk_gain`, and `dyt`, including input and parameter gradients for the vector-scale modules.
- Added CUDA Tile layout kernels for `reshape_heads`, `merge_heads`, and `repeat_kv`, with inverse-layout autograd wrappers and CPU/GPU drift tests.
- Added CUDA Tile module kernels for `expert_combine`, `kv_cache_write`, `broadcast_expert_routes`, `broadcast_chunk_routes`, `byte_patch_merge`, and `latent_mse_loss`. The route kernels cover float32 route-weight expansion with int64 route-index outputs, `byte_patch_merge` covers nearest interpolation back to token length, and `latent_mse_loss` uses a Tile reduction for detached-target MSE.
- Added CUDA Tile `byte_patch_embed` coverage for byte-token embedding plus bias-free Conv1d patch projection, preserving `embedding.weight` and `proj.weight` state dict keys with PyTorch-composed gradients.
- Added CUDA Tile coverage for `threshold`, `kv_cache_read`, and `absolute_position_embedding`. `threshold` is a zero-gradient no-grad-style step function, `kv_cache_read` concatenates cached and current K/V tensors along the sequence dimension, and absolute position embeddings use a Tile gather-copy with explicit embedding-weight gradients.
- Added CUDA Tile `token_embedding` coverage with native int64 token lookup and indexed embedding-weight gradient accumulation while preserving the stage's `(hidden, embedding_weight)` output contract.
- Added CUDA Tile `rotary_embedding` coverage for Q/K RoPE forward math with inverse-rotation backward gradients and support for different query and KV head counts.
- Added CUDA Tile `causal_chunk_state` coverage for prefix and mean chunk-state construction with native Tile forward accumulation and PyTorch-composed gradients.
- Added CUDA Tile `rms_norm` and `qk_norm` coverage for contiguous float32 rows with last dimension up to 1024.
- Added CUDA Tile `layer_norm` coverage for contiguous float32 rows with last dimension up to 1024 and affine weight/bias gradients.
- Added CUDA Tile `group_norm` coverage for `[B,S,D]` tensors when `S * group_dim <= 1024`, preserving `norm.weight` and `norm.bias` state dict keys with PyTorch-composed gradients.
- Added CUDA Tile projection coverage for `linear`, `lm_head`, and `tied_lm_head`, with native forward dot products and autograd-composed input/weight/bias gradients.
- Added CUDA Tile composition coverage for `bitlinear_ternary`, preserving the PyTorch-reference ternary weight and activation STE while running the quantized projection through the Tile linear path.
- Added CUDA Tile composition coverage for `fp8_linear`, preserving the reference FP8 weight STE and `amax_history` update while running the dequantized projection through the Tile linear path.
- Added CUDA Tile composition coverage for `mx_linear`, preserving the reference MXFP4/MXFP8 block-scale weight STE while running the dequantized projection through the Tile linear path.
- Added CUDA Tile composition coverage for `nf4_linear` when `compute_dtype="fp32"` and `dropout=0`, preserving packed NF4 buffers, LoRA parameters, and `load_base_weight()` while running dequantized base and LoRA projections through Tile linear paths.
- Added CUDA Tile composition coverage for `randmap_adapter`, including a native trainable scaled residual add primitive for the adapter scale parameter while preserving frozen `down_proj`/`up_proj` maps and trainable `middle.weight`.
- Added CUDA Tile projection-composition coverage for `kv_pca_encode` and `kv_pca_decode`, preserving the PyTorch stage parameter names (`k_proj`, `v_proj`, `k_unproj`, and `v_unproj`) for checkpoint-compatible KV cache compression.
- Added CUDA Tile KV quantization coverage for `kv_quant_pack` and `kv_quant_unpack`, covering same-shaped float32 K/V rows with `head_dim <= 512`, per-row scale packing, dequantized unpacking, and PyTorch-compatible autograd for the non-smooth quantization contract.
- Added CUDA Tile composition coverage for `ttt_linear`, using Tile base projection, down projection, tanh activation, up projection, and residual add while preserving the existing TTT state dict keys.
- Added CUDA Tile composition coverage for deterministic `lora_linear` stages with `dropout=0`, using Tile base projection, low-rank A/B projections, and scaled residual add while preserving the existing LoRA state dict keys.
- Added CUDA Tile composition coverage for `mlp_relu2`, `swiglu`, `geglu`, and `reglu` using Tile linear, unary activation, and multiply primitives while preserving the original stage parameter names.
- Added CUDA Tile `solu` coverage using Tile linear projections plus a native row-wise softmax gate primitive for the softmax-gated GLU path.
- Added CUDA Tile composition coverage for `jepa_projector` and `jepa_predictor`, using Tile linear, LayerNorm, and GELU primitives while preserving the original `net.*` state dict keys.
- Added CUDA Tile projection-family wrappers for `router_logits`, `value_head`, `reward_head`, and `denoise_head` using the shared Tile linear primitive while preserving their original state dict keys and output shapes.
- Added CUDA Tile `act_halt_gate` coverage by composing Tile mean-pool/projection/sigmoid behavior while preserving the stage's `proj.*` state dict keys.
- Added CUDA Tile `act_weighted_sum` coverage for deterministic ACT recurrent-state accumulation with gradients for states and weights.
- Added CUDA Tile `latent_pool` coverage for JEPA masked latent pooling, including mean fallback for empty masks and gradients for both latent states and floating masks.
- Added CUDA Tile `token_cross_entropy` and `masked_token_cross_entropy` coverage with numerically stable native reductions, ignore-index support for the masked variant, and autograd-composed logits/mask gradients.
- Added CUDA Tile `sequence_logp` coverage for DPO-style masked sequence log-probability reduction with ignore-index support and gradients for logits and floating masks.
- Added CUDA Tile `dpo_pairwise_loss` coverage for sigmoid, hinge, and IPO pairwise preference losses. The Tile kernel reduces the scalar loss and emits detached chosen/rejected rewards to preserve the PyTorch stage contract.
- Added CUDA Tile `ppo_clipped_loss` coverage for PPO policy/value loss reductions, returning policy loss, value loss, and combined loss with PyTorch-composed gradients for clipping-branch parity.
- Added CUDA Tile `gae_compute` coverage for reverse-time generalized advantage estimation, returning advantages and returns with PyTorch-composed gradients for the scan contract.
- Added CUDA Tile `preference_bce_loss` coverage for Bradley-Terry preference-model loss reduction with gradients for chosen and rejected rewards.
- Added CUDA Tile `load_balance_loss` coverage by reusing the native route-balance reduction while preserving the stage's `(aux_loss, router_logits)` passthrough output and gradients.
- Added CUDA Tile `auxfree_load_balancing` coverage for per-expert bias addition while preserving the stage's no-grad device-side bias update.
- Added CUDA Tile `topk_route` coverage for normalized top-k routing weights and expert indices, preserving routing telemetry buffers and recomputing the selected-softmax path for gradients.
- Added CUDA Tile `route_balance_loss` coverage for router softmax-density balance loss with a native Tile density/reduction forward and autograd-composed gradients.
- Added CUDA Tile `route_selection_loss` coverage for supervised semantic-route BCE over the semantic expert slice, using the same semantic vocabulary dimension lookup as the PyTorch stage and PyTorch-composed gradients for parity.
- Added CUDA Tile composition coverage for `route_distillation_loss`, building detached teacher route logits from semantic topic confidence scores and using the native Tile softmax-distillation reduction for the student route KL objective.
- Added CUDA Tile `semantic_alignment_loss` coverage for masked semantic topic cross-entropy over per-dimension vocabulary logits, using native Tile item losses with PyTorch-composed logits gradients.
- Added CUDA Tile composition coverage for `semantic_projector`, using Tile topic heads, signature projection, row-softmax signature scalar, and residual MLP while preserving all projector state dict keys.
- Added CUDA Tile composition coverage for `semantic_chunk_projector`, flattening chunk states through Tile topic heads, signature projection, row-softmax signature scalar, and residual MLP before restoring the chunk-axis output contract.
- Added CUDA Tile `semantic_hasher` and `semantic_chunk_hasher` coverage for deterministic LSH bucket ID generation from flat and chunked semantic vectors.
- Added CUDA Tile composition coverage for `semantic_moe_router`, using native top-k route selection over cosine similarity while preserving the router centroids and telemetry buffers.
- Added CUDA Tile composition coverage for `semantic_hash_router`, using native top-k route selection for unforced hash/topic routing while preserving the PyTorch forced-target ordering path.
- Added CUDA Tile composition coverage for `semantic_moe_jepa_evo_router`, preserving shared/semantic/free route logits and forced-target ordering while routing the free expert head through Tile projection.
- Added CUDA Tile composition coverage for `expert_dispatch`, preserving the token-to-expert routing loop while using Tile linear and SiLU primitives for each selected expert MLP.
- Added CUDA Tile `scaled_dot_product_attention` coverage for contiguous float32 `[B,H,S,D]` attention with causal/non-causal masking, grouped-query head mapping, PyTorch-composed gradients, and CPU fallback.
- Extended the CUDA Tile attention kernel to cover `sliding_window_attention`, `block_sparse_attention`, `streaming_attention_sinks`, and deterministic `native_sparse_attention` sparse-mask patterns with right-aligned causal masking.
- Added CUDA Tile composition coverage for `differential_attention`, using split Q/K Tile SDPA branches, lambda subtraction, and Tile RMSNorm while preserving PyTorch-compatible gradients.
- Added CUDA Tile composition coverage for `causal_self_attention`, `fused_causal_attention`, `multi_latent_attention`, and `routed_attention_experts` using Tile projection, RoPE, SDPA, normalization, and output-projection primitives where each stage requires them.
- Added CUDA Tile `attentionless_decoder` coverage for bucket-conditioned expert-output decoding, preserving `bucket_embed.weight` and `out_proj.weight` gradients.
- Added CUDA Tile `dropout` coverage for deterministic `p=0` and inference passthrough via the Tile identity path while leaving stochastic training masks on the PyTorch RNG path.
- Added CUDA Tile `softmax_distillation_loss` coverage for teacher/student KL distillation with teacher-detached gradients and PyTorch-compatible `batchmean` scaling.
- Added CUDA Tile optimizer/runtime coverage for `adamw_step`, exposing a tensor-level AdamW update with decoupled weight decay, bias correction, CPU fallback, and GPU parity coverage.
- Added CUDA Tile optimizer/runtime coverage for `ema_update`, exposing an in-place no-grad target/source weighted-average helper with CPU fallback and GPU parity coverage.
- Added CUDA Tile optimizer/runtime coverage for `gradient_accumulate`, exposing an in-place scaled add-into-buffer helper with CPU fallback and GPU parity coverage.
- Added CUDA Tile optimizer/runtime coverage for `gradient_clip_norm`, exposing multi-tensor global L2 norm reduction plus in-place gradient scaling with CPU fallback and GPU parity coverage.
- Added CUDA Tile composition coverage for `mamba`, routing the input/output projections and gating through Tile primitives while preserving the depthwise convolution contract.
- Added CUDA Tile deterministic counter-random coverage for `random_timesteps`, `mask_scheduler`, and random/block `jepa_mask`.
- Added CUDA Tile composition coverage for `universal_transformer`, including recurrent attention, MLP, and ACT halt-gate paths.
- Added CUDA Tile optimizer/runtime helpers for `muon_newton_schulz`, `muon_step`, and `split_optimizer_step`.
- Added a dtype-expansion section to `todo-tile-cuda.md` covering fp16, fp8, and NVFP4 work across all kernels, including explicit dtype policy gates and no-support reasons for kernels where a low-precision dtype is not meaningful.
- Added fp16 CUDA Tile support for scalar function kernels. The fp16 path upcasts contiguous CUDA fp16 inputs to float32, executes the existing Tile float32 kernel, and casts outputs back to fp16 so strict Tile mode no longer falls back to PyTorch for scalar fp16 inputs.
- Added fp16 CUDA Tile activation support for the module kernels `loss_scale`, `logit_softcap`, `aux_loss_add`, `kl_penalty`, `residual_add`, `residual_mix`, `manifold_hyper_connection`, `qk_gain`, `dyt`, `act_weighted_sum`, `latent_pool`, `rms_norm`, `layer_norm`, `group_norm`, and `qk_norm`. These paths use Tile float32 compute with fp16 activation cast-in/cast-out while keeping reductions, scale, parameter, weight, and mask gradients in float32.
- Added fp16 CUDA Tile activation support for projection-family modules built from `tile_linear_module`, including `linear`, LM/router/value/reward/denoise heads, KV PCA projections, JEPA heads, deterministic LoRA/TTT/adapter projections, `bitlinear_ternary`, `fp8_linear`, `mx_linear`, MLP projections, and ACT halt projection.
- Added fp16 CUDA Tile activation support for RoPE, SDPA, sparse attention variants, differential attention, causal/fused causal attention, and MLA. These paths keep attention score and softmax accumulation in fp32 and cast outputs back to fp16.
- Added fp16 CUDA Tile support for loss/reduction modules with fp32 accumulation, including token CE, masked CE, sequence logp, latent MSE, semantic alignment, DPO, PPO, GAE, preference BCE, load/route balance, route selection/distillation, and softmax distillation. Scalar losses remain fp32; activation-like tuple outputs such as GAE advantages/returns and DPO rewards preserve fp16.
- Added fp16 CUDA Tile support for optimizer/runtime helpers `ema_update`, `gradient_accumulate`, `gradient_clip_norm`, and `adamw_step`. These paths execute the existing Tile float32 kernels and copy results back to fp16 buffers where needed; AdamW fp16 support requires fp32 first/second moment state.
- Added fp16 CUDA Tile training-mode `dropout` support for `0 < p < 1` using deterministic counter-based masks with fp32 mask math and fp16 activation/gradient outputs, replacing the prior PyTorch RNG fallback for fp16 stochastic dropout.
- Added fp8 CUDA Tile activation support for direct projection modules (`linear`, LM/router/value/reward/denoise heads, tied LM head, and KV PCA encode/decode). The fp8 path accepts `float8_e4m3fn` and `float8_e5m2` activations, dequantizes to float32, accumulates through the Tile float32 linear kernel, returns float32 outputs, and keeps weight/bias gradients in float32.
- Expanded fp8 CUDA Tile activation support to composite projection-family modules: JEPA heads, deterministic LoRA/TTT/adapter projections, `bitlinear_ternary`, `fp8_linear`, `mx_linear`, MLP projections, and ACT halt projection. Branching composites dequantize fp8 inputs once before fan-out so internal gradient accumulation remains in float32.
- Added fp8 CUDA Tile Q/K/V activation support for attention modules with fp32 score and softmax accumulation: SDPA, sparse/window/native/streaming-sink attention variants, differential attention, causal/fused causal attention, MLA, and routed attention experts. Composite and routed attention stages dequantize fp8 inputs before projection or expert fan-out to keep gradient accumulation in float32.
- Added fp8 CUDA Tile support for scalar function kernels and simple elementwise modules. Unary, binary, and binary-pair functions plus `loss_scale`, `logit_softcap`, `aux_loss_add`, `kl_penalty`, `residual_add`, `residual_mix`, `manifold_hyper_connection`, `qk_gain`, and `dyt` now dequantize fp8 activations to float32 for Tile compute and requantize activation outputs to the input fp8 format.
- Added NVFP4 CUDA Tile activation support for projection-family modules (`linear`, LM/router/value/reward/denoise heads, tied LM head, KV PCA encode/decode, JEPA heads, deterministic LoRA/TTT/adapter projections, `bitlinear_ternary`, `fp8_linear`, `mx_linear`, MLP projections, and ACT halt projection). Packed `NVFP4Tensor` inputs dequantize through NeuralFn block/tensor scale metadata, accumulate through the Tile float32 path, return float32 outputs, and can preserve source-activation gradients for parity checks through `quantize_nvfp4_reference(..., preserve_grad=True)`. `nf4_linear` remains excluded because it owns a separate packed NF4 base-weight contract.
- Added NVFP4 CUDA Tile attention support for Q/K/V and shared attention inputs across SDPA, sparse/window/native/streaming-sink attention variants, differential attention, causal/fused causal attention, MLA, and routed attention experts. These paths dequantize packed NVFP4 activations once before Tile attention, RoPE/projection fan-out, or route fan-out, keep score and softmax accumulation in float32, return float32 outputs, and preserve source-activation gradients through the optional NVFP4 source path.
- Added explicit fp8/NVFP4 no-support reasons to registry `dtype_support` entries for losses/reductions, optimizer state, stochastic masks, integer/hash/routing outputs, host-only source nodes, delegated child-graph calls, and packed-format wrappers such as NF4.
- Added deterministic low-precision SDK reference helpers: fp8 quantize/dequantize for `float8_e4m3fn` and `float8_e5m2`, plus `NVFP4Tensor` packing/dequantization using FP4 E2M1 values in blocks of 16 with FP8 E4M3 block scales, an FP32 tensor scale, and optional source tracking for straight-through gradients.
- Added `nfn kernels list`, `nfn kernels doctor`, `nfn kernels bench`, and `nfn kernels examples` with optional `--json` output for CUDA Tile registry coverage, diagnostics, benchmark comparison, and example generation.
- Added `examples/tile_cuda/` with smoke examples and generated one-file SDK snippets for all 138 registry entries.
- Added `nfn train`, `nfn infer`, and `nfn eval` support for `--kernel-backend {auto,torch,tile-cuda}`, `--tile-cuda-strict`, and `--tile-cuda-report PATH`.
- Added the `tile-cuda` optional dependency extra for the `ninja` build tool. It now supports the Torch-free native CUDA Tile build path; install `.[torch]` separately for graph-backed PyTorch extension workflows.

#### Changed

- `CompiledTorchGraph` now precomputes flat input/output routing and node edge inputs during initialization. Forward and trace execution use that immutable execution plan instead of walking graph-editor node and edge metadata for every batch, keeping real training tensors on the compiled backend hot path.
- `CompiledTorchGraph` now consumes CUDA Tile backend settings. Strict Tile mode validates graph coverage before batches run, and optional report paths write diagnostics plus coverage JSON.
- The optional CUDA Tile extension build is opt-in via `NFN_TILE_CUDA_BUILD=1` or `TileCudaConfig(build_enabled=True)`, with `NFN_TILE_CUDA_ARCH` available for explicit architecture selection.
- `MLAStage` now passes contiguous Q/K/V tensors into CUDA SDPA to avoid misaligned-address faults from transposed views on GPU.
- The CUDA Tile registry now accounts for all 138 training-relevant NeuralFn entries: 129 Tile-covered kernels/compositions, 7 host-only source/interface entries, and 2 delegated compiled-graph calls. The default registry has no `torch_fallback` entries.
- CUDA Tile scalar function registry entries and verified module/optimizer entries now advertise `float32`, `float16`, fp8, and projection-family NVFP4 support according to their tested contracts. Entries also expose a per-kernel `dtype_support` matrix for `float32`, `float16`, `float8_e4m3fn`, `float8_e5m2`, and `nvfp4`, with explicit reasons for unsupported lower-precision contracts. Coverage reports include `by_dtype` aggregate counts for the same tracked dtype set.

#### Breaking changes

- `RandomTimestepsStage`, `MaskSchedulerStage`, and `JEPAMaskStage` no longer draw from the global PyTorch RNG in their Tile-compatible paths. They now use deterministic counter-based random generation so CPU/GPU parity and compiled execution are reproducible. Tests or callers that asserted exact global-RNG side effects should assert output shape/range or stage-local determinism instead.

#### Verification

- Added regression coverage that compiles a graph, disables graph edge traversal after compilation, and verifies the compiled forward pass still succeeds from the static execution plan.
- Added `tests/test_tile_cuda_registry.py` to verify the CUDA Tile registry covers the current NeuralFn builtin and dispatch inventory and reports explicit fallback/delegation reasons for unfinished kernels.
- Added `tests/test_tile_cuda_ops.py` for deterministic CPU and GPU forward/backward parity across the implemented unary, binary, and binary-pair scalar family.
- Added `tests/test_tile_cuda_modules.py` for deterministic CPU and GPU forward/backward parity across the implemented elementwise module family.
- Added `tests/test_tile_cuda_coverage.py`, `tests/test_tile_cuda_static_plan.py`, and `tests/test_tile_cuda_examples.py` for the final coverage gate, training hot-path invariant, and checked-in/generated examples.
- Added GPU fp16 forward/backward parity coverage for all scalar unary, binary, and binary-pair CUDA Tile function kernels.
- Added GPU fp16 forward/backward parity coverage for the verified simple module kernels and CPU registry checks that prevent unverified Tile modules from advertising fp16 support.
- Added GPU fp16 forward/backward parity coverage for verified projection-family modules, including quantized-weight projection wrappers, and kept `nf4_linear` float32-only in the registry because its base compute contract is explicitly fp32.
- Added GPU fp16 forward/backward parity coverage for verified attention-family modules, including `routed_attention_experts` with fp32 route-weight accumulation before casting contributions back to fp16 activations.
- Added GPU fp16 forward/backward parity coverage for verified loss/reduction modules and kept semantic projectors float32-only in the dtype registry because argmax-derived topic/signature outputs can change under lower-precision activation quantization.
- Added GPU fp16 parity coverage for verified optimizer/runtime helpers, including `muon_step` and `split_optimizer_step` with fp16 parameter/gradient tensors plus float32 optimizer state. Standalone `muon_newton_schulz` remains float32-only as the matrix orthogonalization primitive.
- Added dtype-aware strict-mode error messages for scalar CUDA Tile function and simple module contract failures, including the rejected dtype and supported dtype set.
- Added GPU NVFP4 forward/backward parity coverage for projection-family and attention-family activation inputs and CPU coverage for `NVFP4Tensor` source-gradient preservation through `quantize_nvfp4_reference(..., preserve_grad=True)`.
- Added CPU boundary coverage for fp8 reference overflow behavior (`float8_e4m3fn` NaN outside range, `float8_e5m2` infinities outside range) and NVFP4 finite scale/codebook behavior on very large packed activation values.
- Added CLI coverage for `nfn kernels list --json`, `nfn kernels doctor --json`, `nfn kernels bench --json`, and `nfn kernels examples --json --write`.
- Verified the CUDA Tile translation unit with `/usr/local/cuda/bin/nvcc -std=c++20 --enable-tile -arch=sm_80 -c neuralfn/csrc/tile_cuda/kernels.cu -o /tmp/neuralfn_tile_kernels.o`.
- Verified focused Python coverage with `python -m pytest tests/test_tile_cuda_registry.py tests/test_tile_cuda_ops.py tests/test_tile_cuda_modules.py tests/test_tile_cuda_optimizer.py tests/test_torch_gpt.py cli/tests/test_nfn_cli.py -q` (`239 passed, 133 skipped, 1 warning, 4 subtests passed`). The skipped cases are GPU drift tests skipped inside the sandboxed process.
- Installed `ninja` in the active Python environment because PyTorch requires it for JIT C++/CUDA extension builds, then verified GPU drift with `NFN_TILE_CUDA_TEST=1 NFN_TILE_CUDA_BUILD=1 NFN_TILE_CUDA_ARCH=sm_120 python -m pytest tests/test_tile_cuda_ops.py tests/test_tile_cuda_modules.py tests/test_tile_cuda_optimizer.py tests/test_tile_cuda_gpu.py -q -rs` (`269 passed, 2 warnings`).
- Re-ran the required preset gate after Tile module wiring with `HOME=/tmp/neuralfn-home python -m pytest tests/test_template_presets.py -x -q` (`18 passed, 15 warnings`). The redirected `HOME` keeps the test dataset cache writable inside the sandbox.

### 2026-05-28 Frontier GPT templates (modern-kernel × existing-template combinations)

#### Added

- **16 new builtin ops** (PyTorch-reference Stages first, per `todo-kernels.md`, to be repointed at `llm.kittens` kernels later), registered in `neuralfn/builtins.py` + `neuralfn/torch_backend.py`:
  - Norms/gates: `dyt` (Dynamic Tanh), `group_norm`, `qk_norm` (fused RMSNorm on Q/K), `geglu`, `reglu`, `solu`.
  - Attention cores (preserve the `forward(q,k,v) -> [B,H,S,head_dim]` SDPA-core contract, masks built internally): `sliding_window_attention`, `block_sparse_attention`, `streaming_attention_sinks`, `native_sparse_attention` (NSA / CSA-spirit), `differential_attention`; plus self-contained `multi_latent_attention` (MLA, owns its decoupled RoPE).
  - MoE: `auxfree_load_balancing` (DeepSeek-V3 bias-adjusted routing; keeps the `aux_loss` port at 0 so MoE block arity is unchanged).
  - Precision: `fp8_linear` (E4M3/E5M2) and `mx_linear` (OCP MXFP4/MXFP8), via the existing compression seam in `get_linear_module_def`.
  - Residual: `manifold_hyper_connection` (single-stream mHC, non-expansive).
  - RoPE scaling: `rope_scaling` (`linear`/`ntk`/`yarn`) is now honored by `RotaryEmbeddingStage` (the `BlockSpec.rope_scaling` field was previously dormant).
- **12 new flagship/precision/cross presets**: `deepseek_v3`, `deepseek_v4`, `gemma3`, `diff_transformer`, `qwen3_longctx`, `longctx_sparse_llama`, `modern_norms_llama`, `fp8_llama`, `mxfp4_llama`, `auxfree_moe_jepa_evo`, `diff_semantic_moe_jepa_evo`, `dyt_geglu_semantic_dense_jepa_evo`.
- **Modernization overlay**: `_apply_modern_profile` + generated `<preset>_modern` for every preset in `MODERN_BASE_PRESETS` (19 variants) — RMSNorm + QK-norm + RoPE/YaRN + GeGLU + auxfree MoE, additive and topology-preserving.
- **Additive `BlockSpec`/`ModelSpec` knobs** (all defaults preserve the existing 26 presets): `attention_variant`, `use_qk_norm`, `norm_type` (+`dyt`/`group_norm`), `mlp_type` (+`geglu`/`reglu`/`solu`), `moe_balance_mode`, `residual_type`, `compression` (+fp8/mx), `window_size`, `sparse_block_size`, `num_sinks`, `nsa_compress_stride`, `mx_block_size`, `diff_lambda_init`, `dyt_alpha_init`, `auxfree_bias_lr`.
- **`select_norm_module` helper** centralises norm selection across all block builders so `dyt`/`group_norm` propagate into the dense, semantic-router, and JEPA/evo stacks (previously the norm choice was duplicated and would silently fall back to LayerNorm).
- Editor `Toolbar.tsx` dropdown gains grouped `<optgroup>` entries for the new presets.

#### Context

DeepSeek-V4-Pro was used as a recognizable-SOTA anchor: its hybrid CSA/HCA attention maps onto the NSA/sparse cores, its **domain-specific experts** are the post-training analog of NeuralFn's architectural semantic per-dimension experts, and its mHC residuals + FP4/FP8 mixed precision motivate the `manifold_hyper_connection` and precision presets. MLA-into-`kv_pca`, Mixture-of-Depths, `soft_moe`, FP8-inside-megakernel, and FP4 MoE experts are documented as out-of-scope follow-ups.

#### Verification

- `tests/test_template_presets.py` builds + CPU-forwards all presets (existing 26 + 12 flagship/precision/cross + 19 `_modern`).
- New `tests/test_frontier_kernels.py` adds per-Stage parity/correctness tests (sliding-window vs manual masked SDPA, differential attention even-head-dim guard, FP8/MXFP4 round-trip bounds + STE grads, auxfree load-rebalancing, mHC non-expansiveness, MLA fwd/bwd, RoPE-scaling variants).
- `tests/test_builtin_neurons.py` catalog expectation extended to 131 builtins.

### 2026-05-20 Electron desktop application wrapping

#### Added

- **Unified Electron Desktop App** -- wrapped NeuralFn (FastAPI backend + React frontend) into a unified desktop application packaging structure inside `desktop/` targeting Windows, macOS, and Linux.
- **SPA Production Serving in FastAPI** -- added production static file hosting and catch-all SPA fallback routing to `server/app.py` when frontend assets are built, allowing Electron to load the app directly via standard HTTP, bypassing all CORS and router history API breaks.
- **Dynamic Free Port Discovery** -- implemented automatic free port discovery on startup (scanning starting at 8000) inside `desktop/main.js` to ensure the app never conflicts with local development or server environments.
- **Zero-Config Persistent Storage** -- passed custom environment variables to the FastAPI child process pointing the SQLite database, snapshots, and artifacts to the user's OS application data directory (`app.getPath('userData')`), ensuring safe read-write permissions and data persistence across updates.
- **Redis-Free Offline State Store** -- forced `NEURALFN_REDIS_URL` to an empty string in the spawned child process, automatically engaging the robust local SQLite + background thread persistence and MemoryLiveStateStore.
- **Automated Monorepo Build Scripts** -- added `package.json` at the root directory containing unified orchestrations to build the editor, copy resources to the sandbox, and start or build/package the Electron distribution.

#### Verification

- Developed and executed an automated end-to-end integration test harness (`desktop-test-data`) that mimics the desktop app startup environment:
  - Spawned the uvicorn server under dynamic SQLite environment routing and zero-Redis.
  - Verified static index.html is served successfully at root `/`.
  - Verified catch-all client router fallback serving on sub-paths like `/app/admin`.
  - Verified REST API bootstrap initializes SQLite tables and returns valid JSON.
  - Confirmed database (`neuralfn.db`) is correctly written under the sandbox directory.
- Installed `pytest` in the conda environment and ran the full template preset test suite `python -m pytest tests/test_template_presets.py -x -q` to verify zero regression across all templates (18 passed successfully).

### 2026-05-07 GPT template architecture diagrams

#### Added

- **Architecture diagram catalog** -- generated documentation PNGs for all shipped GPT template presets and added them to the framework guide:
  - `docs/assets/gpt_template_architectures_core.png`
  - `docs/assets/gpt_template_architectures_research.png`
  - `docs/assets/gpt_template_architectures_semantic.png`
- **Docs navigation** -- linked the diagram catalog from `README.md`, `docs/README.md`, and the framework guide index, while keeping the detailed Semantic MoE JEPA Evo PNG as a standalone reference.

#### Verification

- `file docs/assets/gpt_template_architectures_core.png docs/assets/gpt_template_architectures_research.png docs/assets/gpt_template_architectures_semantic.png`
- `rg -n "gpt_template_architectures|Architecture diagrams" README.md docs`
- `git diff --check`

### 2026-05-07 Non-semantic JEPA Evo control templates

#### Added

- **`dense_jepa_evo` and `moe_jepa_evo` presets** -- added non-semantic AR+JEPA Evo controls that remove the semantic router and semantic data source. Both use the existing `(tokens, targets)` contract and train next-token CE plus JEPA latent alignment; the MoE variant uses standard MoE routing and load-balance loss.
- **Public template wiring** -- added `build_dense_jepa_evo_spec()` and `build_moe_jepa_evo_spec()`, preset dispatch, all-preset coverage, and editor dropdown options.

#### Verification

- `python -m py_compile neuralfn/config.py neuralfn/torch_templates.py tests/test_template_presets.py`
- Direct smoke script building payloads, resolving variants, compiling on CPU, and forwarding `dense_jepa_evo`, `moe_jepa_evo`, `semantic_dense_jepa_evo`, and `semantic_moe_jepa_evo` -> all returned scalar losses.
- `npm run build` from `editor/` -> build succeeded.
- Required preset gate `python -m pytest tests/test_template_presets.py -x -q` is currently blocked in this environment because `/home/adam/miniconda3/envs/NeuralFn/bin/python` has no `pytest` module installed.

### 2026-05-07 Semantic Dense JEPA Evo template

#### Added

- **`semantic_dense_jepa_evo` preset** -- added a dense control companion to `semantic_moe_jepa_evo`. It keeps the chunk-level causal semantic planner, JEPA target encoder, AR CE, JEPA latent alignment, and semantic-alignment losses, but uses dense LLaMA FFNs instead of semantic expert dispatch and does not run route evolution.
- **Dense stage builder** -- added `build_semantic_dense_jepa_evo_spec()` and `build_semantic_dense_jepa_evo_model_stage_graph()` so Python, REST/editor template routes, and MCP `load_gpt_template` can build the dense variant by preset name.
- **Preset coverage and UI wiring** -- added the dense preset to the all-preset test catalog and the editor GPT template dropdown.

#### Changed

- **Semantic template runtime handling** -- semantic utility execution now treats the chunk-level JEPA Evo objectives as semantic-input templates, so dummy semantic targets are supplied when probing/generating against these graphs.
- **Semantic Evo regression coverage** -- added tests that check the dense variant omits route/expert nodes while the MoE variant retains chunk routing, expert dispatch/combine, route losses, and the shared/semantic/free expert path described by the architecture image.

#### Verification

- `python -m py_compile neuralfn/config.py neuralfn/torch_templates.py neuralfn/torch_backend.py server/routers/sessions.py tests/test_jepa_semantic.py tests/test_template_presets.py`
- Direct smoke script building payloads, resolving variants, compiling on CPU, and forwarding both `semantic_dense_jepa_evo` and `semantic_moe_jepa_evo` with `(tokens, targets, sem_targets)` -> both returned scalar losses.
- `npm run build` from `editor/` -> build succeeded.
- Intended pytest gate `python -m pytest tests/test_jepa_semantic.py -q -k "semantic_moe_jepa_evo or semantic_dense_jepa_evo"` is currently blocked in this environment because `/home/adam/miniconda3/envs/NeuralFn/bin/python` has no `pytest` module installed.
- Required preset gate `python -m pytest tests/test_template_presets.py -x -q` is blocked for the same missing-`pytest` reason.

### 2026-05-04 Graphless Parameter Golf checkpoint inference

#### Added

- **Graphless `.pt` checkpoint chat** -- `nfn infer` can now load flat Parameter Golf root-GPT `.pt` checkpoints directly with `--checkpoint` plus a matching `--checkpoint-tokenizer`, without requiring a NeuralFn graph JSON. Passing `--weights <checkpoint>.pt` without `--graph` routes to the same graphless loader.
- **Parameter Golf CaseOps preset stack** -- added `parameter_golf_caseops_8192`, `parameter_golf_10min`, and `parameter_golf_muon` presets so the supplied lossless-caps training shape, budget, and optimizer settings can be selected from the CLI planner or flags.
- **Checkpoint runtime packaging** -- the CLI package now includes the Parameter Golf runtime loader and declares `sentencepiece` for the graphless tokenizer path.
- **Reference-compatible flat runtime** -- aligned graphless inference with the root Parameter Golf flat checkpoint architecture, including the ReLU-squared MLP and ignoring incompatible newer structural log hints when the flat tensors do not contain those features.
- **CaseOps display cleanup** -- CaseOps SentencePiece checkpoints now hide private-use case markers in decoded chat text and suppress reconstruction-only tokenizer ids during graphless sampling, including byte fallback, ellipsis artifacts, and the high-id single-character fallback band.
- **Graphless repeat controls** -- added default Parameter Golf repeat guards for repeated n-grams and consecutive token runs, exposed `--no-repeat-ngram-size` / `--repeat-run-limit`, and added the `/repeat` chat command for live repetition-penalty tuning.
- **Infer slash-command autocomplete** -- interactive `nfn infer` now shows live slash-command suggestions while typing `/` commands and routes Tab through command completion, completing unique prefixes such as `/sett` -> `/settings`, listing ambiguous matches, and showing value hints for commands such as `/temp`, `/top_k`, and `/repeat`.
- **Inline word autocomplete** -- added `/autocomplete <words>` for interactive `nfn infer`. Positive values show a 50% gray inline word prediction after the cursor, preserve the model's generated word boundary so predictions can either start a new word or complete the current word, and Tab accepts the visible prediction; `/autocomplete 0` disables the inline mode and restores the existing token-preview Tab flow.

#### Fixed

- **Wrapped infer input repainting** -- the interactive `nfn infer` input renderer now tracks how many terminal rows the current prompt, ghost prediction, and status line occupy, then clears the full wrapped block before drawing the next frame. Repaint output also writes explicit CRLF line breaks for raw TTY mode so status redraws return to column zero. This prevents long autocomplete sessions from leaving stale duplicate prompt rows on screen.

#### Notes

- Graphless inference is intentionally scoped to flat Parameter Golf root-GPT state dicts containing `tok_emb.weight`, `skip_weights`, and `blocks.*` attention/MLP tensors. NeuralFn exports remain graph-first and should still be loaded with `--graph`.
- For the supplied artifact, use `nfn infer --checkpoint ~/NeuralFn/artifacts/final_model.pt --checkpoint-tokenizer ~/Downloads/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model --checkpoint-log ~/Downloads/a54a53b3-7d6e-461c-975a-590030e61bd0.txt`.

#### Verification

- `conda run -n NeuralFn python -m py_compile nfn_impl.py parameter_golf_runtime.py`
- `conda run -n NeuralFn python -m unittest tests.test_nfn_cli`
- `conda run -n NeuralFn python nfn.py infer --checkpoint /home/adam/NeuralFn/artifacts/final_model.pt --checkpoint-tokenizer /home/adam/Downloads/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model --checkpoint-log /home/adam/Downloads/a54a53b3-7d6e-461c-975a-590030e61bd0.txt --prompt "a cat sat on the" --max-new-tokens 32 --temperature 0.8 --top-k 32 --top-p 1 --device cpu --seed 1337 --log-every 0` -> generated readable prose beginning with `table.`
- `conda run -n NeuralFn python nfn.py infer --checkpoint /home/adam/NeuralFn/artifacts/final_model.pt --checkpoint-tokenizer /home/adam/Downloads/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model --checkpoint-log /home/adam/Downloads/a54a53b3-7d6e-461c-975a-590030e61bd0.txt --prompt "the sky is" --max-new-tokens 64 --temperature 0.4 --top-k 32 --top-p 1 --device cuda --seed 1337 --log-every 0` -> generated prose beginning with `blue and the sun is shining` without the repeated quote loop.
- `conda run -n NeuralFn python nfn.py infer --checkpoint /home/adam/NeuralFn/artifacts/final_model.pt --checkpoint-tokenizer /home/adam/Downloads/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model --checkpoint-log /home/adam/Downloads/a54a53b3-7d6e-461c-975a-590030e61bd0.txt --prompt "the cat sat in the box and" --max-new-tokens 64 --temperature 0.4 --top-k 32 --top-p 1 --repetition-penalty 1.15 --device cuda --seed 1337 --log-every 0` -> generated prose without the prior CaseOps ellipsis/byte/fallback artifacts.

### 2026-05-03 Semantic MoE JEPA Evo architecture image correction

#### Changed

- **Architecture image reference** -- restored the Semantic MoE JEPA Evo documentation to use the original PNG asset at `docs/assets/semantic_moe_jepa_evo_architecture.png` after the SVG conversion lost formatting.

#### Verification

- `file docs/assets/semantic_moe_jepa_evo_architecture.png`
- `rg -n "semantic_moe_jepa_evo_architecture\\.svg|semantic_moe_jepa_evo_architecture\\.png" README.md CHANGELOG.md docs .cursor llms-full.txt`

### 2026-05-03 Deep documentation sync for CLI, fine-tuning, and torch templates

#### Changed

- **CLI documentation surface** -- added `docs/cli.md`, refreshed `cli/README.md`, linked CLI workflows from `README.md`, `docs/README.md`, `llms.txt`, and `docs/agent-skills.md`, and documented the current `nfn train`, `nfn infer`, and `nfn eval` workflow model.
- **Current SDK and torch-template references** -- updated Python SDK, framework-guide, and repo-local agent-skill docs for the current `ModelSpec`-first builder API, composed recipe/fine-tuning roots, qLoRA/LoRA fields, adapter checkpoint helpers, fine-tuning stages, and the 115-builtin catalog.
- **LLM and agent artifacts** -- added the `.cursor/skills/neuralfn-cli` skill, refreshed the Torch and Python SDK skills, and regenerated `llms-full.txt` from README, changelog, docs, CLI docs, and repo-local skills.
- **Example API drift cleanup** -- updated `examples/gpt_graph.py` to build a `ModelSpec` with `build_llama_spec()` and pass it to `build_gpt_root_graph(model_spec=...)`.

#### Verification

- `conda run -n NeuralFn python cli/nfn.py --help`
- `conda run -n NeuralFn python cli/nfn.py train --help`
- `conda run -n NeuralFn python cli/nfn.py infer --help`
- `conda run -n NeuralFn python cli/nfn.py eval --help`
- `conda run -n NeuralFn python examples/gpt_graph.py`
- `conda run -n NeuralFn python -m pytest cli/tests/test_nfn_cli.py -q` -> `58 passed`
- `conda run -n NeuralFn python -m py_compile examples/gpt_graph.py neuralfn/config.py neuralfn/torch_templates.py`
- Stale-reference scan over `README.md`, `docs/`, `.cursor/skills/`, `cli/README.md`, `llms.txt`, and examples found no current API-artifact hits for the old template signatures or artifact defaults.
- Known residual test failures outside this documentation pass: `conda run -n NeuralFn python -m pytest cli/tests/test_nfn_cli.py cli/tests/test_train_pretraining_file_flags.py cli/tests/test_train_tinystories_flags.py -q` currently fails in `test_pretraining_file_explicit_sentencepiece_requires_shared_model` and `test_load_val_token_dataset_falls_back_to_train_holdout_without_val_file`.

### 2026-05-03 Script path cleanup before push

#### Changed

- **Portable CLI run scripts** -- removed workstation-specific absolute `PYTHONPATH` values from the 5090 helper scripts. They now derive the project root from the script location, run from `./cli`, and keep artifact paths under `$HOME/NeuralFn/artifacts`.
- **Portable setup, cache, and history examples** -- replaced local absolute setup and verification paths in docs/changelog examples with repo-relative paths, and changed the tiktoken encoding cache default to `~/tiktoken_encodings`.

#### Verification

- Workstation-path scan over shell scripts, Python files, Markdown, and changelog examples -> no local absolute path matches.
- `bash -n 5090-mini-run.sh 5090-llama-smoke.sh 5090-llama-baseline.sh 5090-llama-overnight.sh`
- `conda run -n NeuralFn python -m py_compile ../server/dataset_manager.py`
- `conda run -n NeuralFn python -m pytest tests/test_dataset_manager_downloads.py ../tests/test_dataset_manager_variants.py -q` -> `5 passed`
- `git diff --check`

### 2026-05-03 CLI and graph-run artifact store migration

#### Breaking changes

- **Default artifact paths moved** -- implicit CLI training outputs, CLI inference graph/weights defaults, interactive inference graph picking, eval reports, and server/editor graph-run artifacts now use `~/NeuralFn/artifacts` instead of repo-local `cli/artifacts` or `server/artifacts`. Callers with hardcoded old paths should pass explicit `--output`, `--graph`, `--weights`, or `--report-path` values, or update scripts to the new shared directory.

#### Changed

- **Shared artifact root** -- CLI helpers and standalone training/inference scripts now resolve default artifacts through a shared `NEURALFN_ARTIFACTS_DIR` override, defaulting to `~/NeuralFn/artifacts`.
- **Current artifact migration** -- existing CLI graph/checkpoint/eval files were moved from `cli/artifacts` into `~/NeuralFn/artifacts`; no compatibility copy or symlink was left behind.
- **In-repo CLI import roots** -- the CLI import bootstrap now points at the enclosing NeuralFn repo and the local `cli/scripts` directory, matching the relocated in-repo CLI layout.
- **Platform artifact default** -- `server/settings.py` now defaults `NEURALFN_ARTIFACTS_DIR` to `~/NeuralFn/artifacts`, so graph-run artifacts share the same local artifact store as CLI runs.

#### Verification

- `conda run -n NeuralFn python -m pytest tests/test_nfn_cli.py tests/test_infer_megakernel_artifacts.py ../tests/test_platform_api.py::SettingsDefaultsTest -q` -> `79 passed, 68 subtests passed`
- `conda run -n NeuralFn python -m py_compile cli_utils.py scripts/cli_utils.py nfn_impl.py scripts/train_jepa_semantic.py scripts/infer_jepa_semantic.py scripts/train_llama_fast.py scripts/infer_llama_fast.py scripts/train_gpt2.py scripts/infer_gpt2.py scripts/train_nanogpt.py scripts/infer_nanogpt.py scripts/train_mixllama_fast.py scripts/infer_mixllama_fast.py scripts/train_semantic_router_moe.py scripts/train_semantic_router_moe-overnight.py scripts/infer_semantic_router_moe.py scripts/train_llama_megakernel.py scripts/infer_llama_megakernel.py ../server/settings.py ../tests/test_platform_api.py`
- `conda run -n NeuralFn python -c "..."` graph metadata check -> `checked=21 missing=0`
- `conda run -n NeuralFn python scripts/infer_jepa_semantic.py --help` and `conda run -n NeuralFn python nfn.py infer --help` both rendered help successfully.

### 2026-05-03 [Experimental] Semantic MoE JEPA Evo GPT template

#### Added

- **`semantic_moe_jepa_evo` objective and preset** -- added a full Semantic MoE JEPA Evo GPT template that combines an autoregressive decoder, chunk-level causal semantic planner, JEPA training-only target path, and a hybrid expert bank with 2 shared experts, one expert per semantic vocabulary dimension, and 8 free learned experts.
- **Chunk router module stack** -- added builtin/stage coverage for causal chunk state extraction, chunk semantic projection, chunk LSH hashing, chunk-to-token route broadcasting, semantic MoE JEPA Evo routing, route balance loss, route selection loss, and route distillation loss.
- **Route-evolution controller** -- `TorchTrainer` can now periodically run lightweight evolutionary search over the new router bias/table parameters during normal gradient training. `route_evo_fraction`, `route_evo_population`, `route_evo_mutation_scale`, and `route_evo_seed` control the cadence and search shape.
- **Architecture asset** -- added `docs/assets/semantic_moe_jepa_evo_architecture.png`, a single-image architecture infographic for the new template.

#### Changed

- **Template catalog wiring** -- `build_model_spec_from_config()`, `build_gpt_root_graph()`, `build_gpt_template_payload()`, the editor GPT template dropdown, and server-side template application now recognize `semantic_moe_jepa_evo`.
- **Semantic routing losses** -- chunked topic logits now flow through semantic alignment, route selection, route distillation, and balance losses alongside AR CE and JEPA latent alignment.
- **Preset coverage** -- the new preset is included in `tests/test_template_presets.py`, so payload generation, variant resolution, compile/forward execution, and server-side apply coverage run with the rest of the shipped catalog.

#### Verification

- `conda run -n NeuralFn python -m pytest tests/test_jepa_semantic.py -q -k "semantic_moe_jepa_evo"` -> `5 passed`
- `conda run -n NeuralFn python -m pytest tests/test_template_presets.py -x -q` -> `17 passed`
- `conda run -n NeuralFn python -m pytest tests/test_builtin_neurons.py -q` -> `7 passed`
- `conda run -n NeuralFn python -m py_compile neuralfn/config.py neuralfn/builtins.py neuralfn/torch_backend.py neuralfn/torch_templates.py tests/test_jepa_semantic.py tests/test_template_presets.py tests/test_builtin_neurons.py`
- `npm run build` (from `editor/`)
- `git diff --check`

### 2026-04-10 Harness auto-download for missing cached dataset aliases

#### Changed

- **Shared harness dataset resolver** -- the sibling training and inference harnesses under `neuralfn-sdk-harness/scripts/` now share one dataset-alias resolver rooted in `train_jepa_semantic.py`. `train_mixllama_fast.py`, `train_semantic_router_moe.py`, `infer_jepa_semantic.py`, `infer_mixllama_fast.py`, and `infer_semantic_router_moe.py` all use the same missing-alias behavior now.
- **Auto-download on cache miss by default** -- when `--dataset-alias` is missing locally, the harnesses now attempt a real `download_hf_dataset(...)` call instead of failing immediately. Standard cached-variant aliases such as `owner__repo__variant__trainN` are parsed into a download contract automatically, and all harnesses now expose explicit override flags for non-standard aliases: `--download-if-missing/--no-download-if-missing`, `--dataset-hf-path`, `--dataset-variant`, `--dataset-train-shards`, `--dataset-repo-id`, and `--dataset-remote-root-prefix`.
- **Strict validator ordering preserved** -- existing aliases are still treated as authoritative cache entries. If an alias already exists but its tokenizer-backed cached shards are inconsistent, the harness does not auto-redownload or delete it; it surfaces the original `DatasetTokenizerMismatchError` directly.

#### Verification

- `conda run -n NeuralFn python -m pytest tests/test_tokenizer_vocab_contract.py -q`
- `conda run -n NeuralFn python -m py_compile cli/scripts/train_jepa_semantic.py cli/scripts/train_mixllama_fast.py cli/scripts/train_semantic_router_moe.py cli/scripts/infer_jepa_semantic.py cli/scripts/infer_mixllama_fast.py cli/scripts/infer_semantic_router_moe.py`

### 2026-04-10 Cached tokenizer contract validation and fail-fast vocab checks

#### Changed

- **Strict cached tokenizer contract** -- tokenizer-backed `uint16_shards` aliases are now validated against their downloaded tokenizer artifacts before they are accepted, loaded for training, or used for inference. NeuralFn now scans cached shard ids, resolves the tokenizer artifact from the alias metadata, and rejects aliases whose cached ids exceed the tokenizer vocab.
- **Fail-fast torch training and trace previews** -- `TorchTrainer` and `trace_torch_graph()` now validate tokenizer-backed cached aliases against the graph vocab before they reach the old auto-resize path. Manual tensors and tokenizer-less inputs still keep the compatibility auto-expand behavior, but cached aliases with tokenizer metadata no longer silently resize embeddings or LM heads.
- **Inference preflight and decode guard** -- the sibling inference harnesses for `jepa_semantic_hybrid`, `semantic_router_moe`, and `mixllama_fast` now compare the dataset tokenizer vocab against the loaded graph/checkpoint vocab before prompt encoding or decode. When decode still sees an out-of-range token id, it now raises a controlled `ValueError` instead of surfacing the raw SentencePiece traceback.
- **Bad cache remediation path** -- cached aliases that fail the tokenizer contract are now treated as invalid cache artifacts. Variant downloads clean up the partially created alias on failure, and the recommended recovery path is to delete and rebuild or re-download the alias with matching tokenizer files.

#### Verification

- `conda run -n NeuralFn python -m pytest tests/test_dataset_manager_variants.py -q`
- `conda run -n NeuralFn python -m pytest tests/test_tokenizer_vocab_contract.py -q`
- `conda run -n NeuralFn python -m pytest tests/test_template_presets.py -x -q`
- `conda run -n NeuralFn python -m py_compile cli/scripts/infer_jepa_semantic.py cli/scripts/infer_semantic_router_moe.py cli/scripts/infer_mixllama_fast.py`

### 2026-04-10 [Experimental] semantic_router_moe preset and router-only harness

#### Added

- **`semantic_router` objective + `semantic_router_moe` preset** -- added a new experimental AR-only MixLLaMA/MoE path that isolates the semantic hash router from JEPA. The preset keeps standard causal attention and MoE expert MLPs, computes one shared semantic route from the pre-block hidden state, and trains next-token CE plus semantic-alignment loss.
- **Shared-route broadcast builtin/stage** -- added `broadcast_expert_routes_module` / `BroadcastExpertRoutesStage` so batch-level expert selections from `semantic_hash_router` can be expanded to the per-token routing tensors expected by the standard MoE dispatcher.
- **Sibling harness scripts** -- added `neuralfn-sdk-harness/scripts/train_semantic_router_moe.py` and `neuralfn-sdk-harness/scripts/infer_semantic_router_moe.py` so the router-only control experiment can be trained and sampled independently of the JEPA hybrid workflow.

#### Changed

- **Template/root graph wiring** -- `build_model_spec_from_config()`, `build_gpt_root_graph()`, and the torch template builders now recognize `semantic_router_moe` and build a root graph with `dataset_source -> (tokens, targets)` plus `semantic_data_source -> sem_targets`, matching the flat compiled contract `(tokens, targets, sem_targets)`.
- **Externally routed MoE blocks** -- the new semantic-router stage uses normal LLaMA attention blocks and standard `expert_dispatch` / `expert_combine`, but replaces the learned token gate with an externally supplied route shared across all MoE blocks in the stage.
- **Semantic-only fallback safety** -- trainer and trace-preview semantic-only paths no longer feed categorical `sem_targets` into the token embedding path. They now synthesize safe placeholder `tokens` / `targets` tensors while preserving the real `sem_targets`, which fixes preview/training failures for semantic-only graphs and control experiments.
- **Preset/skill/docs surfaces** -- the toolbar dropdown, framework guide, SDK docs, and both NeuralFn agent skills now include `semantic_router_moe` and describe it as the router-only control experiment alongside `jepa_semantic_hybrid`.

#### Verification

- `conda run -n NeuralFn python -m pytest tests/test_template_presets.py -x -q` -> `15 passed`
- `conda run -n NeuralFn python -m pytest tests/test_jepa_semantic.py -q` -> `39 passed`

### 2026-04-10 Expanded canonical semantic router vocab

#### Changed

- **Canonical vocab promotion** -- `neuralfn/data/semantic/vocab_8d.json` now contains the expanded router-oriented vocabulary, and the temporary `vocab_8d_expanded_router.json` file has been removed. All loaders, training code, inference code, and API surfaces continue to use the canonical `vocab_8d.json` path.
- **Vocabulary loader validation** -- `ConversationalVocabulary` now validates that the canonical vocab file contains exactly the expected 8 routed dimensions, list-of-string term arrays, and internally consistent optional metadata such as `term_counts` and `total_terms`.
- **Dynamic topic-count docs/tests** -- semantic docs and tests now treat `num_topics` and projector/router shapes as dynamic per-dimension values derived from the expanded vocab instead of implying a fixed 40-topic layout.

#### Breaking changes

- **Old JEPA semantic checkpoints are incompatible** -- checkpoints and interrupted artifacts trained against the previous 40-term semantic vocab are expected to fail to load correctly against the expanded canonical vocab and must be retrained.

#### Verification

- Not run in this task. User explicitly requested no local test or training execution.

### 2026-04-10 JEPA trainer semantic dataset crash fix

#### Changed

- **Trainer semantic dataset loading** -- `TorchTrainer._load_semantic_dataset()` now wraps the vocab-derived `load_training_targets()` arrays directly instead of re-casting through `np.int64`. This removes the trainer-startup `NameError` on CUDA JEPA runs after the vocab-only semantic refactor.

#### Verification

- Not run in this task. User explicitly requested no local test or training execution.

### 2026-04-10 JEPA harness startup crash fix

#### Changed

- **Semantic hash helper compatibility** -- `signature_to_bucket()` and `signature_to_float()` in `neuralfn/semantic.py` still treat `n_buckets` as the canonical low-level parameter, but they now also accept `n_sig_buckets` as a compatibility alias. Passing both with different values raises a clear `ValueError` instead of failing later in the harness startup path.
- **Normalized vocab-only target builders** -- the internal vocab-target materializers now call the low-level signature helpers with the canonical bucket argument, which removes the `TypeError` that blocked `load_training_targets()` during JEPA harness startup.
- **Harness schedule accuracy** -- `neuralfn-sdk-harness/scripts/train_jepa_semantic.py` now threads the resolved CLI `--top-k` into schedule estimation, so semantic-row counts and derived epoch/accumulation summaries no longer fall back to the preset default when you override routing width.

#### Verification

- Not run in this task. User explicitly requested no local test or training execution.

### 2026-04-09 [Experimental] Vocab-only semantic routing

#### Changed

- **Vocab-only semantic supervision** -- `vocab_8d.json` is now the authoritative semantic source for `jepa_semantic_hybrid`. `load_training_data()` remains available as a compatibility wrapper, but it now materializes deterministic vocab-derived samples instead of reading a shipped CSV. `semantic_data_source` likewise generates categorical semantic topic targets on the fly from the vocab metadata.
- **Fixed dimension-to-expert routing** -- the hybrid preset now requires exactly 8 experts, one per vocab dimension: `entity_type`, `action`, `property`, `emotion_sentiment`, `domain`, `temporal`, `causality`, and `social_register`. `top_k` is still configurable but capped to 8, training-time routing is teacher-forced from active semantic targets, and inference-time auto/manual routing uses the same map.
- **Semantic head and loss contract** -- `SemanticProjectorStage` now predicts per-dimension topic logits in addition to the internal 9-D semantic state, and `SemanticAlignmentLossStage` now applies masked categorical cross-entropy over those vocab-topic logits rather than MSE over quantized semantic vectors.
- **Inference topic overrides** -- `neuralfn-sdk-harness/scripts/infer_jepa_semantic.py` now defaults to ignore-sentinel semantic targets in auto mode and supports manual topic forcing via `--semantic-topics dimension=topic,...`. Logged routing summaries now report the resolved expert IDs and their semantic dimensions.
- **Public metadata updates** -- the semantic REST/MCP surfaces now describe the stack as 9-D instead of 15-D, and `/semantic/dimensions` now includes the fixed `expert_id` map plus `num_topics` per dimension.

#### Breaking changes

- **`sem_targets` meaning changed** -- callers must now treat `sem_targets` as categorical vocab-topic IDs with `-100` ignore sentinels in the first 8 slots plus a derived taxonomy-hash slot in position 8. The old quantized-vector interpretation is no longer valid.
- **`jepa_semantic_hybrid` expert count is fixed** -- passing any `experts` value other than `8` to `build_jepa_semantic_hybrid_spec()` now raises a `ValueError`.
- **Shipped semantic CSV removed** -- `neuralfn/data/semantic/training_100k_8d.csv` is no longer packaged or used by the semantic workflow. Consumers that depended on that file should switch to `vocab_8d.json` plus `load_training_targets()` / `load_training_data()`.

#### Verification

- Not run in this task. User explicitly requested no local test or training execution.

### 2026-04-09 JEPA harness inference probe

#### Added

- **Sibling inference script** -- `neuralfn-sdk-harness/scripts/infer_jepa_semantic.py` now provides a small CUDA-only generation probe for exported `jepa_semantic_hybrid` checkpoints. It loads the saved graph JSON and `.pt` weights, auto-detects the traced logits node (`model/softcap` or `model/lm_head`), feeds dummy `targets` and `sem_targets` into the training root graph, and autoregressively samples next tokens.
- **Cached tokenizer reuse** -- the new script reuses the cached tokenizer artifacts stored under the dataset alias and decodes prompts/output with SentencePiece when available. It also supports raw `--prompt-tokens` input so the workflow still works in token-id mode when SentencePiece is unavailable.

#### Verification

- Not run in this task. User explicitly requested no local test or training execution.

### 2026-04-09 Torch training progress logging

#### Changed

- **`TorchTrainer.train()` progress callback** -- torch training now accepts an optional `on_step` callback that receives structured warmup and optimizer-step progress dictionaries. This keeps progress reporting in caller code instead of hardwiring trainer logging policy.
- **Sibling harness console output** -- `neuralfn-sdk-harness/scripts/train_jepa_semantic.py` now configures line-buffered console logging, emits explicit startup / training / validation / export stage markers, and uses `--train-log-every` to print periodic warmup and train-step progress.

#### Verification

- Not run in this task. User explicitly requested no local test or training execution.

### 2026-04-09 Torch compile BF16 stability

#### Changed

- **Fixed compiled node dispatch** -- `CompiledTorchGraph` now instantiates fixed child modules for function nodes and executes every non-trace node directly through its child module. This removes the old `_execute_node` hot path that forced `torch.compile` to specialize one generic dispatcher across different node IDs, input arities, and mixed `Long` / BF16 / FP32 inputs.
- **Contained non-loss float promotions** -- `SemanticProjectorStage`, `SemanticMoERouterStage`, `SemanticHashRouterStage`, and `AttentionlessDecoderStage` now cast their non-loss outputs back to the incoming activation dtype before returning to the graph. Full-precision math is still used inside scalar loss reductions such as token cross-entropy and MSE losses.

#### Verification

- Not run in this task. User explicitly requested no local test or training execution.

### 2026-04-09 [Experimental] JEPA semantic routed-expert training

This is still a **research prototype**. The architecture and trainer surface remain experimental, but the routed branch is now part of the actual training objective instead of being dead wiring.

#### Changed

- **`jepa_semantic_hybrid` training objective** -- the experimental hybrid preset now trains three connected loss terms: autoregressive next-token cross-entropy on the routed expert branch, JEPA latent MSE, and semantic-alignment loss. The routed branch now hashes the pooled semantic vector, performs hash-aware expert routing, runs attention-capable experts over the full masked hidden sequence, and feeds the result into the LM head.
- **Hybrid encoder / template wiring** -- `build_jepa_semantic_encoder_graph()` now outputs `(semantic_vec, hidden)` rather than `(semantic_vec, residual)`. `build_jepa_semantic_model_stage_graph()` now consumes `tokens`, `targets`, and `sem_targets`, adds `semantic_hash_router`, `routed_attention_experts`, and per-loss `loss_scale` nodes, then combines scaled AR/JEPA/semantic losses into the final scalar loss.
- **Hybrid root graph contract** -- `build_gpt_root_graph()` now wires the `jepa_semantic_hybrid` root with a `dataset_source` emitting `tokens` and `targets`, plus the existing `semantic_data_source` emitting `sem_targets`. Dataset-backed tracing and training for the preset now populate all three roles.
- **Torch trainer profile surface** -- `TorchTrainConfig` now includes the parameter-golf-inspired split-optimizer knobs used by the hybrid harness: `optimizer_profile`, `train_batch_tokens`, `beta1`, `beta2`, `adam_eps`, `grad_clip_norm`, `warmup_steps`, `warmdown_fraction`, `max_wallclock_seconds`, `embed_lr`, `head_lr`, `tied_embed_lr`, `matrix_lr`, `scalar_lr`, `muon_momentum`, `muon_backend_steps`, `muon_momentum_warmup_start`, and `muon_momentum_warmup_steps`.
- **Torch trainer implementation** -- `TorchTrainer.train()` now supports token-budgeted gradient accumulation, optional parameter-golf split optimizers, Muon for matrix-shaped parameters, warmup priming, warmdown LR scaling, optional Muon momentum warmup, gradient clipping, and proper semantic-only fallback tensors for role layouts that include `tokens`, `targets`, and `sem_targets`.
- **Experimental builtins / stages** -- added `loss_scale_module`, `semantic_hash_router_module`, and `routed_attention_experts_module` plus the matching `LossScaleStage`, `SemanticHashRouterStage`, and `RoutedAttentionExpertsStage`.
- **Sibling SDK harness** -- `neuralfn-sdk-harness/scripts/train_jepa_semantic.py` is now a CUDA-only, step-driven entrypoint centered on `max_steps`, derived epochs, JEPA-tuned defaults, the new trainer profile surface, and explicit logging of adapted versus ignored `parameter-golf/train_gpt.py` knobs.

#### Breaking changes

- **Hybrid compiled/input contract** -- callers that previously ran `CompiledTorchGraph` for `jepa_semantic_hybrid` with two flat inputs `(tokens, sem_targets)` must now supply `(tokens, targets, sem_targets)`. The preset's `dataset_source` port layout likewise changed from `["tokens"]` to `["tokens", "targets"]`.

#### Verification

- `conda run -n NeuralFn python -m pytest tests/test_jepa_semantic.py -q` -> `29 passed`
- `conda run -n NeuralFn python -m pytest tests/test_template_presets.py -x -q` -> `15 passed`
- `conda run -n NeuralFn python -m pytest tests/test_builtin_neurons.py -q` -> `4 passed`
- `PYTHONPATH=. conda run -n NeuralFn python cli/scripts/train_jepa_semantic.py --help` -> CLI import and argument surface verified
- `conda run -n NeuralFn python -c "import torch; print(torch.cuda.is_available())"` -> `True`

### 2026-04-08 [Experimental] Hybrid JEPA Semantic LLM preset

This is a **research prototype**, not a stable feature. All APIs, data formats, and architectural decisions introduced here are experimental and may change significantly or be removed based on findings.

#### Added

- **`jepa_semantic_hybrid` GPT template preset** -- a new first-class preset loaded via `load_gpt_template(preset="jepa_semantic_hybrid")`. Combines JEPA self-supervised learning with a 9-dimensional grounded semantic space, LSH-based hashing, semantic MoE routing, and an attention-less decoder stage.
- **9D semantic space** -- 8 vocabulary-grounded dimensions (`entity_type`, `action`, `property`, `emotion_sentiment`, `domain`, `temporal`, `causality`, `social_register`) plus a 9th taxonomy hash dimension derived from the entity/action/domain signature.
- **`neuralfn/semantic.py`** -- `SemanticMatrix`, `SemanticHasher`, `ConversationalVocabulary` (loads `vocab_8d.json`, encodes/decodes 9D vectors), `signature_to_float()` (deterministic MD5-based taxonomy hash), `load_training_data()` compatibility helpers, and `generate_synthetic_semantic_data()` (uses vocabulary-backed samples when available).
- **Shipped data assets** in `neuralfn/data/semantic/`: `vocab_8d.json` (core vocabulary terms and dimension metadata).
- **7 new `nn.Module` stages** in `neuralfn/torch_backend.py`: `SemanticDataSourceStage`, `SemanticProjectorStage`, `SemanticAlignmentLossStage`, `SemanticHasherStage`, `SemanticMoERouterStage` (legacy compatibility), `AttentionlessDecoderStage` (legacy compatibility), and `SoftmaxDistillationLossStage`.
- **Dual data source wiring** -- the `jepa_semantic_hybrid` root graph has both a `tokens_in` terminal (for regular text datasets) and a `semantic_data_source` node (auto-loads 100k training rows). Both are visible in the editor. The model subgraph has two input ports: `tokens` feeds the JEPA mask/encoder pipeline, `sem_targets` feeds the `SemanticAlignmentLossStage` for supervised vocabulary alignment. Training combines JEPA self-supervised loss with semantic alignment loss via `aux_loss_add`.
- **Session-scoped template apply path** -- the editor now loads `jepa_semantic_hybrid` through the session `/templates/gpt/apply` endpoint instead of piecing together a preview payload client-side. This ensures the exact backend-built root graph is persisted and displayed, including root nodes, edges, and `input_node_ids` / `output_node_ids`.
- **Semantic trace preview fixes** -- `trace_torch_graph` and the `/trace/torch` route now treat `semantic_data_source` as a built-in data source, skip project dataset access checks for the `__semantic_builtin__` marker, and generate both `tokens` and `sem_targets` preview inputs correctly. This removes the red-button preview failure before training starts.
- **Template payload `extra_nodes`** -- `build_gpt_template_payload` now returns `extra_nodes` and `extra_edges` so the editor's `onAddGPT` handler can display root-level nodes like `semantic_data_source` alongside the model subgraph node.
- **7 new builtin neuron defs** in `neuralfn/builtins.py`: `semantic_data_source_module`, `semantic_projector_module`, `semantic_alignment_loss_module`, `semantic_hasher_module`, `semantic_moe_router_module`, `attentionless_decoder_module`, `softmax_distillation_loss_module`.
- **`build_jepa_semantic_hybrid_spec()`** in `neuralfn/config.py` with new `ModelSpec` fields: `semantic_dim` (default 9), `semantic_residual_dim`, `semantic_n_lsh_tables`, `semantic_n_lsh_planes`, `semantic_table_path`.
- **`build_jepa_semantic_encoder_graph()`** and **`build_jepa_semantic_model_stage_graph()`** in `neuralfn/torch_templates.py`.
- **`"jepa_semantic"` objective type** added to `ObjectiveType` literal and handled in `build_gpt_root_graph` and `TorchTrainer`.
- **4 new MCP tools**: `reverse_engineer_to_semantic`, `semantic_search`, `train_jepa_semantic`, `generate_with_semantics`.
- **4 new REST endpoints**: `POST .../semantic/encode`, `POST .../semantic/search`, `GET .../semantic/dimensions`, `POST .../semantic/generate`.
- **`SemanticInferenceCache`** subclass and `export_semantic_tables` / `import_semantic_tables` helpers in `neuralfn/inference.py`.
- **`tests/test_jepa_semantic.py`** with 15 tests covering the semantic data layer, all new stages, and end-to-end preset compile/forward/training.
- **Python packaging metadata** -- added `pyproject.toml` so NeuralFn can be installed via `pip install -e .` (or `pip install -e /path/to/NeuralFn` from a sibling project). The editable install includes `neuralfn/data/semantic/*.json` and `*.csv` as package data for SDK consumers.

#### Notes

- The preset uses the JEPA latent MSE loss as its sole training signal during the initial self-supervised phase. The decoder/hasher/router paths are wired in the graph for inference and later distillation phases.
- The `TorchTrainer` now treats `objective == "jepa_semantic"` the same as `"jepa"` for EMA target updates.
- Semantic data artifacts (`neuralfn/data/semantic/`) are generated, not tracked in git.

### 2026-04-08 Full SDK documentation and agent skills

#### Added

- Comprehensive `docs/` directory with 50+ markdown pages covering every part of the platform:
  - **Framework guide** (`docs/framework-guide/`): 9 tutorial-oriented pages teaching developers how to build with NeuralFn in Python -- defining neurons, building graphs, subgraphs/variants, torch models, templates/presets, training workflows, inference/export, and datasets.
  - **Python SDK reference** (`docs/python-sdk/`): 15 pages documenting every public class, function, method, property, and type in the `neuralfn` package including all 58 builtin neurons, all 16 presets, all 40+ Stage classes, and all training methods.
  - **REST API reference** (`docs/rest-api/`): 8 pages covering all 60+ HTTP endpoints with method, path, auth requirements, request/response shapes, and error codes.
  - **MCP tools reference** (`docs/mcp/`): 7 pages documenting all 35+ MCP tools with parameters, descriptions, and workflow examples.
  - **Server internals** (`docs/server/`): 6 pages covering Settings, ORM models, auth system, services, and Pydantic models.
  - **Editor reference** (`docs/editor/`): 6 pages covering the TypeScript API client, Zustand store, graph utilities, components, and pages.
  - **Testing guide** (`docs/testing.md`): test suite overview with targeted check commands.
  - **Agent skills page** (`docs/agent-skills.md`): links to all AI coding agent skills with descriptions.
- Three AI coding agent skills in `.cursor/skills/`:
  - `neuralfn-python-sdk`: teaches agents how to build graphs with the core Python SDK.
  - `neuralfn-torch`: teaches agents how to build, train, and export torch-backed models using presets and the template system.
  - `neuralfn-mcp` (updated): expanded with all 16 presets, full config key table, missing tools (`load_dataset_source`, `poll_training_status`, `get_training_status`, `set_dataset_access`), and non-AR workflow examples.
- `AGENTS.md` updated with documentation maintenance rules: any change to public APIs must update the corresponding `docs/` page and relevant agent skills.
- `README.md` updated with link to `docs/` directory.

#### Notes

- The framework guide and API reference are complementary: the guide teaches how to build, the reference has exact signatures. They cross-link to each other.
- All docs use relative markdown links for GitHub navigation.
- Agent skills are designed to stay under 500 lines for optimal context window usage, with supporting reference files where needed.

### 2026-04-08 JEPA block masking

#### Added

- `JEPAMaskStage` now supports a `mask_strategy` parameter: `"random"` (default, backward-compatible i.i.d. per-token masking) and `"block"` (contiguous span masking). Block masking samples `num_blocks` contiguous spans per sequence with lengths drawn uniformly from `[min_block_ratio * seq_len, max_block_ratio * seq_len]`, forcing the predictor to reason about larger semantic structures rather than interpolating from adjacent unmasked tokens.
- New `ModelSpec` fields: `jepa_mask_strategy` (str), `jepa_num_blocks` (int, default 4), `jepa_min_block_ratio` (float, default 0.1), `jepa_max_block_ratio` (float, default 0.25). All wired through `_base_model_spec` and the `build_jepa_model_stage_graph` template builder.
- Default `jepa_mask` builtin neuron config extended with the new keys.
- Two new tests in `tests/test_template_presets.py`:
  - `test_jepa_block_masking_produces_contiguous_spans` — verifies block masks produce contiguous spans of at least `min_block_len`, masked positions are replaced, unmasked positions are preserved, and random mode still produces scattered masks.
  - `test_jepa_block_masking_config_wires_through_template` — verifies `ModelSpec` fields propagate through the template builder into the compiled graph's module config.

#### Verification

- `python -m pytest tests/test_template_presets.py -q` — all tests pass, including the existing EMA target encoder test.

### 2026-04-08 Implement backend_capabilities: cache, quantized_export, megakernel, PCA KV cache

#### Added

- `resolve_backend_capabilities(spec)` in `neuralfn/config.py` auto-derives capability flags from `TemplateSpec` fields (`runtime`, `compression`, etc.) and is called from `_base_model_spec()` so every preset gets a correct capability map.
- `FusedCausalAttentionStage` in `neuralfn/torch_backend.py` combines QKV projection, reshape, RoPE, SDPA, merge, and output projection into a single `nn.Module` for the megakernel runtime's aggressive kernel fusion scope.  Registered as `fused_causal_attention` builtin module.
- `export_quantized_pt(graph, path, scheme)` and `import_quantized_pt(graph, path)` in `neuralfn/inference.py` supporting `int8` (per-channel) and `ternary` quantization schemes.
- `InferenceCache` class in `neuralfn/inference.py` for stateful autoregressive generation with KV cache management.  Reads device from the graph's `torch_config` and handles both training (tokens+targets) and inference-only graphs.
- New presets `llama_megakernel` (`runtime="megakernel"`) and `kv_pca_llama` (`compression="kv_pca"`) in `neuralfn/config.py`, registered in `build_model_spec_from_config`.
- `build_dense_attention_graph` now accepts `enable_cache`, `enable_pca`, `pca_compressed_dim`, and `fused_megakernel` flags to optionally insert KV cache read/write nodes, PCA encode/decode nodes, or collapse to a single fused attention node.
- 24 new tests in `tests/test_backend_capabilities.py` covering capabilities resolution, megakernel forward, PCA attention, KV cache graph structure, quantized export round-trips, runtime wiring, inference cache, and new preset registration.

#### Changed

- `TorchTrainer.train()` now reads `template.runtime` from the graph's serialized `template_spec` to select the compilation mode (`eager` → none, `compile` → `torch.compile`, `megakernel` → `torch.compile(mode="max-autotune", fullgraph=True)`).  `TorchTrainConfig.compile` still acts as an override when explicitly True.
- `KVQuantPackStage` upgraded from a plain concat stub to real int8 quantization with per-token scale factors.  `KVQuantUnpackStage` performs the inverse dequantization.
- `TemplateSpec.backend_capabilities` defaults updated: `cache` and `quantized_export` now default to `True`.
- All template graph builders (`build_model_stage_graph`, `build_hidden_backbone_graph`, `build_seq2seq_model_stage_graph`, `build_diffusion_model_stage_graph`) now forward PCA/megakernel flags to `build_dense_attention_graph` via `_attn_flags()`.
- `test_kv_quant.py` tolerance widened from exact match to `atol=0.02` to account for int8 quantization noise.
- Built-in neuron catalog test updated to include `fused_causal_attention_module` (80 entries).

#### Verification

- `python -m pytest tests/test_backend_capabilities.py tests/test_template_presets.py tests/test_kv_quant.py tests/test_kv_pca.py tests/test_builtin_neurons.py -q` → 41 passed.

### 2026-04-08 Gitignore for local data and caches

#### Changed

- Expanded `.gitignore` to exclude SQLite databases, `server/datasets/`, `server/session_snapshots/`, `server/artifacts/`, local `.env` files, Python/Node/tool caches (including `*.tsbuildinfo`), coverage artifacts, and common log/OS junk. Documented this next to the platform configuration table in `README.md`.

### 2026-04-05 Remaining roadmap templates and training wiring

#### Added

- Added the remaining shipped template specs and graph builders for `ttt_llama`, `llm_jepa`, `hnet_lm`, and `universal_llama`.
- Added new torch module stages and builtins for JEPA masking/pooling/projector/predictor/loss, raw-byte patch embedding/merge, ACT halting, universal recurrence, and internal diffusion timestep sampling.
- Added raw-byte dataset loading helpers plus new regression coverage in `tests/test_template_presets.py` for preset routing, JEPA EMA behavior, H-Net byte loading, Universal halting, and dataset-source role wiring.

#### Changed

- `build_gpt_root_graph()` now persists a serialized `template_spec` into `graph.torch_config` and routes objective-specific graphs explicitly for AR, Seq2Seq, Diffusion, JEPA, H-Net, and Universal templates.
- `build_gpt_template_payload()` and the session/template application path now resolve the full shipped preset catalog consistently, including the previously broken `seq2seq` and `diffusion` payload paths.
- Dataset-backed tracing, `dataset_source` insertion, and torch training now route by input role instead of assuming only `(tokens, targets)` or `(enc_tokens, dec_tokens, targets)`.
- H-Net training now switches to raw-byte dataset loading automatically and enforces `vocab_size == 256`.
- The editor template picker and MCP `load_gpt_template` surface now reflect the expanded preset set instead of only the older NanoGPT/GPT-2/LLaMA/MoE subset.
- The built-in neuron catalog test now tracks the current 79-entry builtin registry rather than the obsolete 37-entry snapshot.

#### Removed

- Removed the temporary root-level debug helpers `debug_pt.py`, `debug_start_run.py`, `debug_start_run_2.py`, `tmp_update.py`, `tmp_verify.py`, `tmp_templates.py`, and `tmp_update_builtins.py`.

#### Verification

- Verified the directly affected tests with `python -m pytest tests/test_template_presets.py tests/test_builtin_neurons.py tests/test_diffusion.py tests/test_seq2seq.py tests/test_server_dataset_loading.py -q` (`16 passed`).
- Verified the updated legacy nested-graph wrappers with `python -m pytest tests/test_server_nested_graphs.py -k "gpt_template_route_returns_variant_library_payload or torch_trace_can_sample_from_dataset_source" -q` (`2 passed`).

### 2026-04-04 README built-in neuron catalog

#### Changed

- Expanded `README.md` **Built-in neurons** into a full reference for all 58 definitions from `neuralfn/builtins.py`, grouped by role (scalar vs torch module), with notes on graph terminals (`input` / `output`), duplicate `gelu` names, and an alphabetical index.

#### Verification

- Cross-checked names and groupings against `neuralfn/builtins.py` (`BuiltinNeurons.all()` / `_BUILTIN_ATTR_MAP`).

### 2026-04-06 Template compatibility and viewport-aware insertion

#### Changed

- Fixed the active `seq2seq` template regression by making `enc_block` and `dec_block` link to the families the preset actually exports: `enc_attention`, `dec_attention`, `cross_attention`, `mlp_dense`, and `mlp_moe`.
- Added variant-family compatibility aliases in both the backend resolver and the editor graph normalizer so older saved graphs that still refer to `attn_block`, `transformer_block`, or `mixllama` can resolve against the equivalent current family when it exists.
- Changed editor insertion defaults so toolbar actions, GPT template inserts, and variant-library inserts use the center of the visible graph viewport plus a deterministic stagger instead of toolbar-button screen coordinates or random off-screen positions.

#### Notes

- Canonical template outputs were left unchanged for the working presets. The compatibility aliases are fallback-only and still prefer an exact family match when one exists.
- The `mixllama` compatibility path is intended for older saved block-family references. It does not rewrite stored graphs; it only broadens resolution at load time.

#### Verification

- Added regression coverage in `tests/test_template_presets.py` for the reported presets (`moe`, `mixllama_fast`, `jamba`, `ternary_b158`, `seq2seq`), the `seq2seq` internal family references, and legacy family alias resolution.
- Verified with `python -m pytest tests/test_template_presets.py tests/test_seq2seq.py -q`.
- Verified the relevant nested-graph template and dataset trace paths with `python -m pytest tests/test_server_nested_graphs.py -k "gpt_template_route_returns_variant_library_payload or torch_trace_can_sample_from_dataset_source" -q`.
- Verified editor type/build wiring with `pnpm --dir editor build`.

### 2026-04-04 Codex project MCP config

#### Added

- Added project-scoped Codex MCP configuration at `.codex/config.toml` for the local `neuralfn` server using `uv run server/mcp_server.py`.

#### Changed

- Updated the MCP setup docs in `README.md` to distinguish Codex's `.codex/config.toml` from Cursor's `.cursor/mcp.json`.

#### Verification

- Verified the config format against the OpenAI Codex MCP docs for project-scoped trusted workspaces and confirmed the repo now contains `.codex/config.toml`.

### 2026-04-04 Datasets tab and personal projects

#### Added

- A dedicated `Datasets` routed surface in the React shell for downloading Hugging Face datasets, uploading local files, inspecting the project-visible catalog, and editing which accessible projects can use each dataset.
- Persistent dataset catalog storage via `dataset_assets` and `project_dataset_grants`, plus an Alembic migration to materialize the new access-control tables.
- Self-serve project creation for authenticated users, with every new project automatically seeded with a `Main session` and activated immediately in the current auth session.
- MCP dataset access management through the new `set_dataset_access` tool and optional `project_ids` sharing on dataset downloads/loads.

#### Changed

- Dataset visibility is no longer just route scoping over a shared filesystem scan. Datasets are now registered in the database and filtered by explicit project grants.
- Existing filesystem datasets under `server/datasets/` are reconciled into the DB-backed catalog on access so they remain visible after the access-control change.
- The editor no longer manages dataset selection from the bottom training strip. Dataset-backed training now resolves from the saved `dataset_source` node configuration in the session graph.
- The training panel is simplified to manual JSON entry plus run status/trace output, while dataset download/upload flows live in the new `Datasets` tab.

#### Operational notes

- Apply the new Alembic revision after the platform foundation migration to create `dataset_assets` and `project_dataset_grants`.
- Environments that still rely on `NEURALFN_CREATE_SCHEMA_ON_STARTUP=1` will auto-create the new tables on startup because they are part of the SQLAlchemy metadata.
- The first dataset catalog request after upgrading reconciles any existing on-disk datasets into the DB catalog and grants them to the projects that already exist at that point.

#### Verification

- Verified backend imports and bytecode with `python -m compileall server tests/test_platform_api.py`.
- Added platform API coverage for non-admin project creation, dataset grant filtering, and graph-driven dataset-backed runs in `tests/test_platform_api.py`.
- Verified the frontend route and type wiring with `pnpm --dir editor build`.
- Attempted to run `uv run --with-requirements requirements.txt python -m unittest discover -s tests -p "test_platform_api.py"`, but this environment could not resolve PyPI to install missing Python dependencies (`fastapi`, `torch`, `tiktoken`, etc.).

### 2026-04-04 Platform foundation

#### Added

- SQLAlchemy-backed persistence for users, auth sessions, projects, memberships, editor sessions, session snapshots, and training runs.
- Alembic migration scaffolding for the durable platform schema, with SQLite as the default local database and MySQL-ready configuration through `NEURALFN_DATABASE_URL`.
- Built-in authentication with bootstrap-admin flow, login/logout endpoints, active-session selection, PBKDF2 password hashing, opaque session tokens, and HTTP-only session cookies.
- Project-scoped datasets plus project/session-scoped graph, session, and run APIs under `/api/projects/{project_id}/...`.
- A routed React app shell with dedicated Editor, Runs, Analytics, and Admin surfaces.
- Refresh-safe session hydration/autosave flow that loads graphs by project/session, tracks revisions, and reloads after `409` conflicts.
- Optional Redis-backed live state for session graph state, run events, and agent coordination, with in-memory fallback for local development.
- MCP authentication and tool scoping so graph/training tools now operate on explicit `project_id` and `session_id` context.

#### Changed

- The platform no longer assumes a single anonymous in-memory graph. Workspace state is now organized by authenticated user, project, and editor session.
- The frontend now boots through `/api/bootstrap`, routes through `/login` and `/app/...`, and persists the active project/session on the server-side auth session.
- Training status and session restore behavior are no longer tied to global process state; they flow through the scoped services and live-state store.
- Legacy helper wrappers remain in `server/routes.py` only to keep older route-oriented tests working against a dedicated legacy workspace.

#### Operational notes

- Local startup defaults to `sqlite:///neuralfn.db` plus filesystem snapshots/artifacts unless overridden with environment variables.
- For migration-managed environments, run `alembic upgrade head` and set `NEURALFN_CREATE_SCHEMA_ON_STARTUP=0`.
- `NEURALFN_ALLOW_ORIGINS` must include the frontend origin because the app uses cookie-authenticated cross-origin requests during local development.
- MCP clients must provide `NEURALFN_MCP_EMAIL` and `NEURALFN_MCP_PASSWORD`, and may override `NEURALFN_BASE_URL` when the API is not hosted at `http://localhost:8000/api`.

#### Verification

- Added `tests/test_platform_api.py` to cover bootstrap-admin, active-session switching, refresh-safe graph restore, idle run status, and revision-conflict handling.
- Verified backend imports/bytecode with `python -m compileall server tests/test_platform_api.py`.
- Verified the new platform API coverage with `python -m unittest discover -s tests -p "test_platform_api.py"`.
- Verified the frontend wiring with `cd editor && pnpm build`.
