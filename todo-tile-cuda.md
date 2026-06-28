# NeuralFn CUDA Tile C++ backend TODO

Goal: implement a CUDA Tile C++ kernel backend for every NeuralFn compute surface that can participate in LLM or neural-network training. This is broader than the current LLM templates: coverage is defined by the public builtin catalog in `neuralfn/builtins.py`, the module dispatch in `neuralfn/torch_backend.py::build_module`, scalar function dispatch in `build_function_module`, and optimizer/runtime math used by `TorchTrainer`.

This TODO is the authoritative checklist for the CUDA Tile backend. Keep `todo-kernels.md` as the older PyTorch-reference / kittens wishlist, but update this file for CUDA Tile implementation status.

References:

- NVIDIA blog: https://developer.nvidia.com/blog/develop-high-performance-gpu-kernels-in-cpp-with-nvidia-cuda-tile/
- CUDA Tile guide: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html

## Hard requirements

- [x] Require CUDA Toolkit 13.3 or newer for the CUDA Tile C++ build path.
- [x] Require `cuda_tile.h`, C++20, `nvcc --enable-tile`, and architecture flags for the active GPU.
- [x] Require compute capability 8.x or newer for Tile kernels; prefer SM120 when available.
- [x] Launch Tile kernels with one logical thread per tile block: `kernel<<<grid, 1>>>(...)`.
- [x] Use `ct::tensor_span`, `ct::partition_view`, `ct::shape`, and masked load/store variants for tails.
- [x] Use `__restrict__` and `ct::assume_aligned(..., 16_ic)` for pointer-heavy kernels.
- [x] Keep PyTorch as the default fallback when CUDA Tile is missing or unsupported.
- [x] Add strict mode that fails if any selected training graph node lacks CUDA Tile coverage.
- [x] Never fake coverage: every unsupported node must have an explicit host-only, delegated, or fallback reason.

## Training hot-path rule

Real training tensors must not pass through graph editor node objects.

- [x] Add a zero-Python SM120 dense GPT training helper. `tools/train_gpt_sm120.sh`
  mirrors `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh` at the process
  boundary by execing `nfn_gpt_native_train` directly instead of starting from
  Python wrappers, preferring llm.kittens TinyStories token bins when they are
  present, and preserving the same 20,000-step, 64x1024, 524,288-token, AdamW,
  validation/sample/checkpoint defaults. This does not close the measured
  kernel-throughput parity gap, but it removes remaining wrapper startup
  overhead from the workstation path used for apples-to-apples SM120 runs.
- [x] Add a compiled SM120 dense GPT launcher. `tools/build_train_gpt_sm120_cli.sh`
  builds `build/nfn_train_gpt_sm120`, and `tools/build_native_gpt2_all.sh`
  includes it so workstation GPT runs can use the same defaults without Bash or
  Python startup before execing the native CUDA Tile trainer.
- [x] Default normal native GPT launcher/SDK device routing to CUDA ordinal `0`.
  `NativeTrainRunConfig`, native GPT checkpoint sampling, `train_gpt.py`,
  `nfn train`, the native training guard, `tools/train_gpt_sm120.sh`, and the
  compiled SM120 launcher now avoid the `dedicated` selector unless explicitly
  requested, so the regular workstation path does not spawn `nvidia-smi` before
  launching CUDA Tile training. Benchmark wrappers keep their explicit
  `dedicated` defaults when they need selected-GPU load evidence.
- [x] Add a generic compiled dense GPT launcher. `tools/build_train_gpt_cli.sh`
  builds `build/nfn_train_gpt`, `tools/build_native_gpt2_all.sh` and
  `tools/rebuild_native_sm120.sh` include it, and
  `tools/install_native_gpt2_commands.sh` links `nfn-train-gpt` /
  `nfn-gpt-train`. It keeps the same no-Python/no-Bash native handoff while
  honoring `NFN_NATIVE_GPT_*` GPT/GPT2/GPT3/NanoGPT/template/custom-graph
  selector, shape, validation, sampling, checkpoint, optimizer, train-loss, and
  device defaults before the older SM120 aliases.
- [x] Compile graph topology into a static execution plan before training.
- [x] Make `CompiledTorchGraph.forward()` use the precompiled plan instead of walking `NeuronGraph.nodes` and `NeuronGraph._incoming()` per batch.
- [x] Add regression coverage proving forward still works after graph edge traversal is made unavailable post-compilation.
- [x] Extend the same invariant to future CUDA Tile graph execution plans.
- [x] Add a benchmark that compares old graph-walk execution, static PyTorch execution, and CUDA Tile execution.
- [x] Add an assertion helper for tests: no training forward/backward path may read editor position, viewport, React store, or mutable graph-editor metadata.
- [x] Gate paired SM120 native benchmarks on the same runtime invariant.
  `tools/paired_kernel_speed.py` now extracts `graph_editor_tensor_flow` and
  `torch_required` from candidate native JSON and fails NeuralFn native
  candidates unless both are exactly `false`. This keeps benchmark promotion
  paths from accepting graph-editor tensor flow or Torch fallback while chasing
  kernel throughput parity.
- [x] Gate CUDA 13.3 SM120 validation on the same no-Torch runtime invariant.
  `tools/validate_sm120_cuda13.sh` now runs
  `tools/check_native_no_torch_deps.py --json` before CUDA smoke checks by
  default, so post-toolkit-reinstall validation fails if compiled native
  artifacts, SDK bindings, or fast Python wrappers regain Torch/c10/Python
  runtime links or slow import paths. Set `NFN_SM120_CUDA13_RUN_NO_TORCH=0`
  only for narrow CUDA-only bisections after the no-Torch gate already passed.
  The 2026-06-28 post-reinstall validator run passed the no-Torch gate, Tile
  smoke, NVFP4 pack smoke, transformer-LM step smoke, focused LM-head
  benchmark, and `tests/test_native_gpt2.py` (`109 passed, 1 skipped`). The
  focused LM-head JSON still reports `candidate_path_class:
  diagnostic-cuda-graph-wrapper`, `candidate_true_fused_capability: false`,
  and `true_fused_launch_count: 0`, so this is a clean CUDA 13.3 baseline, not
  strict LM-head completion.
- [x] Add the focused LM-head backward candidate/current microbench to the
  default CUDA 13.3 SM120 validator. `tools/validate_sm120_cuda13.sh` now runs
  `tools/bench_lm_head_backward_candidate.sh` with the `trainer-chunk` profile
  and writes `/tmp/nfn_sm120_cuda13_lm_head_backward.json`, using a real Tile
  ops shared object for the standalone `dlopen` benchmark even when the trainer
  smoke path uses the linked binary. The standalone LM-head benchmark now
  carries warmup graph-route counters separately, so the warmed trainer-chunk
  validation still proves graph capture body routing even when timed iterations
  are graph-cache hits. This keeps CUDA-reinstall validation tied to the
  current LM-head migration blocker without treating the diagnostic CUDA Graph
  wrapper as strict true-fused completion.
- [x] Make canonical llm.kittens parity runs median-based by default. After the
  CUDA 13.3 WSL reinstall, one-sample runs could fail at sub-percent jitter
  even with zero compute processes on the dedicated RTX 5090; the parity
  wrapper now defaults to three measured samples plus one warmup sample so the
  existing `median:` default gate is used unless a quick smoke explicitly sets
  `NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0`.
- [x] Keep editor graph objects as control-plane data only: authoring, serialization, validation, and compile-time planning.
- [x] Guard the Python SDK native-training boundary against accidental Python or
  shell launcher fallback. `NativeTrainRunConfig.strict_native_command` now
  defaults true, and both the SDK config resolver and generic
  `neuralfn._native_train` C++ binding reject `python`, `bash`, `*.py`, and
  `*.sh` commands unless the caller explicitly sets
  `strict_native_command=False` for diagnostics. The no-Torch dependency gate
  exercises this default so native SDK handoff regressions cannot silently route
  through Torch-era scripts.
- [x] Require compiled GPT fast-path artifacts in the no-Torch verifier. The
  default `tools/check_native_no_torch_deps.py` artifact set now fails if
  `build/nfn_gpt_native_train_linked`, `build/nfn_gpt2_native_train`,
  `build/nfn_train_gpt`, `build/nfn_train_gpt_sm120`,
  `build/nfn_native_train`, `neuralfn/_native_gpt.*.so`,
  `neuralfn/_native_gpt2.*.so`, or `neuralfn/_native_train.*.so` is missing or
  stale, alongside the native GPT trainer and Tile ops library. This makes the
  no-Bash/no-Python workstation launch path and SDK C++ binding surface
  required evidence instead of optional convenience artifacts. The same gate now
  also runs budgeted direct native metadata startup probes for
  `build/nfn_gpt_native_train_linked --list-templates`,
  `build/nfn_gpt2_native_train --list-templates`, and
  `build/nfn_native_train --list-models --json`, plus the unified frontend GPT
  catalog actions `build/nfn_native_train --base-model gpt --list-templates`
  and `build/nfn_native_train --base-model gpt
  --native-cuda-list-templates`.
- [x] Enforce optimized native GPT kernel routes by default. Dense GPT runtime
  JSON now reports `optimized_kernel_contract_*` fields and fails normal
  training if the optimized AdamW ABI is missing, attention falls back to
  row/scalar paths, or the linear backend launches TF32/SGEMM fallback.
  `--allow-basic-kernel-fallback` is reserved for diagnostics and rejected
  same-script candidate bisection.

## Current SM120 parity baseline

- [x] Refresh the no-stage llm.kittens parity gate after restoring the fused
  padded token-weight initializer to opt-in. The 2026-06-28 dedicated RTX 5090
  3-step, 3-sample, 1-warmup run measured NeuralFn at `0.997292x` median
  train-loop wall, `0.996885x` median steady-state CUDA-event time, and
  `1.002498x` median tokens/sec versus
  `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh`, with
  `metric_ratio_gates: passed=true`. The runtime contract stayed clean:
  `graph_editor_tensor_flow=false`, `torch_required=false`,
  `optimized_kernel_contract_passed=true`, and `train_loss_host_d2h_count=0`.
  - 2026-06-28 current-default 3-step, 2-sample no-stage parity still passed the
    configured gate after pinning the llm.kittens reference command to the
    system CUDA 13.3 runtime/WSL driver shim: NeuralFn measured `1.001966x`
    llm.kittens train-loop wall, `1.001668x` steady-state CUDA-event step time,
    and `0.998004x` tokens/sec. The same JSON reports
    `diagnostic-cuda-graph-wrapper`, `true_fused_capability=false`, and
    `graph_body_nodes_per_replay=3`, so keep the strict goal open.
- [x] Preserve rejected heavy cuBLASLt plan retunes as named profiles instead of
  implicit defaults. `NFN_SM120_NATIVE_CANDIDATE_PROFILE=cublaslt_heavy_shape_flip`
  now records the CUDA 13.3.33 dedicated RTX 5090 3-step, 2-sample
  shape-stat gate that changed cuBLASLt plan-cache and linear-shape telemetry
  but regressed train-loop wall time (`1.002525x`), steady-state CUDA-event
  time (`1.005491x`), block backward (`1.011881x`), MLP FC backward
  (`1.031422x`), and QKV backward (`1.029108x`). The default planner stays on
  the current per-shape choices.
- [x] Reduce dense GPT native startup parameter-fill launches. The raw Tile ABI
  now exposes `nfn_native_tile_fill_many_values_mixed_float32_bf16_bits`, and
  `nfn_gpt_native_train` uses it to initialize float32 and BF16 constant
  parameter descriptor groups in one mixed launch. The 2026-06-25 default
  TinyStories SM120 startup preflight reported
  `parameter_initialization_strategy=mixed-float32-bf16-fill-many-values`,
  `mixed_parameter_initialization_kernel_launches=1`, and
  `parameter_initialization_kernel_launches_per_startup=1`, with
  `torch_required=false` and `graph_editor_tensor_flow=false`.
- [x] Make the optimized-attention fallback gate unambiguous in native GPT
  JSON. Default runs now report
  `attention_forward_scalar_launch_fallback_available=true`,
  `attention_forward_scalar_launch_fallback_enabled=false`,
  `attention_forward_scalar_launch_allowed=false`, and fail if scalar attention
  launches. The 2026-06-25 one-step TinyStories GPU assertion reported zero
  scalar launches, zero row fallback launches, `torch_required=false`, and
  `graph_editor_tensor_flow=false`.
- [x] Verify and document linked dense GPT startup as the default SDK/CLI
  dispatch target. The linked binary resolves Tile ops through `RTLD_DEFAULT`
  and self-selects `--tile-ops-lib linked`; the 2026-06-25 one-step TinyStories
  probe reduced `setup.load_tile_ops` from about `63.986 ms` on
  `build/nfn_gpt_native_train` to `0.083 ms` on
  `build/nfn_gpt_native_train_linked`, with `torch_required=false` and
  `graph_editor_tensor_flow=false`.
- [x] Elide stage-timing event-pool preallocation on startup-only probes. When
  `NFN_NATIVE_GPT_STAGE_TIMING=1` is set with `--startup-only`, the trainer now
  reports `stage_timing_prealloc_event_pairs_requested: 0` and does not create
  thousands of CUDA events that cannot be consumed before exit. Normal
  stage-timed training still preallocates the event pool for hot-loop timing.
  Short stage-timed training probes now default the pool to
  `4096 * max_steps`, capped by `NFN_NATIVE_GPT_STAGE_TIMING_MAX_EVENTS`,
  instead of always reserving 16,384 event pairs; explicit
  `NFN_NATIVE_GPT_STAGE_TIMING_PREALLOC_EVENTS` overrides still win. The
  2026-06-28 GPU-visible one-step smoke used 3,987 events from a 4,096-pair
  pool with zero dropped events and zero hot-created pairs, reducing
  `setup.stage_timing_event_pool` to `1.561 ms` for that probe.
- [x] Prefer the installed CUDA 13 runtime path in native C++ startup when no
  explicit `--cuda-runtime-lib` / `NFN_CUDA_RUNTIME_LIB` is supplied. The
  resolver now checks `/usr/local/cuda/lib64/libcudart.so.13` and adjacent
  installed Toolkit paths before generic sonames. A 2026-06-26 linked one-step
  TinyStories run selected `/usr/local/cuda/lib64/libcudart.so.13`, reported
  `setup.load_cuda_runtime=0.174629 ms`, `torch_required=false`, and
  `graph_editor_tensor_flow=false`; explicit generic `libcudart.so` comparison
  measured `0.195069 ms`. This removes a small loader-search cost only; large
  startup buckets remain arena materialization and token-weight initialization.
- [x] Align the llm.kittens parity wrapper with the native-vs-native candidate
  wrapper's dedicated-GPU stale-utilization guard. `tools/bench_native_gpt_sm120_parity.sh`
  now defaults `NFN_SM120_PARITY_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE=1`
  and forwards the paired-runner flag that tolerates high NVML utilization only
  when the selected display-disabled GPU has no compute processes; set it to
  `0` for strict stale-utilization rejection.
- [x] Revisit the LM-head backward microbench after the CUDA 13.3.33 WSL
  reinstall. Sandboxed GPU probes still fail with OS-blocked NVML/runtime
  access, but the same command with GPU access sees the dedicated RTX 5090
  (`NVIDIA-SMI 610.43.02`, CUDA UMD 13.3) and the smoke profile passes. The
  2026-06-25 smoke measured the diagnostic CUDA Graph wrapper candidate at
  `1.807552 ms/iter` versus the legacy cooperative sequence at
  `1.840461 ms/iter` (`0.982119x`) on 2,048 rows, but the ABI still reports
  `candidate_true_fused_capability=false` and
  `candidate_symbol_abi_path_class=diagnostic-cuda-graph-wrapper`. This remains
  a graph-replay optimization, not the final true fused Tile kernel.
  - 2026-06-28 reran the production trainer-chunk and strict true-fused
    trainer-chunk probes after the CUDA reinstall. The graph-wrapper candidate
    measured `0.970829x` versus the cooperative baseline at 32,768 rows, but
    still reports `candidate_path_class=diagnostic-cuda-graph-wrapper`. The
    strict true-fused single-kernel diagnostic reports
    `candidate_path_class=strict-true-fused-tile-kernel` but remains rejected at
    `6.739547x` versus the cooperative baseline and `22.092373x` versus the
    reference CE+dHidden+dWeight component sum. The benchmark now keeps
    reference-component timings warm by default via `max(1, warmup)`, while
    `NFN_LM_HEAD_BACKWARD_REFERENCE_COMPONENT_WARMUP` remains available for
    intentional cold-start diagnostics. The post-fix rerun reported
    `reference_component_warmup=1` with `NFN_LM_HEAD_BACKWARD_WARMUP=0`.
- [x] Make paired full-trainer benchmarks report the LM-head true-fused blocker
  explicitly. `tools/paired_kernel_speed.py` now emits
  `native_lm_head_true_fused_target` whenever the candidate profile is still
  `diagnostic-cuda-graph-wrapper` or the strict true-fused capability is false,
  including graph replay/body-node means and the required next ABI
  `nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16` plus
  `nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused()`.
  The paired tool also accepts `--require-native-lm-head-true-fused` so a run
  can fail on this blocker without relying on incidental timing gates.
  `tools/bench_native_gpt_sm120_parity.sh` forwards that gate when strict parity
  enforcement is on, and the native candidate wrapper exposes opt-in
  `NFN_SM120_NATIVE_REQUIRE_LM_HEAD_TRUE_FUSED` /
  `NFN_SM120_CANDIDATE_REQUIRE_LM_HEAD_TRUE_FUSED` aliases for LM-head-specific
  bisections.
  - 2026-06-27 added the rejected-by-default 4x4 strict true-fused LM-head
    diagnostic profile. Tile CUDA now accepts
    `NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=4`, the CE row-thread selector
    accepts `16`, and `trainer-chunk-true-fused-tile4` /
    `lm_head_true_fused_tile4` measure the matching body with strict route
    gates. The CUDA 13.3.33 dedicated RTX 5090 one-step full-loop gate proved
    true-fused launches moved `0 -> 16`, but rejected promotion at `30.645660x`
    train-loop wall time, `0.032631x` tokens/sec, `129.582841x` LM-head
    backward time, and `186.457823x` cooperative LM-head time versus the CUDA
    Graph wrapper. This is benchmark coverage only; the diagnostic single-kernel
    body remains unusable for training-speed parity.
  - 2026-06-28 reran the focused `trainer-chunk-true-fused-tile4` microbench on
    the dedicated RTX 5090 after the CUDA 13.3 reinstall. It proved the strict
    route again (`candidate_path_class=strict-true-fused-tile-kernel`) but
    rejected promotion at `37.738071x` candidate/current-wrapper and
    `113.697403x` candidate/reference-summed time, with the strict body
    `4510.827989 ms` slower than the reference CE+dHidden+dWeight components.
    The tile4 body is now recorded as a dead-end tuning branch unless a future
    implementation proves both current-wrapper and reference parity.
  - 2026-06-28 added a rejected-by-default 24x24 strict true-fused LM-head
    diagnostic profile. Tile CUDA now accepts
    `NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=24`, the CE row-thread selector
    accepts `576`, and `trainer-chunk-true-fused-tile24` /
    `lm_head_true_fused_tile24` build the matching candidate library in `/tmp`
    for focused and full-loop bisection. The CUDA 13.3.33 dedicated RTX 5090
    focused probe proved `candidate_path_class=strict-true-fused-tile-kernel`
    and moved `true_fused_launch_count` to `1`, but rejected promotion at
    `6.266142x` candidate/current-wrapper and `21.764091x`
    candidate/reference-summed time, with the strict body still `679.228962 ms`
    slower than the reference CE+dHidden+dWeight components.
  - 2026-06-26 tightened this evidence path with an explicit strict launch
    counter. The Tile ops ABI now exports
    `nfn_native_tile_lm_head_classifier_true_fused_launch_count()`, focused
    LM-head benchmark JSON reports per-variant `true_fused_launch_count`, native
    GPT full-loop JSON reports `lm_head_classifier_true_fused_launch_count`, and
    the paired speed gate treats zero observed full-loop true-fused launches as
    a failure even when capability/path metadata says strict true fused.
  - 2026-06-26 tightened the same counter again so it increments only after
    CUDA accepts the cooperative `cudaLaunchCooperativeKernel` call. Validation
    exits, unsupported cooperative launch devices, occupancy failures, and
    rejected prelaunch attempts no longer count as observed true-fused kernel
    launches.
  - 2026-06-26 guarded the current strict cooperative body as smoke-shape-only
    by default. Production GPT row/vocab/hidden shapes now return CUDA
    not-supported unless
    `NFN_NATIVE_GPT_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION=1` (or the
    GPT2/Tile alias) is set for an unsafe diagnostic run. This keeps full-loop
    strict gates from spending time in the tiny proof kernel before the real
    production fused LM-head body exists.
- [x] Add shape-scoped BGRADB first-write diagnostics for transformer block
  dWeight+bias kernels. The global `bgrad_first_write_direct` route stays
  rejected, but Tile-CUDA now accepts
  `NFN_NATIVE_LINEAR_BGRAD_FIRST_WRITE_DIRECT_ENABLE_SHAPE=m,n,k,opA,opB`
  and matching aliases so QKV, attention projection, MLP FC, and MLP projection
  can be isolated in the same paired benchmark harness before any default
  promotion. This is a kernel-selection diagnostic, not a parity completion.
  The 2026-06-25 QKV, attention projection, MLP FC, and MLP projection shape
  gates each moved 36 first-write calls to direct writes and failed timing
  gates, so all four shape profiles are rejected by default.
- [x] Keep SM120 candidate sweep output aligned with current parity evidence.
  The no-argument sweep now includes the promoted `lm_head_graph_prewarm` gate
  and `summary.tsv` reports LM-head graph replay, cooperative sequence, and
  cuBLASLt BGRADB direct/accumulate route deltas in addition to the older QKV,
  loss-bin, and grouped-cuBLASLt proof columns.
- [x] Tighten the promoted QKV default-vs-legacy regression gate without
  overfailing on event-timing noise. A 2026-06-25 short dedicated RTX 5090
  sweep measured `qkv_dinput_ln128` at `0.989796x` train-loop wall and
  `0.979339x` block backward while missing only the strict steady-state
  CUDA-event gate at `1.000114x`; the profile now keeps strict train-loop and
  block-backward thresholds but allows the steady-state event slice up to
  `1.002x`, matching `lm_head_graph_prewarm`. The follow-up short GPU check
  still rejected a real `1.006174x` steady-state event regression while passing
  train-loop wall (`0.984400x`) and block backward (`0.966664x`), so the gate is
  not masking material steady-state regressions.
- [x] Revalidate the promoted QKV default-vs-legacy route after the CUDA
  Toolkit reinstall. The 2026-06-27 dedicated RTX 5090 5-step, 2-sample,
  no-warmup, stage-timed same-script gate kept `qkv_dinput_ln128` accepted:
  train-loop wall was `0.998609x` versus the older
  256-row/QKV-dWeight-first baseline, block backward was `0.996347x`, and
  candidate-over-llm.kittens train-loop wall was `0.999448x`. Route proof moved
  `block_backward_qkv_dinput_before_dweight_count` from `0` to `480` and
  `block_state_layout.layer_norm_backward_affine_row_chunk_size` from `256` to
  `128`; the selected RTX 5090 had no compute processes before samples.
- [x] Refresh canonical llm.kittens parity after promoting the no-loss LM-head
  CE vec8 normal-store specialized kernel. The 2026-06-27 dedicated RTX 5090
  5-step, 2-sample, 1-warmup, stage-timed same-script gate passed the explicit
  `1.003` workstation parity band: median train-loop wall was `1.001834x`,
  median steady-state CUDA-event wall was `1.002183x`, and runtime contract
  checks still reported `graph_editor_tensor_flow=false` and
  `torch_required=false`. Runtime JSON confirmed the default
  `lm_head_ce_kernel_strategy:
  no-loss-specialized-dlogits-vec8-loads-normal-vec8-stores`. The remaining
  parity blocker is unchanged: LM-head backward is still the diagnostic CUDA
  Graph wrapper with `true_fused_capability=false`, so the next material work
  remains a reference-aligned fused CE/dlogits path plus faster separate logits,
  dHidden, and dWeight stages.
- [x] Rechecked the default no-stage parity path after the CUDA 13.3.33 WSL
  reinstall and the latest rejected storage-profile work. The 2026-06-28
  dedicated RTX 5090 3-step, 1-sample same-script run included the llm.kittens
  reference and reported current NeuralFn at `0.999448x` train-loop wall time,
  `0.998914x` steady-state CUDA-event step time, and `1.000456x` tokens/sec
  versus llm.kittens. The selected GPU had display disabled and no compute
  processes before samples, and the runtime contract still passed with
  `graph_editor_tensor_flow=false` and `torch_required=false`. The
  `native_lm_head_true_fused_target` section still reports
  `diagnostic-cuda-graph-wrapper`, `true_fused_capability=false`, and
  `graph_body_nodes_per_replay=3`; treat that as the remaining architectural
  target rather than a current no-stage throughput regression.
- [x] Reject reduced MLP activation storage as a default startup-memory route.
  The 2026-06-27 dedicated RTX 5090 3-step, 2-sample, stage-timed probes for
  `NFN_NATIVE_GPT_STORE_MLP_BLOCKS=3` and `=9` reduced stored MLP activation
  bytes to `0.25x` and `0.75x` respectively, but both failed hot-loop gates by
  forcing recompute work through slower GEMM routes. The 3-block probe improved
  setup wall to `0.908060x` while regressing train-loop wall to `1.278677x`,
  steady-state CUDA-event step time to `1.277148x`, tokens/sec to `0.782060x`,
  block backward to `1.217694x`, and MLP projection backward to `1.458270x`.
  The 9-block probe improved setup wall only to `0.975999x` while regressing
  train-loop wall to `1.084793x`, steady-state CUDA-event step time to
  `1.082810x`, tokens/sec to `0.921834x`, block backward to `1.073931x`, and
  MLP projection backward to `1.155027x`. Keep the default at all 12 stored
  MLP blocks until a replacement avoids the BF16 pack/cache miss and cuBLASLt
  recompute penalty.
- [x] Recheck compile-flag-only SM120 reference alignment before changing
  defaults. The 2026-06-28 dedicated RTX 5090 3-step, 2-sample, stage-timed
  `llmk_sm120_reference_flags` rerun rebuilt a temporary Tile ops library with
  the documented llm.kittens SM120 macro bundle and passed
  candidate-over-llm.kittens gates (`0.999113x` train-loop wall,
  `0.999168x` steady-state CUDA-event step time, `1.000646x` tokens/sec).
  It remains rejected for default promotion because it changed no hot route
  counters or cuBLASLt plan-cache entries and was flat/slightly slower versus
  the current linked native baseline (`1.000196x` train-loop wall,
  `0.999805x` tokens/sec, `1.000278x` block backward). The default build
  already carries the material SM120 flags; the next useful slice remains
  real LM-head/backward kernel work, not another macro-bundle reroute.
- [x] Recheck the direct LM-head cooperative sequence wrapper after the same
  CUDA 13.3.33 refresh. The 2026-06-28 dedicated RTX 5090 3-step, 2-sample,
  stage-timed rerun kept `lm_head_cooperative_sequence_wrapper` rejected:
  disabling CUDA Graph replay for the sequence wrapper regressed train-loop
  wall to `1.012109x`, steady-state CUDA-event step time to `1.005261x`,
  tokens/sec to `0.988038x`, LM-head backward to `1.050922x`, and cooperative
  LM-head body time to `1.073406x`. Keep the CUDA Graph wrapper default until
  the replacement is a true fused/reference-aligned classifier-backward path.
- [x] Revisit the broad native test surface after the CUDA Toolkit reinstall.
  `tools/check_native_no_torch_deps.py --rebuild-stale --json` passed with all
  tracked native binaries and bindings unstale, and
  `python -m pytest tests/test_native_gpt2.py -q` passed with `100 passed, 2
  skipped` after correcting the strict LM-head source-contract assertion to
  match escaped C++ JSON string literals.
- [x] Refresh the native-vs-llm.kittens parity measurement after the CUDA WSL
  reinstall and dedicated RTX 5090 setup. The 2026-06-24 CUDA 13.3.33
  3-step/1-sample same-script run measured NeuralFn at `2512.313 ms/step`
  versus llm.kittens at `2474.457 ms/step` (`1.015299x` train-loop wall,
  `0.984089x` tokens/sec), with the steady-state CUDA-event slice at
  `1.011877x`. The selected GPU was idle before and after the run, with no
  compute processes.
- [x] Refresh stage attribution under the same stack. The 2-step stage-timed
  run measured `1.008363x` train-loop wall and `1.015418x` steady-state
  CUDA-event timing. Remaining hot buckets are native kernel throughput:
  `stage.block_backward.total_ms`, `stage.block_forward.total_ms`,
  `stage.block_backward.attn_sdpa.to_qkv.total_ms`,
  `stage.block_backward.mlp_proj.total_ms`, QKV backward, and
  `stage.lm_head_backward.total_ms`.
- [x] Refresh the stronger current parity sample after the linked rebuilds and
  latest candidate rejections. The 2026-06-24 CUDA 13.3.33 5-step, 3-sample,
  1-warmup same-script run measured NeuralFn at `2525.500 ms/step` versus
  llm.kittens at `2465.055 ms/step` (`1.024520x` train-loop wall,
  `0.975643x` tokens/sec), with the steady-state CUDA-event slice at
  `1.014749x` and first-step CUDA-event slice at `1.061982x`. The selected RTX
  5090 was idle before and after every sample, with no compute processes. Native
  setup averaged `634.259 ms`, mainly float arena materialization
  (`265.065 ms`), token weight initialization (`158.416 ms`), uint16 arena
  materialization (`124.713 ms`), and cuBLASLt plan prewarm (`74.021 ms`).
- [x] Refresh parity after restoring the rejected attention-projection ordering
  route to default-off and rebuilding `nfn_gpt_native_train_linked`. The
  2026-06-24 CUDA 13.3.33 3-step, 2-sample, 1-warmup stage-timed same-script
  run measured NeuralFn at `2555.538 ms/step` versus llm.kittens at
  `2460.547 ms/step` (`1.038651x` train-loop wall, `0.962343x` tokens/sec),
  with the steady-state CUDA-event slice at `1.012345x` and first-step
  CUDA-event slice at `1.089442x`. The selected RTX 5090 was idle before and
  after every sample, with no compute processes. Current hot NeuralFn buckets
  are still `stage.block_backward.total_ms` (`3856.435 ms`),
  `stage.block_forward.total_ms` (`1965.470 ms`), and
  `stage.lm_head_backward.total_ms` (`1753.965 ms`).
- [x] Refresh parity after the latest CUDA reinstall/catalog commits and fix
  the parity wrapper shape override. The 2026-06-24 2-step, 1-sample,
  stage-timed same-script run on the display-disabled RTX 5090 measured
  NeuralFn at `2584.085 ms/step` versus llm.kittens at `2505.865 ms/step`
  (`1.031215x` train-loop wall, `0.968518x` tokens/sec), with steady-state
  CUDA-event timing at `1.014173x`. The selected GPU had zero compute
  processes before and after. `tools/bench_native_gpt_sm120_parity.sh` now
  wires `NFN_SM120_PARITY_TRAIN_BATCH_TOKENS` /
  `NFN_SM120_TRAIN_BATCH_TOKENS` into both llm.kittens `-d` and NeuralFn
  `--train-batch-tokens`, so future shape bisection preserves same-script
  parity instead of silently using hardcoded `524288` on both sides.
- [x] Reject disabling cuBLASLt plan prewarm as a startup-only optimization on
  the current CUDA 13.3 dedicated RTX 5090 stack. The
  `cublaslt_plan_prewarm_off` candidate improved setup wall to `0.834325x`,
  but moved lazy plan work into the hot path: train-loop wall regressed to
  `1.015300x`, first-step CUDA-event time to `1.044809x`, tokens/sec to
  `0.984974x`, LM-head backward to `1.031614x`, and block backward to
  `1.023253x`. Keep full plan prewarm enabled for real training.
- [x] Reject packed-attention dprep warp-count retuning on the current CUDA
  13.3 RTX 5090 stack. `attention_dprep_warps_2` changed
  `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_WARPS` from the default 3 to 2 but
  regressed train-loop wall to `1.004595x`, steady-state CUDA-event timing to
  `1.008058x`, block backward to `1.017877x`, and
  `attention_backward_tk_timing_us` to `1.002175x`. `attention_dprep_warps_4`
  changed the same knob to 4 but regressed train-loop wall to `1.005938x`,
  block backward to `1.020308x`, attention SDPA to `1.001733x`, and
  `attention_backward_tk_timing_us` to `1.001117x`. Keep both as rejected
  named profiles unless CUDA or the TK attention backward implementation
  materially changes.
- [x] Refresh the post-CUDA-reinstall llm.kittens parity gate on 2026-06-25
  with the linked native trainer on the display-disabled RTX 5090. The
  10-step/1-sample same-script run measured NeuralFn at `2521.900 ms/step`
  versus llm.kittens at `2473.866 ms/step` (`1.019417x` train-loop wall,
  `0.977908x` tokens/sec), with steady-state CUDA-event timing at
  `1.007778x`. Disabling NeuralFn train-loop CUDA event timing did not close
  the gap (`1.010888x` wall), so the failure is not a timing-instrumentation
  artifact.
- [x] Recheck the two plausible non-kernel LM-head escape hatches against
  llm.kittens after the CUDA reinstall. The graph-prewarm route eliminated
  runtime LM-head graph captures (`capture_success_mean=0`,
  `prewarm_success_mean=3`) but still failed at `1.009454x` train-loop wall and
  `1.006953x` steady-state CUDA-event timing. The explicit sequence-wrapper
  route was worse (`1.025042x` wall, `1.011626x` steady-state). Keep both
  diagnostic-only.
- [x] Add failed-parity diagnostics to `tools/bench_native_gpt_sm120_parity.sh`.
  When the paired JSON shows native Tile training is active but LM-head backward
  is still the `diagnostic-cuda-graph-ce-fork-join-dhidden-dweight-not-single-kernel`
  route and `lm_head_cooperative_backward_fused_kernel_available=false`, the
  wrapper now prints the exact blocker and points at the strict true-fused
  LM-head Tile kernel ABI. This preserves the failing exit code instead of
  weakening the gate.
- [x] Add a compact LM-head classifier-backward path class to native GPT
  runtime JSON and paired benchmark extraction. As of 2026-06-25 the dedicated
  RTX 5090 JSON-contract parity smoke reports
  `lm_head_classifier_backward_path_class: diagnostic-cuda-graph-wrapper`,
  which keeps the current CUDA Graph wrapper visibly separate from the future
  `strict-true-fused-tile-kernel` route.
- [x] Add an ABI-declared path class for the strict LM-head fused symbol.
  `nfn_native_tile_lm_head_classifier_backward_fused_kernel_path_class()`
  currently returns `diagnostic-cuda-graph-wrapper`; native GPT JSON reports
  `lm_head_cooperative_backward_fused_kernel_abi_path_class`, and the focused
  benchmark reports `candidate_symbol_abi_path_class`. This keeps future
  strict-fused promotions from relying only on counter inference.
- [x] Add explicit CUDA Graph body node attribution for the diagnostic LM-head
  fused symbol. The ABI now reports a three-node graph body: CE/dlogits,
  dHidden, and dWeight. Native GPT runtime JSON reports per-replay and
  replay-total `lm_head_fused_graph_body_*` counts, so the next true fused Tile
  kernel can be compared against the actual wrapper structure instead of an
  opaque replay counter.
- [x] Refresh post-rebuild SM120 parity and make the paired hot-stage report
  show per-call context. After rebuilding the default Tile ops library and
  linked native trainers with the current build-script dependency invalidation,
  the 2026-06-25 2-step/1-sample same-script parity smoke measured NeuralFn at
  `2516.720 ms/step` versus llm.kittens at `2437.755 ms/step`
  (`1.032393x` train-loop wall, `0.968559x` tokens/sec), with steady-state
  CUDA-event timing at `1.007771x`. A follow-up 2-step stage-timed run measured
  `1.039801x` train-loop wall, `1.011402x` steady-state CUDA-event timing, and
  hot native buckets of `stage.block_backward.total_ms=2512.060 ms`
  (`count=192`, `avg_ms=13.0836`),
  `stage.lm_head_backward.total_ms=1141.240 ms` (`count=16`,
  `avg_ms=71.3272`), and
  `stage.lm_head_backward.cooperative.total_ms=791.149 ms` (`count=32`,
  `avg_ms=24.7234`). `tools/paired_kernel_speed.py` now prints `count_mean`
  and `avg_ms_mean` beside hot-stage `*.total_ms` lines so future
  candidate-vs-current gates do not require manual sidecar JSON parsing to see
  whether a regression is per-layer, per-chunk, or one-time setup work.
- [x] Refresh current no-stage SM120 parity after the LM-head graph-upload and
  loss-bin evidence updates. The 2026-06-25 3-step, 2-sample same-script run
  with the display-disabled RTX 5090 measured llm.kittens at `2455.180 ms/step`
  and NeuralFn at `2503.315 ms/step`, or `1.019609x` train-loop wall and
  `0.980724x` tokens/sec. Steady-state CUDA-event timing was much closer at
  `1.002080x`, but the wrapper still failed strict parity gates and printed
  the current blocker: native Tile training is active, LM-head graph replay is
  active (`lm_head_fused_graph_replay_count=48`), but
  `lm_head_cooperative_backward_fused_kernel_available=false` and
  `lm_head_classifier_backward_path_class=diagnostic-cuda-graph-wrapper`.
  The next implementation target remains replacing
  `nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16` with a
  true fused classifier-backward Tile kernel.
- [ ] Close the remaining SM120 parity gap with measured native kernel changes,
  not Torch/Python/graph-editor workarounds. Every candidate must run through
  `tools/bench_native_gpt_sm120_candidate.sh` or
  `tools/bench_native_gpt_sm120_parity.sh` so baseline and candidate execute in
  the same script under the same external GPU load.
  - 2026-06-28 added explicit hot-route launch counters for the default-off
    block-backward concurrent schedules:
    `block_backward_qkv_concurrent_dinput_dweight_count`,
    `block_backward_mlp_fc_concurrent_dinput_dweight_count`, and
    `block_backward_attn_proj_concurrent_dinput_dweight_count`. The dense GPT
    JSON, paired metric extractor, route-change gate, and sweep `summary.tsv`
    now expose those deltas, so side-stream candidates must prove real kernel
    schedule changes inside the same script instead of relying on booleans or
    standalone timings.
  - 2026-06-28 added `--require-native-hot-route-counter NAME` to the paired
    speed tool and wired the SM120 side-stream profiles to require their exact
    launch counter. This keeps future CUDA Tile experiments from passing route
    proof because an unrelated strategy field changed while the intended hot
    kernel schedule did not run.
  - 2026-06-28 reran `qkv_concurrent_dinput_dweight` on the dedicated RTX 5090
    with the hot-route counter gate. The profile proved the intended schedule
    change by moving `block_backward_qkv_concurrent_dinput_dweight_count` from
    `0` to `288` and `block_backward_qkv_dinput_before_dweight_count` from
    `288` to `0`, but still rejected promotion at `1.001725x` train-loop wall,
    `1.002119x` steady-state CUDA-event step time, `0.998283x` tokens/sec, and
    `1.003575x` candidate-over-llm.kittens train-loop wall. Keep this
    side-stream branch diagnostic-only; it is not the remaining block-backward
    kernel fix.
  - 2026-06-28 tightened the paired native runtime contract gate so SM120
    candidate benchmarks must also report `train_loss_host_d2h_count=0`.
    Candidate promotion now fails if the timed training path pulls train-loss
    values back to the host, keeping logging/eval work out of native CUDA Tile
    kernel measurements while the real LM-head fusion gap remains open.
  - 2026-06-25 added first-step versus steady-state stage aggregation to the
    native GPT CUDA-event timing JSON and paired benchmark extractor. The
    rebuilt linked trainer's 3-step TinyStories native-only diagnostic measured
    `train_loop_cuda_event_first_step_wall_ms_per_step=2695.35` versus
    `train_loop_cuda_event_steady_state_wall_ms_per_step=2444.04`. The biggest
    first-step delta was forward attention QKV:
    `stage.block_forward.attention.qkv.first_step_avg_ms=3.315` versus
    `stage.block_forward.attention.qkv.steady_state_avg_ms=1.083`, which keeps
    the next startup/parity work pointed at first-use forward-QKV/TK or plan
    warmup behavior rather than graph-editor, Torch, or validation paths.
  - 2026-06-25 added the default-off
    `NFN_NATIVE_GPT_PREWARM_TK_QKV_FORWARD=1` diagnostic and reproducible
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=tk_qkv_forward_prewarm`. The route
    proof kept `graph_editor_tensor_flow=false` and moved
    `linear_tk_qkv_first_use_prewarm_success_count` from `0` to `1`. The
    same-script split-stage gate improved train-loop wall to `0.982548x`,
    first-step CUDA-event timing to `0.948385x`, and forward-QKV first-step avg
    to `0.435121x`, but it is not a default because setup regressed to
    `1.239921x` and total wall to `1.000981x`. Treat it as evidence that the
    first-step QKV gap is first-use setup work; the real fix needs to remove or
    lower that cost, not move it earlier.
  - 2026-06-26 added
    `NFN_NATIVE_GPT_PREWARM_TK_QKV_FORWARD_ROWS=N` plus the rejected
    `tk_qkv_forward_prewarm_1row` profile. The one-row route proved the row cap
    and moved `linear_tk_qkv_first_use_prewarm_success_count` from `0` to `1`,
    improving forward-QKV first-step avg to `0.477316x`, but setup regressed to
    `1.415877x`, total wall to `1.018835x`, and llm.kittens reference gates
    still failed. Keep QKV prewarm diagnostic-only; startup parity needs a
    cheaper first-use/TK setup path, not another prewarm relocation.
  - 2026-06-25 rejected LM-head graph prewarm as a default despite native-vs-native
    startup and LM-head improvements, and also rejected the explicit cooperative
    sequence wrapper, CUDA event timing changes, and the llm.kittens SM120 macro
    rebuild as complete parity fixes. The latest graph-prewarm refresh improved
    train-loop wall to `0.995921x`, tokens/sec to `1.004099x`, and LM-head
    backward to `0.966626x`, but missed promotion at `1.002619x` steady-state
    CUDA-event timing and `1.009095x` block backward. The macro rebuild improved current
    NeuralFn wall time to `0.996738x` native-vs-native but changed no route
    counters and still missed the steady-state gate, so it remains timing noise
    rather than a kernel implementation. The active implementation target is now
    narrowed to replacing
    `nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16` with a
    real fused classifier-backward CUDA Tile kernel and flipping
    `nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused()`
    only when the strict microbenchmark passes.
  - 2026-06-25 promoted `lm_head_graph_prewarm` after the CUDA 13.3.33
    post-reinstall same-script rerun. The profile still compares only
    `NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_GRAPH_PREWARM=0` vs `=1`; cuBLAS handle
    and BF16 workspace prewarm are already default-on and must not be disabled
    on the baseline side. The graph-only rerun passed at `0.970282x` train-loop
    wall, `1.001894x` steady-state CUDA-event timing, `0.968319x` LM-head
    backward, `0.956792x` block backward, and `0.911989x` MLP projection
    backward. The native trainer now defaults graph prewarm on, with
    `NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_GRAPH_PREWARM=0` as the bisection route.
  - 2026-06-25 added and rejected
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_graph_thread_cache_prewarm`.
    The route primes the LM-head CUDA Graph replay thread cache during graph
    prewarm with `NFN_NATIVE_GPT_LM_HEAD_GRAPH_PREWARM_THREAD_CACHE=1`, moving
    `lm_head_fused_graph_thread_cache_hit_count` from `45` to `48`. The
    same-script 3-step, 1-sample stage-timed RTX 5090 gate rejected default
    promotion because train-loop wall regressed to `1.003958x`, steady-state
    CUDA-event timing to `1.002099x`, LM-head backward to `1.000922x`, and
    tokens/sec to `0.996059x`. Keep the route default-off and diagnostic-only.
  - 2026-06-27 promoted the former
    `lm_head_ce_no_loss_vec8_normal_store_specialized` CUDA Tile kernel
    candidate as the default no-loss CE+dlogits route. The default now reports
    `lm_head_ce_kernel_strategy:
    no-loss-specialized-dlogits-vec8-loads-normal-vec8-stores`; set
    `NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_VEC8_NORMAL_STORE_SPECIALIZED=0` to
    compare against the older scalar-store specialized route. The accepted
    default-vs-legacy profile gates train-loop wall, steady-state CUDA-event
    wall, graph-wrapper LM-head aggregate time, and tokens/sec inside a tight
    `0.1%` same-script band because the CUDA Graph LM-head wrapper does not
    emit a standalone CE body substage. The dedicated RTX 5090 rerun passed
    with route strategy changes, `1.000131x` train-loop wall, `0.999777x`
    steady-state CUDA-event wall, `1.000699x` LM-head aggregate, and
    `0.999870x` tokens/sec.
  - 2026-06-25 catalogued
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_cooperative_backward_off` as
    the rejected direct CE+dHidden+dWeight fallback. A 2026-06-28 CUDA 13.3.33
    dedicated RTX 5090 current-default 3-step, 2-sample no-stage rerun improved
    setup wall to `0.868531x`, but rejected disabling the graph wrapper because
    train-loop wall regressed to `1.010400x`, first-step CUDA-event time to
    `1.028008x`, tokens/sec dropped to `0.989710x`, candidate-over-llm.kittens
    train-loop wall was `1.005850x`, and candidate-over-llm.kittens tokens/sec
    was `0.994488x`. Disabling the graph wrapper is still not a parity fix.
  - 2026-06-24 hardened `tools/bench_native_gpt_sm120_parity.sh` against
    no-op profile evidence: it now exits before GPU work when
    `NFN_SM120_PARITY_CANDIDATE_PROFILE` or `NFN_SM120_PARITY_PROFILE` is set.
    Use the native-vs-native candidate wrapper for named profile expansion, or
    pass explicit `NFN_SM120_PARITY_CANDIDATE_ENV` values when comparing
    NeuralFn against llm.kittens.
  - 2026-06-24 added metadata-only `nfn kernels list --kind ... --status ...`
    filters on both the lightweight Torch-free CLI path and the full CLI path.
    `nfn kernels list --status host_only --kind module --json` now emits only
    the source/orchestration module specs while preserving global coverage
    totals, making the non-kernel data boundaries explicit during the native
    migration.
  - 2026-06-24 added the metadata-only compiled GPT template catalog action:
    `nfn_gpt_native_train --list-templates`, `nfn train --base-model gpt
    --list-templates`, and wrapper alias `--native-cuda-list-templates` print
    shipped template/native support status without resolving token shards,
    opening datasets, importing Torch, or routing real training data through
    graph-editor nodes.
    The no-Torch verifier now runs the `train_gpt.py` wrapper alias and the
    top-level `nfn train --base-model gpt --list-templates` path under the
    import blocker so that catalog lookups cannot regress into Python graph or
    dataset startup.
    The `train_gpt.py --native-cuda-list-templates` wrapper also strips its
    default dataset alias and eval cadence flags before launching native C++.
  - 2026-06-24 rechecked `qkv_dinput_before_dweight` after the CUDA reinstall:
    it improved train-loop wall to `0.994580x` but still missed strict
    steady-state, LM-head, MLP-projection, and QKV gates, so it remains
    rejected.
  - 2026-06-24 added the named `qkv_dinput_ln64` paired profile for the best
    current near-miss: QKV dInput-before-dWeight plus 64-row LayerNorm affine.
    The stronger 5-step, 3-sample same-script confirmation improved
    steady-state CUDA-event timing to `0.998529x`, but regressed train-loop
    wall to `1.000261x`, total block backward to `1.000938x`, MLP projection to
    `1.004308x`, and QKV backward to `1.007310x`, so it remains rejected.
  - 2026-06-24 corrected the same-script benchmark contract for the promoted
    `qkv_dinput_ln128` route. It is now an allowed promoted comparison profile,
    not a rejected profile requiring an override, and compares the current
    default QKV dInput-before-dWeight plus 128-row LayerNorm affine path against
    the old 256-row/QKV-dWeight-first route while emitting `candidate_note`
    metadata in paired results. Its promoted-profile gates now match the
    promotion evidence: train-loop wall, steady-state CUDA-event wall, total
    block backward, and train tokens/sec.
  - 2026-06-24 rechecked `lm_head_row_loss_partial_reduce` after the CUDA
    reinstall and dedicated RTX 5090 setup. The paired 3-step, 2-sample run
    changed only the row-loss accumulation strategy
    (`NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE=1 -> 0`) and improved the
    steady-state CUDA-event slice to `0.999166x`, but failed the promotion gate
    because train-loop wall time regressed to `1.002484x` and tokens/sec fell
    to `0.997528x`. Keep row-loss sum accumulation enabled by default; this is
    not a parity-closing LM-head kernel.
  - 2026-06-24 tested `linear_bias_row_chunk_256` after rebuilding both native
    GPT CLIs. The paired 2-step, 2-sample stage-timed run changed
    `block_state_layout.linear_backward_bias_row_chunk_size` from `512` to
    `256` and improved train-loop wall to `0.997526x` plus steady-state
    CUDA-event timing to `0.997142x`, but was too noisy as a fresh strict
    confirmation sample because
    `stage.block_backward.total_ms` regressed to `1.009570x` and
    `stage.block_backward.mlp_fc.dweight_bias.total_ms` regressed to
    `1.000482x`. Keep the promoted 256-row bias reducer default from the
    stronger prior 5-step, 3-sample gate; this profile remains a normal
    default-confirmation comparison against the old 512-row route, not a
    rejected rollback.
  - 2026-06-24 rechecked `cublaslt_grouped_probe` after the current CUDA 13.3
    rebuilds. The dedicated RTX 5090 same-script capability run still reports
    `linear_cublaslt_grouped_layout_probe_status: 0`, but grouped matmul
    execution remains unsupported at
    `linear_cublaslt_grouped_matmul_probe_status: 15`. Keep grouped block
    dWeight/BGRADB work blocked until execution status is `0`; the active
    parity direction remains a true fused/cooperative LM-head kernel or another
    measured block-backward Tile kernel route.
  - 2026-06-24 tightened the cuBLASLt grouped matmul probe to use 32-bit
    grouped rows/columns/leading-dimension arrays, matching cuBLASLt's default
    grouped descriptor integer width. The strict
    `cublaslt_grouped_probe_required` profile still fails correctly with
    layout status `0` and matmul status `15`; a temporary explicit 64-bit
    grouped-width experiment returned status `7`, so grouped block dWeight and
    BGRADB work remains blocked on execution support, not descriptor creation.
  - 2026-06-24 added list parsing for
    `NFN_NATIVE_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE` /
    `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE` and the paired
    `cublaslt_block_dinput` profile for the dense GPT MLP projection, MLP FC,
    QKV, and attention projection dInput shapes. The isolated linear benchmark
    improved all four raw-symbol comparisons (`0.985876x` to `0.998370x`), but
    the same-script native trainer gate rejected the profile because the
    default loop already used the same cuBLASLt dInput plans: route counters,
    strategy values, shape route names, and plan cache entries did not change.
    Keep the profile rejected/diagnostic-only.
  - 2026-06-24 added
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_bf16_hidden_from_final_norm`.
    The candidate routes final LayerNorm through the BF16-output LayerNorm ABI
    so it writes the full LM-head BF16 hidden buffer directly and elides
    `lm_head_backward.hidden_prepack`. The same-script 3-step, 2-sample
    dedicated RTX 5090 gate proved the strategy change
    (`lm_head_bf16_hidden_from_final_norm_enabled: false -> true`) but rejected
    promotion at `1.009000x` train-loop wall, `1.000147x` steady-state
    CUDA-event timing, and `1.000293x` LM-head dWeight. Keep it
    diagnostic-only; the remaining LM-head parity work is still a real
    fused/cooperative classifier-backward kernel, not moving the prepack cost.
  - 2026-06-24 rechecked
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=bf16_workspace_prewarm` on the current
    rebuilt linked native trainer. The 5-step, 3-sample, 1-warmup same-script
    run changed only setup/prewarm counters
    (`linear_bf16_workspace_prewarm_requested/success_count: 0 -> 1`) and
    failed the route-change gate. It measured `0.999466x` train-loop wall and
    `0.999417x` steady-state CUDA-event timing, but setup regressed to
    `1.005087x` and strict stage gates missed at
    `stage.lm_head_backward.total_ms=1.000361x` and
    `stage.block_backward.mlp_proj.total_ms=1.001043x`. Keep it
    diagnostic-only; it is neither a startup fix nor a parity-closing hot
    kernel route.
  - 2026-06-24 refreshed the current stage-timed parity sample after the
    LM-head microbench graph-counter wiring:
    `NFN_SM120_PARITY_STEPS=3 NFN_SM120_PARITY_SAMPLES=1
    NFN_SM120_PARITY_WARMUP=1 NFN_SM120_PARITY_STAGE_TIMING=1
    NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0
    NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_after_graph_microbench_20260624.json
    bash tools/bench_native_gpt_sm120_parity.sh`. The dedicated RTX 5090 had
    display disabled, zero compute processes, and 0% utilization before/after
    the sample. llm.kittens measured `2467.650000 ms/step`; NeuralFn measured
    `2529.723333 ms/step`, or `1.025155x` train-loop wall, `1.014369x`
    steady-state CUDA-event wall, and `0.975088x` tokens/sec. Current hot
    NeuralFn buckets over three steps remain
    `stage.block_backward.total_ms=3771.620`,
    `stage.train.model_forward.total_ms=1982.190`, and
    `stage.lm_head_backward.total_ms=1755.510`, with LM-head split across
    logits `522.590`, CE `207.662`, dHidden `532.341`, and dWeight `488.064`.

## Native C++ trainer ABI

This section tracks the raw no-Torch C ABI used by compiled model trainers. It is separate from the PyTorch extension bindings and autograd wrappers.

- [x] Build `libnfn_native_train_tile_ops.so` from CUDA Tile kernels without `torch/extension.h`.
  - 2026-06-20 CUDA Toolkit 13.3.33 requires the SM120 trainer-facing Tile ops build to define `LLMK_SM120_USE_CUBLASLT_GEMM` by default and normalize inherited `NFN_TILE_CUDA_ARCH=sm_120` / `compute_120` to `sm_120a` / `compute_120a` when TK attention is enabled. Without that llm.kittens-aligned compile route, `ptxas` rejects several raw TK GEMM instantiations for static shared-memory usage above the default `0xc000` cap.
- [x] Expose AdamW, gradient accumulation, reductions, in-place scaling, and linear forward through `neuralfn/csrc/native_train/tile_ops.h`.
- [x] Extend the no-Torch dependency verifier across GPT, GPT-2-evo, NanoGPT, `nfn train`, native inference, `neuralfn.native_train`, and public SDK native training exports so legacy handoff surfaces cannot import Torch/NumPy/tiktoken/dataset-manager code before delegating to compiled native CLIs.
  - 2026-06-19 also made the pyproject contract enforce the same boundary: the default package install has no hard dependency on Torch, NumPy, tokenizer, dataset, graph-analysis, or server packages. Those workflows are now explicit extras (`tile-cuda`, `datasets`, `graph`, `server`, `torch`, or `all`), and the no-Torch verifier fails if those packages move back into default dependencies.
  - 2026-06-21 extended the default native artifact scan to built SDK binding modules matching `neuralfn/_native*.so`, so `_native_gpt`, `_native_gpt2`, and `_native_train` cannot accidentally regain `libtorch`, `libc10`, or `libpython` links while the SDK moves to C++ binding dispatch. The same verifier now imports those built modules under the blocked-import harness when present.
  - 2026-06-23 added `shell_entrypoints` coverage to the verifier for native
    benchmark planning wrappers. `tools/bench_linear_backward_candidate.sh` now
    has `NFN_LINEAR_BACKWARD_DRY_RUN=1`, and the verifier dry-runs both that
    wrapper and `tools/bench_native_gpt_linear_hot_matrix.sh` so benchmark
    command planning cannot pull in Torch, graph-editor, dataset, build, or CUDA
    startup work.
  - 2026-06-24 extended the verifier across universal dense GPT architecture
    selection: `nfn train --template-name gpt2_moa` and `nfn train
    --base-model gpt3 --graph-file ...` now run under the same import blocker
    and stubbed compiled CLI, proving template/custom-graph selection stays on
    the native command path instead of importing the graph-backed runtime.
  - 2026-06-28 removed the `neuralfn.native_cuda_device` package import from
    the direct native wrappers (`nfn train`, `cli/scripts/train_gpt.py`, and
    `native_training_guard.py`). They now keep the small CUDA-visible-device
    resolver local to the script so default device selection cannot import the
    Python SDK package before the compiled C++ trainer handoff.
- [x] Add a generic native-train binding command resolver so SDK tests and callers can inspect the compiled argv that `neuralfn._native_train` will spawn without importing Torch, dataset managers, or graph payload paths.
- [x] Expose gradient/device-buffer fill through the native ABI for trainer-loop zeroing.
- [x] Expose global gradient norm clip scale finalization and device-scalar gradient scaling through the native ABI.
- [x] Expose token embedding, absolute position embedding, RMSNorm, LayerNorm, softmax, scaled dot-product attention, token CE partials, and masked token CE partials through the native ABI.
- [x] Expose token CE logits backward and masked token CE logits backward through the native ABI.
- [x] Expose low-overhead cuBLASLt plan-cache inspection through
  `nfn_native_tile_trainer_linear_cublaslt_plan_cache_count` and
  `nfn_native_tile_trainer_linear_cublaslt_plan_cache_entry`, with native GPT
  JSON fields `linear_cublaslt_plan_cache_available`,
  `linear_cublaslt_plan_cache_count`, and `linear_cublaslt_plan_cache`. Use
  this for normal same-script parity runs where synchronized
  `linear_shape_stats` timing would perturb the kernel candidate.
- [x] Expose the dense-GPT LM-head classifier row-chunk path as a distinct
  native Tile ABI route with launch/shape counters, separate from the generic
  token CE symbols. The current symbol still performs BF16/u16 public-vocab
  loss+dlogits over each row chunk; the remaining parity work is to replace its
  internals with a cooperative classifier plus LM-head dHidden/dWeight kernel
  without reintroducing full resident logits.
  - 2026-06-26 proved the current tiny strict cooperative kernel in the focused
    smoke path with `true_fused_launch_count=1` and sequence-wrapper/CUDA Graph
    counters at zero, then wired the same launch counter into native GPT
    full-loop route tracking. This closes the evidence gap only; the production
    parity item remains a bounded full-shape fused classifier/dHidden/dWeight
    kernel.
  - 2026-06-22 added a default-off strict parity guard for this route:
    `nfn_gpt_native_train --require-cooperative-lm-head-backward` and
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_cooperative_backward_required`
    now fail explicitly unless the strict cooperative classifier/dHidden/dWeight
    ABI symbol is available, reports a true fused capability, and is integrated.
    The SM120 wrapper treats the named profile as a strict ABI preflight probe,
    not a metric-gated speed candidate. Runtime JSON reports
    `lm_head_cooperative_backward_required`,
    `lm_head_cooperative_backward_requested`,
    `lm_head_cooperative_backward_abi_wrapper_available`,
    `lm_head_cooperative_backward_sequence_wrapper_available`,
    `lm_head_cooperative_backward_kernel_available`,
    `lm_head_cooperative_backward_fused_kernel_available`,
    `lm_head_cooperative_backward_route_integrated`,
    `lm_head_cooperative_backward_kernel_enabled`, and
    `lm_head_cooperative_backward_strategy`.
  - 2026-06-22 corrected the runtime contract so the existing
    `nfn_native_tile_lm_head_classifier_backward_cooperative_fused_bf16_u16`
    symbol is reported as the event-ordered sequence wrapper, not as a true
    fused parity kernel. `lm_head_cooperative_backward_kernel_available` and
    `lm_head_cooperative_backward_fused_kernel_available` now require the
    separate future symbol
    `nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16`; until
    that exists, `--require-cooperative-lm-head-backward` must fail instead of
    accepting wrapper-only CE/dHidden/dWeight sequencing.
  - 2026-06-24 corrected the strict future symbol capability bit as well:
    `nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16` is
    currently a cached CUDA Graph over the existing CE/dHidden/dWeight kernels,
    so `nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused()`
    returns `0` until a real single-kernel/cooperative body replaces it.
  - 2026-06-24 added separate CUDA Graph route reporting for the non-required
    cooperative LM-head path. `NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1`
    can now select the graph symbol when it is present and report
    `lm_head_cooperative_backward_cuda_graph_available` /
    `lm_head_cooperative_backward_cuda_graph_enabled`, while
    `--require-cooperative-lm-head-backward` still requires the true fused
    capability and remains blocked by the current graph wrapper.
  - 2026-06-25 split the strict LM-head contract into the future monolithic
    true-fused bit and the current llm.kittens-equivalent native parity bit.
    The Tile ABI now exports
    `nfn_native_tile_lm_head_classifier_backward_llmk_classifier_matmul_parity()`
    so runtime JSON can distinguish native llm.kittens-equivalent graph/wrapper
    parity from the future monolithic kernel bit.
  - 2026-06-25 tightened strict LM-head capability semantics again:
    `--require-cooperative-lm-head-backward`,
    `lm_head_cooperative_backward_kernel_available`, and
    `lm_head_cooperative_backward_fused_kernel_available` now require
    `nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused()`
    to return nonzero. The current parity bit remains visible as
    `lm_head_llmk_classifier_matmul_parity_available`, but does not satisfy
    strict kernel availability. Current CUDA 13.3 builds therefore keep
    `lm_head_cooperative_backward_kernel_available=false` and
    `lm_head_cooperative_backward_cuda_graph_enabled=true`; the open work
    remains a true fused classifier/dHidden/dWeight kernel under the strict
    callable.
  - 2026-06-22 made the non-required
    `NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1` /
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_cooperative_backward` path
    actually execute the existing event-ordered sequence wrapper when the true
    fused parity kernel is absent. Runtime JSON now reports
    `lm_head_cooperative_backward_sequence_wrapper_enabled` separately from
    `lm_head_cooperative_backward_kernel_enabled`, so the paired route-change
    gate can detect a real wrapper-route change while the strict
    `--require-cooperative-lm-head-backward` profile still fails on
    wrapper-only builds. A one-step dedicated RTX 5090 probe confirmed the
    route-change gate now passes through the strategy and
    `lm_head_cooperative_backward_sequence_wrapper_enabled` changes, but still
    rejected the wrapper at `1.007071x` train-loop wall, `1.000602x` LM-head
    backward, and `1.001183x` block backward. This is a diagnostic route, not
    the parity kernel.
  - 2026-06-23 moved the strict
    `--require-cooperative-lm-head-backward` training guard ahead of cached
    token-shard resolution and CUDA runtime setup. Missing or placeholder true
    fused LM-head kernels now fail immediately with a capability error, while
    `--check-tile-ops --require-cooperative-lm-head-backward` remains the JSON
    inspection path for strict ABI status.
  - [ ] Implement the actual cooperative LM-head backward Tile ABI that fuses or
    co-schedules classifier dlogit production with dHidden and dWeight work
    without materializing full resident logits or routing tensors through Torch.
    - 2026-06-27 wired strict SM120 true-fused LM-head full-loop candidate
      profiles through the focused `tools/bench_lm_head_backward_candidate.sh`
      preflight. `NFN_SM120_NATIVE_LM_HEAD_BACKWARD_PREFLIGHT=auto` selects the
      matching trainer-chunk profile, and the SM120 `*_GAP_MS` aliases forward
      to the focused absolute-gap gates so a known-slow strict body fails before
      the expensive llm.kittens/native paired run.
    - 2026-06-27 extended `build/lm_head_backward_bench` JSON with
      `candidate_reference_gap`: absolute candidate-minus-reference timings for
      generic and cuBLASLt reference component paths plus the current
      bottleneck component names. The benchmark still rejects placeholder
      wrappers, but the next Tile kernel iteration can now identify the
      remaining gap from one same-process run without manual subtraction.
    - 2026-06-28 extended each focused LM-head benchmark variant JSON with
      graph-body cuBLASLt launch and Tile fallback counters for dHidden and
      dWeight. Future candidate runs must use these counters with graph replay
      counters to prove the measured body is the optimized Tile classifier
      route and not a diagnostic cuBLASLt/fallback path.
    - 2026-06-28 made the focused `trainer-chunk` LM-head profile require that
      proof by default via `NFN_LM_HEAD_BACKWARD_REQUIRE_GRAPH_BODY_TILE=1`.
      The profile now fails if the candidate misses graph replay, falls back
      out of the graph, uses cuBLASLt for dHidden/dWeight, or does not increment
      both Tile graph-body counters.
    - 2026-06-28 added opt-in section-cycle gates to the focused LM-head
      wrapper:
      `NFN_LM_HEAD_BACKWARD_MAX_TRUE_FUSED_CE_CYCLES_PER_BLOCK`,
      `NFN_LM_HEAD_BACKWARD_MAX_TRUE_FUSED_DHIDDEN_CYCLES_PER_BLOCK`, and
      `NFN_LM_HEAD_BACKWARD_MAX_TRUE_FUSED_DWEIGHT_CYCLES_PER_BLOCK`. Each gate
      requires a nonzero strict true-fused launch count and then fails on the
      specific CE, dHidden, or dWeight per-block cycle budget, keeping future
      Tile kernel work tied to section attribution instead of aggregate timing
      alone. The SM120 native candidate wrapper forwards the matching
      `NFN_SM120_*_LM_HEAD_BACKWARD_MAX_TRUE_FUSED_*_CYCLES_PER_BLOCK` aliases
      into that focused preflight so known-slow strict sections fail before the
      full paired trainer benchmark starts.
    - 2026-06-28 reran the production-shape focused default 32x32 strict
      true-fused LM-head body at the current 28672-row trainer chunk after the
      latest CUDA 13.3.33/native defaults:
      `NFN_LM_HEAD_BACKWARD_PROFILE=trainer-chunk-true-fused
      NFN_LM_HEAD_BACKWARD_ALLOW_REJECTED_PROFILE=1
      NFN_LM_HEAD_BACKWARD_ITERATIONS=1 NFN_LM_HEAD_BACKWARD_WARMUP=0
      bash tools/bench_lm_head_backward_candidate.sh`. The route still proves
      `candidate_true_fused_capability=true` and
      `candidate_path_class=strict-true-fused-tile-kernel`, but remains rejected
      at `6.155991x` candidate/current-wrapper time,
      `22.242162x` candidate/reference-summed time, and
      `658.998471 ms` slower than the reference CE+dHidden+dWeight components.
      Do not promote the strict body or flip production readiness until this
      focused gate is near parity.
    - 2026-06-27 exposed the generic compiled GPT launcher through the Python
      SDK native-train path. `build_native_gpt_launcher_run_config()` resolves
      `NFN_NATIVE_GPT_TRAIN_CLI`, `build/nfn_train_gpt`, or installed
      `nfn-train-gpt` / `nfn-gpt-train` into a strict native command config, so
      SDK callers can spawn the generic compiled GPT trainer without Python
      launchers or graph-editor tensor flow while the remaining LM-head true
      fused kernel work continues.
    - 2026-06-23 fixed the diagnostic sequence wrapper to preserve the
      optimizer-only no-loss classifier path. The native trainer now passes a
      cooperative no-loss flag when `record_loss` is false, and the raw Tile ABI
      dispatches the existing BF16/u16 no-loss CE+dlogits kernel instead of
      requiring a row-loss buffer. Validation and explicit train-loss logging
      still use row-loss or loss-bin collection. This removes an artificial
      measurement penalty from `lm_head_cooperative_backward`, but the route is
      still diagnostic until the same-script full-loop gate passes. The linked
      trainer rerun on the dedicated RTX 5090 confirmed the candidate now
      reports `lm_head_ce_loss_backward_strategy:
      no-loss-dlogits-public-vocab-no-pad-zero-bf16-u16-targets` and
      `lm_head_classifier_ce_no_loss_enabled: true`, but rejected the wrapper
      at `1.117578x` train-loop wall and `1.294010x` LM-head backward.
    - 2026-06-23 added the native CUDA microbenchmark harness for this exact
      handoff: `bash tools/bench_lm_head_backward_candidate.sh` builds
      `build/lm_head_backward_bench`, loads
      `libnfn_native_train_tile_ops.so`, and compares
      `nfn_native_tile_lm_head_classifier_backward_cooperative_bf16_u16`
      against
      `nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16`
      inside one CUDA process with event timing and cooperative route counters.
      The JSON includes `candidate_true_fused_capability`, so a future symbol
      export alone is not enough to pass the strict fused-kernel evidence gate.
      The wrapper also has `NFN_LM_HEAD_BACKWARD_PROFILE=trainer-chunk`,
      `NFN_LM_HEAD_BACKWARD_PROFILE=trainer-chunk-strict`,
      `NFN_LM_HEAD_BACKWARD_PROFILE=trainer-row-loss`, and
      `NFN_LM_HEAD_BACKWARD_PROFILE=trainer-loss-bins` profiles plus
      `NFN_LM_HEAD_BACKWARD_REQUIRE_TRUE_FUSED=1` /
      `NFN_LM_HEAD_BACKWARD_MAX_RATIO=...` fail-fast gates. Set
      `NFN_LM_HEAD_BACKWARD_CANDIDATE_FIRST=1` to rerun close candidates in
      reverse order; JSON reports `run_order`. Set
      `NFN_LM_HEAD_BACKWARD_DRY_RUN=1` to print the resolved C++ benchmark argv
      without building artifacts or loading CUDA; the native no-Torch verifier
      covers that dry-run and scans `build/lm_head_backward_bench` when present.
      The wrapper defaults
      `NFN_LM_HEAD_BACKWARD_CUDA_VISIBLE_DEVICES=auto` so focused runs use the
      dedicated display-disabled NVIDIA GPU when `nvidia-smi` can select one.
      The JSON also reports `candidate_sequence_wrapper_only` and
      `candidate_strict_symbol_is_placeholder_sequence`, and the true-fused
      requirement gate now fails with a specific CE/dHidden/dWeight sequencing
      reason when the strict symbol is still only the placeholder wrapper.
    - 2026-06-24 added named
      `NFN_LM_HEAD_BACKWARD_PROFILE=trainer-chunk-cublaslt` and
      `trainer-row-loss-cublaslt` microbench profiles for the existing
      `nfn_native_tile_lm_head_classifier_backward_cooperative_cublaslt_bf16_u16`
      candidate. The dedicated RTX 5090 no-loss trainer-chunk checks rejected
      the cuBLASLt cooperative route: baseline-first measured `1.470042x`
      candidate/baseline time and candidate-first measured `1.463026x`, so this
      remains a diagnostic route rather than the fused/cooperative LM-head
      solution.
      The old row-loss profile reruns against the current wrapper-only strict
      symbol measured `trainer-row-loss` at the previously recorded
      `1.008311x` and `trainer-loss-bins` at `1.010302x`, and
      `NFN_LM_HEAD_BACKWARD_REQUIRE_TRUE_FUSED=1` failed with
      `candidate_true_fused_capability is false`; the wrapper remains rejected
      at the trainer chunk and loss-bin scales.
    - 2026-06-25 full-trainer CUDA 13.3.33 rerun kept the same cuBLASLt
      cooperative route rejected at training scale. The 3-step, 2-sample
      same-script profile changed the LM-head strategy to
      `diagnostic-cublaslt-sequence-wrapper-ce-dhidden-dweight-not-parity`, but
      regressed train-loop wall to `1.077251x`, steady-state CUDA-event timing
      to `1.083727x`, LM-head backward to `1.335573x`, and the cooperative
      LM-head substage to `1.477219x`. Do not spend more default-promotion work
      on this cuBLASLt wrapper; the needed work is still a real fused
      classifier-backward Tile kernel.
    - 2026-06-24 replaced the strict LM-head fused-kernel placeholder with a
      cached CUDA Graph body over the optimized CE, dHidden, and dWeight
      launches, then corrected the status contract so the graph wrapper no
      longer claims true fused parity. Current CUDA 13.3 builds expose the graph
      symbol and can run it through the non-required
      `NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1` diagnostic route, but
      `nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused()`
      returns `0`, `lm_head_cooperative_backward_kernel_available` remains
      false, and `--require-cooperative-lm-head-backward` still fails the strict
      preflight. The latest short RTX 5090 route-proof run reported
      `lm_head_cooperative_backward_cuda_graph_enabled: true`, graph replay
      success `32`, graph fallback `0`, and route-change gate passed; keep the
      graph body diagnostic-only and replace it with a lower-overhead
      fused/cooperative body before promotion.
    - 2026-06-24 added explicit strict LM-head CUDA Graph observability to the
      Tile C ABI and trainer JSON. Rebuilt Tile ops libraries now export graph
      capture/cache/replay/fallback counters, and native training JSON plus
      `tools/paired_kernel_speed.py` report them as
      `lm_head_fused_graph_capture_attempt_count`,
      `lm_head_fused_graph_capture_success_count`,
      `lm_head_fused_graph_cache_hit_count`,
      `lm_head_fused_graph_thread_cache_hit_count`,
      `lm_head_fused_graph_cache_entry_count`,
      `lm_head_fused_graph_replay_count`,
      `lm_head_fused_graph_replay_success_count`, and
      `lm_head_fused_graph_fallback_count`. This closes the evidence gap for
      graph-vs-fallback candidate runs; it does not close the parity gap or
      promote the CUDA Graph body as the default. The strict graph path no
      longer increments legacy `lm_head_cooperative_sequence_*` counters on
      successful replay, so those counters continue to identify only the
      fallback/diagnostic sequence wrapper. The thread-cache counter was added
      2026-06-25 to prove hot replays are skipping the mutex-protected graph
      cache scan after the small per-thread graph exec cache is warmed. The
      same-script RTX 5090 run moved
      `lm_head_fused_graph_thread_cache_hit_count` from `0` to `45`, passed
      route and ratio gates, and measured `0.980403x` train-loop wall time,
      `0.998423x` steady-state CUDA-event step time, `1.019998x` tokens/sec,
      and `0.999734x` LM-head backward; the standalone LM-head microbench
      reported candidate `graph_thread_cache_hit_count=5`,
      `graph_replay_success_count=5`, fallback `0`, and `0.983870x`
      candidate/baseline time. A fair no-stage llm.kittens parity rerun still
      failed the steady-state gate (`0.996810x` train-loop wall,
      `1.004028x` steady-state CUDA-event step time, `0.997129x` tokens/sec,
      80 LM-head graph replays and 77 thread-cache hits), so the next parity
      target remains GPU-side LM-head graph-body fusion or block-kernel work,
      not host-side graph cache lookup.
    - 2026-06-25 added and rejected
      `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_graph_serial_body`, which
      flips the strict LM-head CUDA Graph body from the default side-stream
      dHidden/dWeight schedule to serial caller-stream dHidden then dWeight
      through `NFN_NATIVE_GPT_LM_HEAD_GRAPH_BODY_SERIAL=1`. The route-proof run
      reported `lm_head_cooperative_backward_fused_kernel_abi_path_class:
      diagnostic-cuda-graph-wrapper-serial-body`, graph replay success, zero
      fallback, `torch_required=false`, and `graph_editor_tensor_flow=false`.
      The same-script 3-step, 2-sample stage-timed gate rejected default
      promotion because train-loop wall regressed to `1.005992x`,
      steady-state CUDA-event timing to `1.004767x`, LM-head backward to
      `1.022264x`, and the cooperative LM-head substage to `1.031743x`.
      Keep the default side-stream graph body; the next useful LM-head work is
      still a real fused classifier/dHidden/dWeight kernel, not serializing the
      current graph body.
    - 2026-06-25 added default `cudaGraphUpload` for each prewarmed LM-head
      CUDA Graph executable plus upload telemetry. Runtime JSON and
      `tools/paired_kernel_speed.py` now report
      `lm_head_fused_graph_upload_success_count` and
      `lm_head_fused_graph_upload_failure_count`; the inverse profile
      `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_graph_upload_off` sets
      `NFN_NATIVE_GPT_LM_HEAD_GRAPH_UPLOAD=0` so the default upload route can be
      kept only if same-script hot-stage gates do not favor the opt-out path.
      The CUDA 13.3.33 dedicated RTX 5090 3-step, 2-sample stage-timed gate
      proved the route change (`lm_head_fused_graph_upload_success_count:
      3 -> 0`) and rejected the opt-out path: train-loop wall `1.001492x`,
      steady-state CUDA-event timing `1.000055x`, LM-head backward
      `1.000583x`, cooperative LM-head `1.000593x`, block backward
      `1.002290x`, and MLP projection backward `1.008134x`. Keep graph upload
      default-on; this is still launch/setup hygiene, not the true fused
      LM-head classifier/dHidden/dWeight kernel. The opt-out profile is now
      marked rejected and requires
      `NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1` for intentional
      reruns.
    - 2026-06-25 updated the diagnostic graph-vs-sequence control after strict
      llm.kittens-parity became the default integrated LM-head route.
      `NFN_NATIVE_GPT_LM_HEAD_FORCE_SEQUENCE_WRAPPER_DIAGNOSTIC=1` plus
      `NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_CUDA_GRAPH=0` now forces the older
      sequence wrapper while keeping the cooperative route requested,
      runtime/plan JSON reports
      `lm_head_force_sequence_wrapper_diagnostic_enabled` and
      `lm_head_cooperative_backward_cuda_graph_requested`, and
      `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_cooperative_sequence_wrapper`
      compares the current strict parity route against sequence-wrapper
      execution in the same native-vs-native script. This is measurement
      infrastructure for the
      open true fused classifier/dHidden/dWeight kernel, not a default route
      promotion. A 2026-06-25 one-step rerun after the strict parity split
      passed the route-change gate: baseline used
      `strict-llmk-fused-classifier-native-matmul-backward`, candidate used
      `diagnostic-sequence-wrapper-ce-side-stream-dhidden-dweight-not-parity`,
      sequence counters rose from `0` to `16`, graph replay counters fell from
      `16` to `0`, and the diagnostic wrapper remained rejected at
      `1.023585x` train-loop wall and `0.976958x` train tokens/sec. The short
      3-step, 2-sample stage-timed probe passed route and
      timing gates, but the stronger 5-step, 3-sample confirmation rejected the
      sequence wrapper as a default at `1.003739x` train-loop wall,
      `1.001489x` steady-state CUDA-event timing, and `1.000401x` LM-head
      backward, so real reruns now require
      `NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1`.
    - 2026-06-24 fixed the cooperative/strict LM-head backward ABI to use the
      normal aligned padded-vocab GEMM launchers for dHidden and dWeight. The
      previous graph and sequence bodies used the public-vocab strided
      dHidden/dWeight launchers whenever `row_stride > vocab`, which meant
      `lm_head_cooperative_backward` was also benchmarking the rejected
      `NFN_NATIVE_GPT_LM_HEAD_PUBLIC_VOCAB_STRIDED_GEMM=1` route. Rebuild Tile
      ops and rerun the focused trainer-chunk microbench plus full
      `lm_head_cooperative_backward` profile before deciding whether the graph
      body still needs replacement with a monolithic kernel.
      The first rebuilt RTX 5090 focused trainer-chunk run reported strict
      graph replay at `1.000788x` versus the legacy cooperative baseline with
      zero fallbacks. A short 2-step, 1-sample full-loop profile then proved
      graph replay was active (`lm_head_fused_graph_replay_success_count=32`)
      and strided public-vocab counters stayed at zero; train-loop wall
      improved to `0.976058x`, but promotion still failed at `1.000188x`
      steady-state CUDA-event timing and `1.002085x` LM-head backward. A
      stronger 3-step, 2-sample profile kept the route diagnostic-only:
      train-loop wall improved to `0.990440x` and graph replay succeeded 48
      times, but steady-state CUDA-event timing regressed to `1.002035x` and
      LM-head backward to `1.001066x`.
      A later CUDA 13.3 dedicated RTX 5090 rerun on 2026-06-24 used
      `NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1
      NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_cooperative_backward
      NFN_SM120_NATIVE_STEPS=3 NFN_SM120_NATIVE_SAMPLES=2
      NFN_SM120_NATIVE_WARMUP=1 NFN_SM120_NATIVE_STAGE_TIMING=1` and again
      rejected promotion: the cooperative wrapper route counters changed
      (`lm_head_cooperative_sequence_*: 0 -> 48`) and
      `stage.lm_head_backward.total_ms` improved to `0.999819x`, but
      `train_loop_wall_ms_per_step` regressed to `1.002204x`,
      steady-state CUDA-event step time regressed to `1.000412x`, and
      `stage.block_backward.mlp_proj.total_ms` regressed to `1.002147x`.
      Keep the graph/sequence route rejected and replace it with an actual
      lower-overhead fused classifier/dHidden/dWeight body before any default
      promotion.
    - 2026-06-23 changed the focused `trainer-chunk` microbenchmark profile to
      pass the cooperative no-loss flag and use the no-loss CE reference
      symbol, matching the optimizer-only native trainer path. Use
      `trainer-row-loss` when intentionally reproducing the older row-loss
      comparison. The first updated RTX 5090 run reported `no_loss: true`,
      `flags: 2`, and no-loss CE reference timing at about `6.49 ms` for the
      49152-row chunk; `candidate_true_fused_capability` remains false, so this
      is benchmark coverage rather than a fused-kernel promotion.
    - 2026-06-22 prerequisite: the native trainer no longer treats
      `nfn_native_tile_lm_head_classifier_backward_cooperative_bf16_u16` as an
      untyped `int (*)()` probe. The function pointer now encodes the required
      BF16 logit/dlogit chunk, u16 targets, optional row losses, BF16/float
      hidden inputs, BF16/float token weights, dHidden, dWeight, shape metadata,
      loss scale, dWeight beta, flags, and stream. The route remains explicitly
      unintegrated until a Tile implementation with that contract is called.
    - 2026-06-22 tightened the strict fused-callable handoff: the trainer now
      loads `nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16`
      into its own typed function pointer and calls it only when the strict
      fused route is enabled. The older
      `nfn_native_tile_lm_head_classifier_backward_cooperative_fused_bf16_u16`
      wrapper remains a non-required diagnostic sequence route, so wrapper-only
      builds cannot accidentally satisfy
      `--require-cooperative-lm-head-backward`.
    - 2026-06-22 exported the strict Tile ABI symbol
      `nfn_native_tile_lm_head_classifier_backward_cooperative_fused_bf16_u16`
      and integrated it behind `NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1`
      / `--require-cooperative-lm-head-backward`. The implementation still
      sequences the existing row-loss classifier, BF16 dHidden, and BF16 dWeight
      launches, and runtime strategy strings explicitly say
      `strict-cooperative-abi-sequences-existing-ce-dhidden-dweight-kernels-not-yet-parity`.
	      A dedicated RTX 5090 1-step, 3-sample gate measured `0.999224x` train-loop
	      wall time and `0.997052x` block backward, but the route still failed the
	      strict total LM-head gate at `1.000739x`. Keep it
	      default-off; the open work remains replacing that sequenced body with a
	      genuinely fused/cooperative kernel under the now-concrete strict symbol.
	    - 2026-06-22 changed the strict fused cooperative export to use persistent
	      non-blocking CUDA streams plus events: CE still runs first on the caller
	      stream, then dHidden and dWeight are queued on side streams, and the
	      caller stream waits on both completion events. Runtime JSON now reports
	      `strict-cooperative-abi-event-ordered-ce-side-stream-dhidden-dweight-diagnostic-not-yet-parity`
	      or the matching loss-bin variant. The CUDA 13.3 dedicated RTX 5090
	      2-step, 2-sample same-script gate rejected this schedule at `1.003537x`
	      train-loop wall, `1.000399x` LM-head backward, and `1.003647x` block
	      backward versus the normal native baseline. Keep it diagnostic-only; the
	      open work is still a genuinely fused/cooperative LM-head kernel body.
	    - 2026-06-23 confirmed the strict
	      `nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16`
	      symbol is only a placeholder until
	      `nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused()`
	      returns nonzero. Current builds return `0`, so
	      `lm_head_cooperative_backward_kernel_available` remains false and the
	      required profile is a strict ABI preflight probe. The open work is now a
	      true fused/single-kernel body under this strict symbol, not another
	      wrapper or preflight integration step.
	    - 2026-06-23 added public-vocab strided LM-head dHidden/dWeight Tile ABI
	      diagnostics:
	      `nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_strided_float32`
	      and
	      `nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta`.
	      `NFN_NATIVE_GPT_LM_HEAD_PUBLIC_VOCAB_STRIDED_GEMM=1` routes padded
	      BF16 dlogit chunks through logical-vocab GEMMs while preserving the
	      padded row stride, and runtime JSON reports
	      `lm_head_dhidden_strided_vocab_gemm_count` plus
	      `lm_head_dweight_strided_vocab_gemm_count`. The same-binary CUDA 13.3
	      dedicated RTX 5090 gate rejected it as a default:
	      `1.117352x` train-loop wall and `0.895573x` tokens/sec versus the
	      current aligned padded-vocab route. Keep it diagnostic-only; GPT-2's
	      current padding is only `50257 -> 50304`, so the old aligned K wins.
	    - 2026-06-22 added cooperative sequence launch counters to the diagnostic
	      wrapper and paired benchmark extraction. Runtime JSON now reports
	      `lm_head_cooperative_sequence_launch_count`,
	      `lm_head_cooperative_sequence_ce_launch_count`,
	      `lm_head_cooperative_sequence_dhidden_launch_count`,
	      `lm_head_cooperative_sequence_dweight_launch_count`,
	      `lm_head_cooperative_sequence_concurrent_count`,
	      `lm_head_cooperative_sequence_legacy_count`, and
	      `lm_head_cooperative_sequence_loss_bin_count`, so future fused-kernel
	      candidates can prove they are no longer just sequencing the old
	      CE/dHidden/dWeight launches.
	    - 2026-06-25 hardened external-reference promotion checks for native GPT
	      kernel candidates. When `NFN_SM120_NATIVE_INCLUDE_LLMK_REFERENCE=1`
	      is set and the candidate changes the native binary, Tile library,
	      build flags, env, or extra args, the wrapper now adds default
	      candidate-over-llm.kittens gates for train-loop wall time,
	      steady-state CUDA-event wall time on multi-step runs, and tokens/sec.
	      This keeps benchmark evidence from promoting a candidate that only
	      beats the previous NeuralFn route while still missing the llm.kittens
	      SM120 reference in the same selected-GPU window.
	    - 2026-06-22 added `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_full_resident_reuse`
	      as the reproducible wrapper for the current full-resident logits/full-batch
	      LM-head reuse diagnostic. It proves why llm.kittens-style resident logits
	      cannot be adopted directly in the current NeuralFn saved-activation layout:
	      the one-step dedicated RTX 5090 paired run improved LM-head backward to
	      `0.705502x`, but train-loop wall regressed to `21.830567x` and block
	      backward to `44.496727x` after the 6.59 GB resident-logit arena pushed
	      attention backward over the memory cliff. Keep this rejected/default-off;
	      the required fix remains a cooperative LM-head backward kernel that avoids
	      both row-chunk recompute and full resident logits.
	    - 2026-06-22 added the opt-in cooperative loss-bin diagnostic
	      `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_cooperative_loss_bins`.
	      The strict cooperative ABI `flags` field now selects the existing
	      loss-bin classifier launcher before the sequenced dHidden/dWeight work,
	      and the benchmark profile applies `--train-loss-every-steps 1` to both
	      baseline and candidate so the loss-bin route is actually exercised. The
	      CUDA 13.3 dedicated RTX 5090 3-step, 2-sample gate requested the route,
	      but changed no tracked route counter, strategy value, linear shape stat,
	      or cuBLASLt plan entry; the apparent `0.993532x` train-loop timing delta
	      is noise. Keep it rejected/default-off; it is not the fused/cooperative
	      kernel body needed for parity.
		  - 2026-06-20 promoted the row-loss reduction classifier variant to the dense
    GPT default after CUDA 13.3.33 RTX 5090 same-script gating. The new
    `nfn_native_tile_lm_head_classifier_backward_row_losses_inplace_strided_no_pad_zero_bf16_bits_u16_targets`
    ABI writes one loss per classifier row, then the trainer reduces those rows
    on device instead of atomic-adding the scalar loss from each row block. The
    3-step, 3-sample native-vs-native run measured `0.994971x` train-loop wall
    time per step and `0.993379x` LM-head backward time versus the prior default.
    Keep `NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_REDUCTION=0` only as the rollback
    bisection path.
  - 2026-06-20 added and initially rejected-as-default the narrower
    `nfn_native_tile_sum_accumulate_float32` row-loss tail. It replaces the
    generic `sum_partials` plus scalar `gradient_accumulate` launches with one
    CUDA reduction/atomic-accumulate launch per LM-head row chunk, but the
    3-step, 3-sample native-vs-native RTX 5090 gate measured `1.008595x`
    train-loop wall time and `1.015517x` LM-head CE time versus the current
    row-loss default. Superseded on 2026-06-22 by the current CUDA 13.3
    recheck/promotion below.
  - 2026-06-22 added and rejected-as-default the loss-bin train-loss logging
    diagnostic. `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_loss_bins` expands
    to `NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=1`; the new
    `nfn_native_tile_lm_head_classifier_backward_loss_bins_inplace_strided_no_pad_zero_bf16_bits_u16_targets`
    ABI accumulates row losses into a fixed bin workspace before a
    `sum_accumulate` tail. The normal no-loss parity benchmark does not execute
    loss accumulation, and the one-step `--train-loss-every-steps 1` RTX 5090
    check proved the route active (`lm_head_classifier_loss_bin_launch_count:
    0 -> 16`) but rejected it at `1.002592x` train-loop wall, `1.000254x`
    LM-head backward, and `1.007361x` block backward. Keep it diagnostic-only;
    the next LM-head parity route still needs a fused/cooperative
    classifier-backward kernel, not another train-loss tail tweak.
  - 2026-06-20 accepted the BF16/u16 row-loss target-logit prefetch barrier
    removal after the CUDA 13.3.33 rebuild. The fused row-loss+dlogits kernel
    now reads the target logit before in-place dlogit writes and no longer needs
    the post-loss `__syncthreads()` before the gradient-store pass. Correctness
    passed the native GPT `--smoke-lm-step` check and
    `NFN_TILE_CUDA_TEST=1 CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_tile_cuda_gpu.py -q`.
    The 3-step, 5-sample native-vs-native RTX 5090 gate passed against the saved
    rebuilt baseline: train-loop wall `0.988618x`, LM-head backward `0.987088x`,
    CE `0.987688x`, block backward `0.990160x`, and tokens/sec `1.011616x`.
  - 2026-06-20 rejected the follow-up shared-target-logit variant. Caching
    `target` and `target_logit` through shared memory before the existing
    max/sum reductions passed correctness, but the same 3-step, 5-sample
    native-vs-native gate measured train-loop wall `1.001780x`,
    LM-head backward `1.000192x`, block backward `1.003660x`, and CE only
    `0.999485x`. The source was reverted; keep the simpler per-thread target
    read from the accepted barrier-removal kernel.
- [x] Add GPT-2 evo `--print-plan` compiled C++ preflight that reports the AdamW/NVFP4/evo-layer schedule and required candidate-evaluation kernels without Python/Torch.
- [x] Add GPT-2 compiled-CLI SDK handoff config that passes cached dataset alias/path directly to the C++ shard resolver without Python `meta.json` or token-shard validation.
- [x] Add GPT-2 native `--backend tile-cuda` / SDK `kernel_backend="tile-cuda"` preflight that reports required raw Tile ABI symbols and `--check-tile-ops` validation without Python/Torch.
- [x] Default all native Tile-CUDA entrypoints and SDK bindings to `CUDA_MODULE_LOADING=LAZY` when unset, matching the dense GPT trainer before command execution or Tile library/runtime loading.
  - 2026-06-19 extended this startup policy to the master `nfn train` direct native dispatcher as well. The no-Torch verifier now covers both explicit `nfn train --tinystories` and default `nfn train` dry-run/print-command paths under the import blocker, so the default CLI route is checked before `train_gpt_native`, `nfn_impl`, Torch, NumPy, tiktoken, or dataset-manager imports can occur.
  - 2026-06-19 extended this to the direct `cli/scripts/train_gpt.py` / `train_gpt2.py` compiled-CLI fast path so compatibility script execution also forwards lazy CUDA module loading before the compiled native trainer starts, without importing Torch.
  - 2026-06-19 extended the same default to `native_training_guard.py`, covering GPT-2 evo and other legacy training-script native handoffs that exec compiled trainers before Torch imports.
- [x] Keep the combined dense GPT device arena opt-in only. `NFN_NATIVE_GPT_COMBINED_DEVICE_ARENA` defaults to `0`, so the trainer uses one float arena and one BF16/uint16 arena by default. `NFN_SM120_NATIVE_CANDIDATE_PROFILE=combined_device_arena` still forces baseline `NFN_NATIVE_GPT_COMBINED_DEVICE_ARENA=0` and candidate `NFN_NATIVE_GPT_COMBINED_DEVICE_ARENA=1` for same-script bisection. After the CUDA 13.3 reinstall, the route was briefly promoted from a passing 3-step gate, then rechecked and rejected: the current dedicated RTX 5090 rerun measured `1.004991x` train-loop wall time, `0.995098x` tokens/sec, and `1.063067x` setup wall time. Set `NFN_NATIVE_GPT_COMBINED_DEVICE_ARENA=1` only for explicit combined-arena regression checks.
  - 2026-06-26 rejected two additional startup allocation probes on the
    dedicated RTX 5090 before committing any runtime change. Moving stored
    dense-attention LSE sidecars into the float arena did not affect the default
    packed-attention route and failed the startup gate at `1.020346x`
    `setup_wall_ms`; forcing split attention still regressed to `1.019853x`
    setup wall. Moving default direct-u16 token/target staging into the main
    uint16 arena also regressed setup to `1.031604x` and left the route-change
    gate unproven. Keep token staging in its standalone token device arena and
    leave dense-attention LSE allocation alone unless a future measured route
    proves otherwise.
- [x] Add the fused padded-vocab token-weight BF16-shadow init ABI without making it the default. `nfn_native_tile_init_gpt2_token_weight_fast_with_bf16_shadow_padded_float32` can initialize the public token rows and zero the padded-vocab tail plus BF16 shadow in one launch, and the native trainer exposes it behind `NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_PADDED_INIT=1`. The padded kernel also has a precomputed deterministic BF16-pattern variant behind `NFN_NATIVE_GPT_TOKEN_WEIGHT_PADDED_BF16_PATTERN=1`, but it remains rejected. A 2026-06-28 3-step, 2-sample rerun improved `setup.token_weight_init.total_ms` to `0.953426x` while missing candidate-over-reference first-step CUDA-event timing at `1.000443x`; the canonical 3-step, 3-sample, 1-warmup rerun passed candidate/reference throughput and first-step gates but failed the target token-init gate at `1.017866x` despite `setup_wall_ms=0.992765x`. Keep the BF16-pattern route opt-in until the token-init and candidate/reference gates both pass in the same script.
- [x] Keep 512-thread BF16 CE rows rejected as a dense GPT default.
  `NFN_SM120_NATIVE_CANDIDATE_PROFILE=ce_bf16_threads_512` now reproduces
  `NFN_NATIVE_GPT_CE_BF16_THREADS=512`, but the 2026-06-23 dedicated RTX 5090
  stage-timed rerun changed `lm_head_ce_bf16_threads_per_row` from `1024` to
  `512` and regressed train-loop wall to `1.012086x`, LM-head backward to
  `1.051608x`, and LM-head CE to `1.430612x` versus the 1024-thread default.
  Real reruns now require
  `NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1`; dry-run plan expansion
  remains available.
  - 2026-06-22 added `nfn_native_tile_token_cross_entropy_bf16_threads_per_row` plus dense GPT JSON/paired-benchmark reporting as `lm_head_ce_bf16_threads_per_row`, so future CE launch-shape bisections prove the resolved Tile-kernel row-block size instead of rechecking the existing 1024-thread default by timing alone.
- [x] Keep the 32768-row LM-head dHidden BF16 GEMMEx FAST_16BF compute-type probe rejected as a dense GPT default. `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_dhidden_fast16bf_32768` expands to `NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF_SHAPE=768,32768,50304,N,N`, and runtime JSON now reports `linear_bf16_gemm_fast16bf_request_count` so the override is visible. The 2026-06-22 dedicated RTX 5090 same-script gate confirmed the counter moved from `0` to `32`, but the strict stage gate failed because `stage.block_backward.total_ms` regressed to `1.001305x`; leave the route diagnostic-only.
- [x] Run GPT-2 no-data Tile-CUDA preflights before token-shard resolution so `--check-tile-ops`, synthetic smoke steps, and ABI checks work without cached datasets and report `token_shards_resolved: false`.
  - 2026-06-19 extended the same startup rule to unsupported selected-graph/template training exits: real `--train-transformer-lm` runs now reject unsupported templates or custom graphs before token-shard resolution and report `token_shards_resolved: false`.
- [x] Add GPT-2 native Tile parameter layout and forward/backward/optimizer stage plan to the compiled `--backend tile-cuda --print-plan` JSON.
- [x] Add GPT-2 `--smoke-optimizer-step` compiled C++ path that allocates GPT-2-sized param/grad/AdamW buffers, runs one raw Tile AdamW call per registered parameter buffer, and samples copyback values without Python/Torch.
- [x] Add GPT-2 `--smoke-lm-step` compiled C++ path that runs tied token embedding, full-vocab LM logits, CE partials/backward, tied weight backward, and AdamW through raw Tile kernels without Python/Torch.
- [x] Add GPT-2 `--smoke-embedding-lm-step` compiled C++ path that samples cached uint16 tokens and runs token/position embeddings, embedding residual add, final LayerNorm, tied LM head, CE backward, embedding/norm backward, and AdamW through raw Tile kernels without Python/Torch or graph-editor payloads.
- [x] Add GPT-2 `--train-embedding-lm` compiled C++ path that runs the sampled token/position embedding, final LayerNorm, tied LM head, CE backward, embedding/norm backward, AdamW, and periodic validation loop over cached shards without Python/Torch or graph-editor payloads.
- [x] Add GPT-2 `--smoke-attention-step` compiled C++ path that runs qkv projection, QKV split, scaled dot-product attention forward/backward, QKV gradient merge, projection backward, and AdamW through raw Tile kernels without Python/Torch.
- [x] Add GPT-2 `--smoke-mlp-step` compiled C++ path that runs c_fc projection, GELU forward/backward, c_proj backward, and AdamW through raw Tile kernels without Python/Torch.
- [x] Add GPT-2 `--smoke-norm-residual-step` compiled C++ path that runs LayerNorm, scaled residual add, LayerNorm backward, gradient accumulation, and AdamW through raw Tile kernels without Python/Torch.
- [x] Add GPT-2 `--smoke-transformer-block-step` compiled C++ path that composes LayerNorm, fused QKV attention, real 12-head reshape/merge layout, residual adds, MLP, backward passes, gradient accumulation, projection bias gradients, and AdamW updates for all 12 GPT-2 block parameter buffers through raw Tile kernels without Python/Torch.
- [x] Add GPT-2 `--smoke-transformer-lm-step` compiled C++ path that samples cached uint16 tokens, preserves range-checked GPT-2 token IDs, and runs token/position embeddings, one tiny transformer block, final LayerNorm, tied LM head, CE forward/backward, transformer backward, embedding backward, and AdamW for 16 parameter buffers through raw Tile kernels without Python/Torch.
- [x] Implement GPT-2 `--train-transformer-lm` as an initial full-vocab real-dim single-layer multi-step compiled C++ loop over cached shards using token/position embeddings, one transformer block, final norm, a row-chunked tied LM-head/CE workspace, transformer backward, embedding backward, periodic validation, device-side global norm gradient clipping, and 16 AdamW parameter updates without Python/Torch.
- [x] Report GPT-2 `--train-transformer-lm` `trained_layers` / `target_layers` in JSON so trained depth is directly testable.
- [x] Report GPT-2 `--train-transformer-lm` `block_state_layout` in JSON and store trained block weights/gradients/AdamW state behind an explicit per-block C++ structure as the first step toward a 12-block array.
- [x] Drive GPT-2 `--train-transformer-lm` block parameter allocation, initialization, gradient zeroing, gradient clipping, AdamW updates, and trained-weight checkpoint export from the per-block C++ state vector instead of direct `block0` optimizer wiring/export.
- [x] Drive GPT-2 `--train-transformer-lm` block activation storage plus forward/backward execution through a per-block activation tape and block loops instead of a single inline block body.
- [x] Raise GPT-2 `--train-transformer-lm` trained block count from 1 to 12 using a scratch-recompute activation tape plus persistent block outputs instead of allocating a full tape per layer.
- [x] Add GPT-2 `--checkpoint-metadata-smoke` compiled C++ path that writes a sparse version-5 bf16 native checkpoint-format file plus `DONE_########` marker for the requested `--num-layers` target shape without Python/Torch/CUDA.
- [x] Add GPT-2 `--train-transformer-lm` final trained-weight checkpoint export in native version-5 bf16 `.bin` format with `DONE_########` marker and JSON file-size accounting.
- [x] Default dense GPT-2 Python and SDK compiled-CLI handoff to `kernel_backend="tile-cuda"` plus `--train-transformer-lm`; the explicit `llm-kittens` training backend has since been removed from CLI/SDK/C++ trainer dispatch.
- [x] Keep GPT-2 wrapper `--native-cuda-dry-run --native-cuda-print-command` metadata-only on the default compiled CLI runner, with no dataset-manager/NumPy/tiktoken/Torch imports and no raw-text shard materialization.
- [x] Remove the CLI training `NFN_ALLOW_TORCH_TRAINING=1` bypass: `nfn train` and direct `cli/scripts/train_*.py` execution now dispatch to compiled native CUDA/C++ entrypoints or fail before importing Torch, while legacy graph-backed experiments must call Python SDK trainer APIs directly.
- [x] Keep the public CLI parser/help aligned with the enforced native backend:
  `nfn train`, `nfn infer`, and `nfn eval` now accept and advertise
  `--kernel-backend tile-cuda` instead of `{auto,torch,tile-cuda}`. This keeps
  the default workstation training surface from suggesting Torch fallback paths
  that the native dispatcher already rejects.
- [x] Preserve explicit zero cadences from SDK/native GPT compiled-CLI configs (`eval_every_steps=0`, `sample_every_steps=0`, and `checkpoint_every_steps=0`) so same-script kernel benchmarks can disable validation, sampling, and checkpoint cadence without the Python handoff clamping them back on.
- [x] Pin the SM120 parity wrapper's NeuralFn candidate to `--train-batch-tokens 524288`, matching the `llm.kittens/train-sm120.sh` `-d 524288` contract in the same paired script instead of relying on native defaults.
- [x] Add `tools/bench_native_gpt_sm120_candidate.sh` as the native-vs-native SM120 bisection wrapper: it keeps the dense GPT command shape fixed on both sides, compares the current Tile ops library/default env against `NFN_SM120_NATIVE_CANDIDATE_ENV` or `NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB`, preserves the `524288` token-batch contract, and reuses the selected-GPU idle/utilization guards from `tools/paired_kernel_speed.py`.
  - 2026-06-22 added named profiles for remaining raw-env diagnostics so every
    native scheduling/attention candidate can be measured through the same
    guarded wrapper instead of ad hoc shell env strings:
    `bf16_attention_grad_out`, `bf16_attention_dprep_grad_out`,
    `mlp_proj_dinput_before_dweight`, `mlp_fc_dinput_before_dweight`,
    `attn_proj_dinput_before_dweight`, and
    `lm_head_fused_loss_backward_off`. Stage-timed runs now attach the matching
    attention, ordering, or LM-head CE gates automatically.
  - 2026-06-23 added execution counters for the three block-backward ordering
    profiles:
    `block_backward_mlp_proj_dinput_before_dweight_count`,
    `block_backward_mlp_fc_dinput_before_dweight_count`, and
    `block_backward_attn_proj_dinput_before_dweight_count`. The paired speed
    tool now includes them in `native_route_counter_changes`, so these
    scheduling probes are rejected or promoted with route proof instead of
    failing route detection.
  - 2026-06-24 rechecked and kept the attention projection
    dInput-before-dWeight route rejected after the rebuilt-binary confirmation.
    A shorter 3-step, 2-sample run looked favorable, but the required stronger
    5-step, 3-sample same-script gate moved
    `block_backward_attn_proj_dinput_before_dweight_count: 0 -> 480` and then
    failed strict timing gates: train-loop wall `1.001501x`, LM-head backward
    `1.000290x`, block backward `1.003886x`, MLP projection backward
    `1.002417x`, and attention projection backward `1.081569x`.
  - 2026-06-18 added short `NFN_SM120_CANDIDATE_*` aliases for the native-vs-native wrapper controls so ad hoc candidate benchmarks do not silently fall back to default steps/samples/profile settings when the shorter names are used.
  - 2026-06-19 added `native_route_counter_changes` to the paired benchmark JSON/text report so candidate timings are checked against tracked TK/cuBLASLt/BF16/LM-head/attention route counters before being treated as kernel evidence.
  - 2026-06-24 tightened the required route-change gate so setup-only/prewarm
    counters cannot validate a throughput candidate by themselves. Paired JSON
    now splits route evidence into `has_hot_route_counter_change`,
    `hot_changed`, and `setup_only_changed`; cuBLAS handle prewarm, BF16
    workspace prewarm, and device-arena setup deltas remain visible but require
    a hot route counter, strategy value, linear-shape row, or cuBLASLt plan-cache
    change before `--require-native-route-change` passes.
  - 2026-06-22 added `native_cublaslt_plan_cache` to the paired benchmark
    JSON/text report. Normal no-shape-stats runs now show cached cuBLASLt
    shape, selected heuristic, returned heuristic count, workspace, and epilogue
    changes, and plan-cache changes suppress the timing-only warning even when
    aggregate route counters are unchanged.
  - 2026-06-20 added the LM-head classifier row-chunk counters to paired native metric summaries and `native_route_counter_changes`, covering `lm_head_classifier_chunk_launch_count` plus the last rows/vocab/stride so classifier-route candidates are visible without opening sidecar JSON.
  - 2026-06-18 corrected the short extra-args aliases: `NFN_SM120_CANDIDATE_EXTRA_ARGS` now maps to candidate-only CLI flags, while `NFN_SM120_COMMON_EXTRA_ARGS` maps to shared baseline-and-candidate flags. The dry-run regression test verifies `--lm-head-row-chunk-size 32768` appears only in the candidate command.
  - 2026-06-19 added `NFN_SM120_PARITY_*` as a third alias family for shared native-vs-native candidate controls such as steps, samples, warmup, GPU selection, profile directory, stage timing, JSON output, and dry-run plan. Precedence is native-specific, then candidate-short, then parity, then defaults, so quick parity-to-candidate bisections do not silently expand to the candidate wrapper's default 10-step, 3-sample, 1-warmup run. Candidate-only env and candidate-only extra args remain isolated to the candidate command.
  - 2026-06-19 added shared env aliases `NFN_SM120_NATIVE_ENV`, `NFN_SM120_COMMON_ENV`, and `NFN_SM120_PARITY_ENV`; each key-value pair is passed to both baseline and candidate commands. Candidate-only env remains isolated through `NFN_SM120_NATIVE_CANDIDATE_ENV` / `NFN_SM120_CANDIDATE_ENV`. The dry-run regression test covers the common profiling case `NFN_NATIVE_GPT_LINEAR_SHAPE_STATS=1` without requiring an invalid native CLI flag.
  - 2026-06-20 added generic `NFN_SM120_*` fallback aliases to both SM120 benchmark wrappers for shared controls such as steps, samples, warmup, GPU selection, max GPU utilization, profile directory, JSON output, and dry-run plan. Wrapper-specific `NFN_SM120_NATIVE_*`, `NFN_SM120_CANDIDATE_*`, and `NFN_SM120_PARITY_*` names still win, but copied parity/candidate commands no longer silently fall back to long/default runs when using the generic names.
  - 2026-06-20 fixed the native-vs-native candidate wrapper to support compiled trainer executable candidates through `NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN` or `NFN_SM120_CANDIDATE_TRAIN_BIN`, with `NFN_NATIVE_GPT_TRAIN_BIN` kept as the baseline. This closes the saved-binary bisection gap where a candidate trainer path could be ignored and the wrapper would silently compare the baseline executable against itself.
  - 2026-06-20 decoupled stage timing from profile sidecar generation in both SM120 wrappers. `NFN_SM120_STAGE_TIMING=1` and the wrapper-specific aliases now pass `--native-stage-timing` even when `NFN_SM120_PROFILE_DIR=none`, so hot-stage gates can run without writing per-sample native profile files.
  - 2026-06-22 fixed native-vs-native attention candidate gates so stage-timed attention profiles automatically pass `NFN_NATIVE_GPT_ATTENTION_BACKWARD_SECTION_TIMING=1` to both baseline and candidate when they add `attention_backward_tk_timing_us` or `attention_backward_dprep_timing_us` ratio checks.
  - 2026-06-24 added `fused_ln2_bf16_out_off` as an inverse regression guard for the default fused attention-residual-LN2 BF16 output handoff. `tools/paired_kernel_speed.py` now treats `fused_ln2_bf16_out_enabled`, `fused_ln2_bf16_norm_float_store_elision_enabled`, `stored_mlp_ln2_bf16_prepack_strategy`, `stored_mlp_forward_strategy`, and `attention_residual_ln2_strategy` as route evidence. The dedicated RTX 5090 2-step, 2-sample stage-timed gate proved the strategy change but rejected the separate-store rollback at `1.020138x` train-loop wall, `1.013718x` steady-state CUDA-event wall, and `1.119485x` `stage.block_forward.mlp_fc_gelu.total_ms`, so `NFN_NATIVE_GPT_FUSE_LN2_BF16_OUT=1` remains the normal path.
  - 2026-06-24 added `mlp_residual_next_ln1_off` as the matching inverse regression guard for the default MLP projection residual -> next-block LN1 fusion. The paired extractor now reads `block_state_layout.mlp_residual_next_ln1_fusion_*` route fields. A first run exposed the old wrong `memory_strategy` path and failed the route gate; after correction, the dedicated RTX 5090 2-step, 2-sample rerun proved `block_state_layout.mlp_residual_next_ln1_fusion_count: 176 -> 0` but rejected the rollback at `1.000520x` train-loop wall, `1.004202x` steady-state CUDA-event wall, and `0.999479x` tokens/sec.
- [x] Fix GPT-2 evo no-Torch command inspection so startup dry-runs expose the real compiled delegate. `native_training_guard.py`, `nfn train`, `train_gpt.py`, and the compiled `nfn_native_train` frontend now normalize `--native-cuda-dry-run`, `--native-cuda-print-command`, and the high-level `--native-cuda-startup-only` alias, `nfn-native-train --base-model gpt2-evo --dry-run --print-command` preserves `--print-command` in the printed family command, and `nfn_gpt2_evo_native_train --native-cuda-dry-run --native-cuda-print-command --native-cuda-startup-only` prints the final dense GPT CUDA Tile delegate with `--train-transformer-lm --layer-evo --startup-only` before token-shard resolution, dataset scanning, Torch imports, or graph-editor tensor flow. The dense native GPT C++ parser also accepts `--native-cuda-startup-only` for direct startup-only calls.
- [x] Revisit the CUDA 13.3 dedicated RTX 5090 parity baseline before retrying older candidates. The 2026-06-19 5-step same-script run measured llm.kittens at `2429.056 ms/step` and NeuralFn at `2536.100 ms/step`, or `1.044068x` train-loop wall time and `0.957209x` tokens/sec. The hottest remaining buckets were block backward (`6365.680 ms` over 5 steps) and LM-head backward (`3127.700 ms`), so parity work still needs new fused/cooperative kernel work rather than parameter-default retunes.
  - A no-stage-timing rerun after the CUDA 13.3 WSL reinstall measured llm.kittens at `2471.224 ms/step` and NeuralFn at `2541.080 ms/step`, or `1.028292x` train-loop wall time and `0.972465x` tokens/sec. The instrumentation-free parity gap is therefore smaller than the stage-attributed run, but still not closed.
  - 2026-06-20 refreshed the no-stage parity baseline after the CUDA-visible test rerun and candidate-wrapper trainer-binary fix: `NFN_SM120_PARITY_STEPS=5 NFN_SM120_PARITY_SAMPLES=3 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_PROFILE_DIR=none bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2487.459333 ms/step` and NeuralFn at `2517.273333 ms/step`, or `1.012089x` train-loop wall time and `0.987647x` tokens/sec. The selected RTX 5090 had zero compute processes before every sample, so the remaining gap is still real but much narrower than the earlier stage-timed run.
  - 2026-06-22 refreshed a short 2-step, 1-sample stage-timed parity sample
    after the strict fused ABI handoff fix:
    `NFN_SM120_PARITY_STEPS=2 NFN_SM120_PARITY_SAMPLES=1
    NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_STAGE_TIMING=1
    NFN_SM120_PARITY_PROFILE_DIR=/tmp/nfn_parity_continue_20260622_2step
    NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_parity_continue_20260622_2step.json
    NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=30 bash
    tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at
    `2539.325 ms/step` and NeuralFn at `2610.800 ms/step`, or `1.028147x`
    train-loop wall and `0.970733x` tokens/sec. The selected RTX 5090 had zero
    compute processes and 0% utilization before and after the sample. The
    hottest current buckets remain `stage.block_backward.total_ms=2622.810`,
    `stage.lm_head_backward.total_ms=1166.160`, `stage.train.model_forward.total_ms=1378.810`,
    `stage.block_backward.mlp_proj.total_ms=654.517`,
    `stage.block_backward.attn_sdpa.total_ms=529.490`, and
    `stage.block_backward.mlp_fc.total_ms=521.723` over 2 steps.
  - 2026-06-24 kept `NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1` rejected/default-off after the current-tree dedicated RTX 5090 rebuilt verification. The same-script 3-step, 2-sample, 1-warmup stage-timed gate compared the BF16 grad-out route against `NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=0`, proved the route change by moving 288 block dInput GEMMs from cuBLASLt to BF16, and improved steady-state CUDA-event step time to `0.997360x`, attention SDPA to `0.978512x`, attention to-QKV to `0.978520x`, and dprep timing to `0.801362x`, but rejected default promotion because train-loop wall regressed to `1.006042x`, block backward to `1.017695x`, and MLP projection backward to `1.004116x`. `NFN_NATIVE_GPT_BF16_ATTENTION_DPREP_GRAD_OUT=1` remains rejected after the reinstall because it changed no tracked GEMM route counters and regressed train-loop wall to `1.007955x`, block backward to `1.015600x`, and attention SDPA to `1.046114x` once the BF16 prep cost was included.
  - 2026-06-22 added and rejected a default-off float-gradient HD64/H12 packed-attention dprep CUDA Tile candidate behind `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_FLOAT_HD64_SPECIALIZED=1` and `NFN_SM120_NATIVE_CANDIDATE_PROFILE=attention_dprep_float_hd64_specialized`. This targets the normal float dO path without the rejected BF16 pack/handoff cost and reports `attention_backward_float_hd64_dprep_launch_count`. The CUDA 13.3 dedicated RTX 5090 same-script 3-step, 3-sample gate proved the route (`0` baseline launches versus `288` candidate launches) and improved dprep timing to `0.998396x`, but failed promotion on LM-head backward (`1.000271x`), MLP projection (`1.000360x`), attention SDPA (`1.000526x`), attention `to_qkv` (`1.000468x`), and TK timing (`1.000963x`).
  - Rechecked the full-logit LM-head reuse probe after moving display work off the 5090; a three-step native-vs-native probe did not complete in the useful benchmark window, so the chunked LM-head classifier remains the only viable default until a fused/cooperative classifier backward lands.
  - CUDA 13.3.33 shape stats still put LM-head dHidden at the top of the GEMM list (`m=768,n=8192,k=50304`, `cublasGemmEx`), but both replacement routes remain slower: `NFN_NATIVE_LINEAR_TK_DINPUT_ENABLE_SHAPE=768,8192,50304,N,N` measured `1.027905x` train-loop wall, and `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE=768,8192,50304,N,N` measured `1.025715x`. cuBLASLt heuristic policies also stayed rejected (`min_waves` `1.007483x`, `max_waves` `1.009232x`), and `--lm-head-row-chunk-size 4096` remained noise/slower (`1.004271x`). Keep these as diagnostics only; the next useful LM-head step is a real fused/cooperative classifier backward, not a route knob.
  - 2026-06-22 rechecked the current 32768-row LM-head dHidden/dWeight route candidates after the CUDA 13.3 reinstall and cooperative wrapper wiring. `lm_head_dhidden_fast16bf_32768` requested FAST_16BF for 48 GEMMEx calls but failed LM-head backward at `1.001544x`; `lm_head_cublaslt_dhidden_32768` moved 48 calls from GEMMEx to cuBLASLt but failed dHidden at `1.001794x`; `lm_head_tk_dinput_32768` regressed train-loop wall to `1.014791x` and dHidden to `1.132698x`; `lm_head_tk_dweight_32768` regressed train-loop wall to `1.023848x` and dWeight to `1.289424x`. Do not promote or re-run these as default candidates without a CUDA/cuBLAS/TK implementation change. The remaining parity work is a fused/cooperative LM-head logits/CE/dHidden/dWeight kernel, not a substitution among the current per-chunk GEMMs.
  - 2026-06-22 added and rejected the no-loss LM-head CE natural-row diagnostic. `NFN_NATIVE_GPT_LM_HEAD_CE_REVERSE_ROWS=0` flips the hot no-loss BF16 classifier kernel from reverse row order to natural launch order and reports `lm_head_ce_reverse_rows_enabled: false` plus `lm_head_ce_row_order_strategy: natural-row-order-diagnostic`; the SM120 parity wrapper can reproduce it with `NFN_SM120_PARITY_CANDIDATE_ENV='NFN_NATIVE_GPT_LM_HEAD_CE_REVERSE_ROWS=0'`, and the native-vs-native candidate wrapper can reproduce the same rejected route with `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_ce_natural_rows`. The dedicated RTX 5090 10-step same-script parity sample proved the route but rejected it at `1.019563x` all-step CUDA-event wall time, `1.019690x` steady-state CUDA-event wall time, and `0.978913x` tokens/sec versus llm.kittens. Keep reverse row order as the default.
  - 2026-06-23 refreshed the current no-stage llm.kittens parity sample after
    the linked-default and CE-profile cleanup:
    `NFN_SM120_PARITY_STEPS=5 NFN_SM120_PARITY_SAMPLES=1
    NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_PROFILE_DIR=none
    NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_current_after_ce_reject.json
    bash tools/bench_native_gpt_sm120_parity.sh`. With the dedicated RTX 5090
    idle, llm.kittens measured `2451.552 ms/step` and NeuralFn measured
    `2495.740 ms/step`, or `1.018025x` train-loop wall and `0.981253x`
    tokens/sec. The gap remains real and points back to fused/cooperative
    LM-head/block-backward kernel work rather than launch-shape retunes.
  - 2026-06-23 refreshed the current stage-timed llm.kittens parity sample on
    the dedicated RTX 5090 after the captured sampler binding and stricter
    candidate-gate cleanup:
    `NFN_SM120_PARITY_STEPS=3 NFN_SM120_PARITY_SAMPLES=2
    NFN_SM120_PARITY_WARMUP=1 NFN_SM120_PARITY_STAGE_TIMING=1
    NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_refresh_3step.json bash
    tools/bench_native_gpt_sm120_parity.sh`. The selected GPU had zero compute
    processes before every sample. llm.kittens measured `2499.553 ms/step` and
    NeuralFn measured `2581.397 ms/step`, or `1.032719x` train-loop wall and
    `0.967622x` tokens/sec. The steady-state CUDA-event ratio was narrower at
    `1.014368x`, with current hot NeuralFn buckets over three steps:
    `stage.block_backward.total_ms=3910.630`,
    `stage.train.model_forward.total_ms=1996.405`, and
    `stage.lm_head_backward.total_ms=1756.295`. This keeps the open parity
    work focused on fused/cooperative LM-head and block-backward kernels.
  - 2026-06-24 refreshed the short stage-timed parity sample after the MLP
    next-LN1 guard and train-loss counter cleanup:
    `NFN_SM120_PARITY_STEPS=3 NFN_SM120_PARITY_SAMPLES=1
    NFN_SM120_PARITY_WARMUP=1 NFN_SM120_PARITY_STAGE_TIMING=1
    NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0
    NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_continue_20260624.json
    NFN_SM120_PARITY_PROFILE_DIR=/tmp/nfn_sm120_parity_continue_20260624_profiles
    bash tools/bench_native_gpt_sm120_parity.sh`. The selected RTX 5090 had
    display disabled, zero compute processes, 0% utilization before every
    sample, and the benchmark lock held. llm.kittens measured
    `2576.893333 ms/step`; NeuralFn measured `2587.400000 ms/step`, or
    `1.004077x` train-loop wall, `1.015852x` steady-state CUDA-event wall, and
    `0.990007x` tokens/sec. Current hot NeuralFn buckets over three steps are
    `stage.block_backward.total_ms=3961.880`,
    `stage.train.model_forward.total_ms=1968.420`,
    `stage.lm_head_backward.total_ms=1751.740`,
    `stage.block_backward.mlp_proj.total_ms=979.072`,
    `stage.block_backward.attn_sdpa.total_ms=804.018`, and
    `stage.block_backward.mlp_fc.total_ms=787.053`.
  - 2026-06-24 removed the candidate wrapper's default CUDA runtime/driver
    version preflight injection. The compiled trainer already leaves
    `NFN_NATIVE_GPT_CUDA_VERSION_PREFLIGHT` off for normal startup; the wrapper
    now matches that and requires `NFN_SM120_NATIVE_CUDA_VERSION_PREFLIGHT=1`
    for explicit diagnostic preflight. A startup-only setup probe measured
    `cuda_runtime_version_preflight_wall_ms` at `119.471 ms` on the first side
    and `110.187 ms` on the second side, so this avoids adding preflight cost to
    raw startup bisections.
  - 2026-06-24 added
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=llmk_sm120_reference_flags` for
    same-script macro-alignment checks against the documented llm.kittens SM120
    reference build. The profile compiles a temporary Tile ops library with the
    full reference macro bundle, disables route-change enforcement because most
    values match current header/default settings, and still keeps timing gates
    active. The CUDA 13.3 dedicated RTX 5090 3-step, 2-sample gate changed no
    tracked route or strategy counters and rejected default promotion: wall time
    improved to `0.994499x` and tokens/sec to `1.005630x`, but steady-state
    CUDA-event timing missed at `1.000331x` and
    `stage.block_backward.mlp_proj.dinput.total_ms` missed at `1.000032x`. Use
    the profile to prove generated-code movement before changing NeuralFn's
    default build flags.
  - 2026-06-24 added and rejected
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=tk_sm120_super_m7` for compile-time TK
    GEMM swizzle bisection. The profile rebuilds candidate Tile ops with
    `LLMK_SM120_SUPER_M=7` and `LLMK_SM120_DINP_SUPER_M=7`. Runtime strategy
    telemetry proved both values changed from the default `8` to `7`, but the
    CUDA 13.3 dedicated RTX 5090 3-step, 2-sample stage-timed gate rejected the
    change at `1.000992x` steady-state CUDA-event timing,
    `1.000168x stage.lm_head_backward.total_ms`, and
    `1.001198x stage.block_backward.mlp_proj.total_ms`.
  - 2026-06-24 added and rejected
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=tk_sm120_super_m13` for the opposite
    TK GEMM swizzle direction. The profile rebuilds candidate Tile ops with
    `LLMK_SM120_SUPER_M=13` and `LLMK_SM120_DINP_SUPER_M=13`. Runtime strategy
    telemetry proved both values changed from the default `8` to `13`, but the
    CUDA 13.3 dedicated RTX 5090 3-step, 2-sample stage-timed gate rejected the
    change at `1.009116x` train-loop wall, `1.002623x` steady-state CUDA-event
    timing, `1.011813x stage.block_backward.total_ms`, and
    `1.010002x stage.block_backward.mlp_proj.total_ms`.
  - 2026-06-24 added paired benchmark metadata for compile-time candidate
    provenance. `tools/paired_kernel_speed.py --metadata KEY=VALUE` now writes
    wrapper-level details into dry-run and measured JSON/text output, and the
    SM120 native candidate wrapper records `candidate_profile`,
    `candidate_tile_ops_build_flags`, and `candidate_route_change_gate` so
    saved benchmark artifacts identify the exact temporary Tile ops build.
  - 2026-06-24 added read-only SM120 TK GEMM compile-config telemetry to native
    GPT JSON and the paired benchmark strategy-change gate. Runtime artifacts
    now report `linear_tk_sm120_*` tile/swizzle/dGELU settings, which lets
    compile-time Tile ops candidates prove generated-kernel changes even when
    their route counters remain constant.
  - 2026-06-24 extended `tools/bench_lm_head_backward_candidate.sh` output via
    `neuralfn/csrc/native_train/lm_head_backward_bench.cpp` so each
    baseline/candidate variant includes CUDA Graph evidence:
    `graph_capture_attempt_count`, `graph_capture_success_count`,
    `graph_cache_hit_count`, `graph_cache_entry_count`, `graph_replay_count`,
    `graph_replay_success_count`, and `graph_fallback_count`. Future strict
    LM-head ABI probes now prove graph replay directly in the standalone
    candidate-vs-old microbench before entering a full native training gate.
    A trainer-chunk run on the dedicated RTX 5090 reported the strict candidate
    captured once, replayed three times, fell back zero times, and stayed flat
    at `1.000358x`, so CUDA Graph replay is measurable but still not a default
    LM-head speed win at the real chunk shape.
    A full-loop
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_cooperative_backward` rerun with
    the new counters proved the strict graph route executed
    (`lm_head_fused_graph_capture_success_count=3`,
    `lm_head_fused_graph_cache_hit_count=45`,
    `lm_head_fused_graph_replay_success_count=48`, and no sequence-wrapper
    launches), but rejected it even more clearly at `1.100254x` train-loop wall,
    `1.069816x` steady-state CUDA-event timing, and
    `1.295083x stage.lm_head_backward.total_ms`. Keep the graph body as ABI and
    benchmark groundwork only; the default still needs a lower-overhead LM-head
    implementation.
  - 2026-06-24 rechecked the standalone LM-head cooperative backward candidate
    after the CUDA 13.3 reinstall. `tools/bench_lm_head_backward_candidate.sh`
    at trainer-chunk shape reported `candidate_true_fused_capability=false` and
    measured `1.010998x` candidate/baseline in baseline-first order, `0.997661x`
    in candidate-first order, and `1.002528x` with row-loss recording. The
    candidate is noise-level to slower because it only schedules CE, dHidden,
    and dWeight as a cross-stream sequence; it is not the true fused kernel the
    integrated trainer gate requires. Keep `lm_head_cooperative_backward` out of
    defaults until `nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused()`
    returns true and a same-script benchmark shows a durable win.
  - 2026-06-23 rechecked `NFN_SM120_NATIVE_CANDIDATE_PROFILE=tk_dgelu_dinput`
    after the row-chunk 49152 promotion and CUDA 13.3 setup. The 5-step,
    3-sample stage-timed same-script run measured `0.997858x` train-loop wall
    and `1.002209x` tokens/sec, but rejected promotion because no tracked
    route or strategy changed and `stage.lm_head_backward.total_ms` missed the
    strict gate at `1.000159x`. Added
    `linear_tk_dgelu_dinput_gemm_count` as a dedicated route counter so future
    compile-time dGELU candidates must prove fused dInput+dGELU launches rather
    than relying on timing-only noise.
  - 2026-06-27 added `NFN_NATIVE_LINEAR_TK_DGELU_DINPUT_DISABLE_SHAPE` /
    `NFN_TILE_CUDA_LINEAR_TK_DGELU_DINPUT_DISABLE_SHAPE` and
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=mlp_proj_dgelu_fallback` so the fused
    GPT MLP projection dInput+dGELU route can be disabled by exact GEMM shape.
    The dedicated RTX 5090 route gate passed by dropping
    `linear_tk_dgelu_dinput_gemm_count` from `288` to `0`, but the candidate
    regressed train-loop wall to `1.013580x`, block backward to `1.027454x`,
    MLP projection total to `1.107897x`, and MLP projection dInput to
    `1.207964x`. Keep it rejected/default-off; the fused TK dGELU route remains
    the faster default.
  - 2026-06-27 refreshed the normal no-stage parity run against
    `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh` on the dedicated RTX
    5090 after the dGELU route-bisection commit. With 5 steps, 3 measured
    samples, 1 warmup sample, stage timing disabled, and zero selected-GPU
    compute processes before/after samples, current NeuralFn measured
    `0.995097x` candidate-over-llm.kittens train-loop wall time, `0.995827x`
    steady-state CUDA-event wall time, and `1.005236x` tokens/sec. Runtime
    contract checks still reported `graph_editor_tensor_flow=false` and
    `torch_required=false`. Treat normal training-loop SM120 parity as currently
    met for this benchmark; keep remaining work focused on startup policy,
    diagnostic overhead, full no-Torch/native coverage, and replacing the
    LM-head diagnostic CUDA Graph wrapper with a production true-fused Tile
    kernel.
  - Added guarded vec4 CUDA candidates for the multi-buffer `float32_to_bf16_bits_many` packer and stored-MLP activation pack/restore path, but keep them default-off. The CUDA 13.3 dedicated RTX 5090 two-sample paired run measured scalar faster than the vec4-default baseline (`0.994143x` candidate/default train-loop wall, `1.005941x` tokens/sec) and reported no tracked route-counter change. Use `NFN_NATIVE_GPT_F32_TO_BF16_MANY_VEC4=1` or `NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS_VEC4=1` only for future same-script bisection.
  - 2026-06-20 CUDA 13.3.33 WSL revalidation: rebuilt every native CUDA trainer and reran the GPU-visible pytest gates on the dedicated RTX 5090. After reinstalling the latest WSL CUDA toolkit (`cuda-toolkit-13-3`), `NFN_TILE_CUDA_TEST=1 python -m pytest tests/test_native_gpt2.py tests/test_tile_cuda_examples.py tests/test_tile_cuda_gpu.py tests/test_tile_cuda_ops.py tests/test_tile_cuda_optimizer.py -q -rs` passed `247` tests, `python -m pytest tests/test_template_presets.py -x -q` passed `26` tests, `python tools/check_native_no_torch_deps.py --skip-artifacts --json` reported `"passed": true`, and the full repository suite later passed with `1185 passed, 4 skipped, 20 warnings, 468 subtests passed`. No CUDA correctness failure remains from the toolkit reinstall.
  - 2026-06-20 staged 10-step parity still shows performance work remaining rather than a failed test: llm.kittens measured `2447.451 ms/step` and NeuralFn measured `2564.590 ms/step`, or `1.047862x` train-loop wall time and `0.952442x` tokens/sec. The hottest measured buckets were `stage.block_backward.total_ms=12918.1` and `stage.lm_head_backward.total_ms=6298.63`, with LM-head split across logits (`2198.35`), CE (`670.532`), dHidden (`1731.16`), and dWeight (`1675.86`) over 10 steps.
  - 2026-06-20 rejected CUDA 13.3.33 reruns after the reinstall: `NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1` moved 320 LM-head dHidden calls from BF16 GEMMEx to cuBLASLt but regressed train-loop wall to `1.034147x` and targeted dHidden to `1.502430x`; `NFN_TILE_CUDA_CUBLASLT_HEURISTIC_SHAPE=768,65536,3072,N,N,0` regressed train-loop wall to `1.004699x`; `NFN_TILE_CUDA_CUBLASLT_HEURISTIC_SHAPE=768,50304,8192,N,T,0` regressed train-loop wall to `1.003986x`; token-weight vector4 and threaded startup initializers did not improve setup enough to promote; `NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC=1` regressed setup wall to `1.149960x`; and the full-logit reuse probe again exceeded the useful paired-benchmark window. Keep all of these default-off.
  - 2026-06-20 post-CUDA-13.3.33-reinstall startup retile sweep kept the trainer-facing token initializer at the 4096-element Tile default. Temporary libraries built with `NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE=8192`, `2048`, and `1024` all failed the targeted startup gate: 8192 measured `1.013697x` token-init time, 2048 measured `1.010289x`, and 1024 measured `1.016591x` versus the current default. The 2048 run showed a noisy `0.992968x` setup wall, but the direct token-init bucket still regressed, so none of these compile-time tile-size variants should be promoted.
  - 2026-06-20 added train-only shard resolution for disabled-validation native GPT runs. When `--eval-every-steps 0` or `--eval-batches 0` is set, the compiled C++ resolver skips validation shard discovery and reports `validation_shards_required: false` plus `validation_shards_resolved: false`; eval-enabled runs still require validation shards unless `--allow-train-val-fallback` is explicit. The old-vs-new RTX 5090 TinyStories startup gate did not improve (`setup_wall_ms=1.004984x`) because CUDA arena materialization and token-weight initialization dominate this dataset, so this is a startup/workflow cleanup rather than a parity-closing kernel change.
  - 2026-06-20 refreshed the same-script parity check after the CUDA Toolkit 13.3.33 retest: `NFN_SM120_PARITY_STEPS=5 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_PROFILE_DIR=/tmp/nfn_parity_profiles_20260620_current NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=30 NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_20260620_current.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2440.084 ms/step` and NeuralFn at `2521.440 ms/step`, or `1.033341x` train-loop wall time and `0.966919x` tokens/sec with zero compute processes on the selected RTX 5090 before and after the sample. Setup was `579.304 ms`, with `setup.uint16_arena_materialize=118.670 ms` and `setup.token_weight_init=157.781 ms`. A one-step stage/shape profile still put the largest buckets at `block_backward=1499.38 ms`, `lm_head_backward=723.958 ms`, `train.model_forward=634.410 ms`, `lm_head_backward.dhidden=253.449 ms`, `lm_head_backward.logits=221.424 ms`, and `lm_head_backward.dweight=177.271 ms`.
  - 2026-06-20 continued parity check after the CUDA 13.3 correctness pass: `NFN_SM120_PARITY_STEPS=5 NFN_SM120_PARITY_SAMPLES=3 NFN_SM120_PARITY_WARMUP=1 NFN_SM120_PARITY_PROFILE_DIR=/tmp/nfn_parity_20260620_continue NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_parity_20260620_continue.json NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=30 bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2474.982667 ms/step` and NeuralFn at `2544.026667 ms/step`, or `1.027937x` train-loop wall time and `0.972783x` tokens/sec. Startup averaged `540.128333 ms`, with `setup.uint16_arena_materialize=109.113667 ms` and `setup.token_weight_init=153.305000 ms`. A 2-step stage profile kept `block_backward` and `lm_head_backward` as the hot train-loop buckets, with `stage.block_backward.mlp_proj.total_ms=664.660`, `stage.block_backward.attn_sdpa.total_ms=540.982`, `stage.block_backward.mlp_fc.total_ms=534.524`, and `stage.lm_head_backward.total_ms=1261.080` over 2 steps.
  - 2026-06-20 rejected the remaining obvious LM-head dHidden route knob on the current build. `NFN_NATIVE_LINEAR_TK_DINPUT_ENABLE_SHAPE=768,8192,50304,N,N` changed the route counter (`linear_tk_gemm_count` rose from `2720` to `3040` over 5 steps) but the 5-step, 3-sample native-vs-native gate measured `1.018638x` train-loop wall time and `0.981711x` tokens/sec versus the default. Keep the TK dInput path diagnostic-only for this LM-head dHidden shape; the parity gap still needs a fused/cooperative row-chunked classifier-backward kernel or a materially different GEMM implementation.
  - 2026-06-20 added `NFN_NATIVE_GPT_LM_HEAD_FUSED_LOSS_BACKWARD=0` / `NFN_NATIVE_GPT2_LM_HEAD_FUSED_LOSS_BACKWARD=0` as a diagnostic switch for comparing the default fused loss-accumulate+dlogits classifier path against separate loss partial reduction plus CE backward. Keep the fused route enabled by default: the stronger 5-step, 3-sample confirmation run failed gates at `1.001484x` train-loop wall, `1.003244x` block backward, and `1.000194x` MLP projection backward.
  - 2026-06-20 post-CUDA-13.3.33-reinstall LM-head chunk sweep kept the 32768-row default. `--lm-head-row-chunk-size 16384` doubled LM-head logit GEMM launches and regressed train-loop wall to `1.016019x` and LM-head backward to `1.062838x`, mainly because dHidden worsened to `1.244198x`. `--lm-head-row-chunk-size 4096` improved CE to `0.961277x` but increased logit GEMM launches further, regressed dWeight to `1.039507x`, and measured `1.004875x` train-loop wall. Do not revisit LM-head chunk-size retuning unless a future CUDA/cuBLAS build changes the logits/dHidden/dWeight route characteristics materially.
  - 2026-06-20 accepted a narrow BF16 classifier/CE read-path improvement: when vector loads are enabled but vector streaming stores are disabled, the final dlogit pass now loads logits with the same 128-bit BF16 vector reads used by the max/sum passes and writes scalar BF16 gradients. The 3-step, 5-sample stage-timed candidate gate passed against the saved prior Tile ops library: train-loop wall `0.995665x`, LM-head backward `0.997949x`, CE `0.994846x`, block backward `0.994074x`, and tokens/sec `1.004513x`. This is a small real movement toward the llm.kittens fused-classifier memory pattern, while the larger parity item still needs fused/cooperative LM-head logits/CE/dHidden/dWeight work.
  - 2026-06-22 added and rejected the narrower BF16 classifier scalar streaming-store candidate. `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_ce_scalar_streaming_store` expands to `NFN_NATIVE_GPT_CE_BF16_VEC_LOADS=1 NFN_NATIVE_GPT_CE_BF16_SCALAR_STREAMING_STORES=1`, changes runtime strategy from `vec8-loads-scalar-stores` to `vec8-loads-scalar-streaming-stores`, and writes scalar dlogits with `st.global.cs.u16`. The dedicated RTX 5090 3-step, 2-sample stage-timed same-script gate rejected it at `1.005702x` train-loop wall, `1.027580x` LM-head backward, and `1.135829x` LM-head CE time. Keep it diagnostic-only; cached scalar stores remain the default.
  - 2026-06-20 refreshed NeuralFn-vs-llm.kittens parity after the accepted CE final vector-load change. `NFN_SM120_PARITY_STEPS=3 NFN_SM120_PARITY_SAMPLES=3 NFN_SM120_PARITY_WARMUP=1 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=auto NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=25 NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_parity_after_ce_final_vecload_3step_3sample.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2522.613333 ms/step` median and NeuralFn at `2560.936667 ms/step` median, or `1.018678x` train-loop wall time and `0.983520x` median tokens/sec. The goal remains open; the next highest-value work is still a cooperative/fused LM-head classifier-backward schedule or materially faster GEMM route, not more scalar CE load/store retuning.
  - 2026-06-20 added and rejected-as-default the CUDA 13.3 BF16 cuBLASLt plan prewarm diagnostic. `NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLANS=1` successfully prewarmed all 9 target plans (`linear_cublaslt_plan_prewarm_success_count=9`) and improved 5-step train-loop wall time to `0.994375x`, but it added about 75 ms of setup work (`setup_wall_ms=1.158747x`) and still failed strict block-backward gates (`stage.block_backward.total_ms=1.000084x`, `stage.block_backward.mlp_proj.total_ms=1.000116x`). Keep it opt-in for attribution only; this does not close the `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh` parity item.
  - 2026-06-22 added `nfn_native_tile_trainer_linear_cublaslt_grouped_matmul_probe_status` as a non-fatal cuBLASLt grouped execution smoke, separate from the existing grouped-layout probe and the legacy cuBLAS grouped BF16 probe. The current CUDA 13.3 RTX 5090 run reports `linear_cublaslt_grouped_layout_supported: true` but requested `NFN_NATIVE_GPT_PROBE_CUBLASLT_GROUPED_MATMUL=1` returns status `15`, so grouped cuBLASLt execution is not ready for LM-head/block-backward routing yet.
  - 2026-06-20 post-reinstall no-stage parity refresh confirmed the remaining issue is performance, not failing CUDA tests. After the 32768-row LM-head chunk default and unsafe-chunk guard, `NFN_SM120_PARITY_STEPS=5 NFN_SM120_PARITY_SAMPLES=3 NFN_SM120_PARITY_WARMUP=1 NFN_SM120_PARITY_PROFILE_DIR=/tmp/nfn_parity_post_guard_20260620 NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_parity_post_guard_20260620.json NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=30 bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2495.348667 ms/step` and NeuralFn at `2522.440 ms/step`, or `1.010870x` train-loop wall time and `0.988745x` tokens/sec. This supersedes the earlier `1.032455x` no-stage post-reinstall baseline.
  - 2026-06-20 added and rejected-as-default a block-backward MLP FC side-stream diagnostic behind `NFN_NATIVE_GPT_BLOCK_MLP_FC_CONCURRENT_DINPUT_DWEIGHT=1` / `NFN_NATIVE_GPT2_BLOCK_MLP_FC_CONCURRENT_DINPUT_DWEIGHT=1`. The route records a default-stream CUDA event, waits from non-blocking dInput/dWeight streams, launches the independent MLP FC dInput and dWeight+bias kernels on those streams, and synchronizes before LN2 backward. The CUDA 13.3 RTX 5090 same-script gate proved the route enabled (`block_backward_mlp_fc_concurrent_dinput_dweight_enabled: true`) but rejected it at `1.006693x` train-loop wall, `1.012567x` block backward, and `1.028021x` MLP FC backward versus the serial default. Keep this diagnostic-only; the next block-backward work needs fused/cooperative kernels or materially different GEMM implementations, not stream overlap of the current calls.
  - 2026-06-25 added and rejected-as-default the matching MLP projection side-stream candidate behind `NFN_NATIVE_GPT_BLOCK_MLP_PROJ_CONCURRENT_DINPUT_DWEIGHT=1` / `NFN_NATIVE_GPT2_BLOCK_MLP_PROJ_CONCURRENT_DINPUT_DWEIGHT=1` plus the named candidate profile `mlp_proj_concurrent_dinput_dweight`. The route pre-materializes the shared BF16 projection grad-out, then launches fused projection dInput+dGELU and projection dWeight+bias on non-blocking side streams before the MLP FC backward consumer runs. Runtime JSON reports `block_backward_mlp_proj_concurrent_dinput_dweight_requested`, `block_backward_mlp_proj_concurrent_dinput_dweight_enabled`, and `block_backward_mlp_proj_concurrent_dinput_dweight_count`; the same-script wrapper gates `stage.block_backward.mlp_proj.total_ms` for stage-timed runs. The CUDA 13.3.33 dedicated RTX 5090 3-step, 2-sample gate proved the counter moved `0 -> 288`, but rejected promotion because train-loop wall regressed to `1.004101x`, steady-state CUDA-event timing to `1.004144x`, LM-head backward to `1.000889x`, block backward to `1.009823x`, and MLP projection backward to `1.025216x`. Keep this diagnostic-only; like the MLP FC route, current stream overlap is not enough to beat the serial fused/cached GEMM schedule.
  - 2026-06-21 added and rejected-as-default the matching attention-projection side-stream diagnostic behind `NFN_NATIVE_GPT_BLOCK_ATTN_PROJ_CONCURRENT_DINPUT_DWEIGHT=1` / `NFN_NATIVE_GPT2_BLOCK_ATTN_PROJ_CONCURRENT_DINPUT_DWEIGHT=1` plus the named candidate profile `attn_proj_concurrent_dinput_dweight`. The route enabled (`block_backward_attn_proj_concurrent_dinput_dweight_enabled: true`) and emitted `stage.block_backward.attn_proj.dinput_dweight_concurrent`, but the same-script RTX 5090 gate rejected it at `1.000183x` train-loop wall, `1.004192x` block backward, and `1.089203x` attention-projection backward versus the serial default. Keep it diagnostic-only; the attention projection gap also needs a better fused/cooperative kernel or GEMM route, not stream overlap.
  - 2026-06-24 added and rejected-as-default a first-step-only attention projection side-stream diagnostic behind `NFN_NATIVE_GPT_BLOCK_ATTN_PROJ_FIRST_STEP_CONCURRENT_DINPUT_DWEIGHT=1` / `NFN_NATIVE_GPT2_BLOCK_ATTN_PROJ_FIRST_STEP_CONCURRENT_DINPUT_DWEIGHT=1` plus the named candidate profile `attn_proj_first_step_concurrent_dinput_dweight`. The route-change gate proved hot execution (`block_backward_attn_proj_first_step_concurrent_dinput_dweight_count: 0 -> 96`) on the CUDA 13.3 dedicated RTX 5090 5-step, 3-sample run, but rejected default promotion because train-loop wall regressed to `1.002629x`, steady-state CUDA event timing to `1.001028x`, block backward to `1.006184x`, and attention projection to `1.075065x`.
  - 2026-06-21 exposed the existing LM-head dHidden/dWeight side-stream schedule as the named candidate profile `lm_head_concurrent_dhidden_dweight`, expanding to `NFN_NATIVE_GPT_LM_HEAD_CONCURRENT_DHIDDEN_DWEIGHT=1`. Stage-timed wrapper runs report the combined `stage.lm_head_backward.dhidden_dweight_concurrent.total_ms` bucket for candidate-side inspection, but do not ratio-gate that candidate-only bucket because the serial baseline emits split dHidden/dWeight substages. After that unmatched gate was removed, the same-script RTX 5090 run rejected it as a default on comparable gates: train-loop wall `1.012827x`, total LM-head backward `1.009974x`, and block backward `1.021336x`. The QKV side-stream profile was also rejected at `1.005658x` train-loop wall and `1.041664x` QKV stage time. Keep both routes diagnostic-only; the next parity work needs a fused/cooperative block-backward or classifier-backward kernel rather than stream overlap around current GEMMs.
  - 2026-06-21 profiled the current one-step GEMM route after the side-stream diagnostics: LM-head logits, dHidden, and dWeight are still the largest tied-head buckets at roughly `179 ms`, `177 ms`, and `169 ms` per optimizer step, while block backward remains dominated by MLP projection, MLP FC, attention SDPA, and QKV GEMMs. A targeted cuBLASLt heuristic override for LM-head dWeight (`NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,50304,32768,N,T,0`) rejected at `1.005593x` train-loop wall and `1.001167x` total LM-head backward. Do not retest that heuristic path unless CUDA/cuBLASLt changes; the next LM-head work needs a fused/cooperative classifier-backward route or a materially different GEMM implementation.
  - 2026-06-22 added reproducible candidate profiles for two more rejected probes instead of promoting either. `NFN_SM120_NATIVE_CANDIDATE_PROFILE=tk_forward_no_n96` builds a temporary Tile ops library with `-DLLMK_SM120_FORWARD_N96=0`; the dedicated RTX 5090 same-script gate rejected it at `1.001827x` train-loop wall time and `1.004826x` block-backward time. `NFN_SM120_NATIVE_CANDIDATE_PROFILE=cuda_device_max_connections_1` sets `CUDA_DEVICE_MAX_CONNECTIONS=1` only on the candidate command; it changed no tracked route counters and rejected at `1.007508x` train-loop wall time, `1.000579x` LM-head backward time, and `1.015155x` block-backward time. Keep both as benchmark profiles only.
  - 2026-06-22 tested the current 32768-row LM-head logits TK route against the BF16 GEMMEx fallback. `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=50304,32768,768,T,N` changed the route counters (`lm_head_logits_tk_gemm_count` fell from `48` to `0` over three measured steps), but failed the dedicated RTX 5090 gate at `1.001931x` train-loop wall time, `1.004681x` LM-head backward time, and `1.004700x` LM-head logits time. The reproducible profile is `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_logits_bf16_fallback_32768`; keep the current TK logits route as default.
  - 2026-06-24 rechecked `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_logits_bf16_fallback_32768`
    after the LM-head row-chunk default moved to 49152. The candidate still
    expands to `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=50304,32768,768,T,N`,
    but it no longer matches the active logits shape: no route counters,
    strategy strings, linear shape stats, or cuBLASLt plan-cache entries
    changed, and the native route-change gate failed. Keep it rejected by
    default and use `lm_head_logits_bf16_fallback_49152` for current-shape
    diagnostics.
  - 2026-06-22 retested the packed-QKV forward TK route against the BF16 fallback on the current CUDA 13.3 build. `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=2304,65536,768,T,N` changed the route counter (`linear_tk_gemm_count` fell from `1488` to `1200` over three measured steps), but failed the dedicated RTX 5090 gate at `1.009016x` train-loop wall time and regressed `stage.block_forward.attention.total_ms` to `1.091020x`. The reproducible profile is `NFN_SM120_NATIVE_CANDIDATE_PROFILE=qkv_forward_bf16_fallback_65536`; keep the current TK packed-QKV forward route as default.
  - 2026-06-23 refreshed the one-step CUDA 13.3 stage/shape profile and rechecked the hot forward fallback direction with stricter target-stage gates. `NFN_SM120_NATIVE_CANDIDATE_PROFILE=qkv_forward_bf16_fallback_65536` reduced TK forward calls but regressed the target `stage.block_forward.attention.qkv.total_ms` to `1.143374x`; `NFN_SM120_NATIVE_CANDIDATE_PROFILE=mlp_fc_forward_bf16_fallback_65536` expands to `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=3072,65536,768,N,N`, changed no tracked route counters, and regressed train-loop wall to `1.016916x`, block backward to `1.034425x`, and `stage.block_forward.mlp_fc_gelu.total_ms` to `1.000722x`. Keep the TK forward defaults; the remaining parity direction is still a fused/cooperative LM-head or block-backward kernel route, not fallbacking these forward GEMMs.
  - 2026-06-22 rechecked grouped execution after the CUDA 13.3 WSL reinstall. `NFN_SM120_NATIVE_CANDIDATE_PROFILE=cublaslt_grouped_probe` still requests only the non-poisoning cuBLASLt grouped-layout and grouped-matmul probes. The dedicated RTX 5090 paired run reports cuBLASLt layout status `0` but cuBLASLt grouped matmul status `15`. A follow-up attempt to include the classic cuBLAS grouped BF16 probe in the same profile failed the candidate with `cudaMalloc transformer_lm_float_arena failed with CUDA error 700`, confirming the probe still poisons the selected CUDA context when unsupported. Keep grouped-GEMM parity work blocked until execution, not just descriptor creation, passes.
  - 2026-06-24 reran the same non-poisoning `cublaslt_grouped_probe` after the
    latest CUDA reinstall on the dedicated RTX 5090:
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=cublaslt_grouped_probe
    NFN_SM120_NATIVE_STEPS=1 NFN_SM120_NATIVE_SAMPLES=1
    NFN_SM120_NATIVE_WARMUP=0 NFN_SM120_NATIVE_STAGE_TIMING=1`. The selected
    GPU was display-disabled and idle before/after the sample, grouped layout
    still returned `0`, and grouped matmul execution still returned `15`.
    Keep grouped block-backward work blocked; descriptor creation is not enough
    to start replacing the existing per-shape GEMMs.
  - 2026-06-24 added
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=cublaslt_grouped_probe_required` as a
    fail-fast prerequisite gate for future grouped-GEMM implementation work. It
    expands to the same non-poisoning layout/matmul probes as
    `cublaslt_grouped_probe`, but fails after the paired run unless both
    `linear_cublaslt_grouped_layout_probe_status` and
    `linear_cublaslt_grouped_matmul_probe_status` are `0`. Use the normal
    profile for telemetry and the required profile only when grouped execution
    is a hard dependency for a candidate patch.
  - 2026-06-24 added the reproducible rejected wrapper profile
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_public_vocab_strided_gemm`.
    It compares the default aligned padded-vocab LM-head dHidden/dWeight GEMMs
    against `NFN_NATIVE_GPT_LM_HEAD_PUBLIC_VOCAB_STRIDED_GEMM=1` in the same
    native-vs-native harness and gates both LM-head substages. The prior CUDA
    13.3 dedicated RTX 5090 same-binary run measured this route at `1.117352x`
    train-loop wall and `0.895573x` tokens/sec, so it remains diagnostic-only
    unless a future CUDA/cuBLAS implementation materially changes the result.
  - 2026-06-20 rejected the remaining post-reinstall switch candidates as defaults: disabling the combined BF16 arena helped startup (`0.952375x` total) but regressed train-loop wall (`1.001300x`); disabling TK forward for `768,65536,3072,T,N` passed train-loop noise but failed total wall (`1.001649x`); disabling BF16 cuBLASLt broadly was a hard regression (`5.644887x` train-loop wall); token-weight vector4 strided init failed token-init gates (`1.006519x`); disabling the token-weight BF16 shadow regressed train-loop wall (`1.004467x`); `--lm-head-row-chunk-size 4096` again failed (`1.000410x` train-loop wall, `1.013208x` LM-head backward); and `NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1` regressed train-loop wall (`1.002900x`). Do not promote route toggles without a same-script win that changes the targeted counters and passes total-wall gates.
  - 2026-06-20 rejected two post-target-logit scheduling switches as defaults.
    `NFN_NATIVE_GPT_MLP_PROJ_DINPUT_BEFORE_DWEIGHT=1` measured better total
    train-loop noise (`0.996116x`) but failed direct hot gates:
    `stage.lm_head_backward.total_ms=1.000081x` and
    `stage.block_backward.mlp_proj.total_ms=1.001189x`; the MLP projection
    dWeight substage improved only by making dInput `1.100269x` slower.
    `NFN_NATIVE_GPT_ATTN_PROJ_DINPUT_BEFORE_DWEIGHT=1` regressed train-loop wall
    to `1.003056x`, block backward to `1.008069x`, and attention projection
    dInput to `1.159150x`. Neither changed tracked route counters, so keep both
    diagnostic-only and continue with real fused/cooperative kernel work.
  - 2026-06-22 added the named no-loss classifier candidate profile
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_ce_vec8_io`, which expands to
    `NFN_NATIVE_GPT_CE_BF16_VEC_LOADS=1 NFN_NATIVE_GPT_CE_BF16_VEC_STORES=1`.
    Native runtime JSON now reports `lm_head_ce_bf16_vector_io_strategy` and
    the individual CE vec-load/store booleans so paired RTX 5090 runs can prove
    the candidate changed the normal LM-head CE dlogit write route before
    judging timing. Keep it diagnostic-only: the 2026-06-22 5-step, 3-sample
    same-script gate saw the CE substage at `0.999670x`, but failed total
    LM-head and block MLP-proj gates (`1.001033x` and `1.000934x`). A shorter
    parser check confirmed paired output records the strategy transition from
    `vec8-loads-scalar-stores` to `vec8-loads-streaming-stores`.
  - 2026-06-22 fixed the default-off streaming-store candidate so the final
    dlogit write pass reuses packed vec8 BF16 loads whenever vec loads are
    enabled, matching the intended load128/store128cs memory-access shape more
    closely. Keep it diagnostic-only: the post-fix dedicated RTX 5090 2-step,
    2-sample same-script gate proved the strategy transition from
    `vec8-loads-scalar-stores` to `vec8-loads-streaming-stores`, but rejected
    the candidate at `1.001346x` train-loop wall time, `0.998656x` tokens/sec,
    `1.000944x` LM-head backward, and `1.004197x` CE time.
  - 2026-06-23 rechecked `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_ce_vec8_io`
    on the current CUDA 13.3 dedicated RTX 5090 stack. The route still changed
    CE vector I/O, but the stage-timed 5-step, 2-sample gate failed
    `stage.lm_head_backward.ce.total_ms=1.003780x`; the apparent total
    train-loop gain was driven by unrelated block-backward timing noise. The
    wrapper now rejects real runs by default unless
    `NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1` is set.
  - 2026-06-20 rejected CUDA write-combined pinned token staging for the native
    GPT token/target host buffer. The candidate changed the trainer
    `cudaHostAlloc` flags from default to write-combined and passed the
    `--smoke-lm-step` check, but the explicit saved-binary paired benchmark
    measured train-loop wall `1.001426x`, setup `1.004470x`, and total wall
    `1.002054x`. The source was reverted to default pinned host allocation.
  - 2026-06-20 CUDA 13.3 token-weight startup rerun: after rebuilding the trainer-facing Tile ops library and native GPT CLI with the latest WSL toolkit, the attempted fast-int32 rollback was not promoted. The startup-only 5-sample paired gate compared the default vector4 route against `NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT=0` and measured vector4 faster at `0.949565x` token-init time, `0.970270x` setup wall time, and `0.970405x` total wall time. Keep vector4 as the default and use `NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT=0` only for paired bisection against the previous fast int32 Tile initializer.
  - 2026-06-22 added and rejected a narrower vector4 token-weight startup candidate: `NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_PATTERN_INIT=1` stores the BF16 shadow from the deterministic 16-value BF16 pattern instead of recomputing BF16 conversions per element, but the startup-only RTX 5090 gate measured it slower than the default conversion writer (`1.023037x` token-init time, `1.019686x` setup wall, `1.019676x` total wall). Keep it diagnostic-only.
  - 2026-06-20 rejected the LM-head dHidden BF16 GEMMEx ALGO0 override after discovering the candidate wrapper had ignored generic aliases before the fallback fix. The actual default-sized same-script run with `NFN_NATIVE_LINEAR_BF16_GEMM_EX_ALGO_SHAPE=768,8192,50304,N,N,0` measured `1.002367x` train-loop wall time and `0.997642x` tokens/sec with no route-counter change, so keep that shape-specific ALGO0 route diagnostic-only.
  - 2026-06-20 continued the post-reinstall parity audit on the clean tree. A startup-only paired run comparing the compiled C++ command with and without `--tinystories` measured `1.000758x` setup wall time, so dataset alias resolution is not the current startup bottleneck; setup is still CUDA arena materialization plus token-weight initialization. The wrapper dry-run `python cli/scripts/train_gpt.py --tinystories --native-cuda-dry-run --native-cuda-print-command --native-cuda-no-checkpoint` completed in about `0.03s`, proving the native GPT Python wrapper is not doing the old multi-minute schedule estimate before delegating. Rechecking `NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLANS=0` saved setup (`0.851303x`) but regressed train-loop wall (`1.002307x`) and LM-head backward (`1.017995x`), so plan prewarm stays enabled for normal training. Disabling TK forward for the hot QKV shape `2304,65536,768,T,N` reduced TK calls but regressed train-loop wall to `1.012063x`; keep that shape on the TK route. The remaining performance item is still fused/cooperative LM-head and block-backward kernel work rather than a default toggle.
  - 2026-06-24 added the diagnostic cuBLASLt plan-prewarm mode selector (`NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLAN_MODE=all|block_only|lm_head_only`) and rejected both selective modes on the dedicated RTX 5090. Startup-only probes confirmed the counters (`all`: attempted 9/skipped 0, `block_only`: attempted 8/skipped 1, `lm_head_only`: attempted 1/skipped 8). `lm_head_only` saved setup to `0.947250x` but regressed train-loop wall to `1.011688x`, steady-state CUDA-event time to `1.001999x`, LM-head backward to `1.000280x`, block backward to `1.022887x`, and MLP projection backward to `1.021800x`. A 2026-06-25 split-stage rerun of `block_only` improved setup to `0.989552x`, but rejected it again because train-loop wall regressed to `1.001350x`, first-step CUDA-event time to `1.002789x`, forward QKV first-step avg to `1.022751x`, and no route, strategy, or plan-cache change passed the native route gate. Keep the default mode at `all`; the parity gap still needs fused/cooperative kernels rather than selective plan prewarm.
  - 2026-06-24 reran the previously borderline BF16 workspace prewarm route on the current build and kept it rejected. The 5-step, 3-sample dedicated RTX 5090 gate changed only setup/prewarm counters and failed hot-route attribution; timing gates regressed at train-loop wall `1.000826x`, steady-state CUDA-event time `1.000283x`, LM-head backward `1.000252x`, block backward `1.001255x`, and MLP projection backward `1.000923x`.
  - 2026-06-24 refreshed the NeuralFn-vs-llm.kittens parity snapshot after the prewarm audits. `NFN_SM120_PARITY_STEPS=3 NFN_SM120_PARITY_SAMPLES=2 NFN_SM120_PARITY_WARMUP=1 NFN_SM120_PARITY_STAGE_TIMING=1 bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2471.550 ms/step` and NeuralFn at `2546.256667 ms/step`, or `1.030315x` train-loop wall, `1.012454x` steady-state CUDA-event time, and `0.970295x` tokens/sec. The remaining hot buckets are still block backward (`3820.990 ms`), model/block forward (`1981.310 ms` / `1970.060 ms`), and LM-head backward (`1756.410 ms`).
- [x] Harden paired benchmark cleanup for interrupted candidates: `tools/paired_kernel_speed.py` now terminates the active command process group on Ctrl-C or unexpected interruption, not only on timeout, so aborted memory-heavy native probes do not keep running on the selected GPU and contaminate later samples.
- [x] Add forward-stage attribution to paired benchmark text summaries. `--native-stage-timing` now prints the existing `stage.train.model_forward.*`, `stage.block_forward.*`, and `stage.block_recompute.*` totals beside the backward buckets, so the next parity pass can assign the remaining ~2.8% gap without manually parsing profile JSON.
  - 2026-06-20 extended the same stdout/gate allowlist with the missing block-backward children that native JSON already records: MLP FC, MLP projection, LN2 residual, attention projection, attention SDPA grad-out/to-QKV, QKV dInput/dWeight, and LN1 residual substages. Future kernel candidates can now gate direct hot metrics such as `stage.block_backward.qkv.dinput.total_ms`, `stage.block_backward.attn_proj.dinput.total_ms`, or `stage.block_backward.ln1_residual.fused_affine_dinput_add.total_ms` without opening profile sidecars.
  - 2026-06-20 added optimizer support-stage attribution to the same paired stdout/gate allowlist: final norm backward, embedding backward, gradient zero, gradient clip, and AdamW update totals now print as first-class native metrics. Future CUDA Tile work can gate those support costs directly instead of digging through profile JSON when a candidate shifts work outside LM-head or block-backward buckets.
- [x] Add dry-run plan inspection to paired SM120 benchmark helpers so baseline/candidate commands, selected CUDA policy, profile settings, and alternating sample order can be audited before launching long GPU timing jobs.
- [x] Add `NFN_SM120_NATIVE_STARTUP_ONLY=1` to `tools/bench_native_gpt_sm120_candidate.sh` so startup-focused native candidates are compared with the same baseline-vs-candidate command shape, selected-GPU idle guard, and external-load controls as train-loop kernel bisections.
  - 2026-06-23 added `NFN_SM120_NATIVE_CANDIDATE_PROFILE=linked_startup`
    (alias `linked_tile_ops`) for the dynamic-loader versus linked-binary
    startup comparison. The profile selects `build/nfn_gpt_native_train` as
    baseline, `build/nfn_gpt_native_train_linked` as candidate, forces
    startup-only zero-step timing with no warmup by default, and disables the
    route-change gate because the expected change is linked Tile-op symbol
    resolution rather than a training-loop kernel route.
  - 2026-06-25 rechecked `linked_startup` after the CUDA 13.3.33 rebuild on
    the dedicated RTX 5090. With 3 measured startup-only samples, no warmup,
    zero compute processes before/after the run, and the display disabled on
    the 5090, the linked candidate measured `0.839094x` `setup_wall_ms` and
    `0.839648x` `total_wall_ms` versus the dynamic-loader baseline. Route
    counters and strategy values stayed unchanged, so this remains a valid
    loader/linkage startup default and not a training-loop kernel parity fix.
  - 2026-06-19 added an opt-in token-weight vector4 startup candidate behind `NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT=1` / `NFN_TILE_CUDA_TOKEN_WEIGHT_VECTOR4_INIT=1`. It writes FP32 token weights and BF16 shadow bits with vectorized stores. Dedicated RTX 5090 startup-only paired timing rejected it as a default (`1.078363x` token-init time, `1.033082x` total wall), so the current int32 Tile initializer remains the default.
  - 2026-06-19 fixed startup-only validation so `--startup-only --max-steps 0` reaches real Tile-CUDA setup or the relevant Tile ops load error instead of failing the positive-step training validator. Use this for setup-only benchmark smoke commands; non-startup training still requires positive `max_steps`.
  - 2026-06-20 fixed the native-vs-native candidate wrapper so measured startup-only runs auto-gate `setup_wall_ms=1.000` instead of the absent `train_loop_wall_ms_per_step`. The corrected RTX 5090 startup-only rerun for `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1` failed on the right metric (`setup_wall_ms=1.008564x`), even though token init alone improved to `0.985085x`; keep threaded token initialization diagnostic-only and continue targeting either arena materialization or a stronger token init kernel before changing startup defaults.
  - 2026-06-20 added a distinct opt-in vector4 grid-stride token initializer behind `NFN_TILE_CUDA_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT=1` / `NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT=1`. It caps the startup kernel at 4096 CTAs and grid-strides vectorized FP32 plus BF16-shadow stores over the token table. The dedicated RTX 5090 startup-only paired gate rejected it as a default: token init improved to `0.985062x` mean, but setup wall measured `1.000096x` mean and a slower median, failing the strict `setup_wall_ms=1.000` gate. Keep it diagnostic-only.
  - 2026-06-23 split dense GPT native arena materialization diagnostics into `float_arena_cuda_malloc_wall_ms`, `float_arena_pointer_assign_wall_ms`, `uint16_arena_cuda_malloc_wall_ms`, `uint16_arena_pointer_assign_wall_ms`, `transformer_device_arena_cuda_malloc_wall_ms`, and `transformer_device_arena_pointer_assign_wall_ms` so startup candidates can distinguish CUDA allocation time from host pointer assignment inside the broader setup timing phases.
- [x] Add dense GPT native profile JSON `float_arena_request_stats` and `uint16_arena_request_stats` with ranked named suballocations so startup `cudaMalloc` candidates can be selected from measured arena contributors. The 2026-06-18 startup profile showed `setup.float_arena_materialize` at 159.510 ms for an 8.49 GB float arena and `setup.uint16_arena_materialize` at 119.426 ms for a 20.10 GB BF16/uint16 arena; the largest BF16 contributors were `stored_mlp_activation_bf16_arena` (10.87 GB), `stored_packed_attention_bf16_arena` (4.83 GB), `stored_packed_attention_ln1_bf16_arena` (1.11 GB), `stored_residual1_bf16_arena` (1.11 GB), and `lm_head_bf16_logits` (824 MB).
- [x] Add grouped arena-family profile fields (`family_count`, `top_families`, `top_family_elements`, and `top_family_bytes`) to the dense GPT native `float_arena_request_stats` and `uint16_arena_request_stats` JSON. A 2026-06-18 startup-only probe showed the float arena grouped as `transformer_lm_buffer` (3.70 GB across 42 requests) and `block.*.persistent_output` (2.21 GB across 11 requests), while the BF16 arena remained dominated by `stored_mlp_activation_bf16_arena` (10.87 GB) and `stored_packed_attention_bf16_arena` (4.83 GB). A refreshed `NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD=0` paired candidate measured `1.001432x` train-loop wall and `0.998574x` tokens/sec versus default, so no kernel default changed in this slice.
- [x] Fix dense GPT native startup so stored-MLP LayerNorm stats actually reuse the combined float arena instead of silently overwriting the arena suballocation with a second standalone `cudaMalloc`. Runtime JSON now reports `stored_mlp_layer_norm_stats_standalone_cuda_malloc_count`, which is `0` on the default `NFN_NATIVE_GPT_FLOAT_STATS_ARENA=1` path. Verification: rebuilt `build/nfn_gpt_native_train`, ran a one-step RTX 5090 smoke that reported `stored_mlp_layer_norm_stats_elements: 1572864`, `stored_mlp_layer_norm_stats_bytes: 6291456`, and standalone malloc count `0`; `python -m pytest tests/test_native_gpt2.py -q` passed; `python tools/check_native_no_torch_deps.py` passed. A short 5-step parity sample still measured NeuralFn at `1.042290x` train-loop wall versus llm.kittens, so this removes startup allocator waste but does not close the remaining GEMM/TK throughput gap.
- [x] Replace the opaque dense GPT native `transformer_lm_buffer` float-arena request name with per-buffer names for the main transformer-LM globals. The 2026-06-18 startup-only profile now reports concrete float families: `block.*.persistent_output` (2.21 GB), `mlp.fc.grad_out` (805 MB), and activation-sized `attention.grad_out`, `embedding_residual`, `ln1.grad_input`, `ln2.grad_input`, and `lnf.grad_input` buffers (201 MB each), so the next startup/memory candidates no longer require source-code decoding.
- [x] Elide the dense GPT native FP32 `mlp.fc.grad_out` arena buffer when the default BF16-only MLP dGELU handoff covers every trained block. This removes `805,306,368` float-arena bytes at the default shape and reports `block_backward_mlp_fc_grad_out_float_buffer_elided: true`; the 2026-06-18 dedicated RTX 5090 same-script benchmark versus `NFN_NATIVE_GPT_ELIDE_MLP_DGELU_FLOAT_GRAD=0` measured `0.969357x` train-loop wall time, `1.031616x` tokens/sec, and `0.965683x` setup wall time.
  - Same-script llm.kittens parity after this change: the 2026-06-18 3-step dedicated RTX 5090 check measured llm.kittens at `2469.283333 ms/step` and NeuralFn at `2551.806667 ms/step`, or `1.033420x` train-loop wall time and `0.967534x` tokens/sec versus `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh`; remaining parity work is still required.
- [x] Add diagnostic opt-in `NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUTS=1` for dense GPT scratch-recompute inter-block persistent outputs. It stores earlier block outputs as BF16, restores each backward input through one FP32 scratch buffer, and reports BF16 store/restore counts plus `fp32_persistent_block_output_*_elided`; at the default shape it elides `2,214,592,512` FP32 bytes while adding `1,107,296,256` BF16 bytes plus one FP32 restore scratch. Keep it rejected as a default because the 2026-06-18 dedicated RTX 5090 same-script benchmark measured `1.021212x` train-loop wall time and `0.979238x` tokens/sec versus the current default, despite better setup (`0.974595x`) and float-arena materialization (`0.896011x`).
  - Rechecked after the latest parity baseline and keep it diagnostic-only: the 2026-06-18 startup-only 3-sample run improved setup wall to `0.965877x` and total startup to `0.966002x`, but the normal 3-step, 2-sample run regressed train-loop wall time to `1.016063x`, tokens/sec to `0.984201x`, and total wall to `1.012855x`.
  - 2026-06-24 added and rejected the narrower direct-LN1 BF16 block-input sub-route. `NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_INPUT_LN1_BACKWARD=1` only runs with BF16 persistent block outputs and lets LN1 residual backward consume cached BF16 inter-block outputs directly when saved packed-attention LN1 and residual1 BF16 caches make the FP32 restore unnecessary. The reproducible wrapper profile is `NFN_SM120_NATIVE_CANDIDATE_PROFILE=bf16_persistent_block_outputs_direct_ln1`; runtime JSON reports request/enabled flags plus `bf16_persistent_block_input_ln1_backward_count`. The CUDA 13.3 dedicated RTX 5090 3-step, 2-sample stage-timed gate proved route activation (`0 -> 264` BF16 LN1 uses) and improved `stage.block_backward.ln1_residual.total_ms` to `0.890572x`, but rejected the route at `1.029093x` train-loop wall, `1.002462x` steady-state CUDA-event step time, and `1.048736x` block backward.
- [x] Keep the next 2026-06-18 LM-head/default-route rechecks rejected after the MLP grad-out elision: `NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1` measured `1.030374x` train-loop wall time and `0.970525x` tokens/sec, `NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=0` measured `1.007590x` train-loop wall time and `0.992475x` tokens/sec, `NFN_NATIVE_GPT_STORE_MLP_BLOCKS=10` measured `1.057927x` train-loop wall time and `0.945250x` tokens/sec, and `NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_SHADOW=0` measured `1.007306x` train-loop wall time and `0.992750x` tokens/sec versus the current default. No default changed; remaining parity work should focus on new kernel work for the hot GEMM/TK buckets rather than these switches.
- [x] Keep 2026-06-18 LM-head/forward-shape retunes rejected: `--lm-head-row-chunk-size 16384` measured `1.016356x` train-loop wall time and `0.983916x` tokens/sec versus the 8192-row default over a 5-step, 3-sample same-script run, and `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=2304,65536,768,T,N` measured `1.035923x` train-loop wall time and `0.965325x` tokens/sec in a 3-step paired smoke. Keep the current LM-head chunk and TK QKV forward default.
  - Superseded on 2026-06-19 by wiring the no-bias BF16-output wrapper to the TK BF16 forward bridge and removing the old shape-specific default disable. The earlier `NFN_NATIVE_LINEAR_TK_FORWARD_ENABLE_SHAPE=50304,8192,768,T,N` rejection did not exercise the current wrapper path; the current rollback knob is `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=50304,8192,768,T,N`.
  - Rechecked the MLP TK-forward fallback probes after the CUDA 13.3 shape profile. Disabling `768,65536,3072,T,N` measured `1.003165x` train-loop wall time and `0.996849x` tokens/sec over a 5-step, 3-sample dedicated RTX 5090 same-script run. Disabling `3072,65536,768,T,N` fell back to an unusably slow route and measured `13.695228x` train-loop wall time and `0.073019x` tokens/sec. Keep both TK forward routes enabled by default.
  - Rechecked candidate-only `--lm-head-row-chunk-size 32768` after the CUDA 13.3 reinstall and keep it rejected: the corrected 5-step, 3-sample native-vs-native run measured `1.559818x` train-loop wall time and `0.643975x` tokens/sec versus the 8192-row default.
  - Added and rejected diagnostic `NFN_NATIVE_GPT_LM_HEAD_REVERSE_CHUNKS=1`, which traverses row chunks from the end of the sequence toward the beginning to test cache-order alignment with llm.kittens classifier behavior. The CUDA 13.3 dedicated RTX 5090 5-step, 3-sample same-script run measured `1.000499x` train-loop wall time and `0.999506x` tokens/sec versus the default forward chunk order, so the default remains unchanged.
  - Rejected `NFN_NATIVE_GPT_LM_HEAD_DWEIGHT_BEFORE_DHIDDEN=1` as a default LM-head ordering route after the CUDA 13.3 reinstall: the dedicated RTX 5090 native-vs-native 5-step, 3-sample run measured `1.000701x` train-loop wall time and `0.999303x` tokens/sec versus current default, so the LM-head order stays unchanged.
- [x] Re-check the reference-alignment Tile ops build with `NFN_TILE_CUDA_EXTRA_NVCC_FLAGS=-DLLMK_SM120_USE_CUBLASLT_GEMM` after the MLP grad-out elision and keep the NeuralFn default Tile bridge: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.001374x` mean train-loop wall time, `0.999244x` median train-loop wall time, and `0.998638x` tokens/sec versus the default library, so the candidate remains noise-level rather than a promoted build flag.
- [x] Add diagnostic opt-in `NFN_NATIVE_GPT_ELIDE_FLOAT_PROJECTION_OUTPUTS=1` for dense GPT scratch-tape FP32 projection outputs. It skips the unused `tape.attn_proj` and `tape.mlp_out` float buffers when BF16 projection-residual is active, saving two activation-sized float allocations (`402,653,184` bytes at `64 x 1024 x 768`), but it is rejected as a default because the 2026-06-18 dedicated RTX 5090 checks measured train-loop neutral and startup-wall neutral-to-slightly slower for the elided side.
- [x] Add GPT-2 `--train-transformer-lm` CUDA runtime/driver preflight JSON and fail before allocation when the driver is unavailable or older than the loaded runtime, so SM120 benchmarking has a clear native gate.
- [x] Teach the native C++ token resolver to accept llm.kittens `TinyStories_train.bin` / `TinyStories_val.bin` directly for `--tinystories`, with `NFN_LLM_KITTENS_TINYSTORIES_DIR` override and direct train-bin sibling validation inference, so GPT-2 startup can match `train-sm120.sh` without Python dataset scanning or raw-text shard materialization.
- [x] Fuse GPT-2 `--train-transformer-lm` token/target upload into one contiguous pinned-to-device uint16 arena copy and one `nfn_native_tile_uint16_to_int64` launch per microbatch, instead of one copy and one widening launch for tokens plus another pair for targets.
- [x] Add `SequentialTokenBatchSampler::next_into()` and use it in GPT-2 `--train-transformer-lm` train/validation loops so real batches write directly into pinned uint16 arenas without `TokenBatch` vector materialization or vector-to-pinned copies.
- [x] Suballocate GPT-2 `--train-transformer-lm` widened int64 token/target buffers and compact uint16 H2D staging from one aligned device token arena, reducing two token device startup `cudaMalloc` calls to one.
- [x] Skip explicit native GPT CLI exit-time `cudaFree` calls and runtime-library `dlclose()` by default with `NFN_NATIVE_GPT_SKIP_EXIT_CUDA_FREE=1`, relying on CUDA process teardown after JSON/checkpoint output and reporting `device_exit_cuda_free_elision_enabled`, `device_exit_cuda_free_skipped_count`, `runtime_library_dlclose_skipped_count`, and `timing.cleanup_wall_ms`; keep `NFN_NATIVE_GPT_SKIP_EXIT_CUDA_FREE=0` for explicit-free diagnostics. The 2026-06-18 dedicated RTX 5090 same-script checks measured startup-only total wall at `0.695283x` and one-step training total wall at `0.946923x` versus explicit frees, with train-loop time neutral at `0.999895x`.
- [x] Elide native dense GPT `--startup-only` post-train diagnostic D2H sample copies. Startup-only still synchronizes after setup, but skips the token-weight and clip-scale sample copies that only prove real optimizer-step mutation; runtime JSON reports `timing.post_train_diagnostic_samples_elided` and elided D2H counts for startup benchmark sidecars.
- [x] Reject extending the dense GPT lazy validation MLP scratch path to cover FP32 `ln2_out`: although the CUDA 13.3 startup probe removed `block0.ln2.out` from the float arena top requests and lowered requested float-arena elements to `1,869,841,165`, the dedicated RTX 5090 same-script startup-only benchmark measured `1.046171x` total wall time, `1.046179x` setup wall time, and `1.048377x` float-arena materialization versus the current default. Keep lazy validation scratch limited to `fc_out` and `act`.
- [x] Keep the CUDA 13.3 allocator and teardown retunes rejected: `NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC=1 NFN_NATIVE_GPT_SKIP_EXIT_CUDA_FREE=1` measured `1.142716x` setup wall and `1.142263x` total startup versus the default allocator path, while restoring explicit exit-time frees with `NFN_NATIVE_GPT_SKIP_EXIT_CUDA_FREE=0` measured `1.378794x` total startup. Leave default `cudaMalloc` arenas plus skip-exit-free enabled.
- [x] Replace GPT-2 `--train-transformer-lm` startup per-buffer zero fills for zero biases and AdamW state with one float-arena zero fill, eliding 369 zero-fill launches at the default 12-layer shape.
- [x] Fuse GPT-2 `--train-transformer-lm` nonzero constant parameter initialization through `nfn_native_tile_fill_many_values_float32`, reducing the default 12-layer startup path from 75 per-buffer nonzero fill launches to one descriptor-driven Tile launch.
- [x] Promote the GPT token-weight fast int32 Tile-index initializer as the default under CUDA 13.3 when the token table fits in int32. A fresh dedicated RTX 5090 startup-only 5-sample benchmark with `NFN_NATIVE_GPT_TOKEN_WEIGHT_FAST_INT32_INIT=1` measured `0.955863x` token-weight init time, `0.988331x` setup wall time, and `0.988772x` total startup versus the previous int64 Tile-index default; the normal one-step 3-sample run kept train-loop timing neutral at `1.000072x`. Use `NFN_NATIVE_GPT_TOKEN_WEIGHT_FAST_INT32_INIT=0` / `NFN_TILE_CUDA_TOKEN_WEIGHT_FAST_INT32_INIT=0` to restore the older int64 Tile-index route for bisection.
- [x] Replace full-vocab non-workspace token CE backward per-call `cudaMalloc` row-stat buffers with a cached process workspace in the raw trainer Tile ABI, exposing allocation/capacity counters for smoke tests and profiling.
- [x] Fuse GPT-2 `--train-transformer-lm` AdamW updates through `nfn_native_tile_adamw_step_many_with_device_scale_float32`, reducing the default 12-layer optimizer step from 148 per-buffer AdamW launches to one multi-buffer launch.
- [x] Make the dense native GPT optimizer route a hard optimized-kernel startup contract. Full transformer training now requires the many-tensor AdamW, BF16-primary/shadow AdamW, BF16-gradient AdamW, many-buffer sumsq, and device clip-scale Tile-CUDA symbols before setup proceeds; runtime JSON reports `optimized_optimizer_contract_loaded` and `optimized_optimizer_contract_error`, while `--check-tile-ops` reports `tile_ops_check.optimized_optimizer_contract_loaded` plus missing optimizer symbols so same-script gates can reject scalar or per-buffer optimizer fallbacks before dataset resolution.
- [x] Fuse GPT-2 `--train-transformer-lm` accumulation-gradient zeroing through `nfn_native_tile_fill_many_float32`, reducing the default 12-layer optimizer-step zeroing path from 148 per-buffer fill launches to one multi-buffer launch.
- [x] Fuse GPT-2 `--train-transformer-lm` gradient-clipping sumsq partial generation through `nfn_native_tile_sumsq_partials_many_float32`, reducing the default 12-layer optimizer-step clipping path from 148 per-buffer sumsq launches to one multi-buffer launch before the device clip-scale reduction.
- [x] Wire GPT-2 `--train-transformer-lm` opt-in BF16 QKV/MLP-FC dWeight staging to direct BF16-gradient clipping and BF16-primary AdamW descriptors, eliminating the staging flush when the profiling switch is enabled; keep it default-off because the same-script dedicated-RTX-5090 benchmark measured the direct BF16 candidate slower than the optimized float-gradient path.
- [x] Default GPT `--train-transformer-lm` token embedding/LM-head startup to the fast CUDA Tile power-of-two deterministic initializer, keep `NFN_NATIVE_GPT_TOKEN_WEIGHT_INIT_LEGACY_MOD17=1` for paired modulo-17 bisection, and report the selected initializer in runtime JSON.
- [x] Match llm.kittens dense GPT dWeight accumulation semantics by adding beta-capable Tile-CUDA dWeight ABI variants and making the first gradient-accumulation microbatch write with GEMM `beta=0`, then accumulate later microbatches with `beta=1`; keep `NFN_NATIVE_GPT_DWEIGHT_FIRST_MICROBATCH_BETA_ZERO=0` for paired bisection and report the active first-write-then-accumulate strategy in runtime JSON.
- [x] Fix the chunked tied LM-head variant of that first-write path so only the first LM-head row chunk of the first gradient-accumulation microbatch uses `beta=0`; later chunks now use `beta=1` and accumulate into `accum_grad_token_weight`, with runtime JSON reporting `lm_head_dweight_beta_zero_scope`.
- [x] Keep direct first-write cuBLASLt `BGRADB` bias-gradient output as an opt-in diagnostic rather than the default: the 2026-06-17 dedicated RTX 5090 same-script 3-sample check measured `NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1` at `0.999871x` train-loop wall time and `1.000129x` tokens/sec, but the 2026-06-18 re-check measured `1.000529x` train-loop wall time and `0.999486x` tokens/sec versus the current scratch-accumulate default. Use `NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1`, `NFN_NATIVE_GPT2_BGRAD_FIRST_WRITE_DIRECT=1`, or `NFN_TILE_CUDA_LINEAR_BGRAD_FIRST_WRITE_DIRECT=1` only for paired bisection.
- [x] Reuse one BF16 pack of the dense GPT MLP projection incoming gradient for both MLP projection dWeight+bias and fused dInput+dGELU through `nfn_native_tile_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_only_float32`; keep `NFN_NATIVE_GPT_REUSE_MLP_PROJ_BF16_GRAD_OUT=0` for paired bisection and report the active reused-BF16-grad-out strategy in runtime JSON.
- [x] Allow trainer-facing BF16/BF16 cuBLASLt GEMMs for larger dense GPT LM-head backward chunk shapes by default; keep `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_LARGE_SHAPES=0` or `NFN_NATIVE_LINEAR_BF16_CUBLASLT_LARGE_SHAPES=0` for paired bisection against the previous BF16 `cublasGemmEx` fallback.
- [x] Fuse GPT-2 `--train-transformer-lm` QKV projection split plus Q/K/V head reshape through `nfn_native_tile_split_qkv_to_heads_float32`, reducing the default forward layout path from four launches per block to one.
- [x] Fuse GPT-2 `--train-transformer-lm` SDPA backward Q/K/V head-gradient merge plus QKV gradient assembly through `nfn_native_tile_merge_heads_to_qkv_float32`, reducing the default backward layout path from four launches per block to one and removing the full trainer's row-major `grad_q`, `grad_k`, and `grad_v` scratch buffers.
- [x] Add `nfn_native_tile_scaled_dot_product_attention_backward_from_merged_grad_float32` and use it in GPT-2 `--train-transformer-lm` so SDPA backward reads row-major attention-output gradients directly, removing the pre-backward `reshape_heads` launch and `grad_attn_heads` scratch buffer from the full trainer.
- [x] Expose `nfn_native_tile_attention_tk_store_forward_workspace_bf16` and `nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_saved_tk_bf16_from_merged_grad_float32` so native trainer loops can store TK BF16 Q/K/V/O/LSE attention forward state per block and later run attention backward without graph-editor tensors or an extra forward recompute.
- [x] Wire GPT-2 `--train-transformer-lm` to the saved TK attention path behind `NFN_NATIVE_GPT2_STORE_ATTENTION_ACTIVATIONS=1`, with JSON counters for saved attention arena size and store/restore/backward use; keep it off by default because the workstation 64x1024 probe regressed from about 74.4k to about 12.6k tok/s.
- [x] Suballocate GPT-2 `--train-transformer-lm` AdamW, gradient-zero, gradient-clip, and parameter-fill descriptor tables from one device descriptor arena, reducing ten small startup descriptor `cudaMalloc` calls to one.
- [x] Pack GPT-2 `--train-transformer-lm` descriptor tables into one host descriptor arena and upload it with one H2D copy, reducing ten startup descriptor `cudaMemcpy` calls to one.
- [x] Live-validate GPT-2 `--train-transformer-lm` memory/runtime behavior at the SM120 default batch shape and compare throughput against `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh`.
  - 2026-06-16 dedicated RTX 5090 check: `tools/bench_native_gpt_sm120_parity.sh` with `NFN_SM120_PARITY_STEPS=10`, `NFN_SM120_PARITY_SAMPLES=1`, `CUDA_VISIBLE_DEVICES=0`, and idle-GPU guards wrote `/tmp/nfn_sm120_parity_10step_current.json`. llm.kittens averaged `2478.839 ms/step` and `212051.9 tok/s`; NeuralFn averaged `2735.720 ms/step` and `191646 tok/s`, or `1.103630x` train-loop time and `0.903769x` tokens/sec versus the reference.
  - 2026-06-17 dedicated RTX 5090 check after large-BF16 cuBLASLt and MLP grad-out reuse: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=15 bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2475.020 ms/step` and `212201.5 tok/s`; NeuralFn measured `2606.400 ms/step` and `201154 tok/s`, or `1.053082x` train-loop time and `0.947939x` tokens/sec versus the reference. The remaining gap is concentrated in `block_backward`, `lm_head_backward`, and `train.model_forward` stage buckets.
  - 2026-06-17 workflow update: the same parity wrapper now appends NeuralFn native `--profile-json` sidecars through `NFN_SM120_PARITY_PROFILE_DIR` by default without enabling CUDA-event stage timing, so same-script parity runs keep the measured command timing-neutral. Set `NFN_SM120_PARITY_STAGE_TIMING=1` for explicit attribution sidecars that carry `timing.stage_timing` buckets.
  - 2026-06-17 unprofiled dedicated RTX 5090 check: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=15 NFN_SM120_PARITY_PROFILE_DIR=none bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2477.527 ms/step` and `211956.7 tok/s`; NeuralFn measured `2611.690 ms/step` and `200747 tok/s`, or `1.054152x` train-loop time and `0.947113x` tokens/sec versus the reference.
  - 2026-06-17 refreshed dedicated RTX 5090 check after SDK runner cleanup: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=15 NFN_SM120_PARITY_PROFILE_DIR=/tmp/nfn_sm120_parity_profiles_after_auto_guard bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2474.848 ms/step` and `212204.6 tok/s`; NeuralFn measured `2614.030 ms/step` and `200567 tok/s`, or `1.056239x` train-loop time and `0.945159x` tokens/sec versus the reference. The refreshed profile still points at `block_backward` (`13016.7 ms`), `lm_head_backward` (`6361.27 ms`), `block_backward.attn_sdpa` (`2785.26 ms`), `block_backward.mlp_fc` (`2708.38 ms`), and `block_backward.qkv` (`2053.33 ms`) as the highest-value remaining buckets.
  - 2026-06-17 follow-up: the refreshed sidecar run above used the old harness behavior that implicitly enabled `NFN_NATIVE_GPT_STAGE_TIMING=1`, while a plain 5-step NeuralFn run without stage timing measured `174287 tok/s` with `stage_timing_enabled: false`. Re-run parity after this harness correction before treating sidecar-enabled throughput as canonical.
  - 2026-06-17 dedicated RTX 5090 post-alias-guard parity check with no sidecar profiling: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=15 NFN_SM120_PARITY_PROFILE_DIR=none bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2567.572 ms/step` and `206676.9 tok/s`; NeuralFn measured `2692.620 ms/step` and `194713 tok/s`, or `1.048703x` train-loop time and `0.942113x` tokens/sec versus the reference.
  - 2026-06-17 one-step stage/shape profile after that parity run wrote `/tmp/nfn_current_stage_shape_after_alias_guard.json`; the hot buckets remain `block_backward` (`1385.720 ms`), `lm_head_backward` (`723.511 ms`), `train.model_forward` (`684.852 ms`), `block_backward.mlp_proj` (`376.063 ms`), `block_backward.attn_sdpa.to_qkv` (`285.569 ms`), and `block_backward.qkv` (`214.069 ms`). Linear-shape counters still show 96-call transformer GEMMs plus 64-call LM-head chunks: TK BF16 forward shapes `2304x65536x768`, `768x65536x768`, `3072x65536x768`, `768x65536x3072`, LM-head `50304x8192x768`, cuBLASLt dWeight/dInput shapes for the same projections, and GEMMEx LM-head dHidden.
  - 2026-06-17 explicit-batch parity confirmation after pinning the NeuralFn candidate to `--train-batch-tokens 524288`: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=3 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=15 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_explicit_batch_3sample.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2626.415667 ms/step` and `199762.7 tok/s`; NeuralFn measured `2714.493333 ms/step` and `193199.666667 tok/s`, or `1.033656x` train-loop time and `0.967400x` tokens/sec versus the reference. The selected GPU reported zero compute processes for every paired sample.
  - 2026-06-17 post-4096-token-init retile parity check with no sidecar profiling: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=15 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_after_retile.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2555.070 ms/step` and `205539 tok/s`; NeuralFn measured `2676.750 ms/step` and `195867 tok/s`, or `1.047623x` train-loop time and `0.952943x` tokens/sec versus the reference.
  - 2026-06-17 post-MLP-residual-next-LN1-fusion parity check with no sidecar profiling: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=3 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=30 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_after_mlp_next_ln1_3sample.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2616.212333 ms/step` and `200607.766667 tok/s`; NeuralFn measured `2713.243333 ms/step` and `193238.666667 tok/s`, or `1.037126x` train-loop time and `0.963323x` tokens/sec versus the reference. The selected GPU reported zero compute processes for every paired sample.
  - 2026-06-18 attention-backward section profile with `NFN_NATIVE_GPT_ATTENTION_BACKWARD_SECTION_TIMING=1` over one optimizer step reported `attention_backward_tk_timing_us: 234948`, `attention_backward_dprep_timing_us: 30604`, and `attention_backward_tk_launch_count: 96`; `block_backward.attn_sdpa.to_qkv` remained `282.329 ms`, so the current SDPA-backward bucket is dominated by the TK backward kernel rather than host graph/tensor flow.
  - 2026-06-18 post-diagnostic-default refresh with the dedicated RTX 5090 idle and no sidecar profiling: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=30 NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_after_diag_defaults_20260618.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2445.644 ms/step` and `214655.2 tok/s`; NeuralFn measured `2549.320 ms/step` and `205658 tok/s`, or `1.042392x` train-loop time and `0.958085x` tokens/sec versus the reference. A current attribution run with `NFN_NATIVE_GPT_STAGE_TIMING=1 NFN_NATIVE_GPT_LINEAR_SHAPE_STATS=1` wrote `/tmp/nfn_current_stage_shape_20260618_after_diag_defaults.json` and showed the largest buckets as `block_backward` (`1277.15 ms`), `lm_head_backward` (`629.661 ms`), `train.model_forward` (`631.122 ms`), `block_backward.mlp_proj` (`342.920 ms`), `block_backward.mlp_fc` (`265.076 ms`), and `block_backward.attn_sdpa.to_qkv` (`263.480 ms`).
  - 2026-06-18 post-native-only-CLI refresh with the dedicated RTX 5090 idle and no sidecar profiling: `NFN_SM120_PARITY_STEPS=3 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=30 NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_after_cli_native_only_20260618.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2465.030 ms/step` and `212701 tok/s`; NeuralFn measured `2548.513333 ms/step` and `205723 tok/s`, or `1.033867x` train-loop time and `0.967193x` tokens/sec versus the reference. The selected GPU reported zero compute processes and `0%` utilization before the paired sample.
  - 2026-06-18 one-step stage/shape profile after the native-only CLI change wrote `/tmp/nfn_current_stage_shape_20260618_after_cli_native_only.json`; the hot buckets remain `block_backward` (`1291.990 ms`), `train.model_forward` (`639.758 ms`), `lm_head_backward` (`634.402 ms`), `block_forward` (`632.120 ms`), `block_backward.mlp_proj` (`347.437 ms`), `block_forward.attention` (`294.776 ms`), `block_backward.mlp_fc` (`269.836 ms`), `block_backward.attn_sdpa.to_qkv` (`266.074 ms`), and `lm_head_backward.logits` (`221.139 ms`).
  - 2026-06-18 current attention-section profile after the BF16-output cuBLASLt and BF16/BF16 split-BGRADB diagnostics wrote `/tmp/nfn_attention_section_current_20260618.json`; one optimizer step reported `attention_backward_tk_timing_us: 237105`, `attention_backward_dprep_timing_us: 31238`, and `attention_backward_tk_launch_count: 96`. The largest stage buckets were `block_backward` (`1303.880 ms`), `lm_head_backward` (`737.801 ms`), `train.model_forward` (`635.762 ms`), `block_backward.mlp_proj` (`345.095 ms`), `block_backward.attn_sdpa.to_qkv` (`280.175 ms`), `block_backward.mlp_fc` (`266.960 ms`), `lm_head_backward.logits` (`259.179 ms`), and `lm_head_backward.dhidden` (`236.576 ms`).
  - 2026-06-18 MLP projection attribution now records `block_backward.mlp_proj.grad_out_bf16` inside `NFN_NATIVE_GPT_STAGE_TIMING=1` profiles. The one-step dedicated RTX 5090 profile `/tmp/nfn_mlp_grad_out_timing_20260618.json` measured the BF16 pack at `22.774 ms` total (`96` calls, `0.237 ms` average), versus `174.294 ms` for `block_backward.mlp_proj.dweight_bias` and `172.114 ms` for `block_backward.mlp_proj.dinput`, so the projection-gradient bridge is measurable but the remaining MLP gap is still dominated by GEMM/TK kernels.
  - 2026-06-18 `NFN_TILE_CUDA_EXTRA_NVCC_FLAGS=-DLLMK_SM120_ATOMIC_DQ` now builds through a dedicated packed-QKV candidate wrapper that allocates float dQ scratch, zeroes it per batch chunk, launches the internal SM120 atomic-dQ backward, and re-packs the Q slice into the BF16 packed `dQKV` scratch before QKV dWeight handoff. The one-step dedicated RTX 5090 profile regressed TK backward timing to `597872 us` versus the default roughly `237105 us`, and the same-script 5-step, 2-sample candidate benchmark measured `1.134435x` train-loop wall time and `0.881527x` tokens/sec versus default. Keep the non-atomic packed-gradient backward as the default.
  - 2026-06-19 CUDA 13.3 parity refresh after the HD64 dprep default fix measured llm.kittens at `2464.946 ms/step` and `212666.8 tok/s`; NeuralFn measured `2552.820 ms/step` and `205377 tok/s`, or `1.035649x` train-loop wall time and `0.965722x` tokens/sec versus `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh`. The one-step stage/shape profile still points to `block_backward` (`1359.650 ms`), `lm_head_backward` (`849.385 ms`), `train.model_forward` (`659.518 ms`), and LM-head shapes `50304x8192x768`, `768x8192x50304`, and `768x50304x8192` as the largest remaining buckets.
  - 2026-06-19 CUDA 13.3.33 follow-up after the WSL reinstall measured the current default one-step stage/shape profile at `2865.24 ms` train-loop wall, `546.399 ms` setup wall, and `133.577 ms` token-weight init. The largest buckets were `block_backward` (`1337.510 ms`), `lm_head_backward` (`824.768 ms`), `train.model_forward` (`674.512 ms`), `block_backward.mlp_proj` (`354.115 ms`), `lm_head_backward.logits` (`328.920 ms`), `lm_head_backward.dhidden` (`244.192 ms`), and `lm_head_backward.dweight` (`177.415 ms`). Shape stats still show LM-head logits `50304x8192x768` and dHidden `768x8192x50304` on BF16 GEMMEx, while dWeight `768x50304x8192` uses cuBLASLt; the next real parity task remains a fused/cooperative row-chunked LM-head backward kernel or a materially better GEMM/TK route.
  - 2026-06-19 after reinstalling the WSL CUDA 13.3 toolkit, the GPU-enabled failed-test rerun is clean: `env NFN_TILE_CUDA_TEST=1 python -m pytest --lf -q -rs` passed with `1167 passed, 20 warnings, 468 subtests passed in 436.93s`. `nvidia-smi` reported CUDA UMD `13.3` on the dedicated RTX 5090, display disabled, and zero compute processes before the run. The same-script parity refresh `NFN_SM120_PARITY_STEPS=5 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=30 NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_after_cuda133_retest.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2505.190 ms/step` and `209376.4 tok/s`; NeuralFn measured `2589.880 ms/step` and `202437 tok/s`, or `1.033806x` train-loop wall time and `0.966857x` tokens/sec with zero compute processes before and after the paired sample. The current attribution file `/tmp/nfn_stage_shape_after_cuda133_retest.json` reports `lm_head_backward` at `844.033 ms`, with `lm_head_backward.logits` at `343.156 ms`, `lm_head_backward.dhidden` at `247.225 ms`, and `lm_head_backward.dweight` at `179.878 ms`; logits and dHidden are still BF16 GEMMEx on `50304x8192x768` and `768x8192x50304`, so the open performance task remains real kernel work rather than stale CUDA test failure cleanup.
  - 2026-06-19 added and rejected the opt-in LM-head two-stream dHidden/dWeight scheduler (`NFN_NATIVE_GPT_LM_HEAD_CONCURRENT_DHIDDEN_DWEIGHT=1`). The path records a CUDA event after CE+dlogits, waits from two non-blocking streams, launches dHidden and dWeight on separate streams, and synchronizes before the next row chunk; runtime JSON reports the requested/available/enabled fields plus `lm_head_dhidden_dweight_schedule_strategy`. A one-step smoke verified the candidate active with `lm_head_backward.dhidden_dweight_concurrent`, but the dedicated RTX 5090 5-step, 3-sample same-script native-vs-native run measured `1.004893x` train-loop wall time and `0.995133x` tokens/sec versus default. A CUDA 13.3 reinstall follow-up over 5 steps and 2 samples still rejected it at `1.002028x` train-loop wall, `0.997989x` tokens/sec, and `1.008876x` LM-head backward stage time. Keep it diagnostic-only; the remaining LM-head parity work still needs a fused/cooperative classifier-backward kernel or materially better GEMM route, not stream overlap around the existing GEMMs.
  - 2026-06-19 added paired benchmark terminal reporting for backend counters including `linear_tk_gemm_count`, `linear_cublaslt_gemm_count`, `linear_bf16_gemm_count`, and attention TK launch counts so active candidate routes are visible without opening sidecar JSON.
  - 2026-06-19 keep the current LM-head dInput and cuBLASLt hot-shape bisections rejected on CUDA 13.3: `NFN_NATIVE_LINEAR_TK_DINPUT=1` measured `1.053838x` train-loop wall time and `0.949018x` tokens/sec, and `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,50304,8192,N,T,0` measured `1.003965x` train-loop wall time and `0.996050x` tokens/sec. The older `NFN_NATIVE_LINEAR_TK_FORWARD_ENABLE_SHAPE=50304,8192,768,T,N` evidence is superseded by the later BF16-output wrapper wiring and default TK bridge promotion.
  - 2026-06-19 keep the CUDA 13.3.33 LM-head cuBLASLt expansion rejected: `NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1 NFN_NATIVE_LINEAR_BF16_OUTPUT_CUBLASLT=1` moved more LM-head work to cuBLASLt but measured `1.027700x` train-loop wall time and `0.973050x` tokens/sec versus default over the same-script native-vs-native run. Keep logits/dHidden on the current default routes until a fused/cooperative classifier-backward kernel or a better shape-specific GEMM route exists.
  - 2026-06-19 added default-off BF16 `cublasGemmEx` algorithm bisection through `NFN_NATIVE_LINEAR_BF16_GEMM_EX_ALGO` / `NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_ALGO` and exact-shape `*_GEMM_EX_ALGO_SHAPE=m,n,k,opA,opB,algo`. Use it to test LM-head GEMMEx shapes in the same native-vs-native script without rebuilding. This does not close parity by itself; promote no algorithm unless a paired dedicated-RTX-5090 run beats the unset default. The LM-head dHidden algorithm-0 probe `768,8192,50304,N,N,0` looked favorable over 2 samples (`0.993367x` train-loop wall) but the 5-sample confirmation was effectively flat (`0.999739x` mean train-loop wall, `1.000326x` tokens/sec), so it remains diagnostic-only. A follow-up algorithm sweep also kept algorithms 1-4 rejected: algorithm 1 was one-sample noise (`0.998467x` wall), algorithms 2 and 3 were slower (`1.023701x` and `1.010416x` wall), and algorithm 4 regressed in the 5-sample confirmation (`1.002073x` wall, `0.997933x` tokens/sec).
  - 2026-06-19 keep `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1` rejected after the CUDA 13.3.33 startup-only recheck: total startup regressed to `1.044736x`, setup wall time to `1.044749x`, and token-weight init time to `1.133043x` versus the current fast int32 Tile-index initializer. Keep the threaded initializer diagnostic-only.
  - 2026-06-19 rechecked `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1` on the post-fused-embedding build with five one-step paired samples and kept it rejected: setup wall time regressed to `1.034063x`, token-weight init to `1.027574x`, and total wall to `1.001269x` despite a noisy `0.995071x` one-step train-loop mean.
  - 2026-06-19 post-CE-vector-load recheck keeps the remaining LM-head shortcut routes rejected: `NFN_NATIVE_LINEAR_BF16_OUTPUT_CUBLASLT=1 NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=50304,8192,768,T,N,0` moved logits to cuBLASLt but measured `1.000302x` train-loop wall time and `0.999702x` tokens/sec, while `NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS=1` measured `12.602812x` train-loop wall time and `0.079348x` tokens/sec because it keeps the same GEMM count and adds a full resident BF16 logit tape. Do not revisit full-logit reuse for the default shape; implement a fused/cooperative row-chunked classifier-backward kernel instead.
- [x] Default dense GPT native embedding forward to a fused Tile-CUDA token embedding + position embedding + scaled residual ABI (`nfn_native_tile_token_position_embedding_residual_u16_float32` on the direct-u16 token path), with `NFN_NATIVE_GPT_FUSE_EMBEDDING_RESIDUAL=0` / `NFN_NATIVE_GPT2_FUSE_EMBEDDING_RESIDUAL=0` / `NFN_TILE_CUDA_FUSE_EMBEDDING_RESIDUAL=0` for old-path bisection. The fused path elides `token_out` and `position_out`, removing `402653184` FP32 arena bytes at the default `64 x 1024 x 768` shape; the dedicated RTX 5090 5-step, 5-sample same-script run measured the old opt-out path at `1.001318x` train-loop wall time and `0.998685x` tokens/sec versus the fused default.
- [x] Default the dense GPT native BF16 projection residual-add helper to a 768-wide CUDA specialization (`linear_bias_residual_add_bf16_linear_dim768_float32_kernel`) with `NFN_TILE_CUDA_DIM768_BF16_RESIDUAL_ADD=0` / `NFN_NATIVE_GPT_DIM768_BF16_RESIDUAL_ADD=0` / `NFN_NATIVE_GPT2_DIM768_BF16_RESIDUAL_ADD=0` for old generic-helper bisection. The dedicated RTX 5090 5-step, 5-sample same-script run measured the specialized default at `0.998835x` train-loop wall time, `1.001172x` tokens/sec, and `0.997518x` median total wall time versus the generic path.
- [x] Default dense GPT native BF16 classifier/CE scans to vectorized 8x BF16 row loads, matching the llm.kittens fused-classifier row-read shape while keeping the raw C ABI stable. Use `NFN_NATIVE_GPT_CE_BF16_VEC_LOADS=0`, `NFN_NATIVE_GPT2_CE_BF16_VEC_LOADS=0`, or `NFN_TILE_CUDA_CE_BF16_VEC_LOADS=0` for scalar-load bisection. The CUDA 13.3.33 dedicated RTX 5090 5-step, 5-sample same-script run measured vector loads at `0.998750x` train-loop wall time and `1.001257x` tokens/sec versus scalar loads; the inverse 3-sample opt-out run measured scalar loads at `1.002740x` train-loop wall time and `0.997270x` tokens/sec versus the promoted default.
- [x] Recheck and keep `NFN_NATIVE_GPT_CE_BF16_VEC_STORES=1` rejected after the BF16 vector-load default and CUDA 13.3.33 WSL reinstall. The dedicated RTX 5090 5-step, 5-sample same-script run measured `0.999390x` mean train-loop wall time and `1.000618x` mean tokens/sec, but median train-loop wall regressed to `1.000326x` and paired command wall was effectively flat/slower, so scalar dlogit stores stay as the default while the 128-bit streaming-store switch remains diagnostic-only.
- [x] Default dense GPT native forward to fuse MLP projection residual into the next block's LN1 stats/BF16 output through `nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_bf16_norm_float32`, with `NFN_NATIVE_GPT_FUSE_MLP_RESIDUAL_NEXT_LN1=0` for bisection. The 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured `0.995763x` train-loop wall time and `1.004256x` tokens/sec versus opt-out; one-step stage probe reported `mlp_residual_next_ln1_fusion_count: 88`.
- [x] Add and reject default-off MLP projection backward order bisection (`NFN_NATIVE_GPT_MLP_PROJ_DINPUT_BEFORE_DWEIGHT=1`). It mirrors the llm.kittens `matmul_backward` dInput-before-dWeight consumer order for the dense GPT MLP projection, but the 2026-06-18 dedicated RTX 5090 5-step, 3-sample same-script run measured `1.000405x` train-loop wall time and `0.999602x` tokens/sec versus the current dWeight+bias-first default, so it remains diagnostic-only.
  - 2026-06-25 reran the linked-trainer profile with stage timing. The route counter moved `0->288` and mean train-loop wall stayed near-flat at `0.999180x`, but the target `stage.block_backward.mlp_proj.dinput.total_ms` bucket regressed to `1.101843x` and total MLP projection backward regressed to `1.001268x`, so the dWeight+bias-first default remains correct.
- [x] Keep MLP FC backward dInput-before-dWeight ordering rejected by default
  (`NFN_NATIVE_GPT_MLP_FC_DINPUT_BEFORE_DWEIGHT=1`). It mirrors the
  llm.kittens dInput-before-dWeight consumer order for dense GPT MLP FC. The
  original 2026-06-18 dedicated RTX 5090 5-step, 3-sample same-script run
  measured `1.000858x` train-loop wall time and `0.999153x` tokens/sec, so it
  was first kept diagnostic-only.
  - 2026-06-25 reran the profile after the 512-thread bias reducer became the
    default. The first rerun moved
    `block_backward_mlp_fc_dinput_before_dweight_count: 0 -> 288` and improved
    average train-loop wall to `0.979771x`, but still missed steady-state
    timing. The current CUDA 13.3.33 3-step, 2-sample rerun passed the
    whole-loop gates: train-loop wall `0.979044x`, steady-state CUDA-event
    timing `0.997216x`, tokens/sec `1.021478x`, block backward `0.960721x`,
    and LM-head backward `0.998613x`. The named MLP FC bucket regressed to
    `1.063824x`, so the profile was temporarily promoted as a whole-loop gate.
  - 2026-06-26 reran the same profile after the CUDA reinstall. The route still
    proved execution by moving
    `block_backward_mlp_fc_dinput_before_dweight_count: 0 -> 288` and kept
    train-loop wall slightly faster at `0.998065x`, but missed strict gates:
    steady-state CUDA-event timing `1.001167x`, block backward `1.001447x`,
    LM-head backward `1.000127x`, MLP projection backward `1.004199x`, and MLP
    FC backward `1.003817x`. The dense GPT default is restored to
    dWeight+bias-before-dInput.
- [x] Keep attention projection backward dInput-before-dWeight ordering rejected by default (`NFN_NATIVE_GPT_ATTN_PROJ_DINPUT_BEFORE_DWEIGHT=1`). It mirrors the llm.kittens dInput-before-dWeight consumer order for dense GPT attention projection, but the 2026-06-24 CUDA 13.3 dedicated RTX 5090 rebuilt-binary 5-step, 3-sample same-script gate proved `block_backward_attn_proj_dinput_before_dweight_count: 0 -> 480` and rejected default promotion because train-loop wall regressed to `1.001501x`, `stage.lm_head_backward.total_ms` to `1.000290x`, `stage.block_backward.total_ms` to `1.003886x`, `stage.block_backward.mlp_proj.total_ms` to `1.002417x`, and `stage.block_backward.attn_proj.total_ms` to `1.081569x`.
- [x] Keep the full combined projection-order route rejected after the CUDA 13.3 reinstall. Enabling the MLP projection, MLP FC, and attention projection dInput-before-dWeight switches together measured `1.000696x` train-loop wall time and `0.999321x` tokens/sec over a 5-step, 2-sample same-script dedicated RTX 5090 run, so all three dWeight-first projection schedules remain the defaults.
- [x] Add and reject default-off attention dprep-only BF16 grad-out bisection (`NFN_NATIVE_GPT_BF16_ATTENTION_DPREP_GRAD_OUT=1`). The candidate keeps attention projection dInput on the default float output path, packs dO to BF16 just before packed-attention dprep/backward, and reports `attention_backward_grad_out_dtype: "bf16-dprep-pack"`. A one-step attribution run reduced dprep timing from roughly `30.5 ms` to `24.8 ms` but added a `22.473 ms` pack, and the 2026-06-18 dedicated RTX 5090 same-script 10-step, 3-sample benchmark measured `1.007803x` train-loop wall time and `0.992260x` tokens/sec versus default, so the current float-dO dprep route remains the default.
  - CUDA 13.3 recheck after the WSL toolkit/driver update still rejects this route: the dedicated RTX 5090 native-vs-native 10-step, 3-sample run measured `1.006112x` train-loop wall time and `0.993932x` tokens/sec versus the current default.
- [x] Add and reject default-off BF16-output cuBLASLt LM-head logits bisection (`NFN_NATIVE_LINEAR_BF16_OUTPUT_CUBLASLT=1` / `NFN_TILE_CUDA_LINEAR_BF16_OUTPUT_CUBLASLT=1`). A one-step shape-stat smoke moved `50304,8192,768,T,N` to `cublaslt`, but the 2026-06-18 dedicated RTX 5090 10-step, 3-sample paired benchmark measured `1.000629x` train-loop wall time and `0.999382x` tokens/sec versus the current BF16 GEMMEx fallback, so it remains diagnostic-only.
- [x] Add and reject default-off BF16/BF16 split dWeight+bias bisection (`NFN_NATIVE_GPT_FUSE_BF16_BF16_DWEIGHT_BGRAD=0` / `NFN_TILE_CUDA_LINEAR_BF16_BF16_BGRAD=0`). The split path keeps block dWeight on the GEMM route and separates bias reduction instead of falling to tiled dWeight, but the 2026-06-18 dedicated RTX 5090 10-step, 3-sample paired benchmark measured `1.033067x` train-loop wall time and `0.968003x` tokens/sec versus the current fused BGRADB default.
- [x] Keep rejected same-script SM120 kernel bisections documented so future work does not retest slower paths: `NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1` (`1.016116x` train-loop time), `NFN_NATIVE_GPT_BF16_BLOCK_DWEIGHT_STAGING=1` (`1.028320x`), `NFN_NATIVE_GPT_DWEIGHT_FIRST_MICROBATCH_BETA_ZERO=0` (`1.009501x`), compile-time `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` (`0.999900x`, noise-equivalent over four samples), `NFN_NATIVE_GPT_FUSE_QKV_BIAS_TK_GEMM=0` (`1.016504x`), `NFN_NATIVE_GPT_BF16_QKV_DWEIGHT=0` (`1.009870x` train-loop time and `0.990246x` tokens/sec, slower), `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX=0` (`1.006262x`), `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX=2` (`1.007752x`), `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX=3` (`1.005321x`), `NFN_NATIVE_LINEAR_CUBLASLT_DESCRIPTOR_CACHE=0` (`1.003259x`), `NFN_NATIVE_LINEAR_TK_FLOAT_OUT=1` (`0.999929x`, noise-equivalent), `NFN_NATIVE_GPT_CE_BF16_THREADS=512` (`0.999774x`, noise-equivalent), `NFN_NATIVE_GPT_CE_BF16_THREADS=256` (`1.011218x`), `NFN_NATIVE_GPT_CE_BF16_VEC_STORES=1` (`1.000271x`, noise-equivalent/slightly slower), `NFN_NATIVE_GPT_CE_BF16_EXP2=1` (`1.000721x` train-loop time and `0.999293x` tokens/sec, noise-equivalent/slightly slower), `NFN_NATIVE_GPT_BF16_LM_HEAD_LOSS=0` (`0.992803x` mean but `1.000689x` median train-loop time over five samples, not promoted), a full LM-head BF16 logit/dlogit tape prototype (`6593445888` extra logit bytes, startup fit but the one-step candidate saturated the RTX 5090 at about `31899/32607 MiB` and exceeded the useful paired-benchmark window), `NFN_NATIVE_GPT_CUDA_MEMSET_GRAD_ZERO=0` (`1.003087x` train-loop time and `0.996930x` tokens/sec, slower), `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE=0` (`1.004427x`), `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_STATS=0` (`1.008131x`), `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=11` (`1.004875x`), `NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP=128` (`0.999506x` mean but `1.001655x` median train-loop time and `1.001502x` total wall time, not promoted), `NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP=32` (`1.003150x` train-loop time and `0.996880x` tokens/sec, slower), `NFN_NATIVE_GPT_ELIDE_QKV_FLOAT_GRAD_SCRATCH=0` (`1.000265x` train-loop time and `0.999736x` tokens/sec, noise-equivalent/slightly slower), `NFN_NATIVE_GPT_STORE_MLP_BLOCKS=11` (`1.029136x`), `NFN_NATIVE_GPT_FUSE_ATTENTION_RESIDUAL_LN2=0` (`1.021641x`), `NFN_NATIVE_GPT_FUSE_LN2_BF16_OUT=0` (`1.011993x`), `NFN_NATIVE_GPT_ELIDE_LN2_BF16_NORM_FLOAT_STORE=0` (`1.004445x` train-loop time and `0.995581x` tokens/sec, slower), historical `NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=128` checks before the CUDA 13.3 rerun (`1.004171x`, now superseded by the promoted 2026-06-23 same-script gate), `NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=512` (`1.022811x`), `NFN_NATIVE_GPT_LM_HEAD_ROW_CHUNK_SIZE=4096` (`0.999703x`, noise-equivalent), `--lm-head-row-chunk-size 12288` (`1.006439x` train-loop time and `0.993616x` tokens/sec, slower), `--lm-head-row-chunk-size 16384` (`1.018129x`), `--lm-head-row-chunk-size 32768` (`0.999895x` train-loop time but `1.007049x` total wall time and `1.007987x` LM-head backward, not promoted), `NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=0` (`1.009504x` train-loop time and `1.039531x` LM-head backward, slower), `NFN_NATIVE_GPT_BF16_PROJECTION_RESIDUAL=0` (`1.015959x`), `NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF=1` (`1.000141x`, noise-equivalent), `NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC=1` (`1.005702x` train-loop time and `1.125753x` setup wall time), `NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="-DLLMK_SM120_USE_CUBLASLT_GEMM"` (`1.014933x`, slower), `NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1` with `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX=1` for the LM-head dHidden shape (`1.028831x` train-loop time and `0.971991x` tokens/sec, slower), a primary float+uint16 startup arena candidate (`1.075212x` setup wall time and `1.047410x` total startup wall time, slower), and a narrow LM-head dHidden cuBLASLt large-`k` probe that still fell back to `cublas_gemmex_bf16` for `m=768,n=8192,k=50304`.
  - 2026-06-25 follow-up: the default SM120 Tile ops build now defines `LLMK_SM120_USE_TK_FUSED_DGELU_DINP` and `LLMK_SM120_APPROX_DGELU_TANH=1` in `build/libnfn_native_train_tile_ops.so`, matching the documented linked-trainer baseline instead of relying on the diagnostic `_tk` sidecar for those macros.
  - 2026-06-25 telemetry fix: dense GPT `--check-tile-ops` / plan JSON now
    queries `linear_tk_sm120_*` config symbols from the loaded Tile ops library
    instead of hardcoding them to absent, so benchmark preflights can see
    compile-time SM120 route settings before running training.
  - 2026-06-25 rebuild-dependency fix: linked GPT trainer builds, SM120
    parity/candidate wrappers, linear/LM-head microbench wrappers, and
    `tools/check_native_no_torch_deps.py` now include
    `tools/build_native_train_tile_ops.sh` as a Tile ops dependency. Default
    compile-flag changes now force `libnfn_native_train_tile_ops.so` rebuilds
    before normal linked training or candidate timing.
- [x] Add `NFN_SM120_NATIVE_CANDIDATE_PROFILE=packed_attention_bwd_batch_128` as a rejected wrapper profile. A 2026-06-25 CUDA 13.3.33 dedicated RTX 5090 attention-section gate changed `attention_backward_tk_batch_cap` from `64` to `128`, but rejected it at `1.013207x` train-loop wall time, `1.000470x` steady-state CUDA-event timing, `1.029008x` block backward, `1.002850x` attention TK timing, and `1.000088x` attention dprep timing; keep the default cap at 64.
- [x] Add `NFN_SM120_NATIVE_CANDIDATE_PROFILE=packed_attention_bwd_batch_32` as a rejected wrapper profile. A 2026-06-25 CUDA 13.3.33 dedicated RTX 5090 attention-section gate changed `attention_backward_tk_batch_cap` from `64` to `32` and doubled `attention_backward_tk_launch_count` from `288` to `576`; dprep timing improved to `0.943271x`, but the smaller chunk rejected default promotion at `1.010819x` train-loop wall time, `1.009021x` steady-state CUDA-event timing, `1.077143x` attention `to_qkv`, and `1.066848x` attention TK timing. Keep the default cap at 64; the next attention work needs a faster TK backward kernel, not more chunk splitting.
- [x] Add `NFN_SM120_NATIVE_CANDIDATE_PROFILE=cublaslt_attn_proj_dweight_h0_65536` as a rejected/no-op wrapper profile. The 2026-06-25 CUDA 13.3.33 dedicated RTX 5090 stage-timed gate for `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,768,65536,N,T,0` showed attractive-looking attention-projection dWeight timing, but changed no tracked route counters, strategy values, or cuBLASLt plan-cache entries and still failed strict steady-state CUDA-event, LM-head backward, and MLP-projection gates. Treat it as timing noise until a real route or plan change is visible.
- [x] Reject `NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS=0`: the candidate disables BF16 residual1 forward storage. The 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured `1.026566x` train-loop time and `0.974147x` tokens/sec versus the default stored-residual path. A 2026-06-25 CUDA 13.3 rerun exposed that the saved-packed-attention recompute path also needs one float attention-projection scratch buffer when residual1 storage is disabled; that diagnostic now reports `saved_packed_attention_recompute_needs_float_attention_projection=true` and keeps only the attention projection scratch while leaving the MLP projection scratch elided. After the scratch fix, the dedicated RTX 5090 one-step same-script gate no longer crashes, reports `stored_residual1_activation_blocks: 0`, improves setup wall time to `0.810095x`, but still rejects default promotion because train-loop time regressed to `1.059928x` and tokens/sec to `0.943455x`.
- [x] Reject `NFN_NATIVE_GPT_BF16_RESIDUAL1_LN_BACKWARD=0`: the candidate kept residual1 storage but disabled the BF16 residual LayerNorm backward consumer, and the 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured `1.014258x` train-loop time and `0.985954x` tokens/sec versus the default BF16 residual backward path.
- [x] Reject `NFN_NATIVE_GPT_LN1_BF16_QKV_FORWARD=0`: the candidate disabled the BF16 LN1-to-QKV handoff, but the 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured `1.015832x` train-loop time and `0.984423x` tokens/sec versus the default BF16 handoff.
- [x] Reject `NFN_NATIVE_GPT_FULL_ACTIVATION_TAPE=1` as a default dense GPT training strategy: the candidate allocates one activation tape per transformer block and skips backward recompute, but the 2026-06-17 dedicated RTX 5090 one-microbatch same-script run measured `61.335739x` train-loop wall time and `0.016304x` tokens/sec versus the default scratch-recompute tape, so the switch remains diagnostic-only.
- [x] Reject `NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1` as a default LM-head dHidden route: the candidate moved `m=768,n=8192,k=50304` from BF16 `cublasGemmEx` to cuBLASLt, but the 2026-06-17 dedicated RTX 5090 one-microbatch 3-sample run measured `1.021534x` train-loop wall time and `0.978930x` tokens/sec, so the large-K cap remains.
- [x] Historical BF16 attention grad-out rejection remains consistent with the current default-off decision. A 2026-06-17 one-step stage probe rejected `NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1` after it reduced `block_backward.attn_sdpa.to_qkv` only from `272.560 ms` to `269.077 ms`, slowed `block_backward.attn_proj.dinput` from `53.980 ms` to `205.680 ms`, and increased total train-loop wall time from `2658.21 ms` to `2833.25 ms`. The 2026-06-24 current-tree CUDA 13.3 rebuilt verification again kept the route rejected/default-off.
- [x] Keep `NFN_NATIVE_GPT_REUSE_MLP_PROJ_BF16_GRAD_OUT=0` rejected for the default MLP projection backward route: a 2026-06-17 one-step stage probe improved `block_backward.mlp_proj.dweight_bias` from `173.413 ms` to `170.561 ms`, but slowed `block_backward.mlp_proj.dinput` from `176.018 ms` to `196.379 ms` and left total train-loop wall time neutral-to-slower (`2658.21 ms` to `2658.65 ms`). The 2026-06-20 CUDA 13.3 retest also failed: the 3-step, 3-sample same-script candidate measured `1.010036x` train-loop wall and `0.990065x` tokens/sec, while `linear_bf16_a_pack_count` rose from `36` to `324` because the route stopped reusing the one BF16 projection-gradient pack.
- [x] Add `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=m,n,k,opA,opB,index` / `NFN_TILE_CUDA_CUBLASLT_HEURISTIC_SHAPE=m,n,k,opA,opB,index` for one-shape cuBLASLt heuristic bisection, and reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,2304,65536,N,T,0` as a default QKV dWeight route: the 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured `0.999825x` mean train-loop wall time but `1.001401x` median train-loop wall time versus the default heuristic selection. The 2026-06-23 post-CUDA-reinstall stage-timed recheck also rejects it: the plan cache changed `cublaslt:768x2304x65536:N,T` from heuristic `1` to `0`, but measured `0.999904x` train-loop wall time with regressions at `stage.block_backward.total_ms=1.000055x` and `stage.block_backward.qkv.dweight_bias.total_ms=1.003363x`.
- [x] Add and reject the cuBLASLt waves-policy selector as a default dense GPT route: `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_POLICY=min_waves` matches the llm.kittens lowest-`wavesCount` selector but measured `1.001205x` train-loop wall time and `0.998809x` tokens/sec in the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample benchmark, while `max_waves` measured `1.001045x` train-loop wall time and `0.998964x` tokens/sec. A post-atomic-route current build check reconfirmed `min_waves` as slower at `1.009572x` train-loop wall time and `0.990522x` tokens/sec in `/tmp/nfn_cublaslt_min_waves_pair_20260618.json`. The 2026-06-28 CUDA 13.3.33 no-stage rerun still rejected `min_waves`: it changed cuBLASLt selected heuristics but regressed current native train-loop wall time to `1.005895x`, steady-state CUDA-event timing to `1.005848x`, startup-plus-train-loop wall time to `1.006503x`, and tokens/sec to `0.994138x`. Explicit shape/global index overrides still win, and the NeuralFn default remains index 1.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,2304,65536,N,T,2` as a default QKV dWeight route: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.001121x` train-loop wall time and `0.998895x` tokens/sec versus the current cuBLASLt heuristic selection.
- [x] Keep `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=3072,768,65536,N,T,0` diagnostic-only for MLP projection dWeight: the 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured only a small `0.998065x` mean and `0.998219x` median train-loop wall-time ratio, and the 2026-06-18 follow-up 3-step, 2-sample check measured `0.998915x` train-loop wall time with `1.001091x` tokens/sec. The measured delta stays within run-to-run noise, so the default cuBLASLt heuristic route is unchanged.
- [x] Keep `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=3072,768,65536,N,T,2` diagnostic-only for MLP projection dWeight: the 2026-06-18 dedicated RTX 5090 same-script 3-step, 2-sample check measured only `0.998753x` train-loop wall time and `1.001251x` tokens/sec, again inside noise and not enough to replace the default cuBLASLt heuristic.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,3072,65536,N,T,0` as a default MLP FC dWeight route: the first 3-sample run was noisy, and the 2026-06-17 dedicated RTX 5090 5-sample confirmation measured `1.015160x` train-loop wall time and `0.985688x` tokens/sec versus the default heuristic selection.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,50304,8192,N,T,0` as a default LM-head dWeight route: the 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured `1.002294x` train-loop wall time and `0.997718x` tokens/sec versus the default heuristic selection.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,65536,3072,N,N,0` as a default MLP projection dInput route: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.000778x` train-loop wall time, `1.006005x` median train-loop wall time, and `0.999289x` tokens/sec versus the default shape heuristic.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,65536,3072,N,N,4` as a default MLP projection dInput route: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.003512x` train-loop wall time, `1.004505x` median train-loop wall time, and `0.996511x` tokens/sec versus the default shape heuristic.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,65536,3072,N,N,5` as a default MLP projection dInput route: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.002550x` train-loop wall time, `1.002451x` median train-loop wall time, and `0.997462x` tokens/sec versus the default shape heuristic.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,50304,8192,N,T,4` as a default LM-head dWeight route: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.007483x` train-loop wall time, `1.007843x` median train-loop wall time, and `0.992575x` tokens/sec versus the default shape heuristic.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,3072,65536,N,T,0` as a default MLP projection dWeight+bias route: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.012195x` train-loop wall time, `1.021403x` median train-loop wall time, and `0.988237x` tokens/sec versus the default shape heuristic.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,65536,3072,N,N,0`, `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,65536,2304,N,N,0`, and `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,65536,768,N,N,0` as default dInput routes: 2026-06-17 one-step stage probes were noise-level or slowed the targeted `mlp_proj.dinput`, `qkv.dinput`, and `attn_proj.dinput` child stages, so the global default heuristic remains selected for these `N,N` shapes.
- [x] Reject quick `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=3072,65536,768,N,N,0` and `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=3072,65536,768,N,N,4` probes as default MLP-FC dInput routes: the 2026-06-18 dedicated RTX 5090 same-script 3-step single-sample checks measured `1.001214x` and `1.004334x` train-loop wall time respectively, with `0.998787x` and `0.995686x` tokens/sec versus the current default.
- [x] Reject quick `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,65536,2304,N,N,4` and `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,65536,768,N,N,4` probes as default QKV/attention-projection dInput routes: the 2026-06-18 dedicated RTX 5090 same-script 3-step single-sample checks measured `1.001084x` and `1.006493x` train-loop wall time respectively, with `0.998917x` and `0.993546x` tokens/sec versus the current default.
- [x] Add GPT-prefixed aliases for native linear shape stats so future CUDA Tile kernel bisections can use either `NFN_NATIVE_LINEAR_SHAPE_STATS=1`, `NFN_TILE_CUDA_LINEAR_SHAPE_STATS=1`, `NFN_NATIVE_GPT_LINEAR_SHAPE_STATS=1`, or `NFN_NATIVE_GPT2_LINEAR_SHAPE_STATS=1`; a 2026-06-18 one-step profile with the existing linear alias reported the current 15 hot GEMM buckets, including TK BF16 LM-head logits, GEMMEx LM-head dHidden, cuBLASLt LM-head dWeight, and the 96-call transformer block forward/backward shapes.
- [x] Extend native linear shape stats with opt-in CUDA-event timing for kernel bisection: `nfn_native_tile_trainer_linear_shape_stats_entry` now returns `total_us`, native GPT runtime JSON reports `linear_shape_stats[].total_us` and `avg_us`, and the stats mode synchronizes measured GEMMs only when shape stats are enabled.
- [x] Fill the remaining TK shape-stat timing gaps for fused MLP FC+GELU, fused MLP projection dInput+dGELU, and TK BF16-to-float output conversion paths so the CUDA 13.3 profiles rank every active TK bucket by measured elapsed time instead of reporting zero-time rows.
- [x] Promote the wired TK BF16 forward route for the padded LM-head logits shape `50304,8192,768,T,N` by default. Earlier CUDA 13.3 checks had rejected this shape because the BF16-input/BF16-weight/BF16-output wrapper never attempted the existing TK bridge; after wiring that wrapper, the 2026-06-19 dedicated RTX 5090 same-script 5-step, 3-sample candidate measured `0.923926x` train-loop wall time and `1.086108x` tokens/sec versus the GEMMEx fallback. A stage-timed follow-up measured `lm_head_backward.total_ms` at `0.642874x` of fallback, with block stages essentially unchanged. Use `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=50304,8192,768,T,N` or `NFN_TILE_CUDA_LINEAR_TK_FORWARD_DISABLE_SHAPE=50304,8192,768,T,N` to reproduce the old fallback for bisection.
  - Same-script llm.kittens parity after this promotion: `NFN_SM120_PARITY_STEPS=5 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_PROFILE_DIR=none ... bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `3214.092 ms/step` and `164306.8 tok/s`; NeuralFn measured `2834.720 ms/step` and `184952 tok/s`, or `0.881966x` train-loop wall time and `1.125650x` tokens/sec versus the reference. Recheck over fresh multi-sample parity runs before treating this as final parity because the reference sample was slower than earlier CUDA 13.3 measurements.
- [x] Add and reject the default-off `NFN_NATIVE_LINEAR_TK_DINPUT=1` / `NFN_TILE_CUDA_LINEAR_TK_DINPUT=1` BF16-gradient/BF16-weight dInput diagnostic for the LM-head dHidden shape. The one-step shape-stat smoke moved `768,8192,50304,N,N` from GEMMEx (`236725 us` bucket time in the prior smoke) to TK (`215821 us` bucket time), but the dedicated RTX 5090 same-script 5-step, 3-sample wall benchmark measured `1.049216x` train-loop time and `0.953102x` tokens/sec versus the default route, so GEMMEx remains the default.
- [x] Fold dense GPT native train-loss collection into the LM-head backward recompute pass. Training microbatches now skip the separate forward LM-head loss pass when train-loss recording is enabled, accumulate CE loss from the row-chunked logits already recomputed for CE backward, and then overwrite those logits with dLogits. Validation and evo candidate scoring stay on the forward-only LM-head loss path. Verification: rebuilt `build/nfn_gpt_native_train`, passed the focused native GPT test slice, and ran the GPU-visible SM120 parity harness. The parity harness keeps train-loss recording disabled for timing-only parity and still measured NeuralFn at `1.036929x` train-loop wall time versus llm.kittens, so this is a workflow/logging improvement rather than the remaining throughput fix.
- [x] Keep the CUDA 13.3 train-loss scalar-copy cleanup and the near-default MLP activation-store memory probe out of the critical-path candidate list. The loss-copy default now relies on blocking D2H `cudaMemcpy` ordering and keeps `NFN_NATIVE_GPT_LM_HEAD_LOSS_COPY_SYNC=1` only for same-binary diagnostics; a dedicated RTX 5090 5-step, 3-sample run with `--train-loss-every-steps 1` measured the no-sync default as neutral/noise-level (`1.000906x` train-loop wall, `0.999097x` tokens/sec). Reducing stored MLP activation blocks from 12 to 11 (`NFN_NATIVE_GPT_STORE_MLP_BLOCKS=11`) did not improve setup and regressed train-loop wall time to `1.024478x` with `0.976120x` tokens/sec because recompute increased BF16 pack/cache churn. Do not promote either as a parity fix.
- [x] Accumulate native dense GPT train-loss scalar on device across the gradient-accumulation optimizer step and copy it to the host once per logged step. The 2026-06-22 old-vs-new same-script RTX 5090 benchmark with `--train-loss-every-steps 1` passed the strict native-candidate gate at `0.999670x` train-loop wall time and `1.000331x` tokens/sec; the candidate reported `train_loss_host_d2h_count=3` for three logged steps and `train_loss_microbatch_host_d2h_copies_elided_per_logged_step=7` at the default eight-microbatch step.
  - 2026-06-24 clarified the disabled-logging runtime JSON counters: the default `--train-loss-every-steps 0` path still performs zero train-loss D2H copies, and now reports `train_loss_host_d2h_copies_per_logged_step=0` plus `train_loss_microbatch_host_d2h_copies_elided_per_logged_step=0` instead of describing the enabled train-loss logging path.
- [x] Expose native dense GPT train-loss cadence as `--train-loss-every-steps N` with `--train-log-every` / `--train-log-every-steps` aliases, defaulting to `0` for benchmark parity. SDK `NativeGptRunConfig` / `NativeGpt2RunConfig` now forward `train_loss_every_steps` to the compiled CLI so train loss can be sampled without graph-editor node flow or the old duplicate LM-head forward-loss pass.
- [x] Keep the LM-head logits cuBLASLt heuristic override rejected for `50304,8192,768,T,N`: the 2026-06-18 dedicated RTX 5090 same-script 3-step single-sample probe with `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=50304,8192,768,T,N,0` did not change the cuBLASLt/TK dispatch counters and measured `1.001453x` train-loop wall time with `0.998551x` tokens/sec versus the default BF16 GEMMEx fallback.
- [x] Reject the BF16-output cuBLASLt LM-head logits probe for `50304,8192,768,T,N`: a temporary candidate library built successfully, but the 2026-06-18 dedicated RTX 5090 one-step shape-stat smoke still reported the logits bucket as BF16 `cublasGemmEx` (`330916 us` over 64 calls), so the no-op diagnostic flag was removed instead of promoted.
- [x] Reject the older float32/TF32 LM-head route as a current default fallback: `NFN_NATIVE_GPT_LM_HEAD_BF16_LOGITS=0` measured `1.229126x` train-loop wall time and `0.813607x` tokens/sec in the 2026-06-18 dedicated RTX 5090 5-step, 3-sample same-script benchmark versus the BF16 logits/dlogits default.
- [x] Add and reject `NFN_NATIVE_GPT_LM_HEAD_DWEIGHT_BEFORE_DHIDDEN=1` as a default LM-head row-chunk order: the candidate runs dWeight before dHidden after CE writes BF16 dlogits, but the 2026-06-18 dedicated RTX 5090 5-step, 3-sample benchmark measured `1.001048x` train-loop wall time and `0.998959x` tokens/sec versus the default CE -> dHidden -> dWeight order.
- [x] Keep the one-shape TK forward gate diagnostic-only and reject disabling the `3072,65536,768,T,N` MLP FC+GELU bucket: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample benchmark with `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=3072,65536,768,T,N` measured `13.608951x` train-loop wall time and `0.073530x` tokens/sec versus the TK default, proving the scalar fallback is not a viable default route.
- [x] Reject disabling the `768,65536,3072,T,N` MLP projection forward bucket as a default route: the 2026-06-18 dedicated RTX 5090 same-script 3-step, 2-sample benchmark with `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=768,65536,3072,T,N` measured `1.010042x` train-loop wall time and `0.990063x` tokens/sec versus the current TK default.
- [x] Reject disabling the `768,65536,768,T,N` attention projection forward bucket as a default route: the 2026-06-18 dedicated RTX 5090 same-script 3-step, 2-sample benchmark with `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=768,65536,768,T,N` measured `1.009872x` train-loop wall time and `0.990230x` tokens/sec versus the current TK default.
- [x] Keep the 2026-06-18 post-LM-head-beta-fix rechecks rejected: `NFN_NATIVE_GPT_DWEIGHT_FIRST_MICROBATCH_BETA_ZERO=0` still measured slower at `1.009049x` train-loop wall time and `0.991044x` tokens/sec; `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1` was noisy and worsened token-init time to `1.044352x`; a removed `NFN_NATIVE_GPT_TOKEN_WEIGHT_PATTERN16_INIT=1` prototype also worsened token-init time to `1.033752x` mean and `1.058094x` median, so the default Tile initializer and beta-zero route remain unchanged.
- [x] Reject `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_GRID3D=1` as a default packed-attention dprep route: the diagnostic 3D batch/head/time launch removes per-row division/modulo from dprep, but the 2026-06-17 dedicated RTX 5090 5-sample confirmation measured `1.008389x` train-loop wall time and `0.991895x` tokens/sec versus the row-linear dprep default.
- [x] Promote the GPT `heads=12, head_dim=64` BF16-grad packed-attention dprep specialization by default: the 2026-06-17 dedicated RTX 5090 same-script 5-step, 3-sample candidate benchmark measured `0.997290x` mean train-loop wall time and `1.002726x` mean tokens/sec versus the older generic row dprep path; set `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_HD64_SPECIALIZED=0` to reproduce the old path.
- [x] Reject a vectorized pair-load/store variant of the HD64 dprep specialization: the 2026-06-17 dedicated RTX 5090 same-script 5-step, 3-sample candidate benchmark with `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_HD64_VEC2=1` measured `1.001076x` mean train-loop wall time and `0.998961x` mean tokens/sec versus the promoted HD64 default, so the candidate switch was removed.
- [x] Re-check `NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1` with the current HD64 dprep default and keep it as a non-promoted diagnostic for low-level Tile callers: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.000529x` train-loop wall time and `0.999486x` tokens/sec versus the current environment-default path.
- [x] Reject `NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF=1` as the current default fallback compute type: the 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.004222x` train-loop wall time and `0.995808x` tokens/sec versus `CUBLAS_COMPUTE_32F` for non-cuBLASLt BF16 GEMMEx paths.
- [x] Historical HD64-era BF16 attention grad-out rejection remains consistent with the current default-off decision. The 2026-06-18 dedicated RTX 5090 same-script 5-step, 3-sample run measured `1.011370x` train-loop wall time and `0.988829x` tokens/sec versus the then-default float attention-gradient handoff. The 2026-06-24 current-tree CUDA 13.3 rebuilt verification again kept `NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1` rejected/default-off.
- [x] Reject `NFN_NATIVE_LINEAR_CUBLASLT_WORKSPACE_MB=256` as a default cuBLASLt workspace cap: one-microbatch timing was slightly faster, but the 2026-06-17 dedicated RTX 5090 normal 5-step 3-sample run measured `1.000863x` train-loop wall time and `0.999150x` tokens/sec versus the 128 MiB default.
- [x] Add `NFN_NATIVE_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=m,n,k,opA,opB` / `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=m,n,k,opA,opB` for one-bucket BF16 cuBLASLt fallback bisection.
- [x] Reject `NFN_NATIVE_LINEAR_BF16_CUBLASLT=0` as a current default-route fallback: the 2026-06-17 dedicated RTX 5090 one-microbatch 3-sample run measured `6.171959x` train-loop wall time and `0.162057x` tokens/sec versus the default cuBLASLt path.
- [x] Reject `NFN_NATIVE_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=768,65536,3072,N,N` as a default fallback for MLP projection dInput: the 2026-06-17 dedicated RTX 5090 one-microbatch 3-sample run measured `1.007716x` train-loop wall time and `0.992349x` tokens/sec.
- [x] Reject `NFN_NATIVE_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=768,65536,768,N,N` as a default fallback for the smaller dInput bucket: the 2026-06-17 dedicated RTX 5090 one-microbatch 3-sample run measured `1.008810x` train-loop wall time and `0.991317x` tokens/sec.
- [x] Reject `NFN_NATIVE_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=768,3072,65536,N,T` as a default fallback for MLP projection dWeight: the 2026-06-17 dedicated RTX 5090 one-microbatch 3-sample run measured `2.857882x` train-loop wall time and `0.351320x` tokens/sec.
- [x] Do not promote `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1` as the default token-weight startup initializer: the 2026-06-17 dedicated RTX 5090 startup-only 5-sample run with the current default measured noise-equivalent `0.996666x` token init time and `0.997098x` total wall time versus the CUDA Tile initializer, so the threaded kernel remains diagnostic-only.
- [x] Align the low-level token-weight Tile ABI default with the compiled trainer/docs by making `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT` default off inside `libnfn_native_train_tile_ops.so`; the 2026-06-17 dedicated RTX 5090 startup-only 5-sample comparison against `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1` measured the corrected default at `0.940074x` token init time, `0.974488x` setup wall time, and `0.976437x` total wall time.
- [x] Keep the GPT token-weight fast int32 Tile-index initializer as the current default after the CUDA 13.3 reinstall. A same-script startup-only recheck on the dedicated RTX 5090 compared the current default against `NFN_NATIVE_GPT_TOKEN_WEIGHT_FAST_INT32_INIT=0`; the opt-out measured `1.014432x` setup wall time, `1.013372x` token-init time, and `1.014616x` total startup versus the int32 default, so the older int64 Tile-index path remains bisection-only.
- [x] Re-check current dense GPT parity against `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh` after the GPT evo delegation work: the 2026-06-18 dedicated RTX 5090 same-script 3-step run measured llm.kittens at `2464.486667 ms/step` and NeuralFn at `2564.563333 ms/step`, or `1.040608x` train-loop wall time and `0.960888x` tokens/sec versus the reference; startup/setup remains dominated by `setup.float_arena_materialize`, `setup.uint16_arena_materialize`, and `setup.token_weight_init`.
- [x] Keep the 2026-06-18 same-script rechecks rejected for the then-current default route: `NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1` measured `1.002364x` train-loop wall time, `--lm-head-row-chunk-size 16384` measured `1.017689x`, `NFN_NATIVE_GPT_CE_BF16_EXP2=1` measured `1.002993x`, historical `NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=128` measured `1.004624x` before the later CUDA 13.3 promotion, and `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE=0` measured `1.003253x` versus the current default.
- [x] Reject disabling direct uint16 token IDs as a default CE path: the 2026-06-18 dedicated RTX 5090 same-script 10-step check with `NFN_NATIVE_GPT_DIRECT_U16_TOKENS=0` measured `1.007744x` train-loop wall time, `0.992317x` tokens/sec, and `1.006201x` total wall time versus the direct-u16 default.
- [x] Reject the 2026-06-18 reduced-storage startup candidate (`NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS=0 NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_ACTIVATIONS=0 NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS=0`) as a model-quality training default: it reduced setup wall time to `0.652100x` but regressed train-loop wall time to `1.457105x`, so saved activation storage remains the default while startup optimization focuses on cheaper allocation/initialization.
- [x] Retile the trainer-facing native GPT token-weight CUDA Tile initializer from the previous 2048-element default to 4096 elements, with `NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE=1024|2048|4096|8192` reserved for compile-time paired bisection. The 2026-06-17 dedicated RTX 5090 startup-only same-script run measured 4096 at `0.895736x` mean token-init time versus 2048 but noisy/slower total startup (`1.179639x`), while the one-step native training comparison measured `0.991404x` total wall time, `0.999314x` train-loop wall time, and `1.000805x` tokens/sec. The 1024 candidate was rejected after measuring `1.007101x` token-init time and `1.012266x` total wall time versus 2048. The 2026-06-18 8192 candidate compiled successfully but stayed non-default after the dedicated RTX 5090 9-sample startup-only comparison against 4096 measured `1.005585x` token-init time, with total startup `0.990436x` inside broader arena-materialization noise.
- [x] Refresh the live 10-step dedicated RTX 5090 parity baseline after the diagnostic commits: NeuralFn `build/nfn_gpt_native_train --backend tile-cuda --tinystories --max-steps 10 --train-batch-tokens 524288 ...` reported `202662` train tokens/sec, while `/mnt/disk2/dev/open-source/llm.kittens/train_gpt2cu ... -x 10` ended at `210704` tok/s and averaged about `210.8k` tok/s after warmup, putting NeuralFn at roughly `0.962x` of the current llm.kittens run.
- [x] Refresh the live 10-step dedicated RTX 5090 parity baseline after the 2026-06-18 allocation/reporting commits: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=30 NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_after_ln1_arena_20260618.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2590.436 ms/step` and `200148.7 tok/s`; NeuralFn measured `2639.410 ms/step` and `198638 tok/s`, or `1.018906x` train-loop time and `0.992452x` tokens/sec versus the reference. The selected RTX 5090 reported zero compute processes before and after the paired sample.
- [x] Confirm that parity baseline with a 3-sample dedicated RTX 5090 run: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=3 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=30 NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_after_ln1_arena_3sample_20260618.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2564.257 ms/step` and `204478.3 tok/s`; NeuralFn measured `2650.133 ms/step` and `197865.3 tok/s`, or `1.033524x` train-loop time and `0.967701x` tokens/sec versus the reference. The selected RTX 5090 reported zero compute processes before and after every paired sample.
- [x] Refresh the dedicated RTX 5090 parity baseline after the NanoGPT routing and tiled fallback commits: `NFN_SM120_PARITY_STEPS=10 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_MAX_GPU_UTILIZATION_PCT=30 NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_after_nanogpt_route_20260618.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2471.984 ms/step` and `212391.2 tok/s`; NeuralFn measured `2555.440 ms/step` and `205165.0 tok/s`, or `1.033761x` train-loop time and `0.965977x` tokens/sec versus the reference. The selected RTX 5090 reported zero compute processes before and after the paired sample.
- [x] Re-check and reject `NFN_NATIVE_GPT_LM_HEAD_ROW_CHUNK_SIZE=16384` after the NanoGPT routing and tiled fallback commits: the dedicated RTX 5090 same-script 10-step, 3-sample candidate benchmark measured `1.000599x` train-loop wall time and `0.999416x` tokens/sec versus the then-current 8192-row default, so the remaining LM-head work stays focused on the GEMM route instead of row-chunk tuning.
- [x] Revisit CUDA 13.3 failed-test surfaces with GPU-visible execution after the WSL toolkit reinstall: sandboxed `nvidia-smi`/native CUDA preflight still reported OS-blocked GPU access, while unsandboxed `nvidia-smi` reported the dedicated RTX 5090 on CUDA UMD 13.3 with zero compute processes. With real GPU access, `NFN_TILE_CUDA_TEST=1 NFN_TILE_CUDA_BUILD=1 NFN_TILE_CUDA_ARCH=sm_120 python -m pytest tests/test_tile_cuda_ops.py tests/test_tile_cuda_modules.py tests/test_tile_cuda_optimizer.py tests/test_tile_cuda_gpu.py -q -rs` passed (`537 passed`) and `python -m pytest tests/test_native_gpt2.py -q` passed (`52 passed`).
- [x] Promote `NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=0` under CUDA 13.3: the dedicated RTX 5090 same-script 5-step, 3-sample benchmark measured `0.997878x` train-loop wall time, `1.002130x` tokens/sec, and `0.994429x` total wall time versus the old full-final-norm BF16 prepack default, with no route-counter change and only the LM-head dWeight strategy changing to per-chunk hidden packing. Set the env flag to `1` only for regression checks against the older full-microbatch prepack route.
- [x] Add and reject the mirror `lm_head_prepack_bf16_hidden_on` same-script candidate profile: the current per-chunk LM-head route is pinned as baseline and the older full-prepack route as candidate, but the CUDA 13.3 dedicated RTX 5090 5-step, 3-sample gate failed `stage.lm_head_backward.dhidden.total_ms` at `1.000690x` despite train-loop mean `0.997953x`, so full prepack stays non-default and requires `NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1` for intentional reruns.
- [x] Re-check and keep `NFN_NATIVE_GPT_CE_BF16_EXP2=1` rejected under CUDA 13.3: the dedicated RTX 5090 same-script 5-step, 3-sample benchmark measured `1.003930x` train-loop wall time and `0.996089x` tokens/sec versus the current BF16 CE math path.
- [x] Reject `NFN_NATIVE_GPT_PUBLIC_VOCAB_CE=0` as a default CE route: the 2026-06-17 dedicated RTX 5090 normal 5-step 3-sample run measured padded-vocab CE at `1.001998x` train-loop wall time and `0.998146x` tokens/sec versus the default public-vocab strided CE, so the current strided public-vocab path stays enabled.
- [x] Retire `NFN_NATIVE_GPT_REUSE_PACKED_LN2_FC_GELU=0` / `NFN_NATIVE_GPT2_REUSE_PACKED_LN2_FC_GELU=0`: the fallback path crashed the current native trainer with CUDA error 700, and the rebuilt trainer now ignores the override while keeping `reuse_packed_ln2_fc_gelu_enabled: true` and the prepacked-LN2 FC+GELU strategy.
- [x] Default `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_BF16=1` for the dense GPT native trainer: the 2026-06-17 dedicated RTX 5090 normal 5-step 3-sample run measured `0.991351x` train-loop wall time and `1.008731x` tokens/sec versus the previous saved-attention LN1 apply-stats recompute path, while adding about `1107296256` bytes of LN1 BF16 tape at the default shape.
- [x] Reject a narrow TK plain-dInput candidate for supported BF16 block dInput GEMMs: the temporary `NFN_NATIVE_LINEAR_TK_DINPUT=1` branch routed plain BF16 dInput through `matmul_dispatch_tk_ab` with BF16 scratch plus float conversion, but the 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured it at `1.064272x` train-loop wall time and `0.939630x` tokens/sec versus the default cuBLASLt dInput route, so no runtime switch was kept.
- [x] Reject a cuBLASLt BF16-output plain-dInput candidate: the temporary `NFN_NATIVE_LINEAR_BF16_DINPUT_OUT=1` branch routed BF16-grad/BF16-weight dInput GEMMs through BF16 output scratch plus float conversion, but the 2026-06-17 dedicated RTX 5090 same-script 3-sample run measured it at `1.040670x` train-loop wall time and `0.960945x` tokens/sec versus the default float-output cuBLASLt dInput route, so no runtime switch was kept.
- [x] Reject `NFN_NATIVE_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE=3072,768,65536,N,T` as a default fallback for the current MLP projection dWeight bucket: the 2026-06-17 dedicated RTX 5090 normal 5-step 3-sample run measured `3.199175x` train-loop wall time, `0.312590x` tokens/sec, and `5.406332x` `stage.block_backward.total_ms` versus the default cuBLASLt BGRADB route.
- [x] Reject `NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD=0` as a default-route fallback: the 2026-06-17 dedicated RTX 5090 normal 5-step 3-sample run measured `0.998810x` mean train-loop wall time but `1.000652x` median train-loop wall time, with `1.001197x` mean tokens/sec and unchanged hot buckets, so disabling the mixed float32-hidden/BF16-gradient BGRADB route remains a noise-equivalent diagnostic rather than a promoted default.
- [x] Reject `NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP=96` as a default packed-attention backward cap: the 2026-06-17 dedicated RTX 5090 same-script 3-sample one-step bisection against the default cap 64 measured `1.007835x` mean and `1.015260x` median train-loop wall time, with `0.992374x` mean tokens/sec, so the default cap stays 64.
- [x] Reject `--lm-head-row-chunk-size 6144` as a default LM-head chunk route: the 2026-06-17 dedicated RTX 5090 same-script 3-sample one-step bisection against the default 8192 measured `1.003477x` mean and `1.004724x` median train-loop wall time, with `0.996541x` mean tokens/sec, so the default chunk stays 8192.
- [x] Reject `NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1` plus `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,8192,50304,N,N,0` as a default LM-head dHidden route: the 2026-06-17 dedicated RTX 5090 same-script 3-sample one-step bisection measured `1.024628x` train-loop wall time and `0.975979x` tokens/sec versus the default BF16 `cublasGemmEx` dHidden route.
- [x] Add diagnostic `NFN_NATIVE_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE=m,n,k,opA,opB` / `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE=m,n,k,opA,opB` for the inverse BF16 cuBLASLt one-shape bisection, then keep the LM-head dHidden route rejected as a default: under CUDA 13.3 on the dedicated RTX 5090, forcing `768,8192,50304,N,N` through cuBLASLt changed shape stats to `bf16-cublaslt-dinput-dhidden`, but the same-script 5-step, 3-sample benchmark measured `1.024865x` train-loop wall time and `0.975739x` tokens/sec versus the current BF16 `cublasGemmEx` fallback.
- [x] Add diagnostic `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_WARPS=N` / GPT2 fallback for packed-attention backward dprep row grouping, and keep the default at 3: the 2026-06-17 dedicated RTX 5090 same-script 3-sample one-step bisections measured `NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_WARPS=2` at `1.002758x` train-loop wall time and `0.997257x` tokens/sec, while `=4` measured `1.001645x` mean train-loop wall time and `0.998425x` mean tokens/sec despite a faster median, so neither is promoted.
- [x] Reject `NFN_NATIVE_GPT_DIRECT_BF16_QKV_GRAD_SCRATCH=0` as a default-route fallback: the 2026-06-17 dedicated RTX 5090 normal 5-step 3-sample run measured `1.029204x` train-loop wall time and `0.971631x` tokens/sec, with `stage.block_backward.attn_sdpa.total_ms` at `1.129386x` and `stage.block_backward.qkv.total_ms` at `1.091903x` versus the direct BF16 QKV grad scratch default.
- [x] Reject `NFN_NATIVE_GPT_FUSE_LN_BACKWARD_AFFINE_RESIDUAL=0` as a default-route fallback: the 2026-06-17 dedicated RTX 5090 normal 5-step 3-sample run measured `1.012891x` train-loop wall time and `0.987285x` tokens/sec versus the fused LayerNorm affine+dInput+residual backward default.
- [x] Do not promote `NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_INIT=0` as a startup default: the 2026-06-17 dedicated RTX 5090 startup-only 5-sample run measured only `0.993382x` setup wall time and `0.993560x` total wall time versus the fused token-weight BF16 shadow initializer, with noisy sample spread.
- [x] Reject `NFN_NATIVE_GPT_COMBINED_BF16_ARENA=0` as a startup default: the 2026-06-17 dedicated RTX 5090 startup-only 5-sample run measured `0.986733x` setup wall time, but the normal 5-step 3-sample run regressed to `1.020957x` train-loop wall time, `0.979932x` tokens/sec, and `1.019949x` total wall time versus the combined BF16 arena default.
- [x] Suballocate the saved packed-attention LN1 BF16 tape from the default combined uint16 arena instead of issuing a separate BF16 `cudaMalloc`; runtime JSON now counts that tape in `uint16_arena_suballocation_count` while preserving the `NFN_NATIVE_GPT_COMBINED_BF16_ARENA=0` per-buffer fallback.
- [x] Suballocate stored MLP LayerNorm stats and saved packed-attention LN1 stats sidecars from the default float arena instead of issuing separate float `cudaMalloc` calls during startup; keep `NFN_NATIVE_GPT_FLOAT_STATS_ARENA=0` / `NFN_NATIVE_GPT2_FLOAT_STATS_ARENA=0` for paired bisection against the older sidecar allocation path.
- [x] Reject `--lm-head-row-chunk-size 65536` for the default `64 x 1024` dense GPT shape: the candidate drove the dedicated RTX 5090 to 100% utilization and about `31926 MiB / 32607 MiB` used memory without completing a 5-step candidate sample in the expected paired-benchmark window, so the then-current 8192-row default remained the practical LM-head chunk size.
  - 2026-06-18 CUDA toolkit 13.3 WSL reinstall recheck: the full 65,536-row LM-head chunk now completes startup and a one-step training smoke on the dedicated RTX 5090, so the earlier failure mode is stale. It remains rejected because the one-step full-chunk run measured `118009 ms` train-loop wall time and `4442.76 tok/s` with `6593445888` BF16 logit bytes, versus the current 8,192-row default one-step smoke at `3416.57 ms` total wall time and `824180736` BF16 logit bytes.
- [x] Re-check and reject `--lm-head-row-chunk-size 16384` after the saved-LN1-BF16 default: the 2026-06-17 dedicated RTX 5090 normal 5-step 3-sample run measured `1.014421x` train-loop wall time and `0.985784x` tokens/sec versus the 8192-row default.
- [x] Reject the BF16 GEMMEx `FAST_16BF` compute-type candidate for remaining LM-head fallback shapes: the 2026-06-18 dedicated RTX 5090 paired run with `NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF=1` / `NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_FAST_16BF=1` measured `1.001097x` train-loop wall time and `0.998921x` tokens/sec versus the current default.
- [x] Route native GPT `.bin` checkpoint text prompts through the compiled sampler path from `nfn infer --checkpoint ... --prompt ...` and `python cli/scripts/infer_gpt2.py --native-checkpoint ... --prompt ...` by GPT-2-tokenizing in the lightweight wrapper and calling `nfn_gpt_native_train --sample-checkpoint ... --prompt-tokens ...`, while keeping `--native-info` metadata inspection Torch-free.
- [x] Decode successful native GPT checkpoint sampler output in the lightweight wrapper, preserving compiled JSON while printing generated token IDs and GPT-2 text without Torch or graph-editor tensors.
- [x] Add compiled C++ native GPT checkpoint inspection with `nfn_gpt_native_train --native-info --native-checkpoint PATH` / `--inspect-checkpoint PATH`, reporting shape, precision, DONE marker state, and file-size validation before CUDA, Torch, dataset, or graph-node setup.
- [x] Route native GPT `.bin` checkpoint `--prompt-tokens` requests through compiled C++ with `nfn_gpt_native_train --sample-checkpoint PATH --prompt-tokens IDS`, validating checkpoint shape, context, vocab bounds, token parsing, executing autoregressive CUDA Tile checkpoint forward passes, and returning generated token IDs without graph-editor tensor flow.
  - 2026-06-27 aligned native checkpoint sampling and the GPT/GPT-2
    compatibility SDK launch configs with the workstation native trainer's
    `cuda_visible_devices="dedicated"` default. Binding payloads and subprocess
    environments now resolve the selector through `nvidia-smi` before launching
    compiled C++ so `nfn infer --checkpoint model_*.bin --prompt-tokens ...`
    uses the display-disabled compute GPU by default while still honoring an
    explicit `CUDA_VISIBLE_DEVICES` or `cuda_visible_devices="0"` override.
- [x] Add native GPT checkpoint tensor-layout decode with `--checkpoint-layout`, deriving tensor shapes, payload offsets, file offsets, and bounded payload samples from the checkpoint header without CUDA, Torch, datasets, or graph-editor tensor flow.
- [x] Add native GPT checkpoint payload load smoke with `--checkpoint-load-smoke`, moving a bounded bf16 checkpoint slice through CUDA memory and Tile bf16-to-float conversion without Torch, datasets, or graph-editor tensor flow.
- [x] Extend checkpoint payload load smoke with `--checkpoint-load-tensor NAME`, selecting named tensors from the decoded layout before CUDA copy/Tile conversion so native inference can prove per-weight checkpoint loads without graph-editor tensors.
- [x] Add checkpoint-backed logits smoke with `--checkpoint-logits-smoke`, loading native checkpoint embeddings/final norm and running token embedding, position embedding, residual add, final LayerNorm, and tied LM-head logits on CUDA Tile kernels without Torch or graph-editor tensors.
- [x] Add checkpoint-backed QKV smoke with `--checkpoint-qkv-smoke`, loading native checkpoint embeddings plus selected-block `ln_1` and `attn.c_attn` tensors and running embedding residual, block LayerNorm, and QKV projection on CUDA Tile kernels without Torch or graph-editor tensors.
- [x] Add checkpoint-backed attention smoke with `--checkpoint-attention-smoke`, extending the QKV smoke through split-to-heads, causal scaled-dot-product attention, and merge-heads on CUDA Tile kernels without Torch or graph-editor tensors.
- [x] Add checkpoint-backed attention residual smoke with `--checkpoint-attention-residual-smoke`, extending the attention smoke through `attn.c_proj` and residual add on CUDA Tile kernels without Torch or graph-editor tensors.
- [x] Add checkpoint-backed block smoke with `--checkpoint-block-smoke`, extending the attention residual smoke through `ln_2`, MLP fc, GELU+bias, MLP projection, and final block residual add on CUDA Tile kernels without Torch or graph-editor tensors.
- [x] Add checkpoint-backed block logits smoke with `--checkpoint-block-logits-smoke`, extending the block smoke through final LayerNorm and tied LM-head logits for the last prompt token on CUDA Tile kernels without Torch or graph-editor tensors.
- [x] Add checkpoint-backed full-stack forward logits smoke with `--checkpoint-forward-logits-smoke`, running every checkpoint GPT block, final LayerNorm, and tied LM-head logits on CUDA Tile kernels while reporting `transformer_blocks_executed: true` and keeping graph-editor tensors out of the path.
- [x] Make dense GPT `--startup-only` suppress final checkpoint export at the native C++ write gate, reporting `checkpoint_export_enabled: false` and `checkpoint_export_startup_only_elided: true` so startup measurements cannot include BF16 checkpoint packing, D2H checkpoint payload copies, or filesystem writes.
- [x] Mirror `checkpoint_export_enabled` into dense GPT native runtime JSON, not just plan JSON, while keeping `final_checkpoint_export_enabled` as the older benchmark-parser alias. The 2026-06-25 linked startup-only RTX 5090 probe reported both top-level fields as `false`, nested `checkpoint.requested: true`, and `checkpoint.startup_only_elided: true` with `torch_required: false` and `graph_editor_tensor_flow: false`.
- [x] Require optimized attention for dense GPT native training by default: if the Tile ABI launches the scalar attention fallback, the trainer fails before final checkpoint export and reports `optimized_attention_required: true`, with `--allow-scalar-attention-fallback` reserved for diagnostics.
- [x] Remove the dense GPT-2 external `llm.kittens` training bridge from normal CLI, SDK, and C++ trainer dispatch. `tools/bench_native_gpt_sm120_parity.sh` remains as a same-script reference benchmark against llm.kittens, while normal training accepts only `tile-cuda`.
- [x] Move dense GPT startup flag translation into the compiled `nfn_native_train` frontend so `--dataset tinystories`, `--output`, `--kernel-backend`, `--template` / `--preset`, `--graph`, `--native-cuda-*`, default `--train-transformer-lm`, default `--backend tile-cuda`, default TinyStories alias fallback, and GPT-3's implicit 2048 context no longer require the Python `nfn train` / `train_gpt.py` argument shim.
- [x] Add opt-in CUDA-event packed-attention backward section timing (`NFN_NATIVE_GPT_ATTENTION_BACKWARD_SECTION_TIMING=1`) so dprep and TK backward costs are reported separately in native GPT runtime JSON without using Torch, Python tensors, or graph-editor nodes.
- [ ] Close the remaining measured SM120 throughput gap between the NeuralFn-owned `libnfn_native_train_tile_ops.so` loop and the `llm.kittens` SM120 reference script.
  - 2026-06-23 added `tools/bench_linear_backward_candidate.sh` and the native
    `linear_backward_bench` C++ harness so new block-backward and LM-head linear
    kernels can be compared against the current raw Tile C ABI symbols in one
    CUDA process before spending time on full trainer-loop parity. Profiles now
    cover `mlp-proj`, `mlp-fc`, `qkv`, `attn-proj`, and `lm-head` dInput/dWeight
    shapes with CUDA-event timing and optional no-regression gates.
  - 2026-06-23 added `tools/bench_native_gpt_linear_hot_matrix.sh` to run those
    hot GPT linear profiles as one matrix gate. It keeps each profile as a
    same-process baseline-vs-candidate CUDA event comparison, supports
    operation-wide and profile-specific candidate symbols, and emits aggregate
    `native_gpt_linear_hot_matrix` JSON so block-backward/LM-head candidates can
    be rejected before noisy full-loop parity runs.
  - 2026-06-24 fixed the standalone linear and LM-head backward microbench
    wrappers so their default `dedicated` CUDA selector resolves to the
    display-disabled GPU index from `nvidia-smi` instead of exporting the
    literal string `CUDA_VISIBLE_DEVICES=dedicated`, which hid all CUDA devices
    from `cudaSetDevice`.
  - 2026-06-24 reran the trainer-chunk LM-head backward microbench after the
    selector fix:
    `NFN_LM_HEAD_BACKWARD_PROFILE=trainer-chunk
    NFN_LM_HEAD_BACKWARD_CANDIDATE_FIRST=1
    NFN_LM_HEAD_BACKWARD_JSON_OUT=/tmp/nfn_lm_head_trainer_chunk_current.json
    bash tools/bench_lm_head_backward_candidate.sh`. The benchmark now reaches
    CUDA, but the strict candidate still reports
    `candidate_true_fused_capability: false`,
    `candidate_sequence_wrapper_only: true`, and
    `candidate_to_baseline_ms_per_iter_ratio: 1.001063`, so the next parity
    slice remains a real fused/cooperative LM-head classifier-backward kernel
    rather than another wrapper/default promotion.
  - 2026-06-23 hardened the focused LM-head and linear-backward benchmark
    wrappers so `CUDA_VISIBLE_DEVICES=auto` falls back to device `0` when
    `nvidia-smi` cannot query GPUs. This preserves a useful C++ CUDA error in
    sandboxed or driver-mismatch environments instead of silently failing before
    the benchmark starts.
  - 2026-06-23 added `NFN_LINEAR_BACKWARD_CANDIDATE_FIRST=1` /
    `--candidate-first` for the linear-backward C++ benchmark. JSON now records
    `run_order`, so close kernel candidates can be checked in both
    baseline-first and candidate-first order before deciding whether a full
    trainer-loop parity run is meaningful.
  - 2026-06-24 added `candidate_symbol_changed` to the linear-backward C++
    benchmark JSON plus `NFN_LINEAR_BACKWARD_REQUIRE_ROUTE_CHANGE=1` in the
    wrapper. New block-backward/LM-head linear candidates can now fail before
    ratio gates when they accidentally compare a symbol against itself. The hot
    matrix wrapper exposes the same guard as
    `NFN_LINEAR_HOT_MATRIX_REQUIRE_ROUTE_CHANGE=1` and forwards it into each
    per-profile benchmark.
  - 2026-06-24 tightened the hot matrix aggregate JSON to carry
    `candidate_symbol_changed_count`, `same_symbol_profile_count`,
    `measurement_only_profile_count`, and `route_change_failure_reason`. This
    marks same-symbol sweeps as measurement-only evidence, preventing raw
    baseline-repeat timing noise from being mistaken for a promotable kernel
    candidate.
  - 2026-06-23 first isolated profile sweep with the new harness ranked the
    current padded-vocab LM-head linear calls as the largest standalone targets:
    `lm-head-dinput` was about `32.17 ms` per 49152-row chunk and
    `lm-head-dweight` about `15.04 ms`, while block-linear calls were low
    single-digit milliseconds (`mlp-proj-dweight` about `1.69 ms`,
    `mlp-fc-dweight` about `1.72 ms`, `qkv-dweight` about `1.00 ms`). The
    first no-warmup MLP projection dInput run exposed setup/cache bias, so keep
    `NFN_LINEAR_BACKWARD_WARMUP>=1` for candidate comparisons.
  - 2026-06-23 added explicit forced-cuBLASLt raw C ABI candidate symbols for
    the padded-vocab LM-head strided BF16 dInput/dWeight calls and exposed them
    as `NFN_LINEAR_BACKWARD_PROFILE=lm-head-dinput-cublaslt` /
    `lm-head-dweight-cublaslt`. The isolated CUDA 13.3 RTX 5090 benchmark
    rejected default promotion: dInput regressed to `1.017720x` and dWeight to
    `1.000576x` versus the current raw ABI symbols. Keep focusing LM-head work on
    a materially different fused/cooperative logits/CE/dHidden/dWeight route,
    not plain cuBLASLt substitution for the padded-vocab strided GEMMs.
  - 2026-06-22 after the no-loss LM-head CE specialization became the default, a 10-step same-script parity sample still measured NeuralFn at `1.021069x` train-loop wall time and `0.978059x` tokens/sec versus `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh`. A one-step attention-section profile split packed-attention backward into `30.440 ms` dprep and `233.277 ms` TK body over 96 launches, so dprep-only work cannot close the gap. Rechecked compile-time `LLMK_SM120_ATOMIC_DQ` via `NFN_SM120_NATIVE_CANDIDATE_PROFILE=attention_atomic_dq` after the CE default; keep it rejected because the 3-step, 3-sample dedicated RTX 5090 run measured `1.139065x` train-loop wall time, `0.877916x` tokens/sec, `2.251237x` `stage.block_backward.attn_sdpa.total_ms`, and `2.470843x` attention TK timing versus the current default.
  - 2026-06-22 made the same-script parity wrapper pass `--train-loss-every-steps 0` to the NeuralFn side by default, with `NFN_SM120_PARITY_TRAIN_LOSS_EVERY_STEPS` / `NFN_SM120_TRAIN_LOSS_EVERY_STEPS` as opt-ins, so short parity runs no longer inherit the raw C++ trainer's default train-loss interval. The verification rerun confirmed `train_loss_host_d2h_count: 0` and `lm_head_classifier_loss_bin_launch_count: 0` over 10 steps, but still measured a real gap: llm.kittens `2435.148 ms/step` and `215519.5 tok/s` versus NeuralFn `2501.160 ms/step` and `209618 tok/s`, or `1.027108x` train-loop wall time and `0.972617x` tokens/sec.
  - 2026-06-22 added opt-in NeuralFn CUDA-event train-loop timing behind `NFN_NATIVE_GPT_TRAIN_LOOP_EVENT_TIMING=1`, enabled by default from the llm.kittens parity wrapper with `NFN_SM120_PARITY_TRAIN_LOOP_EVENT_TIMING=0` as the opt-out. Runtime JSON now reports all-step and first-step-excluded event timings, and `tools/paired_kernel_speed.py` maps llm.kittens step logs to the same `train_loop_cuda_event_*` metric names. Use these fields on future parity runs to distinguish actual GPU training-loop time from host wall/setup/validation noise.
    - Verification on the dedicated RTX 5090 10-step parity sample measured host wall `1.021398x`, all-step CUDA-event wall `1.021390x`, first-step-excluded CUDA-event wall `1.022219x`, and tokens/sec `0.976866x` versus llm.kittens. The metric alignment confirms the remaining gap is not a first-step or host-wall accounting artifact.
  - 2026-06-21 after the RTX 5090 utilization counter recovered, a short 3-step same-script parity smoke still failed the strict gate: llm.kittens measured `2492.030 ms/step` while NeuralFn measured `2544.767 ms/step`, or `1.021162x` train-loop wall time. A second 3-step run with native stage timing measured llm.kittens at `2458.560 ms/step` and NeuralFn at `2535.820 ms/step` (`1.031425x`). Current hot native buckets over the 3-step stage-timed run were `stage.block_backward.total_ms` `3791.070`, `stage.train.model_forward.total_ms` `1977.110`, `stage.block_forward.total_ms` `1965.690`, and `stage.lm_head_backward.total_ms` `1758.480`; the largest block-backward children were `mlp_proj` `977.502`, `attn_sdpa.to_qkv` `806.098`, `mlp_fc` `790.279`, and `qkv` `598.053`. Continue with new kernel work in block backward plus LM-head logits/dHidden/dWeight rather than default-switch promotion.
  - 2026-06-18 added `lm_head_classifier_strategy_contract` to native dense GPT plan/runtime JSON so parity artifacts explicitly compare the llm.kittens full resident BF16 classifier-logit contract with NeuralFn's row-chunked BF16 in-place CE path. At the default `64 x 1024` shape this reports 65,536 full rows vs the 8,192-row NeuralFn chunk, 6.59GB reference-style BF16 logits vs 825.8MB resident NeuralFn BF16 logits, and an 8x resident-logit reduction. This does not close the throughput gap; it makes the next fused classifier/LM-head-backward or memory-gated full-logit candidate measurable against the correct tradeoff.
  - 2026-06-18 made `tools/paired_kernel_speed.py` print both stdout and stderr tails on nonzero child exits, so CUDA driver/runtime failures from external baselines are visible when same-script SM120 parity runs fail before writing JSON. The current sandbox still reports `nvidia-smi` as blocked by the OS, so fresh GPU parity numbers must be collected only after driver access is visible to the benchmark process.
  - 2026-06-18 rejected a memory-tradeoff attempt to fit full resident LM-head logits by combining `--lm-head-row-chunk-size 65536` with `NFN_NATIVE_GPT_STORE_MLP_BLOCKS=6`. The dedicated RTX 5090 one-step same-script candidate changed the classifier contract to one 65,536-row chunk but regressed train-loop wall time to `3.325363x`, tokens/sec to `0.300719x`, LM-head backward to `1.260466x`, and block backward to `5.086636x` because attention/MLP recompute dominated. Keep the 8,192-row LM-head chunk default and do not trade away stored MLP tape for full logits.
  - 2026-06-18 historical note, superseded by later launcher policy: an early wrapper-only `CUDA_MODULE_LOADING=LAZY` candidate measured `0.997976x` train-loop wall time and `1.002029x` tokens/sec, but total wall time stayed neutral at `0.999180x` and setup regressed to `1.017474x` with token init at `1.029699x`. Current native launchers and the master `nfn train` native dispatcher set lazy module loading by default when the caller has not supplied a value.
  - 2026-06-18 CUDA 13.3 refresh: the dedicated RTX 5090 no-sidecar 10-step parity sample measured llm.kittens at `2467.658 ms/step` and `212912.5 tok/s`; NeuralFn measured `2545.570 ms/step` and `205961 tok/s`, or `1.031573x` train-loop wall time and `0.967350x` tokens/sec versus the reference. A current stage/shape profile still points at `block_backward`, `lm_head_backward`, and LM-head GEMMEx logits/dHidden as the highest-value remaining work.
  - 2026-06-18 CUDA 13.3 one-step stage/shape profile refreshed the hot buckets at `2891.86 ms` train-loop wall and `181298 tok/s` with stage timing enabled: `block_backward` `1365.58 ms`, `lm_head_backward` `834.428 ms`, `train.model_forward` `663.252 ms`, `block_backward.mlp_proj` `361.877 ms`, `lm_head_backward.logits` `334.330 ms`, `block_backward.mlp_fc` `290.295 ms`, `block_backward.attn_sdpa.to_qkv` `269.354 ms`, and `lm_head_backward.dhidden` `246.133 ms`. The top linear buckets were still LM-head BF16 GEMMEx logits `50304,8192,768,T,N` at `330372 us` and LM-head dHidden `768,8192,50304,N,N` at `241937 us`, followed by LM-head dWeight cuBLASLt `768,50304,8192,N,T` at `176041 us`.
  - 2026-06-18 rejected LM-head dWeight cuBLASLt heuristic retunes under CUDA 13.3: shape override `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,50304,8192,N,T,0` measured `1.002237x` train-loop wall time and `0.998182x` tokens/sec, while index `2` measured `1.027506x` train-loop wall time and `0.973415x` tokens/sec versus the current index-1 default. Keep the current heuristic and focus LM-head work on a fused logits/CE/dHidden path rather than more heuristic golf.
  - 2026-06-18 rejected rechecking `NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF=1` under CUDA 13.3 for the hot LM-head GEMMEx buckets: the dedicated RTX 5090 5-step, 3-sample paired run measured `1.007790x` train-loop wall time, `0.992298x` tokens/sec, and `1.007315x` total wall time versus the current `CUBLAS_COMPUTE_32F` GEMMEx default.
  - 2026-06-18 rejected a BF16 CE vector-load candidate for NeuralFn's row-chunked LM-head classifier path. The candidate added int4 vector loads to the u16-target public-vocab BF16 CE kernel and passed a one-step CUDA smoke, but the dedicated RTX 5090 5-step, 3-sample paired run measured `1.017256x` train-loop wall time, `0.983185x` tokens/sec, and `1.017205x` total wall time versus the current scalar-load CE default, so the code was removed rather than kept as another slow diagnostic switch.
  - 2026-06-18 rechecked `NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC=1` after the CUDA 13.3 reinstall and keep it rejected for startup: the 5-sample startup-only native-vs-native run measured `1.180688x` setup wall time, `1.180297x` total startup wall time, `1.384994x` float-arena materialization, and `1.639900x` uint16-arena materialization versus default.
  - 2026-06-18 replaced the earlier hardcoded cuBLASLt heuristic-index note with v2 shape-stat evidence: under CUDA 13.3 on the dedicated RTX 5090, the hot MLP projection dWeight shape `3072,768,65536,N,T` reports `cublaslt_returned_heuristics: 1`, `cublaslt_selected_heuristic: 0`, and a 66 MB workspace, so the previous index-1 fallback was a no-op and has been removed. Keep `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=3072,768,65536,N,T,INDEX` diagnostic-only unless a future CUDA/cuBLASLt build reports multiple returned heuristics and a paired win.
  - 2026-06-18 rejected cuBLASLt heuristic index 2 for the hot MLP FC dWeight shape `768,3072,65536,N,T`: the dedicated RTX 5090 native-vs-native 5-step, 3-sample confirmation with `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,3072,65536,N,T,2` measured `1.012398x` train-loop wall time, `0.988108x` tokens/sec, and `1.011762x` total wall time versus the current default.
  - 2026-06-18 rejected revisiting `--lm-head-row-chunk-size 4096` after the then-current heuristic defaults: the dedicated RTX 5090 native-vs-native 5-step, 2-sample check halved the resident LM-head BF16 logit workspace and improved setup wall time to `0.974574x`, but regressed train-loop wall time to `1.008698x`, tokens/sec to `0.991400x`, and total wall time to `1.007341x` versus the 8192-row default.
  - 2026-06-18 rejected `NFN_NATIVE_LINEAR_BF16_OUTPUT_CUBLASLT=1` for LM-head logits: the dedicated RTX 5090 native-vs-native 5-step, 3-sample run measured `1.003620x` train-loop wall time, `0.996405x` tokens/sec, and `1.002305x` total wall time versus the GEMMEx default.
  - 2026-06-18 historical note: the old pre-wrapper attempt to re-enable TK BF16 forward for the padded LM-head logits shape with `NFN_NATIVE_LINEAR_TK_FORWARD_ENABLE_SHAPE=50304,8192,768,T,N` measured `1.007997x` train-loop wall time, `0.992076x` tokens/sec, and `1.006142x` total wall time versus the then-current GEMMEx fallback. This was superseded by the 2026-06-19 BF16-output wrapper wiring and default TK bridge promotion.
  - 2026-06-18 rechecked the fast int32-index Tile token-weight initializer after the CUDA 13.3 reinstall and kept it as the current default. The dedicated RTX 5090 startup-only native-vs-native 5-sample run compared default int32 against `NFN_NATIVE_GPT_TOKEN_WEIGHT_FAST_INT32_INIT=0`; the opt-out measured `1.014432x` setup wall time, `1.013372x` token-init time, and `1.014616x` total startup versus the int32 default. Keep the int64 Tile-index path opt-in for bisection only.
  - 2026-06-18 refreshed the current stage/shape profile after the token-initializer rejection using CUDA 13.3 on the dedicated RTX 5090. One profiled optimizer step measured `3017.16 ms` train-loop wall and `173769 tok/s` with instrumentation enabled; hot stages remain `block_backward` (`1427.33 ms`), `lm_head_backward` (`875.083 ms`), and `train.model_forward` (`684.007 ms`). LM-head CE itself is only `70.008 ms`; the LM-head cost is dominated by logits GEMMEx `50304,8192,768,T,N` (`352055 us`), dHidden GEMMEx `768,8192,50304,N,N` (`250490 us`), and dWeight cuBLASLt `768,50304,8192,N,T` (`177449 us`). Next useful work is a new fused LM-head classifier/backward kernel or a materially different GEMM route, not another rejected chunk/heuristic switch.
  - 2026-06-18 current no-sidecar parity sample after the CUDA 13.3 rebuild measured llm.kittens at `2782.612 ms/step` and `189520.7 tok/s`; NeuralFn measured `2903.160 ms/step` and `180592 tok/s`, or `1.043322x` train-loop wall time and `0.952888x` tokens/sec versus the reference. The candidate still reports an 8x smaller resident LM-head logit chunk (`824180736` BF16 bytes vs `6593445888` reference-style bytes), so the remaining parity gap is the row-chunked LM-head GEMM schedule plus block-backward buckets rather than Python/Torch or graph-editor flow.
  - 2026-06-19 after reinstalling CUDA Toolkit 13.3 for WSL, reran the CUDA-visible correctness gates on the dedicated RTX 5090: the Tile CUDA GPU suite passed with `537 passed, 6 warnings`, native GPT contract tests passed with `51 passed, 1 skipped`, Tile registry/example tests passed with `28 passed`, `tools/check_native_no_torch_deps.py` reported all native training/inference checks `ok`, and both `build/nfn_gpt_native_train` plus `build/libnfn_native_train_tile_ops.so` rebuilt cleanly. The refreshed 5-step, 2-sample parity run still measured NeuralFn at `2568.120 ms/step` and `204152.5 tok/s` versus llm.kittens at `2486.577 ms/step` and `210984.6 tok/s`, or `1.032833x` train-loop wall time. This confirms the current blocker is a real native kernel throughput gap, not failing tests, Torch dependency leakage, graph-editor tensor flow, or stale CUDA installation state.
  - 2026-06-19 rejected forward-logit reuse memory-trade variants under CUDA 13.3. `NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS=1 NFN_NATIVE_GPT_STORE_MLP_BLOCKS=0` cut `lm_head_backward` to `0.665538x` but regressed total train-loop wall time to `1.364953x` because `block_backward.mlp_fc` grew to `1.628808x`. `NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS=1 NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_ACTIVATIONS=0` cut `lm_head_backward` to `0.680357x` but still regressed train-loop wall time to `1.103069x` because attention recompute increased forward/TK work. Keep full forward-logit reuse diagnostic-only; fitting it by dropping current activation stores loses more than the LM-head recompute saves.
  - 2026-06-19 superseded: the old `NFN_NATIVE_LINEAR_TK_FORWARD_ENABLE_SHAPE=50304,8192,768,T,N` rejection covered the pre-wrapper path. The current CUDA 13.3 default tries the TK BF16 forward bridge for the padded LM-head logit shape; use `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=50304,8192,768,T,N` to reproduce the older GEMMEx route.
  - 2026-06-19 rechecked BF16-output cuBLASLt for LM-head logits with `NFN_NATIVE_LINEAR_BF16_OUTPUT_CUBLASLT=1`. The candidate increased train-loop wall time to `1.002706x` and `lm_head_backward` to `1.005947x` while merely shifting LM-head logits from GEMMEx to cuBLASLt. Keep the GEMMEx fallback default for row-chunked LM-head logits.
  - 2026-06-19 rejected overlapping LM-head dHidden and dWeight with a side CUDA stream. The prototype recorded CE completion on the default stream, launched dHidden on a non-blocking side stream, ran dWeight on the default stream, and synchronized before reusing the row-chunk logit buffer. The dedicated RTX 5090 5-step, 2-sample native-vs-native run measured `1.003463x` train-loop wall time and `0.996573x` tokens/sec versus the serial default, with the range crossing noise. The prototype was removed; keep LM-head work focused on fused/cooperative kernels rather than stream-level overlap of two large GEMMs.
  - 2026-06-19 post-reinstall rechecks kept the remaining easy LM-head switches rejected: no-stage-timing parity still measured NeuralFn at `1.027170x` train-loop wall time versus llm.kittens, `NFN_NATIVE_GPT_CE_BF16_EXP2=1` measured `1.001188x` train-loop wall time, `--lm-head-row-chunk-size 16384` measured `1.017714x`, `--lm-head-row-chunk-size 4096` measured `1.008782x`, `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,50304,8192,N,T,0` measured `1.000315x`, and `NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF=1` measured `1.001155x`. A shape-stats probe reconfirmed the LM-head hot buckets as logits GEMMEx `50304,8192,768,T,N`, dHidden GEMMEx `768,8192,50304,N,N`, and dWeight cuBLASLt `768,50304,8192,N,T`. Do not spend more time on chunk-size/heuristic toggles unless a future CUDA/cuBLAS build changes these shape stats materially.
  - 2026-06-19 after installing CUDA Toolkit 13.3.33 for WSL, reran the cached failure surface directly on the dedicated RTX 5090: `NFN_TILE_CUDA_TEST=1 python -m pytest --lf -q -rs` passed with `1167 passed, 20 warnings, 468 subtests passed`. `nvcc --version` reported `V13.3.33`, `nvidia-smi` reported CUDA UMD `13.3` with no active GPU compute processes before the run, `bash tools/build_native_gpt_cli.sh` and `bash tools/build_native_train_tile_ops.sh` rebuilt the native CUDA artifacts, and `python tools/check_native_no_torch_deps.py` reported all native entrypoints `ok`. The fresh same-script 5-step, 2-sample parity run measured llm.kittens at `2481.764 ms/step` and NeuralFn at `2568.290 ms/step`, or `1.034910x` NeuralFn train-loop wall time; this confirms the retest surface is green and the remaining issue is kernel throughput in `block_backward` plus `lm_head_backward`.
  - 2026-06-19 follow-up after the CUDA 13.3.33 extension-load fix remeasured the dedicated RTX 5090 parity surface with stage timing enabled: llm.kittens measured `2507.726 ms/step` and NeuralFn measured `2602.880 ms/step`, or `1.037944x` train-loop wall time and `0.963012x` tokens/sec. Shape stats on the current default still show LM-head logits `50304,8192,768,T,N` and dHidden `768,8192,50304,N,N` on GEMMEx/SGEMM-family fallback buckets, with LM-head dWeight `768,50304,8192,N,T` on cuBLASLt. Same-script native-vs-native rechecks kept the easy switches rejected: `NFN_NATIVE_GPT_LM_HEAD_DWEIGHT_BEFORE_DHIDDEN=1` measured `1.001614x`, `NFN_NATIVE_GPT_LM_HEAD_REVERSE_CHUNKS=1` measured `1.001020x`, compile-time `NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE=8192` measured `1.002704x` startup total wall and `1.028856x` token init, `NFN_NATIVE_GPT_MLP_PROJ_DINPUT_BEFORE_DWEIGHT=1` measured `0.999577x` train-loop wall but stayed noise-equivalent, the all-projection dInput-before-dWeight combo measured `0.999913x`, `NFN_TILE_CUDA_LINEAR_TK_DINPUT_ENABLE_SHAPE=768,8192,50304,N,N` measured `1.023513x`, `NFN_TILE_CUDA_CUBLASLT_HEURISTIC_SHAPE=768,50304,8192,N,T,0` measured `1.001584x`, and `NFN_NATIVE_GPT_LM_HEAD_ROW_CHUNK_SIZE=16384` measured `1.002609x`. Do not promote these switches; the next implementation must be new fused/cooperative LM-head backward work or a materially different GEMM route.
  - 2026-06-19 post-vector4-commit parity refresh still measured a real gap: `NFN_SM120_PARITY_STEPS=5 NFN_SM120_PARITY_SAMPLES=1 ... /tmp/nfn_sm120_parity_current_after_vector4.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2460.796 ms/step` and NeuralFn at `2600.160 ms/step`, or `1.056634x` train-loop wall and `0.945115x` tokens/sec. The candidate still used BF16 GEMMEx for LM-head logits/dHidden and cuBLASLt for LM-head dWeight.
  - 2026-06-19 after the CUDA 13.3 reinstall, rechecked `NFN_NATIVE_GPT_CE_BF16_VEC_STORES=1` over a 5-step, 3-sample native-vs-native run. The candidate measured `1.006100x` train-loop wall, `0.993948x` tokens/sec, and `1.002658x` `stage.lm_head_backward.total_ms`, with no useful route-counter change, so vector stores stay diagnostic-only. `tools/paired_kernel_speed.py` now prints LM-head substages (`logits`, `ce`, `dhidden`, `dweight`, optional concurrent dHidden/dWeight) plus the key block-backward substages in both per-side metrics and candidate-over-baseline ratios, so the next parity slice can attribute coarse `lm_head_backward` and `block_backward` changes from stdout.
  - 2026-06-19 added metric-ratio gates to the same-script benchmark path. `tools/paired_kernel_speed.py --max-candidate-ratio [STAT:]METRIC=RATIO` and `tools/bench_native_gpt_sm120_candidate.sh` via `NFN_SM120_NATIVE_MAX_CANDIDATE_RATIO` / `NFN_SM120_CANDIDATE_MAX_CANDIDATE_RATIO` now fail candidates that regress required hot metrics such as `stage.lm_head_backward.total_ms` or `train_loop_wall_ms_per_step`; `STAT` defaults to `mean` and can be `median`, `min`, or `max`, so borderline noisy candidates should use median gates too. Missing metrics also fail so future kernel work cannot pass without the intended stage attribution.
  - 2026-06-22 added lower-bound metric gates to the same-script benchmark path. `tools/paired_kernel_speed.py --min-candidate-ratio [STAT:]METRIC=RATIO` and the native candidate wrapper aliases `NFN_SM120_NATIVE_MIN_CANDIDATE_RATIO` / `NFN_SM120_CANDIDATE_MIN_CANDIDATE_RATIO` now fail candidates that do not meet baseline throughput or lose required route counters, complementing the existing max-ratio regression gates.
  - 2026-06-22 refreshed the native-vs-native no-op benchmark on the dedicated RTX 5090 after the CUDA 13.3.33 rebuild and SDK binding-resolver work. `tools/bench_native_gpt_sm120_candidate.sh` selected GPU 0 with no compute processes, measured baseline at `2515.313 ms/step` and candidate at `2517.873 ms/step`, and reported no strategy or route-counter changes; the paired wall-clock candidate-over-baseline ratio was `0.998879x` while the native train-loop ratio was `1.001026x` mean / `1.002885x` median. Treat this as the current no-op noise floor: the next default candidate must change a real route counter or beat strict train-loop/stage gates, not rely on noise-equivalent timing.
  - 2026-06-22 added and rejected shape-selective SM120 TK dWeight routing for the then-current 32768-row LM-head bucket. `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_tk_dweight_32768` expands to `NFN_NATIVE_LINEAR_TK_DWEIGHT_ENABLE_SHAPE=768,50304,32768,N,T`; runtime JSON and the paired benchmark now report `linear_tk_dweight_gemm_count`. The dedicated RTX 5090 5-step, 3-sample same-script benchmark proved the route active (`linear_tk_dweight_gemm_count: 0 -> 80`, `linear_cublaslt_gemm_count: 3440 -> 3360`) but rejected it at `1.022262x` train-loop wall time, `0.978245x` tokens/sec, and `1.279309x` `stage.lm_head_backward.dweight.total_ms`. The 2026-06-24 wrapper update added `lm_head_tk_dweight_49152` for the current 49152-row default shape (`NFN_NATIVE_LINEAR_TK_DWEIGHT_ENABLE_SHAPE=768,50304,49152,N,T`), so new TK dWeight bisections no longer benchmark the stale chunk size. The current-shape gate moved `linear_tk_dweight_gemm_count: 0 -> 16` but rejected the route at `1.303473x` train-loop wall, `1.201790x` LM-head dWeight, and `1.557413x` block backward, so the wrapper rejects both TK dWeight profiles by default. Keep the route diagnostic-only; the next LM-head attempt needs a fused/cooperative classifier-backward kernel rather than a TK scratch dWeight substitution.
  - 2026-06-20 after the CUDA Toolkit 13.3.33 WSL reinstall, reran the failure surface on the dedicated RTX 5090: `bash tools/build_native_gpt2_all.sh` rebuilt all native CUDA artifacts, the GPU-visible native/Tile pytest set passed with `240 passed`, `tests/test_template_presets.py -x -q` passed with `26 passed`, and `tools/check_native_no_torch_deps.py --skip-artifacts --json` reported all native training/inference entrypoints clean. The same-script 10-step native candidate rerun remained operationally green but measured `1.012728x` train-loop wall time against the baseline command, while the 2-step stage-timed rerun showed candidate timing crossing noise (`0.993515x` train-loop wall) with no route-counter changes. To stop noisy or slower candidate routes from slipping through manually, `tools/bench_native_gpt_sm120_candidate.sh` now auto-gates measured candidate runs at `train_loop_wall_ms_per_step=1.000` and, with stage timing enabled, gates `stage.lm_head_backward.total_ms`, `stage.block_backward.total_ms`, and `stage.block_backward.mlp_proj.total_ms` at `1.000`; dry-run planning and no-op baseline-vs-baseline checks remain ungated.
  - 2026-06-20 after installing CUDA Toolkit 13.3.33 for WSL, revisited the failed-test surface again on the dedicated RTX 5090. The focused CUDA group passed with `155 passed`, the broader native GPT/Tile examples group passed with `243 passed`, and the full repository suite passed with `781 passed, 403 skipped, 16 warnings, 468 subtests passed`. A fresh 5-step, 3-sample parity run measured llm.kittens at `2473.361 ms/step` and `211960 tok/s`; NeuralFn measured `2541.100 ms/step` and `206326 tok/s`, or `1.027407x` train-loop wall time and `0.973435x` tokens/sec. Stage timing still ranks `block_backward` and `lm_head_backward` as the main gap buckets, so failing tests and stale CUDA installs are no longer valid explanations for the remaining parity gap.
  - 2026-06-20 rejected the remaining token-weight startup candidates under the fresh CUDA 13.3.33 setup. `NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT=1` regressed token init to `1.010306x`; `NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT=1` improved token init to `0.930195x` but regressed overall setup wall time to `1.039387x`. Keep the default fast int32 Tile initializer. `tools/bench_native_gpt_sm120_candidate.sh` now adds `setup.token_weight_init.total_ms=1.000` automatically for startup-only candidate runs whose candidate library path, env, or candidate-only args mention token-weight initialization, so future token-init bisections must pass their own substage gate as well as setup wall time.
  - 2026-06-20 rejected broad cuBLASLt heuristic-policy changes for the native dense GPT hot path. `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_POLICY=min_waves` failed the 3-step, 2-sample stage-timed gate at `1.007645x` train-loop wall, `1.000305x` LM-head backward, and `1.013865x` block backward. `NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_POLICY=max_waves` failed at `1.023965x` train-loop wall, `1.032178x` LM-head backward, and `1.030164x` block backward. Both runs reported no tracked route-counter change, so keep the default selected cuBLASLt heuristic behavior and focus on a real fused/cooperative LM-head or block-backward kernel route.
  - 2026-06-20 added a shape-stat attribution path for the same-script native candidate benchmark. `NFN_SM120_NATIVE_LINEAR_SHAPE_STATS=1` / candidate / parity aliases enable native linear shape stats on both paired commands, and `tools/paired_kernel_speed.py` now reports `native_linear_shape_stats` with per-shape average time ratios and cuBLASLt selected heuristic lists. Use this for the next LM-head or block-backward candidate so the report proves whether the actual hot GEMM shape changed path or algorithm; leave it off for normal throughput gates because shape stats synchronize measured GEMMs.
  - 2026-06-20 added `NFN_NATIVE_LINEAR_BF16_BF16_BGRAD_DISABLE_SHAPE` / `NFN_TILE_CUDA_LINEAR_BF16_BF16_BGRAD_DISABLE_SHAPE` for shape-selective BF16/BF16 dWeight+bias bisection. The broad `NFN_TILE_CUDA_LINEAR_BF16_BF16_BGRAD=0` probe proved too coarse: it improved the MLP projection dWeight substage but also changed MLP FC and QKV dWeight routes and failed the whole-step gate. The isolated `3072,768,65536,N,T` probe with matching cuBLASLt disable did move only the MLP projection dWeight shape and improved that substage to `0.981120x`, but it still failed the total-step and block-backward gates at `1.000764x` and `1.004350x`. Keep the shape gate diagnostic-only; the next production candidate needs a route that improves the whole step, not just one substage.
  - 2026-06-20 rejected an ephemeral full-batch LM-head backward prototype that allocated a 6.59GB BF16 logit buffer plus full-row CE scratch inside `lm_head_backward`, ran one full-row logits GEMM/CE/dHidden/dWeight schedule per microbatch, and freed the workspace before block backward to avoid the earlier resident-logit memory cliff. The one-step RTX 5090 native-vs-native smoke proved the route was active (`lm_head_logits_tk_gemm_count` fell from `64` to `8`, with `8` temporary allocations and `52.75GB` cumulative temporary workspace), but it regressed train-loop wall time to `5.555959x`, `lm_head_backward` to `19.114675x`, logits to `6.086396x`, CE to `13.482386x`, dHidden to `19.768439x`, and dWeight to `14.609672x`. The prototype was removed. Do not pursue reference-style full-row LM-head GEMMs through temporary full logits; the next LM-head attempt needs a true fused/cooperative row-chunked classifier-backward kernel that preserves the current 8192-row resident cap.
  - 2026-06-20 rechecked the current row-chunked BF16/u16 CE classifier switches under CUDA 13.3.33 after the clean rebuild. `NFN_NATIVE_GPT_CE_BF16_VEC_STORES=1` consistently improved only the CE substage (`0.975309x` mean over a 2-step, 3-sample stage-timed run) but failed the whole-step gate at `1.005911x` train-loop wall time and had no route-counter change. Combining vector stores with `NFN_NATIVE_GPT_CE_BF16_EXP2=1` kept the CE substage at `0.975134x` but still failed the whole-step gate at `1.000262x`. Keep both switches diagnostic-only; CE is now a small slice of `lm_head_backward`, so parity work should not default a timing-only CE tweak that fails the total-step gate.
  - 2026-06-20 rechecked `NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC=1` after confirming CUDA Toolkit 13.3.33 and CUDA UMD 13.3 are loaded. The 1-step, 3-sample native-vs-native startup/runtime gate measured setup wall time at `1.156567x`, total wall time at `1.028093x`, float-arena materialization at `1.338616x`, and uint16-arena materialization at `1.670007x` versus the default allocator, while train-loop timing remained noise-equivalent. Keep cudaMallocAsync opt-in only; it does not solve the startup gap on the dedicated RTX 5090.
  - 2026-06-20 current CUDA 13.3 parity line after rebuilding `build/nfn_gpt_native_train`, `build/nfn_native_train`, `build/libnfn_native_train_tile_ops.so`, and the native GPT Python binding: the 5-step, 3-sample llm.kittens-vs-NeuralFn gate measured llm.kittens at `2495.348667 ms/step` and NeuralFn at `2522.440 ms/step`, or `1.010870x` train-loop wall time / `0.988745x` tokens/sec. The remaining production target is still a memory-safe fused/cooperative LM-head classifier-backward path: llm.kittens uses full resident logits plus fused classifier, while NeuralFn's bounded chunked classifier avoids the 6.59GB resident-logit cliff but pays extra row-chunked logits/CE/dHidden/dWeight work.
  - 2026-06-20 retuned the dense GPT native LM-head row chunk default from 8192 to 32768 rows after the CUDA 13.3 reinstall. The dedicated RTX 5090 native-vs-native 5-step, 3-sample benchmark cut LM-head logits/classifier chunk launches from 320 to 80 and measured `0.998625x` train-loop wall time / `1.001388x` tokens/sec versus 8192 rows. The 16384-row candidate regressed at `1.008471x`; the 65536-row full-batch candidate timed out at the 300s sample limit and left the WSL GPU utilization counter stuck, so it remains rejected.
  - 2026-06-20 hardened the native runner against the rejected 65536-row LM-head chunk path: effective chunks above 32768 rows now fail during native plan validation unless `NFN_NATIVE_GPT_ALLOW_UNSAFE_LM_HEAD_ROW_CHUNK=1` is set for explicit diagnostics. This keeps future paired benchmark runs from accidentally relaunching the path that left the WSL GPU utilization counter stuck at 100% with no visible process.
  - 2026-06-23 promoted the dense GPT native LM-head row chunk default from 32768 to 49152 rows after the CUDA 13.3 dedicated RTX 5090 5-step, 3-sample same-script gate passed at `0.992974x` train-loop wall time and `0.998563x` LM-head backward versus the old 32768-row route. The SM120 candidate wrapper now pins the 32768 baseline for `lm_head_row_chunk_49152` and exposes `lm_head_row_chunk_32768` as the inverse rollback check. A fresh 2026-06-23 CUDA 13.3 dedicated RTX 5090 3-step, 2-sample stage-timed rerun rejected that inverse rollback at `1.000594x` train-loop wall, `1.001939x` steady-state CUDA-event wall, `1.000885x` LM-head backward, `1.000224x` block backward, and `1.000409x` MLP projection, so 49152 remains the default and the rollback profile now requires `NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1` for real launches.
  - 2026-06-25 converted `lm_head_row_chunk_65536` from a timeout-prone profile to a rejected-by-default profile after the CUDA 13.3.33 dedicated RTX 5090 one-step current/native/reference gate proved the unsafe full-row route active but regressed train-loop wall to `7.368793x` versus current native and `8.037520x` versus llm.kittens. It also cut tokens/sec to `0.135708x` versus current and collapsed `stage.block_backward.attn_sdpa.to_qkv.total_ms` to `63.207371x`. Keep this as a historical diagnostic only; intentional reruns require `NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1`.
  - 2026-06-23 added the explicit `lm_head_cooperative_no_loss_backward` same-script profile for the cooperative LM-head sequence wrapper after the no-loss CE flag fix. It expands to `NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1 NFN_NATIVE_GPT_LM_HEAD_CLASSIFIER_CE_NO_LOSS=1 NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_DEFAULT_SPECIALIZED=1` plus `--train-loss-every-steps 0`, but remains rejected by default because the dedicated RTX 5090 one-step stage-timed gate measured `1.117578x` train-loop wall, `1.294010x` LM-head backward, and `0.894788x` tokens/sec versus the serial default. Do not promote the cooperative sequence wrapper without a true fused/cooperative classifier-backward kernel.
  - 2026-06-23 raised the native safe LM-head row-chunk cap to 49152 rows. The 65536-row full-batch route remains unsafe/time-prone and still requires `NFN_NATIVE_GPT_ALLOW_UNSAFE_LM_HEAD_ROW_CHUNK=1`.
  - 2026-06-19 rejected `CUDA_DEVICE_MAX_CONNECTIONS=8` for native GPT defaults: the 5-step, 3-sample same-script native-vs-native run measured `1.004548x` train-loop wall, `0.995488x` tokens/sec, and `1.121564x` setup wall versus the existing `CUDA_DEVICE_MAX_CONNECTIONS=1` default.
  - 2026-06-19 retested `NFN_NATIVE_GPT_MLP_PROJ_DINPUT_BEFORE_DWEIGHT=1` over 5 paired samples. The mean train-loop wall was `0.998206x`, but median was `1.000733x`, total wall was `0.999022x`, and the range crossed noise, so it remains diagnostic-only rather than a default.
  - 2026-06-19 rejected full forward LM-head logit reuse again on the current build: `NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS=1` timed out at the 300-second candidate limit for a 5-step run. The default row-chunked recompute path remains necessary until there is a fused/cooperative classifier-backward kernel that avoids the full resident-logit memory cliff.
  - 2026-06-19 added and rejected a diagnostic full-batch resident LM-head schedule behind `NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS=1 NFN_NATIVE_GPT_FULL_BATCH_LM_HEAD_REUSE=1`. The path replaces the old pattern that paid the full-logit allocation but still ran LM-head work as 8192-row chunks: a one-step CUDA 13.3.33 RTX 5090 smoke measured `stage.lm_head_backward.total_ms` at `0.652349x` versus the default and showed full-row substage averages of `train.lm_head_logits=28.176 ms`, `lm_head_backward.ce=8.663 ms`, `lm_head_backward.dhidden=22.471 ms`, and `lm_head_backward.dweight=20.151 ms` per microbatch. It remains off by default because full resident logits made downstream `block_backward.attn_sdpa` collapse to `271.687281x` and total train-loop wall time to `32.086572x`. Do not promote full-logit reuse without first removing the resident-logit memory cliff.
  - 2026-06-19 added low-overhead native GPT LM-head logits backend counters (`lm_head_logits_tk_gemm_count`, `lm_head_logits_cublaslt_gemm_count`, and `lm_head_logits_bf16_gemm_count`) so normal no-shape-stats benchmark JSON can identify the active logits route. A one-step CUDA smoke with `linear_shape_stats_enabled: false` reported `lm_head_logits_linear_strategy: "tk-sm120-bf16-gemm-default"` and `lm_head_logits_tk_gemm_count: 64`, avoiding the previous misleading GEMMEx label when shape stats were off.
  - 2026-06-19 defaulted `nfn_native_tile_float32_to_bf16_bits` to a guarded vec4 CUDA path for aligned native GPT buffers, with `NFN_NATIVE_GPT_F32_TO_BF16_VEC4=0` / `NFN_TILE_CUDA_F32_TO_BF16_VEC4=0` for scalar fallback bisection. The dedicated RTX 5090 5-step, 3-sample same-script run measured scalar opt-out at `1.003731x` train-loop wall and `0.996281x` tokens/sec versus vec4 default; `block_backward.mlp_proj.grad_out_bf16` improved from roughly `114-122 ms` to `83-85 ms` over the stage-timed runs.
  - 2026-06-19 rechecked `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_BF16=0` as a startup tradeoff. Startup-only timing improved setup wall to `0.982959x` and total wall to `0.982962x`, and the full train benchmark improved setup wall to `0.938130x`, but train-loop wall regressed to `1.005951x` and total wall was `1.003292x`. Keep the saved packed-attention LN1 BF16 tape enabled for the default training path.
  - 2026-06-19 rejected two activation-tape memory/startup candidates as defaults because train-loop throughput regressed. `NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUTS=1` reduced setup wall time to `0.944457x` but increased train-loop wall time to `1.014826x` and reduced tokens/sec to `0.985409x`. `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_BF16=0` reduced setup wall time to `0.960234x` but increased train-loop wall time to `1.004923x` and reduced tokens/sec to `0.995113x`. Keep both diagnostic-only; setup-only memory savings do not satisfy the llm.kittens parity target.
  - 2026-06-19 added shape-selective TK dInput routing for future single-shape bisections, then rejected the LM-head dHidden TK candidate. Broad `NFN_NATIVE_LINEAR_TK_DINPUT=1` regressed train-loop wall time to `1.042491x`; the isolated `NFN_NATIVE_LINEAR_TK_DINPUT_ENABLE_SHAPE=768,8192,50304,N,N` candidate still regressed `lm_head_backward` to `1.080834x`, train-loop wall time to `1.019686x`, and tokens/sec to `0.980699x`. Keep TK dInput shape routing diagnostic-only and keep the LM-head dHidden shape on GEMMEx.
  - 2026-06-20 added and rejected a concrete double-buffered LM-head row-chunk pipeline candidate behind `NFN_NATIVE_GPT_LM_HEAD_PIPELINE_CHUNKS=1`. The path allocates two 8192-row BF16 logit chunks, computes logits and CE/dlogits on the default stream, and queues dHidden plus ordered dWeight on side streams. A one-step TinyStories smoke passed and reported two buffers plus `824180736` extra BF16 logit bytes, but the paired 2-step, 3-sample RTX 5090 gate failed at `1.001057x` train-loop wall and `1.009187x` `stage.lm_head_backward.total_ms`. Keep it diagnostic-only; this simple pipeline is not enough to close the LM-head gap.
  - 2026-06-22 rechecked the same `lm_head_pipeline_chunks` profile after the cooperative wrapper and CUDA 13.3 candidate-profile cleanup. The route still enabled and reported the double-buffered LM-head schedule, but the three-step dedicated RTX 5090 gate rejected it at `22.955358x` train-loop wall and `45.070737x` block backward, with attention projection and SDPA backward exploding after the pipeline. Keep it default-off and do not revisit it without redesigning the cross-stream ownership/synchronization model.
  - 2026-06-22 redesigned that ownership model for the opt-in route: side-stream dHidden/dWeight launches now record per-slot done events, and the default stream waits only on the BF16 logit slot being reused instead of synchronizing whole side streams. Runtime JSON reports `lm_head_pipeline_slot_event_wait_count`, `lm_head_pipeline_done_event_record_count`, and the `double-buffered-logits-ce-default-stream-side-stream-dhidden-ordered-dweight-slot-events` strategy. The dedicated RTX 5090 one-step same-script retest proved the slot-event route active (`32` slot waits and `32` done-event records), but still rejected it at `18.448472x` native train-loop wall time versus the serial baseline. Do not revisit side-stream overlap around the same LM-head GEMMs as a default candidate; the next parity step needs a real fused/cooperative classifier-backward kernel or materially different GEMM route.
  - 2026-06-24 reran `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_pipeline_chunks` intentionally after the CUDA reinstall with a short 3-step, 2-sample stage-timed native-vs-native gate. The candidate command timed out after 300 seconds, so the wrapper now treats the profile as rejected by default instead of only timeout-prone. Keep pipeline scheduling off the default path; the remaining LM-head work is the true cooperative logits/CE/dHidden/dWeight kernel or a materially different GEMM route.
  - 2026-06-24 added and rejected `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_prob_only_corrections`. The route writes probability-only no-loss BF16 classifier dlogits and then applies the target subtraction through separate Tile CUDA dHidden/dWeight correction kernels. The dedicated RTX 5090 CUDA 13.3 3-step, 2-sample same-script gate proved the strategy change (`lm_head_ce_kernel_strategy: no-loss-prob-only-dlogits-plus-target-corrections`) but rejected it at `1.005050x` `stage.lm_head_backward.total_ms` and `1.000994x` steady-state CUDA-event step time. Keep this diagnostic-only; extra correction launches/atomics are slower than folding the one-hot subtraction into CE.
  - 2026-06-24 added and rejected `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_prob_only_combined_corrections`. The route still writes probability-only no-loss BF16 classifier dlogits, but replaces the two separate post-GEMM target-correction launches with one combined Tile CUDA correction kernel so dHidden remains correct after its GEMM. The dedicated RTX 5090 CUDA 13.3 3-step, 2-sample same-script gate proved the strategy change (`lm_head_ce_kernel_strategy: no-loss-prob-only-dlogits-plus-combined-target-correction`) but rejected it at `1.006574x` train-loop wall time and `1.003646x` `stage.lm_head_backward.total_ms`. Keep this diagnostic-only; the extra atomic correction work still loses despite fewer launches.
  - 2026-06-24 promoted the combined prob-only target-correction launch shape from 256 to 512 threads for that diagnostic route only. Runtime JSON now reports `lm_head_prob_only_target_correction_threads`, `tools/paired_kernel_speed.py` treats it as hot route evidence, and `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_prob_only_combined_corrections_threads_512` isolates forced 256 versus forced 512 while disabling cooperative LM-head and fused CE. The CUDA 13.3 dedicated RTX 5090 3-step, 2-sample gate passed at `0.988300x` train-loop wall time, `0.997678x` steady-state CUDA-event step time, `0.999215x` LM-head backward, and `0.999998x` LM-head CE. This does not promote the broader prob-only CE route as default; it only improves the diagnostic fallback if selected.
  - 2026-06-25 optimized the diagnostic probability-only BF16/u16 CE+dlogits Tile kernel so its aligned vec8 loop stores with `store_bf16_vec8_normal` instead of eight scalar BF16 stores. The implementation change was validated with the existing same-script `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_prob_only_combined_corrections` gate over 3 steps, 2 samples, and stage timing on the dedicated RTX 5090. The route change was proven and LM-head timing improved slightly (`stage.lm_head_backward.total_ms=0.999740x`, `stage.lm_head_backward.cooperative.total_ms=0.999728x`), but the candidate remains rejected/default-off because train-loop wall regressed to `1.005718x`, tokens/sec fell to `0.994390x`, block backward regressed to `1.012704x`, and MLP projection regressed to `1.004966x`. Do not promote the prob-only correction schedule; the next useful LM-head work is still the real fused/co-scheduled classifier-backward kernel or a materially different GEMM route.
  - 2026-06-25 corrected the prob-only runtime JSON strategy labels after the vec8 normal-store implementation: future prob-only candidate profiles should report `lm_head_ce_bf16_vector_io_strategy: vec8-loads-normal-vec8-stores` and `lm_head_ce_kernel_strategy` strings that include `vec8-loads-normal-vec8-stores` when the diagnostic route executes. Treat older `no-loss-prob-only-dlogits-plus-*` labels as stale telemetry from before this fix.
  - 2026-06-24 tightened the standalone LM-head strict benchmark diagnostics. `tools/bench_lm_head_backward_candidate.sh` and its JSON now report `candidate_cuda_graph_wrapper_only` separately from `candidate_sequence_wrapper_only`, and the strict true-fused gate names the CUDA Graph wrapper failure directly. A CUDA 13.3 dedicated-GPU one-iteration strict rerun reached the current graph route (`graph_capture_success_count: 1`, `graph_replay_success_count: 1`) and still failed with `candidate_true_fused_capability: false`, confirming the next implementation target remains a real fused kernel rather than treating graph replay as fused capability.
  - 2026-06-24 fixed the standalone LM-head microbench reference component timing so logits/CE/dHidden/dWeight probes run the configured warmup count before timed iterations and JSON reports `reference_component_warmup`. The post-fix dedicated RTX 5090 refresh shows the strict symbol is still only a CUDA Graph wrapper (`candidate_true_fused_capability=false`, `candidate_cuda_graph_wrapper_only=true`) and slightly slower than the cooperative baseline at `1.000975x`; the warmed reference logits probe now reports `13.532 ms/iter` instead of first-use setup noise. The next useful work remains a real fused LM-head body rather than graph replay or reference-timing noise.
  - 2026-06-20 added raw Tile C ABI diagnostics for CUDA 13.3 cuBLASLt grouped-layout support. Native GPT JSON now reports `linear_cublaslt_grouped_layout_probe_available`, raw `linear_cublaslt_grouped_layout_probe_status`, and `linear_cublaslt_grouped_layout_supported`, and the paired speed tool extracts the probe status. This does not change the default route; it creates a concrete gate for the next grouped-GEMM or cooperative-kernel candidate so unsupported local grouped layout support cannot be mistaken for a route regression.
  - 2026-06-21 rebuilt the SM120 native artifacts with the current CUDA 13.3 WSL install using `bash tools/rebuild_native_sm120.sh`. `tools/check_native_no_torch_deps.py` reported all native binaries, Python bindings, and no-Torch SDK entrypoints `ok`; the GPU-visible focused CUDA Tile suite passed with `154 passed`; and a one-step native GPT CUDA smoke reported runtime and driver `13.3` with `graph_editor_tensor_flow: false`. The refreshed stage-timed smoke measured `2638.68 ms` train-loop wall, with `lm_head_backward` still at `582.197 ms` (`logits 173.094 ms`, `ce 69.1886 ms`, `dhidden 176.392 ms`, `dweight 161.898 ms`) and `block_backward` at `1259.7 ms`. This keeps the remaining work pointed at native kernel throughput, not stale CUDA artifacts, Torch dependency leakage, or graph-editor tensor flow.
  - 2026-06-21 changed `tools/bench_native_gpt_sm120_parity.sh` back to measurement-only by default after a post-rebuild parity run measured the current gap at `1.027519x` train-loop wall and exited nonzero solely because the strict llm.kittens gate was enabled. Use `NFN_SM120_PARITY_ENFORCE_GATE=1` or explicit `NFN_SM120_PARITY_MAX_CANDIDATE_RATIO` values when a parity run should be a failing performance gate; leave it unset for normal gap measurement. Native-vs-native candidate runs remain strictly gated by default.
  - 2026-06-21 rejected a current-shape LM-head dHidden cuBLASLt probe after the CUDA 13.3 rebuild. `NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1 NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,32768,50304,N,N,0` moved the dHidden route from BF16 GEMMEx to cuBLASLt for the 32768-row chunk shape, but the same-script RTX 5090 stage-timed gate failed at `1.001298x` `stage.lm_head_backward.total_ms` and `1.001356x` `stage.lm_head_backward.dhidden.total_ms`. Keep the 32768-row dHidden shape on the current GEMMEx route.
  - 2026-06-21 corrected the named `lm_head_cublaslt_dhidden_32768` candidate profile so it expands to the same route-changing env used by the rejected probe: `NFN_NATIVE_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE=768,32768,50304,N,N NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K=1 NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,32768,50304,N,N,0`. This is benchmark tooling only; it does not promote the rejected route.
  - 2026-06-22 CUDA 13.3 WSL reinstall recheck on the dedicated RTX 5090: `nvidia-smi` reported CUDA UMD `13.3`, no compute processes, and the native builds/tests were green (`bash tools/build_native_train_tile_ops.sh build/libnfn_native_train_tile_ops.so`, `bash tools/build_native_gpt_cli.sh`, focused native GPT pytest, and `tools/check_native_no_torch_deps.py --skip-artifacts --json`). Rechecked `lm_head_cooperative_backward`; the route enabled `lm_head_cooperative_backward_sequence_wrapper_enabled: true`, but the gate rejected it at `1.004660x` `stage.lm_head_backward.total_ms`. Rechecked `lm_head_dweight_before_dhidden`; it changed `lm_head_dhidden_dweight_schedule_strategy` to `serial-dweight-before-dhidden`, but failed train-loop (`1.011427x`) and block-backward (`1.020745x`) gates. Rechecked `cublaslt_grouped_probe`; grouped layout still succeeds (`linear_cublaslt_grouped_layout_probe_status: 0`) while grouped matmul execution remains blocked (`linear_cublaslt_grouped_matmul_probe_status: 15`). These are still diagnostic-only and not defaults.
  - 2026-06-22 initially promoted the row-loss sum-accumulate tail to the default after that CUDA 13.3 dedicated RTX 5090 gate passed. `NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE=1` measured `0.998849x` train-loop wall, `1.001155x` tokens/sec, `0.999887x` LM-head backward, `0.998400x` block backward, and `0.999498x` MLP projection versus the older partial-reduction tail. The 2026-06-23 post-row-chunk default rerun superseded that result and restored the partial-reduction default: `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_row_loss_partial_reduce` measured `0.997075x` train-loop wall and `1.002951x` tokens/sec versus sum-accumulate. The old route remains available with `NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE=1` for manual diagnostics, but `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_row_loss_sum_accumulate` is now rejected for normal real launches: the CUDA 13.3 dedicated RTX 5090 3-step, 2-sample stage-timed rerun changed the strategy but failed the strict gates at `1.000970x` steady-state CUDA-event wall time and `1.000304x` LM-head backward.
  - 2026-06-23 added a linked dense GPT native CLI path for startup-sensitive workstation runs. `bash tools/build_native_gpt_cli_linked.sh` builds `build/nfn_gpt_native_train_linked` against `build/libnfn_native_train_tile_ops.so`, and `--tile-ops-lib linked` resolves Tile ABI symbols from `RTLD_DEFAULT` instead of `dlopen`ing the library inside the trainer. A dedicated RTX 5090 4-sample startup-only paired benchmark measured linked setup wall at `0.877545x` of the dynamic-loader baseline while preserving the required Tile symbol scan; keep the normal dynamic `--tile-ops-lib PATH` path for same-script kernel candidate library swaps.
  - 2026-06-23 made that linked path the normal workstation default when built: SDK compiled-CLI resolution, `train_gpt.py`, direct `nfn train`, `nfn_native_train`, and the GPT-2-evo delegate now prefer `build/nfn_gpt_native_train_linked` unless an explicit native GPT CLI override is set. The linked binary self-selects `tile_ops_library: "linked"` from its executable name, and `tools/build_native_gpt2_all.sh` plus `tools/rebuild_native_sm120.sh` now build it so default training runs pick up the measured startup win without extra flags. A same-harness dedicated RTX 5090 paired startup check measured this default route at `0.870426x` setup wall versus the dynamic-loader baseline with no native strategy or route-counter changes.
  - 2026-06-24 continuation validation after the CUDA 13.3 refresh: the broad native GPT suite passed with `82 passed, 2 skipped`, the GPT template preset suite passed with `28 passed`, and `tools/check_native_no_torch_deps.py --skip-artifacts --json` passed with the wrapper print-command paths still around `0.04s` and no Torch/NumPy/tiktoken/dataset-manager imports. A short dedicated RTX 5090 llm.kittens parity smoke (`NFN_SM120_PARITY_STEPS=2 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0`) selected GPU 0 with no compute processes and measured llm.kittens at `2636.340 ms/step` / `200236.5 tok/s` versus NeuralFn at `2573.840 ms/step` / `203699 tok/s` (`0.976293x` train-loop wall, `1.017292x` tokens/sec). Treat this as a green smoke only; the next production kernel target remains the fused row-chunked LM-head backward route below.
  - 2026-06-25 post-CUDA-13.3.33 rebuild validation: `nvidia-smi` reported the dedicated RTX 5090 display disabled, CUDA UMD `13.3`, and no compute processes. Rebuilt `build/libnfn_native_train_tile_ops.so`, `build/nfn_gpt_native_train`, and `build/nfn_gpt_native_train_linked`; focused pytest passed for the LM-head strict microbench and SM120 candidate wrapper; `tools/check_native_no_torch_deps.py --skip-artifacts --json` passed, including native `nfn infer` checkpoint detection and prompt-token sampling. A 3-step, 1-sample llm.kittens parity gate still failed at `1.018331x` train-loop wall, `1.011590x` steady-state CUDA-event step time, and `0.978569x` tokens/sec versus llm.kittens. NeuralFn hot stages remain `stage.block_backward.total_ms=3935.67 ms`, `stage.train.model_forward.total_ms=1980.25 ms`, `stage.lm_head_backward.total_ms=1714.22 ms` (`logits=520.46 ms`, `cooperative=1189.04 ms`), and `stage.block_backward.mlp_proj.total_ms=973.564 ms`. This keeps the next target on true fused LM-head backward or materially faster block-backward kernels, not CUDA installation or Torch leakage.
  - 2026-06-25 post-reinstall graph-prewarm recheck: the full native GPT suite passed with `90 passed, 2 skipped`, confirming the CUDA 13.3 reinstall did not leave native trainer tests failing. A later dedicated RTX 5090 same-script native-vs-native rerun promoted graph prewarm by eliminating runtime LM-head graph captures and passing all configured gates: train-loop wall `0.970282x`, steady-state CUDA-event timing `1.001894x`, LM-head backward `0.968319x`, block backward `0.956792x`, and MLP projection backward `0.911989x`. This is now a default graph-wrapper improvement, not the final strict fused-kernel solution.
  - 2026-06-26 rechecked `lm_head_graph_prewarm` after the MLP FC ordering rollback. The 3-step, 2-sample same-script gate still passed with route proof: graph capture attempts moved `3 -> 0`, graph cache hits moved `45 -> 48`, train-loop wall was `0.985915x`, steady-state CUDA-event timing was `0.999199x`, LM-head backward was `0.957549x`, block backward was `0.997858x`, and MLP projection backward was `0.992403x` versus explicit prewarm opt-out. Keep graph prewarm as the default wrapper path while the strict true-fused LM-head kernel remains the next implementation target.
  - 2026-06-25 current measurement-only llm.kittens parity refresh with selected-GPU locking and stage timing measured llm.kittens at `2429.860 ms/step` and NeuralFn at `2525.015 ms/step`, or `1.039161x` host wall. Steady-state CUDA-event timing was closer but still behind at `1.012150x` (`2441.650 ms/step` NeuralFn versus `2412.340 ms/step` llm.kittens), and tokens/sec was `0.962262x` of reference. The same run reported `lm_head_classifier_backward_path_class=diagnostic-cuda-graph-wrapper`, `lm_head_cooperative_backward_fused_kernel_capability_available=false`, and three graph-body nodes per replay, so strict parity still requires replacing the graph wrapper with a true fused LM-head classifier-backward Tile kernel.
  - Next implementation target: replace the cached CUDA Graph LM-head strict ABI body with a NeuralFn-owned fused row-chunked LM-head backward kernel that keeps the current bounded resident-logit cap but avoids paying separate CE, dHidden, and dWeight kernel bodies per chunk. The correctness surface is now green under CUDA 13.3, and graph prewarm now removes lazy capture from the timed path, but the strict ABI still reports `diagnostic-cuda-graph-wrapper` rather than `strict-true-fused-tile-kernel`.
- [x] Extend native GPT checkpoint text-prompt inference with GPT-2 tokenization in the lightweight wrapper so `.bin` checkpoint inference no longer needs the transitional external sampler bridge.
- [x] Add GPT-2 evo raw Tile-CUDA trainer ABI for device-side candidate mutation, best-loss selection, and best-candidate adoption without graph-editor tensor flow.
- [x] Add GPT-2 evo compiled C++ `--smoke-evo-kernels` path that loads the raw evo ABI, launches mutate/select/adopt on CUDA device buffers, and verifies best-candidate copyback without Python/Torch, datasets, or graph-editor payloads.
- [x] Wire the dense GPT native `--train-transformer-lm` loop to the raw layer-evo mutate/select/adopt ABI cadence behind `--layer-evo`, targeting the selected block's float32 `ln1.weight` on device and reporting `graph_editor_tensor_flow: false` in plan/runtime JSON.
- [x] Make `nfn_gpt2_evo_native_train` delegate dense GPT-2-compatible training runs to `nfn_gpt_native_train --train-transformer-lm --layer-evo`, preserving print-plan/smoke behavior and keeping the runtime JSON explicit about the native evo candidate-loss source.
- [x] Wire GPT-2 evo native layer-evolution forward-only candidate evaluation into the dense GPT trainer loop: after AdamW, the trainer mutates `block_N.ln1.weight`, reuses the current pinned training batch, allocates the same lazy float MLP scratch used by validation-only forwards, scores every candidate through native CUDA forward loss, copies each scalar loss directly from the native loss device buffer into the device candidate-loss array, selects/adopts via the raw evo ABI, and reports `candidate_loss_source: "native-forward-loss-device-resident-current-batch"`, `candidate_loss_transport: "device-to-device"`, `candidate_loss_host_roundtrips_elided`, and `forward_candidate_evals` without graph-editor tensor flow.
- [x] Expose NanoGPT preflight JSON with separate `available_native_kernels` and `required_native_kernels` lists.
- [x] Add NanoGPT `--check-tile-ops` compiled C++ path that `dlopen`s `libnfn_native_train_tile_ops.so` and verifies required raw ABI symbols without Python/Torch.
- [x] Add NanoGPT `--smoke-tile-ops` compiled C++ path that dynamically loads CUDA runtime, executes `nfn_native_tile_fill_float32`, and verifies device-to-host copyback without Python/Torch.
- [x] Add NanoGPT `--smoke-optimizer-step` compiled C++ path that builds the NanoGPT parameter layout, initializes contiguous param/grad/AdamW buffers, executes `nfn_native_tile_adamw_step_float32` once per registered parameter buffer, and verifies param/moment copyback without Python/Torch.
- [x] Add NanoGPT `--smoke-training-loop-step` compiled C++ path that executes gradient zeroing, synthetic gradient fill, global-norm clip scale finalization, device-scalar gradient scaling, and per-buffer AdamW over the registered parameter layout without Python/Torch.
- [x] Add NanoGPT `--smoke-lm-step` compiled C++ path that runs token embedding, tied LM-head linear logits, token CE loss/backward, tied weight backward, and AdamW update through raw native kernels without Python/Torch.
- [x] Add NanoGPT `--smoke-token-train-step` compiled C++ path that samples a real native uint16 token/target batch from cached shards, runs tied-LM forward/backward/update kernels over those IDs, and verifies sampled-batch loss, gradient, and weight update values without Python/Torch.
- [x] Add NanoGPT `--train-token-lm` compiled C++ path that runs a real multi-step tied token-embedding LM training loop over cached native token shards without Python/Torch.
- [x] Add periodic native validation loss to NanoGPT `--train-token-lm` over resolved validation token shards without Torch, Python dataset payloads, or graph-editor node data flow.
- [x] Route NanoGPT `--train-token-lm` through `nfn-native-train` and `neuralfn.native_train.run_native_train()` so CLI and SDK dispatch stay on compiled native artifacts.
- [x] Make normal NanoGPT training entrypoints (`nfn train --base-model nanogpt ...` and `python cli/scripts/train_nanogpt.py ...`) stay inside native C++ before Torch imports. Full-transformer NanoGPT now selects `--train-transformer-lm --template-name nanogpt` and routes through the shared dense GPT loop with NanoGPT width/head/layer geometry, while explicit `--train-token-lm` still reaches the older tied token-embedding native loop for diagnostics.
- [x] Add NanoGPT `--smoke-embedding-norm-step` compiled C++ path that samples native tokens and runs token/position embeddings, residual add, LayerNorm forward/backward, tied logits, CE backward, embedding/position/norm gradients, and AdamW updates through raw native kernels without Python/Torch.
- [x] Add NanoGPT `--smoke-mlp-step` compiled C++ path that runs MLP fc projection, GELU, output projection, backward, and AdamW updates through raw native kernels without Python/Torch.
- [x] Add NanoGPT `--smoke-attention-step` compiled C++ path that runs Q/K/V projections, SDPA forward/backward, output projection, Q/K/V projection backward, and AdamW updates through raw native kernels without Python/Torch.
- [x] Add NanoGPT native parameter/gradient buffer registry and contiguous AdamW-state layout to the C++ preflight.
- [x] Add NanoGPT AdamW parameter-group metadata over the registered C++ buffers.
- [x] Add NanoGPT native execution-stage plan with ready/requires-wiring/missing-ABI status per forward, backward, and optimizer stage.
- [x] Expose scaled residual add through the raw no-Torch native trainer ABI.
- [x] Expose fused QKV split/merge through the raw no-Torch native trainer ABI so NanoGPT can use one `qkv.weight` projection, feed contiguous Q/K/V buffers into SDPA, and pack Q/K/V gradients back into the fused projection gradient.
- [x] Expose reshape-heads and merge-heads through the raw no-Torch native trainer ABI so GPT-style trainers can feed `[batch, heads, seq, head_dim]` attention kernels without PyTorch layout helpers.
- [x] Add NanoGPT `--smoke-qkv-layout-step` compiled C++ path that executes fused-QKV split and merge kernels on device buffers and verifies exact layout copyback without Python/Torch.
- [x] Add NanoGPT `--smoke-fused-qkv-attention-step` compiled C++ path that runs fused `attn.qkv.weight` projection, QKV split, SDPA forward/backward, QKV gradient merge, fused qkv weight backward, output projection backward, and AdamW updates through raw native kernels without Python/Torch.
- [x] Add NanoGPT `--smoke-transformer-block-step` compiled C++ path that composes LayerNorm, fused-QKV attention, residual adds, MLP, backward passes, gradient accumulation, and AdamW updates for one tiny transformer block through raw native kernels without Python/Torch.
- [x] Mark NanoGPT tied LM head input/weight backward as covered by the raw linear backward native ABI.
- [x] Replace the NanoGPT-sized CE backward path (`vocab <= 1024`) with row-wise kernels that compute softmax statistics once per row instead of once per output element.
- [x] Add chunked row-wise CE backward for larger vocabularies so full GPT-class vocabularies do not use the elementwise fallback.
- [x] Expose absolute position embedding backward through the native ABI.
- [x] Expose token embedding weight backward through the native ABI.
- [x] Expose LayerNorm input and affine parameter backward through the native ABI.
- [x] Expose RMSNorm input backward through the native ABI.
- [x] Expose linear input backward through the native ABI.
- [x] Expose linear weight and bias backward through the native ABI.
- [x] Expose LayerNorm affine and Linear bias gradient-accumulate native ABI variants for optimizer-step accumulation buffers.
- [x] Replace serial linear weight/bias backward row loops with row-chunked tiled atomic accumulation for large row counts.
- [x] Remove GPT-2 full-trainer per-microbatch LayerNorm affine / Linear bias scratch buffers and copy loops by writing directly into accumulation buffers.
- [x] Add trainer-build GPU GEMM fast path for native linear forward, dInput, and dWeight behind `NFN_TILE_CUDA_USE_CUBLAS_LINEAR=1` without importing Torch or the PyTorch Tile extension.
- [x] Add trainer-build GPU GEMV fast path for native linear bias backward and accumulate-bias backward behind `NFN_TILE_CUDA_USE_CUBLAS_LINEAR=1`, using a cached device ones vector initialized by a Tile fill kernel instead of the row-chunked atomic bias fallback.
- [x] Replace fallback linear weight backward reduction kernels with GEMM-grade tiled kernels for large row counts when the trainer cuBLAS path is unavailable. The fallback float32-output dWeight paths now use a shared-memory 2D tiled CUDA kernel for float32/BF16 activation and gradient combinations, while the normal trainer build still tries cuBLAS/cuBLASLt first and bias-only fallbacks keep the shared row-chunk reduction path.
- [x] Expose GELU activation forward/backward through the native ABI.
- [x] Add dropout forward/backward native Tile ABI for nonzero NanoGPT dropout preflight coverage, using deterministic inverted-dropout Tile kernels exposed as `nfn_native_tile_dropout_forward_float32` and `nfn_native_tile_dropout_backward_float32`.
- [x] Wire MLP-stage activation and projection backward through the native NanoGPT preflight/smoke path, including fc/proj input and weight backward, GELU backward, and AdamW updates.
- [x] Expose scaled dot-product attention backward through the native ABI.
- [x] Wire attention-stage QKV/output projection backward through the native NanoGPT preflight/smoke path, including fused QKV split, SDPA backward, QKV gradient merge, fused qkv backward, output projection backward, and AdamW updates.
- [x] Replace GPT-2-compatible SDPA forward scalar-output Tile launch with a value-chunked row-vector Tile attempt for `seq_k <= 1024`, reusing each query row's score/softmax across a 2-channel value chunk when CUDA accepts the row kernel and falling back to scalar Tile attention when CUDA rejects that launch.
- [x] Add GPT-2 `--train-transformer-lm` attention-forward launch telemetry and auto-disable repeated row-kernel attempts after the first CUDA launch rejection, so live runs report row attempts, row successes, row fallbacks, scalar launches, and avoid repeated failed-launch overhead.
- [x] Tune GPT-compatible SDPA forward resources so the live SM120 trainer no longer needs row-vector or scalar-launch safety fallbacks; the current packed-QKV TK path reports `attention_forward_row_launch_fallback_count: 0`, `attention_forward_scalar_launch_count: 0`, and `attention_forward_tk_launch_count` for the full loop.
- [x] Keep the older GPT-2-compatible row-vector SDPA kernel out of the hot trainer path when packed SM120 TK attention is active, so the scalar fallback remains unused on the live dedicated-RTX-5090 probe.
- [x] Cover every shipped GPT template name in the native GPT-2 training selector via `--template-name` / `--preset`, and cover custom graph selection via `--graph-file`, returning explicit native-trainer-missing JSON for unsupported templates instead of falling back to Torch or graph-editor tensor flow.
- [x] Admit compatible dense GPT custom graph files into the native trainer when their JSON carries GPT `template_spec` metadata matching the compiled dense loop, while arbitrary graph JSON still returns `custom-graph-native-trainer-missing` and missing paths still return `custom-graph-file-missing`.
- [x] Classify dense GPT modern aliases in the native selector: `gpt2_modern` now reaches the compiled GPT-2 geometry trainer instead of generic missing-template status, while `nanogpt`, `nanogpt_modern`, and `nanogpt_megakernel` are recognized as dense GPT templates and use the selected NanoGPT geometry in the shared native loop.
- [x] Update dense GPT-2 native dry-run/plan JSON to report `native-transformer-lm-ready` and `training_step_plan.status: "ready"` for the implemented compiled Tile-CUDA loop; `remaining_validation` now tracks closing the measured SM120 throughput gap via `tools/bench_native_gpt_sm120_parity.sh` instead of saying live validation has not happened.
- [x] Generalize the shared dense GPT transformer-LM loop so dense GPT dimensions can be driven from the selected template or compatible graph metadata instead of fixed GPT-2 trainer shape defaults. Plan/runtime JSON now exposes `native_geometry_contract.name: "native-dense-gpt-transformer"`, `shape_source: "selected_dense_gpt_geometry"` or `"custom_graph_template_spec"`, `template_geometry_dynamic`, `custom_graph_geometry_dynamic`, `selected_template_geometry`, `geometry_matches_compiled_loop`, and custom graph path metadata (`graph_file_exists`, `graph_file_size_bytes`) so selected-template shape mismatches, arbitrary custom graphs, and missing graph files are not mistaken for missing native trainers.
  - 2026-06-20 unblocked the Python/SDK compiled-CLI handoff from masking compatible custom-graph geometry: `NativeGpt2RunConfig` / `NativeGptRunConfig` now carry `batch_size_explicit`, `seq_len_explicit`, and `num_layers_explicit`, and the high-level native GPT harness sets them from actual argv. When false, the compiled argv omits those flags so the C++ runner can consume `template_spec` `seq_len` / `num_layers` from `--graph-file` and adjust the batch shape without Torch or graph-editor tensor flow.
  - 2026-06-22 extended the compiled dense loop to use selected dense GPT model width, head count, head dim, MLP size, layer count, context length, vocabulary rows, dry-run/plan allocation metadata, and checkpoint metadata. NanoGPT now reports `native-transformer-lm` and uses its 320-wide/5-head/5-layer geometry. Template dropout is recorded in geometry metadata; stochastic dropout inside the full dense transformer loop remains a separate native-kernel integration item.

## Backend scaffolding

- [x] Add `neuralfn/tile_cuda/__init__.py`.
- [x] Add `neuralfn/tile_cuda/config.py` with `TileCudaConfig`.
- [x] Add `neuralfn/tile_cuda/registry.py` with `TileKernelSpec`, `TileKernelRegistry`, and `KernelCoverageReport`.
- [x] Add `neuralfn/tile_cuda/runtime.py` with availability checks, extension loading, and fallback policy.
- [x] Add `neuralfn/tile_cuda/autograd.py` with PyTorch autograd wrappers for Tile kernels.
- [x] Add `neuralfn/csrc/tile_cuda/` for C++/CUDA Tile extension sources.
- [x] Add `neuralfn/csrc/tile_cuda/bindings.cpp` for PyTorch custom op bindings.
- [x] Add `neuralfn/csrc/tile_cuda/kernels.cu` or split files by family once the file becomes too large.
- [x] Add packaging support for optional CUDA extension builds without breaking CPU-only installs.
- [x] Add `NFN_TILE_CUDA_BUILD=1` opt-in source build path.
- [x] Add `NFN_TILE_CUDA_ARCH` override for explicit `-arch`.
- [x] Add runtime diagnostics for CUDA version, driver version, GPU CC, Tile header availability, and extension load status.

## SDK and CLI surface

- [x] Add `TorchTrainConfig.kernel_backend: Literal["auto", "torch", "tile_cuda"] = "auto"`.
- [x] Add `TorchTrainConfig.tile_cuda_strict: bool = False`.
- [x] Add `TorchTrainConfig.tile_cuda_report_path: str | None = None`.
- [x] Let `CompiledTorchGraph` choose Tile-backed stages when requested and available.
- [x] Preserve existing PyTorch stage behavior when `kernel_backend="torch"` or Tile is unavailable in auto mode.
- [x] Add `nfn train --kernel-backend {auto,torch,tile-cuda}`.
- [x] Add `nfn infer --kernel-backend {auto,torch,tile-cuda}` where inference kernels exist.
- [x] Add `nfn eval --kernel-backend {auto,torch,tile-cuda}`.
- [x] Add `--tile-cuda-strict`.
- [x] Add `--tile-cuda-report PATH`.
- [x] Add `nfn kernels list`.
- [x] Add `nfn kernels doctor`.
- [x] Add `nfn kernels bench`.
- [x] Add `nfn kernels examples`.

## Coverage gates

- [x] Generate the coverage inventory from `BuiltinNeurons.all()`.
- [x] Generate the module inventory from `build_module()` dispatch.
- [x] Generate the scalar function inventory from `build_function_module()`.
- [x] Fail coverage tests if a builtin/module/function is missing from the Tile registry.
- [x] Fail strict training if a selected graph has an uncovered node.
- [x] Produce a JSON report with kernel status, fallback reason, dtype support, and tested shapes.
- [x] Include host-only entries for source/reference/orchestration nodes.
- [x] Include delegated entries for fused kernels that cover multiple logical graph nodes.

Per-kernel done criteria:

- [x] Forward CUDA Tile kernel or delegated fused implementation.
- [x] Backward CUDA Tile kernel, autograd composition, or explicit no-grad reason.
- [x] PyTorch custom op binding.
- [x] Autograd wrapper.
- [x] Shape contract.
- [x] Dtype contract.
- [x] CPU/PyTorch parity test.
- [x] CUDA parity test.
- [x] Gradient parity test for trainable kernels.
- [x] SDK example.
- [x] CLI smoke coverage where relevant.
- [x] Docs entry.

## Scalar function kernels

- [x] `input` / `input_node`: host-only interface marker; no compute kernel.
- [x] `output` / `output_node`: host-only interface marker; no compute kernel.
- [x] `identity`: elementwise forward/backward.
- [x] `negate`: elementwise forward/backward.
- [x] `add`: elementwise binary forward/backward.
- [x] `multiply`: elementwise binary forward/backward.
- [x] `sigmoid`: elementwise forward/backward.
- [x] `relu`: elementwise forward/backward.
- [x] `tanh_neuron`: elementwise forward/backward.
- [x] `threshold`: no-grad bool-style output, explicit training limitation.
- [x] `gaussian`: elementwise `exp(-x*x)` forward/backward.
- [x] `log`: clamped log forward/backward matching `max(x, 1e-7)`.
- [x] `leaky_relu`: elementwise forward/backward.
- [x] `prelu`: constant-slope elementwise forward/backward.
- [x] `relu6`: clipped ReLU forward/backward.
- [x] `elu`: elementwise forward/backward.
- [x] `selu`: elementwise forward/backward.
- [x] `gelu`: scalar builtin GELU forward/backward.
- [x] `silu`: elementwise forward/backward.
- [x] `mish`: elementwise forward/backward.
- [x] `softplus`: stable forward/backward.
- [x] `softsign`: elementwise forward/backward.
- [x] `hard_sigmoid`: clipped linear forward/backward.
- [x] `hard_tanh`: clipped linear forward/backward.
- [x] `hard_swish`: elementwise forward/backward.
- [x] `softmax_2`: two-input stable softmax forward/backward.
- [x] `logsoftmax_2`: two-input stable log-softmax forward/backward.

## Core tensor and LLM kernels

- [x] `token_embedding`: gather forward, indexed gradient accumulation.
- [x] `linear`: matmul plus optional bias, backward for input/weight/bias.
- [x] `tied_lm_head`: linear against shared embedding weight.
- [x] `lm_head`: vocab projection.
- [x] `logit_softcap`: `softcap * tanh(logits / softcap)`.
- [x] `token_cross_entropy`: numerically stable CE reduction.
- [x] `masked_token_cross_entropy`: CE masked by response/loss mask.
- [x] `sequence_logp`: gather log-prob sums with mask.
- [x] `residual_add`: scaled residual add.
- [x] `residual_mix`: learned primary/skip scale mix.
- [x] `manifold_hyper_connection`: sigmoid beta, bounded residual mix.

## Norm, activation, and MLP kernels

- [x] `rms_norm`: RMS norm forward/backward over last dimension.
- [x] `layer_norm`: LayerNorm forward/backward.
- [x] `group_norm`: grouped norm for `[B, S, D]`.
- [x] `qk_norm`: fused Q/K RMSNorm.
- [x] `dyt`: dynamic tanh with learnable alpha, weight, bias.
- [x] `dropout`: deterministic `p=0` and inference passthrough; stochastic training mask remains on the PyTorch RNG path.
- [x] `gelu` module: tensor GELU forward/backward.
- [x] `mlp_relu2`: linear, ReLU, square, projection.
- [x] `swiglu`: three-matrix SiLU gate.
- [x] `geglu`: three-matrix GELU gate.
- [x] `reglu`: three-matrix ReLU gate.
- [x] `solu`: softmax-gated linear unit.

## Attention and position kernels

- [x] `reshape_heads`: view/transpose contract with contiguous fallback.
- [x] `merge_heads`: transpose/reshape merge.
- [x] `repeat_kv`: grouped-query KV repeat.
- [x] `rotary_embedding`: RoPE forward/backward for Q/K.
- [x] `qk_gain`: per-head Q scale.
- [x] `scaled_dot_product_attention`: causal/non-causal SDPA.
- [x] `sliding_window_attention`: local causal window SDPA.
- [x] `block_sparse_attention`: block-local plus sink/global pattern.
- [x] `streaming_attention_sinks`: recent window plus persistent sinks.
- [x] `native_sparse_attention`: deterministic NSA reference pattern first, learned sparse selector later.
- [x] `differential_attention`: dual SDPA branches plus lambda subtraction and norm.
- [x] `causal_self_attention`: fused QKV projection, QK norm, RoPE, QK gain, SDPA, output projection.
- [x] `fused_causal_attention`: fused QKV, RoPE, SDPA, output projection.
- [x] `multi_latent_attention`: MLA low-rank KV, decoupled RoPE, SDPA, output projection.
- [x] `absolute_position_embedding`: learned positional lookup.
- [x] `routed_attention_experts`: expert-routed attention path.

## KV cache and compression kernels

- [x] `kv_cache_read`: concat cache and current KV or passthrough.
- [x] `kv_cache_write`: cache output contract, no-grad/pass-through where applicable.
- [x] `kv_pca_encode`: K/V projection to compressed dim.
- [x] `kv_pca_decode`: K/V projection back to head dim.
- [x] `kv_quant_pack`: int8 pack with per-token scale.
- [x] `kv_quant_unpack`: dequantize and split packed K/V.

## Quantization and adapter kernels

- [x] `bitlinear_ternary`: ternary weight quantization and quantized activation STE.
- [x] `fp8_linear`: E4M3/E5M2 quantized weight path, amax history, STE.
- [x] `mx_linear`: MXFP4/MXFP8 block-scale quantized weight path.
- [x] `lora_linear`: base linear plus low-rank delta.
- [x] `nf4_linear`: NF4 unpack/dequant plus LoRA delta.
- [x] `randmap_adapter`: frozen down/up maps plus trainable middle and scale.
- [x] `ttt_linear`: test-time training linear stage.

## MoE and routing kernels

- [x] `router_logits`: router projection.
- [x] `auxfree_load_balancing`: biased router update without host sync.
- [x] `topk_route`: softmax, top-k, normalize, routing stats.
- [x] `expert_dispatch`: token-to-expert dispatch and weighted combine.
- [x] `expert_combine`: identity/combine contract.
- [x] `broadcast_expert_routes`: route expansion over sequence.
- [x] `broadcast_chunk_routes`: route expansion over chunk spans.
- [x] `load_balance_loss`: router density auxiliary loss.
- [x] `aux_loss_add`: scalar loss addition.
- [x] `loss_scale`: scalar loss scaling.
- [x] `route_balance_loss`: route entropy/balance objective.
- [x] `route_selection_loss`: supervised route objective.
- [x] `route_distillation_loss`: route distillation objective.

## Semantic kernels

- [x] `semantic_data_source`: host-only source contract.
- [x] `semantic_projector`: semantic/residual/topic projection.
- [x] `semantic_alignment_loss`: semantic topic CE.
- [x] `semantic_hasher`: LSH bucket hashing.
- [x] `semantic_moe_router`: semantic top-k router.
- [x] `semantic_hash_router`: hash/topic/target-aware router.
- [x] `causal_chunk_state`: causal chunk pooling/state.
- [x] `semantic_chunk_projector`: chunk semantic/residual/topic projection.
- [x] `semantic_chunk_hasher`: chunk LSH bucket hashing.
- [x] `semantic_moe_jepa_evo_router`: shared/semantic/free expert router.
- [x] `attentionless_decoder`: hash/expert output to logits.
- [x] `softmax_distillation_loss`: teacher/student distillation loss.

## JEPA, diffusion, byte, universal, and sequence kernels

- [x] `mamba`: projection, depthwise conv, SiLU gate, output projection.
- [x] `denoise_head`: diffusion denoise projection.
- [x] `mask_scheduler`: timestep-driven token masking.
- [x] `random_timesteps`: device-side random timestep generation.
- [x] `jepa_mask`: random and block mask generation.
- [x] `latent_pool`: masked mean with fallback mean.
- [x] `jepa_projector`: projector MLP.
- [x] `jepa_predictor`: predictor MLP.
- [x] `latent_mse_loss`: detached-target MSE.
- [x] `byte_patch_embed`: byte embedding plus Conv1d patch projection.
- [x] `byte_patch_merge`: nearest interpolation back to target length.
- [x] `act_halt_gate`: mean-pool halt probability.
- [x] `act_weighted_sum`: weighted sum across recurrent states.
- [x] `universal_transformer`: recurrent attention/MLP with ACT weights.

## Fine-tuning and RLHF kernels

- [x] `sft_dataset_source`: host-only source contract.
- [x] `dpo_dataset_source`: host-only source contract.
- [x] `ppo_rollout_source`: host-only source contract.
- [x] `reference_forward`: delegated compiled graph call, no standalone Tile kernel.
- [x] `reward_forward`: delegated compiled graph call, no standalone Tile kernel.
- [x] `dpo_pairwise_loss`: sigmoid, hinge, IPO variants.
- [x] `reward_head`: pooled scalar head.
- [x] `preference_bce_loss`: Bradley-Terry BCE.
- [x] `value_head`: per-token value projection.
- [x] `ppo_clipped_loss`: clipped policy/value loss.
- [x] `kl_penalty`: reward shaping by KL.
- [x] `gae_compute`: reverse-time GAE scan.

## Optimizer and training runtime kernels

- [x] `Muon._zeropower_via_newtonschulz5`: Tile implementation for matrix updates.
- [x] `Muon.step`: fused momentum and Newton-Schulz update where practical.
- [x] AdamW update kernels for normal optimizer profile.
- [x] Split optimizer profile kernels for embedding/head/matrix/scalar parameter groups.
- [x] Gradient accumulation add-into-buffer kernels.
- [x] Gradient clipping norm and scale kernels.
- [x] EMA target update kernels for JEPA objectives.
- [x] Route-evolution evaluation path audit for Tile compatibility or explicit fallback.
- [x] GPT-2 compiled C++ `--smoke-tile-ops` path that loads raw Tile ops, launches `nfn_native_tile_fill_float32`, copies back, and reports JSON without Python/Torch.
- [x] GPT-2 compiled C++ `--smoke-optimizer-step` path over the registered GPT-2 parameter layout.
- [x] GPT-2 compiled C++ `--smoke-lm-step` path over a tiny tied embedding/LM-head forward/backward/update slice.
- [x] GPT-2 compiled C++ `--smoke-embedding-lm-step` path over sampled cached uint16 tokens, token/position embeddings, final norm, tied LM head, CE backward, embedding/norm backward, and AdamW.
- [x] GPT-2 compiled C++ `--train-embedding-lm` partial native loop over cached shards with periodic validation losses.
- [x] GPT-2 compiled C++ `--smoke-attention-step` path over a tiny model-dim qkv/SDPA/projection backward/update slice.
- [x] GPT-2 compiled C++ `--smoke-mlp-step` path over a tiny c_fc/GELU/c_proj backward/update slice.
- [x] GPT-2 compiled C++ `--smoke-norm-residual-step` path over LayerNorm/residual/backward/gradient-accumulation/update block glue kernels.
- [x] GPT-2 compiled C++ `--smoke-transformer-block-step` path over a composed LayerNorm/attention/residual/MLP/backward/update block, including projection bias gradients and AdamW updates for all 12 GPT-2 block parameter buffers.
- [x] GPT-2 compiled C++ `--smoke-transformer-lm-step` path over sampled cached GPT-2 token IDs, embeddings, one transformer block, final norm, tied LM head, CE backward, transformer backward, embedding backward, and 16-buffer AdamW update coverage.
- [x] GPT-2 compiled C++ `--train-transformer-lm` full-vocab real-dim 12-layer multi-step loop over cached train/validation shards with token-to-loss transformer kernels, row-chunked logits workspace, scratch-recompute activation tape, 148-buffer AdamW updates, validation JSON, and no Python/Torch fallback.

## Examples to add

- [x] `examples/tile_cuda/scalar_add_train.py`
- [x] `examples/tile_cuda/dense_llm_smoke_train.py`
- [x] `examples/tile_cuda/moe_router_smoke_train.py`
- [x] `examples/tile_cuda/jepa_smoke_train.py`
- [x] `examples/tile_cuda/strict_mode_report.py`
- [x] `examples/tile_cuda/kernel_bench.py`
- [x] Generated one-file SDK example for every registry entry under `examples/tile_cuda/generated/`.

## Documentation updates

- [x] Update `README.md` with CUDA Tile setup, backend selection, and fallback behavior.
- [x] Update `CHANGELOG.md` for every meaningful backend, API, or CLI change.
- [x] Add `docs/python-sdk/tile-cuda.md`.
- [x] Update `docs/python-sdk/torch-backend.md`.
- [x] Update `docs/framework-guide/training-workflows.md`.
- [x] Update `docs/cli.md`.
- [x] Update relevant `.cursor/skills/` entries if public SDK, CLI, or MCP behavior changes.

## Tests and verification

- [x] `python -m pytest tests/test_tile_cuda_coverage.py -q`
- [x] `python -m pytest tests/test_tile_cuda_static_plan.py -q`
- [x] `python -m pytest tests/test_tile_cuda_registry.py -q`
- [x] `python -m pytest tests/test_tile_cuda_examples.py -q`
- [x] `NFN_TILE_CUDA_TEST=1 python -m pytest tests/test_tile_cuda_gpu.py -q`
- [x] `python -m pytest tests/test_template_presets.py -x -q`
- [x] `python -m pytest tests/test_builtin_neurons.py -q`
- [x] `python -m pytest tests/test_backend_capabilities.py -q`
- [x] `python -m pytest tests/test_torch_gpt.py -q`
- [x] `python -m pytest cli/tests/test_nfn_cli.py -q`
- [x] `git diff --check`

## Migration notes

- [x] Do not break existing graph JSON: Tile backend selection must live in runtime config, not graph schema, unless a later migration explicitly requires it.
- [x] Do not remove the PyTorch path.
- [x] Do not change existing template preset names for Tile support.
- [x] Do not turn variant-library port mismatch fallback back into a hard error.
- [x] If a future public config or graph serialization field changes, add a clearly labeled breaking-change note to `CHANGELOG.md` and matching docs.

## Dtype expansion: fp16, fp8, and NVFP4

Goal: add fp16, fp8, and NVFP4 CUDA Tile variants for every covered kernel where the dtype is meaningful and safe. Keep explicit exclusions for host-only nodes, integer-output/hash kernels, source/orchestration nodes, and kernels whose state contract is inherently another quantization format such as NF4.

### Dtype policy gates

- [x] Add a per-kernel dtype support matrix to the registry instead of only a flat `dtypes` tuple.
- [x] Add strict-mode errors that name the requested dtype and the supported dtype set for scalar functions, simple modules, and projection modules.
- [x] Add dtype-specific coverage reports for `float32`, `float16`, `float8_e4m3fn`, `float8_e5m2`, and NVFP4.
- [x] Keep fp8/NVFP4 accumulation in fp32 unless the kernel has a proven lower-precision accumulation contract.
- [x] Add deterministic CPU reference quantize/dequantize helpers for fp8 and NVFP4.
- [x] Add GPU parity tolerances per dtype family for verified fp16, fp8, and projection-family NVFP4 contracts.

### fp16 coverage

- [x] Scalar function kernels: unary, binary, and binary-pair function nodes use Tile float32 compute with fp16 cast-in/cast-out.
- [x] Scalar module kernels: `loss_scale`, `logit_softcap`, `aux_loss_add`, and `kl_penalty` use Tile float32 compute with fp16 activation cast-in/cast-out.
- [x] Simple vector/elementwise module kernels: `residual_add`, `residual_mix`, `manifold_hyper_connection`, `qk_gain`, and `dyt` support fp16 activations with float32 scale/parameter gradients.
- [x] Reduction-adjacent elementwise module kernels: `act_weighted_sum` and `latent_pool` support fp16 activations with float32 weights or masks.
- [x] Stateful stochastic elementwise module kernels: `dropout` fp16 training masks for `0 < p < 1`; deterministic eval and `p=0` already use the fp16 identity path.
- [x] Norm kernels: `rms_norm`, `layer_norm`, `group_norm`, and `qk_norm` with fp32 reduction.
- [x] Projection kernels: `linear`, `lm_head`, `tied_lm_head`, router/value/reward/denoise heads, KV PCA projections, JEPA heads, deterministic LoRA/TTT/adapter projections, quantized-weight `bitlinear_ternary`/`fp8_linear`/`mx_linear`, MLP projections, and ACT halt projection.
- [x] Semantic projector fp16 discrete topic output contract: keep `semantic_projector` and `semantic_chunk_projector` float32-only because their argmax-derived topic/signature semantics can change under lower-precision activation quantization.
- [x] Attention kernels: SDPA, sliding/window/block/native sparse variants, differential attention, causal/fused causal attention, MLA, and routed attention experts with fp32 score/softmax or route-weight accumulation and fp16 output.
- [x] Routed attention experts fp16 route-weight accumulation contract.
- [x] Loss/reduction kernels with fp32 accumulation and fp16-compatible logits or values: token CE, masked CE, sequence logp, latent MSE, semantic alignment, DPO, PPO, GAE, preference BCE, load/route balance, route selection/distillation, and softmax distillation.
- [x] Optimizer/runtime kernels where fp16 parameter state is meaningful: `ema_update`, `gradient_accumulate`, `gradient_clip_norm`, and `adamw_step` with fp16 parameter/gradient buffers plus fp32 Adam moments.
- [x] Muon and split-optimizer fp16 matrix-state semantics: support fp16 parameter/gradient tensors with float32 momentum/Adam state; keep standalone Newton-Schulz matrix orthogonalization float32-only.
- [x] CPU registry and GPU parity tests for the fp16-supported scalar function, elementwise module, reduction-adjacent module, norm, projection, attention, loss/reduction, and optimizer/runtime families.
- [x] CPU and GPU parity tests for each newly added fp16 family beyond scalar functions and simple modules.

### fp8 coverage

- [x] Define supported fp8 formats: `float8_e4m3fn` and `float8_e5m2`.
- [x] Direct projection fp8 activation kernels with fp32 accumulation for `linear`, LM/router/value/reward/denoise heads, tied LM head, and KV PCA encode/decode.
- [x] Composite projection-family fp8 activation kernels with fp32 accumulation and scale/amax handling for JEPA heads, LoRA/TTT/adapters, quantized-weight wrappers, MLP projections, and ACT halt projection.
- [x] Attention fp8 Q/K/V input support with fp32 score/softmax accumulation where tensor-core or Tile support allows it.
- [x] Elementwise fp8 pass-through/activation kernels where inputs can be safely dequantized to fp32 and requantized.
- [x] Explicit no-fp8 reasons for losses, optimizers, integer/hash outputs, stochastic masks, and source/delegated nodes where fp8 is not meaningful.
- [x] CPU and GPU parity tests with fp8 tolerances plus explicit boundary overflow checks for PyTorch fp8 E4M3FN/E5M2 reference behavior.

### NVFP4 coverage

- [x] Define the NeuralFn NVFP4 packed representation, scale metadata, and row/block granularity.
- [x] Add pack/unpack helpers and CPU references for NVFP4.
- [x] Add native C++ Tile ops ABI pack/dequantize primitives for block-size-16 NVFP4 activation storage (`nfn_native_tile_float32_to_nvfp4_packed`, `nfn_native_tile_nvfp4_packed_to_float32`).
- [x] Projection-family NVFP4 activation kernels with fp32 accumulation for `linear`, LM/router/value/reward/denoise heads, tied LM head, KV PCA encode/decode, JEPA heads, deterministic LoRA/TTT/adapter projections, `bitlinear_ternary`, `fp8_linear`, `mx_linear`, MLP projections, and ACT halt projection; `nf4_linear` stays excluded because it owns a separate packed NF4 base-weight contract.
- [x] Attention Q/K/V NVFP4 support for SDPA, sparse/window/native/streaming-sink attention variants, differential attention, causal/fused causal attention, MLA, and routed attention experts with fp32 score/softmax and route-weight accumulation.
- [x] Explicit no-NVFP4 reasons for losses, optimizers, stochastic masks, integer/hash outputs, and source/delegated nodes where NVFP4 is not meaningful.
- [x] CPU and GPU parity tests with NVFP4 tolerances for projection-family and attention-family activations plus source-gradient preservation.
- [x] NVFP4 saturation-boundary tests for packed projection inputs.
- [ ] Wire dense GPT native training to real packed NVFP4 activation buffers and projection/attention FP4 GEMM routes. The current compiled dense GPT trainer preserves `--tile-cuda-activation-dtype nvfp4` as intent only and reports `native_activation_packing_active: false`; `--require-native-nvfp4-activation-packing` now fails fast until this native storage path is connected to the trainer.
  - [x] Add the intermediate native C++ `--smoke-nvfp4-pack` preflight so the dense GPT binary itself can load and launch `nfn_native_tile_float32_to_nvfp4_packed` plus `nfn_native_tile_nvfp4_packed_to_float32` on a synthetic CUDA activation tile and report pass/fail JSON before datasets are opened. Full packed activation training remains unchecked above.
  - [x] Tighten the required-NVFP4 contract so dense GPT and GPT-2-evo native `--print-plan` / `--dry-run` JSON return a failure status when packed activation storage is required but still missing, while preserving diagnostic JSON with the exact remaining native kernel/storage work.

## CUDA 13.3 RTX 5090 parity status

- [x] Revisited the CUDA-visible tests after installing CUDA Toolkit 13.3 for WSL. `NFN_TILE_CUDA_TEST=1 python -m pytest tests/test_tile_cuda_gpu.py tests/test_tile_cuda_ops.py tests/test_tile_cuda_optimizer.py tests/test_tile_cuda_modules.py -q -rs` passed with `537 passed, 6 warnings` on the dedicated RTX 5090; sandboxed runs still skip/fail GPU discovery because the sandbox does not expose CUDA/NVML.
- [x] Rebuilt and rechecked dense GPT native after the CUDA 13.3 reinstall: `bash tools/build_native_train_tile_ops.sh`, `bash tools/build_native_gpt_cli.sh`, `python -m pytest tests/test_native_gpt2.py -q`, `python -m pytest tests/test_tile_cuda_examples.py -q`, `NFN_TILE_CUDA_TEST=1 python -m pytest tests/test_tile_cuda_gpu.py -q -rs`, `python tools/check_native_no_torch_deps.py`, and `git diff --check` passed.
- [x] Re-ran the full CUDA 13.3 SM120 validator after tightening the required-NVFP4 native preflight contract. `bash tools/validate_sm120_cuda13.sh` passed on the dedicated RTX 5090, including the no-Torch dependency gate, Tile fill smoke, NVFP4 pack smoke, TinyStories transformer-LM CUDA smoke, LM-head benchmark, and `tests/test_native_gpt2.py` (`109 passed, 1 skipped in 486.83s`).
- [x] Added a default one-step runtime-contract probe to `tools/validate_sm120_cuda13.sh` so the normal CUDA health gate proves the promoted dense-GPT speed routes at runtime, not only through optional paired benchmarks. The fast RTX 5090 validation wrote `/tmp/nfn_sm120_cuda13_runtime_contract_summary_20260628.json` with `graph_editor_tensor_flow=false`, `torch_required=false`, `optimized_kernel_contract_passed=true`, `train_loss_host_d2h_count=0`, `linear_tk_qkv_first_use_prewarm_success_count=1`, `block_backward_qkv_dinput_before_dweight_count=96`, `layer_norm_backward_affine_row_chunk_size=128`, `linear_backward_bias_threads_per_block=512`, and `lm_head_classifier_backward_path_class=diagnostic-cuda-graph-wrapper`.
- [x] Split llm.kittens parity gating from candidate-only stage-timing diagnostics. `tools/bench_native_gpt_sm120_parity.sh` now skips its default `1.003x` metric-ratio gates when `NFN_SM120_PARITY_STAGE_TIMING=1` unless an explicit `NFN_SM120_PARITY_MAX_CANDIDATE_RATIO` is provided, and records `default_metric_ratio_gate=disabled_for_candidate_only_stage_timing` in the paired JSON metadata. The 2026-06-28 dedicated RTX 5090 recheck showed why: the 3-step stage-timed diagnostic run failed at `1.005075x` train-loop wall from attribution overhead, while the matching non-stage paired run passed at `1.000735x` train-loop wall and `1.000462x` steady-state CUDA-event time with no external compute load. A stronger 5-step, 3-sample non-stage gate then passed with median ratios `0.992989x` train-loop wall, `0.994042x` steady-state CUDA-event time, and `1.007140x` tokens/sec with zero selected-GPU compute processes before/after every sample.
- [x] Revisited the failed-test surface again after reinstalling `cuda-toolkit-13-3`. `nvcc --version` reports CUDA `13.3.33`, `nvidia-smi` reports CUDA UMD `13.3` with the RTX 5090 idle, `bash tools/build_native_gpt2_all.sh` rebuilt every native trainer and `libnfn_native_train_tile_ops.so`, `NFN_TILE_CUDA_TEST=1 /home/adam/miniconda3/envs/NeuralFn/bin/python -m pytest tests/test_native_gpt2.py tests/test_tile_cuda_examples.py tests/test_tile_cuda_gpu.py tests/test_tile_cuda_ops.py tests/test_tile_cuda_optimizer.py -q -rs` passed with `241 passed`, `/home/adam/miniconda3/envs/NeuralFn/bin/python -m pytest tests/test_template_presets.py -x -q` passed with `26 passed`, `tools/check_native_no_torch_deps.py --skip-artifacts --json` passed, and a one-step `cli/scripts/train_gpt.py --tinystories --max-steps 1 --train-loss-every-steps 1 --eval-every-steps 0 --no-checkpoint --json-out /tmp/nfn_cuda133_gpt_smoke.json` smoke reported `status: "native-transformer-lm-trained"`, `backend: "tile-cuda"`, and CUDA runtime/driver `13.3`.
- [x] Revisited the CUDA 13.3 failed-test surface after the latest WSL toolkit reinstall. The focused GPU/native suite `NFN_TILE_CUDA_TEST=1 /home/adam/miniconda3/envs/NeuralFn/bin/python -m pytest tests/test_native_gpt2.py tests/test_tile_cuda_examples.py tests/test_tile_cuda_gpu.py tests/test_tile_cuda_ops.py tests/test_tile_cuda_optimizer.py -q -rs` passed with `243 passed in 312.46s`, and the full repository suite `/home/adam/miniconda3/envs/NeuralFn/bin/python -m pytest -q` passed with `1180 passed, 4 skipped, 20 warnings, 468 subtests passed in 440.89s`. There are no known red tests from the CUDA reinstall pass.
- [x] Refreshed parity under the same CUDA 13.3 environment with selected-GPU locking and no external compute load: `NFN_SM120_PARITY_STEPS=5 NFN_SM120_PARITY_SAMPLES=3 NFN_SM120_PARITY_WARMUP=1 ... bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2465.408 ms/step` and NeuralFn at `2537.593 ms/step`, or `1.029294x` train-loop wall time and `0.971566x` tokens/sec. The current gap remains native kernel work, not failed CUDA setup.
- [x] Refreshed the CUDA 13.3 one-step stage/shape profile with `NFN_NATIVE_GPT_LINEAR_SHAPE_STATS=1`. The current hot buckets remain `block_backward` at about `1367 ms`, `lm_head_backward` at about `721 ms`, and model forward at about `641-665 ms`; LM-head backward is about `222 ms` logits, `67.6 ms` CE, `251.8 ms` dHidden, and `176.2 ms` dWeight. Shape stats still route LM-head logits through TK BF16 `50304,8192,768,T,N`, LM-head dHidden through BF16 GEMMEx `768,8192,50304,N,N`, and LM-head dWeight through cuBLASLt `768,50304,8192,N,T`, so the remaining work should start at the fused/co-scheduled LM-head backward contract rather than rechecking CUDA install failures.
- [x] Matched the native tied token initialization contract to llm.kittens: only the real 50,257 tokenizer rows are initialized, the 47 padded rows stay zero in FP32 master and BF16 shadow storage, and the default public-vocab BF16 CE path uses no-pad-zero Tile entry points. Runtime JSON reports `lm_head_ce_pad_zero_skipped`, `token_weight_padding_zero_enabled`, `token_weight_init_elements`, and `token_weight_padding_elements`.
- [x] Promote the token-weight vector4-strided initializer as the CUDA 13.3
  RTX 5090 native dense-GPT startup default. The `token_weight_vector4_strided`
  wrapper profile still compares baseline
  `NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT=0` against candidate `=1`,
  and native JSON reports `token_weight_vector4_strided_init_requested` plus
  the distinct
  `device-vector4-strided-power2-deterministic-fused-bf16-shadow-padded-zero`
  strategy by default so route-change gates can see the Tile dispatch. The
  dedicated RTX
  5090 rebuilt 2-sample same-script gate passed at `0.989905x` setup wall,
  `0.987217x` token init, and `0.997838x` total wall versus the old default,
  with
  `graph_editor_tensor_flow=false` and `torch_required=false`.
- [x] Added first-class same-script coverage for the rejected BF16-pattern token-weight shadow writer. The `token_weight_bf16_pattern` wrapper profile now compares baseline `NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_PATTERN_INIT=0` against candidate `=1`, native JSON reports `token_weight_bf16_pattern_init_requested`, and the candidate strategy is labeled `device-vector4-power2-deterministic-fused-bf16-pattern-shadow` so route-change gates can distinguish it from the default conversion-based vector4 BF16-shadow writer. The CUDA 13.3.33 dedicated RTX 5090 5-sample startup-only revalidation changed the route and improved mean setup wall to `0.954257x` plus mean token-weight init to `0.913903x`, but kept the profile rejected because token-weight init was unstable with median `1.021956x` and max `1.095803x`.
  - 2026-06-25 reran the same startup-only profile after confirming the RTX 5090 was visible unsandboxed with CUDA UMD 13.3 and no compute processes. The route still changed, but total setup wall stayed flat at `0.998033x` mean / `1.015865x` median, `setup.token_weight_init.total_ms` stayed flat only on mean at `0.998156x` while regressing to `1.022682x` median, and `setup.uint16_arena_materialize.total_ms` regressed to `1.075542x` mean / `1.032256x` median, so `NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_PATTERN_INIT=1` remains diagnostic-only.
- [x] Refreshed the canonical NeuralFn-vs-llm.kittens parity baseline with selected-GPU locking and idle checks: `NFN_SM120_PARITY_STEPS=5 NFN_SM120_PARITY_SAMPLES=2 NFN_SM120_PARITY_WARMUP=0 ... bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2460.950 ms/step` and NeuralFn at `2533.160 ms/step`, or `1.029377x` train-loop wall time and `0.971123x` tokens/sec versus `/mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh`.
- [x] Current CUDA 13.3 stage timing confirms the remaining gap is native GPU work, not Python or graph-editor traversal. The 5-step candidate profiles ranked `block_backward` at about `6479 ms`, `lm_head_backward` at about `3115 ms`, `train.model_forward` at about `3120 ms`, `block_forward` at about `3082 ms`, then `block_backward.mlp_proj`, `block_backward.attn_sdpa`, `block_backward.mlp_fc`, `lm_head_backward.logits`, `lm_head_backward.dhidden`, and `lm_head_backward.dweight`.
- [x] Added default-off CUDA-event setup timing for startup bisection. `NFN_NATIVE_GPT_SETUP_EVENT_TIMING=1` records `timing.setup_cuda_event_timing` for selected setup kernels and `tools/paired_kernel_speed.py` flattens those as `setup.cuda_event.*` metrics, making it possible to separate host enqueue/sync artifacts from true device time before changing token-weight or BF16-refresh kernels. The first dedicated RTX 5090 startup-only smoke measured `setup.token_weight_init` at about `18.8 ms` of CUDA-event time, confirming that the older 150-170 ms host bucket was not pure token kernel work. The SM120 wrappers expose this as `NFN_SM120_NATIVE_SETUP_EVENT_TIMING=1` and `NFN_SM120_PARITY_SETUP_EVENT_TIMING=1`.
- [x] Keep `NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS=1` diagnostic-only after the CUDA 13.3 recheck. It reduced the `lm_head_backward` bucket but allocated full resident logits, pushed the run near the 32 GiB memory cliff, inflated `block_backward.attn_sdpa`, and measured `13.437483x` train-loop wall time versus the default over the 5-step, 2-sample native-vs-native paired benchmark.
- [x] Keep the no-extra-memory LM-head ordering probes rejected after the CUDA 13.3 recheck. `NFN_NATIVE_GPT_LM_HEAD_DWEIGHT_BEFORE_DHIDDEN=1` measured `1.002230x` train-loop wall time in a 5-step probe, and `NFN_NATIVE_GPT_LM_HEAD_REVERSE_CHUNKS=1` measured `1.003616x`; neither should be promoted.
  - 2026-06-22 added `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_dweight_before_dhidden` as the reproducible same-script wrapper profile for the rejected dWeight-before-dHidden ordering probe. It expands to `NFN_NATIVE_GPT_LM_HEAD_DWEIGHT_BEFORE_DHIDDEN=1`; the dedicated RTX 5090 2-step, 2-sample wrapper gate proved the route change but rejected it at `1.001517x` train-loop wall time, `1.000862x` LM-head backward, and `1.000612x` block backward. Keep it diagnostic-only.
- [x] Keep the LM-head dHidden/dWeight two-stream schedule rejected after the latest CUDA 13.3 dedicated-GPU recheck. A 2-step, 2-sample same-script run of `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_concurrent_dhidden_dweight` selected the display-disabled RTX 5090, changed the schedule from `serial-dhidden-before-dweight` to `two-nonblocking-cuda-streams-after-ce-event`, and measured `0.999265x` train-loop wall time but failed the strict hot-stage gates with `1.009284x` `stage.lm_head_backward.total_ms` and `1.001567x` `stage.block_backward.mlp_proj.total_ms`. Do not promote it; the next LM-head work still needs a real fused/cooperative classifier-backward kernel rather than another stream-ordering default.
- [x] Keep the narrower LM-head last-dWeight overlap schedule rejected after the CUDA 13.3 dedicated-GPU recheck. `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_overlap_last_dweight` expands to `NFN_NATIVE_GPT_LM_HEAD_OVERLAP_LAST_DWEIGHT=1`, queues only the last processed row chunk's LM-head dWeight on the side stream, and waits before the next microbatch/optimizer. The latest 5-step, 3-sample same-script confirmation proved the route active (`lm_head_overlap_last_dweight_queue_count: 40`, `lm_head_overlap_last_dweight_sync_count: 40`) but regressed train-loop wall time to `1.001676x` and train tokens/sec to `0.998350x`; the preceding stage-timed probe also missed the LM-head backward gate at `1.000164x`. The wrapper now rejects the profile unless `NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1` is set.
- [x] Add an optimizer/gradient-clip Tile-size bisection knob without changing the default. `NFN_TILE_CUDA_OPTIMIZER_TILE_SIZE=1024|2048|4096` now retile-builds the native `sumsq`, device-scale, and multi-buffer AdamW kernels, and the Tile ops ABI exposes `nfn_native_tile_optimizer_tile_size()` so dense GPT JSON reports `optimizer_tile_size` / `optimizer_tile_strategy` for same-script route gating. The CUDA 13.3 dedicated RTX 5090 2048 candidate passed the optimizer smoke but stayed rejected: the 3-step, 3-sample native-vs-native benchmark measured `0.999946x` `stage.adamw_update.total_ms` and `0.999543x` `stage.gradient_clip.total_ms`, with no material optimizer win and unrelated hot-stage gates still missed. Keep the default at 1024; the remaining parity work is still LM-head/block-backward kernel work, not optimizer support-stage retile.
- [x] Keep the LayerNorm affine row chunk at 128 after the 64-row recheck. `NFN_SM120_NATIVE_CANDIDATE_PROFILE=layernorm_affine_row_chunk_64` proves the smaller route and improves setup wall (`0.978586x`) but misses current full-loop/reference gates (`1.000214x` train-loop wall, `1.000232x` steady-state CUDA-event timing, `1.000290x` block backward, `1.001836x` MLP projection, `1.000143x` LM-head backward, `1.001641x` candidate-over-llm.kittens train-loop wall), so this is a rejected diagnostic rather than a default.
- [x] Keep the LayerNorm affine row chunk at 128 after the 96-row recheck. `NFN_SM120_NATIVE_CANDIDATE_PROFILE=layernorm_affine_row_chunk_96` proves the intermediate route but still misses hot-stage gates (`1.000296x` MLP projection, `1.000002x` LM-head backward), so this is also a rejected diagnostic rather than a default.
- [x] Keep the hot forward TK fallback probes rejected after the CUDA 13.3 refresh. The current profile has QKV forward and MLP FC forward TK shapes among the largest forward buckets, but disabling those TK routes failed same-script candidate gates: QKV target stage `1.143374x`, and MLP FC train-loop/block-backward/target-stage `1.016916x` / `1.034425x` / `1.000722x`. The next useful work is a new fused/cooperative kernel implementation for the remaining LM-head/block-backward gap.
- [x] Keep the remaining BF16 CE store-policy probes rejected after the latest CUDA 13.3 dedicated-GPU rerun. `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_ce_vec8_normal_store` changed the CE route to `vec8-loads-normal-stores` and improved the narrow CE bucket to `0.999055x`, but missed the strict total LM-head gate at `1.009078x` and regressed LM-head logits to `1.024165x`. `lm_head_ce_scalar_streaming_store` changed the route to `vec8-loads-scalar-streaming-stores` and regressed train-loop wall to `1.020535x`, steady-state CUDA-event wall to `1.026691x`, total LM-head backward to `1.122725x`, and CE to `2.054816x`. Cached scalar stores remain the default.
- [x] Keep the CUDA 13.3 cuBLASLt route probes rejected. `NFN_TILE_CUDA_CUBLASLT_HEURISTIC_INDEX=1` measured `1.006633x` train-loop wall time, and `NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_LARGE_SHAPES=0` measured `1.004884x`; the existing default route remains faster.
- [x] Keep the packed-attention storage toggles rejected as throughput defaults. `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_ACTIVATIONS=0` reduced setup but measured `1.106053x` train-loop wall time because it increased recompute work, and `NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_BF16=0` measured `1.009204x`.
- [x] Keep the BF16/uint16-arena-first startup diagnostic rejected. `NFN_NATIVE_GPT_UINT16_ARENA_FIRST=1` reverses only the default split-arena materialization order and reports `arena_materialize_order`; the CUDA 13.3.33 dedicated RTX 5090 7-sample startup gate changed the route to `uint16-then-float` but regressed setup wall to `1.013035x` mean / `1.010524x` median, uint16 arena materialization to `2.369884x`, and token-weight init to `1.135714x`.
- [x] Refreshed a 1-step CUDA 13.3 stage/shape profile after the dense GPT LM-smoke fix with `NFN_NATIVE_GPT_LINEAR_SHAPE_STATS=1`. The current default still spends about `1349 ms` in `block_backward`, `732 ms` in `lm_head_backward`, and `638 ms` in model forward per optimizer step. LM-head backward breaks down to about `221 ms` logits, `67 ms` CE, `265 ms` dHidden, and `176 ms` dWeight; block backward remains concentrated in MLP projection (`352 ms`), MLP FC (`293 ms`), attention SDPA-to-QKV (`264 ms`), and QKV (`218 ms`). This confirms the remaining gap is not startup, Torch, or graph-node flow.
- [x] Keep the latest existing-route block/MLP probes rejected. `NFN_NATIVE_LINEAR_TK_DINPUT_DISABLE_SHAPE=3072,65536,768,N,N` showed an unconfirmed short-run `0.991338x` train-loop wall result, but route counters did not change, so it is not evidence for a default change. `NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE=768,65536,3072,T,N` did change the route by reducing TK calls from `1632` to `1344`, but regressed train-loop wall to `1.023601x` and tokens/sec to `0.976942x`.
- [x] Removed one SDK startup dispatch hop for dense GPT-family native runs. `neuralfn.native_train.build_native_train_run_config("gpt"|"gpt2"|"gpt3"|"nanogpt", ...)` now resolves directly to `nfn_gpt_native_train --model-family ...` when `NFN_NATIVE_GPT_CLI` is set or `build/nfn_gpt_native_train` exists, while explicit `NFN_NATIVE_TRAIN_CLI` / `native_train_cli=` still forces the unified frontend. This is a startup-path cleanup only; the remaining parity item below is still the native kernel-throughput gap.
- [x] Removed the matching SDK startup dispatch hop for other compiled family
  targets. `build_native_train_run_config("gpt2-evo"|"llama"|"mixllama"|"jepa"|
  "semantic-router-moe"|"deepseek-v4", ...)` now resolves directly to the
  family binary when present, or to `NFN_NATIVE_<FAMILY>_CLI`, while explicit
  `NFN_NATIVE_TRAIN_CLI` / `native_train_cli=` still forces the unified
  frontend. Dense GPT aliases keep passing `--model-family`; family-specific
  binaries receive only their native args.
- [x] Exposed the strict cooperative LM-head backward guard through the Python
  native-training SDK. `build_native_train_run_config(...,
  require_cooperative_lm_head_backward=True)` now appends the dense-GPT
  `--require-cooperative-lm-head-backward` flag without importing Torch,
  suppresses duplicate strict flag spellings, and raises for non-dense family
  targets. This keeps Python launchers on the same native parity gate while the
  real fused LM-head Tile kernel remains the next open performance item.
- [x] Removed the matching top-level `nfn train` dispatch hop for those family
  targets. `nfn train --base-model gpt2-evo ...` and the other compiled-family
  names now go straight to `NFN_NATIVE_<FAMILY>_CLI`, `build/nfn_<family>_native_train`,
  or an installed family binary; the CLI only injects `--base-model` when it
  deliberately falls back to the unified `nfn_native_train` frontend.
- [x] Fixed the all-in-one native build order so
  `tools/build_native_gpt2_all.sh` rebuilds `libnfn_native_train_tile_ops.so`
  before `nfn_gpt_native_train_linked`. The SDK/CLI prefer the linked dense GPT
  binary for lower startup overhead, so the preferred compiled path now links
  against the current Tile kernels instead of a stale library from an earlier
  build.
- [x] Rejected `NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1` as a current block-backward default. Earlier probes were mixed, and the 2026-06-25 CUDA 13.3 dedicated RTX 5090 5-step, 3-sample rerun proved the route change by moving 240 first-write bias gradients to direct writes, but still rejected promotion at `1.005404x` train-loop wall time, `1.001035x` steady-state CUDA-event timing, `0.994864x` tokens/sec, `1.010323x` `stage.block_backward.total_ms`, and `1.020533x` `stage.block_backward.mlp_fc.dweight_bias.total_ms`; keep the BGRADB direct first-write path diagnostic-only unless a future implementation adds stronger route attribution and a durable win.
- [x] Rejected the MLP projection side-stream schedule as a current block-backward direction. A temporary native candidate launched independent MLP projection dInput+dGELU and dWeight+bias on the existing non-blocking block-backward streams after the BF16 grad-out pack. The CUDA 13.3 RTX 5090 same-script gate verified the path but rejected it at `1.007390x` train-loop wall time, `1.013009x` `stage.block_backward.total_ms`, and `1.029721x` `stage.block_backward.mlp_proj.total_ms`; the diagnostic code was removed rather than kept default-off.
- [x] Rejected temporary full-row LM-head allocation as a workaround for the resident full-logit memory cliff. The async-free prototype kept enough pool pressure that `block_backward.attn_sdpa` collapsed to about `9057 ms` in a one-step smoke; the sync `cudaMalloc`/`cudaFree` variant released memory before block backward but spent about `902 ms` in `lm_head_backward` for a single `65536`-token microbatch, including about `195 ms` allocation and `115 ms` free time. This confirms the parity fix cannot be a per-microbatch full-logit allocation; it needs a fused/cooperative row-chunked classifier-backward kernel that keeps the current resident cap.
- [x] Hardened SM120 candidate acceptance so measured candidate changes must also produce a tracked route/strategy/linear-shape/cuBLASLt-plan change. `tools/paired_kernel_speed.py --require-native-route-change` now fails timing-only candidates with no implementation attribution, and `tools/bench_native_gpt_sm120_candidate.sh` enables that gate automatically for real measured candidate changes.
- [x] Removed the artificial row-loss CE fallback from the diagnostic LM-head cooperative sequence wrapper. Optimizer-only cooperative steps now pass a no-loss flag through the raw Tile ABI and reuse the default BF16/u16 no-loss CE+dlogits kernel before sequencing dHidden and dWeight; loss-recording paths still use row-loss or loss-bin collection. This keeps `lm_head_cooperative_backward` measurable without changing validation semantics, but it does not promote the route. The linked trainer one-step RTX 5090 rerun confirmed `lm_head_classifier_ce_no_loss_enabled: true` for the candidate and still rejected it at `1.117578x` train-loop wall and `1.294010x` LM-head backward.
- [x] Rejected the existing LM-head cooperative ABI wrapper as a parity fix. The latest CUDA 13.3 dedicated RTX 5090 one-step same-script probe of `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_cooperative_backward` now proves the diagnostic sequence-wrapper route is active (`lm_head_cooperative_backward_sequence_wrapper_enabled: true`, strategy changed to `diagnostic-sequence-wrapper-ce-side-stream-dhidden-dweight-not-parity`), but still rejects it at `1.007071x` train-loop wall, `1.000602x` `stage.lm_head_backward.total_ms`, `1.001183x` `stage.block_backward.total_ms`, and `1.002039x` `stage.block_backward.mlp_proj.total_ms`. This confirms the current wrapper sequences existing CE/dHidden/dWeight kernels and is not the fused/cooperative classifier-backward kernel needed for parity.
- [x] Promoted the no-loss llm.kittens-style CE+dlogits store route as the dense
  GPT optimizer-step default. `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_ce_no_loss_llmk_style_specialized`
  now compares the default
  `NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_LLMK_STYLE_SPECIALIZED=1` route against
  `=0`, runs the BF16/u16 vec8-load plus streaming-vec8-store kernel, and
  reports `no-loss-llmk-style-dlogits-vec8-loads-streaming-vec8-stores`. The
  CUDA 13.3.33 dedicated RTX 5090 2026-06-28 current-default 3-step, 2-sample
  no-stage rerun passed full trainer and llm.kittens reference gates at
  `0.999669x` train-loop wall, `0.999849x` steady-state CUDA-event wall,
  `1.000333x` train tokens/sec, `0.997332x` candidate-over-llm.kittens
  train-loop wall, and `1.002269x` candidate-over-llm.kittens tokens/sec. This
  closes the CE/dlogits default gap; the strict single-kernel
  classifier/dHidden/dWeight Tile route remains the next experimental LM-head
  item.
- [x] Revisited CUDA 13.3 smoke failures after the WSL toolkit reinstall. The
  rebuilt Tile ops library and dense GPT CLI pass symbol checks, and
  unsandboxed `nvidia-smi` sees the dedicated RTX 5090 with CUDA UMD 13.3 and
  no compute processes. The sandbox still reports CUDA error 35 on device
  allocation, so the native GPT C++ frontend now annotates error 35 as either
  a real runtime/driver mismatch or blocked GPU device access and points at
  unsandboxed `nvidia-smi` plus `--cuda-runtime-lib` /
  `NFN_CUDA_RUNTIME_LIB`. The same optimizer smoke passes unsandboxed with all
  148 AdamW buffer updates.
- [x] Revisited the native trainer test surface after the CUDA 13.3 reinstall.
  `python -m pytest tests/test_native_gpt2.py -q` now passes
  `80 passed, 1 skipped in 351.39s`, and
  `python tools/check_native_no_torch_deps.py --skip-artifacts --json
  --max-entrypoint-seconds 2.0` still passes across GPT/GPT-2-evo, NanoGPT,
  `nfn train`, native inference, SDK exports, binding imports, and benchmark
  dry-run entrypoints without importing Torch/NumPy/tiktoken/dataset-manager or
  graph-runtime shims. `tests/test_tile_cuda_gpu.py` remains skipped in this
  environment, so it is not a live CUDA regression signal.
- [ ] Close the remaining SM120 parity gap with new kernel work, not more default-switch promotion. The next high-value implementation target is the LM-head classifier/backward contract reported in runtime JSON: fuse or otherwise co-schedule row-chunked BF16 logits, public-vocab CE/dlogits, dHidden, and dWeight so the default path is closer to llm.kittens' full-resident fused classifier without triggering the full-logit memory cliff.
  - 2026-06-28 added graph-body route counters for the diagnostic LM-head CUDA Graph wrapper: runtime JSON and paired speed summaries now report cuBLASLt dHidden/dWeight launch counts and Tile dHidden/dWeight fallback counts inside the graph body. This keeps `lm_head_graph_body_cublaslt` and future body candidates measurable, but the true fused LM-head replacement remains unchecked above.
  - 2026-06-28 follow-up: `native_lm_head_true_fused_target` now derives
    `graph_body_cublaslt_launch_mean` and `graph_body_tile_fallback_mean` from
    those counters, so paired benchmark failures show whether the diagnostic
    wrapper body used cuBLASLt or fell back to Tile while the true fused target
    remains open.
  - 2026-06-28 follow-up: added
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=adamw_token_shadow_refresh` so the
    fused token-weight BF16 shadow AdamW refresh default is compared against the
    older two-launch refresh route in the same SM120 candidate harness. This
    closes a benchmark-coverage gap for the optimized optimizer path but does
    not close the LM-head true-fused parity item above.
  - 2026-06-28 follow-up: aligned
    `tools/bench_lm_head_backward_candidate.sh` trainer-chunk profiles with the
    current 28672-row dense GPT LM-head chunk default. Focused
    `trainer-chunk`, strict true-fused, cuBLASLt, row-loss, and loss-bin
    LM-head microbench runs now measure the same row shape as the actual
    trainer unless `NFN_LM_HEAD_BACKWARD_ROWS` overrides it. The corrected
    focused `trainer-chunk` run on the dedicated RTX 5090 measured
    `candidate_to_baseline_ms_per_iter_ratio: 0.999499` at 28672 rows and kept
    `graph_fallback_count: 0`, so the benchmark-shape change preserves the
    current wrapper while making future strict-body probes honest.
  - 2026-06-28 follow-up: added device-side section cycle counters for the
    strict true-fused LM-head cooperative kernel. The raw Tile ABI now exposes
    CE/dHidden/dWeight cycle totals and cooperative-block counts, and
    `build/lm_head_backward_bench` emits `true_fused_*_cycles_per_block` in the
    baseline/candidate JSON. This does not promote the slow strict body, but it
    turns future true-fused failures into targeted CE versus matrix-section
    evidence instead of a single opaque candidate time.
  - 2026-06-28 continuation probe after the CUDA 13.3 reinstall and latest
    validator updates: a one-step, one-sample stage-timed same-script run
    (`/tmp/nfn_sm120_stage_probe_continue_20260628.json`) selected the
    display-disabled RTX 5090 with no compute processes and measured
    llm.kittens at `2449.760 ms/step` versus NeuralFn at
    `2579.740 ms/step` (`1.053058x` train-loop wall,
    `0.949616x` tokens/sec). NeuralFn remained native-only with no route
    counter changes; the current hot buckets were `stage.block_backward` at
    about `1314 ms` and `stage.lm_head_backward` at about `572 ms`
    (`logits` about `177 ms`, cooperative graph body about `394 ms`). This
    keeps the next implementation target on real LM-head/block-backward hot
    kernels, not startup, graph-editor tensor flow, Torch fallback, or external
    GPU load.
  - 2026-06-23 added the dedicated linear-backward microbench to keep the next
    kernel work honest: candidate dInput/dWeight symbols for block backward and
    LM-head can now be timed against the current C ABI without dataset loading,
    graph-editor tensors, Python/Torch, or full trainer-loop noise.
  - 2026-06-24 added and rejected the LM-head CUDA Graph setup-prewarm route.
    The new Tile ABI symbol
    `nfn_native_tile_lm_head_classifier_backward_fused_graph_prewarm_bf16_u16`
    captures graph cache entries without launching CE/dHidden/dWeight, and
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_graph_prewarm` enables the
    required cuBLAS handle, BF16 workspace, and LM-head-only cuBLASLt plan
    prewarm prerequisites. The dedicated RTX 5090 5-step, 2-sample gate proved
    runtime captures moved from `3` to `0` and first-step event time improved
    to `0.969636x`, but rejected default promotion because steady-state event
    time regressed to `1.001249x` and total LM-head backward to `1.000123x`.
    Keep it diagnostic-only; the remaining useful work is still a true fused
    or co-scheduled LM-head classifier-backward kernel body.
  - 2026-06-25 follow-up: graph prewarm now captures both the no-loss graph key
    and the active train-loss graph key, including the loss-bin graph flags when
    configured, so the first logged train-loss step does not pay a separate
    lazy LM-head CUDA Graph capture.
  - 2026-06-25 refreshed the CUDA 13.3.33 post-reinstall LM-head trainer-chunk
    microbench after rebuilding `libnfn_native_train_tile_ops.so` and the
    linked native GPT trainer. The diagnostic non-strict run completed on the
    dedicated RTX 5090 with `candidate_to_baseline_ms_per_iter_ratio:
    1.000324`, `candidate_true_fused_capability: false`, and
    `candidate_cuda_graph_wrapper_only: true`. This keeps the existing graph
    wrapper rejected as parity evidence; the next implementation still needs a
    real fused or co-scheduled LM-head classifier/backward kernel contract.
  - 2026-06-25 changed the default LM-head CUDA Graph body from a single-stream
    CE -> dHidden -> dWeight capture to a post-CE fork/join capture that
    launches dHidden and dWeight on cooperative non-blocking streams before
    joining the caller stream. The focused trainer-chunk microbench improved
    versus the older cooperative sequence (`0.969899x`, and `0.971587x` with
    candidate first), and the full parity probe improved average loop wall to
    `0.989380x`; however strict steady-state parity still failed at
    `1.015315x`, so this is a partial scheduling win, not completion of the
    remaining parity item.
  - 2026-06-25 moved native GPT stage-timing event allocation out of the hot
    diagnostic path. Stage-timed runs now preallocate CUDA event pairs during
    setup with `NFN_NATIVE_GPT_STAGE_TIMING_PREALLOC_EVENTS` /
    `NFN_NATIVE_GPT2_STAGE_TIMING_PREALLOC_EVENTS` defaulting to `16384`, then
    report requested, preallocated, hot-created, and unused-destroyed pair
    counts in native JSON and `tools/paired_kernel_speed.py`. A 3-step
    stage-timed parity sidecar used 11,961 events, so the default is sized to
    cover the common short parity wrapper without hot-path event creation. This
    is diagnostic overhead cleanup only; the remaining parity item is still the
    fused/co-scheduled LM-head classifier-backward kernel body.
  - 2026-06-22 kept the no-loss LM-head classifier CE route default-off after
    retesting it against the current packed-QKV dense GPT default. The route
    sends normal no-loss optimizer steps through the classifier row-loss kernel
    while skipping the loss reduction tail, but the post-default 3-step,
    2-sample dedicated RTX 5090 gate measured `0.998998x` train-loop wall time,
    `1.001029x` tokens/sec, `1.005316x` LM-head backward, and `1.000570x`
    MLP-proj backward, failing the stage gates. The candidate profile now
    forces the older generic no-loss CE path on the baseline side and the
    classifier route on the candidate side, so future runs remain explicit.
  - 2026-06-23 rechecked `lm_head_classifier_ce_no_loss` after the CUDA 13.3
    WSL reinstall and marked real launches as rejected unless
    `NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1` is set. The 3-step,
    2-sample dedicated RTX 5090 stage-timed gate changed the route counters and
    strategies, but regressed train-loop wall to `1.005933x`, LM-head backward
    to `1.087310x`, and LM-head CE to `1.848303x`. Dry-run expansion remains
    available for command inspection.
  - 2026-06-22 fixed native dense GPT runtime attribution for the default
    no-loss LM-head CE+dlogits route. Timing-only runs now increment
    `lm_head_classifier_no_loss_chunk_count`, report
    `lm_head_ce_kernel_strategy:
    "no-loss-dlogits-vec8-loads-scalar-stores"`, and report
    `lm_head_ce_loss_backward_strategy:
    "no-loss-dlogits-public-vocab-no-pad-zero-bf16-u16-targets"` when the
    generic no-loss path is active. The paired speed tool includes the no-loss
    chunk counter in native route changes, so future candidate gates do not
    mistake no-loss timings for row-loss or loss-bin train-loss work.
  - 2026-06-23 rejected the shape-gated cuBLASLt substitution for the current
    LM-head dHidden bucket. `lm_head_cublaslt_dhidden_32768` moved 48 calls from
    BF16 GEMMEx to cuBLASLt on the CUDA 13.3 dedicated RTX 5090 3-step,
    3-sample stage-timed gate, but failed strict gates at `1.000384x`
    train-loop wall, `1.000199x` LM-head dHidden, and `1.001504x` block
    backward. This keeps backend substitution out of the default path and
    reinforces that the remaining parity work needs a real fused/cooperative
    classifier-backward implementation.
  - 2026-06-23 added LM-head dHidden route counters to native JSON and paired
    route-change detection:
    `lm_head_dhidden_tk_gemm_count`,
    `lm_head_dhidden_cublaslt_gemm_count`, and
    `lm_head_dhidden_bf16_gemm_count`. Future candidates that only change the
    dHidden backend no longer need expensive `linear_shape_stats` timing to
    prove they routed.
  - 2026-06-23 added matching block-backward dInput route counters to native
    JSON and paired route-change detection:
    `block_backward_dinput_tk_gemm_count`,
    `block_backward_dinput_cublaslt_gemm_count`, and
    `block_backward_dinput_bf16_gemm_count`. These aggregate the MLP
    projection, MLP FC, attention projection, and QKV dInput bodies so the next
    block-backward candidate must prove it moved the hot dInput GEMM route
    before timing wins are trusted.
  - 2026-06-23 added `NFN_NATIVE_LINEAR_TK_DINPUT_DEFAULT_BLOCK=1` /
    `NFN_TILE_CUDA_LINEAR_TK_DINPUT_DEFAULT_BLOCK=1` as an opt-in diagnostic
    route for block-sized GPT BF16-gradient/BF16-weight dInput shapes with
    `m <= 4096`, `k <= 4096`, valid tile multiples, and `N,N` layout. The
    same-script 3-sample RTX 5090 benchmark moved 384 two-step block dInput
    calls from cuBLASLt to TK, but rejected the route at `1.027648x`
    train-loop wall and `0.973558x` tokens/sec, so keep the cuBLASLt/GEMMEx
    block fallback as the default. Keep the LM-head dHidden classifier shape
    opt-in through `NFN_NATIVE_LINEAR_TK_DINPUT=1` or
    `NFN_NATIVE_LINEAR_TK_DINPUT_ENABLE_SHAPE=...` because the 5090 paired
    checks also rejected that larger shape.
  - 2026-06-22 promoted the no-loss LM-head CE specialization to the default
    behind `NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_DEFAULT_SPECIALIZED=1` /
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_ce_no_loss_default_specialized`.
    The route removes runtime CE mode branches for the timing-only no-loss
    BF16/u16 path and reports `lm_head_ce_kernel_strategy:
    no-loss-default-specialized-dlogits-vec8-loads-scalar-stores`. The first
    dedicated RTX 5090 3-step, 2-sample check proved the route and improved the
    target LM-head buckets but failed strict promotion on unrelated
    block-backward noise; the follow-up 3-step, 5-sample gate passed cleanly at
    `0.975413x` train-loop wall, `1.025216x` tokens/sec, `0.912765x` LM-head
    backward, `0.552364x` LM-head CE, and `0.990994x` block backward. The
    candidate profile now forces the baseline side to
    `NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_DEFAULT_SPECIALIZED=0`; use that opt-out
    only to compare against the older generic no-loss CE+dlogits kernel.
    2026-06-23 tightened the paired wrapper so this default-route profile is no
    longer treated as LM-head-only for stage-timed runs: it now also gates
    `stage.block_backward.total_ms`, preventing a CE-only pass from hiding
    unrelated block-backward regression.
    2026-06-24 refreshed the same default on the current clean tree with a
    3-step, 2-sample same-script RTX 5090 gate. The route change was visible in
    `lm_head_ce_no_loss_default_specialized_*` and `lm_head_ce_kernel_strategy`,
    and the candidate passed whole-loop gates at `0.968164x` train-loop wall,
    `0.979934x` steady-state CUDA-event wall, `1.032888x` tokens/sec,
    `0.913491x` LM-head backward, `0.553762x` LM-head CE, and `0.984587x`
    block backward versus the generic no-loss CE+dlogits baseline.
    2026-06-26 reran the profile after the CUDA 13.3.33 reinstall and MLP FC
    ordering rollback. The candidate still proved the route by changing
    `lm_head_ce_no_loss_default_specialized_*` and
    `lm_head_ce_kernel_strategy`, measured `0.975099x` train-loop wall,
    `0.976966x` steady-state CUDA-event wall, `1.025547x` tokens/sec, and
    `0.911191x` LM-head backward. The benchmark wrapper now skips only the
    missing standalone `stage.lm_head_backward.ce.total_ms` gate for this
    default-vs-legacy no-loss profile because the current LM-head timing path
    reports the CE work through the diagnostic CUDA Graph cooperative stage;
    other CE profiles keep their explicit CE sub-stage gates.
  - 2026-06-22 promoted the LM-head loss-bin train-loss reduction route to the default after the stronger CUDA 13.3 dedicated RTX 5090 same-script check proved a durable logged-loss path win. `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_loss_bins` now forces its baseline command to `NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=0` and candidate command to `=1`, so it still compares the new default against the older row-loss route. The 3-step, 2-sample check measured `0.977282x` train-loop wall time, `1.023250x` tokens/sec, `0.909537x` LM-head backward, and `0.541619x` LM-head CE, with `lm_head_classifier_loss_bin_launch_count` moving from `0` to `48`. Set `NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=0` only for regression checks against the older row-loss tail.
  - 2026-06-22 rechecked the latest CUDA Toolkit 13.3 for WSL install on the dedicated RTX 5090. Sandboxed `nvidia-smi` still reports OS-blocked NVML, but the unsandboxed GPU path reports NVIDIA-SMI `610.43.02`, CUDA UMD `13.3`, no compute processes, and the parity harness runs successfully with selected-GPU locking. A one-step, stage-timed same-script comparison measured llm.kittens at `2651.960 ms/step` and NeuralFn at `2692.140 ms/step`, or `1.015151x` train-loop wall time and `0.985078x` tokens/sec. The remaining hot buckets are now `block_backward` (`1260.570 ms`), `train.model_forward` (`765.722 ms`), and `lm_head_backward` (`639.129 ms`), not CUDA setup, Torch startup, graph-editor node flow, or external GPU load.
  - 2026-06-22 kept the latest existing side-stream schedules diagnostic-only on the dedicated RTX 5090. `lm_head_concurrent_dhidden_dweight` proved the two-stream LM-head schedule but measured `1.000742x` train-loop wall time and `1.007977x` LM-head backward in the one-step gate. `qkv_concurrent_dinput_dweight` proved the block QKV side-stream route but regressed to `1.022447x` train-loop wall time and `1.049762x` block backward. Do not promote these switches; the next useful work is still a real fused/cooperative LM-head or block-backward kernel route.
  - 2026-06-22 current CUDA 13.3 dedicated RTX 5090 parity refresh still shows a real gap: one stage-timed same-script sample measured llm.kittens at `2455.570 ms/step` and NeuralFn at `2655.010 ms/step` (`1.081219x`, `0.924884x` tokens/sec). Hot NeuralFn buckets remain `block_backward` (`1275.380 ms`), `train.model_forward` (`769.952 ms`), and `lm_head_backward` (`583.131 ms`). The LM-head path now uses two `32768`-row chunks, with `lm_head_backward.logits` `173.411 ms`, CE `69.2295 ms`, dHidden `176.722 ms`, and dWeight `162.153 ms`.
  - 2026-06-23 promoted the LayerNorm affine row chunk from 256 to 128 rows for the hot fused residual-backward path after the CUDA 13.3 dedicated RTX 5090 no-stage same-script gate measured `0.997750x` train-loop wall time and `1.002270x` tokens/sec across three interleaved five-step samples. Runtime JSON reports `block_state_layout.layer_norm_backward_affine_row_chunk_size`, the paired tool treats that nested value as a route metric, and the wrapper profile `layernorm_affine_row_chunk_128` now forces its baseline to `NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=256` so future runs still compare the default against the older route. The `512` profile remains diagnostic-only after measuring `1.019837x` train-loop wall time / `1.039994x` block backward.
  - 2026-06-28 refreshed the rejected `NFN_SM120_NATIVE_CANDIDATE_PROFILE=layernorm_affine_row_chunk_64` profile after the CUDA 13.3.33 reinstall. The 3-step, 2-sample same-script native/reference gate changed `block_state_layout.layer_norm_backward_affine_row_chunk_size` from 128 to 64 and improved setup wall to `0.978586x`, but failed promotion gates at `1.000214x` train-loop wall, `1.000232x` steady-state CUDA-event timing, `1.000290x` block backward, `1.001836x` MLP projection, `1.000143x` LM-head backward, and `1.001641x` candidate-over-llm.kittens train-loop wall. Keep the 128-row default unless a future kernel change removes the adjacent-stage regressions.
  - 2026-06-23 added and rejected the intermediate `NFN_SM120_NATIVE_CANDIDATE_PROFILE=layernorm_affine_row_chunk_96` profile. The 5-step, 3-sample dedicated RTX 5090 stage-timed gate changed `block_state_layout.layer_norm_backward_affine_row_chunk_size` from 128 to 96 and improved train-loop wall to `0.999112x`, but still missed hot-stage gates with `1.000296x` MLP projection and `1.000002x` LM-head backward. Keep the 128-row default.
  - 2026-06-23 refreshed llm.kittens parity after the 128-row LayerNorm affine default. The dedicated RTX 5090 no-stage five-step sample measured llm.kittens at `2471.728 ms/step` and NeuralFn at `2496.600 ms/step`, or `1.010063x` train-loop wall time and `0.988567x` tokens/sec, with the selected GPU idle before/after the sample. The remaining gap is now roughly 1.0%, so the next useful work should be a real hot-bucket kernel route rather than parameter/default churn.
  - 2026-06-23 refreshed current no-stage parity again after the row-chunk rejections: `NFN_SM120_PARITY_STEPS=5 NFN_SM120_PARITY_SAMPLES=2 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_current_continue.json bash tools/bench_native_gpt_sm120_parity.sh` measured llm.kittens at `2451.837 ms/step` and NeuralFn at `2499.550 ms/step`, or `1.019517x` train-loop wall time and `0.980526x` tokens/sec. The gap remains real and still points to new LM-head/block-backward kernel work.
  - 2026-06-22 added separate linear-bias row-chunk profiling for the Tile bias reducer used by split block dWeight+bias diagnostics. Runtime JSON now reports `block_state_layout.linear_backward_bias_row_chunk_size`, the paired tool treats it as route proof, and wrapper profiles `linear_bias_row_chunk_256` / `linear_bias_row_chunk_1024` reproduce `NFN_NATIVE_GPT_LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE=256` / `1024`. The focused dedicated RTX 5090 gate showed 256 improving target block buckets (`0.999220x` block backward, `0.998559x` MLP projection dWeight+bias, `0.999280x` MLP FC dWeight+bias), but the first automatic gate rerun still failed on train-loop noise at `1.000970x`. The 2026-06-23 follow-up same-script 5-step, 3-sample gate promoted 256 rows after measuring `0.995306x` train-loop wall time, `1.005073x` tokens/sec, and `0.994292x` total wall time. The 1024 candidate remains rejected at `1.001730x` train-loop, `1.002737x` block backward, and `1.000599x` MLP FC dWeight+bias. The remaining gap still needs real fused/cooperative block or LM-head kernel work.
  - 2026-06-24 added launch-width diagnostics for the same Tile linear-bias reducer. Runtime JSON now reports `block_state_layout.linear_backward_bias_threads_per_block`, the paired tool treats that nested value as route proof, and `NFN_SM120_NATIVE_CANDIDATE_PROFILE=linear_bias_threads_512` compares `NFN_NATIVE_GPT_LINEAR_BACKWARD_BIAS_THREADS=512` against the 256-thread route. The initial CUDA 13.3 dedicated RTX 5090 3-step, 2-sample stage-timed gate proved the route change and improved train-loop wall (`0.989155x`) plus block backward (`0.961836x`), but rejected promotion because steady-state CUDA-event step time regressed to `1.000446x` and MLP FC dWeight+bias regressed to `1.066923x`. A later corrected-lib CUDA 13.3.33 rerun kept the 512-thread route as the default after it improved train-loop wall, steady-state CUDA-event timing, block backward, and the MLP dWeight+bias buckets.
  - 2026-06-23 post-promotion parity refresh with rebuilt linked Tile ops measured NeuralFn effectively even with llm.kittens on train-loop wall: `NFN_SM120_PARITY_STEPS=5 NFN_SM120_PARITY_SAMPLES=2 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_after_linear_bias_256_default.json bash tools/bench_native_gpt_sm120_parity.sh` reported llm.kittens at `2500.759 ms/step`, NeuralFn at `2499.380 ms/step`, `0.999463x` train-loop wall time, and `0.998374x` tokens/sec, with the selected RTX 5090 idle and locked. A follow-up two-step stage-timed sample was noisier (`1.024001x`) but kept the same hot-bucket ordering: block backward (`2546.870 ms`), model forward (`1376.120 ms`), LM-head backward (`1173.540 ms`), MLP projection backward (`655.245 ms`), attention-to-QKV backward (`531.887 ms`), and MLP FC backward (`524.594 ms`). Treat future work as targeted hot-kernel work; do not infer a startup, graph-editor, Torch, or external-load issue from older parity notes.
  - 2026-06-23 refreshed parity again after the LM-head row-loss partial-reduce default: `NFN_SM120_PARITY_STEPS=5 NFN_SM120_PARITY_SAMPLES=2 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=0 NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_after_row_loss_partial_default.json bash tools/bench_native_gpt_sm120_parity.sh` reported llm.kittens at `2500.666 ms/step`, NeuralFn at `2499.970 ms/step`, `0.999869x` train-loop wall time, and `0.997328x` tokens/sec. Runtime JSON confirmed `lm_head_ce_row_loss_sum_accumulate_enabled: false` and `block_state_layout.linear_backward_bias_row_chunk_size: 256`. The remaining actionable work is a real fused/cooperative LM-head or block-backward kernel, not another stale diagnostic rerun.
  - 2026-06-23 reran parity after restoring default NeuralFn CUDA-event train-loop timing: `NFN_SM120_PARITY_STEPS=3 NFN_SM120_PARITY_SAMPLES=1 NFN_SM120_PARITY_WARMUP=0 NFN_SM120_PARITY_CUDA_VISIBLE_DEVICES=auto NFN_SM120_PARITY_PROFILE_DIR=none NFN_SM120_PARITY_JSON_OUT=/tmp/nfn_sm120_parity_event_default_3step.json bash tools/bench_native_gpt_sm120_parity.sh` selected the dedicated RTX 5090, confirmed zero compute processes before/after the sample, and reported llm.kittens at `2590.420 ms/step`, NeuralFn at `2514.607 ms/step`, `0.970733x` train-loop wall time, and `1.022057x` tokens/sec. The CUDA-event totals now appear by default; the steady-state event slice was still slower at `1.015231x`, so future work should target steady-state LM-head/block hot kernels rather than startup, graph-editor flow, or Torch fallbacks.
  - 2026-06-23 tightened the native-vs-native candidate wrapper after that steady-state event signal: measured training comparisons now pass `NFN_NATIVE_GPT_TRAIN_LOOP_EVENT_TIMING=1` to both sides by default and auto-gate `train_loop_cuda_event_steady_state_wall_ms_per_step=1.000` when `max-steps > 1`. This keeps new candidates from passing on first-step/setup movement while regressing the steady-state CUDA-event slice.
  - 2026-06-23 rechecked the existing `cudaMallocAsync` allocator as a startup fix and kept it rejected. `NFN_SM120_NATIVE_CANDIDATE_PROFILE=cuda_malloc_async` now expands to baseline `NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC=0` versus candidate `=1`; the dedicated RTX 5090 3-sample startup-only run enabled `device_allocator_strategy: cudaMallocAsync-null-stream` but failed the setup gate at `1.145549x`, with `setup.uint16_arena_materialize.total_ms` regressing to `1.775565x`. Keep synchronous `cudaMalloc` as the default and focus startup work elsewhere.
  - 2026-06-23 aligned the paired benchmark wrapper with existing rejection evidence: `NFN_SM120_NATIVE_CANDIDATE_PROFILE=layernorm_affine_row_chunk_512` and `linear_bias_row_chunk_1024` now fail before launching a real benchmark unless `NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1` is set. Dry-run plan expansion still works for both profiles so old route evidence remains inspectable.
  - 2026-06-24 kept rejected-profile enforcement on `bf16_attention_grad_out` after the rebuilt verification rejected default promotion. The stale attention bisections `bf16_attention_grad_out`, `bf16_attention_dprep_grad_out`, and `attention_dprep_float_hd64_specialized` still require `NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1` for real paired launches.
  - 2026-06-25 added and rejected the midpoint `packed_attention_bwd_batch_96` profile. The CUDA 13.3.33 dedicated RTX 5090 3-step, 2-sample current/native/reference gate changed `attention_backward_tk_batch_cap` from `64` to `96`, but failed strict attention-section gates at `1.000597x` attention total, `1.000638x` attention `to_qkv`, and `1.000149x` dprep timing, while candidate-over-llm.kittens train-loop wall remained `1.035674x`. Keep the 64 batch cap default; both smaller/larger batch-cap probes remain diagnostic-only.
  - 2026-06-22 added BGRADB-specific route counters for future block dWeight+bias candidates. Runtime JSON and `tools/paired_kernel_speed.py` now track `linear_cublaslt_bgrad_gemm_count`, `linear_cublaslt_bgrad_direct_write_count`, and `linear_cublaslt_bgrad_accumulate_count`, so a BGRADB first-write or split-bias candidate must prove it changed the actual cuBLASLt BGRADB path instead of only changing generic `linear_cublaslt_gemm_count`.
  - 2026-06-23 dedicated RTX 5090 one-step same-script check rejected `NFN_SM120_NATIVE_CANDIDATE_PROFILE=linear_bias_row_chunk_1024`: it changed `block_state_layout.linear_backward_bias_row_chunk_size` from 512 to 1024, but the new BGRADB counters stayed at 384 BGRADB GEMMs, 0 direct writes, and 384 bias accumulates. It failed ratio gates at `train_loop_wall_ms_per_step=1.009736x`, `stage.block_backward.total_ms=1.008488x`, `stage.block_backward.mlp_proj.total_ms=1.034002x`, `stage.block_backward.mlp_proj.dweight_bias.total_ms=1.050950x`, and `stage.block_backward.mlp_fc.dweight_bias.total_ms=1.000630x`; keep the promoted 256-row default.
  - 2026-06-23 initially rechecked `NFN_SM120_NATIVE_CANDIDATE_PROFILE=combined_device_arena` after the CUDA 13.3 reinstall on the dedicated RTX 5090. The first startup-only gate and follow-up 3-step training gate passed, but a later same-day rerun rejected the combined arena at `1.004991x` train-loop wall time, `0.995098x` tokens/sec, and `1.063067x` setup wall time. The split float arena plus split BF16/uint16 arena route is therefore the default again; use `NFN_NATIVE_GPT_COMBINED_DEVICE_ARENA=1` only for combined-arena comparisons.
  - 2026-06-22 added a default-off LM-head row-loss CE specialization behind `NFN_NATIVE_GPT_LM_HEAD_CE_DEFAULT_SPECIALIZED=1` / `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_ce_default_specialized`. The route removes runtime CE mode branches only for the current default shape (`1024` threads, vec8 BF16 loads, scalar cached stores, `expf`) and reports `lm_head_ce_kernel_strategy: default-specialized-row-loss-vec8-loads-scalar-stores`. The 3-step, 3-sample dedicated RTX 5090 gate proved the route change but rejected it at `1.001545x` train-loop wall, `1.000931x` LM-head backward, and `1.000331x` LM-head CE time. Keep it diagnostic-only; the remaining LM-head gap still needs a cooperative logits/CE/dHidden/dWeight schedule or a materially faster GEMM path, not more CE branch specialization.
  - 2026-06-22 added a matching default-off LM-head loss-bin CE specialization behind `NFN_NATIVE_GPT_LM_HEAD_CE_LOSS_BINS_DEFAULT_SPECIALIZED=1` / `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_ce_loss_bins_default_specialized`. The route uses a dedicated default-shape loss-bin BF16/u16 CE+dlogits kernel and reports `lm_head_ce_kernel_strategy: default-specialized-loss-bins-vec8-loads-scalar-stores`. The 3-step, 3-sample dedicated RTX 5090 gate proved the strategy change and passed train-loop wall (`0.999215x`) but rejected it on LM-head backward (`1.000741x`), LM-head CE (`1.000339x`), and MLP projection (`1.001222x`). Keep it diagnostic-only; this confirms branch specialization is not enough to close the remaining parity gap.
  - 2026-06-22 added a default-off llm.kittens-style LM-head CE specialization behind `NFN_NATIVE_GPT_LM_HEAD_CE_LLMK_STYLE_SPECIALIZED=1` / `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_ce_llmk_style_specialized` and the loss-bin variant `lm_head_ce_loss_bins_llmk_style_specialized`. The route keeps the 1024-thread vec8-load classifier shape but overwrites dlogits with vector streaming stores and reports `lm_head_ce_kernel_strategy: llmk-style-row-loss-vec8-loads-streaming-vec8-stores` or `llmk-style-loss-bins-vec8-loads-streaming-vec8-stores`. The CUDA 13.3 dedicated RTX 5090 same-script gate proved the row-loss route but rejected promotion: train-loop wall improved (`0.997562x`) while LM-head backward (`1.000511x`) and CE (`1.000411x`) missed gates. The loss-bin profile proved request/strategy plumbing, but `lm_head_classifier_loss_bin_launch_count` stayed at zero in that short run; final JSON now derives loss-bin enabled/strategy selection from the runtime launch counter. Keep both diagnostic-only.
  - 2026-06-22 fixed the loss-bin candidate wrapper after a current-CUDA recheck showed `NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=1` was a no-op in default throughput runs: train-loss logging was disabled, so no loss-accumulation tail executed and `lm_head_classifier_loss_bin_launch_count` stayed `0`. The `lm_head_loss_bins`, `lm_head_ce_loss_bins_default_specialized`, and `lm_head_ce_loss_bins_llmk_style_specialized` profiles now force `--train-loss-every-steps 1` on both baseline and candidate commands before any timing result is trusted. A corrected 3-step, 2-sample dedicated RTX 5090 `lm_head_loss_bins` recheck proved the route (`lm_head_classifier_loss_bin_launch_count: 0 -> 48`, loss strategy changed to fused loss bins) and improved train-loop wall (`0.986539x`), LM-head backward (`0.908492x`), and LM-head CE (`0.540795x`), but still failed strict gates on block backward (`1.019348x`) and MLP projection (`1.000430x`). Keep it diagnostic-only; do not promote this route until the unrelated backward-stage regression is removed or the same-script gate passes cleanly.
  - 2026-06-24 refreshed `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_loss_bins` after the LM-head graph-counter split on the dedicated RTX 5090. The same-script 3-step, 2-sample stage-timed rerun forced `--train-loss-every-steps 1`, proved the route (`lm_head_classifier_loss_bin_launch_count: 0 -> 48`), and now passed the native metric gates at `0.964602x` train-loop wall, `0.977001x` steady-state CUDA-event timing, `0.909318x` `stage.lm_head_backward.total_ms`, and `0.542775x` LM-head CE. The wrapper no longer rejects this named profile; it is the accepted train-loss logging comparison against the older row-loss tail. This does not close normal no-train-loss optimizer-step parity, because that path still skips the logged loss-reduction tail.
  - 2026-06-25 refreshed `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_loss_bins` after the LM-head graph-upload and side-stream diagnostics. The dedicated RTX 5090 3-step, 2-sample stage-timed rerun again forced `--train-loss-every-steps 1`, moved `lm_head_classifier_loss_bin_launch_count: 0 -> 48`, and passed all strict gates at `0.981541x` train-loop wall, `0.982697x` steady-state CUDA-event timing, `1.018809x` tokens/sec, `0.927229x` LM-head backward, `0.999905x` block backward, and `0.995141x` MLP projection backward. Keep the profile accepted for logged-loss regression checks, but do not count it as closing normal no-train-loss parity; the open implementation target remains the true fused classifier/dHidden/dWeight kernel.
  - 2026-06-24 added the combined diagnostic profile `NFN_SM120_NATIVE_CANDIDATE_PROFILE=lm_head_loss_bins_bf16_workspace_prewarm`. It compares loss-bin train-loss logging plus BF16 workspace prewarm against both routes disabled and forces `--train-loss-every-steps 1` so the loss-bin route actually executes. The dedicated RTX 5090 CUDA 13.3 3-step, 2-sample gate passed for the logged-loss route (`0.960297x` train-loop wall, `0.976567x` steady-state CUDA-event wall, `1.041409x` tokens/sec, `0.909381x` LM-head backward, loss-bin launches `0 -> 48`), but keep it rejected by default because normal no-train-loss throughput does not execute the loss-bin tail and forced train-loss logging adds host loss copies.
  - 2026-06-22 extended the existing shape-gated TK dWeight diagnostic into the BF16/BF16 dWeight+bias ABI and added the named profile `NFN_SM120_NATIVE_CANDIDATE_PROFILE=mlp_proj_tk_dweight_65536` for `NFN_NATIVE_LINEAR_TK_DWEIGHT_ENABLE_SHAPE=3072,768,65536,N,T`. The route computes dWeight through the TK bridge and then runs the existing Tile bias reducer. The one-step dedicated RTX 5090 probe proved the route active (`linear_tk_dweight_gemm_count: 0 -> 96`) but rejected it at `1.019937x` train-loop wall, `1.041530x` block backward, `1.112576x` MLP projection, and `1.229754x` MLP projection dWeight+bias. Keep it diagnostic-only; the current TK dWeight bridge is slower than cuBLASLt BGRADB for this block bucket.
  - 2026-06-22 strengthened the same `mlp_proj_tk_dweight_65536` rejection after the row-loss default promotion and JSON reporting fix. The CUDA 13.3 dedicated RTX 5090 3-step, 2-sample same-script gate moved `linear_tk_dweight_gemm_count` from `0` to `288`, reduced cuBLASLt GEMMs from `2064` to `1776`, and reported `block_backward_mlp_proj_tk_dweight_enabled: true`, but failed at `1.017138x` train-loop wall, `1.037585x` block backward, `1.148117x` MLP projection, and `1.313764x` MLP projection dWeight+bias. Keep this route diagnostic-only; the next block-backward work needs a better fused/cooperative kernel or GEMM route, not the current TK dWeight bridge.
  - 2026-06-23 rechecked `mlp_proj_tk_dweight_65536` after the LM-head prepack default flip and marked the wrapper profile rejected by default. The CUDA 13.3 dedicated RTX 5090 2-step, 2-sample same-script gate moved `linear_tk_dweight_gemm_count` from `0` to `192` and reduced cuBLASLt GEMMs from `1376` to `1184`, but regressed train-loop wall to `1.019797x` and tokens/sec to `0.980596x`. Use `NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1` only for intentional reruns.
  - 2026-06-22 earlier check: `NFN_NATIVE_LINEAR_TK_DWEIGHT_ENABLE_SHAPE=3072,768,65536,N,T` did not route through the hot block dWeight+bias ABI (`linear_tk_dweight_gemm_count` stayed `0`) because the TK dWeight route only covered the no-bias BF16/BF16 dWeight ABI used by LM-head. The follow-up dWeight+bias wrapper above supersedes that blocker; the profile still must be benchmark-rejected or accepted before any default change.
  - 2026-06-22 rechecked grouped cuBLASLt support after CUDA 13.3: `cublaslt_grouped_probe` reported grouped layout status `0` but grouped matmul execution status `15`. A direct attempt to fold the classic cuBLAS grouped BF16 probe into that profile reproduced CUDA error `700` on the following model arena allocation, so grouped block-backward GEMMs are still blocked on this workstation until grouped execution, not just grouped descriptor creation, passes without poisoning the context.
  - 2026-06-23 reran the non-poisoning `cublaslt_grouped_probe` after the CUDA reinstall: cuBLASLt grouped layout still reports status `0`, grouped matmul execution still reports status `15`, and the classic cuBLAS grouped BF16 probe remains omitted. The wrapper now treats this profile as capability-only by keeping the route-change gate but skipping automatic `setup_wall_ms` ratio gates, so startup timing noise does not make readiness checks fail for the wrong reason.
  - 2026-06-23 hardened the BF16-only MLP projection dInput+dGELU raw ABI used
    by the default BF16 MLP grad handoff. The SM120 TK fused route remains the
    first choice, but if a non-default build or shape misses that route,
    `nfn_native_tile_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_only_float32`
    now falls back to BF16-output GEMM plus an in-place BF16-bits dGELU kernel
    instead of leaving the handoff buffer unwritten. This is correctness
    hardening for fallback/candidate builds, not a promoted throughput fix for
    the remaining SM120 parity gap.
  - 2026-06-23 fixed the SM120 paired wrapper's compile-time dGELU profiles so
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=tk_dgelu_dinput` and
    `tk_dgelu_approx_tanh` compare the current fused TK route against the older
    unfused trainer route in one script. The baseline side now forces
    `NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU=0`, the candidate side forces
    `NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU=1`, and the candidate Tile ops shared
    object still rebuilds with the SM120 dGELU compile flag. This is benchmark
    hygiene for future kernel work; it prevents another no-op default-vs-default
    dGELU run before any promotion decision. A one-step dedicated RTX 5090
    route smoke with ratio gates disabled changed
    `linear_tk_dgelu_dinput_gemm_count` from `0` to `96`,
    `block_backward_dinput_tk_gemm_count` from `0` to `96`, and
    `block_backward_dinput_cublaslt_gemm_count` from `384` to `288`, with the
    native route-change gate passing.
  - 2026-06-24 rechecked the same compile-time dGELU profile after the linked
    SM120 baseline had already absorbed the fused TK dInput+dGELU route. The
    baseline now reports `linear_tk_dgelu_dinput_gemm_count=288`, the generated
    candidate reports the same route counters, and the native route-change gate
    fails. Keep `tk_dgelu_dinput` and `tk_dgelu_approx_tanh` rejected/no-op
    historical diagnostics unless a future compile flag proves a distinct route.
  - 2026-06-24 rechecked `NFN_SM120_NATIVE_CANDIDATE_PROFILE=tk_forward_no_n96`
    after the CUDA reinstall. The wrapper built the `-DLLMK_SM120_FORWARD_N96=0`
    Tile ops candidate and ran a short dedicated RTX 5090 paired gate, but no
    route counters, strategy strings, linear shape stats, or cuBLASLt plan-cache
    entries changed. The route-change gate failed, and hot-stage gates regressed
    at `stage.lm_head_backward.total_ms=1.001484x` and
    `stage.block_backward.mlp_proj.total_ms=1.001994x`. Keep it rejected by
    default; rerun only as a historical compile-flag diagnostic.
  - 2026-06-23 added the clearer
    `NFN_SM120_NATIVE_DISABLE_METRIC_RATIO_GATES=1` alias, plus candidate,
    parity, and shared variants, for route-proof-only candidate smokes that
    should keep native route-change checks but skip automatic timing-ratio
    gates. The older `NFN_SM120_NATIVE_AUTO_DISABLE_METRIC_RATIO_GATES=1`
    spelling remains supported.
  - 2026-06-24 made explicitly allowed rejected-profile reruns route-proof
    smokes by default. `NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1`
    still requires intentional opt-in, still prints the paired timing ratios,
    and still keeps the native route-change gate, but it no longer fails solely
    on automatic strict promotion-ratio gates unless
    `NFN_SM120_NATIVE_ENFORCE_REJECTED_CANDIDATE_RATIO_GATES=1` is set.
  - 2026-06-25 rechecked the native GPT path after the CUDA Toolkit 13.3 WSL
    reinstall on the display-disabled RTX 5090. The native-vs-native
    five-step/two-sample stage-timed gate passed with
    `train_loop_wall_ms_per_step=0.998306x` and steady-state CUDA-event timing
    at `1.001334x`, with no route or strategy changes. The no-Torch/default
    dependency gate also passed, and `tests/test_native_gpt2.py` passed
    (`90 passed, 1 skipped in 375.92s`). The external llm.kittens reference
    comparison still shows a real parity gap: default NeuralFn measured
    `1.029364x` llm.kittens train-loop wall and `1.010067x` steady-state
    CUDA-event timing. Intentionally rerunning the rejected
    `llmk_sm120_reference_flags` compile-flag bundle narrowed that to
    `1.012296x` wall and `1.008866x` steady-state, but it still did not close
    parity or prove a distinct hot route. Keep the next work focused on a real
    fused/cooperative LM-head or block-backward kernel, not compile-flag
    promotion or graph-editor/Torch/startup hypotheses.
  - 2026-06-25 kept `linear_bias_threads_512` as the optimized default after
    rerunning it on the display-disabled RTX 5090 with CUDA 13.3.33 and the
    corrected shared Tile helper. The same-script 3-step, 2-sample stage-timed
    gate proved the route changed
    (`block_state_layout.linear_backward_bias_threads_per_block: 256 -> 512`)
    and measured `train_loop_wall_ms_per_step=0.992990x`,
    `train_loop_cuda_event_steady_state_wall_ms_per_step=0.998950x`,
    `train_tokens_per_second=1.007496x`,
    `stage.block_backward.total_ms=0.989262x`,
    `stage.block_backward.mlp_proj.dweight_bias.total_ms=0.984430x`, and
    `stage.block_backward.mlp_fc.dweight_bias.total_ms=0.972707x`. The
    Tile-CUDA bias reducer default remains 512 threads while retaining
    `NFN_NATIVE_GPT_LINEAR_BACKWARD_BIAS_THREADS=256` for regression checks
    against the old route.
  - 2026-06-25 refreshed llm.kittens parity after the now-superseded
    512-thread bias reducer run. The same-script 5-step, 2-sample stage-timed
    run on the
    display-disabled RTX 5090 selected the dedicated GPU with zero compute
    processes before and after each sample, but NeuralFn still missed parity:
    `train_loop_wall_ms_per_step=1.011475x`,
    `train_loop_cuda_event_steady_state_wall_ms_per_step=1.012495x`, and
    `train_tokens_per_second=0.983134x` versus llm.kittens. Candidate JSON
    confirmed `block_state_layout.linear_backward_bias_threads_per_block: 512`,
    zero train-loss host D2H copies, and no graph-editor/Torch data path. The
    hottest buckets remained `stage.block_backward.total_ms=6566.650 ms`,
    `stage.train.model_forward.total_ms=3196.490 ms`, and
    `stage.lm_head_backward.total_ms=2864.970 ms`, with 80 LM-head CUDA Graph
    replays and a three-node CE/dHidden/dWeight graph body. The next parity
    slice remains a true fused LM-head classifier-backward Tile kernel or a real
    block-backward kernel, not another startup or route-toggle rerun.
  - 2026-06-25 rechecked LM-head CUDA Graph prewarm after the CUDA Toolkit
    13.3 reinstall. Earlier short gates were mixed, but the later graph-only
    opt-out-versus-default-on rerun passed the configured same-script gates:
    train-loop wall `0.970282x`, steady-state CUDA-event timing `1.001894x`,
    LM-head backward `0.968319x`, block backward `0.956792x`, and MLP projection
    backward `0.911989x`. `NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_GRAPH_PREWARM`
    now defaults to `1` for real training; set it to `0` only for lazy-capture
    bisection. This is a default graph-wrapper improvement, not a
    parity-closing fused kernel implementation.
  - Historical 2026-06-25 parity note from the lazy-capture default window
    before graph prewarm promotion. The same-script 3-step, 1-sample stage-timed run selected the
    dedicated RTX 5090 with zero compute processes before and after the
    sample. NeuralFn beat llm.kittens on average train-loop wall
    (`0.964922x`) and tokens/sec (`1.011635x`), but the strict steady-state
    CUDA-event gate still failed at `1.013321x`. Runtime JSON confirmed
    `lm_head_cooperative_backward_graph_prewarm_requested=false`,
    `lm_head_fused_graph_capture_success_count=3`,
    `lm_head_fused_graph_thread_cache_hit_count=45`, and 48 graph replays
    through the three-node CE/dHidden/dWeight body. The hot buckets remained
    `stage.block_backward.total_ms=4085.320 ms`,
    `stage.train.model_forward.total_ms=1983.550 ms`, and
    `stage.lm_head_backward.total_ms=1774.670 ms`, with MLP projection
    backward dominated by dWeight+bias (`748.079 ms`) and dInput
    (`509.308 ms`). Keep targeting a real fused/cooperative LM-head kernel or
    block-backward Tile kernel; the graph cache is only a launch/capture
    wrapper.
  - 2026-06-25 reran parity after the corrected 512-thread bias reducer default
    and lazy LM-head graph prewarm change. The same-script 3-step, 1-sample
    stage-timed run again selected the display-disabled dedicated RTX 5090 and
    confirmed zero compute processes before and after the sample. NeuralFn beat
    llm.kittens on average train-loop wall (`0.986263x`), CUDA-event wall
    (`0.986115x`), first-step CUDA-event wall (`0.942728x`), and tokens/sec
    (`1.001695x`), but the strict steady-state CUDA-event gate still failed at
    `1.013543x`. Runtime JSON still reported zero train-loss host D2H copies,
    `lm_head_classifier_backward_path_class=diagnostic-cuda-graph-wrapper`,
    `lm_head_cooperative_backward_graph_prewarm_requested=false`, three graph
    captures, 45 thread-cache hits, and 48 graph replays through the CE,
    dHidden, and dWeight graph body. The hottest buckets remained
    `stage.block_backward.total_ms=3948.050 ms`,
    `stage.train.model_forward.total_ms=1987.080 ms`,
    `stage.lm_head_backward.total_ms=1775.780 ms`, and MLP projection dWeight
    plus dInput work. Keep the next implementation slice on steady-state
    LM-head/block-backward kernels; the no-Torch/no-graph-editor constraint is
    already satisfied by this run.
  - 2026-06-25 rechecked `bf16_attention_grad_out` after the 512-thread bias
    reducer became the default. The same-script 3-step, 2-sample stage-timed
    run proved the BF16 attention grad-out handoff route by moving 288 block
    dInput GEMMs from cuBLASLt to BF16, improved steady-state CUDA-event timing
    to `0.997577x` and attention to-QKV to `0.978000x`, but still rejected
    default promotion because train-loop wall regressed to `1.002882x`,
    tokens/sec to `0.997149x`, block backward to `1.005784x`, MLP FC
    dWeight+bias to `1.062723x`, and attention projection to `1.025902x`.
  - 2026-06-25 reran the 3-step, 1-sample llm.kittens parity check after the
    candidate-profile help fix to make sure the harness-only change did not
    hide the remaining kernel gap. The dedicated display-disabled RTX 5090 was
    idle before and after the sample, and native JSON still confirmed zero
    train-loss host D2H copies plus the no-Torch native path. NeuralFn failed
    strict parity at `1.034824x` train-loop wall and `1.012863x`
    steady-state CUDA-event step time versus llm.kittens. The current LM-head
    path remained `diagnostic-cuda-graph-wrapper` with three captures, 45
    thread-cache hits, and 48 graph replays across the CE, dHidden, and dWeight
    graph body. Hot buckets were unchanged: `stage.block_backward.total_ms`
    `4038.430 ms`, `stage.train.model_forward.total_ms` `1992.760 ms`,
    `stage.lm_head_backward.total_ms` `1774.240 ms`, and
    `stage.lm_head_backward.cooperative.total_ms` `1248.000 ms`. Keep the next
    implementation slice on a true fused/cooperative LM-head classifier-backward
    Tile kernel or a real block-backward GEMM/TK route change.
  - 2026-06-25 reran the rejected `lm_head_row_chunk_49152` route after the
    CUDA Toolkit 13.3 reinstall with
    `NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1`. The route proof
    changed `lm_head_classifier_last_rows` from `32768` to `16384`, but the
    same-script 3-step stage-timed gate rejected it decisively:
    `train_loop_wall_ms_per_step=1.272313x`,
    `train_loop_cuda_event_steady_state_wall_ms_per_step=1.257891x`, and
    `train_tokens_per_second=0.785970x`. LM-head timing stayed flat
    (`stage.lm_head_backward.total_ms=1.000772x`), while block backward
    regressed (`stage.block_backward.total_ms=1.532127x`,
    `stage.block_backward.attn_sdpa.to_qkv.total_ms=3.363801x`,
    `stage.block_backward.attn_proj.total_ms=1.342615x`). Keep the default
    two-`32768`-row LM-head route; larger row chunks are not a viable parity
    path on the current dedicated RTX 5090/CUDA 13.3 setup.
  - 2026-06-25 reran `lm_head_overlap_last_dweight` after cooperative graph
    prewarm became the default. A plain profile run only changed the requested
    flag (`lm_head_overlap_last_dweight_requested=true`) and left
    `lm_head_overlap_last_dweight_enabled=false`, so the wrapper profile now
    disables `NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD` on the candidate to
    force the real side-stream route. The route-enabled same-script 3-step,
    2-sample stage-timed run proved `lm_head_overlap_last_dweight_enabled=true`
    with 24 queue/sync events, but rejected the route at
    `train_loop_wall_ms_per_step=1.020764x`,
    `train_loop_cuda_event_steady_state_wall_ms_per_step=1.002042x`,
    `train_tokens_per_second=0.979861x`, and
    `stage.lm_head_backward.total_ms=1.050532x` versus the default cooperative
    CUDA Graph wrapper. Keep the next implementation target on a true fused
    LM-head body or block-backward kernel, not side-stream scheduling.
  - 2026-06-27 post-CUDA-reinstall 10-step recheck on the dedicated
    display-disabled RTX 5090 left the current native route unchanged and
    narrowed the open target. `llmk_sm120_reference_flags` rebuilt with its
    macro bundle but regressed current native train-loop wall to `1.004713x`
    and stayed `1.001757x` slower than the llm.kittens reference. Plain
    no-stage parity still failed at `1.006483x` train-loop wall and
    `0.992972x` tokens/sec versus llm.kittens. `cublaslt_min_waves` also
    failed, regressing native train-loop wall to `1.010224x`. All runs selected
    GPU 0, observed no compute processes before/after, and passed the native
    runtime contract (`graph_editor_tensor_flow=false`, `torch_required=false`).
    Do not promote these profiles; the remaining implementation work is the
    production true-fused LM-head classifier-backward Tile kernel replacing the
    current three-node diagnostic CUDA Graph wrapper.
  - 2026-06-28 refreshed the current no-stage parity and min-waves evidence
    after the latest native defaults. The canonical 5-step, 2-sample no-stage
    parity wrapper passed on the dedicated display-disabled RTX 5090 with
    NeuralFn at `1.000345x` llm.kittens train-loop wall time and `0.999421x`
    tokens/sec, plus the native runtime contract still clean
    (`graph_editor_tensor_flow=false`, `torch_required=false`,
    `optimized_kernel_contract_passed=true`, `train_loss_host_d2h_count=0`).
    The same-script `cublaslt_min_waves` rerun regressed current native
    train-loop wall to `1.005895x` and tokens/sec to `0.994138x`, so heuristic
    policy churn remains rejected; the next useful performance work is a real
    fused LM-head or block-backward kernel route.
  - 2026-06-27 added an explicit native dense-GPT fast-startup mode instead of
    changing the long-training defaults. `NFN_NATIVE_GPT_FAST_STARTUP=1`
    (aliases: `NFN_NATIVE_GPT2_FAST_STARTUP=1`,
    `NFN_TILE_CUDA_FAST_STARTUP=1`) flips the setup-prewarm defaults so TK QKV
    first-use prewarm and LM-head CUDA Graph prewarm are skipped unless their
    explicit prewarm env vars force them back on. Runtime JSON reports
    `native_fast_startup_requested` and `native_fast_startup_prewarm_policy`;
    `NFN_SM120_NATIVE_CANDIDATE_PROFILE=fast_startup` compares the startup-only
    tradeoff in the same selected-GPU harness. Keep this opt-in for smoke tests
    and low-latency startup checks; the default training path keeps throughput
    prewarms on. The dedicated RTX 5090 3-sample startup-only gate passed at
    `0.736103x` setup wall and proved the strategy-value change in the saved
    paired JSON.
  - 2026-06-28 reran the rejected `token_weight_padded_init` candidate after
    confirming CUDA UMD 13.3 on the dedicated display-disabled RTX 5090 and no
    active compute processes. The explicit rejected-profile rerun used
    `NFN_SM120_NATIVE_ALLOW_REJECTED_CANDIDATE_PROFILE=1` with a 3-step,
    2-sample same-script gate. It proved the route
    (`token_weight_bf16_padding_memset_count: 1 -> 0`) and improved setup wall
    time to `0.965032x`, but still failed promotion against the llm.kittens
    reference: `train_loop_wall_ms_per_step=1.000427x`,
    `train_loop_cuda_event_first_step_wall_ms_per_step=1.000846x`,
    `train_loop_cuda_event_steady_state_wall_ms_per_step=1.000203x`, and
    `train_tokens_per_second=0.999952x`. Superseded by the later 2026-06-28
    5-step, 3-sample parity promotion that made the padded initializer default;
    this earlier result remains useful only as evidence that startup-only wins
    were not enough to promote the route before the full parity rerun.
  - 2026-06-28 refreshed the current default after promoting fused padded
    token-weight initialization. The canonical 5-step, 3-sample no-stage parity
    wrapper selected the dedicated RTX 5090, observed zero selected-GPU compute
    processes before each sample, and passed the runtime contract with
    `graph_editor_tensor_flow=false`, `torch_required=false`,
    `optimized_kernel_contract_passed=true`, and `train_loss_host_d2h_count=0`.
    Median NeuralFn over llm.kittens was `0.999041x` train-loop wall,
    `0.999342x` steady-state CUDA-event step time, and `1.001718x`
    tokens/sec. The remaining setup split is now explicit:
    `setup_wall_ms=714.306 ms` median, float arena materialization
    `181.658 ms`, uint16 arena materialization `125.478 ms`, and token-weight
    init `151.345 ms`. The LM-head path is still
    `diagnostic-cuda-graph-wrapper` with 120 graph replays over the three
    CE/dHidden/dWeight graph-body nodes, so the next implementation work remains
    reducing real arena materialization or replacing that wrapper with a strict
    true-fused Tile classifier-backward kernel.
  - 2026-06-27 added `NFN_SM120_NATIVE_CANDIDATE_PROFILE=fast_startup_full` to
    keep fast-startup default decisions honest. The 5-step, 2-sample dedicated
    RTX 5090 full-training probe improved setup wall time to `0.655522x`, but
    first-step work moved into the training loop: train-loop wall regressed to
    `1.017654x`, first-step CUDA-event time to `1.086326x`, tokens/sec to
    `0.982655x`, and candidate-over-llm.kittens train-loop wall to
    `1.010462x`. Keep `NFN_NATIVE_GPT_FAST_STARTUP=1` for
    startup-only/preflight workflows unless longer runs prove the first-step
    cost is amortized.
