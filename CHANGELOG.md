# Changelog

`README.md` captures the current product and setup story. This file captures the more detailed history behind meaningful changes, including migration notes and verification.

Future updates should append new entries here rather than replacing older notes.

## Unreleased

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
