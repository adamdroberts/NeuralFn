#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_BIN="${NFN_LM_HEAD_BACKWARD_BENCH_BIN:-${ROOT_DIR}/build/lm_head_backward_bench}"
TILE_OPS_LIB="${NFN_NATIVE_TILE_OPS_LIB:-${ROOT_DIR}/build/libnfn_native_train_tile_ops.so}"
JSON_OUT="${NFN_LM_HEAD_BACKWARD_JSON_OUT:-/tmp/nfn_lm_head_backward_bench.json}"
HIDDEN_DIM="${NFN_LM_HEAD_BACKWARD_HIDDEN_DIM:-768}"
VOCAB="${NFN_LM_HEAD_BACKWARD_VOCAB:-50257}"
ROW_STRIDE="${NFN_LM_HEAD_BACKWARD_ROW_STRIDE:-50304}"
CUDA_VISIBLE_DEVICES_VALUE="${NFN_LM_HEAD_BACKWARD_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-dedicated}}"
CUDA_DEVICE_RAW="${NFN_LM_HEAD_BACKWARD_CUDA_DEVICE:-auto}"
REQUIRE_IDLE_SELECTED_GPU="${NFN_LM_HEAD_BACKWARD_REQUIRE_IDLE_SELECTED_GPU:-1}"
MAX_SELECTED_GPU_UTILIZATION_PCT="${NFN_LM_HEAD_BACKWARD_MAX_SELECTED_GPU_UTILIZATION_PCT:-15}"
SELECTED_GPU_UTILIZATION_RETRIES="${NFN_LM_HEAD_BACKWARD_SELECTED_GPU_UTILIZATION_RETRIES:-3}"
SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS="${NFN_LM_HEAD_BACKWARD_SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS:-0.25}"
ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE="${NFN_LM_HEAD_BACKWARD_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE:-1}"
GPU_BENCHMARK_LOCK="${NFN_LM_HEAD_BACKWARD_GPU_BENCHMARK_LOCK:-1}"
GPU_BENCHMARK_LOCK_TIMEOUT_SECONDS="${NFN_LM_HEAD_BACKWARD_GPU_BENCHMARK_LOCK_TIMEOUT_SECONDS:-0}"
BASELINE_SYMBOL_OVERRIDE="${NFN_LM_HEAD_BACKWARD_BASELINE_SYMBOL:-}"
CANDIDATE_SYMBOL_OVERRIDE="${NFN_LM_HEAD_BACKWARD_CANDIDATE_SYMBOL:-}"
PROFILE="${NFN_LM_HEAD_BACKWARD_PROFILE:-smoke}"
DEFAULT_BASELINE_SYMBOL="nfn_native_tile_lm_head_classifier_backward_cooperative_bf16_u16"
DEFAULT_CANDIDATE_SYMBOL="nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16"
REJECTED_PROFILE=""
REJECTED_REASON=""
FORCE_REBUILD_TILE_OPS=0
PROFILE_CHOICES="smoke, trainer-chunk, trainer-chunk-serial-graph-body, trainer-chunk-strict, trainer-chunk-true-fused, trainer-chunk-true-fused-tile16, trainer-chunk-true-fused-tile16-wmma, trainer-chunk-true-fused-tile16-wmma-warp32, trainer-chunk-true-fused-tile16-wmma-exp2-ce, trainer-chunk-true-fused-tile24, trainer-chunk-true-fused-tile8, trainer-chunk-true-fused-tile4, true-fused-cooperative-smoke, trainer-chunk-cublaslt, trainer-row-loss, trainer-row-loss-cublaslt, or trainer-loss-bins"

case "${PROFILE}" in
  smoke)
    DEFAULT_ROWS=2048
    DEFAULT_ITERATIONS=5
    DEFAULT_WARMUP=2
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=0
    ;;
  trainer-chunk|trainer_chunk)
    DEFAULT_ROWS=28672
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=2
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=1
    DEFAULT_REQUIRE_TRUE_FUSED=0
    DEFAULT_REQUIRE_GRAPH_BODY_TILE=1
    ;;
  trainer-chunk-serial-graph-body|trainer_chunk_serial_graph_body|serial-graph-body-trainer-chunk|serial_graph_body_trainer_chunk)
    DEFAULT_ROWS=28672
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=2
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=1
    DEFAULT_REQUIRE_TRUE_FUSED=0
    DEFAULT_REQUIRE_GRAPH_BODY_TILE=1
    DEFAULT_REQUIRE_GRAPH_BODY_SERIAL=1
    REJECTED_PROFILE="${PROFILE}"
    REJECTED_REASON="Production-shape focused serial graph-body LM-head diagnostic. It forces NFN_TILE_CUDA_LM_HEAD_GRAPH_BODY_SERIAL=1 so the CUDA Graph wrapper captures CE, dHidden, and dWeight on the caller stream instead of the default fork-join cooperative streams. Keep rejected as a negative-control profile; production candidates must beat the default concurrent graph-body route, not this serial route."
    export NFN_TILE_CUDA_LM_HEAD_GRAPH_BODY_SERIAL="${NFN_TILE_CUDA_LM_HEAD_GRAPH_BODY_SERIAL:-1}"
    ;;
  trainer-chunk-strict|trainer_chunk_strict|strict-trainer-chunk|strict_trainer_chunk)
    DEFAULT_ROWS=28672
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=2
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=1
    DEFAULT_REQUIRE_TRUE_FUSED=1
    ;;
  trainer-chunk-true-fused|trainer_chunk_true_fused|true-fused-trainer-chunk|true_fused_trainer_chunk)
    DEFAULT_ROWS=28672
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=2
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=1
    DEFAULT_REQUIRE_TRUE_FUSED=1
    DEFAULT_MAX_RATIO=1.000
    DEFAULT_MAX_REFERENCE_RATIO=1.000
    DEFAULT_MAX_CUBLASLT_REFERENCE_RATIO=1.000
    REJECTED_PROFILE="${PROFILE}"
    REJECTED_REASON="Production-shape focused strict true-fused LM-head profile. It forces NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE=1 and NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION=1 so the focused trainer-chunk microbench measures the cooperative single-kernel CE+dHidden+dWeight body. CUDA 13.3.33 dedicated RTX 5090 2026-06-28 post-reinstall one-iteration focused rerun at the current 28672-row trainer chunk proved strict-true-fused-tile-kernel but rejected the default 32x32 body at 32.326054x candidate/current-wrapper and 22.231452x candidate/reference-summed time. The strict body took 690.838257 ms and remained 659.763442 ms slower than the reference CE+dHidden+dWeight components. Keep rejected until this focused gate proves candidate/current-wrapper and candidate/reference parity."
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE:-1}"
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION:-1}"
    ;;
  trainer-chunk-true-fused-tile16|trainer_chunk_true_fused_tile16|true-fused-trainer-chunk-tile16|true_fused_trainer_chunk_tile16)
    DEFAULT_ROWS=28672
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=2
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=1
    DEFAULT_REQUIRE_TRUE_FUSED=1
    DEFAULT_MAX_RATIO=1.000
    DEFAULT_MAX_REFERENCE_RATIO=1.000
    DEFAULT_MAX_CUBLASLT_REFERENCE_RATIO=1.000
    REJECTED_PROFILE="${PROFILE}"
    REJECTED_REASON="Production-shape focused strict true-fused LM-head tile16 profile. It builds the candidate Tile ops library with NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=16, forces CE threads to 256, and measures the cooperative single-kernel CE+dHidden+dWeight body. CUDA 13.3 dedicated RTX 5090 2026-06-27 one-iteration probe proved strict-true-fused-tile-kernel but rejected it at 6.187603x candidate/baseline and 21.078761x candidate/reference-summed time. Keep rejected until this focused gate proves candidate/current-wrapper and candidate/reference parity."
    export NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="${NFN_TILE_CUDA_EXTRA_NVCC_FLAGS:+${NFN_TILE_CUDA_EXTRA_NVCC_FLAGS} }-DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=16"
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE:-1}"
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION:-1}"
    export NFN_TILE_CUDA_CE_BF16_THREADS="${NFN_TILE_CUDA_CE_BF16_THREADS:-256}"
    if [[ -z "${NFN_NATIVE_TILE_OPS_LIB+x}" ]]; then
      TILE_OPS_LIB="${TMPDIR:-/tmp}/nfn_lm_head_backward_tile_ops_true_fused_tile16.so"
      FORCE_REBUILD_TILE_OPS=1
    fi
    ;;
  trainer-chunk-true-fused-tile16-wmma|trainer_chunk_true_fused_tile16_wmma|true-fused-trainer-chunk-tile16-wmma|true_fused_trainer_chunk_tile16_wmma)
    DEFAULT_ROWS=28672
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=2
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=1
    DEFAULT_REQUIRE_TRUE_FUSED=1
    DEFAULT_MAX_RATIO=1.000
    DEFAULT_MAX_REFERENCE_RATIO=1.000
    DEFAULT_MAX_CUBLASLT_REFERENCE_RATIO=1.000
    REJECTED_PROFILE="${PROFILE}"
    REJECTED_REASON="Production-shape focused strict true-fused LM-head tile16 WMMA profile. It builds the candidate Tile ops library with NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=16 and NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_WMMA=1, forces CE threads to 256, and measures the cooperative single-kernel CE+dHidden+dWeight body. CUDA 13.3.33 dedicated RTX 5090 2026-06-29 one-iteration focused probe proved strict-true-fused-tile-kernel with candidate_symbol_abi_implementation_class=wmma-bf16-cooperative-tile-experimental, improving the previous scalar strict body but still rejecting it at 2.585996x candidate/current-wrapper and 7.777548x candidate/reference-summed time. Keep rejected until this focused gate proves candidate/current-wrapper and candidate/reference parity."
    export NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="${NFN_TILE_CUDA_EXTRA_NVCC_FLAGS:+${NFN_TILE_CUDA_EXTRA_NVCC_FLAGS} }-DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=16 -DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_WMMA=1"
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE:-1}"
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION:-1}"
    export NFN_TILE_CUDA_CE_BF16_THREADS="${NFN_TILE_CUDA_CE_BF16_THREADS:-256}"
    if [[ -z "${NFN_NATIVE_TILE_OPS_LIB+x}" ]]; then
      TILE_OPS_LIB="${TMPDIR:-/tmp}/nfn_lm_head_backward_tile_ops_true_fused_tile16_wmma.so"
      FORCE_REBUILD_TILE_OPS=1
    fi
    ;;
  trainer-chunk-true-fused-tile16-wmma-warp32|trainer_chunk_true_fused_tile16_wmma_warp32|true-fused-trainer-chunk-tile16-wmma-warp32|true_fused_trainer_chunk_tile16_wmma_warp32)
    DEFAULT_ROWS=28672
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=2
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=1
    DEFAULT_REQUIRE_TRUE_FUSED=1
    DEFAULT_MAX_RATIO=1.000
    DEFAULT_MAX_REFERENCE_RATIO=1.000
    DEFAULT_MAX_CUBLASLT_REFERENCE_RATIO=1.000
    REJECTED_PROFILE="${PROFILE}"
    REJECTED_REASON="Production-shape focused strict true-fused LM-head tile16 WMMA one-warp profile. It builds the candidate Tile ops library with NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=16, NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_WMMA=1, and NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_THREADS=32, then forces CE threads to 32 so the cooperative body launches one warp per block. The 2026-06-29 focused rerun proved the strict one-warp WMMA ABI path but rejected it at 21.559768x candidate/current-wrapper, 15.444372x candidate/reference-summed, and 12.048745x candidate/reference-summed-with-logits time. dHidden and dWeight remain far slower than the reference components, so keep rejected until a different body proves candidate/current-wrapper and candidate/reference parity."
    export NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="${NFN_TILE_CUDA_EXTRA_NVCC_FLAGS:+${NFN_TILE_CUDA_EXTRA_NVCC_FLAGS} }-DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=16 -DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_WMMA=1 -DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_THREADS=32"
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE:-1}"
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION:-1}"
    export NFN_TILE_CUDA_CE_BF16_THREADS="${NFN_TILE_CUDA_CE_BF16_THREADS:-32}"
    if [[ -z "${NFN_NATIVE_TILE_OPS_LIB+x}" ]]; then
      TILE_OPS_LIB="${TMPDIR:-/tmp}/nfn_lm_head_backward_tile_ops_true_fused_tile16_wmma_warp32.so"
      FORCE_REBUILD_TILE_OPS=1
    fi
    ;;
  trainer-chunk-true-fused-tile16-wmma-exp2-ce|trainer_chunk_true_fused_tile16_wmma_exp2_ce|true-fused-trainer-chunk-tile16-wmma-exp2-ce|true_fused_trainer_chunk_tile16_wmma_exp2_ce)
    DEFAULT_ROWS=28672
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=2
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=1
    DEFAULT_REQUIRE_TRUE_FUSED=1
    DEFAULT_MAX_RATIO=1.000
    DEFAULT_MAX_REFERENCE_RATIO=1.000
    DEFAULT_MAX_CUBLASLT_REFERENCE_RATIO=1.000
    REJECTED_PROFILE="${PROFILE}"
    REJECTED_REASON="Production-shape focused strict true-fused LM-head tile16 WMMA plus exp2 CE profile. It builds the candidate Tile ops library with NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=16, NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_WMMA=1, and NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_CE_EXP2=1, forces CE threads to 256, and measures whether the strict body's CE math is worth further optimization before deeper dHidden/dWeight work. CUDA 13.3.33 dedicated RTX 5090 2026-06-29 one-iteration focused rerun proved strict-true-fused-tile-kernel with candidate_symbol_abi_implementation_class=wmma-bf16-cooperative-tile-exp2-ce-experimental, but rejected it at 11.873753x candidate/current-wrapper, 8.379202x candidate/reference-summed, and 6.472768x candidate/reference-summed-with-logits time. The strict body took 278.149017 ms; dHidden and dWeight remained the dominant gaps with 307486024.709804 and 498988900.996078 cycles/block. Keep rejected until the focused gate proves candidate/current-wrapper and candidate/reference parity."
    export NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="${NFN_TILE_CUDA_EXTRA_NVCC_FLAGS:+${NFN_TILE_CUDA_EXTRA_NVCC_FLAGS} }-DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=16 -DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_WMMA=1 -DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_CE_EXP2=1"
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE:-1}"
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION:-1}"
    export NFN_TILE_CUDA_CE_BF16_THREADS="${NFN_TILE_CUDA_CE_BF16_THREADS:-256}"
    if [[ -z "${NFN_NATIVE_TILE_OPS_LIB+x}" ]]; then
      TILE_OPS_LIB="${TMPDIR:-/tmp}/nfn_lm_head_backward_tile_ops_true_fused_tile16_wmma_exp2_ce.so"
      FORCE_REBUILD_TILE_OPS=1
    fi
    ;;
  trainer-chunk-true-fused-tile24|trainer_chunk_true_fused_tile24|true-fused-trainer-chunk-tile24|true_fused_trainer_chunk_tile24)
    DEFAULT_ROWS=28672
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=2
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=1
    DEFAULT_REQUIRE_TRUE_FUSED=1
    DEFAULT_MAX_RATIO=1.000
    DEFAULT_MAX_REFERENCE_RATIO=1.000
    DEFAULT_MAX_CUBLASLT_REFERENCE_RATIO=1.000
    REJECTED_PROFILE="${PROFILE}"
    REJECTED_REASON="Production-shape focused strict true-fused LM-head tile24 profile. It builds the candidate Tile ops library with NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=24, forces CE threads to 576, and measures the cooperative single-kernel CE+dHidden+dWeight body. CUDA 13.3.33 dedicated RTX 5090 2026-06-28 one-iteration focused probe proved strict-true-fused-tile-kernel but rejected it at 6.266142x candidate/baseline and 21.764091x candidate/reference-summed time, with the strict body still 679.228962 ms slower than the reference CE+dHidden+dWeight components. Keep rejected until this focused gate proves candidate/current-wrapper and candidate/reference parity."
    export NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="${NFN_TILE_CUDA_EXTRA_NVCC_FLAGS:+${NFN_TILE_CUDA_EXTRA_NVCC_FLAGS} }-DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=24"
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE:-1}"
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION:-1}"
    export NFN_TILE_CUDA_CE_BF16_THREADS="${NFN_TILE_CUDA_CE_BF16_THREADS:-576}"
    if [[ -z "${NFN_NATIVE_TILE_OPS_LIB+x}" ]]; then
      TILE_OPS_LIB="${TMPDIR:-/tmp}/nfn_lm_head_backward_tile_ops_true_fused_tile24.so"
      FORCE_REBUILD_TILE_OPS=1
    fi
    ;;
  trainer-chunk-true-fused-tile8|trainer_chunk_true_fused_tile8|true-fused-trainer-chunk-tile8|true_fused_trainer_chunk_tile8)
    DEFAULT_ROWS=28672
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=2
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=1
    DEFAULT_REQUIRE_TRUE_FUSED=1
    DEFAULT_MAX_RATIO=1.000
    DEFAULT_MAX_REFERENCE_RATIO=1.000
    DEFAULT_MAX_CUBLASLT_REFERENCE_RATIO=1.000
    REJECTED_PROFILE="${PROFILE}"
    REJECTED_REASON="Production-shape focused strict true-fused LM-head tile8 profile. It builds the candidate Tile ops library with NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=8, forces CE threads to 64, and measures the cooperative single-kernel CE+dHidden+dWeight body. CUDA 13.3 dedicated RTX 5090 2026-06-27 one-iteration probe proved strict-true-fused-tile-kernel but rejected it at 8.412627x candidate/baseline and 26.425985x candidate/reference-summed time. Keep rejected until this focused gate proves candidate/current-wrapper and candidate/reference parity."
    export NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="${NFN_TILE_CUDA_EXTRA_NVCC_FLAGS:+${NFN_TILE_CUDA_EXTRA_NVCC_FLAGS} }-DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=8"
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE:-1}"
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION:-1}"
    export NFN_TILE_CUDA_CE_BF16_THREADS="${NFN_TILE_CUDA_CE_BF16_THREADS:-64}"
    if [[ -z "${NFN_NATIVE_TILE_OPS_LIB+x}" ]]; then
      TILE_OPS_LIB="${TMPDIR:-/tmp}/nfn_lm_head_backward_tile_ops_true_fused_tile8.so"
      FORCE_REBUILD_TILE_OPS=1
    fi
    ;;
  trainer-chunk-true-fused-tile4|trainer_chunk_true_fused_tile4|true-fused-trainer-chunk-tile4|true_fused_trainer_chunk_tile4)
    DEFAULT_ROWS=28672
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=2
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=1
    DEFAULT_REQUIRE_TRUE_FUSED=1
    DEFAULT_MAX_RATIO=1.000
    DEFAULT_MAX_REFERENCE_RATIO=1.000
    DEFAULT_MAX_CUBLASLT_REFERENCE_RATIO=1.000
    REJECTED_PROFILE="${PROFILE}"
    REJECTED_REASON="Production-shape focused strict true-fused LM-head tile4 profile. It builds the candidate Tile ops library with NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=4, forces a warp-sized NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_THREADS=32 body plus matching CE threads, and measures the cooperative single-kernel CE+dHidden+dWeight body. CUDA 13.3.33 dedicated RTX 5090 2026-06-28 one-iteration focused rerun proved strict-true-fused-tile-kernel but rejected it at 37.738071x candidate/baseline and 113.697403x candidate/reference-summed time, with the strict body still 4510.827989 ms slower than the reference CE+dHidden+dWeight components. Keep rejected until this focused gate proves candidate/current-wrapper and candidate/reference parity."
    export NFN_TILE_CUDA_EXTRA_NVCC_FLAGS="${NFN_TILE_CUDA_EXTRA_NVCC_FLAGS:+${NFN_TILE_CUDA_EXTRA_NVCC_FLAGS} }-DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=4 -DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_THREADS=32"
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE:-1}"
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION:-1}"
    export NFN_TILE_CUDA_CE_BF16_THREADS="${NFN_TILE_CUDA_CE_BF16_THREADS:-32}"
    if [[ -z "${NFN_NATIVE_TILE_OPS_LIB+x}" ]]; then
      TILE_OPS_LIB="${TMPDIR:-/tmp}/nfn_lm_head_backward_tile_ops_true_fused_tile4.so"
      FORCE_REBUILD_TILE_OPS=1
    fi
    ;;
  true-fused-cooperative-smoke|true_fused_cooperative_smoke|strict-true-fused-smoke|strict_true_fused_smoke)
    DEFAULT_ROWS=4
    DEFAULT_ITERATIONS=1
    DEFAULT_WARMUP=0
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=0
    DEFAULT_REQUIRE_TRUE_FUSED=1
    if [[ -z "${NFN_LM_HEAD_BACKWARD_HIDDEN_DIM+x}" ]]; then
      HIDDEN_DIM=8
    fi
    if [[ -z "${NFN_LM_HEAD_BACKWARD_VOCAB+x}" ]]; then
      VOCAB=16
    fi
    if [[ -z "${NFN_LM_HEAD_BACKWARD_ROW_STRIDE+x}" ]]; then
      ROW_STRIDE=16
    fi
    export NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE="${NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE:-1}"
    ;;
  trainer-chunk-cublaslt|trainer_chunk_cublaslt|trainer-cublaslt|trainer_cublaslt)
    DEFAULT_ROWS=28672
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=2
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=1
    DEFAULT_REQUIRE_TRUE_FUSED=0
    DEFAULT_CANDIDATE_SYMBOL="nfn_native_tile_lm_head_classifier_backward_cooperative_cublaslt_bf16_u16"
    REJECTED_PROFILE="${PROFILE}"
    REJECTED_REASON="CUDA 13.3 dedicated RTX 5090 trainer-chunk evidence rejects this cuBLASLt LM-head route: 37.070129 ms/iter vs 25.271233 ms/iter baseline, candidate/baseline ratio 1.466890."
    ;;
  trainer-row-loss-cublaslt|trainer_row_loss_cublaslt)
    DEFAULT_ROWS=28672
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=2
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=0
    DEFAULT_REQUIRE_TRUE_FUSED=0
    DEFAULT_CANDIDATE_SYMBOL="nfn_native_tile_lm_head_classifier_backward_cooperative_cublaslt_bf16_u16"
    REJECTED_PROFILE="${PROFILE}"
    REJECTED_REASON="CUDA 13.3 dedicated RTX 5090 evidence rejects the cuBLASLt LM-head route; keep this profile for intentional diagnostics only."
    ;;
  trainer-row-loss|trainer_row_loss)
    DEFAULT_ROWS=28672
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=2
    DEFAULT_LOSS_BINS=0
    DEFAULT_NO_LOSS=0
    DEFAULT_REQUIRE_TRUE_FUSED=0
    ;;
  trainer-loss-bins|trainer_loss_bins)
    DEFAULT_ROWS=28672
    DEFAULT_ITERATIONS=3
    DEFAULT_WARMUP=2
    DEFAULT_LOSS_BINS=1024
    DEFAULT_NO_LOSS=0
    DEFAULT_REQUIRE_TRUE_FUSED=0
    ;;
  *)
    echo "Unknown NFN_LM_HEAD_BACKWARD_PROFILE='${PROFILE}' (expected ${PROFILE_CHOICES})" >&2
    exit 2
    ;;
esac

BASELINE_SYMBOL="${BASELINE_SYMBOL_OVERRIDE:-${DEFAULT_BASELINE_SYMBOL}}"
CANDIDATE_SYMBOL="${CANDIDATE_SYMBOL_OVERRIDE:-${DEFAULT_CANDIDATE_SYMBOL}}"

ROWS="${NFN_LM_HEAD_BACKWARD_ROWS:-${DEFAULT_ROWS}}"
ITERATIONS="${NFN_LM_HEAD_BACKWARD_ITERATIONS:-${DEFAULT_ITERATIONS}}"
WARMUP="${NFN_LM_HEAD_BACKWARD_WARMUP:-${DEFAULT_WARMUP}}"
REFERENCE_COMPONENT_WARMUP="${NFN_LM_HEAD_BACKWARD_REFERENCE_COMPONENT_WARMUP:-}"
LOSS_BINS="${NFN_LM_HEAD_BACKWARD_LOSS_BINS:-${DEFAULT_LOSS_BINS}}"
NO_LOSS="${NFN_LM_HEAD_BACKWARD_NO_LOSS:-${DEFAULT_NO_LOSS}}"
MAX_RATIO="${NFN_LM_HEAD_BACKWARD_MAX_RATIO:-${DEFAULT_MAX_RATIO:-}}"
MAX_REFERENCE_RATIO="${NFN_LM_HEAD_BACKWARD_MAX_REFERENCE_RATIO:-${DEFAULT_MAX_REFERENCE_RATIO:-}}"
MAX_REFERENCE_WITH_LOGITS_RATIO="${NFN_LM_HEAD_BACKWARD_MAX_REFERENCE_WITH_LOGITS_RATIO:-}"
MAX_CUBLASLT_REFERENCE_RATIO="${NFN_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_RATIO:-${DEFAULT_MAX_CUBLASLT_REFERENCE_RATIO:-}}"
MAX_CUBLASLT_REFERENCE_WITH_LOGITS_RATIO="${NFN_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_WITH_LOGITS_RATIO:-}"
MAX_REFERENCE_GAP_MS="${NFN_LM_HEAD_BACKWARD_MAX_REFERENCE_GAP_MS:-}"
MAX_REFERENCE_WITH_LOGITS_GAP_MS="${NFN_LM_HEAD_BACKWARD_MAX_REFERENCE_WITH_LOGITS_GAP_MS:-}"
MAX_CUBLASLT_REFERENCE_GAP_MS="${NFN_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_GAP_MS:-}"
MAX_CUBLASLT_REFERENCE_WITH_LOGITS_GAP_MS="${NFN_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_WITH_LOGITS_GAP_MS:-}"
MAX_TRUE_FUSED_CE_CYCLES_PER_BLOCK="${NFN_LM_HEAD_BACKWARD_MAX_TRUE_FUSED_CE_CYCLES_PER_BLOCK:-}"
MAX_TRUE_FUSED_DHIDDEN_CYCLES_PER_BLOCK="${NFN_LM_HEAD_BACKWARD_MAX_TRUE_FUSED_DHIDDEN_CYCLES_PER_BLOCK:-}"
MAX_TRUE_FUSED_DWEIGHT_CYCLES_PER_BLOCK="${NFN_LM_HEAD_BACKWARD_MAX_TRUE_FUSED_DWEIGHT_CYCLES_PER_BLOCK:-}"
REQUIRE_TRUE_FUSED="${NFN_LM_HEAD_BACKWARD_REQUIRE_TRUE_FUSED:-${DEFAULT_REQUIRE_TRUE_FUSED:-0}}"
REQUIRE_GRAPH_BODY_TILE="${NFN_LM_HEAD_BACKWARD_REQUIRE_GRAPH_BODY_TILE:-${DEFAULT_REQUIRE_GRAPH_BODY_TILE:-0}}"
REQUIRE_GRAPH_BODY_SERIAL="${NFN_LM_HEAD_BACKWARD_REQUIRE_GRAPH_BODY_SERIAL:-${DEFAULT_REQUIRE_GRAPH_BODY_SERIAL:-0}}"
CANDIDATE_FIRST="${NFN_LM_HEAD_BACKWARD_CANDIDATE_FIRST:-0}"
DRY_RUN="${NFN_LM_HEAD_BACKWARD_DRY_RUN:-0}"
ALLOW_REJECTED_PROFILE="${NFN_LM_HEAD_BACKWARD_ALLOW_REJECTED_PROFILE:-0}"

if [[ -n "${REJECTED_PROFILE}" ]]; then
  case "${DRY_RUN,,}:${ALLOW_REJECTED_PROFILE,,}" in
    1:*|true:*|yes:*|on:*|*:1|*:true|*:yes|*:on)
      ;;
    *)
      echo "NFN_LM_HEAD_BACKWARD_PROFILE=${REJECTED_PROFILE} is a rejected LM-head candidate profile." >&2
      echo "${REJECTED_REASON}" >&2
      echo "Set NFN_LM_HEAD_BACKWARD_ALLOW_REJECTED_PROFILE=1 to rerun it intentionally, or NFN_LM_HEAD_BACKWARD_DRY_RUN=1 to inspect the command only." >&2
      exit 2
      ;;
  esac
fi

select_auto_cuda_device() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' "0"
    return
  fi
  local query_output
  if ! query_output="$(nvidia-smi --query-gpu=index,display_active,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
    printf '%s\n' "0"
    return
  fi
  printf '%s\n' "${query_output}" | awk -F, '
      {
        idx=$1; display=$2; util=$3;
        gsub(/^[ \t]+|[ \t]+$/, "", idx);
        gsub(/^[ \t]+|[ \t]+$/, "", display);
        gsub(/^[ \t]+|[ \t]+$/, "", util);
        if (first == "") first = idx;
        if (display == "Disabled" && (best == "" || util + 0 < best_util + 0)) {
          best = idx;
          best_util = util;
        }
      }
      END {
        if (best != "") print best;
        else if (first != "") print first;
        else print "0";
      }
    '
}

case "${CUDA_VISIBLE_DEVICES_VALUE,,}" in
  ""|"none"|"off")
    ;;
  "auto"|"dedicated"|"dedicated-auto")
    SELECTED_CUDA_VISIBLE_DEVICE="$(select_auto_cuda_device)"
    export CUDA_VISIBLE_DEVICES="${SELECTED_CUDA_VISIBLE_DEVICE}"
    ;;
  *)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
    ;;
esac

case "${CUDA_DEVICE_RAW,,}" in
  "auto")
    CUDA_DEVICE=0
    ;;
  *)
    CUDA_DEVICE="${CUDA_DEVICE_RAW}"
    ;;
esac

snapshot_selected_gpu_load_json() {
  local phase="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    NFN_GPU_PHASE="${phase}" \
    NFN_GPU_VISIBLE="${CUDA_VISIBLE_DEVICES-}" \
    NFN_GPU_DEVICE="${CUDA_DEVICE}" \
    python -c 'import json, os
print(json.dumps({
    "phase": os.environ["NFN_GPU_PHASE"],
    "available": False,
    "reason": "nvidia-smi-not-found",
    "cuda_visible_devices": os.environ.get("NFN_GPU_VISIBLE", ""),
    "cuda_device": os.environ.get("NFN_GPU_DEVICE", ""),
}, sort_keys=True))
'
    return 0
  fi
  local selected_gpu="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"
  selected_gpu="${selected_gpu%%,*}"
  if [[ -z "${selected_gpu}" ]]; then
    selected_gpu="${CUDA_DEVICE}"
  fi
  local gpu_query=""
  local gpu_status=0
  if gpu_query="$(nvidia-smi -i "${selected_gpu}" --query-gpu=index,name,uuid,display_active,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)"; then
    :
  else
    gpu_status=$?
  fi
  local compute_query=""
  local compute_status=0
  if compute_query="$(nvidia-smi -i "${selected_gpu}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null)"; then
    :
  else
    compute_status=$?
  fi
  NFN_GPU_PHASE="${phase}" \
  NFN_GPU_VISIBLE="${CUDA_VISIBLE_DEVICES-}" \
  NFN_GPU_DEVICE="${CUDA_DEVICE}" \
  NFN_GPU_SELECTED="${selected_gpu}" \
  NFN_GPU_QUERY="${gpu_query}" \
  NFN_GPU_QUERY_STATUS="${gpu_status}" \
  NFN_GPU_COMPUTE="${compute_query}" \
  NFN_GPU_COMPUTE_STATUS="${compute_status}" \
  python -c 'import json, os

def parse_gpu(row):
    parts = [part.strip() for part in row.split(",", 6)]
    if len(parts) < 7:
        return {}
    index, name, uuid, display_active, utilization, memory_used, memory_total = parts
    def as_int(value):
        try:
            return int(value)
        except ValueError:
            return None
    return {
        "index": index,
        "name": name,
        "uuid": uuid,
        "display_active": display_active,
        "utilization_gpu_pct": as_int(utilization),
        "memory_used_mib": as_int(memory_used),
        "memory_total_mib": as_int(memory_total),
    }

compute_rows = []
for row in os.environ.get("NFN_GPU_COMPUTE", "").splitlines():
    parts = [part.strip() for part in row.split(",", 2)]
    if len(parts) == 3:
        pid, process_name, used_memory = parts
        try:
            used_memory_mib = int(used_memory)
        except ValueError:
            used_memory_mib = None
        compute_rows.append({
            "pid": pid,
            "process_name": process_name,
            "used_memory_mib": used_memory_mib,
        })

gpu_status = int(os.environ.get("NFN_GPU_QUERY_STATUS", "0") or 0)
compute_status = int(os.environ.get("NFN_GPU_COMPUTE_STATUS", "0") or 0)
print(json.dumps({
    "phase": os.environ["NFN_GPU_PHASE"],
    "available": gpu_status == 0,
    "cuda_visible_devices": os.environ.get("NFN_GPU_VISIBLE", ""),
    "cuda_device": os.environ.get("NFN_GPU_DEVICE", ""),
    "selected_gpu": os.environ.get("NFN_GPU_SELECTED", ""),
    "gpu_query_status": gpu_status,
    "compute_query_status": compute_status,
    "gpu": parse_gpu(os.environ.get("NFN_GPU_QUERY", "").splitlines()[0] if os.environ.get("NFN_GPU_QUERY", "").splitlines() else ""),
    "compute_processes": compute_rows,
    "compute_process_count": len(compute_rows),
}, sort_keys=True))
'
}

merge_gpu_load_context_json() {
  local before_json="$1"
  local after_json="$2"
  if [[ ! -f "${JSON_OUT}" ]]; then
    return 0
  fi
  python - "${JSON_OUT}" "${before_json}" "${after_json}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
data["gpu_load_context"] = {
    "before": json.loads(sys.argv[2]),
    "after": json.loads(sys.argv[3]),
}
path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
PY
}

selected_gpu_for_benchmark_lock() {
  local selected_gpu="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"
  selected_gpu="${selected_gpu%%,*}"
  if [[ -z "${selected_gpu}" ]]; then
    selected_gpu="${CUDA_DEVICE}"
  fi
  printf '%s\n' "${selected_gpu}"
}

acquire_gpu_benchmark_lock() {
  case "${GPU_BENCHMARK_LOCK,,}" in
    1|true|yes|on)
      ;;
    0|false|no|off|"")
      return 0
      ;;
    *)
      echo "Invalid NFN_LM_HEAD_BACKWARD_GPU_BENCHMARK_LOCK='${GPU_BENCHMARK_LOCK}'" >&2
      exit 2
      ;;
  esac
  if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required when NFN_LM_HEAD_BACKWARD_GPU_BENCHMARK_LOCK=1" >&2
    exit 2
  fi
  local selected_gpu
  selected_gpu="$(selected_gpu_for_benchmark_lock)"
  local safe_gpu="${selected_gpu//[^A-Za-z0-9_.-]/_}"
  local lock_path="${TMPDIR:-/tmp}/nfn_lm_head_backward_gpu_${safe_gpu}.lock"
  exec 9>"${lock_path}"
  if [[ "${GPU_BENCHMARK_LOCK_TIMEOUT_SECONDS}" == "0" ]]; then
    if ! flock -n 9; then
      echo "LM-head benchmark GPU lock is already held: ${lock_path}" >&2
      exit 75
    fi
  else
    if ! flock -w "${GPU_BENCHMARK_LOCK_TIMEOUT_SECONDS}" 9; then
      echo "Timed out waiting for LM-head benchmark GPU lock: ${lock_path}" >&2
      exit 75
    fi
  fi
}

validate_selected_gpu_idle_snapshot() {
  local snapshot_json="$1"
  local final_attempt="$2"
  NFN_GPU_SNAPSHOT="${snapshot_json}" \
  NFN_GPU_MAX_UTIL="${MAX_SELECTED_GPU_UTILIZATION_PCT}" \
  NFN_GPU_ALLOW_STALE="${ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE}" \
  NFN_GPU_FINAL_ATTEMPT="${final_attempt}" \
  python - <<'PY'
import json
import os
import sys

snapshot = json.loads(os.environ["NFN_GPU_SNAPSHOT"])
max_util = float(os.environ["NFN_GPU_MAX_UTIL"])
allow_stale = os.environ["NFN_GPU_ALLOW_STALE"].lower() in {"1", "true", "yes", "on"}
final_attempt = os.environ["NFN_GPU_FINAL_ATTEMPT"].lower() in {"1", "true", "yes", "on"}
compute_count = int(snapshot.get("compute_process_count", 0) or 0)
gpu = snapshot.get("gpu") or {}
util = gpu.get("utilization_gpu_pct")
selected = snapshot.get("selected_gpu") or snapshot.get("cuda_visible_devices") or snapshot.get("cuda_device")
if compute_count > 0:
    print(f"selected GPU {selected} has {compute_count} active compute process(es)", file=sys.stderr)
    raise SystemExit(2)
if max_util < 0:
    raise SystemExit(0)
if util is None:
    if final_attempt and not allow_stale:
        print(f"selected GPU {selected} utilization is unavailable", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(0 if final_attempt else 1)
if float(util) <= max_util:
    raise SystemExit(0)
if final_attempt and allow_stale:
    print(
        f"selected GPU {selected} still reports {util}% utilization but has no compute processes; allowing stale NVML sample",
        file=sys.stderr,
    )
    raise SystemExit(0)
print(f"selected GPU {selected} utilization {util}% exceeds limit {max_util}%", file=sys.stderr)
raise SystemExit(1)
PY
}

require_selected_gpu_idle() {
  case "${REQUIRE_IDLE_SELECTED_GPU,,}" in
    1|true|yes|on)
      ;;
    0|false|no|off|"")
      return 0
      ;;
    *)
      echo "Invalid NFN_LM_HEAD_BACKWARD_REQUIRE_IDLE_SELECTED_GPU='${REQUIRE_IDLE_SELECTED_GPU}'" >&2
      exit 2
      ;;
  esac
  local retries="${SELECTED_GPU_UTILIZATION_RETRIES}"
  local attempt=0
  while :; do
    local snapshot
    snapshot="$(snapshot_selected_gpu_load_json before)"
    local final_attempt=0
    if [[ "${attempt}" -ge "${retries}" ]]; then
      final_attempt=1
    fi
    if validate_selected_gpu_idle_snapshot "${snapshot}" "${final_attempt}"; then
      GPU_LOAD_BEFORE="${snapshot}"
      return 0
    fi
    local status=$?
    if [[ "${status}" != "1" || "${final_attempt}" == "1" ]]; then
      exit "${status}"
    fi
    sleep "${SELECTED_GPU_UTILIZATION_RETRY_INTERVAL_SECONDS}"
    attempt=$((attempt + 1))
  done
}

case "${NO_LOSS,,}" in
  1|true|yes|on)
    NO_LOSS_ARG=(--no-loss)
    ;;
  0|false|no|off|"")
    NO_LOSS_ARG=()
    ;;
  *)
    echo "Invalid NFN_LM_HEAD_BACKWARD_NO_LOSS='${NO_LOSS}'" >&2
    exit 2
    ;;
esac

case "${CANDIDATE_FIRST,,}" in
  1|true|yes|on)
    CANDIDATE_FIRST_ARG=(--candidate-first)
    ;;
  0|false|no|off|"")
    CANDIDATE_FIRST_ARG=()
    ;;
  *)
    echo "Invalid NFN_LM_HEAD_BACKWARD_CANDIDATE_FIRST='${CANDIDATE_FIRST}'" >&2
    exit 2
    ;;
esac

case "${REQUIRE_TRUE_FUSED,,}" in
  1|true|yes|on)
    REQUIRE_TRUE_FUSED_ARG=(--require-true-fused-candidate)
    ;;
  0|false|no|off|"")
    REQUIRE_TRUE_FUSED_ARG=()
    ;;
  *)
    echo "Invalid NFN_LM_HEAD_BACKWARD_REQUIRE_TRUE_FUSED='${REQUIRE_TRUE_FUSED}'" >&2
    exit 2
    ;;
esac

case "${REQUIRE_GRAPH_BODY_SERIAL,,}" in
  1|true|yes|on)
    ;;
  0|false|no|off|"")
    ;;
  *)
    echo "Invalid NFN_LM_HEAD_BACKWARD_REQUIRE_GRAPH_BODY_SERIAL='${REQUIRE_GRAPH_BODY_SERIAL}'" >&2
    exit 2
    ;;
esac

if [[ "${#NO_LOSS_ARG[@]}" -gt 0 && "${LOSS_BINS}" != "0" ]]; then
  echo "NFN_LM_HEAD_BACKWARD_NO_LOSS cannot be combined with NFN_LM_HEAD_BACKWARD_LOSS_BINS=${LOSS_BINS}" >&2
  exit 2
fi

BENCH_ARGS=(
  --tile-ops-lib "${TILE_OPS_LIB}"
  --baseline-symbol "${BASELINE_SYMBOL}"
  --candidate-symbol "${CANDIDATE_SYMBOL}"
  --rows "${ROWS}"
  --hidden-dim "${HIDDEN_DIM}"
  --vocab "${VOCAB}"
  --row-stride "${ROW_STRIDE}"
  --iterations "${ITERATIONS}"
  --warmup "${WARMUP}"
)
if [[ -n "${REFERENCE_COMPONENT_WARMUP}" ]]; then
  BENCH_ARGS+=(--reference-component-warmup "${REFERENCE_COMPONENT_WARMUP}")
fi
BENCH_ARGS+=(
  "${NO_LOSS_ARG[@]}"
  --loss-bins "${LOSS_BINS}"
  "${CANDIDATE_FIRST_ARG[@]}"
  "${REQUIRE_TRUE_FUSED_ARG[@]}"
  --cuda-device "${CUDA_DEVICE}"
  --json-out "${JSON_OUT}"
)

case "${DRY_RUN,,}" in
  1|true|yes|on)
    DRY_RUN_ENV_PREFIX=()
    for ENV_NAME in \
      NFN_TILE_CUDA_LM_HEAD_GRAPH_BODY_SERIAL \
      NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE \
      NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION \
      NFN_TILE_CUDA_CE_BF16_THREADS \
      NFN_TILE_CUDA_EXTRA_NVCC_FLAGS; do
      if [[ -n "${!ENV_NAME+x}" ]]; then
        DRY_RUN_ENV_PREFIX+=("${ENV_NAME}=${!ENV_NAME}")
      fi
    done
    if [[ "${#DRY_RUN_ENV_PREFIX[@]}" -gt 0 ]]; then
      printf '%q' env
      for ARG in "${DRY_RUN_ENV_PREFIX[@]}"; do
        printf ' %q' "${ARG}"
      done
      printf ' '
    fi
    printf '%q' "${BENCH_BIN}"
    for ARG in "${BENCH_ARGS[@]}"; do
      printf ' %q' "${ARG}"
    done
    printf '\n'
    exit 0
    ;;
  0|false|no|off|"")
    ;;
  *)
    echo "Invalid NFN_LM_HEAD_BACKWARD_DRY_RUN='${DRY_RUN}'" >&2
    exit 2
    ;;
esac

BENCH_DEPS=(
  "${ROOT_DIR}/neuralfn/csrc/native_train/lm_head_backward_bench.cpp"
  "${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.h"
)
REBUILD_BENCH=0
if [[ ! -x "${BENCH_BIN}" ]]; then
  REBUILD_BENCH=1
else
  for DEP in "${BENCH_DEPS[@]}"; do
    if [[ "${DEP}" -nt "${BENCH_BIN}" ]]; then
      REBUILD_BENCH=1
      break
    fi
  done
fi
if [[ "${REBUILD_BENCH}" == "1" ]]; then
  bash "${ROOT_DIR}/tools/build_lm_head_backward_bench.sh" "${BENCH_BIN}" >&2
fi
TILE_OPS_DEPS=(
  "${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.cu"
  "${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.h"
  "${ROOT_DIR}/neuralfn/csrc/tile_cuda/kernels.cu"
  "${ROOT_DIR}/tools/build_native_train_tile_ops.sh"
)
REBUILD_TILE_OPS="${FORCE_REBUILD_TILE_OPS}"
if [[ ! -f "${TILE_OPS_LIB}" ]]; then
  REBUILD_TILE_OPS=1
else
  for DEP in "${TILE_OPS_DEPS[@]}"; do
    if [[ "${DEP}" -nt "${TILE_OPS_LIB}" ]]; then
      REBUILD_TILE_OPS=1
      break
    fi
  done
fi
if [[ "${REBUILD_TILE_OPS}" == "1" ]]; then
  bash "${ROOT_DIR}/tools/build_native_train_tile_ops.sh" "${TILE_OPS_LIB}" >&2
fi

emit_true_fused_requirement_message() {
  if [[ ! -f "${JSON_OUT}" ]]; then
    return 0
  fi
  python -c 'import json, pathlib, sys
try:
    data = json.loads(pathlib.Path(sys.argv[1]).read_text())
except Exception:
    raise SystemExit(0)
if data.get("candidate_true_fused_capability", False):
    raise SystemExit(0)
required = data.get("next_required_kernel_body")
required_symbol = data.get("next_required_symbol")
required_capability = data.get("next_required_capability_symbol")
required_path = data.get("next_required_path_class")
if not (required or required_symbol or required_capability or required_path):
    raise SystemExit(0)
print(
    "LM-head true-fused replacement required: "
    f"next_required_symbol={required_symbol or 'unknown'}, "
    f"next_required_capability_symbol={required_capability or 'unknown'}, "
    f"next_required_path_class={required_path or 'unknown'}, "
    f"next_required_kernel_body={required or 'unknown'}",
    file=sys.stderr,
)
' "${JSON_OUT}"
}

acquire_gpu_benchmark_lock
GPU_LOAD_BEFORE=""
require_selected_gpu_idle
if [[ -z "${GPU_LOAD_BEFORE}" ]]; then
  GPU_LOAD_BEFORE="$(snapshot_selected_gpu_load_json before)"
fi
BENCH_STDOUT="$(mktemp "${TMPDIR:-/tmp}/nfn_lm_head_backward_stdout.XXXXXX")"
BENCH_STATUS=0
"${BENCH_BIN}" "${BENCH_ARGS[@]}" >"${BENCH_STDOUT}" || BENCH_STATUS=$?
GPU_LOAD_AFTER="$(snapshot_selected_gpu_load_json after)"
merge_gpu_load_context_json "${GPU_LOAD_BEFORE}" "${GPU_LOAD_AFTER}"
if [[ -f "${JSON_OUT}" ]]; then
  python -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).read_text(), end="")' "${JSON_OUT}"
elif [[ -s "${BENCH_STDOUT}" ]]; then
  python -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).read_text(), end="")' "${BENCH_STDOUT}"
fi
rm -f "${BENCH_STDOUT}"
if [[ "${BENCH_STATUS}" != "0" ]]; then
  case "${REQUIRE_TRUE_FUSED,,}" in
    1|true|yes|on)
      emit_true_fused_requirement_message
      ;;
  esac
  exit "${BENCH_STATUS}"
fi

case "${REQUIRE_TRUE_FUSED,,}" in
  1|true|yes|on)
    python -c 'import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text())
if not data.get("candidate_true_fused_capability", False):
    required = data.get("next_required_kernel_body")
    required_symbol = data.get("next_required_symbol")
    required_capability = data.get("next_required_capability_symbol")
    required_path = data.get("next_required_path_class")
    suffix = ""
    if required or required_symbol or required_capability or required_path:
        suffix = (
            f"; next_required_symbol={required_symbol or 'unknown'}, "
            f"next_required_capability_symbol={required_capability or 'unknown'}, "
            f"next_required_path_class={required_path or 'unknown'}, "
            f"next_required_kernel_body={required or 'unknown'}"
        )
    if data.get("candidate_sequence_wrapper_only", False):
        raise SystemExit("candidate strict symbol is still sequencing CE/dHidden/dWeight; candidate_true_fused_capability is false" + suffix)
    if data.get("candidate_cuda_graph_wrapper_only", False):
        raise SystemExit("candidate strict symbol is a CUDA Graph wrapper around CE/dHidden/dWeight; candidate_true_fused_capability is false" + suffix)
    raise SystemExit("candidate_true_fused_capability is false" + suffix)
candidate = data.get("candidate", {})
true_fused_launch_count = int(candidate.get("true_fused_launch_count", 0) or 0)
if true_fused_launch_count <= 0:
    raise SystemExit("candidate_true_fused_capability is true but candidate.true_fused_launch_count is zero")
' "${JSON_OUT}"
    ;;
esac

case "${REQUIRE_GRAPH_BODY_TILE,,}" in
  1|true|yes|on)
    python -c 'import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text())
candidate = data.get("candidate") or {}
graph_replay_success_count = int(candidate.get("graph_replay_success_count", 0) or 0)
warmup_graph_replay_success_count = int(candidate.get("warmup_graph_replay_success_count", 0) or 0)
graph_fallback_count = int(candidate.get("graph_fallback_count", 0) or 0)
warmup_graph_fallback_count = int(candidate.get("warmup_graph_fallback_count", 0) or 0)
cublaslt_dhidden = (
    int(candidate.get("graph_body_cublaslt_dhidden_launch_count", 0) or 0)
    + int(candidate.get("warmup_graph_body_cublaslt_dhidden_launch_count", 0) or 0)
)
cublaslt_dweight = (
    int(candidate.get("graph_body_cublaslt_dweight_launch_count", 0) or 0)
    + int(candidate.get("warmup_graph_body_cublaslt_dweight_launch_count", 0) or 0)
)
tile_dhidden = (
    int(candidate.get("graph_body_tile_dhidden_fallback_count", 0) or 0)
    + int(candidate.get("warmup_graph_body_tile_dhidden_fallback_count", 0) or 0)
)
tile_dweight = (
    int(candidate.get("graph_body_tile_dweight_fallback_count", 0) or 0)
    + int(candidate.get("warmup_graph_body_tile_dweight_fallback_count", 0) or 0)
)
if graph_replay_success_count <= 0:
    raise SystemExit("candidate graph-body Tile gate failed: graph_replay_success_count is zero")
if graph_fallback_count != 0 or warmup_graph_fallback_count != 0:
    raise SystemExit(
        "candidate graph-body Tile gate failed: graph fallback occurred "
        f"(timed={graph_fallback_count}, warmup={warmup_graph_fallback_count})"
    )
if cublaslt_dhidden != 0 or cublaslt_dweight != 0:
    raise SystemExit(
        "candidate graph-body Tile gate failed: cuBLASLt diagnostic body ran "
        f"(dhidden={cublaslt_dhidden}, dweight={cublaslt_dweight})"
    )
if tile_dhidden <= 0 or tile_dweight <= 0:
    raise SystemExit(
        "candidate graph-body Tile gate failed: Tile graph-body counters missing "
        f"(dhidden={tile_dhidden}, dweight={tile_dweight}, "
        f"warmup_graph_replay_success_count={warmup_graph_replay_success_count})"
    )
' "${JSON_OUT}"
    ;;
esac

case "${REQUIRE_GRAPH_BODY_SERIAL,,}" in
  1|true|yes|on)
    python -c 'import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text())
candidate = data.get("candidate") or {}
graph_replay_success_count = int(candidate.get("graph_replay_success_count", 0) or 0)
warmup_graph_replay_success_count = int(candidate.get("warmup_graph_replay_success_count", 0) or 0)
abi_path_class = str(data.get("candidate_symbol_abi_path_class", ""))
candidate_path_class = str(data.get("candidate_path_class", ""))
tile_dhidden = (
    int(candidate.get("graph_body_tile_dhidden_fallback_count", 0) or 0)
    + int(candidate.get("warmup_graph_body_tile_dhidden_fallback_count", 0) or 0)
)
tile_dweight = (
    int(candidate.get("graph_body_tile_dweight_fallback_count", 0) or 0)
    + int(candidate.get("warmup_graph_body_tile_dweight_fallback_count", 0) or 0)
)
if "serial-body" not in abi_path_class:
    raise SystemExit(
        "candidate serial graph-body gate failed: serial ABI path class missing "
        f"(candidate_symbol_abi_path_class={abi_path_class!r}, candidate_path_class={candidate_path_class!r})"
    )
if graph_replay_success_count <= 0:
    raise SystemExit(
        "candidate serial graph-body gate failed: CUDA Graph replay did not occur "
        f"(timed={graph_replay_success_count}, warmup={warmup_graph_replay_success_count})"
    )
if tile_dhidden <= 0 or tile_dweight <= 0:
    raise SystemExit(
        "candidate serial graph-body gate failed: Tile graph-body counters missing "
        f"(dhidden={tile_dhidden}, dweight={tile_dweight})"
    )
' "${JSON_OUT}"
    ;;
esac

check_json_ratio() {
  local key="$1"
  local limit="$2"
  if [[ -z "${limit}" ]]; then
    return 0
  fi
  python -c 'import json, pathlib, sys
path, key, limit_raw = sys.argv[1], sys.argv[2], sys.argv[3]
data = json.loads(pathlib.Path(path).read_text())
ratio = float(data[key])
limit = float(limit_raw)
if ratio > limit:
    gap = data.get("candidate_reference_gap") or {}
    gap_parts = []
    for gap_key in (
        "candidate_minus_reference_summed_ms_per_iter",
        "candidate_minus_reference_summed_with_logits_ms_per_iter",
        "candidate_minus_reference_cublaslt_summed_ms_per_iter",
        "candidate_minus_reference_cublaslt_summed_with_logits_ms_per_iter",
    ):
        if gap_key in gap:
            try:
                gap_parts.append(f"{gap_key}={float(gap[gap_key]):.6f}")
            except (TypeError, ValueError):
                gap_parts.append(f"{gap_key}={gap[gap_key]}")
    for component_key, value_key in (
        ("reference_bottleneck_component", "reference_bottleneck_ms_per_iter"),
        ("reference_cublaslt_bottleneck_component", "reference_cublaslt_bottleneck_ms_per_iter"),
    ):
        if component_key in gap:
            try:
                gap_parts.append(f"{component_key}={gap[component_key]}:{float(gap.get(value_key, 0.0)):.6f}ms")
            except (TypeError, ValueError):
                gap_parts.append(f"{component_key}={gap[component_key]}")
    suffix = "; candidate_reference_gap: " + "; ".join(gap_parts) if gap_parts else ""
    raise SystemExit(f"{key} {ratio:.6f} exceeds limit {limit:.6f}{suffix}")
' "${JSON_OUT}" "${key}" "${limit}"
}

check_json_gap() {
  local key="$1"
  local limit="$2"
  if [[ -z "${limit}" ]]; then
    return 0
  fi
  python -c 'import json, pathlib, sys
path, key, limit_raw = sys.argv[1], sys.argv[2], sys.argv[3]
data = json.loads(pathlib.Path(path).read_text())
gap = data.get("candidate_reference_gap") or {}
value = float(gap[key])
limit = float(limit_raw)
if value > limit:
    reference_component = gap.get("reference_bottleneck_component", "unknown")
    reference_ms = float(gap.get("reference_bottleneck_ms_per_iter", 0.0) or 0.0)
    cublaslt_component = gap.get("reference_cublaslt_bottleneck_component", "unknown")
    cublaslt_ms = float(gap.get("reference_cublaslt_bottleneck_ms_per_iter", 0.0) or 0.0)
    raise SystemExit(
        f"{key} {value:.6f} ms exceeds limit {limit:.6f} ms; "
        f"reference_bottleneck_component={reference_component}:{reference_ms:.6f}ms; "
        f"reference_cublaslt_bottleneck_component={cublaslt_component}:{cublaslt_ms:.6f}ms"
    )
' "${JSON_OUT}" "${key}" "${limit}"
}

check_json_ratio \
  "candidate_to_baseline_ms_per_iter_ratio" \
  "${MAX_RATIO}"
check_json_ratio \
  "candidate_to_reference_summed_ms_per_iter_ratio" \
  "${MAX_REFERENCE_RATIO}"
check_json_ratio \
  "candidate_to_reference_summed_with_logits_ms_per_iter_ratio" \
  "${MAX_REFERENCE_WITH_LOGITS_RATIO}"
check_json_ratio \
  "candidate_to_reference_cublaslt_summed_ms_per_iter_ratio" \
  "${MAX_CUBLASLT_REFERENCE_RATIO}"
check_json_ratio \
  "candidate_to_reference_cublaslt_summed_with_logits_ms_per_iter_ratio" \
  "${MAX_CUBLASLT_REFERENCE_WITH_LOGITS_RATIO}"
check_json_gap \
  "candidate_minus_reference_summed_ms_per_iter" \
  "${MAX_REFERENCE_GAP_MS}"
check_json_gap \
  "candidate_minus_reference_summed_with_logits_ms_per_iter" \
  "${MAX_REFERENCE_WITH_LOGITS_GAP_MS}"
check_json_gap \
  "candidate_minus_reference_cublaslt_summed_ms_per_iter" \
  "${MAX_CUBLASLT_REFERENCE_GAP_MS}"
check_json_gap \
  "candidate_minus_reference_cublaslt_summed_with_logits_ms_per_iter" \
  "${MAX_CUBLASLT_REFERENCE_WITH_LOGITS_GAP_MS}"

check_candidate_section_cycles_per_block() {
  local key="$1"
  local limit="$2"
  if [[ -z "${limit}" ]]; then
    return 0
  fi
  python -c 'import json, pathlib, sys
path, key, limit_raw = sys.argv[1], sys.argv[2], sys.argv[3]
data = json.loads(pathlib.Path(path).read_text())
candidate = data.get("candidate") or {}
launches = int(candidate.get("true_fused_launch_count", 0) or 0)
if launches <= 0:
    raise SystemExit(
        f"{key} gate requires a strict true-fused candidate, but "
        "candidate.true_fused_launch_count is zero"
    )
value = float(candidate[key])
limit = float(limit_raw)
if value > limit:
    section = key.replace("true_fused_", "").replace("_cycles_per_block", "")
    raw_cycles = candidate.get(f"true_fused_{section}_cycles", "unknown")
    blocks = candidate.get(f"true_fused_{section}_blocks", "unknown")
    raise SystemExit(
        f"candidate.{key} {value:.6f} exceeds limit {limit:.6f}; "
        f"raw_cycles={raw_cycles}, blocks={blocks}, "
        f"candidate_path_class={data.get('candidate_path_class', 'unknown')}"
    )
' "${JSON_OUT}" "${key}" "${limit}"
}

check_candidate_section_cycles_per_block \
  "true_fused_ce_cycles_per_block" \
  "${MAX_TRUE_FUSED_CE_CYCLES_PER_BLOCK}"
check_candidate_section_cycles_per_block \
  "true_fused_dhidden_cycles_per_block" \
  "${MAX_TRUE_FUSED_DHIDDEN_CYCLES_PER_BLOCK}"
check_candidate_section_cycles_per_block \
  "true_fused_dweight_cycles_per_block" \
  "${MAX_TRUE_FUSED_DWEIGHT_CYCLES_PER_BLOCK}"
