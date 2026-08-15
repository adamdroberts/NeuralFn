// Exact small-batch K-quant MMQ for the native Muse Glimmer runtime.
//
// The device templates come from the pinned, MIT-licensed llama.cpp subset in
// vendor/llama_mmq_62bf73d. NeuralFn supplies the public ABI, strict typed
// descriptor validation, caller-owned workspace, grouping and stream-K
// scheduling. No ggml runtime or shared library is linked.

#include "mmq.cuh"
#include "tile_ops.h"
#include "vecdotq.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include <limits>

extern "C" void ggml_abort(
    const char* file,
    int line,
    const char* format,
    ...) {
  std::fprintf(stderr, "NeuralFn K-quant MMQ assertion at %s:%d: ", file, line);
  std::va_list arguments;
  va_start(arguments, format);
  std::vfprintf(stderr, format, arguments);
  va_end(arguments);
  std::fputc('\n', stderr);
  std::abort();
}

namespace {

namespace cg = cooperative_groups;

constexpr int kQuantThreads = 128;
constexpr std::int64_t kMaxOperations = 4;
constexpr std::int64_t kPaddingRows = 16;

#ifndef NFN_GLIMMER_MMVQ_WARPS
// Two output warps per block is the accepted RTX 5090 full-model geometry.
// Four warps reduced occupancy for Glimmer's one-row decode workload; callers
// may still override this compile-time development knob for another target.
#define NFN_GLIMMER_MMVQ_WARPS 2
#endif

#ifndef NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK
// Eight verifier rows reuse one decoded K-quant weight block without crossing
// the register-pressure cliff observed at sixteen rows on RTX 5090.
#define NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK 8
#endif

#ifndef NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q4
#define NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q4 \
  NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK
#endif

#ifndef NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q5
#define NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q5 \
  NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK
#endif

#ifndef NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q6
#define NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q6 \
  NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK
#endif

#ifndef NFN_GLIMMER_MMVQ_ROW_GROUPS_PER_BLOCK
// One two-warp group preserves occupancy with the eight-row register set.
#define NFN_GLIMMER_MMVQ_ROW_GROUPS_PER_BLOCK 1
#endif

#ifndef NFN_GLIMMER_MMVQ_HOIST_WEIGHT_BLOCK
// Retain one decoded packed-weight block across independent verifier rows.
// Set this to 0 only for the arithmetic-equivalent benchmark control path.
#define NFN_GLIMMER_MMVQ_HOIST_WEIGHT_BLOCK 1
#endif

#ifndef NFN_GLIMMER_MMVQ_SHARED_ACCUMULATORS
// Development switch for a sixteen-row verifier tile.  The accepted default
// keeps row partials in registers; the shared path trades shared-memory
// round-trips for lower register pressure and one packed-weight traversal.
#define NFN_GLIMMER_MMVQ_SHARED_ACCUMULATORS 0
#endif

#ifndef NFN_GLIMMER_MMVQ_PRECOMPUTED_QSUM
// The verifier owns its Q8 workspace.  Candidate builds may store the exact
// signed-byte sum in the otherwise-unused high 16 bits of ds and avoid
// recomputing that integer reduction in every Q4_K/Q5_K output row.
#define NFN_GLIMMER_MMVQ_PRECOMPUTED_QSUM 0
#endif

#ifndef NFN_GLIMMER_MMVQ_TRANSPOSE_Q8_ROWS
// Candidate verifier layout: keep the private Q8 blocks for the reused input
// rows adjacent.  The packed-weight traversal then reads eight neighboring
// activation blocks instead of eight blocks separated by a complete row.
#define NFN_GLIMMER_MMVQ_TRANSPOSE_Q8_ROWS 0
#endif

#ifndef NFN_GLIMMER_MMVQ_PREDECODE_Q6_WEIGHT
// Candidate path that makes Q6_K's two signed packed vectors and scales
// explicitly loop-invariant across verifier rows.
#define NFN_GLIMMER_MMVQ_PREDECODE_Q6_WEIGHT 0
#endif

#ifndef NFN_GLIMMER_MMVQ_USE_MMQ_MIN_ROWS
// Values above the public verifier maximum keep the exact MMVQ path. Candidate
// builds can lower this to measure the pinned tiled MMQ crossover directly.
#define NFN_GLIMMER_MMVQ_USE_MMQ_MIN_ROWS 17
#endif

#ifndef NFN_GLIMMER_MMVQ_EXACT_FIVE_ROW_TAIL
// Specialize the common final DFlash verifier block. It
// keeps the accepted eight-row kernel for full blocks and avoids executing
// three padded zero rows when exactly five target rows remain.
#define NFN_GLIMMER_MMVQ_EXACT_FIVE_ROW_TAIL 1
#endif

#ifndef NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS
// Fallback independent output-channel groups per CTA for one-row decode.
// Per-encoding and hot-shape defaults below are selected by exact full-model
// A/B; this fallback remains useful for compile-time portability sweeps.
#define NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS 1
#endif

#ifndef NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS_Q4
// Four independent output-channel groups improved the non-FFN Q4_K target
// projections on RTX 5090 while preserving each two-warp reduction.
#define NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS_Q4 4
#endif

#ifndef NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS_Q5
// The 202,048-row LM head is occupancy-sensitive and retains one group.
#define NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS_Q5 1
#endif

#ifndef NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS_Q6
#define NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS_Q6 4
#endif

#ifndef NFN_GLIMMER_MMVQ_ONE_ROW_FFN_PAIR_CHANNEL_GROUPS_Q4
// The paired gate/up projection dominates one-token decode and has a much
// larger output grid than Q/K/V/O.  Keep its launch geometry independently
// tunable so a full-model A/B can improve that hot shape without perturbing
// the other arithmetic-identical Q4_K projections.
// The much larger paired 6,656 -> 19,968 gate/up grid regressed with the
// four-group geometry; one group is the accepted full-model winner.
#define NFN_GLIMMER_MMVQ_ONE_ROW_FFN_PAIR_CHANNEL_GROUPS_Q4 1
#endif

#ifndef NFN_GLIMMER_MMVQ_ONE_ROW_ATTN_PROJ_CHANNEL_GROUPS_Q4
#define NFN_GLIMMER_MMVQ_ONE_ROW_ATTN_PROJ_CHANNEL_GROUPS_Q4 \
  NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS_Q4
#endif

#ifndef NFN_GLIMMER_MMVQ_ONE_ROW_ATTN_OUT_CHANNEL_GROUPS_Q4
#define NFN_GLIMMER_MMVQ_ONE_ROW_ATTN_OUT_CHANNEL_GROUPS_Q4 \
  NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS_Q4
#endif

#ifndef NFN_GLIMMER_MMVQ_ONE_ROW_FFN_DOWN_CHANNEL_GROUPS_Q6
#define NFN_GLIMMER_MMVQ_ONE_ROW_FFN_DOWN_CHANNEL_GROUPS_Q6 \
  NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS_Q6
#endif

static_assert(
    NFN_GLIMMER_MMVQ_WARPS == 1 || NFN_GLIMMER_MMVQ_WARPS == 2 ||
        NFN_GLIMMER_MMVQ_WARPS == 4 || NFN_GLIMMER_MMVQ_WARPS == 8,
    "NFN_GLIMMER_MMVQ_WARPS must be 1, 2, 4, or 8");
static_assert(
    NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK == 1 ||
        NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK == 2 ||
        NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK == 4 ||
        NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK == 8 ||
        NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK == 16,
    "NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK must be 1, 2, 4, 8, or 16");
#define NFN_GLIMMER_VALIDATE_INPUT_ROWS(value) \
  ((value) == 1 || (value) == 2 || (value) == 4 || (value) == 8 || \
   (value) == 16)
static_assert(
    NFN_GLIMMER_VALIDATE_INPUT_ROWS(
        NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q4) &&
        NFN_GLIMMER_VALIDATE_INPUT_ROWS(
            NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q5) &&
        NFN_GLIMMER_VALIDATE_INPUT_ROWS(
            NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q6),
    "per-encoding MMVQ input rows must be 1, 2, 4, 8, or 16");
#undef NFN_GLIMMER_VALIDATE_INPUT_ROWS
static_assert(
    NFN_GLIMMER_MMVQ_ROW_GROUPS_PER_BLOCK == 1 ||
        NFN_GLIMMER_MMVQ_ROW_GROUPS_PER_BLOCK == 2,
    "NFN_GLIMMER_MMVQ_ROW_GROUPS_PER_BLOCK must be 1 or 2");
static_assert(
    NFN_GLIMMER_MMVQ_HOIST_WEIGHT_BLOCK == 0 ||
        NFN_GLIMMER_MMVQ_HOIST_WEIGHT_BLOCK == 1,
    "NFN_GLIMMER_MMVQ_HOIST_WEIGHT_BLOCK must be 0 or 1");
static_assert(
    NFN_GLIMMER_MMVQ_SHARED_ACCUMULATORS == 0 ||
        NFN_GLIMMER_MMVQ_SHARED_ACCUMULATORS == 1,
    "NFN_GLIMMER_MMVQ_SHARED_ACCUMULATORS must be 0 or 1");
static_assert(
    NFN_GLIMMER_MMVQ_PRECOMPUTED_QSUM == 0 ||
        NFN_GLIMMER_MMVQ_PRECOMPUTED_QSUM == 1,
    "NFN_GLIMMER_MMVQ_PRECOMPUTED_QSUM must be 0 or 1");
static_assert(
    NFN_GLIMMER_MMVQ_TRANSPOSE_Q8_ROWS == 0 ||
        NFN_GLIMMER_MMVQ_TRANSPOSE_Q8_ROWS == 1,
    "NFN_GLIMMER_MMVQ_TRANSPOSE_Q8_ROWS must be 0 or 1");
static_assert(
    NFN_GLIMMER_MMVQ_PREDECODE_Q6_WEIGHT == 0 ||
        NFN_GLIMMER_MMVQ_PREDECODE_Q6_WEIGHT == 1,
    "NFN_GLIMMER_MMVQ_PREDECODE_Q6_WEIGHT must be 0 or 1");
static_assert(
    NFN_GLIMMER_MMVQ_USE_MMQ_MIN_ROWS >= 2 &&
        NFN_GLIMMER_MMVQ_USE_MMQ_MIN_ROWS <= 17,
    "NFN_GLIMMER_MMVQ_USE_MMQ_MIN_ROWS must be between 2 and 17");
static_assert(
    NFN_GLIMMER_MMVQ_EXACT_FIVE_ROW_TAIL == 0 ||
        NFN_GLIMMER_MMVQ_EXACT_FIVE_ROW_TAIL == 1,
    "NFN_GLIMMER_MMVQ_EXACT_FIVE_ROW_TAIL must be 0 or 1");
static_assert(
    !NFN_GLIMMER_MMVQ_EXACT_FIVE_ROW_TAIL ||
        NFN_GLIMMER_MMVQ_ROW_GROUPS_PER_BLOCK == 1,
    "the exact five-row tail requires one row group per block");
static_assert(
    NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS >= 1 &&
        NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS <= 8,
    "NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS must be between 1 and 8");
static_assert(
    NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS_Q4 >= 1 &&
        NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS_Q4 <= 8 &&
        NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS_Q5 >= 1 &&
        NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS_Q5 <= 8 &&
        NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS_Q6 >= 1 &&
        NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS_Q6 <= 8,
    "per-encoding MMVQ channel groups must be between 1 and 8");
static_assert(
    NFN_GLIMMER_MMVQ_ONE_ROW_FFN_PAIR_CHANNEL_GROUPS_Q4 >= 1 &&
        NFN_GLIMMER_MMVQ_ONE_ROW_FFN_PAIR_CHANNEL_GROUPS_Q4 <= 8,
    "Q4_K FFN-pair MMVQ channel groups must be between 1 and 8");
static_assert(
    NFN_GLIMMER_MMVQ_ONE_ROW_ATTN_PROJ_CHANNEL_GROUPS_Q4 >= 1 &&
        NFN_GLIMMER_MMVQ_ONE_ROW_ATTN_PROJ_CHANNEL_GROUPS_Q4 <= 8 &&
        NFN_GLIMMER_MMVQ_ONE_ROW_ATTN_OUT_CHANNEL_GROUPS_Q4 >= 1 &&
        NFN_GLIMMER_MMVQ_ONE_ROW_ATTN_OUT_CHANNEL_GROUPS_Q4 <= 8 &&
        NFN_GLIMMER_MMVQ_ONE_ROW_FFN_DOWN_CHANNEL_GROUPS_Q6 >= 1 &&
        NFN_GLIMMER_MMVQ_ONE_ROW_FFN_DOWN_CHANNEL_GROUPS_Q6 <= 8,
    "shape-specific MMVQ channel groups must be between 1 and 8");
static_assert(
    NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K ==
            static_cast<std::uint32_t>(GGML_TYPE_Q4_K) &&
        NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K ==
            static_cast<std::uint32_t>(GGML_TYPE_Q5_K) &&
        NFN_NATIVE_TILE_PACKED_WEIGHT_Q6_K ==
            static_cast<std::uint32_t>(GGML_TYPE_Q6_K),
    "NeuralFn and the pinned MMQ encoding IDs must stay identical");
static_assert(
    sizeof(block_q8_1_mmq) % sizeof(int) == 0,
    "MMQ Q8 padding must be safely writable as aligned 32-bit words");

template <bool StoreSums, bool ApplyGate, bool ApplySwiGlu>
__global__ void quantize_q8_1_mmq(
    const float* __restrict__ input,
    const float* __restrict__ auxiliary,
    block_q8_1_mmq* __restrict__ output,
    std::int64_t rows,
    std::int64_t width) {
  const std::int64_t row = blockIdx.x;
  const std::int64_t i0 =
      (static_cast<std::int64_t>(blockDim.x) * blockIdx.y + threadIdx.x) * 4;
  if (row == 0) {
    const std::int64_t block_count = width / QK8_1_MMQ;
    int* padding = reinterpret_cast<int*>(
        output + block_count * rows);
    const std::int64_t padding_ints =
        block_count * kPaddingRows *
        static_cast<std::int64_t>(sizeof(block_q8_1_mmq) / sizeof(int));
    const std::int64_t padding_thread =
        static_cast<std::int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    const std::int64_t padding_threads =
        static_cast<std::int64_t>(gridDim.y) * blockDim.x;
    for (std::int64_t index = padding_thread; index < padding_ints;
         index += padding_threads) {
      padding[index] = 0;
    }
  }
  if (row >= rows || i0 >= width) return;

  float4 values = reinterpret_cast<const float4*>(input + row * width)[i0 / 4];
  if constexpr (ApplyGate) {
    const float4 other =
        reinterpret_cast<const float4*>(auxiliary + row * width)[i0 / 4];
    if constexpr (ApplySwiGlu) {
      values.x = (other.x * values.x) *
          (1.0f / (1.0f + expf(-values.x)));
      values.y = (other.y * values.y) *
          (1.0f / (1.0f + expf(-values.y)));
      values.z = (other.z * values.z) *
          (1.0f / (1.0f + expf(-values.z)));
      values.w = (other.w * values.w) *
          (1.0f / (1.0f + expf(-values.w)));
    } else {
      values.x = values.x / (1.0f + expf(-other.x));
      values.y = values.y / (1.0f + expf(-other.y));
      values.z = values.z / (1.0f + expf(-other.z));
      values.w = values.w / (1.0f + expf(-other.w));
    }
  }
  float maximum = fabsf(values.x);
  maximum = fmaxf(maximum, fabsf(values.y));
  maximum = fmaxf(maximum, fabsf(values.z));
  maximum = fmaxf(maximum, fabsf(values.w));
  for (int offset = 4; offset > 0; offset >>= 1) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, offset, 32));
  }

  float sum = values.x + values.y + values.z + values.w;
  if constexpr (StoreSums) {
    for (int offset = 4; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xffffffffU, sum, offset, 32);
    }
  }

  const float inverse = 127.0f / maximum;
  char4 quantized;
  quantized.x = roundf(values.x * inverse);
  quantized.y = roundf(values.y * inverse);
  quantized.z = roundf(values.z * inverse);
  quantized.w = roundf(values.w * inverse);
  const float scale = 1.0f / inverse;

  const std::int64_t k_block = i0 / QK8_1_MMQ;
  const std::int64_t within = i0 % QK8_1_MMQ;
  block_q8_1_mmq& block = output[k_block * rows + row];
  reinterpret_cast<char4*>(block.qs)[within / 4] = quantized;
  if (within % 32 == 0) {
    if constexpr (StoreSums) {
      block.ds4[within / 32] = make_half2(scale, sum);
    } else {
      block.d4[within / 32] = scale;
    }
  }
}

// Decode uses llama.cpp's ordinary Q8_1 layout and its four-warp cooperative
// MMVQ reduction.  This is deliberately separate from the transposed MMQ
// layout above: forcing a one-row matrix-vector product through the small-
// batch matrix-matrix kernel leaves most of Blackwell idle.
template <bool ApplyGate, bool ApplySwiGlu>
__global__ void quantize_q8_1_mmvq_one_row(
    const float* __restrict__ input,
    const float* __restrict__ auxiliary,
    block_q8_1* __restrict__ output,
    std::int64_t width) {
  const std::int64_t index =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= width) return;
  const int lane = threadIdx.x & 31;
  float value = input[index];
  if constexpr (ApplyGate) {
    const float other = auxiliary[index];
    if constexpr (ApplySwiGlu) {
      value = (other * value) * (1.0f / (1.0f + expf(-value)));
    } else {
      value = value / (1.0f + expf(-other));
    }
  }
  float maximum = warp_reduce_max<32>(fabsf(value));
  float sum = warp_reduce_sum<32>(value);
  const float scale = maximum / 127.0f;
  block_q8_1& block = output[index / QK8_1];
  block.qs[lane] = maximum == 0.0f
      ? 0
      : static_cast<std::int8_t>(roundf(value / scale));
  if (lane == 0) block.ds = make_half2(scale, sum);
}

template <ggml_type Type>
__device__ __forceinline__ float mmvq_vec_dot(
    const void* weights,
    const block_q8_1* q8,
    int weight_block,
    int quant_index) {
  if constexpr (Type == GGML_TYPE_Q4_K) {
    return vec_dot_q4_K_q8_1(weights, q8, weight_block, quant_index);
  } else if constexpr (Type == GGML_TYPE_Q5_K) {
    return vec_dot_q5_K_q8_1(weights, q8, weight_block, quant_index);
  } else {
    return vec_dot_q6_K_q8_1(weights, q8, weight_block, quant_index);
  }
}

__device__ __forceinline__ std::int64_t mmvq_q8_workspace_index(
    std::int64_t row,
    int q8_block,
    int q8_blocks_per_row,
    int q8_padded_rows) {
#if NFN_GLIMMER_MMVQ_TRANSPOSE_Q8_ROWS
  return static_cast<std::int64_t>(q8_block) * q8_padded_rows + row;
#else
  (void)q8_padded_rows;
  return row * q8_blocks_per_row + q8_block;
#endif
}

#if NFN_GLIMMER_MMVQ_PRECOMPUTED_QSUM
__device__ __forceinline__ float q4_k_q8_1_presummed(
    const int* __restrict__ v,
    const int* __restrict__ u,
    const std::int16_t* __restrict__ qsum,
    const std::uint8_t* __restrict__ scales,
    const std::uint8_t* __restrict__ minimums,
    const half2& dm,
    const float* __restrict__ d8) {
  float scaled_sum = 0.0f;
  float minimum_sum = 0.0f;
#pragma unroll
  for (int index = 0; index < QR4_K; ++index) {
    const int v0 = (v[0] >> (4 * index)) & 0x0F0F0F0F;
    const int v1 = (v[1] >> (4 * index)) & 0x0F0F0F0F;
    const int dot = ggml_cuda_dp4a(
        v1, u[2 * index + 1],
        ggml_cuda_dp4a(v0, u[2 * index], 0));
    scaled_sum += d8[index] * (dot * scales[index]);
    minimum_sum += d8[index] * (qsum[index] * minimums[index]);
  }
  const float2 converted = __half22float2(dm);
  return converted.x * scaled_sum - converted.y * minimum_sum;
}

__device__ __forceinline__ float q5_k_q8_1_presummed(
    const int* __restrict__ low,
    const int* __restrict__ high,
    const int* __restrict__ u,
    const std::int16_t* __restrict__ qsum,
    const std::uint8_t* __restrict__ scales,
    const std::uint8_t* __restrict__ minimums,
    const half2& dm,
    const float* __restrict__ d8) {
  float scaled_sum = 0.0f;
  float minimum_sum = 0.0f;
#pragma unroll
  for (int index = 0; index < QR5_K; ++index) {
    const int low0 = (low[0] >> (4 * index)) & 0x0F0F0F0F;
    const int low1 = (low[1] >> (4 * index)) & 0x0F0F0F0F;
    const int high0 = ((high[0] >> index) << 4) & 0x10101010;
    const int high1 = ((high[1] >> index) << 4) & 0x10101010;
    const int v0 = low0 | high0;
    const int v1 = low1 | high1;
    const int dot = ggml_cuda_dp4a(
        v0, u[2 * index],
        ggml_cuda_dp4a(v1, u[2 * index + 1], 0));
    scaled_sum += d8[index] * (dot * scales[index]);
    minimum_sum += d8[index] * (qsum[index] * minimums[index]);
  }
  const float2 converted = __half22float2(dm);
  return converted.x * scaled_sum - converted.y * minimum_sum;
}
#endif

#if NFN_GLIMMER_MMVQ_HOIST_WEIGHT_BLOCK
// The pinned llama.cpp VMMQ helpers decode the packed-weight side of a dot
// product inside every call. Multi-row speculative verification invokes that
// call for several independent activation rows against the same weight block.
// The accepted path keeps the weight-side values live and only reloads each
// row's Q8 activation. Each row's dot-product implementation and partial-sum
// order remain byte-for-byte the same as the ordinary helper.
template <ggml_type Type, int InputRowsPerBlock, int PartialStride>
__device__ __forceinline__ void mmvq_vec_dot_reuse_weight_block(
    const void* __restrict__ weights,
    const block_q8_1* __restrict__ q8_rows,
    const std::int16_t* __restrict__ q8_sums,
    int q8_blocks_per_row,
    int q8_padded_rows,
    std::int64_t input_row0,
    int q8_block,
    int weight_block,
    int quant_index,
    float* __restrict__ partial) {
  if constexpr (Type == GGML_TYPE_Q4_K) {
    const block_q4_K* block =
        static_cast<const block_q4_K*>(weights) + weight_block;
    const int bq8_offset =
        QR4_K * ((quant_index / 2) / (QI8_1 / 2));
    const int* q4 = reinterpret_cast<const int*>(
        block->qs + 16 * bq8_offset + 4 * ((quant_index / 2) % 4));
    int v[2] = {q4[0], q4[4]};

    const uint16_t* scales =
        reinterpret_cast<const uint16_t*>(block->scales);
    uint16_t auxiliary[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
      auxiliary[0] = scales[j + 0] & 0x3f3f;
      auxiliary[1] = scales[j + 2] & 0x3f3f;
    } else {
      auxiliary[0] = ((scales[j + 2] >> 0) & 0x0f0f) |
          ((scales[j - 2] & 0xc0c0) >> 2);
      auxiliary[1] = ((scales[j + 2] >> 4) & 0x0f0f) |
          ((scales[j - 0] & 0xc0c0) >> 2);
    }
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(auxiliary);
    const uint8_t* m = sc + 2;

#if NFN_GLIMMER_MMVQ_SHARED_ACCUMULATORS
#pragma unroll 1
#else
#pragma unroll
#endif
    for (int row = 0; row < InputRowsPerBlock; ++row) {
      int u[2 * QR4_K];
      float d8[QR4_K];
#if NFN_GLIMMER_MMVQ_PRECOMPUTED_QSUM
      std::int16_t qsum[QR4_K];
#endif
#pragma unroll
      for (int i = 0; i < QR4_K; ++i) {
        const int q8_index = q8_block + bq8_offset + i;
        const block_q8_1* q8i = q8_rows + mmvq_q8_workspace_index(
            input_row0 + row, q8_index, q8_blocks_per_row,
            q8_padded_rows);
        d8[i] = __low2float(q8i->ds);
#if NFN_GLIMMER_MMVQ_PRECOMPUTED_QSUM
        const int subset = (quant_index / 2) % 4;
        qsum[i] = q8_sums[
            (mmvq_q8_workspace_index(
                input_row0 + row, q8_index, q8_blocks_per_row,
                q8_padded_rows) * 4) + subset];
#endif
        const int* values = reinterpret_cast<const int*>(q8i->qs) +
            ((quant_index / 2) % 4);
        u[2 * i + 0] = values[0];
        u[2 * i + 1] = values[4];
      }
#if NFN_GLIMMER_MMVQ_PRECOMPUTED_QSUM
      partial[row * PartialStride] += q4_k_q8_1_presummed(
          v, u, qsum, sc, m, block->dm, d8);
#else
      partial[row * PartialStride] += vec_dot_q4_K_q8_1_impl_vmmq(
          v, u, sc, m, block->dm, d8);
#endif
    }
  } else if constexpr (Type == GGML_TYPE_Q5_K) {
    const block_q5_K* block =
        static_cast<const block_q5_K*>(weights) + weight_block;
    const int bq8_offset =
        QR5_K * ((quant_index / 2) / (QI8_1 / 2));
    const int* ql = reinterpret_cast<const int*>(
        block->qs + 16 * bq8_offset + 4 * ((quant_index / 2) % 4));
    const int* qh = reinterpret_cast<const int*>(
        block->qh + 4 * ((quant_index / 2) % 4));
    int vl[2] = {ql[0], ql[4]};
    int vh[2] = {qh[0] >> bq8_offset, qh[4] >> bq8_offset};

    const uint16_t* scales =
        reinterpret_cast<const uint16_t*>(block->scales);
    uint16_t auxiliary[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
      auxiliary[0] = scales[j + 0] & 0x3f3f;
      auxiliary[1] = scales[j + 2] & 0x3f3f;
    } else {
      auxiliary[0] = ((scales[j + 2] >> 0) & 0x0f0f) |
          ((scales[j - 2] & 0xc0c0) >> 2);
      auxiliary[1] = ((scales[j + 2] >> 4) & 0x0f0f) |
          ((scales[j - 0] & 0xc0c0) >> 2);
    }
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(auxiliary);
    const uint8_t* m = sc + 2;

#if NFN_GLIMMER_MMVQ_SHARED_ACCUMULATORS
#pragma unroll 1
#else
#pragma unroll
#endif
    for (int row = 0; row < InputRowsPerBlock; ++row) {
      int u[2 * QR5_K];
      float d8[QR5_K];
#if NFN_GLIMMER_MMVQ_PRECOMPUTED_QSUM
      std::int16_t qsum[QR5_K];
#endif
#pragma unroll
      for (int i = 0; i < QR5_K; ++i) {
        const int q8_index = q8_block + bq8_offset + i;
        const block_q8_1* q8i = q8_rows + mmvq_q8_workspace_index(
            input_row0 + row, q8_index, q8_blocks_per_row,
            q8_padded_rows);
        d8[i] = __low2float(q8i->ds);
#if NFN_GLIMMER_MMVQ_PRECOMPUTED_QSUM
        const int subset = (quant_index / 2) % 4;
        qsum[i] = q8_sums[
            (mmvq_q8_workspace_index(
                input_row0 + row, q8_index, q8_blocks_per_row,
                q8_padded_rows) * 4) + subset];
#endif
        const int* values = reinterpret_cast<const int*>(q8i->qs) +
            ((quant_index / 2) % 4);
        u[2 * i + 0] = values[0];
        u[2 * i + 1] = values[4];
      }
#if NFN_GLIMMER_MMVQ_PRECOMPUTED_QSUM
      partial[row * PartialStride] += q5_k_q8_1_presummed(
          vl, vh, u, qsum, sc, m, block->dm, d8);
#else
      partial[row * PartialStride] += vec_dot_q5_K_q8_1_impl_vmmq(
          vl, vh, u, sc, m, block->dm, d8);
#endif
    }
  } else {
    const block_q6_K* block =
        static_cast<const block_q6_K*>(weights) + weight_block;
    const int bq8_offset = 2 * QR6_K * (quant_index / (QI6_K / 2)) +
        (quant_index % (QI6_K / 2)) / (QI6_K / 4);
    const int scale_offset = (QI6_K / 4) *
            (quant_index / (QI6_K / 2)) +
        (quant_index % (QI6_K / 2)) / (QI6_K / 8);
    const int vh_shift =
        2 * ((quant_index % (QI6_K / 2)) / (QI6_K / 4));
    const int vl = get_int_b2(block->ql, quant_index);
    const int vh = get_int_b2(
        block->qh,
        (QI6_K / 4) * (quant_index / (QI6_K / 2)) +
            quant_index % (QI6_K / 4)) >> vh_shift;
    const int8_t* scales = block->scales + scale_offset;
#if NFN_GLIMMER_MMVQ_PREDECODE_Q6_WEIGHT
    int decoded[QR6_K];
    int decoded_scales[QR6_K];
    const float decoded_scale = static_cast<float>(block->d);
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
      decoded_scales[i] = scales[4 * i];
      const int low = (vl >> (4 * i)) & 0x0F0F0F0F;
      const int high = ((vh >> (4 * i)) << 4) & 0x30303030;
      decoded[i] = __vsubss4(low | high, 0x20202020);
    }
#endif

#if NFN_GLIMMER_MMVQ_SHARED_ACCUMULATORS
#pragma unroll 1
#else
#pragma unroll
#endif
    for (int row = 0; row < InputRowsPerBlock; ++row) {
      int u[QR6_K];
      float d8[QR6_K];
#pragma unroll
      for (int i = 0; i < QR6_K; ++i) {
        const block_q8_1* q8 = q8_rows + mmvq_q8_workspace_index(
            input_row0 + row, q8_block + bq8_offset + 2 * i,
            q8_blocks_per_row, q8_padded_rows);
        u[i] = get_int_b4(
            q8->qs, quant_index % QI8_1);
        d8[i] = __low2float(q8->ds);
      }
#if NFN_GLIMMER_MMVQ_PREDECODE_Q6_WEIGHT
      float sum = 0.0f;
#pragma unroll
      for (int i = 0; i < QR6_K; ++i) {
        sum += d8[i] *
            (ggml_cuda_dp4a(decoded[i], u[i], 0) * decoded_scales[i]);
      }
      partial[row * PartialStride] += decoded_scale * sum;
#else
      partial[row * PartialStride] += vec_dot_q6_K_q8_1_impl_mmvq(
          vl, vh, u, scales, block->d, d8);
#endif
    }
  }
}
#endif

template <ggml_type Type>
__host__ __device__ constexpr int mmvq_vdr() {
  if constexpr (Type == GGML_TYPE_Q4_K) {
    return VDR_Q4_K_Q8_1_MMVQ;
  } else if constexpr (Type == GGML_TYPE_Q5_K) {
    return VDR_Q5_K_Q8_1_MMVQ;
  } else {
    return VDR_Q6_K_Q8_1_MMVQ;
  }
}

template <ggml_type Type>
__host__ __device__ constexpr int mmvq_input_rows_per_block() {
  if constexpr (Type == GGML_TYPE_Q4_K) {
    return NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q4;
  } else if constexpr (Type == GGML_TYPE_Q5_K) {
    return NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q5;
  } else {
    return NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q6;
  }
}

template <ggml_type Type>
__host__ __device__ constexpr int mmvq_one_row_channel_groups() {
  if constexpr (Type == GGML_TYPE_Q4_K) {
    return NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS_Q4;
  } else if constexpr (Type == GGML_TYPE_Q5_K) {
    return NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS_Q5;
  } else {
    return NFN_GLIMMER_MMVQ_ONE_ROW_CHANNEL_GROUPS_Q6;
  }
}

template <ggml_type Type, bool SmallK, int ChannelGroups>
__launch_bounds__(
    32 * NFN_GLIMMER_MMVQ_WARPS * ChannelGroups,
    1) __global__ void
linear_mmvq_one_row(
    const void* __restrict__ weights,
    const block_q8_1* __restrict__ q8,
    float* __restrict__ output,
    std::int64_t input_dim,
    std::int64_t output_dim) {
  constexpr int warps_per_block = NFN_GLIMMER_MMVQ_WARPS;
  constexpr int channel_groups = ChannelGroups;
  constexpr int rows_per_group = SmallK ? warps_per_block : 1;
  constexpr int rows_per_block = rows_per_group * channel_groups;
  constexpr int qk = ggml_cuda_type_traits<Type>::qk;
  constexpr int qi = ggml_cuda_type_traits<Type>::qi;
  constexpr int vdr = mmvq_vdr<Type>();
  constexpr int blocks_per_iteration = vdr * warps_per_block * 32 / qi;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int channel_group = threadIdx.z;
  const int thread = 32 * warp + lane;
  const std::int64_t output0 =
      static_cast<std::int64_t>(blockIdx.x) * rows_per_block +
      channel_group * rows_per_group;
  const int blocks_per_row = static_cast<int>(input_dim / qk);

  float partial[rows_per_group]{};
  for (int weight_block = thread / (qi / vdr);
       weight_block < blocks_per_row;
       weight_block += blocks_per_iteration) {
    const int q8_block = weight_block * (qk / QK8_1);
    const int quant_index = vdr * (thread % (qi / vdr));
#pragma unroll
    for (int row = 0; row < rows_per_group; ++row) {
      if (output0 + row < output_dim) {
        partial[row] += mmvq_vec_dot<Type>(
            weights, q8 + q8_block,
            static_cast<int>(output0 + row) * blocks_per_row + weight_block,
            quant_index);
      }
    }
  }

  __shared__ float warp_partials[
      channel_groups]
      [warps_per_block > 1 ? warps_per_block - 1 : 1]
      [rows_per_group][32];
  if (warp > 0) {
#pragma unroll
    for (int row = 0; row < rows_per_group; ++row) {
      warp_partials[channel_group][warp - 1][row][lane] = partial[row];
    }
  }
  __syncthreads();
  if (warp > 0) return;
#pragma unroll
  for (int row = 0; row < rows_per_group; ++row) {
#pragma unroll
    for (int other_warp = 0; other_warp < warps_per_block - 1; ++other_warp) {
      partial[row] +=
          warp_partials[channel_group][other_warp][row][lane];
    }
    partial[row] = warp_reduce_sum<32>(partial[row]);
    if (lane == row && output0 + row < output_dim) {
      output[output0 + row] = partial[row];
    }
  }
}

template <ggml_type Type, bool SmallK, int ChannelGroups>
__launch_bounds__(
    32 * NFN_GLIMMER_MMVQ_WARPS * ChannelGroups,
    1) __global__ void
linear_mmvq_multi_one_row(
    const void* __restrict__ weights0,
    const void* __restrict__ weights1,
    const void* __restrict__ weights2,
    const void* __restrict__ weights3,
    const block_q8_1* __restrict__ q8,
    float* __restrict__ output0,
    float* __restrict__ output1,
    float* __restrict__ output2,
    float* __restrict__ output3,
    std::int64_t input_dim,
    std::int64_t output_dim0,
    std::int64_t output_dim1,
    std::int64_t output_dim2,
    std::int64_t output_dim3) {
  constexpr int warps_per_block = NFN_GLIMMER_MMVQ_WARPS;
  constexpr int channel_groups = ChannelGroups;
  constexpr int rows_per_group = SmallK ? warps_per_block : 1;
  constexpr int rows_per_block = rows_per_group * channel_groups;
  constexpr int qk = ggml_cuda_type_traits<Type>::qk;
  constexpr int qi = ggml_cuda_type_traits<Type>::qi;
  constexpr int vdr = mmvq_vdr<Type>();
  constexpr int blocks_per_iteration = vdr * warps_per_block * 32 / qi;
  const std::int64_t operation_blocks0 =
      (output_dim0 + rows_per_block - 1) / rows_per_block;
  const std::int64_t operation_blocks1 =
      (output_dim1 + rows_per_block - 1) / rows_per_block;
  const std::int64_t operation_blocks2 =
      (output_dim2 + rows_per_block - 1) / rows_per_block;
  const std::int64_t block = blockIdx.x;
  const void* weights = weights0;
  float* output = output0;
  std::int64_t output_dim = output_dim0;
  std::int64_t operation_block = block;
  if (block >= operation_blocks0) {
    operation_block -= operation_blocks0;
    weights = weights1;
    output = output1;
    output_dim = output_dim1;
  }
  if (block >= operation_blocks0 + operation_blocks1) {
    operation_block -= operation_blocks1;
    weights = weights2;
    output = output2;
    output_dim = output_dim2;
  }
  if (block >= operation_blocks0 + operation_blocks1 + operation_blocks2) {
    operation_block -= operation_blocks2;
    weights = weights3;
    output = output3;
    output_dim = output_dim3;
  }

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int channel_group = threadIdx.z;
  const int thread = 32 * warp + lane;
  const std::int64_t output_row0 = operation_block * rows_per_block +
      channel_group * rows_per_group;
  const int blocks_per_row = static_cast<int>(input_dim / qk);

  float partial[rows_per_group]{};
  for (int weight_block = thread / (qi / vdr);
       weight_block < blocks_per_row;
       weight_block += blocks_per_iteration) {
    const int q8_block = weight_block * (qk / QK8_1);
    const int quant_index = vdr * (thread % (qi / vdr));
#pragma unroll
    for (int row = 0; row < rows_per_group; ++row) {
      if (output_row0 + row < output_dim) {
        partial[row] += mmvq_vec_dot<Type>(
            weights, q8 + q8_block,
            static_cast<int>(output_row0 + row) * blocks_per_row + weight_block,
            quant_index);
      }
    }
  }

  __shared__ float warp_partials[
      channel_groups]
      [warps_per_block > 1 ? warps_per_block - 1 : 1]
      [rows_per_group][32];
  if (warp > 0) {
#pragma unroll
    for (int row = 0; row < rows_per_group; ++row) {
      warp_partials[channel_group][warp - 1][row][lane] = partial[row];
    }
  }
  __syncthreads();
  if (warp > 0) return;
#pragma unroll
  for (int row = 0; row < rows_per_group; ++row) {
#pragma unroll
    for (int other_warp = 0; other_warp < warps_per_block - 1; ++other_warp) {
      partial[row] +=
          warp_partials[channel_group][other_warp][row][lane];
    }
    partial[row] = warp_reduce_sum<32>(partial[row]);
    if (lane == row && output_row0 + row < output_dim) {
      output[output_row0 + row] = partial[row];
    }
  }
}

// Batched verifier variant with the exact per-row Q8_1 quantization and warp
// accumulation order of the accepted one-row MMVQ path. The second grid axis
// supplies independent activation rows; no reduction ever crosses rows.
template <bool ApplyGate, bool ApplySwiGlu>
__global__ void quantize_q8_1_mmvq_rows(
    const float* __restrict__ input,
    const float* __restrict__ auxiliary,
    block_q8_1* __restrict__ output,
    std::int16_t* __restrict__ q8_sums,
    std::int64_t rows,
    std::int64_t width) {
  const std::int64_t row = blockIdx.y;
  const std::int64_t column =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= width) return;
  const int lane = threadIdx.x & 31;
  const int q8_blocks_per_row = static_cast<int>(width / QK8_1);
  const int q8_padded_rows = static_cast<int>(gridDim.y);
  const int q8_block = static_cast<int>(column / QK8_1);
  const std::int64_t block_index = mmvq_q8_workspace_index(
      row, q8_block, q8_blocks_per_row, q8_padded_rows);
  block_q8_1& block = output[block_index];
  if (row >= rows) {
    block.qs[lane] = 0;
    if (lane == 0) block.ds = make_half2(0.0f, 0.0f);
#if NFN_GLIMMER_MMVQ_PRECOMPUTED_QSUM
    if (lane < 4) {
      q8_sums[block_index * 4 + lane] = 0;
    }
#endif
    return;
  }
  float value = input[row * width + column];
  if constexpr (ApplyGate) {
    const float other = auxiliary[row * width + column];
    if constexpr (ApplySwiGlu) {
      value = (other * value) * (1.0f / (1.0f + expf(-value)));
    } else {
      value = value / (1.0f + expf(-other));
    }
  }
  float maximum = warp_reduce_max<32>(fabsf(value));
  float sum = warp_reduce_sum<32>(value);
  const float scale = maximum / 127.0f;
  const std::int8_t quantized = maximum == 0.0f
      ? 0
      : static_cast<std::int8_t>(roundf(value / scale));
  block.qs[lane] = quantized;
#if NFN_GLIMMER_MMVQ_PRECOMPUTED_QSUM
  const int subset = lane & 3;
  int subset_sum = 0;
#pragma unroll
  for (int element = 0; element < 4; ++element) {
    subset_sum += __shfl_sync(
        0xffffffffU, static_cast<int>(quantized), 4 * subset + element);
    subset_sum += __shfl_sync(
        0xffffffffU, static_cast<int>(quantized),
        16 + 4 * subset + element);
  }
  if (lane < 4) {
    q8_sums[block_index * 4 + lane] =
        static_cast<std::int16_t>(subset_sum);
  }
#endif
  if (lane == 0) block.ds = make_half2(scale, sum);
}

// Keep several activation rows in one block so their back-to-back vec-dot
// calls reuse the same packed-weight cache lines.  Each row still traverses
// weight blocks and performs the two-warp/lane reduction in precisely the
// one-row MMVQ order; only independent rows are co-scheduled.
template <
    ggml_type Type,
    bool SmallK,
    int InputRowsPerBlock>
__launch_bounds__(
    32 * NFN_GLIMMER_MMVQ_WARPS * NFN_GLIMMER_MMVQ_ROW_GROUPS_PER_BLOCK,
    1) __global__ void
linear_mmvq_multi_rows(
    const void* __restrict__ weights0,
    const void* __restrict__ weights1,
    const void* __restrict__ weights2,
    const void* __restrict__ weights3,
    const block_q8_1* __restrict__ q8_rows,
    const std::int16_t* __restrict__ q8_sums,
    float* __restrict__ output0,
    float* __restrict__ output1,
    float* __restrict__ output2,
    float* __restrict__ output3,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim0,
    std::int64_t output_dim1,
    std::int64_t output_dim2,
    std::int64_t output_dim3) {
  constexpr int warps_per_block = NFN_GLIMMER_MMVQ_WARPS;
  constexpr int outputs_per_block = SmallK ? warps_per_block : 1;
  constexpr int qk = ggml_cuda_type_traits<Type>::qk;
  constexpr int qi = ggml_cuda_type_traits<Type>::qi;
  constexpr int vdr = mmvq_vdr<Type>();
  constexpr int blocks_per_iteration = vdr * warps_per_block * 32 / qi;
  const std::int64_t operation_blocks0 =
      (output_dim0 + outputs_per_block - 1) / outputs_per_block;
  const std::int64_t operation_blocks1 =
      (output_dim1 + outputs_per_block - 1) / outputs_per_block;
  const std::int64_t operation_blocks2 =
      (output_dim2 + outputs_per_block - 1) / outputs_per_block;
  const std::int64_t operation_block_count = operation_blocks0 +
      operation_blocks1 + operation_blocks2 +
      (output_dim3 + outputs_per_block - 1) / outputs_per_block;
  const std::int64_t block = blockIdx.x;
  if (block >= operation_block_count) return;
  const void* weights = weights0;
  float* output = output0;
  std::int64_t output_dim = output_dim0;
  std::int64_t operation_block = block;
  if (block >= operation_blocks0) {
    operation_block -= operation_blocks0;
    weights = weights1;
    output = output1;
    output_dim = output_dim1;
  }
  if (block >= operation_blocks0 + operation_blocks1) {
    operation_block -= operation_blocks1;
    weights = weights2;
    output = output2;
    output_dim = output_dim2;
  }
  if (block >= operation_blocks0 + operation_blocks1 + operation_blocks2) {
    operation_block -= operation_blocks2;
    weights = weights3;
    output = output3;
    output_dim = output_dim3;
  }

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int row_group = threadIdx.z;
  const int thread = 32 * warp + lane;
  const std::int64_t output_channel0 =
      operation_block * outputs_per_block;
  const int weight_blocks_per_row = static_cast<int>(input_dim / qk);
  const std::int64_t input_row0 =
      (static_cast<std::int64_t>(blockIdx.y) *
           NFN_GLIMMER_MMVQ_ROW_GROUPS_PER_BLOCK +
       row_group) * InputRowsPerBlock;
  const int q8_padded_rows = static_cast<int>(
      gridDim.y * NFN_GLIMMER_MMVQ_ROW_GROUPS_PER_BLOCK *
      InputRowsPerBlock);
#if NFN_GLIMMER_MMVQ_SHARED_ACCUMULATORS && \
    NFN_GLIMMER_MMVQ_HOIST_WEIGHT_BLOCK
  if constexpr (!SmallK) {
    __shared__ float shared_partials
        [NFN_GLIMMER_MMVQ_ROW_GROUPS_PER_BLOCK]
        [warps_per_block][InputRowsPerBlock][32];
#pragma unroll
    for (int input_offset = 0; input_offset < InputRowsPerBlock;
         ++input_offset) {
      shared_partials[row_group][warp][input_offset][lane] = 0.0f;
    }
    __syncthreads();
    for (int weight_block = thread / (qi / vdr);
         weight_block < weight_blocks_per_row;
         weight_block += blocks_per_iteration) {
      const int q8_block = weight_block * (qk / QK8_1);
      const int quant_index = vdr * (thread % (qi / vdr));
      mmvq_vec_dot_reuse_weight_block<
          Type, InputRowsPerBlock, 32>(
          weights, q8_rows, q8_sums,
          static_cast<int>(input_dim / QK8_1),
          q8_padded_rows,
          input_row0, q8_block,
          static_cast<int>(output_channel0) * weight_blocks_per_row +
              weight_block,
          quant_index,
          &shared_partials[row_group][warp][0][lane]);
    }
    __syncthreads();
    if (warp > 0) return;
#pragma unroll
    for (int input_offset = 0; input_offset < InputRowsPerBlock;
         ++input_offset) {
      const std::int64_t input_row = input_row0 + input_offset;
      float value = shared_partials[row_group][0][input_offset][lane];
#pragma unroll
      for (int other_warp = 1; other_warp < warps_per_block; ++other_warp) {
        value += shared_partials[row_group][other_warp][input_offset][lane];
      }
      value = warp_reduce_sum<32>(value);
      if (lane == 0 && input_row < rows && output_channel0 < output_dim) {
        output[input_row * output_dim + output_channel0] = value;
      }
    }
    return;
  }
#endif
  float partial[InputRowsPerBlock][outputs_per_block]{};
  for (int weight_block = thread / (qi / vdr);
       weight_block < weight_blocks_per_row;
       weight_block += blocks_per_iteration) {
    const int q8_block = weight_block * (qk / QK8_1);
    const int quant_index = vdr * (thread % (qi / vdr));
#if NFN_GLIMMER_MMVQ_HOIST_WEIGHT_BLOCK
    if constexpr (!SmallK) {
      if (input_row0 < rows && output_channel0 < output_dim) {
        mmvq_vec_dot_reuse_weight_block<Type, InputRowsPerBlock, 1>(
            weights, q8_rows, q8_sums,
            static_cast<int>(input_dim / QK8_1),
            q8_padded_rows,
            input_row0, q8_block,
            static_cast<int>(output_channel0) * weight_blocks_per_row +
                weight_block,
            quant_index, &partial[0][0]);
        continue;
      }
    }
#endif
#pragma unroll
    for (int input_offset = 0; input_offset < InputRowsPerBlock;
         ++input_offset) {
      const std::int64_t input_row = input_row0 + input_offset;
      const block_q8_1* q8 = input_row < rows
          ? q8_rows + input_row * (input_dim / QK8_1)
          : q8_rows;
#pragma unroll
      for (int output_index = 0; output_index < outputs_per_block;
           ++output_index) {
        if (input_row < rows &&
            output_channel0 + output_index < output_dim) {
          partial[input_offset][output_index] += mmvq_vec_dot<Type>(
              weights, q8 + q8_block,
              static_cast<int>(output_channel0 + output_index) *
                      weight_blocks_per_row +
                  weight_block,
              quant_index);
        }
      }
    }
  }

  __shared__ float warp_partials[
      NFN_GLIMMER_MMVQ_ROW_GROUPS_PER_BLOCK]
      [warps_per_block > 1 ? warps_per_block - 1 : 1]
      [InputRowsPerBlock][outputs_per_block][32];
  if (warp > 0) {
#pragma unroll
    for (int input_offset = 0; input_offset < InputRowsPerBlock;
         ++input_offset) {
#pragma unroll
      for (int output_index = 0; output_index < outputs_per_block;
           ++output_index) {
        warp_partials[row_group][warp - 1][input_offset][output_index][lane] =
            partial[input_offset][output_index];
      }
    }
  }
  __syncthreads();
  if (warp > 0) return;
#pragma unroll
  for (int input_offset = 0; input_offset < InputRowsPerBlock;
       ++input_offset) {
    const std::int64_t input_row = input_row0 + input_offset;
#pragma unroll
    for (int output_index = 0; output_index < outputs_per_block;
         ++output_index) {
#pragma unroll
      for (int other_warp = 0; other_warp < warps_per_block - 1;
           ++other_warp) {
        partial[input_offset][output_index] +=
            warp_partials[row_group][other_warp][input_offset][output_index][lane];
      }
      partial[input_offset][output_index] = warp_reduce_sum<32>(
          partial[input_offset][output_index]);
      if (lane == output_index && input_row < rows &&
          output_channel0 + output_index < output_dim) {
        output[input_row * output_dim + output_channel0 + output_index] =
            partial[input_offset][output_index];
      }
    }
  }
}

template <ggml_type Type, int J, bool Fallback>
cudaError_t launch_mmq(
    const std::uint8_t* packed_weights,
    const block_q8_1_mmq* q8,
    float* output,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t row_stride_bytes,
    float* fixup_workspace,
    std::size_t fixup_workspace_bytes,
    int multiprocessors,
    cudaStream_t stream) {
  constexpr int threads = 256;
  constexpr ggml_cuda_mmq_config config =
      ggml_cuda_mmq_get_config_ampere(Type, J, Fallback);
  constexpr int tile_i = 128;
  const int shared_bytes = static_cast<int>(mmq_get_nbytes_shared(config, 1200));
  cudaError_t status = cudaFuncSetAttribute(
      mul_mat_q<Type, J, Fallback>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_bytes);
  if (status != cudaSuccess) return status;

  (void)row_stride_bytes;
  const std::int64_t weight_blocks_per_row = input_dim / QK_K;
  const std::int64_t q8_stride_ints =
      rows * input_dim * static_cast<std::int64_t>(sizeof(block_q8_1)) /
      (QK8_1 * static_cast<std::int64_t>(sizeof(int)));
  const int ntx = static_cast<int>((rows + J - 1) / J);
  const int nty = static_cast<int>((output_dim + tile_i - 1) / tile_i);
  if (multiprocessors <= 0) return cudaErrorInvalidValue;
  const int tiles = ntx * nty;
  const int waves = (tiles + multiprocessors - 1) / multiprocessors;
  const int efficiency = 100 * tiles / (multiprocessors * waves);
  int stream_blocks = efficiency >= 90 ? tiles : multiprocessors;
  const bool needs_fixup = tiles % stream_blocks != 0;
  const std::size_t required_fixup = needs_fixup
      ? static_cast<std::size_t>(stream_blocks) * J * tile_i * sizeof(float)
      : 0;
  if (required_fixup > fixup_workspace_bytes) return cudaErrorInvalidValue;
  const dim3 grid(static_cast<unsigned int>(stream_blocks), 1, 1);
  const dim3 block(32, threads / 32, 1);
  mul_mat_q<Type, J, Fallback><<<grid, block, shared_bytes, stream>>>(
      reinterpret_cast<const char*>(packed_weights),
      reinterpret_cast<const int*>(q8),
      nullptr,
      nullptr,
      output,
      needs_fixup ? fixup_workspace : nullptr,
      nullptr,
      init_fastdiv_values(input_dim / QK_K),
      output_dim,
      rows,
      weight_blocks_per_row,
      rows,
      output_dim,
      init_fastdiv_values(1),
      init_fastdiv_values(1),
      output_dim * weight_blocks_per_row,
      q8_stride_ints,
      rows * output_dim,
      init_fastdiv_values(1),
      init_fastdiv_values(1),
      output_dim * weight_blocks_per_row,
      q8_stride_ints,
      rows * output_dim,
      init_fastdiv_values(ntx));
  status = cudaPeekAtLastError();
  if (status != cudaSuccess || !needs_fixup) return status;
  const dim3 fixup_grid(
      static_cast<unsigned int>(stream_blocks),
      static_cast<unsigned int>(tile_i / 32),
      1);
  const dim3 fixup_block(32, (threads / 32) / 2, 1);
  mul_mat_q_stream_k_fixup<Type, J, Fallback>
      <<<fixup_grid, fixup_block, 0, stream>>>(
          nullptr,
          nullptr,
          output,
          fixup_workspace,
          init_fastdiv_values(input_dim / QK_K),
          output_dim,
          rows,
          output_dim,
          init_fastdiv_values(1),
          rows * output_dim,
          init_fastdiv_values(1),
          rows * output_dim,
          init_fastdiv_values(ntx));
  return cudaPeekAtLastError();
}

template <ggml_type Type>
cudaError_t dispatch_rows(
    const std::uint8_t* packed_weights,
    const block_q8_1_mmq* q8,
    float* output,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t row_stride_bytes,
    float* fixup_workspace,
    std::size_t fixup_workspace_bytes,
    int multiprocessors,
    cudaStream_t stream) {
  const bool fallback = output_dim % 128 != 0;
  if (rows <= 8) {
    if (fallback) {
      return launch_mmq<Type, 8, true>(
          packed_weights, q8, output, rows, input_dim, output_dim,
          row_stride_bytes, fixup_workspace, fixup_workspace_bytes,
          multiprocessors, stream);
    }
    return launch_mmq<Type, 8, false>(
        packed_weights, q8, output, rows, input_dim, output_dim,
        row_stride_bytes, fixup_workspace, fixup_workspace_bytes,
        multiprocessors, stream);
  }
  if (fallback) {
    return launch_mmq<Type, 16, true>(
        packed_weights, q8, output, rows, input_dim, output_dim,
        row_stride_bytes, fixup_workspace, fixup_workspace_bytes,
        multiprocessors, stream);
  }
  return launch_mmq<Type, 16, false>(
      packed_weights, q8, output, rows, input_dim, output_dim,
      row_stride_bytes, fixup_workspace, fixup_workspace_bytes,
      multiprocessors, stream);
}

bool query_multiprocessors(int* output) {
  if (output == nullptr) return false;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return false;
  int multiprocessors = 0;
  if (cudaDeviceGetAttribute(
          &multiprocessors, cudaDevAttrMultiProcessorCount, device) !=
          cudaSuccess ||
      multiprocessors <= 0) {
    return false;
  }
  *output = multiprocessors;
  return true;
}

bool workspace_layout(
    std::int64_t rows,
    std::int64_t input_dim,
    int multiprocessors,
    std::size_t* q8_bytes,
    std::size_t* total_bytes) {
  if (q8_bytes == nullptr || total_bytes == nullptr || rows < 1 || rows > 16 ||
      input_dim <= 0 || input_dim % QK8_1_MMQ != 0 || multiprocessors <= 0) {
    return false;
  }
  constexpr std::size_t output_tile = 128;
  const std::size_t blocks = static_cast<std::size_t>(input_dim / QK8_1_MMQ);
  const std::size_t q8_rows =
      static_cast<std::size_t>(rows + kPaddingRows);
  if (blocks > std::numeric_limits<std::size_t>::max() /
          (q8_rows * sizeof(block_q8_1_mmq))) {
    return false;
  }
  const std::size_t raw_q8 = blocks * q8_rows * sizeof(block_q8_1_mmq);
  const std::size_t aligned_q8 = (raw_q8 + 255) & ~std::size_t(255);
  const std::size_t fixup = static_cast<std::size_t>(multiprocessors) *
      16 * output_tile * sizeof(float);
  if (aligned_q8 > std::numeric_limits<std::size_t>::max() - fixup) {
    return false;
  }
  *q8_bytes = aligned_q8;
  *total_bytes = aligned_q8 + fixup;
  return true;
}

std::int64_t block_bytes(std::uint32_t encoding) {
  switch (encoding) {
    case NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K:
      return 144;
    case NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K:
      return 176;
    case NFN_NATIVE_TILE_PACKED_WEIGHT_Q6_K:
      return 210;
    default:
      return 0;
  }
}

bool validate_descriptor(
    const NfnNativeTilePackedWeightDescriptorV1* descriptor,
    std::int64_t input_dim,
    void* cuda_stream) {
  if (descriptor == nullptr ||
      descriptor->struct_size < sizeof(NfnNativeTilePackedWeightDescriptorV1) ||
      descriptor->version != NFN_NATIVE_TILE_PACKED_WEIGHT_V1 ||
      descriptor->flags != 0 || descriptor->reserved0 != 0 ||
      descriptor->reserved1 != 0 || descriptor->data == nullptr ||
      descriptor->data_nbytes <= 0 || descriptor->output_dim <= 0 ||
      descriptor->input_dim != input_dim || input_dim <= 0 || input_dim % QK_K != 0 ||
      (descriptor->cuda_stream != nullptr &&
       descriptor->cuda_stream != cuda_stream)) {
    return false;
  }
  const std::int64_t bytes = block_bytes(descriptor->encoding);
  if (bytes == 0 || input_dim / QK_K >
          std::numeric_limits<std::int64_t>::max() / bytes) {
    return false;
  }
  const std::int64_t expected_stride = input_dim / QK_K * bytes;
  if (descriptor->row_stride_bytes != expected_stride ||
      descriptor->output_dim >
          std::numeric_limits<std::int64_t>::max() / expected_stride) {
    return false;
  }
  return descriptor->data_nbytes == descriptor->output_dim * expected_stride;
}

template <ggml_type Type>
cudaError_t launch_mmvq_one_row(
    const NfnNativeTilePackedWeightDescriptorV1& descriptor,
    const block_q8_1* q8,
    float* output,
    cudaStream_t stream) {
  constexpr int qi = ggml_cuda_type_traits<Type>::qi;
  constexpr int vdr = mmvq_vdr<Type>();
  constexpr int warps_per_block = NFN_GLIMMER_MMVQ_WARPS;
  constexpr int channel_groups = mmvq_one_row_channel_groups<Type>();
  const int blocks_per_row = static_cast<int>(descriptor.input_dim / QK_K);
  constexpr int blocks_per_iteration_one_warp = vdr * 32 / qi;
  const bool small_k = blocks_per_row <
      warps_per_block * blocks_per_iteration_one_warp;
  const dim3 threads(32, warps_per_block, channel_groups);
  if (small_k) {
    constexpr int outputs_per_block = warps_per_block * channel_groups;
    const unsigned int blocks = static_cast<unsigned int>(
        (descriptor.output_dim + outputs_per_block - 1) / outputs_per_block);
    linear_mmvq_one_row<Type, true, channel_groups><<<
        blocks, threads, 0, stream>>>(
        descriptor.data, q8, output, descriptor.input_dim,
        descriptor.output_dim);
  } else {
    const unsigned int blocks = static_cast<unsigned int>(
        (descriptor.output_dim + channel_groups - 1) / channel_groups);
    linear_mmvq_one_row<Type, false, channel_groups><<<
        blocks, threads, 0, stream>>>(
        descriptor.data, q8, output, descriptor.input_dim,
        descriptor.output_dim);
  }
  return cudaPeekAtLastError();
}

template <ggml_type Type, int ChannelGroups>
cudaError_t launch_matching_mmvq_one_row_grouped(
    const std::array<
        const NfnNativeTilePackedWeightDescriptorV1*, kMaxOperations>&
        matching,
    const std::array<float*, kMaxOperations>& matching_outputs,
    std::int64_t matching_count,
    const block_q8_1* q8,
    cudaStream_t stream) {
  if (matching_count == 0) return cudaSuccess;
  const std::int64_t input_dim = matching[0]->input_dim;
  constexpr int qi = ggml_cuda_type_traits<Type>::qi;
  constexpr int vdr = mmvq_vdr<Type>();
  constexpr int warps_per_block = NFN_GLIMMER_MMVQ_WARPS;
  constexpr int channel_groups = ChannelGroups;
  const int blocks_per_row = static_cast<int>(input_dim / QK_K);
  constexpr int blocks_per_iteration_one_warp = vdr * 32 / qi;
  const bool small_k = blocks_per_row <
      warps_per_block * blocks_per_iteration_one_warp;
  const auto output_dim = [&](std::size_t index) {
    return matching[index] == nullptr ? 0 : matching[index]->output_dim;
  };
  const auto weight = [&](std::size_t index) -> const void* {
    return matching[index] == nullptr ? nullptr : matching[index]->data;
  };
  const dim3 threads(32, warps_per_block, channel_groups);
  if (small_k) {
    constexpr int outputs_per_block = warps_per_block * channel_groups;
    const std::int64_t blocks =
        (output_dim(0) + outputs_per_block - 1) / outputs_per_block +
        (output_dim(1) + outputs_per_block - 1) / outputs_per_block +
        (output_dim(2) + outputs_per_block - 1) / outputs_per_block +
        (output_dim(3) + outputs_per_block - 1) / outputs_per_block;
    linear_mmvq_multi_one_row<Type, true, channel_groups><<<
        static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        weight(0), weight(1), weight(2), weight(3), q8,
        matching_outputs[0], matching_outputs[1], matching_outputs[2],
        matching_outputs[3], input_dim, output_dim(0), output_dim(1),
        output_dim(2), output_dim(3));
  } else {
    const std::int64_t blocks =
        (output_dim(0) + channel_groups - 1) / channel_groups +
        (output_dim(1) + channel_groups - 1) / channel_groups +
        (output_dim(2) + channel_groups - 1) / channel_groups +
        (output_dim(3) + channel_groups - 1) / channel_groups;
    linear_mmvq_multi_one_row<Type, false, channel_groups><<<
        static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        weight(0), weight(1), weight(2), weight(3), q8,
        matching_outputs[0], matching_outputs[1], matching_outputs[2],
        matching_outputs[3], input_dim, output_dim(0), output_dim(1),
        output_dim(2), output_dim(3));
  }
  return cudaPeekAtLastError();
}

template <ggml_type Type>
cudaError_t launch_matching_mmvq_one_row(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    float* const* outputs,
    std::int64_t operation_count,
    const block_q8_1* q8,
    cudaStream_t stream) {
  std::array<const NfnNativeTilePackedWeightDescriptorV1*, kMaxOperations>
      matching{};
  std::array<float*, kMaxOperations> matching_outputs{};
  std::int64_t matching_count = 0;
  for (std::int64_t index = 0; index < operation_count; ++index) {
    if (descriptors[index]->encoding != static_cast<std::uint32_t>(Type)) {
      continue;
    }
    matching[static_cast<std::size_t>(matching_count)] = descriptors[index];
    matching_outputs[static_cast<std::size_t>(matching_count)] = outputs[index];
    ++matching_count;
  }
  if constexpr (Type == GGML_TYPE_Q4_K) {
    // Glimmer's SwiGLU gate/up pair has two D -> 3D projections.  It accounts
    // for most target decode time, so dispatch it to a separately compiled
    // launch shape while keeping every dot product and reduction unchanged.
    bool ffn_projection = matching_count >= 1;
    for (std::int64_t index = 0; index < matching_count; ++index) {
      ffn_projection = ffn_projection &&
          matching[static_cast<std::size_t>(index)]->output_dim ==
              3 * matching[static_cast<std::size_t>(index)]->input_dim;
    }
    if (ffn_projection) {
      return launch_matching_mmvq_one_row_grouped<
          Type, NFN_GLIMMER_MMVQ_ONE_ROW_FFN_PAIR_CHANNEL_GROUPS_Q4>(
          matching, matching_outputs, matching_count, q8, stream);
    }
    bool attention_projection = matching_count >= 1 &&
        matching[0]->input_dim == 6656;
    for (std::int64_t index = 0; index < matching_count; ++index) {
      attention_projection = attention_projection &&
          matching[static_cast<std::size_t>(index)]->output_dim <= 4096;
    }
    if (attention_projection) {
      return launch_matching_mmvq_one_row_grouped<
          Type, NFN_GLIMMER_MMVQ_ONE_ROW_ATTN_PROJ_CHANNEL_GROUPS_Q4>(
          matching, matching_outputs, matching_count, q8, stream);
    }
    const bool attention_output = matching_count == 1 &&
        matching[0]->input_dim == 4096 && matching[0]->output_dim == 6656;
    if (attention_output) {
      return launch_matching_mmvq_one_row_grouped<
          Type, NFN_GLIMMER_MMVQ_ONE_ROW_ATTN_OUT_CHANNEL_GROUPS_Q4>(
          matching, matching_outputs, matching_count, q8, stream);
    }
  }
  if constexpr (Type == GGML_TYPE_Q6_K) {
    const bool ffn_down = matching_count == 1 &&
        matching[0]->input_dim == 3 * matching[0]->output_dim;
    if (ffn_down) {
      return launch_matching_mmvq_one_row_grouped<
          Type, NFN_GLIMMER_MMVQ_ONE_ROW_FFN_DOWN_CHANNEL_GROUPS_Q6>(
          matching, matching_outputs, matching_count, q8, stream);
    }
  }
  return launch_matching_mmvq_one_row_grouped<
      Type, mmvq_one_row_channel_groups<Type>()>(
      matching, matching_outputs, matching_count, q8, stream);
}

template <ggml_type Type>
cudaError_t launch_matching_mmvq_rows(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    float* const* outputs,
    std::int64_t operation_count,
    const block_q8_1* q8,
    const std::int16_t* q8_sums,
    std::int64_t rows,
    cudaStream_t stream) {
  std::array<const NfnNativeTilePackedWeightDescriptorV1*, kMaxOperations>
      matching{};
  std::array<float*, kMaxOperations> matching_outputs{};
  std::int64_t matching_count = 0;
  for (std::int64_t index = 0; index < operation_count; ++index) {
    if (descriptors[index]->encoding != static_cast<std::uint32_t>(Type)) {
      continue;
    }
    matching[static_cast<std::size_t>(matching_count)] = descriptors[index];
    matching_outputs[static_cast<std::size_t>(matching_count)] = outputs[index];
    ++matching_count;
  }
  if (matching_count == 0) return cudaSuccess;
  const std::int64_t input_dim = matching[0]->input_dim;
  constexpr int qi = ggml_cuda_type_traits<Type>::qi;
  constexpr int vdr = mmvq_vdr<Type>();
  constexpr int warps_per_block = NFN_GLIMMER_MMVQ_WARPS;
  const int weight_blocks_per_row = static_cast<int>(input_dim / QK_K);
  constexpr int blocks_per_iteration_one_warp = vdr * 32 / qi;
  const bool small_k = weight_blocks_per_row <
      warps_per_block * blocks_per_iteration_one_warp;
  const auto output_dim = [&](std::size_t index) {
    return matching[index] == nullptr ? 0 : matching[index]->output_dim;
  };
  const auto weight = [&](std::size_t index) -> const void* {
    return matching[index] == nullptr ? nullptr : matching[index]->data;
  };
  const dim3 threads(
      32, warps_per_block, NFN_GLIMMER_MMVQ_ROW_GROUPS_PER_BLOCK);
  constexpr int input_rows_per_block = mmvq_input_rows_per_block<Type>();
  const unsigned int input_row_blocks = static_cast<unsigned int>(
      (rows + input_rows_per_block * NFN_GLIMMER_MMVQ_ROW_GROUPS_PER_BLOCK - 1) /
      (input_rows_per_block * NFN_GLIMMER_MMVQ_ROW_GROUPS_PER_BLOCK));
#if NFN_GLIMMER_MMVQ_EXACT_FIVE_ROW_TAIL
  if (rows == 5) {
    constexpr int tail_rows_per_block = 5;
    if (small_k) {
      const std::int64_t blocks =
          (output_dim(0) + warps_per_block - 1) / warps_per_block +
          (output_dim(1) + warps_per_block - 1) / warps_per_block +
          (output_dim(2) + warps_per_block - 1) / warps_per_block +
          (output_dim(3) + warps_per_block - 1) / warps_per_block;
      linear_mmvq_multi_rows<Type, true, tail_rows_per_block><<<
          dim3(static_cast<unsigned int>(blocks), 1, 1),
          threads, 0, stream>>>(
          weight(0), weight(1), weight(2), weight(3), q8, q8_sums,
          matching_outputs[0], matching_outputs[1], matching_outputs[2],
          matching_outputs[3], rows, input_dim, output_dim(0), output_dim(1),
          output_dim(2), output_dim(3));
    } else {
      const std::int64_t blocks =
          output_dim(0) + output_dim(1) + output_dim(2) + output_dim(3);
      linear_mmvq_multi_rows<Type, false, tail_rows_per_block><<<
          dim3(static_cast<unsigned int>(blocks), 1, 1),
          threads, 0, stream>>>(
          weight(0), weight(1), weight(2), weight(3), q8, q8_sums,
          matching_outputs[0], matching_outputs[1], matching_outputs[2],
          matching_outputs[3], rows, input_dim, output_dim(0), output_dim(1),
          output_dim(2), output_dim(3));
    }
    return cudaPeekAtLastError();
  }
#endif
  if (small_k) {
    const std::int64_t blocks =
        (output_dim(0) + warps_per_block - 1) / warps_per_block +
        (output_dim(1) + warps_per_block - 1) / warps_per_block +
        (output_dim(2) + warps_per_block - 1) / warps_per_block +
        (output_dim(3) + warps_per_block - 1) / warps_per_block;
    linear_mmvq_multi_rows<Type, true, input_rows_per_block><<<
        dim3(static_cast<unsigned int>(blocks), input_row_blocks, 1),
        threads, 0, stream>>>(
        weight(0), weight(1), weight(2), weight(3), q8, q8_sums,
        matching_outputs[0], matching_outputs[1], matching_outputs[2],
        matching_outputs[3], rows, input_dim, output_dim(0), output_dim(1),
        output_dim(2), output_dim(3));
  } else {
    const std::int64_t blocks =
        output_dim(0) + output_dim(1) + output_dim(2) + output_dim(3);
    linear_mmvq_multi_rows<Type, false, input_rows_per_block><<<
        dim3(static_cast<unsigned int>(blocks), input_row_blocks, 1),
        threads, 0, stream>>>(
        weight(0), weight(1), weight(2), weight(3), q8, q8_sums,
        matching_outputs[0], matching_outputs[1], matching_outputs[2],
        matching_outputs[3], rows, input_dim, output_dim(0), output_dim(1),
        output_dim(2), output_dim(3));
  }
  return cudaPeekAtLastError();
}

int run_mmvq_multi_one_row(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    const float* input,
    const float* auxiliary,
    bool apply_swiglu,
    float* const* outputs,
    std::int64_t operation_count,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream) {
  if (descriptors == nullptr || input == nullptr || outputs == nullptr ||
      operation_count < 1 || operation_count > kMaxOperations ||
      workspace == nullptr || workspace_nbytes <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::int64_t input_dim = descriptors[0] == nullptr
      ? 0 : descriptors[0]->input_dim;
  const std::size_t q8_bytes = input_dim > 0
      ? static_cast<std::size_t>(input_dim / QK8_1) * sizeof(block_q8_1)
      : 0;
  for (std::int64_t index = 0; index < operation_count; ++index) {
    if (outputs[index] == nullptr ||
        !validate_descriptor(descriptors[index], input_dim, cuda_stream)) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
  }
  if ((apply_swiglu && auxiliary == nullptr) ||
      input_dim % QK8_1 != 0 || q8_bytes == 0 ||
      static_cast<std::uint64_t>(workspace_nbytes) < q8_bytes) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  block_q8_1* q8 = static_cast<block_q8_1*>(workspace);
  constexpr int quant_threads = 256;
  const unsigned int quant_blocks = static_cast<unsigned int>(
      (input_dim + quant_threads - 1) / quant_threads);
  if (auxiliary != nullptr && apply_swiglu) {
    quantize_q8_1_mmvq_one_row<true, true>
        <<<quant_blocks, quant_threads, 0, stream>>>(
            input, auxiliary, q8, input_dim);
  } else if (auxiliary != nullptr) {
    quantize_q8_1_mmvq_one_row<true, false>
        <<<quant_blocks, quant_threads, 0, stream>>>(
            input, auxiliary, q8, input_dim);
  } else {
    quantize_q8_1_mmvq_one_row<false, false>
        <<<quant_blocks, quant_threads, 0, stream>>>(
            input, nullptr, q8, input_dim);
  }
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) return static_cast<int>(status);
  status = launch_matching_mmvq_one_row<GGML_TYPE_Q4_K>(
      descriptors, outputs, operation_count, q8, stream);
  if (status == cudaSuccess) {
    status = launch_matching_mmvq_one_row<GGML_TYPE_Q5_K>(
        descriptors, outputs, operation_count, q8, stream);
  }
  if (status == cudaSuccess) {
    status = launch_matching_mmvq_one_row<GGML_TYPE_Q6_K>(
        descriptors, outputs, operation_count, q8, stream);
  }
  return static_cast<int>(status);
}

int run_mmvq_multi_one_row_prequantized(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    float* const* outputs,
    std::int64_t operation_count,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream) {
  if (descriptors == nullptr || outputs == nullptr || operation_count < 1 ||
      operation_count > kMaxOperations || workspace == nullptr ||
      workspace_nbytes <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::int64_t input_dim = descriptors[0] == nullptr
      ? 0 : descriptors[0]->input_dim;
  const std::size_t q8_bytes = input_dim > 0
      ? static_cast<std::size_t>(input_dim / QK8_1) * sizeof(block_q8_1)
      : 0;
  for (std::int64_t index = 0; index < operation_count; ++index) {
    if (outputs[index] == nullptr ||
        !validate_descriptor(descriptors[index], input_dim, cuda_stream)) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
  }
  if (input_dim <= 0 || input_dim % QK8_1 != 0 || q8_bytes == 0 ||
      static_cast<std::uint64_t>(workspace_nbytes) < q8_bytes) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  const auto* q8 = static_cast<const block_q8_1*>(workspace);
  cudaError_t status = launch_matching_mmvq_one_row<GGML_TYPE_Q4_K>(
      descriptors, outputs, operation_count, q8, stream);
  if (status == cudaSuccess) {
    status = launch_matching_mmvq_one_row<GGML_TYPE_Q5_K>(
        descriptors, outputs, operation_count, q8, stream);
  }
  if (status == cudaSuccess) {
    status = launch_matching_mmvq_one_row<GGML_TYPE_Q6_K>(
        descriptors, outputs, operation_count, q8, stream);
  }
  return static_cast<int>(status);
}

int run_mmvq_multi_rows(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    const float* input,
    const float* auxiliary,
    bool apply_swiglu,
    float* const* outputs,
    std::int64_t operation_count,
    std::int64_t rows,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream) {
  if (descriptors == nullptr || input == nullptr || outputs == nullptr ||
      operation_count < 1 || operation_count > kMaxOperations ||
      rows < 2 || rows > 16 || workspace == nullptr ||
      workspace_nbytes <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::int64_t input_dim = descriptors[0] == nullptr
      ? 0 : descriptors[0]->input_dim;
  if (input_dim <= 0 || input_dim % QK8_1 != 0 ||
      static_cast<std::uint64_t>(rows) >
          std::numeric_limits<std::size_t>::max() /
              (static_cast<std::size_t>(input_dim / QK8_1) *
               sizeof(block_q8_1))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr std::int64_t max_input_rows_per_block =
      NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q4 >
              NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q5
          ? (NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q4 >
                     NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q6
                 ? NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q4
                 : NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q6)
          : (NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q5 >
                     NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q6
                 ? NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q5
                 : NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q6);
  constexpr std::int64_t padded_row_tile = max_input_rows_per_block *
      NFN_GLIMMER_MMVQ_ROW_GROUPS_PER_BLOCK;
  const std::int64_t padded_rows =
      ((rows + padded_row_tile - 1) / padded_row_tile) * padded_row_tile;
  const std::size_t q8_bytes = static_cast<std::size_t>(padded_rows) *
      static_cast<std::size_t>(input_dim / QK8_1) * sizeof(block_q8_1);
  std::size_t q8_sum_bytes = 0;
#if NFN_GLIMMER_MMVQ_PRECOMPUTED_QSUM
  q8_sum_bytes = static_cast<std::size_t>(padded_rows) *
      static_cast<std::size_t>(input_dim / QK8_1) * 4 *
      sizeof(std::int16_t);
#endif
  for (std::int64_t index = 0; index < operation_count; ++index) {
    if (outputs[index] == nullptr ||
        !validate_descriptor(descriptors[index], input_dim, cuda_stream)) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
  }
  if ((apply_swiglu && auxiliary == nullptr) || q8_bytes == 0 ||
      q8_bytes > std::numeric_limits<std::size_t>::max() - q8_sum_bytes ||
      static_cast<std::uint64_t>(workspace_nbytes) <
          q8_bytes + q8_sum_bytes) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  block_q8_1* q8 = static_cast<block_q8_1*>(workspace);
  std::int16_t* q8_sums = nullptr;
#if NFN_GLIMMER_MMVQ_PRECOMPUTED_QSUM
  q8_sums = reinterpret_cast<std::int16_t*>(
      static_cast<std::uint8_t*>(workspace) + q8_bytes);
#endif
  constexpr int quant_threads = 256;
  const dim3 quant_grid(
      static_cast<unsigned int>(
          (input_dim + quant_threads - 1) / quant_threads),
      static_cast<unsigned int>(padded_rows), 1);
  if (auxiliary != nullptr && apply_swiglu) {
    quantize_q8_1_mmvq_rows<true, true><<<
        quant_grid, quant_threads, 0, stream>>>(
        input, auxiliary, q8, q8_sums, rows, input_dim);
  } else if (auxiliary != nullptr) {
    quantize_q8_1_mmvq_rows<true, false><<<
        quant_grid, quant_threads, 0, stream>>>(
        input, auxiliary, q8, q8_sums, rows, input_dim);
  } else {
    quantize_q8_1_mmvq_rows<false, false><<<
        quant_grid, quant_threads, 0, stream>>>(
        input, nullptr, q8, q8_sums, rows, input_dim);
  }
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) return static_cast<int>(status);
  status = launch_matching_mmvq_rows<GGML_TYPE_Q4_K>(
      descriptors, outputs, operation_count, q8, q8_sums, rows, stream);
  if (status == cudaSuccess) {
    status = launch_matching_mmvq_rows<GGML_TYPE_Q5_K>(
        descriptors, outputs, operation_count, q8, q8_sums, rows, stream);
  }
  if (status == cudaSuccess) {
    status = launch_matching_mmvq_rows<GGML_TYPE_Q6_K>(
        descriptors, outputs, operation_count, q8, q8_sums, rows, stream);
  }
  return static_cast<int>(status);
}

int run_mmvq_multi_rows_prequantized(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    float* const* outputs,
    std::int64_t operation_count,
    std::int64_t rows,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream) {
  if (descriptors == nullptr || outputs == nullptr ||
      operation_count < 1 || operation_count > kMaxOperations ||
      rows < 2 || rows > 16 || workspace == nullptr ||
      workspace_nbytes <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::int64_t input_dim = descriptors[0] == nullptr
      ? 0 : descriptors[0]->input_dim;
  if (input_dim <= 0 || input_dim % QK8_1 != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr std::int64_t max_input_rows_per_block =
      NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q4 >
              NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q5
          ? (NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q4 >
                     NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q6
                 ? NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q4
                 : NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q6)
          : (NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q5 >
                     NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q6
                 ? NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q5
                 : NFN_GLIMMER_MMVQ_INPUT_ROWS_PER_BLOCK_Q6);
  constexpr std::int64_t padded_row_tile = max_input_rows_per_block *
      NFN_GLIMMER_MMVQ_ROW_GROUPS_PER_BLOCK;
  const std::int64_t padded_rows =
      ((rows + padded_row_tile - 1) / padded_row_tile) * padded_row_tile;
  const std::size_t q8_bytes = static_cast<std::size_t>(padded_rows) *
      static_cast<std::size_t>(input_dim / QK8_1) * sizeof(block_q8_1);
  std::size_t q8_sum_bytes = 0;
#if NFN_GLIMMER_MMVQ_PRECOMPUTED_QSUM
  q8_sum_bytes = static_cast<std::size_t>(padded_rows) *
      static_cast<std::size_t>(input_dim / QK8_1) * 4 *
      sizeof(std::int16_t);
#endif
  for (std::int64_t index = 0; index < operation_count; ++index) {
    if (outputs[index] == nullptr ||
        !validate_descriptor(descriptors[index], input_dim, cuda_stream)) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
  }
  if (q8_bytes == 0 ||
      q8_bytes > std::numeric_limits<std::size_t>::max() - q8_sum_bytes ||
      static_cast<std::uint64_t>(workspace_nbytes) <
          q8_bytes + q8_sum_bytes) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto* q8 = static_cast<const block_q8_1*>(workspace);
  const std::int16_t* q8_sums = nullptr;
#if NFN_GLIMMER_MMVQ_PRECOMPUTED_QSUM
  q8_sums = reinterpret_cast<const std::int16_t*>(
      static_cast<const std::uint8_t*>(workspace) + q8_bytes);
#endif
  const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  cudaError_t status = launch_matching_mmvq_rows<GGML_TYPE_Q4_K>(
      descriptors, outputs, operation_count, q8, q8_sums, rows, stream);
  if (status == cudaSuccess) {
    status = launch_matching_mmvq_rows<GGML_TYPE_Q5_K>(
        descriptors, outputs, operation_count, q8, q8_sums, rows, stream);
  }
  if (status == cudaSuccess) {
    status = launch_matching_mmvq_rows<GGML_TYPE_Q6_K>(
        descriptors, outputs, operation_count, q8, q8_sums, rows, stream);
  }
  return static_cast<int>(status);
}

constexpr int kSplitAttentionKeysPerTile = 8;
constexpr int kSplitAttentionMaxKeys = 128;

#ifndef NFN_GLIMMER_BATCH_DUAL_RMS_BLOCKS_PER_ROW
#define NFN_GLIMMER_BATCH_DUAL_RMS_BLOCKS_PER_ROW 8
#endif

static_assert(
    NFN_GLIMMER_BATCH_DUAL_RMS_BLOCKS_PER_ROW == 4 ||
        NFN_GLIMMER_BATCH_DUAL_RMS_BLOCKS_PER_ROW == 8,
    "batched dual RMS candidate supports four or eight blocks per row");

// Batched counterpart of the accepted one-row cooperative norm. Every block
// repeats the exact 256-lane accumulation and reduction tree for its row; only
// independent elementwise writes are partitioned. Thus additional occupancy
// cannot perturb the strict normalized values.
__global__ void glimmer_batch_dual_rms_cooperative_float32_v1_kernel(
    const float* __restrict__ input,
    NfnNativeTilePackedWeightDescriptorV1 first_weight,
    bool has_first_weight,
    const float* __restrict__ residual_input,
    float* __restrict__ hidden_output,
    NfnNativeTilePackedWeightDescriptorV1 second_weight,
    bool has_second_weight,
    float* __restrict__ normalized_output,
    float* __restrict__ residual_output,
    std::int64_t rows,
    std::int64_t width,
    float first_eps,
    bool first_centered,
    float second_eps,
    bool second_centered) {
  constexpr int blocks_per_row =
      NFN_GLIMMER_BATCH_DUAL_RMS_BLOCKS_PER_ROW;
  __shared__ float partials[256];
  __shared__ float inverse_rms;
  const std::int64_t row = blockIdx.x / blocks_per_row;
  const int row_block = blockIdx.x % blocks_per_row;
  if (row >= rows) return;
  const std::int64_t row_base = row * width;
  float square_sum = 0.0f;
  for (std::int64_t col = threadIdx.x; col < width; col += blockDim.x) {
    const float value = input[row_base + col];
    square_sum = fmaf(value, value, square_sum);
  }
  partials[threadIdx.x] = square_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partials[threadIdx.x] += partials[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    inverse_rms = rsqrtf(
        partials[0] / static_cast<float>(width) + first_eps);
  }
  __syncthreads();
  const float* first_values =
      reinterpret_cast<const float*>(first_weight.data);
  for (std::int64_t col =
           static_cast<std::int64_t>(row_block) * blockDim.x + threadIdx.x;
       col < width;
       col += static_cast<std::int64_t>(blocks_per_row) * blockDim.x) {
    float scale = has_first_weight ? first_values[col] : 1.0f;
    if (has_first_weight && first_centered) scale += 1.0f;
    const std::int64_t index = row_base + col;
    const float normalized = input[index] * inverse_rms * scale;
    const float hidden = __fadd_rn(normalized, residual_input[index]);
    hidden_output[index] = hidden;
    residual_output[index] = hidden;
  }

  cg::this_grid().sync();

  square_sum = 0.0f;
  for (std::int64_t col = threadIdx.x; col < width; col += blockDim.x) {
    const float value = hidden_output[row_base + col];
    square_sum = fmaf(value, value, square_sum);
  }
  partials[threadIdx.x] = square_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partials[threadIdx.x] += partials[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    inverse_rms = rsqrtf(
        partials[0] / static_cast<float>(width) + second_eps);
  }
  __syncthreads();
  const float* second_values =
      reinterpret_cast<const float*>(second_weight.data);
  for (std::int64_t col =
           static_cast<std::int64_t>(row_block) * blockDim.x + threadIdx.x;
       col < width;
       col += static_cast<std::int64_t>(blocks_per_row) * blockDim.x) {
    float scale = has_second_weight ? second_values[col] : 1.0f;
    if (has_second_weight && second_centered) scale += 1.0f;
    const std::int64_t index = row_base + col;
    normalized_output[index] = hidden_output[index] * inverse_rms * scale;
  }
}

__device__ __forceinline__ void split_attention_key_bounds(
    const NfnNativeTileDFlashBlockAttentionDescriptorV1& descriptor,
    std::int64_t query_row,
    std::int64_t* first_key_position,
    std::int64_t* last_key_position) {
  const std::int64_t query_position = descriptor.context_length + query_row;
  const bool causal =
      (descriptor.flags & NFN_NATIVE_TILE_BLOCK_ATTENTION_CAUSAL) != 0;
  *first_key_position = max(
      static_cast<std::int64_t>(0),
      query_position - descriptor.sliding_window + (causal ? 1 : 0));
  *last_key_position = causal
      ? query_position
      : min(descriptor.context_length + descriptor.block_rows - 1,
            query_position + descriptor.sliding_window);
}

// Exact short-context attention split. The score CTA preserves the ordinary
// kernel's one-warp-per-key FMA and shuffle tree. It exposes independent
// eight-key tiles as CTAs so a five-row target tail is not limited to 160
// blocks on a 170-SM device.
__global__ void dflash_block_attention_scores_short_v1_kernel(
    NfnNativeTileDFlashBlockAttentionDescriptorV1 descriptor,
    float* __restrict__ scores,
    int score_tiles) {
  const std::int64_t flat_tile = blockIdx.x;
  const std::int64_t flat_head = flat_tile / score_tiles;
  const int tile = static_cast<int>(flat_tile % score_tiles);
  const std::int64_t query_row = flat_head / descriptor.query_heads;
  const std::int64_t query_head = flat_head % descriptor.query_heads;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (query_row >= descriptor.query_rows) return;
  std::int64_t first_key_position = 0;
  std::int64_t last_key_position = -1;
  split_attention_key_bounds(
      descriptor, query_row, &first_key_position, &last_key_position);
  const std::int64_t key_position = first_key_position +
      static_cast<std::int64_t>(tile) * kSplitAttentionKeysPerTile + warp;
  float dot = 0.0f;
  if (key_position <= last_key_position) {
    const std::int64_t kv_head =
        query_head * descriptor.kv_heads / descriptor.query_heads;
    const float* query = descriptor.query +
        (query_row * descriptor.query_heads + query_head) *
            descriptor.head_dim;
    const bool block_row = key_position >= descriptor.context_length;
    const std::int64_t source_row = block_row
        ? key_position - descriptor.context_length
        : key_position % descriptor.cache_capacity;
    const bool causal =
        (descriptor.flags & NFN_NATIVE_TILE_BLOCK_ATTENTION_CAUSAL) != 0;
    for (std::int64_t component = lane;
         component < descriptor.head_dim; component += 32) {
      const std::int64_t source_index = block_row
          ? (source_row * descriptor.kv_heads + kv_head) *
                descriptor.head_dim + component
          : source_row * descriptor.cache_row_stride +
                kv_head * descriptor.head_dim + component;
      float key_value = block_row
          ? descriptor.block_key[source_index]
          : __bfloat162float(
                reinterpret_cast<const __nv_bfloat16*>(
                    descriptor.key_cache_bf16)[source_index]);
      if (causal && block_row && source_row < query_row) {
        key_value = __bfloat162float(__float2bfloat16(key_value));
      }
      dot = fmaf(query[component], key_value, dot);
    }
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    dot += __shfl_down_sync(0xffffffffu, dot, offset);
  }
  if (lane == 0) {
    scores[(flat_head * score_tiles + tile) *
               kSplitAttentionKeysPerTile + warp] =
        key_position <= last_key_position
            ? dot * descriptor.scale
            : -CUDART_INF_F;
  }
}

__global__ void dflash_block_attention_values_short_v1_kernel(
    NfnNativeTileDFlashBlockAttentionDescriptorV1 descriptor,
    const float* __restrict__ scores,
    int score_tiles) {
  __shared__ float tile_scores[kSplitAttentionKeysPerTile];
  __shared__ float tile_weights[kSplitAttentionKeysPerTile];
  __shared__ float shared_maximum;
  __shared__ float shared_denominator;
  __shared__ float shared_alpha;
  const std::int64_t flat_head = blockIdx.x;
  const std::int64_t query_row = flat_head / descriptor.query_heads;
  const std::int64_t query_head = flat_head % descriptor.query_heads;
  const std::int64_t dim = threadIdx.x;
  if (query_row >= descriptor.query_rows) return;
  const std::int64_t kv_head =
      query_head * descriptor.kv_heads / descriptor.query_heads;
  std::int64_t first_key_position = 0;
  std::int64_t last_key_position = -1;
  split_attention_key_bounds(
      descriptor, query_row, &first_key_position, &last_key_position);
  const bool causal =
      (descriptor.flags & NFN_NATIVE_TILE_BLOCK_ATTENTION_CAUSAL) != 0;
  float accumulated = 0.0f;
  if (threadIdx.x == 0) {
    shared_maximum = -CUDART_INF_F;
    shared_denominator = 0.0f;
  }
  __syncthreads();
  for (int tile = 0; tile < score_tiles; ++tile) {
    const std::int64_t tile_begin = first_key_position +
        static_cast<std::int64_t>(tile) * kSplitAttentionKeysPerTile;
    if (tile_begin > last_key_position) break;
    if (threadIdx.x < kSplitAttentionKeysPerTile) {
      tile_scores[threadIdx.x] = scores[
          (flat_head * score_tiles + tile) *
              kSplitAttentionKeysPerTile + threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float tile_maximum = -CUDART_INF_F;
      for (int key = 0; key < kSplitAttentionKeysPerTile; ++key) {
        tile_maximum = fmaxf(tile_maximum, tile_scores[key]);
      }
      const float next_maximum = fmaxf(shared_maximum, tile_maximum);
      shared_alpha = expf(shared_maximum - next_maximum);
      float tile_denominator = 0.0f;
      for (int key = 0; key < kSplitAttentionKeysPerTile; ++key) {
        const float weight = tile_scores[key] == -CUDART_INF_F
            ? 0.0f : expf(tile_scores[key] - next_maximum);
        tile_weights[key] = weight;
        tile_denominator += weight;
      }
      shared_denominator =
          shared_denominator * shared_alpha + tile_denominator;
      shared_maximum = next_maximum;
    }
    __syncthreads();
    if (dim < descriptor.head_dim) {
      accumulated *= shared_alpha;
      for (int key = 0; key < kSplitAttentionKeysPerTile; ++key) {
        const std::int64_t key_position = tile_begin + key;
        if (key_position > last_key_position) break;
        const bool block_row = key_position >= descriptor.context_length;
        const std::int64_t source_row = block_row
            ? key_position - descriptor.context_length
            : key_position % descriptor.cache_capacity;
        const std::int64_t source_index = block_row
            ? (source_row * descriptor.kv_heads + kv_head) *
                  descriptor.head_dim + dim
            : source_row * descriptor.cache_row_stride +
                  kv_head * descriptor.head_dim + dim;
        float value = block_row
            ? descriptor.block_value[source_index]
            : __bfloat162float(
                  reinterpret_cast<const __nv_bfloat16*>(
                      descriptor.value_cache_bf16)[source_index]);
        if (causal && block_row && source_row < query_row) {
          value = __bfloat162float(__float2bfloat16(value));
        }
        accumulated = fmaf(tile_weights[key], value, accumulated);
      }
    }
    __syncthreads();
  }
  if (dim < descriptor.head_dim) {
    descriptor.output[
        (query_row * descriptor.query_heads + query_head) *
            descriptor.head_dim + dim] =
        accumulated / shared_denominator;
  }
}

template <ggml_type Type>
cudaError_t launch_matching_operations(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    float* const* outputs,
    std::int64_t operation_count,
    const block_q8_1_mmq* q8,
    std::int64_t rows,
    std::int64_t input_dim,
    float* fixup,
    std::size_t fixup_bytes,
    int multiprocessors,
    cudaStream_t stream) {
  for (std::int64_t index = 0; index < operation_count; ++index) {
    const NfnNativeTilePackedWeightDescriptorV1& descriptor = *descriptors[index];
    if (descriptor.encoding != static_cast<std::uint32_t>(Type)) continue;
    const cudaError_t status = dispatch_rows<Type>(
        descriptor.data, q8, outputs[index], rows, input_dim,
        descriptor.output_dim, descriptor.row_stride_bytes, fixup, fixup_bytes,
        multiprocessors, stream);
    if (status != cudaSuccess) return status;
  }

  return cudaSuccess;
}

template <bool StoreSums>
cudaError_t quantize_input(
    const float* input,
    const float* auxiliary,
    bool apply_swiglu,
    block_q8_1_mmq* q8,
    std::int64_t rows,
    std::int64_t input_dim,
    cudaStream_t stream) {
  const dim3 grid(
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>((input_dim + 4 * kQuantThreads - 1) /
                                (4 * kQuantThreads)),
      1);
  if (auxiliary != nullptr && apply_swiglu) {
    quantize_q8_1_mmq<StoreSums, true, true>
        <<<grid, kQuantThreads, 0, stream>>>(
            input, auxiliary, q8, rows, input_dim);
  } else if (auxiliary != nullptr) {
    quantize_q8_1_mmq<StoreSums, true, false>
        <<<grid, kQuantThreads, 0, stream>>>(
            input, auxiliary, q8, rows, input_dim);
  } else {
    quantize_q8_1_mmq<StoreSums, false, false>
        <<<grid, kQuantThreads, 0, stream>>>(
            input, nullptr, q8, rows, input_dim);
  }
  return cudaPeekAtLastError();
}

int run_multi_linear(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    const float* input,
    const float* auxiliary,
    bool apply_swiglu,
    float* const* outputs,
    std::int64_t operation_count,
    std::int64_t rows,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream) {
  if (descriptors == nullptr || input == nullptr || outputs == nullptr ||
      operation_count < 1 || operation_count > kMaxOperations ||
      workspace == nullptr || workspace_nbytes <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::int64_t input_dim = descriptors[0] == nullptr
      ? 0 : descriptors[0]->input_dim;
  for (std::int64_t index = 0; index < operation_count; ++index) {
    if (outputs[index] == nullptr ||
        !validate_descriptor(descriptors[index], input_dim, cuda_stream)) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
  }

  if (rows == 1) {
    return run_mmvq_multi_one_row(
        descriptors, input, auxiliary, apply_swiglu, outputs,
        operation_count, workspace, workspace_nbytes, cuda_stream);
  }
  int multiprocessors = 0;
  std::size_t q8_bytes = 0;
  std::size_t required_bytes = 0;
  if (!query_multiprocessors(&multiprocessors) ||
      !workspace_layout(
          rows, input_dim, multiprocessors, &q8_bytes, &required_bytes) ||
      static_cast<std::uint64_t>(workspace_nbytes) < required_bytes) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  bool has_q4_or_q5 = false;
  bool has_q6 = false;
  for (std::int64_t index = 0; index < operation_count; ++index) {
    has_q4_or_q5 = has_q4_or_q5 ||
        descriptors[index]->encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K ||
        descriptors[index]->encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K;
    has_q6 = has_q6 ||
        descriptors[index]->encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q6_K;
  }

  const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  block_q8_1_mmq* q8 = static_cast<block_q8_1_mmq*>(workspace);
  float* fixup = reinterpret_cast<float*>(
      static_cast<std::uint8_t*>(workspace) + q8_bytes);
  const std::size_t fixup_bytes = required_bytes - q8_bytes;
  cudaError_t status = cudaSuccess;
  if (has_q4_or_q5) {
    status = quantize_input<true>(
        input, auxiliary, apply_swiglu, q8, rows, input_dim, stream);
  }
  if (has_q4_or_q5) {
    if (status == cudaSuccess) {
      status = launch_matching_operations<GGML_TYPE_Q4_K>(
          descriptors, outputs, operation_count, q8, rows, input_dim, fixup,
          fixup_bytes, multiprocessors, stream);
    }
    if (status == cudaSuccess) {
      status = launch_matching_operations<GGML_TYPE_Q5_K>(
          descriptors, outputs, operation_count, q8, rows, input_dim, fixup,
          fixup_bytes, multiprocessors, stream);
    }
  }
  if (status == cudaSuccess && has_q6) {
    status = quantize_input<false>(
        input, auxiliary, apply_swiglu, q8, rows, input_dim, stream);
    if (status == cudaSuccess) {
      status = launch_matching_operations<GGML_TYPE_Q6_K>(
          descriptors, outputs, operation_count, q8, rows, input_dim, fixup,
          fixup_bytes, multiprocessors, stream);
    }
  }
  return static_cast<int>(status);
}

}  // namespace

extern "C" int nfn_native_tile_k_quant_mmq_abi_version() {
  return NFN_NATIVE_TILE_K_QUANT_MMQ_V1;
}

extern "C" int
nfn_native_tile_glimmer_dual_rms_add_capture_cooperative_batch_float32_v1(
    const float* input,
    const NfnNativeTilePackedWeightDescriptorV1* first_weight,
    const float* residual_input,
    float* hidden_output,
    const NfnNativeTilePackedWeightDescriptorV1* second_weight,
    float* normalized_output,
    float* residual_output,
    std::int64_t rows,
    std::int64_t width,
    float first_eps,
    bool first_centered,
    float second_eps,
    bool second_centered,
    void* cuda_stream) {
  const auto valid_norm = [width](
      const NfnNativeTilePackedWeightDescriptorV1* weight) {
    return weight == nullptr ||
        (weight->struct_size >=
             sizeof(NfnNativeTilePackedWeightDescriptorV1) &&
         weight->version == NFN_NATIVE_TILE_PACKED_WEIGHT_V1 &&
         weight->encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_F32 &&
         weight->flags == 0 && weight->reserved0 == 0 &&
         weight->reserved1 == 0 && weight->data != nullptr &&
         weight->output_dim == 1 && weight->input_dim == width &&
         weight->row_stride_bytes == width *
             static_cast<std::int64_t>(sizeof(float)) &&
         weight->data_nbytes == width *
             static_cast<std::int64_t>(sizeof(float)));
  };
  if (input == nullptr || residual_input == nullptr ||
      hidden_output == nullptr || normalized_output == nullptr ||
      residual_output == nullptr || rows < 2 || rows > 16 ||
      width <= 0 || width > 65536 || !std::isfinite(first_eps) ||
      !(first_eps > 0.0f) || !std::isfinite(second_eps) ||
      !(second_eps > 0.0f) || cuda_stream == nullptr ||
      !valid_norm(first_weight) || !valid_norm(second_weight)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  NfnNativeTilePackedWeightDescriptorV1 first{};
  NfnNativeTilePackedWeightDescriptorV1 second{};
  bool has_first = first_weight != nullptr;
  bool has_second = second_weight != nullptr;
  if (has_first) first = *first_weight;
  if (has_second) second = *second_weight;
  const float* input_arg = input;
  const float* residual_input_arg = residual_input;
  float* hidden_output_arg = hidden_output;
  float* normalized_output_arg = normalized_output;
  float* residual_output_arg = residual_output;
  void* args[] = {
      &input_arg,
      &first,
      &has_first,
      &residual_input_arg,
      &hidden_output_arg,
      &second,
      &has_second,
      &normalized_output_arg,
      &residual_output_arg,
      &rows,
      &width,
      &first_eps,
      &first_centered,
      &second_eps,
      &second_centered,
  };
  constexpr int blocks_per_row =
      NFN_GLIMMER_BATCH_DUAL_RMS_BLOCKS_PER_ROW;
  const cudaError_t status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(
          glimmer_batch_dual_rms_cooperative_float32_v1_kernel),
      static_cast<unsigned int>(rows * blocks_per_row), 256, args, 0,
      static_cast<cudaStream_t>(cuda_stream));
  return static_cast<int>(
      status == cudaSuccess ? cudaPeekAtLastError() : status);
}

extern "C" int
nfn_native_tile_dflash_block_attention_short_split_float32_v1(
    const NfnNativeTileDFlashBlockAttentionDescriptorV1* descriptor,
    float* score_workspace,
    std::int64_t score_workspace_nbytes) {
  if (descriptor == nullptr || score_workspace == nullptr ||
      descriptor->struct_size <
          sizeof(NfnNativeTileDFlashBlockAttentionDescriptorV1) ||
      descriptor->version != NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1 ||
      descriptor->reserved0 != 0 || descriptor->reserved1 != 0 ||
      descriptor->query == nullptr || descriptor->block_key == nullptr ||
      descriptor->block_value == nullptr ||
      descriptor->key_cache_bf16 == nullptr ||
      descriptor->value_cache_bf16 == nullptr ||
      descriptor->output == nullptr || descriptor->query_rows <= 0 ||
      descriptor->query_rows > 16 || descriptor->block_rows <= 0 ||
      descriptor->query_heads != 32 || descriptor->kv_heads <= 0 ||
      descriptor->query_heads % descriptor->kv_heads != 0 ||
      descriptor->head_dim != 128 || descriptor->context_length < 0 ||
      descriptor->sliding_window <= 0 || descriptor->cache_capacity <= 0 ||
      descriptor->cache_row_stride <
          descriptor->kv_heads * descriptor->head_dim ||
      !isfinite(descriptor->scale) || !(descriptor->scale > 0.0f) ||
      descriptor->cuda_stream == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::int64_t maximum_keys = min(
      descriptor->sliding_window,
      descriptor->context_length + descriptor->block_rows);
  if (maximum_keys <= 0 || maximum_keys > kSplitAttentionMaxKeys) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int score_tiles = static_cast<int>(
      (maximum_keys + kSplitAttentionKeysPerTile - 1) /
      kSplitAttentionKeysPerTile);
  const std::int64_t score_count = descriptor->query_rows *
      descriptor->query_heads * score_tiles * kSplitAttentionKeysPerTile;
  if (score_count <= 0 || score_count >
          std::numeric_limits<std::int64_t>::max() /
              static_cast<std::int64_t>(sizeof(float)) ||
      score_workspace_nbytes <
          score_count * static_cast<std::int64_t>(sizeof(float))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(descriptor->cuda_stream);
  const std::int64_t flat_heads =
      descriptor->query_rows * descriptor->query_heads;
  dflash_block_attention_scores_short_v1_kernel<<<
      static_cast<unsigned int>(flat_heads * score_tiles), 256, 0, stream>>>(
      *descriptor, score_workspace, score_tiles);
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) return static_cast<int>(status);
  dflash_block_attention_values_short_v1_kernel<<<
      static_cast<unsigned int>(flat_heads), 256, 0, stream>>>(
      *descriptor, score_workspace, score_tiles);
  return static_cast<int>(cudaPeekAtLastError());
}

extern "C" std::int64_t nfn_native_tile_k_quant_mmq_workspace_bytes_v1(
    std::int64_t rows,
    std::int64_t input_dim) {
  int multiprocessors = 0;
  std::size_t q8_bytes = 0;
  std::size_t total_bytes = 0;
  if (!query_multiprocessors(&multiprocessors) ||
      !workspace_layout(
          rows, input_dim, multiprocessors, &q8_bytes, &total_bytes) ||
      total_bytes > static_cast<std::size_t>(
          std::numeric_limits<std::int64_t>::max())) {
    return 0;
  }
  return static_cast<std::int64_t>(total_bytes);
}

extern "C" int nfn_native_tile_k_quant_mmq_multi_linear_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    const float* input,
    float* const* outputs,
    std::int64_t operation_count,
    std::int64_t rows,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream) {
  return run_multi_linear(
      descriptors, input, nullptr, false, outputs, operation_count, rows,
      workspace, workspace_nbytes, cuda_stream);
}

extern "C" int nfn_native_tile_k_quant_mmq_multi_linear_gated_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    const float* input,
    const float* gate,
    float* const* outputs,
    std::int64_t operation_count,
    std::int64_t rows,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream) {
  if (gate == nullptr) return static_cast<int>(cudaErrorInvalidValue);
  return run_multi_linear(
      descriptors, input, gate, false, outputs, operation_count, rows,
      workspace, workspace_nbytes, cuda_stream);
}

extern "C" int nfn_native_tile_k_quant_mmq_multi_linear_swiglu_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    const float* gate,
    const float* up,
    float* const* outputs,
    std::int64_t operation_count,
    std::int64_t rows,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream) {
  if (gate == nullptr || up == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return run_multi_linear(
      descriptors, gate, up, true, outputs, operation_count, rows, workspace,
      workspace_nbytes, cuda_stream);
}

extern "C" int nfn_native_tile_k_quant_mmvq_multi_linear_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    const float* input,
    float* const* outputs,
    std::int64_t operation_count,
    std::int64_t rows,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream) {
  if (rows >= NFN_GLIMMER_MMVQ_USE_MMQ_MIN_ROWS) {
    return run_multi_linear(
        descriptors, input, nullptr, false, outputs, operation_count, rows,
        workspace, workspace_nbytes, cuda_stream);
  }
  return run_mmvq_multi_rows(
      descriptors, input, nullptr, false, outputs, operation_count, rows,
      workspace, workspace_nbytes, cuda_stream);
}

extern "C" int
nfn_native_tile_k_quant_mmvq_multi_linear_prequantized_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    float* const* outputs,
    std::int64_t operation_count,
    std::int64_t rows,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream) {
  if (rows == 1) {
    return run_mmvq_multi_one_row_prequantized(
        descriptors, outputs, operation_count, workspace, workspace_nbytes,
        cuda_stream);
  }
  return run_mmvq_multi_rows_prequantized(
      descriptors, outputs, operation_count, rows, workspace,
      workspace_nbytes, cuda_stream);
}

extern "C" int nfn_native_tile_k_quant_mmvq_multi_linear_gated_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    const float* input,
    const float* gate,
    float* const* outputs,
    std::int64_t operation_count,
    std::int64_t rows,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream) {
  if (gate == nullptr) return static_cast<int>(cudaErrorInvalidValue);
  if (rows >= NFN_GLIMMER_MMVQ_USE_MMQ_MIN_ROWS) {
    return run_multi_linear(
        descriptors, input, gate, false, outputs, operation_count, rows,
        workspace, workspace_nbytes, cuda_stream);
  }
  return run_mmvq_multi_rows(
      descriptors, input, gate, false, outputs, operation_count, rows,
      workspace, workspace_nbytes, cuda_stream);
}

extern "C" int nfn_native_tile_k_quant_mmvq_multi_linear_swiglu_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    const float* gate,
    const float* up,
    float* const* outputs,
    std::int64_t operation_count,
    std::int64_t rows,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream) {
  if (gate == nullptr || up == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (rows >= NFN_GLIMMER_MMVQ_USE_MMQ_MIN_ROWS) {
    return run_multi_linear(
        descriptors, gate, up, true, outputs, operation_count, rows,
        workspace, workspace_nbytes, cuda_stream);
  }
  return run_mmvq_multi_rows(
      descriptors, gate, up, true, outputs, operation_count, rows, workspace,
      workspace_nbytes, cuda_stream);
}

extern "C" int nfn_native_tile_k_quant_mmvq_linear_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* descriptor,
    const float* input,
    float* output,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream) {
  const NfnNativeTilePackedWeightDescriptorV1* descriptors[]{descriptor};
  float* outputs[]{output};
  return run_mmvq_multi_one_row(
      descriptors, input, nullptr, false, outputs, 1, workspace,
      workspace_nbytes, cuda_stream);
}
