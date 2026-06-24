#include <cuda_tile.h>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
#include "llmc/tk/attention_sm120.cuh"
#include "llmc/matmul.cuh"
#endif
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
#include <cublasLt.h>
#include <cublas_v2.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

namespace neuralfn::tile_cuda {

#ifndef NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE
#define NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE 4096
#endif

static_assert(
    NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE == 1024 ||
        NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE == 2048 ||
        NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE == 4096 ||
        NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE == 8192,
    "NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE must be 1024, 2048, 4096, or 8192");

#ifndef NFN_TILE_CUDA_OPTIMIZER_TILE_SIZE
#define NFN_TILE_CUDA_OPTIMIZER_TILE_SIZE 1024
#endif

static_assert(
    NFN_TILE_CUDA_OPTIMIZER_TILE_SIZE == 1024 ||
        NFN_TILE_CUDA_OPTIMIZER_TILE_SIZE == 2048 ||
        NFN_TILE_CUDA_OPTIMIZER_TILE_SIZE == 4096,
    "NFN_TILE_CUDA_OPTIMIZER_TILE_SIZE must be 1024, 2048, or 4096");

#if NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE == 1024
#define NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE 1024_ic
#elif NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE == 2048
#define NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE 2048_ic
#elif NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE == 4096
#define NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE 4096_ic
#elif NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE == 8192
#define NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE 8192_ic
#endif

#if NFN_TILE_CUDA_OPTIMIZER_TILE_SIZE == 1024
#define NFN_TILE_CUDA_OPTIMIZER_TILE_SHAPE 1024_ic
#elif NFN_TILE_CUDA_OPTIMIZER_TILE_SIZE == 2048
#define NFN_TILE_CUDA_OPTIMIZER_TILE_SHAPE 2048_ic
#elif NFN_TILE_CUDA_OPTIMIZER_TILE_SIZE == 4096
#define NFN_TILE_CUDA_OPTIMIZER_TILE_SHAPE 4096_ic
#endif

namespace {

constexpr int kTileSize = 1024;
constexpr int kOptimizerTileSize = NFN_TILE_CUDA_OPTIMIZER_TILE_SIZE;
constexpr int kAttentionValueChunkSize = 64;
constexpr int kGpt2AttentionHeads = 12;
constexpr int kGpt2AttentionHeadDim = 64;
constexpr int kGpt2AttentionValueChunks = kGpt2AttentionHeadDim / kAttentionValueChunkSize;
constexpr std::int64_t kTkPackedAttentionBackwardDefaultMaxBatchPerLaunch = 64;
constexpr std::int64_t kLayerNormBackwardAffineDefaultRowChunkSize = 128;
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
std::atomic<std::int64_t> g_attention_forward_tk_launch_count{0};
std::atomic<std::int64_t> g_attention_backward_tk_launch_count{0};
std::atomic<std::int64_t> g_attention_backward_float_hd64_dprep_launch_count{0};
std::atomic<std::int64_t> g_attention_backward_dprep_timing_us{0};
std::atomic<std::int64_t> g_attention_backward_dprep_timing_count{0};
std::atomic<std::int64_t> g_attention_backward_tk_timing_us{0};
std::atomic<std::int64_t> g_attention_backward_tk_timing_count{0};
std::atomic<std::int64_t> g_attention_tk_workspace_allocation_count{0};
std::atomic<std::int64_t> g_attention_tk_workspace_element_capacity{0};
std::atomic<std::int64_t> g_attention_tk_workspace_row_capacity{0};
#endif
std::atomic<std::int64_t> g_attention_forward_row_launch_count{0};
std::atomic<std::int64_t> g_attention_forward_row_fallback_count{0};
std::atomic<std::int64_t> g_attention_forward_scalar_launch_count{0};
std::atomic<int> g_attention_forward_row_last_error{0};
std::atomic<int> g_attention_forward_row_prelaunch_clear_error{0};
std::atomic<int> g_attention_forward_row_prelaunch_peek_error{0};
std::atomic<std::int64_t> g_attention_forward_row_grid_x{0};
std::atomic<std::int64_t> g_attention_forward_row_grid_y{0};
std::atomic<std::int64_t> g_attention_forward_row_grid_z{0};
std::atomic<std::int64_t> g_attention_forward_row_block_x{0};
std::atomic<int> g_attention_forward_row_attr_status{-1};
std::atomic<int> g_attention_forward_row_attr_max_threads_per_block{0};
std::atomic<int> g_attention_forward_row_attr_num_regs{0};
std::atomic<std::int64_t> g_attention_forward_row_attr_shared_size_bytes{0};
std::atomic<std::int64_t> g_attention_forward_row_attr_const_size_bytes{0};
std::atomic<std::int64_t> g_attention_forward_row_attr_local_size_bytes{0};
std::atomic<std::int64_t> g_token_cross_entropy_workspace_allocation_count{0};
std::atomic<std::int64_t> g_token_cross_entropy_workspace_row_capacity{0};
std::atomic<std::int64_t> g_lm_head_classifier_chunk_launch_count{0};
std::atomic<std::int64_t> g_lm_head_classifier_last_rows{0};
std::atomic<std::int64_t> g_lm_head_classifier_last_vocab{0};
std::atomic<std::int64_t> g_lm_head_classifier_last_row_stride{0};
std::atomic<std::int64_t> g_lm_head_classifier_loss_bin_launch_count{0};
std::atomic<std::int64_t> g_bf16_to_f32_vec4_count{0};
struct TokenCrossEntropyWorkspace {
  float* row_max = nullptr;
  float* row_denom = nullptr;
  std::int64_t row_capacity = 0;
};
TokenCrossEntropyWorkspace g_token_cross_entropy_workspace;
std::mutex g_token_cross_entropy_workspace_mutex;
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
std::atomic<std::int64_t> g_linear_bf16_gemm_count{0};
std::atomic<std::int64_t> g_linear_bf16_gemm_fast16bf_request_count{0};
std::atomic<std::int64_t> g_linear_tk_gemm_count{0};
std::atomic<std::int64_t> g_linear_tk_float_out_gemm_count{0};
std::atomic<std::int64_t> g_linear_tk_dweight_gemm_count{0};
std::atomic<std::int64_t> g_linear_tk_dgelu_dinput_gemm_count{0};
std::atomic<std::int64_t> g_linear_cublaslt_gemm_count{0};
std::atomic<std::int64_t> g_linear_cublaslt_bgrad_gemm_count{0};
std::atomic<std::int64_t> g_linear_cublaslt_bgrad_direct_write_count{0};
std::atomic<std::int64_t> g_linear_cublaslt_bgrad_accumulate_count{0};
std::atomic<std::int64_t> g_linear_sgemm_count{0};
std::atomic<std::int64_t> g_linear_bf16_a_pack_count{0};
std::atomic<std::int64_t> g_linear_bf16_a_cache_hit_count{0};
std::atomic<std::int64_t> g_linear_bf16_cache_reset_count{0};
std::atomic<std::int64_t> g_linear_bf16_workspace_allocation_count{0};
std::atomic<std::int64_t> g_linear_bf16_workspace_a_capacity{0};
std::atomic<std::int64_t> g_linear_bf16_workspace_b_capacity{0};
std::atomic<std::int64_t> g_linear_bf16_cached_a_capacity{0};
std::atomic<std::int64_t> g_linear_bf16_cache_entry_count{0};
struct LinearShapeStat {
  int path = 0;
  int m = 0;
  int n = 0;
  int k = 0;
  int op_a = 0;
  int op_b = 0;
  int cublaslt_selected_heuristic = -1;
  int cublaslt_returned_heuristics = 0;
  std::int64_t cublaslt_workspace_bytes = 0;
  std::int64_t calls = 0;
  std::int64_t total_us = 0;
};
std::vector<LinearShapeStat> g_linear_shape_stats;
std::mutex g_linear_shape_stats_mutex;
bool trainer_linear_shape_stats_enabled();
void record_linear_shape_stat(
    int path,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    std::int64_t elapsed_us = 0,
    int cublaslt_selected_heuristic = -1,
    int cublaslt_returned_heuristics = 0,
    std::int64_t cublaslt_workspace_bytes = 0);
struct LinearShapeTiming {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaStream_t stream = nullptr;
  bool active = false;
};
LinearShapeTiming begin_linear_shape_timing(cudaStream_t stream);
std::int64_t finish_linear_shape_timing(LinearShapeTiming* timing);
void discard_linear_shape_timing(LinearShapeTiming* timing);
#endif
std::atomic<bool> g_attention_forward_row_launch_disabled{false};

std::int64_t layer_norm_backward_affine_row_chunk_size() {
  static const std::int64_t value = []() {
    const char* raw = std::getenv("NFN_TILE_CUDA_LAYERNORM_AFFINE_ROW_CHUNK_SIZE");
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE");
    }
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT2_LAYERNORM_AFFINE_ROW_CHUNK_SIZE");
    }
    if (raw == nullptr || raw[0] == '\0') {
      return kLayerNormBackwardAffineDefaultRowChunkSize;
    }
    char* end = nullptr;
    const long long parsed = std::strtoll(raw, &end, 10);
    if (end == raw || (end != nullptr && *end != '\0') || parsed <= 0) {
      return kLayerNormBackwardAffineDefaultRowChunkSize;
    }
    return static_cast<std::int64_t>(parsed);
  }();
  return value;
}

int cross_entropy_bf16_threads_per_row() {
  static const int value = []() {
    const char* raw = std::getenv("NFN_TILE_CUDA_CE_BF16_THREADS");
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT_CE_BF16_THREADS");
    }
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT2_CE_BF16_THREADS");
    }
    if (raw == nullptr || raw[0] == '\0') {
      return 1024;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || (end != nullptr && *end != '\0')) {
      return 1024;
    }
    switch (parsed) {
      case 128:
      case 256:
      case 512:
      case 1024:
        return static_cast<int>(parsed);
      default:
        return 1024;
    }
  }();
  return value;
}

int lm_head_prob_only_target_correction_threads_value() {
  static const int value = []() {
    const char* raw = std::getenv("NFN_TILE_CUDA_LM_HEAD_PROB_ONLY_TARGET_CORRECTION_THREADS");
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_TARGET_CORRECTION_THREADS");
    }
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT2_LM_HEAD_PROB_ONLY_TARGET_CORRECTION_THREADS");
    }
    if (raw == nullptr || raw[0] == '\0') {
      return 512;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || (end != nullptr && *end != '\0')) {
      return 512;
    }
    switch (parsed) {
      case 128:
      case 256:
      case 512:
      case 1024:
        return static_cast<int>(parsed);
      default:
        return 512;
    }
  }();
  return value;
}

bool cross_entropy_bf16_vec_stores_enabled() {
  static const bool value = []() {
    const char* raw = std::getenv("NFN_TILE_CUDA_CE_BF16_VEC_STORES");
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT_CE_BF16_VEC_STORES");
    }
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT2_CE_BF16_VEC_STORES");
    }
    return raw != nullptr &&
           (std::strcmp(raw, "1") == 0 || std::strcmp(raw, "true") == 0 ||
            std::strcmp(raw, "TRUE") == 0 || std::strcmp(raw, "on") == 0 ||
            std::strcmp(raw, "ON") == 0);
  }();
  return value;
}

bool cross_entropy_bf16_vec_normal_stores_enabled() {
  static const bool value = []() {
    const char* raw = std::getenv("NFN_TILE_CUDA_CE_BF16_VEC_NORMAL_STORES");
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT_CE_BF16_VEC_NORMAL_STORES");
    }
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT2_CE_BF16_VEC_NORMAL_STORES");
    }
    return raw != nullptr &&
           (std::strcmp(raw, "1") == 0 || std::strcmp(raw, "true") == 0 ||
            std::strcmp(raw, "TRUE") == 0 || std::strcmp(raw, "on") == 0 ||
            std::strcmp(raw, "ON") == 0);
  }();
  return value;
}

bool cross_entropy_bf16_scalar_streaming_stores_enabled() {
  static const bool value = []() {
    const char* raw = std::getenv("NFN_TILE_CUDA_CE_BF16_SCALAR_STREAMING_STORES");
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT_CE_BF16_SCALAR_STREAMING_STORES");
    }
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT2_CE_BF16_SCALAR_STREAMING_STORES");
    }
    return raw != nullptr &&
           (std::strcmp(raw, "1") == 0 || std::strcmp(raw, "true") == 0 ||
            std::strcmp(raw, "TRUE") == 0 || std::strcmp(raw, "on") == 0 ||
            std::strcmp(raw, "ON") == 0);
  }();
  return value;
}

bool cross_entropy_bf16_vec_loads_enabled() {
  static const bool value = []() {
    const char* raw = std::getenv("NFN_TILE_CUDA_CE_BF16_VEC_LOADS");
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT_CE_BF16_VEC_LOADS");
    }
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT2_CE_BF16_VEC_LOADS");
    }
    if (raw == nullptr || raw[0] == '\0') {
      return true;
    }
    if (std::strcmp(raw, "0") == 0 || std::strcmp(raw, "false") == 0 ||
        std::strcmp(raw, "FALSE") == 0 || std::strcmp(raw, "off") == 0 ||
        std::strcmp(raw, "OFF") == 0) {
      return false;
    }
    return std::strcmp(raw, "1") == 0 || std::strcmp(raw, "true") == 0 ||
           std::strcmp(raw, "TRUE") == 0 || std::strcmp(raw, "on") == 0 ||
           std::strcmp(raw, "ON") == 0;
  }();
  return value;
}

bool dim768_bf16_residual_add_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_DIM768_BF16_RESIDUAL_ADD");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT_DIM768_BF16_RESIDUAL_ADD");
    }
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT2_DIM768_BF16_RESIDUAL_ADD");
    }
    if (value == nullptr) {
      return true;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool cross_entropy_bf16_exp2_enabled() {
  static const bool value = []() {
    const char* raw = std::getenv("NFN_TILE_CUDA_CE_BF16_EXP2");
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT_CE_BF16_EXP2");
    }
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT2_CE_BF16_EXP2");
    }
    return raw != nullptr &&
           (std::strcmp(raw, "1") == 0 || std::strcmp(raw, "true") == 0 ||
            std::strcmp(raw, "TRUE") == 0 || std::strcmp(raw, "on") == 0 ||
            std::strcmp(raw, "ON") == 0);
  }();
  return value;
}

bool lm_head_ce_default_specialized_enabled() {
  static const bool value = []() {
    const char* raw = std::getenv("NFN_TILE_CUDA_LM_HEAD_CE_DEFAULT_SPECIALIZED");
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT_LM_HEAD_CE_DEFAULT_SPECIALIZED");
    }
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT2_LM_HEAD_CE_DEFAULT_SPECIALIZED");
    }
    return raw != nullptr &&
           (std::strcmp(raw, "1") == 0 || std::strcmp(raw, "true") == 0 ||
            std::strcmp(raw, "TRUE") == 0 || std::strcmp(raw, "on") == 0 ||
            std::strcmp(raw, "ON") == 0);
  }();
  return value;
}

bool lm_head_ce_no_loss_default_specialized_enabled() {
  static const bool value = []() {
    const char* raw = std::getenv("NFN_TILE_CUDA_LM_HEAD_CE_NO_LOSS_DEFAULT_SPECIALIZED");
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_DEFAULT_SPECIALIZED");
    }
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT2_LM_HEAD_CE_NO_LOSS_DEFAULT_SPECIALIZED");
    }
    if (raw == nullptr) {
      return true;
    }
    if (std::strcmp(raw, "0") == 0 || std::strcmp(raw, "false") == 0 ||
        std::strcmp(raw, "FALSE") == 0 || std::strcmp(raw, "off") == 0 ||
        std::strcmp(raw, "OFF") == 0) {
      return false;
    }
    return std::strcmp(raw, "1") == 0 || std::strcmp(raw, "true") == 0 ||
           std::strcmp(raw, "TRUE") == 0 || std::strcmp(raw, "on") == 0 ||
           std::strcmp(raw, "ON") == 0;
  }();
  return value;
}

bool lm_head_ce_llmk_style_specialized_enabled() {
  static const bool value = []() {
    const char* raw = std::getenv("NFN_TILE_CUDA_LM_HEAD_CE_LLMK_STYLE_SPECIALIZED");
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT_LM_HEAD_CE_LLMK_STYLE_SPECIALIZED");
    }
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT2_LM_HEAD_CE_LLMK_STYLE_SPECIALIZED");
    }
    return raw != nullptr &&
           (std::strcmp(raw, "1") == 0 || std::strcmp(raw, "true") == 0 ||
            std::strcmp(raw, "TRUE") == 0 || std::strcmp(raw, "on") == 0 ||
            std::strcmp(raw, "ON") == 0);
  }();
  return value;
}

bool lm_head_ce_no_loss_llmk_style_specialized_enabled() {
  static const bool value = []() {
    const char* raw = std::getenv("NFN_TILE_CUDA_LM_HEAD_CE_NO_LOSS_LLMK_STYLE_SPECIALIZED");
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_LLMK_STYLE_SPECIALIZED");
    }
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT2_LM_HEAD_CE_NO_LOSS_LLMK_STYLE_SPECIALIZED");
    }
    return raw != nullptr &&
           (std::strcmp(raw, "1") == 0 || std::strcmp(raw, "true") == 0 ||
            std::strcmp(raw, "TRUE") == 0 || std::strcmp(raw, "on") == 0 ||
            std::strcmp(raw, "ON") == 0);
  }();
  return value;
}

bool lm_head_ce_loss_bins_default_specialized_enabled() {
  static const bool value = []() {
    const char* raw = std::getenv("NFN_TILE_CUDA_LM_HEAD_CE_LOSS_BINS_DEFAULT_SPECIALIZED");
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT_LM_HEAD_CE_LOSS_BINS_DEFAULT_SPECIALIZED");
    }
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT2_LM_HEAD_CE_LOSS_BINS_DEFAULT_SPECIALIZED");
    }
    return raw != nullptr &&
           (std::strcmp(raw, "1") == 0 || std::strcmp(raw, "true") == 0 ||
            std::strcmp(raw, "TRUE") == 0 || std::strcmp(raw, "on") == 0 ||
            std::strcmp(raw, "ON") == 0);
  }();
  return value;
}

bool lm_head_ce_reverse_rows_enabled() {
  static const bool value = []() {
    const char* raw = std::getenv("NFN_TILE_CUDA_LM_HEAD_CE_REVERSE_ROWS");
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT_LM_HEAD_CE_REVERSE_ROWS");
    }
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT2_LM_HEAD_CE_REVERSE_ROWS");
    }
    if (raw == nullptr) {
      return true;
    }
    if (std::strcmp(raw, "0") == 0 || std::strcmp(raw, "false") == 0 ||
        std::strcmp(raw, "FALSE") == 0 || std::strcmp(raw, "off") == 0 ||
        std::strcmp(raw, "OFF") == 0 || std::strcmp(raw, "no") == 0 ||
        std::strcmp(raw, "NO") == 0) {
      return false;
    }
    return true;
  }();
  return value;
}

#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION) && defined(LLMK_SM120_USE_CUBLASLT_GEMM) && defined(KITTENS_SM120)
std::once_flag g_llmk_sm120_cublaslt_init_once;

void ensure_llmk_sm120_cublaslt_initialized() {
  std::call_once(g_llmk_sm120_cublaslt_init_once, []() {
    llmk::cublaslt_sm120::init();
  });
}
#else
void ensure_llmk_sm120_cublaslt_initialized() {}
#endif

__tile_global__ void fill_float32_kernel(
    float* __restrict__ values,
    std::int64_t n,
    float value);

__global__ void f32_to_bf16_kernel(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    std::int64_t n) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = __float2bfloat16(src[idx]);
  }
}

__global__ void f32_to_bf16_bits_kernel(
    const float* __restrict__ src,
    std::uint16_t* __restrict__ dst,
    std::int64_t n) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    const unsigned int bits = __float_as_uint(src[idx]);
    const unsigned int rounding_bias = ((bits >> 16) & 1u) + 0x7fffu;
    dst[idx] = static_cast<std::uint16_t>((bits + rounding_bias) >> 16);
  }
}

__device__ __forceinline__ std::uint16_t f32_bits_to_bf16_bits_device(unsigned int bits) {
  const unsigned int rounding_bias = ((bits >> 16) & 1u) + 0x7fffu;
  return static_cast<std::uint16_t>((bits + rounding_bias) >> 16);
}

__global__ void f32_to_bf16_bits_vec4_kernel(
    const float4* __restrict__ src,
    unsigned long long* __restrict__ dst,
    std::int64_t n4) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n4) {
    return;
  }
  const float4 value = src[idx];
  const std::uint16_t a = f32_bits_to_bf16_bits_device(__float_as_uint(value.x));
  const std::uint16_t b = f32_bits_to_bf16_bits_device(__float_as_uint(value.y));
  const std::uint16_t c = f32_bits_to_bf16_bits_device(__float_as_uint(value.z));
  const std::uint16_t d = f32_bits_to_bf16_bits_device(__float_as_uint(value.w));
  dst[idx] = static_cast<unsigned long long>(a) |
      (static_cast<unsigned long long>(b) << 16) |
      (static_cast<unsigned long long>(c) << 32) |
      (static_cast<unsigned long long>(d) << 48);
}

__global__ void bf16_bits_to_f32_kernel(
    const std::uint16_t* __restrict__ src,
    float* __restrict__ dst,
    std::int64_t n) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    const unsigned int bits = static_cast<unsigned int>(src[idx]) << 16;
    dst[idx] = __uint_as_float(bits);
  }
}

__device__ __forceinline__ std::uint16_t f32_to_bf16_bits_device(float value) {
  return f32_bits_to_bf16_bits_device(__float_as_uint(value));
}

__device__ __forceinline__ float bf16_bits_to_f32_device(std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16);
}

__global__ void bf16_bits_to_f32_vec4_kernel(
    const std::uint16_t* __restrict__ src,
    float* __restrict__ dst,
    std::int64_t n4) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n4) {
    return;
  }
  const std::int64_t base = idx * 4;
  dst[base] = bf16_bits_to_f32_device(src[base]);
  dst[base + 1] = bf16_bits_to_f32_device(src[base + 1]);
  dst[base + 2] = bf16_bits_to_f32_device(src[base + 2]);
  dst[base + 3] = bf16_bits_to_f32_device(src[base + 3]);
}

__global__ void bf16_bits_add_bias_inplace_kernel(
    std::uint16_t* __restrict__ values,
    const float* __restrict__ bias,
    std::int64_t n,
    std::int64_t output_dim) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const std::int64_t col = idx % output_dim;
  values[idx] = f32_to_bf16_bits_device(bf16_bits_to_f32_device(values[idx]) + bias[col]);
}

__tile_global__ void bf16_bits_add_bias_inplace_tile_float32_kernel(
    std::uint16_t* __restrict__ values_bf16_bits,
    const float* __restrict__ bias,
    std::int64_t n,
    std::int64_t output_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto* values = ct::assume_aligned(reinterpret_cast<__nv_bfloat16*>(values_bf16_bits), 16_ic);
  bias = ct::assume_aligned(bias, 16_ic);

  const int bx = ct::bid().x;
  using Shape = decltype(ct::shape{1024_ic});
  using IndexTile = ct::tile<std::int64_t, Shape>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto col = idx % ct::full<IndexTile>(output_dim);
  auto value = ct::element_cast<float>(ct::load_masked(values + idx, mask));
  auto bias_value = ct::load_masked(bias + col, mask);
  ct::store_masked(values + idx, ct::element_cast<__nv_bfloat16>(value + bias_value), mask);
}

bool bf16_bits_add_bias_tile_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_BF16_BIAS_INPLACE_TILE");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT_BF16_BIAS_INPLACE_TILE");
    }
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT2_BF16_BIAS_INPLACE_TILE");
    }
    if (value == nullptr) {
      return true;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

constexpr std::int64_t kLinearBackwardBiasRowChunkSize = 256;

std::int64_t linear_backward_bias_row_chunk_size() {
  static const std::int64_t value = []() {
    const char* raw = std::getenv("NFN_TILE_CUDA_LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE");
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT_LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE");
    }
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT2_LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE");
    }
    if (raw == nullptr || raw[0] == '\0') {
      return kLinearBackwardBiasRowChunkSize;
    }
    char* end = nullptr;
    const long long parsed = std::strtoll(raw, &end, 10);
    if (end == raw || (end != nullptr && *end != '\0') || parsed <= 0) {
      return kLinearBackwardBiasRowChunkSize;
    }
    return static_cast<std::int64_t>(parsed);
  }();
  return value;
}

int linear_backward_bias_threads_per_block_value() {
  static const int value = []() {
    const char* raw = std::getenv("NFN_TILE_CUDA_LINEAR_BACKWARD_BIAS_THREADS");
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT_LINEAR_BACKWARD_BIAS_THREADS");
    }
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT2_LINEAR_BACKWARD_BIAS_THREADS");
    }
    if (raw == nullptr || raw[0] == '\0') {
      return 256;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || (end != nullptr && *end != '\0')) {
      return 256;
    }
    if (parsed == 128 || parsed == 256 || parsed == 512 || parsed == 1024) {
      return static_cast<int>(parsed);
    }
    return 256;
  }();
  return value;
}

__global__ void store_mlp_activations_bf16_float32_kernel(
    const float* __restrict__ ln2_out,
    const float* __restrict__ fc_out,
    const float* __restrict__ act,
    std::uint16_t* __restrict__ dest,
    std::int64_t activation_elements,
    std::int64_t hidden_elements) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t total = activation_elements + hidden_elements * 2;
  if (idx >= total) {
    return;
  }
  float value = 0.0f;
  if (idx < activation_elements) {
    value = ln2_out[idx];
  } else if (idx < activation_elements + hidden_elements) {
    value = fc_out[idx - activation_elements];
  } else {
    value = act[idx - activation_elements - hidden_elements];
  }
  dest[idx] = f32_to_bf16_bits_device(value);
}

__global__ void restore_mlp_activations_bf16_float32_kernel(
    const std::uint16_t* __restrict__ source,
    float* __restrict__ ln2_out,
    float* __restrict__ fc_out,
    float* __restrict__ act,
    std::int64_t activation_elements,
    std::int64_t hidden_elements) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t total = activation_elements + hidden_elements * 2;
  if (idx >= total) {
    return;
  }
  const float value = bf16_bits_to_f32_device(source[idx]);
  if (idx < activation_elements) {
    ln2_out[idx] = value;
  } else if (idx < activation_elements + hidden_elements) {
    fc_out[idx - activation_elements] = value;
  } else {
    act[idx - activation_elements - hidden_elements] = value;
  }
}

__device__ __forceinline__ unsigned long long f32x4_to_bf16x4_bits_device(float4 value) {
  const std::uint16_t a = f32_bits_to_bf16_bits_device(__float_as_uint(value.x));
  const std::uint16_t b = f32_bits_to_bf16_bits_device(__float_as_uint(value.y));
  const std::uint16_t c = f32_bits_to_bf16_bits_device(__float_as_uint(value.z));
  const std::uint16_t d = f32_bits_to_bf16_bits_device(__float_as_uint(value.w));
  return static_cast<unsigned long long>(a) |
      (static_cast<unsigned long long>(b) << 16) |
      (static_cast<unsigned long long>(c) << 32) |
      (static_cast<unsigned long long>(d) << 48);
}

__device__ __forceinline__ float4 bf16x4_bits_to_f32x4_device(unsigned long long packed) {
  float4 value;
  value.x = bf16_bits_to_f32_device(static_cast<std::uint16_t>(packed & 0xffffull));
  value.y = bf16_bits_to_f32_device(static_cast<std::uint16_t>((packed >> 16) & 0xffffull));
  value.z = bf16_bits_to_f32_device(static_cast<std::uint16_t>((packed >> 32) & 0xffffull));
  value.w = bf16_bits_to_f32_device(static_cast<std::uint16_t>((packed >> 48) & 0xffffull));
  return value;
}

__global__ void store_mlp_activations_bf16_float32_vec4_kernel(
    const float* __restrict__ ln2_out,
    const float* __restrict__ fc_out,
    const float* __restrict__ act,
    unsigned long long* __restrict__ dest,
    std::int64_t activation_vec4,
    std::int64_t hidden_vec4) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t total_vec4 = activation_vec4 + hidden_vec4 * 2;
  if (idx >= total_vec4) {
    return;
  }
  const float4* src = nullptr;
  std::int64_t src_idx = idx;
  if (idx < activation_vec4) {
    src = reinterpret_cast<const float4*>(ln2_out);
  } else if (idx < activation_vec4 + hidden_vec4) {
    src = reinterpret_cast<const float4*>(fc_out);
    src_idx = idx - activation_vec4;
  } else {
    src = reinterpret_cast<const float4*>(act);
    src_idx = idx - activation_vec4 - hidden_vec4;
  }
  dest[idx] = f32x4_to_bf16x4_bits_device(src[src_idx]);
}

__global__ void restore_mlp_activations_bf16_float32_vec4_kernel(
    const unsigned long long* __restrict__ source,
    float* __restrict__ ln2_out,
    float* __restrict__ fc_out,
    float* __restrict__ act,
    std::int64_t activation_vec4,
    std::int64_t hidden_vec4) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t total_vec4 = activation_vec4 + hidden_vec4 * 2;
  if (idx >= total_vec4) {
    return;
  }
  float4* dst = nullptr;
  std::int64_t dst_idx = idx;
  if (idx < activation_vec4) {
    dst = reinterpret_cast<float4*>(ln2_out);
  } else if (idx < activation_vec4 + hidden_vec4) {
    dst = reinterpret_cast<float4*>(fc_out);
    dst_idx = idx - activation_vec4;
  } else {
    dst = reinterpret_cast<float4*>(act);
    dst_idx = idx - activation_vec4 - hidden_vec4;
  }
  dst[dst_idx] = bf16x4_bits_to_f32x4_device(source[idx]);
}

bool store_mlp_activations_vec4_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_STORE_MLP_ACTIVATIONS_VEC4");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS_VEC4");
    }
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT2_STORE_MLP_ACTIVATIONS_VEC4");
    }
    if (value == nullptr) {
      return false;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

__global__ void f32_to_bf16_bits_many_kernel(
    const float* const* __restrict__ sources,
    const std::int64_t* __restrict__ elements,
    const std::int64_t* __restrict__ offsets,
    std::uint16_t* __restrict__ dst,
    std::int64_t buffer_count,
    std::int64_t max_elements) {
  const std::int64_t buffer = static_cast<std::int64_t>(blockIdx.y);
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (buffer >= buffer_count || idx >= max_elements) {
    return;
  }
  const std::int64_t n = elements[buffer];
  if (idx >= n) {
    return;
  }
  const float* src = sources[buffer];
  dst[offsets[buffer] + idx] = f32_to_bf16_bits_device(src[idx]);
}

__global__ void f32_to_bf16_bits_many_vec4_kernel(
    const float* const* __restrict__ sources,
    const std::int64_t* __restrict__ elements,
    const std::int64_t* __restrict__ offsets,
    std::uint16_t* __restrict__ dst,
    std::int64_t buffer_count,
    std::int64_t max_vec4_elements) {
  const std::int64_t buffer = static_cast<std::int64_t>(blockIdx.y);
  const std::int64_t idx4 = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (buffer >= buffer_count || idx4 >= max_vec4_elements) {
    return;
  }
  const std::int64_t n = elements[buffer];
  const std::int64_t base = idx4 * 4;
  if (base >= n) {
    return;
  }
  const float* src = sources[buffer];
  const std::int64_t dst_offset = offsets[buffer] + base;
  const bool vector_aligned =
      ((reinterpret_cast<std::uintptr_t>(src + base) & 0x0f) == 0) &&
      ((reinterpret_cast<std::uintptr_t>(dst + dst_offset) & 0x07) == 0);
  if (vector_aligned && base + 3 < n) {
    const float4 value = *reinterpret_cast<const float4*>(src + base);
    const std::uint16_t a = f32_bits_to_bf16_bits_device(__float_as_uint(value.x));
    const std::uint16_t b = f32_bits_to_bf16_bits_device(__float_as_uint(value.y));
    const std::uint16_t c = f32_bits_to_bf16_bits_device(__float_as_uint(value.z));
    const std::uint16_t d = f32_bits_to_bf16_bits_device(__float_as_uint(value.w));
    *reinterpret_cast<unsigned long long*>(dst + dst_offset) =
        static_cast<unsigned long long>(a) |
        (static_cast<unsigned long long>(b) << 16) |
        (static_cast<unsigned long long>(c) << 32) |
        (static_cast<unsigned long long>(d) << 48);
    return;
  }
  for (int lane = 0; lane < 4; ++lane) {
    const std::int64_t idx = base + lane;
    if (idx < n) {
      dst[offsets[buffer] + idx] = f32_to_bf16_bits_device(src[idx]);
    }
  }
}

bool f32_to_bf16_bits_many_vec4_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_F32_TO_BF16_MANY_VEC4");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT_F32_TO_BF16_MANY_VEC4");
    }
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT2_F32_TO_BF16_MANY_VEC4");
    }
    if (value == nullptr) {
      return false;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool bf16_bits_to_f32_vec4_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_BF16_TO_F32_VEC4");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT_BF16_TO_F32_VEC4");
    }
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT2_BF16_TO_F32_VEC4");
    }
    if (value == nullptr) {
      return false;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

void launch_bf16_bits_to_float32_internal(
    const std::uint16_t* source,
    float* dest,
    std::int64_t n,
    cudaStream_t stream) {
  if (bf16_bits_to_f32_vec4_enabled() && n > 0 && (n % 4) == 0) {
    const std::int64_t n4 = n / 4;
    const int blocks = static_cast<int>((n4 + kTileSize - 1) / kTileSize);
    bf16_bits_to_f32_vec4_kernel<<<blocks, kTileSize, 0, stream>>>(source, dest, n4);
    g_bf16_to_f32_vec4_count.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  bf16_bits_to_f32_kernel<<<blocks, kTileSize, 0, stream>>>(source, dest, n);
}

void release_token_cross_entropy_workspace(TokenCrossEntropyWorkspace& workspace) {
  if (workspace.row_max != nullptr) cudaFree(workspace.row_max);
  if (workspace.row_denom != nullptr) cudaFree(workspace.row_denom);
  workspace = {};
}

TokenCrossEntropyWorkspace* ensure_token_cross_entropy_workspace(std::int64_t rows) {
  if (rows <= 0) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(g_token_cross_entropy_workspace_mutex);
  if (g_token_cross_entropy_workspace.row_capacity >= rows &&
      g_token_cross_entropy_workspace.row_max != nullptr &&
      g_token_cross_entropy_workspace.row_denom != nullptr) {
    return &g_token_cross_entropy_workspace;
  }

  TokenCrossEntropyWorkspace next;
  next.row_capacity = rows;
  const std::size_t bytes = sizeof(float) * static_cast<std::size_t>(rows);
  if (cudaMalloc(&next.row_max, bytes) != cudaSuccess) {
    return nullptr;
  }
  if (cudaMalloc(&next.row_denom, bytes) != cudaSuccess) {
    release_token_cross_entropy_workspace(next);
    return nullptr;
  }
  release_token_cross_entropy_workspace(g_token_cross_entropy_workspace);
  g_token_cross_entropy_workspace = next;
  g_token_cross_entropy_workspace_allocation_count.fetch_add(1, std::memory_order_relaxed);
  g_token_cross_entropy_workspace_row_capacity.store(rows, std::memory_order_relaxed);
  return &g_token_cross_entropy_workspace;
}

#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
struct TkAttentionWorkspace {
  __nv_bfloat16* q_bf = nullptr;
  __nv_bfloat16* k_bf = nullptr;
  __nv_bfloat16* v_bf = nullptr;
  __nv_bfloat16* o_bf = nullptr;
  __nv_bfloat16* go_bf = nullptr;
  __nv_bfloat16* gq_bf = nullptr;
  __nv_bfloat16* gk_bf = nullptr;
  __nv_bfloat16* gv_bf = nullptr;
  __nv_bfloat16* packed_grad_bf = nullptr;
#if defined(LLMK_SM120_ATOMIC_DQ)
  float* qg_float = nullptr;
#endif
  float* lse = nullptr;
  float* d = nullptr;
  std::int64_t element_capacity = 0;
  std::int64_t row_capacity = 0;
};

TkAttentionWorkspace g_tk_attention_workspace;
std::mutex g_tk_attention_workspace_mutex;

std::int64_t tk_packed_attention_backward_max_batch_per_launch() {
  static const std::int64_t value = []() {
    const char* raw = std::getenv("NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP");
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT2_PACKED_ATTENTION_BACKWARD_BATCH_CAP");
    }
    if (raw == nullptr || raw[0] == '\0') {
      return kTkPackedAttentionBackwardDefaultMaxBatchPerLaunch;
    }
    char* end = nullptr;
    const long long parsed = std::strtoll(raw, &end, 10);
    if (end == raw || parsed <= 0) {
      return kTkPackedAttentionBackwardDefaultMaxBatchPerLaunch;
    }
    return static_cast<std::int64_t>(parsed);
  }();
  return value;
}

bool tk_packed_attention_dprep_grid3d_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_GRID3D");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT2_PACKED_ATTENTION_DPREP_GRID3D");
    }
    if (value == nullptr || value[0] == '\0') {
      return false;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

int tk_packed_attention_dprep_warps_per_block() {
  static const int warps = []() {
    const char* raw = std::getenv("NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_WARPS");
    if (raw == nullptr) {
      raw = std::getenv("NFN_NATIVE_GPT2_PACKED_ATTENTION_DPREP_WARPS");
    }
    if (raw == nullptr || raw[0] == '\0') {
      return 3;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || parsed < 1 || parsed > 8) {
      return 3;
    }
    return static_cast<int>(parsed);
  }();
  return warps;
}

bool tk_packed_attention_dprep_hd64_specialized_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_HD64_SPECIALIZED");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT2_PACKED_ATTENTION_DPREP_HD64_SPECIALIZED");
    }
    if (value == nullptr || value[0] == '\0') {
      return true;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool tk_packed_attention_dprep_float_hd64_specialized_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_FLOAT_HD64_SPECIALIZED");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT2_PACKED_ATTENTION_DPREP_FLOAT_HD64_SPECIALIZED");
    }
    if (value == nullptr || value[0] == '\0') {
      return false;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool tk_attention_backward_section_timing_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_NATIVE_GPT_ATTENTION_BACKWARD_SECTION_TIMING");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT2_ATTENTION_BACKWARD_SECTION_TIMING");
    }
    if (value == nullptr) {
      value = std::getenv("NFN_TILE_CUDA_ATTENTION_BACKWARD_SECTION_TIMING");
    }
    if (value == nullptr || value[0] == '\0') {
      return false;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

class CudaSectionTimer {
 public:
  CudaSectionTimer(
      bool enabled,
      cudaStream_t stream,
      std::atomic<std::int64_t>& total_us,
      std::atomic<std::int64_t>& count)
      : stream_(stream), total_us_(total_us), count_(count) {
    if (!enabled) {
      return;
    }
    if (cudaEventCreateWithFlags(&start_, cudaEventDefault) != cudaSuccess) {
      start_ = nullptr;
      return;
    }
    if (cudaEventCreateWithFlags(&stop_, cudaEventDefault) != cudaSuccess) {
      cudaEventDestroy(start_);
      start_ = nullptr;
      stop_ = nullptr;
      return;
    }
    if (cudaEventRecord(start_, stream_) == cudaSuccess) {
      active_ = true;
    }
  }

  ~CudaSectionTimer() {
    if (active_) {
      if (cudaEventRecord(stop_, stream_) == cudaSuccess &&
          cudaEventSynchronize(stop_) == cudaSuccess) {
        float elapsed_ms = 0.0f;
        if (cudaEventElapsedTime(&elapsed_ms, start_, stop_) == cudaSuccess) {
          const auto elapsed_us =
              static_cast<std::int64_t>(std::llround(static_cast<double>(elapsed_ms) * 1000.0));
          total_us_.fetch_add(elapsed_us, std::memory_order_relaxed);
          count_.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
    if (stop_ != nullptr) {
      cudaEventDestroy(stop_);
    }
    if (start_ != nullptr) {
      cudaEventDestroy(start_);
    }
  }

 private:
  cudaStream_t stream_ = nullptr;
  std::atomic<std::int64_t>& total_us_;
  std::atomic<std::int64_t>& count_;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
  bool active_ = false;
};

void release_tk_attention_workspace(TkAttentionWorkspace& workspace) {
  if (workspace.q_bf != nullptr) cudaFree(workspace.q_bf);
  if (workspace.k_bf != nullptr) cudaFree(workspace.k_bf);
  if (workspace.v_bf != nullptr) cudaFree(workspace.v_bf);
  if (workspace.o_bf != nullptr) cudaFree(workspace.o_bf);
  if (workspace.go_bf != nullptr) cudaFree(workspace.go_bf);
  if (workspace.gq_bf != nullptr) cudaFree(workspace.gq_bf);
  if (workspace.gk_bf != nullptr) cudaFree(workspace.gk_bf);
  if (workspace.gv_bf != nullptr) cudaFree(workspace.gv_bf);
  if (workspace.packed_grad_bf != nullptr) cudaFree(workspace.packed_grad_bf);
#if defined(LLMK_SM120_ATOMIC_DQ)
  if (workspace.qg_float != nullptr) cudaFree(workspace.qg_float);
#endif
  if (workspace.lse != nullptr) cudaFree(workspace.lse);
  if (workspace.d != nullptr) cudaFree(workspace.d);
  workspace = {};
}

bool cuda_malloc_bf16(__nv_bfloat16** ptr, std::int64_t elements) {
  return cudaMalloc(
      reinterpret_cast<void**>(ptr),
      sizeof(__nv_bfloat16) * static_cast<std::size_t>(elements)) == cudaSuccess;
}

bool cuda_malloc_float(float** ptr, std::int64_t elements) {
  return cudaMalloc(
      reinterpret_cast<void**>(ptr),
      sizeof(float) * static_cast<std::size_t>(elements)) == cudaSuccess;
}

TkAttentionWorkspace* ensure_tk_attention_workspace(
    std::int64_t elements,
    std::int64_t row_elements,
    cudaStream_t stream,
    bool require_packed_grad = false) {
  std::lock_guard<std::mutex> lock(g_tk_attention_workspace_mutex);
  if (g_tk_attention_workspace.element_capacity >= elements &&
      g_tk_attention_workspace.row_capacity >= row_elements &&
      (!require_packed_grad || g_tk_attention_workspace.packed_grad_bf != nullptr)) {
    return &g_tk_attention_workspace;
  }
  cudaStreamSynchronize(stream);
  release_tk_attention_workspace(g_tk_attention_workspace);
  TkAttentionWorkspace next;
  if (!cuda_malloc_bf16(&next.q_bf, elements) ||
      !cuda_malloc_bf16(&next.k_bf, elements) ||
      !cuda_malloc_bf16(&next.v_bf, elements) ||
      !cuda_malloc_bf16(&next.o_bf, elements) ||
      !cuda_malloc_bf16(&next.go_bf, elements) ||
      !cuda_malloc_bf16(&next.gq_bf, elements) ||
      !cuda_malloc_bf16(&next.gk_bf, elements) ||
      !cuda_malloc_bf16(&next.gv_bf, elements) ||
      !cuda_malloc_float(&next.lse, row_elements) ||
      !cuda_malloc_float(&next.d, row_elements)) {
    release_tk_attention_workspace(next);
    return nullptr;
  }
  if (require_packed_grad && !cuda_malloc_bf16(&next.packed_grad_bf, elements * 3)) {
    release_tk_attention_workspace(next);
    return nullptr;
  }
#if defined(LLMK_SM120_ATOMIC_DQ)
  if (require_packed_grad && !cuda_malloc_float(&next.qg_float, elements)) {
    release_tk_attention_workspace(next);
    return nullptr;
  }
#endif
  next.element_capacity = elements;
  next.row_capacity = row_elements;
  g_tk_attention_workspace = next;
  g_attention_tk_workspace_allocation_count.fetch_add(1, std::memory_order_relaxed);
  g_attention_tk_workspace_element_capacity.store(elements, std::memory_order_relaxed);
  g_attention_tk_workspace_row_capacity.store(row_elements, std::memory_order_relaxed);
  return &g_tk_attention_workspace;
}

#if defined(LLMK_SM120_ATOMIC_DQ)
__global__ void pack_attention_atomic_dq_to_packed_grad_bf16_kernel(
    const float* qg,
    std::uint16_t* packed_grad_bits,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t elements = batch * heads * seq_len * head_dim;
  if (idx >= elements) {
    return;
  }
  const std::int64_t d = idx % head_dim;
  const std::int64_t t = (idx / head_dim) % seq_len;
  const std::int64_t h = (idx / (head_dim * seq_len)) % heads;
  const std::int64_t b = idx / (head_dim * seq_len * heads);
  const std::int64_t packed_cols = 3 * heads * head_dim;
  const std::int64_t packed_idx = (b * seq_len + t) * packed_cols + h * head_dim + d;
  packed_grad_bits[packed_idx] = f32_to_bf16_bits_device(qg[idx]);
}

template <int D>
void launch_tk_attention_packed_qkv_backward_atomic_dq(
    __nv_bfloat16* qkv,
    __nv_bfloat16* out,
    float* lse,
    __nv_bfloat16* grad_out,
    float* d,
    float* qg,
    __nv_bfloat16* packed_grad,
    int batch,
    int heads,
    int seq_len,
    cudaStream_t stream) {
  using globals = llmk::attention::sm120_detail::bwd_globals<D>;
  const unsigned int Bu = static_cast<unsigned int>(batch);
  const unsigned int NHu = static_cast<unsigned int>(heads);
  const unsigned int Tu = static_cast<unsigned int>(seq_len);
  const unsigned int Du = static_cast<unsigned int>(D);
  const unsigned int packed_cols = static_cast<unsigned int>(3 * heads * D);

  typename globals::q_gl q_arg{qkv, Bu, 1U, Tu, packed_cols};
  typename globals::k_gl k_arg{qkv, Bu, 1U, Tu, packed_cols};
  typename globals::v_gl v_arg{qkv, Bu, 1U, Tu, packed_cols};
  typename globals::o_gl o_arg{out, Bu, NHu, Tu, Du};
  typename globals::og_gl og_arg{grad_out, Bu, NHu, Tu, Du};
  typename globals::l_gl l_arg{lse, Bu, NHu, 1U, Tu};
  typename globals::d_gl d_arg{d, Bu, NHu, 1U, Tu};
  typename globals::qg_gl qg_arg{qg, Bu, NHu, Tu, Du};
  typename globals::grad_gl kg_arg{packed_grad, Bu, 1U, Tu, packed_cols};
  typename globals::grad_gl vg_arg{packed_grad, Bu, 1U, Tu, packed_cols};
  globals g{q_arg, k_arg, v_arg, o_arg, og_arg, l_arg, d_arg, qg_arg, kg_arg, vg_arg, seq_len};

  dim3 grid(seq_len / llmk::attention::sm120_detail::Q_BLOCK, heads, batch);
  llmk::attention::sm120_detail::bwd_main_kernel<D, true, true>
      <<<grid, ::kittens::WARP_THREADS, 0, stream>>>(g);
  constexpr int threads = 256;
  const std::int64_t q_elements =
      static_cast<std::int64_t>(batch) * heads * seq_len * D;
  const int blocks = static_cast<int>((q_elements + threads - 1) / threads);
  pack_attention_atomic_dq_to_packed_grad_bf16_kernel<<<blocks, threads, 0, stream>>>(
      qg,
      reinterpret_cast<std::uint16_t*>(packed_grad),
      batch,
      heads,
      seq_len,
      D);
}
#endif

__global__ void f32_to_bf16_attention_grad_kernel(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    bool src_merged) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t n = batch * heads * seq_len * head_dim;
  if (idx >= n) {
    return;
  }
  if (!src_merged) {
    dst[idx] = __float2bfloat16(src[idx]);
    return;
  }
  const std::int64_t d = idx % head_dim;
  const std::int64_t t = (idx / head_dim) % seq_len;
  const std::int64_t h = (idx / (head_dim * seq_len)) % heads;
  const std::int64_t b = idx / (head_dim * seq_len * heads);
  const std::int64_t merged_idx = ((b * seq_len + t) * heads + h) * head_dim + d;
  dst[idx] = __float2bfloat16(src[merged_idx]);
}

__global__ void packed_attention_dprep_kernel(
    const std::uint16_t* __restrict__ out_btc_bf16_bits,
    const float* __restrict__ grad_out_btc,
    __nv_bfloat16* __restrict__ out_heads_bf16,
    __nv_bfloat16* __restrict__ grad_out_heads_bf16,
    float* __restrict__ d,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim) {
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  const std::int64_t rows = batch * heads * seq_len;
  if (row >= rows) {
    return;
  }
  const std::int64_t t = row % seq_len;
  const std::int64_t h = (row / seq_len) % heads;
  const std::int64_t b = row / (heads * seq_len);
  const std::int64_t merged_base = ((b * seq_len + t) * heads + h) * head_dim;
  const std::int64_t head_base = row * head_dim;
  float acc = 0.0f;
  for (std::int64_t d_idx = threadIdx.x; d_idx < head_dim; d_idx += warpSize) {
    const std::uint16_t out_bits = out_btc_bf16_bits[merged_base + d_idx];
    const float out_value = bf16_bits_to_f32_device(out_bits);
    const float grad_value = grad_out_btc[merged_base + d_idx];
    out_heads_bf16[head_base + d_idx] = reinterpret_cast<const __nv_bfloat16&>(out_bits);
    grad_out_heads_bf16[head_base + d_idx] = __float2bfloat16(grad_value);
    acc += out_value * grad_value;
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    acc += __shfl_down_sync(0xffffffff, acc, offset);
  }
  if (threadIdx.x == 0) {
    d[row] = acc;
  }
}

__global__ void packed_attention_dprep_grid3d_kernel(
    const std::uint16_t* __restrict__ out_btc_bf16_bits,
    const float* __restrict__ grad_out_btc,
    __nv_bfloat16* __restrict__ out_heads_bf16,
    __nv_bfloat16* __restrict__ grad_out_heads_bf16,
    float* __restrict__ d,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim) {
  const std::int64_t t = static_cast<std::int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  const std::int64_t h = static_cast<std::int64_t>(blockIdx.y);
  const std::int64_t b = static_cast<std::int64_t>(blockIdx.z);
  if (b >= batch || h >= heads || t >= seq_len) {
    return;
  }
  const std::int64_t merged_base = ((b * seq_len + t) * heads + h) * head_dim;
  const std::int64_t row = (b * heads + h) * seq_len + t;
  const std::int64_t head_base = row * head_dim;
  float acc = 0.0f;
  for (std::int64_t d_idx = threadIdx.x; d_idx < head_dim; d_idx += warpSize) {
    const std::uint16_t out_bits = out_btc_bf16_bits[merged_base + d_idx];
    const float out_value = bf16_bits_to_f32_device(out_bits);
    const float grad_value = grad_out_btc[merged_base + d_idx];
    out_heads_bf16[head_base + d_idx] = reinterpret_cast<const __nv_bfloat16&>(out_bits);
    grad_out_heads_bf16[head_base + d_idx] = __float2bfloat16(grad_value);
    acc += out_value * grad_value;
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    acc += __shfl_down_sync(0xffffffff, acc, offset);
  }
  if (threadIdx.x == 0) {
    d[row] = acc;
  }
}

__global__ void packed_attention_dprep_bf16_grad_kernel(
    const std::uint16_t* __restrict__ out_btc_bf16_bits,
    const std::uint16_t* __restrict__ grad_out_btc_bf16_bits,
    __nv_bfloat16* __restrict__ out_heads_bf16,
    __nv_bfloat16* __restrict__ grad_out_heads_bf16,
    float* __restrict__ d,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim) {
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  const std::int64_t rows = batch * heads * seq_len;
  if (row >= rows) {
    return;
  }
  const std::int64_t t = row % seq_len;
  const std::int64_t h = (row / seq_len) % heads;
  const std::int64_t b = row / (heads * seq_len);
  const std::int64_t merged_base = ((b * seq_len + t) * heads + h) * head_dim;
  const std::int64_t head_base = row * head_dim;
  float acc = 0.0f;
  for (std::int64_t d_idx = threadIdx.x; d_idx < head_dim; d_idx += warpSize) {
    const std::uint16_t out_bits = out_btc_bf16_bits[merged_base + d_idx];
    const std::uint16_t grad_bits = grad_out_btc_bf16_bits[merged_base + d_idx];
    const float out_value = bf16_bits_to_f32_device(out_bits);
    const float grad_value = bf16_bits_to_f32_device(grad_bits);
    out_heads_bf16[head_base + d_idx] = reinterpret_cast<const __nv_bfloat16&>(out_bits);
    grad_out_heads_bf16[head_base + d_idx] = reinterpret_cast<const __nv_bfloat16&>(grad_bits);
    acc += out_value * grad_value;
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    acc += __shfl_down_sync(0xffffffff, acc, offset);
  }
  if (threadIdx.x == 0) {
    d[row] = acc;
  }
}

__global__ void packed_attention_dprep_bf16_grad_hd64_h12_kernel(
    const std::uint16_t* __restrict__ out_btc_bf16_bits,
    const std::uint16_t* __restrict__ grad_out_btc_bf16_bits,
    __nv_bfloat16* __restrict__ out_heads_bf16,
    __nv_bfloat16* __restrict__ grad_out_heads_bf16,
    float* __restrict__ d,
    std::int64_t batch,
    std::int64_t seq_len) {
  constexpr std::int64_t kHeads = 12;
  constexpr std::int64_t kHeadDim = 64;
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  const std::int64_t rows = batch * kHeads * seq_len;
  if (row >= rows) {
    return;
  }
  const std::int64_t t = row % seq_len;
  const std::int64_t bh = row / seq_len;
  const std::int64_t h = bh % kHeads;
  const std::int64_t b = bh / kHeads;
  const std::int64_t merged_base = ((b * seq_len + t) * kHeads + h) * kHeadDim;
  const std::int64_t head_base = row * kHeadDim;
  const int lane = threadIdx.x;
  const int d0 = lane;
  const int d1 = lane + 32;
  const std::uint16_t out0 = out_btc_bf16_bits[merged_base + d0];
  const std::uint16_t grad0 = grad_out_btc_bf16_bits[merged_base + d0];
  const std::uint16_t out1 = out_btc_bf16_bits[merged_base + d1];
  const std::uint16_t grad1 = grad_out_btc_bf16_bits[merged_base + d1];
  out_heads_bf16[head_base + d0] = reinterpret_cast<const __nv_bfloat16&>(out0);
  grad_out_heads_bf16[head_base + d0] = reinterpret_cast<const __nv_bfloat16&>(grad0);
  out_heads_bf16[head_base + d1] = reinterpret_cast<const __nv_bfloat16&>(out1);
  grad_out_heads_bf16[head_base + d1] = reinterpret_cast<const __nv_bfloat16&>(grad1);
  float acc =
      bf16_bits_to_f32_device(out0) * bf16_bits_to_f32_device(grad0) +
      bf16_bits_to_f32_device(out1) * bf16_bits_to_f32_device(grad1);
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    acc += __shfl_down_sync(0xffffffff, acc, offset);
  }
  if (lane == 0) {
    d[row] = acc;
  }
}

__global__ void packed_attention_dprep_float_grad_hd64_h12_kernel(
    const std::uint16_t* __restrict__ out_btc_bf16_bits,
    const float* __restrict__ grad_out_btc,
    __nv_bfloat16* __restrict__ out_heads_bf16,
    __nv_bfloat16* __restrict__ grad_out_heads_bf16,
    float* __restrict__ d,
    std::int64_t batch,
    std::int64_t seq_len) {
  constexpr std::int64_t kHeads = 12;
  constexpr std::int64_t kHeadDim = 64;
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  const std::int64_t rows = batch * kHeads * seq_len;
  if (row >= rows) {
    return;
  }
  const std::int64_t t = row % seq_len;
  const std::int64_t bh = row / seq_len;
  const std::int64_t h = bh % kHeads;
  const std::int64_t b = bh / kHeads;
  const std::int64_t merged_base = ((b * seq_len + t) * kHeads + h) * kHeadDim;
  const std::int64_t head_base = row * kHeadDim;
  const int lane = threadIdx.x;
  const int d0 = lane;
  const int d1 = lane + 32;
  const std::uint16_t out0 = out_btc_bf16_bits[merged_base + d0];
  const std::uint16_t out1 = out_btc_bf16_bits[merged_base + d1];
  const float grad0 = grad_out_btc[merged_base + d0];
  const float grad1 = grad_out_btc[merged_base + d1];
  out_heads_bf16[head_base + d0] = reinterpret_cast<const __nv_bfloat16&>(out0);
  grad_out_heads_bf16[head_base + d0] = __float2bfloat16(grad0);
  out_heads_bf16[head_base + d1] = reinterpret_cast<const __nv_bfloat16&>(out1);
  grad_out_heads_bf16[head_base + d1] = __float2bfloat16(grad1);
  float acc =
      bf16_bits_to_f32_device(out0) * grad0 +
      bf16_bits_to_f32_device(out1) * grad1;
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    acc += __shfl_down_sync(0xffffffff, acc, offset);
  }
  if (lane == 0) {
    d[row] = acc;
  }
}

__global__ void packed_attention_dprep_bf16_grad_grid3d_kernel(
    const std::uint16_t* __restrict__ out_btc_bf16_bits,
    const std::uint16_t* __restrict__ grad_out_btc_bf16_bits,
    __nv_bfloat16* __restrict__ out_heads_bf16,
    __nv_bfloat16* __restrict__ grad_out_heads_bf16,
    float* __restrict__ d,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim) {
  const std::int64_t t = static_cast<std::int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  const std::int64_t h = static_cast<std::int64_t>(blockIdx.y);
  const std::int64_t b = static_cast<std::int64_t>(blockIdx.z);
  if (b >= batch || h >= heads || t >= seq_len) {
    return;
  }
  const std::int64_t merged_base = ((b * seq_len + t) * heads + h) * head_dim;
  const std::int64_t row = (b * heads + h) * seq_len + t;
  const std::int64_t head_base = row * head_dim;
  float acc = 0.0f;
  for (std::int64_t d_idx = threadIdx.x; d_idx < head_dim; d_idx += warpSize) {
    const std::uint16_t out_bits = out_btc_bf16_bits[merged_base + d_idx];
    const std::uint16_t grad_bits = grad_out_btc_bf16_bits[merged_base + d_idx];
    const float out_value = bf16_bits_to_f32_device(out_bits);
    const float grad_value = bf16_bits_to_f32_device(grad_bits);
    out_heads_bf16[head_base + d_idx] = reinterpret_cast<const __nv_bfloat16&>(out_bits);
    grad_out_heads_bf16[head_base + d_idx] = reinterpret_cast<const __nv_bfloat16&>(grad_bits);
    acc += out_value * grad_value;
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    acc += __shfl_down_sync(0xffffffff, acc, offset);
  }
  if (threadIdx.x == 0) {
    d[row] = acc;
  }
}

__global__ void bf16_to_f32_kernel(
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ dst,
    std::int64_t n) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = __bfloat162float(src[idx]);
  }
}

__global__ void bf16_heads_to_qkv_float32_kernel(
    const __nv_bfloat16* __restrict__ q_heads,
    const __nv_bfloat16* __restrict__ k_heads,
    const __nv_bfloat16* __restrict__ v_heads,
    float* __restrict__ qkv,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t dim = heads * head_dim;
  const std::int64_t n = batch * seq_len * dim;
  if (idx >= n) {
    return;
  }
  const std::int64_t d = idx % head_dim;
  const std::int64_t h = (idx / head_dim) % heads;
  const std::int64_t s = (idx / (head_dim * heads)) % seq_len;
  const std::int64_t b = idx / (head_dim * heads * seq_len);
  const std::int64_t src = ((b * heads + h) * seq_len + s) * head_dim + d;
  const std::int64_t row = b * seq_len + s;
  const std::int64_t col = h * head_dim + d;
  const std::int64_t qkv_base = row * (3 * dim) + col;
  qkv[qkv_base] = __bfloat162float(q_heads[src]);
  qkv[qkv_base + dim] = __bfloat162float(k_heads[src]);
  qkv[qkv_base + 2 * dim] = __bfloat162float(v_heads[src]);
}

bool use_tk_sm120_attention(
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride) {
  return query_heads == key_heads &&
      (qk_dim == 64 || qk_dim == 128) &&
      value_dim == qk_dim &&
      seq_q == seq_k &&
      seq_q > 0 &&
      (seq_q % 16) == 0 &&
      is_causal &&
      !right_align_causal &&
      !use_sparse_rules &&
      window <= 0 &&
      num_sinks <= 0 &&
      block_size <= 0 &&
      compress_stride <= 1;
}

int launch_tk_attention_forward_float32(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    cudaStream_t stream) {
  const std::int64_t elements = batch * heads * seq_len * head_dim;
  const std::int64_t lse_elements = batch * heads * seq_len;
  TkAttentionWorkspace* workspace = ensure_tk_attention_workspace(elements, lse_elements, stream);
  if (workspace == nullptr) {
    return 2;
  }
  const int threads = 256;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(q, workspace->q_bf, elements);
  f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(k, workspace->k_bf, elements);
  f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(v, workspace->v_bf, elements);
  if (head_dim == 64) {
    llmk::attention::launch_forward_causal<64>(
        llmk::to_bf16(workspace->q_bf), llmk::to_bf16(workspace->k_bf),
        llmk::to_bf16(workspace->v_bf), workspace->lse,
        llmk::to_bf16(workspace->o_bf), static_cast<int>(batch), static_cast<int>(heads),
        static_cast<int>(seq_len), stream);
  } else {
    llmk::attention::launch_forward_causal<128>(
        llmk::to_bf16(workspace->q_bf), llmk::to_bf16(workspace->k_bf),
        llmk::to_bf16(workspace->v_bf), workspace->lse,
        llmk::to_bf16(workspace->o_bf), static_cast<int>(batch), static_cast<int>(heads),
        static_cast<int>(seq_len), stream);
  }
  bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(workspace->o_bf, out, elements);
  g_attention_forward_tk_launch_count.fetch_add(1, std::memory_order_relaxed);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_tk_attention_backward_float32(
    const float* q,
    const float* k,
    const float* v,
    const float* grad_out,
    float* grad_q,
    float* grad_k,
    float* grad_v,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    bool grad_out_merged,
    cudaStream_t stream) {
#if defined(LLMK_SM120_ATOMIC_DQ)
  return 2;
#else
  const std::int64_t elements = batch * heads * seq_len * head_dim;
  const std::int64_t row_elements = batch * heads * seq_len;
  TkAttentionWorkspace* workspace = ensure_tk_attention_workspace(elements, row_elements, stream);
  if (workspace == nullptr) {
    return 2;
  }
  const int threads = 256;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(q, workspace->q_bf, elements);
  f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(k, workspace->k_bf, elements);
  f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(v, workspace->v_bf, elements);
  f32_to_bf16_attention_grad_kernel<<<blocks, threads, 0, stream>>>(
      grad_out, workspace->go_bf, batch, heads, seq_len, head_dim, grad_out_merged);
  if (head_dim == 64) {
    llmk::attention::launch_forward_causal<64>(
        llmk::to_bf16(workspace->q_bf), llmk::to_bf16(workspace->k_bf),
        llmk::to_bf16(workspace->v_bf), workspace->lse,
        llmk::to_bf16(workspace->o_bf), static_cast<int>(batch), static_cast<int>(heads),
        static_cast<int>(seq_len), stream);
    llmk::attention::launch_backward_causal<64>(
        llmk::to_bf16(workspace->q_bf), llmk::to_bf16(workspace->k_bf),
        llmk::to_bf16(workspace->v_bf), llmk::to_bf16(workspace->o_bf),
        workspace->lse, llmk::to_bf16(workspace->go_bf), workspace->d,
        llmk::to_bf16(workspace->gq_bf), llmk::to_bf16(workspace->gk_bf),
        llmk::to_bf16(workspace->gv_bf), static_cast<int>(batch), static_cast<int>(heads),
        static_cast<int>(seq_len), stream);
  } else {
    llmk::attention::launch_forward_causal<128>(
        llmk::to_bf16(workspace->q_bf), llmk::to_bf16(workspace->k_bf),
        llmk::to_bf16(workspace->v_bf), workspace->lse,
        llmk::to_bf16(workspace->o_bf), static_cast<int>(batch), static_cast<int>(heads),
        static_cast<int>(seq_len), stream);
    llmk::attention::launch_backward_causal<128>(
        llmk::to_bf16(workspace->q_bf), llmk::to_bf16(workspace->k_bf),
        llmk::to_bf16(workspace->v_bf), llmk::to_bf16(workspace->o_bf),
        workspace->lse, llmk::to_bf16(workspace->go_bf), workspace->d,
        llmk::to_bf16(workspace->gq_bf), llmk::to_bf16(workspace->gk_bf),
        llmk::to_bf16(workspace->gv_bf), static_cast<int>(batch), static_cast<int>(heads),
        static_cast<int>(seq_len), stream);
  }
  bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(workspace->gq_bf, grad_q, elements);
  bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(workspace->gk_bf, grad_k, elements);
  bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(workspace->gv_bf, grad_v, elements);
  g_attention_backward_tk_launch_count.fetch_add(1, std::memory_order_relaxed);
  return static_cast<int>(cudaPeekAtLastError());
#endif
}

int launch_tk_attention_backward_to_qkv_float32(
    const float* q,
    const float* k,
    const float* v,
    const float* grad_out,
    float* grad_qkv,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    bool grad_out_merged,
    cudaStream_t stream) {
#if defined(LLMK_SM120_ATOMIC_DQ)
  return 2;
#else
  const std::int64_t elements = batch * heads * seq_len * head_dim;
  const std::int64_t row_elements = batch * heads * seq_len;
  TkAttentionWorkspace* workspace = ensure_tk_attention_workspace(elements, row_elements, stream);
  if (workspace == nullptr) {
    return 2;
  }
  const int threads = 256;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(q, workspace->q_bf, elements);
  f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(k, workspace->k_bf, elements);
  f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(v, workspace->v_bf, elements);
  f32_to_bf16_attention_grad_kernel<<<blocks, threads, 0, stream>>>(
      grad_out, workspace->go_bf, batch, heads, seq_len, head_dim, grad_out_merged);
  if (head_dim == 64) {
    llmk::attention::launch_forward_causal<64>(
        llmk::to_bf16(workspace->q_bf), llmk::to_bf16(workspace->k_bf),
        llmk::to_bf16(workspace->v_bf), workspace->lse,
        llmk::to_bf16(workspace->o_bf), static_cast<int>(batch), static_cast<int>(heads),
        static_cast<int>(seq_len), stream);
    llmk::attention::launch_backward_causal<64>(
        llmk::to_bf16(workspace->q_bf), llmk::to_bf16(workspace->k_bf),
        llmk::to_bf16(workspace->v_bf), llmk::to_bf16(workspace->o_bf),
        workspace->lse, llmk::to_bf16(workspace->go_bf), workspace->d,
        llmk::to_bf16(workspace->gq_bf), llmk::to_bf16(workspace->gk_bf),
        llmk::to_bf16(workspace->gv_bf), static_cast<int>(batch), static_cast<int>(heads),
        static_cast<int>(seq_len), stream);
  } else {
    llmk::attention::launch_forward_causal<128>(
        llmk::to_bf16(workspace->q_bf), llmk::to_bf16(workspace->k_bf),
        llmk::to_bf16(workspace->v_bf), workspace->lse,
        llmk::to_bf16(workspace->o_bf), static_cast<int>(batch), static_cast<int>(heads),
        static_cast<int>(seq_len), stream);
    llmk::attention::launch_backward_causal<128>(
        llmk::to_bf16(workspace->q_bf), llmk::to_bf16(workspace->k_bf),
        llmk::to_bf16(workspace->v_bf), llmk::to_bf16(workspace->o_bf),
        workspace->lse, llmk::to_bf16(workspace->go_bf), workspace->d,
        llmk::to_bf16(workspace->gq_bf), llmk::to_bf16(workspace->gk_bf),
        llmk::to_bf16(workspace->gv_bf), static_cast<int>(batch), static_cast<int>(heads),
        static_cast<int>(seq_len), stream);
  }
  bf16_heads_to_qkv_float32_kernel<<<blocks, threads, 0, stream>>>(
      workspace->gq_bf,
      workspace->gk_bf,
      workspace->gv_bf,
      grad_qkv,
      batch,
      seq_len,
      heads,
      head_dim);
  g_attention_backward_tk_launch_count.fetch_add(1, std::memory_order_relaxed);
  return static_cast<int>(cudaPeekAtLastError());
#endif
}

int launch_tk_attention_backward_to_qkv_reuse_forward_float32(
    const float* grad_out,
    float* grad_qkv,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    bool grad_out_merged,
    cudaStream_t stream) {
#if defined(LLMK_SM120_ATOMIC_DQ)
  return 2;
#else
  const std::int64_t elements = batch * heads * seq_len * head_dim;
  const std::int64_t row_elements = batch * heads * seq_len;
  TkAttentionWorkspace* workspace = ensure_tk_attention_workspace(elements, row_elements, stream);
  if (workspace == nullptr ||
      workspace->q_bf == nullptr ||
      workspace->k_bf == nullptr ||
      workspace->v_bf == nullptr ||
      workspace->o_bf == nullptr ||
      workspace->lse == nullptr) {
    return 2;
  }
  const int threads = 256;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  f32_to_bf16_attention_grad_kernel<<<blocks, threads, 0, stream>>>(
      grad_out, workspace->go_bf, batch, heads, seq_len, head_dim, grad_out_merged);
  if (head_dim == 64) {
    llmk::attention::launch_backward_causal<64>(
        llmk::to_bf16(workspace->q_bf), llmk::to_bf16(workspace->k_bf),
        llmk::to_bf16(workspace->v_bf), llmk::to_bf16(workspace->o_bf),
        workspace->lse, llmk::to_bf16(workspace->go_bf), workspace->d,
        llmk::to_bf16(workspace->gq_bf), llmk::to_bf16(workspace->gk_bf),
        llmk::to_bf16(workspace->gv_bf), static_cast<int>(batch), static_cast<int>(heads),
        static_cast<int>(seq_len), stream);
  } else {
    llmk::attention::launch_backward_causal<128>(
        llmk::to_bf16(workspace->q_bf), llmk::to_bf16(workspace->k_bf),
        llmk::to_bf16(workspace->v_bf), llmk::to_bf16(workspace->o_bf),
        workspace->lse, llmk::to_bf16(workspace->go_bf), workspace->d,
        llmk::to_bf16(workspace->gq_bf), llmk::to_bf16(workspace->gk_bf),
        llmk::to_bf16(workspace->gv_bf), static_cast<int>(batch), static_cast<int>(heads),
        static_cast<int>(seq_len), stream);
  }
  bf16_heads_to_qkv_float32_kernel<<<blocks, threads, 0, stream>>>(
      workspace->gq_bf,
      workspace->gk_bf,
      workspace->gv_bf,
      grad_qkv,
      batch,
      seq_len,
      heads,
      head_dim);
  g_attention_backward_tk_launch_count.fetch_add(1, std::memory_order_relaxed);
  return static_cast<int>(cudaPeekAtLastError());
#endif
}

int launch_tk_attention_packed_qkv_forward_bf16_float32(
    const std::uint16_t* qkv_bf16_bits,
    std::uint16_t* out_bf16_bits,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    cudaStream_t stream) {
  if (qkv_bf16_bits == nullptr ||
      out_bf16_bits == nullptr ||
      batch <= 0 ||
      heads <= 0 ||
      seq_len <= 0 ||
      head_dim <= 0) {
    return 2;
  }
  const std::int64_t elements = batch * heads * seq_len * head_dim;
  const std::int64_t row_elements = batch * heads * seq_len;
  TkAttentionWorkspace* workspace = ensure_tk_attention_workspace(elements, row_elements, stream, true);
  if (workspace == nullptr || workspace->lse == nullptr) {
    return 2;
  }
  auto* qkv = const_cast<__nv_bfloat16*>(
      reinterpret_cast<const __nv_bfloat16*>(qkv_bf16_bits));
  auto* out = reinterpret_cast<__nv_bfloat16*>(out_bf16_bits);
  llmk::attention::launch_forward_causal_packed_qkv_btc(
      llmk::to_bf16(qkv),
      workspace->lse,
      llmk::to_bf16(out),
      static_cast<int>(batch),
      static_cast<int>(heads),
      static_cast<int>(seq_len),
      static_cast<int>(head_dim),
      stream);
  g_attention_forward_tk_launch_count.fetch_add(1, std::memory_order_relaxed);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_tk_attention_packed_qkv_forward_store_lse_bf16_float32(
    const std::uint16_t* qkv_bf16_bits,
    std::uint16_t* out_bf16_bits,
    float* saved_lse,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    cudaStream_t stream) {
  const int status = launch_tk_attention_packed_qkv_forward_bf16_float32(
      qkv_bf16_bits,
      out_bf16_bits,
      batch,
      heads,
      seq_len,
      head_dim,
      stream);
  if (status != 0) {
    return status;
  }
  const std::int64_t row_elements = batch * heads * seq_len;
  if (saved_lse == nullptr || g_tk_attention_workspace.lse == nullptr || row_elements <= 0) {
    return 2;
  }
  cudaMemcpyAsync(
      saved_lse,
      g_tk_attention_workspace.lse,
      sizeof(float) * static_cast<std::size_t>(row_elements),
      cudaMemcpyDeviceToDevice,
      stream);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_tk_attention_packed_qkv_backward_to_qkv_impl(
    const std::uint16_t* qkv_bf16_bits,
    const std::uint16_t* out_bf16_bits,
    const float* saved_lse,
    const float* grad_out,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_qkv,
    std::uint16_t* grad_qkv_bf16_bits,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    bool grad_out_merged,
    cudaStream_t stream) {
  if (qkv_bf16_bits == nullptr ||
      out_bf16_bits == nullptr ||
      (grad_out == nullptr && grad_out_bf16_bits == nullptr) ||
      (grad_qkv == nullptr && grad_qkv_bf16_bits == nullptr) ||
      !grad_out_merged ||
      batch <= 0 ||
      heads <= 0 ||
      seq_len <= 0 ||
      head_dim <= 0) {
    return 2;
  }
  const std::int64_t elements = batch * heads * seq_len * head_dim;
  const std::int64_t row_elements = batch * heads * seq_len;
  TkAttentionWorkspace* workspace = ensure_tk_attention_workspace(elements, row_elements, stream, true);
  if (workspace == nullptr ||
      workspace->o_bf == nullptr ||
      workspace->go_bf == nullptr ||
      workspace->packed_grad_bf == nullptr ||
#if defined(LLMK_SM120_ATOMIC_DQ)
      workspace->qg_float == nullptr ||
#endif
      workspace->lse == nullptr ||
      workspace->d == nullptr) {
    return 2;
  }
  const std::int64_t max_batch_per_launch =
      std::max<std::int64_t>(1, tk_packed_attention_backward_max_batch_per_launch());
  const std::int64_t head_elements_per_batch = heads * seq_len * head_dim;
  const std::int64_t row_elements_per_batch = heads * seq_len;
  const std::int64_t packed_elements_per_batch = seq_len * heads * head_dim * 3;
  const std::int64_t merged_elements_per_batch = seq_len * heads * head_dim;
  const bool time_backward_sections = tk_attention_backward_section_timing_enabled();
  std::int64_t tk_backward_chunk_launches = 0;
  for (std::int64_t batch_begin = 0; batch_begin < batch; batch_begin += max_batch_per_launch) {
    const std::int64_t chunk_batch = std::min(max_batch_per_launch, batch - batch_begin);
    const std::int64_t chunk_rows = chunk_batch * row_elements_per_batch;
    dim3 dprep_block(32, static_cast<unsigned int>(tk_packed_attention_dprep_warps_per_block()), 1);
    {
      CudaSectionTimer timer(
          time_backward_sections,
          stream,
          g_attention_backward_dprep_timing_us,
          g_attention_backward_dprep_timing_count);
      if (tk_packed_attention_dprep_grid3d_enabled()) {
        dim3 dprep_grid(
            static_cast<unsigned int>((seq_len + dprep_block.y - 1) / dprep_block.y),
            static_cast<unsigned int>(heads),
            static_cast<unsigned int>(chunk_batch));
        if (grad_out_bf16_bits != nullptr) {
          packed_attention_dprep_bf16_grad_grid3d_kernel<<<dprep_grid, dprep_block, 0, stream>>>(
              out_bf16_bits + batch_begin * merged_elements_per_batch,
              grad_out_bf16_bits + batch_begin * merged_elements_per_batch,
              workspace->o_bf + batch_begin * head_elements_per_batch,
              workspace->go_bf + batch_begin * head_elements_per_batch,
              workspace->d + batch_begin * row_elements_per_batch,
              chunk_batch,
              heads,
              seq_len,
              head_dim);
        } else {
          packed_attention_dprep_grid3d_kernel<<<dprep_grid, dprep_block, 0, stream>>>(
              out_bf16_bits + batch_begin * merged_elements_per_batch,
              grad_out + batch_begin * merged_elements_per_batch,
              workspace->o_bf + batch_begin * head_elements_per_batch,
              workspace->go_bf + batch_begin * head_elements_per_batch,
              workspace->d + batch_begin * row_elements_per_batch,
              chunk_batch,
              heads,
              seq_len,
              head_dim);
        }
      } else {
        const int dprep_grid = static_cast<int>((chunk_rows + dprep_block.y - 1) / dprep_block.y);
        if (grad_out_bf16_bits != nullptr &&
            head_dim == 64 &&
            heads == kGpt2AttentionHeads &&
            tk_packed_attention_dprep_hd64_specialized_enabled()) {
          packed_attention_dprep_bf16_grad_hd64_h12_kernel<<<dprep_grid, dprep_block, 0, stream>>>(
              out_bf16_bits + batch_begin * merged_elements_per_batch,
              grad_out_bf16_bits + batch_begin * merged_elements_per_batch,
              workspace->o_bf + batch_begin * head_elements_per_batch,
              workspace->go_bf + batch_begin * head_elements_per_batch,
              workspace->d + batch_begin * row_elements_per_batch,
              chunk_batch,
              seq_len);
        } else if (grad_out_bf16_bits != nullptr) {
          packed_attention_dprep_bf16_grad_kernel<<<dprep_grid, dprep_block, 0, stream>>>(
              out_bf16_bits + batch_begin * merged_elements_per_batch,
              grad_out_bf16_bits + batch_begin * merged_elements_per_batch,
              workspace->o_bf + batch_begin * head_elements_per_batch,
              workspace->go_bf + batch_begin * head_elements_per_batch,
              workspace->d + batch_begin * row_elements_per_batch,
              chunk_batch,
              heads,
              seq_len,
              head_dim);
        } else if (head_dim == 64 &&
            heads == kGpt2AttentionHeads &&
            tk_packed_attention_dprep_float_hd64_specialized_enabled()) {
          g_attention_backward_float_hd64_dprep_launch_count.fetch_add(1, std::memory_order_relaxed);
          packed_attention_dprep_float_grad_hd64_h12_kernel<<<dprep_grid, dprep_block, 0, stream>>>(
              out_bf16_bits + batch_begin * merged_elements_per_batch,
              grad_out + batch_begin * merged_elements_per_batch,
              workspace->o_bf + batch_begin * head_elements_per_batch,
              workspace->go_bf + batch_begin * head_elements_per_batch,
              workspace->d + batch_begin * row_elements_per_batch,
              chunk_batch,
              seq_len);
        } else {
          packed_attention_dprep_kernel<<<dprep_grid, dprep_block, 0, stream>>>(
              out_bf16_bits + batch_begin * merged_elements_per_batch,
              grad_out + batch_begin * merged_elements_per_batch,
              workspace->o_bf + batch_begin * head_elements_per_batch,
              workspace->go_bf + batch_begin * head_elements_per_batch,
              workspace->d + batch_begin * row_elements_per_batch,
              chunk_batch,
              heads,
              seq_len,
              head_dim);
        }
      }
    }
    auto* qkv = const_cast<__nv_bfloat16*>(
        reinterpret_cast<const __nv_bfloat16*>(
            qkv_bf16_bits + batch_begin * packed_elements_per_batch));
    float* lse =
        const_cast<float*>((saved_lse != nullptr ? saved_lse : workspace->lse) +
                           batch_begin * row_elements_per_batch);
    __nv_bfloat16* packed_grad_target = workspace->packed_grad_bf + batch_begin * packed_elements_per_batch;
    const bool direct_bf16_grad_target =
        grad_qkv_bf16_bits != nullptr && grad_qkv_bf16_bits != qkv_bf16_bits;
    if (direct_bf16_grad_target) {
      packed_grad_target = reinterpret_cast<__nv_bfloat16*>(
          grad_qkv_bf16_bits + batch_begin * packed_elements_per_batch);
    }
    {
      CudaSectionTimer timer(
          time_backward_sections,
          stream,
          g_attention_backward_tk_timing_us,
          g_attention_backward_tk_timing_count);
#if defined(LLMK_SM120_ATOMIC_DQ)
      float* qg_float = workspace->qg_float + batch_begin * head_elements_per_batch;
      cudaMemsetAsync(
          qg_float,
          0,
          sizeof(float) * static_cast<std::size_t>(chunk_batch * head_elements_per_batch),
          stream);
      if (head_dim == 64) {
        launch_tk_attention_packed_qkv_backward_atomic_dq<64>(
            qkv,
            workspace->o_bf + batch_begin * head_elements_per_batch,
            lse,
            workspace->go_bf + batch_begin * head_elements_per_batch,
            workspace->d + batch_begin * row_elements_per_batch,
            qg_float,
            packed_grad_target,
            static_cast<int>(chunk_batch),
            static_cast<int>(heads),
            static_cast<int>(seq_len),
            stream);
      } else {
        launch_tk_attention_packed_qkv_backward_atomic_dq<128>(
            qkv,
            workspace->o_bf + batch_begin * head_elements_per_batch,
            lse,
            workspace->go_bf + batch_begin * head_elements_per_batch,
            workspace->d + batch_begin * row_elements_per_batch,
            qg_float,
            packed_grad_target,
            static_cast<int>(chunk_batch),
            static_cast<int>(heads),
            static_cast<int>(seq_len),
            stream);
      }
#else
      if (head_dim == 64) {
        llmk::attention::launch_backward_causal_packed_qkv_packed_grads<64>(
            llmk::to_bf16(qkv),
            llmk::to_bf16(workspace->o_bf + batch_begin * head_elements_per_batch),
            lse,
            llmk::to_bf16(workspace->go_bf + batch_begin * head_elements_per_batch),
            workspace->d + batch_begin * row_elements_per_batch,
            llmk::to_bf16(packed_grad_target),
            static_cast<int>(chunk_batch),
            static_cast<int>(heads),
            static_cast<int>(seq_len),
            stream,
            true);
      } else {
        llmk::attention::launch_backward_causal_packed_qkv_packed_grads<128>(
            llmk::to_bf16(qkv),
            llmk::to_bf16(workspace->o_bf + batch_begin * head_elements_per_batch),
            lse,
            llmk::to_bf16(workspace->go_bf + batch_begin * head_elements_per_batch),
            workspace->d + batch_begin * row_elements_per_batch,
            llmk::to_bf16(packed_grad_target),
            static_cast<int>(chunk_batch),
            static_cast<int>(heads),
            static_cast<int>(seq_len),
            stream,
            true);
      }
#endif
    }
    tk_backward_chunk_launches += 1;
  }
  const std::int64_t packed_elements = elements * 3;
  constexpr int threads = 256;
  if (grad_qkv_bf16_bits != nullptr && grad_qkv_bf16_bits == qkv_bf16_bits) {
    cudaMemcpyAsync(
        grad_qkv_bf16_bits,
        workspace->packed_grad_bf,
        sizeof(std::uint16_t) * static_cast<std::size_t>(packed_elements),
        cudaMemcpyDeviceToDevice,
        stream);
  }
  const __nv_bfloat16* packed_grad_source =
      (grad_qkv_bf16_bits != nullptr && grad_qkv_bf16_bits != qkv_bf16_bits)
          ? reinterpret_cast<const __nv_bfloat16*>(grad_qkv_bf16_bits)
          : workspace->packed_grad_bf;
  if (grad_qkv != nullptr) {
    const int blocks = static_cast<int>((packed_elements + threads - 1) / threads);
    bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(packed_grad_source, grad_qkv, packed_elements);
  }
  g_attention_backward_tk_launch_count.fetch_add(tk_backward_chunk_launches, std::memory_order_relaxed);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_tk_attention_packed_qkv_backward_to_qkv_float32(
    const std::uint16_t* qkv_bf16_bits,
    const std::uint16_t* out_bf16_bits,
    const float* saved_lse,
    const float* grad_out,
    float* grad_qkv,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    bool grad_out_merged,
    cudaStream_t stream) {
  return launch_tk_attention_packed_qkv_backward_to_qkv_impl(
      qkv_bf16_bits,
      out_bf16_bits,
      saved_lse,
      grad_out,
      nullptr,
      grad_qkv,
      nullptr,
      batch,
      heads,
      seq_len,
      head_dim,
      grad_out_merged,
      stream);
}

int launch_tk_attention_packed_qkv_backward_to_qkv_bf16_bits(
    const std::uint16_t* qkv_bf16_bits,
    const std::uint16_t* out_bf16_bits,
    const float* saved_lse,
    const float* grad_out,
    std::uint16_t* grad_qkv_bf16_bits,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    bool grad_out_merged,
    cudaStream_t stream) {
  return launch_tk_attention_packed_qkv_backward_to_qkv_impl(
      qkv_bf16_bits,
      out_bf16_bits,
      saved_lse,
      grad_out,
      nullptr,
      nullptr,
      grad_qkv_bf16_bits,
      batch,
      heads,
      seq_len,
      head_dim,
      grad_out_merged,
      stream);
}

int launch_tk_attention_packed_qkv_backward_to_qkv_bf16_bits_from_bf16_grad_bits(
    const std::uint16_t* qkv_bf16_bits,
    const std::uint16_t* out_bf16_bits,
    const float* saved_lse,
    const std::uint16_t* grad_out_bf16_bits,
    std::uint16_t* grad_qkv_bf16_bits,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    bool grad_out_merged,
    cudaStream_t stream) {
  return launch_tk_attention_packed_qkv_backward_to_qkv_impl(
      qkv_bf16_bits,
      out_bf16_bits,
      saved_lse,
      nullptr,
      grad_out_bf16_bits,
      nullptr,
      grad_qkv_bf16_bits,
      batch,
      heads,
      seq_len,
      head_dim,
      grad_out_merged,
      stream);
}

int launch_tk_attention_forward_store_bf16_float32(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    std::uint16_t* saved_q_bf16_bits,
    std::uint16_t* saved_k_bf16_bits,
    std::uint16_t* saved_v_bf16_bits,
    std::uint16_t* saved_o_bf16_bits,
    float* saved_lse,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    cudaStream_t stream) {
  if (q == nullptr ||
      k == nullptr ||
      v == nullptr ||
      out == nullptr ||
      saved_q_bf16_bits == nullptr ||
      saved_k_bf16_bits == nullptr ||
      saved_v_bf16_bits == nullptr ||
      saved_o_bf16_bits == nullptr ||
      saved_lse == nullptr ||
      batch <= 0 ||
      heads <= 0 ||
      seq_len <= 0 ||
      head_dim <= 0) {
    return 2;
  }
  const std::int64_t elements = batch * heads * seq_len * head_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  auto* saved_q = reinterpret_cast<__nv_bfloat16*>(saved_q_bf16_bits);
  auto* saved_k = reinterpret_cast<__nv_bfloat16*>(saved_k_bf16_bits);
  auto* saved_v = reinterpret_cast<__nv_bfloat16*>(saved_v_bf16_bits);
  auto* saved_o = reinterpret_cast<__nv_bfloat16*>(saved_o_bf16_bits);
  f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(q, saved_q, elements);
  f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(k, saved_k, elements);
  f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(v, saved_v, elements);
  if (head_dim == 64) {
    llmk::attention::launch_forward_causal<64>(
        llmk::to_bf16(saved_q), llmk::to_bf16(saved_k),
        llmk::to_bf16(saved_v), saved_lse,
        llmk::to_bf16(saved_o), static_cast<int>(batch), static_cast<int>(heads),
        static_cast<int>(seq_len), stream);
  } else {
    llmk::attention::launch_forward_causal<128>(
        llmk::to_bf16(saved_q), llmk::to_bf16(saved_k),
        llmk::to_bf16(saved_v), saved_lse,
        llmk::to_bf16(saved_o), static_cast<int>(batch), static_cast<int>(heads),
        static_cast<int>(seq_len), stream);
  }
  bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(saved_o, out, elements);
  g_attention_forward_tk_launch_count.fetch_add(1, std::memory_order_relaxed);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_tk_attention_store_forward_workspace_bf16(
    std::uint16_t* saved_q_bf16_bits,
    std::uint16_t* saved_k_bf16_bits,
    std::uint16_t* saved_v_bf16_bits,
    std::uint16_t* saved_o_bf16_bits,
    float* saved_lse,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    cudaStream_t stream) {
  if (saved_q_bf16_bits == nullptr ||
      saved_k_bf16_bits == nullptr ||
      saved_v_bf16_bits == nullptr ||
      saved_o_bf16_bits == nullptr ||
      saved_lse == nullptr ||
      batch <= 0 ||
      heads <= 0 ||
      seq_len <= 0 ||
      head_dim <= 0) {
    return 2;
  }
  const std::int64_t elements = batch * heads * seq_len * head_dim;
  const std::int64_t row_elements = batch * heads * seq_len;
  std::lock_guard<std::mutex> lock(g_tk_attention_workspace_mutex);
  if (g_tk_attention_workspace.element_capacity < elements ||
      g_tk_attention_workspace.row_capacity < row_elements ||
      g_tk_attention_workspace.q_bf == nullptr ||
      g_tk_attention_workspace.k_bf == nullptr ||
      g_tk_attention_workspace.v_bf == nullptr ||
      g_tk_attention_workspace.o_bf == nullptr ||
      g_tk_attention_workspace.lse == nullptr) {
    return 2;
  }
  const std::size_t bf16_bytes = sizeof(std::uint16_t) * static_cast<std::size_t>(elements);
  const std::size_t lse_bytes = sizeof(float) * static_cast<std::size_t>(row_elements);
  cudaMemcpyAsync(saved_q_bf16_bits, g_tk_attention_workspace.q_bf, bf16_bytes, cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(saved_k_bf16_bits, g_tk_attention_workspace.k_bf, bf16_bytes, cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(saved_v_bf16_bits, g_tk_attention_workspace.v_bf, bf16_bytes, cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(saved_o_bf16_bits, g_tk_attention_workspace.o_bf, bf16_bytes, cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(saved_lse, g_tk_attention_workspace.lse, lse_bytes, cudaMemcpyDeviceToDevice, stream);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_tk_attention_backward_to_qkv_from_saved_bf16_float32(
    const std::uint16_t* saved_q_bf16_bits,
    const std::uint16_t* saved_k_bf16_bits,
    const std::uint16_t* saved_v_bf16_bits,
    const std::uint16_t* saved_o_bf16_bits,
    const float* saved_lse,
    const float* grad_out,
    float* grad_qkv,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    bool grad_out_merged,
    cudaStream_t stream) {
#if defined(LLMK_SM120_ATOMIC_DQ)
  return 2;
#else
  if (saved_q_bf16_bits == nullptr ||
      saved_k_bf16_bits == nullptr ||
      saved_v_bf16_bits == nullptr ||
      saved_o_bf16_bits == nullptr ||
      saved_lse == nullptr ||
      grad_out == nullptr ||
      grad_qkv == nullptr) {
    return 2;
  }
  const std::int64_t elements = batch * heads * seq_len * head_dim;
  const std::int64_t row_elements = batch * heads * seq_len;
  TkAttentionWorkspace* workspace = ensure_tk_attention_workspace(elements, row_elements, stream);
  if (workspace == nullptr ||
      workspace->go_bf == nullptr ||
      workspace->gq_bf == nullptr ||
      workspace->gk_bf == nullptr ||
      workspace->gv_bf == nullptr ||
      workspace->d == nullptr) {
    return 2;
  }
  const int threads = 256;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  f32_to_bf16_attention_grad_kernel<<<blocks, threads, 0, stream>>>(
      grad_out, workspace->go_bf, batch, heads, seq_len, head_dim, grad_out_merged);
  __nv_bfloat16* saved_q = const_cast<__nv_bfloat16*>(
      reinterpret_cast<const __nv_bfloat16*>(saved_q_bf16_bits));
  __nv_bfloat16* saved_k = const_cast<__nv_bfloat16*>(
      reinterpret_cast<const __nv_bfloat16*>(saved_k_bf16_bits));
  __nv_bfloat16* saved_v = const_cast<__nv_bfloat16*>(
      reinterpret_cast<const __nv_bfloat16*>(saved_v_bf16_bits));
  __nv_bfloat16* saved_o = const_cast<__nv_bfloat16*>(
      reinterpret_cast<const __nv_bfloat16*>(saved_o_bf16_bits));
  float* saved_lse_mut = const_cast<float*>(saved_lse);
  if (head_dim == 64) {
    llmk::attention::launch_backward_causal<64>(
        llmk::to_bf16(saved_q), llmk::to_bf16(saved_k),
        llmk::to_bf16(saved_v), llmk::to_bf16(saved_o),
        saved_lse_mut, llmk::to_bf16(workspace->go_bf), workspace->d,
        llmk::to_bf16(workspace->gq_bf), llmk::to_bf16(workspace->gk_bf),
        llmk::to_bf16(workspace->gv_bf), static_cast<int>(batch), static_cast<int>(heads),
        static_cast<int>(seq_len), stream);
  } else {
    llmk::attention::launch_backward_causal<128>(
        llmk::to_bf16(saved_q), llmk::to_bf16(saved_k),
        llmk::to_bf16(saved_v), llmk::to_bf16(saved_o),
        saved_lse_mut, llmk::to_bf16(workspace->go_bf), workspace->d,
        llmk::to_bf16(workspace->gq_bf), llmk::to_bf16(workspace->gk_bf),
        llmk::to_bf16(workspace->gv_bf), static_cast<int>(batch), static_cast<int>(heads),
        static_cast<int>(seq_len), stream);
  }
  bf16_heads_to_qkv_float32_kernel<<<blocks, threads, 0, stream>>>(
      workspace->gq_bf,
      workspace->gk_bf,
      workspace->gv_bf,
      grad_qkv,
      batch,
      seq_len,
      heads,
      head_dim);
  g_attention_backward_tk_launch_count.fetch_add(1, std::memory_order_relaxed);
  return static_cast<int>(cudaPeekAtLastError());
#endif
}
#endif

#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
bool fits_cublas_int(std::int64_t value) {
  return value > 0 && value <= static_cast<std::int64_t>(std::numeric_limits<int>::max());
}

cublasHandle_t trainer_linear_cublas_handle(cudaStream_t stream) {
  static cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      handle = nullptr;
      return nullptr;
    }
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
  }
  if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
    return nullptr;
  }
  return handle;
}

bool trainer_linear_cublas_prewarm_internal(cudaStream_t stream) {
  return trainer_linear_cublas_handle(stream) != nullptr;
}

cublasLtHandle_t trainer_linear_cublaslt_handle() {
  static cublasLtHandle_t handle = nullptr;
  if (handle == nullptr && cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    handle = nullptr;
  }
  return handle;
}

struct TrainerLinearCublasLtPlanKey {
  int m = 0;
  int n = 0;
  int k = 0;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  int op_a = CUBLAS_OP_N;
  int op_b = CUBLAS_OP_N;
  int a_type = CUDA_R_32F;
  int b_type = CUDA_R_32F;
  int c_type = CUDA_R_32F;
  int compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  int epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  bool operator==(const TrainerLinearCublasLtPlanKey& other) const {
    return m == other.m && n == other.n && k == other.k && lda == other.lda &&
        ldb == other.ldb && ldc == other.ldc && op_a == other.op_a && op_b == other.op_b &&
        a_type == other.a_type && b_type == other.b_type && c_type == other.c_type &&
        compute_type == other.compute_type && epilogue == other.epilogue;
  }
};

struct TrainerLinearCublasLtPlan {
  TrainerLinearCublasLtPlanKey key;
  cublasLtMatmulDesc_t matmul_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulAlgo_t algo{};
  std::size_t workspace_size = 0;
  int selected_heuristic = -1;
  int returned_heuristics = 0;
  bool valid = false;
};

struct TrainerLinearCublasLtWorkspace {
  void* data = nullptr;
  std::size_t capacity = 0;
  std::vector<TrainerLinearCublasLtPlan> plans;
};

struct TrainerLinearBgradWorkspace {
  float* data = nullptr;
  std::int64_t capacity = 0;
};

bool trainer_linear_cublaslt_enabled();
bool trainer_linear_cublaslt_descriptor_cache_enabled();
int trainer_linear_cublaslt_heuristic_index_override();
int trainer_linear_cublaslt_heuristic_policy();
int trainer_linear_cublaslt_shape_heuristic_index_override(
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b);

TrainerLinearCublasLtWorkspace g_trainer_linear_cublaslt_workspace;
std::mutex g_trainer_linear_cublaslt_workspace_mutex;
TrainerLinearBgradWorkspace g_trainer_linear_bgrad_workspace;
std::mutex g_trainer_linear_bgrad_workspace_mutex;
constexpr std::size_t kTrainerLinearCublasLtWorkspaceBytes = 128ull * 1024ull * 1024ull;
constexpr std::size_t kTrainerLinearCublasLtPlanLimit = 128;

std::size_t trainer_linear_cublaslt_workspace_limit_bytes() {
  static const std::size_t bytes = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_LINEAR_CUBLASLT_WORKSPACE_MB");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_CUBLASLT_WORKSPACE_MB");
    }
    if (value == nullptr || value[0] == '\0') {
      return kTrainerLinearCublasLtWorkspaceBytes;
    }
    char* end = nullptr;
    const unsigned long long mib = std::strtoull(value, &end, 10);
    if (end == value || mib == 0ull || mib > 2048ull) {
      return kTrainerLinearCublasLtWorkspaceBytes;
    }
    return static_cast<std::size_t>(
        mib * static_cast<unsigned long long>(1024ull * 1024ull));
  }();
  return bytes;
}

bool ensure_trainer_linear_cublaslt_workspace(std::size_t bytes) {
  if (bytes == 0) {
    return true;
  }
  if (g_trainer_linear_cublaslt_workspace.capacity >= bytes) {
    return true;
  }
  void* next = nullptr;
  if (cudaMalloc(&next, bytes) != cudaSuccess) {
    return false;
  }
  if (g_trainer_linear_cublaslt_workspace.data != nullptr) {
    cudaFree(g_trainer_linear_cublaslt_workspace.data);
  }
  g_trainer_linear_cublaslt_workspace.data = next;
  g_trainer_linear_cublaslt_workspace.capacity = bytes;
  return true;
}

float* ensure_trainer_linear_bgrad_workspace(std::int64_t elements) {
  if (elements <= 0) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(g_trainer_linear_bgrad_workspace_mutex);
  if (g_trainer_linear_bgrad_workspace.capacity >= elements &&
      g_trainer_linear_bgrad_workspace.data != nullptr) {
    return g_trainer_linear_bgrad_workspace.data;
  }
  float* next = nullptr;
  if (cudaMalloc(&next, static_cast<std::size_t>(elements) * sizeof(float)) != cudaSuccess) {
    return nullptr;
  }
  if (g_trainer_linear_bgrad_workspace.data != nullptr) {
    cudaFree(g_trainer_linear_bgrad_workspace.data);
  }
  g_trainer_linear_bgrad_workspace.data = next;
  g_trainer_linear_bgrad_workspace.capacity = elements;
  return g_trainer_linear_bgrad_workspace.data;
}

bool create_trainer_linear_cublaslt_layouts(
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    cudaDataType_t a_type,
    cudaDataType_t b_type,
    cudaDataType_t c_type,
    cublasComputeType_t compute_type,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    cublasLtEpilogue_t epilogue,
    float* bias_pointer,
    cublasLtMatmulDesc_t* matmul_desc,
    cublasLtMatrixLayout_t* a_desc,
    cublasLtMatrixLayout_t* b_desc,
    cublasLtMatrixLayout_t* c_desc) {
  *matmul_desc = nullptr;
  *a_desc = nullptr;
  *b_desc = nullptr;
  *c_desc = nullptr;
  if (cublasLtMatmulDescCreate(matmul_desc, compute_type, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) {
    return false;
  }
  if (cublasLtMatmulDescSetAttribute(
          *matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)) != CUBLAS_STATUS_SUCCESS ||
      cublasLtMatmulDescSetAttribute(
          *matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)) != CUBLAS_STATUS_SUCCESS) {
    return false;
  }
  if (epilogue != CUBLASLT_EPILOGUE_DEFAULT &&
      cublasLtMatmulDescSetAttribute(
          *matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)) !=
          CUBLAS_STATUS_SUCCESS) {
    return false;
  }
  if (bias_pointer != nullptr) {
    cudaDataType_t bias_data_type = c_type;
    if (cublasLtMatmulDescSetAttribute(
            *matmul_desc,
            CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
            &bias_data_type,
            sizeof(bias_data_type)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(
            *matmul_desc,
            CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            &bias_pointer,
            sizeof(bias_pointer)) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }
  }

  const int a_rows = op_a == CUBLAS_OP_N ? m : k;
  const int a_cols = op_a == CUBLAS_OP_N ? k : m;
  const int b_rows = op_b == CUBLAS_OP_N ? k : n;
  const int b_cols = op_b == CUBLAS_OP_N ? n : k;
  if (cublasLtMatrixLayoutCreate(a_desc, a_type, a_rows, a_cols, lda) != CUBLAS_STATUS_SUCCESS ||
      cublasLtMatrixLayoutCreate(b_desc, b_type, b_rows, b_cols, ldb) != CUBLAS_STATUS_SUCCESS ||
      cublasLtMatrixLayoutCreate(c_desc, c_type, m, n, ldc) != CUBLAS_STATUS_SUCCESS) {
    return false;
  }
  return true;
}

void destroy_trainer_linear_cublaslt_layouts(
    cublasLtMatmulDesc_t matmul_desc,
    cublasLtMatrixLayout_t a_desc,
    cublasLtMatrixLayout_t b_desc,
    cublasLtMatrixLayout_t c_desc) {
  if (c_desc != nullptr) cublasLtMatrixLayoutDestroy(c_desc);
  if (b_desc != nullptr) cublasLtMatrixLayoutDestroy(b_desc);
  if (a_desc != nullptr) cublasLtMatrixLayoutDestroy(a_desc);
  if (matmul_desc != nullptr) cublasLtMatmulDescDestroy(matmul_desc);
}

void destroy_trainer_linear_cublaslt_plan(TrainerLinearCublasLtPlan& plan) {
  destroy_trainer_linear_cublaslt_layouts(
      plan.matmul_desc,
      plan.a_desc,
      plan.b_desc,
      plan.c_desc);
  plan.matmul_desc = nullptr;
  plan.a_desc = nullptr;
  plan.b_desc = nullptr;
  plan.c_desc = nullptr;
  plan.valid = false;
}

TrainerLinearCublasLtPlan* trainer_linear_cublaslt_plan_for(
    cublasLtHandle_t handle,
    const TrainerLinearCublasLtPlanKey& key,
    cublasLtMatmulDesc_t matmul_desc,
    cublasLtMatrixLayout_t a_desc,
    cublasLtMatrixLayout_t b_desc,
    cublasLtMatrixLayout_t c_desc) {
  for (TrainerLinearCublasLtPlan& plan : g_trainer_linear_cublaslt_workspace.plans) {
    if (plan.valid && plan.key == key) {
      return &plan;
    }
  }
  cublasLtMatmulPreference_t preference = nullptr;
  if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS) {
    return nullptr;
  }
  const std::size_t max_workspace = trainer_linear_cublaslt_workspace_limit_bytes();
  const cublasStatus_t pref_status = cublasLtMatmulPreferenceSetAttribute(
      preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &max_workspace,
      sizeof(max_workspace));
  if (pref_status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulPreferenceDestroy(preference);
    return nullptr;
  }
  std::array<cublasLtMatmulHeuristicResult_t, 32> results{};
  int returned = 0;
  const cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
      handle,
      matmul_desc,
      a_desc,
      b_desc,
      c_desc,
      c_desc,
      preference,
      static_cast<int>(results.size()),
      results.data(),
      &returned);
  cublasLtMatmulPreferenceDestroy(preference);
  if (heuristic_status != CUBLAS_STATUS_SUCCESS || returned <= 0) {
    return nullptr;
  }

  int selected = returned > 1 ? 1 : 0;
  const int policy = trainer_linear_cublaslt_heuristic_policy();
  if (policy == 1 || policy == 2) {
    selected = 0;
    for (int i = 1; i < returned; ++i) {
      const bool better = policy == 1
          ? results[i].wavesCount < results[selected].wavesCount
          : results[i].wavesCount > results[selected].wavesCount;
      if (better) {
        selected = i;
      }
    }
  }
  int requested_index =
      trainer_linear_cublaslt_shape_heuristic_index_override(
          key.m,
          key.n,
          key.k,
          static_cast<cublasOperation_t>(key.op_a),
          static_cast<cublasOperation_t>(key.op_b));
  if (requested_index < 0) {
    requested_index = trainer_linear_cublaslt_heuristic_index_override();
  }
  if (requested_index >= 0 && requested_index < returned) {
    selected = requested_index;
  }
  if (!ensure_trainer_linear_cublaslt_workspace(results[selected].workspaceSize)) {
    return nullptr;
  }
  if (g_trainer_linear_cublaslt_workspace.plans.size() >= kTrainerLinearCublasLtPlanLimit) {
    destroy_trainer_linear_cublaslt_plan(g_trainer_linear_cublaslt_workspace.plans.front());
    g_trainer_linear_cublaslt_workspace.plans.erase(g_trainer_linear_cublaslt_workspace.plans.begin());
  }
  TrainerLinearCublasLtPlan plan{};
  plan.key = key;
  if (trainer_linear_cublaslt_descriptor_cache_enabled()) {
    plan.matmul_desc = matmul_desc;
    plan.a_desc = a_desc;
    plan.b_desc = b_desc;
    plan.c_desc = c_desc;
  }
  plan.algo = results[selected].algo;
  plan.workspace_size = results[selected].workspaceSize;
  plan.selected_heuristic = selected;
  plan.returned_heuristics = returned;
  plan.valid = true;
  g_trainer_linear_cublaslt_workspace.plans.push_back(plan);
  return &g_trainer_linear_cublaslt_workspace.plans.back();
}

bool find_trainer_linear_cublaslt_plan(
    const TrainerLinearCublasLtPlanKey& key,
    TrainerLinearCublasLtPlan* plan_copy) {
  std::lock_guard<std::mutex> lock(g_trainer_linear_cublaslt_workspace_mutex);
  for (const TrainerLinearCublasLtPlan& plan : g_trainer_linear_cublaslt_workspace.plans) {
    if (plan.valid && plan.key == key) {
      *plan_copy = plan;
      return true;
    }
  }
  return false;
}

bool set_trainer_linear_cublaslt_runtime_pointers(
    cublasLtMatmulDesc_t matmul_desc,
    float* bias_pointer) {
  if (bias_pointer == nullptr) {
    return true;
  }
  return cublasLtMatmulDescSetAttribute(
             matmul_desc,
             CUBLASLT_MATMUL_DESC_BIAS_POINTER,
             &bias_pointer,
             sizeof(bias_pointer)) == CUBLAS_STATUS_SUCCESS;
}

bool cublaslt_linear_matmul(
    const void* a,
    const void* b,
    void* c,
    cudaDataType_t a_type,
    cudaDataType_t b_type,
    cudaDataType_t c_type,
    cublasComputeType_t compute_type,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int lda,
    int ldb,
    int ldc,
    float beta_value,
    cublasLtEpilogue_t epilogue,
    float* bias_pointer,
    cudaStream_t stream) {
  cublasLtHandle_t handle = trainer_linear_cublaslt_handle();
  if (handle == nullptr) {
    return false;
  }

  TrainerLinearCublasLtPlanKey key{
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      static_cast<int>(op_a),
      static_cast<int>(op_b),
      static_cast<int>(a_type),
      static_cast<int>(b_type),
      static_cast<int>(c_type),
      static_cast<int>(compute_type),
      static_cast<int>(epilogue)};
  TrainerLinearCublasLtPlan plan_copy;
  bool has_plan = false;
  if (trainer_linear_cublaslt_descriptor_cache_enabled()) {
    has_plan = find_trainer_linear_cublaslt_plan(key, &plan_copy);
  }

  cublasLtMatmulDesc_t matmul_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  if (!has_plan) {
    if (!create_trainer_linear_cublaslt_layouts(
            op_a,
            op_b,
            a_type,
            b_type,
            c_type,
            compute_type,
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            epilogue,
            bias_pointer,
            &matmul_desc,
            &a_desc,
            &b_desc,
            &c_desc)) {
      destroy_trainer_linear_cublaslt_layouts(matmul_desc, a_desc, b_desc, c_desc);
      return false;
    }
    std::lock_guard<std::mutex> lock(g_trainer_linear_cublaslt_workspace_mutex);
    TrainerLinearCublasLtPlan* plan =
        trainer_linear_cublaslt_plan_for(handle, key, matmul_desc, a_desc, b_desc, c_desc);
    if (plan == nullptr) {
      destroy_trainer_linear_cublaslt_layouts(matmul_desc, a_desc, b_desc, c_desc);
      return false;
    }
    plan_copy = *plan;
  }

  cublasLtMatmulDesc_t launch_matmul_desc =
      plan_copy.matmul_desc != nullptr ? plan_copy.matmul_desc : matmul_desc;
  cublasLtMatrixLayout_t launch_a_desc = plan_copy.a_desc != nullptr ? plan_copy.a_desc : a_desc;
  cublasLtMatrixLayout_t launch_b_desc = plan_copy.b_desc != nullptr ? plan_copy.b_desc : b_desc;
  cublasLtMatrixLayout_t launch_c_desc = plan_copy.c_desc != nullptr ? plan_copy.c_desc : c_desc;
  const bool retained_new_descriptors = launch_matmul_desc == matmul_desc && matmul_desc != nullptr &&
      trainer_linear_cublaslt_descriptor_cache_enabled();
  if (retained_new_descriptors) {
    matmul_desc = nullptr;
    a_desc = nullptr;
    b_desc = nullptr;
    c_desc = nullptr;
  }
  if (!set_trainer_linear_cublaslt_runtime_pointers(launch_matmul_desc, bias_pointer)) {
    destroy_trainer_linear_cublaslt_layouts(matmul_desc, a_desc, b_desc, c_desc);
    return false;
  }

  const float alpha = 1.0f;
  const float beta = beta_value;
  void* workspace = plan_copy.workspace_size > 0 ? g_trainer_linear_cublaslt_workspace.data : nullptr;
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  const cublasStatus_t status = cublasLtMatmul(
      handle,
      launch_matmul_desc,
      &alpha,
      a,
      launch_a_desc,
      b,
      launch_b_desc,
      &beta,
      c,
      launch_c_desc,
      c,
      launch_c_desc,
      &plan_copy.algo,
      workspace,
      plan_copy.workspace_size,
      stream);
  destroy_trainer_linear_cublaslt_layouts(matmul_desc, a_desc, b_desc, c_desc);
  if (status == CUBLAS_STATUS_SUCCESS) {
    const std::int64_t elapsed_us = finish_linear_shape_timing(&timing);
    g_linear_cublaslt_gemm_count.fetch_add(1, std::memory_order_relaxed);
    if (epilogue == CUBLASLT_EPILOGUE_BGRADB) {
      g_linear_cublaslt_bgrad_gemm_count.fetch_add(1, std::memory_order_relaxed);
    }
    record_linear_shape_stat(
        1,
        m,
        n,
        k,
        op_a,
        op_b,
        elapsed_us,
        plan_copy.selected_heuristic,
        plan_copy.returned_heuristics,
        static_cast<std::int64_t>(plan_copy.workspace_size));
    return true;
  }
  discard_linear_shape_timing(&timing);
  return false;
}

bool cublaslt_linear_matmul(
    const void* a,
    const void* b,
    void* c,
    cudaDataType_t a_type,
    cudaDataType_t b_type,
    cudaDataType_t c_type,
    cublasComputeType_t compute_type,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int lda,
    int ldb,
    int ldc,
    float beta_value,
    cudaStream_t stream) {
  return cublaslt_linear_matmul(
      a,
      b,
      c,
      a_type,
      b_type,
      c_type,
      compute_type,
      m,
      n,
      k,
      op_a,
      op_b,
      lda,
      ldb,
      ldc,
      beta_value,
      CUBLASLT_EPILOGUE_DEFAULT,
      nullptr,
      stream);
}

bool cublaslt_linear_matmul_float32(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int lda,
    int ldb,
    int ldc,
    float beta_value,
    cudaStream_t stream) {
  if (!trainer_linear_cublaslt_enabled()) {
    return false;
  }
  return cublaslt_linear_matmul(
      a,
      b,
      c,
      CUDA_R_32F,
      CUDA_R_32F,
      CUDA_R_32F,
      CUBLAS_COMPUTE_32F_FAST_TF32,
      m,
      n,
      k,
      op_a,
      op_b,
      lda,
      ldb,
      ldc,
      beta_value,
      stream);
}

bool trainer_linear_bf16_cublaslt_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_NATIVE_LINEAR_BF16_CUBLASLT");
    if (value == nullptr) {
      value = std::getenv("NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT");
    }
    if (value == nullptr) {
      return true;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool trainer_linear_shape_stats_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_NATIVE_LINEAR_SHAPE_STATS");
    if (value == nullptr) {
      value = std::getenv("NFN_TILE_CUDA_LINEAR_SHAPE_STATS");
    }
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT_LINEAR_SHAPE_STATS");
    }
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT2_LINEAR_SHAPE_STATS");
    }
    if (value == nullptr) {
      return false;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

void record_linear_shape_stat(
    int path,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    std::int64_t elapsed_us,
    int cublaslt_selected_heuristic,
    int cublaslt_returned_heuristics,
    std::int64_t cublaslt_workspace_bytes) {
  if (!trainer_linear_shape_stats_enabled()) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_linear_shape_stats_mutex);
  for (LinearShapeStat& stat : g_linear_shape_stats) {
    if (stat.path == path && stat.m == m && stat.n == n && stat.k == k &&
        stat.op_a == static_cast<int>(op_a) && stat.op_b == static_cast<int>(op_b)) {
      stat.calls += 1;
      stat.total_us += elapsed_us;
      if (path == 1 && cublaslt_selected_heuristic >= 0) {
        stat.cublaslt_selected_heuristic = cublaslt_selected_heuristic;
        stat.cublaslt_returned_heuristics = cublaslt_returned_heuristics;
        stat.cublaslt_workspace_bytes = cublaslt_workspace_bytes;
      }
      return;
    }
  }
  if (g_linear_shape_stats.size() >= 256) {
    return;
  }
  g_linear_shape_stats.push_back(
      LinearShapeStat{
          path,
          m,
          n,
          k,
          static_cast<int>(op_a),
          static_cast<int>(op_b),
          path == 1 ? cublaslt_selected_heuristic : -1,
          path == 1 ? cublaslt_returned_heuristics : 0,
          path == 1 ? cublaslt_workspace_bytes : 0,
          1,
          elapsed_us});
}

LinearShapeTiming begin_linear_shape_timing(cudaStream_t stream) {
  LinearShapeTiming timing;
  if (!trainer_linear_shape_stats_enabled()) {
    return timing;
  }
  timing.stream = stream;
  if (cudaEventCreate(&timing.start) != cudaSuccess) {
    timing.start = nullptr;
    return timing;
  }
  if (cudaEventCreate(&timing.stop) != cudaSuccess) {
    cudaEventDestroy(timing.start);
    timing.start = nullptr;
    timing.stop = nullptr;
    return timing;
  }
  if (cudaEventRecord(timing.start, stream) != cudaSuccess) {
    cudaEventDestroy(timing.start);
    cudaEventDestroy(timing.stop);
    timing.start = nullptr;
    timing.stop = nullptr;
    return timing;
  }
  timing.active = true;
  return timing;
}

std::int64_t finish_linear_shape_timing(LinearShapeTiming* timing) {
  if (timing == nullptr || !timing->active) {
    return 0;
  }
  float elapsed_ms = 0.0f;
  if (cudaEventRecord(timing->stop, timing->stream) == cudaSuccess &&
      cudaEventSynchronize(timing->stop) == cudaSuccess &&
      cudaEventElapsedTime(&elapsed_ms, timing->start, timing->stop) == cudaSuccess) {
    // Keep JSON compact and integer-only; sub-microsecond values round up to 1 us.
    const float elapsed_us = elapsed_ms * 1000.0f;
    cudaEventDestroy(timing->start);
    cudaEventDestroy(timing->stop);
    timing->active = false;
    return elapsed_us > 0.0f ? static_cast<std::int64_t>(elapsed_us + 0.5f) : 0;
  }
  cudaEventDestroy(timing->start);
  cudaEventDestroy(timing->stop);
  timing->active = false;
  return 0;
}

std::int64_t finish_linear_shape_timing_with_host_fallback(
    LinearShapeTiming* timing,
    std::chrono::steady_clock::time_point host_start) {
  std::int64_t elapsed_us = finish_linear_shape_timing(timing);
  if (elapsed_us > 0 || !trainer_linear_shape_stats_enabled()) {
    return elapsed_us;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    return elapsed_us;
  }
  const auto host_stop = std::chrono::steady_clock::now();
  const auto host_elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(host_stop - host_start).count();
  return host_elapsed > 0 ? static_cast<std::int64_t>(host_elapsed) : 1;
}

void discard_linear_shape_timing(LinearShapeTiming* timing) {
  if (timing == nullptr || !timing->active) {
    return;
  }
  cudaEventDestroy(timing->start);
  cudaEventDestroy(timing->stop);
  timing->active = false;
}

bool parse_cublas_op_token(const char* token, cublasOperation_t* op) {
  if (token == nullptr || op == nullptr) {
    return false;
  }
  if (std::strcmp(token, "N") == 0 || std::strcmp(token, "n") == 0 ||
      std::strcmp(token, "0") == 0) {
    *op = CUBLAS_OP_N;
    return true;
  }
  if (std::strcmp(token, "T") == 0 || std::strcmp(token, "t") == 0 ||
      std::strcmp(token, "1") == 0) {
    *op = CUBLAS_OP_T;
    return true;
  }
  return false;
}

bool parse_linear_shape_token(const char* value, LinearShapeStat* shape) {
  if (value == nullptr || value[0] == '\0' || shape == nullptr) {
    return false;
  }
  int parsed_m = 0;
  int parsed_n = 0;
  int parsed_k = 0;
  char parsed_op_a[8] = {};
  char parsed_op_b[8] = {};
  if (std::sscanf(
          value,
          "%d,%d,%d,%7[^,],%7s",
          &parsed_m,
          &parsed_n,
          &parsed_k,
          parsed_op_a,
          parsed_op_b) != 5) {
    return false;
  }
  cublasOperation_t parsed_a = CUBLAS_OP_N;
  cublasOperation_t parsed_b = CUBLAS_OP_N;
  if (parsed_m <= 0 || parsed_n <= 0 || parsed_k <= 0 ||
      !parse_cublas_op_token(parsed_op_a, &parsed_a) ||
      !parse_cublas_op_token(parsed_op_b, &parsed_b)) {
    return false;
  }
  shape->path = 2;
  shape->m = parsed_m;
  shape->n = parsed_n;
  shape->k = parsed_k;
  shape->op_a = static_cast<int>(parsed_a);
  shape->op_b = static_cast<int>(parsed_b);
  return true;
}

bool linear_shape_matches(
    const LinearShapeStat& shape,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b) {
  return shape.path == 2 && shape.m == m && shape.n == n && shape.k == k &&
      shape.op_a == static_cast<int>(op_a) &&
      shape.op_b == static_cast<int>(op_b);
}

struct LinearShapeList {
  std::array<LinearShapeStat, 16> shapes{};
  int count = 0;
};

LinearShapeList parse_linear_shape_list(const char* value) {
  LinearShapeList list{};
  if (value == nullptr || value[0] == '\0') {
    return list;
  }
  std::array<char, 1024> buffer{};
  const std::size_t length = std::min(std::strlen(value), buffer.size() - 1);
  std::memcpy(buffer.data(), value, length);
  buffer[length] = '\0';

  char* token_start = buffer.data();
  for (std::size_t index = 0; index <= length; ++index) {
    char& current = buffer[index];
    if (current != ':' && current != ';' && current != ' ' &&
        current != '\t' && current != '\n' && current != '\0') {
      continue;
    }
    const char separator = current;
    current = '\0';
    if (token_start[0] != '\0' &&
        list.count < static_cast<int>(list.shapes.size())) {
      LinearShapeStat shape{};
      if (parse_linear_shape_token(token_start, &shape)) {
        list.shapes[static_cast<std::size_t>(list.count)] = shape;
        ++list.count;
      }
    }
    if (separator == '\0') {
      break;
    }
    token_start = buffer.data() + index + 1;
  }
  return list;
}

bool linear_shape_list_matches(
    const LinearShapeList& list,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b) {
  for (int index = 0; index < list.count; ++index) {
    if (linear_shape_matches(
            list.shapes[static_cast<std::size_t>(index)],
            m,
            n,
            k,
            op_a,
            op_b)) {
      return true;
    }
  }
  return false;
}

struct CublasLtHeuristicShapeOverride {
  int m = 0;
  int n = 0;
  int k = 0;
  int op_a = CUBLAS_OP_N;
  int op_b = CUBLAS_OP_N;
  int index = -1;
  bool valid = false;
};

struct CublasLtHeuristicShapeOverrideList {
  std::array<CublasLtHeuristicShapeOverride, 16> shapes{};
  int count = 0;
};

bool parse_cublaslt_heuristic_shape_override_token(
    const char* value,
    CublasLtHeuristicShapeOverride* shape) {
  if (value == nullptr || value[0] == '\0' || shape == nullptr) {
    return false;
  }
  int parsed_m = 0;
  int parsed_n = 0;
  int parsed_k = 0;
  int parsed_index = -1;
  char parsed_op_a[8] = {};
  char parsed_op_b[8] = {};
  if (std::sscanf(
          value,
          "%d,%d,%d,%7[^,],%7[^,],%d",
          &parsed_m,
          &parsed_n,
          &parsed_k,
          parsed_op_a,
          parsed_op_b,
          &parsed_index) != 6) {
    return false;
  }
  cublasOperation_t parsed_a = CUBLAS_OP_N;
  cublasOperation_t parsed_b = CUBLAS_OP_N;
  if (parsed_m <= 0 || parsed_n <= 0 || parsed_k <= 0 ||
      parsed_index < 0 || parsed_index > 31 ||
      !parse_cublas_op_token(parsed_op_a, &parsed_a) ||
      !parse_cublas_op_token(parsed_op_b, &parsed_b)) {
    return false;
  }
  shape->m = parsed_m;
  shape->n = parsed_n;
  shape->k = parsed_k;
  shape->op_a = static_cast<int>(parsed_a);
  shape->op_b = static_cast<int>(parsed_b);
  shape->index = parsed_index;
  shape->valid = true;
  return true;
}

CublasLtHeuristicShapeOverrideList parse_cublaslt_heuristic_shape_override_list(
    const char* value) {
  CublasLtHeuristicShapeOverrideList list{};
  if (value == nullptr || value[0] == '\0') {
    return list;
  }
  std::array<char, 2048> buffer{};
  const std::size_t length = std::min(std::strlen(value), buffer.size() - 1);
  std::memcpy(buffer.data(), value, length);
  buffer[length] = '\0';

  char* token_start = buffer.data();
  for (std::size_t index = 0; index <= length; ++index) {
    char& current = buffer[index];
    if (current != ':' && current != ';' && current != ' ' &&
        current != '\t' && current != '\n' && current != '\0') {
      continue;
    }
    const char separator = current;
    current = '\0';
    if (token_start[0] != '\0' &&
        list.count < static_cast<int>(list.shapes.size())) {
      CublasLtHeuristicShapeOverride shape{};
      if (parse_cublaslt_heuristic_shape_override_token(token_start, &shape)) {
        list.shapes[static_cast<std::size_t>(list.count)] = shape;
        ++list.count;
      }
    }
    if (separator == '\0') {
      break;
    }
    token_start = buffer.data() + index + 1;
  }
  return list;
}

bool trainer_linear_bf16_cublaslt_shape_disabled(
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b) {
  static const LinearShapeStat disabled_shape = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE");
    }
    LinearShapeStat shape{};
    if (value == nullptr || value[0] == '\0') {
      return shape;
    }
    int parsed_m = 0;
    int parsed_n = 0;
    int parsed_k = 0;
    char parsed_op_a[8] = {};
    char parsed_op_b[8] = {};
    if (std::sscanf(
            value,
            "%d,%d,%d,%7[^,],%7s",
            &parsed_m,
            &parsed_n,
            &parsed_k,
            parsed_op_a,
            parsed_op_b) != 5) {
      return shape;
    }
    cublasOperation_t parsed_a = CUBLAS_OP_N;
    cublasOperation_t parsed_b = CUBLAS_OP_N;
    if (parsed_m <= 0 || parsed_n <= 0 || parsed_k <= 0 ||
        !parse_cublas_op_token(parsed_op_a, &parsed_a) ||
        !parse_cublas_op_token(parsed_op_b, &parsed_b)) {
      return shape;
    }
    shape.path = 1;
    shape.m = parsed_m;
    shape.n = parsed_n;
    shape.k = parsed_k;
    shape.op_a = static_cast<int>(parsed_a);
    shape.op_b = static_cast<int>(parsed_b);
    return shape;
  }();
  return disabled_shape.path == 1 && disabled_shape.m == m && disabled_shape.n == n &&
      disabled_shape.k == k && disabled_shape.op_a == static_cast<int>(op_a) &&
      disabled_shape.op_b == static_cast<int>(op_b);
}

bool trainer_linear_bf16_cublaslt_shape_enabled(
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b) {
  static const LinearShapeList enabled_shapes = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE");
    }
    return parse_linear_shape_list(value);
  }();
  return linear_shape_list_matches(enabled_shapes, m, n, k, op_a, op_b);
}

bool trainer_linear_bf16_cublaslt_shape_supported(
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b) {
  if (trainer_linear_bf16_cublaslt_shape_disabled(m, n, k, op_a, op_b)) {
    return false;
  }
  if (trainer_linear_bf16_cublaslt_shape_enabled(m, n, k, op_a, op_b)) {
    return true;
  }
  if (m <= 3072 && (n <= 3072 || k <= 3072)) {
    return true;
  }
  static const bool large_shape_enabled = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_LARGE_SHAPES");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_BF16_CUBLASLT_LARGE_SHAPES");
    }
    if (value == nullptr) {
      return true;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  static const bool extra_large_k_enabled = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K");
    }
    if (value == nullptr) {
      return false;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return large_shape_enabled &&
      m <= 50304 &&
      n <= 50304 &&
      (k <= 32768 || (extra_large_k_enabled && k <= 65536));
}

bool trainer_linear_bf16_output_cublaslt_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_LINEAR_BF16_OUTPUT_CUBLASLT");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_BF16_OUTPUT_CUBLASLT");
    }
    if (value == nullptr) {
      return false;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool trainer_linear_tk_forward_shape_disabled(
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b) {
  static const LinearShapeStat enabled_shape = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_LINEAR_TK_FORWARD_ENABLE_SHAPE");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_TK_FORWARD_ENABLE_SHAPE");
    }
    LinearShapeStat shape{};
    parse_linear_shape_token(value, &shape);
    return shape;
  }();
  static const LinearShapeStat disabled_shape = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_LINEAR_TK_FORWARD_DISABLE_SHAPE");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE");
    }
    LinearShapeStat shape{};
    parse_linear_shape_token(value, &shape);
    return shape;
  }();
  if (linear_shape_matches(enabled_shape, m, n, k, op_a, op_b)) {
    return false;
  }
  return linear_shape_matches(disabled_shape, m, n, k, op_a, op_b);
}

bool trainer_linear_float32_bf16_bgrad_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT2_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD");
    }
    if (value == nullptr) {
      value = std::getenv("NFN_TILE_CUDA_LINEAR_FLOAT32_BF16_BGRAD");
    }
    if (value == nullptr) {
      return true;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool trainer_linear_bf16_bf16_bgrad_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_NATIVE_GPT_FUSE_BF16_BF16_DWEIGHT_BGRAD");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT2_FUSE_BF16_BF16_DWEIGHT_BGRAD");
    }
    if (value == nullptr) {
      value = std::getenv("NFN_TILE_CUDA_LINEAR_BF16_BF16_BGRAD");
    }
    if (value == nullptr) {
      return true;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool trainer_linear_bf16_bf16_bgrad_disabled_for_shape(
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b) {
  static const LinearShapeList disabled_shapes = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_LINEAR_BF16_BF16_BGRAD_DISABLE_SHAPE");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_BF16_BF16_BGRAD_DISABLE_SHAPE");
    }
    return parse_linear_shape_list(value);
  }();
  return linear_shape_list_matches(disabled_shapes, m, n, k, op_a, op_b);
}

bool trainer_linear_bf16_bridge_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_LINEAR_BF16");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_BF16");
    }
    if (value == nullptr) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool trainer_linear_bgrad_first_write_direct_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT2_BGRAD_FIRST_WRITE_DIRECT");
    }
    if (value == nullptr) {
      value = std::getenv("NFN_TILE_CUDA_LINEAR_BGRAD_FIRST_WRITE_DIRECT");
    }
    if (value == nullptr || value[0] == '\0') {
      return false;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool trainer_linear_cublaslt_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_LINEAR_CUBLASLT");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_CUBLASLT");
    }
    if (value == nullptr) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool trainer_linear_cublaslt_descriptor_cache_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_CUBLASLT_DESCRIPTOR_CACHE");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_CUBLASLT_DESCRIPTOR_CACHE");
    }
    if (value == nullptr) {
      return true;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

int trainer_linear_cublaslt_heuristic_index_override() {
  static const int index = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_CUBLASLT_HEURISTIC_INDEX");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX");
    }
    if (value == nullptr || value[0] == '\0') {
      return -1;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || (end != nullptr && *end != '\0') || parsed < 0 || parsed > 31) {
      return -1;
    }
    return static_cast<int>(parsed);
  }();
  return index;
}

int trainer_linear_cublaslt_heuristic_policy() {
  static const int policy = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_CUBLASLT_HEURISTIC_POLICY");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_POLICY");
    }
    if (value == nullptr || value[0] == '\0') {
      return 0;
    }
    if (std::strcmp(value, "min_waves") == 0 ||
        std::strcmp(value, "MIN_WAVES") == 0 ||
        std::strcmp(value, "llmk") == 0 ||
        std::strcmp(value, "LLMK") == 0) {
      return 1;
    }
    if (std::strcmp(value, "max_waves") == 0 ||
        std::strcmp(value, "MAX_WAVES") == 0) {
      return 2;
    }
    return 0;
  }();
  return policy;
}

int trainer_linear_cublaslt_shape_heuristic_index_override(
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b) {
  static const CublasLtHeuristicShapeOverrideList overrides = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_CUBLASLT_HEURISTIC_SHAPE");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE");
    }
    return parse_cublaslt_heuristic_shape_override_list(value);
  }();
  for (int index = 0; index < overrides.count; ++index) {
    const CublasLtHeuristicShapeOverride& override =
        overrides.shapes[static_cast<std::size_t>(index)];
    if (override.valid && override.m == m && override.n == n && override.k == k &&
        override.op_a == static_cast<int>(op_a) &&
        override.op_b == static_cast<int>(op_b)) {
      return override.index;
    }
  }
  return -1;
}

cublasComputeType_t trainer_linear_bf16_gemm_ex_compute_type(
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b) {
  static const LinearShapeStat fast_16bf_shape = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_FAST_16BF_SHAPE");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF_SHAPE");
    }
    LinearShapeStat shape{};
    parse_linear_shape_token(value, &shape);
    return shape;
  }();
  if (linear_shape_matches(fast_16bf_shape, m, n, k, op_a, op_b)) {
    g_linear_bf16_gemm_fast16bf_request_count.fetch_add(1, std::memory_order_relaxed);
    return CUBLAS_COMPUTE_32F_FAST_16BF;
  }
  static const bool fast_16bf = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_FAST_16BF");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF");
    }
    if (value == nullptr || value[0] == '\0') {
      return false;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  if (fast_16bf) {
    g_linear_bf16_gemm_fast16bf_request_count.fetch_add(1, std::memory_order_relaxed);
    return CUBLAS_COMPUTE_32F_FAST_16BF;
  }
  return CUBLAS_COMPUTE_32F;
}

bool parse_bf16_gemm_ex_algo_token(const char* token, cublasGemmAlgo_t* algo) {
  if (token == nullptr || token[0] == '\0' || algo == nullptr) {
    return false;
  }
  if (std::strcmp(token, "default_tensor_op") == 0 ||
      std::strcmp(token, "DEFAULT_TENSOR_OP") == 0 ||
      std::strcmp(token, "tensor_op") == 0 ||
      std::strcmp(token, "TENSOR_OP") == 0) {
    *algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    return true;
  }
  if (std::strcmp(token, "default") == 0 || std::strcmp(token, "DEFAULT") == 0) {
    *algo = CUBLAS_GEMM_DEFAULT;
    return true;
  }
  char* end = nullptr;
  const long parsed = std::strtol(token, &end, 10);
  if (end == token || (end != nullptr && *end != '\0')) {
    return false;
  }
  if (parsed >= 0 && parsed <= 15) {
    *algo = static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_ALGO0_TENSOR_OP + parsed);
    return true;
  }
  if (parsed == static_cast<long>(CUBLAS_GEMM_DEFAULT) ||
      parsed == static_cast<long>(CUBLAS_GEMM_DEFAULT_TENSOR_OP) ||
      (parsed >= static_cast<long>(CUBLAS_GEMM_ALGO0_TENSOR_OP) &&
       parsed <= static_cast<long>(CUBLAS_GEMM_ALGO15_TENSOR_OP))) {
    *algo = static_cast<cublasGemmAlgo_t>(parsed);
    return true;
  }
  return false;
}

cublasGemmAlgo_t trainer_linear_bf16_gemm_ex_algo(
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    cublasGemmAlgo_t default_algo) {
  struct ShapeAlgoOverride {
    int m = 0;
    int n = 0;
    int k = 0;
    int op_a = CUBLAS_OP_N;
    int op_b = CUBLAS_OP_N;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    bool valid = false;
  };
  static const ShapeAlgoOverride shape_override = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_ALGO_SHAPE");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_BF16_GEMM_EX_ALGO_SHAPE");
    }
    ShapeAlgoOverride shape{};
    if (value == nullptr || value[0] == '\0') {
      return shape;
    }
    int parsed_m = 0;
    int parsed_n = 0;
    int parsed_k = 0;
    char parsed_op_a[8] = {};
    char parsed_op_b[8] = {};
    char parsed_algo[32] = {};
    if (std::sscanf(
            value,
            "%d,%d,%d,%7[^,],%7[^,],%31s",
            &parsed_m,
            &parsed_n,
            &parsed_k,
            parsed_op_a,
            parsed_op_b,
            parsed_algo) != 6) {
      return shape;
    }
    cublasOperation_t parsed_a = CUBLAS_OP_N;
    cublasOperation_t parsed_b = CUBLAS_OP_N;
    cublasGemmAlgo_t parsed_algo_value = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    if (parsed_m <= 0 || parsed_n <= 0 || parsed_k <= 0 ||
        !parse_cublas_op_token(parsed_op_a, &parsed_a) ||
        !parse_cublas_op_token(parsed_op_b, &parsed_b) ||
        !parse_bf16_gemm_ex_algo_token(parsed_algo, &parsed_algo_value)) {
      return shape;
    }
    shape.m = parsed_m;
    shape.n = parsed_n;
    shape.k = parsed_k;
    shape.op_a = static_cast<int>(parsed_a);
    shape.op_b = static_cast<int>(parsed_b);
    shape.algo = parsed_algo_value;
    shape.valid = true;
    return shape;
  }();
  if (shape_override.valid && shape_override.m == m && shape_override.n == n &&
      shape_override.k == k && shape_override.op_a == static_cast<int>(op_a) &&
      shape_override.op_b == static_cast<int>(op_b)) {
    return shape_override.algo;
  }
  struct GlobalAlgoOverride {
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    bool valid = false;
  };
  static const GlobalAlgoOverride global_override = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_ALGO");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_BF16_GEMM_EX_ALGO");
    }
    GlobalAlgoOverride override{};
    cublasGemmAlgo_t parsed = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    if (!parse_bf16_gemm_ex_algo_token(value, &parsed)) {
      return override;
    }
    override.algo = parsed;
    override.valid = true;
    return override;
  }();
  if (global_override.valid) {
    return global_override.algo;
  }
  return default_algo;
}

struct TrainerLinearBf16Workspace {
  struct CacheEntry {
    __nv_bfloat16* data = nullptr;
    const float* source = nullptr;
    std::int64_t capacity = 0;
    std::int64_t elements = 0;
    std::uint64_t last_use = 0;
    bool valid = false;
  };

  __nv_bfloat16* a = nullptr;
  __nv_bfloat16* b = nullptr;
  std::uint16_t* c_bits = nullptr;
  std::int64_t a_capacity = 0;
  std::int64_t b_capacity = 0;
  std::int64_t c_capacity = 0;
  std::uint64_t cache_clock = 0;
  std::vector<CacheEntry> cached_a_entries;
};

TrainerLinearBf16Workspace g_trainer_linear_bf16_workspace;
std::mutex g_trainer_linear_bf16_workspace_mutex;
constexpr std::size_t kTrainerLinearBf16CacheEntryLimit = 128;

std::int64_t trainer_linear_bf16_cached_a_total_capacity(const TrainerLinearBf16Workspace& workspace) {
  std::int64_t total = 0;
  for (const TrainerLinearBf16Workspace::CacheEntry& entry : workspace.cached_a_entries) {
    total += entry.capacity;
  }
  return total;
}

void update_trainer_linear_bf16_cache_stats(const TrainerLinearBf16Workspace& workspace) {
  g_linear_bf16_cache_entry_count.store(
      static_cast<std::int64_t>(workspace.cached_a_entries.size()),
      std::memory_order_relaxed);
  g_linear_bf16_cached_a_capacity.store(
      trainer_linear_bf16_cached_a_total_capacity(workspace),
      std::memory_order_relaxed);
}

void release_trainer_linear_bf16_workspace(TrainerLinearBf16Workspace& workspace) {
  if (workspace.a != nullptr) cudaFree(workspace.a);
  if (workspace.b != nullptr) cudaFree(workspace.b);
  if (workspace.c_bits != nullptr) cudaFree(workspace.c_bits);
  for (TrainerLinearBf16Workspace::CacheEntry& entry : workspace.cached_a_entries) {
    if (entry.data != nullptr) {
      cudaFree(entry.data);
    }
  }
  workspace = TrainerLinearBf16Workspace{};
  update_trainer_linear_bf16_cache_stats(workspace);
}

void invalidate_trainer_linear_bf16_cache(TrainerLinearBf16Workspace& workspace) {
  for (TrainerLinearBf16Workspace::CacheEntry& entry : workspace.cached_a_entries) {
    entry.source = nullptr;
    entry.elements = 0;
    entry.valid = false;
  }
}

TrainerLinearBf16Workspace* ensure_trainer_linear_bf16_workspace(
    std::int64_t a_elements,
    std::int64_t b_elements,
    std::int64_t c_elements = 0) {
  if (a_elements <= 0 || b_elements <= 0) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(g_trainer_linear_bf16_workspace_mutex);
  if (g_trainer_linear_bf16_workspace.a_capacity >= a_elements &&
      g_trainer_linear_bf16_workspace.b_capacity >= b_elements &&
      g_trainer_linear_bf16_workspace.c_capacity >= c_elements) {
    return &g_trainer_linear_bf16_workspace;
  }
  TrainerLinearBf16Workspace next;
  next.a_capacity = std::max(a_elements, g_trainer_linear_bf16_workspace.a_capacity);
  next.b_capacity = std::max(b_elements, g_trainer_linear_bf16_workspace.b_capacity);
  next.c_capacity = std::max(c_elements, g_trainer_linear_bf16_workspace.c_capacity);
  if (cudaMalloc(
          reinterpret_cast<void**>(&next.a),
          sizeof(__nv_bfloat16) * static_cast<std::size_t>(next.a_capacity)) != cudaSuccess ||
      cudaMalloc(
          reinterpret_cast<void**>(&next.b),
          sizeof(__nv_bfloat16) * static_cast<std::size_t>(next.b_capacity)) != cudaSuccess) {
    release_trainer_linear_bf16_workspace(next);
    return nullptr;
  }
  if (next.c_capacity > 0 &&
      cudaMalloc(
          reinterpret_cast<void**>(&next.c_bits),
          sizeof(std::uint16_t) * static_cast<std::size_t>(next.c_capacity)) != cudaSuccess) {
    release_trainer_linear_bf16_workspace(next);
    return nullptr;
  }
  next.cache_clock = g_trainer_linear_bf16_workspace.cache_clock;
  next.cached_a_entries = std::move(g_trainer_linear_bf16_workspace.cached_a_entries);
  g_trainer_linear_bf16_workspace.cached_a_entries.clear();
  release_trainer_linear_bf16_workspace(g_trainer_linear_bf16_workspace);
  g_trainer_linear_bf16_workspace = next;
  g_linear_bf16_workspace_allocation_count.fetch_add(1, std::memory_order_relaxed);
  g_linear_bf16_workspace_a_capacity.store(next.a_capacity, std::memory_order_relaxed);
  g_linear_bf16_workspace_b_capacity.store(next.b_capacity, std::memory_order_relaxed);
  update_trainer_linear_bf16_cache_stats(g_trainer_linear_bf16_workspace);
  return &g_trainer_linear_bf16_workspace;
}

bool prewarm_trainer_linear_bf16_workspace(
    std::int64_t a_elements,
    std::int64_t b_elements,
    std::int64_t c_elements) {
  return ensure_trainer_linear_bf16_workspace(a_elements, b_elements, c_elements) != nullptr;
}

TrainerLinearBf16Workspace::CacheEntry* trainer_linear_bf16_cache_entry_for(
    TrainerLinearBf16Workspace* workspace,
    const float* source,
    std::int64_t elements,
    bool* cache_hit) {
  if (cache_hit != nullptr) {
    *cache_hit = false;
  }
  workspace->cache_clock += 1;
  for (TrainerLinearBf16Workspace::CacheEntry& entry : workspace->cached_a_entries) {
    if (entry.valid && entry.source == source && entry.elements == elements) {
      entry.last_use = workspace->cache_clock;
      g_linear_bf16_a_cache_hit_count.fetch_add(1, std::memory_order_relaxed);
      if (cache_hit != nullptr) {
        *cache_hit = true;
      }
      return &entry;
    }
  }

  TrainerLinearBf16Workspace::CacheEntry* selected = nullptr;
  for (TrainerLinearBf16Workspace::CacheEntry& entry : workspace->cached_a_entries) {
    if (!entry.valid) {
      selected = &entry;
      break;
    }
  }
  if (selected == nullptr && workspace->cached_a_entries.size() < kTrainerLinearBf16CacheEntryLimit) {
    workspace->cached_a_entries.emplace_back();
    selected = &workspace->cached_a_entries.back();
    update_trainer_linear_bf16_cache_stats(*workspace);
  }
  if (selected == nullptr) {
    selected = &workspace->cached_a_entries.front();
    for (TrainerLinearBf16Workspace::CacheEntry& entry : workspace->cached_a_entries) {
      if (entry.last_use < selected->last_use) {
        selected = &entry;
      }
    }
  }
  if (selected->capacity < elements) {
    __nv_bfloat16* next = nullptr;
    if (cudaMalloc(
            reinterpret_cast<void**>(&next),
            sizeof(__nv_bfloat16) * static_cast<std::size_t>(elements)) != cudaSuccess) {
      return nullptr;
    }
    if (selected->data != nullptr) {
      cudaFree(selected->data);
    }
    selected->data = next;
    selected->capacity = elements;
    update_trainer_linear_bf16_cache_stats(*workspace);
  }
  selected->source = source;
  selected->elements = elements;
  selected->last_use = workspace->cache_clock;
  selected->valid = true;
  return selected;
}

__nv_bfloat16* trainer_linear_bf16_a_operand(
    TrainerLinearBf16Workspace* workspace,
    const float* a,
    std::int64_t a_elements,
    bool cache_a_operand,
    cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = static_cast<int>((a_elements + threads - 1) / threads);
  if (cache_a_operand) {
    bool cache_hit = false;
    TrainerLinearBf16Workspace::CacheEntry* entry =
        trainer_linear_bf16_cache_entry_for(workspace, a, a_elements, &cache_hit);
    if (entry == nullptr || entry->data == nullptr) {
      return nullptr;
    }
    if (cache_hit) {
      return entry->data;
    }
    f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(a, entry->data, a_elements);
    g_linear_bf16_a_pack_count.fetch_add(1, std::memory_order_relaxed);
    return entry->data;
  }
  f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(a, workspace->a, a_elements);
  g_linear_bf16_a_pack_count.fetch_add(1, std::memory_order_relaxed);
  return workspace->a;
}

__nv_bfloat16* trainer_linear_bf16_b_operand(
    TrainerLinearBf16Workspace* workspace,
    const float* b,
    std::int64_t b_elements,
    bool cache_b_operand,
    cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = static_cast<int>((b_elements + threads - 1) / threads);
  if (cache_b_operand) {
    bool cache_hit = false;
    TrainerLinearBf16Workspace::CacheEntry* entry =
        trainer_linear_bf16_cache_entry_for(workspace, b, b_elements, &cache_hit);
    if (entry == nullptr || entry->data == nullptr) {
      return nullptr;
    }
    if (cache_hit) {
      return entry->data;
    }
    f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(b, entry->data, b_elements);
    g_linear_bf16_a_pack_count.fetch_add(1, std::memory_order_relaxed);
    return entry->data;
  }
  f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(b, workspace->b, b_elements);
  g_linear_bf16_a_pack_count.fetch_add(1, std::memory_order_relaxed);
  return workspace->b;
}

bool tk_linear_gemm_bf16_forward_to_float32(
    const __nv_bfloat16* weight_bf16,
    const __nv_bfloat16* x_bf16,
    std::uint16_t* out_bf16_bits,
    float* out,
    int rows,
    int input_dim,
    int output_dim,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    cudaStream_t stream);

bool cublas_linear_gemm_ex_bf16_float32(
    const float* a,
    const float* b,
    float* c,
    std::int64_t a_elements,
    std::int64_t b_elements,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int lda,
    int ldb,
    int ldc,
    float beta_value,
    bool cache_a_operand,
    bool cache_b_operand,
    bool force_bf16,
    cudaStream_t stream) {
  if (!force_bf16 && !trainer_linear_bf16_bridge_enabled()) {
    return false;
  }
  const std::int64_t c_elements = static_cast<std::int64_t>(m) * n;
  TrainerLinearBf16Workspace* workspace = ensure_trainer_linear_bf16_workspace(a_elements, b_elements, c_elements);
  if (workspace == nullptr) {
    return false;
  }
  cublasHandle_t handle = trainer_linear_cublas_handle(stream);
  if (handle == nullptr) {
    return false;
  }
  __nv_bfloat16* a_bf16 = trainer_linear_bf16_a_operand(workspace, a, a_elements, cache_a_operand, stream);
  if (a_bf16 == nullptr) {
    return false;
  }
  __nv_bfloat16* b_bf16 = trainer_linear_bf16_b_operand(workspace, b, b_elements, cache_b_operand, stream);
  if (b_bf16 == nullptr) {
    return false;
  }
  if (beta_value == 0.0f &&
      tk_linear_gemm_bf16_forward_to_float32(
          a_bf16,
          b_bf16,
          workspace->c_bits,
          c,
          n,
          k,
          m,
          op_a,
          op_b,
          stream)) {
    return true;
  }
  if (trainer_linear_bf16_cublaslt_enabled() &&
      trainer_linear_bf16_cublaslt_shape_supported(m, n, k, op_a, op_b) &&
      cublaslt_linear_matmul(
          a_bf16,
          b_bf16,
          c,
          CUDA_R_16BF,
          CUDA_R_16BF,
          CUDA_R_32F,
          CUBLAS_COMPUTE_32F_FAST_16BF,
          m,
          n,
          k,
          op_a,
          op_b,
          lda,
          ldb,
          ldc,
          beta_value,
          stream)) {
    return true;
  }
  const float alpha = 1.0f;
  const float beta = beta_value;
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  const cublasStatus_t status = cublasGemmEx(
      handle,
      op_a,
      op_b,
      m,
      n,
      k,
      &alpha,
      a_bf16,
      CUDA_R_16BF,
      lda,
      b_bf16,
      CUDA_R_16BF,
      ldb,
      &beta,
      c,
      CUDA_R_32F,
      ldc,
      trainer_linear_bf16_gemm_ex_compute_type(m, n, k, op_a, op_b),
      trainer_linear_bf16_gemm_ex_algo(
          m, n, k, op_a, op_b, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  if (status == CUBLAS_STATUS_SUCCESS) {
    const std::int64_t elapsed_us = finish_linear_shape_timing(&timing);
    g_linear_bf16_gemm_count.fetch_add(1, std::memory_order_relaxed);
    record_linear_shape_stat(4, m, n, k, op_a, op_b, elapsed_us);
    return true;
  }
  discard_linear_shape_timing(&timing);
  return false;
}

bool cublas_linear_gemm_ex_bf16_bits_a_float32(
    const std::uint16_t* a_bf16_bits,
    const float* b,
    float* c,
    std::int64_t b_elements,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int lda,
    int ldb,
    int ldc,
    float beta_value,
    bool force_bf16,
    cudaStream_t stream) {
  if (!force_bf16 && !trainer_linear_bf16_bridge_enabled()) {
    return false;
  }
  TrainerLinearBf16Workspace* workspace = ensure_trainer_linear_bf16_workspace(1, b_elements);
  if (workspace == nullptr) {
    return false;
  }
  cublasHandle_t handle = trainer_linear_cublas_handle(stream);
  if (handle == nullptr) {
    return false;
  }
  constexpr int threads = 256;
  const int b_blocks = static_cast<int>((b_elements + threads - 1) / threads);
  f32_to_bf16_kernel<<<b_blocks, threads, 0, stream>>>(b, workspace->b, b_elements);
  const auto* a_bf16 = reinterpret_cast<const __nv_bfloat16*>(a_bf16_bits);
  if (trainer_linear_bf16_cublaslt_enabled() &&
      trainer_linear_bf16_cublaslt_shape_supported(m, n, k, op_a, op_b) &&
      cublaslt_linear_matmul(
          a_bf16,
          workspace->b,
          c,
          CUDA_R_16BF,
          CUDA_R_16BF,
          CUDA_R_32F,
          CUBLAS_COMPUTE_32F_FAST_16BF,
          m,
          n,
          k,
          op_a,
          op_b,
          lda,
          ldb,
          ldc,
          beta_value,
          stream)) {
    return true;
  }
  const float alpha = 1.0f;
  const float beta = beta_value;
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  const cublasStatus_t status = cublasGemmEx(
      handle,
      op_a,
      op_b,
      m,
      n,
      k,
      &alpha,
      a_bf16,
      CUDA_R_16BF,
      lda,
      workspace->b,
      CUDA_R_16BF,
      ldb,
      &beta,
      c,
      CUDA_R_32F,
      ldc,
      trainer_linear_bf16_gemm_ex_compute_type(m, n, k, op_a, op_b),
      trainer_linear_bf16_gemm_ex_algo(
          m, n, k, op_a, op_b, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  if (status == CUBLAS_STATUS_SUCCESS) {
    const std::int64_t elapsed_us = finish_linear_shape_timing(&timing);
    g_linear_bf16_gemm_count.fetch_add(1, std::memory_order_relaxed);
    record_linear_shape_stat(4, m, n, k, op_a, op_b, elapsed_us);
    return true;
  }
  discard_linear_shape_timing(&timing);
  return false;
}

bool cublas_linear_gemm_ex_bf16_float32_with_bgrad(
    const float* a,
    const float* b,
    float* c,
    float* bias_gradient,
    std::int64_t a_elements,
    std::int64_t b_elements,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int lda,
    int ldb,
    int ldc,
    float beta_value,
    bool cache_a_operand,
    bool cache_b_operand,
    bool force_bf16,
    cudaStream_t stream) {
  if (bias_gradient == nullptr || (!force_bf16 && !trainer_linear_bf16_bridge_enabled())) {
    return false;
  }
  const std::int64_t c_elements = static_cast<std::int64_t>(m) * n;
  TrainerLinearBf16Workspace* workspace = ensure_trainer_linear_bf16_workspace(a_elements, b_elements, c_elements);
  if (workspace == nullptr) {
    return false;
  }
  __nv_bfloat16* a_bf16 = trainer_linear_bf16_a_operand(workspace, a, a_elements, cache_a_operand, stream);
  if (a_bf16 == nullptr) {
    return false;
  }
  __nv_bfloat16* b_bf16 = trainer_linear_bf16_b_operand(workspace, b, b_elements, cache_b_operand, stream);
  if (b_bf16 == nullptr) {
    return false;
  }
  return trainer_linear_bf16_cublaslt_enabled() &&
      trainer_linear_bf16_cublaslt_shape_supported(m, n, k, op_a, op_b) &&
      cublaslt_linear_matmul(
          a_bf16,
          b_bf16,
          c,
          CUDA_R_16BF,
          CUDA_R_16BF,
          CUDA_R_32F,
          CUBLAS_COMPUTE_32F_FAST_16BF,
          m,
          n,
          k,
          op_a,
          op_b,
          lda,
          ldb,
          ldc,
          beta_value,
          CUBLASLT_EPILOGUE_BGRADB,
          bias_gradient,
          stream);
}

bool cublas_linear_gemm_ex_bf16_bits_a_float32_with_bgrad(
    const std::uint16_t* a_bf16_bits,
    const float* b,
    float* c,
    float* bias_gradient,
    std::int64_t b_elements,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int lda,
    int ldb,
    int ldc,
    float beta_value,
    bool force_bf16,
    cudaStream_t stream) {
  if (bias_gradient == nullptr || (!force_bf16 && !trainer_linear_bf16_bridge_enabled())) {
    return false;
  }
  TrainerLinearBf16Workspace* workspace = ensure_trainer_linear_bf16_workspace(1, b_elements);
  if (workspace == nullptr) {
    return false;
  }
  constexpr int threads = 256;
  const int b_blocks = static_cast<int>((b_elements + threads - 1) / threads);
  f32_to_bf16_kernel<<<b_blocks, threads, 0, stream>>>(b, workspace->b, b_elements);
  const auto* a_bf16 = reinterpret_cast<const __nv_bfloat16*>(a_bf16_bits);
  return trainer_linear_bf16_cublaslt_enabled() &&
      trainer_linear_bf16_cublaslt_shape_supported(m, n, k, op_a, op_b) &&
      cublaslt_linear_matmul(
          a_bf16,
          workspace->b,
          c,
          CUDA_R_16BF,
          CUDA_R_16BF,
          CUDA_R_32F,
          CUBLAS_COMPUTE_32F_FAST_16BF,
          m,
          n,
          k,
          op_a,
          op_b,
          lda,
          ldb,
          ldc,
          beta_value,
          CUBLASLT_EPILOGUE_BGRADB,
          bias_gradient,
          stream);
}

bool cublas_linear_gemm_ex_bf16_bits_ab_float32_with_bgrad(
    const std::uint16_t* a_bf16_bits,
    const std::uint16_t* b_bf16_bits,
    float* c,
    float* bias_gradient,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int lda,
    int ldb,
    int ldc,
    float beta_value,
    bool force_bf16,
    cudaStream_t stream) {
  if (bias_gradient == nullptr || (!force_bf16 && !trainer_linear_bf16_bridge_enabled())) {
    return false;
  }
  const auto* a_bf16 = reinterpret_cast<const __nv_bfloat16*>(a_bf16_bits);
  const auto* b_bf16 = reinterpret_cast<const __nv_bfloat16*>(b_bf16_bits);
  return trainer_linear_bf16_cublaslt_enabled() &&
      trainer_linear_bf16_cublaslt_shape_supported(m, n, k, op_a, op_b) &&
      cublaslt_linear_matmul(
          a_bf16,
          b_bf16,
          c,
          CUDA_R_16BF,
          CUDA_R_16BF,
          CUDA_R_32F,
          CUBLAS_COMPUTE_32F_FAST_16BF,
          m,
          n,
          k,
          op_a,
          op_b,
          lda,
          ldb,
          ldc,
          beta_value,
          CUBLASLT_EPILOGUE_BGRADB,
          bias_gradient,
          stream);
}

bool cublas_linear_gemm_ex_bf16_bits_b_float32(
    const float* a,
    const std::uint16_t* b_bf16_bits,
    float* c,
    std::int64_t a_elements,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int lda,
    int ldb,
    int ldc,
    float beta_value,
    bool cache_a_operand,
    bool force_bf16,
    cudaStream_t stream) {
  if (!force_bf16 && !trainer_linear_bf16_bridge_enabled()) {
    return false;
  }
  TrainerLinearBf16Workspace* workspace = ensure_trainer_linear_bf16_workspace(a_elements, 1);
  if (workspace == nullptr) {
    return false;
  }
  cublasHandle_t handle = trainer_linear_cublas_handle(stream);
  if (handle == nullptr) {
    return false;
  }
  __nv_bfloat16* a_bf16 = trainer_linear_bf16_a_operand(workspace, a, a_elements, cache_a_operand, stream);
  if (a_bf16 == nullptr) {
    return false;
  }
  const auto* b_bf16 = reinterpret_cast<const __nv_bfloat16*>(b_bf16_bits);
  if (trainer_linear_bf16_cublaslt_enabled() &&
      trainer_linear_bf16_cublaslt_shape_supported(m, n, k, op_a, op_b) &&
      cublaslt_linear_matmul(
          a_bf16,
          b_bf16,
          c,
          CUDA_R_16BF,
          CUDA_R_16BF,
          CUDA_R_32F,
          CUBLAS_COMPUTE_32F_FAST_16BF,
          m,
          n,
          k,
          op_a,
          op_b,
          lda,
          ldb,
          ldc,
          beta_value,
          stream)) {
    return true;
  }
  const float alpha = 1.0f;
  const float beta = beta_value;
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  const cublasStatus_t status = cublasGemmEx(
      handle,
      op_a,
      op_b,
      m,
      n,
      k,
      &alpha,
      a_bf16,
      CUDA_R_16BF,
      lda,
      b_bf16,
      CUDA_R_16BF,
      ldb,
      &beta,
      c,
      CUDA_R_32F,
      ldc,
      trainer_linear_bf16_gemm_ex_compute_type(m, n, k, op_a, op_b),
      trainer_linear_bf16_gemm_ex_algo(
          m, n, k, op_a, op_b, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  if (status == CUBLAS_STATUS_SUCCESS) {
    const std::int64_t elapsed_us = finish_linear_shape_timing(&timing);
    g_linear_bf16_gemm_count.fetch_add(1, std::memory_order_relaxed);
    record_linear_shape_stat(4, m, n, k, op_a, op_b, elapsed_us);
    return true;
  }
  discard_linear_shape_timing(&timing);
  return false;
}

bool cublas_linear_gemm_ex_bf16_bits_b_float32_with_bgrad(
    const float* a,
    const std::uint16_t* b_bf16_bits,
    float* c,
    float* bias_gradient,
    std::int64_t a_elements,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int lda,
    int ldb,
    int ldc,
    float beta_value,
    bool cache_a_operand,
    bool force_bf16,
    cudaStream_t stream) {
  if (bias_gradient == nullptr || (!force_bf16 && !trainer_linear_bf16_bridge_enabled())) {
    return false;
  }
  TrainerLinearBf16Workspace* workspace = ensure_trainer_linear_bf16_workspace(a_elements, 1);
  if (workspace == nullptr) {
    return false;
  }
  __nv_bfloat16* a_bf16 = trainer_linear_bf16_a_operand(workspace, a, a_elements, cache_a_operand, stream);
  if (a_bf16 == nullptr) {
    return false;
  }
  const auto* b_bf16 = reinterpret_cast<const __nv_bfloat16*>(b_bf16_bits);
  return trainer_linear_bf16_cublaslt_enabled() &&
      trainer_linear_bf16_cublaslt_shape_supported(m, n, k, op_a, op_b) &&
      cublaslt_linear_matmul(
          a_bf16,
          b_bf16,
          c,
          CUDA_R_16BF,
          CUDA_R_16BF,
          CUDA_R_32F,
          CUBLAS_COMPUTE_32F_FAST_16BF,
          m,
          n,
          k,
          op_a,
          op_b,
          lda,
          ldb,
          ldc,
          beta_value,
          CUBLASLT_EPILOGUE_BGRADB,
          bias_gradient,
          stream);
}

bool cublas_linear_gemm_ex_bf16_bits_ab_float32(
    const std::uint16_t* a_bf16_bits,
    const std::uint16_t* b_bf16_bits,
    float* c,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int lda,
    int ldb,
    int ldc,
    float beta_value,
    bool force_bf16,
    cudaStream_t stream) {
  if (!force_bf16 && !trainer_linear_bf16_bridge_enabled()) {
    return false;
  }
  cublasHandle_t handle = trainer_linear_cublas_handle(stream);
  if (handle == nullptr) {
    return false;
  }
  const auto* a_bf16 = reinterpret_cast<const __nv_bfloat16*>(a_bf16_bits);
  const auto* b_bf16 = reinterpret_cast<const __nv_bfloat16*>(b_bf16_bits);
  if (trainer_linear_bf16_cublaslt_enabled() &&
      trainer_linear_bf16_cublaslt_shape_supported(m, n, k, op_a, op_b) &&
      cublaslt_linear_matmul(
          a_bf16,
          b_bf16,
          c,
          CUDA_R_16BF,
          CUDA_R_16BF,
          CUDA_R_32F,
          CUBLAS_COMPUTE_32F_FAST_16BF,
          m,
          n,
          k,
          op_a,
          op_b,
          lda,
          ldb,
          ldc,
          beta_value,
          stream)) {
    return true;
  }
  const float alpha = 1.0f;
  const float beta = beta_value;
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  const cublasStatus_t status = cublasGemmEx(
      handle,
      op_a,
      op_b,
      m,
      n,
      k,
      &alpha,
      a_bf16,
      CUDA_R_16BF,
      lda,
      b_bf16,
      CUDA_R_16BF,
      ldb,
      &beta,
      c,
      CUDA_R_32F,
      ldc,
      trainer_linear_bf16_gemm_ex_compute_type(m, n, k, op_a, op_b),
      trainer_linear_bf16_gemm_ex_algo(
          m, n, k, op_a, op_b, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  if (status == CUBLAS_STATUS_SUCCESS) {
    const std::int64_t elapsed_us = finish_linear_shape_timing(&timing);
    g_linear_bf16_gemm_count.fetch_add(1, std::memory_order_relaxed);
    record_linear_shape_stat(4, m, n, k, op_a, op_b, elapsed_us);
    return true;
  }
  discard_linear_shape_timing(&timing);
  return false;
}

bool trainer_linear_tk_gemm_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_NATIVE_LINEAR_TK_GEMM");
    if (value == nullptr) {
      value = std::getenv("NFN_TILE_CUDA_LINEAR_TK_GEMM");
    }
    if (value == nullptr) {
      return true;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool trainer_linear_tk_float_out_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_NATIVE_LINEAR_TK_FLOAT_OUT");
    if (value == nullptr) {
      value = std::getenv("NFN_TILE_CUDA_LINEAR_TK_FLOAT_OUT");
    }
    if (value == nullptr) {
      return false;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool trainer_linear_tk_dinput_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_NATIVE_LINEAR_TK_DINPUT");
    if (value == nullptr) {
      value = std::getenv("NFN_TILE_CUDA_LINEAR_TK_DINPUT");
    }
    if (value == nullptr) {
      return false;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool trainer_linear_tk_dinput_default_block_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_NATIVE_LINEAR_TK_DINPUT_DEFAULT_BLOCK");
    if (value == nullptr) {
      value = std::getenv("NFN_TILE_CUDA_LINEAR_TK_DINPUT_DEFAULT_BLOCK");
    }
    if (value == nullptr) {
      return false;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool trainer_linear_tk_dinput_shape_enabled(
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b) {
  static const LinearShapeStat enabled_shape = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_LINEAR_TK_DINPUT_ENABLE_SHAPE");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_TK_DINPUT_ENABLE_SHAPE");
    }
    LinearShapeStat shape{};
    parse_linear_shape_token(value, &shape);
    return shape;
  }();
  return linear_shape_matches(enabled_shape, m, n, k, op_a, op_b);
}

bool trainer_linear_tk_dinput_default_shape_enabled(
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b) {
  if (op_a != CUBLAS_OP_N || op_b != CUBLAS_OP_N) {
    return false;
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    return false;
  }
  if (n % 128 != 0 || m % 128 != 0 || k % 64 != 0) {
    return false;
  }
  return m <= 4096 && k <= 4096;
}

bool trainer_linear_tk_dinput_shape_disabled(
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b) {
  static const LinearShapeStat disabled_shape = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_LINEAR_TK_DINPUT_DISABLE_SHAPE");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_TK_DINPUT_DISABLE_SHAPE");
    }
    LinearShapeStat shape{};
    parse_linear_shape_token(value, &shape);
    return shape;
  }();
  return linear_shape_matches(disabled_shape, m, n, k, op_a, op_b);
}

bool trainer_linear_tk_dweight_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_NATIVE_LINEAR_TK_DWEIGHT");
    if (value == nullptr) {
      value = std::getenv("NFN_TILE_CUDA_LINEAR_TK_DWEIGHT");
    }
    if (value == nullptr) {
      return false;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool trainer_linear_tk_dweight_shape_enabled(
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b) {
  static const LinearShapeStat enabled_shape = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_LINEAR_TK_DWEIGHT_ENABLE_SHAPE");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_TK_DWEIGHT_ENABLE_SHAPE");
    }
    LinearShapeStat shape{};
    parse_linear_shape_token(value, &shape);
    return shape;
  }();
  return linear_shape_matches(enabled_shape, m, n, k, op_a, op_b);
}

bool trainer_linear_tk_dweight_shape_disabled(
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b) {
  static const LinearShapeStat disabled_shape = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_LINEAR_TK_DWEIGHT_DISABLE_SHAPE");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_LINEAR_TK_DWEIGHT_DISABLE_SHAPE");
    }
    LinearShapeStat shape{};
    parse_linear_shape_token(value, &shape);
    return shape;
  }();
  return linear_shape_matches(disabled_shape, m, n, k, op_a, op_b);
}

bool tk_linear_gemm_bf16_forward_to_bf16_bits(
    const __nv_bfloat16* weight_bf16,
    const __nv_bfloat16* x_bf16,
    const __nv_bfloat16* bias_bf16,
    std::uint16_t* out_bf16_bits,
    int rows,
    int input_dim,
    int output_dim,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (!trainer_linear_tk_gemm_enabled()) {
    return false;
  }
  if (trainer_linear_tk_forward_shape_disabled(
          output_dim, rows, input_dim, op_a, op_b)) {
    return false;
  }
  if (op_a != CUBLAS_OP_T || op_b != CUBLAS_OP_N) {
    return false;
  }
  if (rows <= 0 || input_dim <= 0 || output_dim <= 0 ||
      rows % 128 != 0 || input_dim % 64 != 0 || output_dim % 128 != 0) {
    return false;
  }
  auto* out = reinterpret_cast<floatX*>(out_bf16_bits);
  const auto* inp = reinterpret_cast<const floatX*>(x_bf16);
  const auto* weight = reinterpret_cast<const floatX*>(weight_bf16);
  const auto* bias = reinterpret_cast<const floatX*>(bias_bf16);
  ensure_llmk_sm120_cublaslt_initialized();
  const auto host_start = std::chrono::steady_clock::now();
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  ::matmul_forward(out, inp, weight, bias, 1, rows, input_dim, output_dim, stream);
  const std::int64_t elapsed_us = finish_linear_shape_timing_with_host_fallback(&timing, host_start);
  g_linear_tk_gemm_count.fetch_add(1, std::memory_order_relaxed);
  g_linear_bf16_gemm_count.fetch_add(1, std::memory_order_relaxed);
  record_linear_shape_stat(2, output_dim, rows, input_dim, op_a, op_b, elapsed_us);
  return true;
#else
  (void)weight_bf16;
  (void)x_bf16;
  (void)bias_bf16;
  (void)out_bf16_bits;
  (void)rows;
  (void)input_dim;
  (void)output_dim;
  (void)op_a;
  (void)op_b;
  (void)stream;
  return false;
#endif
}

bool tk_linear_gemm_bf16_forward_to_float32(
    const __nv_bfloat16* weight_bf16,
    const __nv_bfloat16* x_bf16,
    std::uint16_t* out_bf16_bits,
    float* out,
    int rows,
    int input_dim,
    int output_dim,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (!trainer_linear_tk_gemm_enabled() || !trainer_linear_tk_float_out_enabled()) {
    return false;
  }
  if (out_bf16_bits == nullptr || out == nullptr) {
    return false;
  }
  if (!tk_linear_gemm_bf16_forward_to_bf16_bits(
          weight_bf16, x_bf16, nullptr, out_bf16_bits, rows, input_dim, output_dim, op_a, op_b, stream)) {
    return false;
  }
  const std::int64_t elements = static_cast<std::int64_t>(rows) * output_dim;
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  launch_bf16_bits_to_float32_internal(out_bf16_bits, out, elements, stream);
  const std::int64_t elapsed_us = finish_linear_shape_timing(&timing);
  g_linear_tk_float_out_gemm_count.fetch_add(1, std::memory_order_relaxed);
  record_linear_shape_stat(3, output_dim, rows, input_dim, op_a, op_b, elapsed_us);
  return true;
#else
  (void)weight_bf16;
  (void)x_bf16;
  (void)out_bf16_bits;
  (void)out;
  (void)rows;
  (void)input_dim;
  (void)output_dim;
  (void)op_a;
  (void)op_b;
  (void)stream;
  return false;
#endif
}

bool tk_linear_backward_input_bf16_bits_weight_bf16_bits_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    float* grad_x,
    int rows,
    int input_dim,
    int output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  constexpr cublasOperation_t kOpA = CUBLAS_OP_N;
  constexpr cublasOperation_t kOpB = CUBLAS_OP_N;
  const bool shape_enabled = trainer_linear_tk_dinput_shape_enabled(
      input_dim, rows, output_dim, kOpA, kOpB);
  const bool default_shape_enabled = trainer_linear_tk_dinput_default_shape_enabled(
      input_dim, rows, output_dim, kOpA, kOpB);
  const bool default_block_enabled =
      default_shape_enabled && trainer_linear_tk_dinput_default_block_enabled();
  if (!trainer_linear_tk_gemm_enabled() ||
      trainer_linear_tk_dinput_shape_disabled(input_dim, rows, output_dim, kOpA, kOpB) ||
      (!trainer_linear_tk_dinput_enabled() && !shape_enabled && !default_block_enabled)) {
    return false;
  }
  if (grad_out_bf16_bits == nullptr || weight_bf16_bits == nullptr || grad_x == nullptr) {
    return false;
  }
  if (rows <= 0 || input_dim <= 0 || output_dim <= 0 ||
      rows % 128 != 0 || input_dim % 128 != 0 || output_dim % 64 != 0) {
    return false;
  }
  const std::int64_t elements = static_cast<std::int64_t>(rows) * input_dim;
  TrainerLinearBf16Workspace* workspace = ensure_trainer_linear_bf16_workspace(elements, 1);
  if (workspace == nullptr || workspace->a == nullptr) {
    return false;
  }
  ensure_llmk_sm120_cublaslt_initialized();
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  ::matmul_dispatch_tk_ab(
      reinterpret_cast<floatX*>(workspace->a),
      reinterpret_cast<const floatX*>(grad_out_bf16_bits),
      reinterpret_cast<const floatX*>(weight_bf16_bits),
      rows,
      input_dim,
      output_dim,
      stream,
      nullptr,
      false);
  launch_bf16_bits_to_float32_internal(
      reinterpret_cast<const std::uint16_t*>(workspace->a),
      grad_x,
      elements,
      stream);
  const std::int64_t elapsed_us = finish_linear_shape_timing(&timing);
  g_linear_tk_gemm_count.fetch_add(1, std::memory_order_relaxed);
  g_linear_bf16_gemm_count.fetch_add(1, std::memory_order_relaxed);
  record_linear_shape_stat(2, input_dim, rows, output_dim, kOpA, kOpB, elapsed_us);
  return true;
#else
  (void)grad_out_bf16_bits;
  (void)weight_bf16_bits;
  (void)grad_x;
  (void)rows;
  (void)input_dim;
  (void)output_dim;
  (void)stream;
  return false;
#endif
}

bool tk_linear_gemm_bf16_forward_gelu_to_bf16_bits(
    const float* x,
    const float* weight,
    const float* bias,
    std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* gelu_bf16_bits,
    std::int64_t x_elements,
    std::int64_t weight_elements,
    int rows,
    int input_dim,
    int output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (!trainer_linear_tk_gemm_enabled()) {
    return false;
  }
  if (trainer_linear_tk_forward_shape_disabled(
          output_dim, rows, input_dim, CUBLAS_OP_T, CUBLAS_OP_N)) {
    return false;
  }
  if (x == nullptr || weight == nullptr || bias == nullptr ||
      pre_gelu_bf16_bits == nullptr || gelu_bf16_bits == nullptr) {
    return false;
  }
  if (rows <= 0 || input_dim <= 0 || output_dim <= 0 ||
      rows % 128 != 0 || input_dim % 64 != 0 || output_dim % 128 != 0) {
    return false;
  }
  if (!matmul_forward_gelu_supported(1, rows, input_dim, output_dim)) {
    return false;
  }
  TrainerLinearBf16Workspace* workspace =
      ensure_trainer_linear_bf16_workspace(x_elements, weight_elements);
  if (workspace == nullptr) {
    return false;
  }
  __nv_bfloat16* x_bf16 = trainer_linear_bf16_b_operand(workspace, x, x_elements, false, stream);
  if (x_bf16 == nullptr) {
    return false;
  }
  __nv_bfloat16* weight_bf16 =
      trainer_linear_bf16_a_operand(workspace, weight, weight_elements, true, stream);
  if (weight_bf16 == nullptr) {
    return false;
  }
  bool bias_cache_hit = false;
  TrainerLinearBf16Workspace::CacheEntry* bias_entry =
      trainer_linear_bf16_cache_entry_for(workspace, bias, output_dim, &bias_cache_hit);
  if (bias_entry == nullptr || bias_entry->data == nullptr) {
    return false;
  }
  if (!bias_cache_hit) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((output_dim + threads - 1) / threads);
    f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(bias, bias_entry->data, output_dim);
    g_linear_bf16_a_pack_count.fetch_add(1, std::memory_order_relaxed);
  }
  auto* out = reinterpret_cast<floatX*>(gelu_bf16_bits);
  auto* pre_gelu = reinterpret_cast<floatX*>(pre_gelu_bf16_bits);
  ensure_llmk_sm120_cublaslt_initialized();
  const auto host_start = std::chrono::steady_clock::now();
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  ::matmul_forward_gelu(
      out,
      pre_gelu,
      reinterpret_cast<const floatX*>(x_bf16),
      reinterpret_cast<const floatX*>(weight_bf16),
      reinterpret_cast<const floatX*>(bias_entry->data),
      1,
      rows,
      input_dim,
      output_dim,
      stream);
  const std::int64_t elapsed_us = finish_linear_shape_timing_with_host_fallback(&timing, host_start);
  g_linear_tk_gemm_count.fetch_add(1, std::memory_order_relaxed);
  g_linear_bf16_gemm_count.fetch_add(1, std::memory_order_relaxed);
  record_linear_shape_stat(2, output_dim, rows, input_dim, CUBLAS_OP_T, CUBLAS_OP_N, elapsed_us);
  return true;
#else
  (void)x;
  (void)weight;
  (void)bias;
  (void)pre_gelu_bf16_bits;
  (void)gelu_bf16_bits;
  (void)x_elements;
  (void)weight_elements;
  (void)rows;
  (void)input_dim;
  (void)output_dim;
  (void)stream;
  return false;
#endif
}

bool tk_linear_gemm_bf16_forward_gelu_weight_bf16_to_bf16_bits(
    const float* x,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* gelu_bf16_bits,
    std::int64_t x_elements,
    int rows,
    int input_dim,
    int output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (!trainer_linear_tk_gemm_enabled()) {
    return false;
  }
  if (trainer_linear_tk_forward_shape_disabled(
          output_dim, rows, input_dim, CUBLAS_OP_T, CUBLAS_OP_N)) {
    return false;
  }
  if (x == nullptr || weight_bf16_bits == nullptr || bias == nullptr ||
      pre_gelu_bf16_bits == nullptr || gelu_bf16_bits == nullptr) {
    return false;
  }
  if (rows <= 0 || input_dim <= 0 || output_dim <= 0 ||
      rows % 128 != 0 || input_dim % 64 != 0 || output_dim % 128 != 0) {
    return false;
  }
  if (!matmul_forward_gelu_supported(1, rows, input_dim, output_dim)) {
    return false;
  }
  TrainerLinearBf16Workspace* workspace =
      ensure_trainer_linear_bf16_workspace(x_elements, output_dim);
  if (workspace == nullptr) {
    return false;
  }
  __nv_bfloat16* x_bf16 = trainer_linear_bf16_b_operand(workspace, x, x_elements, false, stream);
  if (x_bf16 == nullptr) {
    return false;
  }
  bool bias_cache_hit = false;
  TrainerLinearBf16Workspace::CacheEntry* bias_entry =
      trainer_linear_bf16_cache_entry_for(workspace, bias, output_dim, &bias_cache_hit);
  if (bias_entry == nullptr || bias_entry->data == nullptr) {
    return false;
  }
  if (!bias_cache_hit) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((output_dim + threads - 1) / threads);
    f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(bias, bias_entry->data, output_dim);
    g_linear_bf16_a_pack_count.fetch_add(1, std::memory_order_relaxed);
  }
  auto* out = reinterpret_cast<floatX*>(gelu_bf16_bits);
  auto* pre_gelu = reinterpret_cast<floatX*>(pre_gelu_bf16_bits);
  ensure_llmk_sm120_cublaslt_initialized();
  const auto host_start = std::chrono::steady_clock::now();
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  ::matmul_forward_gelu(
      out,
      pre_gelu,
      reinterpret_cast<const floatX*>(x_bf16),
      reinterpret_cast<const floatX*>(weight_bf16_bits),
      reinterpret_cast<const floatX*>(bias_entry->data),
      1,
      rows,
      input_dim,
      output_dim,
      stream);
  const std::int64_t elapsed_us = finish_linear_shape_timing_with_host_fallback(&timing, host_start);
  g_linear_tk_gemm_count.fetch_add(1, std::memory_order_relaxed);
  g_linear_bf16_gemm_count.fetch_add(1, std::memory_order_relaxed);
  record_linear_shape_stat(2, output_dim, rows, input_dim, CUBLAS_OP_T, CUBLAS_OP_N, elapsed_us);
  return true;
#else
  (void)x;
  (void)weight_bf16_bits;
  (void)bias;
  (void)pre_gelu_bf16_bits;
  (void)gelu_bf16_bits;
  (void)x_elements;
  (void)rows;
  (void)input_dim;
  (void)output_dim;
  (void)stream;
  return false;
#endif
}

bool tk_linear_backward_input_dgelu_bf16_bits_float32(
    const float* grad_out,
    const float* weight,
    const std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    float* grad_x,
    std::int64_t grad_out_elements,
    std::int64_t weight_elements,
    int rows,
    int input_dim,
    int output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (!trainer_linear_tk_gemm_enabled()) {
    return false;
  }
  if (grad_out == nullptr || weight == nullptr || pre_gelu_bf16_bits == nullptr ||
      grad_x_bf16_bits == nullptr || grad_x == nullptr) {
    return false;
  }
  if (rows <= 0 || input_dim <= 0 || output_dim <= 0 ||
      rows % 128 != 0 || input_dim % 128 != 0 || output_dim % 64 != 0) {
    return false;
  }
  TrainerLinearBf16Workspace* workspace =
      ensure_trainer_linear_bf16_workspace(grad_out_elements, weight_elements);
  if (workspace == nullptr) {
    return false;
  }
  __nv_bfloat16* grad_out_bf16 =
      trainer_linear_bf16_a_operand(workspace, grad_out, grad_out_elements, false, stream);
  if (grad_out_bf16 == nullptr) {
    return false;
  }
  __nv_bfloat16* weight_bf16 =
      trainer_linear_bf16_b_operand(workspace, weight, weight_elements, true, stream);
  if (weight_bf16 == nullptr) {
    return false;
  }
  ensure_llmk_sm120_cublaslt_initialized();
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  ::matmul_dispatch_tk_ab(
      reinterpret_cast<floatX*>(grad_x_bf16_bits),
      reinterpret_cast<const floatX*>(grad_out_bf16),
      reinterpret_cast<const floatX*>(weight_bf16),
      rows,
      input_dim,
      output_dim,
      stream,
      reinterpret_cast<const floatX*>(pre_gelu_bf16_bits),
      true);
  const std::int64_t elements = static_cast<std::int64_t>(rows) * input_dim;
  launch_bf16_bits_to_float32_internal(grad_x_bf16_bits, grad_x, elements, stream);
  const std::int64_t elapsed_us = finish_linear_shape_timing(&timing);
  g_linear_tk_gemm_count.fetch_add(1, std::memory_order_relaxed);
  g_linear_tk_dgelu_dinput_gemm_count.fetch_add(1, std::memory_order_relaxed);
  g_linear_bf16_gemm_count.fetch_add(1, std::memory_order_relaxed);
  record_linear_shape_stat(2, input_dim, rows, output_dim, CUBLAS_OP_N, CUBLAS_OP_N, elapsed_us);
  return true;
#else
  (void)grad_out;
  (void)weight;
  (void)pre_gelu_bf16_bits;
  (void)grad_x_bf16_bits;
  (void)grad_x;
  (void)grad_out_elements;
  (void)weight_elements;
  (void)rows;
  (void)input_dim;
  (void)output_dim;
  (void)stream;
  return false;
#endif
}

bool tk_linear_backward_input_dgelu_weight_bf16_bits_float32(
    const float* grad_out,
    const std::uint16_t* weight_bf16_bits,
    const std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    float* grad_x,
    std::int64_t grad_out_elements,
    int rows,
    int input_dim,
    int output_dim,
    bool write_float_grad,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (!trainer_linear_tk_gemm_enabled()) {
    return false;
  }
  if (grad_out == nullptr || weight_bf16_bits == nullptr || pre_gelu_bf16_bits == nullptr ||
      grad_x_bf16_bits == nullptr || (write_float_grad && grad_x == nullptr)) {
    return false;
  }
  if (rows <= 0 || input_dim <= 0 || output_dim <= 0 ||
      rows % 128 != 0 || input_dim % 128 != 0 || output_dim % 64 != 0) {
    return false;
  }
  TrainerLinearBf16Workspace* workspace =
      ensure_trainer_linear_bf16_workspace(grad_out_elements, 1);
  if (workspace == nullptr) {
    return false;
  }
  __nv_bfloat16* grad_out_bf16 =
      trainer_linear_bf16_a_operand(workspace, grad_out, grad_out_elements, false, stream);
  if (grad_out_bf16 == nullptr) {
    return false;
  }
  ensure_llmk_sm120_cublaslt_initialized();
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  ::matmul_dispatch_tk_ab(
      reinterpret_cast<floatX*>(grad_x_bf16_bits),
      reinterpret_cast<const floatX*>(grad_out_bf16),
      reinterpret_cast<const floatX*>(weight_bf16_bits),
      rows,
      input_dim,
      output_dim,
      stream,
      reinterpret_cast<const floatX*>(pre_gelu_bf16_bits),
      true);
  if (write_float_grad) {
    const std::int64_t elements = static_cast<std::int64_t>(rows) * input_dim;
    launch_bf16_bits_to_float32_internal(grad_x_bf16_bits, grad_x, elements, stream);
  }
  const std::int64_t elapsed_us = finish_linear_shape_timing(&timing);
  g_linear_tk_gemm_count.fetch_add(1, std::memory_order_relaxed);
  g_linear_tk_dgelu_dinput_gemm_count.fetch_add(1, std::memory_order_relaxed);
  g_linear_bf16_gemm_count.fetch_add(1, std::memory_order_relaxed);
  record_linear_shape_stat(2, input_dim, rows, output_dim, CUBLAS_OP_N, CUBLAS_OP_N, elapsed_us);
  return true;
#else
  (void)grad_out;
  (void)weight_bf16_bits;
  (void)pre_gelu_bf16_bits;
  (void)grad_x_bf16_bits;
  (void)grad_x;
  (void)grad_out_elements;
  (void)rows;
  (void)input_dim;
  (void)output_dim;
  (void)stream;
  return false;
#endif
}

bool tk_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    const std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    int rows,
    int input_dim,
    int output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (!trainer_linear_tk_gemm_enabled()) {
    return false;
  }
  if (grad_out_bf16_bits == nullptr || weight_bf16_bits == nullptr ||
      pre_gelu_bf16_bits == nullptr || grad_x_bf16_bits == nullptr) {
    return false;
  }
  if (rows <= 0 || input_dim <= 0 || output_dim <= 0 ||
      rows % 128 != 0 || input_dim % 128 != 0 || output_dim % 64 != 0) {
    return false;
  }
  ensure_llmk_sm120_cublaslt_initialized();
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  ::matmul_dispatch_tk_ab(
      reinterpret_cast<floatX*>(grad_x_bf16_bits),
      reinterpret_cast<const floatX*>(grad_out_bf16_bits),
      reinterpret_cast<const floatX*>(weight_bf16_bits),
      rows,
      input_dim,
      output_dim,
      stream,
      reinterpret_cast<const floatX*>(pre_gelu_bf16_bits),
      true);
  const std::int64_t elapsed_us = finish_linear_shape_timing(&timing);
  g_linear_tk_gemm_count.fetch_add(1, std::memory_order_relaxed);
  g_linear_tk_dgelu_dinput_gemm_count.fetch_add(1, std::memory_order_relaxed);
  g_linear_bf16_gemm_count.fetch_add(1, std::memory_order_relaxed);
  record_linear_shape_stat(2, input_dim, rows, output_dim, CUBLAS_OP_N, CUBLAS_OP_N, elapsed_us);
  return true;
#else
  (void)grad_out_bf16_bits;
  (void)weight_bf16_bits;
  (void)pre_gelu_bf16_bits;
  (void)grad_x_bf16_bits;
  (void)rows;
  (void)input_dim;
  (void)output_dim;
  (void)stream;
  return false;
#endif
}

bool cublas_linear_gemm_ex_bf16_float32_to_bf16_bits(
    const float* a,
    const float* b,
    std::uint16_t* c_bf16_bits,
    std::int64_t a_elements,
    std::int64_t b_elements,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int lda,
    int ldb,
    int ldc,
    bool cache_a_operand,
    bool cache_b_operand,
    bool force_bf16,
    cudaStream_t stream) {
  if (!force_bf16 && !trainer_linear_bf16_bridge_enabled()) {
    return false;
  }
  TrainerLinearBf16Workspace* workspace = ensure_trainer_linear_bf16_workspace(a_elements, b_elements);
  if (workspace == nullptr) {
    return false;
  }
  cublasHandle_t handle = trainer_linear_cublas_handle(stream);
  if (handle == nullptr) {
    return false;
  }
  __nv_bfloat16* a_bf16 = trainer_linear_bf16_a_operand(workspace, a, a_elements, cache_a_operand, stream);
  if (a_bf16 == nullptr) {
    return false;
  }
  __nv_bfloat16* b_bf16 = trainer_linear_bf16_b_operand(workspace, b, b_elements, cache_b_operand, stream);
  if (b_bf16 == nullptr) {
    return false;
  }
  if (tk_linear_gemm_bf16_forward_to_bf16_bits(
          a_bf16,
          b_bf16,
          nullptr,
          c_bf16_bits,
          n,
          k,
          m,
          op_a,
          op_b,
          stream)) {
    return true;
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto* c_bf16 = reinterpret_cast<__nv_bfloat16*>(c_bf16_bits);
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  const cublasStatus_t status = cublasGemmEx(
      handle,
      op_a,
      op_b,
      m,
      n,
      k,
      &alpha,
      a_bf16,
      CUDA_R_16BF,
      lda,
      b_bf16,
      CUDA_R_16BF,
      ldb,
      &beta,
      c_bf16,
      CUDA_R_16BF,
      ldc,
      trainer_linear_bf16_gemm_ex_compute_type(m, n, k, op_a, op_b),
      trainer_linear_bf16_gemm_ex_algo(
          m, n, k, op_a, op_b, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  if (status == CUBLAS_STATUS_SUCCESS) {
    const std::int64_t elapsed_us = finish_linear_shape_timing(&timing);
    g_linear_bf16_gemm_count.fetch_add(1, std::memory_order_relaxed);
    record_linear_shape_stat(4, m, n, k, op_a, op_b, elapsed_us);
    return true;
  }
  discard_linear_shape_timing(&timing);
  return false;
}

bool cublas_linear_gemm_ex_bf16_bits_a_float32_to_bf16_bits(
    const std::uint16_t* a_bf16_bits,
    const float* b,
    const float* bias,
    std::uint16_t* c_bf16_bits,
    std::int64_t b_elements,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int lda,
    int ldb,
    int ldc,
    bool force_bf16,
    bool has_bias,
    cudaStream_t stream) {
  if (!force_bf16 && !trainer_linear_bf16_bridge_enabled()) {
    return false;
  }
  const std::int64_t bias_elements = has_bias && bias != nullptr ? static_cast<std::int64_t>(m) : 1;
  TrainerLinearBf16Workspace* workspace = ensure_trainer_linear_bf16_workspace(bias_elements, b_elements);
  if (workspace == nullptr) {
    return false;
  }
  cublasHandle_t handle = trainer_linear_cublas_handle(stream);
  if (handle == nullptr) {
    return false;
  }
  constexpr int threads = 256;
  const int b_blocks = static_cast<int>((b_elements + threads - 1) / threads);
  f32_to_bf16_kernel<<<b_blocks, threads, 0, stream>>>(b, workspace->b, b_elements);
  __nv_bfloat16* bias_bf16 = nullptr;
  if (has_bias && bias != nullptr) {
    const int bias_blocks = static_cast<int>((bias_elements + threads - 1) / threads);
    f32_to_bf16_kernel<<<bias_blocks, threads, 0, stream>>>(bias, workspace->a, bias_elements);
    bias_bf16 = workspace->a;
  }
  const auto* a_bf16 = reinterpret_cast<const __nv_bfloat16*>(a_bf16_bits);
  if (tk_linear_gemm_bf16_forward_to_bf16_bits(
          a_bf16,
          workspace->b,
          bias_bf16,
          c_bf16_bits,
          n,
          k,
          m,
          op_a,
          op_b,
          stream)) {
    return true;
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto* c_bf16 = reinterpret_cast<__nv_bfloat16*>(c_bf16_bits);
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  const cublasStatus_t status = cublasGemmEx(
      handle,
      op_a,
      op_b,
      m,
      n,
      k,
      &alpha,
      a_bf16,
      CUDA_R_16BF,
      lda,
      workspace->b,
      CUDA_R_16BF,
      ldb,
      &beta,
      c_bf16,
      CUDA_R_16BF,
      ldc,
      trainer_linear_bf16_gemm_ex_compute_type(m, n, k, op_a, op_b),
      trainer_linear_bf16_gemm_ex_algo(
          m, n, k, op_a, op_b, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  if (status == CUBLAS_STATUS_SUCCESS) {
    const std::int64_t elapsed_us = finish_linear_shape_timing(&timing);
    g_linear_bf16_gemm_count.fetch_add(1, std::memory_order_relaxed);
    record_linear_shape_stat(4, m, n, k, op_a, op_b, elapsed_us);
    if (has_bias && bias != nullptr) {
      const std::int64_t elements = static_cast<std::int64_t>(n) * m;
      if (bf16_bits_add_bias_tile_enabled()) {
        const int blocks = static_cast<int>((elements + kTileSize - 1) / kTileSize);
        bf16_bits_add_bias_inplace_tile_float32_kernel<<<blocks, 1, 0, stream>>>(
            c_bf16_bits, bias, elements, m);
      } else {
        const int blocks = static_cast<int>((elements + threads - 1) / threads);
        bf16_bits_add_bias_inplace_kernel<<<blocks, threads, 0, stream>>>(c_bf16_bits, bias, elements, m);
      }
    }
    return true;
  }
  discard_linear_shape_timing(&timing);
  return false;
}

bool cublas_linear_gemm_ex_float32_a_bf16_bits_b_to_bf16_bits(
    const float* a,
    const std::uint16_t* b_bf16_bits,
    const float* bias,
    std::uint16_t* c_bf16_bits,
    std::int64_t a_elements,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int lda,
    int ldb,
    int ldc,
    bool cache_a_operand,
    bool force_bf16,
    bool has_bias,
    cudaStream_t stream) {
  if (!force_bf16 && !trainer_linear_bf16_bridge_enabled()) {
    return false;
  }
  const std::int64_t bias_elements = has_bias && bias != nullptr ? static_cast<std::int64_t>(m) : 1;
  TrainerLinearBf16Workspace* workspace = ensure_trainer_linear_bf16_workspace(a_elements, bias_elements);
  if (workspace == nullptr) {
    return false;
  }
  cublasHandle_t handle = trainer_linear_cublas_handle(stream);
  if (handle == nullptr) {
    return false;
  }
  __nv_bfloat16* a_bf16 = trainer_linear_bf16_a_operand(workspace, a, a_elements, cache_a_operand, stream);
  if (a_bf16 == nullptr) {
    return false;
  }
  constexpr int threads = 256;
  __nv_bfloat16* bias_bf16 = nullptr;
  if (has_bias && bias != nullptr) {
    const int bias_blocks = static_cast<int>((bias_elements + threads - 1) / threads);
    f32_to_bf16_kernel<<<bias_blocks, threads, 0, stream>>>(bias, workspace->b, bias_elements);
    bias_bf16 = workspace->b;
  }
  const auto* b_bf16 = reinterpret_cast<const __nv_bfloat16*>(b_bf16_bits);
  if (tk_linear_gemm_bf16_forward_to_bf16_bits(
          a_bf16,
          b_bf16,
          bias_bf16,
          c_bf16_bits,
          n,
          k,
          m,
          op_a,
          op_b,
          stream)) {
    return true;
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto* c_bf16 = reinterpret_cast<__nv_bfloat16*>(c_bf16_bits);
  if (!has_bias &&
      trainer_linear_bf16_output_cublaslt_enabled() &&
      trainer_linear_bf16_cublaslt_enabled() &&
      trainer_linear_bf16_cublaslt_shape_supported(m, n, k, op_a, op_b) &&
      cublaslt_linear_matmul(
          a_bf16,
          b_bf16,
          c_bf16,
          CUDA_R_16BF,
          CUDA_R_16BF,
          CUDA_R_16BF,
          CUBLAS_COMPUTE_32F_FAST_16BF,
          m,
          n,
          k,
          op_a,
          op_b,
          lda,
          ldb,
          ldc,
          0.0f,
          stream)) {
    return true;
  }
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  const cublasStatus_t status = cublasGemmEx(
      handle,
      op_a,
      op_b,
      m,
      n,
      k,
      &alpha,
      a_bf16,
      CUDA_R_16BF,
      lda,
      b_bf16,
      CUDA_R_16BF,
      ldb,
      &beta,
      c_bf16,
      CUDA_R_16BF,
      ldc,
      trainer_linear_bf16_gemm_ex_compute_type(m, n, k, op_a, op_b),
      trainer_linear_bf16_gemm_ex_algo(
          m, n, k, op_a, op_b, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  if (status == CUBLAS_STATUS_SUCCESS) {
    const std::int64_t elapsed_us = finish_linear_shape_timing(&timing);
    g_linear_bf16_gemm_count.fetch_add(1, std::memory_order_relaxed);
    record_linear_shape_stat(4, m, n, k, op_a, op_b, elapsed_us);
    if (has_bias && bias != nullptr) {
      const std::int64_t elements = static_cast<std::int64_t>(n) * m;
      if (bf16_bits_add_bias_tile_enabled()) {
        const int blocks = static_cast<int>((elements + kTileSize - 1) / kTileSize);
        bf16_bits_add_bias_inplace_tile_float32_kernel<<<blocks, 1, 0, stream>>>(
            c_bf16_bits, bias, elements, m);
      } else {
        const int blocks = static_cast<int>((elements + threads - 1) / threads);
        bf16_bits_add_bias_inplace_kernel<<<blocks, threads, 0, stream>>>(c_bf16_bits, bias, elements, m);
      }
    }
    return true;
  }
  discard_linear_shape_timing(&timing);
  return false;
}

bool cublas_linear_gemm_ex_bf16_bits_ab_to_bf16_bits(
    const std::uint16_t* a_bf16_bits,
    const std::uint16_t* b_bf16_bits,
    const float* bias,
    std::uint16_t* c_bf16_bits,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int lda,
    int ldb,
    int ldc,
    bool has_bias,
    cudaStream_t stream) {
  cublasHandle_t handle = trainer_linear_cublas_handle(stream);
  if (handle == nullptr) {
    return false;
  }
  constexpr int threads = 256;
  const std::int64_t bias_elements = has_bias && bias != nullptr ? static_cast<std::int64_t>(m) : 1;
  TrainerLinearBf16Workspace* workspace = ensure_trainer_linear_bf16_workspace(bias_elements, 1);
  if (workspace == nullptr) {
    return false;
  }
  __nv_bfloat16* bias_bf16 = nullptr;
  if (has_bias && bias != nullptr) {
    const int bias_blocks = static_cast<int>((bias_elements + threads - 1) / threads);
    f32_to_bf16_kernel<<<bias_blocks, threads, 0, stream>>>(bias, workspace->a, bias_elements);
    bias_bf16 = workspace->a;
  }
  const auto* a_bf16 = reinterpret_cast<const __nv_bfloat16*>(a_bf16_bits);
  const auto* b_bf16 = reinterpret_cast<const __nv_bfloat16*>(b_bf16_bits);
  if (tk_linear_gemm_bf16_forward_to_bf16_bits(
          a_bf16,
          b_bf16,
          bias_bf16,
          c_bf16_bits,
          n,
          k,
          m,
          op_a,
          op_b,
          stream)) {
    return true;
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto* c_bf16 = reinterpret_cast<__nv_bfloat16*>(c_bf16_bits);
  if (!has_bias &&
      trainer_linear_bf16_output_cublaslt_enabled() &&
      trainer_linear_bf16_cublaslt_enabled() &&
      trainer_linear_bf16_cublaslt_shape_supported(m, n, k, op_a, op_b) &&
      cublaslt_linear_matmul(
          a_bf16,
          b_bf16,
          c_bf16,
          CUDA_R_16BF,
          CUDA_R_16BF,
          CUDA_R_16BF,
          CUBLAS_COMPUTE_32F_FAST_16BF,
          m,
          n,
          k,
          op_a,
          op_b,
          lda,
          ldb,
          ldc,
          0.0f,
          stream)) {
    return true;
  }
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  const cublasStatus_t status = cublasGemmEx(
      handle,
      op_a,
      op_b,
      m,
      n,
      k,
      &alpha,
      a_bf16,
      CUDA_R_16BF,
      lda,
      b_bf16,
      CUDA_R_16BF,
      ldb,
      &beta,
      c_bf16,
      CUDA_R_16BF,
      ldc,
      trainer_linear_bf16_gemm_ex_compute_type(m, n, k, op_a, op_b),
      trainer_linear_bf16_gemm_ex_algo(
          m, n, k, op_a, op_b, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  if (status == CUBLAS_STATUS_SUCCESS) {
    const std::int64_t elapsed_us = finish_linear_shape_timing(&timing);
    g_linear_bf16_gemm_count.fetch_add(1, std::memory_order_relaxed);
    record_linear_shape_stat(4, m, n, k, op_a, op_b, elapsed_us);
    if (has_bias && bias != nullptr) {
      const std::int64_t elements = static_cast<std::int64_t>(n) * m;
      if (bf16_bits_add_bias_tile_enabled()) {
        const int blocks = static_cast<int>((elements + kTileSize - 1) / kTileSize);
        bf16_bits_add_bias_inplace_tile_float32_kernel<<<blocks, 1, 0, stream>>>(
            c_bf16_bits, bias, elements, m);
      } else {
        const int blocks = static_cast<int>((elements + threads - 1) / threads);
        bf16_bits_add_bias_inplace_kernel<<<blocks, threads, 0, stream>>>(c_bf16_bits, bias, elements, m);
      }
    }
    return true;
  }
  discard_linear_shape_timing(&timing);
  return false;
}

bool cublas_linear_gemm_ex_bf16_bits_ab_to_bf16_bits_accumulate(
    const std::uint16_t* a_bf16_bits,
    const std::uint16_t* b_bf16_bits,
    std::uint16_t* c_bf16_bits,
    int m,
    int n,
    int k,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int lda,
    int ldb,
    int ldc,
    cudaStream_t stream) {
  cublasHandle_t handle = trainer_linear_cublas_handle(stream);
  if (handle == nullptr) {
    return false;
  }
  const auto* a_bf16 = reinterpret_cast<const __nv_bfloat16*>(a_bf16_bits);
  const auto* b_bf16 = reinterpret_cast<const __nv_bfloat16*>(b_bf16_bits);
  auto* c_bf16 = reinterpret_cast<__nv_bfloat16*>(c_bf16_bits);
  const float alpha = 1.0f;
  const float beta = 1.0f;
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  const cublasStatus_t status = cublasGemmEx(
      handle,
      op_a,
      op_b,
      m,
      n,
      k,
      &alpha,
      a_bf16,
      CUDA_R_16BF,
      lda,
      b_bf16,
      CUDA_R_16BF,
      ldb,
      &beta,
      c_bf16,
      CUDA_R_16BF,
      ldc,
      trainer_linear_bf16_gemm_ex_compute_type(m, n, k, op_a, op_b),
      trainer_linear_bf16_gemm_ex_algo(
          m, n, k, op_a, op_b, CUBLAS_GEMM_DEFAULT));
  if (status == CUBLAS_STATUS_SUCCESS) {
    const std::int64_t elapsed_us = finish_linear_shape_timing(&timing);
    g_linear_bf16_gemm_count.fetch_add(1, std::memory_order_relaxed);
    record_linear_shape_stat(4, m, n, k, op_a, op_b, elapsed_us);
    return true;
  }
  discard_linear_shape_timing(&timing);
  return false;
}

bool cublas_linear_forward_float32(
    const float* x,
    const float* weight,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
  if (!fits_cublas_int(rows) || !fits_cublas_int(input_dim) || !fits_cublas_int(output_dim)) {
    return false;
  }
  cublasHandle_t handle = trainer_linear_cublas_handle(stream);
  if (handle == nullptr) {
    return false;
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int m = static_cast<int>(output_dim);
  const int n = static_cast<int>(rows);
  const int k = static_cast<int>(input_dim);
  if (cublas_linear_gemm_ex_bf16_float32(
          weight,
          x,
          out,
          output_dim * input_dim,
          rows * input_dim,
          m,
          n,
          k,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          k,
          k,
          m,
          0.0f,
          true,
          false,
          false,
          stream)) {
    return true;
  }
  if (cublaslt_linear_matmul_float32(
          weight,
          x,
          out,
          m,
          n,
          k,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          k,
          k,
          m,
          0.0f,
          stream)) {
    return true;
  }
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  const cublasStatus_t status = cublasSgemm(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      m,
      n,
      k,
      &alpha,
      weight,
      k,
      x,
      k,
      &beta,
      out,
      m);
  if (status == CUBLAS_STATUS_SUCCESS) {
    const std::int64_t elapsed_us = finish_linear_shape_timing(&timing);
    g_linear_sgemm_count.fetch_add(1, std::memory_order_relaxed);
    record_linear_shape_stat(5, m, n, k, CUBLAS_OP_T, CUBLAS_OP_N, elapsed_us);
    return true;
  }
  discard_linear_shape_timing(&timing);
  return false;
}

bool cublas_linear_backward_input_float32(
    const float* grad_out,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool force_bf16,
    cudaStream_t stream) {
  if (!fits_cublas_int(rows) || !fits_cublas_int(input_dim) || !fits_cublas_int(output_dim)) {
    return false;
  }
  cublasHandle_t handle = trainer_linear_cublas_handle(stream);
  if (handle == nullptr) {
    return false;
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int m = static_cast<int>(input_dim);
  const int n = static_cast<int>(rows);
  const int k = static_cast<int>(output_dim);
  if (cublas_linear_gemm_ex_bf16_float32(
          weight,
          grad_out,
          grad_x,
          output_dim * input_dim,
          rows * output_dim,
          m,
          n,
          k,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          m,
          k,
          m,
          0.0f,
          true,
          false,
          force_bf16,
          stream)) {
    return true;
  }
  if (cublaslt_linear_matmul_float32(
          weight,
          grad_out,
          grad_x,
          m,
          n,
          k,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          m,
          k,
          m,
          0.0f,
          stream)) {
    return true;
  }
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  const cublasStatus_t status = cublasSgemm(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      m,
      n,
      k,
      &alpha,
      weight,
      m,
      grad_out,
      k,
      &beta,
      grad_x,
      m);
  if (status == CUBLAS_STATUS_SUCCESS) {
    const std::int64_t elapsed_us = finish_linear_shape_timing(&timing);
    g_linear_sgemm_count.fetch_add(1, std::memory_order_relaxed);
    record_linear_shape_stat(5, m, n, k, CUBLAS_OP_N, CUBLAS_OP_N, elapsed_us);
    return true;
  }
  discard_linear_shape_timing(&timing);
  return false;
}

bool cublas_linear_backward_weight_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta_value,
    cudaStream_t stream) {
  if (!fits_cublas_int(rows) || !fits_cublas_int(input_dim) || !fits_cublas_int(output_dim)) {
    return false;
  }
  cublasHandle_t handle = trainer_linear_cublas_handle(stream);
  if (handle == nullptr) {
    return false;
  }
  const float alpha = 1.0f;
  const float beta = beta_value;
  const int m = static_cast<int>(input_dim);
  const int n = static_cast<int>(output_dim);
  const int k = static_cast<int>(rows);
  if (cublas_linear_gemm_ex_bf16_float32(
          x,
          grad_out,
          grad_weight,
          rows * input_dim,
          rows * output_dim,
          m,
          n,
          k,
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          m,
          n,
          m,
          beta_value,
          false,
          false,
          false,
          stream)) {
    return true;
  }
  if (cublaslt_linear_matmul_float32(
          x,
          grad_out,
          grad_weight,
          m,
          n,
          k,
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          m,
          n,
          m,
          beta_value,
          stream)) {
    return true;
  }
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  const cublasStatus_t status = cublasSgemm(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      m,
      n,
      k,
      &alpha,
      x,
      m,
      grad_out,
      n,
      &beta,
      grad_weight,
      m);
  if (status == CUBLAS_STATUS_SUCCESS) {
    const std::int64_t elapsed_us = finish_linear_shape_timing(&timing);
    g_linear_sgemm_count.fetch_add(1, std::memory_order_relaxed);
    record_linear_shape_stat(5, m, n, k, CUBLAS_OP_N, CUBLAS_OP_T, elapsed_us);
    return true;
  }
  discard_linear_shape_timing(&timing);
  return false;
}

bool ensure_trainer_bias_ones(std::int64_t rows, cudaStream_t stream, float** ones_out) {
  static float* ones = nullptr;
  static std::int64_t capacity = 0;
  if (rows <= 0) {
    return false;
  }
  if (capacity < rows) {
    float* next = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&next), sizeof(float) * static_cast<std::size_t>(rows)) !=
        cudaSuccess) {
      return false;
    }
    const int blocks = static_cast<int>((rows + kTileSize - 1) / kTileSize);
    fill_float32_kernel<<<blocks, 1, 0, stream>>>(next, rows, 1.0f);
    if (ones != nullptr) {
      cudaFree(ones);
    }
    ones = next;
    capacity = rows;
  }
  *ones_out = ones;
  return true;
}

bool cublas_linear_backward_bias_float32(
    const float* grad_out,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t output_dim,
    float beta_value,
    cudaStream_t stream) {
  if (!fits_cublas_int(rows) || !fits_cublas_int(output_dim)) {
    return false;
  }
  float* ones = nullptr;
  if (!ensure_trainer_bias_ones(rows, stream, &ones)) {
    return false;
  }
  cublasHandle_t handle = trainer_linear_cublas_handle(stream);
  if (handle == nullptr) {
    return false;
  }
  const float alpha = 1.0f;
  const float beta = beta_value;
  const int m = static_cast<int>(output_dim);
  const int n = static_cast<int>(rows);
  const cublasStatus_t status = cublasSgemv(
      handle,
      CUBLAS_OP_N,
      m,
      n,
      &alpha,
      grad_out,
      m,
      ones,
      1,
      &beta,
      grad_bias,
      1);
  return status == CUBLAS_STATUS_SUCCESS;
}
#endif

__tile_global__ void unary_float32_kernel(const float* __restrict__ x, float* __restrict__ out, std::int64_t n, int op) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  auto x_tile = ct::partition_view{ct::tensor_span{x, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto zero = ct::full<decltype(x_tile)>(0.0f);
  auto one = ct::full<decltype(x_tile)>(1.0f);
  auto result = x_tile;

  if (op == 0) {
    result = x_tile;
  } else if (op == 1) {
    result = -x_tile;
  } else if (op == 2) {
    result = ct::select(x_tile > zero, x_tile, zero);
  } else if (op == 3) {
    result = one / (one + ct::exp(-x_tile));
  } else if (op == 4) {
    result = ct::tanh(x_tile);
  } else if (op == 5) {
    result = ct::select(x_tile >= zero, x_tile, x_tile * ct::full<decltype(x_tile)>(0.01f));
  } else if (op == 6) {
    auto sigmoid = one / (one + ct::exp(-x_tile));
    result = x_tile * sigmoid;
  } else if (op == 7) {
    result = ct::log(one + ct::exp(x_tile));
  } else if (op == 8) {
    auto neg_one = ct::full<decltype(x_tile)>(-1.0f);
    result = ct::select(x_tile > one, one, ct::select(x_tile < neg_one, neg_one, x_tile));
  } else if (op == 9) {
    result = ct::exp(-(x_tile * x_tile));
  } else if (op == 10) {
    auto floor = ct::full<decltype(x_tile)>(1.0e-7f);
    result = ct::log(ct::select(x_tile > floor, x_tile, floor));
  } else if (op == 11) {
    result = ct::select(x_tile >= zero, x_tile, x_tile * ct::full<decltype(x_tile)>(0.25f));
  } else if (op == 12) {
    auto six = ct::full<decltype(x_tile)>(6.0f);
    result = ct::select(x_tile > six, six, ct::select(x_tile < zero, zero, x_tile));
  } else if (op == 13) {
    result = ct::select(x_tile >= zero, x_tile, ct::exp(x_tile) - one);
  } else if (op == 14) {
    auto alpha = ct::full<decltype(x_tile)>(1.6732632423543772f);
    auto scale = ct::full<decltype(x_tile)>(1.0507009873554805f);
    auto inner = ct::select(x_tile >= zero, x_tile, alpha * (ct::exp(x_tile) - one));
    result = scale * inner;
  } else if (op == 15) {
    auto softplus = ct::log(one + ct::exp(x_tile));
    result = x_tile * ct::tanh(softplus);
  } else if (op == 16) {
    result = x_tile / (one + ct::abs(x_tile));
  } else if (op == 17) {
    auto raw = x_tile / ct::full<decltype(x_tile)>(6.0f) + ct::full<decltype(x_tile)>(0.5f);
    result = ct::select(raw > one, one, ct::select(raw < zero, zero, raw));
  } else if (op == 18) {
    auto raw = x_tile / ct::full<decltype(x_tile)>(6.0f) + ct::full<decltype(x_tile)>(0.5f);
    auto hard_sigmoid = ct::select(raw > one, one, ct::select(raw < zero, zero, raw));
    result = x_tile * hard_sigmoid;
  } else if (op == 19) {
    result = ct::select(x_tile >= zero, one, zero);
  } else if (op == 20) {
    auto inv_sqrt2 = ct::full<decltype(x_tile)>(0.7071067811865476f);
    auto z = x_tile * inv_sqrt2;
    auto sign = ct::select(z < zero, ct::full<decltype(z)>(-1.0f), one);
    auto abs_z = ct::abs(z);
    auto t = one / (one + ct::full<decltype(z)>(0.3275911f) * abs_z);
    auto a1 = ct::full<decltype(z)>(0.254829592f);
    auto a2 = ct::full<decltype(z)>(-0.284496736f);
    auto a3 = ct::full<decltype(z)>(1.421413741f);
    auto a4 = ct::full<decltype(z)>(-1.453152027f);
    auto a5 = ct::full<decltype(z)>(1.061405429f);
    auto poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;
    auto erf_approx = sign * (one - poly * ct::exp(-(abs_z * abs_z)));
    result = ct::full<decltype(x_tile)>(0.5f) * x_tile * (one + erf_approx);
  }

  auto out_view = ct::partition_view{ct::tensor_span{out, ct::extents{n}}, ct::shape{1024_ic}};
  out_view.store_masked(result, bx);
}

__tile_global__ void binary_float32_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out,
    std::int64_t n,
    int op) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  lhs = ct::assume_aligned(lhs, 16_ic);
  rhs = ct::assume_aligned(rhs, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  auto lhs_tile = ct::partition_view{ct::tensor_span{lhs, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto rhs_tile = ct::partition_view{ct::tensor_span{rhs, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto result = op == 0 ? lhs_tile + rhs_tile : lhs_tile * rhs_tile;
  auto out_view = ct::partition_view{ct::tensor_span{out, ct::extents{n}}, ct::shape{1024_ic}};
  out_view.store_masked(result, bx);
}

__tile_global__ void binary_pair_float32_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out0,
    float* __restrict__ out1,
    std::int64_t n,
    int op) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  lhs = ct::assume_aligned(lhs, 16_ic);
  rhs = ct::assume_aligned(rhs, 16_ic);
  out0 = ct::assume_aligned(out0, 16_ic);
  out1 = ct::assume_aligned(out1, 16_ic);

  const int bx = ct::bid().x;
  auto lhs_tile = ct::partition_view{ct::tensor_span{lhs, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto rhs_tile = ct::partition_view{ct::tensor_span{rhs, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto max_tile = ct::select(lhs_tile > rhs_tile, lhs_tile, rhs_tile);
  auto lhs_exp = ct::exp(lhs_tile - max_tile);
  auto rhs_exp = ct::exp(rhs_tile - max_tile);
  auto denom = lhs_exp + rhs_exp;
  auto result0 = lhs_exp / denom;
  auto result1 = rhs_exp / denom;
  if (op == 1) {
    auto lse = max_tile + ct::log(denom);
    result0 = lhs_tile - lse;
    result1 = rhs_tile - lse;
  }
  auto out0_view = ct::partition_view{ct::tensor_span{out0, ct::extents{n}}, ct::shape{1024_ic}};
  auto out1_view = ct::partition_view{ct::tensor_span{out1, ct::extents{n}}, ct::shape{1024_ic}};
  out0_view.store_masked(result0, bx);
  out1_view.store_masked(result1, bx);
}

__tile_global__ void scalar_unary_float32_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    std::int64_t n,
    float value,
    int op) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  auto x_tile = ct::partition_view{ct::tensor_span{x, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto scalar = ct::full<decltype(x_tile)>(value);
  auto result = op == 0 ? scalar * x_tile : scalar * ct::tanh(x_tile / scalar);
  auto out_view = ct::partition_view{ct::tensor_span{out, ct::extents{n}}, ct::shape{1024_ic}};
  out_view.store_masked(result, bx);
}

__tile_global__ void scalar_binary_float32_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out,
    std::int64_t n,
    float value,
    int op) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  lhs = ct::assume_aligned(lhs, 16_ic);
  rhs = ct::assume_aligned(rhs, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  auto lhs_tile = ct::partition_view{ct::tensor_span{lhs, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto rhs_tile = ct::partition_view{ct::tensor_span{rhs, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto scalar = ct::full<decltype(lhs_tile)>(value);
  auto result = op == 0 ? lhs_tile + scalar * rhs_tile : lhs_tile - scalar * rhs_tile;
  auto out_view = ct::partition_view{ct::tensor_span{out, ct::extents{n}}, ct::shape{1024_ic}};
  out_view.store_masked(result, bx);
}

__tile_global__ void ema_update_float32_kernel(
    float* __restrict__ target,
    const float* __restrict__ source,
    std::int64_t n,
    float decay) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  target = ct::assume_aligned(target, 16_ic);
  source = ct::assume_aligned(source, 16_ic);

  const int bx = ct::bid().x;
  auto target_tile = ct::partition_view{ct::tensor_span{target, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto source_tile = ct::partition_view{ct::tensor_span{source, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto decay_tile = ct::full<decltype(target_tile)>(decay);
  auto one_minus_decay = ct::full<decltype(target_tile)>(1.0f - decay);
  auto result = decay_tile * target_tile + one_minus_decay * source_tile;
  auto target_view = ct::partition_view{ct::tensor_span{target, ct::extents{n}}, ct::shape{1024_ic}};
  target_view.store_masked(result, bx);
}

__tile_global__ void gradient_accumulate_float32_kernel(
    float* __restrict__ buffer,
    const float* __restrict__ grad,
    std::int64_t n,
    float scale) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  buffer = ct::assume_aligned(buffer, 16_ic);
  grad = ct::assume_aligned(grad, 16_ic);

  const int bx = ct::bid().x;
  auto buffer_tile = ct::partition_view{ct::tensor_span{buffer, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto grad_tile = ct::partition_view{ct::tensor_span{grad, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto result = buffer_tile + ct::full<decltype(buffer_tile)>(scale) * grad_tile;
  auto buffer_view = ct::partition_view{ct::tensor_span{buffer, ct::extents{n}}, ct::shape{1024_ic}};
  buffer_view.store_masked(result, bx);
}

__tile_global__ void copy_float32_kernel(
    const float* __restrict__ source,
    float* __restrict__ dest,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  source = ct::assume_aligned(source, 16_ic);
  dest = ct::assume_aligned(dest, 16_ic);

  const int bx = ct::bid().x;
  auto source_tile = ct::partition_view{ct::tensor_span{source, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto dest_view = ct::partition_view{ct::tensor_span{dest, ct::extents{n}}, ct::shape{1024_ic}};
  dest_view.store_masked(source_tile, bx);
}

__device__ float deterministic_uniform_value(std::int64_t linear_idx, std::int64_t counter, std::int64_t salt);

__global__ void evo_mutate_candidates_float32_kernel(
    const float* __restrict__ base,
    float* __restrict__ candidates,
    std::int64_t elements,
    std::int64_t candidate_count,
    float mutation_scale,
    std::int64_t seed) {
  const std::int64_t total = elements * candidate_count;
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total || elements <= 0 || candidate_count <= 0) {
    return;
  }
  const std::int64_t candidate = idx / elements;
  const std::int64_t element = idx - candidate * elements;
  const float value = base[element];
  if (candidate == 0 || mutation_scale == 0.0f) {
    candidates[idx] = value;
    return;
  }
  const std::int64_t salt = 65537 + candidate * 131071;
  const float u1 = fmaxf(deterministic_uniform_value(element, seed, salt), 1.0e-7f);
  const float u2 = deterministic_uniform_value(element, seed, salt + 17);
  const float radius = sqrtf(-2.0f * logf(u1));
  const float theta = 6.2831853071795864769f * u2;
  candidates[idx] = value + mutation_scale * radius * cosf(theta);
}

__global__ void evo_select_best_loss_float32_kernel(
    const float* __restrict__ losses,
    std::int64_t candidate_count,
    std::int64_t* __restrict__ best_index,
    float* __restrict__ best_loss) {
  if (threadIdx.x != 0 || blockIdx.x != 0 || candidate_count <= 0) {
    return;
  }
  std::int64_t index = 0;
  float value = losses[0];
  for (std::int64_t i = 1; i < candidate_count; ++i) {
    const float candidate = losses[i];
    if (candidate < value) {
      value = candidate;
      index = i;
    }
  }
  *best_index = index;
  *best_loss = value;
}

__global__ void evo_adopt_candidate_float32_kernel(
    const float* __restrict__ candidates,
    const std::int64_t* __restrict__ best_index,
    float* __restrict__ target,
    std::int64_t elements,
    std::int64_t candidate_count) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elements || elements <= 0 || candidate_count <= 0) {
    return;
  }
  std::int64_t selected = *best_index;
  if (selected < 0) {
    selected = 0;
  } else if (selected >= candidate_count) {
    selected = candidate_count - 1;
  }
  target[idx] = candidates[selected * elements + idx];
}

__tile_global__ void uint16_to_int64_kernel(
    const std::uint16_t* __restrict__ source,
    std::int64_t* __restrict__ dest,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  source = ct::assume_aligned(source, 16_ic);
  dest = ct::assume_aligned(dest, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto token_tile = ct::load_masked(source + idx, mask);
  ct::store_masked(dest + idx, ct::element_cast<std::int64_t>(token_tile), mask);
}

__tile_global__ void fill_float32_kernel(
    float* __restrict__ values,
    std::int64_t n,
    float value) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  values = ct::assume_aligned(values, 16_ic);

  const int bx = ct::bid().x;
  auto view = ct::partition_view{ct::tensor_span{values, ct::extents{n}}, ct::shape{1024_ic}};
  auto tile = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(value);
  view.store_masked(tile, bx);
}

__tile_global__ void fill_many_float32_kernel(
    float* const* __restrict__ buffers,
    const std::int64_t* __restrict__ elements,
    std::int64_t buffer_count,
    float value) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const int tensor = ct::bid().x;
  const int chunk = ct::bid().y;
  if (static_cast<std::int64_t>(tensor) >= buffer_count) {
    return;
  }

  float* values = ct::assume_aligned(buffers[tensor], 16_ic);
  const std::int64_t n = elements[tensor];
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(chunk) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto tile = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(value);
  ct::store_masked(values + idx, tile, mask);
}

__tile_global__ void fill_many_values_float32_kernel(
    float* const* __restrict__ buffers,
    const std::int64_t* __restrict__ elements,
    const float* __restrict__ fill_values,
    std::int64_t buffer_count) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const int tensor = ct::bid().x;
  const int chunk = ct::bid().y;
  if (static_cast<std::int64_t>(tensor) >= buffer_count) {
    return;
  }

  float* values = ct::assume_aligned(buffers[tensor], 16_ic);
  const std::int64_t n = elements[tensor];
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(chunk) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto tile = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(fill_values[tensor]);
  ct::store_masked(values + idx, tile, mask);
}

__tile_global__ void fill_many_values_bf16_bits_float32_kernel(
    std::uint16_t* const* __restrict__ buffers,
    const std::int64_t* __restrict__ elements,
    const float* __restrict__ fill_values,
    std::int64_t buffer_count) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const int tensor = ct::bid().x;
  const int chunk = ct::bid().y;
  if (static_cast<std::int64_t>(tensor) >= buffer_count) {
    return;
  }

  auto* values = ct::assume_aligned(
      reinterpret_cast<__nv_bfloat16*>(buffers[tensor]), 16_ic);
  const std::int64_t n = elements[tensor];
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(chunk) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto tile = ct::element_cast<__nv_bfloat16>(
      ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(fill_values[tensor]));
  ct::store_masked(values + idx, tile, mask);
}

__tile_global__ void init_gpt2_token_weight_float32_kernel(
    float* __restrict__ values,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  values = ct::assume_aligned(values, 16_ic);

  const int bx = ct::bid().x;
  constexpr int kTokenInitTileSize = NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE;
  using IndexTile =
      ct::tile<std::int64_t, decltype(ct::shape{NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE})>;
  auto idx = ct::iota<IndexTile>() +
      ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTokenInitTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto bucket = idx % ct::full<IndexTile>(17);
  auto shifted = bucket - ct::full<IndexTile>(8);
  auto value = ct::element_cast<float>(shifted) *
      ct::full<ct::tile<float, decltype(ct::shape{NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE})>>(0.01f);
  ct::store_masked(values + idx, value, mask);
}

__tile_global__ void init_gpt2_token_weight_fast_float32_kernel(
    float* __restrict__ values,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  values = ct::assume_aligned(values, 16_ic);

  const int bx = ct::bid().x;
  constexpr int kTokenInitTileSize = NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE;
  using IndexTile =
      ct::tile<std::int64_t, decltype(ct::shape{NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE})>;
  auto idx = ct::iota<IndexTile>() +
      ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTokenInitTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto bucket = idx % ct::full<IndexTile>(16);
  auto shifted = bucket - ct::full<IndexTile>(8);
  auto value = ct::element_cast<float>(shifted) *
      ct::full<ct::tile<float, decltype(ct::shape{NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE})>>(0.01f);
  ct::store_masked(values + idx, value, mask);
}

__tile_global__ void init_gpt2_token_weight_fast_int32_float32_kernel(
    float* __restrict__ values,
    int n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  values = ct::assume_aligned(values, 16_ic);

  const int bx = ct::bid().x;
  constexpr int kTokenInitTileSize = NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE;
  using IndexTile =
      ct::tile<int, decltype(ct::shape{NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE})>;
  auto idx = ct::iota<IndexTile>() +
      ct::full<IndexTile>(bx * kTokenInitTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto bucket = idx % ct::full<IndexTile>(16);
  auto shifted = bucket - ct::full<IndexTile>(8);
  auto value = ct::element_cast<float>(shifted) *
      ct::full<ct::tile<float, decltype(ct::shape{NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE})>>(0.01f);
  ct::store_masked(values + ct::element_cast<std::int64_t>(idx), value, mask);
}

__tile_global__ void init_gpt2_token_weight_with_bf16_shadow_float32_kernel(
    float* __restrict__ values,
    std::uint16_t* __restrict__ shadow_bf16_bits,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  values = ct::assume_aligned(values, 16_ic);
  auto* shadow = ct::assume_aligned(
      reinterpret_cast<__nv_bfloat16*>(shadow_bf16_bits), 16_ic);

  const int bx = ct::bid().x;
  constexpr int kTokenInitTileSize = NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE;
  using IndexTile =
      ct::tile<std::int64_t, decltype(ct::shape{NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE})>;
  auto idx = ct::iota<IndexTile>() +
      ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTokenInitTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto bucket = idx % ct::full<IndexTile>(17);
  auto shifted = bucket - ct::full<IndexTile>(8);
  auto value = ct::element_cast<float>(shifted) *
      ct::full<ct::tile<float, decltype(ct::shape{NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE})>>(0.01f);
  ct::store_masked(values + idx, value, mask);
  ct::store_masked(shadow + idx, ct::element_cast<__nv_bfloat16>(value), mask);
}

__tile_global__ void init_gpt2_token_weight_fast_with_bf16_shadow_float32_kernel(
    float* __restrict__ values,
    std::uint16_t* __restrict__ shadow_bf16_bits,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  values = ct::assume_aligned(values, 16_ic);
  auto* shadow = ct::assume_aligned(
      reinterpret_cast<__nv_bfloat16*>(shadow_bf16_bits), 16_ic);

  const int bx = ct::bid().x;
  constexpr int kTokenInitTileSize = NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE;
  using IndexTile =
      ct::tile<std::int64_t, decltype(ct::shape{NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE})>;
  auto idx = ct::iota<IndexTile>() +
      ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTokenInitTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto bucket = idx % ct::full<IndexTile>(16);
  auto shifted = bucket - ct::full<IndexTile>(8);
  auto value = ct::element_cast<float>(shifted) *
      ct::full<ct::tile<float, decltype(ct::shape{NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE})>>(0.01f);
  ct::store_masked(values + idx, value, mask);
  ct::store_masked(shadow + idx, ct::element_cast<__nv_bfloat16>(value), mask);
}

__tile_global__ void init_gpt2_token_weight_fast_int32_with_bf16_shadow_float32_kernel(
    float* __restrict__ values,
    std::uint16_t* __restrict__ shadow_bf16_bits,
    int n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  values = ct::assume_aligned(values, 16_ic);
  auto* shadow = ct::assume_aligned(
      reinterpret_cast<__nv_bfloat16*>(shadow_bf16_bits), 16_ic);

  const int bx = ct::bid().x;
  constexpr int kTokenInitTileSize = NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE;
  using IndexTile =
      ct::tile<int, decltype(ct::shape{NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE})>;
  auto idx = ct::iota<IndexTile>() +
      ct::full<IndexTile>(bx * kTokenInitTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto bucket = idx % ct::full<IndexTile>(16);
  auto shifted = bucket - ct::full<IndexTile>(8);
  auto value = ct::element_cast<float>(shifted) *
      ct::full<ct::tile<float, decltype(ct::shape{NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE})>>(0.01f);
  auto idx64 = ct::element_cast<std::int64_t>(idx);
  ct::store_masked(values + idx64, value, mask);
  ct::store_masked(shadow + idx64, ct::element_cast<__nv_bfloat16>(value), mask);
}

__global__ void init_gpt2_token_weight_threaded_float32_kernel(
    float* __restrict__ values,
    std::int64_t n,
    int bucket_mask,
    int bucket_mod) {
  const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
  for (std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n;
       idx += stride) {
    const int bucket = bucket_mask > 0
        ? (static_cast<int>(idx) & bucket_mask)
        : static_cast<int>(idx % bucket_mod);
    values[idx] = static_cast<float>(bucket - 8) * 0.01f;
  }
}

__global__ void init_gpt2_token_weight_threaded_with_bf16_shadow_float32_kernel(
    float* __restrict__ values,
    std::uint16_t* __restrict__ shadow_bf16_bits,
    std::int64_t n,
    int bucket_mask,
    int bucket_mod) {
  auto* shadow = reinterpret_cast<__nv_bfloat16*>(shadow_bf16_bits);
  const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
  for (std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n;
       idx += stride) {
    const int bucket = bucket_mask > 0
        ? (static_cast<int>(idx) & bucket_mask)
        : static_cast<int>(idx % bucket_mod);
    const float value = static_cast<float>(bucket - 8) * 0.01f;
    values[idx] = value;
    shadow[idx] = __float2bfloat16(value);
  }
}

__device__ __forceinline__ std::uint16_t bf16_bits_from_float(float value) {
  const __nv_bfloat16 bf16_value = __float2bfloat16(value);
  return *reinterpret_cast<const std::uint16_t*>(&bf16_value);
}

__device__ __forceinline__ ushort4 gpt2_token_weight_init_bf16_pattern4(int bucket) {
  switch (bucket & 15) {
    case 0:
      return make_ushort4(0xbda4u, 0xbd8fu, 0xbd76u, 0xbd4du);
    case 4:
      return make_ushort4(0xbd24u, 0xbcf6u, 0xbca4u, 0xbc24u);
    case 8:
      return make_ushort4(0x0000u, 0x3c24u, 0x3ca4u, 0x3cf6u);
    case 12:
      return make_ushort4(0x3d24u, 0x3d4du, 0x3d76u, 0x3d8fu);
    default:
      return make_ushort4(
          bf16_bits_from_float(static_cast<float>(bucket - 8) * 0.01f),
          bf16_bits_from_float(static_cast<float>(((bucket + 1) & 15) - 8) * 0.01f),
          bf16_bits_from_float(static_cast<float>(((bucket + 2) & 15) - 8) * 0.01f),
          bf16_bits_from_float(static_cast<float>(((bucket + 3) & 15) - 8) * 0.01f));
  }
}

__device__ __forceinline__ std::uint16_t gpt2_token_weight_init_bf16_pattern1(int bucket) {
  switch (bucket & 15) {
    case 0: return 0xbda4u;
    case 1: return 0xbd8fu;
    case 2: return 0xbd76u;
    case 3: return 0xbd4du;
    case 4: return 0xbd24u;
    case 5: return 0xbcf6u;
    case 6: return 0xbca4u;
    case 7: return 0xbc24u;
    case 8: return 0x0000u;
    case 9: return 0x3c24u;
    case 10: return 0x3ca4u;
    case 11: return 0x3cf6u;
    case 12: return 0x3d24u;
    case 13: return 0x3d4du;
    case 14: return 0x3d76u;
    default: return 0x3d8fu;
  }
}

__global__ void init_gpt2_token_weight_vector4_float32_kernel(
    float* __restrict__ values,
    std::int64_t n) {
  const std::int64_t idx = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (idx + 3 < n) {
    const int bucket = static_cast<int>(idx) & 15;
    reinterpret_cast<float4*>(values)[idx / 4] = make_float4(
        static_cast<float>(bucket - 8) * 0.01f,
        static_cast<float>(((bucket + 1) & 15) - 8) * 0.01f,
        static_cast<float>(((bucket + 2) & 15) - 8) * 0.01f,
        static_cast<float>(((bucket + 3) & 15) - 8) * 0.01f);
    return;
  }
  for (std::int64_t tail = idx; tail < n; ++tail) {
    const int bucket = static_cast<int>(tail) & 15;
    values[tail] = static_cast<float>(bucket - 8) * 0.01f;
  }
}

__global__ void init_gpt2_token_weight_vector4_with_bf16_shadow_float32_kernel(
    float* __restrict__ values,
    std::uint16_t* __restrict__ shadow_bf16_bits,
    std::int64_t n) {
  const std::int64_t idx = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (idx + 3 < n) {
    const int bucket = static_cast<int>(idx) & 15;
    const float value0 = static_cast<float>(bucket - 8) * 0.01f;
    const float value1 = static_cast<float>(((bucket + 1) & 15) - 8) * 0.01f;
    const float value2 = static_cast<float>(((bucket + 2) & 15) - 8) * 0.01f;
    const float value3 = static_cast<float>(((bucket + 3) & 15) - 8) * 0.01f;
    reinterpret_cast<float4*>(values)[idx / 4] = make_float4(value0, value1, value2, value3);
    reinterpret_cast<ushort4*>(shadow_bf16_bits)[idx / 4] =
        gpt2_token_weight_init_bf16_pattern4(bucket);
    return;
  }
  for (std::int64_t tail = idx; tail < n; ++tail) {
    const int bucket = static_cast<int>(tail) & 15;
    const float value = static_cast<float>(bucket - 8) * 0.01f;
    values[tail] = value;
    shadow_bf16_bits[tail] = gpt2_token_weight_init_bf16_pattern1(bucket);
  }
}

__global__ void init_gpt2_token_weight_vector4_with_bf16_shadow_convert_float32_kernel(
    float* __restrict__ values,
    std::uint16_t* __restrict__ shadow_bf16_bits,
    std::int64_t n) {
  const std::int64_t idx = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (idx + 3 < n) {
    const int bucket = static_cast<int>(idx) & 15;
    const float value0 = static_cast<float>(bucket - 8) * 0.01f;
    const float value1 = static_cast<float>(((bucket + 1) & 15) - 8) * 0.01f;
    const float value2 = static_cast<float>(((bucket + 2) & 15) - 8) * 0.01f;
    const float value3 = static_cast<float>(((bucket + 3) & 15) - 8) * 0.01f;
    reinterpret_cast<float4*>(values)[idx / 4] = make_float4(value0, value1, value2, value3);
    reinterpret_cast<ushort4*>(shadow_bf16_bits)[idx / 4] = make_ushort4(
        bf16_bits_from_float(value0),
        bf16_bits_from_float(value1),
        bf16_bits_from_float(value2),
        bf16_bits_from_float(value3));
    return;
  }
  for (std::int64_t tail = idx; tail < n; ++tail) {
    const int bucket = static_cast<int>(tail) & 15;
    const float value = static_cast<float>(bucket - 8) * 0.01f;
    values[tail] = value;
    shadow_bf16_bits[tail] = bf16_bits_from_float(value);
  }
}

__global__ void init_gpt2_token_weight_vector4_with_bf16_shadow_padded_float32_kernel(
    float* __restrict__ values,
    std::uint16_t* __restrict__ shadow_bf16_bits,
    std::int64_t public_n,
    std::int64_t total_n) {
  const std::int64_t idx = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (idx + 3 < total_n) {
    if (idx >= public_n) {
      reinterpret_cast<float4*>(values)[idx / 4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      reinterpret_cast<ushort4*>(shadow_bf16_bits)[idx / 4] = make_ushort4(0, 0, 0, 0);
      return;
    }
    if (idx + 3 < public_n) {
      const int bucket = static_cast<int>(idx) & 15;
      const float value0 = static_cast<float>(bucket - 8) * 0.01f;
      const float value1 = static_cast<float>(((bucket + 1) & 15) - 8) * 0.01f;
      const float value2 = static_cast<float>(((bucket + 2) & 15) - 8) * 0.01f;
      const float value3 = static_cast<float>(((bucket + 3) & 15) - 8) * 0.01f;
      reinterpret_cast<float4*>(values)[idx / 4] = make_float4(value0, value1, value2, value3);
      reinterpret_cast<ushort4*>(shadow_bf16_bits)[idx / 4] = make_ushort4(
          bf16_bits_from_float(value0),
          bf16_bits_from_float(value1),
          bf16_bits_from_float(value2),
          bf16_bits_from_float(value3));
      return;
    }
  }
  for (std::int64_t tail = idx; tail < total_n; ++tail) {
    if (tail < public_n) {
      const int bucket = static_cast<int>(tail) & 15;
      const float value = static_cast<float>(bucket - 8) * 0.01f;
      values[tail] = value;
      shadow_bf16_bits[tail] = bf16_bits_from_float(value);
    } else {
      values[tail] = 0.0f;
      shadow_bf16_bits[tail] = 0;
    }
  }
}

__global__ void init_gpt2_token_weight_vector4_strided_float32_kernel(
    float* __restrict__ values,
    std::int64_t n) {
  const std::int64_t vector_count = (n + 3) / 4;
  const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
  for (std::int64_t vector_idx =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       vector_idx < vector_count;
       vector_idx += stride) {
    const std::int64_t idx = vector_idx * 4;
    if (idx + 3 < n) {
      const int bucket = static_cast<int>(idx) & 15;
      reinterpret_cast<float4*>(values)[vector_idx] = make_float4(
          static_cast<float>(bucket - 8) * 0.01f,
          static_cast<float>(((bucket + 1) & 15) - 8) * 0.01f,
          static_cast<float>(((bucket + 2) & 15) - 8) * 0.01f,
          static_cast<float>(((bucket + 3) & 15) - 8) * 0.01f);
      continue;
    }
    for (std::int64_t tail = idx; tail < n; ++tail) {
      const int bucket = static_cast<int>(tail) & 15;
      values[tail] = static_cast<float>(bucket - 8) * 0.01f;
    }
  }
}

__global__ void init_gpt2_token_weight_vector4_strided_with_bf16_shadow_float32_kernel(
    float* __restrict__ values,
    std::uint16_t* __restrict__ shadow_bf16_bits,
    std::int64_t n) {
  const std::int64_t vector_count = (n + 3) / 4;
  const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
  for (std::int64_t vector_idx =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       vector_idx < vector_count;
       vector_idx += stride) {
    const std::int64_t idx = vector_idx * 4;
    if (idx + 3 < n) {
      const int bucket = static_cast<int>(idx) & 15;
      const float value0 = static_cast<float>(bucket - 8) * 0.01f;
      const float value1 = static_cast<float>(((bucket + 1) & 15) - 8) * 0.01f;
      const float value2 = static_cast<float>(((bucket + 2) & 15) - 8) * 0.01f;
      const float value3 = static_cast<float>(((bucket + 3) & 15) - 8) * 0.01f;
      reinterpret_cast<float4*>(values)[vector_idx] = make_float4(value0, value1, value2, value3);
      reinterpret_cast<ushort4*>(shadow_bf16_bits)[vector_idx] = make_ushort4(
          bf16_bits_from_float(value0),
          bf16_bits_from_float(value1),
          bf16_bits_from_float(value2),
          bf16_bits_from_float(value3));
      continue;
    }
    for (std::int64_t tail = idx; tail < n; ++tail) {
      const int bucket = static_cast<int>(tail) & 15;
      const float value = static_cast<float>(bucket - 8) * 0.01f;
      values[tail] = value;
      shadow_bf16_bits[tail] = bf16_bits_from_float(value);
    }
  }
}

__tile_global__ void sumsq_partials_float32_kernel(
    const float* __restrict__ values,
    float* __restrict__ partials,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  values = ct::assume_aligned(values, 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{NFN_TILE_CUDA_OPTIMIZER_TILE_SHAPE})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kOptimizerTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto tile = ct::load_masked(values + idx, mask);
  auto zero = ct::full<decltype(tile)>(0.0f);
  auto squared = ct::select(mask, tile * tile, zero);
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(partials + out_idx, ct::sum(squared, 0_ic));
}

__tile_global__ void sumsq_partials_many_float32_kernel(
    const float* const* __restrict__ buffers,
    const std::int64_t* __restrict__ elements,
    const std::int64_t* __restrict__ partial_offsets,
    float* __restrict__ partials,
    std::int64_t buffer_count) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const int tensor = ct::bid().x;
  const int chunk = ct::bid().y;
  if (static_cast<std::int64_t>(tensor) >= buffer_count) {
    return;
  }

  const float* values = ct::assume_aligned(buffers[tensor], 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);
  const std::int64_t n = elements[tensor];
  if (static_cast<std::int64_t>(chunk) * kOptimizerTileSize >= n) {
    return;
  }
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{NFN_TILE_CUDA_OPTIMIZER_TILE_SHAPE})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(chunk) * kOptimizerTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto tile = ct::load_masked(values + idx, mask);
  auto zero = ct::full<decltype(tile)>(0.0f);
  auto squared = ct::select(mask, tile * tile, zero);
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(partial_offsets[tensor] + static_cast<std::int64_t>(chunk));
  ct::store(partials + out_idx, ct::sum(squared, 0_ic));
}

__tile_global__ void sumsq_partials_many_bf16_bits_float32_kernel(
    const std::uint16_t* const* __restrict__ buffers,
    const std::int64_t* __restrict__ elements,
    const std::int64_t* __restrict__ partial_offsets,
    float* __restrict__ partials,
    std::int64_t buffer_count) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const int tensor = ct::bid().x;
  const int chunk = ct::bid().y;
  if (static_cast<std::int64_t>(tensor) >= buffer_count) {
    return;
  }

  const auto* values = ct::assume_aligned(
      reinterpret_cast<const __nv_bfloat16*>(buffers[tensor]), 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);
  const std::int64_t n = elements[tensor];
  if (static_cast<std::int64_t>(chunk) * kOptimizerTileSize >= n) {
    return;
  }
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{NFN_TILE_CUDA_OPTIMIZER_TILE_SHAPE})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(chunk) * kOptimizerTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto tile = ct::element_cast<float>(ct::load_masked(values + idx, mask));
  auto zero = ct::full<decltype(tile)>(0.0f);
  auto squared = ct::select(mask, tile * tile, zero);
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(partial_offsets[tensor] + static_cast<std::int64_t>(chunk));
  ct::store(partials + out_idx, ct::sum(squared, 0_ic));
}

__tile_global__ void scale_inplace_float32_kernel(
    float* __restrict__ values,
    std::int64_t n,
    float scale) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  values = ct::assume_aligned(values, 16_ic);

  const int bx = ct::bid().x;
  auto view = ct::partition_view{ct::tensor_span{values, ct::extents{n}}, ct::shape{NFN_TILE_CUDA_OPTIMIZER_TILE_SHAPE}};
  auto tile = view.load_masked(bx);
  view.store_masked(tile * ct::full<decltype(tile)>(scale), bx);
}

__tile_global__ void global_norm_clip_scale_float32_kernel(
    const float* __restrict__ sumsq_partials,
    float* __restrict__ clip_scale,
    std::int64_t partial_count,
    float max_norm,
    float eps) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  sumsq_partials = ct::assume_aligned(sumsq_partials, 16_ic);
  clip_scale = ct::assume_aligned(clip_scale, 16_ic);

  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{NFN_TILE_CUDA_OPTIMIZER_TILE_SHAPE})>;
  auto idx = ct::iota<IndexTile>();
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{NFN_TILE_CUDA_OPTIMIZER_TILE_SHAPE})>>(0.0f);
  for (std::int64_t offset = 0; offset < partial_count; offset += kOptimizerTileSize) {
    auto pos = idx + ct::full<IndexTile>(offset);
    auto mask = pos < ct::full<IndexTile>(partial_count);
    auto value = ct::load_masked(sumsq_partials + pos, mask);
    acc = acc + ct::select(mask, value, ct::full<decltype(value)>(0.0f));
  }
  auto total = ct::sum(acc, 0_ic);
  auto norm = ct::sqrt(total);
  auto one = ct::full<decltype(norm)>(1.0f);
  auto raw_scale = ct::full<decltype(norm)>(max_norm) / (norm + ct::full<decltype(norm)>(eps));
  auto scale = ct::select(raw_scale < one, raw_scale, one);
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(0);
  ct::store(clip_scale + out_idx, scale);
}

__tile_global__ void scale_inplace_by_device_float32_kernel(
    float* __restrict__ values,
    const float* __restrict__ scale,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  values = ct::assume_aligned(values, 16_ic);
  scale = ct::assume_aligned(scale, 16_ic);

  const int bx = ct::bid().x;
  auto view = ct::partition_view{ct::tensor_span{values, ct::extents{n}}, ct::shape{NFN_TILE_CUDA_OPTIMIZER_TILE_SHAPE}};
  auto tile = view.load_masked(bx);
  auto scale_tile = ct::full<decltype(tile)>(*scale);
  view.store_masked(tile * scale_tile, bx);
}

__tile_global__ void adamw_step_float32_kernel(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    std::int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float sqrt_bias_correction2) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  param = ct::assume_aligned(param, 16_ic);
  grad = ct::assume_aligned(grad, 16_ic);
  exp_avg = ct::assume_aligned(exp_avg, 16_ic);
  exp_avg_sq = ct::assume_aligned(exp_avg_sq, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{NFN_TILE_CUDA_OPTIMIZER_TILE_SHAPE})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kOptimizerTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto p = ct::load_masked(param + idx, mask);
  auto g = ct::load_masked(grad + idx, mask);
  auto m = ct::load_masked(exp_avg + idx, mask);
  auto v = ct::load_masked(exp_avg_sq + idx, mask);
  auto one = ct::full<decltype(p)>(1.0f);
  auto beta1_tile = ct::full<decltype(p)>(beta1);
  auto beta2_tile = ct::full<decltype(p)>(beta2);
  auto next_m = beta1_tile * m + (one - beta1_tile) * g;
  auto next_v = beta2_tile * v + (one - beta2_tile) * g * g;
  auto decayed = p * (one - ct::full<decltype(p)>(lr * weight_decay));
  auto denom = ct::sqrt(next_v) / ct::full<decltype(p)>(sqrt_bias_correction2) + ct::full<decltype(p)>(eps);
  auto step_size = ct::full<decltype(p)>(lr / bias_correction1);
  auto next_p = decayed - step_size * next_m / denom;
  ct::store_masked(param + idx, next_p, mask);
  ct::store_masked(exp_avg + idx, next_m, mask);
  ct::store_masked(exp_avg_sq + idx, next_v, mask);
}

__tile_global__ void adamw_step_with_device_scale_float32_kernel(
    float* __restrict__ param,
    const float* __restrict__ grad,
    const float* __restrict__ grad_scale,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    std::int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float sqrt_bias_correction2) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  param = ct::assume_aligned(param, 16_ic);
  grad = ct::assume_aligned(grad, 16_ic);
  grad_scale = ct::assume_aligned(grad_scale, 16_ic);
  exp_avg = ct::assume_aligned(exp_avg, 16_ic);
  exp_avg_sq = ct::assume_aligned(exp_avg_sq, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{NFN_TILE_CUDA_OPTIMIZER_TILE_SHAPE})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kOptimizerTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto p = ct::load_masked(param + idx, mask);
  auto g = ct::load_masked(grad + idx, mask) * ct::full<decltype(p)>(*grad_scale);
  auto m = ct::load_masked(exp_avg + idx, mask);
  auto v = ct::load_masked(exp_avg_sq + idx, mask);
  auto one = ct::full<decltype(p)>(1.0f);
  auto beta1_tile = ct::full<decltype(p)>(beta1);
  auto beta2_tile = ct::full<decltype(p)>(beta2);
  auto next_m = beta1_tile * m + (one - beta1_tile) * g;
  auto next_v = beta2_tile * v + (one - beta2_tile) * g * g;
  auto decayed = p * (one - ct::full<decltype(p)>(lr * weight_decay));
  auto denom = ct::sqrt(next_v) / ct::full<decltype(p)>(sqrt_bias_correction2) + ct::full<decltype(p)>(eps);
  auto step_size = ct::full<decltype(p)>(lr / bias_correction1);
  auto next_p = decayed - step_size * next_m / denom;
  ct::store_masked(param + idx, next_p, mask);
  ct::store_masked(exp_avg + idx, next_m, mask);
  ct::store_masked(exp_avg_sq + idx, next_v, mask);
}

__tile_global__ void adamw_step_many_with_device_scale_float32_kernel(
    float* const* __restrict__ params,
    const float* const* __restrict__ grads,
    const float* __restrict__ grad_scale,
    float* const* __restrict__ exp_avgs,
    float* const* __restrict__ exp_avg_sqs,
    const std::int64_t* __restrict__ elements,
    const float* __restrict__ weight_decays,
    std::int64_t buffer_count,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bias_correction1,
    float sqrt_bias_correction2) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const int tensor = ct::bid().x;
  const int chunk = ct::bid().y;
  if (static_cast<std::int64_t>(tensor) >= buffer_count) {
    return;
  }

  float* param = ct::assume_aligned(params[tensor], 16_ic);
  const float* grad = ct::assume_aligned(grads[tensor], 16_ic);
  grad_scale = ct::assume_aligned(grad_scale, 16_ic);
  float* exp_avg = ct::assume_aligned(exp_avgs[tensor], 16_ic);
  float* exp_avg_sq = ct::assume_aligned(exp_avg_sqs[tensor], 16_ic);

  const std::int64_t n = elements[tensor];
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{NFN_TILE_CUDA_OPTIMIZER_TILE_SHAPE})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(chunk) * kOptimizerTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto p = ct::load_masked(param + idx, mask);
  auto g = ct::load_masked(grad + idx, mask) * ct::full<decltype(p)>(*grad_scale);
  auto m = ct::load_masked(exp_avg + idx, mask);
  auto v = ct::load_masked(exp_avg_sq + idx, mask);
  auto one = ct::full<decltype(p)>(1.0f);
  auto beta1_tile = ct::full<decltype(p)>(beta1);
  auto beta2_tile = ct::full<decltype(p)>(beta2);
  auto next_m = beta1_tile * m + (one - beta1_tile) * g;
  auto next_v = beta2_tile * v + (one - beta2_tile) * g * g;
  auto decayed = p * (one - ct::full<decltype(p)>(lr * weight_decays[tensor]));
  auto denom = ct::sqrt(next_v) / ct::full<decltype(p)>(sqrt_bias_correction2) + ct::full<decltype(p)>(eps);
  auto step_size = ct::full<decltype(p)>(lr / bias_correction1);
  auto next_p = decayed - step_size * next_m / denom;
  ct::store_masked(param + idx, next_p, mask);
  ct::store_masked(exp_avg + idx, next_m, mask);
  ct::store_masked(exp_avg_sq + idx, next_v, mask);
}

__tile_global__ void adamw_step_many_with_device_scale_bf16_shadow_float32_kernel(
    float* const* __restrict__ params,
    const float* const* __restrict__ grads,
    const float* __restrict__ grad_scale,
    float* const* __restrict__ exp_avgs,
    float* const* __restrict__ exp_avg_sqs,
    const std::int64_t* __restrict__ elements,
    const float* __restrict__ weight_decays,
    const std::int64_t* __restrict__ bf16_shadow_offsets,
    std::uint16_t* __restrict__ bf16_shadow_bits,
    std::int64_t buffer_count,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bias_correction1,
    float sqrt_bias_correction2) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const int tensor = ct::bid().x;
  const int chunk = ct::bid().y;
  if (static_cast<std::int64_t>(tensor) >= buffer_count) {
    return;
  }

  float* param = ct::assume_aligned(params[tensor], 16_ic);
  const float* grad = ct::assume_aligned(grads[tensor], 16_ic);
  grad_scale = ct::assume_aligned(grad_scale, 16_ic);
  float* exp_avg = ct::assume_aligned(exp_avgs[tensor], 16_ic);
  float* exp_avg_sq = ct::assume_aligned(exp_avg_sqs[tensor], 16_ic);
  auto* shadow = ct::assume_aligned(reinterpret_cast<__nv_bfloat16*>(bf16_shadow_bits), 16_ic);

  const std::int64_t n = elements[tensor];
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{NFN_TILE_CUDA_OPTIMIZER_TILE_SHAPE})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(chunk) * kOptimizerTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto p = ct::load_masked(param + idx, mask);
  auto g = ct::load_masked(grad + idx, mask) * ct::full<decltype(p)>(*grad_scale);
  auto m = ct::load_masked(exp_avg + idx, mask);
  auto v = ct::load_masked(exp_avg_sq + idx, mask);
  auto one = ct::full<decltype(p)>(1.0f);
  auto beta1_tile = ct::full<decltype(p)>(beta1);
  auto beta2_tile = ct::full<decltype(p)>(beta2);
  auto next_m = beta1_tile * m + (one - beta1_tile) * g;
  auto next_v = beta2_tile * v + (one - beta2_tile) * g * g;
  auto decayed = p * (one - ct::full<decltype(p)>(lr * weight_decays[tensor]));
  auto denom = ct::sqrt(next_v) / ct::full<decltype(p)>(sqrt_bias_correction2) + ct::full<decltype(p)>(eps);
  auto step_size = ct::full<decltype(p)>(lr / bias_correction1);
  auto next_p = decayed - step_size * next_m / denom;
  ct::store_masked(param + idx, next_p, mask);
  ct::store_masked(exp_avg + idx, next_m, mask);
  ct::store_masked(exp_avg_sq + idx, next_v, mask);

  const std::int64_t shadow_offset = bf16_shadow_offsets[tensor];
  if (shadow_offset >= 0) {
    ct::store_masked(shadow + ct::full<IndexTile>(shadow_offset) + idx,
                     ct::element_cast<__nv_bfloat16>(next_p),
                     mask);
  }
}

__tile_global__ void adamw_step_many_with_device_scale_bf16_param_float32_kernel(
    std::uint16_t* const* __restrict__ params_bf16_bits,
    const float* const* __restrict__ grads,
    const float* __restrict__ grad_scale,
    float* const* __restrict__ exp_avgs,
    float* const* __restrict__ exp_avg_sqs,
    const std::int64_t* __restrict__ elements,
    const float* __restrict__ weight_decays,
    std::int64_t buffer_count,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bias_correction1,
    float sqrt_bias_correction2) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const int tensor = ct::bid().x;
  const int chunk = ct::bid().y;
  if (static_cast<std::int64_t>(tensor) >= buffer_count) {
    return;
  }

  auto* param = ct::assume_aligned(
      reinterpret_cast<__nv_bfloat16*>(params_bf16_bits[tensor]), 16_ic);
  const float* grad = ct::assume_aligned(grads[tensor], 16_ic);
  grad_scale = ct::assume_aligned(grad_scale, 16_ic);
  float* exp_avg = ct::assume_aligned(exp_avgs[tensor], 16_ic);
  float* exp_avg_sq = ct::assume_aligned(exp_avg_sqs[tensor], 16_ic);

  const std::int64_t n = elements[tensor];
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{NFN_TILE_CUDA_OPTIMIZER_TILE_SHAPE})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(chunk) * kOptimizerTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto p = ct::element_cast<float>(ct::load_masked(param + idx, mask));
  auto g = ct::load_masked(grad + idx, mask) * ct::full<decltype(p)>(*grad_scale);
  auto m = ct::load_masked(exp_avg + idx, mask);
  auto v = ct::load_masked(exp_avg_sq + idx, mask);
  auto one = ct::full<decltype(p)>(1.0f);
  auto beta1_tile = ct::full<decltype(p)>(beta1);
  auto beta2_tile = ct::full<decltype(p)>(beta2);
  auto next_m = beta1_tile * m + (one - beta1_tile) * g;
  auto next_v = beta2_tile * v + (one - beta2_tile) * g * g;
  auto decayed = p * (one - ct::full<decltype(p)>(lr * weight_decays[tensor]));
  auto denom = ct::sqrt(next_v) / ct::full<decltype(p)>(sqrt_bias_correction2) + ct::full<decltype(p)>(eps);
  auto step_size = ct::full<decltype(p)>(lr / bias_correction1);
  auto next_p = decayed - step_size * next_m / denom;
  ct::store_masked(param + idx, ct::element_cast<__nv_bfloat16>(next_p), mask);
  ct::store_masked(exp_avg + idx, next_m, mask);
  ct::store_masked(exp_avg_sq + idx, next_v, mask);
}

__tile_global__ void adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32_kernel(
    std::uint16_t* const* __restrict__ params_bf16_bits,
    const std::uint16_t* const* __restrict__ grads_bf16_bits,
    const float* __restrict__ grad_scale,
    float* const* __restrict__ exp_avgs,
    float* const* __restrict__ exp_avg_sqs,
    const std::int64_t* __restrict__ elements,
    const float* __restrict__ weight_decays,
    std::int64_t buffer_count,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bias_correction1,
    float sqrt_bias_correction2) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const int tensor = ct::bid().x;
  const int chunk = ct::bid().y;
  if (static_cast<std::int64_t>(tensor) >= buffer_count) {
    return;
  }

  auto* param = ct::assume_aligned(
      reinterpret_cast<__nv_bfloat16*>(params_bf16_bits[tensor]), 16_ic);
  const auto* grad = ct::assume_aligned(
      reinterpret_cast<const __nv_bfloat16*>(grads_bf16_bits[tensor]), 16_ic);
  grad_scale = ct::assume_aligned(grad_scale, 16_ic);
  float* exp_avg = ct::assume_aligned(exp_avgs[tensor], 16_ic);
  float* exp_avg_sq = ct::assume_aligned(exp_avg_sqs[tensor], 16_ic);

  const std::int64_t n = elements[tensor];
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{NFN_TILE_CUDA_OPTIMIZER_TILE_SHAPE})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(chunk) * kOptimizerTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto p = ct::element_cast<float>(ct::load_masked(param + idx, mask));
  auto g = ct::element_cast<float>(ct::load_masked(grad + idx, mask)) *
      ct::full<decltype(p)>(*grad_scale);
  auto m = ct::load_masked(exp_avg + idx, mask);
  auto v = ct::load_masked(exp_avg_sq + idx, mask);
  auto one = ct::full<decltype(p)>(1.0f);
  auto beta1_tile = ct::full<decltype(p)>(beta1);
  auto beta2_tile = ct::full<decltype(p)>(beta2);
  auto next_m = beta1_tile * m + (one - beta1_tile) * g;
  auto next_v = beta2_tile * v + (one - beta2_tile) * g * g;
  auto decayed = p * (one - ct::full<decltype(p)>(lr * weight_decays[tensor]));
  auto denom = ct::sqrt(next_v) / ct::full<decltype(p)>(sqrt_bias_correction2) + ct::full<decltype(p)>(eps);
  auto step_size = ct::full<decltype(p)>(lr / bias_correction1);
  auto next_p = decayed - step_size * next_m / denom;
  ct::store_masked(param + idx, ct::element_cast<__nv_bfloat16>(next_p), mask);
  ct::store_masked(exp_avg + idx, next_m, mask);
  ct::store_masked(exp_avg_sq + idx, next_v, mask);
}

__tile_global__ void scalar_ternary_float32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ out,
    std::int64_t n,
    float value,
    int op) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  a = ct::assume_aligned(a, 16_ic);
  b = ct::assume_aligned(b, 16_ic);
  c = ct::assume_aligned(c, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  auto a_tile = ct::partition_view{ct::tensor_span{a, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto b_tile = ct::partition_view{ct::tensor_span{b, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto c_tile = ct::partition_view{ct::tensor_span{c, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto scalar = ct::full<decltype(a_tile)>(value);
  auto result = op == 0 ? c_tile - scalar * (a_tile - b_tile) : a_tile + scalar * b_tile + c_tile;
  auto out_view = ct::partition_view{ct::tensor_span{out, ct::extents{n}}, ct::shape{1024_ic}};
  out_view.store_masked(result, bx);
}

__tile_global__ void vector_binary_float32_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    const float* __restrict__ scale0,
    const float* __restrict__ scale1,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t dim,
    int op) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  lhs = ct::assume_aligned(lhs, 16_ic);
  rhs = ct::assume_aligned(rhs, 16_ic);
  scale0 = ct::assume_aligned(scale0, 16_ic);
  out = ct::assume_aligned(out, 16_ic);
  if (scale1 != nullptr) {
    scale1 = ct::assume_aligned(scale1, 16_ic);
  }

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<decltype(idx)>(n);
  auto lhs_ptrs = lhs + idx;
  auto rhs_ptrs = rhs + idx;
  auto lane = idx % ct::full<decltype(idx)>(dim);
  auto scale0_ptrs = scale0 + lane;
  auto lhs_tile = ct::load_masked(lhs_ptrs, mask);
  auto rhs_tile = ct::load_masked(rhs_ptrs, mask);
  auto scale0_tile = ct::load_masked(scale0_ptrs, mask);
  auto result = lhs_tile + scale0_tile * rhs_tile;
  if (op == 1 && scale1 != nullptr) {
    auto scale1_ptrs = scale1 + lane;
    auto scale1_tile = ct::load_masked(scale1_ptrs, mask);
    result = scale0_tile * lhs_tile + scale1_tile * rhs_tile;
  } else if (op == 2) {
    auto one = ct::full<decltype(scale0_tile)>(1.0f);
    auto beta = one / (one + ct::exp(-scale0_tile));
    auto alpha = ct::sqrt(one - beta * beta);
    result = alpha * lhs_tile + beta * rhs_tile;
  }
  ct::store_masked(out + idx, result, mask);
}

__tile_global__ void qk_gain_float32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ gain,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t heads,
    std::int64_t inner) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  q = ct::assume_aligned(q, 16_ic);
  gain = ct::assume_aligned(gain, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto head = (idx / ct::full<IndexTile>(inner)) % ct::full<IndexTile>(heads);
  auto q_tile = ct::load_masked(q + idx, mask);
  auto gain_tile = ct::load_masked(gain + head, mask);
  ct::store_masked(out + idx, q_tile * gain_tile, mask);
}

__tile_global__ void dyt_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t dim,
    float alpha) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  weight = ct::assume_aligned(weight, 16_ic);
  bias = ct::assume_aligned(bias, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto lane = idx % ct::full<IndexTile>(dim);
  auto x_tile = ct::load_masked(x + idx, mask);
  auto weight_tile = ct::load_masked(weight + lane, mask);
  auto bias_tile = ct::load_masked(bias + lane, mask);
  auto alpha_tile = ct::full<decltype(x_tile)>(alpha);
  auto t = ct::tanh(alpha_tile * x_tile);
  ct::store_masked(out + idx, weight_tile * t + bias_tile, mask);
}

__tile_global__ void reshape_heads_float32_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto head_dim_tile = ct::full<IndexTile>(head_dim);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto heads_tile = ct::full<IndexTile>(heads);
  auto d = idx % head_dim_tile;
  auto s = (idx / head_dim_tile) % seq_tile;
  auto h = (idx / (head_dim_tile * seq_tile)) % heads_tile;
  auto b = idx / (head_dim_tile * seq_tile * heads_tile);
  auto src = (b * seq_tile + s) * (heads_tile * head_dim_tile) + h * head_dim_tile + d;
  auto value = ct::load_masked(x + src, mask);
  ct::store_masked(out + idx, value, mask);
}

__tile_global__ void merge_heads_float32_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto head_dim_tile = ct::full<IndexTile>(head_dim);
  auto heads_tile = ct::full<IndexTile>(heads);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto d = idx % head_dim_tile;
  auto h = (idx / head_dim_tile) % heads_tile;
  auto s = (idx / (head_dim_tile * heads_tile)) % seq_tile;
  auto b = idx / (head_dim_tile * heads_tile * seq_tile);
  auto src = ((b * heads_tile + h) * seq_tile + s) * head_dim_tile + d;
  auto value = ct::load_masked(x + src, mask);
  ct::store_masked(out + idx, value, mask);
}

__tile_global__ void repeat_kv_float32_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t kv_heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    std::int64_t repeats) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto head_dim_tile = ct::full<IndexTile>(head_dim);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto kv_heads_tile = ct::full<IndexTile>(kv_heads);
  auto repeats_tile = ct::full<IndexTile>(repeats);
  auto d = idx % head_dim_tile;
  auto s = (idx / head_dim_tile) % seq_tile;
  auto h = (idx / (head_dim_tile * seq_tile)) % (kv_heads_tile * repeats_tile);
  auto b = idx / (head_dim_tile * seq_tile * kv_heads_tile * repeats_tile);
  auto src_h = h / repeats_tile;
  auto src = ((b * kv_heads_tile + src_h) * seq_tile + s) * head_dim_tile + d;
  auto value = ct::load_masked(x + src, mask);
  ct::store_masked(out + idx, value, mask);
}

__tile_global__ void broadcast_expert_routes_float32_kernel(
    const float* __restrict__ weights,
    const std::int64_t* __restrict__ indices,
    float* __restrict__ out_weights,
    std::int64_t* __restrict__ out_indices,
    std::int64_t n,
    std::int64_t route_seq,
    std::int64_t seq_len,
    std::int64_t route_width) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  weights = ct::assume_aligned(weights, 16_ic);
  indices = ct::assume_aligned(indices, 16_ic);
  out_weights = ct::assume_aligned(out_weights, 16_ic);
  out_indices = ct::assume_aligned(out_indices, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto width_tile = ct::full<IndexTile>(route_width);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto route_seq_tile = ct::full<IndexTile>(route_seq);
  auto k = idx % width_tile;
  auto s = (idx / width_tile) % seq_tile;
  auto b = idx / (width_tile * seq_tile);
  auto src_s = ct::select(route_seq_tile == ct::full<IndexTile>(1), ct::full<IndexTile>(0), s);
  auto src = (b * route_seq_tile + src_s) * width_tile + k;
  auto weight = ct::load_masked(weights + src, mask);
  auto index = ct::load_masked(indices + src, mask);
  ct::store_masked(out_weights + idx, weight, mask);
  ct::store_masked(out_indices + idx, index, mask);
}

__tile_global__ void broadcast_chunk_routes_float32_kernel(
    const float* __restrict__ weights,
    const std::int64_t* __restrict__ indices,
    float* __restrict__ out_weights,
    std::int64_t* __restrict__ out_indices,
    std::int64_t n,
    std::int64_t chunks,
    std::int64_t seq_len,
    std::int64_t route_width,
    std::int64_t chunk_size) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  weights = ct::assume_aligned(weights, 16_ic);
  indices = ct::assume_aligned(indices, 16_ic);
  out_weights = ct::assume_aligned(out_weights, 16_ic);
  out_indices = ct::assume_aligned(out_indices, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto width_tile = ct::full<IndexTile>(route_width);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto chunks_tile = ct::full<IndexTile>(chunks);
  auto chunk_size_tile = ct::full<IndexTile>(chunk_size);
  auto k = idx % width_tile;
  auto s = (idx / width_tile) % seq_tile;
  auto b = idx / (width_tile * seq_tile);
  auto raw_chunk = s / chunk_size_tile;
  auto chunk = ct::select(raw_chunk >= chunks_tile, chunks_tile - ct::full<IndexTile>(1), raw_chunk);
  auto src = (b * chunks_tile + chunk) * width_tile + k;
  auto weight = ct::load_masked(weights + src, mask);
  auto index = ct::load_masked(indices + src, mask);
  ct::store_masked(out_weights + idx, weight, mask);
  ct::store_masked(out_indices + idx, index, mask);
}

__tile_global__ void byte_patch_merge_float32_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t source_len,
    std::int64_t target_len,
    std::int64_t dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto dim_tile = ct::full<IndexTile>(dim);
  auto target_tile = ct::full<IndexTile>(target_len);
  auto source_tile = ct::full<IndexTile>(source_len);
  auto d = idx % dim_tile;
  auto t = (idx / dim_tile) % target_tile;
  auto b = idx / (dim_tile * target_tile);
  auto src_t = (t * source_tile) / target_tile;
  auto src = (b * source_tile + src_t) * dim_tile + d;
  auto value = ct::load_masked(x + src, mask);
  ct::store_masked(out + idx, value, mask);
}

__tile_global__ void byte_patch_embed_float32_kernel(
    const std::int64_t* __restrict__ tokens,
    const float* __restrict__ embedding,
    const float* __restrict__ proj,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t model_dim,
    std::int64_t patch_size,
    std::int64_t stride,
    std::int64_t out_len,
    std::int64_t vocab_size) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  tokens = ct::assume_aligned(tokens, 16_ic);
  embedding = ct::assume_aligned(embedding, 16_ic);
  proj = ct::assume_aligned(proj, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto model_dim_tile = ct::full<IndexTile>(model_dim);
  auto out_len_tile = ct::full<IndexTile>(out_len);
  auto out_c = idx % model_dim_tile;
  auto patch_idx = (idx / model_dim_tile) % out_len_tile;
  auto batch_idx = idx / (model_dim_tile * out_len_tile);
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto seq_len_tile = ct::full<IndexTile>(seq_len);
  auto vocab_tile = ct::full<IndexTile>(vocab_size);
  auto zero_i = ct::full<IndexTile>(0);
  for (std::int64_t k = 0; k < patch_size; ++k) {
    auto token_pos = patch_idx * ct::full<IndexTile>(stride) + ct::full<IndexTile>(k);
    auto valid_token = mask & (token_pos < seq_len_tile);
    auto token_raw = ct::load_masked(tokens + batch_idx * seq_len_tile + token_pos, valid_token);
    auto token_hi = ct::select(token_raw >= vocab_tile, vocab_tile - ct::full<IndexTile>(1), token_raw);
    auto token_id = ct::select(token_hi < zero_i, zero_i, token_hi);
    for (std::int64_t in_c = 0; in_c < model_dim; ++in_c) {
      auto embed = ct::load_masked(embedding + token_id * model_dim_tile + ct::full<IndexTile>(in_c), valid_token);
      auto weight = ct::load_masked(
          proj + (out_c * model_dim_tile + ct::full<IndexTile>(in_c)) * ct::full<IndexTile>(patch_size) + ct::full<IndexTile>(k),
          mask);
      acc = acc + ct::select(valid_token, embed * weight, ct::full<decltype(embed)>(0.0f));
    }
  }
  ct::store_masked(out + idx, acc, mask);
}

__tile_global__ void causal_chunk_state_float32_kernel(
    const float* __restrict__ hidden,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t dim,
    std::int64_t chunk_size,
    std::int64_t chunks,
    int mode) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  hidden = ct::assume_aligned(hidden, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto dim_tile = ct::full<IndexTile>(dim);
  auto chunks_tile = ct::full<IndexTile>(chunks);
  auto d = idx % dim_tile;
  auto chunk = (idx / dim_tile) % chunks_tile;
  auto b = idx / (dim_tile * chunks_tile);
  auto chunk_size_tile = ct::full<IndexTile>(chunk_size);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto zero_i = ct::full<IndexTile>(0);
  auto one_i = ct::full<IndexTile>(1);
  auto start = chunk * chunk_size_tile;
  auto end_mean = start + chunk_size_tile;
  end_mean = ct::select(end_mean > seq_tile, seq_tile, end_mean);
  auto boundary = start - one_i;
  boundary = ct::select(boundary < zero_i, zero_i, boundary);
  boundary = ct::select(boundary >= seq_tile, seq_tile - one_i, boundary);

  auto sum = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto count = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t t = 0; t < seq_len; ++t) {
    auto t_tile = ct::full<IndexTile>(t);
    auto active = mode == 0 ? ((t_tile >= start) & (t_tile < end_mean)) : (t_tile <= boundary);
    active = active & mask;
    auto src = (b * seq_tile + t_tile) * dim_tile + d;
    auto value = ct::load_masked(hidden + src, active);
    sum = sum + ct::select(active, value, ct::full<decltype(value)>(0.0f));
    count = count + ct::select(active, ct::full<decltype(value)>(1.0f), ct::full<decltype(value)>(0.0f));
  }
  auto denom = ct::select(count > ct::full<decltype(count)>(1.0f), count, ct::full<decltype(count)>(1.0f));
  ct::store_masked(out + idx, sum / denom, mask);
}

__tile_global__ void latent_mse_partials_float32_kernel(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ partials,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  pred = ct::assume_aligned(pred, 16_ic);
  target = ct::assume_aligned(target, 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto pred_tile = ct::load_masked(pred + idx, mask);
  auto target_tile = ct::load_masked(target + idx, mask);
  auto zero = ct::full<decltype(pred_tile)>(0.0f);
  auto diff = ct::select(mask, pred_tile - target_tile, zero);
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(partials + out_idx, ct::sum(diff * diff, 0_ic));
}

__tile_global__ void semantic_alignment_loss_items_float32_kernel(
    const float* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    const std::int64_t* __restrict__ term_counts,
    float* __restrict__ losses,
    float* __restrict__ counts,
    std::int64_t n,
    std::int64_t dims,
    std::int64_t terms,
    std::int64_t ignore_index) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logits = ct::assume_aligned(logits, 16_ic);
  targets = ct::assume_aligned(targets, 16_ic);
  term_counts = ct::assume_aligned(term_counts, 16_ic);
  losses = ct::assume_aligned(losses, 16_ic);
  counts = ct::assume_aligned(counts, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto dims_tile = ct::full<IndexTile>(dims);
  auto terms_tile = ct::full<IndexTile>(terms);
  auto dim_idx = idx % dims_tile;
  auto row_idx = idx / dims_tile;
  auto target = ct::load_masked(targets + row_idx * dims_tile + dim_idx, mask);
  auto term_count_raw = ct::load_masked(term_counts + dim_idx, mask);
  auto term_count = ct::select(term_count_raw > terms_tile, terms_tile, term_count_raw);
  auto valid = mask & (target != ct::full<IndexTile>(ignore_index)) & (target >= ct::full<IndexTile>(0)) & (target < term_count);
  auto base = (row_idx * dims_tile + dim_idx) * terms_tile;
  auto max_val = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(-3.4028234663852886e38f);
  for (std::int64_t term = 0; term < terms; ++term) {
    auto term_tile = ct::full<IndexTile>(term);
    auto active = valid & (term_tile < term_count);
    auto value = ct::load_masked(logits + base + term_tile, active);
    max_val = ct::select(active & (value > max_val), value, max_val);
  }
  auto sum_exp = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t term = 0; term < terms; ++term) {
    auto term_tile = ct::full<IndexTile>(term);
    auto active = valid & (term_tile < term_count);
    auto value = ct::load_masked(logits + base + term_tile, active);
    sum_exp = sum_exp + ct::select(active, ct::exp(value - max_val), ct::full<decltype(value)>(0.0f));
  }
  auto safe_target = ct::select(valid, target, ct::full<IndexTile>(0));
  auto target_logit = ct::load_masked(logits + base + safe_target, valid);
  auto loss = ct::log(sum_exp) + max_val - target_logit;
  ct::store_masked(losses + idx, ct::select(valid, loss, ct::full<decltype(loss)>(0.0f)), mask);
  ct::store_masked(counts + idx, ct::select(valid, ct::full<decltype(loss)>(1.0f), ct::full<decltype(loss)>(0.0f)), mask);
}

__tile_global__ void sum_partials_float32_kernel(
    const float* __restrict__ values,
    float* __restrict__ partials,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  values = ct::assume_aligned(values, 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto value_tile = ct::load_masked(values + idx, mask);
  auto zero = ct::full<decltype(value_tile)>(0.0f);
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(partials + out_idx, ct::sum(ct::select(mask, value_tile, zero), 0_ic));
}

__global__ void sum_accumulate_float32_kernel(
    const float* __restrict__ values,
    float* __restrict__ total,
    std::int64_t n) {
  __shared__ float scratch[256];
  const int tid = threadIdx.x;
  const std::int64_t start = static_cast<std::int64_t>(blockIdx.x) * blockDim.x * 2 + tid;
  float sum = 0.0f;
  if (start < n) {
    sum += values[start];
  }
  const std::int64_t second = start + blockDim.x;
  if (second < n) {
    sum += values[second];
  }
  scratch[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(total, scratch[0]);
  }
}

__tile_global__ void kv_cache_read_float32_kernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ cache_k,
    const float* __restrict__ cache_v,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    std::int64_t n,
    std::int64_t heads,
    std::int64_t cache_seq,
    std::int64_t current_seq,
    std::int64_t head_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  k = ct::assume_aligned(k, 16_ic);
  v = ct::assume_aligned(v, 16_ic);
  cache_k = ct::assume_aligned(cache_k, 16_ic);
  cache_v = ct::assume_aligned(cache_v, 16_ic);
  out_k = ct::assume_aligned(out_k, 16_ic);
  out_v = ct::assume_aligned(out_v, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto head_dim_tile = ct::full<IndexTile>(head_dim);
  auto heads_tile = ct::full<IndexTile>(heads);
  auto cache_seq_tile = ct::full<IndexTile>(cache_seq);
  auto total_seq_tile = ct::full<IndexTile>(cache_seq + current_seq);
  auto d = idx % head_dim_tile;
  auto s = (idx / head_dim_tile) % total_seq_tile;
  auto h = (idx / (head_dim_tile * total_seq_tile)) % heads_tile;
  auto b = idx / (head_dim_tile * total_seq_tile * heads_tile);
  auto use_cache = s < cache_seq_tile;
  auto current_s = s - cache_seq_tile;
  auto cache_src = ((b * heads_tile + h) * cache_seq_tile + s) * head_dim_tile + d;
  auto current_src = ((b * heads_tile + h) * ct::full<IndexTile>(current_seq) + current_s) * head_dim_tile + d;
  auto cache_mask = mask & use_cache;
  auto current_mask = mask & (!use_cache);
  auto zero = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto k_value = ct::load_masked(cache_k + cache_src, cache_mask) + ct::load_masked(k + current_src, current_mask);
  auto v_value = ct::load_masked(cache_v + cache_src, cache_mask) + ct::load_masked(v + current_src, current_mask);
  k_value = ct::select(mask, k_value, zero);
  v_value = ct::select(mask, v_value, zero);
  ct::store_masked(out_k + idx, k_value, mask);
  ct::store_masked(out_v + idx, v_value, mask);
}

__tile_global__ void kv_quant_pack_float32_kernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    std::int64_t head_dim) {
  namespace ct = cuda::tiles;

  const int row = ct::bid().x;
  const std::int64_t kv_dim = head_dim * 2;
  const std::int64_t packed_dim = kv_dim + 1;
  const std::int64_t row_base = static_cast<std::int64_t>(row) * head_dim;
  const std::int64_t out_base = static_cast<std::int64_t>(row) * packed_dim;

  float amax = 0.0f;
  for (std::int64_t d = 0; d < head_dim; ++d) {
    const float kval = k[row_base + d];
    const float vval = v[row_base + d];
    const float kabs = kval < 0.0f ? -kval : kval;
    const float vabs = vval < 0.0f ? -vval : vval;
    amax = kabs > amax ? kabs : amax;
    amax = vabs > amax ? vabs : amax;
  }
  const float scale = (amax > 1.0e-7f ? amax : 1.0e-7f) / 127.0f;
  for (std::int64_t d = 0; d < head_dim; ++d) {
    const float raw_k = k[row_base + d] / scale;
    const float raw_v = v[row_base + d] / scale;
    float qk = static_cast<float>(raw_k >= 0.0f ? static_cast<int>(raw_k + 0.5f) : static_cast<int>(raw_k - 0.5f));
    float qv = static_cast<float>(raw_v >= 0.0f ? static_cast<int>(raw_v + 0.5f) : static_cast<int>(raw_v - 0.5f));
    qk = qk > 127.0f ? 127.0f : qk;
    qk = qk < -128.0f ? -128.0f : qk;
    qv = qv > 127.0f ? 127.0f : qv;
    qv = qv < -128.0f ? -128.0f : qv;
    out[out_base + d] = qk;
    out[out_base + head_dim + d] = qv;
  }
  out[out_base + kv_dim] = scale;
}

__tile_global__ void kv_quant_unpack_float32_kernel(
    const float* __restrict__ packed,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    std::int64_t n,
    std::int64_t head_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  packed = ct::assume_aligned(packed, 16_ic);
  out_k = ct::assume_aligned(out_k, 16_ic);
  out_v = ct::assume_aligned(out_v, 16_ic);

  const int bx = ct::bid().x;
  const std::int64_t kv_dim = head_dim * 2;
  const std::int64_t packed_dim = kv_dim + 1;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto head_dim_tile = ct::full<IndexTile>(head_dim);
  auto kv_dim_tile = ct::full<IndexTile>(kv_dim);
  auto d = idx % kv_dim_tile;
  auto row = idx / kv_dim_tile;
  auto scale = ct::load_masked(packed + row * ct::full<IndexTile>(packed_dim) + ct::full<IndexTile>(kv_dim), mask);
  auto quantized = ct::load_masked(packed + row * ct::full<IndexTile>(packed_dim) + d, mask);
  auto value = quantized * scale;
  auto k_mask = mask & (d < head_dim_tile);
  auto v_mask = mask & (!k_mask);
  auto out_row_base = row * head_dim_tile;
  ct::store_masked(out_k + out_row_base + d, value, k_mask);
  ct::store_masked(out_v + out_row_base + (d - head_dim_tile), value, v_mask);
}

__tile_global__ void absolute_position_embedding_float32_kernel(
    const float* __restrict__ weight,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t model_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  weight = ct::assume_aligned(weight, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto dim_tile = ct::full<IndexTile>(model_dim);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto d = idx % dim_tile;
  auto s = (idx / dim_tile) % seq_tile;
  auto src = s * dim_tile + d;
  auto value = ct::load_masked(weight + src, mask);
  ct::store_masked(out + idx, value, mask);
}

__tile_global__ void absolute_position_embedding_backward_float32_kernel(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_weight,
    std::int64_t seq_len,
    std::int64_t model_dim,
    std::int64_t batch) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_weight = ct::assume_aligned(grad_weight, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(seq_len * model_dim);
  auto dim_tile = ct::full<IndexTile>(model_dim);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto s = idx / dim_tile;
  auto d = idx % dim_tile;
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
    auto src = (ct::full<IndexTile>(batch_idx) * seq_tile + s) * dim_tile + d;
    auto value = ct::load_masked(grad_out + src, mask);
    acc = acc + value;
  }
  ct::store_masked(grad_weight + idx, acc, mask);
}

__tile_global__ void absolute_position_embedding_backward_accumulate_float32_kernel(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_weight,
    std::int64_t seq_len,
    std::int64_t model_dim,
    std::int64_t batch) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_weight = ct::assume_aligned(grad_weight, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(seq_len * model_dim);
  auto dim_tile = ct::full<IndexTile>(model_dim);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto s = idx / dim_tile;
  auto d = idx % dim_tile;
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
    auto src = (ct::full<IndexTile>(batch_idx) * seq_tile + s) * dim_tile + d;
    auto value = ct::load_masked(grad_out + src, mask);
    acc = acc + value;
  }
  auto current = ct::load_masked(grad_weight + idx, mask);
  ct::store_masked(grad_weight + idx, current + acc, mask);
}

__tile_global__ void token_embedding_float32_kernel(
    const float* __restrict__ weight,
    const std::int64_t* __restrict__ token_ids,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t model_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  weight = ct::assume_aligned(weight, 16_ic);
  token_ids = ct::assume_aligned(token_ids, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto dim_tile = ct::full<IndexTile>(model_dim);
  auto d = idx % dim_tile;
  auto token_offset = idx / dim_tile;
  auto token = ct::load_masked(token_ids + token_offset, mask);
  auto src = token * dim_tile + d;
  auto value = ct::load_masked(weight + src, mask);
  ct::store_masked(out + idx, value, mask);
}

__tile_global__ void token_embedding_u16_float32_kernel(
    const float* __restrict__ weight,
    const std::uint16_t* __restrict__ token_ids,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t model_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  weight = ct::assume_aligned(weight, 16_ic);
  token_ids = ct::assume_aligned(token_ids, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto dim_tile = ct::full<IndexTile>(model_dim);
  auto d = idx % dim_tile;
  auto token_offset = idx / dim_tile;
  auto token_u16 = ct::load_masked(token_ids + token_offset, mask);
  auto token = ct::element_cast<std::int64_t>(token_u16);
  auto src = token * dim_tile + d;
  auto value = ct::load_masked(weight + src, mask);
  ct::store_masked(out + idx, value, mask);
}

__tile_global__ void token_position_embedding_residual_float32_kernel(
    const float* __restrict__ token_weight,
    const std::int64_t* __restrict__ token_ids,
    const float* __restrict__ position_weight,
    const float* __restrict__ scale,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t model_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  token_weight = ct::assume_aligned(token_weight, 16_ic);
  token_ids = ct::assume_aligned(token_ids, 16_ic);
  position_weight = ct::assume_aligned(position_weight, 16_ic);
  scale = ct::assume_aligned(scale, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto dim_tile = ct::full<IndexTile>(model_dim);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto d = idx % dim_tile;
  auto token_offset = idx / dim_tile;
  auto s = token_offset % seq_tile;
  auto token = ct::load_masked(token_ids + token_offset, mask);
  auto token_value = ct::load_masked(token_weight + token * dim_tile + d, mask);
  auto position_value = ct::load_masked(position_weight + s * dim_tile + d, mask);
  auto scale_tile = ct::full<decltype(token_value)>(*scale);
  ct::store_masked(out + idx, token_value + scale_tile * position_value, mask);
}

__tile_global__ void token_position_embedding_residual_u16_float32_kernel(
    const float* __restrict__ token_weight,
    const std::uint16_t* __restrict__ token_ids,
    const float* __restrict__ position_weight,
    const float* __restrict__ scale,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t model_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  token_weight = ct::assume_aligned(token_weight, 16_ic);
  token_ids = ct::assume_aligned(token_ids, 16_ic);
  position_weight = ct::assume_aligned(position_weight, 16_ic);
  scale = ct::assume_aligned(scale, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto dim_tile = ct::full<IndexTile>(model_dim);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto d = idx % dim_tile;
  auto token_offset = idx / dim_tile;
  auto s = token_offset % seq_tile;
  auto token_u16 = ct::load_masked(token_ids + token_offset, mask);
  auto token = ct::element_cast<std::int64_t>(token_u16);
  auto token_value = ct::load_masked(token_weight + token * dim_tile + d, mask);
  auto position_value = ct::load_masked(position_weight + s * dim_tile + d, mask);
  auto scale_tile = ct::full<decltype(token_value)>(*scale);
  ct::store_masked(out + idx, token_value + scale_tile * position_value, mask);
}

__tile_global__ void token_embedding_backward_weight_float32_kernel(
    const std::int64_t* __restrict__ token_ids,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_weight,
    std::int64_t n,
    std::int64_t model_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  token_ids = ct::assume_aligned(token_ids, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_weight = ct::assume_aligned(grad_weight, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto dim_tile = ct::full<IndexTile>(model_dim);
  auto d = idx % dim_tile;
  auto token_offset = idx / dim_tile;
  auto token = ct::load_masked(token_ids + token_offset, mask);
  auto dst = token * dim_tile + d;
  auto grad = ct::load_masked(grad_out + idx, mask);
  ct::atomic_add_masked<ct::memory_order::relaxed>(grad_weight + dst, grad, mask);
}

__tile_global__ void token_embedding_backward_weight_u16_float32_kernel(
    const std::uint16_t* __restrict__ token_ids,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_weight,
    std::int64_t n,
    std::int64_t model_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  token_ids = ct::assume_aligned(token_ids, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_weight = ct::assume_aligned(grad_weight, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto dim_tile = ct::full<IndexTile>(model_dim);
  auto d = idx % dim_tile;
  auto token_offset = idx / dim_tile;
  auto token_u16 = ct::load_masked(token_ids + token_offset, mask);
  auto token = ct::element_cast<std::int64_t>(token_u16);
  auto dst = token * dim_tile + d;
  auto grad = ct::load_masked(grad_out + idx, mask);
  ct::atomic_add_masked<ct::memory_order::relaxed>(grad_weight + dst, grad, mask);
}

__tile_global__ void rotary_embedding_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ inv_freq,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  inv_freq = ct::assume_aligned(inv_freq, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto head_dim_tile = ct::full<IndexTile>(head_dim);
  auto half_tile = ct::full<IndexTile>(head_dim / 2);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto d = idx % head_dim_tile;
  auto s = (idx / head_dim_tile) % seq_tile;
  auto pair_d = d % half_tile;
  auto first_half = d < half_tile;
  auto base = idx - d;
  auto x1 = ct::load_masked(x + base + pair_d, mask);
  auto x2 = ct::load_masked(x + base + pair_d + half_tile, mask);
  auto inv = ct::load_masked(inv_freq + pair_d, mask);
  auto angle = ct::tile<float, decltype(ct::shape{1024_ic})>(s) * inv;
  auto c = ct::cos(angle);
  auto sn = ct::sin(angle);
  auto first = x1 * c + x2 * sn;
  auto second = -x1 * sn + x2 * c;
  auto value = ct::select(first_half, first, second);
  ct::store_masked(out + idx, value, mask);
}

__tile_global__ void rms_norm_float32_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    std::int64_t rows,
    std::int64_t dim,
    float eps) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int row = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>();
  auto mask = d < ct::full<IndexTile>(dim);
  auto base = ct::full<IndexTile>(static_cast<std::int64_t>(row) * dim);
  auto x_tile = ct::load_masked(x + base + d, mask);
  auto zero = ct::full<decltype(x_tile)>(0.0f);
  auto valid_x = ct::select(mask, x_tile, zero);
  auto sum_sq = ct::sum(valid_x * valid_x, 0_ic);
  auto mean_sq = sum_sq / ct::full<decltype(sum_sq)>(static_cast<float>(dim));
  auto scale = ct::rsqrt(mean_sq + ct::full<decltype(sum_sq)>(eps));
  ct::store_masked(out + base + d, valid_x * scale, mask);
}

__tile_global__ void rms_norm_backward_input_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_x,
    std::int64_t rows,
    std::int64_t dim,
    float eps) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_x = ct::assume_aligned(grad_x, 16_ic);

  const int row = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>();
  auto mask = d < ct::full<IndexTile>(dim);
  auto base = ct::full<IndexTile>(static_cast<std::int64_t>(row) * dim);
  auto x_tile = ct::load_masked(x + base + d, mask);
  auto grad_tile = ct::load_masked(grad_out + base + d, mask);
  auto zero = ct::full<decltype(x_tile)>(0.0f);
  auto valid_x = ct::select(mask, x_tile, zero);
  auto valid_grad = ct::select(mask, grad_tile, zero);
  auto sum_sq = ct::sum(valid_x * valid_x, 0_ic);
  auto dim_f = ct::full<decltype(sum_sq)>(static_cast<float>(dim));
  auto mean_sq = sum_sq / dim_f;
  auto inv_rms = ct::rsqrt(mean_sq + ct::full<decltype(sum_sq)>(eps));
  auto dot = ct::sum(valid_grad * valid_x, 0_ic);
  auto correction = valid_x * (inv_rms * inv_rms * inv_rms) * (dot / dim_f);
  auto grad_x_tile = valid_grad * inv_rms - correction;
  ct::store_masked(grad_x + base + d, grad_x_tile, mask);
}

__tile_global__ void layer_norm_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    std::int64_t rows,
    std::int64_t dim,
    float eps) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  weight = ct::assume_aligned(weight, 16_ic);
  bias = ct::assume_aligned(bias, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int row = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>();
  auto mask = d < ct::full<IndexTile>(dim);
  auto base = ct::full<IndexTile>(static_cast<std::int64_t>(row) * dim);
  auto x_tile = ct::load_masked(x + base + d, mask);
  auto zero = ct::full<decltype(x_tile)>(0.0f);
  auto valid_x = ct::select(mask, x_tile, zero);
  auto sum_x = ct::sum(valid_x, 0_ic);
  auto mean = sum_x / ct::full<decltype(sum_x)>(static_cast<float>(dim));
  auto centered = ct::select(mask, x_tile - mean, zero);
  auto sum_sq = ct::sum(centered * centered, 0_ic);
  auto var = sum_sq / ct::full<decltype(sum_sq)>(static_cast<float>(dim));
  auto scale = ct::rsqrt(var + ct::full<decltype(var)>(eps));
  auto weight_tile = ct::load_masked(weight + d, mask);
  auto bias_tile = ct::load_masked(bias + d, mask);
  ct::store_masked(out + base + d, centered * scale * weight_tile + bias_tile, mask);
}

__tile_global__ void layer_norm_with_stats_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    std::int64_t rows,
    std::int64_t dim,
    float eps) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  weight = ct::assume_aligned(weight, 16_ic);
  bias = ct::assume_aligned(bias, 16_ic);
  out = ct::assume_aligned(out, 16_ic);
  mean_out = ct::assume_aligned(mean_out, 16_ic);
  rstd_out = ct::assume_aligned(rstd_out, 16_ic);

  const int row = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>();
  auto mask = d < ct::full<IndexTile>(dim);
  auto base = ct::full<IndexTile>(static_cast<std::int64_t>(row) * dim);
  auto x_tile = ct::load_masked(x + base + d, mask);
  auto zero = ct::full<decltype(x_tile)>(0.0f);
  auto valid_x = ct::select(mask, x_tile, zero);
  auto sum_x = ct::sum(valid_x, 0_ic);
  auto mean = sum_x / ct::full<decltype(sum_x)>(static_cast<float>(dim));
  auto centered = ct::select(mask, x_tile - mean, zero);
  auto sum_sq = ct::sum(centered * centered, 0_ic);
  auto var = sum_sq / ct::full<decltype(sum_sq)>(static_cast<float>(dim));
  auto scale = ct::rsqrt(var + ct::full<decltype(var)>(eps));
  auto weight_tile = ct::load_masked(weight + d, mask);
  auto bias_tile = ct::load_masked(bias + d, mask);
  ct::store_masked(out + base + d, centered * scale * weight_tile + bias_tile, mask);
  if (row < rows) {
    mean_out[row] = static_cast<float>(mean);
    rstd_out[row] = static_cast<float>(scale);
  }
}

__tile_global__ void layer_norm_with_stats_bf16_out_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    std::uint16_t* __restrict__ out_bf16_bits,
    std::int64_t rows,
    std::int64_t dim,
    float eps) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  weight = ct::assume_aligned(weight, 16_ic);
  bias = ct::assume_aligned(bias, 16_ic);
  out = ct::assume_aligned(out, 16_ic);
  mean_out = ct::assume_aligned(mean_out, 16_ic);
  rstd_out = ct::assume_aligned(rstd_out, 16_ic);
  auto* out_bf16 = ct::assume_aligned(reinterpret_cast<__nv_bfloat16*>(out_bf16_bits), 16_ic);

  const int row = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>();
  auto mask = d < ct::full<IndexTile>(dim);
  auto base = ct::full<IndexTile>(static_cast<std::int64_t>(row) * dim);
  auto x_tile = ct::load_masked(x + base + d, mask);
  auto zero = ct::full<decltype(x_tile)>(0.0f);
  auto valid_x = ct::select(mask, x_tile, zero);
  auto sum_x = ct::sum(valid_x, 0_ic);
  auto mean = sum_x / ct::full<decltype(sum_x)>(static_cast<float>(dim));
  auto centered = ct::select(mask, x_tile - mean, zero);
  auto sum_sq = ct::sum(centered * centered, 0_ic);
  auto var = sum_sq / ct::full<decltype(sum_sq)>(static_cast<float>(dim));
  auto scale = ct::rsqrt(var + ct::full<decltype(var)>(eps));
  auto weight_tile = ct::load_masked(weight + d, mask);
  auto bias_tile = ct::load_masked(bias + d, mask);
  auto norm_value = centered * scale * weight_tile + bias_tile;
  ct::store_masked(out + base + d, norm_value, mask);
  ct::store_masked(out_bf16 + base + d, ct::element_cast<__nv_bfloat16>(norm_value), mask);
  if (row < rows) {
    mean_out[row] = static_cast<float>(mean);
    rstd_out[row] = static_cast<float>(scale);
  }
}

__tile_global__ void layer_norm_apply_stats_bf16_out_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    std::uint16_t* __restrict__ out_bf16_bits,
    std::int64_t rows,
    std::int64_t dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  weight = ct::assume_aligned(weight, 16_ic);
  bias = ct::assume_aligned(bias, 16_ic);
  mean = ct::assume_aligned(mean, 16_ic);
  rstd = ct::assume_aligned(rstd, 16_ic);
  auto* out_bf16 = ct::assume_aligned(reinterpret_cast<__nv_bfloat16*>(out_bf16_bits), 16_ic);

  const int row = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>();
  auto mask = d < ct::full<IndexTile>(dim);
  auto base = ct::full<IndexTile>(static_cast<std::int64_t>(row) * dim);
  auto x_tile = ct::load_masked(x + base + d, mask);
  auto mean_tile = ct::full<decltype(x_tile)>(row < rows ? mean[row] : 0.0f);
  auto rstd_tile = ct::full<decltype(x_tile)>(row < rows ? rstd[row] : 0.0f);
  auto weight_tile = ct::load_masked(weight + d, mask);
  auto bias_tile = ct::load_masked(bias + d, mask);
  auto norm_value = (x_tile - mean_tile) * rstd_tile * weight_tile + bias_tile;
  ct::store_masked(out_bf16 + base + d, ct::element_cast<__nv_bfloat16>(norm_value), mask);
}

__tile_global__ void layer_norm_backward_input_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    float* __restrict__ grad_x,
    std::int64_t rows,
    std::int64_t dim,
    float eps) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  weight = ct::assume_aligned(weight, 16_ic);
  grad_x = ct::assume_aligned(grad_x, 16_ic);

  const int row = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>();
  auto mask = d < ct::full<IndexTile>(dim);
  auto base = ct::full<IndexTile>(static_cast<std::int64_t>(row) * dim);
  auto zero = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto x_tile = ct::load_masked(x + base + d, mask);
  auto valid_x = ct::select(mask, x_tile, zero);
  auto sum_x = ct::sum(valid_x, 0_ic);
  auto dim_f = ct::full<decltype(sum_x)>(static_cast<float>(dim));
  auto mean = sum_x / dim_f;
  auto centered = ct::select(mask, x_tile - mean, zero);
  auto sum_sq = ct::sum(centered * centered, 0_ic);
  auto var = sum_sq / dim_f;
  auto inv_std = ct::rsqrt(var + ct::full<decltype(var)>(eps));
  auto xhat = centered * inv_std;
  auto grad_tile = ct::load_masked(grad_out + base + d, mask);
  auto weight_tile = ct::load_masked(weight + d, mask);
  auto grad_norm = ct::select(mask, grad_tile * weight_tile, zero);
  auto sum_grad = ct::sum(grad_norm, 0_ic);
  auto sum_grad_xhat = ct::sum(grad_norm * xhat, 0_ic);
  auto grad_x_tile = (grad_norm * dim_f - sum_grad - xhat * sum_grad_xhat) * (inv_std / dim_f);
  ct::store_masked(grad_x + base + d, grad_x_tile, mask);
}

__tile_global__ void layer_norm_backward_input_with_stats_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_x,
    std::int64_t rows,
    std::int64_t dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  weight = ct::assume_aligned(weight, 16_ic);
  mean = ct::assume_aligned(mean, 16_ic);
  rstd = ct::assume_aligned(rstd, 16_ic);
  grad_x = ct::assume_aligned(grad_x, 16_ic);

  const int row = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>();
  auto mask = d < ct::full<IndexTile>(dim);
  auto base = ct::full<IndexTile>(static_cast<std::int64_t>(row) * dim);
  auto zero = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto x_tile = ct::load_masked(x + base + d, mask);
  auto mean_tile = ct::full<decltype(zero)>(row < rows ? mean[row] : 0.0f);
  auto inv_std = ct::full<decltype(zero)>(row < rows ? rstd[row] : 0.0f);
  auto centered = ct::select(mask, x_tile - mean_tile, zero);
  auto xhat = centered * inv_std;
  auto grad_tile = ct::load_masked(grad_out + base + d, mask);
  auto weight_tile = ct::load_masked(weight + d, mask);
  auto grad_norm = ct::select(mask, grad_tile * weight_tile, zero);
  auto sum_grad = ct::sum(grad_norm, 0_ic);
  auto sum_grad_xhat = ct::sum(grad_norm * xhat, 0_ic);
  auto dim_f = ct::full<decltype(sum_grad)>(static_cast<float>(dim));
  auto grad_x_tile = (grad_norm * dim_f - sum_grad - xhat * sum_grad_xhat) * (inv_std / dim_f);
  ct::store_masked(grad_x + base + d, grad_x_tile, mask);
}

__tile_global__ void layer_norm_backward_input_residual_add_with_stats_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ residual_grad,
    const float* __restrict__ residual_scale,
    float* __restrict__ out,
    std::int64_t rows,
    std::int64_t dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  weight = ct::assume_aligned(weight, 16_ic);
  mean = ct::assume_aligned(mean, 16_ic);
  rstd = ct::assume_aligned(rstd, 16_ic);
  residual_grad = ct::assume_aligned(residual_grad, 16_ic);
  residual_scale = ct::assume_aligned(residual_scale, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int row = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>();
  auto mask = d < ct::full<IndexTile>(dim);
  auto base = ct::full<IndexTile>(static_cast<std::int64_t>(row) * dim);
  auto zero = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto x_tile = ct::load_masked(x + base + d, mask);
  auto mean_tile = ct::full<decltype(zero)>(row < rows ? mean[row] : 0.0f);
  auto inv_std = ct::full<decltype(zero)>(row < rows ? rstd[row] : 0.0f);
  auto centered = ct::select(mask, x_tile - mean_tile, zero);
  auto xhat = centered * inv_std;
  auto grad_tile = ct::load_masked(grad_out + base + d, mask);
  auto weight_tile = ct::load_masked(weight + d, mask);
  auto grad_norm = ct::select(mask, grad_tile * weight_tile, zero);
  auto sum_grad = ct::sum(grad_norm, 0_ic);
  auto sum_grad_xhat = ct::sum(grad_norm * xhat, 0_ic);
  auto dim_f = ct::full<decltype(sum_grad)>(static_cast<float>(dim));
  auto grad_x_tile = (grad_norm * dim_f - sum_grad - xhat * sum_grad_xhat) * (inv_std / dim_f);
  auto residual_tile = ct::load_masked(residual_grad + base + d, mask);
  auto scale_tile = ct::full<decltype(grad_x_tile)>(*residual_scale);
  ct::store_masked(out + base + d, residual_tile + scale_tile * grad_x_tile, mask);
}

__tile_global__ void layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32_kernel(
    const std::uint16_t* __restrict__ x_bf16_bits,
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ residual_grad,
    const float* __restrict__ residual_scale,
    float* __restrict__ out,
    std::int64_t rows,
    std::int64_t dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const auto* x_bf16 = ct::assume_aligned(reinterpret_cast<const __nv_bfloat16*>(x_bf16_bits), 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  weight = ct::assume_aligned(weight, 16_ic);
  mean = ct::assume_aligned(mean, 16_ic);
  rstd = ct::assume_aligned(rstd, 16_ic);
  residual_grad = ct::assume_aligned(residual_grad, 16_ic);
  residual_scale = ct::assume_aligned(residual_scale, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int row = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>();
  auto mask = d < ct::full<IndexTile>(dim);
  auto base = ct::full<IndexTile>(static_cast<std::int64_t>(row) * dim);
  auto zero = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto x_tile = ct::element_cast<float>(ct::load_masked(x_bf16 + base + d, mask));
  auto mean_tile = ct::full<decltype(zero)>(row < rows ? mean[row] : 0.0f);
  auto inv_std = ct::full<decltype(zero)>(row < rows ? rstd[row] : 0.0f);
  auto centered = ct::select(mask, x_tile - mean_tile, zero);
  auto xhat = centered * inv_std;
  auto grad_tile = ct::load_masked(grad_out + base + d, mask);
  auto weight_tile = ct::load_masked(weight + d, mask);
  auto grad_norm = ct::select(mask, grad_tile * weight_tile, zero);
  auto sum_grad = ct::sum(grad_norm, 0_ic);
  auto sum_grad_xhat = ct::sum(grad_norm * xhat, 0_ic);
  auto dim_f = ct::full<decltype(sum_grad)>(static_cast<float>(dim));
  auto grad_x_tile = (grad_norm * dim_f - sum_grad - xhat * sum_grad_xhat) * (inv_std / dim_f);
  auto residual_tile = ct::load_masked(residual_grad + base + d, mask);
  auto scale_tile = ct::full<decltype(grad_x_tile)>(*residual_scale);
  ct::store_masked(out + base + d, residual_tile + scale_tile * grad_x_tile, mask);
}

__tile_global__ void layer_norm_backward_affine_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    float eps) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_weight = ct::assume_aligned(grad_weight, 16_ic);
  grad_bias = ct::assume_aligned(grad_bias, 16_ic);

  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>();
  auto mask = d < ct::full<IndexTile>(dim);
  auto zero = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto grad_weight_acc = zero;
  auto grad_bias_acc = zero;
  for (std::int64_t row = 0; row < rows; ++row) {
    auto base = ct::full<IndexTile>(row * dim);
    auto x_tile = ct::load_masked(x + base + d, mask);
    auto grad_tile = ct::load_masked(grad_out + base + d, mask);
    auto valid_x = ct::select(mask, x_tile, zero);
    auto valid_grad = ct::select(mask, grad_tile, zero);
    auto sum_x = ct::sum(valid_x, 0_ic);
    auto dim_f = ct::full<decltype(sum_x)>(static_cast<float>(dim));
    auto mean = sum_x / dim_f;
    auto centered = ct::select(mask, x_tile - mean, zero);
    auto sum_sq = ct::sum(centered * centered, 0_ic);
    auto var = sum_sq / dim_f;
    auto inv_std = ct::rsqrt(var + ct::full<decltype(var)>(eps));
    grad_weight_acc = grad_weight_acc + valid_grad * centered * inv_std;
    grad_bias_acc = grad_bias_acc + valid_grad;
  }
  ct::store_masked(grad_weight + d, grad_weight_acc, mask);
  ct::store_masked(grad_bias + d, grad_bias_acc, mask);
}

__tile_global__ void layer_norm_backward_affine_accumulate_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    float eps) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_weight = ct::assume_aligned(grad_weight, 16_ic);
  grad_bias = ct::assume_aligned(grad_bias, 16_ic);

  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>();
  auto mask = d < ct::full<IndexTile>(dim);
  auto zero = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto grad_weight_acc = zero;
  auto grad_bias_acc = zero;
  for (std::int64_t row = 0; row < rows; ++row) {
    auto base = ct::full<IndexTile>(row * dim);
    auto x_tile = ct::load_masked(x + base + d, mask);
    auto grad_tile = ct::load_masked(grad_out + base + d, mask);
    auto valid_x = ct::select(mask, x_tile, zero);
    auto valid_grad = ct::select(mask, grad_tile, zero);
    auto sum_x = ct::sum(valid_x, 0_ic);
    auto dim_f = ct::full<decltype(sum_x)>(static_cast<float>(dim));
    auto mean = sum_x / dim_f;
    auto centered = ct::select(mask, x_tile - mean, zero);
    auto sum_sq = ct::sum(centered * centered, 0_ic);
    auto var = sum_sq / dim_f;
    auto inv_std = ct::rsqrt(var + ct::full<decltype(var)>(eps));
    grad_weight_acc = grad_weight_acc + valid_grad * centered * inv_std;
    grad_bias_acc = grad_bias_acc + valid_grad;
  }
  auto current_weight = ct::load_masked(grad_weight + d, mask);
  auto current_bias = ct::load_masked(grad_bias + d, mask);
  ct::store_masked(grad_weight + d, current_weight + grad_weight_acc, mask);
  ct::store_masked(grad_bias + d, current_bias + grad_bias_acc, mask);
}

__tile_global__ void layer_norm_backward_affine_accumulate_with_stats_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_out,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    std::int64_t rows,
    std::int64_t dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  mean = ct::assume_aligned(mean, 16_ic);
  rstd = ct::assume_aligned(rstd, 16_ic);
  grad_weight = ct::assume_aligned(grad_weight, 16_ic);
  grad_bias = ct::assume_aligned(grad_bias, 16_ic);

  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>();
  auto mask = d < ct::full<IndexTile>(dim);
  auto zero = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto grad_weight_acc = zero;
  auto grad_bias_acc = zero;
  for (std::int64_t row = 0; row < rows; ++row) {
    auto base = ct::full<IndexTile>(row * dim);
    auto x_tile = ct::load_masked(x + base + d, mask);
    auto grad_tile = ct::load_masked(grad_out + base + d, mask);
    auto valid_grad = ct::select(mask, grad_tile, zero);
    auto mean_tile = ct::full<decltype(zero)>(mean[row]);
    auto inv_std = ct::full<decltype(zero)>(rstd[row]);
    auto centered = ct::select(mask, x_tile - mean_tile, zero);
    grad_weight_acc = grad_weight_acc + valid_grad * centered * inv_std;
    grad_bias_acc = grad_bias_acc + valid_grad;
  }
  auto current_weight = ct::load_masked(grad_weight + d, mask);
  auto current_bias = ct::load_masked(grad_bias + d, mask);
  ct::store_masked(grad_weight + d, current_weight + grad_weight_acc, mask);
  ct::store_masked(grad_bias + d, current_bias + grad_bias_acc, mask);
}

__tile_global__ void layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32_kernel(
    const std::uint16_t* __restrict__ x_bf16_bits,
    const float* __restrict__ grad_out,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    std::int64_t rows,
    std::int64_t dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const auto* x_bf16 = ct::assume_aligned(reinterpret_cast<const __nv_bfloat16*>(x_bf16_bits), 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  mean = ct::assume_aligned(mean, 16_ic);
  rstd = ct::assume_aligned(rstd, 16_ic);
  grad_weight = ct::assume_aligned(grad_weight, 16_ic);
  grad_bias = ct::assume_aligned(grad_bias, 16_ic);

  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>();
  auto mask = d < ct::full<IndexTile>(dim);
  auto zero = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto grad_weight_acc = zero;
  auto grad_bias_acc = zero;
  for (std::int64_t row = 0; row < rows; ++row) {
    auto base = ct::full<IndexTile>(row * dim);
    auto x_tile = ct::element_cast<float>(ct::load_masked(x_bf16 + base + d, mask));
    auto grad_tile = ct::load_masked(grad_out + base + d, mask);
    auto valid_grad = ct::select(mask, grad_tile, zero);
    auto mean_tile = ct::full<decltype(zero)>(mean[row]);
    auto inv_std = ct::full<decltype(zero)>(rstd[row]);
    auto centered = ct::select(mask, x_tile - mean_tile, zero);
    grad_weight_acc = grad_weight_acc + valid_grad * centered * inv_std;
    grad_bias_acc = grad_bias_acc + valid_grad;
  }
  auto current_weight = ct::load_masked(grad_weight + d, mask);
  auto current_bias = ct::load_masked(grad_bias + d, mask);
  ct::store_masked(grad_weight + d, current_weight + grad_weight_acc, mask);
  ct::store_masked(grad_bias + d, current_bias + grad_bias_acc, mask);
}

__tile_global__ void layer_norm_backward_affine_chunked_atomic_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    std::int64_t row_chunk_size,
    std::int64_t row_chunks) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_weight = ct::assume_aligned(grad_weight, 16_ic);
  grad_bias = ct::assume_aligned(grad_bias, 16_ic);

  const std::int64_t dim_block = static_cast<std::int64_t>(ct::bid().x);
  const std::int64_t row_chunk = static_cast<std::int64_t>(ct::bid().y);
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>() + ct::full<IndexTile>(dim_block * kTileSize);
  auto mask = d < ct::full<IndexTile>(dim);
  auto zero = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto grad_weight_acc = zero;
  auto grad_bias_acc = zero;
  const std::int64_t row_start = row_chunk * row_chunk_size;
  const std::int64_t row_end = (row_start + row_chunk_size < rows) ? row_start + row_chunk_size : rows;
  for (std::int64_t row = row_start; row < row_end; ++row) {
    auto base = ct::full<IndexTile>(row * dim);
    auto x_tile = ct::load_masked(x + base + d, mask);
    auto grad_tile = ct::load_masked(grad_out + base + d, mask);
    auto valid_x = ct::select(mask, x_tile, zero);
    auto valid_grad = ct::select(mask, grad_tile, zero);
    auto sum_x = ct::sum(valid_x, 0_ic);
    auto dim_f = ct::full<decltype(sum_x)>(static_cast<float>(dim));
    auto mean = sum_x / dim_f;
    auto centered = ct::select(mask, x_tile - mean, zero);
    auto sum_sq = ct::sum(centered * centered, 0_ic);
    auto var = sum_sq / dim_f;
    auto inv_std = ct::rsqrt(var + ct::full<decltype(var)>(eps));
    grad_weight_acc = grad_weight_acc + valid_grad * centered * inv_std;
    grad_bias_acc = grad_bias_acc + valid_grad;
  }
  auto active = row_chunk < row_chunks ? mask : ct::full<decltype(mask)>(false);
  ct::atomic_add_masked<ct::memory_order::relaxed>(grad_weight + d, grad_weight_acc, active);
  ct::atomic_add_masked<ct::memory_order::relaxed>(grad_bias + d, grad_bias_acc, active);
}

__tile_global__ void layer_norm_backward_affine_chunked_atomic_with_stats_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_out,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    std::int64_t row_chunk_size,
    std::int64_t row_chunks) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  mean = ct::assume_aligned(mean, 16_ic);
  rstd = ct::assume_aligned(rstd, 16_ic);
  grad_weight = ct::assume_aligned(grad_weight, 16_ic);
  grad_bias = ct::assume_aligned(grad_bias, 16_ic);

  const std::int64_t dim_block = static_cast<std::int64_t>(ct::bid().x);
  const std::int64_t row_chunk = static_cast<std::int64_t>(ct::bid().y);
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>() + ct::full<IndexTile>(dim_block * kTileSize);
  auto mask = d < ct::full<IndexTile>(dim);
  auto zero = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto grad_weight_acc = zero;
  auto grad_bias_acc = zero;
  const std::int64_t row_start = row_chunk * row_chunk_size;
  const std::int64_t row_end = (row_start + row_chunk_size < rows) ? row_start + row_chunk_size : rows;
  for (std::int64_t row = row_start; row < row_end; ++row) {
    auto base = ct::full<IndexTile>(row * dim);
    auto x_tile = ct::load_masked(x + base + d, mask);
    auto grad_tile = ct::load_masked(grad_out + base + d, mask);
    auto valid_grad = ct::select(mask, grad_tile, zero);
    auto mean_tile = ct::full<decltype(zero)>(mean[row]);
    auto inv_std = ct::full<decltype(zero)>(rstd[row]);
    auto centered = ct::select(mask, x_tile - mean_tile, zero);
    grad_weight_acc = grad_weight_acc + valid_grad * centered * inv_std;
    grad_bias_acc = grad_bias_acc + valid_grad;
  }
  auto active = row_chunk < row_chunks ? mask : ct::full<decltype(mask)>(false);
  ct::atomic_add_masked<ct::memory_order::relaxed>(grad_weight + d, grad_weight_acc, active);
  ct::atomic_add_masked<ct::memory_order::relaxed>(grad_bias + d, grad_bias_acc, active);
}

__tile_global__ void layer_norm_backward_affine_chunked_atomic_with_stats_bf16_bits_float32_kernel(
    const std::uint16_t* __restrict__ x_bf16_bits,
    const float* __restrict__ grad_out,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    std::int64_t row_chunk_size,
    std::int64_t row_chunks) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const auto* x_bf16 = ct::assume_aligned(reinterpret_cast<const __nv_bfloat16*>(x_bf16_bits), 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  mean = ct::assume_aligned(mean, 16_ic);
  rstd = ct::assume_aligned(rstd, 16_ic);
  grad_weight = ct::assume_aligned(grad_weight, 16_ic);
  grad_bias = ct::assume_aligned(grad_bias, 16_ic);

  const std::int64_t dim_block = static_cast<std::int64_t>(ct::bid().x);
  const std::int64_t row_chunk = static_cast<std::int64_t>(ct::bid().y);
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>() + ct::full<IndexTile>(dim_block * kTileSize);
  auto mask = d < ct::full<IndexTile>(dim);
  auto zero = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto grad_weight_acc = zero;
  auto grad_bias_acc = zero;
  const std::int64_t row_start = row_chunk * row_chunk_size;
  const std::int64_t row_end = (row_start + row_chunk_size < rows) ? row_start + row_chunk_size : rows;
  for (std::int64_t row = row_start; row < row_end; ++row) {
    auto base = ct::full<IndexTile>(row * dim);
    auto x_tile = ct::element_cast<float>(ct::load_masked(x_bf16 + base + d, mask));
    auto grad_tile = ct::load_masked(grad_out + base + d, mask);
    auto valid_grad = ct::select(mask, grad_tile, zero);
    auto mean_tile = ct::full<decltype(zero)>(mean[row]);
    auto inv_std = ct::full<decltype(zero)>(rstd[row]);
    auto centered = ct::select(mask, x_tile - mean_tile, zero);
    grad_weight_acc = grad_weight_acc + valid_grad * centered * inv_std;
    grad_bias_acc = grad_bias_acc + valid_grad;
  }
  auto active = row_chunk < row_chunks ? mask : ct::full<decltype(mask)>(false);
  ct::atomic_add_masked<ct::memory_order::relaxed>(grad_weight + d, grad_weight_acc, active);
  ct::atomic_add_masked<ct::memory_order::relaxed>(grad_bias + d, grad_bias_acc, active);
}

__tile_global__ void layer_norm_backward_affine_residual_add_chunked_atomic_with_stats_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ residual_grad,
    const float* __restrict__ residual_scale,
    float* __restrict__ out,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    std::int64_t row_chunk_size,
    std::int64_t row_chunks) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  weight = ct::assume_aligned(weight, 16_ic);
  mean = ct::assume_aligned(mean, 16_ic);
  rstd = ct::assume_aligned(rstd, 16_ic);
  residual_grad = ct::assume_aligned(residual_grad, 16_ic);
  residual_scale = ct::assume_aligned(residual_scale, 16_ic);
  out = ct::assume_aligned(out, 16_ic);
  grad_weight = ct::assume_aligned(grad_weight, 16_ic);
  grad_bias = ct::assume_aligned(grad_bias, 16_ic);

  const std::int64_t dim_block = static_cast<std::int64_t>(ct::bid().x);
  const std::int64_t row_chunk = static_cast<std::int64_t>(ct::bid().y);
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>() + ct::full<IndexTile>(dim_block * kTileSize);
  auto mask = d < ct::full<IndexTile>(dim);
  auto zero = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto weight_tile = ct::load_masked(weight + d, mask);
  auto scale_tile = ct::full<decltype(zero)>(*residual_scale);
  auto dim_f = ct::full<decltype(zero)>(static_cast<float>(dim));
  auto grad_weight_acc = zero;
  auto grad_bias_acc = zero;
  const std::int64_t row_start = row_chunk * row_chunk_size;
  const std::int64_t row_end = (row_start + row_chunk_size < rows) ? row_start + row_chunk_size : rows;
  for (std::int64_t row = row_start; row < row_end; ++row) {
    auto base = ct::full<IndexTile>(row * dim);
    auto x_tile = ct::load_masked(x + base + d, mask);
    auto grad_tile = ct::load_masked(grad_out + base + d, mask);
    auto valid_grad = ct::select(mask, grad_tile, zero);
    auto mean_tile = ct::full<decltype(zero)>(mean[row]);
    auto inv_std = ct::full<decltype(zero)>(rstd[row]);
    auto centered = ct::select(mask, x_tile - mean_tile, zero);
    auto xhat = centered * inv_std;
    grad_weight_acc = grad_weight_acc + valid_grad * xhat;
    grad_bias_acc = grad_bias_acc + valid_grad;
    auto grad_norm = ct::select(mask, grad_tile * weight_tile, zero);
    auto sum_grad = ct::sum(grad_norm, 0_ic);
    auto sum_grad_xhat = ct::sum(grad_norm * xhat, 0_ic);
    auto grad_x_tile = (grad_norm * dim_f - sum_grad - xhat * sum_grad_xhat) * (inv_std / dim_f);
    auto residual_tile = ct::load_masked(residual_grad + base + d, mask);
    ct::store_masked(out + base + d, residual_tile + scale_tile * grad_x_tile, mask);
  }
  auto active = row_chunk < row_chunks ? mask : ct::full<decltype(mask)>(false);
  ct::atomic_add_masked<ct::memory_order::relaxed>(grad_weight + d, grad_weight_acc, active);
  ct::atomic_add_masked<ct::memory_order::relaxed>(grad_bias + d, grad_bias_acc, active);
}

__tile_global__ void layer_norm_backward_affine_residual_add_chunked_atomic_with_stats_bf16_bits_float32_kernel(
    const std::uint16_t* __restrict__ x_bf16_bits,
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ residual_grad,
    const float* __restrict__ residual_scale,
    float* __restrict__ out,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    std::int64_t row_chunk_size,
    std::int64_t row_chunks) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const auto* x_bf16 = ct::assume_aligned(reinterpret_cast<const __nv_bfloat16*>(x_bf16_bits), 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  weight = ct::assume_aligned(weight, 16_ic);
  mean = ct::assume_aligned(mean, 16_ic);
  rstd = ct::assume_aligned(rstd, 16_ic);
  residual_grad = ct::assume_aligned(residual_grad, 16_ic);
  residual_scale = ct::assume_aligned(residual_scale, 16_ic);
  out = ct::assume_aligned(out, 16_ic);
  grad_weight = ct::assume_aligned(grad_weight, 16_ic);
  grad_bias = ct::assume_aligned(grad_bias, 16_ic);

  const std::int64_t dim_block = static_cast<std::int64_t>(ct::bid().x);
  const std::int64_t row_chunk = static_cast<std::int64_t>(ct::bid().y);
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>() + ct::full<IndexTile>(dim_block * kTileSize);
  auto mask = d < ct::full<IndexTile>(dim);
  auto zero = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto weight_tile = ct::load_masked(weight + d, mask);
  auto scale_tile = ct::full<decltype(zero)>(*residual_scale);
  auto dim_f = ct::full<decltype(zero)>(static_cast<float>(dim));
  auto grad_weight_acc = zero;
  auto grad_bias_acc = zero;
  const std::int64_t row_start = row_chunk * row_chunk_size;
  const std::int64_t row_end = (row_start + row_chunk_size < rows) ? row_start + row_chunk_size : rows;
  for (std::int64_t row = row_start; row < row_end; ++row) {
    auto base = ct::full<IndexTile>(row * dim);
    auto x_tile = ct::element_cast<float>(ct::load_masked(x_bf16 + base + d, mask));
    auto grad_tile = ct::load_masked(grad_out + base + d, mask);
    auto valid_grad = ct::select(mask, grad_tile, zero);
    auto mean_tile = ct::full<decltype(zero)>(mean[row]);
    auto inv_std = ct::full<decltype(zero)>(rstd[row]);
    auto centered = ct::select(mask, x_tile - mean_tile, zero);
    auto xhat = centered * inv_std;
    grad_weight_acc = grad_weight_acc + valid_grad * xhat;
    grad_bias_acc = grad_bias_acc + valid_grad;
    auto grad_norm = ct::select(mask, grad_tile * weight_tile, zero);
    auto sum_grad = ct::sum(grad_norm, 0_ic);
    auto sum_grad_xhat = ct::sum(grad_norm * xhat, 0_ic);
    auto grad_x_tile = (grad_norm * dim_f - sum_grad - xhat * sum_grad_xhat) * (inv_std / dim_f);
    auto residual_tile = ct::load_masked(residual_grad + base + d, mask);
    ct::store_masked(out + base + d, residual_tile + scale_tile * grad_x_tile, mask);
  }
  auto active = row_chunk < row_chunks ? mask : ct::full<decltype(mask)>(false);
  ct::atomic_add_masked<ct::memory_order::relaxed>(grad_weight + d, grad_weight_acc, active);
  ct::atomic_add_masked<ct::memory_order::relaxed>(grad_bias + d, grad_bias_acc, active);
}

__tile_global__ void softmax_lastdim_float32_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    std::int64_t rows,
    std::int64_t dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int row = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>();
  auto mask = d < ct::full<IndexTile>(dim);
  auto base = ct::full<IndexTile>(static_cast<std::int64_t>(row) * dim);
  auto x_tile = ct::load_masked(x + base + d, mask);
  auto neg_inf = ct::full<decltype(x_tile)>(-3.4028234663852886e38f);
  auto safe_x = ct::select(mask, x_tile, neg_inf);
  auto max_val = ct::reduce_max(safe_x, 0_ic);
  auto exp_x = ct::select(mask, ct::exp(x_tile - max_val), ct::full<decltype(x_tile)>(0.0f));
  auto denom = ct::sum(exp_x, 0_ic);
  ct::store_masked(out + base + d, exp_x / denom, mask);
}

__tile_global__ void semantic_hash_int64_kernel(
    const float* __restrict__ sem_vec,
    const float* __restrict__ proj,
    std::int64_t* __restrict__ out,
    std::int64_t batch,
    std::int64_t dim,
    std::int64_t tables,
    std::int64_t planes) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  sem_vec = ct::assume_aligned(sem_vec, 16_ic);
  proj = ct::assume_aligned(proj, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const std::int64_t job = static_cast<std::int64_t>(ct::bid().x);
  const std::int64_t row = job / tables;
  const std::int64_t table = job % tables;
  std::int64_t hash = 0;
  for (std::int64_t plane = 0; plane < planes; ++plane) {
    float dot = 0.0f;
    const std::int64_t proj_base = (table * planes + plane) * dim;
    const std::int64_t row_base = row * dim;
    for (std::int64_t d = 0; d < dim; ++d) {
      dot += sem_vec[row_base + d] * proj[proj_base + d];
    }
    if (dot > 0.0f) {
      hash |= (static_cast<std::int64_t>(1) << plane);
    }
  }
  out[row * tables + table] = hash;
}

__global__ void topk_route_float32_kernel(
    const float* __restrict__ logits,
    float* __restrict__ weights,
    std::int64_t* __restrict__ indices,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t top_k) {
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  float top_vals[64];
  std::int64_t top_idx[64];
  for (std::int64_t k = 0; k < top_k; ++k) {
    top_vals[k] = -3.4028234663852886e38f;
    top_idx[k] = 0;
  }
  const std::int64_t base = row * experts;
  for (std::int64_t expert = 0; expert < experts; ++expert) {
    const float value = logits[base + expert];
    for (std::int64_t slot = 0; slot < top_k; ++slot) {
      if (value > top_vals[slot]) {
        for (std::int64_t move = top_k - 1; move > slot; --move) {
          top_vals[move] = top_vals[move - 1];
          top_idx[move] = top_idx[move - 1];
        }
        top_vals[slot] = value;
        top_idx[slot] = expert;
        break;
      }
    }
  }
  const float max_val = top_vals[0];
  float denom = 0.0f;
  for (std::int64_t k = 0; k < top_k; ++k) {
    denom += expf(top_vals[k] - max_val);
  }
  const std::int64_t out_base = row * top_k;
  for (std::int64_t k = 0; k < top_k; ++k) {
    weights[out_base + k] = expf(top_vals[k] - max_val) / denom;
    indices[out_base + k] = top_idx[k];
  }
}

__tile_global__ void attentionless_decoder_float32_kernel(
    const std::int64_t* __restrict__ bucket_indices,
    const float* __restrict__ expert_output,
    const float* __restrict__ bucket_embed,
    const float* __restrict__ out_weight,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t residual_dim,
    std::int64_t vocab_size,
    std::int64_t n_buckets) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  bucket_indices = ct::assume_aligned(bucket_indices, 16_ic);
  expert_output = ct::assume_aligned(expert_output, 16_ic);
  bucket_embed = ct::assume_aligned(bucket_embed, 16_ic);
  out_weight = ct::assume_aligned(out_weight, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto vocab_tile = ct::full<IndexTile>(vocab_size);
  auto residual_tile = ct::full<IndexTile>(residual_dim);
  auto out_col = idx % vocab_tile;
  auto row = idx / vocab_tile;
  auto raw_bucket = ct::load_masked(bucket_indices + row, mask);
  auto bucket = raw_bucket % ct::full<IndexTile>(n_buckets);
  bucket = ct::select(bucket < ct::full<IndexTile>(0), bucket + ct::full<IndexTile>(n_buckets), bucket);
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t r = 0; r < residual_dim; ++r) {
    auto r_tile = ct::full<IndexTile>(r);
    auto expert_val = ct::load_masked(expert_output + row * residual_tile + r_tile, mask);
    auto bucket_val = ct::load_masked(bucket_embed + bucket * residual_tile + r_tile, mask);
    auto weight_val = ct::load_masked(out_weight + out_col * residual_tile + r_tile, mask);
    acc = acc + (expert_val + bucket_val) * weight_val;
  }
  ct::store_masked(out + idx, acc, mask);
}

__tile_global__ void expert_bias_add_float32_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t experts) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logits = ct::assume_aligned(logits, 16_ic);
  bias = ct::assume_aligned(bias, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto expert = idx % ct::full<IndexTile>(experts);
  auto value = ct::load_masked(logits + idx, mask);
  auto bias_value = ct::load_masked(bias + expert, mask);
  ct::store_masked(out + idx, value + bias_value, mask);
}

__tile_global__ void group_norm_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    std::int64_t seq_len,
    std::int64_t dim,
    std::int64_t num_groups,
    float eps) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  weight = ct::assume_aligned(weight, 16_ic);
  bias = ct::assume_aligned(bias, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bg = ct::bid().x;
  const std::int64_t b_scalar = bg / num_groups;
  const std::int64_t g_scalar = bg % num_groups;
  const std::int64_t group_dim = dim / num_groups;
  const std::int64_t group_elems = seq_len * group_dim;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto e = ct::iota<IndexTile>();
  auto mask = e < ct::full<IndexTile>(group_elems);
  auto group_dim_tile = ct::full<IndexTile>(group_dim);
  auto seq = e / group_dim_tile;
  auto c = e % group_dim_tile;
  auto d = ct::full<IndexTile>(g_scalar * group_dim) + c;
  auto src = (ct::full<IndexTile>(b_scalar) * ct::full<IndexTile>(seq_len) + seq) * ct::full<IndexTile>(dim) + d;
  auto x_tile = ct::load_masked(x + src, mask);
  auto zero = ct::full<decltype(x_tile)>(0.0f);
  auto valid_x = ct::select(mask, x_tile, zero);
  auto sum_x = ct::sum(valid_x, 0_ic);
  auto mean = sum_x / ct::full<decltype(sum_x)>(static_cast<float>(group_elems));
  auto centered = ct::select(mask, x_tile - mean, zero);
  auto sum_sq = ct::sum(centered * centered, 0_ic);
  auto var = sum_sq / ct::full<decltype(sum_sq)>(static_cast<float>(group_elems));
  auto scale = ct::rsqrt(var + ct::full<decltype(var)>(eps));
  auto weight_tile = ct::load_masked(weight + d, mask);
  auto bias_tile = ct::load_masked(bias + d, mask);
  ct::store_masked(out + src, centered * scale * weight_tile + bias_tile, mask);
}

__tile_global__ void scaled_residual_add_float32_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    const float* __restrict__ scale,
    float* __restrict__ out,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  lhs = ct::assume_aligned(lhs, 16_ic);
  rhs = ct::assume_aligned(rhs, 16_ic);
  scale = ct::assume_aligned(scale, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  auto lhs_tile = ct::partition_view{ct::tensor_span{lhs, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto rhs_tile = ct::partition_view{ct::tensor_span{rhs, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto scale_tile = ct::full<decltype(lhs_tile)>(*scale);
  auto result = lhs_tile + scale_tile * rhs_tile;
  ct::partition_view{ct::tensor_span{out, ct::extents{n}}, ct::shape{1024_ic}}.store_masked(result, bx);
}

__tile_global__ void split_qkv_float32_kernel(
    const float* __restrict__ qkv,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    std::int64_t rows,
    std::int64_t dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  qkv = ct::assume_aligned(qkv, 16_ic);
  q = ct::assume_aligned(q, 16_ic);
  k = ct::assume_aligned(k, 16_ic);
  v = ct::assume_aligned(v, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto local = ct::iota<IndexTile>();
  auto flat = ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize) + local;
  const std::int64_t n = rows * dim;
  auto mask = flat < ct::full<IndexTile>(n);
  auto dim_tile = ct::full<IndexTile>(dim);
  auto row = flat / dim_tile;
  auto col = flat % dim_tile;
  auto qkv_base = row * ct::full<IndexTile>(3 * dim) + col;
  auto q_tile = ct::load_masked(qkv + qkv_base, mask);
  auto k_tile = ct::load_masked(qkv + qkv_base + dim_tile, mask);
  auto v_tile = ct::load_masked(qkv + qkv_base + ct::full<IndexTile>(2 * dim), mask);
  ct::store_masked(q + flat, q_tile, mask);
  ct::store_masked(k + flat, k_tile, mask);
  ct::store_masked(v + flat, v_tile, mask);
}

__tile_global__ void split_qkv_to_heads_float32_kernel(
    const float* __restrict__ qkv,
    float* __restrict__ q_heads,
    float* __restrict__ k_heads,
    float* __restrict__ v_heads,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  qkv = ct::assume_aligned(qkv, 16_ic);
  q_heads = ct::assume_aligned(q_heads, 16_ic);
  k_heads = ct::assume_aligned(k_heads, 16_ic);
  v_heads = ct::assume_aligned(v_heads, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto local = ct::iota<IndexTile>();
  auto flat = ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize) + local;
  const std::int64_t dim = heads * head_dim;
  const std::int64_t n = batch * seq_len * dim;
  auto mask = flat < ct::full<IndexTile>(n);
  auto head_dim_tile = ct::full<IndexTile>(head_dim);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto heads_tile = ct::full<IndexTile>(heads);
  auto dim_tile = ct::full<IndexTile>(dim);
  auto d = flat % head_dim_tile;
  auto h = (flat / head_dim_tile) % heads_tile;
  auto s = (flat / (head_dim_tile * heads_tile)) % seq_tile;
  auto b = flat / (head_dim_tile * heads_tile * seq_tile);
  auto row = b * seq_tile + s;
  auto col = h * head_dim_tile + d;
  auto qkv_base = row * ct::full<IndexTile>(3 * dim) + col;
  auto dst = ((b * heads_tile + h) * seq_tile + s) * head_dim_tile + d;
  auto q_tile = ct::load_masked(qkv + qkv_base, mask);
  auto k_tile = ct::load_masked(qkv + qkv_base + dim_tile, mask);
  auto v_tile = ct::load_masked(qkv + qkv_base + ct::full<IndexTile>(2 * dim), mask);
  ct::store_masked(q_heads + dst, q_tile, mask);
  ct::store_masked(k_heads + dst, k_tile, mask);
  ct::store_masked(v_heads + dst, v_tile, mask);
}

__tile_global__ void split_qkv_to_heads_add_bias_float32_kernel(
    const float* __restrict__ qkv,
    const float* __restrict__ bias,
    float* __restrict__ q_heads,
    float* __restrict__ k_heads,
    float* __restrict__ v_heads,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  qkv = ct::assume_aligned(qkv, 16_ic);
  bias = ct::assume_aligned(bias, 16_ic);
  q_heads = ct::assume_aligned(q_heads, 16_ic);
  k_heads = ct::assume_aligned(k_heads, 16_ic);
  v_heads = ct::assume_aligned(v_heads, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto local = ct::iota<IndexTile>();
  auto flat = ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize) + local;
  const std::int64_t dim = heads * head_dim;
  const std::int64_t n = batch * seq_len * dim;
  auto mask = flat < ct::full<IndexTile>(n);
  auto head_dim_tile = ct::full<IndexTile>(head_dim);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto heads_tile = ct::full<IndexTile>(heads);
  auto dim_tile = ct::full<IndexTile>(dim);
  auto d = flat % head_dim_tile;
  auto h = (flat / head_dim_tile) % heads_tile;
  auto s = (flat / (head_dim_tile * heads_tile)) % seq_tile;
  auto b = flat / (head_dim_tile * heads_tile * seq_tile);
  auto row = b * seq_tile + s;
  auto col = h * head_dim_tile + d;
  auto qkv_base = row * ct::full<IndexTile>(3 * dim) + col;
  auto dst = ((b * heads_tile + h) * seq_tile + s) * head_dim_tile + d;
  auto q_tile = ct::load_masked(qkv + qkv_base, mask) + ct::load_masked(bias + col, mask);
  auto k_tile = ct::load_masked(qkv + qkv_base + dim_tile, mask) + ct::load_masked(bias + dim_tile + col, mask);
  auto v_tile = ct::load_masked(qkv + qkv_base + ct::full<IndexTile>(2 * dim), mask) +
                ct::load_masked(bias + ct::full<IndexTile>(2 * dim) + col, mask);
  ct::store_masked(q_heads + dst, q_tile, mask);
  ct::store_masked(k_heads + dst, k_tile, mask);
  ct::store_masked(v_heads + dst, v_tile, mask);
}

__tile_global__ void merge_qkv_float32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ qkv,
    std::int64_t rows,
    std::int64_t dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  q = ct::assume_aligned(q, 16_ic);
  k = ct::assume_aligned(k, 16_ic);
  v = ct::assume_aligned(v, 16_ic);
  qkv = ct::assume_aligned(qkv, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto local = ct::iota<IndexTile>();
  auto flat = ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize) + local;
  const std::int64_t n = rows * dim;
  auto mask = flat < ct::full<IndexTile>(n);
  auto dim_tile = ct::full<IndexTile>(dim);
  auto row = flat / dim_tile;
  auto col = flat % dim_tile;
  auto qkv_base = row * ct::full<IndexTile>(3 * dim) + col;
  auto q_tile = ct::load_masked(q + flat, mask);
  auto k_tile = ct::load_masked(k + flat, mask);
  auto v_tile = ct::load_masked(v + flat, mask);
  ct::store_masked(qkv + qkv_base, q_tile, mask);
  ct::store_masked(qkv + qkv_base + dim_tile, k_tile, mask);
  ct::store_masked(qkv + qkv_base + ct::full<IndexTile>(2 * dim), v_tile, mask);
}

__tile_global__ void merge_heads_to_qkv_float32_kernel(
    const float* __restrict__ q_heads,
    const float* __restrict__ k_heads,
    const float* __restrict__ v_heads,
    float* __restrict__ qkv,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  q_heads = ct::assume_aligned(q_heads, 16_ic);
  k_heads = ct::assume_aligned(k_heads, 16_ic);
  v_heads = ct::assume_aligned(v_heads, 16_ic);
  qkv = ct::assume_aligned(qkv, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto local = ct::iota<IndexTile>();
  auto flat = ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize) + local;
  const std::int64_t dim = heads * head_dim;
  const std::int64_t n = batch * seq_len * dim;
  auto mask = flat < ct::full<IndexTile>(n);
  auto head_dim_tile = ct::full<IndexTile>(head_dim);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto heads_tile = ct::full<IndexTile>(heads);
  auto dim_tile = ct::full<IndexTile>(dim);
  auto d = flat % head_dim_tile;
  auto h = (flat / head_dim_tile) % heads_tile;
  auto s = (flat / (head_dim_tile * heads_tile)) % seq_tile;
  auto b = flat / (head_dim_tile * heads_tile * seq_tile);
  auto src = ((b * heads_tile + h) * seq_tile + s) * head_dim_tile + d;
  auto row = b * seq_tile + s;
  auto col = h * head_dim_tile + d;
  auto qkv_base = row * ct::full<IndexTile>(3 * dim) + col;
  auto q_tile = ct::load_masked(q_heads + src, mask);
  auto k_tile = ct::load_masked(k_heads + src, mask);
  auto v_tile = ct::load_masked(v_heads + src, mask);
  ct::store_masked(qkv + qkv_base, q_tile, mask);
  ct::store_masked(qkv + qkv_base + dim_tile, k_tile, mask);
  ct::store_masked(qkv + qkv_base + ct::full<IndexTile>(2 * dim), v_tile, mask);
}

__tile_global__ void linear_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  weight = ct::assume_aligned(weight, 16_ic);
  out = ct::assume_aligned(out, 16_ic);
  if (has_bias) {
    bias = ct::assume_aligned(bias, 16_ic);
  }

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto output_dim_tile = ct::full<IndexTile>(output_dim);
  auto out_col = idx % output_dim_tile;
  auto row = idx / output_dim_tile;
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t in_col = 0; in_col < input_dim; ++in_col) {
    auto x_value = ct::load_masked(x + row * ct::full<IndexTile>(input_dim) + ct::full<IndexTile>(in_col), mask);
    auto w_value = ct::load_masked(weight + out_col * ct::full<IndexTile>(input_dim) + ct::full<IndexTile>(in_col), mask);
    acc = acc + x_value * w_value;
  }
  if (has_bias) {
    acc = acc + ct::load_masked(bias + out_col, mask);
  }
  ct::store_masked(out + idx, acc, mask);
}

__tile_global__ void linear_add_bias_float32_kernel(
    float* __restrict__ out,
    const float* __restrict__ bias,
    std::int64_t n,
    std::int64_t output_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  out = ct::assume_aligned(out, 16_ic);
  bias = ct::assume_aligned(bias, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto output_dim_tile = ct::full<IndexTile>(output_dim);
  auto out_col = idx % output_dim_tile;
  auto values = ct::load_masked(out + idx, mask) + ct::load_masked(bias + out_col, mask);
  ct::store_masked(out + idx, values, mask);
}

__tile_global__ void linear_backward_input_float32_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    float* __restrict__ grad_x,
    std::int64_t n,
    std::int64_t input_dim,
    std::int64_t output_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  grad_out = ct::assume_aligned(grad_out, 16_ic);
  weight = ct::assume_aligned(weight, 16_ic);
  grad_x = ct::assume_aligned(grad_x, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto input_dim_tile = ct::full<IndexTile>(input_dim);
  auto output_dim_tile = ct::full<IndexTile>(output_dim);
  auto in_col = idx % input_dim_tile;
  auto row = idx / input_dim_tile;
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t out_col = 0; out_col < output_dim; ++out_col) {
    auto grad_value = ct::load_masked(
        grad_out + row * output_dim_tile + ct::full<IndexTile>(out_col),
        mask);
    auto weight_value = ct::load_masked(
        weight + ct::full<IndexTile>(out_col) * input_dim_tile + in_col,
        mask);
    acc = acc + grad_value * weight_value;
  }
  ct::store_masked(grad_x + idx, acc, mask);
}

__tile_global__ void linear_backward_weight_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_weight,
    std::int64_t n,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_weight = ct::assume_aligned(grad_weight, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto input_dim_tile = ct::full<IndexTile>(input_dim);
  auto output_dim_tile = ct::full<IndexTile>(output_dim);
  auto out_col = idx / input_dim_tile;
  auto in_col = idx % input_dim_tile;
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t row_idx = 0; row_idx < rows; ++row_idx) {
    auto x_value = ct::load_masked(
        x + ct::full<IndexTile>(row_idx) * input_dim_tile + in_col,
        mask);
    auto grad_value = ct::load_masked(
        grad_out + ct::full<IndexTile>(row_idx) * output_dim_tile + out_col,
        mask);
    acc = acc + grad_value * x_value;
  }
  ct::store_masked(grad_weight + idx, acc, mask);
}

template <bool X_BF16, bool GRAD_BF16>
__global__ void linear_backward_weight_tiled_float32_kernel(
    const float* __restrict__ x,
    const std::uint16_t* __restrict__ x_bf16_bits,
    const float* __restrict__ grad_out,
    const std::uint16_t* __restrict__ grad_out_bf16_bits,
    float* __restrict__ grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool accumulate) {
  constexpr int kTile = 16;
  __shared__ float x_tile[kTile][kTile];
  __shared__ float grad_tile[kTile][kTile];

  const int in_col = static_cast<int>(blockIdx.x) * kTile + threadIdx.x;
  const int out_col = static_cast<int>(blockIdx.y) * kTile + threadIdx.y;
  float acc = 0.0f;

  for (std::int64_t row_base = 0; row_base < rows; row_base += kTile) {
    const std::int64_t row = row_base + threadIdx.y;
    if (row < rows && in_col < input_dim) {
      const std::int64_t x_idx = row * input_dim + in_col;
      if constexpr (X_BF16) {
        x_tile[threadIdx.y][threadIdx.x] = bf16_bits_to_f32_device(x_bf16_bits[x_idx]);
      } else {
        x_tile[threadIdx.y][threadIdx.x] = x[x_idx];
      }
    } else {
      x_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    const int grad_out_col = static_cast<int>(blockIdx.y) * kTile + threadIdx.x;
    if (row < rows && grad_out_col < output_dim) {
      const std::int64_t grad_idx = row * output_dim + grad_out_col;
      if constexpr (GRAD_BF16) {
        grad_tile[threadIdx.y][threadIdx.x] = bf16_bits_to_f32_device(grad_out_bf16_bits[grad_idx]);
      } else {
        grad_tile[threadIdx.y][threadIdx.x] = grad_out[grad_idx];
      }
    } else {
      grad_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();
    if (in_col < input_dim && out_col < output_dim) {
      for (int k = 0; k < kTile; ++k) {
        acc += x_tile[k][threadIdx.x] * grad_tile[k][threadIdx.y];
      }
    }
    __syncthreads();
  }

  if (in_col < input_dim && out_col < output_dim) {
    const std::int64_t weight_idx = static_cast<std::int64_t>(out_col) * input_dim + in_col;
    grad_weight[weight_idx] = accumulate ? grad_weight[weight_idx] + acc : acc;
  }
}

void launch_linear_backward_weight_tiled_float32_fallback(
    const float* x,
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool x_bf16,
    bool grad_bf16,
    bool accumulate,
    cudaStream_t stream) {
  constexpr int kTile = 16;
  const dim3 block(kTile, kTile, 1);
  const dim3 grid(
      static_cast<unsigned int>((input_dim + kTile - 1) / kTile),
      static_cast<unsigned int>((output_dim + kTile - 1) / kTile),
      1);
  if (x_bf16 && grad_bf16) {
    linear_backward_weight_tiled_float32_kernel<true, true><<<grid, block, 0, stream>>>(
        nullptr, x_bf16_bits, nullptr, grad_out_bf16_bits, grad_weight, rows, input_dim, output_dim, accumulate);
  } else if (x_bf16) {
    linear_backward_weight_tiled_float32_kernel<true, false><<<grid, block, 0, stream>>>(
        nullptr, x_bf16_bits, grad_out, nullptr, grad_weight, rows, input_dim, output_dim, accumulate);
  } else if (grad_bf16) {
    linear_backward_weight_tiled_float32_kernel<false, true><<<grid, block, 0, stream>>>(
        x, nullptr, nullptr, grad_out_bf16_bits, grad_weight, rows, input_dim, output_dim, accumulate);
  } else {
    linear_backward_weight_tiled_float32_kernel<false, false><<<grid, block, 0, stream>>>(
        x, nullptr, grad_out, nullptr, grad_weight, rows, input_dim, output_dim, accumulate);
  }
}

template <bool X_BF16, bool GRAD_BF16>
__global__ void linear_backward_weight_tiled_strided_float32_kernel(
    const float* __restrict__ x,
    const std::uint16_t* __restrict__ x_bf16_bits,
    const float* __restrict__ grad_out,
    const std::uint16_t* __restrict__ grad_out_bf16_bits,
    float* __restrict__ grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t grad_out_row_stride,
    bool accumulate) {
  constexpr int kTile = 16;
  __shared__ float x_tile[kTile][kTile];
  __shared__ float grad_tile[kTile][kTile];

  const int in_col = static_cast<int>(blockIdx.x) * kTile + threadIdx.x;
  const int out_col = static_cast<int>(blockIdx.y) * kTile + threadIdx.y;
  float acc = 0.0f;

  for (std::int64_t row_base = 0; row_base < rows; row_base += kTile) {
    const std::int64_t row = row_base + threadIdx.y;
    if (row < rows && in_col < input_dim) {
      const std::int64_t x_idx = row * input_dim + in_col;
      if constexpr (X_BF16) {
        x_tile[threadIdx.y][threadIdx.x] = bf16_bits_to_f32_device(x_bf16_bits[x_idx]);
      } else {
        x_tile[threadIdx.y][threadIdx.x] = x[x_idx];
      }
    } else {
      x_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    const int grad_out_col = static_cast<int>(blockIdx.y) * kTile + threadIdx.x;
    if (row < rows && grad_out_col < output_dim) {
      const std::int64_t grad_idx = row * grad_out_row_stride + grad_out_col;
      if constexpr (GRAD_BF16) {
        grad_tile[threadIdx.y][threadIdx.x] = bf16_bits_to_f32_device(grad_out_bf16_bits[grad_idx]);
      } else {
        grad_tile[threadIdx.y][threadIdx.x] = grad_out[grad_idx];
      }
    } else {
      grad_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();
    if (in_col < input_dim && out_col < output_dim) {
      for (int k = 0; k < kTile; ++k) {
        acc += x_tile[k][threadIdx.x] * grad_tile[k][threadIdx.y];
      }
    }
    __syncthreads();
  }

  if (in_col < input_dim && out_col < output_dim) {
    const std::int64_t weight_idx = static_cast<std::int64_t>(out_col) * input_dim + in_col;
    grad_weight[weight_idx] = accumulate ? grad_weight[weight_idx] + acc : acc;
  }
}

void launch_linear_backward_weight_tiled_strided_float32_fallback(
    const float* x,
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t grad_out_row_stride,
    bool x_bf16,
    bool grad_bf16,
    bool accumulate,
    cudaStream_t stream) {
  constexpr int kTile = 16;
  const dim3 block(kTile, kTile, 1);
  const dim3 grid(
      static_cast<unsigned int>((input_dim + kTile - 1) / kTile),
      static_cast<unsigned int>((output_dim + kTile - 1) / kTile),
      1);
  if (x_bf16 && grad_bf16) {
    linear_backward_weight_tiled_strided_float32_kernel<true, true><<<grid, block, 0, stream>>>(
        nullptr, x_bf16_bits, nullptr, grad_out_bf16_bits, grad_weight, rows, input_dim, output_dim, grad_out_row_stride, accumulate);
  } else if (x_bf16) {
    linear_backward_weight_tiled_strided_float32_kernel<true, false><<<grid, block, 0, stream>>>(
        nullptr, x_bf16_bits, grad_out, nullptr, grad_weight, rows, input_dim, output_dim, grad_out_row_stride, accumulate);
  } else if (grad_bf16) {
    linear_backward_weight_tiled_strided_float32_kernel<false, true><<<grid, block, 0, stream>>>(
        x, nullptr, nullptr, grad_out_bf16_bits, grad_weight, rows, input_dim, output_dim, grad_out_row_stride, accumulate);
  } else {
    linear_backward_weight_tiled_strided_float32_kernel<false, false><<<grid, block, 0, stream>>>(
        x, nullptr, grad_out, nullptr, grad_weight, rows, input_dim, output_dim, grad_out_row_stride, accumulate);
  }
}

__global__ void linear_backward_weight_accumulate_bf16_bits_bf16_bits_bf16_bits_kernel(
    const std::uint16_t* __restrict__ x_bf16_bits,
    const std::uint16_t* __restrict__ grad_out_bf16_bits,
    std::uint16_t* __restrict__ grad_weight_bf16_bits,
    std::int64_t n,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const std::int64_t out_col = idx / input_dim;
  const std::int64_t in_col = idx % input_dim;
  float acc = bf16_bits_to_f32_device(grad_weight_bf16_bits[idx]);
  for (std::int64_t row_idx = 0; row_idx < rows; ++row_idx) {
    const float x_value = bf16_bits_to_f32_device(x_bf16_bits[row_idx * input_dim + in_col]);
    const float grad_value = bf16_bits_to_f32_device(grad_out_bf16_bits[row_idx * output_dim + out_col]);
    acc += x_value * grad_value;
  }
  grad_weight_bf16_bits[idx] = f32_to_bf16_bits_device(acc);
}

__global__ void bf16_bits_accumulate_to_float32_kernel(
    float* __restrict__ dst,
    const std::uint16_t* __restrict__ src_bf16_bits,
    std::int64_t n,
    float beta) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const float src = bf16_bits_to_f32_device(src_bf16_bits[idx]);
  dst[idx] = beta == 0.0f ? src : (beta * dst[idx] + src);
}

__global__ void linear_backward_input_bf16_bits_float32_kernel(
    const std::uint16_t* __restrict__ grad_out_bf16_bits,
    const float* __restrict__ weight,
    float* __restrict__ grad_x,
    std::int64_t n,
    std::int64_t input_dim,
    std::int64_t output_dim) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const std::int64_t row = idx / input_dim;
  const std::int64_t in_col = idx % input_dim;
  float acc = 0.0f;
  for (std::int64_t out_col = 0; out_col < output_dim; ++out_col) {
    const float grad_value = bf16_bits_to_f32_device(grad_out_bf16_bits[row * output_dim + out_col]);
    acc += grad_value * weight[out_col * input_dim + in_col];
  }
  grad_x[idx] = acc;
}

__global__ void linear_backward_input_bf16_bits_weight_bf16_bits_float32_kernel(
    const std::uint16_t* __restrict__ grad_out_bf16_bits,
    const std::uint16_t* __restrict__ weight_bf16_bits,
    float* __restrict__ grad_x,
    std::int64_t n,
    std::int64_t input_dim,
    std::int64_t output_dim) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const std::int64_t row = idx / input_dim;
  const std::int64_t in_col = idx % input_dim;
  float acc = 0.0f;
  for (std::int64_t out_col = 0; out_col < output_dim; ++out_col) {
    const float grad_value = bf16_bits_to_f32_device(grad_out_bf16_bits[row * output_dim + out_col]);
    const float weight_value = bf16_bits_to_f32_device(weight_bf16_bits[out_col * input_dim + in_col]);
    acc += grad_value * weight_value;
  }
  grad_x[idx] = acc;
}

__global__ void linear_backward_input_bf16_bits_weight_bf16_bits_strided_float32_kernel(
    const std::uint16_t* __restrict__ grad_out_bf16_bits,
    const std::uint16_t* __restrict__ weight_bf16_bits,
    float* __restrict__ grad_x,
    std::int64_t n,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t grad_out_row_stride) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const std::int64_t row = idx / input_dim;
  const std::int64_t in_col = idx % input_dim;
  float acc = 0.0f;
  for (std::int64_t out_col = 0; out_col < output_dim; ++out_col) {
    const float grad_value = bf16_bits_to_f32_device(grad_out_bf16_bits[row * grad_out_row_stride + out_col]);
    const float weight_value = bf16_bits_to_f32_device(weight_bf16_bits[out_col * input_dim + in_col]);
    acc += grad_value * weight_value;
  }
  grad_x[idx] = acc;
}

__global__ void linear_backward_input_weight_bf16_bits_float32_kernel(
    const float* __restrict__ grad_out,
    const std::uint16_t* __restrict__ weight_bf16_bits,
    float* __restrict__ grad_x,
    std::int64_t n,
    std::int64_t input_dim,
    std::int64_t output_dim) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const std::int64_t row = idx / input_dim;
  const std::int64_t in_col = idx % input_dim;
  float acc = 0.0f;
  for (std::int64_t out_col = 0; out_col < output_dim; ++out_col) {
    const float grad_value = grad_out[row * output_dim + out_col];
    const float weight_value = bf16_bits_to_f32_device(weight_bf16_bits[out_col * input_dim + in_col]);
    acc += grad_value * weight_value;
  }
  grad_x[idx] = acc;
}

__global__ void linear_backward_input_weight_bf16_bits_to_bf16_bits_float32_kernel(
    const float* __restrict__ grad_out,
    const std::uint16_t* __restrict__ weight_bf16_bits,
    std::uint16_t* __restrict__ grad_x_bf16_bits,
    std::int64_t n,
    std::int64_t input_dim,
    std::int64_t output_dim) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const std::int64_t row = idx / input_dim;
  const std::int64_t in_col = idx % input_dim;
  float acc = 0.0f;
  for (std::int64_t out_col = 0; out_col < output_dim; ++out_col) {
    const float grad_value = grad_out[row * output_dim + out_col];
    const float weight_value = bf16_bits_to_f32_device(weight_bf16_bits[out_col * input_dim + in_col]);
    acc += grad_value * weight_value;
  }
  grad_x_bf16_bits[idx] = f32_to_bf16_bits_device(acc);
}

__global__ void linear_bf16_output_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    std::uint16_t* __restrict__ out_bf16_bits,
    std::int64_t n,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const std::int64_t row = idx / output_dim;
  const std::int64_t col = idx % output_dim;
  const float* x_row = x + row * input_dim;
  const float* w_row = weight + col * input_dim;
  float acc = has_bias ? bias[col] : 0.0f;
  for (std::int64_t i = 0; i < input_dim; ++i) {
    acc += x_row[i] * w_row[i];
  }
  out_bf16_bits[idx] = f32_to_bf16_bits_device(acc);
}

__global__ void linear_weight_bf16_output_float32_kernel(
    const float* __restrict__ x,
    const std::uint16_t* __restrict__ weight_bf16_bits,
    const float* __restrict__ bias,
    std::uint16_t* __restrict__ out_bf16_bits,
    std::int64_t n,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const std::int64_t row = idx / output_dim;
  const std::int64_t col = idx % output_dim;
  const float* x_row = x + row * input_dim;
  const std::uint16_t* w_row = weight_bf16_bits + col * input_dim;
  float acc = has_bias ? bias[col] : 0.0f;
  for (std::int64_t i = 0; i < input_dim; ++i) {
    acc += x_row[i] * bf16_bits_to_f32_device(w_row[i]);
  }
  out_bf16_bits[idx] = f32_to_bf16_bits_device(acc);
}

__global__ void linear_bf16_input_weight_bf16_output_float32_kernel(
    const std::uint16_t* __restrict__ x_bf16_bits,
    const std::uint16_t* __restrict__ weight_bf16_bits,
    const float* __restrict__ bias,
    std::uint16_t* __restrict__ out_bf16_bits,
    std::int64_t n,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const std::int64_t row = idx / output_dim;
  const std::int64_t col = idx % output_dim;
  const std::uint16_t* x_row = x_bf16_bits + row * input_dim;
  const std::uint16_t* w_row = weight_bf16_bits + col * input_dim;
  float acc = has_bias ? bias[col] : 0.0f;
  for (std::int64_t i = 0; i < input_dim; ++i) {
    acc += bf16_bits_to_f32_device(x_row[i]) * bf16_bits_to_f32_device(w_row[i]);
  }
  out_bf16_bits[idx] = f32_to_bf16_bits_device(acc);
}

__global__ void linear_bf16_input_float_weight_bf16_output_float32_kernel(
    const std::uint16_t* __restrict__ x_bf16_bits,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    std::uint16_t* __restrict__ out_bf16_bits,
    std::int64_t n,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const std::int64_t row = idx / output_dim;
  const std::int64_t col = idx % output_dim;
  const std::uint16_t* x_row = x_bf16_bits + row * input_dim;
  const float* w_row = weight + col * input_dim;
  float acc = has_bias ? bias[col] : 0.0f;
  for (std::int64_t i = 0; i < input_dim; ++i) {
    acc += bf16_bits_to_f32_device(x_row[i]) * w_row[i];
  }
  out_bf16_bits[idx] = f32_to_bf16_bits_device(acc);
}

__global__ void linear_bf16_gelu_bf16_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    std::uint16_t* __restrict__ pre_gelu_bf16_bits,
    std::uint16_t* __restrict__ gelu_bf16_bits,
    std::int64_t n,
    std::int64_t input_dim,
    std::int64_t output_dim) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const std::int64_t row = idx / output_dim;
  const std::int64_t col = idx % output_dim;
  const float* x_row = x + row * input_dim;
  const float* w_row = weight + col * input_dim;
  float acc = bias[col];
  for (std::int64_t i = 0; i < input_dim; ++i) {
    acc += x_row[i] * w_row[i];
  }
  const float x2 = acc * acc;
  const float tanh_out = tanhf(0.7978845608028654f * (acc + 0.044715f * acc * x2));
  const float gelu = 0.5f * acc * (1.0f + tanh_out);
  pre_gelu_bf16_bits[idx] = f32_to_bf16_bits_device(acc);
  gelu_bf16_bits[idx] = f32_to_bf16_bits_device(gelu);
}

__global__ void linear_weight_bf16_gelu_bf16_float32_kernel(
    const float* __restrict__ x,
    const std::uint16_t* __restrict__ weight_bf16_bits,
    const float* __restrict__ bias,
    std::uint16_t* __restrict__ pre_gelu_bf16_bits,
    std::uint16_t* __restrict__ gelu_bf16_bits,
    std::int64_t n,
    std::int64_t input_dim,
    std::int64_t output_dim) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const std::int64_t row = idx / output_dim;
  const std::int64_t col = idx % output_dim;
  const float* x_row = x + row * input_dim;
  const std::uint16_t* w_row = weight_bf16_bits + col * input_dim;
  float acc = bias[col];
  for (std::int64_t i = 0; i < input_dim; ++i) {
    acc += x_row[i] * bf16_bits_to_f32_device(w_row[i]);
  }
  const float x2 = acc * acc;
  const float tanh_out = tanhf(0.7978845608028654f * (acc + 0.044715f * acc * x2));
  const float gelu = 0.5f * acc * (1.0f + tanh_out);
  pre_gelu_bf16_bits[idx] = f32_to_bf16_bits_device(acc);
  gelu_bf16_bits[idx] = f32_to_bf16_bits_device(gelu);
}

__global__ void linear_bf16_input_weight_bf16_gelu_bf16_float32_kernel(
    const std::uint16_t* __restrict__ x_bf16_bits,
    const std::uint16_t* __restrict__ weight_bf16_bits,
    const float* __restrict__ bias,
    std::uint16_t* __restrict__ pre_gelu_bf16_bits,
    std::uint16_t* __restrict__ gelu_bf16_bits,
    std::int64_t n,
    std::int64_t input_dim,
    std::int64_t output_dim) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const std::int64_t row = idx / output_dim;
  const std::int64_t col = idx % output_dim;
  const std::uint16_t* x_row = x_bf16_bits + row * input_dim;
  const std::uint16_t* w_row = weight_bf16_bits + col * input_dim;
  float acc = bias[col];
  for (std::int64_t i = 0; i < input_dim; ++i) {
    acc += bf16_bits_to_f32_device(x_row[i]) * bf16_bits_to_f32_device(w_row[i]);
  }
  const float x2 = acc * acc;
  const float tanh_out = tanhf(0.7978845608028654f * (acc + 0.044715f * acc * x2));
  const float gelu = 0.5f * acc * (1.0f + tanh_out);
  pre_gelu_bf16_bits[idx] = f32_to_bf16_bits_device(acc);
  gelu_bf16_bits[idx] = f32_to_bf16_bits_device(gelu);
}

__global__ void linear_bf16_input_bits_float32_kernel(
    const std::uint16_t* __restrict__ x_bf16_bits,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const std::int64_t row = idx / output_dim;
  const std::int64_t col = idx % output_dim;
  const float* w_row = weight + col * input_dim;
  float acc = has_bias ? bias[col] : 0.0f;
  for (std::int64_t i = 0; i < input_dim; ++i) {
    acc += bf16_bits_to_f32_device(x_bf16_bits[row * input_dim + i]) * w_row[i];
  }
  out[idx] = acc;
}

__global__ void linear_weight_bf16_bits_float32_kernel(
    const float* __restrict__ x,
    const std::uint16_t* __restrict__ weight_bf16_bits,
    const float* __restrict__ bias,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const std::int64_t row = idx / output_dim;
  const std::int64_t col = idx % output_dim;
  const float* x_row = x + row * input_dim;
  const std::uint16_t* w_row = weight_bf16_bits + col * input_dim;
  float acc = has_bias ? bias[col] : 0.0f;
  for (std::int64_t i = 0; i < input_dim; ++i) {
    acc += x_row[i] * bf16_bits_to_f32_device(w_row[i]);
  }
  out[idx] = acc;
}

__global__ void linear_bf16_input_weight_bf16_bits_float32_kernel(
    const std::uint16_t* __restrict__ x_bf16_bits,
    const std::uint16_t* __restrict__ weight_bf16_bits,
    const float* __restrict__ bias,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const std::int64_t row = idx / output_dim;
  const std::int64_t col = idx % output_dim;
  const std::uint16_t* w_row = weight_bf16_bits + col * input_dim;
  float acc = has_bias ? bias[col] : 0.0f;
  for (std::int64_t i = 0; i < input_dim; ++i) {
    acc += bf16_bits_to_f32_device(x_bf16_bits[row * input_dim + i]) *
        bf16_bits_to_f32_device(w_row[i]);
  }
  out[idx] = acc;
}

__tile_global__ void linear_backward_bias_float32_kernel(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_bias,
    std::int64_t output_dim,
    std::int64_t rows) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_bias = ct::assume_aligned(grad_bias, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto out_col = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = out_col < ct::full<IndexTile>(output_dim);
  auto output_dim_tile = ct::full<IndexTile>(output_dim);
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t row_idx = 0; row_idx < rows; ++row_idx) {
    auto grad_value = ct::load_masked(
        grad_out + ct::full<IndexTile>(row_idx) * output_dim_tile + out_col,
        mask);
    acc = acc + grad_value;
  }
  ct::store_masked(grad_bias + out_col, acc, mask);
}

__tile_global__ void linear_backward_bias_accumulate_float32_kernel(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_bias,
    std::int64_t output_dim,
    std::int64_t rows) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_bias = ct::assume_aligned(grad_bias, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto out_col = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = out_col < ct::full<IndexTile>(output_dim);
  auto output_dim_tile = ct::full<IndexTile>(output_dim);
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t row_idx = 0; row_idx < rows; ++row_idx) {
    auto grad_value = ct::load_masked(
        grad_out + ct::full<IndexTile>(row_idx) * output_dim_tile + out_col,
        mask);
    acc = acc + grad_value;
  }
  auto current = ct::load_masked(grad_bias + out_col, mask);
  ct::store_masked(grad_bias + out_col, current + acc, mask);
}

__tile_global__ void linear_backward_bias_chunked_atomic_float32_kernel(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_bias,
    std::int64_t output_dim,
    std::int64_t rows,
    std::int64_t row_chunk_size,
    std::int64_t row_chunks) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_bias = ct::assume_aligned(grad_bias, 16_ic);

  const std::int64_t out_block = static_cast<std::int64_t>(ct::bid().x);
  const std::int64_t row_chunk = static_cast<std::int64_t>(ct::bid().y);
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto out_col = ct::iota<IndexTile>() + ct::full<IndexTile>(out_block * kTileSize);
  auto mask = out_col < ct::full<IndexTile>(output_dim);
  auto output_dim_tile = ct::full<IndexTile>(output_dim);
  const std::int64_t row_start = row_chunk * row_chunk_size;
  const std::int64_t row_end = (row_start + row_chunk_size < rows) ? row_start + row_chunk_size : rows;
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t row_idx = row_start; row_idx < row_end; ++row_idx) {
    auto grad_value = ct::load_masked(grad_out + ct::full<IndexTile>(row_idx) * output_dim_tile + out_col, mask);
    acc = acc + grad_value;
  }
  auto active = row_chunk < row_chunks ? mask : ct::full<decltype(mask)>(false);
  ct::atomic_add_masked<ct::memory_order::relaxed>(grad_bias + out_col, acc, active);
}

__global__ void linear_backward_bias_chunked_atomic_bf16_bits_float32_kernel(
    const std::uint16_t* __restrict__ grad_out_bf16_bits,
    float* __restrict__ grad_bias,
    std::int64_t output_dim,
    std::int64_t rows,
    std::int64_t row_chunk_size) {
  const std::int64_t out_col = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out_col >= output_dim) {
    return;
  }
  const std::int64_t row_chunk = static_cast<std::int64_t>(blockIdx.y);
  const std::int64_t row_start = row_chunk * row_chunk_size;
  const std::int64_t row_end = (row_start + row_chunk_size < rows) ? row_start + row_chunk_size : rows;
  float acc = 0.0f;
  for (std::int64_t row_idx = row_start; row_idx < row_end; ++row_idx) {
    acc += bf16_bits_to_f32_device(grad_out_bf16_bits[row_idx * output_dim + out_col]);
  }
  atomicAdd(grad_bias + out_col, acc);
}

__tile_global__ void gelu_float32_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  auto x_tile = ct::partition_view{ct::tensor_span{x, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto one = ct::full<decltype(x_tile)>(1.0f);
  auto half = ct::full<decltype(x_tile)>(0.5f);
  auto gelu_scale = ct::full<decltype(x_tile)>(0.7978845608028654f);
  auto gelu_cubic = ct::full<decltype(x_tile)>(0.044715f);
  auto x2 = x_tile * x_tile;
  auto tanh_arg = gelu_scale * (x_tile + gelu_cubic * x_tile * x2);
  auto result = half * x_tile * (one + ct::tanh(tanh_arg));
  ct::partition_view{ct::tensor_span{out, ct::extents{n}}, ct::shape{1024_ic}}.store_masked(result, bx);
}

__tile_global__ void gelu_add_bias_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ biased_out,
    float* __restrict__ gelu_out,
    std::int64_t n,
    std::int64_t output_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  bias = ct::assume_aligned(bias, 16_ic);
  biased_out = ct::assume_aligned(biased_out, 16_ic);
  gelu_out = ct::assume_aligned(gelu_out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto out_col = idx % ct::full<IndexTile>(output_dim);
  auto x_tile = ct::load_masked(x + idx, mask) + ct::load_masked(bias + out_col, mask);
  auto one = ct::full<decltype(x_tile)>(1.0f);
  auto half = ct::full<decltype(x_tile)>(0.5f);
  auto gelu_scale = ct::full<decltype(x_tile)>(0.7978845608028654f);
  auto gelu_cubic = ct::full<decltype(x_tile)>(0.044715f);
  auto x2 = x_tile * x_tile;
  auto tanh_arg = gelu_scale * (x_tile + gelu_cubic * x_tile * x2);
  auto result = half * x_tile * (one + ct::tanh(tanh_arg));
  ct::store_masked(biased_out + idx, x_tile, mask);
  ct::store_masked(gelu_out + idx, result, mask);
}

__tile_global__ void gelu_add_bias_bf16_act_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ biased_out,
    float* __restrict__ gelu_out,
    std::uint16_t* __restrict__ gelu_bf16_bits,
    std::int64_t n,
    std::int64_t output_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  bias = ct::assume_aligned(bias, 16_ic);
  biased_out = ct::assume_aligned(biased_out, 16_ic);
  gelu_out = ct::assume_aligned(gelu_out, 16_ic);
  auto* gelu_bf16 = ct::assume_aligned(reinterpret_cast<__nv_bfloat16*>(gelu_bf16_bits), 16_ic);

  const int bx = ct::bid().x;
  using Shape = decltype(ct::shape{1024_ic});
  using IndexTile = ct::tile<std::int64_t, Shape>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto out_col = idx % ct::full<IndexTile>(output_dim);
  auto biased = ct::load_masked(x + idx, mask) + ct::load_masked(bias + out_col, mask);
  auto one = ct::full<decltype(biased)>(1.0f);
  auto half = ct::full<decltype(biased)>(0.5f);
  auto gelu_scale = ct::full<decltype(biased)>(0.7978845608028654f);
  auto gelu_cubic = ct::full<decltype(biased)>(0.044715f);
  auto x2 = biased * biased;
  auto tanh_arg = gelu_scale * (biased + gelu_cubic * biased * x2);
  auto result = half * biased * (one + ct::tanh(tanh_arg));
  ct::store_masked(biased_out + idx, biased, mask);
  ct::store_masked(gelu_out + idx, result, mask);
  ct::store_masked(gelu_bf16 + idx, ct::element_cast<__nv_bfloat16>(result), mask);
}

__tile_global__ void linear_bias_residual_add_float32_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ linear_out,
    const float* __restrict__ bias,
    const float* __restrict__ residual_scale,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t output_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  residual = ct::assume_aligned(residual, 16_ic);
  linear_out = ct::assume_aligned(linear_out, 16_ic);
  bias = ct::assume_aligned(bias, 16_ic);
  residual_scale = ct::assume_aligned(residual_scale, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto out_col = idx % ct::full<IndexTile>(output_dim);
  auto scale = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(*residual_scale);
  auto projected = ct::load_masked(linear_out + idx, mask) + ct::load_masked(bias + out_col, mask);
  auto result = ct::load_masked(residual + idx, mask) + projected * scale;
  ct::store_masked(out + idx, result, mask);
}

__tile_global__ void linear_bias_residual_add_bf16_linear_float32_kernel(
    const float* __restrict__ residual,
    const std::uint16_t* __restrict__ linear_out_bf16_bits,
    const float* __restrict__ bias,
    const float* __restrict__ residual_scale,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t output_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  residual = ct::assume_aligned(residual, 16_ic);
  auto* linear_out = ct::assume_aligned(
      reinterpret_cast<const __nv_bfloat16*>(linear_out_bf16_bits), 16_ic);
  bias = ct::assume_aligned(bias, 16_ic);
  residual_scale = ct::assume_aligned(residual_scale, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using Shape = decltype(ct::shape{1024_ic});
  using IndexTile = ct::tile<std::int64_t, Shape>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto out_col = idx % ct::full<IndexTile>(output_dim);
  auto scale = ct::full<ct::tile<float, Shape>>(*residual_scale);
  auto projected =
      ct::element_cast<float>(ct::load_masked(linear_out + idx, mask)) +
      ct::load_masked(bias + out_col, mask);
  auto result = ct::load_masked(residual + idx, mask) + projected * scale;
  ct::store_masked(out + idx, result, mask);
}

__tile_global__ void linear_bias_residual_add_bf16_linear_bf16_residual_float32_kernel(
    const float* __restrict__ residual,
    const std::uint16_t* __restrict__ linear_out_bf16_bits,
    const float* __restrict__ bias,
    const float* __restrict__ residual_scale,
    float* __restrict__ out,
    std::uint16_t* __restrict__ residual_bf16_out,
    std::int64_t n,
    std::int64_t output_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  residual = ct::assume_aligned(residual, 16_ic);
  auto* linear_out = ct::assume_aligned(
      reinterpret_cast<const __nv_bfloat16*>(linear_out_bf16_bits), 16_ic);
  bias = ct::assume_aligned(bias, 16_ic);
  residual_scale = ct::assume_aligned(residual_scale, 16_ic);
  out = ct::assume_aligned(out, 16_ic);
  auto* residual_bf16 = ct::assume_aligned(
      reinterpret_cast<__nv_bfloat16*>(residual_bf16_out), 16_ic);

  const int bx = ct::bid().x;
  using Shape = decltype(ct::shape{1024_ic});
  using IndexTile = ct::tile<std::int64_t, Shape>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto out_col = idx % ct::full<IndexTile>(output_dim);
  auto scale = ct::full<ct::tile<float, Shape>>(*residual_scale);
  auto projected =
      ct::element_cast<float>(ct::load_masked(linear_out + idx, mask)) +
      ct::load_masked(bias + out_col, mask);
  auto result = ct::load_masked(residual + idx, mask) + projected * scale;
  ct::store_masked(out + idx, result, mask);
  ct::store_masked(residual_bf16 + idx, ct::element_cast<__nv_bfloat16>(result), mask);
}

__global__ void linear_bias_residual_add_bf16_linear_dim768_float32_kernel(
    const float* __restrict__ residual,
    const std::uint16_t* __restrict__ linear_out_bf16_bits,
    const float* __restrict__ bias,
    const float* __restrict__ residual_scale,
    float* __restrict__ out,
    std::int64_t rows) {
  constexpr int kDim = 768;
  constexpr int kRowsPerBlock = 2;
  constexpr int kValuesPerBlock = kDim * kRowsPerBlock;
  const std::int64_t row_base = static_cast<std::int64_t>(blockIdx.x) * kRowsPerBlock;
  const float scale = *residual_scale;
  const __nv_bfloat16* linear_out =
      reinterpret_cast<const __nv_bfloat16*>(linear_out_bf16_bits);
  for (int local = threadIdx.x; local < kValuesPerBlock; local += blockDim.x) {
    const std::int64_t row = row_base + local / kDim;
    if (row >= rows) {
      continue;
    }
    const int col = local - (local / kDim) * kDim;
    const std::int64_t idx = row * kDim + col;
    const float projected = __bfloat162float(linear_out[idx]) + bias[col];
    out[idx] = residual[idx] + projected * scale;
  }
}

__tile_global__ void linear_bias_residual_layer_norm_float32_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ linear_out,
    const float* __restrict__ linear_bias,
    const float* __restrict__ residual_scale,
    const float* __restrict__ norm_weight,
    const float* __restrict__ norm_bias,
    float* __restrict__ residual_out,
    float* __restrict__ norm_out,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    std::uint16_t* __restrict__ residual_bf16_out,
    std::uint16_t* __restrict__ norm_bf16_out,
    std::int64_t rows,
    std::int64_t dim,
    float eps) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  residual = ct::assume_aligned(residual, 16_ic);
  linear_out = ct::assume_aligned(linear_out, 16_ic);
  linear_bias = ct::assume_aligned(linear_bias, 16_ic);
  residual_scale = ct::assume_aligned(residual_scale, 16_ic);
  norm_weight = ct::assume_aligned(norm_weight, 16_ic);
  norm_bias = ct::assume_aligned(norm_bias, 16_ic);
  residual_out = ct::assume_aligned(residual_out, 16_ic);
  if (norm_out != nullptr) {
    norm_out = ct::assume_aligned(norm_out, 16_ic);
  }

  const int row = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>();
  auto mask = (ct::full<IndexTile>(row) < ct::full<IndexTile>(rows)) && (d < ct::full<IndexTile>(dim));
  auto base = ct::full<IndexTile>(static_cast<std::int64_t>(row) * dim);
  auto scale = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(*residual_scale);
  auto residual_tile = ct::load_masked(residual + base + d, mask);
  auto projected = ct::load_masked(linear_out + base + d, mask) + ct::load_masked(linear_bias + d, mask);
  auto combined = residual_tile + projected * scale;
  auto zero = ct::full<decltype(combined)>(0.0f);
  auto valid = ct::select(mask, combined, zero);
  ct::store_masked(residual_out + base + d, combined, mask);
  if (residual_bf16_out != nullptr) {
    auto* residual_bf16 = reinterpret_cast<__nv_bfloat16*>(residual_bf16_out);
    ct::store_masked(residual_bf16 + base + d, ct::element_cast<__nv_bfloat16>(combined), mask);
  }
  auto sum_x = ct::sum(valid, 0_ic);
  auto dim_f = ct::full<decltype(sum_x)>(static_cast<float>(dim));
  auto mean = sum_x / dim_f;
  auto centered = ct::select(mask, combined - mean, zero);
  auto sum_sq = ct::sum(centered * centered, 0_ic);
  auto var = sum_sq / ct::full<decltype(sum_sq)>(static_cast<float>(dim));
  auto norm_scale = ct::rsqrt(var + ct::full<decltype(var)>(eps));
  if (mean_out != nullptr && row < rows) {
    mean_out[row] = static_cast<float>(mean);
  }
  if (rstd_out != nullptr && row < rows) {
    rstd_out[row] = static_cast<float>(norm_scale);
  }
  auto weight_tile = ct::load_masked(norm_weight + d, mask);
  auto bias_tile = ct::load_masked(norm_bias + d, mask);
  auto norm_value = centered * norm_scale * weight_tile + bias_tile;
  if (norm_out != nullptr) {
    ct::store_masked(norm_out + base + d, norm_value, mask);
  }
  if (norm_bf16_out != nullptr) {
    auto* norm_bf16 = reinterpret_cast<__nv_bfloat16*>(norm_bf16_out);
    ct::store_masked(norm_bf16 + base + d, ct::element_cast<__nv_bfloat16>(norm_value), mask);
  }
}

__tile_global__ void linear_bias_residual_layer_norm_bf16_linear_float32_kernel(
    const float* __restrict__ residual,
    const std::uint16_t* __restrict__ linear_out_bf16_bits,
    const float* __restrict__ linear_bias,
    const float* __restrict__ residual_scale,
    const float* __restrict__ norm_weight,
    const float* __restrict__ norm_bias,
    float* __restrict__ residual_out,
    float* __restrict__ norm_out,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    std::uint16_t* __restrict__ residual_bf16_out,
    std::uint16_t* __restrict__ norm_bf16_out,
    std::int64_t rows,
    std::int64_t dim,
    float eps) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  residual = ct::assume_aligned(residual, 16_ic);
  auto* linear_out = ct::assume_aligned(
      reinterpret_cast<const __nv_bfloat16*>(linear_out_bf16_bits), 16_ic);
  linear_bias = ct::assume_aligned(linear_bias, 16_ic);
  residual_scale = ct::assume_aligned(residual_scale, 16_ic);
  norm_weight = ct::assume_aligned(norm_weight, 16_ic);
  norm_bias = ct::assume_aligned(norm_bias, 16_ic);
  residual_out = ct::assume_aligned(residual_out, 16_ic);
  if (norm_out != nullptr) {
    norm_out = ct::assume_aligned(norm_out, 16_ic);
  }

  const int row = ct::bid().x;
  using Shape = decltype(ct::shape{1024_ic});
  using IndexTile = ct::tile<std::int64_t, Shape>;
  auto d = ct::iota<IndexTile>();
  auto mask = (ct::full<IndexTile>(row) < ct::full<IndexTile>(rows)) && (d < ct::full<IndexTile>(dim));
  auto base = ct::full<IndexTile>(static_cast<std::int64_t>(row) * dim);
  auto scale = ct::full<ct::tile<float, Shape>>(*residual_scale);
  auto residual_tile = ct::load_masked(residual + base + d, mask);
  auto projected =
      ct::element_cast<float>(ct::load_masked(linear_out + base + d, mask)) +
      ct::load_masked(linear_bias + d, mask);
  auto combined = residual_tile + projected * scale;
  auto zero = ct::full<decltype(combined)>(0.0f);
  auto valid = ct::select(mask, combined, zero);
  ct::store_masked(residual_out + base + d, combined, mask);
  if (residual_bf16_out != nullptr) {
    auto* residual_bf16 = reinterpret_cast<__nv_bfloat16*>(residual_bf16_out);
    ct::store_masked(residual_bf16 + base + d, ct::element_cast<__nv_bfloat16>(combined), mask);
  }
  auto sum_x = ct::sum(valid, 0_ic);
  auto dim_f = ct::full<decltype(sum_x)>(static_cast<float>(dim));
  auto mean = sum_x / dim_f;
  auto centered = ct::select(mask, combined - mean, zero);
  auto sum_sq = ct::sum(centered * centered, 0_ic);
  auto var = sum_sq / ct::full<decltype(sum_sq)>(static_cast<float>(dim));
  auto norm_scale = ct::rsqrt(var + ct::full<decltype(var)>(eps));
  if (mean_out != nullptr && row < rows) {
    mean_out[row] = static_cast<float>(mean);
  }
  if (rstd_out != nullptr && row < rows) {
    rstd_out[row] = static_cast<float>(norm_scale);
  }
  auto weight_tile = ct::load_masked(norm_weight + d, mask);
  auto bias_tile = ct::load_masked(norm_bias + d, mask);
  auto norm_value = centered * norm_scale * weight_tile + bias_tile;
  if (norm_out != nullptr) {
    ct::store_masked(norm_out + base + d, norm_value, mask);
  }
  if (norm_bf16_out != nullptr) {
    auto* norm_bf16 = reinterpret_cast<__nv_bfloat16*>(norm_bf16_out);
    ct::store_masked(norm_bf16 + base + d, ct::element_cast<__nv_bfloat16>(norm_value), mask);
  }
}

__tile_global__ void gelu_backward_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_x,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_x = ct::assume_aligned(grad_x, 16_ic);

  const int bx = ct::bid().x;
  auto x_tile = ct::partition_view{ct::tensor_span{x, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto grad_tile = ct::partition_view{ct::tensor_span{grad_out, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto one = ct::full<decltype(x_tile)>(1.0f);
  auto half = ct::full<decltype(x_tile)>(0.5f);
  auto gelu_scale = ct::full<decltype(x_tile)>(0.7978845608028654f);
  auto gelu_cubic = ct::full<decltype(x_tile)>(0.044715f);
  auto x2 = x_tile * x_tile;
  auto tanh_out = ct::tanh(gelu_scale * (x_tile + gelu_cubic * x_tile * x2));
  auto sech_out = one - tanh_out * tanh_out;
  auto grad_local = half * (one + tanh_out)
      + half * x_tile * sech_out * gelu_scale
          * (one + ct::full<decltype(x_tile)>(3.0f) * gelu_cubic * x2);
  auto grad = grad_tile * grad_local;
  ct::partition_view{ct::tensor_span{grad_x, ct::extents{n}}, ct::shape{1024_ic}}.store_masked(grad, bx);
}

__tile_global__ void gelu_backward_inplace_float32_kernel(
    const float* __restrict__ x,
    float* __restrict__ grad,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  grad = ct::assume_aligned(grad, 16_ic);

  const int bx = ct::bid().x;
  auto x_tile = ct::partition_view{ct::tensor_span{x, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto grad_tile = ct::partition_view{ct::tensor_span{grad, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto one = ct::full<decltype(x_tile)>(1.0f);
  auto half = ct::full<decltype(x_tile)>(0.5f);
  auto gelu_scale = ct::full<decltype(x_tile)>(0.7978845608028654f);
  auto gelu_cubic = ct::full<decltype(x_tile)>(0.044715f);
  auto x2 = x_tile * x_tile;
  auto tanh_out = ct::tanh(gelu_scale * (x_tile + gelu_cubic * x_tile * x2));
  auto sech_out = one - tanh_out * tanh_out;
  auto grad_local = half * (one + tanh_out)
      + half * x_tile * sech_out * gelu_scale
          * (one + ct::full<decltype(x_tile)>(3.0f) * gelu_cubic * x2);
  auto result = grad_tile * grad_local;
  ct::partition_view{ct::tensor_span{grad, ct::extents{n}}, ct::shape{1024_ic}}.store_masked(result, bx);
}

__tile_global__ void gelu_backward_inplace_bf16_bits_float32_kernel(
    const std::uint16_t* __restrict__ x_bf16_bits,
    float* __restrict__ grad,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const auto* x_bf16 = ct::assume_aligned(reinterpret_cast<const __nv_bfloat16*>(x_bf16_bits), 16_ic);
  grad = ct::assume_aligned(grad, 16_ic);

  const int bx = ct::bid().x;
  using Shape = decltype(ct::shape{1024_ic});
  using IndexTile = ct::tile<std::int64_t, Shape>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto x = ct::element_cast<float>(ct::load_masked(x_bf16 + idx, mask));
  auto grad_tile = ct::load_masked(grad + idx, mask);
  auto one = ct::full<decltype(x)>(1.0f);
  auto half = ct::full<decltype(x)>(0.5f);
  auto gelu_scale = ct::full<decltype(x)>(0.7978845608028654f);
  auto gelu_cubic = ct::full<decltype(x)>(0.044715f);
  auto x2 = x * x;
  auto tanh_out = ct::tanh(gelu_scale * (x + gelu_cubic * x * x2));
  auto sech_out = one - tanh_out * tanh_out;
  auto grad_local = half * (one + tanh_out)
      + half * x * sech_out * gelu_scale
          * (one + ct::full<decltype(x)>(3.0f) * gelu_cubic * x2);
  ct::store_masked(grad + idx, grad_tile * grad_local, mask);
}

__global__ void gelu_backward_inplace_bf16_bits_to_bf16_bits_float32_kernel(
    const std::uint16_t* __restrict__ x_bf16_bits,
    std::uint16_t* __restrict__ grad_bf16_bits,
    std::int64_t n) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const float x = bf16_bits_to_f32_device(x_bf16_bits[idx]);
  const float grad = bf16_bits_to_f32_device(grad_bf16_bits[idx]);
  constexpr float kHalf = 0.5f;
  constexpr float kGeluScale = 0.7978845608028654f;
  constexpr float kGeluCubic = 0.044715f;
  const float x2 = x * x;
  const float tanh_out = tanhf(kGeluScale * (x + kGeluCubic * x * x2));
  const float sech_out = 1.0f - tanh_out * tanh_out;
  const float grad_local = kHalf * (1.0f + tanh_out) +
      kHalf * x * sech_out * kGeluScale * (1.0f + 3.0f * kGeluCubic * x2);
  grad_bf16_bits[idx] = f32_to_bf16_bits_device(grad * grad_local);
}

__tile_global__ void dropout_forward_float32_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    std::int64_t n,
    float dropout_p,
    std::int64_t seed) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using Shape = decltype(ct::shape{1024_ic});
  using IndexTile = ct::tile<std::int64_t, Shape>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto bucket = (idx * ct::full<IndexTile>(1103515245) + ct::full<IndexTile>(seed + 12345)) %
      ct::full<IndexTile>(10000);
  auto keep = bucket >= ct::full<IndexTile>(static_cast<std::int64_t>(dropout_p * 10000.0f));
  auto values = ct::load_masked(x + idx, mask);
  auto scaled = values * ct::full<decltype(values)>(1.0f / (1.0f - dropout_p));
  auto zero = ct::full<decltype(values)>(0.0f);
  ct::store_masked(out + idx, ct::select(keep, scaled, zero), mask);
}

__tile_global__ void dropout_backward_float32_kernel(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_x,
    std::int64_t n,
    float dropout_p,
    std::int64_t seed) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_x = ct::assume_aligned(grad_x, 16_ic);

  const int bx = ct::bid().x;
  using Shape = decltype(ct::shape{1024_ic});
  using IndexTile = ct::tile<std::int64_t, Shape>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto bucket = (idx * ct::full<IndexTile>(1103515245) + ct::full<IndexTile>(seed + 12345)) %
      ct::full<IndexTile>(10000);
  auto keep = bucket >= ct::full<IndexTile>(static_cast<std::int64_t>(dropout_p * 10000.0f));
  auto values = ct::load_masked(grad_out + idx, mask);
  auto scaled = values * ct::full<decltype(values)>(1.0f / (1.0f - dropout_p));
  auto zero = ct::full<decltype(values)>(0.0f);
  ct::store_masked(grad_x + idx, ct::select(keep, scaled, zero), mask);
}

__tile_global__ void act_weighted_sum_float32_kernel(
    const float* __restrict__ states,
    const float* __restrict__ weights,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t steps,
    std::int64_t inner) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  states = ct::assume_aligned(states, 16_ic);
  weights = ct::assume_aligned(weights, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto inner_tile = ct::full<IndexTile>(inner);
  auto offset = idx % inner_tile;
  auto batch = idx / inner_tile;
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t step = 0; step < steps; ++step) {
    auto weight = ct::load_masked(weights + batch * ct::full<IndexTile>(steps) + ct::full<IndexTile>(step), mask);
    auto state = ct::load_masked(
        states + (batch * ct::full<IndexTile>(steps) + ct::full<IndexTile>(step)) * inner_tile + offset,
        mask);
    acc = acc + state * weight;
  }
  ct::store_masked(out + idx, acc, mask);
}

__tile_global__ void latent_pool_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mask_values,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  mask_values = ct::assume_aligned(mask_values, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto active = idx < ct::full<IndexTile>(n);
  auto dim_tile = ct::full<IndexTile>(dim);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto d = idx % dim_tile;
  auto batch = idx / dim_tile;
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto fallback = ct::full<decltype(acc)>(0.0f);
  auto denom = ct::full<decltype(acc)>(0.0f);
  for (std::int64_t step = 0; step < seq_len; ++step) {
    auto mask_value = ct::load_masked(mask_values + batch * seq_tile + ct::full<IndexTile>(step), active);
    auto value = ct::load_masked(x + (batch * seq_tile + ct::full<IndexTile>(step)) * dim_tile + d, active);
    acc = acc + value * mask_value;
    fallback = fallback + value;
    denom = denom + mask_value;
  }
  auto zero = ct::full<decltype(acc)>(0.0f);
  auto pooled = acc / denom;
  auto mean = fallback / ct::full<decltype(acc)>(static_cast<float>(seq_len));
  auto result = ct::select(denom > zero, pooled, mean);
  ct::store_masked(out + idx, result, active);
}

__tile_global__ void token_cross_entropy_partials_float32_kernel(
    const float* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    float* __restrict__ partials,
    std::int64_t rows,
    std::int64_t vocab) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logits = ct::assume_aligned(logits, 16_ic);
  targets = ct::assume_aligned(targets, 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto row = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto active = row < ct::full<IndexTile>(rows);
  auto vocab_tile = ct::full<IndexTile>(vocab);
  auto maxv = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(-3.4028234663852886e38f);
  for (std::int64_t col = 0; col < vocab; ++col) {
    auto value = ct::load_masked(logits + row * vocab_tile + ct::full<IndexTile>(col), active);
    maxv = ct::select(value > maxv, value, maxv);
  }
  auto denom = ct::full<decltype(maxv)>(0.0f);
  for (std::int64_t col = 0; col < vocab; ++col) {
    auto value = ct::load_masked(logits + row * vocab_tile + ct::full<IndexTile>(col), active);
    denom = denom + ct::exp(value - maxv);
  }
  auto target = ct::load_masked(targets + row, active);
  auto target_value = ct::load_masked(logits + row * vocab_tile + target, active);
  auto loss = ct::log(denom) + maxv - target_value;
  loss = ct::select(active, loss, ct::full<decltype(loss)>(0.0f));
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(partials + out_idx, ct::sum(loss, 0_ic));
}

__tile_global__ void token_cross_entropy_partials_bf16_bits_kernel(
    const std::uint16_t* __restrict__ logits_bf16_bits,
    const std::int64_t* __restrict__ targets,
    float* __restrict__ partials,
    std::int64_t rows,
    std::int64_t vocab) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const auto* logits = ct::assume_aligned(reinterpret_cast<const __nv_bfloat16*>(logits_bf16_bits), 16_ic);
  targets = ct::assume_aligned(targets, 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto row = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto active = row < ct::full<IndexTile>(rows);
  auto vocab_tile = ct::full<IndexTile>(vocab);
  auto maxv = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(-3.4028234663852886e38f);
  for (std::int64_t col = 0; col < vocab; ++col) {
    auto value = ct::element_cast<float>(
        ct::load_masked(logits + row * vocab_tile + ct::full<IndexTile>(col), active));
    maxv = ct::select(value > maxv, value, maxv);
  }
  auto denom = ct::full<decltype(maxv)>(0.0f);
  for (std::int64_t col = 0; col < vocab; ++col) {
    auto value = ct::element_cast<float>(
        ct::load_masked(logits + row * vocab_tile + ct::full<IndexTile>(col), active));
    denom = denom + ct::exp(value - maxv);
  }
  auto target = ct::load_masked(targets + row, active);
  auto target_value = ct::element_cast<float>(ct::load_masked(logits + row * vocab_tile + target, active));
  auto loss = ct::log(denom) + maxv - target_value;
  loss = ct::select(active, loss, ct::full<decltype(loss)>(0.0f));
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(partials + out_idx, ct::sum(loss, 0_ic));
}

__tile_global__ void token_cross_entropy_partials_strided_float32_kernel(
    const float* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    float* __restrict__ partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logits = ct::assume_aligned(logits, 16_ic);
  targets = ct::assume_aligned(targets, 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto row = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto active = row < ct::full<IndexTile>(rows);
  auto stride_tile = ct::full<IndexTile>(row_stride);
  auto maxv = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(-3.4028234663852886e38f);
  for (std::int64_t col = 0; col < vocab; ++col) {
    auto value = ct::load_masked(logits + row * stride_tile + ct::full<IndexTile>(col), active);
    maxv = ct::select(value > maxv, value, maxv);
  }
  auto denom = ct::full<decltype(maxv)>(0.0f);
  for (std::int64_t col = 0; col < vocab; ++col) {
    auto value = ct::load_masked(logits + row * stride_tile + ct::full<IndexTile>(col), active);
    denom = denom + ct::exp(value - maxv);
  }
  auto target = ct::load_masked(targets + row, active);
  auto target_value = ct::load_masked(logits + row * stride_tile + target, active);
  auto loss = ct::log(denom) + maxv - target_value;
  loss = ct::select(active, loss, ct::full<decltype(loss)>(0.0f));
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(partials + out_idx, ct::sum(loss, 0_ic));
}

__tile_global__ void token_cross_entropy_partials_strided_bf16_bits_kernel(
    const std::uint16_t* __restrict__ logits_bf16_bits,
    const std::int64_t* __restrict__ targets,
    float* __restrict__ partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const auto* logits = ct::assume_aligned(reinterpret_cast<const __nv_bfloat16*>(logits_bf16_bits), 16_ic);
  targets = ct::assume_aligned(targets, 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto row = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto active = row < ct::full<IndexTile>(rows);
  auto stride_tile = ct::full<IndexTile>(row_stride);
  auto maxv = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(-3.4028234663852886e38f);
  for (std::int64_t col = 0; col < vocab; ++col) {
    auto value = ct::element_cast<float>(
        ct::load_masked(logits + row * stride_tile + ct::full<IndexTile>(col), active));
    maxv = ct::select(value > maxv, value, maxv);
  }
  auto denom = ct::full<decltype(maxv)>(0.0f);
  for (std::int64_t col = 0; col < vocab; ++col) {
    auto value = ct::element_cast<float>(
        ct::load_masked(logits + row * stride_tile + ct::full<IndexTile>(col), active));
    denom = denom + ct::exp(value - maxv);
  }
  auto target = ct::load_masked(targets + row, active);
  auto target_value = ct::element_cast<float>(ct::load_masked(logits + row * stride_tile + target, active));
  auto loss = ct::log(denom) + maxv - target_value;
  loss = ct::select(active, loss, ct::full<decltype(loss)>(0.0f));
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(partials + out_idx, ct::sum(loss, 0_ic));
}

__tile_global__ void token_cross_entropy_partials_strided_bf16_bits_u16_targets_kernel(
    const std::uint16_t* __restrict__ logits_bf16_bits,
    const std::uint16_t* __restrict__ targets,
    float* __restrict__ partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const auto* logits = ct::assume_aligned(reinterpret_cast<const __nv_bfloat16*>(logits_bf16_bits), 16_ic);
  targets = ct::assume_aligned(targets, 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto row = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto active = row < ct::full<IndexTile>(rows);
  auto stride_tile = ct::full<IndexTile>(row_stride);
  auto maxv = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(-3.4028234663852886e38f);
  for (std::int64_t col = 0; col < vocab; ++col) {
    auto value = ct::element_cast<float>(
        ct::load_masked(logits + row * stride_tile + ct::full<IndexTile>(col), active));
    maxv = ct::select(value > maxv, value, maxv);
  }
  auto denom = ct::full<decltype(maxv)>(0.0f);
  for (std::int64_t col = 0; col < vocab; ++col) {
    auto value = ct::element_cast<float>(
        ct::load_masked(logits + row * stride_tile + ct::full<IndexTile>(col), active));
    denom = denom + ct::exp(value - maxv);
  }
  auto target_u16 = ct::load_masked(targets + row, active);
  auto target = ct::element_cast<std::int64_t>(target_u16);
  auto target_value = ct::element_cast<float>(ct::load_masked(logits + row * stride_tile + target, active));
  auto loss = ct::log(denom) + maxv - target_value;
  loss = ct::select(active, loss, ct::full<decltype(loss)>(0.0f));
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(partials + out_idx, ct::sum(loss, 0_ic));
}

__tile_global__ void masked_token_cross_entropy_partials_float32_kernel(
    const float* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    const float* __restrict__ loss_mask,
    float* __restrict__ loss_partials,
    float* __restrict__ mask_partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t ignore_index) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logits = ct::assume_aligned(logits, 16_ic);
  targets = ct::assume_aligned(targets, 16_ic);
  loss_mask = ct::assume_aligned(loss_mask, 16_ic);
  loss_partials = ct::assume_aligned(loss_partials, 16_ic);
  mask_partials = ct::assume_aligned(mask_partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto row = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto active = row < ct::full<IndexTile>(rows);
  auto vocab_tile = ct::full<IndexTile>(vocab);
  auto target = ct::load_masked(targets + row, active);
  auto valid = active && (target != ct::full<IndexTile>(ignore_index));
  auto safe_target = ct::select(valid, target, ct::full<IndexTile>(0));
  auto maxv = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(-3.4028234663852886e38f);
  for (std::int64_t col = 0; col < vocab; ++col) {
    auto value = ct::load_masked(logits + row * vocab_tile + ct::full<IndexTile>(col), active);
    maxv = ct::select(value > maxv, value, maxv);
  }
  auto denom = ct::full<decltype(maxv)>(0.0f);
  for (std::int64_t col = 0; col < vocab; ++col) {
    auto value = ct::load_masked(logits + row * vocab_tile + ct::full<IndexTile>(col), active);
    denom = denom + ct::exp(value - maxv);
  }
  auto target_value = ct::load_masked(logits + row * vocab_tile + safe_target, active);
  auto per_token = ct::select(valid, ct::log(denom) + maxv - target_value, ct::full<decltype(maxv)>(0.0f));
  auto mask_value = ct::load_masked(loss_mask + row, active);
  auto weighted = ct::select(active, per_token * mask_value, ct::full<decltype(per_token)>(0.0f));
  auto mask_sum = ct::select(active, mask_value, ct::full<decltype(mask_value)>(0.0f));
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(loss_partials + out_idx, ct::sum(weighted, 0_ic));
  ct::store(mask_partials + out_idx, ct::sum(mask_sum, 0_ic));
}

__tile_global__ void token_cross_entropy_row_stats_float32_kernel(
    const float* __restrict__ logits,
    float* __restrict__ row_max_out,
    float* __restrict__ row_denom_out,
    std::int64_t rows,
    std::int64_t vocab) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logits = ct::assume_aligned(logits, 16_ic);
  row_max_out = ct::assume_aligned(row_max_out, 16_ic);
  row_denom_out = ct::assume_aligned(row_denom_out, 16_ic);

  const std::int64_t row = static_cast<std::int64_t>(ct::bid().x);
  if (row >= rows) {
    return;
  }
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  using ScalarTile = ct::tile<float, decltype(ct::shape{1_ic})>;
  auto local_col = ct::iota<IndexTile>();
  auto row_max = ct::full<ScalarTile>(-3.4028234663852886e38f);
  for (std::int64_t start = 0; start < vocab; start += kTileSize) {
    auto col = local_col + ct::full<IndexTile>(start);
    auto active = col < ct::full<IndexTile>(vocab);
    auto value = ct::load_masked(logits + ct::full<IndexTile>(row * vocab) + col, active);
    auto neg_inf = ct::full<decltype(value)>(-3.4028234663852886e38f);
    auto safe_value = ct::select(active, value, neg_inf);
    auto chunk_max = ct::reduce_max(safe_value, 0_ic);
    row_max = ct::select(chunk_max > row_max, chunk_max, row_max);
  }
  auto denom = ct::full<ScalarTile>(0.0f);
  for (std::int64_t start = 0; start < vocab; start += kTileSize) {
    auto col = local_col + ct::full<IndexTile>(start);
    auto active = col < ct::full<IndexTile>(vocab);
    auto value = ct::load_masked(logits + ct::full<IndexTile>(row * vocab) + col, active);
    auto exp_value = ct::select(active, ct::exp(value - row_max), ct::full<decltype(value)>(0.0f));
    denom = denom + ct::sum(exp_value, 0_ic);
  }
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(row);
  ct::store(row_max_out + out_idx, row_max);
  ct::store(row_denom_out + out_idx, denom);
}

__global__ void token_cross_entropy_bf16_bits_row_stats_kernel(
    const std::uint16_t* __restrict__ logits,
    float* __restrict__ row_max_out,
    float* __restrict__ row_denom_out,
    std::int64_t rows,
    std::int64_t vocab) {
  extern __shared__ float scratch[];
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
  const int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }
  float local_max = -3.4028234663852886e38f;
  for (std::int64_t col = tid; col < vocab; col += blockDim.x) {
    const float value = bf16_bits_to_f32_device(logits[row * vocab + col]);
    local_max = fmaxf(local_max, value);
  }
  scratch[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    }
    __syncthreads();
  }
  const float row_max = scratch[0];
  float local_denom = 0.0f;
  for (std::int64_t col = tid; col < vocab; col += blockDim.x) {
    const float value = bf16_bits_to_f32_device(logits[row * vocab + col]);
    local_denom += expf(value - row_max);
  }
  scratch[tid] = local_denom;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    row_max_out[row] = row_max;
    row_denom_out[row] = scratch[0];
  }
}

__global__ void token_cross_entropy_backward_inplace_bf16_bits_kernel(
    std::uint16_t* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    const float* __restrict__ row_max,
    const float* __restrict__ row_denom,
    std::int64_t elements,
    std::int64_t vocab,
    float loss_scale) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  const std::int64_t row = idx / vocab;
  const std::int64_t col = idx - row * vocab;
  const float value = bf16_bits_to_f32_device(logits[idx]);
  const float prob = expf(value - row_max[row]) / row_denom[row];
  const float onehot = col == targets[row] ? 1.0f : 0.0f;
  logits[idx] = f32_to_bf16_bits_device((prob - onehot) * loss_scale);
}

__device__ __forceinline__ float block_reduce_max_f32(float value, float* shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warp_count = (blockDim.x + 31) >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  }
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  value = lane < warp_count ? shared[lane] : -INFINITY;
  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    if (lane == 0) {
      shared[0] = value;
    }
  }
  __syncthreads();
  return shared[0];
}

__device__ __forceinline__ float block_reduce_sum_f32(float value, float* shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warp_count = (blockDim.x + 31) >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  value = lane < warp_count ? shared[lane] : 0.0f;
  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) {
      shared[0] = value;
    }
  }
  __syncthreads();
  return shared[0];
}

__device__ __forceinline__ float cross_entropy_exp_device(float value, bool use_exp2) {
  constexpr float kLog2E = 1.44269504088896340736f;
  return use_exp2 ? exp2f(value * kLog2E) : expf(value);
}

__device__ __forceinline__ int pack_two_u16_to_i32(std::uint16_t low, std::uint16_t high) {
  const std::uint32_t packed =
      static_cast<std::uint32_t>(low) | (static_cast<std::uint32_t>(high) << 16);
  return static_cast<int>(packed);
}

__device__ __forceinline__ int4 pack_eight_u16_to_int4(
    std::uint16_t v0,
    std::uint16_t v1,
    std::uint16_t v2,
    std::uint16_t v3,
    std::uint16_t v4,
    std::uint16_t v5,
    std::uint16_t v6,
    std::uint16_t v7) {
  return make_int4(
      pack_two_u16_to_i32(v0, v1),
      pack_two_u16_to_i32(v2, v3),
      pack_two_u16_to_i32(v4, v5),
      pack_two_u16_to_i32(v6, v7));
}

__device__ __forceinline__ std::uint16_t int4_u16_at(const int4& packed, int index) {
  const unsigned int lane =
      index < 2 ? static_cast<unsigned int>(packed.x)
                : (index < 4 ? static_cast<unsigned int>(packed.y)
                             : (index < 6 ? static_cast<unsigned int>(packed.z)
                                          : static_cast<unsigned int>(packed.w)));
  return static_cast<std::uint16_t>((index & 1) == 0 ? (lane & 0xffffu) : ((lane >> 16) & 0xffffu));
}

__device__ __forceinline__ int4 load_bf16_vec8(const std::uint16_t* __restrict__ src) {
  return *reinterpret_cast<const int4*>(src);
}

__device__ __forceinline__ float bf16_row_max_vec8_or_scalar(
    const std::uint16_t* __restrict__ row_logits,
    std::int64_t vocab,
    bool vec_loads) {
  float thread_max = -INFINITY;
  if (vec_loads) {
    constexpr std::int64_t kVec = 8;
    const std::int64_t aligned_vocab = vocab & ~(kVec - 1);
    for (std::int64_t col = static_cast<std::int64_t>(threadIdx.x) * kVec;
         col < aligned_vocab;
         col += static_cast<std::int64_t>(blockDim.x) * kVec) {
      const int4 packed = load_bf16_vec8(row_logits + col);
#pragma unroll
      for (int offset = 0; offset < 8; ++offset) {
        thread_max = fmaxf(thread_max, bf16_bits_to_f32_device(int4_u16_at(packed, offset)));
      }
    }
    for (std::int64_t col = aligned_vocab + threadIdx.x; col < vocab; col += blockDim.x) {
      thread_max = fmaxf(thread_max, bf16_bits_to_f32_device(row_logits[col]));
    }
  } else {
    for (std::int64_t col = threadIdx.x; col < vocab; col += blockDim.x) {
      thread_max = fmaxf(thread_max, bf16_bits_to_f32_device(row_logits[col]));
    }
  }
  return thread_max;
}

__device__ __forceinline__ float bf16_row_exp_sum_vec8_or_scalar(
    const std::uint16_t* __restrict__ row_logits,
    std::int64_t vocab,
    float row_max,
    bool vec_loads,
    bool use_exp2) {
  float thread_sum = 0.0f;
  if (vec_loads) {
    constexpr std::int64_t kVec = 8;
    const std::int64_t aligned_vocab = vocab & ~(kVec - 1);
    for (std::int64_t col = static_cast<std::int64_t>(threadIdx.x) * kVec;
         col < aligned_vocab;
         col += static_cast<std::int64_t>(blockDim.x) * kVec) {
      const int4 packed = load_bf16_vec8(row_logits + col);
#pragma unroll
      for (int offset = 0; offset < 8; ++offset) {
        thread_sum += cross_entropy_exp_device(
            bf16_bits_to_f32_device(int4_u16_at(packed, offset)) - row_max,
            use_exp2);
      }
    }
    for (std::int64_t col = aligned_vocab + threadIdx.x; col < vocab; col += blockDim.x) {
      thread_sum += cross_entropy_exp_device(bf16_bits_to_f32_device(row_logits[col]) - row_max, use_exp2);
    }
  } else {
    for (std::int64_t col = threadIdx.x; col < vocab; col += blockDim.x) {
      thread_sum += cross_entropy_exp_device(bf16_bits_to_f32_device(row_logits[col]) - row_max, use_exp2);
    }
  }
  return thread_sum;
}

__device__ __forceinline__ void store_bf16_vec8_streaming(
    std::uint16_t* __restrict__ dst,
    std::uint16_t v0,
    std::uint16_t v1,
    std::uint16_t v2,
    std::uint16_t v3,
    std::uint16_t v4,
    std::uint16_t v5,
    std::uint16_t v6,
    std::uint16_t v7) {
  __stcs(reinterpret_cast<int4*>(dst), pack_eight_u16_to_int4(v0, v1, v2, v3, v4, v5, v6, v7));
}

__device__ __forceinline__ void store_bf16_vec8_normal(
    std::uint16_t* __restrict__ dst,
    std::uint16_t v0,
    std::uint16_t v1,
    std::uint16_t v2,
    std::uint16_t v3,
    std::uint16_t v4,
    std::uint16_t v5,
    std::uint16_t v6,
    std::uint16_t v7) {
  *reinterpret_cast<int4*>(dst) = pack_eight_u16_to_int4(v0, v1, v2, v3, v4, v5, v6, v7);
}

__device__ __forceinline__ void store_bf16_vec8(
    std::uint16_t* __restrict__ dst,
    const std::uint16_t* __restrict__ values,
    bool streaming) {
  if (streaming) {
    store_bf16_vec8_streaming(
        dst,
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7]);
  } else {
    store_bf16_vec8_normal(
        dst,
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7]);
  }
}

__device__ __forceinline__ void store_bf16_scalar(
    std::uint16_t* __restrict__ dst,
    std::uint16_t value,
    bool streaming) {
  if (streaming) {
    asm volatile("st.global.cs.u16 [%0], %1;" :: "l"(dst), "h"(value) : "memory");
  } else {
    *dst = value;
  }
}

__global__ void token_cross_entropy_backward_inplace_bf16_bits_fused_kernel(
    std::uint16_t* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    std::int64_t rows,
    std::int64_t vocab,
    float loss_scale,
    bool vec_stores,
    bool vec_normal_stores,
    bool vec_loads,
    bool scalar_streaming_stores,
    bool use_exp2) {
  const std::int64_t row = rows - static_cast<std::int64_t>(blockIdx.x) - 1;
  if (row >= rows) {
    return;
  }
  __shared__ float reduce_max_shared[32];
  __shared__ float reduce_sum_shared[32];

  std::uint16_t* row_logits = logits + row * vocab;
  float thread_max = -INFINITY;
  for (std::int64_t col = threadIdx.x; col < vocab; col += blockDim.x) {
    thread_max = fmaxf(thread_max, bf16_bits_to_f32_device(row_logits[col]));
  }
  const float row_max = block_reduce_max_f32(thread_max, reduce_max_shared);

  float thread_sum = 0.0f;
  for (std::int64_t col = threadIdx.x; col < vocab; col += blockDim.x) {
    thread_sum += cross_entropy_exp_device(bf16_bits_to_f32_device(row_logits[col]) - row_max, use_exp2);
  }
  const float row_denom = block_reduce_sum_f32(thread_sum, reduce_sum_shared);
  const std::int64_t target = targets[row];

  if (vec_stores || vec_normal_stores) {
    constexpr std::int64_t kVec = 8;
    const std::int64_t aligned_vocab = vocab & ~(kVec - 1);
    for (std::int64_t col = static_cast<std::int64_t>(threadIdx.x) * kVec;
         col < aligned_vocab;
         col += static_cast<std::int64_t>(blockDim.x) * kVec) {
      std::uint16_t grad[8];
      const int4 packed = vec_loads ? load_bf16_vec8(row_logits + col) : make_int4(0, 0, 0, 0);
#pragma unroll
      for (int offset = 0; offset < 8; ++offset) {
        const std::int64_t current_col = col + offset;
        const std::uint16_t raw = vec_loads ? int4_u16_at(packed, offset) : row_logits[current_col];
        const float value = bf16_bits_to_f32_device(raw);
        const float prob = cross_entropy_exp_device(value - row_max, use_exp2) / row_denom;
        const float onehot = current_col == target ? 1.0f : 0.0f;
        grad[offset] = f32_to_bf16_bits_device((prob - onehot) * loss_scale);
      }
      store_bf16_vec8(row_logits + col, grad, vec_stores);
    }
    for (std::int64_t col = aligned_vocab + threadIdx.x; col < vocab; col += blockDim.x) {
      const float value = bf16_bits_to_f32_device(row_logits[col]);
      const float prob = cross_entropy_exp_device(value - row_max, use_exp2) / row_denom;
      const float onehot = col == target ? 1.0f : 0.0f;
      store_bf16_scalar(
          row_logits + col,
          f32_to_bf16_bits_device((prob - onehot) * loss_scale),
          scalar_streaming_stores);
    }
  } else {
    for (std::int64_t col = threadIdx.x; col < vocab; col += blockDim.x) {
      const float value = bf16_bits_to_f32_device(row_logits[col]);
      const float prob = cross_entropy_exp_device(value - row_max, use_exp2) / row_denom;
      const float onehot = col == target ? 1.0f : 0.0f;
      store_bf16_scalar(
          row_logits + col,
          f32_to_bf16_bits_device((prob - onehot) * loss_scale),
          scalar_streaming_stores);
    }
  }
}

__global__ void token_cross_entropy_backward_inplace_strided_float32_fused_kernel(
    float* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    bool zero_padding) {
  const std::int64_t row = rows - static_cast<std::int64_t>(blockIdx.x) - 1;
  if (row >= rows) {
    return;
  }
  __shared__ float reduce_max_shared[32];
  __shared__ float reduce_sum_shared[32];

  float* row_logits = logits + row * row_stride;
  float thread_max = -INFINITY;
  for (std::int64_t col = threadIdx.x; col < vocab; col += blockDim.x) {
    thread_max = fmaxf(thread_max, row_logits[col]);
  }
  const float row_max = block_reduce_max_f32(thread_max, reduce_max_shared);

  float thread_sum = 0.0f;
  for (std::int64_t col = threadIdx.x; col < vocab; col += blockDim.x) {
    thread_sum += expf(row_logits[col] - row_max);
  }
  const float row_denom = block_reduce_sum_f32(thread_sum, reduce_sum_shared);
  const std::int64_t target = targets[row];

  for (std::int64_t col = threadIdx.x; col < vocab; col += blockDim.x) {
    const float value = row_logits[col];
    const float prob = expf(value - row_max) / row_denom;
    const float onehot = col == target ? 1.0f : 0.0f;
    row_logits[col] = (prob - onehot) * loss_scale;
  }
  if (zero_padding) {
    for (std::int64_t col = vocab + threadIdx.x; col < row_stride; col += blockDim.x) {
      row_logits[col] = 0.0f;
    }
  }
}

__global__ void token_cross_entropy_backward_inplace_strided_bf16_bits_fused_kernel(
    std::uint16_t* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    bool vec_stores,
    bool vec_normal_stores,
    bool vec_loads,
    bool scalar_streaming_stores,
    bool use_exp2,
    bool zero_padding) {
  const std::int64_t row = rows - static_cast<std::int64_t>(blockIdx.x) - 1;
  if (row >= rows) {
    return;
  }
  __shared__ float reduce_max_shared[32];
  __shared__ float reduce_sum_shared[32];

  std::uint16_t* row_logits = logits + row * row_stride;
  float thread_max = bf16_row_max_vec8_or_scalar(row_logits, vocab, vec_loads);
  const float row_max = block_reduce_max_f32(thread_max, reduce_max_shared);

  float thread_sum = bf16_row_exp_sum_vec8_or_scalar(row_logits, vocab, row_max, vec_loads, use_exp2);
  const float row_denom = block_reduce_sum_f32(thread_sum, reduce_sum_shared);
  const std::int64_t target = targets[row];

  if (vec_stores || vec_normal_stores) {
    constexpr std::int64_t kVec = 8;
    const std::int64_t aligned_vocab = vocab & ~(kVec - 1);
    for (std::int64_t col = static_cast<std::int64_t>(threadIdx.x) * kVec;
         col < aligned_vocab;
         col += static_cast<std::int64_t>(blockDim.x) * kVec) {
      std::uint16_t grad[8];
      const int4 packed = vec_loads ? load_bf16_vec8(row_logits + col) : make_int4(0, 0, 0, 0);
#pragma unroll
      for (int offset = 0; offset < 8; ++offset) {
        const std::int64_t current_col = col + offset;
        const std::uint16_t raw = vec_loads ? int4_u16_at(packed, offset) : row_logits[current_col];
        const float value = bf16_bits_to_f32_device(raw);
        const float prob = cross_entropy_exp_device(value - row_max, use_exp2) / row_denom;
        const float onehot = current_col == target ? 1.0f : 0.0f;
        grad[offset] = f32_to_bf16_bits_device((prob - onehot) * loss_scale);
      }
      store_bf16_vec8(row_logits + col, grad, vec_stores);
    }
    for (std::int64_t col = aligned_vocab + threadIdx.x; col < vocab; col += blockDim.x) {
      const float value = bf16_bits_to_f32_device(row_logits[col]);
      const float prob = cross_entropy_exp_device(value - row_max, use_exp2) / row_denom;
      const float onehot = col == target ? 1.0f : 0.0f;
      store_bf16_scalar(
          row_logits + col,
          f32_to_bf16_bits_device((prob - onehot) * loss_scale),
          scalar_streaming_stores);
    }
  } else {
    for (std::int64_t col = threadIdx.x; col < vocab; col += blockDim.x) {
      const float value = bf16_bits_to_f32_device(row_logits[col]);
      const float prob = cross_entropy_exp_device(value - row_max, use_exp2) / row_denom;
      const float onehot = col == target ? 1.0f : 0.0f;
      store_bf16_scalar(
          row_logits + col,
          f32_to_bf16_bits_device((prob - onehot) * loss_scale),
          scalar_streaming_stores);
    }
  }
  if (zero_padding) {
    for (std::int64_t col = vocab + threadIdx.x; col < row_stride; col += blockDim.x) {
      row_logits[col] = f32_to_bf16_bits_device(0.0f);
    }
  }
}

__global__ void token_cross_entropy_backward_inplace_strided_bf16_bits_u16_targets_fused_kernel(
    std::uint16_t* __restrict__ logits,
    const std::uint16_t* __restrict__ targets,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    bool vec_stores,
    bool vec_normal_stores,
    bool vec_loads,
    bool scalar_streaming_stores,
    bool use_exp2,
    bool zero_padding) {
  const std::int64_t row = rows - static_cast<std::int64_t>(blockIdx.x) - 1;
  if (row >= rows) {
    return;
  }
  __shared__ float reduce_max_shared[32];
  __shared__ float reduce_sum_shared[32];

  std::uint16_t* row_logits = logits + row * row_stride;
  float thread_max = bf16_row_max_vec8_or_scalar(row_logits, vocab, vec_loads);
  const float row_max = block_reduce_max_f32(thread_max, reduce_max_shared);

  float thread_sum = bf16_row_exp_sum_vec8_or_scalar(row_logits, vocab, row_max, vec_loads, use_exp2);
  const float row_denom = block_reduce_sum_f32(thread_sum, reduce_sum_shared);
  const std::uint16_t target = targets[row];

  if (vec_stores || vec_normal_stores) {
    constexpr std::int64_t kVec = 8;
    const std::int64_t aligned_vocab = vocab & ~(kVec - 1);
    for (std::int64_t col = static_cast<std::int64_t>(threadIdx.x) * kVec;
         col < aligned_vocab;
         col += static_cast<std::int64_t>(blockDim.x) * kVec) {
      std::uint16_t grad[8];
      const int4 packed = vec_loads ? load_bf16_vec8(row_logits + col) : make_int4(0, 0, 0, 0);
#pragma unroll
      for (int offset = 0; offset < 8; ++offset) {
        const std::int64_t current_col = col + offset;
        const std::uint16_t raw = vec_loads ? int4_u16_at(packed, offset) : row_logits[current_col];
        const float value = bf16_bits_to_f32_device(raw);
        const float prob = cross_entropy_exp_device(value - row_max, use_exp2) / row_denom;
        const float onehot = current_col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
        grad[offset] = f32_to_bf16_bits_device((prob - onehot) * loss_scale);
      }
      store_bf16_vec8(row_logits + col, grad, vec_stores);
    }
    for (std::int64_t col = aligned_vocab + threadIdx.x; col < vocab; col += blockDim.x) {
      const float value = bf16_bits_to_f32_device(row_logits[col]);
      const float prob = cross_entropy_exp_device(value - row_max, use_exp2) / row_denom;
      const float onehot = col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
      store_bf16_scalar(
          row_logits + col,
          f32_to_bf16_bits_device((prob - onehot) * loss_scale),
          scalar_streaming_stores);
    }
  } else {
    if (vec_loads) {
      constexpr std::int64_t kVec = 8;
      const std::int64_t aligned_vocab = vocab & ~(kVec - 1);
      for (std::int64_t col = static_cast<std::int64_t>(threadIdx.x) * kVec;
           col < aligned_vocab;
           col += static_cast<std::int64_t>(blockDim.x) * kVec) {
        const int4 packed = load_bf16_vec8(row_logits + col);
#pragma unroll
        for (int offset = 0; offset < 8; ++offset) {
          const std::int64_t current_col = col + offset;
          const float value = bf16_bits_to_f32_device(int4_u16_at(packed, offset));
          const float prob = cross_entropy_exp_device(value - row_max, use_exp2) / row_denom;
          const float onehot = current_col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
          store_bf16_scalar(
              row_logits + current_col,
              f32_to_bf16_bits_device((prob - onehot) * loss_scale),
              scalar_streaming_stores);
        }
      }
      for (std::int64_t col = aligned_vocab + threadIdx.x; col < vocab; col += blockDim.x) {
        const float value = bf16_bits_to_f32_device(row_logits[col]);
        const float prob = cross_entropy_exp_device(value - row_max, use_exp2) / row_denom;
        const float onehot = col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
        store_bf16_scalar(
            row_logits + col,
            f32_to_bf16_bits_device((prob - onehot) * loss_scale),
            scalar_streaming_stores);
      }
    } else {
      for (std::int64_t col = threadIdx.x; col < vocab; col += blockDim.x) {
        const float value = bf16_bits_to_f32_device(row_logits[col]);
        const float prob = cross_entropy_exp_device(value - row_max, use_exp2) / row_denom;
        const float onehot = col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
        store_bf16_scalar(
            row_logits + col,
            f32_to_bf16_bits_device((prob - onehot) * loss_scale),
            scalar_streaming_stores);
      }
    }
  }
  if (zero_padding) {
    for (std::int64_t col = vocab + threadIdx.x; col < row_stride; col += blockDim.x) {
      row_logits[col] = f32_to_bf16_bits_device(0.0f);
    }
  }
}

__global__ void token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_default_kernel(
    std::uint16_t* __restrict__ logits,
    const std::uint16_t* __restrict__ targets,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    bool reverse_rows) {
  const std::int64_t launch_row = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t row = reverse_rows ? rows - launch_row - 1 : launch_row;
  if (row >= rows) {
    return;
  }
  __shared__ float reduce_max_shared[32];
  __shared__ float reduce_sum_shared[32];

  std::uint16_t* row_logits = logits + row * row_stride;
  float thread_max = bf16_row_max_vec8_or_scalar(row_logits, vocab, true);
  const float row_max = block_reduce_max_f32(thread_max, reduce_max_shared);

  float thread_sum = bf16_row_exp_sum_vec8_or_scalar(row_logits, vocab, row_max, true, false);
  const float row_denom = block_reduce_sum_f32(thread_sum, reduce_sum_shared);
  const std::uint16_t target = targets[row];

  constexpr std::int64_t kVec = 8;
  const std::int64_t aligned_vocab = vocab & ~(kVec - 1);
  for (std::int64_t col = static_cast<std::int64_t>(threadIdx.x) * kVec;
       col < aligned_vocab;
       col += static_cast<std::int64_t>(blockDim.x) * kVec) {
    const int4 packed = load_bf16_vec8(row_logits + col);
#pragma unroll
    for (int offset = 0; offset < 8; ++offset) {
      const std::int64_t current_col = col + offset;
      const float value = bf16_bits_to_f32_device(int4_u16_at(packed, offset));
      const float prob = expf(value - row_max) / row_denom;
      const float onehot = current_col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
      row_logits[current_col] = f32_to_bf16_bits_device((prob - onehot) * loss_scale);
    }
  }
  for (std::int64_t col = aligned_vocab + threadIdx.x; col < vocab; col += blockDim.x) {
    const float value = bf16_bits_to_f32_device(row_logits[col]);
    const float prob = expf(value - row_max) / row_denom;
    const float onehot = col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
    row_logits[col] = f32_to_bf16_bits_device((prob - onehot) * loss_scale);
  }
}

__global__ void token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_prob_only_kernel(
    std::uint16_t* __restrict__ logits,
    const std::uint16_t* __restrict__ targets,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    bool reverse_rows) {
  (void)targets;
  const std::int64_t launch_row = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t row = reverse_rows ? rows - launch_row - 1 : launch_row;
  if (row >= rows) {
    return;
  }
  __shared__ float reduce_max_shared[32];
  __shared__ float reduce_sum_shared[32];

  std::uint16_t* row_logits = logits + row * row_stride;
  float thread_max = bf16_row_max_vec8_or_scalar(row_logits, vocab, true);
  const float row_max = block_reduce_max_f32(thread_max, reduce_max_shared);

  float thread_sum = bf16_row_exp_sum_vec8_or_scalar(row_logits, vocab, row_max, true, false);
  const float row_denom = block_reduce_sum_f32(thread_sum, reduce_sum_shared);

  constexpr std::int64_t kVec = 8;
  const std::int64_t aligned_vocab = vocab & ~(kVec - 1);
  for (std::int64_t col = static_cast<std::int64_t>(threadIdx.x) * kVec;
       col < aligned_vocab;
       col += static_cast<std::int64_t>(blockDim.x) * kVec) {
    const int4 packed = load_bf16_vec8(row_logits + col);
#pragma unroll
    for (int offset = 0; offset < 8; ++offset) {
      const std::int64_t current_col = col + offset;
      const float value = bf16_bits_to_f32_device(int4_u16_at(packed, offset));
      const float prob = expf(value - row_max) / row_denom;
      row_logits[current_col] = f32_to_bf16_bits_device(prob * loss_scale);
    }
  }
  for (std::int64_t col = aligned_vocab + threadIdx.x; col < vocab; col += blockDim.x) {
    const float value = bf16_bits_to_f32_device(row_logits[col]);
    const float prob = expf(value - row_max) / row_denom;
    row_logits[col] = f32_to_bf16_bits_device(prob * loss_scale);
  }
}

__global__ void token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_llmk_style_kernel(
    std::uint16_t* __restrict__ logits,
    const std::uint16_t* __restrict__ targets,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    bool reverse_rows) {
  const std::int64_t launch_row = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t row = reverse_rows ? rows - launch_row - 1 : launch_row;
  if (row >= rows) {
    return;
  }
  __shared__ float reduce_max_shared[32];
  __shared__ float reduce_sum_shared[32];

  std::uint16_t* row_logits = logits + row * row_stride;
  float thread_max = bf16_row_max_vec8_or_scalar(row_logits, vocab, true);
  const float row_max = block_reduce_max_f32(thread_max, reduce_max_shared);

  float thread_sum = bf16_row_exp_sum_vec8_or_scalar(row_logits, vocab, row_max, true, false);
  const float row_denom = block_reduce_sum_f32(thread_sum, reduce_sum_shared);
  const std::uint16_t target = targets[row];

  constexpr std::int64_t kVec = 8;
  const std::int64_t aligned_vocab = vocab & ~(kVec - 1);
  for (std::int64_t col = static_cast<std::int64_t>(threadIdx.x) * kVec;
       col < aligned_vocab;
       col += static_cast<std::int64_t>(blockDim.x) * kVec) {
    const int4 packed = load_bf16_vec8(row_logits + col);
    std::uint16_t grad[8];
#pragma unroll
    for (int offset = 0; offset < 8; ++offset) {
      const std::int64_t current_col = col + offset;
      const float value = bf16_bits_to_f32_device(int4_u16_at(packed, offset));
      const float prob = expf(value - row_max) / row_denom;
      const float onehot = current_col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
      grad[offset] = f32_to_bf16_bits_device((prob - onehot) * loss_scale);
    }
    store_bf16_vec8_streaming(
        row_logits + col,
        grad[0],
        grad[1],
        grad[2],
        grad[3],
        grad[4],
        grad[5],
        grad[6],
        grad[7]);
  }
  for (std::int64_t col = aligned_vocab + threadIdx.x; col < vocab; col += blockDim.x) {
    const float value = bf16_bits_to_f32_device(row_logits[col]);
    const float prob = expf(value - row_max) / row_denom;
    const float onehot = col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
    store_bf16_scalar(
        row_logits + col,
        f32_to_bf16_bits_device((prob - onehot) * loss_scale),
        true);
  }
}

__global__ void lm_head_prob_only_dhidden_target_correction_bf16_bits_kernel(
    const std::uint16_t* __restrict__ targets,
    const std::uint16_t* __restrict__ token_weight_bf16,
    float* __restrict__ grad_hidden,
    std::int64_t rows,
    std::int64_t hidden_dim,
    std::int64_t token_weight_row_stride,
    float loss_scale) {
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t total = rows * hidden_dim;
  if (index >= total) {
    return;
  }
  const std::int64_t row = index / hidden_dim;
  const std::int64_t dim = index - row * hidden_dim;
  const std::uint16_t target = targets[row];
  const float weight = bf16_bits_to_f32_device(
      token_weight_bf16[static_cast<std::int64_t>(target) * token_weight_row_stride + dim]);
  grad_hidden[index] -= loss_scale * weight;
}

__global__ void lm_head_prob_only_dweight_target_correction_bf16_bits_kernel(
    const std::uint16_t* __restrict__ targets,
    const std::uint16_t* __restrict__ hidden_bf16,
    float* __restrict__ grad_weight,
    std::int64_t rows,
    std::int64_t hidden_dim,
    std::int64_t grad_weight_row_stride,
    float loss_scale) {
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t total = rows * hidden_dim;
  if (index >= total) {
    return;
  }
  const std::int64_t row = index / hidden_dim;
  const std::int64_t dim = index - row * hidden_dim;
  const std::uint16_t target = targets[row];
  const float hidden = bf16_bits_to_f32_device(hidden_bf16[index]);
  atomicAdd(grad_weight + static_cast<std::int64_t>(target) * grad_weight_row_stride + dim, -loss_scale * hidden);
}

__global__ void lm_head_prob_only_combined_target_correction_bf16_bits_kernel(
    const std::uint16_t* __restrict__ targets,
    const std::uint16_t* __restrict__ token_weight_bf16,
    const std::uint16_t* __restrict__ hidden_bf16,
    float* __restrict__ grad_hidden,
    float* __restrict__ grad_weight,
    std::int64_t rows,
    std::int64_t hidden_dim,
    std::int64_t token_weight_row_stride,
    std::int64_t grad_weight_row_stride,
    float loss_scale) {
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t total = rows * hidden_dim;
  if (index >= total) {
    return;
  }
  const std::int64_t row = index / hidden_dim;
  const std::int64_t dim = index - row * hidden_dim;
  const std::uint16_t target = targets[row];
  const float weight = bf16_bits_to_f32_device(
      token_weight_bf16[static_cast<std::int64_t>(target) * token_weight_row_stride + dim]);
  const float hidden = bf16_bits_to_f32_device(hidden_bf16[index]);
  grad_hidden[index] -= loss_scale * weight;
  atomicAdd(grad_weight + static_cast<std::int64_t>(target) * grad_weight_row_stride + dim, -loss_scale * hidden);
}

__global__ void token_cross_entropy_backward_loss_inplace_strided_bf16_bits_u16_targets_fused_kernel(
    std::uint16_t* __restrict__ logits,
    const std::uint16_t* __restrict__ targets,
    float* __restrict__ loss_out,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    bool vec_stores,
    bool vec_normal_stores,
    bool vec_loads,
    bool scalar_streaming_stores,
    bool use_exp2,
    bool zero_padding,
    bool write_row_losses) {
  const std::int64_t row = rows - static_cast<std::int64_t>(blockIdx.x) - 1;
  if (row >= rows) {
    return;
  }
  __shared__ float reduce_max_shared[32];
  __shared__ float reduce_sum_shared[32];

  std::uint16_t* row_logits = logits + row * row_stride;
  float thread_max = bf16_row_max_vec8_or_scalar(row_logits, vocab, vec_loads);
  const float row_max = block_reduce_max_f32(thread_max, reduce_max_shared);

  float thread_sum = bf16_row_exp_sum_vec8_or_scalar(row_logits, vocab, row_max, vec_loads, use_exp2);
  const float row_denom = block_reduce_sum_f32(thread_sum, reduce_sum_shared);
  const std::uint16_t target = targets[row];
  const float target_logit = bf16_bits_to_f32_device(row_logits[static_cast<std::int64_t>(target)]);

  if (threadIdx.x == 0 && loss_out != nullptr) {
    const float loss = logf(row_denom) + row_max - target_logit;
    if (write_row_losses) {
      loss_out[row] = loss;
    } else {
      atomicAdd(loss_out, loss);
    }
  }

  if (vec_stores || vec_normal_stores) {
    constexpr std::int64_t kVec = 8;
    const std::int64_t aligned_vocab = vocab & ~(kVec - 1);
    for (std::int64_t col = static_cast<std::int64_t>(threadIdx.x) * kVec;
         col < aligned_vocab;
         col += static_cast<std::int64_t>(blockDim.x) * kVec) {
      std::uint16_t grad[8];
      const int4 packed = vec_loads ? load_bf16_vec8(row_logits + col) : make_int4(0, 0, 0, 0);
#pragma unroll
      for (int offset = 0; offset < 8; ++offset) {
        const std::int64_t current_col = col + offset;
        const std::uint16_t raw = vec_loads ? int4_u16_at(packed, offset) : row_logits[current_col];
        const float value = bf16_bits_to_f32_device(raw);
        const float prob = cross_entropy_exp_device(value - row_max, use_exp2) / row_denom;
        const float onehot = current_col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
        grad[offset] = f32_to_bf16_bits_device((prob - onehot) * loss_scale);
      }
      store_bf16_vec8(row_logits + col, grad, vec_stores);
    }
    for (std::int64_t col = aligned_vocab + threadIdx.x; col < vocab; col += blockDim.x) {
      const float value = bf16_bits_to_f32_device(row_logits[col]);
      const float prob = cross_entropy_exp_device(value - row_max, use_exp2) / row_denom;
      const float onehot = col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
      store_bf16_scalar(
          row_logits + col,
          f32_to_bf16_bits_device((prob - onehot) * loss_scale),
          scalar_streaming_stores);
    }
  } else {
    if (vec_loads) {
      constexpr std::int64_t kVec = 8;
      const std::int64_t aligned_vocab = vocab & ~(kVec - 1);
      for (std::int64_t col = static_cast<std::int64_t>(threadIdx.x) * kVec;
           col < aligned_vocab;
           col += static_cast<std::int64_t>(blockDim.x) * kVec) {
        const int4 packed = load_bf16_vec8(row_logits + col);
#pragma unroll
        for (int offset = 0; offset < 8; ++offset) {
          const std::int64_t current_col = col + offset;
          const float value = bf16_bits_to_f32_device(int4_u16_at(packed, offset));
          const float prob = cross_entropy_exp_device(value - row_max, use_exp2) / row_denom;
          const float onehot = current_col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
          store_bf16_scalar(
              row_logits + current_col,
              f32_to_bf16_bits_device((prob - onehot) * loss_scale),
              scalar_streaming_stores);
        }
      }
      for (std::int64_t col = aligned_vocab + threadIdx.x; col < vocab; col += blockDim.x) {
        const float value = bf16_bits_to_f32_device(row_logits[col]);
        const float prob = cross_entropy_exp_device(value - row_max, use_exp2) / row_denom;
        const float onehot = col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
        store_bf16_scalar(
            row_logits + col,
            f32_to_bf16_bits_device((prob - onehot) * loss_scale),
            scalar_streaming_stores);
      }
    } else {
      for (std::int64_t col = threadIdx.x; col < vocab; col += blockDim.x) {
        const float value = bf16_bits_to_f32_device(row_logits[col]);
        const float prob = cross_entropy_exp_device(value - row_max, use_exp2) / row_denom;
        const float onehot = col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
        store_bf16_scalar(
            row_logits + col,
            f32_to_bf16_bits_device((prob - onehot) * loss_scale),
            scalar_streaming_stores);
      }
    }
  }
  if (zero_padding) {
    for (std::int64_t col = vocab + threadIdx.x; col < row_stride; col += blockDim.x) {
      row_logits[col] = f32_to_bf16_bits_device(0.0f);
    }
  }
}

__global__ void lm_head_classifier_backward_row_losses_default_bf16_bits_u16_targets_kernel(
    std::uint16_t* __restrict__ logits,
    const std::uint16_t* __restrict__ targets,
    float* __restrict__ row_losses,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale) {
  const std::int64_t row = rows - static_cast<std::int64_t>(blockIdx.x) - 1;
  if (row >= rows) {
    return;
  }
  __shared__ float reduce_max_shared[32];
  __shared__ float reduce_sum_shared[32];

  std::uint16_t* row_logits = logits + row * row_stride;
  float thread_max = bf16_row_max_vec8_or_scalar(row_logits, vocab, true);
  const float row_max = block_reduce_max_f32(thread_max, reduce_max_shared);

  float thread_sum = bf16_row_exp_sum_vec8_or_scalar(row_logits, vocab, row_max, true, false);
  const float row_denom = block_reduce_sum_f32(thread_sum, reduce_sum_shared);
  const std::uint16_t target = targets[row];
  const float target_logit = bf16_bits_to_f32_device(row_logits[static_cast<std::int64_t>(target)]);

  if (threadIdx.x == 0 && row_losses != nullptr) {
    row_losses[row] = logf(row_denom) + row_max - target_logit;
  }

  constexpr std::int64_t kVec = 8;
  const std::int64_t aligned_vocab = vocab & ~(kVec - 1);
  for (std::int64_t col = static_cast<std::int64_t>(threadIdx.x) * kVec;
       col < aligned_vocab;
       col += static_cast<std::int64_t>(blockDim.x) * kVec) {
    const int4 packed = load_bf16_vec8(row_logits + col);
#pragma unroll
    for (int offset = 0; offset < 8; ++offset) {
      const std::int64_t current_col = col + offset;
      const float value = bf16_bits_to_f32_device(int4_u16_at(packed, offset));
      const float prob = expf(value - row_max) / row_denom;
      const float onehot = current_col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
      row_logits[current_col] = f32_to_bf16_bits_device((prob - onehot) * loss_scale);
    }
  }
  for (std::int64_t col = aligned_vocab + threadIdx.x; col < vocab; col += blockDim.x) {
    const float value = bf16_bits_to_f32_device(row_logits[col]);
    const float prob = expf(value - row_max) / row_denom;
    const float onehot = col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
    row_logits[col] = f32_to_bf16_bits_device((prob - onehot) * loss_scale);
  }
}

__global__ void lm_head_classifier_backward_row_losses_llmk_style_bf16_bits_u16_targets_kernel(
    std::uint16_t* __restrict__ logits,
    const std::uint16_t* __restrict__ targets,
    float* __restrict__ row_losses,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale) {
  const std::int64_t row = rows - static_cast<std::int64_t>(blockIdx.x) - 1;
  if (row >= rows) {
    return;
  }
  __shared__ float reduce_max_shared[32];
  __shared__ float reduce_sum_shared[32];

  std::uint16_t* row_logits = logits + row * row_stride;
  float thread_max = bf16_row_max_vec8_or_scalar(row_logits, vocab, true);
  const float row_max = block_reduce_max_f32(thread_max, reduce_max_shared);

  float thread_sum = bf16_row_exp_sum_vec8_or_scalar(row_logits, vocab, row_max, true, false);
  const float row_denom = block_reduce_sum_f32(thread_sum, reduce_sum_shared);
  const std::uint16_t target = targets[row];
  const float target_logit = bf16_bits_to_f32_device(row_logits[static_cast<std::int64_t>(target)]);

  if (threadIdx.x == 0 && row_losses != nullptr) {
    row_losses[row] = logf(row_denom) + row_max - target_logit;
  }

  constexpr std::int64_t kVec = 8;
  const std::int64_t aligned_vocab = vocab & ~(kVec - 1);
  for (std::int64_t col = static_cast<std::int64_t>(threadIdx.x) * kVec;
       col < aligned_vocab;
       col += static_cast<std::int64_t>(blockDim.x) * kVec) {
    const int4 packed = load_bf16_vec8(row_logits + col);
    std::uint16_t grad[8];
#pragma unroll
    for (int offset = 0; offset < 8; ++offset) {
      const std::int64_t current_col = col + offset;
      const float value = bf16_bits_to_f32_device(int4_u16_at(packed, offset));
      const float prob = expf(value - row_max) / row_denom;
      const float onehot = current_col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
      grad[offset] = f32_to_bf16_bits_device((prob - onehot) * loss_scale);
    }
    store_bf16_vec8_streaming(
        row_logits + col,
        grad[0],
        grad[1],
        grad[2],
        grad[3],
        grad[4],
        grad[5],
        grad[6],
        grad[7]);
  }
  for (std::int64_t col = aligned_vocab + threadIdx.x; col < vocab; col += blockDim.x) {
    const float value = bf16_bits_to_f32_device(row_logits[col]);
    const float prob = expf(value - row_max) / row_denom;
    const float onehot = col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
    store_bf16_scalar(
        row_logits + col,
        f32_to_bf16_bits_device((prob - onehot) * loss_scale),
        true);
  }
}

__global__ void lm_head_classifier_backward_loss_bins_default_bf16_bits_u16_targets_kernel(
    std::uint16_t* __restrict__ logits,
    const std::uint16_t* __restrict__ targets,
    float* __restrict__ loss_bins,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    std::int64_t loss_bin_count,
    float loss_scale) {
  const std::int64_t row = rows - static_cast<std::int64_t>(blockIdx.x) - 1;
  if (row >= rows) {
    return;
  }
  __shared__ float reduce_max_shared[32];
  __shared__ float reduce_sum_shared[32];

  std::uint16_t* row_logits = logits + row * row_stride;
  float thread_max = bf16_row_max_vec8_or_scalar(row_logits, vocab, true);
  const float row_max = block_reduce_max_f32(thread_max, reduce_max_shared);

  float thread_sum = bf16_row_exp_sum_vec8_or_scalar(row_logits, vocab, row_max, true, false);
  const float row_denom = block_reduce_sum_f32(thread_sum, reduce_sum_shared);
  const std::uint16_t target = targets[row];
  const float target_logit = bf16_bits_to_f32_device(row_logits[static_cast<std::int64_t>(target)]);

  if (threadIdx.x == 0 && loss_bins != nullptr && loss_bin_count > 0) {
    const float loss = logf(row_denom) + row_max - target_logit;
    atomicAdd(loss_bins + (row % loss_bin_count), loss);
  }

  constexpr std::int64_t kVec = 8;
  const std::int64_t aligned_vocab = vocab & ~(kVec - 1);
  for (std::int64_t col = static_cast<std::int64_t>(threadIdx.x) * kVec;
       col < aligned_vocab;
       col += static_cast<std::int64_t>(blockDim.x) * kVec) {
    const int4 packed = load_bf16_vec8(row_logits + col);
#pragma unroll
    for (int offset = 0; offset < 8; ++offset) {
      const std::int64_t current_col = col + offset;
      const float value = bf16_bits_to_f32_device(int4_u16_at(packed, offset));
      const float prob = expf(value - row_max) / row_denom;
      const float onehot = current_col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
      row_logits[current_col] = f32_to_bf16_bits_device((prob - onehot) * loss_scale);
    }
  }
  for (std::int64_t col = aligned_vocab + threadIdx.x; col < vocab; col += blockDim.x) {
    const float value = bf16_bits_to_f32_device(row_logits[col]);
    const float prob = expf(value - row_max) / row_denom;
    const float onehot = col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
    row_logits[col] = f32_to_bf16_bits_device((prob - onehot) * loss_scale);
  }
}

__global__ void lm_head_classifier_backward_loss_bins_llmk_style_bf16_bits_u16_targets_kernel(
    std::uint16_t* __restrict__ logits,
    const std::uint16_t* __restrict__ targets,
    float* __restrict__ loss_bins,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    std::int64_t loss_bin_count,
    float loss_scale) {
  const std::int64_t row = rows - static_cast<std::int64_t>(blockIdx.x) - 1;
  if (row >= rows) {
    return;
  }
  __shared__ float reduce_max_shared[32];
  __shared__ float reduce_sum_shared[32];

  std::uint16_t* row_logits = logits + row * row_stride;
  float thread_max = bf16_row_max_vec8_or_scalar(row_logits, vocab, true);
  const float row_max = block_reduce_max_f32(thread_max, reduce_max_shared);

  float thread_sum = bf16_row_exp_sum_vec8_or_scalar(row_logits, vocab, row_max, true, false);
  const float row_denom = block_reduce_sum_f32(thread_sum, reduce_sum_shared);
  const std::uint16_t target = targets[row];
  const float target_logit = bf16_bits_to_f32_device(row_logits[static_cast<std::int64_t>(target)]);

  if (threadIdx.x == 0 && loss_bins != nullptr && loss_bin_count > 0) {
    const float loss = logf(row_denom) + row_max - target_logit;
    atomicAdd(loss_bins + (row % loss_bin_count), loss);
  }

  constexpr std::int64_t kVec = 8;
  const std::int64_t aligned_vocab = vocab & ~(kVec - 1);
  for (std::int64_t col = static_cast<std::int64_t>(threadIdx.x) * kVec;
       col < aligned_vocab;
       col += static_cast<std::int64_t>(blockDim.x) * kVec) {
    const int4 packed = load_bf16_vec8(row_logits + col);
    std::uint16_t grad[8];
#pragma unroll
    for (int offset = 0; offset < 8; ++offset) {
      const std::int64_t current_col = col + offset;
      const float value = bf16_bits_to_f32_device(int4_u16_at(packed, offset));
      const float prob = expf(value - row_max) / row_denom;
      const float onehot = current_col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
      grad[offset] = f32_to_bf16_bits_device((prob - onehot) * loss_scale);
    }
    store_bf16_vec8_streaming(
        row_logits + col,
        grad[0],
        grad[1],
        grad[2],
        grad[3],
        grad[4],
        grad[5],
        grad[6],
        grad[7]);
  }
  for (std::int64_t col = aligned_vocab + threadIdx.x; col < vocab; col += blockDim.x) {
    const float value = bf16_bits_to_f32_device(row_logits[col]);
    const float prob = expf(value - row_max) / row_denom;
    const float onehot = col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
    store_bf16_scalar(
        row_logits + col,
        f32_to_bf16_bits_device((prob - onehot) * loss_scale),
        true);
  }
}

__global__ void token_cross_entropy_backward_loss_bins_inplace_strided_bf16_bits_u16_targets_fused_kernel(
    std::uint16_t* __restrict__ logits,
    const std::uint16_t* __restrict__ targets,
    float* __restrict__ loss_bins,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    std::int64_t loss_bin_count,
    float loss_scale,
    bool vec_stores,
    bool vec_normal_stores,
    bool vec_loads,
    bool scalar_streaming_stores,
    bool use_exp2) {
  const std::int64_t row = rows - static_cast<std::int64_t>(blockIdx.x) - 1;
  if (row >= rows) {
    return;
  }
  __shared__ float reduce_max_shared[32];
  __shared__ float reduce_sum_shared[32];

  std::uint16_t* row_logits = logits + row * row_stride;
  float thread_max = bf16_row_max_vec8_or_scalar(row_logits, vocab, vec_loads);
  const float row_max = block_reduce_max_f32(thread_max, reduce_max_shared);

  float thread_sum = bf16_row_exp_sum_vec8_or_scalar(row_logits, vocab, row_max, vec_loads, use_exp2);
  const float row_denom = block_reduce_sum_f32(thread_sum, reduce_sum_shared);
  const std::uint16_t target = targets[row];
  const float target_logit = bf16_bits_to_f32_device(row_logits[static_cast<std::int64_t>(target)]);

  if (threadIdx.x == 0 && loss_bins != nullptr && loss_bin_count > 0) {
    const float loss = logf(row_denom) + row_max - target_logit;
    atomicAdd(loss_bins + (row % loss_bin_count), loss);
  }

  if (vec_stores || vec_normal_stores) {
    constexpr std::int64_t kVec = 8;
    const std::int64_t aligned_vocab = vocab & ~(kVec - 1);
    for (std::int64_t col = static_cast<std::int64_t>(threadIdx.x) * kVec;
         col < aligned_vocab;
         col += static_cast<std::int64_t>(blockDim.x) * kVec) {
      std::uint16_t grad[8];
      const int4 packed = vec_loads ? load_bf16_vec8(row_logits + col) : make_int4(0, 0, 0, 0);
#pragma unroll
      for (int offset = 0; offset < 8; ++offset) {
        const std::int64_t current_col = col + offset;
        const std::uint16_t raw = vec_loads ? int4_u16_at(packed, offset) : row_logits[current_col];
        const float value = bf16_bits_to_f32_device(raw);
        const float prob = cross_entropy_exp_device(value - row_max, use_exp2) / row_denom;
        const float onehot = current_col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
        grad[offset] = f32_to_bf16_bits_device((prob - onehot) * loss_scale);
      }
      store_bf16_vec8(row_logits + col, grad, vec_stores);
    }
    for (std::int64_t col = aligned_vocab + threadIdx.x; col < vocab; col += blockDim.x) {
      const float value = bf16_bits_to_f32_device(row_logits[col]);
      const float prob = cross_entropy_exp_device(value - row_max, use_exp2) / row_denom;
      const float onehot = col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
      store_bf16_scalar(
          row_logits + col,
          f32_to_bf16_bits_device((prob - onehot) * loss_scale),
          scalar_streaming_stores);
    }
  } else {
    for (std::int64_t col = threadIdx.x; col < vocab; col += blockDim.x) {
      const float value = bf16_bits_to_f32_device(row_logits[col]);
      const float prob = cross_entropy_exp_device(value - row_max, use_exp2) / row_denom;
      const float onehot = col == static_cast<std::int64_t>(target) ? 1.0f : 0.0f;
      store_bf16_scalar(
          row_logits + col,
          f32_to_bf16_bits_device((prob - onehot) * loss_scale),
          scalar_streaming_stores);
    }
  }
}

__tile_global__ void token_cross_entropy_backward_chunked_float32_kernel(
    const float* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    const float* __restrict__ row_max,
    const float* __restrict__ row_denom,
    float* __restrict__ grad_logits,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t chunks_per_row,
    float loss_scale) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logits = ct::assume_aligned(logits, 16_ic);
  targets = ct::assume_aligned(targets, 16_ic);
  row_max = ct::assume_aligned(row_max, 16_ic);
  row_denom = ct::assume_aligned(row_denom, 16_ic);
  grad_logits = ct::assume_aligned(grad_logits, 16_ic);

  const std::int64_t block = static_cast<std::int64_t>(ct::bid().x);
  const std::int64_t row = block / chunks_per_row;
  const std::int64_t chunk = block - row * chunks_per_row;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto col = ct::iota<IndexTile>() + ct::full<IndexTile>(chunk * kTileSize);
  auto active = (row < rows) && (col < ct::full<IndexTile>(vocab));
  auto base = ct::full<IndexTile>(row * vocab);
  auto target = ct::full<IndexTile>(row < rows ? targets[row] : 0);
  auto maxv = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(row < rows ? row_max[row] : 0.0f);
  auto denom = ct::full<decltype(maxv)>(row < rows ? row_denom[row] : 1.0f);
  auto value = ct::load_masked(logits + base + col, active);
  auto prob = ct::exp(value - maxv) / denom;
  auto onehot = ct::select(col == target, ct::full<decltype(prob)>(1.0f), ct::full<decltype(prob)>(0.0f));
  auto grad = (prob - onehot) * ct::full<decltype(prob)>(loss_scale);
  ct::store_masked(grad_logits + base + col, grad, active);
}

__tile_global__ void token_cross_entropy_backward_chunked_inplace_float32_kernel(
    float* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    const float* __restrict__ row_max,
    const float* __restrict__ row_denom,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t chunks_per_row,
    float loss_scale) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logits = ct::assume_aligned(logits, 16_ic);
  targets = ct::assume_aligned(targets, 16_ic);
  row_max = ct::assume_aligned(row_max, 16_ic);
  row_denom = ct::assume_aligned(row_denom, 16_ic);

  const std::int64_t block = static_cast<std::int64_t>(ct::bid().x);
  const std::int64_t row = block / chunks_per_row;
  const std::int64_t chunk = block - row * chunks_per_row;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto col = ct::iota<IndexTile>() + ct::full<IndexTile>(chunk * kTileSize);
  auto active = (row < rows) && (col < ct::full<IndexTile>(vocab));
  auto base = ct::full<IndexTile>(row * vocab);
  auto target = ct::full<IndexTile>(row < rows ? targets[row] : 0);
  auto maxv = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(row < rows ? row_max[row] : 0.0f);
  auto denom = ct::full<decltype(maxv)>(row < rows ? row_denom[row] : 1.0f);
  auto value = ct::load_masked(logits + base + col, active);
  auto prob = ct::exp(value - maxv) / denom;
  auto onehot = ct::select(col == target, ct::full<decltype(prob)>(1.0f), ct::full<decltype(prob)>(0.0f));
  auto grad = (prob - onehot) * ct::full<decltype(prob)>(loss_scale);
  ct::store_masked(logits + base + col, grad, active);
}

__tile_global__ void token_cross_entropy_backward_rowwise_float32_kernel(
    const float* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    float* __restrict__ grad_logits,
    std::int64_t rows,
    std::int64_t vocab,
    float loss_scale) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logits = ct::assume_aligned(logits, 16_ic);
  targets = ct::assume_aligned(targets, 16_ic);
  grad_logits = ct::assume_aligned(grad_logits, 16_ic);

  const std::int64_t row = static_cast<std::int64_t>(ct::bid().x);
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto col = ct::iota<IndexTile>();
  auto active = (row < rows) && (col < ct::full<IndexTile>(vocab));
  auto base = ct::full<IndexTile>(row * vocab);
  auto target = ct::full<IndexTile>(targets[row]);
  auto value = ct::load_masked(logits + base + col, active);
  auto neg_inf = ct::full<decltype(value)>(-3.4028234663852886e38f);
  auto safe_value = ct::select(active, value, neg_inf);
  auto maxv = ct::reduce_max(safe_value, 0_ic);
  auto exp_value = ct::select(active, ct::exp(value - maxv), ct::full<decltype(value)>(0.0f));
  auto denom = ct::sum(exp_value, 0_ic);
  auto prob = exp_value / denom;
  auto onehot = ct::select(col == target, ct::full<decltype(prob)>(1.0f), ct::full<decltype(prob)>(0.0f));
  auto grad = (prob - onehot) * ct::full<decltype(prob)>(loss_scale);
  ct::store_masked(grad_logits + base + col, grad, active);
}

__tile_global__ void token_cross_entropy_backward_rowwise_inplace_float32_kernel(
    float* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    std::int64_t rows,
    std::int64_t vocab,
    float loss_scale) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logits = ct::assume_aligned(logits, 16_ic);
  targets = ct::assume_aligned(targets, 16_ic);

  const std::int64_t row = static_cast<std::int64_t>(ct::bid().x);
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto col = ct::iota<IndexTile>();
  auto active = (row < rows) && (col < ct::full<IndexTile>(vocab));
  auto base = ct::full<IndexTile>(row * vocab);
  auto target = ct::full<IndexTile>(targets[row]);
  auto value = ct::load_masked(logits + base + col, active);
  auto neg_inf = ct::full<decltype(value)>(-3.4028234663852886e38f);
  auto safe_value = ct::select(active, value, neg_inf);
  auto maxv = ct::reduce_max(safe_value, 0_ic);
  auto exp_value = ct::select(active, ct::exp(value - maxv), ct::full<decltype(value)>(0.0f));
  auto denom = ct::sum(exp_value, 0_ic);
  auto prob = exp_value / denom;
  auto onehot = ct::select(col == target, ct::full<decltype(prob)>(1.0f), ct::full<decltype(prob)>(0.0f));
  auto grad = (prob - onehot) * ct::full<decltype(prob)>(loss_scale);
  ct::store_masked(logits + base + col, grad, active);
}

__tile_global__ void masked_token_cross_entropy_row_stats_float32_kernel(
    const float* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    float* __restrict__ row_max_out,
    float* __restrict__ row_denom_out,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t ignore_index) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logits = ct::assume_aligned(logits, 16_ic);
  targets = ct::assume_aligned(targets, 16_ic);
  row_max_out = ct::assume_aligned(row_max_out, 16_ic);
  row_denom_out = ct::assume_aligned(row_denom_out, 16_ic);

  const std::int64_t row = static_cast<std::int64_t>(ct::bid().x);
  if (row >= rows) {
    return;
  }
  const std::int64_t target_scalar = targets[row];
  const bool valid_scalar = target_scalar != ignore_index;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  using ScalarTile = ct::tile<float, decltype(ct::shape{1_ic})>;
  auto local_col = ct::iota<IndexTile>();
  auto row_max = ct::full<ScalarTile>(-3.4028234663852886e38f);
  for (std::int64_t start = 0; start < vocab; start += kTileSize) {
    auto col = local_col + ct::full<IndexTile>(start);
    auto in_vocab = col < ct::full<IndexTile>(vocab);
    auto active = valid_scalar ? in_vocab : ct::full<decltype(in_vocab)>(false);
    auto value = ct::load_masked(logits + ct::full<IndexTile>(row * vocab) + col, active);
    auto neg_inf = ct::full<decltype(value)>(-3.4028234663852886e38f);
    auto safe_value = ct::select(active, value, neg_inf);
    auto chunk_max = ct::reduce_max(safe_value, 0_ic);
    row_max = ct::select(chunk_max > row_max, chunk_max, row_max);
  }
  auto denom = ct::full<ScalarTile>(valid_scalar ? 0.0f : 1.0f);
  for (std::int64_t start = 0; start < vocab; start += kTileSize) {
    auto col = local_col + ct::full<IndexTile>(start);
    auto in_vocab = col < ct::full<IndexTile>(vocab);
    auto active = valid_scalar ? in_vocab : ct::full<decltype(in_vocab)>(false);
    auto value = ct::load_masked(logits + ct::full<IndexTile>(row * vocab) + col, active);
    auto exp_value = ct::select(active, ct::exp(value - row_max), ct::full<decltype(value)>(0.0f));
    denom = denom + ct::sum(exp_value, 0_ic);
  }
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(row);
  auto safe_max = valid_scalar ? row_max : ct::full<ScalarTile>(0.0f);
  ct::store(row_max_out + out_idx, safe_max);
  ct::store(row_denom_out + out_idx, denom);
}

__tile_global__ void masked_token_cross_entropy_backward_chunked_float32_kernel(
    const float* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    const float* __restrict__ loss_mask,
    const float* __restrict__ row_max,
    const float* __restrict__ row_denom,
    float* __restrict__ grad_logits,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t ignore_index,
    std::int64_t chunks_per_row,
    float loss_scale) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logits = ct::assume_aligned(logits, 16_ic);
  targets = ct::assume_aligned(targets, 16_ic);
  loss_mask = ct::assume_aligned(loss_mask, 16_ic);
  row_max = ct::assume_aligned(row_max, 16_ic);
  row_denom = ct::assume_aligned(row_denom, 16_ic);
  grad_logits = ct::assume_aligned(grad_logits, 16_ic);

  const std::int64_t block = static_cast<std::int64_t>(ct::bid().x);
  const std::int64_t row = block / chunks_per_row;
  const std::int64_t chunk = block - row * chunks_per_row;
  const std::int64_t target_scalar = row < rows ? targets[row] : ignore_index;
  const bool valid_scalar = row < rows && target_scalar != ignore_index;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto col = ct::iota<IndexTile>() + ct::full<IndexTile>(chunk * kTileSize);
  auto active = (row < rows) && (col < ct::full<IndexTile>(vocab));
  auto valid = valid_scalar ? active : ct::full<decltype(active)>(false);
  auto base = ct::full<IndexTile>(row * vocab);
  auto target = ct::full<IndexTile>(valid_scalar ? target_scalar : 0);
  auto maxv = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(row < rows ? row_max[row] : 0.0f);
  auto denom = ct::full<decltype(maxv)>(row < rows ? row_denom[row] : 1.0f);
  auto value = ct::load_masked(logits + base + col, valid);
  auto prob = ct::exp(value - maxv) / denom;
  auto onehot = ct::select(col == target, ct::full<decltype(prob)>(1.0f), ct::full<decltype(prob)>(0.0f));
  auto mask_value = ct::full<decltype(prob)>(valid_scalar ? loss_mask[row] : 0.0f);
  auto grad = (prob - onehot) * mask_value * ct::full<decltype(prob)>(loss_scale);
  grad = ct::select(valid, grad, ct::full<decltype(grad)>(0.0f));
  ct::store_masked(grad_logits + base + col, grad, active);
}

__tile_global__ void masked_token_cross_entropy_backward_rowwise_float32_kernel(
    const float* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    const float* __restrict__ loss_mask,
    float* __restrict__ grad_logits,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t ignore_index,
    float loss_scale) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logits = ct::assume_aligned(logits, 16_ic);
  targets = ct::assume_aligned(targets, 16_ic);
  loss_mask = ct::assume_aligned(loss_mask, 16_ic);
  grad_logits = ct::assume_aligned(grad_logits, 16_ic);

  const std::int64_t row = static_cast<std::int64_t>(ct::bid().x);
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto col = ct::iota<IndexTile>();
  auto active = (row < rows) && (col < ct::full<IndexTile>(vocab));
  auto base = ct::full<IndexTile>(row * vocab);
  const std::int64_t target_scalar = targets[row];
  const bool valid_scalar = row < rows && target_scalar != ignore_index;
  auto valid = active && ct::full<IndexTile>(valid_scalar);
  auto safe_target = ct::full<IndexTile>(valid_scalar ? target_scalar : 0);
  auto value = ct::load_masked(logits + base + col, valid);
  auto neg_inf = ct::full<decltype(value)>(-3.4028234663852886e38f);
  auto safe_value = ct::select(valid, value, neg_inf);
  auto maxv = ct::reduce_max(safe_value, 0_ic);
  auto exp_value = ct::select(valid, ct::exp(value - maxv), ct::full<decltype(value)>(0.0f));
  auto denom = ct::sum(exp_value, 0_ic);
  auto denom_safe = denom + ct::full<decltype(denom)>(valid_scalar ? 0.0f : 1.0f);
  auto prob = exp_value / denom_safe;
  auto onehot = ct::select(col == safe_target, ct::full<decltype(prob)>(1.0f), ct::full<decltype(prob)>(0.0f));
  auto mask_value = ct::full<decltype(prob)>(valid_scalar ? loss_mask[row] : 0.0f);
  auto grad = (prob - onehot) * mask_value * ct::full<decltype(prob)>(loss_scale);
  grad = ct::select(valid, grad, ct::full<decltype(grad)>(0.0f));
  ct::store_masked(grad_logits + base + col, grad, active);
}

__tile_global__ void sequence_logp_float32_kernel(
    const float* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    const float* __restrict__ loss_mask,
    float* __restrict__ out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t vocab,
    std::int64_t ignore_index) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logits = ct::assume_aligned(logits, 16_ic);
  targets = ct::assume_aligned(targets, 16_ic);
  loss_mask = ct::assume_aligned(loss_mask, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto batch_idx = ct::iota<IndexTile>();
  auto active = batch_idx < ct::full<IndexTile>(batch);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto vocab_tile = ct::full<IndexTile>(vocab);
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t step = 0; step < seq_len; ++step) {
    auto row = batch_idx * seq_tile + ct::full<IndexTile>(step);
    auto target = ct::load_masked(targets + row, active);
    auto valid = active && (target != ct::full<IndexTile>(ignore_index));
    auto safe_target = ct::select(valid, target, ct::full<IndexTile>(0));
    auto maxv = ct::full<decltype(acc)>(-3.4028234663852886e38f);
    for (std::int64_t col = 0; col < vocab; ++col) {
      auto value = ct::load_masked(logits + row * vocab_tile + ct::full<IndexTile>(col), active);
      maxv = ct::select(value > maxv, value, maxv);
    }
    auto denom = ct::full<decltype(acc)>(0.0f);
    for (std::int64_t col = 0; col < vocab; ++col) {
      auto value = ct::load_masked(logits + row * vocab_tile + ct::full<IndexTile>(col), active);
      denom = denom + ct::exp(value - maxv);
    }
    auto target_value = ct::load_masked(logits + row * vocab_tile + safe_target, active);
    auto logp = target_value - maxv - ct::log(denom);
    auto mask_value = ct::load_masked(loss_mask + row, active);
    acc = acc + ct::select(valid, logp * mask_value, ct::full<decltype(acc)>(0.0f));
  }
  ct::store_masked(out + batch_idx, acc, active);
}

__tile_global__ void preference_bce_partials_float32_kernel(
    const float* __restrict__ reward_chosen,
    const float* __restrict__ reward_rejected,
    float* __restrict__ partials,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  reward_chosen = ct::assume_aligned(reward_chosen, 16_ic);
  reward_rejected = ct::assume_aligned(reward_rejected, 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto chosen = ct::load_masked(reward_chosen + idx, mask);
  auto rejected = ct::load_masked(reward_rejected + idx, mask);
  auto zero = ct::full<decltype(chosen)>(0.0f);
  auto diff = chosen - rejected;
  auto loss = ct::log(ct::full<decltype(diff)>(1.0f) + ct::exp(-diff));
  loss = ct::select(mask, loss, zero);
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(partials + out_idx, ct::sum(loss, 0_ic));
}

__tile_global__ void ppo_clipped_loss_partials_float32_kernel(
    const float* __restrict__ logp_new,
    const float* __restrict__ logp_old,
    const float* __restrict__ advantages,
    const float* __restrict__ value_new,
    const float* __restrict__ value_old,
    const float* __restrict__ returns,
    float* __restrict__ policy_partials,
    float* __restrict__ value_partials,
    std::int64_t n,
    float clip_range) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logp_new = ct::assume_aligned(logp_new, 16_ic);
  logp_old = ct::assume_aligned(logp_old, 16_ic);
  advantages = ct::assume_aligned(advantages, 16_ic);
  value_new = ct::assume_aligned(value_new, 16_ic);
  value_old = ct::assume_aligned(value_old, 16_ic);
  returns = ct::assume_aligned(returns, 16_ic);
  policy_partials = ct::assume_aligned(policy_partials, 16_ic);
  value_partials = ct::assume_aligned(value_partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto new_logp = ct::load_masked(logp_new + idx, mask);
  auto old_logp = ct::load_masked(logp_old + idx, mask);
  auto adv = ct::load_masked(advantages + idx, mask);
  auto v_new = ct::load_masked(value_new + idx, mask);
  auto v_old = ct::load_masked(value_old + idx, mask);
  auto ret = ct::load_masked(returns + idx, mask);
  auto zero = ct::full<decltype(new_logp)>(0.0f);
  auto one = ct::full<decltype(new_logp)>(1.0f);
  auto clip = ct::full<decltype(new_logp)>(clip_range);

  auto ratio = ct::exp(new_logp - old_logp);
  auto ratio_clipped = ct::select(ratio > one + clip, one + clip, ct::select(ratio < one - clip, one - clip, ratio));
  auto unclipped = ratio * adv;
  auto clipped = ratio_clipped * adv;
  auto policy = -ct::select(unclipped < clipped, unclipped, clipped);

  auto delta = v_new - v_old;
  auto delta_clipped = ct::select(delta > clip, clip, ct::select(delta < -clip, -clip, delta));
  auto v_clipped = v_old + delta_clipped;
  auto vf_sq1 = (v_new - ret) * (v_new - ret);
  auto vf_sq2 = (v_clipped - ret) * (v_clipped - ret);
  auto value = ct::full<decltype(vf_sq1)>(0.5f) * ct::select(vf_sq1 > vf_sq2, vf_sq1, vf_sq2);

  policy = ct::select(mask, policy, zero);
  value = ct::select(mask, value, zero);
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(policy_partials + out_idx, ct::sum(policy, 0_ic));
  ct::store(value_partials + out_idx, ct::sum(value, 0_ic));
}

__tile_global__ void gae_compute_float32_kernel(
    const float* __restrict__ rewards,
    const float* __restrict__ values,
    float* __restrict__ advantages,
    float* __restrict__ returns,
    std::int64_t seq_len,
    float gamma,
    float lambda_value) {
  namespace ct = cuda::tiles;

  const int batch = ct::bid().x;
  const std::int64_t base = static_cast<std::int64_t>(batch) * seq_len;
  float next_adv = 0.0f;
  float next_value = 0.0f;
  for (std::int64_t t = seq_len - 1; t >= 0; --t) {
    const std::int64_t idx = base + t;
    const float delta = rewards[idx] + gamma * next_value - values[idx];
    next_adv = delta + gamma * lambda_value * next_adv;
    advantages[idx] = next_adv;
    returns[idx] = next_adv + values[idx];
    next_value = values[idx];
  }
}

__tile_global__ void dpo_pairwise_partials_float32_kernel(
    const float* __restrict__ policy_logp_chosen,
    const float* __restrict__ policy_logp_rejected,
    const float* __restrict__ ref_logp_chosen,
    const float* __restrict__ ref_logp_rejected,
    float* __restrict__ partials,
    float* __restrict__ chosen_reward_out,
    float* __restrict__ rejected_reward_out,
    std::int64_t n,
    float beta,
    float label_smoothing,
    int loss_type) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  policy_logp_chosen = ct::assume_aligned(policy_logp_chosen, 16_ic);
  policy_logp_rejected = ct::assume_aligned(policy_logp_rejected, 16_ic);
  ref_logp_chosen = ct::assume_aligned(ref_logp_chosen, 16_ic);
  ref_logp_rejected = ct::assume_aligned(ref_logp_rejected, 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);
  chosen_reward_out = ct::assume_aligned(chosen_reward_out, 16_ic);
  rejected_reward_out = ct::assume_aligned(rejected_reward_out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto policy_chosen = ct::load_masked(policy_logp_chosen + idx, mask);
  auto policy_rejected = ct::load_masked(policy_logp_rejected + idx, mask);
  auto ref_chosen = ct::load_masked(ref_logp_chosen + idx, mask);
  auto ref_rejected = ct::load_masked(ref_logp_rejected + idx, mask);
  auto beta_tile = ct::full<decltype(policy_chosen)>(beta);
  auto one = ct::full<decltype(policy_chosen)>(1.0f);
  auto zero = ct::full<decltype(policy_chosen)>(0.0f);
  auto chosen_logratio = policy_chosen - ref_chosen;
  auto rejected_logratio = policy_rejected - ref_rejected;
  auto logits = beta_tile * (chosen_logratio - rejected_logratio);
  auto loss = ct::log(one + ct::exp(-logits));
  if (loss_type == 1) {
    loss = ct::select(one - logits > zero, one - logits, zero);
  } else if (loss_type == 2) {
    auto target = ct::full<decltype(policy_chosen)>(1.0f / (2.0f * beta));
    auto delta = logits - target;
    loss = delta * delta;
  } else if (label_smoothing > 0.0f) {
    auto smoothing = ct::full<decltype(policy_chosen)>(label_smoothing);
    loss = loss * (one - smoothing) + ct::log(one + ct::exp(logits)) * smoothing;
  }
  loss = ct::select(mask, loss, zero);
  ct::store_masked(chosen_reward_out + idx, beta_tile * chosen_logratio, mask);
  ct::store_masked(rejected_reward_out + idx, beta_tile * rejected_logratio, mask);
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(partials + out_idx, ct::sum(loss, 0_ic));
}

__tile_global__ void route_selection_loss_partials_float32_kernel(
    const float* __restrict__ route_logits,
    const std::int64_t* __restrict__ sem_targets,
    float* __restrict__ loss_partials,
    float* __restrict__ count_partials,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t experts,
    std::int64_t num_vocab_dims,
    std::int64_t shared_experts,
    std::int64_t ignore_index) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  route_logits = ct::assume_aligned(route_logits, 16_ic);
  sem_targets = ct::assume_aligned(sem_targets, 16_ic);
  loss_partials = ct::assume_aligned(loss_partials, 16_ic);
  count_partials = ct::assume_aligned(count_partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto dims_tile = ct::full<IndexTile>(num_vocab_dims);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto experts_tile = ct::full<IndexTile>(experts);
  auto shared_tile = ct::full<IndexTile>(shared_experts);
  auto d = idx % dims_tile;
  auto s = (idx / dims_tile) % seq_tile;
  auto b = idx / (dims_tile * seq_tile);
  auto target = ct::load_masked(sem_targets + b * dims_tile + d, mask);
  auto valid = mask & (target != ct::full<IndexTile>(ignore_index));
  auto logit = ct::load_masked(route_logits + (b * seq_tile + s) * experts_tile + shared_tile + d, valid);
  auto zero = ct::full<decltype(logit)>(0.0f);
  auto one = ct::full<decltype(logit)>(1.0f);
  auto loss = ct::select(valid, ct::log(one + ct::exp(-logit)), zero);
  auto count = ct::select(valid, one, zero);
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(loss_partials + out_idx, ct::sum(loss, 0_ic));
  ct::store(count_partials + out_idx, ct::sum(count, 0_ic));
}

__tile_global__ void route_balance_density_float32_kernel(
    const float* __restrict__ route_logits,
    float* __restrict__ density,
    std::int64_t rows,
    std::int64_t experts) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  route_logits = ct::assume_aligned(route_logits, 16_ic);
  density = ct::assume_aligned(density, 16_ic);

  const int expert = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto row = ct::iota<IndexTile>();
  auto mask = row < ct::full<IndexTile>(rows);
  auto maxv = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(-3.4028234663852886e38f);
  for (std::int64_t col = 0; col < experts; ++col) {
    auto value = ct::load_masked(route_logits + row * ct::full<IndexTile>(experts) + ct::full<IndexTile>(col), mask);
    maxv = ct::select(value > maxv, value, maxv);
  }
  auto denom = ct::full<decltype(maxv)>(0.0f);
  for (std::int64_t col = 0; col < experts; ++col) {
    auto value = ct::load_masked(route_logits + row * ct::full<IndexTile>(experts) + ct::full<IndexTile>(col), mask);
    denom = denom + ct::exp(value - maxv);
  }
  auto selected = ct::load_masked(route_logits + row * ct::full<IndexTile>(experts) + ct::full<IndexTile>(expert), mask);
  auto prob = ct::exp(selected - maxv) / denom;
  prob = ct::select(mask, prob, ct::full<decltype(prob)>(0.0f));
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(expert));
  auto scale = ct::full<ct::tile<float, decltype(ct::shape{1_ic})>>(1.0f / static_cast<float>(rows));
  ct::store(density + out_idx, ct::sum(prob, 0_ic) * scale);
}

__tile_global__ void route_balance_loss_float32_kernel(
    const float* __restrict__ density,
    float* __restrict__ out,
    std::int64_t experts) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  density = ct::assume_aligned(density, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>();
  auto mask = idx < ct::full<IndexTile>(experts);
  auto value = ct::load_masked(density + idx, mask);
  auto sq = ct::select(mask, value * value, ct::full<decltype(value)>(0.0f));
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(0);
  auto scale = ct::full<ct::tile<float, decltype(ct::shape{1_ic})>>(static_cast<float>(experts));
  ct::store(out + out_idx, ct::sum(sq, 0_ic) * scale);
}

__tile_global__ void softmax_distillation_partials_float32_kernel(
    const float* __restrict__ teacher_logits,
    const float* __restrict__ student_logits,
    float* __restrict__ partials,
    std::int64_t rows,
    std::int64_t vocab) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  teacher_logits = ct::assume_aligned(teacher_logits, 16_ic);
  student_logits = ct::assume_aligned(student_logits, 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);

  const int row_id = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto col = ct::iota<IndexTile>();
  auto mask = col < ct::full<IndexTile>(vocab);
  auto base = ct::full<IndexTile>(static_cast<std::int64_t>(row_id)) * ct::full<IndexTile>(vocab);
  auto teacher = ct::load_masked(teacher_logits + base + col, mask);
  auto student = ct::load_masked(student_logits + base + col, mask);
  auto teacher_exp = ct::select(mask, ct::exp(teacher), ct::full<decltype(teacher)>(0.0f));
  auto student_exp = ct::select(mask, ct::exp(student), ct::full<decltype(student)>(0.0f));
  auto teacher_logsum = ct::log(ct::sum(teacher_exp, 0_ic));
  auto student_logsum = ct::log(ct::sum(student_exp, 0_ic));
  auto teacher_prob = ct::exp(teacher - teacher_logsum);
  auto kl = teacher_prob * ((teacher - teacher_logsum) - (student - student_logsum));
  kl = ct::select(mask, kl, ct::full<decltype(kl)>(0.0f));
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(row_id));
  ct::store(partials + out_idx, ct::sum(kl, 0_ic));
}

__tile_global__ void scaled_dot_product_attention_float32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  q = ct::assume_aligned(q, 16_ic);
  k = ct::assume_aligned(k, 16_ic);
  v = ct::assume_aligned(v, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const std::int64_t out_idx_scalar = static_cast<std::int64_t>(ct::bid().x);
  if (out_idx_scalar >= n) {
    return;
  }

  const std::int64_t d_out = out_idx_scalar % value_dim;
  const std::int64_t q_pos = (out_idx_scalar / value_dim) % seq_q;
  const std::int64_t q_head = (out_idx_scalar / (value_dim * seq_q)) % query_heads;
  const std::int64_t batch = out_idx_scalar / (value_dim * seq_q * query_heads);
  const std::int64_t k_head = (q_head * key_heads) / query_heads;
  const std::int64_t q_base_scalar = ((batch * query_heads + q_head) * seq_q + q_pos) * qk_dim;
  const std::int64_t k_base_scalar = (batch * key_heads + k_head) * seq_k * qk_dim;
  const std::int64_t v_base_scalar = (batch * key_heads + k_head) * seq_k * value_dim;

  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto key_pos = ct::iota<IndexTile>();
  auto valid = key_pos < ct::full<IndexTile>(seq_k);
  if (is_causal) {
    const std::int64_t offset = right_align_causal ? (seq_k - seq_q) : 0;
    valid = valid & (key_pos <= ct::full<IndexTile>(q_pos + offset));
  }
  if (use_sparse_rules) {
    const std::int64_t offset = right_align_causal ? (seq_k - seq_q) : 0;
    auto keep = ct::full<decltype(valid)>(false);
    bool any_rule = false;
    if (window > 0) {
      keep = keep | (key_pos > ct::full<IndexTile>(q_pos + offset - window));
      any_rule = true;
    }
    if (num_sinks > 0) {
      keep = keep | (key_pos < ct::full<IndexTile>(num_sinks));
      any_rule = true;
    }
    if (block_size > 0) {
      keep = keep | (ct::full<IndexTile>((q_pos + offset) / block_size) == (key_pos / ct::full<IndexTile>(block_size)));
      any_rule = true;
    }
    if (compress_stride > 1) {
      keep = keep | ((key_pos % ct::full<IndexTile>(compress_stride)) == ct::full<IndexTile>(0));
      any_rule = true;
    }
    if (any_rule) {
      valid = valid & keep;
    }
  }

  auto score = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t d = 0; d < qk_dim; ++d) {
    auto q_val = ct::full<decltype(score)>(q[q_base_scalar + d]);
    auto k_val = ct::load_masked(k + ct::full<IndexTile>(k_base_scalar + d) + key_pos * ct::full<IndexTile>(qk_dim), valid);
    score = score + q_val * k_val;
  }
  score = score * ct::full<decltype(score)>(scale);
  auto neg_inf = ct::full<decltype(score)>(-3.4028234663852886e38f);
  auto safe_score = ct::select(valid, score, neg_inf);
  auto max_score = ct::reduce_max(safe_score, 0_ic);
  auto exp_score = ct::select(valid, ct::exp(score - max_score), ct::full<decltype(score)>(0.0f));
  auto denom = ct::sum(exp_score, 0_ic);
  auto v_value = ct::load_masked(
      v + ct::full<IndexTile>(v_base_scalar + d_out) + key_pos * ct::full<IndexTile>(value_dim),
      valid);
  auto weighted = exp_score * v_value;
  auto result = ct::sum(weighted, 0_ic) / denom;

  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(out_idx_scalar);
  ct::store(out + out_idx, result);
}

__tile_global__ void scaled_dot_product_attention_row_float32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    std::int64_t seq_len,
    float scale) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  q = ct::assume_aligned(q, 16_ic);
  k = ct::assume_aligned(k, 16_ic);
  v = ct::assume_aligned(v, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const std::int64_t row_idx_scalar = static_cast<std::int64_t>(ct::bid().x);
  const std::int64_t q_pos = row_idx_scalar % seq_len;
  const std::int64_t q_head = (row_idx_scalar / seq_len) % kGpt2AttentionHeads;
  const std::int64_t batch = row_idx_scalar / (seq_len * kGpt2AttentionHeads);
  const std::int64_t q_base_scalar = ((batch * kGpt2AttentionHeads + q_head) * seq_len + q_pos) * kGpt2AttentionHeadDim;
  const std::int64_t k_base_scalar = (batch * kGpt2AttentionHeads + q_head) * seq_len * kGpt2AttentionHeadDim;
  const std::int64_t v_base_scalar = (batch * kGpt2AttentionHeads + q_head) * seq_len * kGpt2AttentionHeadDim;
  const std::int64_t out_base_scalar = ((batch * kGpt2AttentionHeads + q_head) * seq_len + q_pos) * kGpt2AttentionHeadDim;

  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto key_pos = ct::iota<IndexTile>();
  auto valid = (key_pos < ct::full<IndexTile>(seq_len)) & (key_pos <= ct::full<IndexTile>(q_pos));

  auto score = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t d = 0; d < kGpt2AttentionHeadDim; ++d) {
    auto q_val = ct::full<decltype(score)>(q[q_base_scalar + d]);
    auto k_val = ct::load_masked(
        k + ct::full<IndexTile>(k_base_scalar + d) + key_pos * ct::full<IndexTile>(kGpt2AttentionHeadDim),
        valid);
    score = score + q_val * k_val;
  }
  score = score * ct::full<decltype(score)>(scale);
  auto neg_inf = ct::full<decltype(score)>(-3.4028234663852886e38f);
  auto safe_score = ct::select(valid, score, neg_inf);
  auto max_score = ct::reduce_max(safe_score, 0_ic);
  auto exp_score = ct::select(valid, ct::exp(score - max_score), ct::full<decltype(score)>(0.0f));
  auto denom = ct::sum(exp_score, 0_ic);

  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  for (std::int64_t d_out = 0; d_out < kGpt2AttentionHeadDim; ++d_out) {
    auto v_value = ct::load_masked(
        v + ct::full<IndexTile>(v_base_scalar + d_out) + key_pos * ct::full<IndexTile>(kGpt2AttentionHeadDim),
        valid);
    auto result = ct::sum(exp_score * v_value, 0_ic) / denom;
    auto out_idx = ct::full<OneIndexTile>(out_base_scalar + d_out);
    ct::store(out + out_idx, result);
  }
}

__tile_global__ void scaled_dot_product_attention_backward_q_float32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_q,
    std::int64_t n,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    bool grad_out_merged) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  q = ct::assume_aligned(q, 16_ic);
  k = ct::assume_aligned(k, 16_ic);
  v = ct::assume_aligned(v, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_q = ct::assume_aligned(grad_q, 16_ic);

  const std::int64_t q_idx_scalar = static_cast<std::int64_t>(ct::bid().x);
  if (q_idx_scalar >= n) {
    return;
  }

  const std::int64_t d_q = q_idx_scalar % qk_dim;
  const std::int64_t q_pos = (q_idx_scalar / qk_dim) % seq_q;
  const std::int64_t q_head = (q_idx_scalar / (qk_dim * seq_q)) % query_heads;
  const std::int64_t batch = q_idx_scalar / (qk_dim * seq_q * query_heads);
  const std::int64_t k_head = (q_head * key_heads) / query_heads;
  const std::int64_t q_base_scalar = ((batch * query_heads + q_head) * seq_q + q_pos) * qk_dim;
  const std::int64_t k_base_scalar = (batch * key_heads + k_head) * seq_k * qk_dim;
  const std::int64_t v_base_scalar = (batch * key_heads + k_head) * seq_k * value_dim;
  const std::int64_t go_base_scalar = grad_out_merged
      ? (batch * seq_q + q_pos) * query_heads * value_dim + q_head * value_dim
      : ((batch * query_heads + q_head) * seq_q + q_pos) * value_dim;

  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto key_pos = ct::iota<IndexTile>();
  auto valid = key_pos < ct::full<IndexTile>(seq_k);
  if (is_causal) {
    const std::int64_t offset = right_align_causal ? (seq_k - seq_q) : 0;
    valid = valid & (key_pos <= ct::full<IndexTile>(q_pos + offset));
  }
  if (use_sparse_rules) {
    const std::int64_t offset = right_align_causal ? (seq_k - seq_q) : 0;
    auto keep = ct::full<decltype(valid)>(false);
    bool any_rule = false;
    if (window > 0) {
      keep = keep | (key_pos > ct::full<IndexTile>(q_pos + offset - window));
      any_rule = true;
    }
    if (num_sinks > 0) {
      keep = keep | (key_pos < ct::full<IndexTile>(num_sinks));
      any_rule = true;
    }
    if (block_size > 0) {
      keep = keep | (ct::full<IndexTile>((q_pos + offset) / block_size) == (key_pos / ct::full<IndexTile>(block_size)));
      any_rule = true;
    }
    if (compress_stride > 1) {
      keep = keep | ((key_pos % ct::full<IndexTile>(compress_stride)) == ct::full<IndexTile>(0));
      any_rule = true;
    }
    if (any_rule) {
      valid = valid & keep;
    }
  }

  auto score = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t d = 0; d < qk_dim; ++d) {
    auto q_val = ct::full<decltype(score)>(q[q_base_scalar + d]);
    auto k_val = ct::load_masked(k + ct::full<IndexTile>(k_base_scalar + d) + key_pos * ct::full<IndexTile>(qk_dim), valid);
    score = score + q_val * k_val;
  }
  score = score * ct::full<decltype(score)>(scale);
  auto neg_inf = ct::full<decltype(score)>(-3.4028234663852886e38f);
  auto safe_score = ct::select(valid, score, neg_inf);
  auto max_score = ct::reduce_max(safe_score, 0_ic);
  auto exp_score = ct::select(valid, ct::exp(score - max_score), ct::full<decltype(score)>(0.0f));
  auto denom = ct::sum(exp_score, 0_ic);
  auto prob = exp_score / denom;
  auto dprob = ct::full<decltype(score)>(0.0f);
  for (std::int64_t dv = 0; dv < value_dim; ++dv) {
    auto go_val = ct::full<decltype(score)>(grad_out[go_base_scalar + dv]);
    auto v_val = ct::load_masked(v + ct::full<IndexTile>(v_base_scalar + dv) + key_pos * ct::full<IndexTile>(value_dim), valid);
    dprob = dprob + go_val * v_val;
  }
  auto dprob_mean = ct::sum(prob * dprob, 0_ic);
  auto dscore = prob * (dprob - dprob_mean);
  auto k_selected_dim = ct::load_masked(
      k + ct::full<IndexTile>(k_base_scalar + d_q) + key_pos * ct::full<IndexTile>(qk_dim),
      valid);
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(q_idx_scalar);
  auto grad = ct::sum(dscore * k_selected_dim, 0_ic) * ct::full<ct::tile<float, decltype(ct::shape{1_ic})>>(scale);
  ct::store(grad_q + out_idx, grad);
}

__tile_global__ void scaled_dot_product_attention_backward_k_float32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_k,
    std::int64_t n,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    bool grad_out_merged) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  q = ct::assume_aligned(q, 16_ic);
  k = ct::assume_aligned(k, 16_ic);
  v = ct::assume_aligned(v, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_k = ct::assume_aligned(grad_k, 16_ic);

  const std::int64_t k_idx_scalar = static_cast<std::int64_t>(ct::bid().x);
  if (k_idx_scalar >= n) {
    return;
  }

  const std::int64_t d_k = k_idx_scalar % qk_dim;
  const std::int64_t k_pos_scalar = (k_idx_scalar / qk_dim) % seq_k;
  const std::int64_t k_head_scalar = (k_idx_scalar / (qk_dim * seq_k)) % key_heads;
  const std::int64_t batch_scalar = k_idx_scalar / (qk_dim * seq_k * key_heads);
  const std::int64_t k_base_scalar = (batch_scalar * key_heads + k_head_scalar) * seq_k * qk_dim;
  const std::int64_t v_base_scalar = (batch_scalar * key_heads + k_head_scalar) * seq_k * value_dim;

  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto key_pos = ct::iota<IndexTile>();
  auto key_match = key_pos == ct::full<IndexTile>(k_pos_scalar);
  auto acc_scalar = ct::full<ct::tile<float, decltype(ct::shape{1_ic})>>(0.0f);
  for (std::int64_t q_head = 0; q_head < query_heads; ++q_head) {
    if ((q_head * key_heads) / query_heads != k_head_scalar) {
      continue;
    }
    for (std::int64_t q_pos = 0; q_pos < seq_q; ++q_pos) {
      auto valid = key_pos < ct::full<IndexTile>(seq_k);
      if (is_causal) {
        const std::int64_t offset = right_align_causal ? (seq_k - seq_q) : 0;
        valid = valid & (key_pos <= ct::full<IndexTile>(q_pos + offset));
      }
      if (use_sparse_rules) {
        const std::int64_t offset = right_align_causal ? (seq_k - seq_q) : 0;
        auto keep = ct::full<decltype(valid)>(false);
        bool any_rule = false;
        if (window > 0) {
          keep = keep | (key_pos > ct::full<IndexTile>(q_pos + offset - window));
          any_rule = true;
        }
        if (num_sinks > 0) {
          keep = keep | (key_pos < ct::full<IndexTile>(num_sinks));
          any_rule = true;
        }
        if (block_size > 0) {
          keep = keep | (ct::full<IndexTile>((q_pos + offset) / block_size) == (key_pos / ct::full<IndexTile>(block_size)));
          any_rule = true;
        }
        if (compress_stride > 1) {
          keep = keep | ((key_pos % ct::full<IndexTile>(compress_stride)) == ct::full<IndexTile>(0));
          any_rule = true;
        }
        if (any_rule) {
          valid = valid & keep;
        }
      }
      const std::int64_t q_base_scalar = ((batch_scalar * query_heads + q_head) * seq_q + q_pos) * qk_dim;
      const std::int64_t go_base_scalar = grad_out_merged
          ? (batch_scalar * seq_q + q_pos) * query_heads * value_dim + q_head * value_dim
          : ((batch_scalar * query_heads + q_head) * seq_q + q_pos) * value_dim;
      auto score = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
      for (std::int64_t d = 0; d < qk_dim; ++d) {
        auto q_val = ct::full<decltype(score)>(q[q_base_scalar + d]);
        auto k_val = ct::load_masked(k + ct::full<IndexTile>(k_base_scalar + d) + key_pos * ct::full<IndexTile>(qk_dim), valid);
        score = score + q_val * k_val;
      }
      score = score * ct::full<decltype(score)>(scale);
      auto neg_inf = ct::full<decltype(score)>(-3.4028234663852886e38f);
      auto safe_score = ct::select(valid, score, neg_inf);
      auto max_score = ct::reduce_max(safe_score, 0_ic);
      auto exp_score = ct::select(valid, ct::exp(score - max_score), ct::full<decltype(score)>(0.0f));
      auto denom = ct::sum(exp_score, 0_ic);
      auto prob = exp_score / denom;
      auto dprob = ct::full<decltype(score)>(0.0f);
      for (std::int64_t dv = 0; dv < value_dim; ++dv) {
        auto go_val = ct::full<decltype(score)>(grad_out[go_base_scalar + dv]);
        auto v_val = ct::load_masked(v + ct::full<IndexTile>(v_base_scalar + dv) + key_pos * ct::full<IndexTile>(value_dim), valid);
        dprob = dprob + go_val * v_val;
      }
      auto dprob_mean = ct::sum(prob * dprob, 0_ic);
      auto dscore = prob * (dprob - dprob_mean);
      auto selected = ct::sum(ct::select(valid & key_match, dscore, ct::full<decltype(dscore)>(0.0f)), 0_ic);
      auto q_dim = ct::full<decltype(selected)>(q[q_base_scalar + d_k]);
      acc_scalar = acc_scalar + selected * q_dim * ct::full<decltype(selected)>(scale);
    }
  }
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(k_idx_scalar);
  ct::store(grad_k + out_idx, acc_scalar);
}

__tile_global__ void scaled_dot_product_attention_backward_v_float32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_v,
    std::int64_t n,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    bool grad_out_merged) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  q = ct::assume_aligned(q, 16_ic);
  k = ct::assume_aligned(k, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_v = ct::assume_aligned(grad_v, 16_ic);

  const std::int64_t v_idx_scalar = static_cast<std::int64_t>(ct::bid().x);
  if (v_idx_scalar >= n) {
    return;
  }

  const std::int64_t d_v = v_idx_scalar % value_dim;
  const std::int64_t k_pos_scalar = (v_idx_scalar / value_dim) % seq_k;
  const std::int64_t k_head_scalar = (v_idx_scalar / (value_dim * seq_k)) % key_heads;
  const std::int64_t batch_scalar = v_idx_scalar / (value_dim * seq_k * key_heads);
  const std::int64_t k_base_scalar = (batch_scalar * key_heads + k_head_scalar) * seq_k * qk_dim;

  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto key_pos = ct::iota<IndexTile>();
  auto key_match = key_pos == ct::full<IndexTile>(k_pos_scalar);
  auto acc_scalar = ct::full<ct::tile<float, decltype(ct::shape{1_ic})>>(0.0f);
  for (std::int64_t q_head = 0; q_head < query_heads; ++q_head) {
    if ((q_head * key_heads) / query_heads != k_head_scalar) {
      continue;
    }
    for (std::int64_t q_pos = 0; q_pos < seq_q; ++q_pos) {
      auto valid = key_pos < ct::full<IndexTile>(seq_k);
      if (is_causal) {
        const std::int64_t offset = right_align_causal ? (seq_k - seq_q) : 0;
        valid = valid & (key_pos <= ct::full<IndexTile>(q_pos + offset));
      }
      if (use_sparse_rules) {
        const std::int64_t offset = right_align_causal ? (seq_k - seq_q) : 0;
        auto keep = ct::full<decltype(valid)>(false);
        bool any_rule = false;
        if (window > 0) {
          keep = keep | (key_pos > ct::full<IndexTile>(q_pos + offset - window));
          any_rule = true;
        }
        if (num_sinks > 0) {
          keep = keep | (key_pos < ct::full<IndexTile>(num_sinks));
          any_rule = true;
        }
        if (block_size > 0) {
          keep = keep | (ct::full<IndexTile>((q_pos + offset) / block_size) == (key_pos / ct::full<IndexTile>(block_size)));
          any_rule = true;
        }
        if (compress_stride > 1) {
          keep = keep | ((key_pos % ct::full<IndexTile>(compress_stride)) == ct::full<IndexTile>(0));
          any_rule = true;
        }
        if (any_rule) {
          valid = valid & keep;
        }
      }
      const std::int64_t q_base_scalar = ((batch_scalar * query_heads + q_head) * seq_q + q_pos) * qk_dim;
      const std::int64_t go_base_scalar = grad_out_merged
          ? (batch_scalar * seq_q + q_pos) * query_heads * value_dim + q_head * value_dim
          : ((batch_scalar * query_heads + q_head) * seq_q + q_pos) * value_dim;
      auto score = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
      for (std::int64_t d = 0; d < qk_dim; ++d) {
        auto q_val = ct::full<decltype(score)>(q[q_base_scalar + d]);
        auto k_val = ct::load_masked(k + ct::full<IndexTile>(k_base_scalar + d) + key_pos * ct::full<IndexTile>(qk_dim), valid);
        score = score + q_val * k_val;
      }
      score = score * ct::full<decltype(score)>(scale);
      auto neg_inf = ct::full<decltype(score)>(-3.4028234663852886e38f);
      auto safe_score = ct::select(valid, score, neg_inf);
      auto max_score = ct::reduce_max(safe_score, 0_ic);
      auto exp_score = ct::select(valid, ct::exp(score - max_score), ct::full<decltype(score)>(0.0f));
      auto denom = ct::sum(exp_score, 0_ic);
      auto prob = exp_score / denom;
      auto selected_prob = ct::sum(ct::select(valid & key_match, prob, ct::full<decltype(prob)>(0.0f)), 0_ic);
      auto go_dim = ct::full<decltype(selected_prob)>(grad_out[go_base_scalar + d_v]);
      acc_scalar = acc_scalar + selected_prob * go_dim;
    }
  }
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(v_idx_scalar);
  ct::store(grad_v + out_idx, acc_scalar);
}

__tile_global__ void zero_three_float32_kernel(
    float* __restrict__ a,
    float* __restrict__ b,
    float* __restrict__ c,
    std::int64_t n_a,
    std::int64_t n_b,
    std::int64_t n_c) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  a = ct::assume_aligned(a, 16_ic);
  b = ct::assume_aligned(b, 16_ic);
  c = ct::assume_aligned(c, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto zero = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto mask_a = idx < ct::full<IndexTile>(n_a);
  auto idx_b = idx - ct::full<IndexTile>(n_a);
  auto mask_b = (idx >= ct::full<IndexTile>(n_a)) & (idx_b < ct::full<IndexTile>(n_b));
  auto idx_c = idx_b - ct::full<IndexTile>(n_b);
  auto mask_c = (idx >= ct::full<IndexTile>(n_a + n_b)) & (idx_c < ct::full<IndexTile>(n_c));
  ct::store_masked(a + idx, zero, mask_a);
  ct::store_masked(b + idx_b, zero, mask_b);
  ct::store_masked(c + idx_c, zero, mask_c);
}

__tile_global__ void scaled_dot_product_attention_backward_query_row_atomic_float32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_q,
    float* __restrict__ grad_k,
    float* __restrict__ grad_v,
    std::int64_t rows,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    bool grad_out_merged) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  q = ct::assume_aligned(q, 16_ic);
  k = ct::assume_aligned(k, 16_ic);
  v = ct::assume_aligned(v, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_q = ct::assume_aligned(grad_q, 16_ic);
  grad_k = ct::assume_aligned(grad_k, 16_ic);
  grad_v = ct::assume_aligned(grad_v, 16_ic);

  const std::int64_t row_idx_scalar = static_cast<std::int64_t>(ct::bid().x);
  if (row_idx_scalar >= rows) {
    return;
  }

  const std::int64_t q_pos = row_idx_scalar % seq_q;
  const std::int64_t q_head = (row_idx_scalar / seq_q) % query_heads;
  const std::int64_t batch = row_idx_scalar / (seq_q * query_heads);
  const std::int64_t k_head = (q_head * key_heads) / query_heads;
  const std::int64_t q_base_scalar = ((batch * query_heads + q_head) * seq_q + q_pos) * qk_dim;
  const std::int64_t k_base_scalar = (batch * key_heads + k_head) * seq_k * qk_dim;
  const std::int64_t v_base_scalar = (batch * key_heads + k_head) * seq_k * value_dim;
  const std::int64_t go_base_scalar = grad_out_merged
      ? (batch * seq_q + q_pos) * query_heads * value_dim + q_head * value_dim
      : ((batch * query_heads + q_head) * seq_q + q_pos) * value_dim;

  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto key_pos = ct::iota<IndexTile>();
  auto valid = key_pos < ct::full<IndexTile>(seq_k);
  if (is_causal) {
    const std::int64_t offset = right_align_causal ? (seq_k - seq_q) : 0;
    valid = valid & (key_pos <= ct::full<IndexTile>(q_pos + offset));
  }
  if (use_sparse_rules) {
    const std::int64_t offset = right_align_causal ? (seq_k - seq_q) : 0;
    auto keep = ct::full<decltype(valid)>(false);
    bool any_rule = false;
    if (window > 0) {
      keep = keep | (key_pos > ct::full<IndexTile>(q_pos + offset - window));
      any_rule = true;
    }
    if (num_sinks > 0) {
      keep = keep | (key_pos < ct::full<IndexTile>(num_sinks));
      any_rule = true;
    }
    if (block_size > 0) {
      keep = keep | (ct::full<IndexTile>((q_pos + offset) / block_size) == (key_pos / ct::full<IndexTile>(block_size)));
      any_rule = true;
    }
    if (compress_stride > 1) {
      keep = keep | ((key_pos % ct::full<IndexTile>(compress_stride)) == ct::full<IndexTile>(0));
      any_rule = true;
    }
    if (any_rule) {
      valid = valid & keep;
    }
  }

  auto score = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t d = 0; d < qk_dim; ++d) {
    auto q_val = ct::full<decltype(score)>(q[q_base_scalar + d]);
    auto k_val = ct::load_masked(k + ct::full<IndexTile>(k_base_scalar + d) + key_pos * ct::full<IndexTile>(qk_dim), valid);
    score = score + q_val * k_val;
  }
  score = score * ct::full<decltype(score)>(scale);
  auto neg_inf = ct::full<decltype(score)>(-3.4028234663852886e38f);
  auto safe_score = ct::select(valid, score, neg_inf);
  auto max_score = ct::reduce_max(safe_score, 0_ic);
  auto exp_score = ct::select(valid, ct::exp(score - max_score), ct::full<decltype(score)>(0.0f));
  auto denom = ct::sum(exp_score, 0_ic);
  auto prob = exp_score / denom;
  auto dprob = ct::full<decltype(score)>(0.0f);
  for (std::int64_t dv = 0; dv < value_dim; ++dv) {
    auto go_val = ct::full<decltype(score)>(grad_out[go_base_scalar + dv]);
    auto v_val = ct::load_masked(v + ct::full<IndexTile>(v_base_scalar + dv) + key_pos * ct::full<IndexTile>(value_dim), valid);
    dprob = dprob + go_val * v_val;
  }
  auto dprob_mean = ct::sum(prob * dprob, 0_ic);
  auto dscore = prob * (dprob - dprob_mean);

  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  using OneFloatTile = ct::tile<float, decltype(ct::shape{1_ic})>;
  for (std::int64_t d_q = 0; d_q < qk_dim; ++d_q) {
    auto k_selected_dim = ct::load_masked(
        k + ct::full<IndexTile>(k_base_scalar + d_q) + key_pos * ct::full<IndexTile>(qk_dim),
        valid);
    auto grad = ct::sum(dscore * k_selected_dim, 0_ic) * ct::full<OneFloatTile>(scale);
    ct::store(grad_q + ct::full<OneIndexTile>(q_base_scalar + d_q), grad);

    auto k_update = dscore * ct::full<decltype(dscore)>(q[q_base_scalar + d_q]) * ct::full<decltype(dscore)>(scale);
    ct::atomic_add_masked<ct::memory_order::relaxed>(
        grad_k + ct::full<IndexTile>(k_base_scalar + d_q) + key_pos * ct::full<IndexTile>(qk_dim),
        k_update,
        valid);
  }
  for (std::int64_t dv = 0; dv < value_dim; ++dv) {
    auto go_val = ct::full<decltype(prob)>(grad_out[go_base_scalar + dv]);
    auto v_update = prob * go_val;
    ct::atomic_add_masked<ct::memory_order::relaxed>(
        grad_v + ct::full<IndexTile>(v_base_scalar + dv) + key_pos * ct::full<IndexTile>(value_dim),
        v_update,
        valid);
  }
}

__tile_global__ void scaled_dot_product_attention_backward_q_row_float32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_q,
    std::int64_t rows,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    bool grad_out_merged) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  q = ct::assume_aligned(q, 16_ic);
  k = ct::assume_aligned(k, 16_ic);
  v = ct::assume_aligned(v, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_q = ct::assume_aligned(grad_q, 16_ic);

  const std::int64_t row_idx_scalar = static_cast<std::int64_t>(ct::bid().x);
  if (row_idx_scalar >= rows) {
    return;
  }

  const std::int64_t q_pos = row_idx_scalar % seq_q;
  const std::int64_t q_head = (row_idx_scalar / seq_q) % query_heads;
  const std::int64_t batch = row_idx_scalar / (seq_q * query_heads);
  const std::int64_t k_head = (q_head * key_heads) / query_heads;
  const std::int64_t q_base_scalar = ((batch * query_heads + q_head) * seq_q + q_pos) * qk_dim;
  const std::int64_t k_base_scalar = (batch * key_heads + k_head) * seq_k * qk_dim;
  const std::int64_t v_base_scalar = (batch * key_heads + k_head) * seq_k * value_dim;
  const std::int64_t go_base_scalar = grad_out_merged
      ? (batch * seq_q + q_pos) * query_heads * value_dim + q_head * value_dim
      : ((batch * query_heads + q_head) * seq_q + q_pos) * value_dim;

  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto key_pos = ct::iota<IndexTile>();
  auto valid = key_pos < ct::full<IndexTile>(seq_k);
  if (is_causal) {
    const std::int64_t offset = right_align_causal ? (seq_k - seq_q) : 0;
    valid = valid & (key_pos <= ct::full<IndexTile>(q_pos + offset));
  }
  if (use_sparse_rules) {
    const std::int64_t offset = right_align_causal ? (seq_k - seq_q) : 0;
    auto keep = ct::full<decltype(valid)>(false);
    bool any_rule = false;
    if (window > 0) {
      keep = keep | (key_pos > ct::full<IndexTile>(q_pos + offset - window));
      any_rule = true;
    }
    if (num_sinks > 0) {
      keep = keep | (key_pos < ct::full<IndexTile>(num_sinks));
      any_rule = true;
    }
    if (block_size > 0) {
      keep = keep | (ct::full<IndexTile>((q_pos + offset) / block_size) == (key_pos / ct::full<IndexTile>(block_size)));
      any_rule = true;
    }
    if (compress_stride > 1) {
      keep = keep | ((key_pos % ct::full<IndexTile>(compress_stride)) == ct::full<IndexTile>(0));
      any_rule = true;
    }
    if (any_rule) {
      valid = valid & keep;
    }
  }

  auto score = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t d = 0; d < qk_dim; ++d) {
    auto q_val = ct::full<decltype(score)>(q[q_base_scalar + d]);
    auto k_val = ct::load_masked(k + ct::full<IndexTile>(k_base_scalar + d) + key_pos * ct::full<IndexTile>(qk_dim), valid);
    score = score + q_val * k_val;
  }
  score = score * ct::full<decltype(score)>(scale);
  auto neg_inf = ct::full<decltype(score)>(-3.4028234663852886e38f);
  auto safe_score = ct::select(valid, score, neg_inf);
  auto max_score = ct::reduce_max(safe_score, 0_ic);
  auto exp_score = ct::select(valid, ct::exp(score - max_score), ct::full<decltype(score)>(0.0f));
  auto denom = ct::sum(exp_score, 0_ic);
  auto prob = exp_score / denom;
  auto dprob = ct::full<decltype(score)>(0.0f);
  for (std::int64_t dv = 0; dv < value_dim; ++dv) {
    auto go_val = ct::full<decltype(score)>(grad_out[go_base_scalar + dv]);
    auto v_val = ct::load_masked(v + ct::full<IndexTile>(v_base_scalar + dv) + key_pos * ct::full<IndexTile>(value_dim), valid);
    dprob = dprob + go_val * v_val;
  }
  auto dprob_mean = ct::sum(prob * dprob, 0_ic);
  auto dscore = prob * (dprob - dprob_mean);

  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  using OneFloatTile = ct::tile<float, decltype(ct::shape{1_ic})>;
  for (std::int64_t d_q = 0; d_q < qk_dim; ++d_q) {
    auto k_selected_dim = ct::load_masked(
        k + ct::full<IndexTile>(k_base_scalar + d_q) + key_pos * ct::full<IndexTile>(qk_dim),
        valid);
    auto grad = ct::sum(dscore * k_selected_dim, 0_ic) * ct::full<OneFloatTile>(scale);
    ct::store(grad_q + ct::full<OneIndexTile>(q_base_scalar + d_q), grad);
  }
}

__tile_global__ void scaled_dot_product_attention_backward_k_row_float32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_k,
    std::int64_t rows,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    bool grad_out_merged) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  q = ct::assume_aligned(q, 16_ic);
  k = ct::assume_aligned(k, 16_ic);
  v = ct::assume_aligned(v, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_k = ct::assume_aligned(grad_k, 16_ic);

  const std::int64_t row_idx_scalar = static_cast<std::int64_t>(ct::bid().x);
  if (row_idx_scalar >= rows) {
    return;
  }

  const std::int64_t k_pos_scalar = row_idx_scalar % seq_k;
  const std::int64_t k_head_scalar = (row_idx_scalar / seq_k) % key_heads;
  const std::int64_t batch_scalar = row_idx_scalar / (seq_k * key_heads);
  const std::int64_t k_base_scalar = (batch_scalar * key_heads + k_head_scalar) * seq_k * qk_dim;
  const std::int64_t v_base_scalar = (batch_scalar * key_heads + k_head_scalar) * seq_k * value_dim;

  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  using OneFloatTile = ct::tile<float, decltype(ct::shape{1_ic})>;
  auto key_pos = ct::iota<IndexTile>();
  auto key_match = key_pos == ct::full<IndexTile>(k_pos_scalar);
  for (std::int64_t d_k = 0; d_k < qk_dim; ++d_k) {
    ct::store(
        grad_k + ct::full<OneIndexTile>(k_base_scalar + k_pos_scalar * qk_dim + d_k),
        ct::full<OneFloatTile>(0.0f));
  }

  for (std::int64_t q_head = 0; q_head < query_heads; ++q_head) {
    if ((q_head * key_heads) / query_heads != k_head_scalar) {
      continue;
    }
    for (std::int64_t q_pos = 0; q_pos < seq_q; ++q_pos) {
      auto valid = key_pos < ct::full<IndexTile>(seq_k);
      if (is_causal) {
        const std::int64_t offset = right_align_causal ? (seq_k - seq_q) : 0;
        valid = valid & (key_pos <= ct::full<IndexTile>(q_pos + offset));
      }
      if (use_sparse_rules) {
        const std::int64_t offset = right_align_causal ? (seq_k - seq_q) : 0;
        auto keep = ct::full<decltype(valid)>(false);
        bool any_rule = false;
        if (window > 0) {
          keep = keep | (key_pos > ct::full<IndexTile>(q_pos + offset - window));
          any_rule = true;
        }
        if (num_sinks > 0) {
          keep = keep | (key_pos < ct::full<IndexTile>(num_sinks));
          any_rule = true;
        }
        if (block_size > 0) {
          keep = keep | (ct::full<IndexTile>((q_pos + offset) / block_size) == (key_pos / ct::full<IndexTile>(block_size)));
          any_rule = true;
        }
        if (compress_stride > 1) {
          keep = keep | ((key_pos % ct::full<IndexTile>(compress_stride)) == ct::full<IndexTile>(0));
          any_rule = true;
        }
        if (any_rule) {
          valid = valid & keep;
        }
      }
      const std::int64_t q_base_scalar = ((batch_scalar * query_heads + q_head) * seq_q + q_pos) * qk_dim;
      const std::int64_t go_base_scalar = grad_out_merged
          ? (batch_scalar * seq_q + q_pos) * query_heads * value_dim + q_head * value_dim
          : ((batch_scalar * query_heads + q_head) * seq_q + q_pos) * value_dim;
      auto score = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
      for (std::int64_t d = 0; d < qk_dim; ++d) {
        auto q_val = ct::full<decltype(score)>(q[q_base_scalar + d]);
        auto k_val = ct::load_masked(k + ct::full<IndexTile>(k_base_scalar + d) + key_pos * ct::full<IndexTile>(qk_dim), valid);
        score = score + q_val * k_val;
      }
      score = score * ct::full<decltype(score)>(scale);
      auto neg_inf = ct::full<decltype(score)>(-3.4028234663852886e38f);
      auto safe_score = ct::select(valid, score, neg_inf);
      auto max_score = ct::reduce_max(safe_score, 0_ic);
      auto exp_score = ct::select(valid, ct::exp(score - max_score), ct::full<decltype(score)>(0.0f));
      auto denom = ct::sum(exp_score, 0_ic);
      auto prob = exp_score / denom;
      auto dprob = ct::full<decltype(score)>(0.0f);
      for (std::int64_t dv = 0; dv < value_dim; ++dv) {
        auto go_val = ct::full<decltype(score)>(grad_out[go_base_scalar + dv]);
        auto v_val = ct::load_masked(v + ct::full<IndexTile>(v_base_scalar + dv) + key_pos * ct::full<IndexTile>(value_dim), valid);
        dprob = dprob + go_val * v_val;
      }
      auto dprob_mean = ct::sum(prob * dprob, 0_ic);
      auto dscore = prob * (dprob - dprob_mean);
      auto selected = ct::sum(ct::select(valid & key_match, dscore, ct::full<decltype(dscore)>(0.0f)), 0_ic);
      for (std::int64_t d_k = 0; d_k < qk_dim; ++d_k) {
        const std::int64_t out_idx_scalar = k_base_scalar + k_pos_scalar * qk_dim + d_k;
        auto current = ct::full<OneFloatTile>(grad_k[out_idx_scalar]);
        auto q_dim = ct::full<OneFloatTile>(q[q_base_scalar + d_k]);
        ct::store(
            grad_k + ct::full<OneIndexTile>(out_idx_scalar),
            current + selected * q_dim * ct::full<OneFloatTile>(scale));
      }
    }
  }
}

__tile_global__ void scaled_dot_product_attention_backward_v_row_float32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_v,
    std::int64_t rows,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    bool grad_out_merged) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  q = ct::assume_aligned(q, 16_ic);
  k = ct::assume_aligned(k, 16_ic);
  grad_out = ct::assume_aligned(grad_out, 16_ic);
  grad_v = ct::assume_aligned(grad_v, 16_ic);

  const std::int64_t row_idx_scalar = static_cast<std::int64_t>(ct::bid().x);
  if (row_idx_scalar >= rows) {
    return;
  }

  const std::int64_t k_pos_scalar = row_idx_scalar % seq_k;
  const std::int64_t k_head_scalar = (row_idx_scalar / seq_k) % key_heads;
  const std::int64_t batch_scalar = row_idx_scalar / (seq_k * key_heads);
  const std::int64_t k_base_scalar = (batch_scalar * key_heads + k_head_scalar) * seq_k * qk_dim;
  const std::int64_t v_base_scalar = (batch_scalar * key_heads + k_head_scalar) * seq_k * value_dim;

  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  using OneFloatTile = ct::tile<float, decltype(ct::shape{1_ic})>;
  auto key_pos = ct::iota<IndexTile>();
  auto key_match = key_pos == ct::full<IndexTile>(k_pos_scalar);
  for (std::int64_t d_v = 0; d_v < value_dim; ++d_v) {
    ct::store(
        grad_v + ct::full<OneIndexTile>(v_base_scalar + k_pos_scalar * value_dim + d_v),
        ct::full<OneFloatTile>(0.0f));
  }

  for (std::int64_t q_head = 0; q_head < query_heads; ++q_head) {
    if ((q_head * key_heads) / query_heads != k_head_scalar) {
      continue;
    }
    for (std::int64_t q_pos = 0; q_pos < seq_q; ++q_pos) {
      auto valid = key_pos < ct::full<IndexTile>(seq_k);
      if (is_causal) {
        const std::int64_t offset = right_align_causal ? (seq_k - seq_q) : 0;
        valid = valid & (key_pos <= ct::full<IndexTile>(q_pos + offset));
      }
      if (use_sparse_rules) {
        const std::int64_t offset = right_align_causal ? (seq_k - seq_q) : 0;
        auto keep = ct::full<decltype(valid)>(false);
        bool any_rule = false;
        if (window > 0) {
          keep = keep | (key_pos > ct::full<IndexTile>(q_pos + offset - window));
          any_rule = true;
        }
        if (num_sinks > 0) {
          keep = keep | (key_pos < ct::full<IndexTile>(num_sinks));
          any_rule = true;
        }
        if (block_size > 0) {
          keep = keep | (ct::full<IndexTile>((q_pos + offset) / block_size) == (key_pos / ct::full<IndexTile>(block_size)));
          any_rule = true;
        }
        if (compress_stride > 1) {
          keep = keep | ((key_pos % ct::full<IndexTile>(compress_stride)) == ct::full<IndexTile>(0));
          any_rule = true;
        }
        if (any_rule) {
          valid = valid & keep;
        }
      }
      const std::int64_t q_base_scalar = ((batch_scalar * query_heads + q_head) * seq_q + q_pos) * qk_dim;
      const std::int64_t go_base_scalar = grad_out_merged
          ? (batch_scalar * seq_q + q_pos) * query_heads * value_dim + q_head * value_dim
          : ((batch_scalar * query_heads + q_head) * seq_q + q_pos) * value_dim;
      auto score = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
      for (std::int64_t d = 0; d < qk_dim; ++d) {
        auto q_val = ct::full<decltype(score)>(q[q_base_scalar + d]);
        auto k_val = ct::load_masked(k + ct::full<IndexTile>(k_base_scalar + d) + key_pos * ct::full<IndexTile>(qk_dim), valid);
        score = score + q_val * k_val;
      }
      score = score * ct::full<decltype(score)>(scale);
      auto neg_inf = ct::full<decltype(score)>(-3.4028234663852886e38f);
      auto safe_score = ct::select(valid, score, neg_inf);
      auto max_score = ct::reduce_max(safe_score, 0_ic);
      auto exp_score = ct::select(valid, ct::exp(score - max_score), ct::full<decltype(score)>(0.0f));
      auto denom = ct::sum(exp_score, 0_ic);
      auto prob = exp_score / denom;
      auto selected_prob = ct::sum(ct::select(valid & key_match, prob, ct::full<decltype(prob)>(0.0f)), 0_ic);
      for (std::int64_t d_v = 0; d_v < value_dim; ++d_v) {
        const std::int64_t out_idx_scalar = v_base_scalar + k_pos_scalar * value_dim + d_v;
        auto current = ct::full<OneFloatTile>(grad_v[out_idx_scalar]);
        auto go_dim = ct::full<OneFloatTile>(grad_out[go_base_scalar + d_v]);
        ct::store(
            grad_v + ct::full<OneIndexTile>(out_idx_scalar),
            current + selected_prob * go_dim);
      }
    }
  }
}

__device__ float deterministic_uniform_value(std::int64_t linear_idx, std::int64_t counter, std::int64_t salt) {
  const std::uint64_t modulus = 16777216ULL;
  std::uint64_t value =
      (static_cast<std::uint64_t>(linear_idx) * 1103515245ULL +
       static_cast<std::uint64_t>(counter) * 12345ULL +
       static_cast<std::uint64_t>(salt)) %
      modulus;
  return static_cast<float>(value) / 16777216.0f;
}

__global__ void random_timesteps_float32_kernel(float* out, std::int64_t batch, std::int64_t counter) {
  const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * kTileSize;
  for (std::int64_t offset = 0; offset < kTileSize; ++offset) {
    const std::int64_t idx = base + offset;
    if (idx < batch) {
      out[idx] = deterministic_uniform_value(idx, counter, 17);
    }
  }
}

__global__ void mask_scheduler_int64_kernel(
    const std::int64_t* tokens,
    const float* timesteps,
    std::int64_t* out,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t mask_token_id,
    std::int64_t counter) {
  const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * kTileSize;
  for (std::int64_t offset = 0; offset < kTileSize; ++offset) {
    const std::int64_t idx = base + offset;
    if (idx < n) {
      const std::int64_t batch = idx / seq_len;
      const float noise = deterministic_uniform_value(idx, counter, 53);
      out[idx] = noise < timesteps[batch] ? mask_token_id : tokens[idx];
    }
  }
}

__global__ void jepa_mask_int64_kernel(
    const std::int64_t* tokens,
    std::int64_t* masked_tokens,
    float* mask_values,
    std::int64_t n,
    std::int64_t seq_len,
    float mask_ratio,
    std::int64_t mask_token_id,
    int strategy,
    std::int64_t num_blocks,
    float min_block_ratio,
    float max_block_ratio,
    std::int64_t counter) {
  const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * kTileSize;
  for (std::int64_t offset = 0; offset < kTileSize; ++offset) {
    const std::int64_t idx = base + offset;
    if (idx < n) {
      const std::int64_t batch = idx / seq_len;
      const std::int64_t pos = idx - batch * seq_len;
      bool is_masked = false;
      if (strategy == 1) {
        const std::int64_t min_len = max(static_cast<std::int64_t>(1), static_cast<std::int64_t>(min_block_ratio * seq_len));
        const std::int64_t max_len = max(min_len, static_cast<std::int64_t>(max_block_ratio * seq_len));
        const std::int64_t len_span = max(static_cast<std::int64_t>(1), max_len - min_len + 1);
        for (std::int64_t block = 0; block < num_blocks; ++block) {
          const float block_noise = deterministic_uniform_value(batch, counter, 101 + block * 2);
          const float start_noise = deterministic_uniform_value(batch, counter, 102 + block * 2);
          const std::int64_t block_len = min_len + min(len_span - 1, static_cast<std::int64_t>(block_noise * len_span));
          const std::int64_t max_start = max(static_cast<std::int64_t>(0), seq_len - block_len);
          const std::int64_t start = min(max_start, static_cast<std::int64_t>(start_noise * (static_cast<float>(max_start) + 1.0f)));
          if (pos >= start && pos < start + block_len) {
            is_masked = true;
          }
        }
      } else {
        is_masked = deterministic_uniform_value(idx, counter, 29) < mask_ratio;
      }
      masked_tokens[idx] = is_masked ? mask_token_id : tokens[idx];
      mask_values[idx] = is_masked ? 1.0f : 0.0f;
    }
  }
}

}  // namespace

int linear_backward_bias_threads_per_block() {
  return linear_backward_bias_threads_per_block_value();
}

bool cublaslt_linear_backward_input_bf16_bits_weight_bf16_strided_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t grad_out_row_stride,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (grad_out_bf16_bits == nullptr ||
      weight_bf16_bits == nullptr ||
      grad_x == nullptr ||
      !fits_cublas_int(rows) ||
      !fits_cublas_int(input_dim) ||
      !fits_cublas_int(output_dim) ||
      !fits_cublas_int(grad_out_row_stride) ||
      grad_out_row_stride < output_dim) {
    return false;
  }
  if (!trainer_linear_bf16_cublaslt_enabled()) {
    return false;
  }
  const auto* weight_bf16 = reinterpret_cast<const __nv_bfloat16*>(weight_bf16_bits);
  const auto* grad_out_bf16 = reinterpret_cast<const __nv_bfloat16*>(grad_out_bf16_bits);
  return cublaslt_linear_matmul(
      weight_bf16,
      grad_out_bf16,
      grad_x,
      CUDA_R_16BF,
      CUDA_R_16BF,
      CUDA_R_32F,
      CUBLAS_COMPUTE_32F_FAST_16BF,
      static_cast<int>(input_dim),
      static_cast<int>(rows),
      static_cast<int>(output_dim),
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      static_cast<int>(input_dim),
      static_cast<int>(grad_out_row_stride),
      static_cast<int>(input_dim),
      0.0f,
      stream);
#else
  (void)grad_out_bf16_bits;
  (void)weight_bf16_bits;
  (void)grad_x;
  (void)rows;
  (void)input_dim;
  (void)output_dim;
  (void)grad_out_row_stride;
  (void)stream;
  return false;
#endif
}

bool cublaslt_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t grad_out_row_stride,
    float beta,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (x_bf16_bits == nullptr ||
      grad_out_bf16_bits == nullptr ||
      grad_weight == nullptr ||
      !fits_cublas_int(rows) ||
      !fits_cublas_int(input_dim) ||
      !fits_cublas_int(output_dim) ||
      !fits_cublas_int(grad_out_row_stride) ||
      grad_out_row_stride < output_dim) {
    return false;
  }
  if (!trainer_linear_bf16_cublaslt_enabled()) {
    return false;
  }
  const auto* x_bf16 = reinterpret_cast<const __nv_bfloat16*>(x_bf16_bits);
  const auto* grad_out_bf16 = reinterpret_cast<const __nv_bfloat16*>(grad_out_bf16_bits);
  return cublaslt_linear_matmul(
      x_bf16,
      grad_out_bf16,
      grad_weight,
      CUDA_R_16BF,
      CUDA_R_16BF,
      CUDA_R_32F,
      CUBLAS_COMPUTE_32F_FAST_16BF,
      static_cast<int>(input_dim),
      static_cast<int>(output_dim),
      static_cast<int>(rows),
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      static_cast<int>(input_dim),
      static_cast<int>(grad_out_row_stride),
      static_cast<int>(input_dim),
      beta,
      stream);
#else
  (void)x_bf16_bits;
  (void)grad_out_bf16_bits;
  (void)grad_weight;
  (void)rows;
  (void)input_dim;
  (void)output_dim;
  (void)grad_out_row_stride;
  (void)beta;
  (void)stream;
  return false;
#endif
}

std::int64_t token_cross_entropy_bf16_threads_per_row() {
  return static_cast<std::int64_t>(cross_entropy_bf16_threads_per_row());
}

std::int64_t lm_head_prob_only_target_correction_threads() {
  return static_cast<std::int64_t>(lm_head_prob_only_target_correction_threads_value());
}

void launch_unary_float32(const float* x, float* out, std::int64_t n, int op, cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  unary_float32_kernel<<<blocks, 1, 0, stream>>>(x, out, n, op);
}

void launch_binary_float32(
    const float* lhs,
    const float* rhs,
    float* out,
    std::int64_t n,
    int op,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  binary_float32_kernel<<<blocks, 1, 0, stream>>>(lhs, rhs, out, n, op);
}

void launch_binary_pair_float32(
    const float* lhs,
    const float* rhs,
    float* out0,
    float* out1,
    std::int64_t n,
    int op,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  binary_pair_float32_kernel<<<blocks, 1, 0, stream>>>(lhs, rhs, out0, out1, n, op);
}

void launch_scalar_unary_float32(
    const float* x,
    float* out,
    std::int64_t n,
    float value,
    int op,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  scalar_unary_float32_kernel<<<blocks, 1, 0, stream>>>(x, out, n, value, op);
}

void launch_scalar_binary_float32(
    const float* lhs,
    const float* rhs,
    float* out,
    std::int64_t n,
    float value,
    int op,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  scalar_binary_float32_kernel<<<blocks, 1, 0, stream>>>(lhs, rhs, out, n, value, op);
}

void launch_ema_update_float32(
    float* target,
    const float* source,
    std::int64_t n,
    float decay,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  ema_update_float32_kernel<<<blocks, 1, 0, stream>>>(target, source, n, decay);
}

void launch_gradient_accumulate_float32(
    float* buffer,
    const float* grad,
    std::int64_t n,
    float scale,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  gradient_accumulate_float32_kernel<<<blocks, 1, 0, stream>>>(buffer, grad, n, scale);
}

void launch_copy_float32(
    const float* source,
    float* dest,
    std::int64_t n,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  copy_float32_kernel<<<blocks, 1, 0, stream>>>(source, dest, n);
}

void launch_evo_mutate_candidates_float32(
    const float* base,
    float* candidates,
    std::int64_t elements,
    std::int64_t candidate_count,
    float mutation_scale,
    std::int64_t seed,
    cudaStream_t stream) {
  if (elements <= 0 || candidate_count <= 0) {
    return;
  }
  constexpr int threads = 256;
  const std::int64_t total = elements * candidate_count;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  evo_mutate_candidates_float32_kernel<<<blocks, threads, 0, stream>>>(
      base, candidates, elements, candidate_count, mutation_scale, seed);
}

void launch_evo_select_best_loss_float32(
    const float* losses,
    std::int64_t candidate_count,
    std::int64_t* best_index,
    float* best_loss,
    cudaStream_t stream) {
  if (candidate_count <= 0) {
    return;
  }
  evo_select_best_loss_float32_kernel<<<1, 1, 0, stream>>>(
      losses, candidate_count, best_index, best_loss);
}

void launch_evo_adopt_candidate_float32(
    const float* candidates,
    const std::int64_t* best_index,
    float* target,
    std::int64_t elements,
    std::int64_t candidate_count,
    cudaStream_t stream) {
  if (elements <= 0 || candidate_count <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  evo_adopt_candidate_float32_kernel<<<blocks, threads, 0, stream>>>(
      candidates, best_index, target, elements, candidate_count);
}

void launch_uint16_to_int64(
    const std::uint16_t* source,
    std::int64_t* dest,
    std::int64_t n,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  uint16_to_int64_kernel<<<blocks, 1, 0, stream>>>(source, dest, n);
}

void launch_float32_to_bf16_bits(
    const float* source,
    std::uint16_t* dest,
    std::int64_t n,
    cudaStream_t stream) {
  static const bool vec4_enabled = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_F32_TO_BF16_VEC4");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT_F32_TO_BF16_VEC4");
    }
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT2_F32_TO_BF16_VEC4");
    }
    if (value == nullptr) {
      return true;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  if (vec4_enabled && n > 0 && (n % 4) == 0 &&
      (reinterpret_cast<std::uintptr_t>(source) % alignof(float4)) == 0 &&
      (reinterpret_cast<std::uintptr_t>(dest) % alignof(unsigned long long)) == 0) {
    const std::int64_t n4 = n / 4;
    const int blocks = static_cast<int>((n4 + kTileSize - 1) / kTileSize);
    f32_to_bf16_bits_vec4_kernel<<<blocks, kTileSize, 0, stream>>>(
        reinterpret_cast<const float4*>(source),
        reinterpret_cast<unsigned long long*>(dest),
        n4);
    return;
  }
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  f32_to_bf16_bits_kernel<<<blocks, kTileSize, 0, stream>>>(source, dest, n);
}

void launch_bf16_bits_to_float32(
    const std::uint16_t* source,
    float* dest,
    std::int64_t n,
    cudaStream_t stream) {
  launch_bf16_bits_to_float32_internal(source, dest, n, stream);
}

void launch_store_mlp_activations_bf16_float32(
    const float* ln2_out,
    const float* fc_out,
    const float* act,
    std::uint16_t* dest,
    std::int64_t activation_elements,
    std::int64_t hidden_elements,
    cudaStream_t stream) {
  const std::int64_t total = activation_elements + hidden_elements * 2;
  const bool vec4_eligible =
      store_mlp_activations_vec4_enabled() &&
      activation_elements % 4 == 0 &&
      hidden_elements % 4 == 0 &&
      ((reinterpret_cast<std::uintptr_t>(ln2_out) & 0x0f) == 0) &&
      ((reinterpret_cast<std::uintptr_t>(fc_out) & 0x0f) == 0) &&
      ((reinterpret_cast<std::uintptr_t>(act) & 0x0f) == 0) &&
      ((reinterpret_cast<std::uintptr_t>(dest) & 0x07) == 0);
  if (vec4_eligible) {
    const std::int64_t total_vec4 = total / 4;
    const int blocks = static_cast<int>((total_vec4 + kTileSize - 1) / kTileSize);
    store_mlp_activations_bf16_float32_vec4_kernel<<<blocks, kTileSize, 0, stream>>>(
        ln2_out,
        fc_out,
        act,
        reinterpret_cast<unsigned long long*>(dest),
        activation_elements / 4,
        hidden_elements / 4);
    return;
  }
  const int blocks = static_cast<int>((total + kTileSize - 1) / kTileSize);
  store_mlp_activations_bf16_float32_kernel<<<blocks, kTileSize, 0, stream>>>(
      ln2_out, fc_out, act, dest, activation_elements, hidden_elements);
}

void launch_restore_mlp_activations_bf16_float32(
    const std::uint16_t* source,
    float* ln2_out,
    float* fc_out,
    float* act,
    std::int64_t activation_elements,
    std::int64_t hidden_elements,
    cudaStream_t stream) {
  const std::int64_t total = activation_elements + hidden_elements * 2;
  const bool vec4_eligible =
      store_mlp_activations_vec4_enabled() &&
      activation_elements % 4 == 0 &&
      hidden_elements % 4 == 0 &&
      ((reinterpret_cast<std::uintptr_t>(source) & 0x07) == 0) &&
      ((reinterpret_cast<std::uintptr_t>(ln2_out) & 0x0f) == 0) &&
      ((reinterpret_cast<std::uintptr_t>(fc_out) & 0x0f) == 0) &&
      ((reinterpret_cast<std::uintptr_t>(act) & 0x0f) == 0);
  if (vec4_eligible) {
    const std::int64_t total_vec4 = total / 4;
    const int blocks = static_cast<int>((total_vec4 + kTileSize - 1) / kTileSize);
    restore_mlp_activations_bf16_float32_vec4_kernel<<<blocks, kTileSize, 0, stream>>>(
        reinterpret_cast<const unsigned long long*>(source),
        ln2_out,
        fc_out,
        act,
        activation_elements / 4,
        hidden_elements / 4);
    return;
  }
  const int blocks = static_cast<int>((total + kTileSize - 1) / kTileSize);
  restore_mlp_activations_bf16_float32_kernel<<<blocks, kTileSize, 0, stream>>>(
      source, ln2_out, fc_out, act, activation_elements, hidden_elements);
}

void launch_float32_to_bf16_bits_many(
    const float* const* sources,
    const std::int64_t* elements,
    const std::int64_t* offsets,
    std::uint16_t* dest,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    cudaStream_t stream) {
  if (buffer_count <= 0 || max_elements <= 0) {
    return;
  }
  if (f32_to_bf16_bits_many_vec4_enabled()) {
    const std::int64_t max_vec4_elements = (max_elements + 3) / 4;
    const int chunks = static_cast<int>((max_vec4_elements + kTileSize - 1) / kTileSize);
    dim3 grid(static_cast<unsigned int>(chunks), static_cast<unsigned int>(buffer_count), 1);
    f32_to_bf16_bits_many_vec4_kernel<<<grid, kTileSize, 0, stream>>>(
        sources, elements, offsets, dest, buffer_count, max_vec4_elements);
    return;
  }
  const int chunks = static_cast<int>((max_elements + kTileSize - 1) / kTileSize);
  dim3 grid(static_cast<unsigned int>(chunks), static_cast<unsigned int>(buffer_count), 1);
  f32_to_bf16_bits_many_kernel<<<grid, kTileSize, 0, stream>>>(
      sources, elements, offsets, dest, buffer_count, max_elements);
}

void launch_fill_float32(
    float* values,
    std::int64_t n,
    float value,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  fill_float32_kernel<<<blocks, 1, 0, stream>>>(values, n, value);
}

void launch_fill_many_float32(
    float* const* buffers,
    const std::int64_t* elements,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    float value,
    cudaStream_t stream) {
  if (buffer_count <= 0 || max_elements <= 0) {
    return;
  }
  const int tensor_blocks = static_cast<int>(buffer_count);
  const int element_blocks = static_cast<int>((max_elements + kTileSize - 1) / kTileSize);
  fill_many_float32_kernel<<<dim3(tensor_blocks, element_blocks), 1, 0, stream>>>(
      buffers,
      elements,
      buffer_count,
      value);
}

void launch_fill_many_values_float32(
    float* const* buffers,
    const std::int64_t* elements,
    const float* values,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    cudaStream_t stream) {
  if (buffer_count <= 0 || max_elements <= 0) {
    return;
  }
  const int tensor_blocks = static_cast<int>(buffer_count);
  const int element_blocks = static_cast<int>((max_elements + kTileSize - 1) / kTileSize);
  fill_many_values_float32_kernel<<<dim3(tensor_blocks, element_blocks), 1, 0, stream>>>(
      buffers,
      elements,
      values,
      buffer_count);
}

void launch_fill_many_values_bf16_bits_float32(
    std::uint16_t* const* buffers,
    const std::int64_t* elements,
    const float* values,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    cudaStream_t stream) {
  if (buffer_count <= 0 || max_elements <= 0) {
    return;
  }
  const int tensor_blocks = static_cast<int>(buffer_count);
  const int element_blocks = static_cast<int>((max_elements + kTileSize - 1) / kTileSize);
  fill_many_values_bf16_bits_float32_kernel<<<dim3(tensor_blocks, element_blocks), 1, 0, stream>>>(
      buffers,
      elements,
      values,
      buffer_count);
}

bool token_weight_threaded_init_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_TOKEN_WEIGHT_THREADED_INIT");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT");
    }
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT2_TOKEN_WEIGHT_THREADED_INIT");
    }
    if (value == nullptr || value[0] == '\0') {
      return false;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool token_weight_fast_int32_tile_init_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_TOKEN_WEIGHT_FAST_INT32_INIT");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT_TOKEN_WEIGHT_FAST_INT32_INIT");
    }
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT2_TOKEN_WEIGHT_FAST_INT32_INIT");
    }
    if (value == nullptr || value[0] == '\0') {
      return true;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool token_weight_vector4_init_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_TOKEN_WEIGHT_VECTOR4_INIT");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT");
    }
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT2_TOKEN_WEIGHT_VECTOR4_INIT");
    }
    if (value == nullptr || value[0] == '\0') {
      return true;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool token_weight_bf16_pattern_init_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_TOKEN_WEIGHT_BF16_PATTERN_INIT");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_PATTERN_INIT");
    }
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT2_TOKEN_WEIGHT_BF16_PATTERN_INIT");
    }
    if (value == nullptr || value[0] == '\0') {
      return false;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

bool token_weight_vector4_strided_init_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("NFN_TILE_CUDA_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT");
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT");
    }
    if (value == nullptr) {
      value = std::getenv("NFN_NATIVE_GPT2_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT");
    }
    if (value == nullptr || value[0] == '\0') {
      return false;
    }
    if (std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
      return false;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
  }();
  return enabled;
}

void launch_init_gpt2_token_weight_vector4_strided_float32(
    float* values,
    std::uint16_t* shadow_bf16_bits,
    std::int64_t n,
    cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  constexpr int kThreads = 256;
  constexpr int kMaxBlocks = 4096;
  const int blocks = std::min<int>(
      kMaxBlocks,
      static_cast<int>(((n + 3) / 4 + kThreads - 1) / kThreads));
  if (shadow_bf16_bits != nullptr) {
    init_gpt2_token_weight_vector4_strided_with_bf16_shadow_float32_kernel<<<
        blocks, kThreads, 0, stream>>>(values, shadow_bf16_bits, n);
  } else {
    init_gpt2_token_weight_vector4_strided_float32_kernel<<<blocks, kThreads, 0, stream>>>(
        values, n);
  }
}

void launch_init_gpt2_token_weight_threaded_float32(
    float* values,
    std::uint16_t* shadow_bf16_bits,
    std::int64_t n,
    bool power2_bucket,
    cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  constexpr int kThreads = 256;
  constexpr int kMaxBlocks = 4096;
  const int blocks = std::min<int>(
      kMaxBlocks,
      static_cast<int>((n + kThreads - 1) / kThreads));
  const int bucket_mask = power2_bucket ? 15 : 0;
  const int bucket_mod = power2_bucket ? 16 : 17;
  if (shadow_bf16_bits != nullptr) {
    init_gpt2_token_weight_threaded_with_bf16_shadow_float32_kernel<<<blocks, kThreads, 0, stream>>>(
        values, shadow_bf16_bits, n, bucket_mask, bucket_mod);
  } else {
    init_gpt2_token_weight_threaded_float32_kernel<<<blocks, kThreads, 0, stream>>>(
        values, n, bucket_mask, bucket_mod);
  }
}

void launch_init_gpt2_token_weight_float32(
    float* values,
    std::int64_t n,
    cudaStream_t stream) {
  if (token_weight_threaded_init_enabled()) {
    launch_init_gpt2_token_weight_threaded_float32(values, nullptr, n, false, stream);
    return;
  }
  constexpr int kTokenInitTileSize = NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE;
  const int blocks = static_cast<int>((n + kTokenInitTileSize - 1) / kTokenInitTileSize);
  init_gpt2_token_weight_float32_kernel<<<blocks, 1, 0, stream>>>(values, n);
}

void launch_init_gpt2_token_weight_fast_float32(
    float* values,
    std::int64_t n,
    cudaStream_t stream) {
  if (token_weight_threaded_init_enabled()) {
    launch_init_gpt2_token_weight_threaded_float32(values, nullptr, n, true, stream);
    return;
  }
  if (token_weight_vector4_strided_init_enabled()) {
    launch_init_gpt2_token_weight_vector4_strided_float32(values, nullptr, n, stream);
    return;
  }
  if (token_weight_vector4_init_enabled()) {
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((n + (kThreads * 4 - 1)) / (kThreads * 4));
    init_gpt2_token_weight_vector4_float32_kernel<<<blocks, kThreads, 0, stream>>>(values, n);
    return;
  }
  constexpr int kTokenInitTileSize = NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE;
  const int blocks = static_cast<int>((n + kTokenInitTileSize - 1) / kTokenInitTileSize);
  if (token_weight_fast_int32_tile_init_enabled() &&
      n <= static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    init_gpt2_token_weight_fast_int32_float32_kernel<<<blocks, 1, 0, stream>>>(
        values, static_cast<int>(n));
    return;
  }
  init_gpt2_token_weight_fast_float32_kernel<<<blocks, 1, 0, stream>>>(values, n);
}

void launch_init_gpt2_token_weight_with_bf16_shadow_float32(
    float* values,
    std::uint16_t* shadow_bf16_bits,
    std::int64_t n,
    cudaStream_t stream) {
  if (shadow_bf16_bits == nullptr) {
    launch_init_gpt2_token_weight_float32(values, n, stream);
    return;
  }
  if (token_weight_threaded_init_enabled()) {
    launch_init_gpt2_token_weight_threaded_float32(values, shadow_bf16_bits, n, false, stream);
    return;
  }
  constexpr int kTokenInitTileSize = NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE;
  const int blocks = static_cast<int>((n + kTokenInitTileSize - 1) / kTokenInitTileSize);
  init_gpt2_token_weight_with_bf16_shadow_float32_kernel<<<blocks, 1, 0, stream>>>(
      values, shadow_bf16_bits, n);
}

void launch_init_gpt2_token_weight_fast_with_bf16_shadow_float32(
    float* values,
    std::uint16_t* shadow_bf16_bits,
    std::int64_t n,
    cudaStream_t stream) {
  if (shadow_bf16_bits == nullptr) {
    launch_init_gpt2_token_weight_fast_float32(values, n, stream);
    return;
  }
  if (token_weight_threaded_init_enabled()) {
    launch_init_gpt2_token_weight_threaded_float32(values, shadow_bf16_bits, n, true, stream);
    return;
  }
  if (token_weight_vector4_strided_init_enabled()) {
    launch_init_gpt2_token_weight_vector4_strided_float32(values, shadow_bf16_bits, n, stream);
    return;
  }
  if (token_weight_vector4_init_enabled()) {
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((n + (kThreads * 4 - 1)) / (kThreads * 4));
    if (token_weight_bf16_pattern_init_enabled()) {
      init_gpt2_token_weight_vector4_with_bf16_shadow_float32_kernel<<<blocks, kThreads, 0, stream>>>(
          values, shadow_bf16_bits, n);
    } else {
      init_gpt2_token_weight_vector4_with_bf16_shadow_convert_float32_kernel<<<blocks, kThreads, 0, stream>>>(
          values, shadow_bf16_bits, n);
    }
    return;
  }
  constexpr int kTokenInitTileSize = NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE;
  const int blocks = static_cast<int>((n + kTokenInitTileSize - 1) / kTokenInitTileSize);
  if (token_weight_fast_int32_tile_init_enabled() &&
      n <= static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    init_gpt2_token_weight_fast_int32_with_bf16_shadow_float32_kernel<<<blocks, 1, 0, stream>>>(
        values, shadow_bf16_bits, static_cast<int>(n));
    return;
  }
  init_gpt2_token_weight_fast_with_bf16_shadow_float32_kernel<<<blocks, 1, 0, stream>>>(
      values, shadow_bf16_bits, n);
}

void launch_init_gpt2_token_weight_fast_with_bf16_shadow_padded_float32(
    float* values,
    std::uint16_t* shadow_bf16_bits,
    std::int64_t public_n,
    std::int64_t total_n,
    cudaStream_t stream) {
  if (total_n <= 0) {
    return;
  }
  if (shadow_bf16_bits == nullptr || public_n >= total_n) {
    launch_init_gpt2_token_weight_fast_with_bf16_shadow_float32(
        values, shadow_bf16_bits, total_n, stream);
    return;
  }
  if (public_n < 0) {
    public_n = 0;
  }
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((total_n + (kThreads * 4 - 1)) / (kThreads * 4));
  init_gpt2_token_weight_vector4_with_bf16_shadow_padded_float32_kernel<<<blocks, kThreads, 0, stream>>>(
      values, shadow_bf16_bits, public_n, total_n);
}

void launch_sumsq_partials_float32(
    const float* values,
    float* partials,
    std::int64_t n,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kOptimizerTileSize - 1) / kOptimizerTileSize);
  sumsq_partials_float32_kernel<<<blocks, 1, 0, stream>>>(values, partials, n);
}

void launch_sumsq_partials_many_float32(
    const float* const* buffers,
    const std::int64_t* elements,
    const std::int64_t* partial_offsets,
    float* partials,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    cudaStream_t stream) {
  if (buffer_count <= 0 || max_elements <= 0) {
    return;
  }
  const int tensor_blocks = static_cast<int>(buffer_count);
  const int element_blocks = static_cast<int>((max_elements + kOptimizerTileSize - 1) / kOptimizerTileSize);
  sumsq_partials_many_float32_kernel<<<dim3(tensor_blocks, element_blocks), 1, 0, stream>>>(
      buffers,
      elements,
      partial_offsets,
      partials,
      buffer_count);
}

void launch_sumsq_partials_many_bf16_bits_float32(
    const std::uint16_t* const* buffers,
    const std::int64_t* elements,
    const std::int64_t* partial_offsets,
    float* partials,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    cudaStream_t stream) {
  if (buffer_count <= 0 || max_elements <= 0) {
    return;
  }
  const int tensor_blocks = static_cast<int>(buffer_count);
  const int element_blocks = static_cast<int>((max_elements + kOptimizerTileSize - 1) / kOptimizerTileSize);
  sumsq_partials_many_bf16_bits_float32_kernel<<<dim3(tensor_blocks, element_blocks), 1, 0, stream>>>(
      buffers,
      elements,
      partial_offsets,
      partials,
      buffer_count);
}

void launch_scale_inplace_float32(
    float* values,
    std::int64_t n,
    float scale,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kOptimizerTileSize - 1) / kOptimizerTileSize);
  scale_inplace_float32_kernel<<<blocks, 1, 0, stream>>>(values, n, scale);
}

void launch_global_norm_clip_scale_float32(
    const float* sumsq_partials,
    float* clip_scale,
    std::int64_t partial_count,
    float max_norm,
    float eps,
    cudaStream_t stream) {
  global_norm_clip_scale_float32_kernel<<<1, 1, 0, stream>>>(
      sumsq_partials, clip_scale, partial_count, max_norm, eps);
}

void launch_scale_inplace_by_device_float32(
    float* values,
    const float* scale,
    std::int64_t n,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kOptimizerTileSize - 1) / kOptimizerTileSize);
  scale_inplace_by_device_float32_kernel<<<blocks, 1, 0, stream>>>(values, scale, n);
}

void launch_adamw_step_float32(
    float* param,
    const float* grad,
    float* exp_avg,
    float* exp_avg_sq,
    std::int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float sqrt_bias_correction2,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kOptimizerTileSize - 1) / kOptimizerTileSize);
  adamw_step_float32_kernel<<<blocks, 1, 0, stream>>>(
      param,
      grad,
      exp_avg,
      exp_avg_sq,
      n,
      lr,
      beta1,
      beta2,
      eps,
      weight_decay,
      bias_correction1,
      sqrt_bias_correction2);
}

void launch_adamw_step_with_device_scale_float32(
    float* param,
    const float* grad,
    const float* grad_scale,
    float* exp_avg,
    float* exp_avg_sq,
    std::int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float sqrt_bias_correction2,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kOptimizerTileSize - 1) / kOptimizerTileSize);
  adamw_step_with_device_scale_float32_kernel<<<blocks, 1, 0, stream>>>(
      param,
      grad,
      grad_scale,
      exp_avg,
      exp_avg_sq,
      n,
      lr,
      beta1,
      beta2,
      eps,
      weight_decay,
      bias_correction1,
      sqrt_bias_correction2);
}

void launch_adamw_step_many_with_device_scale_float32(
    float* const* params,
    const float* const* grads,
    const float* grad_scale,
    float* const* exp_avgs,
    float* const* exp_avg_sqs,
    const std::int64_t* elements,
    const float* weight_decays,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bias_correction1,
    float sqrt_bias_correction2,
    cudaStream_t stream) {
  if (buffer_count <= 0 || max_elements <= 0) {
    return;
  }
  const int tensor_blocks = static_cast<int>(buffer_count);
  const int element_blocks = static_cast<int>((max_elements + kOptimizerTileSize - 1) / kOptimizerTileSize);
  adamw_step_many_with_device_scale_float32_kernel<<<dim3(tensor_blocks, element_blocks), 1, 0, stream>>>(
      params,
      grads,
      grad_scale,
      exp_avgs,
      exp_avg_sqs,
      elements,
      weight_decays,
      buffer_count,
      lr,
      beta1,
      beta2,
      eps,
      bias_correction1,
      sqrt_bias_correction2);
}

void launch_adamw_step_many_with_device_scale_bf16_shadow_float32(
    float* const* params,
    const float* const* grads,
    const float* grad_scale,
    float* const* exp_avgs,
    float* const* exp_avg_sqs,
    const std::int64_t* elements,
    const float* weight_decays,
    const std::int64_t* bf16_shadow_offsets,
    std::uint16_t* bf16_shadow_bits,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bias_correction1,
    float sqrt_bias_correction2,
    cudaStream_t stream) {
  if (buffer_count <= 0 || max_elements <= 0) {
    return;
  }
  const int tensor_blocks = static_cast<int>(buffer_count);
  const int element_blocks = static_cast<int>((max_elements + kOptimizerTileSize - 1) / kOptimizerTileSize);
  adamw_step_many_with_device_scale_bf16_shadow_float32_kernel<<<dim3(tensor_blocks, element_blocks), 1, 0, stream>>>(
      params,
      grads,
      grad_scale,
      exp_avgs,
      exp_avg_sqs,
      elements,
      weight_decays,
      bf16_shadow_offsets,
      bf16_shadow_bits,
      buffer_count,
      lr,
      beta1,
      beta2,
      eps,
      bias_correction1,
      sqrt_bias_correction2);
}

void launch_adamw_step_many_with_device_scale_bf16_param_float32(
    std::uint16_t* const* params_bf16_bits,
    const float* const* grads,
    const float* grad_scale,
    float* const* exp_avgs,
    float* const* exp_avg_sqs,
    const std::int64_t* elements,
    const float* weight_decays,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bias_correction1,
    float sqrt_bias_correction2,
    cudaStream_t stream) {
  if (buffer_count <= 0 || max_elements <= 0) {
    return;
  }
  const int tensor_blocks = static_cast<int>(buffer_count);
  const int element_blocks = static_cast<int>((max_elements + kOptimizerTileSize - 1) / kOptimizerTileSize);
  adamw_step_many_with_device_scale_bf16_param_float32_kernel<<<dim3(tensor_blocks, element_blocks), 1, 0, stream>>>(
      params_bf16_bits,
      grads,
      grad_scale,
      exp_avgs,
      exp_avg_sqs,
      elements,
      weight_decays,
      buffer_count,
      lr,
      beta1,
      beta2,
      eps,
      bias_correction1,
      sqrt_bias_correction2);
}

void launch_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32(
    std::uint16_t* const* params_bf16_bits,
    const std::uint16_t* const* grads_bf16_bits,
    const float* grad_scale,
    float* const* exp_avgs,
    float* const* exp_avg_sqs,
    const std::int64_t* elements,
    const float* weight_decays,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bias_correction1,
    float sqrt_bias_correction2,
    cudaStream_t stream) {
  if (buffer_count <= 0 || max_elements <= 0) {
    return;
  }
  const int tensor_blocks = static_cast<int>(buffer_count);
  const int element_blocks = static_cast<int>((max_elements + kOptimizerTileSize - 1) / kOptimizerTileSize);
  adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32_kernel<<<dim3(tensor_blocks, element_blocks), 1, 0, stream>>>(
      params_bf16_bits,
      grads_bf16_bits,
      grad_scale,
      exp_avgs,
      exp_avg_sqs,
      elements,
      weight_decays,
      buffer_count,
      lr,
      beta1,
      beta2,
      eps,
      bias_correction1,
      sqrt_bias_correction2);
}

void launch_scalar_ternary_float32(
    const float* a,
    const float* b,
    const float* c,
    float* out,
    std::int64_t n,
    float value,
    int op,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  scalar_ternary_float32_kernel<<<blocks, 1, 0, stream>>>(a, b, c, out, n, value, op);
}

void launch_vector_binary_float32(
    const float* lhs,
    const float* rhs,
    const float* scale0,
    const float* scale1,
    float* out,
    std::int64_t n,
    std::int64_t dim,
    int op,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  vector_binary_float32_kernel<<<blocks, 1, 0, stream>>>(lhs, rhs, scale0, scale1, out, n, dim, op);
}

void launch_qk_gain_float32(
    const float* q,
    const float* gain,
    float* out,
    std::int64_t n,
    std::int64_t heads,
    std::int64_t inner,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  qk_gain_float32_kernel<<<blocks, 1, 0, stream>>>(q, gain, out, n, heads, inner);
}

void launch_dyt_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t n,
    std::int64_t dim,
    float alpha,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  dyt_float32_kernel<<<blocks, 1, 0, stream>>>(x, weight, bias, out, n, dim, alpha);
}

void launch_reshape_heads_float32(
    const float* x,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim,
    cudaStream_t stream) {
  const std::int64_t n = batch * seq_len * heads * head_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  reshape_heads_float32_kernel<<<blocks, 1, 0, stream>>>(x, out, n, seq_len, heads, head_dim);
}

void launch_merge_heads_float32(
    const float* x,
    float* out,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    cudaStream_t stream) {
  const std::int64_t n = batch * heads * seq_len * head_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  merge_heads_float32_kernel<<<blocks, 1, 0, stream>>>(x, out, n, heads, seq_len, head_dim);
}

void launch_repeat_kv_float32(
    const float* x,
    float* out,
    std::int64_t batch,
    std::int64_t kv_heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    std::int64_t repeats,
    cudaStream_t stream) {
  const std::int64_t n = batch * kv_heads * repeats * seq_len * head_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  repeat_kv_float32_kernel<<<blocks, 1, 0, stream>>>(x, out, n, kv_heads, seq_len, head_dim, repeats);
}

void launch_broadcast_expert_routes_float32(
    const float* weights,
    const std::int64_t* indices,
    float* out_weights,
    std::int64_t* out_indices,
    std::int64_t batch,
    std::int64_t route_seq,
    std::int64_t seq_len,
    std::int64_t route_width,
    cudaStream_t stream) {
  const std::int64_t n = batch * seq_len * route_width;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  broadcast_expert_routes_float32_kernel<<<blocks, 1, 0, stream>>>(
      weights, indices, out_weights, out_indices, n, route_seq, seq_len, route_width);
}

void launch_broadcast_chunk_routes_float32(
    const float* weights,
    const std::int64_t* indices,
    float* out_weights,
    std::int64_t* out_indices,
    std::int64_t batch,
    std::int64_t chunks,
    std::int64_t seq_len,
    std::int64_t route_width,
    std::int64_t chunk_size,
    cudaStream_t stream) {
  const std::int64_t n = batch * seq_len * route_width;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  broadcast_chunk_routes_float32_kernel<<<blocks, 1, 0, stream>>>(
      weights, indices, out_weights, out_indices, n, chunks, seq_len, route_width, chunk_size);
}

void launch_byte_patch_merge_float32(
    const float* x,
    float* out,
    std::int64_t batch,
    std::int64_t source_len,
    std::int64_t target_len,
    std::int64_t dim,
    cudaStream_t stream) {
  const std::int64_t n = batch * target_len * dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  byte_patch_merge_float32_kernel<<<blocks, 1, 0, stream>>>(x, out, n, source_len, target_len, dim);
}

void launch_byte_patch_embed_float32(
    const std::int64_t* tokens,
    const float* embedding,
    const float* proj,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    std::int64_t patch_size,
    std::int64_t stride,
    std::int64_t out_len,
    std::int64_t vocab_size,
    cudaStream_t stream) {
  const std::int64_t n = batch * out_len * model_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  byte_patch_embed_float32_kernel<<<blocks, 1, 0, stream>>>(
      tokens, embedding, proj, out, n, seq_len, model_dim, patch_size, stride, out_len, vocab_size);
}

void launch_causal_chunk_state_float32(
    const float* hidden,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t dim,
    std::int64_t chunk_size,
    std::int64_t chunks,
    int mode,
    cudaStream_t stream) {
  const std::int64_t n = batch * chunks * dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  causal_chunk_state_float32_kernel<<<blocks, 1, 0, stream>>>(hidden, out, n, seq_len, dim, chunk_size, chunks, mode);
}

void launch_latent_mse_partials_float32(
    const float* pred,
    const float* target,
    float* partials,
    std::int64_t n,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  latent_mse_partials_float32_kernel<<<blocks, 1, 0, stream>>>(pred, target, partials, n);
}

void launch_semantic_alignment_loss_items_float32(
    const float* logits,
    const std::int64_t* targets,
    const std::int64_t* term_counts,
    float* losses,
    float* counts,
    std::int64_t n,
    std::int64_t dims,
    std::int64_t terms,
    std::int64_t ignore_index,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  semantic_alignment_loss_items_float32_kernel<<<blocks, 1, 0, stream>>>(
      logits, targets, term_counts, losses, counts, n, dims, terms, ignore_index);
}

void launch_sum_partials_float32(
    const float* values,
    float* partials,
    std::int64_t n,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  sum_partials_float32_kernel<<<blocks, 1, 0, stream>>>(values, partials, n);
}

void launch_sum_accumulate_float32(
    const float* values,
    float* total,
    std::int64_t n,
    cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((n + (kThreads * 2 - 1)) / (kThreads * 2));
  sum_accumulate_float32_kernel<<<blocks, kThreads, 0, stream>>>(values, total, n);
}

void launch_scale_float32(const float* x, float* out, std::int64_t n, float value, cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  scalar_unary_float32_kernel<<<blocks, 1, 0, stream>>>(x, out, n, value, 0);
}

void launch_kv_cache_read_float32(
    const float* k,
    const float* v,
    const float* cache_k,
    const float* cache_v,
    float* out_k,
    float* out_v,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t cache_seq,
    std::int64_t current_seq,
    std::int64_t head_dim,
    cudaStream_t stream) {
  const std::int64_t n = batch * heads * (cache_seq + current_seq) * head_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  kv_cache_read_float32_kernel<<<blocks, 1, 0, stream>>>(
      k, v, cache_k, cache_v, out_k, out_v, n, heads, cache_seq, current_seq, head_dim);
}

void launch_kv_quant_pack_float32(
    const float* k,
    const float* v,
    float* out,
    std::int64_t rows,
    std::int64_t head_dim,
    cudaStream_t stream) {
  kv_quant_pack_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(k, v, out, head_dim);
}

void launch_kv_quant_unpack_float32(
    const float* packed,
    float* out_k,
    float* out_v,
    std::int64_t rows,
    std::int64_t head_dim,
    cudaStream_t stream) {
  const std::int64_t n = rows * head_dim * 2;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  kv_quant_unpack_float32_kernel<<<blocks, 1, 0, stream>>>(packed, out_k, out_v, n, head_dim);
}

void launch_absolute_position_embedding_float32(
    const float* weight,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    cudaStream_t stream) {
  const std::int64_t n = batch * seq_len * model_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  absolute_position_embedding_float32_kernel<<<blocks, 1, 0, stream>>>(weight, out, n, seq_len, model_dim);
}

void launch_absolute_position_embedding_backward_float32(
    const float* grad_out,
    float* grad_weight,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    cudaStream_t stream) {
  const std::int64_t n = seq_len * model_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  absolute_position_embedding_backward_float32_kernel<<<blocks, 1, 0, stream>>>(
      grad_out, grad_weight, seq_len, model_dim, batch);
}

void launch_absolute_position_embedding_backward_accumulate_float32(
    const float* grad_out,
    float* grad_weight,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    cudaStream_t stream) {
  const std::int64_t n = seq_len * model_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  absolute_position_embedding_backward_accumulate_float32_kernel<<<blocks, 1, 0, stream>>>(
      grad_out, grad_weight, seq_len, model_dim, batch);
}

void launch_token_embedding_float32(
    const float* weight,
    const std::int64_t* token_ids,
    float* out,
    std::int64_t tokens,
    std::int64_t model_dim,
    cudaStream_t stream) {
  const std::int64_t n = tokens * model_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  token_embedding_float32_kernel<<<blocks, 1, 0, stream>>>(weight, token_ids, out, n, model_dim);
}

void launch_token_embedding_u16_float32(
    const float* weight,
    const std::uint16_t* token_ids,
    float* out,
    std::int64_t tokens,
    std::int64_t model_dim,
    cudaStream_t stream) {
  const std::int64_t n = tokens * model_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  token_embedding_u16_float32_kernel<<<blocks, 1, 0, stream>>>(weight, token_ids, out, n, model_dim);
}

void launch_token_position_embedding_residual_float32(
    const float* token_weight,
    const std::int64_t* token_ids,
    const float* position_weight,
    const float* scale,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    cudaStream_t stream) {
  const std::int64_t n = batch * seq_len * model_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  token_position_embedding_residual_float32_kernel<<<blocks, 1, 0, stream>>>(
      token_weight, token_ids, position_weight, scale, out, n, seq_len, model_dim);
}

void launch_token_position_embedding_residual_u16_float32(
    const float* token_weight,
    const std::uint16_t* token_ids,
    const float* position_weight,
    const float* scale,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    cudaStream_t stream) {
  const std::int64_t n = batch * seq_len * model_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  token_position_embedding_residual_u16_float32_kernel<<<blocks, 1, 0, stream>>>(
      token_weight, token_ids, position_weight, scale, out, n, seq_len, model_dim);
}

void launch_token_embedding_backward_weight_float32(
    const std::int64_t* token_ids,
    const float* grad_out,
    float* grad_weight,
    std::int64_t tokens,
    std::int64_t model_dim,
    cudaStream_t stream) {
  const std::int64_t n = tokens * model_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  token_embedding_backward_weight_float32_kernel<<<blocks, 1, 0, stream>>>(
      token_ids, grad_out, grad_weight, n, model_dim);
}

void launch_token_embedding_backward_weight_u16_float32(
    const std::uint16_t* token_ids,
    const float* grad_out,
    float* grad_weight,
    std::int64_t tokens,
    std::int64_t model_dim,
    cudaStream_t stream) {
  const std::int64_t n = tokens * model_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  token_embedding_backward_weight_u16_float32_kernel<<<blocks, 1, 0, stream>>>(
      token_ids, grad_out, grad_weight, n, model_dim);
}

void launch_rotary_embedding_float32(
    const float* x,
    const float* inv_freq,
    float* out,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    cudaStream_t stream) {
  const std::int64_t n = batch * heads * seq_len * head_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  rotary_embedding_float32_kernel<<<blocks, 1, 0, stream>>>(x, inv_freq, out, n, heads, seq_len, head_dim);
}

void launch_rms_norm_float32(
    const float* x,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    cudaStream_t stream) {
  rms_norm_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(x, out, rows, dim, eps);
}

void launch_rms_norm_backward_input_float32(
    const float* x,
    const float* grad_out,
    float* grad_x,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    cudaStream_t stream) {
  rms_norm_backward_input_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      x, grad_out, grad_x, rows, dim, eps);
}

void launch_layer_norm_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    cudaStream_t stream) {
  layer_norm_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(x, weight, bias, out, rows, dim, eps);
}

void launch_layer_norm_with_stats_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    float* mean,
    float* rstd,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    cudaStream_t stream) {
  layer_norm_with_stats_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      x, weight, bias, out, mean, rstd, rows, dim, eps);
}

void launch_layer_norm_with_stats_bf16_out_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    float* mean,
    float* rstd,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    cudaStream_t stream) {
  layer_norm_with_stats_bf16_out_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      x, weight, bias, out, mean, rstd, out_bf16_bits, rows, dim, eps);
}

void launch_layer_norm_apply_stats_bf16_out_float32(
    const float* x,
    const float* weight,
    const float* bias,
    const float* mean,
    const float* rstd,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream) {
  layer_norm_apply_stats_bf16_out_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      x, weight, bias, mean, rstd, out_bf16_bits, rows, dim);
}

void launch_layer_norm_backward_input_float32(
    const float* x,
    const float* grad_out,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    cudaStream_t stream) {
  layer_norm_backward_input_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      x, grad_out, weight, grad_x, rows, dim, eps);
}

void launch_layer_norm_backward_input_with_stats_float32(
    const float* x,
    const float* grad_out,
    const float* weight,
    const float* mean,
    const float* rstd,
    float* grad_x,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream) {
  layer_norm_backward_input_with_stats_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      x, grad_out, weight, mean, rstd, grad_x, rows, dim);
}

void launch_layer_norm_backward_input_residual_add_with_stats_float32(
    const float* x,
    const float* grad_out,
    const float* weight,
    const float* mean,
    const float* rstd,
    const float* residual_grad,
    const float* residual_scale,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream) {
  layer_norm_backward_input_residual_add_with_stats_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      x, grad_out, weight, mean, rstd, residual_grad, residual_scale, out, rows, dim);
}

void launch_layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    const float* weight,
    const float* mean,
    const float* rstd,
    const float* residual_grad,
    const float* residual_scale,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream) {
  layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      x_bf16_bits, grad_out, weight, mean, rstd, residual_grad, residual_scale, out, rows, dim);
}

void launch_layer_norm_backward_affine_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    cudaStream_t stream) {
  if (rows > 1024) {
    const std::int64_t kRowChunkSize = layer_norm_backward_affine_row_chunk_size();
    const std::int64_t dim_blocks = (dim + kTileSize - 1) / kTileSize;
    const std::int64_t row_chunks = (rows + kRowChunkSize - 1) / kRowChunkSize;
    cudaMemsetAsync(grad_weight, 0, sizeof(float) * static_cast<std::size_t>(dim), stream);
    cudaMemsetAsync(grad_bias, 0, sizeof(float) * static_cast<std::size_t>(dim), stream);
    layer_norm_backward_affine_chunked_atomic_float32_kernel<<<
        dim3(static_cast<unsigned int>(dim_blocks), static_cast<unsigned int>(row_chunks), 1),
        1,
        0,
        stream>>>(x, grad_out, grad_weight, grad_bias, rows, dim, eps, kRowChunkSize, row_chunks);
    return;
  }
  layer_norm_backward_affine_float32_kernel<<<1, 1, 0, stream>>>(
      x, grad_out, grad_weight, grad_bias, rows, dim, eps);
}

void launch_layer_norm_backward_affine_accumulate_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    cudaStream_t stream) {
  if (rows > 1024) {
    const std::int64_t kRowChunkSize = layer_norm_backward_affine_row_chunk_size();
    const std::int64_t dim_blocks = (dim + kTileSize - 1) / kTileSize;
    const std::int64_t row_chunks = (rows + kRowChunkSize - 1) / kRowChunkSize;
    layer_norm_backward_affine_chunked_atomic_float32_kernel<<<
        dim3(static_cast<unsigned int>(dim_blocks), static_cast<unsigned int>(row_chunks), 1),
        1,
        0,
        stream>>>(x, grad_out, grad_weight, grad_bias, rows, dim, eps, kRowChunkSize, row_chunks);
    return;
  }
  layer_norm_backward_affine_accumulate_float32_kernel<<<1, 1, 0, stream>>>(
      x, grad_out, grad_weight, grad_bias, rows, dim, eps);
}

void launch_layer_norm_backward_affine_accumulate_with_stats_float32(
    const float* x,
    const float* grad_out,
    const float* mean,
    const float* rstd,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream) {
  if (rows > 1024) {
    const std::int64_t kRowChunkSize = layer_norm_backward_affine_row_chunk_size();
    const std::int64_t dim_blocks = (dim + kTileSize - 1) / kTileSize;
    const std::int64_t row_chunks = (rows + kRowChunkSize - 1) / kRowChunkSize;
    layer_norm_backward_affine_chunked_atomic_with_stats_float32_kernel<<<
        dim3(static_cast<unsigned int>(dim_blocks), static_cast<unsigned int>(row_chunks), 1),
        1,
        0,
        stream>>>(x, grad_out, mean, rstd, grad_weight, grad_bias, rows, dim, kRowChunkSize, row_chunks);
    return;
  }
  layer_norm_backward_affine_accumulate_with_stats_float32_kernel<<<1, 1, 0, stream>>>(
      x, grad_out, mean, rstd, grad_weight, grad_bias, rows, dim);
}

void launch_layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    const float* mean,
    const float* rstd,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream) {
  if (rows > 1024) {
    const std::int64_t kRowChunkSize = layer_norm_backward_affine_row_chunk_size();
    const std::int64_t dim_blocks = (dim + kTileSize - 1) / kTileSize;
    const std::int64_t row_chunks = (rows + kRowChunkSize - 1) / kRowChunkSize;
    layer_norm_backward_affine_chunked_atomic_with_stats_bf16_bits_float32_kernel<<<
        dim3(static_cast<unsigned int>(dim_blocks), static_cast<unsigned int>(row_chunks), 1),
        1,
        0,
        stream>>>(x_bf16_bits, grad_out, mean, rstd, grad_weight, grad_bias, rows, dim, kRowChunkSize, row_chunks);
    return;
  }
  layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32_kernel<<<1, 1, 0, stream>>>(
      x_bf16_bits, grad_out, mean, rstd, grad_weight, grad_bias, rows, dim);
}

bool launch_layer_norm_backward_affine_residual_add_accumulate_with_stats_float32(
    const float* x,
    const float* grad_out,
    const float* weight,
    const float* mean,
    const float* rstd,
    const float* residual_grad,
    const float* residual_scale,
    float* out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream) {
  if (dim > kTileSize) {
    return false;
  }
  const std::int64_t kRowChunkSize = layer_norm_backward_affine_row_chunk_size();
  const std::int64_t dim_blocks = (dim + kTileSize - 1) / kTileSize;
  const std::int64_t row_chunks = (rows + kRowChunkSize - 1) / kRowChunkSize;
  layer_norm_backward_affine_residual_add_chunked_atomic_with_stats_float32_kernel<<<
      dim3(static_cast<unsigned int>(dim_blocks), static_cast<unsigned int>(row_chunks), 1),
      1,
      0,
      stream>>>(
      x,
      grad_out,
      weight,
      mean,
      rstd,
      residual_grad,
      residual_scale,
      out,
      grad_weight,
      grad_bias,
      rows,
      dim,
      kRowChunkSize,
      row_chunks);
  return true;
}

bool launch_layer_norm_backward_affine_residual_add_accumulate_with_stats_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    const float* weight,
    const float* mean,
    const float* rstd,
    const float* residual_grad,
    const float* residual_scale,
    float* out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream) {
  if (dim > kTileSize) {
    return false;
  }
  const std::int64_t kRowChunkSize = layer_norm_backward_affine_row_chunk_size();
  const std::int64_t dim_blocks = (dim + kTileSize - 1) / kTileSize;
  const std::int64_t row_chunks = (rows + kRowChunkSize - 1) / kRowChunkSize;
  layer_norm_backward_affine_residual_add_chunked_atomic_with_stats_bf16_bits_float32_kernel<<<
      dim3(static_cast<unsigned int>(dim_blocks), static_cast<unsigned int>(row_chunks), 1),
      1,
      0,
      stream>>>(
      x_bf16_bits,
      grad_out,
      weight,
      mean,
      rstd,
      residual_grad,
      residual_scale,
      out,
      grad_weight,
      grad_bias,
      rows,
      dim,
      kRowChunkSize,
      row_chunks);
  return true;
}

void launch_softmax_lastdim_float32(
    const float* x,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream) {
  softmax_lastdim_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(x, out, rows, dim);
}

void launch_semantic_hash_int64(
    const float* sem_vec,
    const float* proj,
    std::int64_t* out,
    std::int64_t batch,
    std::int64_t dim,
    std::int64_t tables,
    std::int64_t planes,
    cudaStream_t stream) {
  semantic_hash_int64_kernel<<<static_cast<int>(batch * tables), 1, 0, stream>>>(
      sem_vec, proj, out, batch, dim, tables, planes);
}

void launch_topk_route_float32(
    const float* logits,
    float* weights,
    std::int64_t* indices,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t top_k,
    cudaStream_t stream) {
  topk_route_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      logits, weights, indices, rows, experts, top_k);
}

void launch_attentionless_decoder_float32(
    const std::int64_t* bucket_indices,
    const float* expert_output,
    const float* bucket_embed,
    const float* out_weight,
    float* out,
    std::int64_t batch,
    std::int64_t residual_dim,
    std::int64_t vocab_size,
    std::int64_t n_buckets,
    cudaStream_t stream) {
  const std::int64_t n = batch * vocab_size;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  attentionless_decoder_float32_kernel<<<blocks, 1, 0, stream>>>(
      bucket_indices, expert_output, bucket_embed, out_weight, out, n, residual_dim, vocab_size, n_buckets);
}

void launch_expert_bias_add_float32(
    const float* logits,
    const float* bias,
    float* out,
    std::int64_t n,
    std::int64_t experts,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  expert_bias_add_float32_kernel<<<blocks, 1, 0, stream>>>(logits, bias, out, n, experts);
}

void launch_group_norm_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t dim,
    std::int64_t num_groups,
    float eps,
    cudaStream_t stream) {
  group_norm_float32_kernel<<<static_cast<int>(batch * num_groups), 1, 0, stream>>>(
      x, weight, bias, out, seq_len, dim, num_groups, eps);
}

void launch_scaled_residual_add_float32(
    const float* lhs,
    const float* rhs,
    const float* scale,
    float* out,
    std::int64_t n,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  scaled_residual_add_float32_kernel<<<blocks, 1, 0, stream>>>(lhs, rhs, scale, out, n);
}

void launch_split_qkv_float32(
    const float* qkv,
    float* q,
    float* k,
    float* v,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream) {
  const std::int64_t n = rows * dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  split_qkv_float32_kernel<<<blocks, 1, 0, stream>>>(qkv, q, k, v, rows, dim);
}

void launch_split_qkv_to_heads_float32(
    const float* qkv,
    float* q_heads,
    float* k_heads,
    float* v_heads,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim,
    cudaStream_t stream) {
  const std::int64_t n = batch * seq_len * heads * head_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  split_qkv_to_heads_float32_kernel<<<blocks, 1, 0, stream>>>(
      qkv, q_heads, k_heads, v_heads, batch, seq_len, heads, head_dim);
}

void launch_split_qkv_to_heads_add_bias_float32(
    const float* qkv,
    const float* bias,
    float* q_heads,
    float* k_heads,
    float* v_heads,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim,
    cudaStream_t stream) {
  const std::int64_t n = batch * seq_len * heads * head_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  split_qkv_to_heads_add_bias_float32_kernel<<<blocks, 1, 0, stream>>>(
      qkv, bias, q_heads, k_heads, v_heads, batch, seq_len, heads, head_dim);
}

void launch_merge_qkv_float32(
    const float* q,
    const float* k,
    const float* v,
    float* qkv,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream) {
  const std::int64_t n = rows * dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  merge_qkv_float32_kernel<<<blocks, 1, 0, stream>>>(q, k, v, qkv, rows, dim);
}

void launch_merge_heads_to_qkv_float32(
    const float* q_heads,
    const float* k_heads,
    const float* v_heads,
    float* qkv,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim,
    cudaStream_t stream) {
  const std::int64_t n = batch * seq_len * heads * head_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  merge_heads_to_qkv_float32_kernel<<<blocks, 1, 0, stream>>>(
      q_heads, k_heads, v_heads, qkv, batch, seq_len, heads, head_dim);
}

void launch_linear_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    cudaStream_t stream) {
  const std::int64_t n = rows * output_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (cublas_linear_forward_float32(x, weight, out, rows, input_dim, output_dim, stream)) {
    if (has_bias) {
      linear_add_bias_float32_kernel<<<blocks, 1, 0, stream>>>(out, bias, n, output_dim);
    }
    return;
  }
#endif
  linear_float32_kernel<<<blocks, 1, 0, stream>>>(x, weight, bias, out, n, input_dim, output_dim, has_bias);
}

void launch_linear_bf16_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    cudaStream_t stream) {
  const std::int64_t n = rows * output_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  const int m = static_cast<int>(output_dim);
  const int cols = static_cast<int>(rows);
  const int k = static_cast<int>(input_dim);
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      cublas_linear_gemm_ex_bf16_float32(
          weight,
          x,
          out,
          output_dim * input_dim,
          rows * input_dim,
          m,
          cols,
          k,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          k,
          k,
          m,
          0.0f,
          true,
          false,
          true,
          stream)) {
    if (has_bias) {
      linear_add_bias_float32_kernel<<<blocks, 1, 0, stream>>>(out, bias, n, output_dim);
    }
    return;
  }
#endif
  linear_float32_kernel<<<blocks, 1, 0, stream>>>(x, weight, bias, out, n, input_dim, output_dim, has_bias);
}

void launch_linear_weight_bf16_float32(
    const float* x,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    cudaStream_t stream) {
  const std::int64_t n = rows * output_dim;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      cublas_linear_gemm_ex_bf16_bits_a_float32(
          weight_bf16_bits,
          x,
          out,
          rows * input_dim,
          static_cast<int>(output_dim),
          static_cast<int>(rows),
          static_cast<int>(input_dim),
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          static_cast<int>(input_dim),
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          0.0f,
          true,
          stream)) {
    if (has_bias) {
      linear_add_bias_float32_kernel<<<blocks, 1, 0, stream>>>(out, bias, n, output_dim);
    }
    return;
  }
#endif
  linear_weight_bf16_bits_float32_kernel<<<blocks, threads, 0, stream>>>(
      x, weight_bf16_bits, bias, out, n, input_dim, output_dim, has_bias);
}

void launch_linear_bf16_output_float32(
    const float* x,
    const float* weight,
    const float* bias,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    cudaStream_t stream) {
  const std::int64_t n = rows * output_dim;
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (!has_bias && fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      cublas_linear_gemm_ex_bf16_float32_to_bf16_bits(
          weight,
          x,
          out_bf16_bits,
          output_dim * input_dim,
          rows * input_dim,
          static_cast<int>(output_dim),
          static_cast<int>(rows),
          static_cast<int>(input_dim),
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          static_cast<int>(input_dim),
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          // Cache the stable weight operand only; activation pointers are reused with new contents.
          true,
          false,
          true,
          stream)) {
    return;
  }
#endif
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  linear_bf16_output_float32_kernel<<<blocks, threads, 0, stream>>>(
      x, weight, bias, out_bf16_bits, n, input_dim, output_dim, has_bias);
}

void launch_linear_weight_bf16_output_float32(
    const float* x,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    cudaStream_t stream) {
  const std::int64_t n = rows * output_dim;
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      cublas_linear_gemm_ex_bf16_bits_a_float32_to_bf16_bits(
          weight_bf16_bits,
          x,
          bias,
          out_bf16_bits,
          rows * input_dim,
          static_cast<int>(output_dim),
          static_cast<int>(rows),
          static_cast<int>(input_dim),
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          static_cast<int>(input_dim),
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          true,
          has_bias,
          stream)) {
    return;
  }
#endif
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  linear_weight_bf16_output_float32_kernel<<<blocks, threads, 0, stream>>>(
      x, weight_bf16_bits, bias, out_bf16_bits, n, input_dim, output_dim, has_bias);
}

void launch_linear_bf16_input_weight_bf16_output_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    cudaStream_t stream) {
  const std::int64_t n = rows * output_dim;
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (!has_bias && fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      tk_linear_gemm_bf16_forward_to_bf16_bits(
          reinterpret_cast<const __nv_bfloat16*>(weight_bf16_bits),
          reinterpret_cast<const __nv_bfloat16*>(x_bf16_bits),
          nullptr,
          out_bf16_bits,
          static_cast<int>(rows),
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          stream)) {
    return;
  }
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      cublas_linear_gemm_ex_bf16_bits_ab_to_bf16_bits(
          weight_bf16_bits,
          x_bf16_bits,
          bias,
          out_bf16_bits,
          static_cast<int>(output_dim),
          static_cast<int>(rows),
          static_cast<int>(input_dim),
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          static_cast<int>(input_dim),
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          has_bias,
          stream)) {
    return;
  }
#endif
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  linear_bf16_input_weight_bf16_output_float32_kernel<<<blocks, threads, 0, stream>>>(
      x_bf16_bits, weight_bf16_bits, bias, out_bf16_bits, n, input_dim, output_dim, has_bias);
}

void launch_linear_bf16_input_float_weight_bf16_output_float32(
    const std::uint16_t* x_bf16_bits,
    const float* weight,
    const float* bias,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    cudaStream_t stream) {
  const std::int64_t n = rows * output_dim;
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      cublas_linear_gemm_ex_float32_a_bf16_bits_b_to_bf16_bits(
          weight,
          x_bf16_bits,
          bias,
          out_bf16_bits,
          output_dim * input_dim,
          static_cast<int>(output_dim),
          static_cast<int>(rows),
          static_cast<int>(input_dim),
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          static_cast<int>(input_dim),
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          true,
          true,
          has_bias,
          stream)) {
    return;
  }
#endif
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  linear_bf16_input_float_weight_bf16_output_float32_kernel<<<blocks, threads, 0, stream>>>(
      x_bf16_bits, weight, bias, out_bf16_bits, n, input_dim, output_dim, has_bias);
}

void launch_bf16_bits_add_bias_inplace_float32(
    std::uint16_t* values,
    const float* bias,
    std::int64_t rows,
    std::int64_t output_dim,
    cudaStream_t stream) {
  const std::int64_t n = rows * output_dim;
  if (bf16_bits_add_bias_tile_enabled()) {
    const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
    bf16_bits_add_bias_inplace_tile_float32_kernel<<<blocks, 1, 0, stream>>>(values, bias, n, output_dim);
  } else {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    bf16_bits_add_bias_inplace_kernel<<<blocks, threads, 0, stream>>>(values, bias, n, output_dim);
  }
}

void launch_linear_bf16_input_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    cudaStream_t stream) {
  const std::int64_t n = rows * output_dim;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      cublas_linear_gemm_ex_bf16_bits_b_float32(
          weight,
          x_bf16_bits,
          out,
          output_dim * input_dim,
          static_cast<int>(output_dim),
          static_cast<int>(rows),
          static_cast<int>(input_dim),
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          static_cast<int>(input_dim),
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          0.0f,
          true,
          true,
          stream)) {
    if (has_bias) {
      linear_add_bias_float32_kernel<<<blocks, 1, 0, stream>>>(out, bias, n, output_dim);
    }
    return;
  }
#endif
  linear_bf16_input_bits_float32_kernel<<<blocks, threads, 0, stream>>>(
      x_bf16_bits, weight, bias, out, n, input_dim, output_dim, has_bias);
}

void launch_linear_bf16_input_weight_bf16_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    cudaStream_t stream) {
  const std::int64_t n = rows * output_dim;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      cublas_linear_gemm_ex_bf16_bits_ab_float32(
          weight_bf16_bits,
          x_bf16_bits,
          out,
          static_cast<int>(output_dim),
          static_cast<int>(rows),
          static_cast<int>(input_dim),
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          static_cast<int>(input_dim),
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          0.0f,
          true,
          stream)) {
    if (has_bias) {
      linear_add_bias_float32_kernel<<<blocks, 1, 0, stream>>>(out, bias, n, output_dim);
    }
    return;
  }
#endif
  linear_bf16_input_weight_bf16_bits_float32_kernel<<<blocks, threads, 0, stream>>>(
      x_bf16_bits, weight_bf16_bits, bias, out, n, input_dim, output_dim, has_bias);
}

void launch_linear_bf16_gelu_bf16_float32(
    const float* x,
    const float* weight,
    const float* bias,
    std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* gelu_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      tk_linear_gemm_bf16_forward_gelu_to_bf16_bits(
          x,
          weight,
          bias,
          pre_gelu_bf16_bits,
          gelu_bf16_bits,
          rows * input_dim,
          output_dim * input_dim,
          static_cast<int>(rows),
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          stream)) {
    return;
  }
#endif
  const std::int64_t n = rows * output_dim;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  linear_bf16_gelu_bf16_float32_kernel<<<blocks, threads, 0, stream>>>(
      x, weight, bias, pre_gelu_bf16_bits, gelu_bf16_bits, n, input_dim, output_dim);
}

void launch_linear_weight_bf16_gelu_bf16_float32(
    const float* x,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* gelu_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      tk_linear_gemm_bf16_forward_gelu_weight_bf16_to_bf16_bits(
          x,
          weight_bf16_bits,
          bias,
          pre_gelu_bf16_bits,
          gelu_bf16_bits,
          rows * input_dim,
          static_cast<int>(rows),
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          stream)) {
    return;
  }
#endif
  const std::int64_t n = rows * output_dim;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  linear_weight_bf16_gelu_bf16_float32_kernel<<<blocks, threads, 0, stream>>>(
      x, weight_bf16_bits, bias, pre_gelu_bf16_bits, gelu_bf16_bits, n, input_dim, output_dim);
}

void launch_linear_bf16_input_weight_bf16_gelu_bf16_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* gelu_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (trainer_linear_tk_gemm_enabled() &&
      x_bf16_bits != nullptr &&
      weight_bf16_bits != nullptr &&
      bias != nullptr &&
      pre_gelu_bf16_bits != nullptr &&
      gelu_bf16_bits != nullptr &&
      fits_cublas_int(rows) &&
      fits_cublas_int(input_dim) &&
      fits_cublas_int(output_dim) &&
      rows > 0 &&
      input_dim > 0 &&
      output_dim > 0 &&
      rows % 128 == 0 &&
      input_dim % 64 == 0 &&
      output_dim % 128 == 0 &&
      !trainer_linear_tk_forward_shape_disabled(
          static_cast<int>(output_dim),
          static_cast<int>(rows),
          static_cast<int>(input_dim),
          CUBLAS_OP_T,
          CUBLAS_OP_N) &&
      matmul_forward_gelu_supported(
          1,
          static_cast<int>(rows),
          static_cast<int>(input_dim),
          static_cast<int>(output_dim))) {
    bool bias_cache_hit = false;
    TrainerLinearBf16Workspace* workspace =
        ensure_trainer_linear_bf16_workspace(output_dim, output_dim);
    TrainerLinearBf16Workspace::CacheEntry* bias_entry =
        workspace == nullptr ? nullptr : trainer_linear_bf16_cache_entry_for(workspace, bias, output_dim, &bias_cache_hit);
    if (bias_entry != nullptr && bias_entry->data != nullptr) {
      if (!bias_cache_hit) {
        constexpr int pack_threads = 256;
        const int pack_blocks = static_cast<int>((output_dim + pack_threads - 1) / pack_threads);
        f32_to_bf16_kernel<<<pack_blocks, pack_threads, 0, stream>>>(bias, bias_entry->data, output_dim);
        g_linear_bf16_a_pack_count.fetch_add(1, std::memory_order_relaxed);
      }
      ensure_llmk_sm120_cublaslt_initialized();
      const auto host_start = std::chrono::steady_clock::now();
      LinearShapeTiming timing = begin_linear_shape_timing(stream);
      ::matmul_forward_gelu(
          reinterpret_cast<floatX*>(gelu_bf16_bits),
          reinterpret_cast<floatX*>(pre_gelu_bf16_bits),
          reinterpret_cast<const floatX*>(x_bf16_bits),
          reinterpret_cast<const floatX*>(weight_bf16_bits),
          reinterpret_cast<const floatX*>(bias_entry->data),
          1,
          static_cast<int>(rows),
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          stream);
      const std::int64_t elapsed_us = finish_linear_shape_timing_with_host_fallback(&timing, host_start);
      g_linear_tk_gemm_count.fetch_add(1, std::memory_order_relaxed);
      g_linear_bf16_gemm_count.fetch_add(1, std::memory_order_relaxed);
      record_linear_shape_stat(
          2,
          static_cast<int>(output_dim),
          static_cast<int>(rows),
          static_cast<int>(input_dim),
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          elapsed_us);
      return;
    }
  }
#endif
  const std::int64_t n = rows * output_dim;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  linear_bf16_input_weight_bf16_gelu_bf16_float32_kernel<<<blocks, threads, 0, stream>>>(
      x_bf16_bits, weight_bf16_bits, bias, pre_gelu_bf16_bits, gelu_bf16_bits, n, input_dim, output_dim);
}

void launch_linear_backward_input_float32(
    const float* grad_out,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
  const std::int64_t n = rows * input_dim;
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (cublas_linear_backward_input_float32(grad_out, weight, grad_x, rows, input_dim, output_dim, false, stream)) {
    return;
  }
#endif
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  linear_backward_input_float32_kernel<<<blocks, 1, 0, stream>>>(
      grad_out, weight, grad_x, n, input_dim, output_dim);
}

void launch_linear_backward_input_bf16_float32(
    const float* grad_out,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (cublas_linear_backward_input_float32(grad_out, weight, grad_x, rows, input_dim, output_dim, true, stream)) {
    return;
  }
#endif
  launch_linear_backward_input_float32(grad_out, weight, grad_x, rows, input_dim, output_dim, stream);
}

void launch_linear_backward_input_weight_bf16_float32(
    const float* grad_out,
    const std::uint16_t* weight_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      cublas_linear_gemm_ex_bf16_bits_a_float32(
          weight_bf16_bits,
          grad_out,
          grad_x,
          rows * output_dim,
          static_cast<int>(input_dim),
          static_cast<int>(rows),
          static_cast<int>(output_dim),
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          static_cast<int>(input_dim),
          0.0f,
          true,
          stream)) {
    return;
  }
#endif
  const std::int64_t n = rows * input_dim;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  linear_backward_input_weight_bf16_bits_float32_kernel<<<blocks, threads, 0, stream>>>(
      grad_out, weight_bf16_bits, grad_x, n, input_dim, output_dim);
}

void launch_linear_backward_input_weight_bf16_to_bf16_bits_float32(
    const float* grad_out,
    const std::uint16_t* weight_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
  const std::int64_t n = rows * input_dim;
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      cublas_linear_gemm_ex_bf16_bits_a_float32_to_bf16_bits(
          weight_bf16_bits,
          grad_out,
          nullptr,
          grad_x_bf16_bits,
          rows * output_dim,
          static_cast<int>(input_dim),
          static_cast<int>(rows),
          static_cast<int>(output_dim),
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          static_cast<int>(input_dim),
          true,
          false,
          stream)) {
    return;
  }
#endif
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  linear_backward_input_weight_bf16_bits_to_bf16_bits_float32_kernel<<<blocks, threads, 0, stream>>>(
      grad_out, weight_bf16_bits, grad_x_bf16_bits, n, input_dim, output_dim);
}

void launch_linear_backward_input_bf16_bits_weight_bf16_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      tk_linear_backward_input_bf16_bits_weight_bf16_bits_float32(
          grad_out_bf16_bits,
          weight_bf16_bits,
          grad_x,
          static_cast<int>(rows),
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          stream)) {
    return;
  }
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      cublas_linear_gemm_ex_bf16_bits_ab_float32(
          weight_bf16_bits,
          grad_out_bf16_bits,
          grad_x,
          static_cast<int>(input_dim),
          static_cast<int>(rows),
          static_cast<int>(output_dim),
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          static_cast<int>(input_dim),
          0.0f,
          true,
          stream)) {
    return;
  }
#endif
  const std::int64_t n = rows * input_dim;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  linear_backward_input_bf16_bits_weight_bf16_bits_float32_kernel<<<blocks, threads, 0, stream>>>(
      grad_out_bf16_bits, weight_bf16_bits, grad_x, n, input_dim, output_dim);
}

void launch_linear_backward_input_bf16_bits_weight_bf16_strided_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t grad_out_row_stride,
    cudaStream_t stream) {
  if (grad_out_row_stride == output_dim) {
    launch_linear_backward_input_bf16_bits_weight_bf16_float32(
        grad_out_bf16_bits, weight_bf16_bits, grad_x, rows, input_dim, output_dim, stream);
    return;
  }
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      fits_cublas_int(grad_out_row_stride) &&
      cublas_linear_gemm_ex_bf16_bits_ab_float32(
          weight_bf16_bits,
          grad_out_bf16_bits,
          grad_x,
          static_cast<int>(input_dim),
          static_cast<int>(rows),
          static_cast<int>(output_dim),
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          static_cast<int>(input_dim),
          static_cast<int>(grad_out_row_stride),
          static_cast<int>(input_dim),
          0.0f,
          true,
          stream)) {
    return;
  }
#endif
  const std::int64_t n = rows * input_dim;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  linear_backward_input_bf16_bits_weight_bf16_bits_strided_float32_kernel<<<blocks, threads, 0, stream>>>(
      grad_out_bf16_bits, weight_bf16_bits, grad_x, n, input_dim, output_dim, grad_out_row_stride);
}

void launch_linear_backward_input_bf16_bits_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      cublas_linear_gemm_ex_bf16_bits_b_float32(
          weight,
          grad_out_bf16_bits,
          grad_x,
          output_dim * input_dim,
          static_cast<int>(input_dim),
          static_cast<int>(rows),
          static_cast<int>(output_dim),
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          static_cast<int>(input_dim),
          0.0f,
          true,
          true,
          stream)) {
    return;
  }
#endif
  const std::int64_t n = rows * input_dim;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  linear_backward_input_bf16_bits_float32_kernel<<<blocks, threads, 0, stream>>>(
      grad_out_bf16_bits, weight, grad_x, n, input_dim, output_dim);
}

void launch_gelu_backward_inplace_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    float* grad_in_out,
    std::int64_t n,
    cudaStream_t stream);
void launch_gelu_backward_inplace_bf16_bits_to_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    std::uint16_t* grad_bf16_bits,
    std::int64_t n,
    cudaStream_t stream);

void launch_linear_backward_input_dgelu_bf16_bits_float32(
    const float* grad_out,
    const float* weight,
    const std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      tk_linear_backward_input_dgelu_bf16_bits_float32(
          grad_out,
          weight,
          pre_gelu_bf16_bits,
          grad_x_bf16_bits,
          grad_x,
          rows * output_dim,
          output_dim * input_dim,
          static_cast<int>(rows),
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          stream)) {
    return;
  }
#endif
  launch_linear_backward_input_bf16_float32(grad_out, weight, grad_x, rows, input_dim, output_dim, stream);
  launch_gelu_backward_inplace_bf16_bits_float32(pre_gelu_bf16_bits, grad_x, rows * input_dim, stream);
}

void launch_linear_backward_input_dgelu_weight_bf16_bits_float32(
    const float* grad_out,
    const std::uint16_t* weight_bf16_bits,
    const std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      tk_linear_backward_input_dgelu_weight_bf16_bits_float32(
          grad_out,
          weight_bf16_bits,
          pre_gelu_bf16_bits,
          grad_x_bf16_bits,
          grad_x,
          rows * output_dim,
          static_cast<int>(rows),
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          true,
          stream)) {
    return;
  }
#endif
  launch_linear_backward_input_weight_bf16_float32(
      grad_out, weight_bf16_bits, grad_x, rows, input_dim, output_dim, stream);
  launch_gelu_backward_inplace_bf16_bits_float32(pre_gelu_bf16_bits, grad_x, rows * input_dim, stream);
}

void launch_linear_backward_input_dgelu_weight_bf16_bits_only_float32(
    const float* grad_out,
    const std::uint16_t* weight_bf16_bits,
    const std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    float* grad_x_fallback,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
  const std::int64_t grad_x_elements = rows * input_dim;
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      tk_linear_backward_input_dgelu_weight_bf16_bits_float32(
          grad_out,
          weight_bf16_bits,
          pre_gelu_bf16_bits,
          grad_x_bf16_bits,
          nullptr,
          rows * output_dim,
          static_cast<int>(rows),
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          false,
          stream)) {
    return;
  }
#endif
  launch_linear_backward_input_weight_bf16_float32(
      grad_out, weight_bf16_bits, grad_x_fallback, rows, input_dim, output_dim, stream);
  launch_gelu_backward_inplace_bf16_bits_float32(pre_gelu_bf16_bits, grad_x_fallback, grad_x_elements, stream);
  constexpr int threads = 256;
  const int blocks = static_cast<int>((grad_x_elements + threads - 1) / threads);
  f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
      grad_x_fallback, reinterpret_cast<__nv_bfloat16*>(grad_x_bf16_bits), grad_x_elements);
}

void launch_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_only_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    const std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      tk_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_float32(
          grad_out_bf16_bits,
          weight_bf16_bits,
          pre_gelu_bf16_bits,
          grad_x_bf16_bits,
          static_cast<int>(rows),
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          stream)) {
    return;
  }
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      cublas_linear_gemm_ex_bf16_bits_ab_to_bf16_bits(
          weight_bf16_bits,
          grad_out_bf16_bits,
          nullptr,
          grad_x_bf16_bits,
          static_cast<int>(input_dim),
          static_cast<int>(rows),
          static_cast<int>(output_dim),
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          static_cast<int>(input_dim),
          false,
          stream)) {
    launch_gelu_backward_inplace_bf16_bits_to_bf16_bits_float32(
        pre_gelu_bf16_bits,
        grad_x_bf16_bits,
        rows * input_dim,
        stream);
    return;
  }
#endif
}

void launch_linear_backward_weight_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
  const std::int64_t n = output_dim * input_dim;
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (cublas_linear_backward_weight_float32(x, grad_out, grad_weight, rows, input_dim, output_dim, 0.0f, stream)) {
    return;
  }
#endif
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  if (rows <= kTileSize) {
    linear_backward_weight_float32_kernel<<<blocks, 1, 0, stream>>>(
        x, grad_out, grad_weight, n, rows, input_dim, output_dim);
    return;
  }
  launch_linear_backward_weight_tiled_float32_fallback(
      x, nullptr, grad_out, nullptr, grad_weight, rows, input_dim, output_dim, false, false, false, stream);
}

void launch_linear_backward_weight_accumulate_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
  const std::int64_t n = output_dim * input_dim;
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (cublas_linear_backward_weight_float32(x, grad_out, grad_weight, rows, input_dim, output_dim, 1.0f, stream)) {
    return;
  }
#endif
  (void)n;
  launch_linear_backward_weight_tiled_float32_fallback(
      x, nullptr, grad_out, nullptr, grad_weight, rows, input_dim, output_dim, false, false, true, stream);
}

void launch_linear_backward_weight_accumulate_bf16_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      cublas_linear_gemm_ex_bf16_float32(
          x,
          grad_out,
          grad_weight,
          rows * input_dim,
          rows * output_dim,
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          static_cast<int>(rows),
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          static_cast<int>(input_dim),
          1.0f,
          false,
          false,
          true,
          stream)) {
    return;
  }
#endif
  launch_linear_backward_weight_accumulate_float32(
      x, grad_out, grad_weight, rows, input_dim, output_dim, stream);
}

void launch_linear_backward_weight_accumulate_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      cublas_linear_gemm_ex_bf16_bits_a_float32(
          x_bf16_bits,
          grad_out,
          grad_weight,
          rows * output_dim,
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          static_cast<int>(rows),
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          static_cast<int>(input_dim),
          1.0f,
          true,
          stream)) {
    return;
  }
#endif
  launch_linear_backward_weight_tiled_float32_fallback(
      nullptr, x_bf16_bits, grad_out, nullptr, grad_weight, rows, input_dim, output_dim, true, false, true, stream);
}

void
launch_linear_backward_bias_accumulate_float32(
    const float* grad_out,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t output_dim,
    cudaStream_t stream);

void launch_linear_backward_weight_bias_accumulate_bf16_bits_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    cudaStream_t stream);

void launch_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    cudaStream_t stream);

void launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    cudaStream_t stream);

void launch_linear_backward_weight_bias_accumulate_float32_bf16_bits_beta(
    const float* x,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    cudaStream_t stream);

void launch_linear_backward_weight_bias_accumulate_bf16_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim)) {
    float* bias_gradient = ensure_trainer_linear_bgrad_workspace(output_dim);
    if (bias_gradient != nullptr &&
        cublas_linear_gemm_ex_bf16_float32_with_bgrad(
            x,
            grad_out,
            grad_weight,
            bias_gradient,
            rows * input_dim,
            rows * output_dim,
            static_cast<int>(input_dim),
            static_cast<int>(output_dim),
            static_cast<int>(rows),
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            static_cast<int>(input_dim),
            static_cast<int>(output_dim),
            static_cast<int>(input_dim),
            1.0f,
            false,
            false,
            true,
            stream)) {
      g_linear_cublaslt_bgrad_accumulate_count.fetch_add(1, std::memory_order_relaxed);
      launch_gradient_accumulate_float32(grad_bias, bias_gradient, output_dim, 1.0f, stream);
      return;
    }
  }
#endif
  launch_linear_backward_weight_accumulate_bf16_float32(
      x, grad_out, grad_weight, rows, input_dim, output_dim, stream);
  launch_linear_backward_bias_accumulate_float32(grad_out, grad_bias, rows, output_dim, stream);
}

void launch_linear_backward_weight_bias_accumulate_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
  launch_linear_backward_weight_bias_accumulate_bf16_bits_float32_beta(
      x_bf16_bits, grad_out, grad_weight, grad_bias, rows, input_dim, output_dim, 1.0f, stream);
}

void launch_linear_backward_weight_bias_accumulate_bf16_bits_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim)) {
    const bool first_write_bias = beta == 0.0f && trainer_linear_bgrad_first_write_direct_enabled();
    float* bias_gradient = first_write_bias ? grad_bias : ensure_trainer_linear_bgrad_workspace(output_dim);
    if (bias_gradient != nullptr &&
        cublas_linear_gemm_ex_bf16_bits_a_float32_with_bgrad(
            x_bf16_bits,
            grad_out,
            grad_weight,
            bias_gradient,
            rows * output_dim,
            static_cast<int>(input_dim),
            static_cast<int>(output_dim),
            static_cast<int>(rows),
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            static_cast<int>(input_dim),
            static_cast<int>(output_dim),
            static_cast<int>(input_dim),
            beta,
            true,
            stream)) {
      if (!first_write_bias) {
        g_linear_cublaslt_bgrad_accumulate_count.fetch_add(1, std::memory_order_relaxed);
        launch_gradient_accumulate_float32(grad_bias, bias_gradient, output_dim, 1.0f, stream);
      } else {
        g_linear_cublaslt_bgrad_direct_write_count.fetch_add(1, std::memory_order_relaxed);
      }
      return;
    }
  }
#endif
  launch_linear_backward_weight_accumulate_bf16_bits_float32(
      x_bf16_bits, grad_out, grad_weight, rows, input_dim, output_dim, stream);
  launch_linear_backward_bias_accumulate_float32(grad_out, grad_bias, rows, output_dim, stream);
}

void launch_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
  launch_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32_beta(
      x_bf16_bits, grad_out_bf16_bits, grad_weight, grad_bias, rows, input_dim, output_dim, 1.0f, stream);
}

bool tk_linear_backward_weight_bf16_bits_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    cudaStream_t stream);

void launch_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    cudaStream_t stream) {
  if (tk_linear_backward_weight_bf16_bits_bf16_bits_float32(
          x_bf16_bits, grad_out_bf16_bits, grad_weight, rows, input_dim, output_dim, beta, stream)) {
    const std::int64_t kRowChunkSize = linear_backward_bias_row_chunk_size();
    const std::int64_t row_chunks = (rows + kRowChunkSize - 1) / kRowChunkSize;
    const int threads = linear_backward_bias_threads_per_block_value();
    const int bias_blocks = static_cast<int>((output_dim + threads - 1) / threads);
    dim3 bias_grid(static_cast<unsigned int>(bias_blocks), static_cast<unsigned int>(row_chunks), 1);
    linear_backward_bias_chunked_atomic_bf16_bits_float32_kernel<<<bias_grid, threads, 0, stream>>>(
        grad_out_bf16_bits, grad_bias, output_dim, rows, kRowChunkSize);
    return;
  }
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim)) {
    const int m = static_cast<int>(input_dim);
    const int n = static_cast<int>(output_dim);
    const int k = static_cast<int>(rows);
    const cublasOperation_t op_a = CUBLAS_OP_N;
    const cublasOperation_t op_b = CUBLAS_OP_T;
    if ((!trainer_linear_bf16_bf16_bgrad_enabled() ||
         trainer_linear_bf16_bf16_bgrad_disabled_for_shape(m, n, k, op_a, op_b)) &&
        cublas_linear_gemm_ex_bf16_bits_ab_float32(
            x_bf16_bits,
            grad_out_bf16_bits,
            grad_weight,
            m,
            n,
            k,
            op_a,
            op_b,
            m,
            n,
            m,
            beta,
            true,
            stream)) {
      const std::int64_t kRowChunkSize = linear_backward_bias_row_chunk_size();
      const std::int64_t row_chunks = (rows + kRowChunkSize - 1) / kRowChunkSize;
      const int threads = linear_backward_bias_threads_per_block_value();
      const int bias_blocks = static_cast<int>((output_dim + threads - 1) / threads);
      dim3 bias_grid(static_cast<unsigned int>(bias_blocks), static_cast<unsigned int>(row_chunks), 1);
      linear_backward_bias_chunked_atomic_bf16_bits_float32_kernel<<<bias_grid, threads, 0, stream>>>(
          grad_out_bf16_bits, grad_bias, output_dim, rows, kRowChunkSize);
      return;
    }
    const bool first_write_bias = beta == 0.0f && trainer_linear_bgrad_first_write_direct_enabled();
    float* bias_gradient = first_write_bias ? grad_bias : ensure_trainer_linear_bgrad_workspace(output_dim);
    if (bias_gradient != nullptr &&
        cublas_linear_gemm_ex_bf16_bits_ab_float32_with_bgrad(
            x_bf16_bits,
            grad_out_bf16_bits,
            grad_weight,
            bias_gradient,
            m,
            n,
            k,
            op_a,
            op_b,
            m,
            n,
            m,
            beta,
            true,
            stream)) {
      if (!first_write_bias) {
        g_linear_cublaslt_bgrad_accumulate_count.fetch_add(1, std::memory_order_relaxed);
        launch_gradient_accumulate_float32(grad_bias, bias_gradient, output_dim, 1.0f, stream);
      } else {
        g_linear_cublaslt_bgrad_direct_write_count.fetch_add(1, std::memory_order_relaxed);
      }
      return;
    }
  }
#endif
  launch_linear_backward_weight_tiled_float32_fallback(
      nullptr, x_bf16_bits, nullptr, grad_out_bf16_bits, grad_weight, rows, input_dim, output_dim, true, true, beta != 0.0f, stream);
  const std::int64_t kRowChunkSize = linear_backward_bias_row_chunk_size();
  const std::int64_t row_chunks = (rows + kRowChunkSize - 1) / kRowChunkSize;
  const int threads = linear_backward_bias_threads_per_block_value();
  const int bias_blocks = static_cast<int>((output_dim + threads - 1) / threads);
  dim3 bias_grid(static_cast<unsigned int>(bias_blocks), static_cast<unsigned int>(row_chunks), 1);
  linear_backward_bias_chunked_atomic_bf16_bits_float32_kernel<<<bias_grid, threads, 0, stream>>>(
      grad_out_bf16_bits, grad_bias, output_dim, rows, kRowChunkSize);
}

void launch_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_to_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    std::uint16_t* grad_weight_bf16_bits,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
  constexpr int threads = 256;
  const int bias_threads = linear_backward_bias_threads_per_block_value();
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      cublas_linear_gemm_ex_bf16_bits_ab_to_bf16_bits_accumulate(
          x_bf16_bits,
          grad_out_bf16_bits,
          grad_weight_bf16_bits,
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          static_cast<int>(rows),
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          static_cast<int>(input_dim),
          stream)) {
    const std::int64_t kRowChunkSize = linear_backward_bias_row_chunk_size();
    const std::int64_t row_chunks = (rows + kRowChunkSize - 1) / kRowChunkSize;
    const int bias_blocks = static_cast<int>((output_dim + bias_threads - 1) / bias_threads);
    dim3 bias_grid(static_cast<unsigned int>(bias_blocks), static_cast<unsigned int>(row_chunks), 1);
    linear_backward_bias_chunked_atomic_bf16_bits_float32_kernel<<<bias_grid, bias_threads, 0, stream>>>(
        grad_out_bf16_bits, grad_bias, output_dim, rows, kRowChunkSize);
    return;
  }
#endif
  const std::int64_t n = output_dim * input_dim;
  const int weight_blocks = static_cast<int>((n + threads - 1) / threads);
  linear_backward_weight_accumulate_bf16_bits_bf16_bits_bf16_bits_kernel<<<weight_blocks, threads, 0, stream>>>(
      x_bf16_bits, grad_out_bf16_bits, grad_weight_bf16_bits, n, rows, input_dim, output_dim);
  const std::int64_t kRowChunkSize = linear_backward_bias_row_chunk_size();
  const std::int64_t row_chunks = (rows + kRowChunkSize - 1) / kRowChunkSize;
  const int bias_blocks = static_cast<int>((output_dim + bias_threads - 1) / bias_threads);
  dim3 bias_grid(static_cast<unsigned int>(bias_blocks), static_cast<unsigned int>(row_chunks), 1);
  linear_backward_bias_chunked_atomic_bf16_bits_float32_kernel<<<bias_grid, bias_threads, 0, stream>>>(
      grad_out_bf16_bits, grad_bias, output_dim, rows, kRowChunkSize);
}

void launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
  launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta(
      x_bf16_bits, grad_out_bf16_bits, grad_weight, rows, input_dim, output_dim, 1.0f, stream);
}

bool tk_linear_backward_weight_bf16_bits_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  constexpr cublasOperation_t kOpA = CUBLAS_OP_N;
  constexpr cublasOperation_t kOpB = CUBLAS_OP_T;
  if (rows <= 0 || input_dim <= 0 || output_dim <= 0 ||
      rows > std::numeric_limits<int>::max() ||
      input_dim > std::numeric_limits<int>::max() ||
      output_dim > std::numeric_limits<int>::max()) {
    return false;
  }
  const int m = static_cast<int>(input_dim);
  const int n = static_cast<int>(output_dim);
  const int k = static_cast<int>(rows);
  const bool shape_enabled = trainer_linear_tk_dweight_shape_enabled(m, n, k, kOpA, kOpB);
  if (!trainer_linear_tk_gemm_enabled() ||
      trainer_linear_tk_dweight_shape_disabled(m, n, k, kOpA, kOpB) ||
      (!trainer_linear_tk_dweight_enabled() && !shape_enabled)) {
    return false;
  }
  if (x_bf16_bits == nullptr || grad_out_bf16_bits == nullptr || grad_weight == nullptr) {
    return false;
  }
  if (rows % 64 != 0 || input_dim % 128 != 0 || output_dim % 128 != 0) {
    return false;
  }
  const std::int64_t weight_elements = input_dim * output_dim;
  TrainerLinearBf16Workspace* workspace = ensure_trainer_linear_bf16_workspace(1, 1, weight_elements);
  if (workspace == nullptr || workspace->c_bits == nullptr) {
    return false;
  }
  ensure_llmk_sm120_cublaslt_initialized();
  const auto host_start = std::chrono::steady_clock::now();
  LinearShapeTiming timing = begin_linear_shape_timing(stream);
  ::matmul_dispatch_tk_atb(
      reinterpret_cast<floatX*>(workspace->c_bits),
      reinterpret_cast<const floatX*>(grad_out_bf16_bits),
      reinterpret_cast<const floatX*>(x_bf16_bits),
      static_cast<int>(output_dim),
      static_cast<int>(input_dim),
      static_cast<int>(rows),
      stream);
  constexpr int threads = 256;
  const int blocks = static_cast<int>((weight_elements + threads - 1) / threads);
  bf16_bits_accumulate_to_float32_kernel<<<blocks, threads, 0, stream>>>(
      grad_weight, workspace->c_bits, weight_elements, beta);
  const std::int64_t elapsed_us = finish_linear_shape_timing_with_host_fallback(&timing, host_start);
  g_linear_tk_gemm_count.fetch_add(1, std::memory_order_relaxed);
  g_linear_tk_dweight_gemm_count.fetch_add(1, std::memory_order_relaxed);
  g_linear_bf16_gemm_count.fetch_add(1, std::memory_order_relaxed);
  record_linear_shape_stat(2, m, n, k, kOpA, kOpB, elapsed_us);
  return true;
#else
  (void)x_bf16_bits;
  (void)grad_out_bf16_bits;
  (void)grad_weight;
  (void)rows;
  (void)input_dim;
  (void)output_dim;
  (void)beta;
  (void)stream;
  return false;
#endif
}

void launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    cudaStream_t stream) {
  if (tk_linear_backward_weight_bf16_bits_bf16_bits_float32(
          x_bf16_bits, grad_out_bf16_bits, grad_weight, rows, input_dim, output_dim, beta, stream)) {
    return;
  }
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      cublas_linear_gemm_ex_bf16_bits_ab_float32(
          x_bf16_bits,
          grad_out_bf16_bits,
          grad_weight,
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          static_cast<int>(rows),
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          static_cast<int>(input_dim),
          beta,
          true,
          stream)) {
    return;
  }
#endif
  launch_linear_backward_weight_tiled_float32_fallback(
      nullptr, x_bf16_bits, nullptr, grad_out_bf16_bits, grad_weight, rows, input_dim, output_dim, true, true, beta != 0.0f, stream);
}

void launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t grad_out_row_stride,
    float beta,
    cudaStream_t stream) {
  if (grad_out_row_stride == output_dim) {
    launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta(
        x_bf16_bits, grad_out_bf16_bits, grad_weight, rows, input_dim, output_dim, beta, stream);
    return;
  }
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      fits_cublas_int(grad_out_row_stride) &&
      cublas_linear_gemm_ex_bf16_bits_ab_float32(
          x_bf16_bits,
          grad_out_bf16_bits,
          grad_weight,
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          static_cast<int>(rows),
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          static_cast<int>(input_dim),
          static_cast<int>(grad_out_row_stride),
          static_cast<int>(input_dim),
          beta,
          true,
          stream)) {
    return;
  }
#endif
  launch_linear_backward_weight_tiled_strided_float32_fallback(
      nullptr,
      x_bf16_bits,
      nullptr,
      grad_out_bf16_bits,
      grad_weight,
      rows,
      input_dim,
      output_dim,
      grad_out_row_stride,
      true,
      true,
      beta != 0.0f,
      stream);
}

void launch_linear_backward_weight_accumulate_float32_bf16_bits(
    const float* x,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim) &&
      cublas_linear_gemm_ex_bf16_bits_b_float32(
          x,
          grad_out_bf16_bits,
          grad_weight,
          rows * input_dim,
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          static_cast<int>(rows),
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          static_cast<int>(input_dim),
          static_cast<int>(output_dim),
          static_cast<int>(input_dim),
          1.0f,
          // x is a reused activation scratch pointer whose contents change across
          // gradient-accumulation microbatches; caching by pointer would reuse stale BF16 data.
          false,
          true,
          stream)) {
    return;
  }
#endif
  launch_linear_backward_weight_tiled_float32_fallback(
      x, nullptr, nullptr, grad_out_bf16_bits, grad_weight, rows, input_dim, output_dim, false, true, true, stream);
}

void launch_linear_backward_weight_bias_accumulate_float32_bf16_bits(
    const float* x,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream) {
  // Fallback chunking is applied by the beta overload with
  // kRowChunkSize = linear_backward_bias_row_chunk_size().
  launch_linear_backward_weight_bias_accumulate_float32_bf16_bits_beta(
      x, grad_out_bf16_bits, grad_weight, grad_bias, rows, input_dim, output_dim, 1.0f, stream);
}

void launch_linear_backward_weight_bias_accumulate_float32_bf16_bits_beta(
    const float* x,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (trainer_linear_float32_bf16_bgrad_enabled() &&
      fits_cublas_int(rows) && fits_cublas_int(input_dim) && fits_cublas_int(output_dim)) {
    const bool first_write_bias = beta == 0.0f && trainer_linear_bgrad_first_write_direct_enabled();
    float* bias_gradient = first_write_bias ? grad_bias : ensure_trainer_linear_bgrad_workspace(output_dim);
    if (bias_gradient != nullptr &&
        cublas_linear_gemm_ex_bf16_bits_b_float32_with_bgrad(
            x,
            grad_out_bf16_bits,
            grad_weight,
            bias_gradient,
            rows * input_dim,
            static_cast<int>(input_dim),
            static_cast<int>(output_dim),
            static_cast<int>(rows),
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            static_cast<int>(input_dim),
            static_cast<int>(output_dim),
            static_cast<int>(input_dim),
            beta,
            // x is a reused activation scratch pointer whose contents change across
            // gradient-accumulation microbatches; caching by pointer would reuse stale BF16 data.
            false,
            true,
            stream)) {
      if (!first_write_bias) {
        g_linear_cublaslt_bgrad_accumulate_count.fetch_add(1, std::memory_order_relaxed);
        launch_gradient_accumulate_float32(grad_bias, bias_gradient, output_dim, 1.0f, stream);
      } else {
        g_linear_cublaslt_bgrad_direct_write_count.fetch_add(1, std::memory_order_relaxed);
      }
      return;
    }
  }
#endif
  launch_linear_backward_weight_tiled_float32_fallback(
      x, nullptr, nullptr, grad_out_bf16_bits, grad_weight, rows, input_dim, output_dim, false, true, beta != 0.0f, stream);
  const std::int64_t kRowChunkSize = linear_backward_bias_row_chunk_size();
  const std::int64_t row_chunks = (rows + kRowChunkSize - 1) / kRowChunkSize;
  const int threads = linear_backward_bias_threads_per_block_value();
  const int bias_blocks = static_cast<int>((output_dim + threads - 1) / threads);
  dim3 bias_grid(static_cast<unsigned int>(bias_blocks), static_cast<unsigned int>(row_chunks), 1);
  linear_backward_bias_chunked_atomic_bf16_bits_float32_kernel<<<bias_grid, threads, 0, stream>>>(
      grad_out_bf16_bits, grad_bias, output_dim, rows, kRowChunkSize);
}

void launch_linear_backward_bias_float32(
    const float* grad_out,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (rows <= kTileSize && cublas_linear_backward_bias_float32(grad_out, grad_bias, rows, output_dim, 0.0f, stream)) {
    return;
  }
#endif
  const int blocks = static_cast<int>((output_dim + kTileSize - 1) / kTileSize);
  if (rows <= kTileSize) {
    linear_backward_bias_float32_kernel<<<blocks, 1, 0, stream>>>(grad_out, grad_bias, output_dim, rows);
    return;
  }
  const std::int64_t kRowChunkSize = linear_backward_bias_row_chunk_size();
  const std::int64_t row_chunks = (rows + kRowChunkSize - 1) / kRowChunkSize;
  fill_float32_kernel<<<blocks, 1, 0, stream>>>(grad_bias, output_dim, 0.0f);
  dim3 grid(static_cast<unsigned int>(blocks), static_cast<unsigned int>(row_chunks), 1);
  linear_backward_bias_chunked_atomic_float32_kernel<<<grid, 1, 0, stream>>>(
      grad_out, grad_bias, output_dim, rows, kRowChunkSize, row_chunks);
}

void launch_linear_backward_bias_accumulate_float32(
    const float* grad_out,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t output_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  if (rows <= kTileSize && cublas_linear_backward_bias_float32(grad_out, grad_bias, rows, output_dim, 1.0f, stream)) {
    return;
  }
#endif
  const int blocks = static_cast<int>((output_dim + kTileSize - 1) / kTileSize);
  if (rows <= kTileSize) {
    linear_backward_bias_accumulate_float32_kernel<<<blocks, 1, 0, stream>>>(
        grad_out, grad_bias, output_dim, rows);
    return;
  }
  const std::int64_t kRowChunkSize = linear_backward_bias_row_chunk_size();
  const std::int64_t row_chunks = (rows + kRowChunkSize - 1) / kRowChunkSize;
  dim3 grid(static_cast<unsigned int>(blocks), static_cast<unsigned int>(row_chunks), 1);
  linear_backward_bias_chunked_atomic_float32_kernel<<<grid, 1, 0, stream>>>(
      grad_out, grad_bias, output_dim, rows, kRowChunkSize, row_chunks);
}

void launch_gelu_float32(
    const float* x,
    float* out,
    std::int64_t n,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  gelu_float32_kernel<<<blocks, 1, 0, stream>>>(x, out, n);
}

void launch_gelu_add_bias_float32(
    const float* x,
    const float* bias,
    float* biased_out,
    float* gelu_out,
    std::int64_t rows,
    std::int64_t output_dim,
    cudaStream_t stream) {
  const std::int64_t n = rows * output_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  gelu_add_bias_float32_kernel<<<blocks, 1, 0, stream>>>(
      x, bias, biased_out, gelu_out, n, output_dim);
}

void launch_gelu_add_bias_bf16_act_float32(
    const float* x,
    const float* bias,
    float* biased_out,
    float* gelu_out,
    std::uint16_t* gelu_bf16_bits,
    std::int64_t rows,
    std::int64_t output_dim,
    cudaStream_t stream) {
  const std::int64_t n = rows * output_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  gelu_add_bias_bf16_act_float32_kernel<<<blocks, 1, 0, stream>>>(
      x, bias, biased_out, gelu_out, gelu_bf16_bits, n, output_dim);
}

void launch_linear_bias_residual_add_float32(
    const float* residual,
    const float* linear_out,
    const float* bias,
    const float* residual_scale,
    float* out,
    std::int64_t rows,
    std::int64_t output_dim,
    cudaStream_t stream) {
  const std::int64_t n = rows * output_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  linear_bias_residual_add_float32_kernel<<<blocks, 1, 0, stream>>>(
      residual, linear_out, bias, residual_scale, out, n, output_dim);
}

void launch_linear_bias_residual_add_bf16_linear_float32(
    const float* residual,
    const std::uint16_t* linear_out_bf16_bits,
    const float* bias,
    const float* residual_scale,
    float* out,
    std::int64_t rows,
    std::int64_t output_dim,
    cudaStream_t stream) {
  if (output_dim == 768 && dim768_bf16_residual_add_enabled()) {
    constexpr int kRowsPerBlock = 2;
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
    linear_bias_residual_add_bf16_linear_dim768_float32_kernel<<<blocks, kThreads, 0, stream>>>(
        residual, linear_out_bf16_bits, bias, residual_scale, out, rows);
    return;
  }
  const std::int64_t n = rows * output_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  linear_bias_residual_add_bf16_linear_float32_kernel<<<blocks, 1, 0, stream>>>(
      residual, linear_out_bf16_bits, bias, residual_scale, out, n, output_dim);
}

void launch_linear_bias_residual_add_bf16_linear_bf16_residual_float32(
    const float* residual,
    const std::uint16_t* linear_out_bf16_bits,
    const float* bias,
    const float* residual_scale,
    float* out,
    std::uint16_t* residual_bf16_out,
    std::int64_t rows,
    std::int64_t output_dim,
    cudaStream_t stream) {
  const std::int64_t n = rows * output_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  linear_bias_residual_add_bf16_linear_bf16_residual_float32_kernel<<<blocks, 1, 0, stream>>>(
      residual, linear_out_bf16_bits, bias, residual_scale, out, residual_bf16_out, n, output_dim);
}

void launch_linear_bias_residual_layer_norm_float32(
    const float* residual,
    const float* linear_out,
    const float* linear_bias,
    const float* residual_scale,
    const float* norm_weight,
    const float* norm_bias,
    float* residual_out,
    float* norm_out,
    std::int64_t rows,
    std::int64_t output_dim,
    float eps,
    cudaStream_t stream) {
  linear_bias_residual_layer_norm_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      residual,
      linear_out,
      linear_bias,
      residual_scale,
      norm_weight,
      norm_bias,
      residual_out,
      norm_out,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      rows,
      output_dim,
      eps);
}

void launch_linear_bias_residual_layer_norm_with_stats_bf16_linear_float32(
    const float* residual,
    const std::uint16_t* linear_out_bf16_bits,
    const float* linear_bias,
    const float* residual_scale,
    const float* norm_weight,
    const float* norm_bias,
    float* residual_out,
    float* norm_out,
    float* mean_out,
    float* rstd_out,
    std::int64_t rows,
    std::int64_t output_dim,
    float eps,
    cudaStream_t stream) {
  linear_bias_residual_layer_norm_bf16_linear_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      residual,
      linear_out_bf16_bits,
      linear_bias,
      residual_scale,
      norm_weight,
      norm_bias,
      residual_out,
      norm_out,
      mean_out,
      rstd_out,
      nullptr,
      nullptr,
      rows,
      output_dim,
      eps);
}

void launch_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_float32(
    const float* residual,
    const std::uint16_t* linear_out_bf16_bits,
    const float* linear_bias,
    const float* residual_scale,
    const float* norm_weight,
    const float* norm_bias,
    float* residual_out,
    float* norm_out,
    float* mean_out,
    float* rstd_out,
    std::uint16_t* residual_bf16_out,
    std::int64_t rows,
    std::int64_t output_dim,
    float eps,
    cudaStream_t stream) {
  linear_bias_residual_layer_norm_bf16_linear_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      residual,
      linear_out_bf16_bits,
      linear_bias,
      residual_scale,
      norm_weight,
      norm_bias,
      residual_out,
      norm_out,
      mean_out,
      rstd_out,
      residual_bf16_out,
      nullptr,
      rows,
      output_dim,
      eps);
}

void launch_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_bf16_norm_float32(
    const float* residual,
    const std::uint16_t* linear_out_bf16_bits,
    const float* linear_bias,
    const float* residual_scale,
    const float* norm_weight,
    const float* norm_bias,
    float* residual_out,
    float* norm_out,
    float* mean_out,
    float* rstd_out,
    std::uint16_t* residual_bf16_out,
    std::uint16_t* norm_bf16_out,
    std::int64_t rows,
    std::int64_t output_dim,
    float eps,
    cudaStream_t stream) {
  linear_bias_residual_layer_norm_bf16_linear_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      residual,
      linear_out_bf16_bits,
      linear_bias,
      residual_scale,
      norm_weight,
      norm_bias,
      residual_out,
      norm_out,
      mean_out,
      rstd_out,
      residual_bf16_out,
      norm_bf16_out,
      rows,
      output_dim,
      eps);
}

void launch_linear_bias_residual_layer_norm_with_stats_float32(
    const float* residual,
    const float* linear_out,
    const float* linear_bias,
    const float* residual_scale,
    const float* norm_weight,
    const float* norm_bias,
    float* residual_out,
    float* norm_out,
    float* mean_out,
    float* rstd_out,
    std::int64_t rows,
    std::int64_t output_dim,
    float eps,
    cudaStream_t stream) {
  linear_bias_residual_layer_norm_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      residual,
      linear_out,
      linear_bias,
      residual_scale,
      norm_weight,
      norm_bias,
      residual_out,
      norm_out,
      mean_out,
      rstd_out,
      nullptr,
      nullptr,
      rows,
      output_dim,
      eps);
}

void launch_linear_bias_residual_layer_norm_with_stats_bf16_residual_float32(
    const float* residual,
    const float* linear_out,
    const float* linear_bias,
    const float* residual_scale,
    const float* norm_weight,
    const float* norm_bias,
    float* residual_out,
    float* norm_out,
    float* mean_out,
    float* rstd_out,
    std::uint16_t* residual_bf16_out,
    std::int64_t rows,
    std::int64_t output_dim,
    float eps,
    cudaStream_t stream) {
  linear_bias_residual_layer_norm_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      residual,
      linear_out,
      linear_bias,
      residual_scale,
      norm_weight,
      norm_bias,
      residual_out,
      norm_out,
      mean_out,
      rstd_out,
      residual_bf16_out,
      nullptr,
      rows,
      output_dim,
      eps);
}

void launch_linear_bias_residual_layer_norm_with_stats_bf16_residual_bf16_norm_float32(
    const float* residual,
    const float* linear_out,
    const float* linear_bias,
    const float* residual_scale,
    const float* norm_weight,
    const float* norm_bias,
    float* residual_out,
    float* norm_out,
    float* mean_out,
    float* rstd_out,
    std::uint16_t* residual_bf16_out,
    std::uint16_t* norm_bf16_out,
    std::int64_t rows,
    std::int64_t output_dim,
    float eps,
    cudaStream_t stream) {
  linear_bias_residual_layer_norm_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      residual,
      linear_out,
      linear_bias,
      residual_scale,
      norm_weight,
      norm_bias,
      residual_out,
      norm_out,
      mean_out,
      rstd_out,
      residual_bf16_out,
      norm_bf16_out,
      rows,
      output_dim,
      eps);
}

void launch_gelu_backward_float32(
    const float* x,
    const float* grad_out,
    float* grad_x,
    std::int64_t n,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  gelu_backward_float32_kernel<<<blocks, 1, 0, stream>>>(x, grad_out, grad_x, n);
}

void launch_gelu_backward_inplace_float32(
    const float* x,
    float* grad,
    std::int64_t n,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  gelu_backward_inplace_float32_kernel<<<blocks, 1, 0, stream>>>(x, grad, n);
}

void launch_gelu_backward_inplace_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    float* grad,
    std::int64_t n,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  gelu_backward_inplace_bf16_bits_float32_kernel<<<blocks, 1, 0, stream>>>(
      x_bf16_bits, grad, n);
}

void launch_gelu_backward_inplace_bf16_bits_to_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    std::uint16_t* grad_bf16_bits,
    std::int64_t n,
    cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  gelu_backward_inplace_bf16_bits_to_bf16_bits_float32_kernel<<<blocks, threads, 0, stream>>>(
      x_bf16_bits, grad_bf16_bits, n);
}

void launch_dropout_forward_float32(
    const float* x,
    float* out,
    std::int64_t n,
    float dropout_p,
    std::int64_t seed,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  dropout_forward_float32_kernel<<<blocks, 1, 0, stream>>>(x, out, n, dropout_p, seed);
}

void launch_dropout_backward_float32(
    const float* grad_out,
    float* grad_x,
    std::int64_t n,
    float dropout_p,
    std::int64_t seed,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  dropout_backward_float32_kernel<<<blocks, 1, 0, stream>>>(
      grad_out, grad_x, n, dropout_p, seed);
}

void launch_act_weighted_sum_float32(
    const float* states,
    const float* weights,
    float* out,
    std::int64_t batch,
    std::int64_t steps,
    std::int64_t inner,
    cudaStream_t stream) {
  const std::int64_t n = batch * inner;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  act_weighted_sum_float32_kernel<<<blocks, 1, 0, stream>>>(states, weights, out, n, steps, inner);
}

void launch_latent_pool_float32(
    const float* x,
    const float* mask_values,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t dim,
    cudaStream_t stream) {
  const std::int64_t n = batch * dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  latent_pool_float32_kernel<<<blocks, 1, 0, stream>>>(x, mask_values, out, n, seq_len, dim);
}

void launch_token_cross_entropy_partials_float32(
    const float* logits,
    const std::int64_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((rows + kTileSize - 1) / kTileSize);
  token_cross_entropy_partials_float32_kernel<<<blocks, 1, 0, stream>>>(logits, targets, partials, rows, vocab);
}

void launch_token_cross_entropy_partials_bf16_bits(
    const std::uint16_t* logits_bf16_bits,
    const std::int64_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((rows + kTileSize - 1) / kTileSize);
  token_cross_entropy_partials_bf16_bits_kernel<<<blocks, 1, 0, stream>>>(
      logits_bf16_bits, targets, partials, rows, vocab);
}

void launch_token_cross_entropy_partials_strided_float32(
    const float* logits,
    const std::int64_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    cudaStream_t stream) {
  if (row_stride == vocab) {
    launch_token_cross_entropy_partials_float32(logits, targets, partials, rows, vocab, stream);
    return;
  }
  const int blocks = static_cast<int>((rows + kTileSize - 1) / kTileSize);
  token_cross_entropy_partials_strided_float32_kernel<<<blocks, 1, 0, stream>>>(
      logits, targets, partials, rows, vocab, row_stride);
}

void launch_token_cross_entropy_partials_strided_bf16_bits(
    const std::uint16_t* logits_bf16_bits,
    const std::int64_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    cudaStream_t stream) {
  if (row_stride == vocab) {
    launch_token_cross_entropy_partials_bf16_bits(logits_bf16_bits, targets, partials, rows, vocab, stream);
    return;
  }
  const int blocks = static_cast<int>((rows + kTileSize - 1) / kTileSize);
  token_cross_entropy_partials_strided_bf16_bits_kernel<<<blocks, 1, 0, stream>>>(
      logits_bf16_bits, targets, partials, rows, vocab, row_stride);
}

void launch_token_cross_entropy_partials_strided_bf16_bits_u16_targets(
    const std::uint16_t* logits_bf16_bits,
    const std::uint16_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((rows + kTileSize - 1) / kTileSize);
  token_cross_entropy_partials_strided_bf16_bits_u16_targets_kernel<<<blocks, 1, 0, stream>>>(
      logits_bf16_bits, targets, partials, rows, vocab, row_stride);
}

void launch_masked_token_cross_entropy_partials_float32(
    const float* logits,
    const std::int64_t* targets,
    const float* loss_mask,
    float* loss_partials,
    float* mask_partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t ignore_index,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((rows + kTileSize - 1) / kTileSize);
  masked_token_cross_entropy_partials_float32_kernel<<<blocks, 1, 0, stream>>>(
      logits, targets, loss_mask, loss_partials, mask_partials, rows, vocab, ignore_index);
}

void launch_token_cross_entropy_backward_float32(
    const float* logits,
    const std::int64_t* targets,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t vocab,
    float loss_scale,
    cudaStream_t stream) {
  if (vocab <= kTileSize) {
    token_cross_entropy_backward_rowwise_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
        logits, targets, grad_logits, rows, vocab, loss_scale);
  } else {
    TokenCrossEntropyWorkspace* workspace = ensure_token_cross_entropy_workspace(rows);
    if (workspace == nullptr) {
      return;
    }
    const std::int64_t chunks_per_row = (vocab + kTileSize - 1) / kTileSize;
    token_cross_entropy_row_stats_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
        logits, workspace->row_max, workspace->row_denom, rows, vocab);
    token_cross_entropy_backward_chunked_float32_kernel<<<static_cast<int>(rows * chunks_per_row), 1, 0, stream>>>(
        logits,
        targets,
        workspace->row_max,
        workspace->row_denom,
        grad_logits,
        rows,
        vocab,
        chunks_per_row,
        loss_scale);
  }
}

void launch_token_cross_entropy_backward_with_workspace_float32(
    const float* logits,
    const std::int64_t* targets,
    float* row_max,
    float* row_denom,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t vocab,
    float loss_scale,
    cudaStream_t stream) {
  if (vocab <= kTileSize) {
    token_cross_entropy_backward_rowwise_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
        logits, targets, grad_logits, rows, vocab, loss_scale);
    return;
  }
  const std::int64_t chunks_per_row = (vocab + kTileSize - 1) / kTileSize;
  token_cross_entropy_row_stats_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      logits, row_max, row_denom, rows, vocab);
  token_cross_entropy_backward_chunked_float32_kernel<<<static_cast<int>(rows * chunks_per_row), 1, 0, stream>>>(
      logits, targets, row_max, row_denom, grad_logits, rows, vocab, chunks_per_row, loss_scale);
}

void launch_token_cross_entropy_backward_inplace_with_workspace_float32(
    float* logits,
    const std::int64_t* targets,
    float* row_max,
    float* row_denom,
    std::int64_t rows,
    std::int64_t vocab,
    float loss_scale,
    cudaStream_t stream) {
  if (vocab <= kTileSize) {
    token_cross_entropy_backward_rowwise_inplace_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
        logits, targets, rows, vocab, loss_scale);
    return;
  }
  const std::int64_t chunks_per_row = (vocab + kTileSize - 1) / kTileSize;
  token_cross_entropy_row_stats_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      logits, row_max, row_denom, rows, vocab);
  token_cross_entropy_backward_chunked_inplace_float32_kernel<<<static_cast<int>(rows * chunks_per_row), 1, 0, stream>>>(
      logits, targets, row_max, row_denom, rows, vocab, chunks_per_row, loss_scale);
}

void launch_token_cross_entropy_backward_inplace_bf16_bits_with_workspace(
    std::uint16_t* logits,
    const std::int64_t* targets,
    float* row_max,
    float* row_denom,
    std::int64_t rows,
    std::int64_t vocab,
    float loss_scale,
    cudaStream_t stream) {
  (void)row_max;
  (void)row_denom;
  const int threads = cross_entropy_bf16_threads_per_row();
  const bool vec_stores = cross_entropy_bf16_vec_stores_enabled();
  const bool vec_normal_stores = cross_entropy_bf16_vec_normal_stores_enabled();
  const bool vec_loads = cross_entropy_bf16_vec_loads_enabled();
  const bool scalar_streaming_stores = cross_entropy_bf16_scalar_streaming_stores_enabled();
  const bool use_exp2 = cross_entropy_bf16_exp2_enabled();
  token_cross_entropy_backward_inplace_bf16_bits_fused_kernel<<<static_cast<int>(rows), threads, 0, stream>>>(
      logits, targets, rows, vocab, loss_scale, vec_stores, vec_normal_stores, vec_loads, scalar_streaming_stores, use_exp2);
}

void launch_token_cross_entropy_backward_inplace_strided_with_workspace_float32(
    float* logits,
    const std::int64_t* targets,
    float* row_max,
    float* row_denom,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream) {
  if (row_stride == vocab) {
    launch_token_cross_entropy_backward_inplace_with_workspace_float32(
        logits, targets, row_max, row_denom, rows, vocab, loss_scale, stream);
    return;
  }
  (void)row_max;
  (void)row_denom;
  constexpr int threads = 1024;
  token_cross_entropy_backward_inplace_strided_float32_fused_kernel<<<static_cast<int>(rows), threads, 0, stream>>>(
      logits, targets, rows, vocab, row_stride, loss_scale, true);
}

void launch_token_cross_entropy_backward_inplace_strided_no_pad_zero_with_workspace_float32(
    float* logits,
    const std::int64_t* targets,
    float* row_max,
    float* row_denom,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream) {
  if (row_stride == vocab) {
    launch_token_cross_entropy_backward_inplace_with_workspace_float32(
        logits, targets, row_max, row_denom, rows, vocab, loss_scale, stream);
    return;
  }
  (void)row_max;
  (void)row_denom;
  constexpr int threads = 1024;
  token_cross_entropy_backward_inplace_strided_float32_fused_kernel<<<static_cast<int>(rows), threads, 0, stream>>>(
      logits, targets, rows, vocab, row_stride, loss_scale, false);
}

void launch_token_cross_entropy_backward_inplace_strided_bf16_bits_with_workspace(
    std::uint16_t* logits,
    const std::int64_t* targets,
    float* row_max,
    float* row_denom,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream) {
  if (row_stride == vocab) {
    launch_token_cross_entropy_backward_inplace_bf16_bits_with_workspace(
        logits, targets, row_max, row_denom, rows, vocab, loss_scale, stream);
    return;
  }
  (void)row_max;
  (void)row_denom;
  const int threads = cross_entropy_bf16_threads_per_row();
  const bool vec_stores = cross_entropy_bf16_vec_stores_enabled();
  const bool vec_normal_stores = cross_entropy_bf16_vec_normal_stores_enabled();
  const bool vec_loads = cross_entropy_bf16_vec_loads_enabled();
  const bool scalar_streaming_stores = cross_entropy_bf16_scalar_streaming_stores_enabled();
  const bool use_exp2 = cross_entropy_bf16_exp2_enabled();
  token_cross_entropy_backward_inplace_strided_bf16_bits_fused_kernel<<<static_cast<int>(rows), threads, 0, stream>>>(
      logits, targets, rows, vocab, row_stride, loss_scale, vec_stores, vec_normal_stores, vec_loads, scalar_streaming_stores, use_exp2, true);
}

void launch_token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_with_workspace(
    std::uint16_t* logits,
    const std::int64_t* targets,
    float* row_max,
    float* row_denom,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream) {
  if (row_stride == vocab) {
    launch_token_cross_entropy_backward_inplace_bf16_bits_with_workspace(
        logits, targets, row_max, row_denom, rows, vocab, loss_scale, stream);
    return;
  }
  (void)row_max;
  (void)row_denom;
  const int threads = cross_entropy_bf16_threads_per_row();
  const bool vec_stores = cross_entropy_bf16_vec_stores_enabled();
  const bool vec_normal_stores = cross_entropy_bf16_vec_normal_stores_enabled();
  const bool vec_loads = cross_entropy_bf16_vec_loads_enabled();
  const bool scalar_streaming_stores = cross_entropy_bf16_scalar_streaming_stores_enabled();
  const bool use_exp2 = cross_entropy_bf16_exp2_enabled();
  token_cross_entropy_backward_inplace_strided_bf16_bits_fused_kernel<<<static_cast<int>(rows), threads, 0, stream>>>(
      logits, targets, rows, vocab, row_stride, loss_scale, vec_stores, vec_normal_stores, vec_loads, scalar_streaming_stores, use_exp2, false);
}

void launch_token_cross_entropy_backward_inplace_strided_bf16_bits_u16_targets_with_workspace(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* row_max,
    float* row_denom,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream) {
  (void)row_max;
  (void)row_denom;
  const int threads = cross_entropy_bf16_threads_per_row();
  const bool vec_stores = cross_entropy_bf16_vec_stores_enabled();
  const bool vec_normal_stores = cross_entropy_bf16_vec_normal_stores_enabled();
  const bool vec_loads = cross_entropy_bf16_vec_loads_enabled();
  const bool scalar_streaming_stores = cross_entropy_bf16_scalar_streaming_stores_enabled();
  const bool use_exp2 = cross_entropy_bf16_exp2_enabled();
  token_cross_entropy_backward_inplace_strided_bf16_bits_u16_targets_fused_kernel<<<static_cast<int>(rows), threads, 0, stream>>>(
      logits, targets, rows, vocab, row_stride, loss_scale, vec_stores, vec_normal_stores, vec_loads, scalar_streaming_stores, use_exp2, true);
}

void launch_token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* row_max,
    float* row_denom,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream) {
  (void)row_max;
  (void)row_denom;
  const int threads = cross_entropy_bf16_threads_per_row();
  const bool vec_stores = cross_entropy_bf16_vec_stores_enabled();
  const bool vec_normal_stores = cross_entropy_bf16_vec_normal_stores_enabled();
  const bool vec_loads = cross_entropy_bf16_vec_loads_enabled();
  const bool scalar_streaming_stores = cross_entropy_bf16_scalar_streaming_stores_enabled();
  const bool use_exp2 = cross_entropy_bf16_exp2_enabled();
  if (lm_head_ce_no_loss_llmk_style_specialized_enabled() &&
      threads == 1024 &&
      vec_loads &&
      !use_exp2) {
    token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_llmk_style_kernel<<<static_cast<int>(rows), threads, 0, stream>>>(
        logits, targets, rows, vocab, row_stride, loss_scale, lm_head_ce_reverse_rows_enabled());
    return;
  }
  if (lm_head_ce_no_loss_default_specialized_enabled() &&
      threads == 1024 &&
      vec_loads &&
      !vec_stores &&
      !vec_normal_stores &&
      !scalar_streaming_stores &&
      !use_exp2) {
    token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_default_kernel<<<static_cast<int>(rows), threads, 0, stream>>>(
        logits, targets, rows, vocab, row_stride, loss_scale, lm_head_ce_reverse_rows_enabled());
    return;
  }
  token_cross_entropy_backward_inplace_strided_bf16_bits_u16_targets_fused_kernel<<<static_cast<int>(rows), threads, 0, stream>>>(
      logits, targets, rows, vocab, row_stride, loss_scale, vec_stores, vec_normal_stores, vec_loads, scalar_streaming_stores, use_exp2, false);
}

void launch_lm_head_classifier_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* row_max,
    float* row_denom,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream) {
  g_lm_head_classifier_chunk_launch_count.fetch_add(1, std::memory_order_relaxed);
  g_lm_head_classifier_last_rows.store(rows, std::memory_order_relaxed);
  g_lm_head_classifier_last_vocab.store(vocab, std::memory_order_relaxed);
  g_lm_head_classifier_last_row_stride.store(row_stride, std::memory_order_relaxed);
  launch_token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace(
      logits, targets, row_max, row_denom, rows, vocab, row_stride, loss_scale, stream);
}

void launch_lm_head_classifier_backward_prob_only_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream) {
  g_lm_head_classifier_chunk_launch_count.fetch_add(1, std::memory_order_relaxed);
  g_lm_head_classifier_last_rows.store(rows, std::memory_order_relaxed);
  g_lm_head_classifier_last_vocab.store(vocab, std::memory_order_relaxed);
  g_lm_head_classifier_last_row_stride.store(row_stride, std::memory_order_relaxed);
  const int threads = cross_entropy_bf16_threads_per_row();
  token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_prob_only_kernel<<<static_cast<int>(rows), threads, 0, stream>>>(
      logits, targets, rows, vocab, row_stride, loss_scale, lm_head_ce_reverse_rows_enabled());
}

void launch_lm_head_prob_only_dhidden_target_correction_bf16_bits(
    const std::uint16_t* targets,
    const std::uint16_t* token_weight_bf16,
    float* grad_hidden,
    std::int64_t rows,
    std::int64_t hidden_dim,
    std::int64_t token_weight_row_stride,
    float loss_scale,
    cudaStream_t stream) {
  const int threads = lm_head_prob_only_target_correction_threads_value();
  const std::int64_t total = rows * hidden_dim;
  if (total <= 0) {
    return;
  }
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  lm_head_prob_only_dhidden_target_correction_bf16_bits_kernel<<<blocks, threads, 0, stream>>>(
      targets, token_weight_bf16, grad_hidden, rows, hidden_dim, token_weight_row_stride, loss_scale);
}

void launch_lm_head_prob_only_dweight_target_correction_bf16_bits(
    const std::uint16_t* targets,
    const std::uint16_t* hidden_bf16,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t hidden_dim,
    std::int64_t grad_weight_row_stride,
    float loss_scale,
    cudaStream_t stream) {
  const int threads = lm_head_prob_only_target_correction_threads_value();
  const std::int64_t total = rows * hidden_dim;
  if (total <= 0) {
    return;
  }
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  lm_head_prob_only_dweight_target_correction_bf16_bits_kernel<<<blocks, threads, 0, stream>>>(
      targets, hidden_bf16, grad_weight, rows, hidden_dim, grad_weight_row_stride, loss_scale);
}

void launch_lm_head_prob_only_combined_target_correction_bf16_bits(
    const std::uint16_t* targets,
    const std::uint16_t* token_weight_bf16,
    const std::uint16_t* hidden_bf16,
    float* grad_hidden,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t hidden_dim,
    std::int64_t token_weight_row_stride,
    std::int64_t grad_weight_row_stride,
    float loss_scale,
    cudaStream_t stream) {
  const int threads = lm_head_prob_only_target_correction_threads_value();
  const std::int64_t total = rows * hidden_dim;
  if (total <= 0) {
    return;
  }
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  lm_head_prob_only_combined_target_correction_bf16_bits_kernel<<<blocks, threads, 0, stream>>>(
      targets,
      token_weight_bf16,
      hidden_bf16,
      grad_hidden,
      grad_weight,
      rows,
      hidden_dim,
      token_weight_row_stride,
      grad_weight_row_stride,
      loss_scale);
}

void launch_token_cross_entropy_backward_loss_inplace_strided_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* loss_total,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream) {
  const int threads = cross_entropy_bf16_threads_per_row();
  const bool vec_stores = cross_entropy_bf16_vec_stores_enabled();
  const bool vec_normal_stores = cross_entropy_bf16_vec_normal_stores_enabled();
  const bool vec_loads = cross_entropy_bf16_vec_loads_enabled();
  const bool scalar_streaming_stores = cross_entropy_bf16_scalar_streaming_stores_enabled();
  const bool use_exp2 = cross_entropy_bf16_exp2_enabled();
  token_cross_entropy_backward_loss_inplace_strided_bf16_bits_u16_targets_fused_kernel<<<static_cast<int>(rows), threads, 0, stream>>>(
      logits, targets, loss_total, rows, vocab, row_stride, loss_scale, vec_stores, vec_normal_stores, vec_loads, scalar_streaming_stores, use_exp2, true, false);
}

void launch_token_cross_entropy_backward_loss_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* loss_total,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream) {
  const int threads = cross_entropy_bf16_threads_per_row();
  const bool vec_stores = cross_entropy_bf16_vec_stores_enabled();
  const bool vec_normal_stores = cross_entropy_bf16_vec_normal_stores_enabled();
  const bool vec_loads = cross_entropy_bf16_vec_loads_enabled();
  const bool scalar_streaming_stores = cross_entropy_bf16_scalar_streaming_stores_enabled();
  const bool use_exp2 = cross_entropy_bf16_exp2_enabled();
  token_cross_entropy_backward_loss_inplace_strided_bf16_bits_u16_targets_fused_kernel<<<static_cast<int>(rows), threads, 0, stream>>>(
      logits, targets, loss_total, rows, vocab, row_stride, loss_scale, vec_stores, vec_normal_stores, vec_loads, scalar_streaming_stores, use_exp2, false, false);
}

void launch_lm_head_classifier_backward_loss_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* loss_total,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream) {
  g_lm_head_classifier_chunk_launch_count.fetch_add(1, std::memory_order_relaxed);
  g_lm_head_classifier_last_rows.store(rows, std::memory_order_relaxed);
  g_lm_head_classifier_last_vocab.store(vocab, std::memory_order_relaxed);
  g_lm_head_classifier_last_row_stride.store(row_stride, std::memory_order_relaxed);
  launch_token_cross_entropy_backward_loss_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
      logits, targets, loss_total, rows, vocab, row_stride, loss_scale, stream);
}

void launch_lm_head_classifier_backward_row_losses_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* row_losses,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream) {
  g_lm_head_classifier_chunk_launch_count.fetch_add(1, std::memory_order_relaxed);
  g_lm_head_classifier_last_rows.store(rows, std::memory_order_relaxed);
  g_lm_head_classifier_last_vocab.store(vocab, std::memory_order_relaxed);
  g_lm_head_classifier_last_row_stride.store(row_stride, std::memory_order_relaxed);
  const int threads = cross_entropy_bf16_threads_per_row();
  const bool vec_stores = cross_entropy_bf16_vec_stores_enabled();
  const bool vec_normal_stores = cross_entropy_bf16_vec_normal_stores_enabled();
  const bool vec_loads = cross_entropy_bf16_vec_loads_enabled();
  const bool scalar_streaming_stores = cross_entropy_bf16_scalar_streaming_stores_enabled();
  const bool use_exp2 = cross_entropy_bf16_exp2_enabled();
  if (lm_head_ce_llmk_style_specialized_enabled() &&
      threads == 1024 &&
      vec_loads &&
      !use_exp2) {
    lm_head_classifier_backward_row_losses_llmk_style_bf16_bits_u16_targets_kernel<<<static_cast<int>(rows), threads, 0, stream>>>(
        logits, targets, row_losses, rows, vocab, row_stride, loss_scale);
    return;
  }
  if (lm_head_ce_default_specialized_enabled() &&
      threads == 1024 &&
      vec_loads &&
      !vec_stores &&
      !vec_normal_stores &&
      !scalar_streaming_stores &&
      !use_exp2) {
    lm_head_classifier_backward_row_losses_default_bf16_bits_u16_targets_kernel<<<static_cast<int>(rows), threads, 0, stream>>>(
        logits, targets, row_losses, rows, vocab, row_stride, loss_scale);
    return;
  }
  token_cross_entropy_backward_loss_inplace_strided_bf16_bits_u16_targets_fused_kernel<<<static_cast<int>(rows), threads, 0, stream>>>(
      logits, targets, row_losses, rows, vocab, row_stride, loss_scale, vec_stores, vec_normal_stores, vec_loads, scalar_streaming_stores, use_exp2, false, true);
}

void launch_lm_head_classifier_backward_loss_bins_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* loss_bins,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    std::int64_t loss_bin_count,
    float loss_scale,
    cudaStream_t stream) {
  g_lm_head_classifier_chunk_launch_count.fetch_add(1, std::memory_order_relaxed);
  g_lm_head_classifier_loss_bin_launch_count.fetch_add(1, std::memory_order_relaxed);
  g_lm_head_classifier_last_rows.store(rows, std::memory_order_relaxed);
  g_lm_head_classifier_last_vocab.store(vocab, std::memory_order_relaxed);
  g_lm_head_classifier_last_row_stride.store(row_stride, std::memory_order_relaxed);
  const int threads = cross_entropy_bf16_threads_per_row();
  const bool vec_stores = cross_entropy_bf16_vec_stores_enabled();
  const bool vec_normal_stores = cross_entropy_bf16_vec_normal_stores_enabled();
  const bool vec_loads = cross_entropy_bf16_vec_loads_enabled();
  const bool scalar_streaming_stores = cross_entropy_bf16_scalar_streaming_stores_enabled();
  const bool use_exp2 = cross_entropy_bf16_exp2_enabled();
  if (lm_head_ce_llmk_style_specialized_enabled() &&
      threads == 1024 &&
      vec_loads &&
      !use_exp2) {
    lm_head_classifier_backward_loss_bins_llmk_style_bf16_bits_u16_targets_kernel<<<static_cast<int>(rows), threads, 0, stream>>>(
        logits, targets, loss_bins, rows, vocab, row_stride, loss_bin_count, loss_scale);
    return;
  }
  if (lm_head_ce_loss_bins_default_specialized_enabled() &&
      threads == 1024 &&
      vec_loads &&
      !vec_stores &&
      !vec_normal_stores &&
      !scalar_streaming_stores &&
      !use_exp2) {
    lm_head_classifier_backward_loss_bins_default_bf16_bits_u16_targets_kernel<<<static_cast<int>(rows), threads, 0, stream>>>(
        logits, targets, loss_bins, rows, vocab, row_stride, loss_bin_count, loss_scale);
    return;
  }
  token_cross_entropy_backward_loss_bins_inplace_strided_bf16_bits_u16_targets_fused_kernel<<<static_cast<int>(rows), threads, 0, stream>>>(
      logits, targets, loss_bins, rows, vocab, row_stride, loss_bin_count, loss_scale, vec_stores, vec_normal_stores, vec_loads, scalar_streaming_stores, use_exp2);
}

void launch_masked_token_cross_entropy_backward_float32(
    const float* logits,
    const std::int64_t* targets,
    const float* loss_mask,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t ignore_index,
    float loss_scale,
    cudaStream_t stream) {
  if (vocab <= kTileSize) {
    masked_token_cross_entropy_backward_rowwise_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
        logits, targets, loss_mask, grad_logits, rows, vocab, ignore_index, loss_scale);
  } else {
    TokenCrossEntropyWorkspace* workspace = ensure_token_cross_entropy_workspace(rows);
    if (workspace == nullptr) {
      return;
    }
    const std::int64_t chunks_per_row = (vocab + kTileSize - 1) / kTileSize;
    masked_token_cross_entropy_row_stats_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
        logits, targets, workspace->row_max, workspace->row_denom, rows, vocab, ignore_index);
    masked_token_cross_entropy_backward_chunked_float32_kernel<<<static_cast<int>(rows * chunks_per_row), 1, 0, stream>>>(
        logits,
        targets,
        loss_mask,
        workspace->row_max,
        workspace->row_denom,
        grad_logits,
        rows,
        vocab,
        ignore_index,
        chunks_per_row,
        loss_scale);
  }
}

void launch_masked_token_cross_entropy_backward_with_workspace_float32(
    const float* logits,
    const std::int64_t* targets,
    const float* loss_mask,
    float* row_max,
    float* row_denom,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t ignore_index,
    float loss_scale,
    cudaStream_t stream) {
  if (vocab <= kTileSize) {
    masked_token_cross_entropy_backward_rowwise_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
        logits, targets, loss_mask, grad_logits, rows, vocab, ignore_index, loss_scale);
    return;
  }
  const std::int64_t chunks_per_row = (vocab + kTileSize - 1) / kTileSize;
  masked_token_cross_entropy_row_stats_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      logits, targets, row_max, row_denom, rows, vocab, ignore_index);
  masked_token_cross_entropy_backward_chunked_float32_kernel<<<static_cast<int>(rows * chunks_per_row), 1, 0, stream>>>(
      logits, targets, loss_mask, row_max, row_denom, grad_logits, rows, vocab, ignore_index, chunks_per_row, loss_scale);
}

void launch_sequence_logp_float32(
    const float* logits,
    const std::int64_t* targets,
    const float* loss_mask,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t vocab,
    std::int64_t ignore_index,
    cudaStream_t stream) {
  sequence_logp_float32_kernel<<<1, 1, 0, stream>>>(logits, targets, loss_mask, out, batch, seq_len, vocab, ignore_index);
}

void launch_preference_bce_partials_float32(
    const float* reward_chosen,
    const float* reward_rejected,
    float* partials,
    std::int64_t n,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  preference_bce_partials_float32_kernel<<<blocks, 1, 0, stream>>>(reward_chosen, reward_rejected, partials, n);
}

void launch_ppo_clipped_loss_partials_float32(
    const float* logp_new,
    const float* logp_old,
    const float* advantages,
    const float* value_new,
    const float* value_old,
    const float* returns,
    float* policy_partials,
    float* value_partials,
    std::int64_t n,
    float clip_range,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  ppo_clipped_loss_partials_float32_kernel<<<blocks, 1, 0, stream>>>(
      logp_new, logp_old, advantages, value_new, value_old, returns, policy_partials, value_partials, n, clip_range);
}

void launch_gae_compute_float32(
    const float* rewards,
    const float* values,
    float* advantages,
    float* returns,
    std::int64_t batch,
    std::int64_t seq_len,
    float gamma,
    float lambda_value,
    cudaStream_t stream) {
  gae_compute_float32_kernel<<<static_cast<int>(batch), 1, 0, stream>>>(
      rewards, values, advantages, returns, seq_len, gamma, lambda_value);
}

void launch_dpo_pairwise_partials_float32(
    const float* policy_logp_chosen,
    const float* policy_logp_rejected,
    const float* ref_logp_chosen,
    const float* ref_logp_rejected,
    float* partials,
    float* chosen_reward_out,
    float* rejected_reward_out,
    std::int64_t n,
    float beta,
    float label_smoothing,
    int loss_type,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  dpo_pairwise_partials_float32_kernel<<<blocks, 1, 0, stream>>>(
      policy_logp_chosen,
      policy_logp_rejected,
      ref_logp_chosen,
      ref_logp_rejected,
      partials,
      chosen_reward_out,
      rejected_reward_out,
      n,
      beta,
      label_smoothing,
      loss_type);
}

void launch_route_selection_loss_partials_float32(
    const float* route_logits,
    const std::int64_t* sem_targets,
    float* loss_partials,
    float* count_partials,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t experts,
    std::int64_t num_vocab_dims,
    std::int64_t shared_experts,
    std::int64_t ignore_index,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  route_selection_loss_partials_float32_kernel<<<blocks, 1, 0, stream>>>(
      route_logits,
      sem_targets,
      loss_partials,
      count_partials,
      n,
      seq_len,
      experts,
      num_vocab_dims,
      shared_experts,
      ignore_index);
}

void launch_route_balance_density_float32(
    const float* route_logits,
    float* density,
    std::int64_t rows,
    std::int64_t experts,
    cudaStream_t stream) {
  route_balance_density_float32_kernel<<<static_cast<int>(experts), 1, 0, stream>>>(route_logits, density, rows, experts);
}

void launch_route_balance_loss_float32(
    const float* density,
    float* out,
    std::int64_t experts,
    cudaStream_t stream) {
  route_balance_loss_float32_kernel<<<1, 1, 0, stream>>>(density, out, experts);
}

void launch_softmax_distillation_partials_float32(
    const float* teacher_logits,
    const float* student_logits,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    cudaStream_t stream) {
  softmax_distillation_partials_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      teacher_logits, student_logits, partials, rows, vocab);
}

void launch_scaled_dot_product_attention_float32(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    std::int64_t n,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (use_tk_sm120_attention(
          query_heads,
          key_heads,
          seq_q,
          seq_k,
          qk_dim,
          value_dim,
          is_causal,
          right_align_causal,
          use_sparse_rules,
          window,
          num_sinks,
          block_size,
          compress_stride)) {
    const std::int64_t per_batch_n = query_heads * seq_q * value_dim;
    const std::int64_t batch = per_batch_n > 0 ? n / per_batch_n : 0;
    const std::int64_t expected_n = batch * per_batch_n;
    if (batch > 0 && n == expected_n &&
        launch_tk_attention_forward_float32(
            q, k, v, out, batch, query_heads, seq_q, qk_dim, stream) == 0) {
      return;
    }
  }
#endif
  if (!g_attention_forward_row_launch_disabled.load(std::memory_order_relaxed) &&
      seq_k <= kTileSize && seq_k > 0 && qk_dim > 0 && value_dim > 0 &&
      query_heads == kGpt2AttentionHeads && key_heads == kGpt2AttentionHeads &&
      seq_q == seq_k && qk_dim == kGpt2AttentionHeadDim && value_dim == kGpt2AttentionHeadDim &&
      is_causal &&
      !right_align_causal && !use_sparse_rules && window <= 0 && num_sinks <= 0 &&
      block_size <= 0 && compress_stride <= 1) {
    const std::int64_t row_count = n / value_dim;
    cudaFuncAttributes row_attrs{};
    const cudaError_t attr_status = cudaFuncGetAttributes(&row_attrs, scaled_dot_product_attention_row_float32_kernel);
    g_attention_forward_row_attr_status.store(static_cast<int>(attr_status), std::memory_order_relaxed);
    if (attr_status == cudaSuccess) {
      g_attention_forward_row_attr_max_threads_per_block.store(row_attrs.maxThreadsPerBlock, std::memory_order_relaxed);
      g_attention_forward_row_attr_num_regs.store(row_attrs.numRegs, std::memory_order_relaxed);
      g_attention_forward_row_attr_shared_size_bytes.store(row_attrs.sharedSizeBytes, std::memory_order_relaxed);
      g_attention_forward_row_attr_const_size_bytes.store(row_attrs.constSizeBytes, std::memory_order_relaxed);
      g_attention_forward_row_attr_local_size_bytes.store(row_attrs.localSizeBytes, std::memory_order_relaxed);
    }
    g_attention_forward_row_launch_count.fetch_add(1, std::memory_order_relaxed);
    const dim3 row_grid(
        static_cast<unsigned int>(row_count),
        static_cast<unsigned int>(kGpt2AttentionValueChunks),
        1);
    constexpr unsigned int row_block = 1;
    g_attention_forward_row_grid_x.store(static_cast<std::int64_t>(row_grid.x), std::memory_order_relaxed);
    g_attention_forward_row_grid_y.store(static_cast<std::int64_t>(row_grid.y), std::memory_order_relaxed);
    g_attention_forward_row_grid_z.store(static_cast<std::int64_t>(row_grid.z), std::memory_order_relaxed);
    g_attention_forward_row_block_x.store(static_cast<std::int64_t>(row_block), std::memory_order_relaxed);
    const cudaError_t clear_error = cudaGetLastError();
    g_attention_forward_row_prelaunch_clear_error.store(static_cast<int>(clear_error), std::memory_order_relaxed);
    const cudaError_t prelaunch_error = cudaPeekAtLastError();
    g_attention_forward_row_prelaunch_peek_error.store(static_cast<int>(prelaunch_error), std::memory_order_relaxed);
    scaled_dot_product_attention_row_float32_kernel<<<
        row_grid, row_block, 0, stream>>>(
        q,
        k,
        v,
        out,
        seq_k,
        scale);
    if (cudaPeekAtLastError() == cudaSuccess) {
      return;
    }
    cudaError_t row_error = cudaGetLastError();
    g_attention_forward_row_last_error.store(static_cast<int>(row_error), std::memory_order_relaxed);
    g_attention_forward_row_fallback_count.fetch_add(1, std::memory_order_relaxed);
    g_attention_forward_row_launch_disabled.store(true, std::memory_order_relaxed);
  }
  g_attention_forward_scalar_launch_count.fetch_add(1, std::memory_order_relaxed);
  scaled_dot_product_attention_float32_kernel<<<static_cast<int>(n), 1, 0, stream>>>(
      q,
      k,
      v,
      out,
      n,
      query_heads,
      key_heads,
      seq_q,
      seq_k,
      qk_dim,
      value_dim,
      scale,
      is_causal,
      right_align_causal,
      use_sparse_rules,
      window,
      num_sinks,
      block_size,
      compress_stride);
}

void reset_attention_forward_launch_stats() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  g_attention_forward_tk_launch_count.store(0, std::memory_order_relaxed);
  g_attention_backward_tk_launch_count.store(0, std::memory_order_relaxed);
  g_attention_backward_float_hd64_dprep_launch_count.store(0, std::memory_order_relaxed);
  g_attention_backward_dprep_timing_us.store(0, std::memory_order_relaxed);
  g_attention_backward_dprep_timing_count.store(0, std::memory_order_relaxed);
  g_attention_backward_tk_timing_us.store(0, std::memory_order_relaxed);
  g_attention_backward_tk_timing_count.store(0, std::memory_order_relaxed);
  g_attention_tk_workspace_allocation_count.store(0, std::memory_order_relaxed);
#endif
  g_attention_forward_row_launch_count.store(0, std::memory_order_relaxed);
  g_attention_forward_row_fallback_count.store(0, std::memory_order_relaxed);
  g_attention_forward_scalar_launch_count.store(0, std::memory_order_relaxed);
  g_attention_forward_row_last_error.store(0, std::memory_order_relaxed);
  g_attention_forward_row_prelaunch_clear_error.store(0, std::memory_order_relaxed);
  g_attention_forward_row_prelaunch_peek_error.store(0, std::memory_order_relaxed);
  g_attention_forward_row_grid_x.store(0, std::memory_order_relaxed);
  g_attention_forward_row_grid_y.store(0, std::memory_order_relaxed);
  g_attention_forward_row_grid_z.store(0, std::memory_order_relaxed);
  g_attention_forward_row_block_x.store(0, std::memory_order_relaxed);
  g_attention_forward_row_attr_status.store(-1, std::memory_order_relaxed);
  g_attention_forward_row_attr_max_threads_per_block.store(0, std::memory_order_relaxed);
  g_attention_forward_row_attr_num_regs.store(0, std::memory_order_relaxed);
  g_attention_forward_row_attr_shared_size_bytes.store(0, std::memory_order_relaxed);
  g_attention_forward_row_attr_const_size_bytes.store(0, std::memory_order_relaxed);
  g_attention_forward_row_attr_local_size_bytes.store(0, std::memory_order_relaxed);
  g_attention_forward_row_launch_disabled.store(false, std::memory_order_relaxed);
}

void reset_trainer_linear_launch_stats() {
  g_bf16_to_f32_vec4_count.store(0, std::memory_order_relaxed);
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  g_linear_bf16_gemm_count.store(0, std::memory_order_relaxed);
  g_linear_bf16_gemm_fast16bf_request_count.store(0, std::memory_order_relaxed);
  g_linear_tk_gemm_count.store(0, std::memory_order_relaxed);
  g_linear_tk_float_out_gemm_count.store(0, std::memory_order_relaxed);
  g_linear_tk_dweight_gemm_count.store(0, std::memory_order_relaxed);
  g_linear_tk_dgelu_dinput_gemm_count.store(0, std::memory_order_relaxed);
  g_linear_cublaslt_gemm_count.store(0, std::memory_order_relaxed);
  g_linear_cublaslt_bgrad_gemm_count.store(0, std::memory_order_relaxed);
  g_linear_cublaslt_bgrad_direct_write_count.store(0, std::memory_order_relaxed);
  g_linear_cublaslt_bgrad_accumulate_count.store(0, std::memory_order_relaxed);
  g_linear_sgemm_count.store(0, std::memory_order_relaxed);
  g_linear_bf16_a_pack_count.store(0, std::memory_order_relaxed);
  g_linear_bf16_a_cache_hit_count.store(0, std::memory_order_relaxed);
  g_linear_bf16_cache_reset_count.store(0, std::memory_order_relaxed);
  g_linear_bf16_workspace_allocation_count.store(0, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(g_trainer_linear_bf16_workspace_mutex);
    invalidate_trainer_linear_bf16_cache(g_trainer_linear_bf16_workspace);
  }
  {
    std::lock_guard<std::mutex> lock(g_linear_shape_stats_mutex);
    g_linear_shape_stats.clear();
  }
#endif
}

std::int64_t trainer_bf16_to_f32_vec4_count() {
  return g_bf16_to_f32_vec4_count.load(std::memory_order_relaxed);
}

void reset_trainer_linear_bf16_cache() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  std::lock_guard<std::mutex> lock(g_trainer_linear_bf16_workspace_mutex);
  invalidate_trainer_linear_bf16_cache(g_trainer_linear_bf16_workspace);
  g_linear_bf16_cache_reset_count.fetch_add(1, std::memory_order_relaxed);
#endif
}

bool trainer_linear_cublas_prewarm(cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return trainer_linear_cublas_prewarm_internal(stream);
#else
  (void)stream;
  return false;
#endif
}

bool trainer_linear_bf16_workspace_prewarm(
    std::int64_t a_elements,
    std::int64_t b_elements,
    std::int64_t c_elements) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return prewarm_trainer_linear_bf16_workspace(a_elements, b_elements, c_elements);
#else
  (void)a_elements;
  (void)b_elements;
  (void)c_elements;
  return false;
#endif
}

std::int64_t trainer_linear_bf16_gemm_count() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return g_linear_bf16_gemm_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t trainer_linear_bf16_gemm_fast16bf_request_count() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return g_linear_bf16_gemm_fast16bf_request_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t trainer_linear_tk_gemm_count() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return g_linear_tk_gemm_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t trainer_linear_tk_float_out_gemm_count() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return g_linear_tk_float_out_gemm_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t trainer_linear_tk_dweight_gemm_count() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return g_linear_tk_dweight_gemm_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t trainer_linear_tk_dgelu_dinput_gemm_count() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return g_linear_tk_dgelu_dinput_gemm_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

int trainer_linear_tk_sm120_k_tile() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION) && defined(LLMK_SM120_K_TILE)
  return LLMK_SM120_K_TILE;
#elif defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return 32;
#else
  return 0;
#endif
}

int trainer_linear_tk_sm120_grad_k_tile() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION) && defined(LLMK_SM120_GRAD_K_TILE)
  return LLMK_SM120_GRAD_K_TILE;
#elif defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return 64;
#else
  return 0;
#endif
}

int trainer_linear_tk_sm120_super_m() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION) && defined(LLMK_SM120_SUPER_M)
  return LLMK_SM120_SUPER_M;
#elif defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return 8;
#else
  return 0;
#endif
}

int trainer_linear_tk_sm120_dinput_super_m() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION) && defined(LLMK_SM120_DINP_SUPER_M)
  return LLMK_SM120_DINP_SUPER_M;
#elif defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return trainer_linear_tk_sm120_super_m();
#else
  return 0;
#endif
}

int trainer_linear_tk_sm120_dweight_super_m() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION) && defined(LLMK_SM120_DWEIGHT_SUPER_M)
  return LLMK_SM120_DWEIGHT_SUPER_M;
#elif defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return 2;
#else
  return 0;
#endif
}

int trainer_linear_tk_sm120_huge_n_k_tile() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION) && defined(LLMK_SM120_HUGE_N_K_TILE)
  return LLMK_SM120_HUGE_N_K_TILE;
#elif defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return 64;
#else
  return 0;
#endif
}

int trainer_linear_tk_sm120_fast_dgelu_enabled() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION) && defined(LLMK_SM120_FAST_DGELU)
  return LLMK_SM120_FAST_DGELU != 0 ? 1 : 0;
#elif defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return 1;
#else
  return 0;
#endif
}

int trainer_linear_tk_sm120_approx_dgelu_tanh_enabled() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION) && defined(LLMK_SM120_APPROX_DGELU_TANH)
  return LLMK_SM120_APPROX_DGELU_TANH != 0 ? 1 : 0;
#elif defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return 1;
#else
  return 0;
#endif
}

std::int64_t trainer_linear_sgemm_count() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return g_linear_sgemm_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t trainer_linear_cublaslt_gemm_count() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return g_linear_cublaslt_gemm_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t trainer_linear_cublaslt_bgrad_gemm_count() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return g_linear_cublaslt_bgrad_gemm_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t trainer_linear_cublaslt_bgrad_direct_write_count() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return g_linear_cublaslt_bgrad_direct_write_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t trainer_linear_cublaslt_bgrad_accumulate_count() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return g_linear_cublaslt_bgrad_accumulate_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t trainer_linear_bf16_a_pack_count() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return g_linear_bf16_a_pack_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t trainer_linear_bf16_a_cache_hit_count() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return g_linear_bf16_a_cache_hit_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t trainer_linear_bf16_cache_reset_count() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return g_linear_bf16_cache_reset_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t trainer_linear_bf16_workspace_allocation_count() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return g_linear_bf16_workspace_allocation_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t trainer_linear_bf16_workspace_a_capacity() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return g_linear_bf16_workspace_a_capacity.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t trainer_linear_bf16_workspace_b_capacity() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return g_linear_bf16_workspace_b_capacity.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t trainer_linear_bf16_cached_a_capacity() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return g_linear_bf16_cached_a_capacity.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t trainer_linear_bf16_cache_entry_count() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  return g_linear_bf16_cache_entry_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

int trainer_linear_cublaslt_grouped_layout_probe_status() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  std::int64_t rows_host = 1;
  std::int64_t cols_host = 1;
  std::int64_t ld_host = 1;
  std::int64_t* rows_device = nullptr;
  std::int64_t* cols_device = nullptr;
  std::int64_t* ld_device = nullptr;
  cublasLtMatrixLayout_t layout = nullptr;
  int status = static_cast<int>(cudaMalloc(&rows_device, sizeof(rows_host)));
  if (status == 0) {
    status = static_cast<int>(cudaMalloc(&cols_device, sizeof(cols_host)));
  }
  if (status == 0) {
    status = static_cast<int>(cudaMalloc(&ld_device, sizeof(ld_host)));
  }
  if (status == 0) {
    status = static_cast<int>(cudaMemcpy(rows_device, &rows_host, sizeof(rows_host), cudaMemcpyHostToDevice));
  }
  if (status == 0) {
    status = static_cast<int>(cudaMemcpy(cols_device, &cols_host, sizeof(cols_host), cudaMemcpyHostToDevice));
  }
  if (status == 0) {
    status = static_cast<int>(cudaMemcpy(ld_device, &ld_host, sizeof(ld_host), cudaMemcpyHostToDevice));
  }
  if (status == 0) {
    status = static_cast<int>(cublasLtGroupedMatrixLayoutCreate(
        &layout,
        CUDA_R_32F,
        1,
        rows_device,
        cols_device,
        ld_device));
  }
  if (layout != nullptr) {
    const int destroy_status = static_cast<int>(cublasLtMatrixLayoutDestroy(layout));
    if (status == 0) {
      status = destroy_status;
    }
  }
  if (rows_device != nullptr) {
    const int free_status = static_cast<int>(cudaFree(rows_device));
    if (status == 0) {
      status = free_status;
    }
  }
  if (cols_device != nullptr) {
    const int free_status = static_cast<int>(cudaFree(cols_device));
    if (status == 0) {
      status = free_status;
    }
  }
  if (ld_device != nullptr) {
    const int free_status = static_cast<int>(cudaFree(ld_device));
    if (status == 0) {
      status = free_status;
    }
  }
  return status;
#else
  return -1;
#endif
}

int trainer_linear_cublaslt_grouped_matmul_probe_status() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  cublasLtHandle_t handle = trainer_linear_cublaslt_handle();
  if (handle == nullptr) {
    return -2;
  }

  constexpr int group_count = 2;
  constexpr int m_host[group_count] = {16, 16};
  constexpr int n_host[group_count] = {2, 2};
  constexpr int k_host[group_count] = {16, 16};
  constexpr int a_ld_host[group_count] = {16, 16};
  constexpr int b_ld_host[group_count] = {16, 16};
  constexpr int c_ld_host[group_count] = {16, 16};
  constexpr std::size_t a_elements = 16 * 16;
  constexpr std::size_t b_elements = 16 * 2;
  constexpr std::size_t c_elements = 16 * 2;
  constexpr std::uint16_t bf16_one = 0x3f80;
  constexpr std::uint16_t expected_bf16 = 0x4180;

  std::array<std::uint16_t*, group_count> a_device = {nullptr, nullptr};
  std::array<std::uint16_t*, group_count> b_device = {nullptr, nullptr};
  std::array<std::uint16_t*, group_count> c_device = {nullptr, nullptr};
  const void** a_ptrs_device = nullptr;
  const void** b_ptrs_device = nullptr;
  void** c_ptrs_device = nullptr;
  std::int32_t* a_rows_device = nullptr;
  std::int32_t* a_cols_device = nullptr;
  std::int32_t* a_ld_device = nullptr;
  std::int32_t* b_rows_device = nullptr;
  std::int32_t* b_cols_device = nullptr;
  std::int32_t* b_ld_device = nullptr;
  std::int32_t* c_rows_device = nullptr;
  std::int32_t* c_cols_device = nullptr;
  std::int32_t* c_ld_device = nullptr;
  cublasLtMatmulDesc_t matmul_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned_results = 0;
  int status = 0;

  auto free_all = [&]() {
    if (preference != nullptr) cublasLtMatmulPreferenceDestroy(preference);
    if (a_desc != nullptr) cublasLtMatrixLayoutDestroy(a_desc);
    if (b_desc != nullptr) cublasLtMatrixLayoutDestroy(b_desc);
    if (c_desc != nullptr) cublasLtMatrixLayoutDestroy(c_desc);
    if (matmul_desc != nullptr) cublasLtMatmulDescDestroy(matmul_desc);
    if (a_ptrs_device != nullptr) cudaFree(a_ptrs_device);
    if (b_ptrs_device != nullptr) cudaFree(b_ptrs_device);
    if (c_ptrs_device != nullptr) cudaFree(c_ptrs_device);
    if (a_rows_device != nullptr) cudaFree(a_rows_device);
    if (a_cols_device != nullptr) cudaFree(a_cols_device);
    if (a_ld_device != nullptr) cudaFree(a_ld_device);
    if (b_rows_device != nullptr) cudaFree(b_rows_device);
    if (b_cols_device != nullptr) cudaFree(b_cols_device);
    if (b_ld_device != nullptr) cudaFree(b_ld_device);
    if (c_rows_device != nullptr) cudaFree(c_rows_device);
    if (c_cols_device != nullptr) cudaFree(c_cols_device);
    if (c_ld_device != nullptr) cudaFree(c_ld_device);
    for (int i = 0; i < group_count; ++i) {
      if (a_device[i] != nullptr) cudaFree(a_device[i]);
      if (b_device[i] != nullptr) cudaFree(b_device[i]);
      if (c_device[i] != nullptr) cudaFree(c_device[i]);
    }
  };
  auto fail = [&](int value) {
    free_all();
    return value;
  };
  auto cuda_step = [&](cudaError_t value) {
    if (status == 0 && value != cudaSuccess) {
      status = static_cast<int>(value);
    }
  };
  auto cublas_step = [&](cublasStatus_t value) {
    if (status == 0 && value != CUBLAS_STATUS_SUCCESS) {
      status = static_cast<int>(value);
    }
  };

  const std::array<std::int32_t, group_count> a_rows = {m_host[0], m_host[1]};
  const std::array<std::int32_t, group_count> a_cols = {k_host[0], k_host[1]};
  const std::array<std::int32_t, group_count> a_ld = {a_ld_host[0], a_ld_host[1]};
  const std::array<std::int32_t, group_count> b_rows = {k_host[0], k_host[1]};
  const std::array<std::int32_t, group_count> b_cols = {n_host[0], n_host[1]};
  const std::array<std::int32_t, group_count> b_ld = {b_ld_host[0], b_ld_host[1]};
  const std::array<std::int32_t, group_count> c_rows = {m_host[0], m_host[1]};
  const std::array<std::int32_t, group_count> c_cols = {n_host[0], n_host[1]};
  const std::array<std::int32_t, group_count> c_ld = {c_ld_host[0], c_ld_host[1]};

  for (int i = 0; i < group_count && status == 0; ++i) {
    cuda_step(cudaMalloc(reinterpret_cast<void**>(&a_device[i]), a_elements * sizeof(std::uint16_t)));
    cuda_step(cudaMalloc(reinterpret_cast<void**>(&b_device[i]), b_elements * sizeof(std::uint16_t)));
    cuda_step(cudaMalloc(reinterpret_cast<void**>(&c_device[i]), c_elements * sizeof(std::uint16_t)));
    if (status == 0) {
      std::vector<std::uint16_t> a_host(a_elements, bf16_one);
      std::vector<std::uint16_t> b_host(b_elements, bf16_one);
      cuda_step(cudaMemcpy(a_device[i], a_host.data(), a_host.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
      cuda_step(cudaMemcpy(b_device[i], b_host.data(), b_host.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
      cuda_step(cudaMemset(c_device[i], 0, c_elements * sizeof(std::uint16_t)));
    }
  }
  if (status != 0) return fail(status);

  const void* a_ptrs_host[group_count] = {a_device[0], a_device[1]};
  const void* b_ptrs_host[group_count] = {b_device[0], b_device[1]};
  void* c_ptrs_host[group_count] = {c_device[0], c_device[1]};
  cuda_step(cudaMalloc(reinterpret_cast<void**>(&a_ptrs_device), sizeof(a_ptrs_host)));
  cuda_step(cudaMalloc(reinterpret_cast<void**>(&b_ptrs_device), sizeof(b_ptrs_host)));
  cuda_step(cudaMalloc(reinterpret_cast<void**>(&c_ptrs_device), sizeof(c_ptrs_host)));
  cuda_step(cudaMalloc(reinterpret_cast<void**>(&a_rows_device), sizeof(std::int32_t) * group_count));
  cuda_step(cudaMalloc(reinterpret_cast<void**>(&a_cols_device), sizeof(std::int32_t) * group_count));
  cuda_step(cudaMalloc(reinterpret_cast<void**>(&a_ld_device), sizeof(std::int32_t) * group_count));
  cuda_step(cudaMalloc(reinterpret_cast<void**>(&b_rows_device), sizeof(std::int32_t) * group_count));
  cuda_step(cudaMalloc(reinterpret_cast<void**>(&b_cols_device), sizeof(std::int32_t) * group_count));
  cuda_step(cudaMalloc(reinterpret_cast<void**>(&b_ld_device), sizeof(std::int32_t) * group_count));
  cuda_step(cudaMalloc(reinterpret_cast<void**>(&c_rows_device), sizeof(std::int32_t) * group_count));
  cuda_step(cudaMalloc(reinterpret_cast<void**>(&c_cols_device), sizeof(std::int32_t) * group_count));
  cuda_step(cudaMalloc(reinterpret_cast<void**>(&c_ld_device), sizeof(std::int32_t) * group_count));
  if (status != 0) return fail(status);

  cuda_step(cudaMemcpy(a_ptrs_device, a_ptrs_host, sizeof(a_ptrs_host), cudaMemcpyHostToDevice));
  cuda_step(cudaMemcpy(b_ptrs_device, b_ptrs_host, sizeof(b_ptrs_host), cudaMemcpyHostToDevice));
  cuda_step(cudaMemcpy(c_ptrs_device, c_ptrs_host, sizeof(c_ptrs_host), cudaMemcpyHostToDevice));
  cuda_step(cudaMemcpy(a_rows_device, a_rows.data(), sizeof(std::int32_t) * group_count, cudaMemcpyHostToDevice));
  cuda_step(cudaMemcpy(a_cols_device, a_cols.data(), sizeof(std::int32_t) * group_count, cudaMemcpyHostToDevice));
  cuda_step(cudaMemcpy(a_ld_device, a_ld.data(), sizeof(std::int32_t) * group_count, cudaMemcpyHostToDevice));
  cuda_step(cudaMemcpy(b_rows_device, b_rows.data(), sizeof(std::int32_t) * group_count, cudaMemcpyHostToDevice));
  cuda_step(cudaMemcpy(b_cols_device, b_cols.data(), sizeof(std::int32_t) * group_count, cudaMemcpyHostToDevice));
  cuda_step(cudaMemcpy(b_ld_device, b_ld.data(), sizeof(std::int32_t) * group_count, cudaMemcpyHostToDevice));
  cuda_step(cudaMemcpy(c_rows_device, c_rows.data(), sizeof(std::int32_t) * group_count, cudaMemcpyHostToDevice));
  cuda_step(cudaMemcpy(c_cols_device, c_cols.data(), sizeof(std::int32_t) * group_count, cudaMemcpyHostToDevice));
  cuda_step(cudaMemcpy(c_ld_device, c_ld.data(), sizeof(std::int32_t) * group_count, cudaMemcpyHostToDevice));
  if (status != 0) return fail(status);

  cublas_step(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F_FAST_16BF, CUDA_R_32F));
  cublas_step(cublasLtGroupedMatrixLayoutCreate(
      &a_desc, CUDA_R_16BF, group_count, a_rows_device, a_cols_device, a_ld_device));
  cublas_step(cublasLtGroupedMatrixLayoutCreate(
      &b_desc, CUDA_R_16BF, group_count, b_rows_device, b_cols_device, b_ld_device));
  cublas_step(cublasLtGroupedMatrixLayoutCreate(
      &c_desc, CUDA_R_16BF, group_count, c_rows_device, c_cols_device, c_ld_device));
  const int batch_count = group_count;
  const cublasLtBatchMode_t grouped_mode = CUBLASLT_BATCH_MODE_GROUPED;
  if (status == 0) {
    cublas_step(cublasLtMatrixLayoutSetAttribute(
        a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    cublas_step(cublasLtMatrixLayoutSetAttribute(
        b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    cublas_step(cublasLtMatrixLayoutSetAttribute(
        c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    cublas_step(cublasLtMatrixLayoutSetAttribute(
        a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &grouped_mode, sizeof(grouped_mode)));
    cublas_step(cublasLtMatrixLayoutSetAttribute(
        b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &grouped_mode, sizeof(grouped_mode)));
    cublas_step(cublasLtMatrixLayoutSetAttribute(
        c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &grouped_mode, sizeof(grouped_mode)));
  }
  if (status != 0) return fail(status);

  cublas_step(cublasLtMatmulPreferenceCreate(&preference));
  if (status == 0) {
    cublas_step(cublasLtMatmulAlgoGetHeuristic(
        handle, matmul_desc, a_desc, b_desc, c_desc, c_desc, preference, 1, &heuristic, &returned_results));
  }
  if (status != 0) return fail(status);
  if (returned_results <= 0) return fail(-3);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublas_step(cublasLtMatmul(
      handle,
      matmul_desc,
      &alpha,
      a_ptrs_device,
      a_desc,
      b_ptrs_device,
      b_desc,
      &beta,
      c_ptrs_device,
      c_desc,
      c_ptrs_device,
      c_desc,
      &heuristic.algo,
      nullptr,
      0,
      nullptr));
  cuda_step(cudaDeviceSynchronize());
  if (status != 0) return fail(status);

  for (int i = 0; i < group_count && status == 0; ++i) {
    std::vector<std::uint16_t> c_host(c_elements, 0);
    cuda_step(cudaMemcpy(c_host.data(), c_device[i], c_host.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
    for (std::uint16_t value : c_host) {
      if (status == 0 && value != expected_bf16) {
        status = -4;
      }
    }
  }
  free_all();
  return status;
#else
  return -1;
#endif
}

int trainer_linear_cublas_grouped_bf16_gemm_probe_status() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  cublasHandle_t handle = trainer_linear_cublas_handle(nullptr);
  if (handle == nullptr) {
    return -2;
  }

  constexpr int group_count = 2;
  constexpr int batch_count = 2;
  const cublasOperation_t transa[group_count] = {CUBLAS_OP_N, CUBLAS_OP_N};
  const cublasOperation_t transb[group_count] = {CUBLAS_OP_N, CUBLAS_OP_N};
  const int m[group_count] = {16, 16};
  const int n[group_count] = {2, 2};
  const int k[group_count] = {16, 16};
  const int lda[group_count] = {16, 16};
  const int ldb[group_count] = {16, 16};
  const int ldc[group_count] = {16, 16};
  const int group_size[group_count] = {1, 1};
  const float alpha[group_count] = {1.0f, 1.0f};
  const float beta[group_count] = {0.0f, 0.0f};
  const std::uint16_t bf16_one = 0x3f80;
  const std::array<std::size_t, batch_count> a_elements = {256, 256};
  const std::array<std::size_t, batch_count> b_elements = {32, 32};
  const std::array<std::size_t, batch_count> c_elements = {32, 32};
  const std::array<std::uint16_t, batch_count> expected_bf16 = {0x4180, 0x4180};
  std::array<std::uint16_t*, batch_count> a_device = {nullptr, nullptr};
  std::array<std::uint16_t*, batch_count> b_device = {nullptr, nullptr};
  std::array<std::uint16_t*, batch_count> c_device = {nullptr, nullptr};
  int status = 0;

  for (int i = 0; i < batch_count && status == 0; ++i) {
    status = static_cast<int>(cudaMalloc(&a_device[i], a_elements[i] * sizeof(std::uint16_t)));
    if (status == 0) {
      status = static_cast<int>(cudaMalloc(&b_device[i], b_elements[i] * sizeof(std::uint16_t)));
    }
    if (status == 0) {
      status = static_cast<int>(cudaMalloc(&c_device[i], c_elements[i] * sizeof(std::uint16_t)));
    }
    if (status == 0) {
      std::vector<std::uint16_t> a_host(a_elements[i], bf16_one);
      status = static_cast<int>(cudaMemcpy(
          a_device[i],
          a_host.data(),
          a_host.size() * sizeof(std::uint16_t),
          cudaMemcpyHostToDevice));
    }
    if (status == 0) {
      std::vector<std::uint16_t> b_host(b_elements[i], bf16_one);
      status = static_cast<int>(cudaMemcpy(
          b_device[i],
          b_host.data(),
          b_host.size() * sizeof(std::uint16_t),
          cudaMemcpyHostToDevice));
    }
    if (status == 0) {
      status = static_cast<int>(cudaMemset(c_device[i], 0, c_elements[i] * sizeof(std::uint16_t)));
    }
  }

  if (status == 0) {
    const void* a_ptrs[batch_count] = {
        static_cast<const void*>(a_device[0]),
        static_cast<const void*>(a_device[1])};
    const void* b_ptrs[batch_count] = {
        static_cast<const void*>(b_device[0]),
        static_cast<const void*>(b_device[1])};
    void* c_ptrs[batch_count] = {
        static_cast<void*>(c_device[0]),
        static_cast<void*>(c_device[1])};
    status = static_cast<int>(cublasGemmGroupedBatchedEx(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a_ptrs,
        CUDA_R_16BF,
        lda,
        b_ptrs,
        CUDA_R_16BF,
        ldb,
        beta,
        c_ptrs,
        CUDA_R_16BF,
        ldc,
        group_count,
        group_size,
        CUBLAS_COMPUTE_32F));
  }
  if (status == 0) {
    status = static_cast<int>(cudaDeviceSynchronize());
  }
  for (int i = 0; i < batch_count && status == 0; ++i) {
    std::vector<std::uint16_t> c_host(c_elements[i], 0);
    status = static_cast<int>(cudaMemcpy(
        c_host.data(),
        c_device[i],
        c_host.size() * sizeof(std::uint16_t),
        cudaMemcpyDeviceToHost));
    for (std::uint16_t value : c_host) {
      if (status == 0 && value != expected_bf16[i]) {
        status = -3;
      }
    }
  }

  for (int i = 0; i < batch_count; ++i) {
    if (a_device[i] != nullptr) {
      const int free_status = static_cast<int>(cudaFree(a_device[i]));
      if (status == 0) {
        status = free_status;
      }
    }
    if (b_device[i] != nullptr) {
      const int free_status = static_cast<int>(cudaFree(b_device[i]));
      if (status == 0) {
        status = free_status;
      }
    }
    if (c_device[i] != nullptr) {
      const int free_status = static_cast<int>(cudaFree(c_device[i]));
      if (status == 0) {
        status = free_status;
      }
    }
  }
  return status;
#else
  return -1;
#endif
}

bool trainer_linear_cublaslt_prewarm_bf16_plan(
    int m,
    int n,
    int k,
    int op_a_value,
    int op_b_value,
    int lda,
    int ldb,
    int ldc,
    bool bgrad_epilogue) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  cublasLtHandle_t handle = trainer_linear_cublaslt_handle();
  if (handle == nullptr || m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
    return false;
  }
  const cublasOperation_t op_a = static_cast<cublasOperation_t>(op_a_value);
  const cublasOperation_t op_b = static_cast<cublasOperation_t>(op_b_value);
  const cublasLtEpilogue_t epilogue =
      bgrad_epilogue ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_DEFAULT;
  TrainerLinearCublasLtPlanKey key{
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      static_cast<int>(op_a),
      static_cast<int>(op_b),
      static_cast<int>(CUDA_R_16BF),
      static_cast<int>(CUDA_R_16BF),
      static_cast<int>(CUDA_R_32F),
      static_cast<int>(CUBLAS_COMPUTE_32F_FAST_16BF),
      static_cast<int>(epilogue)};
  TrainerLinearCublasLtPlan cached_plan;
  if (trainer_linear_cublaslt_descriptor_cache_enabled() &&
      find_trainer_linear_cublaslt_plan(key, &cached_plan)) {
    return true;
  }
  cublasLtMatmulDesc_t matmul_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  if (!create_trainer_linear_cublaslt_layouts(
          op_a,
          op_b,
          CUDA_R_16BF,
          CUDA_R_16BF,
          CUDA_R_32F,
          CUBLAS_COMPUTE_32F_FAST_16BF,
          m,
          n,
          k,
          lda,
          ldb,
          ldc,
          epilogue,
          nullptr,
          &matmul_desc,
          &a_desc,
          &b_desc,
          &c_desc)) {
    destroy_trainer_linear_cublaslt_layouts(matmul_desc, a_desc, b_desc, c_desc);
    return false;
  }
  std::lock_guard<std::mutex> lock(g_trainer_linear_cublaslt_workspace_mutex);
  TrainerLinearCublasLtPlan* plan =
      trainer_linear_cublaslt_plan_for(handle, key, matmul_desc, a_desc, b_desc, c_desc);
  const bool retained =
      plan != nullptr &&
      trainer_linear_cublaslt_descriptor_cache_enabled() &&
      plan->matmul_desc == matmul_desc;
  if (!retained) {
    destroy_trainer_linear_cublaslt_layouts(matmul_desc, a_desc, b_desc, c_desc);
  }
  return plan != nullptr;
#else
  (void)m;
  (void)n;
  (void)k;
  (void)op_a_value;
  (void)op_b_value;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)bgrad_epilogue;
  return false;
#endif
}

std::int64_t trainer_linear_shape_stats_count() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  std::lock_guard<std::mutex> lock(g_linear_shape_stats_mutex);
  return static_cast<std::int64_t>(g_linear_shape_stats.size());
#else
  return 0;
#endif
}

bool trainer_linear_shape_stats_entry(
    std::int64_t index,
    int* path,
    int* m,
    int* n,
    int* k,
    int* op_a,
    int* op_b,
    std::int64_t* calls,
    std::int64_t* total_us) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  std::lock_guard<std::mutex> lock(g_linear_shape_stats_mutex);
  if (index < 0 || index >= static_cast<std::int64_t>(g_linear_shape_stats.size())) {
    return false;
  }
  const LinearShapeStat& stat = g_linear_shape_stats[static_cast<std::size_t>(index)];
  if (path != nullptr) {
    *path = stat.path;
  }
  if (m != nullptr) {
    *m = stat.m;
  }
  if (n != nullptr) {
    *n = stat.n;
  }
  if (k != nullptr) {
    *k = stat.k;
  }
  if (op_a != nullptr) {
    *op_a = stat.op_a;
  }
  if (op_b != nullptr) {
    *op_b = stat.op_b;
  }
  if (calls != nullptr) {
    *calls = stat.calls;
  }
  if (total_us != nullptr) {
    *total_us = stat.total_us;
  }
  return true;
#else
  (void)index;
  (void)path;
  (void)m;
  (void)n;
  (void)k;
  (void)op_a;
  (void)op_b;
  (void)calls;
  (void)total_us;
  return false;
#endif
}

bool trainer_linear_shape_stats_entry_v2(
    std::int64_t index,
    int* path,
    int* m,
    int* n,
    int* k,
    int* op_a,
    int* op_b,
    std::int64_t* calls,
    std::int64_t* total_us,
    int* cublaslt_selected_heuristic,
    int* cublaslt_returned_heuristics,
    std::int64_t* cublaslt_workspace_bytes) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  std::lock_guard<std::mutex> lock(g_linear_shape_stats_mutex);
  if (index < 0 || index >= static_cast<std::int64_t>(g_linear_shape_stats.size())) {
    return false;
  }
  const LinearShapeStat& stat = g_linear_shape_stats[static_cast<std::size_t>(index)];
  if (path != nullptr) {
    *path = stat.path;
  }
  if (m != nullptr) {
    *m = stat.m;
  }
  if (n != nullptr) {
    *n = stat.n;
  }
  if (k != nullptr) {
    *k = stat.k;
  }
  if (op_a != nullptr) {
    *op_a = stat.op_a;
  }
  if (op_b != nullptr) {
    *op_b = stat.op_b;
  }
  if (calls != nullptr) {
    *calls = stat.calls;
  }
  if (total_us != nullptr) {
    *total_us = stat.total_us;
  }
  if (cublaslt_selected_heuristic != nullptr) {
    *cublaslt_selected_heuristic = stat.cublaslt_selected_heuristic;
  }
  if (cublaslt_returned_heuristics != nullptr) {
    *cublaslt_returned_heuristics = stat.cublaslt_returned_heuristics;
  }
  if (cublaslt_workspace_bytes != nullptr) {
    *cublaslt_workspace_bytes = stat.cublaslt_workspace_bytes;
  }
  return true;
#else
  (void)index;
  (void)path;
  (void)m;
  (void)n;
  (void)k;
  (void)op_a;
  (void)op_b;
  (void)calls;
  (void)total_us;
  (void)cublaslt_selected_heuristic;
  (void)cublaslt_returned_heuristics;
  (void)cublaslt_workspace_bytes;
  return false;
#endif
}

std::int64_t trainer_linear_cublaslt_plan_cache_count() {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  std::lock_guard<std::mutex> lock(g_trainer_linear_cublaslt_workspace_mutex);
  std::int64_t count = 0;
  for (const TrainerLinearCublasLtPlan& plan : g_trainer_linear_cublaslt_workspace.plans) {
    if (plan.valid) {
      count += 1;
    }
  }
  return count;
#else
  return 0;
#endif
}

bool trainer_linear_cublaslt_plan_cache_entry(
    std::int64_t index,
    int* m,
    int* n,
    int* k,
    int* op_a,
    int* op_b,
    int* selected_heuristic,
    int* returned_heuristics,
    std::int64_t* workspace_bytes,
    int* epilogue) {
#if defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR)
  std::lock_guard<std::mutex> lock(g_trainer_linear_cublaslt_workspace_mutex);
  if (index < 0) {
    return false;
  }
  std::int64_t valid_index = 0;
  for (const TrainerLinearCublasLtPlan& plan : g_trainer_linear_cublaslt_workspace.plans) {
    if (!plan.valid) {
      continue;
    }
    if (valid_index == index) {
      if (m != nullptr) {
        *m = plan.key.m;
      }
      if (n != nullptr) {
        *n = plan.key.n;
      }
      if (k != nullptr) {
        *k = plan.key.k;
      }
      if (op_a != nullptr) {
        *op_a = plan.key.op_a;
      }
      if (op_b != nullptr) {
        *op_b = plan.key.op_b;
      }
      if (selected_heuristic != nullptr) {
        *selected_heuristic = plan.selected_heuristic;
      }
      if (returned_heuristics != nullptr) {
        *returned_heuristics = plan.returned_heuristics;
      }
      if (workspace_bytes != nullptr) {
        *workspace_bytes = static_cast<std::int64_t>(plan.workspace_size);
      }
      if (epilogue != nullptr) {
        *epilogue = plan.key.epilogue;
      }
      return true;
    }
    valid_index += 1;
  }
  return false;
#else
  (void)index;
  (void)m;
  (void)n;
  (void)k;
  (void)op_a;
  (void)op_b;
  (void)selected_heuristic;
  (void)returned_heuristics;
  (void)workspace_bytes;
  (void)epilogue;
  return false;
#endif
}

std::int64_t attention_forward_row_launch_count() {
  return g_attention_forward_row_launch_count.load(std::memory_order_relaxed);
}

std::int64_t attention_forward_tk_launch_count() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return g_attention_forward_tk_launch_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t attention_backward_tk_launch_count() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return g_attention_backward_tk_launch_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

int attention_backward_tk_block_size() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION) && defined(LLMK_SM120_ATTN_BWD_BLOCK)
  return LLMK_SM120_ATTN_BWD_BLOCK;
#elif defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return 16;
#else
  return 0;
#endif
}

std::int64_t attention_backward_float_hd64_dprep_launch_count() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return g_attention_backward_float_hd64_dprep_launch_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t attention_backward_dprep_timing_us() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return g_attention_backward_dprep_timing_us.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t attention_backward_dprep_timing_count() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return g_attention_backward_dprep_timing_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t attention_backward_tk_timing_us() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return g_attention_backward_tk_timing_us.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t attention_backward_tk_timing_count() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return g_attention_backward_tk_timing_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t attention_tk_workspace_allocation_count() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return g_attention_tk_workspace_allocation_count.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t attention_tk_workspace_element_capacity() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return g_attention_tk_workspace_element_capacity.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t attention_tk_workspace_row_capacity() {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return g_attention_tk_workspace_row_capacity.load(std::memory_order_relaxed);
#else
  return 0;
#endif
}

std::int64_t token_cross_entropy_workspace_allocation_count() {
  return g_token_cross_entropy_workspace_allocation_count.load(std::memory_order_relaxed);
}

std::int64_t token_cross_entropy_workspace_row_capacity() {
  return g_token_cross_entropy_workspace_row_capacity.load(std::memory_order_relaxed);
}

void reset_lm_head_classifier_chunk_stats() {
  g_lm_head_classifier_chunk_launch_count.store(0, std::memory_order_relaxed);
  g_lm_head_classifier_last_rows.store(0, std::memory_order_relaxed);
  g_lm_head_classifier_last_vocab.store(0, std::memory_order_relaxed);
  g_lm_head_classifier_last_row_stride.store(0, std::memory_order_relaxed);
  g_lm_head_classifier_loss_bin_launch_count.store(0, std::memory_order_relaxed);
}

std::int64_t lm_head_classifier_chunk_launch_count() {
  return g_lm_head_classifier_chunk_launch_count.load(std::memory_order_relaxed);
}

std::int64_t lm_head_classifier_last_rows() {
  return g_lm_head_classifier_last_rows.load(std::memory_order_relaxed);
}

std::int64_t lm_head_classifier_last_vocab() {
  return g_lm_head_classifier_last_vocab.load(std::memory_order_relaxed);
}

std::int64_t lm_head_classifier_last_row_stride() {
  return g_lm_head_classifier_last_row_stride.load(std::memory_order_relaxed);
}

std::int64_t lm_head_classifier_loss_bin_launch_count() {
  return g_lm_head_classifier_loss_bin_launch_count.load(std::memory_order_relaxed);
}

std::int64_t attention_forward_row_fallback_count() {
  return g_attention_forward_row_fallback_count.load(std::memory_order_relaxed);
}

std::int64_t attention_forward_scalar_launch_count() {
  return g_attention_forward_scalar_launch_count.load(std::memory_order_relaxed);
}

int attention_forward_row_last_error() {
  return g_attention_forward_row_last_error.load(std::memory_order_relaxed);
}

int attention_forward_row_prelaunch_clear_error() {
  return g_attention_forward_row_prelaunch_clear_error.load(std::memory_order_relaxed);
}

int attention_forward_row_prelaunch_peek_error() {
  return g_attention_forward_row_prelaunch_peek_error.load(std::memory_order_relaxed);
}

std::int64_t attention_forward_row_grid_x() {
  return g_attention_forward_row_grid_x.load(std::memory_order_relaxed);
}

std::int64_t attention_forward_row_grid_y() {
  return g_attention_forward_row_grid_y.load(std::memory_order_relaxed);
}

std::int64_t attention_forward_row_grid_z() {
  return g_attention_forward_row_grid_z.load(std::memory_order_relaxed);
}

std::int64_t attention_forward_row_block_x() {
  return g_attention_forward_row_block_x.load(std::memory_order_relaxed);
}

int attention_forward_row_attr_status() {
  return g_attention_forward_row_attr_status.load(std::memory_order_relaxed);
}

int attention_forward_row_attr_max_threads_per_block() {
  return g_attention_forward_row_attr_max_threads_per_block.load(std::memory_order_relaxed);
}

int attention_forward_row_attr_num_regs() {
  return g_attention_forward_row_attr_num_regs.load(std::memory_order_relaxed);
}

std::int64_t attention_forward_row_attr_shared_size_bytes() {
  return g_attention_forward_row_attr_shared_size_bytes.load(std::memory_order_relaxed);
}

std::int64_t attention_forward_row_attr_const_size_bytes() {
  return g_attention_forward_row_attr_const_size_bytes.load(std::memory_order_relaxed);
}

std::int64_t attention_forward_row_attr_local_size_bytes() {
  return g_attention_forward_row_attr_local_size_bytes.load(std::memory_order_relaxed);
}

void launch_scaled_dot_product_attention_backward_float32_impl(
    const float* q,
    const float* k,
    const float* v,
    const float* grad_out,
    float* grad_q,
    float* grad_k,
    float* grad_v,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    bool grad_out_merged,
    cudaStream_t stream) {
  const std::int64_t q_n = batch * query_heads * seq_q * qk_dim;
  const std::int64_t k_n = batch * key_heads * seq_k * qk_dim;
  const std::int64_t v_n = batch * key_heads * seq_k * value_dim;
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (use_tk_sm120_attention(
          query_heads,
          key_heads,
          seq_q,
          seq_k,
          qk_dim,
          value_dim,
          is_causal,
          right_align_causal,
          use_sparse_rules,
          window,
          num_sinks,
          block_size,
          compress_stride) &&
      launch_tk_attention_backward_float32(
          q,
          k,
          v,
          grad_out,
          grad_q,
          grad_k,
          grad_v,
          batch,
          query_heads,
          seq_q,
          qk_dim,
          grad_out_merged,
          stream) == 0) {
    return;
  }
#endif
  const bool use_row_kernels =
      qk_dim <= kGpt2AttentionHeadDim &&
      value_dim <= kGpt2AttentionHeadDim &&
      seq_q <= 1024 &&
      seq_k <= 1024;
  if (use_row_kernels) {
    const std::int64_t q_rows = batch * query_heads * seq_q;
    const std::int64_t total_grad_n = q_n + k_n + v_n;
    const int zero_blocks = static_cast<int>((total_grad_n + kTileSize - 1) / kTileSize);
    zero_three_float32_kernel<<<zero_blocks, 1, 0, stream>>>(
        grad_q,
        grad_k,
        grad_v,
        q_n,
        k_n,
        v_n);
    scaled_dot_product_attention_backward_query_row_atomic_float32_kernel<<<static_cast<int>(q_rows), 1, 0, stream>>>(
        q,
        k,
        v,
        grad_out,
        grad_q,
        grad_k,
        grad_v,
        q_rows,
        query_heads,
        key_heads,
        seq_q,
        seq_k,
        qk_dim,
        value_dim,
        scale,
        is_causal,
        right_align_causal,
        use_sparse_rules,
        window,
        num_sinks,
        block_size,
        compress_stride,
        grad_out_merged);
    return;
  }
  scaled_dot_product_attention_backward_q_float32_kernel<<<static_cast<int>(q_n), 1, 0, stream>>>(
      q,
      k,
      v,
      grad_out,
      grad_q,
      q_n,
      query_heads,
      key_heads,
      seq_q,
      seq_k,
      qk_dim,
      value_dim,
      scale,
      is_causal,
      right_align_causal,
      use_sparse_rules,
      window,
      num_sinks,
      block_size,
      compress_stride,
      grad_out_merged);
  scaled_dot_product_attention_backward_k_float32_kernel<<<static_cast<int>(k_n), 1, 0, stream>>>(
      q,
      k,
      v,
      grad_out,
      grad_k,
      k_n,
      query_heads,
      key_heads,
      seq_q,
      seq_k,
      qk_dim,
      value_dim,
      scale,
      is_causal,
      right_align_causal,
      use_sparse_rules,
      window,
      num_sinks,
      block_size,
      compress_stride,
      grad_out_merged);
  scaled_dot_product_attention_backward_v_float32_kernel<<<static_cast<int>(v_n), 1, 0, stream>>>(
      q,
      k,
      grad_out,
      grad_v,
      v_n,
      query_heads,
      key_heads,
      seq_q,
      seq_k,
      qk_dim,
      value_dim,
      scale,
      is_causal,
      right_align_causal,
      use_sparse_rules,
      window,
      num_sinks,
      block_size,
      compress_stride,
      grad_out_merged);
}

void launch_scaled_dot_product_attention_backward_float32(
    const float* q,
    const float* k,
    const float* v,
    const float* grad_out,
    float* grad_q,
    float* grad_k,
    float* grad_v,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    cudaStream_t stream) {
  launch_scaled_dot_product_attention_backward_float32_impl(
      q,
      k,
      v,
      grad_out,
      grad_q,
      grad_k,
      grad_v,
      batch,
      query_heads,
      key_heads,
      seq_q,
      seq_k,
      qk_dim,
      value_dim,
      scale,
      is_causal,
      right_align_causal,
      use_sparse_rules,
      window,
      num_sinks,
      block_size,
      compress_stride,
      false,
      stream);
}

void launch_scaled_dot_product_attention_backward_from_merged_grad_float32(
    const float* q,
    const float* k,
    const float* v,
    const float* grad_out,
    float* grad_q,
    float* grad_k,
    float* grad_v,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    cudaStream_t stream) {
  launch_scaled_dot_product_attention_backward_float32_impl(
      q,
      k,
      v,
      grad_out,
      grad_q,
      grad_k,
      grad_v,
      batch,
      query_heads,
      key_heads,
      seq_q,
      seq_k,
      qk_dim,
      value_dim,
      scale,
      is_causal,
      right_align_causal,
      use_sparse_rules,
      window,
      num_sinks,
      block_size,
      compress_stride,
      true,
      stream);
}

void launch_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32(
    const float* q,
    const float* k,
    const float* v,
    const float* grad_out,
    float* grad_qkv,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (use_tk_sm120_attention(
          query_heads,
          key_heads,
          seq_q,
          seq_k,
          qk_dim,
          value_dim,
          is_causal,
          right_align_causal,
          use_sparse_rules,
          window,
          num_sinks,
          block_size,
          compress_stride) &&
      launch_tk_attention_backward_to_qkv_float32(
          q,
          k,
          v,
          grad_out,
          grad_qkv,
          batch,
          query_heads,
          seq_q,
          qk_dim,
          true,
          stream) == 0) {
    return;
  }
#endif
}

void launch_scaled_dot_product_attention_backward_to_qkv_reuse_forward_from_merged_grad_float32(
    const float* grad_out,
    float* grad_qkv,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    cudaStream_t stream) {
  (void)scale;
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (use_tk_sm120_attention(
          query_heads,
          key_heads,
          seq_q,
          seq_k,
          qk_dim,
          value_dim,
          is_causal,
          right_align_causal,
          use_sparse_rules,
          window,
          num_sinks,
          block_size,
          compress_stride) &&
      launch_tk_attention_backward_to_qkv_reuse_forward_float32(
          grad_out,
          grad_qkv,
          batch,
          query_heads,
          seq_q,
          qk_dim,
          true,
          stream) == 0) {
    return;
  }
#endif
}

int launch_scaled_dot_product_attention_packed_qkv_bf16_float32(
    const std::uint16_t* qkv_bf16_bits,
    std::uint16_t* out_bf16_bits,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    cudaStream_t stream) {
  (void)scale;
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (!use_tk_sm120_attention(
          query_heads,
          key_heads,
          seq_q,
          seq_k,
          qk_dim,
          value_dim,
          is_causal,
          right_align_causal,
          use_sparse_rules,
          window,
          num_sinks,
          block_size,
          compress_stride)) {
    return 2;
  }
  return launch_tk_attention_packed_qkv_forward_bf16_float32(
      qkv_bf16_bits,
      out_bf16_bits,
      batch,
      query_heads,
      seq_q,
      qk_dim,
      stream);
#else
  (void)qkv_bf16_bits;
  (void)out_bf16_bits;
  (void)batch;
  (void)query_heads;
  (void)key_heads;
  (void)seq_q;
  (void)seq_k;
  (void)qk_dim;
  (void)value_dim;
  (void)is_causal;
  (void)right_align_causal;
  (void)use_sparse_rules;
  (void)window;
  (void)num_sinks;
  (void)block_size;
  (void)compress_stride;
  (void)stream;
  return 2;
#endif
}

int launch_scaled_dot_product_attention_packed_qkv_store_lse_bf16_float32(
    const std::uint16_t* qkv_bf16_bits,
    std::uint16_t* out_bf16_bits,
    float* saved_lse,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    cudaStream_t stream) {
  (void)scale;
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (!use_tk_sm120_attention(
          query_heads,
          key_heads,
          seq_q,
          seq_k,
          qk_dim,
          value_dim,
          is_causal,
          right_align_causal,
          use_sparse_rules,
          window,
          num_sinks,
          block_size,
          compress_stride)) {
    return 2;
  }
  return launch_tk_attention_packed_qkv_forward_store_lse_bf16_float32(
      qkv_bf16_bits,
      out_bf16_bits,
      saved_lse,
      batch,
      query_heads,
      seq_q,
      qk_dim,
      stream);
#else
  (void)qkv_bf16_bits;
  (void)out_bf16_bits;
  (void)saved_lse;
  (void)batch;
  (void)query_heads;
  (void)key_heads;
  (void)seq_q;
  (void)seq_k;
  (void)qk_dim;
  (void)value_dim;
  (void)is_causal;
  (void)right_align_causal;
  (void)use_sparse_rules;
  (void)window;
  (void)num_sinks;
  (void)block_size;
  (void)compress_stride;
  (void)stream;
  return 2;
#endif
}

int launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_merged_grad_float32(
    const std::uint16_t* qkv_bf16_bits,
    const std::uint16_t* out_bf16_bits,
    const float* grad_out,
    float* grad_qkv,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    cudaStream_t stream) {
  (void)scale;
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (!use_tk_sm120_attention(
          query_heads,
          key_heads,
          seq_q,
          seq_k,
          qk_dim,
          value_dim,
          is_causal,
          right_align_causal,
          use_sparse_rules,
          window,
          num_sinks,
          block_size,
          compress_stride)) {
    return 2;
  }
  return launch_tk_attention_packed_qkv_backward_to_qkv_float32(
      qkv_bf16_bits,
      out_bf16_bits,
      nullptr,
      grad_out,
      grad_qkv,
      batch,
      query_heads,
      seq_q,
      qk_dim,
      true,
      stream);
#else
  (void)qkv_bf16_bits;
  (void)out_bf16_bits;
  (void)grad_out;
  (void)grad_qkv;
  (void)batch;
  (void)query_heads;
  (void)key_heads;
  (void)seq_q;
  (void)seq_k;
  (void)qk_dim;
  (void)value_dim;
  (void)is_causal;
  (void)right_align_causal;
  (void)use_sparse_rules;
  (void)window;
  (void)num_sinks;
  (void)block_size;
  (void)compress_stride;
  (void)stream;
  return 2;
#endif
}

int launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_saved_lse_bf16_from_merged_grad_float32(
    const std::uint16_t* qkv_bf16_bits,
    const std::uint16_t* out_bf16_bits,
    const float* saved_lse,
    const float* grad_out,
    float* grad_qkv,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    cudaStream_t stream) {
  (void)scale;
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (!use_tk_sm120_attention(
          query_heads,
          key_heads,
          seq_q,
          seq_k,
          qk_dim,
          value_dim,
          is_causal,
          right_align_causal,
          use_sparse_rules,
          window,
          num_sinks,
          block_size,
          compress_stride)) {
    return 2;
  }
  return launch_tk_attention_packed_qkv_backward_to_qkv_float32(
      qkv_bf16_bits,
      out_bf16_bits,
      saved_lse,
      grad_out,
      grad_qkv,
      batch,
      query_heads,
      seq_q,
      qk_dim,
      true,
      stream);
#else
  (void)qkv_bf16_bits;
  (void)out_bf16_bits;
  (void)saved_lse;
  (void)grad_out;
  (void)grad_qkv;
  (void)batch;
  (void)query_heads;
  (void)key_heads;
  (void)seq_q;
  (void)seq_k;
  (void)qk_dim;
  (void)value_dim;
  (void)is_causal;
  (void)right_align_causal;
  (void)use_sparse_rules;
  (void)window;
  (void)num_sinks;
  (void)block_size;
  (void)compress_stride;
  (void)stream;
  return 2;
#endif
}

int launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_merged_grad_float32(
    const std::uint16_t* qkv_bf16_bits,
    const std::uint16_t* out_bf16_bits,
    const float* grad_out,
    std::uint16_t* grad_qkv_bf16_bits,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    cudaStream_t stream) {
  (void)scale;
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (!use_tk_sm120_attention(
          query_heads,
          key_heads,
          seq_q,
          seq_k,
          qk_dim,
          value_dim,
          is_causal,
          right_align_causal,
          use_sparse_rules,
          window,
          num_sinks,
          block_size,
          compress_stride)) {
    return 2;
  }
  return launch_tk_attention_packed_qkv_backward_to_qkv_bf16_bits(
      qkv_bf16_bits,
      out_bf16_bits,
      nullptr,
      grad_out,
      grad_qkv_bf16_bits,
      batch,
      query_heads,
      seq_q,
      qk_dim,
      true,
      stream);
#else
  (void)qkv_bf16_bits;
  (void)out_bf16_bits;
  (void)grad_out;
  (void)grad_qkv_bf16_bits;
  (void)batch;
  (void)query_heads;
  (void)key_heads;
  (void)seq_q;
  (void)seq_k;
  (void)qk_dim;
  (void)value_dim;
  (void)is_causal;
  (void)right_align_causal;
  (void)use_sparse_rules;
  (void)window;
  (void)num_sinks;
  (void)block_size;
  (void)compress_stride;
  (void)stream;
  return 2;
#endif
}

int launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_merged_grad_float32(
    const std::uint16_t* qkv_bf16_bits,
    const std::uint16_t* out_bf16_bits,
    const float* saved_lse,
    const float* grad_out,
    std::uint16_t* grad_qkv_bf16_bits,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    cudaStream_t stream) {
  (void)scale;
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (!use_tk_sm120_attention(
          query_heads,
          key_heads,
          seq_q,
          seq_k,
          qk_dim,
          value_dim,
          is_causal,
          right_align_causal,
          use_sparse_rules,
          window,
          num_sinks,
          block_size,
          compress_stride)) {
    return 2;
  }
  return launch_tk_attention_packed_qkv_backward_to_qkv_bf16_bits(
      qkv_bf16_bits,
      out_bf16_bits,
      saved_lse,
      grad_out,
      grad_qkv_bf16_bits,
      batch,
      query_heads,
      seq_q,
      qk_dim,
      true,
      stream);
#else
  (void)qkv_bf16_bits;
  (void)out_bf16_bits;
  (void)saved_lse;
  (void)grad_out;
  (void)grad_qkv_bf16_bits;
  (void)batch;
  (void)query_heads;
  (void)key_heads;
  (void)seq_q;
  (void)seq_k;
  (void)qk_dim;
  (void)value_dim;
  (void)is_causal;
  (void)right_align_causal;
  (void)use_sparse_rules;
  (void)window;
  (void)num_sinks;
  (void)block_size;
  (void)compress_stride;
  (void)stream;
  return 2;
#endif
}

int launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_bf16_merged_grad_float32(
    const std::uint16_t* qkv_bf16_bits,
    const std::uint16_t* out_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    std::uint16_t* grad_qkv_bf16_bits,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    cudaStream_t stream) {
  (void)scale;
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (!use_tk_sm120_attention(
          query_heads,
          key_heads,
          seq_q,
          seq_k,
          qk_dim,
          value_dim,
          is_causal,
          right_align_causal,
          use_sparse_rules,
          window,
          num_sinks,
          block_size,
          compress_stride)) {
    return 2;
  }
  return launch_tk_attention_packed_qkv_backward_to_qkv_bf16_bits_from_bf16_grad_bits(
      qkv_bf16_bits,
      out_bf16_bits,
      nullptr,
      grad_out_bf16_bits,
      grad_qkv_bf16_bits,
      batch,
      query_heads,
      seq_q,
      qk_dim,
      true,
      stream);
#else
  (void)qkv_bf16_bits;
  (void)out_bf16_bits;
  (void)grad_out_bf16_bits;
  (void)grad_qkv_bf16_bits;
  (void)batch;
  (void)query_heads;
  (void)key_heads;
  (void)seq_q;
  (void)seq_k;
  (void)qk_dim;
  (void)value_dim;
  (void)is_causal;
  (void)right_align_causal;
  (void)use_sparse_rules;
  (void)window;
  (void)num_sinks;
  (void)block_size;
  (void)compress_stride;
  (void)stream;
  return 2;
#endif
}

int launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_bf16_merged_grad_float32(
    const std::uint16_t* qkv_bf16_bits,
    const std::uint16_t* out_bf16_bits,
    const float* saved_lse,
    const std::uint16_t* grad_out_bf16_bits,
    std::uint16_t* grad_qkv_bf16_bits,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    cudaStream_t stream) {
  (void)scale;
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (!use_tk_sm120_attention(
          query_heads,
          key_heads,
          seq_q,
          seq_k,
          qk_dim,
          value_dim,
          is_causal,
          right_align_causal,
          use_sparse_rules,
          window,
          num_sinks,
          block_size,
          compress_stride)) {
    return 2;
  }
  return launch_tk_attention_packed_qkv_backward_to_qkv_bf16_bits_from_bf16_grad_bits(
      qkv_bf16_bits,
      out_bf16_bits,
      saved_lse,
      grad_out_bf16_bits,
      grad_qkv_bf16_bits,
      batch,
      query_heads,
      seq_q,
      qk_dim,
      true,
      stream);
#else
  (void)qkv_bf16_bits;
  (void)out_bf16_bits;
  (void)saved_lse;
  (void)grad_out_bf16_bits;
  (void)grad_qkv_bf16_bits;
  (void)batch;
  (void)query_heads;
  (void)key_heads;
  (void)seq_q;
  (void)seq_k;
  (void)qk_dim;
  (void)value_dim;
  (void)is_causal;
  (void)right_align_causal;
  (void)use_sparse_rules;
  (void)window;
  (void)num_sinks;
  (void)block_size;
  (void)compress_stride;
  (void)stream;
  return 2;
#endif
}

int launch_scaled_dot_product_attention_store_tk_bf16_float32(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    std::uint16_t* saved_q_bf16_bits,
    std::uint16_t* saved_k_bf16_bits,
    std::uint16_t* saved_v_bf16_bits,
    std::uint16_t* saved_o_bf16_bits,
    float* saved_lse,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    cudaStream_t stream) {
  (void)scale;
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (use_tk_sm120_attention(
          query_heads,
          key_heads,
          seq_q,
          seq_k,
          qk_dim,
          value_dim,
          is_causal,
          right_align_causal,
          use_sparse_rules,
          window,
          num_sinks,
          block_size,
          compress_stride)) {
    return launch_tk_attention_forward_store_bf16_float32(
        q,
        k,
        v,
        out,
        saved_q_bf16_bits,
        saved_k_bf16_bits,
        saved_v_bf16_bits,
        saved_o_bf16_bits,
        saved_lse,
        batch,
        query_heads,
        seq_q,
        qk_dim,
        stream);
  }
#else
  (void)q;
  (void)k;
  (void)v;
  (void)out;
  (void)saved_q_bf16_bits;
  (void)saved_k_bf16_bits;
  (void)saved_v_bf16_bits;
  (void)saved_o_bf16_bits;
  (void)saved_lse;
  (void)batch;
  (void)query_heads;
  (void)key_heads;
  (void)seq_q;
  (void)seq_k;
  (void)qk_dim;
  (void)value_dim;
  (void)is_causal;
  (void)right_align_causal;
  (void)use_sparse_rules;
  (void)window;
  (void)num_sinks;
  (void)block_size;
  (void)compress_stride;
  (void)stream;
#endif
  return 2;
}

int launch_attention_tk_store_forward_workspace_bf16(
    std::uint16_t* saved_q_bf16_bits,
    std::uint16_t* saved_k_bf16_bits,
    std::uint16_t* saved_v_bf16_bits,
    std::uint16_t* saved_o_bf16_bits,
    float* saved_lse,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    cudaStream_t stream) {
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  return launch_tk_attention_store_forward_workspace_bf16(
      saved_q_bf16_bits,
      saved_k_bf16_bits,
      saved_v_bf16_bits,
      saved_o_bf16_bits,
      saved_lse,
      batch,
      heads,
      seq_len,
      head_dim,
      stream);
#else
  (void)saved_q_bf16_bits;
  (void)saved_k_bf16_bits;
  (void)saved_v_bf16_bits;
  (void)saved_o_bf16_bits;
  (void)saved_lse;
  (void)batch;
  (void)heads;
  (void)seq_len;
  (void)head_dim;
  (void)stream;
  return 2;
#endif
}

int launch_scaled_dot_product_attention_backward_to_qkv_from_saved_tk_bf16_from_merged_grad_float32(
    const std::uint16_t* saved_q_bf16_bits,
    const std::uint16_t* saved_k_bf16_bits,
    const std::uint16_t* saved_v_bf16_bits,
    const std::uint16_t* saved_o_bf16_bits,
    const float* saved_lse,
    const float* grad_out,
    float* grad_qkv,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    cudaStream_t stream) {
  (void)scale;
#if defined(NFN_TILE_CUDA_USE_TK_ATTENTION)
  if (!use_tk_sm120_attention(
          query_heads,
          key_heads,
          seq_q,
          seq_k,
          qk_dim,
          value_dim,
          is_causal,
          right_align_causal,
          use_sparse_rules,
          window,
          num_sinks,
          block_size,
          compress_stride)) {
    return 2;
  }
  return launch_tk_attention_backward_to_qkv_from_saved_bf16_float32(
      saved_q_bf16_bits,
      saved_k_bf16_bits,
      saved_v_bf16_bits,
      saved_o_bf16_bits,
      saved_lse,
      grad_out,
      grad_qkv,
      batch,
      query_heads,
      seq_q,
      qk_dim,
      true,
      stream);
#else
  (void)saved_q_bf16_bits;
  (void)saved_k_bf16_bits;
  (void)saved_v_bf16_bits;
  (void)saved_o_bf16_bits;
  (void)saved_lse;
  (void)grad_out;
  (void)grad_qkv;
  (void)batch;
  (void)query_heads;
  (void)key_heads;
  (void)seq_q;
  (void)seq_k;
  (void)qk_dim;
  (void)value_dim;
  (void)is_causal;
  (void)right_align_causal;
  (void)use_sparse_rules;
  (void)window;
  (void)num_sinks;
  (void)block_size;
  (void)compress_stride;
  (void)stream;
  return 2;
#endif
}

void launch_random_timesteps_float32(float* out, std::int64_t batch, std::int64_t counter, cudaStream_t stream) {
  const int blocks = static_cast<int>((batch + kTileSize - 1) / kTileSize);
  random_timesteps_float32_kernel<<<blocks, 1, 0, stream>>>(out, batch, counter);
}

void launch_mask_scheduler_int64(
    const std::int64_t* tokens,
    const float* timesteps,
    std::int64_t* out,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t mask_token_id,
    std::int64_t counter,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  mask_scheduler_int64_kernel<<<blocks, 1, 0, stream>>>(tokens, timesteps, out, n, seq_len, mask_token_id, counter);
}

void launch_jepa_mask_int64(
    const std::int64_t* tokens,
    std::int64_t* masked_tokens,
    float* mask_values,
    std::int64_t n,
    std::int64_t seq_len,
    float mask_ratio,
    std::int64_t mask_token_id,
    int strategy,
    std::int64_t num_blocks,
    float min_block_ratio,
    float max_block_ratio,
    std::int64_t counter,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  jepa_mask_int64_kernel<<<blocks, 1, 0, stream>>>(
      tokens, masked_tokens, mask_values, n, seq_len, mask_ratio, mask_token_id, strategy, num_blocks, min_block_ratio, max_block_ratio, counter);
}

}  // namespace neuralfn::tile_cuda
