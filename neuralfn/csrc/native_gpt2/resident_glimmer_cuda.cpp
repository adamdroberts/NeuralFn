#include "resident_glimmer_cuda.h"

#include "../native_train/tile_ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace neuralfn::resident_glimmer_cuda {
namespace {

constexpr int kCudaSuccess = 0;
constexpr int kCudaMemcpyHostToDevice = 1;
constexpr int kCudaMemcpyDeviceToHost = 2;
constexpr int kCudaMemcpyDeviceToDevice = 3;

bool feature_enabled_default_on(const char* name) noexcept {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return true;
    return std::strcmp(value, "0") != 0 &&
        std::strcmp(value, "false") != 0 &&
        std::strcmp(value, "False") != 0 &&
        std::strcmp(value, "FALSE") != 0 &&
        std::strcmp(value, "off") != 0 &&
        std::strcmp(value, "Off") != 0 &&
        std::strcmp(value, "OFF") != 0;
}

using CudaGetDeviceCountFn = int (*)(int*);
using CudaSetDeviceFn = int (*)(int);
using CudaMallocFn = int (*)(void**, std::size_t);
using CudaFreeFn = int (*)(void*);
using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
using CudaMemcpyAsyncFn = int (*)(void*, const void*, std::size_t, int, void*);
using CudaMemsetAsyncFn = int (*)(void*, int, std::size_t, void*);
using CudaStreamCreateFn = int (*)(void**);
using CudaStreamDestroyFn = int (*)(void*);
using CudaStreamSynchronizeFn = int (*)(void*);
using CudaStreamWaitEventFn = int (*)(void*, void*, unsigned int);
using CudaEventCreateWithFlagsFn = int (*)(void**, unsigned int);
using CudaEventDestroyFn = int (*)(void*);
using CudaEventRecordFn = int (*)(void*, void*);
using CudaStreamBeginCaptureFn = int (*)(void*, int);
using CudaStreamEndCaptureFn = int (*)(void*, void**);
using CudaGraphInstantiateWithFlagsFn = int (*)(void**, void*, unsigned long long);
using CudaGraphDestroyFn = int (*)(void*);
using CudaGraphExecDestroyFn = int (*)(void*);
using CudaGraphLaunchFn = int (*)(void*, void*);
using CudaGetErrorStringFn = const char* (*)(int);

using AbiVersionFn = int (*)();
using ErrorStringFn = const char* (*)(int);
using PackedValidateFn = int (*)(const NfnNativeTilePackedWeightDescriptorV1*);
using PackedLinearFn = int (*)(
    const NfnNativeTilePackedWeightDescriptorV1*, const float*, const float*,
    float*, std::int64_t, bool);
using QuantizeQ8Fn = int (*)(
    const float*, std::int8_t*, float*, float*, std::int64_t, std::int64_t,
    void*);
using PackedLinearQ8Fn = int (*)(
    const NfnNativeTilePackedWeightDescriptorV1*, const std::int8_t*,
    const float*, const float*, const float*, float*, std::int64_t, bool);
using PackedLinearQ8MultiDecodeFn = int (*)(
    const NfnNativeTilePackedWeightDescriptorV1*,
    const NfnNativeTilePackedWeightDescriptorV1*,
    const NfnNativeTilePackedWeightDescriptorV1*,
    const NfnNativeTilePackedWeightDescriptorV1*, const std::int8_t*,
    const float*, const float*, float*, float*, float*, float*, std::int64_t,
    void*);
using KQuantMmqWorkspaceBytesFn = std::int64_t (*)(
    std::int64_t, std::int64_t);
using KQuantMmqMultiLinearFn = int (*)(
    const NfnNativeTilePackedWeightDescriptorV1* const*, const float*,
    float* const*, std::int64_t, std::int64_t, void*, std::int64_t, void*);
using KQuantMmqMultiLinearGatedFn = int (*)(
    const NfnNativeTilePackedWeightDescriptorV1* const*, const float*,
    const float*, float* const*, std::int64_t, std::int64_t, void*,
    std::int64_t, void*);
using KQuantMmqMultiLinearSwiGluFn = KQuantMmqMultiLinearGatedFn;
using KQuantMmvqMultiLinearPrequantizedFn = int (*)(
    const NfnNativeTilePackedWeightDescriptorV1* const*, float* const*,
    std::int64_t, std::int64_t, void*, std::int64_t, void*);
using KQuantMmvqLinearFn = int (*)(
    const NfnNativeTilePackedWeightDescriptorV1*, const float*, float*, void*,
    std::int64_t, void*);
using ExperimentalMmqAbiFn = std::uint32_t (*)();
using ExperimentalMmqCreateFn = void* (*)(int);
using ExperimentalMmqDestroyFn = void (*)(void*);
using ExperimentalMmqLinearFn = int (*)(
    void*, std::uint32_t, const std::uint8_t*, std::int64_t, std::int64_t,
    const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
using ExperimentalMmqMultiLinearFn = int (*)(
    void*, const std::uint32_t*, const std::uint8_t* const*,
    const std::int64_t*, const std::int64_t*, const float*, float* const*,
    const std::int64_t*, std::int64_t, std::int64_t, std::int64_t, void*);
using ArgmaxRowsFn = int (*)(
    const float*, std::int64_t*, float*, std::int64_t, std::int64_t, void*);
using EmbeddingFn = int (*)(
    const NfnNativeTilePackedWeightDescriptorV1*, std::int64_t, float*);
using EmbeddingDeviceI64Fn = int (*)(
    const NfnNativeTilePackedWeightDescriptorV1*, const std::int64_t*, float*);
using EmbeddingBatchFn = int (*)(
    const NfnNativeTilePackedWeightDescriptorV1*, const std::int32_t*, float*,
    std::int64_t);
using RmsNormFn = int (*)(
    const float*, const NfnNativeTilePackedWeightDescriptorV1*, float*,
    std::int64_t, std::int64_t, float, bool, void*);
using RmsCaptureFn = int (*)(
    const float*, const NfnNativeTilePackedWeightDescriptorV1*, float*, float*,
    std::int64_t, std::int64_t, float, bool, void*);
using RmsCaptureQ8Fn = int (*)(
    const float*, const NfnNativeTilePackedWeightDescriptorV1*, float*, float*,
    std::int8_t*, float*, float*, std::int64_t, std::int64_t, float, bool,
    void*);
using RmsAddFn = int (*)(
    const float*, const NfnNativeTilePackedWeightDescriptorV1*, const float*,
    float*, std::int64_t, std::int64_t, float, bool, void*);
using DualRmsAddCaptureFn = int (*)(
    const float*, const NfnNativeTilePackedWeightDescriptorV1*, const float*,
    float*, const NfnNativeTilePackedWeightDescriptorV1*, float*, float*,
    std::int64_t, std::int64_t, float, bool, float, bool, void*);
using DualRmsAddCaptureMmvqQ8Fn = int (*)(
    const float*, const NfnNativeTilePackedWeightDescriptorV1*, const float*,
    float*, const NfnNativeTilePackedWeightDescriptorV1*, float*, float*,
    std::int64_t, std::int64_t, float, bool, float, bool, void*, std::int64_t,
    void*);
using RopeFn = int (*)(
    float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    float, std::uint32_t, void*);
using RopeBatchFn = int (*)(
    float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    std::int64_t, float, std::uint32_t, bool, void*);
using QkNormScaleRopeFn = int (*)(
    float*, float*, const NfnNativeTilePackedWeightDescriptorV1*,
    const NfnNativeTilePackedWeightDescriptorV1*, std::int64_t, std::int64_t,
    std::int64_t, float, bool, bool, float, std::int64_t, float,
    std::uint32_t, bool, void*);
using QkNormScaleRopeBatchFn = int (*)(
    float*, float*, const NfnNativeTilePackedWeightDescriptorV1*,
    const NfnNativeTilePackedWeightDescriptorV1*, std::int64_t, std::int64_t,
    std::int64_t, std::int64_t, float, bool, bool, float, std::int64_t, float,
    std::uint32_t, bool, void*);
using GqaFn = int (*)(const NfnNativeTileGlimmerGqaDecodeDescriptorV1*);
using FusedDecodeAttentionFn = int (*)(
    const NfnNativeTileGlimmerFusedDecodeAttentionDescriptorV1*);
using FusedDecodeAttentionDevicePositionFn = int (*)(
    const NfnNativeTileGlimmerFusedDecodeAttentionDescriptorV1*,
    const std::int64_t*, std::int64_t);
using CacheCommitFn = int (*)(const NfnNativeTileGlimmerCacheCommitDescriptorV1*);
using CacheCommitRowsFn = int (*)(
    const NfnNativeTileGlimmerCacheCommitDescriptorV1*, std::int64_t);
using CacheCommitLayersFn = int (*)(
    const NfnNativeTileGlimmerCacheCommitLayersDescriptorV1*);
using PackTargetTapsFn = int (*)(
    const float*, float*, std::int64_t, std::int64_t, std::int64_t,
    std::int64_t, std::int64_t, void*);
using DFlashAttentionFn = int (*)(
    const NfnNativeTileDFlashBlockAttentionDescriptorV1*);
using DFlashAttentionShortSplitFn = int (*)(
    const NfnNativeTileDFlashBlockAttentionDescriptorV1*, float*,
    std::int64_t);
using GateFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
using LogitTransformFn = int (*)(float*, std::int64_t, float, float, void*);
using SwiGluFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
using AddFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
using ScaleFn = int (*)(float*, std::int64_t, float, void*);
using VisionPrepareFn = int (*)(
    const NfnNativeTileGlimmerVisionPrepareDescriptorV1*);
using VisionLayerNormFn = int (*)(
    const float*, const float*, const float*, float*, std::int64_t,
    std::int64_t, float, void*);
using VisionAttentionFn = int (*)(
    const NfnNativeTileGlimmerVisionAttentionDescriptorV1*);
using VisionPixelShuffleFn = int (*)(
    const NfnNativeTileGlimmerVisionPixelShuffleDescriptorV1*);
using GeluFn = int (*)(const float*, float*, std::int64_t, void*);

std::size_t checked_size(std::int64_t value, const char* label) {
    if (value < 0 || static_cast<std::uint64_t>(value) >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(std::string("Glimmer CUDA size overflow at ") + label);
    }
    return static_cast<std::size_t>(value);
}

std::int64_t checked_mul(std::int64_t left, std::int64_t right, const char* label) {
    if (left < 0 || right < 0 ||
        (left != 0 && right > std::numeric_limits<std::int64_t>::max() / left)) {
        throw std::runtime_error(std::string("Glimmer CUDA size overflow at ") + label);
    }
    return left * right;
}

class DynamicLibrary final {
public:
    DynamicLibrary() = default;
    ~DynamicLibrary() { close(); }
    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;

    bool try_open(const std::string& path, std::string* error) {
        close();
#if defined(_WIN32)
        handle_ = LoadLibraryA(path.c_str());
        if (handle_ == nullptr) {
            if (error != nullptr) *error = "LoadLibrary failed";
            return false;
        }
#else
        dlerror();
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle_ == nullptr) {
            const char* raw = dlerror();
            if (error != nullptr) *error = raw == nullptr ? "dlopen failed" : raw;
            return false;
        }
#endif
        path_ = path;
        return true;
    }

    template <typename Function>
    Function require(const char* symbol) const {
        if (handle_ == nullptr) {
            throw std::runtime_error("Glimmer CUDA library is not open");
        }
#if defined(_WIN32)
        void* raw = reinterpret_cast<void*>(GetProcAddress(
            static_cast<HMODULE>(handle_), symbol));
#else
        dlerror();
        void* raw = dlsym(handle_, symbol);
        const char* error = dlerror();
        if (error != nullptr) raw = nullptr;
#endif
        if (raw == nullptr) {
            throw std::runtime_error(
                "Glimmer CUDA library " + path_ + " is missing required symbol " + symbol);
        }
        static_assert(sizeof(Function) == sizeof(raw));
        Function result = nullptr;
        std::memcpy(&result, &raw, sizeof(result));
        return result;
    }

    template <typename Function>
    Function optional(const char* symbol) const noexcept {
        if (handle_ == nullptr) return nullptr;
#if defined(_WIN32)
        void* raw = reinterpret_cast<void*>(GetProcAddress(
            static_cast<HMODULE>(handle_), symbol));
#else
        dlerror();
        void* raw = dlsym(handle_, symbol);
        if (dlerror() != nullptr) raw = nullptr;
#endif
        static_assert(sizeof(Function) == sizeof(raw));
        Function result = nullptr;
        std::memcpy(&result, &raw, sizeof(result));
        return result;
    }

    const std::string& path() const noexcept { return path_; }

private:
    void close() noexcept {
        if (handle_ != nullptr) {
#if defined(_WIN32)
            FreeLibrary(static_cast<HMODULE>(handle_));
#else
            dlclose(handle_);
#endif
        }
        handle_ = nullptr;
        path_.clear();
    }
    void* handle_ = nullptr;
    std::string path_;
};

std::string canonical_tile_library(const std::string& raw) {
    if (raw.empty()) {
        throw std::runtime_error("whole-model Glimmer CUDA requires tile_ops_lib");
    }
    const std::filesystem::path requested(raw);
    if (!requested.is_absolute()) {
        throw std::runtime_error("whole-model Glimmer CUDA tile_ops_lib must be absolute");
    }
    std::error_code error;
    const std::filesystem::path resolved = std::filesystem::canonical(requested, error);
    if (error || !std::filesystem::is_regular_file(resolved, error) || error) {
        throw std::runtime_error(
            "whole-model Glimmer CUDA tile_ops_lib is not a regular file: " + raw);
    }
    return resolved.string();
}

std::string canonical_experimental_mmq_library(const std::string& raw) {
    if (raw.empty()) {
        throw std::runtime_error("experimental Glimmer MMQ provider path is empty");
    }
    const std::filesystem::path requested(raw);
    if (!requested.is_absolute()) {
        throw std::runtime_error(
            "experimental Glimmer MMQ provider path must be absolute");
    }
    std::error_code error;
    const std::filesystem::path resolved = std::filesystem::canonical(requested, error);
    if (error || !std::filesystem::is_regular_file(resolved, error) || error) {
        throw std::runtime_error(
            "experimental Glimmer MMQ provider is not a regular file: " + raw);
    }
    return resolved.string();
}

class Runtime final : public std::enable_shared_from_this<Runtime> {
public:
    explicit Runtime(const Config& config)
        : device_(config.cuda_device) {
        if (device_ < 0) {
            throw std::runtime_error("Glimmer CUDA device must be non-negative");
        }
        std::string error;
        if (!tile_.try_open(canonical_tile_library(config.tile_ops_lib), &error)) {
            throw std::runtime_error("could not load Glimmer CUDA Tile ops: " + error);
        }
        base_abi_ = tile_.require<AbiVersionFn>("nfn_native_tile_ops_abi_version");
        strict_abi_ = tile_.require<AbiVersionFn>("nfn_native_tile_strict_math_abi_version");
        packed_abi_ = tile_.require<AbiVersionFn>("nfn_native_tile_packed_weight_abi_version");
        feature_abi_ = tile_.require<AbiVersionFn>(
            "nfn_native_tile_glimmer_inference_abi_version");
        tile_error_ = tile_.require<ErrorStringFn>("nfn_native_tile_ops_error_string");
        packed_validate_ = tile_.require<PackedValidateFn>(
            "nfn_native_tile_packed_weight_validate_v1");
        linear_ = tile_.require<PackedLinearFn>(
            "nfn_native_tile_linear_packed_weight_float32_v1");
        quantize_q8_ = tile_.optional<QuantizeQ8Fn>(
            "nfn_native_tile_quantize_q8_1_float32_v1");
        linear_q8_ = tile_.optional<PackedLinearQ8Fn>(
            "nfn_native_tile_linear_packed_weight_q8_1_float32_v1");
        linear_q8_multi_decode_ = tile_.optional<PackedLinearQ8MultiDecodeFn>(
            "nfn_native_tile_linear_packed_weight_q8_1_multi_decode_float32_v1");
        if ((quantize_q8_ == nullptr) != (linear_q8_ == nullptr)) {
            throw std::runtime_error(
                "Glimmer CUDA sidecar exposes an incomplete Q8 activation fast path");
        }
        k_quant_mmq_abi_ = tile_.optional<AbiVersionFn>(
            "nfn_native_tile_k_quant_mmq_abi_version");
        k_quant_mmq_workspace_bytes_ = tile_.optional<KQuantMmqWorkspaceBytesFn>(
            "nfn_native_tile_k_quant_mmq_workspace_bytes_v1");
        k_quant_mmq_multi_linear_ = tile_.optional<KQuantMmqMultiLinearFn>(
            "nfn_native_tile_k_quant_mmq_multi_linear_float32_v1");
        k_quant_mmq_multi_linear_gated_ =
            tile_.optional<KQuantMmqMultiLinearGatedFn>(
                "nfn_native_tile_k_quant_mmq_multi_linear_gated_float32_v1");
        k_quant_mmq_multi_linear_swiglu_ =
            tile_.optional<KQuantMmqMultiLinearSwiGluFn>(
                "nfn_native_tile_k_quant_mmq_multi_linear_swiglu_float32_v1");
        k_quant_mmvq_multi_linear_ = tile_.optional<KQuantMmqMultiLinearFn>(
            "nfn_native_tile_k_quant_mmvq_multi_linear_float32_v1");
        k_quant_mmvq_multi_linear_prequantized_ =
            tile_.optional<KQuantMmvqMultiLinearPrequantizedFn>(
                "nfn_native_tile_k_quant_mmvq_multi_linear_prequantized_float32_v1");
        k_quant_mmvq_multi_linear_gated_ =
            tile_.optional<KQuantMmqMultiLinearGatedFn>(
                "nfn_native_tile_k_quant_mmvq_multi_linear_gated_float32_v1");
        k_quant_mmvq_multi_linear_swiglu_ =
            tile_.optional<KQuantMmqMultiLinearSwiGluFn>(
                "nfn_native_tile_k_quant_mmvq_multi_linear_swiglu_float32_v1");
        k_quant_mmvq_linear_ = tile_.optional<KQuantMmvqLinearFn>(
            "nfn_native_tile_k_quant_mmvq_linear_float32_v1");
        const bool has_any_k_quant_mmq = k_quant_mmq_abi_ != nullptr ||
            k_quant_mmq_workspace_bytes_ != nullptr ||
            k_quant_mmq_multi_linear_ != nullptr;
        if (has_any_k_quant_mmq &&
            (k_quant_mmq_abi_ == nullptr ||
             k_quant_mmq_workspace_bytes_ == nullptr ||
             k_quant_mmq_multi_linear_ == nullptr)) {
            throw std::runtime_error(
                "Glimmer CUDA sidecar exposes an incomplete exact K-quant MMQ ABI");
        }
        if (k_quant_mmq_multi_linear_gated_ != nullptr && !has_any_k_quant_mmq) {
            throw std::runtime_error(
                "Glimmer CUDA sidecar exposes gated MMQ without the base MMQ ABI");
        }
        if (k_quant_mmq_multi_linear_swiglu_ != nullptr && !has_any_k_quant_mmq) {
            throw std::runtime_error(
                "Glimmer CUDA sidecar exposes SwiGLU MMQ without the base MMQ ABI");
        }
        if (k_quant_mmvq_linear_ != nullptr && !has_any_k_quant_mmq) {
            throw std::runtime_error(
                "Glimmer CUDA sidecar exposes MMVQ without the base MMQ ABI");
        }
        const bool has_any_batched_mmvq =
            k_quant_mmvq_multi_linear_ != nullptr ||
            k_quant_mmvq_multi_linear_gated_ != nullptr ||
            k_quant_mmvq_multi_linear_swiglu_ != nullptr;
        if (has_any_batched_mmvq &&
            (k_quant_mmvq_multi_linear_ == nullptr ||
             k_quant_mmvq_multi_linear_gated_ == nullptr ||
             k_quant_mmvq_multi_linear_swiglu_ == nullptr ||
             !has_any_k_quant_mmq)) {
            throw std::runtime_error(
                "Glimmer CUDA sidecar exposes an incomplete batched MMVQ verifier ABI");
        }
        if (k_quant_mmvq_multi_linear_prequantized_ != nullptr &&
            !has_any_batched_mmvq) {
            throw std::runtime_error(
                "Glimmer CUDA sidecar exposes a prequantized MMVQ handoff without MMVQ");
        }
        if (has_any_k_quant_mmq &&
            k_quant_mmq_abi_() != NFN_NATIVE_TILE_K_QUANT_MMQ_V1) {
            throw std::runtime_error(
                "Glimmer CUDA sidecar exposes an unsupported exact K-quant MMQ ABI");
        }
        argmax_rows_ = tile_.optional<ArgmaxRowsFn>(
            "nfn_native_tile_argmax_rows_float32_v1");
        embedding_ = tile_.require<EmbeddingFn>(
            "nfn_native_tile_glimmer_embedding_gather_float32_v1");
        embedding_device_i64_ = tile_.optional<EmbeddingDeviceI64Fn>(
            "nfn_native_tile_glimmer_embedding_gather_device_i64_float32_v1");
        embedding_batch_ = tile_.require<EmbeddingBatchFn>(
            "nfn_native_tile_glimmer_embedding_batch_i32_float32_v1");
        rms_ = tile_.require<RmsNormFn>(
            "nfn_native_tile_glimmer_rms_norm_affine_float32_v1");
        rms_capture_ = tile_.optional<RmsCaptureFn>(
            "nfn_native_tile_glimmer_rms_norm_affine_capture_residual_float32_v1");
        rms_capture_q8_ = tile_.optional<RmsCaptureQ8Fn>(
            "nfn_native_tile_glimmer_rms_norm_affine_capture_residual_q8_1_float32_v1");
        rms_add_ = tile_.optional<RmsAddFn>(
            "nfn_native_tile_glimmer_rms_norm_affine_add_residual_float32_v1");
        dual_rms_add_capture_ = tile_.optional<DualRmsAddCaptureFn>(
            "nfn_native_tile_glimmer_dual_rms_add_capture_float32_v1");
        dual_rms_add_capture_cooperative_batch_ =
            tile_.optional<DualRmsAddCaptureFn>(
                "nfn_native_tile_glimmer_dual_rms_add_capture_cooperative_batch_float32_v1");
        dual_rms_add_capture_mmvq_q8_ =
            tile_.optional<DualRmsAddCaptureMmvqQ8Fn>(
                "nfn_native_tile_glimmer_dual_rms_add_capture_mmvq_q8_float32_v1");
        if ((rms_capture_ == nullptr) != (rms_add_ == nullptr)) {
            throw std::runtime_error(
                "Glimmer CUDA sidecar exposes an incomplete fused residual-norm path");
        }
        if ((dual_rms_add_capture_mmvq_q8_ == nullptr) !=
            (k_quant_mmvq_multi_linear_prequantized_ == nullptr)) {
            throw std::runtime_error(
                "Glimmer CUDA sidecar exposes an incomplete RMS-to-MMVQ handoff");
        }
        rope_ = tile_.require<RopeFn>(
            "nfn_native_tile_glimmer_positioned_rope_float32_v1");
        rope_batch_ = tile_.require<RopeBatchFn>(
            "nfn_native_tile_glimmer_positioned_rope_batch_float32_v1");
        qk_norm_scale_rope_ = tile_.optional<QkNormScaleRopeFn>(
            "nfn_native_tile_glimmer_qk_norm_scale_rope_float32_v1");
        qk_norm_scale_rope_batch_ = tile_.optional<QkNormScaleRopeBatchFn>(
            "nfn_native_tile_glimmer_qk_norm_scale_rope_batch_float32_v1");
        gqa_ = tile_.require<GqaFn>("nfn_native_tile_glimmer_gqa_decode_float32_v1");
        fused_decode_attention_ = tile_.optional<FusedDecodeAttentionFn>(
            "nfn_native_tile_glimmer_fused_decode_attention_float32_v1");
        fused_decode_attention_device_position_ =
            tile_.optional<FusedDecodeAttentionDevicePositionFn>(
                "nfn_native_tile_glimmer_fused_decode_attention_device_position_float32_v1");
        cache_commit_ = tile_.require<CacheCommitFn>(
            "nfn_native_tile_glimmer_cache_commit_bf16_v1");
        cache_commit_rows_ = tile_.optional<CacheCommitRowsFn>(
            "nfn_native_tile_glimmer_cache_commit_rows_bf16_v1");
        cache_commit_layers_ = tile_.optional<CacheCommitLayersFn>(
            "nfn_native_tile_glimmer_cache_commit_layers_bf16_v1");
        pack_target_taps_ = tile_.optional<PackTargetTapsFn>(
            "nfn_native_tile_glimmer_pack_target_taps_float32_v1");
        dflash_attention_ = tile_.require<DFlashAttentionFn>(
            "nfn_native_tile_dflash_block_attention_float32_v1");
        dflash_attention_short_split_ =
            tile_.optional<DFlashAttentionShortSplitFn>(
                "nfn_native_tile_dflash_block_attention_short_split_float32_v1");
        gate_ = tile_.require<GateFn>(
            "nfn_native_tile_glimmer_sigmoid_gate_float32_v1");
        logit_transform_ = tile_.require<LogitTransformFn>(
            "nfn_native_tile_glimmer_logit_transform_float32_v1");
        swiglu_ = tile_.require<SwiGluFn>("nfn_native_tile_swiglu_float32");
        add_ = tile_.require<AddFn>("nfn_native_tile_add_float32");
        scale_ = tile_.require<ScaleFn>("nfn_native_tile_scale_inplace_float32");
        if (base_abi_() != 1 || strict_abi_() != 1 ||
            packed_abi_() != NFN_NATIVE_TILE_PACKED_WEIGHT_V1 ||
            feature_abi_() != NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1) {
            throw std::runtime_error(
                "whole-model Glimmer CUDA requires base/strict/packed/feature ABI version 1");
        }

        const std::vector<std::string> candidates = config.cuda_runtime_lib.empty()
            ? std::vector<std::string>{
                  "libcudart.so.13", "libcudart.so", "libcudart.so.12",
                  "/usr/local/cuda/lib64/libcudart.so"}
            : std::vector<std::string>{config.cuda_runtime_lib};
        for (const std::string& candidate : candidates) {
            if (cuda_.try_open(candidate, &error)) break;
        }
        if (cuda_.path().empty()) {
            throw std::runtime_error("could not load CUDA runtime for Glimmer: " + error);
        }
        get_device_count_ = cuda_.require<CudaGetDeviceCountFn>("cudaGetDeviceCount");
        set_device_ = cuda_.require<CudaSetDeviceFn>("cudaSetDevice");
        malloc_ = cuda_.require<CudaMallocFn>("cudaMalloc");
        free_ = cuda_.require<CudaFreeFn>("cudaFree");
        memcpy_ = cuda_.require<CudaMemcpyFn>("cudaMemcpy");
        memcpy_async_ = cuda_.require<CudaMemcpyAsyncFn>("cudaMemcpyAsync");
        memset_async_ = cuda_.require<CudaMemsetAsyncFn>("cudaMemsetAsync");
        stream_create_ = cuda_.require<CudaStreamCreateFn>("cudaStreamCreate");
        stream_destroy_ = cuda_.require<CudaStreamDestroyFn>("cudaStreamDestroy");
        stream_sync_ = cuda_.require<CudaStreamSynchronizeFn>("cudaStreamSynchronize");
        stream_wait_event_ = cuda_.require<CudaStreamWaitEventFn>(
            "cudaStreamWaitEvent");
        event_create_with_flags_ = cuda_.require<CudaEventCreateWithFlagsFn>(
            "cudaEventCreateWithFlags");
        event_destroy_ = cuda_.require<CudaEventDestroyFn>("cudaEventDestroy");
        event_record_ = cuda_.require<CudaEventRecordFn>("cudaEventRecord");
        stream_begin_capture_ = cuda_.optional<CudaStreamBeginCaptureFn>(
            "cudaStreamBeginCapture");
        stream_end_capture_ = cuda_.optional<CudaStreamEndCaptureFn>(
            "cudaStreamEndCapture");
        graph_instantiate_with_flags_ =
            cuda_.optional<CudaGraphInstantiateWithFlagsFn>(
                "cudaGraphInstantiateWithFlags");
        graph_destroy_ = cuda_.optional<CudaGraphDestroyFn>("cudaGraphDestroy");
        graph_exec_destroy_ = cuda_.optional<CudaGraphExecDestroyFn>(
            "cudaGraphExecDestroy");
        graph_launch_ = cuda_.optional<CudaGraphLaunchFn>("cudaGraphLaunch");
        error_string_ = cuda_.require<CudaGetErrorStringFn>("cudaGetErrorString");
        int count = 0;
        check_cuda(get_device_count_(&count), "cudaGetDeviceCount");
        if (device_ >= count) {
            throw std::runtime_error("requested Glimmer CUDA device is unavailable");
        }
        set_device();
        check_cuda(stream_create_(&stream_), "cudaStreamCreate");
        verifier_projection_overlap_enabled_ = feature_enabled_default_on(
            "NFN_GLIMMER_VERIFIER_PROJECTION_OVERLAP");
        short_attention_split_enabled_ = feature_enabled_default_on(
            "NFN_GLIMMER_SHORT_ATTENTION_SPLIT");
        cooperative_batch_rms_enabled_ = feature_enabled_default_on(
            "NFN_GLIMMER_COOPERATIVE_BATCH_RMS");
        if (verifier_projection_overlap_enabled_) {
            try {
                for (void*& auxiliary_stream : verifier_aux_streams_) {
                    check_cuda(
                        stream_create_(&auxiliary_stream),
                        "cudaStreamCreate verifier auxiliary");
                }
                // cudaEventDisableTiming: these events are stream-ordering
                // primitives, never timing sources.
                constexpr unsigned int disable_timing = 2U;
                check_cuda(event_create_with_flags_(
                    &verifier_ready_event_, disable_timing),
                    "cudaEventCreate verifier ready");
                for (void*& done_event : verifier_done_events_) {
                    check_cuda(event_create_with_flags_(
                        &done_event, disable_timing),
                        "cudaEventCreate verifier done");
                }
            } catch (...) {
                for (void*& done_event : verifier_done_events_) {
                    if (done_event != nullptr) (void)event_destroy_(done_event);
                    done_event = nullptr;
                }
                if (verifier_ready_event_ != nullptr) {
                    (void)event_destroy_(verifier_ready_event_);
                    verifier_ready_event_ = nullptr;
                }
                for (void*& auxiliary_stream : verifier_aux_streams_) {
                    if (auxiliary_stream != nullptr) {
                        (void)stream_destroy_(auxiliary_stream);
                    }
                    auxiliary_stream = nullptr;
                }
                throw;
            }
        }

        // Benchmark-only escape hatch used to quantify a pinned independent
        // MMQ implementation on the exact resident device pointers.  It is
        // deliberately absent from public configuration and capability bits:
        // shipped support must land behind NeuralFn's reviewed raw Tile ABI,
        // not this upstream-private C++ provider boundary.
        const char* experimental_provider =
            std::getenv("NFN_GLIMMER_EXPERIMENTAL_MMQ_PROVIDER");
        if (experimental_provider != nullptr && *experimental_provider != '\0') {
            const std::string provider_path =
                canonical_experimental_mmq_library(experimental_provider);
            if (!experimental_mmq_.try_open(provider_path, &error)) {
                throw std::runtime_error(
                    "could not load experimental Glimmer MMQ provider: " + error);
            }
            experimental_mmq_abi_ = experimental_mmq_.require<ExperimentalMmqAbiFn>(
                "nfn_experimental_llama_mmq_provider_abi");
            experimental_mmq_create_ =
                experimental_mmq_.require<ExperimentalMmqCreateFn>(
                    "nfn_experimental_llama_mmq_provider_create");
            experimental_mmq_destroy_ =
                experimental_mmq_.require<ExperimentalMmqDestroyFn>(
                    "nfn_experimental_llama_mmq_provider_destroy");
            experimental_mmq_linear_ =
                experimental_mmq_.require<ExperimentalMmqLinearFn>(
                    "nfn_experimental_llama_mmq_provider_linear_f32");
            experimental_mmq_multi_linear_ =
                experimental_mmq_.optional<ExperimentalMmqMultiLinearFn>(
                    "nfn_experimental_llama_mmq_provider_multi_linear_f32");
            if (experimental_mmq_abi_() != 1) {
                throw std::runtime_error(
                    "experimental Glimmer MMQ provider ABI is unsupported");
            }
            experimental_mmq_context_ = experimental_mmq_create_(device_);
            if (experimental_mmq_context_ == nullptr) {
                throw std::runtime_error(
                    "experimental Glimmer MMQ provider initialization failed");
            }
        }
    }

    ~Runtime() {
        set_device_noexcept();
        if (experimental_mmq_context_ != nullptr) {
            (void)stream_sync_(stream_);
            experimental_mmq_destroy_(experimental_mmq_context_);
            experimental_mmq_context_ = nullptr;
        }
        for (void* auxiliary_stream : verifier_aux_streams_) {
            if (auxiliary_stream != nullptr) (void)stream_sync_(auxiliary_stream);
        }
        for (void*& done_event : verifier_done_events_) {
            if (done_event != nullptr) (void)event_destroy_(done_event);
            done_event = nullptr;
        }
        if (verifier_ready_event_ != nullptr) {
            (void)event_destroy_(verifier_ready_event_);
            verifier_ready_event_ = nullptr;
        }
        for (void*& auxiliary_stream : verifier_aux_streams_) {
            if (auxiliary_stream != nullptr) (void)stream_destroy_(auxiliary_stream);
            auxiliary_stream = nullptr;
        }
        if (stream_ != nullptr) stream_destroy_(stream_);
    }

    void set_device() const { check_cuda(set_device_(device_), "cudaSetDevice"); }
    void set_device_noexcept() const noexcept { (void)set_device_(device_); }
    void* stream() const noexcept { return stream_; }
    bool verifier_projection_overlap_enabled() const noexcept {
        return verifier_projection_overlap_enabled_;
    }
    bool short_attention_split_enabled() const noexcept {
        return short_attention_split_enabled_ &&
            dflash_attention_short_split_ != nullptr;
    }
    bool cooperative_batch_rms_enabled() const noexcept {
        return cooperative_batch_rms_enabled_ &&
            dual_rms_add_capture_cooperative_batch_ != nullptr;
    }
    void* verifier_aux_stream(std::size_t index) const {
        if (!verifier_projection_overlap_enabled_ ||
            index >= verifier_aux_streams_.size() ||
            verifier_aux_streams_[index] == nullptr) {
            throw std::runtime_error(
                "Glimmer verifier auxiliary CUDA stream is unavailable");
        }
        return verifier_aux_streams_[index];
    }
    void begin_verifier_projection_overlap() const {
        if (!verifier_projection_overlap_enabled_) return;
        check_cuda(
            event_record_(verifier_ready_event_, stream_),
            "cudaEventRecord verifier ready");
        for (void* auxiliary_stream : verifier_aux_streams_) {
            check_cuda(stream_wait_event_(
                auxiliary_stream, verifier_ready_event_, 0),
                "cudaStreamWaitEvent verifier ready");
        }
    }
    void end_verifier_projection_overlap() const {
        if (!verifier_projection_overlap_enabled_) return;
        for (std::size_t index = 0; index < verifier_aux_streams_.size(); ++index) {
            check_cuda(event_record_(
                verifier_done_events_[index], verifier_aux_streams_[index]),
                "cudaEventRecord verifier done");
            check_cuda(stream_wait_event_(
                stream_, verifier_done_events_[index], 0),
                "cudaStreamWaitEvent verifier done");
        }
    }
    int device() const noexcept { return device_; }
    const std::string& tile_path() const noexcept { return tile_.path(); }
    const std::string& cuda_path() const noexcept { return cuda_.path(); }

    void* allocate(std::size_t bytes) const {
        set_device();
        void* result = nullptr;
        check_cuda(malloc_(&result, std::max<std::size_t>(bytes, 1)), "cudaMalloc");
        if (result == nullptr) throw std::runtime_error("cudaMalloc returned null");
        return result;
    }
    void free_noexcept(void* pointer) const noexcept {
        if (pointer == nullptr) return;
        set_device_noexcept();
        (void)free_(pointer);
    }
    void copy_h2d_async(void* target, const void* source, std::size_t bytes) const {
        if (bytes == 0) return;
        check_cuda(memcpy_async_(target, source, bytes, kCudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync H2D");
    }
    void copy_d2d_async(void* target, const void* source, std::size_t bytes) const {
        if (bytes == 0) return;
        check_cuda(memcpy_async_(target, source, bytes, kCudaMemcpyDeviceToDevice, stream_),
                   "cudaMemcpyAsync D2D");
    }
    void copy_d2h(void* target, const void* source, std::size_t bytes) const {
        if (bytes == 0) return;
        check_cuda(memcpy_(target, source, bytes, kCudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    }
    void zero_async(void* target, std::size_t bytes) const {
        check_cuda(memset_async_(target, 0, bytes, stream_), "cudaMemsetAsync");
    }
    void synchronize() const { check_cuda(stream_sync_(stream_), "cudaStreamSynchronize"); }
    bool has_decode_graphs() const noexcept {
        const char* disabled = std::getenv("NFN_GLIMMER_CUDA_GRAPHS");
        const bool explicitly_disabled = disabled != nullptr &&
            (std::strcmp(disabled, "0") == 0 ||
             std::strcmp(disabled, "false") == 0 ||
             std::strcmp(disabled, "off") == 0);
        return !explicitly_disabled && argmax_rows_ != nullptr &&
            embedding_device_i64_ != nullptr &&
            fused_decode_attention_device_position_ != nullptr &&
            stream_begin_capture_ != nullptr && stream_end_capture_ != nullptr &&
            graph_instantiate_with_flags_ != nullptr && graph_destroy_ != nullptr &&
            graph_exec_destroy_ != nullptr && graph_launch_ != nullptr;
    }
    void begin_decode_graph_capture() const {
        if (!has_decode_graphs()) {
            throw std::runtime_error("Glimmer CUDA decode graphs are unavailable");
        }
        // Thread-local capture keeps unrelated CUDA work in another host
        // thread from invalidating a model-local decode capture.
        check_cuda(stream_begin_capture_(stream_, 1), "cudaStreamBeginCapture");
    }
    void* end_decode_graph_capture() const {
        void* graph = nullptr;
        check_cuda(stream_end_capture_(stream_, &graph), "cudaStreamEndCapture");
        if (graph == nullptr) {
            throw std::runtime_error("Glimmer CUDA decode graph capture returned null");
        }
        void* graph_exec = nullptr;
        const int instantiate_status =
            graph_instantiate_with_flags_(&graph_exec, graph, 0);
        const int destroy_status = graph_destroy_(graph);
        check_cuda(instantiate_status, "cudaGraphInstantiateWithFlags");
        if (destroy_status != kCudaSuccess) {
            if (graph_exec != nullptr) (void)graph_exec_destroy_(graph_exec);
            check_cuda(destroy_status, "cudaGraphDestroy");
        }
        if (graph_exec == nullptr) {
            throw std::runtime_error("Glimmer CUDA decode graph executable is null");
        }
        return graph_exec;
    }
    void abort_decode_graph_capture_noexcept() const noexcept {
        if (stream_end_capture_ == nullptr) return;
        void* graph = nullptr;
        if (stream_end_capture_(stream_, &graph) == kCudaSuccess &&
            graph != nullptr && graph_destroy_ != nullptr) {
            (void)graph_destroy_(graph);
        }
    }
    void launch_decode_graph(void* graph_exec) const {
        if (graph_exec == nullptr || graph_launch_ == nullptr) {
            throw std::runtime_error("Glimmer CUDA decode graph is invalid");
        }
        check_cuda(graph_launch_(graph_exec, stream_), "cudaGraphLaunch");
    }
    void destroy_decode_graph_noexcept(void* graph_exec) const noexcept {
        if (graph_exec == nullptr || graph_exec_destroy_ == nullptr) return;
        set_device_noexcept();
        (void)graph_exec_destroy_(graph_exec);
    }
    void check_tile(int status, const char* operation) const {
        if (status == kCudaSuccess) return;
        const char* message = tile_error_(status);
        throw std::runtime_error(
            std::string(operation) + " failed: " +
            (message == nullptr ? "unknown Tile-CUDA error" : message));
    }

    PackedValidateFn packed_validate() const noexcept { return packed_validate_; }
    PackedLinearFn linear() const noexcept { return linear_; }
    bool has_q8_linear() const noexcept {
        return quantize_q8_ != nullptr && linear_q8_ != nullptr;
    }
    QuantizeQ8Fn quantize_q8() const noexcept { return quantize_q8_; }
    PackedLinearQ8Fn linear_q8() const noexcept { return linear_q8_; }
    PackedLinearQ8MultiDecodeFn linear_q8_multi_decode() const noexcept {
        return linear_q8_multi_decode_;
    }
    bool has_k_quant_mmq() const noexcept {
        return k_quant_mmq_abi_ != nullptr &&
            k_quant_mmq_workspace_bytes_ != nullptr &&
            k_quant_mmq_multi_linear_ != nullptr;
    }
    std::int64_t k_quant_mmq_workspace_bytes(
        std::int64_t rows,
        std::int64_t input_dim) const {
        if (!has_k_quant_mmq()) return 0;
        set_device();
        return k_quant_mmq_workspace_bytes_(rows, input_dim);
    }
    int k_quant_mmq_multi_linear(
        const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
        const float* input,
        float* const* outputs,
        std::int64_t operation_count,
        std::int64_t rows,
        void* workspace,
        std::int64_t workspace_nbytes) const {
        if (!has_k_quant_mmq()) return 1;
        return k_quant_mmq_multi_linear_(
            descriptors, input, outputs, operation_count, rows, workspace,
            workspace_nbytes, stream_);
    }
    bool has_k_quant_mmq_gated() const noexcept {
        return has_k_quant_mmq() && k_quant_mmq_multi_linear_gated_ != nullptr;
    }
    int k_quant_mmq_multi_linear_gated(
        const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
        const float* input,
        const float* gate,
        float* const* outputs,
        std::int64_t operation_count,
        std::int64_t rows,
        void* workspace,
        std::int64_t workspace_nbytes) const {
        if (!has_k_quant_mmq_gated()) return 1;
        return k_quant_mmq_multi_linear_gated_(
            descriptors, input, gate, outputs, operation_count, rows,
            workspace, workspace_nbytes, stream_);
    }
    bool has_k_quant_mmq_swiglu() const noexcept {
        return has_k_quant_mmq() && k_quant_mmq_multi_linear_swiglu_ != nullptr;
    }
    bool has_k_quant_mmvq() const noexcept {
        return has_k_quant_mmq() && k_quant_mmvq_linear_ != nullptr;
    }
    bool has_k_quant_mmvq_rows() const noexcept {
        return has_k_quant_mmq() && k_quant_mmvq_multi_linear_ != nullptr &&
            k_quant_mmvq_multi_linear_gated_ != nullptr &&
            k_quant_mmvq_multi_linear_swiglu_ != nullptr;
    }
    bool has_k_quant_mmvq_prequantized() const noexcept {
        return has_k_quant_mmvq_rows() &&
            k_quant_mmvq_multi_linear_prequantized_ != nullptr;
    }
    int k_quant_mmvq_multi_linear(
        const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
        const float* input,
        float* const* outputs,
        std::int64_t operation_count,
        std::int64_t rows,
        void* workspace,
        std::int64_t workspace_nbytes) const {
        if (!has_k_quant_mmvq_rows()) return 1;
        return k_quant_mmvq_multi_linear_(
            descriptors, input, outputs, operation_count, rows, workspace,
            workspace_nbytes, stream_);
    }
    bool has_rms_to_mmvq_handoff() const noexcept {
        return has_k_quant_mmvq_prequantized() &&
            dual_rms_add_capture_mmvq_q8_ != nullptr;
    }
    int k_quant_mmvq_multi_linear_prequantized(
        const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
        float* const* outputs,
        std::int64_t operation_count,
        std::int64_t rows,
        void* workspace,
        std::int64_t workspace_nbytes,
        void* cuda_stream = nullptr) const {
        if (!has_k_quant_mmvq_prequantized()) return 1;
        return k_quant_mmvq_multi_linear_prequantized_(
            descriptors, outputs, operation_count, rows, workspace,
            workspace_nbytes, cuda_stream == nullptr ? stream_ : cuda_stream);
    }
    int k_quant_mmvq_multi_linear_gated(
        const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
        const float* input,
        const float* gate,
        float* const* outputs,
        std::int64_t operation_count,
        std::int64_t rows,
        void* workspace,
        std::int64_t workspace_nbytes) const {
        if (!has_k_quant_mmvq_rows()) return 1;
        return k_quant_mmvq_multi_linear_gated_(
            descriptors, input, gate, outputs, operation_count, rows,
            workspace, workspace_nbytes, stream_);
    }
    int k_quant_mmvq_multi_linear_swiglu(
        const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
        const float* gate,
        const float* up,
        float* const* outputs,
        std::int64_t operation_count,
        std::int64_t rows,
        void* workspace,
        std::int64_t workspace_nbytes) const {
        if (!has_k_quant_mmvq_rows()) return 1;
        return k_quant_mmvq_multi_linear_swiglu_(
            descriptors, gate, up, outputs, operation_count, rows, workspace,
            workspace_nbytes, stream_);
    }
    int k_quant_mmvq_linear(
        const NfnNativeTilePackedWeightDescriptorV1* descriptor,
        const float* input,
        float* output,
        void* workspace,
        std::int64_t workspace_nbytes) const {
        if (!has_k_quant_mmvq()) return 1;
        return k_quant_mmvq_linear_(
            descriptor, input, output, workspace, workspace_nbytes, stream_);
    }
    int k_quant_mmq_multi_linear_swiglu(
        const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
        const float* gate,
        const float* up,
        float* const* outputs,
        std::int64_t operation_count,
        std::int64_t rows,
        void* workspace,
        std::int64_t workspace_nbytes) const {
        if (!has_k_quant_mmq_swiglu()) return 1;
        return k_quant_mmq_multi_linear_swiglu_(
            descriptors, gate, up, outputs, operation_count, rows, workspace,
            workspace_nbytes, stream_);
    }
    bool has_experimental_mmq() const noexcept {
        return experimental_mmq_context_ != nullptr &&
            experimental_mmq_linear_ != nullptr;
    }
    bool has_experimental_mmq_multi() const noexcept {
        return has_experimental_mmq() && experimental_mmq_multi_linear_ != nullptr;
    }
    int experimental_mmq_linear(
        const NfnNativeTilePackedWeightDescriptorV1* descriptor,
        const float* input,
        float* output,
        std::int64_t rows) const {
        if (!has_experimental_mmq() || descriptor == nullptr) {
            return 1;
        }
        return experimental_mmq_linear_(
            experimental_mmq_context_, descriptor->encoding, descriptor->data,
            descriptor->data_nbytes, descriptor->row_stride_bytes, input,
            output, rows, descriptor->input_dim, descriptor->output_dim,
            stream_);
    }
    int experimental_mmq_multi_linear(
        const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
        const float* input,
        float* const* outputs,
        std::int64_t operation_count,
        std::int64_t rows) const {
        if (!has_experimental_mmq_multi() || descriptors == nullptr ||
            outputs == nullptr || input == nullptr || operation_count <= 0 ||
            operation_count > 4 || descriptors[0] == nullptr) {
            return 1;
        }
        std::array<std::uint32_t, 4> encodings{};
        std::array<const std::uint8_t*, 4> weights{};
        std::array<std::int64_t, 4> nbytes{};
        std::array<std::int64_t, 4> strides{};
        std::array<std::int64_t, 4> output_dims{};
        const std::int64_t input_dim = descriptors[0]->input_dim;
        for (std::int64_t index = 0; index < operation_count; ++index) {
            const auto* descriptor = descriptors[index];
            if (descriptor == nullptr || descriptor->input_dim != input_dim ||
                outputs[index] == nullptr) {
                return 1;
            }
            encodings[static_cast<std::size_t>(index)] = descriptor->encoding;
            weights[static_cast<std::size_t>(index)] = descriptor->data;
            nbytes[static_cast<std::size_t>(index)] = descriptor->data_nbytes;
            strides[static_cast<std::size_t>(index)] = descriptor->row_stride_bytes;
            output_dims[static_cast<std::size_t>(index)] = descriptor->output_dim;
        }
        return experimental_mmq_multi_linear_(
            experimental_mmq_context_, encodings.data(), weights.data(),
            nbytes.data(), strides.data(), input, outputs, output_dims.data(),
            operation_count, rows, input_dim, stream_);
    }
    ArgmaxRowsFn argmax_rows() const noexcept { return argmax_rows_; }
    EmbeddingFn embedding() const noexcept { return embedding_; }
    EmbeddingDeviceI64Fn embedding_device_i64() const noexcept {
        return embedding_device_i64_;
    }
    EmbeddingBatchFn embedding_batch() const noexcept { return embedding_batch_; }
    RmsNormFn rms() const noexcept { return rms_; }
    bool has_fused_residual_norm() const noexcept {
        return rms_capture_ != nullptr && rms_add_ != nullptr;
    }
    RmsCaptureFn rms_capture() const noexcept { return rms_capture_; }
    RmsCaptureQ8Fn rms_capture_q8() const noexcept { return rms_capture_q8_; }
    RmsAddFn rms_add() const noexcept { return rms_add_; }
    DualRmsAddCaptureFn dual_rms_add_capture() const noexcept {
        return dual_rms_add_capture_;
    }
    DualRmsAddCaptureFn dual_rms_add_capture_cooperative_batch() const noexcept {
        return dual_rms_add_capture_cooperative_batch_;
    }
    DualRmsAddCaptureMmvqQ8Fn dual_rms_add_capture_mmvq_q8() const noexcept {
        return dual_rms_add_capture_mmvq_q8_;
    }
    RopeFn rope() const noexcept { return rope_; }
    RopeBatchFn rope_batch() const noexcept { return rope_batch_; }
    QkNormScaleRopeFn qk_norm_scale_rope() const noexcept {
        return qk_norm_scale_rope_;
    }
    QkNormScaleRopeBatchFn qk_norm_scale_rope_batch() const noexcept {
        return qk_norm_scale_rope_batch_;
    }
    GqaFn gqa() const noexcept { return gqa_; }
    FusedDecodeAttentionFn fused_decode_attention() const noexcept {
        return fused_decode_attention_;
    }
    FusedDecodeAttentionDevicePositionFn
    fused_decode_attention_device_position() const noexcept {
        return fused_decode_attention_device_position_;
    }
    CacheCommitFn cache_commit() const noexcept { return cache_commit_; }
    CacheCommitRowsFn cache_commit_rows() const noexcept {
        return cache_commit_rows_;
    }
    CacheCommitLayersFn cache_commit_layers() const noexcept {
        return cache_commit_layers_;
    }
    PackTargetTapsFn pack_target_taps() const noexcept {
        return pack_target_taps_;
    }
    DFlashAttentionFn dflash_attention() const noexcept { return dflash_attention_; }
    DFlashAttentionShortSplitFn dflash_attention_short_split() const noexcept {
        return dflash_attention_short_split_;
    }
    GateFn gate() const noexcept { return gate_; }
    LogitTransformFn logit_transform() const noexcept { return logit_transform_; }
    SwiGluFn swiglu() const noexcept { return swiglu_; }
    AddFn add() const noexcept { return add_; }
    ScaleFn scale() const noexcept { return scale_; }

    void enable_vision() {
        std::lock_guard<std::mutex> lock(vision_mutex_);
        if (vision_prepare_ != nullptr) return;
        const auto vision_abi = tile_.require<AbiVersionFn>(
            "nfn_native_tile_glimmer_vision_abi_version");
        if (vision_abi() != NFN_NATIVE_TILE_GLIMMER_VISION_V1) {
            throw std::runtime_error(
                "whole-model Glimmer CUDA vision requires feature ABI version 1");
        }
        vision_prepare_ = tile_.require<VisionPrepareFn>(
            "nfn_native_tile_glimmer_vision_prepare_float32_v1");
        vision_layer_norm_ = tile_.require<VisionLayerNormFn>(
            "nfn_native_tile_glimmer_vision_layer_norm_float32_v1");
        vision_attention_ = tile_.require<VisionAttentionFn>(
            "nfn_native_tile_glimmer_vision_attention_float32_v1");
        vision_pixel_shuffle_ = tile_.require<VisionPixelShuffleFn>(
            "nfn_native_tile_glimmer_vision_pixel_shuffle_float32_v1");
        gelu_ = tile_.require<GeluFn>("nfn_native_tile_gelu_float32");
    }

    VisionPrepareFn vision_prepare() const noexcept { return vision_prepare_; }
    VisionLayerNormFn vision_layer_norm() const noexcept { return vision_layer_norm_; }
    VisionAttentionFn vision_attention() const noexcept { return vision_attention_; }
    VisionPixelShuffleFn vision_pixel_shuffle() const noexcept {
        return vision_pixel_shuffle_;
    }
    GeluFn gelu() const noexcept { return gelu_; }

private:
    void check_cuda(int status, const char* operation) const {
        if (status == kCudaSuccess) return;
        const char* message = error_string_ == nullptr ? nullptr : error_string_(status);
        throw std::runtime_error(
            std::string(operation) + " failed: " +
            (message == nullptr ? "unknown CUDA error" : message));
    }

    int device_ = 0;
    DynamicLibrary tile_;
    DynamicLibrary cuda_;
    DynamicLibrary experimental_mmq_;
    void* stream_ = nullptr;
    std::array<void*, 2> verifier_aux_streams_{};
    void* verifier_ready_event_ = nullptr;
    std::array<void*, 2> verifier_done_events_{};
    bool verifier_projection_overlap_enabled_ = false;
    bool short_attention_split_enabled_ = false;
    bool cooperative_batch_rms_enabled_ = false;
    AbiVersionFn base_abi_ = nullptr;
    AbiVersionFn strict_abi_ = nullptr;
    AbiVersionFn packed_abi_ = nullptr;
    AbiVersionFn feature_abi_ = nullptr;
    ErrorStringFn tile_error_ = nullptr;
    PackedValidateFn packed_validate_ = nullptr;
    PackedLinearFn linear_ = nullptr;
    QuantizeQ8Fn quantize_q8_ = nullptr;
    PackedLinearQ8Fn linear_q8_ = nullptr;
    PackedLinearQ8MultiDecodeFn linear_q8_multi_decode_ = nullptr;
    AbiVersionFn k_quant_mmq_abi_ = nullptr;
    KQuantMmqWorkspaceBytesFn k_quant_mmq_workspace_bytes_ = nullptr;
    KQuantMmqMultiLinearFn k_quant_mmq_multi_linear_ = nullptr;
    KQuantMmqMultiLinearGatedFn k_quant_mmq_multi_linear_gated_ = nullptr;
    KQuantMmqMultiLinearSwiGluFn k_quant_mmq_multi_linear_swiglu_ = nullptr;
    KQuantMmqMultiLinearFn k_quant_mmvq_multi_linear_ = nullptr;
    KQuantMmvqMultiLinearPrequantizedFn
        k_quant_mmvq_multi_linear_prequantized_ = nullptr;
    KQuantMmqMultiLinearGatedFn k_quant_mmvq_multi_linear_gated_ = nullptr;
    KQuantMmqMultiLinearSwiGluFn k_quant_mmvq_multi_linear_swiglu_ = nullptr;
    KQuantMmvqLinearFn k_quant_mmvq_linear_ = nullptr;
    ExperimentalMmqAbiFn experimental_mmq_abi_ = nullptr;
    ExperimentalMmqCreateFn experimental_mmq_create_ = nullptr;
    ExperimentalMmqDestroyFn experimental_mmq_destroy_ = nullptr;
    ExperimentalMmqLinearFn experimental_mmq_linear_ = nullptr;
    ExperimentalMmqMultiLinearFn experimental_mmq_multi_linear_ = nullptr;
    void* experimental_mmq_context_ = nullptr;
    ArgmaxRowsFn argmax_rows_ = nullptr;
    EmbeddingFn embedding_ = nullptr;
    EmbeddingDeviceI64Fn embedding_device_i64_ = nullptr;
    EmbeddingBatchFn embedding_batch_ = nullptr;
    RmsNormFn rms_ = nullptr;
    RmsCaptureFn rms_capture_ = nullptr;
    RmsCaptureQ8Fn rms_capture_q8_ = nullptr;
    RmsAddFn rms_add_ = nullptr;
    DualRmsAddCaptureFn dual_rms_add_capture_ = nullptr;
    DualRmsAddCaptureFn dual_rms_add_capture_cooperative_batch_ = nullptr;
    DualRmsAddCaptureMmvqQ8Fn dual_rms_add_capture_mmvq_q8_ = nullptr;
    RopeFn rope_ = nullptr;
    RopeBatchFn rope_batch_ = nullptr;
    QkNormScaleRopeFn qk_norm_scale_rope_ = nullptr;
    QkNormScaleRopeBatchFn qk_norm_scale_rope_batch_ = nullptr;
    GqaFn gqa_ = nullptr;
    FusedDecodeAttentionFn fused_decode_attention_ = nullptr;
    FusedDecodeAttentionDevicePositionFn
        fused_decode_attention_device_position_ = nullptr;
    CacheCommitFn cache_commit_ = nullptr;
    CacheCommitRowsFn cache_commit_rows_ = nullptr;
    CacheCommitLayersFn cache_commit_layers_ = nullptr;
    PackTargetTapsFn pack_target_taps_ = nullptr;
    DFlashAttentionFn dflash_attention_ = nullptr;
    DFlashAttentionShortSplitFn dflash_attention_short_split_ = nullptr;
    GateFn gate_ = nullptr;
    LogitTransformFn logit_transform_ = nullptr;
    SwiGluFn swiglu_ = nullptr;
    AddFn add_ = nullptr;
    ScaleFn scale_ = nullptr;
    std::mutex vision_mutex_;
    VisionPrepareFn vision_prepare_ = nullptr;
    VisionLayerNormFn vision_layer_norm_ = nullptr;
    VisionAttentionFn vision_attention_ = nullptr;
    VisionPixelShuffleFn vision_pixel_shuffle_ = nullptr;
    GeluFn gelu_ = nullptr;
    CudaGetDeviceCountFn get_device_count_ = nullptr;
    CudaSetDeviceFn set_device_ = nullptr;
    CudaMallocFn malloc_ = nullptr;
    CudaFreeFn free_ = nullptr;
    CudaMemcpyFn memcpy_ = nullptr;
    CudaMemcpyAsyncFn memcpy_async_ = nullptr;
    CudaMemsetAsyncFn memset_async_ = nullptr;
    CudaStreamCreateFn stream_create_ = nullptr;
    CudaStreamDestroyFn stream_destroy_ = nullptr;
    CudaStreamSynchronizeFn stream_sync_ = nullptr;
    CudaStreamWaitEventFn stream_wait_event_ = nullptr;
    CudaEventCreateWithFlagsFn event_create_with_flags_ = nullptr;
    CudaEventDestroyFn event_destroy_ = nullptr;
    CudaEventRecordFn event_record_ = nullptr;
    CudaStreamBeginCaptureFn stream_begin_capture_ = nullptr;
    CudaStreamEndCaptureFn stream_end_capture_ = nullptr;
    CudaGraphInstantiateWithFlagsFn graph_instantiate_with_flags_ = nullptr;
    CudaGraphDestroyFn graph_destroy_ = nullptr;
    CudaGraphExecDestroyFn graph_exec_destroy_ = nullptr;
    CudaGraphLaunchFn graph_launch_ = nullptr;
    CudaGetErrorStringFn error_string_ = nullptr;
};

class DeviceBuffer final {
public:
    DeviceBuffer() = default;
    DeviceBuffer(std::shared_ptr<Runtime> runtime, std::size_t bytes)
        : runtime_(std::move(runtime)), bytes_(bytes), pointer_(runtime_->allocate(bytes)) {}
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(DeviceBuffer&& other) noexcept { *this = std::move(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            runtime_ = std::move(other.runtime_);
            bytes_ = std::exchange(other.bytes_, 0);
            pointer_ = std::exchange(other.pointer_, nullptr);
        }
        return *this;
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void reset() noexcept {
        if (runtime_) runtime_->free_noexcept(pointer_);
        pointer_ = nullptr;
        bytes_ = 0;
        runtime_.reset();
    }
    void* get() const noexcept { return pointer_; }
    template <typename Value> Value* as() const noexcept {
        return static_cast<Value*>(pointer_);
    }
    std::size_t bytes() const noexcept { return bytes_; }
private:
    std::shared_ptr<Runtime> runtime_;
    std::size_t bytes_ = 0;
    void* pointer_ = nullptr;
};

struct DeviceWeight {
    DeviceBuffer storage;
    NfnNativeTilePackedWeightDescriptorV1 descriptor{};
    bool centered = false;
};

struct DeviceLayer {
    DeviceWeight input_norm;
    DeviceWeight post_attention_norm;
    DeviceWeight pre_feedforward_norm;
    DeviceWeight post_feedforward_norm;
    DeviceWeight q;
    DeviceWeight k;
    DeviceWeight v;
    DeviceWeight gate;
    DeviceWeight output;
    DeviceWeight mlp_gate;
    DeviceWeight mlp_up;
    DeviceWeight mlp_down;
    std::optional<DeviceWeight> q_norm;
    std::optional<DeviceWeight> k_norm;
};

struct DeviceLoraWeight {
    DeviceWeight a;
    DeviceWeight b;
    float scaling = 1.0f;
};

struct DeviceLoraLayer {
    std::optional<DeviceLoraWeight> q;
    std::optional<DeviceLoraWeight> k;
    std::optional<DeviceLoraWeight> v;
    std::optional<DeviceLoraWeight> gate;
    std::optional<DeviceLoraWeight> output;
    std::optional<DeviceLoraWeight> mlp_gate;
    std::optional<DeviceLoraWeight> mlp_up;
    std::optional<DeviceLoraWeight> mlp_down;
};

DeviceWeight upload_weight(
    const std::shared_ptr<Runtime>& runtime,
    const HostWeightView& source,
    std::int64_t* total_bytes) {
    if (source.data == nullptr || source.rows <= 0 || source.cols <= 0 ||
        source.row_stride_bytes <= 0 || source.nbytes <= 0 ||
        source.nbytes != checked_mul(source.rows, source.row_stride_bytes, "weight extent")) {
        throw std::runtime_error("Glimmer CUDA weight descriptor is invalid");
    }
    DeviceWeight result;
    result.storage = DeviceBuffer(runtime, checked_size(source.nbytes, "weight bytes"));
    runtime->copy_h2d_async(result.storage.get(), source.data, result.storage.bytes());
    result.descriptor.struct_size = sizeof(result.descriptor);
    result.descriptor.version = NFN_NATIVE_TILE_PACKED_WEIGHT_V1;
    result.descriptor.encoding = source.encoding;
    result.descriptor.data = result.storage.as<const std::uint8_t>();
    result.descriptor.data_nbytes = source.nbytes;
    result.descriptor.output_dim = source.rows;
    result.descriptor.input_dim = source.cols;
    result.descriptor.row_stride_bytes = source.row_stride_bytes;
    result.descriptor.cuda_stream = runtime->stream();
    result.centered = source.centered;
    runtime->check_tile(runtime->packed_validate()(&result.descriptor), "packed weight validate");
    if (*total_bytes > std::numeric_limits<std::int64_t>::max() - source.nbytes) {
        throw std::runtime_error("Glimmer CUDA resident weight byte count overflow");
    }
    *total_bytes += source.nbytes;
    return result;
}

void throw_if_cancelled(const std::atomic<bool>& cancelled) {
    if (cancelled.load(std::memory_order_relaxed)) {
        throw std::runtime_error("resident inference session was cancelled");
    }
}

struct DeviceVisionLinear {
    DeviceWeight weight;
    DeviceBuffer bias;
    bool has_bias = false;
};

struct DeviceVisionLayer {
    DeviceVisionLinear query;
    DeviceVisionLinear key;
    DeviceVisionLinear value;
    DeviceVisionLinear output;
    DeviceBuffer norm1_weight;
    DeviceBuffer norm1_bias;
    DeviceBuffer norm2_weight;
    DeviceBuffer norm2_bias;
    DeviceVisionLinear fc1;
    DeviceVisionLinear fc2;
};

DeviceBuffer upload_float_vector(
    const std::shared_ptr<Runtime>& runtime,
    const std::vector<float>& source,
    std::int64_t expected,
    std::int64_t* weight_bytes,
    const char* label) {
    if (expected <= 0 || source.size() != checked_size(expected, label)) {
        throw std::runtime_error(std::string("Glimmer CUDA vision ") + label +
                                 " extent is invalid");
    }
    const std::size_t bytes = checked_size(checked_mul(expected, 4, label), label);
    DeviceBuffer result(runtime, bytes);
    runtime->copy_h2d_async(result.get(), source.data(), bytes);
    if (*weight_bytes > std::numeric_limits<std::int64_t>::max() -
            static_cast<std::int64_t>(bytes)) {
        throw std::runtime_error("Glimmer CUDA vision weight byte count overflow");
    }
    *weight_bytes += static_cast<std::int64_t>(bytes);
    return result;
}

class VisionExecutor final {
public:
    VisionExecutor(
        std::shared_ptr<Runtime> source_runtime,
        const VisionConfig& source_config,
        const VisionHostWeightPlan& source)
        : runtime_(std::move(source_runtime)), config_(source_config) {
        if (config_.hidden_size <= 0 || config_.intermediate_size <= 0 ||
            config_.num_layers <= 0 || config_.num_heads <= 0 ||
            config_.hidden_size % config_.num_heads != 0 ||
            (config_.hidden_size / config_.num_heads) % 4 != 0 ||
            config_.patch_width <= 0 || config_.merge_size <= 0 ||
            config_.position_side <= 0 || config_.adapter_size <= 0 ||
            config_.output_size <= 0 || source.layers.size() !=
                checked_size(config_.num_layers, "vision layers") ||
            !std::isfinite(config_.rope_theta) || !(config_.rope_theta > 0.0f) ||
            !std::isfinite(config_.norm_eps) || !(config_.norm_eps > 0.0f)) {
            throw std::runtime_error("Glimmer CUDA vision geometry is invalid");
        }
        runtime_->enable_vision();
        patch_ = upload_weight(runtime_, source.patch, &weight_bytes_);
        const std::int64_t position_rows = checked_mul(
            config_.position_side, config_.position_side, "vision position rows");
        position_ = upload_float_vector(
            runtime_, source.position,
            checked_mul(position_rows, config_.hidden_size, "vision position"),
            &weight_bytes_, "position table");
        pre_norm_weight_ = upload_float_vector(
            runtime_, source.pre_norm_weight, config_.hidden_size,
            &weight_bytes_, "pre norm weight");
        pre_norm_bias_ = upload_float_vector(
            runtime_, source.pre_norm_bias, config_.hidden_size,
            &weight_bytes_, "pre norm bias");
        post_norm_weight_ = upload_float_vector(
            runtime_, source.post_norm_weight, config_.hidden_size,
            &weight_bytes_, "post norm weight");
        post_norm_bias_ = upload_float_vector(
            runtime_, source.post_norm_bias, config_.hidden_size,
            &weight_bytes_, "post norm bias");
        layers_.reserve(source.layers.size());
        for (const VisionHostLayer& host : source.layers) {
            DeviceVisionLayer layer;
            layer.query = upload_linear(host.query);
            layer.key = upload_linear(host.key);
            layer.value = upload_linear(host.value);
            layer.output = upload_linear(host.output);
            layer.norm1_weight = upload_float_vector(
                runtime_, host.norm1_weight, config_.hidden_size,
                &weight_bytes_, "layer norm1 weight");
            layer.norm1_bias = upload_float_vector(
                runtime_, host.norm1_bias, config_.hidden_size,
                &weight_bytes_, "layer norm1 bias");
            layer.norm2_weight = upload_float_vector(
                runtime_, host.norm2_weight, config_.hidden_size,
                &weight_bytes_, "layer norm2 weight");
            layer.norm2_bias = upload_float_vector(
                runtime_, host.norm2_bias, config_.hidden_size,
                &weight_bytes_, "layer norm2 bias");
            layer.fc1 = upload_linear(host.fc1);
            layer.fc2 = upload_linear(host.fc2);
            layers_.push_back(std::move(layer));
        }
        adapter_fc1_ = upload_linear(source.adapter_fc1);
        adapter_fc2_ = upload_linear(source.adapter_fc2);
        projection_ = upload_linear(source.projection);
        runtime_->synchronize();
    }

    std::vector<float> encode(
        const std::vector<float>& patches,
        const std::vector<std::int64_t>& grid_thw,
        const std::atomic<bool>& cancelled) {
        throw_if_cancelled(cancelled);
        if (patches.empty() || patches.size() %
                checked_size(config_.patch_width, "vision patch width") != 0) {
            throw std::runtime_error("Glimmer CUDA vision patch extent is invalid");
        }
        const std::int64_t rows = static_cast<std::int64_t>(patches.size()) /
            config_.patch_width;
        const Layout layout = make_layout(grid_thw, rows);
        const auto floats = [&](std::int64_t count, const char* label) {
            return checked_size(checked_mul(count, 4, label), label);
        };
        DeviceBuffer input(runtime_, floats(
            checked_mul(rows, config_.patch_width, "vision input"), "vision input"));
        DeviceBuffer projected(runtime_, floats(
            checked_mul(rows, config_.hidden_size, "vision projected"), "vision projected"));
        DeviceBuffer hidden(runtime_, projected.bytes());
        DeviceBuffer normalized(runtime_, projected.bytes());
        DeviceBuffer query(runtime_, projected.bytes());
        DeviceBuffer key(runtime_, projected.bytes());
        DeviceBuffer value(runtime_, projected.bytes());
        DeviceBuffer attended(runtime_, projected.bytes());
        DeviceBuffer branch(runtime_, projected.bytes());
        DeviceBuffer mlp(runtime_, floats(
            checked_mul(rows, config_.intermediate_size, "vision MLP"), "vision MLP"));
        DeviceBuffer permutation = upload_i32(layout.permutation, "vision permutation");
        DeviceBuffer corner_indices = upload_i32(layout.corner_indices, "vision corners");
        DeviceBuffer corner_weights = upload_f32(layout.corner_weights, "vision corner weights");
        DeviceBuffer position_width = upload_i32(layout.position_width, "vision width positions");
        DeviceBuffer position_height = upload_i32(layout.position_height, "vision height positions");
        DeviceBuffer window_begin = upload_i32(layout.window_begin, "vision window begin");
        DeviceBuffer window_end = upload_i32(layout.window_end, "vision window end");
        DeviceBuffer full_begin = upload_i32(layout.full_begin, "vision full begin");
        DeviceBuffer full_end = upload_i32(layout.full_end, "vision full end");
        DeviceBuffer pixel_sources = upload_i32(layout.pixel_sources, "vision pixel sources");
        input_copy(input, patches);
        linear(patch_, nullptr, input, projected, rows);
        NfnNativeTileGlimmerVisionPrepareDescriptorV1 prepare{
            .struct_size = sizeof(prepare),
            .version = NFN_NATIVE_TILE_GLIMMER_VISION_V1,
            .projected = projected.as<const float>(),
            .position_table = position_.as<const float>(),
            .corner_indices = corner_indices.as<const std::int32_t>(),
            .corner_weights = corner_weights.as<const float>(),
            .permutation = permutation.as<const std::int32_t>(),
            .output = hidden.as<float>(),
            .rows = rows,
            .width = config_.hidden_size,
            .position_rows = checked_mul(
                config_.position_side, config_.position_side, "vision position rows"),
            .cuda_stream = runtime_->stream(),
        };
        call(runtime_->vision_prepare()(&prepare), "Glimmer vision prepare");
        layer_norm(
            hidden, pre_norm_weight_, pre_norm_bias_, normalized, rows,
            config_.hidden_size);
        std::swap(hidden, normalized);

        for (std::int64_t index = 0; index < config_.num_layers; ++index) {
            throw_if_cancelled(cancelled);
            const DeviceVisionLayer& layer = layers_.at(
                checked_size(index, "vision layer"));
            layer_norm(
                hidden, layer.norm1_weight, layer.norm1_bias, normalized,
                rows, config_.hidden_size);
            linear(layer.query.weight, &layer.query, normalized, query, rows);
            linear(layer.key.weight, &layer.key, normalized, key, rows);
            linear(layer.value.weight, &layer.value, normalized, value, rows);
            const bool full = (index + 1) % 4 == 0 || index + 1 == config_.num_layers;
            NfnNativeTileGlimmerVisionAttentionDescriptorV1 attention{
                .struct_size = sizeof(attention),
                .version = NFN_NATIVE_TILE_GLIMMER_VISION_V1,
                .interleaved_rope = config_.interleaved_rope ? 1U : 0U,
                .reserved0 = 0,
                .query = query.as<const float>(),
                .key = key.as<const float>(),
                .value = value.as<const float>(),
                .position_width = position_width.as<const std::int32_t>(),
                .position_height = position_height.as<const std::int32_t>(),
                .row_begin = (full ? full_begin : window_begin).as<const std::int32_t>(),
                .row_end = (full ? full_end : window_end).as<const std::int32_t>(),
                .output = attended.as<float>(),
                .rows = rows,
                .heads = config_.num_heads,
                .head_dim = config_.hidden_size / config_.num_heads,
                .rope_theta = config_.rope_theta,
                .reserved1 = 0,
                .cuda_stream = runtime_->stream(),
            };
            call(runtime_->vision_attention()(&attention), "Glimmer vision attention");
            linear(layer.output.weight, &layer.output, attended, branch, rows);
            add_inplace(hidden, branch, rows * config_.hidden_size);
            layer_norm(
                hidden, layer.norm2_weight, layer.norm2_bias, normalized,
                rows, config_.hidden_size);
            linear(layer.fc1.weight, &layer.fc1, normalized, mlp, rows);
            call(runtime_->gelu()(
                mlp.as<const float>(), mlp.as<float>(),
                rows * config_.intermediate_size, runtime_->stream()),
                "Glimmer vision GELU");
            linear(layer.fc2.weight, &layer.fc2, mlp, branch, rows);
            add_inplace(hidden, branch, rows * config_.hidden_size);
        }
        layer_norm(
            hidden, post_norm_weight_, post_norm_bias_, normalized, rows,
            config_.hidden_size);
        const std::int64_t merge_area = checked_mul(
            config_.merge_size, config_.merge_size, "vision merge area");
        DeviceBuffer merged(runtime_, floats(checked_mul(
            checked_mul(layout.merged_rows, merge_area, "vision merged rows"),
            config_.hidden_size, "vision merged"), "vision merged"));
        NfnNativeTileGlimmerVisionPixelShuffleDescriptorV1 shuffle{
            .struct_size = sizeof(shuffle),
            .version = NFN_NATIVE_TILE_GLIMMER_VISION_V1,
            .reordered_hidden = normalized.as<const float>(),
            .source_rows = pixel_sources.as<const std::int32_t>(),
            .output = merged.as<float>(),
            .merged_rows = layout.merged_rows,
            .hidden_size = config_.hidden_size,
            .merge_area = merge_area,
            .cuda_stream = runtime_->stream(),
        };
        call(runtime_->vision_pixel_shuffle()(&shuffle), "Glimmer vision pixel shuffle");
        DeviceBuffer adapted(runtime_, floats(checked_mul(
            layout.merged_rows, config_.adapter_size, "vision adapted"),
            "vision adapted"));
        DeviceBuffer adapter_output(runtime_, adapted.bytes());
        linear(adapter_fc1_.weight, &adapter_fc1_, merged, adapted, layout.merged_rows);
        call(runtime_->gelu()(
            adapted.as<const float>(), adapted.as<float>(),
            layout.merged_rows * config_.adapter_size, runtime_->stream()),
            "Glimmer vision adapter GELU 1");
        linear(
            adapter_fc2_.weight, &adapter_fc2_, adapted, adapter_output,
            layout.merged_rows);
        call(runtime_->gelu()(
            adapter_output.as<const float>(), adapter_output.as<float>(),
            layout.merged_rows * config_.adapter_size, runtime_->stream()),
            "Glimmer vision adapter GELU 2");
        DeviceBuffer output(runtime_, floats(checked_mul(
            layout.merged_rows, config_.output_size, "vision output"),
            "vision output"));
        linear(
            projection_.weight, &projection_, adapter_output, output,
            layout.merged_rows);
        call(runtime_->rms()(
            output.as<const float>(), nullptr, output.as<float>(),
            layout.merged_rows, config_.output_size, config_.norm_eps, false,
            runtime_->stream()), "Glimmer vision output RMSNorm");
        runtime_->synchronize();
        throw_if_cancelled(cancelled);
        std::vector<float> host(checked_size(
            checked_mul(layout.merged_rows, config_.output_size, "vision host output"),
            "vision host output"));
        runtime_->copy_d2h(host.data(), output.get(), output.bytes());
        workspace_bytes_ = static_cast<std::int64_t>(
            input.bytes() + projected.bytes() + hidden.bytes() + normalized.bytes() +
            query.bytes() + key.bytes() + value.bytes() + attended.bytes() +
            branch.bytes() + mlp.bytes() + merged.bytes() + adapted.bytes() +
            adapter_output.bytes() + output.bytes() + permutation.bytes() +
            corner_indices.bytes() + corner_weights.bytes() +
            position_width.bytes() + position_height.bytes() +
            window_begin.bytes() + window_end.bytes() + full_begin.bytes() +
            full_end.bytes() + pixel_sources.bytes());
        return host;
    }

    std::int64_t weight_bytes() const noexcept { return weight_bytes_; }
    std::int64_t workspace_bytes() const noexcept { return workspace_bytes_; }
    std::int64_t launches() const noexcept { return launches_; }

private:
    struct Layout {
        std::vector<std::int32_t> permutation;
        std::vector<std::int32_t> corner_indices;
        std::vector<float> corner_weights;
        std::vector<std::int32_t> position_width;
        std::vector<std::int32_t> position_height;
        std::vector<std::int32_t> window_begin;
        std::vector<std::int32_t> window_end;
        std::vector<std::int32_t> full_begin;
        std::vector<std::int32_t> full_end;
        std::vector<std::int32_t> pixel_sources;
        std::int64_t merged_rows = 0;
    };

    struct GridRow {
        std::int64_t temporal;
        std::int64_t height;
        std::int64_t width;
        std::int64_t offset;
    };

    static std::int32_t i32(std::int64_t value, const char* label) {
        if (value < 0 || value > std::numeric_limits<std::int32_t>::max()) {
            throw std::runtime_error(std::string("Glimmer CUDA vision ") + label +
                                     " exceeds int32");
        }
        return static_cast<std::int32_t>(value);
    }

    Layout make_layout(
        const std::vector<std::int64_t>& grid_thw,
        std::int64_t supplied_rows) const {
        if (grid_thw.empty() || grid_thw.size() % 3 != 0) {
            throw std::runtime_error("Glimmer CUDA vision grid is invalid");
        }
        Layout result;
        std::vector<GridRow> grid;
        std::vector<std::int64_t> window_boundaries{0};
        std::vector<std::int64_t> full_boundaries{0};
        std::int64_t offset = 0;
        for (std::size_t media = 0; media < grid_thw.size(); media += 3) {
            const std::int64_t temporal = grid_thw[media];
            const std::int64_t height = grid_thw[media + 1];
            const std::int64_t width = grid_thw[media + 2];
            if (temporal <= 0 || height <= 0 || width <= 0 ||
                height % config_.merge_size != 0 ||
                width % config_.merge_size != 0) {
                throw std::runtime_error("Glimmer CUDA vision grid is unmergeable");
            }
            const std::int64_t spatial = checked_mul(height, width, "vision spatial");
            grid.push_back({temporal, height, width, offset});
            offset += checked_mul(temporal, spatial, "vision media rows");
            full_boundaries.push_back(offset);
            result.merged_rows += checked_mul(
                temporal, checked_mul(height / config_.merge_size,
                                      width / config_.merge_size,
                                      "vision merged spatial"),
                "vision merged rows");
            const std::int64_t window = config_.position_side;
            for (std::int64_t time = 0; time < temporal; ++time) {
                for (std::int64_t wh = 0; wh < (height + window - 1) / window; ++wh) {
                    for (std::int64_t ww = 0; ww < (width + window - 1) / window; ++ww) {
                        std::int64_t count = 0;
                        for (std::int64_t lh = 0; lh < window; ++lh) {
                            const std::int64_t h = wh * window + lh;
                            if (h >= height) continue;
                            for (std::int64_t lw = 0; lw < window; ++lw) {
                                const std::int64_t w = ww * window + lw;
                                if (w >= width) continue;
                                result.permutation.push_back(i32(
                                    grid.back().offset + time * spatial + h * width + w,
                                    "permutation"));
                                result.position_width.push_back(i32(w + 1, "width position"));
                                result.position_height.push_back(i32(h + 1, "height position"));
                                ++count;
                            }
                        }
                        if (count > 0) window_boundaries.push_back(
                            window_boundaries.back() + count);
                    }
                }
            }
        }
        if (offset != supplied_rows ||
            result.permutation.size() != checked_size(supplied_rows, "vision rows")) {
            throw std::runtime_error("Glimmer CUDA vision grid/patch rows differ");
        }
        result.corner_indices.assign(
            checked_size(checked_mul(supplied_rows, 4, "vision corners"),
                         "vision corners"), -1);
        result.corner_weights.assign(result.corner_indices.size(), 0.0f);
        const float side = static_cast<float>(config_.position_side);
        for (const GridRow& media : grid) {
            const std::int64_t spatial = media.height * media.width;
            for (std::int64_t h = 0; h < media.height; ++h) {
                const float hc = (static_cast<float>(h) + 0.5f) *
                    side / static_cast<float>(media.height) - 0.5f;
                const std::int64_t hf = static_cast<std::int64_t>(std::floor(hc));
                const float hd = hc - static_cast<float>(hf);
                for (std::int64_t w = 0; w < media.width; ++w) {
                    const float wc = (static_cast<float>(w) + 0.5f) *
                        side / static_cast<float>(media.width) - 0.5f;
                    const std::int64_t wf = static_cast<std::int64_t>(std::floor(wc));
                    const float wd = wc - static_cast<float>(wf);
                    const std::int64_t hs[4] = {hf, hf, hf + 1, hf + 1};
                    const std::int64_t ws[4] = {wf, wf + 1, wf, wf + 1};
                    const float weights[4] = {
                        (1.0f-hd)*(1.0f-wd), (1.0f-hd)*wd,
                        hd*(1.0f-wd), hd*wd};
                    for (std::int64_t time = 0; time < media.temporal; ++time) {
                        const std::int64_t row = media.offset + time * spatial +
                            h * media.width + w;
                        for (int corner = 0; corner < 4; ++corner) {
                            const std::size_t target = checked_size(
                                row * 4 + corner, "vision corner");
                            result.corner_weights[target] = weights[corner];
                            if (hs[corner] >= 0 && hs[corner] < config_.position_side &&
                                ws[corner] >= 0 && ws[corner] < config_.position_side) {
                                result.corner_indices[target] = i32(
                                    hs[corner] * config_.position_side + ws[corner],
                                    "position table row");
                            }
                        }
                    }
                }
            }
        }
        const auto fill_ranges = [&](const std::vector<std::int64_t>& boundaries,
                                     std::vector<std::int32_t>* begin,
                                     std::vector<std::int32_t>* end) {
            begin->resize(checked_size(supplied_rows, "vision range rows"));
            end->resize(begin->size());
            for (std::size_t segment = 0; segment + 1 < boundaries.size(); ++segment) {
                for (std::int64_t row = boundaries[segment];
                     row < boundaries[segment + 1]; ++row) {
                    begin->at(checked_size(row, "vision range row")) =
                        i32(boundaries[segment], "range begin");
                    end->at(checked_size(row, "vision range row")) =
                        i32(boundaries[segment + 1], "range end");
                }
            }
        };
        fill_ranges(window_boundaries, &result.window_begin, &result.window_end);
        fill_ranges(full_boundaries, &result.full_begin, &result.full_end);
        std::vector<std::int32_t> inverse(
            checked_size(supplied_rows, "vision inverse permutation"));
        for (std::int64_t row = 0; row < supplied_rows; ++row) {
            inverse.at(static_cast<std::size_t>(result.permutation[row])) = i32(row, "inverse row");
        }
        result.pixel_sources.reserve(checked_size(
            checked_mul(result.merged_rows,
                        config_.merge_size * config_.merge_size,
                        "vision pixel source rows"),
            "vision pixel source rows"));
        for (const GridRow& media : grid) {
            const std::int64_t spatial = media.height * media.width;
            for (std::int64_t time = 0; time < media.temporal; ++time) {
                for (std::int64_t bh = 0; bh < media.height / config_.merge_size; ++bh) {
                    for (std::int64_t bw = 0; bw < media.width / config_.merge_size; ++bw) {
                        for (std::int64_t lh = 0; lh < config_.merge_size; ++lh) {
                            for (std::int64_t lw = 0; lw < config_.merge_size; ++lw) {
                                const std::int64_t source = media.offset + time * spatial +
                                    (bh * config_.merge_size + lh) * media.width +
                                    bw * config_.merge_size + lw;
                                result.pixel_sources.push_back(
                                    inverse.at(checked_size(source, "vision pixel source")));
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    DeviceVisionLinear upload_linear(const VisionHostLinear& source) {
        DeviceVisionLinear result;
        result.weight = upload_weight(runtime_, source.weight, &weight_bytes_);
        if (!source.bias.empty()) {
            result.bias = upload_float_vector(
                runtime_, source.bias, source.weight.rows, &weight_bytes_,
                "linear bias");
            result.has_bias = true;
        }
        return result;
    }

    DeviceBuffer upload_i32(
        const std::vector<std::int32_t>& source, const char* label) const {
        if (source.empty()) throw std::runtime_error(std::string(label) + " is empty");
        DeviceBuffer result(runtime_, source.size() * sizeof(std::int32_t));
        runtime_->copy_h2d_async(result.get(), source.data(), result.bytes());
        return result;
    }

    DeviceBuffer upload_f32(
        const std::vector<float>& source, const char* label) const {
        if (source.empty()) throw std::runtime_error(std::string(label) + " is empty");
        DeviceBuffer result(runtime_, source.size() * sizeof(float));
        runtime_->copy_h2d_async(result.get(), source.data(), result.bytes());
        return result;
    }

    void input_copy(
        DeviceBuffer& target, const std::vector<float>& source) const {
        if (target.bytes() != source.size() * sizeof(float)) {
            throw std::runtime_error("Glimmer CUDA vision input copy extent mismatch");
        }
        runtime_->copy_h2d_async(target.get(), source.data(), target.bytes());
    }

    void call(int status, const char* label) {
        runtime_->check_tile(status, label);
        ++launches_;
    }

    void linear(
        const DeviceWeight& weight,
        const DeviceVisionLinear* linear,
        const DeviceBuffer& input,
        DeviceBuffer& output,
        std::int64_t rows) {
        const float* bias = linear != nullptr && linear->has_bias
            ? linear->bias.as<const float>() : nullptr;
        call(runtime_->linear()(
            &weight.descriptor, input.as<const float>(), bias,
            output.as<float>(), rows, bias != nullptr), "Glimmer vision packed linear");
    }

    void layer_norm(
        const DeviceBuffer& input,
        const DeviceBuffer& weight,
        const DeviceBuffer& bias,
        DeviceBuffer& output,
        std::int64_t rows,
        std::int64_t width) {
        call(runtime_->vision_layer_norm()(
            input.as<const float>(), weight.as<const float>(),
            bias.as<const float>(), output.as<float>(), rows, width,
            config_.norm_eps, runtime_->stream()), "Glimmer vision LayerNorm");
    }

    void add_inplace(
        DeviceBuffer& target, const DeviceBuffer& branch,
        std::int64_t count) {
        call(runtime_->add()(
            target.as<const float>(), branch.as<const float>(), target.as<float>(),
            count, runtime_->stream()), "Glimmer vision residual add");
    }

    std::shared_ptr<Runtime> runtime_;
    VisionConfig config_;
    DeviceWeight patch_;
    DeviceBuffer position_;
    DeviceBuffer pre_norm_weight_;
    DeviceBuffer pre_norm_bias_;
    DeviceBuffer post_norm_weight_;
    DeviceBuffer post_norm_bias_;
    std::vector<DeviceVisionLayer> layers_;
    DeviceVisionLinear adapter_fc1_;
    DeviceVisionLinear adapter_fc2_;
    DeviceVisionLinear projection_;
    std::int64_t weight_bytes_ = 0;
    std::int64_t workspace_bytes_ = 0;
    std::int64_t launches_ = 0;
};

}  // namespace

class Cache::Impl final {
public:
    struct Layer {
        bool local = false;
        std::int64_t capacity = 0;
        DeviceBuffer keys;
        DeviceBuffer values;
    };

    Impl(std::shared_ptr<Runtime> runtime, const Config& config)
        : runtime(std::move(runtime)),
          final_hidden(this->runtime, checked_size(
              checked_mul(config.model_dim, 4, "final hidden"), "final hidden")),
          decode_graph_position(this->runtime, sizeof(std::int64_t)) {
        const std::int64_t kv_width = checked_mul(
            config.num_kv_heads, config.head_dim, "KV width");
        layers.reserve(checked_size(config.num_layers, "cache layer count"));
        layer_descriptors.reserve(
            checked_size(config.num_layers, "cache layer descriptor count"));
        for (std::int64_t index = 0; index < config.num_layers; ++index) {
            Layer layer;
            layer.local = index % 4 != 3;
            layer.capacity = layer.local ? config.sliding_window : config.max_seq_len;
            const std::int64_t elements = checked_mul(layer.capacity, kv_width, "cache elements");
            const std::size_t bytes = checked_size(checked_mul(elements, 2, "cache bytes"), "cache bytes");
            layer.keys = DeviceBuffer(this->runtime, bytes);
            layer.values = DeviceBuffer(this->runtime, bytes);
            this->runtime->zero_async(layer.keys.get(), bytes);
            this->runtime->zero_async(layer.values.get(), bytes);
            allocated_bytes += static_cast<std::int64_t>(bytes) * 2;
            layers.push_back(std::move(layer));
            const Layer& stored = layers.back();
            NfnNativeTileGlimmerCacheLayerV1 descriptor{};
            descriptor.key_cache_bf16 = stored.keys.as<std::uint16_t>();
            descriptor.value_cache_bf16 = stored.values.as<std::uint16_t>();
            descriptor.cache_capacity = stored.capacity;
            descriptor.cache_row_stride = kv_width;
            layer_descriptors.push_back(descriptor);
        }
        allocated_bytes += static_cast<std::int64_t>(
            final_hidden.bytes() + decode_graph_position.bytes());
        this->runtime->zero_async(final_hidden.get(), final_hidden.bytes());
        this->runtime->zero_async(
            decode_graph_position.get(), decode_graph_position.bytes());
        this->runtime->synchronize();
    }

    ~Impl() {
        runtime->destroy_decode_graph_noexcept(decode_graph_exec);
        decode_graph_exec = nullptr;
    }

    std::shared_ptr<Runtime> runtime;
    std::vector<Layer> layers;
    std::vector<NfnNativeTileGlimmerCacheLayerV1> layer_descriptors;
    DeviceBuffer final_hidden;
    DeviceBuffer decode_graph_position;
    void* decode_graph_exec = nullptr;
    bool decode_graph_disabled = false;
    std::int64_t logical_length = 0;
    std::int64_t allocated_bytes = 0;
};

Cache::Cache(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Cache::~Cache() = default;
std::int64_t Cache::logical_length() const noexcept { return impl_->logical_length; }
std::int64_t Cache::allocated_bytes() const noexcept { return impl_->allocated_bytes; }

class VerificationScratch final {
public:
    VerificationScratch(
        const std::shared_ptr<Runtime>& runtime,
        const Config& config,
        std::int64_t rows,
        std::int64_t tap_count)
        : staged_keys(runtime, checked_size(checked_mul(checked_mul(
              checked_mul(config.num_layers, rows, "verification layer rows"),
              checked_mul(config.num_kv_heads, config.head_dim,
                          "verification KV width"),
              "verification KV elements"), 4, "verification KV bytes"),
              "verification KV bytes")),
          staged_values(runtime, staged_keys.bytes()),
          staged_final(runtime, checked_size(checked_mul(checked_mul(
              rows, config.model_dim, "verification final rows"), 4,
              "verification final bytes"), "verification final bytes")),
          staged_taps(runtime, checked_size(checked_mul(checked_mul(checked_mul(
              rows, tap_count, "verification tap rows"), config.model_dim,
              "verification taps"), 4, "verification tap bytes"),
              "verification tap bytes")),
          attention_scores(runtime, checked_size(checked_mul(checked_mul(
              checked_mul(rows, config.num_heads,
                          "verification attention rows"),
              128, "verification attention score capacity"),
              4, "verification attention score bytes"),
              "verification attention score bytes")) {}

    std::size_t bytes() const noexcept {
        return staged_keys.bytes() + staged_values.bytes() +
            staged_final.bytes() + staged_taps.bytes() +
            attention_scores.bytes();
    }

    std::atomic<bool> leased{false};
    DeviceBuffer staged_keys;
    DeviceBuffer staged_values;
    DeviceBuffer staged_final;
    DeviceBuffer staged_taps;
    DeviceBuffer attention_scores;
};

class VerificationScratchLease final {
public:
    VerificationScratchLease(
        std::shared_ptr<VerificationScratch> source,
        bool leased)
        : scratch_(std::move(source)), leased_(leased) {}
    ~VerificationScratchLease() {
        if (leased_ && scratch_) {
            scratch_->leased.store(false, std::memory_order_release);
        }
    }
    VerificationScratchLease(const VerificationScratchLease&) = delete;
    VerificationScratchLease& operator=(const VerificationScratchLease&) = delete;

private:
    std::shared_ptr<VerificationScratch> scratch_;
    bool leased_ = false;
};

class Verification::Impl final {
public:
    Impl(
        std::shared_ptr<Runtime> source_runtime,
        const Config& config,
        std::int64_t source_position,
        std::int64_t source_rows,
        std::int64_t tap_count,
        bool compute_logits,
        bool compute_argmax,
        bool copy_taps_to_host,
        std::shared_ptr<VerificationScratch> source_scratch,
        bool source_scratch_lease)
        : runtime(std::move(source_runtime)),
          position(source_position),
          row_count(source_rows),
          tap_count(tap_count),
          kv_width(checked_mul(config.num_kv_heads, config.head_dim, "verification KV width")),
          scratch(std::move(source_scratch)),
          scratch_lease(scratch, source_scratch_lease),
          host_logits(compute_logits ? checked_size(checked_mul(
              source_rows, config.vocab_size, "verification logits"),
              "verification logits") : 0),
          host_argmax_indices(compute_argmax
              ? checked_size(source_rows, "verification argmax indices") : 0),
          host_argmax_values(compute_argmax
              ? checked_size(source_rows, "verification argmax values") : 0),
          host_tap_major(copy_taps_to_host ? checked_size(checked_mul(checked_mul(
              source_rows, tap_count, "verification tap-major rows"),
              config.model_dim, "verification tap-major values"),
              "verification tap-major values") : 0),
          host_taps(copy_taps_to_host ? checked_size(checked_mul(checked_mul(
              source_rows, tap_count, "verification tap rows"), config.model_dim,
              "verification taps"), "verification taps") : 0),
          cuda_device(config.cuda_device) {
        if (!scratch) {
            throw std::runtime_error("verification scratch is unavailable");
        }
    }

    std::shared_ptr<Runtime> runtime;
    std::int64_t position = 0;
    std::int64_t row_count = 0;
    std::int64_t tap_count = 0;
    std::int64_t kv_width = 0;
    std::shared_ptr<VerificationScratch> scratch;
    VerificationScratchLease scratch_lease;
    std::vector<float> host_logits;
    std::vector<std::int64_t> host_argmax_indices;
    std::vector<float> host_argmax_values;
    std::vector<float> host_tap_major;
    std::vector<float> host_taps;
    int cuda_device = -1;
};

Verification::Verification(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Verification::~Verification() = default;
std::int64_t Verification::rows() const noexcept { return impl_ ? impl_->row_count : 0; }
std::int64_t Verification::position() const noexcept { return impl_ ? impl_->position : 0; }
const std::vector<float>& Verification::logits() const noexcept {
    return impl_->host_logits;
}
const std::vector<std::int64_t>& Verification::argmax_indices() const noexcept {
    return impl_->host_argmax_indices;
}
const std::vector<float>& Verification::argmax_values() const noexcept {
    return impl_->host_argmax_values;
}
const std::vector<float>& Verification::target_taps() const noexcept {
    return impl_->host_taps;
}
const float* Verification::device_target_taps() const noexcept {
    return impl_ && impl_->tap_count > 0
        ? impl_->scratch->staged_taps.as<const float>() : nullptr;
}
int Verification::cuda_device() const noexcept {
    return impl_ ? impl_->cuda_device : -1;
}

class Model::Impl final {
public:
    Impl(const Config& source_config, const HostWeightPlan& source)
        : config(source_config), runtime(std::make_shared<Runtime>(config)) {
        if (source.layers.size() != checked_size(config.num_layers, "model layer count") ||
            config.num_layers <= 0 || config.num_layers % 4 != 0 || config.max_seq_len <= 0 ||
            config.vocab_size <= 0 || config.model_dim <= 0 ||
            config.intermediate_dim <= 0 || config.num_heads <= 0 ||
            config.num_kv_heads <= 0 || config.num_heads % config.num_kv_heads != 0 ||
            config.head_dim <= 0 || config.head_dim > 256 ||
            config.sliding_window <= 0 || config.sliding_window > config.max_seq_len) {
            throw std::runtime_error("Glimmer CUDA model geometry is invalid");
        }
        token_embedding = upload_weight(runtime, source.token_embedding, &weight_bytes);
        final_norm = upload_weight(runtime, source.final_norm, &weight_bytes);
        lm_head = upload_weight(runtime, source.lm_head, &weight_bytes);
        layers.reserve(source.layers.size());
        for (const HostLayerWeights& host : source.layers) {
            DeviceLayer layer;
            layer.input_norm = upload_weight(runtime, host.input_norm, &weight_bytes);
            layer.post_attention_norm = upload_weight(
                runtime, host.post_attention_norm, &weight_bytes);
            layer.pre_feedforward_norm = upload_weight(
                runtime, host.pre_feedforward_norm, &weight_bytes);
            layer.post_feedforward_norm = upload_weight(
                runtime, host.post_feedforward_norm, &weight_bytes);
            layer.q = upload_weight(runtime, host.q, &weight_bytes);
            layer.k = upload_weight(runtime, host.k, &weight_bytes);
            layer.v = upload_weight(runtime, host.v, &weight_bytes);
            layer.gate = upload_weight(runtime, host.gate, &weight_bytes);
            layer.output = upload_weight(runtime, host.output, &weight_bytes);
            layer.mlp_gate = upload_weight(runtime, host.mlp_gate, &weight_bytes);
            layer.mlp_up = upload_weight(runtime, host.mlp_up, &weight_bytes);
            layer.mlp_down = upload_weight(runtime, host.mlp_down, &weight_bytes);
            if (host.q_norm) layer.q_norm = upload_weight(runtime, *host.q_norm, &weight_bytes);
            if (host.k_norm) layer.k_norm = upload_weight(runtime, *host.k_norm, &weight_bytes);
            layers.push_back(std::move(layer));
        }
        runtime->synchronize();

        const auto floats = [&](std::int64_t count, const char* label) {
            return checked_size(checked_mul(count, 4, label), label);
        };
        hidden = DeviceBuffer(runtime, floats(config.model_dim, "hidden"));
        normalized = DeviceBuffer(runtime, floats(config.model_dim, "normalized"));
        residual = DeviceBuffer(runtime, floats(config.model_dim, "residual"));
        branch = DeviceBuffer(runtime, floats(config.model_dim, "branch"));
        query = DeviceBuffer(runtime, floats(
            checked_mul(config.num_heads, config.head_dim, "query width"), "query"));
        key = DeviceBuffer(runtime, floats(
            checked_mul(config.num_kv_heads, config.head_dim, "key width"), "key"));
        value = DeviceBuffer(runtime, key.bytes());
        gate = DeviceBuffer(runtime, query.bytes());
        attention = DeviceBuffer(runtime, query.bytes());
        mlp_gate = DeviceBuffer(runtime, floats(config.intermediate_dim, "MLP gate"));
        mlp_up = DeviceBuffer(runtime, mlp_gate.bytes());
        mlp_activated = DeviceBuffer(runtime, mlp_gate.bytes());
        staged_final = DeviceBuffer(runtime, hidden.bytes());
        logits_buffer = DeviceBuffer(runtime, floats(config.vocab_size, "logits"));
        raw_head_hidden = DeviceBuffer(runtime, floats(
            checked_mul(16, config.model_dim, "DFlash raw head hidden"),
            "DFlash raw head hidden"));
        raw_head_logits = DeviceBuffer(runtime, floats(
            checked_mul(16, config.vocab_size, "DFlash raw head logits"),
            "DFlash raw head logits"));
        argmax_indices = DeviceBuffer(runtime, checked_size(
            checked_mul(16, static_cast<std::int64_t>(sizeof(std::int64_t)),
                        "argmax indices"),
            "argmax indices"));
        argmax_values = DeviceBuffer(runtime, floats(16, "argmax values"));
        raw_embedding_token_ids = DeviceBuffer(runtime, checked_size(
            checked_mul(16, static_cast<std::int64_t>(sizeof(std::int32_t)),
                        "raw embedding token IDs"),
            "raw embedding token IDs"));
        target_tap_staging = DeviceBuffer(runtime, floats(checked_mul(
            config.num_layers, config.model_dim, "target tap staging"),
            "target tap staging"));
        verify_normalized = DeviceBuffer(runtime, raw_head_hidden.bytes());
        verify_residual = DeviceBuffer(runtime, raw_head_hidden.bytes());
        verify_branch = DeviceBuffer(runtime, raw_head_hidden.bytes());
        verify_query = DeviceBuffer(runtime, floats(checked_mul(
            16, checked_mul(config.num_heads, config.head_dim,
                            "verification query width"),
            "verification query rows"), "verification query rows"));
        verify_key = DeviceBuffer(runtime, floats(checked_mul(
            16, checked_mul(config.num_kv_heads, config.head_dim,
                            "verification key width"),
            "verification key rows"), "verification key rows"));
        verify_value = DeviceBuffer(runtime, verify_key.bytes());
        verify_gate = DeviceBuffer(runtime, verify_query.bytes());
        verify_attention = DeviceBuffer(runtime, verify_query.bytes());
        verify_mlp_gate = DeviceBuffer(runtime, floats(checked_mul(
            16, config.intermediate_dim, "verification MLP rows"),
            "verification MLP rows"));
        verify_mlp_up = DeviceBuffer(runtime, verify_mlp_gate.bytes());
        verify_mlp_activated = DeviceBuffer(runtime, verify_mlp_gate.bytes());
        const std::int64_t q8_width = std::max(
            config.model_dim, std::max(config.intermediate_dim,
                                       checked_mul(config.num_heads, config.head_dim,
                                                   "Q8 attention width")));
        constexpr std::int64_t q8_max_rows = 16;
        q8_values = DeviceBuffer(runtime, checked_size(checked_mul(
            q8_max_rows, q8_width, "Q8 values"), "Q8 values"));
        q8_scales = DeviceBuffer(runtime, floats(
            checked_mul(q8_max_rows, (q8_width + 31) / 32,
                        "Q8 activation scales"),
            "Q8 activation scales"));
        q8_sums = DeviceBuffer(runtime, q8_scales.bytes());
        const std::int64_t k_quant_mmq_bytes =
            runtime->k_quant_mmq_workspace_bytes(q8_max_rows, q8_width);
        if (runtime->has_k_quant_mmq() && k_quant_mmq_bytes <= 0) {
            throw std::runtime_error(
                "Glimmer CUDA exact K-quant MMQ workspace query failed");
        }
        if (k_quant_mmq_bytes > 0) {
            k_quant_mmq_workspace = DeviceBuffer(
                runtime, checked_size(k_quant_mmq_bytes, "exact K-quant MMQ workspace"));
        }
        verification_scratch = std::make_shared<VerificationScratch>(
            runtime, config, 16, 5);
        workspace = static_cast<std::int64_t>(
            hidden.bytes() + normalized.bytes() + residual.bytes() + branch.bytes() +
            query.bytes() + key.bytes() + value.bytes() + gate.bytes() + attention.bytes() +
            mlp_gate.bytes() + mlp_up.bytes() + mlp_activated.bytes() +
            staged_final.bytes() +
            logits_buffer.bytes() + raw_head_hidden.bytes() + raw_head_logits.bytes() +
            argmax_indices.bytes() + argmax_values.bytes() +
            raw_embedding_token_ids.bytes() + target_tap_staging.bytes() +
            verify_normalized.bytes() + verify_residual.bytes() + verify_branch.bytes() +
            verify_query.bytes() + verify_key.bytes() + verify_value.bytes() +
            verify_gate.bytes() + verify_attention.bytes() + verify_mlp_gate.bytes() +
            verify_mlp_up.bytes() + verify_mlp_activated.bytes() +
            q8_values.bytes() + q8_scales.bytes() + q8_sums.bytes() +
            k_quant_mmq_workspace.bytes() + verification_scratch->bytes());

    }

    std::shared_ptr<VerificationScratch> acquire_verification_scratch(
        std::int64_t rows,
        std::int64_t tap_count,
        bool* leased) {
        if (leased == nullptr) {
            throw std::runtime_error("verification scratch lease output is null");
        }
        *leased = false;
        if (rows <= 16 && tap_count <= 5) {
            bool expected = false;
            if (verification_scratch->leased.compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel)) {
                *leased = true;
                return verification_scratch;
            }
        }
        return std::make_shared<VerificationScratch>(
            runtime, config, rows, tap_count);
    }

    void call(int status, const char* label) {
        runtime->check_tile(status, label);
        ++launches;
    }

    void linear(
        const DeviceWeight& weight,
        const DeviceBuffer& input,
        DeviceBuffer& output,
        std::int64_t rows = 1,
        bool exact_verifier_rows = false) {
        if (can_linear_native_mmvq(weight, rows)) {
            call(runtime->k_quant_mmvq_linear(
                &weight.descriptor, input.as<const float>(), output.as<float>(),
                k_quant_mmq_workspace.get(),
                static_cast<std::int64_t>(k_quant_mmq_workspace.bytes())),
                "Glimmer exact K-quant MMVQ linear");
            ++mmq_linears;
            return;
        }
        if (exact_verifier_rows && can_linear_native_mmvq_rows(weight, rows)) {
            const NfnNativeTilePackedWeightDescriptorV1* descriptors[]{
                &weight.descriptor};
            float* outputs[]{output.as<float>()};
            call(runtime->k_quant_mmvq_multi_linear(
                descriptors, input.as<const float>(), outputs, 1, rows,
                k_quant_mmq_workspace.get(),
                static_cast<std::int64_t>(k_quant_mmq_workspace.bytes())),
                "Glimmer verifier-exact K-quant MMVQ linear");
            ++mmq_linears;
            return;
        }
        if (can_linear_native_mmq(weight, rows)) {
            const NfnNativeTilePackedWeightDescriptorV1* descriptors[]{
                &weight.descriptor};
            float* outputs[]{output.as<float>()};
            call(runtime->k_quant_mmq_multi_linear(
                descriptors, input.as<const float>(), outputs, 1, rows,
                k_quant_mmq_workspace.get(),
                static_cast<std::int64_t>(k_quant_mmq_workspace.bytes())),
                "Glimmer exact K-quant MMQ linear");
            ++mmq_linears;
            return;
        }
        if (can_linear_experimental_mmq(weight, rows)) {
            call(runtime->experimental_mmq_linear(
                &weight.descriptor, input.as<const float>(), output.as<float>(),
                rows), "Glimmer experimental packed MMQ linear");
            return;
        }
        call(runtime->linear()(
            &weight.descriptor, input.as<const float>(), nullptr, output.as<float>(), rows, false),
            "Glimmer packed linear");
    }

    void linear_mmvq_prequantized(
        const DeviceWeight& weight,
        DeviceBuffer& output,
        std::int64_t rows,
        void* cuda_stream = nullptr) {
        linear_mmvq_prequantized_raw(
            weight, output.as<float>(), rows, cuda_stream);
    }

    void linear_mmvq_prequantized_raw(
        const DeviceWeight& weight,
        float* output,
        std::int64_t rows,
        void* cuda_stream = nullptr) {
        if (!can_linear_native_mmvq_rows(weight, rows) ||
            !runtime->has_k_quant_mmvq_prequantized() || output == nullptr) {
            throw std::runtime_error(
                "Glimmer verifier prequantized MMVQ linear is unavailable");
        }
        NfnNativeTilePackedWeightDescriptorV1 descriptor = weight.descriptor;
        if (cuda_stream != nullptr) descriptor.cuda_stream = cuda_stream;
        const NfnNativeTilePackedWeightDescriptorV1* descriptors[]{
            &descriptor};
        float* outputs[]{output};
        call(runtime->k_quant_mmvq_multi_linear_prequantized(
            descriptors, outputs, 1, rows, k_quant_mmq_workspace.get(),
            static_cast<std::int64_t>(k_quant_mmq_workspace.bytes()),
            cuda_stream),
            "Glimmer verifier prequantized K-quant MMVQ linear");
        ++mmq_linears;
    }

    template <std::size_t Count>
    bool can_linear_mmq_group(
        const std::array<const DeviceWeight*, Count>& weights,
        std::int64_t rows,
        bool exact_verifier_rows = false) const noexcept {
        // Grouped projections reduce launch overhead for prompt/decode work,
        // but the 2..16-row speculative verifier is dominated by packed-weight
        // MMVQ occupancy. On RTX 5090, fusing its independent projections cut
        // launches yet regressed end-to-end DFlash throughput. Keep verifier
        // projections separate while retaining the profitable ordinary path.
        if (!mmq_megakernels_enabled() || exact_verifier_rows) return false;
        const bool native =
            std::all_of(weights.begin(), weights.end(), [&](const DeviceWeight* weight) {
                return weight != nullptr &&
                    ((exact_verifier_rows &&
                      can_linear_native_mmvq_rows(*weight, rows)) ||
                     (!exact_verifier_rows && can_linear_native_mmq(*weight, rows)) ||
                     can_linear_native_mmvq(*weight, rows));
            });
        if (native) return true;
        return runtime->has_experimental_mmq_multi() &&
            std::all_of(weights.begin(), weights.end(), [&](const DeviceWeight* weight) {
                return weight != nullptr && can_linear_experimental_mmq(*weight, rows);
            });
    }

    template <std::size_t Count>
    void linear_group(
        const std::array<const DeviceWeight*, Count>& weights,
        const DeviceBuffer& input,
        const std::array<DeviceBuffer*, Count>& outputs,
        std::int64_t rows,
        bool exact_verifier_rows = false,
        bool mmvq_prequantized = false) {
        if (mmvq_prequantized && (rows != 1 || exact_verifier_rows ||
                                  !runtime->has_rms_to_mmvq_handoff())) {
            throw std::runtime_error(
                "Glimmer prequantized MMVQ handoff is unavailable");
        }
        if (!can_linear_mmq_group(weights, rows, exact_verifier_rows)) {
            for (std::size_t index = 0; index < Count; ++index) {
                linear(
                    *weights[index], input, *outputs[index], rows,
                    exact_verifier_rows);
            }
            return;
        }
        std::array<const NfnNativeTilePackedWeightDescriptorV1*, Count> descriptors{};
        std::array<float*, Count> output_pointers{};
        for (std::size_t index = 0; index < Count; ++index) {
            descriptors[index] = &weights[index]->descriptor;
            output_pointers[index] = outputs[index]->template as<float>();
        }
        const bool use_native = std::all_of(
            weights.begin(), weights.end(), [&](const DeviceWeight* weight) {
                return weight != nullptr &&
                    (can_linear_native_mmq(*weight, rows) ||
                     can_linear_native_mmvq(*weight, rows));
            });
        if (use_native) {
            const int status = mmvq_prequantized
                ? runtime->k_quant_mmvq_multi_linear_prequantized(
                      descriptors.data(), output_pointers.data(),
                      static_cast<std::int64_t>(Count), rows,
                      k_quant_mmq_workspace.get(),
                      static_cast<std::int64_t>(k_quant_mmq_workspace.bytes()))
                : exact_verifier_rows
                    ? runtime->k_quant_mmvq_multi_linear(
                      descriptors.data(), input.as<const float>(),
                      output_pointers.data(), static_cast<std::int64_t>(Count),
                      rows, k_quant_mmq_workspace.get(),
                      static_cast<std::int64_t>(k_quant_mmq_workspace.bytes()))
                    : runtime->k_quant_mmq_multi_linear(
                      descriptors.data(), input.as<const float>(),
                      output_pointers.data(), static_cast<std::int64_t>(Count),
                      rows, k_quant_mmq_workspace.get(),
                      static_cast<std::int64_t>(k_quant_mmq_workspace.bytes()));
            call(status, mmvq_prequantized
                ? "Glimmer prequantized grouped K-quant MMVQ linears"
                : exact_verifier_rows
                    ? "Glimmer verifier-exact grouped K-quant MMVQ linears"
                    : "Glimmer grouped K-quant MMQ linears");
            mmq_linears += static_cast<std::int64_t>(Count);
        } else {
            call(runtime->experimental_mmq_multi_linear(
                descriptors.data(), input.as<const float>(), output_pointers.data(),
                static_cast<std::int64_t>(Count), rows),
                "Glimmer experimental grouped packed MMQ linears");
        }
    }

    template <std::size_t Count>
    void linear_group_raw(
        const std::array<const DeviceWeight*, Count>& weights,
        const DeviceBuffer& input,
        const std::array<float*, Count>& outputs,
        std::int64_t rows,
        bool exact_verifier_rows = false) {
        if (!can_linear_mmq_group(weights, rows, exact_verifier_rows) ||
            std::any_of(outputs.begin(), outputs.end(), [](const float* output) {
                return output == nullptr;
            })) {
            throw std::runtime_error(
                "Glimmer raw grouped MMQ output is unavailable");
        }
        std::array<const NfnNativeTilePackedWeightDescriptorV1*, Count>
            descriptors{};
        for (std::size_t index = 0; index < Count; ++index) {
            descriptors[index] = &weights[index]->descriptor;
        }
        const bool use_native = std::all_of(
            weights.begin(), weights.end(), [&](const DeviceWeight* weight) {
                return weight != nullptr &&
                    (can_linear_native_mmq(*weight, rows) ||
                     can_linear_native_mmvq(*weight, rows));
            });
        if (use_native) {
            const int status = exact_verifier_rows
                ? runtime->k_quant_mmvq_multi_linear(
                      descriptors.data(), input.as<const float>(), outputs.data(),
                      static_cast<std::int64_t>(Count), rows,
                      k_quant_mmq_workspace.get(),
                      static_cast<std::int64_t>(k_quant_mmq_workspace.bytes()))
                : runtime->k_quant_mmq_multi_linear(
                      descriptors.data(), input.as<const float>(), outputs.data(),
                      static_cast<std::int64_t>(Count), rows,
                      k_quant_mmq_workspace.get(),
                      static_cast<std::int64_t>(k_quant_mmq_workspace.bytes()));
            call(status, exact_verifier_rows
                ? "Glimmer verifier-exact grouped K-quant MMVQ linears to raw outputs"
                : "Glimmer grouped K-quant MMQ linears to raw outputs");
            mmq_linears += static_cast<std::int64_t>(Count);
        } else {
            call(runtime->experimental_mmq_multi_linear(
                descriptors.data(), input.as<const float>(), outputs.data(),
                static_cast<std::int64_t>(Count), rows),
                "Glimmer experimental grouped MMQ linears to raw outputs");
        }
    }

    bool linear_gated_native_mmq(
        const DeviceWeight& weight,
        const DeviceBuffer& input,
        const DeviceBuffer& gate_values,
        DeviceBuffer& output,
        std::int64_t rows,
        bool exact_verifier_rows = false) {
        // The exact verifier's row-co-scheduled MMVQ kernel is profitable, but
        // folding its sigmoid handoff into the packed projection was neutral
        // to slightly slower in matched end-to-end DFlash A/B runs. Preserve
        // the standalone composition there; keep the fused path for ordinary
        // target decode where the same-binary A/B is a clear win.
        if (!mmq_megakernels_enabled() || exact_verifier_rows ||
            !runtime->has_k_quant_mmq_gated() ||
            (!can_linear_native_mmq(weight, rows) &&
             !can_linear_native_mmvq(weight, rows))) {
            return false;
        }
        const NfnNativeTilePackedWeightDescriptorV1* descriptors[]{
            &weight.descriptor};
        float* outputs[]{output.as<float>()};
        const int status = exact_verifier_rows
            ? runtime->k_quant_mmvq_multi_linear_gated(
                  descriptors, input.as<const float>(),
                  gate_values.as<const float>(), outputs, 1, rows,
                  k_quant_mmq_workspace.get(),
                  static_cast<std::int64_t>(k_quant_mmq_workspace.bytes()))
            : runtime->k_quant_mmq_multi_linear_gated(
                  descriptors, input.as<const float>(),
                  gate_values.as<const float>(), outputs, 1, rows,
                  k_quant_mmq_workspace.get(),
                  static_cast<std::int64_t>(k_quant_mmq_workspace.bytes()));
        call(status, exact_verifier_rows
            ? "Glimmer verifier-exact sigmoid-gated K-quant MMVQ linear"
            : "Glimmer sigmoid-gated K-quant MMQ linear");
        ++mmq_linears;
        return true;
    }

    bool linear_swiglu_native_mmq(
        const DeviceWeight& weight,
        const DeviceBuffer& gate_values,
        const DeviceBuffer& up_values,
        DeviceBuffer& output,
        std::int64_t rows,
        bool exact_verifier_rows = false) {
        // See linear_gated_native_mmq: exact verifier fusion reduced launches
        // without improving measured throughput, so fail over to the exact
        // standalone handoff while retaining the profitable target path.
        if (!mmq_megakernels_enabled() || exact_verifier_rows ||
            !runtime->has_k_quant_mmq_swiglu() ||
            (!can_linear_native_mmq(weight, rows) &&
             !can_linear_native_mmvq(weight, rows))) {
            return false;
        }
        const NfnNativeTilePackedWeightDescriptorV1* descriptors[]{
            &weight.descriptor};
        float* outputs[]{output.as<float>()};
        const int status = exact_verifier_rows
            ? runtime->k_quant_mmvq_multi_linear_swiglu(
                  descriptors, gate_values.as<const float>(),
                  up_values.as<const float>(), outputs, 1, rows,
                  k_quant_mmq_workspace.get(),
                  static_cast<std::int64_t>(k_quant_mmq_workspace.bytes()))
            : runtime->k_quant_mmq_multi_linear_swiglu(
                  descriptors, gate_values.as<const float>(),
                  up_values.as<const float>(), outputs, 1, rows,
                  k_quant_mmq_workspace.get(),
                  static_cast<std::int64_t>(k_quant_mmq_workspace.bytes()));
        call(status, exact_verifier_rows
            ? "Glimmer verifier-exact SwiGLU K-quant MMVQ linear"
            : "Glimmer SwiGLU K-quant MMQ linear");
        ++mmq_linears;
        return true;
    }

    static bool is_k_quant(const DeviceWeight& weight) noexcept {
        return weight.descriptor.encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K ||
            weight.descriptor.encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K ||
            weight.descriptor.encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q6_K;
    }

    static bool mmq_megakernels_enabled() noexcept {
        // Development A/B gate. The default combines projections sharing one
        // activation quantization and folds gate/SwiGLU preparation into the
        // packed down projection. Setting the variable to an explicit false
        // value restores the exact standalone kernel composition so full-model
        // output and timing can be compared on identical artifacts.
        const char* value = std::getenv("NFN_GLIMMER_MMQ_MEGAKERNELS");
        return value == nullptr ||
            (std::strcmp(value, "0") != 0 &&
             std::strcmp(value, "false") != 0 &&
             std::strcmp(value, "off") != 0);
    }

    bool can_linear_native_mmq(
        const DeviceWeight& weight,
        std::int64_t rows) const noexcept {
        return rows >= 2 && rows <= 16 && runtime->has_k_quant_mmq() &&
            k_quant_mmq_workspace.get() != nullptr && is_k_quant(weight);
    }

    bool can_linear_native_mmvq(
        const DeviceWeight& weight,
        std::int64_t rows) const noexcept {
        return rows == 1 && runtime->has_k_quant_mmvq() &&
            k_quant_mmq_workspace.get() != nullptr && is_k_quant(weight);
    }

    bool can_linear_native_mmvq_rows(
        const DeviceWeight& weight,
        std::int64_t rows) const noexcept {
        return rows >= 2 && rows <= 16 && runtime->has_k_quant_mmvq_rows() &&
            k_quant_mmq_workspace.get() != nullptr && is_k_quant(weight);
    }

    bool can_linear_experimental_mmq(
        const DeviceWeight& weight,
        std::int64_t rows) const noexcept {
        return rows >= 2 && rows <= 16 && runtime->has_experimental_mmq() &&
            is_k_quant(weight);
    }

    bool can_linear_q8(const DeviceWeight& weight, std::int64_t rows) const noexcept {
        return rows > 0 && rows <= 16 &&
            !can_linear_native_mmvq(weight, rows) &&
            !can_linear_native_mmq(weight, rows) &&
            !can_linear_experimental_mmq(weight, rows) &&
            runtime->has_q8_linear() &&
            is_k_quant(weight);
    }

    bool prefer_fused_rms_q8(std::int64_t rows) const noexcept {
        // One-row decode benefits from the standalone quantizer's 208-way
        // block parallelism. Batched speculative verification amortizes the
        // row-local fused loop and wins by avoiding a separate launch/read.
        return rows > 1 && runtime->rms_capture_q8() != nullptr;
    }

    void quantize_q8(
        const DeviceBuffer& input,
        std::int64_t width,
        std::int64_t rows = 1) {
        if (!runtime->has_q8_linear() || rows <= 0 || rows > 16 ||
            width <= 0 || width % 32 != 0 ||
            checked_mul(rows, width, "Q8 values") >
                static_cast<std::int64_t>(q8_values.bytes()) ||
            checked_mul(checked_mul(rows, width / 32, "Q8 scale rows"), 4,
                        "Q8 scales") >
                static_cast<std::int64_t>(q8_scales.bytes())) {
            throw std::runtime_error("Glimmer CUDA Q8 activation workspace is invalid");
        }
        call(runtime->quantize_q8()(
            input.as<const float>(), q8_values.as<std::int8_t>(),
            q8_scales.as<float>(), q8_sums.as<float>(), rows, width,
            runtime->stream()), "Glimmer Q8 activation quantize");
        ++q8_quantizations;
    }

    void linear_q8(
        const DeviceWeight& weight,
        DeviceBuffer& output,
        std::int64_t rows = 1) {
        if (!can_linear_q8(weight, rows)) {
            throw std::runtime_error("Glimmer CUDA Q8 packed linear is unavailable");
        }
        call(runtime->linear_q8()(
            &weight.descriptor, q8_values.as<const std::int8_t>(),
            q8_scales.as<const float>(), q8_sums.as<const float>(), nullptr,
            output.as<float>(), rows, false), "Glimmer Q8 packed linear");
        ++q8_linears;
    }

    void linear_q8_multi_decode(
        const DeviceWeight& weight0,
        DeviceBuffer& output0,
        const DeviceWeight& weight1,
        DeviceBuffer& output1,
        const DeviceWeight* weight2 = nullptr,
        DeviceBuffer* output2 = nullptr,
        const DeviceWeight* weight3 = nullptr,
        DeviceBuffer* output3 = nullptr) {
        const bool has2 = weight2 != nullptr && output2 != nullptr;
        const bool has3 = weight3 != nullptr && output3 != nullptr;
        const std::int64_t projection_count = has3 ? 4 : (has2 ? 3 : 2);
        if (runtime->linear_q8_multi_decode() == nullptr ||
            (weight2 == nullptr) != (output2 == nullptr) ||
            (weight3 == nullptr) != (output3 == nullptr) || has3 != has2 ||
            !can_linear_q8(weight0, 1) || !can_linear_q8(weight1, 1) ||
            (has2 && !can_linear_q8(*weight2, 1)) ||
            (has3 && !can_linear_q8(*weight3, 1))) {
            throw std::runtime_error(
                "Glimmer CUDA fused decode projection geometry is invalid");
        }
        call(runtime->linear_q8_multi_decode()(
            &weight0.descriptor, &weight1.descriptor,
            has2 ? &weight2->descriptor : nullptr,
            has3 ? &weight3->descriptor : nullptr,
            q8_values.as<const std::int8_t>(), q8_scales.as<const float>(),
            q8_sums.as<const float>(), output0.as<float>(), output1.as<float>(),
            has2 ? output2->as<float>() : nullptr,
            has3 ? output3->as<float>() : nullptr, projection_count,
            runtime->stream()), "Glimmer fused Q8 decode projections");
        q8_linears += projection_count;
    }

    ArgmaxRows argmax(
        const DeviceBuffer& values,
        std::int64_t rows,
        std::int64_t width) {
        enqueue_argmax(values, rows, width);
        runtime->synchronize();
        return copy_argmax(rows);
    }

    void enqueue_argmax(
        const DeviceBuffer& values,
        std::int64_t rows,
        std::int64_t width,
        bool count = true) {
        if (runtime->argmax_rows() == nullptr || rows <= 0 || rows > 16 ||
            width <= 0 || checked_mul(rows, width, "argmax values") * 4 >
                static_cast<std::int64_t>(values.bytes())) {
            throw std::runtime_error("Glimmer CUDA device argmax is unavailable");
        }
        call(runtime->argmax_rows()(
            values.as<const float>(), argmax_indices.as<std::int64_t>(),
            argmax_values.as<float>(), rows, width, runtime->stream()),
            "Glimmer row argmax");
        if (count) {
            ++argmax_calls;
            argmax_rows_selected += rows;
        }
    }

    ArgmaxRows copy_argmax(std::int64_t rows) {
        if (rows <= 0 || rows > 16) {
            throw std::runtime_error("Glimmer CUDA argmax result row count is invalid");
        }
        ArgmaxRows result;
        result.indices.resize(checked_size(rows, "argmax result indices"));
        result.values.resize(checked_size(rows, "argmax result values"));
        runtime->copy_d2h(
            result.indices.data(), argmax_indices.get(),
            result.indices.size() * sizeof(std::int64_t));
        runtime->copy_d2h(
            result.values.data(), argmax_values.get(),
            result.values.size() * sizeof(float));
        return result;
    }

    void linear_with_lora(
        const DeviceWeight& weight,
        const DeviceLoraWeight* adapter,
        const DeviceBuffer& input,
        DeviceBuffer& output,
        std::int64_t rows = 1,
        bool exact_verifier_rows = false) {
        linear(weight, input, output, rows, exact_verifier_rows);
        if (adapter == nullptr) return;
        const DeviceLoraWeight& lora = *adapter;
        if (rows <= 0 || rows > 16 ||
            lora.a.descriptor.input_dim != weight.descriptor.input_dim ||
            lora.b.descriptor.output_dim != weight.descriptor.output_dim ||
            lora.b.descriptor.input_dim != lora.a.descriptor.output_dim ||
            !std::isfinite(lora.scaling) || !(lora.scaling > 0.0f)) {
            throw std::runtime_error("Glimmer CUDA LoRA projection geometry is invalid");
        }
        const std::int64_t rank_count = checked_mul(
            rows, lora.a.descriptor.output_dim, "LoRA rank rows");
        const std::int64_t delta_count = checked_mul(
            rows, weight.descriptor.output_dim, "LoRA output rows");
        if (checked_mul(rank_count, 4, "LoRA rank bytes") >
                static_cast<std::int64_t>(lora_rank.bytes()) ||
            checked_mul(delta_count, 4, "LoRA delta bytes") >
                static_cast<std::int64_t>(lora_delta.bytes())) {
            throw std::runtime_error("Glimmer CUDA LoRA workspace is too small");
        }
        call(runtime->linear()(
            &lora.a.descriptor, input.as<const float>(), nullptr,
            lora_rank.as<float>(), rows, false), "Glimmer LoRA A linear");
        call(runtime->linear()(
            &lora.b.descriptor, lora_rank.as<const float>(), nullptr,
            lora_delta.as<float>(), rows, false), "Glimmer LoRA B linear");
        call(runtime->scale()(
            lora_delta.as<float>(), delta_count, lora.scaling, runtime->stream()),
            "Glimmer LoRA scale");
        call(runtime->add()(
            output.as<const float>(), lora_delta.as<const float>(), output.as<float>(),
            delta_count, runtime->stream()), "Glimmer LoRA residual");
    }

    void load_lora(const HostLoraPlan& source) {
        if (!lora_layers.empty()) {
            throw std::runtime_error("Glimmer CUDA LoRA adapter is already loaded");
        }
        if (source.layers.size() != layers.size()) {
            throw std::runtime_error("Glimmer CUDA LoRA layer count is invalid");
        }
        std::vector<DeviceLoraLayer> loaded;
        loaded.reserve(source.layers.size());
        std::int64_t max_rank = 0;
        std::int64_t max_output = 0;
        auto upload = [&](const HostLoraWeight& host) {
            if (host.a.encoding != 30 || host.b.encoding != 30 ||
                host.a.rows <= 0 || host.a.cols <= 0 ||
                host.b.rows <= 0 || host.b.cols != host.a.rows ||
                !std::isfinite(host.scaling) || !(host.scaling > 0.0f)) {
                throw std::runtime_error("Glimmer CUDA LoRA weight contract is invalid");
            }
            DeviceLoraWeight result;
            result.a = upload_weight(runtime, host.a, &weight_bytes);
            result.b = upload_weight(runtime, host.b, &weight_bytes);
            result.scaling = host.scaling;
            max_rank = std::max(max_rank, host.a.rows);
            max_output = std::max(max_output, host.b.rows);
            return result;
        };
        for (const HostLoraLayer& host : source.layers) {
            DeviceLoraLayer layer;
            if (host.q) layer.q = upload(*host.q);
            if (host.k) layer.k = upload(*host.k);
            if (host.v) layer.v = upload(*host.v);
            if (host.gate) layer.gate = upload(*host.gate);
            if (host.output) layer.output = upload(*host.output);
            if (host.mlp_gate) layer.mlp_gate = upload(*host.mlp_gate);
            if (host.mlp_up) layer.mlp_up = upload(*host.mlp_up);
            if (host.mlp_down) layer.mlp_down = upload(*host.mlp_down);
            loaded.push_back(std::move(layer));
        }
        if (max_rank <= 0 || max_output <= 0) {
            throw std::runtime_error("Glimmer CUDA LoRA plan has no target projections");
        }
        const std::size_t rank_bytes = checked_size(checked_mul(
            checked_mul(16, max_rank, "LoRA rank workspace"), 4,
            "LoRA rank workspace bytes"), "LoRA rank workspace bytes");
        const std::size_t delta_bytes = checked_size(checked_mul(
            checked_mul(16, max_output, "LoRA delta workspace"), 4,
            "LoRA delta workspace bytes"), "LoRA delta workspace bytes");
        lora_rank = DeviceBuffer(runtime, rank_bytes);
        lora_delta = DeviceBuffer(runtime, delta_bytes);
        workspace += static_cast<std::int64_t>(rank_bytes + delta_bytes);
        lora_layers = std::move(loaded);
        runtime->synchronize();
    }

    void rms(
        const DeviceBuffer& input,
        const DeviceWeight* weight,
        DeviceBuffer& output,
        std::int64_t rows,
        std::int64_t width,
        float eps,
        bool centered = false) {
        call(runtime->rms()(
            input.as<const float>(), weight == nullptr ? nullptr : &weight->descriptor,
            output.as<float>(), rows, width, eps,
            weight == nullptr ? centered : weight->centered, runtime->stream()),
            "Glimmer wide RMSNorm");
    }

    void rms_capture(
        const DeviceBuffer& input,
        const DeviceWeight* weight,
        DeviceBuffer& output,
        DeviceBuffer& residual_output,
        std::int64_t rows,
        std::int64_t width,
        float eps) {
        if (!runtime->has_fused_residual_norm()) {
            runtime->copy_d2d_async(
                residual_output.get(), input.get(), checked_size(checked_mul(
                    checked_mul(rows, width, "RMS capture values"), 4,
                    "RMS capture bytes"), "RMS capture bytes"));
            rms(input, weight, output, rows, width, eps);
            return;
        }
        call(runtime->rms_capture()(
            input.as<const float>(), weight == nullptr ? nullptr : &weight->descriptor,
            output.as<float>(), residual_output.as<float>(), rows, width, eps,
            weight == nullptr ? false : weight->centered, runtime->stream()),
            "Glimmer fused RMSNorm residual capture");
    }

    void rms_capture_q8(
        const DeviceBuffer& input,
        const DeviceWeight* weight,
        DeviceBuffer& output,
        DeviceBuffer& residual_output,
        std::int64_t rows,
        std::int64_t width,
        float eps) {
        if (runtime->rms_capture_q8() == nullptr || !runtime->has_q8_linear() ||
            rows <= 0 || rows > 16 || width <= 0 || width % 32 != 0 ||
            checked_mul(rows, width, "fused RMS Q8 values") >
                static_cast<std::int64_t>(q8_values.bytes()) ||
            checked_mul(checked_mul(rows, width / 32, "fused RMS Q8 rows"), 4,
                        "fused RMS Q8 metadata") >
                static_cast<std::int64_t>(q8_scales.bytes())) {
            throw std::runtime_error(
                "Glimmer fused RMSNorm Q8 workspace is invalid");
        }
        call(runtime->rms_capture_q8()(
            input.as<const float>(), weight == nullptr ? nullptr : &weight->descriptor,
            output.as<float>(), residual_output.as<float>(),
            q8_values.as<std::int8_t>(), q8_scales.as<float>(), q8_sums.as<float>(),
            rows, width, eps, weight == nullptr ? false : weight->centered,
            runtime->stream()), "Glimmer fused RMSNorm residual capture and Q8");
        ++q8_quantizations;
    }

    void rms_add(
        const DeviceBuffer& input,
        const DeviceWeight* weight,
        const DeviceBuffer& residual_input,
        DeviceBuffer& output,
        DeviceBuffer& scratch,
        std::int64_t rows,
        std::int64_t width,
        float eps) {
        if (!runtime->has_fused_residual_norm()) {
            rms(input, weight, scratch, rows, width, eps);
            call(runtime->add()(
                residual_input.as<const float>(), scratch.as<const float>(),
                output.as<float>(), checked_mul(rows, width, "RMS residual values"),
                runtime->stream()), "Glimmer residual add");
            return;
        }
        call(runtime->rms_add()(
            input.as<const float>(), weight == nullptr ? nullptr : &weight->descriptor,
            residual_input.as<const float>(), output.as<float>(), rows, width, eps,
            weight == nullptr ? false : weight->centered, runtime->stream()),
            "Glimmer fused RMSNorm residual add");
    }

    bool dual_rms_add_capture(
        const DeviceBuffer& input,
        const DeviceWeight* first_weight,
        const DeviceBuffer& residual_input,
        DeviceBuffer& hidden_output,
        const DeviceWeight* second_weight,
        DeviceBuffer& normalized_output,
        DeviceBuffer& residual_output,
        std::int64_t rows,
        std::int64_t width,
        float first_eps,
        float second_eps) {
        if (runtime->dual_rms_add_capture() == nullptr) {
            rms_add(
                input, first_weight, residual_input, hidden_output,
                normalized_output, rows, width, first_eps);
            rms_capture(
                hidden_output, second_weight, normalized_output,
                residual_output, rows, width, second_eps);
            return false;
        }
        if (rows == 1 && mmq_megakernels_enabled() &&
            runtime->has_rms_to_mmvq_handoff()) {
            call(runtime->dual_rms_add_capture_mmvq_q8()(
                input.as<const float>(),
                first_weight == nullptr ? nullptr : &first_weight->descriptor,
                residual_input.as<const float>(), hidden_output.as<float>(),
                second_weight == nullptr ? nullptr : &second_weight->descriptor,
                normalized_output.as<float>(), residual_output.as<float>(), rows,
                width, first_eps,
                first_weight == nullptr ? false : first_weight->centered,
                second_eps,
                second_weight == nullptr ? false : second_weight->centered,
                k_quant_mmq_workspace.get(),
                static_cast<std::int64_t>(k_quant_mmq_workspace.bytes()),
                runtime->stream()),
                "Glimmer fused dual RMSNorm-to-MMVQ handoff");
            return true;
        }
        const auto cooperative_norm_supported = [](const DeviceWeight* weight) {
            return weight == nullptr ||
                weight->descriptor.encoding ==
                    NFN_NATIVE_TILE_PACKED_WEIGHT_F32;
        };
        const bool use_cooperative_batch_rms =
            rows > 1 && runtime->cooperative_batch_rms_enabled() &&
            cooperative_norm_supported(first_weight) &&
            cooperative_norm_supported(second_weight);
        DualRmsAddCaptureFn dual_rms =
            use_cooperative_batch_rms
            ? runtime->dual_rms_add_capture_cooperative_batch()
            : runtime->dual_rms_add_capture();
        call(dual_rms(
            input.as<const float>(),
            first_weight == nullptr ? nullptr : &first_weight->descriptor,
            residual_input.as<const float>(), hidden_output.as<float>(),
            second_weight == nullptr ? nullptr : &second_weight->descriptor,
            normalized_output.as<float>(), residual_output.as<float>(), rows,
            width, first_eps,
            first_weight == nullptr ? false : first_weight->centered,
            second_eps,
            second_weight == nullptr ? false : second_weight->centered,
            runtime->stream()), use_cooperative_batch_rms
                ? "Glimmer cooperative batched dual RMSNorm residual path"
                : "Glimmer fused dual RMSNorm residual path");
        return false;
    }

    Config config;
    std::shared_ptr<Runtime> runtime;
    DeviceWeight token_embedding;
    DeviceWeight final_norm;
    DeviceWeight lm_head;
    std::vector<DeviceLayer> layers;
    std::vector<DeviceLoraLayer> lora_layers;
    std::unique_ptr<VisionExecutor> vision;
    DeviceBuffer hidden;
    DeviceBuffer normalized;
    DeviceBuffer residual;
    DeviceBuffer branch;
    DeviceBuffer query;
    DeviceBuffer key;
    DeviceBuffer value;
    DeviceBuffer gate;
    DeviceBuffer attention;
    DeviceBuffer mlp_gate;
    DeviceBuffer mlp_up;
    DeviceBuffer mlp_activated;
    DeviceBuffer staged_final;
    DeviceBuffer logits_buffer;
    DeviceBuffer raw_head_hidden;
    DeviceBuffer raw_head_logits;
    DeviceBuffer argmax_indices;
    DeviceBuffer argmax_values;
    DeviceBuffer raw_embedding_token_ids;
    DeviceBuffer target_tap_staging;
    DeviceBuffer verify_normalized;
    DeviceBuffer verify_residual;
    DeviceBuffer verify_branch;
    DeviceBuffer verify_query;
    DeviceBuffer verify_key;
    DeviceBuffer verify_value;
    DeviceBuffer verify_gate;
    DeviceBuffer verify_attention;
    DeviceBuffer verify_mlp_gate;
    DeviceBuffer verify_mlp_up;
    DeviceBuffer verify_mlp_activated;
    DeviceBuffer q8_values;
    DeviceBuffer q8_scales;
    DeviceBuffer q8_sums;
    DeviceBuffer k_quant_mmq_workspace;
    std::shared_ptr<VerificationScratch> verification_scratch;
    DeviceBuffer lora_rank;
    DeviceBuffer lora_delta;
    std::int64_t weight_bytes = 0;
    std::int64_t workspace = 0;
    std::int64_t launches = 0;
    std::int64_t q8_quantizations = 0;
    std::int64_t q8_linears = 0;
    std::int64_t mmq_linears = 0;
    std::int64_t argmax_calls = 0;
    std::int64_t argmax_rows_selected = 0;
    bool closed = false;
    std::mutex mutex;
};

Model::Model(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Model::~Model() { close(); }

std::shared_ptr<Model> Model::load(const Config& config, const HostWeightPlan& weights) {
    return std::shared_ptr<Model>(new Model(std::make_unique<Impl>(config, weights)));
}

void Model::load_lora_adapter(const HostLoraPlan& weights) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed) {
        throw std::runtime_error("Glimmer CUDA model is closed");
    }
    impl_->load_lora(weights);
}

void Model::load_vision(
    const VisionConfig& config,
    const VisionHostWeightPlan& weights) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed) {
        throw std::runtime_error("Glimmer CUDA model is closed");
    }
    if (impl_->vision) {
        throw std::runtime_error("Glimmer CUDA vision weights are already loaded");
    }
    auto loaded = std::make_unique<VisionExecutor>(impl_->runtime, config, weights);
    impl_->vision = std::move(loaded);
}

std::vector<float> Model::encode_vision(
    const std::vector<float>& packed_patches,
    const std::vector<std::int64_t>& grid_thw,
    const std::atomic<bool>& cancelled) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || !impl_->vision) {
        throw std::runtime_error(
            "Glimmer CUDA vision encoding requires loaded vision weights");
    }
    return impl_->vision->encode(packed_patches, grid_thw, cancelled);
}

bool Model::has_vision() const noexcept {
    return impl_ && static_cast<bool>(impl_->vision);
}

std::int64_t Model::vision_weight_bytes() const noexcept {
    return impl_ && impl_->vision ? impl_->vision->weight_bytes() : 0;
}

std::shared_ptr<Cache> Model::create_cache() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed) throw std::runtime_error("Glimmer CUDA model is closed");
    return std::shared_ptr<Cache>(new Cache(std::make_unique<Cache::Impl>(
        impl_->runtime, impl_->config)));
}

void Model::append_token(
    std::int64_t token_id,
    std::int64_t position,
    const std::shared_ptr<Cache>& cache,
    const std::atomic<bool>& cancelled,
    const std::vector<std::int64_t>* tap_layers,
    std::vector<float>* target_taps,
    bool fast_k_quant) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    append_input_unlocked(
        token_id, nullptr, position, cache, cancelled, tap_layers, target_taps,
        fast_k_quant, nullptr, nullptr, true, true);
}

void Model::append_embedding(
    const std::vector<float>& embedding,
    std::int64_t position,
    const std::shared_ptr<Cache>& cache,
    const std::atomic<bool>& cancelled,
    const std::vector<std::int64_t>* tap_layers,
    std::vector<float>* target_taps,
    bool fast_k_quant) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    append_input_unlocked(
        -1, &embedding, position, cache, cancelled, tap_layers, target_taps,
        fast_k_quant, nullptr, nullptr, true, true);
}

void Model::append_input_unlocked(
    std::int64_t token_id,
    const std::vector<float>* embedding,
    std::int64_t position,
    const std::shared_ptr<Cache>& cache,
    const std::atomic<bool>& cancelled,
    const std::vector<std::int64_t>* tap_layers,
    std::vector<float>* target_taps,
    bool fast_k_quant,
    const std::int64_t* device_token_id,
    const std::int64_t* device_position,
    bool synchronize,
    bool commit_logical_length) {
    if (impl_->closed || !cache || !cache->impl_) {
        throw std::runtime_error("Glimmer CUDA model/cache is closed");
    }
    if (cache->impl_->runtime.get() != impl_->runtime.get() ||
        cache->impl_->logical_length != position ||
        (embedding != nullptr && device_token_id != nullptr) ||
        (embedding == nullptr && device_token_id == nullptr &&
         (token_id < 0 || token_id >= impl_->config.vocab_size)) ||
        (embedding != nullptr && embedding->size() !=
            static_cast<std::size_t>(impl_->config.model_dim)) || position < 0 ||
        position >= impl_->config.max_seq_len ||
        (device_token_id != nullptr &&
         impl_->runtime->embedding_device_i64() == nullptr) ||
        (device_position != nullptr &&
         impl_->runtime->fused_decode_attention_device_position() == nullptr)) {
        throw std::runtime_error("Glimmer CUDA token/cache position is invalid");
    }
    if ((tap_layers == nullptr) != (target_taps == nullptr)) {
        throw std::runtime_error("Glimmer CUDA tap layers/output must be supplied together");
    }
    if (tap_layers != nullptr) {
        if (tap_layers->empty() || !std::is_sorted(tap_layers->begin(), tap_layers->end()) ||
            std::adjacent_find(tap_layers->begin(), tap_layers->end()) != tap_layers->end() ||
            tap_layers->front() < 0 || tap_layers->back() >= impl_->config.num_layers) {
            throw std::runtime_error("Glimmer CUDA target tap layer list is invalid");
        }
        target_taps->clear();
        target_taps->resize(checked_size(checked_mul(
            static_cast<std::int64_t>(tap_layers->size()), impl_->config.model_dim,
            "target tap extent"), "target tap extent"));
    }
    throw_if_cancelled(cancelled);
    if (device_token_id != nullptr) {
        impl_->call(impl_->runtime->embedding_device_i64()(
            &impl_->token_embedding.descriptor, device_token_id,
            impl_->hidden.as<float>()), "Glimmer device-token embedding gather");
    } else if (embedding == nullptr) {
        impl_->call(impl_->runtime->embedding()(
            &impl_->token_embedding.descriptor, token_id, impl_->hidden.as<float>()),
            "Glimmer embedding gather");
    } else {
        impl_->runtime->copy_h2d_async(
            impl_->hidden.get(), embedding->data(), impl_->hidden.bytes());
    }
    impl_->rms(
        impl_->hidden, nullptr, impl_->normalized, 1, impl_->config.model_dim,
        impl_->config.norm_eps);
    impl_->runtime->copy_d2d_async(
        impl_->hidden.get(), impl_->normalized.get(), impl_->hidden.bytes());

    const std::int64_t query_width = checked_mul(
        impl_->config.num_heads, impl_->config.head_dim, "query width");
    const std::int64_t kv_width = checked_mul(
        impl_->config.num_kv_heads, impl_->config.head_dim, "KV width");
    bool attention_norm_ready = false;
    bool attention_mmvq_q8_ready = false;
    for (std::int64_t layer_index = 0; layer_index < impl_->config.num_layers; ++layer_index) {
        throw_if_cancelled(cancelled);
        DeviceLayer& layer = impl_->layers[static_cast<std::size_t>(layer_index)];
        const DeviceLoraLayer* adapter = impl_->lora_layers.empty()
            ? nullptr : &impl_->lora_layers[static_cast<std::size_t>(layer_index)];
        const std::array<const DeviceWeight*, 4> attention_weights{
            &layer.q, &layer.k, &layer.v, &layer.gate};
        const bool exact_attention = adapter == nullptr &&
            impl_->can_linear_mmq_group(attention_weights, 1);
        const bool fast_attention = fast_k_quant && adapter == nullptr &&
            impl_->can_linear_q8(layer.q, 1) && impl_->can_linear_q8(layer.k, 1) &&
            impl_->can_linear_q8(layer.v, 1) && impl_->can_linear_q8(layer.gate, 1);
        const bool fused_attention_q8 = fast_attention &&
            impl_->prefer_fused_rms_q8(1);
        const bool attention_input_mmvq_prequantized =
            attention_norm_ready && attention_mmvq_q8_ready;
        if (!attention_norm_ready) {
            if (fused_attention_q8) {
                impl_->rms_capture_q8(
                    impl_->hidden, &layer.input_norm, impl_->normalized,
                    impl_->residual, 1, impl_->config.model_dim,
                    impl_->config.norm_eps);
            } else {
                impl_->rms_capture(
                    impl_->hidden, &layer.input_norm, impl_->normalized,
                    impl_->residual, 1, impl_->config.model_dim,
                    impl_->config.norm_eps);
            }
        }
        attention_norm_ready = false;
        attention_mmvq_q8_ready = false;
        if (exact_attention) {
            impl_->linear_group(
                attention_weights, impl_->normalized,
                std::array<DeviceBuffer*, 4>{
                    &impl_->query, &impl_->key, &impl_->value, &impl_->gate},
                1, false, attention_input_mmvq_prequantized);
        } else if (fast_attention) {
            if (!fused_attention_q8) {
                impl_->quantize_q8(impl_->normalized, impl_->config.model_dim);
            }
            if (impl_->runtime->linear_q8_multi_decode() != nullptr) {
                impl_->linear_q8_multi_decode(
                    layer.q, impl_->query, layer.k, impl_->key,
                    &layer.v, &impl_->value, &layer.gate, &impl_->gate);
            } else {
                impl_->linear_q8(layer.q, impl_->query);
                impl_->linear_q8(layer.k, impl_->key);
                impl_->linear_q8(layer.v, impl_->value);
                impl_->linear_q8(layer.gate, impl_->gate);
            }
        } else {
            impl_->linear_with_lora(
                layer.q, adapter && adapter->q ? &*adapter->q : nullptr,
                impl_->normalized, impl_->query);
            impl_->linear_with_lora(
                layer.k, adapter && adapter->k ? &*adapter->k : nullptr,
                impl_->normalized, impl_->key);
            impl_->linear_with_lora(
                layer.v, adapter && adapter->v ? &*adapter->v : nullptr,
                impl_->normalized, impl_->value);
            impl_->linear_with_lora(
                layer.gate, adapter && adapter->gate ? &*adapter->gate : nullptr,
                impl_->normalized, impl_->gate);
        }
        const bool local = layer_index % 4 != 3;
        Cache::Impl::Layer& cache_layer =
            cache->impl_->layers[static_cast<std::size_t>(layer_index)];
        const std::int64_t first_key_position = local
            ? std::max<std::int64_t>(
                  0, position - impl_->config.sliding_window + 1)
            : 0;
        const float attention_scale =
            1.0f / std::sqrt(static_cast<float>(impl_->config.head_dim));
        const std::uint32_t rope_layout = impl_->config.gguf_interleaved
            ? NFN_NATIVE_TILE_GLIMMER_ROPE_INTERLEAVED
            : NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT;
        if (impl_->runtime->fused_decode_attention() != nullptr) {
            NfnNativeTileGlimmerFusedDecodeAttentionDescriptorV1 fused{};
            fused.struct_size = sizeof(fused);
            fused.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
            fused.query = impl_->query.as<float>();
            fused.key = impl_->key.as<float>();
            fused.current_value = impl_->value.as<const float>();
            fused.key_cache_bf16 = cache_layer.keys.as<std::uint16_t>();
            fused.value_cache_bf16 = cache_layer.values.as<std::uint16_t>();
            fused.output = impl_->attention.as<float>();
            if (layer.q_norm) {
                fused.query_norm_weight = layer.q_norm->descriptor;
                fused.has_query_norm_weight = 1;
                fused.query_norm_centered = layer.q_norm->centered ? 1 : 0;
            }
            if (layer.k_norm) {
                fused.key_norm_weight = layer.k_norm->descriptor;
                fused.has_key_norm_weight = 1;
                fused.key_norm_centered = layer.k_norm->centered ? 1 : 0;
            }
            fused.query_heads = impl_->config.num_heads;
            fused.kv_heads = impl_->config.num_kv_heads;
            fused.head_dim = impl_->config.head_dim;
            fused.position = position;
            fused.first_key_position = first_key_position;
            fused.cache_capacity = cache_layer.capacity;
            fused.cache_row_stride = kv_width;
            fused.norm_eps = impl_->config.norm_eps;
            fused.query_scale = impl_->config.gguf_interleaved
                ? 1.0f
                : impl_->config.q_scale_factor;
            fused.rope_theta = impl_->config.rope_theta;
            fused.attention_scale = attention_scale;
            fused.rope_layout = rope_layout;
            fused.apply_rope = local ? 1 : 0;
            fused.cuda_stream = impl_->runtime->stream();
            if (device_position != nullptr) {
                impl_->call(
                    impl_->runtime->fused_decode_attention_device_position()(
                        &fused, device_position,
                        local ? impl_->config.sliding_window
                              : impl_->config.max_seq_len),
                    "Glimmer graph-position fused decode attention/cache");
            } else {
                impl_->call(
                    impl_->runtime->fused_decode_attention()(&fused),
                    "Glimmer fused decode attention/cache");
            }
        } else {
            if (impl_->runtime->qk_norm_scale_rope() != nullptr) {
                impl_->call(impl_->runtime->qk_norm_scale_rope()(
                    impl_->query.as<float>(), impl_->key.as<float>(),
                    layer.q_norm ? &layer.q_norm->descriptor : nullptr,
                    layer.k_norm ? &layer.k_norm->descriptor : nullptr,
                    impl_->config.num_heads, impl_->config.num_kv_heads,
                    impl_->config.head_dim, impl_->config.norm_eps,
                    layer.q_norm ? layer.q_norm->centered : false,
                    layer.k_norm ? layer.k_norm->centered : false,
                    impl_->config.gguf_interleaved
                        ? 1.0f : impl_->config.q_scale_factor,
                    position, impl_->config.rope_theta, rope_layout, local,
                    impl_->runtime->stream()),
                    "Glimmer fused QK norm/scale/RoPE");
            } else {
                impl_->rms(
                    impl_->query, layer.q_norm ? &*layer.q_norm : nullptr,
                    impl_->query, impl_->config.num_heads,
                    impl_->config.head_dim, impl_->config.norm_eps);
                impl_->rms(
                    impl_->key, layer.k_norm ? &*layer.k_norm : nullptr,
                    impl_->key, impl_->config.num_kv_heads,
                    impl_->config.head_dim, impl_->config.norm_eps);
                if (!impl_->config.gguf_interleaved) {
                    impl_->call(impl_->runtime->scale()(
                        impl_->query.as<float>(), query_width,
                        impl_->config.q_scale_factor,
                        impl_->runtime->stream()), "Glimmer query scale");
                }
                if (local) {
                    impl_->call(impl_->runtime->rope()(
                        impl_->query.as<float>(), impl_->key.as<float>(),
                        impl_->config.num_heads, impl_->config.num_kv_heads,
                        impl_->config.head_dim, position,
                        impl_->config.rope_theta, rope_layout,
                        impl_->runtime->stream()), "Glimmer positioned RoPE");
                }
            }
            NfnNativeTileGlimmerGqaDecodeDescriptorV1 attention{};
            attention.struct_size = sizeof(attention);
            attention.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
            attention.query = impl_->query.as<const float>();
            attention.current_key = impl_->key.as<const float>();
            attention.current_value = impl_->value.as<const float>();
            attention.key_cache_bf16 =
                cache_layer.keys.as<const std::uint16_t>();
            attention.value_cache_bf16 =
                cache_layer.values.as<const std::uint16_t>();
            attention.output = impl_->attention.as<float>();
            attention.query_heads = impl_->config.num_heads;
            attention.kv_heads = impl_->config.num_kv_heads;
            attention.head_dim = impl_->config.head_dim;
            attention.position = position;
            attention.first_key_position = first_key_position;
            attention.cache_capacity = cache_layer.capacity;
            attention.cache_row_stride = kv_width;
            attention.scale = attention_scale;
            attention.cuda_stream = impl_->runtime->stream();
            impl_->call(
                impl_->runtime->gqa()(&attention), "Glimmer GQA decode");

            // The cache's logical length remains unchanged until the entire
            // token succeeds, so this row is still transactionally invisible.
            // A local-ring retry cannot observe the overwritten row; global
            // storage writes a fresh slot.
            NfnNativeTileGlimmerCacheCommitDescriptorV1 commit{};
            commit.struct_size = sizeof(commit);
            commit.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
            commit.current_key = impl_->key.as<const float>();
            commit.current_value = impl_->value.as<const float>();
            commit.key_cache_bf16 = cache_layer.keys.as<std::uint16_t>();
            commit.value_cache_bf16 = cache_layer.values.as<std::uint16_t>();
            commit.kv_heads = impl_->config.num_kv_heads;
            commit.head_dim = impl_->config.head_dim;
            commit.position = position;
            commit.cache_capacity = cache_layer.capacity;
            commit.cache_row_stride = kv_width;
            commit.cuda_stream = impl_->runtime->stream();
            impl_->call(
                impl_->runtime->cache_commit()(&commit),
                "Glimmer cache commit");
        }
        const bool fused_attention_output = adapter == nullptr &&
            impl_->linear_gated_native_mmq(
                layer.output, impl_->attention, impl_->gate, impl_->branch, 1);
        if (!fused_attention_output) {
            impl_->call(impl_->runtime->gate()(
                impl_->attention.as<const float>(), impl_->gate.as<const float>(),
                impl_->attention.as<float>(), query_width, impl_->runtime->stream()),
                "Glimmer attention gate");
            const bool fast_attention_output = fast_k_quant &&
                tap_layers == nullptr && adapter == nullptr &&
                impl_->can_linear_q8(layer.output, 1);
            if (fast_attention_output) {
                impl_->quantize_q8(impl_->attention, query_width);
                impl_->linear_q8(layer.output, impl_->branch);
            } else {
                impl_->linear_with_lora(
                    layer.output,
                    adapter && adapter->output ? &*adapter->output : nullptr,
                    impl_->attention, impl_->branch);
            }
        }
        const bool mlp_input_mmvq_prequantized = impl_->dual_rms_add_capture(
            impl_->branch, &layer.post_attention_norm, impl_->residual,
            impl_->hidden, &layer.pre_feedforward_norm, impl_->normalized,
            impl_->residual, 1, impl_->config.model_dim,
            impl_->config.post_norm_eps, impl_->config.norm_eps);
        const bool fast_mlp = fast_k_quant && adapter == nullptr &&
            impl_->can_linear_q8(layer.mlp_gate, 1) &&
            impl_->can_linear_q8(layer.mlp_up, 1);
        const std::array<const DeviceWeight*, 2> mlp_weights{
            &layer.mlp_gate, &layer.mlp_up};
        const bool exact_mlp = adapter == nullptr &&
            impl_->can_linear_mmq_group(mlp_weights, 1);
        if (exact_mlp) {
            impl_->linear_group(
                mlp_weights, impl_->normalized,
                std::array<DeviceBuffer*, 2>{&impl_->mlp_gate, &impl_->mlp_up},
                1, false, mlp_input_mmvq_prequantized);
        } else if (fast_mlp) {
            impl_->quantize_q8(impl_->normalized, impl_->config.model_dim);
            if (impl_->runtime->linear_q8_multi_decode() != nullptr) {
                impl_->linear_q8_multi_decode(
                    layer.mlp_gate, impl_->mlp_gate,
                    layer.mlp_up, impl_->mlp_up);
            } else {
                impl_->linear_q8(layer.mlp_gate, impl_->mlp_gate);
                impl_->linear_q8(layer.mlp_up, impl_->mlp_up);
            }
        } else {
            impl_->linear_with_lora(
                layer.mlp_gate, adapter && adapter->mlp_gate ? &*adapter->mlp_gate : nullptr,
                impl_->normalized, impl_->mlp_gate);
            impl_->linear_with_lora(
                layer.mlp_up, adapter && adapter->mlp_up ? &*adapter->mlp_up : nullptr,
                impl_->normalized, impl_->mlp_up);
        }
        const bool fused_mlp_down = adapter == nullptr &&
            impl_->linear_swiglu_native_mmq(
                layer.mlp_down, impl_->mlp_gate, impl_->mlp_up,
                impl_->branch, 1);
        if (!fused_mlp_down) {
            impl_->call(impl_->runtime->swiglu()(
                impl_->mlp_gate.as<const float>(), impl_->mlp_up.as<const float>(),
                impl_->mlp_activated.as<float>(), impl_->config.intermediate_dim,
                impl_->runtime->stream()), "Glimmer SwiGLU");
            const bool fast_mlp_down = fast_k_quant && tap_layers == nullptr &&
                adapter == nullptr && impl_->can_linear_q8(layer.mlp_down, 1);
            if (fast_mlp_down) {
                impl_->quantize_q8(
                    impl_->mlp_activated, impl_->config.intermediate_dim);
                impl_->linear_q8(layer.mlp_down, impl_->branch);
            } else {
                impl_->linear_with_lora(
                    layer.mlp_down,
                    adapter && adapter->mlp_down ? &*adapter->mlp_down : nullptr,
                    impl_->mlp_activated, impl_->branch);
            }
        }
        if (layer_index + 1 < impl_->config.num_layers) {
            DeviceLayer& next_layer =
                impl_->layers[static_cast<std::size_t>(layer_index + 1)];
            attention_mmvq_q8_ready = impl_->dual_rms_add_capture(
                impl_->branch, &layer.post_feedforward_norm, impl_->residual,
                impl_->hidden, &next_layer.input_norm, impl_->normalized,
                impl_->residual, 1, impl_->config.model_dim,
                impl_->config.post_norm_eps, impl_->config.norm_eps);
            attention_norm_ready = true;
        } else {
            impl_->rms_add(
                impl_->branch, &layer.post_feedforward_norm, impl_->residual,
                impl_->hidden, impl_->normalized, 1, impl_->config.model_dim,
                impl_->config.post_norm_eps);
        }
        if (tap_layers != nullptr && std::binary_search(
                tap_layers->begin(), tap_layers->end(), layer_index)) {
            const auto found = std::lower_bound(
                tap_layers->begin(), tap_layers->end(), layer_index);
            const std::int64_t tap_index = static_cast<std::int64_t>(
                std::distance(tap_layers->begin(), found));
            impl_->runtime->copy_d2d_async(
                impl_->target_tap_staging.as<float>() +
                    tap_index * impl_->config.model_dim,
                impl_->hidden.get(), impl_->hidden.bytes());
        }
    }
    impl_->rms(
        impl_->hidden, &impl_->final_norm, impl_->staged_final, 1,
        impl_->config.model_dim, impl_->config.norm_eps);
    throw_if_cancelled(cancelled);

    impl_->runtime->copy_d2d_async(
        cache->impl_->final_hidden.get(), impl_->staged_final.get(), impl_->staged_final.bytes());
    if (synchronize) impl_->runtime->synchronize();
    if (target_taps != nullptr) {
        impl_->runtime->copy_d2h(
            target_taps->data(), impl_->target_tap_staging.get(),
            checked_size(checked_mul(
                static_cast<std::int64_t>(target_taps->size()), 4,
                "target tap output bytes"), "target tap output bytes"));
    }
    if (commit_logical_length) cache->impl_->logical_length = position + 1;
}

std::shared_ptr<Verification> Model::verify_tokens(
    const std::vector<std::int64_t>& token_ids,
    std::int64_t position,
    const std::shared_ptr<Cache>& cache,
    const std::atomic<bool>& cancelled,
    const std::vector<std::int64_t>* tap_layers,
    bool compute_logits,
    bool compute_argmax,
    bool fast_k_quant,
    bool copy_taps_to_host,
    bool exact_verifier_rows) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const std::int64_t rows = static_cast<std::int64_t>(token_ids.size());
    if (impl_->closed || !cache || !cache->impl_ ||
        cache->impl_->runtime.get() != impl_->runtime.get() ||
        cache->impl_->logical_length != position || rows <= 0 || rows > 16 ||
        position < 0 || position + rows > impl_->config.max_seq_len ||
        std::any_of(token_ids.begin(), token_ids.end(), [&](std::int64_t token) {
            return token < 0 || token >= impl_->config.vocab_size;
        })) {
        throw std::runtime_error("Glimmer CUDA verification block/cache is invalid");
    }
    if ((compute_logits && compute_argmax) ||
        (compute_argmax && impl_->runtime->argmax_rows() == nullptr)) {
        throw std::runtime_error("Glimmer CUDA verification output mode is invalid");
    }
    const bool k_quant_verifier = exact_verifier_rows &&
        Impl::is_k_quant(impl_->lm_head);
    if (k_quant_verifier && !impl_->runtime->has_k_quant_mmvq_rows()) {
        throw std::runtime_error(
            "Glimmer K-quant speculative verification requires the batched MMVQ ABI");
    }
    exact_verifier_rows = k_quant_verifier;
    if (tap_layers != nullptr &&
        (tap_layers->empty() || !std::is_sorted(tap_layers->begin(), tap_layers->end()) ||
         std::adjacent_find(tap_layers->begin(), tap_layers->end()) != tap_layers->end() ||
         tap_layers->front() < 0 || tap_layers->back() >= impl_->config.num_layers)) {
        throw std::runtime_error("Glimmer CUDA verification tap layers are invalid");
    }
    throw_if_cancelled(cancelled);
    const std::int64_t d = impl_->config.model_dim;
    const std::int64_t query_width = checked_mul(
        impl_->config.num_heads, impl_->config.head_dim, "verification query width");
    const std::int64_t kv_width = checked_mul(
        impl_->config.num_kv_heads, impl_->config.head_dim, "verification KV width");
    const auto floats = [&](std::int64_t count, const char* label) {
        return checked_size(checked_mul(count, 4, label), label);
    };
    const std::int64_t hidden_count = checked_mul(rows, d, "verification hidden");
    const std::size_t hidden_bytes = floats(hidden_count, "verification hidden");
    const std::size_t kv_bytes = floats(checked_mul(
        rows, kv_width, "verification KV"), "verification KV");
    const std::size_t logits_bytes = floats(checked_mul(
        rows, impl_->config.vocab_size, "verification logits"),
        "verification logits");
    DeviceBuffer& hidden = impl_->raw_head_hidden;
    DeviceBuffer& normalized = impl_->verify_normalized;
    DeviceBuffer& residual = impl_->verify_residual;
    DeviceBuffer& branch = impl_->verify_branch;
    DeviceBuffer& query = impl_->verify_query;
    DeviceBuffer& key = impl_->verify_key;
    DeviceBuffer& value = impl_->verify_value;
    DeviceBuffer& gate = impl_->verify_gate;
    DeviceBuffer& attention = impl_->verify_attention;
    DeviceBuffer& mlp_gate = impl_->verify_mlp_gate;
    DeviceBuffer& mlp_up = impl_->verify_mlp_up;
    DeviceBuffer& mlp_activated = impl_->verify_mlp_activated;
    DeviceBuffer& logits = impl_->raw_head_logits;
    const std::int64_t tap_count = tap_layers == nullptr
        ? 0 : static_cast<std::int64_t>(tap_layers->size());
    bool scratch_lease = false;
    std::shared_ptr<VerificationScratch> scratch =
        impl_->acquire_verification_scratch(rows, tap_count, &scratch_lease);
    auto verification = std::shared_ptr<Verification>(new Verification(
        std::make_unique<Verification::Impl>(
            impl_->runtime, impl_->config, position, rows,
            tap_count, compute_logits, compute_argmax, copy_taps_to_host,
            std::move(scratch), scratch_lease)));

    std::vector<std::int32_t> embedding_ids;
    embedding_ids.reserve(token_ids.size());
    for (std::int64_t token : token_ids) {
        embedding_ids.push_back(static_cast<std::int32_t>(token));
    }
    impl_->runtime->copy_h2d_async(
        impl_->raw_embedding_token_ids.get(), embedding_ids.data(),
        embedding_ids.size() * sizeof(std::int32_t));
    impl_->call(impl_->runtime->embedding_batch()(
        &impl_->token_embedding.descriptor,
        impl_->raw_embedding_token_ids.as<const std::int32_t>(),
        hidden.as<float>(), rows), "Glimmer verification batched embedding");
    impl_->rms(
        hidden, nullptr, normalized, rows, d, impl_->config.norm_eps);
    impl_->runtime->copy_d2d_async(hidden.get(), normalized.get(), hidden_bytes);

    bool attention_norm_ready = false;
    for (std::int64_t layer_index = 0; layer_index < impl_->config.num_layers; ++layer_index) {
        throw_if_cancelled(cancelled);
        DeviceLayer& layer = impl_->layers[static_cast<std::size_t>(layer_index)];
        const DeviceLoraLayer* adapter = impl_->lora_layers.empty()
            ? nullptr : &impl_->lora_layers[static_cast<std::size_t>(layer_index)];
        const bool fast_attention = fast_k_quant && adapter == nullptr &&
            impl_->can_linear_q8(layer.q, rows) &&
            impl_->can_linear_q8(layer.k, rows) &&
            impl_->can_linear_q8(layer.v, rows) &&
            impl_->can_linear_q8(layer.gate, rows);
        const std::array<const DeviceWeight*, 4> attention_weights{
            &layer.q, &layer.k, &layer.v, &layer.gate};
        const bool experimental_attention = adapter == nullptr &&
            impl_->can_linear_mmq_group(
                attention_weights, rows, exact_verifier_rows);
        const bool reuse_exact_attention_q8 = adapter == nullptr &&
            exact_verifier_rows &&
            impl_->runtime->has_k_quant_mmvq_prequantized() &&
            std::all_of(
                attention_weights.begin(), attention_weights.end(),
                [&](const DeviceWeight* weight) {
                    return weight != nullptr &&
                        impl_->can_linear_native_mmvq_rows(*weight, rows);
                });
        float* staged_key = verification->impl_->scratch->staged_keys.as<float>() +
            layer_index * rows * kv_width;
        float* staged_value = verification->impl_->scratch->staged_values.as<float>() +
            layer_index * rows * kv_width;
        const bool direct_staged_attention =
            (experimental_attention || reuse_exact_attention_q8) &&
            impl_->runtime->qk_norm_scale_rope_batch() != nullptr;
        float* key_values = direct_staged_attention ? staged_key : key.as<float>();
        float* value_values =
            direct_staged_attention ? staged_value : value.as<float>();
        const bool fused_attention_q8 = fast_attention &&
            impl_->prefer_fused_rms_q8(rows);
        bool attention_q8_ready = false;
        if (!attention_norm_ready) {
            if (fused_attention_q8) {
                impl_->rms_capture_q8(
                    hidden, &layer.input_norm, normalized, residual, rows, d,
                    impl_->config.norm_eps);
                attention_q8_ready = true;
            } else {
                impl_->rms_capture(
                    hidden, &layer.input_norm, normalized, residual, rows, d,
                    impl_->config.norm_eps);
            }
        } else if (fast_attention) {
            impl_->quantize_q8(normalized, d, rows);
            attention_q8_ready = true;
        }
        attention_norm_ready = false;
        if (reuse_exact_attention_q8) {
            // The four projections consume the identical normalized rows.
            // Quantize once for Q, then preserve and reuse that exact Q8_1
            // workspace for K/V/gate without changing projection reduction
            // order or fusing independent output kernels. K/V land directly
            // in transactional cache staging, avoiding two D2D copies/layer.
            impl_->linear(layer.q, normalized, query, rows, true);
            const bool overlap =
                impl_->runtime->verifier_projection_overlap_enabled();
            if (overlap) impl_->runtime->begin_verifier_projection_overlap();
            impl_->linear_mmvq_prequantized_raw(
                layer.k, key_values, rows,
                overlap ? impl_->runtime->verifier_aux_stream(0) : nullptr);
            impl_->linear_mmvq_prequantized_raw(
                layer.v, value_values, rows,
                overlap ? impl_->runtime->verifier_aux_stream(1) : nullptr);
            impl_->linear_mmvq_prequantized(layer.gate, gate, rows);
            if (overlap) impl_->runtime->end_verifier_projection_overlap();
        } else if (direct_staged_attention) {
            impl_->linear_group_raw(
                attention_weights, normalized,
                std::array<float*, 4>{
                    query.as<float>(), key_values, value_values, gate.as<float>()},
                rows, exact_verifier_rows);
        } else if (experimental_attention) {
            impl_->linear_group(
                attention_weights, normalized,
                std::array<DeviceBuffer*, 4>{&query, &key, &value, &gate}, rows,
                exact_verifier_rows);
        } else if (fast_attention) {
            if (!attention_q8_ready) impl_->quantize_q8(normalized, d, rows);
            impl_->linear_q8(layer.q, query, rows);
            impl_->linear_q8(layer.k, key, rows);
            impl_->linear_q8(layer.v, value, rows);
            impl_->linear_q8(layer.gate, gate, rows);
        } else {
            impl_->linear_with_lora(
                layer.q, adapter && adapter->q ? &*adapter->q : nullptr,
                normalized, query, rows, exact_verifier_rows);
            impl_->linear_with_lora(
                layer.k, adapter && adapter->k ? &*adapter->k : nullptr,
                normalized, key, rows, exact_verifier_rows);
            impl_->linear_with_lora(
                layer.v, adapter && adapter->v ? &*adapter->v : nullptr,
                normalized, value, rows, exact_verifier_rows);
            impl_->linear_with_lora(
                layer.gate, adapter && adapter->gate ? &*adapter->gate : nullptr,
                normalized, gate, rows, exact_verifier_rows);
        }
        const bool local = layer_index % 4 != 3;
        if (impl_->runtime->qk_norm_scale_rope_batch() != nullptr) {
            impl_->call(impl_->runtime->qk_norm_scale_rope_batch()(
                query.as<float>(), key_values,
                layer.q_norm ? &layer.q_norm->descriptor : nullptr,
                layer.k_norm ? &layer.k_norm->descriptor : nullptr,
                rows,
                impl_->config.num_heads, impl_->config.num_kv_heads,
                impl_->config.head_dim, impl_->config.norm_eps,
                layer.q_norm ? layer.q_norm->centered : false,
                layer.k_norm ? layer.k_norm->centered : false,
                impl_->config.gguf_interleaved
                    ? 1.0f : impl_->config.q_scale_factor,
                position, impl_->config.rope_theta,
                impl_->config.gguf_interleaved
                    ? NFN_NATIVE_TILE_GLIMMER_ROPE_INTERLEAVED
                    : NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT,
                local, impl_->runtime->stream()),
                "Glimmer verification fused batched QK norm/scale/RoPE");
        } else {
            impl_->rms(
                query, layer.q_norm ? &*layer.q_norm : nullptr, query,
                checked_mul(rows, impl_->config.num_heads,
                            "verification Q norm rows"),
                impl_->config.head_dim, impl_->config.norm_eps);
            impl_->rms(
                key, layer.k_norm ? &*layer.k_norm : nullptr, key,
                checked_mul(rows, impl_->config.num_kv_heads,
                            "verification K norm rows"),
                impl_->config.head_dim, impl_->config.norm_eps);
            if (!impl_->config.gguf_interleaved) {
                impl_->call(impl_->runtime->scale()(
                    query.as<float>(),
                    checked_mul(rows, query_width, "verification Q scale"),
                    impl_->config.q_scale_factor, impl_->runtime->stream()),
                    "Glimmer verification query scale");
            }
            if (local) {
                impl_->call(impl_->runtime->rope_batch()(
                    query.as<float>(), key.as<float>(), rows,
                    impl_->config.num_heads, impl_->config.num_kv_heads,
                    impl_->config.head_dim, position, impl_->config.rope_theta,
                    impl_->config.gguf_interleaved
                        ? NFN_NATIVE_TILE_GLIMMER_ROPE_INTERLEAVED
                        : NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT,
                    false, impl_->runtime->stream()),
                    "Glimmer verification batched positioned RoPE");
            }
        }
        if (!direct_staged_attention) {
            impl_->runtime->copy_d2d_async(staged_key, key.get(), kv_bytes);
            impl_->runtime->copy_d2d_async(staged_value, value.get(), kv_bytes);
        }
        Cache::Impl::Layer& cache_layer =
            cache->impl_->layers[static_cast<std::size_t>(layer_index)];
        NfnNativeTileDFlashBlockAttentionDescriptorV1 descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
        descriptor.flags = NFN_NATIVE_TILE_BLOCK_ATTENTION_CAUSAL;
        descriptor.query = query.as<const float>();
        descriptor.block_key = key_values;
        descriptor.block_value = value_values;
        descriptor.key_cache_bf16 = cache_layer.keys.as<const std::uint16_t>();
        descriptor.value_cache_bf16 = cache_layer.values.as<const std::uint16_t>();
        descriptor.output = attention.as<float>();
        descriptor.query_rows = rows;
        descriptor.block_rows = rows;
        descriptor.query_heads = impl_->config.num_heads;
        descriptor.kv_heads = impl_->config.num_kv_heads;
        descriptor.head_dim = impl_->config.head_dim;
        descriptor.context_length = position;
        descriptor.sliding_window = local
            ? impl_->config.sliding_window : impl_->config.max_seq_len;
        descriptor.cache_capacity = cache_layer.capacity;
        descriptor.cache_row_stride = kv_width;
        descriptor.scale = 1.0f / std::sqrt(static_cast<float>(impl_->config.head_dim));
        descriptor.cuda_stream = impl_->runtime->stream();
        const std::int64_t maximum_attention_keys = std::min(
            descriptor.sliding_window,
            descriptor.context_length + descriptor.block_rows);
        const bool split_short_attention =
            impl_->runtime->short_attention_split_enabled() &&
            rows < 16 && maximum_attention_keys <= 128;
        if (split_short_attention) {
            impl_->call(impl_->runtime->dflash_attention_short_split()(
                &descriptor,
                verification->impl_->scratch->attention_scores.as<float>(),
                static_cast<std::int64_t>(
                    verification->impl_->scratch->attention_scores.bytes())),
                "Glimmer split short causal block verification attention");
        } else {
            impl_->call(impl_->runtime->dflash_attention()(&descriptor),
                        "Glimmer causal block verification attention");
        }
        const bool fused_gated_output = adapter == nullptr &&
            impl_->linear_gated_native_mmq(
                layer.output, attention, gate, branch, rows,
                exact_verifier_rows);
        if (!fused_gated_output) {
            impl_->call(impl_->runtime->gate()(
                attention.as<const float>(), gate.as<const float>(),
                attention.as<float>(),
                checked_mul(rows, query_width, "verification gate"),
                impl_->runtime->stream()),
                "Glimmer verification attention gate");
            if (adapter == nullptr) {
                impl_->linear(
                    layer.output, attention, branch, rows,
                    exact_verifier_rows);
            } else {
                impl_->linear_with_lora(
                    layer.output,
                    adapter && adapter->output ? &*adapter->output : nullptr,
                    attention, branch, rows, exact_verifier_rows);
            }
        }
        impl_->dual_rms_add_capture(
            branch, &layer.post_attention_norm, residual, hidden,
            &layer.pre_feedforward_norm, normalized, residual, rows, d,
            impl_->config.post_norm_eps, impl_->config.norm_eps);
        const bool fast_mlp = fast_k_quant && adapter == nullptr &&
            impl_->can_linear_q8(layer.mlp_gate, rows) &&
            impl_->can_linear_q8(layer.mlp_up, rows);
        const std::array<const DeviceWeight*, 2> mlp_weights{
            &layer.mlp_gate, &layer.mlp_up};
        const bool experimental_mlp = adapter == nullptr &&
            impl_->can_linear_mmq_group(
                mlp_weights, rows, exact_verifier_rows);
        const bool reuse_exact_mlp_q8 = adapter == nullptr &&
            exact_verifier_rows &&
            impl_->runtime->has_k_quant_mmvq_prequantized() &&
            std::all_of(
                mlp_weights.begin(), mlp_weights.end(),
                [&](const DeviceWeight* weight) {
                    return weight != nullptr &&
                        impl_->can_linear_native_mmvq_rows(*weight, rows);
                });
        if (reuse_exact_mlp_q8) {
            impl_->linear(layer.mlp_gate, normalized, mlp_gate, rows, true);
            impl_->linear_mmvq_prequantized(layer.mlp_up, mlp_up, rows);
        } else if (experimental_mlp) {
            impl_->linear_group(
                mlp_weights, normalized,
                std::array<DeviceBuffer*, 2>{&mlp_gate, &mlp_up}, rows,
                exact_verifier_rows);
        } else if (fast_mlp) {
            impl_->quantize_q8(normalized, d, rows);
            impl_->linear_q8(layer.mlp_gate, mlp_gate, rows);
            impl_->linear_q8(layer.mlp_up, mlp_up, rows);
        } else {
            impl_->linear_with_lora(
                layer.mlp_gate,
                adapter && adapter->mlp_gate ? &*adapter->mlp_gate : nullptr,
                normalized, mlp_gate, rows, exact_verifier_rows);
            impl_->linear_with_lora(
                layer.mlp_up,
                adapter && adapter->mlp_up ? &*adapter->mlp_up : nullptr,
                normalized, mlp_up, rows, exact_verifier_rows);
        }
        const bool fused_swiglu_down = adapter == nullptr &&
            impl_->linear_swiglu_native_mmq(
                layer.mlp_down, mlp_gate, mlp_up, branch, rows,
                exact_verifier_rows);
        if (!fused_swiglu_down) {
            impl_->call(impl_->runtime->swiglu()(
                mlp_gate.as<const float>(), mlp_up.as<const float>(),
                mlp_activated.as<float>(), checked_mul(
                    rows, impl_->config.intermediate_dim,
                    "verification SwiGLU"),
                impl_->runtime->stream()), "Glimmer verification SwiGLU");
            if (adapter == nullptr) {
                impl_->linear(
                    layer.mlp_down, mlp_activated, branch, rows,
                    exact_verifier_rows);
            } else {
                impl_->linear_with_lora(
                    layer.mlp_down,
                    adapter && adapter->mlp_down ? &*adapter->mlp_down : nullptr,
                    mlp_activated, branch, rows, exact_verifier_rows);
            }
        }
        if (layer_index + 1 < impl_->config.num_layers) {
            DeviceLayer& next_layer =
                impl_->layers[static_cast<std::size_t>(layer_index + 1)];
            impl_->dual_rms_add_capture(
                branch, &layer.post_feedforward_norm, residual, hidden,
                &next_layer.input_norm, normalized, residual, rows, d,
                impl_->config.post_norm_eps, impl_->config.norm_eps);
            attention_norm_ready = true;
        } else {
            impl_->rms_add(
                branch, &layer.post_feedforward_norm, residual, hidden,
                normalized, rows, d, impl_->config.post_norm_eps);
        }
        if (tap_layers != nullptr) {
            const auto found = std::lower_bound(
                tap_layers->begin(), tap_layers->end(), layer_index);
            if (found != tap_layers->end() && *found == layer_index) {
                const std::int64_t tap_index = static_cast<std::int64_t>(
                    std::distance(tap_layers->begin(), found));
                impl_->runtime->copy_d2d_async(
                    verification->impl_->scratch->staged_taps.as<float>() +
                        tap_index * rows * d,
                    hidden.get(), hidden_bytes);
            }
        }
    }
    impl_->rms(
        hidden, &impl_->final_norm, normalized, rows, d, impl_->config.norm_eps);
    impl_->runtime->copy_d2d_async(
        verification->impl_->scratch->staged_final.get(), normalized.get(), hidden_bytes);
    if (compute_logits || compute_argmax) {
        impl_->linear(
            impl_->lm_head, normalized, logits, rows, exact_verifier_rows);
    }
    if (compute_logits) {
        impl_->call(impl_->runtime->logit_transform()(
            logits.as<float>(), checked_mul(
                rows, impl_->config.vocab_size, "verification logits"),
            impl_->config.output_multiplier, impl_->config.logit_softcap,
            impl_->runtime->stream()), "Glimmer verification logit transform");
    }
    if (compute_argmax) {
        ArgmaxRows argmax = impl_->argmax(logits, rows, impl_->config.vocab_size);
        verification->impl_->host_argmax_indices = std::move(argmax.indices);
        verification->impl_->host_argmax_values = std::move(argmax.values);
        for (float& value : verification->impl_->host_argmax_values) {
            value = static_cast<float>(impl_->config.logit_softcap * std::tanh(
                (impl_->config.output_multiplier * static_cast<double>(value)) /
                impl_->config.logit_softcap));
        }
    }
    throw_if_cancelled(cancelled);
    impl_->runtime->synchronize();
    if (compute_logits) {
        impl_->runtime->copy_d2h(
            verification->impl_->host_logits.data(), logits.get(), logits_bytes);
    }
    if (verification->impl_->tap_count > 0 && copy_taps_to_host) {
        impl_->runtime->copy_d2h(
            verification->impl_->host_tap_major.data(),
            verification->impl_->scratch->staged_taps.get(),
            verification->impl_->host_tap_major.size() * sizeof(float));
        for (std::int64_t row = 0; row < rows; ++row) {
            for (std::int64_t tap = 0; tap < verification->impl_->tap_count; ++tap) {
                const float* source = verification->impl_->host_tap_major.data() +
                    (tap * rows + row) * d;
                float* target = verification->impl_->host_taps.data() +
                    (row * verification->impl_->tap_count + tap) * d;
                std::memcpy(target, source, checked_size(
                    checked_mul(d, 4, "verification tap row bytes"),
                    "verification tap row bytes"));
            }
        }
    }
    return verification;
}

void Model::commit_verification(
    const std::shared_ptr<Cache>& cache,
    const std::shared_ptr<Verification>& verification,
    std::int64_t accepted_rows,
    bool synchronize) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || !cache || !cache->impl_ || !verification ||
        !verification->impl_ || cache->impl_->runtime.get() != impl_->runtime.get() ||
        verification->impl_->runtime.get() != impl_->runtime.get() ||
        cache->impl_->logical_length != verification->impl_->position ||
        accepted_rows < 0 || accepted_rows > verification->impl_->row_count) {
        throw std::runtime_error("Glimmer CUDA verification commit is invalid");
    }
    if (accepted_rows == 0) return;
    const std::int64_t rows = verification->impl_->row_count;
    const std::int64_t kv_width = verification->impl_->kv_width;
    if (impl_->runtime->cache_commit_layers() != nullptr) {
        NfnNativeTileGlimmerCacheCommitLayersDescriptorV1 commit{};
        commit.struct_size = sizeof(commit);
        commit.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
        commit.staged_keys =
            verification->impl_->scratch->staged_keys.as<const float>();
        commit.staged_values =
            verification->impl_->scratch->staged_values.as<const float>();
        commit.layers = cache->impl_->layer_descriptors.data();
        commit.layer_count = impl_->config.num_layers;
        commit.source_rows = rows;
        commit.rows = accepted_rows;
        commit.kv_heads = impl_->config.num_kv_heads;
        commit.head_dim = impl_->config.head_dim;
        commit.position = verification->impl_->position;
        commit.source_layer_stride = checked_mul(
            rows, kv_width, "verification staged layer stride");
        commit.cuda_stream = impl_->runtime->stream();
        impl_->call(impl_->runtime->cache_commit_layers()(&commit),
                    "Glimmer all-layer verification cache commit");
    } else {
        for (std::int64_t layer_index = 0;
             layer_index < impl_->config.num_layers; ++layer_index) {
            Cache::Impl::Layer& cache_layer =
                cache->impl_->layers[static_cast<std::size_t>(layer_index)];
            const float* staged_key =
                verification->impl_->scratch->staged_keys.as<const float>() +
                layer_index * rows * kv_width;
            const float* staged_value =
                verification->impl_->scratch->staged_values.as<const float>() +
                layer_index * rows * kv_width;
            NfnNativeTileGlimmerCacheCommitDescriptorV1 commit{};
            commit.struct_size = sizeof(commit);
            commit.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
            commit.current_key = staged_key;
            commit.current_value = staged_value;
            commit.key_cache_bf16 = cache_layer.keys.as<std::uint16_t>();
            commit.value_cache_bf16 = cache_layer.values.as<std::uint16_t>();
            commit.kv_heads = impl_->config.num_kv_heads;
            commit.head_dim = impl_->config.head_dim;
            commit.position = verification->impl_->position;
            commit.cache_capacity = cache_layer.capacity;
            commit.cache_row_stride = kv_width;
            commit.cuda_stream = impl_->runtime->stream();
            if (impl_->runtime->cache_commit_rows() != nullptr) {
                impl_->call(impl_->runtime->cache_commit_rows()(
                    &commit, accepted_rows),
                    "Glimmer batched verification cache commit");
            } else {
                for (std::int64_t row = 0; row < accepted_rows; ++row) {
                    commit.current_key = staged_key + row * kv_width;
                    commit.current_value = staged_value + row * kv_width;
                    commit.position = verification->impl_->position + row;
                    impl_->call(impl_->runtime->cache_commit()(&commit),
                                "Glimmer verification cache commit");
                }
            }
        }
    }
    const std::size_t hidden_bytes = checked_size(checked_mul(
        impl_->config.model_dim, 4, "verification final hidden"),
        "verification final hidden");
    impl_->runtime->copy_d2d_async(
        cache->impl_->final_hidden.get(),
        verification->impl_->scratch->staged_final.as<const float>() +
            (accepted_rows - 1) * impl_->config.model_dim,
        hidden_bytes);
    if (synchronize) impl_->runtime->synchronize();
    cache->impl_->logical_length += accepted_rows;
}

void Model::synchronize() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed) throw std::runtime_error("Glimmer CUDA model is closed");
    impl_->runtime->synchronize();
}

std::vector<float> Model::logits(const std::shared_ptr<Cache>& cache) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || !cache || !cache->impl_ || cache->impl_->logical_length <= 0 ||
        cache->impl_->runtime.get() != impl_->runtime.get()) {
        throw std::runtime_error("Glimmer CUDA logits require a non-empty owned cache");
    }
    impl_->linear(impl_->lm_head, cache->impl_->final_hidden, impl_->logits_buffer);
    impl_->call(impl_->runtime->logit_transform()(
        impl_->logits_buffer.as<float>(), impl_->config.vocab_size,
        impl_->config.output_multiplier, impl_->config.logit_softcap,
        impl_->runtime->stream()), "Glimmer final logit transform");
    impl_->runtime->synchronize();
    std::vector<float> result(static_cast<std::size_t>(impl_->config.vocab_size));
    impl_->runtime->copy_d2h(
        result.data(), impl_->logits_buffer.get(), result.size() * sizeof(float));
    return result;
}

ArgmaxRows Model::argmax_logits(const std::shared_ptr<Cache>& cache) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || !cache || !cache->impl_ ||
        cache->impl_->logical_length <= 0 ||
        cache->impl_->runtime.get() != impl_->runtime.get() ||
        impl_->runtime->argmax_rows() == nullptr) {
        throw std::runtime_error(
            "Glimmer CUDA argmax requires a non-empty owned cache and device support");
    }
    impl_->linear(impl_->lm_head, cache->impl_->final_hidden, impl_->logits_buffer);
    ArgmaxRows result = impl_->argmax(
        impl_->logits_buffer, 1, impl_->config.vocab_size);
    for (float& value : result.values) {
        value = static_cast<float>(impl_->config.logit_softcap * std::tanh(
            (impl_->config.output_multiplier * static_cast<double>(value)) /
            impl_->config.logit_softcap));
    }
    return result;
}

ArgmaxRows Model::decode_argmax_and_append(
    const std::shared_ptr<Cache>& cache,
    const std::atomic<bool>& cancelled) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || !cache || !cache->impl_ ||
        cache->impl_->logical_length <= 0 ||
        cache->impl_->logical_length >= impl_->config.max_seq_len ||
        cache->impl_->runtime.get() != impl_->runtime.get() ||
        cache->impl_->decode_graph_disabled ||
        !impl_->runtime->has_decode_graphs()) {
        throw std::runtime_error(
            "Glimmer CUDA graph decode requires a non-empty compatible cache");
    }
    throw_if_cancelled(cancelled);
    const std::int64_t position = cache->impl_->logical_length;
    impl_->runtime->copy_h2d_async(
        cache->impl_->decode_graph_position.get(), &position, sizeof(position));

    if (cache->impl_->decode_graph_exec == nullptr) {
        // Capture operates on the same stable model workspace and cache
        // allocations used for replay. The position and selected token remain
        // device indirections, so neither changes the graph topology.
        impl_->runtime->synchronize();
        bool capture_started = false;
        try {
            impl_->runtime->begin_decode_graph_capture();
            capture_started = true;
            impl_->linear(
                impl_->lm_head, cache->impl_->final_hidden,
                impl_->logits_buffer);
            impl_->enqueue_argmax(
                impl_->logits_buffer, 1, impl_->config.vocab_size, false);
            append_input_unlocked(
                -1, nullptr, position, cache, cancelled, nullptr, nullptr,
                false, impl_->argmax_indices.as<const std::int64_t>(),
                cache->impl_->decode_graph_position.as<const std::int64_t>(),
                false, false);
            cache->impl_->decode_graph_exec =
                impl_->runtime->end_decode_graph_capture();
            capture_started = false;
        } catch (...) {
            if (capture_started) {
                impl_->runtime->abort_decode_graph_capture_noexcept();
            }
            cache->impl_->decode_graph_disabled = true;
            throw;
        }
    }

    // Cancellation is deliberately a token-boundary contract for captured
    // decode: once launched, the graph atomically produces and commits one
    // token before control returns to the session.
    impl_->runtime->launch_decode_graph(cache->impl_->decode_graph_exec);
    impl_->runtime->synchronize();
    ArgmaxRows result = impl_->copy_argmax(1);
    ++impl_->argmax_calls;
    ++impl_->argmax_rows_selected;
    cache->impl_->logical_length = position + 1;
    for (float& value : result.values) {
        value = static_cast<float>(impl_->config.logit_softcap * std::tanh(
            (impl_->config.output_multiplier * static_cast<double>(value)) /
            impl_->config.logit_softcap));
    }
    return result;
}

std::vector<float> Model::raw_logits(const float* hidden) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || hidden == nullptr) {
        throw std::runtime_error("Glimmer CUDA raw logits require an open model and hidden row");
    }
    impl_->runtime->copy_h2d_async(
        impl_->hidden.get(), hidden, impl_->hidden.bytes());
    impl_->linear(impl_->lm_head, impl_->hidden, impl_->logits_buffer);
    impl_->runtime->synchronize();
    std::vector<float> result(static_cast<std::size_t>(impl_->config.vocab_size));
    impl_->runtime->copy_d2h(
        result.data(), impl_->logits_buffer.get(), result.size() * sizeof(float));
    return result;
}

std::vector<float> Model::raw_logits_rows(
    const float* hidden,
    std::int64_t rows,
    bool fast_k_quant) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || hidden == nullptr || rows <= 0 || rows > 16) {
        throw std::runtime_error(
            "Glimmer CUDA raw batched logits require 1..16 hidden rows");
    }
    const std::size_t hidden_bytes = checked_size(checked_mul(
        checked_mul(rows, impl_->config.model_dim, "raw head hidden rows"), 4,
        "raw head hidden bytes"), "raw head hidden bytes");
    const std::size_t logits_bytes = checked_size(checked_mul(
        checked_mul(rows, impl_->config.vocab_size, "raw head logit rows"), 4,
        "raw head logit bytes"), "raw head logit bytes");
    impl_->runtime->copy_h2d_async(
        impl_->raw_head_hidden.get(), hidden, hidden_bytes);
    if (fast_k_quant && impl_->can_linear_q8(impl_->lm_head, rows)) {
        impl_->quantize_q8(impl_->raw_head_hidden, impl_->config.model_dim, rows);
        impl_->linear_q8(impl_->lm_head, impl_->raw_head_logits, rows);
    } else {
        impl_->linear(
            impl_->lm_head, impl_->raw_head_hidden, impl_->raw_head_logits, rows);
    }
    impl_->runtime->synchronize();
    std::vector<float> result(checked_size(checked_mul(
        rows, impl_->config.vocab_size, "raw head output"), "raw head output"));
    impl_->runtime->copy_d2h(
        result.data(), impl_->raw_head_logits.get(), logits_bytes);
    return result;
}

ArgmaxRows Model::raw_argmax_rows(
    const float* hidden,
    std::int64_t rows,
    bool fast_k_quant) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || hidden == nullptr || rows <= 0 || rows > 16 ||
        impl_->runtime->argmax_rows() == nullptr) {
        throw std::runtime_error(
            "Glimmer CUDA raw batched argmax requires 1..16 rows and device support");
    }
    const std::size_t hidden_bytes = checked_size(checked_mul(
        checked_mul(rows, impl_->config.model_dim, "raw argmax hidden rows"), 4,
        "raw argmax hidden bytes"), "raw argmax hidden bytes");
    impl_->runtime->copy_h2d_async(
        impl_->raw_head_hidden.get(), hidden, hidden_bytes);
    if (fast_k_quant && impl_->can_linear_q8(impl_->lm_head, rows)) {
        impl_->quantize_q8(impl_->raw_head_hidden, impl_->config.model_dim, rows);
        impl_->linear_q8(impl_->lm_head, impl_->raw_head_logits, rows);
    } else {
        impl_->linear(
            impl_->lm_head, impl_->raw_head_hidden, impl_->raw_head_logits, rows);
    }
    return impl_->argmax(impl_->raw_head_logits, rows, impl_->config.vocab_size);
}

ArgmaxRows Model::raw_argmax_rows_device(
    const float* device_hidden,
    int source_cuda_device,
    std::int64_t rows,
    bool fast_k_quant) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || device_hidden == nullptr || rows <= 0 || rows > 16 ||
        source_cuda_device != impl_->config.cuda_device ||
        impl_->runtime->argmax_rows() == nullptr) {
        throw std::runtime_error(
            "Glimmer CUDA device-hidden argmax requires compatible 1..16 rows");
    }
    const std::size_t hidden_bytes = checked_size(checked_mul(
        checked_mul(rows, impl_->config.model_dim,
                    "device-hidden argmax rows"), 4,
        "device-hidden argmax bytes"), "device-hidden argmax bytes");
    impl_->runtime->copy_d2d_async(
        impl_->raw_head_hidden.get(), device_hidden, hidden_bytes);
    if (fast_k_quant && impl_->can_linear_q8(impl_->lm_head, rows)) {
        impl_->quantize_q8(impl_->raw_head_hidden, impl_->config.model_dim, rows);
        impl_->linear_q8(impl_->lm_head, impl_->raw_head_logits, rows);
    } else {
        impl_->linear(
            impl_->lm_head, impl_->raw_head_hidden, impl_->raw_head_logits, rows);
    }
    return impl_->argmax(impl_->raw_head_logits, rows, impl_->config.vocab_size);
}

std::vector<float> Model::raw_embedding(std::int64_t token_id) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || token_id < 0 || token_id >= impl_->config.vocab_size) {
        throw std::runtime_error("Glimmer CUDA raw embedding token/model is invalid");
    }
    impl_->call(impl_->runtime->embedding()(
        &impl_->token_embedding.descriptor, token_id, impl_->hidden.as<float>()),
        "Glimmer raw embedding gather");
    impl_->runtime->synchronize();
    std::vector<float> result(static_cast<std::size_t>(impl_->config.model_dim));
    impl_->runtime->copy_d2h(result.data(), impl_->hidden.get(), impl_->hidden.bytes());
    return result;
}

std::vector<float> Model::raw_embeddings(
    const std::vector<std::int64_t>& token_ids) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const std::int64_t rows = static_cast<std::int64_t>(token_ids.size());
    if (impl_->closed || rows <= 0 || rows > 16 ||
        std::any_of(token_ids.begin(), token_ids.end(), [&](std::int64_t token) {
            return token < 0 || token >= impl_->config.vocab_size;
        })) {
        throw std::runtime_error(
            "Glimmer CUDA raw batched embedding tokens/model are invalid");
    }
    std::vector<std::int32_t> ids;
    ids.reserve(token_ids.size());
    for (std::int64_t token : token_ids) {
        ids.push_back(static_cast<std::int32_t>(token));
    }
    impl_->runtime->copy_h2d_async(
        impl_->raw_embedding_token_ids.get(), ids.data(),
        ids.size() * sizeof(std::int32_t));
    impl_->call(impl_->runtime->embedding_batch()(
        &impl_->token_embedding.descriptor,
        impl_->raw_embedding_token_ids.as<const std::int32_t>(),
        impl_->raw_head_hidden.as<float>(), rows),
        "Glimmer raw batched embedding gather");
    impl_->runtime->synchronize();
    std::vector<float> result(checked_size(checked_mul(
        rows, impl_->config.model_dim, "raw batched embeddings"),
        "raw batched embeddings"));
    impl_->runtime->copy_d2h(
        result.data(), impl_->raw_head_hidden.get(),
        result.size() * sizeof(float));
    return result;
}

const float* Model::raw_embeddings_device(
    const std::vector<std::int64_t>& token_ids) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const std::int64_t rows = static_cast<std::int64_t>(token_ids.size());
    if (impl_->closed || rows <= 0 || rows > 16 ||
        std::any_of(token_ids.begin(), token_ids.end(), [&](std::int64_t token) {
            return token < 0 || token >= impl_->config.vocab_size;
        })) {
        throw std::runtime_error(
            "Glimmer CUDA raw device embedding tokens/model are invalid");
    }
    std::vector<std::int32_t> ids;
    ids.reserve(token_ids.size());
    for (std::int64_t token : token_ids) {
        ids.push_back(static_cast<std::int32_t>(token));
    }
    impl_->runtime->copy_h2d_async(
        impl_->raw_embedding_token_ids.get(), ids.data(),
        ids.size() * sizeof(std::int32_t));
    impl_->call(impl_->runtime->embedding_batch()(
        &impl_->token_embedding.descriptor,
        impl_->raw_embedding_token_ids.as<const std::int32_t>(),
        impl_->raw_head_hidden.as<float>(), rows),
        "Glimmer raw batched device embedding gather");
    impl_->runtime->synchronize();
    return impl_->raw_head_hidden.as<const float>();
}

void Model::close() noexcept {
    if (!impl_) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->closed = true;
}

std::int64_t Model::resident_weight_bytes() const noexcept {
    return impl_->weight_bytes +
        (impl_->vision ? impl_->vision->weight_bytes() : 0);
}
std::int64_t Model::workspace_bytes() const noexcept {
    return impl_->workspace +
        (impl_->vision ? impl_->vision->workspace_bytes() : 0);
}
std::int64_t Model::kernel_launches() const noexcept {
    return impl_->launches +
        (impl_->vision ? impl_->vision->launches() : 0);
}
std::int64_t Model::k_quant_mmq_linears() const noexcept {
    return impl_->mmq_linears;
}
std::int64_t Model::q8_activation_quantizations() const noexcept {
    return impl_->q8_quantizations;
}
std::int64_t Model::q8_packed_linears() const noexcept {
    return impl_->q8_linears;
}
std::int64_t Model::device_argmax_calls() const noexcept {
    return impl_->argmax_calls;
}
std::int64_t Model::device_argmax_rows() const noexcept {
    return impl_->argmax_rows_selected;
}
bool Model::has_device_argmax() const noexcept {
    return impl_ && impl_->runtime->argmax_rows() != nullptr;
}
bool Model::has_decode_graphs() const noexcept {
    return impl_ && impl_->runtime->has_decode_graphs();
}
int Model::cuda_device() const noexcept { return impl_->runtime->device(); }
const std::string& Model::tile_ops_library() const noexcept { return impl_->runtime->tile_path(); }
const std::string& Model::cuda_runtime_library() const noexcept { return impl_->runtime->cuda_path(); }

class DFlashCache::Impl final {
public:
    struct Layer {
        DeviceBuffer keys;
        DeviceBuffer values;
    };

    Impl(std::shared_ptr<Runtime> source_runtime, const DFlashConfig& config)
        : runtime(std::move(source_runtime)), capacity(config.sliding_window) {
        const std::int64_t kv_width = checked_mul(
            config.num_kv_heads, config.head_dim, "DFlash KV width");
        const std::size_t bytes = checked_size(checked_mul(
            checked_mul(capacity, kv_width, "DFlash cache elements"), 2,
            "DFlash BF16 cache bytes"), "DFlash BF16 cache bytes");
        layers.reserve(checked_size(config.num_layers, "DFlash cache layers"));
        for (std::int64_t index = 0; index < config.num_layers; ++index) {
            Layer layer;
            layer.keys = DeviceBuffer(runtime, bytes);
            layer.values = DeviceBuffer(runtime, bytes);
            runtime->zero_async(layer.keys.get(), bytes);
            runtime->zero_async(layer.values.get(), bytes);
            allocated_bytes += static_cast<std::int64_t>(bytes) * 2;
            layers.push_back(std::move(layer));
        }
        runtime->synchronize();
    }

    std::shared_ptr<Runtime> runtime;
    std::vector<Layer> layers;
    std::int64_t capacity = 0;
    std::int64_t logical_length = 0;
    std::int64_t allocated_bytes = 0;
};

DFlashCache::DFlashCache(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
DFlashCache::~DFlashCache() = default;
std::int64_t DFlashCache::logical_length() const noexcept {
    return impl_ ? impl_->logical_length : 0;
}
std::int64_t DFlashCache::allocated_bytes() const noexcept {
    return impl_ ? impl_->allocated_bytes : 0;
}

struct DFlashDeviceLayer {
    DeviceWeight input_norm;
    DeviceWeight post_attention_norm;
    DeviceWeight q;
    DeviceWeight k;
    DeviceWeight v;
    DeviceWeight output;
    DeviceWeight q_norm;
    DeviceWeight k_norm;
    DeviceWeight mlp_gate;
    DeviceWeight mlp_up;
    DeviceWeight mlp_down;
};

class DFlashModel::Impl final {
public:
    Impl(const DFlashConfig& source_config, const DFlashHostWeightPlan& source)
        : config(source_config), runtime(std::make_shared<Runtime>(target_runtime_config(source_config))) {
        if (config.max_seq_len <= 0 || config.model_dim <= 0 ||
            config.intermediate_dim <= 0 || config.num_layers <= 0 ||
            config.num_heads <= 0 || config.num_kv_heads <= 0 ||
            config.num_heads % config.num_kv_heads != 0 || config.head_dim <= 0 ||
            config.head_dim > 256 || config.block_size < 2 || config.block_size > 64 ||
            config.tap_count <= 0 || config.sliding_window <= 0 ||
            config.sliding_window > config.max_seq_len ||
            source.layers.size() != checked_size(config.num_layers, "DFlash layer count")) {
            throw std::runtime_error("DFlash CUDA model geometry is invalid");
        }
        context_projection = upload_weight(runtime, source.context_projection, &weight_bytes);
        context_norm = upload_weight(runtime, source.context_norm, &weight_bytes);
        final_norm = upload_weight(runtime, source.final_norm, &weight_bytes);
        layers.reserve(source.layers.size());
        for (const DFlashHostLayerWeights& host : source.layers) {
            DFlashDeviceLayer layer;
            layer.input_norm = upload_weight(runtime, host.input_norm, &weight_bytes);
            layer.post_attention_norm = upload_weight(
                runtime, host.post_attention_norm, &weight_bytes);
            layer.q = upload_weight(runtime, host.q, &weight_bytes);
            layer.k = upload_weight(runtime, host.k, &weight_bytes);
            layer.v = upload_weight(runtime, host.v, &weight_bytes);
            layer.output = upload_weight(runtime, host.output, &weight_bytes);
            layer.q_norm = upload_weight(runtime, host.q_norm, &weight_bytes);
            layer.k_norm = upload_weight(runtime, host.k_norm, &weight_bytes);
            layer.mlp_gate = upload_weight(runtime, host.mlp_gate, &weight_bytes);
            layer.mlp_up = upload_weight(runtime, host.mlp_up, &weight_bytes);
            layer.mlp_down = upload_weight(runtime, host.mlp_down, &weight_bytes);
            layers.push_back(std::move(layer));
        }
        runtime->synchronize();

        const auto floats = [&](std::int64_t count, const char* label) {
            return checked_size(checked_mul(count, 4, label), label);
        };
        const std::int64_t rows = config.block_size;
        const std::int64_t hidden_count = checked_mul(rows, config.model_dim, "DFlash hidden");
        const std::int64_t query_width = checked_mul(
            config.num_heads, config.head_dim, "DFlash query width");
        const std::int64_t kv_width = checked_mul(
            config.num_kv_heads, config.head_dim, "DFlash KV width");
        context_input = DeviceBuffer(runtime, floats(checked_mul(rows, checked_mul(
            config.tap_count, config.model_dim, "DFlash tap width"),
            "DFlash tap rows"), "DFlash tap input"));
        context_projected = DeviceBuffer(runtime, floats(checked_mul(
            rows, config.model_dim, "DFlash context rows"), "DFlash context"));
        context_normalized = DeviceBuffer(runtime, context_projected.bytes());
        hidden = DeviceBuffer(runtime, floats(hidden_count, "DFlash hidden"));
        normalized = DeviceBuffer(runtime, hidden.bytes());
        residual = DeviceBuffer(runtime, hidden.bytes());
        branch = DeviceBuffer(runtime, hidden.bytes());
        query = DeviceBuffer(runtime, floats(checked_mul(rows, query_width, "DFlash Q"), "DFlash Q"));
        key = DeviceBuffer(runtime, floats(checked_mul(rows, kv_width, "DFlash K"), "DFlash K"));
        value = DeviceBuffer(runtime, key.bytes());
        attention = DeviceBuffer(runtime, query.bytes());
        mlp_gate = DeviceBuffer(runtime, floats(checked_mul(
            rows, config.intermediate_dim, "DFlash MLP"), "DFlash MLP"));
        mlp_up = DeviceBuffer(runtime, mlp_gate.bytes());
        mlp_activated = DeviceBuffer(runtime, mlp_gate.bytes());
        const std::int64_t k_quant_mmq_width = std::max(
            checked_mul(config.tap_count, config.model_dim, "DFlash MMQ tap width"),
            std::max(config.intermediate_dim,
                     std::max(config.model_dim, query_width)));
        const std::int64_t k_quant_mmq_bytes =
            runtime->k_quant_mmq_workspace_bytes(rows, k_quant_mmq_width);
        if (runtime->has_k_quant_mmq() && k_quant_mmq_bytes <= 0) {
            throw std::runtime_error(
                "DFlash CUDA exact K-quant MMQ workspace query failed");
        }
        if (k_quant_mmq_bytes > 0) {
            k_quant_mmq_workspace = DeviceBuffer(
                runtime, checked_size(
                    k_quant_mmq_bytes, "DFlash exact K-quant MMQ workspace"));
        }
        workspace = static_cast<std::int64_t>(
            context_input.bytes() + context_projected.bytes() + context_normalized.bytes() +
            hidden.bytes() + normalized.bytes() + residual.bytes() + branch.bytes() +
            query.bytes() + key.bytes() + value.bytes() + attention.bytes() +
            mlp_gate.bytes() + mlp_up.bytes() + mlp_activated.bytes() +
            k_quant_mmq_workspace.bytes());
    }

    static Config target_runtime_config(const DFlashConfig& source) {
        Config result;
        result.cuda_device = source.cuda_device;
        result.tile_ops_lib = source.tile_ops_lib;
        result.cuda_runtime_lib = source.cuda_runtime_lib;
        return result;
    }

    void call(int status, const char* label) {
        runtime->check_tile(status, label);
        ++launches;
    }

    void linear(
        const DeviceWeight& weight,
        const DeviceBuffer& input,
        DeviceBuffer& output,
        std::int64_t rows) {
        if (can_linear_native_mmq(weight, rows)) {
            const NfnNativeTilePackedWeightDescriptorV1* descriptors[]{
                &weight.descriptor};
            float* outputs[]{output.as<float>()};
            call(runtime->k_quant_mmq_multi_linear(
                descriptors, input.as<const float>(), outputs, 1, rows,
                k_quant_mmq_workspace.get(),
                static_cast<std::int64_t>(k_quant_mmq_workspace.bytes())),
                "DFlash exact K-quant MMQ linear");
            ++mmq_linears;
            return;
        }
        if (can_linear_experimental_mmq(weight, rows)) {
            call(runtime->experimental_mmq_linear(
                &weight.descriptor, input.as<const float>(), output.as<float>(),
                rows), "DFlash experimental packed MMQ linear");
            return;
        }
        call(runtime->linear()(
            &weight.descriptor, input.as<const float>(), nullptr, output.as<float>(), rows, false),
            "DFlash packed linear");
    }

    static bool is_k_quant(const DeviceWeight& weight) noexcept {
        return weight.descriptor.encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K ||
            weight.descriptor.encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K ||
            weight.descriptor.encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q6_K;
    }

    bool can_linear_native_mmq(
        const DeviceWeight& weight,
        std::int64_t rows) const noexcept {
        return rows >= 2 && rows <= 16 && runtime->has_k_quant_mmq() &&
            k_quant_mmq_workspace.get() != nullptr && is_k_quant(weight);
    }

    bool can_linear_experimental_mmq(
        const DeviceWeight& weight,
        std::int64_t rows) const noexcept {
        return rows >= 2 && rows <= 16 && runtime->has_experimental_mmq() &&
            is_k_quant(weight);
    }

    template <std::size_t Count>
    bool can_linear_mmq_group(
        const std::array<const DeviceWeight*, Count>& weights,
        std::int64_t rows) const noexcept {
        const bool native =
            std::all_of(weights.begin(), weights.end(), [&](const DeviceWeight* weight) {
                return weight != nullptr && can_linear_native_mmq(*weight, rows);
            });
        if (native) return true;
        return runtime->has_experimental_mmq_multi() &&
            std::all_of(weights.begin(), weights.end(), [&](const DeviceWeight* weight) {
                return weight != nullptr && can_linear_experimental_mmq(*weight, rows);
            });
    }

    template <std::size_t Count>
    void linear_group(
        const std::array<const DeviceWeight*, Count>& weights,
        const DeviceBuffer& input,
        const std::array<DeviceBuffer*, Count>& outputs,
        std::int64_t rows) {
        if (!can_linear_mmq_group(weights, rows)) {
            for (std::size_t index = 0; index < Count; ++index) {
                linear(*weights[index], input, *outputs[index], rows);
            }
            return;
        }
        std::array<const NfnNativeTilePackedWeightDescriptorV1*, Count> descriptors{};
        std::array<float*, Count> output_pointers{};
        for (std::size_t index = 0; index < Count; ++index) {
            descriptors[index] = &weights[index]->descriptor;
            output_pointers[index] = outputs[index]->template as<float>();
        }
        const bool use_native = std::all_of(
            weights.begin(), weights.end(), [&](const DeviceWeight* weight) {
                return weight != nullptr && can_linear_native_mmq(*weight, rows);
            });
        if (use_native) {
            call(runtime->k_quant_mmq_multi_linear(
                descriptors.data(), input.as<const float>(), output_pointers.data(),
                static_cast<std::int64_t>(Count), rows,
                k_quant_mmq_workspace.get(),
                static_cast<std::int64_t>(k_quant_mmq_workspace.bytes())),
                "DFlash exact grouped K-quant MMQ linears");
            mmq_linears += static_cast<std::int64_t>(Count);
        } else {
            call(runtime->experimental_mmq_multi_linear(
                descriptors.data(), input.as<const float>(), output_pointers.data(),
                static_cast<std::int64_t>(Count), rows),
                "DFlash experimental grouped packed MMQ linears");
        }
    }

    void rms(
        const DeviceBuffer& input,
        const DeviceWeight& weight,
        DeviceBuffer& output,
        std::int64_t rows,
        std::int64_t width) {
        call(runtime->rms()(
            input.as<const float>(), &weight.descriptor, output.as<float>(), rows, width,
            config.norm_eps, weight.centered, runtime->stream()),
            "DFlash RMSNorm");
    }

    void rms_capture(
        const DeviceBuffer& input,
        const DeviceWeight& weight,
        DeviceBuffer& output,
        DeviceBuffer& residual_output,
        std::int64_t rows,
        std::int64_t width) {
        if (!runtime->has_fused_residual_norm()) {
            runtime->copy_d2d_async(
                residual_output.get(), input.get(), checked_size(checked_mul(
                    checked_mul(rows, width, "DFlash RMS capture values"), 4,
                    "DFlash RMS capture bytes"), "DFlash RMS capture bytes"));
            rms(input, weight, output, rows, width);
            return;
        }
        call(runtime->rms_capture()(
            input.as<const float>(), &weight.descriptor, output.as<float>(),
            residual_output.as<float>(), rows, width, config.norm_eps,
            weight.centered, runtime->stream()),
            "DFlash fused RMSNorm residual capture");
    }

    DFlashConfig config;
    std::shared_ptr<Runtime> runtime;
    DeviceWeight context_projection;
    DeviceWeight context_norm;
    DeviceWeight final_norm;
    std::vector<DFlashDeviceLayer> layers;
    DeviceBuffer context_input;
    DeviceBuffer context_projected;
    DeviceBuffer context_normalized;
    DeviceBuffer hidden;
    DeviceBuffer normalized;
    DeviceBuffer residual;
    DeviceBuffer branch;
    DeviceBuffer query;
    DeviceBuffer key;
    DeviceBuffer value;
    DeviceBuffer attention;
    DeviceBuffer mlp_gate;
    DeviceBuffer mlp_up;
    DeviceBuffer mlp_activated;
    DeviceBuffer k_quant_mmq_workspace;
    std::int64_t weight_bytes = 0;
    std::int64_t workspace = 0;
    std::int64_t launches = 0;
    std::int64_t mmq_linears = 0;
    bool closed = false;
    std::mutex mutex;
};

DFlashModel::DFlashModel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
DFlashModel::~DFlashModel() { close(); }

std::shared_ptr<DFlashModel> DFlashModel::load(
    const DFlashConfig& config,
    const DFlashHostWeightPlan& weights) {
    return std::shared_ptr<DFlashModel>(
        new DFlashModel(std::make_unique<Impl>(config, weights)));
}

std::shared_ptr<DFlashCache> DFlashModel::create_cache() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed) throw std::runtime_error("DFlash CUDA model is closed");
    return std::shared_ptr<DFlashCache>(new DFlashCache(std::make_unique<DFlashCache::Impl>(
        impl_->runtime, impl_->config)));
}

void DFlashModel::append_context(
    const float* concatenated_target_taps,
    std::int64_t position,
    const std::shared_ptr<DFlashCache>& cache,
    const std::atomic<bool>& cancelled) {
    append_contexts(
        concatenated_target_taps, 1, position, cache, cancelled);
}

void DFlashModel::append_contexts(
    const float* concatenated_target_taps,
    std::int64_t rows,
    std::int64_t start_position,
    const std::shared_ptr<DFlashCache>& cache,
    const std::atomic<bool>& cancelled) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || !cache || !cache->impl_ || concatenated_target_taps == nullptr ||
        cache->impl_->runtime.get() != impl_->runtime.get() ||
        cache->impl_->logical_length != start_position || rows <= 0 ||
        rows > impl_->config.block_size || start_position < 0 ||
        start_position + rows > impl_->config.max_seq_len) {
        throw std::runtime_error("DFlash CUDA context batch/cache position is invalid");
    }
    throw_if_cancelled(cancelled);
    const std::int64_t tap_width = checked_mul(
        impl_->config.tap_count, impl_->config.model_dim, "DFlash tap width");
    const std::size_t tap_bytes = checked_size(checked_mul(checked_mul(
        rows, tap_width, "DFlash tap rows"), 4, "DFlash tap bytes"),
        "DFlash tap bytes");
    impl_->runtime->copy_h2d_async(
        impl_->context_input.get(), concatenated_target_taps, tap_bytes);
    append_context_rows_locked(rows, start_position, cache, cancelled);
}

std::vector<float> DFlashModel::append_contexts_device_tap_major(
    const float* tap_major_device,
    int source_cuda_device,
    std::int64_t source_rows,
    std::int64_t rows,
    std::int64_t start_position,
    const std::shared_ptr<DFlashCache>& cache,
    const std::atomic<bool>& cancelled) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || !cache || !cache->impl_ || tap_major_device == nullptr ||
        cache->impl_->runtime.get() != impl_->runtime.get() ||
        cache->impl_->logical_length != start_position ||
        source_cuda_device != impl_->config.cuda_device || source_rows <= 0 ||
        source_rows > impl_->config.block_size || rows <= 0 || rows > source_rows ||
        start_position < 0 || start_position + rows > impl_->config.max_seq_len ||
        impl_->runtime->pack_target_taps() == nullptr) {
        throw std::runtime_error(
            "DFlash CUDA device target-tap batch/cache position is invalid");
    }
    throw_if_cancelled(cancelled);
    const std::int64_t tap_width = checked_mul(
        impl_->config.tap_count, impl_->config.model_dim, "DFlash tap width");
    impl_->call(impl_->runtime->pack_target_taps()(
        tap_major_device, impl_->context_input.as<float>(), source_rows, 0,
        rows, impl_->config.tap_count, impl_->config.model_dim,
        impl_->runtime->stream()), "DFlash target tap pack");
    const std::int64_t append_rows = rows - 1;
    if (append_rows > 0) {
        append_context_rows_locked(
            append_rows, start_position, cache, cancelled);
    } else {
        impl_->runtime->synchronize();
    }
    std::vector<float> last(checked_size(tap_width, "DFlash last target taps"));
    impl_->runtime->copy_d2h(
        last.data(),
        impl_->context_input.as<const float>() + (rows - 1) * tap_width,
        checked_size(checked_mul(tap_width, 4, "DFlash last target tap bytes"),
                     "DFlash last target tap bytes"));
    return last;
}

void DFlashModel::append_contexts_device_tap_major_all(
    const float* tap_major_device,
    int source_cuda_device,
    std::int64_t source_rows,
    std::int64_t rows,
    std::int64_t start_position,
    const std::shared_ptr<DFlashCache>& cache,
    const std::atomic<bool>& cancelled) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || !cache || !cache->impl_ || tap_major_device == nullptr ||
        cache->impl_->runtime.get() != impl_->runtime.get() ||
        cache->impl_->logical_length != start_position ||
        source_cuda_device != impl_->config.cuda_device || source_rows <= 0 ||
        source_rows > impl_->config.block_size || rows <= 0 || rows > source_rows ||
        start_position < 0 || start_position + rows > impl_->config.max_seq_len ||
        impl_->runtime->pack_target_taps() == nullptr) {
        throw std::runtime_error(
            "DFlash CUDA all-row target-tap batch/cache position is invalid");
    }
    throw_if_cancelled(cancelled);
    impl_->call(impl_->runtime->pack_target_taps()(
        tap_major_device, impl_->context_input.as<float>(), source_rows, 0,
        rows, impl_->config.tap_count, impl_->config.model_dim,
        impl_->runtime->stream()), "DFlash all-row target tap pack");
    append_context_rows_locked(rows, start_position, cache, cancelled);
}

bool DFlashModel::has_device_tap_pack() const noexcept {
    return impl_ && impl_->runtime->pack_target_taps() != nullptr;
}

void DFlashModel::append_context_rows_locked(
    std::int64_t rows,
    std::int64_t start_position,
    const std::shared_ptr<DFlashCache>& cache,
    const std::atomic<bool>& cancelled) {
    impl_->linear(
        impl_->context_projection, impl_->context_input, impl_->context_projected, rows);
    impl_->rms(
        impl_->context_projected, impl_->context_norm, impl_->context_normalized,
        rows, impl_->config.model_dim);
    const std::int64_t query_width = checked_mul(
        impl_->config.num_heads, impl_->config.head_dim, "DFlash query width");
    const std::int64_t kv_width = checked_mul(
        impl_->config.num_kv_heads, impl_->config.head_dim, "DFlash KV width");
    impl_->runtime->zero_async(
        impl_->query.get(), checked_size(checked_mul(checked_mul(
            rows, query_width, "DFlash dummy Q rows"), 4,
            "DFlash dummy Q bytes"), "DFlash dummy Q bytes"));
    for (std::int64_t layer_index = 0; layer_index < impl_->config.num_layers; ++layer_index) {
        throw_if_cancelled(cancelled);
        DFlashDeviceLayer& layer = impl_->layers[static_cast<std::size_t>(layer_index)];
        impl_->linear_group(
            std::array<const DeviceWeight*, 2>{&layer.k, &layer.v},
            impl_->context_normalized,
            std::array<DeviceBuffer*, 2>{&impl_->key, &impl_->value}, rows);
        if (impl_->runtime->qk_norm_scale_rope_batch() != nullptr) {
            impl_->call(impl_->runtime->qk_norm_scale_rope_batch()(
                impl_->query.as<float>(), impl_->key.as<float>(), nullptr,
                &layer.k_norm.descriptor, rows, impl_->config.num_heads,
                impl_->config.num_kv_heads, impl_->config.head_dim,
                impl_->config.norm_eps, false, layer.k_norm.centered, 1.0f,
                start_position, impl_->config.rope_theta,
                impl_->config.gguf_interleaved
                    ? NFN_NATIVE_TILE_GLIMMER_ROPE_INTERLEAVED
                    : NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT,
                true, impl_->runtime->stream()),
                "DFlash fused batched context K norm/RoPE");
        } else {
            impl_->rms(
                impl_->key, layer.k_norm, impl_->key,
                checked_mul(rows, impl_->config.num_kv_heads,
                            "DFlash context K norm rows"),
                impl_->config.head_dim);
            impl_->call(impl_->runtime->rope_batch()(
                impl_->query.as<float>(), impl_->key.as<float>(), rows,
                impl_->config.num_heads, impl_->config.num_kv_heads,
                impl_->config.head_dim, start_position,
                impl_->config.rope_theta,
                impl_->config.gguf_interleaved
                    ? NFN_NATIVE_TILE_GLIMMER_ROPE_INTERLEAVED
                    : NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT,
                false, impl_->runtime->stream()),
                "DFlash batched positioned context RoPE");
        }
        DFlashCache::Impl::Layer& cache_layer =
            cache->impl_->layers[static_cast<std::size_t>(layer_index)];
        NfnNativeTileGlimmerCacheCommitDescriptorV1 commit{};
        commit.struct_size = sizeof(commit);
        commit.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
        commit.current_key = impl_->key.as<const float>();
        commit.current_value = impl_->value.as<const float>();
        commit.key_cache_bf16 = cache_layer.keys.as<std::uint16_t>();
        commit.value_cache_bf16 = cache_layer.values.as<std::uint16_t>();
        commit.kv_heads = impl_->config.num_kv_heads;
        commit.head_dim = impl_->config.head_dim;
        commit.position = start_position;
        commit.cache_capacity = cache->impl_->capacity;
        commit.cache_row_stride = kv_width;
        commit.cuda_stream = impl_->runtime->stream();
        if (impl_->runtime->cache_commit_rows() != nullptr) {
            impl_->call(impl_->runtime->cache_commit_rows()(&commit, rows),
                        "DFlash batched context cache commit");
        } else {
            for (std::int64_t row = 0; row < rows; ++row) {
                commit.current_key = impl_->key.as<const float>() + row * kv_width;
                commit.current_value = impl_->value.as<const float>() + row * kv_width;
                commit.position = start_position + row;
                impl_->call(impl_->runtime->cache_commit()(&commit),
                            "DFlash context cache commit");
            }
        }
    }
    impl_->runtime->synchronize();
    cache->impl_->logical_length = start_position + rows;
}

std::vector<float> DFlashModel::forward_block(
    const float* raw_target_embeddings,
    std::int64_t rows,
    const std::shared_ptr<DFlashCache>& cache,
    const std::atomic<bool>& cancelled) {
    std::vector<float> result;
    forward_block_input(
        raw_target_embeddings, false, impl_->config.cuda_device, rows,
        cache, cancelled, &result);
    return result;
}

const float* DFlashModel::forward_block_device(
    const float* raw_target_embeddings_device,
    int source_cuda_device,
    std::int64_t rows,
    const std::shared_ptr<DFlashCache>& cache,
    const std::atomic<bool>& cancelled) {
    return forward_block_input(
        raw_target_embeddings_device, true, source_cuda_device, rows,
        cache, cancelled, nullptr);
}

const float* DFlashModel::forward_block_input(
    const float* raw_target_embeddings,
    bool input_is_device,
    int source_cuda_device,
    std::int64_t rows,
    const std::shared_ptr<DFlashCache>& cache,
    const std::atomic<bool>& cancelled,
    std::vector<float>* host_result) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || !cache || !cache->impl_ || raw_target_embeddings == nullptr ||
        cache->impl_->runtime.get() != impl_->runtime.get() ||
        (input_is_device && source_cuda_device != impl_->config.cuda_device) ||
        rows != impl_->config.block_size || cache->impl_->logical_length < 0 ||
        cache->impl_->logical_length + rows > impl_->config.max_seq_len) {
        throw std::runtime_error("DFlash CUDA proposal block/cache is invalid");
    }
    throw_if_cancelled(cancelled);
    if (input_is_device) {
        impl_->runtime->copy_d2d_async(
            impl_->hidden.get(), raw_target_embeddings, impl_->hidden.bytes());
    } else {
        impl_->runtime->copy_h2d_async(
            impl_->hidden.get(), raw_target_embeddings, impl_->hidden.bytes());
    }
    const std::int64_t hidden_count = checked_mul(rows, impl_->config.model_dim, "DFlash hidden");
    const std::int64_t kv_width = checked_mul(
        impl_->config.num_kv_heads, impl_->config.head_dim, "DFlash KV width");
    for (std::int64_t layer_index = 0; layer_index < impl_->config.num_layers; ++layer_index) {
        throw_if_cancelled(cancelled);
        DFlashDeviceLayer& layer = impl_->layers[static_cast<std::size_t>(layer_index)];
        impl_->rms_capture(
            impl_->hidden, layer.input_norm, impl_->normalized, impl_->residual,
            rows, impl_->config.model_dim);
        impl_->linear_group(
            std::array<const DeviceWeight*, 3>{&layer.q, &layer.k, &layer.v},
            impl_->normalized,
            std::array<DeviceBuffer*, 3>{
                &impl_->query, &impl_->key, &impl_->value}, rows);
        if (impl_->runtime->qk_norm_scale_rope_batch() != nullptr) {
            impl_->call(impl_->runtime->qk_norm_scale_rope_batch()(
                impl_->query.as<float>(), impl_->key.as<float>(),
                &layer.q_norm.descriptor, &layer.k_norm.descriptor, rows,
                impl_->config.num_heads, impl_->config.num_kv_heads,
                impl_->config.head_dim, impl_->config.norm_eps,
                layer.q_norm.centered, layer.k_norm.centered, 1.0f,
                cache->impl_->logical_length, impl_->config.rope_theta,
                impl_->config.gguf_interleaved
                    ? NFN_NATIVE_TILE_GLIMMER_ROPE_INTERLEAVED
                    : NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT,
                true, impl_->runtime->stream()),
                "DFlash fused batched QK norm/RoPE");
        } else {
            impl_->rms(
                impl_->query, layer.q_norm, impl_->query,
                checked_mul(rows, impl_->config.num_heads, "DFlash Q norm rows"),
                impl_->config.head_dim);
            impl_->rms(
                impl_->key, layer.k_norm, impl_->key,
                checked_mul(rows, impl_->config.num_kv_heads, "DFlash K norm rows"),
                impl_->config.head_dim);
            impl_->call(impl_->runtime->rope_batch()(
                impl_->query.as<float>(), impl_->key.as<float>(), rows,
                impl_->config.num_heads, impl_->config.num_kv_heads,
                impl_->config.head_dim, cache->impl_->logical_length,
                impl_->config.rope_theta,
                impl_->config.gguf_interleaved
                    ? NFN_NATIVE_TILE_GLIMMER_ROPE_INTERLEAVED
                    : NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT,
                false, impl_->runtime->stream()),
                "DFlash batched positioned block RoPE");
        }
        DFlashCache::Impl::Layer& cache_layer =
            cache->impl_->layers[static_cast<std::size_t>(layer_index)];
        NfnNativeTileDFlashBlockAttentionDescriptorV1 attention_descriptor{};
        attention_descriptor.struct_size = sizeof(attention_descriptor);
        attention_descriptor.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
        attention_descriptor.query = impl_->query.as<const float>();
        attention_descriptor.block_key = impl_->key.as<const float>();
        attention_descriptor.block_value = impl_->value.as<const float>();
        attention_descriptor.key_cache_bf16 = cache_layer.keys.as<const std::uint16_t>();
        attention_descriptor.value_cache_bf16 = cache_layer.values.as<const std::uint16_t>();
        attention_descriptor.output = impl_->attention.as<float>();
        attention_descriptor.query_rows = rows;
        attention_descriptor.block_rows = rows;
        attention_descriptor.query_heads = impl_->config.num_heads;
        attention_descriptor.kv_heads = impl_->config.num_kv_heads;
        attention_descriptor.head_dim = impl_->config.head_dim;
        attention_descriptor.context_length = cache->impl_->logical_length;
        attention_descriptor.sliding_window = impl_->config.sliding_window;
        attention_descriptor.cache_capacity = cache->impl_->capacity;
        attention_descriptor.cache_row_stride = kv_width;
        attention_descriptor.scale = 1.0f /
            std::sqrt(static_cast<float>(impl_->config.head_dim));
        attention_descriptor.cuda_stream = impl_->runtime->stream();
        impl_->call(impl_->runtime->dflash_attention()(&attention_descriptor),
                    "DFlash block attention");
        impl_->linear(layer.output, impl_->attention, impl_->branch, rows);
        impl_->call(impl_->runtime->add()(
            impl_->residual.as<const float>(), impl_->branch.as<const float>(),
            impl_->hidden.as<float>(), hidden_count, impl_->runtime->stream()),
            "DFlash attention residual");
        impl_->rms_capture(
            impl_->hidden, layer.post_attention_norm, impl_->normalized,
            impl_->residual, rows, impl_->config.model_dim);
        impl_->linear_group(
            std::array<const DeviceWeight*, 2>{&layer.mlp_gate, &layer.mlp_up},
            impl_->normalized,
            std::array<DeviceBuffer*, 2>{&impl_->mlp_gate, &impl_->mlp_up}, rows);
        impl_->call(impl_->runtime->swiglu()(
            impl_->mlp_gate.as<const float>(), impl_->mlp_up.as<const float>(),
            impl_->mlp_activated.as<float>(),
            checked_mul(rows, impl_->config.intermediate_dim, "DFlash SwiGLU"),
            impl_->runtime->stream()), "DFlash SwiGLU");
        impl_->linear(layer.mlp_down, impl_->mlp_activated, impl_->branch, rows);
        impl_->call(impl_->runtime->add()(
            impl_->residual.as<const float>(), impl_->branch.as<const float>(),
            impl_->hidden.as<float>(), hidden_count, impl_->runtime->stream()),
            "DFlash feedforward residual");
    }
    impl_->rms(
        impl_->hidden, impl_->final_norm, impl_->normalized,
        rows, impl_->config.model_dim);
    impl_->runtime->synchronize();
    if (host_result != nullptr) {
        host_result->resize(checked_size(hidden_count, "DFlash result"));
        impl_->runtime->copy_d2h(
            host_result->data(), impl_->normalized.get(), impl_->normalized.bytes());
    }
    return impl_->normalized.as<const float>();
}

void DFlashModel::close() noexcept {
    if (!impl_) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->closed = true;
}

std::int64_t DFlashModel::resident_weight_bytes() const noexcept {
    return impl_->weight_bytes;
}
std::int64_t DFlashModel::workspace_bytes() const noexcept { return impl_->workspace; }
std::int64_t DFlashModel::kernel_launches() const noexcept { return impl_->launches; }
std::int64_t DFlashModel::k_quant_mmq_linears() const noexcept {
    return impl_->mmq_linears;
}

}  // namespace neuralfn::resident_glimmer_cuda
