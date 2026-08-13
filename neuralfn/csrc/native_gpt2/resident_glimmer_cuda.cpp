#include "resident_glimmer_cuda.h"

#include "../native_train/tile_ops.h"

#include <algorithm>
#include <cmath>
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
using CudaGetErrorStringFn = const char* (*)(int);

using AbiVersionFn = int (*)();
using ErrorStringFn = const char* (*)(int);
using PackedValidateFn = int (*)(const NfnNativeTilePackedWeightDescriptorV1*);
using PackedLinearFn = int (*)(
    const NfnNativeTilePackedWeightDescriptorV1*, const float*, const float*,
    float*, std::int64_t, bool);
using EmbeddingFn = int (*)(
    const NfnNativeTilePackedWeightDescriptorV1*, std::int64_t, float*);
using RmsNormFn = int (*)(
    const float*, const NfnNativeTilePackedWeightDescriptorV1*, float*,
    std::int64_t, std::int64_t, float, bool, void*);
using RopeFn = int (*)(
    float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    float, std::uint32_t, void*);
using GqaFn = int (*)(const NfnNativeTileGlimmerGqaDecodeDescriptorV1*);
using CacheCommitFn = int (*)(const NfnNativeTileGlimmerCacheCommitDescriptorV1*);
using DFlashAttentionFn = int (*)(
    const NfnNativeTileDFlashBlockAttentionDescriptorV1*);
using GateFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
using LogitTransformFn = int (*)(float*, std::int64_t, float, float, void*);
using SwiGluFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
using AddFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
using ScaleFn = int (*)(float*, std::int64_t, float, void*);

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
        embedding_ = tile_.require<EmbeddingFn>(
            "nfn_native_tile_glimmer_embedding_gather_float32_v1");
        rms_ = tile_.require<RmsNormFn>(
            "nfn_native_tile_glimmer_rms_norm_affine_float32_v1");
        rope_ = tile_.require<RopeFn>(
            "nfn_native_tile_glimmer_positioned_rope_float32_v1");
        gqa_ = tile_.require<GqaFn>("nfn_native_tile_glimmer_gqa_decode_float32_v1");
        cache_commit_ = tile_.require<CacheCommitFn>(
            "nfn_native_tile_glimmer_cache_commit_bf16_v1");
        dflash_attention_ = tile_.require<DFlashAttentionFn>(
            "nfn_native_tile_dflash_block_attention_float32_v1");
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
        error_string_ = cuda_.require<CudaGetErrorStringFn>("cudaGetErrorString");
        int count = 0;
        check_cuda(get_device_count_(&count), "cudaGetDeviceCount");
        if (device_ >= count) {
            throw std::runtime_error("requested Glimmer CUDA device is unavailable");
        }
        set_device();
        check_cuda(stream_create_(&stream_), "cudaStreamCreate");
    }

    ~Runtime() {
        set_device_noexcept();
        if (stream_ != nullptr) stream_destroy_(stream_);
    }

    void set_device() const { check_cuda(set_device_(device_), "cudaSetDevice"); }
    void set_device_noexcept() const noexcept { (void)set_device_(device_); }
    void* stream() const noexcept { return stream_; }
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
    void check_tile(int status, const char* operation) const {
        if (status == kCudaSuccess) return;
        const char* message = tile_error_(status);
        throw std::runtime_error(
            std::string(operation) + " failed: " +
            (message == nullptr ? "unknown Tile-CUDA error" : message));
    }

    PackedValidateFn packed_validate() const noexcept { return packed_validate_; }
    PackedLinearFn linear() const noexcept { return linear_; }
    EmbeddingFn embedding() const noexcept { return embedding_; }
    RmsNormFn rms() const noexcept { return rms_; }
    RopeFn rope() const noexcept { return rope_; }
    GqaFn gqa() const noexcept { return gqa_; }
    CacheCommitFn cache_commit() const noexcept { return cache_commit_; }
    DFlashAttentionFn dflash_attention() const noexcept { return dflash_attention_; }
    GateFn gate() const noexcept { return gate_; }
    LogitTransformFn logit_transform() const noexcept { return logit_transform_; }
    SwiGluFn swiglu() const noexcept { return swiglu_; }
    AddFn add() const noexcept { return add_; }
    ScaleFn scale() const noexcept { return scale_; }

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
    void* stream_ = nullptr;
    AbiVersionFn base_abi_ = nullptr;
    AbiVersionFn strict_abi_ = nullptr;
    AbiVersionFn packed_abi_ = nullptr;
    AbiVersionFn feature_abi_ = nullptr;
    ErrorStringFn tile_error_ = nullptr;
    PackedValidateFn packed_validate_ = nullptr;
    PackedLinearFn linear_ = nullptr;
    EmbeddingFn embedding_ = nullptr;
    RmsNormFn rms_ = nullptr;
    RopeFn rope_ = nullptr;
    GqaFn gqa_ = nullptr;
    CacheCommitFn cache_commit_ = nullptr;
    DFlashAttentionFn dflash_attention_ = nullptr;
    GateFn gate_ = nullptr;
    LogitTransformFn logit_transform_ = nullptr;
    SwiGluFn swiglu_ = nullptr;
    AddFn add_ = nullptr;
    ScaleFn scale_ = nullptr;
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
              checked_mul(config.model_dim, 4, "final hidden"), "final hidden")) {
        const std::int64_t kv_width = checked_mul(
            config.num_kv_heads, config.head_dim, "KV width");
        layers.reserve(checked_size(config.num_layers, "cache layer count"));
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
        }
        allocated_bytes += static_cast<std::int64_t>(final_hidden.bytes());
        this->runtime->zero_async(final_hidden.get(), final_hidden.bytes());
        this->runtime->synchronize();
    }

    std::shared_ptr<Runtime> runtime;
    std::vector<Layer> layers;
    DeviceBuffer final_hidden;
    std::int64_t logical_length = 0;
    std::int64_t allocated_bytes = 0;
};

Cache::Cache(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Cache::~Cache() = default;
std::int64_t Cache::logical_length() const noexcept { return impl_->logical_length; }
std::int64_t Cache::allocated_bytes() const noexcept { return impl_->allocated_bytes; }

class Verification::Impl final {
public:
    Impl(
        std::shared_ptr<Runtime> source_runtime,
        const Config& config,
        std::int64_t source_position,
        std::int64_t source_rows,
        std::int64_t tap_count)
        : runtime(std::move(source_runtime)),
          position(source_position),
          row_count(source_rows),
          kv_width(checked_mul(config.num_kv_heads, config.head_dim, "verification KV width")),
          staged_keys(runtime, checked_size(checked_mul(checked_mul(
              checked_mul(config.num_layers, source_rows, "verification layer rows"),
              kv_width, "verification KV elements"), 4, "verification KV bytes"),
              "verification KV bytes")),
          staged_values(runtime, staged_keys.bytes()),
          staged_final(runtime, checked_size(checked_mul(checked_mul(
              source_rows, config.model_dim, "verification final rows"), 4,
              "verification final bytes"), "verification final bytes")),
          host_logits(checked_size(checked_mul(
              source_rows, config.vocab_size, "verification logits"),
              "verification logits")),
          host_taps(checked_size(checked_mul(checked_mul(
              source_rows, tap_count, "verification tap rows"), config.model_dim,
              "verification taps"), "verification taps")) {}

    std::shared_ptr<Runtime> runtime;
    std::int64_t position = 0;
    std::int64_t row_count = 0;
    std::int64_t kv_width = 0;
    DeviceBuffer staged_keys;
    DeviceBuffer staged_values;
    DeviceBuffer staged_final;
    std::vector<float> host_logits;
    std::vector<float> host_taps;
};

Verification::Verification(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Verification::~Verification() = default;
std::int64_t Verification::rows() const noexcept { return impl_ ? impl_->row_count : 0; }
std::int64_t Verification::position() const noexcept { return impl_ ? impl_->position : 0; }
const std::vector<float>& Verification::logits() const noexcept {
    return impl_->host_logits;
}
const std::vector<float>& Verification::target_taps() const noexcept {
    return impl_->host_taps;
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
        staged_keys = DeviceBuffer(runtime, checked_size(checked_mul(
            checked_mul(config.num_layers, checked_mul(config.num_kv_heads, config.head_dim, "KV width"),
                        "staged rows"), 4, "staged keys"), "staged keys"));
        staged_values = DeviceBuffer(runtime, staged_keys.bytes());
        staged_final = DeviceBuffer(runtime, hidden.bytes());
        logits_buffer = DeviceBuffer(runtime, floats(config.vocab_size, "logits"));
        workspace = static_cast<std::int64_t>(
            hidden.bytes() + normalized.bytes() + residual.bytes() + branch.bytes() +
            query.bytes() + key.bytes() + value.bytes() + gate.bytes() + attention.bytes() +
            mlp_gate.bytes() + mlp_up.bytes() + mlp_activated.bytes() +
            staged_keys.bytes() + staged_values.bytes() + staged_final.bytes() +
            logits_buffer.bytes());
    }

    void call(int status, const char* label) {
        runtime->check_tile(status, label);
        ++launches;
    }

    void linear(
        const DeviceWeight& weight,
        const DeviceBuffer& input,
        DeviceBuffer& output,
        std::int64_t rows = 1) {
        call(runtime->linear()(
            &weight.descriptor, input.as<const float>(), nullptr, output.as<float>(), rows, false),
            "Glimmer packed linear");
    }

    void linear_with_lora(
        const DeviceWeight& weight,
        const DeviceLoraWeight* adapter,
        const DeviceBuffer& input,
        DeviceBuffer& output,
        std::int64_t rows = 1) {
        linear(weight, input, output, rows);
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

    Config config;
    std::shared_ptr<Runtime> runtime;
    DeviceWeight token_embedding;
    DeviceWeight final_norm;
    DeviceWeight lm_head;
    std::vector<DeviceLayer> layers;
    std::vector<DeviceLoraLayer> lora_layers;
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
    DeviceBuffer staged_keys;
    DeviceBuffer staged_values;
    DeviceBuffer staged_final;
    DeviceBuffer logits_buffer;
    DeviceBuffer lora_rank;
    DeviceBuffer lora_delta;
    std::int64_t weight_bytes = 0;
    std::int64_t workspace = 0;
    std::int64_t launches = 0;
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
    std::vector<float>* target_taps) {
    append_input(token_id, nullptr, position, cache, cancelled, tap_layers, target_taps);
}

void Model::append_embedding(
    const std::vector<float>& embedding,
    std::int64_t position,
    const std::shared_ptr<Cache>& cache,
    const std::atomic<bool>& cancelled,
    const std::vector<std::int64_t>* tap_layers,
    std::vector<float>* target_taps) {
    append_input(-1, &embedding, position, cache, cancelled, tap_layers, target_taps);
}

void Model::append_input(
    std::int64_t token_id,
    const std::vector<float>* embedding,
    std::int64_t position,
    const std::shared_ptr<Cache>& cache,
    const std::atomic<bool>& cancelled,
    const std::vector<std::int64_t>* tap_layers,
    std::vector<float>* target_taps) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || !cache || !cache->impl_) {
        throw std::runtime_error("Glimmer CUDA model/cache is closed");
    }
    if (cache->impl_->runtime.get() != impl_->runtime.get() ||
        cache->impl_->logical_length != position ||
        (embedding == nullptr && (token_id < 0 || token_id >= impl_->config.vocab_size)) ||
        (embedding != nullptr && embedding->size() !=
            static_cast<std::size_t>(impl_->config.model_dim)) || position < 0 ||
        position >= impl_->config.max_seq_len) {
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
        target_taps->reserve(checked_size(checked_mul(
            static_cast<std::int64_t>(tap_layers->size()), impl_->config.model_dim,
            "target tap extent"), "target tap extent"));
    }
    throw_if_cancelled(cancelled);
    if (embedding == nullptr) {
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
    const std::size_t kv_bytes = checked_size(checked_mul(kv_width, 4, "KV bytes"), "KV bytes");
    for (std::int64_t layer_index = 0; layer_index < impl_->config.num_layers; ++layer_index) {
        throw_if_cancelled(cancelled);
        DeviceLayer& layer = impl_->layers[static_cast<std::size_t>(layer_index)];
        const DeviceLoraLayer* adapter = impl_->lora_layers.empty()
            ? nullptr : &impl_->lora_layers[static_cast<std::size_t>(layer_index)];
        impl_->runtime->copy_d2d_async(
            impl_->residual.get(), impl_->hidden.get(), impl_->hidden.bytes());
        impl_->rms(
            impl_->hidden, &layer.input_norm, impl_->normalized, 1,
            impl_->config.model_dim, impl_->config.norm_eps);
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
        impl_->rms(
            impl_->query, layer.q_norm ? &*layer.q_norm : nullptr, impl_->query,
            impl_->config.num_heads, impl_->config.head_dim, impl_->config.norm_eps);
        impl_->rms(
            impl_->key, layer.k_norm ? &*layer.k_norm : nullptr, impl_->key,
            impl_->config.num_kv_heads, impl_->config.head_dim, impl_->config.norm_eps);
        if (!impl_->config.gguf_interleaved) {
            impl_->call(impl_->runtime->scale()(
                impl_->query.as<float>(), query_width, impl_->config.q_scale_factor,
                impl_->runtime->stream()), "Glimmer query scale");
        }
        const bool local = layer_index % 4 != 3;
        if (local) {
            impl_->call(impl_->runtime->rope()(
                impl_->query.as<float>(), impl_->key.as<float>(), impl_->config.num_heads,
                impl_->config.num_kv_heads, impl_->config.head_dim, position,
                impl_->config.rope_theta,
                impl_->config.gguf_interleaved
                    ? NFN_NATIVE_TILE_GLIMMER_ROPE_INTERLEAVED
                    : NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT,
                impl_->runtime->stream()), "Glimmer positioned RoPE");
        }
        auto* staged_key = impl_->staged_keys.as<float>() + layer_index * kv_width;
        auto* staged_value = impl_->staged_values.as<float>() + layer_index * kv_width;
        impl_->runtime->copy_d2d_async(staged_key, impl_->key.get(), kv_bytes);
        impl_->runtime->copy_d2d_async(staged_value, impl_->value.get(), kv_bytes);
        Cache::Impl::Layer& cache_layer = cache->impl_->layers[static_cast<std::size_t>(layer_index)];
        NfnNativeTileGlimmerGqaDecodeDescriptorV1 attention{};
        attention.struct_size = sizeof(attention);
        attention.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
        attention.query = impl_->query.as<const float>();
        attention.current_key = impl_->key.as<const float>();
        attention.current_value = impl_->value.as<const float>();
        attention.key_cache_bf16 = cache_layer.keys.as<const std::uint16_t>();
        attention.value_cache_bf16 = cache_layer.values.as<const std::uint16_t>();
        attention.output = impl_->attention.as<float>();
        attention.query_heads = impl_->config.num_heads;
        attention.kv_heads = impl_->config.num_kv_heads;
        attention.head_dim = impl_->config.head_dim;
        attention.position = position;
        attention.first_key_position = local
            ? std::max<std::int64_t>(0, position - impl_->config.sliding_window + 1)
            : 0;
        attention.cache_capacity = cache_layer.capacity;
        attention.cache_row_stride = kv_width;
        attention.scale = 1.0f / std::sqrt(static_cast<float>(impl_->config.head_dim));
        attention.cuda_stream = impl_->runtime->stream();
        impl_->call(impl_->runtime->gqa()(&attention), "Glimmer GQA decode");
        impl_->call(impl_->runtime->gate()(
            impl_->attention.as<const float>(), impl_->gate.as<const float>(),
            impl_->attention.as<float>(), query_width, impl_->runtime->stream()),
            "Glimmer attention gate");
        impl_->linear_with_lora(
            layer.output, adapter && adapter->output ? &*adapter->output : nullptr,
            impl_->attention, impl_->branch);
        impl_->rms(
            impl_->branch, &layer.post_attention_norm, impl_->normalized, 1,
            impl_->config.model_dim, impl_->config.post_norm_eps);
        impl_->call(impl_->runtime->add()(
            impl_->residual.as<const float>(), impl_->normalized.as<const float>(),
            impl_->hidden.as<float>(), impl_->config.model_dim, impl_->runtime->stream()),
            "Glimmer attention residual");

        impl_->runtime->copy_d2d_async(
            impl_->residual.get(), impl_->hidden.get(), impl_->hidden.bytes());
        impl_->rms(
            impl_->hidden, &layer.pre_feedforward_norm, impl_->normalized, 1,
            impl_->config.model_dim, impl_->config.norm_eps);
        impl_->linear_with_lora(
            layer.mlp_gate, adapter && adapter->mlp_gate ? &*adapter->mlp_gate : nullptr,
            impl_->normalized, impl_->mlp_gate);
        impl_->linear_with_lora(
            layer.mlp_up, adapter && adapter->mlp_up ? &*adapter->mlp_up : nullptr,
            impl_->normalized, impl_->mlp_up);
        impl_->call(impl_->runtime->swiglu()(
            impl_->mlp_gate.as<const float>(), impl_->mlp_up.as<const float>(),
            impl_->mlp_activated.as<float>(), impl_->config.intermediate_dim,
            impl_->runtime->stream()), "Glimmer SwiGLU");
        impl_->linear_with_lora(
            layer.mlp_down, adapter && adapter->mlp_down ? &*adapter->mlp_down : nullptr,
            impl_->mlp_activated, impl_->branch);
        impl_->rms(
            impl_->branch, &layer.post_feedforward_norm, impl_->normalized, 1,
            impl_->config.model_dim, impl_->config.post_norm_eps);
        impl_->call(impl_->runtime->add()(
            impl_->residual.as<const float>(), impl_->normalized.as<const float>(),
            impl_->hidden.as<float>(), impl_->config.model_dim, impl_->runtime->stream()),
            "Glimmer feedforward residual");
        if (tap_layers != nullptr && std::binary_search(
                tap_layers->begin(), tap_layers->end(), layer_index)) {
            impl_->runtime->synchronize();
            const std::size_t offset = target_taps->size();
            target_taps->resize(offset + checked_size(
                impl_->config.model_dim, "target tap width"));
            impl_->runtime->copy_d2h(
                target_taps->data() + static_cast<std::ptrdiff_t>(offset),
                impl_->hidden.get(), impl_->hidden.bytes());
        }
    }
    impl_->rms(
        impl_->hidden, &impl_->final_norm, impl_->staged_final, 1,
        impl_->config.model_dim, impl_->config.norm_eps);
    throw_if_cancelled(cancelled);

    // Commit every layer only after all model math and the final norm have
    // launched successfully.  A caller cancellation cannot expose a partial
    // token cache; runtime failures cause the owning session to rebuild.
    for (std::int64_t layer_index = 0; layer_index < impl_->config.num_layers; ++layer_index) {
        Cache::Impl::Layer& cache_layer = cache->impl_->layers[static_cast<std::size_t>(layer_index)];
        NfnNativeTileGlimmerCacheCommitDescriptorV1 commit{};
        commit.struct_size = sizeof(commit);
        commit.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
        commit.current_key = impl_->staged_keys.as<const float>() + layer_index * kv_width;
        commit.current_value = impl_->staged_values.as<const float>() + layer_index * kv_width;
        commit.key_cache_bf16 = cache_layer.keys.as<std::uint16_t>();
        commit.value_cache_bf16 = cache_layer.values.as<std::uint16_t>();
        commit.kv_heads = impl_->config.num_kv_heads;
        commit.head_dim = impl_->config.head_dim;
        commit.position = position;
        commit.cache_capacity = cache_layer.capacity;
        commit.cache_row_stride = kv_width;
        commit.cuda_stream = impl_->runtime->stream();
        impl_->call(impl_->runtime->cache_commit()(&commit), "Glimmer cache commit");
    }
    impl_->runtime->copy_d2d_async(
        cache->impl_->final_hidden.get(), impl_->staged_final.get(), impl_->staged_final.bytes());
    impl_->runtime->synchronize();
    cache->impl_->logical_length = position + 1;
}

std::shared_ptr<Verification> Model::verify_tokens(
    const std::vector<std::int64_t>& token_ids,
    std::int64_t position,
    const std::shared_ptr<Cache>& cache,
    const std::atomic<bool>& cancelled,
    const std::vector<std::int64_t>* tap_layers) {
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
    DeviceBuffer hidden(impl_->runtime, floats(hidden_count, "verification hidden"));
    DeviceBuffer normalized(impl_->runtime, hidden.bytes());
    DeviceBuffer residual(impl_->runtime, hidden.bytes());
    DeviceBuffer branch(impl_->runtime, hidden.bytes());
    DeviceBuffer query(impl_->runtime, floats(checked_mul(
        rows, query_width, "verification query"), "verification query"));
    DeviceBuffer key(impl_->runtime, floats(checked_mul(
        rows, kv_width, "verification key"), "verification key"));
    DeviceBuffer value(impl_->runtime, key.bytes());
    DeviceBuffer gate(impl_->runtime, query.bytes());
    DeviceBuffer attention(impl_->runtime, query.bytes());
    DeviceBuffer mlp_gate(impl_->runtime, floats(checked_mul(
        rows, impl_->config.intermediate_dim, "verification MLP"), "verification MLP"));
    DeviceBuffer mlp_up(impl_->runtime, mlp_gate.bytes());
    DeviceBuffer mlp_activated(impl_->runtime, mlp_gate.bytes());
    DeviceBuffer logits(impl_->runtime, floats(checked_mul(
        rows, impl_->config.vocab_size, "verification logits"), "verification logits"));
    auto verification = std::shared_ptr<Verification>(new Verification(
        std::make_unique<Verification::Impl>(
            impl_->runtime, impl_->config, position, rows,
            tap_layers == nullptr ? 0 : static_cast<std::int64_t>(tap_layers->size()))));

    const std::size_t hidden_row_bytes = floats(d, "verification hidden row");
    for (std::int64_t row = 0; row < rows; ++row) {
        impl_->call(impl_->runtime->embedding()(
            &impl_->token_embedding.descriptor,
            token_ids[static_cast<std::size_t>(row)],
            hidden.as<float>() + row * d), "Glimmer verification embedding");
    }
    impl_->rms(
        hidden, nullptr, normalized, rows, d, impl_->config.norm_eps);
    impl_->runtime->copy_d2d_async(hidden.get(), normalized.get(), hidden.bytes());

    for (std::int64_t layer_index = 0; layer_index < impl_->config.num_layers; ++layer_index) {
        throw_if_cancelled(cancelled);
        DeviceLayer& layer = impl_->layers[static_cast<std::size_t>(layer_index)];
        const DeviceLoraLayer* adapter = impl_->lora_layers.empty()
            ? nullptr : &impl_->lora_layers[static_cast<std::size_t>(layer_index)];
        impl_->runtime->copy_d2d_async(residual.get(), hidden.get(), hidden.bytes());
        impl_->rms(
            hidden, &layer.input_norm, normalized, rows, d, impl_->config.norm_eps);
        impl_->linear_with_lora(
            layer.q, adapter && adapter->q ? &*adapter->q : nullptr,
            normalized, query, rows);
        impl_->linear_with_lora(
            layer.k, adapter && adapter->k ? &*adapter->k : nullptr,
            normalized, key, rows);
        impl_->linear_with_lora(
            layer.v, adapter && adapter->v ? &*adapter->v : nullptr,
            normalized, value, rows);
        impl_->linear_with_lora(
            layer.gate, adapter && adapter->gate ? &*adapter->gate : nullptr,
            normalized, gate, rows);
        impl_->rms(
            query, layer.q_norm ? &*layer.q_norm : nullptr, query,
            checked_mul(rows, impl_->config.num_heads, "verification Q norm rows"),
            impl_->config.head_dim, impl_->config.norm_eps);
        impl_->rms(
            key, layer.k_norm ? &*layer.k_norm : nullptr, key,
            checked_mul(rows, impl_->config.num_kv_heads, "verification K norm rows"),
            impl_->config.head_dim, impl_->config.norm_eps);
        if (!impl_->config.gguf_interleaved) {
            impl_->call(impl_->runtime->scale()(
                query.as<float>(), checked_mul(rows, query_width, "verification Q scale"),
                impl_->config.q_scale_factor, impl_->runtime->stream()),
                "Glimmer verification query scale");
        }
        const bool local = layer_index % 4 != 3;
        if (local) {
            for (std::int64_t row = 0; row < rows; ++row) {
                impl_->call(impl_->runtime->rope()(
                    query.as<float>() + row * query_width,
                    key.as<float>() + row * kv_width,
                    impl_->config.num_heads, impl_->config.num_kv_heads,
                    impl_->config.head_dim, position + row, impl_->config.rope_theta,
                    impl_->config.gguf_interleaved
                        ? NFN_NATIVE_TILE_GLIMMER_ROPE_INTERLEAVED
                        : NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT,
                    impl_->runtime->stream()), "Glimmer verification positioned RoPE");
            }
        }
        float* staged_key = verification->impl_->staged_keys.as<float>() +
            layer_index * rows * kv_width;
        float* staged_value = verification->impl_->staged_values.as<float>() +
            layer_index * rows * kv_width;
        impl_->runtime->copy_d2d_async(
            staged_key, key.get(), floats(checked_mul(rows, kv_width, "staged K"), "staged K"));
        impl_->runtime->copy_d2d_async(
            staged_value, value.get(), floats(checked_mul(rows, kv_width, "staged V"), "staged V"));
        Cache::Impl::Layer& cache_layer =
            cache->impl_->layers[static_cast<std::size_t>(layer_index)];
        NfnNativeTileDFlashBlockAttentionDescriptorV1 descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
        descriptor.flags = NFN_NATIVE_TILE_BLOCK_ATTENTION_CAUSAL;
        descriptor.query = query.as<const float>();
        descriptor.block_key = key.as<const float>();
        descriptor.block_value = value.as<const float>();
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
        impl_->call(impl_->runtime->dflash_attention()(&descriptor),
                    "Glimmer causal block verification attention");
        impl_->call(impl_->runtime->gate()(
            attention.as<const float>(), gate.as<const float>(), attention.as<float>(),
            checked_mul(rows, query_width, "verification gate"), impl_->runtime->stream()),
            "Glimmer verification attention gate");
        impl_->linear_with_lora(
            layer.output, adapter && adapter->output ? &*adapter->output : nullptr,
            attention, branch, rows);
        impl_->rms(
            branch, &layer.post_attention_norm, normalized, rows, d,
            impl_->config.post_norm_eps);
        impl_->call(impl_->runtime->add()(
            residual.as<const float>(), normalized.as<const float>(), hidden.as<float>(),
            hidden_count, impl_->runtime->stream()),
            "Glimmer verification attention residual");
        impl_->runtime->copy_d2d_async(residual.get(), hidden.get(), hidden.bytes());
        impl_->rms(
            hidden, &layer.pre_feedforward_norm, normalized, rows, d,
            impl_->config.norm_eps);
        impl_->linear_with_lora(
            layer.mlp_gate, adapter && adapter->mlp_gate ? &*adapter->mlp_gate : nullptr,
            normalized, mlp_gate, rows);
        impl_->linear_with_lora(
            layer.mlp_up, adapter && adapter->mlp_up ? &*adapter->mlp_up : nullptr,
            normalized, mlp_up, rows);
        impl_->call(impl_->runtime->swiglu()(
            mlp_gate.as<const float>(), mlp_up.as<const float>(),
            mlp_activated.as<float>(), checked_mul(
                rows, impl_->config.intermediate_dim, "verification SwiGLU"),
            impl_->runtime->stream()), "Glimmer verification SwiGLU");
        impl_->linear_with_lora(
            layer.mlp_down, adapter && adapter->mlp_down ? &*adapter->mlp_down : nullptr,
            mlp_activated, branch, rows);
        impl_->rms(
            branch, &layer.post_feedforward_norm, normalized, rows, d,
            impl_->config.post_norm_eps);
        impl_->call(impl_->runtime->add()(
            residual.as<const float>(), normalized.as<const float>(), hidden.as<float>(),
            hidden_count, impl_->runtime->stream()),
            "Glimmer verification feedforward residual");
        if (tap_layers != nullptr) {
            const auto found = std::lower_bound(
                tap_layers->begin(), tap_layers->end(), layer_index);
            if (found != tap_layers->end() && *found == layer_index) {
                const std::int64_t tap_index = static_cast<std::int64_t>(
                    std::distance(tap_layers->begin(), found));
                for (std::int64_t row = 0; row < rows; ++row) {
                    float* target = verification->impl_->host_taps.data() +
                        (row * static_cast<std::int64_t>(tap_layers->size()) + tap_index) * d;
                    impl_->runtime->copy_d2h(
                        target, hidden.as<const float>() + row * d, hidden_row_bytes);
                }
            }
        }
    }
    impl_->rms(
        hidden, &impl_->final_norm, normalized, rows, d, impl_->config.norm_eps);
    impl_->runtime->copy_d2d_async(
        verification->impl_->staged_final.get(), normalized.get(), normalized.bytes());
    impl_->linear(impl_->lm_head, normalized, logits, rows);
    impl_->call(impl_->runtime->logit_transform()(
        logits.as<float>(), checked_mul(rows, impl_->config.vocab_size, "verification logits"),
        impl_->config.output_multiplier, impl_->config.logit_softcap,
        impl_->runtime->stream()), "Glimmer verification logit transform");
    throw_if_cancelled(cancelled);
    impl_->runtime->synchronize();
    impl_->runtime->copy_d2h(
        verification->impl_->host_logits.data(), logits.get(), logits.bytes());
    return verification;
}

void Model::commit_verification(
    const std::shared_ptr<Cache>& cache,
    const std::shared_ptr<Verification>& verification,
    std::int64_t accepted_rows) {
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
    for (std::int64_t layer_index = 0; layer_index < impl_->config.num_layers; ++layer_index) {
        Cache::Impl::Layer& cache_layer =
            cache->impl_->layers[static_cast<std::size_t>(layer_index)];
        const float* staged_key = verification->impl_->staged_keys.as<const float>() +
            layer_index * rows * kv_width;
        const float* staged_value = verification->impl_->staged_values.as<const float>() +
            layer_index * rows * kv_width;
        for (std::int64_t row = 0; row < accepted_rows; ++row) {
            NfnNativeTileGlimmerCacheCommitDescriptorV1 commit{};
            commit.struct_size = sizeof(commit);
            commit.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
            commit.current_key = staged_key + row * kv_width;
            commit.current_value = staged_value + row * kv_width;
            commit.key_cache_bf16 = cache_layer.keys.as<std::uint16_t>();
            commit.value_cache_bf16 = cache_layer.values.as<std::uint16_t>();
            commit.kv_heads = impl_->config.num_kv_heads;
            commit.head_dim = impl_->config.head_dim;
            commit.position = verification->impl_->position + row;
            commit.cache_capacity = cache_layer.capacity;
            commit.cache_row_stride = kv_width;
            commit.cuda_stream = impl_->runtime->stream();
            impl_->call(impl_->runtime->cache_commit()(&commit),
                        "Glimmer verification cache commit");
        }
    }
    const std::size_t hidden_bytes = checked_size(checked_mul(
        impl_->config.model_dim, 4, "verification final hidden"),
        "verification final hidden");
    impl_->runtime->copy_d2d_async(
        cache->impl_->final_hidden.get(),
        verification->impl_->staged_final.as<const float>() +
            (accepted_rows - 1) * impl_->config.model_dim,
        hidden_bytes);
    impl_->runtime->synchronize();
    cache->impl_->logical_length += accepted_rows;
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

void Model::close() noexcept {
    if (!impl_) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->closed = true;
}

std::int64_t Model::resident_weight_bytes() const noexcept { return impl_->weight_bytes; }
std::int64_t Model::workspace_bytes() const noexcept { return impl_->workspace; }
std::int64_t Model::kernel_launches() const noexcept { return impl_->launches; }
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
        context_input = DeviceBuffer(runtime, floats(checked_mul(
            config.tap_count, config.model_dim, "DFlash tap width"), "DFlash tap input"));
        context_projected = DeviceBuffer(runtime, floats(config.model_dim, "DFlash context"));
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
        workspace = static_cast<std::int64_t>(
            context_input.bytes() + context_projected.bytes() + context_normalized.bytes() +
            hidden.bytes() + normalized.bytes() + residual.bytes() + branch.bytes() +
            query.bytes() + key.bytes() + value.bytes() + attention.bytes() +
            mlp_gate.bytes() + mlp_up.bytes() + mlp_activated.bytes());
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
        call(runtime->linear()(
            &weight.descriptor, input.as<const float>(), nullptr, output.as<float>(), rows, false),
            "DFlash packed linear");
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
    std::int64_t weight_bytes = 0;
    std::int64_t workspace = 0;
    std::int64_t launches = 0;
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
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || !cache || !cache->impl_ || concatenated_target_taps == nullptr ||
        cache->impl_->runtime.get() != impl_->runtime.get() ||
        cache->impl_->logical_length != position || position < 0 ||
        position >= impl_->config.max_seq_len) {
        throw std::runtime_error("DFlash CUDA context/cache position is invalid");
    }
    throw_if_cancelled(cancelled);
    impl_->runtime->copy_h2d_async(
        impl_->context_input.get(), concatenated_target_taps, impl_->context_input.bytes());
    impl_->linear(
        impl_->context_projection, impl_->context_input, impl_->context_projected, 1);
    impl_->rms(
        impl_->context_projected, impl_->context_norm, impl_->context_normalized,
        1, impl_->config.model_dim);
    const std::int64_t query_width = checked_mul(
        impl_->config.num_heads, impl_->config.head_dim, "DFlash query width");
    const std::int64_t kv_width = checked_mul(
        impl_->config.num_kv_heads, impl_->config.head_dim, "DFlash KV width");
    impl_->runtime->zero_async(
        impl_->query.get(), checked_size(checked_mul(query_width, 4, "DFlash dummy Q"),
                                         "DFlash dummy Q"));
    for (std::int64_t layer_index = 0; layer_index < impl_->config.num_layers; ++layer_index) {
        throw_if_cancelled(cancelled);
        DFlashDeviceLayer& layer = impl_->layers[static_cast<std::size_t>(layer_index)];
        impl_->linear(layer.k, impl_->context_normalized, impl_->key, 1);
        impl_->linear(layer.v, impl_->context_normalized, impl_->value, 1);
        impl_->rms(
            impl_->key, layer.k_norm, impl_->key,
            impl_->config.num_kv_heads, impl_->config.head_dim);
        impl_->call(impl_->runtime->rope()(
            impl_->query.as<float>(), impl_->key.as<float>(), impl_->config.num_heads,
            impl_->config.num_kv_heads, impl_->config.head_dim, position,
            impl_->config.rope_theta,
            impl_->config.gguf_interleaved
                ? NFN_NATIVE_TILE_GLIMMER_ROPE_INTERLEAVED
                : NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT,
            impl_->runtime->stream()), "DFlash positioned context RoPE");
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
        commit.position = position;
        commit.cache_capacity = cache->impl_->capacity;
        commit.cache_row_stride = kv_width;
        commit.cuda_stream = impl_->runtime->stream();
        impl_->call(impl_->runtime->cache_commit()(&commit), "DFlash context cache commit");
    }
    impl_->runtime->synchronize();
    cache->impl_->logical_length = position + 1;
}

std::vector<float> DFlashModel::forward_block(
    const float* raw_target_embeddings,
    std::int64_t rows,
    const std::shared_ptr<DFlashCache>& cache,
    const std::atomic<bool>& cancelled) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || !cache || !cache->impl_ || raw_target_embeddings == nullptr ||
        cache->impl_->runtime.get() != impl_->runtime.get() ||
        rows != impl_->config.block_size || cache->impl_->logical_length < 0 ||
        cache->impl_->logical_length + rows > impl_->config.max_seq_len) {
        throw std::runtime_error("DFlash CUDA proposal block/cache is invalid");
    }
    throw_if_cancelled(cancelled);
    impl_->runtime->copy_h2d_async(
        impl_->hidden.get(), raw_target_embeddings, impl_->hidden.bytes());
    const std::int64_t hidden_count = checked_mul(rows, impl_->config.model_dim, "DFlash hidden");
    const std::int64_t query_width = checked_mul(
        impl_->config.num_heads, impl_->config.head_dim, "DFlash query width");
    const std::int64_t kv_width = checked_mul(
        impl_->config.num_kv_heads, impl_->config.head_dim, "DFlash KV width");
    for (std::int64_t layer_index = 0; layer_index < impl_->config.num_layers; ++layer_index) {
        throw_if_cancelled(cancelled);
        DFlashDeviceLayer& layer = impl_->layers[static_cast<std::size_t>(layer_index)];
        impl_->runtime->copy_d2d_async(
            impl_->residual.get(), impl_->hidden.get(), impl_->hidden.bytes());
        impl_->rms(
            impl_->hidden, layer.input_norm, impl_->normalized,
            rows, impl_->config.model_dim);
        impl_->linear(layer.q, impl_->normalized, impl_->query, rows);
        impl_->linear(layer.k, impl_->normalized, impl_->key, rows);
        impl_->linear(layer.v, impl_->normalized, impl_->value, rows);
        impl_->rms(
            impl_->query, layer.q_norm, impl_->query,
            checked_mul(rows, impl_->config.num_heads, "DFlash Q norm rows"),
            impl_->config.head_dim);
        impl_->rms(
            impl_->key, layer.k_norm, impl_->key,
            checked_mul(rows, impl_->config.num_kv_heads, "DFlash K norm rows"),
            impl_->config.head_dim);
        for (std::int64_t row = 0; row < rows; ++row) {
            impl_->call(impl_->runtime->rope()(
                impl_->query.as<float>() + row * query_width,
                impl_->key.as<float>() + row * kv_width,
                impl_->config.num_heads, impl_->config.num_kv_heads,
                impl_->config.head_dim, cache->impl_->logical_length + row,
                impl_->config.rope_theta,
                impl_->config.gguf_interleaved
                    ? NFN_NATIVE_TILE_GLIMMER_ROPE_INTERLEAVED
                    : NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT,
                impl_->runtime->stream()), "DFlash positioned block RoPE");
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
        impl_->runtime->copy_d2d_async(
            impl_->residual.get(), impl_->hidden.get(), impl_->hidden.bytes());
        impl_->rms(
            impl_->hidden, layer.post_attention_norm, impl_->normalized,
            rows, impl_->config.model_dim);
        impl_->linear(layer.mlp_gate, impl_->normalized, impl_->mlp_gate, rows);
        impl_->linear(layer.mlp_up, impl_->normalized, impl_->mlp_up, rows);
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
    std::vector<float> result(checked_size(hidden_count, "DFlash result"));
    impl_->runtime->copy_d2h(result.data(), impl_->normalized.get(), impl_->normalized.bytes());
    return result;
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

}  // namespace neuralfn::resident_glimmer_cuda
