#include "resident_tile_turboquant.h"

#include "resident_turboquant.h"
#include "../native_train/tile_ops.h"

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace neuralfn::resident_dense {
namespace {

constexpr int kCudaSuccess = 0;
constexpr int kCudaMemcpyHostToDevice = 1;
constexpr int kCudaMemcpyDeviceToHost = 2;
constexpr std::int64_t kMaxSequenceLength = 16384;
constexpr std::int64_t kMaxHeadDimension = 256;

static_assert(std::is_standard_layout_v<NfnNativeTileTurboQuantAttentionDescriptorV1>);
static_assert(NFN_NATIVE_TILE_TURBOQUANT_ATTENTION_V1 == 1);

using AbiVersionFn = int (*)();
using TurboQuantForwardFn = int (*)(
    const NfnNativeTileTurboQuantAttentionDescriptorV1*);
using TileErrorStringFn = const char* (*)(int);
using TileStatsResetFn = void (*)();
using TileLaunchCountFn = std::int64_t (*)();

using CudaGetDeviceCountFn = int (*)(int*);
using CudaSetDeviceFn = int (*)(int);
using CudaMallocFn = int (*)(void**, std::size_t);
using CudaFreeFn = int (*)(void*);
using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
using CudaMemcpyAsyncFn = int (*)(void*, const void*, std::size_t, int, void*);
using CudaStreamCreateFn = int (*)(void**);
using CudaStreamDestroyFn = int (*)(void*);
using CudaStreamSynchronizeFn = int (*)(void*);
using CudaGetErrorStringFn = const char* (*)(int);

std::size_t checked_size_mul(
    std::size_t left,
    std::size_t right,
    const char* label) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::runtime_error(std::string("Tile-CUDA TurboQuant size overflow at ") + label);
    }
    return left * right;
}

std::size_t as_size(std::int64_t value, const char* label) {
    if (value < 0 || static_cast<std::uint64_t>(value) >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(std::string("Tile-CUDA TurboQuant invalid size at ") + label);
    }
    return static_cast<std::size_t>(value);
}

void throw_if_cancelled(const std::atomic<bool>& cancelled) {
    if (cancelled.load(std::memory_order_relaxed)) {
        throw ResidentCancellationError("resident inference session was cancelled");
    }
}

class SharedLibrary final {
public:
    SharedLibrary() = default;

    explicit SharedLibrary(const std::string& path) {
        open(path);
    }

    ~SharedLibrary() {
        if (handle_ != nullptr) {
            dlclose(handle_);
        }
    }

    SharedLibrary(const SharedLibrary&) = delete;
    SharedLibrary& operator=(const SharedLibrary&) = delete;

    bool try_open(const std::string& path, std::string* error) {
        if (handle_ != nullptr) {
            throw std::runtime_error("shared library handle is already open");
        }
        dlerror();
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle_ == nullptr) {
            const char* raw = dlerror();
            if (error != nullptr) {
                *error = raw == nullptr ? "dlopen failed" : raw;
            }
            return false;
        }
        path_ = path;
        return true;
    }

    void open(const std::string& path) {
        std::string error;
        if (!try_open(path, &error)) {
            throw std::runtime_error(
                "failed to load Tile-CUDA TurboQuant shared library " + path + ": " + error);
        }
    }

    template <typename Fn>
    Fn require(const char* symbol) const {
        dlerror();
        void* raw = dlsym(handle_, symbol);
        const char* error = dlerror();
        if (raw == nullptr || error != nullptr) {
            throw std::runtime_error(
                "Tile-CUDA TurboQuant shared library " + path_ +
                " is missing required symbol " + symbol);
        }
        static_assert(sizeof(Fn) == sizeof(raw));
        Fn result = nullptr;
        std::memcpy(&result, &raw, sizeof(result));
        return result;
    }

    const std::string& path() const noexcept { return path_; }

private:
    void* handle_ = nullptr;
    std::string path_;
};

std::string canonical_tile_library_path(const std::string& raw) {
    if (raw.empty()) {
        throw std::runtime_error("Tile-CUDA TurboQuant requires tile_ops_lib");
    }
    const std::filesystem::path requested(raw);
    if (!requested.is_absolute()) {
        throw std::runtime_error("Tile-CUDA TurboQuant tile_ops_lib must be an absolute path");
    }
    std::error_code error;
    const std::filesystem::path resolved = std::filesystem::canonical(requested, error);
    if (error || !std::filesystem::is_regular_file(resolved, error) || error) {
        throw std::runtime_error(
            "Tile-CUDA TurboQuant tile_ops_lib is not a readable regular file: " + raw);
    }
    return resolved.string();
}

class Runtime final : public std::enable_shared_from_this<Runtime> {
public:
    explicit Runtime(const TileTurboQuantConfig& config)
        : device_(config.device),
          tile_library_(canonical_tile_library_path(config.tile_ops_lib)) {
        tile_ops_abi_version_ = tile_library_.require<AbiVersionFn>(
            "nfn_native_tile_ops_abi_version");
        strict_math_abi_version_ = tile_library_.require<AbiVersionFn>(
            "nfn_native_tile_strict_math_abi_version");
        feature_abi_version_ = tile_library_.require<AbiVersionFn>(
            "nfn_native_tile_turboquant_attention_abi_version");
        forward_ = tile_library_.require<TurboQuantForwardFn>(
            "nfn_native_tile_turboquant_attention_forward_v1");
        tile_error_string_ = tile_library_.require<TileErrorStringFn>(
            "nfn_native_tile_ops_error_string");
        stats_reset_ = tile_library_.require<TileStatsResetFn>(
            "nfn_native_tile_turboquant_attention_stats_reset");
        launch_count_ = tile_library_.require<TileLaunchCountFn>(
            "nfn_native_tile_turboquant_attention_launch_count");
        if (tile_ops_abi_version_() != 1) {
            throw std::runtime_error("Tile-CUDA TurboQuant requires base Tile ops ABI version 1");
        }
        if (strict_math_abi_version_() != 1) {
            throw std::runtime_error("Tile-CUDA TurboQuant requires strict-math ABI version 1");
        }
        if (feature_abi_version_() != 1) {
            throw std::runtime_error("Tile-CUDA TurboQuant requires attention feature ABI version 1");
        }

        const std::vector<std::string> candidates = config.cuda_runtime_lib.empty()
            ? std::vector<std::string>{
                  "/usr/local/cuda/lib64/libcudart.so.13",
                  "/usr/local/cuda/lib64/libcudart.so",
                  "/usr/local/cuda-13/lib64/libcudart.so.13",
                  "/usr/local/cuda-13/lib64/libcudart.so",
                  "libcudart.so.13",
                  "libcudart.so",
                  "libcudart.so.12",
              }
            : std::vector<std::string>{config.cuda_runtime_lib};
        std::string last_error;
        for (const std::string& candidate : candidates) {
            if (cuda_library_.try_open(candidate, &last_error)) {
                break;
            }
        }
        if (cuda_library_.path().empty()) {
            throw std::runtime_error(
                "Tile-CUDA TurboQuant could not load a CUDA runtime: " + last_error);
        }

        cuda_get_device_count_ = cuda_library_.require<CudaGetDeviceCountFn>("cudaGetDeviceCount");
        cuda_set_device_ = cuda_library_.require<CudaSetDeviceFn>("cudaSetDevice");
        cuda_malloc_ = cuda_library_.require<CudaMallocFn>("cudaMalloc");
        cuda_free_ = cuda_library_.require<CudaFreeFn>("cudaFree");
        cuda_memcpy_ = cuda_library_.require<CudaMemcpyFn>("cudaMemcpy");
        cuda_memcpy_async_ = cuda_library_.require<CudaMemcpyAsyncFn>("cudaMemcpyAsync");
        cuda_stream_create_ = cuda_library_.require<CudaStreamCreateFn>("cudaStreamCreate");
        cuda_stream_destroy_ = cuda_library_.require<CudaStreamDestroyFn>("cudaStreamDestroy");
        cuda_stream_synchronize_ = cuda_library_.require<CudaStreamSynchronizeFn>(
            "cudaStreamSynchronize");
        cuda_error_string_ = cuda_library_.require<CudaGetErrorStringFn>("cudaGetErrorString");

        if (device_ < 0 || device_ > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Tile-CUDA TurboQuant device must be a non-negative int");
        }
        int device_count = 0;
        check_cuda(cuda_get_device_count_(&device_count), "cudaGetDeviceCount");
        if (device_ >= device_count) {
            throw std::runtime_error(
                "Tile-CUDA TurboQuant device " + std::to_string(device_) +
                " is unavailable; visible device count is " + std::to_string(device_count));
        }
        set_device();
    }

    const std::string& tile_library_path() const noexcept { return tile_library_.path(); }
    const std::string& cuda_library_path() const noexcept { return cuda_library_.path(); }
    std::int64_t device() const noexcept { return device_; }

    void set_device() const {
        check_cuda(cuda_set_device_(static_cast<int>(device_)), "cudaSetDevice");
    }

    void* allocate(std::size_t bytes) const {
        set_device();
        void* pointer = nullptr;
        check_cuda(cuda_malloc_(&pointer, std::max<std::size_t>(bytes, 1)), "cudaMalloc");
        if (pointer == nullptr) {
            throw std::runtime_error("Tile-CUDA TurboQuant cudaMalloc returned a null pointer");
        }
        return pointer;
    }

    void free_noexcept(void* pointer) const noexcept {
        if (pointer == nullptr) {
            return;
        }
        cuda_set_device_(static_cast<int>(device_));
        cuda_free_(pointer);
    }

    void copy_h2d(void* target, const void* source, std::size_t bytes) const {
        if (bytes == 0) {
            return;
        }
        set_device();
        check_cuda(
            cuda_memcpy_(target, source, bytes, kCudaMemcpyHostToDevice),
            "cudaMemcpy H2D");
    }

    void copy_h2d_async(
        void* target,
        const void* source,
        std::size_t bytes,
        void* stream) const {
        if (bytes == 0) {
            return;
        }
        check_cuda(
            cuda_memcpy_async_(target, source, bytes, kCudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync H2D");
    }

    void copy_d2h_async(
        void* target,
        const void* source,
        std::size_t bytes,
        void* stream) const {
        if (bytes == 0) {
            return;
        }
        check_cuda(
            cuda_memcpy_async_(target, source, bytes, kCudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync D2H");
    }

    void* create_stream() const {
        set_device();
        void* stream = nullptr;
        check_cuda(cuda_stream_create_(&stream), "cudaStreamCreate");
        if (stream == nullptr) {
            throw std::runtime_error("Tile-CUDA TurboQuant cudaStreamCreate returned null");
        }
        return stream;
    }

    void destroy_stream_noexcept(void* stream) const noexcept {
        if (stream == nullptr) {
            return;
        }
        cuda_set_device_(static_cast<int>(device_));
        cuda_stream_destroy_(stream);
    }

    void synchronize(void* stream) const {
        check_cuda(cuda_stream_synchronize_(stream), "cudaStreamSynchronize");
    }

    void launch(const NfnNativeTileTurboQuantAttentionDescriptorV1& descriptor) const {
        set_device();
        const int status = forward_(&descriptor);
        if (status != kCudaSuccess) {
            const char* detail = tile_error_string_(status);
            std::ostringstream message;
            message << "Tile-CUDA TurboQuant attention launch failed with status " << status;
            if (detail != nullptr) {
                message << ": " << detail;
            }
            throw std::runtime_error(message.str());
        }
    }

private:
    void check_cuda(int status, const char* operation) const {
        if (status == kCudaSuccess) {
            return;
        }
        const char* detail = cuda_error_string_ == nullptr
            ? nullptr
            : cuda_error_string_(status);
        std::ostringstream message;
        message << "Tile-CUDA TurboQuant " << operation << " failed with status " << status;
        if (detail != nullptr) {
            message << ": " << detail;
        }
        throw std::runtime_error(message.str());
    }

    std::int64_t device_ = 0;
    SharedLibrary tile_library_;
    SharedLibrary cuda_library_;
    AbiVersionFn tile_ops_abi_version_ = nullptr;
    AbiVersionFn strict_math_abi_version_ = nullptr;
    AbiVersionFn feature_abi_version_ = nullptr;
    TurboQuantForwardFn forward_ = nullptr;
    TileErrorStringFn tile_error_string_ = nullptr;
    TileStatsResetFn stats_reset_ = nullptr;
    TileLaunchCountFn launch_count_ = nullptr;
    CudaGetDeviceCountFn cuda_get_device_count_ = nullptr;
    CudaSetDeviceFn cuda_set_device_ = nullptr;
    CudaMallocFn cuda_malloc_ = nullptr;
    CudaFreeFn cuda_free_ = nullptr;
    CudaMemcpyFn cuda_memcpy_ = nullptr;
    CudaMemcpyAsyncFn cuda_memcpy_async_ = nullptr;
    CudaStreamCreateFn cuda_stream_create_ = nullptr;
    CudaStreamDestroyFn cuda_stream_destroy_ = nullptr;
    CudaStreamSynchronizeFn cuda_stream_synchronize_ = nullptr;
    CudaGetErrorStringFn cuda_error_string_ = nullptr;
};

class DeviceAllocation final {
public:
    DeviceAllocation() = default;

    DeviceAllocation(std::shared_ptr<Runtime> runtime, std::size_t bytes)
        : runtime_(std::move(runtime)), pointer_(runtime_->allocate(bytes)), bytes_(bytes) {}

    ~DeviceAllocation() {
        reset();
    }

    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    DeviceAllocation(DeviceAllocation&& other) noexcept
        : runtime_(std::move(other.runtime_)),
          pointer_(std::exchange(other.pointer_, nullptr)),
          bytes_(std::exchange(other.bytes_, 0)) {}

    DeviceAllocation& operator=(DeviceAllocation&& other) noexcept {
        if (this != &other) {
            reset();
            runtime_ = std::move(other.runtime_);
            pointer_ = std::exchange(other.pointer_, nullptr);
            bytes_ = std::exchange(other.bytes_, 0);
        }
        return *this;
    }

    void* get() const noexcept { return pointer_; }
    std::size_t size() const noexcept { return bytes_; }

private:
    void reset() noexcept {
        if (runtime_ != nullptr && pointer_ != nullptr) {
            runtime_->free_noexcept(pointer_);
        }
        pointer_ = nullptr;
        bytes_ = 0;
        runtime_.reset();
    }

    std::shared_ptr<Runtime> runtime_;
    void* pointer_ = nullptr;
    std::size_t bytes_ = 0;
};

class Stream final {
public:
    explicit Stream(std::shared_ptr<Runtime> runtime)
        : runtime_(std::move(runtime)), stream_(runtime_->create_stream()) {}

    ~Stream() {
        if (runtime_ != nullptr && stream_ != nullptr) {
            runtime_->destroy_stream_noexcept(stream_);
        }
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    void* get() const noexcept { return stream_; }

private:
    std::shared_ptr<Runtime> runtime_;
    void* stream_ = nullptr;
};

DeviceAllocation upload_doubles(
    const std::shared_ptr<Runtime>& runtime,
    const std::vector<double>& values) {
    const std::size_t bytes = checked_size_mul(
        std::max<std::size_t>(values.size(), 1), sizeof(double), "table bytes");
    DeviceAllocation result(runtime, bytes);
    if (values.empty()) {
        const double zero = 0.0;
        runtime->copy_h2d(result.get(), &zero, sizeof(zero));
    } else {
        runtime->copy_h2d(result.get(), values.data(), values.size() * sizeof(double));
    }
    return result;
}

struct ProfileState final {
    ProfileState(
        std::shared_ptr<Runtime> runtime_value,
        std::shared_ptr<const TurboQuantCodec> codec_value)
        : runtime(std::move(runtime_value)),
          codec(std::move(codec_value)),
          rotation(upload_doubles(runtime, codec->tables().rotation)),
          qjl_projection(upload_doubles(runtime, codec->tables().qjl_projection)),
          centroids_2(upload_doubles(runtime, codec->tables().centroids.at(2))),
          centroids_3(upload_doubles(runtime, codec->tables().centroids.at(3))),
          centroids_4(upload_doubles(runtime, codec->tables().centroids.at(4))) {}

    std::shared_ptr<Runtime> runtime;
    std::shared_ptr<const TurboQuantCodec> codec;
    DeviceAllocation rotation;
    DeviceAllocation qjl_projection;
    DeviceAllocation centroids_2;
    DeviceAllocation centroids_3;
    DeviceAllocation centroids_4;
};

std::size_t cache_bytes(
    std::int64_t num_layers,
    std::int64_t max_seq_len,
    std::int64_t num_heads,
    std::size_t record_bytes,
    const char* label) {
    std::size_t result = checked_size_mul(
        as_size(num_layers, label), as_size(max_seq_len, label), label);
    result = checked_size_mul(result, as_size(num_heads, label), label);
    return checked_size_mul(result, record_bytes, label);
}

}  // namespace

class TileTurboQuantModel::Impl final {
public:
    Impl(
        TileTurboQuantConfig config_value,
        std::int64_t num_layers_value,
        std::int64_t num_heads_value,
        std::int64_t channels_value,
        std::int64_t max_seq_len_value)
        : config(std::move(config_value)),
          num_layers(num_layers_value),
          num_heads(num_heads_value),
          channels(channels_value),
          max_seq_len(max_seq_len_value) {
        if (config.backend != "tile-cuda") {
            throw std::runtime_error(
                "resident TurboQuant attention configuration backend must be 'tile-cuda'");
        }
        if (num_layers <= 0 || num_heads <= 0 || channels <= 0 ||
            channels % num_heads != 0 || max_seq_len <= 0) {
            throw std::runtime_error("Tile-CUDA TurboQuant model geometry is invalid");
        }
        const std::int64_t head_dim = channels / num_heads;
        if (head_dim < 2 || head_dim > kMaxHeadDimension || head_dim % 2 != 0) {
            throw std::runtime_error(
                "Tile-CUDA TurboQuant requires an even attention head dimension in 2..256");
        }
        if (max_seq_len > kMaxSequenceLength) {
            throw std::runtime_error(
                "Tile-CUDA TurboQuant requires a model context length no greater than 16384");
        }
        runtime = std::make_shared<Runtime>(config);
    }

    std::shared_ptr<ProfileState> resolve_profile(
        std::shared_ptr<const TurboQuantCodec> codec) {
        if (!codec || codec->dimension() != channels / num_heads) {
            throw std::runtime_error(
                "Tile-CUDA TurboQuant codec geometry does not match the configured model");
        }
        const std::size_t index = codec->profile() == TurboQuantProfile::Qjl35 ? 1 : 0;
        std::lock_guard<std::mutex> lock(profile_mutex);
        std::shared_ptr<ProfileState>& slot = profiles[index];
        if (slot != nullptr) {
            if (!slot->codec->matches(codec->tables())) {
                throw std::runtime_error(
                    "Tile-CUDA TurboQuant tables changed after this profile was uploaded");
            }
            return slot;
        }
        slot = std::make_shared<ProfileState>(runtime, std::move(codec));
        return slot;
    }

    TileTurboQuantConfig config;
    std::int64_t num_layers = 0;
    std::int64_t num_heads = 0;
    std::int64_t channels = 0;
    std::int64_t max_seq_len = 0;
    std::shared_ptr<Runtime> runtime;
    std::mutex profile_mutex;
    std::array<std::shared_ptr<ProfileState>, 2> profiles;
};

class TileTurboQuantSession::Impl final {
public:
    Impl(
        std::shared_ptr<TileTurboQuantModel::Impl> model_value,
        std::shared_ptr<ProfileState> profile_value)
        : model(std::move(model_value)),
          profile(std::move(profile_value)),
          runtime(model->runtime),
          stream(runtime),
          key_cache(
              runtime,
              cache_bytes(
                  model->num_layers,
                  model->max_seq_len,
                  model->num_heads,
                  profile->codec->key_record_bytes(),
                  "key cache")),
          value_cache(
              runtime,
              cache_bytes(
                  model->num_layers,
                  model->max_seq_len,
                  model->num_heads,
                  profile->codec->value_record_bytes(),
                  "value cache")),
          query(runtime, vector_bytes()),
          current_key(runtime, vector_bytes()),
          current_value(runtime, vector_bytes()),
          output(runtime, vector_bytes()) {}

    std::size_t vector_bytes() const {
        return checked_size_mul(as_size(model->channels, "attention vector"), sizeof(float),
                                "attention vector bytes");
    }

    std::size_t row_bytes(std::size_t record_bytes) const {
        return checked_size_mul(
            as_size(model->num_heads, "cache row heads"), record_bytes, "cache row bytes");
    }

    std::size_t row_offset(
        std::int64_t layer,
        std::int64_t position,
        std::size_t record_bytes) const {
        std::size_t record = checked_size_mul(
            as_size(layer, "cache layer"),
            as_size(model->max_seq_len, "cache capacity"),
            "cache layer positions");
        record += as_size(position, "cache position");
        record = checked_size_mul(
            record, as_size(model->num_heads, "cache heads"), "cache record index");
        return checked_size_mul(record, record_bytes, "cache record offset");
    }

    std::shared_ptr<TileTurboQuantModel::Impl> model;
    std::shared_ptr<ProfileState> profile;
    std::shared_ptr<Runtime> runtime;
    Stream stream;
    DeviceAllocation key_cache;
    DeviceAllocation value_cache;
    DeviceAllocation query;
    DeviceAllocation current_key;
    DeviceAllocation current_value;
    DeviceAllocation output;
    std::atomic<std::int64_t> gpu_launches{0};
    std::atomic<std::int64_t> row_uploads{0};
    std::atomic<std::int64_t> h2d_bytes{0};
    std::atomic<std::int64_t> d2h_bytes{0};
};

TileTurboQuantSession::TileTurboQuantSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

TileTurboQuantSession::~TileTurboQuantSession() = default;

void TileTurboQuantSession::attention(
    std::int64_t layer,
    std::int64_t past_sequence_length,
    const float* query,
    const float* current_key,
    const float* current_value,
    float* output,
    float scale,
    const std::atomic<bool>& cancelled) {
    if (impl_ == nullptr || query == nullptr || current_key == nullptr ||
        current_value == nullptr || output == nullptr ||
        layer < 0 || layer >= impl_->model->num_layers ||
        past_sequence_length < 0 || past_sequence_length >= impl_->model->max_seq_len ||
        !std::isfinite(scale) || !(scale > 0.0f)) {
        throw std::runtime_error("Tile-CUDA TurboQuant attention received invalid inputs");
    }
    throw_if_cancelled(cancelled);
    impl_->runtime->set_device();
    const std::size_t vector_bytes = impl_->vector_bytes();
    impl_->runtime->copy_h2d_async(impl_->query.get(), query, vector_bytes, impl_->stream.get());
    impl_->runtime->copy_h2d_async(
        impl_->current_key.get(), current_key, vector_bytes, impl_->stream.get());
    impl_->runtime->copy_h2d_async(
        impl_->current_value.get(), current_value, vector_bytes, impl_->stream.get());
    impl_->h2d_bytes.fetch_add(
        static_cast<std::int64_t>(vector_bytes * 3), std::memory_order_relaxed);

    NfnNativeTileTurboQuantAttentionDescriptorV1 descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.version = NFN_NATIVE_TILE_TURBOQUANT_ATTENTION_V1;
    descriptor.profile = impl_->profile->codec->profile() == TurboQuantProfile::Qjl35
        ? NFN_NATIVE_TILE_TURBOQUANT_PROFILE_QJL_3_5
        : NFN_NATIVE_TILE_TURBOQUANT_PROFILE_MSE_3_5;
    descriptor.flags = 0;
    descriptor.query = static_cast<const float*>(impl_->query.get());
    descriptor.key_records = static_cast<const std::uint8_t*>(impl_->key_cache.get());
    descriptor.value_records = static_cast<const std::uint8_t*>(impl_->value_cache.get());
    descriptor.current_key = static_cast<const float*>(impl_->current_key.get());
    descriptor.current_value = static_cast<const float*>(impl_->current_value.get());
    descriptor.output = static_cast<float*>(impl_->output.get());
    descriptor.rotation = static_cast<const double*>(impl_->profile->rotation.get());
    descriptor.qjl_projection = impl_->profile->codec->profile() == TurboQuantProfile::Qjl35
        ? static_cast<const double*>(impl_->profile->qjl_projection.get())
        : nullptr;
    descriptor.centroids_2bit = static_cast<const double*>(impl_->profile->centroids_2.get());
    descriptor.centroids_3bit = static_cast<const double*>(impl_->profile->centroids_3.get());
    descriptor.centroids_4bit = static_cast<const double*>(impl_->profile->centroids_4.get());
    descriptor.batch_size = 1;
    descriptor.layer_index = layer;
    descriptor.num_layers = impl_->model->num_layers;
    descriptor.query_heads = impl_->model->num_heads;
    descriptor.kv_heads = impl_->model->num_heads;
    descriptor.head_dim = impl_->model->channels / impl_->model->num_heads;
    descriptor.past_sequence_length = past_sequence_length;
    descriptor.cache_capacity = impl_->model->max_seq_len;
    descriptor.key_record_bytes = static_cast<std::int64_t>(
        impl_->profile->codec->key_record_bytes());
    descriptor.value_record_bytes = static_cast<std::int64_t>(
        impl_->profile->codec->value_record_bytes());
    descriptor.scale = scale;
    descriptor.cuda_stream = impl_->stream.get();

    impl_->runtime->launch(descriptor);
    impl_->gpu_launches.fetch_add(1, std::memory_order_relaxed);
    impl_->runtime->copy_d2h_async(
        output, impl_->output.get(), vector_bytes, impl_->stream.get());
    impl_->d2h_bytes.fetch_add(
        static_cast<std::int64_t>(vector_bytes), std::memory_order_relaxed);
    impl_->runtime->synchronize(impl_->stream.get());
    throw_if_cancelled(cancelled);
}

void TileTurboQuantSession::upload_row(
    std::int64_t layer,
    std::int64_t position,
    const std::uint8_t* key_records,
    std::size_t key_bytes,
    const std::uint8_t* value_records,
    std::size_t value_bytes,
    const std::atomic<bool>& cancelled) {
    if (impl_ == nullptr || key_records == nullptr || value_records == nullptr ||
        layer < 0 || layer >= impl_->model->num_layers ||
        position < 0 || position >= impl_->model->max_seq_len ||
        key_bytes != impl_->row_bytes(impl_->profile->codec->key_record_bytes()) ||
        value_bytes != impl_->row_bytes(impl_->profile->codec->value_record_bytes())) {
        throw std::runtime_error("Tile-CUDA TurboQuant row upload received invalid inputs");
    }
    throw_if_cancelled(cancelled);
    impl_->runtime->set_device();
    auto* key_target = static_cast<std::uint8_t*>(impl_->key_cache.get()) +
        impl_->row_offset(layer, position, impl_->profile->codec->key_record_bytes());
    auto* value_target = static_cast<std::uint8_t*>(impl_->value_cache.get()) +
        impl_->row_offset(layer, position, impl_->profile->codec->value_record_bytes());
    impl_->runtime->copy_h2d_async(
        key_target, key_records, key_bytes, impl_->stream.get());
    impl_->runtime->copy_h2d_async(
        value_target, value_records, value_bytes, impl_->stream.get());
    impl_->runtime->synchronize(impl_->stream.get());
    impl_->row_uploads.fetch_add(1, std::memory_order_relaxed);
    impl_->h2d_bytes.fetch_add(
        static_cast<std::int64_t>(key_bytes + value_bytes), std::memory_order_relaxed);
    throw_if_cancelled(cancelled);
}

TileTurboQuantSessionStats TileTurboQuantSession::stats() const noexcept {
    TileTurboQuantSessionStats result;
    if (impl_ == nullptr) {
        return result;
    }
    result.backend = "tile-cuda";
    result.tile_ops_lib = impl_->runtime->tile_library_path();
    result.cuda_runtime_lib = impl_->runtime->cuda_library_path();
    result.device = impl_->runtime->device();
    result.gpu_launches = impl_->gpu_launches.load(std::memory_order_relaxed);
    result.row_uploads = impl_->row_uploads.load(std::memory_order_relaxed);
    result.h2d_bytes = impl_->h2d_bytes.load(std::memory_order_relaxed);
    result.d2h_bytes = impl_->d2h_bytes.load(std::memory_order_relaxed);
    return result;
}

TileTurboQuantModel::TileTurboQuantModel(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

TileTurboQuantModel::~TileTurboQuantModel() = default;

std::shared_ptr<TileTurboQuantModel> TileTurboQuantModel::configure(
    TileTurboQuantConfig config,
    std::int64_t num_layers,
    std::int64_t num_heads,
    std::int64_t channels,
    std::int64_t max_seq_len) {
    return std::shared_ptr<TileTurboQuantModel>(new TileTurboQuantModel(
        std::make_shared<Impl>(
            std::move(config), num_layers, num_heads, channels, max_seq_len)));
}

std::unique_ptr<TileTurboQuantSession> TileTurboQuantModel::create_session(
    std::shared_ptr<const TurboQuantCodec> codec) {
    if (impl_ == nullptr) {
        throw std::runtime_error("Tile-CUDA TurboQuant model is not configured");
    }
    std::shared_ptr<ProfileState> profile = impl_->resolve_profile(std::move(codec));
    return std::unique_ptr<TileTurboQuantSession>(new TileTurboQuantSession(
        std::make_unique<TileTurboQuantSession::Impl>(impl_, std::move(profile))));
}

TileTurboQuantModelStats TileTurboQuantModel::stats() const noexcept {
    TileTurboQuantModelStats result;
    if (impl_ == nullptr) {
        return result;
    }
    result.configured = true;
    result.backend = "tile-cuda";
    result.tile_ops_lib = impl_->runtime->tile_library_path();
    result.cuda_runtime_lib = impl_->runtime->cuda_library_path();
    result.device = impl_->runtime->device();
    return result;
}

}  // namespace neuralfn::resident_dense
