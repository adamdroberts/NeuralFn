#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace neuralfn::native_train {

#ifndef NFN_TILE_CUDA_OPTIMIZER_TILE_SIZE
#define NFN_TILE_CUDA_OPTIMIZER_TILE_SIZE 1024
#endif

struct FamilyParameterBufferSpec {
    std::string name;
    std::int64_t elements = 0;
    bool trainable = true;
};

struct FamilyDeviceParameterBuffer {
    FamilyParameterBufferSpec spec;
    std::int64_t element_offset = 0;
    void* device_ptr = nullptr;
};

struct FamilyDeviceParameterView {
    std::string_view name;
    float* parameter = nullptr;
    std::int64_t elements = 0;
    std::int64_t element_offset = 0;
    bool trainable = true;

    bool valid() const {
        return parameter != nullptr || elements == 0;
    }
};

struct FamilyFullParameterCheckpointInfo {
    std::filesystem::path path;
    std::int64_t parameter_elements = 0;
    std::int64_t parameter_bytes = 0;
    std::int64_t trained_parameter_elements = 0;
    std::uint64_t parameter_update_checksum = 1469598103934665603ull;
    bool sidecar_written = false;
};

inline void mix_family_float_checksum(std::uint64_t* checksum, std::int64_t index, float value) {
    if (checksum == nullptr) {
        return;
    }
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(value));
    *checksum ^= static_cast<std::uint64_t>(index);
    *checksum *= 1099511628211ull;
    *checksum ^= static_cast<std::uint64_t>(bits);
    *checksum *= 1099511628211ull;
}

inline bool write_family_full_parameter_sidecar(
    const std::filesystem::path& path,
    const std::vector<float>& host_parameters,
    FamilyFullParameterCheckpointInfo* info,
    std::string* error) {
    if (info == nullptr || error == nullptr) {
        return false;
    }
    *info = FamilyFullParameterCheckpointInfo{};
    info->path = path;
    if (host_parameters.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        *error = "family full-parameter checkpoint has too many elements";
        return false;
    }
    info->parameter_elements = static_cast<std::int64_t>(host_parameters.size());
    if (info->parameter_elements > std::numeric_limits<std::int64_t>::max() / 4) {
        *error = "family full-parameter checkpoint byte count overflowed";
        return false;
    }
    info->parameter_bytes = info->parameter_elements * 4;
    info->trained_parameter_elements = info->parameter_elements;
    for (std::int64_t index = 0; index < info->parameter_elements; ++index) {
        mix_family_float_checksum(
            &info->parameter_update_checksum,
            index,
            host_parameters[static_cast<std::size_t>(index)]);
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        *error = "failed to open family full-parameter sidecar for writing";
        return false;
    }
    out.write(
        reinterpret_cast<const char*>(host_parameters.data()),
        static_cast<std::streamsize>(host_parameters.size() * sizeof(float)));
    if (!out) {
        *error = "failed to write family full-parameter sidecar";
        return false;
    }
    info->sidecar_written = true;
    return true;
}

struct FamilyCudaRuntimeApi {
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaMemcpyAsyncFn = int (*)(void*, const void*, std::size_t, int, void*);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);
    using CudaStreamCreateWithFlagsFn = int (*)(void**, unsigned int);
    using CudaStreamDestroyFn = int (*)(void*);
    using CudaStreamSynchronizeFn = int (*)(void*);
    using CudaStreamBeginCaptureFn = int (*)(void*, int);
    using CudaStreamEndCaptureFn = int (*)(void*, void**);
    using CudaGraphInstantiateFn = int (*)(void**, void*, void*, char*, std::size_t);
    using CudaGraphUploadFn = int (*)(void*, void*);
    using CudaGraphLaunchFn = int (*)(void*, void*);
    using CudaGraphDestroyFn = int (*)(void*);
    using CudaGraphExecDestroyFn = int (*)(void*);

    void* handle = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaMemcpyAsyncFn cuda_memcpy_async = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;
    CudaStreamCreateWithFlagsFn cuda_stream_create_with_flags = nullptr;
    CudaStreamDestroyFn cuda_stream_destroy = nullptr;
    CudaStreamSynchronizeFn cuda_stream_synchronize = nullptr;
    CudaStreamBeginCaptureFn cuda_stream_begin_capture = nullptr;
    CudaStreamEndCaptureFn cuda_stream_end_capture = nullptr;
    CudaGraphInstantiateFn cuda_graph_instantiate = nullptr;
    CudaGraphUploadFn cuda_graph_upload = nullptr;
    CudaGraphLaunchFn cuda_graph_launch = nullptr;
    CudaGraphDestroyFn cuda_graph_destroy = nullptr;
    CudaGraphExecDestroyFn cuda_graph_exec_destroy = nullptr;
};

inline std::vector<std::string> family_cuda_runtime_candidates(std::string_view explicit_path = {}) {
    std::vector<std::string> out;
    if (!explicit_path.empty()) {
        out.push_back(std::string(explicit_path));
    }
    out.push_back("libcudart.so");
    out.push_back("libcudart.so.13");
    out.push_back("libcudart.so.12");
    out.push_back("/usr/local/cuda/lib64/libcudart.so");
    return out;
}

template <typename Fn>
inline Fn family_load_symbol(void* handle, const char* name) {
    if (handle == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<Fn>(dlsym(handle, name));
}

inline std::string family_cuda_error(const FamilyCudaRuntimeApi& api, int code, std::string_view context) {
    std::ostringstream out;
    out << context << " failed";
    if (code != 0) {
        out << " with code " << code;
        if (api.cuda_get_error_string != nullptr) {
            const char* text = api.cuda_get_error_string(code);
            if (text != nullptr) {
                out << " (" << text << ")";
            }
        }
    }
    return out.str();
}

inline bool load_family_cuda_runtime(
    std::string_view explicit_path,
    FamilyCudaRuntimeApi* api,
    std::string* error) {
    if (api == nullptr || error == nullptr) {
        return false;
    }
    *api = FamilyCudaRuntimeApi{};
    for (const std::string& candidate : family_cuda_runtime_candidates(explicit_path)) {
        api->handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (api->handle != nullptr) {
            break;
        }
    }
    if (api->handle == nullptr) {
        *error = "failed to load CUDA runtime for family parameter store";
        return false;
    }
    api->cuda_malloc = family_load_symbol<FamilyCudaRuntimeApi::CudaMallocFn>(api->handle, "cudaMalloc");
    api->cuda_free = family_load_symbol<FamilyCudaRuntimeApi::CudaFreeFn>(api->handle, "cudaFree");
    api->cuda_memcpy = family_load_symbol<FamilyCudaRuntimeApi::CudaMemcpyFn>(api->handle, "cudaMemcpy");
    api->cuda_memcpy_async = family_load_symbol<FamilyCudaRuntimeApi::CudaMemcpyAsyncFn>(
        api->handle, "cudaMemcpyAsync");
    api->cuda_device_synchronize =
        family_load_symbol<FamilyCudaRuntimeApi::CudaDeviceSynchronizeFn>(api->handle, "cudaDeviceSynchronize");
    api->cuda_get_error_string =
        family_load_symbol<FamilyCudaRuntimeApi::CudaGetErrorStringFn>(api->handle, "cudaGetErrorString");
    api->cuda_stream_create_with_flags =
        family_load_symbol<FamilyCudaRuntimeApi::CudaStreamCreateWithFlagsFn>(
            api->handle,
            "cudaStreamCreateWithFlags");
    api->cuda_stream_destroy =
        family_load_symbol<FamilyCudaRuntimeApi::CudaStreamDestroyFn>(api->handle, "cudaStreamDestroy");
    api->cuda_stream_synchronize =
        family_load_symbol<FamilyCudaRuntimeApi::CudaStreamSynchronizeFn>(api->handle, "cudaStreamSynchronize");
    api->cuda_stream_begin_capture =
        family_load_symbol<FamilyCudaRuntimeApi::CudaStreamBeginCaptureFn>(api->handle, "cudaStreamBeginCapture");
    api->cuda_stream_end_capture =
        family_load_symbol<FamilyCudaRuntimeApi::CudaStreamEndCaptureFn>(api->handle, "cudaStreamEndCapture");
    api->cuda_graph_instantiate =
        family_load_symbol<FamilyCudaRuntimeApi::CudaGraphInstantiateFn>(api->handle, "cudaGraphInstantiate");
    api->cuda_graph_upload =
        family_load_symbol<FamilyCudaRuntimeApi::CudaGraphUploadFn>(api->handle, "cudaGraphUpload");
    api->cuda_graph_launch =
        family_load_symbol<FamilyCudaRuntimeApi::CudaGraphLaunchFn>(api->handle, "cudaGraphLaunch");
    api->cuda_graph_destroy =
        family_load_symbol<FamilyCudaRuntimeApi::CudaGraphDestroyFn>(api->handle, "cudaGraphDestroy");
    api->cuda_graph_exec_destroy =
        family_load_symbol<FamilyCudaRuntimeApi::CudaGraphExecDestroyFn>(api->handle, "cudaGraphExecDestroy");
    if (api->cuda_malloc == nullptr || api->cuda_free == nullptr || api->cuda_memcpy == nullptr ||
        api->cuda_device_synchronize == nullptr) {
        *error = "CUDA runtime is missing family parameter store allocation/copy symbols";
        dlclose(api->handle);
        *api = FamilyCudaRuntimeApi{};
        return false;
    }
    return true;
}

class FamilyDeviceParameterStore {
public:
    using Initializer = float (*)(std::string_view buffer_name, std::int64_t element_index);

    FamilyDeviceParameterStore() = default;

    explicit FamilyDeviceParameterStore(std::vector<FamilyParameterBufferSpec> specs)
        : specs_(std::move(specs)) {
        std::int64_t offset = 0;
        buffers_.reserve(specs_.size());
        for (const FamilyParameterBufferSpec& spec : specs_) {
            buffers_.push_back({spec, offset, nullptr});
            offset += std::max<std::int64_t>(0, spec.elements);
        }
        total_elements_ = offset;
    }

    FamilyDeviceParameterStore(const FamilyDeviceParameterStore&) = delete;
    FamilyDeviceParameterStore& operator=(const FamilyDeviceParameterStore&) = delete;

    FamilyDeviceParameterStore(FamilyDeviceParameterStore&& other) noexcept {
        move_from(std::move(other));
    }

    FamilyDeviceParameterStore& operator=(FamilyDeviceParameterStore&& other) noexcept {
        if (this != &other) {
            release();
            move_from(std::move(other));
        }
        return *this;
    }

    ~FamilyDeviceParameterStore() {
        release();
    }

    bool allocate(std::string_view cuda_runtime_path, std::string* error) {
        if (error == nullptr) {
            return false;
        }
        if (!runtime_.handle && !load_family_cuda_runtime(cuda_runtime_path, &runtime_, error)) {
            return false;
        }
        if (total_elements_ < 0 || total_elements_ > std::numeric_limits<std::int64_t>::max() / kFloat32Bytes) {
            *error = "family parameter store layout is too large";
            return false;
        }
        for (FamilyDeviceParameterBuffer& buffer : buffers_) {
            if (buffer.spec.elements < 0) {
                *error = "family parameter buffer has negative element count";
                return false;
            }
            if (buffer.device_ptr != nullptr || buffer.spec.elements == 0) {
                continue;
            }
            const std::size_t bytes = checked_bytes(buffer.spec.elements, error);
            if (!error->empty()) {
                return false;
            }
            int status = runtime_.cuda_malloc(&buffer.device_ptr, bytes);
            if (status != 0) {
                *error = family_cuda_error(runtime_, status, "cudaMalloc family parameter " + buffer.spec.name);
                return false;
            }
        }
        allocated_ = true;
        return true;
    }

    bool initialize_deterministic(
        Initializer initializer,
        std::string_view cuda_runtime_path,
        std::string* error) {
        if (initializer == nullptr) {
            if (error != nullptr) {
                *error = "family parameter store initializer is null";
            }
            return false;
        }
        if (!allocate(cuda_runtime_path, error)) {
            return false;
        }
        std::vector<float> chunk(static_cast<std::size_t>(kChunkFloats), 0.0f);
        for (FamilyDeviceParameterBuffer& buffer : buffers_) {
            for (std::int64_t offset = 0; offset < buffer.spec.elements; offset += kChunkFloats) {
                const std::int64_t count = std::min<std::int64_t>(kChunkFloats, buffer.spec.elements - offset);
                for (std::int64_t i = 0; i < count; ++i) {
                    chunk[static_cast<std::size_t>(i)] = initializer(buffer.spec.name, offset + i);
                }
                if (!copy_host_to_device(buffer.device_ptr, offset, chunk.data(), count, error)) {
                    return false;
                }
            }
        }
        return synchronize(error);
    }

    bool load_from_sidecar(
        const std::filesystem::path& sidecar_path,
        std::string_view cuda_runtime_path,
        std::string* error) {
        if (!allocate(cuda_runtime_path, error)) {
            return false;
        }
        const std::int64_t expected_bytes = total_bytes();
        if (expected_bytes < 0) {
            if (error != nullptr) {
                *error = "family parameter store total byte count overflowed";
            }
            return false;
        }
        std::error_code ec;
        const bool exists = std::filesystem::exists(sidecar_path, ec);
        if (ec || !exists) {
            if (error != nullptr) {
                *error = ec ? "failed to stat family parameter sidecar"
                            : "family parameter sidecar does not exist";
            }
            return false;
        }
        const std::uintmax_t actual_size = std::filesystem::file_size(sidecar_path, ec);
        if (ec || actual_size != static_cast<std::uintmax_t>(expected_bytes)) {
            if (error != nullptr) {
                *error = ec ? "failed to stat family parameter sidecar size"
                            : "family parameter sidecar size does not match parameter store layout";
            }
            return false;
        }
        std::ifstream in(sidecar_path, std::ios::binary);
        if (!in) {
            if (error != nullptr) {
                *error = "failed to open family parameter sidecar";
            }
            return false;
        }
        std::vector<float> chunk(static_cast<std::size_t>(kChunkFloats), 0.0f);
        for (FamilyDeviceParameterBuffer& buffer : buffers_) {
            for (std::int64_t offset = 0; offset < buffer.spec.elements; offset += kChunkFloats) {
                const std::int64_t count = std::min<std::int64_t>(kChunkFloats, buffer.spec.elements - offset);
                const std::size_t bytes = checked_bytes(count, error);
                if (!error->empty()) {
                    return false;
                }
                in.read(reinterpret_cast<char*>(chunk.data()), static_cast<std::streamsize>(bytes));
                if (!in) {
                    if (error != nullptr) {
                        *error = "failed to read family parameter sidecar";
                    }
                    return false;
                }
                if (!copy_host_to_device(buffer.device_ptr, offset, chunk.data(), count, error)) {
                    return false;
                }
            }
        }
        loaded_from_sidecar_ = true;
        return synchronize(error);
    }

    bool copy_to_host(std::vector<float>* out, std::string* error) const {
        if (out == nullptr || error == nullptr) {
            return false;
        }
        if (!allocated_) {
            *error = "family parameter store is not allocated";
            return false;
        }
        out->assign(static_cast<std::size_t>(total_elements_), 0.0f);
        for (const FamilyDeviceParameterBuffer& buffer : buffers_) {
            if (buffer.spec.elements == 0) {
                continue;
            }
            const std::size_t bytes = checked_bytes(buffer.spec.elements, error);
            if (!error->empty()) {
                return false;
            }
            int status = runtime_.cuda_memcpy(
                out->data() + buffer.element_offset,
                buffer.device_ptr,
                bytes,
                kCudaMemcpyDeviceToHost);
            if (status != 0) {
                *error = family_cuda_error(runtime_, status, "cudaMemcpy family parameter D2H");
                return false;
            }
        }
        return true;
    }

    bool allocate_temporary(std::size_t bytes, void** out, std::string* error) const {
        if (out == nullptr || error == nullptr) {
            return false;
        }
        *out = nullptr;
        if (runtime_.cuda_malloc == nullptr) {
            *error = "family parameter store CUDA runtime is not loaded";
            return false;
        }
        if (temporary_replay_leases_enabled_) {
            if (!temporary_replay_validation_enabled_) {
                const TemporaryAllocation& allocation =
                    temporary_replay_plan_[temporary_replay_allocate_position_];
                *out = allocation.pointer;
                temporary_replay_allocate_position_ += 1;
                temporary_replay_outstanding_count_ += 1;
                temporary_replay_lease_count_ += 1;
                return true;
            }
            if (temporary_replay_allocate_position_ >= temporary_replay_plan_.size()) {
                *error = "family temporary replay lease exhausted the warmed sequence";
                return false;
            }
            const TemporaryAllocation& allocation =
                temporary_replay_plan_[temporary_replay_allocate_position_];
            if (allocation.bytes != bytes || allocation.pointer == nullptr) {
                *error = "family temporary replay lease size did not match warmed sequence";
                return false;
            }
            *out = allocation.pointer;
            temporary_replay_allocate_position_ += 1;
            temporary_replay_outstanding_count_ += 1;
            temporary_replay_lease_count_ += 1;
            return true;
        }
        for (std::size_t index = 0; index < temporary_pool_.size(); ++index) {
            if (temporary_pool_[index].bytes == bytes) {
                *out = temporary_pool_[index].pointer;
                temporary_pool_.erase(temporary_pool_.begin() + static_cast<std::ptrdiff_t>(index));
                temporary_active_.push_back({bytes, *out});
                temporary_active_high_water_count_ = std::max(
                    temporary_active_high_water_count_,
                    temporary_active_.size());
                if (temporary_replay_recording_enabled_) {
                    temporary_replay_plan_.push_back({bytes, *out});
                }
                return true;
            }
        }
        const int status = runtime_.cuda_malloc(out, bytes);
        if (status != 0) {
            *error = family_cuda_error(runtime_, status, "cudaMalloc family temporary buffer");
            return false;
        }
        temporary_active_.push_back({bytes, *out});
        temporary_active_high_water_count_ = std::max(
            temporary_active_high_water_count_,
            temporary_active_.size());
        if (temporary_replay_recording_enabled_) {
            temporary_replay_plan_.push_back({bytes, *out});
        }
        return true;
    }

    bool free_temporary(void* pointer, std::string* error) const {
        if (error == nullptr) {
            return false;
        }
        if (pointer == nullptr) {
            return true;
        }
        if (runtime_.cuda_free == nullptr) {
            *error = "family parameter store CUDA runtime is not loaded";
            return false;
        }
        if (temporary_replay_leases_enabled_) {
            if (!temporary_replay_validation_enabled_) {
                temporary_replay_free_position_ += 1;
                if (temporary_replay_outstanding_count_ > 0) {
                    temporary_replay_outstanding_count_ -= 1;
                }
                return true;
            }
            if (temporary_replay_free_position_ >= temporary_replay_plan_.size()) {
                *error = "family temporary replay lease freed beyond warmed sequence";
                return false;
            }
            if (temporary_replay_free_position_ >= temporary_replay_free_plan_.size()) {
                *error = "family temporary replay lease freed beyond warmed free sequence";
                return false;
            }
            const TemporaryAllocation& allocation =
                temporary_replay_free_plan_[temporary_replay_free_position_];
            if (allocation.pointer != pointer) {
                *error = "family temporary replay lease free order did not match warmed sequence";
                return false;
            }
            if (temporary_replay_outstanding_count_ == 0) {
                *error = "family temporary replay lease was freed without an outstanding allocation";
                return false;
            }
            temporary_replay_free_position_ += 1;
            temporary_replay_outstanding_count_ -= 1;
            return true;
        }
        for (std::size_t index = 0; index < temporary_active_.size(); ++index) {
            if (temporary_active_[index].pointer == pointer) {
                if (temporary_replay_recording_enabled_) {
                    temporary_replay_free_plan_.push_back(temporary_active_[index]);
                }
                temporary_pool_.push_back(temporary_active_[index]);
                temporary_active_.erase(temporary_active_.begin() + static_cast<std::ptrdiff_t>(index));
                return true;
            }
        }
        *error = "family temporary buffer was freed without an active allocation record";
        return false;
    }

    bool copy_host_bytes_to_device(
        void* destination,
        const void* source,
        std::size_t bytes,
        std::string* error,
        void* stream = nullptr) const {
        if (error == nullptr || (bytes > 0 && (destination == nullptr || source == nullptr))) {
            return false;
        }
        if (runtime_.cuda_memcpy == nullptr) {
            *error = "family parameter store CUDA runtime is not loaded";
            return false;
        }
        const int status = runtime_.cuda_memcpy_async != nullptr
            ? runtime_.cuda_memcpy_async(destination, source, bytes, kCudaMemcpyHostToDevice, stream)
            : runtime_.cuda_memcpy(destination, source, bytes, kCudaMemcpyHostToDevice);
        if (status != 0) {
            *error = family_cuda_error(runtime_, status, "cudaMemcpy family temporary H2D");
            return false;
        }
        return true;
    }

    bool allocate_persistent_workspace(std::size_t bytes, void** out, std::string* error) const {
        if (out == nullptr || error == nullptr) {
            return false;
        }
        *out = nullptr;
        if (bytes == 0) {
            return true;
        }
        if (runtime_.cuda_malloc == nullptr) {
            *error = "family parameter store CUDA runtime is not loaded";
            return false;
        }
        for (const TemporaryAllocation& allocation : persistent_workspace_) {
            if (allocation.bytes >= bytes) {
                *out = allocation.pointer;
                return true;
            }
        }
        void* pointer = nullptr;
        const int status = runtime_.cuda_malloc(&pointer, bytes);
        if (status != 0) {
            *error = family_cuda_error(runtime_, status, "cudaMalloc family persistent workspace");
            return false;
        }
        persistent_workspace_.push_back({bytes, pointer});
        *out = pointer;
        return true;
    }

    bool copy_device_bytes_to_host(
        void* destination,
        const void* source,
        std::size_t bytes,
        std::string* error,
        void* stream = nullptr) const {
        if (error == nullptr || (bytes > 0 && (destination == nullptr || source == nullptr))) {
            return false;
        }
        if (runtime_.cuda_memcpy == nullptr) {
            *error = "family parameter store CUDA runtime is not loaded";
            return false;
        }
        if (stream != nullptr) {
            if (runtime_.cuda_stream_synchronize == nullptr) {
                *error = "CUDA runtime is missing stream synchronization for family temporary D2H";
                return false;
            }
            const int sync_status = runtime_.cuda_stream_synchronize(stream);
            if (sync_status != 0) {
                *error = family_cuda_error(runtime_, sync_status, "cudaStreamSynchronize family temporary D2H");
                return false;
            }
        }
        const int status = runtime_.cuda_memcpy(destination, source, bytes, kCudaMemcpyDeviceToHost);
        if (status != 0) {
            *error = family_cuda_error(runtime_, status, "cudaMemcpy family temporary D2H");
            return false;
        }
        return true;
    }

    bool synchronize_device(std::string* error) const {
        return synchronize(error);
    }

    bool write_host_sidecar(const std::filesystem::path& path, std::string* error) const {
        std::vector<float> host;
        if (!copy_to_host(&host, error)) {
            return false;
        }
        FamilyFullParameterCheckpointInfo info;
        return write_family_full_parameter_sidecar(path, host, &info, error);
    }

    bool overwrite_buffer_from_host(
        std::string_view name,
        const std::vector<float>& host_values,
        std::string* error) {
        if (error == nullptr) {
            return false;
        }
        if (!allocated_) {
            *error = "family parameter store is not allocated";
            return false;
        }
        for (FamilyDeviceParameterBuffer& buffer : buffers_) {
            if (buffer.spec.name != name) {
                continue;
            }
            if (host_values.size() != static_cast<std::size_t>(buffer.spec.elements)) {
                *error = "family parameter overwrite size mismatch for " + buffer.spec.name;
                return false;
            }
            if (!copy_host_to_device(
                    buffer.device_ptr,
                    0,
                    host_values.data(),
                    buffer.spec.elements,
                    error)) {
                return false;
            }
            return synchronize(error);
        }
        *error = "family parameter overwrite buffer not found";
        return false;
    }

    const std::vector<FamilyDeviceParameterBuffer>& buffers() const {
        return buffers_;
    }

    const FamilyDeviceParameterBuffer* find_buffer(std::string_view name) const {
        for (const FamilyDeviceParameterBuffer& buffer : buffers_) {
            if (buffer.spec.name == name) {
                return &buffer;
            }
        }
        return nullptr;
    }

    float* parameter_ptr(std::string_view name) const {
        const FamilyDeviceParameterBuffer* buffer = find_buffer(name);
        return buffer == nullptr ? nullptr : static_cast<float*>(buffer->device_ptr);
    }

    std::int64_t parameter_elements(std::string_view name) const {
        const FamilyDeviceParameterBuffer* buffer = find_buffer(name);
        return buffer == nullptr ? -1 : buffer->spec.elements;
    }

    std::int64_t parameter_offset(std::string_view name) const {
        const FamilyDeviceParameterBuffer* buffer = find_buffer(name);
        return buffer == nullptr ? -1 : buffer->element_offset;
    }

    FamilyDeviceParameterView parameter_view(std::string_view name) const {
        const FamilyDeviceParameterBuffer* buffer = find_buffer(name);
        if (buffer == nullptr) {
            return {};
        }
        return {
            buffer->spec.name,
            static_cast<float*>(buffer->device_ptr),
            buffer->spec.elements,
            buffer->element_offset,
            buffer->spec.trainable,
        };
    }

    std::int64_t total_elements() const {
        return total_elements_;
    }

    std::int64_t total_bytes() const {
        if (total_elements_ < 0 || total_elements_ > std::numeric_limits<std::int64_t>::max() / kFloat32Bytes) {
            return -1;
        }
        return total_elements_ * kFloat32Bytes;
    }

    bool allocated() const {
        return allocated_;
    }

    std::int64_t temporary_pool_buffer_count() const {
        return static_cast<std::int64_t>(temporary_pool_.size());
    }

    std::int64_t temporary_active_buffer_count() const {
        return static_cast<std::int64_t>(temporary_active_.size());
    }

    std::int64_t temporary_metadata_reserved_buffer_count() const {
        return static_cast<std::int64_t>(std::min(
            temporary_pool_.capacity(),
            temporary_active_.capacity()));
    }

    std::int64_t temporary_active_buffer_high_water_count() const {
        return static_cast<std::int64_t>(temporary_active_high_water_count_);
    }

    bool temporary_replay_leases_enabled() const {
        return temporary_replay_leases_enabled_;
    }

    bool temporary_replay_validation_enabled() const {
        return temporary_replay_validation_enabled_;
    }

    std::int64_t temporary_replay_lease_count() const {
        return static_cast<std::int64_t>(temporary_replay_lease_count_);
    }

    bool temporary_replay_leases_ready() const {
        if (temporary_replay_plan_.empty() || temporary_pool_.empty() || !temporary_active_.empty()) {
            return false;
        }
        if (temporary_replay_free_plan_.size() != temporary_replay_plan_.size()) {
            return false;
        }
        for (const TemporaryAllocation& planned : temporary_replay_plan_) {
            if (planned.pointer == nullptr) {
                return false;
            }
            bool found = false;
            for (const TemporaryAllocation& pooled : temporary_pool_) {
                if (pooled.pointer == planned.pointer && pooled.bytes == planned.bytes) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return false;
            }
        }
        for (const TemporaryAllocation& planned_free : temporary_replay_free_plan_) {
            if (planned_free.pointer == nullptr) {
                return false;
            }
            bool found = false;
            for (const TemporaryAllocation& pooled : temporary_pool_) {
                if (pooled.pointer == planned_free.pointer && pooled.bytes == planned_free.bytes) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return false;
            }
        }
        return true;
    }

    std::int64_t temporary_replay_plan_buffer_count() const {
        return static_cast<std::int64_t>(temporary_replay_plan_.size());
    }

    bool begin_temporary_replay_recording(std::string* error) const {
        if (error == nullptr) {
            return false;
        }
        if (!temporary_active_.empty()) {
            *error = "family temporary replay recording requires no active temporary buffers";
            return false;
        }
        temporary_replay_plan_.clear();
        temporary_replay_free_plan_.clear();
        temporary_replay_plan_.reserve(std::max<std::size_t>(temporary_active_high_water_count_, 1));
        temporary_replay_free_plan_.reserve(std::max<std::size_t>(temporary_active_high_water_count_, 1));
        temporary_replay_recording_enabled_ = true;
        return true;
    }

    bool end_temporary_replay_recording(std::string* error) const {
        if (error == nullptr) {
            return false;
        }
        temporary_replay_recording_enabled_ = false;
        if (!temporary_active_.empty()) {
            *error = "family temporary replay recording ended with active temporary buffers";
            temporary_replay_plan_.clear();
            temporary_replay_free_plan_.clear();
            return false;
        }
        if (temporary_replay_plan_.empty()) {
            *error = "family temporary replay recording did not capture any leases";
            return false;
        }
        if (temporary_replay_free_plan_.size() != temporary_replay_plan_.size()) {
            *error = "family temporary replay recording did not capture matching free leases";
            temporary_replay_plan_.clear();
            temporary_replay_free_plan_.clear();
            return false;
        }
        return true;
    }

    bool begin_temporary_replay_leases(std::string* error) const {
        if (error == nullptr) {
            return false;
        }
        if (!temporary_replay_leases_ready()) {
            *error = "family temporary replay leases require a warmed inactive temporary pool";
            return false;
        }
        temporary_replay_allocate_position_ = 0;
        temporary_replay_free_position_ = 0;
        temporary_replay_outstanding_count_ = 0;
        temporary_replay_leases_enabled_ = true;
        return true;
    }

    bool arm_temporary_replay_leases(std::string* error) const {
        if (error == nullptr) {
            return false;
        }
        if (!temporary_replay_leases_ready()) {
            *error = "family temporary replay leases require a warmed inactive temporary pool";
            return false;
        }
        temporary_replay_allocate_position_ = 0;
        temporary_replay_free_position_ = 0;
        temporary_replay_outstanding_count_ = 0;
        temporary_replay_leases_enabled_ = true;
        return true;
    }

    bool arm_temporary_replay_leases(bool validate_each_lease, std::string* error) const {
        if (!arm_temporary_replay_leases(error)) {
            return false;
        }
        temporary_replay_validation_enabled_ = validate_each_lease;
        return true;
    }

    void disarm_temporary_replay_leases() const {
        temporary_replay_leases_enabled_ = false;
        temporary_replay_validation_enabled_ = true;
        temporary_replay_allocate_position_ = 0;
        temporary_replay_free_position_ = 0;
        temporary_replay_outstanding_count_ = 0;
    }

    bool end_temporary_replay_leases(std::string* error) const {
        if (error == nullptr) {
            return false;
        }
        if (!temporary_replay_leases_enabled_) {
            return true;
        }
        if (temporary_replay_outstanding_count_ != 0) {
            *error = "family temporary replay leases ended with outstanding buffers";
            temporary_replay_leases_enabled_ = false;
            temporary_replay_validation_enabled_ = true;
            temporary_replay_allocate_position_ = 0;
            temporary_replay_free_position_ = 0;
            temporary_replay_outstanding_count_ = 0;
            return false;
        }
        if (temporary_replay_allocate_position_ != temporary_replay_plan_.size() ||
            temporary_replay_free_position_ != temporary_replay_free_plan_.size()) {
            *error = "family temporary replay leases did not consume the warmed sequence";
            temporary_replay_leases_enabled_ = false;
            temporary_replay_validation_enabled_ = true;
            temporary_replay_allocate_position_ = 0;
            temporary_replay_free_position_ = 0;
            return false;
        }
        temporary_replay_leases_enabled_ = false;
        temporary_replay_validation_enabled_ = true;
        temporary_replay_allocate_position_ = 0;
        temporary_replay_free_position_ = 0;
        return true;
    }

    void reserve_temporary_metadata_records(std::size_t count) const {
        temporary_pool_.reserve(count);
        temporary_active_.reserve(count);
    }

    bool loaded_from_sidecar() const {
        return loaded_from_sidecar_;
    }

    void release() {
        if (runtime_.cuda_free != nullptr) {
            for (FamilyDeviceParameterBuffer& buffer : buffers_) {
                if (buffer.device_ptr != nullptr) {
                    runtime_.cuda_free(buffer.device_ptr);
                    buffer.device_ptr = nullptr;
                }
            }
            for (const TemporaryAllocation& allocation : temporary_pool_) {
                if (allocation.pointer != nullptr) {
                    runtime_.cuda_free(allocation.pointer);
                }
            }
            for (const TemporaryAllocation& allocation : temporary_active_) {
                if (allocation.pointer != nullptr) {
                    runtime_.cuda_free(allocation.pointer);
                }
            }
            for (const TemporaryAllocation& allocation : persistent_workspace_) {
                if (allocation.pointer != nullptr) {
                    runtime_.cuda_free(allocation.pointer);
                }
            }
        }
        temporary_pool_.clear();
        temporary_active_.clear();
        persistent_workspace_.clear();
        temporary_active_high_water_count_ = 0;
        temporary_replay_leases_enabled_ = false;
        temporary_replay_validation_enabled_ = true;
        temporary_replay_recording_enabled_ = false;
        temporary_replay_allocate_position_ = 0;
        temporary_replay_free_position_ = 0;
        temporary_replay_outstanding_count_ = 0;
        temporary_replay_lease_count_ = 0;
        temporary_replay_plan_.clear();
        temporary_replay_free_plan_.clear();
        if (runtime_.handle != nullptr) {
            dlclose(runtime_.handle);
        }
        runtime_ = FamilyCudaRuntimeApi{};
        allocated_ = false;
        loaded_from_sidecar_ = false;
    }

private:
    static constexpr std::int64_t kFloat32Bytes = 4;
    static constexpr std::int64_t kChunkFloats = 1 << 20;
    static constexpr int kCudaMemcpyHostToDevice = 1;
    static constexpr int kCudaMemcpyDeviceToHost = 2;

    static std::size_t checked_bytes(std::int64_t elements, std::string* error) {
        if (error != nullptr) {
            error->clear();
        }
        if (elements < 0 || elements > std::numeric_limits<std::int64_t>::max() / kFloat32Bytes) {
            if (error != nullptr) {
                *error = "family parameter byte count overflowed";
            }
            return 0;
        }
        const std::int64_t bytes = elements * kFloat32Bytes;
        if (static_cast<std::uint64_t>(bytes) > std::numeric_limits<std::size_t>::max()) {
            if (error != nullptr) {
                *error = "family parameter byte count exceeds size_t";
            }
            return 0;
        }
        return static_cast<std::size_t>(bytes);
    }

    bool copy_host_to_device(
        void* device_ptr,
        std::int64_t element_offset,
        const float* host_ptr,
        std::int64_t elements,
        std::string* error) const {
        if (device_ptr == nullptr && elements > 0) {
            if (error != nullptr) {
                *error = "family parameter device pointer is null";
            }
            return false;
        }
        const std::size_t bytes = checked_bytes(elements, error);
        if (error != nullptr && !error->empty()) {
            return false;
        }
        char* destination = static_cast<char*>(device_ptr) + static_cast<std::size_t>(element_offset * kFloat32Bytes);
        int status = runtime_.cuda_memcpy(destination, host_ptr, bytes, kCudaMemcpyHostToDevice);
        if (status != 0) {
            if (error != nullptr) {
                *error = family_cuda_error(runtime_, status, "cudaMemcpy family parameter H2D");
            }
            return false;
        }
        return true;
    }

    bool synchronize(std::string* error) const {
        int status = runtime_.cuda_device_synchronize();
        if (status != 0) {
            if (error != nullptr) {
                *error = family_cuda_error(runtime_, status, "cudaDeviceSynchronize family parameter store");
            }
            return false;
        }
        return true;
    }

    void move_from(FamilyDeviceParameterStore&& other) {
        specs_ = std::move(other.specs_);
        buffers_ = std::move(other.buffers_);
        runtime_ = other.runtime_;
        total_elements_ = other.total_elements_;
        allocated_ = other.allocated_;
        loaded_from_sidecar_ = other.loaded_from_sidecar_;
        temporary_pool_ = std::move(other.temporary_pool_);
        temporary_active_ = std::move(other.temporary_active_);
        persistent_workspace_ = std::move(other.persistent_workspace_);
        temporary_active_high_water_count_ = other.temporary_active_high_water_count_;
        temporary_replay_leases_enabled_ = other.temporary_replay_leases_enabled_;
        temporary_replay_validation_enabled_ = other.temporary_replay_validation_enabled_;
        temporary_replay_recording_enabled_ = other.temporary_replay_recording_enabled_;
        temporary_replay_allocate_position_ = other.temporary_replay_allocate_position_;
        temporary_replay_free_position_ = other.temporary_replay_free_position_;
        temporary_replay_outstanding_count_ = other.temporary_replay_outstanding_count_;
        temporary_replay_lease_count_ = other.temporary_replay_lease_count_;
        temporary_replay_plan_ = std::move(other.temporary_replay_plan_);
        temporary_replay_free_plan_ = std::move(other.temporary_replay_free_plan_);
        other.runtime_ = FamilyCudaRuntimeApi{};
        other.total_elements_ = 0;
        other.allocated_ = false;
        other.loaded_from_sidecar_ = false;
        other.temporary_pool_.clear();
        other.temporary_active_.clear();
        other.persistent_workspace_.clear();
        other.temporary_active_high_water_count_ = 0;
        other.temporary_replay_leases_enabled_ = false;
        other.temporary_replay_validation_enabled_ = true;
        other.temporary_replay_recording_enabled_ = false;
        other.temporary_replay_allocate_position_ = 0;
        other.temporary_replay_free_position_ = 0;
        other.temporary_replay_outstanding_count_ = 0;
        other.temporary_replay_lease_count_ = 0;
        other.temporary_replay_plan_.clear();
        other.temporary_replay_free_plan_.clear();
        for (FamilyDeviceParameterBuffer& buffer : other.buffers_) {
            buffer.device_ptr = nullptr;
        }
    }

    std::vector<FamilyParameterBufferSpec> specs_;
    std::vector<FamilyDeviceParameterBuffer> buffers_;
    struct TemporaryAllocation {
        std::size_t bytes = 0;
        void* pointer = nullptr;
    };
    mutable std::vector<TemporaryAllocation> temporary_pool_;
    mutable std::vector<TemporaryAllocation> temporary_active_;
    mutable std::vector<TemporaryAllocation> persistent_workspace_;
    mutable std::size_t temporary_active_high_water_count_ = 0;
    mutable bool temporary_replay_leases_enabled_ = false;
    mutable bool temporary_replay_validation_enabled_ = true;
    mutable bool temporary_replay_recording_enabled_ = false;
    mutable std::vector<TemporaryAllocation> temporary_replay_plan_;
    mutable std::vector<TemporaryAllocation> temporary_replay_free_plan_;
    mutable std::size_t temporary_replay_allocate_position_ = 0;
    mutable std::size_t temporary_replay_free_position_ = 0;
    mutable std::size_t temporary_replay_outstanding_count_ = 0;
    mutable std::size_t temporary_replay_lease_count_ = 0;
    FamilyCudaRuntimeApi runtime_;
    std::int64_t total_elements_ = 0;
    bool allocated_ = false;
    bool loaded_from_sidecar_ = false;
};

struct FamilyTileOptimizerApi {
    using GradientAccumulateFn = int (*)(float*, const float*, std::int64_t, float, void*);
    using FillManyFn = int (*)(float* const*, const std::int64_t*, std::int64_t, std::int64_t, float, void*);
    using SumsqPartialsManyFn =
        int (*)(const float* const*, const std::int64_t*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using GlobalNormClipScaleFn = int (*)(const float*, float*, std::int64_t, float, float, void*);
    using AdamWManyWithDeviceScaleFn = int (*)(
        float* const*,
        const float* const*,
        const float*,
        float* const*,
        float* const*,
        const std::int64_t*,
        const float*,
        std::int64_t,
        std::int64_t,
        float,
        float,
        float,
        float,
        float,
        float,
        void*);
    using AdamWManyWithDeviceScaleHyperFn = int (*)(
        float* const*,
        const float* const*,
        const float*,
        float* const*,
        float* const*,
        const std::int64_t*,
        const float*,
        const float*,
        std::int64_t,
        std::int64_t,
        void*);
    using AdamWManyWithDeviceScaleBf16ShadowFn = int (*)(
        float* const*,
        const float* const*,
        const float*,
        float* const*,
        float* const*,
        const std::int64_t*,
        const float*,
        const std::int64_t*,
        std::uint16_t*,
        std::int64_t,
        std::int64_t,
        float,
        float,
        float,
        float,
        float,
        float,
        void*);
    using AdamWManyWithDeviceScaleBf16ShadowHyperFn = int (*)(
        float* const*,
        const float* const*,
        const float*,
        float* const*,
        float* const*,
        const std::int64_t*,
        const float*,
        const std::int64_t*,
        std::uint16_t*,
        const float*,
        std::int64_t,
        std::int64_t,
        void*);

    void* handle = nullptr;
    GradientAccumulateFn gradient_accumulate_float32 = nullptr;
    FillManyFn fill_many_float32 = nullptr;
    SumsqPartialsManyFn sumsq_partials_many_float32 = nullptr;
    GlobalNormClipScaleFn global_norm_clip_scale_float32 = nullptr;
    AdamWManyWithDeviceScaleFn adamw_step_many_with_device_scale_float32 = nullptr;
    AdamWManyWithDeviceScaleHyperFn adamw_step_many_with_device_scale_hyper_float32 = nullptr;
    AdamWManyWithDeviceScaleBf16ShadowFn adamw_step_many_with_device_scale_bf16_shadow_float32 = nullptr;
    AdamWManyWithDeviceScaleBf16ShadowHyperFn adamw_step_many_with_device_scale_bf16_shadow_hyper_float32 = nullptr;
};

inline bool load_family_tile_optimizer_api(
    std::string_view tile_ops_path,
    FamilyTileOptimizerApi* api,
    std::string* error) {
    if (api == nullptr || error == nullptr) {
        return false;
    }
    *api = FamilyTileOptimizerApi{};
    if (tile_ops_path.empty()) {
        *error = "family optimizer Tile ops path is empty";
        return false;
    }
    api->handle = dlopen(std::string(tile_ops_path).c_str(), RTLD_NOW | RTLD_LOCAL);
    if (api->handle == nullptr) {
        const char* raw = dlerror();
        *error = raw == nullptr ? "failed to load Tile ops library for family optimizer" : raw;
        return false;
    }
    api->gradient_accumulate_float32 =
        family_load_symbol<FamilyTileOptimizerApi::GradientAccumulateFn>(
            api->handle,
            "nfn_native_tile_gradient_accumulate_float32");
    api->fill_many_float32 =
        family_load_symbol<FamilyTileOptimizerApi::FillManyFn>(
            api->handle,
            "nfn_native_tile_fill_many_float32");
    api->sumsq_partials_many_float32 =
        family_load_symbol<FamilyTileOptimizerApi::SumsqPartialsManyFn>(
            api->handle,
            "nfn_native_tile_sumsq_partials_many_float32");
    api->global_norm_clip_scale_float32 =
        family_load_symbol<FamilyTileOptimizerApi::GlobalNormClipScaleFn>(
            api->handle,
            "nfn_native_tile_global_norm_clip_scale_float32");
    api->adamw_step_many_with_device_scale_float32 =
        family_load_symbol<FamilyTileOptimizerApi::AdamWManyWithDeviceScaleFn>(
            api->handle,
            "nfn_native_tile_adamw_step_many_with_device_scale_float32");
    api->adamw_step_many_with_device_scale_hyper_float32 =
        family_load_symbol<FamilyTileOptimizerApi::AdamWManyWithDeviceScaleHyperFn>(
            api->handle,
            "nfn_native_tile_adamw_step_many_with_device_scale_hyper_float32");
    api->adamw_step_many_with_device_scale_bf16_shadow_float32 =
        family_load_symbol<FamilyTileOptimizerApi::AdamWManyWithDeviceScaleBf16ShadowFn>(
            api->handle,
            "nfn_native_tile_adamw_step_many_with_device_scale_bf16_shadow_float32");
    api->adamw_step_many_with_device_scale_bf16_shadow_hyper_float32 =
        family_load_symbol<FamilyTileOptimizerApi::AdamWManyWithDeviceScaleBf16ShadowHyperFn>(
            api->handle,
            "nfn_native_tile_adamw_step_many_with_device_scale_bf16_shadow_hyper_float32");
    if (api->gradient_accumulate_float32 == nullptr ||
        api->fill_many_float32 == nullptr ||
        api->sumsq_partials_many_float32 == nullptr ||
        api->global_norm_clip_scale_float32 == nullptr ||
        api->adamw_step_many_with_device_scale_float32 == nullptr) {
        *error = "Tile ops library is missing family optimizer many-tensor symbols";
        dlclose(api->handle);
        *api = FamilyTileOptimizerApi{};
        return false;
    }
    return true;
}

inline double family_scheduled_learning_rate(
    double base_lr,
    std::int64_t step,
    std::int64_t max_steps,
    std::int64_t warmup_steps,
    std::string_view schedule,
    double final_lr_fraction) {
    if (base_lr <= 0.0) {
        return 0.0;
    }
    const std::int64_t warmup = std::max<std::int64_t>(0, warmup_steps);
    if (warmup > 0 && step <= warmup) {
        return base_lr * (static_cast<double>(std::max<std::int64_t>(1, step)) / static_cast<double>(warmup));
    }
    if (schedule == "constant") {
        return base_lr;
    }
    const double floor_fraction = std::clamp(final_lr_fraction, 0.0, 1.0);
    const double floor_lr = base_lr * floor_fraction;
    const std::int64_t decay_steps = std::max<std::int64_t>(1, max_steps - warmup);
    const double progress = std::clamp(
        static_cast<double>(std::max<std::int64_t>(0, step - warmup)) / static_cast<double>(decay_steps),
        0.0,
        1.0);
    constexpr double kPi = 3.14159265358979323846264338327950288;
    const double cosine = 0.5 * (1.0 + std::cos(kPi * progress));
    return floor_lr + (base_lr - floor_lr) * cosine;
}

struct FamilyOptimizerHyperparameters {
    float learning_rate = 0.0006f;
    float beta1 = 0.9f;
    float beta2 = 0.95f;
    float eps = 1.0e-8f;
    float weight_decay = 0.02f;
    float grad_clip_norm = 1.0f;
    float grad_clip_eps = 1.0e-6f;
    float bias_correction1 = 0.1f;
    float sqrt_bias_correction2 = 0.22360679775f;
};

struct FamilyOptimizerBuffer {
    std::string name;
    float* parameter = nullptr;
    float* grad = nullptr;
    float* exp_avg = nullptr;
    float* exp_avg_sq = nullptr;
    std::int64_t elements = 0;
    std::int64_t element_offset = 0;
    std::int64_t bf16_shadow_offset = -1;
    std::int64_t partial_offset = 0;
    float weight_decay = 0.02f;
};

class FamilyOptimizerState;

enum class FamilyProductionBatchFormat {
    kUint16Tokens,
    kUint8Bytes,
};

struct FamilyProductionBatchView {
    FamilyProductionBatchFormat format = FamilyProductionBatchFormat::kUint16Tokens;
    std::int64_t batch_size = 0;
    std::int64_t seq_len = 0;
    const std::uint16_t* tokens_u16 = nullptr;
    const std::uint16_t* targets_u16 = nullptr;
    const std::uint8_t* tokens_u8 = nullptr;
    const std::uint8_t* targets_u8 = nullptr;
    const std::int64_t* semantic_targets = nullptr;
    const std::int64_t* device_semantic_targets = nullptr;
    std::int64_t semantic_dims = 0;
    std::int64_t semantic_terms = 0;
    bool derive_device_semantic_targets_from_tokens = false;
};

struct FamilyProductionHostWorkspace {
    std::vector<std::uint16_t> tokens_u16;
    std::vector<std::int64_t> targets_i64;
    std::vector<float> jepa_mask;
    std::vector<float> jepa_pred;
    std::vector<float> jepa_target;
    std::vector<float> jepa_grad_pred;
    std::vector<float> jepa_grad_target;
    std::vector<float> jepa_grad_target_expanded;
    std::vector<float> jepa_grad_online_pooled;
    std::vector<float> jepa_grad_final_norm;
    std::vector<float> loss_reporting_totals;
    std::vector<float> lm_chunk_loss;
    std::vector<float> jepa_loss;
    std::vector<float> semantic_loss;
    std::vector<float> embedding_gradient;
    std::vector<float> gradient_collect;
    std::vector<std::int64_t> semantic_targets;
    std::vector<std::uint8_t> semantic_target_valid;
    std::vector<float> route_logits;
    std::vector<float> route_weights;
    std::vector<std::int64_t> route_indices;
    std::vector<float> route_weight_gradients;
    std::vector<float> route_gradient;
    std::vector<float> chunk_route_weights;
    std::vector<std::int64_t> chunk_route_indices;
    std::vector<std::int64_t> semantic_hash_indices;
    std::vector<float> semantic_hash_embedding;
    std::vector<float> semantic_table_gate;
    std::vector<float> semantic_dimension_bias;
    std::vector<float> semantic_gate_score_gradient;
};

struct FamilyProductionLosses {
    float total = 0.0f;
    float autoregressive = 0.0f;
    float jepa = 0.0f;
    float router = 0.0f;
    float semantic = 0.0f;
    float auxiliary = 0.0f;
};

struct FamilyProductionStepContext {
    FamilyDeviceParameterStore* parameters = nullptr;
    FamilyOptimizerState* optimizer = nullptr;
    FamilyProductionHostWorkspace* host_workspace = nullptr;
    void* cuda_stream = nullptr;
    std::string tile_ops_lib;
    std::int64_t optimizer_step = 0;
    std::int64_t accumulation_step = 0;
    std::int64_t accumulation_steps = 1;
    bool report_loss_to_host = true;
    float* loss_reporting_device_scalars = nullptr;
    std::int64_t loss_reporting_scalar_count = 0;
};

struct FamilyProductionStepResult {
    bool ok = false;
    bool optimizer_step_applied = false;
    bool optimizer_step_same_stream = false;
    bool family_step_binding_verified = false;
    bool loss_reporting_deferred_to_wrapper = false;
    const float* loss_reporting_device_scalars = nullptr;
    std::int64_t loss_reporting_scalar_count = 0;
    FamilyProductionLosses losses;
    std::int64_t gradient_buffer_count = 0;
    std::int64_t parameter_dependent_gradient_buffer_count = 0;
    std::int64_t persistent_parameter_buffer_count = 0;
    std::int64_t chained_block_layer_count = 0;
    std::int64_t chained_block_row_count = 0;
    std::int64_t sampled_attention_row_count = 0;
    std::int64_t sampled_causal_attention_context_count = 0;
    std::int64_t sampled_seq2seq_cross_attention_row_count = 0;
    std::int64_t sampled_diffusion_masked_denoise_row_count = 0;
    std::int64_t sampled_ttt_inner_update_row_count = 0;
    std::int64_t sampled_universal_recurrent_step_row_count = 0;
    std::int64_t sampled_hnet_byte_patch_row_count = 0;
    std::int64_t sampled_jamba_mamba_state_row_count = 0;
    std::int64_t tile_lm_row_count = 0;
    std::int64_t semantic_target_batch_count = 0;
    std::int64_t semantic_target_row_count = 0;
    std::int64_t semantic_route_bias_count = 0;
    std::int64_t semantic_route_forced_count = 0;
    std::int64_t semantic_route_distillation_count = 0;
    std::int64_t semantic_route_broadcast_count = 0;
    std::int64_t semantic_route_evo_adoption_count = 0;
    std::int64_t auxfree_bias_refresh_count = 0;
    std::int64_t updated_parameter_elements = 0;
    std::string error;
};

class FamilyProductionStep {
public:
    virtual ~FamilyProductionStep() = default;

    virtual const char* name() const = 0;

    virtual FamilyProductionStepResult forward_backward(
        const FamilyProductionBatchView& batch,
        FamilyProductionStepContext* context) = 0;
};

class FamilyOptimizerState {
public:
    FamilyOptimizerState() = default;

    explicit FamilyOptimizerState(const std::vector<FamilyDeviceParameterBuffer>& parameter_buffers) {
        configure(parameter_buffers);
    }

    FamilyOptimizerState(const FamilyOptimizerState&) = delete;
    FamilyOptimizerState& operator=(const FamilyOptimizerState&) = delete;

    FamilyOptimizerState(FamilyOptimizerState&& other) noexcept {
        move_from(std::move(other));
    }

    FamilyOptimizerState& operator=(FamilyOptimizerState&& other) noexcept {
        if (this != &other) {
            release();
            move_from(std::move(other));
        }
        return *this;
    }

    ~FamilyOptimizerState() {
        release();
    }

    void configure(const std::vector<FamilyDeviceParameterBuffer>& parameter_buffers) {
        release();
        buffers_.clear();
        max_elements_ = 0;
        partial_count_ = 0;
        bf16_shadow_elements_ = 0;
        for (const FamilyDeviceParameterBuffer& buffer : parameter_buffers) {
            if (!buffer.spec.trainable || buffer.spec.elements <= 0) {
                continue;
            }
            FamilyOptimizerBuffer item;
            item.name = buffer.spec.name;
            item.parameter = static_cast<float*>(buffer.device_ptr);
            item.elements = buffer.spec.elements;
            item.element_offset = buffer.element_offset;
            item.bf16_shadow_offset = bf16_shadow_elements_;
            item.partial_offset = partial_count_;
            item.weight_decay = default_weight_decay_for_buffer(buffer.spec.name);
            partial_count_ += optimizer_tile_count(buffer.spec.elements);
            bf16_shadow_elements_ += buffer.spec.elements;
            max_elements_ = std::max(max_elements_, buffer.spec.elements);
            buffers_.push_back(item);
        }
    }

    bool allocate(
        std::string_view cuda_runtime_path,
        std::string_view tile_ops_path,
        std::string* error) {
        if (error == nullptr) {
            return false;
        }
        if (buffers_.empty()) {
            *error = "family optimizer has no trainable parameter buffers";
            return false;
        }
        if (!runtime_.handle && !load_family_cuda_runtime(cuda_runtime_path, &runtime_, error)) {
            return false;
        }
        if (!tile_api_.handle && !load_family_tile_optimizer_api(tile_ops_path, &tile_api_, error)) {
            return false;
        }
        for (FamilyOptimizerBuffer& buffer : buffers_) {
            if (buffer.parameter == nullptr) {
                *error = "family optimizer parameter pointer is null for " + buffer.name;
                return false;
            }
            if (!allocate_float_buffer(&buffer.grad, buffer.elements, "grad " + buffer.name, error) ||
                !allocate_float_buffer(&buffer.exp_avg, buffer.elements, "exp_avg " + buffer.name, error) ||
                !allocate_float_buffer(&buffer.exp_avg_sq, buffer.elements, "exp_avg_sq " + buffer.name, error)) {
                return false;
            }
        }
        if (!allocate_float_buffer(&clip_scale_, 1, "clip_scale", error) ||
            !allocate_float_buffer(&optimizer_hyperparameters_, kOptimizerHyperparameterCount, "optimizer_hyperparameters", error) ||
            !allocate_float_buffer(&sumsq_partials_, partial_count_, "sumsq_partials", error)) {
            return false;
        }
        if (tile_api_.adamw_step_many_with_device_scale_bf16_shadow_float32 != nullptr &&
            bf16_shadow_elements_ > 0 &&
            !allocate_device_array(
                &bf16_shadow_bits_,
                static_cast<std::size_t>(bf16_shadow_elements_),
                "bf16_shadow_bits",
                error)) {
            return false;
        }
        if (!allocate_descriptor_arrays(error) || !zero_state(error) || !set_clip_scale_host(1.0f, error)) {
            return false;
        }
        allocated_ = true;
        return synchronize(error);
    }

    bool zero_gradients(std::string* error) {
        if (allocated_ && tile_api_.fill_many_float32 != nullptr &&
            d_grad_ptrs_ != nullptr && d_elements_ != nullptr) {
            std::string device_error;
            if (zero_gradients_device(nullptr, &device_error)) {
                return true;
            }
            if (error != nullptr) {
                error->clear();
            }
        }
        return zero_gradients_host(error);
    }

    bool set_sparse_global_gradients(
        const std::vector<std::pair<std::int64_t, float>>& gradients,
        float scale,
        std::string* error) {
        return accumulate_sparse_global_gradients(gradients, scale, true, error);
    }

    bool accumulate_sparse_global_gradients(
        const std::vector<std::pair<std::int64_t, float>>& gradients,
        float scale,
        bool reset_gradients,
        std::string* error) {
        if (!ready(error)) {
            return false;
        }
        if (gradients.empty()) {
            if (error != nullptr) {
                *error = "family optimizer sparse gradient list is empty";
            }
            return false;
        }
        if (!std::isfinite(scale)) {
            if (error != nullptr) {
                *error = "family optimizer sparse gradient scale is non-finite";
            }
            return false;
        }
        if (reset_gradients && !zero_gradients(error)) {
            return false;
        }
        for (const auto& item : gradients) {
            if (!std::isfinite(item.second)) {
                if (error != nullptr) {
                    *error = "family optimizer sparse gradient value is non-finite";
                }
                return false;
            }
            const std::int64_t global_index = item.first;
            bool copied = false;
            for (FamilyOptimizerBuffer& buffer : buffers_) {
                if (global_index < buffer.element_offset ||
                    global_index >= buffer.element_offset + buffer.elements) {
                    continue;
                }
                const std::int64_t local_index = global_index - buffer.element_offset;
                float value = item.second * scale;
                if (!std::isfinite(value)) {
                    if (error != nullptr) {
                        *error = "family optimizer sparse gradient scaled value is non-finite";
                    }
                    return false;
                }
                if (!reset_gradients) {
                    float existing = 0.0f;
                    const int read_status = runtime_.cuda_memcpy(
                        &existing,
                        buffer.grad + local_index,
                        sizeof(float),
                        kCudaMemcpyDeviceToHost);
                    if (read_status != 0) {
                        if (error != nullptr) {
                            *error = family_cuda_error(runtime_, read_status, "cudaMemcpy family sparse gradient D2H");
                        }
                        return false;
                    }
                    value += existing;
                    if (!std::isfinite(value)) {
                        if (error != nullptr) {
                            *error = "family optimizer sparse accumulated gradient is non-finite";
                        }
                        return false;
                    }
                }
                const int status = runtime_.cuda_memcpy(
                    buffer.grad + local_index,
                    &value,
                    sizeof(float),
                    kCudaMemcpyHostToDevice);
                if (status != 0) {
                    if (error != nullptr) {
                        *error = family_cuda_error(runtime_, status, "cudaMemcpy family sparse gradient H2D");
                    }
                    return false;
                }
                copied = true;
                break;
            }
            if (!copied) {
                if (error != nullptr) {
                    *error = "family optimizer sparse gradient index is outside trainable parameter buffers";
                }
                return false;
            }
        }
        return synchronize(error);
    }

    bool accumulate_gradient(
        std::size_t buffer_index,
        const float* source_grad,
        float scale,
        void* cuda_stream,
        std::string* error) {
        if (!ready(error)) {
            return false;
        }
        if (buffer_index >= buffers_.size() || source_grad == nullptr) {
            if (error != nullptr) {
                *error = "family optimizer gradient accumulation input is invalid";
            }
            return false;
        }
        FamilyOptimizerBuffer& buffer = buffers_[buffer_index];
        const int status =
            tile_api_.gradient_accumulate_float32(buffer.grad, source_grad, buffer.elements, scale, cuda_stream);
        if (status != 0) {
            if (error != nullptr) {
                *error = "nfn_native_tile_gradient_accumulate_float32 failed";
            }
            return false;
        }
        return true;
    }

    float* loss_reporting_device_scalars(std::int64_t count, std::string* error) {
        if (!ready(error)) {
            return nullptr;
        }
        if (count <= 0) {
            if (error != nullptr) {
                *error = "family optimizer loss reporting scalar count is invalid";
            }
            return nullptr;
        }
        if (loss_reporting_scalar_count_ < count) {
            free_device(loss_reporting_device_scalars_);
            if (!allocate_float_buffer(
                    &loss_reporting_device_scalars_,
                    count,
                    "loss_reporting_scalars",
                    error)) {
                loss_reporting_scalar_count_ = 0;
                return nullptr;
            }
            loss_reporting_scalar_count_ = count;
        }
        return loss_reporting_device_scalars_;
    }

    bool compute_global_clip_scale(float max_norm, float eps, void* cuda_stream, std::string* error) {
        if (!ready(error)) {
            return false;
        }
        int status = tile_api_.sumsq_partials_many_float32(
            d_grad_ptrs_,
            d_elements_,
            d_partial_offsets_,
            sumsq_partials_,
            static_cast<std::int64_t>(buffers_.size()),
            max_elements_,
            cuda_stream);
        if (status == 0) {
            status = tile_api_.global_norm_clip_scale_float32(
                sumsq_partials_,
                clip_scale_,
                partial_count_,
                max_norm,
                eps,
                cuda_stream);
        }
        if (status != 0) {
            if (error != nullptr) {
                *error = "family optimizer global norm clip scale kernels failed";
            }
            return false;
        }
        return true;
    }

    bool adamw_step(const FamilyOptimizerHyperparameters& hyper, void* cuda_stream, std::string* error) {
        if (!ready(error)) {
            return false;
        }
        const bool use_device_hyperparameters =
            optimizer_hyperparameters_ != nullptr &&
            ((bf16_shadow_enabled() &&
              tile_api_.adamw_step_many_with_device_scale_bf16_shadow_hyper_float32 != nullptr) ||
             (!bf16_shadow_enabled() &&
              tile_api_.adamw_step_many_with_device_scale_hyper_float32 != nullptr));
        if (use_device_hyperparameters &&
            !copy_optimizer_hyperparameters_to_device(hyper, cuda_stream, error)) {
            return false;
        }
        int status = 0;
        if (bf16_shadow_enabled()) {
            if (use_device_hyperparameters) {
                status = tile_api_.adamw_step_many_with_device_scale_bf16_shadow_hyper_float32(
                    d_param_ptrs_,
                    d_grad_ptrs_,
                    clip_scale_,
                    d_exp_avg_ptrs_,
                    d_exp_avg_sq_ptrs_,
                    d_elements_,
                    d_weight_decays_,
                    d_bf16_shadow_offsets_,
                    bf16_shadow_bits_,
                    optimizer_hyperparameters_,
                    static_cast<std::int64_t>(buffers_.size()),
                    max_elements_,
                    cuda_stream);
            } else {
                status = tile_api_.adamw_step_many_with_device_scale_bf16_shadow_float32(
                    d_param_ptrs_,
                    d_grad_ptrs_,
                    clip_scale_,
                    d_exp_avg_ptrs_,
                    d_exp_avg_sq_ptrs_,
                    d_elements_,
                    d_weight_decays_,
                    d_bf16_shadow_offsets_,
                    bf16_shadow_bits_,
                    static_cast<std::int64_t>(buffers_.size()),
                    max_elements_,
                    hyper.learning_rate,
                    hyper.beta1,
                    hyper.beta2,
                    hyper.eps,
                    hyper.bias_correction1,
                    hyper.sqrt_bias_correction2,
                    cuda_stream);
            }
            if (status != 0) {
                if (error != nullptr) {
                    *error = use_device_hyperparameters
                        ? "nfn_native_tile_adamw_step_many_with_device_scale_bf16_shadow_hyper_float32 failed"
                        : "nfn_native_tile_adamw_step_many_with_device_scale_bf16_shadow_float32 failed";
                }
                return false;
            }
            return true;
        }
        if (use_device_hyperparameters) {
            status = tile_api_.adamw_step_many_with_device_scale_hyper_float32(
                d_param_ptrs_,
                d_grad_ptrs_,
                clip_scale_,
                d_exp_avg_ptrs_,
                d_exp_avg_sq_ptrs_,
                d_elements_,
                d_weight_decays_,
                optimizer_hyperparameters_,
                static_cast<std::int64_t>(buffers_.size()),
                max_elements_,
                cuda_stream);
        } else {
            status = tile_api_.adamw_step_many_with_device_scale_float32(
                d_param_ptrs_,
                d_grad_ptrs_,
                clip_scale_,
                d_exp_avg_ptrs_,
                d_exp_avg_sq_ptrs_,
                d_elements_,
                d_weight_decays_,
                static_cast<std::int64_t>(buffers_.size()),
                max_elements_,
                hyper.learning_rate,
                hyper.beta1,
                hyper.beta2,
                hyper.eps,
                hyper.bias_correction1,
                hyper.sqrt_bias_correction2,
                cuda_stream);
        }
        if (status != 0) {
            if (error != nullptr) {
                *error = use_device_hyperparameters
                    ? "nfn_native_tile_adamw_step_many_with_device_scale_hyper_float32 failed"
                    : "nfn_native_tile_adamw_step_many_with_device_scale_float32 failed";
            }
            return false;
        }
        return true;
    }

    bool optimizer_step(
        const FamilyOptimizerHyperparameters& hyper,
        void* cuda_stream,
        std::string* error) {
        if (cuda_graph_capture_enabled_ && cuda_stream == nullptr && supports_cuda_graph_capture()) {
            return optimizer_step_cuda_graph(hyper, error);
        }
        return optimizer_step_uncaptured(hyper, cuda_stream, error);
    }

    bool optimizer_step_uncaptured(
        const FamilyOptimizerHyperparameters& hyper,
        void* cuda_stream,
        std::string* error) {
        if (!compute_global_clip_scale(hyper.grad_clip_norm, hyper.grad_clip_eps, cuda_stream, error)) {
            return false;
        }
        if (!adamw_step(hyper, cuda_stream, error)) {
            return false;
        }
        if (zero_gradients_device(cuda_stream, error)) {
            return true;
        }
        if (cuda_stream == nullptr) {
            if (error != nullptr) {
                error->clear();
            }
            return zero_gradients_host(error);
        }
        return false;
    }

    void set_cuda_graph_capture_enabled(bool enabled) {
        cuda_graph_capture_enabled_ = enabled;
    }

    bool cuda_graph_capture_enabled() const {
        return cuda_graph_capture_enabled_;
    }

    bool supports_cuda_graph_capture() const {
        return runtime_.cuda_stream_create_with_flags != nullptr &&
               runtime_.cuda_stream_destroy != nullptr &&
               runtime_.cuda_stream_synchronize != nullptr &&
               runtime_.cuda_stream_begin_capture != nullptr &&
               runtime_.cuda_stream_end_capture != nullptr &&
               runtime_.cuda_graph_instantiate != nullptr &&
               runtime_.cuda_graph_upload != nullptr &&
               runtime_.cuda_graph_launch != nullptr &&
               runtime_.cuda_graph_destroy != nullptr &&
               runtime_.cuda_graph_exec_destroy != nullptr &&
               tile_api_.fill_many_float32 != nullptr;
    }

    bool supports_production_step_stream() const {
        return runtime_.cuda_stream_create_with_flags != nullptr &&
               runtime_.cuda_stream_destroy != nullptr &&
               runtime_.cuda_stream_synchronize != nullptr;
    }

    void* production_step_stream(std::string* error) {
        if (!ready(error)) {
            return nullptr;
        }
        if (!supports_production_step_stream()) {
            if (error != nullptr) {
                *error = "family optimizer CUDA runtime symbols are missing production-step stream support";
            }
            return nullptr;
        }
        if (production_step_stream_ == nullptr) {
            const int status =
                runtime_.cuda_stream_create_with_flags(&production_step_stream_, kCudaStreamNonBlocking);
            if (status != 0) {
                if (error != nullptr) {
                    *error = family_cuda_error(runtime_, status, "cudaStreamCreateWithFlags family production step");
                }
                return nullptr;
            }
        }
        return production_step_stream_;
    }

    bool synchronize_stream(void* cuda_stream, std::string* error) const {
        if (cuda_stream == nullptr) {
            return synchronize(error);
        }
        if (runtime_.cuda_stream_synchronize == nullptr) {
            if (error != nullptr) {
                *error = "family optimizer CUDA runtime is missing stream synchronization";
            }
            return false;
        }
        const int status = runtime_.cuda_stream_synchronize(cuda_stream);
        if (status != 0) {
            if (error != nullptr) {
                *error = family_cuda_error(runtime_, status, "cudaStreamSynchronize family production step");
            }
            return false;
        }
        return true;
    }

    bool zero_gradients_on_stream(void* cuda_stream, std::string* error) {
        if (zero_gradients_device(cuda_stream, error)) {
            return true;
        }
        if (cuda_stream == nullptr) {
            if (error != nullptr) {
                error->clear();
            }
            return zero_gradients_host(error);
        }
        return false;
    }

    bool cuda_graph_instantiated() const {
        return optimizer_graph_exec_ != nullptr;
    }

    std::int64_t cuda_graph_capture_count() const {
        return cuda_graph_capture_count_;
    }

    std::int64_t cuda_graph_replay_count() const {
        return cuda_graph_replay_count_;
    }

    bool cuda_graph_uses_device_hyperparameters() const {
        return optimizer_graph_uses_device_hyperparameters_;
    }

    bool cuda_graph_device_hyperparameters_supported() const {
        return device_hyperparameter_adamw_available();
    }

    std::int64_t production_step_graph_capture_count() const {
        return production_step_graph_capture_count_;
    }

    std::int64_t production_step_graph_replay_count() const {
        return production_step_graph_replay_count_;
    }

    bool production_step_graph_replay_ready(bool report_loss_to_host) const {
        return production_step_graph_exec_ != nullptr &&
               production_step_graph_uses_device_hyperparameters_ &&
               production_step_graph_report_loss_to_host_ == report_loss_to_host;
    }

    bool stage_optimizer_hyperparameters_for_graph_replay(
        const FamilyOptimizerHyperparameters& hyper,
        std::string* error) {
        if (optimizer_hyperparameters_ == nullptr) {
            if (error != nullptr) {
                *error = "family optimizer device hyperparameter buffer is not allocated";
            }
            return false;
        }
        set_optimizer_hyperparameters_host(hyper);
        return true;
    }

    bool begin_production_step_graph_capture(void* cuda_stream, std::string* error) {
        if (!ready(error)) {
            return false;
        }
        if (cuda_stream == nullptr) {
            if (error != nullptr) {
                *error = "family production-step graph capture requires a CUDA stream";
            }
            return false;
        }
        if (!supports_cuda_graph_capture()) {
            if (error != nullptr) {
                *error = "family production-step CUDA graph runtime symbols are missing";
            }
            return false;
        }
        const int status = runtime_.cuda_stream_begin_capture(cuda_stream, kCudaStreamCaptureModeGlobal);
        if (status != 0) {
            if (error != nullptr) {
                *error = family_cuda_error(runtime_, status, "cudaStreamBeginCapture family production step");
            }
            return false;
        }
        return true;
    }

    bool abandon_production_step_graph_capture(void* cuda_stream) {
        if (cuda_stream == nullptr || runtime_.cuda_stream_end_capture == nullptr) {
            return false;
        }
        void* graph = nullptr;
        const int status = runtime_.cuda_stream_end_capture(cuda_stream, &graph);
        if (graph != nullptr && runtime_.cuda_graph_destroy != nullptr) {
            runtime_.cuda_graph_destroy(graph);
        }
        return status == 0;
    }

    bool end_retain_launch_production_step_graph_capture(
        void* cuda_stream,
        bool report_loss_to_host,
        std::string* error) {
        if (cuda_stream == nullptr) {
            if (error != nullptr) {
                *error = "family production-step graph launch requires a CUDA stream";
            }
            return false;
        }
        void* graph = nullptr;
        int status = runtime_.cuda_stream_end_capture(cuda_stream, &graph);
        if (status != 0 || graph == nullptr) {
            if (error != nullptr) {
                *error = family_cuda_error(runtime_, status, "cudaStreamEndCapture family production step");
            }
            return false;
        }
        void* exec = nullptr;
        status = runtime_.cuda_graph_instantiate(&exec, graph, nullptr, nullptr, 0);
        const int destroy_status = runtime_.cuda_graph_destroy(graph);
        if (status != 0 || destroy_status != 0 || exec == nullptr) {
            if (exec != nullptr) {
                runtime_.cuda_graph_exec_destroy(exec);
            }
            if (error != nullptr) {
                *error = status != 0
                    ? family_cuda_error(runtime_, status, "cudaGraphInstantiate family production step")
                    : family_cuda_error(runtime_, destroy_status, "cudaGraphDestroy family production step");
            }
            return false;
        }
        status = runtime_.cuda_graph_upload(exec, cuda_stream);
        if (status != 0) {
            runtime_.cuda_graph_exec_destroy(exec);
            if (error != nullptr) {
                *error = family_cuda_error(runtime_, status, "cudaGraphUpload family production step");
            }
            return false;
        }
        destroy_production_step_graph();
        production_step_graph_exec_ = exec;
        production_step_graph_report_loss_to_host_ = report_loss_to_host;
        production_step_graph_uses_device_hyperparameters_ = device_hyperparameter_adamw_available();
        production_step_graph_capture_count_ += 1;
        return launch_retained_production_step_graph(cuda_stream, error);
    }

    bool end_launch_destroy_production_step_graph_capture(void* cuda_stream, std::string* error) {
        return end_retain_launch_production_step_graph_capture(cuda_stream, true, error);
    }

    bool launch_retained_production_step_graph(void* cuda_stream, std::string* error) {
        if (cuda_stream == nullptr) {
            if (error != nullptr) {
                *error = "family retained production-step graph launch requires a CUDA stream";
            }
            return false;
        }
        if (production_step_graph_exec_ == nullptr) {
            if (error != nullptr) {
                *error = "family retained production-step graph is not instantiated";
            }
            return false;
        }
        int status = runtime_.cuda_graph_launch(production_step_graph_exec_, cuda_stream);
        if (status != 0) {
            if (error != nullptr) {
                *error = family_cuda_error(runtime_, status, "cudaGraphLaunch family production step");
            }
            return false;
        }
        status = runtime_.cuda_stream_synchronize(cuda_stream);
        if (status != 0) {
            if (error != nullptr) {
                *error = family_cuda_error(runtime_, status, "cudaStreamSynchronize family production step graph");
            }
            return false;
        }
        production_step_graph_replay_count_ += 1;
        return true;
    }

    std::string cuda_graph_disable_reason() const {
        if (!cuda_graph_capture_enabled_) {
            return "disabled";
        }
        if (!supports_cuda_graph_capture()) {
            return "cuda-graph-runtime-symbols-missing";
        }
        return "";
    }

    bool has_bf16_shadow_adamw() const {
        return tile_api_.adamw_step_many_with_device_scale_bf16_shadow_float32 != nullptr;
    }

    bool bf16_shadow_enabled() const {
        return tile_api_.adamw_step_many_with_device_scale_bf16_shadow_float32 != nullptr &&
               bf16_shadow_bits_ != nullptr &&
               d_bf16_shadow_offsets_ != nullptr;
    }

    std::int64_t bf16_shadow_elements() const {
        return bf16_shadow_elements_;
    }

    const std::vector<FamilyOptimizerBuffer>& buffers() const {
        return buffers_;
    }

    std::size_t trainable_buffer_index(std::string_view name) const {
        for (std::size_t index = 0; index < buffers_.size(); ++index) {
            if (buffers_[index].name == name) {
                return index;
            }
        }
        return buffers_.size();
    }

    std::int64_t partial_count() const {
        return partial_count_;
    }

    std::int64_t max_elements() const {
        return max_elements_;
    }

    float* clip_scale_device() const {
        return clip_scale_;
    }

    bool write_checkpoint(
        const std::filesystem::path& path,
        std::int64_t completed_optimizer_steps,
        std::string* error) const {
        if (error == nullptr || completed_optimizer_steps < 0) {
            return false;
        }
        if (!allocated_ || runtime_.cuda_memcpy == nullptr) {
            *error = "family optimizer is not allocated";
            return false;
        }
        constexpr std::uint64_t kMagic = 0x314D4954504F4E46ull;
        constexpr std::int64_t kVersion = 2;
        std::int64_t total_elements = 0;
        for (const FamilyOptimizerBuffer& buffer : buffers_) {
            total_elements += buffer.elements;
        }
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            *error = "failed to open family optimizer checkpoint for writing";
            return false;
        }
        const std::int64_t buffer_count = static_cast<std::int64_t>(buffers_.size());
        out.write(reinterpret_cast<const char*>(&kMagic), sizeof(kMagic));
        out.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
        out.write(reinterpret_cast<const char*>(&completed_optimizer_steps), sizeof(completed_optimizer_steps));
        out.write(reinterpret_cast<const char*>(&buffer_count), sizeof(buffer_count));
        out.write(reinterpret_cast<const char*>(&total_elements), sizeof(total_elements));
        out.write(reinterpret_cast<const char*>(&bf16_shadow_elements_), sizeof(bf16_shadow_elements_));
        constexpr std::int64_t kChunkFloats = 1 << 20;
        std::vector<float> chunk(static_cast<std::size_t>(kChunkFloats), 0.0f);
        auto write_device_state = [&](const float* device, std::int64_t elements, std::string_view name) {
            for (std::int64_t offset = 0; offset < elements; offset += kChunkFloats) {
                const std::int64_t count = std::min<std::int64_t>(kChunkFloats, elements - offset);
                const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);
                const int status = runtime_.cuda_memcpy(
                    chunk.data(), device + offset, bytes, kCudaMemcpyDeviceToHost);
                if (status != 0) {
                    *error = family_cuda_error(
                        runtime_, status, "cudaMemcpy family optimizer " + std::string(name) + " D2H");
                    return false;
                }
                out.write(reinterpret_cast<const char*>(chunk.data()), static_cast<std::streamsize>(bytes));
                if (!out) {
                    *error = "failed to write family optimizer " + std::string(name);
                    return false;
                }
            }
            return true;
        };
        for (const FamilyOptimizerBuffer& buffer : buffers_) {
            if (!write_device_state(buffer.exp_avg, buffer.elements, "exp_avg")) {
                return false;
            }
        }
        for (const FamilyOptimizerBuffer& buffer : buffers_) {
            if (!write_device_state(buffer.exp_avg_sq, buffer.elements, "exp_avg_sq")) {
                return false;
            }
        }
        if (bf16_shadow_elements_ > 0 && bf16_shadow_bits_ != nullptr) {
            constexpr std::int64_t kChunkValues = 1 << 20;
            std::vector<std::uint16_t> shadow_chunk(static_cast<std::size_t>(kChunkValues), 0);
            for (std::int64_t offset = 0; offset < bf16_shadow_elements_; offset += kChunkValues) {
                const std::int64_t count = std::min<std::int64_t>(kChunkValues, bf16_shadow_elements_ - offset);
                const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(std::uint16_t);
                const int status = runtime_.cuda_memcpy(
                    shadow_chunk.data(), bf16_shadow_bits_ + offset, bytes, kCudaMemcpyDeviceToHost);
                if (status != 0) {
                    *error = family_cuda_error(runtime_, status, "cudaMemcpy family optimizer BF16 shadow D2H");
                    return false;
                }
                out.write(reinterpret_cast<const char*>(shadow_chunk.data()), static_cast<std::streamsize>(bytes));
                if (!out) {
                    *error = "failed to write family optimizer BF16 shadow";
                    return false;
                }
            }
        }
        out.close();
        if (!out) {
            *error = "failed to flush family optimizer checkpoint";
            return false;
        }
        return true;
    }

    bool load_checkpoint(
        const std::filesystem::path& path,
        std::int64_t* completed_optimizer_steps,
        std::string* error) {
        if (completed_optimizer_steps == nullptr || error == nullptr) {
            return false;
        }
        *completed_optimizer_steps = 0;
        if (!allocated_ || runtime_.cuda_memcpy == nullptr) {
            *error = "family optimizer is not allocated";
            return false;
        }
        constexpr std::uint64_t kMagic = 0x314D4954504F4E46ull;
        constexpr std::int64_t kCurrentVersion = 2;
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            *error = "failed to open family optimizer checkpoint";
            return false;
        }
        std::uint64_t magic = 0;
        std::int64_t version = 0;
        std::int64_t step = 0;
        std::int64_t buffer_count = 0;
        std::int64_t total_elements = 0;
        std::int64_t checkpoint_shadow_elements = 0;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        in.read(reinterpret_cast<char*>(&step), sizeof(step));
        in.read(reinterpret_cast<char*>(&buffer_count), sizeof(buffer_count));
        in.read(reinterpret_cast<char*>(&total_elements), sizeof(total_elements));
        if (version >= 2) {
            in.read(reinterpret_cast<char*>(&checkpoint_shadow_elements), sizeof(checkpoint_shadow_elements));
        }
        std::int64_t expected_elements = 0;
        for (const FamilyOptimizerBuffer& buffer : buffers_) {
            expected_elements += buffer.elements;
        }
        if (!in || magic != kMagic || version < 1 || version > kCurrentVersion || step < 0 ||
            buffer_count != static_cast<std::int64_t>(buffers_.size()) ||
            total_elements != expected_elements ||
            (version >= 2 && checkpoint_shadow_elements != bf16_shadow_elements_)) {
            *error = "family optimizer checkpoint header does not match optimizer layout";
            return false;
        }
        constexpr std::int64_t kChunkFloats = 1 << 20;
        std::vector<float> chunk(static_cast<std::size_t>(kChunkFloats), 0.0f);
        auto read_device_state = [&](float* device, std::int64_t elements, std::string_view name) {
            for (std::int64_t offset = 0; offset < elements; offset += kChunkFloats) {
                const std::int64_t count = std::min<std::int64_t>(kChunkFloats, elements - offset);
                const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);
                in.read(reinterpret_cast<char*>(chunk.data()), static_cast<std::streamsize>(bytes));
                if (!in) {
                    *error = "failed to read family optimizer " + std::string(name);
                    return false;
                }
                const int status = runtime_.cuda_memcpy(
                    device + offset, chunk.data(), bytes, kCudaMemcpyHostToDevice);
                if (status != 0) {
                    *error = family_cuda_error(
                        runtime_, status, "cudaMemcpy family optimizer " + std::string(name) + " H2D");
                    return false;
                }
            }
            return true;
        };
        for (FamilyOptimizerBuffer& buffer : buffers_) {
            if (!read_device_state(buffer.exp_avg, buffer.elements, "exp_avg")) {
                return false;
            }
        }
        for (FamilyOptimizerBuffer& buffer : buffers_) {
            if (!read_device_state(buffer.exp_avg_sq, buffer.elements, "exp_avg_sq")) {
                return false;
            }
        }
        if (version >= 2 && checkpoint_shadow_elements > 0 && bf16_shadow_bits_ != nullptr) {
            constexpr std::int64_t kChunkValues = 1 << 20;
            std::vector<std::uint16_t> shadow_chunk(static_cast<std::size_t>(kChunkValues), 0);
            for (std::int64_t offset = 0; offset < checkpoint_shadow_elements; offset += kChunkValues) {
                const std::int64_t count = std::min<std::int64_t>(kChunkValues, checkpoint_shadow_elements - offset);
                const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(std::uint16_t);
                in.read(reinterpret_cast<char*>(shadow_chunk.data()), static_cast<std::streamsize>(bytes));
                if (!in) {
                    *error = "failed to read family optimizer BF16 shadow";
                    return false;
                }
                const int status = runtime_.cuda_memcpy(
                    bf16_shadow_bits_ + offset, shadow_chunk.data(), bytes, kCudaMemcpyHostToDevice);
                if (status != 0) {
                    *error = family_cuda_error(runtime_, status, "cudaMemcpy family optimizer BF16 shadow H2D");
                    return false;
                }
            }
        }
        char trailing = 0;
        if (in.read(&trailing, 1)) {
            *error = "family optimizer checkpoint has trailing data";
            return false;
        }
        *completed_optimizer_steps = step;
        return synchronize(error);
    }

    void release() {
        destroy_optimizer_graph();
        destroy_production_step_graph();
        if (runtime_.cuda_stream_destroy != nullptr && production_step_stream_ != nullptr) {
            runtime_.cuda_stream_destroy(production_step_stream_);
            production_step_stream_ = nullptr;
        }
        if (runtime_.cuda_stream_destroy != nullptr && optimizer_graph_stream_ != nullptr) {
            runtime_.cuda_stream_destroy(optimizer_graph_stream_);
            optimizer_graph_stream_ = nullptr;
        }
        if (runtime_.cuda_free != nullptr) {
            for (FamilyOptimizerBuffer& buffer : buffers_) {
                free_device(buffer.grad);
                free_device(buffer.exp_avg);
                free_device(buffer.exp_avg_sq);
            }
            free_device(clip_scale_);
            free_device(optimizer_hyperparameters_);
            free_device(sumsq_partials_);
            free_device(d_param_ptrs_);
            free_device(d_grad_ptrs_);
            free_device(d_exp_avg_ptrs_);
            free_device(d_exp_avg_sq_ptrs_);
            free_device(d_elements_);
            free_device(d_partial_offsets_);
            free_device(d_bf16_shadow_offsets_);
            free_device(d_weight_decays_);
            free_device(bf16_shadow_bits_);
            free_device(loss_reporting_device_scalars_);
        }
        if (tile_api_.handle != nullptr) {
            dlclose(tile_api_.handle);
        }
        if (runtime_.handle != nullptr) {
            dlclose(runtime_.handle);
        }
        runtime_ = FamilyCudaRuntimeApi{};
        tile_api_ = FamilyTileOptimizerApi{};
        bf16_shadow_elements_ = 0;
        loss_reporting_scalar_count_ = 0;
        cuda_graph_capture_enabled_ = false;
        cuda_graph_capture_count_ = 0;
        cuda_graph_replay_count_ = 0;
        production_step_graph_capture_count_ = 0;
        production_step_graph_replay_count_ = 0;
        allocated_ = false;
    }

private:
    static constexpr std::int64_t kFloat32Bytes = 4;
    static constexpr std::int64_t kOptimizerHyperparameterCount = 6;
    static constexpr int kCudaMemcpyHostToDevice = 1;
    static constexpr int kCudaMemcpyDeviceToHost = 2;
    static constexpr unsigned int kCudaStreamNonBlocking = 1;
    static constexpr int kCudaStreamCaptureModeGlobal = 0;

    static bool optimizer_hyperparameters_equal(
        const FamilyOptimizerHyperparameters& left,
        const FamilyOptimizerHyperparameters& right) {
        return left.learning_rate == right.learning_rate &&
               left.beta1 == right.beta1 &&
               left.beta2 == right.beta2 &&
               left.eps == right.eps &&
               left.weight_decay == right.weight_decay &&
               left.grad_clip_norm == right.grad_clip_norm &&
               left.grad_clip_eps == right.grad_clip_eps &&
               left.bias_correction1 == right.bias_correction1 &&
               left.sqrt_bias_correction2 == right.sqrt_bias_correction2;
    }

    static bool optimizer_graph_scalar_launch_hyperparameters_equal(
        const FamilyOptimizerHyperparameters& left,
        const FamilyOptimizerHyperparameters& right) {
        return left.weight_decay == right.weight_decay &&
               left.grad_clip_norm == right.grad_clip_norm &&
               left.grad_clip_eps == right.grad_clip_eps;
    }

    bool device_hyperparameter_adamw_available() const {
        return optimizer_hyperparameters_ != nullptr &&
               ((bf16_shadow_enabled() &&
                 tile_api_.adamw_step_many_with_device_scale_bf16_shadow_hyper_float32 != nullptr) ||
                (!bf16_shadow_enabled() &&
                 tile_api_.adamw_step_many_with_device_scale_hyper_float32 != nullptr));
    }

    bool optimizer_graph_hyperparameters_require_recapture(
        const FamilyOptimizerHyperparameters& captured,
        const FamilyOptimizerHyperparameters& current) const {
        if (!optimizer_graph_uses_device_hyperparameters_) {
            return !optimizer_hyperparameters_equal(captured, current);
        }
        return !optimizer_graph_scalar_launch_hyperparameters_equal(captured, current);
    }

    bool optimizer_step_cuda_graph(const FamilyOptimizerHyperparameters& hyper, std::string* error) {
        if (!ready(error)) {
            return false;
        }
        if (!supports_cuda_graph_capture()) {
            if (error != nullptr) {
                *error = "family optimizer CUDA graph runtime symbols are missing";
            }
            return false;
        }
        if (optimizer_graph_exec_ == nullptr ||
            !optimizer_graph_hyper_valid_ ||
            optimizer_graph_hyperparameters_require_recapture(optimizer_graph_hyper_, hyper)) {
            if (!capture_optimizer_step_graph(hyper, error)) {
                return false;
            }
        } else if (optimizer_graph_uses_device_hyperparameters_ &&
                   !copy_optimizer_hyperparameters_to_device(hyper, optimizer_graph_stream_, error)) {
            return false;
        }
        const int launch_status = runtime_.cuda_graph_launch(optimizer_graph_exec_, optimizer_graph_stream_);
        if (launch_status != 0) {
            if (error != nullptr) {
                *error = family_cuda_error(runtime_, launch_status, "cudaGraphLaunch family optimizer step");
            }
            return false;
        }
        const int sync_status = runtime_.cuda_stream_synchronize(optimizer_graph_stream_);
        if (sync_status != 0) {
            if (error != nullptr) {
                *error = family_cuda_error(runtime_, sync_status, "cudaStreamSynchronize family optimizer graph");
            }
            return false;
        }
        cuda_graph_replay_count_ += 1;
        return true;
    }

    bool copy_optimizer_hyperparameters_to_device(
        const FamilyOptimizerHyperparameters& hyper,
        void* cuda_stream,
        std::string* error) {
        if (optimizer_hyperparameters_ == nullptr) {
            if (error != nullptr) {
                *error = "family optimizer device hyperparameter buffer is not allocated";
            }
            return false;
        }
        set_optimizer_hyperparameters_host(hyper);
        int status = 0;
        if (cuda_stream != nullptr && runtime_.cuda_memcpy_async != nullptr) {
            status = runtime_.cuda_memcpy_async(
                optimizer_hyperparameters_,
                optimizer_hyperparameters_host_,
                sizeof(optimizer_hyperparameters_host_),
                kCudaMemcpyHostToDevice,
                cuda_stream);
        } else {
            status = runtime_.cuda_memcpy(
                optimizer_hyperparameters_,
                optimizer_hyperparameters_host_,
                sizeof(optimizer_hyperparameters_host_),
                kCudaMemcpyHostToDevice);
        }
        if (status != 0) {
            if (error != nullptr) {
                *error = family_cuda_error(runtime_, status, "cudaMemcpy family optimizer hyperparameters H2D");
            }
            return false;
        }
        return true;
    }

    void set_optimizer_hyperparameters_host(const FamilyOptimizerHyperparameters& hyper) {
        optimizer_hyperparameters_host_[0] = hyper.learning_rate;
        optimizer_hyperparameters_host_[1] = hyper.beta1;
        optimizer_hyperparameters_host_[2] = hyper.beta2;
        optimizer_hyperparameters_host_[3] = hyper.eps;
        optimizer_hyperparameters_host_[4] = hyper.bias_correction1;
        optimizer_hyperparameters_host_[5] = hyper.sqrt_bias_correction2;
    }

    bool capture_optimizer_step_graph(const FamilyOptimizerHyperparameters& hyper, std::string* error) {
        destroy_optimizer_graph();
        if (optimizer_graph_stream_ == nullptr) {
            const int stream_status =
                runtime_.cuda_stream_create_with_flags(&optimizer_graph_stream_, kCudaStreamNonBlocking);
            if (stream_status != 0) {
                if (error != nullptr) {
                    *error = family_cuda_error(runtime_, stream_status, "cudaStreamCreateWithFlags family optimizer graph");
                }
                return false;
            }
        }
        void* graph = nullptr;
        int status = runtime_.cuda_stream_begin_capture(optimizer_graph_stream_, kCudaStreamCaptureModeGlobal);
        if (status != 0) {
            if (error != nullptr) {
                *error = family_cuda_error(runtime_, status, "cudaStreamBeginCapture family optimizer step");
            }
            return false;
        }
        if (!compute_global_clip_scale(hyper.grad_clip_norm, hyper.grad_clip_eps, optimizer_graph_stream_, error) ||
            !adamw_step(hyper, optimizer_graph_stream_, error) ||
            !zero_gradients_device(optimizer_graph_stream_, error)) {
            void* abandoned_graph = nullptr;
            runtime_.cuda_stream_end_capture(optimizer_graph_stream_, &abandoned_graph);
            if (abandoned_graph != nullptr) {
                runtime_.cuda_graph_destroy(abandoned_graph);
            }
            return false;
        }
        status = runtime_.cuda_stream_end_capture(optimizer_graph_stream_, &graph);
        if (status != 0 || graph == nullptr) {
            if (error != nullptr) {
                *error = family_cuda_error(runtime_, status, "cudaStreamEndCapture family optimizer step");
            }
            return false;
        }
        void* exec = nullptr;
        status = runtime_.cuda_graph_instantiate(&exec, graph, nullptr, nullptr, 0);
        const int destroy_status = runtime_.cuda_graph_destroy(graph);
        if (status != 0 || destroy_status != 0 || exec == nullptr) {
            if (exec != nullptr) {
                runtime_.cuda_graph_exec_destroy(exec);
            }
            if (error != nullptr) {
                *error = status != 0
                    ? family_cuda_error(runtime_, status, "cudaGraphInstantiate family optimizer step")
                    : family_cuda_error(runtime_, destroy_status, "cudaGraphDestroy family optimizer step");
            }
            return false;
        }
        status = runtime_.cuda_graph_upload(exec, optimizer_graph_stream_);
        if (status != 0) {
            runtime_.cuda_graph_exec_destroy(exec);
            if (error != nullptr) {
                *error = family_cuda_error(runtime_, status, "cudaGraphUpload family optimizer step");
            }
            return false;
        }
        optimizer_graph_exec_ = exec;
        optimizer_graph_hyper_ = hyper;
        optimizer_graph_hyper_valid_ = true;
        optimizer_graph_uses_device_hyperparameters_ = device_hyperparameter_adamw_available();
        cuda_graph_capture_count_ += 1;
        return true;
    }

    void destroy_optimizer_graph() {
        if (runtime_.cuda_graph_exec_destroy != nullptr && optimizer_graph_exec_ != nullptr) {
            runtime_.cuda_graph_exec_destroy(optimizer_graph_exec_);
        }
        optimizer_graph_exec_ = nullptr;
        optimizer_graph_hyper_valid_ = false;
        optimizer_graph_uses_device_hyperparameters_ = false;
    }

    void destroy_production_step_graph() {
        if (runtime_.cuda_graph_exec_destroy != nullptr && production_step_graph_exec_ != nullptr) {
            runtime_.cuda_graph_exec_destroy(production_step_graph_exec_);
        }
        production_step_graph_exec_ = nullptr;
        production_step_graph_report_loss_to_host_ = false;
        production_step_graph_uses_device_hyperparameters_ = false;
    }

    bool zero_gradients_device(void* cuda_stream, std::string* error) {
        if (!ready(error)) {
            return false;
        }
        if (tile_api_.fill_many_float32 == nullptr || d_grad_ptrs_ == nullptr || d_elements_ == nullptr) {
            if (error != nullptr) {
                *error = "Tile ops library is missing family optimizer gradient fill-many symbol";
            }
            return false;
        }
        constexpr std::int64_t kFillManyMaxDescriptorsPerLaunch = 32;
        const std::int64_t buffer_count = static_cast<std::int64_t>(buffers_.size());
        for (std::int64_t offset = 0; offset < buffer_count; offset += kFillManyMaxDescriptorsPerLaunch) {
            const std::int64_t chunk_count =
                std::min<std::int64_t>(kFillManyMaxDescriptorsPerLaunch, buffer_count - offset);
            const int status = tile_api_.fill_many_float32(
                const_cast<float* const*>(d_grad_ptrs_) + offset,
                d_elements_ + offset,
                chunk_count,
                max_elements_,
                0.0f,
                cuda_stream);
            if (status != 0) {
                if (error != nullptr) {
                    *error = "nfn_native_tile_fill_many_float32 failed for family optimizer gradient chunk";
                }
                return false;
            }
        }
        return true;
    }

    bool zero_gradients_host(std::string* error) {
        for (FamilyOptimizerBuffer& buffer : buffers_) {
            if (!copy_zero_float_buffer(buffer.grad, buffer.elements, error)) {
                return false;
            }
        }
        return true;
    }

    static std::int64_t optimizer_tile_count(std::int64_t elements) {
        if (elements <= 0) {
            return 0;
        }
        return (elements + NFN_TILE_CUDA_OPTIMIZER_TILE_SIZE - 1) / NFN_TILE_CUDA_OPTIMIZER_TILE_SIZE;
    }

    static float default_weight_decay_for_buffer(const std::string& name) {
        if (name.find("norm.weight") != std::string::npos ||
            (name.size() >= 5 && name.rfind(".bias") == name.size() - 5)) {
            return 0.0f;
        }
        return 0.02f;
    }

    static std::size_t checked_float_bytes(std::int64_t elements, std::string* error) {
        if (error != nullptr) {
            error->clear();
        }
        if (elements < 0 || elements > std::numeric_limits<std::int64_t>::max() / kFloat32Bytes) {
            if (error != nullptr) {
                *error = "family optimizer float byte count overflowed";
            }
            return 0;
        }
        return static_cast<std::size_t>(elements * kFloat32Bytes);
    }

    template <typename T>
    bool copy_descriptor_to_device(T* device, const std::vector<T>& host, std::string_view name, std::string* error) {
        if (host.empty()) {
            return true;
        }
        const std::size_t bytes = host.size() * sizeof(T);
        const int status = runtime_.cuda_memcpy(device, host.data(), bytes, kCudaMemcpyHostToDevice);
        if (status != 0) {
            if (error != nullptr) {
                *error = family_cuda_error(runtime_, status, std::string("cudaMemcpy family optimizer ") + std::string(name));
            }
            return false;
        }
        return true;
    }

    bool allocate_float_buffer(float** ptr, std::int64_t elements, const std::string& name, std::string* error) {
        const std::size_t bytes = checked_float_bytes(elements, error);
        if (error != nullptr && !error->empty()) {
            return false;
        }
        const int status = runtime_.cuda_malloc(reinterpret_cast<void**>(ptr), bytes);
        if (status != 0) {
            if (error != nullptr) {
                *error = family_cuda_error(runtime_, status, "cudaMalloc family optimizer " + name);
            }
            return false;
        }
        return true;
    }

    template <typename T>
    bool allocate_device_array(T** ptr, std::size_t count, const std::string& name, std::string* error) {
        if (count == 0) {
            *ptr = nullptr;
            return true;
        }
        const int status = runtime_.cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(T));
        if (status != 0) {
            if (error != nullptr) {
                *error = family_cuda_error(runtime_, status, "cudaMalloc family optimizer descriptor " + name);
            }
            return false;
        }
        return true;
    }

    bool allocate_descriptor_arrays(std::string* error) {
        const std::size_t count = buffers_.size();
        std::vector<float*> params;
        std::vector<const float*> grads;
        std::vector<float*> exp_avgs;
        std::vector<float*> exp_avg_sqs;
        std::vector<std::int64_t> elements;
        std::vector<std::int64_t> partial_offsets;
        std::vector<std::int64_t> bf16_shadow_offsets;
        std::vector<float> weight_decays;
        params.reserve(count);
        grads.reserve(count);
        exp_avgs.reserve(count);
        exp_avg_sqs.reserve(count);
        elements.reserve(count);
        partial_offsets.reserve(count);
        bf16_shadow_offsets.reserve(count);
        weight_decays.reserve(count);
        for (const FamilyOptimizerBuffer& buffer : buffers_) {
            params.push_back(buffer.parameter);
            grads.push_back(buffer.grad);
            exp_avgs.push_back(buffer.exp_avg);
            exp_avg_sqs.push_back(buffer.exp_avg_sq);
            elements.push_back(buffer.elements);
            partial_offsets.push_back(buffer.partial_offset);
            bf16_shadow_offsets.push_back(buffer.bf16_shadow_offset);
            weight_decays.push_back(buffer.weight_decay);
        }
        return allocate_device_array(&d_param_ptrs_, count, "params", error) &&
               allocate_device_array(&d_grad_ptrs_, count, "grads", error) &&
               allocate_device_array(&d_exp_avg_ptrs_, count, "exp_avgs", error) &&
               allocate_device_array(&d_exp_avg_sq_ptrs_, count, "exp_avg_sqs", error) &&
               allocate_device_array(&d_elements_, count, "elements", error) &&
               allocate_device_array(&d_partial_offsets_, count, "partial_offsets", error) &&
               allocate_device_array(&d_bf16_shadow_offsets_, count, "bf16_shadow_offsets", error) &&
               allocate_device_array(&d_weight_decays_, count, "weight_decays", error) &&
               copy_descriptor_to_device(d_param_ptrs_, params, "params", error) &&
               copy_descriptor_to_device(d_grad_ptrs_, grads, "grads", error) &&
               copy_descriptor_to_device(d_exp_avg_ptrs_, exp_avgs, "exp_avgs", error) &&
               copy_descriptor_to_device(d_exp_avg_sq_ptrs_, exp_avg_sqs, "exp_avg_sqs", error) &&
               copy_descriptor_to_device(d_elements_, elements, "elements", error) &&
               copy_descriptor_to_device(d_partial_offsets_, partial_offsets, "partial_offsets", error) &&
               copy_descriptor_to_device(d_bf16_shadow_offsets_, bf16_shadow_offsets, "bf16_shadow_offsets", error) &&
               copy_descriptor_to_device(d_weight_decays_, weight_decays, "weight_decays", error);
    }

    bool copy_zero_float_buffer(float* ptr, std::int64_t elements, std::string* error) {
        std::vector<float> zeros(static_cast<std::size_t>(std::min<std::int64_t>(elements, 1 << 20)), 0.0f);
        for (std::int64_t offset = 0; offset < elements; offset += static_cast<std::int64_t>(zeros.size())) {
            const std::int64_t count = std::min<std::int64_t>(static_cast<std::int64_t>(zeros.size()), elements - offset);
            const std::size_t bytes = checked_float_bytes(count, error);
            if (error != nullptr && !error->empty()) {
                return false;
            }
            const int status =
                runtime_.cuda_memcpy(ptr + offset, zeros.data(), bytes, kCudaMemcpyHostToDevice);
            if (status != 0) {
                if (error != nullptr) {
                    *error = family_cuda_error(runtime_, status, "cudaMemcpy zero family optimizer buffer");
                }
                return false;
            }
        }
        return true;
    }

    bool zero_state(std::string* error) {
        for (FamilyOptimizerBuffer& buffer : buffers_) {
            if (!copy_zero_float_buffer(buffer.grad, buffer.elements, error) ||
                !copy_zero_float_buffer(buffer.exp_avg, buffer.elements, error) ||
                !copy_zero_float_buffer(buffer.exp_avg_sq, buffer.elements, error)) {
                return false;
            }
        }
        return copy_zero_float_buffer(sumsq_partials_, partial_count_, error);
    }

    bool set_clip_scale_host(float value, std::string* error) {
        const int status = runtime_.cuda_memcpy(clip_scale_, &value, sizeof(float), kCudaMemcpyHostToDevice);
        if (status != 0) {
            if (error != nullptr) {
                *error = family_cuda_error(runtime_, status, "cudaMemcpy family optimizer clip_scale");
            }
            return false;
        }
        return true;
    }

    bool ready(std::string* error) const {
        if (!allocated_) {
            if (error != nullptr) {
                *error = "family optimizer state is not allocated";
            }
            return false;
        }
        return true;
    }

    bool synchronize(std::string* error) const {
        const int status = runtime_.cuda_device_synchronize();
        if (status != 0) {
            if (error != nullptr) {
                *error = family_cuda_error(runtime_, status, "cudaDeviceSynchronize family optimizer state");
            }
            return false;
        }
        return true;
    }

    template <typename T>
    void free_device(T*& ptr) {
        if (ptr != nullptr) {
            runtime_.cuda_free(ptr);
            ptr = nullptr;
        }
    }

    void move_from(FamilyOptimizerState&& other) {
        buffers_ = std::move(other.buffers_);
        runtime_ = other.runtime_;
        tile_api_ = other.tile_api_;
        max_elements_ = other.max_elements_;
        partial_count_ = other.partial_count_;
        bf16_shadow_elements_ = other.bf16_shadow_elements_;
        clip_scale_ = other.clip_scale_;
        optimizer_hyperparameters_ = other.optimizer_hyperparameters_;
        std::copy(
            other.optimizer_hyperparameters_host_,
            other.optimizer_hyperparameters_host_ + kOptimizerHyperparameterCount,
            optimizer_hyperparameters_host_);
        sumsq_partials_ = other.sumsq_partials_;
        bf16_shadow_bits_ = other.bf16_shadow_bits_;
        loss_reporting_device_scalars_ = other.loss_reporting_device_scalars_;
        loss_reporting_scalar_count_ = other.loss_reporting_scalar_count_;
        production_step_stream_ = other.production_step_stream_;
        optimizer_graph_stream_ = other.optimizer_graph_stream_;
        optimizer_graph_exec_ = other.optimizer_graph_exec_;
        production_step_graph_exec_ = other.production_step_graph_exec_;
        production_step_graph_report_loss_to_host_ = other.production_step_graph_report_loss_to_host_;
        production_step_graph_uses_device_hyperparameters_ = other.production_step_graph_uses_device_hyperparameters_;
        optimizer_graph_hyper_ = other.optimizer_graph_hyper_;
        optimizer_graph_hyper_valid_ = other.optimizer_graph_hyper_valid_;
        optimizer_graph_uses_device_hyperparameters_ = other.optimizer_graph_uses_device_hyperparameters_;
        cuda_graph_capture_enabled_ = other.cuda_graph_capture_enabled_;
        cuda_graph_capture_count_ = other.cuda_graph_capture_count_;
        cuda_graph_replay_count_ = other.cuda_graph_replay_count_;
        production_step_graph_capture_count_ = other.production_step_graph_capture_count_;
        production_step_graph_replay_count_ = other.production_step_graph_replay_count_;
        d_param_ptrs_ = other.d_param_ptrs_;
        d_grad_ptrs_ = other.d_grad_ptrs_;
        d_exp_avg_ptrs_ = other.d_exp_avg_ptrs_;
        d_exp_avg_sq_ptrs_ = other.d_exp_avg_sq_ptrs_;
        d_elements_ = other.d_elements_;
        d_partial_offsets_ = other.d_partial_offsets_;
        d_bf16_shadow_offsets_ = other.d_bf16_shadow_offsets_;
        d_weight_decays_ = other.d_weight_decays_;
        allocated_ = other.allocated_;
        other.runtime_ = FamilyCudaRuntimeApi{};
        other.tile_api_ = FamilyTileOptimizerApi{};
        other.max_elements_ = 0;
        other.partial_count_ = 0;
        other.bf16_shadow_elements_ = 0;
        other.loss_reporting_scalar_count_ = 0;
        other.clip_scale_ = nullptr;
        other.optimizer_hyperparameters_ = nullptr;
        other.sumsq_partials_ = nullptr;
        other.bf16_shadow_bits_ = nullptr;
        other.loss_reporting_device_scalars_ = nullptr;
        other.production_step_stream_ = nullptr;
        other.optimizer_graph_stream_ = nullptr;
        other.optimizer_graph_exec_ = nullptr;
        other.production_step_graph_exec_ = nullptr;
        other.production_step_graph_report_loss_to_host_ = false;
        other.production_step_graph_uses_device_hyperparameters_ = false;
        other.optimizer_graph_hyper_valid_ = false;
        other.optimizer_graph_uses_device_hyperparameters_ = false;
        other.cuda_graph_capture_enabled_ = false;
        other.cuda_graph_capture_count_ = 0;
        other.cuda_graph_replay_count_ = 0;
        other.production_step_graph_capture_count_ = 0;
        other.production_step_graph_replay_count_ = 0;
        other.d_param_ptrs_ = nullptr;
        other.d_grad_ptrs_ = nullptr;
        other.d_exp_avg_ptrs_ = nullptr;
        other.d_exp_avg_sq_ptrs_ = nullptr;
        other.d_elements_ = nullptr;
        other.d_partial_offsets_ = nullptr;
        other.d_bf16_shadow_offsets_ = nullptr;
        other.d_weight_decays_ = nullptr;
        other.allocated_ = false;
    }

    std::vector<FamilyOptimizerBuffer> buffers_;
    FamilyCudaRuntimeApi runtime_;
    FamilyTileOptimizerApi tile_api_;
    std::int64_t max_elements_ = 0;
    std::int64_t partial_count_ = 0;
    std::int64_t bf16_shadow_elements_ = 0;
    float* clip_scale_ = nullptr;
    float* optimizer_hyperparameters_ = nullptr;
    float optimizer_hyperparameters_host_[kOptimizerHyperparameterCount] = {};
    float* sumsq_partials_ = nullptr;
    std::uint16_t* bf16_shadow_bits_ = nullptr;
    float* loss_reporting_device_scalars_ = nullptr;
    std::int64_t loss_reporting_scalar_count_ = 0;
    void* production_step_stream_ = nullptr;
    void* optimizer_graph_stream_ = nullptr;
    void* optimizer_graph_exec_ = nullptr;
    void* production_step_graph_exec_ = nullptr;
    bool production_step_graph_report_loss_to_host_ = false;
    bool production_step_graph_uses_device_hyperparameters_ = false;
    FamilyOptimizerHyperparameters optimizer_graph_hyper_;
    bool optimizer_graph_hyper_valid_ = false;
    bool optimizer_graph_uses_device_hyperparameters_ = false;
    bool cuda_graph_capture_enabled_ = false;
    std::int64_t cuda_graph_capture_count_ = 0;
    std::int64_t cuda_graph_replay_count_ = 0;
    std::int64_t production_step_graph_capture_count_ = 0;
    std::int64_t production_step_graph_replay_count_ = 0;
    float** d_param_ptrs_ = nullptr;
    const float** d_grad_ptrs_ = nullptr;
    float** d_exp_avg_ptrs_ = nullptr;
    float** d_exp_avg_sq_ptrs_ = nullptr;
    std::int64_t* d_elements_ = nullptr;
    std::int64_t* d_partial_offsets_ = nullptr;
    std::int64_t* d_bf16_shadow_offsets_ = nullptr;
    float* d_weight_decays_ = nullptr;
    bool allocated_ = false;
};

}  // namespace neuralfn::native_train
