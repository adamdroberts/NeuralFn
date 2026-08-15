// Experimental benchmark provider for the pinned llama.cpp CUDA MMQ kernels.
//
// This file is intentionally outside NeuralFn's product build.  It links to a
// single pinned ggml CUDA shared library and exposes a tiny C ABI over existing
// device pointers.  The purpose is to quantify the attainable projection
// speed before porting an independently reviewed kernel into the native Tile
// ABI; shipping NeuralFn must not depend on this private upstream C++ ABI.

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include "ggml-backend-impl.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace {

constexpr std::uint32_t kProviderAbi = 1;
constexpr std::uint32_t kNeuralFnQ4K = 12;
constexpr std::uint32_t kNeuralFnQ5K = 13;
constexpr std::uint32_t kNeuralFnQ6K = 14;

struct Provider {
    explicit Provider(int requested_device) : device(requested_device) {
        backend = ggml_backend_cuda_init(device);
        if (backend == nullptr) {
            throw std::runtime_error("ggml_backend_cuda_init failed");
        }
        context = static_cast<ggml_backend_cuda_context*>(backend->context);
        if (context == nullptr) {
            ggml_backend_free(backend);
            backend = nullptr;
            throw std::runtime_error("ggml CUDA backend context is null");
        }
    }

    ~Provider() {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }

    int device = 0;
    ggml_backend_t backend = nullptr;
    ggml_backend_cuda_context* context = nullptr;
    std::mutex mutex;
};

ggml_type map_encoding(std::uint32_t encoding) {
    switch (encoding) {
        case kNeuralFnQ4K:
            return GGML_TYPE_Q4_K;
        case kNeuralFnQ5K:
            return GGML_TYPE_Q5_K;
        case kNeuralFnQ6K:
            return GGML_TYPE_Q6_K;
        default:
            return GGML_TYPE_COUNT;
    }
}

template <ggml_type Type>
void launch_mmq_from_q8(
    Provider& provider,
    const std::uint8_t* packed_weights,
    const char* q8,
    float* output,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t row_stride_bytes,
    cudaStream_t stream) {
    const std::int64_t weight_blocks_per_row = row_stride_bytes /
        static_cast<std::int64_t>(ggml_type_size(Type));
    const std::int64_t q8_stride_ints =
        rows * input_dim * static_cast<std::int64_t>(sizeof(block_q8_1)) /
        (QK8_1 * static_cast<std::int64_t>(sizeof(int)));
    const mmq_args arguments = {
        reinterpret_cast<const char*>(packed_weights),
        Type,
        reinterpret_cast<const int*>(q8),
        nullptr,
        nullptr,
        output,
        nullptr,
        input_dim,
        output_dim,
        rows,
        weight_blocks_per_row,
        rows,
        output_dim,
        1,
        1,
        output_dim * weight_blocks_per_row,
        q8_stride_ints,
        rows * output_dim,
        1,
        1,
        output_dim * weight_blocks_per_row,
        q8_stride_ints,
        rows * output_dim,
        rows,
    };
    mul_mat_q_case<Type>(*provider.context, arguments, stream);
}

template <ggml_type Type>
void launch_mmq_group(
    Provider& provider,
    const std::uint32_t* encodings,
    const std::uint8_t* const* packed_weights,
    const std::int64_t* row_stride_bytes,
    const float* input,
    float* const* outputs,
    const std::int64_t* output_dims,
    std::int64_t operation_count,
    std::int64_t rows,
    std::int64_t input_dim,
    cudaStream_t stream) {
    bool used = false;
    int j_max = 0;
    const int cc = ggml_cuda_info().devices[provider.device].cc;
    for (std::int64_t index = 0; index < operation_count; ++index) {
        if (map_encoding(encodings[index]) != Type) continue;
        used = true;
        const bool fallback = output_dims[index] % 128 != 0;
        j_max = std::max(
            j_max, ggml_cuda_mmq_get_J_max(Type, fallback, cc, rows));
    }
    if (!used) return;
    const std::size_t q8_nbytes =
        static_cast<std::size_t>(rows * input_dim) * sizeof(block_q8_1_mmq) /
            QK8_1_MMQ +
        static_cast<std::size_t>(j_max) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> q8(provider.context->pool(), q8_nbytes);
    quantize_mmq_q8_1_cuda(
        input, nullptr, q8.get(), Type,
        input_dim,
        input_dim,
        rows * input_dim,
        rows * input_dim,
        input_dim,
        rows,
        1,
        1,
        stream);
    for (std::int64_t index = 0; index < operation_count; ++index) {
        if (map_encoding(encodings[index]) != Type) continue;
        launch_mmq_from_q8<Type>(
            provider, packed_weights[index], q8.get(), outputs[index], rows,
            input_dim, output_dims[index], row_stride_bytes[index], stream);
    }
}

}  // namespace

extern "C" {

std::uint32_t nfn_experimental_llama_mmq_provider_abi(void) {
    return kProviderAbi;
}

void* nfn_experimental_llama_mmq_provider_create(int device) {
    try {
        return new Provider(device);
    } catch (...) {
        return nullptr;
    }
}

void nfn_experimental_llama_mmq_provider_destroy(void* opaque) {
    delete static_cast<Provider*>(opaque);
}

int nfn_experimental_llama_mmq_provider_linear_f32(
    void* opaque,
    std::uint32_t encoding,
    const std::uint8_t* packed_weights,
    std::int64_t packed_nbytes,
    std::int64_t row_stride_bytes,
    const float* input,
    float* output,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    Provider* provider = static_cast<Provider*>(opaque);
    const ggml_type type = map_encoding(encoding);
    if (provider == nullptr || packed_weights == nullptr || input == nullptr ||
        output == nullptr || type == GGML_TYPE_COUNT || rows < 2 || rows > 16 ||
        input_dim <= 0 || input_dim % 256 != 0 || output_dim <= 0 ||
        row_stride_bytes != input_dim / 256 *
            static_cast<std::int64_t>(ggml_type_size(type)) ||
        packed_nbytes < output_dim * row_stride_bytes) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    std::lock_guard<std::mutex> lock(provider->mutex);
    ggml_cuda_set_device(provider->device);
    const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
    const std::uint32_t encodings[] = { encoding };
    const std::uint8_t* weights[] = { packed_weights };
    const std::int64_t strides[] = { row_stride_bytes };
    float* outputs[] = { output };
    const std::int64_t output_dims[] = { output_dim };
    switch (type) {
        case GGML_TYPE_Q4_K:
            launch_mmq_group<GGML_TYPE_Q4_K>(
                *provider, encodings, weights, strides, input, outputs,
                output_dims, 1, rows, input_dim, stream);
            break;
        case GGML_TYPE_Q5_K:
            launch_mmq_group<GGML_TYPE_Q5_K>(
                *provider, encodings, weights, strides, input, outputs,
                output_dims, 1, rows, input_dim, stream);
            break;
        case GGML_TYPE_Q6_K:
            launch_mmq_group<GGML_TYPE_Q6_K>(
                *provider, encodings, weights, strides, input, outputs,
                output_dims, 1, rows, input_dim, stream);
            break;
        default:
            return static_cast<int>(cudaErrorInvalidValue);
    }
    return static_cast<int>(cudaPeekAtLastError());
}

int nfn_experimental_llama_mmq_provider_multi_linear_f32(
    void* opaque,
    const std::uint32_t* encodings,
    const std::uint8_t* const* packed_weights,
    const std::int64_t* packed_nbytes,
    const std::int64_t* row_stride_bytes,
    const float* input,
    float* const* outputs,
    const std::int64_t* output_dims,
    std::int64_t operation_count,
    std::int64_t rows,
    std::int64_t input_dim,
    void* cuda_stream) {
    Provider* provider = static_cast<Provider*>(opaque);
    if (provider == nullptr || encodings == nullptr || packed_weights == nullptr ||
        packed_nbytes == nullptr || row_stride_bytes == nullptr || input == nullptr ||
        outputs == nullptr || output_dims == nullptr || operation_count <= 0 ||
        operation_count > 4 || rows < 2 || rows > 16 || input_dim <= 0 ||
        input_dim % 256 != 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    for (std::int64_t index = 0; index < operation_count; ++index) {
        const ggml_type type = map_encoding(encodings[index]);
        if (type == GGML_TYPE_COUNT || packed_weights[index] == nullptr ||
            outputs[index] == nullptr || output_dims[index] <= 0 ||
            row_stride_bytes[index] != input_dim / 256 *
                static_cast<std::int64_t>(ggml_type_size(type)) ||
            packed_nbytes[index] < output_dims[index] * row_stride_bytes[index]) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
    }
    std::lock_guard<std::mutex> lock(provider->mutex);
    ggml_cuda_set_device(provider->device);
    const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
    launch_mmq_group<GGML_TYPE_Q4_K>(
        *provider, encodings, packed_weights, row_stride_bytes, input, outputs,
        output_dims, operation_count, rows, input_dim, stream);
    launch_mmq_group<GGML_TYPE_Q5_K>(
        *provider, encodings, packed_weights, row_stride_bytes, input, outputs,
        output_dims, operation_count, rows, input_dim, stream);
    launch_mmq_group<GGML_TYPE_Q6_K>(
        *provider, encodings, packed_weights, row_stride_bytes, input, outputs,
        output_dims, operation_count, rows, input_dim, stream);
    return static_cast<int>(cudaPeekAtLastError());
}

}  // extern "C"
