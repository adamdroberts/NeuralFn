#include "tile_ops.h"

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
                   "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data_; }

    void upload(const std::vector<T>& values) {
        if (values.size() != count_) {
            throw std::runtime_error("device upload size mismatch");
        }
        cuda_check(cudaMemcpy(data_, values.data(), count_ * sizeof(T),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy host-to-device");
    }

    std::vector<T> download() const {
        std::vector<T> values(count_);
        cuda_check(cudaMemcpy(values.data(), data_, count_ * sizeof(T),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy device-to-host");
        return values;
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

template <typename Function>
Function required_symbol(void* library, const char* name) {
    dlerror();
    auto* symbol = dlsym(library, name);
    if (const char* error = dlerror(); error != nullptr || symbol == nullptr) {
        throw std::runtime_error(
            std::string("missing Tile symbol ") + name + ": " +
            (error == nullptr ? "unknown error" : error));
    }
    return reinterpret_cast<Function>(symbol);
}

void require_close(float actual, float expected, float tolerance,
                   const std::string& label) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            label + " mismatch: actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
    }
}

void check_sigmoid_gate(void* library) {
    using Fn = int (*)(const float*, const float*, float*, std::int64_t, void*);
    const auto operation = required_symbol<Fn>(
        library, "nfn_native_tile_glimmer_sigmoid_gate_float32_v1");
    const std::vector<float> values{1.0f, 2.0f, -3.0f, 4.0f};
    const std::vector<float> gates{0.0f, 1.0f, -1.0f, 2.0f};
    DeviceBuffer<float> device_values(values.size());
    DeviceBuffer<float> device_gates(gates.size());
    DeviceBuffer<float> device_output(values.size());
    device_values.upload(values);
    device_gates.upload(gates);
    cuda_check(static_cast<cudaError_t>(operation(
                   device_values.get(), device_gates.get(), device_output.get(),
                   static_cast<std::int64_t>(values.size()), nullptr)),
               "glimmer sigmoid gate launch");
    cuda_check(cudaDeviceSynchronize(), "glimmer sigmoid gate synchronize");
    const auto output = device_output.download();
    for (std::size_t index = 0; index < values.size(); ++index) {
        const float expected = values[index] /
            (1.0f + std::exp(-gates[index]));
        require_close(output[index], expected, 2.0e-6f, "sigmoid gate");
    }
}

void check_wide_rms_norm(void* library) {
    using Fn = int (*)(const float*, const NfnNativeTilePackedWeightDescriptorV1*,
                       float*, std::int64_t, std::int64_t, float, bool, void*);
    const auto operation = required_symbol<Fn>(
        library, "nfn_native_tile_glimmer_rms_norm_affine_float32_v1");
    constexpr std::int64_t rows = 2;
    constexpr std::int64_t width = 6656;
    constexpr float epsilon = 1.0e-8f;
    std::vector<float> input(static_cast<std::size_t>(rows * width));
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(static_cast<int>(index % 37) - 18) /
            11.0f;
    }
    DeviceBuffer<float> device_input(input.size());
    DeviceBuffer<float> device_output(input.size());
    device_input.upload(input);
    cuda_check(static_cast<cudaError_t>(operation(
                   device_input.get(), nullptr, device_output.get(), rows, width,
                   epsilon, false, nullptr)),
               "glimmer wide RMSNorm launch");
    cuda_check(cudaDeviceSynchronize(), "glimmer wide RMSNorm synchronize");
    const auto output = device_output.download();
    for (std::int64_t row = 0; row < rows; ++row) {
        double square_sum = 0.0;
        for (std::int64_t column = 0; column < width; ++column) {
            const double value = input[static_cast<std::size_t>(row * width + column)];
            square_sum += value * value;
        }
        const float inverse_rms = static_cast<float>(
            1.0 / std::sqrt(square_sum / static_cast<double>(width) + epsilon));
        for (std::int64_t column = 0; column < width; ++column) {
            const std::size_t index = static_cast<std::size_t>(row * width + column);
            require_close(output[index], input[index] * inverse_rms, 2.0e-5f,
                          "wide RMSNorm");
        }
    }
}

void rotate_half_split(std::vector<float>* values, std::int64_t heads,
                       std::int64_t head_dim, std::int64_t position,
                       float theta) {
    const std::int64_t half = head_dim / 2;
    for (std::int64_t head = 0; head < heads; ++head) {
        const std::int64_t base = head * head_dim;
        for (std::int64_t lane = 0; lane < half; ++lane) {
            const double frequency = std::pow(
                static_cast<double>(theta),
                -2.0 * static_cast<double>(lane) / static_cast<double>(head_dim));
            const double angle = static_cast<double>(position) * frequency;
            const float cosine = static_cast<float>(std::cos(angle));
            const float sine = static_cast<float>(std::sin(angle));
            const float first = (*values)[static_cast<std::size_t>(base + lane)];
            const float second = (*values)[static_cast<std::size_t>(base + lane + half)];
            (*values)[static_cast<std::size_t>(base + lane)] =
                first * cosine - second * sine;
            (*values)[static_cast<std::size_t>(base + lane + half)] =
                second * cosine + first * sine;
        }
    }
}

void check_positioned_rope(void* library) {
    using Fn = int (*)(float*, float*, std::int64_t, std::int64_t, std::int64_t,
                       std::int64_t, float, std::uint32_t, void*);
    const auto operation = required_symbol<Fn>(
        library, "nfn_native_tile_glimmer_positioned_rope_float32_v1");
    constexpr std::int64_t query_heads = 32;
    constexpr std::int64_t kv_heads = 2;
    constexpr std::int64_t head_dim = 128;
    constexpr std::int64_t position = 2051;
    constexpr float theta = 500000.0f;
    std::vector<float> query(static_cast<std::size_t>(query_heads * head_dim));
    std::vector<float> key(static_cast<std::size_t>(kv_heads * head_dim));
    for (std::size_t index = 0; index < query.size(); ++index) {
        query[index] = static_cast<float>(static_cast<int>(index % 29) - 14) / 17.0f;
    }
    for (std::size_t index = 0; index < key.size(); ++index) {
        key[index] = static_cast<float>(static_cast<int>(index % 19) - 9) / 13.0f;
    }
    auto expected_query = query;
    auto expected_key = key;
    rotate_half_split(&expected_query, query_heads, head_dim, position, theta);
    rotate_half_split(&expected_key, kv_heads, head_dim, position, theta);
    DeviceBuffer<float> device_query(query.size());
    DeviceBuffer<float> device_key(key.size());
    device_query.upload(query);
    device_key.upload(key);
    cuda_check(static_cast<cudaError_t>(operation(
                   device_query.get(), device_key.get(), query_heads, kv_heads,
                   head_dim, position, theta,
                   NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT, nullptr)),
               "glimmer positioned RoPE launch");
    cuda_check(cudaDeviceSynchronize(), "glimmer positioned RoPE synchronize");
    const auto actual_query = device_query.download();
    const auto actual_key = device_key.download();
    for (std::size_t index = 0; index < actual_query.size(); ++index) {
        require_close(actual_query[index], expected_query[index], 2.0e-4f,
                      "positioned query RoPE");
    }
    for (std::size_t index = 0; index < actual_key.size(); ++index) {
        require_close(actual_key[index], expected_key[index], 2.0e-4f,
                      "positioned key RoPE");
    }
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<std::uint16_t>(bits >> 16u);
}

float bf16_to_float(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16u);
}

float patterned_value(std::int64_t first, std::int64_t second,
                      std::int64_t modulus, float divisor) {
    return static_cast<float>((first * 17 + second * 13) % modulus - modulus / 2) /
        divisor;
}

std::pair<int, int> q4_scale_min(const std::vector<std::uint8_t>& block,
                                 int index) {
    const std::uint8_t* scales = block.data() + 4;
    if (index < 4) {
        return {scales[index] & 63, scales[index + 4] & 63};
    }
    return {
        (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4),
        (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
    };
}

float q4_value(const std::vector<std::uint8_t>& block, int column) {
    const int group = column / 64;
    const int lane = column % 32;
    const bool high = (column % 64) >= 32;
    const auto [scale, minimum] = q4_scale_min(block, group * 2 + (high ? 1 : 0));
    const std::uint8_t packed = block[static_cast<std::size_t>(16 + group * 32 + lane)];
    const int quant = high ? packed >> 4 : packed & 0x0f;
    return static_cast<float>(scale * quant) - 0.5f * static_cast<float>(minimum);
}

void check_q4_k_packed_linear(void* library) {
    using ValidateFn = int (*)(const NfnNativeTilePackedWeightDescriptorV1*);
    using DequantFn = int (*)(const NfnNativeTilePackedWeightDescriptorV1*, float*);
    using LinearFn = int (*)(const NfnNativeTilePackedWeightDescriptorV1*,
                             const float*, const float*, float*, std::int64_t, bool);
    using BackwardFn = int (*)(const NfnNativeTilePackedWeightDescriptorV1*,
                               const float*, float*, std::int64_t);
    const auto validate = required_symbol<ValidateFn>(
        library, "nfn_native_tile_packed_weight_validate_v1");
    const auto dequantize = required_symbol<DequantFn>(
        library, "nfn_native_tile_packed_weight_dequantize_float32_v1");
    const auto linear = required_symbol<LinearFn>(
        library, "nfn_native_tile_linear_packed_weight_float32_v1");
    const auto backward = required_symbol<BackwardFn>(
        library, "nfn_native_tile_linear_backward_input_packed_weight_float32_v1");
    std::vector<std::uint8_t> block(144, 0);
    block[0] = 0x00;
    block[1] = 0x3c;  // fp16 1.0
    block[2] = 0x00;
    block[3] = 0x38;  // fp16 0.5
    for (int index = 0; index < 8; ++index) {
        block[static_cast<std::size_t>(4 + index)] =
            static_cast<std::uint8_t>(index + 1);
    }
    std::fill(block.begin() + 16, block.end(), static_cast<std::uint8_t>(0x21));
    std::vector<float> expected(256);
    std::vector<float> input(256);
    for (int column = 0; column < 256; ++column) {
        expected[static_cast<std::size_t>(column)] = q4_value(block, column);
        input[static_cast<std::size_t>(column)] =
            static_cast<float>(column % 17 - 8) / 19.0f;
    }
    DeviceBuffer<std::uint8_t> device_block(block.size());
    DeviceBuffer<float> device_dequantized(expected.size());
    DeviceBuffer<float> device_input(input.size());
    DeviceBuffer<float> device_output(1);
    DeviceBuffer<float> device_grad_output(1);
    DeviceBuffer<float> device_grad_input(input.size());
    device_block.upload(block);
    device_input.upload(input);
    device_grad_output.upload(std::vector<float>{1.25f});
    NfnNativeTilePackedWeightDescriptorV1 descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.version = NFN_NATIVE_TILE_PACKED_WEIGHT_V1;
    descriptor.encoding = NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K;
    descriptor.data = device_block.get();
    descriptor.data_nbytes = static_cast<std::int64_t>(block.size());
    descriptor.output_dim = 1;
    descriptor.input_dim = 256;
    descriptor.row_stride_bytes = 144;
    cuda_check(static_cast<cudaError_t>(validate(&descriptor)),
               "Q4_K descriptor validation");
    cuda_check(static_cast<cudaError_t>(dequantize(&descriptor,
                                                   device_dequantized.get())),
               "Q4_K dequantize launch");
    cuda_check(static_cast<cudaError_t>(linear(
                   &descriptor, device_input.get(), nullptr, device_output.get(), 1,
                   false)),
               "Q4_K linear launch");
    cuda_check(static_cast<cudaError_t>(backward(
                   &descriptor, device_grad_output.get(), device_grad_input.get(), 1)),
               "Q4_K backward-input launch");
    cuda_check(cudaDeviceSynchronize(), "Q4_K kernels synchronize");
    const auto dequantized = device_dequantized.download();
    const auto output = device_output.download();
    const auto grad_input = device_grad_input.download();
    double expected_output = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require_close(dequantized[index], expected[index], 1.0e-6f,
                      "Q4_K dequantize");
        require_close(grad_input[index], expected[index] * 1.25f, 1.0e-6f,
                      "Q4_K backward input");
        expected_output += static_cast<double>(expected[index]) * input[index];
    }
    require_close(output[0], static_cast<float>(expected_output), 2.0e-4f,
                  "Q4_K linear");
}

void check_local_gqa_2048(void* library) {
    using DecodeFn = int (*)(const NfnNativeTileGlimmerGqaDecodeDescriptorV1*);
    using CommitFn = int (*)(const NfnNativeTileGlimmerCacheCommitDescriptorV1*);
    const auto decode = required_symbol<DecodeFn>(
        library, "nfn_native_tile_glimmer_gqa_decode_float32_v1");
    const auto commit = required_symbol<CommitFn>(
        library, "nfn_native_tile_glimmer_cache_commit_bf16_v1");
    constexpr std::int64_t query_heads = 32;
    constexpr std::int64_t kv_heads = 2;
    constexpr std::int64_t head_dim = 128;
    constexpr std::int64_t position = 2048;
    constexpr std::int64_t first_position = 1;
    constexpr std::int64_t capacity = 2048;
    constexpr std::int64_t row_stride = kv_heads * head_dim;
    constexpr float scale = 0.08838834764831845f;
    const std::size_t query_count = static_cast<std::size_t>(query_heads * head_dim);
    const std::size_t kv_count = static_cast<std::size_t>(kv_heads * head_dim);
    const std::size_t cache_count = static_cast<std::size_t>(capacity * row_stride);
    std::vector<float> query(query_count);
    std::vector<float> current_key(kv_count);
    std::vector<float> current_value(kv_count);
    std::vector<std::uint16_t> key_cache(cache_count, 0);
    std::vector<std::uint16_t> value_cache(cache_count, 0);
    for (std::size_t index = 0; index < query.size(); ++index) {
        query[index] = patterned_value(static_cast<std::int64_t>(index / head_dim),
                                       static_cast<std::int64_t>(index % head_dim),
                                       31, 29.0f);
    }
    for (std::int64_t head = 0; head < kv_heads; ++head) {
        for (std::int64_t dim = 0; dim < head_dim; ++dim) {
            const std::size_t index = static_cast<std::size_t>(head * head_dim + dim);
            current_key[index] = patterned_value(position + head, dim, 37, 41.0f);
            current_value[index] = patterned_value(position + head * 3, dim, 43, 47.0f);
        }
    }
    for (std::int64_t key_position = first_position;
         key_position < position; ++key_position) {
        const std::int64_t slot = key_position % capacity;
        for (std::int64_t head = 0; head < kv_heads; ++head) {
            for (std::int64_t dim = 0; dim < head_dim; ++dim) {
                const std::size_t index = static_cast<std::size_t>(
                    slot * row_stride + head * head_dim + dim);
                key_cache[index] = float_to_bf16(
                    patterned_value(key_position + head, dim, 37, 41.0f));
                value_cache[index] = float_to_bf16(
                    patterned_value(key_position + head * 3, dim, 43, 47.0f));
            }
        }
    }
    DeviceBuffer<float> device_query(query.size());
    DeviceBuffer<float> device_current_key(current_key.size());
    DeviceBuffer<float> device_current_value(current_value.size());
    DeviceBuffer<std::uint16_t> device_key_cache(key_cache.size());
    DeviceBuffer<std::uint16_t> device_value_cache(value_cache.size());
    DeviceBuffer<float> device_output(query.size());
    device_query.upload(query);
    device_current_key.upload(current_key);
    device_current_value.upload(current_value);
    device_key_cache.upload(key_cache);
    device_value_cache.upload(value_cache);
    NfnNativeTileGlimmerGqaDecodeDescriptorV1 descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
    descriptor.query = device_query.get();
    descriptor.current_key = device_current_key.get();
    descriptor.current_value = device_current_value.get();
    descriptor.key_cache_bf16 = device_key_cache.get();
    descriptor.value_cache_bf16 = device_value_cache.get();
    descriptor.output = device_output.get();
    descriptor.query_heads = query_heads;
    descriptor.kv_heads = kv_heads;
    descriptor.head_dim = head_dim;
    descriptor.position = position;
    descriptor.first_key_position = first_position;
    descriptor.cache_capacity = capacity;
    descriptor.cache_row_stride = row_stride;
    descriptor.scale = scale;
    cuda_check(static_cast<cudaError_t>(decode(&descriptor)),
               "Glimmer local GQA-2048 launch");
    cuda_check(cudaDeviceSynchronize(), "Glimmer local GQA-2048 synchronize");
    const auto output = device_output.download();
    std::vector<double> scores(static_cast<std::size_t>(position - first_position + 1));
    for (std::int64_t query_head = 0; query_head < query_heads; ++query_head) {
        const std::int64_t kv_head = query_head * kv_heads / query_heads;
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::int64_t key_position = first_position;
             key_position <= position; ++key_position) {
            double dot = 0.0;
            for (std::int64_t dim = 0; dim < head_dim; ++dim) {
                const float key_value = key_position == position
                    ? current_key[static_cast<std::size_t>(kv_head * head_dim + dim)]
                    : bf16_to_float(key_cache[static_cast<std::size_t>(
                          (key_position % capacity) * row_stride +
                          kv_head * head_dim + dim)]);
                dot += static_cast<double>(query[static_cast<std::size_t>(
                           query_head * head_dim + dim)]) * key_value;
            }
            const std::size_t score_index = static_cast<std::size_t>(
                key_position - first_position);
            scores[score_index] = dot * scale;
            maximum = std::max(maximum, scores[score_index]);
        }
        double denominator = 0.0;
        for (double& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (std::int64_t dim = 0; dim < head_dim; ++dim) {
            double expected = 0.0;
            for (std::int64_t key_position = first_position;
                 key_position <= position; ++key_position) {
                const float value = key_position == position
                    ? current_value[static_cast<std::size_t>(kv_head * head_dim + dim)]
                    : bf16_to_float(value_cache[static_cast<std::size_t>(
                          (key_position % capacity) * row_stride +
                          kv_head * head_dim + dim)]);
                expected += scores[static_cast<std::size_t>(key_position - first_position)] *
                    value;
            }
            require_close(output[static_cast<std::size_t>(query_head * head_dim + dim)],
                          static_cast<float>(expected / denominator), 4.0e-4f,
                          "local GQA-2048");
        }
    }
    NfnNativeTileGlimmerCacheCommitDescriptorV1 commit_descriptor{};
    commit_descriptor.struct_size = sizeof(commit_descriptor);
    commit_descriptor.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
    commit_descriptor.current_key = device_current_key.get();
    commit_descriptor.current_value = device_current_value.get();
    commit_descriptor.key_cache_bf16 = device_key_cache.get();
    commit_descriptor.value_cache_bf16 = device_value_cache.get();
    commit_descriptor.kv_heads = kv_heads;
    commit_descriptor.head_dim = head_dim;
    commit_descriptor.position = position;
    commit_descriptor.cache_capacity = capacity;
    commit_descriptor.cache_row_stride = row_stride;
    cuda_check(static_cast<cudaError_t>(commit(&commit_descriptor)),
               "Glimmer cache commit launch");
    cuda_check(cudaDeviceSynchronize(), "Glimmer cache commit synchronize");
    const auto committed_key = device_key_cache.download();
    const auto committed_value = device_value_cache.download();
    for (std::size_t index = 0; index < kv_count; ++index) {
        if (committed_key[index] != float_to_bf16(current_key[index]) ||
            committed_value[index] != float_to_bf16(current_value[index])) {
            throw std::runtime_error("transactional BF16 cache commit mismatch");
        }
    }
}

void check_dflash_block_attention(void* library) {
    using Fn = int (*)(const NfnNativeTileDFlashBlockAttentionDescriptorV1*);
    const auto operation = required_symbol<Fn>(
        library, "nfn_native_tile_dflash_block_attention_float32_v1");
    constexpr std::int64_t query_rows = 16;
    constexpr std::int64_t block_rows = 16;
    constexpr std::int64_t query_heads = 32;
    constexpr std::int64_t kv_heads = 8;
    constexpr std::int64_t head_dim = 128;
    constexpr std::int64_t context_length = 32;
    constexpr std::int64_t sliding_window = 64;
    constexpr std::int64_t capacity = 64;
    constexpr std::int64_t row_stride = kv_heads * head_dim;
    constexpr float scale = 0.08838834764831845f;
    std::vector<float> query(static_cast<std::size_t>(
        query_rows * query_heads * head_dim));
    std::vector<float> block_key(static_cast<std::size_t>(
        block_rows * kv_heads * head_dim));
    std::vector<float> block_value(block_key.size());
    std::vector<std::uint16_t> key_cache(static_cast<std::size_t>(
        capacity * row_stride), 0);
    std::vector<std::uint16_t> value_cache(key_cache.size(), 0);
    for (std::size_t index = 0; index < query.size(); ++index) {
        query[index] = patterned_value(static_cast<std::int64_t>(index / head_dim),
                                       static_cast<std::int64_t>(index % head_dim),
                                       29, 37.0f);
    }
    for (std::int64_t row = 0; row < block_rows; ++row) {
        for (std::int64_t head = 0; head < kv_heads; ++head) {
            for (std::int64_t dim = 0; dim < head_dim; ++dim) {
                const std::size_t index = static_cast<std::size_t>(
                    (row * kv_heads + head) * head_dim + dim);
                block_key[index] = patterned_value(context_length + row + head,
                                                   dim, 31, 43.0f);
                block_value[index] = patterned_value(context_length + row + head * 2,
                                                     dim, 41, 53.0f);
            }
        }
    }
    for (std::int64_t row = 0; row < context_length; ++row) {
        for (std::int64_t head = 0; head < kv_heads; ++head) {
            for (std::int64_t dim = 0; dim < head_dim; ++dim) {
                const std::size_t index = static_cast<std::size_t>(
                    row * row_stride + head * head_dim + dim);
                key_cache[index] = float_to_bf16(
                    patterned_value(row + head, dim, 31, 43.0f));
                value_cache[index] = float_to_bf16(
                    patterned_value(row + head * 2, dim, 41, 53.0f));
            }
        }
    }
    DeviceBuffer<float> device_query(query.size());
    DeviceBuffer<float> device_block_key(block_key.size());
    DeviceBuffer<float> device_block_value(block_value.size());
    DeviceBuffer<std::uint16_t> device_key_cache(key_cache.size());
    DeviceBuffer<std::uint16_t> device_value_cache(value_cache.size());
    DeviceBuffer<float> device_output(query.size());
    device_query.upload(query);
    device_block_key.upload(block_key);
    device_block_value.upload(block_value);
    device_key_cache.upload(key_cache);
    device_value_cache.upload(value_cache);
    NfnNativeTileDFlashBlockAttentionDescriptorV1 descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
    descriptor.query = device_query.get();
    descriptor.block_key = device_block_key.get();
    descriptor.block_value = device_block_value.get();
    descriptor.key_cache_bf16 = device_key_cache.get();
    descriptor.value_cache_bf16 = device_value_cache.get();
    descriptor.output = device_output.get();
    descriptor.query_rows = query_rows;
    descriptor.block_rows = block_rows;
    descriptor.query_heads = query_heads;
    descriptor.kv_heads = kv_heads;
    descriptor.head_dim = head_dim;
    descriptor.context_length = context_length;
    descriptor.sliding_window = sliding_window;
    descriptor.cache_capacity = capacity;
    descriptor.cache_row_stride = row_stride;
    descriptor.scale = scale;
    cuda_check(static_cast<cudaError_t>(operation(&descriptor)),
               "DFlash block attention launch");
    cuda_check(cudaDeviceSynchronize(), "DFlash block attention synchronize");
    const auto output = device_output.download();
    std::vector<double> weights(static_cast<std::size_t>(
        context_length + block_rows));
    for (std::int64_t query_row = 0; query_row < query_rows; ++query_row) {
        for (std::int64_t query_head = 0; query_head < query_heads; ++query_head) {
            const std::int64_t kv_head = query_head * kv_heads / query_heads;
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::int64_t key_position = 0;
                 key_position < context_length + block_rows; ++key_position) {
                double dot = 0.0;
                for (std::int64_t dim = 0; dim < head_dim; ++dim) {
                    const float key_value = key_position < context_length
                        ? bf16_to_float(key_cache[static_cast<std::size_t>(
                              key_position * row_stride + kv_head * head_dim + dim)])
                        : block_key[static_cast<std::size_t>(
                              ((key_position - context_length) * kv_heads + kv_head) *
                                  head_dim + dim)];
                    dot += static_cast<double>(query[static_cast<std::size_t>(
                               (query_row * query_heads + query_head) * head_dim + dim)]) *
                        key_value;
                }
                weights[static_cast<std::size_t>(key_position)] = dot * scale;
                maximum = std::max(maximum,
                                   weights[static_cast<std::size_t>(key_position)]);
            }
            double denominator = 0.0;
            for (double& weight : weights) {
                weight = std::exp(weight - maximum);
                denominator += weight;
            }
            for (std::int64_t dim = 0; dim < head_dim; ++dim) {
                double expected = 0.0;
                for (std::int64_t key_position = 0;
                     key_position < context_length + block_rows; ++key_position) {
                    const float value = key_position < context_length
                        ? bf16_to_float(value_cache[static_cast<std::size_t>(
                              key_position * row_stride + kv_head * head_dim + dim)])
                        : block_value[static_cast<std::size_t>(
                              ((key_position - context_length) * kv_heads + kv_head) *
                                  head_dim + dim)];
                    expected += weights[static_cast<std::size_t>(key_position)] * value;
                }
                require_close(output[static_cast<std::size_t>(
                                  (query_row * query_heads + query_head) * head_dim + dim)],
                              static_cast<float>(expected / denominator), 5.0e-4f,
                              "DFlash block attention");
            }
        }
    }
}

void check_post_training_losses(void* library) {
    using CeFn = int (*)(const NfnNativeTileGlimmerMaskedCeDescriptorV1*);
    using DpoFn = int (*)(const NfnNativeTileDpoPairwiseDescriptorV1*);
    using RewardFn = int (*)(const NfnNativeTileMaskedRewardHeadDescriptorV1*);
    using PreferenceFn = int (*)(const NfnNativeTilePreferenceBceDescriptorV1*);
    using PpoFn = int (*)(const NfnNativeTileMaskedPpoLossDescriptorV1*);
    const auto masked_ce = required_symbol<CeFn>(
        library, "nfn_native_tile_glimmer_masked_cross_entropy_i32_float32_v1");
    const auto dpo_forward = required_symbol<DpoFn>(
        library, "nfn_native_tile_dpo_pairwise_loss_float32_forward_v1");
    const auto dpo_backward = required_symbol<DpoFn>(
        library, "nfn_native_tile_dpo_pairwise_loss_float32_backward_v1");
    const auto reward_forward = required_symbol<RewardFn>(
        library, "nfn_native_tile_masked_reward_head_float32_forward_v1");
    const auto reward_backward = required_symbol<RewardFn>(
        library, "nfn_native_tile_masked_reward_head_float32_backward_v1");
    const auto preference_forward = required_symbol<PreferenceFn>(
        library, "nfn_native_tile_preference_bce_loss_float32_forward_v1");
    const auto preference_backward = required_symbol<PreferenceFn>(
        library, "nfn_native_tile_preference_bce_loss_float32_backward_v1");
    const auto ppo_forward = required_symbol<PpoFn>(
        library, "nfn_native_tile_masked_ppo_loss_float32_forward_v1");
    const auto ppo_backward = required_symbol<PpoFn>(
        library, "nfn_native_tile_masked_ppo_loss_float32_backward_v1");

    constexpr std::int64_t vocab_size = 202048;
    constexpr std::int32_t target = 201818;
    std::vector<float> logits(static_cast<std::size_t>(vocab_size), 0.0f);
    logits[static_cast<std::size_t>(target)] = 1.0f;
    DeviceBuffer<float> device_logits(logits.size());
    DeviceBuffer<std::int32_t> device_target(1);
    DeviceBuffer<float> device_mask(1);
    DeviceBuffer<float> device_row_loss(1);
    DeviceBuffer<float> device_grad_logits(logits.size());
    device_logits.upload(logits);
    device_target.upload(std::vector<std::int32_t>{target});
    device_mask.upload(std::vector<float>{1.0f});
    NfnNativeTileGlimmerMaskedCeDescriptorV1 ce_descriptor{};
    ce_descriptor.struct_size = sizeof(ce_descriptor);
    ce_descriptor.version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1;
    ce_descriptor.transformed_logits = device_logits.get();
    ce_descriptor.targets = device_target.get();
    ce_descriptor.loss_mask = device_mask.get();
    ce_descriptor.row_loss = device_row_loss.get();
    ce_descriptor.grad_transformed_logits = device_grad_logits.get();
    ce_descriptor.rows = 1;
    ce_descriptor.vocab_size = vocab_size;
    ce_descriptor.ignore_index = -100;
    ce_descriptor.grad_scale = 1.0f;
    cuda_check(static_cast<cudaError_t>(masked_ce(&ce_descriptor)),
               "wide masked CE launch");
    cuda_check(cudaDeviceSynchronize(), "wide masked CE synchronize");
    const auto row_loss = device_row_loss.download();
    const auto grad_logits = device_grad_logits.download();
    const double denominator = static_cast<double>(vocab_size - 1) + std::exp(1.0);
    require_close(row_loss[0], static_cast<float>(std::log(denominator) - 1.0),
                  2.0e-5f, "wide masked CE loss");
    require_close(grad_logits[static_cast<std::size_t>(target)],
                  static_cast<float>(std::exp(1.0) / denominator - 1.0),
                  2.0e-6f, "wide masked CE target gradient");
    require_close(grad_logits[0], static_cast<float>(1.0 / denominator),
                  2.0e-8f, "wide masked CE non-target gradient");

    const std::vector<float> policy_chosen{-1.0f, -2.0f, -0.7f};
    const std::vector<float> policy_rejected{-1.5f, -1.8f, -1.2f};
    const std::vector<float> reference_chosen{-1.1f, -2.1f, -0.9f};
    const std::vector<float> reference_rejected{-1.4f, -1.9f, -1.0f};
    DeviceBuffer<float> device_policy_chosen(policy_chosen.size());
    DeviceBuffer<float> device_policy_rejected(policy_rejected.size());
    DeviceBuffer<float> device_reference_chosen(reference_chosen.size());
    DeviceBuffer<float> device_reference_rejected(reference_rejected.size());
    DeviceBuffer<float> device_dpo_loss(policy_chosen.size());
    DeviceBuffer<float> device_chosen_reward(policy_chosen.size());
    DeviceBuffer<float> device_rejected_reward(policy_chosen.size());
    DeviceBuffer<float> device_grad_chosen(policy_chosen.size());
    DeviceBuffer<float> device_grad_rejected(policy_chosen.size());
    device_policy_chosen.upload(policy_chosen);
    device_policy_rejected.upload(policy_rejected);
    device_reference_chosen.upload(reference_chosen);
    device_reference_rejected.upload(reference_rejected);
    NfnNativeTileDpoPairwiseDescriptorV1 dpo_descriptor{};
    dpo_descriptor.struct_size = sizeof(dpo_descriptor);
    dpo_descriptor.version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1;
    dpo_descriptor.loss_type = NFN_NATIVE_TILE_DPO_LOSS_SIGMOID;
    dpo_descriptor.policy_logp_chosen = device_policy_chosen.get();
    dpo_descriptor.policy_logp_rejected = device_policy_rejected.get();
    dpo_descriptor.reference_logp_chosen = device_reference_chosen.get();
    dpo_descriptor.reference_logp_rejected = device_reference_rejected.get();
    dpo_descriptor.row_loss = device_dpo_loss.get();
    dpo_descriptor.chosen_reward = device_chosen_reward.get();
    dpo_descriptor.rejected_reward = device_rejected_reward.get();
    dpo_descriptor.grad_policy_logp_chosen = device_grad_chosen.get();
    dpo_descriptor.grad_policy_logp_rejected = device_grad_rejected.get();
    dpo_descriptor.examples = static_cast<std::int64_t>(policy_chosen.size());
    dpo_descriptor.beta = 0.2f;
    dpo_descriptor.label_smoothing = 0.1f;
    dpo_descriptor.grad_scale = 0.5f;
    cuda_check(static_cast<cudaError_t>(dpo_forward(&dpo_descriptor)),
               "DPO forward launch");
    cuda_check(static_cast<cudaError_t>(dpo_backward(&dpo_descriptor)),
               "DPO backward launch");
    cuda_check(cudaDeviceSynchronize(), "DPO synchronize");
    const auto dpo_loss = device_dpo_loss.download();
    const auto chosen_reward = device_chosen_reward.download();
    const auto rejected_reward = device_rejected_reward.download();
    const auto grad_chosen = device_grad_chosen.download();
    const auto grad_rejected = device_grad_rejected.download();
    for (std::size_t index = 0; index < policy_chosen.size(); ++index) {
        const float chosen_ratio = policy_chosen[index] - reference_chosen[index];
        const float rejected_ratio = policy_rejected[index] - reference_rejected[index];
        const float dpo_logit = 0.2f * (chosen_ratio - rejected_ratio);
        const float expected_loss = 0.9f * std::log1p(std::exp(-dpo_logit)) +
            0.1f * std::log1p(std::exp(dpo_logit));
        const float derivative = 1.0f / (1.0f + std::exp(-dpo_logit)) - 0.9f;
        const float expected_gradient = 0.5f * 0.2f * derivative;
        require_close(dpo_loss[index], expected_loss, 2.0e-6f, "DPO loss");
        require_close(chosen_reward[index], 0.2f * chosen_ratio, 1.0e-7f,
                      "DPO chosen reward");
        require_close(rejected_reward[index], 0.2f * rejected_ratio, 1.0e-7f,
                      "DPO rejected reward");
        require_close(grad_chosen[index], expected_gradient, 1.0e-7f,
                      "DPO chosen gradient");
        require_close(grad_rejected[index], -expected_gradient, 1.0e-7f,
                      "DPO rejected gradient");
    }

    constexpr std::int64_t reward_batch = 2;
    constexpr std::int64_t reward_sequence = 3;
    constexpr std::int64_t reward_hidden = 6656;
    std::vector<float> reward_weight(static_cast<std::size_t>(reward_hidden));
    std::vector<std::uint16_t> reward_weight_bf16(
        static_cast<std::size_t>(reward_hidden));
    std::vector<float> reward_hidden_values(static_cast<std::size_t>(
        reward_batch * reward_sequence * reward_hidden));
    const std::vector<float> reward_mask{1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f};
    for (std::size_t index = 0; index < reward_weight.size(); ++index) {
        reward_weight[index] = static_cast<float>(static_cast<int>(index % 13) - 6) /
            97.0f;
        reward_weight_bf16[index] = float_to_bf16(reward_weight[index]);
        reward_weight[index] = bf16_to_float(reward_weight_bf16[index]);
    }
    for (std::size_t index = 0; index < reward_hidden_values.size(); ++index) {
        reward_hidden_values[index] = patterned_value(
            static_cast<std::int64_t>(index / reward_hidden),
            static_cast<std::int64_t>(index % reward_hidden), 23, 89.0f);
    }
    DeviceBuffer<std::uint16_t> device_reward_weight(reward_weight_bf16.size());
    DeviceBuffer<float> device_reward_hidden(reward_hidden_values.size());
    DeviceBuffer<float> device_reward_mask(reward_mask.size());
    DeviceBuffer<float> device_reward(reward_batch);
    DeviceBuffer<std::int32_t> device_selected(reward_batch);
    DeviceBuffer<float> device_grad_reward(reward_batch);
    DeviceBuffer<float> device_grad_hidden(reward_hidden_values.size());
    DeviceBuffer<float> device_grad_weight(reward_weight.size());
    device_reward_weight.upload(reward_weight_bf16);
    device_reward_hidden.upload(reward_hidden_values);
    device_reward_mask.upload(reward_mask);
    device_grad_reward.upload(std::vector<float>{0.5f, -0.25f});
    NfnNativeTilePackedWeightDescriptorV1 reward_weight_descriptor{};
    reward_weight_descriptor.struct_size = sizeof(reward_weight_descriptor);
    reward_weight_descriptor.version = NFN_NATIVE_TILE_PACKED_WEIGHT_V1;
    reward_weight_descriptor.encoding = NFN_NATIVE_TILE_PACKED_WEIGHT_BF16;
    reward_weight_descriptor.data = reinterpret_cast<const std::uint8_t*>(
        device_reward_weight.get());
    reward_weight_descriptor.data_nbytes = reward_hidden * sizeof(std::uint16_t);
    reward_weight_descriptor.output_dim = 1;
    reward_weight_descriptor.input_dim = reward_hidden;
    reward_weight_descriptor.row_stride_bytes = reward_hidden * sizeof(std::uint16_t);
    NfnNativeTileMaskedRewardHeadDescriptorV1 reward_descriptor{};
    reward_descriptor.struct_size = sizeof(reward_descriptor);
    reward_descriptor.version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1;
    reward_descriptor.hidden = device_reward_hidden.get();
    reward_descriptor.sequence_mask = device_reward_mask.get();
    reward_descriptor.weight = &reward_weight_descriptor;
    reward_descriptor.reward = device_reward.get();
    reward_descriptor.selected_positions = device_selected.get();
    reward_descriptor.grad_reward = device_grad_reward.get();
    reward_descriptor.grad_hidden = device_grad_hidden.get();
    reward_descriptor.grad_weight = device_grad_weight.get();
    reward_descriptor.batch_size = reward_batch;
    reward_descriptor.sequence_length = reward_sequence;
    reward_descriptor.hidden_size = reward_hidden;
    cuda_check(static_cast<cudaError_t>(reward_forward(&reward_descriptor)),
               "reward forward launch");
    cuda_check(cudaDeviceSynchronize(), "reward forward synchronize");
    cuda_check(static_cast<cudaError_t>(reward_backward(&reward_descriptor)),
               "reward backward launch");
    cuda_check(cudaDeviceSynchronize(), "reward backward synchronize");
    const auto rewards = device_reward.download();
    const auto selected = device_selected.download();
    const auto reward_grad_hidden = device_grad_hidden.download();
    const auto reward_grad_weight = device_grad_weight.download();
    const std::vector<std::int32_t> expected_selected{1, 2};
    const std::vector<float> reward_upstream{0.5f, -0.25f};
    for (std::int64_t example = 0; example < reward_batch; ++example) {
        if (selected[static_cast<std::size_t>(example)] !=
            expected_selected[static_cast<std::size_t>(example)]) {
            throw std::runtime_error("reward selected-position mismatch");
        }
        double expected_reward = 0.0;
        const std::int64_t row = example * reward_sequence +
            expected_selected[static_cast<std::size_t>(example)];
        for (std::int64_t column = 0; column < reward_hidden; ++column) {
            expected_reward += static_cast<double>(
                reward_hidden_values[static_cast<std::size_t>(row * reward_hidden + column)]) *
                reward_weight[static_cast<std::size_t>(column)];
        }
        require_close(rewards[static_cast<std::size_t>(example)],
                      static_cast<float>(expected_reward), 2.0e-5f,
                      "reward value");
    }
    for (std::int64_t column = 0; column < reward_hidden; ++column) {
        float expected_weight_gradient = 0.0f;
        for (std::int64_t example = 0; example < reward_batch; ++example) {
            const std::int64_t row = example * reward_sequence +
                expected_selected[static_cast<std::size_t>(example)];
            expected_weight_gradient += reward_upstream[static_cast<std::size_t>(example)] *
                reward_hidden_values[static_cast<std::size_t>(
                    row * reward_hidden + column)];
        }
        require_close(reward_grad_weight[static_cast<std::size_t>(column)],
                      expected_weight_gradient, 2.0e-6f,
                      "reward weight gradient");
        for (std::int64_t example = 0; example < reward_batch; ++example) {
            for (std::int64_t position = 0; position < reward_sequence; ++position) {
                const float expected = position ==
                        expected_selected[static_cast<std::size_t>(example)]
                    ? reward_upstream[static_cast<std::size_t>(example)] *
                          reward_weight[static_cast<std::size_t>(column)]
                    : 0.0f;
                require_close(reward_grad_hidden[static_cast<std::size_t>(
                                  (example * reward_sequence + position) * reward_hidden +
                                  column)],
                              expected, 1.0e-7f, "reward hidden gradient");
            }
        }
    }

    DeviceBuffer<float> device_preference_loss(reward_batch);
    DeviceBuffer<float> device_preference_grad_chosen(reward_batch);
    DeviceBuffer<float> device_preference_grad_rejected(reward_batch);
    NfnNativeTilePreferenceBceDescriptorV1 preference_descriptor{};
    preference_descriptor.struct_size = sizeof(preference_descriptor);
    preference_descriptor.version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1;
    preference_descriptor.reward_chosen = device_reward.get();
    preference_descriptor.reward_rejected = device_chosen_reward.get();
    preference_descriptor.row_loss = device_preference_loss.get();
    preference_descriptor.grad_reward_chosen = device_preference_grad_chosen.get();
    preference_descriptor.grad_reward_rejected = device_preference_grad_rejected.get();
    preference_descriptor.examples = reward_batch;
    preference_descriptor.grad_scale = 0.5f;
    cuda_check(static_cast<cudaError_t>(preference_forward(&preference_descriptor)),
               "preference BCE forward launch");
    cuda_check(static_cast<cudaError_t>(preference_backward(&preference_descriptor)),
               "preference BCE backward launch");
    cuda_check(cudaDeviceSynchronize(), "preference BCE synchronize");
    const auto preference_loss = device_preference_loss.download();
    const auto preference_grad_chosen = device_preference_grad_chosen.download();
    const auto preference_grad_rejected = device_preference_grad_rejected.download();
    for (std::int64_t example = 0; example < reward_batch; ++example) {
        const float difference = rewards[static_cast<std::size_t>(example)] -
            chosen_reward[static_cast<std::size_t>(example)];
        const float expected_loss = std::log1p(std::exp(-difference));
        const float expected_gradient = 0.5f *
            (1.0f / (1.0f + std::exp(-difference)) - 1.0f);
        require_close(preference_loss[static_cast<std::size_t>(example)],
                      expected_loss, 2.0e-6f, "preference BCE loss");
        require_close(preference_grad_chosen[static_cast<std::size_t>(example)],
                      expected_gradient, 2.0e-7f,
                      "preference BCE chosen gradient");
        require_close(preference_grad_rejected[static_cast<std::size_t>(example)],
                      -expected_gradient, 2.0e-7f,
                      "preference BCE rejected gradient");
    }

    const std::vector<float> logp_new{-0.2f, -0.1f, -0.5f, -0.3f};
    const std::vector<float> logp_old{-0.25f, -0.05f, -0.6f, -0.4f};
    const std::vector<float> advantages{1.0f, -1.0f, 0.5f, -0.2f};
    const std::vector<float> value_new{0.4f, 0.2f, 0.8f, -0.1f};
    const std::vector<float> value_old{0.3f, 0.25f, 0.7f, 0.0f};
    const std::vector<float> returns{0.5f, 0.0f, 1.0f, 0.2f};
    const std::vector<float> ppo_mask{1.0f, 1.0f, 0.0f, 1.0f};
    const std::vector<float> entropy{0.6f, 0.5f, 0.4f, 0.3f};
    DeviceBuffer<float> device_logp_new(logp_new.size());
    DeviceBuffer<float> device_logp_old(logp_old.size());
    DeviceBuffer<float> device_advantages(advantages.size());
    DeviceBuffer<float> device_value_new(value_new.size());
    DeviceBuffer<float> device_value_old(value_old.size());
    DeviceBuffer<float> device_returns(returns.size());
    DeviceBuffer<float> device_ppo_mask(ppo_mask.size());
    DeviceBuffer<float> device_entropy(entropy.size());
    DeviceBuffer<float> device_policy_loss(1);
    DeviceBuffer<float> device_value_loss(1);
    DeviceBuffer<float> device_entropy_bonus(1);
    DeviceBuffer<float> device_total_loss(1);
    DeviceBuffer<float> device_grad_logp_new(logp_new.size());
    DeviceBuffer<float> device_grad_value_new(value_new.size());
    DeviceBuffer<float> device_grad_entropy(entropy.size());
    device_logp_new.upload(logp_new);
    device_logp_old.upload(logp_old);
    device_advantages.upload(advantages);
    device_value_new.upload(value_new);
    device_value_old.upload(value_old);
    device_returns.upload(returns);
    device_ppo_mask.upload(ppo_mask);
    device_entropy.upload(entropy);
    NfnNativeTileMaskedPpoLossDescriptorV1 ppo_descriptor{};
    ppo_descriptor.struct_size = sizeof(ppo_descriptor);
    ppo_descriptor.version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1;
    ppo_descriptor.logp_new = device_logp_new.get();
    ppo_descriptor.logp_old = device_logp_old.get();
    ppo_descriptor.advantages = device_advantages.get();
    ppo_descriptor.value_new = device_value_new.get();
    ppo_descriptor.value_old = device_value_old.get();
    ppo_descriptor.returns = device_returns.get();
    ppo_descriptor.loss_mask = device_ppo_mask.get();
    ppo_descriptor.entropy = device_entropy.get();
    ppo_descriptor.policy_loss = device_policy_loss.get();
    ppo_descriptor.value_loss = device_value_loss.get();
    ppo_descriptor.entropy_bonus = device_entropy_bonus.get();
    ppo_descriptor.total_loss = device_total_loss.get();
    ppo_descriptor.grad_logp_new = device_grad_logp_new.get();
    ppo_descriptor.grad_value_new = device_grad_value_new.get();
    ppo_descriptor.grad_entropy = device_grad_entropy.get();
    ppo_descriptor.rows = static_cast<std::int64_t>(logp_new.size());
    ppo_descriptor.clip_range = 0.2f;
    ppo_descriptor.value_coefficient = 0.5f;
    ppo_descriptor.entropy_coefficient = 0.01f;
    ppo_descriptor.epsilon = 1.0e-8f;
    cuda_check(static_cast<cudaError_t>(ppo_forward(&ppo_descriptor)),
               "PPO forward launch");
    cuda_check(static_cast<cudaError_t>(ppo_backward(&ppo_descriptor)),
               "PPO backward launch");
    cuda_check(cudaDeviceSynchronize(), "PPO synchronize");
    const float active = 3.0f;
    double expected_policy_sum = 0.0;
    double expected_value_sum = 0.0;
    double expected_entropy_sum = 0.0;
    for (std::size_t index = 0; index < logp_new.size(); ++index) {
        if (!(ppo_mask[index] > 0.0f)) continue;
        const double ratio = std::exp(static_cast<double>(logp_new[index] - logp_old[index]));
        const double clipped_ratio = std::min(1.2, std::max(0.8, ratio));
        expected_policy_sum -= std::min(ratio * advantages[index],
                                        clipped_ratio * advantages[index]);
        const double delta = value_new[index] - value_old[index];
        const double clipped_delta = std::min(0.2, std::max(-0.2, delta));
        const double raw_error = value_new[index] - returns[index];
        const double clipped_error = value_old[index] + clipped_delta - returns[index];
        expected_value_sum += 0.5 * std::max(raw_error * raw_error,
                                             clipped_error * clipped_error);
        expected_entropy_sum += entropy[index];
    }
    const float expected_policy = static_cast<float>(expected_policy_sum / active);
    const float expected_value = static_cast<float>(expected_value_sum / active);
    const float expected_entropy = static_cast<float>(expected_entropy_sum / active);
    require_close(device_policy_loss.download()[0], expected_policy, 2.0e-6f,
                  "PPO policy loss");
    require_close(device_value_loss.download()[0], expected_value, 2.0e-6f,
                  "PPO value loss");
    require_close(device_entropy_bonus.download()[0], expected_entropy, 2.0e-6f,
                  "PPO entropy bonus");
    require_close(device_total_loss.download()[0],
                  expected_policy + 0.5f * expected_value - 0.01f * expected_entropy,
                  2.0e-6f, "PPO total loss");
    const auto ppo_grad_logp = device_grad_logp_new.download();
    const auto ppo_grad_value = device_grad_value_new.download();
    const auto ppo_grad_entropy = device_grad_entropy.download();
    for (std::size_t index = 0; index < logp_new.size(); ++index) {
        if (!(ppo_mask[index] > 0.0f)) {
            require_close(ppo_grad_logp[index], 0.0f, 0.0f, "PPO masked logp gradient");
            require_close(ppo_grad_value[index], 0.0f, 0.0f, "PPO masked value gradient");
            require_close(ppo_grad_entropy[index], 0.0f, 0.0f,
                          "PPO masked entropy gradient");
            continue;
        }
        const double ratio = std::exp(static_cast<double>(logp_new[index] - logp_old[index]));
        const bool policy_active = advantages[index] >= 0.0f
            ? ratio <= 1.2 : ratio >= 0.8;
        const float expected_logp_gradient = policy_active
            ? static_cast<float>(-advantages[index] * ratio / active) : 0.0f;
        const double delta = value_new[index] - value_old[index];
        const double clipped_delta = std::min(0.2, std::max(-0.2, delta));
        const double raw_error = value_new[index] - returns[index];
        const double clipped_error = value_old[index] + clipped_delta - returns[index];
        double value_gradient = 0.0;
        if (raw_error * raw_error >= clipped_error * clipped_error) {
            value_gradient = raw_error;
        } else if (delta >= -0.2 && delta <= 0.2) {
            value_gradient = clipped_error;
        }
        require_close(ppo_grad_logp[index], expected_logp_gradient, 2.0e-6f,
                      "PPO logp gradient");
        require_close(ppo_grad_value[index],
                      static_cast<float>(0.5 * value_gradient / active), 2.0e-6f,
                      "PPO value gradient");
        require_close(ppo_grad_entropy[index], -0.01f / active, 1.0e-8f,
                      "PPO entropy gradient");
    }
}

void check_vision_layer_norm(void* library) {
    using Fn = int (*)(const float*, const float*, const float*, float*,
                       std::int64_t, std::int64_t, float, void*);
    const auto operation = required_symbol<Fn>(
        library, "nfn_native_tile_glimmer_vision_layer_norm_float32_v1");
    constexpr std::int64_t rows = 2;
    constexpr std::int64_t width = 1536;
    constexpr float epsilon = 1.0e-6f;
    std::vector<float> input(static_cast<std::size_t>(rows * width));
    std::vector<float> weight(static_cast<std::size_t>(width));
    std::vector<float> bias(static_cast<std::size_t>(width));
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(static_cast<int>(index % 23) - 11) / 7.0f;
    }
    for (std::size_t index = 0; index < weight.size(); ++index) {
        weight[index] = 0.75f + static_cast<float>(index % 7) / 20.0f;
        bias[index] = static_cast<float>(static_cast<int>(index % 5) - 2) / 50.0f;
    }
    DeviceBuffer<float> device_input(input.size());
    DeviceBuffer<float> device_weight(weight.size());
    DeviceBuffer<float> device_bias(bias.size());
    DeviceBuffer<float> device_output(input.size());
    device_input.upload(input);
    device_weight.upload(weight);
    device_bias.upload(bias);
    cuda_check(static_cast<cudaError_t>(operation(
                   device_input.get(), device_weight.get(), device_bias.get(),
                   device_output.get(), rows, width, epsilon, nullptr)),
               "glimmer vision LayerNorm launch");
    cuda_check(cudaDeviceSynchronize(), "glimmer vision LayerNorm synchronize");
    const auto output = device_output.download();
    for (std::int64_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        double square_sum = 0.0;
        for (std::int64_t column = 0; column < width; ++column) {
            const double value = input[static_cast<std::size_t>(row * width + column)];
            sum += value;
            square_sum += value * value;
        }
        const double mean = sum / static_cast<double>(width);
        const double variance = square_sum / static_cast<double>(width) - mean * mean;
        const float inverse_std = static_cast<float>(1.0 / std::sqrt(variance + epsilon));
        for (std::int64_t column = 0; column < width; ++column) {
            const std::size_t index = static_cast<std::size_t>(row * width + column);
            const float expected =
                (input[index] - static_cast<float>(mean)) * inverse_std *
                    weight[static_cast<std::size_t>(column)] +
                bias[static_cast<std::size_t>(column)];
            require_close(output[index], expected, 3.0e-5f,
                          "vision LayerNorm");
        }
    }
}

void check_vision_prepare_attention_shuffle(void* library) {
    using PrepareFn = int (*)(const NfnNativeTileGlimmerVisionPrepareDescriptorV1*);
    using AttentionFn = int (*)(const NfnNativeTileGlimmerVisionAttentionDescriptorV1*);
    using ShuffleFn = int (*)(const NfnNativeTileGlimmerVisionPixelShuffleDescriptorV1*);
    const auto prepare = required_symbol<PrepareFn>(
        library, "nfn_native_tile_glimmer_vision_prepare_float32_v1");
    const auto attention = required_symbol<AttentionFn>(
        library, "nfn_native_tile_glimmer_vision_attention_float32_v1");
    const auto shuffle = required_symbol<ShuffleFn>(
        library, "nfn_native_tile_glimmer_vision_pixel_shuffle_float32_v1");
    constexpr std::int64_t rows = 4;
    constexpr std::int64_t heads = 12;
    constexpr std::int64_t head_dim = 128;
    constexpr std::int64_t width = heads * head_dim;
    constexpr std::int64_t position_rows = 8;
    std::vector<float> projected(static_cast<std::size_t>(rows * width));
    std::vector<float> positions(static_cast<std::size_t>(position_rows * width));
    std::vector<std::int32_t> corners(static_cast<std::size_t>(rows * 4));
    std::vector<float> corner_weights(corners.size());
    std::vector<std::int32_t> permutation{2, 0, 3, 1};
    for (std::size_t index = 0; index < projected.size(); ++index) {
        projected[index] = patterned_value(static_cast<std::int64_t>(index / width),
                                           static_cast<std::int64_t>(index % width),
                                           31, 41.0f);
    }
    for (std::size_t index = 0; index < positions.size(); ++index) {
        positions[index] = patterned_value(static_cast<std::int64_t>(index / width),
                                           static_cast<std::int64_t>(index % width),
                                           19, 71.0f);
    }
    for (std::int64_t row = 0; row < rows; ++row) {
        for (int corner = 0; corner < 4; ++corner) {
            corners[static_cast<std::size_t>(row * 4 + corner)] =
                static_cast<std::int32_t>((row + corner) % position_rows);
            corner_weights[static_cast<std::size_t>(row * 4 + corner)] =
                static_cast<float>(corner + 1) / 10.0f;
        }
    }
    DeviceBuffer<float> device_projected(projected.size());
    DeviceBuffer<float> device_positions(positions.size());
    DeviceBuffer<std::int32_t> device_corners(corners.size());
    DeviceBuffer<float> device_corner_weights(corner_weights.size());
    DeviceBuffer<std::int32_t> device_permutation(permutation.size());
    DeviceBuffer<float> device_prepared(projected.size());
    device_projected.upload(projected);
    device_positions.upload(positions);
    device_corners.upload(corners);
    device_corner_weights.upload(corner_weights);
    device_permutation.upload(permutation);
    NfnNativeTileGlimmerVisionPrepareDescriptorV1 prepare_descriptor{};
    prepare_descriptor.struct_size = sizeof(prepare_descriptor);
    prepare_descriptor.version = NFN_NATIVE_TILE_GLIMMER_VISION_V1;
    prepare_descriptor.projected = device_projected.get();
    prepare_descriptor.position_table = device_positions.get();
    prepare_descriptor.corner_indices = device_corners.get();
    prepare_descriptor.corner_weights = device_corner_weights.get();
    prepare_descriptor.permutation = device_permutation.get();
    prepare_descriptor.output = device_prepared.get();
    prepare_descriptor.rows = rows;
    prepare_descriptor.width = width;
    prepare_descriptor.position_rows = position_rows;
    cuda_check(static_cast<cudaError_t>(prepare(&prepare_descriptor)),
               "vision prepare launch");
    cuda_check(cudaDeviceSynchronize(), "vision prepare synchronize");
    const auto prepared = device_prepared.download();
    for (std::int64_t output_row = 0; output_row < rows; ++output_row) {
        const std::int64_t source_row = permutation[static_cast<std::size_t>(output_row)];
        for (std::int64_t dim = 0; dim < width; ++dim) {
            float expected = projected[static_cast<std::size_t>(source_row * width + dim)];
            for (int corner = 0; corner < 4; ++corner) {
                const std::size_t metadata = static_cast<std::size_t>(source_row * 4 + corner);
                expected += corner_weights[metadata] * positions[static_cast<std::size_t>(
                    corners[metadata] * width + dim)];
            }
            require_close(prepared[static_cast<std::size_t>(output_row * width + dim)],
                          expected, 2.0e-6f, "vision prepare");
        }
    }

    std::vector<float> query(static_cast<std::size_t>(rows * width));
    std::vector<float> key(query.size());
    std::vector<float> value(query.size());
    std::vector<std::int32_t> position_width(static_cast<std::size_t>(rows), 0);
    std::vector<std::int32_t> position_height(static_cast<std::size_t>(rows), 0);
    std::vector<std::int32_t> row_begin(static_cast<std::size_t>(rows), 0);
    std::vector<std::int32_t> row_end(static_cast<std::size_t>(rows),
                                      static_cast<std::int32_t>(rows));
    for (std::size_t index = 0; index < query.size(); ++index) {
        query[index] = patterned_value(static_cast<std::int64_t>(index / width),
                                       static_cast<std::int64_t>(index % width),
                                       23, 53.0f);
        key[index] = patterned_value(static_cast<std::int64_t>(index / width) + 3,
                                     static_cast<std::int64_t>(index % width),
                                     29, 61.0f);
        value[index] = patterned_value(static_cast<std::int64_t>(index / width) + 7,
                                       static_cast<std::int64_t>(index % width),
                                       31, 67.0f);
    }
    DeviceBuffer<float> device_query(query.size());
    DeviceBuffer<float> device_key(key.size());
    DeviceBuffer<float> device_value(value.size());
    DeviceBuffer<std::int32_t> device_position_width(position_width.size());
    DeviceBuffer<std::int32_t> device_position_height(position_height.size());
    DeviceBuffer<std::int32_t> device_row_begin(row_begin.size());
    DeviceBuffer<std::int32_t> device_row_end(row_end.size());
    DeviceBuffer<float> device_attention_output(query.size());
    device_query.upload(query);
    device_key.upload(key);
    device_value.upload(value);
    device_position_width.upload(position_width);
    device_position_height.upload(position_height);
    device_row_begin.upload(row_begin);
    device_row_end.upload(row_end);
    NfnNativeTileGlimmerVisionAttentionDescriptorV1 attention_descriptor{};
    attention_descriptor.struct_size = sizeof(attention_descriptor);
    attention_descriptor.version = NFN_NATIVE_TILE_GLIMMER_VISION_V1;
    attention_descriptor.query = device_query.get();
    attention_descriptor.key = device_key.get();
    attention_descriptor.value = device_value.get();
    attention_descriptor.position_width = device_position_width.get();
    attention_descriptor.position_height = device_position_height.get();
    attention_descriptor.row_begin = device_row_begin.get();
    attention_descriptor.row_end = device_row_end.get();
    attention_descriptor.output = device_attention_output.get();
    attention_descriptor.rows = rows;
    attention_descriptor.heads = heads;
    attention_descriptor.head_dim = head_dim;
    attention_descriptor.rope_theta = 10000.0f;
    cuda_check(static_cast<cudaError_t>(attention(&attention_descriptor)),
               "vision attention launch");
    cuda_check(cudaDeviceSynchronize(), "vision attention synchronize");
    const auto attention_output = device_attention_output.download();
    const double attention_scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    std::vector<double> weights(static_cast<std::size_t>(rows));
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t head = 0; head < heads; ++head) {
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::int64_t key_row = 0; key_row < rows; ++key_row) {
                double dot = 0.0;
                for (std::int64_t dim = 0; dim < head_dim; ++dim) {
                    dot += static_cast<double>(query[static_cast<std::size_t>(
                               row * width + head * head_dim + dim)]) *
                        key[static_cast<std::size_t>(
                            key_row * width + head * head_dim + dim)];
                }
                weights[static_cast<std::size_t>(key_row)] = dot * attention_scale;
                maximum = std::max(maximum, weights[static_cast<std::size_t>(key_row)]);
            }
            double denominator = 0.0;
            for (double& weight : weights) {
                weight = std::exp(weight - maximum);
                denominator += weight;
            }
            for (std::int64_t dim = 0; dim < head_dim; ++dim) {
                double expected = 0.0;
                for (std::int64_t key_row = 0; key_row < rows; ++key_row) {
                    expected += weights[static_cast<std::size_t>(key_row)] *
                        value[static_cast<std::size_t>(
                            key_row * width + head * head_dim + dim)];
                }
                require_close(attention_output[static_cast<std::size_t>(
                                  row * width + head * head_dim + dim)],
                              static_cast<float>(expected / denominator), 2.0e-4f,
                              "vision attention");
            }
        }
    }

    constexpr std::int64_t merge_area = 4;
    constexpr std::int64_t merged_rows = 1;
    const std::vector<std::int32_t> source_rows{2, 0, 3, 1};
    DeviceBuffer<std::int32_t> device_source_rows(source_rows.size());
    DeviceBuffer<float> device_shuffle_output(static_cast<std::size_t>(width * merge_area));
    device_source_rows.upload(source_rows);
    NfnNativeTileGlimmerVisionPixelShuffleDescriptorV1 shuffle_descriptor{};
    shuffle_descriptor.struct_size = sizeof(shuffle_descriptor);
    shuffle_descriptor.version = NFN_NATIVE_TILE_GLIMMER_VISION_V1;
    shuffle_descriptor.reordered_hidden = device_prepared.get();
    shuffle_descriptor.source_rows = device_source_rows.get();
    shuffle_descriptor.output = device_shuffle_output.get();
    shuffle_descriptor.merged_rows = merged_rows;
    shuffle_descriptor.hidden_size = width;
    shuffle_descriptor.merge_area = merge_area;
    cuda_check(static_cast<cudaError_t>(shuffle(&shuffle_descriptor)),
               "vision pixel shuffle launch");
    cuda_check(cudaDeviceSynchronize(), "vision pixel shuffle synchronize");
    const auto shuffled = device_shuffle_output.download();
    for (std::int64_t dim = 0; dim < width; ++dim) {
        for (std::int64_t slot = 0; slot < merge_area; ++slot) {
            require_close(shuffled[static_cast<std::size_t>(dim * merge_area + slot)],
                          prepared[static_cast<std::size_t>(
                              source_rows[static_cast<std::size_t>(slot)] * width + dim)],
                          0.0f, "vision pixel shuffle");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 3) {
            throw std::runtime_error(
                "usage: muse_glimmer_cuda_kernel_probe STRICT_TILE_OPS_SO [CUDA_DEVICE]");
        }
        const int cuda_device = argc == 3 ? std::stoi(argv[2]) : 0;
        if (cuda_device < 0) {
            throw std::runtime_error("CUDA_DEVICE must be non-negative");
        }
        void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (library == nullptr) {
            throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
        }
        using AbiFn = int (*)();
        const auto strict_abi = required_symbol<AbiFn>(
            library, "nfn_native_tile_strict_math_abi_version");
        const auto inference_abi = required_symbol<AbiFn>(
            library, "nfn_native_tile_glimmer_inference_abi_version");
        const auto vision_abi = required_symbol<AbiFn>(
            library, "nfn_native_tile_glimmer_vision_abi_version");
        if (strict_abi() != 1 || inference_abi() != 1 || vision_abi() != 1) {
            throw std::runtime_error("strict/inference/vision ABI mismatch");
        }
        cuda_check(cudaSetDevice(cuda_device), "cudaSetDevice");
        check_sigmoid_gate(library);
        check_q4_k_packed_linear(library);
        check_wide_rms_norm(library);
        check_positioned_rope(library);
        check_local_gqa_2048(library);
        check_dflash_block_attention(library);
        check_post_training_losses(library);
        check_vision_layer_norm(library);
        check_vision_prepare_attention_shuffle(library);
        cuda_check(cudaDeviceSynchronize(), "final cudaDeviceSynchronize");
        dlclose(library);
        std::cout
            << "{\"status\":\"passed\",\"device\":" << cuda_device << ","
            << "\"kernels\":[\"q4_k_dequant_linear_dx\","
            << "\"sigmoid_gate\",\"rms_norm_6656\","
            << "\"positioned_rope_q32_kv2_h128\","
            << "\"gqa_decode_q32_kv2_h128_window2048\","
            << "\"cache_commit_bf16\","
            << "\"dflash_block_attention_q16_q32_kv8_h128\","
            << "\"masked_ce_vocab202048_i32\",\"dpo_forward_backward\","
            << "\"reward_head_6656_forward_backward\","
            << "\"preference_bce_forward_backward\","
            << "\"ppo_forward_backward\","
            << "\"vision_prepare_1536\",\"vision_layer_norm_1536\","
            << "\"vision_attention_12x128\","
            << "\"vision_pixel_shuffle_6144\"]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Muse Glimmer CUDA kernel probe failed: " << error.what() << '\n';
        return 1;
    }
}
