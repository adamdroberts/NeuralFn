#include "tile_ops.h"

#include <cuda_fp16.h>
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

void check_device_indirect_embedding(void* library) {
    using HostFn = int (*)(
        const NfnNativeTilePackedWeightDescriptorV1*, std::int64_t, float*);
    using DeviceFn = int (*)(
        const NfnNativeTilePackedWeightDescriptorV1*, const std::int64_t*,
        float*);
    const auto host_gather = required_symbol<HostFn>(
        library, "nfn_native_tile_glimmer_embedding_gather_float32_v1");
    const auto device_gather = required_symbol<DeviceFn>(
        library,
        "nfn_native_tile_glimmer_embedding_gather_device_i64_float32_v1");

    constexpr std::int64_t vocab_size = 3;
    constexpr std::int64_t width = 8;
    constexpr std::int64_t token_id = 2;
    std::vector<float> weight(static_cast<std::size_t>(vocab_size * width));
    for (std::int64_t row = 0; row < vocab_size; ++row) {
        for (std::int64_t column = 0; column < width; ++column) {
            weight[static_cast<std::size_t>(row * width + column)] =
                static_cast<float>(row * 17 + column - 11) / 23.0f;
        }
    }
    DeviceBuffer<float> device_weight(weight.size());
    DeviceBuffer<std::int64_t> device_token_id(1);
    DeviceBuffer<float> host_output(static_cast<std::size_t>(width));
    DeviceBuffer<float> device_output(static_cast<std::size_t>(width));
    device_weight.upload(weight);
    device_token_id.upload(std::vector<std::int64_t>{token_id});

    NfnNativeTilePackedWeightDescriptorV1 descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.version = NFN_NATIVE_TILE_PACKED_WEIGHT_V1;
    descriptor.encoding = NFN_NATIVE_TILE_PACKED_WEIGHT_F32;
    descriptor.data = reinterpret_cast<const std::uint8_t*>(
        device_weight.get());
    descriptor.data_nbytes = static_cast<std::int64_t>(
        weight.size() * sizeof(float));
    descriptor.output_dim = vocab_size;
    descriptor.input_dim = width;
    descriptor.row_stride_bytes = width * static_cast<std::int64_t>(sizeof(float));

    cuda_check(static_cast<cudaError_t>(host_gather(
                   &descriptor, token_id, host_output.get())),
               "host-token embedding gather launch");
    cuda_check(static_cast<cudaError_t>(device_gather(
                   &descriptor, device_token_id.get(), device_output.get())),
               "device-token embedding gather launch");
    cuda_check(cudaDeviceSynchronize(),
               "device-token embedding gather synchronize");
    const auto expected = host_output.download();
    const auto actual = device_output.download();
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require_close(actual[index], expected[index], 0.0f,
                      "device-token embedding gather");
    }
    if (device_gather(&descriptor, nullptr, device_output.get()) == cudaSuccess) {
        throw std::runtime_error(
            "device-token embedding gather accepted a null token pointer");
    }
}

void check_wide_rms_norm(void* library) {
    using Fn = int (*)(const float*, const NfnNativeTilePackedWeightDescriptorV1*,
                       float*, std::int64_t, std::int64_t, float, bool, void*);
    using CaptureFn = int (*)(
        const float*, const NfnNativeTilePackedWeightDescriptorV1*, float*,
        float*, std::int64_t, std::int64_t, float, bool, void*);
    using CaptureQ8Fn = int (*)(
        const float*, const NfnNativeTilePackedWeightDescriptorV1*, float*,
        float*, std::int8_t*, float*, float*, std::int64_t, std::int64_t,
        float, bool, void*);
    using QuantizeFn = int (*)(
        const float*, std::int8_t*, float*, float*, std::int64_t,
        std::int64_t, void*);
    using AddFn = int (*)(
        const float*, const NfnNativeTilePackedWeightDescriptorV1*, const float*,
        float*, std::int64_t, std::int64_t, float, bool, void*);
    using DualFn = int (*)(
        const float*, const NfnNativeTilePackedWeightDescriptorV1*, const float*,
        float*, const NfnNativeTilePackedWeightDescriptorV1*, float*, float*,
        std::int64_t, std::int64_t, float, bool, float, bool, void*);
    using PlainAddFn = int (*)(
        const float*, const float*, float*, std::int64_t, void*);
    const auto operation = required_symbol<Fn>(
        library, "nfn_native_tile_glimmer_rms_norm_affine_float32_v1");
    const auto capture = required_symbol<CaptureFn>(
        library,
        "nfn_native_tile_glimmer_rms_norm_affine_capture_residual_float32_v1");
    const auto capture_q8 = required_symbol<CaptureQ8Fn>(
        library,
        "nfn_native_tile_glimmer_rms_norm_affine_capture_residual_q8_1_float32_v1");
    const auto quantize = required_symbol<QuantizeFn>(
        library, "nfn_native_tile_quantize_q8_1_float32_v1");
    const auto add = required_symbol<AddFn>(
        library,
        "nfn_native_tile_glimmer_rms_norm_affine_add_residual_float32_v1");
    const auto dual = required_symbol<DualFn>(
        library,
        "nfn_native_tile_glimmer_dual_rms_add_capture_float32_v1");
    const auto cooperative_dual = required_symbol<DualFn>(
        library,
        "nfn_native_tile_glimmer_dual_rms_add_capture_cooperative_batch_float32_v1");
    const auto plain_add = required_symbol<PlainAddFn>(
        library, "nfn_native_tile_add_float32");
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
    DeviceBuffer<float> device_residual(input.size());
    DeviceBuffer<float> device_baseline_added(input.size());
    DeviceBuffer<float> device_baseline_normalized(input.size());
    DeviceBuffer<float> device_baseline_captured(input.size());
    DeviceBuffer<float> device_dual_hidden(input.size());
    DeviceBuffer<float> device_dual_normalized(input.size());
    DeviceBuffer<float> device_cooperative_residual_input(input.size());
    DeviceBuffer<float> device_cooperative_hidden(input.size());
    DeviceBuffer<float> device_cooperative_normalized(input.size());
    DeviceBuffer<float> device_cooperative_captured(input.size());
    device_cooperative_residual_input.upload(input);
    constexpr std::int64_t q8_blocks = rows * width / 32;
    DeviceBuffer<std::int8_t> baseline_q8(input.size());
    DeviceBuffer<float> baseline_q8_scales(static_cast<std::size_t>(q8_blocks));
    DeviceBuffer<float> baseline_q8_sums(static_cast<std::size_t>(q8_blocks));
    DeviceBuffer<std::int8_t> fused_q8(input.size());
    DeviceBuffer<float> fused_q8_scales(static_cast<std::size_t>(q8_blocks));
    DeviceBuffer<float> fused_q8_sums(static_cast<std::size_t>(q8_blocks));
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
    cuda_check(static_cast<cudaError_t>(capture(
                   device_input.get(), nullptr, device_output.get(),
                   device_residual.get(), rows, width, epsilon, false, nullptr)),
               "glimmer fused RMSNorm capture launch");
    cuda_check(cudaDeviceSynchronize(), "glimmer fused RMSNorm capture synchronize");
    const auto captured = device_residual.download();
    const auto captured_output = device_output.download();
    for (std::size_t index = 0; index < input.size(); ++index) {
        require_close(captured[index], input[index], 0.0f,
                      "fused RMSNorm captured residual");
        require_close(captured_output[index], output[index], 0.0f,
                      "fused RMSNorm capture output");
    }
    cuda_check(static_cast<cudaError_t>(quantize(
                   device_output.get(), baseline_q8.get(),
                   baseline_q8_scales.get(), baseline_q8_sums.get(), rows,
                   width, nullptr)),
               "standalone normalized Q8 quantize launch");
    cuda_check(static_cast<cudaError_t>(capture_q8(
                   device_input.get(), nullptr, device_output.get(),
                   device_residual.get(), fused_q8.get(), fused_q8_scales.get(),
                   fused_q8_sums.get(), rows, width, epsilon, false, nullptr)),
               "fused RMSNorm Q8 capture launch");
    cuda_check(cudaDeviceSynchronize(), "fused RMSNorm Q8 capture synchronize");
    const auto fused_captured = device_residual.download();
    const auto fused_output = device_output.download();
    const auto baseline_codes = baseline_q8.download();
    const auto fused_codes = fused_q8.download();
    const auto baseline_scales = baseline_q8_scales.download();
    const auto fused_scales = fused_q8_scales.download();
    const auto baseline_sums = baseline_q8_sums.download();
    const auto fused_sums = fused_q8_sums.download();
    for (std::size_t index = 0; index < input.size(); ++index) {
        require_close(fused_captured[index], captured[index], 0.0f,
                      "fused RMSNorm Q8 residual");
        require_close(fused_output[index], captured_output[index], 0.0f,
                      "fused RMSNorm Q8 normalized output");
        if (fused_codes[index] != baseline_codes[index]) {
            throw std::runtime_error("fused RMSNorm Q8 code mismatch");
        }
    }
    for (std::size_t index = 0; index < baseline_scales.size(); ++index) {
        require_close(fused_scales[index], baseline_scales[index], 0.0f,
                      "fused RMSNorm Q8 scale");
        require_close(fused_sums[index], baseline_sums[index], 0.0f,
                      "fused RMSNorm Q8 sum");
    }
    cuda_check(static_cast<cudaError_t>(plain_add(
                   device_residual.get(), device_output.get(),
                   device_baseline_added.get(), rows * width, nullptr)),
               "baseline RMSNorm residual-add launch");
    cuda_check(cudaDeviceSynchronize(), "baseline RMSNorm residual-add synchronize");
    const auto baseline_added = device_baseline_added.download();
    cuda_check(static_cast<cudaError_t>(add(
                   device_input.get(), nullptr, device_residual.get(),
                   device_output.get(), rows, width, epsilon, false, nullptr)),
               "glimmer fused RMSNorm residual-add launch");
    cuda_check(cudaDeviceSynchronize(), "glimmer fused RMSNorm residual-add synchronize");
    const auto added = device_output.download();
    for (std::size_t index = 0; index < input.size(); ++index) {
        require_close(added[index], baseline_added[index], 0.0f,
                      "fused RMSNorm residual-add output");
    }
    constexpr float second_epsilon = 1.0e-5f;
    cuda_check(static_cast<cudaError_t>(capture(
                   device_output.get(), nullptr, device_baseline_normalized.get(),
                   device_baseline_captured.get(), rows, width, second_epsilon,
                   false, nullptr)),
               "baseline second RMSNorm capture launch");
    cuda_check(static_cast<cudaError_t>(dual(
                   device_input.get(), nullptr, device_residual.get(),
                   device_dual_hidden.get(), nullptr,
                   device_dual_normalized.get(), device_residual.get(), rows,
                   width, epsilon, false, second_epsilon, false, nullptr)),
               "dual RMSNorm add-capture launch");
    cuda_check(cudaDeviceSynchronize(),
               "dual RMSNorm add-capture synchronize");
    const auto baseline_normalized = device_baseline_normalized.download();
    const auto baseline_captured = device_baseline_captured.download();
    const auto dual_hidden = device_dual_hidden.download();
    const auto dual_normalized = device_dual_normalized.download();
    const auto dual_captured = device_residual.download();
    for (std::size_t index = 0; index < input.size(); ++index) {
        if (std::bit_cast<std::uint32_t>(dual_hidden[index]) !=
                std::bit_cast<std::uint32_t>(added[index]) ||
            std::bit_cast<std::uint32_t>(dual_captured[index]) !=
                std::bit_cast<std::uint32_t>(baseline_captured[index]) ||
            std::bit_cast<std::uint32_t>(dual_normalized[index]) !=
                std::bit_cast<std::uint32_t>(baseline_normalized[index])) {
            throw std::runtime_error(
                "dual RMSNorm add-capture is not bit-exact at " +
                std::to_string(index));
        }
    }
    cudaStream_t cooperative_stream = nullptr;
    cuda_check(cudaStreamCreate(&cooperative_stream),
               "cooperative dual RMS stream create");
    cuda_check(static_cast<cudaError_t>(cooperative_dual(
                   device_input.get(), nullptr,
                   device_cooperative_residual_input.get(),
                   device_cooperative_hidden.get(), nullptr,
                   device_cooperative_normalized.get(),
                   device_cooperative_captured.get(), rows, width, epsilon,
                   false, second_epsilon, false, cooperative_stream)),
               "cooperative dual RMSNorm add-capture launch");
    cuda_check(cudaStreamSynchronize(cooperative_stream),
               "cooperative dual RMSNorm synchronize");
    cuda_check(cudaStreamDestroy(cooperative_stream),
               "cooperative dual RMS stream destroy");
    const auto cooperative_hidden = device_cooperative_hidden.download();
    const auto cooperative_normalized =
        device_cooperative_normalized.download();
    const auto cooperative_captured = device_cooperative_captured.download();
    for (std::size_t index = 0; index < input.size(); ++index) {
        if (std::bit_cast<std::uint32_t>(cooperative_hidden[index]) !=
                std::bit_cast<std::uint32_t>(dual_hidden[index]) ||
            std::bit_cast<std::uint32_t>(cooperative_captured[index]) !=
                std::bit_cast<std::uint32_t>(dual_captured[index]) ||
            std::bit_cast<std::uint32_t>(cooperative_normalized[index]) !=
                std::bit_cast<std::uint32_t>(dual_normalized[index])) {
            throw std::runtime_error(
                "cooperative dual RMSNorm is not bit-exact at " +
                std::to_string(index));
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

void check_fused_qk_norm_scale_rope(void* library) {
    using RmsFn = int (*)(
        const float*, const NfnNativeTilePackedWeightDescriptorV1*, float*,
        std::int64_t, std::int64_t, float, bool, void*);
    using ScaleFn = int (*)(float*, std::int64_t, float, void*);
    using RopeFn = int (*)(float*, float*, std::int64_t, std::int64_t,
                           std::int64_t, std::int64_t, float, std::uint32_t,
                           void*);
    using FusedFn = int (*)(
        float*, float*, const NfnNativeTilePackedWeightDescriptorV1*,
        const NfnNativeTilePackedWeightDescriptorV1*, std::int64_t,
        std::int64_t, std::int64_t, float, bool, bool, float, std::int64_t,
        float, std::uint32_t, bool, void*);
    using BatchFn = int (*)(
        float*, float*, const NfnNativeTilePackedWeightDescriptorV1*,
        const NfnNativeTilePackedWeightDescriptorV1*, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, float, bool, bool, float,
        std::int64_t, float, std::uint32_t, bool, void*);
    const auto rms = required_symbol<RmsFn>(
        library, "nfn_native_tile_glimmer_rms_norm_affine_float32_v1");
    const auto scale = required_symbol<ScaleFn>(
        library, "nfn_native_tile_scale_inplace_float32");
    const auto rope = required_symbol<RopeFn>(
        library, "nfn_native_tile_glimmer_positioned_rope_float32_v1");
    const auto fused = required_symbol<FusedFn>(
        library, "nfn_native_tile_glimmer_qk_norm_scale_rope_float32_v1");
    const auto batch = required_symbol<BatchFn>(
        library,
        "nfn_native_tile_glimmer_qk_norm_scale_rope_batch_float32_v1");
    constexpr std::int64_t query_heads = 32;
    constexpr std::int64_t kv_heads = 2;
    constexpr std::int64_t head_dim = 128;
    constexpr std::int64_t query_width = query_heads * head_dim;
    constexpr std::int64_t key_width = kv_heads * head_dim;
    constexpr float epsilon = 1.0e-5f;
    constexpr float query_scale = 3.87f;
    constexpr std::int64_t position = 2051;
    constexpr float theta = 500000.0f;
    std::vector<float> query(static_cast<std::size_t>(query_width));
    std::vector<float> key(static_cast<std::size_t>(key_width));
    for (std::size_t index = 0; index < query.size(); ++index) {
        query[index] = static_cast<float>(
            static_cast<int>((index * 31 + 11) % 257) - 128) / 73.0f;
    }
    for (std::size_t index = 0; index < key.size(); ++index) {
        key[index] = static_cast<float>(
            static_cast<int>((index * 23 + 5) % 193) - 96) / 61.0f;
    }
    DeviceBuffer<float> baseline_query(query.size());
    DeviceBuffer<float> baseline_key(key.size());
    DeviceBuffer<float> fused_query(query.size());
    DeviceBuffer<float> fused_key(key.size());
    const auto require_exact = [](const std::vector<float>& expected,
                                  const std::vector<float>& actual,
                                  const std::string& label) {
        for (std::size_t index = 0; index < expected.size(); ++index) {
            if (std::bit_cast<std::uint32_t>(expected[index]) !=
                std::bit_cast<std::uint32_t>(actual[index])) {
                throw std::runtime_error(
                    label + " is not bit-exact at " + std::to_string(index));
            }
        }
    };
    for (bool apply_rope : {false, true}) {
        baseline_query.upload(query);
        baseline_key.upload(key);
        fused_query.upload(query);
        fused_key.upload(key);
        cuda_check(static_cast<cudaError_t>(rms(
                       baseline_query.get(), nullptr, baseline_query.get(),
                       query_heads, head_dim, epsilon, false, nullptr)),
                   "baseline Q head norm launch");
        cuda_check(static_cast<cudaError_t>(rms(
                       baseline_key.get(), nullptr, baseline_key.get(),
                       kv_heads, head_dim, epsilon, false, nullptr)),
                   "baseline K head norm launch");
        cuda_check(static_cast<cudaError_t>(scale(
                       baseline_query.get(), query_width, query_scale, nullptr)),
                   "baseline query scale launch");
        if (apply_rope) {
            cuda_check(static_cast<cudaError_t>(rope(
                           baseline_query.get(), baseline_key.get(), query_heads,
                           kv_heads, head_dim, position, theta,
                           NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT, nullptr)),
                       "baseline QK RoPE launch");
        }
        cuda_check(static_cast<cudaError_t>(fused(
                       fused_query.get(), fused_key.get(), nullptr, nullptr,
                       query_heads, kv_heads, head_dim, epsilon, false, false,
                       query_scale, position, theta,
                       NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT, apply_rope,
                       nullptr)),
                   "fused QK norm/scale/RoPE launch");
        cuda_check(cudaDeviceSynchronize(),
                   "fused QK norm/scale/RoPE synchronize");
        require_exact(
            baseline_query.download(), fused_query.download(),
            apply_rope ? "fused local query" : "fused global query");
        require_exact(
            baseline_key.download(), fused_key.download(),
            apply_rope ? "fused local key" : "fused global key");
    }

    constexpr std::int64_t rows = 3;
    std::vector<float> batch_query(static_cast<std::size_t>(rows * query_width));
    std::vector<float> batch_key(static_cast<std::size_t>(rows * key_width));
    for (std::size_t index = 0; index < batch_query.size(); ++index) {
        batch_query[index] = static_cast<float>(
            static_cast<int>((index * 37 + 19) % 263) - 131) / 79.0f;
    }
    for (std::size_t index = 0; index < batch_key.size(); ++index) {
        batch_key[index] = static_cast<float>(
            static_cast<int>((index * 29 + 7) % 199) - 99) / 67.0f;
    }
    DeviceBuffer<float> rowwise_query(batch_query.size());
    DeviceBuffer<float> rowwise_key(batch_key.size());
    DeviceBuffer<float> fused_batch_query(batch_query.size());
    DeviceBuffer<float> fused_batch_key(batch_key.size());
    for (bool apply_rope : {false, true}) {
        rowwise_query.upload(batch_query);
        rowwise_key.upload(batch_key);
        fused_batch_query.upload(batch_query);
        fused_batch_key.upload(batch_key);
        for (std::int64_t row = 0; row < rows; ++row) {
            cuda_check(static_cast<cudaError_t>(fused(
                           rowwise_query.get() + row * query_width,
                           rowwise_key.get() + row * key_width, nullptr, nullptr,
                           query_heads, kv_heads, head_dim, epsilon, false, false,
                           query_scale, position + row, theta,
                           NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT, apply_rope,
                           nullptr)),
                       "row-wise QK norm/scale/RoPE launch");
        }
        cuda_check(static_cast<cudaError_t>(batch(
                       fused_batch_query.get(), fused_batch_key.get(), nullptr,
                       nullptr, rows, query_heads, kv_heads, head_dim, epsilon,
                       false, false, query_scale, position, theta,
                       NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT, apply_rope,
                       nullptr)),
                   "batched QK norm/scale/RoPE launch");
        cuda_check(cudaDeviceSynchronize(),
                   "batched QK norm/scale/RoPE synchronize");
        require_exact(
            rowwise_query.download(), fused_batch_query.download(),
            apply_rope ? "batched local query" : "batched global query");
        require_exact(
            rowwise_key.download(), fused_batch_key.download(),
            apply_rope ? "batched local key" : "batched global key");
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

void check_packed_k_multilinear(
    void* library,
    std::uint32_t encoding,
    std::int64_t block_bytes,
    const std::string& label) {
    using ValidateFn = int (*)(const NfnNativeTilePackedWeightDescriptorV1*);
    using DequantFn = int (*)(const NfnNativeTilePackedWeightDescriptorV1*, float*);
    using LinearFn = int (*)(const NfnNativeTilePackedWeightDescriptorV1*,
                             const float*, const float*, float*, std::int64_t, bool);
    const auto validate = required_symbol<ValidateFn>(
        library, "nfn_native_tile_packed_weight_validate_v1");
    const auto dequantize = required_symbol<DequantFn>(
        library, "nfn_native_tile_packed_weight_dequantize_float32_v1");
    const auto linear = required_symbol<LinearFn>(
        library, "nfn_native_tile_linear_packed_weight_float32_v1");
    constexpr std::int64_t rows = 5;
    constexpr std::int64_t output_dim = 5;
    constexpr std::int64_t input_dim = 512;
    constexpr std::int64_t blocks_per_row = input_dim / 256;
    const std::int64_t row_stride = blocks_per_row * block_bytes;
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(
        output_dim * row_stride));
    for (std::size_t index = 0; index < packed.size(); ++index) {
        packed[index] = static_cast<std::uint8_t>((index * 29 + 17) & 0xff);
    }
    for (std::int64_t row = 0; row < output_dim; ++row) {
        for (std::int64_t block_index = 0; block_index < blocks_per_row;
             ++block_index) {
            const std::size_t offset = static_cast<std::size_t>(
                row * row_stride + block_index * block_bytes);
            if (encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K ||
                encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K) {
                packed[offset] = 0x00;
                packed[offset + 1] = 0x28;  // fp16 0.03125
                packed[offset + 2] = 0x00;
                packed[offset + 3] = 0x24;  // fp16 0.015625
                for (std::size_t scale = 0; scale < 12; ++scale) {
                    packed[offset + 4 + scale] =
                        static_cast<std::uint8_t>(1 + (scale % 4));
                }
            } else {
                for (std::size_t scale = 0; scale < 16; ++scale) {
                    packed[offset + 192 + scale] = static_cast<std::uint8_t>(
                        static_cast<std::int8_t>(static_cast<int>(scale % 9) - 4));
                }
                packed[offset + 208] = 0x00;
                packed[offset + 209] = 0x28;  // fp16 0.03125
            }
        }
    }
    std::vector<float> input(static_cast<std::size_t>(rows * input_dim));
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t column = 0; column < input_dim; ++column) {
            input[static_cast<std::size_t>(row * input_dim + column)] =
                static_cast<float>((row * 37 + column * 13) % 257 - 128) /
                257.0f;
        }
    }
    std::vector<float> bias(static_cast<std::size_t>(output_dim));
    for (std::int64_t column = 0; column < output_dim; ++column) {
        bias[static_cast<std::size_t>(column)] =
            static_cast<float>(column - 2) / 7.0f;
    }
    DeviceBuffer<std::uint8_t> device_packed(packed.size());
    DeviceBuffer<float> device_dequantized(
        static_cast<std::size_t>(output_dim * input_dim));
    DeviceBuffer<float> device_input(input.size());
    DeviceBuffer<float> device_bias(bias.size());
    DeviceBuffer<float> device_output(
        static_cast<std::size_t>(rows * output_dim));
    device_packed.upload(packed);
    device_input.upload(input);
    device_bias.upload(bias);
    NfnNativeTilePackedWeightDescriptorV1 descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.version = NFN_NATIVE_TILE_PACKED_WEIGHT_V1;
    descriptor.encoding = encoding;
    descriptor.data = device_packed.get();
    descriptor.data_nbytes = static_cast<std::int64_t>(packed.size());
    descriptor.output_dim = output_dim;
    descriptor.input_dim = input_dim;
    descriptor.row_stride_bytes = row_stride;
    cuda_check(static_cast<cudaError_t>(validate(&descriptor)),
               (label + " descriptor validation").c_str());
    cuda_check(static_cast<cudaError_t>(dequantize(
                   &descriptor, device_dequantized.get())),
               (label + " dequantize launch").c_str());
    cuda_check(static_cast<cudaError_t>(linear(
                   &descriptor, device_input.get(), device_bias.get(),
                   device_output.get(), rows, true)),
               (label + " multilinear launch").c_str());
    cuda_check(cudaDeviceSynchronize(),
               (label + " multilinear synchronize").c_str());
    const auto dequantized = device_dequantized.download();
    const auto output = device_output.download();
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t output_column = 0; output_column < output_dim;
             ++output_column) {
            double expected = bias[static_cast<std::size_t>(output_column)];
            for (std::int64_t input_column = 0; input_column < input_dim;
                 ++input_column) {
                expected += static_cast<double>(input[static_cast<std::size_t>(
                    row * input_dim + input_column)]) *
                    dequantized[static_cast<std::size_t>(
                        output_column * input_dim + input_column)];
            }
            require_close(
                output[static_cast<std::size_t>(row * output_dim + output_column)],
                static_cast<float>(expected), 5.0e-3f,
                label + " rows=5 packed linear");
        }
    }
}

void check_q8_activation_packed_linear(
    void* library,
    std::uint32_t encoding,
    std::int64_t block_bytes,
    std::int64_t rows,
    std::int64_t output_dim,
    std::int64_t input_dim,
    const std::string& label) {
    using DequantFn = int (*)(const NfnNativeTilePackedWeightDescriptorV1*, float*);
    using QuantizeFn = int (*)(const float*, std::int8_t*, float*, float*,
                               std::int64_t, std::int64_t, void*);
    using LinearFn = int (*)(const NfnNativeTilePackedWeightDescriptorV1*,
                             const std::int8_t*, const float*, const float*,
                             const float*, float*, std::int64_t, bool);
    const auto dequantize = required_symbol<DequantFn>(
        library, "nfn_native_tile_packed_weight_dequantize_float32_v1");
    const auto quantize = required_symbol<QuantizeFn>(
        library, "nfn_native_tile_quantize_q8_1_float32_v1");
    const auto linear = required_symbol<LinearFn>(
        library, "nfn_native_tile_linear_packed_weight_q8_1_float32_v1");
    if (rows <= 0 || output_dim <= 0 || input_dim <= 0 || input_dim % 256 != 0) {
        throw std::runtime_error(label + " Q8 test geometry is invalid");
    }
    const std::int64_t blocks_per_row = input_dim / 256;
    const std::int64_t q8_blocks_per_row = input_dim / 32;
    const std::int64_t row_stride = blocks_per_row * block_bytes;
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(
        output_dim * row_stride));
    for (std::size_t index = 0; index < packed.size(); ++index) {
        packed[index] = static_cast<std::uint8_t>((index * 41 + 23) & 0xff);
    }
    for (std::int64_t row = 0; row < output_dim; ++row) {
        for (std::int64_t block_index = 0; block_index < blocks_per_row;
             ++block_index) {
            const std::size_t offset = static_cast<std::size_t>(
                row * row_stride + block_index * block_bytes);
            if (encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K ||
                encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K) {
                packed[offset] = 0x00;
                packed[offset + 1] = 0x28;  // fp16 0.03125
                packed[offset + 2] = 0x00;
                packed[offset + 3] = 0x24;  // fp16 0.015625 exercises Q8 sum/min
                for (std::size_t scale = 0; scale < 12; ++scale) {
                    packed[offset + 4 + scale] =
                        static_cast<std::uint8_t>(1 + ((scale + row) % 4));
                }
            } else {
                for (std::size_t scale = 0; scale < 16; ++scale) {
                    packed[offset + 192 + scale] = static_cast<std::uint8_t>(
                        static_cast<std::int8_t>(
                            static_cast<int>((scale + row) % 11) - 5));
                }
                packed[offset + 208] = 0x00;
                packed[offset + 209] = 0x28;  // fp16 0.03125
            }
        }
    }
    // Every 32-wide block has an exact Q8 scale of 1/32.  Paired codes cancel,
    // leaving a small, exactly-half-representable nonzero sum so Q4_K/Q5_K's
    // half2 d/s path and minimum correction are both exercised.
    std::vector<float> input(static_cast<std::size_t>(rows * input_dim));
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t block = 0; block < q8_blocks_per_row; ++block) {
            for (std::int64_t lane = 0; lane < 32; ++lane) {
                int code = 0;
                if (lane == 0) code = -127;
                else if (lane == 1) code = 127;
                else if (lane == 2) code = 64 - 2 * static_cast<int>(row % 5);
                else if (lane >= 4) {
                    const int magnitude = 1 + static_cast<int>(
                        (block * 11 + (lane / 2) * 7 + row * 3) % 96);
                    code = (lane & 1) == 0 ? magnitude : -magnitude;
                }
                input[static_cast<std::size_t>(
                    row * input_dim + block * 32 + lane)] =
                    static_cast<float>(code) * 0.03125f;
            }
        }
    }
    std::vector<float> bias(static_cast<std::size_t>(output_dim));
    for (std::int64_t column = 0; column < output_dim; ++column) {
        bias[static_cast<std::size_t>(column)] =
            static_cast<float>(column - 3) / 9.0f;
    }
    DeviceBuffer<std::uint8_t> device_packed(packed.size());
    DeviceBuffer<float> device_dequantized(
        static_cast<std::size_t>(output_dim * input_dim));
    DeviceBuffer<float> device_input(input.size());
    DeviceBuffer<std::int8_t> device_q8(input.size());
    DeviceBuffer<float> device_scales(static_cast<std::size_t>(
        rows * q8_blocks_per_row));
    DeviceBuffer<float> device_sums(static_cast<std::size_t>(
        rows * q8_blocks_per_row));
    DeviceBuffer<float> device_bias(bias.size());
    DeviceBuffer<float> device_output(static_cast<std::size_t>(rows * output_dim));
    device_packed.upload(packed);
    device_input.upload(input);
    device_bias.upload(bias);
    NfnNativeTilePackedWeightDescriptorV1 descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.version = NFN_NATIVE_TILE_PACKED_WEIGHT_V1;
    descriptor.encoding = encoding;
    descriptor.data = device_packed.get();
    descriptor.data_nbytes = static_cast<std::int64_t>(packed.size());
    descriptor.output_dim = output_dim;
    descriptor.input_dim = input_dim;
    descriptor.row_stride_bytes = row_stride;
    cuda_check(static_cast<cudaError_t>(dequantize(
                   &descriptor, device_dequantized.get())),
               (label + " Q8 oracle dequantize launch").c_str());
    cuda_check(static_cast<cudaError_t>(quantize(
                   device_input.get(), device_q8.get(), device_scales.get(),
                   device_sums.get(), rows, input_dim, nullptr)),
               (label + " Q8 activation quantize launch").c_str());
    cuda_check(static_cast<cudaError_t>(linear(
                   &descriptor, device_q8.get(), device_scales.get(),
                   device_sums.get(), device_bias.get(), device_output.get(),
                   rows, true)),
               (label + " Q8 packed linear launch").c_str());
    cuda_check(cudaDeviceSynchronize(),
               (label + " Q8 packed linear synchronize").c_str());
    const auto dequantized = device_dequantized.download();
    const auto q8 = device_q8.download();
    const auto scales = device_scales.download();
    const auto sums = device_sums.download();
    const auto output = device_output.download();
    std::vector<float> reconstructed(input.size());
    for (std::int64_t block = 0; block < rows * q8_blocks_per_row; ++block) {
        float maximum = 0.0f;
        float expected_sum = 0.0f;
        for (std::int64_t lane = 0; lane < 32; ++lane) {
            const float value = input[static_cast<std::size_t>(block * 32 + lane)];
            maximum = std::max(maximum, std::abs(value));
            expected_sum += value;
        }
        const float expected_inverse_scale =
            maximum > 0.0f ? 127.0f / maximum : 0.0f;
        const float expected_scale = expected_inverse_scale > 0.0f
            ? 1.0f / expected_inverse_scale
            : 0.0f;
        require_close(scales[static_cast<std::size_t>(block)], expected_scale,
                      1.0e-7f, label + " Q8 scale");
        require_close(sums[static_cast<std::size_t>(block)], expected_sum,
                      2.0e-5f, label + " Q8 sum");
        for (std::int64_t lane = 0; lane < 32; ++lane) {
            const std::size_t index = static_cast<std::size_t>(block * 32 + lane);
            const int expected_quantized = expected_inverse_scale > 0.0f
                ? std::clamp(static_cast<int>(std::round(
                                 input[index] * expected_inverse_scale)),
                             -127, 127)
                : 0;
            if (static_cast<int>(q8[index]) != expected_quantized) {
                throw std::runtime_error(
                    label + " Q8 activation code mismatch at " +
                    std::to_string(index) + ": actual=" +
                    std::to_string(static_cast<int>(q8[index])) + " expected=" +
                    std::to_string(expected_quantized));
            }
            const float linear_scale =
                encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q6_K
                ? expected_scale
                : __half2float(__float2half_rn(expected_scale));
            reconstructed[index] =
                linear_scale * static_cast<float>(expected_quantized);
        }
    }
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t output_column = 0; output_column < output_dim;
             ++output_column) {
            double expected = bias[static_cast<std::size_t>(output_column)];
            for (std::int64_t input_column = 0; input_column < input_dim;
                 ++input_column) {
                expected += static_cast<double>(reconstructed[static_cast<std::size_t>(
                    row * input_dim + input_column)]) * dequantized[static_cast<std::size_t>(
                        output_column * input_dim + input_column)];
            }
            require_close(output[static_cast<std::size_t>(
                              row * output_dim + output_column)],
                          static_cast<float>(expected), 8.0e-3f,
                          label + " batched Q8 packed linear");
        }
    }
}

void check_q8_multi_decode(void* library) {
    using QuantizeFn = int (*)(const float*, std::int8_t*, float*, float*,
                               std::int64_t, std::int64_t, void*);
    using LinearFn = int (*)(const NfnNativeTilePackedWeightDescriptorV1*,
                             const std::int8_t*, const float*, const float*,
                             const float*, float*, std::int64_t, bool);
    using MultiFn = int (*)(
        const NfnNativeTilePackedWeightDescriptorV1*,
        const NfnNativeTilePackedWeightDescriptorV1*,
        const NfnNativeTilePackedWeightDescriptorV1*,
        const NfnNativeTilePackedWeightDescriptorV1*, const std::int8_t*,
        const float*, const float*, float*, float*, float*, float*,
        std::int64_t, void*);
    const auto quantize = required_symbol<QuantizeFn>(
        library, "nfn_native_tile_quantize_q8_1_float32_v1");
    const auto linear = required_symbol<LinearFn>(
        library, "nfn_native_tile_linear_packed_weight_q8_1_float32_v1");
    const auto multi = required_symbol<MultiFn>(
        library,
        "nfn_native_tile_linear_packed_weight_q8_1_multi_decode_float32_v1");

    constexpr std::int64_t input_dim = 512;
    constexpr std::int64_t output_dim0 = 5;
    constexpr std::int64_t output_dim1 = 7;
    constexpr std::int64_t output_dim2 = 3;
    constexpr std::int64_t output_dim3 = 9;
    constexpr std::int64_t q8_blocks = input_dim / 32;
    const auto make_packed = [](std::uint32_t encoding,
                                std::int64_t block_bytes,
                                std::int64_t output_dim,
                                int seed) {
        const std::int64_t blocks_per_row = input_dim / 256;
        std::vector<std::uint8_t> packed(static_cast<std::size_t>(
            output_dim * blocks_per_row * block_bytes));
        for (std::size_t index = 0; index < packed.size(); ++index) {
            packed[index] = static_cast<std::uint8_t>(
                (index * static_cast<std::size_t>(seed * 2 + 17) + seed) & 0xff);
        }
        for (std::int64_t row = 0; row < output_dim; ++row) {
            for (std::int64_t block_index = 0; block_index < blocks_per_row;
                 ++block_index) {
                const std::size_t offset = static_cast<std::size_t>(
                    (row * blocks_per_row + block_index) * block_bytes);
                if (encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K ||
                    encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K) {
                    packed[offset] = 0x00;
                    packed[offset + 1] = 0x28;
                    packed[offset + 2] = 0x00;
                    packed[offset + 3] = 0x24;
                    for (std::size_t scale = 0; scale < 12; ++scale) {
                        packed[offset + 4 + scale] = static_cast<std::uint8_t>(
                            1 + ((scale + static_cast<std::size_t>(row) + seed) % 4));
                    }
                } else {
                    for (std::size_t scale = 0; scale < 16; ++scale) {
                        packed[offset + 192 + scale] = static_cast<std::uint8_t>(
                            static_cast<std::int8_t>(
                                static_cast<int>((scale + row + seed) % 11) - 5));
                    }
                    packed[offset + 208] = 0x00;
                    packed[offset + 209] = 0x28;
                }
            }
        }
        return packed;
    };
    auto packed0 = make_packed(
        NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K, 144, output_dim0, 3);
    auto packed1 = make_packed(
        NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K, 176, output_dim1, 5);
    auto packed2 = make_packed(
        NFN_NATIVE_TILE_PACKED_WEIGHT_Q6_K, 210, output_dim2, 7);
    auto packed3 = make_packed(
        NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K, 144, output_dim3, 11);
    DeviceBuffer<std::uint8_t> device_packed0(packed0.size());
    DeviceBuffer<std::uint8_t> device_packed1(packed1.size());
    DeviceBuffer<std::uint8_t> device_packed2(packed2.size());
    DeviceBuffer<std::uint8_t> device_packed3(packed3.size());
    device_packed0.upload(packed0);
    device_packed1.upload(packed1);
    device_packed2.upload(packed2);
    device_packed3.upload(packed3);
    const auto descriptor = [](std::uint32_t encoding, const std::uint8_t* data,
                               std::int64_t bytes, std::int64_t output_dim,
                               std::int64_t row_stride) {
        NfnNativeTilePackedWeightDescriptorV1 value{};
        value.struct_size = sizeof(value);
        value.version = NFN_NATIVE_TILE_PACKED_WEIGHT_V1;
        value.encoding = encoding;
        value.data = data;
        value.data_nbytes = bytes;
        value.output_dim = output_dim;
        value.input_dim = input_dim;
        value.row_stride_bytes = row_stride;
        return value;
    };
    const auto descriptor0 = descriptor(
        NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K, device_packed0.get(),
        static_cast<std::int64_t>(packed0.size()), output_dim0, 2 * 144);
    const auto descriptor1 = descriptor(
        NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K, device_packed1.get(),
        static_cast<std::int64_t>(packed1.size()), output_dim1, 2 * 176);
    const auto descriptor2 = descriptor(
        NFN_NATIVE_TILE_PACKED_WEIGHT_Q6_K, device_packed2.get(),
        static_cast<std::int64_t>(packed2.size()), output_dim2, 2 * 210);
    const auto descriptor3 = descriptor(
        NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K, device_packed3.get(),
        static_cast<std::int64_t>(packed3.size()), output_dim3, 2 * 144);

    std::vector<float> input(static_cast<std::size_t>(input_dim));
    for (std::int64_t index = 0; index < input_dim; ++index) {
        input[static_cast<std::size_t>(index)] =
            static_cast<float>(static_cast<int>((index * 19 + 7) % 251) - 125) /
            47.0f;
    }
    DeviceBuffer<float> device_input(input.size());
    DeviceBuffer<std::int8_t> device_q8(input.size());
    DeviceBuffer<float> device_scales(static_cast<std::size_t>(q8_blocks));
    DeviceBuffer<float> device_sums(static_cast<std::size_t>(q8_blocks));
    DeviceBuffer<float> baseline0(static_cast<std::size_t>(output_dim0));
    DeviceBuffer<float> baseline1(static_cast<std::size_t>(output_dim1));
    DeviceBuffer<float> baseline2(static_cast<std::size_t>(output_dim2));
    DeviceBuffer<float> baseline3(static_cast<std::size_t>(output_dim3));
    DeviceBuffer<float> fused0(static_cast<std::size_t>(output_dim0));
    DeviceBuffer<float> fused1(static_cast<std::size_t>(output_dim1));
    DeviceBuffer<float> fused2(static_cast<std::size_t>(output_dim2));
    DeviceBuffer<float> fused3(static_cast<std::size_t>(output_dim3));
    device_input.upload(input);
    cuda_check(static_cast<cudaError_t>(quantize(
                   device_input.get(), device_q8.get(), device_scales.get(),
                   device_sums.get(), 1, input_dim, nullptr)),
               "multi-decode Q8 quantize launch");
    const auto run_single = [&](const NfnNativeTilePackedWeightDescriptorV1& value,
                                float* output) {
        cuda_check(static_cast<cudaError_t>(linear(
                       &value, device_q8.get(), device_scales.get(),
                       device_sums.get(), nullptr, output, 1, false)),
                   "multi-decode single projection launch");
    };
    run_single(descriptor0, baseline0.get());
    run_single(descriptor1, baseline1.get());
    run_single(descriptor2, baseline2.get());
    run_single(descriptor3, baseline3.get());
    cuda_check(static_cast<cudaError_t>(multi(
                   &descriptor0, &descriptor1, &descriptor2, &descriptor3,
                   device_q8.get(), device_scales.get(), device_sums.get(),
                   fused0.get(), fused1.get(), fused2.get(), fused3.get(), 4,
                   nullptr)),
               "four-projection multi-decode launch");
    cuda_check(cudaDeviceSynchronize(), "four-projection multi-decode synchronize");
    const auto require_exact = [](const std::vector<float>& expected,
                                  const std::vector<float>& actual,
                                  const std::string& label) {
        if (actual.size() != expected.size()) {
            throw std::runtime_error(label + " output size mismatch");
        }
        for (std::size_t index = 0; index < actual.size(); ++index) {
            if (std::bit_cast<std::uint32_t>(actual[index]) !=
                std::bit_cast<std::uint32_t>(expected[index])) {
                throw std::runtime_error(
                    label + " is not bit-exact at " + std::to_string(index));
            }
        }
    };
    require_exact(baseline0.download(), fused0.download(), "projection 0");
    require_exact(baseline1.download(), fused1.download(), "projection 1");
    require_exact(baseline2.download(), fused2.download(), "projection 2");
    require_exact(baseline3.download(), fused3.download(), "projection 3");
    cuda_check(static_cast<cudaError_t>(multi(
                   &descriptor0, &descriptor1, nullptr, nullptr,
                   device_q8.get(), device_scales.get(), device_sums.get(),
                   fused0.get(), fused1.get(), nullptr, nullptr, 2, nullptr)),
               "two-projection multi-decode launch");
    cuda_check(cudaDeviceSynchronize(), "two-projection multi-decode synchronize");
    require_exact(baseline0.download(), fused0.download(), "two-projection 0");
    require_exact(baseline1.download(), fused1.download(), "two-projection 1");
}

void check_exact_k_quant_mmq(void* library) {
    using AbiFn = int (*)();
    using WorkspaceFn = std::int64_t (*)(std::int64_t, std::int64_t);
    using MmqFn = int (*)(
        const NfnNativeTilePackedWeightDescriptorV1* const*, const float*,
        float* const*, std::int64_t, std::int64_t, void*, std::int64_t, void*);
    using GatedMmqFn = int (*)(
        const NfnNativeTilePackedWeightDescriptorV1* const*, const float*,
        const float*, float* const*, std::int64_t, std::int64_t, void*,
        std::int64_t, void*);
    using SwiGluMmqFn = GatedMmqFn;
    using MmvqFn = int (*)(
        const NfnNativeTilePackedWeightDescriptorV1*, const float*, float*,
        void*, std::int64_t, void*);
    using PrequantizedMmvqFn = int (*)(
        const NfnNativeTilePackedWeightDescriptorV1* const*, float* const*,
        std::int64_t, std::int64_t, void*, std::int64_t, void*);
    using DualFn = int (*)(
        const float*, const NfnNativeTilePackedWeightDescriptorV1*, const float*,
        float*, const NfnNativeTilePackedWeightDescriptorV1*, float*, float*,
        std::int64_t, std::int64_t, float, bool, float, bool, void*);
    using DualHandoffFn = int (*)(
        const float*, const NfnNativeTilePackedWeightDescriptorV1*, const float*,
        float*, const NfnNativeTilePackedWeightDescriptorV1*, float*, float*,
        std::int64_t, std::int64_t, float, bool, float, bool, void*,
        std::int64_t, void*);
    using GateFn = int (*)(
        const float*, const float*, float*, std::int64_t, void*);
    using QuantizeFn = int (*)(const float*, std::int8_t*, float*, float*,
                               std::int64_t, std::int64_t, void*);
    using LinearFn = int (*)(const NfnNativeTilePackedWeightDescriptorV1*,
                             const std::int8_t*, const float*, const float*,
                             const float*, float*, std::int64_t, bool);
    const auto abi = required_symbol<AbiFn>(
        library, "nfn_native_tile_k_quant_mmq_abi_version");
    const auto workspace_bytes = required_symbol<WorkspaceFn>(
        library, "nfn_native_tile_k_quant_mmq_workspace_bytes_v1");
    const auto mmq = required_symbol<MmqFn>(
        library, "nfn_native_tile_k_quant_mmq_multi_linear_float32_v1");
    const auto gated_mmq = required_symbol<GatedMmqFn>(
        library,
        "nfn_native_tile_k_quant_mmq_multi_linear_gated_float32_v1");
    const auto swiglu_mmq = required_symbol<SwiGluMmqFn>(
        library,
        "nfn_native_tile_k_quant_mmq_multi_linear_swiglu_float32_v1");
    const auto batched_mmvq = required_symbol<MmqFn>(
        library,
        "nfn_native_tile_k_quant_mmvq_multi_linear_float32_v1");
    const auto prequantized_mmvq = required_symbol<PrequantizedMmvqFn>(
        library,
        "nfn_native_tile_k_quant_mmvq_multi_linear_prequantized_float32_v1");
    const auto batched_gated_mmvq = required_symbol<GatedMmqFn>(
        library,
        "nfn_native_tile_k_quant_mmvq_multi_linear_gated_float32_v1");
    const auto batched_swiglu_mmvq = required_symbol<SwiGluMmqFn>(
        library,
        "nfn_native_tile_k_quant_mmvq_multi_linear_swiglu_float32_v1");
    const auto mmvq = required_symbol<MmvqFn>(
        library, "nfn_native_tile_k_quant_mmvq_linear_float32_v1");
    const auto dual = required_symbol<DualFn>(
        library,
        "nfn_native_tile_glimmer_dual_rms_add_capture_float32_v1");
    const auto dual_handoff = required_symbol<DualHandoffFn>(
        library,
        "nfn_native_tile_glimmer_dual_rms_add_capture_mmvq_q8_float32_v1");
    const auto gate_op = required_symbol<GateFn>(
        library, "nfn_native_tile_glimmer_sigmoid_gate_float32_v1");
    const auto swiglu_op = required_symbol<GateFn>(
        library, "nfn_native_tile_swiglu_float32");
    const auto quantize = required_symbol<QuantizeFn>(
        library, "nfn_native_tile_quantize_q8_1_float32_v1");
    const auto linear = required_symbol<LinearFn>(
        library, "nfn_native_tile_linear_packed_weight_q8_1_float32_v1");
    if (abi() != NFN_NATIVE_TILE_K_QUANT_MMQ_V1) {
        throw std::runtime_error("exact K-quant MMQ ABI mismatch");
    }

    constexpr std::int64_t rows = 5;
    constexpr std::int64_t input_dim = 512;
    constexpr std::int64_t output_dim0 = 129;
    constexpr std::int64_t output_dim1 = 17;
    constexpr std::int64_t output_dim2 = 33;
    constexpr std::int64_t q8_blocks = rows * input_dim / 32;
    const auto make_packed = [](std::uint32_t encoding,
                                std::int64_t block_bytes,
                                std::int64_t output_dim,
                                int seed) {
        constexpr std::int64_t blocks_per_row = input_dim / 256;
        std::vector<std::uint8_t> packed(static_cast<std::size_t>(
            output_dim * blocks_per_row * block_bytes));
        for (std::size_t index = 0; index < packed.size(); ++index) {
            packed[index] = static_cast<std::uint8_t>(
                (index * static_cast<std::size_t>(seed * 2 + 17) + seed) & 0xff);
        }
        for (std::int64_t row = 0; row < output_dim; ++row) {
            for (std::int64_t block = 0; block < blocks_per_row; ++block) {
                const std::size_t offset = static_cast<std::size_t>(
                    (row * blocks_per_row + block) * block_bytes);
                if (encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K ||
                    encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K) {
                    packed[offset] = 0x00;
                    packed[offset + 1] = 0x28;
                    packed[offset + 2] = 0x00;
                    packed[offset + 3] = 0x24;
                    for (std::size_t scale = 0; scale < 12; ++scale) {
                        packed[offset + 4 + scale] = static_cast<std::uint8_t>(
                            1 + ((scale + static_cast<std::size_t>(row) + seed) % 4));
                    }
                } else {
                    for (std::size_t scale = 0; scale < 16; ++scale) {
                        packed[offset + 192 + scale] = static_cast<std::uint8_t>(
                            static_cast<std::int8_t>(
                                static_cast<int>((scale + row + seed) % 11) - 5));
                    }
                    packed[offset + 208] = 0x00;
                    packed[offset + 209] = 0x28;
                }
            }
        }
        return packed;
    };
    auto packed0 = make_packed(
        NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K, 144, output_dim0, 3);
    auto packed1 = make_packed(
        NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K, 176, output_dim1, 5);
    auto packed2 = make_packed(
        NFN_NATIVE_TILE_PACKED_WEIGHT_Q6_K, 210, output_dim2, 7);
    DeviceBuffer<std::uint8_t> device_packed0(packed0.size());
    DeviceBuffer<std::uint8_t> device_packed1(packed1.size());
    DeviceBuffer<std::uint8_t> device_packed2(packed2.size());
    device_packed0.upload(packed0);
    device_packed1.upload(packed1);
    device_packed2.upload(packed2);
    const auto descriptor = [](std::uint32_t encoding, const std::uint8_t* data,
                               std::int64_t nbytes, std::int64_t output_dim,
                               std::int64_t row_stride) {
        NfnNativeTilePackedWeightDescriptorV1 value{};
        value.struct_size = sizeof(value);
        value.version = NFN_NATIVE_TILE_PACKED_WEIGHT_V1;
        value.encoding = encoding;
        value.data = data;
        value.data_nbytes = nbytes;
        value.output_dim = output_dim;
        value.input_dim = input_dim;
        value.row_stride_bytes = row_stride;
        return value;
    };
    const auto descriptor0 = descriptor(
        NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K, device_packed0.get(),
        static_cast<std::int64_t>(packed0.size()), output_dim0, 2 * 144);
    const auto descriptor1 = descriptor(
        NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K, device_packed1.get(),
        static_cast<std::int64_t>(packed1.size()), output_dim1, 2 * 176);
    const auto descriptor2 = descriptor(
        NFN_NATIVE_TILE_PACKED_WEIGHT_Q6_K, device_packed2.get(),
        static_cast<std::int64_t>(packed2.size()), output_dim2, 2 * 210);

    std::vector<float> input(static_cast<std::size_t>(rows * input_dim));
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t column = 0; column < input_dim; ++column) {
            input[static_cast<std::size_t>(row * input_dim + column)] =
                static_cast<float>(
                    static_cast<int>((column * 19 + row * 23 + 7) % 251) - 125) /
                47.0f;
        }
    }
    DeviceBuffer<float> device_input(input.size());
    DeviceBuffer<std::int8_t> device_q8(input.size());
    DeviceBuffer<float> device_scales(static_cast<std::size_t>(q8_blocks));
    DeviceBuffer<float> device_sums(static_cast<std::size_t>(q8_blocks));
    DeviceBuffer<float> baseline0(static_cast<std::size_t>(rows * output_dim0));
    DeviceBuffer<float> baseline1(static_cast<std::size_t>(rows * output_dim1));
    DeviceBuffer<float> baseline2(static_cast<std::size_t>(rows * output_dim2));
    DeviceBuffer<float> result0(static_cast<std::size_t>(rows * output_dim0));
    DeviceBuffer<float> result1(static_cast<std::size_t>(rows * output_dim1));
    DeviceBuffer<float> result2(static_cast<std::size_t>(rows * output_dim2));
    DeviceBuffer<float> device_gate(input.size());
    DeviceBuffer<float> gated_input(input.size());
    DeviceBuffer<float> gated_baseline0(
        static_cast<std::size_t>(rows * output_dim0));
    DeviceBuffer<float> gated_baseline1(
        static_cast<std::size_t>(rows * output_dim1));
    DeviceBuffer<float> gated_baseline2(
        static_cast<std::size_t>(rows * output_dim2));
    DeviceBuffer<float> gated_result0(
        static_cast<std::size_t>(rows * output_dim0));
    DeviceBuffer<float> gated_result1(
        static_cast<std::size_t>(rows * output_dim1));
    DeviceBuffer<float> gated_result2(
        static_cast<std::size_t>(rows * output_dim2));
    DeviceBuffer<float> swiglu_input(input.size());
    DeviceBuffer<float> swiglu_baseline0(
        static_cast<std::size_t>(rows * output_dim0));
    DeviceBuffer<float> swiglu_baseline1(
        static_cast<std::size_t>(rows * output_dim1));
    DeviceBuffer<float> swiglu_baseline2(
        static_cast<std::size_t>(rows * output_dim2));
    DeviceBuffer<float> swiglu_result0(
        static_cast<std::size_t>(rows * output_dim0));
    DeviceBuffer<float> swiglu_result1(
        static_cast<std::size_t>(rows * output_dim1));
    DeviceBuffer<float> swiglu_result2(
        static_cast<std::size_t>(rows * output_dim2));
    device_input.upload(input);
    std::vector<float> gates(input.size());
    for (std::size_t index = 0; index < gates.size(); ++index) {
        gates[index] = static_cast<float>(
            static_cast<int>((index * 13 + 11) % 101) - 50) / 17.0f;
    }
    device_gate.upload(gates);
    cuda_check(static_cast<cudaError_t>(quantize(
                   device_input.get(), device_q8.get(), device_scales.get(),
                   device_sums.get(), rows, input_dim, nullptr)),
               "MMQ oracle Q8 quantize launch");
    const auto run_baseline = [&](const NfnNativeTilePackedWeightDescriptorV1& value,
                                  float* output) {
        cuda_check(static_cast<cudaError_t>(linear(
                       &value, device_q8.get(), device_scales.get(),
                       device_sums.get(), nullptr, output, rows, false)),
                   "MMQ oracle packed linear launch");
    };
    run_baseline(descriptor0, baseline0.get());
    run_baseline(descriptor1, baseline1.get());
    run_baseline(descriptor2, baseline2.get());

    const std::int64_t required_workspace = workspace_bytes(rows, input_dim);
    if (required_workspace <= 0 || workspace_bytes(17, input_dim) != 0 ||
        workspace_bytes(rows, input_dim - 1) != 0) {
        throw std::runtime_error("exact K-quant MMQ workspace contract mismatch");
    }
    DeviceBuffer<std::uint8_t> workspace(
        static_cast<std::size_t>(required_workspace));
    const NfnNativeTilePackedWeightDescriptorV1* descriptors[]{
        &descriptor0, &descriptor1, &descriptor2};
    float* outputs[]{result0.get(), result1.get(), result2.get()};
    cuda_check(static_cast<cudaError_t>(mmq(
                   descriptors, device_input.get(), outputs, 3, rows,
                   workspace.get(), required_workspace, nullptr)),
               "mixed Q4/Q5/Q6 exact MMQ launch");
    cuda_check(cudaDeviceSynchronize(), "mixed exact MMQ synchronize");
    const auto require_mmq_close = [](const std::vector<float>& expected,
                                      const std::vector<float>& actual,
                                      const std::string& label) {
        if (expected.size() != actual.size()) {
            throw std::runtime_error(label + " output size mismatch");
        }
        for (std::size_t index = 0; index < expected.size(); ++index) {
            const float tolerance = 2.0e-2f + 4.0e-4f * std::abs(expected[index]);
            require_close(actual[index], expected[index], tolerance,
                          label + " at " + std::to_string(index));
        }
    };
    require_mmq_close(baseline0.download(), result0.download(), "Q4_K exact MMQ");
    require_mmq_close(baseline1.download(), result1.download(), "Q5_K exact MMQ");
    require_mmq_close(baseline2.download(), result2.download(), "Q6_K exact MMQ");

    const std::int64_t single_workspace_bytes = workspace_bytes(1, input_dim);
    if (single_workspace_bytes <= 0) {
        throw std::runtime_error("exact K-quant MMVQ workspace contract mismatch");
    }
    DeviceBuffer<std::uint8_t> single_workspace(
        static_cast<std::size_t>(single_workspace_bytes));
    DeviceBuffer<float> single_baseline0(static_cast<std::size_t>(output_dim0));
    DeviceBuffer<float> single_baseline1(static_cast<std::size_t>(output_dim1));
    DeviceBuffer<float> single_baseline2(static_cast<std::size_t>(output_dim2));
    DeviceBuffer<float> single_result0(static_cast<std::size_t>(output_dim0));
    DeviceBuffer<float> single_result1(static_cast<std::size_t>(output_dim1));
    DeviceBuffer<float> single_result2(static_cast<std::size_t>(output_dim2));
    DeviceBuffer<float> single_direct0(static_cast<std::size_t>(output_dim0));
    cuda_check(static_cast<cudaError_t>(quantize(
                   device_input.get(), device_q8.get(), device_scales.get(),
                   device_sums.get(), 1, input_dim, nullptr)),
               "MMVQ oracle Q8 quantize launch");
    const auto run_single_baseline = [&] (
        const NfnNativeTilePackedWeightDescriptorV1& value, float* output) {
        cuda_check(static_cast<cudaError_t>(linear(
                       &value, device_q8.get(), device_scales.get(),
                       device_sums.get(), nullptr, output, 1, false)),
                   "MMVQ oracle packed linear launch");
    };
    run_single_baseline(descriptor0, single_baseline0.get());
    run_single_baseline(descriptor1, single_baseline1.get());
    run_single_baseline(descriptor2, single_baseline2.get());
    float* single_outputs[]{
        single_result0.get(), single_result1.get(), single_result2.get()};
    cuda_check(static_cast<cudaError_t>(mmq(
                   descriptors, device_input.get(), single_outputs, 3, 1,
                   single_workspace.get(), single_workspace_bytes, nullptr)),
               "mixed Q4/Q5/Q6 exact MMVQ launch");
    cuda_check(static_cast<cudaError_t>(mmvq(
                   &descriptor0, device_input.get(), single_direct0.get(),
                   single_workspace.get(), single_workspace_bytes, nullptr)),
               "dedicated exact Q4_K MMVQ launch");
    cuda_check(cudaDeviceSynchronize(), "mixed exact MMVQ synchronize");
    const auto require_mmvq_close = [](const std::vector<float>& expected,
                                       const std::vector<float>& actual,
                                       const std::string& label) {
        if (expected.size() != actual.size()) {
            throw std::runtime_error(label + " output size mismatch");
        }
        for (std::size_t index = 0; index < expected.size(); ++index) {
            // MMVQ distributes the K reduction across four warps, whereas the
            // scalar packed-linear oracle accumulates it serially.  Bound that
            // FP32 reassociation independently from the bit-exact fusion tests
            // below; full-model greedy hashes are checked by the chat harness.
            const float tolerance = 2.5e-1f + 1.0e-3f * std::abs(expected[index]);
            require_close(actual[index], expected[index], tolerance,
                          label + " at " + std::to_string(index));
        }
    };
    require_mmvq_close(single_baseline0.download(), single_result0.download(),
                       "Q4_K exact MMVQ");
    require_mmvq_close(single_baseline1.download(), single_result1.download(),
                       "Q5_K exact MMVQ");
    require_mmvq_close(single_baseline2.download(), single_result2.download(),
                       "Q6_K exact MMVQ");
    const auto single_multi0 = single_result0.download();
    const auto single_direct = single_direct0.download();
    for (std::size_t index = 0; index < single_multi0.size(); ++index) {
        if (std::bit_cast<std::uint32_t>(single_multi0[index]) !=
            std::bit_cast<std::uint32_t>(single_direct[index])) {
            throw std::runtime_error(
                "dedicated and grouped MMVQ differ at " +
                std::to_string(index));
        }
    }

    const auto require_rows_bit_exact = [](
        const std::vector<float>& expected,
        const std::vector<float>& actual,
        const std::string& label) {
        if (expected.size() != actual.size()) {
            throw std::runtime_error(label + " output size mismatch");
        }
        for (std::size_t index = 0; index < expected.size(); ++index) {
            if (std::bit_cast<std::uint32_t>(expected[index]) !=
                std::bit_cast<std::uint32_t>(actual[index])) {
                throw std::runtime_error(
                    label + " is not bit-exact at " + std::to_string(index));
            }
        }
    };
    const auto run_rowwise = [&] (
        GatedMmqFn operation,
        const float* first,
        const float* second,
        DeviceBuffer<float>& output0,
        DeviceBuffer<float>& output1,
        DeviceBuffer<float>& output2,
        const std::string& label) {
        for (std::int64_t row = 0; row < rows; ++row) {
            const std::string operation_label =
                label + " row " + std::to_string(row);
            float* row_outputs[]{
                output0.get() + row * output_dim0,
                output1.get() + row * output_dim1,
                output2.get() + row * output_dim2};
            cuda_check(static_cast<cudaError_t>(operation(
                           descriptors, first + row * input_dim,
                           second == nullptr ? nullptr : second + row * input_dim,
                           row_outputs, 3, 1, single_workspace.get(),
                           single_workspace_bytes, nullptr)),
                       operation_label.c_str());
        }
    };
    const auto run_plain_rowwise = [&] (
        DeviceBuffer<float>& output0,
        DeviceBuffer<float>& output1,
        DeviceBuffer<float>& output2) {
        for (std::int64_t row = 0; row < rows; ++row) {
            const std::string operation_label =
                "row-wise exact MMVQ row " + std::to_string(row);
            float* row_outputs[]{
                output0.get() + row * output_dim0,
                output1.get() + row * output_dim1,
                output2.get() + row * output_dim2};
            cuda_check(static_cast<cudaError_t>(mmq(
                           descriptors, device_input.get() + row * input_dim,
                           row_outputs, 3, 1, single_workspace.get(),
                           single_workspace_bytes, nullptr)),
                       operation_label.c_str());
        }
    };

    run_plain_rowwise(baseline0, baseline1, baseline2);
    cuda_check(static_cast<cudaError_t>(batched_mmvq(
                   descriptors, device_input.get(), outputs, 3, rows,
                   workspace.get(), required_workspace, nullptr)),
               "batched verifier MMVQ launch");
    cuda_check(cudaDeviceSynchronize(), "batched verifier MMVQ synchronize");
    require_rows_bit_exact(
        baseline0.download(), result0.download(),
        "Q4_K batched verifier MMVQ");
    require_rows_bit_exact(
        baseline1.download(), result1.download(),
        "Q5_K batched verifier MMVQ");
    require_rows_bit_exact(
        baseline2.download(), result2.download(),
        "Q6_K batched verifier MMVQ");

    run_rowwise(
        gated_mmq, device_input.get(), device_gate.get(), baseline0, baseline1,
        baseline2, "row-wise gated MMVQ");
    cuda_check(static_cast<cudaError_t>(batched_gated_mmvq(
                   descriptors, device_input.get(), device_gate.get(), outputs,
                   3, rows, workspace.get(), required_workspace, nullptr)),
               "batched gated verifier MMVQ launch");
    cuda_check(cudaDeviceSynchronize(),
               "batched gated verifier MMVQ synchronize");
    require_rows_bit_exact(
        baseline0.download(), result0.download(),
        "Q4_K batched gated verifier MMVQ");
    require_rows_bit_exact(
        baseline1.download(), result1.download(),
        "Q5_K batched gated verifier MMVQ");
    require_rows_bit_exact(
        baseline2.download(), result2.download(),
        "Q6_K batched gated verifier MMVQ");

    run_rowwise(
        swiglu_mmq, device_input.get(), device_gate.get(), baseline0, baseline1,
        baseline2, "row-wise SwiGLU MMVQ");
    cuda_check(static_cast<cudaError_t>(batched_swiglu_mmvq(
                   descriptors, device_input.get(), device_gate.get(), outputs,
                   3, rows, workspace.get(), required_workspace, nullptr)),
               "batched SwiGLU verifier MMVQ launch");
    cuda_check(cudaDeviceSynchronize(),
               "batched SwiGLU verifier MMVQ synchronize");
    require_rows_bit_exact(
        baseline0.download(), result0.download(),
        "Q4_K batched SwiGLU verifier MMVQ");
    require_rows_bit_exact(
        baseline1.download(), result1.download(),
        "Q5_K batched SwiGLU verifier MMVQ");
    require_rows_bit_exact(
        baseline2.download(), result2.download(),
        "Q6_K batched SwiGLU verifier MMVQ");

    DeviceBuffer<float> single_materialized(static_cast<std::size_t>(input_dim));
    DeviceBuffer<float> single_fused_baseline(static_cast<std::size_t>(output_dim0));
    DeviceBuffer<float> single_fused_result(static_cast<std::size_t>(output_dim0));
    const NfnNativeTilePackedWeightDescriptorV1* single_descriptor[]{&descriptor0};
    float* single_baseline_output[]{single_fused_baseline.get()};
    float* single_fused_output[]{single_fused_result.get()};
    const auto require_single_fused_exact = [&] (const std::string& label) {
        const auto expected = single_fused_baseline.download();
        const auto actual = single_fused_result.download();
        for (std::size_t index = 0; index < expected.size(); ++index) {
            if (std::bit_cast<std::uint32_t>(expected[index]) !=
                std::bit_cast<std::uint32_t>(actual[index])) {
                throw std::runtime_error(
                    label + " is not bit-exact at " + std::to_string(index));
            }
        }
    };
    cuda_check(static_cast<cudaError_t>(gate_op(
                   device_input.get(), device_gate.get(), single_materialized.get(),
                   input_dim, nullptr)),
               "materialized one-row MMVQ sigmoid gate launch");
    cuda_check(static_cast<cudaError_t>(mmq(
                   single_descriptor, single_materialized.get(),
                   single_baseline_output, 1, 1, single_workspace.get(),
                   single_workspace_bytes, nullptr)),
               "materialized one-row sigmoid-gated MMVQ launch");
    cuda_check(static_cast<cudaError_t>(gated_mmq(
                   single_descriptor, device_input.get(), device_gate.get(),
                   single_fused_output, 1, 1, single_workspace.get(),
                   single_workspace_bytes, nullptr)),
               "fused one-row sigmoid-gated MMVQ launch");
    cuda_check(cudaDeviceSynchronize(),
               "fused one-row sigmoid-gated MMVQ synchronize");
    require_single_fused_exact("Q4_K fused one-row sigmoid-gated MMVQ");

    cuda_check(static_cast<cudaError_t>(swiglu_op(
                   device_input.get(), device_gate.get(), single_materialized.get(),
                   input_dim, nullptr)),
               "materialized one-row MMVQ SwiGLU launch");
    cuda_check(static_cast<cudaError_t>(mmq(
                   single_descriptor, single_materialized.get(),
                   single_baseline_output, 1, 1, single_workspace.get(),
                   single_workspace_bytes, nullptr)),
               "materialized one-row SwiGLU MMVQ launch");
    cuda_check(static_cast<cudaError_t>(swiglu_mmq(
                   single_descriptor, device_input.get(), device_gate.get(),
                   single_fused_output, 1, 1, single_workspace.get(),
                   single_workspace_bytes, nullptr)),
               "fused one-row SwiGLU MMVQ launch");
    cuda_check(cudaDeviceSynchronize(), "fused one-row MMVQ SwiGLU synchronize");
    require_single_fused_exact("Q4_K fused one-row SwiGLU MMVQ");

    DeviceBuffer<float> ordinary_hidden(static_cast<std::size_t>(input_dim));
    DeviceBuffer<float> ordinary_normalized(static_cast<std::size_t>(input_dim));
    DeviceBuffer<float> ordinary_capture(static_cast<std::size_t>(input_dim));
    DeviceBuffer<float> handoff_hidden(static_cast<std::size_t>(input_dim));
    DeviceBuffer<float> handoff_normalized(static_cast<std::size_t>(input_dim));
    DeviceBuffer<float> handoff_capture(static_cast<std::size_t>(input_dim));
    DeviceBuffer<float> ordinary_mmvq(static_cast<std::size_t>(output_dim0));
    DeviceBuffer<float> handoff_mmvq(static_cast<std::size_t>(output_dim0));
    float* ordinary_mmvq_output[]{ordinary_mmvq.get()};
    float* handoff_mmvq_output[]{handoff_mmvq.get()};
    cuda_check(static_cast<cudaError_t>(dual(
                   device_input.get(), nullptr, device_gate.get(),
                   ordinary_hidden.get(), nullptr, ordinary_normalized.get(),
                   ordinary_capture.get(), 1, input_dim, 1.0e-8f, false,
                   1.0e-5f, false, nullptr)),
               "ordinary dual RMS handoff oracle launch");
    cuda_check(static_cast<cudaError_t>(mmq(
                   single_descriptor, ordinary_normalized.get(),
                   ordinary_mmvq_output, 1, 1, single_workspace.get(),
                   single_workspace_bytes, nullptr)),
               "ordinary post-dual-RMS MMVQ launch");
    cuda_check(static_cast<cudaError_t>(dual_handoff(
                   device_input.get(), nullptr, device_gate.get(),
                   handoff_hidden.get(), nullptr, handoff_normalized.get(),
                   handoff_capture.get(), 1, input_dim, 1.0e-8f, false,
                   1.0e-5f, false, single_workspace.get(),
                   single_workspace_bytes, nullptr)),
               "dual RMS to prequantized MMVQ handoff launch");
    cuda_check(static_cast<cudaError_t>(prequantized_mmvq(
                   single_descriptor, handoff_mmvq_output, 1, 1,
                   single_workspace.get(), single_workspace_bytes, nullptr)),
               "prequantized MMVQ handoff launch");
    cuda_check(cudaDeviceSynchronize(),
               "dual RMS to MMVQ handoff synchronize");
    require_rows_bit_exact(
        ordinary_hidden.download(), handoff_hidden.download(),
        "dual RMS handoff hidden output");
    require_rows_bit_exact(
        ordinary_normalized.download(), handoff_normalized.download(),
        "dual RMS handoff normalized output");
    require_rows_bit_exact(
        ordinary_capture.download(), handoff_capture.download(),
        "dual RMS handoff residual capture");
    require_rows_bit_exact(
        ordinary_mmvq.download(), handoff_mmvq.download(),
        "dual RMS prequantized MMVQ handoff");
    if (dual_handoff(
            device_input.get(), nullptr, device_gate.get(),
            handoff_hidden.get(), nullptr, handoff_normalized.get(),
            handoff_capture.get(), 2, input_dim, 1.0e-8f, false, 1.0e-5f,
            false, single_workspace.get(), single_workspace_bytes, nullptr) !=
        static_cast<int>(cudaErrorInvalidValue)) {
        throw std::runtime_error("dual RMS MMVQ handoff accepted rows != 1");
    }
    if (dual_handoff(
            device_input.get(), nullptr, device_gate.get(),
            handoff_hidden.get(), nullptr, handoff_normalized.get(),
            handoff_capture.get(), 1, input_dim, 1.0e-8f, false, 1.0e-5f,
            false, single_workspace.get(), (input_dim / 32) * 36 - 1,
            nullptr) != static_cast<int>(cudaErrorInvalidValue)) {
        throw std::runtime_error("dual RMS MMVQ handoff accepted short workspace");
    }
    if (prequantized_mmvq(
            single_descriptor, handoff_mmvq_output, 1, 17,
            single_workspace.get(), single_workspace_bytes, nullptr) !=
        static_cast<int>(cudaErrorInvalidValue)) {
        throw std::runtime_error(
            "prequantized MMVQ handoff accepted more than 16 rows");
    }

    cuda_check(static_cast<cudaError_t>(gate_op(
                   device_input.get(), device_gate.get(), gated_input.get(),
                   rows * input_dim, nullptr)),
               "materialized MMQ sigmoid gate launch");
    float* gated_baseline_outputs[]{
        gated_baseline0.get(), gated_baseline1.get(), gated_baseline2.get()};
    cuda_check(static_cast<cudaError_t>(mmq(
                   descriptors, gated_input.get(), gated_baseline_outputs, 3,
                   rows, workspace.get(), required_workspace, nullptr)),
               "materialized sigmoid-gated exact MMQ launch");
    float* gated_outputs[]{
        gated_result0.get(), gated_result1.get(), gated_result2.get()};
    cuda_check(static_cast<cudaError_t>(gated_mmq(
                   descriptors, device_input.get(), device_gate.get(),
                   gated_outputs, 3, rows, workspace.get(), required_workspace,
                   nullptr)),
               "fused sigmoid-gated exact MMQ launch");
    cuda_check(cudaDeviceSynchronize(),
               "fused sigmoid-gated exact MMQ synchronize");
    const auto require_bit_exact = [](const std::vector<float>& expected,
                                      const std::vector<float>& actual,
                                      const std::string& label) {
        if (expected.size() != actual.size()) {
            throw std::runtime_error(label + " output size mismatch");
        }
        for (std::size_t index = 0; index < expected.size(); ++index) {
            if (std::bit_cast<std::uint32_t>(expected[index]) !=
                std::bit_cast<std::uint32_t>(actual[index])) {
                throw std::runtime_error(
                    label + " is not bit-exact at " + std::to_string(index));
            }
        }
    };
    require_bit_exact(gated_baseline0.download(), gated_result0.download(),
                      "Q4_K fused sigmoid-gated MMQ");
    require_bit_exact(gated_baseline1.download(), gated_result1.download(),
                      "Q5_K fused sigmoid-gated MMQ");
    require_bit_exact(gated_baseline2.download(), gated_result2.download(),
                      "Q6_K fused sigmoid-gated MMQ");
    if (gated_mmq(
            descriptors, device_input.get(), nullptr, gated_outputs, 3, rows,
            workspace.get(), required_workspace, nullptr) !=
        static_cast<int>(cudaErrorInvalidValue)) {
        throw std::runtime_error("gated exact MMQ accepted a null gate");
    }

    cuda_check(static_cast<cudaError_t>(swiglu_op(
                   device_input.get(), device_gate.get(), swiglu_input.get(),
                   rows * input_dim, nullptr)),
               "materialized MMQ SwiGLU launch");
    float* swiglu_baseline_outputs[]{
        swiglu_baseline0.get(), swiglu_baseline1.get(), swiglu_baseline2.get()};
    cuda_check(static_cast<cudaError_t>(mmq(
                   descriptors, swiglu_input.get(), swiglu_baseline_outputs, 3,
                   rows, workspace.get(), required_workspace, nullptr)),
               "materialized SwiGLU exact MMQ launch");
    float* swiglu_outputs[]{
        swiglu_result0.get(), swiglu_result1.get(), swiglu_result2.get()};
    cuda_check(static_cast<cudaError_t>(swiglu_mmq(
                   descriptors, device_input.get(), device_gate.get(),
                   swiglu_outputs, 3, rows, workspace.get(), required_workspace,
                   nullptr)),
               "fused SwiGLU exact MMQ launch");
    cuda_check(cudaDeviceSynchronize(), "fused SwiGLU exact MMQ synchronize");
    require_bit_exact(swiglu_baseline0.download(), swiglu_result0.download(),
                      "Q4_K fused SwiGLU MMQ");
    require_bit_exact(swiglu_baseline1.download(), swiglu_result1.download(),
                      "Q5_K fused SwiGLU MMQ");
    require_bit_exact(swiglu_baseline2.download(), swiglu_result2.download(),
                      "Q6_K fused SwiGLU MMQ");
    if (swiglu_mmq(
            descriptors, device_input.get(), nullptr, swiglu_outputs, 3, rows,
            workspace.get(), required_workspace, nullptr) !=
        static_cast<int>(cudaErrorInvalidValue)) {
        throw std::runtime_error("SwiGLU exact MMQ accepted a null up tensor");
    }

    const auto require_invalid = [&](const NfnNativeTilePackedWeightDescriptorV1* value,
                                     std::int64_t test_rows,
                                     std::int64_t test_workspace,
                                     const std::string& label) {
        const NfnNativeTilePackedWeightDescriptorV1* one[]{value};
        float* one_output[]{result0.get()};
        const int status = mmq(
            one, device_input.get(), one_output, 1, test_rows, workspace.get(),
            test_workspace, nullptr);
        if (status != static_cast<int>(cudaErrorInvalidValue)) {
            throw std::runtime_error(label + " did not fail closed");
        }
    };
    auto malformed = descriptor0;
    --malformed.data_nbytes;
    require_invalid(&malformed, rows, required_workspace, "truncated MMQ tensor");
    malformed = descriptor0;
    malformed.encoding = 0xffffu;
    require_invalid(&malformed, rows, required_workspace, "unknown MMQ encoding");
    malformed = descriptor0;
    --malformed.row_stride_bytes;
    require_invalid(&malformed, rows, required_workspace, "bad MMQ row stride");
    require_invalid(&descriptor0, rows, required_workspace - 1,
                    "short MMQ workspace");
    require_invalid(&descriptor0, 17, required_workspace,
                    "unsupported MMQ row count");
}

void check_argmax_rows(void* library) {
    using Fn = int (*)(const float*, std::int64_t*, float*, std::int64_t,
                       std::int64_t, void*);
    const auto operation = required_symbol<Fn>(
        library, "nfn_native_tile_argmax_rows_float32_v1");
    constexpr std::int64_t rows = 3;
    constexpr std::int64_t width = 202048;
    std::vector<float> values(static_cast<std::size_t>(rows * width), -9.0f);
    values[17] = 10.0f;
    values[123] = 10.0f;  // Equal maximum must retain the lower index.
    values[static_cast<std::size_t>(width + width - 1)] = 5.5f;
    values[static_cast<std::size_t>(2 * width)] = -1.0f;
    DeviceBuffer<float> device_values(values.size());
    DeviceBuffer<std::int64_t> device_indices(static_cast<std::size_t>(rows));
    DeviceBuffer<float> device_maxima(static_cast<std::size_t>(rows));
    device_values.upload(values);
    cuda_check(static_cast<cudaError_t>(operation(
                   device_values.get(), device_indices.get(), device_maxima.get(),
                   rows, width, nullptr)),
               "Glimmer row argmax launch");
    cuda_check(cudaDeviceSynchronize(), "Glimmer row argmax synchronize");
    const auto indices = device_indices.download();
    const auto maxima = device_maxima.download();
    const std::vector<std::int64_t> expected_indices{17, width - 1, 0};
    const std::vector<float> expected_values{10.0f, 5.5f, -1.0f};
    for (std::int64_t row = 0; row < rows; ++row) {
        if (indices[static_cast<std::size_t>(row)] !=
            expected_indices[static_cast<std::size_t>(row)]) {
            throw std::runtime_error("Glimmer row argmax index mismatch");
        }
        require_close(
            maxima[static_cast<std::size_t>(row)],
            expected_values[static_cast<std::size_t>(row)], 0.0f,
            "Glimmer row argmax value");
    }
}

void check_batched_cache_commit(void* library) {
    using Fn = int (*)(const NfnNativeTileGlimmerCacheCommitDescriptorV1*,
                       std::int64_t);
    const auto operation = required_symbol<Fn>(
        library, "nfn_native_tile_glimmer_cache_commit_rows_bf16_v1");
    constexpr std::int64_t rows = 3;
    constexpr std::int64_t kv_heads = 2;
    constexpr std::int64_t head_dim = 128;
    constexpr std::int64_t width = kv_heads * head_dim;
    constexpr std::int64_t capacity = 4;
    constexpr std::int64_t position = 3;
    std::vector<float> keys(static_cast<std::size_t>(rows * width));
    std::vector<float> values(keys.size());
    for (std::size_t index = 0; index < keys.size(); ++index) {
        keys[index] = static_cast<float>(static_cast<int>(index % 113) - 56) / 31.0f;
        values[index] = static_cast<float>(static_cast<int>(index % 97) - 48) / 29.0f;
    }
    std::vector<std::uint16_t> zero(static_cast<std::size_t>(capacity * width), 0);
    DeviceBuffer<float> device_keys(keys.size());
    DeviceBuffer<float> device_values(values.size());
    DeviceBuffer<std::uint16_t> device_key_cache(zero.size());
    DeviceBuffer<std::uint16_t> device_value_cache(zero.size());
    device_keys.upload(keys);
    device_values.upload(values);
    device_key_cache.upload(zero);
    device_value_cache.upload(zero);
    NfnNativeTileGlimmerCacheCommitDescriptorV1 descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
    descriptor.current_key = device_keys.get();
    descriptor.current_value = device_values.get();
    descriptor.key_cache_bf16 = device_key_cache.get();
    descriptor.value_cache_bf16 = device_value_cache.get();
    descriptor.kv_heads = kv_heads;
    descriptor.head_dim = head_dim;
    descriptor.position = position;
    descriptor.cache_capacity = capacity;
    descriptor.cache_row_stride = width;
    cuda_check(static_cast<cudaError_t>(operation(&descriptor, rows)),
               "batched cache commit launch");
    cuda_check(cudaDeviceSynchronize(), "batched cache commit synchronize");
    const auto key_cache = device_key_cache.download();
    const auto value_cache = device_value_cache.download();
    for (std::int64_t row = 0; row < rows; ++row) {
        const std::int64_t slot = (position + row) % capacity;
        for (std::int64_t column = 0; column < width; ++column) {
            const std::size_t source = static_cast<std::size_t>(row * width + column);
            const std::size_t target = static_cast<std::size_t>(slot * width + column);
            require_close(bf16_to_float(key_cache[target]),
                          bf16_to_float(float_to_bf16(keys[source])), 0.0f,
                          "batched cache key");
            require_close(bf16_to_float(value_cache[target]),
                          bf16_to_float(float_to_bf16(values[source])), 0.0f,
                          "batched cache value");
        }
    }
    for (std::int64_t column = 0; column < width; ++column) {
        if (key_cache[static_cast<std::size_t>(2 * width + column)] != 0 ||
            value_cache[static_cast<std::size_t>(2 * width + column)] != 0) {
            throw std::runtime_error("batched cache commit changed an untouched slot");
        }
    }
}

void check_all_layer_cache_commit(void* library) {
    using Fn = int (*)(
        const NfnNativeTileGlimmerCacheCommitLayersDescriptorV1*);
    const auto operation = required_symbol<Fn>(
        library, "nfn_native_tile_glimmer_cache_commit_layers_bf16_v1");
    constexpr std::int64_t layer_count = 3;
    constexpr std::int64_t source_rows = 3;
    constexpr std::int64_t rows = 2;
    constexpr std::int64_t kv_heads = 2;
    constexpr std::int64_t head_dim = 128;
    constexpr std::int64_t width = kv_heads * head_dim;
    constexpr std::int64_t layer_stride = source_rows * width;
    constexpr std::int64_t position = 3;
    std::vector<float> keys(
        static_cast<std::size_t>(layer_count * layer_stride));
    std::vector<float> values(keys.size());
    for (std::int64_t layer = 0; layer < layer_count; ++layer) {
        for (std::int64_t row = 0; row < source_rows; ++row) {
            for (std::int64_t column = 0; column < width; ++column) {
                const std::size_t index = static_cast<std::size_t>(
                    layer * layer_stride + row * width + column);
                keys[index] = static_cast<float>(
                    1000 * layer + 100 * row + column + 1) / 257.0f;
                values[index] = -static_cast<float>(
                    1000 * layer + 100 * row + column + 1) / 263.0f;
            }
        }
    }
    DeviceBuffer<float> device_keys(keys.size());
    DeviceBuffer<float> device_values(values.size());
    device_keys.upload(keys);
    device_values.upload(values);

    constexpr std::int64_t capacity0 = 4;
    constexpr std::int64_t capacity1 = 5;
    constexpr std::int64_t capacity2 = 2;
    constexpr std::int64_t stride0 = width;
    constexpr std::int64_t stride1 = width + 3;
    constexpr std::int64_t stride2 = width + 7;
    std::vector<std::uint16_t> zero0(
        static_cast<std::size_t>(capacity0 * stride0), 0);
    std::vector<std::uint16_t> zero1(
        static_cast<std::size_t>(capacity1 * stride1), 0);
    std::vector<std::uint16_t> zero2(
        static_cast<std::size_t>(capacity2 * stride2), 0);
    DeviceBuffer<std::uint16_t> key_cache0(zero0.size());
    DeviceBuffer<std::uint16_t> value_cache0(zero0.size());
    DeviceBuffer<std::uint16_t> key_cache1(zero1.size());
    DeviceBuffer<std::uint16_t> value_cache1(zero1.size());
    DeviceBuffer<std::uint16_t> key_cache2(zero2.size());
    DeviceBuffer<std::uint16_t> value_cache2(zero2.size());
    key_cache0.upload(zero0);
    value_cache0.upload(zero0);
    key_cache1.upload(zero1);
    value_cache1.upload(zero1);
    key_cache2.upload(zero2);
    value_cache2.upload(zero2);

    NfnNativeTileGlimmerCacheLayerV1 layers[layer_count]{};
    layers[0] = {key_cache0.get(), value_cache0.get(), capacity0, stride0};
    layers[1] = {key_cache1.get(), value_cache1.get(), capacity1, stride1};
    layers[2] = {key_cache2.get(), value_cache2.get(), capacity2, stride2};
    NfnNativeTileGlimmerCacheCommitLayersDescriptorV1 descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
    descriptor.staged_keys = device_keys.get();
    descriptor.staged_values = device_values.get();
    descriptor.layers = layers;
    descriptor.layer_count = layer_count;
    descriptor.source_rows = source_rows;
    descriptor.rows = rows;
    descriptor.kv_heads = kv_heads;
    descriptor.head_dim = head_dim;
    descriptor.position = position;
    descriptor.source_layer_stride = layer_stride;
    cuda_check(static_cast<cudaError_t>(operation(&descriptor)),
               "all-layer cache commit launch");
    cuda_check(cudaDeviceSynchronize(), "all-layer cache commit synchronize");

    const auto verify_layer = [&](
        std::int64_t layer, std::int64_t capacity, std::int64_t stride,
        const std::vector<std::uint16_t>& key_cache,
        const std::vector<std::uint16_t>& value_cache) {
        for (std::int64_t slot = 0; slot < capacity; ++slot) {
            std::int64_t source_row = -1;
            for (std::int64_t row = 0; row < rows; ++row) {
                if ((position + row) % capacity == slot) source_row = row;
            }
            for (std::int64_t column = 0; column < stride; ++column) {
                const std::size_t target =
                    static_cast<std::size_t>(slot * stride + column);
                if (source_row < 0 || column >= width) {
                    if (key_cache[target] != 0 || value_cache[target] != 0) {
                        throw std::runtime_error(
                            "all-layer cache commit changed an untouched value");
                    }
                    continue;
                }
                const std::size_t source = static_cast<std::size_t>(
                    layer * layer_stride + source_row * width + column);
                require_close(
                    bf16_to_float(key_cache[target]),
                    bf16_to_float(float_to_bf16(keys[source])), 0.0f,
                    "all-layer cache key");
                require_close(
                    bf16_to_float(value_cache[target]),
                    bf16_to_float(float_to_bf16(values[source])), 0.0f,
                    "all-layer cache value");
            }
        }
    };
    verify_layer(0, capacity0, stride0, key_cache0.download(),
                 value_cache0.download());
    verify_layer(1, capacity1, stride1, key_cache1.download(),
                 value_cache1.download());
    verify_layer(2, capacity2, stride2, key_cache2.download(),
                 value_cache2.download());

    auto invalid = descriptor;
    invalid.rows = source_rows + 1;
    if (operation(&invalid) == cudaSuccess) {
        throw std::runtime_error("all-layer cache commit accepted too many rows");
    }
    invalid = descriptor;
    invalid.source_layer_stride = layer_stride - 1;
    if (operation(&invalid) == cudaSuccess) {
        throw std::runtime_error("all-layer cache commit accepted a short stride");
    }
    const std::int64_t saved_capacity = layers[1].cache_capacity;
    layers[1].cache_capacity = 0;
    if (operation(&descriptor) == cudaSuccess) {
        throw std::runtime_error("all-layer cache commit accepted zero capacity");
    }
    layers[1].cache_capacity = saved_capacity;
}

void check_fused_decode_attention(void* library) {
    using QkFn = int (*)(
        float*, float*, const NfnNativeTilePackedWeightDescriptorV1*,
        const NfnNativeTilePackedWeightDescriptorV1*, std::int64_t,
        std::int64_t, std::int64_t, float, bool, bool, float, std::int64_t,
        float, std::uint32_t, bool, void*);
    using DecodeFn = int (*)(const NfnNativeTileGlimmerGqaDecodeDescriptorV1*);
    using CommitFn = int (*)(const NfnNativeTileGlimmerCacheCommitDescriptorV1*);
    using FusedFn = int (*)(
        const NfnNativeTileGlimmerFusedDecodeAttentionDescriptorV1*);
    using DevicePositionFusedFn = int (*)(
        const NfnNativeTileGlimmerFusedDecodeAttentionDescriptorV1*,
        const std::int64_t*, std::int64_t);
    const auto qk = required_symbol<QkFn>(
        library, "nfn_native_tile_glimmer_qk_norm_scale_rope_float32_v1");
    const auto decode = required_symbol<DecodeFn>(
        library, "nfn_native_tile_glimmer_gqa_decode_float32_v1");
    const auto commit = required_symbol<CommitFn>(
        library, "nfn_native_tile_glimmer_cache_commit_bf16_v1");
    const auto fused = required_symbol<FusedFn>(
        library,
        "nfn_native_tile_glimmer_fused_decode_attention_float32_v1");
    const auto fused_device_position = required_symbol<DevicePositionFusedFn>(
        library,
        "nfn_native_tile_glimmer_fused_decode_attention_device_position_float32_v1");

    constexpr std::int64_t query_heads = 32;
    constexpr std::int64_t kv_heads = 2;
    constexpr std::int64_t head_dim = 128;
    constexpr std::int64_t query_width = query_heads * head_dim;
    constexpr std::int64_t kv_width = kv_heads * head_dim;
    constexpr std::int64_t position = 31;
    constexpr std::int64_t capacity = 64;
    constexpr float norm_eps = 1.0e-5f;
    constexpr float query_scale = 3.87f;
    constexpr float theta = 500000.0f;
    constexpr float attention_scale = 0.08838834764831845f;
    const std::size_t cache_count =
        static_cast<std::size_t>(capacity * kv_width);

    std::vector<float> query(static_cast<std::size_t>(query_width));
    std::vector<float> key(static_cast<std::size_t>(kv_width));
    std::vector<float> value(static_cast<std::size_t>(kv_width));
    std::vector<std::uint16_t> key_cache(cache_count);
    std::vector<std::uint16_t> value_cache(cache_count);
    for (std::size_t index = 0; index < query.size(); ++index) {
        query[index] = patterned_value(
            static_cast<std::int64_t>(index / head_dim),
            static_cast<std::int64_t>(index % head_dim), 31, 37.0f);
    }
    for (std::size_t index = 0; index < key.size(); ++index) {
        key[index] = patterned_value(
            position + static_cast<std::int64_t>(index / head_dim),
            static_cast<std::int64_t>(index % head_dim), 43, 47.0f);
        value[index] = patterned_value(
            position + static_cast<std::int64_t>(index / head_dim) * 3,
            static_cast<std::int64_t>(index % head_dim), 53, 59.0f);
    }
    for (std::int64_t row = 0; row < position; ++row) {
        for (std::int64_t column = 0; column < kv_width; ++column) {
            const std::size_t index = static_cast<std::size_t>(
                row * kv_width + column);
            key_cache[index] = float_to_bf16(patterned_value(
                row + column / head_dim, column % head_dim, 61, 67.0f));
            value_cache[index] = float_to_bf16(patterned_value(
                row + 3 * (column / head_dim), column % head_dim, 71, 73.0f));
        }
    }

    const auto require_exact_float = [](
        const std::vector<float>& expected,
        const std::vector<float>& actual,
        const std::string& label) {
        if (expected.size() != actual.size()) {
            throw std::runtime_error(label + " size mismatch");
        }
        for (std::size_t index = 0; index < expected.size(); ++index) {
            if (std::bit_cast<std::uint32_t>(expected[index]) !=
                std::bit_cast<std::uint32_t>(actual[index])) {
                throw std::runtime_error(
                    label + " is not bit-exact at " + std::to_string(index));
            }
        }
    };

    for (bool apply_rope : {false, true}) {
        DeviceBuffer<float> baseline_query(query.size());
        DeviceBuffer<float> baseline_key(key.size());
        DeviceBuffer<float> baseline_value(value.size());
        DeviceBuffer<float> baseline_output(query.size());
        DeviceBuffer<std::uint16_t> baseline_key_cache(key_cache.size());
        DeviceBuffer<std::uint16_t> baseline_value_cache(value_cache.size());
        DeviceBuffer<float> fused_query(query.size());
        DeviceBuffer<float> fused_key(key.size());
        DeviceBuffer<float> fused_value(value.size());
        DeviceBuffer<float> fused_output(query.size());
        DeviceBuffer<std::uint16_t> fused_key_cache(key_cache.size());
        DeviceBuffer<std::uint16_t> fused_value_cache(value_cache.size());
        DeviceBuffer<float> device_position_query(query.size());
        DeviceBuffer<float> device_position_key(key.size());
        DeviceBuffer<float> device_position_value(value.size());
        DeviceBuffer<float> device_position_output(query.size());
        DeviceBuffer<std::uint16_t> device_position_key_cache(key_cache.size());
        DeviceBuffer<std::uint16_t> device_position_value_cache(value_cache.size());
        DeviceBuffer<std::int64_t> device_position(1);
        baseline_query.upload(query);
        baseline_key.upload(key);
        baseline_value.upload(value);
        baseline_key_cache.upload(key_cache);
        baseline_value_cache.upload(value_cache);
        fused_query.upload(query);
        fused_key.upload(key);
        fused_value.upload(value);
        fused_key_cache.upload(key_cache);
        fused_value_cache.upload(value_cache);
        device_position_query.upload(query);
        device_position_key.upload(key);
        device_position_value.upload(value);
        device_position_key_cache.upload(key_cache);
        device_position_value_cache.upload(value_cache);
        device_position.upload(std::vector<std::int64_t>{position});

        cuda_check(static_cast<cudaError_t>(qk(
                       baseline_query.get(), baseline_key.get(), nullptr,
                       nullptr, query_heads, kv_heads, head_dim, norm_eps,
                       false, false, query_scale, position, theta,
                       NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT, apply_rope,
                       nullptr)),
                   "baseline fused-decode QK launch");
        NfnNativeTileGlimmerGqaDecodeDescriptorV1 attention{};
        attention.struct_size = sizeof(attention);
        attention.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
        attention.query = baseline_query.get();
        attention.current_key = baseline_key.get();
        attention.current_value = baseline_value.get();
        attention.key_cache_bf16 = baseline_key_cache.get();
        attention.value_cache_bf16 = baseline_value_cache.get();
        attention.output = baseline_output.get();
        attention.query_heads = query_heads;
        attention.kv_heads = kv_heads;
        attention.head_dim = head_dim;
        attention.position = position;
        attention.first_key_position = 0;
        attention.cache_capacity = capacity;
        attention.cache_row_stride = kv_width;
        attention.scale = attention_scale;
        cuda_check(static_cast<cudaError_t>(decode(&attention)),
                   "baseline fused-decode GQA launch");
        NfnNativeTileGlimmerCacheCommitDescriptorV1 cache_commit{};
        cache_commit.struct_size = sizeof(cache_commit);
        cache_commit.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
        cache_commit.current_key = baseline_key.get();
        cache_commit.current_value = baseline_value.get();
        cache_commit.key_cache_bf16 = baseline_key_cache.get();
        cache_commit.value_cache_bf16 = baseline_value_cache.get();
        cache_commit.kv_heads = kv_heads;
        cache_commit.head_dim = head_dim;
        cache_commit.position = position;
        cache_commit.cache_capacity = capacity;
        cache_commit.cache_row_stride = kv_width;
        cuda_check(static_cast<cudaError_t>(commit(&cache_commit)),
                   "baseline fused-decode cache launch");

        NfnNativeTileGlimmerFusedDecodeAttentionDescriptorV1 descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
        descriptor.query = fused_query.get();
        descriptor.key = fused_key.get();
        descriptor.current_value = fused_value.get();
        descriptor.key_cache_bf16 = fused_key_cache.get();
        descriptor.value_cache_bf16 = fused_value_cache.get();
        descriptor.output = fused_output.get();
        descriptor.query_heads = query_heads;
        descriptor.kv_heads = kv_heads;
        descriptor.head_dim = head_dim;
        descriptor.position = position;
        descriptor.first_key_position = 0;
        descriptor.cache_capacity = capacity;
        descriptor.cache_row_stride = kv_width;
        descriptor.norm_eps = norm_eps;
        descriptor.query_scale = query_scale;
        descriptor.rope_theta = theta;
        descriptor.attention_scale = attention_scale;
        descriptor.rope_layout =
            NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT;
        descriptor.apply_rope = apply_rope ? 1 : 0;
        cuda_check(static_cast<cudaError_t>(fused(&descriptor)),
                   "fused decode-attention launch");

        auto device_position_descriptor = descriptor;
        device_position_descriptor.query = device_position_query.get();
        device_position_descriptor.key = device_position_key.get();
        device_position_descriptor.current_value = device_position_value.get();
        device_position_descriptor.key_cache_bf16 =
            device_position_key_cache.get();
        device_position_descriptor.value_cache_bf16 =
            device_position_value_cache.get();
        device_position_descriptor.output = device_position_output.get();
        const std::int64_t sliding_window =
            apply_rope ? position + 1 : capacity;
        cuda_check(static_cast<cudaError_t>(fused_device_position(
                       &device_position_descriptor, device_position.get(),
                       sliding_window)),
                   "device-position fused decode-attention launch");
        cuda_check(cudaDeviceSynchronize(),
                   "fused decode-attention synchronize");

        require_exact_float(
            baseline_query.download(), fused_query.download(),
            "fused decode query");
        require_exact_float(
            baseline_key.download(), fused_key.download(),
            "fused decode key");
        require_exact_float(
            baseline_output.download(), fused_output.download(),
            "fused decode attention");
        if (baseline_key_cache.download() != fused_key_cache.download() ||
            baseline_value_cache.download() != fused_value_cache.download()) {
            throw std::runtime_error(
                "fused decode cache write is not bit-exact");
        }
        require_exact_float(
            baseline_query.download(), device_position_query.download(),
            "device-position fused decode query");
        require_exact_float(
            baseline_key.download(), device_position_key.download(),
            "device-position fused decode key");
        require_exact_float(
            baseline_output.download(), device_position_output.download(),
            "device-position fused decode attention");
        if (baseline_key_cache.download() !=
                device_position_key_cache.download() ||
            baseline_value_cache.download() !=
                device_position_value_cache.download()) {
            throw std::runtime_error(
                "device-position fused decode cache write is not bit-exact");
        }
        if (fused_device_position(
                &device_position_descriptor, nullptr, sliding_window) ==
            cudaSuccess) {
            throw std::runtime_error(
                "device-position fused decode accepted a null position");
        }
        if (fused_device_position(
                &device_position_descriptor, device_position.get(), 0) ==
            cudaSuccess) {
            throw std::runtime_error(
                "device-position fused decode accepted a zero window");
        }
    }

    NfnNativeTileGlimmerFusedDecodeAttentionDescriptorV1 invalid{};
    invalid.struct_size = sizeof(invalid);
    invalid.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
    if (fused(&invalid) == cudaSuccess) {
        throw std::runtime_error(
            "fused decode attention accepted null storage");
    }
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
    using ShortSplitFn = int (*)(
        const NfnNativeTileDFlashBlockAttentionDescriptorV1*, float*,
        std::int64_t);
    using DecodeFn = int (*)(const NfnNativeTileGlimmerGqaDecodeDescriptorV1*);
    using CommitFn = int (*)(const NfnNativeTileGlimmerCacheCommitDescriptorV1*);
    const auto operation = required_symbol<Fn>(
        library, "nfn_native_tile_dflash_block_attention_float32_v1");
    const auto short_split = required_symbol<ShortSplitFn>(
        library,
        "nfn_native_tile_dflash_block_attention_short_split_float32_v1");
    const auto decode = required_symbol<DecodeFn>(
        library, "nfn_native_tile_glimmer_gqa_decode_float32_v1");
    const auto commit = required_symbol<CommitFn>(
        library, "nfn_native_tile_glimmer_cache_commit_bf16_v1");
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
    DeviceBuffer<float> device_short_split_output(query.size());
    constexpr std::int64_t split_score_count =
        query_rows * query_heads * 6 * 8;
    DeviceBuffer<float> device_split_scores(
        static_cast<std::size_t>(split_score_count));
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

    cudaStream_t split_stream = nullptr;
    cuda_check(cudaStreamCreate(&split_stream),
               "DFlash short-split stream create");
    descriptor.output = device_short_split_output.get();
    descriptor.cuda_stream = split_stream;
    cuda_check(static_cast<cudaError_t>(short_split(
                   &descriptor, device_split_scores.get(),
                   split_score_count * static_cast<std::int64_t>(sizeof(float)))),
               "DFlash short-split attention launch");
    cuda_check(cudaStreamSynchronize(split_stream),
               "DFlash short-split attention synchronize");
    cuda_check(cudaStreamDestroy(split_stream),
               "DFlash short-split stream destroy");
    const auto short_split_output = device_short_split_output.download();
    for (std::size_t index = 0; index < output.size(); ++index) {
        require_close(short_split_output[index], output[index], 5.0e-4f,
                      "DFlash short-split attention");
    }
    descriptor.cuda_stream = nullptr;

    // Target verification uses the same block ABI with a causal mask. Each
    // earlier tentative K/V row must cross the BF16 cache boundary before a
    // later query observes it, and the resulting block output must be exactly
    // the same as token-at-a-time GQA plus cache commit.
    DeviceBuffer<float> device_causal_output(query.size());
    DeviceBuffer<float> device_serial_output(query.size());
    descriptor.flags = NFN_NATIVE_TILE_BLOCK_ATTENTION_CAUSAL;
    descriptor.output = device_causal_output.get();
    cuda_check(static_cast<cudaError_t>(operation(&descriptor)),
               "causal target block attention launch");
    for (std::int64_t row = 0; row < query_rows; ++row) {
        NfnNativeTileGlimmerGqaDecodeDescriptorV1 serial{};
        serial.struct_size = sizeof(serial);
        serial.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
        serial.query = device_query.get() + row * query_heads * head_dim;
        serial.current_key = device_block_key.get() + row * kv_heads * head_dim;
        serial.current_value =
            device_block_value.get() + row * kv_heads * head_dim;
        serial.key_cache_bf16 = device_key_cache.get();
        serial.value_cache_bf16 = device_value_cache.get();
        serial.output =
            device_serial_output.get() + row * query_heads * head_dim;
        serial.query_heads = query_heads;
        serial.kv_heads = kv_heads;
        serial.head_dim = head_dim;
        serial.position = context_length + row;
        serial.first_key_position = 0;
        serial.cache_capacity = capacity;
        serial.cache_row_stride = row_stride;
        serial.scale = scale;
        cuda_check(static_cast<cudaError_t>(decode(&serial)),
                   "serial target attention launch");

        NfnNativeTileGlimmerCacheCommitDescriptorV1 cache_commit{};
        cache_commit.struct_size = sizeof(cache_commit);
        cache_commit.version = NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
        cache_commit.current_key = serial.current_key;
        cache_commit.current_value = serial.current_value;
        cache_commit.key_cache_bf16 = device_key_cache.get();
        cache_commit.value_cache_bf16 = device_value_cache.get();
        cache_commit.kv_heads = kv_heads;
        cache_commit.head_dim = head_dim;
        cache_commit.position = serial.position;
        cache_commit.cache_capacity = capacity;
        cache_commit.cache_row_stride = row_stride;
        cuda_check(static_cast<cudaError_t>(commit(&cache_commit)),
                   "serial target cache commit launch");
    }
    cuda_check(cudaDeviceSynchronize(),
               "causal target block parity synchronize");
    const auto causal_output = device_causal_output.download();
    const auto serial_output = device_serial_output.download();
    for (std::size_t index = 0; index < causal_output.size(); ++index) {
        if (std::bit_cast<std::uint32_t>(causal_output[index]) !=
            std::bit_cast<std::uint32_t>(serial_output[index])) {
            throw std::runtime_error(
                "causal target block attention differs from serial GQA at " +
                std::to_string(index));
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
        check_device_indirect_embedding(library);
        check_q4_k_packed_linear(library);
        check_packed_k_multilinear(
            library, NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K, 144, "Q4_K");
        check_packed_k_multilinear(
            library, NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K, 176, "Q5_K");
        check_packed_k_multilinear(
            library, NFN_NATIVE_TILE_PACKED_WEIGHT_Q6_K, 210, "Q6_K");
        check_q8_activation_packed_linear(
            library, NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K, 144,
            5, 7, 512, "Q4_K");
        check_q8_activation_packed_linear(
            library, NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K, 176,
            5, 8193, 512, "Q5_K head-dispatch");
        check_q8_activation_packed_linear(
            library, NFN_NATIVE_TILE_PACKED_WEIGHT_Q6_K, 210,
            5, 7, 6656, "Q6_K production-width");
        check_q8_activation_packed_linear(
            library, NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K, 176,
            1, 7, 512, "Q5_K cooperative decode");
        check_q8_activation_packed_linear(
            library, NFN_NATIVE_TILE_PACKED_WEIGHT_Q6_K, 210,
            1, 7, 6656, "Q6_K cooperative decode");
        check_q8_multi_decode(library);
        check_exact_k_quant_mmq(library);
        check_argmax_rows(library);
        check_batched_cache_commit(library);
        check_all_layer_cache_commit(library);
        check_wide_rms_norm(library);
        check_positioned_rope(library);
        check_fused_qk_norm_scale_rope(library);
        check_fused_decode_attention(library);
        check_local_gqa_2048(library);
        check_dflash_block_attention(library);
        check_post_training_losses(library);
        check_vision_layer_norm(library);
        check_vision_prepare_attention_shuffle(library);
        cuda_check(cudaDeviceSynchronize(), "final cudaDeviceSynchronize");
        dlclose(library);
        std::cout
            << "{\"status\":\"passed\",\"device\":" << cuda_device << ","
            << "\"kernels\":[\"q4_q5_q6_k_dequant_multilinear_dx\","
            << "\"q8_1_activation_q4_q5_q6_k_linear\","
            << "\"q8_1_multi_decode_2_4_projection\","
            << "\"exact_mmq_mixed_q4_q5_q6_rows5_tail\","
            << "\"exact_mmq_sigmoid_gate_bit_exact\","
            << "\"exact_mmq_swiglu_bit_exact\","
            << "\"exact_mmvq_mixed_q4_q5_q6_row1\","
            << "\"exact_mmvq_sigmoid_gate_bit_exact\","
            << "\"exact_mmvq_swiglu_bit_exact\","
            << "\"exact_batched_mmvq_q4_q5_q6_rows5_bit_exact\","
            << "\"exact_batched_mmvq_sigmoid_gate_rows5_bit_exact\","
            << "\"exact_batched_mmvq_swiglu_rows5_bit_exact\","
            << "\"argmax_rows_vocab202048\","
            << "\"device_indirect_embedding_i64\","
            << "\"sigmoid_gate\",\"rms_norm_6656\","
            << "\"rms_norm_q8_capture_6656\","
            << "\"dual_rms_add_capture_6656_bit_exact\","
            << "\"cooperative_dual_rms_add_capture_6656_bit_exact\","
            << "\"dual_rms_mmvq_q8_handoff_bit_exact\","
            << "\"positioned_rope_q32_kv2_h128\","
            << "\"qk_norm_scale_rope_q32_kv2_h128\","
            << "\"qk_norm_scale_rope_batch_rows3\","
            << "\"fused_decode_attention_q32_kv2_h128_bit_exact\","
            << "\"fused_decode_attention_device_position_bit_exact\","
            << "\"gqa_decode_q32_kv2_h128_window2048\","
            << "\"cache_commit_bf16\",\"cache_commit_rows_bf16\","
            << "\"cache_commit_layers_bf16\","
            << "\"dflash_block_attention_q16_q32_kv8_h128\","
            << "\"dflash_short_split_attention_q16_q32_kv8_h128\","
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
