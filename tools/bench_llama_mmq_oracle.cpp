// Copyright (c) NeuralFn contributors.
//
// Focused projection benchmark for a pinned llama.cpp/ggml CUDA build.  This
// utility deliberately uses only ggml's public backend API so NeuralFn's
// packed-linear benchmark can be compared with the independent implementation
// at identical K-quant geometry.  It is compiled manually against the pinned
// oracle build; it is not part of the NeuralFn product binaries.

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    ggml_type type = GGML_TYPE_Q4_K;
    std::string type_name = "q4_k";
    std::int64_t input_dim = 6656;
    std::int64_t output_dim = 4096;
    std::int64_t rows = 1;
    int device = 0;
    int warmups = 10;
    int repetitions = 100;
};

[[noreturn]] void usage(const char* program, const std::string& error = {}) {
    if (!error.empty()) {
        std::cerr << "error: " << error << "\n";
    }
    std::cerr
        << "usage: " << program
        << " [--encoding q4_k|q5_k|q6_k] [--input-dim N]"
           " [--output-dim N] [--rows N] [--device N]"
           " [--warmups N] [--repetitions N]\n";
    std::exit(error.empty() ? 0 : 2);
}

std::int64_t parse_i64(const char* value, const char* label) {
    char* end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + label + ": " + value);
    }
    return static_cast<std::int64_t>(parsed);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "-h" || argument == "--help") {
            usage(argv[0]);
        }
        if (index + 1 >= argc) {
            usage(argv[0], "missing value for " + argument);
        }
        const char* value = argv[++index];
        if (argument == "--encoding") {
            options.type_name = value;
            if (options.type_name == "q4_k") {
                options.type = GGML_TYPE_Q4_K;
            } else if (options.type_name == "q5_k") {
                options.type = GGML_TYPE_Q5_K;
            } else if (options.type_name == "q6_k") {
                options.type = GGML_TYPE_Q6_K;
            } else {
                usage(argv[0], "unsupported encoding: " + options.type_name);
            }
        } else if (argument == "--input-dim") {
            options.input_dim = parse_i64(value, "input dimension");
        } else if (argument == "--output-dim") {
            options.output_dim = parse_i64(value, "output dimension");
        } else if (argument == "--rows") {
            options.rows = parse_i64(value, "row count");
        } else if (argument == "--device") {
            options.device = static_cast<int>(parse_i64(value, "device"));
        } else if (argument == "--warmups") {
            options.warmups = static_cast<int>(parse_i64(value, "warmups"));
        } else if (argument == "--repetitions") {
            options.repetitions = static_cast<int>(parse_i64(value, "repetitions"));
        } else {
            usage(argv[0], "unknown option: " + argument);
        }
    }
    if (options.input_dim <= 0 || options.input_dim % 256 != 0 ||
        options.output_dim <= 0 || options.rows <= 0 || options.warmups < 0 ||
        options.repetitions <= 0 || options.device < 0) {
        usage(argv[0], "dimensions must be positive, K divisible by 256, and counts valid");
    }
    return options;
}

void put_u16_le(std::vector<std::uint8_t>& data, std::size_t offset, std::uint16_t value) {
    data.at(offset) = static_cast<std::uint8_t>(value & 0xffU);
    data.at(offset + 1) = static_cast<std::uint8_t>(value >> 8U);
}

std::vector<std::uint8_t> make_packed_weights(
    const Options& options,
    std::size_t* row_stride) {
    const std::size_t block_bytes = options.type == GGML_TYPE_Q4_K
        ? 144U
        : options.type == GGML_TYPE_Q5_K ? 176U : 210U;
    const std::size_t blocks_per_row = static_cast<std::size_t>(options.input_dim / 256);
    *row_stride = blocks_per_row * block_bytes;
    const std::size_t total = static_cast<std::size_t>(options.output_dim) * *row_stride;
    std::vector<std::uint8_t> result(total);
    for (std::size_t index = 0; index < total; ++index) {
        result[index] = static_cast<std::uint8_t>((index * 29U + 17U) & 0xffU);
    }
    for (std::int64_t row = 0; row < options.output_dim; ++row) {
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            const std::size_t offset = static_cast<std::size_t>(row) * *row_stride +
                block * block_bytes;
            if (options.type == GGML_TYPE_Q4_K || options.type == GGML_TYPE_Q5_K) {
                put_u16_le(result, offset, 0x2800U);      // fp16(0.03125)
                put_u16_le(result, offset + 2, 0x2400U);  // fp16(0.015625)
            } else {
                put_u16_le(result, offset + 208, 0x2800U);
            }
        }
    }
    return result;
}

std::vector<float> make_inputs(const Options& options) {
    std::vector<float> result(
        static_cast<std::size_t>(options.rows * options.input_dim));
    for (std::int64_t row = 0; row < options.rows; ++row) {
        for (std::int64_t column = 0; column < options.input_dim; ++column) {
            result[static_cast<std::size_t>(row * options.input_dim + column)] =
                static_cast<float>((row * 37 + column * 13) % 257 - 128) / 257.0F;
        }
    }
    return result;
}

double percentile(std::vector<double> values, double quantile) {
    std::sort(values.begin(), values.end());
    const double position = quantile * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

std::uint64_t output_digest(const std::vector<float>& values) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const float value : values) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= static_cast<std::uint8_t>(bits >> (byte * 8));
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        ggml_backend_t backend = ggml_backend_cuda_init(options.device);
        if (backend == nullptr) {
            throw std::runtime_error("ggml_backend_cuda_init returned null");
        }

        constexpr std::size_t graph_nodes = 16;
        const ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * 16 +
                ggml_graph_overhead_custom(graph_nodes, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc = */ true,
        };
        ggml_context* context = ggml_init(params);
        if (context == nullptr) {
            ggml_backend_free(backend);
            throw std::runtime_error("ggml_init returned null");
        }

        ggml_tensor* weights = ggml_new_tensor_2d(
            context, options.type, options.input_dim, options.output_dim);
        ggml_tensor* input = ggml_new_tensor_2d(
            context, GGML_TYPE_F32, options.input_dim, options.rows);
        ggml_tensor* output = ggml_mul_mat(context, weights, input);
        ggml_set_name(weights, "oracle_weights");
        ggml_set_name(input, "oracle_input");
        ggml_set_name(output, "oracle_output");

        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
        if (buffer == nullptr) {
            ggml_free(context);
            ggml_backend_free(backend);
            throw std::runtime_error("ggml_backend_alloc_ctx_tensors returned null");
        }

        std::size_t row_stride = 0;
        const std::vector<std::uint8_t> packed = make_packed_weights(options, &row_stride);
        const std::vector<float> inputs = make_inputs(options);
        if (packed.size() != ggml_nbytes(weights) ||
            inputs.size() * sizeof(float) != ggml_nbytes(input)) {
            throw std::runtime_error("synthetic tensor byte layout does not match ggml");
        }
        ggml_backend_tensor_set(weights, packed.data(), 0, packed.size());
        ggml_backend_tensor_set(input, inputs.data(), 0, inputs.size() * sizeof(float));
        ggml_backend_synchronize(backend);

        ggml_cgraph* graph = ggml_new_graph_custom(context, graph_nodes, false);
        ggml_build_forward_expand(graph, output);
        for (int iteration = 0; iteration < options.warmups; ++iteration) {
            const ggml_status status = ggml_backend_graph_compute(backend, graph);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error(std::string("warmup graph compute failed: ") +
                    ggml_status_to_string(status));
            }
        }

        std::vector<double> samples_ms;
        samples_ms.reserve(static_cast<std::size_t>(options.repetitions));
        for (int iteration = 0; iteration < options.repetitions; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            const ggml_status status = ggml_backend_graph_compute(backend, graph);
            const auto stop = std::chrono::steady_clock::now();
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error(std::string("timed graph compute failed: ") +
                    ggml_status_to_string(status));
            }
            samples_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
        }

        std::vector<float> outputs(static_cast<std::size_t>(options.rows * options.output_dim));
        ggml_backend_tensor_get(output, outputs.data(), 0, outputs.size() * sizeof(float));
        ggml_backend_synchronize(backend);
        const double median_ms = percentile(samples_ms, 0.5);
        const double p05_ms = percentile(samples_ms, 0.05);
        const double p95_ms = percentile(samples_ms, 0.95);
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"encoding\":\"" << options.type_name << "\""
                  << ",\"rows\":" << options.rows
                  << ",\"input_dim\":" << options.input_dim
                  << ",\"output_dim\":" << options.output_dim
                  << ",\"row_stride_bytes\":" << row_stride
                  << ",\"median_ms\":" << median_ms
                  << ",\"p05_ms\":" << p05_ms
                  << ",\"p95_ms\":" << p95_ms
                  << ",\"digest\":\"0x" << std::hex << output_digest(outputs)
                  << std::dec << "\"}\n";

        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bench_llama_mmq_oracle: " << error.what() << "\n";
        return 1;
    }
}
