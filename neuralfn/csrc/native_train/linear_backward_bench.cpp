#include "tile_ops.h"

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace {

constexpr const char* kDinputStridedSymbol =
    "nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_strided_float32";
constexpr const char* kDweightStridedSymbol =
    "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta";

using LinearDinputStridedFn = int (*)(
    const std::uint16_t*,
    const std::uint16_t*,
    float*,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    void*);

using LinearDweightStridedFn = int (*)(
    const std::uint16_t*,
    const std::uint16_t*,
    float*,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    float,
    void*);

struct Options {
    std::string tile_ops_lib = "build/libnfn_native_train_tile_ops.so";
    std::string operation = "dinput-strided";
    std::string baseline_symbol = kDinputStridedSymbol;
    std::string candidate_symbol = kDinputStridedSymbol;
    std::string json_out;
    int cuda_device = 0;
    std::int64_t rows = 512;
    std::int64_t input_dim = 128;
    std::int64_t output_dim = 256;
    std::int64_t grad_out_row_stride = 0;
    int iterations = 5;
    int warmup = 1;
    float beta = 0.0f;
    bool candidate_first = false;
};

struct VariantResult {
    std::string name;
    std::string symbol;
    float total_ms = 0.0f;
    double ms_per_iter = 0.0;
};

[[noreturn]] void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " [options]\n"
        << "  --tile-ops-lib PATH\n"
        << "  --operation dinput-strided|dweight-strided\n"
        << "  --baseline-symbol NAME\n"
        << "  --candidate-symbol NAME\n"
        << "  --rows N\n"
        << "  --input-dim N\n"
        << "  --output-dim N\n"
        << "  --grad-out-row-stride N\n"
        << "  --iterations N\n"
        << "  --warmup N\n"
        << "  --beta FLOAT\n"
        << "  --candidate-first\n"
        << "  --cuda-device N\n"
        << "  --json-out PATH\n";
    std::exit(2);
}

std::int64_t parse_i64(std::string_view value, std::string_view name) {
    char* end = nullptr;
    const long long parsed = std::strtoll(std::string(value).c_str(), &end, 10);
    if (end == nullptr || *end != '\0') {
        throw std::runtime_error("invalid integer for " + std::string(name) + ": " + std::string(value));
    }
    return static_cast<std::int64_t>(parsed);
}

float parse_float(std::string_view value, std::string_view name) {
    char* end = nullptr;
    const float parsed = std::strtof(std::string(value).c_str(), &end);
    if (end == nullptr || *end != '\0') {
        throw std::runtime_error("invalid float for " + std::string(name) + ": " + std::string(value));
    }
    return parsed;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](std::string_view name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + std::string(name));
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
        } else if (arg == "--tile-ops-lib") {
            options.tile_ops_lib = require_value(arg);
        } else if (arg == "--operation") {
            options.operation = require_value(arg);
        } else if (arg == "--baseline-symbol") {
            options.baseline_symbol = require_value(arg);
        } else if (arg == "--candidate-symbol") {
            options.candidate_symbol = require_value(arg);
        } else if (arg == "--rows") {
            options.rows = parse_i64(require_value(arg), arg);
        } else if (arg == "--input-dim") {
            options.input_dim = parse_i64(require_value(arg), arg);
        } else if (arg == "--output-dim") {
            options.output_dim = parse_i64(require_value(arg), arg);
        } else if (arg == "--grad-out-row-stride") {
            options.grad_out_row_stride = parse_i64(require_value(arg), arg);
        } else if (arg == "--iterations") {
            options.iterations = static_cast<int>(parse_i64(require_value(arg), arg));
        } else if (arg == "--warmup") {
            options.warmup = static_cast<int>(parse_i64(require_value(arg), arg));
        } else if (arg == "--beta") {
            options.beta = parse_float(require_value(arg), arg);
        } else if (arg == "--candidate-first") {
            options.candidate_first = true;
        } else if (arg == "--cuda-device") {
            options.cuda_device = static_cast<int>(parse_i64(require_value(arg), arg));
        } else if (arg == "--json-out") {
            options.json_out = require_value(arg);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.grad_out_row_stride <= 0) {
        options.grad_out_row_stride = options.output_dim;
    }
    if ((options.operation != "dinput-strided" && options.operation != "dweight-strided") ||
        options.rows <= 0 ||
        options.input_dim <= 0 ||
        options.output_dim <= 0 ||
        options.grad_out_row_stride < options.output_dim ||
        options.iterations <= 0 ||
        options.warmup < 0) {
        throw std::runtime_error("invalid benchmark operation, shape, stride, or iteration count");
    }
    if (options.operation == "dweight-strided") {
        if (options.baseline_symbol == kDinputStridedSymbol) {
            options.baseline_symbol = kDweightStridedSymbol;
        }
        if (options.candidate_symbol == kDinputStridedSymbol) {
            options.candidate_symbol = kDweightStridedSymbol;
        }
    }
    return options;
}

void cuda_check(cudaError_t status, std::string_view label) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
T load_symbol(void* handle, const std::string& name) {
    dlerror();
    void* symbol = dlsym(handle, name.c_str());
    const char* error = dlerror();
    if (error != nullptr || symbol == nullptr) {
        throw std::runtime_error("missing Tile ops symbol " + name + (error != nullptr ? ": " + std::string(error) : ""));
    }
    return reinterpret_cast<T>(symbol);
}

std::string json_escape(std::string_view text) {
    std::ostringstream out;
    for (const char ch : text) {
        switch (ch) {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\n':
                out << "\\n";
                break;
            default:
                out << ch;
                break;
        }
    }
    return out.str();
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t bytes) {
        allocate(bytes);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }
    void allocate(std::size_t bytes) {
        bytes_ = bytes;
        cuda_check(cudaMalloc(&ptr_, bytes), "cudaMalloc");
    }
    void* get() const {
        return ptr_;
    }
    std::size_t bytes() const {
        return bytes_;
    }

private:
    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
};

template <typename Fn>
VariantResult run_variant(
    const std::string& name,
    const std::string& symbol,
    Fn fn,
    const Options& options,
    const std::uint16_t* x_bf16,
    const std::uint16_t* grad_out_bf16,
    const std::uint16_t* weight_bf16,
    float* grad_x,
    float* grad_weight) {
    const std::size_t grad_x_bytes =
        static_cast<std::size_t>(options.rows * options.input_dim) * sizeof(float);
    const std::size_t grad_weight_bytes =
        static_cast<std::size_t>(options.input_dim * options.output_dim) * sizeof(float);
    for (int i = 0; i < options.warmup; ++i) {
        cuda_check(cudaMemset(grad_x, 0, grad_x_bytes), "warmup grad_x memset");
        cuda_check(cudaMemset(grad_weight, 0, grad_weight_bytes), "warmup grad_weight memset");
        int status = 0;
        if constexpr (std::is_same_v<Fn, LinearDinputStridedFn>) {
            status = fn(
                grad_out_bf16,
                weight_bf16,
                grad_x,
                options.rows,
                options.input_dim,
                options.output_dim,
                options.grad_out_row_stride,
                nullptr);
        } else {
            status = fn(
                x_bf16,
                grad_out_bf16,
                grad_weight,
                options.rows,
                options.input_dim,
                options.output_dim,
                options.grad_out_row_stride,
                options.beta,
                nullptr);
        }
        if (status != 0) {
            throw std::runtime_error(name + " warmup returned status " + std::to_string(status));
        }
        cuda_check(cudaDeviceSynchronize(), name + " warmup synchronize");
    }

    cuda_check(cudaMemset(grad_x, 0, grad_x_bytes), "timed pre-reset grad_x memset");
    cuda_check(cudaMemset(grad_weight, 0, grad_weight_bytes), "timed pre-reset grad_weight memset");
    cuda_check(cudaDeviceSynchronize(), name + " timed pre-reset synchronize");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");
    cuda_check(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < options.iterations; ++i) {
        int status = 0;
        if constexpr (std::is_same_v<Fn, LinearDinputStridedFn>) {
            status = fn(
                grad_out_bf16,
                weight_bf16,
                grad_x,
                options.rows,
                options.input_dim,
                options.output_dim,
                options.grad_out_row_stride,
                nullptr);
        } else {
            status = fn(
                x_bf16,
                grad_out_bf16,
                grad_weight,
                options.rows,
                options.input_dim,
                options.output_dim,
                options.grad_out_row_stride,
                options.beta,
                nullptr);
        }
        if (status != 0) {
            throw std::runtime_error(name + " timed run returned status " + std::to_string(status));
        }
    }
    cuda_check(cudaEventRecord(stop), "cudaEventRecord stop");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    VariantResult result;
    result.name = name;
    result.symbol = symbol;
    cuda_check(cudaEventElapsedTime(&result.total_ms, start, stop), "cudaEventElapsedTime");
    result.ms_per_iter = static_cast<double>(result.total_ms) / static_cast<double>(options.iterations);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return result;
}

std::string render_json(
    const Options& options,
    const VariantResult& baseline,
    const VariantResult& candidate) {
    const double ratio =
        baseline.ms_per_iter > 0.0 ? candidate.ms_per_iter / baseline.ms_per_iter : 0.0;
    const bool candidate_symbol_changed = baseline.symbol != candidate.symbol;
    auto variant_json = [](const VariantResult& value) {
        std::ostringstream out;
        out << "{"
            << "\"name\":\"" << json_escape(value.name) << "\","
            << "\"symbol\":\"" << json_escape(value.symbol) << "\","
            << "\"total_ms\":" << std::fixed << std::setprecision(6) << value.total_ms << ","
            << "\"ms_per_iter\":" << std::fixed << std::setprecision(6) << value.ms_per_iter
            << "}";
        return out.str();
    };
    std::ostringstream out;
    out << "{\n"
        << "  \"benchmark\": \"linear_backward_tile_ops\",\n"
        << "  \"tile_ops_lib\": \"" << json_escape(options.tile_ops_lib) << "\",\n"
        << "  \"operation\": \"" << json_escape(options.operation) << "\",\n"
        << "  \"rows\": " << options.rows << ",\n"
        << "  \"input_dim\": " << options.input_dim << ",\n"
        << "  \"output_dim\": " << options.output_dim << ",\n"
        << "  \"grad_out_row_stride\": " << options.grad_out_row_stride << ",\n"
        << "  \"iterations\": " << options.iterations << ",\n"
        << "  \"warmup\": " << options.warmup << ",\n"
        << "  \"beta\": " << std::fixed << std::setprecision(6) << options.beta << ",\n"
        << "  \"run_order\": \"" << (options.candidate_first ? "candidate-first" : "baseline-first") << "\",\n"
        << "  \"timed_reset_between_iterations\": false,\n"
        << "  \"candidate_symbol_changed\": " << (candidate_symbol_changed ? "true" : "false") << ",\n"
        << "  \"baseline\": " << variant_json(baseline) << ",\n"
        << "  \"candidate\": " << variant_json(candidate) << ",\n"
        << "  \"candidate_to_baseline_ms_per_iter_ratio\": " << std::fixed << std::setprecision(6) << ratio << "\n"
        << "}\n";
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        cuda_check(cudaSetDevice(options.cuda_device), "cudaSetDevice");
        void* handle = dlopen(options.tile_ops_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            throw std::runtime_error("dlopen failed for " + options.tile_ops_lib + ": " + dlerror());
        }

        const std::size_t x_bytes =
            static_cast<std::size_t>(options.rows * options.input_dim) * sizeof(std::uint16_t);
        const std::size_t grad_out_bytes =
            static_cast<std::size_t>(options.rows * options.grad_out_row_stride) * sizeof(std::uint16_t);
        const std::size_t weight_bytes =
            static_cast<std::size_t>(options.input_dim * options.output_dim) * sizeof(std::uint16_t);
        const std::size_t grad_x_bytes =
            static_cast<std::size_t>(options.rows * options.input_dim) * sizeof(float);
        const std::size_t grad_weight_bytes =
            static_cast<std::size_t>(options.input_dim * options.output_dim) * sizeof(float);

        DeviceBuffer x_bf16(x_bytes);
        DeviceBuffer grad_out_bf16(grad_out_bytes);
        DeviceBuffer weight_bf16(weight_bytes);
        DeviceBuffer grad_x(grad_x_bytes);
        DeviceBuffer grad_weight(grad_weight_bytes);
        cuda_check(cudaMemset(x_bf16.get(), 0, x_bytes), "memset x_bf16");
        cuda_check(cudaMemset(grad_out_bf16.get(), 0, grad_out_bytes), "memset grad_out_bf16");
        cuda_check(cudaMemset(weight_bf16.get(), 0, weight_bytes), "memset weight_bf16");

        VariantResult baseline;
        VariantResult candidate;
        if (options.operation == "dinput-strided") {
            auto baseline_fn = load_symbol<LinearDinputStridedFn>(handle, options.baseline_symbol);
            auto candidate_fn = load_symbol<LinearDinputStridedFn>(handle, options.candidate_symbol);
            if (options.candidate_first) {
                candidate = run_variant(
                    "candidate",
                    options.candidate_symbol,
                    candidate_fn,
                    options,
                    static_cast<const std::uint16_t*>(x_bf16.get()),
                    static_cast<const std::uint16_t*>(grad_out_bf16.get()),
                    static_cast<const std::uint16_t*>(weight_bf16.get()),
                    static_cast<float*>(grad_x.get()),
                    static_cast<float*>(grad_weight.get()));
                baseline = run_variant(
                    "baseline",
                    options.baseline_symbol,
                    baseline_fn,
                    options,
                    static_cast<const std::uint16_t*>(x_bf16.get()),
                    static_cast<const std::uint16_t*>(grad_out_bf16.get()),
                    static_cast<const std::uint16_t*>(weight_bf16.get()),
                    static_cast<float*>(grad_x.get()),
                    static_cast<float*>(grad_weight.get()));
            } else {
                baseline = run_variant(
                    "baseline",
                    options.baseline_symbol,
                    baseline_fn,
                    options,
                    static_cast<const std::uint16_t*>(x_bf16.get()),
                    static_cast<const std::uint16_t*>(grad_out_bf16.get()),
                    static_cast<const std::uint16_t*>(weight_bf16.get()),
                    static_cast<float*>(grad_x.get()),
                    static_cast<float*>(grad_weight.get()));
                candidate = run_variant(
                    "candidate",
                    options.candidate_symbol,
                    candidate_fn,
                    options,
                    static_cast<const std::uint16_t*>(x_bf16.get()),
                    static_cast<const std::uint16_t*>(grad_out_bf16.get()),
                    static_cast<const std::uint16_t*>(weight_bf16.get()),
                    static_cast<float*>(grad_x.get()),
                    static_cast<float*>(grad_weight.get()));
            }
        } else {
            auto baseline_fn = load_symbol<LinearDweightStridedFn>(handle, options.baseline_symbol);
            auto candidate_fn = load_symbol<LinearDweightStridedFn>(handle, options.candidate_symbol);
            if (options.candidate_first) {
                candidate = run_variant(
                    "candidate",
                    options.candidate_symbol,
                    candidate_fn,
                    options,
                    static_cast<const std::uint16_t*>(x_bf16.get()),
                    static_cast<const std::uint16_t*>(grad_out_bf16.get()),
                    static_cast<const std::uint16_t*>(weight_bf16.get()),
                    static_cast<float*>(grad_x.get()),
                    static_cast<float*>(grad_weight.get()));
                baseline = run_variant(
                    "baseline",
                    options.baseline_symbol,
                    baseline_fn,
                    options,
                    static_cast<const std::uint16_t*>(x_bf16.get()),
                    static_cast<const std::uint16_t*>(grad_out_bf16.get()),
                    static_cast<const std::uint16_t*>(weight_bf16.get()),
                    static_cast<float*>(grad_x.get()),
                    static_cast<float*>(grad_weight.get()));
            } else {
                baseline = run_variant(
                    "baseline",
                    options.baseline_symbol,
                    baseline_fn,
                    options,
                    static_cast<const std::uint16_t*>(x_bf16.get()),
                    static_cast<const std::uint16_t*>(grad_out_bf16.get()),
                    static_cast<const std::uint16_t*>(weight_bf16.get()),
                    static_cast<float*>(grad_x.get()),
                    static_cast<float*>(grad_weight.get()));
                candidate = run_variant(
                    "candidate",
                    options.candidate_symbol,
                    candidate_fn,
                    options,
                    static_cast<const std::uint16_t*>(x_bf16.get()),
                    static_cast<const std::uint16_t*>(grad_out_bf16.get()),
                    static_cast<const std::uint16_t*>(weight_bf16.get()),
                    static_cast<float*>(grad_x.get()),
                    static_cast<float*>(grad_weight.get()));
            }
        }

        const std::string json = render_json(options, baseline, candidate);
        if (!options.json_out.empty()) {
            std::filesystem::path out_path(options.json_out);
            if (!out_path.parent_path().empty()) {
                std::filesystem::create_directories(out_path.parent_path());
            }
            std::ofstream file(options.json_out);
            file << json;
        }
        std::cout << json;
        dlclose(handle);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "linear_backward_bench: " << exc.what() << "\n";
        return 2;
    }
}
