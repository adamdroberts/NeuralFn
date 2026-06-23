#include "tile_ops.h"

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using LmHeadBackwardFn = int (*)(
    std::uint16_t*,
    const std::uint16_t*,
    float*,
    const std::uint16_t*,
    const float*,
    const std::uint16_t*,
    const float*,
    float*,
    float*,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    float,
    float,
    int,
    void*);

using IntFn = int (*)();
using VoidFn = void (*)();
using CountFn = std::int64_t (*)();

struct Options {
    std::string tile_ops_lib = "build/libnfn_native_train_tile_ops.so";
    std::string baseline_symbol = "nfn_native_tile_lm_head_classifier_backward_cooperative_bf16_u16";
    std::string candidate_symbol = "nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16";
    std::string json_out;
    int cuda_device = 0;
    std::int64_t rows = 2048;
    std::int64_t hidden_dim = 768;
    std::int64_t vocab = 50257;
    std::int64_t row_stride = 0;
    int iterations = 5;
    int warmup = 1;
    int flags = 0;
};

struct VariantResult {
    std::string name;
    std::string symbol;
    float total_ms = 0.0f;
    double ms_per_iter = 0.0;
    std::int64_t launch_count = 0;
    std::int64_t ce_launch_count = 0;
    std::int64_t dhidden_launch_count = 0;
    std::int64_t dweight_launch_count = 0;
    std::int64_t concurrent_count = 0;
    std::int64_t legacy_count = 0;
    std::int64_t loss_bin_count = 0;
};

[[noreturn]] void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " [options]\n"
        << "  --tile-ops-lib PATH\n"
        << "  --baseline-symbol NAME\n"
        << "  --candidate-symbol NAME\n"
        << "  --rows N\n"
        << "  --hidden-dim N\n"
        << "  --vocab N\n"
        << "  --row-stride N\n"
        << "  --iterations N\n"
        << "  --warmup N\n"
        << "  --loss-bins N\n"
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
        } else if (arg == "--baseline-symbol") {
            options.baseline_symbol = require_value(arg);
        } else if (arg == "--candidate-symbol") {
            options.candidate_symbol = require_value(arg);
        } else if (arg == "--rows") {
            options.rows = parse_i64(require_value(arg), arg);
        } else if (arg == "--hidden-dim") {
            options.hidden_dim = parse_i64(require_value(arg), arg);
        } else if (arg == "--vocab") {
            options.vocab = parse_i64(require_value(arg), arg);
        } else if (arg == "--row-stride") {
            options.row_stride = parse_i64(require_value(arg), arg);
        } else if (arg == "--iterations") {
            options.iterations = static_cast<int>(parse_i64(require_value(arg), arg));
        } else if (arg == "--warmup") {
            options.warmup = static_cast<int>(parse_i64(require_value(arg), arg));
        } else if (arg == "--loss-bins") {
            const int loss_bins = static_cast<int>(parse_i64(require_value(arg), arg));
            if (loss_bins > 0) {
                options.flags = 1 | (loss_bins << 8);
            }
        } else if (arg == "--cuda-device") {
            options.cuda_device = static_cast<int>(parse_i64(require_value(arg), arg));
        } else if (arg == "--json-out") {
            options.json_out = require_value(arg);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.row_stride <= 0) {
        options.row_stride = ((options.vocab + 127) / 128) * 128;
    }
    if (options.rows <= 0 || options.hidden_dim <= 0 || options.vocab <= 0 ||
        options.row_stride < options.vocab || options.iterations <= 0 || options.warmup < 0) {
        throw std::runtime_error("invalid benchmark shape or iteration count");
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

VariantResult run_variant(
    const std::string& name,
    const std::string& symbol,
    LmHeadBackwardFn fn,
    VoidFn reset_stats,
    CountFn launch_count,
    CountFn ce_launch_count,
    CountFn dhidden_launch_count,
    CountFn dweight_launch_count,
    CountFn concurrent_count,
    CountFn legacy_count,
    CountFn loss_bin_count,
    const Options& options,
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* row_losses,
    const std::uint16_t* hidden_bf16,
    const float* hidden_float,
    const std::uint16_t* token_weight_bf16,
    const float* token_weight_float,
    float* grad_hidden,
    float* grad_weight) {
    reset_stats();
    const std::size_t logits_bytes =
        static_cast<std::size_t>(options.rows * options.row_stride) * sizeof(std::uint16_t);
    const std::size_t grad_hidden_bytes =
        static_cast<std::size_t>(options.rows * options.hidden_dim) * sizeof(float);
    const std::size_t grad_weight_bytes =
        static_cast<std::size_t>(options.hidden_dim * options.vocab) * sizeof(float);
    for (int i = 0; i < options.warmup; ++i) {
        cuda_check(cudaMemset(logits, 0, logits_bytes), "warmup logits memset");
        cuda_check(cudaMemset(grad_hidden, 0, grad_hidden_bytes), "warmup grad_hidden memset");
        cuda_check(cudaMemset(grad_weight, 0, grad_weight_bytes), "warmup grad_weight memset");
        const int status = fn(
            logits,
            targets,
            row_losses,
            hidden_bf16,
            hidden_float,
            token_weight_bf16,
            token_weight_float,
            grad_hidden,
            grad_weight,
            options.rows,
            options.hidden_dim,
            options.vocab,
            options.row_stride,
            1.0f / static_cast<float>(options.rows),
            0.0f,
            options.flags,
            nullptr);
        if (status != 0) {
            throw std::runtime_error(name + " warmup returned status " + std::to_string(status));
        }
        cuda_check(cudaDeviceSynchronize(), name + " warmup synchronize");
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");
    cuda_check(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < options.iterations; ++i) {
        cuda_check(cudaMemset(logits, 0, logits_bytes), "timed logits memset");
        cuda_check(cudaMemset(grad_hidden, 0, grad_hidden_bytes), "timed grad_hidden memset");
        cuda_check(cudaMemset(grad_weight, 0, grad_weight_bytes), "timed grad_weight memset");
        const int status = fn(
            logits,
            targets,
            row_losses,
            hidden_bf16,
            hidden_float,
            token_weight_bf16,
            token_weight_float,
            grad_hidden,
            grad_weight,
            options.rows,
            options.hidden_dim,
            options.vocab,
            options.row_stride,
            1.0f / static_cast<float>(options.rows),
            0.0f,
            options.flags,
            nullptr);
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
    result.launch_count = launch_count();
    result.ce_launch_count = ce_launch_count();
    result.dhidden_launch_count = dhidden_launch_count();
    result.dweight_launch_count = dweight_launch_count();
    result.concurrent_count = concurrent_count();
    result.legacy_count = legacy_count();
    result.loss_bin_count = loss_bin_count();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return result;
}

std::string render_json(
    const Options& options,
    bool true_fused_capability,
    const VariantResult& baseline,
    const VariantResult& candidate) {
    const double ratio =
        baseline.ms_per_iter > 0.0 ? candidate.ms_per_iter / baseline.ms_per_iter : 0.0;
    auto variant_json = [](const VariantResult& value) {
        std::ostringstream out;
        out << "{"
            << "\"name\":\"" << json_escape(value.name) << "\","
            << "\"symbol\":\"" << json_escape(value.symbol) << "\","
            << "\"total_ms\":" << std::fixed << std::setprecision(6) << value.total_ms << ","
            << "\"ms_per_iter\":" << std::fixed << std::setprecision(6) << value.ms_per_iter << ","
            << "\"launch_count\":" << value.launch_count << ","
            << "\"ce_launch_count\":" << value.ce_launch_count << ","
            << "\"dhidden_launch_count\":" << value.dhidden_launch_count << ","
            << "\"dweight_launch_count\":" << value.dweight_launch_count << ","
            << "\"concurrent_count\":" << value.concurrent_count << ","
            << "\"legacy_count\":" << value.legacy_count << ","
            << "\"loss_bin_count\":" << value.loss_bin_count
            << "}";
        return out.str();
    };
    std::ostringstream out;
    out << "{\n"
        << "  \"benchmark\": \"lm_head_backward_tile_ops\",\n"
        << "  \"tile_ops_lib\": \"" << json_escape(options.tile_ops_lib) << "\",\n"
        << "  \"rows\": " << options.rows << ",\n"
        << "  \"hidden_dim\": " << options.hidden_dim << ",\n"
        << "  \"vocab\": " << options.vocab << ",\n"
        << "  \"row_stride\": " << options.row_stride << ",\n"
        << "  \"iterations\": " << options.iterations << ",\n"
        << "  \"warmup\": " << options.warmup << ",\n"
        << "  \"flags\": " << options.flags << ",\n"
        << "  \"candidate_true_fused_capability\": " << (true_fused_capability ? "true" : "false") << ",\n"
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

        auto baseline_fn = load_symbol<LmHeadBackwardFn>(handle, options.baseline_symbol);
        auto candidate_fn = load_symbol<LmHeadBackwardFn>(handle, options.candidate_symbol);
        auto true_fused_capability =
            load_symbol<IntFn>(handle, "nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused");
        auto reset_stats = load_symbol<VoidFn>(handle, "nfn_native_tile_lm_head_classifier_stats_reset");
        auto launch_count =
            load_symbol<CountFn>(handle, "nfn_native_tile_lm_head_cooperative_sequence_launch_count");
        auto ce_launch_count =
            load_symbol<CountFn>(handle, "nfn_native_tile_lm_head_cooperative_sequence_ce_launch_count");
        auto dhidden_launch_count =
            load_symbol<CountFn>(handle, "nfn_native_tile_lm_head_cooperative_sequence_dhidden_launch_count");
        auto dweight_launch_count =
            load_symbol<CountFn>(handle, "nfn_native_tile_lm_head_cooperative_sequence_dweight_launch_count");
        auto concurrent_count =
            load_symbol<CountFn>(handle, "nfn_native_tile_lm_head_cooperative_sequence_concurrent_count");
        auto legacy_count =
            load_symbol<CountFn>(handle, "nfn_native_tile_lm_head_cooperative_sequence_legacy_count");
        auto loss_bin_count =
            load_symbol<CountFn>(handle, "nfn_native_tile_lm_head_cooperative_sequence_loss_bin_count");

        const std::size_t logits_bytes =
            static_cast<std::size_t>(options.rows * options.row_stride) * sizeof(std::uint16_t);
        const std::size_t target_bytes = static_cast<std::size_t>(options.rows) * sizeof(std::uint16_t);
        const std::size_t row_losses_bytes = static_cast<std::size_t>(options.rows) * sizeof(float);
        const std::size_t hidden_bf16_bytes =
            static_cast<std::size_t>(options.rows * options.hidden_dim) * sizeof(std::uint16_t);
        const std::size_t hidden_float_bytes =
            static_cast<std::size_t>(options.rows * options.hidden_dim) * sizeof(float);
        const std::size_t token_weight_bf16_bytes =
            static_cast<std::size_t>(options.hidden_dim * options.vocab) * sizeof(std::uint16_t);
        const std::size_t token_weight_float_bytes =
            static_cast<std::size_t>(options.hidden_dim * options.vocab) * sizeof(float);
        const std::size_t grad_hidden_bytes =
            static_cast<std::size_t>(options.rows * options.hidden_dim) * sizeof(float);
        const std::size_t grad_weight_bytes =
            static_cast<std::size_t>(options.hidden_dim * options.vocab) * sizeof(float);

        DeviceBuffer logits(logits_bytes);
        DeviceBuffer targets(target_bytes);
        DeviceBuffer row_losses(row_losses_bytes);
        DeviceBuffer hidden_bf16(hidden_bf16_bytes);
        DeviceBuffer hidden_float(hidden_float_bytes);
        DeviceBuffer token_weight_bf16(token_weight_bf16_bytes);
        DeviceBuffer token_weight_float(token_weight_float_bytes);
        DeviceBuffer grad_hidden(grad_hidden_bytes);
        DeviceBuffer grad_weight(grad_weight_bytes);

        std::vector<std::uint16_t> host_targets(static_cast<std::size_t>(options.rows));
        for (std::int64_t row = 0; row < options.rows; ++row) {
            host_targets[static_cast<std::size_t>(row)] = static_cast<std::uint16_t>(row % options.vocab);
        }
        cuda_check(cudaMemcpy(targets.get(), host_targets.data(), target_bytes, cudaMemcpyHostToDevice),
                   "copy targets");
        cuda_check(cudaMemset(hidden_bf16.get(), 0, hidden_bf16_bytes), "memset hidden_bf16");
        cuda_check(cudaMemset(hidden_float.get(), 0, hidden_float_bytes), "memset hidden_float");
        cuda_check(cudaMemset(token_weight_bf16.get(), 0, token_weight_bf16_bytes), "memset token_weight_bf16");
        cuda_check(cudaMemset(token_weight_float.get(), 0, token_weight_float_bytes), "memset token_weight_float");

        const VariantResult baseline = run_variant(
            "baseline",
            options.baseline_symbol,
            baseline_fn,
            reset_stats,
            launch_count,
            ce_launch_count,
            dhidden_launch_count,
            dweight_launch_count,
            concurrent_count,
            legacy_count,
            loss_bin_count,
            options,
            static_cast<std::uint16_t*>(logits.get()),
            static_cast<const std::uint16_t*>(targets.get()),
            static_cast<float*>(row_losses.get()),
            static_cast<const std::uint16_t*>(hidden_bf16.get()),
            static_cast<const float*>(hidden_float.get()),
            static_cast<const std::uint16_t*>(token_weight_bf16.get()),
            static_cast<const float*>(token_weight_float.get()),
            static_cast<float*>(grad_hidden.get()),
            static_cast<float*>(grad_weight.get()));
        const VariantResult candidate = run_variant(
            "candidate",
            options.candidate_symbol,
            candidate_fn,
            reset_stats,
            launch_count,
            ce_launch_count,
            dhidden_launch_count,
            dweight_launch_count,
            concurrent_count,
            legacy_count,
            loss_bin_count,
            options,
            static_cast<std::uint16_t*>(logits.get()),
            static_cast<const std::uint16_t*>(targets.get()),
            static_cast<float*>(row_losses.get()),
            static_cast<const std::uint16_t*>(hidden_bf16.get()),
            static_cast<const float*>(hidden_float.get()),
            static_cast<const std::uint16_t*>(token_weight_bf16.get()),
            static_cast<const float*>(token_weight_float.get()),
            static_cast<float*>(grad_hidden.get()),
            static_cast<float*>(grad_weight.get()));

        const std::string json = render_json(options, true_fused_capability() != 0, baseline, candidate);
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
        std::cerr << "lm_head_backward_bench: " << exc.what() << "\n";
        return 2;
    }
}
