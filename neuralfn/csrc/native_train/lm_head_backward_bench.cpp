#include "tile_ops.h"

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
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
using LmHeadCeRowLossFn = int (*)(
    std::uint16_t*,
    const std::uint16_t*,
    float*,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    float,
    void*);
using LmHeadCeNoLossFn = int (*)(
    std::uint16_t*,
    const std::uint16_t*,
    float*,
    float*,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    float,
    void*);
using LinearBackwardInputStridedFn = int (*)(
    const std::uint16_t*,
    const std::uint16_t*,
    float*,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    void*);
using LinearBackwardWeightStridedBetaFn = int (*)(
    const std::uint16_t*,
    const std::uint16_t*,
    float*,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    float,
    void*);
using LinearBf16InputWeightBf16OutputFn = int (*)(
    const std::uint16_t*,
    const std::uint16_t*,
    const float*,
    std::uint16_t*,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    bool,
    void*);

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
    bool no_loss = false;
    bool candidate_first = false;
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
    std::int64_t graph_capture_attempt_count = 0;
    std::int64_t graph_capture_success_count = 0;
    std::int64_t graph_cache_hit_count = 0;
    std::int64_t graph_cache_entry_count = 0;
    std::int64_t graph_replay_count = 0;
    std::int64_t graph_replay_success_count = 0;
    std::int64_t graph_fallback_count = 0;
};

struct ComponentResult {
    double logits_ms_per_iter = 0.0;
    double ce_ms_per_iter = 0.0;
    double dhidden_ms_per_iter = 0.0;
    double dweight_ms_per_iter = 0.0;
    double summed_ms_per_iter = 0.0;
    double summed_with_logits_ms_per_iter = 0.0;
};

[[noreturn]] void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " [options]\n"
        << "  --tile-ops-lib PATH\n"
        << "  --baseline-symbol NAME\n"
        << "  --candidate-symbol NAME\n"
        << "     e.g. nfn_native_tile_lm_head_classifier_backward_cooperative_cublaslt_bf16_u16\n"
        << "  --rows N\n"
        << "  --hidden-dim N\n"
        << "  --vocab N\n"
        << "  --row-stride N\n"
        << "  --iterations N\n"
        << "  --warmup N\n"
        << "  --no-loss\n"
        << "  --loss-bins N\n"
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
        } else if (arg == "--no-loss") {
            if (options.flags != 0) {
                throw std::runtime_error("--no-loss cannot be combined with --loss-bins");
            }
            options.no_loss = true;
            options.flags = 1 << 1;
        } else if (arg == "--loss-bins") {
            const int loss_bins = static_cast<int>(parse_i64(require_value(arg), arg));
            if (options.no_loss && loss_bins > 0) {
                throw std::runtime_error("--loss-bins cannot be combined with --no-loss");
            }
            if (loss_bins > 0) {
                options.flags = 1 | (loss_bins << 8);
            }
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
    CountFn graph_capture_attempt_count,
    CountFn graph_capture_success_count,
    CountFn graph_cache_hit_count,
    CountFn graph_cache_entry_count,
    CountFn graph_replay_count,
    CountFn graph_replay_success_count,
    CountFn graph_fallback_count,
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
    reset_stats();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_check(cudaMemset(logits, 0, logits_bytes), "timed pre-reset logits memset");
    cuda_check(cudaMemset(grad_hidden, 0, grad_hidden_bytes), "timed pre-reset grad_hidden memset");
    cuda_check(cudaMemset(grad_weight, 0, grad_weight_bytes), "timed pre-reset grad_weight memset");
    cuda_check(cudaDeviceSynchronize(), name + " timed pre-reset synchronize");
    cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");
    cuda_check(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < options.iterations; ++i) {
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
    result.graph_capture_attempt_count = graph_capture_attempt_count();
    result.graph_capture_success_count = graph_capture_success_count();
    result.graph_cache_hit_count = graph_cache_hit_count();
    result.graph_cache_entry_count = graph_cache_entry_count();
    result.graph_replay_count = graph_replay_count();
    result.graph_replay_success_count = graph_replay_success_count();
    result.graph_fallback_count = graph_fallback_count();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return result;
}

double time_component(
    const std::string& label,
    int iterations,
    int warmup,
    const std::function<void()>& prepare,
    const std::function<int()>& launch) {
    for (int i = 0; i < warmup; ++i) {
        prepare();
        cuda_check(cudaDeviceSynchronize(), label + " warmup prepare synchronize");
        const int status = launch();
        if (status != 0) {
            throw std::runtime_error(label + " warmup returned status " + std::to_string(status));
        }
        cuda_check(cudaDeviceSynchronize(), label + " warmup synchronize");
    }
    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        prepare();
        cuda_check(cudaDeviceSynchronize(), label + " prepare synchronize");
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        cuda_check(cudaEventCreate(&start), label + " cudaEventCreate start");
        cuda_check(cudaEventCreate(&stop), label + " cudaEventCreate stop");
        cuda_check(cudaEventRecord(start), label + " cudaEventRecord start");
        const int status = launch();
        if (status != 0) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            throw std::runtime_error(label + " returned status " + std::to_string(status));
        }
        cuda_check(cudaEventRecord(stop), label + " cudaEventRecord stop");
        cuda_check(cudaEventSynchronize(stop), label + " cudaEventSynchronize stop");
        float elapsed_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), label + " cudaEventElapsedTime");
        total_ms += static_cast<double>(elapsed_ms);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    return iterations > 0 ? total_ms / static_cast<double>(iterations) : 0.0;
}

ComponentResult run_reference_components(
    LinearBf16InputWeightBf16OutputFn logits_fn,
    LmHeadCeRowLossFn ce_fn,
    LmHeadCeNoLossFn ce_no_loss_fn,
    LinearBackwardInputStridedFn dhidden_fn,
    LinearBackwardWeightStridedBetaFn dweight_fn,
    const Options& options,
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* row_losses,
    const std::uint16_t* hidden_bf16,
    const std::uint16_t* token_weight_bf16,
    float* grad_hidden,
    float* grad_weight) {
    const std::size_t logits_bytes =
        static_cast<std::size_t>(options.rows * options.row_stride) * sizeof(std::uint16_t);
    const std::size_t grad_hidden_bytes =
        static_cast<std::size_t>(options.rows * options.hidden_dim) * sizeof(float);
    const std::size_t grad_weight_bytes =
        static_cast<std::size_t>(options.hidden_dim * options.vocab) * sizeof(float);
    const float loss_scale = 1.0f / static_cast<float>(options.rows);
    ComponentResult result;
    result.logits_ms_per_iter = time_component(
        "reference.logits",
        options.iterations,
        options.warmup,
        [&]() {
            cuda_check(cudaMemset(logits, 0, logits_bytes), "reference logits output memset");
        },
        [&]() {
            return logits_fn(
                hidden_bf16,
                token_weight_bf16,
                nullptr,
                logits,
                options.rows,
                options.hidden_dim,
                options.row_stride,
                false,
                nullptr);
        });
    result.ce_ms_per_iter = time_component(
        "reference.ce_row_losses",
        options.iterations,
        options.warmup,
        [&]() {
            cuda_check(cudaMemset(logits, 0, logits_bytes), "reference ce logits memset");
        },
        [&]() {
            if (options.no_loss) {
                return ce_no_loss_fn(
                    logits,
                    targets,
                    nullptr,
                    nullptr,
                    options.rows,
                    options.vocab,
                    options.row_stride,
                    loss_scale,
                    nullptr);
            }
            return ce_fn(
                logits,
                targets,
                row_losses,
                options.rows,
                options.vocab,
                options.row_stride,
                loss_scale,
                nullptr);
        });
    cuda_check(cudaMemset(logits, 0, logits_bytes), "reference dlogits logits memset");
    const int ce_status =
        options.no_loss
            ? ce_no_loss_fn(
                  logits,
                  targets,
                  nullptr,
                  nullptr,
                  options.rows,
                  options.vocab,
                  options.row_stride,
                  loss_scale,
                  nullptr)
            : ce_fn(
                  logits,
                  targets,
                  row_losses,
                  options.rows,
                  options.vocab,
                  options.row_stride,
                  loss_scale,
                  nullptr);
    if (ce_status != 0) {
        throw std::runtime_error("reference dlogits preparation returned status " + std::to_string(ce_status));
    }
    cuda_check(cudaDeviceSynchronize(), "reference dlogits preparation synchronize");
    result.dhidden_ms_per_iter = time_component(
        "reference.dhidden",
        options.iterations,
        options.warmup,
        [&]() {
            cuda_check(cudaMemset(grad_hidden, 0, grad_hidden_bytes), "reference dhidden grad memset");
        },
        [&]() {
            return dhidden_fn(
                logits,
                token_weight_bf16,
                grad_hidden,
                options.rows,
                options.hidden_dim,
                options.vocab,
                options.row_stride,
                nullptr);
        });
    result.dweight_ms_per_iter = time_component(
        "reference.dweight",
        options.iterations,
        options.warmup,
        [&]() {
            cuda_check(cudaMemset(grad_weight, 0, grad_weight_bytes), "reference dweight grad memset");
        },
        [&]() {
            return dweight_fn(
                hidden_bf16,
                logits,
                grad_weight,
                options.rows,
                options.hidden_dim,
                options.vocab,
                options.row_stride,
                0.0f,
                nullptr);
        });
    result.summed_ms_per_iter =
        result.ce_ms_per_iter + result.dhidden_ms_per_iter + result.dweight_ms_per_iter;
    result.summed_with_logits_ms_per_iter =
        result.logits_ms_per_iter + result.summed_ms_per_iter;
    return result;
}

std::string render_json(
    const Options& options,
    bool true_fused_capability,
    const VariantResult& baseline,
    const VariantResult& candidate,
    const ComponentResult& reference_components,
    const ComponentResult& reference_cublaslt_components) {
    const double ratio =
        baseline.ms_per_iter > 0.0 ? candidate.ms_per_iter / baseline.ms_per_iter : 0.0;
    const double candidate_reference_ratio =
        reference_components.summed_ms_per_iter > 0.0
            ? candidate.ms_per_iter / reference_components.summed_ms_per_iter
            : 0.0;
    const double candidate_reference_with_logits_ratio =
        reference_components.summed_with_logits_ms_per_iter > 0.0
            ? candidate.ms_per_iter / reference_components.summed_with_logits_ms_per_iter
            : 0.0;
    const double candidate_cublaslt_reference_ratio =
        reference_cublaslt_components.summed_ms_per_iter > 0.0
            ? candidate.ms_per_iter / reference_cublaslt_components.summed_ms_per_iter
            : 0.0;
    const double candidate_cublaslt_reference_with_logits_ratio =
        reference_cublaslt_components.summed_with_logits_ms_per_iter > 0.0
            ? candidate.ms_per_iter / reference_cublaslt_components.summed_with_logits_ms_per_iter
            : 0.0;
    const bool candidate_sequence_wrapper_only =
        !true_fused_capability &&
        candidate.ce_launch_count > 0 &&
        candidate.dhidden_launch_count > 0 &&
        candidate.dweight_launch_count > 0;
    const bool candidate_strict_symbol_is_placeholder_sequence =
        candidate_sequence_wrapper_only &&
        candidate.symbol == "nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16";
    const bool candidate_cuda_graph_wrapper_only =
        !true_fused_capability &&
        candidate.graph_replay_count > 0 &&
        candidate.graph_replay_success_count > 0 &&
        candidate.ce_launch_count == 0 &&
        candidate.dhidden_launch_count == 0 &&
        candidate.dweight_launch_count == 0;
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
            << "\"loss_bin_count\":" << value.loss_bin_count << ","
            << "\"graph_capture_attempt_count\":" << value.graph_capture_attempt_count << ","
            << "\"graph_capture_success_count\":" << value.graph_capture_success_count << ","
            << "\"graph_cache_hit_count\":" << value.graph_cache_hit_count << ","
            << "\"graph_cache_entry_count\":" << value.graph_cache_entry_count << ","
            << "\"graph_replay_count\":" << value.graph_replay_count << ","
            << "\"graph_replay_success_count\":" << value.graph_replay_success_count << ","
            << "\"graph_fallback_count\":" << value.graph_fallback_count
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
        << "  \"reference_component_warmup\": " << options.warmup << ",\n"
        << "  \"no_loss\": " << (options.no_loss ? "true" : "false") << ",\n"
        << "  \"flags\": " << options.flags << ",\n"
        << "  \"run_order\": \"" << (options.candidate_first ? "candidate-first" : "baseline-first") << "\",\n"
        << "  \"timed_reset_between_iterations\": false,\n"
        << "  \"candidate_true_fused_capability\": " << (true_fused_capability ? "true" : "false") << ",\n"
        << "  \"candidate_sequence_wrapper_only\": " << (candidate_sequence_wrapper_only ? "true" : "false") << ",\n"
        << "  \"candidate_strict_symbol_is_placeholder_sequence\": " << (candidate_strict_symbol_is_placeholder_sequence ? "true" : "false") << ",\n"
        << "  \"candidate_cuda_graph_wrapper_only\": " << (candidate_cuda_graph_wrapper_only ? "true" : "false") << ",\n"
        << "  \"reference_components\": {"
        << "\"logits_ms_per_iter\":" << std::fixed << std::setprecision(6) << reference_components.logits_ms_per_iter << ","
        << "\"ce_ms_per_iter\":" << std::fixed << std::setprecision(6) << reference_components.ce_ms_per_iter << ","
        << "\"dhidden_ms_per_iter\":" << std::fixed << std::setprecision(6) << reference_components.dhidden_ms_per_iter << ","
        << "\"dweight_ms_per_iter\":" << std::fixed << std::setprecision(6) << reference_components.dweight_ms_per_iter << ","
        << "\"summed_ms_per_iter\":" << std::fixed << std::setprecision(6) << reference_components.summed_ms_per_iter << ","
        << "\"summed_with_logits_ms_per_iter\":" << std::fixed << std::setprecision(6) << reference_components.summed_with_logits_ms_per_iter
        << "},\n"
        << "  \"reference_cublaslt_components\": {"
        << "\"logits_ms_per_iter\":" << std::fixed << std::setprecision(6) << reference_cublaslt_components.logits_ms_per_iter << ","
        << "\"ce_ms_per_iter\":" << std::fixed << std::setprecision(6) << reference_cublaslt_components.ce_ms_per_iter << ","
        << "\"dhidden_ms_per_iter\":" << std::fixed << std::setprecision(6) << reference_cublaslt_components.dhidden_ms_per_iter << ","
        << "\"dweight_ms_per_iter\":" << std::fixed << std::setprecision(6) << reference_cublaslt_components.dweight_ms_per_iter << ","
        << "\"summed_ms_per_iter\":" << std::fixed << std::setprecision(6) << reference_cublaslt_components.summed_ms_per_iter << ","
        << "\"summed_with_logits_ms_per_iter\":" << std::fixed << std::setprecision(6) << reference_cublaslt_components.summed_with_logits_ms_per_iter
        << "},\n"
        << "  \"baseline\": " << variant_json(baseline) << ",\n"
        << "  \"candidate\": " << variant_json(candidate) << ",\n"
        << "  \"candidate_to_baseline_ms_per_iter_ratio\": " << std::fixed << std::setprecision(6) << ratio << ",\n"
        << "  \"candidate_to_reference_summed_ms_per_iter_ratio\": "
        << std::fixed << std::setprecision(6) << candidate_reference_ratio << ",\n"
        << "  \"candidate_to_reference_summed_with_logits_ms_per_iter_ratio\": "
        << std::fixed << std::setprecision(6) << candidate_reference_with_logits_ratio << ",\n"
        << "  \"candidate_to_reference_cublaslt_summed_ms_per_iter_ratio\": "
        << std::fixed << std::setprecision(6) << candidate_cublaslt_reference_ratio << ",\n"
        << "  \"candidate_to_reference_cublaslt_summed_with_logits_ms_per_iter_ratio\": "
        << std::fixed << std::setprecision(6) << candidate_cublaslt_reference_with_logits_ratio << "\n"
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
        auto graph_capture_attempt_count =
            load_symbol<CountFn>(handle, "nfn_native_tile_lm_head_fused_graph_capture_attempt_count");
        auto graph_capture_success_count =
            load_symbol<CountFn>(handle, "nfn_native_tile_lm_head_fused_graph_capture_success_count");
        auto graph_cache_hit_count =
            load_symbol<CountFn>(handle, "nfn_native_tile_lm_head_fused_graph_cache_hit_count");
        auto graph_cache_entry_count =
            load_symbol<CountFn>(handle, "nfn_native_tile_lm_head_fused_graph_cache_entry_count");
        auto graph_replay_count =
            load_symbol<CountFn>(handle, "nfn_native_tile_lm_head_fused_graph_replay_count");
        auto graph_replay_success_count =
            load_symbol<CountFn>(handle, "nfn_native_tile_lm_head_fused_graph_replay_success_count");
        auto graph_fallback_count =
            load_symbol<CountFn>(handle, "nfn_native_tile_lm_head_fused_graph_fallback_count");
        auto reference_logits_fn = load_symbol<LinearBf16InputWeightBf16OutputFn>(
            handle,
            "nfn_native_tile_linear_bf16_input_weight_bf16_output_float32");
        auto reference_ce_fn = load_symbol<LmHeadCeRowLossFn>(
            handle,
            "nfn_native_tile_lm_head_classifier_backward_row_losses_inplace_strided_no_pad_zero_bf16_bits_u16_targets");
        auto reference_ce_no_loss_fn = load_symbol<LmHeadCeNoLossFn>(
            handle,
            "nfn_native_tile_lm_head_classifier_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace");
        auto reference_dhidden_fn = load_symbol<LinearBackwardInputStridedFn>(
            handle,
            "nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_strided_float32");
        auto reference_dweight_fn = load_symbol<LinearBackwardWeightStridedBetaFn>(
            handle,
            "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta");
        auto reference_cublaslt_dhidden_fn = load_symbol<LinearBackwardInputStridedFn>(
            handle,
            "nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_strided_cublaslt_float32");
        auto reference_cublaslt_dweight_fn = load_symbol<LinearBackwardWeightStridedBetaFn>(
            handle,
            "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_cublaslt_float32_beta");

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

        auto run_named_variant = [&](const char* name, const std::string& symbol, LmHeadBackwardFn fn) {
            return run_variant(
                name,
                symbol,
                fn,
                reset_stats,
                launch_count,
                ce_launch_count,
                dhidden_launch_count,
                dweight_launch_count,
                concurrent_count,
                legacy_count,
                loss_bin_count,
                graph_capture_attempt_count,
                graph_capture_success_count,
                graph_cache_hit_count,
                graph_cache_entry_count,
                graph_replay_count,
                graph_replay_success_count,
                graph_fallback_count,
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
        };
        VariantResult baseline;
        VariantResult candidate;
        if (options.candidate_first) {
            candidate = run_named_variant("candidate", options.candidate_symbol, candidate_fn);
            baseline = run_named_variant("baseline", options.baseline_symbol, baseline_fn);
        } else {
            baseline = run_named_variant("baseline", options.baseline_symbol, baseline_fn);
            candidate = run_named_variant("candidate", options.candidate_symbol, candidate_fn);
        }
        const ComponentResult reference_components = run_reference_components(
            reference_logits_fn,
            reference_ce_fn,
            reference_ce_no_loss_fn,
            reference_dhidden_fn,
            reference_dweight_fn,
            options,
            static_cast<std::uint16_t*>(logits.get()),
            static_cast<const std::uint16_t*>(targets.get()),
            static_cast<float*>(row_losses.get()),
            static_cast<const std::uint16_t*>(hidden_bf16.get()),
            static_cast<const std::uint16_t*>(token_weight_bf16.get()),
            static_cast<float*>(grad_hidden.get()),
            static_cast<float*>(grad_weight.get()));
        const ComponentResult reference_cublaslt_components = run_reference_components(
            reference_logits_fn,
            reference_ce_fn,
            reference_ce_no_loss_fn,
            reference_cublaslt_dhidden_fn,
            reference_cublaslt_dweight_fn,
            options,
            static_cast<std::uint16_t*>(logits.get()),
            static_cast<const std::uint16_t*>(targets.get()),
            static_cast<float*>(row_losses.get()),
            static_cast<const std::uint16_t*>(hidden_bf16.get()),
            static_cast<const std::uint16_t*>(token_weight_bf16.get()),
            static_cast<float*>(grad_hidden.get()),
            static_cast<float*>(grad_weight.get()));

        const std::string json =
            render_json(
                options,
                true_fused_capability() != 0,
                baseline,
                candidate,
                reference_components,
                reference_cublaslt_components);
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
