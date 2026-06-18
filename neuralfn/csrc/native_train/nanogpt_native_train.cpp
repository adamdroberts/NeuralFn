#include "token_shards.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct NanoGptPlan {
    std::string dataset_alias = "roneneldan__TinyStories__TinyStoriesV2-GPT4";
    std::string output = "artifacts/nanogpt.bin";
    std::string optimizer_profile = "adamw";
    std::int64_t max_steps = 20000;
    std::int64_t train_seq_len = 1024;
    std::int64_t batch_size = 64;
    std::int64_t train_batch_tokens = 524288;
    std::int64_t eval_batches = 20;
    std::int64_t eval_batch_size = 64;
    std::int64_t eval_every_steps = 250;
    std::int64_t warmup_steps = 60;
    std::int64_t vocab_size = 50257;
    std::int64_t num_layers = 5;
    std::int64_t model_dim = 320;
    std::int64_t num_heads = 5;
    bool bias = false;
    double dropout_p = 0.0;
    double learning_rate = 0.0006;
    double weight_decay = 0.1;
    double beta1 = 0.9;
    double beta2 = 0.95;
    double adam_eps = 1e-8;
    double grad_clip_norm = 1.0;
    bool allow_train_as_val = false;
    bool require_token_shards = false;
    bool sample_token_batch = false;
    bool check_tile_ops = false;
    bool smoke_tile_ops = false;
    bool smoke_optimizer_step = false;
    bool smoke_training_loop_step = false;
    bool smoke_lm_step = false;
    bool smoke_token_train_step = false;
    bool smoke_embedding_norm_step = false;
    bool smoke_qkv_layout_step = false;
    bool smoke_fused_qkv_attention_step = false;
    bool smoke_transformer_block_step = false;
    bool smoke_mlp_step = false;
    bool smoke_attention_step = false;
    bool train_token_lm = false;
    std::string tile_ops_lib;
    std::string cuda_runtime_lib;
    std::vector<std::string> unparsed_args;
};

struct ParameterBuffer {
    std::string name;
    std::vector<std::int64_t> shape;
    std::int64_t offset = 0;
    std::int64_t count = 0;
    bool weight_decay = true;
};

struct TrainingStage {
    std::string name;
    std::string phase;
    std::string status;
    std::string kernel_abi;
    std::int64_t elements = 0;
};

struct ValidationLossRecord {
    std::int64_t step = 0;
    std::int64_t batches = 0;
    std::int64_t tokens = 0;
    double loss_sum = 0.0;
    double loss_mean = 0.0;
};

struct TileOpsSymbolCheck {
    std::string name;
    bool found = false;
};

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    for (char ch : value) {
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
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                out << ch;
        }
    }
    return out.str();
}

std::string require_value(int argc, char** argv, int* index, const std::string& flag) {
    if (*index + 1 >= argc) {
        std::cerr << flag << " requires a value\n";
        std::exit(2);
    }
    *index += 1;
    return argv[*index];
}

std::int64_t parse_i64(const std::string& value, const std::string& flag) {
    try {
        std::size_t pos = 0;
        long long parsed = std::stoll(value, &pos, 10);
        if (pos != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return static_cast<std::int64_t>(parsed);
    } catch (const std::exception&) {
        std::cerr << flag << " expects an integer, got '" << value << "'\n";
        std::exit(2);
    }
}

double parse_f64(const std::string& value, const std::string& flag) {
    try {
        std::size_t pos = 0;
        double parsed = std::stod(value, &pos);
        if (pos != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return parsed;
    } catch (const std::exception&) {
        std::cerr << flag << " expects a number, got '" << value << "'\n";
        std::exit(2);
    }
}

std::int64_t ceil_div(std::int64_t lhs, std::int64_t rhs) {
    return (lhs + rhs - 1) / rhs;
}

std::int64_t shape_count(const std::vector<std::int64_t>& shape) {
    std::int64_t total = 1;
    for (const std::int64_t dim : shape) {
        total *= dim;
    }
    return total;
}

std::vector<ParameterBuffer> build_parameter_layout(const NanoGptPlan& plan) {
    std::vector<ParameterBuffer> layout;
    std::int64_t offset = 0;
    const std::int64_t dim = plan.model_dim;
    const std::int64_t hidden = dim * 4;

    auto add = [&](std::string name, std::vector<std::int64_t> shape, bool weight_decay) {
        ParameterBuffer buffer;
        buffer.name = std::move(name);
        buffer.shape = std::move(shape);
        buffer.offset = offset;
        buffer.count = shape_count(buffer.shape);
        buffer.weight_decay = weight_decay;
        offset += buffer.count;
        layout.push_back(std::move(buffer));
    };

    add("tok_emb.weight", {plan.vocab_size, dim}, true);
    add("pos_emb.weight", {plan.train_seq_len, dim}, true);
    for (std::int64_t layer = 0; layer < plan.num_layers; ++layer) {
        const std::string prefix = "blocks." + std::to_string(layer);
        add(prefix + ".attn.qkv.weight", {3 * dim, dim}, true);
        if (plan.bias) {
            add(prefix + ".attn.qkv.bias", {3 * dim}, false);
        }
        add(prefix + ".attn.out.weight", {dim, dim}, true);
        if (plan.bias) {
            add(prefix + ".attn.out.bias", {dim}, false);
        }
        add(prefix + ".mlp.fc.weight", {hidden, dim}, true);
        if (plan.bias) {
            add(prefix + ".mlp.fc.bias", {hidden}, false);
        }
        add(prefix + ".mlp.proj.weight", {dim, hidden}, true);
        if (plan.bias) {
            add(prefix + ".mlp.proj.bias", {dim}, false);
        }
        add(prefix + ".ln1.weight", {dim}, false);
        add(prefix + ".ln1.bias", {dim}, false);
        add(prefix + ".ln2.weight", {dim}, false);
        add(prefix + ".ln2.bias", {dim}, false);
    }
    add("ln_f.weight", {dim}, false);
    return layout;
}

std::int64_t parameter_layout_count(const std::vector<ParameterBuffer>& layout) {
    if (layout.empty()) {
        return 0;
    }
    const ParameterBuffer& last = layout.back();
    return last.offset + last.count;
}

std::int64_t estimate_parameter_count(const NanoGptPlan& plan) {
    return parameter_layout_count(build_parameter_layout(plan));
}

std::string shape_json(const std::vector<std::int64_t>& shape) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << shape[i];
    }
    out << "]";
    return out.str();
}

std::string parameter_layout_json(const std::vector<ParameterBuffer>& layout) {
    const std::int64_t total = parameter_layout_count(layout);
    std::ostringstream out;
    out << "{\n"
        << "    \"total_parameters\": " << total << ",\n"
        << "    \"buffer_count\": " << layout.size() << ",\n"
        << "    \"parameter_dtype\": \"float32\",\n"
        << "    \"gradient_dtype\": \"float32\",\n"
        << "    \"optimizer_state_dtype\": \"float32\",\n"
        << "    \"required_device_buffers\": {\n"
        << "      \"parameters\": " << total << ",\n"
        << "      \"gradients\": " << total << ",\n"
        << "      \"adamw_exp_avg\": " << total << ",\n"
        << "      \"adamw_exp_avg_sq\": " << total << ",\n"
        << "      \"clip_scale\": 1\n"
        << "    },\n"
        << "    \"buffers\": [\n";
    for (std::size_t i = 0; i < layout.size(); ++i) {
        const ParameterBuffer& buffer = layout[i];
        out << "      {\"name\": \"" << json_escape(buffer.name) << "\", "
            << "\"shape\": " << shape_json(buffer.shape) << ", "
            << "\"offset\": " << buffer.offset << ", "
            << "\"count\": " << buffer.count << ", "
            << "\"weight_decay\": " << (buffer.weight_decay ? "true" : "false") << "}";
        if (i + 1 != layout.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "    ]\n"
        << "  }";
    return out.str();
}

std::string optimizer_groups_json(const NanoGptPlan& plan, const std::vector<ParameterBuffer>& layout) {
    std::vector<std::size_t> decay_indices;
    std::vector<std::size_t> no_decay_indices;
    std::int64_t decay_count = 0;
    std::int64_t no_decay_count = 0;
    for (std::size_t i = 0; i < layout.size(); ++i) {
        if (layout[i].weight_decay) {
            decay_indices.push_back(i);
            decay_count += layout[i].count;
        } else {
            no_decay_indices.push_back(i);
            no_decay_count += layout[i].count;
        }
    }
    auto indices_json = [](const std::vector<std::size_t>& indices) {
        std::ostringstream out;
        out << "[";
        for (std::size_t i = 0; i < indices.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            out << indices[i];
        }
        out << "]";
        return out.str();
    };
    std::ostringstream out;
    out << "{\n"
        << "    \"groups\": [\n"
        << "      {\"name\": \"decay\", \"weight_decay\": " << plan.weight_decay
        << ", \"buffer_indices\": " << indices_json(decay_indices)
        << ", \"total_elements\": " << decay_count << "},\n"
        << "      {\"name\": \"no_decay\", \"weight_decay\": 0"
        << ", \"buffer_indices\": " << indices_json(no_decay_indices)
        << ", \"total_elements\": " << no_decay_count << "}\n"
        << "    ],\n"
        << "    \"adamw_step_abi\": \"nfn_native_tile_adamw_step_float32\"\n"
        << "  }";
    return out.str();
}

std::vector<std::string> required_tile_ops_symbols() {
    return {
        "nfn_native_tile_ops_abi_version",
        "nfn_native_tile_fill_float32",
        "nfn_native_tile_gradient_accumulate_float32",
        "nfn_native_tile_sumsq_partials_float32",
        "nfn_native_tile_sum_partials_float32",
        "nfn_native_tile_scale_inplace_float32",
        "nfn_native_tile_global_norm_clip_scale_float32",
        "nfn_native_tile_scale_inplace_by_device_float32",
        "nfn_native_tile_scaled_residual_add_float32",
        "nfn_native_tile_split_qkv_float32",
        "nfn_native_tile_merge_qkv_float32",
        "nfn_native_tile_adamw_step_float32",
        "nfn_native_tile_linear_float32",
        "nfn_native_tile_linear_backward_input_float32",
        "nfn_native_tile_linear_backward_weight_float32",
        "nfn_native_tile_linear_backward_bias_float32",
        "nfn_native_tile_gelu_float32",
        "nfn_native_tile_gelu_backward_float32",
        "nfn_native_tile_dropout_forward_float32",
        "nfn_native_tile_dropout_backward_float32",
        "nfn_native_tile_absolute_position_embedding_float32",
        "nfn_native_tile_absolute_position_embedding_backward_float32",
        "nfn_native_tile_token_embedding_float32",
        "nfn_native_tile_token_embedding_backward_weight_float32",
        "nfn_native_tile_layer_norm_float32",
        "nfn_native_tile_layer_norm_backward_input_float32",
        "nfn_native_tile_layer_norm_backward_affine_float32",
        "nfn_native_tile_rms_norm_float32",
        "nfn_native_tile_rms_norm_backward_input_float32",
        "nfn_native_tile_softmax_lastdim_float32",
        "nfn_native_tile_token_cross_entropy_partials_float32",
        "nfn_native_tile_token_cross_entropy_backward_float32",
        "nfn_native_tile_token_cross_entropy_backward_with_workspace_float32",
        "nfn_native_tile_masked_token_cross_entropy_partials_float32",
        "nfn_native_tile_masked_token_cross_entropy_backward_float32",
        "nfn_native_tile_masked_token_cross_entropy_backward_with_workspace_float32",
        "nfn_native_tile_scaled_dot_product_attention_float32",
        "nfn_native_tile_scaled_dot_product_attention_backward_float32",
    };
}

std::string resolve_tile_ops_lib(const NanoGptPlan& plan, const char* program) {
    if (!plan.tile_ops_lib.empty()) {
        return plan.tile_ops_lib;
    }
    const char* env = std::getenv("NFN_NATIVE_TRAIN_TILE_OPS_LIB");
    if (env != nullptr && std::string_view(env).size() > 0) {
        return std::string(env);
    }
    std::filesystem::path exe_path(program);
    if (exe_path.has_parent_path()) {
        std::filesystem::path sibling = exe_path.parent_path() / "libnfn_native_train_tile_ops.so";
        if (std::filesystem::exists(sibling)) {
            return sibling.string();
        }
    }
    std::filesystem::path build_path = std::filesystem::current_path() / "build" / "libnfn_native_train_tile_ops.so";
    if (std::filesystem::exists(build_path)) {
        return build_path.string();
    }
    return "libnfn_native_train_tile_ops.so";
}

std::vector<std::string> cuda_runtime_candidates(const NanoGptPlan& plan) {
    if (!plan.cuda_runtime_lib.empty()) {
        return {plan.cuda_runtime_lib};
    }
    const char* env = std::getenv("NFN_CUDA_RUNTIME_LIB");
    if (env != nullptr && std::string_view(env).size() > 0) {
        return {std::string(env)};
    }
    return {"libcudart.so", "libcudart.so.13", "libcudart.so.12"};
}

template <typename Fn>
Fn load_symbol(void* handle, const char* name) {
    dlerror();
    void* symbol = dlsym(handle, name);
    if (symbol == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<Fn>(symbol);
}

std::string dl_last_error(const char* fallback) {
    const char* error = dlerror();
    return error == nullptr ? std::string(fallback) : std::string(error);
}

int print_tile_ops_check_json(const NanoGptPlan& plan, const char* program) {
    const std::string lib_path = resolve_tile_ops_lib(plan, program);
    std::vector<TileOpsSymbolCheck> checks;
    for (const std::string& symbol : required_tile_ops_symbols()) {
        checks.push_back(TileOpsSymbolCheck{symbol, false});
    }

    void* handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    std::string error;
    int abi_version = 0;
    if (handle == nullptr) {
        const char* dl_error = dlerror();
        error = dl_error == nullptr ? "dlopen failed" : dl_error;
    } else {
        for (TileOpsSymbolCheck& check : checks) {
            dlerror();
            void* symbol = dlsym(handle, check.name.c_str());
            check.found = symbol != nullptr;
        }
        void* abi_symbol = dlsym(handle, "nfn_native_tile_ops_abi_version");
        if (abi_symbol != nullptr) {
            using AbiFn = int (*)();
            abi_version = reinterpret_cast<AbiFn>(abi_symbol)();
        }
    }

    bool all_found = handle != nullptr;
    for (const TileOpsSymbolCheck& check : checks) {
        all_found = all_found && check.found;
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"nanogpt\",\n"
        << "  \"tile_ops_library\": \"" << json_escape(lib_path) << "\",\n"
        << "  \"loaded\": " << (handle != nullptr ? "true" : "false") << ",\n"
        << "  \"abi_version\": " << abi_version << ",\n"
        << "  \"required_symbol_count\": " << checks.size() << ",\n"
        << "  \"all_required_symbols_found\": " << (all_found ? "true" : "false") << ",\n"
        << "  \"symbols\": [\n";
    for (std::size_t i = 0; i < checks.size(); ++i) {
        std::cout
            << "    {\"name\": \"" << json_escape(checks[i].name)
            << "\", \"found\": " << (checks[i].found ? "true" : "false") << "}";
        if (i + 1 != checks.size()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "  ]";
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    if (handle != nullptr) {
        dlclose(handle);
    }
    return all_found ? 0 : 2;
}

int print_tile_ops_smoke_json(const NanoGptPlan& plan, const char* program) {
    constexpr std::int64_t kElements = 16;
    constexpr float kExpectedValue = 3.25f;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    const std::string tile_lib_path = resolve_tile_ops_lib(plan, program);
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(plan);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool kernel_loaded = false;
    bool passed = false;
    double max_abs_error = 0.0;
    std::string error;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (tile_handle == nullptr) {
        error = dl_last_error("dlopen tile ops failed");
    } else {
        tile_loaded = true;
        fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
        kernel_loaded = fill != nullptr;
        if (fill == nullptr) {
            error = dl_last_error("dlsym nfn_native_tile_fill_float32 failed");
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* device_values = nullptr;
    if (error.empty()) {
        int status = cuda_malloc(reinterpret_cast<void**>(&device_values), sizeof(float) * kElements);
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc");
        }
    }

    if (error.empty()) {
        int status = fill(device_values, kElements, kExpectedValue, nullptr);
        if (status != 0) {
            error = cuda_error(status, "nfn_native_tile_fill_float32");
        }
    }

    if (error.empty()) {
        int status = cuda_device_synchronize();
        if (status != 0) {
            error = cuda_error(status, "cudaDeviceSynchronize");
        }
    }

    std::vector<float> host_values(static_cast<std::size_t>(kElements), 0.0f);
    if (error.empty()) {
        int status = cuda_memcpy(
            host_values.data(), device_values, sizeof(float) * kElements, kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy device-to-host");
        }
    }

    if (error.empty()) {
        for (float value : host_values) {
            double abs_error = std::fabs(static_cast<double>(value) - static_cast<double>(kExpectedValue));
            if (abs_error > max_abs_error) {
                max_abs_error = abs_error;
            }
        }
        passed = max_abs_error <= 1e-6;
        if (!passed) {
            std::ostringstream out;
            out << "fill smoke max_abs_error " << max_abs_error << " exceeded tolerance";
            error = out.str();
        }
    }

    if (device_values != nullptr && cuda_free != nullptr) {
        int status = cuda_free(device_values);
        if (status != 0 && error.empty()) {
            error = cuda_error(status, "cudaFree");
        }
    }
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"nanogpt\",\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"kernel\": \"nfn_native_tile_fill_float32\",\n"
        << "  \"kernel_loaded\": " << (kernel_loaded ? "true" : "false") << ",\n"
        << "  \"elements\": " << kElements << ",\n"
        << "  \"expected_value\": " << kExpectedValue << ",\n"
        << "  \"max_abs_error\": " << max_abs_error << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

int print_optimizer_smoke_json(const NanoGptPlan& plan, const char* program) {
    constexpr float kInitialParam = 1.0f;
    constexpr float kInitialGrad = 0.5f;
    constexpr float kInitialMoment = 0.0f;
    constexpr float kLearningRate = 0.1f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.95f;
    constexpr float kEps = 1e-8f;
    constexpr float kWeightDecay = 0.1f;
    constexpr float kBiasCorrection1 = 1.0f - kBeta1;
    constexpr float kSqrtBiasCorrection2 = 0.22360679775f;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    const std::vector<ParameterBuffer> layout = build_parameter_layout(plan);
    const std::int64_t total_parameters = parameter_layout_count(layout);
    std::int64_t decay_buffer_count = 0;
    std::int64_t no_decay_buffer_count = 0;
    std::int64_t decay_elements = 0;
    std::int64_t no_decay_elements = 0;
    for (const ParameterBuffer& buffer : layout) {
        if (buffer.weight_decay) {
            decay_buffer_count += 1;
            decay_elements += buffer.count;
        } else {
            no_decay_buffer_count += 1;
            no_decay_elements += buffer.count;
        }
    }

    const float expected_m = kBeta1 * kInitialMoment + (1.0f - kBeta1) * kInitialGrad;
    const float expected_v = kBeta2 * kInitialMoment + (1.0f - kBeta2) * kInitialGrad * kInitialGrad;
    auto expected_param_for_decay = [&](float weight_decay) {
        return kInitialParam * (1.0f - kLearningRate * weight_decay) -
               (kLearningRate / kBiasCorrection1) * expected_m / (std::sqrt(expected_v) / kSqrtBiasCorrection2 + kEps);
    };

    const std::string tile_lib_path = resolve_tile_ops_lib(plan, program);
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(plan);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool fill_loaded = false;
    bool adamw_loaded = false;
    bool passed = false;
    double max_param_abs_error = 0.0;
    double max_exp_avg_abs_error = 0.0;
    double max_exp_avg_sq_abs_error = 0.0;
    std::string error;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using AdamWFn = int (*)(
        float*,
        const float*,
        float*,
        float*,
        std::int64_t,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        void*);
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    AdamWFn adamw = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (tile_handle == nullptr) {
        error = dl_last_error("dlopen tile ops failed");
    } else {
        tile_loaded = true;
        fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
        adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
        fill_loaded = fill != nullptr;
        adamw_loaded = adamw != nullptr;
        if (fill == nullptr || adamw == nullptr) {
            error = dl_last_error("dlsym fill/adamw failed");
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* param = nullptr;
    float* grad = nullptr;
    float* exp_avg = nullptr;
    float* exp_avg_sq = nullptr;
    auto allocate = [&](float** ptr, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_malloc(reinterpret_cast<void**>(ptr), sizeof(float) * total_parameters);
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(&param, "param");
    allocate(&grad, "grad");
    allocate(&exp_avg, "exp_avg");
    allocate(&exp_avg_sq, "exp_avg_sq");

    auto fill_buffer = [&](float* ptr, float value, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = fill(ptr, total_parameters, value, nullptr);
        if (status != 0) {
            error = cuda_error(status, "fill " + name);
        }
    };
    fill_buffer(param, kInitialParam, "param");
    fill_buffer(grad, kInitialGrad, "grad");
    fill_buffer(exp_avg, kInitialMoment, "exp_avg");
    fill_buffer(exp_avg_sq, kInitialMoment, "exp_avg_sq");

    std::int64_t adamw_step_calls = 0;
    if (error.empty()) {
        for (const ParameterBuffer& buffer : layout) {
            const float buffer_weight_decay = buffer.weight_decay ? kWeightDecay : 0.0f;
            int status = adamw(
                param + buffer.offset,
                grad + buffer.offset,
                exp_avg + buffer.offset,
                exp_avg_sq + buffer.offset,
                buffer.count,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                buffer_weight_decay,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr);
            if (status != 0) {
                error = cuda_error(status, "nfn_native_tile_adamw_step_float32 " + buffer.name);
                break;
            }
            adamw_step_calls += 1;
        }
    }

    if (error.empty()) {
        int status = cuda_device_synchronize();
        if (status != 0) {
            error = cuda_error(status, "cudaDeviceSynchronize");
        }
    }

    std::vector<float> host_param(static_cast<std::size_t>(total_parameters), 0.0f);
    std::vector<float> host_exp_avg(static_cast<std::size_t>(total_parameters), 0.0f);
    std::vector<float> host_exp_avg_sq(static_cast<std::size_t>(total_parameters), 0.0f);
    auto copy_back = [&](float* device_ptr, std::vector<float>& host, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_memcpy(host.data(), device_ptr, sizeof(float) * total_parameters, kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy " + name);
        }
    };
    copy_back(param, host_param, "param");
    copy_back(exp_avg, host_exp_avg, "exp_avg");
    copy_back(exp_avg_sq, host_exp_avg_sq, "exp_avg_sq");

    if (error.empty()) {
        for (const ParameterBuffer& buffer : layout) {
            const float expected_param = expected_param_for_decay(buffer.weight_decay ? kWeightDecay : 0.0f);
            for (std::int64_t local_index = 0; local_index < buffer.count; ++local_index) {
                const std::size_t index = static_cast<std::size_t>(buffer.offset + local_index);
                max_param_abs_error = std::max(
                    max_param_abs_error,
                    std::fabs(static_cast<double>(host_param[index]) - expected_param));
                max_exp_avg_abs_error = std::max(
                    max_exp_avg_abs_error,
                    std::fabs(static_cast<double>(host_exp_avg[index]) - expected_m));
                max_exp_avg_sq_abs_error = std::max(
                    max_exp_avg_sq_abs_error,
                    std::fabs(static_cast<double>(host_exp_avg_sq[index]) - expected_v));
            }
        }
        passed = max_param_abs_error <= 1e-5 && max_exp_avg_abs_error <= 1e-6 && max_exp_avg_sq_abs_error <= 1e-6;
        if (!passed) {
            std::ostringstream out;
            out << "AdamW smoke exceeded tolerance: param=" << max_param_abs_error
                << " exp_avg=" << max_exp_avg_abs_error
                << " exp_avg_sq=" << max_exp_avg_sq_abs_error;
            error = out.str();
        }
    }

    if (param != nullptr && cuda_free != nullptr) {
        int status = cuda_free(param);
        if (status != 0 && error.empty()) {
            error = cuda_error(status, "cudaFree param");
        }
    }
    if (grad != nullptr && cuda_free != nullptr) {
        int status = cuda_free(grad);
        if (status != 0 && error.empty()) {
            error = cuda_error(status, "cudaFree grad");
        }
    }
    if (exp_avg != nullptr && cuda_free != nullptr) {
        int status = cuda_free(exp_avg);
        if (status != 0 && error.empty()) {
            error = cuda_error(status, "cudaFree exp_avg");
        }
    }
    if (exp_avg_sq != nullptr && cuda_free != nullptr) {
        int status = cuda_free(exp_avg_sq);
        if (status != 0 && error.empty()) {
            error = cuda_error(status, "cudaFree exp_avg_sq");
        }
    }
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"nanogpt\",\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"fill_kernel_loaded\": " << (fill_loaded ? "true" : "false") << ",\n"
        << "  \"optimizer_kernel\": \"nfn_native_tile_adamw_step_float32\",\n"
        << "  \"optimizer_kernel_loaded\": " << (adamw_loaded ? "true" : "false") << ",\n"
        << "  \"parameter_buffer_count\": " << layout.size() << ",\n"
        << "  \"total_parameters\": " << total_parameters << ",\n"
        << "  \"adamw_step_calls\": " << adamw_step_calls << ",\n"
        << "  \"decay_buffer_count\": " << decay_buffer_count << ",\n"
        << "  \"no_decay_buffer_count\": " << no_decay_buffer_count << ",\n"
        << "  \"decay_elements\": " << decay_elements << ",\n"
        << "  \"no_decay_elements\": " << no_decay_elements << ",\n"
        << "  \"expected_decay_param\": " << expected_param_for_decay(kWeightDecay) << ",\n"
        << "  \"expected_no_decay_param\": " << expected_param_for_decay(0.0f) << ",\n"
        << "  \"expected_exp_avg\": " << expected_m << ",\n"
        << "  \"expected_exp_avg_sq\": " << expected_v << ",\n"
        << "  \"max_param_abs_error\": " << max_param_abs_error << ",\n"
        << "  \"max_exp_avg_abs_error\": " << max_exp_avg_abs_error << ",\n"
        << "  \"max_exp_avg_sq_abs_error\": " << max_exp_avg_sq_abs_error << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

int print_training_loop_step_smoke_json(const NanoGptPlan& plan, const char* program) {
    constexpr float kInitialParam = 1.0f;
    constexpr float kSyntheticGrad = 0.25f;
    constexpr float kInitialMoment = 0.0f;
    constexpr float kLearningRate = 0.1f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.95f;
    constexpr float kEps = 1e-8f;
    constexpr float kClipEps = 1e-6f;
    constexpr float kWeightDecay = 0.1f;
    constexpr float kMaxNorm = 1.0f;
    constexpr float kBiasCorrection1 = 1.0f - kBeta1;
    constexpr float kSqrtBiasCorrection2 = 0.22360679775f;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    constexpr std::int64_t kTileSize = 1024;

    const std::vector<ParameterBuffer> layout = build_parameter_layout(plan);
    const std::int64_t total_parameters = parameter_layout_count(layout);
    const std::int64_t partial_count = ceil_div(total_parameters, kTileSize);
    std::int64_t decay_buffer_count = 0;
    std::int64_t no_decay_buffer_count = 0;
    for (const ParameterBuffer& buffer : layout) {
        if (buffer.weight_decay) {
            decay_buffer_count += 1;
        } else {
            no_decay_buffer_count += 1;
        }
    }

    const float expected_clip_scale = std::min(
        1.0f,
        kMaxNorm / (std::sqrt(static_cast<float>(total_parameters) * kSyntheticGrad * kSyntheticGrad) + kClipEps));
    const float expected_scaled_grad = kSyntheticGrad * expected_clip_scale;
    const float expected_m = (1.0f - kBeta1) * expected_scaled_grad;
    const float expected_v = (1.0f - kBeta2) * expected_scaled_grad * expected_scaled_grad;
    auto expected_param_for_decay = [&](float weight_decay) {
        return kInitialParam * (1.0f - kLearningRate * weight_decay) -
               (kLearningRate / kBiasCorrection1) * expected_m /
                   (std::sqrt(expected_v) / kSqrtBiasCorrection2 + kEps);
    };

    const std::string tile_lib_path = resolve_tile_ops_lib(plan, program);
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(plan);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool passed = false;
    double clip_scale_abs_error = 0.0;
    double max_grad_abs_error = 0.0;
    double max_param_abs_error = 0.0;
    double max_exp_avg_abs_error = 0.0;
    double max_exp_avg_sq_abs_error = 0.0;
    std::string error;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using SumsqPartialsFn = int (*)(const float*, float*, std::int64_t, void*);
    using ClipScaleFn = int (*)(const float*, float*, std::int64_t, float, float, void*);
    using ScaleByDeviceFn = int (*)(float*, const float*, std::int64_t, void*);
    using AdamWFn = int (*)(
        float*,
        const float*,
        float*,
        float*,
        std::int64_t,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        void*);
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    SumsqPartialsFn sumsq_partials = nullptr;
    ClipScaleFn clip_scale = nullptr;
    ScaleByDeviceFn scale_by_device = nullptr;
    AdamWFn adamw = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (tile_handle == nullptr) {
        error = dl_last_error("dlopen tile ops failed");
    } else {
        tile_loaded = true;
        fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
        sumsq_partials = load_symbol<SumsqPartialsFn>(tile_handle, "nfn_native_tile_sumsq_partials_float32");
        clip_scale = load_symbol<ClipScaleFn>(tile_handle, "nfn_native_tile_global_norm_clip_scale_float32");
        scale_by_device =
            load_symbol<ScaleByDeviceFn>(tile_handle, "nfn_native_tile_scale_inplace_by_device_float32");
        adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
        if (fill == nullptr || sumsq_partials == nullptr || clip_scale == nullptr ||
            scale_by_device == nullptr || adamw == nullptr) {
            error = dl_last_error("dlsym training-loop-step kernels failed");
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr ||
            cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* param = nullptr;
    float* grad = nullptr;
    float* exp_avg = nullptr;
    float* exp_avg_sq = nullptr;
    float* sumsq = nullptr;
    float* clip = nullptr;
    auto allocate = [&](float** ptr, std::int64_t elements, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_malloc(reinterpret_cast<void**>(ptr), sizeof(float) * static_cast<std::size_t>(elements));
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(&param, total_parameters, "param");
    allocate(&grad, total_parameters, "grad");
    allocate(&exp_avg, total_parameters, "exp_avg");
    allocate(&exp_avg_sq, total_parameters, "exp_avg_sq");
    allocate(&sumsq, partial_count, "sumsq_partials");
    allocate(&clip, 1, "clip_scale");

    auto run = [&](int status, const std::string& name) {
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    if (error.empty()) {
        run(fill(param, total_parameters, kInitialParam, nullptr), "param.fill");
    }
    if (error.empty()) {
        run(fill(grad, total_parameters, 0.0f, nullptr), "gradient_zero");
    }
    if (error.empty()) {
        run(fill(exp_avg, total_parameters, kInitialMoment, nullptr), "exp_avg.fill");
    }
    if (error.empty()) {
        run(fill(exp_avg_sq, total_parameters, kInitialMoment, nullptr), "exp_avg_sq.fill");
    }
    if (error.empty()) {
        run(fill(grad, total_parameters, kSyntheticGrad, nullptr), "synthetic_backward_gradient_fill");
    }
    if (error.empty()) {
        run(sumsq_partials(grad, sumsq, total_parameters, nullptr), "gradient_sumsq_partials");
    }
    if (error.empty()) {
        run(clip_scale(sumsq, clip, partial_count, kMaxNorm, kClipEps, nullptr), "gradient_clip_scale");
    }
    if (error.empty()) {
        run(scale_by_device(grad, clip, total_parameters, nullptr), "gradient_scale");
    }

    std::int64_t adamw_step_calls = 0;
    if (error.empty()) {
        for (const ParameterBuffer& buffer : layout) {
            const float buffer_weight_decay = buffer.weight_decay ? kWeightDecay : 0.0f;
            int status = adamw(
                param + buffer.offset,
                grad + buffer.offset,
                exp_avg + buffer.offset,
                exp_avg_sq + buffer.offset,
                buffer.count,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                buffer_weight_decay,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr);
            if (status != 0) {
                error = cuda_error(status, "adamw_step " + buffer.name);
                break;
            }
            adamw_step_calls += 1;
        }
    }
    if (error.empty()) {
        run(cuda_device_synchronize(), "cudaDeviceSynchronize");
    }

    std::vector<float> host_param(static_cast<std::size_t>(total_parameters), 0.0f);
    std::vector<float> host_grad(static_cast<std::size_t>(total_parameters), 0.0f);
    std::vector<float> host_exp_avg(static_cast<std::size_t>(total_parameters), 0.0f);
    std::vector<float> host_exp_avg_sq(static_cast<std::size_t>(total_parameters), 0.0f);
    std::vector<float> host_clip(1, 0.0f);
    auto copy_back = [&](float* device_ptr, std::vector<float>& host, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_memcpy(host.data(), device_ptr, sizeof(float) * host.size(), kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy device-to-host " + name);
        }
    };
    copy_back(param, host_param, "param");
    copy_back(grad, host_grad, "grad");
    copy_back(exp_avg, host_exp_avg, "exp_avg");
    copy_back(exp_avg_sq, host_exp_avg_sq, "exp_avg_sq");
    copy_back(clip, host_clip, "clip_scale");

    if (error.empty()) {
        clip_scale_abs_error = std::fabs(static_cast<double>(host_clip[0]) - expected_clip_scale);
        for (const ParameterBuffer& buffer : layout) {
            const float expected_param = expected_param_for_decay(buffer.weight_decay ? kWeightDecay : 0.0f);
            for (std::int64_t local_index = 0; local_index < buffer.count; ++local_index) {
                const std::size_t index = static_cast<std::size_t>(buffer.offset + local_index);
                max_grad_abs_error = std::max(
                    max_grad_abs_error,
                    std::fabs(static_cast<double>(host_grad[index]) - expected_scaled_grad));
                max_param_abs_error = std::max(
                    max_param_abs_error,
                    std::fabs(static_cast<double>(host_param[index]) - expected_param));
                max_exp_avg_abs_error = std::max(
                    max_exp_avg_abs_error,
                    std::fabs(static_cast<double>(host_exp_avg[index]) - expected_m));
                max_exp_avg_sq_abs_error = std::max(
                    max_exp_avg_sq_abs_error,
                    std::fabs(static_cast<double>(host_exp_avg_sq[index]) - expected_v));
            }
        }
        passed = clip_scale_abs_error <= 1e-6 && max_grad_abs_error <= 1e-6 &&
                 max_param_abs_error <= 1e-5 && max_exp_avg_abs_error <= 1e-6 &&
                 max_exp_avg_sq_abs_error <= 1e-6 && adamw_step_calls == static_cast<std::int64_t>(layout.size());
        if (!passed) {
            std::ostringstream out;
            out << "training-loop smoke exceeded tolerance: clip=" << clip_scale_abs_error
                << " grad=" << max_grad_abs_error
                << " param=" << max_param_abs_error
                << " exp_avg=" << max_exp_avg_abs_error
                << " exp_avg_sq=" << max_exp_avg_sq_abs_error;
            error = out.str();
        }
    }

    auto free_device = [&](void* ptr, const std::string& name) {
        if (ptr != nullptr && cuda_free != nullptr) {
            int status = cuda_free(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFree " + name);
            }
        }
    };
    free_device(param, "param");
    free_device(grad, "grad");
    free_device(exp_avg, "exp_avg");
    free_device(exp_avg_sq, "exp_avg_sq");
    free_device(sumsq, "sumsq_partials");
    free_device(clip, "clip_scale");
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"nanogpt\",\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"parameter_buffer_count\": " << layout.size() << ",\n"
        << "  \"total_parameters\": " << total_parameters << ",\n"
        << "  \"partial_count\": " << partial_count << ",\n"
        << "  \"adamw_step_calls\": " << adamw_step_calls << ",\n"
        << "  \"decay_buffer_count\": " << decay_buffer_count << ",\n"
        << "  \"no_decay_buffer_count\": " << no_decay_buffer_count << ",\n"
        << "  \"kernels\": [\n"
        << "    \"nfn_native_tile_fill_float32\",\n"
        << "    \"nfn_native_tile_sumsq_partials_float32\",\n"
        << "    \"nfn_native_tile_global_norm_clip_scale_float32\",\n"
        << "    \"nfn_native_tile_scale_inplace_by_device_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"expected_clip_scale\": " << expected_clip_scale << ",\n"
        << "  \"expected_scaled_grad\": " << expected_scaled_grad << ",\n"
        << "  \"clip_scale_abs_error\": " << clip_scale_abs_error << ",\n"
        << "  \"max_grad_abs_error\": " << max_grad_abs_error << ",\n"
        << "  \"max_param_abs_error\": " << max_param_abs_error << ",\n"
        << "  \"max_exp_avg_abs_error\": " << max_exp_avg_abs_error << ",\n"
        << "  \"max_exp_avg_sq_abs_error\": " << max_exp_avg_sq_abs_error << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

int print_lm_step_smoke_json(const NanoGptPlan& plan, const char* program) {
    constexpr std::int64_t kRows = 2;
    constexpr std::int64_t kVocab = 4;
    constexpr std::int64_t kDim = 4;
    constexpr std::int64_t kWeightElements = kVocab * kDim;
    constexpr std::int64_t kActivationElements = kRows * kDim;
    constexpr std::int64_t kLogitElements = kRows * kVocab;
    constexpr float kInitialWeight = 0.1f;
    constexpr float kLearningRate = 0.1f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.95f;
    constexpr float kEps = 1e-8f;
    constexpr float kWeightDecay = 0.1f;
    constexpr float kBiasCorrection1 = 1.0f - kBeta1;
    constexpr float kSqrtBiasCorrection2 = 0.22360679775f;
    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    const std::int64_t host_tokens[kRows] = {0, 1};
    const std::int64_t host_targets[kRows] = {1, 2};
    const float expected_loss = static_cast<float>(kRows) * std::log(static_cast<float>(kVocab));

    auto expected_grad_for_token = [](std::int64_t token) {
        if (token == 0 || token == 3) {
            return 0.025f;
        }
        return -0.025f;
    };
    auto expected_param_for_grad = [&](float grad) {
        const float next_m = (1.0f - kBeta1) * grad;
        const float next_v = (1.0f - kBeta2) * grad * grad;
        return kInitialWeight * (1.0f - kLearningRate * kWeightDecay) -
               (kLearningRate / kBiasCorrection1) * next_m / (std::sqrt(next_v) / kSqrtBiasCorrection2 + kEps);
    };

    const std::string tile_lib_path = resolve_tile_ops_lib(plan, program);
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(plan);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool passed = false;
    double loss_abs_error = 0.0;
    double max_grad_abs_error = 0.0;
    double max_weight_abs_error = 0.0;
    std::string error;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using TokenEmbeddingFn = int (*)(const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenEmbeddingBackwardWeightFn = int (*)(
        const std::int64_t*, const float*, float*, std::int64_t, std::int64_t, void*);
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyPartialsFn = int (*)(
        const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyBackwardFn = int (*)(
        const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, float, void*);
    using AdamWFn = int (*)(
        float*,
        const float*,
        float*,
        float*,
        std::int64_t,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        void*);
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    TokenEmbeddingFn token_embedding = nullptr;
    TokenEmbeddingBackwardWeightFn token_embedding_backward_weight = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightFn linear_backward_weight = nullptr;
    TokenCrossEntropyPartialsFn ce_partials = nullptr;
    TokenCrossEntropyBackwardFn ce_backward = nullptr;
    AdamWFn adamw = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (tile_handle == nullptr) {
        error = dl_last_error("dlopen tile ops failed");
    } else {
        tile_loaded = true;
        fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
        token_embedding = load_symbol<TokenEmbeddingFn>(tile_handle, "nfn_native_tile_token_embedding_float32");
        token_embedding_backward_weight = load_symbol<TokenEmbeddingBackwardWeightFn>(
            tile_handle, "nfn_native_tile_token_embedding_backward_weight_float32");
        linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
        linear_backward_input = load_symbol<LinearBackwardInputFn>(
            tile_handle, "nfn_native_tile_linear_backward_input_float32");
        linear_backward_weight = load_symbol<LinearBackwardWeightFn>(
            tile_handle, "nfn_native_tile_linear_backward_weight_float32");
        ce_partials = load_symbol<TokenCrossEntropyPartialsFn>(
            tile_handle, "nfn_native_tile_token_cross_entropy_partials_float32");
        ce_backward = load_symbol<TokenCrossEntropyBackwardFn>(
            tile_handle, "nfn_native_tile_token_cross_entropy_backward_float32");
        adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
        if (fill == nullptr || token_embedding == nullptr || token_embedding_backward_weight == nullptr ||
            linear == nullptr || linear_backward_input == nullptr || linear_backward_weight == nullptr ||
            ce_partials == nullptr || ce_backward == nullptr || adamw == nullptr) {
            error = dl_last_error("dlsym LM-step kernels failed");
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* weight = nullptr;
    float* grad_weight = nullptr;
    float* exp_avg = nullptr;
    float* exp_avg_sq = nullptr;
    float* hidden = nullptr;
    float* logits = nullptr;
    float* loss_partials = nullptr;
    float* grad_logits = nullptr;
    float* grad_hidden = nullptr;
    std::int64_t* token_ids = nullptr;
    std::int64_t* targets = nullptr;

    auto allocate = [&](auto** ptr, std::size_t bytes, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_malloc(reinterpret_cast<void**>(ptr), bytes);
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(&weight, sizeof(float) * kWeightElements, "weight");
    allocate(&grad_weight, sizeof(float) * kWeightElements, "grad_weight");
    allocate(&exp_avg, sizeof(float) * kWeightElements, "exp_avg");
    allocate(&exp_avg_sq, sizeof(float) * kWeightElements, "exp_avg_sq");
    allocate(&hidden, sizeof(float) * kActivationElements, "hidden");
    allocate(&logits, sizeof(float) * kLogitElements, "logits");
    allocate(&loss_partials, sizeof(float), "loss_partials");
    allocate(&grad_logits, sizeof(float) * kLogitElements, "grad_logits");
    allocate(&grad_hidden, sizeof(float) * kActivationElements, "grad_hidden");
    allocate(&token_ids, sizeof(std::int64_t) * kRows, "token_ids");
    allocate(&targets, sizeof(std::int64_t) * kRows, "targets");

    auto fill_buffer = [&](float* ptr, std::int64_t elements, float value, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = fill(ptr, elements, value, nullptr);
        if (status != 0) {
            error = cuda_error(status, "fill " + name);
        }
    };
    fill_buffer(weight, kWeightElements, kInitialWeight, "weight");
    fill_buffer(grad_weight, kWeightElements, 0.0f, "grad_weight");
    fill_buffer(exp_avg, kWeightElements, 0.0f, "exp_avg");
    fill_buffer(exp_avg_sq, kWeightElements, 0.0f, "exp_avg_sq");
    fill_buffer(grad_hidden, kActivationElements, 0.0f, "grad_hidden");

    auto copy_to_device = [&](void* dst, const void* src, std::size_t bytes, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_memcpy(dst, src, bytes, kCudaMemcpyHostToDevice);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy host-to-device " + name);
        }
    };
    copy_to_device(token_ids, host_tokens, sizeof(host_tokens), "token_ids");
    copy_to_device(targets, host_targets, sizeof(host_targets), "targets");

    auto run = [&](int status, const std::string& name) {
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    if (error.empty()) {
        run(token_embedding(weight, token_ids, hidden, kRows, kDim, nullptr), "token_embedding");
    }
    if (error.empty()) {
        run(linear(hidden, weight, nullptr, logits, kRows, kDim, kVocab, false, nullptr), "linear");
    }
    if (error.empty()) {
        run(ce_partials(logits, targets, loss_partials, kRows, kVocab, nullptr), "token_cross_entropy_partials");
    }
    if (error.empty()) {
        run(ce_backward(logits, targets, grad_logits, kRows, kVocab, 1.0f / static_cast<float>(kRows), nullptr),
            "token_cross_entropy_backward");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_logits, weight, grad_hidden, kRows, kDim, kVocab, nullptr),
            "linear_backward_input");
    }
    if (error.empty()) {
        run(linear_backward_weight(hidden, grad_logits, grad_weight, kRows, kDim, kVocab, nullptr),
            "linear_backward_weight");
    }
    if (error.empty()) {
        run(token_embedding_backward_weight(token_ids, grad_hidden, grad_weight, kRows, kDim, nullptr),
            "token_embedding_backward_weight");
    }
    if (error.empty()) {
        run(adamw(
                weight,
                grad_weight,
                exp_avg,
                exp_avg_sq,
                kWeightElements,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                kWeightDecay,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "adamw_step");
    }
    if (error.empty()) {
        run(cuda_device_synchronize(), "cudaDeviceSynchronize");
    }

    std::vector<float> host_weight(static_cast<std::size_t>(kWeightElements), 0.0f);
    std::vector<float> host_grad_weight(static_cast<std::size_t>(kWeightElements), 0.0f);
    std::vector<float> host_loss(1, 0.0f);
    auto copy_back = [&](float* device_ptr, std::vector<float>& host, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_memcpy(host.data(), device_ptr, sizeof(float) * host.size(), kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy device-to-host " + name);
        }
    };
    copy_back(weight, host_weight, "weight");
    copy_back(grad_weight, host_grad_weight, "grad_weight");
    copy_back(loss_partials, host_loss, "loss_partials");

    if (error.empty()) {
        loss_abs_error = std::fabs(static_cast<double>(host_loss[0]) - expected_loss);
        for (std::int64_t token = 0; token < kVocab; ++token) {
            const float expected_grad = expected_grad_for_token(token);
            const float expected_weight = expected_param_for_grad(expected_grad);
            for (std::int64_t dim = 0; dim < kDim; ++dim) {
                const std::size_t index = static_cast<std::size_t>(token * kDim + dim);
                max_grad_abs_error = std::max(
                    max_grad_abs_error,
                    std::fabs(static_cast<double>(host_grad_weight[index]) - expected_grad));
                max_weight_abs_error = std::max(
                    max_weight_abs_error,
                    std::fabs(static_cast<double>(host_weight[index]) - expected_weight));
            }
        }
        passed = loss_abs_error <= 1e-5 && max_grad_abs_error <= 1e-5 && max_weight_abs_error <= 1e-5;
        if (!passed) {
            std::ostringstream out;
            out << "LM smoke exceeded tolerance: loss=" << loss_abs_error
                << " grad=" << max_grad_abs_error
                << " weight=" << max_weight_abs_error;
            error = out.str();
        }
    }

    auto free_device = [&](void* ptr, const std::string& name) {
        if (ptr != nullptr && cuda_free != nullptr) {
            int status = cuda_free(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFree " + name);
            }
        }
    };
    free_device(weight, "weight");
    free_device(grad_weight, "grad_weight");
    free_device(exp_avg, "exp_avg");
    free_device(exp_avg_sq, "exp_avg_sq");
    free_device(hidden, "hidden");
    free_device(logits, "logits");
    free_device(loss_partials, "loss_partials");
    free_device(grad_logits, "grad_logits");
    free_device(grad_hidden, "grad_hidden");
    free_device(token_ids, "token_ids");
    free_device(targets, "targets");
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"nanogpt\",\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"rows\": " << kRows << ",\n"
        << "  \"vocab\": " << kVocab << ",\n"
        << "  \"model_dim\": " << kDim << ",\n"
        << "  \"kernels\": [\n"
        << "    \"nfn_native_tile_token_embedding_float32\",\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_token_cross_entropy_partials_float32\",\n"
        << "    \"nfn_native_tile_token_cross_entropy_backward_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_token_embedding_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"expected_loss\": " << expected_loss << ",\n"
        << "  \"loss_abs_error\": " << loss_abs_error << ",\n"
        << "  \"max_grad_abs_error\": " << max_grad_abs_error << ",\n"
        << "  \"max_weight_abs_error\": " << max_weight_abs_error << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

float adamw_expected_param_scalar(
    float initial_param,
    float grad,
    float learning_rate,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float sqrt_bias_correction2);

int print_token_train_step_smoke_json(const NanoGptPlan& plan, const char* program) {
    constexpr float kInitialWeight = 0.1f;
    constexpr float kLearningRate = 0.1f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.95f;
    constexpr float kEps = 1e-8f;
    constexpr float kWeightDecay = 0.1f;
    constexpr float kBiasCorrection1 = 1.0f - kBeta1;
    constexpr float kSqrtBiasCorrection2 = 0.22360679775f;
    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    neuralfn::native_train::TokenShardDataset dataset;
    neuralfn::native_train::BatchPlan batch_plan;
    neuralfn::native_train::TokenBatch batch;
    bool have_dataset = false;
    bool have_batch = false;
    std::string error;

    try {
        dataset = neuralfn::native_train::resolve_token_shards(plan.dataset_alias, plan.allow_train_as_val);
        batch_plan = neuralfn::native_train::build_batch_plan(
            dataset, plan.train_seq_len, plan.batch_size, plan.train_batch_tokens);
        have_dataset = true;
        neuralfn::native_train::SequentialTokenBatchSampler sampler(
            dataset.train_shards, plan.train_seq_len, plan.batch_size);
        have_batch = sampler.next(batch);
        if (!have_batch) {
            error = "not enough train tokens to build one native token batch";
        }
    } catch (const std::exception& exc) {
        error = exc.what();
    }

    const std::int64_t rows = have_batch ? static_cast<std::int64_t>(batch.tokens.size()) : 0;
    const std::int64_t vocab = plan.vocab_size;
    const std::int64_t dim = plan.model_dim;
    const std::int64_t weight_elements = vocab * dim;
    const std::int64_t activation_elements = rows * dim;
    const std::int64_t logit_elements = rows * vocab;
    std::vector<std::int64_t> host_tokens(static_cast<std::size_t>(rows), 0);
    std::vector<std::int64_t> host_targets(static_cast<std::size_t>(rows), 0);
    std::vector<std::int64_t> target_counts(static_cast<std::size_t>(vocab), 0);

    if (error.empty()) {
        if (rows <= 0) {
            error = "sampled token batch is empty";
        } else if (vocab <= 0 || dim <= 0) {
            error = "vocab and model_dim must be positive";
        } else {
            for (std::int64_t i = 0; i < rows; ++i) {
                host_tokens[static_cast<std::size_t>(i)] = static_cast<std::int64_t>(batch.tokens[static_cast<std::size_t>(i)]);
                host_targets[static_cast<std::size_t>(i)] =
                    static_cast<std::int64_t>(batch.targets[static_cast<std::size_t>(i)]);
                if (host_tokens[static_cast<std::size_t>(i)] < 0 || host_tokens[static_cast<std::size_t>(i)] >= vocab ||
                    host_targets[static_cast<std::size_t>(i)] < 0 || host_targets[static_cast<std::size_t>(i)] >= vocab) {
                    std::ostringstream out;
                    out << "sampled token/target id exceeds --vocab-size at row " << i
                        << ": token=" << host_tokens[static_cast<std::size_t>(i)]
                        << " target=" << host_targets[static_cast<std::size_t>(i)]
                        << " vocab=" << vocab;
                    error = out.str();
                    break;
                }
                target_counts[static_cast<std::size_t>(host_targets[static_cast<std::size_t>(i)])] += 1;
            }
        }
    }

    const float expected_loss = rows > 0 && vocab > 0
        ? static_cast<float>(rows) * std::log(static_cast<float>(vocab))
        : 0.0f;
    std::vector<float> expected_grad(static_cast<std::size_t>(std::max<std::int64_t>(weight_elements, 0)), 0.0f);
    if (error.empty()) {
        for (std::int64_t token = 0; token < vocab; ++token) {
            const float grad =
                kInitialWeight *
                ((1.0f / static_cast<float>(vocab)) -
                 (static_cast<float>(target_counts[static_cast<std::size_t>(token)]) / static_cast<float>(rows)));
            for (std::int64_t d = 0; d < dim; ++d) {
                expected_grad[static_cast<std::size_t>(token * dim + d)] = grad;
            }
        }
    }

    const std::string tile_lib_path = resolve_tile_ops_lib(plan, program);
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(plan);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool passed = false;
    double loss_abs_error = 0.0;
    double max_grad_abs_error = 0.0;
    double max_weight_abs_error = 0.0;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using TokenEmbeddingFn = int (*)(const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenEmbeddingBackwardWeightFn = int (*)(
        const std::int64_t*, const float*, float*, std::int64_t, std::int64_t, void*);
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyPartialsFn = int (*)(
        const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyBackwardFn = int (*)(
        const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, float, void*);
    using AdamWFn = int (*)(
        float*,
        const float*,
        float*,
        float*,
        std::int64_t,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        void*);
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = nullptr;
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    TokenEmbeddingFn token_embedding = nullptr;
    TokenEmbeddingBackwardWeightFn token_embedding_backward_weight = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightFn linear_backward_weight = nullptr;
    TokenCrossEntropyPartialsFn ce_partials = nullptr;
    TokenCrossEntropyBackwardFn ce_backward = nullptr;
    AdamWFn adamw = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (error.empty()) {
        tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            error = dl_last_error("dlopen tile ops failed");
        } else {
            tile_loaded = true;
            fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
            token_embedding = load_symbol<TokenEmbeddingFn>(tile_handle, "nfn_native_tile_token_embedding_float32");
            token_embedding_backward_weight = load_symbol<TokenEmbeddingBackwardWeightFn>(
                tile_handle, "nfn_native_tile_token_embedding_backward_weight_float32");
            linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
            linear_backward_input = load_symbol<LinearBackwardInputFn>(
                tile_handle, "nfn_native_tile_linear_backward_input_float32");
            linear_backward_weight = load_symbol<LinearBackwardWeightFn>(
                tile_handle, "nfn_native_tile_linear_backward_weight_float32");
            ce_partials = load_symbol<TokenCrossEntropyPartialsFn>(
                tile_handle, "nfn_native_tile_token_cross_entropy_partials_float32");
            ce_backward = load_symbol<TokenCrossEntropyBackwardFn>(
                tile_handle, "nfn_native_tile_token_cross_entropy_backward_float32");
            adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
            if (fill == nullptr || token_embedding == nullptr || token_embedding_backward_weight == nullptr ||
                linear == nullptr || linear_backward_input == nullptr || linear_backward_weight == nullptr ||
                ce_partials == nullptr || ce_backward == nullptr || adamw == nullptr) {
                error = dl_last_error("dlsym token-train-step kernels failed");
            }
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* weight = nullptr;
    float* grad_weight = nullptr;
    float* exp_avg = nullptr;
    float* exp_avg_sq = nullptr;
    float* hidden = nullptr;
    float* logits = nullptr;
    float* loss_partials = nullptr;
    float* grad_logits = nullptr;
    float* grad_hidden = nullptr;
    std::int64_t* token_ids = nullptr;
    std::int64_t* targets = nullptr;

    auto allocate = [&](auto** ptr, std::size_t bytes, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_malloc(reinterpret_cast<void**>(ptr), bytes);
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(&weight, sizeof(float) * static_cast<std::size_t>(weight_elements), "weight");
    allocate(&grad_weight, sizeof(float) * static_cast<std::size_t>(weight_elements), "grad_weight");
    allocate(&exp_avg, sizeof(float) * static_cast<std::size_t>(weight_elements), "exp_avg");
    allocate(&exp_avg_sq, sizeof(float) * static_cast<std::size_t>(weight_elements), "exp_avg_sq");
    allocate(&hidden, sizeof(float) * static_cast<std::size_t>(activation_elements), "hidden");
    allocate(&logits, sizeof(float) * static_cast<std::size_t>(logit_elements), "logits");
    allocate(&loss_partials, sizeof(float), "loss_partials");
    allocate(&grad_logits, sizeof(float) * static_cast<std::size_t>(logit_elements), "grad_logits");
    allocate(&grad_hidden, sizeof(float) * static_cast<std::size_t>(activation_elements), "grad_hidden");
    allocate(&token_ids, sizeof(std::int64_t) * static_cast<std::size_t>(rows), "token_ids");
    allocate(&targets, sizeof(std::int64_t) * static_cast<std::size_t>(rows), "targets");

    auto fill_buffer = [&](float* ptr, std::int64_t elements, float value, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = fill(ptr, elements, value, nullptr);
        if (status != 0) {
            error = cuda_error(status, "fill " + name);
        }
    };
    fill_buffer(weight, weight_elements, kInitialWeight, "weight");
    fill_buffer(grad_weight, weight_elements, 0.0f, "grad_weight");
    fill_buffer(exp_avg, weight_elements, 0.0f, "exp_avg");
    fill_buffer(exp_avg_sq, weight_elements, 0.0f, "exp_avg_sq");
    fill_buffer(grad_hidden, activation_elements, 0.0f, "grad_hidden");

    auto copy_to_device = [&](void* dst, const void* src, std::size_t bytes, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_memcpy(dst, src, bytes, kCudaMemcpyHostToDevice);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy host-to-device " + name);
        }
    };
    copy_to_device(token_ids, host_tokens.data(), sizeof(std::int64_t) * host_tokens.size(), "token_ids");
    copy_to_device(targets, host_targets.data(), sizeof(std::int64_t) * host_targets.size(), "targets");

    auto run = [&](int status, const std::string& name) {
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    if (error.empty()) {
        run(token_embedding(weight, token_ids, hidden, rows, dim, nullptr), "token_embedding");
    }
    if (error.empty()) {
        run(linear(hidden, weight, nullptr, logits, rows, dim, vocab, false, nullptr), "linear");
    }
    if (error.empty()) {
        run(ce_partials(logits, targets, loss_partials, rows, vocab, nullptr), "token_cross_entropy_partials");
    }
    if (error.empty()) {
        run(ce_backward(logits, targets, grad_logits, rows, vocab, 1.0f / static_cast<float>(rows), nullptr),
            "token_cross_entropy_backward");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_logits, weight, grad_hidden, rows, dim, vocab, nullptr),
            "linear_backward_input");
    }
    if (error.empty()) {
        run(linear_backward_weight(hidden, grad_logits, grad_weight, rows, dim, vocab, nullptr),
            "linear_backward_weight");
    }
    if (error.empty()) {
        run(token_embedding_backward_weight(token_ids, grad_hidden, grad_weight, rows, dim, nullptr),
            "token_embedding_backward_weight");
    }
    if (error.empty()) {
        run(adamw(
                weight,
                grad_weight,
                exp_avg,
                exp_avg_sq,
                weight_elements,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                kWeightDecay,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "adamw_step");
    }
    if (error.empty()) {
        run(cuda_device_synchronize(), "cudaDeviceSynchronize");
    }

    std::vector<float> host_weight(static_cast<std::size_t>(std::max<std::int64_t>(weight_elements, 0)), 0.0f);
    std::vector<float> host_grad_weight(static_cast<std::size_t>(std::max<std::int64_t>(weight_elements, 0)), 0.0f);
    std::vector<float> host_loss(1, 0.0f);
    auto copy_back = [&](float* device_ptr, std::vector<float>& host, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_memcpy(host.data(), device_ptr, sizeof(float) * host.size(), kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy device-to-host " + name);
        }
    };
    copy_back(weight, host_weight, "weight");
    copy_back(grad_weight, host_grad_weight, "grad_weight");
    copy_back(loss_partials, host_loss, "loss_partials");

    if (error.empty()) {
        loss_abs_error = std::fabs(static_cast<double>(host_loss[0]) - expected_loss);
        for (std::int64_t token = 0; token < vocab; ++token) {
            for (std::int64_t d = 0; d < dim; ++d) {
                const std::size_t index = static_cast<std::size_t>(token * dim + d);
                const float expected_weight = adamw_expected_param_scalar(
                    kInitialWeight,
                    expected_grad[index],
                    kLearningRate,
                    kBeta1,
                    kBeta2,
                    kEps,
                    kWeightDecay,
                    kBiasCorrection1,
                    kSqrtBiasCorrection2);
                max_grad_abs_error = std::max(
                    max_grad_abs_error,
                    std::fabs(static_cast<double>(host_grad_weight[index]) - expected_grad[index]));
                max_weight_abs_error = std::max(
                    max_weight_abs_error,
                    std::fabs(static_cast<double>(host_weight[index]) - expected_weight));
            }
        }
        passed = loss_abs_error <= 1e-5 && max_grad_abs_error <= 2e-5 && max_weight_abs_error <= 2e-5;
        if (!passed) {
            std::ostringstream out;
            out << "token train-step smoke exceeded tolerance: loss=" << loss_abs_error
                << " grad=" << max_grad_abs_error
                << " weight=" << max_weight_abs_error;
            error = out.str();
        }
    }

    auto free_device = [&](void* ptr, const std::string& name) {
        if (ptr != nullptr && cuda_free != nullptr) {
            int status = cuda_free(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFree " + name);
            }
        }
    };
    free_device(weight, "weight");
    free_device(grad_weight, "grad_weight");
    free_device(exp_avg, "exp_avg");
    free_device(exp_avg_sq, "exp_avg_sq");
    free_device(hidden, "hidden");
    free_device(logits, "logits");
    free_device(loss_partials, "loss_partials");
    free_device(grad_logits, "grad_logits");
    free_device(grad_hidden, "grad_hidden");
    free_device(token_ids, "token_ids");
    free_device(targets, "targets");
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"nanogpt\",\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"dataset_loaded\": " << (have_dataset ? "true" : "false") << ",\n"
        << "  \"batch_loaded\": " << (have_batch ? "true" : "false") << ",\n"
        << "  \"token_shards\": ";
    if (have_dataset) {
        std::cout << neuralfn::native_train::token_shard_dataset_json(dataset, &batch_plan);
    } else {
        std::cout << "null";
    }
    std::cout << ",\n"
        << "  \"sample_batch\": ";
    if (have_batch) {
        std::cout << neuralfn::native_train::token_batch_json(batch);
    } else {
        std::cout << "null";
    }
    std::cout << ",\n"
        << "  \"rows\": " << rows << ",\n"
        << "  \"vocab\": " << vocab << ",\n"
        << "  \"model_dim\": " << dim << ",\n"
        << "  \"kernels\": [\n"
        << "    \"nfn_native_tile_token_embedding_float32\",\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_token_cross_entropy_partials_float32\",\n"
        << "    \"nfn_native_tile_token_cross_entropy_backward_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_token_embedding_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"expected_loss\": " << expected_loss << ",\n"
        << "  \"loss_abs_error\": " << loss_abs_error << ",\n"
        << "  \"max_grad_abs_error\": " << max_grad_abs_error << ",\n"
        << "  \"max_weight_abs_error\": " << max_weight_abs_error << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

int run_token_lm_training_json(const NanoGptPlan& plan, const char* program) {
    constexpr float kInitialWeight = 0.1f;
    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    neuralfn::native_train::TokenShardDataset dataset;
    neuralfn::native_train::BatchPlan batch_plan;
    std::string error;
    bool dataset_loaded = false;
    try {
        dataset = neuralfn::native_train::resolve_token_shards(plan.dataset_alias, plan.allow_train_as_val);
        batch_plan = neuralfn::native_train::build_batch_plan(
            dataset, plan.train_seq_len, plan.batch_size, plan.train_batch_tokens);
        dataset_loaded = true;
    } catch (const std::exception& exc) {
        error = exc.what();
    }

    const std::int64_t train_rows = plan.batch_size * plan.train_seq_len;
    const std::int64_t eval_rows = plan.eval_batch_size * plan.train_seq_len;
    const std::int64_t max_rows = std::max(train_rows, eval_rows);
    const std::int64_t vocab = plan.vocab_size;
    const std::int64_t dim = plan.model_dim;
    const std::int64_t weight_elements = vocab * dim;
    const std::int64_t activation_elements = max_rows * dim;
    const std::int64_t logit_elements = max_rows * vocab;
    if (error.empty() && (train_rows <= 0 || eval_rows <= 0 || vocab <= 0 || dim <= 0)) {
        error = "train rows, eval rows, vocab, and model_dim must be positive";
    }

    const std::string tile_lib_path = resolve_tile_ops_lib(plan, program);
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(plan);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool passed = false;
    std::int64_t steps_completed = 0;
    std::int64_t epochs_completed = 0;
    std::int64_t tokens_processed = 0;
    double final_loss_sum = 0.0;
    double final_loss_mean = 0.0;
    std::vector<ValidationLossRecord> validation_losses;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using TokenEmbeddingFn = int (*)(const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenEmbeddingBackwardWeightFn = int (*)(
        const std::int64_t*, const float*, float*, std::int64_t, std::int64_t, void*);
    using LinearFn = int (*)(const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightFn = int (*)(const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyPartialsFn = int (*)(const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyBackwardFn = int (*)(const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, float, void*);
    using TokenCrossEntropyBackwardWorkspaceFn = int (*)(
        const float*,
        const std::int64_t*,
        float*,
        float*,
        float*,
        std::int64_t,
        std::int64_t,
        float,
        void*);
    using AdamWFn = int (*)(
        float*,
        const float*,
        float*,
        float*,
        std::int64_t,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        void*);
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = nullptr;
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    TokenEmbeddingFn token_embedding = nullptr;
    TokenEmbeddingBackwardWeightFn token_embedding_backward_weight = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightFn linear_backward_weight = nullptr;
    TokenCrossEntropyPartialsFn ce_partials = nullptr;
    TokenCrossEntropyBackwardFn ce_backward = nullptr;
    TokenCrossEntropyBackwardWorkspaceFn ce_backward_workspace = nullptr;
    AdamWFn adamw = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (error.empty()) {
        tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            error = dl_last_error("dlopen tile ops failed");
        } else {
            tile_loaded = true;
            fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
            token_embedding = load_symbol<TokenEmbeddingFn>(tile_handle, "nfn_native_tile_token_embedding_float32");
            token_embedding_backward_weight = load_symbol<TokenEmbeddingBackwardWeightFn>(
                tile_handle, "nfn_native_tile_token_embedding_backward_weight_float32");
            linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
            linear_backward_input = load_symbol<LinearBackwardInputFn>(
                tile_handle, "nfn_native_tile_linear_backward_input_float32");
            linear_backward_weight = load_symbol<LinearBackwardWeightFn>(
                tile_handle, "nfn_native_tile_linear_backward_weight_float32");
            ce_partials = load_symbol<TokenCrossEntropyPartialsFn>(
                tile_handle, "nfn_native_tile_token_cross_entropy_partials_float32");
            ce_backward = load_symbol<TokenCrossEntropyBackwardFn>(
                tile_handle, "nfn_native_tile_token_cross_entropy_backward_float32");
            ce_backward_workspace = load_symbol<TokenCrossEntropyBackwardWorkspaceFn>(
                tile_handle, "nfn_native_tile_token_cross_entropy_backward_with_workspace_float32");
            adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
            if (fill == nullptr || token_embedding == nullptr || token_embedding_backward_weight == nullptr ||
                linear == nullptr || linear_backward_input == nullptr || linear_backward_weight == nullptr ||
                ce_partials == nullptr || ce_backward == nullptr || ce_backward_workspace == nullptr || adamw == nullptr) {
                error = dl_last_error("dlsym token-LM training kernels failed");
            }
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* weight = nullptr;
    float* grad_weight = nullptr;
    float* exp_avg = nullptr;
    float* exp_avg_sq = nullptr;
    float* hidden = nullptr;
    float* logits = nullptr;
    float* loss_partials = nullptr;
    float* grad_logits = nullptr;
    float* grad_hidden = nullptr;
    float* ce_row_max = nullptr;
    float* ce_row_denom = nullptr;
    std::int64_t* token_ids = nullptr;
    std::int64_t* targets = nullptr;
    auto allocate = [&](void** ptr, std::size_t bytes, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_malloc(ptr, bytes);
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(reinterpret_cast<void**>(&weight), sizeof(float) * static_cast<std::size_t>(weight_elements), "weight");
    allocate(reinterpret_cast<void**>(&grad_weight), sizeof(float) * static_cast<std::size_t>(weight_elements), "grad_weight");
    allocate(reinterpret_cast<void**>(&exp_avg), sizeof(float) * static_cast<std::size_t>(weight_elements), "exp_avg");
    allocate(reinterpret_cast<void**>(&exp_avg_sq), sizeof(float) * static_cast<std::size_t>(weight_elements), "exp_avg_sq");
    allocate(reinterpret_cast<void**>(&hidden), sizeof(float) * static_cast<std::size_t>(activation_elements), "hidden");
    allocate(reinterpret_cast<void**>(&logits), sizeof(float) * static_cast<std::size_t>(logit_elements), "logits");
    allocate(reinterpret_cast<void**>(&loss_partials), sizeof(float), "loss_partials");
    allocate(reinterpret_cast<void**>(&grad_logits), sizeof(float) * static_cast<std::size_t>(logit_elements), "grad_logits");
    allocate(reinterpret_cast<void**>(&grad_hidden), sizeof(float) * static_cast<std::size_t>(activation_elements), "grad_hidden");
    allocate(reinterpret_cast<void**>(&ce_row_max), sizeof(float) * static_cast<std::size_t>(max_rows), "ce_row_max");
    allocate(reinterpret_cast<void**>(&ce_row_denom), sizeof(float) * static_cast<std::size_t>(max_rows), "ce_row_denom");
    allocate(reinterpret_cast<void**>(&token_ids), sizeof(std::int64_t) * static_cast<std::size_t>(max_rows), "token_ids");
    allocate(reinterpret_cast<void**>(&targets), sizeof(std::int64_t) * static_cast<std::size_t>(max_rows), "targets");

    auto run = [&](int status, const std::string& name) {
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    if (error.empty()) {
        run(fill(weight, weight_elements, kInitialWeight, nullptr), "weight.fill");
    }
    if (error.empty()) {
        run(fill(grad_weight, weight_elements, 0.0f, nullptr), "grad_weight.fill");
    }
    if (error.empty()) {
        run(fill(exp_avg, weight_elements, 0.0f, nullptr), "exp_avg.fill");
    }
    if (error.empty()) {
        run(fill(exp_avg_sq, weight_elements, 0.0f, nullptr), "exp_avg_sq.fill");
    }
    if (error.empty()) {
        run(fill(grad_hidden, activation_elements, 0.0f, nullptr), "grad_hidden.fill");
    }

    neuralfn::native_train::SequentialTokenBatchSampler sampler(dataset.train_shards, plan.train_seq_len, plan.batch_size);
    neuralfn::native_train::SequentialTokenBatchSampler val_sampler(
        dataset.val_shards, plan.train_seq_len, plan.eval_batch_size);
    neuralfn::native_train::TokenBatch batch;
    neuralfn::native_train::TokenBatch val_batch;
    std::vector<std::int64_t> host_tokens(static_cast<std::size_t>(max_rows), 0);
    std::vector<std::int64_t> host_targets(static_cast<std::size_t>(max_rows), 0);
    std::vector<float> host_loss(1, 0.0f);

    auto load_batch = [&](const neuralfn::native_train::TokenBatch& source, std::int64_t row_count, std::string_view label) {
        if (!error.empty()) {
            return;
        }
        if (static_cast<std::int64_t>(source.tokens.size()) != row_count ||
            static_cast<std::int64_t>(source.targets.size()) != row_count) {
            std::ostringstream out;
            out << label << " token batch has wrong size: tokens=" << source.tokens.size()
                << " targets=" << source.targets.size() << " expected=" << row_count;
            error = out.str();
            return;
        }
        for (std::int64_t i = 0; i < row_count; ++i) {
            const std::uint16_t token = source.tokens[static_cast<std::size_t>(i)];
            const std::uint16_t target = source.targets[static_cast<std::size_t>(i)];
            if (token >= vocab || target >= vocab) {
                std::ostringstream out;
                out << label << " token/target id exceeds --vocab-size at row " << i
                    << ": token=" << token << " target=" << target << " vocab=" << vocab;
                error = out.str();
                return;
            }
            host_tokens[static_cast<std::size_t>(i)] = static_cast<std::int64_t>(token);
            host_targets[static_cast<std::size_t>(i)] = static_cast<std::int64_t>(target);
        }
        run(cuda_memcpy(token_ids, host_tokens.data(), sizeof(std::int64_t) * static_cast<std::size_t>(row_count),
                kCudaMemcpyHostToDevice),
            std::string(label) + ".cudaMemcpy token_ids");
        run(cuda_memcpy(targets, host_targets.data(), sizeof(std::int64_t) * static_cast<std::size_t>(row_count),
                kCudaMemcpyHostToDevice),
            std::string(label) + ".cudaMemcpy targets");
    };

    auto forward_loss = [&](const neuralfn::native_train::TokenBatch& source, std::int64_t row_count, std::string_view label)
        -> double {
        load_batch(source, row_count, label);
        if (error.empty()) {
            run(token_embedding(weight, token_ids, hidden, row_count, dim, nullptr), std::string(label) + ".embedding.forward");
        }
        if (error.empty()) {
            run(linear(hidden, weight, nullptr, logits, row_count, dim, vocab, false, nullptr),
                std::string(label) + ".lm_head.forward");
        }
        if (error.empty()) {
            run(ce_partials(logits, targets, loss_partials, row_count, vocab, nullptr),
                std::string(label) + ".token_cross_entropy.partials");
        }
        if (error.empty()) {
            run(cuda_device_synchronize(), std::string(label) + ".cudaDeviceSynchronize");
        }
        if (error.empty()) {
            run(cuda_memcpy(host_loss.data(), loss_partials, sizeof(float), kCudaMemcpyDeviceToHost),
                std::string(label) + ".cudaMemcpy loss");
        }
        return error.empty() ? static_cast<double>(host_loss[0]) : 0.0;
    };

    auto run_validation = [&](std::int64_t step) {
        if (!error.empty() || plan.eval_every_steps <= 0 || plan.eval_batches <= 0) {
            return;
        }
        ValidationLossRecord record;
        record.step = step;
        for (std::int64_t batch_index = 0; batch_index < plan.eval_batches; ++batch_index) {
            if (!val_sampler.next(val_batch)) {
                val_sampler.reset();
                if (!val_sampler.next(val_batch)) {
                    error = "not enough validation tokens to build one native token batch";
                    break;
                }
            }
            const double loss_sum = forward_loss(val_batch, eval_rows, "validation");
            if (!error.empty()) {
                break;
            }
            record.batches += 1;
            record.tokens += eval_rows;
            record.loss_sum += loss_sum;
        }
        if (error.empty() && record.batches > 0 && record.tokens > 0) {
            record.loss_mean = record.loss_sum / static_cast<double>(record.tokens);
            validation_losses.push_back(record);
        }
    };

    for (std::int64_t step = 1; step <= plan.max_steps && error.empty(); ++step) {
        if (!sampler.next(batch)) {
            sampler.reset();
            epochs_completed += 1;
            if (!sampler.next(batch)) {
                error = "not enough train tokens to build one native token batch";
                break;
            }
        }
        const double train_loss_sum = forward_loss(batch, train_rows, "train");
        if (error.empty()) {
            run(fill(grad_weight, weight_elements, 0.0f, nullptr), "grad_weight.zero");
        }
        if (error.empty()) {
            run(fill(grad_hidden, activation_elements, 0.0f, nullptr), "grad_hidden.zero");
        }
        if (error.empty()) {
            run(ce_backward_workspace(
                    logits,
                    targets,
                    ce_row_max,
                    ce_row_denom,
                    grad_logits,
                    train_rows,
                    vocab,
                    1.0f / static_cast<float>(train_rows),
                    nullptr),
                "token_cross_entropy.backward");
        }
        if (error.empty()) {
            run(linear_backward_input(grad_logits, weight, grad_hidden, train_rows, dim, vocab, nullptr),
                "tied_lm_head.backward_input");
        }
        if (error.empty()) {
            run(linear_backward_weight(hidden, grad_logits, grad_weight, train_rows, dim, vocab, nullptr),
                "tied_lm_head.backward_weight");
        }
        if (error.empty()) {
            run(token_embedding_backward_weight(token_ids, grad_hidden, grad_weight, train_rows, dim, nullptr),
                "token_embedding.backward_weight");
        }
        const float bias_correction1 = 1.0f - std::pow(static_cast<float>(plan.beta1), static_cast<float>(step));
        const float sqrt_bias_correction2 =
            std::sqrt(1.0f - std::pow(static_cast<float>(plan.beta2), static_cast<float>(step)));
        if (error.empty()) {
            run(adamw(
                    weight,
                    grad_weight,
                    exp_avg,
                    exp_avg_sq,
                    weight_elements,
                    static_cast<float>(plan.learning_rate),
                    static_cast<float>(plan.beta1),
                    static_cast<float>(plan.beta2),
                    static_cast<float>(plan.adam_eps),
                    static_cast<float>(plan.weight_decay),
                    bias_correction1,
                    sqrt_bias_correction2,
                    nullptr),
                "adamw.step");
        }
        if (error.empty()) {
            run(cuda_device_synchronize(), "train.cudaDeviceSynchronize");
        }
        if (!error.empty()) {
            break;
        }
        steps_completed = step;
        tokens_processed += train_rows;
        final_loss_sum = train_loss_sum;
        final_loss_mean = final_loss_sum / static_cast<double>(train_rows);
        if (plan.eval_every_steps > 0 && (step % plan.eval_every_steps) == 0) {
            run_validation(step);
        }
    }

    passed = error.empty() && steps_completed == plan.max_steps && std::isfinite(final_loss_mean);

    auto free_device = [&](void* ptr, const std::string& name) {
        if (ptr != nullptr && cuda_free != nullptr) {
            int status = cuda_free(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFree " + name);
            }
        }
    };
    free_device(weight, "weight");
    free_device(grad_weight, "grad_weight");
    free_device(exp_avg, "exp_avg");
    free_device(exp_avg_sq, "exp_avg_sq");
    free_device(hidden, "hidden");
    free_device(logits, "logits");
    free_device(loss_partials, "loss_partials");
    free_device(grad_logits, "grad_logits");
    free_device(grad_hidden, "grad_hidden");
    free_device(ce_row_max, "ce_row_max");
    free_device(ce_row_denom, "ce_row_denom");
    free_device(token_ids, "token_ids");
    free_device(targets, "targets");
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"nanogpt\",\n"
        << "  \"status\": \"" << (passed ? "native-token-lm-trained" : "native-token-lm-failed") << "\",\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"dataset_loaded\": " << (dataset_loaded ? "true" : "false") << ",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"token_shards\": ";
    if (dataset_loaded) {
        std::cout << neuralfn::native_train::token_shard_dataset_json(dataset, &batch_plan);
    } else {
        std::cout << "null";
    }
    std::cout << ",\n"
        << "  \"rows\": " << train_rows << ",\n"
        << "  \"eval_rows\": " << eval_rows << ",\n"
        << "  \"vocab\": " << vocab << ",\n"
        << "  \"model_dim\": " << dim << ",\n"
        << "  \"max_steps\": " << plan.max_steps << ",\n"
        << "  \"eval_every_steps\": " << plan.eval_every_steps << ",\n"
        << "  \"eval_batches\": " << plan.eval_batches << ",\n"
        << "  \"eval_batch_size\": " << plan.eval_batch_size << ",\n"
        << "  \"steps_completed\": " << steps_completed << ",\n"
        << "  \"epochs_completed\": " << epochs_completed << ",\n"
        << "  \"tokens_processed\": " << tokens_processed << ",\n"
        << "  \"learning_rate\": " << plan.learning_rate << ",\n"
        << "  \"weight_decay\": " << plan.weight_decay << ",\n"
        << "  \"initial_weight\": " << kInitialWeight << ",\n"
        << "  \"final_loss_sum\": " << final_loss_sum << ",\n"
        << "  \"final_loss_mean\": " << final_loss_mean << ",\n"
        << "  \"validation\": {\n"
        << "    \"eval_every_steps\": " << plan.eval_every_steps << ",\n"
        << "    \"eval_batches\": " << plan.eval_batches << ",\n"
        << "    \"eval_batch_size\": " << plan.eval_batch_size << ",\n"
        << "    \"eval_count\": " << validation_losses.size() << ",\n"
        << "    \"losses\": [\n";
    for (std::size_t i = 0; i < validation_losses.size(); ++i) {
        const ValidationLossRecord& record = validation_losses[i];
        std::cout
            << "      {\"step\": " << record.step
            << ", \"batches\": " << record.batches
            << ", \"tokens\": " << record.tokens
            << ", \"loss_sum\": " << record.loss_sum
            << ", \"loss_mean\": " << record.loss_mean << "}";
        if (i + 1 != validation_losses.size()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout
        << "    ]\n"
        << "  },\n"
        << "  \"kernels\": [\n"
        << "    \"nfn_native_tile_fill_float32\",\n"
        << "    \"nfn_native_tile_token_embedding_float32\",\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_token_cross_entropy_partials_float32\",\n"
        << "    \"nfn_native_tile_token_cross_entropy_backward_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_token_embedding_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

int print_embedding_norm_step_smoke_json(const NanoGptPlan& plan, const char* program) {
    constexpr float kInitialTokenWeight = 0.1f;
    constexpr float kInitialPositionWeight = 0.2f;
    constexpr float kInitialLnWeight = 1.0f;
    constexpr float kInitialLnBias = 0.0f;
    constexpr float kResidualScale = 1.0f;
    constexpr float kLearningRate = 0.1f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.95f;
    constexpr float kEps = 1e-8f;
    constexpr float kNormEps = 1e-5f;
    constexpr float kWeightDecay = 0.1f;
    constexpr float kBiasCorrection1 = 1.0f - kBeta1;
    constexpr float kSqrtBiasCorrection2 = 0.22360679775f;
    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    neuralfn::native_train::TokenShardDataset dataset;
    neuralfn::native_train::BatchPlan batch_plan;
    neuralfn::native_train::TokenBatch batch;
    bool have_dataset = false;
    bool have_batch = false;
    std::string error;

    try {
        dataset = neuralfn::native_train::resolve_token_shards(plan.dataset_alias, plan.allow_train_as_val);
        batch_plan = neuralfn::native_train::build_batch_plan(
            dataset, plan.train_seq_len, plan.batch_size, plan.train_batch_tokens);
        have_dataset = true;
        neuralfn::native_train::SequentialTokenBatchSampler sampler(
            dataset.train_shards, plan.train_seq_len, plan.batch_size);
        have_batch = sampler.next(batch);
        if (!have_batch) {
            error = "not enough train tokens to build one native token batch";
        }
    } catch (const std::exception& exc) {
        error = exc.what();
    }

    const std::int64_t rows = have_batch ? static_cast<std::int64_t>(batch.tokens.size()) : 0;
    const std::int64_t vocab = plan.vocab_size;
    const std::int64_t dim = plan.model_dim;
    const std::int64_t seq_len = plan.train_seq_len;
    const std::int64_t batch_size = plan.batch_size;
    const std::int64_t token_weight_elements = vocab * dim;
    const std::int64_t position_weight_elements = seq_len * dim;
    const std::int64_t activation_elements = rows * dim;
    const std::int64_t logit_elements = rows * vocab;
    std::vector<std::int64_t> host_tokens(static_cast<std::size_t>(rows), 0);
    std::vector<std::int64_t> host_targets(static_cast<std::size_t>(rows), 0);

    if (error.empty()) {
        if (rows <= 0) {
            error = "sampled token batch is empty";
        } else if (vocab <= 0 || dim <= 0 || seq_len <= 0 || batch_size <= 0) {
            error = "vocab, model_dim, seq_len, and batch_size must be positive";
        } else {
            for (std::int64_t i = 0; i < rows; ++i) {
                host_tokens[static_cast<std::size_t>(i)] = static_cast<std::int64_t>(batch.tokens[static_cast<std::size_t>(i)]);
                host_targets[static_cast<std::size_t>(i)] =
                    static_cast<std::int64_t>(batch.targets[static_cast<std::size_t>(i)]);
                if (host_tokens[static_cast<std::size_t>(i)] < 0 || host_tokens[static_cast<std::size_t>(i)] >= vocab ||
                    host_targets[static_cast<std::size_t>(i)] < 0 || host_targets[static_cast<std::size_t>(i)] >= vocab) {
                    std::ostringstream out;
                    out << "sampled token/target id exceeds --vocab-size at row " << i
                        << ": token=" << host_tokens[static_cast<std::size_t>(i)]
                        << " target=" << host_targets[static_cast<std::size_t>(i)]
                        << " vocab=" << vocab;
                    error = out.str();
                    break;
                }
            }
        }
    }

    const float expected_residual = kInitialTokenWeight + kResidualScale * kInitialPositionWeight;
    const float expected_norm = 0.0f;
    const float expected_loss = rows > 0 && vocab > 0
        ? static_cast<float>(rows) * std::log(static_cast<float>(vocab))
        : 0.0f;
    const float expected_token_weight = adamw_expected_param_scalar(
        kInitialTokenWeight, 0.0f, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2);
    const float expected_position_weight = adamw_expected_param_scalar(
        kInitialPositionWeight, 0.0f, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2);
    const float expected_ln_weight = adamw_expected_param_scalar(
        kInitialLnWeight, 0.0f, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2);
    const float expected_ln_bias = adamw_expected_param_scalar(
        kInitialLnBias, 0.0f, kLearningRate, kBeta1, kBeta2, kEps, 0.0f, kBiasCorrection1, kSqrtBiasCorrection2);

    const std::string tile_lib_path = resolve_tile_ops_lib(plan, program);
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(plan);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool passed = false;
    double max_residual_abs_error = 0.0;
    double max_norm_abs_error = 0.0;
    double loss_abs_error = 0.0;
    double max_token_grad_abs_error = 0.0;
    double max_position_grad_abs_error = 0.0;
    double max_ln_grad_abs_error = 0.0;
    double max_token_weight_abs_error = 0.0;
    double max_position_weight_abs_error = 0.0;
    double max_ln_weight_abs_error = 0.0;
    double max_ln_bias_abs_error = 0.0;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using TokenEmbeddingFn = int (*)(const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenEmbeddingBackwardWeightFn = int (*)(
        const std::int64_t*, const float*, float*, std::int64_t, std::int64_t, void*);
    using PositionEmbeddingFn = int (*)(const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using PositionEmbeddingBackwardFn = int (*)(const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using ResidualAddFn = int (*)(const float*, const float*, const float*, float*, std::int64_t, void*);
    using LayerNormFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, float, void*);
    using LayerNormBackwardInputFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, float, void*);
    using LayerNormBackwardAffineFn = int (*)(
        const float*, const float*, float*, float*, std::int64_t, std::int64_t, float, void*);
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyPartialsFn = int (*)(
        const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyBackwardFn = int (*)(
        const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, float, void*);
    using AdamWFn = int (*)(
        float*,
        const float*,
        float*,
        float*,
        std::int64_t,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        void*);
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = nullptr;
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    TokenEmbeddingFn token_embedding = nullptr;
    TokenEmbeddingBackwardWeightFn token_embedding_backward_weight = nullptr;
    PositionEmbeddingFn position_embedding = nullptr;
    PositionEmbeddingBackwardFn position_embedding_backward = nullptr;
    ResidualAddFn residual_add = nullptr;
    LayerNormFn layer_norm = nullptr;
    LayerNormBackwardInputFn layer_norm_backward_input = nullptr;
    LayerNormBackwardAffineFn layer_norm_backward_affine = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightFn linear_backward_weight = nullptr;
    TokenCrossEntropyPartialsFn ce_partials = nullptr;
    TokenCrossEntropyBackwardFn ce_backward = nullptr;
    AdamWFn adamw = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (error.empty()) {
        tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            error = dl_last_error("dlopen tile ops failed");
        } else {
            tile_loaded = true;
            fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
            token_embedding = load_symbol<TokenEmbeddingFn>(tile_handle, "nfn_native_tile_token_embedding_float32");
            token_embedding_backward_weight = load_symbol<TokenEmbeddingBackwardWeightFn>(
                tile_handle, "nfn_native_tile_token_embedding_backward_weight_float32");
            position_embedding = load_symbol<PositionEmbeddingFn>(
                tile_handle, "nfn_native_tile_absolute_position_embedding_float32");
            position_embedding_backward = load_symbol<PositionEmbeddingBackwardFn>(
                tile_handle, "nfn_native_tile_absolute_position_embedding_backward_float32");
            residual_add = load_symbol<ResidualAddFn>(tile_handle, "nfn_native_tile_scaled_residual_add_float32");
            layer_norm = load_symbol<LayerNormFn>(tile_handle, "nfn_native_tile_layer_norm_float32");
            layer_norm_backward_input = load_symbol<LayerNormBackwardInputFn>(
                tile_handle, "nfn_native_tile_layer_norm_backward_input_float32");
            layer_norm_backward_affine = load_symbol<LayerNormBackwardAffineFn>(
                tile_handle, "nfn_native_tile_layer_norm_backward_affine_float32");
            linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
            linear_backward_input = load_symbol<LinearBackwardInputFn>(
                tile_handle, "nfn_native_tile_linear_backward_input_float32");
            linear_backward_weight = load_symbol<LinearBackwardWeightFn>(
                tile_handle, "nfn_native_tile_linear_backward_weight_float32");
            ce_partials = load_symbol<TokenCrossEntropyPartialsFn>(
                tile_handle, "nfn_native_tile_token_cross_entropy_partials_float32");
            ce_backward = load_symbol<TokenCrossEntropyBackwardFn>(
                tile_handle, "nfn_native_tile_token_cross_entropy_backward_float32");
            adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
            if (fill == nullptr || token_embedding == nullptr || token_embedding_backward_weight == nullptr ||
                position_embedding == nullptr || position_embedding_backward == nullptr || residual_add == nullptr ||
                layer_norm == nullptr || layer_norm_backward_input == nullptr || layer_norm_backward_affine == nullptr ||
                linear == nullptr || linear_backward_input == nullptr || linear_backward_weight == nullptr ||
                ce_partials == nullptr || ce_backward == nullptr || adamw == nullptr) {
                error = dl_last_error("dlsym embedding-norm-step kernels failed");
            }
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* token_weight = nullptr;
    float* position_weight = nullptr;
    float* ln_weight = nullptr;
    float* ln_bias = nullptr;
    float* grad_token_weight = nullptr;
    float* grad_position_weight = nullptr;
    float* grad_ln_weight = nullptr;
    float* grad_ln_bias = nullptr;
    float* token_exp_avg = nullptr;
    float* token_exp_avg_sq = nullptr;
    float* position_exp_avg = nullptr;
    float* position_exp_avg_sq = nullptr;
    float* ln_weight_exp_avg = nullptr;
    float* ln_weight_exp_avg_sq = nullptr;
    float* ln_bias_exp_avg = nullptr;
    float* ln_bias_exp_avg_sq = nullptr;
    float* token_out = nullptr;
    float* position_out = nullptr;
    float* residual = nullptr;
    float* ln_out = nullptr;
    float* logits = nullptr;
    float* loss_partials = nullptr;
    float* grad_logits = nullptr;
    float* grad_ln = nullptr;
    float* grad_residual = nullptr;
    float* residual_scale = nullptr;
    std::int64_t* token_ids = nullptr;
    std::int64_t* targets = nullptr;

    auto allocate = [&](auto** ptr, std::size_t bytes, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_malloc(reinterpret_cast<void**>(ptr), bytes);
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(&token_weight, sizeof(float) * static_cast<std::size_t>(token_weight_elements), "token_weight");
    allocate(&position_weight, sizeof(float) * static_cast<std::size_t>(position_weight_elements), "position_weight");
    allocate(&ln_weight, sizeof(float) * static_cast<std::size_t>(dim), "ln_weight");
    allocate(&ln_bias, sizeof(float) * static_cast<std::size_t>(dim), "ln_bias");
    allocate(&grad_token_weight, sizeof(float) * static_cast<std::size_t>(token_weight_elements), "grad_token_weight");
    allocate(&grad_position_weight, sizeof(float) * static_cast<std::size_t>(position_weight_elements), "grad_position_weight");
    allocate(&grad_ln_weight, sizeof(float) * static_cast<std::size_t>(dim), "grad_ln_weight");
    allocate(&grad_ln_bias, sizeof(float) * static_cast<std::size_t>(dim), "grad_ln_bias");
    allocate(&token_exp_avg, sizeof(float) * static_cast<std::size_t>(token_weight_elements), "token_exp_avg");
    allocate(&token_exp_avg_sq, sizeof(float) * static_cast<std::size_t>(token_weight_elements), "token_exp_avg_sq");
    allocate(&position_exp_avg, sizeof(float) * static_cast<std::size_t>(position_weight_elements), "position_exp_avg");
    allocate(&position_exp_avg_sq, sizeof(float) * static_cast<std::size_t>(position_weight_elements), "position_exp_avg_sq");
    allocate(&ln_weight_exp_avg, sizeof(float) * static_cast<std::size_t>(dim), "ln_weight_exp_avg");
    allocate(&ln_weight_exp_avg_sq, sizeof(float) * static_cast<std::size_t>(dim), "ln_weight_exp_avg_sq");
    allocate(&ln_bias_exp_avg, sizeof(float) * static_cast<std::size_t>(dim), "ln_bias_exp_avg");
    allocate(&ln_bias_exp_avg_sq, sizeof(float) * static_cast<std::size_t>(dim), "ln_bias_exp_avg_sq");
    allocate(&token_out, sizeof(float) * static_cast<std::size_t>(activation_elements), "token_out");
    allocate(&position_out, sizeof(float) * static_cast<std::size_t>(activation_elements), "position_out");
    allocate(&residual, sizeof(float) * static_cast<std::size_t>(activation_elements), "residual");
    allocate(&ln_out, sizeof(float) * static_cast<std::size_t>(activation_elements), "ln_out");
    allocate(&logits, sizeof(float) * static_cast<std::size_t>(logit_elements), "logits");
    allocate(&loss_partials, sizeof(float), "loss_partials");
    allocate(&grad_logits, sizeof(float) * static_cast<std::size_t>(logit_elements), "grad_logits");
    allocate(&grad_ln, sizeof(float) * static_cast<std::size_t>(activation_elements), "grad_ln");
    allocate(&grad_residual, sizeof(float) * static_cast<std::size_t>(activation_elements), "grad_residual");
    allocate(&residual_scale, sizeof(float), "residual_scale");
    allocate(&token_ids, sizeof(std::int64_t) * static_cast<std::size_t>(rows), "token_ids");
    allocate(&targets, sizeof(std::int64_t) * static_cast<std::size_t>(rows), "targets");

    auto fill_buffer = [&](float* ptr, std::int64_t elements, float value, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = fill(ptr, elements, value, nullptr);
        if (status != 0) {
            error = cuda_error(status, "fill " + name);
        }
    };
    fill_buffer(token_weight, token_weight_elements, kInitialTokenWeight, "token_weight");
    fill_buffer(position_weight, position_weight_elements, kInitialPositionWeight, "position_weight");
    fill_buffer(ln_weight, dim, kInitialLnWeight, "ln_weight");
    fill_buffer(ln_bias, dim, kInitialLnBias, "ln_bias");
    fill_buffer(grad_token_weight, token_weight_elements, 0.0f, "grad_token_weight");
    fill_buffer(grad_position_weight, position_weight_elements, 0.0f, "grad_position_weight");
    fill_buffer(grad_ln_weight, dim, 0.0f, "grad_ln_weight");
    fill_buffer(grad_ln_bias, dim, 0.0f, "grad_ln_bias");
    fill_buffer(token_exp_avg, token_weight_elements, 0.0f, "token_exp_avg");
    fill_buffer(token_exp_avg_sq, token_weight_elements, 0.0f, "token_exp_avg_sq");
    fill_buffer(position_exp_avg, position_weight_elements, 0.0f, "position_exp_avg");
    fill_buffer(position_exp_avg_sq, position_weight_elements, 0.0f, "position_exp_avg_sq");
    fill_buffer(ln_weight_exp_avg, dim, 0.0f, "ln_weight_exp_avg");
    fill_buffer(ln_weight_exp_avg_sq, dim, 0.0f, "ln_weight_exp_avg_sq");
    fill_buffer(ln_bias_exp_avg, dim, 0.0f, "ln_bias_exp_avg");
    fill_buffer(ln_bias_exp_avg_sq, dim, 0.0f, "ln_bias_exp_avg_sq");
    fill_buffer(residual_scale, 1, kResidualScale, "residual_scale");

    auto copy_to_device = [&](void* dst, const void* src, std::size_t bytes, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_memcpy(dst, src, bytes, kCudaMemcpyHostToDevice);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy host-to-device " + name);
        }
    };
    copy_to_device(token_ids, host_tokens.data(), sizeof(std::int64_t) * host_tokens.size(), "token_ids");
    copy_to_device(targets, host_targets.data(), sizeof(std::int64_t) * host_targets.size(), "targets");

    auto run = [&](int status, const std::string& name) {
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    if (error.empty()) {
        run(token_embedding(token_weight, token_ids, token_out, rows, dim, nullptr), "token_embedding.forward");
    }
    if (error.empty()) {
        run(position_embedding(position_weight, position_out, batch_size, seq_len, dim, nullptr),
            "position_embedding.forward");
    }
    if (error.empty()) {
        run(residual_add(token_out, position_out, residual_scale, residual, activation_elements, nullptr),
            "embedding_residual_add.forward");
    }
    if (error.empty()) {
        run(layer_norm(residual, ln_weight, ln_bias, ln_out, rows, dim, kNormEps, nullptr), "layer_norm.forward");
    }
    if (error.empty()) {
        run(linear(ln_out, token_weight, nullptr, logits, rows, dim, vocab, false, nullptr), "lm_head.forward");
    }
    if (error.empty()) {
        run(ce_partials(logits, targets, loss_partials, rows, vocab, nullptr), "token_cross_entropy_partials");
    }
    if (error.empty()) {
        run(ce_backward(logits, targets, grad_logits, rows, vocab, 1.0f / static_cast<float>(rows), nullptr),
            "token_cross_entropy_backward");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_logits, token_weight, grad_ln, rows, dim, vocab, nullptr),
            "lm_head.backward_input");
    }
    if (error.empty()) {
        run(linear_backward_weight(ln_out, grad_logits, grad_token_weight, rows, dim, vocab, nullptr),
            "lm_head.backward_weight");
    }
    if (error.empty()) {
        run(layer_norm_backward_affine(residual, grad_ln, grad_ln_weight, grad_ln_bias, rows, dim, kNormEps, nullptr),
            "layer_norm.backward_affine");
    }
    if (error.empty()) {
        run(layer_norm_backward_input(residual, grad_ln, ln_weight, grad_residual, rows, dim, kNormEps, nullptr),
            "layer_norm.backward_input");
    }
    if (error.empty()) {
        run(token_embedding_backward_weight(token_ids, grad_residual, grad_token_weight, rows, dim, nullptr),
            "token_embedding.backward_weight");
    }
    if (error.empty()) {
        run(position_embedding_backward(grad_residual, grad_position_weight, batch_size, seq_len, dim, nullptr),
            "position_embedding.backward");
    }
    if (error.empty()) {
        run(adamw(token_weight, grad_token_weight, token_exp_avg, token_exp_avg_sq, token_weight_elements, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "token_weight.adamw");
    }
    if (error.empty()) {
        run(adamw(position_weight, grad_position_weight, position_exp_avg, position_exp_avg_sq, position_weight_elements, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "position_weight.adamw");
    }
    if (error.empty()) {
        run(adamw(ln_weight, grad_ln_weight, ln_weight_exp_avg, ln_weight_exp_avg_sq, dim, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "ln_weight.adamw");
    }
    if (error.empty()) {
        run(adamw(ln_bias, grad_ln_bias, ln_bias_exp_avg, ln_bias_exp_avg_sq, dim, kLearningRate, kBeta1, kBeta2, kEps, 0.0f, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "ln_bias.adamw");
    }
    if (error.empty()) {
        run(cuda_device_synchronize(), "cudaDeviceSynchronize");
    }

    std::vector<float> host_residual(static_cast<std::size_t>(std::max<std::int64_t>(activation_elements, 0)), 0.0f);
    std::vector<float> host_ln_out(static_cast<std::size_t>(std::max<std::int64_t>(activation_elements, 0)), 0.0f);
    std::vector<float> host_loss(1, 0.0f);
    std::vector<float> host_grad_token(static_cast<std::size_t>(std::max<std::int64_t>(token_weight_elements, 0)), 0.0f);
    std::vector<float> host_grad_position(static_cast<std::size_t>(std::max<std::int64_t>(position_weight_elements, 0)), 0.0f);
    std::vector<float> host_grad_ln_weight(static_cast<std::size_t>(std::max<std::int64_t>(dim, 0)), 0.0f);
    std::vector<float> host_grad_ln_bias(static_cast<std::size_t>(std::max<std::int64_t>(dim, 0)), 0.0f);
    std::vector<float> host_token_weight(static_cast<std::size_t>(std::max<std::int64_t>(token_weight_elements, 0)), 0.0f);
    std::vector<float> host_position_weight(static_cast<std::size_t>(std::max<std::int64_t>(position_weight_elements, 0)), 0.0f);
    std::vector<float> host_ln_weight(static_cast<std::size_t>(std::max<std::int64_t>(dim, 0)), 0.0f);
    std::vector<float> host_ln_bias(static_cast<std::size_t>(std::max<std::int64_t>(dim, 0)), 0.0f);
    auto copy_back = [&](float* device_ptr, std::vector<float>& host, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_memcpy(host.data(), device_ptr, sizeof(float) * host.size(), kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy device-to-host " + name);
        }
    };
    copy_back(residual, host_residual, "residual");
    copy_back(ln_out, host_ln_out, "ln_out");
    copy_back(loss_partials, host_loss, "loss_partials");
    copy_back(grad_token_weight, host_grad_token, "grad_token_weight");
    copy_back(grad_position_weight, host_grad_position, "grad_position_weight");
    copy_back(grad_ln_weight, host_grad_ln_weight, "grad_ln_weight");
    copy_back(grad_ln_bias, host_grad_ln_bias, "grad_ln_bias");
    copy_back(token_weight, host_token_weight, "token_weight");
    copy_back(position_weight, host_position_weight, "position_weight");
    copy_back(ln_weight, host_ln_weight, "ln_weight");
    copy_back(ln_bias, host_ln_bias, "ln_bias");

    if (error.empty()) {
        auto max_error = [](const std::vector<float>& values, float expected) {
            double result = 0.0;
            for (float value : values) {
                result = std::max(result, std::fabs(static_cast<double>(value) - expected));
            }
            return result;
        };
        max_residual_abs_error = max_error(host_residual, expected_residual);
        max_norm_abs_error = max_error(host_ln_out, expected_norm);
        loss_abs_error = std::fabs(static_cast<double>(host_loss[0]) - expected_loss);
        max_token_grad_abs_error = max_error(host_grad_token, 0.0f);
        max_position_grad_abs_error = max_error(host_grad_position, 0.0f);
        max_ln_grad_abs_error = std::max(max_error(host_grad_ln_weight, 0.0f), max_error(host_grad_ln_bias, 0.0f));
        max_token_weight_abs_error = max_error(host_token_weight, expected_token_weight);
        max_position_weight_abs_error = max_error(host_position_weight, expected_position_weight);
        max_ln_weight_abs_error = max_error(host_ln_weight, expected_ln_weight);
        max_ln_bias_abs_error = max_error(host_ln_bias, expected_ln_bias);
        passed = max_residual_abs_error <= 1e-6 && max_norm_abs_error <= 1e-6 && loss_abs_error <= 1e-5 &&
                 max_token_grad_abs_error <= 1e-6 && max_position_grad_abs_error <= 1e-6 &&
                 max_ln_grad_abs_error <= 1e-6 && max_token_weight_abs_error <= 1e-5 &&
                 max_position_weight_abs_error <= 1e-5 && max_ln_weight_abs_error <= 1e-5 &&
                 max_ln_bias_abs_error <= 1e-6;
        if (!passed) {
            std::ostringstream out;
            out << "embedding/norm smoke exceeded tolerance: residual=" << max_residual_abs_error
                << " norm=" << max_norm_abs_error
                << " loss=" << loss_abs_error
                << " token_grad=" << max_token_grad_abs_error
                << " position_grad=" << max_position_grad_abs_error
                << " ln_grad=" << max_ln_grad_abs_error;
            error = out.str();
        }
    }

    auto free_device = [&](void* ptr, const std::string& name) {
        if (ptr != nullptr && cuda_free != nullptr) {
            int status = cuda_free(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFree " + name);
            }
        }
    };
    free_device(token_weight, "token_weight");
    free_device(position_weight, "position_weight");
    free_device(ln_weight, "ln_weight");
    free_device(ln_bias, "ln_bias");
    free_device(grad_token_weight, "grad_token_weight");
    free_device(grad_position_weight, "grad_position_weight");
    free_device(grad_ln_weight, "grad_ln_weight");
    free_device(grad_ln_bias, "grad_ln_bias");
    free_device(token_exp_avg, "token_exp_avg");
    free_device(token_exp_avg_sq, "token_exp_avg_sq");
    free_device(position_exp_avg, "position_exp_avg");
    free_device(position_exp_avg_sq, "position_exp_avg_sq");
    free_device(ln_weight_exp_avg, "ln_weight_exp_avg");
    free_device(ln_weight_exp_avg_sq, "ln_weight_exp_avg_sq");
    free_device(ln_bias_exp_avg, "ln_bias_exp_avg");
    free_device(ln_bias_exp_avg_sq, "ln_bias_exp_avg_sq");
    free_device(token_out, "token_out");
    free_device(position_out, "position_out");
    free_device(residual, "residual");
    free_device(ln_out, "ln_out");
    free_device(logits, "logits");
    free_device(loss_partials, "loss_partials");
    free_device(grad_logits, "grad_logits");
    free_device(grad_ln, "grad_ln");
    free_device(grad_residual, "grad_residual");
    free_device(residual_scale, "residual_scale");
    free_device(token_ids, "token_ids");
    free_device(targets, "targets");
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"nanogpt\",\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"dataset_loaded\": " << (have_dataset ? "true" : "false") << ",\n"
        << "  \"batch_loaded\": " << (have_batch ? "true" : "false") << ",\n"
        << "  \"token_shards\": ";
    if (have_dataset) {
        std::cout << neuralfn::native_train::token_shard_dataset_json(dataset, &batch_plan);
    } else {
        std::cout << "null";
    }
    std::cout << ",\n"
        << "  \"sample_batch\": ";
    if (have_batch) {
        std::cout << neuralfn::native_train::token_batch_json(batch);
    } else {
        std::cout << "null";
    }
    std::cout << ",\n"
        << "  \"rows\": " << rows << ",\n"
        << "  \"batch_size\": " << batch_size << ",\n"
        << "  \"seq_len\": " << seq_len << ",\n"
        << "  \"vocab\": " << vocab << ",\n"
        << "  \"model_dim\": " << dim << ",\n"
        << "  \"kernels\": [\n"
        << "    \"nfn_native_tile_token_embedding_float32\",\n"
        << "    \"nfn_native_tile_absolute_position_embedding_float32\",\n"
        << "    \"nfn_native_tile_scaled_residual_add_float32\",\n"
        << "    \"nfn_native_tile_layer_norm_float32\",\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_token_cross_entropy_partials_float32\",\n"
        << "    \"nfn_native_tile_token_cross_entropy_backward_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_layer_norm_backward_affine_float32\",\n"
        << "    \"nfn_native_tile_layer_norm_backward_input_float32\",\n"
        << "    \"nfn_native_tile_token_embedding_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_absolute_position_embedding_backward_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"expected_loss\": " << expected_loss << ",\n"
        << "  \"max_residual_abs_error\": " << max_residual_abs_error << ",\n"
        << "  \"max_norm_abs_error\": " << max_norm_abs_error << ",\n"
        << "  \"loss_abs_error\": " << loss_abs_error << ",\n"
        << "  \"max_token_grad_abs_error\": " << max_token_grad_abs_error << ",\n"
        << "  \"max_position_grad_abs_error\": " << max_position_grad_abs_error << ",\n"
        << "  \"max_ln_grad_abs_error\": " << max_ln_grad_abs_error << ",\n"
        << "  \"max_token_weight_abs_error\": " << max_token_weight_abs_error << ",\n"
        << "  \"max_position_weight_abs_error\": " << max_position_weight_abs_error << ",\n"
        << "  \"max_ln_weight_abs_error\": " << max_ln_weight_abs_error << ",\n"
        << "  \"max_ln_bias_abs_error\": " << max_ln_bias_abs_error << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

int print_qkv_layout_step_smoke_json(const NanoGptPlan& plan, const char* program) {
    constexpr std::int64_t kRows = 2;
    constexpr std::int64_t kDim = 4;
    constexpr std::int64_t kQkvElements = kRows * kDim * 3;
    constexpr std::int64_t kProjectionElements = kRows * kDim;
    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    std::vector<float> host_qkv(static_cast<std::size_t>(kQkvElements), 0.0f);
    for (std::int64_t row = 0; row < kRows; ++row) {
        for (std::int64_t dim = 0; dim < kDim; ++dim) {
            host_qkv[static_cast<std::size_t>(row * 3 * kDim + dim)] = 100.0f * static_cast<float>(row) + static_cast<float>(dim);
            host_qkv[static_cast<std::size_t>(row * 3 * kDim + kDim + dim)] =
                100.0f * static_cast<float>(row) + 10.0f + static_cast<float>(dim);
            host_qkv[static_cast<std::size_t>(row * 3 * kDim + 2 * kDim + dim)] =
                100.0f * static_cast<float>(row) + 20.0f + static_cast<float>(dim);
        }
    }

    const std::string tile_lib_path = resolve_tile_ops_lib(plan, program);
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(plan);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool passed = false;
    double max_q_abs_error = 0.0;
    double max_k_abs_error = 0.0;
    double max_v_abs_error = 0.0;
    double max_merged_abs_error = 0.0;
    std::string error;

    using SplitQkvFn = int (*)(const float*, float*, float*, float*, std::int64_t, std::int64_t, void*);
    using MergeQkvFn = int (*)(const float*, const float*, const float*, float*, std::int64_t, std::int64_t, void*);
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    void* cuda_handle = nullptr;
    SplitQkvFn split_qkv = nullptr;
    MergeQkvFn merge_qkv = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (tile_handle == nullptr) {
        error = dl_last_error("dlopen tile ops failed");
    } else {
        tile_loaded = true;
        split_qkv = load_symbol<SplitQkvFn>(tile_handle, "nfn_native_tile_split_qkv_float32");
        merge_qkv = load_symbol<MergeQkvFn>(tile_handle, "nfn_native_tile_merge_qkv_float32");
        if (split_qkv == nullptr || merge_qkv == nullptr) {
            error = dl_last_error("dlsym QKV layout kernels failed");
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* qkv = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* merged = nullptr;
    auto allocate = [&](float** ptr, std::int64_t elements, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_malloc(reinterpret_cast<void**>(ptr), sizeof(float) * static_cast<std::size_t>(elements));
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(&qkv, kQkvElements, "qkv");
    allocate(&q, kProjectionElements, "q");
    allocate(&k, kProjectionElements, "k");
    allocate(&v, kProjectionElements, "v");
    allocate(&merged, kQkvElements, "merged");

    if (error.empty()) {
        int status = cuda_memcpy(qkv, host_qkv.data(), sizeof(float) * host_qkv.size(), kCudaMemcpyHostToDevice);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy host-to-device qkv");
        }
    }
    auto run = [&](int status, const std::string& name) {
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    if (error.empty()) {
        run(split_qkv(qkv, q, k, v, kRows, kDim, nullptr), "split_qkv");
    }
    if (error.empty()) {
        run(merge_qkv(q, k, v, merged, kRows, kDim, nullptr), "merge_qkv");
    }
    if (error.empty()) {
        run(cuda_device_synchronize(), "cudaDeviceSynchronize");
    }

    std::vector<float> host_q(static_cast<std::size_t>(kProjectionElements), 0.0f);
    std::vector<float> host_k(static_cast<std::size_t>(kProjectionElements), 0.0f);
    std::vector<float> host_v(static_cast<std::size_t>(kProjectionElements), 0.0f);
    std::vector<float> host_merged(static_cast<std::size_t>(kQkvElements), 0.0f);
    auto copy_back = [&](float* device_ptr, std::vector<float>& host, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_memcpy(host.data(), device_ptr, sizeof(float) * host.size(), kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy device-to-host " + name);
        }
    };
    copy_back(q, host_q, "q");
    copy_back(k, host_k, "k");
    copy_back(v, host_v, "v");
    copy_back(merged, host_merged, "merged");

    if (error.empty()) {
        for (std::int64_t row = 0; row < kRows; ++row) {
            for (std::int64_t dim = 0; dim < kDim; ++dim) {
                const std::size_t flat = static_cast<std::size_t>(row * kDim + dim);
                const std::size_t base = static_cast<std::size_t>(row * 3 * kDim + dim);
                max_q_abs_error = std::max(max_q_abs_error, std::fabs(static_cast<double>(host_q[flat] - host_qkv[base])));
                max_k_abs_error =
                    std::max(max_k_abs_error, std::fabs(static_cast<double>(host_k[flat] - host_qkv[base + kDim])));
                max_v_abs_error =
                    std::max(max_v_abs_error, std::fabs(static_cast<double>(host_v[flat] - host_qkv[base + 2 * kDim])));
            }
        }
        for (std::size_t i = 0; i < host_qkv.size(); ++i) {
            max_merged_abs_error =
                std::max(max_merged_abs_error, std::fabs(static_cast<double>(host_merged[i] - host_qkv[i])));
        }
        passed = max_q_abs_error <= 1e-6 && max_k_abs_error <= 1e-6 && max_v_abs_error <= 1e-6 &&
                 max_merged_abs_error <= 1e-6;
        if (!passed) {
            std::ostringstream out;
            out << "QKV layout smoke exceeded tolerance: q=" << max_q_abs_error
                << " k=" << max_k_abs_error
                << " v=" << max_v_abs_error
                << " merged=" << max_merged_abs_error;
            error = out.str();
        }
    }

    auto free_device = [&](void* ptr, const std::string& name) {
        if (ptr != nullptr && cuda_free != nullptr) {
            int status = cuda_free(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFree " + name);
            }
        }
    };
    free_device(qkv, "qkv");
    free_device(q, "q");
    free_device(k, "k");
    free_device(v, "v");
    free_device(merged, "merged");
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"nanogpt\",\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"rows\": " << kRows << ",\n"
        << "  \"model_dim\": " << kDim << ",\n"
        << "  \"qkv_elements\": " << kQkvElements << ",\n"
        << "  \"kernels\": [\n"
        << "    \"nfn_native_tile_split_qkv_float32\",\n"
        << "    \"nfn_native_tile_merge_qkv_float32\"\n"
        << "  ],\n"
        << "  \"max_q_abs_error\": " << max_q_abs_error << ",\n"
        << "  \"max_k_abs_error\": " << max_k_abs_error << ",\n"
        << "  \"max_v_abs_error\": " << max_v_abs_error << ",\n"
        << "  \"max_merged_abs_error\": " << max_merged_abs_error << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

float gelu_approx_scalar(float x) {
    const float z = x * 0.7071067811865476f;
    const float sign = z < 0.0f ? -1.0f : 1.0f;
    const float abs_z = std::fabs(z);
    const float t = 1.0f / (1.0f + 0.3275911f * abs_z);
    const float poly =
        (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    const float erf_approx = sign * (1.0f - poly * std::exp(-(abs_z * abs_z)));
    return 0.5f * x * (1.0f + erf_approx);
}

float gelu_backward_approx_scalar(float x, float grad_out) {
    const float z = x * 0.7071067811865476f;
    const float sign = z < 0.0f ? -1.0f : 1.0f;
    const float abs_z = std::fabs(z);
    const float t = 1.0f / (1.0f + 0.3275911f * abs_z);
    const float poly =
        (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    const float erf_approx = sign * (1.0f - poly * std::exp(-(abs_z * abs_z)));
    const float cdf = 0.5f * (1.0f + erf_approx);
    const float pdf_term = x * std::exp(-0.5f * x * x) * 0.3989422804014327f;
    return grad_out * (cdf + pdf_term);
}

float adamw_expected_param_scalar(
    float initial_param,
    float grad,
    float learning_rate,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float sqrt_bias_correction2) {
    const float next_m = (1.0f - beta1) * grad;
    const float next_v = (1.0f - beta2) * grad * grad;
    return initial_param * (1.0f - learning_rate * weight_decay) -
           (learning_rate / bias_correction1) * next_m / (std::sqrt(next_v) / sqrt_bias_correction2 + eps);
}

int print_mlp_step_smoke_json(const NanoGptPlan& plan, const char* program) {
    constexpr std::int64_t kRows = 2;
    constexpr std::int64_t kDim = 4;
    constexpr std::int64_t kHidden = 8;
    constexpr float kInputValue = 0.5f;
    constexpr float kFcWeight = 0.1f;
    constexpr float kProjWeight = 0.2f;
    constexpr float kGradOut = 0.25f;
    constexpr float kLearningRate = 0.1f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.95f;
    constexpr float kEps = 1e-8f;
    constexpr float kWeightDecay = 0.1f;
    constexpr float kBiasCorrection1 = 1.0f - kBeta1;
    constexpr float kSqrtBiasCorrection2 = 0.22360679775f;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    const float expected_fc_pre = static_cast<float>(kDim) * kInputValue * kFcWeight;
    const float expected_act = gelu_approx_scalar(expected_fc_pre);
    const float expected_out = static_cast<float>(kHidden) * expected_act * kProjWeight;
    const float expected_proj_grad = static_cast<float>(kRows) * kGradOut * expected_act;
    const float expected_grad_act = static_cast<float>(kDim) * kGradOut * kProjWeight;
    const float expected_grad_fc_pre = gelu_backward_approx_scalar(expected_fc_pre, expected_grad_act);
    const float expected_grad_x = static_cast<float>(kHidden) * expected_grad_fc_pre * kFcWeight;
    const float expected_fc_grad = static_cast<float>(kRows) * expected_grad_fc_pre * kInputValue;
    const float expected_fc_weight =
        adamw_expected_param_scalar(kFcWeight, expected_fc_grad, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2);
    const float expected_proj_weight =
        adamw_expected_param_scalar(kProjWeight, expected_proj_grad, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2);

    const std::string tile_lib_path = resolve_tile_ops_lib(plan, program);
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(plan);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool passed = false;
    double max_out_abs_error = 0.0;
    double max_grad_x_abs_error = 0.0;
    double max_fc_grad_abs_error = 0.0;
    double max_proj_grad_abs_error = 0.0;
    double max_fc_weight_abs_error = 0.0;
    double max_proj_weight_abs_error = 0.0;
    std::string error;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using GeluFn = int (*)(const float*, float*, std::int64_t, void*);
    using GeluBackwardFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
    using AdamWFn = int (*)(
        float*,
        const float*,
        float*,
        float*,
        std::int64_t,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        void*);
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightFn linear_backward_weight = nullptr;
    GeluFn gelu = nullptr;
    GeluBackwardFn gelu_backward = nullptr;
    AdamWFn adamw = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (tile_handle == nullptr) {
        error = dl_last_error("dlopen tile ops failed");
    } else {
        tile_loaded = true;
        fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
        linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
        linear_backward_input = load_symbol<LinearBackwardInputFn>(
            tile_handle, "nfn_native_tile_linear_backward_input_float32");
        linear_backward_weight = load_symbol<LinearBackwardWeightFn>(
            tile_handle, "nfn_native_tile_linear_backward_weight_float32");
        gelu = load_symbol<GeluFn>(tile_handle, "nfn_native_tile_gelu_float32");
        gelu_backward = load_symbol<GeluBackwardFn>(tile_handle, "nfn_native_tile_gelu_backward_float32");
        adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
        if (fill == nullptr || linear == nullptr || linear_backward_input == nullptr ||
            linear_backward_weight == nullptr || gelu == nullptr || gelu_backward == nullptr || adamw == nullptr) {
            error = dl_last_error("dlsym MLP-step kernels failed");
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* x = nullptr;
    float* fc_weight = nullptr;
    float* proj_weight = nullptr;
    float* fc_out = nullptr;
    float* act = nullptr;
    float* out = nullptr;
    float* grad_out = nullptr;
    float* grad_act = nullptr;
    float* grad_fc_out = nullptr;
    float* grad_x = nullptr;
    float* grad_fc_weight = nullptr;
    float* grad_proj_weight = nullptr;
    float* fc_exp_avg = nullptr;
    float* fc_exp_avg_sq = nullptr;
    float* proj_exp_avg = nullptr;
    float* proj_exp_avg_sq = nullptr;

    auto allocate = [&](float** ptr, std::int64_t elements, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_malloc(reinterpret_cast<void**>(ptr), sizeof(float) * static_cast<std::size_t>(elements));
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(&x, kRows * kDim, "x");
    allocate(&fc_weight, kHidden * kDim, "fc_weight");
    allocate(&proj_weight, kDim * kHidden, "proj_weight");
    allocate(&fc_out, kRows * kHidden, "fc_out");
    allocate(&act, kRows * kHidden, "act");
    allocate(&out, kRows * kDim, "out");
    allocate(&grad_out, kRows * kDim, "grad_out");
    allocate(&grad_act, kRows * kHidden, "grad_act");
    allocate(&grad_fc_out, kRows * kHidden, "grad_fc_out");
    allocate(&grad_x, kRows * kDim, "grad_x");
    allocate(&grad_fc_weight, kHidden * kDim, "grad_fc_weight");
    allocate(&grad_proj_weight, kDim * kHidden, "grad_proj_weight");
    allocate(&fc_exp_avg, kHidden * kDim, "fc_exp_avg");
    allocate(&fc_exp_avg_sq, kHidden * kDim, "fc_exp_avg_sq");
    allocate(&proj_exp_avg, kDim * kHidden, "proj_exp_avg");
    allocate(&proj_exp_avg_sq, kDim * kHidden, "proj_exp_avg_sq");

    auto fill_buffer = [&](float* ptr, std::int64_t elements, float value, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = fill(ptr, elements, value, nullptr);
        if (status != 0) {
            error = cuda_error(status, "fill " + name);
        }
    };
    fill_buffer(x, kRows * kDim, kInputValue, "x");
    fill_buffer(fc_weight, kHidden * kDim, kFcWeight, "fc_weight");
    fill_buffer(proj_weight, kDim * kHidden, kProjWeight, "proj_weight");
    fill_buffer(grad_out, kRows * kDim, kGradOut, "grad_out");
    fill_buffer(grad_fc_weight, kHidden * kDim, 0.0f, "grad_fc_weight");
    fill_buffer(grad_proj_weight, kDim * kHidden, 0.0f, "grad_proj_weight");
    fill_buffer(fc_exp_avg, kHidden * kDim, 0.0f, "fc_exp_avg");
    fill_buffer(fc_exp_avg_sq, kHidden * kDim, 0.0f, "fc_exp_avg_sq");
    fill_buffer(proj_exp_avg, kDim * kHidden, 0.0f, "proj_exp_avg");
    fill_buffer(proj_exp_avg_sq, kDim * kHidden, 0.0f, "proj_exp_avg_sq");

    auto run = [&](int status, const std::string& name) {
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    if (error.empty()) {
        run(linear(x, fc_weight, nullptr, fc_out, kRows, kDim, kHidden, false, nullptr), "mlp.fc.forward");
    }
    if (error.empty()) {
        run(gelu(fc_out, act, kRows * kHidden, nullptr), "gelu.forward");
    }
    if (error.empty()) {
        run(linear(act, proj_weight, nullptr, out, kRows, kHidden, kDim, false, nullptr), "mlp.proj.forward");
    }
    if (error.empty()) {
        run(linear_backward_weight(act, grad_out, grad_proj_weight, kRows, kHidden, kDim, nullptr),
            "mlp.proj.backward_weight");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_out, proj_weight, grad_act, kRows, kHidden, kDim, nullptr),
            "mlp.proj.backward_input");
    }
    if (error.empty()) {
        run(gelu_backward(fc_out, grad_act, grad_fc_out, kRows * kHidden, nullptr), "gelu.backward");
    }
    if (error.empty()) {
        run(linear_backward_weight(x, grad_fc_out, grad_fc_weight, kRows, kDim, kHidden, nullptr),
            "mlp.fc.backward_weight");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_fc_out, fc_weight, grad_x, kRows, kDim, kHidden, nullptr),
            "mlp.fc.backward_input");
    }
    if (error.empty()) {
        run(adamw(
                fc_weight,
                grad_fc_weight,
                fc_exp_avg,
                fc_exp_avg_sq,
                kHidden * kDim,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                kWeightDecay,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "mlp.fc.adamw");
    }
    if (error.empty()) {
        run(adamw(
                proj_weight,
                grad_proj_weight,
                proj_exp_avg,
                proj_exp_avg_sq,
                kDim * kHidden,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                kWeightDecay,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "mlp.proj.adamw");
    }
    if (error.empty()) {
        run(cuda_device_synchronize(), "cudaDeviceSynchronize");
    }

    std::vector<float> host_out(static_cast<std::size_t>(kRows * kDim), 0.0f);
    std::vector<float> host_grad_x(static_cast<std::size_t>(kRows * kDim), 0.0f);
    std::vector<float> host_fc_grad(static_cast<std::size_t>(kHidden * kDim), 0.0f);
    std::vector<float> host_proj_grad(static_cast<std::size_t>(kDim * kHidden), 0.0f);
    std::vector<float> host_fc_weight(static_cast<std::size_t>(kHidden * kDim), 0.0f);
    std::vector<float> host_proj_weight(static_cast<std::size_t>(kDim * kHidden), 0.0f);
    auto copy_back = [&](float* device_ptr, std::vector<float>& host, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_memcpy(host.data(), device_ptr, sizeof(float) * host.size(), kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy device-to-host " + name);
        }
    };
    copy_back(out, host_out, "out");
    copy_back(grad_x, host_grad_x, "grad_x");
    copy_back(grad_fc_weight, host_fc_grad, "grad_fc_weight");
    copy_back(grad_proj_weight, host_proj_grad, "grad_proj_weight");
    copy_back(fc_weight, host_fc_weight, "fc_weight");
    copy_back(proj_weight, host_proj_weight, "proj_weight");

    if (error.empty()) {
        for (float value : host_out) {
            max_out_abs_error = std::max(max_out_abs_error, std::fabs(static_cast<double>(value) - expected_out));
        }
        for (float value : host_grad_x) {
            max_grad_x_abs_error =
                std::max(max_grad_x_abs_error, std::fabs(static_cast<double>(value) - expected_grad_x));
        }
        for (float value : host_fc_grad) {
            max_fc_grad_abs_error = std::max(max_fc_grad_abs_error, std::fabs(static_cast<double>(value) - expected_fc_grad));
        }
        for (float value : host_proj_grad) {
            max_proj_grad_abs_error =
                std::max(max_proj_grad_abs_error, std::fabs(static_cast<double>(value) - expected_proj_grad));
        }
        for (float value : host_fc_weight) {
            max_fc_weight_abs_error =
                std::max(max_fc_weight_abs_error, std::fabs(static_cast<double>(value) - expected_fc_weight));
        }
        for (float value : host_proj_weight) {
            max_proj_weight_abs_error =
                std::max(max_proj_weight_abs_error, std::fabs(static_cast<double>(value) - expected_proj_weight));
        }
        passed = max_out_abs_error <= 1e-4 && max_grad_x_abs_error <= 1e-4 &&
                 max_fc_grad_abs_error <= 1e-4 &&
                 max_proj_grad_abs_error <= 1e-5 && max_fc_weight_abs_error <= 1e-5 &&
                 max_proj_weight_abs_error <= 1e-5;
        if (!passed) {
            std::ostringstream out_message;
            out_message << "MLP smoke exceeded tolerance: out=" << max_out_abs_error
                        << " grad_x=" << max_grad_x_abs_error
                        << " fc_grad=" << max_fc_grad_abs_error
                        << " proj_grad=" << max_proj_grad_abs_error
                        << " fc_weight=" << max_fc_weight_abs_error
                        << " proj_weight=" << max_proj_weight_abs_error;
            error = out_message.str();
        }
    }

    auto free_device = [&](void* ptr, const std::string& name) {
        if (ptr != nullptr && cuda_free != nullptr) {
            int status = cuda_free(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFree " + name);
            }
        }
    };
    free_device(x, "x");
    free_device(fc_weight, "fc_weight");
    free_device(proj_weight, "proj_weight");
    free_device(fc_out, "fc_out");
    free_device(act, "act");
    free_device(out, "out");
    free_device(grad_out, "grad_out");
    free_device(grad_act, "grad_act");
    free_device(grad_fc_out, "grad_fc_out");
    free_device(grad_x, "grad_x");
    free_device(grad_fc_weight, "grad_fc_weight");
    free_device(grad_proj_weight, "grad_proj_weight");
    free_device(fc_exp_avg, "fc_exp_avg");
    free_device(fc_exp_avg_sq, "fc_exp_avg_sq");
    free_device(proj_exp_avg, "proj_exp_avg");
    free_device(proj_exp_avg_sq, "proj_exp_avg_sq");
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"nanogpt\",\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"rows\": " << kRows << ",\n"
        << "  \"model_dim\": " << kDim << ",\n"
        << "  \"hidden_dim\": " << kHidden << ",\n"
        << "  \"kernels\": [\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_gelu_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_gelu_backward_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"expected_fc_pre\": " << expected_fc_pre << ",\n"
        << "  \"expected_act\": " << expected_act << ",\n"
        << "  \"expected_out\": " << expected_out << ",\n"
        << "  \"max_out_abs_error\": " << max_out_abs_error << ",\n"
        << "  \"max_grad_x_abs_error\": " << max_grad_x_abs_error << ",\n"
        << "  \"max_fc_grad_abs_error\": " << max_fc_grad_abs_error << ",\n"
        << "  \"max_proj_grad_abs_error\": " << max_proj_grad_abs_error << ",\n"
        << "  \"max_fc_weight_abs_error\": " << max_fc_weight_abs_error << ",\n"
        << "  \"max_proj_weight_abs_error\": " << max_proj_weight_abs_error << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

int print_attention_step_smoke_json(const NanoGptPlan& plan, const char* program) {
    constexpr std::int64_t kBatch = 1;
    constexpr std::int64_t kHeads = 1;
    constexpr std::int64_t kSeq = 2;
    constexpr std::int64_t kDim = 4;
    constexpr std::int64_t kRows = kBatch * kSeq;
    constexpr float kInputValue = 0.5f;
    constexpr float kQWeight = 0.1f;
    constexpr float kKWeight = 0.2f;
    constexpr float kVWeight = 0.3f;
    constexpr float kOutWeight = 0.25f;
    constexpr float kGradFinal = 0.125f;
    constexpr float kLearningRate = 0.1f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.95f;
    constexpr float kEps = 1e-8f;
    constexpr float kWeightDecay = 0.1f;
    constexpr float kBiasCorrection1 = 1.0f - kBeta1;
    constexpr float kSqrtBiasCorrection2 = 0.22360679775f;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    const float expected_q = static_cast<float>(kDim) * kInputValue * kQWeight;
    const float expected_k = static_cast<float>(kDim) * kInputValue * kKWeight;
    const float expected_v = static_cast<float>(kDim) * kInputValue * kVWeight;
    const float expected_attn = expected_v;
    const float expected_out = static_cast<float>(kDim) * expected_attn * kOutWeight;
    const float expected_grad_attn = static_cast<float>(kDim) * kGradFinal * kOutWeight;
    const float expected_grad_v = static_cast<float>(kSeq) * 0.5f * expected_grad_attn;
    const float expected_v_weight_grad = static_cast<float>(kRows) * expected_grad_v * kInputValue;
    const float expected_out_weight_grad = static_cast<float>(kRows) * expected_attn * kGradFinal;
    const float expected_q_weight =
        adamw_expected_param_scalar(kQWeight, 0.0f, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2);
    const float expected_k_weight =
        adamw_expected_param_scalar(kKWeight, 0.0f, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2);
    const float expected_v_weight =
        adamw_expected_param_scalar(kVWeight, expected_v_weight_grad, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2);
    const float expected_out_weight =
        adamw_expected_param_scalar(kOutWeight, expected_out_weight_grad, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2);

    const std::string tile_lib_path = resolve_tile_ops_lib(plan, program);
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(plan);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool passed = false;
    double max_q_abs_error = 0.0;
    double max_k_abs_error = 0.0;
    double max_v_abs_error = 0.0;
    double max_attn_abs_error = 0.0;
    double max_out_abs_error = 0.0;
    double max_grad_q_weight_abs_error = 0.0;
    double max_grad_k_weight_abs_error = 0.0;
    double max_grad_v_weight_abs_error = 0.0;
    double max_grad_out_weight_abs_error = 0.0;
    double max_q_weight_abs_error = 0.0;
    double max_k_weight_abs_error = 0.0;
    double max_v_weight_abs_error = 0.0;
    double max_out_weight_abs_error = 0.0;
    std::string error;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using AttentionFn = int (*)(
        const float*,
        const float*,
        const float*,
        float*,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        float,
        bool,
        bool,
        bool,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        void*);
    using AttentionBackwardFn = int (*)(
        const float*,
        const float*,
        const float*,
        const float*,
        float*,
        float*,
        float*,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        float,
        bool,
        bool,
        bool,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        void*);
    using AdamWFn = int (*)(
        float*,
        const float*,
        float*,
        float*,
        std::int64_t,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        void*);
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightFn linear_backward_weight = nullptr;
    AttentionFn attention = nullptr;
    AttentionBackwardFn attention_backward = nullptr;
    AdamWFn adamw = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (tile_handle == nullptr) {
        error = dl_last_error("dlopen tile ops failed");
    } else {
        tile_loaded = true;
        fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
        linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
        linear_backward_input = load_symbol<LinearBackwardInputFn>(
            tile_handle, "nfn_native_tile_linear_backward_input_float32");
        linear_backward_weight = load_symbol<LinearBackwardWeightFn>(
            tile_handle, "nfn_native_tile_linear_backward_weight_float32");
        attention = load_symbol<AttentionFn>(tile_handle, "nfn_native_tile_scaled_dot_product_attention_float32");
        attention_backward = load_symbol<AttentionBackwardFn>(
            tile_handle, "nfn_native_tile_scaled_dot_product_attention_backward_float32");
        adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
        if (fill == nullptr || linear == nullptr || linear_backward_input == nullptr ||
            linear_backward_weight == nullptr || attention == nullptr || attention_backward == nullptr || adamw == nullptr) {
            error = dl_last_error("dlsym attention-step kernels failed");
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* x = nullptr;
    float* q_weight = nullptr;
    float* k_weight = nullptr;
    float* v_weight = nullptr;
    float* out_weight = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* attn_out = nullptr;
    float* out = nullptr;
    float* grad_out = nullptr;
    float* grad_attn = nullptr;
    float* grad_q = nullptr;
    float* grad_k = nullptr;
    float* grad_v = nullptr;
    float* grad_x_tmp = nullptr;
    float* grad_q_weight = nullptr;
    float* grad_k_weight = nullptr;
    float* grad_v_weight = nullptr;
    float* grad_out_weight = nullptr;
    float* q_exp_avg = nullptr;
    float* q_exp_avg_sq = nullptr;
    float* k_exp_avg = nullptr;
    float* k_exp_avg_sq = nullptr;
    float* v_exp_avg = nullptr;
    float* v_exp_avg_sq = nullptr;
    float* out_exp_avg = nullptr;
    float* out_exp_avg_sq = nullptr;

    auto allocate = [&](float** ptr, std::int64_t elements, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_malloc(reinterpret_cast<void**>(ptr), sizeof(float) * static_cast<std::size_t>(elements));
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(&x, kRows * kDim, "x");
    allocate(&q_weight, kDim * kDim, "q_weight");
    allocate(&k_weight, kDim * kDim, "k_weight");
    allocate(&v_weight, kDim * kDim, "v_weight");
    allocate(&out_weight, kDim * kDim, "out_weight");
    allocate(&q, kRows * kDim, "q");
    allocate(&k, kRows * kDim, "k");
    allocate(&v, kRows * kDim, "v");
    allocate(&attn_out, kRows * kDim, "attn_out");
    allocate(&out, kRows * kDim, "out");
    allocate(&grad_out, kRows * kDim, "grad_out");
    allocate(&grad_attn, kRows * kDim, "grad_attn");
    allocate(&grad_q, kRows * kDim, "grad_q");
    allocate(&grad_k, kRows * kDim, "grad_k");
    allocate(&grad_v, kRows * kDim, "grad_v");
    allocate(&grad_x_tmp, kRows * kDim, "grad_x_tmp");
    allocate(&grad_q_weight, kDim * kDim, "grad_q_weight");
    allocate(&grad_k_weight, kDim * kDim, "grad_k_weight");
    allocate(&grad_v_weight, kDim * kDim, "grad_v_weight");
    allocate(&grad_out_weight, kDim * kDim, "grad_out_weight");
    allocate(&q_exp_avg, kDim * kDim, "q_exp_avg");
    allocate(&q_exp_avg_sq, kDim * kDim, "q_exp_avg_sq");
    allocate(&k_exp_avg, kDim * kDim, "k_exp_avg");
    allocate(&k_exp_avg_sq, kDim * kDim, "k_exp_avg_sq");
    allocate(&v_exp_avg, kDim * kDim, "v_exp_avg");
    allocate(&v_exp_avg_sq, kDim * kDim, "v_exp_avg_sq");
    allocate(&out_exp_avg, kDim * kDim, "out_exp_avg");
    allocate(&out_exp_avg_sq, kDim * kDim, "out_exp_avg_sq");

    auto fill_buffer = [&](float* ptr, std::int64_t elements, float value, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = fill(ptr, elements, value, nullptr);
        if (status != 0) {
            error = cuda_error(status, "fill " + name);
        }
    };
    fill_buffer(x, kRows * kDim, kInputValue, "x");
    fill_buffer(q_weight, kDim * kDim, kQWeight, "q_weight");
    fill_buffer(k_weight, kDim * kDim, kKWeight, "k_weight");
    fill_buffer(v_weight, kDim * kDim, kVWeight, "v_weight");
    fill_buffer(out_weight, kDim * kDim, kOutWeight, "out_weight");
    fill_buffer(grad_out, kRows * kDim, kGradFinal, "grad_out");
    fill_buffer(q_exp_avg, kDim * kDim, 0.0f, "q_exp_avg");
    fill_buffer(q_exp_avg_sq, kDim * kDim, 0.0f, "q_exp_avg_sq");
    fill_buffer(k_exp_avg, kDim * kDim, 0.0f, "k_exp_avg");
    fill_buffer(k_exp_avg_sq, kDim * kDim, 0.0f, "k_exp_avg_sq");
    fill_buffer(v_exp_avg, kDim * kDim, 0.0f, "v_exp_avg");
    fill_buffer(v_exp_avg_sq, kDim * kDim, 0.0f, "v_exp_avg_sq");
    fill_buffer(out_exp_avg, kDim * kDim, 0.0f, "out_exp_avg");
    fill_buffer(out_exp_avg_sq, kDim * kDim, 0.0f, "out_exp_avg_sq");

    auto run = [&](int status, const std::string& name) {
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    if (error.empty()) {
        run(linear(x, q_weight, nullptr, q, kRows, kDim, kDim, false, nullptr), "q_proj.forward");
    }
    if (error.empty()) {
        run(linear(x, k_weight, nullptr, k, kRows, kDim, kDim, false, nullptr), "k_proj.forward");
    }
    if (error.empty()) {
        run(linear(x, v_weight, nullptr, v, kRows, kDim, kDim, false, nullptr), "v_proj.forward");
    }
    if (error.empty()) {
        run(attention(
                q,
                k,
                v,
                attn_out,
                kRows * kDim,
                kHeads,
                kHeads,
                kSeq,
                kSeq,
                kDim,
                kDim,
                1.0f / std::sqrt(static_cast<float>(kDim)),
                false,
                false,
                false,
                0,
                0,
                0,
                0,
                nullptr),
            "attention.forward");
    }
    if (error.empty()) {
        run(linear(attn_out, out_weight, nullptr, out, kRows, kDim, kDim, false, nullptr), "out_proj.forward");
    }
    if (error.empty()) {
        run(linear_backward_weight(attn_out, grad_out, grad_out_weight, kRows, kDim, kDim, nullptr),
            "out_proj.backward_weight");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_out, out_weight, grad_attn, kRows, kDim, kDim, nullptr),
            "out_proj.backward_input");
    }
    if (error.empty()) {
        run(attention_backward(
                q,
                k,
                v,
                grad_attn,
                grad_q,
                grad_k,
                grad_v,
                kBatch,
                kHeads,
                kHeads,
                kSeq,
                kSeq,
                kDim,
                kDim,
                1.0f / std::sqrt(static_cast<float>(kDim)),
                false,
                false,
                false,
                0,
                0,
                0,
                0,
                nullptr),
            "attention.backward");
    }
    if (error.empty()) {
        run(linear_backward_weight(x, grad_q, grad_q_weight, kRows, kDim, kDim, nullptr), "q_proj.backward_weight");
    }
    if (error.empty()) {
        run(linear_backward_weight(x, grad_k, grad_k_weight, kRows, kDim, kDim, nullptr), "k_proj.backward_weight");
    }
    if (error.empty()) {
        run(linear_backward_weight(x, grad_v, grad_v_weight, kRows, kDim, kDim, nullptr), "v_proj.backward_weight");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_q, q_weight, grad_x_tmp, kRows, kDim, kDim, nullptr), "q_proj.backward_input");
    }
    if (error.empty()) {
        run(adamw(q_weight, grad_q_weight, q_exp_avg, q_exp_avg_sq, kDim * kDim, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "q_proj.adamw");
    }
    if (error.empty()) {
        run(adamw(k_weight, grad_k_weight, k_exp_avg, k_exp_avg_sq, kDim * kDim, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "k_proj.adamw");
    }
    if (error.empty()) {
        run(adamw(v_weight, grad_v_weight, v_exp_avg, v_exp_avg_sq, kDim * kDim, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "v_proj.adamw");
    }
    if (error.empty()) {
        run(adamw(out_weight, grad_out_weight, out_exp_avg, out_exp_avg_sq, kDim * kDim, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "out_proj.adamw");
    }
    if (error.empty()) {
        run(cuda_device_synchronize(), "cudaDeviceSynchronize");
    }

    std::vector<float> host_q(static_cast<std::size_t>(kRows * kDim), 0.0f);
    std::vector<float> host_k(static_cast<std::size_t>(kRows * kDim), 0.0f);
    std::vector<float> host_v(static_cast<std::size_t>(kRows * kDim), 0.0f);
    std::vector<float> host_attn(static_cast<std::size_t>(kRows * kDim), 0.0f);
    std::vector<float> host_out(static_cast<std::size_t>(kRows * kDim), 0.0f);
    std::vector<float> host_grad_q_weight(static_cast<std::size_t>(kDim * kDim), 0.0f);
    std::vector<float> host_grad_k_weight(static_cast<std::size_t>(kDim * kDim), 0.0f);
    std::vector<float> host_grad_v_weight(static_cast<std::size_t>(kDim * kDim), 0.0f);
    std::vector<float> host_grad_out_weight(static_cast<std::size_t>(kDim * kDim), 0.0f);
    std::vector<float> host_q_weight(static_cast<std::size_t>(kDim * kDim), 0.0f);
    std::vector<float> host_k_weight(static_cast<std::size_t>(kDim * kDim), 0.0f);
    std::vector<float> host_v_weight(static_cast<std::size_t>(kDim * kDim), 0.0f);
    std::vector<float> host_out_weight(static_cast<std::size_t>(kDim * kDim), 0.0f);
    auto copy_back = [&](float* device_ptr, std::vector<float>& host, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_memcpy(host.data(), device_ptr, sizeof(float) * host.size(), kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy device-to-host " + name);
        }
    };
    copy_back(q, host_q, "q");
    copy_back(k, host_k, "k");
    copy_back(v, host_v, "v");
    copy_back(attn_out, host_attn, "attn_out");
    copy_back(out, host_out, "out");
    copy_back(grad_q_weight, host_grad_q_weight, "grad_q_weight");
    copy_back(grad_k_weight, host_grad_k_weight, "grad_k_weight");
    copy_back(grad_v_weight, host_grad_v_weight, "grad_v_weight");
    copy_back(grad_out_weight, host_grad_out_weight, "grad_out_weight");
    copy_back(q_weight, host_q_weight, "q_weight");
    copy_back(k_weight, host_k_weight, "k_weight");
    copy_back(v_weight, host_v_weight, "v_weight");
    copy_back(out_weight, host_out_weight, "out_weight");

    if (error.empty()) {
        auto max_error = [](const std::vector<float>& values, float expected) {
            double result = 0.0;
            for (float value : values) {
                result = std::max(result, std::fabs(static_cast<double>(value) - expected));
            }
            return result;
        };
        max_q_abs_error = max_error(host_q, expected_q);
        max_k_abs_error = max_error(host_k, expected_k);
        max_v_abs_error = max_error(host_v, expected_v);
        max_attn_abs_error = max_error(host_attn, expected_attn);
        max_out_abs_error = max_error(host_out, expected_out);
        max_grad_q_weight_abs_error = max_error(host_grad_q_weight, 0.0f);
        max_grad_k_weight_abs_error = max_error(host_grad_k_weight, 0.0f);
        max_grad_v_weight_abs_error = max_error(host_grad_v_weight, expected_v_weight_grad);
        max_grad_out_weight_abs_error = max_error(host_grad_out_weight, expected_out_weight_grad);
        max_q_weight_abs_error = max_error(host_q_weight, expected_q_weight);
        max_k_weight_abs_error = max_error(host_k_weight, expected_k_weight);
        max_v_weight_abs_error = max_error(host_v_weight, expected_v_weight);
        max_out_weight_abs_error = max_error(host_out_weight, expected_out_weight);
        passed = max_q_abs_error <= 1e-4 && max_k_abs_error <= 1e-4 && max_v_abs_error <= 1e-4 &&
                 max_attn_abs_error <= 1e-4 && max_out_abs_error <= 1e-4 &&
                 max_grad_q_weight_abs_error <= 1e-6 && max_grad_k_weight_abs_error <= 1e-6 &&
                 max_grad_v_weight_abs_error <= 1e-4 && max_grad_out_weight_abs_error <= 1e-4 &&
                 max_q_weight_abs_error <= 1e-5 && max_k_weight_abs_error <= 1e-5 &&
                 max_v_weight_abs_error <= 1e-5 && max_out_weight_abs_error <= 1e-5;
        if (!passed) {
            std::ostringstream out_message;
            out_message << "attention smoke exceeded tolerance: q=" << max_q_abs_error
                        << " k=" << max_k_abs_error
                        << " v=" << max_v_abs_error
                        << " attn=" << max_attn_abs_error
                        << " out=" << max_out_abs_error
                        << " grad_v_weight=" << max_grad_v_weight_abs_error
                        << " grad_out_weight=" << max_grad_out_weight_abs_error;
            error = out_message.str();
        }
    }

    auto free_device = [&](void* ptr, const std::string& name) {
        if (ptr != nullptr && cuda_free != nullptr) {
            int status = cuda_free(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFree " + name);
            }
        }
    };
    free_device(x, "x");
    free_device(q_weight, "q_weight");
    free_device(k_weight, "k_weight");
    free_device(v_weight, "v_weight");
    free_device(out_weight, "out_weight");
    free_device(q, "q");
    free_device(k, "k");
    free_device(v, "v");
    free_device(attn_out, "attn_out");
    free_device(out, "out");
    free_device(grad_out, "grad_out");
    free_device(grad_attn, "grad_attn");
    free_device(grad_q, "grad_q");
    free_device(grad_k, "grad_k");
    free_device(grad_v, "grad_v");
    free_device(grad_x_tmp, "grad_x_tmp");
    free_device(grad_q_weight, "grad_q_weight");
    free_device(grad_k_weight, "grad_k_weight");
    free_device(grad_v_weight, "grad_v_weight");
    free_device(grad_out_weight, "grad_out_weight");
    free_device(q_exp_avg, "q_exp_avg");
    free_device(q_exp_avg_sq, "q_exp_avg_sq");
    free_device(k_exp_avg, "k_exp_avg");
    free_device(k_exp_avg_sq, "k_exp_avg_sq");
    free_device(v_exp_avg, "v_exp_avg");
    free_device(v_exp_avg_sq, "v_exp_avg_sq");
    free_device(out_exp_avg, "out_exp_avg");
    free_device(out_exp_avg_sq, "out_exp_avg_sq");
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"nanogpt\",\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"batch\": " << kBatch << ",\n"
        << "  \"heads\": " << kHeads << ",\n"
        << "  \"seq\": " << kSeq << ",\n"
        << "  \"model_dim\": " << kDim << ",\n"
        << "  \"kernels\": [\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_scaled_dot_product_attention_float32\",\n"
        << "    \"nfn_native_tile_scaled_dot_product_attention_backward_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"max_q_abs_error\": " << max_q_abs_error << ",\n"
        << "  \"max_k_abs_error\": " << max_k_abs_error << ",\n"
        << "  \"max_v_abs_error\": " << max_v_abs_error << ",\n"
        << "  \"max_attn_abs_error\": " << max_attn_abs_error << ",\n"
        << "  \"max_out_abs_error\": " << max_out_abs_error << ",\n"
        << "  \"max_grad_q_weight_abs_error\": " << max_grad_q_weight_abs_error << ",\n"
        << "  \"max_grad_k_weight_abs_error\": " << max_grad_k_weight_abs_error << ",\n"
        << "  \"max_grad_v_weight_abs_error\": " << max_grad_v_weight_abs_error << ",\n"
        << "  \"max_grad_out_weight_abs_error\": " << max_grad_out_weight_abs_error << ",\n"
        << "  \"max_q_weight_abs_error\": " << max_q_weight_abs_error << ",\n"
        << "  \"max_k_weight_abs_error\": " << max_k_weight_abs_error << ",\n"
        << "  \"max_v_weight_abs_error\": " << max_v_weight_abs_error << ",\n"
        << "  \"max_out_weight_abs_error\": " << max_out_weight_abs_error << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

int print_fused_qkv_attention_step_smoke_json(const NanoGptPlan& plan, const char* program) {
    constexpr std::int64_t kBatch = 1;
    constexpr std::int64_t kHeads = 1;
    constexpr std::int64_t kSeq = 2;
    constexpr std::int64_t kDim = 4;
    constexpr std::int64_t kRows = kBatch * kSeq;
    constexpr std::int64_t kQkvDim = 3 * kDim;
    constexpr float kInputValue = 0.5f;
    constexpr float kQWeight = 0.1f;
    constexpr float kKWeight = 0.2f;
    constexpr float kVWeight = 0.3f;
    constexpr float kOutWeight = 0.25f;
    constexpr float kGradFinal = 0.125f;
    constexpr float kLearningRate = 0.1f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.95f;
    constexpr float kEps = 1e-8f;
    constexpr float kWeightDecay = 0.1f;
    constexpr float kBiasCorrection1 = 1.0f - kBeta1;
    constexpr float kSqrtBiasCorrection2 = 0.22360679775f;
    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    const float expected_q = static_cast<float>(kDim) * kInputValue * kQWeight;
    const float expected_k = static_cast<float>(kDim) * kInputValue * kKWeight;
    const float expected_v = static_cast<float>(kDim) * kInputValue * kVWeight;
    const float expected_attn = expected_v;
    const float expected_out = static_cast<float>(kDim) * expected_attn * kOutWeight;
    const float expected_grad_attn = static_cast<float>(kDim) * kGradFinal * kOutWeight;
    const float expected_grad_v = static_cast<float>(kSeq) * 0.5f * expected_grad_attn;
    const float expected_grad_x = static_cast<float>(kDim) * expected_grad_v * kVWeight;
    const float expected_v_weight_grad = static_cast<float>(kRows) * expected_grad_v * kInputValue;
    const float expected_out_weight_grad = static_cast<float>(kRows) * expected_attn * kGradFinal;
    const float expected_q_weight =
        adamw_expected_param_scalar(kQWeight, 0.0f, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2);
    const float expected_k_weight =
        adamw_expected_param_scalar(kKWeight, 0.0f, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2);
    const float expected_v_weight =
        adamw_expected_param_scalar(kVWeight, expected_v_weight_grad, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2);
    const float expected_out_weight =
        adamw_expected_param_scalar(kOutWeight, expected_out_weight_grad, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2);

    std::vector<float> host_qkv_weight(static_cast<std::size_t>(kQkvDim * kDim), 0.0f);
    for (std::int64_t row = 0; row < kQkvDim; ++row) {
        const float value = row < kDim ? kQWeight : (row < 2 * kDim ? kKWeight : kVWeight);
        for (std::int64_t col = 0; col < kDim; ++col) {
            host_qkv_weight[static_cast<std::size_t>(row * kDim + col)] = value;
        }
    }

    const std::string tile_lib_path = resolve_tile_ops_lib(plan, program);
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(plan);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool passed = false;
    double max_q_abs_error = 0.0;
    double max_k_abs_error = 0.0;
    double max_v_abs_error = 0.0;
    double max_attn_abs_error = 0.0;
    double max_out_abs_error = 0.0;
    double max_grad_x_abs_error = 0.0;
    double max_grad_qkv_weight_abs_error = 0.0;
    double max_grad_out_weight_abs_error = 0.0;
    double max_qkv_weight_abs_error = 0.0;
    double max_out_weight_abs_error = 0.0;
    std::string error;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using SplitQkvFn = int (*)(const float*, float*, float*, float*, std::int64_t, std::int64_t, void*);
    using MergeQkvFn = int (*)(const float*, const float*, const float*, float*, std::int64_t, std::int64_t, void*);
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using AttentionFn = int (*)(
        const float*,
        const float*,
        const float*,
        float*,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        float,
        bool,
        bool,
        bool,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        void*);
    using AttentionBackwardFn = int (*)(
        const float*,
        const float*,
        const float*,
        const float*,
        float*,
        float*,
        float*,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        float,
        bool,
        bool,
        bool,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        void*);
    using AdamWFn = int (*)(
        float*,
        const float*,
        float*,
        float*,
        std::int64_t,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        void*);
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    SplitQkvFn split_qkv = nullptr;
    MergeQkvFn merge_qkv = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightFn linear_backward_weight = nullptr;
    AttentionFn attention = nullptr;
    AttentionBackwardFn attention_backward = nullptr;
    AdamWFn adamw = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (tile_handle == nullptr) {
        error = dl_last_error("dlopen tile ops failed");
    } else {
        tile_loaded = true;
        fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
        split_qkv = load_symbol<SplitQkvFn>(tile_handle, "nfn_native_tile_split_qkv_float32");
        merge_qkv = load_symbol<MergeQkvFn>(tile_handle, "nfn_native_tile_merge_qkv_float32");
        linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
        linear_backward_input = load_symbol<LinearBackwardInputFn>(
            tile_handle, "nfn_native_tile_linear_backward_input_float32");
        linear_backward_weight = load_symbol<LinearBackwardWeightFn>(
            tile_handle, "nfn_native_tile_linear_backward_weight_float32");
        attention = load_symbol<AttentionFn>(tile_handle, "nfn_native_tile_scaled_dot_product_attention_float32");
        attention_backward = load_symbol<AttentionBackwardFn>(
            tile_handle, "nfn_native_tile_scaled_dot_product_attention_backward_float32");
        adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
        if (fill == nullptr || split_qkv == nullptr || merge_qkv == nullptr || linear == nullptr ||
            linear_backward_input == nullptr || linear_backward_weight == nullptr ||
            attention == nullptr || attention_backward == nullptr || adamw == nullptr) {
            error = dl_last_error("dlsym fused-QKV attention kernels failed");
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* x = nullptr;
    float* qkv_weight = nullptr;
    float* out_weight = nullptr;
    float* qkv = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* attn_out = nullptr;
    float* out = nullptr;
    float* grad_out = nullptr;
    float* grad_attn = nullptr;
    float* grad_q = nullptr;
    float* grad_k = nullptr;
    float* grad_v = nullptr;
    float* grad_qkv = nullptr;
    float* grad_x = nullptr;
    float* grad_qkv_weight = nullptr;
    float* grad_out_weight = nullptr;
    float* qkv_exp_avg = nullptr;
    float* qkv_exp_avg_sq = nullptr;
    float* out_exp_avg = nullptr;
    float* out_exp_avg_sq = nullptr;

    auto allocate = [&](float** ptr, std::int64_t elements, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_malloc(reinterpret_cast<void**>(ptr), sizeof(float) * static_cast<std::size_t>(elements));
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(&x, kRows * kDim, "x");
    allocate(&qkv_weight, kQkvDim * kDim, "qkv_weight");
    allocate(&out_weight, kDim * kDim, "out_weight");
    allocate(&qkv, kRows * kQkvDim, "qkv");
    allocate(&q, kRows * kDim, "q");
    allocate(&k, kRows * kDim, "k");
    allocate(&v, kRows * kDim, "v");
    allocate(&attn_out, kRows * kDim, "attn_out");
    allocate(&out, kRows * kDim, "out");
    allocate(&grad_out, kRows * kDim, "grad_out");
    allocate(&grad_attn, kRows * kDim, "grad_attn");
    allocate(&grad_q, kRows * kDim, "grad_q");
    allocate(&grad_k, kRows * kDim, "grad_k");
    allocate(&grad_v, kRows * kDim, "grad_v");
    allocate(&grad_qkv, kRows * kQkvDim, "grad_qkv");
    allocate(&grad_x, kRows * kDim, "grad_x");
    allocate(&grad_qkv_weight, kQkvDim * kDim, "grad_qkv_weight");
    allocate(&grad_out_weight, kDim * kDim, "grad_out_weight");
    allocate(&qkv_exp_avg, kQkvDim * kDim, "qkv_exp_avg");
    allocate(&qkv_exp_avg_sq, kQkvDim * kDim, "qkv_exp_avg_sq");
    allocate(&out_exp_avg, kDim * kDim, "out_exp_avg");
    allocate(&out_exp_avg_sq, kDim * kDim, "out_exp_avg_sq");

    auto fill_buffer = [&](float* ptr, std::int64_t elements, float value, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = fill(ptr, elements, value, nullptr);
        if (status != 0) {
            error = cuda_error(status, "fill " + name);
        }
    };
    fill_buffer(x, kRows * kDim, kInputValue, "x");
    fill_buffer(out_weight, kDim * kDim, kOutWeight, "out_weight");
    fill_buffer(grad_out, kRows * kDim, kGradFinal, "grad_out");
    fill_buffer(qkv_exp_avg, kQkvDim * kDim, 0.0f, "qkv_exp_avg");
    fill_buffer(qkv_exp_avg_sq, kQkvDim * kDim, 0.0f, "qkv_exp_avg_sq");
    fill_buffer(out_exp_avg, kDim * kDim, 0.0f, "out_exp_avg");
    fill_buffer(out_exp_avg_sq, kDim * kDim, 0.0f, "out_exp_avg_sq");
    if (error.empty()) {
        int status = cuda_memcpy(
            qkv_weight,
            host_qkv_weight.data(),
            sizeof(float) * host_qkv_weight.size(),
            kCudaMemcpyHostToDevice);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy host-to-device qkv_weight");
        }
    }

    auto run = [&](int status, const std::string& name) {
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    if (error.empty()) {
        run(linear(x, qkv_weight, nullptr, qkv, kRows, kDim, kQkvDim, false, nullptr), "qkv.forward");
    }
    if (error.empty()) {
        run(split_qkv(qkv, q, k, v, kRows, kDim, nullptr), "qkv.split");
    }
    if (error.empty()) {
        run(attention(
                q,
                k,
                v,
                attn_out,
                kRows * kDim,
                kHeads,
                kHeads,
                kSeq,
                kSeq,
                kDim,
                kDim,
                1.0f / std::sqrt(static_cast<float>(kDim)),
                false,
                false,
                false,
                0,
                0,
                0,
                0,
                nullptr),
            "attention.forward");
    }
    if (error.empty()) {
        run(linear(attn_out, out_weight, nullptr, out, kRows, kDim, kDim, false, nullptr), "out_proj.forward");
    }
    if (error.empty()) {
        run(linear_backward_weight(attn_out, grad_out, grad_out_weight, kRows, kDim, kDim, nullptr),
            "out_proj.backward_weight");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_out, out_weight, grad_attn, kRows, kDim, kDim, nullptr),
            "out_proj.backward_input");
    }
    if (error.empty()) {
        run(attention_backward(
                q,
                k,
                v,
                grad_attn,
                grad_q,
                grad_k,
                grad_v,
                kBatch,
                kHeads,
                kHeads,
                kSeq,
                kSeq,
                kDim,
                kDim,
                1.0f / std::sqrt(static_cast<float>(kDim)),
                false,
                false,
                false,
                0,
                0,
                0,
                0,
                nullptr),
            "attention.backward");
    }
    if (error.empty()) {
        run(merge_qkv(grad_q, grad_k, grad_v, grad_qkv, kRows, kDim, nullptr), "qkv.grad_merge");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_qkv, qkv_weight, grad_x, kRows, kDim, kQkvDim, nullptr),
            "qkv.backward_input");
    }
    if (error.empty()) {
        run(linear_backward_weight(x, grad_qkv, grad_qkv_weight, kRows, kDim, kQkvDim, nullptr),
            "qkv.backward_weight");
    }
    if (error.empty()) {
        run(adamw(qkv_weight, grad_qkv_weight, qkv_exp_avg, qkv_exp_avg_sq, kQkvDim * kDim, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "qkv.adamw");
    }
    if (error.empty()) {
        run(adamw(out_weight, grad_out_weight, out_exp_avg, out_exp_avg_sq, kDim * kDim, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "out_proj.adamw");
    }
    if (error.empty()) {
        run(cuda_device_synchronize(), "cudaDeviceSynchronize");
    }

    std::vector<float> host_q(static_cast<std::size_t>(kRows * kDim), 0.0f);
    std::vector<float> host_k(static_cast<std::size_t>(kRows * kDim), 0.0f);
    std::vector<float> host_v(static_cast<std::size_t>(kRows * kDim), 0.0f);
    std::vector<float> host_attn(static_cast<std::size_t>(kRows * kDim), 0.0f);
    std::vector<float> host_out(static_cast<std::size_t>(kRows * kDim), 0.0f);
    std::vector<float> host_grad_x(static_cast<std::size_t>(kRows * kDim), 0.0f);
    std::vector<float> host_grad_qkv_weight(static_cast<std::size_t>(kQkvDim * kDim), 0.0f);
    std::vector<float> host_grad_out_weight(static_cast<std::size_t>(kDim * kDim), 0.0f);
    std::vector<float> host_updated_qkv_weight(static_cast<std::size_t>(kQkvDim * kDim), 0.0f);
    std::vector<float> host_updated_out_weight(static_cast<std::size_t>(kDim * kDim), 0.0f);
    auto copy_back = [&](float* device_ptr, std::vector<float>& host, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_memcpy(host.data(), device_ptr, sizeof(float) * host.size(), kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy device-to-host " + name);
        }
    };
    copy_back(q, host_q, "q");
    copy_back(k, host_k, "k");
    copy_back(v, host_v, "v");
    copy_back(attn_out, host_attn, "attn_out");
    copy_back(out, host_out, "out");
    copy_back(grad_x, host_grad_x, "grad_x");
    copy_back(grad_qkv_weight, host_grad_qkv_weight, "grad_qkv_weight");
    copy_back(grad_out_weight, host_grad_out_weight, "grad_out_weight");
    copy_back(qkv_weight, host_updated_qkv_weight, "qkv_weight");
    copy_back(out_weight, host_updated_out_weight, "out_weight");

    if (error.empty()) {
        auto max_error = [](const std::vector<float>& values, float expected) {
            double result = 0.0;
            for (float value : values) {
                result = std::max(result, std::fabs(static_cast<double>(value) - expected));
            }
            return result;
        };
        max_q_abs_error = max_error(host_q, expected_q);
        max_k_abs_error = max_error(host_k, expected_k);
        max_v_abs_error = max_error(host_v, expected_v);
        max_attn_abs_error = max_error(host_attn, expected_attn);
        max_out_abs_error = max_error(host_out, expected_out);
        max_grad_x_abs_error = max_error(host_grad_x, expected_grad_x);
        max_grad_out_weight_abs_error = max_error(host_grad_out_weight, expected_out_weight_grad);
        max_out_weight_abs_error = max_error(host_updated_out_weight, expected_out_weight);
        for (std::int64_t row = 0; row < kQkvDim; ++row) {
            const bool is_q = row < kDim;
            const bool is_k = row >= kDim && row < 2 * kDim;
            const float expected_grad = is_q || is_k ? 0.0f : expected_v_weight_grad;
            const float expected_weight = is_q
                ? expected_q_weight
                : (is_k ? expected_k_weight : expected_v_weight);
            for (std::int64_t col = 0; col < kDim; ++col) {
                const std::size_t index = static_cast<std::size_t>(row * kDim + col);
                max_grad_qkv_weight_abs_error = std::max(
                    max_grad_qkv_weight_abs_error,
                    std::fabs(static_cast<double>(host_grad_qkv_weight[index]) - expected_grad));
                max_qkv_weight_abs_error = std::max(
                    max_qkv_weight_abs_error,
                    std::fabs(static_cast<double>(host_updated_qkv_weight[index]) - expected_weight));
            }
        }
        passed = max_q_abs_error <= 1e-4 && max_k_abs_error <= 1e-4 && max_v_abs_error <= 1e-4 &&
                 max_attn_abs_error <= 1e-4 && max_out_abs_error <= 1e-4 &&
                 max_grad_x_abs_error <= 1e-4 &&
                 max_grad_qkv_weight_abs_error <= 1e-5 && max_grad_out_weight_abs_error <= 1e-4 &&
                 max_qkv_weight_abs_error <= 1e-5 && max_out_weight_abs_error <= 1e-5;
        if (!passed) {
            std::ostringstream out_message;
            out_message << "fused-QKV attention smoke exceeded tolerance: q=" << max_q_abs_error
                        << " k=" << max_k_abs_error
                        << " v=" << max_v_abs_error
                        << " attn=" << max_attn_abs_error
                        << " out=" << max_out_abs_error
                        << " grad_x=" << max_grad_x_abs_error
                        << " grad_qkv=" << max_grad_qkv_weight_abs_error
                        << " grad_out=" << max_grad_out_weight_abs_error
                        << " qkv_weight=" << max_qkv_weight_abs_error;
            error = out_message.str();
        }
    }

    auto free_device = [&](void* ptr, const std::string& name) {
        if (ptr != nullptr && cuda_free != nullptr) {
            int status = cuda_free(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFree " + name);
            }
        }
    };
    free_device(x, "x");
    free_device(qkv_weight, "qkv_weight");
    free_device(out_weight, "out_weight");
    free_device(qkv, "qkv");
    free_device(q, "q");
    free_device(k, "k");
    free_device(v, "v");
    free_device(attn_out, "attn_out");
    free_device(out, "out");
    free_device(grad_out, "grad_out");
    free_device(grad_attn, "grad_attn");
    free_device(grad_q, "grad_q");
    free_device(grad_k, "grad_k");
    free_device(grad_v, "grad_v");
    free_device(grad_qkv, "grad_qkv");
    free_device(grad_x, "grad_x");
    free_device(grad_qkv_weight, "grad_qkv_weight");
    free_device(grad_out_weight, "grad_out_weight");
    free_device(qkv_exp_avg, "qkv_exp_avg");
    free_device(qkv_exp_avg_sq, "qkv_exp_avg_sq");
    free_device(out_exp_avg, "out_exp_avg");
    free_device(out_exp_avg_sq, "out_exp_avg_sq");
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"nanogpt\",\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"batch\": " << kBatch << ",\n"
        << "  \"heads\": " << kHeads << ",\n"
        << "  \"seq\": " << kSeq << ",\n"
        << "  \"model_dim\": " << kDim << ",\n"
        << "  \"kernels\": [\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_split_qkv_float32\",\n"
        << "    \"nfn_native_tile_scaled_dot_product_attention_float32\",\n"
        << "    \"nfn_native_tile_scaled_dot_product_attention_backward_float32\",\n"
        << "    \"nfn_native_tile_merge_qkv_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"max_q_abs_error\": " << max_q_abs_error << ",\n"
        << "  \"max_k_abs_error\": " << max_k_abs_error << ",\n"
        << "  \"max_v_abs_error\": " << max_v_abs_error << ",\n"
        << "  \"max_attn_abs_error\": " << max_attn_abs_error << ",\n"
        << "  \"max_out_abs_error\": " << max_out_abs_error << ",\n"
        << "  \"max_grad_x_abs_error\": " << max_grad_x_abs_error << ",\n"
        << "  \"max_grad_qkv_weight_abs_error\": " << max_grad_qkv_weight_abs_error << ",\n"
        << "  \"max_grad_out_weight_abs_error\": " << max_grad_out_weight_abs_error << ",\n"
        << "  \"max_qkv_weight_abs_error\": " << max_qkv_weight_abs_error << ",\n"
        << "  \"max_out_weight_abs_error\": " << max_out_weight_abs_error << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

int print_transformer_block_step_smoke_json(const NanoGptPlan& plan, const char* program) {
    constexpr std::int64_t kBatch = 1;
    constexpr std::int64_t kHeads = 1;
    constexpr std::int64_t kSeq = 2;
    constexpr std::int64_t kDim = 4;
    constexpr std::int64_t kRows = kBatch * kSeq;
    constexpr std::int64_t kHidden = 8;
    constexpr std::int64_t kQkvDim = 3 * kDim;
    constexpr float kLnWeight = 1.0f;
    constexpr float kLnBias = 0.0f;
    constexpr float kQWeight = 0.05f;
    constexpr float kKWeight = 0.07f;
    constexpr float kVWeight = 0.11f;
    constexpr float kAttnProjWeight = 0.13f;
    constexpr float kFcWeight = 0.17f;
    constexpr float kMlpProjWeight = 0.19f;
    constexpr float kGradFinal = 0.125f;
    constexpr float kResidualScale = 1.0f;
    constexpr float kLearningRate = 0.1f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.95f;
    constexpr float kEps = 1e-8f;
    constexpr float kNormEps = 1e-5f;
    constexpr float kWeightDecay = 0.1f;
    constexpr float kBiasCorrection1 = 1.0f - kBeta1;
    constexpr float kSqrtBiasCorrection2 = 0.22360679775f;
    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    const std::vector<float> host_x = {
        0.3f, -0.1f, 0.5f, -0.2f,
        -0.4f, 0.2f, 0.1f, 0.6f,
    };
    std::vector<float> host_initial_qkv_weight(static_cast<std::size_t>(kQkvDim * kDim), 0.0f);
    for (std::int64_t row = 0; row < kQkvDim; ++row) {
        const float value = row < kDim ? kQWeight : (row < 2 * kDim ? kKWeight : kVWeight);
        for (std::int64_t col = 0; col < kDim; ++col) {
            host_initial_qkv_weight[static_cast<std::size_t>(row * kDim + col)] = value;
        }
    }
    const std::vector<float> host_initial_attn_proj_weight(static_cast<std::size_t>(kDim * kDim), kAttnProjWeight);
    const std::vector<float> host_initial_fc_weight(static_cast<std::size_t>(kHidden * kDim), kFcWeight);
    const std::vector<float> host_initial_mlp_proj_weight(static_cast<std::size_t>(kDim * kHidden), kMlpProjWeight);
    const std::vector<float> host_initial_ln_weight(static_cast<std::size_t>(kDim), kLnWeight);
    const std::vector<float> host_initial_ln_bias(static_cast<std::size_t>(kDim), kLnBias);

    const std::string tile_lib_path = resolve_tile_ops_lib(plan, program);
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(plan);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool passed = false;
    bool forward_finite = false;
    bool backward_finite = false;
    bool optimizer_finite = false;
    double residual2_max_abs = 0.0;
    double grad_x_max_abs = 0.0;
    double grad_qkv_weight_max_abs = 0.0;
    double grad_attn_proj_weight_max_abs = 0.0;
    double grad_fc_weight_max_abs = 0.0;
    double grad_mlp_proj_weight_max_abs = 0.0;
    double qkv_weight_max_delta = 0.0;
    double attn_proj_weight_max_delta = 0.0;
    double fc_weight_max_delta = 0.0;
    double mlp_proj_weight_max_delta = 0.0;
    std::string error;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using GradientAccumulateFn = int (*)(float*, const float*, std::int64_t, float, void*);
    using ResidualAddFn = int (*)(const float*, const float*, const float*, float*, std::int64_t, void*);
    using SplitQkvFn = int (*)(const float*, float*, float*, float*, std::int64_t, std::int64_t, void*);
    using MergeQkvFn = int (*)(const float*, const float*, const float*, float*, std::int64_t, std::int64_t, void*);
    using LayerNormFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, float, void*);
    using LayerNormBackwardInputFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, float, void*);
    using LayerNormBackwardAffineFn = int (*)(
        const float*, const float*, float*, float*, std::int64_t, std::int64_t, float, void*);
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using GeluFn = int (*)(const float*, float*, std::int64_t, void*);
    using GeluBackwardFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
    using AttentionFn = int (*)(
        const float*,
        const float*,
        const float*,
        float*,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        float,
        bool,
        bool,
        bool,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        void*);
    using AttentionBackwardFn = int (*)(
        const float*,
        const float*,
        const float*,
        const float*,
        float*,
        float*,
        float*,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        float,
        bool,
        bool,
        bool,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        void*);
    using AdamWFn = int (*)(
        float*,
        const float*,
        float*,
        float*,
        std::int64_t,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        void*);
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    GradientAccumulateFn gradient_accumulate = nullptr;
    ResidualAddFn residual_add = nullptr;
    SplitQkvFn split_qkv = nullptr;
    MergeQkvFn merge_qkv = nullptr;
    LayerNormFn layer_norm = nullptr;
    LayerNormBackwardInputFn layer_norm_backward_input = nullptr;
    LayerNormBackwardAffineFn layer_norm_backward_affine = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightFn linear_backward_weight = nullptr;
    GeluFn gelu = nullptr;
    GeluBackwardFn gelu_backward = nullptr;
    AttentionFn attention = nullptr;
    AttentionBackwardFn attention_backward = nullptr;
    AdamWFn adamw = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (tile_handle == nullptr) {
        error = dl_last_error("dlopen tile ops failed");
    } else {
        tile_loaded = true;
        fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
        gradient_accumulate = load_symbol<GradientAccumulateFn>(
            tile_handle, "nfn_native_tile_gradient_accumulate_float32");
        residual_add = load_symbol<ResidualAddFn>(tile_handle, "nfn_native_tile_scaled_residual_add_float32");
        split_qkv = load_symbol<SplitQkvFn>(tile_handle, "nfn_native_tile_split_qkv_float32");
        merge_qkv = load_symbol<MergeQkvFn>(tile_handle, "nfn_native_tile_merge_qkv_float32");
        layer_norm = load_symbol<LayerNormFn>(tile_handle, "nfn_native_tile_layer_norm_float32");
        layer_norm_backward_input = load_symbol<LayerNormBackwardInputFn>(
            tile_handle, "nfn_native_tile_layer_norm_backward_input_float32");
        layer_norm_backward_affine = load_symbol<LayerNormBackwardAffineFn>(
            tile_handle, "nfn_native_tile_layer_norm_backward_affine_float32");
        linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
        linear_backward_input = load_symbol<LinearBackwardInputFn>(
            tile_handle, "nfn_native_tile_linear_backward_input_float32");
        linear_backward_weight = load_symbol<LinearBackwardWeightFn>(
            tile_handle, "nfn_native_tile_linear_backward_weight_float32");
        gelu = load_symbol<GeluFn>(tile_handle, "nfn_native_tile_gelu_float32");
        gelu_backward = load_symbol<GeluBackwardFn>(tile_handle, "nfn_native_tile_gelu_backward_float32");
        attention = load_symbol<AttentionFn>(tile_handle, "nfn_native_tile_scaled_dot_product_attention_float32");
        attention_backward = load_symbol<AttentionBackwardFn>(
            tile_handle, "nfn_native_tile_scaled_dot_product_attention_backward_float32");
        adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
        if (fill == nullptr || gradient_accumulate == nullptr || residual_add == nullptr ||
            split_qkv == nullptr || merge_qkv == nullptr || layer_norm == nullptr ||
            layer_norm_backward_input == nullptr || layer_norm_backward_affine == nullptr ||
            linear == nullptr || linear_backward_input == nullptr || linear_backward_weight == nullptr ||
            gelu == nullptr || gelu_backward == nullptr || attention == nullptr ||
            attention_backward == nullptr || adamw == nullptr) {
            error = dl_last_error("dlsym transformer-block kernels failed");
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* x = nullptr;
    float* residual_scale = nullptr;
    float* ln1_weight = nullptr;
    float* ln1_bias = nullptr;
    float* ln2_weight = nullptr;
    float* ln2_bias = nullptr;
    float* qkv_weight = nullptr;
    float* attn_proj_weight = nullptr;
    float* fc_weight = nullptr;
    float* mlp_proj_weight = nullptr;
    float* ln1_out = nullptr;
    float* qkv = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* attn_out = nullptr;
    float* attn_proj = nullptr;
    float* residual1 = nullptr;
    float* ln2_out = nullptr;
    float* fc_out = nullptr;
    float* act = nullptr;
    float* mlp_out = nullptr;
    float* residual2 = nullptr;
    float* grad_residual2 = nullptr;
    float* grad_act = nullptr;
    float* grad_fc_out = nullptr;
    float* grad_ln2 = nullptr;
    float* grad_residual1_from_mlp = nullptr;
    float* grad_residual1 = nullptr;
    float* grad_attn_out = nullptr;
    float* grad_q = nullptr;
    float* grad_k = nullptr;
    float* grad_v = nullptr;
    float* grad_qkv = nullptr;
    float* grad_ln1 = nullptr;
    float* grad_x_from_attn = nullptr;
    float* grad_x = nullptr;
    float* grad_ln1_weight = nullptr;
    float* grad_ln1_bias = nullptr;
    float* grad_ln2_weight = nullptr;
    float* grad_ln2_bias = nullptr;
    float* grad_qkv_weight = nullptr;
    float* grad_attn_proj_weight = nullptr;
    float* grad_fc_weight = nullptr;
    float* grad_mlp_proj_weight = nullptr;
    float* ln1_weight_exp_avg = nullptr;
    float* ln1_weight_exp_avg_sq = nullptr;
    float* ln1_bias_exp_avg = nullptr;
    float* ln1_bias_exp_avg_sq = nullptr;
    float* ln2_weight_exp_avg = nullptr;
    float* ln2_weight_exp_avg_sq = nullptr;
    float* ln2_bias_exp_avg = nullptr;
    float* ln2_bias_exp_avg_sq = nullptr;
    float* qkv_exp_avg = nullptr;
    float* qkv_exp_avg_sq = nullptr;
    float* attn_proj_exp_avg = nullptr;
    float* attn_proj_exp_avg_sq = nullptr;
    float* fc_exp_avg = nullptr;
    float* fc_exp_avg_sq = nullptr;
    float* mlp_proj_exp_avg = nullptr;
    float* mlp_proj_exp_avg_sq = nullptr;

    auto allocate = [&](float** ptr, std::int64_t elements, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_malloc(reinterpret_cast<void**>(ptr), sizeof(float) * static_cast<std::size_t>(elements));
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(&x, kRows * kDim, "x");
    allocate(&residual_scale, 1, "residual_scale");
    allocate(&ln1_weight, kDim, "ln1_weight");
    allocate(&ln1_bias, kDim, "ln1_bias");
    allocate(&ln2_weight, kDim, "ln2_weight");
    allocate(&ln2_bias, kDim, "ln2_bias");
    allocate(&qkv_weight, kQkvDim * kDim, "qkv_weight");
    allocate(&attn_proj_weight, kDim * kDim, "attn_proj_weight");
    allocate(&fc_weight, kHidden * kDim, "fc_weight");
    allocate(&mlp_proj_weight, kDim * kHidden, "mlp_proj_weight");
    allocate(&ln1_out, kRows * kDim, "ln1_out");
    allocate(&qkv, kRows * kQkvDim, "qkv");
    allocate(&q, kRows * kDim, "q");
    allocate(&k, kRows * kDim, "k");
    allocate(&v, kRows * kDim, "v");
    allocate(&attn_out, kRows * kDim, "attn_out");
    allocate(&attn_proj, kRows * kDim, "attn_proj");
    allocate(&residual1, kRows * kDim, "residual1");
    allocate(&ln2_out, kRows * kDim, "ln2_out");
    allocate(&fc_out, kRows * kHidden, "fc_out");
    allocate(&act, kRows * kHidden, "act");
    allocate(&mlp_out, kRows * kDim, "mlp_out");
    allocate(&residual2, kRows * kDim, "residual2");
    allocate(&grad_residual2, kRows * kDim, "grad_residual2");
    allocate(&grad_act, kRows * kHidden, "grad_act");
    allocate(&grad_fc_out, kRows * kHidden, "grad_fc_out");
    allocate(&grad_ln2, kRows * kDim, "grad_ln2");
    allocate(&grad_residual1_from_mlp, kRows * kDim, "grad_residual1_from_mlp");
    allocate(&grad_residual1, kRows * kDim, "grad_residual1");
    allocate(&grad_attn_out, kRows * kDim, "grad_attn_out");
    allocate(&grad_q, kRows * kDim, "grad_q");
    allocate(&grad_k, kRows * kDim, "grad_k");
    allocate(&grad_v, kRows * kDim, "grad_v");
    allocate(&grad_qkv, kRows * kQkvDim, "grad_qkv");
    allocate(&grad_ln1, kRows * kDim, "grad_ln1");
    allocate(&grad_x_from_attn, kRows * kDim, "grad_x_from_attn");
    allocate(&grad_x, kRows * kDim, "grad_x");
    allocate(&grad_ln1_weight, kDim, "grad_ln1_weight");
    allocate(&grad_ln1_bias, kDim, "grad_ln1_bias");
    allocate(&grad_ln2_weight, kDim, "grad_ln2_weight");
    allocate(&grad_ln2_bias, kDim, "grad_ln2_bias");
    allocate(&grad_qkv_weight, kQkvDim * kDim, "grad_qkv_weight");
    allocate(&grad_attn_proj_weight, kDim * kDim, "grad_attn_proj_weight");
    allocate(&grad_fc_weight, kHidden * kDim, "grad_fc_weight");
    allocate(&grad_mlp_proj_weight, kDim * kHidden, "grad_mlp_proj_weight");
    allocate(&ln1_weight_exp_avg, kDim, "ln1_weight_exp_avg");
    allocate(&ln1_weight_exp_avg_sq, kDim, "ln1_weight_exp_avg_sq");
    allocate(&ln1_bias_exp_avg, kDim, "ln1_bias_exp_avg");
    allocate(&ln1_bias_exp_avg_sq, kDim, "ln1_bias_exp_avg_sq");
    allocate(&ln2_weight_exp_avg, kDim, "ln2_weight_exp_avg");
    allocate(&ln2_weight_exp_avg_sq, kDim, "ln2_weight_exp_avg_sq");
    allocate(&ln2_bias_exp_avg, kDim, "ln2_bias_exp_avg");
    allocate(&ln2_bias_exp_avg_sq, kDim, "ln2_bias_exp_avg_sq");
    allocate(&qkv_exp_avg, kQkvDim * kDim, "qkv_exp_avg");
    allocate(&qkv_exp_avg_sq, kQkvDim * kDim, "qkv_exp_avg_sq");
    allocate(&attn_proj_exp_avg, kDim * kDim, "attn_proj_exp_avg");
    allocate(&attn_proj_exp_avg_sq, kDim * kDim, "attn_proj_exp_avg_sq");
    allocate(&fc_exp_avg, kHidden * kDim, "fc_exp_avg");
    allocate(&fc_exp_avg_sq, kHidden * kDim, "fc_exp_avg_sq");
    allocate(&mlp_proj_exp_avg, kDim * kHidden, "mlp_proj_exp_avg");
    allocate(&mlp_proj_exp_avg_sq, kDim * kHidden, "mlp_proj_exp_avg_sq");

    auto fill_buffer = [&](float* ptr, std::int64_t elements, float value, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = fill(ptr, elements, value, nullptr);
        if (status != 0) {
            error = cuda_error(status, "fill " + name);
        }
    };
    auto copy_to_device = [&](float* dst, const std::vector<float>& host, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_memcpy(dst, host.data(), sizeof(float) * host.size(), kCudaMemcpyHostToDevice);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy host-to-device " + name);
        }
    };
    copy_to_device(x, host_x, "x");
    fill_buffer(residual_scale, 1, kResidualScale, "residual_scale");
    copy_to_device(ln1_weight, host_initial_ln_weight, "ln1_weight");
    copy_to_device(ln1_bias, host_initial_ln_bias, "ln1_bias");
    copy_to_device(ln2_weight, host_initial_ln_weight, "ln2_weight");
    copy_to_device(ln2_bias, host_initial_ln_bias, "ln2_bias");
    copy_to_device(qkv_weight, host_initial_qkv_weight, "qkv_weight");
    copy_to_device(attn_proj_weight, host_initial_attn_proj_weight, "attn_proj_weight");
    copy_to_device(fc_weight, host_initial_fc_weight, "fc_weight");
    copy_to_device(mlp_proj_weight, host_initial_mlp_proj_weight, "mlp_proj_weight");
    fill_buffer(grad_residual2, kRows * kDim, kGradFinal, "grad_residual2");
    fill_buffer(grad_ln1_weight, kDim, 0.0f, "grad_ln1_weight");
    fill_buffer(grad_ln1_bias, kDim, 0.0f, "grad_ln1_bias");
    fill_buffer(grad_ln2_weight, kDim, 0.0f, "grad_ln2_weight");
    fill_buffer(grad_ln2_bias, kDim, 0.0f, "grad_ln2_bias");
    fill_buffer(grad_qkv_weight, kQkvDim * kDim, 0.0f, "grad_qkv_weight");
    fill_buffer(grad_attn_proj_weight, kDim * kDim, 0.0f, "grad_attn_proj_weight");
    fill_buffer(grad_fc_weight, kHidden * kDim, 0.0f, "grad_fc_weight");
    fill_buffer(grad_mlp_proj_weight, kDim * kHidden, 0.0f, "grad_mlp_proj_weight");
    fill_buffer(ln1_weight_exp_avg, kDim, 0.0f, "ln1_weight_exp_avg");
    fill_buffer(ln1_weight_exp_avg_sq, kDim, 0.0f, "ln1_weight_exp_avg_sq");
    fill_buffer(ln1_bias_exp_avg, kDim, 0.0f, "ln1_bias_exp_avg");
    fill_buffer(ln1_bias_exp_avg_sq, kDim, 0.0f, "ln1_bias_exp_avg_sq");
    fill_buffer(ln2_weight_exp_avg, kDim, 0.0f, "ln2_weight_exp_avg");
    fill_buffer(ln2_weight_exp_avg_sq, kDim, 0.0f, "ln2_weight_exp_avg_sq");
    fill_buffer(ln2_bias_exp_avg, kDim, 0.0f, "ln2_bias_exp_avg");
    fill_buffer(ln2_bias_exp_avg_sq, kDim, 0.0f, "ln2_bias_exp_avg_sq");
    fill_buffer(qkv_exp_avg, kQkvDim * kDim, 0.0f, "qkv_exp_avg");
    fill_buffer(qkv_exp_avg_sq, kQkvDim * kDim, 0.0f, "qkv_exp_avg_sq");
    fill_buffer(attn_proj_exp_avg, kDim * kDim, 0.0f, "attn_proj_exp_avg");
    fill_buffer(attn_proj_exp_avg_sq, kDim * kDim, 0.0f, "attn_proj_exp_avg_sq");
    fill_buffer(fc_exp_avg, kHidden * kDim, 0.0f, "fc_exp_avg");
    fill_buffer(fc_exp_avg_sq, kHidden * kDim, 0.0f, "fc_exp_avg_sq");
    fill_buffer(mlp_proj_exp_avg, kDim * kHidden, 0.0f, "mlp_proj_exp_avg");
    fill_buffer(mlp_proj_exp_avg_sq, kDim * kHidden, 0.0f, "mlp_proj_exp_avg_sq");

    auto run = [&](int status, const std::string& name) {
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    if (error.empty()) {
        run(layer_norm(x, ln1_weight, ln1_bias, ln1_out, kRows, kDim, kNormEps, nullptr), "ln1.forward");
    }
    if (error.empty()) {
        run(linear(ln1_out, qkv_weight, nullptr, qkv, kRows, kDim, kQkvDim, false, nullptr), "attn.qkv.forward");
    }
    if (error.empty()) {
        run(split_qkv(qkv, q, k, v, kRows, kDim, nullptr), "attn.qkv.split");
    }
    if (error.empty()) {
        run(attention(
                q,
                k,
                v,
                attn_out,
                kRows * kDim,
                kHeads,
                kHeads,
                kSeq,
                kSeq,
                kDim,
                kDim,
                1.0f / std::sqrt(static_cast<float>(kDim)),
                false,
                false,
                false,
                0,
                0,
                0,
                0,
                nullptr),
            "attn.sdpa.forward");
    }
    if (error.empty()) {
        run(linear(attn_out, attn_proj_weight, nullptr, attn_proj, kRows, kDim, kDim, false, nullptr),
            "attn.out.forward");
    }
    if (error.empty()) {
        run(residual_add(x, attn_proj, residual_scale, residual1, kRows * kDim, nullptr),
            "attn.residual_add.forward");
    }
    if (error.empty()) {
        run(layer_norm(residual1, ln2_weight, ln2_bias, ln2_out, kRows, kDim, kNormEps, nullptr),
            "ln2.forward");
    }
    if (error.empty()) {
        run(linear(ln2_out, fc_weight, nullptr, fc_out, kRows, kDim, kHidden, false, nullptr),
            "mlp.fc.forward");
    }
    if (error.empty()) {
        run(gelu(fc_out, act, kRows * kHidden, nullptr), "mlp.gelu.forward");
    }
    if (error.empty()) {
        run(linear(act, mlp_proj_weight, nullptr, mlp_out, kRows, kHidden, kDim, false, nullptr),
            "mlp.proj.forward");
    }
    if (error.empty()) {
        run(residual_add(residual1, mlp_out, residual_scale, residual2, kRows * kDim, nullptr),
            "mlp.residual_add.forward");
    }
    if (error.empty()) {
        run(linear_backward_weight(act, grad_residual2, grad_mlp_proj_weight, kRows, kHidden, kDim, nullptr),
            "mlp.proj.backward_weight");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_residual2, mlp_proj_weight, grad_act, kRows, kHidden, kDim, nullptr),
            "mlp.proj.backward_input");
    }
    if (error.empty()) {
        run(gelu_backward(fc_out, grad_act, grad_fc_out, kRows * kHidden, nullptr), "mlp.gelu.backward");
    }
    if (error.empty()) {
        run(linear_backward_weight(ln2_out, grad_fc_out, grad_fc_weight, kRows, kDim, kHidden, nullptr),
            "mlp.fc.backward_weight");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_fc_out, fc_weight, grad_ln2, kRows, kDim, kHidden, nullptr),
            "mlp.fc.backward_input");
    }
    if (error.empty()) {
        run(layer_norm_backward_affine(residual1, grad_ln2, grad_ln2_weight, grad_ln2_bias, kRows, kDim, kNormEps, nullptr),
            "ln2.backward_affine");
    }
    if (error.empty()) {
        run(layer_norm_backward_input(residual1, grad_ln2, ln2_weight, grad_residual1_from_mlp, kRows, kDim, kNormEps, nullptr),
            "ln2.backward_input");
    }
    if (error.empty()) {
        fill_buffer(grad_residual1, kRows * kDim, kGradFinal, "grad_residual1");
    }
    if (error.empty()) {
        run(gradient_accumulate(grad_residual1, grad_residual1_from_mlp, kRows * kDim, 1.0f, nullptr),
            "mlp.residual.backward_accumulate");
    }
    if (error.empty()) {
        run(linear_backward_weight(attn_out, grad_residual1, grad_attn_proj_weight, kRows, kDim, kDim, nullptr),
            "attn.out.backward_weight");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_residual1, attn_proj_weight, grad_attn_out, kRows, kDim, kDim, nullptr),
            "attn.out.backward_input");
    }
    if (error.empty()) {
        run(attention_backward(
                q,
                k,
                v,
                grad_attn_out,
                grad_q,
                grad_k,
                grad_v,
                kBatch,
                kHeads,
                kHeads,
                kSeq,
                kSeq,
                kDim,
                kDim,
                1.0f / std::sqrt(static_cast<float>(kDim)),
                false,
                false,
                false,
                0,
                0,
                0,
                0,
                nullptr),
            "attn.sdpa.backward");
    }
    if (error.empty()) {
        run(merge_qkv(grad_q, grad_k, grad_v, grad_qkv, kRows, kDim, nullptr), "attn.qkv.grad_merge");
    }
    if (error.empty()) {
        run(linear_backward_weight(ln1_out, grad_qkv, grad_qkv_weight, kRows, kDim, kQkvDim, nullptr),
            "attn.qkv.backward_weight");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_qkv, qkv_weight, grad_ln1, kRows, kDim, kQkvDim, nullptr),
            "attn.qkv.backward_input");
    }
    if (error.empty()) {
        run(layer_norm_backward_affine(x, grad_ln1, grad_ln1_weight, grad_ln1_bias, kRows, kDim, kNormEps, nullptr),
            "ln1.backward_affine");
    }
    if (error.empty()) {
        run(layer_norm_backward_input(x, grad_ln1, ln1_weight, grad_x_from_attn, kRows, kDim, kNormEps, nullptr),
            "ln1.backward_input");
    }
    if (error.empty()) {
        fill_buffer(grad_x, kRows * kDim, kGradFinal, "grad_x");
    }
    if (error.empty()) {
        run(gradient_accumulate(grad_x, grad_x_from_attn, kRows * kDim, 1.0f, nullptr),
            "attn.residual.backward_accumulate");
    }
    if (error.empty()) {
        run(gradient_accumulate(grad_x, grad_residual1_from_mlp, kRows * kDim, 1.0f, nullptr),
            "mlp.direct_residual.backward_accumulate");
    }
    if (error.empty()) {
        run(adamw(ln1_weight, grad_ln1_weight, ln1_weight_exp_avg, ln1_weight_exp_avg_sq, kDim, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "ln1.weight.adamw");
    }
    if (error.empty()) {
        run(adamw(ln1_bias, grad_ln1_bias, ln1_bias_exp_avg, ln1_bias_exp_avg_sq, kDim, kLearningRate, kBeta1, kBeta2, kEps, 0.0f, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "ln1.bias.adamw");
    }
    if (error.empty()) {
        run(adamw(ln2_weight, grad_ln2_weight, ln2_weight_exp_avg, ln2_weight_exp_avg_sq, kDim, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "ln2.weight.adamw");
    }
    if (error.empty()) {
        run(adamw(ln2_bias, grad_ln2_bias, ln2_bias_exp_avg, ln2_bias_exp_avg_sq, kDim, kLearningRate, kBeta1, kBeta2, kEps, 0.0f, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "ln2.bias.adamw");
    }
    if (error.empty()) {
        run(adamw(qkv_weight, grad_qkv_weight, qkv_exp_avg, qkv_exp_avg_sq, kQkvDim * kDim, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "qkv.weight.adamw");
    }
    if (error.empty()) {
        run(adamw(attn_proj_weight, grad_attn_proj_weight, attn_proj_exp_avg, attn_proj_exp_avg_sq, kDim * kDim, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "attn.out.weight.adamw");
    }
    if (error.empty()) {
        run(adamw(fc_weight, grad_fc_weight, fc_exp_avg, fc_exp_avg_sq, kHidden * kDim, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "mlp.fc.weight.adamw");
    }
    if (error.empty()) {
        run(adamw(mlp_proj_weight, grad_mlp_proj_weight, mlp_proj_exp_avg, mlp_proj_exp_avg_sq, kDim * kHidden, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "mlp.proj.weight.adamw");
    }
    if (error.empty()) {
        run(cuda_device_synchronize(), "cudaDeviceSynchronize");
    }

    std::vector<float> host_residual2(static_cast<std::size_t>(kRows * kDim), 0.0f);
    std::vector<float> host_grad_x(static_cast<std::size_t>(kRows * kDim), 0.0f);
    std::vector<float> host_grad_qkv_weight(static_cast<std::size_t>(kQkvDim * kDim), 0.0f);
    std::vector<float> host_grad_attn_proj_weight(static_cast<std::size_t>(kDim * kDim), 0.0f);
    std::vector<float> host_grad_fc_weight(static_cast<std::size_t>(kHidden * kDim), 0.0f);
    std::vector<float> host_grad_mlp_proj_weight(static_cast<std::size_t>(kDim * kHidden), 0.0f);
    std::vector<float> host_qkv_weight(static_cast<std::size_t>(kQkvDim * kDim), 0.0f);
    std::vector<float> host_attn_proj_weight(static_cast<std::size_t>(kDim * kDim), 0.0f);
    std::vector<float> host_fc_weight(static_cast<std::size_t>(kHidden * kDim), 0.0f);
    std::vector<float> host_mlp_proj_weight(static_cast<std::size_t>(kDim * kHidden), 0.0f);
    auto copy_back = [&](float* device_ptr, std::vector<float>& host, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_memcpy(host.data(), device_ptr, sizeof(float) * host.size(), kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy device-to-host " + name);
        }
    };
    copy_back(residual2, host_residual2, "residual2");
    copy_back(grad_x, host_grad_x, "grad_x");
    copy_back(grad_qkv_weight, host_grad_qkv_weight, "grad_qkv_weight");
    copy_back(grad_attn_proj_weight, host_grad_attn_proj_weight, "grad_attn_proj_weight");
    copy_back(grad_fc_weight, host_grad_fc_weight, "grad_fc_weight");
    copy_back(grad_mlp_proj_weight, host_grad_mlp_proj_weight, "grad_mlp_proj_weight");
    copy_back(qkv_weight, host_qkv_weight, "qkv_weight");
    copy_back(attn_proj_weight, host_attn_proj_weight, "attn_proj_weight");
    copy_back(fc_weight, host_fc_weight, "fc_weight");
    copy_back(mlp_proj_weight, host_mlp_proj_weight, "mlp_proj_weight");

    if (error.empty()) {
        auto all_finite = [](const std::vector<float>& values) {
            return std::all_of(values.begin(), values.end(), [](float value) {
                return std::isfinite(value);
            });
        };
        auto max_abs = [](const std::vector<float>& values) {
            double result = 0.0;
            for (float value : values) {
                result = std::max(result, std::fabs(static_cast<double>(value)));
            }
            return result;
        };
        auto max_delta = [](const std::vector<float>& before, const std::vector<float>& after) {
            double result = 0.0;
            const std::size_t n = std::min(before.size(), after.size());
            for (std::size_t i = 0; i < n; ++i) {
                result = std::max(
                    result,
                    std::fabs(static_cast<double>(after[i]) - static_cast<double>(before[i])));
            }
            return result;
        };
        residual2_max_abs = max_abs(host_residual2);
        grad_x_max_abs = max_abs(host_grad_x);
        grad_qkv_weight_max_abs = max_abs(host_grad_qkv_weight);
        grad_attn_proj_weight_max_abs = max_abs(host_grad_attn_proj_weight);
        grad_fc_weight_max_abs = max_abs(host_grad_fc_weight);
        grad_mlp_proj_weight_max_abs = max_abs(host_grad_mlp_proj_weight);
        qkv_weight_max_delta = max_delta(host_initial_qkv_weight, host_qkv_weight);
        attn_proj_weight_max_delta = max_delta(host_initial_attn_proj_weight, host_attn_proj_weight);
        fc_weight_max_delta = max_delta(host_initial_fc_weight, host_fc_weight);
        mlp_proj_weight_max_delta = max_delta(host_initial_mlp_proj_weight, host_mlp_proj_weight);
        forward_finite = all_finite(host_residual2);
        backward_finite = all_finite(host_grad_x) && all_finite(host_grad_qkv_weight) &&
                          all_finite(host_grad_attn_proj_weight) && all_finite(host_grad_fc_weight) &&
                          all_finite(host_grad_mlp_proj_weight);
        optimizer_finite = all_finite(host_qkv_weight) && all_finite(host_attn_proj_weight) &&
                           all_finite(host_fc_weight) && all_finite(host_mlp_proj_weight);
        passed = forward_finite && backward_finite && optimizer_finite &&
                 residual2_max_abs > 1e-8 && grad_x_max_abs > 1e-8 &&
                 grad_qkv_weight_max_abs > 1e-8 && grad_attn_proj_weight_max_abs > 1e-8 &&
                 grad_fc_weight_max_abs > 1e-8 && grad_mlp_proj_weight_max_abs > 1e-8 &&
                 qkv_weight_max_delta > 1e-8 && attn_proj_weight_max_delta > 1e-8 &&
                 fc_weight_max_delta > 1e-8 && mlp_proj_weight_max_delta > 1e-8;
        if (!passed) {
            std::ostringstream out_message;
            out_message << "transformer-block smoke failed finite/nonzero checks: residual2=" << residual2_max_abs
                        << " grad_x=" << grad_x_max_abs
                        << " qkv_grad=" << grad_qkv_weight_max_abs
                        << " attn_proj_grad=" << grad_attn_proj_weight_max_abs
                        << " fc_grad=" << grad_fc_weight_max_abs
                        << " mlp_proj_grad=" << grad_mlp_proj_weight_max_abs;
            error = out_message.str();
        }
    }

    auto free_device = [&](void* ptr, const std::string& name) {
        if (ptr != nullptr && cuda_free != nullptr) {
            int status = cuda_free(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFree " + name);
            }
        }
    };
    free_device(x, "x");
    free_device(residual_scale, "residual_scale");
    free_device(ln1_weight, "ln1_weight");
    free_device(ln1_bias, "ln1_bias");
    free_device(ln2_weight, "ln2_weight");
    free_device(ln2_bias, "ln2_bias");
    free_device(qkv_weight, "qkv_weight");
    free_device(attn_proj_weight, "attn_proj_weight");
    free_device(fc_weight, "fc_weight");
    free_device(mlp_proj_weight, "mlp_proj_weight");
    free_device(ln1_out, "ln1_out");
    free_device(qkv, "qkv");
    free_device(q, "q");
    free_device(k, "k");
    free_device(v, "v");
    free_device(attn_out, "attn_out");
    free_device(attn_proj, "attn_proj");
    free_device(residual1, "residual1");
    free_device(ln2_out, "ln2_out");
    free_device(fc_out, "fc_out");
    free_device(act, "act");
    free_device(mlp_out, "mlp_out");
    free_device(residual2, "residual2");
    free_device(grad_residual2, "grad_residual2");
    free_device(grad_act, "grad_act");
    free_device(grad_fc_out, "grad_fc_out");
    free_device(grad_ln2, "grad_ln2");
    free_device(grad_residual1_from_mlp, "grad_residual1_from_mlp");
    free_device(grad_residual1, "grad_residual1");
    free_device(grad_attn_out, "grad_attn_out");
    free_device(grad_q, "grad_q");
    free_device(grad_k, "grad_k");
    free_device(grad_v, "grad_v");
    free_device(grad_qkv, "grad_qkv");
    free_device(grad_ln1, "grad_ln1");
    free_device(grad_x_from_attn, "grad_x_from_attn");
    free_device(grad_x, "grad_x");
    free_device(grad_ln1_weight, "grad_ln1_weight");
    free_device(grad_ln1_bias, "grad_ln1_bias");
    free_device(grad_ln2_weight, "grad_ln2_weight");
    free_device(grad_ln2_bias, "grad_ln2_bias");
    free_device(grad_qkv_weight, "grad_qkv_weight");
    free_device(grad_attn_proj_weight, "grad_attn_proj_weight");
    free_device(grad_fc_weight, "grad_fc_weight");
    free_device(grad_mlp_proj_weight, "grad_mlp_proj_weight");
    free_device(ln1_weight_exp_avg, "ln1_weight_exp_avg");
    free_device(ln1_weight_exp_avg_sq, "ln1_weight_exp_avg_sq");
    free_device(ln1_bias_exp_avg, "ln1_bias_exp_avg");
    free_device(ln1_bias_exp_avg_sq, "ln1_bias_exp_avg_sq");
    free_device(ln2_weight_exp_avg, "ln2_weight_exp_avg");
    free_device(ln2_weight_exp_avg_sq, "ln2_weight_exp_avg_sq");
    free_device(ln2_bias_exp_avg, "ln2_bias_exp_avg");
    free_device(ln2_bias_exp_avg_sq, "ln2_bias_exp_avg_sq");
    free_device(qkv_exp_avg, "qkv_exp_avg");
    free_device(qkv_exp_avg_sq, "qkv_exp_avg_sq");
    free_device(attn_proj_exp_avg, "attn_proj_exp_avg");
    free_device(attn_proj_exp_avg_sq, "attn_proj_exp_avg_sq");
    free_device(fc_exp_avg, "fc_exp_avg");
    free_device(fc_exp_avg_sq, "fc_exp_avg_sq");
    free_device(mlp_proj_exp_avg, "mlp_proj_exp_avg");
    free_device(mlp_proj_exp_avg_sq, "mlp_proj_exp_avg_sq");
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"nanogpt\",\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"batch\": " << kBatch << ",\n"
        << "  \"heads\": " << kHeads << ",\n"
        << "  \"seq\": " << kSeq << ",\n"
        << "  \"model_dim\": " << kDim << ",\n"
        << "  \"hidden_dim\": " << kHidden << ",\n"
        << "  \"weight_update_count\": 8,\n"
        << "  \"kernels\": [\n"
        << "    \"nfn_native_tile_layer_norm_float32\",\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_split_qkv_float32\",\n"
        << "    \"nfn_native_tile_scaled_dot_product_attention_float32\",\n"
        << "    \"nfn_native_tile_scaled_residual_add_float32\",\n"
        << "    \"nfn_native_tile_gelu_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_gelu_backward_float32\",\n"
        << "    \"nfn_native_tile_layer_norm_backward_affine_float32\",\n"
        << "    \"nfn_native_tile_layer_norm_backward_input_float32\",\n"
        << "    \"nfn_native_tile_gradient_accumulate_float32\",\n"
        << "    \"nfn_native_tile_scaled_dot_product_attention_backward_float32\",\n"
        << "    \"nfn_native_tile_merge_qkv_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"forward_finite\": " << (forward_finite ? "true" : "false") << ",\n"
        << "  \"backward_finite\": " << (backward_finite ? "true" : "false") << ",\n"
        << "  \"optimizer_finite\": " << (optimizer_finite ? "true" : "false") << ",\n"
        << "  \"residual2_max_abs\": " << residual2_max_abs << ",\n"
        << "  \"grad_x_max_abs\": " << grad_x_max_abs << ",\n"
        << "  \"grad_qkv_weight_max_abs\": " << grad_qkv_weight_max_abs << ",\n"
        << "  \"grad_attn_proj_weight_max_abs\": " << grad_attn_proj_weight_max_abs << ",\n"
        << "  \"grad_fc_weight_max_abs\": " << grad_fc_weight_max_abs << ",\n"
        << "  \"grad_mlp_proj_weight_max_abs\": " << grad_mlp_proj_weight_max_abs << ",\n"
        << "  \"qkv_weight_max_delta\": " << qkv_weight_max_delta << ",\n"
        << "  \"attn_proj_weight_max_delta\": " << attn_proj_weight_max_delta << ",\n"
        << "  \"fc_weight_max_delta\": " << fc_weight_max_delta << ",\n"
        << "  \"mlp_proj_weight_max_delta\": " << mlp_proj_weight_max_delta << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

std::vector<TrainingStage> build_training_stages(const NanoGptPlan& plan) {
    std::vector<TrainingStage> stages;
    const std::int64_t tokens = plan.batch_size * plan.train_seq_len;
    const std::int64_t hidden = tokens * plan.model_dim;
    const std::int64_t mlp_hidden = tokens * plan.model_dim * 4;
    const std::int64_t qkv = tokens * plan.model_dim * 3;
    const std::int64_t attention = tokens * plan.model_dim;
    const std::int64_t logits = tokens * plan.vocab_size;
    const std::int64_t parameters = parameter_layout_count(build_parameter_layout(plan));

    auto add = [&](std::string name, std::string phase, std::string status, std::string kernel_abi, std::int64_t elements) {
        TrainingStage stage;
        stage.name = std::move(name);
        stage.phase = std::move(phase);
        stage.status = std::move(status);
        stage.kernel_abi = std::move(kernel_abi);
        stage.elements = elements;
        stages.push_back(std::move(stage));
    };

    add("token_embedding.forward", "forward", "ready", "nfn_native_tile_token_embedding_float32", hidden);
    add("absolute_position_embedding.forward", "forward", "ready", "nfn_native_tile_absolute_position_embedding_float32", hidden);
    add("embedding_residual_add.forward", "forward", "ready", "nfn_native_tile_scaled_residual_add_float32", hidden);
    for (std::int64_t layer = 0; layer < plan.num_layers; ++layer) {
        const std::string prefix = "blocks." + std::to_string(layer) + ".";
        add(prefix + "ln1.forward", "forward", "ready", "nfn_native_tile_layer_norm_float32", hidden);
        add(prefix + "attn.qkv.forward", "forward", "ready", "nfn_native_tile_linear_float32", qkv);
        add(prefix + "attn.qkv.split", "forward", "ready", "nfn_native_tile_split_qkv_float32", qkv);
        add(prefix + "attn.sdpa.forward", "forward", "ready", "nfn_native_tile_scaled_dot_product_attention_float32", attention);
        add(prefix + "attn.out.forward", "forward", "ready", "nfn_native_tile_linear_float32", hidden);
        add(prefix + "attn.residual_add.forward", "forward", "ready", "nfn_native_tile_scaled_residual_add_float32", hidden);
        add(prefix + "ln2.forward", "forward", "ready", "nfn_native_tile_layer_norm_float32", hidden);
        add(prefix + "mlp.fc.forward", "forward", "ready", "nfn_native_tile_linear_float32", mlp_hidden);
        add(prefix + "mlp.gelu.forward", "forward", "ready", "nfn_native_tile_gelu_float32", mlp_hidden);
        add(prefix + "mlp.proj.forward", "forward", "ready", "nfn_native_tile_linear_float32", hidden);
        add(prefix + "mlp.residual_add.forward", "forward", "ready", "nfn_native_tile_scaled_residual_add_float32", hidden);
        if (plan.dropout_p > 0.0) {
            add(prefix + "dropout.forward", "forward", "ready", "nfn_native_tile_dropout_forward_float32", hidden);
            add(prefix + "dropout.backward", "backward", "ready", "nfn_native_tile_dropout_backward_float32", hidden);
        }
    }
    add("ln_f.forward", "forward", "ready", "nfn_native_tile_layer_norm_float32", hidden);
    add("lm_head.forward", "forward", "ready", "nfn_native_tile_linear_float32", logits);
    add("token_cross_entropy.forward", "forward", "ready", "nfn_native_tile_token_cross_entropy_partials_float32", tokens);
    add("token_cross_entropy.backward", "backward", "ready", "nfn_native_tile_token_cross_entropy_backward_float32", logits);
    add("lm_head.backward_input", "backward", "ready", "nfn_native_tile_linear_backward_input_float32", hidden);
    add("lm_head.backward_weight_tied", "backward", "ready", "nfn_native_tile_linear_backward_weight_float32", plan.vocab_size * plan.model_dim);
    for (std::int64_t layer = plan.num_layers - 1; layer >= 0; --layer) {
        const std::string prefix = "blocks." + std::to_string(layer) + ".";
        add(prefix + "mlp.proj.backward", "backward", "ready", "linear input/weight/bias backward native ABI", hidden + mlp_hidden);
        add(prefix + "mlp.gelu.backward", "backward", "ready", "nfn_native_tile_gelu_backward_float32", mlp_hidden);
        add(prefix + "mlp.fc.backward", "backward", "ready", "linear input/weight/bias backward native ABI", hidden + mlp_hidden);
        add(prefix + "ln2.backward", "backward", "ready", "nfn_native_tile_layer_norm_backward_input_float32", hidden);
        add(prefix + "attn.out.backward", "backward", "ready", "linear input/weight/bias backward native ABI", hidden);
        add(prefix + "attn.sdpa.backward", "backward", "ready", "nfn_native_tile_scaled_dot_product_attention_backward_float32", attention);
        add(prefix + "attn.qkv.grad_merge", "backward", "ready", "nfn_native_tile_merge_qkv_float32", qkv);
        add(prefix + "attn.qkv.backward", "backward", "ready", "linear input/weight/bias backward native ABI", hidden + qkv);
        add(prefix + "ln1.backward", "backward", "ready", "nfn_native_tile_layer_norm_backward_input_float32", hidden);
    }
    add("embedding.backward", "backward", "ready", "token+position embedding backward native ABI", hidden);
    add("gradient_zero", "optimizer", "ready", "nfn_native_tile_fill_float32", parameters);
    add("gradient_clip", "optimizer", "ready", "nfn_native_tile_global_norm_clip_scale_float32", 1);
    add("gradient_scale", "optimizer", "ready", "nfn_native_tile_scale_inplace_by_device_float32", parameters);
    add("adamw_step", "optimizer", "ready", "nfn_native_tile_adamw_step_float32", parameters);
    return stages;
}

std::string training_stages_json(const std::vector<TrainingStage>& stages) {
    std::int64_t ready = 0;
    std::int64_t requires_wiring = 0;
    std::int64_t missing_abi = 0;
    std::int64_t max_elements = 0;
    for (const TrainingStage& stage : stages) {
        if (stage.status == "ready") {
            ++ready;
        } else if (stage.status == "requires_wiring") {
            ++requires_wiring;
        } else if (stage.status == "missing_abi") {
            ++missing_abi;
        }
        if (stage.elements > max_elements) {
            max_elements = stage.elements;
        }
    }
    std::ostringstream out;
    out << "{\n"
        << "    \"stage_count\": " << stages.size() << ",\n"
        << "    \"ready_stage_count\": " << ready << ",\n"
        << "    \"requires_wiring_stage_count\": " << requires_wiring << ",\n"
        << "    \"missing_abi_stage_count\": " << missing_abi << ",\n"
        << "    \"max_stage_elements\": " << max_elements << ",\n"
        << "    \"stages\": [\n";
    for (std::size_t i = 0; i < stages.size(); ++i) {
        const TrainingStage& stage = stages[i];
        out << "      {\"name\": \"" << json_escape(stage.name) << "\", "
            << "\"phase\": \"" << json_escape(stage.phase) << "\", "
            << "\"status\": \"" << json_escape(stage.status) << "\", "
            << "\"kernel_abi\": \"" << json_escape(stage.kernel_abi) << "\", "
            << "\"elements\": " << stage.elements << "}";
        if (i + 1 != stages.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "    ]\n"
        << "  }";
    return out.str();
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [native nanogpt options]\n\n"
        << "Compiled NeuralFn NanoGPT native trainer.\n"
        << "Compiled NeuralFn NanoGPT native training preflight.\n"
        << "This binary parses the NanoGPT training contract in C++, emits native plans,\n"
        << "and runs the partial tied token-LM loop with raw CUDA Tile kernels.\n\n"
        << "Core options:\n"
        << "  --dataset-alias PATH_OR_ALIAS   Dataset alias or cached shard directory\n"
        << "  --tinystories                   Use TinyStoriesV2 GPT-4 alias\n"
        << "  --output PATH                   Native checkpoint/output path\n"
        << "  --max-steps N                   Optimizer steps, default 20000\n"
        << "  --train-seq-len N               Sequence length, default 1024\n"
        << "  --batch-size N                  Microbatch rows, default 64\n"
        << "  --train-batch-tokens N          Effective tokens/step, default 524288\n"
        << "  --eval-every-steps N            Validation-loss cadence over val shards, default 250\n"
        << "  --eval-batches N                Validation batches per eval, default 10\n"
        << "  --eval-batch-size N             Validation microbatch rows, default 8\n"
        << "  --vocab-size N                  Vocabulary size, default 50257\n"
        << "  --num-layers N                  Transformer layers, default 5\n"
        << "  --model-dim N                   Width, default 320\n"
        << "  --num-heads N                   Attention heads, default 5\n"
        << "  --optimizer-profile adamw       Native optimizer profile; only adamw is accepted\n"
        << "  --require-token-shards          Require fineweb_train_*.bin and fineweb_val_*.bin shards\n"
        << "  --sample-token-batch            Include the first native token/target batch in --print-plan JSON\n"
        << "  --allow-train-val-fallback      Reuse train shard when no validation shard exists\n"
        << "  --tile-ops-lib PATH             Raw libnfn_native_train_tile_ops.so path for Tile ops checks/smokes\n"
        << "  --cuda-runtime-lib PATH         CUDA runtime path for Tile ops smokes; env NFN_CUDA_RUNTIME_LIB also works\n"
        << "  --check-tile-ops                dlopen the raw Tile ops C ABI, verify NanoGPT-required symbols, and exit\n"
        << "  --smoke-tile-ops                Execute a tiny raw Tile fill kernel through CUDA runtime dlopen and exit\n"
        << "  --smoke-optimizer-step          Execute one raw Tile AdamW update over tiny device buffers and exit\n"
        << "  --smoke-training-loop-step      Execute gradient zero/clip/scale/AdamW loop kernels and exit\n"
        << "  --smoke-lm-step                 Execute tiny tied-embedding LM forward/backward/update kernels and exit\n"
        << "  --smoke-token-train-step        Execute one native sampled-token tied-LM train step and exit\n"
        << "  --smoke-embedding-norm-step     Execute sampled embedding/residual/LayerNorm train-step kernels and exit\n"
        << "  --smoke-qkv-layout-step         Execute fused-QKV split/merge layout kernels and exit\n"
        << "  --smoke-fused-qkv-attention-step Execute fused-QKV attention backward/update kernels and exit\n"
        << "  --smoke-transformer-block-step  Execute tiny transformer block forward/backward/update kernels and exit\n"
        << "  --smoke-mlp-step                Execute tiny MLP projection/GELU backward/update kernels and exit\n"
        << "  --smoke-attention-step          Execute tiny attention projection/SDPA backward/update kernels and exit\n"
        << "  --train-token-lm                Train a tied token-embedding LM over cached shards using raw Tile kernels\n"
        << "  --print-plan                    Print the native JSON plan and exit 0\n"
        << "  --dry-run                       Print the native JSON plan and exit 0 before loading Tile ops\n";
}

void validate_plan(const NanoGptPlan& plan) {
    if (plan.optimizer_profile != "adamw") {
        std::cerr << "--optimizer-profile must be adamw for the native NanoGPT trainer, got '"
                  << plan.optimizer_profile << "'\n";
        std::exit(2);
    }
    if (plan.train_seq_len <= 0 || plan.batch_size <= 0 || plan.train_batch_tokens <= 0) {
        std::cerr << "train sequence length, batch size, and train batch tokens must be positive\n";
        std::exit(2);
    }
    if (plan.max_steps <= 0 || plan.warmup_steps < 0) {
        std::cerr << "max steps must be positive and warmup steps must be non-negative\n";
        std::exit(2);
    }
    if (plan.vocab_size <= 0 || plan.vocab_size > 65535) {
        std::cerr << "native NanoGPT currently requires 1 <= --vocab-size <= 65535 for uint16 token shards\n";
        std::exit(2);
    }
    if (plan.num_layers <= 0 || plan.model_dim <= 0 || plan.num_heads <= 0) {
        std::cerr << "model depth, width, and heads must be positive\n";
        std::exit(2);
    }
    if (plan.model_dim % plan.num_heads != 0) {
        std::cerr << "--model-dim must be divisible by --num-heads for native NanoGPT\n";
        std::exit(2);
    }
    if (plan.dropout_p < 0.0 || plan.dropout_p >= 1.0) {
        std::cerr << "--dropout-p must be in [0, 1)\n";
        std::exit(2);
    }
}

void print_plan_json(
    const NanoGptPlan& plan,
    const neuralfn::native_train::TokenShardDataset* dataset,
    const neuralfn::native_train::BatchPlan* dataset_batch_plan,
    const neuralfn::native_train::TokenBatch* sample_batch) {
    const std::int64_t microbatch_tokens = plan.batch_size * plan.train_seq_len;
    const std::int64_t grad_accum_steps = ceil_div(plan.train_batch_tokens, microbatch_tokens);
    const std::int64_t effective_tokens = grad_accum_steps * microbatch_tokens;
    const std::int64_t head_dim = plan.model_dim / plan.num_heads;
    const std::vector<ParameterBuffer> parameter_layout = build_parameter_layout(plan);
    const std::vector<TrainingStage> training_stages = build_training_stages(plan);
    const std::int64_t parameter_count = parameter_layout_count(parameter_layout);
    std::cout
        << "{\n"
        << "  \"model_family\": \"nanogpt\",\n"
        << "  \"status\": \"native-preflight-missing-trainer\",\n"
        << "  \"dataset_alias\": \"" << json_escape(plan.dataset_alias) << "\",\n"
        << "  \"output\": \"" << json_escape(plan.output) << "\",\n"
        << "  \"shape\": {\n"
        << "    \"vocab_size\": " << plan.vocab_size << ",\n"
        << "    \"num_layers\": " << plan.num_layers << ",\n"
        << "    \"model_dim\": " << plan.model_dim << ",\n"
        << "    \"num_heads\": " << plan.num_heads << ",\n"
        << "    \"head_dim\": " << head_dim << ",\n"
        << "    \"bias\": " << (plan.bias ? "true" : "false") << ",\n"
        << "    \"dropout_p\": " << plan.dropout_p << "\n"
        << "  },\n"
        << "  \"schedule\": {\n"
        << "    \"max_steps\": " << plan.max_steps << ",\n"
        << "    \"train_seq_len\": " << plan.train_seq_len << ",\n"
        << "    \"batch_size\": " << plan.batch_size << ",\n"
        << "    \"microbatch_tokens\": " << microbatch_tokens << ",\n"
        << "    \"requested_train_batch_tokens\": " << plan.train_batch_tokens << ",\n"
        << "    \"grad_accum_steps\": " << grad_accum_steps << ",\n"
        << "    \"effective_train_batch_tokens\": " << effective_tokens << ",\n"
        << "    \"eval_every_steps\": " << plan.eval_every_steps << ",\n"
        << "    \"eval_batches\": " << plan.eval_batches << ",\n"
        << "    \"eval_batch_size\": " << plan.eval_batch_size << ",\n"
        << "    \"warmup_steps\": " << plan.warmup_steps << "\n"
        << "  },\n"
        << "  \"optimizer\": {\n"
        << "    \"profile\": \"" << json_escape(plan.optimizer_profile) << "\",\n"
        << "    \"learning_rate\": " << plan.learning_rate << ",\n"
        << "    \"weight_decay\": " << plan.weight_decay << ",\n"
        << "    \"beta1\": " << plan.beta1 << ",\n"
        << "    \"beta2\": " << plan.beta2 << ",\n"
        << "    \"adam_eps\": " << plan.adam_eps << ",\n"
        << "    \"grad_clip_norm\": " << plan.grad_clip_norm << ",\n"
        << "    \"parameter_groups\": " << optimizer_groups_json(plan, parameter_layout) << "\n"
        << "  },\n"
        << "  \"token_shards\": ";
    if (dataset != nullptr && dataset_batch_plan != nullptr) {
        std::cout << neuralfn::native_train::token_shard_dataset_json(*dataset, dataset_batch_plan);
    } else {
        std::cout << "null";
    }
    std::cout << ",\n"
        << "  \"sample_batch\": ";
    if (sample_batch != nullptr) {
        std::cout << neuralfn::native_train::token_batch_json(*sample_batch);
    } else {
        std::cout << "null";
    }
    std::cout << ",\n"
        << "  \"parameter_layout\": " << parameter_layout_json(parameter_layout) << ",\n"
        << "  \"training_step_plan\": " << training_stages_json(training_stages) << ",\n"
        << "  \"estimated_parameters\": " << parameter_count << ",\n"
        << "  \"available_native_kernels\": [\n"
        << "    \"native uint16 cached-token batch sampler over resolved train/validation shards\",\n"
        << "    \"trainer-wide parameter/gradient buffer registry\",\n"
        << "    \"AdamW parameter-group planner over registered buffers\",\n"
        << "    \"token embedding forward\",\n"
        << "    \"token embedding weight backward\",\n"
        << "    \"absolute position embedding forward\",\n"
        << "    \"absolute position embedding backward\",\n"
        << "    \"LayerNorm forward\",\n"
        << "    \"LayerNorm input backward\",\n"
        << "    \"LayerNorm affine parameter backward\",\n"
        << "    \"RMSNorm forward\",\n"
        << "    \"RMSNorm input backward\",\n"
        << "    \"linear forward\",\n"
        << "    \"linear input backward\",\n"
        << "    \"linear weight backward\",\n"
        << "    \"linear bias backward\",\n"
        << "    \"tied LM head input and weight backward via linear native ABI\",\n"
        << "    \"fused QKV projection split and gradient merge for NanoGPT qkv.weight layout\",\n"
        << "    \"scaled residual add forward\",\n"
        << "    \"GELU activation forward\",\n"
        << "    \"GELU activation backward\",\n"
        << "    \"dropout forward/backward native Tile ABI for nonzero dropout_p\",\n"
        << "    \"MLP projection/GELU forward/backward/update smoke over raw native kernels\",\n"
        << "    \"softmax forward\",\n"
        << "    \"scaled dot-product attention forward\",\n"
        << "    \"scaled dot-product attention backward\",\n"
        << "    \"fused QKV attention forward/backward/update smoke over raw native kernels\",\n"
        << "    \"transformer block forward/backward/update smoke over raw native kernels\",\n"
        << "    \"multi-step tied token-LM trainer loop over cached native token shards\",\n"
        << "    \"periodic native validation loss over resolved validation token shards\",\n"
        << "    \"token and masked token cross entropy loss partials\",\n"
        << "    \"chunked row-wise token and masked token cross entropy logits backward for full GPT-class vocabularies\",\n"
        << "    \"gradient accumulation, partial reductions, scaling, and fused AdamW update\",\n"
        << "    \"global norm clipping scale finalizer and device-scalar gradient scaling\",\n"
        << "    \"registered-buffer AdamW iteration over decay and no-decay parameter groups\"\n"
        << "  ],\n"
        << "  \"required_native_kernels\": [\n"
        << "    \"full trainer loop integration over ready forward, backward, and optimizer stages\"\n"
        << "  ],\n"
        << "  \"unparsed_args\": [";
    for (std::size_t i = 0; i < plan.unparsed_args.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << "\"" << json_escape(plan.unparsed_args[i]) << "\"";
    }
    std::cout << "]\n}\n";
}

NanoGptPlan parse_args(int argc, char** argv, bool* print_plan, bool* dry_run) {
    NanoGptPlan plan;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto after_equals = [&](std::string_view prefix) {
            return arg.substr(prefix.size());
        };
        auto value_for = [&](const std::string& flag) {
            return require_value(argc, argv, &i, flag);
        };
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--print-plan" || arg == "--json") {
            *print_plan = true;
            continue;
        }
        if (arg == "--dry-run" || arg == "--native-cuda-dry-run") {
            *dry_run = true;
            continue;
        }
        if (arg == "--check-tile-ops") {
            plan.check_tile_ops = true;
            continue;
        }
        if (arg == "--smoke-tile-ops") {
            plan.smoke_tile_ops = true;
            continue;
        }
        if (arg == "--smoke-optimizer-step") {
            plan.smoke_optimizer_step = true;
            continue;
        }
        if (arg == "--smoke-training-loop-step") {
            plan.smoke_training_loop_step = true;
            continue;
        }
        if (arg == "--smoke-lm-step") {
            plan.smoke_lm_step = true;
            continue;
        }
        if (arg == "--smoke-token-train-step") {
            plan.require_token_shards = true;
            plan.smoke_token_train_step = true;
            continue;
        }
        if (arg == "--smoke-embedding-norm-step") {
            plan.require_token_shards = true;
            plan.smoke_embedding_norm_step = true;
            continue;
        }
        if (arg == "--smoke-qkv-layout-step") {
            plan.smoke_qkv_layout_step = true;
            continue;
        }
        if (arg == "--smoke-fused-qkv-attention-step") {
            plan.smoke_fused_qkv_attention_step = true;
            continue;
        }
        if (arg == "--smoke-transformer-block-step") {
            plan.smoke_transformer_block_step = true;
            continue;
        }
        if (arg == "--smoke-mlp-step") {
            plan.smoke_mlp_step = true;
            continue;
        }
        if (arg == "--smoke-attention-step") {
            plan.smoke_attention_step = true;
            continue;
        }
        if (arg == "--train-token-lm") {
            plan.require_token_shards = true;
            plan.train_token_lm = true;
            continue;
        }
        if (arg == "--tile-ops-lib") {
            plan.tile_ops_lib = value_for(arg);
            continue;
        }
        if (arg.rfind("--tile-ops-lib=", 0) == 0) {
            plan.tile_ops_lib = after_equals("--tile-ops-lib=");
            continue;
        }
        if (arg == "--cuda-runtime-lib") {
            plan.cuda_runtime_lib = value_for(arg);
            continue;
        }
        if (arg.rfind("--cuda-runtime-lib=", 0) == 0) {
            plan.cuda_runtime_lib = after_equals("--cuda-runtime-lib=");
            continue;
        }
        if (arg == "--allow-train-val-fallback" || arg == "--native-cuda-allow-train-val-fallback") {
            plan.allow_train_as_val = true;
            continue;
        }
        if (arg == "--require-token-shards") {
            plan.require_token_shards = true;
            continue;
        }
        if (arg == "--sample-token-batch") {
            plan.require_token_shards = true;
            plan.sample_token_batch = true;
            continue;
        }
        if (arg == "--tinystories") {
            plan.dataset_alias = "roneneldan__TinyStories__TinyStoriesV2-GPT4";
            continue;
        }
        if (arg == "--dataset-alias" || arg == "--dataset") {
            plan.dataset_alias = value_for(arg);
            continue;
        }
        if (arg.rfind("--dataset-alias=", 0) == 0) {
            plan.dataset_alias = after_equals("--dataset-alias=");
            continue;
        }
        if (arg.rfind("--dataset=", 0) == 0) {
            plan.dataset_alias = after_equals("--dataset=");
            continue;
        }
        if (arg == "--output") {
            plan.output = value_for(arg);
            continue;
        }
        if (arg.rfind("--output=", 0) == 0) {
            plan.output = after_equals("--output=");
            continue;
        }
        if (arg == "--max-steps" || arg == "--iterations") {
            plan.max_steps = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg.rfind("--max-steps=", 0) == 0) {
            plan.max_steps = parse_i64(after_equals("--max-steps="), "--max-steps");
            continue;
        }
        if (arg == "--train-seq-len") {
            plan.train_seq_len = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--batch-size") {
            plan.batch_size = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--train-batch-tokens") {
            plan.train_batch_tokens = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--eval-batches") {
            plan.eval_batches = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--eval-batch-size") {
            plan.eval_batch_size = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--eval-every-steps") {
            plan.eval_every_steps = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--warmup-steps") {
            plan.warmup_steps = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--vocab-size") {
            plan.vocab_size = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--num-layers") {
            plan.num_layers = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--model-dim") {
            plan.model_dim = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--num-heads") {
            plan.num_heads = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--bias") {
            plan.bias = true;
            continue;
        }
        if (arg == "--no-bias") {
            plan.bias = false;
            continue;
        }
        if (arg == "--dropout-p") {
            plan.dropout_p = parse_f64(value_for(arg), arg);
            continue;
        }
        if (arg == "--optimizer-profile") {
            plan.optimizer_profile = value_for(arg);
            continue;
        }
        if (arg == "--learning-rate") {
            plan.learning_rate = parse_f64(value_for(arg), arg);
            continue;
        }
        if (arg == "--weight-decay") {
            plan.weight_decay = parse_f64(value_for(arg), arg);
            continue;
        }
        if (arg == "--beta1") {
            plan.beta1 = parse_f64(value_for(arg), arg);
            continue;
        }
        if (arg == "--beta2") {
            plan.beta2 = parse_f64(value_for(arg), arg);
            continue;
        }
        if (arg == "--adam-eps") {
            plan.adam_eps = parse_f64(value_for(arg), arg);
            continue;
        }
        if (arg == "--grad-clip-norm") {
            plan.grad_clip_norm = parse_f64(value_for(arg), arg);
            continue;
        }
        plan.unparsed_args.push_back(arg);
    }
    return plan;
}

}  // namespace

int main(int argc, char** argv) {
    bool print_plan = false;
    bool dry_run = false;
    NanoGptPlan plan = parse_args(argc, argv, &print_plan, &dry_run);
    validate_plan(plan);
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
        setenv("CUDA_MODULE_LOADING", "LAZY", 0);
    }
    if (plan.smoke_tile_ops) {
        return print_tile_ops_smoke_json(plan, argv[0]);
    }
    if (plan.smoke_optimizer_step) {
        return print_optimizer_smoke_json(plan, argv[0]);
    }
    if (plan.smoke_training_loop_step) {
        return print_training_loop_step_smoke_json(plan, argv[0]);
    }
    if (plan.smoke_lm_step) {
        return print_lm_step_smoke_json(plan, argv[0]);
    }
    if (plan.smoke_token_train_step) {
        return print_token_train_step_smoke_json(plan, argv[0]);
    }
    if (plan.smoke_embedding_norm_step) {
        return print_embedding_norm_step_smoke_json(plan, argv[0]);
    }
    if (plan.smoke_qkv_layout_step) {
        return print_qkv_layout_step_smoke_json(plan, argv[0]);
    }
    if (plan.smoke_fused_qkv_attention_step) {
        return print_fused_qkv_attention_step_smoke_json(plan, argv[0]);
    }
    if (plan.smoke_transformer_block_step) {
        return print_transformer_block_step_smoke_json(plan, argv[0]);
    }
    if (plan.smoke_mlp_step) {
        return print_mlp_step_smoke_json(plan, argv[0]);
    }
    if (plan.smoke_attention_step) {
        return print_attention_step_smoke_json(plan, argv[0]);
    }
    neuralfn::native_train::TokenShardDataset token_shards;
    neuralfn::native_train::BatchPlan token_batch_plan;
    neuralfn::native_train::TokenBatch sample_batch;
    bool have_token_shards = false;
    bool have_sample_batch = false;
    if (plan.require_token_shards || std::filesystem::is_directory(neuralfn::native_train::resolve_dataset_path(plan.dataset_alias))) {
        try {
            token_shards = neuralfn::native_train::resolve_token_shards(plan.dataset_alias, plan.allow_train_as_val);
            token_batch_plan = neuralfn::native_train::build_batch_plan(
                token_shards, plan.train_seq_len, plan.batch_size, plan.train_batch_tokens);
            have_token_shards = true;
            if (plan.sample_token_batch) {
                neuralfn::native_train::SequentialTokenBatchSampler sampler(
                    token_shards.train_shards, plan.train_seq_len, plan.batch_size);
                have_sample_batch = sampler.next(sample_batch);
                if (!have_sample_batch) {
                    std::cerr << "not enough train tokens to build one native token batch\n";
                    return 2;
                }
            }
        } catch (const std::exception& exc) {
            if (plan.require_token_shards) {
                std::cerr << exc.what() << "\n";
                return 2;
            }
        }
    }
    if (print_plan || dry_run) {
        print_plan_json(
            plan,
            have_token_shards ? &token_shards : nullptr,
            have_token_shards ? &token_batch_plan : nullptr,
            have_sample_batch ? &sample_batch : nullptr);
    }
    if (print_plan && !dry_run) {
        return 0;
    }
    if (dry_run && plan.train_token_lm) {
        return 0;
    }
    if (plan.train_token_lm) {
        return run_token_lm_training_json(plan, argv[0]);
    }
    if (plan.check_tile_ops) {
        return print_tile_ops_check_json(plan, argv[0]);
    }
    std::cerr
        << "nfn_nanogpt_native_train: native CUDA Tile trainer for nanogpt is not implemented yet.\n"
        << "The C++ preflight parsed the NanoGPT plan; implement the required kernels printed by --print-plan before production training.\n";
    return 2;
}
