#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <exception>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "token_shards.h"

#ifndef NFN_NATIVE_MODEL_FAMILY
#define NFN_NATIVE_MODEL_FAMILY "unknown"
#endif

#ifndef NFN_NATIVE_TARGET_NAME
#define NFN_NATIVE_TARGET_NAME "nfn_unknown_native_train"
#endif

#ifndef NFN_NATIVE_REQUIRED_KERNELS
#define NFN_NATIVE_REQUIRED_KERNELS "model-specific CUDA Tile kernels"
#endif

#ifndef NFN_NATIVE_REQUIRED_SYMBOLS
#define NFN_NATIVE_REQUIRED_SYMBOLS ""
#endif

#ifndef NFN_NATIVE_COVERAGE_CLASS
#define NFN_NATIVE_COVERAGE_CLASS "family-native-loop-missing"
#endif

#ifndef NFN_NATIVE_MISSING_REQUIREMENTS
#define NFN_NATIVE_MISSING_REQUIREMENTS ""
#endif

namespace {

struct Config {
    std::string dataset_alias = "roneneldan__TinyStories__TinyStoriesV2-GPT4";
    std::string output_dir = "artifacts";
    std::string template_name = NFN_NATIVE_MODEL_FAMILY;
    std::string graph_file;
    std::string tile_ops_lib;
    std::int64_t max_steps = 20000;
    std::int64_t batch_size = 64;
    std::int64_t train_seq_len = 1024;
    std::int64_t train_batch_tokens = 524288;
    std::int64_t eval_every_steps = 250;
    double learning_rate = 0.0006;
    bool print_plan = false;
    bool check_tile_ops = false;
    bool dry_run = false;
    bool sample_token_batch = false;
    bool allow_train_as_val = false;
    bool smoke_llama_loop = false;
    bool smoke_llama_train_step = false;
    std::vector<std::string> unparsed_args;
    std::string cuda_runtime_lib;
};

struct SymbolResult {
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

std::vector<std::string> split_csv(std::string_view value) {
    std::vector<std::string> out;
    std::string current;
    for (char ch : value) {
        if (ch == ',') {
            if (!current.empty()) {
                out.push_back(current);
            }
            current.clear();
        } else if (!std::isspace(static_cast<unsigned char>(ch))) {
            current.push_back(ch);
        }
    }
    if (!current.empty()) {
        out.push_back(current);
    }
    return out;
}

template <typename Fn>
Fn load_symbol(void* handle, const char* name) {
    return reinterpret_cast<Fn>(dlsym(handle, name));
}

std::vector<std::string> cuda_runtime_candidates(const Config& cfg) {
    if (!cfg.cuda_runtime_lib.empty()) {
        return {cfg.cuda_runtime_lib};
    }
    const char* env = std::getenv("NFN_CUDA_RUNTIME_LIB");
    if (env != nullptr && env[0] != '\0') {
        return {env};
    }
    return {
        "/usr/local/cuda/lib64/libcudart.so.13",
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/local/cuda-13/lib64/libcudart.so.13",
        "/usr/local/cuda-13/lib64/libcudart.so",
        "libcudart.so.13",
        "libcudart.so",
        "libcudart.so.12",
    };
}

std::string resolve_tile_ops_lib(const Config& cfg, const char* program) {
    if (!cfg.tile_ops_lib.empty()) {
        return cfg.tile_ops_lib;
    }
    const char* env = std::getenv("NFN_NATIVE_TRAIN_TILE_OPS_LIB");
    if (env != nullptr && env[0] != '\0') {
        return env;
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

std::vector<SymbolResult> check_symbols(const std::string& lib_path, std::string* error) {
    std::vector<SymbolResult> results;
    const std::vector<std::string> symbols = split_csv(NFN_NATIVE_REQUIRED_SYMBOLS);
    if (symbols.empty()) {
        return results;
    }
    void* handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        if (error != nullptr) {
            const char* raw = dlerror();
            *error = raw == nullptr ? "dlopen failed" : raw;
        }
        for (const std::string& symbol : symbols) {
            results.push_back({symbol, false});
        }
        return results;
    }
    for (const std::string& symbol : symbols) {
        dlerror();
        void* found = dlsym(handle, symbol.c_str());
        results.push_back({symbol, found != nullptr});
    }
    dlclose(handle);
    return results;
}

bool all_symbols_found(const std::vector<SymbolResult>& results) {
    return std::all_of(results.begin(), results.end(), [](const SymbolResult& result) {
        return result.found;
    });
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [native options]\n\n"
        << NFN_NATIVE_TARGET_NAME << " is a compiled NeuralFn native preflight for "
        << NFN_NATIVE_MODEL_FAMILY << ".\n"
        << "It intentionally fails for real training until the required CUDA Tile C++ trainer is implemented.\n\n"
        << "Useful preflight options:\n"
        << "  --print-plan              Emit JSON with native work and schedule metadata\n"
        << "  --check-tile-ops          Check required raw Tile symbols from the trainer ABI\n"
        << "  --sample-token-batch      Resolve native token shards and emit the first token/target batch\n"
        << "  --smoke-llama-loop        Launch RMSNorm, RoPE, and SwiGLU loop-composition kernels on CUDA\n"
        << "  --smoke-llama-train-step  Launch the LLaMA loop-composition kernels plus one AdamW update on CUDA\n"
        << "  --tile-ops-lib PATH       Override libnfn_native_train_tile_ops.so\n"
        << "  --cuda-runtime-lib PATH   Override libcudart for CUDA smoke commands\n"
        << "  --dry-run                 Emit the same JSON plan without training\n\n"
        << "Required native work:\n"
        << "  " << NFN_NATIVE_REQUIRED_KERNELS << "\n\n"
        << "This command keeps CLI, SDK, and install paths on compiled native boundaries\n"
        << "instead of entering graph-backed TorchTrainer code.\n";
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--print-plan" || arg == "--native-cuda-print-plan") {
            cfg.print_plan = true;
        } else if (arg == "--check-tile-ops" || arg == "--native-cuda-check-tile-ops") {
            cfg.check_tile_ops = true;
        } else if (arg == "--dry-run" || arg == "--native-cuda-dry-run") {
            cfg.dry_run = true;
        } else if (arg == "--sample-token-batch") {
            cfg.sample_token_batch = true;
        } else if (arg == "--smoke-llama-loop" || arg == "--native-cuda-smoke-llama-loop") {
            cfg.smoke_llama_loop = true;
        } else if (arg == "--smoke-llama-train-step" || arg == "--native-cuda-smoke-llama-train-step") {
            cfg.smoke_llama_train_step = true;
        } else if (arg == "--allow-train-val-fallback" || arg == "--native-cuda-allow-train-val-fallback") {
            cfg.allow_train_as_val = true;
        } else if (arg == "--dataset-alias" || arg == "--dataset") {
            cfg.dataset_alias = require_value(argc, argv, &i, arg);
        } else if (arg == "--tinystories") {
            cfg.dataset_alias = "roneneldan__TinyStories__TinyStoriesV2-GPT4";
        } else if (arg == "--output-dir" || arg == "--output") {
            cfg.output_dir = require_value(argc, argv, &i, arg);
        } else if (arg == "--template-name" || arg == "--template" || arg == "--preset") {
            cfg.template_name = require_value(argc, argv, &i, arg);
        } else if (arg == "--graph-file" || arg == "--graph") {
            cfg.graph_file = require_value(argc, argv, &i, arg);
        } else if (arg == "--tile-ops-lib" || arg == "--native-cuda-tile-ops-lib") {
            cfg.tile_ops_lib = require_value(argc, argv, &i, arg);
        } else if (arg == "--cuda-runtime-lib" || arg == "--native-cuda-cuda-runtime-lib") {
            cfg.cuda_runtime_lib = require_value(argc, argv, &i, arg);
        } else if (arg == "--max-steps") {
            cfg.max_steps = parse_i64(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--batch-size") {
            cfg.batch_size = parse_i64(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--train-seq-len" || arg == "--seq-len") {
            cfg.train_seq_len = parse_i64(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--train-batch-tokens") {
            cfg.train_batch_tokens = parse_i64(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--eval-every-steps") {
            cfg.eval_every_steps = parse_i64(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--learning-rate" || arg == "--lr") {
            cfg.learning_rate = parse_f64(require_value(argc, argv, &i, arg), arg);
        } else {
            cfg.unparsed_args.push_back(arg);
        }
    }
    return cfg;
}

void print_json(const Config& cfg, const char* program) {
    const std::string tile_ops_lib = resolve_tile_ops_lib(cfg, program);
    std::string tile_ops_error;
    const std::vector<std::string> required_symbols = split_csv(NFN_NATIVE_REQUIRED_SYMBOLS);
    const bool symbol_check_requested = cfg.check_tile_ops || cfg.print_plan || cfg.dry_run;
    const std::vector<SymbolResult> symbols = cfg.check_tile_ops
        ? check_symbols(tile_ops_lib, &tile_ops_error)
        : std::vector<SymbolResult>{};
    const bool symbols_ok = cfg.check_tile_ops && all_symbols_found(symbols);
    bool have_dataset = false;
    bool have_batch = false;
    neuralfn::native_train::TokenShardDataset dataset;
    neuralfn::native_train::BatchPlan batch_plan;
    neuralfn::native_train::TokenBatch sample_batch;
    if (cfg.sample_token_batch) {
        dataset = neuralfn::native_train::resolve_token_shards(
            cfg.dataset_alias,
            cfg.allow_train_as_val,
            false);
        batch_plan = neuralfn::native_train::build_batch_plan(
            dataset,
            cfg.train_seq_len,
            cfg.batch_size,
            cfg.train_batch_tokens);
        have_dataset = true;
        neuralfn::native_train::SequentialTokenBatchSampler sampler(
            dataset.train_shards,
            cfg.train_seq_len,
            cfg.batch_size);
        have_batch = sampler.next(sample_batch);
    }
    const std::string kernel_status =
        required_symbols.empty()
            ? "no-required-tile-symbols-declared"
            : (!cfg.check_tile_ops
                   ? "required-tile-symbols-unchecked"
                   : (symbols_ok ? "required-tile-symbols-present" : "required-tile-symbols-missing"));

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(NFN_NATIVE_MODEL_FAMILY) << "\",\n"
        << "  \"native_target\": \"" << json_escape(NFN_NATIVE_TARGET_NAME) << "\",\n"
        << "  \"status\": \"family-native-trainer-missing\",\n"
        << "  \"kernel_status\": \"" << json_escape(kernel_status) << "\",\n"
        << "  \"trainer_loop_status\": \"family-native-loop-missing\",\n"
        << "  \"native_training_coverage_class\": \"" << json_escape(NFN_NATIVE_COVERAGE_CLASS) << "\",\n"
        << "  \"compiled_native_boundary\": true,\n"
        << "  \"torch_required\": false,\n"
        << "  \"graph_editor_tensor_flow\": false,\n"
        << "  \"native_token_batch_preflight\": " << (cfg.sample_token_batch ? "true" : "false") << ",\n"
        << "  \"template_name\": \"" << json_escape(cfg.template_name) << "\",\n"
        << "  \"graph_file\": \"" << json_escape(cfg.graph_file) << "\",\n"
        << "  \"dataset_alias\": \"" << json_escape(cfg.dataset_alias) << "\",\n"
        << "  \"output_dir\": \"" << json_escape(cfg.output_dir) << "\",\n"
        << "  \"schedule\": {\n"
        << "    \"max_steps\": " << cfg.max_steps << ",\n"
        << "    \"batch_size\": " << cfg.batch_size << ",\n"
        << "    \"train_seq_len\": " << cfg.train_seq_len << ",\n"
        << "    \"train_batch_tokens\": " << cfg.train_batch_tokens << ",\n"
        << "    \"eval_every_steps\": " << cfg.eval_every_steps << ",\n"
        << "    \"learning_rate\": " << cfg.learning_rate << "\n"
        << "  },\n"
        << "  \"required_native_work\": [\n"
        << "    \"" << json_escape(NFN_NATIVE_REQUIRED_KERNELS) << "\",\n"
        << "    \"wire the family forward/backward/optimizer loop to raw Tile ABI calls\",\n"
        << "    \"write native checkpoints and native inference metadata for this family\"\n"
        << "  ],\n"
        << "  \"native_training_missing_requirements\": [\n";
    const std::vector<std::string> missing_requirements = split_csv(NFN_NATIVE_MISSING_REQUIREMENTS);
    for (std::size_t i = 0; i < missing_requirements.size(); ++i) {
        std::cout << "    \"" << json_escape(missing_requirements[i]) << "\"";
        if (i + 1 != missing_requirements.size()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout
        << "  ],\n"
        << "  \"required_tile_symbols\": [\n";
    for (std::size_t i = 0; i < required_symbols.size(); ++i) {
        std::cout << "    \"" << json_escape(required_symbols[i]) << "\"";
        if (i + 1 != required_symbols.size()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout
        << "  ],\n"
        << "  \"unparsed_args\": [\n";
    for (std::size_t i = 0; i < cfg.unparsed_args.size(); ++i) {
        std::cout << "    \"" << json_escape(cfg.unparsed_args[i]) << "\"";
        if (i + 1 != cfg.unparsed_args.size()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "  ],\n"
        << "  \"token_shards\": ";
    if (have_dataset) {
        std::cout << neuralfn::native_train::token_shard_dataset_json(dataset, &batch_plan);
    } else {
        std::cout << "null";
    }
    std::cout << ",\n"
        << "  \"sample_batch\": ";
    if (have_batch) {
        std::cout << neuralfn::native_train::token_batch_json(sample_batch);
    } else {
        std::cout << "null";
    }
    if (symbol_check_requested) {
        std::cout
            << ",\n"
            << "  \"tile_ops_check\": {\n"
            << "    \"tile_ops_lib\": \"" << json_escape(tile_ops_lib) << "\",\n"
            << "    \"checked\": " << (cfg.check_tile_ops ? "true" : "false") << ",\n"
            << "    \"all_required_symbols_found\": " << (symbols_ok ? "true" : "false") << ",\n"
            << "    \"error\": \"" << json_escape(tile_ops_error) << "\",\n"
            << "    \"symbols\": [\n";
        for (std::size_t i = 0; i < symbols.size(); ++i) {
            std::cout
                << "      {\"name\": \"" << json_escape(symbols[i].name)
                << "\", \"found\": " << (symbols[i].found ? "true" : "false") << "}";
            if (i + 1 != symbols.size()) {
                std::cout << ",";
            }
            std::cout << "\n";
        }
        std::cout
            << "    ]\n"
            << "  }";
    }
    std::cout << "\n}\n";
}

int print_llama_loop_smoke_json(const Config& cfg, const char* program) {
    const std::string family = NFN_NATIVE_MODEL_FAMILY;
    const bool llama_family =
        family.find("llama") != std::string::npos ||
        family == "unknown";
    const std::string tile_ops_lib = resolve_tile_ops_lib(cfg, program);
    const std::vector<std::string> runtime_candidates = cuda_runtime_candidates(cfg);
    std::string cuda_lib_path;
    std::string error;
    void* tile_handle = nullptr;
    void* cuda_handle = nullptr;
    bool tile_ops_loaded = false;
    bool cuda_runtime_loaded = false;

    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);
    using RmsNormFn = int (*)(const float*, float*, std::int64_t, std::int64_t, float, void*);
    using RmsNormBackwardFn = int (*)(const float*, const float*, float*, std::int64_t, std::int64_t, float, void*);
    using RotaryFn = int (*)(const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using SwiGluFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
    using SwiGluBackwardFn = int (*)(const float*, const float*, const float*, float*, float*, std::int64_t, void*);
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

    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;
    RmsNormFn rms_norm = nullptr;
    RmsNormBackwardFn rms_norm_backward = nullptr;
    RotaryFn rotary = nullptr;
    RotaryFn rotary_backward = nullptr;
    SwiGluFn swiglu = nullptr;
    SwiGluBackwardFn swiglu_backward = nullptr;
    FillFn fill = nullptr;
    AdamWFn adamw = nullptr;

    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    constexpr std::int64_t kRows = 2;
    constexpr std::int64_t kDim = 8;
    constexpr std::int64_t kSeqLen = 2;
    constexpr std::int64_t kHeads = 2;
    constexpr std::int64_t kHeadDim = 4;
    constexpr std::int64_t kElements = kRows * kDim;
    constexpr float kEps = 1e-6f;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed";
        if (code != 0) {
            out << " with code " << code;
            if (cuda_get_error_string != nullptr) {
                const char* text = cuda_get_error_string(code);
                if (text != nullptr) {
                    out << " (" << text << ")";
                }
            }
        }
        return out.str();
    };
    auto close_handles = [&]() {
        if (tile_handle != nullptr) {
            dlclose(tile_handle);
            tile_handle = nullptr;
        }
        if (cuda_handle != nullptr) {
            dlclose(cuda_handle);
            cuda_handle = nullptr;
        }
    };
    auto max_abs_error = [](const std::vector<float>& actual, const std::vector<float>& expected) {
        float max_err = 0.0f;
        const std::size_t n = std::min(actual.size(), expected.size());
        for (std::size_t i = 0; i < n; ++i) {
            max_err = std::max(max_err, std::fabs(actual[i] - expected[i]));
        }
        return max_err;
    };
    bool passed = false;
    float rms_max_error = 0.0f;
    float rms_backward_max_error = 0.0f;
    float rotary_max_error = 0.0f;
    float rotary_backward_max_error = 0.0f;
    float swiglu_max_error = 0.0f;
    float swiglu_backward_gate_max_error = 0.0f;
    float swiglu_backward_up_max_error = 0.0f;
    float adamw_param_max_error = 0.0f;
    float adamw_moment_max_error = 0.0f;

    std::vector<void*> allocated;
    auto free_allocated = [&]() {
        if (cuda_free == nullptr) {
            return;
        }
        for (void* ptr : allocated) {
            if (ptr != nullptr) {
                cuda_free(ptr);
            }
        }
        allocated.clear();
    };

    if (!llama_family) {
        error = "LLaMA smoke commands are only valid for LLaMA-family native preflights";
    } else {
        tile_handle = dlopen(tile_ops_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            const char* raw = dlerror();
            error = raw == nullptr ? "failed to load Tile ops library" : raw;
        } else {
            tile_ops_loaded = true;
            rms_norm = load_symbol<RmsNormFn>(tile_handle, "nfn_native_tile_rms_norm_float32");
            rms_norm_backward = load_symbol<RmsNormBackwardFn>(
                tile_handle, "nfn_native_tile_rms_norm_backward_input_float32");
            rotary = load_symbol<RotaryFn>(tile_handle, "nfn_native_tile_rotary_embedding_float32");
            rotary_backward = load_symbol<RotaryFn>(
                tile_handle, "nfn_native_tile_rotary_embedding_backward_float32");
            swiglu = load_symbol<SwiGluFn>(tile_handle, "nfn_native_tile_swiglu_float32");
            swiglu_backward = load_symbol<SwiGluBackwardFn>(
                tile_handle, "nfn_native_tile_swiglu_backward_float32");
            fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
            adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
            if (rms_norm == nullptr || rms_norm_backward == nullptr || rotary == nullptr ||
                rotary_backward == nullptr || swiglu == nullptr || swiglu_backward == nullptr ||
                (cfg.smoke_llama_train_step && (fill == nullptr || adamw == nullptr))) {
                error = "Tile ops library is missing one or more LLaMA loop-composition symbols";
            }
        }
    }
    if (error.empty()) {
        for (const std::string& candidate : runtime_candidates) {
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_lib_path = candidate;
                cuda_runtime_loaded = true;
                break;
            }
        }
        if (!cuda_runtime_loaded) {
            error = "failed to load CUDA runtime";
        } else {
            cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
            cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
            cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
            cuda_device_synchronize =
                load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
            cuda_get_error_string =
                load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
            if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr ||
                cuda_device_synchronize == nullptr) {
                error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
            }
        }
    }
    if (error.empty()) {
        std::vector<float> x(static_cast<std::size_t>(kElements));
        std::vector<float> grad(static_cast<std::size_t>(kElements));
        std::vector<float> inv_freq(static_cast<std::size_t>(kHeadDim / 2));
        for (std::int64_t i = 0; i < kElements; ++i) {
            x[static_cast<std::size_t>(i)] = 0.05f * static_cast<float>(i + 1);
            grad[static_cast<std::size_t>(i)] = 0.02f * static_cast<float>((i % 7) + 1);
        }
        for (std::int64_t i = 0; i < kHeadDim / 2; ++i) {
            inv_freq[static_cast<std::size_t>(i)] = 1.0f / std::pow(10000.0f, (2.0f * static_cast<float>(i)) / static_cast<float>(kHeadDim));
        }

        float* d_x = nullptr;
        float* d_grad = nullptr;
        float* d_out = nullptr;
        float* d_grad_x = nullptr;
        float* d_inv = nullptr;
        float* d_gate_grad = nullptr;
        float* d_up_grad = nullptr;
        float* d_param = nullptr;
        float* d_param_grad = nullptr;
        float* d_exp_avg = nullptr;
        float* d_exp_avg_sq = nullptr;
        auto alloc = [&](float** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(float));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        if (alloc(&d_x, x.size(), "x") &&
            alloc(&d_grad, grad.size(), "grad") &&
            alloc(&d_out, x.size(), "out") &&
            alloc(&d_grad_x, x.size(), "grad_x") &&
            alloc(&d_inv, inv_freq.size(), "inv_freq") &&
            alloc(&d_gate_grad, x.size(), "grad_gate") &&
            alloc(&d_up_grad, x.size(), "grad_up") &&
            (!cfg.smoke_llama_train_step ||
             (alloc(&d_param, 4, "adamw_param") &&
              alloc(&d_param_grad, 4, "adamw_grad") &&
              alloc(&d_exp_avg, 4, "adamw_exp_avg") &&
              alloc(&d_exp_avg_sq, 4, "adamw_exp_avg_sq")))) {
            int status = cuda_memcpy(d_x, x.data(), x.size() * sizeof(float), kCudaMemcpyHostToDevice);
            if (status != 0) {
                error = cuda_error(status, "cudaMemcpy x H2D");
            }
            if (error.empty()) {
                status = cuda_memcpy(d_grad, grad.data(), grad.size() * sizeof(float), kCudaMemcpyHostToDevice);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy grad H2D");
                }
            }
            if (error.empty()) {
                status = cuda_memcpy(d_inv, inv_freq.data(), inv_freq.size() * sizeof(float), kCudaMemcpyHostToDevice);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy inv_freq H2D");
                }
            }
            if (error.empty()) {
                status = rms_norm(d_x, d_out, kRows, kDim, kEps, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_rms_norm_float32");
                }
            }
            if (error.empty()) {
                status = rms_norm_backward(d_x, d_grad, d_grad_x, kRows, kDim, kEps, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_rms_norm_backward_input_float32");
                }
            }
            if (error.empty()) {
                status = rotary(d_x, d_inv, d_out, kElements, kHeads, kSeqLen, kHeadDim, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_rotary_embedding_float32");
                }
            }
            if (error.empty()) {
                status = rotary_backward(d_grad, d_inv, d_grad_x, kElements, kHeads, kSeqLen, kHeadDim, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_rotary_embedding_backward_float32");
                }
            }
            if (error.empty()) {
                status = swiglu(d_x, d_grad, d_out, kElements, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_swiglu_float32");
                }
            }
            if (error.empty()) {
                status = swiglu_backward(d_x, d_grad, d_grad, d_gate_grad, d_up_grad, kElements, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_swiglu_backward_float32");
                }
            }
            if (error.empty() && cfg.smoke_llama_train_step) {
                status = fill(d_param, 4, 0.5f, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "fill adamw_param");
                }
            }
            if (error.empty() && cfg.smoke_llama_train_step) {
                status = fill(d_param_grad, 4, 0.25f, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "fill adamw_grad");
                }
            }
            if (error.empty() && cfg.smoke_llama_train_step) {
                status = fill(d_exp_avg, 4, 0.0f, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "fill adamw_exp_avg");
                }
            }
            if (error.empty() && cfg.smoke_llama_train_step) {
                status = fill(d_exp_avg_sq, 4, 0.0f, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "fill adamw_exp_avg_sq");
                }
            }
            if (error.empty() && cfg.smoke_llama_train_step) {
                status = adamw(
                    d_param,
                    d_param_grad,
                    d_exp_avg,
                    d_exp_avg_sq,
                    4,
                    0.001f,
                    0.9f,
                    0.999f,
                    1e-8f,
                    0.01f,
                    0.1f,
                    std::sqrt(0.001f),
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_adamw_step_float32");
                }
            }
            if (error.empty()) {
                status = cuda_device_synchronize();
                if (status != 0) {
                    error = cuda_error(status, "cudaDeviceSynchronize");
                }
            }
            std::vector<float> actual_out(x.size(), 0.0f);
            std::vector<float> actual_grad_x(x.size(), 0.0f);
            std::vector<float> actual_gate_grad(x.size(), 0.0f);
            std::vector<float> actual_up_grad(x.size(), 0.0f);
            std::vector<float> actual_param(4, 0.0f);
            std::vector<float> actual_exp_avg(4, 0.0f);
            std::vector<float> actual_exp_avg_sq(4, 0.0f);
            if (error.empty()) {
                status = cuda_memcpy(actual_out.data(), d_out, actual_out.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy out D2H");
                }
            }
            if (error.empty()) {
                status = cuda_memcpy(actual_grad_x.data(), d_grad_x, actual_grad_x.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy grad_x D2H");
                }
            }
            if (error.empty()) {
                status = cuda_memcpy(actual_gate_grad.data(), d_gate_grad, actual_gate_grad.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy grad_gate D2H");
                }
            }
            if (error.empty()) {
                status = cuda_memcpy(actual_up_grad.data(), d_up_grad, actual_up_grad.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy grad_up D2H");
                }
            }
            if (error.empty() && cfg.smoke_llama_train_step) {
                status = cuda_memcpy(actual_param.data(), d_param, actual_param.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy adamw_param D2H");
                }
            }
            if (error.empty() && cfg.smoke_llama_train_step) {
                status = cuda_memcpy(actual_exp_avg.data(), d_exp_avg, actual_exp_avg.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy adamw_exp_avg D2H");
                }
            }
            if (error.empty() && cfg.smoke_llama_train_step) {
                status = cuda_memcpy(actual_exp_avg_sq.data(), d_exp_avg_sq, actual_exp_avg_sq.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy adamw_exp_avg_sq D2H");
                }
            }

            std::vector<float> rms_expected(x.size(), 0.0f);
            std::vector<float> rms_bwd_expected(x.size(), 0.0f);
            for (std::int64_t row = 0; row < kRows; ++row) {
                float sum_sq = 0.0f;
                float dot = 0.0f;
                for (std::int64_t d = 0; d < kDim; ++d) {
                    const std::size_t idx = static_cast<std::size_t>(row * kDim + d);
                    sum_sq += x[idx] * x[idx];
                    dot += grad[idx] * x[idx];
                }
                const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(kDim) + kEps);
                for (std::int64_t d = 0; d < kDim; ++d) {
                    const std::size_t idx = static_cast<std::size_t>(row * kDim + d);
                    rms_expected[idx] = x[idx] * inv_rms;
                    rms_bwd_expected[idx] =
                        grad[idx] * inv_rms -
                        x[idx] * inv_rms * inv_rms * inv_rms * (dot / static_cast<float>(kDim));
                }
            }
            std::vector<float> rotary_expected(x.size(), 0.0f);
            std::vector<float> rotary_bwd_expected(x.size(), 0.0f);
            for (std::int64_t idx = 0; idx < kElements; ++idx) {
                const std::int64_t d = idx % kHeadDim;
                const std::int64_t s = (idx / kHeadDim) % kSeqLen;
                const std::int64_t pair = d % (kHeadDim / 2);
                const std::int64_t base = idx - d;
                const float angle = static_cast<float>(s) * inv_freq[static_cast<std::size_t>(pair)];
                const float c = std::cos(angle);
                const float sn = std::sin(angle);
                const float x1 = x[static_cast<std::size_t>(base + pair)];
                const float x2 = x[static_cast<std::size_t>(base + pair + kHeadDim / 2)];
                const float g1 = grad[static_cast<std::size_t>(base + pair)];
                const float g2 = grad[static_cast<std::size_t>(base + pair + kHeadDim / 2)];
                rotary_expected[static_cast<std::size_t>(idx)] =
                    d < kHeadDim / 2 ? (x1 * c + x2 * sn) : (-x1 * sn + x2 * c);
                rotary_bwd_expected[static_cast<std::size_t>(idx)] =
                    d < kHeadDim / 2 ? (g1 * c - g2 * sn) : (g1 * sn + g2 * c);
            }
            std::vector<float> swiglu_expected(x.size(), 0.0f);
            std::vector<float> swiglu_gate_grad_expected(x.size(), 0.0f);
            std::vector<float> swiglu_up_grad_expected(x.size(), 0.0f);
            for (std::size_t i = 0; i < x.size(); ++i) {
                const float sig = 1.0f / (1.0f + std::exp(-x[i]));
                const float act = x[i] * sig;
                const float dact = sig * (1.0f + x[i] * (1.0f - sig));
                swiglu_expected[i] = act * grad[i];
                swiglu_gate_grad_expected[i] = grad[i] * grad[i] * dact;
                swiglu_up_grad_expected[i] = grad[i] * act;
            }

            if (error.empty()) {
                rms_max_error = max_abs_error(actual_out, swiglu_expected);
                swiglu_max_error = rms_max_error;
                swiglu_backward_gate_max_error = max_abs_error(actual_gate_grad, swiglu_gate_grad_expected);
                swiglu_backward_up_max_error = max_abs_error(actual_up_grad, swiglu_up_grad_expected);
                // Re-run and copy each earlier stage for exact per-stage checks.
                status = rms_norm(d_x, d_out, kRows, kDim, kEps, nullptr);
                if (status == 0) {
                    status = cuda_device_synchronize();
                }
                if (status == 0) {
                    status = cuda_memcpy(actual_out.data(), d_out, actual_out.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                }
                if (status != 0) {
                    error = cuda_error(status, "RMSNorm verification copy");
                } else {
                    rms_max_error = max_abs_error(actual_out, rms_expected);
                }
            }
            if (error.empty()) {
                status = rms_norm_backward(d_x, d_grad, d_grad_x, kRows, kDim, kEps, nullptr);
                if (status == 0) {
                    status = cuda_device_synchronize();
                }
                if (status == 0) {
                    status = cuda_memcpy(actual_grad_x.data(), d_grad_x, actual_grad_x.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                }
                if (status != 0) {
                    error = cuda_error(status, "RMSNorm backward verification copy");
                } else {
                    rms_backward_max_error = max_abs_error(actual_grad_x, rms_bwd_expected);
                }
            }
            if (error.empty()) {
                status = rotary(d_x, d_inv, d_out, kElements, kHeads, kSeqLen, kHeadDim, nullptr);
                if (status == 0) {
                    status = cuda_device_synchronize();
                }
                if (status == 0) {
                    status = cuda_memcpy(actual_out.data(), d_out, actual_out.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                }
                if (status != 0) {
                    error = cuda_error(status, "RoPE verification copy");
                } else {
                    rotary_max_error = max_abs_error(actual_out, rotary_expected);
                }
            }
            if (error.empty()) {
                status = rotary_backward(d_grad, d_inv, d_grad_x, kElements, kHeads, kSeqLen, kHeadDim, nullptr);
                if (status == 0) {
                    status = cuda_device_synchronize();
                }
                if (status == 0) {
                    status = cuda_memcpy(actual_grad_x.data(), d_grad_x, actual_grad_x.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                }
                if (status != 0) {
                    error = cuda_error(status, "RoPE backward verification copy");
                } else {
                    rotary_backward_max_error = max_abs_error(actual_grad_x, rotary_bwd_expected);
                }
            }
            const float tolerance = 2e-6f;
            if (error.empty() && cfg.smoke_llama_train_step) {
                const float grad_value = 0.25f;
                const float expected_exp_avg = 0.9f * 0.0f + 0.1f * grad_value;
                const float expected_exp_avg_sq = 0.999f * 0.0f + 0.001f * grad_value * grad_value;
                const float denom = std::sqrt(expected_exp_avg_sq) / std::sqrt(0.001f) + 1e-8f;
                const float decayed_param = 0.5f * (1.0f - 0.001f * 0.01f);
                const float expected_param = decayed_param - 0.001f * (expected_exp_avg / 0.1f) / denom;
                for (std::size_t i = 0; i < actual_param.size(); ++i) {
                    adamw_param_max_error = std::max(adamw_param_max_error, std::fabs(actual_param[i] - expected_param));
                    adamw_moment_max_error = std::max(
                        adamw_moment_max_error,
                        std::max(
                            std::fabs(actual_exp_avg[i] - expected_exp_avg),
                            std::fabs(actual_exp_avg_sq[i] - expected_exp_avg_sq)));
                }
            }
            passed = error.empty() &&
                rms_max_error <= tolerance &&
                rms_backward_max_error <= tolerance &&
                rotary_max_error <= tolerance &&
                rotary_backward_max_error <= tolerance &&
                swiglu_max_error <= tolerance &&
                swiglu_backward_gate_max_error <= tolerance &&
                swiglu_backward_up_max_error <= tolerance &&
                (!cfg.smoke_llama_train_step ||
                 (adamw_param_max_error <= tolerance && adamw_moment_max_error <= tolerance));
            if (!passed && error.empty()) {
                error = "LLaMA loop-composition smoke exceeded tolerance";
            }
        }
    }
    free_allocated();
    close_handles();

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(NFN_NATIVE_MODEL_FAMILY) << "\",\n"
        << "  \"native_target\": \"" << json_escape(NFN_NATIVE_TARGET_NAME) << "\",\n"
        << "  \"smoke\": \"" << (cfg.smoke_llama_train_step ? "llama_train_step_slice" : "llama_loop_composition") << "\",\n"
        << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(error) << "\",\n"
        << "  \"compiled_native_boundary\": true,\n"
        << "  \"torch_required\": false,\n"
        << "  \"graph_editor_tensor_flow\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_ops_lib) << "\",\n"
        << "  \"tile_ops_loaded\": " << (tile_ops_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"shape\": {\"rows\": " << kRows << ", \"dim\": " << kDim
        << ", \"heads\": " << kHeads << ", \"seq_len\": " << kSeqLen
        << ", \"head_dim\": " << kHeadDim << "},\n"
        << "  \"loop_composition_stages\": [\n"
        << "    \"nfn_native_tile_rms_norm_float32\",\n"
        << "    \"nfn_native_tile_rms_norm_backward_input_float32\",\n"
        << "    \"nfn_native_tile_rotary_embedding_float32\",\n"
        << "    \"nfn_native_tile_rotary_embedding_backward_float32\",\n"
        << "    \"nfn_native_tile_swiglu_float32\",\n"
        << "    \"nfn_native_tile_swiglu_backward_float32\"";
    if (cfg.smoke_llama_train_step) {
        std::cout << ",\n"
                  << "    \"nfn_native_tile_fill_float32\",\n"
                  << "    \"nfn_native_tile_adamw_step_float32\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout
        << "  ],\n"
        << "  \"max_errors\": {"
        << "\"rms_norm\":" << rms_max_error
        << ", \"rms_norm_backward\":" << rms_backward_max_error
        << ", \"rotary\":" << rotary_max_error
        << ", \"rotary_backward\":" << rotary_backward_max_error
        << ", \"swiglu\":" << swiglu_max_error
        << ", \"swiglu_backward_gate\":" << swiglu_backward_gate_max_error
        << ", \"swiglu_backward_up\":" << swiglu_backward_up_max_error
        << ", \"adamw_param\":" << adamw_param_max_error
        << ", \"adamw_moment\":" << adamw_moment_max_error
        << "}\n"
        << "}\n";
    return passed ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
    const Config cfg = parse_args(argc, argv);
    try {
        if (cfg.smoke_llama_loop || cfg.smoke_llama_train_step) {
            return print_llama_loop_smoke_json(cfg, argv[0]);
        }
        if (cfg.print_plan || cfg.check_tile_ops || cfg.dry_run || cfg.sample_token_batch) {
            print_json(cfg, argv[0]);
            return 0;
        }
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 2;
    }

    try {
        print_json(cfg, argv[0]);
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 2;
    }
    std::cerr
        << NFN_NATIVE_TARGET_NAME << ": native CUDA Tile trainer for " << NFN_NATIVE_MODEL_FAMILY
        << " is not implemented yet.\n"
        << "Required native work: " << NFN_NATIVE_REQUIRED_KERNELS << "\n"
        << "Do not use the graph-backed TorchTrainer path for production training; implement this "
        << "family's CUDA Tile C++ trainer loop first. For local graph-backed debugging, call the Python "
        << "SDK trainer APIs directly instead of routing through nfn train.\n";
    return 2;
}
