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

#ifndef NFN_NATIVE_COMPLETED_REQUIREMENTS
#define NFN_NATIVE_COMPLETED_REQUIREMENTS ""
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
    bool smoke_llama_lm_head_step = false;
    bool smoke_llama_packed_attention_step = false;
    bool smoke_llama_attention_block_step = false;
    bool smoke_moe_route_expert_step = false;
    bool smoke_moe_transformer_block_step = false;
    bool smoke_jepa_projector_step = false;
    bool smoke_jepa_target_encoder_step = false;
    bool smoke_jepa_ar_loss_step = false;
    bool smoke_semantic_alignment_step = false;
    bool smoke_diffusion_denoise_step = false;
    bool smoke_seq2seq_cross_attention_step = false;
    bool smoke_ttt_linear_inner_step = false;
    bool smoke_universal_recurrent_step = false;
    bool smoke_universal_act_halt_step = false;
    bool smoke_hnet_byte_patch_step = false;
    bool smoke_jamba_chunk_state_step = false;
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
        << "  --smoke-llama-lm-head-step Launch LLaMA loop plus LM-head CE/backward/AdamW kernels on CUDA\n"
        << "  --smoke-llama-packed-attention-step Launch packed-QKV BF16 attention forward/backward kernels on CUDA\n"
        << "  --smoke-llama-attention-block-step Launch RMSNorm, QKV projection, packed attention, and residual kernels on CUDA\n"
        << "  --smoke-moe-route-expert-step Launch MoE routing, expert, balance-loss, and AdamW kernels on CUDA\n"
        << "  --smoke-moe-transformer-block-step Launch attention, routing, MoE expert, and residual kernels on CUDA\n"
        << "  --smoke-jepa-projector-step Launch JEPA projector/predictor, latent loss, backward, and AdamW kernels on CUDA\n"
        << "  --smoke-jepa-target-encoder-step Launch JEPA target latent-pool and projection kernels on CUDA\n"
        << "  --smoke-jepa-ar-loss-step Launch AR CE plus JEPA latent-loss composition kernels on CUDA\n"
        << "  --smoke-semantic-alignment-step Launch semantic hash and alignment-loss kernels on CUDA\n"
        << "  --smoke-diffusion-denoise-step Launch diffusion denoise head, loss, backward, and AdamW kernels on CUDA\n"
        << "  --smoke-seq2seq-cross-attention-step Launch seq2seq cross-attention, CE, backward, and AdamW kernels on CUDA\n"
        << "  --smoke-ttt-linear-inner-step Launch TTT inner linear loss, backward, and AdamW kernels on CUDA\n"
        << "  --smoke-universal-recurrent-step Launch universal recurrent linear loss, backward, and AdamW kernels on CUDA\n"
        << "  --smoke-universal-act-halt-step Launch universal ACT halt gate, weighted sum, BCE-gradient, and AdamW kernels on CUDA\n"
        << "  --smoke-hnet-byte-patch-step Launch HNet byte patch embed/merge, head loss, backward, and AdamW kernels on CUDA\n"
        << "  --smoke-jamba-chunk-state-step Launch Jamba chunk state, head loss, backward, and AdamW kernels on CUDA\n"
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
        } else if (arg == "--smoke-llama-lm-head-step" || arg == "--native-cuda-smoke-llama-lm-head-step") {
            cfg.smoke_llama_lm_head_step = true;
        } else if (arg == "--smoke-llama-packed-attention-step" || arg == "--native-cuda-smoke-llama-packed-attention-step") {
            cfg.smoke_llama_packed_attention_step = true;
        } else if (arg == "--smoke-llama-attention-block-step" || arg == "--native-cuda-smoke-llama-attention-block-step") {
            cfg.smoke_llama_attention_block_step = true;
        } else if (arg == "--smoke-moe-route-expert-step" || arg == "--native-cuda-smoke-moe-route-expert-step") {
            cfg.smoke_moe_route_expert_step = true;
        } else if (arg == "--smoke-moe-transformer-block-step" || arg == "--native-cuda-smoke-moe-transformer-block-step") {
            cfg.smoke_moe_transformer_block_step = true;
        } else if (arg == "--smoke-jepa-projector-step" || arg == "--native-cuda-smoke-jepa-projector-step") {
            cfg.smoke_jepa_projector_step = true;
        } else if (arg == "--smoke-jepa-target-encoder-step" || arg == "--native-cuda-smoke-jepa-target-encoder-step") {
            cfg.smoke_jepa_target_encoder_step = true;
        } else if (arg == "--smoke-jepa-ar-loss-step" || arg == "--native-cuda-smoke-jepa-ar-loss-step") {
            cfg.smoke_jepa_ar_loss_step = true;
        } else if (arg == "--smoke-semantic-alignment-step" || arg == "--native-cuda-smoke-semantic-alignment-step") {
            cfg.smoke_semantic_alignment_step = true;
        } else if (arg == "--smoke-diffusion-denoise-step" || arg == "--native-cuda-smoke-diffusion-denoise-step") {
            cfg.smoke_diffusion_denoise_step = true;
        } else if (arg == "--smoke-seq2seq-cross-attention-step" || arg == "--native-cuda-smoke-seq2seq-cross-attention-step") {
            cfg.smoke_seq2seq_cross_attention_step = true;
        } else if (arg == "--smoke-ttt-linear-inner-step" || arg == "--native-cuda-smoke-ttt-linear-inner-step") {
            cfg.smoke_ttt_linear_inner_step = true;
        } else if (arg == "--smoke-universal-recurrent-step" || arg == "--native-cuda-smoke-universal-recurrent-step") {
            cfg.smoke_universal_recurrent_step = true;
        } else if (arg == "--smoke-universal-act-halt-step" || arg == "--native-cuda-smoke-universal-act-halt-step") {
            cfg.smoke_universal_act_halt_step = true;
        } else if (arg == "--smoke-hnet-byte-patch-step" || arg == "--native-cuda-smoke-hnet-byte-patch-step") {
            cfg.smoke_hnet_byte_patch_step = true;
        } else if (arg == "--smoke-jamba-chunk-state-step" || arg == "--native-cuda-smoke-jamba-chunk-state-step") {
            cfg.smoke_jamba_chunk_state_step = true;
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
        << "  \"native_training_completed_requirements\": [\n";
    const std::vector<std::string> completed_requirements = split_csv(NFN_NATIVE_COMPLETED_REQUIREMENTS);
    for (std::size_t i = 0; i < completed_requirements.size(); ++i) {
        std::cout << "    \"" << json_escape(completed_requirements[i]) << "\"";
        if (i + 1 != completed_requirements.size()) {
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
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightFn linear_backward_weight = nullptr;
    TokenCrossEntropyPartialsFn ce_partials = nullptr;
    TokenCrossEntropyBackwardFn ce_backward = nullptr;
    AdamWFn adamw = nullptr;

    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    constexpr std::int64_t kRows = 2;
    constexpr std::int64_t kDim = 8;
    constexpr std::int64_t kSeqLen = 2;
    constexpr std::int64_t kHeads = 2;
    constexpr std::int64_t kHeadDim = 4;
    constexpr std::int64_t kElements = kRows * kDim;
    constexpr std::int64_t kLmVocab = 4;
    constexpr std::int64_t kLmWeightElements = kLmVocab * kDim;
    constexpr std::int64_t kLmLogitElements = kRows * kLmVocab;
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
    float lm_head_loss_max_error = 0.0f;
    float lm_head_grad_weight_max_error = 0.0f;
    float lm_head_weight_max_error = 0.0f;

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
            if (rms_norm == nullptr || rms_norm_backward == nullptr || rotary == nullptr ||
                rotary_backward == nullptr || swiglu == nullptr || swiglu_backward == nullptr ||
                ((cfg.smoke_llama_train_step || cfg.smoke_llama_lm_head_step) &&
                 (fill == nullptr || adamw == nullptr)) ||
                (cfg.smoke_llama_lm_head_step &&
                 (linear == nullptr || linear_backward_input == nullptr || linear_backward_weight == nullptr ||
                  ce_partials == nullptr || ce_backward == nullptr))) {
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
        float* d_lm_weight = nullptr;
        float* d_lm_grad_weight = nullptr;
        float* d_lm_exp_avg = nullptr;
        float* d_lm_exp_avg_sq = nullptr;
        float* d_lm_logits = nullptr;
        float* d_lm_loss = nullptr;
        float* d_lm_grad_logits = nullptr;
        float* d_lm_grad_hidden = nullptr;
        std::int64_t* d_lm_targets = nullptr;
        auto alloc = [&](float** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(float));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        auto alloc_i64 = [&](std::int64_t** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(std::int64_t));
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
              alloc(&d_exp_avg_sq, 4, "adamw_exp_avg_sq"))) &&
            (!cfg.smoke_llama_lm_head_step ||
             (alloc(&d_lm_weight, kLmWeightElements, "lm_weight") &&
              alloc(&d_lm_grad_weight, kLmWeightElements, "lm_grad_weight") &&
              alloc(&d_lm_exp_avg, kLmWeightElements, "lm_exp_avg") &&
              alloc(&d_lm_exp_avg_sq, kLmWeightElements, "lm_exp_avg_sq") &&
              alloc(&d_lm_logits, kLmLogitElements, "lm_logits") &&
              alloc(&d_lm_loss, 1, "lm_loss") &&
              alloc(&d_lm_grad_logits, kLmLogitElements, "lm_grad_logits") &&
              alloc(&d_lm_grad_hidden, kElements, "lm_grad_hidden") &&
              alloc_i64(&d_lm_targets, kRows, "lm_targets")))) {
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
            if (error.empty() && cfg.smoke_llama_lm_head_step) {
                status = fill(d_lm_weight, kLmWeightElements, 0.1f, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "fill lm_weight");
                }
            }
            if (error.empty() && cfg.smoke_llama_lm_head_step) {
                status = fill(d_lm_grad_weight, kLmWeightElements, 0.0f, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "fill lm_grad_weight");
                }
            }
            if (error.empty() && cfg.smoke_llama_lm_head_step) {
                status = fill(d_lm_exp_avg, kLmWeightElements, 0.0f, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "fill lm_exp_avg");
                }
            }
            if (error.empty() && cfg.smoke_llama_lm_head_step) {
                status = fill(d_lm_exp_avg_sq, kLmWeightElements, 0.0f, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "fill lm_exp_avg_sq");
                }
            }
            if (error.empty() && cfg.smoke_llama_lm_head_step) {
                const std::int64_t lm_targets[2] = {1, 2};
                status = cuda_memcpy(d_lm_targets, lm_targets, sizeof(lm_targets), kCudaMemcpyHostToDevice);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy lm_targets H2D");
                }
            }
            if (error.empty() && cfg.smoke_llama_lm_head_step) {
                status = linear(d_x, d_lm_weight, nullptr, d_lm_logits, kRows, kDim, kLmVocab, false, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_float32");
                }
            }
            if (error.empty() && cfg.smoke_llama_lm_head_step) {
                status = ce_partials(d_lm_logits, d_lm_targets, d_lm_loss, kRows, kLmVocab, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_token_cross_entropy_partials_float32");
                }
            }
            if (error.empty() && cfg.smoke_llama_lm_head_step) {
                status = ce_backward(
                    d_lm_logits,
                    d_lm_targets,
                    d_lm_grad_logits,
                    kRows,
                    kLmVocab,
                    1.0f / static_cast<float>(kRows),
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_token_cross_entropy_backward_float32");
                }
            }
            if (error.empty() && cfg.smoke_llama_lm_head_step) {
                status = linear_backward_input(
                    d_lm_grad_logits,
                    d_lm_weight,
                    d_lm_grad_hidden,
                    kRows,
                    kDim,
                    kLmVocab,
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_backward_input_float32");
                }
            }
            if (error.empty() && cfg.smoke_llama_lm_head_step) {
                status = linear_backward_weight(
                    d_x,
                    d_lm_grad_logits,
                    d_lm_grad_weight,
                    kRows,
                    kDim,
                    kLmVocab,
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_backward_weight_float32");
                }
            }
            if (error.empty() && cfg.smoke_llama_lm_head_step) {
                status = adamw(
                    d_lm_weight,
                    d_lm_grad_weight,
                    d_lm_exp_avg,
                    d_lm_exp_avg_sq,
                    kLmWeightElements,
                    0.1f,
                    0.9f,
                    0.95f,
                    1e-8f,
                    0.1f,
                    0.1f,
                    std::sqrt(0.05f),
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_adamw_step_float32 lm_head");
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
            std::vector<float> actual_lm_weight(static_cast<std::size_t>(kLmWeightElements), 0.0f);
            std::vector<float> actual_lm_grad_weight(static_cast<std::size_t>(kLmWeightElements), 0.0f);
            std::vector<float> actual_lm_loss(1, 0.0f);
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
            if (error.empty() && cfg.smoke_llama_lm_head_step) {
                status = cuda_memcpy(actual_lm_weight.data(), d_lm_weight, actual_lm_weight.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy lm_weight D2H");
                }
            }
            if (error.empty() && cfg.smoke_llama_lm_head_step) {
                status = cuda_memcpy(actual_lm_grad_weight.data(), d_lm_grad_weight, actual_lm_grad_weight.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy lm_grad_weight D2H");
                }
            }
            if (error.empty() && cfg.smoke_llama_lm_head_step) {
                status = cuda_memcpy(actual_lm_loss.data(), d_lm_loss, actual_lm_loss.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy lm_loss D2H");
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
            const float lm_head_tolerance = 1e-4f;
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
            if (error.empty() && cfg.smoke_llama_lm_head_step) {
                const std::int64_t lm_targets[2] = {1, 2};
                const float expected_loss = static_cast<float>(kRows) * std::log(static_cast<float>(kLmVocab));
                lm_head_loss_max_error = std::fabs(actual_lm_loss[0] - expected_loss);
                for (std::int64_t token = 0; token < kLmVocab; ++token) {
                    for (std::int64_t dim = 0; dim < kDim; ++dim) {
                        float expected_grad = 0.0f;
                        for (std::int64_t row = 0; row < kRows; ++row) {
                            const float prob = 1.0f / static_cast<float>(kLmVocab);
                            const float target_term = lm_targets[row] == token ? 1.0f : 0.0f;
                            const float grad_logit =
                                (prob - target_term) / static_cast<float>(kRows);
                            expected_grad += x[static_cast<std::size_t>(row * kDim + dim)] * grad_logit;
                        }
                        const std::size_t idx = static_cast<std::size_t>(token * kDim + dim);
                        const float adamw_grad = actual_lm_grad_weight[idx];
                        const float next_m = 0.1f * adamw_grad;
                        const float next_v = 0.05f * adamw_grad * adamw_grad;
                        const float denom = std::sqrt(next_v) / std::sqrt(0.05f) + 1e-8f;
                        const float decayed_weight = 0.1f * (1.0f - 0.1f * 0.1f);
                        const float expected_weight = decayed_weight - 0.1f * (next_m / 0.1f) / denom;
                        lm_head_grad_weight_max_error = std::max(
                            lm_head_grad_weight_max_error,
                            std::fabs(actual_lm_grad_weight[idx] - expected_grad));
                        lm_head_weight_max_error = std::max(
                            lm_head_weight_max_error,
                            std::fabs(actual_lm_weight[idx] - expected_weight));
                    }
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
                 (adamw_param_max_error <= tolerance && adamw_moment_max_error <= tolerance)) &&
                (!cfg.smoke_llama_lm_head_step ||
                 (lm_head_loss_max_error <= lm_head_tolerance &&
                  lm_head_grad_weight_max_error <= lm_head_tolerance &&
                  lm_head_weight_max_error <= lm_head_tolerance));
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
        << "  \"smoke\": \""
        << (cfg.smoke_llama_lm_head_step
                ? "llama_lm_head_train_step_slice"
                : (cfg.smoke_llama_train_step ? "llama_train_step_slice" : "llama_loop_composition"))
        << "\",\n"
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
    if (cfg.smoke_llama_train_step || cfg.smoke_llama_lm_head_step) {
        std::cout << ",\n"
                  << "    \"nfn_native_tile_fill_float32\"";
        if (cfg.smoke_llama_lm_head_step) {
            std::cout << ",\n"
                      << "    \"nfn_native_tile_linear_float32\",\n"
                      << "    \"nfn_native_tile_token_cross_entropy_partials_float32\",\n"
                      << "    \"nfn_native_tile_token_cross_entropy_backward_float32\",\n"
                      << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
                      << "    \"nfn_native_tile_linear_backward_weight_float32\",\n"
                      << "    \"nfn_native_tile_adamw_step_float32\"\n";
        } else {
            std::cout << ",\n"
                      << "    \"nfn_native_tile_adamw_step_float32\"\n";
        }
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
        << ", \"lm_head_loss\":" << lm_head_loss_max_error
        << ", \"lm_head_grad_weight\":" << lm_head_grad_weight_max_error
        << ", \"lm_head_weight\":" << lm_head_weight_max_error
        << "}\n"
        << "}\n";
    return passed ? 0 : 2;
}

int print_llama_packed_attention_smoke_json(const Config& cfg, const char* program) {
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
    using Float32ToBf16Fn = int (*)(const float*, std::uint16_t*, std::int64_t, void*);
    using Bf16ToFloat32Fn = int (*)(const std::uint16_t*, float*, std::int64_t, void*);
    using PackedAttentionStoreLseFn = int (*)(
        const std::uint16_t*,
        std::uint16_t*,
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
    using PackedAttentionBackwardFn = int (*)(
        const std::uint16_t*,
        const std::uint16_t*,
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

    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;
    Float32ToBf16Fn float32_to_bf16 = nullptr;
    Bf16ToFloat32Fn bf16_to_float32 = nullptr;
    PackedAttentionStoreLseFn packed_attention = nullptr;
    PackedAttentionBackwardFn packed_attention_backward = nullptr;

    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    constexpr std::int64_t kBatch = 1;
    constexpr std::int64_t kHeads = 12;
    constexpr std::int64_t kSeqLen = 32;
    constexpr std::int64_t kHeadDim = 64;
    constexpr std::int64_t kElements = kBatch * kSeqLen * kHeads * kHeadDim;
    constexpr std::int64_t kPackedElements = kElements * 3;
    constexpr float kScale = 1.0f / 8.0f;

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
    auto packed_index = [](std::int64_t b, std::int64_t t, std::int64_t part, std::int64_t h, std::int64_t d) {
        return (((b * kSeqLen + t) * 3 + part) * kHeads + h) * kHeadDim + d;
    };
    auto merged_index = [](std::int64_t b, std::int64_t t, std::int64_t h, std::int64_t d) {
        return ((b * kSeqLen + t) * kHeads + h) * kHeadDim + d;
    };

    bool passed = false;
    float forward_max_error = 0.0f;
    float lse_finite_error = 0.0f;
    float grad_qkv_nonzero_sum = 0.0f;
    float grad_qkv_max_abs = 0.0f;

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
        error = "LLaMA packed-attention smoke commands are only valid for LLaMA-family native preflights";
    } else {
        tile_handle = dlopen(tile_ops_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            const char* raw = dlerror();
            error = raw == nullptr ? "failed to load Tile ops library" : raw;
        } else {
            tile_ops_loaded = true;
            float32_to_bf16 = load_symbol<Float32ToBf16Fn>(tile_handle, "nfn_native_tile_float32_to_bf16_bits");
            bf16_to_float32 = load_symbol<Bf16ToFloat32Fn>(tile_handle, "nfn_native_tile_bf16_bits_to_float32");
            packed_attention = load_symbol<PackedAttentionStoreLseFn>(
                tile_handle, "nfn_native_tile_scaled_dot_product_attention_packed_qkv_store_lse_bf16_float32");
            packed_attention_backward = load_symbol<PackedAttentionBackwardFn>(
                tile_handle, "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_saved_lse_bf16_from_merged_grad_float32");
            if (float32_to_bf16 == nullptr || bf16_to_float32 == nullptr ||
                packed_attention == nullptr || packed_attention_backward == nullptr) {
                error = "Tile ops library is missing one or more LLaMA packed-attention symbols";
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
        std::vector<float> qkv(static_cast<std::size_t>(kPackedElements), 0.0f);
        std::vector<float> grad_out(static_cast<std::size_t>(kElements), 0.0f);
        for (std::int64_t t = 0; t < kSeqLen; ++t) {
            for (std::int64_t d = 0; d < kHeadDim; ++d) {
                qkv[static_cast<std::size_t>(packed_index(0, t, 0, 0, d))] =
                    0.01f * static_cast<float>((d % 11) + 1) + 0.02f * static_cast<float>(t);
                qkv[static_cast<std::size_t>(packed_index(0, t, 1, 0, d))] =
                    0.015f * static_cast<float>((d % 7) + 1) - 0.01f * static_cast<float>(t);
                qkv[static_cast<std::size_t>(packed_index(0, t, 2, 0, d))] =
                    0.02f * static_cast<float>((d % 5) + 1) + 0.03f * static_cast<float>(t);
                grad_out[static_cast<std::size_t>(merged_index(0, t, 0, d))] =
                    0.004f * static_cast<float>((d % 13) + 1) + 0.01f * static_cast<float>(t + 1);
            }
        }

        float* d_qkv_float = nullptr;
        std::uint16_t* d_qkv_bf16 = nullptr;
        std::uint16_t* d_out_bf16 = nullptr;
        float* d_out_float = nullptr;
        float* d_saved_lse = nullptr;
        float* d_grad_out = nullptr;
        float* d_grad_qkv = nullptr;
        auto alloc_float = [&](float** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(float));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        auto alloc_u16 = [&](std::uint16_t** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(std::uint16_t));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        if (alloc_float(&d_qkv_float, qkv.size(), "qkv_float") &&
            alloc_u16(&d_qkv_bf16, qkv.size(), "qkv_bf16") &&
            alloc_u16(&d_out_bf16, grad_out.size(), "out_bf16") &&
            alloc_float(&d_out_float, grad_out.size(), "out_float") &&
            alloc_float(&d_saved_lse, static_cast<std::size_t>(kBatch * kHeads * kSeqLen), "saved_lse") &&
            alloc_float(&d_grad_out, grad_out.size(), "grad_out") &&
            alloc_float(&d_grad_qkv, qkv.size(), "grad_qkv")) {
            int status = cuda_memcpy(d_qkv_float, qkv.data(), qkv.size() * sizeof(float), kCudaMemcpyHostToDevice);
            if (status != 0) {
                error = cuda_error(status, "cudaMemcpy qkv H2D");
            }
            if (error.empty()) {
                status = cuda_memcpy(d_grad_out, grad_out.data(), grad_out.size() * sizeof(float), kCudaMemcpyHostToDevice);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy grad_out H2D");
                }
            }
            if (error.empty()) {
                status = float32_to_bf16(d_qkv_float, d_qkv_bf16, kPackedElements, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_float32_to_bf16_bits qkv");
                }
            }
            if (error.empty()) {
                status = packed_attention(
                    d_qkv_bf16,
                    d_out_bf16,
                    d_saved_lse,
                    kBatch,
                    kHeads,
                    kHeads,
                    kSeqLen,
                    kSeqLen,
                    kHeadDim,
                    kHeadDim,
                    kScale,
                    true,
                    false,
                    false,
                    0,
                    0,
                    0,
                    0,
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_scaled_dot_product_attention_packed_qkv_store_lse_bf16_float32");
                }
            }
            if (error.empty()) {
                status = bf16_to_float32(d_out_bf16, d_out_float, kElements, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_bf16_bits_to_float32 out");
                }
            }
            if (error.empty()) {
                status = packed_attention_backward(
                    d_qkv_bf16,
                    d_out_bf16,
                    d_saved_lse,
                    d_grad_out,
                    d_grad_qkv,
                    kBatch,
                    kHeads,
                    kHeads,
                    kSeqLen,
                    kSeqLen,
                    kHeadDim,
                    kHeadDim,
                    kScale,
                    true,
                    false,
                    false,
                    0,
                    0,
                    0,
                    0,
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_saved_lse_bf16_from_merged_grad_float32");
                }
            }
            if (error.empty()) {
                status = cuda_device_synchronize();
                if (status != 0) {
                    error = cuda_error(status, "cudaDeviceSynchronize");
                }
            }
            std::vector<float> actual_out(grad_out.size(), 0.0f);
            std::vector<float> actual_lse(static_cast<std::size_t>(kBatch * kHeads * kSeqLen), 0.0f);
            std::vector<float> actual_grad_qkv(qkv.size(), 0.0f);
            if (error.empty()) {
                status = cuda_memcpy(actual_out.data(), d_out_float, actual_out.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy out D2H");
                }
            }
            if (error.empty()) {
                status = cuda_memcpy(actual_lse.data(), d_saved_lse, actual_lse.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy lse D2H");
                }
            }
            if (error.empty()) {
                status = cuda_memcpy(actual_grad_qkv.data(), d_grad_qkv, actual_grad_qkv.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy grad_qkv D2H");
                }
            }
            if (error.empty()) {
                std::vector<float> expected_out(grad_out.size(), 0.0f);
                for (std::int64_t t = 0; t < kSeqLen; ++t) {
                    std::vector<float> scores(static_cast<std::size_t>(t + 1), 0.0f);
                    float max_score = -std::numeric_limits<float>::infinity();
                    for (std::int64_t s = 0; s <= t; ++s) {
                        float dot = 0.0f;
                        for (std::int64_t d = 0; d < kHeadDim; ++d) {
                            dot += qkv[static_cast<std::size_t>(packed_index(0, t, 0, 0, d))] *
                                   qkv[static_cast<std::size_t>(packed_index(0, s, 1, 0, d))];
                        }
                        scores[static_cast<std::size_t>(s)] = dot * kScale;
                        max_score = std::max(max_score, scores[static_cast<std::size_t>(s)]);
                    }
                    float denom = 0.0f;
                    for (float score : scores) {
                        denom += std::exp(score - max_score);
                    }
                    for (std::int64_t s = 0; s <= t; ++s) {
                        const float prob = std::exp(scores[static_cast<std::size_t>(s)] - max_score) / denom;
                        for (std::int64_t d = 0; d < kHeadDim; ++d) {
                            expected_out[static_cast<std::size_t>(merged_index(0, t, 0, d))] +=
                                prob * qkv[static_cast<std::size_t>(packed_index(0, s, 2, 0, d))];
                        }
                    }
                }
                forward_max_error = max_abs_error(actual_out, expected_out);
                for (float value : actual_lse) {
                    if (!std::isfinite(value)) {
                        lse_finite_error = 1.0f;
                    }
                }
                for (float value : actual_grad_qkv) {
                    if (!std::isfinite(value)) {
                        grad_qkv_max_abs = std::numeric_limits<float>::infinity();
                        break;
                    }
                    grad_qkv_nonzero_sum += std::fabs(value);
                    grad_qkv_max_abs = std::max(grad_qkv_max_abs, std::fabs(value));
                }
                passed =
                    forward_max_error <= 6e-3f &&
                    lse_finite_error == 0.0f &&
                    grad_qkv_nonzero_sum > 1e-5f &&
                    std::isfinite(grad_qkv_max_abs);
                if (!passed) {
                    error = "LLaMA packed-attention smoke exceeded tolerance";
                }
            }
        }
    }
    free_allocated();
    close_handles();

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(NFN_NATIVE_MODEL_FAMILY) << "\",\n"
        << "  \"native_target\": \"" << json_escape(NFN_NATIVE_TARGET_NAME) << "\",\n"
        << "  \"smoke\": \"llama_packed_qkv_attention_step_slice\",\n"
        << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(error) << "\",\n"
        << "  \"compiled_native_boundary\": true,\n"
        << "  \"torch_required\": false,\n"
        << "  \"graph_editor_tensor_flow\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_ops_lib) << "\",\n"
        << "  \"tile_ops_loaded\": " << (tile_ops_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"shape\": {\"batch\": " << kBatch << ", \"heads\": " << kHeads
        << ", \"seq_len\": " << kSeqLen << ", \"head_dim\": " << kHeadDim << "},\n"
        << "  \"loop_composition_stages\": [\n"
        << "    \"nfn_native_tile_float32_to_bf16_bits\",\n"
        << "    \"nfn_native_tile_scaled_dot_product_attention_packed_qkv_store_lse_bf16_float32\",\n"
        << "    \"nfn_native_tile_bf16_bits_to_float32\",\n"
        << "    \"nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_saved_lse_bf16_from_merged_grad_float32\"\n"
        << "  ],\n"
        << "  \"max_errors\": {"
        << "\"packed_attention_forward\":" << forward_max_error
        << ", \"saved_lse_finite\":" << lse_finite_error
        << ", \"grad_qkv_nonzero_sum\":" << grad_qkv_nonzero_sum
        << ", \"grad_qkv_max_abs\":" << grad_qkv_max_abs
        << "}\n"
        << "}\n";
    return passed ? 0 : 2;
}

int print_llama_attention_block_smoke_json(const Config& cfg, const char* program) {
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
    using LinearBf16OutputFn = int (*)(
        const float*,
        const float*,
        const float*,
        std::uint16_t*,
        std::int64_t,
        std::int64_t,
        std::int64_t,
        bool,
        void*);
    using PackedAttentionFn = int (*)(
        const std::uint16_t*,
        std::uint16_t*,
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
    using Bf16ToFloat32Fn = int (*)(const std::uint16_t*, float*, std::int64_t, void*);
    using ScaledResidualAddFn = int (*)(const float*, const float*, const float*, float*, std::int64_t, void*);

    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;
    RmsNormFn rms_norm = nullptr;
    LinearBf16OutputFn linear_bf16_output = nullptr;
    PackedAttentionFn packed_attention = nullptr;
    Bf16ToFloat32Fn bf16_to_float32 = nullptr;
    ScaledResidualAddFn residual_add = nullptr;

    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    constexpr std::int64_t kBatch = 1;
    constexpr std::int64_t kHeads = 12;
    constexpr std::int64_t kSeqLen = 32;
    constexpr std::int64_t kHeadDim = 64;
    constexpr std::int64_t kDim = kHeads * kHeadDim;
    constexpr std::int64_t kRows = kBatch * kSeqLen;
    constexpr std::int64_t kQkvDim = 3 * kDim;
    constexpr std::int64_t kElements = kRows * kDim;
    constexpr std::int64_t kQkvElements = kRows * kQkvDim;
    constexpr float kEps = 1e-6f;
    constexpr float kScale = 1.0f / 8.0f;

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
    auto max_abs = [](const std::vector<float>& values) {
        float out = 0.0f;
        for (float value : values) {
            out = std::max(out, std::fabs(value));
        }
        return out;
    };

    bool passed = false;
    float qkv_max_abs = 0.0f;
    float attention_max_abs = 0.0f;
    float residual_delta_max_abs = 0.0f;
    float residual_scale_value = 1.0f;

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
        error = "LLaMA attention-block smoke commands are only valid for LLaMA-family native preflights";
    } else {
        tile_handle = dlopen(tile_ops_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            const char* raw = dlerror();
            error = raw == nullptr ? "failed to load Tile ops library" : raw;
        } else {
            tile_ops_loaded = true;
            rms_norm = load_symbol<RmsNormFn>(tile_handle, "nfn_native_tile_rms_norm_float32");
            linear_bf16_output = load_symbol<LinearBf16OutputFn>(
                tile_handle, "nfn_native_tile_linear_bf16_output_float32");
            packed_attention = load_symbol<PackedAttentionFn>(
                tile_handle, "nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32");
            bf16_to_float32 = load_symbol<Bf16ToFloat32Fn>(tile_handle, "nfn_native_tile_bf16_bits_to_float32");
            residual_add = load_symbol<ScaledResidualAddFn>(tile_handle, "nfn_native_tile_scaled_residual_add_float32");
            if (rms_norm == nullptr || linear_bf16_output == nullptr || packed_attention == nullptr ||
                bf16_to_float32 == nullptr || residual_add == nullptr) {
                error = "Tile ops library is missing one or more LLaMA attention-block symbols";
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
        std::vector<float> hidden(static_cast<std::size_t>(kElements), 0.0f);
        std::vector<float> qkv_weight(static_cast<std::size_t>(kDim * kQkvDim), 0.0f);
        std::vector<float> residual_scale = {residual_scale_value};
        for (std::int64_t row = 0; row < kRows; ++row) {
            for (std::int64_t dim = 0; dim < kDim; ++dim) {
                hidden[static_cast<std::size_t>(row * kDim + dim)] =
                    0.001f * static_cast<float>((row % 17) + 1) +
                    0.0002f * static_cast<float>((dim % 31) + 1);
            }
        }
        for (std::int64_t out_dim = 0; out_dim < kQkvDim; ++out_dim) {
            const std::int64_t in_dim = out_dim % kDim;
            qkv_weight[static_cast<std::size_t>(out_dim * kDim + in_dim)] =
                0.08f + 0.001f * static_cast<float>(out_dim % 13);
        }

        float* d_hidden = nullptr;
        float* d_normed = nullptr;
        float* d_attention_float = nullptr;
        float* d_residual_out = nullptr;
        float* d_qkv_weight = nullptr;
        float* d_residual_scale = nullptr;
        std::uint16_t* d_qkv_bf16 = nullptr;
        std::uint16_t* d_attention_bf16 = nullptr;
        auto alloc_float = [&](float** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(float));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        auto alloc_bf16 = [&](std::uint16_t** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(std::uint16_t));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        if (alloc_float(&d_hidden, hidden.size(), "hidden") &&
            alloc_float(&d_normed, hidden.size(), "normed") &&
            alloc_float(&d_attention_float, hidden.size(), "attention_float") &&
            alloc_float(&d_residual_out, hidden.size(), "residual_out") &&
            alloc_float(&d_qkv_weight, qkv_weight.size(), "qkv_weight") &&
            alloc_float(&d_residual_scale, residual_scale.size(), "residual_scale") &&
            alloc_bf16(&d_qkv_bf16, static_cast<std::size_t>(kQkvElements), "qkv_bf16") &&
            alloc_bf16(&d_attention_bf16, hidden.size(), "attention_bf16")) {
            int status = cuda_memcpy(d_hidden, hidden.data(), hidden.size() * sizeof(float), kCudaMemcpyHostToDevice);
            if (status != 0) {
                error = cuda_error(status, "cudaMemcpy hidden H2D");
            }
            if (error.empty()) {
                status = cuda_memcpy(
                    d_qkv_weight, qkv_weight.data(), qkv_weight.size() * sizeof(float), kCudaMemcpyHostToDevice);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy qkv_weight H2D");
                }
            }
            if (error.empty()) {
                status = cuda_memcpy(
                    d_residual_scale, residual_scale.data(), residual_scale.size() * sizeof(float), kCudaMemcpyHostToDevice);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy residual_scale H2D");
                }
            }
            if (error.empty()) {
                status = rms_norm(d_hidden, d_normed, kRows, kDim, kEps, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_rms_norm_float32");
                }
            }
            if (error.empty()) {
                status = linear_bf16_output(
                    d_normed, d_qkv_weight, nullptr, d_qkv_bf16, kRows, kDim, kQkvDim, false, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_bf16_output_float32");
                }
            }
            if (error.empty()) {
                status = packed_attention(
                    d_qkv_bf16,
                    d_attention_bf16,
                    kBatch,
                    kHeads,
                    kHeads,
                    kSeqLen,
                    kSeqLen,
                    kHeadDim,
                    kHeadDim,
                    kScale,
                    true,
                    false,
                    false,
                    0,
                    0,
                    0,
                    0,
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32");
                }
            }
            if (error.empty()) {
                status = bf16_to_float32(d_attention_bf16, d_attention_float, kElements, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_bf16_bits_to_float32");
                }
            }
            if (error.empty()) {
                status = residual_add(d_hidden, d_attention_float, d_residual_scale, d_residual_out, kElements, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_scaled_residual_add_float32");
                }
            }
            if (error.empty()) {
                status = cuda_device_synchronize();
                if (status != 0) {
                    error = cuda_error(status, "cudaDeviceSynchronize attention block smoke");
                }
            }
            if (error.empty()) {
                std::vector<std::uint16_t> qkv_actual(static_cast<std::size_t>(kQkvElements), 0);
                std::vector<float> attention_actual(hidden.size(), 0.0f);
                std::vector<float> residual_actual(hidden.size(), 0.0f);
                status = cuda_memcpy(
                    qkv_actual.data(), d_qkv_bf16, qkv_actual.size() * sizeof(std::uint16_t), kCudaMemcpyDeviceToHost);
                if (status == 0) {
                    status = cuda_memcpy(
                        attention_actual.data(), d_attention_float, attention_actual.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                }
                if (status == 0) {
                    status = cuda_memcpy(
                        residual_actual.data(), d_residual_out, residual_actual.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                }
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy attention block D2H");
                } else {
                    for (std::uint16_t value : qkv_actual) {
                        qkv_max_abs = std::max(qkv_max_abs, value == 0 ? 0.0f : 1.0f);
                    }
                    attention_max_abs = max_abs(attention_actual);
                    for (std::size_t i = 0; i < residual_actual.size(); ++i) {
                        residual_delta_max_abs = std::max(
                            residual_delta_max_abs,
                            std::fabs(residual_actual[i] - hidden[i]));
                    }
                }
            }
            passed = error.empty() &&
                qkv_max_abs > 0.0f &&
                attention_max_abs > 0.0f &&
                residual_delta_max_abs > 0.0f;
            if (!passed && error.empty()) {
                error = "LLaMA attention-block smoke produced a degenerate output";
            }
        }
    }
    free_allocated();
    close_handles();

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(NFN_NATIVE_MODEL_FAMILY) << "\",\n"
        << "  \"native_target\": \"" << json_escape(NFN_NATIVE_TARGET_NAME) << "\",\n"
        << "  \"smoke\": \"llama_packed_qkv_attention_block_forward_slice\",\n"
        << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(error) << "\",\n"
        << "  \"compiled_native_boundary\": true,\n"
        << "  \"torch_required\": false,\n"
        << "  \"graph_editor_tensor_flow\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_ops_lib) << "\",\n"
        << "  \"tile_ops_loaded\": " << (tile_ops_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"shape\": {\"batch\": " << kBatch << ", \"heads\": " << kHeads
        << ", \"seq_len\": " << kSeqLen << ", \"head_dim\": " << kHeadDim
        << ", \"model_dim\": " << kDim << "},\n"
        << "  \"loop_composition_stages\": [\n"
        << "    \"nfn_native_tile_rms_norm_float32\",\n"
        << "    \"nfn_native_tile_linear_bf16_output_float32\",\n"
        << "    \"nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32\",\n"
        << "    \"nfn_native_tile_bf16_bits_to_float32\",\n"
        << "    \"nfn_native_tile_scaled_residual_add_float32\"\n"
        << "  ],\n"
        << "  \"max_errors\": {"
        << "\"qkv_nonzero\":" << qkv_max_abs
        << ", \"attention_max_abs\":" << attention_max_abs
        << ", \"residual_delta_max_abs\":" << residual_delta_max_abs
        << "}\n"
        << "}\n";
    return passed ? 0 : 2;
}

int print_moe_transformer_block_smoke_json(const Config& cfg, const char* program) {
    const std::string family = NFN_NATIVE_MODEL_FAMILY;
    const bool moe_family =
        family.find("moe") != std::string::npos ||
        family.find("mixllama") != std::string::npos ||
        family.find("deepseek") != std::string::npos ||
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
    using LinearBf16OutputFn = int (*)(
        const float*, const float*, const float*, std::uint16_t*,
        std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using PackedAttentionFn = int (*)(
        const std::uint16_t*, std::uint16_t*, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        float, bool, bool, bool, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t, void*);
    using Bf16ToFloat32Fn = int (*)(const std::uint16_t*, float*, std::int64_t, void*);
    using ScaledResidualAddFn = int (*)(const float*, const float*, const float*, float*, std::int64_t, void*);
    using TopKRouteFn = int (*)(const float*, float*, std::int64_t*, std::int64_t, std::int64_t, std::int64_t, void*);
    using MoeSwiGluForwardFn = int (*)(
        const float*, const float*, const std::int64_t*, const float*, const float*, const float*, float*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);

    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;
    RmsNormFn rms_norm = nullptr;
    LinearBf16OutputFn linear_bf16_output = nullptr;
    PackedAttentionFn packed_attention = nullptr;
    Bf16ToFloat32Fn bf16_to_float32 = nullptr;
    ScaledResidualAddFn residual_add = nullptr;
    TopKRouteFn topk_route = nullptr;
    MoeSwiGluForwardFn moe_forward = nullptr;

    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    constexpr std::int64_t kBatch = 1;
    constexpr std::int64_t kHeads = 12;
    constexpr std::int64_t kSeqLen = 32;
    constexpr std::int64_t kHeadDim = 64;
    constexpr std::int64_t kDim = kHeads * kHeadDim;
    constexpr std::int64_t kRows = kBatch * kSeqLen;
    constexpr std::int64_t kQkvDim = 3 * kDim;
    constexpr std::int64_t kElements = kRows * kDim;
    constexpr std::int64_t kQkvElements = kRows * kQkvDim;
    constexpr std::int64_t kExperts = 4;
    constexpr std::int64_t kTopK = 2;
    constexpr std::int64_t kMoeHidden = 4;
    constexpr std::int64_t kRouteElements = kRows * kTopK;
    constexpr std::int64_t kW13Elements = kExperts * kDim * kMoeHidden;
    constexpr std::int64_t kW2Elements = kExperts * kMoeHidden * kDim;
    constexpr float kEps = 1e-6f;
    constexpr float kScale = 1.0f / 8.0f;

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
    auto max_abs = [](const std::vector<float>& values) {
        float out = 0.0f;
        for (float value : values) {
            out = std::max(out, std::fabs(value));
        }
        return out;
    };

    bool passed = false;
    float attention_max_abs = 0.0f;
    float attn_residual_delta_max_abs = 0.0f;
    float route_weight_max_abs = 0.0f;
    float moe_max_abs = 0.0f;
    float block_residual_delta_max_abs = 0.0f;
    std::int64_t route_index_max = -1;
    const float residual_scale_value = 1.0f;

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

    if (!moe_family) {
        error = "MoE transformer-block smoke commands are only valid for MoE-family native preflights";
    } else {
        tile_handle = dlopen(tile_ops_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            const char* raw = dlerror();
            error = raw == nullptr ? "failed to load Tile ops library" : raw;
        } else {
            tile_ops_loaded = true;
            rms_norm = load_symbol<RmsNormFn>(tile_handle, "nfn_native_tile_rms_norm_float32");
            linear_bf16_output = load_symbol<LinearBf16OutputFn>(
                tile_handle, "nfn_native_tile_linear_bf16_output_float32");
            packed_attention = load_symbol<PackedAttentionFn>(
                tile_handle, "nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32");
            bf16_to_float32 = load_symbol<Bf16ToFloat32Fn>(tile_handle, "nfn_native_tile_bf16_bits_to_float32");
            residual_add = load_symbol<ScaledResidualAddFn>(tile_handle, "nfn_native_tile_scaled_residual_add_float32");
            topk_route = load_symbol<TopKRouteFn>(tile_handle, "nfn_native_tile_topk_route_float32");
            moe_forward = load_symbol<MoeSwiGluForwardFn>(
                tile_handle, "nfn_native_tile_moe_swiglu_forward_float32");
            if (rms_norm == nullptr || linear_bf16_output == nullptr || packed_attention == nullptr ||
                bf16_to_float32 == nullptr || residual_add == nullptr || topk_route == nullptr ||
                moe_forward == nullptr) {
                error = "Tile ops library is missing one or more MoE transformer-block symbols";
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
        std::vector<float> hidden(static_cast<std::size_t>(kElements), 0.0f);
        std::vector<float> qkv_weight(static_cast<std::size_t>(kDim * kQkvDim), 0.0f);
        std::vector<float> route_logits(static_cast<std::size_t>(kRows * kExperts), 0.0f);
        std::vector<float> w1(static_cast<std::size_t>(kW13Elements), 0.0f);
        std::vector<float> w2(static_cast<std::size_t>(kW2Elements), 0.0f);
        std::vector<float> w3(static_cast<std::size_t>(kW13Elements), 0.0f);
        std::vector<float> residual_scale = {residual_scale_value};
        for (std::int64_t row = 0; row < kRows; ++row) {
            for (std::int64_t dim = 0; dim < kDim; ++dim) {
                hidden[static_cast<std::size_t>(row * kDim + dim)] =
                    0.001f * static_cast<float>((row % 19) + 1) +
                    0.0001f * static_cast<float>((dim % 37) + 1);
            }
            for (std::int64_t expert = 0; expert < kExperts; ++expert) {
                route_logits[static_cast<std::size_t>(row * kExperts + expert)] =
                    0.02f * static_cast<float>((row + expert * 3) % 11) -
                    0.01f * static_cast<float>(expert);
            }
        }
        for (std::int64_t out_dim = 0; out_dim < kQkvDim; ++out_dim) {
            const std::int64_t in_dim = out_dim % kDim;
            qkv_weight[static_cast<std::size_t>(out_dim * kDim + in_dim)] =
                0.07f + 0.001f * static_cast<float>(out_dim % 17);
        }
        for (std::int64_t expert = 0; expert < kExperts; ++expert) {
            for (std::int64_t dim = 0; dim < kDim; ++dim) {
                for (std::int64_t hidden_dim = 0; hidden_dim < kMoeHidden; ++hidden_dim) {
                    const std::size_t idx = static_cast<std::size_t>((expert * kDim + dim) * kMoeHidden + hidden_dim);
                    w1[idx] = 0.0005f * static_cast<float>((dim + hidden_dim + expert) % 13 + 1);
                    w3[idx] = -0.0004f * static_cast<float>((dim * 2 + hidden_dim + expert) % 11 + 1);
                }
            }
            for (std::int64_t hidden_dim = 0; hidden_dim < kMoeHidden; ++hidden_dim) {
                for (std::int64_t dim = 0; dim < kDim; ++dim) {
                    const std::size_t idx = static_cast<std::size_t>((expert * kMoeHidden + hidden_dim) * kDim + dim);
                    w2[idx] = 0.0006f * static_cast<float>((dim + hidden_dim * 3 + expert) % 7 + 1);
                }
            }
        }

        float* d_hidden = nullptr;
        float* d_normed = nullptr;
        float* d_attention_float = nullptr;
        float* d_attn_residual = nullptr;
        float* d_moe_out = nullptr;
        float* d_block_out = nullptr;
        float* d_qkv_weight = nullptr;
        float* d_route_logits = nullptr;
        float* d_route_weights = nullptr;
        float* d_w1 = nullptr;
        float* d_w2 = nullptr;
        float* d_w3 = nullptr;
        float* d_residual_scale = nullptr;
        std::uint16_t* d_qkv_bf16 = nullptr;
        std::uint16_t* d_attention_bf16 = nullptr;
        std::int64_t* d_route_indices = nullptr;
        auto alloc_float = [&](float** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(float));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        auto alloc_bf16 = [&](std::uint16_t** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(std::uint16_t));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        auto alloc_i64 = [&](std::int64_t** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(std::int64_t));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        if (alloc_float(&d_hidden, hidden.size(), "hidden") &&
            alloc_float(&d_normed, hidden.size(), "normed") &&
            alloc_float(&d_attention_float, hidden.size(), "attention_float") &&
            alloc_float(&d_attn_residual, hidden.size(), "attn_residual") &&
            alloc_float(&d_moe_out, hidden.size(), "moe_out") &&
            alloc_float(&d_block_out, hidden.size(), "block_out") &&
            alloc_float(&d_qkv_weight, qkv_weight.size(), "qkv_weight") &&
            alloc_float(&d_route_logits, route_logits.size(), "route_logits") &&
            alloc_float(&d_route_weights, static_cast<std::size_t>(kRouteElements), "route_weights") &&
            alloc_float(&d_w1, w1.size(), "w1") &&
            alloc_float(&d_w2, w2.size(), "w2") &&
            alloc_float(&d_w3, w3.size(), "w3") &&
            alloc_float(&d_residual_scale, residual_scale.size(), "residual_scale") &&
            alloc_bf16(&d_qkv_bf16, static_cast<std::size_t>(kQkvElements), "qkv_bf16") &&
            alloc_bf16(&d_attention_bf16, hidden.size(), "attention_bf16") &&
            alloc_i64(&d_route_indices, static_cast<std::size_t>(kRouteElements), "route_indices")) {
            int status = cuda_memcpy(d_hidden, hidden.data(), hidden.size() * sizeof(float), kCudaMemcpyHostToDevice);
            if (status == 0) {
                status = cuda_memcpy(d_qkv_weight, qkv_weight.data(), qkv_weight.size() * sizeof(float), kCudaMemcpyHostToDevice);
            }
            if (status == 0) {
                status = cuda_memcpy(d_route_logits, route_logits.data(), route_logits.size() * sizeof(float), kCudaMemcpyHostToDevice);
            }
            if (status == 0) {
                status = cuda_memcpy(d_w1, w1.data(), w1.size() * sizeof(float), kCudaMemcpyHostToDevice);
            }
            if (status == 0) {
                status = cuda_memcpy(d_w2, w2.data(), w2.size() * sizeof(float), kCudaMemcpyHostToDevice);
            }
            if (status == 0) {
                status = cuda_memcpy(d_w3, w3.data(), w3.size() * sizeof(float), kCudaMemcpyHostToDevice);
            }
            if (status == 0) {
                status = cuda_memcpy(d_residual_scale, residual_scale.data(), sizeof(float), kCudaMemcpyHostToDevice);
            }
            if (status != 0) {
                error = cuda_error(status, "cudaMemcpy MoE transformer block H2D");
            }
            if (error.empty()) {
                status = rms_norm(d_hidden, d_normed, kRows, kDim, kEps, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_rms_norm_float32");
                }
            }
            if (error.empty()) {
                status = linear_bf16_output(d_normed, d_qkv_weight, nullptr, d_qkv_bf16, kRows, kDim, kQkvDim, false, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_bf16_output_float32");
                }
            }
            if (error.empty()) {
                status = packed_attention(
                    d_qkv_bf16, d_attention_bf16, kBatch, kHeads, kHeads, kSeqLen, kSeqLen,
                    kHeadDim, kHeadDim, kScale, true, false, false, 0, 0, 0, 0, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32");
                }
            }
            if (error.empty()) {
                status = bf16_to_float32(d_attention_bf16, d_attention_float, kElements, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_bf16_bits_to_float32");
                }
            }
            if (error.empty()) {
                status = residual_add(d_hidden, d_attention_float, d_residual_scale, d_attn_residual, kElements, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_scaled_residual_add_float32 attention");
                }
            }
            if (error.empty()) {
                status = topk_route(d_route_logits, d_route_weights, d_route_indices, kRows, kExperts, kTopK, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_topk_route_float32");
                }
            }
            if (error.empty()) {
                status = moe_forward(
                    d_attn_residual, d_route_weights, d_route_indices, d_w1, d_w2, d_w3, d_moe_out,
                    kRows, kDim, kMoeHidden, kExperts, kTopK, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_moe_swiglu_forward_float32");
                }
            }
            if (error.empty()) {
                status = residual_add(d_attn_residual, d_moe_out, d_residual_scale, d_block_out, kElements, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_scaled_residual_add_float32 moe");
                }
            }
            if (error.empty()) {
                status = cuda_device_synchronize();
                if (status != 0) {
                    error = cuda_error(status, "cudaDeviceSynchronize MoE transformer block smoke");
                }
            }
            if (error.empty()) {
                std::vector<float> attention_actual(hidden.size(), 0.0f);
                std::vector<float> attn_residual_actual(hidden.size(), 0.0f);
                std::vector<float> route_weight_actual(static_cast<std::size_t>(kRouteElements), 0.0f);
                std::vector<std::int64_t> route_index_actual(static_cast<std::size_t>(kRouteElements), -1);
                std::vector<float> moe_actual(hidden.size(), 0.0f);
                std::vector<float> block_actual(hidden.size(), 0.0f);
                status = cuda_memcpy(attention_actual.data(), d_attention_float, attention_actual.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (status == 0) {
                    status = cuda_memcpy(attn_residual_actual.data(), d_attn_residual, attn_residual_actual.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                }
                if (status == 0) {
                    status = cuda_memcpy(route_weight_actual.data(), d_route_weights, route_weight_actual.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                }
                if (status == 0) {
                    status = cuda_memcpy(route_index_actual.data(), d_route_indices, route_index_actual.size() * sizeof(std::int64_t), kCudaMemcpyDeviceToHost);
                }
                if (status == 0) {
                    status = cuda_memcpy(moe_actual.data(), d_moe_out, moe_actual.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                }
                if (status == 0) {
                    status = cuda_memcpy(block_actual.data(), d_block_out, block_actual.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                }
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy MoE transformer block D2H");
                } else {
                    attention_max_abs = max_abs(attention_actual);
                    route_weight_max_abs = max_abs(route_weight_actual);
                    moe_max_abs = max_abs(moe_actual);
                    for (std::int64_t index : route_index_actual) {
                        route_index_max = std::max(route_index_max, index);
                    }
                    for (std::size_t i = 0; i < hidden.size(); ++i) {
                        attn_residual_delta_max_abs = std::max(
                            attn_residual_delta_max_abs,
                            std::fabs(attn_residual_actual[i] - hidden[i]));
                        block_residual_delta_max_abs = std::max(
                            block_residual_delta_max_abs,
                            std::fabs(block_actual[i] - attn_residual_actual[i]));
                    }
                }
            }
            passed = error.empty() &&
                attention_max_abs > 0.0f &&
                attn_residual_delta_max_abs > 0.0f &&
                route_weight_max_abs > 0.0f &&
                route_index_max >= 0 &&
                moe_max_abs > 0.0f &&
                block_residual_delta_max_abs > 0.0f;
            if (!passed && error.empty()) {
                error = "MoE transformer-block smoke produced a degenerate output";
            }
        }
    }
    free_allocated();
    close_handles();

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(NFN_NATIVE_MODEL_FAMILY) << "\",\n"
        << "  \"native_target\": \"" << json_escape(NFN_NATIVE_TARGET_NAME) << "\",\n"
        << "  \"smoke\": \"moe_transformer_block_forward_slice\",\n"
        << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(error) << "\",\n"
        << "  \"compiled_native_boundary\": true,\n"
        << "  \"torch_required\": false,\n"
        << "  \"graph_editor_tensor_flow\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_ops_lib) << "\",\n"
        << "  \"tile_ops_loaded\": " << (tile_ops_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"shape\": {\"batch\": " << kBatch << ", \"seq_len\": " << kSeqLen
        << ", \"model_dim\": " << kDim << ", \"heads\": " << kHeads
        << ", \"head_dim\": " << kHeadDim << ", \"experts\": " << kExperts
        << ", \"top_k\": " << kTopK << ", \"moe_hidden_dim\": " << kMoeHidden << "},\n"
        << "  \"loop_composition_stages\": [\n"
        << "    \"nfn_native_tile_rms_norm_float32\",\n"
        << "    \"nfn_native_tile_linear_bf16_output_float32\",\n"
        << "    \"nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32\",\n"
        << "    \"nfn_native_tile_bf16_bits_to_float32\",\n"
        << "    \"nfn_native_tile_scaled_residual_add_float32\",\n"
        << "    \"nfn_native_tile_topk_route_float32\",\n"
        << "    \"nfn_native_tile_moe_swiglu_forward_float32\"\n"
        << "  ],\n"
        << "  \"max_errors\": {"
        << "\"attention_max_abs\":" << attention_max_abs
        << ", \"attn_residual_delta_max_abs\":" << attn_residual_delta_max_abs
        << ", \"route_weight_max_abs\":" << route_weight_max_abs
        << ", \"route_index_max\":" << route_index_max
        << ", \"moe_max_abs\":" << moe_max_abs
        << ", \"block_residual_delta_max_abs\":" << block_residual_delta_max_abs
        << "}\n"
        << "}\n";
    return passed ? 0 : 2;
}

int print_moe_route_expert_smoke_json(const Config& cfg, const char* program) {
    const std::string family = NFN_NATIVE_MODEL_FAMILY;
    const bool moe_family =
        family.find("moe") != std::string::npos ||
        family.find("mixllama") != std::string::npos ||
        family.find("deepseek") != std::string::npos ||
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
    using TopKRouteFn = int (*)(const float*, float*, std::int64_t*, std::int64_t, std::int64_t, std::int64_t, void*);
    using BroadcastExpertRoutesFn = int (*)(
        const float*, const std::int64_t*, float*, std::int64_t*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using MoeSwiGluForwardFn = int (*)(
        const float*, const float*, const std::int64_t*, const float*, const float*, const float*, float*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using MoeSwiGluBackwardFn = int (*)(
        const float*, const float*, const std::int64_t*, const float*, const float*, const float*, const float*,
        float*, float*, float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using RouteBalanceDensityFn = int (*)(const float*, float*, std::int64_t, std::int64_t, void*);
    using RouteBalanceLossFn = int (*)(const float*, float*, std::int64_t, void*);
    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using AdamWFn = int (*)(
        float*, const float*, float*, float*, std::int64_t, float, float, float, float, float, float, float, void*);

    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;
    TopKRouteFn topk_route = nullptr;
    BroadcastExpertRoutesFn broadcast_routes = nullptr;
    MoeSwiGluForwardFn moe_forward = nullptr;
    MoeSwiGluBackwardFn moe_backward = nullptr;
    RouteBalanceDensityFn route_density = nullptr;
    RouteBalanceLossFn route_loss = nullptr;
    FillFn fill = nullptr;
    AdamWFn adamw = nullptr;

    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    constexpr std::int64_t kRouteRows = 1;
    constexpr std::int64_t kBatch = 1;
    constexpr std::int64_t kSeqLen = 2;
    constexpr std::int64_t kTokens = kBatch * kSeqLen;
    constexpr std::int64_t kDim = 3;
    constexpr std::int64_t kHidden = 2;
    constexpr std::int64_t kExperts = 3;
    constexpr std::int64_t kTopK = 2;
    constexpr std::int64_t kRouteElements = kRouteRows * kTopK;
    constexpr std::int64_t kBroadcastRouteElements = kTokens * kTopK;
    constexpr std::int64_t kTokenElements = kTokens * kDim;
    constexpr std::int64_t kW13Elements = kExperts * kDim * kHidden;
    constexpr std::int64_t kW2Elements = kExperts * kHidden * kDim;

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
    auto max_i64_error = [](const std::vector<std::int64_t>& actual, const std::vector<std::int64_t>& expected) {
        std::int64_t max_err = 0;
        const std::size_t n = std::min(actual.size(), expected.size());
        for (std::size_t i = 0; i < n; ++i) {
            const std::int64_t err = static_cast<std::int64_t>(std::llabs(actual[i] - expected[i]));
            max_err = std::max(max_err, err);
        }
        return max_err;
    };
    auto silu = [](float value) {
        const float sig = 1.0f / (1.0f + std::exp(-value));
        return value * sig;
    };
    auto dsilu = [](float value) {
        const float sig = 1.0f / (1.0f + std::exp(-value));
        return sig * (1.0f + value * (1.0f - sig));
    };

    bool passed = false;
    float topk_weight_max_error = 0.0f;
    std::int64_t topk_index_max_error = 0;
    float broadcast_weight_max_error = 0.0f;
    std::int64_t broadcast_index_max_error = 0;
    float moe_forward_max_error = 0.0f;
    float moe_grad_x_max_error = 0.0f;
    float moe_grad_w1_max_error = 0.0f;
    float moe_grad_w2_max_error = 0.0f;
    float moe_grad_w3_max_error = 0.0f;
    float balance_density_max_error = 0.0f;
    float balance_loss_max_error = 0.0f;
    float adamw_w1_max_error = 0.0f;

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

    if (!moe_family) {
        error = "MoE smoke commands are only valid for MoE-family native preflights";
    } else {
        tile_handle = dlopen(tile_ops_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            const char* raw = dlerror();
            error = raw == nullptr ? "failed to load Tile ops library" : raw;
        } else {
            tile_ops_loaded = true;
            topk_route = load_symbol<TopKRouteFn>(tile_handle, "nfn_native_tile_topk_route_float32");
            broadcast_routes = load_symbol<BroadcastExpertRoutesFn>(
                tile_handle, "nfn_native_tile_broadcast_expert_routes_float32");
            moe_forward = load_symbol<MoeSwiGluForwardFn>(
                tile_handle, "nfn_native_tile_moe_swiglu_forward_float32");
            moe_backward = load_symbol<MoeSwiGluBackwardFn>(
                tile_handle, "nfn_native_tile_moe_swiglu_backward_float32");
            route_density = load_symbol<RouteBalanceDensityFn>(
                tile_handle, "nfn_native_tile_route_balance_density_float32");
            route_loss = load_symbol<RouteBalanceLossFn>(
                tile_handle, "nfn_native_tile_route_balance_loss_float32");
            fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
            adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
            if (topk_route == nullptr || broadcast_routes == nullptr || moe_forward == nullptr ||
                moe_backward == nullptr || route_density == nullptr || route_loss == nullptr ||
                fill == nullptr || adamw == nullptr) {
                error = "Tile ops library is missing one or more MoE route/expert train-step symbols";
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
        const std::vector<float> route_logits = {0.1f, 0.5f, -0.2f};
        const std::vector<float> x = {
            0.2f, -0.1f, 0.3f,
            -0.4f, 0.5f, 0.1f,
        };
        const std::vector<float> grad_out = {
            0.1f, -0.2f, 0.05f,
            0.03f, 0.07f, -0.04f,
        };
        std::vector<float> w1(static_cast<std::size_t>(kW13Elements));
        std::vector<float> w2(static_cast<std::size_t>(kW2Elements));
        std::vector<float> w3(static_cast<std::size_t>(kW13Elements));
        for (std::size_t i = 0; i < w1.size(); ++i) {
            w1[i] = 0.02f * static_cast<float>((i % 7) + 1);
            w3[i] = -0.015f * static_cast<float>((i % 5) + 1);
        }
        for (std::size_t i = 0; i < w2.size(); ++i) {
            w2[i] = 0.01f * static_cast<float>(static_cast<int>(i % 11) - 5);
        }

        float* d_route_logits = nullptr;
        float* d_route_weights = nullptr;
        float* d_broadcast_weights = nullptr;
        float* d_x = nullptr;
        float* d_w1 = nullptr;
        float* d_w2 = nullptr;
        float* d_w3 = nullptr;
        float* d_out = nullptr;
        float* d_grad_out = nullptr;
        float* d_grad_x = nullptr;
        float* d_grad_w1 = nullptr;
        float* d_grad_w2 = nullptr;
        float* d_grad_w3 = nullptr;
        float* d_density = nullptr;
        float* d_balance_loss = nullptr;
        float* d_w1_exp_avg = nullptr;
        float* d_w1_exp_avg_sq = nullptr;
        std::int64_t* d_route_indices = nullptr;
        std::int64_t* d_broadcast_indices = nullptr;
        auto alloc = [&](float** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(float));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        auto alloc_i64 = [&](std::int64_t** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(std::int64_t));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        if (alloc(&d_route_logits, route_logits.size(), "route_logits") &&
            alloc(&d_route_weights, kRouteElements, "route_weights") &&
            alloc_i64(&d_route_indices, kRouteElements, "route_indices") &&
            alloc(&d_broadcast_weights, kBroadcastRouteElements, "broadcast_weights") &&
            alloc_i64(&d_broadcast_indices, kBroadcastRouteElements, "broadcast_indices") &&
            alloc(&d_x, x.size(), "x") &&
            alloc(&d_w1, w1.size(), "w1") &&
            alloc(&d_w2, w2.size(), "w2") &&
            alloc(&d_w3, w3.size(), "w3") &&
            alloc(&d_out, kTokenElements, "out") &&
            alloc(&d_grad_out, grad_out.size(), "grad_out") &&
            alloc(&d_grad_x, kTokenElements, "grad_x") &&
            alloc(&d_grad_w1, w1.size(), "grad_w1") &&
            alloc(&d_grad_w2, w2.size(), "grad_w2") &&
            alloc(&d_grad_w3, w3.size(), "grad_w3") &&
            alloc(&d_density, kExperts, "density") &&
            alloc(&d_balance_loss, 1, "balance_loss") &&
            alloc(&d_w1_exp_avg, w1.size(), "w1_exp_avg") &&
            alloc(&d_w1_exp_avg_sq, w1.size(), "w1_exp_avg_sq")) {
            auto copy_float = [&](float* dst, const std::vector<float>& src, const std::string& name) {
                int status = cuda_memcpy(dst, src.data(), src.size() * sizeof(float), kCudaMemcpyHostToDevice);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy " + name + " H2D");
                    return false;
                }
                return true;
            };
            if (copy_float(d_route_logits, route_logits, "route_logits") &&
                copy_float(d_x, x, "x") &&
                copy_float(d_w1, w1, "w1") &&
                copy_float(d_w2, w2, "w2") &&
                copy_float(d_w3, w3, "w3") &&
                copy_float(d_grad_out, grad_out, "grad_out")) {
                int status = fill(d_grad_x, kTokenElements, 0.0f, nullptr);
                if (status == 0) {
                    status = fill(d_grad_w1, kW13Elements, 0.0f, nullptr);
                }
                if (status == 0) {
                    status = fill(d_grad_w2, kW2Elements, 0.0f, nullptr);
                }
                if (status == 0) {
                    status = fill(d_grad_w3, kW13Elements, 0.0f, nullptr);
                }
                if (status == 0) {
                    status = fill(d_w1_exp_avg, kW13Elements, 0.0f, nullptr);
                }
                if (status == 0) {
                    status = fill(d_w1_exp_avg_sq, kW13Elements, 0.0f, nullptr);
                }
                if (status != 0) {
                    error = cuda_error(status, "zero MoE gradients/moments");
                }
            }
            int status = 0;
            if (error.empty()) {
                status = topk_route(d_route_logits, d_route_weights, d_route_indices, kRouteRows, kExperts, kTopK, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_topk_route_float32");
                }
            }
            if (error.empty()) {
                status = broadcast_routes(
                    d_route_weights,
                    d_route_indices,
                    d_broadcast_weights,
                    d_broadcast_indices,
                    kBatch,
                    kRouteRows,
                    kSeqLen,
                    kTopK,
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_broadcast_expert_routes_float32");
                }
            }
            if (error.empty()) {
                status = moe_forward(
                    d_x,
                    d_broadcast_weights,
                    d_broadcast_indices,
                    d_w1,
                    d_w2,
                    d_w3,
                    d_out,
                    kTokens,
                    kDim,
                    kHidden,
                    kExperts,
                    kTopK,
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_moe_swiglu_forward_float32");
                }
            }
            if (error.empty()) {
                status = moe_backward(
                    d_x,
                    d_broadcast_weights,
                    d_broadcast_indices,
                    d_w1,
                    d_w2,
                    d_w3,
                    d_grad_out,
                    d_grad_x,
                    d_grad_w1,
                    d_grad_w2,
                    d_grad_w3,
                    kTokens,
                    kDim,
                    kHidden,
                    kExperts,
                    kTopK,
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_moe_swiglu_backward_float32");
                }
            }
            if (error.empty()) {
                status = route_density(d_route_logits, d_density, kRouteRows, kExperts, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_route_balance_density_float32");
                }
            }
            if (error.empty()) {
                status = route_loss(d_density, d_balance_loss, kExperts, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_route_balance_loss_float32");
                }
            }
            if (error.empty()) {
                status = adamw(
                    d_w1,
                    d_grad_w1,
                    d_w1_exp_avg,
                    d_w1_exp_avg_sq,
                    kW13Elements,
                    0.01f,
                    0.9f,
                    0.95f,
                    1e-8f,
                    0.02f,
                    0.1f,
                    std::sqrt(0.05f),
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_adamw_step_float32 w1");
                }
            }
            if (error.empty()) {
                status = cuda_device_synchronize();
                if (status != 0) {
                    error = cuda_error(status, "cudaDeviceSynchronize");
                }
            }

            std::vector<float> actual_route_weights(static_cast<std::size_t>(kRouteElements), 0.0f);
            std::vector<std::int64_t> actual_route_indices(static_cast<std::size_t>(kRouteElements), 0);
            std::vector<float> actual_broadcast_weights(static_cast<std::size_t>(kBroadcastRouteElements), 0.0f);
            std::vector<std::int64_t> actual_broadcast_indices(static_cast<std::size_t>(kBroadcastRouteElements), 0);
            std::vector<float> actual_out(static_cast<std::size_t>(kTokenElements), 0.0f);
            std::vector<float> actual_grad_x(static_cast<std::size_t>(kTokenElements), 0.0f);
            std::vector<float> actual_grad_w1(w1.size(), 0.0f);
            std::vector<float> actual_grad_w2(w2.size(), 0.0f);
            std::vector<float> actual_grad_w3(w3.size(), 0.0f);
            std::vector<float> actual_density(static_cast<std::size_t>(kExperts), 0.0f);
            std::vector<float> actual_balance_loss(1, 0.0f);
            std::vector<float> actual_w1(w1.size(), 0.0f);
            auto copy_back_float = [&](std::vector<float>& dst, const float* src, const std::string& name) {
                int copy_status = cuda_memcpy(dst.data(), src, dst.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (copy_status != 0) {
                    error = cuda_error(copy_status, "cudaMemcpy " + name + " D2H");
                    return false;
                }
                return true;
            };
            auto copy_back_i64 = [&](std::vector<std::int64_t>& dst, const std::int64_t* src, const std::string& name) {
                int copy_status = cuda_memcpy(dst.data(), src, dst.size() * sizeof(std::int64_t), kCudaMemcpyDeviceToHost);
                if (copy_status != 0) {
                    error = cuda_error(copy_status, "cudaMemcpy " + name + " D2H");
                    return false;
                }
                return true;
            };
            if (error.empty()) {
                copy_back_float(actual_route_weights, d_route_weights, "route_weights") &&
                    copy_back_i64(actual_route_indices, d_route_indices, "route_indices") &&
                    copy_back_float(actual_broadcast_weights, d_broadcast_weights, "broadcast_weights") &&
                    copy_back_i64(actual_broadcast_indices, d_broadcast_indices, "broadcast_indices") &&
                    copy_back_float(actual_out, d_out, "out") &&
                    copy_back_float(actual_grad_x, d_grad_x, "grad_x") &&
                    copy_back_float(actual_grad_w1, d_grad_w1, "grad_w1") &&
                    copy_back_float(actual_grad_w2, d_grad_w2, "grad_w2") &&
                    copy_back_float(actual_grad_w3, d_grad_w3, "grad_w3") &&
                    copy_back_float(actual_density, d_density, "density") &&
                    copy_back_float(actual_balance_loss, d_balance_loss, "balance_loss") &&
                    copy_back_float(actual_w1, d_w1, "w1_updated");
            }

            std::vector<float> expected_route_weights(static_cast<std::size_t>(kRouteElements), 0.0f);
            std::vector<std::int64_t> expected_route_indices = {1, 0};
            const float top_max = route_logits[1];
            const float denom = std::exp(route_logits[1] - top_max) + std::exp(route_logits[0] - top_max);
            expected_route_weights[0] = std::exp(route_logits[1] - top_max) / denom;
            expected_route_weights[1] = std::exp(route_logits[0] - top_max) / denom;
            std::vector<float> expected_broadcast_weights(static_cast<std::size_t>(kBroadcastRouteElements), 0.0f);
            std::vector<std::int64_t> expected_broadcast_indices(static_cast<std::size_t>(kBroadcastRouteElements), 0);
            for (std::int64_t token = 0; token < kTokens; ++token) {
                for (std::int64_t k = 0; k < kTopK; ++k) {
                    expected_broadcast_weights[static_cast<std::size_t>(token * kTopK + k)] =
                        expected_route_weights[static_cast<std::size_t>(k)];
                    expected_broadcast_indices[static_cast<std::size_t>(token * kTopK + k)] =
                        expected_route_indices[static_cast<std::size_t>(k)];
                }
            }

            std::vector<float> expected_out(static_cast<std::size_t>(kTokenElements), 0.0f);
            std::vector<float> expected_grad_x(static_cast<std::size_t>(kTokenElements), 0.0f);
            std::vector<float> expected_grad_w1(w1.size(), 0.0f);
            std::vector<float> expected_grad_w2(w2.size(), 0.0f);
            std::vector<float> expected_grad_w3(w3.size(), 0.0f);
            for (std::int64_t token = 0; token < kTokens; ++token) {
                for (std::int64_t k = 0; k < kTopK; ++k) {
                    const std::int64_t expert = expected_broadcast_indices[static_cast<std::size_t>(token * kTopK + k)];
                    const float route_weight = expected_broadcast_weights[static_cast<std::size_t>(token * kTopK + k)];
                    std::vector<float> gate(static_cast<std::size_t>(kHidden), 0.0f);
                    std::vector<float> up(static_cast<std::size_t>(kHidden), 0.0f);
                    std::vector<float> hidden(static_cast<std::size_t>(kHidden), 0.0f);
                    for (std::int64_t h = 0; h < kHidden; ++h) {
                        for (std::int64_t d = 0; d < kDim; ++d) {
                            const float xv = x[static_cast<std::size_t>(token * kDim + d)];
                            const std::size_t w13_idx = static_cast<std::size_t>((expert * kDim + d) * kHidden + h);
                            gate[static_cast<std::size_t>(h)] += xv * w1[w13_idx];
                            up[static_cast<std::size_t>(h)] += xv * w3[w13_idx];
                        }
                        hidden[static_cast<std::size_t>(h)] = silu(gate[static_cast<std::size_t>(h)]) * up[static_cast<std::size_t>(h)];
                    }
                    for (std::int64_t out_d = 0; out_d < kDim; ++out_d) {
                        float expert_acc = 0.0f;
                        for (std::int64_t h = 0; h < kHidden; ++h) {
                            const std::size_t w2_idx = static_cast<std::size_t>((expert * kHidden + h) * kDim + out_d);
                            expert_acc += hidden[static_cast<std::size_t>(h)] * w2[w2_idx];
                        }
                        expected_out[static_cast<std::size_t>(token * kDim + out_d)] += route_weight * expert_acc;
                    }
                    for (std::int64_t h = 0; h < kHidden; ++h) {
                        float grad_hidden = 0.0f;
                        for (std::int64_t out_d = 0; out_d < kDim; ++out_d) {
                            const std::size_t w2_idx = static_cast<std::size_t>((expert * kHidden + h) * kDim + out_d);
                            const float grad_expert_out =
                                route_weight * grad_out[static_cast<std::size_t>(token * kDim + out_d)];
                            expected_grad_w2[w2_idx] += hidden[static_cast<std::size_t>(h)] * grad_expert_out;
                            grad_hidden += grad_expert_out * w2[w2_idx];
                        }
                        const float grad_up = grad_hidden * silu(gate[static_cast<std::size_t>(h)]);
                        const float grad_gate =
                            grad_hidden * up[static_cast<std::size_t>(h)] * dsilu(gate[static_cast<std::size_t>(h)]);
                        for (std::int64_t d = 0; d < kDim; ++d) {
                            const float xv = x[static_cast<std::size_t>(token * kDim + d)];
                            const std::size_t w13_idx = static_cast<std::size_t>((expert * kDim + d) * kHidden + h);
                            expected_grad_w1[w13_idx] += xv * grad_gate;
                            expected_grad_w3[w13_idx] += xv * grad_up;
                            expected_grad_x[static_cast<std::size_t>(token * kDim + d)] +=
                                grad_gate * w1[w13_idx] + grad_up * w3[w13_idx];
                        }
                    }
                }
            }

            std::vector<float> expected_density(static_cast<std::size_t>(kExperts), 0.0f);
            float all_denom = 0.0f;
            const float all_max = *std::max_element(route_logits.begin(), route_logits.end());
            for (float value : route_logits) {
                all_denom += std::exp(value - all_max);
            }
            for (std::int64_t expert = 0; expert < kExperts; ++expert) {
                expected_density[static_cast<std::size_t>(expert)] =
                    std::exp(route_logits[static_cast<std::size_t>(expert)] - all_max) / all_denom;
            }
            float expected_balance_loss = 0.0f;
            for (float value : expected_density) {
                expected_balance_loss += value * value;
            }
            expected_balance_loss *= static_cast<float>(kExperts);

            std::vector<float> expected_w1 = w1;
            for (std::size_t i = 0; i < expected_w1.size(); ++i) {
                const float grad = expected_grad_w1[i];
                const float next_m = 0.1f * grad;
                const float next_v = 0.05f * grad * grad;
                const float denom_adam = std::sqrt(next_v) / std::sqrt(0.05f) + 1e-8f;
                const float decayed = expected_w1[i] * (1.0f - 0.01f * 0.02f);
                expected_w1[i] = decayed - 0.01f * (next_m / 0.1f) / denom_adam;
            }

            if (error.empty()) {
                topk_weight_max_error = max_abs_error(actual_route_weights, expected_route_weights);
                topk_index_max_error = max_i64_error(actual_route_indices, expected_route_indices);
                broadcast_weight_max_error = max_abs_error(actual_broadcast_weights, expected_broadcast_weights);
                broadcast_index_max_error = max_i64_error(actual_broadcast_indices, expected_broadcast_indices);
                moe_forward_max_error = max_abs_error(actual_out, expected_out);
                moe_grad_x_max_error = max_abs_error(actual_grad_x, expected_grad_x);
                moe_grad_w1_max_error = max_abs_error(actual_grad_w1, expected_grad_w1);
                moe_grad_w2_max_error = max_abs_error(actual_grad_w2, expected_grad_w2);
                moe_grad_w3_max_error = max_abs_error(actual_grad_w3, expected_grad_w3);
                balance_density_max_error = max_abs_error(actual_density, expected_density);
                balance_loss_max_error = std::fabs(actual_balance_loss[0] - expected_balance_loss);
                adamw_w1_max_error = max_abs_error(actual_w1, expected_w1);
            }
            constexpr float kTolerance = 2e-6f;
            constexpr float kAdamTolerance = 2e-5f;
            passed = error.empty() &&
                topk_weight_max_error <= kTolerance &&
                topk_index_max_error == 0 &&
                broadcast_weight_max_error <= kTolerance &&
                broadcast_index_max_error == 0 &&
                moe_forward_max_error <= kTolerance &&
                moe_grad_x_max_error <= kTolerance &&
                moe_grad_w1_max_error <= kTolerance &&
                moe_grad_w2_max_error <= kTolerance &&
                moe_grad_w3_max_error <= kTolerance &&
                balance_density_max_error <= kTolerance &&
                balance_loss_max_error <= kTolerance &&
                adamw_w1_max_error <= kAdamTolerance;
            if (!passed && error.empty()) {
                error = "MoE route/expert train-step smoke exceeded tolerance";
            }
        }
    }
    free_allocated();
    close_handles();

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(NFN_NATIVE_MODEL_FAMILY) << "\",\n"
        << "  \"native_target\": \"" << json_escape(NFN_NATIVE_TARGET_NAME) << "\",\n"
        << "  \"smoke\": \"moe_route_expert_train_step_slice\",\n"
        << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(error) << "\",\n"
        << "  \"compiled_native_boundary\": true,\n"
        << "  \"torch_required\": false,\n"
        << "  \"graph_editor_tensor_flow\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_ops_lib) << "\",\n"
        << "  \"tile_ops_loaded\": " << (tile_ops_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"shape\": {\"tokens\": " << kTokens << ", \"dim\": " << kDim
        << ", \"hidden_dim\": " << kHidden << ", \"experts\": " << kExperts
        << ", \"top_k\": " << kTopK << "},\n"
        << "  \"loop_composition_stages\": [\n"
        << "    \"nfn_native_tile_topk_route_float32\",\n"
        << "    \"nfn_native_tile_broadcast_expert_routes_float32\",\n"
        << "    \"nfn_native_tile_moe_swiglu_forward_float32\",\n"
        << "    \"nfn_native_tile_moe_swiglu_backward_float32\",\n"
        << "    \"nfn_native_tile_route_balance_density_float32\",\n"
        << "    \"nfn_native_tile_route_balance_loss_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"max_errors\": {"
        << "\"topk_weight\":" << topk_weight_max_error
        << ", \"topk_index\":" << topk_index_max_error
        << ", \"broadcast_weight\":" << broadcast_weight_max_error
        << ", \"broadcast_index\":" << broadcast_index_max_error
        << ", \"moe_forward\":" << moe_forward_max_error
        << ", \"moe_grad_x\":" << moe_grad_x_max_error
        << ", \"moe_grad_w1\":" << moe_grad_w1_max_error
        << ", \"moe_grad_w2\":" << moe_grad_w2_max_error
        << ", \"moe_grad_w3\":" << moe_grad_w3_max_error
        << ", \"balance_density\":" << balance_density_max_error
        << ", \"balance_loss\":" << balance_loss_max_error
        << ", \"adamw_w1\":" << adamw_w1_max_error
        << "}\n"
        << "}\n";
    return passed ? 0 : 2;
}

int print_jepa_target_encoder_smoke_json(const Config& cfg, const char* program) {
    const std::string family = NFN_NATIVE_MODEL_FAMILY;
    const bool jepa_family = family.find("jepa") != std::string::npos || family == "unknown";
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
    using LatentPoolFn = int (*)(const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);

    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;
    LatentPoolFn latent_pool = nullptr;
    LinearFn linear = nullptr;

    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    constexpr std::int64_t kBatch = 2;
    constexpr std::int64_t kSeqLen = 4;
    constexpr std::int64_t kDim = 3;
    constexpr std::int64_t kLatentDim = 2;
    constexpr std::int64_t kInputElements = kBatch * kSeqLen * kDim;
    constexpr std::int64_t kPooledElements = kBatch * kDim;
    constexpr std::int64_t kProjectedElements = kBatch * kLatentDim;

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
    float pooled_max_error = 0.0f;
    float projected_max_error = 0.0f;

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

    if (!jepa_family) {
        error = "JEPA target-encoder smoke commands are only valid for JEPA-family native preflights";
    } else {
        tile_handle = dlopen(tile_ops_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            const char* raw = dlerror();
            error = raw == nullptr ? "failed to load Tile ops library" : raw;
        } else {
            tile_ops_loaded = true;
            latent_pool = load_symbol<LatentPoolFn>(tile_handle, "nfn_native_tile_latent_pool_float32");
            linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
            if (latent_pool == nullptr || linear == nullptr) {
                error = "Tile ops library is missing one or more JEPA target-encoder symbols";
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
        std::vector<float> target_hidden(static_cast<std::size_t>(kInputElements));
        const std::vector<float> mask = {
            1.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
        };
        const std::vector<float> weight = {
            0.2f, -0.1f, 0.05f,
            -0.03f, 0.07f, 0.11f,
        };
        const std::vector<float> bias = {0.01f, -0.02f};
        for (std::int64_t i = 0; i < kInputElements; ++i) {
            target_hidden[static_cast<std::size_t>(i)] =
                0.05f * static_cast<float>((i % 9) + 1) - 0.01f * static_cast<float>(i / 3);
        }

        float* d_target_hidden = nullptr;
        float* d_mask = nullptr;
        float* d_pooled = nullptr;
        float* d_weight = nullptr;
        float* d_bias = nullptr;
        float* d_projected = nullptr;
        auto alloc = [&](float** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(float));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        if (alloc(&d_target_hidden, target_hidden.size(), "target_hidden") &&
            alloc(&d_mask, mask.size(), "mask") &&
            alloc(&d_pooled, static_cast<std::size_t>(kPooledElements), "pooled") &&
            alloc(&d_weight, weight.size(), "weight") &&
            alloc(&d_bias, bias.size(), "bias") &&
            alloc(&d_projected, static_cast<std::size_t>(kProjectedElements), "projected")) {
            int status = cuda_memcpy(
                d_target_hidden, target_hidden.data(), target_hidden.size() * sizeof(float), kCudaMemcpyHostToDevice);
            if (status == 0) {
                status = cuda_memcpy(d_mask, mask.data(), mask.size() * sizeof(float), kCudaMemcpyHostToDevice);
            }
            if (status == 0) {
                status = cuda_memcpy(d_weight, weight.data(), weight.size() * sizeof(float), kCudaMemcpyHostToDevice);
            }
            if (status == 0) {
                status = cuda_memcpy(d_bias, bias.data(), bias.size() * sizeof(float), kCudaMemcpyHostToDevice);
            }
            if (status != 0) {
                error = cuda_error(status, "cudaMemcpy JEPA target encoder H2D");
            }
            if (error.empty()) {
                status = latent_pool(d_target_hidden, d_mask, d_pooled, kBatch, kSeqLen, kDim, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_latent_pool_float32");
                }
            }
            if (error.empty()) {
                status = linear(d_pooled, d_weight, d_bias, d_projected, kBatch, kDim, kLatentDim, true, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_float32 target projection");
                }
            }
            if (error.empty()) {
                status = cuda_device_synchronize();
                if (status != 0) {
                    error = cuda_error(status, "cudaDeviceSynchronize JEPA target encoder");
                }
            }
            std::vector<float> actual_pooled(static_cast<std::size_t>(kPooledElements), 0.0f);
            std::vector<float> actual_projected(static_cast<std::size_t>(kProjectedElements), 0.0f);
            if (error.empty()) {
                status = cuda_memcpy(actual_pooled.data(), d_pooled, actual_pooled.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (status == 0) {
                    status = cuda_memcpy(
                        actual_projected.data(), d_projected, actual_projected.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                }
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy JEPA target encoder D2H");
                }
            }

            std::vector<float> expected_pooled(static_cast<std::size_t>(kPooledElements), 0.0f);
            for (std::int64_t batch = 0; batch < kBatch; ++batch) {
                float count = 0.0f;
                for (std::int64_t pos = 0; pos < kSeqLen; ++pos) {
                    count += mask[static_cast<std::size_t>(batch * kSeqLen + pos)] > 0.0f ? 1.0f : 0.0f;
                }
                const bool fallback_mean = count == 0.0f;
                const float denom = fallback_mean ? static_cast<float>(kSeqLen) : count;
                for (std::int64_t pos = 0; pos < kSeqLen; ++pos) {
                    const float include =
                        fallback_mean || mask[static_cast<std::size_t>(batch * kSeqLen + pos)] > 0.0f ? 1.0f : 0.0f;
                    for (std::int64_t dim = 0; dim < kDim; ++dim) {
                        expected_pooled[static_cast<std::size_t>(batch * kDim + dim)] +=
                            include *
                            target_hidden[static_cast<std::size_t>((batch * kSeqLen + pos) * kDim + dim)] / denom;
                    }
                }
            }
            std::vector<float> expected_projected(static_cast<std::size_t>(kProjectedElements), 0.0f);
            for (std::int64_t batch = 0; batch < kBatch; ++batch) {
                for (std::int64_t out_dim = 0; out_dim < kLatentDim; ++out_dim) {
                    float value = bias[static_cast<std::size_t>(out_dim)];
                    for (std::int64_t dim = 0; dim < kDim; ++dim) {
                        value += expected_pooled[static_cast<std::size_t>(batch * kDim + dim)] *
                            weight[static_cast<std::size_t>(out_dim * kDim + dim)];
                    }
                    expected_projected[static_cast<std::size_t>(batch * kLatentDim + out_dim)] = value;
                }
            }
            if (error.empty()) {
                pooled_max_error = max_abs_error(actual_pooled, expected_pooled);
                projected_max_error = max_abs_error(actual_projected, expected_projected);
            }
            constexpr float kTolerance = 2e-6f;
            passed = error.empty() && pooled_max_error <= kTolerance && projected_max_error <= kTolerance;
            if (!passed && error.empty()) {
                error = "JEPA target-encoder smoke exceeded tolerance";
            }
        }
    }
    free_allocated();
    close_handles();

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(NFN_NATIVE_MODEL_FAMILY) << "\",\n"
        << "  \"native_target\": \"" << json_escape(NFN_NATIVE_TARGET_NAME) << "\",\n"
        << "  \"smoke\": \"jepa_target_encoder_forward_slice\",\n"
        << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(error) << "\",\n"
        << "  \"compiled_native_boundary\": true,\n"
        << "  \"torch_required\": false,\n"
        << "  \"graph_editor_tensor_flow\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_ops_lib) << "\",\n"
        << "  \"tile_ops_loaded\": " << (tile_ops_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"shape\": {\"batch\": " << kBatch << ", \"seq_len\": " << kSeqLen
        << ", \"dim\": " << kDim << ", \"latent_dim\": " << kLatentDim << "},\n"
        << "  \"loop_composition_stages\": [\n"
        << "    \"nfn_native_tile_latent_pool_float32\",\n"
        << "    \"nfn_native_tile_linear_float32\"\n"
        << "  ],\n"
        << "  \"max_errors\": {"
        << "\"pooled\":" << pooled_max_error
        << ", \"projected\":" << projected_max_error
        << "}\n"
        << "}\n";
    return passed ? 0 : 2;
}

int print_jepa_ar_loss_smoke_json(const Config& cfg, const char* program) {
    const std::string family = NFN_NATIVE_MODEL_FAMILY;
    const bool jepa_family = family.find("jepa") != std::string::npos || family == "unknown";
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
    using TokenCrossEntropyPartialsFn =
        int (*)(const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using LatentMseLossFn = int (*)(const float*, const float*, float*, std::int64_t, void*);

    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;
    TokenCrossEntropyPartialsFn ce_partials = nullptr;
    LatentMseLossFn latent_mse_loss = nullptr;

    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    constexpr std::int64_t kRows = 3;
    constexpr std::int64_t kVocab = 5;
    constexpr std::int64_t kLatentElements = 6;
    constexpr float kArLossCoef = 1.0f;
    constexpr float kJepaLossCoef = 0.25f;

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

    bool passed = false;
    float ar_loss_max_error = 0.0f;
    float jepa_loss_max_error = 0.0f;
    float total_loss_max_error = 0.0f;

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

    if (!jepa_family) {
        error = "JEPA AR+loss smoke commands are only valid for JEPA-family native preflights";
    } else {
        tile_handle = dlopen(tile_ops_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            const char* raw = dlerror();
            error = raw == nullptr ? "failed to load Tile ops library" : raw;
        } else {
            tile_ops_loaded = true;
            ce_partials = load_symbol<TokenCrossEntropyPartialsFn>(
                tile_handle, "nfn_native_tile_token_cross_entropy_partials_float32");
            latent_mse_loss = load_symbol<LatentMseLossFn>(
                tile_handle, "nfn_native_tile_latent_mse_loss_float32");
            if (ce_partials == nullptr || latent_mse_loss == nullptr) {
                error = "Tile ops library is missing one or more JEPA AR+loss composition symbols";
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
        const std::vector<float> logits = {
            0.10f, -0.20f, 0.30f, 0.05f, -0.10f,
            -0.15f, 0.25f, 0.05f, -0.05f, 0.10f,
            0.20f, 0.15f, -0.25f, 0.35f, -0.30f,
        };
        const std::vector<std::int64_t> targets = {2, 1, 3};
        const std::vector<float> pred = {0.05f, -0.10f, 0.20f, 0.30f, -0.15f, 0.12f};
        const std::vector<float> latent_target = {0.01f, -0.05f, 0.25f, 0.10f, -0.20f, 0.08f};

        float* d_logits = nullptr;
        float* d_ar_loss = nullptr;
        float* d_pred = nullptr;
        float* d_latent_target = nullptr;
        float* d_jepa_loss = nullptr;
        std::int64_t* d_targets = nullptr;
        auto alloc = [&](float** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(float));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        auto alloc_i64 = [&](std::int64_t** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(std::int64_t));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        if (alloc(&d_logits, logits.size(), "logits") &&
            alloc_i64(&d_targets, targets.size(), "targets") &&
            alloc(&d_ar_loss, 1, "ar_loss") &&
            alloc(&d_pred, pred.size(), "pred") &&
            alloc(&d_latent_target, latent_target.size(), "latent_target") &&
            alloc(&d_jepa_loss, 1, "jepa_loss")) {
            int status = cuda_memcpy(
                d_logits, logits.data(), logits.size() * sizeof(float), kCudaMemcpyHostToDevice);
            if (status == 0) {
                status = cuda_memcpy(
                    d_targets, targets.data(), targets.size() * sizeof(std::int64_t), kCudaMemcpyHostToDevice);
            }
            if (status == 0) {
                status = cuda_memcpy(d_pred, pred.data(), pred.size() * sizeof(float), kCudaMemcpyHostToDevice);
            }
            if (status == 0) {
                status = cuda_memcpy(
                    d_latent_target, latent_target.data(), latent_target.size() * sizeof(float), kCudaMemcpyHostToDevice);
            }
            if (status != 0) {
                error = cuda_error(status, "cudaMemcpy JEPA AR+loss H2D");
            }
            if (error.empty()) {
                status = ce_partials(d_logits, d_targets, d_ar_loss, kRows, kVocab, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_token_cross_entropy_partials_float32");
                }
            }
            if (error.empty()) {
                status = latent_mse_loss(d_pred, d_latent_target, d_jepa_loss, kLatentElements, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_latent_mse_loss_float32");
                }
            }
            if (error.empty()) {
                status = cuda_device_synchronize();
                if (status != 0) {
                    error = cuda_error(status, "cudaDeviceSynchronize JEPA AR+loss");
                }
            }
            std::vector<float> actual_ar_loss(1, 0.0f);
            std::vector<float> actual_jepa_loss(1, 0.0f);
            if (error.empty()) {
                status = cuda_memcpy(
                    actual_ar_loss.data(), d_ar_loss, sizeof(float), kCudaMemcpyDeviceToHost);
                if (status == 0) {
                    status = cuda_memcpy(
                        actual_jepa_loss.data(), d_jepa_loss, sizeof(float), kCudaMemcpyDeviceToHost);
                }
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy JEPA AR+loss D2H");
                }
            }

            float expected_ar_loss = 0.0f;
            for (std::int64_t row = 0; row < kRows; ++row) {
                float row_max = logits[static_cast<std::size_t>(row * kVocab)];
                for (std::int64_t col = 1; col < kVocab; ++col) {
                    row_max = std::max(row_max, logits[static_cast<std::size_t>(row * kVocab + col)]);
                }
                float denom = 0.0f;
                for (std::int64_t col = 0; col < kVocab; ++col) {
                    denom += std::exp(logits[static_cast<std::size_t>(row * kVocab + col)] - row_max);
                }
                const float target_logit =
                    logits[static_cast<std::size_t>(row * kVocab + targets[static_cast<std::size_t>(row)])];
                expected_ar_loss += std::log(denom) + row_max - target_logit;
            }
            float expected_jepa_loss = 0.0f;
            for (std::size_t i = 0; i < pred.size(); ++i) {
                const float diff = pred[i] - latent_target[i];
                expected_jepa_loss += diff * diff;
            }
            const float actual_total =
                kArLossCoef * actual_ar_loss[0] + kJepaLossCoef * actual_jepa_loss[0];
            const float expected_total =
                kArLossCoef * expected_ar_loss + kJepaLossCoef * expected_jepa_loss;
            ar_loss_max_error = std::fabs(actual_ar_loss[0] - expected_ar_loss);
            jepa_loss_max_error = std::fabs(actual_jepa_loss[0] - expected_jepa_loss);
            total_loss_max_error = std::fabs(actual_total - expected_total);
            constexpr float kTolerance = 2e-6f;
            passed = error.empty() &&
                ar_loss_max_error <= kTolerance &&
                jepa_loss_max_error <= kTolerance &&
                total_loss_max_error <= kTolerance;
            if (!passed && error.empty()) {
                error = "JEPA AR+loss composition smoke exceeded tolerance";
            }
        }
    }
    free_allocated();
    close_handles();

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(NFN_NATIVE_MODEL_FAMILY) << "\",\n"
        << "  \"native_target\": \"" << json_escape(NFN_NATIVE_TARGET_NAME) << "\",\n"
        << "  \"smoke\": \"jepa_ar_loss_composition_slice\",\n"
        << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(error) << "\",\n"
        << "  \"compiled_native_boundary\": true,\n"
        << "  \"torch_required\": false,\n"
        << "  \"graph_editor_tensor_flow\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_ops_lib) << "\",\n"
        << "  \"tile_ops_loaded\": " << (tile_ops_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"shape\": {\"rows\": " << kRows << ", \"vocab\": " << kVocab
        << ", \"latent_elements\": " << kLatentElements << "},\n"
        << "  \"loss_coefficients\": {\"ar\": " << kArLossCoef
        << ", \"jepa\": " << kJepaLossCoef << "},\n"
        << "  \"loop_composition_stages\": [\n"
        << "    \"nfn_native_tile_token_cross_entropy_partials_float32\",\n"
        << "    \"nfn_native_tile_latent_mse_loss_float32\"\n"
        << "  ],\n"
        << "  \"max_errors\": {"
        << "\"ar_loss\":" << ar_loss_max_error
        << ", \"jepa_loss\":" << jepa_loss_max_error
        << ", \"total_loss\":" << total_loss_max_error
        << "}\n"
        << "}\n";
    return passed ? 0 : 2;
}

int print_jepa_projector_smoke_json(const Config& cfg, const char* program) {
    const std::string family = NFN_NATIVE_MODEL_FAMILY;
    const bool jepa_family = family.find("jepa") != std::string::npos || family == "unknown";
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
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightAccumulateFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LatentMseLossFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using AdamWFn = int (*)(
        float*, const float*, float*, float*, std::int64_t, float, float, float, float, float, float, float, void*);

    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightAccumulateFn linear_backward_weight_accumulate = nullptr;
    LatentMseLossFn latent_mse_loss = nullptr;
    FillFn fill = nullptr;
    AdamWFn adamw = nullptr;

    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    constexpr std::int64_t kRows = 2;
    constexpr std::int64_t kInputDim = 3;
    constexpr std::int64_t kHiddenDim = 4;
    constexpr std::int64_t kLatentDim = 3;
    constexpr std::int64_t kHiddenElements = kRows * kHiddenDim;
    constexpr std::int64_t kLatentElements = kRows * kLatentDim;
    constexpr std::int64_t kProjectorWeightElements = kHiddenDim * kInputDim;
    constexpr std::int64_t kPredictorWeightElements = kLatentDim * kHiddenDim;

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
    float projector_forward_max_error = 0.0f;
    float predictor_forward_max_error = 0.0f;
    float latent_loss_max_error = 0.0f;
    float predictor_grad_hidden_max_error = 0.0f;
    float predictor_grad_weight_max_error = 0.0f;
    float projector_grad_weight_max_error = 0.0f;
    float predictor_weight_update_max_error = 0.0f;

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

    if (!jepa_family) {
        error = "JEPA smoke commands are only valid for JEPA-family native preflights";
    } else {
        tile_handle = dlopen(tile_ops_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            const char* raw = dlerror();
            error = raw == nullptr ? "failed to load Tile ops library" : raw;
        } else {
            tile_ops_loaded = true;
            linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
            linear_backward_input = load_symbol<LinearBackwardInputFn>(
                tile_handle, "nfn_native_tile_linear_backward_input_float32");
            linear_backward_weight_accumulate = load_symbol<LinearBackwardWeightAccumulateFn>(
                tile_handle, "nfn_native_tile_linear_backward_weight_accumulate_float32");
            latent_mse_loss = load_symbol<LatentMseLossFn>(
                tile_handle, "nfn_native_tile_latent_mse_loss_float32");
            fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
            adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
            if (linear == nullptr || linear_backward_input == nullptr ||
                linear_backward_weight_accumulate == nullptr || latent_mse_loss == nullptr ||
                fill == nullptr || adamw == nullptr) {
                error = "Tile ops library is missing one or more JEPA projector train-step symbols";
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
        const std::vector<float> x = {
            0.2f, -0.1f, 0.3f,
            -0.4f, 0.5f, 0.1f,
        };
        const std::vector<float> target = {
            0.05f, -0.02f, 0.08f,
            -0.03f, 0.04f, 0.01f,
        };
        std::vector<float> projector_weight(static_cast<std::size_t>(kProjectorWeightElements));
        std::vector<float> predictor_weight(static_cast<std::size_t>(kPredictorWeightElements));
        for (std::size_t i = 0; i < projector_weight.size(); ++i) {
            projector_weight[i] = 0.03f * static_cast<float>(static_cast<int>(i % 7) - 3);
        }
        for (std::size_t i = 0; i < predictor_weight.size(); ++i) {
            predictor_weight[i] = -0.02f * static_cast<float>(static_cast<int>(i % 5) - 2);
        }

        float* d_x = nullptr;
        float* d_target = nullptr;
        float* d_projector_weight = nullptr;
        float* d_predictor_weight = nullptr;
        float* d_projector_out = nullptr;
        float* d_pred = nullptr;
        float* d_loss = nullptr;
        float* d_grad_pred = nullptr;
        float* d_grad_hidden = nullptr;
        float* d_grad_projector_weight = nullptr;
        float* d_grad_predictor_weight = nullptr;
        float* d_predictor_exp_avg = nullptr;
        float* d_predictor_exp_avg_sq = nullptr;
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
            alloc(&d_target, target.size(), "target") &&
            alloc(&d_projector_weight, projector_weight.size(), "projector_weight") &&
            alloc(&d_predictor_weight, predictor_weight.size(), "predictor_weight") &&
            alloc(&d_projector_out, kHiddenElements, "projector_out") &&
            alloc(&d_pred, kLatentElements, "pred") &&
            alloc(&d_loss, 1, "loss") &&
            alloc(&d_grad_pred, kLatentElements, "grad_pred") &&
            alloc(&d_grad_hidden, kHiddenElements, "grad_hidden") &&
            alloc(&d_grad_projector_weight, projector_weight.size(), "grad_projector_weight") &&
            alloc(&d_grad_predictor_weight, predictor_weight.size(), "grad_predictor_weight") &&
            alloc(&d_predictor_exp_avg, predictor_weight.size(), "predictor_exp_avg") &&
            alloc(&d_predictor_exp_avg_sq, predictor_weight.size(), "predictor_exp_avg_sq")) {
            auto copy_float = [&](float* dst, const std::vector<float>& src, const std::string& name) {
                int status = cuda_memcpy(dst, src.data(), src.size() * sizeof(float), kCudaMemcpyHostToDevice);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy " + name + " H2D");
                    return false;
                }
                return true;
            };
            if (copy_float(d_x, x, "x") &&
                copy_float(d_target, target, "target") &&
                copy_float(d_projector_weight, projector_weight, "projector_weight") &&
                copy_float(d_predictor_weight, predictor_weight, "predictor_weight")) {
                int status = fill(d_grad_projector_weight, kProjectorWeightElements, 0.0f, nullptr);
                if (status == 0) {
                    status = fill(d_grad_predictor_weight, kPredictorWeightElements, 0.0f, nullptr);
                }
                if (status == 0) {
                    status = fill(d_predictor_exp_avg, kPredictorWeightElements, 0.0f, nullptr);
                }
                if (status == 0) {
                    status = fill(d_predictor_exp_avg_sq, kPredictorWeightElements, 0.0f, nullptr);
                }
                if (status != 0) {
                    error = cuda_error(status, "zero JEPA gradients/moments");
                }
            }

            std::vector<float> expected_projector_out(static_cast<std::size_t>(kHiddenElements), 0.0f);
            std::vector<float> expected_pred(static_cast<std::size_t>(kLatentElements), 0.0f);
            for (std::int64_t row = 0; row < kRows; ++row) {
                for (std::int64_t out_d = 0; out_d < kHiddenDim; ++out_d) {
                    float acc = 0.0f;
                    for (std::int64_t in_d = 0; in_d < kInputDim; ++in_d) {
                        acc += x[static_cast<std::size_t>(row * kInputDim + in_d)] *
                            projector_weight[static_cast<std::size_t>(out_d * kInputDim + in_d)];
                    }
                    expected_projector_out[static_cast<std::size_t>(row * kHiddenDim + out_d)] = acc;
                }
                for (std::int64_t out_d = 0; out_d < kLatentDim; ++out_d) {
                    float acc = 0.0f;
                    for (std::int64_t in_d = 0; in_d < kHiddenDim; ++in_d) {
                        acc += expected_projector_out[static_cast<std::size_t>(row * kHiddenDim + in_d)] *
                            predictor_weight[static_cast<std::size_t>(out_d * kHiddenDim + in_d)];
                    }
                    expected_pred[static_cast<std::size_t>(row * kLatentDim + out_d)] = acc;
                }
            }
            std::vector<float> grad_pred(static_cast<std::size_t>(kLatentElements), 0.0f);
            float expected_loss = 0.0f;
            for (std::size_t i = 0; i < expected_pred.size(); ++i) {
                const float diff = expected_pred[i] - target[i];
                expected_loss += diff * diff;
                grad_pred[i] = 2.0f * diff;
            }
            std::vector<float> expected_grad_hidden(static_cast<std::size_t>(kHiddenElements), 0.0f);
            std::vector<float> expected_grad_predictor_weight(predictor_weight.size(), 0.0f);
            std::vector<float> expected_grad_projector_weight(projector_weight.size(), 0.0f);
            for (std::int64_t row = 0; row < kRows; ++row) {
                for (std::int64_t latent = 0; latent < kLatentDim; ++latent) {
                    const float grad = grad_pred[static_cast<std::size_t>(row * kLatentDim + latent)];
                    for (std::int64_t hidden = 0; hidden < kHiddenDim; ++hidden) {
                        expected_grad_predictor_weight[static_cast<std::size_t>(latent * kHiddenDim + hidden)] +=
                            expected_projector_out[static_cast<std::size_t>(row * kHiddenDim + hidden)] * grad;
                        expected_grad_hidden[static_cast<std::size_t>(row * kHiddenDim + hidden)] +=
                            grad * predictor_weight[static_cast<std::size_t>(latent * kHiddenDim + hidden)];
                    }
                }
                for (std::int64_t hidden = 0; hidden < kHiddenDim; ++hidden) {
                    const float grad = expected_grad_hidden[static_cast<std::size_t>(row * kHiddenDim + hidden)];
                    for (std::int64_t in_d = 0; in_d < kInputDim; ++in_d) {
                        expected_grad_projector_weight[static_cast<std::size_t>(hidden * kInputDim + in_d)] +=
                            x[static_cast<std::size_t>(row * kInputDim + in_d)] * grad;
                    }
                }
            }

            if (error.empty() && !copy_float(d_grad_pred, grad_pred, "grad_pred")) {
                // copy_float set error.
            }
            int status = 0;
            if (error.empty()) {
                status = linear(
                    d_x, d_projector_weight, nullptr, d_projector_out,
                    kRows, kInputDim, kHiddenDim, false, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_float32 projector");
                }
            }
            if (error.empty()) {
                status = linear(
                    d_projector_out, d_predictor_weight, nullptr, d_pred,
                    kRows, kHiddenDim, kLatentDim, false, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_float32 predictor");
                }
            }
            if (error.empty()) {
                status = latent_mse_loss(d_pred, d_target, d_loss, kLatentElements, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_latent_mse_loss_float32");
                }
            }
            if (error.empty()) {
                status = linear_backward_input(
                    d_grad_pred, d_predictor_weight, d_grad_hidden,
                    kRows, kHiddenDim, kLatentDim, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_backward_input_float32 predictor");
                }
            }
            if (error.empty()) {
                status = linear_backward_weight_accumulate(
                    d_projector_out, d_grad_pred, d_grad_predictor_weight,
                    kRows, kHiddenDim, kLatentDim, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_backward_weight_accumulate_float32 predictor");
                }
            }
            if (error.empty()) {
                status = linear_backward_weight_accumulate(
                    d_x, d_grad_hidden, d_grad_projector_weight,
                    kRows, kInputDim, kHiddenDim, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_backward_weight_accumulate_float32 projector");
                }
            }
            if (error.empty()) {
                status = adamw(
                    d_predictor_weight,
                    d_grad_predictor_weight,
                    d_predictor_exp_avg,
                    d_predictor_exp_avg_sq,
                    kPredictorWeightElements,
                    0.01f,
                    0.9f,
                    0.95f,
                    1e-8f,
                    0.02f,
                    0.1f,
                    std::sqrt(0.05f),
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_adamw_step_float32 predictor");
                }
            }
            if (error.empty()) {
                status = cuda_device_synchronize();
                if (status != 0) {
                    error = cuda_error(status, "cudaDeviceSynchronize");
                }
            }

            std::vector<float> actual_projector_out(static_cast<std::size_t>(kHiddenElements), 0.0f);
            std::vector<float> actual_pred(static_cast<std::size_t>(kLatentElements), 0.0f);
            std::vector<float> actual_loss(1, 0.0f);
            std::vector<float> actual_grad_hidden(static_cast<std::size_t>(kHiddenElements), 0.0f);
            std::vector<float> actual_grad_projector_weight(projector_weight.size(), 0.0f);
            std::vector<float> actual_grad_predictor_weight(predictor_weight.size(), 0.0f);
            std::vector<float> actual_predictor_weight(predictor_weight.size(), 0.0f);
            auto copy_back_float = [&](std::vector<float>& dst, const float* src, const std::string& name) {
                int copy_status = cuda_memcpy(dst.data(), src, dst.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (copy_status != 0) {
                    error = cuda_error(copy_status, "cudaMemcpy " + name + " D2H");
                    return false;
                }
                return true;
            };
            if (error.empty()) {
                copy_back_float(actual_projector_out, d_projector_out, "projector_out") &&
                    copy_back_float(actual_pred, d_pred, "pred") &&
                    copy_back_float(actual_loss, d_loss, "loss") &&
                    copy_back_float(actual_grad_hidden, d_grad_hidden, "grad_hidden") &&
                    copy_back_float(actual_grad_projector_weight, d_grad_projector_weight, "grad_projector_weight") &&
                    copy_back_float(actual_grad_predictor_weight, d_grad_predictor_weight, "grad_predictor_weight") &&
                    copy_back_float(actual_predictor_weight, d_predictor_weight, "predictor_weight");
            }

            std::vector<float> expected_predictor_weight = predictor_weight;
            for (std::size_t i = 0; i < expected_predictor_weight.size(); ++i) {
                const float grad = expected_grad_predictor_weight[i];
                const float next_m = 0.1f * grad;
                const float next_v = 0.05f * grad * grad;
                const float denom = std::sqrt(next_v) / std::sqrt(0.05f) + 1e-8f;
                const float decayed = expected_predictor_weight[i] * (1.0f - 0.01f * 0.02f);
                expected_predictor_weight[i] = decayed - 0.01f * (next_m / 0.1f) / denom;
            }

            if (error.empty()) {
                projector_forward_max_error = max_abs_error(actual_projector_out, expected_projector_out);
                predictor_forward_max_error = max_abs_error(actual_pred, expected_pred);
                latent_loss_max_error = std::fabs(actual_loss[0] - expected_loss);
                predictor_grad_hidden_max_error = max_abs_error(actual_grad_hidden, expected_grad_hidden);
                predictor_grad_weight_max_error = max_abs_error(actual_grad_predictor_weight, expected_grad_predictor_weight);
                projector_grad_weight_max_error = max_abs_error(actual_grad_projector_weight, expected_grad_projector_weight);
                predictor_weight_update_max_error = max_abs_error(actual_predictor_weight, expected_predictor_weight);
            }
            constexpr float kTolerance = 5e-6f;
            constexpr float kAdamTolerance = 2e-5f;
            passed = error.empty() &&
                projector_forward_max_error <= kTolerance &&
                predictor_forward_max_error <= kTolerance &&
                latent_loss_max_error <= kTolerance &&
                predictor_grad_hidden_max_error <= kTolerance &&
                predictor_grad_weight_max_error <= kTolerance &&
                projector_grad_weight_max_error <= kTolerance &&
                predictor_weight_update_max_error <= kAdamTolerance;
            if (!passed && error.empty()) {
                error = "JEPA projector train-step smoke exceeded tolerance";
            }
        }
    }
    free_allocated();
    close_handles();

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(NFN_NATIVE_MODEL_FAMILY) << "\",\n"
        << "  \"native_target\": \"" << json_escape(NFN_NATIVE_TARGET_NAME) << "\",\n"
        << "  \"smoke\": \"jepa_projector_train_step_slice\",\n"
        << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(error) << "\",\n"
        << "  \"compiled_native_boundary\": true,\n"
        << "  \"torch_required\": false,\n"
        << "  \"graph_editor_tensor_flow\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_ops_lib) << "\",\n"
        << "  \"tile_ops_loaded\": " << (tile_ops_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"shape\": {\"rows\": " << kRows << ", \"input_dim\": " << kInputDim
        << ", \"hidden_dim\": " << kHiddenDim << ", \"latent_dim\": " << kLatentDim << "},\n"
        << "  \"loop_composition_stages\": [\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_latent_mse_loss_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_accumulate_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"max_errors\": {"
        << "\"projector_forward\":" << projector_forward_max_error
        << ", \"predictor_forward\":" << predictor_forward_max_error
        << ", \"latent_loss\":" << latent_loss_max_error
        << ", \"predictor_grad_hidden\":" << predictor_grad_hidden_max_error
        << ", \"predictor_grad_weight\":" << predictor_grad_weight_max_error
        << ", \"projector_grad_weight\":" << projector_grad_weight_max_error
        << ", \"predictor_weight_update\":" << predictor_weight_update_max_error
        << "}\n"
        << "}\n";
    return passed ? 0 : 2;
}

int print_diffusion_denoise_smoke_json(const Config& cfg, const char* program) {
    const std::string family = NFN_NATIVE_MODEL_FAMILY;
    const bool diffusion_family = family.find("diffusion") != std::string::npos || family == "unknown";
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
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightAccumulateFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LatentMseLossFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using AdamWFn = int (*)(
        float*, const float*, float*, float*, std::int64_t, float, float, float, float, float, float, float, void*);

    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightAccumulateFn linear_backward_weight_accumulate = nullptr;
    LatentMseLossFn latent_mse_loss = nullptr;
    FillFn fill = nullptr;
    AdamWFn adamw = nullptr;

    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    constexpr std::int64_t kRows = 2;
    constexpr std::int64_t kDim = 4;
    constexpr std::int64_t kElements = kRows * kDim;
    constexpr std::int64_t kWeightElements = kDim * kDim;

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
    float denoise_forward_max_error = 0.0f;
    float denoise_loss_max_error = 0.0f;
    float denoise_grad_input_max_error = 0.0f;
    float denoise_grad_weight_max_error = 0.0f;
    float denoise_weight_update_max_error = 0.0f;

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

    if (!diffusion_family) {
        error = "Diffusion smoke commands are only valid for diffusion-family native preflights";
    } else {
        tile_handle = dlopen(tile_ops_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            const char* raw = dlerror();
            error = raw == nullptr ? "failed to load Tile ops library" : raw;
        } else {
            tile_ops_loaded = true;
            linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
            linear_backward_input = load_symbol<LinearBackwardInputFn>(
                tile_handle, "nfn_native_tile_linear_backward_input_float32");
            linear_backward_weight_accumulate = load_symbol<LinearBackwardWeightAccumulateFn>(
                tile_handle, "nfn_native_tile_linear_backward_weight_accumulate_float32");
            latent_mse_loss = load_symbol<LatentMseLossFn>(
                tile_handle, "nfn_native_tile_latent_mse_loss_float32");
            fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
            adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
            if (linear == nullptr || linear_backward_input == nullptr ||
                linear_backward_weight_accumulate == nullptr || latent_mse_loss == nullptr ||
                fill == nullptr || adamw == nullptr) {
                error = "Tile ops library is missing one or more diffusion denoise train-step symbols";
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
        const std::vector<float> noisy_latent = {
            0.15f, -0.20f, 0.35f, 0.05f,
            -0.10f, 0.25f, -0.30f, 0.40f,
        };
        const std::vector<float> target_noise = {
            0.02f, -0.04f, 0.06f, 0.08f,
            -0.01f, 0.03f, -0.05f, 0.07f,
        };
        std::vector<float> denoise_weight(static_cast<std::size_t>(kWeightElements));
        for (std::size_t i = 0; i < denoise_weight.size(); ++i) {
            denoise_weight[i] = 0.025f * static_cast<float>(static_cast<int>(i % 9) - 4);
        }

        std::vector<float> expected_pred(static_cast<std::size_t>(kElements), 0.0f);
        for (std::int64_t row = 0; row < kRows; ++row) {
            for (std::int64_t out_d = 0; out_d < kDim; ++out_d) {
                float acc = 0.0f;
                for (std::int64_t in_d = 0; in_d < kDim; ++in_d) {
                    acc += noisy_latent[static_cast<std::size_t>(row * kDim + in_d)] *
                        denoise_weight[static_cast<std::size_t>(out_d * kDim + in_d)];
                }
                expected_pred[static_cast<std::size_t>(row * kDim + out_d)] = acc;
            }
        }
        std::vector<float> grad_pred(static_cast<std::size_t>(kElements), 0.0f);
        float expected_loss = 0.0f;
        for (std::size_t i = 0; i < expected_pred.size(); ++i) {
            const float diff = expected_pred[i] - target_noise[i];
            expected_loss += diff * diff;
            grad_pred[i] = 2.0f * diff;
        }
        std::vector<float> expected_grad_input(static_cast<std::size_t>(kElements), 0.0f);
        std::vector<float> expected_grad_weight(denoise_weight.size(), 0.0f);
        for (std::int64_t row = 0; row < kRows; ++row) {
            for (std::int64_t out_d = 0; out_d < kDim; ++out_d) {
                const float grad = grad_pred[static_cast<std::size_t>(row * kDim + out_d)];
                for (std::int64_t in_d = 0; in_d < kDim; ++in_d) {
                    expected_grad_weight[static_cast<std::size_t>(out_d * kDim + in_d)] +=
                        noisy_latent[static_cast<std::size_t>(row * kDim + in_d)] * grad;
                    expected_grad_input[static_cast<std::size_t>(row * kDim + in_d)] +=
                        grad * denoise_weight[static_cast<std::size_t>(out_d * kDim + in_d)];
                }
            }
        }

        float* d_noisy_latent = nullptr;
        float* d_target_noise = nullptr;
        float* d_denoise_weight = nullptr;
        float* d_pred = nullptr;
        float* d_loss = nullptr;
        float* d_grad_pred = nullptr;
        float* d_grad_input = nullptr;
        float* d_grad_weight = nullptr;
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
        if (alloc(&d_noisy_latent, noisy_latent.size(), "noisy_latent") &&
            alloc(&d_target_noise, target_noise.size(), "target_noise") &&
            alloc(&d_denoise_weight, denoise_weight.size(), "denoise_weight") &&
            alloc(&d_pred, kElements, "pred") &&
            alloc(&d_loss, 1, "loss") &&
            alloc(&d_grad_pred, grad_pred.size(), "grad_pred") &&
            alloc(&d_grad_input, kElements, "grad_input") &&
            alloc(&d_grad_weight, denoise_weight.size(), "grad_weight") &&
            alloc(&d_exp_avg, denoise_weight.size(), "exp_avg") &&
            alloc(&d_exp_avg_sq, denoise_weight.size(), "exp_avg_sq")) {
            auto copy_float = [&](float* dst, const std::vector<float>& src, const std::string& name) {
                int status = cuda_memcpy(dst, src.data(), src.size() * sizeof(float), kCudaMemcpyHostToDevice);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy " + name + " H2D");
                    return false;
                }
                return true;
            };
            copy_float(d_noisy_latent, noisy_latent, "noisy_latent") &&
                copy_float(d_target_noise, target_noise, "target_noise") &&
                copy_float(d_denoise_weight, denoise_weight, "denoise_weight") &&
                copy_float(d_grad_pred, grad_pred, "grad_pred");
            int status = 0;
            if (error.empty()) {
                status = fill(d_grad_weight, kWeightElements, 0.0f, nullptr);
                if (status == 0) {
                    status = fill(d_exp_avg, kWeightElements, 0.0f, nullptr);
                }
                if (status == 0) {
                    status = fill(d_exp_avg_sq, kWeightElements, 0.0f, nullptr);
                }
                if (status != 0) {
                    error = cuda_error(status, "zero diffusion gradients/moments");
                }
            }
            if (error.empty()) {
                status = linear(
                    d_noisy_latent, d_denoise_weight, nullptr, d_pred,
                    kRows, kDim, kDim, false, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_float32 denoise");
                }
            }
            if (error.empty()) {
                status = latent_mse_loss(d_pred, d_target_noise, d_loss, kElements, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_latent_mse_loss_float32");
                }
            }
            if (error.empty()) {
                status = linear_backward_input(
                    d_grad_pred, d_denoise_weight, d_grad_input,
                    kRows, kDim, kDim, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_backward_input_float32 denoise");
                }
            }
            if (error.empty()) {
                status = linear_backward_weight_accumulate(
                    d_noisy_latent, d_grad_pred, d_grad_weight,
                    kRows, kDim, kDim, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_backward_weight_accumulate_float32 denoise");
                }
            }
            if (error.empty()) {
                status = adamw(
                    d_denoise_weight,
                    d_grad_weight,
                    d_exp_avg,
                    d_exp_avg_sq,
                    kWeightElements,
                    0.01f,
                    0.9f,
                    0.95f,
                    1e-8f,
                    0.02f,
                    0.1f,
                    std::sqrt(0.05f),
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_adamw_step_float32 denoise");
                }
            }
            if (error.empty()) {
                status = cuda_device_synchronize();
                if (status != 0) {
                    error = cuda_error(status, "cudaDeviceSynchronize");
                }
            }

            std::vector<float> actual_pred(static_cast<std::size_t>(kElements), 0.0f);
            std::vector<float> actual_loss(1, 0.0f);
            std::vector<float> actual_grad_input(static_cast<std::size_t>(kElements), 0.0f);
            std::vector<float> actual_grad_weight(denoise_weight.size(), 0.0f);
            std::vector<float> actual_denoise_weight(denoise_weight.size(), 0.0f);
            auto copy_back_float = [&](std::vector<float>& dst, const float* src, const std::string& name) {
                int copy_status = cuda_memcpy(dst.data(), src, dst.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (copy_status != 0) {
                    error = cuda_error(copy_status, "cudaMemcpy " + name + " D2H");
                    return false;
                }
                return true;
            };
            if (error.empty()) {
                copy_back_float(actual_pred, d_pred, "pred") &&
                    copy_back_float(actual_loss, d_loss, "loss") &&
                    copy_back_float(actual_grad_input, d_grad_input, "grad_input") &&
                    copy_back_float(actual_grad_weight, d_grad_weight, "grad_weight") &&
                    copy_back_float(actual_denoise_weight, d_denoise_weight, "denoise_weight");
            }

            std::vector<float> expected_denoise_weight = denoise_weight;
            for (std::size_t i = 0; i < expected_denoise_weight.size(); ++i) {
                const float grad = expected_grad_weight[i];
                const float next_m = 0.1f * grad;
                const float next_v = 0.05f * grad * grad;
                const float denom = std::sqrt(next_v) / std::sqrt(0.05f) + 1e-8f;
                const float decayed = expected_denoise_weight[i] * (1.0f - 0.01f * 0.02f);
                expected_denoise_weight[i] = decayed - 0.01f * (next_m / 0.1f) / denom;
            }
            if (error.empty()) {
                denoise_forward_max_error = max_abs_error(actual_pred, expected_pred);
                denoise_loss_max_error = std::fabs(actual_loss[0] - expected_loss);
                denoise_grad_input_max_error = max_abs_error(actual_grad_input, expected_grad_input);
                denoise_grad_weight_max_error = max_abs_error(actual_grad_weight, expected_grad_weight);
                denoise_weight_update_max_error = max_abs_error(actual_denoise_weight, expected_denoise_weight);
            }
            constexpr float kTolerance = 5e-5f;
            constexpr float kAdamTolerance = 2e-5f;
            passed = error.empty() &&
                denoise_forward_max_error <= kTolerance &&
                denoise_loss_max_error <= kTolerance &&
                denoise_grad_input_max_error <= kTolerance &&
                denoise_grad_weight_max_error <= kTolerance &&
                denoise_weight_update_max_error <= kAdamTolerance;
            if (!passed && error.empty()) {
                error = "Diffusion denoise train-step smoke exceeded tolerance";
            }
        }
    }
    free_allocated();
    close_handles();

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(NFN_NATIVE_MODEL_FAMILY) << "\",\n"
        << "  \"native_target\": \"" << json_escape(NFN_NATIVE_TARGET_NAME) << "\",\n"
        << "  \"smoke\": \"diffusion_denoise_train_step_slice\",\n"
        << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(error) << "\",\n"
        << "  \"compiled_native_boundary\": true,\n"
        << "  \"torch_required\": false,\n"
        << "  \"graph_editor_tensor_flow\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_ops_lib) << "\",\n"
        << "  \"tile_ops_loaded\": " << (tile_ops_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"shape\": {\"rows\": " << kRows << ", \"latent_dim\": " << kDim << "},\n"
        << "  \"loop_composition_stages\": [\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_latent_mse_loss_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_accumulate_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"max_errors\": {"
        << "\"denoise_forward\":" << denoise_forward_max_error
        << ", \"denoise_loss\":" << denoise_loss_max_error
        << ", \"denoise_grad_input\":" << denoise_grad_input_max_error
        << ", \"denoise_grad_weight\":" << denoise_grad_weight_max_error
        << ", \"denoise_weight_update\":" << denoise_weight_update_max_error
        << "}\n"
        << "}\n";
    return passed ? 0 : 2;
}

int print_ttt_linear_inner_smoke_json(const Config& cfg, const char* program) {
    const std::string family = NFN_NATIVE_MODEL_FAMILY;
    const bool ttt_family = family.find("ttt") != std::string::npos || family == "unknown";
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
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightAccumulateFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LatentMseLossFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using AdamWFn = int (*)(
        float*, const float*, float*, float*, std::int64_t, float, float, float, float, float, float, float, void*);

    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightAccumulateFn linear_backward_weight_accumulate = nullptr;
    LatentMseLossFn latent_mse_loss = nullptr;
    FillFn fill = nullptr;
    AdamWFn adamw = nullptr;

    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    constexpr std::int64_t kRows = 2;
    constexpr std::int64_t kDim = 3;
    constexpr std::int64_t kElements = kRows * kDim;
    constexpr std::int64_t kWeightElements = kDim * kDim;

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
    float inner_forward_max_error = 0.0f;
    float inner_loss_max_error = 0.0f;
    float inner_grad_input_max_error = 0.0f;
    float inner_grad_weight_max_error = 0.0f;
    float inner_weight_update_max_error = 0.0f;

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

    if (!ttt_family) {
        error = "TTT smoke commands are only valid for ttt-family native preflights";
    } else {
        tile_handle = dlopen(tile_ops_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            const char* raw = dlerror();
            error = raw == nullptr ? "failed to load Tile ops library" : raw;
        } else {
            tile_ops_loaded = true;
            linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
            linear_backward_input = load_symbol<LinearBackwardInputFn>(
                tile_handle, "nfn_native_tile_linear_backward_input_float32");
            linear_backward_weight_accumulate = load_symbol<LinearBackwardWeightAccumulateFn>(
                tile_handle, "nfn_native_tile_linear_backward_weight_accumulate_float32");
            latent_mse_loss = load_symbol<LatentMseLossFn>(
                tile_handle, "nfn_native_tile_latent_mse_loss_float32");
            fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
            adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
            if (linear == nullptr || linear_backward_input == nullptr ||
                linear_backward_weight_accumulate == nullptr || latent_mse_loss == nullptr ||
                fill == nullptr || adamw == nullptr) {
                error = "Tile ops library is missing one or more TTT inner linear train-step symbols";
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
        const std::vector<float> hidden = {
            0.12f, -0.08f, 0.18f,
            -0.14f, 0.21f, 0.06f,
        };
        const std::vector<float> reconstruction_target = {
            0.03f, -0.01f, 0.05f,
            -0.02f, 0.04f, 0.02f,
        };
        std::vector<float> inner_weight(static_cast<std::size_t>(kWeightElements));
        for (std::size_t i = 0; i < inner_weight.size(); ++i) {
            inner_weight[i] = 0.035f * static_cast<float>(static_cast<int>(i % 7) - 3);
        }

        std::vector<float> expected_pred(static_cast<std::size_t>(kElements), 0.0f);
        for (std::int64_t row = 0; row < kRows; ++row) {
            for (std::int64_t out_d = 0; out_d < kDim; ++out_d) {
                float acc = 0.0f;
                for (std::int64_t in_d = 0; in_d < kDim; ++in_d) {
                    acc += hidden[static_cast<std::size_t>(row * kDim + in_d)] *
                        inner_weight[static_cast<std::size_t>(out_d * kDim + in_d)];
                }
                expected_pred[static_cast<std::size_t>(row * kDim + out_d)] = acc;
            }
        }
        std::vector<float> grad_pred(static_cast<std::size_t>(kElements), 0.0f);
        float expected_loss = 0.0f;
        for (std::size_t i = 0; i < expected_pred.size(); ++i) {
            const float diff = expected_pred[i] - reconstruction_target[i];
            expected_loss += diff * diff;
            grad_pred[i] = 2.0f * diff;
        }
        std::vector<float> expected_grad_input(static_cast<std::size_t>(kElements), 0.0f);
        std::vector<float> expected_grad_weight(inner_weight.size(), 0.0f);
        for (std::int64_t row = 0; row < kRows; ++row) {
            for (std::int64_t out_d = 0; out_d < kDim; ++out_d) {
                const float grad = grad_pred[static_cast<std::size_t>(row * kDim + out_d)];
                for (std::int64_t in_d = 0; in_d < kDim; ++in_d) {
                    expected_grad_weight[static_cast<std::size_t>(out_d * kDim + in_d)] +=
                        hidden[static_cast<std::size_t>(row * kDim + in_d)] * grad;
                    expected_grad_input[static_cast<std::size_t>(row * kDim + in_d)] +=
                        grad * inner_weight[static_cast<std::size_t>(out_d * kDim + in_d)];
                }
            }
        }

        float* d_hidden = nullptr;
        float* d_target = nullptr;
        float* d_inner_weight = nullptr;
        float* d_pred = nullptr;
        float* d_loss = nullptr;
        float* d_grad_pred = nullptr;
        float* d_grad_input = nullptr;
        float* d_grad_weight = nullptr;
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
        if (alloc(&d_hidden, hidden.size(), "hidden") &&
            alloc(&d_target, reconstruction_target.size(), "target") &&
            alloc(&d_inner_weight, inner_weight.size(), "inner_weight") &&
            alloc(&d_pred, kElements, "pred") &&
            alloc(&d_loss, 1, "loss") &&
            alloc(&d_grad_pred, grad_pred.size(), "grad_pred") &&
            alloc(&d_grad_input, kElements, "grad_input") &&
            alloc(&d_grad_weight, inner_weight.size(), "grad_weight") &&
            alloc(&d_exp_avg, inner_weight.size(), "exp_avg") &&
            alloc(&d_exp_avg_sq, inner_weight.size(), "exp_avg_sq")) {
            auto copy_float = [&](float* dst, const std::vector<float>& src, const std::string& name) {
                int status = cuda_memcpy(dst, src.data(), src.size() * sizeof(float), kCudaMemcpyHostToDevice);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy " + name + " H2D");
                    return false;
                }
                return true;
            };
            copy_float(d_hidden, hidden, "hidden") &&
                copy_float(d_target, reconstruction_target, "target") &&
                copy_float(d_inner_weight, inner_weight, "inner_weight") &&
                copy_float(d_grad_pred, grad_pred, "grad_pred");
            int status = 0;
            if (error.empty()) {
                status = fill(d_grad_weight, kWeightElements, 0.0f, nullptr);
                if (status == 0) {
                    status = fill(d_exp_avg, kWeightElements, 0.0f, nullptr);
                }
                if (status == 0) {
                    status = fill(d_exp_avg_sq, kWeightElements, 0.0f, nullptr);
                }
                if (status != 0) {
                    error = cuda_error(status, "zero TTT gradients/moments");
                }
            }
            if (error.empty()) {
                status = linear(d_hidden, d_inner_weight, nullptr, d_pred, kRows, kDim, kDim, false, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_float32 ttt inner");
                }
            }
            if (error.empty()) {
                status = latent_mse_loss(d_pred, d_target, d_loss, kElements, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_latent_mse_loss_float32 ttt inner");
                }
            }
            if (error.empty()) {
                status = linear_backward_input(d_grad_pred, d_inner_weight, d_grad_input, kRows, kDim, kDim, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_backward_input_float32 ttt inner");
                }
            }
            if (error.empty()) {
                status = linear_backward_weight_accumulate(d_hidden, d_grad_pred, d_grad_weight, kRows, kDim, kDim, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_backward_weight_accumulate_float32 ttt inner");
                }
            }
            if (error.empty()) {
                status = adamw(
                    d_inner_weight,
                    d_grad_weight,
                    d_exp_avg,
                    d_exp_avg_sq,
                    kWeightElements,
                    0.01f,
                    0.9f,
                    0.95f,
                    1e-8f,
                    0.02f,
                    0.1f,
                    std::sqrt(0.05f),
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_adamw_step_float32 ttt inner");
                }
            }
            if (error.empty()) {
                status = cuda_device_synchronize();
                if (status != 0) {
                    error = cuda_error(status, "cudaDeviceSynchronize");
                }
            }

            std::vector<float> actual_pred(static_cast<std::size_t>(kElements), 0.0f);
            std::vector<float> actual_loss(1, 0.0f);
            std::vector<float> actual_grad_input(static_cast<std::size_t>(kElements), 0.0f);
            std::vector<float> actual_grad_weight(inner_weight.size(), 0.0f);
            std::vector<float> actual_inner_weight(inner_weight.size(), 0.0f);
            auto copy_back_float = [&](std::vector<float>& dst, const float* src, const std::string& name) {
                int copy_status = cuda_memcpy(dst.data(), src, dst.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (copy_status != 0) {
                    error = cuda_error(copy_status, "cudaMemcpy " + name + " D2H");
                    return false;
                }
                return true;
            };
            if (error.empty()) {
                copy_back_float(actual_pred, d_pred, "pred") &&
                    copy_back_float(actual_loss, d_loss, "loss") &&
                    copy_back_float(actual_grad_input, d_grad_input, "grad_input") &&
                    copy_back_float(actual_grad_weight, d_grad_weight, "grad_weight") &&
                    copy_back_float(actual_inner_weight, d_inner_weight, "inner_weight");
            }

            std::vector<float> expected_inner_weight = inner_weight;
            for (std::size_t i = 0; i < expected_inner_weight.size(); ++i) {
                const float grad = expected_grad_weight[i];
                const float next_m = 0.1f * grad;
                const float next_v = 0.05f * grad * grad;
                const float denom = std::sqrt(next_v) / std::sqrt(0.05f) + 1e-8f;
                const float decayed = expected_inner_weight[i] * (1.0f - 0.01f * 0.02f);
                expected_inner_weight[i] = decayed - 0.01f * (next_m / 0.1f) / denom;
            }
            if (error.empty()) {
                inner_forward_max_error = max_abs_error(actual_pred, expected_pred);
                inner_loss_max_error = std::fabs(actual_loss[0] - expected_loss);
                inner_grad_input_max_error = max_abs_error(actual_grad_input, expected_grad_input);
                inner_grad_weight_max_error = max_abs_error(actual_grad_weight, expected_grad_weight);
                inner_weight_update_max_error = max_abs_error(actual_inner_weight, expected_inner_weight);
            }
            constexpr float kTolerance = 5e-5f;
            constexpr float kAdamTolerance = 2e-5f;
            passed = error.empty() &&
                inner_forward_max_error <= kTolerance &&
                inner_loss_max_error <= kTolerance &&
                inner_grad_input_max_error <= kTolerance &&
                inner_grad_weight_max_error <= kTolerance &&
                inner_weight_update_max_error <= kAdamTolerance;
            if (!passed && error.empty()) {
                error = "TTT inner linear train-step smoke exceeded tolerance";
            }
        }
    }
    free_allocated();
    close_handles();

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(NFN_NATIVE_MODEL_FAMILY) << "\",\n"
        << "  \"native_target\": \"" << json_escape(NFN_NATIVE_TARGET_NAME) << "\",\n"
        << "  \"smoke\": \"ttt_linear_inner_train_step_slice\",\n"
        << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(error) << "\",\n"
        << "  \"compiled_native_boundary\": true,\n"
        << "  \"torch_required\": false,\n"
        << "  \"graph_editor_tensor_flow\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_ops_lib) << "\",\n"
        << "  \"tile_ops_loaded\": " << (tile_ops_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"shape\": {\"rows\": " << kRows << ", \"inner_dim\": " << kDim << "},\n"
        << "  \"loop_composition_stages\": [\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_latent_mse_loss_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_accumulate_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"max_errors\": {"
        << "\"inner_forward\":" << inner_forward_max_error
        << ", \"inner_loss\":" << inner_loss_max_error
        << ", \"inner_grad_input\":" << inner_grad_input_max_error
        << ", \"inner_grad_weight\":" << inner_grad_weight_max_error
        << ", \"inner_weight_update\":" << inner_weight_update_max_error
        << "}\n"
        << "}\n";
    return passed ? 0 : 2;
}

int print_universal_recurrent_smoke_json(const Config& cfg, const char* program) {
    const std::string family = NFN_NATIVE_MODEL_FAMILY;
    const bool universal_family = family.find("universal") != std::string::npos || family == "unknown";
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
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightAccumulateFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LatentMseLossFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using AdamWFn = int (*)(
        float*, const float*, float*, float*, std::int64_t, float, float, float, float, float, float, float, void*);

    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightAccumulateFn linear_backward_weight_accumulate = nullptr;
    LatentMseLossFn latent_mse_loss = nullptr;
    FillFn fill = nullptr;
    AdamWFn adamw = nullptr;

    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    constexpr std::int64_t kRows = 2;
    constexpr std::int64_t kDim = 4;
    constexpr std::int64_t kElements = kRows * kDim;
    constexpr std::int64_t kWeightElements = kDim * kDim;

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
    float recurrent_forward_max_error = 0.0f;
    float recurrent_loss_max_error = 0.0f;
    float recurrent_grad_input_max_error = 0.0f;
    float recurrent_grad_weight_max_error = 0.0f;
    float recurrent_weight_update_max_error = 0.0f;

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

    if (!universal_family) {
        error = "Universal smoke commands are only valid for universal-family native preflights";
    } else {
        tile_handle = dlopen(tile_ops_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            const char* raw = dlerror();
            error = raw == nullptr ? "failed to load Tile ops library" : raw;
        } else {
            tile_ops_loaded = true;
            linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
            linear_backward_input = load_symbol<LinearBackwardInputFn>(
                tile_handle, "nfn_native_tile_linear_backward_input_float32");
            linear_backward_weight_accumulate = load_symbol<LinearBackwardWeightAccumulateFn>(
                tile_handle, "nfn_native_tile_linear_backward_weight_accumulate_float32");
            latent_mse_loss = load_symbol<LatentMseLossFn>(
                tile_handle, "nfn_native_tile_latent_mse_loss_float32");
            fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
            adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
            if (linear == nullptr || linear_backward_input == nullptr ||
                linear_backward_weight_accumulate == nullptr || latent_mse_loss == nullptr ||
                fill == nullptr || adamw == nullptr) {
                error = "Tile ops library is missing one or more universal recurrent train-step symbols";
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
        const std::vector<float> recurrent_state = {
            0.11f, -0.07f, 0.19f, 0.03f,
            -0.13f, 0.22f, -0.04f, 0.09f,
        };
        const std::vector<float> target_state = {
            0.02f, -0.01f, 0.04f, 0.03f,
            -0.02f, 0.05f, -0.01f, 0.01f,
        };
        std::vector<float> recurrent_weight(static_cast<std::size_t>(kWeightElements));
        for (std::size_t i = 0; i < recurrent_weight.size(); ++i) {
            recurrent_weight[i] = 0.03f * static_cast<float>(static_cast<int>(i % 9) - 4);
        }

        std::vector<float> expected_next(static_cast<std::size_t>(kElements), 0.0f);
        for (std::int64_t row = 0; row < kRows; ++row) {
            for (std::int64_t out_d = 0; out_d < kDim; ++out_d) {
                float acc = 0.0f;
                for (std::int64_t in_d = 0; in_d < kDim; ++in_d) {
                    acc += recurrent_state[static_cast<std::size_t>(row * kDim + in_d)] *
                        recurrent_weight[static_cast<std::size_t>(out_d * kDim + in_d)];
                }
                expected_next[static_cast<std::size_t>(row * kDim + out_d)] = acc;
            }
        }
        std::vector<float> grad_next(static_cast<std::size_t>(kElements), 0.0f);
        float expected_loss = 0.0f;
        for (std::size_t i = 0; i < expected_next.size(); ++i) {
            const float diff = expected_next[i] - target_state[i];
            expected_loss += diff * diff;
            grad_next[i] = 2.0f * diff;
        }
        std::vector<float> expected_grad_state(static_cast<std::size_t>(kElements), 0.0f);
        std::vector<float> expected_grad_weight(recurrent_weight.size(), 0.0f);
        for (std::int64_t row = 0; row < kRows; ++row) {
            for (std::int64_t out_d = 0; out_d < kDim; ++out_d) {
                const float grad = grad_next[static_cast<std::size_t>(row * kDim + out_d)];
                for (std::int64_t in_d = 0; in_d < kDim; ++in_d) {
                    expected_grad_weight[static_cast<std::size_t>(out_d * kDim + in_d)] +=
                        recurrent_state[static_cast<std::size_t>(row * kDim + in_d)] * grad;
                    expected_grad_state[static_cast<std::size_t>(row * kDim + in_d)] +=
                        grad * recurrent_weight[static_cast<std::size_t>(out_d * kDim + in_d)];
                }
            }
        }

        float* d_state = nullptr;
        float* d_target = nullptr;
        float* d_weight = nullptr;
        float* d_next = nullptr;
        float* d_loss = nullptr;
        float* d_grad_next = nullptr;
        float* d_grad_state = nullptr;
        float* d_grad_weight = nullptr;
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
        if (alloc(&d_state, recurrent_state.size(), "state") &&
            alloc(&d_target, target_state.size(), "target") &&
            alloc(&d_weight, recurrent_weight.size(), "recurrent_weight") &&
            alloc(&d_next, kElements, "next") &&
            alloc(&d_loss, 1, "loss") &&
            alloc(&d_grad_next, grad_next.size(), "grad_next") &&
            alloc(&d_grad_state, kElements, "grad_state") &&
            alloc(&d_grad_weight, recurrent_weight.size(), "grad_weight") &&
            alloc(&d_exp_avg, recurrent_weight.size(), "exp_avg") &&
            alloc(&d_exp_avg_sq, recurrent_weight.size(), "exp_avg_sq")) {
            auto copy_float = [&](float* dst, const std::vector<float>& src, const std::string& name) {
                int status = cuda_memcpy(dst, src.data(), src.size() * sizeof(float), kCudaMemcpyHostToDevice);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy " + name + " H2D");
                    return false;
                }
                return true;
            };
            copy_float(d_state, recurrent_state, "state") &&
                copy_float(d_target, target_state, "target") &&
                copy_float(d_weight, recurrent_weight, "recurrent_weight") &&
                copy_float(d_grad_next, grad_next, "grad_next");
            int status = 0;
            if (error.empty()) {
                status = fill(d_grad_weight, kWeightElements, 0.0f, nullptr);
                if (status == 0) {
                    status = fill(d_exp_avg, kWeightElements, 0.0f, nullptr);
                }
                if (status == 0) {
                    status = fill(d_exp_avg_sq, kWeightElements, 0.0f, nullptr);
                }
                if (status != 0) {
                    error = cuda_error(status, "zero universal gradients/moments");
                }
            }
            if (error.empty()) {
                status = linear(d_state, d_weight, nullptr, d_next, kRows, kDim, kDim, false, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_float32 universal recurrent");
                }
            }
            if (error.empty()) {
                status = latent_mse_loss(d_next, d_target, d_loss, kElements, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_latent_mse_loss_float32 universal recurrent");
                }
            }
            if (error.empty()) {
                status = linear_backward_input(d_grad_next, d_weight, d_grad_state, kRows, kDim, kDim, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_backward_input_float32 universal recurrent");
                }
            }
            if (error.empty()) {
                status = linear_backward_weight_accumulate(d_state, d_grad_next, d_grad_weight, kRows, kDim, kDim, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_backward_weight_accumulate_float32 universal recurrent");
                }
            }
            if (error.empty()) {
                status = adamw(
                    d_weight,
                    d_grad_weight,
                    d_exp_avg,
                    d_exp_avg_sq,
                    kWeightElements,
                    0.01f,
                    0.9f,
                    0.95f,
                    1e-8f,
                    0.02f,
                    0.1f,
                    std::sqrt(0.05f),
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_adamw_step_float32 universal recurrent");
                }
            }
            if (error.empty()) {
                status = cuda_device_synchronize();
                if (status != 0) {
                    error = cuda_error(status, "cudaDeviceSynchronize");
                }
            }

            std::vector<float> actual_next(static_cast<std::size_t>(kElements), 0.0f);
            std::vector<float> actual_loss(1, 0.0f);
            std::vector<float> actual_grad_state(static_cast<std::size_t>(kElements), 0.0f);
            std::vector<float> actual_grad_weight(recurrent_weight.size(), 0.0f);
            std::vector<float> actual_weight(recurrent_weight.size(), 0.0f);
            auto copy_back_float = [&](std::vector<float>& dst, const float* src, const std::string& name) {
                int copy_status = cuda_memcpy(dst.data(), src, dst.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (copy_status != 0) {
                    error = cuda_error(copy_status, "cudaMemcpy " + name + " D2H");
                    return false;
                }
                return true;
            };
            if (error.empty()) {
                copy_back_float(actual_next, d_next, "next") &&
                    copy_back_float(actual_loss, d_loss, "loss") &&
                    copy_back_float(actual_grad_state, d_grad_state, "grad_state") &&
                    copy_back_float(actual_grad_weight, d_grad_weight, "grad_weight") &&
                    copy_back_float(actual_weight, d_weight, "recurrent_weight");
            }

            std::vector<float> expected_weight = recurrent_weight;
            for (std::size_t i = 0; i < expected_weight.size(); ++i) {
                const float grad = expected_grad_weight[i];
                const float next_m = 0.1f * grad;
                const float next_v = 0.05f * grad * grad;
                const float denom = std::sqrt(next_v) / std::sqrt(0.05f) + 1e-8f;
                const float decayed = expected_weight[i] * (1.0f - 0.01f * 0.02f);
                expected_weight[i] = decayed - 0.01f * (next_m / 0.1f) / denom;
            }
            if (error.empty()) {
                recurrent_forward_max_error = max_abs_error(actual_next, expected_next);
                recurrent_loss_max_error = std::fabs(actual_loss[0] - expected_loss);
                recurrent_grad_input_max_error = max_abs_error(actual_grad_state, expected_grad_state);
                recurrent_grad_weight_max_error = max_abs_error(actual_grad_weight, expected_grad_weight);
                recurrent_weight_update_max_error = max_abs_error(actual_weight, expected_weight);
            }
            constexpr float kTolerance = 5e-5f;
            constexpr float kAdamTolerance = 2e-5f;
            passed = error.empty() &&
                recurrent_forward_max_error <= kTolerance &&
                recurrent_loss_max_error <= kTolerance &&
                recurrent_grad_input_max_error <= kTolerance &&
                recurrent_grad_weight_max_error <= kTolerance &&
                recurrent_weight_update_max_error <= kAdamTolerance;
            if (!passed && error.empty()) {
                error = "Universal recurrent train-step smoke exceeded tolerance";
            }
        }
    }
    free_allocated();
    close_handles();

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(NFN_NATIVE_MODEL_FAMILY) << "\",\n"
        << "  \"native_target\": \"" << json_escape(NFN_NATIVE_TARGET_NAME) << "\",\n"
        << "  \"smoke\": \"universal_recurrent_train_step_slice\",\n"
        << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(error) << "\",\n"
        << "  \"compiled_native_boundary\": true,\n"
        << "  \"torch_required\": false,\n"
        << "  \"graph_editor_tensor_flow\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_ops_lib) << "\",\n"
        << "  \"tile_ops_loaded\": " << (tile_ops_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"shape\": {\"rows\": " << kRows << ", \"recurrent_dim\": " << kDim << "},\n"
        << "  \"loop_composition_stages\": [\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_latent_mse_loss_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_accumulate_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"max_errors\": {"
        << "\"recurrent_forward\":" << recurrent_forward_max_error
        << ", \"recurrent_loss\":" << recurrent_loss_max_error
        << ", \"recurrent_grad_input\":" << recurrent_grad_input_max_error
        << ", \"recurrent_grad_weight\":" << recurrent_grad_weight_max_error
        << ", \"recurrent_weight_update\":" << recurrent_weight_update_max_error
        << "}\n"
        << "}\n";
    return passed ? 0 : 2;
}

int print_universal_act_halt_smoke_json(const Config& cfg, const char* program) {
    const std::string family = NFN_NATIVE_MODEL_FAMILY;
    const bool universal_family = family.find("universal") != std::string::npos || family == "unknown";
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
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightAccumulateFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using ActWeightedSumFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using ActHaltingBceGradFn = int (*)(
        const float*, const float*, float*, float*, float*, std::int64_t, void*);
    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using AdamWFn = int (*)(
        float*, const float*, float*, float*, std::int64_t, float, float, float, float, float, float, float, void*);

    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightAccumulateFn linear_backward_weight_accumulate = nullptr;
    ActWeightedSumFn act_weighted_sum = nullptr;
    ActHaltingBceGradFn act_halting_bce_grad = nullptr;
    FillFn fill = nullptr;
    AdamWFn adamw = nullptr;

    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    constexpr std::int64_t kBatch = 2;
    constexpr std::int64_t kSteps = 3;
    constexpr std::int64_t kDim = 4;
    constexpr std::int64_t kRows = kBatch * kSteps;
    constexpr std::int64_t kStateElements = kRows * kDim;
    constexpr std::int64_t kHaltWeightElements = kDim;

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
    auto sigmoid = [](float x) {
        return 1.0f / (1.0f + std::exp(-x));
    };

    bool passed = false;
    float halt_logits_max_error = 0.0f;
    float halt_loss_max_error = 0.0f;
    float halt_grad_input_max_error = 0.0f;
    float halt_grad_weight_max_error = 0.0f;
    float halt_weight_update_max_error = 0.0f;
    float act_weighted_sum_max_error = 0.0f;

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

    if (!universal_family) {
        error = "Universal smoke commands are only valid for universal-family native preflights";
    } else {
        tile_handle = dlopen(tile_ops_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            const char* raw = dlerror();
            error = raw == nullptr ? "failed to load Tile ops library" : raw;
        } else {
            tile_ops_loaded = true;
            linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
            linear_backward_input = load_symbol<LinearBackwardInputFn>(
                tile_handle, "nfn_native_tile_linear_backward_input_float32");
            linear_backward_weight_accumulate = load_symbol<LinearBackwardWeightAccumulateFn>(
                tile_handle, "nfn_native_tile_linear_backward_weight_accumulate_float32");
            act_weighted_sum = load_symbol<ActWeightedSumFn>(
                tile_handle, "nfn_native_tile_act_weighted_sum_float32");
            act_halting_bce_grad = load_symbol<ActHaltingBceGradFn>(
                tile_handle, "nfn_native_tile_act_halting_bce_grad_float32");
            fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
            adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
            if (linear == nullptr || linear_backward_input == nullptr ||
                linear_backward_weight_accumulate == nullptr || act_weighted_sum == nullptr ||
                act_halting_bce_grad == nullptr || fill == nullptr || adamw == nullptr) {
                error = "Tile ops library is missing one or more universal ACT halt train-step symbols";
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
        const std::vector<float> hidden_states = {
            0.11f, -0.07f, 0.19f, 0.03f,
            -0.13f, 0.22f, -0.04f, 0.09f,
            0.05f, 0.08f, -0.10f, 0.17f,
            0.03f, -0.02f, 0.07f, -0.11f,
            0.14f, 0.06f, -0.05f, 0.12f,
            -0.09f, 0.15f, 0.02f, -0.04f,
        };
        const std::vector<float> halt_targets = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f};
        const std::vector<float> halt_weight = {0.18f, -0.09f, 0.07f, 0.13f};

        std::vector<float> expected_logits(static_cast<std::size_t>(kRows), 0.0f);
        std::vector<float> halt_probs(static_cast<std::size_t>(kRows), 0.0f);
        std::vector<float> grad_logits(static_cast<std::size_t>(kRows), 0.0f);
        float expected_loss = 0.0f;
        for (std::int64_t row = 0; row < kRows; ++row) {
            float logit = 0.0f;
            for (std::int64_t dim = 0; dim < kDim; ++dim) {
                logit += hidden_states[static_cast<std::size_t>(row * kDim + dim)] *
                    halt_weight[static_cast<std::size_t>(dim)];
            }
            const float prob = sigmoid(logit);
            const float target = halt_targets[static_cast<std::size_t>(row)];
            expected_logits[static_cast<std::size_t>(row)] = logit;
            halt_probs[static_cast<std::size_t>(row)] = prob;
            grad_logits[static_cast<std::size_t>(row)] = prob - target;
            expected_loss += -target * std::log(std::max(prob, 1e-6f)) -
                (1.0f - target) * std::log(std::max(1.0f - prob, 1e-6f));
        }

        std::vector<float> expected_grad_hidden(static_cast<std::size_t>(kStateElements), 0.0f);
        std::vector<float> expected_grad_weight(static_cast<std::size_t>(kHaltWeightElements), 0.0f);
        for (std::int64_t row = 0; row < kRows; ++row) {
            const float grad = grad_logits[static_cast<std::size_t>(row)];
            for (std::int64_t dim = 0; dim < kDim; ++dim) {
                expected_grad_weight[static_cast<std::size_t>(dim)] +=
                    hidden_states[static_cast<std::size_t>(row * kDim + dim)] * grad;
                expected_grad_hidden[static_cast<std::size_t>(row * kDim + dim)] =
                    grad * halt_weight[static_cast<std::size_t>(dim)];
            }
        }
        std::vector<float> expected_act(static_cast<std::size_t>(kBatch * kDim), 0.0f);
        for (std::int64_t batch = 0; batch < kBatch; ++batch) {
            for (std::int64_t step = 0; step < kSteps; ++step) {
                const std::int64_t row = batch * kSteps + step;
                const float weight = halt_probs[static_cast<std::size_t>(row)];
                for (std::int64_t dim = 0; dim < kDim; ++dim) {
                    expected_act[static_cast<std::size_t>(batch * kDim + dim)] +=
                        hidden_states[static_cast<std::size_t>(row * kDim + dim)] * weight;
                }
            }
        }

        float* d_hidden = nullptr;
        float* d_halt_weight = nullptr;
        float* d_halt_targets = nullptr;
        float* d_logits = nullptr;
        float* d_loss_partials = nullptr;
        float* d_halt_probs = nullptr;
        float* d_grad_logits = nullptr;
        float* d_grad_hidden = nullptr;
        float* d_grad_weight = nullptr;
        float* d_exp_avg = nullptr;
        float* d_exp_avg_sq = nullptr;
        float* d_act = nullptr;
        auto alloc = [&](float** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(float));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        if (alloc(&d_hidden, hidden_states.size(), "hidden") &&
            alloc(&d_halt_weight, halt_weight.size(), "halt_weight") &&
            alloc(&d_halt_targets, halt_targets.size(), "halt_targets") &&
            alloc(&d_logits, expected_logits.size(), "halt_logits") &&
            alloc(&d_loss_partials, 1, "halt_loss_partials") &&
            alloc(&d_halt_probs, halt_probs.size(), "halt_probs") &&
            alloc(&d_grad_logits, grad_logits.size(), "grad_logits") &&
            alloc(&d_grad_hidden, expected_grad_hidden.size(), "grad_hidden") &&
            alloc(&d_grad_weight, expected_grad_weight.size(), "grad_weight") &&
            alloc(&d_exp_avg, halt_weight.size(), "exp_avg") &&
            alloc(&d_exp_avg_sq, halt_weight.size(), "exp_avg_sq") &&
            alloc(&d_act, expected_act.size(), "act_weighted_sum")) {
            auto copy_float = [&](float* dst, const std::vector<float>& src, const std::string& name) {
                int status = cuda_memcpy(dst, src.data(), src.size() * sizeof(float), kCudaMemcpyHostToDevice);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy " + name + " H2D");
                    return false;
                }
                return true;
            };
            copy_float(d_hidden, hidden_states, "hidden") &&
                copy_float(d_halt_weight, halt_weight, "halt_weight") &&
                copy_float(d_halt_targets, halt_targets, "halt_targets");

            int status = 0;
            if (error.empty()) {
                status = fill(d_grad_weight, kHaltWeightElements, 0.0f, nullptr);
                if (status == 0) {
                    status = fill(d_exp_avg, kHaltWeightElements, 0.0f, nullptr);
                }
                if (status == 0) {
                    status = fill(d_exp_avg_sq, kHaltWeightElements, 0.0f, nullptr);
                }
                if (status != 0) {
                    error = cuda_error(status, "zero universal ACT gradients/moments");
                }
            }
            if (error.empty()) {
                status = linear(d_hidden, d_halt_weight, nullptr, d_logits, kRows, kDim, 1, false, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_float32 universal ACT halt gate");
                }
            }
            if (error.empty()) {
                status = act_halting_bce_grad(
                    d_logits, d_halt_targets, d_loss_partials, d_grad_logits, d_halt_probs, kRows, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_act_halting_bce_grad_float32 universal ACT");
                }
            }
            if (error.empty()) {
                status = act_weighted_sum(d_hidden, d_halt_probs, d_act, kBatch, kSteps, kDim, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_act_weighted_sum_float32 universal ACT");
                }
            }
            if (error.empty()) {
                status = linear_backward_input(d_grad_logits, d_halt_weight, d_grad_hidden, kRows, kDim, 1, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_backward_input_float32 universal ACT");
                }
            }
            if (error.empty()) {
                status = linear_backward_weight_accumulate(d_hidden, d_grad_logits, d_grad_weight, kRows, kDim, 1, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_backward_weight_accumulate_float32 universal ACT");
                }
            }
            if (error.empty()) {
                status = adamw(
                    d_halt_weight,
                    d_grad_weight,
                    d_exp_avg,
                    d_exp_avg_sq,
                    kHaltWeightElements,
                    0.01f,
                    0.9f,
                    0.95f,
                    1e-8f,
                    0.02f,
                    0.1f,
                    std::sqrt(0.05f),
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_adamw_step_float32 universal ACT");
                }
            }
            if (error.empty()) {
                status = cuda_device_synchronize();
                if (status != 0) {
                    error = cuda_error(status, "cudaDeviceSynchronize");
                }
            }

            std::vector<float> actual_logits(expected_logits.size(), 0.0f);
            std::vector<float> actual_loss_partials(1, 0.0f);
            std::vector<float> actual_halt_probs(halt_probs.size(), 0.0f);
            std::vector<float> actual_grad_hidden(expected_grad_hidden.size(), 0.0f);
            std::vector<float> actual_grad_weight(expected_grad_weight.size(), 0.0f);
            std::vector<float> actual_weight(halt_weight.size(), 0.0f);
            std::vector<float> actual_act(expected_act.size(), 0.0f);
            auto copy_back_float = [&](std::vector<float>& dst, const float* src, const std::string& name) {
                int copy_status = cuda_memcpy(dst.data(), src, dst.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (copy_status != 0) {
                    error = cuda_error(copy_status, "cudaMemcpy " + name + " D2H");
                    return false;
                }
                return true;
            };
            if (error.empty()) {
                copy_back_float(actual_logits, d_logits, "halt_logits") &&
                    copy_back_float(actual_loss_partials, d_loss_partials, "halt_loss_partials") &&
                    copy_back_float(actual_halt_probs, d_halt_probs, "halt_probs") &&
                    copy_back_float(actual_grad_hidden, d_grad_hidden, "grad_hidden") &&
                    copy_back_float(actual_grad_weight, d_grad_weight, "grad_weight") &&
                    copy_back_float(actual_weight, d_halt_weight, "halt_weight") &&
                    copy_back_float(actual_act, d_act, "act_weighted_sum");
            }
            float actual_loss = 0.0f;
            for (float item : actual_loss_partials) {
                actual_loss += item;
            }

            std::vector<float> expected_weight = halt_weight;
            for (std::size_t i = 0; i < expected_weight.size(); ++i) {
                const float grad = expected_grad_weight[i];
                const float next_m = 0.1f * grad;
                const float next_v = 0.05f * grad * grad;
                const float denom = std::sqrt(next_v) / std::sqrt(0.05f) + 1e-8f;
                const float decayed = expected_weight[i] * (1.0f - 0.01f * 0.02f);
                expected_weight[i] = decayed - 0.01f * (next_m / 0.1f) / denom;
            }
            if (error.empty()) {
                halt_logits_max_error = max_abs_error(actual_logits, expected_logits);
                halt_loss_max_error = std::fabs(actual_loss - expected_loss);
                halt_grad_input_max_error = max_abs_error(actual_grad_hidden, expected_grad_hidden);
                halt_grad_weight_max_error = max_abs_error(actual_grad_weight, expected_grad_weight);
                halt_weight_update_max_error = max_abs_error(actual_weight, expected_weight);
                act_weighted_sum_max_error =
                    std::max(max_abs_error(actual_act, expected_act), max_abs_error(actual_halt_probs, halt_probs));
            }
            constexpr float kTolerance = 5e-5f;
            constexpr float kAdamTolerance = 2e-5f;
            passed = error.empty() &&
                halt_logits_max_error <= kTolerance &&
                halt_loss_max_error <= kTolerance &&
                halt_grad_input_max_error <= kTolerance &&
                halt_grad_weight_max_error <= kTolerance &&
                halt_weight_update_max_error <= kAdamTolerance &&
                act_weighted_sum_max_error <= kTolerance;
            if (!passed && error.empty()) {
                error = "Universal ACT halt train-step smoke exceeded tolerance";
            }
        }
    }
    free_allocated();
    close_handles();

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(NFN_NATIVE_MODEL_FAMILY) << "\",\n"
        << "  \"native_target\": \"" << json_escape(NFN_NATIVE_TARGET_NAME) << "\",\n"
        << "  \"smoke\": \"universal_act_halt_train_step_slice\",\n"
        << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(error) << "\",\n"
        << "  \"compiled_native_boundary\": true,\n"
        << "  \"torch_required\": false,\n"
        << "  \"graph_editor_tensor_flow\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_ops_lib) << "\",\n"
        << "  \"tile_ops_loaded\": " << (tile_ops_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"shape\": {\"batch\": " << kBatch << ", \"steps\": " << kSteps << ", \"dim\": " << kDim << "},\n"
        << "  \"loop_composition_stages\": [\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_act_halting_bce_grad_float32\",\n"
        << "    \"nfn_native_tile_act_weighted_sum_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_accumulate_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"max_errors\": {"
        << "\"halt_logits\":" << halt_logits_max_error
        << ", \"halt_loss\":" << halt_loss_max_error
        << ", \"halt_grad_input\":" << halt_grad_input_max_error
        << ", \"halt_grad_weight\":" << halt_grad_weight_max_error
        << ", \"halt_weight_update\":" << halt_weight_update_max_error
        << ", \"act_weighted_sum\":" << act_weighted_sum_max_error
        << "}\n"
        << "}\n";
    return passed ? 0 : 2;
}

int print_hnet_byte_patch_smoke_json(const Config& cfg, const char* program) {
    const std::string family = NFN_NATIVE_MODEL_FAMILY;
    const bool hnet_family = family.find("hnet") != std::string::npos || family == "unknown";
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
    using BytePatchEmbedFn = int (*)(
        const std::int64_t*, const float*, const float*, float*, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using BytePatchMergeFn = int (*)(
        const float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightAccumulateFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LatentMseLossFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using AdamWFn = int (*)(
        float*, const float*, float*, float*, std::int64_t, float, float, float, float, float, float, float, void*);

    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;
    BytePatchEmbedFn byte_patch_embed = nullptr;
    BytePatchMergeFn byte_patch_merge = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightAccumulateFn linear_backward_weight_accumulate = nullptr;
    LatentMseLossFn latent_mse_loss = nullptr;
    FillFn fill = nullptr;
    AdamWFn adamw = nullptr;

    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    constexpr std::int64_t kBatch = 1;
    constexpr std::int64_t kSeqLen = 4;
    constexpr std::int64_t kDim = 2;
    constexpr std::int64_t kPatchSize = 2;
    constexpr std::int64_t kStride = 2;
    constexpr std::int64_t kOutLen = 2;
    constexpr std::int64_t kVocab = 5;
    constexpr std::int64_t kPatchElements = kBatch * kOutLen * kDim;
    constexpr std::int64_t kMergedElements = kBatch * kSeqLen * kDim;
    constexpr std::int64_t kHeadWeightElements = kDim * kDim;
    constexpr std::int64_t kProjectionElements = kDim * kDim * kPatchSize;

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
    float byte_patch_embed_max_error = 0.0f;
    float byte_patch_merge_max_error = 0.0f;
    float head_forward_max_error = 0.0f;
    float hnet_loss_max_error = 0.0f;
    float head_grad_hidden_max_error = 0.0f;
    float head_grad_weight_max_error = 0.0f;
    float head_weight_update_max_error = 0.0f;

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

    if (!hnet_family) {
        error = "HNet smoke commands are only valid for hnet-family native preflights";
    } else {
        tile_handle = dlopen(tile_ops_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            const char* raw = dlerror();
            error = raw == nullptr ? "failed to load Tile ops library" : raw;
        } else {
            tile_ops_loaded = true;
            byte_patch_embed = load_symbol<BytePatchEmbedFn>(tile_handle, "nfn_native_tile_byte_patch_embed_float32");
            byte_patch_merge = load_symbol<BytePatchMergeFn>(tile_handle, "nfn_native_tile_byte_patch_merge_float32");
            linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
            linear_backward_input = load_symbol<LinearBackwardInputFn>(
                tile_handle, "nfn_native_tile_linear_backward_input_float32");
            linear_backward_weight_accumulate = load_symbol<LinearBackwardWeightAccumulateFn>(
                tile_handle, "nfn_native_tile_linear_backward_weight_accumulate_float32");
            latent_mse_loss = load_symbol<LatentMseLossFn>(
                tile_handle, "nfn_native_tile_latent_mse_loss_float32");
            fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
            adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
            if (byte_patch_embed == nullptr || byte_patch_merge == nullptr || linear == nullptr ||
                linear_backward_input == nullptr || linear_backward_weight_accumulate == nullptr ||
                latent_mse_loss == nullptr || fill == nullptr || adamw == nullptr) {
                error = "Tile ops library is missing one or more HNet byte patch train-step symbols";
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
        const std::vector<std::int64_t> tokens = {0, 1, 2, 3};
        const std::vector<float> embedding = {
            0.10f, -0.05f,
            0.20f, 0.03f,
            -0.08f, 0.12f,
            0.04f, -0.15f,
            0.06f, 0.09f,
        };
        std::vector<float> projection(static_cast<std::size_t>(kProjectionElements));
        std::vector<float> head_weight(static_cast<std::size_t>(kHeadWeightElements));
        for (std::size_t i = 0; i < projection.size(); ++i) {
            projection[i] = 0.025f * static_cast<float>(static_cast<int>(i % 7) - 3);
        }
        for (std::size_t i = 0; i < head_weight.size(); ++i) {
            head_weight[i] = -0.03f * static_cast<float>(static_cast<int>(i % 5) - 2);
        }
        const std::vector<float> target = {
            0.02f, -0.01f,
            0.03f, 0.01f,
            -0.02f, 0.04f,
            0.01f, -0.03f,
        };

        std::vector<float> expected_patch(static_cast<std::size_t>(kPatchElements), 0.0f);
        for (std::int64_t patch = 0; patch < kOutLen; ++patch) {
            for (std::int64_t out_c = 0; out_c < kDim; ++out_c) {
                float acc = 0.0f;
                for (std::int64_t k = 0; k < kPatchSize; ++k) {
                    const std::int64_t token_pos = patch * kStride + k;
                    if (token_pos >= kSeqLen) {
                        continue;
                    }
                    std::int64_t token_id = tokens[static_cast<std::size_t>(token_pos)];
                    token_id = std::max<std::int64_t>(0, std::min<std::int64_t>(token_id, kVocab - 1));
                    for (std::int64_t in_c = 0; in_c < kDim; ++in_c) {
                        acc += embedding[static_cast<std::size_t>(token_id * kDim + in_c)] *
                            projection[static_cast<std::size_t>((out_c * kDim + in_c) * kPatchSize + k)];
                    }
                }
                expected_patch[static_cast<std::size_t>(patch * kDim + out_c)] = acc;
            }
        }
        std::vector<float> expected_merged(static_cast<std::size_t>(kMergedElements), 0.0f);
        for (std::int64_t t = 0; t < kSeqLen; ++t) {
            const std::int64_t src_t = (t * kOutLen) / kSeqLen;
            for (std::int64_t d = 0; d < kDim; ++d) {
                expected_merged[static_cast<std::size_t>(t * kDim + d)] =
                    expected_patch[static_cast<std::size_t>(src_t * kDim + d)];
            }
        }
        std::vector<float> expected_pred(static_cast<std::size_t>(kMergedElements), 0.0f);
        for (std::int64_t row = 0; row < kSeqLen; ++row) {
            for (std::int64_t out_d = 0; out_d < kDim; ++out_d) {
                for (std::int64_t in_d = 0; in_d < kDim; ++in_d) {
                    expected_pred[static_cast<std::size_t>(row * kDim + out_d)] +=
                        expected_merged[static_cast<std::size_t>(row * kDim + in_d)] *
                        head_weight[static_cast<std::size_t>(out_d * kDim + in_d)];
                }
            }
        }
        std::vector<float> grad_pred(static_cast<std::size_t>(kMergedElements), 0.0f);
        float expected_loss = 0.0f;
        for (std::size_t i = 0; i < expected_pred.size(); ++i) {
            const float diff = expected_pred[i] - target[i];
            expected_loss += diff * diff;
            grad_pred[i] = 2.0f * diff;
        }
        std::vector<float> expected_grad_merged(static_cast<std::size_t>(kMergedElements), 0.0f);
        std::vector<float> expected_grad_head_weight(head_weight.size(), 0.0f);
        for (std::int64_t row = 0; row < kSeqLen; ++row) {
            for (std::int64_t out_d = 0; out_d < kDim; ++out_d) {
                const float grad = grad_pred[static_cast<std::size_t>(row * kDim + out_d)];
                for (std::int64_t in_d = 0; in_d < kDim; ++in_d) {
                    expected_grad_head_weight[static_cast<std::size_t>(out_d * kDim + in_d)] +=
                        expected_merged[static_cast<std::size_t>(row * kDim + in_d)] * grad;
                    expected_grad_merged[static_cast<std::size_t>(row * kDim + in_d)] +=
                        grad * head_weight[static_cast<std::size_t>(out_d * kDim + in_d)];
                }
            }
        }

        std::int64_t* d_tokens = nullptr;
        float* d_embedding = nullptr;
        float* d_projection = nullptr;
        float* d_patch = nullptr;
        float* d_merged = nullptr;
        float* d_head_weight = nullptr;
        float* d_pred = nullptr;
        float* d_target = nullptr;
        float* d_loss = nullptr;
        float* d_grad_pred = nullptr;
        float* d_grad_merged = nullptr;
        float* d_grad_head_weight = nullptr;
        float* d_exp_avg = nullptr;
        float* d_exp_avg_sq = nullptr;
        auto alloc_float = [&](float** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(float));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        auto alloc_i64 = [&](std::int64_t** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(std::int64_t));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        if (alloc_i64(&d_tokens, tokens.size(), "tokens") &&
            alloc_float(&d_embedding, embedding.size(), "embedding") &&
            alloc_float(&d_projection, projection.size(), "projection") &&
            alloc_float(&d_patch, kPatchElements, "patch") &&
            alloc_float(&d_merged, kMergedElements, "merged") &&
            alloc_float(&d_head_weight, head_weight.size(), "head_weight") &&
            alloc_float(&d_pred, kMergedElements, "pred") &&
            alloc_float(&d_target, target.size(), "target") &&
            alloc_float(&d_loss, 1, "loss") &&
            alloc_float(&d_grad_pred, grad_pred.size(), "grad_pred") &&
            alloc_float(&d_grad_merged, kMergedElements, "grad_merged") &&
            alloc_float(&d_grad_head_weight, head_weight.size(), "grad_head_weight") &&
            alloc_float(&d_exp_avg, head_weight.size(), "exp_avg") &&
            alloc_float(&d_exp_avg_sq, head_weight.size(), "exp_avg_sq")) {
            auto copy_float = [&](float* dst, const std::vector<float>& src, const std::string& name) {
                int status = cuda_memcpy(dst, src.data(), src.size() * sizeof(float), kCudaMemcpyHostToDevice);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy " + name + " H2D");
                    return false;
                }
                return true;
            };
            auto copy_i64 = [&](std::int64_t* dst, const std::vector<std::int64_t>& src, const std::string& name) {
                int status = cuda_memcpy(dst, src.data(), src.size() * sizeof(std::int64_t), kCudaMemcpyHostToDevice);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy " + name + " H2D");
                    return false;
                }
                return true;
            };
            copy_i64(d_tokens, tokens, "tokens") &&
                copy_float(d_embedding, embedding, "embedding") &&
                copy_float(d_projection, projection, "projection") &&
                copy_float(d_head_weight, head_weight, "head_weight") &&
                copy_float(d_target, target, "target") &&
                copy_float(d_grad_pred, grad_pred, "grad_pred");
            int status = 0;
            if (error.empty()) {
                status = fill(d_grad_head_weight, kHeadWeightElements, 0.0f, nullptr);
                if (status == 0) {
                    status = fill(d_exp_avg, kHeadWeightElements, 0.0f, nullptr);
                }
                if (status == 0) {
                    status = fill(d_exp_avg_sq, kHeadWeightElements, 0.0f, nullptr);
                }
                if (status != 0) {
                    error = cuda_error(status, "zero HNet gradients/moments");
                }
            }
            if (error.empty()) {
                status = byte_patch_embed(
                    d_tokens,
                    d_embedding,
                    d_projection,
                    d_patch,
                    kBatch,
                    kSeqLen,
                    kDim,
                    kPatchSize,
                    kStride,
                    kOutLen,
                    kVocab,
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_byte_patch_embed_float32");
                }
            }
            if (error.empty()) {
                status = byte_patch_merge(d_patch, d_merged, kBatch, kOutLen, kSeqLen, kDim, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_byte_patch_merge_float32");
                }
            }
            if (error.empty()) {
                status = linear(d_merged, d_head_weight, nullptr, d_pred, kSeqLen, kDim, kDim, false, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_float32 HNet head");
                }
            }
            if (error.empty()) {
                status = latent_mse_loss(d_pred, d_target, d_loss, kMergedElements, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_latent_mse_loss_float32 HNet");
                }
            }
            if (error.empty()) {
                status = linear_backward_input(d_grad_pred, d_head_weight, d_grad_merged, kSeqLen, kDim, kDim, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_backward_input_float32 HNet head");
                }
            }
            if (error.empty()) {
                status = linear_backward_weight_accumulate(
                    d_merged, d_grad_pred, d_grad_head_weight, kSeqLen, kDim, kDim, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_backward_weight_accumulate_float32 HNet head");
                }
            }
            if (error.empty()) {
                status = adamw(
                    d_head_weight,
                    d_grad_head_weight,
                    d_exp_avg,
                    d_exp_avg_sq,
                    kHeadWeightElements,
                    0.01f,
                    0.9f,
                    0.95f,
                    1e-8f,
                    0.02f,
                    0.1f,
                    std::sqrt(0.05f),
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_adamw_step_float32 HNet head");
                }
            }
            if (error.empty()) {
                status = cuda_device_synchronize();
                if (status != 0) {
                    error = cuda_error(status, "cudaDeviceSynchronize");
                }
            }

            std::vector<float> actual_patch(static_cast<std::size_t>(kPatchElements), 0.0f);
            std::vector<float> actual_merged(static_cast<std::size_t>(kMergedElements), 0.0f);
            std::vector<float> actual_pred(static_cast<std::size_t>(kMergedElements), 0.0f);
            std::vector<float> actual_loss(1, 0.0f);
            std::vector<float> actual_grad_merged(static_cast<std::size_t>(kMergedElements), 0.0f);
            std::vector<float> actual_grad_head_weight(head_weight.size(), 0.0f);
            std::vector<float> actual_head_weight(head_weight.size(), 0.0f);
            auto copy_back_float = [&](std::vector<float>& dst, const float* src, const std::string& name) {
                int copy_status = cuda_memcpy(dst.data(), src, dst.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (copy_status != 0) {
                    error = cuda_error(copy_status, "cudaMemcpy " + name + " D2H");
                    return false;
                }
                return true;
            };
            if (error.empty()) {
                copy_back_float(actual_patch, d_patch, "patch") &&
                    copy_back_float(actual_merged, d_merged, "merged") &&
                    copy_back_float(actual_pred, d_pred, "pred") &&
                    copy_back_float(actual_loss, d_loss, "loss") &&
                    copy_back_float(actual_grad_merged, d_grad_merged, "grad_merged") &&
                    copy_back_float(actual_grad_head_weight, d_grad_head_weight, "grad_head_weight") &&
                    copy_back_float(actual_head_weight, d_head_weight, "head_weight");
            }

            std::vector<float> expected_head_weight = head_weight;
            for (std::size_t i = 0; i < expected_head_weight.size(); ++i) {
                const float grad = expected_grad_head_weight[i];
                const float next_m = 0.1f * grad;
                const float next_v = 0.05f * grad * grad;
                const float denom = std::sqrt(next_v) / std::sqrt(0.05f) + 1e-8f;
                const float decayed = expected_head_weight[i] * (1.0f - 0.01f * 0.02f);
                expected_head_weight[i] = decayed - 0.01f * (next_m / 0.1f) / denom;
            }
            if (error.empty()) {
                byte_patch_embed_max_error = max_abs_error(actual_patch, expected_patch);
                byte_patch_merge_max_error = max_abs_error(actual_merged, expected_merged);
                head_forward_max_error = max_abs_error(actual_pred, expected_pred);
                hnet_loss_max_error = std::fabs(actual_loss[0] - expected_loss);
                head_grad_hidden_max_error = max_abs_error(actual_grad_merged, expected_grad_merged);
                head_grad_weight_max_error = max_abs_error(actual_grad_head_weight, expected_grad_head_weight);
                head_weight_update_max_error = max_abs_error(actual_head_weight, expected_head_weight);
            }
            constexpr float kTolerance = 5e-5f;
            constexpr float kAdamTolerance = 2e-5f;
            passed = error.empty() &&
                byte_patch_embed_max_error <= kTolerance &&
                byte_patch_merge_max_error <= kTolerance &&
                head_forward_max_error <= kTolerance &&
                hnet_loss_max_error <= kTolerance &&
                head_grad_hidden_max_error <= kTolerance &&
                head_grad_weight_max_error <= kTolerance &&
                head_weight_update_max_error <= kAdamTolerance;
            if (!passed && error.empty()) {
                error = "HNet byte patch train-step smoke exceeded tolerance";
            }
        }
    }
    free_allocated();
    close_handles();

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(NFN_NATIVE_MODEL_FAMILY) << "\",\n"
        << "  \"native_target\": \"" << json_escape(NFN_NATIVE_TARGET_NAME) << "\",\n"
        << "  \"smoke\": \"hnet_byte_patch_train_step_slice\",\n"
        << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(error) << "\",\n"
        << "  \"compiled_native_boundary\": true,\n"
        << "  \"torch_required\": false,\n"
        << "  \"graph_editor_tensor_flow\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_ops_lib) << "\",\n"
        << "  \"tile_ops_loaded\": " << (tile_ops_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"shape\": {\"batch\": " << kBatch << ", \"seq_len\": " << kSeqLen
        << ", \"patch_len\": " << kOutLen << ", \"dim\": " << kDim
        << ", \"patch_size\": " << kPatchSize << ", \"stride\": " << kStride << "},\n"
        << "  \"loop_composition_stages\": [\n"
        << "    \"nfn_native_tile_byte_patch_embed_float32\",\n"
        << "    \"nfn_native_tile_byte_patch_merge_float32\",\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_latent_mse_loss_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_accumulate_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"max_errors\": {"
        << "\"byte_patch_embed\":" << byte_patch_embed_max_error
        << ", \"byte_patch_merge\":" << byte_patch_merge_max_error
        << ", \"head_forward\":" << head_forward_max_error
        << ", \"hnet_loss\":" << hnet_loss_max_error
        << ", \"head_grad_hidden\":" << head_grad_hidden_max_error
        << ", \"head_grad_weight\":" << head_grad_weight_max_error
        << ", \"head_weight_update\":" << head_weight_update_max_error
        << "}\n"
        << "}\n";
    return passed ? 0 : 2;
}

int print_jamba_chunk_state_smoke_json(const Config& cfg, const char* program) {
    const std::string family = NFN_NATIVE_MODEL_FAMILY;
    const bool jamba_family = family.find("jamba") != std::string::npos || family == "unknown";
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
    using CausalChunkStateFn = int (*)(
        const float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, void*);
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightAccumulateFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LatentMseLossFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using AdamWFn = int (*)(
        float*, const float*, float*, float*, std::int64_t, float, float, float, float, float, float, float, void*);

    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;
    CausalChunkStateFn causal_chunk_state = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightAccumulateFn linear_backward_weight_accumulate = nullptr;
    LatentMseLossFn latent_mse_loss = nullptr;
    FillFn fill = nullptr;
    AdamWFn adamw = nullptr;

    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    constexpr std::int64_t kBatch = 1;
    constexpr std::int64_t kSeqLen = 4;
    constexpr std::int64_t kDim = 3;
    constexpr std::int64_t kChunkSize = 2;
    constexpr std::int64_t kChunks = 2;
    constexpr std::int64_t kModeMean = 0;
    constexpr std::int64_t kStateElements = kBatch * kChunks * kDim;
    constexpr std::int64_t kWeightElements = kDim * kDim;

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
    float chunk_state_max_error = 0.0f;
    float head_forward_max_error = 0.0f;
    float jamba_loss_max_error = 0.0f;
    float head_grad_state_max_error = 0.0f;
    float head_grad_weight_max_error = 0.0f;
    float head_weight_update_max_error = 0.0f;

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

    if (!jamba_family) {
        error = "Jamba smoke commands are only valid for jamba-family native preflights";
    } else {
        tile_handle = dlopen(tile_ops_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            const char* raw = dlerror();
            error = raw == nullptr ? "failed to load Tile ops library" : raw;
        } else {
            tile_ops_loaded = true;
            causal_chunk_state = load_symbol<CausalChunkStateFn>(
                tile_handle, "nfn_native_tile_causal_chunk_state_float32");
            linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
            linear_backward_input = load_symbol<LinearBackwardInputFn>(
                tile_handle, "nfn_native_tile_linear_backward_input_float32");
            linear_backward_weight_accumulate = load_symbol<LinearBackwardWeightAccumulateFn>(
                tile_handle, "nfn_native_tile_linear_backward_weight_accumulate_float32");
            latent_mse_loss = load_symbol<LatentMseLossFn>(
                tile_handle, "nfn_native_tile_latent_mse_loss_float32");
            fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
            adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
            if (causal_chunk_state == nullptr || linear == nullptr || linear_backward_input == nullptr ||
                linear_backward_weight_accumulate == nullptr || latent_mse_loss == nullptr ||
                fill == nullptr || adamw == nullptr) {
                error = "Tile ops library is missing one or more Jamba chunk-state train-step symbols";
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
        const std::vector<float> hidden = {
            0.10f, -0.05f, 0.07f,
            0.20f, 0.03f, -0.04f,
            -0.08f, 0.12f, 0.02f,
            0.04f, -0.15f, 0.09f,
        };
        const std::vector<float> target = {
            0.02f, -0.01f, 0.04f,
            -0.03f, 0.05f, 0.01f,
        };
        std::vector<float> head_weight(static_cast<std::size_t>(kWeightElements));
        for (std::size_t i = 0; i < head_weight.size(); ++i) {
            head_weight[i] = 0.025f * static_cast<float>(static_cast<int>(i % 7) - 3);
        }

        std::vector<float> expected_state(static_cast<std::size_t>(kStateElements), 0.0f);
        for (std::int64_t chunk = 0; chunk < kChunks; ++chunk) {
            const std::int64_t start = chunk * kChunkSize;
            const std::int64_t end = std::min<std::int64_t>(start + kChunkSize, kSeqLen);
            const float denom = static_cast<float>(std::max<std::int64_t>(end - start, 1));
            for (std::int64_t d = 0; d < kDim; ++d) {
                float sum = 0.0f;
                for (std::int64_t t = start; t < end; ++t) {
                    sum += hidden[static_cast<std::size_t>(t * kDim + d)];
                }
                expected_state[static_cast<std::size_t>(chunk * kDim + d)] = sum / denom;
            }
        }
        std::vector<float> expected_pred(static_cast<std::size_t>(kStateElements), 0.0f);
        for (std::int64_t row = 0; row < kChunks; ++row) {
            for (std::int64_t out_d = 0; out_d < kDim; ++out_d) {
                for (std::int64_t in_d = 0; in_d < kDim; ++in_d) {
                    expected_pred[static_cast<std::size_t>(row * kDim + out_d)] +=
                        expected_state[static_cast<std::size_t>(row * kDim + in_d)] *
                        head_weight[static_cast<std::size_t>(out_d * kDim + in_d)];
                }
            }
        }
        std::vector<float> grad_pred(static_cast<std::size_t>(kStateElements), 0.0f);
        float expected_loss = 0.0f;
        for (std::size_t i = 0; i < expected_pred.size(); ++i) {
            const float diff = expected_pred[i] - target[i];
            expected_loss += diff * diff;
            grad_pred[i] = 2.0f * diff;
        }
        std::vector<float> expected_grad_state(static_cast<std::size_t>(kStateElements), 0.0f);
        std::vector<float> expected_grad_weight(head_weight.size(), 0.0f);
        for (std::int64_t row = 0; row < kChunks; ++row) {
            for (std::int64_t out_d = 0; out_d < kDim; ++out_d) {
                const float grad = grad_pred[static_cast<std::size_t>(row * kDim + out_d)];
                for (std::int64_t in_d = 0; in_d < kDim; ++in_d) {
                    expected_grad_weight[static_cast<std::size_t>(out_d * kDim + in_d)] +=
                        expected_state[static_cast<std::size_t>(row * kDim + in_d)] * grad;
                    expected_grad_state[static_cast<std::size_t>(row * kDim + in_d)] +=
                        grad * head_weight[static_cast<std::size_t>(out_d * kDim + in_d)];
                }
            }
        }

        float* d_hidden = nullptr;
        float* d_state = nullptr;
        float* d_head_weight = nullptr;
        float* d_pred = nullptr;
        float* d_target = nullptr;
        float* d_loss = nullptr;
        float* d_grad_pred = nullptr;
        float* d_grad_state = nullptr;
        float* d_grad_weight = nullptr;
        float* d_exp_avg = nullptr;
        float* d_exp_avg_sq = nullptr;
        auto alloc_float = [&](float** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(float));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        if (alloc_float(&d_hidden, hidden.size(), "hidden") &&
            alloc_float(&d_state, kStateElements, "state") &&
            alloc_float(&d_head_weight, head_weight.size(), "head_weight") &&
            alloc_float(&d_pred, kStateElements, "pred") &&
            alloc_float(&d_target, target.size(), "target") &&
            alloc_float(&d_loss, 1, "loss") &&
            alloc_float(&d_grad_pred, grad_pred.size(), "grad_pred") &&
            alloc_float(&d_grad_state, kStateElements, "grad_state") &&
            alloc_float(&d_grad_weight, head_weight.size(), "grad_weight") &&
            alloc_float(&d_exp_avg, head_weight.size(), "exp_avg") &&
            alloc_float(&d_exp_avg_sq, head_weight.size(), "exp_avg_sq")) {
            auto copy_float = [&](float* dst, const std::vector<float>& src, const std::string& name) {
                int status = cuda_memcpy(dst, src.data(), src.size() * sizeof(float), kCudaMemcpyHostToDevice);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy " + name + " H2D");
                    return false;
                }
                return true;
            };
            copy_float(d_hidden, hidden, "hidden") &&
                copy_float(d_head_weight, head_weight, "head_weight") &&
                copy_float(d_target, target, "target") &&
                copy_float(d_grad_pred, grad_pred, "grad_pred");
            int status = 0;
            if (error.empty()) {
                status = fill(d_grad_weight, kWeightElements, 0.0f, nullptr);
                if (status == 0) {
                    status = fill(d_exp_avg, kWeightElements, 0.0f, nullptr);
                }
                if (status == 0) {
                    status = fill(d_exp_avg_sq, kWeightElements, 0.0f, nullptr);
                }
                if (status != 0) {
                    error = cuda_error(status, "zero Jamba gradients/moments");
                }
            }
            if (error.empty()) {
                status = causal_chunk_state(
                    d_hidden, d_state, kBatch, kSeqLen, kDim, kChunkSize, kChunks, kModeMean, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_causal_chunk_state_float32");
                }
            }
            if (error.empty()) {
                status = linear(d_state, d_head_weight, nullptr, d_pred, kChunks, kDim, kDim, false, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_float32 Jamba head");
                }
            }
            if (error.empty()) {
                status = latent_mse_loss(d_pred, d_target, d_loss, kStateElements, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_latent_mse_loss_float32 Jamba");
                }
            }
            if (error.empty()) {
                status = linear_backward_input(d_grad_pred, d_head_weight, d_grad_state, kChunks, kDim, kDim, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_backward_input_float32 Jamba head");
                }
            }
            if (error.empty()) {
                status = linear_backward_weight_accumulate(
                    d_state, d_grad_pred, d_grad_weight, kChunks, kDim, kDim, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_backward_weight_accumulate_float32 Jamba head");
                }
            }
            if (error.empty()) {
                status = adamw(
                    d_head_weight,
                    d_grad_weight,
                    d_exp_avg,
                    d_exp_avg_sq,
                    kWeightElements,
                    0.01f,
                    0.9f,
                    0.95f,
                    1e-8f,
                    0.02f,
                    0.1f,
                    std::sqrt(0.05f),
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_adamw_step_float32 Jamba head");
                }
            }
            if (error.empty()) {
                status = cuda_device_synchronize();
                if (status != 0) {
                    error = cuda_error(status, "cudaDeviceSynchronize");
                }
            }

            std::vector<float> actual_state(static_cast<std::size_t>(kStateElements), 0.0f);
            std::vector<float> actual_pred(static_cast<std::size_t>(kStateElements), 0.0f);
            std::vector<float> actual_loss(1, 0.0f);
            std::vector<float> actual_grad_state(static_cast<std::size_t>(kStateElements), 0.0f);
            std::vector<float> actual_grad_weight(head_weight.size(), 0.0f);
            std::vector<float> actual_head_weight(head_weight.size(), 0.0f);
            auto copy_back_float = [&](std::vector<float>& dst, const float* src, const std::string& name) {
                int copy_status = cuda_memcpy(dst.data(), src, dst.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (copy_status != 0) {
                    error = cuda_error(copy_status, "cudaMemcpy " + name + " D2H");
                    return false;
                }
                return true;
            };
            if (error.empty()) {
                copy_back_float(actual_state, d_state, "state") &&
                    copy_back_float(actual_pred, d_pred, "pred") &&
                    copy_back_float(actual_loss, d_loss, "loss") &&
                    copy_back_float(actual_grad_state, d_grad_state, "grad_state") &&
                    copy_back_float(actual_grad_weight, d_grad_weight, "grad_weight") &&
                    copy_back_float(actual_head_weight, d_head_weight, "head_weight");
            }

            std::vector<float> expected_head_weight = head_weight;
            for (std::size_t i = 0; i < expected_head_weight.size(); ++i) {
                const float grad = expected_grad_weight[i];
                const float next_m = 0.1f * grad;
                const float next_v = 0.05f * grad * grad;
                const float denom = std::sqrt(next_v) / std::sqrt(0.05f) + 1e-8f;
                const float decayed = expected_head_weight[i] * (1.0f - 0.01f * 0.02f);
                expected_head_weight[i] = decayed - 0.01f * (next_m / 0.1f) / denom;
            }
            if (error.empty()) {
                chunk_state_max_error = max_abs_error(actual_state, expected_state);
                head_forward_max_error = max_abs_error(actual_pred, expected_pred);
                jamba_loss_max_error = std::fabs(actual_loss[0] - expected_loss);
                head_grad_state_max_error = max_abs_error(actual_grad_state, expected_grad_state);
                head_grad_weight_max_error = max_abs_error(actual_grad_weight, expected_grad_weight);
                head_weight_update_max_error = max_abs_error(actual_head_weight, expected_head_weight);
            }
            constexpr float kTolerance = 5e-5f;
            constexpr float kAdamTolerance = 2e-5f;
            passed = error.empty() &&
                chunk_state_max_error <= kTolerance &&
                head_forward_max_error <= kTolerance &&
                jamba_loss_max_error <= kTolerance &&
                head_grad_state_max_error <= kTolerance &&
                head_grad_weight_max_error <= kTolerance &&
                head_weight_update_max_error <= kAdamTolerance;
            if (!passed && error.empty()) {
                error = "Jamba chunk-state train-step smoke exceeded tolerance";
            }
        }
    }
    free_allocated();
    close_handles();

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(NFN_NATIVE_MODEL_FAMILY) << "\",\n"
        << "  \"native_target\": \"" << json_escape(NFN_NATIVE_TARGET_NAME) << "\",\n"
        << "  \"smoke\": \"jamba_chunk_state_train_step_slice\",\n"
        << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(error) << "\",\n"
        << "  \"compiled_native_boundary\": true,\n"
        << "  \"torch_required\": false,\n"
        << "  \"graph_editor_tensor_flow\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_ops_lib) << "\",\n"
        << "  \"tile_ops_loaded\": " << (tile_ops_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"shape\": {\"batch\": " << kBatch << ", \"seq_len\": " << kSeqLen
        << ", \"chunks\": " << kChunks << ", \"chunk_size\": " << kChunkSize
        << ", \"dim\": " << kDim << "},\n"
        << "  \"loop_composition_stages\": [\n"
        << "    \"nfn_native_tile_causal_chunk_state_float32\",\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_latent_mse_loss_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_accumulate_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"max_errors\": {"
        << "\"chunk_state\":" << chunk_state_max_error
        << ", \"head_forward\":" << head_forward_max_error
        << ", \"jamba_loss\":" << jamba_loss_max_error
        << ", \"head_grad_state\":" << head_grad_state_max_error
        << ", \"head_grad_weight\":" << head_grad_weight_max_error
        << ", \"head_weight_update\":" << head_weight_update_max_error
        << "}\n"
        << "}\n";
    return passed ? 0 : 2;
}

int print_seq2seq_cross_attention_smoke_json(const Config& cfg, const char* program) {
    const std::string family = NFN_NATIVE_MODEL_FAMILY;
    const bool seq2seq_family = family.find("seq2seq") != std::string::npos || family == "unknown";
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
    using AttentionFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, float,
        bool, bool, bool, std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using AttentionBackwardFn = int (*)(
        const float*, const float*, const float*, const float*, float*, float*, float*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, float, bool, bool, bool, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightAccumulateFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyPartialsFn = int (*)(
        const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyBackwardFn = int (*)(
        const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, float, void*);
    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using AdamWFn = int (*)(
        float*, const float*, float*, float*, std::int64_t, float, float, float, float, float, float, float, void*);

    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;
    AttentionFn attention = nullptr;
    AttentionBackwardFn attention_backward = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightAccumulateFn linear_backward_weight_accumulate = nullptr;
    TokenCrossEntropyPartialsFn ce_partials = nullptr;
    TokenCrossEntropyBackwardFn ce_backward = nullptr;
    FillFn fill = nullptr;
    AdamWFn adamw = nullptr;

    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    constexpr std::int64_t kBatch = 1;
    constexpr std::int64_t kHeads = 1;
    constexpr std::int64_t kSeqQ = 2;
    constexpr std::int64_t kSeqK = 3;
    constexpr std::int64_t kDim = 2;
    constexpr std::int64_t kVocab = 3;
    constexpr std::int64_t kQueryElements = kBatch * kHeads * kSeqQ * kDim;
    constexpr std::int64_t kKeyElements = kBatch * kHeads * kSeqK * kDim;
    constexpr std::int64_t kValueElements = kBatch * kHeads * kSeqK * kDim;
    constexpr std::int64_t kAttnElements = kQueryElements;
    constexpr std::int64_t kLogitElements = kSeqQ * kVocab;
    constexpr std::int64_t kWeightElements = kVocab * kDim;
    constexpr float kScale = 0.70710678118f;

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
    float cross_attention_forward_max_error = 0.0f;
    float seq2seq_loss_max_error = 0.0f;
    float lm_grad_hidden_max_error = 0.0f;
    float lm_grad_weight_max_error = 0.0f;
    float cross_attention_grad_q_max_error = 0.0f;
    float cross_attention_grad_k_max_error = 0.0f;
    float cross_attention_grad_v_max_error = 0.0f;
    float lm_weight_update_max_error = 0.0f;

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

    if (!seq2seq_family) {
        error = "Seq2seq smoke commands are only valid for seq2seq-family native preflights";
    } else {
        tile_handle = dlopen(tile_ops_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            const char* raw = dlerror();
            error = raw == nullptr ? "failed to load Tile ops library" : raw;
        } else {
            tile_ops_loaded = true;
            attention = load_symbol<AttentionFn>(tile_handle, "nfn_native_tile_scaled_dot_product_attention_float32");
            attention_backward = load_symbol<AttentionBackwardFn>(
                tile_handle, "nfn_native_tile_scaled_dot_product_attention_backward_float32");
            linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
            linear_backward_input = load_symbol<LinearBackwardInputFn>(
                tile_handle, "nfn_native_tile_linear_backward_input_float32");
            linear_backward_weight_accumulate = load_symbol<LinearBackwardWeightAccumulateFn>(
                tile_handle, "nfn_native_tile_linear_backward_weight_accumulate_float32");
            ce_partials = load_symbol<TokenCrossEntropyPartialsFn>(
                tile_handle, "nfn_native_tile_token_cross_entropy_partials_float32");
            ce_backward = load_symbol<TokenCrossEntropyBackwardFn>(
                tile_handle, "nfn_native_tile_token_cross_entropy_backward_float32");
            fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
            adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
            if (attention == nullptr || attention_backward == nullptr || linear == nullptr ||
                linear_backward_input == nullptr || linear_backward_weight_accumulate == nullptr ||
                ce_partials == nullptr || ce_backward == nullptr || fill == nullptr || adamw == nullptr) {
                error = "Tile ops library is missing one or more seq2seq cross-attention train-step symbols";
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
        const std::vector<float> q = {0.20f, -0.10f, 0.05f, 0.30f};
        const std::vector<float> k = {0.10f, 0.00f, -0.20f, 0.25f, 0.15f, -0.05f};
        const std::vector<float> v = {-0.10f, 0.35f, 0.20f, -0.15f, 0.05f, 0.10f};
        const std::vector<std::int64_t> targets = {1, 2};
        std::vector<float> lm_weight(static_cast<std::size_t>(kWeightElements));
        for (std::size_t i = 0; i < lm_weight.size(); ++i) {
            lm_weight[i] = 0.04f * static_cast<float>(static_cast<int>(i % 5) - 2);
        }

        std::vector<float> expected_attn(static_cast<std::size_t>(kAttnElements), 0.0f);
        std::vector<float> probs(static_cast<std::size_t>(kSeqQ * kSeqK), 0.0f);
        for (std::int64_t tq = 0; tq < kSeqQ; ++tq) {
            float max_score = -3.4028234663852886e38f;
            for (std::int64_t tk = 0; tk < kSeqK; ++tk) {
                float score = 0.0f;
                for (std::int64_t d = 0; d < kDim; ++d) {
                    score += q[static_cast<std::size_t>(tq * kDim + d)] *
                        k[static_cast<std::size_t>(tk * kDim + d)];
                }
                score *= kScale;
                probs[static_cast<std::size_t>(tq * kSeqK + tk)] = score;
                max_score = std::max(max_score, score);
            }
            float denom = 0.0f;
            for (std::int64_t tk = 0; tk < kSeqK; ++tk) {
                float p = std::exp(probs[static_cast<std::size_t>(tq * kSeqK + tk)] - max_score);
                probs[static_cast<std::size_t>(tq * kSeqK + tk)] = p;
                denom += p;
            }
            for (std::int64_t tk = 0; tk < kSeqK; ++tk) {
                const float p = probs[static_cast<std::size_t>(tq * kSeqK + tk)] / denom;
                probs[static_cast<std::size_t>(tq * kSeqK + tk)] = p;
                for (std::int64_t d = 0; d < kDim; ++d) {
                    expected_attn[static_cast<std::size_t>(tq * kDim + d)] +=
                        p * v[static_cast<std::size_t>(tk * kDim + d)];
                }
            }
        }
        std::vector<float> expected_logits(static_cast<std::size_t>(kLogitElements), 0.0f);
        for (std::int64_t row = 0; row < kSeqQ; ++row) {
            for (std::int64_t token = 0; token < kVocab; ++token) {
                for (std::int64_t d = 0; d < kDim; ++d) {
                    expected_logits[static_cast<std::size_t>(row * kVocab + token)] +=
                        expected_attn[static_cast<std::size_t>(row * kDim + d)] *
                        lm_weight[static_cast<std::size_t>(token * kDim + d)];
                }
            }
        }
        std::vector<float> expected_grad_logits(static_cast<std::size_t>(kLogitElements), 0.0f);
        float expected_loss = 0.0f;
        for (std::int64_t row = 0; row < kSeqQ; ++row) {
            float max_logit = -3.4028234663852886e38f;
            for (std::int64_t token = 0; token < kVocab; ++token) {
                max_logit = std::max(max_logit, expected_logits[static_cast<std::size_t>(row * kVocab + token)]);
            }
            float denom = 0.0f;
            for (std::int64_t token = 0; token < kVocab; ++token) {
                denom += std::exp(expected_logits[static_cast<std::size_t>(row * kVocab + token)] - max_logit);
            }
            expected_loss += std::log(denom) + max_logit -
                expected_logits[static_cast<std::size_t>(row * kVocab + targets[static_cast<std::size_t>(row)])];
            for (std::int64_t token = 0; token < kVocab; ++token) {
                const float p = std::exp(expected_logits[static_cast<std::size_t>(row * kVocab + token)] - max_logit) / denom;
                const float target_term = targets[static_cast<std::size_t>(row)] == token ? 1.0f : 0.0f;
                expected_grad_logits[static_cast<std::size_t>(row * kVocab + token)] =
                    (p - target_term) / static_cast<float>(kSeqQ);
            }
        }
        std::vector<float> expected_grad_attn(static_cast<std::size_t>(kAttnElements), 0.0f);
        std::vector<float> expected_grad_lm_weight(lm_weight.size(), 0.0f);
        for (std::int64_t row = 0; row < kSeqQ; ++row) {
            for (std::int64_t token = 0; token < kVocab; ++token) {
                const float grad = expected_grad_logits[static_cast<std::size_t>(row * kVocab + token)];
                for (std::int64_t d = 0; d < kDim; ++d) {
                    expected_grad_lm_weight[static_cast<std::size_t>(token * kDim + d)] +=
                        expected_attn[static_cast<std::size_t>(row * kDim + d)] * grad;
                    expected_grad_attn[static_cast<std::size_t>(row * kDim + d)] +=
                        grad * lm_weight[static_cast<std::size_t>(token * kDim + d)];
                }
            }
        }
        std::vector<float> expected_grad_q(static_cast<std::size_t>(kQueryElements), 0.0f);
        std::vector<float> expected_grad_k(static_cast<std::size_t>(kKeyElements), 0.0f);
        std::vector<float> expected_grad_v(static_cast<std::size_t>(kValueElements), 0.0f);
        for (std::int64_t tq = 0; tq < kSeqQ; ++tq) {
            std::vector<float> grad_p(static_cast<std::size_t>(kSeqK), 0.0f);
            float weighted_grad_p = 0.0f;
            for (std::int64_t tk = 0; tk < kSeqK; ++tk) {
                for (std::int64_t d = 0; d < kDim; ++d) {
                    grad_p[static_cast<std::size_t>(tk)] +=
                        expected_grad_attn[static_cast<std::size_t>(tq * kDim + d)] *
                        v[static_cast<std::size_t>(tk * kDim + d)];
                    expected_grad_v[static_cast<std::size_t>(tk * kDim + d)] +=
                        probs[static_cast<std::size_t>(tq * kSeqK + tk)] *
                        expected_grad_attn[static_cast<std::size_t>(tq * kDim + d)];
                }
                weighted_grad_p += probs[static_cast<std::size_t>(tq * kSeqK + tk)] *
                    grad_p[static_cast<std::size_t>(tk)];
            }
            for (std::int64_t tk = 0; tk < kSeqK; ++tk) {
                const float grad_score =
                    probs[static_cast<std::size_t>(tq * kSeqK + tk)] *
                    (grad_p[static_cast<std::size_t>(tk)] - weighted_grad_p);
                for (std::int64_t d = 0; d < kDim; ++d) {
                    expected_grad_q[static_cast<std::size_t>(tq * kDim + d)] +=
                        grad_score * kScale * k[static_cast<std::size_t>(tk * kDim + d)];
                    expected_grad_k[static_cast<std::size_t>(tk * kDim + d)] +=
                        grad_score * kScale * q[static_cast<std::size_t>(tq * kDim + d)];
                }
            }
        }

        float* d_q = nullptr;
        float* d_k = nullptr;
        float* d_v = nullptr;
        float* d_attn = nullptr;
        float* d_lm_weight = nullptr;
        float* d_logits = nullptr;
        float* d_loss = nullptr;
        float* d_grad_logits = nullptr;
        float* d_grad_attn = nullptr;
        float* d_grad_lm_weight = nullptr;
        float* d_grad_q = nullptr;
        float* d_grad_k = nullptr;
        float* d_grad_v = nullptr;
        float* d_exp_avg = nullptr;
        float* d_exp_avg_sq = nullptr;
        std::int64_t* d_targets = nullptr;
        auto alloc = [&](float** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(float));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        auto alloc_i64 = [&](std::int64_t** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(std::int64_t));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        if (alloc(&d_q, q.size(), "q") &&
            alloc(&d_k, k.size(), "k") &&
            alloc(&d_v, v.size(), "v") &&
            alloc(&d_attn, kAttnElements, "attn") &&
            alloc(&d_lm_weight, lm_weight.size(), "lm_weight") &&
            alloc(&d_logits, kLogitElements, "logits") &&
            alloc(&d_loss, 1, "loss") &&
            alloc(&d_grad_logits, kLogitElements, "grad_logits") &&
            alloc(&d_grad_attn, kAttnElements, "grad_attn") &&
            alloc(&d_grad_lm_weight, lm_weight.size(), "grad_lm_weight") &&
            alloc(&d_grad_q, kQueryElements, "grad_q") &&
            alloc(&d_grad_k, kKeyElements, "grad_k") &&
            alloc(&d_grad_v, kValueElements, "grad_v") &&
            alloc(&d_exp_avg, lm_weight.size(), "exp_avg") &&
            alloc(&d_exp_avg_sq, lm_weight.size(), "exp_avg_sq") &&
            alloc_i64(&d_targets, targets.size(), "targets")) {
            auto copy_float = [&](float* dst, const std::vector<float>& src, const std::string& name) {
                int status = cuda_memcpy(dst, src.data(), src.size() * sizeof(float), kCudaMemcpyHostToDevice);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy " + name + " H2D");
                    return false;
                }
                return true;
            };
            auto copy_i64 = [&](std::int64_t* dst, const std::vector<std::int64_t>& src, const std::string& name) {
                int status = cuda_memcpy(dst, src.data(), src.size() * sizeof(std::int64_t), kCudaMemcpyHostToDevice);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy " + name + " H2D");
                    return false;
                }
                return true;
            };
            copy_float(d_q, q, "q") &&
                copy_float(d_k, k, "k") &&
                copy_float(d_v, v, "v") &&
                copy_float(d_lm_weight, lm_weight, "lm_weight") &&
                copy_i64(d_targets, targets, "targets");
            int status = 0;
            if (error.empty()) {
                status = fill(d_grad_lm_weight, kWeightElements, 0.0f, nullptr);
                if (status == 0) {
                    status = fill(d_exp_avg, kWeightElements, 0.0f, nullptr);
                }
                if (status == 0) {
                    status = fill(d_exp_avg_sq, kWeightElements, 0.0f, nullptr);
                }
                if (status != 0) {
                    error = cuda_error(status, "zero seq2seq gradients/moments");
                }
            }
            if (error.empty()) {
                status = attention(d_q, d_k, d_v, d_attn, kQueryElements, kHeads, kHeads, kSeqQ, kSeqK, kDim, kDim, kScale, false, false, false, 0, 0, 0, 0, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_scaled_dot_product_attention_float32 cross");
                }
            }
            if (error.empty()) {
                status = linear(d_attn, d_lm_weight, nullptr, d_logits, kSeqQ, kDim, kVocab, false, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_float32 seq2seq lm");
                }
            }
            if (error.empty()) {
                status = ce_partials(d_logits, d_targets, d_loss, kSeqQ, kVocab, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_token_cross_entropy_partials_float32 seq2seq");
                }
            }
            if (error.empty()) {
                status = ce_backward(d_logits, d_targets, d_grad_logits, kSeqQ, kVocab, 1.0f / static_cast<float>(kSeqQ), nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_token_cross_entropy_backward_float32 seq2seq");
                }
            }
            if (error.empty()) {
                status = linear_backward_input(d_grad_logits, d_lm_weight, d_grad_attn, kSeqQ, kDim, kVocab, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_backward_input_float32 seq2seq lm");
                }
            }
            if (error.empty()) {
                status = linear_backward_weight_accumulate(d_attn, d_grad_logits, d_grad_lm_weight, kSeqQ, kDim, kVocab, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_linear_backward_weight_accumulate_float32 seq2seq lm");
                }
            }
            if (error.empty()) {
                status = attention_backward(d_q, d_k, d_v, d_grad_attn, d_grad_q, d_grad_k, d_grad_v, kBatch, kHeads, kHeads, kSeqQ, kSeqK, kDim, kDim, kScale, false, false, false, 0, 0, 0, 0, nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_scaled_dot_product_attention_backward_float32 cross");
                }
            }
            if (error.empty()) {
                status = adamw(
                    d_lm_weight,
                    d_grad_lm_weight,
                    d_exp_avg,
                    d_exp_avg_sq,
                    kWeightElements,
                    0.01f,
                    0.9f,
                    0.95f,
                    1e-8f,
                    0.02f,
                    0.1f,
                    std::sqrt(0.05f),
                    nullptr);
                if (status != 0) {
                    error = cuda_error(status, "nfn_native_tile_adamw_step_float32 seq2seq lm");
                }
            }
            if (error.empty()) {
                status = cuda_device_synchronize();
                if (status != 0) {
                    error = cuda_error(status, "cudaDeviceSynchronize");
                }
            }

            std::vector<float> actual_attn(static_cast<std::size_t>(kAttnElements), 0.0f);
            std::vector<float> actual_loss(1, 0.0f);
            std::vector<float> actual_grad_attn(static_cast<std::size_t>(kAttnElements), 0.0f);
            std::vector<float> actual_grad_lm_weight(lm_weight.size(), 0.0f);
            std::vector<float> actual_grad_q(static_cast<std::size_t>(kQueryElements), 0.0f);
            std::vector<float> actual_grad_k(static_cast<std::size_t>(kKeyElements), 0.0f);
            std::vector<float> actual_grad_v(static_cast<std::size_t>(kValueElements), 0.0f);
            std::vector<float> actual_lm_weight(lm_weight.size(), 0.0f);
            auto copy_back_float = [&](std::vector<float>& dst, const float* src, const std::string& name) {
                int copy_status = cuda_memcpy(dst.data(), src, dst.size() * sizeof(float), kCudaMemcpyDeviceToHost);
                if (copy_status != 0) {
                    error = cuda_error(copy_status, "cudaMemcpy " + name + " D2H");
                    return false;
                }
                return true;
            };
            if (error.empty()) {
                copy_back_float(actual_attn, d_attn, "attn") &&
                    copy_back_float(actual_loss, d_loss, "loss") &&
                    copy_back_float(actual_grad_attn, d_grad_attn, "grad_attn") &&
                    copy_back_float(actual_grad_lm_weight, d_grad_lm_weight, "grad_lm_weight") &&
                    copy_back_float(actual_grad_q, d_grad_q, "grad_q") &&
                    copy_back_float(actual_grad_k, d_grad_k, "grad_k") &&
                    copy_back_float(actual_grad_v, d_grad_v, "grad_v") &&
                    copy_back_float(actual_lm_weight, d_lm_weight, "lm_weight");
            }
            std::vector<float> expected_lm_weight = lm_weight;
            for (std::size_t i = 0; i < expected_lm_weight.size(); ++i) {
                const float grad = expected_grad_lm_weight[i];
                const float next_m = 0.1f * grad;
                const float next_v = 0.05f * grad * grad;
                const float denom = std::sqrt(next_v) / std::sqrt(0.05f) + 1e-8f;
                const float decayed = expected_lm_weight[i] * (1.0f - 0.01f * 0.02f);
                expected_lm_weight[i] = decayed - 0.01f * (next_m / 0.1f) / denom;
            }
            if (error.empty()) {
                cross_attention_forward_max_error = max_abs_error(actual_attn, expected_attn);
                seq2seq_loss_max_error = std::fabs(actual_loss[0] - expected_loss);
                lm_grad_hidden_max_error = max_abs_error(actual_grad_attn, expected_grad_attn);
                lm_grad_weight_max_error = max_abs_error(actual_grad_lm_weight, expected_grad_lm_weight);
                cross_attention_grad_q_max_error = max_abs_error(actual_grad_q, expected_grad_q);
                cross_attention_grad_k_max_error = max_abs_error(actual_grad_k, expected_grad_k);
                cross_attention_grad_v_max_error = max_abs_error(actual_grad_v, expected_grad_v);
                lm_weight_update_max_error = max_abs_error(actual_lm_weight, expected_lm_weight);
            }
            constexpr float kTolerance = 2e-4f;
            passed = error.empty() &&
                cross_attention_forward_max_error <= kTolerance &&
                seq2seq_loss_max_error <= kTolerance &&
                lm_grad_hidden_max_error <= kTolerance &&
                lm_grad_weight_max_error <= kTolerance &&
                cross_attention_grad_q_max_error <= kTolerance &&
                cross_attention_grad_k_max_error <= kTolerance &&
                cross_attention_grad_v_max_error <= kTolerance &&
                lm_weight_update_max_error <= kTolerance;
            if (!passed && error.empty()) {
                error = "Seq2seq cross-attention train-step smoke exceeded tolerance";
            }
        }
    }
    free_allocated();
    close_handles();

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(NFN_NATIVE_MODEL_FAMILY) << "\",\n"
        << "  \"native_target\": \"" << json_escape(NFN_NATIVE_TARGET_NAME) << "\",\n"
        << "  \"smoke\": \"seq2seq_cross_attention_train_step_slice\",\n"
        << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(error) << "\",\n"
        << "  \"compiled_native_boundary\": true,\n"
        << "  \"torch_required\": false,\n"
        << "  \"graph_editor_tensor_flow\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_ops_lib) << "\",\n"
        << "  \"tile_ops_loaded\": " << (tile_ops_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"shape\": {\"batch\": " << kBatch << ", \"heads\": " << kHeads
        << ", \"decoder_seq\": " << kSeqQ << ", \"encoder_seq\": " << kSeqK
        << ", \"dim\": " << kDim << ", \"vocab\": " << kVocab << "},\n"
        << "  \"loop_composition_stages\": [\n"
        << "    \"nfn_native_tile_scaled_dot_product_attention_float32\",\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_token_cross_entropy_partials_float32\",\n"
        << "    \"nfn_native_tile_token_cross_entropy_backward_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_accumulate_float32\",\n"
        << "    \"nfn_native_tile_scaled_dot_product_attention_backward_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"max_errors\": {"
        << "\"cross_attention_forward\":" << cross_attention_forward_max_error
        << ", \"seq2seq_loss\":" << seq2seq_loss_max_error
        << ", \"lm_grad_hidden\":" << lm_grad_hidden_max_error
        << ", \"lm_grad_weight\":" << lm_grad_weight_max_error
        << ", \"cross_attention_grad_q\":" << cross_attention_grad_q_max_error
        << ", \"cross_attention_grad_k\":" << cross_attention_grad_k_max_error
        << ", \"cross_attention_grad_v\":" << cross_attention_grad_v_max_error
        << ", \"lm_weight_update\":" << lm_weight_update_max_error
        << "}\n"
        << "}\n";
    return passed ? 0 : 2;
}

int print_semantic_alignment_smoke_json(const Config& cfg, const char* program) {
    const std::string family = NFN_NATIVE_MODEL_FAMILY;
    const bool semantic_family = family.find("semantic") != std::string::npos || family == "unknown";
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
    using SemanticHashFn = int (*)(
        const float*, const float*, std::int64_t*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using SemanticAlignmentLossItemsFn = int (*)(
        const float*, const std::int64_t*, const std::int64_t*, float*, float*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);

    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;
    SemanticHashFn semantic_hash = nullptr;
    SemanticAlignmentLossItemsFn semantic_alignment_loss_items = nullptr;

    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    constexpr std::int64_t kRows = 2;
    constexpr std::int64_t kDims = 3;
    constexpr std::int64_t kTerms = 4;
    constexpr std::int64_t kTables = 2;
    constexpr std::int64_t kPlanes = 3;
    constexpr std::int64_t kIgnoreIndex = -1;
    constexpr std::int64_t kLogitElements = kRows * kDims * kTerms;
    constexpr std::int64_t kItemElements = kRows * kDims;
    constexpr std::int64_t kHashElements = kRows * kTables;

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
    auto max_i64_error = [](const std::vector<std::int64_t>& actual, const std::vector<std::int64_t>& expected) {
        std::int64_t max_err = 0;
        const std::size_t n = std::min(actual.size(), expected.size());
        for (std::size_t i = 0; i < n; ++i) {
            const std::int64_t err = static_cast<std::int64_t>(std::llabs(actual[i] - expected[i]));
            max_err = std::max(max_err, err);
        }
        return max_err;
    };

    bool passed = false;
    std::int64_t semantic_hash_max_error = 0;
    float semantic_loss_items_max_error = 0.0f;
    float semantic_count_items_max_error = 0.0f;
    float semantic_loss_sum_max_error = 0.0f;
    float semantic_count_sum_max_error = 0.0f;

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

    if (!semantic_family) {
        error = "Semantic smoke commands are only valid for semantic-family native preflights";
    } else {
        tile_handle = dlopen(tile_ops_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            const char* raw = dlerror();
            error = raw == nullptr ? "failed to load Tile ops library" : raw;
        } else {
            tile_ops_loaded = true;
            semantic_hash = load_symbol<SemanticHashFn>(tile_handle, "nfn_native_tile_semantic_hash_int64");
            semantic_alignment_loss_items = load_symbol<SemanticAlignmentLossItemsFn>(
                tile_handle, "nfn_native_tile_semantic_alignment_loss_items_float32");
            if (semantic_hash == nullptr || semantic_alignment_loss_items == nullptr) {
                error = "Tile ops library is missing one or more semantic alignment symbols";
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
        const std::vector<float> sem_vec = {
            0.25f, -0.5f, 0.75f,
            -0.2f, 0.4f, 0.1f,
        };
        const std::vector<float> projection = {
            0.5f, 0.25f, -0.1f,
            -0.4f, 0.2f, 0.3f,
            0.1f, -0.7f, 0.4f,
            -0.3f, 0.6f, 0.2f,
            0.8f, -0.1f, -0.5f,
            0.2f, 0.3f, 0.4f,
        };
        std::vector<float> logits(static_cast<std::size_t>(kLogitElements));
        for (std::int64_t i = 0; i < kLogitElements; ++i) {
            logits[static_cast<std::size_t>(i)] = 0.07f * static_cast<float>(static_cast<int>(i % 9) - 4);
        }
        const std::vector<std::int64_t> targets = {
            1, 2, kIgnoreIndex,
            0, 1, 2,
        };
        const std::vector<std::int64_t> term_counts = {3, 2, 4};

        std::vector<std::int64_t> expected_hash(static_cast<std::size_t>(kHashElements), 0);
        for (std::int64_t row = 0; row < kRows; ++row) {
            for (std::int64_t table = 0; table < kTables; ++table) {
                std::int64_t hash = 0;
                for (std::int64_t plane = 0; plane < kPlanes; ++plane) {
                    float dot = 0.0f;
                    for (std::int64_t dim = 0; dim < kDims; ++dim) {
                        dot += sem_vec[static_cast<std::size_t>(row * kDims + dim)] *
                            projection[static_cast<std::size_t>((table * kPlanes + plane) * kDims + dim)];
                    }
                    if (dot > 0.0f) {
                        hash |= (static_cast<std::int64_t>(1) << plane);
                    }
                }
                expected_hash[static_cast<std::size_t>(row * kTables + table)] = hash;
            }
        }

        std::vector<float> expected_losses(static_cast<std::size_t>(kItemElements), 0.0f);
        std::vector<float> expected_counts(static_cast<std::size_t>(kItemElements), 0.0f);
        float expected_loss_sum = 0.0f;
        float expected_count_sum = 0.0f;
        for (std::int64_t row = 0; row < kRows; ++row) {
            for (std::int64_t dim = 0; dim < kDims; ++dim) {
                const std::size_t item_idx = static_cast<std::size_t>(row * kDims + dim);
                const std::int64_t target = targets[item_idx];
                const std::int64_t term_count = std::min<std::int64_t>(term_counts[static_cast<std::size_t>(dim)], kTerms);
                if (target == kIgnoreIndex || target < 0 || target >= term_count) {
                    continue;
                }
                const std::int64_t base = (row * kDims + dim) * kTerms;
                float max_value = -3.4028234663852886e38f;
                for (std::int64_t term = 0; term < term_count; ++term) {
                    max_value = std::max(max_value, logits[static_cast<std::size_t>(base + term)]);
                }
                float sum_exp = 0.0f;
                for (std::int64_t term = 0; term < term_count; ++term) {
                    sum_exp += std::exp(logits[static_cast<std::size_t>(base + term)] - max_value);
                }
                const float loss =
                    std::log(sum_exp) + max_value - logits[static_cast<std::size_t>(base + target)];
                expected_losses[item_idx] = loss;
                expected_counts[item_idx] = 1.0f;
                expected_loss_sum += loss;
                expected_count_sum += 1.0f;
            }
        }

        float* d_sem_vec = nullptr;
        float* d_projection = nullptr;
        float* d_logits = nullptr;
        float* d_losses = nullptr;
        float* d_counts = nullptr;
        std::int64_t* d_hash = nullptr;
        std::int64_t* d_targets = nullptr;
        std::int64_t* d_term_counts = nullptr;
        auto alloc = [&](float** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(float));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        auto alloc_i64 = [&](std::int64_t** ptr, std::size_t count, const std::string& name) {
            int status = cuda_malloc(reinterpret_cast<void**>(ptr), count * sizeof(std::int64_t));
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc " + name);
                return false;
            }
            allocated.push_back(*ptr);
            return true;
        };
        if (alloc(&d_sem_vec, sem_vec.size(), "sem_vec") &&
            alloc(&d_projection, projection.size(), "projection") &&
            alloc(&d_logits, logits.size(), "logits") &&
            alloc(&d_losses, kItemElements, "losses") &&
            alloc(&d_counts, kItemElements, "counts") &&
            alloc_i64(&d_hash, kHashElements, "hash") &&
            alloc_i64(&d_targets, targets.size(), "targets") &&
            alloc_i64(&d_term_counts, term_counts.size(), "term_counts")) {
            auto copy_float = [&](float* dst, const std::vector<float>& src, const std::string& name) {
                int status = cuda_memcpy(dst, src.data(), src.size() * sizeof(float), kCudaMemcpyHostToDevice);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy " + name + " H2D");
                    return false;
                }
                return true;
            };
            auto copy_i64 = [&](std::int64_t* dst, const std::vector<std::int64_t>& src, const std::string& name) {
                int status = cuda_memcpy(dst, src.data(), src.size() * sizeof(std::int64_t), kCudaMemcpyHostToDevice);
                if (status != 0) {
                    error = cuda_error(status, "cudaMemcpy " + name + " H2D");
                    return false;
                }
                return true;
            };
            copy_float(d_sem_vec, sem_vec, "sem_vec") &&
                copy_float(d_projection, projection, "projection") &&
                copy_float(d_logits, logits, "logits") &&
                copy_i64(d_targets, targets, "targets") &&
                copy_i64(d_term_counts, term_counts, "term_counts");
        }
        int status = 0;
        if (error.empty()) {
            status = semantic_hash(d_sem_vec, d_projection, d_hash, kRows, kDims, kTables, kPlanes, nullptr);
            if (status != 0) {
                error = cuda_error(status, "nfn_native_tile_semantic_hash_int64");
            }
        }
        if (error.empty()) {
            status = semantic_alignment_loss_items(
                d_logits,
                d_targets,
                d_term_counts,
                d_losses,
                d_counts,
                kItemElements,
                kDims,
                kTerms,
                kIgnoreIndex,
                nullptr);
            if (status != 0) {
                error = cuda_error(status, "nfn_native_tile_semantic_alignment_loss_items_float32");
            }
        }
        if (error.empty()) {
            status = cuda_device_synchronize();
            if (status != 0) {
                error = cuda_error(status, "cudaDeviceSynchronize");
            }
        }

        std::vector<std::int64_t> actual_hash(static_cast<std::size_t>(kHashElements), 0);
        std::vector<float> actual_losses(static_cast<std::size_t>(kItemElements), 0.0f);
        std::vector<float> actual_counts(static_cast<std::size_t>(kItemElements), 0.0f);
        auto copy_back_float = [&](std::vector<float>& dst, const float* src, const std::string& name) {
            int copy_status = cuda_memcpy(dst.data(), src, dst.size() * sizeof(float), kCudaMemcpyDeviceToHost);
            if (copy_status != 0) {
                error = cuda_error(copy_status, "cudaMemcpy " + name + " D2H");
                return false;
            }
            return true;
        };
        auto copy_back_i64 = [&](std::vector<std::int64_t>& dst, const std::int64_t* src, const std::string& name) {
            int copy_status = cuda_memcpy(dst.data(), src, dst.size() * sizeof(std::int64_t), kCudaMemcpyDeviceToHost);
            if (copy_status != 0) {
                error = cuda_error(copy_status, "cudaMemcpy " + name + " D2H");
                return false;
            }
            return true;
        };
        if (error.empty()) {
            copy_back_i64(actual_hash, d_hash, "hash") &&
                copy_back_float(actual_losses, d_losses, "losses") &&
                copy_back_float(actual_counts, d_counts, "counts");
        }
        if (error.empty()) {
            float actual_loss_sum = 0.0f;
            float actual_count_sum = 0.0f;
            for (float value : actual_losses) {
                actual_loss_sum += value;
            }
            for (float value : actual_counts) {
                actual_count_sum += value;
            }
            semantic_hash_max_error = max_i64_error(actual_hash, expected_hash);
            semantic_loss_items_max_error = max_abs_error(actual_losses, expected_losses);
            semantic_count_items_max_error = max_abs_error(actual_counts, expected_counts);
            semantic_loss_sum_max_error = std::fabs(actual_loss_sum - expected_loss_sum);
            semantic_count_sum_max_error = std::fabs(actual_count_sum - expected_count_sum);
        }
        constexpr float kTolerance = 2e-6f;
        passed = error.empty() &&
            semantic_hash_max_error == 0 &&
            semantic_loss_items_max_error <= kTolerance &&
            semantic_count_items_max_error <= kTolerance &&
            semantic_loss_sum_max_error <= kTolerance &&
            semantic_count_sum_max_error <= kTolerance;
        if (!passed && error.empty()) {
            error = "Semantic alignment smoke exceeded tolerance";
        }
    }
    free_allocated();
    close_handles();

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(NFN_NATIVE_MODEL_FAMILY) << "\",\n"
        << "  \"native_target\": \"" << json_escape(NFN_NATIVE_TARGET_NAME) << "\",\n"
        << "  \"smoke\": \"semantic_alignment_step_slice\",\n"
        << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(error) << "\",\n"
        << "  \"compiled_native_boundary\": true,\n"
        << "  \"torch_required\": false,\n"
        << "  \"graph_editor_tensor_flow\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_ops_lib) << "\",\n"
        << "  \"tile_ops_loaded\": " << (tile_ops_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"shape\": {\"rows\": " << kRows << ", \"dims\": " << kDims
        << ", \"terms\": " << kTerms << ", \"tables\": " << kTables
        << ", \"planes\": " << kPlanes << "},\n"
        << "  \"loop_composition_stages\": [\n"
        << "    \"nfn_native_tile_semantic_hash_int64\",\n"
        << "    \"nfn_native_tile_semantic_alignment_loss_items_float32\"\n"
        << "  ],\n"
        << "  \"max_errors\": {"
        << "\"semantic_hash\":" << semantic_hash_max_error
        << ", \"semantic_loss_items\":" << semantic_loss_items_max_error
        << ", \"semantic_count_items\":" << semantic_count_items_max_error
        << ", \"semantic_loss_sum\":" << semantic_loss_sum_max_error
        << ", \"semantic_count_sum\":" << semantic_count_sum_max_error
        << "}\n"
        << "}\n";
    return passed ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
    const Config cfg = parse_args(argc, argv);
    try {
        if (cfg.smoke_llama_loop || cfg.smoke_llama_train_step || cfg.smoke_llama_lm_head_step) {
            return print_llama_loop_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_llama_packed_attention_step) {
            return print_llama_packed_attention_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_llama_attention_block_step) {
            return print_llama_attention_block_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_moe_route_expert_step) {
            return print_moe_route_expert_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_moe_transformer_block_step) {
            return print_moe_transformer_block_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_jepa_projector_step) {
            return print_jepa_projector_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_jepa_target_encoder_step) {
            return print_jepa_target_encoder_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_jepa_ar_loss_step) {
            return print_jepa_ar_loss_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_semantic_alignment_step) {
            return print_semantic_alignment_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_diffusion_denoise_step) {
            return print_diffusion_denoise_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_seq2seq_cross_attention_step) {
            return print_seq2seq_cross_attention_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_ttt_linear_inner_step) {
            return print_ttt_linear_inner_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_universal_recurrent_step) {
            return print_universal_recurrent_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_universal_act_halt_step) {
            return print_universal_act_halt_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_hnet_byte_patch_step) {
            return print_hnet_byte_patch_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_jamba_chunk_state_step) {
            return print_jamba_chunk_state_smoke_json(cfg, argv[0]);
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
