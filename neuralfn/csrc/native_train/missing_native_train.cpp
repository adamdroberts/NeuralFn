#include <algorithm>
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
    std::vector<std::string> unparsed_args;
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
        << "  --tile-ops-lib PATH       Override libnfn_native_train_tile_ops.so\n"
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

}  // namespace

int main(int argc, char** argv) {
    const Config cfg = parse_args(argc, argv);
    try {
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
        << "family's CUDA Tile C++ kernels first. For local graph-backed debugging, call the Python "
        << "SDK trainer APIs directly instead of routing through nfn train.\n";
    return 2;
}
