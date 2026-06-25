#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#error "nfn_native_train currently targets POSIX execvp environments."
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

struct ModelEntry {
    std::string_view name;
    std::string_view status;
    std::string_view native_target;
    std::string_view transformer_lm_status;
    std::string_view token_lm_status;
    std::string_view geometry_status;
    std::string_view notes;
};

constexpr ModelEntry MODEL_REGISTRY[] = {
    {
        "gpt",
        "implemented",
        "nfn_gpt_native_train",
        "native-transformer-lm",
        "not-applicable",
        "gpt2-compatible-fixed-dense-transformer",
        "Dense GPT aliases to the NeuralFn Tile-CUDA transformer-LM loop; template/custom graph selection decides the GPT architecture.",
    },
    {
        "gpt2",
        "implemented",
        "nfn_gpt_native_train",
        "native-transformer-lm",
        "not-applicable",
        "gpt2-compatible-fixed-dense-transformer",
        "GPT-2 is a dense GPT template/default shape on the NeuralFn Tile-CUDA transformer-LM loop.",
    },
    {
        "gpt3",
        "implemented",
        "nfn_gpt_native_train",
        "native-transformer-lm",
        "not-applicable",
        "gpt2-compatible-fixed-dense-transformer-with-gpt3-context",
        "GPT-3-style dense decoder training uses the same GPT native target; context/window and width come from the selected template or custom graph.",
    },
    {
        "gpt2-evo",
        "implemented",
        "nfn_gpt2_evo_native_train",
        "native-dense-gpt-layer-evo-delegate",
        "not-applicable",
        "dense-gpt2-compatible-layer-evo-delegate",
        "GPT-2 evo is a model-aware native C++ preflight/delegate that dispatches dense GPT-2-compatible runs to the CUDA Tile transformer-LM loop with --layer-evo.",
    },
    {
        "nanogpt",
        "implemented",
        "nfn_gpt_native_train",
        "native-transformer-lm",
        "implemented",
        "dense-gpt-template-geometry",
        "NanoGPT routes to the shared dense GPT target with --template-name nanogpt; the native loop now uses the selected 320-wide/5-head/5-layer dense GPT geometry. Pass --train-token-lm for the token-only native preflight.",
    },
    {
        "llama",
        "missing-native-trainer",
        "nfn_llama_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-rope-swiglu-native-loop",
        "LLaMA/RoPE/SwiGLU training needs a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "mixllama",
        "missing-native-trainer",
        "nfn_mixllama_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-moe-routing-native-loop",
        "MoE routing and expert kernels need a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "jepa",
        "missing-native-trainer",
        "nfn_jepa_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-jepa-objective-native-loop",
        "Semantic/JEPA objectives need a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "semantic-router-moe",
        "missing-native-trainer",
        "nfn_semantic_router_moe_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-semantic-router-moe-native-loop",
        "Semantic router MoE training needs a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "deepseek-v4",
        "missing-native-trainer",
        "nfn_deepseek_v4_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-deepseek-sparse-moe-native-loop",
        "DeepSeek-style sparse/MoE variants need a dedicated native CUDA Tile C++ trainer.",
    },
};

constexpr std::string_view DEFAULT_TINYSTORIES_ALIAS = "roneneldan__TinyStories__TinyStoriesV2-GPT4";

std::string env_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

std::string lower_model(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (ch == '_') {
            ch = '-';
        }
    }
    if (value == "nano-gpt") {
        return "nanogpt";
    }
    return value;
}

std::string shell_quote(const std::string& value) {
    if (value.empty()) {
        return "''";
    }
    bool simple = true;
    for (char ch : value) {
        if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '-' || ch == '.' || ch == '/' || ch == ':' || ch == '=')) {
            simple = false;
            break;
        }
    }
    if (simple) {
        return value;
    }
    std::ostringstream out;
    out << '\'';
    for (char ch : value) {
        if (ch == '\'') {
            out << "'\\''";
        } else {
            out << ch;
        }
    }
    out << '\'';
    return out.str();
}

void print_command(const std::vector<std::string>& command) {
    for (std::size_t i = 0; i < command.size(); ++i) {
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << shell_quote(command[i]);
    }
    std::cout << '\n';
    std::cout.flush();
}

const ModelEntry* find_model(std::string_view name) {
    for (const ModelEntry& entry : MODEL_REGISTRY) {
        if (entry.name == name) {
            return &entry;
        }
    }
    return nullptr;
}

bool has_forwarded_flag(const std::vector<std::string>& args, const std::string_view flag) {
    return std::find(args.begin(), args.end(), flag) != args.end();
}

bool has_forwarded_value_flag(const std::vector<std::string>& args, const std::string_view flag) {
    const std::string prefix(flag);
    for (const std::string& arg : args) {
        if (arg == prefix || arg.rfind(prefix + "=", 0) == 0) {
            return true;
        }
    }
    return false;
}

bool has_template_or_graph_selector(const std::vector<std::string>& args) {
    return has_forwarded_value_flag(args, "--template-name") ||
           has_forwarded_value_flag(args, "--template") ||
           has_forwarded_value_flag(args, "--preset") ||
           has_forwarded_value_flag(args, "--graph-file") ||
           has_forwarded_value_flag(args, "--graph");
}

bool has_any_forwarded_value_flag(const std::vector<std::string>& args, const std::vector<std::string_view>& flags) {
    for (std::string_view flag : flags) {
        if (has_forwarded_value_flag(args, flag)) {
            return true;
        }
    }
    return false;
}

bool has_native_train_action(const std::vector<std::string>& args) {
    static constexpr std::string_view ACTION_FLAGS[] = {
        "--check-tile-ops",
        "--print-plan",
        "--sample-token-batch",
        "--smoke-attention-step",
        "--smoke-embedding-lm-step",
        "--smoke-embedding-norm-step",
        "--smoke-fused-qkv-attention-step",
        "--smoke-lm-step",
        "--smoke-mlp-step",
        "--smoke-norm-residual-step",
        "--smoke-optimizer-step",
        "--smoke-qkv-layout-step",
        "--smoke-tile-ops",
        "--smoke-token-train-step",
        "--smoke-training-loop-step",
        "--smoke-transformer-block-step",
        "--smoke-transformer-lm-step",
        "--train-embedding-lm",
        "--train-token-lm",
        "--train-transformer-lm",
    };
    for (std::string_view flag : ACTION_FLAGS) {
        if (has_forwarded_flag(args, flag)) {
            return true;
        }
    }
    return false;
}

void append_value_arg(std::vector<std::string>& args, std::string flag, std::string value) {
    args.push_back(std::move(flag));
    args.push_back(std::move(value));
}

std::string output_dir_from_output(const std::string& value) {
    fs::path path(value);
    if (path.has_extension()) {
        path.replace_extension();
    }
    return path.string();
}

bool arg_is_any(const std::string& arg, const std::vector<std::string_view>& flags) {
    for (std::string_view flag : flags) {
        if (arg == flag) {
            return true;
        }
    }
    return false;
}

std::string matched_equals_flag(const std::string& arg, const std::vector<std::string_view>& flags) {
    for (std::string_view flag : flags) {
        const std::string prefix(flag);
        if (arg.rfind(prefix + "=", 0) == 0) {
            return prefix;
        }
    }
    return std::string();
}

void append_dataset_alias(std::vector<std::string>& args, const std::string& raw_dataset) {
    const std::string dataset = lower_model(raw_dataset);
    if (dataset == "tinystories") {
        args.push_back("--tinystories");
    } else if (dataset == "golf1") {
        append_value_arg(args, "--dataset-alias", "willdepueoai__parameter-golf__sp1024__train1");
    } else if (dataset == "golf10") {
        append_value_arg(args, "--dataset-alias", "willdepueoai__parameter-golf__sp1024__train10");
    } else {
        append_value_arg(args, "--dataset-alias", raw_dataset);
    }
}

void print_model_table() {
    std::cout << "Native NeuralFn training coverage:\n";
    for (const ModelEntry& entry : MODEL_REGISTRY) {
        std::cout << "  " << entry.name << ": " << entry.status << " -> " << entry.native_target << '\n';
        std::cout << "    transformer_lm=" << entry.transformer_lm_status
                  << " token_lm=" << entry.token_lm_status
                  << " geometry=" << entry.geometry_status << '\n';
        std::cout << "    " << entry.notes << '\n';
    }
}

void print_model_json() {
    std::cout << "{\n  \"models\": [\n";
    for (std::size_t i = 0; i < std::size(MODEL_REGISTRY); ++i) {
        const ModelEntry& entry = MODEL_REGISTRY[i];
        std::cout
            << "    {\"name\": \"" << entry.name
            << "\", \"status\": \"" << entry.status
            << "\", \"native_target\": \"" << entry.native_target
            << "\", \"transformer_lm_status\": \"" << entry.transformer_lm_status
            << "\", \"token_lm_status\": \"" << entry.token_lm_status
            << "\", \"geometry_status\": \"" << entry.geometry_status
            << "\", \"notes\": \"" << entry.notes << "\"}";
        if (i + 1 != std::size(MODEL_REGISTRY)) {
            std::cout << ',';
        }
        std::cout << '\n';
    }
    std::cout << "  ]\n}\n";
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [train] --base-model MODEL [native options]\n\n"
        << "Unified no-Python NeuralFn native training frontend.\n"
        << "Dispatches dense GPT/GPT-2/GPT-3/NanoGPT aliases to nfn_gpt_native_train and missing\n"
        << "families to compiled per-family targets before any Python/Torch runtime can start.\n"
        << "Options:\n"
        << "  --base-model, --model NAME      Model family. Dense GPT aliases: gpt, gpt2, gpt3, nanogpt\n"
        << "  --native-gpt-cli PATH           Override the dense GPT native cached-shard CLI\n"
        << "  --native-gpt2-cli PATH          Compatibility override for the dense GPT native cached-shard CLI\n"
        << "  NFN_NATIVE_<MODEL>_CLI=PATH     Override a per-family native trainer, for example NFN_NATIVE_NANOGPT_CLI\n"
        << "  --list-models                   Print native training coverage\n"
        << "  --json                          Use JSON with --list-models\n"
        << "  --help                          Show this help\n\n"
        << "Python wrapper aliases such as --tinystories, --dataset tinystories, --output,\n"
        << "--kernel-backend, --native-cuda-*, --template, --preset, and --graph are\n"
        << "normalized here so dense GPT training can start without Python argument shims.\n"
        << "All other options are forwarded to the selected native model trainer.\n";
}

std::string require_value(int argc, char** argv, int* index, const std::string& flag) {
    if (*index + 1 >= argc) {
        std::cerr << flag << " requires a value\n";
        std::exit(2);
    }
    *index += 1;
    return argv[*index];
}

std::string sibling_gpt_cli(const char* program) {
    std::string env_cli = env_or_empty("NFN_NATIVE_GPT_CLI");
    if (!env_cli.empty()) {
        return env_cli;
    }
    env_cli = env_or_empty("NFN_NATIVE_GPT2_CLI");
    if (!env_cli.empty()) {
        return env_cli;
    }
    fs::path exe_path(program);
    if (exe_path.has_parent_path()) {
        fs::path linked_sibling = exe_path.parent_path() / "nfn_gpt_native_train_linked";
        if (fs::exists(linked_sibling)) {
            return linked_sibling.string();
        }
        fs::path sibling = exe_path.parent_path() / "nfn_gpt_native_train";
        if (fs::exists(sibling)) {
            return sibling.string();
        }
        fs::path legacy_sibling = exe_path.parent_path() / "nfn_gpt2_native_train";
        if (fs::exists(legacy_sibling)) {
            return legacy_sibling.string();
        }
    }
    fs::path linked_local_build = fs::current_path() / "build" / "nfn_gpt_native_train_linked";
    if (fs::exists(linked_local_build)) {
        return linked_local_build.string();
    }
    fs::path local_build = fs::current_path() / "build" / "nfn_gpt_native_train";
    if (fs::exists(local_build)) {
        return local_build.string();
    }
    fs::path legacy_local_build = fs::current_path() / "build" / "nfn_gpt2_native_train";
    if (fs::exists(legacy_local_build)) {
        return legacy_local_build.string();
    }
    return "nfn_gpt_native_train";
}

std::string env_name_for_model(std::string_view model) {
    std::string env = "NFN_NATIVE_";
    for (char ch : model) {
        char out = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        if (out == '-') {
            out = '_';
        }
        env.push_back(out);
    }
    env += "_CLI";
    return env;
}

std::string hyphen_command_name(std::string_view target) {
    std::string command(target);
    for (char& ch : command) {
        if (ch == '_') {
            ch = '-';
        }
    }
    return command;
}

std::string resolve_native_target_cli(const char* program, const ModelEntry& entry) {
    const std::string env_name = env_name_for_model(entry.name);
    std::string env_cli = env_or_empty(env_name.c_str());
    if (!env_cli.empty()) {
        return env_cli;
    }
    const std::string native_target(entry.native_target);
    const std::string hyphen_target = hyphen_command_name(entry.native_target);
    fs::path exe_path(program);
    if (exe_path.has_parent_path()) {
        fs::path sibling = exe_path.parent_path() / native_target;
        if (fs::exists(sibling)) {
            return sibling.string();
        }
        fs::path hyphen_sibling = exe_path.parent_path() / hyphen_target;
        if (fs::exists(hyphen_sibling)) {
            return hyphen_sibling.string();
        }
    }
    fs::path local_build = fs::current_path() / "build" / native_target;
    if (fs::exists(local_build)) {
        return local_build.string();
    }
    fs::path local_hyphen_build = fs::current_path() / "build" / hyphen_target;
    if (fs::exists(local_hyphen_build)) {
        return local_hyphen_build.string();
    }
    return std::string();
}

int exec_command(std::vector<std::string>& command) {
    std::vector<char*> exec_args;
    exec_args.reserve(command.size() + 1);
    for (std::string& item : command) {
        exec_args.push_back(item.data());
    }
    exec_args.push_back(nullptr);
    execvp(command[0].c_str(), exec_args.data());
    std::cerr << "Failed to exec " << command[0] << ": " << std::strerror(errno) << '\n';
    return errno == ENOENT ? 127 : 126;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model = "gpt";
    std::string gpt_cli = sibling_gpt_cli(argv[0]);
    std::vector<std::string> forwarded;
    bool print_command_requested = false;
    bool list_models = false;
    bool json_output = false;
    bool saw_separator = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto after_equals = [&](const std::string& prefix) -> std::string {
            return arg.substr(prefix.size());
        };
        if (i == 1 && arg == "train") {
            continue;
        }
        if (arg == "--") {
            saw_separator = true;
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (!saw_separator && arg == "--list-models") {
            list_models = true;
            continue;
        }
        if (!saw_separator && arg == "--json") {
            json_output = true;
            continue;
        }
        if (!saw_separator && arg_is_any(arg, {
                "--runtime",
                "--device",
                "--dataset-hf-path",
                "--dataset-variant",
                "--dataset-train-shards",
                "--dataset-train-file",
                "--dataset-val-file",
                "--tokenizer",
                "--native-cuda-runner",
                "--run-id",
                "--seed",
                "--min-lr",
            })) {
            require_value(argc, argv, &i, arg);
            continue;
        }
        if (!saw_separator && !matched_equals_flag(arg, {
                "--runtime",
                "--device",
                "--dataset-hf-path",
                "--dataset-variant",
                "--dataset-train-shards",
                "--dataset-train-file",
                "--dataset-val-file",
                "--tokenizer",
                "--native-cuda-runner",
                "--run-id",
                "--seed",
                "--min-lr",
            }).empty()) {
            continue;
        }
        if (!saw_separator && arg_is_any(arg, {
                "--download-if-missing",
                "--no-download-if-missing",
                "--tokgpt2",
                "--cl100k",
                "--o200k",
                "--tile-cuda-strict",
                "--no-tile-cuda-strict",
            })) {
            continue;
        }
        if (!saw_separator && (arg == "--base-model" || arg == "--model")) {
            model = lower_model(require_value(argc, argv, &i, arg));
            continue;
        }
        if (!saw_separator && arg.rfind("--base-model=", 0) == 0) {
            model = lower_model(after_equals("--base-model="));
            continue;
        }
        if (!saw_separator && arg.rfind("--model=", 0) == 0) {
            model = lower_model(after_equals("--model="));
            continue;
        }
        if (!saw_separator && (arg == "--native-gpt-cli" || arg == "--native-gpt2-cli")) {
            gpt_cli = require_value(argc, argv, &i, arg);
            continue;
        }
        if (!saw_separator && arg.rfind("--native-gpt-cli=", 0) == 0) {
            gpt_cli = after_equals("--native-gpt-cli=");
            continue;
        }
        if (!saw_separator && arg.rfind("--native-gpt2-cli=", 0) == 0) {
            gpt_cli = after_equals("--native-gpt2-cli=");
            continue;
        }
        if (!saw_separator && arg == "--dataset") {
            append_dataset_alias(forwarded, require_value(argc, argv, &i, arg));
            continue;
        }
        if (!saw_separator && arg.rfind("--dataset=", 0) == 0) {
            append_dataset_alias(forwarded, after_equals("--dataset="));
            continue;
        }
        if (!saw_separator && arg == "--output") {
            append_value_arg(forwarded, "--output-dir", output_dir_from_output(require_value(argc, argv, &i, arg)));
            continue;
        }
        if (!saw_separator && arg.rfind("--output=", 0) == 0) {
            append_value_arg(forwarded, "--output-dir", output_dir_from_output(after_equals("--output=")));
            continue;
        }
        if (!saw_separator && arg_is_any(arg, {
                "--kernel-backend",
                "--native-cuda-kernel-backend",
                "--native-cuda-executable",
                "--native-cuda-output-dir",
                "--native-cuda-tile-ops-lib",
                "--native-cuda-cuda-runtime-lib",
                "--native-cuda-lm-head-row-chunk-size",
                "--template",
                "--preset",
                "--graph",
            })) {
            const std::string value = require_value(argc, argv, &i, arg);
            if (arg == "--kernel-backend" || arg == "--native-cuda-kernel-backend") {
                append_value_arg(forwarded, "--backend", value);
            } else if (arg == "--native-cuda-executable") {
                append_value_arg(forwarded, "--target", value);
            } else if (arg == "--native-cuda-output-dir") {
                append_value_arg(forwarded, "--output-dir", value);
            } else if (arg == "--native-cuda-tile-ops-lib") {
                append_value_arg(forwarded, "--tile-ops-lib", value);
            } else if (arg == "--native-cuda-cuda-runtime-lib") {
                append_value_arg(forwarded, "--cuda-runtime-lib", value);
            } else if (arg == "--native-cuda-lm-head-row-chunk-size") {
                append_value_arg(forwarded, "--lm-head-row-chunk-size", value);
            } else if (arg == "--template" || arg == "--preset") {
                append_value_arg(forwarded, "--template-name", value);
            } else if (arg == "--graph") {
                append_value_arg(forwarded, "--graph-file", value);
            }
            continue;
        }
        const std::string value_alias = matched_equals_flag(arg, {
            "--kernel-backend",
            "--native-cuda-kernel-backend",
            "--native-cuda-executable",
            "--native-cuda-output-dir",
            "--native-cuda-tile-ops-lib",
            "--native-cuda-cuda-runtime-lib",
            "--native-cuda-lm-head-row-chunk-size",
            "--template",
            "--preset",
            "--graph",
        });
        if (!saw_separator && !value_alias.empty()) {
            const std::string value = arg.substr(value_alias.size() + 1);
            if (value_alias == "--kernel-backend" || value_alias == "--native-cuda-kernel-backend") {
                append_value_arg(forwarded, "--backend", value);
            } else if (value_alias == "--native-cuda-executable") {
                append_value_arg(forwarded, "--target", value);
            } else if (value_alias == "--native-cuda-output-dir") {
                append_value_arg(forwarded, "--output-dir", value);
            } else if (value_alias == "--native-cuda-tile-ops-lib") {
                append_value_arg(forwarded, "--tile-ops-lib", value);
            } else if (value_alias == "--native-cuda-cuda-runtime-lib") {
                append_value_arg(forwarded, "--cuda-runtime-lib", value);
            } else if (value_alias == "--native-cuda-lm-head-row-chunk-size") {
                append_value_arg(forwarded, "--lm-head-row-chunk-size", value);
            } else if (value_alias == "--template" || value_alias == "--preset") {
                append_value_arg(forwarded, "--template-name", value);
            } else if (value_alias == "--graph") {
                append_value_arg(forwarded, "--graph-file", value);
            }
            continue;
        }
        if (!saw_separator && arg_is_any(arg, {
                "--native-cuda-print-plan",
                "--native-cuda-check-tile-ops",
                "--native-cuda-smoke-tile-ops",
                "--native-cuda-smoke-optimizer-step",
                "--native-cuda-smoke-lm-step",
                "--native-cuda-smoke-attention-step",
                "--native-cuda-smoke-mlp-step",
                "--native-cuda-smoke-norm-residual-step",
                "--native-cuda-smoke-transformer-block-step",
                "--native-cuda-smoke-transformer-lm-step",
                "--native-cuda-smoke-embedding-lm-step",
                "--native-cuda-allow-train-val-fallback",
                "--native-cuda-no-checkpoint",
                "--native-cuda-write-checkpoint",
                "--native-cuda-require-cooperative-lm-head-backward",
                "--require-cooperative-lm-head-backward",
                "--native-cuda-startup-only",
            })) {
            if (arg == "--native-cuda-print-plan") {
                forwarded.push_back("--print-plan");
            } else if (arg == "--native-cuda-check-tile-ops") {
                forwarded.push_back("--check-tile-ops");
            } else if (arg == "--native-cuda-smoke-tile-ops") {
                forwarded.push_back("--smoke-tile-ops");
            } else if (arg == "--native-cuda-smoke-optimizer-step") {
                forwarded.push_back("--smoke-optimizer-step");
            } else if (arg == "--native-cuda-smoke-lm-step") {
                forwarded.push_back("--smoke-lm-step");
            } else if (arg == "--native-cuda-smoke-attention-step") {
                forwarded.push_back("--smoke-attention-step");
            } else if (arg == "--native-cuda-smoke-mlp-step") {
                forwarded.push_back("--smoke-mlp-step");
            } else if (arg == "--native-cuda-smoke-norm-residual-step") {
                forwarded.push_back("--smoke-norm-residual-step");
            } else if (arg == "--native-cuda-smoke-transformer-block-step") {
                forwarded.push_back("--smoke-transformer-block-step");
            } else if (arg == "--native-cuda-smoke-transformer-lm-step") {
                forwarded.push_back("--smoke-transformer-lm-step");
            } else if (arg == "--native-cuda-smoke-embedding-lm-step") {
                forwarded.push_back("--smoke-embedding-lm-step");
            } else if (arg == "--native-cuda-allow-train-val-fallback") {
                forwarded.push_back("--allow-train-val-fallback");
            } else if (arg == "--native-cuda-no-checkpoint") {
                forwarded.push_back("--no-checkpoint");
            } else if (arg == "--native-cuda-write-checkpoint") {
                forwarded.push_back("--write-checkpoint");
            } else if (arg == "--native-cuda-require-cooperative-lm-head-backward" ||
                       arg == "--require-cooperative-lm-head-backward") {
                forwarded.push_back("--require-cooperative-lm-head-backward");
            } else if (arg == "--native-cuda-startup-only") {
                forwarded.push_back("--startup-only");
            }
            continue;
        }
        if (arg == "--dry-run" || arg == "--native-cuda-dry-run") {
            forwarded.push_back("--dry-run");
            continue;
        }
        if (arg == "--print-command" || arg == "--native-cuda-print-command") {
            print_command_requested = true;
            continue;
        }
        forwarded.push_back(arg);
    }

    if (list_models) {
        if (json_output) {
            print_model_json();
        } else {
            print_model_table();
        }
        return 0;
    }

    const ModelEntry* model_entry = find_model(model);
    if (model_entry == nullptr) {
        std::cerr
            << "No native C++ trainer is registered for model family '" << model << "'.\n"
            << "Current native training coverage:\n";
        for (const ModelEntry& entry : MODEL_REGISTRY) {
            std::cerr << "  " << entry.name << ": " << entry.status << " -> " << entry.native_target << '\n';
        }
        std::cerr << "Build a native CUDA Tile C++ trainer for this family before running it.\n";
        return 2;
    }

    const bool dense_gpt =
        model_entry->name == std::string_view("gpt") ||
        model_entry->name == std::string_view("gpt2") ||
        model_entry->name == std::string_view("gpt3") ||
        model_entry->name == std::string_view("nanogpt");

    const bool nanogpt_token_lm =
        model_entry->name == std::string_view("nanogpt") &&
        has_forwarded_flag(forwarded, "--train-token-lm");

    if (dense_gpt && !has_native_train_action(forwarded)) {
        forwarded.push_back("--train-transformer-lm");
    }
    if (dense_gpt && !has_forwarded_value_flag(forwarded, "--backend")) {
        append_value_arg(forwarded, "--backend", "tile-cuda");
    }
    if (
        dense_gpt &&
        !has_any_forwarded_value_flag(forwarded, {"--dataset-alias", "--dataset-path"}) &&
        !has_forwarded_flag(forwarded, "--tinystories")
    ) {
        const std::string dataset_alias = env_or_empty("DATASET_ALIAS");
        append_value_arg(
            forwarded,
            "--dataset-alias",
            dataset_alias.empty() ? std::string(DEFAULT_TINYSTORIES_ALIAS) : dataset_alias);
    }
    if (
        model_entry->name == std::string_view("gpt3") &&
        !has_forwarded_value_flag(forwarded, "--train-seq-len") &&
        !has_template_or_graph_selector(forwarded)
    ) {
        append_value_arg(forwarded, "--train-seq-len", "2048");
    }

    const bool dispatchable_native_target =
        model_entry->status == std::string_view("implemented") ||
        model_entry->status == std::string_view("partial-native-trainer") ||
        model_entry->status == std::string_view("external-fast-path");
    if (!dispatchable_native_target) {
        const std::string target_cli =
            dense_gpt && !gpt_cli.empty()
                ? gpt_cli
                : resolve_native_target_cli(argv[0], *model_entry);
        if (!target_cli.empty()) {
            if (std::getenv("CUDA_VISIBLE_DEVICES") == nullptr) {
                setenv("CUDA_VISIBLE_DEVICES", "0", 0);
            }
            if (std::getenv("CUDA_DEVICE_MAX_CONNECTIONS") == nullptr) {
                setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1", 0);
            }
            if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
                setenv("CUDA_MODULE_LOADING", "LAZY", 0);
            }
            std::vector<std::string> command;
            command.push_back(target_cli);
            if (dense_gpt) {
                command.push_back("--model-family");
                command.push_back(std::string(model_entry->name));
            }
            command.insert(command.end(), forwarded.begin(), forwarded.end());
            if (print_command_requested) {
                command.push_back("--print-command");
                print_command(command);
                return 0;
            }
            return exec_command(command);
        }
        std::cerr
            << "No native C++ trainer is registered for model family '" << model << "'.\n"
            << "Current native training coverage:\n";
        for (const ModelEntry& entry : MODEL_REGISTRY) {
            std::cerr << "  " << entry.name << ": " << entry.status << " -> " << entry.native_target << '\n';
        }
        std::cerr << "Build a native CUDA Tile C++ trainer for this family before running it.\n";
        return 2;
    }

    if (nanogpt_token_lm) {
        const std::string target_cli = resolve_native_target_cli(
            argv[0],
            ModelEntry{
                "nanogpt",
                "implemented",
                "nfn_nanogpt_native_train",
                "not-applicable",
                "implemented",
                "token-lm-only",
                "Explicit NanoGPT token-only native trainer.",
            });
        if (target_cli.empty()) {
            std::cerr << "No NanoGPT token-LM native CLI configured.\n";
            return 2;
        }
        std::vector<std::string> command;
        command.push_back(target_cli);
        command.insert(command.end(), forwarded.begin(), forwarded.end());
        if (print_command_requested) {
            command.push_back("--print-command");
            print_command(command);
            return 0;
        }
        return exec_command(command);
    }

    if (!dense_gpt) {
        const std::string target_cli = resolve_native_target_cli(argv[0], *model_entry);
        if (target_cli.empty()) {
            std::cerr
                << "No native C++ trainer is available for model family '" << model << "'.\n"
                << "Expected target: " << model_entry->native_target << '\n';
            return 2;
        }
        std::vector<std::string> command;
        command.push_back(target_cli);
        command.insert(command.end(), forwarded.begin(), forwarded.end());
        if (print_command_requested) {
            command.push_back("--print-command");
            print_command(command);
            return 0;
        }
        return exec_command(command);
    }

    if (gpt_cli.empty()) {
        std::cerr << "No GPT native CLI configured.\n";
        return 2;
    }

    if (std::getenv("CUDA_VISIBLE_DEVICES") == nullptr) {
        setenv("CUDA_VISIBLE_DEVICES", "0", 0);
    }
    if (std::getenv("CUDA_DEVICE_MAX_CONNECTIONS") == nullptr) {
        setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1", 0);
    }
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
        setenv("CUDA_MODULE_LOADING", "LAZY", 0);
    }

    std::vector<std::string> command;
    command.push_back(gpt_cli);
    if (model_entry->name == std::string_view("gpt") ||
        model_entry->name == std::string_view("gpt2") ||
        model_entry->name == std::string_view("gpt3")) {
        command.push_back("--model-family");
        command.push_back(std::string(model_entry->name));
    } else if (model_entry->name == std::string_view("nanogpt")) {
        command.push_back("--model-family");
        command.push_back("gpt");
        if (!has_template_or_graph_selector(forwarded)) {
            command.push_back("--template-name");
            command.push_back("nanogpt");
        }
    }
    command.insert(command.end(), forwarded.begin(), forwarded.end());
    if (print_command_requested) {
        print_command(command);
        return 0;
    }
    return exec_command(command);
}
