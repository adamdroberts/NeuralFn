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
    std::string_view notes;
};

constexpr ModelEntry MODEL_REGISTRY[] = {
    {
        "gpt",
        "implemented",
        "nfn_gpt_native_train",
        "Dense GPT aliases to the NeuralFn Tile-CUDA transformer-LM loop; template/custom graph selection decides the GPT architecture.",
    },
    {
        "gpt2",
        "implemented",
        "nfn_gpt_native_train",
        "GPT-2 is a dense GPT template/default shape on the NeuralFn Tile-CUDA transformer-LM loop.",
    },
    {
        "gpt3",
        "implemented",
        "nfn_gpt_native_train",
        "GPT-3-style dense decoder training uses the same GPT native target; context/window and width come from the selected template or custom graph.",
    },
    {
        "gpt2-evo",
        "missing-native-trainer",
        "nfn_gpt2_evo_native_train",
        "Layer-evo mutation/evaluation still needs a native CUDA Tile C++ trainer.",
    },
    {
        "nanogpt",
        "partial-native-trainer",
        "nfn_nanogpt_native_train",
        "NanoGPT has a native --train-token-lm loop over cached shards; full transformer training still needs model-wide trainer integration.",
    },
    {
        "llama",
        "missing-native-trainer",
        "nfn_llama_native_train",
        "LLaMA/RoPE/SwiGLU training needs a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "mixllama",
        "missing-native-trainer",
        "nfn_mixllama_native_train",
        "MoE routing and expert kernels need a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "jepa",
        "missing-native-trainer",
        "nfn_jepa_native_train",
        "Semantic/JEPA objectives need a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "semantic-router-moe",
        "missing-native-trainer",
        "nfn_semantic_router_moe_native_train",
        "Semantic router MoE training needs a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "deepseek-v4",
        "missing-native-trainer",
        "nfn_deepseek_v4_native_train",
        "DeepSeek-style sparse/MoE variants need a dedicated native CUDA Tile C++ trainer.",
    },
};

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

void print_model_table() {
    std::cout << "Native NeuralFn training coverage:\n";
    for (const ModelEntry& entry : MODEL_REGISTRY) {
        std::cout << "  " << entry.name << ": " << entry.status << " -> " << entry.native_target << '\n';
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
        << "Dispatches dense GPT/GPT-2/GPT-3 aliases to nfn_gpt_native_train and partial/missing\n"
        << "families to compiled per-family targets before any Python/Torch runtime can start.\n"
        << "Options:\n"
        << "  --base-model, --model NAME      Model family. Dense GPT aliases: gpt, gpt2, gpt3; partial: nanogpt --train-token-lm\n"
        << "  --native-gpt-cli PATH           Override the dense GPT native cached-shard CLI\n"
        << "  --native-gpt2-cli PATH          Compatibility override for the dense GPT native cached-shard CLI\n"
        << "  NFN_NATIVE_<MODEL>_CLI=PATH     Override a per-family native trainer, for example NFN_NATIVE_NANOGPT_CLI\n"
        << "  --list-models                   Print native training coverage\n"
        << "  --json                          Use JSON with --list-models\n"
        << "  --help                          Show this help\n\n"
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
        fs::path sibling = exe_path.parent_path() / "nfn_gpt_native_train";
        if (fs::exists(sibling)) {
            return sibling.string();
        }
        fs::path legacy_sibling = exe_path.parent_path() / "nfn_gpt2_native_train";
        if (fs::exists(legacy_sibling)) {
            return legacy_sibling.string();
        }
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
        if (arg == "--dry-run" || arg == "--native-cuda-dry-run") {
            forwarded.push_back(arg);
            continue;
        }
        if (arg == "--print-command" || arg == "--native-cuda-print-command") {
            print_command_requested = true;
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

    if (model_entry->status != std::string_view("implemented") &&
        model_entry->status != std::string_view("external-fast-path")) {
        const bool dense_gpt =
            model_entry->name == std::string_view("gpt") ||
            model_entry->name == std::string_view("gpt2") ||
            model_entry->name == std::string_view("gpt3");
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
    }
    command.insert(command.end(), forwarded.begin(), forwarded.end());
    if (print_command_requested) {
        print_command(command);
        return 0;
    }
    return exec_command(command);
}
