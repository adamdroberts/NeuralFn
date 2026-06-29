#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#error "nfn_native currently targets POSIX execvp environments."
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

constexpr std::string_view kDefaultTrainCommand = "nfn_native_train";
constexpr std::string_view kDefaultGptCommand = "nfn_gpt_native_train";

std::string env_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

bool env_is_empty(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr || value[0] == '\0';
}

void setenv_default_if_empty(const char* name, const char* value) {
    if (value != nullptr && value[0] != '\0' && env_is_empty(name)) {
        setenv(name, value, 1);
    }
}

std::string shell_quote(const std::string& value) {
    if (value.empty()) {
        return "''";
    }
    bool simple = true;
    for (char ch : value) {
        if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '-' ||
              ch == '.' || ch == '/' || ch == ':' || ch == '=')) {
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

std::string sibling_or_path(const char* program, const char* env_name, std::string_view command_name) {
    std::string override = env_or_empty(env_name);
    if (!override.empty()) {
        return override;
    }
    fs::path exe_path(program);
    if (exe_path.has_parent_path()) {
        fs::path sibling = exe_path.parent_path() / std::string(command_name);
        if (fs::exists(sibling)) {
            return sibling.string();
        }
    }
    fs::path local_build = fs::current_path() / "build" / std::string(command_name);
    if (fs::exists(local_build)) {
        return local_build.string();
    }
    return std::string(command_name);
}

bool is_model_checkpoint_name(const std::string& name) {
    if (name.size() != std::string("model_00000000.bin").size()) {
        return false;
    }
    if (name.rfind("model_", 0) != 0 || name.substr(name.size() - 4) != ".bin") {
        return false;
    }
    for (std::size_t i = 6; i < 14; ++i) {
        if (!std::isdigit(static_cast<unsigned char>(name[i]))) {
            return false;
        }
    }
    return true;
}

long long checkpoint_step_from_name(const std::string& name) {
    if (!is_model_checkpoint_name(name)) {
        return -1;
    }
    return std::stoll(name.substr(6, 8));
}

fs::path resolve_latest_checkpoint(const fs::path& input) {
    std::error_code ec;
    if (!fs::is_directory(input, ec)) {
        return input;
    }
    fs::path best;
    long long best_step = -1;
    for (const fs::directory_entry& entry : fs::directory_iterator(input, ec)) {
        if (ec || !entry.is_regular_file(ec)) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        const long long step = checkpoint_step_from_name(name);
        if (step > best_step) {
            best_step = step;
            best = entry.path();
        }
    }
    return best.empty() ? input : best;
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " train [native training args]\n"
        << "       " << program << " infer --checkpoint PATH --prompt-tokens IDS [sampler args]\n\n"
        << "Compiled NeuralFn native CLI shim. It covers the no-Python native train and\n"
        << "native GPT checkpoint inference paths; graph-backed workflows should use the\n"
        << "Python nfn command.\n\n"
        << "Commands:\n"
        << "  train    Exec nfn_native_train before Python/Torch can start.\n"
        << "  infer    Exec nfn_gpt_native_train --sample-checkpoint/--native-info.\n\n"
        << "Native infer options:\n"
        << "  --checkpoint, --native-checkpoint PATH  Native model_*.bin file or directory.\n"
        << "  --prompt-tokens IDS                     Comma-separated token ids for sampling.\n"
        << "  --native-info                           Inspect checkpoint metadata instead of sampling.\n"
        << "  --print-command                         Print the compiled delegate command.\n";
}

std::string require_value(int argc, char** argv, int* index, const std::string& flag) {
    if (*index + 1 >= argc) {
        std::cerr << flag << " requires a value\n";
        std::exit(2);
    }
    *index += 1;
    return argv[*index];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc <= 1 || std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h") {
        print_usage(argv[0]);
        return 0;
    }

    const std::string command_name(argv[1]);
    if (command_name == "train") {
        std::vector<std::string> command;
        command.push_back(sibling_or_path(argv[0], "NFN_NATIVE_TRAIN_CLI", kDefaultTrainCommand));
        for (int i = 2; i < argc; ++i) {
            command.emplace_back(argv[i]);
        }
        setenv_default_if_empty("CUDA_VISIBLE_DEVICES", "0");
        setenv_default_if_empty("CUDA_DEVICE_MAX_CONNECTIONS", "1");
        setenv_default_if_empty("CUDA_MODULE_LOADING", "LAZY");
        return exec_command(command);
    }

    if (command_name != "infer") {
        std::cerr << "Unsupported native nfn command '" << command_name
                  << "'. Use 'train' or 'infer'.\n";
        return 2;
    }

    std::vector<std::string> forwarded;
    std::string checkpoint;
    bool native_info = false;
    bool print_delegate = false;
    bool has_sampling_action = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg(argv[i]);
        auto after_equals = [&](const std::string& prefix) -> std::string {
            return arg.substr(prefix.size());
        };
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--checkpoint" || arg == "--native-checkpoint") {
            checkpoint = require_value(argc, argv, &i, arg);
            continue;
        }
        if (arg.rfind("--checkpoint=", 0) == 0) {
            checkpoint = after_equals("--checkpoint=");
            continue;
        }
        if (arg.rfind("--native-checkpoint=", 0) == 0) {
            checkpoint = after_equals("--native-checkpoint=");
            continue;
        }
        if (arg == "--runtime" || arg == "--kernel-backend" || arg == "--device") {
            require_value(argc, argv, &i, arg);
            continue;
        }
        if (arg.rfind("--runtime=", 0) == 0 ||
            arg.rfind("--kernel-backend=", 0) == 0 ||
            arg.rfind("--device=", 0) == 0) {
            continue;
        }
        if (arg == "--native-info") {
            native_info = true;
            continue;
        }
        if (arg == "--print-command" || arg == "--native-cuda-print-command") {
            print_delegate = true;
            continue;
        }
        if (arg == "--prompt-tokens" || arg.rfind("--prompt-tokens=", 0) == 0) {
            has_sampling_action = true;
        }
        forwarded.push_back(std::move(arg));
    }

    if (checkpoint.empty()) {
        std::cerr << "nfn-native infer requires --checkpoint or --native-checkpoint\n";
        return 2;
    }

    const fs::path resolved_checkpoint = resolve_latest_checkpoint(fs::path(checkpoint));
    if (!fs::exists(resolved_checkpoint)) {
        std::cerr << "native checkpoint does not exist: " << resolved_checkpoint.string() << '\n';
        return 2;
    }

    std::vector<std::string> command;
    command.push_back(sibling_or_path(argv[0], "NFN_NATIVE_GPT_CLI", kDefaultGptCommand));
    if (native_info) {
        command.push_back("--native-info");
        command.push_back("--native-checkpoint");
        command.push_back(resolved_checkpoint.string());
    } else {
        if (!has_sampling_action) {
            std::cerr << "nfn-native infer sampling requires --prompt-tokens IDS\n";
            return 2;
        }
        command.push_back("--sample-checkpoint");
        command.push_back(resolved_checkpoint.string());
        command.insert(command.end(), forwarded.begin(), forwarded.end());
    }

    setenv_default_if_empty("CUDA_VISIBLE_DEVICES", "0");
    setenv_default_if_empty("CUDA_DEVICE_MAX_CONNECTIONS", "1");
    setenv_default_if_empty("CUDA_MODULE_LOADING", "LAZY");
    if (print_delegate) {
        print_command(command);
        return 0;
    }
    return exec_command(command);
}
