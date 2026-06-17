#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#error "nfn_gpt2_tile_train currently targets POSIX execvp environments."
#else
#include <unistd.h>
#endif

namespace {

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

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [--target PATH] [--dry-run] [--print-command] -- <train_gpt2cu args...>\n"
        << "\n"
        << "Sets CUDA_VISIBLE_DEVICES=0, CUDA_DEVICE_MAX_CONNECTIONS=1, and CUDA_MODULE_LOADING=LAZY when unset, then execs the target GPT-2 CUDA trainer.\n"
        << "Target resolution: --target, then NFN_NATIVE_GPT2_TRAIN_BIN, then train_gpt2cu.\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string target;
    const char* env_target = std::getenv("NFN_NATIVE_GPT2_TRAIN_BIN");
    if (env_target != nullptr && env_target[0] != '\0') {
        target = env_target;
    } else {
        target = "train_gpt2cu";
    }

    bool dry_run = false;
    bool print_command = false;
    std::vector<std::string> passthrough;
    bool after_separator = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (after_separator) {
            passthrough.push_back(arg);
            continue;
        }
        if (arg == "--") {
            after_separator = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--dry-run") {
            dry_run = true;
        } else if (arg == "--print-command") {
            print_command = true;
        } else if (arg == "--target") {
            if (i + 1 >= argc) {
                std::cerr << "--target requires a value\n";
                return 2;
            }
            target = argv[++i];
        } else if (arg.rfind("--target=", 0) == 0) {
            target = arg.substr(std::strlen("--target="));
        } else {
            passthrough.push_back(arg);
        }
    }

    if (target.empty()) {
        std::cerr << "No GPT-2 CUDA trainer target configured.\n";
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
    command.push_back(target);
    command.insert(command.end(), passthrough.begin(), passthrough.end());

    if (print_command || dry_run) {
        for (std::size_t i = 0; i < command.size(); ++i) {
            if (i != 0) {
                std::cout << ' ';
            }
            std::cout << shell_quote(command[i]);
        }
        std::cout << '\n';
    }
    if (dry_run) {
        return 0;
    }

    std::vector<char*> exec_args;
    exec_args.reserve(command.size() + 1);
    for (std::string& item : command) {
        exec_args.push_back(item.data());
    }
    exec_args.push_back(nullptr);

    execvp(target.c_str(), exec_args.data());
    std::cerr << "Failed to exec " << target << ": " << std::strerror(errno) << '\n';
    return errno == ENOENT ? 127 : 126;
}
