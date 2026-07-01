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
constexpr std::string_view kDefaultTinyStoriesAlias = "roneneldan__TinyStories__TinyStoriesV2-GPT4";

std::string require_value(int argc, char** argv, int* index, const std::string& flag);

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

std::string sibling_gpt_cli(const char* program) {
    std::string override = env_or_empty("NFN_NATIVE_GPT_CLI");
    if (!override.empty()) {
        return override;
    }
    override = env_or_empty("NFN_NATIVE_GPT2_CLI");
    if (!override.empty()) {
        return override;
    }
    fs::path exe_path(program);
    if (exe_path.has_parent_path()) {
        for (std::string_view name : {
                 "nfn_gpt_native_train_linked",
                 "nfn_gpt_native_train",
                 "nfn_gpt2_native_train",
             }) {
            fs::path sibling = exe_path.parent_path() / std::string(name);
            if (fs::exists(sibling)) {
                return sibling.string();
            }
        }
    }
    for (std::string_view name : {
             "nfn_gpt_native_train_linked",
             "nfn_gpt_native_train",
             "nfn_gpt2_native_train",
         }) {
        fs::path local_build = fs::current_path() / "build" / std::string(name);
        if (fs::exists(local_build)) {
            return local_build.string();
        }
    }
    return std::string(kDefaultGptCommand);
}

bool is_dense_gpt_model(std::string_view model) {
    return model == "gpt" || model == "gpt2" || model == "gpt3" || model == "nanogpt";
}

bool has_forwarded_flag(const std::vector<std::string>& args, std::string_view flag) {
    return std::find(args.begin(), args.end(), flag) != args.end();
}

bool has_forwarded_value_flag(const std::vector<std::string>& args, std::string_view flag) {
    const std::string prefix(flag);
    for (const std::string& arg : args) {
        if (arg == prefix || arg.rfind(prefix + "=", 0) == 0) {
            return true;
        }
    }
    return false;
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

bool has_any_forwarded_value_flag(
    const std::vector<std::string>& args,
    const std::vector<std::string_view>& flags) {
    for (std::string_view flag : flags) {
        if (has_forwarded_value_flag(args, flag)) {
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

bool has_native_train_action(const std::vector<std::string>& args) {
    static constexpr std::string_view kActionFlags[] = {
        "--check-tile-ops",
        "--list-templates",
        "--native-cuda-list-templates",
        "--print-plan",
        "--sample-token-batch",
        "--smoke-attention-step",
        "--smoke-embedding-lm-step",
        "--smoke-embedding-norm-step",
        "--smoke-family-layout-checkpoint-step",
        "--smoke-dense-jepa-train-step",
        "--smoke-hnet-byte-patch-step",
        "--smoke-hnet-byte-patch-backward-step",
        "--smoke-hnet-byte-lm-loop-step",
        "--smoke-jamba-chunk-state-step",
        "--smoke-jamba-mamba-state-step",
        "--smoke-jamba-layer-schedule-step",
        "--smoke-jepa-ar-loss-step",
        "--smoke-jepa-target-encoder-step",
        "--smoke-diffusion-denoise-step",
        "--smoke-diffusion-objective-step",
        "--smoke-diffusion-full-loop-step",
        "--smoke-fused-qkv-attention-step",
        "--smoke-llama-attention-block-step",
        "--smoke-llama-token-lm-train-step",
        "--smoke-llama-composed-train-step",
        "--smoke-llama-full-loop-step",
        "--smoke-llama-rope-attention-block-step",
        "--smoke-llama-rope-block-train-step",
        "--smoke-llama-packed-attention-step",
        "--smoke-lm-step",
        "--smoke-mlp-step",
        "--smoke-moe-transformer-block-step",
        "--smoke-moe-transformer-block-train-step",
        "--smoke-moe-transformer-lm-train-step",
        "--smoke-norm-residual-step",
        "--smoke-nvfp4-pack",
        "--smoke-optimizer-step",
        "--smoke-qkv-layout-step",
        "--smoke-semantic-dense-jepa-train-step",
        "--smoke-semantic-router-moe-train-step",
        "--smoke-seq2seq-cross-attention-step",
        "--smoke-seq2seq-full-encoder-decoder-loop-step",
        "--smoke-seq2seq-loss-composition-step",
        "--smoke-ttt-composite-inner-step",
        "--smoke-ttt-full-transformer-loop-step",
        "--smoke-ttt-linear-inner-step",
        "--smoke-universal-act-halt-step",
        "--smoke-universal-recurrent-step",
        "--smoke-universal-transformer-loop-step",
        "--smoke-tile-ops",
        "--smoke-token-train-step",
        "--smoke-training-loop-step",
        "--smoke-transformer-block-step",
        "--smoke-transformer-lm-step",
        "--train-embedding-lm",
        "--train-token-lm",
        "--train-transformer-lm",
    };
    for (std::string_view flag : kActionFlags) {
        if (has_forwarded_flag(args, flag)) {
            return true;
        }
    }
    return false;
}

bool has_template_catalog_action(const std::vector<std::string>& args) {
    return has_forwarded_flag(args, "--list-templates") ||
           has_forwarded_flag(args, "--native-cuda-list-templates");
}

bool has_dataset_free_preflight_action(const std::vector<std::string>& args) {
    return has_forwarded_flag(args, "--check-tile-ops") ||
           has_forwarded_flag(args, "--smoke-nvfp4-pack") ||
           has_forwarded_flag(args, "--smoke-tile-ops");
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

struct DenseTrainCommand {
    bool can_bypass = false;
    bool print_command_requested = false;
    std::vector<std::string> argv;
};

DenseTrainCommand build_dense_gpt_train_command(int argc, char** argv) {
    DenseTrainCommand result;
    std::string model = "gpt";
    std::vector<std::string> forwarded;
    bool saw_separator = false;
    bool list_models = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg(argv[i]);
        auto after_equals = [&](const std::string& prefix) -> std::string {
            return arg.substr(prefix.size());
        };
        if (arg == "--") {
            saw_separator = true;
            continue;
        }
        if (!saw_separator && arg == "--help") {
            forwarded.push_back(arg);
            continue;
        }
        if (!saw_separator && arg == "--list-models") {
            list_models = true;
            forwarded.push_back(arg);
            continue;
        }
        if (!saw_separator && arg == "--json") {
            forwarded.push_back(arg);
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
            // The top-level CLI resolves the same overrides through env vars; keep
            // this option for the unified fallback instead of silently ignoring it.
            return result;
        }
        if (!saw_separator && (arg.rfind("--native-gpt-cli=", 0) == 0 ||
                               arg.rfind("--native-gpt2-cli=", 0) == 0)) {
            return result;
        }
        if (!saw_separator && (arg == "--dataset")) {
            append_dataset_alias(forwarded, require_value(argc, argv, &i, arg));
            continue;
        }
        if (!saw_separator && arg.rfind("--dataset=", 0) == 0) {
            append_dataset_alias(forwarded, after_equals("--dataset="));
            continue;
        }
        if (!saw_separator && (arg == "--output" || arg == "--artifact-path")) {
            append_value_arg(forwarded, "--output-dir", output_dir_from_output(require_value(argc, argv, &i, arg)));
            continue;
        }
        if (!saw_separator && arg.rfind("--output=", 0) == 0) {
            append_value_arg(forwarded, "--output-dir", output_dir_from_output(after_equals("--output=")));
            continue;
        }
        if (!saw_separator && arg.rfind("--artifact-path=", 0) == 0) {
            append_value_arg(forwarded, "--output-dir", output_dir_from_output(after_equals("--artifact-path=")));
            continue;
        }
        if (!saw_separator && (arg == "--kernel-backend" || arg == "--native-cuda-kernel-backend")) {
            append_value_arg(forwarded, "--backend", require_value(argc, argv, &i, arg));
            continue;
        }
        if (!saw_separator && arg.rfind("--kernel-backend=", 0) == 0) {
            append_value_arg(forwarded, "--backend", after_equals("--kernel-backend="));
            continue;
        }
        if (!saw_separator && arg.rfind("--native-cuda-kernel-backend=", 0) == 0) {
            append_value_arg(forwarded, "--backend", after_equals("--native-cuda-kernel-backend="));
            continue;
        }
        if (!saw_separator && (arg == "--template" || arg == "--preset")) {
            append_value_arg(forwarded, "--template-name", require_value(argc, argv, &i, arg));
            continue;
        }
        if (!saw_separator && arg.rfind("--template=", 0) == 0) {
            append_value_arg(forwarded, "--template-name", after_equals("--template="));
            continue;
        }
        if (!saw_separator && arg.rfind("--preset=", 0) == 0) {
            append_value_arg(forwarded, "--template-name", after_equals("--preset="));
            continue;
        }
        if (!saw_separator && arg == "--graph") {
            append_value_arg(forwarded, "--graph-file", require_value(argc, argv, &i, arg));
            continue;
        }
        if (!saw_separator && arg.rfind("--graph=", 0) == 0) {
            append_value_arg(forwarded, "--graph-file", after_equals("--graph="));
            continue;
        }
        if (!saw_separator && (arg == "--native-cuda-executable" ||
                               arg == "--native-cuda-output-dir" ||
                               arg == "--native-cuda-tile-ops-lib" ||
                               arg == "--native-cuda-cuda-runtime-lib" ||
                               arg == "--native-cuda-lm-head-row-chunk-size")) {
            const std::string value = require_value(argc, argv, &i, arg);
            if (arg == "--native-cuda-executable") {
                append_value_arg(forwarded, "--target", value);
            } else if (arg == "--native-cuda-output-dir") {
                append_value_arg(forwarded, "--output-dir", value);
            } else if (arg == "--native-cuda-tile-ops-lib") {
                append_value_arg(forwarded, "--tile-ops-lib", value);
            } else if (arg == "--native-cuda-cuda-runtime-lib") {
                append_value_arg(forwarded, "--cuda-runtime-lib", value);
            } else {
                append_value_arg(forwarded, "--lm-head-row-chunk-size", value);
            }
            continue;
        }
        if (!saw_separator && arg.rfind("--native-cuda-output-dir=", 0) == 0) {
            append_value_arg(forwarded, "--output-dir", after_equals("--native-cuda-output-dir="));
            continue;
        }
        if (!saw_separator && arg.rfind("--native-cuda-executable=", 0) == 0) {
            append_value_arg(forwarded, "--target", after_equals("--native-cuda-executable="));
            continue;
        }
        if (!saw_separator && arg.rfind("--native-cuda-tile-ops-lib=", 0) == 0) {
            append_value_arg(forwarded, "--tile-ops-lib", after_equals("--native-cuda-tile-ops-lib="));
            continue;
        }
        if (!saw_separator && arg.rfind("--native-cuda-cuda-runtime-lib=", 0) == 0) {
            append_value_arg(forwarded, "--cuda-runtime-lib", after_equals("--native-cuda-cuda-runtime-lib="));
            continue;
        }
        if (!saw_separator && arg.rfind("--native-cuda-lm-head-row-chunk-size=", 0) == 0) {
            append_value_arg(forwarded, "--lm-head-row-chunk-size", after_equals("--native-cuda-lm-head-row-chunk-size="));
            continue;
        }
        if (!saw_separator && (arg == "--native-cuda-print-plan" ||
                               arg == "--native-cuda-list-templates" ||
                               arg == "--native-cuda-check-tile-ops" ||
                               arg == "--native-cuda-smoke-tile-ops" ||
                               arg == "--native-cuda-smoke-hnet-byte-patch-step" ||
                               arg == "--native-cuda-smoke-hnet-byte-patch-backward-step" ||
                               arg == "--native-cuda-smoke-hnet-byte-lm-loop-step" ||
                               arg == "--native-cuda-smoke-jamba-chunk-state-step" ||
                               arg == "--native-cuda-smoke-jamba-mamba-state-step" ||
                               arg == "--native-cuda-smoke-jamba-layer-schedule-step" ||
                               arg == "--native-cuda-smoke-family-layout-checkpoint-step" ||
                               arg == "--native-cuda-smoke-dense-jepa-train-step" ||
                               arg == "--native-cuda-smoke-llama-loop" ||
                               arg == "--native-cuda-smoke-llama-lm-head-step" ||
                               arg == "--native-cuda-smoke-llama-token-lm-train-step" ||
                               arg == "--native-cuda-smoke-llama-composed-train-step" ||
                               arg == "--native-cuda-smoke-llama-full-loop-step" ||
                               arg == "--native-cuda-smoke-llama-attention-block-step" ||
                               arg == "--native-cuda-smoke-llama-rope-attention-block-step" ||
                               arg == "--native-cuda-smoke-llama-rope-block-train-step" ||
                               arg == "--native-cuda-smoke-llama-packed-attention-step" ||
                               arg == "--native-cuda-smoke-llama-train-step" ||
                               arg == "--native-cuda-smoke-jepa-ar-loss-step" ||
                               arg == "--native-cuda-smoke-jepa-projector-step" ||
                               arg == "--native-cuda-smoke-jepa-target-encoder-step" ||
                               arg == "--native-cuda-smoke-diffusion-denoise-step" ||
                               arg == "--native-cuda-smoke-diffusion-objective-step" ||
                               arg == "--native-cuda-smoke-diffusion-full-loop-step" ||
                               arg == "--native-cuda-smoke-moe-route-expert-step" ||
                               arg == "--native-cuda-smoke-moe-transformer-block-step" ||
                               arg == "--native-cuda-smoke-moe-transformer-block-train-step" ||
                               arg == "--native-cuda-smoke-moe-transformer-lm-train-step" ||
                               arg == "--native-cuda-smoke-semantic-alignment-step" ||
                               arg == "--native-cuda-smoke-semantic-dense-jepa-train-step" ||
                               arg == "--native-cuda-smoke-semantic-router-moe-train-step" ||
                               arg == "--native-cuda-smoke-semantic-route-loss-step" ||
                               arg == "--native-cuda-smoke-seq2seq-cross-attention-step" ||
                               arg == "--native-cuda-smoke-seq2seq-full-encoder-decoder-loop-step" ||
                               arg == "--native-cuda-smoke-seq2seq-loss-composition-step" ||
                               arg == "--native-cuda-smoke-ttt-composite-inner-step" ||
                               arg == "--native-cuda-smoke-ttt-full-transformer-loop-step" ||
                               arg == "--native-cuda-smoke-ttt-linear-inner-step" ||
                               arg == "--native-cuda-smoke-universal-act-halt-step" ||
                               arg == "--native-cuda-smoke-universal-recurrent-step" ||
                               arg == "--native-cuda-smoke-universal-transformer-loop-step" ||
                               arg == "--native-cuda-smoke-optimizer-step" ||
                               arg == "--native-cuda-smoke-lm-step" ||
                               arg == "--native-cuda-smoke-attention-step" ||
                               arg == "--native-cuda-smoke-mlp-step" ||
                               arg == "--native-cuda-smoke-norm-residual-step" ||
                               arg == "--native-cuda-smoke-nvfp4-pack" ||
                               arg == "--native-cuda-smoke-transformer-block-step" ||
                               arg == "--native-cuda-smoke-transformer-lm-step" ||
                               arg == "--native-cuda-smoke-embedding-lm-step" ||
                               arg == "--native-cuda-allow-train-val-fallback" ||
                               arg == "--native-cuda-no-checkpoint" ||
                               arg == "--native-cuda-write-checkpoint" ||
                               arg == "--native-cuda-require-native-nvfp4-activation-packing" ||
                               arg == "--require-native-nvfp4-activation-packing" ||
                               arg == "--native-cuda-require-cooperative-lm-head-backward" ||
                               arg == "--require-cooperative-lm-head-backward" ||
                               arg == "--native-cuda-startup-only" ||
                               arg == "--native-cuda-fast-startup" ||
                               arg == "--fast-startup" ||
                               arg == "--native-cuda-dry-run")) {
            if (arg == "--native-cuda-print-plan") {
                forwarded.push_back("--print-plan");
            } else if (arg == "--native-cuda-dry-run") {
                forwarded.push_back("--dry-run");
            } else if (arg == "--native-cuda-check-tile-ops") {
                forwarded.push_back("--check-tile-ops");
            } else if (arg == "--native-cuda-smoke-tile-ops") {
                forwarded.push_back("--smoke-tile-ops");
            } else if (arg == "--native-cuda-smoke-hnet-byte-patch-step") {
                forwarded.push_back("--smoke-hnet-byte-patch-step");
            } else if (arg == "--native-cuda-smoke-hnet-byte-patch-backward-step") {
                forwarded.push_back("--smoke-hnet-byte-patch-backward-step");
            } else if (arg == "--native-cuda-smoke-hnet-byte-lm-loop-step") {
                forwarded.push_back("--smoke-hnet-byte-lm-loop-step");
            } else if (arg == "--native-cuda-smoke-jamba-chunk-state-step") {
                forwarded.push_back("--smoke-jamba-chunk-state-step");
            } else if (arg == "--native-cuda-smoke-jamba-mamba-state-step") {
                forwarded.push_back("--smoke-jamba-mamba-state-step");
            } else if (arg == "--native-cuda-smoke-jamba-layer-schedule-step") {
                forwarded.push_back("--smoke-jamba-layer-schedule-step");
            } else if (arg == "--native-cuda-smoke-family-layout-checkpoint-step") {
                forwarded.push_back("--smoke-family-layout-checkpoint-step");
            } else if (arg == "--native-cuda-smoke-dense-jepa-train-step") {
                forwarded.push_back("--smoke-dense-jepa-train-step");
            } else if (arg == "--native-cuda-smoke-llama-loop") {
                forwarded.push_back("--smoke-llama-loop");
            } else if (arg == "--native-cuda-smoke-llama-lm-head-step") {
                forwarded.push_back("--smoke-llama-lm-head-step");
            } else if (arg == "--native-cuda-smoke-llama-token-lm-train-step") {
                forwarded.push_back("--smoke-llama-token-lm-train-step");
            } else if (arg == "--native-cuda-smoke-llama-composed-train-step") {
                forwarded.push_back("--smoke-llama-composed-train-step");
            } else if (arg == "--native-cuda-smoke-llama-full-loop-step") {
                forwarded.push_back("--smoke-llama-full-loop-step");
            } else if (arg == "--native-cuda-smoke-llama-attention-block-step") {
                forwarded.push_back("--smoke-llama-attention-block-step");
            } else if (arg == "--native-cuda-smoke-llama-rope-attention-block-step") {
                forwarded.push_back("--smoke-llama-rope-attention-block-step");
            } else if (arg == "--native-cuda-smoke-llama-rope-block-train-step") {
                forwarded.push_back("--smoke-llama-rope-block-train-step");
            } else if (arg == "--native-cuda-smoke-llama-packed-attention-step") {
                forwarded.push_back("--smoke-llama-packed-attention-step");
            } else if (arg == "--native-cuda-smoke-llama-train-step") {
                forwarded.push_back("--smoke-llama-train-step");
            } else if (arg == "--native-cuda-smoke-jepa-ar-loss-step") {
                forwarded.push_back("--smoke-jepa-ar-loss-step");
            } else if (arg == "--native-cuda-smoke-jepa-projector-step") {
                forwarded.push_back("--smoke-jepa-projector-step");
            } else if (arg == "--native-cuda-smoke-jepa-target-encoder-step") {
                forwarded.push_back("--smoke-jepa-target-encoder-step");
            } else if (arg == "--native-cuda-smoke-diffusion-denoise-step") {
                forwarded.push_back("--smoke-diffusion-denoise-step");
            } else if (arg == "--native-cuda-smoke-diffusion-objective-step") {
                forwarded.push_back("--smoke-diffusion-objective-step");
            } else if (arg == "--native-cuda-smoke-diffusion-full-loop-step") {
                forwarded.push_back("--smoke-diffusion-full-loop-step");
            } else if (arg == "--native-cuda-smoke-moe-route-expert-step") {
                forwarded.push_back("--smoke-moe-route-expert-step");
            } else if (arg == "--native-cuda-smoke-moe-transformer-block-step") {
                forwarded.push_back("--smoke-moe-transformer-block-step");
            } else if (arg == "--native-cuda-smoke-moe-transformer-block-train-step") {
                forwarded.push_back("--smoke-moe-transformer-block-train-step");
            } else if (arg == "--native-cuda-smoke-moe-transformer-lm-train-step") {
                forwarded.push_back("--smoke-moe-transformer-lm-train-step");
            } else if (arg == "--native-cuda-smoke-semantic-alignment-step") {
                forwarded.push_back("--smoke-semantic-alignment-step");
            } else if (arg == "--native-cuda-smoke-semantic-dense-jepa-train-step") {
                forwarded.push_back("--smoke-semantic-dense-jepa-train-step");
            } else if (arg == "--native-cuda-smoke-semantic-router-moe-train-step") {
                forwarded.push_back("--smoke-semantic-router-moe-train-step");
            } else if (arg == "--native-cuda-smoke-semantic-route-loss-step") {
                forwarded.push_back("--smoke-semantic-route-loss-step");
            } else if (arg == "--native-cuda-smoke-seq2seq-cross-attention-step") {
                forwarded.push_back("--smoke-seq2seq-cross-attention-step");
            } else if (arg == "--native-cuda-smoke-seq2seq-full-encoder-decoder-loop-step") {
                forwarded.push_back("--smoke-seq2seq-full-encoder-decoder-loop-step");
            } else if (arg == "--native-cuda-smoke-seq2seq-loss-composition-step") {
                forwarded.push_back("--smoke-seq2seq-loss-composition-step");
            } else if (arg == "--native-cuda-smoke-ttt-composite-inner-step") {
                forwarded.push_back("--smoke-ttt-composite-inner-step");
            } else if (arg == "--native-cuda-smoke-ttt-full-transformer-loop-step") {
                forwarded.push_back("--smoke-ttt-full-transformer-loop-step");
            } else if (arg == "--native-cuda-smoke-ttt-linear-inner-step") {
                forwarded.push_back("--smoke-ttt-linear-inner-step");
            } else if (arg == "--native-cuda-smoke-universal-act-halt-step") {
                forwarded.push_back("--smoke-universal-act-halt-step");
            } else if (arg == "--native-cuda-smoke-universal-recurrent-step") {
                forwarded.push_back("--smoke-universal-recurrent-step");
            } else if (arg == "--native-cuda-smoke-universal-transformer-loop-step") {
                forwarded.push_back("--smoke-universal-transformer-loop-step");
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
            } else if (arg == "--native-cuda-smoke-nvfp4-pack") {
                forwarded.push_back("--smoke-nvfp4-pack");
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
            } else if (arg == "--native-cuda-require-native-nvfp4-activation-packing" ||
                       arg == "--require-native-nvfp4-activation-packing") {
                forwarded.push_back("--require-native-nvfp4-activation-packing");
            } else if (arg == "--native-cuda-require-cooperative-lm-head-backward" ||
                       arg == "--require-cooperative-lm-head-backward") {
                forwarded.push_back("--require-cooperative-lm-head-backward");
            } else if (arg == "--native-cuda-startup-only") {
                forwarded.push_back("--startup-only");
            } else if (arg == "--native-cuda-fast-startup" || arg == "--fast-startup") {
                forwarded.push_back("--fast-startup");
            } else {
                forwarded.push_back("--list-templates");
            }
            continue;
        }
        if (!saw_separator && (arg == "--print-command" || arg == "--native-cuda-print-command")) {
            result.print_command_requested = true;
            continue;
        }
        forwarded.push_back(std::move(arg));
    }

    if (list_models || !is_dense_gpt_model(model)) {
        return result;
    }
    if (model == "nanogpt" && has_forwarded_flag(forwarded, "--train-token-lm")) {
        return result;
    }

    std::vector<std::string> command;
    command.push_back(sibling_gpt_cli(argv[0]));
    if (model == "nanogpt") {
        append_value_arg(command, "--model-family", "gpt");
        if (!has_template_or_graph_selector(forwarded)) {
            append_value_arg(command, "--template-name", "nanogpt");
        }
    } else {
        append_value_arg(command, "--model-family", model);
    }
    command.insert(command.end(), forwarded.begin(), forwarded.end());
    if (!has_native_train_action(command) && !has_template_catalog_action(command)) {
        command.push_back("--train-transformer-lm");
    }
    if (!has_any_forwarded_value_flag(command, {"--backend", "--runtime", "--device"})) {
        append_value_arg(command, "--backend", "tile-cuda");
    }
    if (!has_template_catalog_action(command) &&
        !has_dataset_free_preflight_action(command) &&
        !has_any_forwarded_value_flag(command, {"--dataset-alias", "--dataset-path", "--train-bin", "--val-bin"}) &&
        !has_forwarded_flag(command, "--tinystories")) {
        append_value_arg(command, "--dataset-alias", env_or_empty("DATASET_ALIAS").empty()
                                                  ? std::string(kDefaultTinyStoriesAlias)
                                                  : env_or_empty("DATASET_ALIAS"));
    }
    if (model == "gpt3" &&
        !has_template_or_graph_selector(command) &&
        !has_any_forwarded_value_flag(command, {"--train-seq-len", "--seq-len"})) {
        append_value_arg(command, "--train-seq-len", "2048");
    }
    result.can_bypass = true;
    result.argv = std::move(command);
    return result;
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
        << "  train    Exec dense GPT directly when possible, otherwise nfn_native_train.\n"
        << "  infer    Exec nfn_gpt_native_train --sample-checkpoint/--native-info.\n\n"
        << "Native infer options:\n"
        << "  --checkpoint, --native-checkpoint, --weights PATH\n"
        << "                                        Native model_*.bin file or directory.\n"
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
        if (env_or_empty("NFN_NATIVE_TRAIN_CLI").empty()) {
            DenseTrainCommand dense_command = build_dense_gpt_train_command(argc, argv);
            if (dense_command.can_bypass) {
                setenv_default_if_empty("CUDA_VISIBLE_DEVICES", "0");
                setenv_default_if_empty("CUDA_DEVICE_MAX_CONNECTIONS", "1");
                setenv_default_if_empty("CUDA_MODULE_LOADING", "LAZY");
                if (dense_command.print_command_requested) {
                    print_command(dense_command.argv);
                    return 0;
                }
                return exec_command(dense_command.argv);
            }
        }
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
        if (arg == "--checkpoint" || arg == "--native-checkpoint" || arg == "--weights") {
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
        if (arg.rfind("--weights=", 0) == 0) {
            checkpoint = after_equals("--weights=");
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
        std::cerr << "nfn-native infer requires --checkpoint, --native-checkpoint, or --weights\n";
        return 2;
    }

    const fs::path resolved_checkpoint = resolve_latest_checkpoint(fs::path(checkpoint));
    if (!fs::exists(resolved_checkpoint)) {
        std::cerr << "native checkpoint does not exist: " << resolved_checkpoint.string() << '\n';
        return 2;
    }

    std::vector<std::string> command;
    command.push_back(sibling_gpt_cli(argv[0]));
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
