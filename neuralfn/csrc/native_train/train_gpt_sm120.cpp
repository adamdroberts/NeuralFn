#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string env_or(const char* name, std::string fallback = {}) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' ? std::string(value) : std::move(fallback);
}

void setenv_default_if_empty(const char* name, const std::string& value) {
    const char* current = std::getenv(name);
    if (value.empty() || (current != nullptr && current[0] != '\0')) {
        return;
    }
    setenv(name, value.c_str(), 1);
}

std::string env_first(std::initializer_list<const char*> names, std::string fallback = {}) {
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr && value[0] != '\0') {
            return std::string(value);
        }
    }
    return fallback;
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool executable(const fs::path& path) {
    std::error_code ec;
    if (!fs::exists(path, ec) || fs::is_directory(path, ec)) {
        return false;
    }
    return access(path.c_str(), X_OK) == 0;
}

bool is_dense_gpt_model_family(const std::string& name) {
    return name == "gpt" || name == "gpt2" || name == "gpt3" || name == "nanogpt";
}

bool is_dense_gpt_template_selector(const std::string& name) {
    return is_dense_gpt_model_family(name) ||
        name == "gpt2_modern" ||
        name == "gpt2_megakernel" ||
        name == "gpt2_moa" ||
        name == "nanogpt_modern" ||
        name == "nanogpt_megakernel";
}

std::string trim(std::string value) {
    const auto begin = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    const auto end = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    if (begin >= end) {
        return {};
    }
    return std::string(begin, end);
}

std::string select_display_disabled_cuda_device() {
    FILE* pipe = popen(
        "nvidia-smi --query-gpu=index,display_active,utilization.gpu --format=csv,noheader,nounits 2>/dev/null",
        "r");
    if (pipe == nullptr) {
        return "0";
    }
    char buffer[256];
    std::string first_index;
    std::string best_index;
    int best_util = 0;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::string line(buffer);
        const std::size_t first_comma = line.find(',');
        const std::size_t second_comma = first_comma == std::string::npos ? std::string::npos : line.find(',', first_comma + 1);
        if (first_comma == std::string::npos || second_comma == std::string::npos) {
            continue;
        }
        std::string index = trim(line.substr(0, first_comma));
        std::string display = trim(line.substr(first_comma + 1, second_comma - first_comma - 1));
        std::string util_text = trim(line.substr(second_comma + 1));
        if (index.empty()) {
            continue;
        }
        if (first_index.empty()) {
            first_index = index;
        }
        int util = 0;
        try {
            util = std::stoi(util_text);
        } catch (...) {
            util = 0;
        }
        if (display == "Disabled" && (best_index.empty() || util < best_util)) {
            best_index = index;
            best_util = util;
        }
    }
    const int status = pclose(pipe);
    if (status != 0 && first_index.empty() && best_index.empty()) {
        return "0";
    }
    if (!best_index.empty()) {
        return best_index;
    }
    return first_index.empty() ? "0" : first_index;
}

std::string resolve_cuda_visible_devices_default() {
    std::string requested = env_first(
        {"NFN_NATIVE_GPT_CUDA_VISIBLE_DEVICES", "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES", "NFN_SM120_CUDA_VISIBLE_DEVICES"},
        "0");
    std::string normalized = lower(trim(requested));
    if (normalized.empty() || normalized == "none" || normalized == "off") {
        return {};
    }
    if (normalized == "auto" || normalized == "dedicated" || normalized == "dedicated-auto") {
        return select_display_disabled_cuda_device();
    }
    return requested;
}

fs::path repo_root_from_argv0(const char* argv0) {
    std::error_code ec;
    fs::path self = fs::absolute(argv0, ec);
    if (ec) {
        self = fs::path(argv0);
    }
    fs::path dir = self.parent_path();
    if (dir.filename() == "build") {
        return dir.parent_path();
    }
    if (fs::exists(dir / "build" / "nfn_gpt_native_train", ec) ||
        fs::exists(dir / "tools" / "train_gpt_sm120.sh", ec)) {
        return dir;
    }
    return fs::current_path(ec);
}

void append(std::vector<std::string>& args, std::string value) {
    args.emplace_back(std::move(value));
}

void append_pair(std::vector<std::string>& args, std::string flag, std::string value) {
    args.emplace_back(std::move(flag));
    args.emplace_back(std::move(value));
}

std::string value_after_equals(const std::string& arg) {
    const std::size_t pos = arg.find('=');
    return pos == std::string::npos ? std::string() : arg.substr(pos + 1);
}

long long parse_nonnegative_i64_or(const std::string& value, long long fallback) {
    try {
        std::size_t consumed = 0;
        const long long parsed = std::stoll(value, &consumed);
        if (consumed == value.size() && parsed >= 0) {
            return parsed;
        }
    } catch (...) {
    }
    return fallback;
}

[[noreturn]] void usage() {
    std::cout
        << "Usage: nfn_train_gpt [options] [-- extra native args]\n\n"
        << "Compiled dense GPT training helper. It calls nfn_gpt_native_train\n"
        << "directly with CUDA Tile defaults matching tools/train_gpt_sm120.sh.\n\n"
        << "Default cadence, shape, and optimizer values can be overridden with\n"
        << "NFN_NATIVE_GPT_*, NFN_SM120_NATIVE_*, or NFN_SM120_* environment variables before launch.\n\n"
        << "Options:\n"
        << "  --activation NAME\n"
        << "  --moa-interval N\n"
        << "  --base-model NAME | --model-family NAME\n"
        << "  --template-name NAME | --template NAME | --preset NAME\n"
        << "  --graph-file PATH | --graph PATH\n"
        << "  --dataset-alias PATH | --dataset-path PATH\n"
        << "  --output-dir PATH\n"
        << "  --native-cuda-fast-startup | --fast-startup\n"
        << "  -h, --help\n";
    std::exit(0);
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path root = repo_root_from_argv0(argv[0]);
    const fs::path default_llmk_tinystories =
        "/mnt/disk2/dev/open-source/llm.kittens/dev/data/tinystories";
    const fs::path llmk_tinystories =
        env_or("NFN_LLM_KITTENS_TINYSTORIES_DIR", default_llmk_tinystories.string());

    std::string native_bin = env_or("NFN_NATIVE_GPT_TRAIN_BIN");
    if (native_bin.empty()) {
        const fs::path linked = root / "build" / "nfn_gpt_native_train_linked";
        native_bin = executable(linked) ? linked.string() : (root / "build" / "nfn_gpt_native_train").string();
    }

    setenv_default_if_empty("CUDA_VISIBLE_DEVICES", resolve_cuda_visible_devices_default());
    setenv_default_if_empty("CUDA_DEVICE_MAX_CONNECTIONS", "1");
    setenv_default_if_empty("CUDA_MODULE_LOADING", "LAZY");

    std::string activation = env_first({"NFN_NATIVE_GPT_ACTIVATION", "NFN_SM120_ACTIVATION"}, "gelu");
    std::string moa_interval = env_first({"NFN_NATIVE_GPT_MOA_INTERVAL", "NFN_SM120_MOA_INTERVAL"}, "50");
    std::string output_dir =
        env_first({"NFN_NATIVE_GPT_OUTPUT_DIR", "NFN_SM120_OUTPUT_DIR"}, (root / "artifacts" / "gpt_sm120").string());
    std::string dataset_alias = env_first({"NFN_NATIVE_GPT_DATASET_ALIAS", "NFN_SM120_DATASET_ALIAS"});
    std::string model_family = env_first({"NFN_NATIVE_GPT_MODEL_FAMILY", "NFN_SM120_MODEL_FAMILY"}, "gpt");
    std::string template_name = env_first({"NFN_NATIVE_GPT_TEMPLATE_NAME", "NFN_SM120_TEMPLATE_NAME"}, "gpt");
    std::string graph_file = env_first({"NFN_NATIVE_GPT_GRAPH_FILE", "NFN_SM120_GRAPH_FILE"});
    std::string train_seq_len =
        env_first({"NFN_NATIVE_GPT_TRAIN_SEQ_LEN", "NFN_SM120_NATIVE_TRAIN_SEQ_LEN", "NFN_SM120_TRAIN_SEQ_LEN"}, "1024");
    std::string batch_size =
        env_first({"NFN_NATIVE_GPT_BATCH_SIZE", "NFN_SM120_NATIVE_BATCH_SIZE", "NFN_SM120_BATCH_SIZE"}, "64");
    std::string eval_every_steps =
        env_first({"NFN_NATIVE_GPT_EVAL_EVERY_STEPS", "NFN_SM120_NATIVE_EVAL_EVERY_STEPS", "NFN_SM120_EVAL_EVERY_STEPS"}, "1000");
    std::string eval_batches =
        env_first({"NFN_NATIVE_GPT_EVAL_BATCHES", "NFN_SM120_NATIVE_EVAL_BATCHES", "NFN_SM120_EVAL_BATCHES"}, "20");
    std::string sample_every =
        env_first({"NFN_NATIVE_GPT_SAMPLE_EVERY", "NFN_SM120_NATIVE_SAMPLE_EVERY", "NFN_SM120_SAMPLE_EVERY"}, "20000");
    std::string generate_tokens =
        env_first({"NFN_NATIVE_GPT_GENERATE_TOKENS", "NFN_SM120_NATIVE_GENERATE_TOKENS", "NFN_SM120_GENERATE_TOKENS"}, "144");
    std::string checkpoint_every =
        env_first({"NFN_NATIVE_GPT_CHECKPOINT_EVERY", "NFN_SM120_NATIVE_CHECKPOINT_EVERY", "NFN_SM120_CHECKPOINT_EVERY"}, "200");
    std::string train_batch_tokens =
        env_first({"NFN_NATIVE_GPT_TRAIN_BATCH_TOKENS", "NFN_SM120_NATIVE_TRAIN_BATCH_TOKENS", "NFN_SM120_TRAIN_BATCH_TOKENS"}, "524288");
    std::string learning_rate =
        env_first({"NFN_NATIVE_GPT_LEARNING_RATE", "NFN_SM120_NATIVE_LEARNING_RATE", "NFN_SM120_LEARNING_RATE"}, "0.0006");
    std::string final_lr_fraction =
        env_first({"NFN_NATIVE_GPT_FINAL_LR_FRACTION", "NFN_SM120_NATIVE_FINAL_LR_FRACTION", "NFN_SM120_FINAL_LR_FRACTION"}, "0.0");
    std::string weight_decay =
        env_first({"NFN_NATIVE_GPT_WEIGHT_DECAY", "NFN_SM120_NATIVE_WEIGHT_DECAY", "NFN_SM120_WEIGHT_DECAY"}, "0.1");
    std::string warmup_steps =
        env_first({"NFN_NATIVE_GPT_WARMUP_STEPS", "NFN_SM120_NATIVE_WARMUP_STEPS", "NFN_SM120_WARMUP_STEPS"}, "600");
    std::string max_steps =
        env_first({"NFN_NATIVE_GPT_MAX_STEPS", "NFN_SM120_NATIVE_MAX_STEPS", "NFN_SM120_MAX_STEPS"}, "20000");
    std::string train_loss_every_steps =
        env_first({"NFN_NATIVE_GPT_TRAIN_LOSS_EVERY_STEPS", "NFN_SM120_NATIVE_TRAIN_LOSS_EVERY_STEPS", "NFN_SM120_TRAIN_LOSS_EVERY_STEPS"});
    bool seq_len_explicit = false;
    bool batch_size_explicit = false;
    bool template_explicit = false;
    bool fast_startup_explicit = false;
    bool activation_explicit =
        std::getenv("NFN_NATIVE_GPT_ACTIVATION") != nullptr || std::getenv("NFN_SM120_ACTIVATION") != nullptr;
    std::vector<std::string> extra_args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << flag << " requires a value\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            usage();
        } else if (arg == "--activation") {
            activation = require_value("--activation");
            activation_explicit = true;
        } else if (arg.rfind("--activation=", 0) == 0) {
            activation = value_after_equals(arg);
            activation_explicit = true;
        } else if (arg == "--moa-interval") {
            moa_interval = require_value("--moa-interval");
        } else if (arg.rfind("--moa-interval=", 0) == 0) {
            moa_interval = value_after_equals(arg);
        } else if (arg == "--base-model" || arg == "--model-family") {
            model_family = require_value(arg.c_str());
            if (!template_explicit) {
                template_name = model_family;
            }
        } else if (arg.rfind("--base-model=", 0) == 0 || arg.rfind("--model-family=", 0) == 0) {
            model_family = value_after_equals(arg);
            if (!template_explicit) {
                template_name = model_family;
            }
        } else if (arg == "--template-name" || arg == "--template" || arg == "--preset") {
            template_name = require_value(arg.c_str());
            template_explicit = true;
        } else if (arg.rfind("--template-name=", 0) == 0 || arg.rfind("--template=", 0) == 0 ||
                   arg.rfind("--preset=", 0) == 0) {
            template_name = value_after_equals(arg);
            template_explicit = true;
        } else if (arg == "--graph-file" || arg == "--graph") {
            graph_file = require_value(arg.c_str());
        } else if (arg.rfind("--graph-file=", 0) == 0 || arg.rfind("--graph=", 0) == 0) {
            graph_file = value_after_equals(arg);
        } else if (arg == "--train-seq-len" || arg == "--seq-len") {
            train_seq_len = require_value(arg.c_str());
            seq_len_explicit = true;
            append_pair(extra_args, arg, train_seq_len);
        } else if (arg.rfind("--train-seq-len=", 0) == 0 || arg.rfind("--seq-len=", 0) == 0) {
            train_seq_len = value_after_equals(arg);
            seq_len_explicit = true;
            append(extra_args, arg);
        } else if (arg == "--batch-size") {
            batch_size = require_value("--batch-size");
            batch_size_explicit = true;
            append_pair(extra_args, arg, batch_size);
        } else if (arg.rfind("--batch-size=", 0) == 0) {
            batch_size = value_after_equals(arg);
            batch_size_explicit = true;
            append(extra_args, arg);
        } else if (arg == "--max-steps") {
            max_steps = require_value("--max-steps");
        } else if (arg.rfind("--max-steps=", 0) == 0) {
            max_steps = value_after_equals(arg);
        } else if (arg == "--dataset-alias" || arg == "--dataset-path") {
            dataset_alias = require_value(arg.c_str());
        } else if (arg.rfind("--dataset-alias=", 0) == 0 || arg.rfind("--dataset-path=", 0) == 0) {
            dataset_alias = value_after_equals(arg);
        } else if (arg == "--output-dir") {
            output_dir = require_value("--output-dir");
        } else if (arg.rfind("--output-dir=", 0) == 0) {
            output_dir = value_after_equals(arg);
        } else if (arg == "--native-cuda-fast-startup" || arg == "--fast-startup") {
            fast_startup_explicit = true;
            append(extra_args, "--fast-startup");
        } else if (arg == "--") {
            while (++i < argc) {
                std::string passthrough = argv[i];
                if (passthrough == "--native-cuda-fast-startup" || passthrough == "--fast-startup") {
                    fast_startup_explicit = true;
                }
                append(extra_args, passthrough == "--native-cuda-fast-startup" ? "--fast-startup" : passthrough);
            }
            break;
        } else {
            append(extra_args, arg);
        }
    }

    model_family = lower(model_family);
    template_name = lower(template_name);
    activation = lower(activation);
    if (!(activation == "gelu" || activation == "relu" || activation == "silu" ||
          activation == "relu2" || activation == "prelu" || activation == "sd-prelu" ||
          activation == "swiglu" || activation == "geglu" || activation == "ensemble" ||
          activation == "moa")) {
        std::cerr << "Invalid --activation '" << activation << "'\n";
        return 2;
    }
    if (!is_dense_gpt_model_family(model_family)) {
        if (is_dense_gpt_template_selector(model_family)) {
            if (!template_explicit) {
                template_name = model_family;
            }
            model_family = "gpt";
        } else {
            std::cerr << "Invalid --base-model/--model-family '" << model_family << "'\n";
            return 2;
        }
    }
    if (!activation_explicit && template_name.find("moa") != std::string::npos) {
        activation = "moa";
    }
    if (template_name == "gpt3") {
        if (!seq_len_explicit) {
            train_seq_len = "2048";
        }
        if (!batch_size_explicit) {
            batch_size = "32";
        }
    }
    if (!executable(native_bin)) {
        std::cerr << "Native GPT trainer is not executable: " << native_bin << "\n"
                  << "Build it with: bash tools/build_native_gpt_cli.sh\n";
        return 127;
    }

    const bool fast_startup_env_explicit =
        std::getenv("NFN_NATIVE_GPT_FAST_STARTUP") != nullptr ||
        std::getenv("NFN_NATIVE_GPT2_FAST_STARTUP") != nullptr ||
        std::getenv("NFN_TILE_CUDA_FAST_STARTUP") != nullptr;
    const long long defer_prewarm_after_steps = parse_nonnegative_i64_or(
        env_first({"NFN_NATIVE_GPT_DEFER_PREWARM_AFTER_STEPS",
                   "NFN_NATIVE_GPT2_DEFER_PREWARM_AFTER_STEPS",
                   "NFN_TILE_CUDA_DEFER_PREWARM_AFTER_STEPS"},
                  "1024"),
        1024);
    const long long resolved_max_steps = parse_nonnegative_i64_or(max_steps, 0);
    const bool auto_fast_startup_short_run =
        !fast_startup_explicit &&
        !fast_startup_env_explicit &&
        defer_prewarm_after_steps > 0 &&
        resolved_max_steps > 0 &&
        resolved_max_steps <= defer_prewarm_after_steps;

    std::vector<std::string> out;
    append(out, native_bin);
    append_pair(out, "--model-family", model_family);
    append_pair(out, "--template-name", template_name);
    if (!graph_file.empty()) {
        append_pair(out, "--graph-file", graph_file);
    }
    if (!dataset_alias.empty()) {
        append_pair(out, "--dataset-alias", dataset_alias);
    } else if (fs::exists(llmk_tinystories / "TinyStories_train.bin") &&
               fs::exists(llmk_tinystories / "TinyStories_val.bin")) {
        append_pair(out, "--dataset-alias", llmk_tinystories.string());
    } else {
        append(out, "--tinystories");
    }
    append_pair(out, "--backend", "tile-cuda");
    if (fs::path(native_bin).filename() == "nfn_gpt_native_train_linked") {
        append_pair(out, "--tile-ops-lib", "linked");
    }
    append_pair(out, "--output-dir", output_dir);
    append_pair(out, "--eval-every-steps", eval_every_steps);
    append_pair(out, "--eval-batches", eval_batches);
    if (!train_loss_every_steps.empty()) {
        append_pair(out, "--train-loss-every-steps", train_loss_every_steps);
    }
    append_pair(out, "--native-cuda-sample-every", sample_every);
    append_pair(out, "--native-cuda-generate-tokens", generate_tokens);
    append_pair(out, "--batch-size", batch_size);
    append_pair(out, "--train-seq-len", train_seq_len);
    append_pair(out, "--train-batch-tokens", train_batch_tokens);
    append_pair(out, "--learning-rate", learning_rate);
    append_pair(out, "--final-lr-fraction", final_lr_fraction);
    append_pair(out, "--weight-decay", weight_decay);
    append_pair(out, "--warmup-steps", warmup_steps);
    append_pair(out, "--native-cuda-checkpoint-every", checkpoint_every);
    append_pair(out, "--max-steps", max_steps);
    append_pair(out, "--native-cuda-activation", activation);
    if (auto_fast_startup_short_run) {
        append(out, "--fast-startup");
    }
    if (activation == "moa") {
        append_pair(out, "--native-cuda-moa-interval", moa_interval);
    }
    append(out, "--train-transformer-lm");
    out.insert(out.end(), extra_args.begin(), extra_args.end());

    std::vector<char*> exec_args;
    exec_args.reserve(out.size() + 1);
    for (std::string& item : out) {
        exec_args.push_back(item.data());
    }
    exec_args.push_back(nullptr);
    execvp(exec_args[0], exec_args.data());
    std::cerr << "failed to exec " << exec_args[0] << ": " << std::strerror(errno) << "\n";
    return 127;
}
