#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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

[[noreturn]] void usage() {
    std::cout
        << "Usage: nfn_train_gpt_sm120 [options] [-- extra native args]\n\n"
        << "Compiled SM120 dense GPT training helper. It calls nfn_gpt_native_train\n"
        << "directly with the same core defaults as tools/train_gpt_sm120.sh.\n\n"
        << "Options:\n"
        << "  --activation NAME\n"
        << "  --moa-interval N\n"
        << "  --base-model NAME | --model-family NAME\n"
        << "  --template-name NAME | --template NAME | --preset NAME\n"
        << "  --graph-file PATH | --graph PATH\n"
        << "  --dataset-alias PATH | --dataset-path PATH\n"
        << "  --output-dir PATH\n"
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

    setenv("CUDA_DEVICE_MAX_CONNECTIONS", env_or("CUDA_DEVICE_MAX_CONNECTIONS", "1").c_str(), 0);
    setenv("CUDA_MODULE_LOADING", env_or("CUDA_MODULE_LOADING", "LAZY").c_str(), 0);

    std::string activation = env_or("NFN_SM120_ACTIVATION", "gelu");
    std::string moa_interval = env_or("NFN_SM120_MOA_INTERVAL", "50");
    std::string output_dir = env_or("NFN_SM120_OUTPUT_DIR", (root / "artifacts" / "gpt_sm120").string());
    std::string dataset_alias = env_or("NFN_SM120_DATASET_ALIAS");
    std::string model_family = env_or("NFN_SM120_MODEL_FAMILY", "gpt");
    std::string template_name = env_or("NFN_SM120_TEMPLATE_NAME", "gpt");
    std::string graph_file = env_or("NFN_SM120_GRAPH_FILE");
    std::string train_seq_len = env_or("NFN_SM120_TRAIN_SEQ_LEN", "1024");
    std::string batch_size = env_or("NFN_SM120_BATCH_SIZE", "64");
    bool seq_len_explicit = false;
    bool batch_size_explicit = false;
    bool template_explicit = false;
    bool activation_explicit = std::getenv("NFN_SM120_ACTIVATION") != nullptr;
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
        } else if (arg == "--dataset-alias" || arg == "--dataset-path") {
            dataset_alias = require_value(arg.c_str());
        } else if (arg.rfind("--dataset-alias=", 0) == 0 || arg.rfind("--dataset-path=", 0) == 0) {
            dataset_alias = value_after_equals(arg);
        } else if (arg == "--output-dir") {
            output_dir = require_value("--output-dir");
        } else if (arg.rfind("--output-dir=", 0) == 0) {
            output_dir = value_after_equals(arg);
        } else if (arg == "--") {
            while (++i < argc) {
                append(extra_args, argv[i]);
            }
            break;
        } else {
            append(extra_args, arg);
        }
    }

    model_family = lower(model_family);
    template_name = lower(template_name);
    activation = lower(activation);
    if (!activation_explicit && template_name.find("moa") != std::string::npos) {
        activation = "moa";
    }
    if (!(activation == "gelu" || activation == "relu" || activation == "silu" ||
          activation == "relu2" || activation == "prelu" || activation == "sd-prelu" ||
          activation == "swiglu" || activation == "geglu" || activation == "ensemble" ||
          activation == "moa")) {
        std::cerr << "Invalid --activation '" << activation << "'\n";
        return 2;
    }
    if (!(model_family == "gpt" || model_family == "gpt2" || model_family == "gpt3" ||
          model_family == "nanogpt")) {
        std::cerr << "Invalid --base-model/--model-family '" << model_family << "'\n";
        return 2;
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
    append_pair(out, "--eval-every-steps", "250");
    append_pair(out, "--eval-batches", "20");
    append_pair(out, "--native-cuda-sample-every", "20000");
    append_pair(out, "--native-cuda-generate-tokens", "144");
    append_pair(out, "--batch-size", batch_size);
    append_pair(out, "--train-seq-len", train_seq_len);
    append_pair(out, "--train-batch-tokens", "524288");
    append_pair(out, "--learning-rate", "0.0006");
    append_pair(out, "--final-lr-fraction", "0.0");
    append_pair(out, "--weight-decay", "0.1");
    append_pair(out, "--warmup-steps", "60");
    append_pair(out, "--native-cuda-checkpoint-every", "200");
    append_pair(out, "--max-steps", "20000");
    append_pair(out, "--native-cuda-activation", activation);
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
